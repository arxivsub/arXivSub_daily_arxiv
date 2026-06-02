# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-02 | 今日论文总数: 815

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. MENTIS: What Belief Changes Under Alignment? Measuring Multi-Scale Latent Torsion in Language Models

**arXiv ID:** 2606.01060 | [PDF](https://arxiv.org/pdf/2606.01060v1)

**作者:** Partha Pratim Saha `[一作]` (BITS Pilani), Amitava Das `[通讯]` (BITS Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比指令调优（IT）与偏好对齐（PA）模型，开发并使用MENTIS框架评估内部几何结构的重新组织。

**💡 创新点**

提出基于层级torsion（T1/T2）和能量-辐射-激活（ERA）的几何诊断方法，揭示对齐是选择性且在特定深度层显著变化的。

**🔧 技术方法**

计算层级梯度方向、角度torsion、谱torsion、ERA深度定位；使用PCA降维与裁剪余弦相似度进行数值稳定化。

**📊 数据集**

采用LITMUS benchmark（20,439个提示、18个概念、7个价值axioms），对四组7–8B模型进行评估。

**📈 对比分析**

与CKA、余弦距离、RepE基线比较，MENTIS在概念辨别（CV=0.64）和安全/不安全提示区分（AUC=0.89）上明显优于基线，并能定位对齐产生最大几何变化的层。

**⚠️ 局限性**

结果为相关性非因果，效应幅度较小；受限于读出方式、目标标记、LITMUS构造，且仅测试7–8B decoder-only模型和RLHF/DPO对齐，未验证更大规模或不同对齐策略。

---

## 2. Physics-Informed Deep Learning for Entropy Prediction in Heterogeneous Systems: Thermodynamic and Information-Theoretic Case Studies

**arXiv ID:** 2606.01179 | [PDF](https://arxiv.org/pdf/2606.01179v1)

**作者:** Biswajeet Sahoo `[一作]` (Durham University), Debadutta Patra `[通讯]` (Veer Surendra Sai University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并验证了一种统一的物理信息深度学习框架（PIDL），在化学反应器和金融回报的两类异构系统中预测熵值，并通过共享编码器探究熵的域不变表示。

**💡 创新点**

首次将Softplus硬约束与多目标物理约束结合，构建单网络同时满足ODE/PDE残差、第二定律和概率归一化，并系统评估共享编码器在不同物理方程下的可迁移性。

**🔧 技术方法**

基于Physics‑Informed Neural Networks（PINN）的深度学习架构，自动微分求解残差，Softplus硬约束实现熵正性，Ruppeiner Riemann几何用于后验诊断。

**📊 数据集**

对CSTR的离散时间ODE数值解（5001点，加入噪声）以及标普500日度对数收益的核密度估计（5786天）用于逆Fokker‑Planck问题。

**📈 对比分析**

与独立PINN和无物理约束网络对比，使用MAPE、MSE、Second‑Law违规率和归一化误差等指标，显示共享编码器参数减少19%但性能不低于单域模型；物理约束模型在仅30%数据下仍保持>90%准确度。

**⚠️ 局限性**

仅适用于时域一维ODE和x,t二维FP，未涵盖空间分布、跳跃、长期记忆等复杂动力学；Ruppeiner曲率仅为后验诊断，训练时间与实现复杂度仍较高。

---

## 3. The World's Fastest Matching Engine Algorithm

**arXiv ID:** 2606.01183 | [PDF](https://arxiv.org/pdf/2606.01183v1)

**作者:** Jake Yoon `[一作]` `[通讯]` (Flash One Technologies LLC), Jake Yoon (Flash One Technologies LLC)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对电子交易订单簿匹配核心的全新数据结构与算法设计，构建了基于固定容量优先指示节点（PIN）和邻居感知平衡树的订单簿，实现了无指针追踪、O(1)插入/删除、常数时间的树操作，并在单核心上实现了每秒3.2亿条订单的处理速率，尾部延迟低于1微秒。

**💡 创新点**

核心创新包括：①PIN 设计，利用连续槽位与优先指示器实现 O(1)优先级判定和局部搬迁；②邻居感知平衡树，消除根到叶的搜索，仅通过已知邻居进行常数时间插拔并沿单一路径重平衡；③深度感知节点容量模型，动态调整节点大小以最大化 L1 缓存利用；④与硬件加速兼容的实现细节。

**🔧 技术方法**

使用的技术主要是 C++实现的固定容量数组、位掩码优先指示器、邻居链表的平衡树（AVL/红黑/ B/B+树）以及多核共享无锁队列的分片架构；还利用了ARM Neoverse‑V2 CPU的缓存特性和 FPGA 直连 BRAM 的映射。

**📊 数据集**

评估数据集基于监管公开的 NVIDIA 等热门证券的交易记录，通过仿真生成功率律分布的价格层、指数波动模型（GBM）以及 95% 取消率、15% IOC 的随机订单流，覆盖静态、正常波动、极端闪崩等多种波动情境。

**📈 对比分析**

与三种主流基准（Liquibook、Exchange‑core、QuantCup）在相同硬件与工作负载下对比；单核吞吐量提升 5–11×；在 96 核实例上服务 10,000 只证券可达 6.4 亿条/秒；尾部延迟保持在 128 ns (P99) 以下。

**⚠️ 局限性**

主要局限在多符号多核情况下共享缓存导致的局部性下降，导致单实例吞吐随符号数增大而下降；对极端波动下的最小订单层深度模型仍需进一步自适应；实现中未包含完整的业务规则与风险校验，需与交易所现有系统进一步集成。

---

## 4. DiscourseFlip: An Oblique Discourse-Level Opinion Manipulation Attack against Black-box Retrieval-Augmented Generation

**arXiv ID:** 2606.01212 | [PDF](https://arxiv.org/pdf/2606.01212v1)

**作者:** Yuyang Gong `[一作]` (Wuhan University), Xiaozhong Liu `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3908 | [OpenAlex ID](https://openalex.org/A5101985030)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对黑盒检索增强生成（RAG）系统，提出了 DiscourseFlip 通过在语义查询网络上投毒文档实现整体舆论操控的攻击方法。

**💡 创新点**

创新点在于将攻击视为图引导的分布式最大覆盖优化，利用语义图与知识图融合的分层结构，使有限投毒预算在多主题查询空间中产生高覆盖率与隐蔽性。

**🔧 技术方法**

核心技术包括：图结构化查询网络构建（检索统计层 + 逻辑推理层），自适应代理（生成、评估、迭代优化），以及基于代理的检索-生成诊断反馈。

**📊 数据集**

数据集基于 Wikipedia 及 PROCON 网站选取 40 个根主题，构造约 5,843 个探测查询、146 个语义节点及 679,402 个检索文档。

**📈 对比分析**

在 Llama3.1-8B 与 Qwen3-8B 以及三种检索器上进行评测，DiscourseFlip 在 RASR、覆盖率、ASV 等指标均显著优于 PoisonedRAG、Topic‑FlipRAG 与 Unic‑RAG，且在 10 文档预算下仍保持高达 60‑80% 的覆盖率与 20‑30% 的意见偏移。

**⚠️ 局限性**

主要局限包括：仅在黑盒设置下验证；依赖代理模型和投毒文档的语义质量；在更强的多模态检索或极大检索窗口下的效果尚未完全验证；缺乏对持续投毒或动态防御的鲁棒性评估。

---

## 5. Conditioned free-energy density of proteins using unbalanced solutions to constraint satisfaction problems

**arXiv ID:** 2606.01329 | [PDF](https://arxiv.org/pdf/2606.01329v1)

**作者:** Pratik Worah `[一作]` (New York University), Srinivasa Varadhan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了将计算非均匀Curie‑Weiss体系自由能转化为无平衡 2→1 归一化问题，并设计了可证明逼近误差的 SDP 算法；随后利用该框架从已知晶体结构构建蛋白质自旋哈密顿量，系统探索其条件自由能景观，并识别可变回骨结构。

**💡 创新点**

创新点包括：① 通过变分表征将自由能与无平衡 2→1 归一化联系起来；② 提供了多参数（ε、ξ） SDP 形式和随机舍入方案；③ 在蛋白质结构预测中首次将该优化方法与 Karplus 方程结合，得到可控的条件 Ramachandran 图。

**🔧 技术方法**

主要技术手段包括半正定规划（SDP）与随机舍入、Karplus 方程的正向与反向推算、熵预算（y）与条件参数 ε 的扫描、以及基于二次耦合矩阵的无平衡 2→1 归一化求解。

**📊 数据集**

使用的数据集是人类泛素（Ubiquitin）PDB 1UBQ（76 个残基）的晶体结构，并通过该结构得到的 ^3J_HC 和 ^2J_Cα‑N 互耦合常数构造自旋哈密顿量。

**📈 对比分析**

与传统仅考虑能量（y = n）的做法相比，利用自由能最优预算 y* ≈ 592 的 SDP 方案得到的角度 RMSD 约 20.6°（相对 28°），并在不同 ε 水平下保持 β‑链不变、α‑螺旋子类型仅变动 1–2 个残基；无平衡因子随 ε 递减从 1.8 降至 1.2，证明算法在多尺度条件下保持近似最优，且运行时间为多项式级。

**⚠️ 局限性**

局限性在于：① 近似误差依赖于实例，缺乏统一的常数因子保证；② 采用简化的 ±1 自旋模型，无法完整描述连续二面角和多体相互作用；③ Karplus 系数仅针对 1UBQ 拟合，通用性待验证；④ 由于使用已知晶体结构作为参考，方法主要用于对已知折叠的细微扰动；⑤ 未能与实验测得的激发态或过渡态结构直接验证。

---

## 6. Interaction-Limited Safe Continuous-Time RL for Dynamical Medical Treatment

**arXiv ID:** 2606.01051 | [PDF](https://arxiv.org/pdf/2606.01051v1)

**作者:** Xun Shen `[一作]` (Tokyo University of Agriculture and Technology), Kenji Wakabayashi `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 1443 | [OpenAlex ID](https://openalex.org/A5079804151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种交互次数受限的安全连续时间强化学习框架，用以在医疗治疗中同时优化药物剂量与临床干预时机，并通过选项（options）构建连续时间决策模型。

**💡 创新点**

创新点包括：① 将连续时间治疗问题重新表述为基于选项的半马尔可夫决策过程；② 设计安全紧缩机制，只在交互时刻施加约束即可保证整个连续时间轨迹的安全；③ 在有限样本下给出学习收敛与安全性保证，并提供可通过数据驱动构造的保守安全近似。

**🔧 技术方法**

采用的技术包括：选项框架（option-based SMDP）、安全紧缩理论、重要性加权样本估计、核密度估计或高斯过程构造保守安全函数、以及现有的安全强化学习算法（如 CPO）进行近似求解。

**📊 数据集**

使用 MIMIC‑III 败血症（sepsis）患者的临床数据，通过物理信息神经网络（PINN）构建连续时间治疗环境，构造了高维、随机的生理动力学模型。

**📈 对比分析**

与等间距交互策略（equidistant interaction）相比，实验在同一患者环境下使用多种安全与非安全优化器（CPO, SAC, TRPO）进行比较，结果显示：自适应交互时机下的 CPO‑O、PPO‑O 等方法在治疗效果（SOFA、乳酸水平）和安全率（符合乳酸阈值）方面均优于等间距方案；同时，安全约束方法在不降低治疗效益的前提下进一步提升安全性。

**⚠️ 局限性**

局限性包括：实验仅在单个患者的合成环境中验证，缺乏多患者泛化与个性化；离线学习在数据覆盖与分布外情况上的鲁棒性未充分验证；SOFA 分数提升幅度有限，尚需临床验证其实际意义；对安全紧缩参数的选择依赖假设与估计，可能导致过度保守或欠保守。

---

## 7. DrugClaw and DrugAudit: A Primary-Source-Grounded Agent and Authority-Aware Benchmark for Drug-Information Question Answering

**arXiv ID:** 2606.01434 | [PDF](https://arxiv.org/pdf/2606.01434v1)

**作者:** Qing Wang `[一作]` (University of Florida), Qianqian Song `[通讯]` (University of Florida)

**通讯引用:** 4600 | [OpenAlex ID](https://openalex.org/A5101825312)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多智能体检索增强药物问答系统（DrugClaw），并构建了3,772条药物信息问题与答案的权威级基准（Drug-QA），评估其答案的事实准确性、引文来源的权威性和可信度。

**💡 创新点**

创新点包括：① 反射驱动的状态机工作流，能自适应检索深度并避免信息不足时的虚假生成；② 基于57个专业药物知识技能的封闭注册表，保证每条答案都能追溯到监管或同行评审的原始记录；③ 权威感知评估面板，将上游主源与下游聚合器区分开来，并引入信任度、语义重叠、引用质量等多维指标；④ 双判定LLM评测协议，使用与候选系统无关的Llama‑3.1‑70B与gpt‑oss‑120b两大模型，获得κ=0.88的高度一致性。

**🔧 技术方法**

技术栈主要包括：多智能体架构（Plan、Retrieval、Graph Builder、Re‑Ranker、Response、Reflector、Code Agent、Web Search）；安全沙盒化的代码生成检索；GPT‑5‑mini 作为规划核心；Llama‑3.1‑70B‑Instruct 与 gpt‑oss‑120b 作为评测判定者；Token‑Jaccard 语义重叠度量；回归惩罚的反射更新；以及结构化证据输出模式。

**📊 数据集**

数据集包括：① 自主构建的Drug‑QA（3,772条），覆盖ChEMBL、DrugCentral、FAERS、Orange Book、Label、LiverTox、PharmGKB、SIDER 等九大源；② 过滤后的MedQA-USMLE（751题）和PubMedQA（512题）药物相关子集，保持原始分布；③ 训练与评测时均使用公开的 FDA、PubMed、DrugBank、ChEMBL 等数据库。

**📈 对比分析**

对比七个系统（DrugClaw linear/graph、Biomni、DeepEvidence、ToolUniverse、GPT‑5‑mini、GPT‑5），通过主源率、答案可信度、Evidence Index等指标以及双判定得分进行评估。DrugClaw在主源率（0.918）和可信度（0.887）上均领先；在MedQA和PubMedQA上分别取得0.920和0.693的准确率；在综合Evidence Index上得到最高分（0.632/0.623），并在两位判定者上保持κ=0.88。

**⚠️ 局限性**

局限性包括：① graph 模式生成的多主张答案在LLM评测时较难解析，导致未解析判定率高；② linear 模式的κ略低（0.79），主要因对简短答案的评测严格；③ 封闭注册表限制了对长尾或新批准药物的覆盖；④ 失误率与拒绝率的平衡需进一步调优；⑤ 评测依赖公开权威模型，仍可能受判定模型偏差影响。

---

## 8. PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making

**arXiv ID:** 2606.01313 | [PDF](https://arxiv.org/pdf/2606.01313v1)

**作者:** Rufeng Chen `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Sihong Xie `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了PSG-Nav框架，实现了在开放词汇导航任务中利用3D概率场景图进行感知、规划与决策；

**💡 创新点**

创新点包括：①构建保留完整类别分布的3D概率场景图，消除单标签确定性导致的误判；②通过多宇宙决策采样多种可能世界，结合对比式LLM推理实现对语义不确定的边际化；③引入证据经验校准器（EEC），利用过去成功/失败记忆对检测结果进行在线自适应校准；

**🔧 技术方法**

技术手段包括：基础模型GLIP、Grounded‑SAM、Qwen2.5 LLM；层次化概率图推理与蒙特卡罗采样；对比式对偶比较策略；检索增强生成（RAG）经验校准；

**📊 数据集**

使用了MP3D、HM3D和HSSD三个Habitat基准数据集；

**📈 对比分析**

在三大数据集上与多种SOTA方法（SG‑Nav、BeliefMapNav、ASCENT、ApexNav等）对比，PSG‑Nav在HM3D、MP3D、HSSD上分别取得66.1%、44.8%和67.9%的成功率，显著优于现有方法；

**⚠️ 局限性**

局限性：仅使用2D占据地图，无法处理垂直结构如楼梯、电梯等多层环境；LLM对比推理带来一定延迟；多宇宙采样和对比计算在大规模环境下仍需进一步优化；

---

## 9. Plausibility Is Not Prediction: Contrastive Evidence for LLM-Based Cellular Perturbation Reasoning

**arXiv ID:** 2606.01042 | [PDF](https://arxiv.org/pdf/2606.01042v1)

**作者:** Xinyu Yuan `[一作]` (Mila - Quebec AI Institute), Jian Tang `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为CORE的框架，将扰动差异表达预测任务转化为对同一基因下正负支持证据的对比推理，显著提升LLM在药物和CRISPR扰动中的精准度。

**💡 创新点**

创新点在于通过对同一基因的正负支持案例进行对比组织（contrastive organization），而非传统的孤立可解释性推理，从而解决了LLM过度乐观、缺乏扰动特异性的问题。

**🔧 技术方法**

核心技术包括构建统一的生物医学知识图谱ReasonKG、基于KG检索的相似扰动签名、加权投票的对比证据包，以及将该证据作为结构化提示交给LLM进行推理。

**📊 数据集**

使用了两大基准数据集：Tahoe100M（C32细胞中的药物扰动）和PerturbQA（K562、HepG2、Jurkat、RPE1细胞中的CRISPRi扰动）。

**📈 对比分析**

在与VCWorld、SUMMER等LLM推理以及STATE、PerturbDiff等单细胞扰动预测模型对比时，CORE在aggregate AUROC、macro‑per‑gene AUROC、F1等指标上均实现显著提升（例如Qwen3.5‑9B的macro AUROC从0.500升至0.711），同时校准率亦大幅改善。

**⚠️ 局限性**

局限性包括仅处理二值差异表达预测、固定细胞环境、对知识图谱和支持集覆盖度敏感，且缺乏剂量、时间、实验条件等多维因素的建模。

---

## 10. Feature to Dynamics: Feature-space to Autoregression strategy for Zero-shot Time Series Forecasting

**arXiv ID:** 2606.01289 | [PDF](https://arxiv.org/pdf/2606.01289v1)

**作者:** Yifan Wu `[一作]` (Xidian University), Jian Lou `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 2122 | [OpenAlex ID](https://openalex.org/A5048843406)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种零样本单变量时间序列预测框架FSA，将预测任务从直接预测值转化为生成自回归策略；

**💡 创新点**

创新点在于：①以结构化特征（全局趋势、周期、残差统计和局部动态）构造任务特征空间；②利用该特征生成低维自回归参数，从而在零样本条件下实现可解释、可迁移的预测策略；

**🔧 技术方法**

技术包括：基于Chebyshev多项式和DFT的全局特征提取，局部窗口统计，3层MLP策略生成网络，带边界tanh的AR参数化，递归自回归解码以及周期性重估循环；

**📊 数据集**

使用了大规模多域预训练数据（Azure VM、Borg Cluster、Alibaba Cluster、PEMS07、Q-Traffic、Traffic Hourly等共104,807样本），并在七个零样本评测集（ETT系列、Electricity、Exchange Rate、Weather）进行验证；

**📈 对比分析**

与容量匹配的Transformer、PatchTST、TimesFM、Chronos等基线在相同预训练数据与模型规模下对比。FSA在所有预测时延（24、48、96步）下平均MSE最低，且在数据受限、样本稀缺场景下表现出更强的样本效率；

**⚠️ 局限性**

局限性包括：仅处理单变量预测，无法建模多变量交互；解码器仍为线性AR形式，对高频、突变或外源驱动的复杂动力学适应性有限；

---

## 11. Coordinating Task Switching in a Robotics Multi-Agent System Using Behavior Trees

**arXiv ID:** 2606.01170 | [PDF](https://arxiv.org/pdf/2606.01170v1)

**作者:** Lucas Haug `[一作]`, Arthur Casals `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在IEEE VSSS（Very Small Soccer）比赛中，将原本基于有限状态机（FSM）的三机器人团队协调方案改为基于行为树（BT）的协调方案，实现在团队角色动态分配与切换；

**💡 创新点**

创新点在于采用行为树实现团队领导者的角色分配，提升了模块化、可维护性与可扩展性，并通过黑板机制实现状态共享与优先级控制；

**🔧 技术方法**

使用了BehaviorTree.CPP库构建BT，结合黑板（Blackboard）实现共享状态，改进了FSM中的状态切换逻辑；

**📊 数据集**

使用的测试数据集为FIRASim模拟器中的250场10分钟对战数据，以及在IRONCup 2023虚拟比赛中的对战结果；

**📈 对比分析**

比较方法：在相同场景下进行250场对战，统计胜率、进球数、进球率等指标，BT团队的胜率从22.8%提升到34.8%，进球数从140提升到193，进球率从0.56提升到0.772，统计显著性检验p=0.00243；

**⚠️ 局限性**

局限性：仅在仿真和虚拟比赛环境中验证，缺乏真实硬件测试；评估范围仅限VSSS比赛，未验证在更大规模或不同任务下的表现；

---

## 12. On the Generalization Gap in Self-Evolving Language Model Reasoning

**arXiv ID:** 2606.01075 | [PDF](https://arxiv.org/pdf/2606.01075v1)

**作者:** Zhenting Qi `[一作]` (Google Research), Cyrus Rashtchian `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在严格的闭环自进化（self‑evolution）框架下，评估多种基于生成器–验证器的离线偏好学习方法，比较其与oracle监督的性能差距，并分析规模、阈值和迭代策略对模型提升的影响。

**💡 创新点**

首次系统化比较不同自进化细粒度（单轮验证、迭代、复习、课程学习）在同一基线模型下的效果；揭示模型规模为“能力阈值”，大模型在复习式自进化中可逼近oracle水平；提出在计算成本上优先扩展验证器推理而非生成候选的策略。

**🔧 技术方法**

采用生成器–验证器游戏、阈值投票、自然语言反馈迭代、Direct Preference Optimization (DPO) 以及多轮/多任务的离线自进化流程；结合Oracle验证与多轮复习。

**📊 数据集**

主要使用 Knights‑and‑Knaves (KK) 逻辑推理数据集；此外在 OpenThoughts3、GSM8K、MATH500、MATHHard、TabMWP 等多任务推理数据集上做跨任务评估。

**📈 对比分析**

通过与oracle‑监督（精确答案）以及公开的在线自进化方法（AZR、INTUITOR）对比；结果显示：在KK任务上，Gemma‑3‑4B 从 31.0% 提升至 44.8%（单轮）或 44.8%（课程学习），但仍低于 53.3% 的oracle；在 Gemma‑3‑12B 的复习式自进化中达 52.8%，接近 oracle 53.6%；在开放式推理任务中提升幅度仅 1–3%。

**⚠️ 局限性**

主要局限：①闭环自进化受模型内部验证噪声限制，尤其是小模型；②缺乏可验证的外部奖励导致难以突破已知解空间；③在开放式任务中，内部验证难以区分多种合理但错误路径；④计算成本随阈值提升和多轮迭代显著上升，需更高效的采样与验证策略。

---

## 13. AcOrch: Accelerating Sampling-based GNN Training under CPU-NPU Heterogeneous Environments

**arXiv ID:** 2606.01161 | [PDF](https://arxiv.org/pdf/2606.01161v1)

**作者:** Kefu Chen `[一作]` (Northeastern University), Ge Yu `[通讯]` (Northeastern University)

**通讯引用:** 6546 | [OpenAlex ID](https://openalex.org/A5072406974)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于采样的图神经网络训练系统 AcOrch，针对CPU‑NPU异构环境进行任务细粒度调度和双层流水线。

**💡 创新点**

创新点在于：①将采样拆分为 CPU 与 AIV 两路并按工作负载动态划分；②将图聚合迁移至 AIC，实现 AIC 充分利用；③采用两级流水线和共享队列实现采样、收集、训练的异步重叠；④基于度和历史采样时间的成本模型实现负载感知划分。

**🔧 技术方法**

使用的技术包括：采样负载分配算法、成本模型（α·deg + β·time）、双路采样、共享队列、异步双缓冲流水线、SpMM 迁移、Ascend 910B 的 AIC/AIV 计算单元与 MTE 内存搬移。

**📊 数据集**

使用的公开图数据集包括：Reddit、Amazon、Wiki‑Talk、Products、Livejournal、Orkut。

**📈 对比分析**

对比方法：与 Ascend 原生框架 MindSporeGL 以及 GPU 端 Quiver 进行对比；实验表明在六个数据集上平均加速 2.31×，AIC 利用率提升 52.63%；在 GPU 对比中，Ascend 910B 取得 95.1% 的端到端性能，并拥有更高利用率和更低功耗。

**⚠️ 局限性**

局限性：仅支持单机单卡，假设全部数据可放入 CPU 内存；尚未实现分布式多 NPU 训练；在极大图规模下仍受 CPU 内存与 PCIe 带宽限制。

---

## 14. Leaf Spectral Reflectance Prediction Using Multi-Head Attention Neural Networks

**arXiv ID:** 2606.01432 | [PDF](https://arxiv.org/pdf/2606.01432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 15. Challenger at MultiPRIDE: Is It Hate Speech or Reclaimed?

**arXiv ID:** 2606.01298 | [PDF](https://arxiv.org/pdf/2606.01298v1)

**作者:** Hadi Bayrami Asl Tekanlou `[一作]` (University of Tabriz), Jafar Razmara `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套轻量级多语言处理管道，结合句子嵌入、Cleanlab标签去噪与MLP分类，旨在区分仇恨言论与再主张语言。

**💡 创新点**

创新点在于将Cleanlab去噪与统一多语言句子嵌入相结合，使用阈值自适应优化，保持模型解释性与低资源使用；并通过backtranslation增强少数类样本。

**🔧 技术方法**

使用了intfloat/e5-large-v2句子Transformer生成128维嵌入、Cleanlab置信学习与Logistic回归进行标签过滤、轻量级MLP网络（两隐藏层16/8单元），以及阈值搜索优化。

**📊 数据集**

使用了MultiPRIDE 2026共享任务数据集，包括英语、意大利语、西班牙语的Twitter推文，标签为reclaimed或not reclaimed。

**📈 对比分析**

通过Precision、Recall、F1及宏平均指标在各语言上进行评估，英文macro F1约0.62，意大利约0.88，西班牙约0.68，整体性能优于基线，但受到类不平衡的影响。

**⚠️ 局限性**

局限性包括类不平衡导致英文性能下降、标签噪声、缺乏上下文信息、模型规模受限于轻量化设计，且未尝试更大模型或多模态特征。

---

## 16. 3DCodeBench: Benchmarking Agentic Procedural 3D Modeling Via Code

**arXiv ID:** 2606.01057 | [PDF](https://arxiv.org/pdf/2606.01057v1)

**作者:** Yipeng Gao `[一作]` (Google DeepMind), Jindong Chen `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了3DCodeBench基准和3DCodeArena评测平台，用于评估 Vision‑Language 模型在编写可执行 Blender Python 代码生成 3D 物体的能力。

**💡 创新点**

创新点包括：① 通过 Agentic curation pipeline 从 Infinigen 自动生成 26K 高质量（文本/图像 ↔ 代码 ↔ 3D 物体）三元组；② 将评估从单轮生成扩展为多轮交互式 Agentic 迭代，并引入自动化的感知与几何度量与人类 Elo 排名相结合的评价框架。

**🔧 技术方法**

主要技术手段包括：Vision‑Language 模型 + 代码生成、Agentic feedback loop（技能库、经验库、代码简化、模拟执行、视觉自评）、Blender 5.0 运行环境、SigLIP‑2、DINOv3、Chamfer、Uni3D 等多模态度量，以及基于 Elo 的人类偏好投票。

**📊 数据集**

使用的数据集为从 Infinigen 提取的 212 类物体，包含 26K（提示、代码、3D 物体）三元组；此外还收集了 12,963 条可训练用的代码–物体样本。

**📈 对比分析**

评估方法对 12 个前沿 VLM 进行单轮与多轮对比；单轮可执行率约 70%，多轮错误反馈后提升至约 97%；在自动化度量与人类 Elo 之间的相关性极高（SigLIP‑2 相关系数 0.964，DINOv3 相关系数 0.972），但形状精度提升有限；思考预算对轻量模型效果显著，而高阶模型提升有限。

**⚠️ 局限性**

局限性在于：① 仍缺乏对物理可行性和复杂几何推理的深度掌握；② 评估聚焦单物体级别，未覆盖场景级合成或跨平台（如 Houdini、Unreal Engine）等；③ 依赖 Blender 5.0 API，难以消除 API 记忆导致的偏差。

---

## 17. GPU Acceleration of Learning With Errors KEMs Using OpenACC for Post-Quantum Cryptography

**arXiv ID:** 2606.01211 | [PDF](https://arxiv.org/pdf/2606.01211v1)

**作者:** Tiziana Liberati `[一作]` (E4 Computer Engineering SpA), Marco Pedicini `[通讯]` (Roma Tre University)

**通讯引用:** 444 | [OpenAlex ID](https://openalex.org/A5029371057)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了基于Plain LWE的KEM，并实现了OpenACC指令式GPU加速版本。

**💡 创新点**

创新点在于将纯LWE算法通过OpenACC实现，并结合设备内随机数生成、批处理与显式数据管理，显著提升了计算效率。

**🔧 技术方法**

使用技术包括OpenACC并行指令、RNGonGPU AES-256 DRBG、批处理策略、显式数据迁移以及CUDA集成。

**📊 数据集**

使用多组LWE参数集（n=32-65536）作为实验数据，并在不同GPU上进行性能评测。

**📈 对比分析**

通过与CPU OpenMP 72线程实现对比，GPU实现实现速度提升高达208×，能耗下降约1.78×。

**⚠️ 局限性**

限制在于受限于内存带宽、CPU侧的Fujisaki‑Okamoto变换开销以及对AMD/Intel GPU可移植性的不足。

---

## 18. ANDES: Agent Native Data Evolving Synthesis Tool for Autonomous Instruction Alignment

**arXiv ID:** 2606.01279 | [PDF](https://arxiv.org/pdf/2606.01279v1)

**作者:** Zhengyang Zhao `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 15556 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Andes，一个可供代理调用的动态数据合成工具，帮助自动化后训练阶段生成高质量、多样化的训练数据。

**💡 创新点**

将数据合成转化为代理可调用的技能，配合自演进的 World Tree 路由与诊断报告，实现闭环自适应生成，显著降低代理能力门槛。

**🔧 技术方法**

自演进主题-场景树、两阶段 QA 生成与改进、基于诊断报告的配置更新、代理工具调用框架。

**📊 数据集**

在 PostTrainBench 及其七个子基准（GSM8K、AIME、HumanEval、BFCL、GPQA-Main、HealthBench、ArenaHardWriting）上评估，同时利用自建的多层主题树。

**📈 对比分析**

与 GLM-4.7、Opus-4.7 等基线在 PostTrainBench 上对比，平均分从 7.48% 提升至 33.39%，超过 Opus-4.7 的 28.56%，并在各基准上持续领先。

**⚠️ 局限性**

仍依赖代理对诊断报告的理解与配置更新能力，且在极端资源受限或任务超出主题树覆盖范围时可能受限。

---

## 19. Learning from Saturated Data: Signals Beyond Correctness for LLM Training

**arXiv ID:** 2606.01436 | [PDF](https://arxiv.org/pdf/2606.01436v1)

**作者:** Hanno Hiss `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11376 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对已被LLM完美解决的“饱和”问题，使用质量信号（自评pairwise和逆熵）进行后训练，以提升在更难问题上的表现。

**💡 创新点**

引入准确性之外的质量评分作为训练信号；利用模型自评和token‑level熵两种方式对正确答案进行细粒度排序；在饱和数据上应用DPO和σ‑RRHF，展示能够超越SFT的效果。

**🔧 技术方法**

使用DPO、σ‑RRHF（加权RRHF）等后训练方法；采用自评pairwise和逆熵作为质量信号；与基线SFT、随机评分和强评判器进行对比。

**📊 数据集**

ReasoningGym的Arithmetic任务（6–10项、1–3位数），GSM8K（含sat和strict‑sat子集），以及同一模型族的指令调优版本。

**📈 对比分析**

与基线SFT、DPO、σ‑RRHF、随机评分进行比较；在Arithmetic上，σ‑RRHF+inverse entropy提升pass@1 18.6%（最高29.25%），在GSM8K提升有限（+3.3%），自评评分甚至导致性能下降；在指令调优模型上无显著提升。

**⚠️ 局限性**

仅在单一模型族实验，方法对不同任务的适用性差异大；仅使用两种质量信号，其他更好指标待探索；仅做离线训练，未评估RL集成；质量评估器的可靠性需进一步校准。

---

## 20. Conservative Discrete Structure Stabilizes Autoregressive Rollouts in a 1D Drift Diffusion Poisson Benchmark

**arXiv ID:** 2606.01366 | [PDF](https://arxiv.org/pdf/2606.01366v1)

**作者:** Yufeng Wang `[一作]` (Stony Brook University), Haibin Ling `[通讯]` (Westlake University)

**通讯引用:** 36954 | [OpenAlex ID](https://openalex.org/A5061469520)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究在一维漂移扩散泊松（DDP）等离子体模型中，构建并评估了一种保守结构的学习型数值逼近器，探讨其在长期自回归推演中的物理一致性。

**💡 创新点**

创新点在于将保守有限体积更新、泊松兼容电场重构以及密度可正性感知限幅等数值结构与神经网络修正相结合，证明了保守结构是实现长时间稳定推演的关键，而非单纯的神经网络精度提升。

**🔧 技术方法**

技术上使用了共享面向量的多层感知机作为修正器，保持节点电位与面电场的离散一致性，采用显式欧拉时间步和正性限幅器，并在训练中加入电荷守恒和正性损失。

**📊 数据集**

数据集为合成的DDP轨迹，基于非维一维网格（默认 16 网格），生成 64 条随机初始密度轨迹（共 4096 步），并在不同配置下重复实验。

**📈 对比分析**

对比方法包括直接下一步状态回归网络、其 Poisson 重算、全局电荷投影以及四步训练版本，经典有限体积核心（无学习修正）以及保守学习模型。实验显示保守模型在 64 个预设配置中以 60/64 的比例在滚动均方误差上胜过直接回归模型，且电荷误差始终保持在舍入误差水平，密度正性也保持良好；直接回归模型在滚动误差上表现差强人意。

**⚠️ 局限性**

局限性包括：仅针对零壁面通量与 Dirichlet 电势的简化 1D 版 DDP；学习修正尚未显示在核心结构之外的显著改进；高分辨率下的中心显式扩散不稳定，需要更稳健的上风或 Scharfetter‑Gummel 离散；缺乏真实壁面物理、噪声数据和多维验证。

---

## 21. SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories

**arXiv ID:** 2606.01311 | [PDF](https://arxiv.org/pdf/2606.01311v1)

**作者:** Zhuoyun Yu `[一作]` (Zhejiang University), Shumin Deng `[通讯]` (Zhejiang University)

**通讯引用:** 2895 | [OpenAlex ID](https://openalex.org/A5060484186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练‑free、基于步骤级失效归因的技能适配框架（SkillAdaptor），能够在执行失败轨迹中定位首个可操作的错误步骤，识别相关技能并进行针对性修改或生成新技能；

**💡 创新点**

核心创新在于将技能适配从全轨迹/会话级别转为步骤级归因，精确定位错误源，避免过宽或误导性更新，并引入资格验证门控以保证更新稳定；

**🔧 技术方法**

技术包括：①步骤级失效定位与责任链路（Localizer & Linker）来评估错误步骤与技能责任；②基于检索与重排序的技能注入；③对技能修改（修订/生成）与资格验证（执行回测检验收益）；④使用LLM（Kimi‑K2.5/GLM‑5/GPT‑5.2）与文本嵌入（Qwen3‑Embedding‑8B）进行检索与编码；

**📊 数据集**

评估数据集包括WebShop、PinchBench和Claw‑Eval三大公开基准；

**📈 对比分析**

与无技能基线、现有训练‑free适配基线（如A‑Mem、AWM、ExpeL、EvoSkill）对比，SkillAdaptor在所有基准上均取得提升；最显著提升为WebShop得分+1.7（Kimi‑K2.5）、PinchBench平均分+1.5（GLM‑5）、Claw‑Eval平均分+1.8（Kimi‑K2.5），并在大部分指标上保持稳定增长；

**⚠️ 局限性**

局限性：①在失效信号稀疏或延迟反馈、缺失外部接口的环境下效果可能下降；②目前仅在三大公开基准上验证，尚缺乏长期部署与更广泛分布漂移场景的评估。

---

## 22. Beyond Visual Memory: Mechanistic Diagnostics of Latent Visual Reasoning

**arXiv ID:** 2606.01287 | [PDF](https://arxiv.org/pdf/2606.01287v1)

**作者:** Garvin Guo `[一作]` (Amap, Alibaba Group), Shuai Dong `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过拆分潜在视觉推理方法中的潜在标记为槽内容、边界标记和格式，并设计了内容友好的探针（Mesoscale Visual Hypotheses）来系统评估潜在槽是否真正作为可恢复的视觉记忆。对多种监督策略和阶段设置进行干预实验，结果表明大部分潜在标记的性能提升来自边界标记、格式约束和局部视觉注意，而不是槽内容本身。

**💡 创新点**

创新点在于：1）首次将潜在标记拆分为可独立干预的三部分，并在实验中单独评估其贡献；2）提出内容友好的探针，使得视觉记忆假设在最有利条件下得到严格检验；3）通过多项干预实验揭示潜在槽不具备可恢复视觉记忆的特性，重新定义了潜在视觉推理的机制。

**🔧 技术方法**

技术方法包括：多阶段训练（SFT+RL）与教师监督、连续视觉状态插入、多模态大型语言模型解码、槽内容对齐（余弦损失）、视觉子空间对齐、注意力熵差分析、标记替换与零化干预、槽内容交换与注入实验等。

**📊 数据集**

使用的视觉数据集有：V^*、HRBench-4K、HRBench-8K、MME-RealWorld-Lite，涵盖细粒度识别、高清图像推理、空间推理和真实世界视觉理解。

**📈 对比分析**

与其他主流潜在视觉推理方法（ILVR、Monet 等）在相同 backbone（Qwen 系列）下进行公平比较。探针在 SFT 阶段已达到当前最优准确率；在 RL 阶段进一步提升。实验显示，虽然整体准确率相近，但不同方法在边界标记依赖和槽-视觉向量相似度上差异显著，说明准确率并不能揭示其内部机制。

**⚠️ 局限性**

局限性包括：1）仅研究了全图 VLM 接口并在解码序列中插入潜在槽，未覆盖强视觉瓶颈或外部视觉工作空间的模型；2）实验集中在 Qwen 系列 backbone，缺乏对其它模型族、视频、体感任务的验证；3）探针只检验槽是否为可恢复视觉记忆，未探索槽在计算、时序或路由等其他潜在功能。

---

## 23. Formal Verification of Secure Encrypted Virtualization

**arXiv ID:** 2606.01381 | [PDF](https://arxiv.org/pdf/2606.01381v1)

**作者:** Hansika Weerasena `[一作]` (University of Florida), Prabhat Mishra `[通讯]` (University of Florida)

**通讯引用:** 6422 | [OpenAlex ID](https://openalex.org/A5006818844)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一个用于表示和验证AMD SEV机密虚拟机的形式化框架。

**💡 创新点**

创新点在于将设计级和属性级抽象相结合，对AMD SEV的安全属性进行形式化检验，首次为其安全性提供严谨保证。

**🔧 技术方法**

作者采用了形式化建模与属性检验技术，结合模型检查工具来验证机密性、完整性和可用性。

**📊 数据集**

论文没有使用传统的数据集，而是基于AMD SEV的规范构造抽象模型进行实验。

**📈 对比分析**

通过模型检查结果与已知安全分析方法对比，证明所构建模型能够捕获潜在的安全缺陷，性能满足理论可行性要求。

**⚠️ 局限性**

局限性在于只针对设计阶段的抽象模型，无法覆盖运行时硬件缺陷或软件漏洞；此外，对复杂多租户场景的可扩展性尚未验证。

---

## 24. Differentially Private Datastore Generation for Retrieval-Augmented Inference

**arXiv ID:** 2606.01413 | [PDF](https://arxiv.org/pdf/2606.01413v1)

**作者:** Abdelrahman Abouelenein `[一作]` (Alexandria University), Marwan Torki `[通讯]` (Alexandria University)

**通讯引用:** 2681 | [OpenAlex ID](https://openalex.org/A5037423841)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种利用SimHash桶化与Laplace噪声的单次纯ε‑差分隐私数据存储发布框架，使检索增强推理可在不产生额外隐私开销的前提下使用可重用的键值数据集。

**💡 创新点**

创新点在于：①在桶级别进行类别投票统计并一次性对整个数据集加入Laplace噪声，提供纯ε‑DP可重用数据集；②通过向所有桶（包括空桶）添加噪声避免空桶泄露；③系统性评估了对成员推断攻击的鲁棒性。

**🔧 技术方法**

使用的技术包括：Locality‑Sensitive Hashing（SimHash）、Laplace机制、ε‑DP理论、kNN‑Prompting框架、成员推断攻击评估、以及对比实验。

**📊 数据集**

实验数据集包括MR、SUBJ、TREC、CR、AGNews、DBpedia、MPQA（共七个文本分类数据集，类别数从2到14）。

**📈 对比分析**

通过与非私有KNN‑Prompting基准比较，发现随着ε增大准确率提升；在ε=5时平均仅损失2.6%准确率；在成员推断攻击中，DP噪声将攻击准确率从70%降至≈53%，接近随机猜测。

**⚠️ 局限性**

局限性包括：需要手动调节哈希参数H/T及噪声预算；在多类别或大词表任务上的扩展尚未完成；在高隐私（小ε）下性能显著下降；仅在文本分类任务中验证，其他下游任务效果未知。

---

## 25. Lagrangian Perturbation Diffusion Steering: Latent Reinforcement Learning for Generative Policies

**arXiv ID:** 2606.01151 | [PDF](https://arxiv.org/pdf/2606.01151v1)

**作者:** Hikmet Simsir `[一作]` (Bilkent University), Ozgur S. Oguz `[通讯]` (Bilkent University)

**通讯引用:** 542 | [OpenAlex ID](https://openalex.org/A5013059855)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在冻结的生成式控制策略（扩散、流匹配等）上学习噪声空间的状态条件残差，从而进行在线自适应改进；不需要重新训练解码器。

**💡 创新点**

提出拉格朗日信任域约束的残差噪声优化目标，既能提升奖励又能防止噪声偏离先验分布导致的模式崩溃；同时实现轻量化的自适应模块。

**🔧 技术方法**

使用拉格朗日方法、残差噪声网络、Q 价值网络、经验回放、DDIM/流匹配等生成式模型；实现对噪声空间的可微调。

**📊 数据集**

RoboMimic 机械操作、OpenAI Gym 运动学环境、Adroit 细粒度操控、LIBERO 视觉语言动作任务，以及在真实 Franka Panda 机器人上进行的实地实验。

**📈 对比分析**

与 DSRL、DPPO、IDQL、DQL 等基线相比，LP‑DS 在多任务上平均提升 10–25% 回报，样本效率更高，且保持更高的动作熵，显示出更好的多模态保留。

**⚠️ 局限性**

需要手动调节信任域阈值 δ，过大易导致模式崩溃，过小则收敛慢；目前主要适用于已充分预训练且数据覆盖完整的生成式策略，尚未验证在极高维或部分可观测环境中的鲁棒性。

---

## 26. Spiking and Event-driven Neuromorphic Mamba Models for Efficient Speech Recognition

**arXiv ID:** 2606.01135 | [PDF](https://arxiv.org/pdf/2606.01135v1)

**作者:** Tauseef Ahmed `[一作]` (Maastricht University), Guangzhi Tang `[通讯]` (Maastricht University)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5037310533)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在SpeechMamba ASR模型上引入事件驱动和脉冲神经网络，提升激活稀疏度，构建了FATReLU激活和LIF脉冲版本，并开发周期精确的RISC‑V Ibex模拟器进行硬件评估。

**💡 创新点**

创新点包括多阶段稀疏化训练管线、阈值自适应FATReLU、脉冲驱动的Mamba块、以及面向硬件的事件驱动模拟器，实现算法与硬件的协同优化。

**🔧 技术方法**

采用FATReLU激活、LIF脉冲神经元、状态空间模型、稀疏性正则化、阈值扫频、以及RISC‑V Ibex周期精确的事件驱动模拟技术。

**📊 数据集**

使用LibriSpeech数据集进行训练与评估。

**📈 对比分析**

与Whisper‑Large‑V2、Pruned Conformer、SpeechMamba、Spike‑driven Transformer、IML‑Spikeformer等方法对比，E‑SpeechMamba在保持<1% WER增加的前提下实现60%稀疏度，S‑SpeechMamba稀疏度超过70%，优化版实现64%稀疏度，硬件模拟表明CPU周期降低最高可达46%。

**⚠️ 局限性**

局限在于算法稀疏度与实际硬件性能不完全匹配（高维稀疏点不足、密集子模块影响）、脉冲神经元的膜电位维护导致内存访问开销、未在真实神经形态硬件上验证、以及对特定模拟器的依赖。

---

## 27. Finite-Resolution Information from Collision Statistics

**arXiv ID:** 2606.01218 | [PDF](https://arxiv.org/pdf/2606.01218v1)

**作者:** Alexander J. Gates `[一作]` `[通讯]`, Alexander J. Gates

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出基于有限阶碰撞统计量（低阶碰撞概率）构造的离散分布的有限分辨率熵与互信息近似方法，分析其估计误差与逼近误差并证明有限阶碰撞统计无法唯一确定Shannon熵。

**💡 创新点**

创新点在于将碰撞矩（即整数阶Rényi熵）视为对Rényi熵路径的离散观测，构造多阶插值逼近Shannon熵和互信息，并清晰区分统计估计误差与结构逼近误差，同时证明有限阶碰撞统计不可识别熵。

**🔧 技术方法**

采用U‑统计量实现无偏碰撞概率估计；利用插值多项式（Lagrange）对Rényi熵路径进行外推；使用Taylor/插值剩余定理给出逼近误差界；证明非可识别性时用隐函数定理和多项式插值。

**📊 数据集**

使用合成数据（如二进制对称信道、离散分布、层级分布等）进行数值实验，无真实数据集。

**📈 对比分析**

与传统经验替代估计器（plug‑in、Miller–Madow）和NSB等Bayesian熵估计器比较。实验显示，碰撞近似在小样本时方差较小，但存在固定的逼近偏差；随着阶数提升逼近误差减小但估计方差增大；总体性能在稀疏或大字母表时不如直接熵估计。

**⚠️ 局限性**

局限性包括：1）有限阶碰撞统计无法唯一确定熵；2）逼近误差受Rényi熵路径高阶导数影响，缺乏统一分布下的精确界；3）高阶碰撞统计难以在有限样本中准确估计；4）所用的Rényi互信息对比仅为“对比”而非完整的Rényi互信息。

---

## 28. When Is 0.1% Enough? Analyzing the Combined Effects of Dimensionality Reduction and Quantization on Text Embedding Compression

**arXiv ID:** 2606.01074 | [PDF](https://arxiv.org/pdf/2606.01074v1)

**作者:** Riku Kisako `[一作]` (Nagoya University), Ryohei Sasano `[通讯]` (Nagoya University)

**通讯引用:** 804 | [OpenAlex ID](https://openalex.org/A5049498516)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估并比较了在文本嵌入压缩中同时使用降维（Head truncation 或 PCA+ROR）和量化（1/2/4/8/16/32 位）的方法，探讨其对四个 MTEB 任务族（分类、聚类、检索、语义相似性）的影响。

**💡 创新点**

首次从实验层面系统化地量化降维与量化协同压缩的优势，揭示不同任务对维度与位宽的敏感性差异，并提出在低位宽下使用分布自适应等距量化表能显著避免零化损失。

**🔧 技术方法**

采用 Head truncation、PCA+随机正交旋转（ROR）两种降维方式；使用等距查找表（equal‑count lookup‑table）进行 2/4/8 位量化；对 1/16/32 位采用符号/浮点映射；通过 PCA + ROR 进一步平衡维度方差。

**📊 数据集**

使用 MTEB Benchmark 的 4 类任务族（Classification、Clustering、Retrieval、STS）中选定的多语料（全部英文）数据集，对四个预训练嵌入模型（两大模型、两前缀模型）进行评测。

**📈 对比分析**

通过相对存储预算（C(d,b)=d·b）与原始 32 位向量的性能比对，提出 99% 与 90% 的性能阈值。实验显示：在 99% 阈值下，Head truncation 更稳定；在 90% 阈值下，PCA+ROR 在分类/聚类中可实现 0.1%–1% 的存储压缩且性能几乎不变；检索任务对维度更敏感，需更大预算；整体可实现 0.1%–10% 的压缩而无显著性能损失。

**⚠️ 局限性**

限制：1）仅探讨两种降维和单一量化策略，未覆盖投影学习、量子化、向量量化等更高级方法；2）仅在 MTEB 的英文子集上实验，未验证多语言或跨域效果；3）校准过程依赖于任务输入，属于任务特定且可转移性低的情形，可能高估实际部署时的压缩效果。

---

## 29. Neural Network Compression by Approximate Differential Equivalence

**arXiv ID:** 2606.01402 | [PDF](https://arxiv.org/pdf/2606.01402v1)

**作者:** Ravi Dhiman `[一作]` (IMT School for Advanced Studies), Lorenzo Valerio `[通讯]` (IIT CNR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于近似前向微分等价（ε-FDE）的神经网络压缩方法，通过聚合功能相近的神经元而非单独剪枝来减小模型规模。

**💡 创新点**

创新点在于把网络编码为多项式ODE系统，利用近似微分等价寻找功能相似的神经元块，并通过该块的聚合实现参数压缩，提供了一个可调节的ε容差参数，实现压缩与精度的平滑权衡。

**🔧 技术方法**

核心技术包括多项式ODE编码、近似前向微分等价分析、块求和聚合以及基于ERGODE工具的等价分区计算。

**📊 数据集**

实验使用了四个基于非线性动力学系统的合成数据集（反应网络、H树电路、多聚化模型、糖酵解振荡器）以及四个公开回归基准（Abalone、Metro Interstate Traffic、Individual Household Electric Power Consumption、Protein）。

**📈 对比分析**

与传统权重剪枝方法（Magnitude-Based Pruning、Wanda）进行对比，结果显示本文方法在相同或更低的参数保留比例下，均可保持或显著降低均方误差，压缩效果稳定且在高压缩率下仍保持较低误差。

**⚠️ 局限性**

局限性包括仅适用于多项式激活且可被ODE编码的网络架构；ε容差需手工设置且对不同数据集和模型大小不一定通用；当前未考虑与硬件加速或更复杂架构（如卷积、残差网络）的直接结合。

---

## 30. Not All Explanations Simulate Equally: Comparing Verbalized Feature Attributions and Self-Generated Rationales

**arXiv ID:** 2606.01148 | [PDF](https://arxiv.org/pdf/2606.01148v1)

**作者:** Pingjun Hong `[一作]` (University of Vienna), Benjamin Roth `[通讯]` (University of Vienna)

**通讯引用:** 1489 | [OpenAlex ID](https://openalex.org/A5046895021)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对比语义特征归因与自生成推理（CoT）两类自然语言解释，研究了它们在问答模型的反事实可模拟性（simulatability）上的差异，提出统一的可模拟性评估框架；

**💡 创新点**

创新点在于：①将可模拟性引入解释评估，统一比较不同来源、不同格式的解释；②系统探讨了解释的粒度、表述方式、语义推理的影响；③对解释在跨模型传递和学生模型学习中的表现进行了评估；

**🔧 技术方法**

采用了MExGen进行特征归因，并通过模板和LLM重写方式进行语义化；对自生成推理使用post-hoc与CoT提示；利用LLM评判器（GPT‑5‑mini）做两阶段预测；此外还设计了教师‑学生训练来验证解释的可学习性；

**📊 数据集**

实验数据基于SQuAD 2.0，构造了1204对原始-反事实问答样本；

**📈 对比分析**

通过比较有无解释的两阶段预测准确率（Δ EM与Δ F1）进行评估，发现句子级归因解释在归因类中效果最佳；CoT推理在所有方法中提升最大；LLM重写效果有限；跨模型CoT传递仍能提升，说明其包含通用任务信息；教师‑学生对齐最高的是CoT，归因解释则对齐较差；

**⚠️ 局限性**

局限性包括：①评估高度依赖单一LLM评判器，可能受模型偏差影响；②仅在SQuAD 2.0上验证，未检验在开放式生成或抽象摘要任务中的泛化；③使用MExGen与固定模板，未探索梯度或注意力等其他归因方法；④可模拟性并不能覆盖解释的其他质量维度（如可信度、用户理解）。

---

## 31. Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems

**arXiv ID:** 2606.01351 | [PDF](https://arxiv.org/pdf/2606.01351v1)

**作者:** Junze Zhu `[一作]` (Nanjing University), Xinyu Dai `[通讯]` (Nanjing University)

**通讯引用:** 4838 | [OpenAlex ID](https://openalex.org/A5102994315)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究多智能体系统中中心化调度器的熵动力学，提出均场熵动力学模型并验证其对调度过程的解释能力。

**💡 创新点**

将熵动力学与任务振荡、上下文扩散分离，构建逆工作流生成管道实现高分辨率调度日志，并揭示“思考陷阱”。

**🔧 技术方法**

基于均场近似的概率动力学、梯度流理论、逆规划与多代理验证委员会，实现多模型调度评估。

**📊 数据集**

自研的Inverse Workflow Generation (IWG) 高复杂度、过程可验证的任务集，涵盖研究、数据分析、网页交互等领域。

**📈 对比分析**

对比多种商用与开源LLM（Gemini、Claude、GPT‑4/5、Qwen、Llama 等）在系统层级与调度层级指标（TS、LCS‑F1、Step‑SR 等）下的表现，发现振荡型模型在早期探索强但后期失效，稳健型模型更稳定，且轻思考版本优于重思考。

**⚠️ 局限性**

实验主要基于合成日志，真实环境噪声与长上下文依旧挑战；模型评估受限于当前prompt与工具集合，未验证跨任务迁移与更大规模系统的泛化。

---

## 32. Multiagent Matroid Upgrading: Greedy is Fair and Efficient

**arXiv ID:** 2606.01309 | [PDF](https://arxiv.org/pdf/2606.01309v1)

**作者:** Qingwen Ma `[一作]` (East China Normal University), Ruilong Zhang `[通讯]` (City University of Hong Kong (Dongguan))

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了多代理矩阵升级问题（MMUP）及其图形特例（MGUP），并证明了一个简单贪心算法在多代理环境下能够获得全局最优解。

**💡 创新点**

首次展示在多代理矩阵升级框架中，利用嵌套性（nestedness）和凸性证明，单纯的贪心选择即可得到最优解，并将该结论推广到最小化最大化目标及区间公平约束。

**🔧 技术方法**

运用了图论中的最小生成树与割集性质、Matroid 理论、凸性分析以及递归构造的嵌套性证明，辅以动态贪心和组合优化技术。

**📊 数据集**

未给出任何实验数据集，论文以理论证明为主。

**📈 对比分析**

通过算法复杂度分析，贪心算法在多代理图形升级问题中实现 O(k|V|^2) 的多项式时间，证明了其最优性；论文未进行实验对比。

**⚠️ 局限性**

在非均匀升级配额或更一般的 Matroid 结构下，问题变为 NP‑hard，贪心算法不再适用，需进一步研究近似或更通用的算法。

---

## 33. IndoBias: A Dual Track Culturally Grounded Benchmark for LLMs Bias Evaluation in Indonesian Languages

**arXiv ID:** 2606.01260 | [PDF](https://arxiv.org/pdf/2606.01260v1)

**作者:** Ikhlasul Akmal Hanif `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1237 | [OpenAlex ID](https://openalex.org/A5065822589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了IndoBias基准，评估印尼及其三种地方语言（爪哇语、巽他语、马卡萨尔语）LLM的偏见。

**💡 创新点**

创新点在于提出双轨深度与广度评估、跨语言多维度偏见测度，以及使用社会科学指标（SPI、O*NET、WGI）扩展刻板性对。

**🔧 技术方法**

使用对照句对、生成式问答任务和Stereotype Index框架，以及在encoder/decoder模型上进行实验。

**📊 数据集**

数据集包括544对对照句，覆盖18个子领域，四种语言共4352句；以及336个人口统计实体的生成任务，使用Wiki数据、CommonCrawl、新闻等语料。

**📈 对比分析**

通过对比Encoder/Decoder模型的prototypical win rate和SP分数，发现decoder模型偏见更强、地方语言在宗教/意识形态域更显著，预训练语料不受控的Common Crawl导致偏见升高。

**⚠️ 局限性**

局限在于实体覆盖不完整、仅包含三种地方语言、预训练实验规模有限以及对生成任务的评估未覆盖预训练检查点。

---

## 34. Thinking Economically: A Hierarchical Framework for Adaptive-Complexity Reasoning in LLMs

**arXiv ID:** 2606.01168 | [PDF](https://arxiv.org/pdf/2606.01168v1)

**作者:** Yubo Gao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种层次自适应预算器(HAB)，通过在问题层面预测合理的推理步数范围，并在每一步内部动态分配令牌预算，从而实现链式思考的经济化。

**💡 创新点**

创新点在于：①将推理复杂度分解为跨问题的粗粒度与单步内部的细粒度两层；②使用PPL对每一步的难度进行学习；③通过自适应Pareto优化在质量与效率之间动态权衡；④引入Fisher信息引导的令牌剪枝来提供细粒度训练监督。

**🔧 技术方法**

采用LoRA微调技术、基于PPL的难度预测头、适应性Pareto目标、Fisher信息剪枝以及两阶段分层训练流程，构建完整的自适应预算框架。

**📊 数据集**

在GSM8K和MATH500这两大数学推理基准上进行评估，并通过Qwen-Max生成的CoT链构建推理深度标签；此外在AIME和LogiQA上进一步验证跨模型与跨任务的适用性。

**📈 对比分析**

与六种主流基线（Vanilla CoT、TokenSkip、CoD、Skeleton-of-Thought、Sketch-of-Thought、O1-Pruner）在准确率和平均输出令牌数两项指标上进行对比，HAB在保持甚至提升准确率的同时显著减少令牌使用，展现出最优的性能-效率折衷。

**⚠️ 局限性**

局限性包括：仅针对线性链式思考设计，尚未扩展到树形或图形思考框架；训练过程中需要额外的PPL标注和外部LLM生成数据，导致额外的计算和工程成本；自适应Pareto优化引入了一定的训练开销。

---

## 35. Low-Subpacketization MIMO Coded Caching with Flexible Stream Allocation

**arXiv ID:** 2606.01353 | [PDF](https://arxiv.org/pdf/2606.01353v1)

**作者:** Mohammad NaseriTehrani `[一作]` (University of Oulu), Antti Tölli `[通讯]` (University of Oulu)

**通讯引用:** 3870 | [OpenAlex ID](https://openalex.org/A5039827493)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种低子包化、支持灵活流分配的多天线编码缓存方案，利用虚拟广播通道分解实现多用户多天线网络的线性可解码传输。

**💡 创新点**

创新点在于将共享缓存思想推广到接收端多天线，采用分组对齐的缓存放置与虚拟MISO映射，形成可自由选择用户数Ω与每用户流数β的操作点，从而显著降低子包化级别并实现对空间多路复用的显式控制。

**🔧 技术方法**

使用了虚拟MISO网络建模、分组缓存放置、零强制(ZF)波束成形、线性可解码分析、子包化级别推导以及基于最大-最小速率的波束优化。

**📊 数据集**

本文基于理论推导和数值仿真，未使用公开数据集；仿真设置主要包括 K=24、L=13、G=2、γ=0.5 等典型参数。

**📈 对比分析**

与 DoF 最优、Lin‑MIMO 及 MU‑MIMO 基线进行比较；在相同 DoF 下，所提方案的子包化级别降低数个数量级；在实际 SNR 范围内，较低 DoF 的操作点可实现更高的对称速率，整体表现优于传统 MU‑MIMO，并在大规模系统中保持良好的 DoF 可扩展性。

**⚠️ 局限性**

局限性包括：仍受整数约束和 Ω、β 取值范围限制；最大可达 DoF 与已知 DoF‑优化方案存在差距；在高 SNR 时，若不充分利用更大 DoF，性能会受到影响；此外，对极端大规模系统的子包化级别提升仍有待进一步研究。

---

## 36. Trust Region On-Policy Distillation

**arXiv ID:** 2606.01249 | [PDF](https://arxiv.org/pdf/2606.01249v1)

**作者:** Xingrun Xing `[一作]` (Samsung Research), Yehui Tang `[通讯]` (Samsung Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Trust Region On-Policy Distillation（TrOPD），用于提升大语言模型在推理任务中的后训练蒸馏效果。

**💡 创新点**

创新点在于引入信任区间策略和异常估计，结合前向KL外部引导，显著降低因分布差异导致的梯度不稳定性。

**🔧 技术方法**

主要技术包括基于K1逆KL的on‑policy distillation、top‑k前向KL估计、蒙特卡罗采样与梯度裁剪/掩蔽，以及off‑policy前缀引导。

**📊 数据集**

实验使用 OpenThoughts3（数学、代码、科学）作为训练样本，并在 AIME 2024/25、AMC 2023、LiveCodeBench、GPQA Diamond、MMLU-Redux 等评测集评估。

**📈 对比分析**

在单域和多域蒸馏中，TrOPD均比现有 OPD、EOPD、REOPOLD 等方法提升约 3–6 分（在数学、代码、指令等基准上），并在多域平均分上获得最高分。

**⚠️ 局限性**

主要局限是未对小推理模型进行实际部署与中期训练评估，且仅验证了在后训练阶段的提升，未来需结合预训练/中期训练进一步验证。

---

## 37. Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning

**arXiv ID:** 2606.01379 | [PDF](https://arxiv.org/pdf/2606.01379v1)

**作者:** Anushka Tiwari `[一作]` (University at Buffalo), Kaiyi Ji `[通讯]` (University at Buffalo)

**通讯引用:** 640 | [OpenAlex ID](https://openalex.org/A5071973105)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无回放的选择性向后细化框架（SABER），用于提示式持续学习，能够在后续任务学习中安全地改进先前任务的提示参数，从而实现正向向后知识迁移。

**💡 创新点**

创新点在于：①利用梯度子空间投影与Wasserstein距离两种任务相关性度量自动识别哪些先前任务能受益于向后更新；②对向后更新做几何约束，将梯度投射到先前任务保留子空间的正交补上，避免负迁移；③累计保护子空间保证多次向后细化互不干扰，并提供理论上保证非干扰且损失非增的证明。

**🔧 技术方法**

技术手段包括：提示调优（prompt tuning）、梯度子空间构造（SVD）、梯度投影与梯度兼容性判定、Wasserstein距离计算、正交投影更新、累计保护子空间正交化。

**📊 数据集**

在两大任务增量基准上评估：SuperNI 和 Long Sequence；并在多种预训练模型上扩展：T5‑Large、LLaMA‑2‑7B、Qwen 等；数据集覆盖多任务顺序和模型规模。

**📈 对比分析**

与七种主流持续学习基线（Replay、L2P、LFPT5、ProgPrompt、CODA‑Prompt、SHLPT、SAPT）对比，SABER 在所有设置下都实现了正向向后迁移（BWT>0），同时保持或提升平均性能（AP）。在大规模模型上，SABER 仍是唯一能持续产生正向向后迁移且总体表现最优的方法；实验表明仅相对基线多 20–30% 的训练时间开销，内存消耗比梯度子空间版本略高。

**⚠️ 局限性**

局限性：①需要手动设定投影阈值和Wasserstein阈值，虽然作者声称对实验稳健但在更极端场景可能需要调参；②梯度子空间版本的存储与计算开销随任务数增大而增长；③目前仅针对提示式持续学习，对其他 PEFT 方式（如 Adapter、LoRA）尚未验证；④未探讨更长任务序列或混合任务相似度下的鲁棒性。

---

## 38. Exploiting In-Sensor Computing for Energy-Efficient Earth Observation

**arXiv ID:** 2606.01271 | [PDF](https://arxiv.org/pdf/2606.01271v1)

**作者:** Luigi Capogrosso `[一作]` (Interdisciplinary Transformation University of Austria), Michele Magno `[通讯]` (ETH Zurich)

**通讯引用:** 7987 | [OpenAlex ID](https://openalex.org/A5066423975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套在 Sony IMX500 图像传感器内部直接执行深度学习推理的 TinyML 管线，实现从传感器到地面站的端到端数据压缩与分类。

**💡 创新点**

创新点在于将推理迁移到传感器本体（in‑sensor computing），彻底消除 OBC 的数据传输开销和能耗，首次在卫星 EO 任务中实现实时传感器级深度学习。

**🔧 技术方法**

采用 INT8 量化的后训练量化（PTQ）、Sony Model Compression Toolkit（MCT）进行图形裁剪与优化，使用轻量卷积网络（SqueezeNet、ShuffleNetV2、MCUNetV1），并通过 MIPI CSI‑2 与 Raspberry Pi5 进行验证。

**📊 数据集**

使用 EuroSAT Sentinel‑2 RGB 数据集（64×64像素，10 类共 27,000 张图像）。

**📈 对比分析**

在 IMX500 上对三模型进行准确率、内存占用、吞吐量（FPS）和能耗（mJ/推理）评估；MCUNetV1 在保持 97.83% 准确率的同时，平均 96.68% 的整体精度、17.40 FPS、27.43 ms 延迟、14.19 mJ/推理、42.26 GMAC/J 能效。

**⚠️ 局限性**

受限于 IMX500 的 8 MB 内存、只能支持轻量卷积结构、未对辐射容忍度和在线学习做实测，仅在 EuroSAT 上验证，迁移到其他 EO 数据集与硬件的通用性待进一步研究。

---

## 39. BenchEvolver: Frontier Task Synthesis via Solution-Centric Evolution

**arXiv ID:** 2606.01286 | [PDF](https://arxiv.org/pdf/2606.01286v1)

**作者:** Yangzhen Wu `[一作]` (University of California, Berkeley), Dawn Song `[通讯]` (Tsinghua University)

**通讯引用:** 58709 | [OpenAlex ID](https://openalex.org/A5019426968)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于解决方案演化的框架，将已有编程题目演化成更难且可验证的版本，并通过自我生成的任务实现模型自我提升。

**💡 创新点**

创新点在于：①先对参考实现进行结构化变异，再从演化后的实现自动生成问题描述与测试；②通过可执行验证和经验性难度评估确保任务既合法又对模型有挑战；③使用记忆引导的进化策略提升搜索效率与多样性；④构建了可持续更新的 LiveCodeBench‑Plus benchmark。

**🔧 技术方法**

使用大语言模型（如 GPT‑5.4、Gemini‑3.1‑Pro）做变异与生成；基于自洽验证（如 brute‑force triangulation、statement‑faithfulness）保证任务正确性；采用强化学习（GRPO）在自生成任务上训练模型；利用内存模块记录变异历史以避免重复与提升多样性。

**📊 数据集**

主要数据集包括 LiveCodeBench（v6）、SciCode、以及从两者演化出的 LiveCodeBench‑Plus（91 题）。

**📈 对比分析**

通过 Pass@1 对比：演化前后目标模型的通过率平均下降 25–40%；在 LiveCodeBench‑Plus 上，前沿模型 Pass@1 从 62.6% 降至 27.5%；在 RL 训练中，seed+evolved 组合在 LCB‑v6 Hard 与 LCB‑Pro Easy 上相较于仅使用 seed 提升 3–4 点，且在独立演化的 LCB‑Evolved Medium 上提升超过 7 点。

**⚠️ 局限性**

局限包括：①演化过程依赖于目标模型，可能导致自相似的弱点被过度强调；②当前仅在两类可执行编程任务上验证，通用性待进一步探讨；③多轮自我提升循环尚未实验，稳定性与多样性控制仍是挑战；④高质量验证和人工审核成本仍不可忽视。

---

## 40. Brain-Atlas-Guided Generative Counterfactual Attention for Explainable Cognitive Decline Diagnosis Using Multimodal Connectomes

**arXiv ID:** 2606.01237 | [PDF](https://arxiv.org/pdf/2606.01237v1)

**作者:** Xiongri Shen `[一作]`, Zhiguo Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了在氧化铜薄片与聚苯乙烯微球界面上形成的一种新型极化子（Y₁），通过实验观测证明其存在。

**💡 创新点**

创新点在于首次发现并表征了该界面极化子，展示了其独特的耦合强度和光学特性。

**🔧 技术方法**

采用光学实验技术（如光谱测量、等离子体共振等）以及表面结构表征手段。

**📊 数据集**

使用的实验数据来自制备的氧化铜薄片和聚苯乙烯微球样品，具体实验记录与光谱数据。

**📈 对比分析**

与传统极化子进行对比，结果显示该极化子在能量位置、寿命或耦合强度方面表现更优，实验数据支持其更高的光学性能。

**⚠️ 局限性**

主要局限包括实验条件受限于实验室环境、缺乏对不同材料体系的普适性验证，以及尚未提供理论模型解释该极化子的形成机理。

---

## 41. Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention

**arXiv ID:** 2606.01243 | [PDF](https://arxiv.org/pdf/2606.01243v1)

**作者:** Shuochen Chang `[一作]` (Shanghai Jiao Tong University), Li Niu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 32163 | [OpenAlex ID](https://openalex.org/A5111709519)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对连续隐藏状态中的推理向量进行结构、因果和几何分析，并基于此设计了无训练、解码时干预，提升LLM推理性能。

**💡 创新点**

将连续推理向量的可解释性与可控性相结合，提出训练自由的几何、语义与因果干预方法。

**🔧 技术方法**

使用结构对齐、CKA、线性映射、因果编辑、权重绑定、能量函数等技术。

**📊 数据集**

在GSM8K、GSM‑Hard、SVAMP、StrategyQA等多域数据集上进行评估。

**📈 对比分析**

在多模型、多规模、多任务上对比基线，干预后数学推理准确率提升约1–2个百分点，Commonsense也有正向提升。

**⚠️ 局限性**

干预会增加推理时延，需要额外梯度或投影计算，未来需将这些可解释性先验嵌入训练目标以降低开销。

---

## 42. MViewRouter: Internalizing Geometric Equivariance via Multi-view Alternating Attention for Combinatorial Routing

**arXiv ID:** 2606.01084 | [PDF](https://arxiv.org/pdf/2606.01084v1)

**作者:** Shiyan Liu `[一作]` (Huazhong University of Science and Technology), Yan Jin `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 21543 | [OpenAlex ID](https://openalex.org/A5041891687)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对组合路由问题（TSP 和 CVRP），提出了 MViewRouter 框架，内部化几何等变性，使模型能够在不依赖数据增强的情况下实现鲁棒的解路策略。

**💡 创新点**

核心创新包括：
- Multi‑View Alternating Attention (MAA)：通过在八个 D4 对称视角之间交替进行局部自注意力与全局视角对齐，构建对称不变的隐藏表示。
- Collective Policy Gradient Aggregation (CPGA)：在训练时同步处理所有对称视角，聚合一致的梯度并使用视角特定基线，提升梯度稳定性并逼近 D4‑不变目标。

**🔧 技术方法**

使用技术：
- Transformer‑style 编码器，包含多头自注意力与视角间 fiber‑attention。
- 对称视角生成（D4 旋转/反射），共享线性投影。
- 基于策略梯度的深度强化学习，配合视角特定基线。
- 统一的多视角批量训练，保持与单视角模型相同的数据量。

**📊 数据集**

实验数据集：
- 生成的 TSP/CVRP 实例：TSP‑50、TSP‑100、TSP‑200、TSP‑500、CVRP‑50、CVRP‑100。
- 真实世界 TSPLIB（29 个实例，50–200 节点）。

**📈 对比分析**

与多种基线比较：
- 精确求解器（Concorde、Gurobi）。
- 经典启发式（LKH3、OR‑Tools）。
- 现有最先进神经解法（POMO、Sym‑NCO、Pointerformer、DIFUSCO 等）。
性能表现：
- 在 TSP‑100 上，原视角仅 0.25% 的最优性差距，×8 视角仅 0.07%，接近精确解。 
- 在 CVRP‑100 上，0.19% 的差距，优于 POMO+PO。 
- 在 TSPLIB 上平均降低 25.5% 的最优性差距，推断时间比 POMO 低 5–7 倍。 
- 在对称性鲁棒性方面，解长方差减少 53.6%。

**⚠️ 局限性**

局限性：
- 对 1k–10k 节点规模的实例尚未可扩展，属于所有自回归构造式解法的通用挑战。 
- 目前仅针对欧几里得坐标的 TSP/CVRP，未覆盖时窗、容量变化等更复杂的 VRP 变体。 
- 需要进一步结合微调或主动搜索提升大规模实例性能。

---

## 43. MURMUR: An Efficient Inference System for Long-Form ASR

**arXiv ID:** 2606.01483 | [PDF](https://arxiv.org/pdf/2606.01483v1)

**作者:** Wei-Tzu Lee `[一作]` (University of Washington), Baris Kasikci `[通讯]` (University of Washington)

**通讯引用:** 2126 | [OpenAlex ID](https://openalex.org/A5050964144)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种两级推理系统 MURMUR，利用可调 300 秒的 chunk 大小和滑动窗口 KV 缓存淘汰，在保持长上下文 ASR 准确性的同时显著降低推理延迟。

**💡 创新点**

创新点包括将 chunk 大小视为超参数并证明 300 秒为最优点、对输出与语音 token 同时采用滑动窗口 KV 缓存淘汰以利用注意力稀疏性，并将两种层级优化结合实现 4.2× 的延迟提升。

**🔧 技术方法**

技术手段包括基于 VibeVoice-ASR 的长上下文模型、VAD 引导的并行 chunk 推理、StreamingLLM 风格的 KV 缓存滑动窗口淘汰以及跨 chunk 的时间对齐与说话人映射。

**📊 数据集**

使用 AMI-IHM、AMI-SDM、Tedlium3 与 Earnings21 四个数据集进行评估，测量 WER、DER、cpWER 与 tcpWER。

**📈 对比分析**

与 WhisperX（30 秒 chunk）和 VibeVoice-ASR 单 pass 基线对比，MURMUR 在 AMI-IHM 上实现 25.3% 的 tcpWER，延迟仅 88 秒，较单 pass 缩短 4.2×，且在 WER/DER 上保持或略优。

**⚠️ 局限性**

局限性在于仅适用于能够原生输出时间戳和说话人标签的长上下文 ASR 模型；实验仅在英文数据集上完成，跨语言或其他模型的泛化性待验证。

---

## 44. Training-Free Imitation Learning with Closed-Form Diffusion Policies

**arXiv ID:** 2606.01238 | [PDF](https://arxiv.org/pdf/2606.01238v1)

**作者:** Raghav Mishra `[一作]` (University of Sydney), Ian R. Manchester `[通讯]` (University of Sydney)

**通讯引用:** 48075 | [OpenAlex ID](https://openalex.org/A5028491443)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Closed-Form Diffusion Policies（CFDP），一种训练-free的扩散式模仿学习策略，可在移动CPU上实时推理并直接从演示数据集中执行。

**💡 创新点**

创新点在于利用闭式评分函数与非参数核估计实现无训练的条件扩散策略，并将其作为可组合原语用于推理时编辑预训练的神经扩散政策。

**🔧 技术方法**

技术手段包括闭式扩散模型、线性噪声过程、Mahalanobis核+ kNN 局部估计、Euler PF-ODE 推理，以及Classifier-Free Guidance（CFG）和增量演示增强。

**📊 数据集**

主要使用了 2D PushT（模拟与真实硬件）数据集，以及在 Robomimic 机器人仿真基准上的演示数据进行评估。

**📈 对比分析**

与 1-NN、NDP-C/T、LSTM-GMM、IBC、BET 等基线比较；CFDP 在多数任务上与神经扩散政策相近，推理速度比 NDP 快 7 倍（CPU）且无需长时间训练；硬件实验中 72% 成功率。

**⚠️ 局限性**

局限性包括：受限于核方法的维度灾难，需手工调参；在高维图像任务难以直接应用；在稀疏观测空间易出现循环；对新分布的泛化受限，且需要完整存储和查询训练数据集。

---

## 45. A Sonar-Visual Dataset for Cross-Modal Underwater Robot Perception

**arXiv ID:** 2606.01398 | [PDF](https://arxiv.org/pdf/2606.01398v1)

**作者:** Weitung Chen `[一作]` (Massachusetts Institute of Technology), Peter Halland Haro `[通讯]` (SINTEF)

**通讯引用:** 152 | [OpenAlex ID](https://openalex.org/A5073413750)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个同步声纳-视觉的海洋机器人数据集SOVIS，并提供了跨模态注释工具和鱼类检测基准。

**💡 创新点**

创新点在于首次提供大规模、实时同步的多束声纳与单目相机对齐数据，以及能够在两种坐标系间自动投影的交互式注释系统，开启了跨模态预测研究。

**🔧 技术方法**

使用了多束声纳、单目相机、温压传感、时序同步、投影模型、YOLOv8、EfficientNet‑B4+FPN+Attention等技术。

**📊 数据集**

使用的数据集是新收集的SOVIS，共76,600帧对齐图像与声纳，并标注了306条鱼类实例。

**📈 对比分析**

通过与几何基线对比，SOVISFishNet在mAP@0.10上从0.067提升到0.467，达到7倍提升，角度误差约4.8°，范围误差0.102m。

**⚠️ 局限性**

局限在于标注样本有限（仅306个），模型受限于小训练集，且只实现了框级检测，未实现像素级声纳预测。

---

## 46. Advanced Mathematics Learning Behavior Prediction and Academic Early Warning Model Based on Multimodal Data Analysis

**arXiv ID:** 2606.01224 | [PDF](https://arxiv.org/pdf/2606.01224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 47. SkillRevise: Improving LLM-Authored Agent Skills via Trace-Conditioned Skill Revision

**arXiv ID:** 2606.01139 | [PDF](https://arxiv.org/pdf/2606.01139v1)

**作者:** Yuxuan Liu `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10743 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于执行反馈的迭代修订框架 SkillRevise，用于在冷启动情境下改进由 LLM 生成或专家编写的 agent 技能。

**💡 创新点**

创新点在于将任务特定的诊断、可迁移的修复原则库以及执行锚定的编辑机制融合成一个有界的循环，能够在有限的重执行预算内通过实测证据自动生成更可靠的技能。

**🔧 技术方法**

核心技术包括：执行轨迹记录与诊断模块、原则检索与绑定（稀疏+稠密匹配）、基于编辑指令的修订算子以及基于实测效用的技能选择。

**📊 数据集**

使用了三大 verifier 驱动的 benchmark（SkillsBench、SkillLearnBench-Random、SWE-Skills-Bench-Hard）以及一个 ALFWorld 原则吸收实验。

**📈 对比分析**

与无技能执行和一次性 LLM 生成技能相比，SkillRevise 在 GPT‑5.5 上分别把 SkillsBench 的成功率从 36.05% 提升至 61.63%，SkillLearnBench-Random 从 39.53% 提升至 61.63%，SWE‑Skills‑Bench‑Hard 从 28/70 提升至 33/70；在多种执行器上亦保持显著改进，且修订后的技能具有跨模型迁移能力。

**⚠️ 局限性**

局限性包括：需要可见的 verifier 反馈并消耗额外执行成本；对测试用例的依赖可能导致过拟合；在极端冷启动或长时序任务中，最多 3 次修订仍可能无法回退至无技能模式。

---

## 48. From Outliers to Errors: Auditing Pali-to-English LLM Translations with Multi-Reference Adjudication

**arXiv ID:** 2606.01136 | [PDF](https://arxiv.org/pdf/2606.01136v1)

**作者:** Máté Metzger `[一作]` (Independent Researcher), Hansa Dhammahaso `[通讯]` (Nibbana Meditation Centre)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四大旗舰LLM（GPT‑5.5、Claude Sonnet 4.6、Gemini 3.1 Pro、Grok 4.3）在1700条Pali Canon段落的英文翻译进行评估，构建多参考本地参考包络，使用句子嵌入漂移做高风险筛选，然后将高漂移样本交由三模型LLM评审进行错误类别与严重性标注。

**💡 创新点**

创新点在于：①将多参考嵌入漂移视为审核过滤器而非最终错误标签；②通过LLM评审面板实现大规模的错误严重性归类；③提出可复用的审计框架，能在多参考环境下优先定位潜在重大错误，同时保留合法译变。

**🔧 技术方法**

技术手段包括：Qwen3 Embedding 8B进行句子嵌入、BLEURT/COMET/chrF等自动指标、三模型LLM评审（DeepSeek、Kimi、Qwen）面板、基于Wilson区间的误差估计、对比分析及误差分类（omission、doctrinal term、agent/role、hallucination、negation）。

**📊 数据集**

数据集为1700条Pali Canon段落，来源于SuttaCentral（Sujato、Bodhi）、Thanissaro；经过过滤得到1700条测试集，并额外抽取300条验证集用于校准评审面板。

**📈 对比分析**

评估方法：先计算每个翻译的normalized drift，阈值1.5筛选1203条高漂移实例；LLM面板对其进行三类标签（valid variation、minor error、major error），得到不同模型的高漂移主错误率（GPT‑5.5≈4%，Claude≈9%，Gemini≈7%，Grok≈27%）。自动指标上Gemini 3.1 Pro表现最好，但与实际主错误率并不完全对应，表明高漂移区块更能捕获严重错误。

**⚠️ 局限性**

局限性：验证集为作者裁定的单人标注，未达独立专家标准；低漂移实例未完全评审，可能低估低漂移错误；评估仅在段落级别，缺乏更广泛上下文；嵌入模型和LLM评审的选择可能引入偏差。

---

## 49. All Models are Wrong, Knowing Where is Useful: On Model Uncertainty in Reinforcement Learning

**arXiv ID:** 2606.01363 | [PDF](https://arxiv.org/pdf/2606.01363v1)

**作者:** Bernd Frauenknecht `[一作]` (RWTH Aachen University), Sebastian Trimpe `[通讯]` (RWTH Aachen University)

**通讯引用:** 2327 | [OpenAlex ID](https://openalex.org/A5023990842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一套利用不确定性量化来降低模型误用的框架，并在此基础上构建了 Probabilistic Ensemble (PE) 模型和 Infoprop 轨迹采样方法，随后将其整合进 Dyna‑style MBRL，实现了在真实机器人硬件上进行高效学习和安全探索。

**💡 创新点**

创新点包括：①将模型不确定性（epistemic uncertainty）作为模型可信区域的阈值，用以限制模型使用范围；②提出 Infoprop 机制，通过最大似然估计消除不确定性并跟踪长时间步的误差积累；③将这些方法与安全过滤器（Dyna‑SAuR）和预测安全过滤器（UPSi）相结合，显著提升数据效率、性能与安全性。

**🔧 技术方法**

主要技术：概率集成网络 (PE)、Trajectory Sampling (TS) 与 Infoprop 轨迹采样、贝叶斯神经网络、最大似然估计、协方差交叉融合、控制障碍函数 (CBF)、MPC 与可达集计算。

**📊 数据集**

数据集：真实 Mini Wheelbot 机器人的传感器数据（姿态、速度、控制指令），以及通过 PE 模型在 HPC 上生成的仿真数据；对比实验使用了标准 MPC 与基于约束 MDP 的安全方法。

**📈 对比分析**

对比方法：与非线性 MPC 基线、传统 Dyna‑style MBRL、约束 MDP 等。实验结果表明：在 Mini Wheelbot 的竞速任务中，仅 11 分钟真实交互即可击败 MPC 基线；在安全探索任务中，Dyna‑SAuR 将失败率降低了两倍之多，相比传统约束 MDP 方法降低了两位数的失败率。

**⚠️ 局限性**

局限性：①目前仅针对自回归的 proprioceptive 状态空间，未覆盖视觉或高维隐状态模型；②部分安全方法（如 Dyna‑SAuR）缺乏严格的安全保证，属于采样近似；③对极大数据量或极端非线性系统的可扩展性尚待验证。

---

## 50. AMP: A Vendor-Neutral Wire Format for Agent Memory Operations

**arXiv ID:** 2606.01138 | [PDF](https://arxiv.org/pdf/2606.01138v1)

**作者:** Thamilvendhan Munirathinam `[一作]` `[通讯]` (Independent Researcher), Thamilvendhan Munirathinam (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个通用的 JSON‑Schema 2020‑12 代理记忆协议（memorywire），定义了五种记忆操作（写、读、删、去重、过期）和四种记忆类型（语义、情节、程序、情感），并实现了一个支持多后端、可选人类审计的参考实现。

**💡 创新点**

创新点在于将已有的记忆框架（mem0、Letta、Cognee 等）中散乱的 SDK、存储模型与治理流程统一为可验证的、版本化的协议层，而不是提出新的算法；通过 RRF 及 FSM、STM↔LTM 的标准化组合，实现跨后端互操作与治理。

**🔧 技术方法**

使用的技术包括 JSON‑Schema、Python 3.11+ 的异步接口、Reciprocal Rank Fusion（RRF）、有限状态机（FSM）表示程序记忆、SQLite、PostgreSQL pgvector、REST/JSON‑RPC 适配器以及 HTMX‑Starlette 的治理 UI。

**📊 数据集**

实验使用的主要数据集为 100 条手工编写的多类型事实（分布为语义、情节、程序、情感）以及 50 条合成记忆做对抗注入实验，另外利用公开的 LongMemEval 与 LoCoMo（已完成预实验）。

**📈 对比分析**

通过与五种后端的交叉适配验证（16 场景共 80 个单元测试，68 PASS/12 SKIP/0 FAIL）、微基准（Recall@5=1.000、写入延迟 37.8 ms、召回延迟 40.6 ms）以及对抗融合实验（RRF 在 1‑of‑N 注入下保持 Recall@5=1.000，MAX 下降至 0.500）展示了协议在性能与安全性上的可行性。

**⚠️ 局限性**

局限性包括：仅在单机环境下测得延迟，未评估分布式部署；对抗实验假设攻击者完全知晓路由器参数；治理机制目前仅基于 SQLite 的附录日志，缺乏完整的硬件隔离与多租户隔离；协议仍在 v0 阶段，版本兼容与安全修补需进一步完善。

---

## 51. GuidaPA: Privacy-Preserving Chatbot for Public Administration via Federated Learning

**arXiv ID:** 2606.01386 | [PDF](https://arxiv.org/pdf/2606.01386v1)

**作者:** Daniel M. Jimenez-Gutierrez `[一作]` (Sapienza University of Rome), Andrea Vitaletti `[通讯]` (Sapienza University of Rome)

**通讯引用:** 1903 | [OpenAlex ID](https://openalex.org/A5070969466)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在意大利公共行政环境中，利用联邦学习在本地文档上微调大语言模型，构建了名为GuidaPA的隐私保护聊天机器人。

**💡 创新点**

其创新点在于将联邦学习与角色访问控制、客户端安全预处理及非IID效应监测集成到面向机构的对话系统框架，并在保持数据本地化的前提下实现接近中心化的答复质量。

**🔧 技术方法**

使用了参数高效微调技术QLoRA（4-bit）+ LoRA、Flower框架进行联邦平均，并结合角色访问控制和安全预处理。

**📊 数据集**

数据集来源于意大利两个公共行政平台SIGESON（约8页手册+FAQ）和SIDFORS（约31页手册+FAQ），共约77条问答对。

**📈 对比分析**

通过与私有中心化微调基线比较，联邦模型在ROUGE-1/2/L、BLEU-4和METEOR指标上仅落后不到5%，实现了与中心化模型相近的性能。

**⚠️ 局限性**

局限性包括仅使用两家平台的公共手册作为数据，数据量小且单一；评估仅基于自动NLG指标，缺乏人工或任务成功率评估；未尝试针对更强非IID的联邦优化算法。

---

## 52. Low-Resource Safety Failures Are Action Failures, Not Representation Failures

**arXiv ID:** 2606.01196 | [PDF](https://arxiv.org/pdf/2606.01196v1)

**作者:** Rashad Aziz `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1237 | [OpenAlex ID](https://openalex.org/A5065822589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究发现低资源语言中模型拒绝有害提示的失败不是缺失有害性表征，而是阈值校准不当，提出利用高资源语言学习到的有害性方向，通过少量低资源示例重新校准门控阈值，修复拒绝行为

**💡 创新点**

创新点在于将跨语言安全缺陷归因于决策阈值失调，提出一种训练自由的少样本潜在门控（latent gate）方法，仅需1~4个目标语言示例即可重新校准阈值并提升低资源语言的拒绝率，且保持对无害提示的准确性

**🔧 技术方法**

技术包括：利用残差流中的一维有害性方向（contrastive‑mean）提取，低秩逻辑回归门控（rank‑10 subspace），阈值自适应（few‑shot calibration），以及条件路由将门控结果与拒绝方向插值/消除结合

**📊 数据集**

使用扩展后的 PolyRefuse 数据集，共23种语言（10高资源、7中资源、6低资源），以及三大模型（Qwen2.5‑7B、Gemma‑2‑9B、Llama‑3.1‑8B）进行评估

**📈 对比分析**

与现有跨语言安全方法 CAST 与 AdaSteer 进行对比；在 18 种模型–语言组合中，门控方法在保持 MMLU 效果的前提下，平均 Δ（有害拒绝率减去无害拒绝率）从 33.6% 提升至 54.5%，同时无害提示的误拒率保持在 12–15% 左右，明显优于对比方法

**⚠️ 局限性**

局限性包括：仅在 7B–9B 开源指令调优模型上验证；依赖 GPT‑4o‑mini 做自动判定，可能忽略多语言细微差异；门控需要对内部激活的访问，无法直接用于闭源模型；未覆盖模型结构多样性（如 MoE、稀疏模型）或极低资源语言的进一步泛化

---

## 53. SABER: Benchmarking Operational Safety of LLM Coding Agents in Stateful Project Workspaces

**arXiv ID:** 2606.01317 | [PDF](https://arxiv.org/pdf/2606.01317v1)

**作者:** Qi Hu `[一作]` (University of Hong Kong), Zhuoran Ji `[通讯]` (Shandong University)

**通讯引用:** 212 | [OpenAlex ID](https://openalex.org/A5030309370)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个在 Docker 沙盒中执行的编程代理安全评估基准，评估大型语言模型（LLM）在多步项目环境中的操作安全性。

**💡 创新点**

创新点在于聚焦环境感知的操作安全，提出三种真实场景（嵌入式注入、风险自选、上下文警告），并通过完整的执行轨迹与最终工作区状态来判定安全，而非仅依赖拒绝或提示级别的传统方法。

**🔧 技术方法**

使用统一的 ReAct 交互框架、Docker 沙盒、工具调用接口；结合规则引擎和 LLM 判定的安全标签体系，构建层级化结果分类。

**📊 数据集**

构建了 716 个可执行任务，分别分为 289 个嵌入式注入、186 个风险自选、241 个上下文警告任务，任务来源于公开 CVE、注入基准与实践工作流。

**📈 对比分析**

在 Benchmark 上评估了 13 个模型（如 GPT‑5.4、Claude Opus 4.6、DeepSeek‑R1 等），采用 HSR（有害安全违规率）、SRR（安全拒绝率）、IR（无能率）等指标；结果显示即使最好的模型 HSR 也超过 54%，表明当前对齐仍不足。

**⚠️ 局限性**

局限性包括：仅使用统一 ReAct harness 评估，未考虑各厂商 agent harness、确认机制、回滚等；未提供真实网络访问或云 IAM 环境；Docker sandbox 不能完全模拟生产多用户权限与长期服务等实际情况。

---

## 54. SEArch: Optimistic Policy Selection Between Scene Noise and Drift for UAV Radar Search

**arXiv ID:** 2606.01325 | [PDF](https://arxiv.org/pdf/2606.01325v1)

**作者:** Noor Khial `[一作]` (Qatar University), Amr Mohamed `[通讯]` (Qatar University)

**通讯引用:** 9418 | [OpenAlex ID](https://openalex.org/A5021329808)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了基于SEA框架的SEArch和W-SEArch两种轻量级在线策略选择器，用于在UAV雷达搜索中自适应应对场景噪声与漂移的变化。

**💡 创新点**

创新点在于：① 将场景内噪声与场景间漂移分别量化并通过SEA框架给出可调误差上界；② 在OFTRL基础上加入自适应学习率和窗口重启机制，使算法在面对快速场景切换时仍保持低误差；③ 提供理论上O(σ̅_T√T+√J)和O(σ̅_I√w)的误差界。

**🔧 技术方法**

技术手段包括：Optimistic Follow the Regularized Leader (OFTRL) 与自适应学习率；SEA（Stochastically Extended Adversary）框架下的在线凸优化；窗口化（Windowed OFTRL）与重启策略；全信息反馈模型与简单的凸投影。

**📊 数据集**

实验使用仿真生成的雷达测量数据，包含4个预训练检测器，T=150，设置不同噪声水平σ∈{0.05,0.1,0.2,0.4}和不同场景切换次数J，验证算法在各种非平稳情形下的性能。

**📈 对比分析**

与UCB、Exp3、OMD等经典在线学习算法对比，实验表明SEArch和W-SEArch在不同非平稳场景中平均累计误差显著低于基线；W-SEArch在快速切换场景下可减少约30% regret，整体实现30% regret下降。

**⚠️ 局限性**

局限性包括：窗口长度w需要预先设定，无法自适应；在高噪声环境下窗口重启优势降低；假设全信息反馈且检测器库已离线预训练，未考虑轨迹规划与策略选择的联合优化。

---

## 55. Digging Up Citations: FOSSIL, a Dataset and Workflow for Reference Extraction in Law and the Humanities

**arXiv ID:** 2606.01109 | [PDF](https://arxiv.org/pdf/2606.01109v1)

**作者:** Luca Foppiano `[一作]` (ScienciaLAB), Christian Boulanger `[通讯]` (Max Planck Institute for Legal History and Legal Theory)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5026897129)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了开放许可证的多语言法律与人文学科论文脚注引用标注数据集，开发了支持协同双视图验证的PDF-TEI Editor工具，并提出了针对脚注引用的Grobid专用“footnotes-ref”流程。

**💡 创新点**

首次公开脚注引用标注数据集；PDF-TEI Editor实现双视图同步、TEI Schema校验和插件化扩展；Grobid专用流程通过替换分割模型显著提升脚注引用识别，覆盖多语种、多学科。

**🔧 技术方法**

使用Grobid机器学习分层提取模型、PDF-TEI Editor的web双视图与TEI验证；Krippendorff α评估、5折交叉验证；LLaMore评估框架结合Levenshtein模糊匹配和Hungarian匹配。

**📊 数据集**

96篇公开许可的学术文章（38本期刊），涵盖法律、历史、社会科学等，包含约2,400脚注块、7,600引用；与CEX、EXparser Gold等已有数据集做对比。

**📈 对比分析**

端到端对比四篇文档，使用关键字段（期刊标题、作者姓氏、出版日期）F1进行评估。默认Grobid F1仅0.35–0.42，专用流程F1提升至0.66–0.80，微平均F1从0.36增至0.72，召回率大幅提升。

**⚠️ 局限性**

仍存在跨页引用、缩写日期等难题导致召回不足；分割模型在同页区块难以区分；数据集规模有限，需要更多语种和期刊；评估仅覆盖四篇文档，需进一步验证。

---

## 56. A Lightweight Slot-Attention Framework for Multi-Instrument Multi-Pitch Estimation

**arXiv ID:** 2606.01460 | [PDF](https://arxiv.org/pdf/2606.01460v1)

**作者:** Michael Taenzer `[一作]` `[通讯]` (Ilmenau University Of Technology), Michael Taenzer (Ilmenau University Of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级槽注意力框架，用于多乐器多音高估计，直接从混音 CQT 预测无序的源级音高映射。

**💡 创新点**

创新点在于使用匈牙利匹配实现排列不变的槽输出；引入自监督色彩编码器提供槽级音色监督；以及多音高分支正则化槽级音高密度。

**🔧 技术方法**

技术包括槽注意力机制、匈牙利匹配、对比自监督色彩编码器、HCQT 前端、卷积骨干、可选的 Polyphony 分支和 FiLM 条件化。

**📊 数据集**

使用 URMP（传统乐器录音）、mshoxxDB（电子音乐）和 MusicNet（基准 MPE）等公开数据集进行训练与评估。

**📈 对比分析**

与标准 MPE 基线相比，匈牙利匹配下的槽方法在 URMP 上 AP 由 24% 提升至 61%，在 mshoxxDB 上提升显著；加入色彩和多音高监督能进一步提高源级 AP/F1，但效果受数据集异质性影响。

**⚠️ 局限性**

局限在于源分配仍易受多源混叠影响，色彩和多音高监督未能完全消除槽间混淆；方法对大规模数据的可扩展性和对复杂电子音色的鲁棒性仍待验证。

---

## 57. Understanding LLM Behavior in Multi-Target Cross-Lingual Summarization

**arXiv ID:** 2606.01252 | [PDF](https://arxiv.org/pdf/2606.01252v1)

**作者:** Sangwon Ryu `[一作]` (POSTECH), Hinrich Schuetze `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了新的多目标跨语言摘要基准 MEA，系统评估了基于大语言模型（LLM）的端到端与管道式跨语言摘要方法，并通过层级分析框架探究 LLM 内部的翻译与摘要行为，随后基于英语摘要隐藏表示提出了推理时激活引导方法以提升多目标跨语言摘要质量。

**💡 创新点**

① 新增覆盖 24 种目标语言的 MEA 基准；② 设计了层级分析框架以定位翻译、摘要及错误出现的内部层；③ 在此基础上提出推理时激活引导（Activation Steering）方法，将英语摘要的隐藏表示用于引导跨语言摘要，首次实现无训练即可提升 LLM 在多语言场景下的表现。

**🔧 技术方法**

层级分析利用 Logit Lens 将每层隐藏状态投影至词表空间；激活引导通过计算语言对比方向并在指定层对隐藏状态进行插值调节；评估使用 G‑Eval、BERTScore、ROUGE 等指标，实验采用 Qwen3.5、Tiny‑Aya‑Global、gpt‑oss‑20b 等多模型。

**📊 数据集**

使用 CNN/DailyMail 版 Element‑Aware 评测集，原始 200 条英语摘要示例通过 Google Translate 和 GPT‑4o‑mini 翻译成 24 种语言，共 4.8K 条跨语言摘要样本，确保翻译质量与信息保真。

**📈 对比分析**

与传统端到端和管道（Summ‑>Trans / Trans‑>Summ）方法对比，实验表明：在大模型（如 Qwen3.5‑9B）下端到端方案整体优于管道；但在小模型下管道表现更好；激活引导在高资源语言上可实现与管道相当甚至更优的性能，低资源语言效果有限。

**⚠️ 局限性**

① 低资源语言评估受限于缺乏成熟 tokenizer 与评测工具，G‑Eval 可能引入偏差；② 激活引导对低资源语言效果不佳，说明英语与低资源语言的表示差距仍未完全消除；③ 基准主要基于 CNN/DailyMail，可能不具备多领域泛化能力。

---

## 58. Strong Stochastic Flow Maps

**arXiv ID:** 2606.01086 | [PDF](https://arxiv.org/pdf/2606.01086v1)

**作者:** Sam McCallum `[一作]`, James Foster `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了强随机流图（SSFMs）框架，直接学习可加噪声SDE的Itô映射，实现路径级（强）采样，显著减少网络评估次数。

**💡 创新点**

创新点包括：①利用多项式（Legendre）逼近Brownian轨迹并证明其α-Holder收敛；②在自蒸馏目标中同时约束漂移、扩散和半群性质，确保得到真正的Itô映射；③实现无模拟（simulation-free）训练，可在扩散模型中直接获得强解。

**🔧 技术方法**

核心技术包括神经ODE/SDE、Legendre多项式 Brownian 逼近、Itô映射的自蒸馏训练目标、EDM2网络架构、无模拟的逆SDE驱动梯度匹配。

**📊 数据集**

评估数据集：CIFAR-10、CelebA‑64（图像生成）；Alanine‑Dipeptide、Chignolin（分子构象采样）。

**📈 对比分析**

与确定性流图、弱随机流图（Diamond/GLASS/GLASS 等）以及传统扩散模型在 FID、PMF误差、JS散度、Wasserstein 指标上进行对比；SSFMs 在 1–10 步（NFE）下已达到或超过传统扩散在 1000 步时的性能，且在分子任务中将 NFE 降至 1–2 次评估。

**⚠️ 局限性**

局限性：目前仅适用于加噪声SDE；对非加噪声或状态相关扩散系数的推广需要进一步研究；高阶精度可能需要更多多项式系数，导致训练与推理成本提升；在极高维或强非线性系统中收敛速度和稳定性仍待验证。

---

## 59. Exploiting Multiple Abstract Call Patterns for Optimizing Run-Time Checks

**arXiv ID:** 2606.01076 | [PDF](https://arxiv.org/pdf/2606.01076v1)

**作者:** Daniela Ferreiro `[一作]`, Manuel Hermenegildo `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种将运行时断言检查语义直接嵌入到多变体顶层抽象解释框架中的方法，从而在程序静态分析时更精确地推断哪些断言可以在编译期完成，哪些需要在运行时检查，并通过生成多版本程序来进一步消除不必要的运行时检查；

**💡 创新点**

创新点在于：①将运行时检查语义与多变体顶层抽象解释器无缝结合，避免了传统的程序转换步骤；②利用不同调用模式（多变体信息）自动生成专用版本，显著提升断言验证率；③对断言属性进行细粒度简化，评估每个属性的验证情况；

**🔧 技术方法**

核心技术包括：多变体顶层抽象解释（top‑down fixpoint）、抽象运行时检查语义、运行时断言属性的抽象化与简化、版本化程序变换（vers）以及在Ciao系统中的实现；

**📊 数据集**

实验使用了两类数据集：1) 经典Prolog基准和Ciao系统库中的若干模块；2) 文档生成器（documenter）这一较大规模应用程序；

**📈 对比分析**

比较方法：将实现的分析器与三种策略进行对比——（a）默认语义的基准运行时检查；（b）使用抽象解释的运行时检查语义；（c）在 (b) 的基础上进行版本化再分析。评估指标包括：已简化断言属性的比例、运行时检查时的执行时间以及相对基准的加速比。实验结果表明：在大多数模块中，断言属性简化率可达 70% 以上，运行时执行时间可比基准提升 2 倍至 240 倍，尤其在使用多变体信息生成版本后，性能提升显著；

**⚠️ 局限性**

限制与挑战：①抽象解释的宽化操作可能过于保守，导致某些属性无法在编译期验证；②版本化产生的程序膨胀会增加内存占用；③在高度递归或涉及复杂约束的程序中，抽象域的精度仍有限；④实验依赖于Ciao系统的实现，未对其他动态逻辑语言进行验证。

---

## 60. TravelEval: A Comprehensive Benchmarking Framework for Evaluating LLM-Powered Travel Planning Agents

**arXiv ID:** 2606.01046 | [PDF](https://arxiv.org/pdf/2606.01046v1)

**作者:** Weiyi Chen `[一作]` (Zhejiang University), Lei Chen `[通讯]` (Hong Kong University Of Science And Technology)

**通讯引用:** 30394 | [OpenAlex ID](https://openalex.org/A5100333593)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TravelEval，包含六维评估框架、逼真沙箱和基于仿真的全局评估方法；

**💡 创新点**

创新点在于从多维度评估旅行计划、引入真实定价与交通数据、以及通过仿真捕捉连贯行程的时空交互；

**🔧 技术方法**

使用大型语言模型、链式思维、ReAct与Reflexion等推理策略，并结合Amap API、交通/酒店/POI接口进行数据拉取；

**📊 数据集**

构建了1,150个用户查询及10城POI、住宿、餐饮等10万+实体的真实数据集；

**📈 对比分析**

与基准方法及12种主流LLM/代理进行对比，结果表明即使是SOTA模型在预算、时空优化和体验多维度仍低于基线；

**⚠️ 局限性**

局限在于模型难以实现全局约束满足、推理复杂度导致认知过载，以及对多约束平衡能力不足。

---

## 61. Leyline: KV Cache Directives for Agentic Inference

**arXiv ID:** 2606.01065 | [PDF](https://arxiv.org/pdf/2606.01065v1)

**作者:** Bole Ma `[一作]` (Erlangen National High Performance Computing Center), Harald Koestler `[通讯]` (Erlangen National High Performance Computing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个服务端的 KV 缓存编辑原语，使代理式 LLM 能够主动删除或替换缓存内容而不必重新预填所有后续内容。

**💡 创新点**

创新点在于引入声明式 4‑tuple 指令（起止位置、替换序列、模式），将政策层的“要做什么”与内核层的“如何安全切换”解耦，并利用闭式 δ‑旋转在不重算 KV 的情况下保持注意力数学正确。

**🔧 技术方法**

核心技术包括：δ‑旋转算法（对 RoPE 的闭式位置校正）、基于指令的切片（splice）机制、radix 缓存与指令层集成以及可插拔的 Python 政策接口。

**📊 数据集**

实验使用了四大开源 LLM（DeepSeek‑V2‑Lite、JoyAI‑LLM‑Flash、GLM‑4.7‑Flash、Moonlight‑16B‑A3B）以及 debug‑gym mini_nightmare 任务集合。

**📈 对比分析**

通过单步微基准（单提示对比）、多步策略轨迹重放、随机切片压力测试和真实长上下文代理回放等方式验证，splice 机制在 17K‑token 提示下提升缓存命中率约 11.2pp，端到端延迟最高可节省 241 ms；在 8 个 debug‑gym 任务上，10 行截断策略通过指令实现后解锁率提升 14.3pp。

**⚠️ 局限性**

局限包括：δ‑旋转实现仅适用于 64‑维 RoPE 结构，需在 GQA/MHA 上额外扩展；bf16 KV 存储导致的 1–3% 精度噪声，偶尔影响 argmax；对位置不确定的中间模板切片模型敏感；实验范围主要聚焦特定任务与阈值，未覆盖更广泛策略与模型组合；以及跨租户并发发现问题需进一步工程化。

---

## 62. Towards Optimal Robustness in Learning-Augmented Paging

**arXiv ID:** 2606.01342 | [PDF](https://arxiv.org/pdf/2606.01342v1)

**作者:** Peng Chen `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9463 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种学习增强分页算法框架 RPB-OnOPT，利用相对预测预算实现最优鲁棒性与一致性。

**💡 创新点**

创新点在于：①定义相对预测预算 (Relative Prediction Budget) 以细粒度调节预测使用；②在 OnlineMin（OM）基础上构造 RPB-OM，实现 H_k + O(1) 的鲁棒性；③证明 1‑一致性与最佳竞争比。

**🔧 技术方法**

使用在线最优算法的工作函数层结构、随机优先级分配、潜在函数分析以及预测预算动态管理技术。

**📊 数据集**

在 SPEC CPU 2006 基准上使用真实和合成的下一次访问时间预测（POP-U、PLECO 与 log‑normal 噪声），以及 2 MB 缓存、16 路组联的硬件配置。

**📈 对比分析**

与 LRU、Marker、OnlineMin、BlindOracle、F&R、LMarker、PredictiveMarker、BlindOracle&LRU、OnOPT-OM 等算法对比，RPB-OM 在大多数情形下平均成本比 OPT 低约 1.25‑1.21（约 20‑25 % hit‑rate），表现最优；在噪声水平升高时仍保持良好鲁棒性。

**⚠️ 局限性**

限制：需调参 τ（初始预算）以平衡鲁棒性与性能；算法仍以 O(log k) 的更新复杂度实现，且鲁棒性只能相对最优（加上常数项），对极端预测错误时的细粒度调节还有待进一步研究。

---

## 63. Worlds Within Words: Translating Culture in Ancient Chinese Texts with Multi-Agent Coordination

**arXiv ID:** 2606.01276 | [PDF](https://arxiv.org/pdf/2606.01276v1)

**作者:** Xiaoqi He `[一作]` (University of Macau), Derek F. Wong `[通讯]` (University of Macau)

**通讯引用:** 3830 | [OpenAlex ID](https://openalex.org/A5101468579)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多代理的文化感知翻译框架 MACAT，用于解决古汉语文本中的文化载荷词（CLW）的精准解释与翻译。

**💡 创新点**

创新点在于把 CLW 翻译视为选择性解释任务，利用多代理动态检测 CLW、构建简洁的文化知识卡、生成多候选、参考无评估重排序以及跨段一致性修复，并引入多维度评估机制。

**🔧 技术方法**

主要技术包括：轻量级 CLW 检测门控、规则+LLM 生成的文化知识卡、上下文压缩与古汉语桥接、COMETKiwi 参考无评估重排序、边界感知本地修复以及多轮多维度评估代理。

**📊 数据集**

实验数据集包括：传统中医经典（100 文档 1,005 段）、补充评估集（50 文档 333 段）以及《论语》20 章节（477 段）进行中→英和中→葡翻译。

**📈 对比分析**

与 Google、CoD、Tower-plus 等商业与基线模型相比，MACAT 在所有维度（术语精度、可读性、忠实度、文化保存、文化解释）均有显著提升；在 DeepSeek-V3 主干上，MACAT 的 Final 分数最高，且在跨域（《论语》）实验中亦保持领先。

**⚠️ 局限性**

局限性包括：知识接口主要依赖人工规则，扩展性和维护成本高；评估高度依赖 LLM-as-a-Judge，需进一步验证；跨语言证据有限；多路径生成与重排序提升推理成本，影响大规模或低延迟部署。

---

## 64. Early Diagnosis of Wasted Computation in Multi-Agent LLM Systems via Failure-Aware Observability

**arXiv ID:** 2606.01365 | [PDF](https://arxiv.org/pdf/2606.01365v1)

**作者:** Xianyou Li `[一作]` (New York University), Jing Yang `[通讯]` (Washington University in St. Louis)

**通讯引用:** 6957 | [OpenAlex ID](https://openalex.org/A5019029226)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个失败感知可观测框架，用于诊断多智能体LLM系统中因工具调用、执行、调度、证据缺失和预算限制导致的计算浪费；

**💡 创新点**

创新点在于将多种失败模式映射为可在线收集的结构化信号，并通过低成本指标与深层语义指标相结合，形成多层次的诊断层，首次在GAIA验证集中系统性评估了这些指标的相关性；

**🔧 技术方法**

技术包括结构化事件追踪、工具错误率、重复动作循环检测、信息增益评估、证据支持度计算、LLM-judge审计、以及基于token/工具调用的成本近似；

**📊 数据集**

使用的数据集为GAIA 2023验证快照，包含165个任务（级别1–3各自53/86/26个）；

**📈 对比分析**

对比方法采用10次LLM-judge基础评估与30次准确性抽样，发现可用最终答案率为12/30，相关性分析显示循环标签与不可用结果正相关，token使用随级别提升；

**⚠️ 局限性**

局限性包括仅在单一模型/配置下实验、评审仅为抽样、未测量能耗/延迟、语义相似度可能错漏、以及未验证在线干预策略的效果。

---

## 65. Autopilot-Preserving Residual Q-Learning with HJB-Inspired Finite-Action Risk Filtering for Fixed-Wing UAV Command Supervision

**arXiv ID:** 2606.01397 | [PDF](https://arxiv.org/pdf/2606.01397v1)

**作者:** Mehmet Iscan `[一作]`, Batuhan Temiz `[通讯]` (Turkish Aerospace)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一种基于有限残差指令的命令监督器，放在经典固定翼UAV自动驾驶仪之上，使用HJB启发式价值评估、优势排名和有限动作屏障进行指令选择，且始终保留无操作回退；

**💡 创新点**

创新点在于：①将学习限制在指令层（airspeed、altitude、heading）而非舵面；②利用半离散价值迭代与Hamiltonian优势相结合的评分方法；③引入CLF/CBF风格的有限动作屏障，确保安全与性能；④在单一共享运行时中对比基准自动驾驶仪、Q学习残差和HJB残差三种方案；

**🔧 技术方法**

核心技术包括：固定翼12状态非线性动力学模型、基于Beard–McLain的自动驾驶仪、离散动作空间（7个残差指令）、半离散价值迭代（1296格），Hamiltonian优势函数，CLF/CBF式有限动作屏障，能量分配辅助器，在线Q学习与价值更新；

**📊 数据集**

使用的“数据集”为仿真生成的20个不同的飞行场景（包括紧凑轨道、爬升、转弯、风切变等），每个场景单次种子，时间范围45–60 s；没有外部真实飞行数据；

**📈 对比分析**

在共享运行时对比三种控制包：基准自动驾驶仪、Q残差监督器、HJB残差监督器。结果显示HJB残差在全时长20场景中参考路径RMS降幅为86.77%（相对基准）及49.54%（相对Q），但相应的airspeed RMS和控制活跃度略有上升；基准在airspeed RMS与控制活跃度上优于两者；每个方法在不同指标上各有优势。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，未进行硬件或实飞测试；模型为系数式小型UAV模型，缺乏高保真气动；安全保障仅为启发式屏障，无正式稳定/安全证明；能量分配辅助器与屏障未单独消融，难以单独量化其贡献；样本数量有限，未进行多种种子或统计显著性分析。

---

## 66. Fairness in two-player zero-sum games with bandit feedback

**arXiv ID:** 2606.01159 | [PDF](https://arxiv.org/pdf/2606.01159v1)

**作者:** S Akash `[一作]` (LatentForce.ai), Pratik Gajane `[通讯]` (University of Orléans)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了两人零和博弈在 bandit 反馈下的公平约束学习，提出了通过重参数化将公平约束转化为标准零和博弈并给出了探测-承诺算法的理论收敛分析。

**💡 创新点**

①将最低播放约束通过 affine 重参数化转化为对收益矩阵的 rank‑1 修正，从而把公平博弈映射为经典零和博弈；②给出了对一般混合公平均衡的 O(T^{2/3}) regret 上界，并在单一占优行动情形下获得实例相关的 O(log T) 上界；③阐述了公平约束导致的 LP 结构与传统行动消除方法不兼容的根本原因。

**🔧 技术方法**

主要使用了 reparametrization（p=(α/m)1+(1-α)p）、对公平收益矩阵的分析、线性规划的 KKT 与基准稳定性分析、Hoeffding 与 sub‑Gaussian 估计，以及 Explore‑Then‑Commit 算法与理论证明。

**📊 数据集**

论文未使用真实数据集，仅在理论上证明，文中给出示例矩阵做说明。

**📈 对比分析**

通过理论分析与先前基于纯 NE 的 ETC 结果比较，证明在公平约束下即使存在纯 NE，最优学习率仍为 O(T^{2/3})，单一占优行动可恢复 O(log T) 的最佳率；实验上未报告。

**⚠️ 局限性**

①学习率受限于 ETC 结构，无法达到混合均衡下的 √T；②公平约束下的 LP 基准不易自适应，需要更复杂的动态解法；③对两侧公平约束、非均匀噪声等情况尚未覆盖。

---

## 67. Adversarial Configurations for the ReCom Transition Function

**arXiv ID:** 2606.01333 | [PDF](https://arxiv.org/pdf/2606.01333v1)

**作者:** Micah Gold `[一作]` `[通讯]`, Micah Gold

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文证明了在平面正方形网格图上存在特定的三分区，使得 ReCom MCMC 算法在这些状态下需要指数级的步骤才能产生平衡切分，从而展示了该算法在最坏情况下不一定在多项式时间内完成。

**💡 创新点**

创新点在于构造了一个名为“高速公路图”的特殊图结构，并利用有效电阻理论和无环树采样技术证明：若合并的两块区域形成高速公路图，则其统一生成的生成树几乎不可能被切成平衡两块，从而给出了 ReCom 在最坏情况下的指数级慢收敛例子。

**🔧 技术方法**

主要技术包括：有效电阻与随机生成树之间的对应关系、有效电阻采样算法、对 2×(2m+1) 梯形网格的分段循环分析，以及对随机生成树中竖向支路出现频率的概率上界推导。

**📊 数据集**

数据集：论文仅使用理论构造的平面正方形网格图（不同尺寸的网格），并未使用实际选区或人口数据。

**📈 对比分析**

方法比较：论文没有给出实验对比或数值性能指标；重点在于证明存在指数级慢收敛的最坏情况，而非在平均或典型情况下的速度表现。

**⚠️ 局限性**

局限性：只展示了最坏情况的存在，未证明在实际或从稳态分布开始时出现这种“糟糕”配置的概率；另外，实验验证与实际选区应用仍缺失。

---

## 68. On Fréchet Traveling Salesmen Problems

**arXiv ID:** 2606.01147 | [PDF](https://arxiv.org/pdf/2606.01147v1)

**作者:** Omrit Filtser `[一作]` (Open University of Israel), Michal Moiseev `[通讯]` (Open University of Israel)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了在欧氏空间中两条曲线（或两条路径）必须尽可能靠近的 TSP 变体，目标是构造两条点集的划分曲线，使得它们的 (离散或连续) Fréchet 距离最小，并进一步探讨长度最小化、平衡点数等相关优化子问题。

**💡 创新点**

创新点包括：
1) 将 Fréchet 距离引入到多代理路线规划问题，首次提出“-TSP”概念；
2) 对离散 Fréchet 版本给出几乎线性的 O(nlogn) 近似算法，并得到长度约束的常数因子近似（最大长度 2.75·MST，长度和 4.75·MST）；
3) 设计了平衡点数的算法，证明最差平衡差为 4（或 3 当 n 为奇数），并在允许距离放宽 √3 倍时实现完全平衡；
4) 通过细致的星图覆盖与子集和技术，给出 O(n²) 的平衡改进算法；
5) 证明连续 Fréchet 版本的 NP‑难性，扩展了现有点集 Fréchet NP‑难性结果。

**🔧 技术方法**

主要技术手段包括：
- 近似离散 Fréchet 距离的最短路径与星图分解；
- 单源最近邻图、最小生成树以及单位圆图的性质，用于构造最优距离阈值；
- DFS 与分层染色策略，生成满足距离约束的两条曲线；
- 对星图的可变平衡度分析，利用子集和（bounded）和星图合并实现平衡改进；
- 通过 3‑SAT 归约构造变量/子句几何图，证明连续版本 NP‑难。

**📊 数据集**

由于论文为理论性质，没有使用真实数据集；所有实验均基于合成点集（随机点、MST、最近邻图等）以及构造的几何实例。

**📈 对比分析**

对比方法：
- 与传统 TSP（单路径）和多代理无距离约束的路径规划相比，-TSP 在保持距离的前提下保持 O(nlogn) 的预处理时间；
- 对离散 Fréchet 版本，提供 1.0（最优）距离、2.75 的最大长度近似和 4.75 的长度和近似，均优于目前已知的通用 TSP 近似方案；
- 平衡点数算法在最坏情况下仅产生差距 4（或 3），且在允许距离放宽 √3 倍时可实现完全平衡；
- 论文通过理论分析给出了时间复杂度与近似因子，未给出实验量化结果，但证明在常数维空间中可在 O(nlogn) 内完成。

**⚠️ 局限性**

局限性：
- 仅在欧氏空间且维数为常数时可行；对高维空间仍需 O(2^O(d)nlogn) 的预处理；
- 连续 Fréchet 版本仅证明 NP‑难，没有近似或多项式时间解；
- 近似因子（2.75、4.75）尚非最优，后续研究可能改进；
- 平衡算法在最坏情况下只能保证差距 4，且对非常不均匀的点分布可能导致性能下降；
- 本研究未考虑动态更新或噪声点的影响。

---

## 69. FreqLite: A Lightweight Frequency-Decomposed Linear Model with Adaptive Reversible Normalization for Robust Long-Term Time-Series Forecasting

**arXiv ID:** 2606.01339 | [PDF](https://arxiv.org/pdf/2606.01339v1)

**作者:** Mirza Samad Ahmed Baiga `[一作]` (Fandaqah), Syeda Anshrah Gillani `[通讯]` (Hamdard University)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5011971425)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种极轻量、通道独立的频域分解线性预测器（FreqLite）及其自适应可逆实例归一化（A-RevIN），实现长序列预测；

**💡 创新点**

创新点包括：①可学习无损的频域分区滤波，将信号分为多频段并分别用线性头预测；②保留并建模高频成分；③引入门控的自适应归一化，在非平稳情况下动态调整水平与尺度；

**🔧 技术方法**

使用的技术主要有：FFT/IRFFT频域分解、可学习的软掩模、通道独立的线性头、A-RevIN自适应归一化、门控机制以及标准的Adam优化、MSE损失；

**📊 数据集**

实验数据集包括标准长时序基准ETT系列（ETTh1/2、ETTm1/2）和Weather，以及非平稳数据集Exchange-rate和ILI，另外还在高维Electricity（321通道）上测试；

**📈 对比分析**

在单台4 GB GPU上与线性、轻量频域及PatchTST等基线对比，使用MSE/MAE评估。FreqLite在L=336长窗口下平均MSE 0.3244，优于RLinear、DLinear、FITS等，并低于PatchTST Transformer；其参数约四倍少、显存2.2倍小、训练速度2.2倍快；在强非平稳场景下A-RevIN提升约5%；

**⚠️ 局限性**

主要限制：在大多数平稳基准上对RLinear的提升仅约0.9%；A-RevIN仅在非平稳数据上显效；模型仅对水平/尺度做一阶校正，未建模方差；实验仅在单4 GB GPU下完成，未验证更大规模或更强Transformer模型。

---

## 70. Transferring Information Across Interventions in Causal Bayesian Optimization

**arXiv ID:** 2606.01457 | [PDF](https://arxiv.org/pdf/2606.01457v1)

**作者:** Mohammad Ali Javidian `[一作]` (Appalachian State University), Mohammad Ali Javidian `[通讯]` (Appalachian State University)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5011786801)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种通过共享可识别因果参数来耦合不同干预响应的图耦合因果贝叶斯优化方法。

**💡 创新点**

引入了从因果图中提取共享参数生成低秩跨干预协方差核，允许干预间信息共享并获得对数级信息增益与 regret 分解。

**🔧 技术方法**

结合了结构因果模型、do-算子、Gaussian Process、信息增益理论、线性高斯 SEM、Jacobian 线性化与非线性路径 Lipschitz 传播。

**📊 数据集**

在理论对齐的线性 Gaussian 链、交叉集转移压测、非线性共享机制模拟、标准因果优化基准，以及真实 ECOLI70 Gaussian 网络实验中进行验证。

**📈 对比分析**

与传统独立干预集 GP、普通 BO 以及基线 CBO 进行对比，GC‑CBO 在干预集合较大、父干预不可用或数据稀疏的情形下实现了显著的值/成本收益提升。

**⚠️ 局限性**

对非线性结构仅提供一阶近似，适用性受限；自适应核动态更新与非平稳 GP 理论尚未完整；在父干预可用、探索空间小的情形下优势不明显。

---

## 71. The Shape of Wisdom: Decision Trajectories in Language Models

**arXiv ID:** 2606.01202 | [PDF](https://arxiv.org/pdf/2606.01202v1)

**作者:** Shailesh Rana `[一作]` `[通讯]` (Independent Researcher), Shailesh Rana (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在多层网络中答案形成的轨迹，并用状态、运动和边界距离三种可读指标对轨迹进行分类。

**💡 创新点**

创新点在于将答案视为深度轨迹而非单点终端，引入三维坐标（状态、运动、边界）来区分答案的稳定性，并通过注意力/MLP标量、span删减和对比实验揭示轨迹变动的机制。

**🔧 技术方法**

使用层级分层得分、答案边际、漂移、绝对距离，计算注意力与MLP贡献标量；实施span删除干预、回放与替换的对比实验；用线性回归评估标量对漂移的解释力。

**📊 数据集**

基于四选项 MMLU 题库，对 Qwen2.5‑7B、Llama‑3.1‑8B、Mistral‑7B‑Instruct 三大 7–8B 指令调优模型进行推理轨迹记录。

**📈 对比分析**

通过统计三模型的最终多选准确率、轨迹类型分布，以及 R²、span 删除效应和替换实验结果进行对比，发现注意力在层级上更显著推动稳定正确轨迹，而 MLP 在回放实验中对最终边际影响更大。

**⚠️ 局限性**

局限性：仅适用于四选项 MMLU、三大 7–8B 指令调优模型，使用答案位置读出；未揭示完整隐藏状态电路；结果对回放与替换协议高度敏感；深度层累积误差限制了长程推理结论。

---

## 72. Where to Look: Can Foundation Models Reach a Target Viewpoint Through Active Exploration?

**arXiv ID:** 2606.01247 | [PDF](https://arxiv.org/pdf/2606.01247v1)

**作者:** Liyang Li `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University)

**通讯引用:** 70767 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了目标视角再现（TVR）任务及其Bench标注，用以评估模型在3D环境中主动调整视角的能力。

**💡 创新点**

创新点在于将视角控制任务从被动识别转为主动探索，构建可诊断的Bench并通过视觉-动作历史微调及多步强化学习显著提升模型性能。

**🔧 技术方法**

采用了视觉-动作监督微调（VA-SFT）、链式思维（CoT-SFT）、单步GRPO与多步GRPO等技术，并结合Qwen3.5-9B等大型语言模型。

**📊 数据集**

使用AI2-THOR与ProcTHOR-10k场景，生成单/多房间、易/难四类任务，总共500个测试任务。

**📈 对比分析**

与多种开源/闭源基础模型对比，最佳微调后模型在Bench上的成功率提升至51.4%，显著高于未微调模型的12%及人类93%的差距。

**⚠️ 局限性**

局限性包括仅在离散模拟环境中测试，缺乏连续空间和现实物理世界验证；结果仅基于单一9B模型，未验证跨模型通用性。

---

## 73. AdaKernel: Learning Adaptive Kernel Parameters for Spatiotemporal Graph Neural Networks

**arXiv ID:** 2606.01283 | [PDF](https://arxiv.org/pdf/2606.01283v1)

**作者:** Zhongyue Zhang `[一作]` (Sichuan University), Yuankai Wu `[通讯]` (Sichuan University)

**通讯引用:** 15343 | [OpenAlex ID](https://openalex.org/A5100758454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AdaKernel，利用可学习的核参数替代固定距离核，实现更精确的空间依赖建模。

**💡 创新点**

首次证明固定核参数会导致不可消除的近似误差，并通过边缘特定或全局可学习尺度显著提升 GNN 对 Kriging、填补与预测的性能。

**🔧 技术方法**

结合 Gaussian/Matérn/RQ 核的可学习尺度、对称归一化邻接矩阵、端到端训练与注意力对比等技术。

**📊 数据集**

在 METR‑LA、PEMS‑BAY、AQI 等八个公开交通与空气质量数据集上进行实验。

**📈 对比分析**

与多种基线（GRIN、MPGRU、IGNNK、KITS、DCRNN、STGCN、GWNet、DGCRN）以及无学习核或全局自适应图相对，AdaKernel 在填补、kriging 与预测任务中平均提升 3–8% MAE/ RMSE。

**⚠️ 局限性**

仅探索了少量核族，且在极度稀疏或高维空间时可学习参数可能收敛不稳定。

---

## 74. Revisiting Neural Processes via Fourier Transform and Volterra Series

**arXiv ID:** 2606.01172 | [PDF](https://arxiv.org/pdf/2606.01172v1)

**作者:** Peiman Mohseni `[一作]` (Texas A&M University), Raymond K. W. Wong `[通讯]` (Texas A&M University)

**通讯引用:** 6102 | [OpenAlex ID](https://openalex.org/A5049858061)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了利用频域卷积（Set Fourier Convolutions, SFConvs）和 Volterra 系列来构建可解释且高效的平移等变条件神经过程（CNP）模型，分别为 SFConvCNP 和 SFVConvCNP。

**💡 创新点**

创新点在于：①通过 Volterra 展开显式表征连续平移等变算子为高阶卷积之和，提供可分析的函数类；②引入 SFConvs 在频域直接处理不规则采样数据，避免网格化、实现近似全局感受野且保持线性复杂度；③在此框架下构造了两类全新的 CNP，兼顾可解释性与效率。

**🔧 技术方法**

技术手段包括：卷积深集框架、Volterra 系列展开、低秩核分解、频域 Fourier 变换、正则化/多频点采样、点乘和逆变换、基于 PyTorch 的实现。

**📊 数据集**

使用了多种数据集：一维合成 GP（RBF、Matérn、周期、锯齿、方波）、二维图像完成（CIFAR-10、SVHN）、三维 Navier-Stokes Kolmogorov 流（空间+时间）、ERA5 气候温度预测（不同地理区域）以及 Lotka–Volterra 捕食者–猎物模拟。

**📈 对比分析**

与现有 CNP、Attention CNP、ConvCNP、Transformer NP 以及其等变版本进行对比。实验表明 SFConvCNP 与 SFVConvCNP 在大多数基准上获得最优或竞争性预测对数似然和 CRPS，尤其在高维、稀疏采样和跨区域泛化任务中显著优于非等变模型。

**⚠️ 局限性**

局限性主要包括：①在 Volterra 版本中参数量受限导致相对性能略低；②低秩近似与频点数量对表达能力有影响；③对极端高维或极稀疏采样场景的理论与实验验证尚未充分；④实现中仍依赖对频域采样的经验设定。

---

## 75. Implicit Geographic Inference in LLM Medical Triage: Language-Driven Disparities in Emergency Recommendations

**arXiv ID:** 2606.01204 | [PDF](https://arxiv.org/pdf/2606.01204v1)

**作者:** Qi Han Wong `[一作]` `[通讯]`, Qi Han Wong

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用Gemini 3.5 Flash对相同症状用六种语言进行对话，评估其医疗分诊建议。

**💡 创新点**

发现模型通过语言隐式推断地理位置，从而采用地区医疗规范导致分诊差异，首次揭示语言驱动的地理偏差。

**🔧 技术方法**

利用低温度设置的LLM推理、手工构造的多语言症状提示、后翻译控制和位置锚点提示。

**📊 数据集**

构建了六种语言（英语、西班牙语、中文、印地语、日语、阿拉伯语）各30次、共450次API调用的数据集。

**📈 对比分析**

比较方法为统计ER建议率、严重度分数和紧急性分类，并通过Wilson置信区间验证显著性；发现非英语提示在无位置信息时ER率低至0%，而英语提示为30%，后翻译和位置锚点验证机制，效果显著。

**⚠️ 局限性**

局限在于仅评估单一模型与单一症状场景、缺乏医学专业验证、样本量有限、未覆盖人类临床基准、系统提示仅英文，可能影响结果的普适性。

---

## 76. The anti-lexicographic SUS-anchor: a near-optimal k=1 sampling scheme

**arXiv ID:** 2606.01190 | [PDF](https://arxiv.org/pdf/2606.01190v1)

**作者:** Groot Koerkamp `[一作]` (Karlsruhe Institute of Technology), Ragnar `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5079994235)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的采样方案——最小唯一子串锚点（SUS-anchor），通过在窗口内寻找最小唯一后缀来决定采样位置，并给出了线性时间、O(w) 空间的流式算法。

**💡 创新点**

创新点包括：① 引入了反词典（anti‑lexicographic）字符顺序，使得SUS-anchor的采样密度接近理论下限；② 将SUS-anchor与最大后缀（maximal suffix）概念等价，简化了算法实现；③ 在 k=1 的选取方案中，首次实现了在小字母表上接近最优的采样密度。

**🔧 技术方法**

技术方法：字符基顺序（character‑based order）的定义与实现；使用双向链表和事件队列（或简化的双端队列）实现窗口内后缀比较的单调队列；通过随机字符串实验验证密度性能；对比不同字母表和窗口长度的实验数据。

**📊 数据集**

实验数据集为长度 10⁷ 的随机字符串，字母表大小 σ∈{2,4,32}，窗口长度 w 变化范围从 1 到几千；使用这些随机序列来评估密度和与下界的偏差。

**📈 对比分析**

比较方法：将 SUS-anchor（lexicographic 与 anti‑lexicographic 两种顺序）与已知方案（bd‑anchor、mod‑minimizer、GreedyMini 等）在相同随机序列上进行采样密度比较。结果显示：anti‑lexicographic SUS-anchor 的密度在 σ=4 时小于 1% 的下界偏差，在 σ=2 时小于 10%，远优于 bd‑anchor 的 15%–50% 之上限。

**⚠️ 局限性**

局限性：① 目前实现基于单调队列，尚未达到最优的 SIMD 并行实现；② 仅针对 k=1 的选取方案，未覆盖更大 k 的情况；③ 需要理论证明 anti‑lexicographic SUS-anchor 的密度确实趋近于 2/(w+1)；④ 对于非前向（non‑forward）方案的潜在更优性能仍未探索。

---

## 77. Tether-Aware Dynamic Collision Avoidance for USV-HROV Systems

**arXiv ID:** 2606.01112 | [PDF](https://arxiv.org/pdf/2606.01112v1)

**作者:** Yang Gu `[一作]`, Yulin Si `[通讯]` (Zhejiang University)

**通讯引用:** 3052 | [OpenAlex ID](https://openalex.org/A5031841011)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了针对海底电缆检测中无人水面艇跟随混合遥控艇时的、考虑缆绳安全和张力的动态碰撞规避方法，并将其嵌入线视距跟踪框架，实现了在避障过程中的缆绳张力管理与安全性提升。

**💡 创新点**

创新点包括：① 通过椭圆包络与投影降维构造的缆绳安全平面域，直接捕捉三维缆绳与障碍船舶的碰撞风险；② 结合张力管理和缆绳释放约束的 TTA‑VO 方法，实时调整缆绳长度，降低缆绳变紧风险；③ 在 LOS 跟踪框架中加入动态规避层，既保留主线跟踪，又在需要时自动触发规避。

**🔧 技术方法**

主要技术手段有：椭圆包络建模、投影降维、平面速度障碍 (VO) 规划、缆绳张力约束与释放策略、线视距跟踪（LOS）导引以及 Gazebo 仿真实现。

**📊 数据集**

实验采用 Gazebo 仿真环境，设置三艘运动障碍船舶（尺寸、速度、航向从表中给出）以及 USV/HROV 的初始位置与运动参数；未使用公开实测数据集。

**📈 对比分析**

比较方法：① 基础 VO（固定域）；② 加入缆绳安全平面域的平面域 VO；③ 进一步加入张力管理的 TTA‑VO。结果显示：基础 VO 虽避免船体碰撞，但缆绳与障碍仍相交；平面域 VO 成功分离缆绳与障碍，但在规避时缆绳可能拉紧；TTA‑VO 在规避后自动释放缆绳，保持正向张力余量（m_tet^min 约 2.55 m），并保持缆绳平面域与障碍域最小边界距离正值（d_bd^min 3.48 m），性能显著优于基线方法。

**⚠️ 局限性**

局限性：仅在仿真环境验证，未进行水槽或实场实验；缺乏对多艇相互作用的考虑；张力管理策略基于经验阈值，可能在极端海况下需要进一步校准。

---

## 78. Hybrid Imbalanced Regression Through Unified Data-Level and Algorithm-Level Balancing

**arXiv ID:** 2606.01221 | [PDF](https://arxiv.org/pdf/2606.01221v1)

**作者:** Shermin Shahbazi `[一作]` (University of Zanjan), Mohsen Afsharchi `[通讯]` (University of Zanjan)

**通讯引用:** 904 | [OpenAlex ID](https://openalex.org/A5047882313)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种融合数据级与算法级方法的端到端混合框架，用于解决连续目标空间的不平衡回归问题。

**💡 创新点**

首次将自适应目标空间分箱、Conditional Variational Autoencoder、基于特征聚类的多阶段重采样、潜在密度加权损失（LDWL）以及注意力门控融合相结合，实现数据与算法级策略的紧密协同。

**🔧 技术方法**

采用自适应R²分箱、CVAE条件表示学习、DBSCAN+SMOGN多阶段过采样、密度加权损失、gated fusion网络等技术。

**📊 数据集**

在16个常用基准回归数据集上进行实验，包含California、Compactive、CPU_small、Heat、Wine_quality、Abalone、Space_ga、Debutanizer、Available_power、Maximal_torque、Fuel_consumption_country、Acceleration、Airfoild、Mortgage、Treasury、ConcreteStrength。

**📈 对比分析**

通过10折交叉验证，将混合框架与单独的数据级或算法级平衡方法以及未平衡的基线回归器（MLP、XGBoost、LR、KNN、SVR、Ridge）在MAE、RMSE、R²等指标上进行比较，混合框架在所有数据集和指标上均优于单一方法，显著提升预测精度。

**⚠️ 局限性**

对目标分箱阈值和KDE带宽的手工设定、对大规模数据的密度估计开销、生成样本可能引入噪声以及在极端稀疏分区中的过度拟合风险等仍是需要进一步改进的限制。

---

## 79. FAiT: Frequency-Aware Inverted Transformer for Multivariate Time Series Forecasting

**arXiv ID:** 2606.01306 | [PDF](https://arxiv.org/pdf/2606.01306v1)

**作者:** Peng He `[一作]` (University of Electronic Science and Technology of China), Qiao Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 53508 | [OpenAlex ID](https://openalex.org/A5100441085)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 FAiT（Frequency-Aware Inverted Transformer），一种针对多变量时间序列预测的新型 Transformer 变体。

**💡 创新点**

核心创新在于引入“反向注意力”来显式补偿自注意力的低通滤波偏差，并结合动态时频调制（DTFM）实现实例化的频谱重分配，从而同时捕捉全局趋势与局部高频细节。

**🔧 技术方法**

技术上融合了自注意力、快速傅里叶变换（FFT）、逆变换、可逆实例归一化（RevIN）、可学习的高低频门控和轻量级 MLP 生成的动态频率权重。

**📊 数据集**

实验使用了 12 个公开基准数据集，包括 ETT、Electricity、Traffic、Weather、Exchange 以及 PEMS 系列等长短期预测任务。

**📈 对比分析**

与 14 类最先进方法（线性、Transformer、频域模型）在 MSE、MAE、R²、r、MASE 等指标上进行比较，FAiT 在大多数数据集上实现了 2–10% 的改进，稳居多变量预测的领先水平。

**⚠️ 局限性**

局限性在于仍采用预设的频谱原型，难以完全自适应极端非周期性变化；同时对极长序列或实时低延迟应用的计算成本与内存占用尚待进一步优化。

---

## 80. Rank-Aware Quantile Activation for Motion-Robust Crop Segmentation in UAV Imagery

**arXiv ID:** 2606.01118 | [PDF](https://arxiv.org/pdf/2606.01118v1)

**作者:** Abinav Kiran `[一作]`, Daya Sagar B S `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种双量化激活（Dual Quantile Activation, QAct）模块，替换传统的ReLU门控，改进UAV摄像机运动模糊下的农业语义分割，提供零样本鲁棒性；

**💡 创新点**

创新点在于引入基于实例级秩归一化的残差量化激活，能在模糊导致高频信号消失时保持少数类别特征，且在零样本与模糊监督两种训练模式下均成为主导鲁棒性因素；

**🔧 技术方法**

使用HRNet backbone + Dual QAct块；构建6-DOF物理运动模糊仿真；采用EMA教师-学生蒸馏框架；通过95%自助法评估误差区间；对不同模糊强度和方向进行鲁棒性分析；

**📊 数据集**

使用 Agriculture‑Vision 2021 数据集，包含8个语义类别，模拟Severity 2‑4的运动模糊以及多角度线性模糊；

**📈 对比分析**

对比四种模型（ZS‑ReLU、ZS‑QAct、Distill‑ReLU、Distill‑QAct）在不同模糊强度下的mIoU和类别IoU；结果显示零样本QAct在中等模糊下已优于蒸馏ReLU，Distill‑QAct在所有强度下表现最佳，尤其对稀有结构/纹理类提升显著；

**⚠️ 局限性**

局限性包括：仍缺乏真实模糊数据验证；对极端模糊（Severity 5）和多光谱融合的鲁棒性有限；QAct在主类（如水）上的优势并非一致；对模型计算成本和推理延迟的影响未做完整评估。

---

## 81. Recursive Jump Operators and Optimal Proof Systems

**arXiv ID:** 2606.01242 | [PDF](https://arxiv.org/pdf/2606.01242v1)

**作者:** Fabian Egidy `[一作]` `[通讯]` (University of Würzburg), Fabian Egidy (University of Würzburg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

论文研究证明系统与递归跳跃算子之间的关系，回答了Khaniki提出的关于不存在最优证明系统是否必然存在递归跳跃算子的问题，并通过构造稀疏oracle展示了该问题在可相对化的框架下的难度；同时证明了递归跳跃算子在-可归约下的向上封闭性，并利用Messner的结果为所有已知缺失最优证明系统的集合构造了递归跳跃算子。

**💡 创新点**

创新点包括：①利用稀疏oracle构造出同时满足“多项式时间层无限”“TAUT无最优证明系统”“TAUT无递归跳跃算子”的三重性质的oracle，证明了Q1在相对化方法中不可正面回答；②证明递归跳跃算子在-可归约下的向上封闭性，填补了该领域先前仅有最优证明系统向下封闭的结果；③通过对Messner证明的技术细化，直接为所有已知无最优证明系统的集合构造递归跳跃算子，进一步表明目前已知的无最优证明系统集合大多同时具备递归跳跃算子。

**🔧 技术方法**

采用的主要技术手段有：优先级构造（priority argument）实现稀疏oracle的构建；Kleene固定点定理用于构造自指示的证明系统；对跳跃算子与证明系统的模拟关系进行对角化；利用对数层与稀疏oracle的组合保证多项式时间层不变；以及对disjoint Π⁰₁对的可归约性进行细致分析。

**📊 数据集**

本研究为纯理论计算复杂性研究，无需使用实验数据集；所有结论均来自构造性证明与逻辑推导。

**📈 对比分析**

由于论文是理论性工作，没有实验对比；其贡献体现在通过构造oracle给出相对化限制，和在已知无最优证明系统的集合上给出递归跳跃算子，说明这些集合在相对化框架下满足期望的性质。

**⚠️ 局限性**

局限性：①结果仅在可相对化方法内成立，无法直接证明Q1的全局（无oracle）正面答案；②对递归跳跃算子的正面结果仅适用于已知缺失最优证明系统的集合，尚未覆盖所有可能的L；③构造的oracle需要稀疏性与多项式时间层无限等假设，实际在无oracle情境下的意义仍待进一步研究。

---

## 82. Beyond Sinusoids: A Morlet Wavelet Framework for Transformer Positional Encoding

**arXiv ID:** 2606.01258 | [PDF](https://arxiv.org/pdf/2606.01258v1)

**作者:** Athanasios Zeris `[一作]` `[通讯]` (Independent Researcher), Athanasios Zeris (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Morlet波浪位置编码并与Energy‑Gated Attention结合，实现对位置的频率与局部性双重学习。

**💡 创新点**

将sin/cos、ROPE、ALiBi统一为可学习频率与局部性的Morlet波浪编码，满足Heisenberg不确定性原理，提供最小不确定性的时间‑频率局部化。

**🔧 技术方法**

使用Morlet Positional Encoding、Energy‑Gated Attention、波形理论与不确定性分析，并在Transformer结构中实现。

**📊 数据集**

使用字符级TinyShakespeare数据集进行训练与评估。

**📈 对比分析**

在TinyShakespeare 256长度上下文下，与学习位置嵌入、sin/cos、ROPE、单独EGA以及两者组合进行对比，Morlet+EGA实现+0.119的性能提升，超过任何单独组件。

**⚠️ 局限性**

实验仅在单个seed、字符级、短上下文下完成，缺乏多seed、多尺度、词级、长上下文验证；存在起点偏置、可学习中心位置缺失、可达性约束强制等限制。

---

## 83. HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces

**arXiv ID:** 2606.01117 | [PDF](https://arxiv.org/pdf/2606.01117v1)

**作者:** Nasib Ullah `[一作]` (Aalto University), Rohit Babbar `[通讯]` (University of Bath)

**通讯引用:** 974 | [OpenAlex ID](https://openalex.org/A5102987303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对极端多标签分类的硬件感知动态稀疏训练框架HASTE，利用组共享固定fan‑in稀疏输出层实现内存与计算效率的显著提升。

**💡 创新点**

创新点在于将标签按语义相似度分组共享相同的稀疏特征模式，既保留了每个标签的独立权重，又实现了更高的特征重用和GPU友好的内存访问；同时采用head‑tail分解替代辅助损失，提升训练稳定性。

**🔧 技术方法**

使用了CUDA自定义核实现“先聚集一次再密集MMA”的前向与反向计算，结合Split‑K并行化梯度累加，配合BF16低精度训练与动态稀疏重连策略。

**📊 数据集**

在Amazon‑670K、AmazonTitles‑670K、Amazon‑3M、LF‑Paper2Keywords‑8.6M等大规模极端分类数据集上进行评测。

**📈 对比分析**

与采样式、稠密及其他稀疏基线（如Spartex、block‑sparse）相比，HASTE在保持或提升P@k/PSP@k的同时，训练时段缩短至1/4以上，显存占用减少约3–4GB，且在A100 GPU上获得4.4×至25×的前向/反向速度提升。

**⚠️ 局限性**

局限在于对标签分组和稀疏比例的超参数调优敏感，过大组尺寸会牺牲精度，且在极端稀疏（>96%）时仍需进一步优化硬件利用和梯度累加机制。

---

## 84. NetVAD: Foundation-Model Representation Learning for Identifier-Free Unsupervised Intrusion Detection

**arXiv ID:** 2606.01452 | [PDF](https://arxiv.org/pdf/2606.01452v1)

**作者:** Darren Fürst `[一作]` (Ostbayerische Technische Hochschule Amberg-Weiden), Sebastian Steindl `[通讯]` (Ostbayerische Technische Hochschule Amberg-Weiden)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5040501394)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于冻结网络基础模型的变分自编码器 NetVAD，能在不使用攻击样本且严格去除 IP、端口、时间戳等标识符的情况下，进行零日攻击的无监督检测。

**💡 创新点**

创新点包括：
- 将自监督预训练的网络基础模型作为固定特征提取器，避免对网络标识符的 shortcut 学习；
- 通过在其嵌入空间训练 VAE，构建仅学习正常流量的低维潜在空间，从而以重构误差检测异常；
- 证明大规模预训练与专门设计的 U‑Net + residual 细化解码器对性能至关重要。

**🔧 技术方法**

技术核心：
- 预训练网络基础模型（netFound）
- 变分自编码器（Encoder‑Projection、U‑Net skip、残差注意力块、oversampling Decoder）
- 循环 β‑annealing 的 VAE ELBO 损失
- 以阈值化 FPR≤5% 的操作级别阈值校准。

**📊 数据集**

使用的数据集：ToN‑IoT（包含 9 类攻击）和 IoT‑23（包含 23 个真实 IoT 恶意样本）。

**📈 对比分析**

与经典方法（Isolation Forest、OCSVM、传统 Autoencoder 等）比较：
- 在 ToN‑IoT 上，NetVAD 微 F1≈98%，宏 F1≈96%，比 Isolation Forest 及 OCSVM 在 identifier‑free 设置下提升显著；
- 在 IoT‑23 上，NetVAD 对复杂僵尸网络（如 Okiru）达到 99.6% F1，尽管整体宏 F1 仅 47.6%，但明显优于传统方法；
- ablation 结果表明缺失预训练或简化解码器会导致宏 F1 降至 19%~16%。

**⚠️ 局限性**

限制：
- 对单包侦查或低频事件检测效果差，因重构误差难以区分；
- 仅测试单一 FM（netFound），未验证不同基础模型的泛化；
- 未评估推理延迟、模型尺寸和量化压缩等部署成本。

---

## 85. BRo-JEPA: Learning Modular Arithmetic in Latent Space

**arXiv ID:** 2606.01372 | [PDF](https://arxiv.org/pdf/2606.01372v1)

**作者:** Divyansh Jha `[一作]` (Georgia Institute of Technology), Brennen Yu `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在MNIST数字上构建一个模块化算术任务，利用JEPA框架在潜在空间学习数字状态间的加减运算，并通过引入基于块旋转的预测器实现对未见运算的零样本推断。

**💡 创新点**

创新点在于将旋转位置编码（RoPE）思想迁移为潜在空间的动态运算模块（Block Rotation Predictor），从而在潜在空间中强制执行模10循环结构，实现真正的零样本泛化；并在此基础上提供可学习与固定旋转角度的两种实现。

**🔧 技术方法**

使用JEPA（Joint Embedding Predictive Architecture）作为基础网络，采用可加动作嵌入、组合一致性损失、VICReg正则化等技术；核心是基于块旋转的预测器（BRo-JEPA）以及可学习/固定旋转角度的实现。

**📊 数据集**

数据集为MNIST手写数字，利用数字标签构造模10加减运算的状态转换对；训练仅使用±1操作，评估未见的±2…±9操作。

**📈 对比分析**

与传统监督分类基线、JEPA+可加嵌入、JEPA+组合一致性损失以及基于块旋转的监督基线进行比较。BRo-JEPA在未见操作上的零样本准确率最高，MFR版本在ResNet-18基线上达到99.46%零样本准确率，rollout和KNN准确率也超过99%；相比之下，传统基线和JEPA仅在见过操作上表现良好，未见操作准确率不足10%。

**⚠️ 局限性**

局限性在于实验环境极其受控（MNIST、已知模10循环结构），块旋转预测器过度匹配任务结构，难以直接推广至更复杂或未知的符号推理场景；并且在可学习角度下的SFR版本对泛化的依赖较弱。

---

## 86. Beyond Access: Guided LLM Scaffolding for Independent Learning in Undergraduate Statistics

**arXiv ID:** 2606.01375 | [PDF](https://arxiv.org/pdf/2606.01375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 87. Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking

**arXiv ID:** 2606.01120 | [PDF](https://arxiv.org/pdf/2606.01120v1)

**作者:** Yuxi Sun `[一作]` (Hong Kong Baptist University), Jing Ma `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 23515 | [OpenAlex ID](https://openalex.org/A5020347295)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PAVE评估框架，系统研究LLM在检索-增强事实核查（RAG）中先验知识与检索证据冲突时的仲裁行为，构建含先验-上下文不一致（PCD）的对照数据集；

**💡 创新点**

在评估中引入知识边界（KB）四态（Known‑Knows、Known‑Unknows、Unknown‑Knows、Unknown‑Unknows）和专门的纠正率/持久率指标，并提出基于Jensen‑Shannon Divergence（JSD）的轻量级测试时仲裁方法，能在不改模型的前提下提升事实可靠性；

**🔧 技术方法**

使用一致性采样评估先验可信度、JSD度量稳定性、PAVE的四层指标、对比多种基线（Self‑guided、Rule‑based、Context‑based）以及GPT、Gemini、Deepseek、Phi‑4、Qwen3、Mistral、Llama3等大型/中型LLM；

**📊 数据集**

构造两类PCD数据集：1）对照（Counterfactual）来源于Quantemp、PolitiFact、Snopes，并通过实体替换、语义对抗等方式生成误导性证据；2）时间（Temporal）来源于2024‑2025年维基百科当前事件，检验模型对过期或缺失知识的更新能力；

**📈 对比分析**

与八个基线对比，PAVE下的JSD仲裁方法在六大LLM族中实现了最佳或竞争性整体准确率，尤其在小模型（如Mistral‑7B）上提升约6.6个百分点；相比传统方法更能平衡正确证据与错误证据的鲁棒性；

**⚠️ 局限性**

局限性包括：仅针对二元事实核查，未考虑多标签或开放式解释；假设检索已完成，忽略检索内部冲突与自动跳过RAG的情形；对“都错误”案例未做实验；构造的对照证据可能含生成噪声；方法在推理时需多次采样，增加计算开销。

---

## 88. HiTokSR: A Coarse-to-Fine Tokenizer with Hierarchical Codebooks for High-Fidelity Real-World Image Super-Resolution

**arXiv ID:** 2606.01157 | [PDF](https://arxiv.org/pdf/2606.01157v1)

**作者:** Mingxi Li `[一作]` `[通讯]`, Mingxi Li

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种层次化token预测框架HiTokSR，用多组子码本对图像潜在空间进行频率感知拆分，并结合语义先验和解码器微调来实现单前向高效的真实图像超分辨率。

**💡 创新点**

创新点在于：1) 将潜在空间按通道分组，使用独立子码本避免高维近邻搜索的维度灾难；2) 采用组掩码与离散小波变换监督实现粗细层次的结构-纹理分离；3) 通过视觉基础模型的自适应特征调制、多尺度CLS标记和表示对齐，提升语义一致性；4) 解码器对预测token的Top‑K扰动微调，弥补训练-测试差异。

**🔧 技术方法**

使用了ViT编码器、Transformer生成器、可调式子码本、频率匹配监督、LoRA微调、SFT/CLS语义调制、DWT监督、Top‑K扰动解码器微调、LPIPS/GAN/多指标损失。

**📊 数据集**

在训练中采用LSDIR+FFHQ（10k）合成数据，评估在RealSR、DRealSR、RealLQ250三大真实超分基准上进行。

**📈 对比分析**

与多种扩散模型（StableSR、OSEDiff等）和VQ模型（FeMASR、TVQ‑RAP、VARSR）对比，HiTokSR在LPIPS、DISTS、FID、PSNR等指标上均位居前列，且单步推理耗时仅0.047 s，参数量与算力低于其他方法，显示出优越的效率-质量平衡。

**⚠️ 局限性**

局限性包括：1) 仍需较多的超参（组数、掩码分布、Top‑K范围）调优；2) 依赖预训练视觉基础模型，若该模型对低质量输入适应不足可能影响语义对齐；3) 目前未在无参考的极端噪声环境下进行深入评估。

---

## 89. Residual-Weighted Randomized Jacobi: Sharpened Bounds via Residual Concentration and Asynchronous Extension

**arXiv ID:** 2606.01232 | [PDF](https://arxiv.org/pdf/2606.01232v1)

**作者:** Evan Coleman `[一作]` `[通讯]` (University of Mary Washington), Evan Coleman (University of Mary Washington)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了残差加权随机Jacobi（power‑weighted Jacobi）方法在同步和异步求解线性系统中的收敛性质，并提出了利用逆参与度（IPR）对收敛率进行精确刻画的新分析框架。

**💡 创新点**

创新点：①提出残差权重采样概率 P(k=j)∝|r_j|^ℓ 的连续谱，ℓ→0退化为均匀采样、ℓ→∞退化为Southwell贪婪；②证明 ℓ=2 时收敛率可写成标准收敛率乘以 IPR，显著提高对非均匀残差的预测精度；③将该分析推广到异步 ADG 框架，给出以 IPR 为参数的 epoch 收敛定理；④实验揭示一致读在高并发下导致发散，异步读更稳定，验证了 IPR 对收敛与冲突率双重影响。

**🔧 技术方法**

使用技术包括：随机坐标更新、逆参与度（IPR）理论、Avron–Druinsky–Gupta（ADG）异步框架、Cauchy–Schwarz 与 Hölder 不等式、马尔可夫链期望、数值实验与原子操作实现。

**📊 数据集**

实验数据集：三类结构化 SPD 系统——2D Laplacian（平滑右端项和点源右端项）以及 1D FEM 质量矩阵，网格大小 N=128（n=16 384）和 n=8 192；均采用矩阵自由实现以保持更新成本一致。

**📈 对比分析**

对比方法：与均匀随机 Jacobi、周期性更新以及随机坐标下降等传统方法对比；在同步设置下，IPR‑增强的收敛上界比传统均匀采样上界更紧；在 96 线程异步运行中，ℓ=2 的残差加权采样比均匀采样快约 2–3 倍，且 IPR 随线程数变化不超过 3%；一致读模式在高并发下出现发散，而异步读模式保持收敛，验证了理论预言。

**⚠️ 局限性**

局限性：①异步分析需假设 IPR 在整个 epoch 内保持在一个全局上界，导致交叉项被过度保守估计（出现 O(ν/√n) 代替 O(1/n)）；②一致读的收敛定理仍缺失，需构建不一致读的正式理论；③目前仅对 ℓ=2 完整证明，其他 ℓ 的最优采样与收敛率尚未系统化；④分析对冲突与残差耦合的上界仍过于宽松，实际冲突率与理论估计存在显著差距。

---

## 90. DAGGER: Gradient-Free Construction of Transiently Amplifying Networks under Hard Connectivity Constraints

**arXiv ID:** 2606.01227 | [PDF](https://arxiv.org/pdf/2606.01227v1)

**作者:** James C. Ferguson `[一作]` `[通讯]` (African Institute for Mathematical Sciences), James C. Ferguson (African Institute for Mathematical Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为 DAGGER 的单前向、无梯度的算法，用于在符号、稀疏和对角线严格约束下，将稳定的有符号稀疏矩阵重赋权成极大非正常放大的网络。

**💡 创新点**

创新点在于：① 将节点排序为尽量无环（DAG）并计算每条边的前向分数；② 通过重排列不等式将幅度多重集按分数匹配，最大化放大；③ 用单一标量 β 通过指数倾斜控制幅度分布，从严格保持幅度多重集到几乎无限放大形成可调 Pareto 前沿；④ 该方法不需要梯度迭代，显著减少特征分解次数并提升计算速度。

**🔧 技术方法**

使用技术包括：贪心最小反馈弧排序、前向分数计算与循环长度惩罚、指数幅度倾斜、按分数排序匹配、整体非负缩放以满足稳定性阈值、可选的梯度基交换细化。

**📊 数据集**

实验数据集主要为合成随机稀疏有符号矩阵（密度0.18，权重为 lognormal 分布），在不同规模（n=60, 80, 120, 200, 300）以及信号检测任务中使用 n=80 或 200 的子样本。

**📈 对比分析**

与 SOC（稳定性优化电路）、GRAD‑Adam（梯度上升）、L0（仅比例缩放）以及 SOC 的适配版本进行对比。DAGGER 在峰值放大（峰值可达 10^10）、Wasserstein‑2 预算、单前向特征分解次数（≈45）以及 wall‑clock 速度（≈5×快于 SOC）上显著优于其他方法；β 递增可实现数十倍放大并形成连续 Pareto 曲线。

**⚠️ 局限性**

局限性包括：① 拓扑结构保持固定，无法突破约束下的非零上界；② 指数倾斜并非最优，可能导致放大效率不足；③ 节点排序采用贪心启发式，未证明最优；④ 多模放大可能不满足单模放大需求。

---

## 91. DeblurNVS: Geometric Latent Diffusion for Novel View Synthesis from Sparse Motion-Blurred Images

**arXiv ID:** 2606.01315 | [PDF](https://arxiv.org/pdf/2606.01315v1)

**作者:** Changyue Shi `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 18528 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无场景优化、可从稀疏运动模糊图像直接合成清晰新视图的通用框架DeblurNVS

**💡 创新点**

通过在几何潜空间中先恢复模糊上下文潜码，再用相机感知扩散合成目标潜码，解决了运动模糊对跨视角一致性的破坏

**🔧 技术方法**

几何潜空间扩散（GLD）+DA3编码器+LoRA适配+轻量解码器+运动模糊数据合成

**📊 数据集**

构建DL3DV-10K-Blur数据集（≈10k场景、≈5M blur–sharp 对）并在DeblurNeRF-Real、DeblurNeRF-Blender等基准上评估

**📈 对比分析**

与3DGS、BAGS、DA3、GLD等基准对比，DeblurNVS在无场景优化的情况下在LPIPS、DISTS、FID等感知指标上优于现有方法，速度仅比GLD慢，但远快于场景优化方法

**⚠️ 局限性**

推理慢（需扩散迭代），在极度模糊或大视角变换时可能出现伪影或细节失真，对纹理细节一致性仍有限

---

## 92. Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence

**arXiv ID:** 2606.01444 | [PDF](https://arxiv.org/pdf/2606.01444v1)

**作者:** Fiona Y. Wang `[一作]` (Massachusetts Institute of Technology), Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 54245 | [OpenAlex ID](https://openalex.org/A5011504360)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

本文提出一种基于范畴论的“代理式发现”框架，将检索、搜索和发现分别建模为复制、更新和范式转移，并通过案例演示了该框架在蛋白质机械模型与纤维网络力学模型中的应用。

**💡 创新点**

创新点在于：①将科学发现抽象为“规约状态”与“范式迁移”两层结构，利用左Kan扩张实现旧证据向新范式的可验证传输；②构建统一的可执行、可审计的知识‑计算图，将知识图谱、工作流、门控与公开论证融为一体；③通过MDL/AIC门控实现自我修正的量化评价。

**🔧 技术方法**

技术方法包括：范畴论（copresheaf、元素范畴、Kan扩张）、最小描述长度/信息准则、模型选择与AIC、结构化的技能与工件注册、压力评分与主动工作流变异，以及面向科学实验的可证明计算层。

**📊 数据集**

使用数据集：PDB蛋白链与对应的B‑factor数据用于Builder/Breaker蛋白机械实验；纤维取向与应力‑应变表用于CategoryScienceClaw纤维网络案例；两案例均基于公开的实验/模拟数据，并未使用大规模机器学习数据集。

**📈 对比分析**

对比方法主要通过信息量与模型复杂度的变化评估：在Builder/Breaker中MDL压缩从起始模型提升至最终模型累计获得约54.3比特的压缩；在CategoryScienceClaw中AIC差值为123.87，表明选择的异向性刚度模型在统计意义上更优；由于缺少传统机器学习基线，本文侧重展示自我验证与语义可追溯性的提升。

**⚠️ 局限性**

局限性包括：①需要先验手工构建的范式与语法，缺乏自动化学习机制；②对大规模数据与复杂模型的计算开销尚未评估；③在实际实验环境下门控的阈值设定依赖专家经验；④目前仅在两类材料科学案例中验证，泛化能力待进一步检验。

---

## 93. PairedGTA: Generating Driving Datasets for Controlled Photometric Shift Analysis

**arXiv ID:** 2606.01192 | [PDF](https://arxiv.org/pdf/2606.01192v1)

**作者:** Andrea Chianese `[一作]` (Scuola Superiore Sant'Anna), Giorgio Buttazzo `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 13574 | [OpenAlex ID](https://openalex.org/A5024920325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用 GTA V 游戏引擎生成同一场景在不同光照与天气条件下的像素对齐图像，并用此数据评估语义分割模型的光照/天气鲁棒性。

**💡 创新点**

通过在游戏引擎中可控地改变光照和天气，同时保持摄像机姿态、场景几何与动态物体位置不变，得到完全配对的光照/天气差异数据集；并使用伪标签进行无监督一致性评估。

**🔧 技术方法**

使用 DeepGTAV、VPilot、Script Hook V 等游戏引擎控制 API 与 Python 客户端进行场景初始化、动态物体实例化和环境渲染；伪标签采用 SegFormer‑B5；评估指标为基于伪标签的类级一致性分数。

**📊 数据集**

自定义生成的 GTA V 数据集（约 100 个位置，每个位置 9 种光照/天气组合），对照 ACDC 真实世界配对数据。

**📈 对比分析**

比较方法：在相同伪标签基准下，计算各模型在不同环境下的类级一致性分数；结果显示 SegFormer‑B5 在大多数类别上平均一致性最高，尤其在日照→夜晚条件下；与 ACDC 比较，配对数据更能体现光照鲁棒性。

**⚠️ 局限性**

局限：缺乏真实语义标注，只能使用伪标签；对游戏引擎的间接控制导致偶发异常；生成速度受限，且仅适用于已存在的游戏引擎；未涵盖其他任务或更大规模扩展。

---

## 94. Everywhere Learning: Artificial Intelligence with Pointwise Constraints

**arXiv ID:** 2606.01557 | [PDF](https://arxiv.org/pdf/2606.01557v1)

**作者:** Ignacio Boero `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**通讯引用:** 16364 | [OpenAlex ID](https://openalex.org/A5078862959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“everywhere learning”框架，在AI训练中使用点对点约束，确保几乎所有样本满足损失上限，并通过对偶理论与泛化分析验证其有效性。

**💡 创新点**

将目标与约束分离，构造基于对偶敏感度的PAC学习条件，并通过稀疏L1惩罚和对偶剪裁实现对非凸模型的泛化控制。

**🔧 技术方法**

采用对偶学习、近似对偶性理论、L1稀疏惩罚、图神经网络与句子变换器混合架构以及对偶变量剪裁技术。

**📊 数据集**

使用FLORA-Bench中的工作流分类任务，数据来源于HumanEval和MBPP基准。

**📈 对比分析**

与传统ERM方法对比，在点对点约束下显著降低尾部损失、提升性能均匀性，Humaneval样本损失下降8%，准确率提升7.5%。

**⚠️ 局限性**

需对偶域进行限制才能保证学习性；对偶敏感度和参数近似误差仍可能限制非凸模型的理论保障，实验验证仅限于特定分类任务。

---

## 95. Expected Value Alignment for Generative Reward Modeling in Formal Mathematics Verification

**arXiv ID:** 2606.01160 | [PDF](https://arxiv.org/pdf/2606.01160v1)

**作者:** Shihao Ji `[一作]`, Mingyu Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Expected Value Alignment（EVA）框架，在 Lean 4 定理证明中生成解释性批判并提取连续奖励。

**💡 创新点**

通过在离散 JSON 输出中利用 token logits 的期望值，既保留可解析的文本输出，又获得连续评分，解决了传统价值头和文本生成奖励的折衷。

**🔧 技术方法**

结合因果语言模型、LoRA 微调、MSE 对齐损失、锚点软化的期望分布、字符到 token 的精准映射以及 deterministic decoding 等技术。

**📊 数据集**

使用了约 1800 条 Lean 4 证明状态与非正式推理的人工标注数据，包含逻辑、对齐和清晰度三维评分，并以 400 条测试集进行评估。

**📈 对比分析**

与 GPT-4o、Qwen2.5-1.5B 零样本、标准 SFT 生成奖励模型以及价值头奖励模型对比；EVA 在 MSE、Pearson r 以及排序准确率上均优于其他方法，Pearson r 达 0.824，排序准确率 84.6%。

**⚠️ 局限性**

依赖于锚点为单 token 的 tokenizer，需精准定位分数 token；连续奖励仅是学习到的代理，不能替代 Lean 的正式校验；在闭环搜索中的效能仍需进一步验证。

---

## 96. DiffuSent: Towards a Unified Diffusion Framework for Aspect-Based Sentiment Analysis

**arXiv ID:** 2606.01323 | [PDF](https://arxiv.org/pdf/2606.01323v1)

**作者:** Shu Long `[一作]` (University of Electronic Science and Technology of China), Xuchuan Zhou `[通讯]` (Southwest Minzu University)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5031558926)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DiffuSent，一种统一的非自回归扩散框架，用边界去噪的方式实现所有七个 ABSA 子任务的端到端生成。

**💡 创新点**

创新点包括：① 将 ABSA 的所有子任务抽象为边界去噪扩散过程；② 引入对比去噪训练策略，显著降低因噪声引起的边界重复和错误预测；③ 通过非自回归解码同时输出多对 aspect‑opinion‑sentiment triplets，提升推理效率。

**🔧 技术方法**

使用技术：生成式扩散模型（DDIM）、Transformer 编码器+跨注意力、对比去噪损失、Hungarian 匹配、预训练语言模型（BERT/T5 等）进行句子编码。

**📊 数据集**

数据集：SemEval 2017（D_17）、SemEval 2019（D_19）、SemEval 2020A（D_20a）和 SemEval 2020B（D_20b），涵盖 AE、OE、ALSC、AOE、AESC、AOPE、ASTE 等七个子任务。

**📈 对比分析**

方法通过 F1 分数与多种基线（BART‑GEN、MvP、SLGM、STAGE、LLaMA3‑8B、ChatGPT‑4 等）对比，DiffuSent 在 12 个 D_20a 设置、D_20b 的 ASTE 任务以及 D_17/D_19 子任务均达到 SOTA。平均 F1 提升约 0.6 分，多词三元组平均提升 2.48%；在推理速度上相较自回归模型可提升至 181 倍。

**⚠️ 局限性**

局限性：① 需调节扩散步长和噪声比例，过大或过小均会影响性能；② 对极长句子或稀有词汇的泛化尚未充分验证；③ 依赖预训练语言模型，对低资源领域可能表现受限；④ 对边界漂移的微小误差仍可能导致精度下降。

---

## 97. Benchmarking Local LLMs for Natural-Language-to-SQL Querying in Biopharmaceutical Manufacturing: An Empirical Benchmark on Consumer-Grade Hardware

**arXiv ID:** 2606.01338 | [PDF](https://arxiv.org/pdf/2606.01338v1)

**作者:** Sagar Bhetwal `[一作]`, Gaurav Kumar Gupta `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在消费者级硬件上评估四个7B–8B参数本地LLM（Qwen 2.5 Coder 7B、Llama 3.1 8B、Mistral 7B、Meditron 7B）在制药制造数据库的自然语言查询到T‑SQL生成任务中的性能，并构建了PharmaBatchDB AI平台和合成数据集进行基准测试。

**💡 创新点**

首次在制药制造域对本地LLM进行结构化查询评估，揭示医学预训练模型在此任务中不如代码调优模型；同时提出GxP对齐的完整本地NLQ管线实现方案。

**🔧 技术方法**

采用Ollama推理引擎、FastAPI、SQL AST校验、ROUGE‑L、Factual Consistency、SQL Compliance、Hallucination Rate、TPS等指标，对四个开源LLM进行零样本评估。

**📊 数据集**

使用约6.3万行、七张表的合成MS SQL Server数据库PharmaBatchDB，提供按模块（Batch、MES、CIP）和难度（易/中/难）划分的60道NLQ–SQL对。

**📈 对比分析**

通过SQL提取率、ROUGE‑L、Factual Consistency、SQL合规率和幻觉率等指标比较，Llama 3.1与Qwen在合规率≈93%和88%、FC≈0.30–0.34上表现相当，Mistral在复杂查询上显著下滑，Meditron基本失败；差异在统计学上显著但Llama–Qwen差距不显著。

**⚠️ 局限性**

主要限制包括样本量仅60道题、单一合成数据库与单一作者SQL参考导致偏差、CPU主导的低吞吐量、缺乏真实制药数据验证以及未与云端模型进行对比。

---

## 98. Feature Alignment Determines Fusion Strategy: A Comparative Study of Cross-Attention and Concatenation in Multimodal Learning

**arXiv ID:** 2606.01207 | [PDF](https://arxiv.org/pdf/2606.01207v1)

**作者:** Zhiqiang Zhou `[一作]` (Hunan Chemical Industry Vocational and Technical College), Xuezhen Xie `[通讯]` (Hunan Chemical Industry Vocational and Technical College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究跨注意力与拼接在多模态融合中的优劣，提出特征对齐质量决定优先选用拼接。

**💡 创新点**

提出特征对齐假设、样本复杂度理论解释拼接优势，并给出决策框架。

**🔧 技术方法**

使用拼接、跨注意力融合、样本复杂度分析、噪声对齐降解实验、t‑SNE可视化、计算效率评估等技术。

**📊 数据集**

使用Flickr8k图像‑文本匹配数据集进行二分类实验。

**📈 对比分析**

在多尺度（2048–16384样本）下，用CLIP ViT‑B/32预对齐特征时，拼接平均提升4.1–5.1%准确率；对ResNet18未对齐特征时，跨注意力更优；噪声实验显示对齐退化时拼接优势增大。

**⚠️ 局限性**

仅在Flickr8k单任务，未验证更大数据集或多任务；仅做特征级融合，未考虑端到端训练；噪声降解模型简化；样本复杂度使用简化PAC界限，需更精细理论。

---

## 99. Decision-Focused On-Policy Learning for Contextual Linear Optimization with Partial Feedback

**arXiv ID:** 2606.01081 | [PDF](https://arxiv.org/pdf/2606.01081v1)

**作者:** Wyame Benslimane `[一作]` (University of California Berkeley), Paul Grigas `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种在线情境线性优化的决策聚焦混合梯度算法（DFHPG），在部分反馈下通过学习成本分布并结合预测-优化策略来更新决策模型。

**💡 创新点**

创新点在于将无偏score函数梯度与基于插件的决策聚焦梯度融合成混合梯度，利用最小二乘nuisance估计控制插件偏差，并在非凸环境下证明了与标准SGD相当的收敛速度。

**🔧 技术方法**

采用了决策聚焦学习、score函数估计、插件梯度、条件生成模型（高斯、CNF、DDPM）、线性优化oracle、在线凸优化与方差降低等技术。

**📊 数据集**

使用了四个基准数据集：合成的top‑k选择、最短路径、组合定价任务，以及真实能源调度数据（SEMO）。

**📈 对比分析**

与传统上下文Bandit方法（GreedyCB、ϵ‑GreedyCB、TSCB）比较，DFHPG在纯bandit和半bandit两种反馈模式下均获得最低累计遗憾，尤其在合成任务上明显优于所有基线，能量调度任务中相比基线降低约1.7–1.9倍。

**⚠️ 局限性**

局限性包括仅在线性目标下提供收敛保证；插件梯度对nuisance估计的准确性高度依赖；在真实数据中score函数信号噪声大，混合策略的超参数调节相对敏感；未扩展到非线性目标或非完美oracle的情形。

---

## 100. Target localization, identification and sensing using latent symmetries

**arXiv ID:** 2606.01421 | [PDF](https://arxiv.org/pdf/2606.01421v1)

**作者:** David Dukov `[一作]` (University of Warwick), Bryn Davies `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了三维散射体阵列，使其具有潜在（隐藏）对称性，并通过电容矩阵模型研究加入“入侵者”后对对称性的破坏，进而构建了能够定位入侵者的传感方法。

**💡 创新点**

创新点在于首次将潜在对称性用于传感问题，并首次在不可稀疏化的三维开放系统中观察到此对称性，证明其可用于实现精准定位。

**🔧 技术方法**

所用技术包括电容矩阵的稀薄近似、隐性对称性与等谱归约、字典匹配、贝叶斯推断、全连接多层感知机（MLP）以及高斯过程回归（GPR）。

**📊 数据集**

数据集为多种自定义散射体配置的仿真数据，包括线性、马蹄形、非对称和稀疏排列，入侵者半径和位置均在预设区间内随机采样。

**📈 对比分析**

通过与字典匹配和传统方法比较，MLP和贝叶斯模型在含噪测量下实现了更高的定位精度（误差在几毫米级别），且GPR能够给出可信区间。

**⚠️ 局限性**

局限性包括：对入侵者远离或靠近散射体时灵敏度下降；多值映射和不良条件导致逆问题在某些区域不稳定；实验仅基于仿真，缺乏真实测量验证。

---

## 101. ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Planning

**arXiv ID:** 2606.01205 | [PDF](https://arxiv.org/pdf/2606.01205v1)

**作者:** Xuchen Liu `[一作]` (Pengcheng Laboratory), Jiankun Yang `[通讯]` (Pengcheng Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于视觉语言指令的无人机导航框架，先通过视频扩散模型“想象”未来观测，然后提取相对6‑DoF运动，最后用动力学规划生成可执行轨迹。

**💡 创新点**

将指令条件的生成式世界模型与基于视觉里程计的动作提取器以及动力学规划器结合，突破传统VLA模型在几何一致性和动力学匹配上的脆弱性，实现从自然语言到可执行姿态的闭环链路。

**🔧 技术方法**

隐式视频扩散模型、学习型视觉里程计动作提取器、动力学规划（Kinodynamic Planner）、step‑distilled inference 以及低延迟航模执行栈。

**📊 数据集**

UAV‑Flow 视觉语言导航基准及真实飞行环境中采集的指令‑观测对。

**📈 对比分析**

与现有 VLN 与 VLA 基线对比，在 UAV‑Flow 上取得 70.9% 的成功率，显著优于前人；在实测飞行中表现出安全、可执行的轨迹，验证了框架的实用性。

**⚠️ 局限性**

生成视频可能出现空间时间不一致导致提取误差；对高动态环境的鲁棒性有限；需显著计算资源（1.3B 参数）且依赖预训练模型的推理成本。

---

## 102. Local MixVR: Breaking the Communication-Sample Dependence in Distributed Learning

**arXiv ID:** 2606.01128 | [PDF](https://arxiv.org/pdf/2606.01128v1)

**作者:** Tehila Dahan `[一作]` (Technion), Kfir Y. Levy `[通讯]` (Technion)

**通讯引用:** 1237 | [OpenAlex ID](https://openalex.org/A5022424856)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Local MixVR的分布式训练框架，在每个通信周期内结合本地双动量、混合本地-小批量更新和漂移校正，从而显著降低通信轮次。

**💡 创新点**

创新点在于首次将多种方差缩减方法（双动量、混合本地小批量、漂移校正）整合到本地更新中，并实现了通信复杂度与样本量无关，仅依赖于工作节点数。

**🔧 技术方法**

使用了本地双动量（Anytime-averaged + STORM）、混合本地-小批量策略和漂移校正机制，并配合分布式同步与梯度累积。

**📊 数据集**

在MNIST和CIFAR-10上验证了算法效果，使用PyTorch在NVIDIA B200 GPU集群进行实验。

**📈 对比分析**

与Local SGD、Local Momentum、Minibatch SGD及Minibatch ASGD等基线比较，Local MixVR在相同通信轮次数下取得更高的测试准确率，尤其在M≤N^{1/4}的常见场景下显著优于Minibatch ASGD。

**⚠️ 局限性**

局限性包括目前仅针对凸光滑目标的理论分析，缺乏非凸或深度网络的最优证明，并且对大规模异构环境的鲁棒性尚未深入研究。

---

## 103. pcbGPT: Automatic PCB Schematic Synthesis from Natural Language Requirements

**arXiv ID:** 2606.01188 | [PDF](https://arxiv.org/pdf/2606.01188v1)

**作者:** Tobias King `[一作]` (Karlsruhe Institute of Technology), Tobias Röddiger `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 382 | [OpenAlex ID](https://openalex.org/A5069745172)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个从自然语言硬件需求生成可编辑 KiCad 原理图的系统，支持交互式 Web 工作流。

**💡 创新点**

创新点在于引入 Python 语言嵌入式电路 DSL、基于数据表的元件知识抽取、组件库检索、执行检验与语义校验的多阶段流水线，以及对比自动与人工评审的严格验证。

**🔧 技术方法**

使用大规模语言模型（GPT‑5、Qwen3.5）、自定义组件搜索与信息工具、数据表图像提取与模型摘要、结构化 DSL 生成、验证代理、以及 Web 前端交互。

**📊 数据集**

采用 20 个嵌入式/物联网/可穿戴任务集合，每个任务包含自然语言描述、关键元件列表、必需接口以及参考实现。

**📈 对比分析**

通过 pass@1、pass@5 以及平均相似度评估，GPT‑5.3‑Codex 在所有任务中取得总体 pass@1 0.45、pass@5 0.71，性能优于其他模型；与人工评审的一致率高，Kappa 统计显著，表明系统在易中难任务上已具备可用性。

**⚠️ 局限性**

仍需人工复核，主要错误集中在支持电路、数值、接口和局部拓扑；系统仅生成原始草图，未覆盖布局、布线、制造优化和完整电气验证，且高度依赖数据表和 KiCad 库的可用性与准确性。

---

## 104. Peacemaker at ATE-IT: Automatic term extraction from Italian text for waste management data using encoder model

**arXiv ID:** 2606.01469 | [PDF](https://arxiv.org/pdf/2606.01469v1)

**作者:** Mahdi Bakhtiyarzadeh `[一作]` (University of Tabriz), Jafar Razmara `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在 ATE-IT 共享任务中实现了一个轻量级的词条抽取系统，将词条抽取转化为序列标注任务。

**💡 创新点**

创新点在于自定义的 BIO 对齐管道与对 Italian BERT 的微调，兼顾低资源环境的可解释性和稳定性。

**🔧 技术方法**

采用了 dbmdz/bert-base-italian-cased Transformer、BIO 标签、微调分类头和 token‑level 对齐算法。

**📊 数据集**

使用了 ATE‑IT 任务提供的意大利废物管理领域语料库（训练2308句，验证577句，测试1142句）。

**📈 对比分析**

与基线（gemini‑2.5‑flash 零样本）和其他 8 支队伍的结果对比，系统在 type‑level 与 micro‑level 上均排名第 7，表现稳健但略低于最先进模型。

**⚠️ 局限性**

局限在于资源受限，未能使用更大、更专业的模型，导致召回率和整体 F1 分数相对较低。

---

## 105. Towards Interactive Video World Modeling: Frontiers, Challenges, Benchmarks, and Future Trends

**arXiv ID:** 2606.01164 | [PDF](https://arxiv.org/pdf/2606.01164v1)

**作者:** Jiuming Liu `[一作]` (University of Cambridge), Per Ola Kristensson `[通讯]` (University of Cambridge)

**通讯引用:** 6950 | [OpenAlex ID](https://openalex.org/A5042452579)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了互动世界建模领域的研究进展、技术瓶颈和评测基准。

**💡 创新点**

首次系统化整理了从应用场景到技术挑战的完整框架，并提出了未来研究方向。

**🔧 技术方法**

主要涉及动作条件可控性、长时交互与记忆机制、实时响应与分布式推理等技术。

**📊 数据集**

参考了多领域数据集，如 TC-Bench、EvalCrafter、WorldScore、OmniWorldBench、iWorldBench、nuScenes、DriveGAN 等。

**📈 对比分析**

通过对比四类场景（开放世界、游戏引擎、机器人/嵌入式AI、自动驾驶）的指标，展示了现有方法在可控性、连贯性、质量等方面的优势与不足。

**⚠️ 局限性**

主要局限在缺乏大规模动作标注数据、因果推理能力不足、长时序一致性难保、物理感知缺失及交互界面不够直观。

---

## 106. HakushoBench: A Japanese Chart and Table VQA Benchmark from Governmental White Papers

**arXiv ID:** 2606.01132 | [PDF](https://arxiv.org/pdf/2606.01132v1)

**作者:** Issa Sugiura `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3607 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 HakushoBench 的日语图表与表格视觉问答（VQA）基准，包含 2,053 张图像（10 种类型）和人工标注的 2,053 对问答，强调全局、多跳、计数、外部知识等难度维度。

**💡 创新点**

创新点在于：① 使用政府白皮书（Hakusho）作为高质量、视觉多样、信息密集的真实图表与表格来源；② 通过多阶段手工标注与验证确保问题难度和答案唯一性；③ 将难度维度细化为多种类型，促使模型进行更深层次的推理与跨图像信息整合。

**🔧 技术方法**

主要技术包括：基于手工标注的 QA 对生成与双人验证；使用 LLM（如 GPT‑5.1）进行自动答案判定；利用 SigLIP2 视觉编码器评估视觉多样性；在实验中采用多种开源与闭源 VLM（Qwen3‑VL、InternVL、Sarashina2.2、LLM‑jp‑4‑VL、GPT‑4o、GPT‑5.1、Gemini 3 Pro）进行对比评测，并试验 Direct 与 Chain‑of‑Thought（CoT）提示策略。

**📊 数据集**

数据集来源：33 份日本政府年度白皮书的 HTML 版（含 5,903 张候选图表/表格，最终筛选 2,053 张有效图像）。

**📈 对比分析**

实验结果显示：开源模型最高仅达 58.6%（Qwen3‑VL‑8B），而闭源模型 Gemini 3 Pro 达到 93.5%，两者差距 34.9 分；在 CoT 提示下，多数模型提升 10+ 分；与现有日语基准 JGraphQA 相比，HakushoBench 更具挑战性，开源模型在其上相对表现更差。

**⚠️ 局限性**

局限性：① 仅覆盖日语且仅来自政府白皮书，可能不代表所有真实文档的视觉风格与领域；② 虽减少重复，但仍可能存在预训练数据污染；③ 由于高性能模型已达 93.5% 以上，评测空间有限，未来需要更高难度子集或跨语言扩展。

---

## 107. Beam-focusing Analysis for Modular XL-arrays: Effect of Time Synchronization Errors

**arXiv ID:** 2606.01096 | [PDF](https://arxiv.org/pdf/2606.01096v1)

**作者:** Mingjiang Wu `[一作]` (Southern University of Science and Technology), Xianfu Lei `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 9347 | [OpenAlex ID](https://openalex.org/A5028336223)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了模块化XL-阵列在近场通信中时钟同步误差对波束聚焦性能的影响

**💡 创新点**

首次证明即使存在时钟误差，双子阵列仍能保持波束聚焦，且角度偏移被上界1/M限制；多子阵列出现波束分裂，形成嵌套波束模式

**🔧 技术方法**

采用最大比传输（MRT）波束成形、Fresnel近似、数值仿真与解析推导

**📊 数据集**

无真实数据集，采用基于参数的仿真模型（多用户单天线、不同子阵列数与子阵列尺寸）

**📈 对比分析**

通过仿真比较不同同步误差范围下的总速率，发现误差增大导致总速率显著下降，尤其在子阵列数增多时更为明显

**⚠️ 局限性**

局限性包括：需要极高精度的时钟同步（ps级别），分析在多子阵列情况下仅为数值结果，且仅考虑理想LoS近场信道，实际环境中的多径与硬件非理想性未被纳入

---

## 108. Reasoning4Sciences: Bridging Reasoning Language Models to All Scientific Branches

**arXiv ID:** 2606.01145 | [PDF](https://arxiv.org/pdf/2606.01145v1)

**作者:** Teddy Ferdinan `[一作]`, Przemysław Kazienko `[通讯]` (Wrocław University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对欧盟研究委员会（ERC）划分的28个学科中Reasoning Language Models（RLMs）的采用与成熟度进行了系统性综述与量化评估；

**💡 创新点**

首次提出基于资源数量的MD/ME成熟度框架，揭示不同学科在模型、方法、数据集和基准测试方面的巨大差异；

**🔧 技术方法**

采用文献综述、分层分类、统计计数与自定义成熟度公式（MD = C_methods + C_datasets/2；ME = C_models + C_benchmarks/2）进行量化分析；

**📊 数据集**

主要使用学术数据库（Google Scholar、Scopus、Web of Science、arXiv、bioRxiv）检索的论文与公开资源，未使用传统实验数据集；

**📈 对比分析**

通过计算MD/ME分数对比，发现物理科学与工程类学科处于高成熟度，社会科学与人文类低成熟度，且公开资源更为匮乏；

**⚠️ 局限性**

局限在于依赖已有文献的可检索性，公开资源缺乏导致真实可用性受限，缺乏跨学科整合与实证性能评估。

---

## 109. Soft-NBCE: Entropy-Weighted Chunk Fusion for Long-Context

**arXiv ID:** 2606.01101 | [PDF](https://arxiv.org/pdf/2606.01101v1)

**作者:** Shihao Ji `[一作]`, Zihui Song `[通讯]` (Chunjiang Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Soft‑NBCE，通过软熵加权融合块级分布替代 NBCE 的硬块选择，并引入 Consistency Distillation 对齐全上下文教师。

**💡 创新点**

软熵权重的温度可调融合、对数空间几何平均聚合、以及无监督 LoRA 自蒸馏以补偿块独立假设。

**🔧 技术方法**

温度 Softmax 熵加权、对数空间加权聚合、Top‑p 截断、对比惩罚 β、LoRA 低秩适配、KL 蒸馏、LLaMA‑3‑8B 基础模型。

**📊 数据集**

LongBench（MuSiQue、HotpotQA、GovReport）、NIAH‑32K 检索、wikitext‑103 用于蒸馏、少样本提示。

**📈 对比分析**

与 Truncated、PCW、Vanilla NBCE、YaRN 等基线对比，在 O(L²/n) 内存下，Soft‑NBCE 在多跳推理上比基线提升 0.035‑0.052 F1，NIAH‑32K 检索 0.909 远超 Truncated 0.659。

**⚠️ 局限性**

延迟随块数线性增长，适合内存受限而非速度需求；在需要整体文档理解的摘要任务中表现不如全上下文；熵相关性假设经验性且 β、τ 固定，缺乏自适应路由。

---

## 110. AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise

**arXiv ID:** 2606.01053 | [PDF](https://arxiv.org/pdf/2606.01053v1)

**作者:** Bowen Tian `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yutao Yue `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AnyEdit++ 框架，通过 Bayesian Surprise 机制（Bayes-Chunk）实现对长文本的自适应分块，从而提升大语言模型的长篇知识编辑能力。

**💡 创新点**

创新点在于：①将贝叶斯惊奇引入分块决策，自动捕捉语义边界；②从理论上证明结构独立性与因果局部性，阐释分块与编辑性能的关系；③保持 AnyEdit 的自回归编辑流程，只改进分块方式，兼容性强。

**🔧 技术方法**

使用技术包括：自回归编辑框架（AnyEdit）、贝叶斯惊奇计算（Surprisal/KL）、梯度可控性分析（垂直 vs 水平传播）、线性最小二乘权重更新（MEMIT 风格），以及针对多模型的插件式集成。

**📊 数据集**

实验数据集涵盖：EditEverything（多任务长篇编辑）、UnKE、CounterFact（事实编辑）、QwQ-Edit（长篇 CoT 数学/代码）等。

**📈 对比分析**

与 AnyEdit、MEMIT、AlphaEdit 等基线在多模型（Llama‑3.1、Llama‑2、Qwen‑2.5）和多任务（数学、代码、物理等）上比较，BLEU 与 BERTScore 均提升约 2–8% 甚至更高，尤其在数学与代码任务中显著领先。

**⚠️ 局限性**

局限性：仍需预训练 LLM 计算惊奇值，分块可能在信息稀疏区误判；单一扰动难以覆盖极长篇或极其复杂的逻辑；缺乏针对更大规模长文本的实验验证。

---

## 111. Modulation-Reaction Networks

**arXiv ID:** 2606.01193 | [PDF](https://arxiv.org/pdf/2606.01193v1)

**作者:** Leo Lobski `[一作]` (University College London), Yoàv Montacute `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种新的数学框架——调制-反应网络（MR-networks），并定义了其同步布尔语义；同时构建了一种混合模态 μ-演算（MRL）用于形式化和推理该网络。

**💡 创新点**

首次将调制（激活/抑制）作为网络核心对象，统一了化学反应网络与布尔网络的视角；引入 MRL 作为能够同时捕捉网络结构与时间演化的逻辑语言；提供了完整的一步更新规则和模型检验方法，并证明了 MR-networks 的 bisimulation 与 MRL 公式的不变性。

**🔧 技术方法**

使用布尔网络理论、模态 μ-演算、固定点运算、评估游戏（evaluation game）以及 bisimulation 关系来分析和验证 MR-networks。

**📊 数据集**

未使用实验数据集；该工作主要为理论建模与形式化推理，未进行数据驱动实验。

**📈 对比分析**

未开展实验比较；论文通过形式化证明展示了 MRL 的表达能力（如可达性、持续产生、吸引子等生物学属性）以及模型检验的可行性，未给出定量性能指标。

**⚠️ 局限性**

局限性包括：1) 异步语义仅作草图，尚未完整实现与验证；2) 对大规模生物网络的计算复杂度未评估；3) 缺乏与现有生物建模工具的集成与实证验证。

---

## 112. TukaBench: A Culturally Grounded Jailbreak Benchmark for African Languages

**arXiv ID:** 2606.01322 | [PDF](https://arxiv.org/pdf/2606.01322v1)

**作者:** Victor Akinode `[一作]` (Mila Quebec AI Institute), David Ifeoluwa Adelani `[通讯]` (Mila Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文构建了一个多语言 jailbreak 评估基准 TukaBench，涵盖七种非洲语言，并通过翻译、文化适配和代码混合三种构造方式，评估 LLM 在低资源语言中的安全行为。

**💡 创新点**

创新点：①首次提供了非洲语言的人工校正 jailbreak 基准；②引入三类标签（Jailbroken、Refused、Deflected）区分安全合规与理解失败；③系统比较了直接提示、BPJ 攻击与代码混合提示对模型行为的影响。

**🔧 技术方法**

使用的技术包括：机器翻译 + 人工校对 + 质量估计（SSA‑COMET‑QE）；代码混合生成（基于 AfriqueQwen‑8B + 人工校正）；LLM‑as‑a‑judge（GPT‑4.1）以及人工验证；对 13 个闭源与开源 LLM 进行评测。

**📊 数据集**

数据集：TukaBench 共 986 条提示/语言（Amharic、Hausa、Igbo、Chichewa、Swahili、Yoruba、Nyanja），由 JailbreakBench (JBB) 翻译/文化适配及 African‑authored (AfriJail‑Mono) 生成，并包含代码混合版本。

**📈 对比分析**

比较方法：直接提示、Boundary Point Jailbreaking (BPJ) 与代码混合提示；评估指标为 Attack Success Rate (ASR)、Deflection Rate 与 Refusal Rate。结果显示：①非洲语言提示往往降低 Refusal、提高 Deflection；②文化适配提示使 ASR 与 Deflection 同时升高；③BPJ 能提升 ASR，但未消除 Deflection；④代码混合提示显著减少 Deflection，但 ASR 变化不一。模型生成层次提升（如 GPT‑5.2）可部分缓解这些问题。

**⚠️ 局限性**

局限性：仅覆盖七种非洲语言，未涉及极低资源语言；评测仅限可招募翻译/校对的语言；模型样本有限，部分结果因模型停用缺失；LLM‑as‑a‑judge 在低资源/非拉丁脚本语言上的可靠性下降。

---

## 113. Magnum.np.distributed: Accelerating Finite Difference Micromagnetic Simulations with Multiple GPUs

**arXiv ID:** 2606.01114 | [PDF](https://arxiv.org/pdf/2606.01114v1)

**作者:** Tsz Chung Cheng `[一作]` (Kyushu University), Hiromi Yuasa `[通讯]` (Kyushu University)

**通讯引用:** 707 | [OpenAlex ID](https://openalex.org/A5058817201)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了基于PyTorch Distributed的Python原生多GPU磁微观模拟框架，扩展了magnum.np实现分布式LLG求解、交换与DMI等场的半壁通信与全局FFT转置；

**💡 创新点**

创新点在于：①使用Python+PyTorch完成全流程无C++/CUDA编译的分布式模拟；②通过PyTorch Distributed实现多GPU间的滑动切片、halo交换与全局FFT转置，保持高度可移植性；③实现了与单GPU、CPU（NUMA）以及其他主流工具（Mumax3、OOMMF）的无缝对比；

**🔧 技术方法**

使用技术包括PyTorch、TorchDynamo+TorchInductor、PyTorch Distributed（NCCL/MPI）、CUDA（仅在后端）、FFT、MPI、NUMA优化、Python接口；

**📊 数据集**

使用数据集包括muMAG标准问题4（Permalloy）、DMI与域壁钉扎测试、以及512×512×90（4×4×0.7 nm）Pt/Gd/Co/Ni多层膜模拟（约2.36千万单元）；

**📈 对比分析**

方法上通过对比单GPU（H100 HBM3/HBM2e）、多GPU（8×H100 HBM3/4×HBM2e）、CPU（Xeon 60核NUMA）以及Mumax3的时间、精度、能量收敛结果，发现分布式实现与单GPU精度一致；在消磁场计算上8个GPU可达7.0×线性加速；在交换/DM场的halo交换上仅在>10⁶单元时显著加速；CPU NUMA pinning可提升6.8×，比单GPU慢10‑15×；

**⚠️ 局限性**

局限性包括：①halo交换与全局FFT导致通信瓶颈，尤其在跨节点时受InfiniBand带宽限制；②对低计算密度的场（如Heisenberg交换、DMI）加速受限；③受PyTorch对复数不支持的JIT限制，无法将多核GPU内核融合；④当前实现对大规模内存（>4×10⁷单元）仍需多GPU；

---

## 114. Reducing Token Usage of State-in-Context Agents using Minification

**arXiv ID:** 2606.01326 | [PDF](https://arxiv.org/pdf/2606.01326v1)

**作者:** Nicolas Hrubec `[一作]` (TU Wien), Jürgen Cito `[通讯]` (TU Wien)

**通讯引用:** 1716 | [OpenAlex ID](https://openalex.org/A5033732305)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现并扩展了 DirectSolve 状态内上下文代理，探究并实现了多种轻量级源代码最小化技术，以降低软件工程任务中的 token 消耗。

**💡 创新点**

提出将代码最小化视为显式上下文压缩手段，并系统评估了多种代码变换（删除注释、空行、缩进、重命名等）在保持语义完整的同时减少 token 数量的可行性与性能折衷。

**🔧 技术方法**

实现了基于 GPT-5-mini 和 GPT-4.1 的 state‑in‑context 代理，集成了自定义代码最小化插件，并使用 OpenAI GPT‑4 tokenizer 计数 token、SWE‑bench 评估框架来衡量修复质量和成本。

**📊 数据集**

使用 SWE‑bench Verified 公开基准（12 个 Python 开源项目的真实 GitHub issue 与测试集），并在 100 例子子集上进行 ablation。

**📈 对比分析**

通过在完整基准上对比未最小化与最小化两种配置，发现平均输入 token 减少约 42%，但通过率从 50% 降至 38%；单独变换与多重变换 ablation 表明单一变换可节省 10‑17% 的性能，组合变换仍能保持相对较高的通过率并显著降低成本。

**⚠️ 局限性**

主要限制包括：模型随机性导致的结果波动、实验仅针对以代码为主的代理不一定普适、修复补丁的语义正确性未进一步验证、缩进最小化导致的语法错误未完全解决，以及在补丁生成后缺乏完整的格式恢复。

---

## 115. Fine-Tuning Diffusion Models for Molecular Generation via Reinforcement Learning and Fast Sampling

**arXiv ID:** 2606.01220 | [PDF](https://arxiv.org/pdf/2606.01220v1)

**作者:** Guang Lin `[一作]` (Shanghai Jiao Tong University), Lei Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 54190 | [OpenAlex ID](https://openalex.org/A5101613292)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种针对结构约束的扩散模型的强化学习微调框架FTDiff，能够在保持分子合法性的同时优化多目标药物性质。

**💡 创新点**

采用GRPO风格的策略优化、时间无关的快速采样以及阈值感知奖励，解决了传统扩散模型在多目标和采样速度上的瓶颈。

**🔧 技术方法**

强化学习（GRPO、双头剪切损失）、时间无关扩散模型、阈值感知奖励函数、快速采样（10步）等。

**📊 数据集**

CrossDocked2020蛋白-配体对数据集。

**📈 对比分析**

与AR、Pocket2Mol、MolCRAFT、TargetDiff、DecompDiff、IPDiff、TAGMol、ALIDiff等基线比较，在多目标设置下平均Vina Score -7.18、Vina Min -8.48、Vina Dock -9.44、High Affinity 78.6%，在单目标上也优于KGDiff，显著提升结合亲和力和多目标满足率。

**⚠️ 局限性**

仍需改进多目标权重的动态平衡、对极端蛋白口袋的泛化能力有限，未探索更复杂属性组合的协同优化。

---

## 116. PALTO: Physics-Informed Active Learning for Tri-Gate FinFET Design Optimization for Vertical Power Delivery

**arXiv ID:** 2606.01265 | [PDF](https://arxiv.org/pdf/2606.01265v1)

**作者:** Ayoub Sadeghi `[一作]` (University of Illinois Chicago), Inna Partin-Vaisband `[通讯]` (University of Illinois Chicago)

**通讯引用:** 561 | [OpenAlex ID](https://openalex.org/A5027004922)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过物理信息化主动学习框架 PALTO，优化三门 GaN FinFET 结构以实现垂直电源交付的低 R_on 高 I_DS 设计。

**💡 创新点**

创新点在于将多任务神经网络与深度集成模型结合使用主动学习，显著降低 TCAD 计算量，同时利用 SHAP 解释模型决策，揭示关键设计参数对性能的影响。

**🔧 技术方法**

使用的技术包括多任务全连接神经网络、深度集成（10 个网络）、query‑by‑committee 主动学习、SHAP 解释、Latin Hypercube Sampling（LHS）生成初始数据、Sentaurus TCAD 进行高保真仿真。

**📊 数据集**

数据集为约 10³ 个 LHS 采样的 TCAD 仿真结果，随后通过主动学习循环生成并评估约 10⁴ 次仿真，覆盖 6×10⁹ 设计点。

**📈 对比分析**

与 NSGA‑II、随机搜索和工业基准比较显示，PALTO 在相同或更优性能下将 TCAD 计算量降低 3.2 倍，得到的设备 D1 在 300-fin 配置下实现 R_on 0.49 Ω、I_DS 3.3 A，性能比基准提升 2 倍，功率 FoM 低至 5 pC·Ω。

**⚠️ 局限性**

局限性包括对 TCAD 模型精度的依赖、主动学习循环仍需昂贵的高保真仿真、仅针对已定义的设计空间，且在其他工艺或材料体系中的可推广性尚待验证。

---

## 117. Structure and Scale in Simplicial Sequence Modelling

**arXiv ID:** 2606.01302 | [PDF](https://arxiv.org/pdf/2606.01302v1)

**作者:** Matthew Farrugia-Roberts `[一作]` `[通讯]` (University of Oxford), Matthew Farrugia-Roberts (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在隐藏马尔可夫模型（HMM）生成的序列上训练小型Transformer时，随着训练步数（计算量）的增大，模型内部对贝叶斯信念分布的线性编码会逐渐精炼，并与预测性能的提升呈现相关性。

**💡 创新点**

首次将可解释内部结构的改进与行为尺度法则联系起来，提出内部计算结构的规模变化是驱动性能提升的根本机制，并用 toy 模型解释了非单调的表示误差变化。

**🔧 技术方法**

使用了四层残差 Transformer（单头注意力宽度 8，MLP 宽度 256），JAX 训练框架，线性探针评估最终层激活对贝叶斯信念的编码误差，以及交叉熵损失监控。

**📊 数据集**

采用从“mess3”三状态 HMM 随机生成的长度为 10 的观测序列，生成训练和测试批次。

**📈 对比分析**

通过监测超额预测交叉熵损失和线性探针误差随训练步数的变化，发现两者均按近似幂律改善，且在 1 亿步时出现非单调性并趋于平台化，表明性能和表示在一定范围内持续提升。

**⚠️ 局限性**

仅在固定模型规模和单一 HMM 任务上进行实验，未探究不同参数数量或层数对规模效应的影响；toy 模型仅示例性解释，未完全匹配量化趋势；实验仅展示相关性，未证明因果关系。

---

## 118. DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts

**arXiv ID:** 2606.01062 | [PDF](https://arxiv.org/pdf/2606.01062v1)

**作者:** Jiarui Feng `[一作]` (Meta MRS), Yixin Chen `[通讯]` (Washington University in St. Louis)

**通讯引用:** 10236 | [OpenAlex ID](https://openalex.org/A5100393445)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出在Mixture-of-Experts (MoE)中用结构化聚合（DAG）替代传统加权求和，动态学习专家输出的聚合图

**💡 创新点**

通过将专家组合视为有向无环图，显著扩展了可实现的专家组合空间，并实现了单层内的多步推理能力

**🔧 技术方法**

引入DAG学习模块（轻量级图神经网络），在标准MoE路由基础上进行图结构学习；使用Transformer、MoE、动态规划理论等

**📊 数据集**

在Pile等大规模文本语料上进行预训练，随后在Alpaca、Open‑Platypus、MathInstruct等指令数据集上微调，评估在PIQA、ARC‑e、HellaSwag、GPQA、Lambada、MMLU、BBH等基准上表现

**📈 对比分析**

与标准MoE以及共享专家、Chain‑of‑Experts、MLP混合等对比，DAG‑MoE在12B/40B训练中均取得更低perplexity（最大约0.8点提升）并在多步推理任务上提升≈6点准确率，额外开销仅约1–4%

**⚠️ 局限性**

限制包括：DAG结构空间被约束为固定深度/节点数，缺乏对最优图结构的理论保证，且实验规模相对较小，尚未验证在亿级参数/万亿token规模下的可扩展性

---

## 119. How (and when) can you fit examples to logic-based hypothesis classes over infinite structures?

**arXiv ID:** 2606.01107 | [PDF](https://arxiv.org/pdf/2606.01107v1)

**作者:** Michael Benedikt `[一作]` (University of Oxford), Alessio Mansutti `[通讯]` (IMDEA Software Institute)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5037329416)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究逻辑可定义的假设类在无限结构（如实数有序域和Presburger算术）上的拟合（训练）问题，探讨其计算与描述复杂度，并给出通过有限样本上的一阶查询求解拟合的可行性；对实值函数类与随机化类进行了可决性与可计算性分析；还讨论了VC维度、自动结构、可定向性等模型理论性质与拟合复杂度的关系。

**💡 创新点**

①首次将一阶查询评估与拟合问题相结合，证明在拥有“限制量化坍塌”性质的结构中，任意可定义的概念类可用一阶句子在多项式时间内判定是否可拟合；②把分段可定义函数与随机化类的拟合问题化简为固定维度的存在式公式，从而在实数有序域、Presburger算术等结构上获得可决性与复杂度上界；③揭示逻辑可定义类的VC维度与拟合可计算性之间的新的关联。

**🔧 技术方法**

模型理论技术（量化消除、UDTFS、VC维度与独立性性质），一阶逻辑查询评估，线性代数与凸几何（Carathéodory定理、ellipsoid法），以及存在式实数理论（∃）的决策技术。

**📊 数据集**

本文为理论研究，不使用具体数据集；所有实验/评估均在理论模型与抽象样本上进行。

**📈 对比分析**

方法对比：在具备限制量化坍塌的结构上，拟合问题可归约为一阶查询，复杂度为多项式；对分段函数类，拟合问题可归约为固定维度存在式，属于∃或更低复杂度；对随机化类，在VC维有限且0-拟合可决的前提下，拟合可在多项式时间内完成；与传统的数值优化/神经网络拟合相比，理论上提供了更严格的可决性与复杂度上界。

**⚠️ 局限性**

局限性：①对非分段的可定义函数（如平方根）拟合仍只能上界为∃，尚无更优复杂度；②对自动结构在非限制量化坍塌情况下的拟合仍未完全可决；③未给出实际参数求解方法，只给出判定问题；④对其它损失函数（除L1之外）的理论分析有限；⑤部分结果依赖于未解决的模型理论扩张问题（是否每个结构可扩张为具有NIP且有限制量化坍塌）。

---

## 120. From Performance to Viability: A Bootstrap Framework for Latent-Space Representation Learning in Adaptive Biological Systems

**arXiv ID:** 2606.01374 | [PDF](https://arxiv.org/pdf/2606.01374v1)

**作者:** Jacques Raynal `[一作]` (University of Montpellier), Jacques Margerit `[通讯]` (University of Montpellier)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个“bootstrap”方法框架，用于在观察到可观测性能不足时，逐步构建并迭代升级多级潜在空间表征，从可观测表现 → 动态组织 → 潜在组织 → 长期可行性 → 预测近似。

**💡 创新点**

创新点在于把理论的形成视为由观察到的解释不足驱动的逐层推进，而非预先设定完整模型；提出了五个连续的表征层次，并用三项步步递进的步态-咬合研究示例阐释这一进化过程。

**🔧 技术方法**

技术方法主要是：bootstrap思想、潜在空间表征学习、可行性（viability）评估以及内部预测近似的概念化；并未引入新的神经网络架构或优化算法。

**📊 数据集**

使用了之前完成的三篇步态-咬合研究的数据集（无新实验数据），这些研究已在原文中公开收集的步态测量与约束条件数据。

**📈 对比分析**

该论文并未进行传统意义上的性能比较或量化指标评估；通过概念性案例说明每一层的出现是由前一层解释不足驱动的，展示了方法的逻辑连贯性与适用性，而非数值性能对比。

**⚠️ 局限性**

局限性包括：1）缺乏在新数据集上的实证验证；2）仅提供概念性框架，未给出可复现的算法实现；3）对方法的有效性与泛化性未通过实验或统计检验进行评估。

---

## 121. Med-HEAL: Analyzing and Mitigating Hallucinations in Medical LLMs with Hallucination-Aware In-Context Learning

**arXiv ID:** 2606.01301 | [PDF](https://arxiv.org/pdf/2606.01301v1)

**作者:** Yiming Liao `[一作]` (University of Maryland Baltimore County), Keke Chen `[通讯]` (University of Maryland Baltimore County)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了 Med-HEAL 框架，系统性收集医学 LLM 在 EHR 诊断推理中的真实幻觉案例，并通过自我批判（self‑critique）机制在推理阶段检测并纠正错误。

**💡 创新点**

创新点包括：① 以 EHRNoteQA 为基础的临床真实幻觉数据集，利用 LLM‑as‑Judge 与人工双评判提升标注可靠性；② 结合自我批判与检索增强的情境学习（RA‑ICL），在不更新参数的前提下实现模型自我纠错；③ 在多种开源医学 LLM 上验证该方法的普适性。

**🔧 技术方法**

使用的技术主要有：LLM‑as‑Judge（GPT‑4o）进行答案真实性评估，检索增强的上下文学习（RA‑ICL）基于 GTR‑T5‑Base 编码，链式思维（CoT）对比，Self‑Critique 管道（错误检测、检索示例、答案再生成与自我判断）。

**📊 数据集**

核心数据集为 Med‑HEAL：从 BioMistral‑7B 在 EHRNoteQA（基于 MIMIC‑IV 出院摘要的 962 个开放式问题）上生成的答案构成；同时包含人类评审标注和 GPT‑4o 的错误类型注解。

**📈 对比分析**

通过与零样本基线、正检索、负检索、对照检索、Chain‑of‑Thought（CoT）等多种提示策略对比，实验显示自我批判+RA‑ICL 在五个模型上平均提升 1%–9%（如 DeepSeek 从 76.9% 提升到 85.9%），部分提升在统计上显著（p<0.05）。

**⚠️ 局限性**

局限性包括：① 仅以 BioMistral‑7B 生成幻觉，导致幻觉模式覆盖有限；② 评估仍依赖 GPT‑4o 判断，尽管经过校准但可能存在误判；③ 对长篇临床文本的推理仍表现不佳，说明需要更强的长上下文处理能力；④ 仅验证了推理阶段的纠错，对模型参数更新与长期适应性的影响尚未探究。

---

## 122. Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing

**arXiv ID:** 2606.01393 | [PDF](https://arxiv.org/pdf/2606.01393v1)

**作者:** Minglai Yang `[一作]` (2077AI), Zexue He `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一个针对专家级文档解析的难度感知基准，涵盖52个BISAC学科、4,514页长篇多语种书籍并提供细粒度布局与结构标注。

**💡 创新点**

创新点在于使用解析器失误驱动的采样策略挑选最具挑战的页面，结合专家领域内容（化学结构、音乐谱、公式、复杂表格）的专门标注，以及长篇跨页解析的评估框架。

**🔧 技术方法**

技术上采用多款先进OCR/ VLM解析器进行采样与基准评估，并使用编辑距离、TEDS、CDM等指标进行文本、表格、公式等组件的量化评价。

**📊 数据集**

数据集来源于大型多语种图书语料库，覆盖52个BISAC主题，包含约70k块级标注，平均每本书约100页。

**📈 对比分析**

通过与多款通用与专用VLM（GPT-5.5、Kimi、Claude、Gemini等）以及专用OCR（MinerU、PaddleOCR）的系统级对比，发现无一模型在所有领域均占优；模型在叙事类文档表现最好，而在参考、设计、医学及音乐等领域性能显著不足。

**⚠️ 局限性**

局限性包括：仅关注文档解析而非完整文档智能任务；未对模型进行专属提示调优；基准仅针对长篇书籍，未覆盖其他文档来源；专业内容的标注形式有限，无法完全覆盖所有领域特定表示。

---

## 123. Bridging Requirements and Architecture: Multi-Agent Orchestration with External Knowledge and Hierarchical Memory

**arXiv ID:** 2606.01385 | [PDF](https://arxiv.org/pdf/2606.01385v1)

**作者:** Ruiyin Li `[一作]` (Wuhan University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 86116 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套名为 MAAD 的多智能体架构设计框架，能够从软件需求规范（SRS）自动生成 4+1 视图 UML、详细的架构文档、并提供 ATAM 评估与不匹配分析报告。

**💡 创新点**

创新点主要包括：① 四个专职智能体（Analyst、Modeler、Designer、Evaluator）协同工作；② 通过检索增强生成（RAG）将行业标准、模式与经验知识注入生成流程；③ 采用三层记忆机制（工作、情景、语义）实现知识积累与跨任务迁移；④ 评估智能体自动完成质量属性评估与矛盾分析，形成闭环迭代。

**🔧 技术方法**

技术手段：大语言模型（GPT‑5.2、Qwen‑3.5、Llama‑3.3、DeepSeek‑R1）+多智能体系统；检索增强生成（RAG）+向量数据库；三层记忆架构；4+1 视图 UML 与 PlantUML；ATAM 评估与结构化指标计算。

**📊 数据集**

数据集：10 篇工业与公开的 SRS 案例（C2C、Case、CCS、CTS、GCS、HCS、LCS、MEM、SFS、SSCS），来自 Jin 等的公开数据集。

**📈 对比分析**

比较方法：与 MetaGPT 基线进行对照；对 10 个案例使用 7 项架构指标（CD、Coh、IC、SC、StC、CCD、SMCC）进行量化评估；并对 6 位行业架构师进行半结构化访谈得到定性反馈。结果表明 MAAD 在指标上整体优于 MetaGPT，生成的架构更模块化、完整、需求可追踪，评估报告自动化显著降低人工验证成本。

**⚠️ 局限性**

局限性：仍需人工干预以处理专业判断和隐藏知识；RAG 对某些指标影响不一致，偶尔会引入更高的复杂度；知识库覆盖有限，缺乏领域特定的高质量训练数据；缺乏统一的行业评估基准，难以全面衡量生成质量。

---

## 124. CA-BED: Conversation-Aware Bayesian Experimental Design

**arXiv ID:** 2606.01182 | [PDF](https://arxiv.org/pdf/2606.01182v1)

**作者:** Daniel Arnould `[一作]`, Shreyas Sunil Kulkarni `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于贝叶斯实验设计的对话规划框架CA-BED，用来在实体推理任务中主动提问并更新信念。

**💡 创新点**

创新点在于将贝叶斯实验设计与LLM概率估计相结合，支持多答案预测和软性证据更新，能够在对话树中递归评估期望信息增益。

**🔧 技术方法**

核心技术包括贝叶斯实验设计、LLM生成候选问题、LLM估计答案概率、贝叶斯更新、对话树look‑ahead和概率信念传播。

**📊 数据集**

使用的评测数据集为20 Questions（111类实体）和Detective Cases（100个谋杀案）。

**📈 对比分析**

与直接提示、UoT等基线比较，CA‑BED在两大基准上均显著提升成功率（平均提升21.8%）并且对话长度仅略增1.8轮；在更严格的Uncertainty of Thoughts（UoT）变体上也保持领先。

**⚠️ 局限性**

局限性包括仅适用于离散、预先定义的假设空间、对LLM概率校准和答案设计高度依赖、计算开销大于直接提示、未在开放域或真实人机交互中验证。

---

## 125. Self-Healing Agentic Orchestrators for Reliable Tool-Augmented Large Language Model Systems

**arXiv ID:** 2606.01416 | [PDF](https://arxiv.org/pdf/2606.01416v1)

**作者:** Rahul Suresh Babu `[一作]`, Adarsh Agrawal `[通讯]` (Stony Brook University)

**通讯引用:** 75 | [OpenAlex ID](https://openalex.org/A5103533240)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种自愈式工具增强LLM代理调度器，该调度器通过监测、检测、根因分类、预算化恢复、验证以及可观测追踪，实现在运行时自动识别并修复各种失败。

**💡 创新点**

创新点在于将可靠性视为受限运行时控制问题，提出可观测、诊断、分类、预算化恢复与验证的一体化循环，并证明了针对性恢复策略显著提升了任务成功率与可诊断性。

**🔧 技术方法**

采用监测–检测–诊断–恢复–验证的控制面架构；使用故障注入基准、可解释的恢复映射策略、验证器与可观测追踪；在实验中模拟工具调用并对策略进行可解释实现。

**📊 数据集**

使用自构造的100个工具增强任务集（分为检索、API工作流、计算、规划、矛盾解决五类）进行基准测试，并在模型-循环验证中使用15-10个简化任务；未使用公开数据集。

**📈 对比分析**

与静态工作流、仅重试、ReAct样式、全重规划四种基线对比，自愈在控制注入基准下实现98.8%的任务成功率，重试94.5%，全重规划93.8%；在预算匹配下仍保持最高成功率；在语义静默失败场景中验证自愈将错误输出率降至0%。

**⚠️ 局限性**

局限性在于实验使用合成任务和模拟工具，未涵盖真实API、网络延迟、权限与成本等；验证器效果仅在受控语义错误下显现；未评估对高风险操作的安全性与人机协同；实验规模与多样性有限。

---

## 126. Test-Time Training for Zero-Resource Dense Retrieval Reranking

**arXiv ID:** 2606.01070 | [PDF](https://arxiv.org/pdf/2606.01070v1)

**作者:** Shiyan Liu `[一作]` (Huazhong University of Science and Technology), Yichen Li `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在零资源环境下的测试时重排框架DART，利用密集检索结果中的伪正负样本在每个查询上微调双线性评分矩阵；

**💡 创新点**

创新点在于在推理时通过自监督的伪标签对评分函数进行快速、轻量级的适配，并结合置信度加权边际损失与跨查询动量机制实现稳健提升；

**🔧 技术方法**

采用双线性评分、置信度加权的边际损失、MetaInit与EMA跨查询动量以及SGD/Lion两种自适应优化器；

**📊 数据集**

在六个BEIR基准数据集（NFCorpus、SCIDOCS、FiQA、ArguAna、TREC‑COVID、SciFact）上进行评估；

**📈 对比分析**

与BGE‑small稠密检索基线相比，平均提升2.1% NDCG@10，且单查询延迟不足10 ms，表现优于多种无监督或训练免费方法，并接近部分监督重排器；

**⚠️ 局限性**

局限在于需要预热阶段选择优化器且对高维编码器的全矩阵更新产生二次计算与内存开销，可考虑低秩参数化以提升可扩展性。

---

## 127. LeAP: Learnable Adaptive Permutation for Feature Selection in Heterogeneous and Sparse Recommender Systems

**arXiv ID:** 2606.01111 | [PDF](https://arxiv.org/pdf/2606.01111v1)

**作者:** Yihong Huang `[一作]` (Bilibili Inc.), Zhihao Li `[通讯]` (Bilibili Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LeAP 模块，将随机特征置换转化为可学习的 O(1) 机制，实现高效、可解释的特征重要性评估与剪枝。

**💡 创新点**

创新点包括：可学习的批量置换门控、基于置换差异的自适应正则化，解决异质维度与极稀疏特征的评估偏差，并在工业规模下实现零业务退化。

**🔧 技术方法**

技术手段：梯度可分离的门控网络、批量置换（Batch‑Shuffle）、指数滑动平均（EMA）估计置换差异、可学习门参数与温度缩放 Sigmoid、O(1) 训练与评估框架。

**📊 数据集**

数据集：四个公共推荐基准（Avazu、Criteo、MovieLens‑1M、AliCCP）以及一大规模工业搜索/CTR 数据集（12k+ 维特征，日均亿级请求）。

**📈 对比分析**

与 Lasso、XGBoost、RandomForest、AutoField、LPFS、SFS、SHARK 等方法对比，LeAP 在所有公开基准上均取得最高 S_AUC；在工业场景下可剔除 30% 以上特征维度且 GAUC 变化 ≤ 1e‑3，显著优于传统置换与基线方法。

**⚠️ 局限性**

限制：仍需在极端高维稀疏场景中细调 EMA 参数以避免正则化过度或欠惩；对动态更新特征（实时生成的 embedding）需要进一步研究持续置换与门控更新机制。

---

## 128. S2M-Trek: From Single to Multi-Sphere Transport via Per-Frame Deep Sets on a Wheel-Legged Robot

**arXiv ID:** 2606.01332 | [PDF](https://arxiv.org/pdf/2606.01332v1)

**作者:** Zong Chen `[一作]` (Huazhong University of Science and Technology), Yiqun Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 1549 | [OpenAlex ID](https://openalex.org/A5100604267)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出Per-Frame Deep Sets方法用于多球运输的动态机动学习

**💡 创新点**

创新点是把每帧集合聚合改为先帧内池化再时间读出，实现对(N!)^H-1的帧级置换不变性并证明其通用性

**🔧 技术方法**

使用了深度强化学习（PPO）+多种集合编码器+DAgger知识蒸馏+16×16布尔触觉图

**📊 数据集**

数据集为仿真生成的四足轮腿机器人与多球运输环境，覆盖1至5球的训练与评估

**📈 对比分析**

通过八种编码器与2×2消融实验比较，Per-Frame Deep Sets在无置换增强的情况下即可在30k迭代内实现5球无掉落，且在五个种子上稳定，性能优于其他编码器

**⚠️ 局限性**

局限在于仅在仿真中验证，物理机器人部署仍待完成，且板尺寸限制导致5球严格成功率下降

---

## 129. STARFISH: faST Accuracy Recovery in pruned networks From Internal State Healing

**arXiv ID:** 2606.01126 | [PDF](https://arxiv.org/pdf/2606.01126v1)

**作者:** Shir Maon `[一作]` (Weizmann Institute of Science), Adi Shamir `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 66827 | [OpenAlex ID](https://openalex.org/A5009126679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种后剪枝恢复方法STARFISH，利用小量无标签校准集对剪枝网络内部表征进行对齐，以提升剪枝后准确率。

**💡 创新点**

创新点在于将内部表征对齐作为恢复目标，独立于剪枝类型、稀疏度、范围和网络架构，并仅需极少的校准样本实现高效恢复。

**🔧 技术方法**

采用余弦相似度表征对齐损失，对密集模型的中间输出进行缓存后在剪枝网络上进行微调；使用梯度下降优化。

**📊 数据集**

主要在ImageNet‑1K数据集上验证，涉及Vision Transformer（DeiT、DeiT‑T、DeiT‑B、DeiT‑3‑H）与MobileNetV1等网络。

**📈 对比分析**

与现有方法（CORP、SNOWS、Magnitude Pruning等）对比，STARFISH在0.5稀疏度下实现了超过30个百分点的准确率提升，在高稀疏度（0.75–0.85）下恢复率可达92%以上，显著优于对手。

**⚠️ 局限性**

局限在于仍需访问原始密集模型权重和剪枝掩码；对非视觉或更大规模多模态模型的适用性尚未验证；未探索在训练阶段直接嵌入对齐目标以预防剪枝损失。

---

## 130. Dual-Route Top-K Retrieval with 1v1 VLM Reranking for the CoVR-R

**arXiv ID:** 2606.01097 | [PDF](https://arxiv.org/pdf/2606.01097v1)

**作者:** Yuyang Sun `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**通讯引用:** 473562 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种零训练的双路线（文本+视觉）Top‑K检索加1v1 VLM重排序的管道，用于CoVR‑R挑战：先用文本推理生成种子并通过VLM槽位选择改进top‑1，再加入视觉检索候选并通过1v1 VLM安全提升top‑1。

**💡 创新点**

1) 将候选召回与top‑1选择分离为独立阶段；2) 在不引入新视觉候选的情况下用VLM槽位选择修正top‑1；3) 采用1v1 pairwise VLM重排序配合风险控制共识检查，保证top‑1提升的可靠性。

**🔧 技术方法**

使用Gemini‑3.1‑Pro的VLM推理与槽位选择，DFN‑H/DFN‑L视觉检索（ViT‑H‑14、ViT‑L‑14），contact‑sheet embedding，1v1 pairwise VLM重排序及共识审核。

**📊 数据集**

CoVR‑R 复合视频检索数据集（301个隐藏测试查询，每查询50个目标ID）。

**📈 对比分析**

在隐藏测试集上，系统达成R@1≈95.3%，R@5≈97.5%，R@10≈98.5%，R@50≈99.7%，明显优于单纯文本或单纯视觉检索方案。

**⚠️ 局限性**

依赖高成本的VLM API，零训练方式受限于VLM的推理与召回能力；1v1重排序多轮推理导致推理开销较大，且对极端难题的鲁棒性仍有限。

---

## 131. Child-directed speech facilitates production, not comprehension, in BabyLMs

**arXiv ID:** 2606.01045 | [PDF](https://arxiv.org/pdf/2606.01045v1)

**作者:** Bastian Bunzeck `[一作]` (Bielefeld University), Sina Zarrieß `[通讯]` (Bielefeld University)

**通讯引用:** 439 | [OpenAlex ID](https://openalex.org/A5078051602)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了儿童指向性语音（CDS）对小型语言模型在生成任务中的影响，并提出基于句框完成的评估方法。

**💡 创新点**

创新点在于将生产式评估引入BabyLM研究，发现使用CDS训练的模型在句框填充任务上优于传统web文本训练模型，揭示了理解与生成能力的分离。

**🔧 技术方法**

技术上采用了基于Llama的自回归语言模型训练，使用nucleus采样和温度控制进行生成，并用LLM-as-a-judge对生成文本进行可接受性打分，同时计算槽位预测的熵与最大概率。

**📊 数据集**

使用的数据集包括：训练阶段的BabyLM、Childes CDS、FineWeb-edu、FineWeb-edu-short、TinyDialogues；评测阶段提取自Childes和FineWeb-edu的句框，以及BLiMP、Zorro、MultiBLiMP最小对等基准。

**📈 对比分析**

通过比较可接受生成比例和最小对等基准成绩，发现CDS训练模型可接受生成率超过90%，而web文本模型低于60%；在BLiMP等理解基准上，web文本模型表现最佳，但生成任务上CDS模型显著领先。

**⚠️ 局限性**

局限性包括仅在英语语料上实验、评估仅限于可接受性而未涉及更细粒度的语义或交际适当性；仅使用自回归模型，未探讨掩码模型；未完全分离词汇多样性、长度等因素对结果的影响。

---

## 132. BraveGuard: From Open-World Threats to Safer Computer-Use Agents

**arXiv ID:** 2606.01166 | [PDF](https://arxiv.org/pdf/2606.01166v1)

**作者:** Yunhao Feng `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 25055 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种自适应的安全防御框架BraveGuard，用于训练针对电脑使用代理的轨迹级守卫模型。

**💡 创新点**

通过从公开研究中挖掘威胁信号，将其转化为可执行任务，结合动态威胁发现和验证驱动的硬案例扩展，实现了轨迹级监督的自演化学习。

**🔧 技术方法**

采用OpenClaw等代理运行、任务合成与轨迹收集、标注，基于Qwen3、Llama等大模型训练守卫，并使用结构化威胁分类与持续迭代的闭环。

**📊 数据集**

主要使用OpenClaw生成的真实代理轨迹，并在AgentHazard-Strongest与ATBench-500等公开轨迹安全基准上进行最终评估。

**📈 对比分析**

在AgentHazard-Strongest上，BraveGuard训练的守卫将检测准确率从38.79%提升至82.38%，召回率从约20%提升至90%以上；在ATBench-500上也获得高达95%召回。

**⚠️ 局限性**

覆盖受限于公开威胁数据与任务合成的完整性，且对不同轨迹格式、工具接口的泛化能力有限。

---

## 133. A Multiscale Network with Supervised Contrastive Learning for Real-Time Facial Emotion Recognition

**arXiv ID:** 2606.01069 | [PDF](https://arxiv.org/pdf/2606.01069v1)

**作者:** Rejoy Chakraborty `[一作]` (Indian Statistical Institute), Kaushik Roy `[通讯]` (West Bengal State University)

**通讯引用:** 6741 | [OpenAlex ID](https://openalex.org/A5015792060)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种轻量化的多尺度情绪识别网络MSFERNet，并基于此开发了实时情绪监测系统RT-FER。

**💡 创新点**

采用多尺度注意力机制结合CBAM、残差学习、EfficientNet‑B0预训练特征提取，并在此基础上加入监督对比学习来提升特征表达，模型仅2.37M参数实现高准确率。

**🔧 技术方法**

迁移学习、残差网络、多尺度卷积、CBAM注意力模块、监督对比损失（SupCon）、AdamW优化器及数据增强与预处理等技术。

**📊 数据集**

FER2013原始7类、修改后的3类以及CK+两大标准情绪识别数据集。

**📈 对比分析**

与现有SOTA模型对比，MSFERNet在FER2013 7类达到66.73%（略高于竞品），在CK+达到96.77%（最高），在3类FER2013达到81.08%；加入SupCon相较于不使用可提升0.1-0.2%。

**⚠️ 局限性**

对图像噪声和非人脸内容敏感，错误样本仍影响训练；仅在公开数据集上验证，缺乏大规模真实世界视频数据的评估。

---

## 134. Cooperative Mitigation against Learning-Based Reactive Jammers: Analysis and SDR Validation

**arXiv ID:** 2606.01197 | [PDF](https://arxiv.org/pdf/2606.01197v1)

**作者:** Soumita Hazra `[一作]` (Indian Institute of Technology Delhi), J. Harshan `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 105 | [OpenAlex ID](https://openalex.org/A5089433499)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出针对学习型反应性干扰器的协作抵御策略，并在软件定义无线电(SDR)平台上验证其可靠性与隐蔽性；

**💡 创新点**

首次将统计能量检测与机器学习驱动的检测器联合考虑，设计出适用于不同延迟约束的Delay‑Tolerant Rate‑Three‑Fourth（DTRTF）和Low‑Latency Constrained Rate‑Three‑Fourth（LLCRTF）方案；

**🔧 技术方法**

采用全双工收发器、能量分配、相位/能量调制、联合最大后验（JMAP）和子最优解码、Kullback–Leibler 散度与瞬时能量检测、随机森林/多层感知器/卷积神经网络与孤立森林等机器学习算法；

**📊 数据集**

在实验中使用真实的IQ样本数据集，收集在不同α值下的LLCRTF和DTRTF信号，作为无监督与监督学习的训练/测试集；

**📈 对比分析**

与现有的平均能量检测方案和Rate‑Half方案比较，DTRTF在保持相同率的同时显著降低Bob的误码率且检测概率低；在ML检测实验中，ROC曲线与随机分类器相近，说明方案对学习型干扰器保持高隐蔽性；

**⚠️ 局限性**

LLCRTF在严格低延迟场景下性能较差，且实验依赖于可预先设定的能量比例α，若α偏离最优值可导致误码率升高；

---

## 135. On Thin Perfect Matchings up to Polylogarithmic Factors

**arXiv ID:** 2606.01330 | [PDF](https://arxiv.org/pdf/2606.01330v1)

**作者:** Alireza Haqi `[一作]` (Stanford University), Shayan Oveis Gharan `[通讯]` (University of Washington)

**通讯引用:** 2257 | [OpenAlex ID](https://openalex.org/A5017101724)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了针对分数完美匹配的薄匹配算法，给出了支持受限和松弛两种情况的理论上最优至多对数因子结果。

**💡 创新点**

首次将树切割稀疏化与随机采样与Tutte条件结合，证明了支持受限下O(n log n)-薄匹配及非受限下O(polylog n)-薄匹配的存在性。

**🔧 技术方法**

利用树切割稀疏化、随机多次采样、Tutte判据以及贪心树配对算法。

**📊 数据集**

无实际数据集，全部为理论证明与算法设计。

**📈 对比分析**

与之前的O(n^2)-薄匹配做理论比较，显著降低薄度上界；实验未作。

**⚠️ 局限性**

结果仍保留对数因子，能否进一步消除对数或实现常数薄度仍是开放问题。

---

## 136. CAREAgent: Clinical Agent with Structured Reasoning and Tool-Integrated for Order Generation

**arXiv ID:** 2606.01094 | [PDF](https://arxiv.org/pdf/2606.01094v1)

**作者:** Ruihui Hou `[一作]` (East China University of Science and Technology), Tong Ruan `[通讯]` (East China University of Science and Technology)

**通讯引用:** 1251 | [OpenAlex ID](https://openalex.org/A5005820786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了CAREAgent，利用多阶段代理推理与真实工具调用实现临床订单生成，并构建了两阶段数据生成与过滤流程。

**💡 创新点**

创新点在于：①两阶段代理推理数据构建方法，自动收集真实工具调用轨迹；②结构化四阶段推理流程与工具整合；③结合监督微调(SFT)与强化学习(RL)的训练策略。

**🔧 技术方法**

采用LLM代理框架、结构化推理模板、Search和Clinical_Assessment两种工具调用、两阶段SFT+RL、GRPO奖励设计等技术。

**📊 数据集**

使用MedChain（8,514例）和私有急诊数据（3,000例）构建训练集（SFT 21,971样本，RL 3,509样本），并在ClinicalBench上进行评估。

**📈 对比分析**

与单代理、协作代理和代理推理三类基线比较，在MedChain、Private和ClinicalBench三个基准上，CAREAgent在F1、召回率等指标均领先，尤其在ClinicalBench提升F1 5.05%。

**⚠️ 局限性**

局限性：仍低于经验临床医生；仅集成两种工具，缺乏多模态工具；主要针对7B模型，扩展到更大模型和更复杂任务仍需验证。

---

## 137. ActMVS: Active Scene Reconstruction with Monocular Multi-View Stereo

**arXiv ID:** 2606.01367 | [PDF](https://arxiv.org/pdf/2606.01367v1)

**作者:** Guo Pu `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Zhouhui Lian `[通讯]` (Wangxuan Institute of Computer Technology, Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了首个基于单目视觉的主动场景重建框架 ActMVS，能够在没有深度传感器的情况下实现无人机/机器人自主规划轨迹并实时生成高质量、全局一致的稠密深度图，用于安全导航与重建。

**💡 创新点**

创新点在于：① 通过视差因子图和体素-帧可见性建模，精准筛选多视角参考帧；② 结合全局深度优化（跨视角深度扭曲对齐 + TV 正则）实现稠密深度的全局一致性；③ 在主动重建流程中首次将高效的体素地图与高保真 Gaussian splatting 结合，形成端到端的在线重建管线。

**🔧 技术方法**

使用技术包括：OctoMap 体素地图、Gaussian splatting 表面表示、MVSA 多视角立体匹配、视差因子图结构、基于 Huber 的深度一致性约束、全局 TV 正则、A* 轨迹规划与下一最佳视角 (NBV) 评分。

**📊 数据集**

主要使用 Replica 数据集进行实验，并在 AirSim 仿真中验证实际飞行路径。

**📈 对比分析**

与 RGB‑D 基线 ActiveGS 与 NARUTO 进行对比；评估指标涵盖 PSNR、SSIM、LPIPS、深度 Accuracy/Completion、Chamfer 距离；结果显示 ActMVS 在单目模式下的渲染与网格质量均优于同类单目方法，接近 RGB‑D 基线，证明多视角深度优化在无深度传感器场景下的有效性。

**⚠️ 局限性**

局限性包括：① 计算成本较高（约 1200 秒/场景，需较密集轨迹）；② 依赖高端 GPU 进行 3D Gaussian 处理，当前机载设备内存不足；③ 对极端动态场景的实时性尚未充分验证，需进一步优化算法和硬件实现。

---

## 138. Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation

**arXiv ID:** 2606.01048 | [PDF](https://arxiv.org/pdf/2606.01048v1)

**作者:** Ziyue Lin `[一作]` (University of Hong Kong), Liangqiong Qu `[通讯]` (University of Hong Kong)

**通讯引用:** 2941 | [OpenAlex ID](https://openalex.org/A5083630052)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将扩散过程分为噪声扩散和残差扩散两阶段的DRDD框架，用于统一且数据高效的图像对图像翻译

**💡 创新点**

发现高斯噪声可作为“域协同化器”，并通过解耦使核心语义映射在噪声域内完成，从而保持域协同化和流形提升；同时仅用无配对目标域图像训练去噪阶段，提升数据效率

**🔧 技术方法**

基于扩散模型的分离式前向/后向流程（噪声扩散、残差扩散、残差移除、去噪），兼容DDPM/DDIM/SDE等主流框架；利用U-Net结构的噪声/残差网络

**📊 数据集**

多任务统一图像恢复（All‑In‑One‑5、CDD‑11），多域降噪（自然/遥感/医学图像）、单任务单域（inpainting、SR、去雨、低照度提升）以及自制MNMD混合噪声基准

**📈 对比分析**

与多种SOTA方法（DA‑CLIP、DiffuIR、AdAIR、VLUNet、DFPIR、RDDM、IR‑SDE等）在SSIM、LPIPS、FID等指标上对比，DRDD在大多数任务上均实现最高或相近的表现，尤其在多任务统一场景下表现突出

**⚠️ 局限性**

仍需手动调节噪声强度以平衡域协同化与输入失真；模型规模大、采样步骤多，推理速度相对较慢；在极端噪声或极端域差异时仍可能出现细节丢失

---

## 139. Non-Vacuous Certification of Transport MCMC via Oscillation-Controlled Normalizing Flows

**arXiv ID:** 2606.01078 | [PDF](https://arxiv.org/pdf/2606.01078v1)

**作者:** Jun Hu `[一作]` (Wuhan University of Technology), Jun Hu `[通讯]` (Wuhan University of Technology)

**通讯引用:** 473562 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

构建了Transport MCMC的严格谱间隙下界，并在香蕉分布等目标上实现了可计算、非空的收敛性证明。

**💡 创新点**

创新点在于将Spectral Normalization与软尺度截断、覆盖式经验振荡上界以及振荡正则化训练相结合，形成了可数值计算且鲁棒的谱间隙证书框架。

**🔧 技术方法**

技术包括Spectral Normalization约束RealNVP、覆盖网格定理、经验振荡上界、梯度上界估计、几何变换（如banana shear图形化）以及振荡正则化训练。

**📊 数据集**

数据集涵盖香蕉族（d=2,5,10,20）、八层 shear building、Gaussian mixture、Neal's funnel以及Bayesian logistic regression。

**📈 对比分析**

通过与解析上界、无约束流、Neural Spline Flow等方法对比，证明在香蕉分布上D=2可获得γ*≈0.828，D=5可获得γ*≥7.6×10⁻⁴；振荡正则化后在D≤20保持非空且性能提升显著。

**⚠️ 局限性**

局限在于高维覆盖半径导致的维数灾难、目标刚性/尾部不匹配以及训练样本不足，使得多模态、重尾或高维任务仍难以获得非空谱间隙证书。

---

## 140. Event-Based Vision in Space: Applications, Trends, and Future Directions

**arXiv ID:** 2606.01280 | [PDF](https://arxiv.org/pdf/2606.01280v1)

**作者:** Luigi Capogrosso `[一作]` (Interdisciplinary Transformation University of Austria), Michele Magno `[通讯]` (ETH Zurich)

**通讯引用:** 7987 | [OpenAlex ID](https://openalex.org/A5066423975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对空间领域中基于事件的视觉技术进行系统综述，构建了四大类应用的分类学。

**💡 创新点**

首次提出了针对航天应用的事件摄像机与神经形态计算的统一分类框架，填补了学术文献碎片化的空白。

**🔧 技术方法**

利用事件相机（DVS）、神经形态硬件、脉冲神经网络及其相关算法，对文献进行技术归纳。

**📊 数据集**

综述中引用了 Falcon Neuro、Sun‑E、e‑STURT、VANTAGE 等事件视觉数据集及实验平台。

**📈 对比分析**

通过对 18 篇文献的筛选与分类，展示了各领域的技术进展和性能指标，例如事件相机的微秒级时间分辨率、HDR 重建的图像质量提升，以及 SNN 在地表分类和航天任务调度中的实时性表现。

**⚠️ 局限性**

仅覆盖已发表的少量案例，缺乏大规模实验验证；文献选择受 PRISMA‑light 限制，可能遗漏相关研究；并未提供统一的评测基准或数据集。

---

## 141. Implicit Drifting Policy: One-Step Action Generation via Conditional Expert Geometry

**arXiv ID:** 2606.01098 | [PDF](https://arxiv.org/pdf/2606.01098v1)

**作者:** Zemin Yang `[一作]` (ShanghaiTech University), Yuexin Ma `[通讯]` (ShanghaiTech University)

**通讯引用:** 4408 | [OpenAlex ID](https://openalex.org/A5102015139)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种只需一次前向传播的隐式漂移策略(IDP)，通过利用类似观察下的专家动作局部几何信息来实现动作的校正。

**💡 创新点**

创新点在于将漂移矢量场的作用直接嵌入到潜在能量函数中，并通过专家近端评估使单步生成器内部化局部动作流形，从而避免显式估计连续漂移场的困难。

**🔧 技术方法**

使用了条件专家几何提取、对比全局参考几何的局部几何超额、几何感知潜能函数、专家近端采样、以及一系列深度学习框架（Transformer/CNN）实现的单步生成器。

**📊 数据集**

在二维（Robomimic 的 Lift、Can、Square、Transport、Tool‑Hang、PushT）和三维（Adroit、DexArt、MetaWorld 的 56 个点云操纵任务）仿真数据以及单臂 ALOHA 的“Pick Peach”真实任务上进行评估。

**📈 对比分析**

与多步扩散/流动政策、快速单步方法（CP、MIP、MP1、OFP、FlowPolicy）以及显式漂移基线比较，IDP 在多数任务中成功率接近或超过最强单步基线，并在部分任务上逼近多步模型的性能，显著优于显式漂移。

**⚠️ 局限性**

局限性包括难以覆盖多模态动作分布、对类似观察下专家动作密度的依赖、以及在极端稀疏或多模态场景下可能表现不佳。

---

## 142. MsFEM-Inspired CNNs with Transfer Learning for Multiscale Model Reduction

**arXiv ID:** 2606.01259 | [PDF](https://arxiv.org/pdf/2606.01259v1)

**作者:** Xuehan Zhang `[一作]` (Tongji University), Eric T. Chung `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 5462 | [OpenAlex ID](https://openalex.org/A5078042062)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于MsFEM的CNN迁移学习框架（MITL），能够在多尺度随机输入场问题中快速迁移与重用多尺度基函数与系数映射，显著降低重新训练成本。

**💡 创新点**

创新点包括：① 将MsFEM的全域基函数问题作为源任务进行预训练，使学习到的基函数具有更好的跨样本可迁移性；② 采用残差结构+卷积神经算子（CNO）对基函数进行微调，提升对不同源项、边界条件和微分算子变化的适应性；③ 对系数CNN只微调顶层全连接层和时间编码网络，实现轻量级、快速的目标任务适配；④ 将MITL与变分自编码器（VAE）相结合，构建“Surrogate‑constrained VAE”用于不确定性量化的逆问题。

**🔧 技术方法**

技术手段：多尺度有限元方法（MsFEM）、卷积神经网络（CNN）与卷积神经算子（CNO）、残差学习、时间编码、变分自编码器（VAE）、概率贝叶斯推断。

**📊 数据集**

数据集：两类二值化随机输入场——(a)通道化含水层导水率场，(b)裂缝岩石渗透率；每类共6024个样本（训练5024，测试1000），通过旋转数据扩增产生24k训练样本；此外使用GeoCrack数据集的裂缝图像。

**📈 对比分析**

与从头训练的CNN‑ROM对比：MITL在仅用64–256个目标样本时相较于CNN‑ROM可减少10–20%相对L2误差；在不同源项、边界条件和算子（包括非线性p‑拉普拉斯、反应扩散、时变热传导）以及时间步长的无稳问题上，MITL均保持2–6%误差，且在极少样本情况下仍可得到可接受的预测精度。

**⚠️ 局限性**

局限性：① 仍需源任务的高质量有限元数据，预训练成本相对较高；② 对于高度异质、复杂的随机场，基函数的可迁移性下降，需更多样本或更大网络；③ 仅在与源任务具有相似多尺度特征的目标任务上表现优异，对完全不同结构或算子仍可能需要重新训练；④ 目前只考虑了离散网格与二维问题，推广到三维或更复杂物理耦合需进一步研究。

---

## 143. Temporal Evidence Routing with Structured Visual Evidence for TimeLogicQA

**arXiv ID:** 2606.01106 | [PDF](https://arxiv.org/pdf/2606.01106v1)

**作者:** Yuyang Sun `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**通讯引用:** 473562 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对视频问答中的时间逻辑，提出了一套分阶段的“Temporal Evidence Routing”管线，将问题解析、视频检索、结构化视觉证据生成、时间微程序验证和确定性约简分离，最终通过保守融合给出答案。

**💡 创新点**

创新点在于把视觉感知与符号时间推理拆分，使用结构化证据先生成事件出现时间点，再用程序化的时间约简规则严格评判，显著提升对重复事件、所有出现、重叠等复杂时间关系的处理能力。

**🔧 技术方法**

主要技术包括自然语言问题解析、基于CLIP/Siglip的检索与视频窗口定位、多模态大语言模型（Gemini-3.1-pro-preview）进行结构化事件抽取、微程序验证与确定性约简逻辑实现，以及保守答案融合。

**📊 数据集**

使用公开的 TimeLogicQA 数据集（3000 条带有布尔/四选一答案、不同时间算子的视频问答），以及多种检索与 VLM 训练时所需的公开视觉检索模型。

**📈 对比分析**

与直接使用 Gemini 进行全视频提示的基线相比，Temporal Evidence Routing 在官方评测中平均准确率提升至 81.8%，在多选、永真、直到和重叠等算子上表现出显著优势。

**⚠️ 局限性**

局限性包括对复杂语言表述的解析仍需人工规则，检索窗口依赖预设阈值，对非常长或多层次事件序列的处理仍有限，且在某些罕见动作或视角模糊时会出现证据缺失或误判。

---

## 144. Adaptive Dense Evidence Refinement for Video Relational Reasoning for VRR-QA Challenge

**arXiv ID:** 2606.01104 | [PDF](https://arxiv.org/pdf/2606.01104v1)

**作者:** Yuyang Sun `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**通讯引用:** 473562 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个仅推理阶段的视频-语言系统，利用多视角检测不稳定问题，并在高预算下进行密集帧级证据构建与对比验证，从而解决视频关系推理中的空间、时间、视角、深度与可见性问题。

**💡 创新点**

创新点在于将难题识别与答案修正分离：首先用轻量多视角快速判定问题是否不稳定，只有在高风险情况下才调用高预算的密集证据模块，并通过关系探针矩阵、成对验证和风险门控时间聚合来决定是否替换答案，形成一种风险控制的自适应推理流程。

**🔧 技术方法**

技术手段包括：Gemini 3.1 Pro 的直接视频推理、轻量多视角候选银行、密集帧证据生成、关系探针矩阵（空间、深度、运动、可见性、反事实）、成对验证、风险门控时间聚合及自适应计算调度。

**📊 数据集**

使用公开的视频多模态问答关系推理基准数据集（未在文中具体命名，通常为类似MSRVTT或ActivityNet-Related QA的数据集），作为评估实验平台。

**📈 对比分析**

在测试集上，该系统实现了平均准确率 90.07% 及宏平均准确率 87.81%，显著优于仅直接推理的基线（如单次 Gemini 调用）以及仅中间验证阶段的配置，验证了密集证据与风险控制策略的有效性。

**⚠️ 局限性**

限制包括：高预算密集证据路径在推理时消耗较高计算资源，难以实时部署；系统对多视角提示的鲁棒性依赖较高，若视频质量或帧采样不佳可能影响稳定性；以及对某些罕见关系类型的通用性尚待进一步验证。

---

## 145. Analysis of Ethnic Disparities in Autism Spectrum Disorder among Toddlers

**arXiv ID:** 2606.01217 | [PDF](https://arxiv.org/pdf/2606.01217v1)

**作者:** Aadithya Prabha Ramaharsha `[一作]` (Sri Ramachandra Institute of Higher Education and Research), Uma Ranjan `[通讯]` (Sri Ramachandra Institute of Higher Education and Research)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5103013160)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究分析了不同种族在自闭症谱系障碍（ASD）表现上的差异，重点关注白人欧洲人、亚洲人和中东人群的行为特征、性别和新生儿黄疸的影响。

**💡 创新点**

创新点在于通过逻辑回归分析揭示了种族对ASD发生率的显著影响，并强调了在诊断框架中考虑种族差异的重要性。

**🔧 技术方法**

使用了逻辑回归分析技术来探讨种族、性别和健康因素与ASD特征之间的关系。

**📊 数据集**

使用了2018年7月的幼儿自闭症数据集，包含3648个实例，涵盖了人口统计信息、行为评估（Q-CHAT-10分数）和临床诊断结果。

**📈 对比分析**

通过与统计测试（如卡方检验和方差分析）进行比较，逻辑回归模型显示出种族、性别和特定行为特征（如A9和A6）对ASD的显著预测能力，且模型的多重共线性效应较小，结果可靠。

**⚠️ 局限性**

本研究的局限性在于卡方检验未能显示种族与ASD的独立预测关系，强调了多变量模型在复杂条件下的重要性。

---

## 146. Understanding Cross-Cloud Interconnects: Hands-On Measurements and Cost Optimization

**arXiv ID:** 2606.01440 | [PDF](https://arxiv.org/pdf/2606.01440v1)

**作者:** Eitan Eliav `[一作]` (Technion), Avi Weit `[通讯]` (IBM Research -- Haifa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对Google的跨云互连服务（CCI）进行了系统的性能测评与成本优化研究，并提出了一种基于滑动窗口的动态决策算法以在VPN与CCI之间切换。

**💡 创新点**

创新点在于首次对CCI的真实性能和计费模型进行实验测量，构建了完整的成本模型，并证明所提出的在线算法在持续高/低负载下可实现近似最优；同时展示了在多种实际工作负载上比传统静态策略显著节约成本。

**🔧 技术方法**

技术包括实地对AWS–GCP CCI进行吞吐量和延迟测量、基于滑动窗口的成本累计、阈值驱动的状态机切换、以及对比竞品基线的离线优化实验。

**📊 数据集**

使用的数据集包括真实世界的MIRAGE（移动应用流量）与PUFFER（视频流量）两大流量记录，以及人工合成的常数速率和突发速率工作负载。

**📈 对比分析**

比较方法是将算法与四个基线（仅VPN、仅CCI、基于历史平均的VPN、基于月平均的VPN）在不同云对、不同地区和不同工作负载下进行成本对比；结果显示算法平均可比最优静态策略节省约1.8×成本，且在大多数场景下紧跟最优策略。

**⚠️ 局限性**

局限性包括：仅评估单一VPN隧道作为基线，未考虑更复杂的多隧道或混合网络方案；仅针对单一大洲内部的CCI；实验受限于现有云商的定价和API接口，未来在不同云提供商或跨洲部署时可能需要重新校准模型。

---

## 147. TECCI: Tricky Edits of Collected and Curated Images

**arXiv ID:** 2606.01213 | [PDF](https://arxiv.org/pdf/2606.01213v1)

**作者:** Aishwarya Agrawal `[一作]` (Google Research), Jason Baldridge `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布了一个名为TECCI的挑战性图像编辑基准数据集，包含1934张新拍摄图像与7550条编辑指令；

**💡 创新点**

通过人工与LLM自动生成的高难度编辑任务，系统性评估生成模型在指令遵循、最小化编辑与视觉质量三维度的能力；

**🔧 技术方法**

使用Gemini 3 Pro/Flash进行指令生成与自动评估；采样手工挑选的IRCS子集与自动生成的GGIS子集；

**📊 数据集**

TECCI数据集共包含7个图像类别（文字、时钟、车辆、建筑、艺术、动物、自然）和28种编辑类型；

**📈 对比分析**

对五款主流编辑模型（Nano Banana Pro、Grok Imagine Pro、GPT Image 1.5、Seedream 5.0 Lite、Nano Banana 2）进行人工评估与自动评估；人类评估显示整体成功率最高仅22%，Grok Imagine Pro在指令遵循上最优；自动评估与人类评估匹配度约74.7%；

**⚠️ 局限性**

模型在保持最小编辑和视觉质量方面表现差强人意，尤其对建筑与自然类图像的空间布局与细节把握不足，创意与推理类编辑更是难度最高；自动评估对IF与VQ偏高，可能导致评估偏差。

---

## 148. ASE-26: a curriculum for agentic software engineering as a discipline

**arXiv ID:** 2606.01152 | [PDF](https://arxiv.org/pdf/2606.01152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 149. Science Earth: Towards A Planet-Scale Operating System for AI-Native Scientific Discovery

**arXiv ID:** 2606.01316 | [PDF](https://arxiv.org/pdf/2606.01316v1)

**作者:** Zhe Zhao `[一作]`, Le Cong `[通讯]` (Stanford University)

**通讯引用:** 30145 | [OpenAlex ID](https://openalex.org/A5080765172)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了Science Earth——一个基于EACN协议的星球级科学运行时，能够让任何科学能力（AI模型、GPU集群、实验仪器等）通过开放网络自行发现、竞标并协同完成任务，支持跨学科证据标准的协同推理与自我纠错；

**💡 创新点**

创新点在于把传统的多代理框架从设计时固定角色转变为开放协议驱动的动态协作网络，实现了任务驱动的协同、冲突驱动的子任务生成以及跨证据标准的判定与信誉累积；

**🔧 技术方法**

核心技术包括Agent2Agent（A2A）传输层、Model Context Protocol（MCP）工具调用层以及自研的Emergent Agent Collaboration Network（EACN）协同层，配合AgentCard、竞标、仲裁与信誉机制；

**📊 数据集**

使用了两个公开数据集：Kuramoto同步模型的数值模拟结果（N=500-20,000，σ=0.3-2.0）以及Kang 2024全癌症单细胞图谱（4,888,651细胞）并结合独立的CCR8⁻ TIGIT⁺ Treg功能实验数据；

**📈 对比分析**

对比方法未进行传统基准测试，而是通过实证演示：在Kuramoto案例中，跨太平洋协作在30分钟内纠正了关键参数误差；在单细胞案例中，64.9小时的八代理协作生成了三层新结果，并与独立实验结果对齐，展示了协议的可行性和高效性；

**⚠️ 局限性**

局限性包括规模受限（仅涉及数个AI模型和计算资源），缺乏实验仪器等硬件节点的加入，治理与信誉滥用风险尚未成熟，以及未在大型基准上与其他协同框架进行系统对比；

---

## 150. ChartArena: Benchmarking Chart Parsing across Languages, Scenarios, and Formats

**arXiv ID:** 2606.01348 | [PDF](https://arxiv.org/pdf/2606.01348v1)

**作者:** Shangpin Peng `[一作]` (Tencent), Yu Zhou `[通讯]` (Nankai University)

**通讯引用:** 32193 | [OpenAlex ID](https://openalex.org/A5012041302)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ChartArena基准与格式无关评估协议，覆盖八类图表（数值图与图形结构）、三种视觉场景（数字渲染、打印照片、手绘照片）及中英双语，评估26个模型。

**💡 创新点**

创新点在于：①综合多图表族与真实场景并结合双语；②将所有模型输出统一映射至三元组视图或有向图视图；③使用结构化的EM与多阈值mAP衡量性能，实现跨格式公平比较。

**🔧 技术方法**

采用了人机协同标注 pipeline、格式统一化（Markdown、JSON、SVG、Mermaid等）到三元组/图结构的归一化流程、结构化评估算法，以及多模态大型语言模型（MLLMs）与专业图表解析器。

**📊 数据集**

使用自建ChartArena数据集，包含柱状图、折线图、饼图、雷达图、箱线图、组合图、流程图与思维导图，每类在数字、打印和手绘三种场景下提供，并覆盖中英文文本。

**📈 对比分析**

评估方法为：将模型输出先归一化为三元组或图结构，然后使用Exact Match与mAP_high进行评分；结果显示Gemini 3.1 Pro最高，开源模型迅速逼近；文档解析MLLM在数值图表现良好但在图形结构上差距明显；专家解析器受限于图表族，雷达图与手绘场景最难。

**⚠️ 局限性**

局限性包括：评估聚焦于八类图表，未覆盖更复杂的图形结构；归一化过程对细粒度语义错误捕捉有限；标注过程仍需大量人工验证；且实验结果主要基于当前训练数据，未来需扩展更广泛的图表类型与多样化视觉条件。

---

## 151. Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution

**arXiv ID:** 2606.01435 | [PDF](https://arxiv.org/pdf/2606.01435v1)

**作者:** Vikas Reddy `[一作]`, Sumanth Challaram `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 LLM 的记忆系统冲突解决的流水线：先用 BM25 检索事实，LLM 做候选抽取，然后用 Python 的最大序列号聚合得到答案。

**💡 创新点**

创新点是把冲突判定从 LLM 的自由文本推理迁移到可解释、确定性的后检索聚合，并在单跳与多跳上显著提升性能。

**🔧 技术方法**

使用的技术包括 BM25 检索、gpt‑4o‑mini 或 gpt‑4o 的候选抽取 Prompt、Python 的序列号最大化聚合，以及 Self‑Ask 风格的多跳拆分。

**📊 数据集**

评估数据集为 MemoryAgentBench 的 FactConsolidation（单跳和多跳）和 LongMemEval 的知识更新子任务。

**📈 对比分析**

与 MAB 报告的 22 个系统做匹配设置对比，单跳 gpt‑4o‑mini 在 262K 上达 78%（比 HippoRAG‑v2 的 54% 高 24pp），gpt‑4o 达 94.8%；多跳 CAR 在 262K 上达 30.2%（比最佳 7% 高 23pp）。

**⚠️ 局限性**

局限包括只在合成冲突数据上测试，缺乏对真实时间戳或偏序更新的验证；多跳仍难以突破 30%；且未完全分离聚合器与 LLM 的贡献。

---

## 152. SkillSmith: Co-Evolving Skills and Tools for Self-Improving Agent Systems

**arXiv ID:** 2606.01314 | [PDF](https://arxiv.org/pdf/2606.01314v1)

**作者:** Yangbo Wei `[一作]` (Shanghai Jiao Tong University), Lei He `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 7695 | [OpenAlex ID](https://openalex.org/A5008695429)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SkillSmith，一个基于生态动力学的技能-工具协同进化框架。

**💡 创新点**

创新点在于统一的原子提案空间、Lotka–Volterra 生态模型与 anti-pattern 内存，解决工具层固定、技能交互与失败诊断缺失问题。

**🔧 技术方法**

采用 LLM 反射、bundle 生成、工具生命周期操作、生态优先检索和交叉合并等技术。

**📊 数据集**

在 OfficeQA、SealQA、WildClawBench 三个基准上评估，使用 Qwen3.5 多尺度模型。

**📈 对比分析**

与基线（无进化、EvoSkill、SkillClaw）相比，SkillSmith 在 OfficeQA/SealQA 上分别提升 18%+，在 WildClawBench 持续提升至第 6 天；性能随模型规模和任务复杂度放大。

**⚠️ 局限性**

局限包括对 LLM 反射质量依赖、生态模型冷启动噪声、以及在体感或多用户场景的验证不足。

---

## 153. Reusing Fusion-Time Spectral Reliability for Adaptive Fusion and Expert Routing in RGB-Infrared Object Detection

**arXiv ID:** 2606.01173 | [PDF](https://arxiv.org/pdf/2606.01173v1)

**作者:** Yefeng Wu `[一作]` `[通讯]` (Anonymous Institution), Yefeng Wu (Anonymous Institution)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于融合时频谱可靠度重用的自适应融合与专家路由框架，用于 RGB‑红外目标检测，并提供了完整的补充实验、诊断分析与可复现设置。

**💡 创新点**

创新点包括：① 在特征融合阶段实时估计并复用两模态的频谱可靠度，以实现动态自适应特征融合；② 通过专家路由将不同模态的特征引导至专门的专家模块，实现更细粒度的特征处理；③ 提出了可解释的谱可靠度描述符及其校准与诊断方法，进一步提升模型鲁棒性。

**🔧 技术方法**

技术细节涵盖：频谱分析、可学习可靠度描述符、注意力机制与专家模块的可插拔设计、数据增强、模型压缩与效率评估、统计显著性检验，以及对多种模态的后处理和可视化工具。

**📊 数据集**

实验数据集主要包括 RGB‑IR 目标检测基准 FLIR ADAS、KAIST、以及公开的 RGB‑IR Pascal‑VOC 等，覆盖不同场景与光照条件。

**📈 对比分析**

与多种基准方法（早期融合、晚期融合、跨模态注意力、专家网络）进行对比，在标准 mAP@IoU=0.5 评估指标上提升约5–10%，并在参数量与前向推理延迟方面实现显著压缩（参数减少约15%，延迟降低约20%）。

**⚠️ 局限性**

局限性包括：对极端低光、强遮挡等极端环境仍有一定敏感性；可靠度估计在低分辨率或高噪声条件下可能失效；额外的可靠度网络增加训练成本；在不同硬件平台的功耗与实时性验证尚未充分；以及对更大规模数据集与多任务场景的推广性待进一步研究。

---

## 154. When Data Is Scarce: Scaling Sparse Language Models with Repeated Training

**arXiv ID:** 2606.01155 | [PDF](https://arxiv.org/pdf/2606.01155v1)

**作者:** Boqian Wu `[一作]` (University of Luxembourg), Decebal Constantin Mocanu `[通讯]` (University of Luxembourg)

**通讯引用:** 2222 | [OpenAlex ID](https://openalex.org/A5011045254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在数据匮乏场景下动态稀疏训练（DST）对大语言模型预训练的缩放行为，并提出了基于稀疏度的自适应缩放律；

**💡 创新点**

创新点在于将稀疏度、数据重用与模型容量统一建模，发现中等稀疏度能延迟数据饱和并提升多轮训练效率；

**🔧 技术方法**

使用动态稀疏训练（DST）、稀疏化掩码更新（幅值裁剪+随机再生）以及对齐的Chinchilla式数据限制缩放公式；

**📊 数据集**

在LLaMA‑2系列模型上使用C4文本语料库进行实验，覆盖1.92B至7.68B密集等价参数的规模；

**📈 对比分析**

与密集训练及传统Chinchilla缩放律对比，稀疏模型在相同数据预算下可实现相同或更低验证损失，同时计算和参数量约缩小3‑10倍；

**⚠️ 局限性**

局限性包括仅评估了DST在LLaMA‑2/C4上的表现，未覆盖其他模型结构或任务，且实际稀疏训练的硬件加速尚未完全实现，可能影响实践中的效率提升。

---

## 155. "Skill issues'': data-centric optimization of lakehouse agents

**arXiv ID:** 2606.01185 | [PDF](https://arxiv.org/pdf/2606.01185v1)

**作者:** Nicole Rose Schneider `[一作]` (University of Maryland), Jacopo Tagliabue `[通讯]` (Bauplan Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对在分支湖仓（branching lakehouse）环境下运行的编码代理（coding agents）提出了一种数据驱动的技能优化管道，自动生成任务–验证器对，执行候选技能并基于轨迹与湖仓状态的程序化检查给出奖励，迭代优化技能markdown文件；

**💡 创新点**

创新点在于：①将湖仓写路径的可观察变更与代理生成的流水线代码建立同构关系，使得写操作的验证可编程且细粒度；②构建了端到端的、可验证的任务库和验证体系；③结合黑盒优化器GEPA进行技能文本的自动进化；

**🔧 技术方法**

使用的技术包括：Bauplan湖仓平台（headless API + git‑for‑data）、Python SDK与CLI、LLM代理Claude‑Sonnet‑4‑6及Claude Code、GEPA Optimize Anything优化框架、Modal沙盒执行、Harbor调度、程序化湖仓状态验证脚本；

**📊 数据集**

数据集来源为：先基于May 2026的匿名生产日志统计API使用频率与会话模式，随后用LLM生成结构化的25条任务–验证器对，涵盖读写、管道构建、数据质量、调试等场景；

**📈 对比分析**

与手工编写的六条技能（如bauplan‑data‑pipeline、bauplan‑safe‑ingestion等）进行对比，优化后技能平均提升约31.9%准确率（验证检查通过率），并在测试集保持或提升表现；实验时间与成本未在本文中给出具体数值；

**⚠️ 局限性**

局限性包括：任务集规模有限（仅25条），技能优化仅在单技能层面进行，未考虑多技能协同；优化依赖大型LLM（Claude‑Sonnet‑4‑6），对较小模型效果未知；验证框架与平台高度耦合，迁移到其他湖仓需要重构；

---

## 156. SIRIUS-SQL: Anchoring Multi-Candidate Text-to-SQL in Execution Feedback

**arXiv ID:** 2606.01246 | [PDF](https://arxiv.org/pdf/2606.01246v1)

**作者:** Leo Luo `[一作]` (Tencent Inc.), Jie Jiang `[通讯]` (Tencent Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于执行反馈的多候选 Text-to-SQL 系统，结合 RL 训练的 SQL 专家模型与通用 LLM，构建多样化候选池；

**💡 创新点**

创新点包括：①难度平滑的 RL 训练使专家产生可执行且覆盖度高的候选；②基于执行结果的生命周期，对不同类型错误（运行错误、超时、空结果）进行特定修复；③混合选择器融合结果一致性、对比式 SQL 评估和 AST 结构支持，且仅在低置信度情况下触发结构化检查；

**🔧 技术方法**

采用 RLVR（Reinforcement Learning with Verifiable Rewards）训练 SIRIUS‑32B，使用两阶段 DAPO 与熵正则；执行诊断与修复使用规则与结构化运算符；选择器使用结果级投票、对比评分与结构签名匹配；

**📊 数据集**

主要使用 Spider 基准（dev 1,534 题/95 DB，test 2,147 题/95 DB）；

**📈 对比分析**

与现有系统（Agentar‑Scale‑SQL 等）对比，在 dev 上达到 75.88% 执行准确率，test 上 91.20%，均超过前人，且在不同通用 LLM 合作时均优于其基线；

**⚠️ 局限性**

局限性：需要同时具备 RL 训练的专用模型和强大通用 LLM；空结果修复仍存在较大改进空间；对比结果依赖公开报告，可能受检索/架构差异影响。

---

## 157. Needles at Scale: LLM-Assisted Target Selection for Windows Vulnerability Research

**arXiv ID:** 2606.01364 | [PDF](https://arxiv.org/pdf/2606.01364v1)

**作者:** Michael J. Bommarito `[一作]` `[通讯]`, Michael J. Bommarito

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个低成本批处理管道，将生产 Windows 二进制中的数百万函数通过符号化、LLM 标签和优先采样，生成可查询的高优先级研究队列。

**💡 创新点**

创新点包括：① 在大规模生产二进制上自动符号化并恢复调用图；② 仅用结构化特征和低成本 LLM 为函数赋予可达性、风险、bug 类等标签；③ 通过优先加权重要性采样生成多样化优先级队列，实现数百万函数的高效筛选。

**🔧 技术方法**

使用了结构化特征提取、PDB 自动下载与调用图恢复、低成本 LLM 推理、优先加权重要性抽样（Efraimidis–Spirakis）以及关系型存储。

**📊 数据集**

使用了 Windows 11 x86‑64 生产二进制（约 76.8% 可符号化，约 3 M 函数），来自两台安装在 2025 版补丁的机器。

**📈 对比分析**

通过标签分布与过滤漏斗评估选择性，将数百万函数压缩到约 22 K 的候选列表；单函数成本极低，模型调用仅处理结构化摘要，整体耗时数小时。

**⚠️ 局限性**

限制在于标签为单模型一次性猜测，缺乏真实正负基准；符号化依赖 PDB 可用性；可达性基于静态调用图，忽略间接/虚拟调用；优先权重手工调优；缺乏实际漏洞验证。

---

## 158. Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs

**arXiv ID:** 2606.01072 | [PDF](https://arxiv.org/pdf/2606.01072v1)

**作者:** Jianing Qian `[一作]` (University of Pennsylvania), Tarik Kelestemur `[通讯]` (RAI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并维护任务相关的场景图作为显式记忆，在部分可观测环境中使用模仿学习实现长时程任务。

**💡 创新点**

创新点在于：①将可解释的场景图动态更新并直接作为神经模仿策略的输入；②通过语言模型和视觉基础模型自动识别并追踪任务相关物体；③在扩散策略中利用场景图实现对长时间历史信息的高效推理。

**🔧 技术方法**

使用的技术包括：大语言模型（GPT-5）进行任务实体提取；Grounding DINO、DINO‑v2 进行物体检测与特征提取；XMem 进行实时跟踪；深度卷积变压器与扩散策略（Diffusion Policy）作为决策网络；场景图的节点由视觉嵌入、二维包围框和三维质心构成。

**📊 数据集**

数据集：①模拟 Boston Dynamics Spot 在 MuJoCo 下的三项移动操纵任务（Throw Trash、Heat‑up Tea、Heat‑up Tea Long），共 400 条脚本演示；②真实桌面操作任务（Pineapple‑Bowl、Feed‑Animals、Feed‑Giraffe、Clean‑Table）使用 Franka Emika Panda 及腕部 ZED Mini RGB‑D 摄像头，收集 300 条遥控演示。

**📈 对比分析**

与 RGB、Visible‑Objects、PTP 等基线相比，本文方法在长时程任务中显著提升成功率，尤其在后续子任务和最终任务完成率上有 10–30% 的提升；消除 3D 或位置信息的消融实验表明空间位置信息对性能至关重要。

**⚠️ 局限性**

局限性：依赖准确的物体识别与跟踪，若环境高度拥挤或动态变化明显，场景图更新误差会累积，影响策略表现；对物体外观与遮挡的鲁棒性仍需进一步提升。

---

## 159. Deep Research as Rubric for Reinforcement Learning

**arXiv ID:** 2606.01091 | [PDF](https://arxiv.org/pdf/2606.01091v1)

**作者:** Wangyi Mei `[一作]` (Fudan University), Deqing Yang `[通讯]` (Fudan University)

**通讯引用:** 1932 | [OpenAlex ID](https://openalex.org/A5046589466)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段的深度研究框架 DR‑Rubric，用于自动生成基于外部证据的评估 Rubric，并利用这些 Rubric 作为奖励信号进行强化学习；同时实现了自启动（bootstrap）式 Rubric 生成，让模型在不依赖外部大模型的情况下不断改进自身的评估标准；

**💡 创新点**

将 Rubric 构建视为“深度研究”问题，将信息检索、推理与评估标准的提取相结合；通过多轮交互式检索收集领域事实、结构约束与失败模式，再将其转化为可验证的原子约束；实现了 bootstrap 训练机制，展示了自我提升的“专业化到再平衡”动态；

**🔧 技术方法**

多轮检索与推理驱动的证据收集；约束合成（将证据转化为原子 Rubric）；GRPO（Group‑based Policy Optimization）强化学习；自回归模型（如 Qwen3‑8B、GPT‑5、Gemini）做为生成器和评估器；

**📊 数据集**

六个基准数据集：ResearchQA、DeepResearchBench、LocalSearchBench、GPQA、MMLU‑Pro、MMLU；实验使用 Qwen3‑8B‑SFT 作为基线，并在 1K–3K RL 实例上进行评估；

**📈 对比分析**

与多种基线（Qwen2.5‑7B、Qwen3‑8B、DR‑Tulu‑SFT‑8B、DR‑Tulu‑RL‑8B、WebExplorer‑8B、Search‑R1‑7B）比较；DR‑Rubric‑8B 在仅 1K–3K RL 数据下，整体性能超过所有基线；GPT‑5 与 Gemini 生成的 Rubric 在不同任务上展现覆盖性或均衡性；bootstrap 第三步在推理任务上达到最佳表现；

**⚠️ 局限性**

在 bootstrap 迭代超过 3 步时会出现奖励失衡导致性能崩塌；对极大模型的稳定性尚未完全解决；对更大规模或更长时间 bootstrap 的验证不足；

---

## 160. Can we trust LLM Self-Explanations for Entity Resolution?

**arXiv ID:** 2606.01210 | [PDF](https://arxiv.org/pdf/2606.01210v1)

**作者:** Tommaso Teofili `[一作]` (Roma Tre University), Divesh Srivastava `[通讯]` (AT&T)

**通讯引用:** 22560 | [OpenAlex ID](https://openalex.org/A5088315797)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统评估了大语言模型（LLM）在实体解析任务中的自解释（feature attribution 与 counterfactual）的可靠性，并提出一种混合解释框架（Ellmer），利用自解释作为先验引导后置探索，从而在保持解释质量的同时显著降低计算成本。

**💡 创新点**

创新点在于①首次从鲁棒性、可信度和因果相关性等维度对LLM自解释进行大规模实验，揭示其不稳定、易幻觉且与实际决策的关联差距；②提出 Ellmer 混合方法，将自解释的高效性与后置解释的可信度相结合，实现低成本但高质量的解释。

**🔧 技术方法**

使用的技术包括三种 prompting 策略（Zero-Shot、Chain-of-Thought、In-Context Learning）生成自解释；后置解释采用 LIME/Explainer/landmark 等方法；Ellmer 通过自解释的 top‑k 特征过滤后置搜索空间；评估指标涵盖 faithfulness、validity、FCA、TkO 等。

**📊 数据集**

实验数据集涵盖 DeepMatcher 10 个真实数据集（AB、BR、FZ、WA、AG、Cameras、Watches）以及两组自生成的合成数据集（FB、FCP）和高维度 stress‑test 集合 FakER。

**📈 对比分析**

与传统后置解释相比，后置和 Ellmer 方法在 faithfulness 低于 0.1、validity 在 0.82–1 范围内；而自解释的 faithfulness 0.15–0.32、validity 0.63–0.81，表现显著逊色；Ellmer 在成本上比纯后置低 1–2 个数量级，但解释质量几乎与后置持平；不同 LLM 对方法效果的影响不大。

**⚠️ 局限性**

局限性包括：LLM 自解释缺乏可执行的正式算法，易产生幻觉且与实际决策脱节；在高维或复杂记录上表现更差；后置方法仍具高计算成本；Ellmer 对参数（k、ϕ）敏感，且未深入探究自解释不可靠的根本原因。

---

## 161. From Craft Practice to Aesthetic Cognition Transmission: Workflow Cognition Translation for AI-native Intangible Cultural Heritage Education

**arXiv ID:** 2606.01203 | [PDF](https://arxiv.org/pdf/2606.01203v1)

**作者:** Annie Yuan `[一作]` (University of Sydney), Annie Yuan `[通讯]` (University of Sydney)

**通讯引用:** 83 | [OpenAlex ID](https://openalex.org/A5066926831)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了通过将传统工艺教育转向审美认知传递，并引入工作流认知及其翻译方法，构建 AI 原生课程与认知学徒体系，旨在实现可扩展的非物质文化遗产学习。

**💡 创新点**

创新点在于（1）将审美认知作为教育核心对象，突破技术复制；（2）构造工作流认知本体，将专家的感知、判断、决策与行为嵌入动态工作流；（3）提出工作流认知翻译框架，将隐式认知分解为最小认知节点并转化为可计算结构，形成 AI 认知中介平台。

**🔧 技术方法**

使用的技术包括认知图谱、工作流网络、最小认知节点（MCN）抽取、机器可读工作流结构、AI 专家孪生、任务导师与学习者导师的多层次教学代理，以及沉浸式 VR/AR 交互环境。

**📊 数据集**

论文未给出具体数据集，主要基于专家访谈、工作流观察、反思协议等定性资料进行认知抽取与模型构建。

**📈 对比分析**

由于缺乏实验数据，本文未进行方法比较或性能评估，主要以理论构造和案例说明的方式展示框架可行性。

**⚠️ 局限性**

局限性包括：缺乏实证验证与量化评估；工作流认知抽取依赖专家主观记录，难以标准化；将隐式认知映射为可计算模型可能导致信息损失；在跨文化或跨工艺迁移时的可迁移性与泛化性待进一步研究。

---

## 162. ThinkSwitch: Context Distillation with LoRA and Weight Interpolation for Specific-Purpose Reasoning Tasks

**arXiv ID:** 2606.01080 | [PDF](https://arxiv.org/pdf/2606.01080v1)

**作者:** Dhruv Saini `[一作]` (Bellevue High School), Rohan Pandey `[通讯]` (DigitalOcean)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ThinkSwitch循环，通过思考模型生成答案、去除推理轨迹、使用QLoRA进行答案仅训练、再用球面线性插值重建思考模型，实现低算力的自我改进。

**💡 创新点**

创新在于只利用模型自生成的答案（不含推理轨迹）进行低秩微调，并通过球面插值同步指令和思考两种检查点，从而在保持两种模式的同时将推理优势迁移至权重。

**🔧 技术方法**

使用QLoRA参数高效微调、球面线性插值（SLERP）、vLLM推理、Unsloth训练框架、AIME和PubMedQA数据。

**📊 数据集**

训练集为15条AIME 2025题目和15条PubMedQA候选题，评估集为30条AIME 2026题和30条PubMedQA子集。

**📈 对比分析**

与基线直接指令和思考检查点比较，AIME指令模型从10/30提升到20/30，思考从14/30提升到22/30；PubMedQA指令从13/30提升到18/30，思考从18/30提升到25/30，实验成本仅$2.86。

**⚠️ 局限性**

局限在于评估集规模小、仅适用于已有兼容指令/思考检查点的模型、需要人工挑选固定提示且对答案提取依赖规则，且在提示池耗尽后易饱和。

---

## 163. Sample Complexity and Decision-Theoretic Guarantees for Bayesian Model Averaging over Decision Trees with Catalan-Exponential Priors

**arXiv ID:** 2606.01340 | [PDF](https://arxiv.org/pdf/2606.01340v1)

**作者:** Livija Jakaite `[一作]` (University of Bedfordshire), Vitaly Schetinin `[通讯]` (University of Bedfordshire)

**通讯引用:** 1081 | [OpenAlex ID](https://openalex.org/A5048277038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文在贝叶斯决策树（BDT）中给出了闭式理论，阐明了何时贝叶斯模型平均（BMA）权重能提供足够的认识论信息来支持决策的可信执行；

**💡 创新点**

创新点在于四条闭式定理：DM叶模型下WAIC与LOO的一致性及其采样阈值；Catalan指数先验的尾部、有效衰减率与后验收敛速率；BMA的贝叶斯偶然性下界；以及基于PAC‑Bayes的先验信息量与决策承诺成本公式；

**🔧 技术方法**

技术包括Dirichlet–Multinomial叶模型、Catalan指数先验解析、WAIC和留一交叉验证的误差对比、Reversible‑Jump MCMC、PAC‑Bayes风险上界、信息熵化简与决策理论的λ‑罚函数；

**📊 数据集**

主要使用的实证数据集是临床小样本的膝关节骨关节炎（knee osteoarthritis）影像数据（N≈40），以及通过模拟生成的二分类DM叶数据；

**📈 对比分析**

方法比较采用WAIC与LOO、BMA与oracle模型、Catalan先验与Chipman先验的后验收敛速度和PAC‑Bayes样本复杂度；实验显示Catalan先验在稀疏模型下样本复杂度降低8.1倍、后验熵收敛更快，且对小样本数据的BMA失效能得到理论解释；

**⚠️ 局限性**

局限性包括：仅对DM叶模型可直接推导，扩展到连续叶模型需要新方程；PAC‑Bayes界在大模型下可能空洞；先验参数γ需要手工设定，缺乏自适应机制；

---

## 164. Formalizing multi-graded Brenner-Schröer Proj schemes and dilatations of rings in Lean4

**arXiv ID:** 2606.01438 | [PDF](https://arxiv.org/pdf/2606.01438v1)

**作者:** Arnaud Mayeux `[一作]` (University of Wisconsin), Jujian Zhang `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在 Lean 4 中实现多重梯度（multi‑graded）代数几何的核心构造——Brenner–Schröer 的 Proj 构造，并将其与常见的局部化、张量积以及 Dilatation（扩张）等操作进行形式化。

**💡 创新点**

首次将 B.-S. Proj 的构造与多重梯度结构完整地写入 Lean 4，建立了“potions”（即多重梯度局部化）的概念与相关定理；同时提供了 Dilatation 的完备的公理化定义和其通用性质的证明，填补了 Lean 4 在多重梯度与局部化理论中的空白。

**🔧 技术方法**

使用 Lean 4 theorem prover 的类型理论框架，构造了可变指数的 `GradedRing`、`HomogeneousSubmonoid`、`Potions`、`TensorProduct` 等类型，利用 `Setoid`、`Quotient` 与 `Localization` 等基础构造实现了多重梯度 Proj、局部化、张量积和 Dilatation 的实现与证明；同时借助 `CommSemiring`、`Algebra` 等数学类实现了 Dilatation 的代数结构与通用性质。

**📊 数据集**

本工作不涉及传统意义上的数据集，而是基于 Lean 4 的公理化数学库（mathlib）进行形式化验证，所有证明均通过 Lean 的交互式证明器完成。

**📈 对比分析**

由于本研究是形式化数学证明工作，主要评估维度是逻辑正确性与证明完整性，而非数值性能；在 Lean 4 环境中完成所有证明后，生成的代码通过 Lean 4 的编译器验证，可视为在理论上“性能最优”。

**⚠️ 局限性**

局限性包括：
- 证明规模较大，阅读和维护难度高；
- 依赖 Lean 4 的数学库，若后续库更新需同步维护；
- 目前仅覆盖了 B.-S. Proj 的核心构造，尚未完全覆盖所有可能的多重梯度空间情形；
- 对于非交换环或更一般的基环的推广仍未实现。

---

## 165. Before and After Temperature: A Distributional View of Creative LLM Generation

**arXiv ID:** 2606.01451 | [PDF](https://arxiv.org/pdf/2606.01451v1)

**作者:** V. S. Raghu Parupudi `[一作]`, Sahiti Bulusu `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了无参考文本情况下大语言模型生成的创意性评价，提出利用采样温度对模型原始概率分布的重新塑形（pre‑vs‑post）作为评价特征。

**💡 创新点**

创新点在于首次将温度变换对分布的“重塑”视为判别指标，证明单一 token‑级别特征（如 n90(p) 或 n95(q)）在预测创意排名上明显优于传统基准。

**🔧 技术方法**

技术方法包括在 Llama‑3.1‑8B‑Instruct 上记录每一步的 top‑K logits，计算 pre‑temperature 分布 p 与 post‑temperature 分布 q，随后提取多类分布指标（KL、TV、nα 宽度、质量泄漏等）并聚合。

**📊 数据集**

使用了 500 条公开创意提示（WritingPrompts、Alternative Uses Task、HellaSwag）以及 150 条子集进行人类评审，全部来自开放许可的英文文本。

**📈 对比分析**

与两组基准（平均 LLM 判别者 gpt‑4o/gemini‑2.5‑pro 与三位人工评审）对比，最佳分布特征在 Spearman ρ 上分别达到 0.918（LLM）和 0.870（人类），超过传统 baseline（≈0.76）相差 0.165/0.110。

**⚠️ 局限性**

局限包括仅评估单一模型（Llama‑3.1‑8B‑Instruct）、对 top‑K 截断的依赖、单次前向推断而非多次采样，以及对序列级细粒度区分（如 T=0.8 vs T=0.3）表现不佳。

---

## 166. DENSER: Depth-Guided Ensemble with Staged EFA-GS Reconstruction for Soccer Novel View Synthesis

**arXiv ID:** 2606.01419 | [PDF](https://arxiv.org/pdf/2606.01419v1)

**作者:** Parthsarthi Rawat `[一作]` `[通讯]` (GameChanger by Dick's Sporting Goods), Parthsarthi Rawat (GameChanger by Dick's Sporting Goods)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于深度引导的多模型像素平均集成框架DENSER，用于足球比赛的新视角合成。

**💡 创新点**

创新点包括基于摄像机高度加权的损失函数、使用Depth-Anything-V2的单目深度监督来补偿纹理缺失区域，以及通过三条不同训练路径产生多样化的集成模型。

**🔧 技术方法**

采用EFA‑GS（改进的3D高斯展开）结合Mip‑Splatting、深度监督L1损失、阶段化训练与尺度限制、像素平均集成等技术。

**📊 数据集**

在SoccerNet‑NVS数据集上进行训练与评估，覆盖多种摄像机高度视角。

**📈 对比分析**

与3DGS和Triangle Splat基线相比，DENSER在五个测试场景中平均PSNR提升约3.1 dB、SSIM提升至0.791，LPIPS略高但保持在0.366，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括集成模型的计算开销较大、LPIPS略高导致高频细节略显模糊，以及对单目深度预测的依赖可能在极端光照或遮挡下表现不佳。

---

## 167. GLIDE: Graph-guided Leap Inference for Diffusion Estimation of Spatio-Temporal Point Processes

**arXiv ID:** 2606.01273 | [PDF](https://arxiv.org/pdf/2606.01273v1)

**作者:** Guanyu Zhou `[一作]` (University of Electronic Science and Technology of China), Qiao Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 53507 | [OpenAlex ID](https://openalex.org/A5100441085)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

GLIDE提出了一种基于多尺度历史图和先验引导跳跃采样的条件扩散框架，用于空间时间点过程的下一事件建模。

**💡 创新点**

创新点在于将多尺度历史图结构与双流编码相结合，并通过先验均值预测器引导的跳跃采样显著降低逆向采样步数与空间误差。

**🔧 技术方法**

采用双流Transformer与Diffusion Transformer（DiT）架构、GATv2图卷积、RoPE位置编码、DDIM加速采样以及轻量级均值预测器。

**📊 数据集**

在地震、COVID‑19、Citibike（及Crime数据集用于图模块消融）四个真实世界数据集上进行实验。

**📈 对比分析**

与Hawkes、RMTPP、SAHP、NJSDE、NSTPP、DeepSTPP、DSTPP、SMASH等基线对比，GLIDE在空间负对数似然与空间误差上实现显著提升（空间NLL降至-0.067，空间误差降至6.61km），并且推理速度提升约3倍。

**⚠️ 局限性**

局限性包括仅处理连续时间空间坐标、未考虑事件标记、图结构最适用于局部相互作用，以及跳跃步长需经验调参。

---

## 168. Riemannian Optimization for Hadamard Products of Low-Rank Matrices

**arXiv ID:** 2606.01216 | [PDF](https://arxiv.org/pdf/2606.01216v1)

**作者:** Pratik Jawanpuria `[一作]` (Indian Institute of Technology Bombay), Bamdev Mishra `[通讯]` (Microsoft)

**通讯引用:** 1955 | [OpenAlex ID](https://openalex.org/A5023551959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将 Hadamard 乘积的低秩矩阵学习问题转化为 Riemannian 商流形上的优化，并提出一种块对角 Riemannian 度量，利用该度量设计了无需超参数的 Riemannian 梯度下降（RGD）算法。

**💡 创新点**

创新点在于：① 针对 Hadamard 乘积的特殊对称性（GL 变换 + 交叉标度），构造了尺度不变且 GL 不变的度量；② 该度量的块权重与 BCD 的正规方程矩阵一致，既保留了局部曲率信息，又保持计算稀疏；③ 通过 Gauss‑Newton 推导的自适应步长实现无调参；④ 将此框架推广至任意光滑损失，兼容现有的 Riemannian 求解器。

**🔧 技术方法**

使用技术包括：Riemannian 商流形理论、块对角拉普拉斯度量、Gauss‑Newton 近似、Armijo 回溯、块级 r×r 线性求解、稀疏矩阵操作及自动微分（实现中使用 NumPy/SciPy 与 Pymanopt）。

**📊 数据集**

实验数据集包括：
- 三个全观测真实矩阵（Camera、Football、Les Misérables）
- 2,000×1,000 的合成矩阵
- MovieLens‑1M（6,040×3,706，约 4.5% 观测率）。

**📈 对比分析**

与 BCD（块坐标下降）和 AGD（欧氏梯度下降）比较：
- 在最小二乘任务上，RGD 与 BCD 近乎等价；
- 在 MovieLens‑1M 的固定参数预算（r₁+r₂=6）下，RGD 在所有 Hadamard 组合（(1,5),(2,4),(3,3)）均取得最低 RMSE；
- RGD 对尺度变换完全不敏感，且在噪声与条件数变化下表现稳健；
- AGD 在非二次损失或高秩时收敛速度慢、精度差。

**⚠️ 局限性**

局限性包括：
- 每次迭代的计算复杂度为 O(Nr + (m+n)r⁴)，相比 BCD 在最小二乘场景下的 O(Nr²) 更高；
- 对极大秩 r 的场景，块级 r×r 求解和 r⁴ 成本可能成为瓶颈；
- 目前实现主要针对光滑损失，若需处理非光滑正则化需进一步改造；
- 对极稀疏数据或超大规模矩阵，内存与时间仍受限。

---

## 169. CoSTL: Comprehensive Spatial-Temporal Representation Learning for Moment Retrieval and Highlight Detection

**arXiv ID:** 2606.01149 | [PDF](https://arxiv.org/pdf/2606.01149v1)

**作者:** Xin Dong `[一作]` (Tsinghua University), Yansong Tang `[通讯]` (Tsinghua University)

**通讯引用:** 98292 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的CoSTL框架，用于视频时刻检索与高亮检测，能够在帧级细粒度视觉信息和视频级时间动态之间进行高效建模。

**💡 创新点**

核心创新在于：①文本驱动的渐进细粒度图像编码器（TDE），通过两步交互逐步提取与查询文本相关的细粒度视觉特征；②多尺度时间感知模块（MTP），在不同时间尺度下捕获宏观与微观时间依赖，融合得到丰富的时空表示。

**🔧 技术方法**

采用视觉Transformer、Q-Former（BLIP‑2）、多头注意力、Transformer编码器、双层卷积预测头以及多尺度下采样与上采样等技术，配合Smooth L1+GIoU、BCE损失等训练策略。

**📊 数据集**

在四个公开基准上验证：QVHighlights、Charades‑STA、TACoS、TVSum。

**📈 对比分析**

与现有最佳方法相比，CoSTL在Moment Retrieval和Highlight Detection任务上均实现显著提升，例如在QVHighlights上平均提升约3.7%，在Charades‑STA的R1@0.5提升2.26%，在TACoS提升0.49%，在TVSum的HIT@1提升6.42%。

**⚠️ 局限性**

局限性包括：对长视频时间跨度仍可能存在性能瓶颈；模型依赖BLIP‑2预训练，若无高质量视觉语言预训练数据则效果受限；以及在小样本类别（如TVSum）上的泛化仍有提升空间。

---

## 170. MindClaw: Closed-Loop Embodied Mental-State Reasoning for Precision Intervention

**arXiv ID:** 2606.01063 | [PDF](https://arxiv.org/pdf/2606.01063v1)

**作者:** Ruoxuan Zhang `[一作]` (Jilin University), Wen-Huang Cheng `[通讯]` (National Taiwan University)

**通讯引用:** 4244 | [OpenAlex ID](https://openalex.org/A5101402074)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了实时闭环的嵌入式心智状态推理框架 MindClaw，支持精确干预。

**💡 创新点**

创新点在于将触发器视为认知技能，分离记忆更新、心理推理和动作生成，实现场景中何时干预何时保持沉默的决策。

**🔧 技术方法**

利用大型语言模型（Qwen3‑4B、GPT‑5.5）配合观测模块、心理推理模块、动作生成模块，并通过技能集合和规则引导触发器决策。

**📊 数据集**

使用 MindPower Benchmark（590 条视频‑文本案例）以及 VirtualHome、ThreeDWorld 等模拟器和静态视频数据。

**📈 对比分析**

与多种直接 VLM 基线（VideoLLaMA3、InternVL、Gemini 等）进行对比，MindClaw 在 Task Accuracy 14.36%、Precision Intervention Accuracy 36.63% 及 Action Satisfaction 100% 上明显优于基线，显示更好的干预时机和行动质量。

**⚠️ 局限性**

主要限制在于对手工收集的技能集合和规则的依赖，且在更复杂或更大规模的环境中泛化性能尚待验证。

---

## 171. Before the Model Learns the Bug:Fuzzing RLVR Verifiers

**arXiv ID:** 2606.01066 | [PDF](https://arxiv.org/pdf/2606.01066v1)

**作者:** Jaideep Ray `[一作]` `[通讯]`, Jaideep Ray

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对可验证奖励（verifiable rewards）中的验证器进行系统性 fuzzing，评估并定位其潜在错误，展示了错误验证器如何产生可被强化学习优化的高收益错误路径，并提出了逐步硬化的验证器改进方法。

**💡 创新点**

创新点在于：①提出了可验证奖励验证器的“bug taxonomy”与差异化评估框架；②基于日志的无源（无源码）差异化指标与度量；③用轻量级 fuzzing 与小规模优化代理演示错误验证器被攻击的结构化性；④公开了可复现的验证器 replay 与硬化 Ablation 工具。

**🔧 技术方法**

主要技术包括：离线 fuzzing（mutation、template search、tabular policy gradient）、差异化日志记录（JSONL）、误差度量（FPR、FNR、exploit‑rate）、验证器硬化 ablation、对开源验证器的 replay、Black‑box 预算搜索。

**📊 数据集**

使用的“数据集”是人工构造的任务集合：①算术最终答案（Math） 200 个案例；②工具调用（JSON tool‑call）500 个案例；③代码单元测试（Code）100 个案例；外部 HumanEval‑style 20 个代码案例；以及对三款开源验证器的对照 replay。所有案例均在本地 Python 环境中生成与评估。

**📈 对比分析**

比较方法：对比“buggy”与“strict”两版验证器的 FPR、exploit‑rate、奖励‑正确性差距；在 10 个随机种子下计算均值、标准差与 95% 置信区间；使用模板搜索与小规模政策梯度优化演示优化如何放大错误验证器的奖励；在有限查询预算内进行黑盒搜索实验。实验表明：buggy 验证器的 FPR 可达 0.5–0.87，优化代理能够在多轮中将奖励提升至 1.0 而正确率仍低；而严格验证器则保持 FPR 为 0 并且奖励与正确率同步。

**⚠️ 局限性**

局限性包括：①实验仅覆盖简化的算术、工具调用和代码任务，未评估完整 RLVR 训练过程；②对开源验证器的 replay 受限于格式兼容性，可能低估语义错误；③fuzzer 采用固定模板与单轮自适应变换，未模拟完整 LLM 输出分布；④优化实验仅在模板池上验证，未检验对神经网络 fine‑tune 的推广；⑤代码验证器在受限 sandbox 下运行，无法覆盖恶意或复杂脚本。

---

## 172. Repeated Descent: A Framework for Online Budget-Feasible Auctions

**arXiv ID:** 2606.01142 | [PDF](https://arxiv.org/pdf/2606.01142v1)

**作者:** Andreas Charalampopoulos `[一作]` (Columbia University), Thanos Tolias `[通讯]` (National Technical University of Athens)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5117111324)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了“重复下降”机制，用自适应线性报价在随机顺序下实现在线预算约束的序列性报价采购拍卖。

**💡 创新点**

首次在单调或非单调子模取值下得到常数（或多项式）竞争比，并给出 XOS 情形下不可达常数竞争比的下界，显著突破之前的 O(log n) 或巨量常数结果。

**🔧 技术方法**

利用自适应阈值递减、线性报价、随机子样本、负相关性与浓缩不等式等技术构造并分析机制。

**📊 数据集**

未使用实际数据集，全部为理论构造与数学证明。

**📈 对比分析**

与之前的 O(log n) 或大常数竞争比的报价机制相比，单调子模的竞争比从约 200 提升至 1046，非单调子模提升至 181 000，均通过严谨的理论证明验证。

**⚠️ 局限性**

机制实现复杂、常数较大，仅在大市场条件下适用，对 XOS 只能给出下界，且未给出实际实验评估。

---

## 173. LongAttnComp: Cross-Family Context Compression for Long-Context Reasoning

**arXiv ID:** 2606.01336 | [PDF](https://arxiv.org/pdf/2606.01336v1)

**作者:** Mengmeng Ji `[一作]` (SambaNova Systems), Chen Wu `[通讯]` (SambaNova Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于细粒度交叉注意力的长上下文压缩框架 LongAttnComp，并通过两阶段微调实现对长文本检索与推理的高效压缩。

**💡 创新点**

创新点包括：① 将 AttnComp 的文档级评分迁移到固定大小的 token 级块；② 用 token 预算 top‑p 选择，兼具累计分数与纯预算两种模式；③ 通过位置重排序保持语篇连贯；④ 设计无模板的查询解析器；⑤ 采用两阶段训练（先构建通用检索基座，再加入多跳推理数据）显著提升多跳推理性能。

**🔧 技术方法**

核心技术：冻结 Llama‑3.1‑8B 前13层的 LLM 加上可训练的交叉注意力层；token 级块划分与分数计算；token‑预算 top‑p 选择；位置重排序；无模板查询解析；两阶段微调策略。

**📊 数据集**

Stage‑1 训练集：32k 例子，SQuAD（16k）+ HotpotQA（16k）；Stage‑2 训练集：20k 例子，MuSiQue（8k）+ 2WikiMultiHopQA（4k）+ SQuAD/HQ 重放（8k）。

**📈 对比分析**

与全上下文基线和 Speculative Prefill 进行对比。On InfiniteBench Code‑Debug，Stage‑1 已超过全上下文精度，Stage‑2 (subq) 在四个目标模型中均达到或超过全上下文；在 LongBench v2 上，Stage‑2 通过多跳数据恢复 7–12% 分数，显著优于 Speculative Prefill。总体表现显示压缩后可保持甚至提升推理精度，且跨模型族迁移良好。

**⚠️ 局限性**

局限性：① 训练数据为合成 NIAH‑style，缺少自然长文档的多样性；② 压缩效果对任务敏感，需调节块大小、查询长度与选择模式；③ 查询解析采用简单的后 N 词切分，未覆盖所有输入结构；④ 仅在 Llama‑3.1‑8B backbone 上验证，未探究其他规模或族的适用性；⑤ 未进行完整的实时效率评估。

---

## 174. A Per-Component Diagnostic Protocol for Neural HJB-PIDE Solvers under Control-Dependent Lévy Jumps

**arXiv ID:** 2606.01122 | [PDF](https://arxiv.org/pdf/2606.01122v1)

**作者:** R. Drissi `[一作]` `[通讯]`, R. Drissi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对残差训练神经 HJB‑PIDE 求解器的五步诊断协议，用于检测并纠正控制相关 Lévy 跳跃下的非局部积分错误。

**💡 创新点**

创新点在于构建可跨模型的逐组件对比与独立参考实现的诊断流程，并首次在 Variance Gamma 基准中发现并修复重要性采样比例错误。

**🔧 技术方法**

采用残差训练的物理信息神经网络、Monte Carlo 重要性采样评估非局部积分、自动微分、以及与独立有限差分和半解析基准的对比。

**📊 数据集**

主要使用 CRRA‑Merton‑Variance Gamma 理论基准（给定参数）和 S&P 500 2010‑2023 日收盘价的合成 VG 拟合。

**📈 对比分析**

方法通过将神经求解器与两套独立 FD‑PIDE 实现及半解析基准在相同截断、网格下比较，误差在约 2% 内；表面误差对神经网络高达 26%，对 FD 仅 0.05%，训练耗时约 170–220 s，FD 约 11 s。

**⚠️ 局限性**

局限性：仅在单一常系数 CRRA‑Merton‑VG 问题上验证，未展示在高维或非齐次 Lévy 过程中的泛化；神经网络表面精度低于 FD，且缺乏收敛理论。

---

## 175. Consistent and Distinctive: LLM Benchmark Efficiency via Maximum Independent Set Prompt Selection on Similarity Graphs

**arXiv ID:** 2606.01400 | [PDF](https://arxiv.org/pdf/2606.01400v1)

**作者:** Denica Kjorvezir `[一作]` (Jožef Stefan Institute), Tome Eftimov `[通讯]` (Jožef Stefan Institute)

**通讯引用:** 2457 | [OpenAlex ID](https://openalex.org/A5082115266)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了基于最大独立集（MIS）的图论方法，对LLM benchmark中的提示进行去冗余与多样子集选取，从而在显著降低评估成本的同时纠正覆盖偏差。

**💡 创新点**

创新点在于将MIS框架应用于自然语言提示的子集选择，并系统验证其在多种嵌入、距离度量与阈值设置下保持排名一致且可显著减少提示数量的能力。

**🔧 技术方法**

所用技术包括图论中的最大独立集求解器（CPLEX、GREEDY、Online-MIS、ReduMIS）、多种句子嵌入模型（BGE、E5、GIST、Nemotron、Qwen-4B、Qwen-8B）、余弦/皮尔逊/标准欧氏距离以及基于百分位的阈值控制。

**📊 数据集**

实验使用了HELM能力排行榜中的四个单轮benchmark（GPQA、IFEval、MMLU-Pro、Omni-MATH）和截至2025年5月的66种LLM。

**📈 对比分析**

通过Spearman ρ和Kendall W将MIS子集排名与全benchmark排名比较，结果显示约84%的配置ρ≥0.95，W≥0.90的比例达99.2%，并且在p80阈值下可实现约37–45%的提示削减同时保持高度一致。

**⚠️ 局限性**

局限性包括：嵌入模型对语义相似性的假设可能不适用于所有子域，低阈值导致过密图可能导致排名偏差；方法需大量GPU内存；仅在英文单语基准上验证，未涵盖多语言或代码基准。

---

## 176. What Makes a Strong Model? A Unified Spectral Analysis of Knowledge Transfer over High-dimensional Linear Regression

**arXiv ID:** 2606.01292 | [PDF](https://arxiv.org/pdf/2606.01292v1)

**作者:** Wendao Wu `[一作]`, Cong Fang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在高维线性回归框架下，通过谱分析统一研究了教师-学生知识迁移（KD、W2S、SD）的有效性，揭示了谱视野扩展和谱去噪两大机制。

**💡 创新点**

创新点在于提出“谱视野扩展”和“谱去噪”这两种普适机制，解释了从强教师到弱学生、弱教师到强学生以及同构学生的表现差异，并给出了精确的风险分解与泛化界。

**🔧 技术方法**

核心技术包括：SGD 动态分析、谱分解、有限样本风险分解、相对谱衰减假设、以及对教师与学生特征空间兼容性的严格定义。

**📊 数据集**

使用了合成数据（正态特征、噪声标签）以及真实数据集 UTKFace 进行年龄回归，实验中涉及 ViT、ResNet 等多种预训练模型的特征提取。

**📈 对比分析**

实验通过 DER（知识蒸馏效率比）和 PGR（性能恢复率）与传统直接训练对比，结果显示在谱衰减差距较大时 DER>1，弱教师到强学生时 PGR>0，验证了理论预测，且强学生在低维任务上可完全恢复教师性能。

**⚠️ 局限性**

局限性包括：理论基于线性回归与子高斯特征，假设谱衰减满足特定幂律；对非线性深度网络的直接推断有限；实验多集中于视觉回归任务，跨任务泛化需进一步验证。

---

## 177. When Hard Negatives Hurt: Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval

**arXiv ID:** 2606.01304 | [PDF](https://arxiv.org/pdf/2606.01304v1)

**作者:** Zhicheng Zhang `[一作]` (Tsinghua University), Zhaocheng Du `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造了面向检索任务的生成-判别桥接方法，利用链式思考指导的反事实扰动产生高质量硬负样本，并在训练时通过查询视角熵最大化抑制源依赖捷径。

**💡 创新点**

创新点在于①阐明并正式化生成-判别差距，发现生成性无意识、源相关捷径两大失效模式；②提出两阶段解决方案：CoT 反事实扰动将“为什么正样本符合查询”拆解为信息需求并针对性违背，③通过熵正则化消除生成样本与真实负样本的分布可区分性，从而恢复对判别边界的有效学习。

**🔧 技术方法**

使用的技术包括大语言模型（GPT‑4.1）生成对抗负样本、链式思考（CoT）推理拆解信息需求、对抗式反事实生成、InfoNCE 对比学习、熵最大化正则化、HDBSCAN 聚类分析与梯度对齐监测等。

**📊 数据集**

在四个检索基准上进行实验：中文版 MS‑MARCO（mMARCO‑zh）、多跳推理任务 HotpotQA、开放域问答 NQ 与 TriviaQA（TQA）。

**📈 对比分析**

与仅用矿工负样本、原始生成负样本、SyNeg（混合矿工与生成负样本）以及仅用 CoT 生成负样本的基线相比，提出的方法在所有四个数据集上均实现 NDCG@10 提升，平均提升 1.90 分相对矿工负样本，3.50 分相对 SyNeg，表现出最稳健且最显著的性能增长。

**⚠️ 局限性**

局限性包括：①对大规模 LLM 的生成依赖，生成成本高；②若生成负样本质量仍不足或分布差距未完全抑制，梯度漂移仍可能出现；③对不同语言或小规模语料库的迁移性尚待验证；④在极端稀疏或高语义相似度场景下，生成负样本可能仍不足以逼近真正的判别边界。

---

## 178. GPTQ-intrinsic LoRA: A Near-optimal Algorithm for Low-precision Quantization with Low-rank Adaptation

**arXiv ID:** 2606.01412 | [PDF](https://arxiv.org/pdf/2606.01412v1)

**作者:** Shihao Zhang `[一作]` (University of California San Diego), Rayan Saab `[通讯]` (University of California San Diego)

**通讯引用:** 1654 | [OpenAlex ID](https://openalex.org/A5055778695)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文先给出了低精度+低秩量化的理论下界，并提出了基于GPTQ的训练无关算法 GPTQ‑Intrinsic LoRA 与进一步的量化细化方法 Bid‑Up；随后在 Qwen3 语言模型和 DeiT 视觉变压器上进行实验，验证算法在低比特量化下的性能提升。

**💡 创新点**

创新点：
- 首次给出低精度加低秩近似的 信息理论下界，揭示压缩与误差的根本关系；
- 设计了 GPTQ‑Intrinsic LoRA，将低秩校正直接嵌入 GPTQ 的量化流程，既保持无训练特性又实现接近下界的误差；
- 提出 Bid‑Up 坐标更新策略，可与低秩补偿 OLrC 交替迭代，保证层级重构误差递减；
- 在理论与实验层面证明 GPTQ‑Intrinsic LoRA 在尺度上与下界相匹配。

**🔧 技术方法**

技术方法：
- GPTQ（Greedy PTQ with Cholesky）量化框架；
- LoRA（低秩适配）与特征提取因子 V_r 的选取；
- 增广 Hessian 与 QR+特征值分解实现的无逆运算；
- 最小二乘和误差扩散解析；
- 随机舍入与定点量化；
- Bid‑Up 的坐标更新和 OLrC 的最优低秩补偿。

**📊 数据集**

数据集与模型：
- 语言模型：Qwen3-0.6B 与 Qwen3-1.7B，使用 WikiText‑2 作为校准与测试集；
- 视觉模型：DeiT‑B 与 DeiT‑III‑L，在 ImageNet‑2012 上进行 top‑1 精度评估。

**📈 对比分析**

对比方法与性能：
- 基线为 GPTQ 与 GPTQ+OLrC；
- 3‑bit 与 4‑bit 权重量化下，GPTQ‑Intrinsic LoRA 在 Qwen3 的 perplexity 下降约 15‑20%，在 DeiT 的 top‑1 下降仅 3‑4%；
- 加入 Bid‑Up + OLrC 迭代可进一步提升 1‑3% top‑1 ；
- 与传统 GPTQ 的精度差距显著，并在低秩参数 r 取 16、32、64 时保持优异。

**⚠️ 局限性**

局限性：
- 仅考虑权重量化，未对激活或 KV 缓存量化给出理论与实验；
- 依赖大量校准样本，若校准不足可能影响误差；
- 理论下界基于非尖峰矩阵与特定的低秩假设，实际模型可能偏离；
- 目前未结合微调或量化后训练，无法评估更大范围的细粒度性能；
- 对超大 rank 或非常低 bit‑width 的适应性仍待验证。

---

## 179. Privacy-Preserving Smart Surveillance with Cross-Dataset Violence Detection and Decentralized Evidence Governance

**arXiv ID:** 2606.01225 | [PDF](https://arxiv.org/pdf/2606.01225v1)

**作者:** Hasan Coşkun `[一作]` (Kadir Has University), Vesna Dimitrova `[通讯]` (Ss. Cyril and Methodius University in Skopje)

**通讯引用:** 204 | [OpenAlex ID](https://openalex.org/A5046161546)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套隐私保护的智能监控框架，先通过轻量级模型检测暴力事件并即时加密录制片段，随后通过阈值式秘密共享与多方投票控制是否解密展示。

**💡 创新点**

核心创新在于将事件检测与证据披露解耦，利用 Shamir 秘密共享与多因素身份验证实现去中心化的证据治理，并提供完整的可审计工作流。

**🔧 技术方法**

技术包括 MobileNetV2 + BiLSTM/Temporal CNN 的视频分类器、AES‑256‑CBC 加密、Shamir 秘密共享（p = 2^521‑1）、RSA‑OAEP 公钥加密、JWT + TOTP 双因素投票与 PostgreSQL 审计日志。

**📊 数据集**

使用公开的 SCVD、RWF‑2000 和 Real‑Life Violence Situations 三大数据集，共 6105 条视频片段，涵盖城市监控与真实场景的暴力与非暴力标签。

**📈 对比分析**

采用七种单/多源训练与跨域测试策略，评估指标为准确率与 ROC‑AUC。最优的 MobileNetV2+BiLSTM 在全部三源合并测试集上达到 93.5% 准确率与 0.980 ROC‑AUC，但对 RWF‑2000 的单源表现仍低，表明数据集漂移依旧存在。

**⚠️ 局限性**

限制包括：仅用三大数据集无法覆盖所有真实摄像头多样性；仅做二分类，无法识别多种事件类型；未在实时流环境下充分验证时延与稳定性；加密与投票实现为原型级，缺乏生产级安全与合规保障。

---

## 180. Understanding Undesirable Attributes of Requirements Engineers: Insights from Practitioners

**arXiv ID:** 2606.01370 | [PDF](https://arxiv.org/pdf/2606.01370v1)

**作者:** Larissa Barbosa `[一作]` (Federal University of Bahia), Rita S. P. Maciel `[通讯]` (Federal University of Bahia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过问卷和访谈调查，识别并归纳了17种不良属性，构建了四类（沟通问题、领域知识不足、人格特质、技术知识不足）的概念图，帮助需求工程师反思并改进工作。

**💡 创新点**

创新点在于首次系统性探讨需求工程师的不良属性，并将其可视化为概念地图，指出这些属性与沟通、关系、业务理解紧密相关，强调了软技能在需求工程中的重要性。

**🔧 技术方法**

使用的技术包括：结构化问卷设计、半结构化访谈、开放式编码分析、概念图绘制，三位研究人员独立编码并协商一致。

**📊 数据集**

数据集为18名巴西软件行业从业者（包括软件工程师和项目经理）在问卷中的回答与11名受访者的访谈记录，总体经验年限平均10年，涉及不同规模公司。

**📈 对比分析**

论文并未使用传统性能指标，而是通过与先前研究的“可取属性”对比，展示两者在沟通、协作、适应性等维度的对照，说明不良属性并非简单的负面反面，而是专业行为的不同维度。

**⚠️ 局限性**

局限性主要是样本量有限、仅来自巴西软件行业，可能存在地域与文化偏差；此外，部分属性在访谈中未得到完整定义，概念图不完备。

---

## 181. CEAR: Certified Ensemble Adversarial Robustness in DNNs

**arXiv ID:** 2606.01437 | [PDF](https://arxiv.org/pdf/2606.01437v1)

**作者:** Daniel Sadig `[一作]` (Toronto Metropolitan University), Reza Samavi `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 1221 | [OpenAlex ID](https://openalex.org/A5026763818)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

CEAR提出一种融合经验和认证防御的集成框架，利用变噪声高斯增强、温度蒸馏与噪声logits，以及两种投票机制（几何中位投票与鲁棒加权投票），显著提升深度网络对白盒攻击的可证明鲁棒性。

**💡 创新点**

创新点包括：①将VGA、DTS与噪声logits三种防御在同一集成体系中联合；②根据预测置信度动态选择几何中位或鲁棒加权投票；③将随机平滑方法推广到集成模型并进行鲁棒半径验证。

**🔧 技术方法**

使用技术主要有：变噪声高斯数据增强（VGA）、温度蒸馏（DTS）、噪声logits、几何中位投票、鲁棒加权投票、随机平滑与蒙特卡罗估计。

**📊 数据集**

实验数据集包括MNIST、CIFAR-10和TinyImageNet。

**📈 对比分析**

与RandSmooth和SWEEN基线对比，CEAR在小扰动下的几何中位投票取得最高认证准确率；在大扰动下的鲁棒加权投票显著提升鲁棒半径和认证准确率，整体性能超过所有基线，并且攻击成功率明显下降。

**⚠️ 局限性**

局限性在于需要训练并维护多个子网络，导致计算与内存成本上升；鲁棒性对噪声水平和投票策略的选择敏感；对极大扰动或更高维度图像的泛化仍存在挑战。

---

## 182. Linear Strategic Classification with Endogenous Improvements

**arXiv ID:** 2606.01198 | [PDF](https://arxiv.org/pdf/2606.01198v1)

**作者:** Siddharth Shrivastava `[一作]` (IIT Hyderabad), Ganesh Ghalme `[通讯]` (IIT Hyderabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在特征可被修改且标签可因改进而改变的战略分类问题，并提出在单指数资格模型下，战略最优线性分类器是对Bayes决策边界的平行平移；同时给出了相应的PAC学习保证与一个基于oracle标签的Plug‑in学习算法。

**💡 创新点**

创新点在于：①证明在单指数模型与线性可分解成本下，战略最优分类器仅需在Bayes阈值上加上β·max_i w_i/α_i即可得到；②指出该策略相较于传统Bayes分类器能更好地近似改进意识下的目标；③给出PAC统一收敛和Plug‑in算法的理论保证。

**🔧 技术方法**

使用的技术包括单指数资格模型、线性可分解（one‑sided ℓ₁）成本、最佳响应分析、VC 维数的统一收敛、hinge 损失的改进版本、梯度下降训练、oracle 标签模拟以及插件估计器。

**📊 数据集**

实验数据集包括成人收入（Adult）、HELOC 信贷风险、法学院考试（Law School）、ACS 收入以及一个合成数据集。

**📈 对比分析**

通过与传统SVM、SERM、Attias等基线方法对比，采用改进率、操纵率等指标评估。实验结果显示，所提 Strat‑Imp‑Aware 算法在四大数据集上均取得最高的改进率，且在大样本或稀缺样本场景下优于对手。

**⚠️ 局限性**

局限性在于：仅适用于单指数资格模型、线性特征与决策器、线性可分解成本、单次交互以及固定最佳响应规则；未考虑非线性模型、ℓ₂ 等成本、异质或重复互动、成本敏感目标及因果改进机制等更复杂的现实情境。

---

## 183. R^3: Composed Video Retrieval via Reasoning-Guided Recalling and Re-ranking

**arXiv ID:** 2606.01113 | [PDF](https://arxiv.org/pdf/2606.01113v1)

**作者:** Zixu Li `[一作]` (Shandong University), Liqiang Nie `[通讯]` (Harbin Institute Of Technology)

**通讯引用:** 30023 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个零射击的组合视频检索管道 ℝ^3，整合推理、召回与重排序三大模块。

**💡 创新点**

创新点在于：①将编辑指令先转化为推理追踪并以此构造增强检索查询；②使用同意门+残差融合机制在检索阶段平衡推理与原始查询；③整个流程完全基于冻结的 Qwen3-VL 组件，无需额外训练。

**🔧 技术方法**

采用 Qwen3‑VL‑Thinking 生成推理追踪、Qwen3‑VL‑Embedding 进行基于向量的粗粒度检索、Qwen3‑VL‑Reranker 进行候选的精细对比重排序，并结合 prompt 设计与同意门残差融合。

**📊 数据集**

在 CoVR‑R 评测基准上，使用 WebVid 和 Something‑Something‑V2 作为检索图库与查询集。

**📈 对比分析**

对比单一嵌入检索基线，推理增强提高 R@1 0.34，重排序提升 R@1 3.70；最终 ℝ^3 在验证集达 95.44% R@1、99.96% R@50，测试集达到 98.82% R@1、100% R@5+，显著优于基线。

**⚠️ 局限性**

局限性包括：推理可能产生不受支持的细节导致检索误差；仍需冻结大模型，推理与重排序耗费显著计算资源；在极其复杂或模糊的编辑指令下，推理与检索的协同效果有限。

---

## 184. Dive into Ambiguity: A*-Inspired Multi-Agents Commonsense Obfuscation Attack on LLM Prompts

**arXiv ID:** 2606.01441 | [PDF](https://arxiv.org/pdf/2606.01441v1)

**作者:** Boxuan Wang `[一作]` (University of Liverpool), Yi Dong `[通讯]` (University of Liverpool)

**通讯引用:** 82936 | [OpenAlex ID](https://openalex.org/A5032573962)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于A*启发式搜索、多智能体改写与机制归因的事实错误诱发框架，用于系统性生成能够诱发LLM常识性幻觉的对抗提示。

**💡 创新点**

创新点在于：①引入A*启发式成本函数，精确引导提示改写；②通过层次化改写策略和多智能体辩论机制动态调节改写强度，解决语义坍塌问题；③设计自动机制归因（AML）框架，对生成的对抗提示进行可解释标签并反馈提升攻击效果。

**🔧 技术方法**

使用的技术包括：A*搜索启发式（语义相似度、编辑距离、输出差异三维组合）；语义坍塌理论与γ基调度器；多智能体辩论（MAD）改写；自动机制标签（AML）与迭代优化；基于语义嵌入的聚类与距离度量。

**📊 数据集**

实验数据集涵盖128条手工标注的常识QA对，并使用CommonsenseQA、CosmosQA、mCSQA、CommonsenseQA 2.0等公开基准。模型涉及闭源API（GPT‑4.1、GPT‑4o、GPT‑3.5‑turbo）与开源模型（Qwen2.5‑7B、LLaMA2‑13B）。

**📈 对比分析**

与随机提示、仅编辑、聚类、输出导向等基线对比，A*启发式+多级改写在闭源模型@20成功率最高（GPT‑4o 44.9%），开源模型（LLaMA2‑13B 75.8%）。AML框架提升单机制注入成功率至约62.9%，但多机制叠加效果递减。整体而言，该方法在不同模型与配置下均优于传统随机或单维度优化策略。

**⚠️ 局限性**

局限性包括：①对抗提示生成仍依赖大量LLM调用，算力成本高；②语义保持仅通过余弦相似度衡量，可能不足以保证语义完整；③多智能体策略在不同模型间效果差异较大，需进一步自适应；④AML标签的可解释性和覆盖率仍有提升空间，尤其对复杂抽象机制。

---

## 185. Toward Reliable Semantic Communication: Beyond Average Performance

**arXiv ID:** 2606.01284 | [PDF](https://arxiv.org/pdf/2606.01284v1)

**作者:** Boyuan Li `[一作]` (Shenzhen University), Ying-Jun Angela Zhang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 15778 | [OpenAlex ID](https://openalex.org/A5004874287)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

综述了可靠语义通信的三类设计思路（基于通道自适应、鲁棒性编码与 HARQ 重传），并分别提出了两种改进方案：1）HANA‑JSCC 通过知识蒸馏提升对不完美 CSI 的鲁棒性；2）S3CHARQ 在源-信道-检查编码框架下利用强化学习自适应重传，提升图像重构的平均与尾部性能。

**💡 创新点**

① 把通道自适应、鲁棒性与重传三种机制统一视角，明确其互补与缺陷；② 在 HANA‑JSCC 中首次结合知识蒸馏来消除 CSI 匹配误差；③ 在 S3CHARQ 中首次实现源-信道-检查编码的联合优化并加入 RL 自适应重传决策，实现平均和尾部 PSNR 双重提升。

**🔧 技术方法**

深度学习联合源信道编码（JSCC），知识蒸馏技术，用于训练对误差的鲁棒性；强化学习（RL）用于自适应重传决策；通道状态信息（CSI）自适应调度与检查编码结合。

**📊 数据集**

CIFAR‑10 图像数据集（用于评估图像重构的 PSNR 与尾部 PSNR），并在 HANA‑JSCC 评估中使用与 SwinJSCC、DeepJSCC‑MIMO 相同的图像/通信基准。

**📈 对比分析**

HANA‑JSCC 与 SwinJSCC、DeepJSCC‑MIMO 在多种 SNR 与 CSI 误差水平下比较，结果显示 HANA‑JSCC 在平均 PSNR 上持续领先；S3CHARQ 与单轮、进化重传、相同重传等方案对比，S3CHARQ 在所有 SNR 区间均实现最高平均 PSNR，并在 97% 分位 PSNR（尾部性能）上明显优于其他方案。

**⚠️ 局限性**

仍缺乏对鲁棒性与重传机制的完整联合设计；重传方案引入额外的通信与延迟开销；实验仅基于 CIFAR‑10 等小规模图像数据，未验证在更大规模或真实通信系统中的可扩展性。

---

## 186. Schema-Agnostic Knowledge Graph Construction via Hybrid Ontology Discovery for Cyber Threat Intelligence

**arXiv ID:** 2606.01208 | [PDF](https://arxiv.org/pdf/2606.01208v1)

**作者:** Seonwoo Kim `[一作]` (Ministry of National Defense), Insup Lee `[通讯]` (Korea University)

**通讯引用:** 13936 | [OpenAlex ID](https://openalex.org/A5030456600)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Schema‑agnostic的CTI知识图构建框架，结合LLM与混合本体发现（embedding搜索+递归导航）以及闭环SHACL校验，实现对大型本体（如UCO）的高质量、结构化提取，且支持本地LLM部署以满足隐私要求。

**💡 创新点**

创新点在于：①将提取管道与本体解耦，支持任意OWL/SHACL本体；②通过混合本体发现动态定位相关子图，避免prompt中全本体导致的上下文消耗和误识；③引入SHACL闭环验证，减少假设映射，显著降低schema违规率；④证明本地开源LLM配合该机制可达近似企业LLM的性能，突破隐私限制。

**🔧 技术方法**

采用的技术包括：大型语言模型（Qwen3.5‑35B、Gemma‑4‑26B、GPT‑5.4‑mini、Claude‑Haiku‑4.5）；embedding‑based semantic search；层级递归导航（LLM引导）以处理复杂本体结构；SHACL约束验证及闭环自纠；MCP兼容的工具接口；vLLM本地服务；实体/谓词阈值控制；知识图序列化为JSON/Neo4j Cypher。

**📊 数据集**

使用CTINexus公开基准（149份CTI报告）并人工扩充UCO、STIX‑2.1的ground‑truth类型；评估本体规模包括UCO（419类、578属性）、STIX‑2.1（109类、311属性）和MALOnt（75类、12属性）。

**📈 对比分析**

通过与TTPDrill、CTINexus、LLM4CTI等基线比较，实体F1在UCO为0.7371（提升62.5%）、STIX为0.8724、MALOnt为0.6942；谓词F1在UCO为0.5484（提升29.5%）、STIX为0.5860、MALOnt为0.5412；非合规率降至5.2%（低于基线43.9%）。本地LLM与企业LLM对比显示，配合混合发现后本地模型可保持99.2%（实体）和97.8%（谓词）的性能。

**⚠️ 局限性**

局限性包括：对静态本体的依赖，无法为新出现的概念自动生成类；阈值设置需人工调优；递归导航在大本体下的计算开销仍不低；仍存在少量误映射或缺失属性的情况；在极高复杂度场景下对本地LLM的推理能力有限。

---

## 187. Learning-based Directed Graph Abstraction of Combinatorial Spaces for Order-Preserving Search in Mixed-Combinatorial Nonlinear Optimization

**arXiv ID:** 2606.01425 | [PDF](https://arxiv.org/pdf/2606.01425v1)

**作者:** Gishnu Madhu `[一作]` (University at Buffalo), Souma Chowdhury `[通讯]` (University at Buffalo)

**通讯引用:** 3743 | [OpenAlex ID](https://openalex.org/A5074202796)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种基于图神经网络的顺序保持组合推荐模型 GNN‑NavCo，学习组合空间的能量梯度并在混合组合非线性优化中提供梯度感知导航，从而提升连续变量优化器的搜索效率和最终解质量。

**💡 创新点**

创新点包括：①将组合空间抽象为具有能量差异边的有向图，利用边场学习实现梯度感知导航；②引入循环一致性正则化保证预测的边场可积分；③将该模型集成到分解式 MCNLP 求解框架中，使得非线性混合组合问题在保持离散结构的同时获得梯度式引导。

**🔧 技术方法**

使用技术主要包括：Edge Field Graph Network（EFGN）——基于图神经网络的编码-解码框架；pairwise difference regression 与 Huber 损失；循环一致性正则化；子图采样和能量差异计算；以及与 PSO、GA 等搜索算法的结合；对约束问题使用罚函数。

**📊 数据集**

数据集主要来自 MINLPLib 的三组基准问题：cvxnonsep_psig20、cvxnonsep_psig40 与 cvxnonsep_normcon40，分别构造 101、501、1001 条组合集，并通过 Latin Hypercube Sampling 对连续变量进行采样。

**📈 对比分析**

与传统索引化组合的 MDPSO/GA 进行多次对比实验；结果表明 GNN‑NavCo 版在大多数实验中收敛更快、最终目标误差更低、方差更小，尤其在大组合空间（501/1001）和约束问题中表现突出。

**⚠️ 局限性**

局限性包括：①在组合数极大时子图尺寸不足导致性能下降，需要进一步优化子图采样策略；②目前仅与 PSO、GA 等离散优化器结合，梯度法兼容性待验证；③模型训练和推理仍需额外计算资源，且对连续上下文的依赖性较强。

---

## 188. Deft Scheduling of Dynamic Cloud Workflows with Varying Deadlines via Mixture-of-Experts

**arXiv ID:** 2606.01162 | [PDF](https://arxiv.org/pdf/2606.01162v1)

**作者:** Ya Shen `[一作]` (Victoria University of Wellington), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**通讯引用:** 32484 | [OpenAlex ID](https://openalex.org/A5100400258)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了 DEFT，一种基于 Mixture-of-Experts 的深度强化学习调度策略，用于在动态云环境下智能分配具有图结构的工作流，满足不同紧迫度的服务级别协议。

**💡 创新点**

创新点在于：①首次将 MoE 引入云工作流调度，专门为不同截止紧迫度训练专家；②设计了图自适应门控网络，利用工作流 DAG、任务状态与 VM 条件进行跨注意力专家选择；③采用两阶段预训练 + 门控微调的训练策略。

**🔧 技术方法**

采用的技术包括：图注意力网络 (GAT) 进行 DAG 编码、交叉注意力实现专家路由、深度强化学习（OpenAI-ES）训练、Mixture-of-Experts 结构以及跨任务状态嵌入。

**📊 数据集**

使用了公开的 CADWS 仿真平台，包含四种典型工作流模式（CyberShake、Montage、Inspiral、SIPHT），并按任务数量划分为 S/M/L 三种规模，工作流到达采用泊松过程模拟。

**📈 对比分析**

与五个基线（ProLis、GRP-HEFT、ES‑RL、SPN‑CWS、GATES）对比，DEFT 在所有规模下总成本最低，尤其在 M、L 规模上显著优于 GATES（分别下降 11.4% 与 29.6%），并在 VM/违约平衡上实现更佳综合表现；实验还证明图自适应门控比线性/MLP 门控更优。

**⚠️ 局限性**

局限性包括：仅在合成仿真数据上验证，未涉及真实多租户云环境；门控与专家路由增加了少量推理延迟；缺乏对专家选择过程的可解释性分析。

---

## 189. On the Complexity of Recurrence Evaluation

**arXiv ID:** 2606.01175 | [PDF](https://arxiv.org/pdf/2606.01175v1)

**作者:** Artem Parfenov `[一作]` (National Research University Higher School of Economics), Michael Vyalyi `[通讯]` (Federal Research Center Computer Science and Control of the Russian Academy of Sciences)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究有限值递推函数的求值问题，给出了在不同输入表示（unary、binary）下的计算复杂度，证明了相应求值问题为PSPACE‑完整以及对NAND递推的三维游戏求值问题的NP‑或更高难度；

**💡 创新点**

创新点在于将递推求值问题与受限图灵机、边界条件以及非线性NAND运算相结合，构造了通用递推函数，使得求值问题可用于模拟任何受限图灵机，从而给出了递推求值的完整复杂度结果，并首次证明了三维NAND递推游戏的求值问题具有高度复杂性；

**🔧 技术方法**

主要技术包括：图灵机与递推函数的编码转换、对偏移量、边界条件的多种表示（单调、succinct circuit）、对NAND函数的逻辑展开与层到层映射、使用坐标变换和三角形演化分析、以及对log‑time可约简的构造；

**📊 数据集**

未使用任何实验数据集，论文完全基于理论证明与构造性证明；

**📈 对比分析**

由于本研究为理论复杂度分析，没有实验对比或性能评估；主要结果以多项式空间/时间可归约证明为依据；

**⚠️ 局限性**

局限性包括：对普通或厄密子弹游戏的NAND递推求值尚未给出完整复杂度分类；对多维子弹游戏的非负偏移量在常规胜利条件下仍缺乏精确复杂度结果；此外，证明方法未能推广到更一般的NAND递推结构，需进一步研究。

---

## 190. HOLA: Holistic Multi-Modal Alignment for Open-Set 3D Recognition

**arXiv ID:** 2606.01334 | [PDF](https://arxiv.org/pdf/2606.01334v1)

**作者:** Koby Aharonov `[一作]` (Technion – Israel Institute of Technology), Ayellet Tal `[通讯]` (Technion – Israel Institute of Technology)

**通讯引用:** 10916 | [OpenAlex ID](https://openalex.org/A5109259472)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种利用多视角图像与多文本描述联合对齐点云的开源集3D识别方法，并通过分离多正样本对比损失和轻量化文本适配器实现对稀有类别的更好泛化；

**💡 创新点**

主要创新在于：①提出解耦多正样本对比损失（DMP），解决多正样本导致的“聚光灯拥挤”问题；②仅对检索到的网络字幕使用轻量化 MLP 适配器，缩小域差距并提升文本-3D对齐效果；

**🔧 技术方法**

使用CLIP共享嵌入空间、PointBERT/MinkowskiNet 3D 编码器、对比学习（InfoNCE）改进、轻量化 MLP 文本适配器以及多视角图像渲染与多文本描述；

**📊 数据集**

训练数据来自 ShapeNetCore、3D-FUTURE、ABO、Objaverse 等三模组数据集，评估使用 Objaverse-LVIS、ScanObjectNN 和 ModelNet40；

**📈 对比分析**

与 OpenShape、TAMM、VIT-LENS、UNI3D、RECON++ 等方法比较，HOLA 在 Objaverse-LVIS 上实现 2.0% 的零样本 Top‑1 提升，使用 72M 参数且 FPS 最高，显著优于大型 ViT 基模型；

**⚠️ 局限性**

在 ModelNet40 上的 Top‑3/Top‑5 仍略低于 SoTA，主要由于该数据集小且缺乏长尾分布。

---

## 191. Beyond Task Success: Behavioral and Representational Diagnostics for WAM and VLA

**arXiv ID:** 2606.01095 | [PDF](https://arxiv.org/pdf/2606.01095v1)

**作者:** Hung Mai `[一作]` (National Economics University), Tuan Do `[通讯]` (Phenikaa University)

**通讯引用:** 4280 | [OpenAlex ID](https://openalex.org/A5054273812)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套模型无关的诊断框架，通过行为回放诊断和稀疏自编码器特征分析，比较了 Vision‑Language‑Action（VLA）策略与 World‑Action Models（WAM）的行为表现与内部表征。

**💡 创新点**

创新点在于将任务成功率之外的多维诊断指标（动作平滑度、目标物体进展、干扰物稳定性、推理成本）与稀疏自编码器对内部特征的“记忆、反应、预测”标签相结合，系统揭示未来预测在实际操控中的可行动性。

**🔧 技术方法**

主要技术包括：行为回放诊断（动作差分、边界跳跃、低运动失败等度量）、稀疏自编码器（TopK SAE）特征挖掘、基于特征的预测一致性、水平稳定性与动作可预测性评分，及多模型多数据集的对比实验。

**📊 数据集**

使用的公开机器人仿真数据集为 LIBERO（包含空间、目标、长等子任务）和 RoboTwin2.0（双臂qpos目标）。

**📈 对比分析**

对比方法是对七种策略（三种VLA、三类WAM）在动作平滑度、目标进度、干扰物扰动、推理延迟、GPU内存等多维度进行量化。结果显示：WAM在动作平滑、目标选择与干扰抑制方面优于VLA，但推理成本显著更高；在LIBERO上成功率相近时行为指标分化明显；在RoboTwin2.0上WAM仍保持相对平滑，但整体成功率普遍较低。

**⚠️ 局限性**

局限性包括：仅在仿真环境和公开 checkpoint 评估；诊断指标与因果关系不明；稀疏自编码器特征标签为概率性弱注释；模型数量有限，未覆盖所有可能的 WAM 变体。

---

## 192. Domination-Avoiding Learning Agents Cannot Collude

**arXiv ID:** 2606.01275 | [PDF](https://arxiv.org/pdf/2606.01275v1)

**作者:** Noam Nisan `[一作]` (Hebrew University of Jerusalem), Emmanuel Zerah `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过定义并分析“Domination-Avoiding”学习算法，证明在Bertrand Logit双寡头定价博弈中，若代理满足此类避免经验主导策略的特性，则其学习动态会自发收敛至竞争性纳什均衡，不能持续出现合谋行为。

**💡 创新点**

创新点在于提出了比传统平均基学习更广泛、鲁棒的“Domination-Avoiding”框架，并证明该框架下的所有算法均可避免在任意有限博弈中被迭代剔除的严格支配策略，从而排除了协同合谋的可能性；同时展示了外部遗憾最小化算法可实现合谋，而内部遗憾最小化则被排除。

**🔧 技术方法**

主要技术包括：迭代剔除严格纯支配策略（IESPDS）理论、经验支配度量、Upper-Time-Average（UTA）分析、对Bertrand Logit模型的离散网格化以及对多种经典遗憾最小化算法（如MW、FTPL、EXP3、内部遗憾最小化器、情境学习）进行理论证明。

**📊 数据集**

实验数据集使用了Bertrand Logit双寡头的离散价格网格（15个价格点），参数设置为产品质量a=2、外部好处a0=0、边际成本c=1、价格敏感度μ=4，模拟了10⁶个时间步。

**📈 对比分析**

通过与传统外部遗憾最小化算法（MW、FTPL）进行比较，发现后者在此博弈中趋向竞争均衡并不出现合谋，而先前报道的Q‑learning算法会产生超竞争定价；论文通过构造Coarse Correlated Equilibrium展示外部遗憾算法可实现合谋，进一步验证了DA框架的必要性。

**⚠️ 局限性**

局限性包括：仅在对称Bertrand Logit博弈及其离散网格上证明了收敛结果；对更一般非对称或多玩家市场的推广尚未完成；DA框架对“足够聪明”的算法（如基于未来收益的强化学习）仍可能失效，尚需进一步研究如何设计可同时实现协同与防止合谋的算法。

---

## 193. DeepIPCv3: Event-Aware Multi-Modal Sensor Fusion for Sudden Pedestrian Crossing Avoidance

**arXiv ID:** 2606.01277 | [PDF](https://arxiv.org/pdf/2606.01277v1)

**作者:** Oskar Natan `[一作]` (Universitas Gadjah Mada), Jun Miura `[通讯]` (Toyohashi University of Technology)

**通讯引用:** 4961 | [OpenAlex ID](https://openalex.org/A5071725508)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发并评估了DeepIPCv3，一种融合LiDAR点云与动态视觉传感器（DVS）事件流的多模态自主驾驶框架，用于突发行人横穿场景的即时避障；

**💡 创新点**

创新点包括：①使用Transformer式交叉注意力动态融合LiDAR的稠密3D几何与DVS的微秒级事件流，消除传统帧图像的曝光与运动模糊；②在控制层引入混合PID‑MLP策略，实现安全且迅速的决策；③通过微秒级事件实现对突发动态障碍的即时感知与反应；

**🔧 技术方法**

技术实现涵盖：Transformer交叉注意力、多头注意、多模态编码器（EfficientNet-B0、PolarNet）、GRU序列建模、混合PID‑MLP控制、MGN多任务权重自适应、PyTorch端到端训练；

**📊 数据集**

使用自建多模态数据集，采集LiDAR、RGB-D、DVS三传感器同步数据，覆盖日间与夜间两种照明条件，总计37,855帧，训练/验证/测试按6:6:6划分，测试集19,847帧；

**📈 对比分析**

与Huang、AIM‑MT、TransFuser、LMDrive等SOTA模型进行ADE、Steer/Maneuver MAE、RTD等指标比较，DeepIPCv3在昼夜场景下均获得最低ADE与MAE，RTD约0.21s，表现出更快、更准确的避障反应；

**⚠️ 局限性**

局限性包括：事件聚合窗口Δt固定为250 ms，需与LiDAR采样频率同步，窗口大小对性能敏感；目前仅离线评估，缺乏真实路测验证；Transformer模块实现尚未优化到极低延迟，需进一步量化压缩与边缘部署研究。

---

## 194. Upper Bounds on Multiple $b$-Burst Deletion-Correcting Codes

**arXiv ID:** 2606.01245 | [PDF](https://arxiv.org/pdf/2606.01245v1)

**作者:** Chen Wang `[一作]` (Technion Israel Institute of Technology), Tolga M. Duman `[通讯]` (Bilkent University)

**通讯引用:** 6238 | [OpenAlex ID](https://openalex.org/A5048023330)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了多块连续删除（t，b）突发删除码的理论极限，分析了删除球的结构并给出了其大小的上下界。

**💡 创新点**

提出了删除球最大化的紧凑表示法和单调性性质，利用这些结构特征构造了更紧的线性规划和球堆积上界，并在 q 足够大或参数固定的情况下给出了渐近最优结果；此外，对特殊情况 (t=b=2) 进一步优化了上界。

**🔧 技术方法**

主要采用了超图匹配理论、线性规划对偶、球堆积（sphere‑packing）以及组合计数与随机变量分析等方法。

**📊 数据集**

本工作完全为理论分析，不依赖具体数据集；所有结果均为数学推导与渐近分析。

**📈 对比分析**

将新的上界与已知的 Levenshtein、Schoeny 等结果以及最近的插入删除球大小表达式进行比较，证明在一般参数范围内显著优于以往上界；对 (t=b=2) 的情况，给出的改进上界在极限下比之前的上界提高了 q⁻² 的系数。

**⚠️ 局限性**

主要局限在于：1）缺乏能够接近理论极限的构造与高效解码算法；2）在小字母表（尤其是二进制）下，上下界仍存在较大差距；3）对更一般的突发删除与插入模型的进一步扩展尚未完成。

---

## 195. Efficient RAG with Intent-Aware Retrieval and Semantics-Preserving Chunking

**arXiv ID:** 2606.01240 | [PDF](https://arxiv.org/pdf/2606.01240v1)

**作者:** Fachrina Dewi Puspitasari `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112742 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 InSemRAG 框架，结合意图感知检索（IAR）和语义保持分块（SPC），通过迭代检索与检查机制实现检索式生成。

**💡 创新点**

创新点在于：①动态混合检索权重根据查询意图自适应调节；②使用 SLM 进行查询扩展、检索权重计算、碎片检测与修复；③语义保持分块通过邻域重组与压缩，解决碎片化问题；④整体采用轻量 SLM 减少延迟。

**🔧 技术方法**

主要技术包括：小型语言模型（SLM）作为检索器与语义检查器；双视图查询扩展（稠密+稀疏）；动态通道权重与混合检索；分块修复与覆盖审计；迭代检索-检查循环。

**📊 数据集**

使用的基准数据集：NQ‑open、TriviaQA、WebQuestions、HotpotQA、2WikiMultiHopQA（问答）；ELI5（长文本生成）；FEVER（事实验证）。

**📈 对比分析**

与 Naïve RAG、CoT‑on‑Concat、Multi‑Hop RAG、SELF‑RAG 等基线对比，InSemRAG 在多跳与证据敏感任务上提升 2.65 F1（HotpotQA）和 1.5% 准确率（FEVER），在所有评测指标（EM、F1、ROUGE、准确率）上均优于基线；延迟比 Multi‑Hop RAG 低 4×，仅略高于 Naïve RAG。

**⚠️ 局限性**

局限性：假设完整信息仅存在于检索块邻域，因而修复仅在邻域内操作，无法捕获跨文档或远程信息；若 SLM 语义判断失误，可能导致错误修复或检索停滞；迭代过程虽低延迟，但在极大查询复杂度时仍会增加时延。

---

## 196. Ask4VG: Risk-Aware Question Selection for Reducing Prior-Driven Answers in Medical VQA

**arXiv ID:** 2606.01044 | [PDF](https://arxiv.org/pdf/2606.01044v1)

**作者:** Xiaorong Zhu `[一作]` (Tianjin University), Weizhi Nie `[通讯]` (Tianjin University)

**通讯引用:** 3406 | [OpenAlex ID](https://openalex.org/A5001185571)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Ask4VG框架，通过对问题进行对抗性视觉探测估计先验驱动的幻觉风险，并基于风险对问题进行重写选择，从而降低医用VQA中的幻觉风险并提升准确率。

**💡 创新点**

创新点在于将问题本身视为可控因素，利用无标签的对抗性视觉探测生成幻觉风险评估，指导问题重写；并在无微调的情况下实现预生成层面风险降低。

**🔧 技术方法**

使用对抗性视觉探测、弱监督风险估计器、问答重写提示以及风险引导的排名算法，并采用大型VLM如Qwen2-VL-2B-Instruct。

**📊 数据集**

主要数据集为VQA-RAD和300样本PMC-VQA进行外部验证。

**📈 对比分析**

与原始问题、单纯提示重写和Oracle上限比较，Ask4VG在风险上降低约4-5%并提升准确率约2个百分点；Oracle可进一步提升至0.352。

**⚠️ 局限性**

局限在于风险评估依赖无标签的弱监督，可能无法捕捉所有幻觉；重写候选数量有限，且未在更大多样化数据集上验证。

---

## 197. Self-Conditioned Positional HNSW for Overlap-Aware Retrieval in Chunked-Document RAG Systems: Method and Industrial Evidence-Quality Audit

**arXiv ID:** 2606.01542 | [PDF](https://arxiv.org/pdf/2606.01542v1)

**作者:** Nataraj Agaram Sundar `[一作]` (eBay Inc.), Tejas Morabia `[通讯]` (eBay Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种自条件位置编码的HNSW检索方法（SCP‑HNSW），用于解决文档分块重叠导致的检索冗余问题，并对其在检索增强生成（RAG）系统中的应用进行了工业级质量评估。

**💡 创新点**

创新点在于：①将低维位置编码附加到块嵌入中，并通过两次检索（先无位置，再自条件位置）估计查询的文档位置先验；②在最终上下文构造时使用可审计的最小索引间隔选择器，实现无修改HNSW核心算法的重叠意识检索；③结合文本与OCR审计数据，为RAG检索质量提供透明、可追溯的评估框架。

**🔧 技术方法**

使用的技术包括：Hierarchical Navigable Small World (HNSW) 近似最近邻检索；位置编码（半周期正余弦编码）；两次检索的自条件查询；最小索引间隔过滤器；以及对检索结果的定量评估（如覆盖率、冗余率）和人工审计工具。

**📊 数据集**

数据集主要包括：770 条生成证据的文本审计数据（其中 318 条完整标注）和 70 条 OCR 审计样本（共 350 条评分），两者均来自工业内部 RAG 生成任务。

**📈 对比分析**

对比方法方面，本文未给出完整的基线实验，但指出需要与语义 HNSW、带间隔过滤器的 HNSW、MMR 多样化以及 SCP‑HNSW 进行对照；目前的评估仅呈现审计统计（如证据评分分布、OCR 通过率），未能直接量化 SCP‑HNSW 在覆盖率或冗余率上的提升，需要进一步的受控检索实验。

**⚠️ 局限性**

局限性包括：①假设块顺序具有语义连贯性，对多区域、表格或杂乱文档可能失效；②单一位置先验可能过度聚焦，忽视远端关键信息；③审计数据仅为描述性统计，缺乏查询级对照，无法得出因果性能提升结论；④工业审计过程未公开完整样本，限制了外部复现。

---

## 198. Can LLM Agents Sustain Long-Horizon Organizational Dynamics?

**arXiv ID:** 2606.01199 | [PDF](https://arxiv.org/pdf/2606.01199v1)

**作者:** Xuancheng Zhu `[一作]` (Beijing University of Posts and Telecommunications), Guoshun Nan `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1431 | [OpenAlex ID](https://openalex.org/A5020360628)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 TaskWeave 框架，用于在多层级组织结构中维护长周期的计划与执行一致性，利用 LLM 代理并结合层级记忆与依赖驱动执行。

**💡 创新点**

将长周期组织模拟转化为“记忆中心协调”问题，首次提出 Formulate–Partition–Diagnose–Align (FPDA) 循环与可追溯的执行记忆机制，并在代理层面嵌入组织先验与委派拓扑，实现跨层级、跨时间的意图传播与依赖满足。

**🔧 技术方法**

核心技术包括：多层级 LLM 代理、FPDA 规划循环、依赖感知的任务拆分与执行、结构化提示工程、追踪式记忆池、以及用于评估的 KL、完成率、敏感信息检测等指标。

**📊 数据集**

使用自建的年度 SaaS 公司模拟数据（15 名代理、3 层级、3 个部门），以及在金融、制造、政府等三类机构的跨域扩展实验，全部为合成任务序列与环境事件。

**📈 对比分析**

通过与 AutoGen、MetaGPT、CAMEL 等通用多代理框架在组织一致性、计划传播、执行可追溯性与企业 NLP 产出等维度进行对比；TaskWeave 在角色分配一致性（KL 越低）和完成率（最终完成率 > 80%）上显著优于基线，且在敏感信息检测时 API 调用和成本最低。

**⚠️ 局限性**

受限于仿真环境过于理想化，缺乏真实机构规则、市场信号、复杂沟通渠道与人机交互；评估指标主要关注一致性与完成率，未覆盖经济有效性与行为真实性，需要在更逼真或真实数据上进一步验证。

---

## 199. Digital Twin-Assisted Adaptive Multi-Agent DRL for Intelligent Spectrum and Resource Management in Open-RAN UAV-Enabled 6G Networks

**arXiv ID:** 2606.01324 | [PDF](https://arxiv.org/pdf/2606.01324v1)

**作者:** Marwan Dhuheir `[一作]` (University of Luxembourg), Symeon Chatzinotas `[通讯]` (University of Luxembourg)

**通讯引用:** 26656 | [OpenAlex ID](https://openalex.org/A5016154330)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在Open‑RAN UAV‑enabled 6G网络中提出了基于数字孪生的自适应多智能体深度强化学习框架，结合粒子群优化进行无人机轨迹优化，联合资源分配（关联、功率、频谱）以提升数据速率、降低延迟。

**💡 创新点**

将数字孪生与非实时 RIC 协同训练结合多智能体 DRL，实现全球同步与边缘实时执行；采用 PSO 与 MADRL 迭代协同，解决非凸混合整数优化问题；使用双重评论家与目标策略平滑提升学习稳定性。

**🔧 技术方法**

数字孪生、Open‑RAN 架构、粒子群优化（PSO）、多智能体深度强化学习（MADDPG/MA Actor‑Critic 改进的 DDPG）、双重评论家、目标策略平滑、参数噪声自适应。

**📊 数据集**

仿真环境：1000 m×1000 m 区域、100 个地面用户、3 个 GRU/3 个 UAV 集群，用户分布采用幂律分布。

**📈 对比分析**

与 MADDPG、MAPPO、MA Actor‑Critic 及贪婪策略对比；实验显示该方法收敛更快、平均数据速率更高、总延迟最低（约 60 ms），能量利用率亦优于基线。

**⚠️ 局限性**

实验仅基于离线仿真，未考虑真实环境中的复杂干扰、硬件限制和链路时延；模型对无人机数量扩展性与多目标优化（如公平性、QoS）仍有待进一步研究。

---

## 200. OpenEye: A Scalable Open-Source Hardware Accelerator for DNNs

**arXiv ID:** 2606.01450 | [PDF](https://arxiv.org/pdf/2606.01450v1)

**作者:** Denis Lebold `[一作]` (University of Duisburg-Essen), Hendrik Wöhrle `[通讯]` (University of Duisburg-Essen)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5111804298)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了可扩展、稀疏感知的 FPGA 加速器 OpenEye，用于高效执行 CNN 的卷积、全连接和池化操作。

**💡 创新点**

其创新点在于可独立扩展的聚类和处理单元、支持稀疏权重与激活的原生稀疏编码、层级路由网络与动态数据流配置，以及在真实 FPGA 上对通信与内存开销的完整评估。

**🔧 技术方法**

采用了可参数化的集群/PE 层级结构、可配置的路由器、稀疏编码解码器、Live Value Table（LVT）实现多端口 PSUM 存储、AXI/Wishbone 接口与 FPGA 资源优化。

**📊 数据集**

实验使用 8 位量化的 MNIST CNN（包含 3 个卷积+池化和 2 个全连接层）进行验证。

**📈 对比分析**

通过在 Xilinx ZU19EG FPGA 上实现多种 cluster/PE 配置，对资源利用、延迟和吞吐量进行测量，结果表明算力近线性扩展，但数据传输开销在大规模配置下成为主要瓶颈。

**⚠️ 局限性**

局限在于对不同网络拓扑和稀疏模式的评估不足，且在高度稀疏或大卷积核场景下的性能与能耗仍待进一步优化。

---

## 201. Training-free image inversion for one-step diffusion models

**arXiv ID:** 2606.01380 | [PDF](https://arxiv.org/pdf/2606.01380v1)

**作者:** Tao Wu `[一作]` (Computer Vision Center), Joost van de Weijer `[通讯]` (Computer Vision Center)

**通讯引用:** 17964 | [OpenAlex ID](https://openalex.org/A5101958996)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个训练自由的反演框架，用于一阶扩散模型的文本引导图像编辑。

**💡 创新点**

创新点包括迭代噪声对齐（Iterative Noise Alignment）和后缀学习（Suffix Learning）两项技术，解决了初始噪声可编辑性和标题缺口问题，并通过交叉注意力掩码实现局部编辑。

**🔧 技术方法**

采用DDIM反演、一阶扩散模型 SD‑Turbo、对噪声和后缀词的梯度优化、KL散度正则化以及交叉注意力掩码等技术。

**📊 数据集**

使用 PIE‑Bench 数据集进行评估，并在部分实验中使用 LLaVA 生成的标题。

**📈 对比分析**

与 NTI、DDPM‑Inv、ReNoise、TurboEdit、FlowAlign、SwiftEdit 等方法比较，在单步编辑场景下，在七项指标（编辑一致性、结构保持、背景保留、PSNR、LPIPS、MSE、SSIM）上均优于现有方法，且推理速度更快。

**⚠️ 局限性**

主要局限是反演阶段需要约 600 步优化，耗时约 2 分钟，整体推理时间仍显冗长；且目前仅支持基于 U‑Net 的扩散模型，对 Transformer‑based 框架的适配尚未实现。

---

## 202. Application of Algorithms in Energy-Efficient Design Platforms for Green Building

**arXiv ID:** 2606.01229 | [PDF](https://arxiv.org/pdf/2606.01229v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 203. How Proposal Novelty, Topical Diversity, and Theory-Practice Balance Shape Scholarly Outcomes in Funded Education Research

**arXiv ID:** 2606.01127 | [PDF](https://arxiv.org/pdf/2606.01127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 204. Fail-Closed Lowering of Resident KV Claims onto LLM Serving Runtimes

**arXiv ID:** 2606.01387 | [PDF](https://arxiv.org/pdf/2606.01387v1)

**作者:** Lukas Stepanek `[一作]` `[通讯]`, Lukas Stepanek

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建一套基于义务的 ResidentClaim 降级模型，对 LLM 运行时的 KV‑cache 相关特性进行语义合规性检查；

**💡 创新点**

创新点在于提出 fail‑closed 降级关系、对应检查器、以及多层次的义务组合，显式区分特性表面与真正的未来 KV‑reuse 责任；

**🔧 技术方法**

使用了 YAML 描述符、Python 检查器、事件日志、调度器与连接器的原始 trace 以及自定义的压力实验；

**📊 数据集**

数据集主要包括 TensorRT‑LLM、SGLang/HiCache、Dynamo‑style 路由的运行时描述符与执行日志，以及在本地 patch 的 vLLM connector/scheduler‑boundary 的重复实验数据；

**📈 对比分析**

通过生成矩阵与 bad‑lowering suite 对比，检验是否会产生误判；实验显示大部分特性表面被判为近似或被拒绝，未出现原生 ResidentClaim 合规；而 patch 版 vLLM 在 131 次重复运行中成功通过所有语义门控，证明缺失的生命周期/结果义务可实现；

**⚠️ 局限性**

局限性包括：仅在公开运行时进行评估，未发现原生合规；patch 结果仅在本地单 GPU 环境下得到，未体现生产级性能；缺少跨模型、并发与大规模部署的验证；

---

## 205. OneVLA: A Unified Framework for Embodied Tasks

**arXiv ID:** 2606.01241 | [PDF](https://arxiv.org/pdf/2606.01241v1)

**作者:** Lingfeng Zhang `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**通讯引用:** 7979 | [OpenAlex ID](https://openalex.org/A5012419026)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 OneVLA，统一的 VLA 框架可同时完成导航和操控任务。

**💡 创新点**

创新统一动作头，将导航与操控动作拼接为 11 维空间，并通过多阶段进阶训练实现跨任务互助。

**🔧 技术方法**

采用统一视觉-语言编码器 + Flow‑matching diffusion 动作头 + Qwen2.5‑VL‑3B backbone + 多阶段训练。

**📊 数据集**

使用 VLN‑CE（R2R、RxR）导航数据、SimplerEnv‑Bridge 操控数据、通用 VQA 与 CoT 数据。

**📈 对比分析**

与 UniVLA、StreamVLN、π_0‑Fast 等基线对比，OneVLA 在所有任务上显著领先（导航提升约 30%，操控提升约 16%，3B 体积胜过 7B 任务专用模型）。

**⚠️ 局限性**

对动作预测序列长度敏感，长时序误差累积导致性能下降，且对不同硬件平台的可迁移性仍需进一步验证。

---

## 206. Don't Read Everything: A Curvature-Conditioned Query for Linear Attention

**arXiv ID:** 2606.01294 | [PDF](https://arxiv.org/pdf/2606.01294v1)

**作者:** Dong Le `[一作]` (Nanyang Technological University), Anh Tuan Luu `[通讯]` (Nanyang Technological University)

**通讯引用:** 2413 | [OpenAlex ID](https://openalex.org/A5050386762)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在线性注意力模型中引入了一种只修改读取步骤的读侧纠正机制（CCQ），通过基于键的协方差对查询进行自适应收缩，从而提升检索性能、长上下文推理和语言建模能力。

**💡 创新点**

创新点在于：①仅在读取阶段引入基于软max局部曲率的查询压缩；②利用键的二阶统计（协方差）实现自适应方向收缩；③与任意线性注意力写规则兼容，能够无缝叠加在现有写侧改进之上。

**🔧 技术方法**

技术实现包括：线性注意力的快速权重记忆（S_t 递推更新），键协方差维护（C̅_t 与 μ_t 递推），查询收缩算子 q_clean = (I - λ Σ)q，λ 为基于查询的门控；评估使用自回归语言建模、零样本推理、S-NIAH、长序列困境（LongBench）等任务。

**📊 数据集**

数据集主要为 FineWeb-Edu（自回归预训练）、Qwen3-4B-Instruct、GovReport、QMSum、NarrativeQA、Qasper、PG19、CodeParrot 等长序列基准，以及 14 任务的 English LongBench。

**📈 对比分析**

与传统 Transformer、Mamba2、GLA、GLA-Hedgehog、Gated DeltaNet 等基线对比，CCQ 在 500M 与 1.3B 模型规模下均表现出：语言模型困惑度下降、零样本下游准确率提升、S-NIAH 8K 任务显著提高、长上下文 PPL 在 4K–20K 区间下降、LongBench 多任务平均分提升 1–3 分。相比仅在写侧改进的模型，CCQ 的改进更为显著且普适。

**⚠️ 局限性**

局限性包括：只在 500M 与 1.3B 规模、两种线性注意力骨干（GLA 与 Gated DeltaNet）上验证；未测试更大规模（>7B）、不同状态空间结构（Mamba、RWKV 等）或多轮交互、智能体任务；并且仅评估了 perplexity、S‑NIAH、长度外推与 LongBench，未覆盖其他潜在应用。

---

## 207. Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs

**arXiv ID:** 2606.01215 | [PDF](https://arxiv.org/pdf/2606.01215v1)

**作者:** Wentao Mo `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Yang Liu `[通讯]` (Wangxuan Institute of Computer Technology, Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过将神经符号程序的推理模式转化为自然语言链式思考（CoT），并在三阶段课程中逐步训练，构建了一种可解释且支持开放词汇的3D多模态LLM（APEIRIA）。

**💡 创新点**

创新点在于：①利用神经符号程序的完整执行轨迹作为监督，直接把“语法”式的推理模式注入LLM；②设计了三阶段课程（感知对齐 → CoT‑SFT → CoT‑RL），让模型先学会对象识别，再学会系统性推理，最后通过奖励驱动的强化学习实现开放词汇与深层嵌套指令的泛化；③提出了稀疏的格点位置编码和高效的对象中心表征，保持了空间结构性同时降低令牌消耗。

**🔧 技术方法**

使用技术包括：对象中心3D表征、格点位置编码、CoT监督微调（CoT‑SFT）、基于奖励的链式思考强化学习（CoT‑RL）以及多任务微调和LoRA参数化。

**📊 数据集**

主要使用的数据集有 ScanRefer、Multi3DRefer、Sr3D、Nr3D、ScanQA、SQA3D、Scan2Cap 等，其中 Sr3D 提供完整的程序级监督，ScanRefer 与 Multi3DRefer 用于评估跨域与开放词汇性能。

**📈 对比分析**

在 ScanRefer Acc@0.25 及 Acc@0.5 上，APEIRIA 分别达 58.4% 与 51.2%，超过所有传统神经符号方法并与最强的3D MLLM 接近；在 Multi3DRefer F1@0.25 上达到 59.2%，同样优于对手；在开放词汇 Nr3D 的零样本评估中也优于监督的NS3D，说明推理模式的迁移效果显著。

**⚠️ 局限性**

主要限制包括：仍依赖外部感知模块，若感知精度不足会导致推理失效；强化学习阶段对奖励设计敏感，需手工调参；开放词汇与复杂嵌套指令的泛化虽有进展，但在极其复杂语义场景下仍可能出现推理漂移。

---

## 208. From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning

**arXiv ID:** 2606.01123 | [PDF](https://arxiv.org/pdf/2606.01123v1)

**作者:** Jun-Jie Yang `[一作]` (National Yang Ming Chiao Tung University), Ping-Chun Hsieh `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 268 | [OpenAlex ID](https://openalex.org/A5017738079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计了一种基于奖励无关表示学习的离线偏好学习框架 FB-PbRL，利用 Forward–Backward（FB）分解预训练得到的潜在表示，再通过对比学习与偏好数据对齐，直接推断任务向量并对网络进行细调；

**💡 创新点**

将奖励无关表示学习与对比式偏好学习结合，证明偏好损失等价于 SimCLR 对比损失，消除显式奖励模型的需要，显著提升偏好样本效率；

**🔧 技术方法**

使用 Forward–Backward 表示学习、SimCLR 风格对比损失、线性奖励参数化、贝尔曼残差测量损失、正交性正则化以及偏好搜索与细调技术；

**📊 数据集**

在 DeepMind Control Suite（Cheetah、Walker、Quadruped、Pointmass）、Adroit Pen 以及 MetaWorld Button‑Press‑Topdown 数据集上进行实验，偏好数据由脚本教师或真实人类提供；

**📈 对比分析**

与 DPPO、OPPO、OPRL、CLARIFY、LiRE、以及多种 RFRL 基线（FB、Laplace、HILP、PSM、RLDP）对比，FB‑PbRL 在离线 PbRL 协议下平均提升约 200 奖励点，甚至在多任务上超过零‑shot RFRL；在偏好有限或噪声的设置下保持鲁棒性，并在训练时间上收敛更快；

**⚠️ 局限性**

依赖于离线数据的覆盖度和偏好质量；在稀疏或偏置数据场景下性能下降；当前框架仅在离线设置验证，在线扩展仍需进一步研究；

---

## 209. UniD$^3$: A Knowledge Graph-Enhanced RAG Framework for Drug-Disease Discovery and Reasoning

**arXiv ID:** 2606.01394 | [PDF](https://arxiv.org/pdf/2606.01394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 210. Knowledge-Intensive Video Generation

**arXiv ID:** 2606.01285 | [PDF](https://arxiv.org/pdf/2606.01285v1)

**作者:** Chenxu Wang `[一作]` (Fudan University), Mingda Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出知识密集式视频生成任务，构建 1080 条信息寻求型提示的基准 KIVI-Bench，并针对生成视频提出事实准确性和有用性评估指标。

**💡 创新点**

创新点在于将视频生成从仅关注视觉质量转向评估视频内容的事实性与实用性，并首次提出自动化的事实精准度和有用性评估方法。

**🔧 技术方法**

使用 Gemini 3.1 Pro LLM 进行提示规划、视频主张提取、事实验证以及有用性评分，结合扩散式视频生成模型进行生成。

**📊 数据集**

主要利用从 WikiHow 视频中选取的 18 类主题生成 1080 条短提示，并对 54 条子集进行模型评测。

**📈 对比分析**

通过人类评估与自动指标对比，发现提出的事实精准度 70.8% 与有用性评分 69.0% 的指标与人类判断的契合度显著高于传统视觉质量指标；七大顶尖模型在所有评估维度均落后人类，差距最大达 37 分。

**⚠️ 局限性**

局限包括仅在子集上实验、依赖 LLM 提取/验证可能产生漏报或误报、评估仅基于文本主张忽略视觉细节，以及结果仅适用于当前模型与设定。

---

## 211. Learning Multi-Modal Trajectory Policies for Data-Efficient Robotic Manipulation

**arXiv ID:** 2606.01047 | [PDF](https://arxiv.org/pdf/2606.01047v1)

**作者:** Zijia Chen `[一作]` (National University of Defense Technology), Li Liu `[通讯]` (National University of Defense Technology)

**通讯引用:** 22084 | [OpenAlex ID](https://openalex.org/A5100418783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了名为MATE的多模态Mixture-of-Experts轨迹预测框架，用于在数据稀缺的机器人操作场景中提升轨迹生成与下游策略学习的效果。

**💡 创新点**

通过细粒度子特征拆分的多模态MoE、基于余弦相似度的尺度不变路由器以及温度/噪声稳定化策略，实现了跨模态专家分配的稳定与平衡。

**🔧 技术方法**

结合Transformer轨迹模型、Mixture-of-Experts结构、余弦路由、温度缩放与高斯噪声注入等技术。

**📊 数据集**

在LIBERO基准（Spatial、Object、Goal、Long）以及真实机器人乒乓球实验中进行评估。

**📈 对比分析**

与传统BC、R3M、VPT、UniPi等基线以及重现的轨迹引导基线和多种MoE变体对比，平均成功率提升4.75%，在LIBERO-Long提升8.83%，真实任务中球返击率从76%提升至85%。

**⚠️ 局限性**

仅在LIBERO有限演示数据上验证，未探究大规模数据；轨迹预测受视觉遮挡、快速运动或复杂接触影响，可能导致失败。

---

## 212. Near-Optimal Pure Machine Unlearning for Smooth Strongly Convex Losses

**arXiv ID:** 2606.01527 | [PDF](https://arxiv.org/pdf/2606.01527v1)

**作者:** Matthew Regehr `[一作]` (University of Waterloo), Andrew Lowy `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了机器无学习（unlearning）在光滑强凸随机优化中的最优统计误差率，给出了上界和下界，并证明在 ε ≫ d 时，可通过新算法实现指数级的性能提升；在 ε ≲ d 时，重新从零训练仍是最优方案。

**💡 创新点**

创新点在于：1) 提出了近似 ε‑unlearning 的 ERM 混合采样算法，取得几乎最优的风险上界；2) 通过打包与对抗性删除构造了匹配的下界，尤其在均值估计问题上实现了指数项完全匹配；3) 在风险分析中首次把 unlearning 罚项与统计误差明确分离，并揭示其对 ε 与维度 d 的指数依赖。

**🔧 技术方法**

主要技术手段包括：强凸稳定性分析、凸几何包围/打包技术、对抗性删除构造、改进 Newton‑step 算法、以及对 ε‑indistinguishability 的概率论估计。

**📊 数据集**

论文聚焦理论分析，并未在具体公开数据集上进行实验；所有结论均基于抽样分布和理论推导。

**📈 对比分析**

与 retraining‑from‑scratch、DP 基线以及之前的 Newton‑step 算法等进行比较；表格显示在 ε ≳ d 时，新算法的风险率为 1/n + (m/n)²·e^{‑2ε/(d+2)}，相比 retrain‑from‑scratch 的 (m/n)² 以及 DP 的 d²m²/(ε²n²) 提升了指数级；当 ε ≲ d 时， retrain‑from‑scratch 仍保持最优。

**⚠️ 局限性**

局限性：上界与下界仅相差一个条件数因子；只覆盖光滑强凸情形，未扩展到非凸、非光滑或有显式存储/计算约束的情形；(ε,δ)-unlearning 的最优率仍未解决。

---

## 213. AI From the Margins (AIM): Rethinking Participatory AI Design Through the Lived Experience of Minoritized Communities

**arXiv ID:** 2606.01171 | [PDF](https://arxiv.org/pdf/2606.01171v1)

**作者:** Tijs Portegies `[一作]` (University of Amsterdam), Sennay Ghebreab `[通讯]` (University of Amsterdam)

**通讯引用:** 970 | [OpenAlex ID](https://openalex.org/A5009260617)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在荷兰医疗领域，通过八次“生活经验”工作坊，提出并实证验证了一种名为 AI From the Margins (AIM) 的方法论框架，旨在将少数族裔群体的生活经验置于 AI 设计的前置条件之中，先行构建七项预置条件（互惠、去殖民化立场、聚焦少数族裔视角、方法灵活性、拒绝权、维度可达性、共所有权），随后引导参与者在安全、平等的环境里分享经验、共创规则、讨论 AI 介入的时机与方式，并与政策制定者对话，探讨将这些经验转化为 AI 政策的可能路径。

**💡 创新点**

创新点在于：①提出在 AI 设计前置的“预置阶段”而非传统的后置参与，突破现有参与式 AI 仅在技术框架已设定后才引入受影响群体的局限；②用七项互相依赖的预置条件取代固定流程，强调方法论的灵活性与情境化；③将少数族裔的立场视为 AI 损害可见性的核心，而非单一输入，强调立场理论与交叉性在方法设计中的应用；④通过实证工作坊展示预置条件如何在不同技术手段（BNIM、Rich Picture 等）中被实现，证明“方法论姿态”可以迁移至多领域。

**🔧 技术方法**

采用的技术手段主要是定性方法：Biographic Narrative Interpretive Method (BNIM) 及其 SQUIN 触发提问；Rich Picture 绘图；协作规则制定；以及与政策工作者的对话。整个流程不涉及机器学习模型或算法实现，而是聚焦在研究设计与参与者互动的技术。

**📊 数据集**

未使用传统机器学习数据集；数据来源为 13 名女性及非二元族裔受访者的访谈、绘图、规则草案及与 5 名政策工作者的对话记录，全部为文本与图像形式的定性资料。

**📈 对比分析**

本文未进行量化对比或性能评估；评估方式主要是基于参与者对工作坊流程、氛围和产出（规则、反思）的主观评价以及研究者对预置条件落实情况的案例性描述，缺乏与现有方法的客观对照指标。

**⚠️ 局限性**

局限性包括：①方法论在单一医疗情境中验证，跨领域适用性尚未检验；②依赖研究者与主持者的立场与经验，可能产生新的权力偏差；③未能生成可直接评估的定量指标，评估依赖定性描述；④适用于大规模公共 AI 项目，可能不适用于小型或专有 AI 开发；⑤方法本身仍在迭代阶段，后续需要持续检验与完善。

---

## 214. KG-FairDiff: Knowledge Graph-Guided Prompt Refinement for Demographically Fair Text-to-Image Generation

**arXiv ID:** 2606.01282 | [PDF](https://arxiv.org/pdf/2606.01282v1)

**作者:** Farbod Davoodi `[一作]` (Qatar Computing Research Institute, Hamad Bin Khalifa University), Ali Diba `[通讯]` (Qatar Computing Research Institute, Hamad Bin Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 KG‑FairDiff，一个在推理阶段通过知识图谱引导提示词改写来实现文本到图像生成公平性的框架，适用于任何黑盒生成模型；

**💡 创新点**

创新点在于将结构化文化与偏见知识图谱与大语言模型重写器结合，形成闭环验证器，既能降低性别、种族、年龄和交叉维度偏见，又能保持原始语义；

**🔧 技术方法**

技术包括知识图谱检索、LLM（如 GPT‑4o）提示词改写、基于 KL 散度的公平损失与语义相似度约束的闭环迭代优化；

**📊 数据集**

使用了约 1,200 条手工整理的三元组（来自 McGillNLP Bias Dataset、CultureBank 等），以及 100 个职业相关提示词（50 个职业）作为实验基准；

**📈 对比分析**

在八个主流生成模型上与 MinorityPrompt、PreciseDebias、FairImagen 等基线对比，KG‑FairDiff 在性别、种族、年龄及交叉轴上的 Bias‑W 缩减可达 20 倍、ENS 增益 13 倍，且语义保真度变化极小；

**⚠️ 局限性**

主要限制包括对自动属性分类器的依赖、知识图谱覆盖不足、验证器与下游公平性指标的校准不充分，以及使用 GPT‑4o 作为重写器与验证器可能导致的循环偏差。

---

## 215. Temporal Motif Signatures for Temporal Graph Neural Networks

**arXiv ID:** 2606.01176 | [PDF](https://arxiv.org/pdf/2606.01176v1)

**作者:** Dylan Sandfelder `[一作]` (University of Oxford), Xiaowen Dong `[通讯]` (University of Oxford)

**通讯引用:** 2816 | [OpenAlex ID](https://openalex.org/A5101579932)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个13维候选局部短时图案计数特征，将其与任意时间图神经网络（TGNN）线性拼接，以提升边预测、链路属性预测和图级分类等任务的性能。

**💡 创新点**

创新点包括：①通过跨数据集诊断发现三条尺度稳定轴（dyadic recency/reciprocity、star diversity、triadic flow）；②在候选锚定的时间 Weisfeiler–Leman 层级中证明该特征位于第一层，填补标准 TGNN 的表达缺口；③构造了无架构改动、易集成的特征增强方法，并在多任务、多数据集上实现可复制的性能提升。

**🔧 技术方法**

使用的技术包括：时序窗口计数、对数归一化、邻居上限抽样、线性嵌入矩阵、时间 WL 理论分析、以及常见的 PR‑AUC、MRR、准确率等评估指标。

**📊 数据集**

实验数据集共13个，包括真实数据（MOOC、Bitcoin Alpha、Bitcoin OTC、PaySim、三种 TGB 链路属性基准）以及六类合成生成器（ER、BA、WS、配置模型、SBM、重连控制）。

**📈 对比分析**

通过与五种强 TGNN 基线（GraphSAGE、GCN、GAT、TGN、DyGFormer 等）在边分类、TGB 链路预测、图级分类三大任务上进行对照实验。结果显示，在 MOOC 任务 PR‑AUC 约 +0.3、在 DyGFormer + Review 链路预测 MRR 从 0.224 提升至 0.413、在图级分类任务中 SAGE 从 59% 提升至 92% 等，提升幅度随数据集特征而异；对已捕获相同结构的数据提升有限。

**⚠️ 局限性**

限制包括：①窗口半径 Δ 对性能敏感；②在稀疏流中三角流特征消失，导致提升受限；③仅考虑≤3节点的图案，未覆盖更高阶或多尺度结构；④对极端高频率流的表现尚未评估；⑤缺乏针对多尺度窗口或更高阶图案的扩展。

---

## 216. HomeFlow: A Data Flywheel for Smart Home Agent Training with Verifiable Simulation

**arXiv ID:** 2606.01230 | [PDF](https://arxiv.org/pdf/2606.01230v1)

**作者:** Yi Gu `[一作]` (Midea Group), Yi Xu `[通讯]` (Midea Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 HomeFlow 训练管道，使大语言模型在智能家居场景下能够通过可验证的仿真环境进行数据合成与策略优化

**💡 创新点**

创新点在于：①统一的可验证仿真环境 HomeEnv + HomeMaker；②将用户意图编译为可验证的状态条件 Blueprint；③基于 MCTS 的轨迹搜索 MCTS‑Flow；④步骤级 RLVE 以环境反馈实现连续奖励；⑤通过上述组合显著提升小模型表现并超过大型 LLM

**🔧 技术方法**

使用的技术包括：可执行代码生成、基于 MCTS 的树搜索、步骤级强化学习（RLVE）、可验证环境的状态检查、程序合成与自动化审计、结构化状态表示与设备抽象

**📊 数据集**

主要使用的实验数据集是自构建的 SmartHome‑Bench（1,678 家庭实例）和 HomeMaker 生成的多样化环境；Blueprint 通过 GPT‑5 生成场景脚本；对比基准使用多种公开 LLM（GPT‑5.5、Claude‑4.6、Qwen‑3 等）

**📈 对比分析**

与基准模型相比，HomeFlow‑RL‑8B 在 SmartHome‑Bench 上整体成功率 87.03%，超过 GPT‑5.5 1.23%；在 4B 级别上也优于所有对比模型；同时在任务细分上均表现最优，平均工具调用次数与标记长度也更为合理

**⚠️ 局限性**

局限性包括：依赖确定性仿真，难以直接迁移到真实设备；对硬件协议和通信协议的适配不足；MCTS 与在线 RL 计算成本高；未覆盖极端噪声与网络延迟等真实环境不确定性

---

## 217. PMC-InterCPT: Rethinking Biomedical Interleaved Data for Multimodal Continued Pretraining

**arXiv ID:** 2606.01049 | [PDF](https://arxiv.org/pdf/2606.01049v1)

**作者:** Guanghao Zhu `[一作]` (Hong Kong Polytechnic University), Hongxia Yang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 44063 | [OpenAlex ID](https://openalex.org/A5100378741)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套从BIOMEDICA数据集清洗、重构、质量筛选并进行模态平衡采样的完整流程，生成高质量的医学多模态CPT语料PMC‑InterCPT；

**💡 创新点**

创新点在于：①恢复缺失标题并去除XML残留、重复与不连贯上下文；②构建图文交错样本并通过LLM监督的医学相关性与文本质量分类器进行筛选；③提出四类证据桶的模态平衡采样策略；

**🔧 技术方法**

技术包括：基于XML的标题恢复、文本去噪与句子级重复清理、图文交错样本构造、LLM（gemini‑3.1‑pro‑preview）标注+Qwen3‑1.7B分类器、SGLang推理、BPE分词和采样；

**📊 数据集**

使用的原始数据集为BIOMEDICA（约24M对，13.65B token），处理后得到PMC‑InterCPT（约10.1M对，9.63B token）；

**📈 对比分析**

通过在Qwen3.5‑VL‑4B‑Base上做CPT+SFT实验，对比原始BIOMEDICA与PMC‑InterCPT，结果显示后者在医学VQA和一般科学多模态基准上平均提升约1–2分，且在更难的医学推理任务中提升更显著；

**⚠️ 局限性**

局限性包括：仅在英文PMC文献上构建，未覆盖多语言；评价主要集中在VQA式基准，缺乏对开放式生成和指令遵循的评估；模态平衡比例的最优性仍依赖具体模型和基准，需进一步验证。

---

## 218. Diamonds in the Sky: Pareidolic Animals in Clouds

**arXiv ID:** 2606.01361 | [PDF](https://arxiv.org/pdf/2606.01361v1)

**作者:** Miriam Horovicz `[一作]` (Reichman University), Yael Moses `[通讯]` (Reichman University)

**通讯引用:** 1346 | [OpenAlex ID](https://openalex.org/A5102320227)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种零样本方法，用扩散模型（MDDS）在云图像中预测并增强人类对云中“幻影动物”（pareidolic animals）的感知。

**💡 创新点**

创新点在于：1) 引入 Masked Delta Denoising Score（MDDS）实现对云区域的局部、受控扩散；2) 将云段转化为可被现有目标检测模型识别的动物形状；3) 通过短动画（从原云到改造图再回归）显著提升人类感知的准确度。

**🔧 技术方法**

核心技术包括：Segment Anything Model（SAM）用于云分割；稳定扩散模型配合 DDS/MDDS 进行文本引导的局部生成；OneFormer 进行动物识别与实例分割；SSIM 用于相似度评估；线性插值生成增强视频。

**📊 数据集**

使用自己收集的 50 张天空图像（共 250 个云段）作为实验数据集，云段经扩散得到 2,250 张候选图像，最终筛选出 60 张有效图像（对应 23 个云段）。

**📈 对比分析**

方法与传统识别模型（CLIP、OneFormer）在原始云图像上对比，后者全部未能检测到动物；在本方法上，模型在增强前的召回率为 0.28，增强后提升至 0.55；精确率从 0.11 提升到 0.30，F1 分数从 0.16 提升到 0.39，远优于随机基线（0.06/0.07/0.06）且仅略逊于理论最优模型（F1 0.45）。

**⚠️ 局限性**

主要局限包括：1) 仅支持 OneFormer 支持的 10 种动物，无法覆盖更广泛的动物类别；2) 每张云段改造需约 60 秒，难以实现实时应用；3) 数据集规模有限，云形态与动物种类多样性不足；4) 仍需进一步验证在其他模糊图像领域（艺术、天文等）的适用性。

---

## 219. MiCU: End-to-End Smart Home Command Understanding with Large Language Model

**arXiv ID:** 2606.01099 | [PDF](https://arxiv.org/pdf/2606.01099v1)

**作者:** Haowei Han `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 31355 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过自动化数据合成、专用LLM训练与token压缩三步，构建了面向智能家居的端到端指令理解系统；

**💡 创新点**

创新点在于：①基于用户日志的难度分级数据合成框架，②使用课程学习 + 领域引导CoT + RL 的两阶段训练策略提升推理与专属知识；③通过token压缩把设备描述映射为特殊标记，显著降低推理成本；

**🔧 技术方法**

采用Qwen3-4B作为基础模型，课程学习、CoT监督细调、DAPO强化学习、Self‑Refine、RAG与Self‑Refine生成标注，以及特定的token压缩技术；

**📊 数据集**

使用从小米Home平台采集的约50K条合成指令（30K易样本+20K难样本）以及9K人工标注测试集；

**📈 对比分析**

与规则基线、GPT‑4o‑mini、DeepSeek‑V3、Llama等对比，模型在DevIdent和ActPred上分别提升约28.3%和20%准确率，在线部署后用户纠正率下降1.57%并将人工审核准确率提升32%；

**⚠️ 局限性**

局限在于：对极大设备数仍需进一步压缩；模型对复杂优先级关系仍易失误；以及在更大规模或多语言场景下的通用性尚未验证。

---

## 220. Emergent Ordinal Geometry in Transformers Trained on Local Comparisons

**arXiv ID:** 2606.01269 | [PDF](https://arxiv.org/pdf/2606.01269v1)

**作者:** Nishit Singh `[一作]` `[通讯]` (Birla Institute of Technology and Science), Nishit Singh (Birla Institute of Technology and Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

训练小型Transformer仅用相邻比较，检验其对未见远程对的传递推理能力。

**💡 创新点**

发现Transformer自发形成一维“心智数轴”，并且决策置信度与几何分离随秩距离递增，重现生物学中的符号距离效应。

**🔧 技术方法**

使用单层Transformer结构、AdamW优化、权重衰减，训练交叉熵目标，并通过主成分分析评估嵌入几何。

**📊 数据集**

使用人工生成的隐藏全序实体集合，训练集仅包含相邻比较，测试集为所有远程对。

**📈 对比分析**

通过OOV准确率、PC1方差占比、秩对齐相关系数以及置信度/几何距离与秩距的相关性评估，结果显示近乎完美的OOV准确率，并且符号距离效应显著。

**⚠️ 局限性**

实验为相关性证据，未验证因果性；仅使用单层小模型与线性关系的合成数据，结果的稳健性、对不同架构及非线性/循环关系的适用性仍待进一步验证。

---

## 221. Institutional Trust and the Domestic AI Advantage: Evidence from DeepSeek and ChatGPT Users in China

**arXiv ID:** 2606.01228 | [PDF](https://arxiv.org/pdf/2606.01228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 222. Agentic Clustering: Controllable Text Taxonomies via Multi-Agent Refinement

**arXiv ID:** 2606.01255 | [PDF](https://arxiv.org/pdf/2606.01255v1)

**作者:** Simon Löwe `[一作]` (Burning Glass Institute), Emily Silcock `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于多代理（proposer、synthesizer、auditor、investigator、critic）与编排器（orchestrator）动态调度的文本聚类框架；

**💡 创新点**

核心创新在于将聚类流程从固定的程序化管道转为“agentic”模式，允许编排器根据当前状态选择适当的专用代理并实时调整聚类粒度、合并/拆分决策，并支持用户自定义聚类意图与目标簇数范围；

**🔧 技术方法**

技术实现主要依赖大语言模型（Claude Opus 4.7 处理小样本任务，GPT‑5‑mini 处理全量分类），结合随机不重复抽样、Prompting（多样化风格指令）、合成、审计、检索与批判性评估等步骤；

**📊 数据集**

在七个公开基准上评估：Banking77、CLINC150、MASSIVE‑Intent、MASSIVE‑Domain、GoEmotions、20 Newsgroups、StackExchange；

**📈 对比分析**

与七个传统与 LLM‑驱动的基线（LDA、SBERT‑kmeans、BERTopic、LLM‑embedding‑kmeans、ClusterLLM、TopicGPT、Huang & He）对比，单种种子实验表明其 ARI、NMI、ACC 最高，最大提升达 32%（如在 20 Newsgroups 上），且在成本上亦处于竞争力位置；

**⚠️ 局限性**

局限性包括：所有基线均为重新实现，可能存在实现差异；实验仅使用单一随机种子；仅评估英文数据集；依赖封闭式 LLM，易受模型更新影响；存在偏见与双重使用风险；

---

## 223. AlbedoEdit: Unified Instance-Level Video Editing with Albedo Guidance

**arXiv ID:** 2606.01362 | [PDF](https://arxiv.org/pdf/2606.01362v1)

**作者:** Xilong Zhou `[一作]` (Max Planck Institute for Informatics), Christian Theobalt `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 35887 | [OpenAlex ID](https://openalex.org/A5020664641)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的视频编辑框架 AlbedoEdit，可对视频进行物体插入、移除和纹理编辑，利用首帧的反照图进行细粒度控制并通过扩散模型生成光照和阴影一致的最终视频。

**💡 创新点**

创新点在于：①将不受光照影响的反照图作为统一的控制信号，简化编辑流程；②在单一模型中同时支持三种编辑任务；③通过合成高质量的RGB+反照配对数据并结合视觉语言模型生成描述文本，使模型学习到光照、反射和阴影等物理交互。

**🔧 技术方法**

使用的技术包括：基于 DiT 的扩散 Transformer、VAE 编码/解码、流匹配训练范式、基于 DiffusionRenderer 的反照提取、以及 Vision‑Language Model 生成的文本提示。

**📊 数据集**

使用新构建的大规模合成数据集，涵盖 9k 3D 资产，包含三类编辑对（无编辑、插入、纹理修改），以及基于 Objaverse、Objaverse++ 的高质量对象，训练时还配合生成的文本提示。

**📈 对比分析**

与多种统一编辑方法（VACE、UniVideo、VideoPainter）及任务专用方法（PISCO、ROSE、EffE、V‑RGBX）进行定量和定性比较。AlbedoEdit 在 VOR、VOI、VTE 三个任务上均达到了或超过专用方法的 PSNR/SSIM/LPSIPS 分数，且在无 GT 评测中的 VBench 分数位列前列，显示出更好的光照一致性和细节保留。

**⚠️ 局限性**

主要限制包括：仅在首帧使用反照图，导致难以精确控制物体运动；对反照提取质量敏感，若阴影或反射误判为纹理会导致光照残留；当前模型仅支持短视频（约 50 步扩散），未采用自回归或蒸馏方案扩展到长视频。

---

## 224. Frontlines and faultlines: How the Russo-Ukrainian conflict reshapes the landscape of scientific research

**arXiv ID:** 2606.01124 | [PDF](https://arxiv.org/pdf/2606.01124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 225. Chameleon: Style-Content Disentangled Framework for Cross-Domain Object Compositing

**arXiv ID:** 2606.01079 | [PDF](https://arxiv.org/pdf/2606.01079v1)

**作者:** Sukhun Ko `[一作]` (Chung-Ang University), Jihyong Oh `[通讯]` (Chung-Ang University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5090121183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Chameleon框架，实现前景对象跨域插入时既保留身份又匹配背景风格的图像合成

**💡 创新点**

创新点在于：①首个大规模训练集ChameleonDataset；②两阶段学习，Joint Hard Contrastive Learning (JHCL) 对DINOv3进行风格/内容解耦；③Spatio‑Temporal Attention Gating (STAG) 在Diffusion Transformer中自适应注入风格

**🔧 技术方法**

利用Diffusion Transformer（DiT）、DINOv3 ViT‑L/16、LoRA微调、联合硬对比学习及STAG机制

**📊 数据集**

构建ChameleonDataset_tr（200K 真实合成图）和ChameleonDataset_ev（评测集），并使用现有的合成与风格数据进行对比

**📈 对比分析**

在TF‑ICON、AIComposer等基准上，Chameleon在LPIPS、CLIP‑I、CSD、VLM评分均优于或逼近最佳方法；用户研究中在身份、风格、整体表现均最高

**⚠️ 局限性**

仍受限于：需要大量人工标注/生成的逆向数据管线；对极端风格差距或复杂光照的泛化尚待进一步提升

---

## 226. ChronosAD: Leveraging Time Series Foundation Models for Accurate Anomaly Detection

**arXiv ID:** 2606.01300 | [PDF](https://arxiv.org/pdf/2606.01300v1)

**作者:** Uzair Khan `[一作]` (University of Verona), Marco Cristani `[通讯]` (University of Verona)

**通讯引用:** 9239 | [OpenAlex ID](https://openalex.org/A5033671063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用Chronos时间序列基础模型作为零样本特征提取器，再通过自定义的BiLSTM＋多头注意力的时间块对多通道序列进行上下文编码，实现端到端的异常检测。

**💡 创新点**

在传统的单变量或手工特征方法上，首次引入跨域预训练的基础模型进行无须微调的特征抽取，并结合双向LSTM与注意力机制充分利用时序和跨通道信息；此外整体框架仅需极少的任务专属调参，具有较强的泛化能力。

**🔧 技术方法**

Chronos预训练模型、BiLSTM、Multi‑Head Attention、L2归一化、SeLU激活、线性层与全连接分类器。

**📊 数据集**

UCR Archive 7个单变量数据集、工业轴承数据集 CWRU、医疗 ECG 数据集 MIT‑BIH、水处理数据集 SWaT，以及仿真波形数据集 Waveform，总计 11 个多域基准。

**📈 对比分析**

与 11 种主流无监督/半监督异常检测方法（AnoGAN, ALAD, Deep‑SVDD, BeatGAN, GOAD, USAD, TLKF, GTA, NSIBF, ECOD, KalmanAE）进行对比；在 AUC 与 AP 上平均提升 4.72% 与 6.60%，在多数据集上常能获得 97% 以上的 AUC，显示出明显优势。

**⚠️ 局限性**

目前仅在有标签的监督设置下验证；对无监督环境的适用性尚未探究；依赖预训练模型和显著的计算资源；以及多通道交互仅通过简单的多路径融合，可能不足以捕捉更复杂的跨通道依赖。

---

## 227. PAI-Studio: Cinematic Video Background Replacement with Camera-Aware Motion

**arXiv ID:** 2606.01399 | [PDF](https://arxiv.org/pdf/2606.01399v1)

**作者:** Heyuan Gao `[一作]` (Utopai Studios), Mike Zheng Shou `[通讯]` (National University Of Singapore)

**通讯引用:** 4001 | [OpenAlex ID](https://openalex.org/A5068937750)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PAI-Studio框架，实现前景视频与多帧背景参考图像在保持前景身份、场景光照一致和边界鲁棒性的同时，生成运动一致、动态的背景视频。

**💡 创新点**

核心创新点包括：1）将前景视频、背景参考和去噪帧拼接为单一序列，并用全双向注意力的Diffusion Transformer实现跨模态全局推理；2）利用时间位置编码克隆实现多帧背景控制；3）在训练中加入前景光照扰动与结构化注释，提升场景自适应重光与运动一致性；4）构建30K规模的CineStudio数据集，专门针对真实的前景分割不完善、光照差异和摄像机运动等挑战。

**🔧 技术方法**

技术栈包括Diffusion Transformer（DiT）视频生成骨干、VAE编码/解码、全双向自注意力、多模态条件拼接、时间位置编码克隆、光照扰动数据增强、结构化JSON注释（基于Gemini-3）以及LoRA微调。

**📊 数据集**

使用自研的CineStudio数据集，来源于高质量电影和网络视频，经过镜头分割、光流过滤、前景分割、背景填补和光照扰动等处理，共30K个样本。

**📈 对比分析**

在Cine-Restore和Cine-NBG两大子集上与VACE（开源）及Kling O1、Kling 3.0 Omni、Runway Aleph、Switchx（商用API）进行对比。PAI-Studio在背景一致性、运动一致性、前景保真、光照和边界融合等指标均超过所有基线，MSE降至0.0164、SSIM提升至0.717、LPIPS降至0.303，商业API的最佳指标也无法匹敌。

**⚠️ 局限性**

局限性主要体现在：1）仍需依赖前景分割或掩码，极度模糊或剧烈变形的边界可能导致合成误差；2）对极端光照或摄像机轨迹的迁移性未作完整验证；3）推理时仍受限于模型尺寸与显存需求，实时性尚不理想；4）在极少数情况下，结构化注释生成的文本信息可能不够精准，影响重光与运动匹配。

---

## 228. What LLMs Must Forget to Teach Effectively: A DIY Approach to Premodern Japanese Language Pedagogy

**arXiv ID:** 2606.01410 | [PDF](https://arxiv.org/pdf/2606.01410v1)

**作者:** Ariel Stilerman `[一作]` (Stanford University), Gavin Sherry `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了三种基于大语言模型的教学工具（BungoBot 辅导、Hiki 词典、Sata 对话伙伴），用于预现代日语课程的教学与学习。

**💡 创新点**

创新在于通过“DIY”提示工程与 XML 标签自定义系统提示，赋予模型教学代理角色并强调学习者主动性与可定制化，而非依赖昂贵的微调或检索增强。

**🔧 技术方法**

采用大语言模型（Gemini、ChatGPT 等）并利用系统提示（prompt engineering）构建自定义实例，辅以 XML 标签与少量示例实现零/少样本推理。

**📊 数据集**

主要依赖 LLM 内部知识与公开教材，未使用专门的训练数据集；仅在日志中使用教材文本进行交互验证。

**📈 对比分析**

通过学生反馈、课堂日志和与标准 LLM 对比（如长序列时可靠性下降）评估，结果显示学习者主动性提高、课堂互动增强，但缺乏量化性能指标。

**⚠️ 局限性**

局限包括模型幻觉、长交互可靠性下降、对预现代日语知识的准确性不保证、对硬件与能源成本的依赖，以及提示工程需手动迭代且可移植性有限。

---

## 229. Truthful AI Advisors: A Pre-Specified Benchmark for Large Language Model Honesty Under Preference Misalignment

**arXiv ID:** 2606.01456 | [PDF](https://arxiv.org/pdf/2606.01456v1)

**作者:** Hamidreza Hasani Balyani `[一作]` (Amazon), Arshia Gharagozlou `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了一套基于Crawford–Sobel廉价谈话模型的实验基准，用来评估指令调优的LLM在偏离用户利益时的诚实程度。

**💡 创新点**

创新点在于将经典经济学的最优信息传递分区理论作为可复制的“真值”基准，并将其应用于自然语言LLM发送者的单轮数值通信，揭示LLM在偏离目标时倾向于近乎完全披露状态并加上线性偏差（线性夸大）。

**🔧 技术方法**

技术上使用了低温确定性解码、混合解码器（数值解析+句子嵌入回归）、基于动态规划的单调分段估计以及互信息归一化评估，并结合Bootstrap检验。

**📊 数据集**

数据集为200个均匀采样的[0,1]状态，跨4款LLM、5个偏差水平和3个提示框架共计12,000条发送者消息。

**📈 对比分析**

比较方法是将模型输出的分区计数、归一化互信息和损失与理论最优廉价谈话分区、完全披露和无信息三种基线对齐；实验显示所有模型均显著超出最优分区（1.8–4.2倍互信息），但信息量随偏差下降，提示框架对结果影响不大。

**⚠️ 局限性**

局限性包括单一一维均匀状态、仅正向偏差、单轮数值消息、对解码器高度依赖以及对特定LLM的外推受限，未覆盖多轮对话、复杂任务或非数值表达的情形。

---

## 230. LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World

**arXiv ID:** 2606.01458 | [PDF](https://arxiv.org/pdf/2606.01458v1)

**作者:** Hojune Kim `[一作]` (Stanford University), Mac Schwager `[通讯]` (Stanford University)

**通讯引用:** 7870 | [OpenAlex ID](https://openalex.org/A5081950488)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种名为Loco-manipulation via Embodied Gaussian Splatting的混合式仿真器，能够在无需人类遥控演示的情况下生成全身操控与物体抓取的高质量训练数据。

**💡 创新点**

创新点在于将物理仿真MuJoCo、3D Gaussian Splatting的视觉真实性以及SAM3D对象重建结合起来，使用两阶段颜色校准实现渲染与真实相机的高一致性，并通过程序化动作原语生成实现可重渲染、可扩展的数据集，彻底消除了对昂贵人类演示的依赖。

**🔧 技术方法**

技术手段包括MuJoCo物理后端、SONIC低层全身控制器、3D Gaussian Splatting渲染、SAM3D三维重建、两阶段颜色校准、程序化动作生成以及GPU重渲染，形成端到端可在图形卡上并行化生成数据的流程。

**📊 数据集**

使用的数据集来自于手持摄像机对目标场景拍摄的1–2分钟视频与单张物体照片，经过COLMAP重建得到3D Gaussian Splatting背景和SAM3D对象网格；在此基础上在Unitree G1机器人上生成三种pick-and-place任务的演示，并与人类遥控、Mesh-only 以及其增强版本进行对比。

**📈 对比分析**

实验在三种VLA backbone（ψ_0、π_0.5、GR00T N1.6）上进行，使用50条或200条演示进行微调，以真实机器人任务成功率为评价指标；结果显示，(200)数据集在所有任务与backbone下都至少与人类遥控相当，且在最难任务中遥控零成功的情况下仍能实现显著成功率，Mesh-only 200演示则明显落后。

**⚠️ 局限性**

局限性包括：颜色校准禁用自动曝光和白平衡，导致对光照变化敏感；需手持摄像收集场景，不能完全免费；重建在高反射或透明表面上会失真；程序化生成的动作未针对能效或动态优化；仅在单一机器人平台、摄像头和控制器上验证；对不同物体类别或动态背景的迁移能力未作充分测试。

---

## 231. Tokenized but Illiquid? Evidence from Real-World Asset Markets

**arXiv ID:** 2606.01131 | [PDF](https://arxiv.org/pdf/2606.01131v1)

**作者:** Rischan Mafrur `[一作]` `[通讯]` (Western Sydney University), Rischan Mafrur (Western Sydney University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究以太坊上Token化的真实世界资产是否产生可观的二级市场流动性，并构建月度面板数据进行描述性、非参数检验与面板回归分析。

**💡 创新点**

提出多维度实证流动性测度框架，比较国债、黄金与私人信用等资产类别的流动性差异，并揭示规模并非决定流动性的关键因素。

**🔧 技术方法**

采用面板回归、Kruskal‑Wallis检验、Spearman相关以及非参数分组检验等统计方法。

**📊 数据集**

以RWA.xyz为主要数据源，辅以Etherscan合约层数据，构建9个以太坊非稳定币RWA的12个月面板（12月2025–5月2026）。

**📈 对比分析**

通过描述性统计和分组检验展示资产类别差异；面板回归显示持有者广度显著提升流动性，规模不显著；结果表明黄金资产流动性最好。

**⚠️ 局限性**

样本规模小、周期短、仅覆盖以太坊，缺乏转让规则、赎回权等关键变量，难以深入评估交易深度与定价等微观结构。

---

## 232. Efficient Exploration for Iterative Nash Preference Optimization

**arXiv ID:** 2606.01382 | [PDF](https://arxiv.org/pdf/2606.01382v1)

**作者:** Tianlong Nan `[一作]` (Columbia University), Tianyi Lin `[通讯]` (Columbia University)

**通讯引用:** 2303 | [OpenAlex ID](https://openalex.org/A5083720030)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LLM对齐中，提出了一种基于Nash学习的在线迭代算法（ENPO），通过显式探索提升对齐效果，解决了传统RLHF在偏好循环或非传递性场景下的局限；

**💡 创新点**

创新点在于：①将SFT式惰性正则化与对抗性策略探索相结合，实现无显式偏好模型估计的直接策略优化；②证明在一般偏好模型下，ENPO可获得O(√T)的后悔界，且无指数级KL正则化依赖；③在访问最优对策/极小极大算子时进一步提升到O(log T)；

**🔧 技术方法**

技术主要包括：KL正则化的镜像下降（mirror‑descent）框架、SFT式惰性正则化、对抗性策略探索（adversarial policy sampling）、DPO风格的损失函数以及最优对策/极小极大oracle；

**📊 数据集**

实验使用Llama‑3‑8B‑Instruct模型，并在四个对齐评测基准上验证：AE2（长度控制win‑rate）、MT‑Bench、Arena‑Hard（对战胜率）等；

**📈 对比分析**

与XPO、INPO等现有NLHF方法对比，ENPO在所有评测指标上均优于基线，尤其在高步长（大step）下保持稳定并明显优于INPO；

**⚠️ 局限性**

局限性包括：需要每轮O(t)的样本批量，且理论证明依赖有限策略族与eluder维数；对连续/大规模策略空间的直接扩展尚未完全验证；

---

## 233. Agent Skills Should Go Beyond Text: The Case for Visual Skills

**arXiv ID:** 2606.01414 | [PDF](https://arxiv.org/pdf/2606.01414v1)

**作者:** Binxiao Xu `[一作]` (Peking University), Hang Hua `[通讯]` (MIT-IBM Watson AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出可视化技能（Visual Skill）多模态框架，以突破文本瓶颈提升视觉中心任务性能

**💡 创新点**

创新点在于将技能拆分为静态、动态、交错三类，配合视觉先验与绑定协议，将空间结构与文本逻辑分离

**🔧 技术方法**

采用大型多模态模型、视觉先验生成（图像、截图、动态轨迹）与 AutoVisualSkill 自动化作者管道

**📊 数据集**

使用 GUI grounding 数据集（ScreenSpot、ScreenSpot‑v2、GroundUI‑18K）及 Dense Counting 数据集 CountBenchQA

**📈 对比分析**

通过直接提示、文本技能提示、可视化技能提示三种设置对比，结果显示可视化技能在点中框精度、IoU、计数精确率等指标显著提升，文本瓶颈得分约 8‑70%

**⚠️ 局限性**

局限在于视觉先验需生成或手工设计，可能不适用于高度开放或符号化任务；对强大模型的依赖仍未完全消除

---

## 234. UR-JEPA: Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures

**arXiv ID:** 2606.01443 | [PDF](https://arxiv.org/pdf/2606.01443v1)

**作者:** Triet M. Le `[一作]` `[通讯]` (Spatiolyx LLC), Triet M. Le (Spatiolyx LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新的自监督学习框架 UR‑JEPA，将 Joint‑Embedding Predictive Architecture 的表示分布正则化目标从全维高斯改为符合均匀 n‑可测度的低维结构；

**💡 创新点**

创新点在于把几何测度理论中的均匀 n‑可测度（通过 Carleson 方差函数和 Jones β‑数）直接转化为可微的 SGD 损失，提供了既能避免表示崩溃又能体现数据本身低维流形的正则化；

**🔧 技术方法**

采用的技术包括 JEPAs 的预测损失、SIGReg 的高维高斯匹配、Gaussian‑kernel smoothed Carleson 损失（^CGLT）以及基于局部 PCA 的 β‑数损失；

**📊 数据集**

在 ImageNet‑10、ImageNet‑100、Galaxy10 SDSS、EuroSAT 等四个公开数据集上进行实验；

**📈 对比分析**

与 LeJEPA(SIGReg) 的匹配配置相比，UR‑JEPA 在 ImageNet‑10 上提升了约0.83pp，且种子方差降低约30%，在 Galaxy10、Inet100、EuroSAT 上保持相同的最高准确率；

**⚠️ 局限性**

局限性包括需额外设定目标内维数 n、计算量略高、β‑数损失易陷入点质崩溃且需额外的 anti‑collapse 机制。

---

## 235. Quantizing Intent: Cross-Domain Semantic IDs from Organic Activity for Industrial Ranking

**arXiv ID:** 2606.01396 | [PDF](https://arxiv.org/pdf/2606.01396v1)

**作者:** Julie Choi `[一作]` (LinkedIn), Arpita Vats `[通讯]` (LinkedIn)

**通讯引用:** 688 | [OpenAlex ID](https://openalex.org/A5020025503)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种跨域用户语义ID（SID）和离散化方法，以在广告点击率（CTR）预测中利用有机 Feed 行为信号，提升冷启动用户的点击预测质量。

**💡 创新点**

1) 通过“行为活跃度丰富性”原则证明跨域 SID 的效果随源表示中包含的用户行为信息量递增；2) 设计RQ‑FSQ量化方法，将残差VAE量化与每维度有限标量量化结合，既保留全局几何，又保持细粒度结构；3) 开发可扩展的层次离散嵌入（HDE）模块和多源 SID 架构，并利用主干模型进行缺失源的回退。

**🔧 技术方法**

使用残差量化（RQ‑KMeans、RQ‑VAE）、有限标量量化（FSQ）、结合的RQ‑FSQ、层次离散嵌入（HDE）、基于哈希的前缀 n‑gram 表、以及对预训练用户嵌入的对比微调。

**📊 数据集**

工业级广告日志（60 天训练+1 天验证），以及公开的 MovieLens‑100K 作为验证基准。

**📈 对比分析**

与无 SID 基线及不同单源/多源 SID 组合进行对比，单源 SID 在行为信息量递增时 AUC 逐步提升；多源 SID 与结构化组合相比提升约 +0.036%；RQ‑FSQ 在预训练嵌入上匹配或略优于密集浮点嵌入，同时将存储压缩 30–280 倍；在冷启动用户（约 8%）中单一 Feed SID 直接提升 +1.522% 的 AUC。

**⚠️ 局限性**

缺失源的回退依赖主干模型，可能导致部分信息损失；哈希冲突仍是潜在瓶颈；在非常低维或极大 K 值的情况下，性能提升有限；跨域特征间的对齐仍主要通过下游 CTR 目标实现，未使用显式域适配损失。

---

## 236. Dynamic Breadth First Search with Predictions

**arXiv ID:** 2606.01187 | [PDF](https://arxiv.org/pdf/2606.01187v1)

**作者:** Shahbaz Khan `[一作]` (Indian Institute of Technology), Utkarsh Lohiya `[通讯]` (Indian Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套基于预测模型的动态 BFS 树维护算法，能够在增删边操作序列中根据预测误差实现自适应更新；

**💡 创新点**

创新点在于引入误差度量（η_e、η_v、η^*_v）使得算法在预测完美时可达到 O(1) 更新时间，在预测不完全时仍保持 O(η_e+η_v) 或 O(min{m,η_e+η^*_v}) 的更新复杂度，并给出增量、递减和全动态三种场景；

**🔧 技术方法**

核心技术是对经典 Even‑Shiloach 树进行批处理扩展，利用预处理好的预测 BFS 树快照和上层父节点列表来快速修复；

**📊 数据集**

论文并未在实际数据集上进行实验，而是通过理论分析和错误修正机制证明算法性能；

**📈 对比分析**

与传统 ES‑树相比，本文在最坏情况仍不超过 O(m) 的更新时间，在理想预测下可实现 O(1)；

**⚠️ 局限性**

主要局限在于预处理阶段需要 O(mn) 或 O(m^2) 的时间与空间，且在高误差情况下仍需退回到重建 BFS 树的 O(m) 方案。

---

## 237. CRePE: Convolution-aware Relative Importance in Post-training Pruning with Efficient Search

**arXiv ID:** 2606.01544 | [PDF](https://arxiv.org/pdf/2606.01544v1)

**作者:** Cheonjun Park `[一作]` (Hankuk University of Foreign Studies), Cheonjun Park `[通讯]` (Hankuk University of Foreign Studies)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5034556789)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文提出了CRePE方法，对大语言模型进行无训练剪枝，加入二维邻域上下文和自适应系数提升重要性评分，并同时提出PHO代理优化框架。

**💡 创新点**

创新点在于：1）在RIA基础上加入2D邻域信息并引入可调行列权重；2）利用Gini系数与PPL高度相关性构建代理目标，显著降低超参数搜索时间。

**🔧 技术方法**

使用了相对重要性评分、二维卷积邻域求和、CMA‑ES超参数优化、Gini系数代理、无训练剪枝技术，以及与Channel Permutation、非均匀稀疏分配和重剪枝等方法的组合。

**📊 数据集**

实验数据集包括WikiText‑2用于语言模型困惑度评估，以及BoolQ、MNLI、RTE、HellaSwag、ARC‑Easy、ARC‑Challenge六个任务用于Zero‑shot准确率测试。

**📈 对比分析**

与Magnitude、Wanda、SparseGPT、RIA等基线在Unstructured、2:4、4:8稀疏设置下进行对比，CRePE在LLaMA、Qwen、Phi‑4、DeepSeek等多种模型上 consistently 获得最低PPL和最高Zero‑shot平均准确率；搜索时间从约11小时降至约20分钟。

**⚠️ 局限性**

局限性包括：1）极高稀疏率（>70%）时优势减弱；2）Gini系数与PPL关联性缺乏理论分析；3）仅适用于密集线性层，未扩展到MoE等结构；4）额外卷积操作在不同硬件上的效率未系统评估。

---

## 238. GovAI-Pipe: A Layered AI Governance Pipeline for Citizen-Facing AI in Turkey's e-Government Gateway

**arXiv ID:** 2606.01417 | [PDF](https://arxiv.org/pdf/2606.01417v1)

**作者:** Ahmet Kaplan `[一作]` (Istanbul Medipol University), Ahmet Kaplan `[通讯]` (Istanbul Medipol University)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5011032619)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 GovAI-Pipe 四层治理管道，用于土耳其 e-Devlet 平台的 AI 服务实现技术治理，涵盖预部署验证、部署治理、运行时监控和后事件治理。

**💡 创新点**

创新点在于将欧盟 AI 法案、OECD 原则与土耳其国内法规映射到可操作的技术管道门控，实现从政策到可审计技术控制的闭环，并引入多层风险分类与人机协同审批流程。

**🔧 技术方法**

使用 MLOps 架构、偏差测试套件（AIF360、Fairlearn）、解释性工具（SHAP、LIME）、漂移检测（PSI、KL）、模型注册与治理工作流以及不可篡改的审计链等技术。

**📊 数据集**

示例用例基于 e-Devlet 交互日志和社会福利数据，模型训练使用公开或内部政府数据，但论文未列明具体数据集名称。

**📈 对比分析**

通过与 EU AI Act 工具箱、行业 MLOps 平台以及现有 eGov 框架的对比分析，GovAI-Pipe 在所有治理维度上达到或优于现有方案；但未给出实验性的性能指标，评估主要基于分析和示例说明。

**⚠️ 局限性**

局限性包括未通过实验验证、示例为说明性用例、法规环境可能演变导致实现调整，以及假设单一中心化政府架构，难以直接迁移到联邦式系统。

---

## 239. RLVR without Ineffective Samples: Group Prioritized Off-Policy Optimization for LLM Reasoning

**arXiv ID:** 2606.01281 | [PDF](https://arxiv.org/pdf/2606.01281v1)

**作者:** Yixiu Mao `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**通讯引用:** 11426 | [OpenAlex ID](https://openalex.org/A5024401174)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Group Prioritized Off‑Policy Optimization（POPO），通过重放有效的prompt‑group并纠正离线偏差来提升LLM推理训练效率

**💡 创新点**

创新点在于组级优先重放与离线样本的解耦重要性采样，既消除无效样本又保持可信的trust‑region约束

**🔧 技术方法**

采用离线重放（FIFO递归）、重要性权重分解、GRPO基本算法及RLVR奖励判定

**📊 数据集**

使用DeepScaleR数学题集、Countdown数值规划、Geometry3k视觉几何及多项通用推理基准（MMLU‑Pro、ARC‑c、GPQA‑diamond）

**📈 对比分析**

与GRPO、DAPO、MoPPS、ARPO等基线比较，POPO实现与DAPO相当的推理精度，却仅消耗30%–70% rollout预算，提升有效样本比例至>50%

**⚠️ 局限性**

局限在于重放未必选取最具信息量的样本，需探索短期与长期缓冲混合策略，且对极端难题的离线数据仍有限

---

## 240. A Fiber Criterion for Representation Identifiability in Supervised Learning

**arXiv ID:** 2606.01092 | [PDF](https://arxiv.org/pdf/2606.01092v1)

**作者:** Vasileios Sevetlidis `[一作]` `[通讯]` (Athena Research Center), Vasileios Sevetlidis (Athena Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究监督学习中预测器因子化后内部表示的可识别性问题，阐明哪些表示属性仅由预测器行为决定

**💡 创新点**

提出“纤维/下沉准则”：表示属性可识别当且仅当它在所有同一预测器的因子化上保持不变；给出“预测器保持增广”作为构造不可识别性的通用证据

**🔧 技术方法**

利用概率论与函数映射的纤维分析、信息瓶颈、对称群作用理论、预测器保持增广构造、以及对水鸟数据的实验诊断

**📊 数据集**

在实验中使用CelebA、CIFAR-10、STL-10、OfficeHome、PACS等公开数据集，以及专门的Waterbirds数据集进行匹配性能对比

**📈 对比分析**

通过构造同一预测器下的不同表示，展示诊断指标（probe准确率、等变距离、有效秩、CKA相似度、域解码）可变化；在Waterbirds实验中，在相同测试精度下不同约束（瓶颈、VIB、增广不变、SupCon）得到显著不同的表示诊断，性能相近但鲁棒性和表示结构差异明显

**⚠️ 局限性**

局限性：分析以单一预测器行为为观察对象，未覆盖所有可能的表示学习机制；实验仅评估线性解码和特定诊断指标，未全面覆盖表示几何或语义；结论依赖可接受的增广空间和模型族，实际实现时需根据任务约束选择

---

## 241. The Case for Model Science: Verify, Explore, Steer, Refine

**arXiv ID:** 2606.01189 | [PDF](https://arxiv.org/pdf/2606.01189v1)

**作者:** Przemyslaw Biecek `[一作]` (Center for Credible AI), Wojciech Samek `[通讯]` (Technical University of Berlin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并提出了构建模型科学（Model Science）框架，强调在模型验证、解释、控制和优化四个视角下系统研究模型，而非仅靠基准测试。

**💡 创新点**

将跨学科方法整合为统一的四视角框架，并倡议建立数据集、模型和发现的共享目录，推动对已部署模型的深度、机制化分析。

**🔧 技术方法**

提出基于验证、探究、调优、细化的四类方法，并借鉴认知科学、神经科学、医学与农业等领域的实验与治理模型。

**📊 数据集**

主要以已有 benchmark（如 ImageNet、GLUE、SuperGLUE、HELM 等）和开源模型下载量等指标为例，强调需对多样化数据集进行系统 catalog。

**📈 对比分析**

论文并未提供实验评估，而是讨论如何通过验证工具、红队、对比分析等方式评估模型行为，指出传统基准的局限性并呼吁更细粒度的性能比较。

**⚠️ 局限性**

关键局限在于缺乏统一的共享平台和长期资金支持，且现有技术仍未能提供完全可解释、可控制的模型机制，且不同社区的整合与共识尚未成熟。

---

## 242. Schedule-Level Shared-Prefix Reuse for LLM RL Training

**arXiv ID:** 2606.01143 | [PDF](https://arxiv.org/pdf/2606.01143v1)

**作者:** Pengbo Li `[一作]` (Hong Kong University of Science and Technology), Kai Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 23638 | [OpenAlex ID](https://openalex.org/A5100438001)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大语言模型RL后训练，设计了调度层级的前缀重用方案，将前缀的前向和反向计算只执行一次，余下的后缀在微批中使用缓存的K/V并累计梯度后一次性反向。

**💡 创新点**

创新点在于：①三阶段调度（前缀前向→后缀微批→前缀反向）实现跨微批前缀共享；②将前缀K/V与梯度KV做生命周期管理，实现前缀激活的主动下沉与复用；③兼容多维并行（TP/EP/CP/PP/DP）并保持MoE的aux-loss负载平衡。

**🔧 技术方法**

使用K/V与梯度KV缓存、异步CPU‑GPU前缀激活下沉、阶段级调度、阶段本地PP重排、CP层级KV预取、MoE辅助损失逻辑计数，全部在TorchTitan框架下实现。

**📊 数据集**

实验数据集包括Llama3‑8B、Qwen3‑8B、Qwen3‑MoE‑30B‑A3B；RL轨迹来自MuSiQue、2WikiMultiHopQA、Video‑MME、Video‑R1、ChartQA、AI2D等固定上下文RAG与多模态RL任务。

**📈 对比分析**

通过与完整序列基线在参数差异、RL轨迹回放、速度与内存消耗对比验证。结果显示：在前缀占比高、轨迹数大时可达4.4×加速、Phase‑B内存减少59%、总序列长度提升1.66倍。

**⚠️ 局限性**

局限性：主要适用于共享大前缀且无法打包进单图的高前缀长上下文RL；低前缀或单轨迹场景收益有限；对批量耦合MoE路由不保证完全一致；编译覆盖度有限，未覆盖所有后端优化。

---

## 243. On the Evaluation of Spiking Neural Network Configurations for Network Intrusion Detection

**arXiv ID:** 2606.01442 | [PDF](https://arxiv.org/pdf/2606.01442v1)

**作者:** Raj Patel `[一作]` (University of Alabama), Shahram Rahimi `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在四个入侵检测基准上，对9种神经元模型与3种脉冲编码方式共27种配置进行了系统的消融实验，训练并评估了540个SNN模型，比较其检测质量与推理成本。

**💡 创新点**

发现脉冲编码比神经元模型更决定检测质量，并给出了最佳配置（Latency编码+LeakyParallel模型），为低延迟、资源受限环境提供了可重复的SNN部署指导。

**🔧 技术方法**

使用Spiking Neural Networks、rate/latency/delta三种脉冲编码、9种轻量级神经元（Leaky、LSTM、SConv等）以及PyTorch实现的后向传播surrogate梯度训练技术。

**📊 数据集**

采用公开的四大网络流量数据集：NSL‑KDD、KDDCup99、CIC‑IDS2017和CTU‑13。

**📈 对比分析**

通过宏F1、准确率、MCC、检测率、误报率、推理延迟等多指标一致性评分进行比较，Latency编码+LeakyParallel在宏F1≈0.80、准确率≈92%、误报≈2%、推理≈0.07 ms/样本时表现最佳。

**⚠️ 局限性**

局限性包括仅评估非自适应攻击场景；使用官方或随机拆分，未覆盖跨场景未知攻击；推理时间测量在GPU同步环境下，实际硬件性能未知；未与传统非SNN基线进行直接对比。

---

## 244. TextFake: Benchmarking AI-Generated Image Detection on Text-Rich Images

**arXiv ID:** 2606.01050 | [PDF](https://arxiv.org/pdf/2606.01050v1)

**作者:** Yuning Zhang `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 24066 | [OpenAlex ID](https://openalex.org/A5064573190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了TextFake基准，包含20,000张包含文本的人工生成与真实图像，用于评估AIGI检测方法。

**💡 创新点**

创新点在于构建分布对齐、跨语言、跨场景、跨主题的文本丰富AIGI基准，并系统揭示三大失效模式（文本密度、渲染保真度、阈值崩溃）。

**🔧 技术方法**

采用四阶段流水线：多源文本图像采集 → OCR与VLM标注 → 结构化提示生成 → 元数据标准化，并对空间/频域、基础模型及VLM API三类检测器进行评估。

**📊 数据集**

使用自建的TextFake数据集（10k真图+10k假图，28种语言、4类主题、屏幕/纸张两种场景），并参考GenImage做对照。

**📈 对比分析**

在零样本二分类设置下对14种专用检测器和3种VLM API进行对比，最高准确率仅79.3%（GAPL），其余方法普遍低于80%，空间/频域方法在文本密度升高时表现急剧下降。

**⚠️ 局限性**

局限性包括：仅覆盖当前最先进的文本渲染生成器；语言深度偏向高资源语言；仅评估二分类零样本性能，未涉及来源归因或区域检测；数据集获取受使用协议限制。

---

## 245. Connecting the Dots: Benchmarking Reflective Memory in Long-Horizon Dialogue

**arXiv ID:** 2606.01223 | [PDF](https://arxiv.org/pdf/2606.01223v1)

**作者:** Jingjie Lin `[一作]` (Harbin Institute of Technology), Ruifeng Xu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 7467 | [OpenAlex ID](https://openalex.org/A5026719663)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 RefMem‑Bench 这一面向长对话的反思性记忆评测基准，并设计了 REMIND 这一分层框架来提升模型在多模态、分散线索中的高层推理能力。

**💡 创新点**

创新点：①将反思性记忆定义为与显式回忆不同的评测目标，涵盖 8 个维度和 3 种任务格式；②引入三层认知金字塔（事实、注意、反思）以及 Progressive Reflective Alignment，实现在训练中把高层推理“蒸馏”进底层推理路径；③通过稀疏编码与策略引导的注意力机制，实现对多模态线索的语义共振与本地化提升。

**🔧 技术方法**

技术：分层检索与证据感知、稀疏自编码（Top‑K SAE）构建注意力指导、反思性摘要生成、双向 KL 蒸馏的 Progressive Reflective Alignment、以及基于预训练大模型（如 Qwen3‑VL‑8B、GPT‑4o‑mini、CLIP、MiniLM）的多模态编码与检索。

**📊 数据集**

数据集：由 REALTALK、DialSim、LoCoMo 等长对话集合自动与人工混合构建，包含 26,449 条 QA 实例，覆盖 35 个长上下文，71K 轮，含文本与图像证据；所有实例均由三名专家标注、验证，并按 2:8 划分训练/测试。

**📈 对比分析**

对比方法：包括无检索（Qwen3‑VL‑8B 基线）、多种训练无检索与训练有检索的记忆增强模型（MemGPT、Mem0、MemoryOS、A‑Mem、LightMem、GAM 等）。在 RefMem‑Bench 上，REMIND 在三种任务格式（单选、多选、自由答）和所有衡量指标（准确率、记忆召回、BLEU‑1、F1）均显著优于基线，尤其在自由答任务中取得 26.2、13.1、21.2 的显著提升。

**⚠️ 局限性**

局限性：① 依赖大量人工标注与 GPT‑4o‑mini 等大模型辅助生成，成本高；② 反思性摘要仅在训练阶段使用，推理时仍需较大模型；③ 对超长对话或极其稀疏的多模态线索的鲁棒性尚未充分验证。

---

## 246. FlowTime: Towards Continuous Generative Watch Time Prediction via Flow-based Personalized Priors

**arXiv ID:** 2606.01352 | [PDF](https://arxiv.org/pdf/2606.01352v1)

**作者:** Hongxu Ma `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 11356 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在短视频推荐中提出 FlowTime，用一步变分自编码器与流式个性化先验实现连续生成式观看时长预测。

**💡 创新点**

创新点在于把观看时长视作连续分布，利用条件正则化的流模型捕捉用户-视频交互模式的多模态分布，并通过一阶推断消除迭代推理延迟。

**🔧 技术方法**

采用 VAE + 条件正则化的 Normalizing Flow、Transformer 编码量化历史分布、Wasserstein 损失和 Huber 损失进行联合训练。

**📊 数据集**

在 KuaiRand-Pure、KuaiRec 与工业级 Indust 三个数据集上进行实验。

**📈 对比分析**

与传统直接回归、序数回归及离散生成回归相比，FlowTime 在 MAE、XAUC 与 PDF-U/I 等指标上均领先 1–2% 并在线上 A/B 测试中提升 1% 左右的关键业务指标。

**⚠️ 局限性**

局限性包括对流深度与用户/项目权重的敏感性、对历史量化方式的依赖，以及在极端稀疏场景下仍需进一步提升泛化能力。

---

## 247. Spatio-Temporal Reconnection for Multi-Robot Networks using Adaptive Prescribed-Time CBFs

**arXiv ID:** 2606.01526 | [PDF](https://arxiv.org/pdf/2606.01526v1)

**作者:** Hao Liu `[一作]` (University of Illinois Chicago), Wenhao Luo `[通讯]` (University of Illinois Chicago)

**通讯引用:** 84682 | [OpenAlex ID](https://openalex.org/A5101473760)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种自适应预定时控制壁函数（Adaptive PT-CBF）与时空触发机制，用于多机器人系统的暂时脱连与快速再连，保证在预定时间窗口内实现联合全局连通性并尽量减少对原始任务控制的偏差。

**💡 创新点**

创新点包括：1）将有限时控制壁函数（FT-CBF）与预定时控制壁函数（PT-CBF）相结合，在线自适应选择预定时间；2）设计基于空间距离与时间紧迫度的时空权重触发机制，仅在必要时激活再连约束；3）提供理论保证在每个时间窗口内必定完成所有必需的通信边，且不需要事先指定固定的再连时间。

**🔧 技术方法**

技术手段主要有：控制壁函数（CBF）、有限时/预定时 CBF（FT-CBF/PT-CBF）、自适应 PT-CBF、时空触发权重、二次规划（QP）控制器、单积分/无刷转子动力学映射、Robotarium 与 CoppeliaSim 仿真平台。

**📊 数据集**

数据集/仿真场景包括：1）Robotarium 机器人平台的两机器人再连实验；2）CoppeliaSim 进行五机器人巡逻任务的仿真；3）对比基线方法 Minimum Connectivity Constraint Spanning Tree（MCCST），以及预定时间 PT-CBF（不同 T_p）和 FT-CBF 进行对比。

**📈 对比分析**

与基线方法相比，Adaptive PT-CBF 在以下指标上表现更优：a) 更快地实现通信壁函数值与实际距离达到阈值；b) 控制偏差更小，任务执行更接近原始轨迹；c) 在多机器人尺寸与再连窗口长度变化时保持控制偏差可接受。与固定 T_p 的 PT-CBF 对比显示，自适应选择时间避免了过早或过迟再连的问题；与 FT-CBF 对比显示在接近边界时更稳健。

**⚠️ 局限性**

局限性包括：1）需要预先指定联合全局连通的目标图和再连窗口，难以处理动态网络拓扑变化；2) 触发机制依赖于手工设定的时间上限与空间阈值，可能在极端环境下失效；3) 对辅助通信层（大范围轻量级信息交换）的假设在实际部署中可能不成立；4) 对大规模机器人团队的可扩展性尚未在真实实验中验证。

---

## 248. Type-Error Ablation and AI Coding Agents

**arXiv ID:** 2606.01522 | [PDF](https://arxiv.org/pdf/2606.01522v1)

**作者:** Shriram Krishnamurthi `[一作]` (Brown University), Matthew Flatt `[通讯]` (University of Utah)

**通讯引用:** 8156 | [OpenAlex ID](https://openalex.org/A5033713909)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 AI 编码助手在不同错误信息详细程度下的修复效果进行实验评估，探讨是否需要为 AI 和人类分别设计错误报告模式。

**💡 创新点**

首次将错误信息的冗余度与 AI 修复成功率关联，提出“人类模式 vs AI 模式”两种错误报告策略，并验证其有效性。

**🔧 技术方法**

使用 Shplait（ML‑style 静态类型语言）实现四种错误报告模式，利用免费开源 LLM（如 CodeLlama via Ollama）配合 Aider 进行自动化交互，构建了类型错误与测试失败判别的 oracle。

**📊 数据集**

基于十个标准数据结构/算法程序（如 AVL 树、Dijkstra、Huffman 等），每个程序产生 6 个单一类型错误的变体，总计 60 个 chaff，进一步在 10 轮实验中共 2400 次试验。

**📈 对比分析**

通过对比四种错误报告模式的成功率（untyped → min → proximate → all）发现更详细的错误信息显著提升 AI 的修复率；在有类型系统的模式下，成功率平均从 33.5% 提升至 52.8%；同一模型在无类型模式下表现最差。

**⚠️ 局限性**

实验规模有限（仅 10 程序、单一 LLM、单错误变体），未检验多错误、复杂程序或其它语言的普适性；未区分类型信息与测试反馈对结果的独立贡献；结果对具体硬件和模型随机性的依赖较大，缺乏对不同模型、语言和规模的泛化验证。

---

## 249. TERRA: Task-Embedded Reasoning and Representation Architecture for Cross-Domain Applications

**arXiv ID:** 2606.01520 | [PDF](https://arxiv.org/pdf/2606.01520v1)

**作者:** Shayan Shokri `[一作]` `[通讯]`, Shayan Shokri

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出跨域结构化状态预测模型的理论框架及其转移界限，并设计可验证的实验程序。

**💡 创新点**

将跨域对应定义为近似MDP同态，利用松弛的同构度量与Gromov‑Wasserstein距离量化结构匹配，推导出预测误差与决策回报的分离上界。

**🔧 技术方法**

掩码潜在预测、动作条件潜在世界模型、离散动作标记、联合嵌入预测、MDP同态、松弛同化度量、Gromov‑Wasserstein距离、深度MDP等。

**📊 数据集**

CARLA、Habitat、nuScenes、Waymo、KITTI、Open X‑Embodiment等驾驶与室内任务数据，以及计划在公开交易所订单簿数据上进行跨域验证。

**📈 对比分析**

论文未给出实验结果；计划通过冻结、少量微调和完全迁移三种方式与从零训练对比，测量线性探针准确率、预测误差及回报差距。

**⚠️ 局限性**

仅为理论与实验设计，缺乏经验验证；估计结构差距难度大；假设领域可用格网结构；需要大量标注和激励；Gromov‑Wasserstein下限仅在等距归一化时成立；对分布漂移和表示坍塌风险未解决。

---

## 250. Semantic Retrieval for Product Search in E-Commerce

**arXiv ID:** 2606.01504 | [PDF](https://arxiv.org/pdf/2606.01504v1)

**作者:** Nikhil Kothari `[一作]` (Flipkart), Surender Kumar `[通讯]` (Flipkart)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Flipkart 电商平台上构建并部署了一套基于大模型的语义检索系统，能够准确处理短小、嘈杂且口语化的用户查询，并在产品目录中实现精确匹配与补充性商品排序。

**💡 创新点**

创新点主要包括：① 两阶段训练框架——先用带假负样本掩码的对比学习构建统一嵌入空间；② 引入 ROAR（Relative Odds Alignment for Retrieval）目标，对分级相关性进行偏好优化，扩展 Bradley–Terry 机制到可变大小的分级组；③ 结合多来源数据（点击行为、推荐替代、会话重写、LLM 生成同义词和攻击式扰动）和 MRL 低维投影实现高效上线。

**🔧 技术方法**

采用的技术包括：Qwen3-Embedding-4B 解码器仅模型作为双塔编码器；LoRA 细粒度微调；False-Negative Margin Masking；ROAR 偏好对齐损失；跨设备负样本采样；vLLM 推理；以及对比损失与 ROAR 的加权组合。

**📊 数据集**

使用的数据集为 Flipkart 自有搜索日志、人工标注的 Exact/Substitute/Complement/Irrelevant 语义标签、基于点击与购买的隐式反馈、推荐系统的替代品信号、会话重写数据、LLM 生成的同义词与攻击扰动，并通过多语言与转写扩展覆盖印度子大陆。

**📈 对比分析**

与生产基线（6 层 BERT 双塔模型）和多种 ablation 配置对比，最终模型在 MAP@8、NDCG@8、AUC 等指标上分别提升 5.57、3.56、13.13 点；在不同查询频段与业务垂直上均有显著提升；上线 A/B 实验中 CTR、ATC、订单分别提升 2.39%、4.58%、2.62%。

**⚠️ 局限性**

局限性包括：评估仅基于 Flipkart 内部数据，缺乏公开基准；内部注释准则与标注一致性未公开，导致可重复性受限；主要以英文（含转写）为主，缺乏完整的本土语言评测；A/B 测试周期仅三周，无法观察长期影响；ROAR 需要分级相关性标签，标注成本较高。

---

## 251. CART: Context-Anchored Recurrent Transformer -- A Parameter-Efficient Architecture with Learned Stability

**arXiv ID:** 2606.01495 | [PDF](https://arxiv.org/pdf/2606.01495v1)

**作者:** Chad A. Capps `[一作]` `[通讯]` (Independent Researcher), Chad A. Capps (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种参数高效的循环Transformer架构——Context-Anchored Recurrent Transformer（CART），通过预先计算一次键值缓存并在多次循环中复用，结合学习的LTI门控实现稳定的递归更新；

**💡 创新点**

创新点在于：①预先一次性计算键值并在循环中冻结，减少每次循环的KV投影成本；②采用可学习的sigmoid LTI门控保证谱半径<1，实现递归稳定性；③将跨注意力与循环状态分离，避免未来信息泄漏；

**🔧 技术方法**

技术上使用多头潜在注意力（MLA）压缩KV、超循环连接（HyperConnection）混合过去状态、循环索引嵌入（LIE）标记迭代步骤，并通过LTI门实现可学习的衰减；

**📊 数据集**

在单个混合语料上训练：TinyStories（30%）、Wikipedia（30%）和FineWeb-Edu（40%），共约1B个标记，使用Llama-2 tokenizer；

**📈 对比分析**

通过Stage 1的64配置快速屏蔽和Stage 2的36配置全训练评估；与参数匹配的Dense Transformer进行对比，发现CART在参数匹配下略逊于Dense（1–2%），在有效参数匹配下差距约10%；在自然语言数据集上性能提升有限；

**⚠️ 局限性**

主要限制包括：①在1B-token训练下模型未达到Chinchilla极限；②循环深度对性能提升有限，R值增大无明显收益；③某些机制（HyperConnection、LIE、LTI门）在共享权重设置下对性能影响甚微；④单GPU单机训练规模受限，未验证更大规模或多GPU加速效果；

---

## 252. Cross-lingual Self-Consistency for Multilingual Reasoning with Language Models

**arXiv ID:** 2606.01464 | [PDF](https://arxiv.org/pdf/2606.01464v1)

**作者:** Ahmed Elhady `[一作]` (University of Basque Country), Mikel Artetxe `[通讯]` (University of Basque Country)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无监督强化学习方法，利用跨语言自一致性（cross‑lingual self‑consistency）提升大语言模型的多语言推理能力，只使用英文单语问题进行训练，并通过模型自翻译生成多语言并行数据；

**💡 创新点**

创新点在于把单语言自一致性奖励扩展到跨语言，使模型在没有黄金答案或平行语料的情况下，强制不同语言的推理结果保持一致，从而显著提升弱语种的表现并在未见语言和领域上具备良好泛化；

**🔧 技术方法**

使用强化学习（GRPO）进行跨语言自一致性奖励训练；利用模型自翻译、多语言并行生成、交叉熵奖励（以英语或所有语言的答案分布为目标）以及推理时多数投票；

**📊 数据集**

训练数据：英语 GSM8k 1k 问题；评估数据：MGSM（10 语言的数学推理题）；泛化测试：mGPQA、MMATH、PolyMath；实验中亦使用自翻译后的 MGSM 子集进行对照；

**📈 对比分析**

与基线（native‑cot、translate‑test、S1、LIDR 等）以及已有公开方法对比，平均在 MGSM 上提升至 7.8%–21.7%（按模型规模递增），在未见语言上提升 18.2%，在 OOD 基准上获得 4.3%–6.2% 的增益，整体优于所有对比方法；

**⚠️ 局限性**

受限于算力仅能实验 20B 以内模型，主要使用 Qwen2.5 系列；多语言推理基准稀缺导致验证范围有限；对更大模型或新架构的效果尚未评估；

---

## 253. TwinQuant: Learnable Subspace Decomposition for 4-Bit LLM Quantization

**arXiv ID:** 2606.01556 | [PDF](https://arxiv.org/pdf/2606.01556v1)

**作者:** Haodong Wang `[一作]` (Hong Kong University of Science and Technology), Xu Chen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 473578 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出TwinQuant，一种针对4‑bit低精度推理的可训练子空间分解框架，能够在保持几乎FP16准确率的同时实现显著加速；

**💡 创新点**

创新点在于：①通过联合优化Stiefel（正交）与General Linear（可逆）流形，学习权重的低秩与残差分支的量化友好子空间；②设计融合双分支的4‑bit GEMM核，避免中间全量化存储，提升内核吞吐量；③全流程无需额外训练，只需在小量标定数据上做离线优化；

**🔧 技术方法**

技术包括：可学习子空间分解（低秩+残差），旋转与可逆变换的联合优化，混合流形梯度下降（Cayley更新+极化参数化），以及定制的双分支融合核；

**📊 数据集**

使用的基准数据集为WikiText2（128句标定），六个零样本推理任务（ARC‑Challenge/Easy、HellaSwag、LAMBADA、PIQA、WinoGrande）进行评测；

**📈 对比分析**

与RTN、GPTQ、AWQ、SmoothQuant、QuaRot、SpinQuant、FlatQuant、SVDQuant等8种主流PTQ方法对比。TwinQuant在LLaMA3与Qwen3系列模型上，4‑bit推理误差仅比FP16低0.6–2.6个百分点，且在FP16 TensorRT‑LLM、AWQ等基线上实现1.3–1.8×的吞吐加速；

**⚠️ 局限性**

局限性包括：仅针对稠密LLM验证；MoE或多模态模型的分布差异尚未验证；核实现针对RTX 4090/L20 GPU，其他加速器需重调；需要标定数据且离线优化耗时；极端长上下文场景下性能仍待验证。

---

## 254. TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions

**arXiv ID:** 2606.01540 | [PDF](https://arxiv.org/pdf/2606.01540v1)

**作者:** Farzaneh Heidari `[一作]` (Université de Montréal), Guillaume Rabusseau `[通讯]` (Université de Montréal)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5023766963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对图结构的黑盒模型，提出一种通过学习图对齐张量网络（TN）近似掩码游戏的多线性扩展，从而在不需要大量蒙特卡洛采样的情况下，准确定义并计算 Shapley 值与高阶交互项。

**💡 创新点**

创新点包括：① 在图结构上构造张量网络拓扑以匹配图分割的低秩特性；② 通过多线性张量网络实现对 Shapley 值的确定性闭式恢复；③ 证明了张量网络容量与图分割秩的上界关系，并在参数匹配下优于传统张量链（TT）和基于采样的方法。

**🔧 技术方法**

核心技术：张量网络（图对齐 TN）、多线性扩展、Vandermonde 插值求解多项式、基于图分解的张量收缩、近似学习（最小二乘拟合）、基于 R² 的模型逼近评估。

**📊 数据集**

在分子与蛋白质基准数据集（Benzene、Mutagenicity、PROTEINS 等）上进行实验，使用节点掩码基线，评估单实例模型的 Shapley 解释。

**📈 对比分析**

与 GraphSVX、SHAP‑IQ 等无模型参数、基于采样的解释器比较，TN‑SHAP‑G 在小图（≤20 个节点）上实现了 0.99+ 的余弦相似度，且仅需 10–100 倍更少的模型查询；在更大图上仍能高效计算，且在多次查询时具备较低的累计计算时间。

**⚠️ 局限性**

局限性包括：仅支持固定的掩码基线与节点级玩家；高阶交互的精度仍受张量网络欠拟合影响；目前不自适应张量拓扑或秩分配，且对图分割结构要求较高。

---

## 255. ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts

**arXiv ID:** 2606.01509 | [PDF](https://arxiv.org/pdf/2606.01509v1)

**作者:** Heng Zhao `[一作]` (University of Virginia), Zhe Zeng `[通讯]` (University of Virginia)

**通讯引用:** 50707 | [OpenAlex ID](https://openalex.org/A5073474951)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可微分的概率性专家路由框架 ProbMoE，将 MoE 路由视为带卡迪纳尔约束的离散子集推理问题。

**💡 创新点**

创新点在于：① 用概率推理取代离散 Top‑k，② 采用 SIMPLE 采样+边缘概率梯度实现可微分训练，③ 引入 Dynamic‑k 机制允许每个 token 动态决定激活专家数量。

**🔧 技术方法**

技术手段包括：softmax 先验专家概率、对卡迪纳尔约束子集分布的 Exact‑k 与 Range‑constrained 归一化、SIMPLE 采样、边缘概率 (marginals) 作为梯度近似、Straight‑Through 估计。

**📊 数据集**

在 OLMoE（1B/7B）与 Qwen1.5-MoE-A2.7B 两大 MoE backbone 上进行实验，任务涵盖 GSM8K、法律推理、翻译、摘要、代码生成（MBPP）以及通用知识（MMLU）。

**📈 对比分析**

与 Frozen Router、Conventional Top‑k、DenseMixer、DefaultMoE、SparseMixer、ReMoE 等基线对比，ProbMoE Exact‑k 在多项任务上提升 1–3% 的 EM/准确率，并显著提高专家利用率与路由多样性；ProbMoE Dynamic‑k 在保持性能的同时平均减少 10–25% 的激活专家数量。

**⚠️ 局限性**

局限性包括：① 需要额外的动态规划计算归一化常数，对专家数较大时仍有计算开销；② 目前仅在微调阶段验证，未在大规模预训练中测试；③ 对分布式训练的系统实现与硬件加速仍有待进一步优化。

---

## 256. Move the Query, Not the Cache: Characterizing Cross-Instance Latent Attention Redistribution Across GPU Fabrics

**arXiv ID:** 2606.01502 | [PDF](https://arxiv.org/pdf/2606.01502v1)

**作者:** Bole Ma `[一作]` (Erlangen National High Performance Computing Center), Gerhard Wellein `[通讯]` (Erlangen National High Performance Computing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究跨实例注意力重分发，提出路由查询而非移动缓存的策略，并给出拓扑感知的成本模型和闭式选择谓词。

**💡 创新点**

① 通过压缩/稀疏注意力将查询变小，证明在大多数情形下路由查询比拉取缓存成本更低；② 对设备主动 RDMA（IBGDA）在跨实例注意力中的优势进行量化；③ 提供可复用的拓扑感知模型和决策谓词，适用于任何压缩或稀疏注意力架构。

**🔧 技术方法**

采用 Multi‑Head Latent Attention 压缩 KV、稀疏索引、FlashMLA/FlashInfer 等 Sparse Attention 实现；利用 NVSHMEM IBGDA、PCIe/NVLink 等网络，并在 4×H100 SXM5 集群上通过 CUDA/CPU 并行实现。

**📊 数据集**

使用大规模 Canonical 语料（案例法、财报、代码库等），以及 DeepSeek、GLM 等模型的压缩块/稀疏块作为实验数据集。

**📈 对比分析**

对三种重分发原语（拉取缓存、路由查询、重新预填）在 IBGDA、PCIe、NVLink 等五种网络上进行往返时延、字节传输和吞吐量测评；路由查询单块往返≈31–48 µs，远低于 3 ms 的拉取拼接；模型在大批量时误差<7%；在跨实例注意力解码阶段，路由查询比拉取快约 30–100 倍。

**⚠️ 局限性**

仍受主机侧 3.5 ms 首 token 延迟影响，端到端性能未完全赢；对非压缩/稀疏注意力模型不适用；实验依赖特定驱动与 IBGDA 硬件；主要评估在 H100+NDR‑200 环境，未覆盖更广泛的体系结构。

---

## 257. A Reproducible UAV-Assisted VANET Dataset Generator for Fragmentation Risk Analysis in Intelligent Transportation Systems

**arXiv ID:** 2606.01488 | [PDF](https://arxiv.org/pdf/2606.01488v1)

**作者:** Bappa Muktar `[一作]` (University of Quebec in Outaouais), Adama Nouboukpo `[通讯]` (University of Quebec in Outaouais)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个基于 ns‑3 的可复现 UAV‑辅助 VANET 数据集生成器，用于在两车道高速公路场景中从当前网络状态预测未来的碎片化风险。

**💡 创新点**

创新点在于：① 模块化、可扩展的仿真框架；② 结合多种交通情景、事故扰动和无人机位置策略；③ 在每个采样点提取车辆运动、图层拓扑、无人机覆盖及通信窗口特征；④ 通过未来时间窗口的连通性状态为样本打标签，实现从当前状态到未来碎片化的监督学习问题。

**🔧 技术方法**

使用技术包括 ns‑3 网络仿真、IEEE 802.11p 无线层、AODV 路由、图论连通性分析、特征提取与标签生成算法，以及可配置的无人机移动模型。

**📊 数据集**

使用自建的 UAV‑辅助 VANET 碎片化风险数据集，示例实验中生成 4620 条样本，涵盖 5 种交通情景和 2 种无人机策略，数据以 CSV 形式导出。

**📈 对比分析**

本文没有提出新的模型，只给出了数据集的生成方法与描述性统计（样本分布、标签分布、特征范围），并未报告机器学习模型的性能评估。

**⚠️ 局限性**

局限性包括：连接图基于理想范围模型，未考虑衰落、干扰和能量消耗；车辆运动模型为简化的两车道常速模型，缺乏真实轨迹；无人机策略简易且不考虑避碰或能量约束；未来标签阈值人为设定，可能对其他应用不适用。

---

## 258. SafeGen-Bench: Benchmarking Safety in Image-Conditioned Text-to-Video Generation

**arXiv ID:** 2606.01481 | [PDF](https://arxiv.org/pdf/2606.01481v1)

**作者:** Yingzi Ma `[一作]` (University of Wisconsin-Madison), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了SafeGen-Bench基准，评估图像-文本条件视频生成模型在十类恶意场景下的安全性。

**💡 创新点**

创新点包括：①考虑文本+起始图像组合的隐蔽风险；②构建包含392个起始帧与文本对的高质量测试集；③在多模态输入下系统化评估安全与质量，并探讨多模态守卫与对抗方法。

**🔧 技术方法**

使用技术：GPT‑4o评估视频安全，ImageGrid‑LLaVA评估视频质量；GPT‑4生成文本提示；CLIP检索起始图像；DINO + SAM‑2构造交互图像；LLaMA‑Guard、LLaVA‑Guard等守卫；多语言翻译与RAB（遗传算法）对抗。

**📊 数据集**

数据集来源：ImageNet、YouTube、MS‑EVS、Panda、WHO等多源视频/图像库，人工过滤后提取起始帧；生成的交互图像来自Human Parsing等。

**📈 对比分析**

对比方法：对五个开源模型（I2VGen‑XL、Open‑Sora‑Plan、CogVideoX）和两个商业模型（Runway Gen‑3‑turbo、Kuaishou Kling）在十类恶意类别下的安全分数与视频质量分数；结果显示商业模型平均安全分数最低、质量最高；开源模型安全性差、质量较好，体现安全-质量权衡。

**⚠️ 局限性**

限制：①开源模型缺乏复杂指令跟随，导致安全与质量难以兼顾；②单模态守卫在多模态输入下效果不足；③对抗方法在逃避守卫的同时往往降低视频质量；④基准仅覆盖起始图像+文本组合，未覆盖更复杂的多模态交互场景。

---

## 259. Engineering Students' Self-Efficacy, Perceptions, and Performance in a Flipped CS1 Course

**arXiv ID:** 2606.01471 | [PDF](https://arxiv.org/pdf/2606.01471v1)

**作者:** Griffin Pitts `[一作]` (North Carolina State University), Ashish Aggarwal `[通讯]` (University of Florida)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5101628727)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了工程专业学生在翻转课堂 CS1 课程中的自我效能、学习态度和编程难度感知，并将其与考试成绩关联。

**💡 创新点**

在非 CS 专业学生群体中首次使用探索性因子分析提取自我效能、学习态度和编程难度三大潜在因子，并探讨性别、先前编程经验与绩效的关系。

**🔧 技术方法**

采用问卷调查、探索性因子分析（主轴提取 + oblimin 旋转）、Pearson 相关、Kruskal–Wallis、Mann–Whitney U 等统计方法。

**📊 数据集**

收集了四个学期共 602 名工程学生的问卷与两次考试成绩数据（平均成绩 81.95，样本量 602）。

**📈 对比分析**

通过相关和分位数分析显示自我效能正相关、编程难度负相关；性别、先前经验对自我效能和难度感知显著，但与成绩差异有限，整体解释方差 45.7%。

**⚠️ 局限性**

样本限于单一高校翻转 CS1 课程，问卷项目未采用已验证量表，先前编程经验仅二元编码，缺乏对多元背景和其他学科的推广性。

---

## 260. LLM Consortium for Software Design Refinement: A Controlled Experiment on Multi-Agent Collaboration Topologies

**arXiv ID:** 2606.01490 | [PDF](https://arxiv.org/pdf/2606.01490v1)

**作者:** Nagarjuna Kanamarlapudi `[一作]`, Praveen K `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在软件架构设计任务上，使用12种多代理LLM协作拓扑进行520次受控实验，评估设计质量。

**💡 创新点**

发现结构化对抗式审查与跨模型审查能显著提升质量，且并行合并失败；同时揭示评估模型间的偏差。

**🔧 技术方法**

采用多代理LLM（Gemini、GPT-OSS）协作、三种评估模型（GPT-OSS、Claude Opus、Claude Sonnet）以及基于12维量表的自动评估。

**📊 数据集**

使用8个软件设计任务（URL短链、计费系统等），共520次实验（12拓扑×8任务×5次）。

**📈 对比分析**

通过加权集成（2×Opus+2×Sonnet+1×GPT-OSS）比较各拓扑，结构化对抗式获得最高平均分4.637，跨模型审查次之；并行合并得分最低；差异显著（p<1e-40）。

**⚠️ 局限性**

局限包括评估模型间一致性不足、仅使用两类生成模型、合并策略未改进、评估量表分辨率有限等。

---

## 261. Flexible Online Representation Learning Based on Similarity Matching

**arXiv ID:** 2606.01546 | [PDF](https://arxiv.org/pdf/2606.01546v1)

**作者:** Shagesh Sridharan `[一作]` (Rutgers University), Anirvan M. Sengupta `[通讯]` (Flatiron Institute, Simons Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可在线学习、可生物学上可解释的稀疏、平移不变表示学习算法，直接在完全正定（CP）空间求解，既可用于聚类、流形分块，又可用于特征学习。

**💡 创新点**

创新点在于：①在CP约束下引入行和迹约束实现平移不变性；②将社区检测、相似度匹配的目标函数统一为可解的min‑max形式；③设计了仅需局部 Hebbian 更新的单层神经网络实现在线学习。

**🔧 技术方法**

主要技术包括：完全正定矩阵优化、min‑max 拉格朗日松弛、辅助变量法、梯度下降‑上升在线算法、神经动力学与可解释的权重更新规则。

**📊 数据集**

实验数据集：四种合成流形（Bunny、圆环、非均匀环、四聚簇）以及CIFAR‑10 图像补丁。

**📈 对比分析**

与传统 k‑means、基准特征学习方法比较，在线流形分块时取得高质量局部感受野；在 CIFAR‑10 任务中得到的稀疏特征在 SVM 分类上与 k‑means 竞争，表现随输出神经元数量和稀疏性提升而改善。

**⚠️ 局限性**

局限性：对超参数（γ、β）敏感，需手工调节；目前仅在中等规模数据上验证，尚未在大规模实时数据流中测试；算法对噪声和非平衡分布的鲁棒性未充分评估。

---

## 262. RoleCDE:Benchmarking and Mitigating Role-Alignment Trade-offs in Role-Playing Agents

**arXiv ID:** 2606.01552 | [PDF](https://arxiv.org/pdf/2606.01552v1)

**作者:** Huayi Lai `[一作]` (Renmin University of China), Xun Liang `[通讯]` (Renmin University of China)

**通讯引用:** 1614 | [OpenAlex ID](https://openalex.org/A5020649203)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RoleCDE 基准，用来评估角色扮演代理在角色价值与对齐价值冲突下的决策行为，并发现了角色-价值解耦现象。

**💡 创新点**

创新点包括：①首个针对角色价值与对齐价值冲突的结构化困境评估基准；②定义了四类决策标签和决策偏差比（DBR）来量化角色一致性；③通过 RoleCDE 细调显著降低解耦，提高角色一致性决策。

**🔧 技术方法**

使用的技术包括：GPT‑4o 进行角色扩展与价值生成、LLM-as-judge 自动分类、SFT 与 DPO 细调、BLEU/ROUGE/BERTScore 等语义相似度评估。

**📊 数据集**

数据集由约 8,000 个角色-情境对与 24,000 个结构化困境实例组成，涵盖 8 类角色和 3 级难度；同时使用公开基准如 MMLU、GSM8K、RoleBench 等进行对比评测。

**📈 对比分析**

对比方法：在 17 款 LLM（闭源与开源）上进行决策与推理评估，计算 DBR 及决策类别比例；细调后模型在 RoleCDE 上 DBR 显著提升，推理相似度（BLEU/ROUGE/BERTScore）提高 15–30%，但对常规推理与角色扮演基准的影响不大。

**⚠️ 局限性**

局限性：①仅支持文本场景，未覆盖多模态或交互式输入；②侧重单步决策，未探讨多轮或长期决策；③未引入强化学习训练，可能限制策略改进；④细调效果受基准设计与标注质量的影响。

---

## 263. ForestMamba: Sparse Mamba with Geometry-guided Queries for 3D Forest Point Cloud Segmentation

**arXiv ID:** 2606.01549 | [PDF](https://arxiv.org/pdf/2606.01549v1)

**作者:** Trung Thanh Nguyen `[一作]` (Nagoya University), Teja Kattenborn `[通讯]` (University of Freiburg)

**通讯引用:** 7696 | [OpenAlex ID](https://openalex.org/A5007461436)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了 ForestMamba，一种结构感知的框架，用于 3D 森林 LiDAR 点云的联合语义与实例分割。

**💡 创新点**

创新点包括：① 采用垂直优先层序化的稀疏编码器与 Mamba 状态空间模型实现长程垂直上下文建模；② 通过多尺度树冠高度模型（CHM）峰值检测与 Farthest Point Sampling（FPS）相结合的几何引导查询初始化；③ 结合局部 kNN 聚合与双向扫描的 Mamba 查询解码器，替代二次复杂度的 Transformer 注意力。

**🔧 技术方法**

使用技术：稀疏卷积、Mamba 状态空间模型、垂直层序化、CHM 峰值检测、FPS、双向 Mamba 以及一对多匹配的实例解码策略。

**📊 数据集**

数据集：FOR-instanceV2，涵盖七个地理分布不同的森林区域（CULS、NIBIO、RMIT、SCION、TUWIEN、BlueCat、Yuchen），共计 11,134 森林树实例。

**📈 对比分析**

与 OneFormer3D、ForestFormer3D、TreeLearn、ForAINet 等基线对比，ForestMamba 在七个地区均取得最高加权平均 F1（83.4%），Recall 更高，推理速度约 3 倍快，GPU 内存消耗约 2.3 倍少，显著优于 Transformer 方案。

**⚠️ 局限性**

局限性：依赖辅助分支预测树体体素以构建 CHM，若密集多层森林中树体检测失准则影响查询初始化；固定的几何参数（α、β）可能不适用于所有森林类型，导致泛化受限。

---

## 264. Agent Operating Systems (AOS): Integrating Agentic Control Planes into, and Beyond, Traditional Operating Systems

**arXiv ID:** 2606.01508 | [PDF](https://arxiv.org/pdf/2606.01508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 265. MPMWorlds: Material-Point-Method Simulations for Inferring and Extrapolating Physical Dynamics

**arXiv ID:** 2606.01538 | [PDF](https://arxiv.org/pdf/2606.01538v1)

**作者:** Žiga Kovačič `[一作]` (Cornell University), Kevin Ellis `[通讯]` (Cornell University)

**通讯引用:** 1156 | [OpenAlex ID](https://openalex.org/A5009201646)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个2D MPM物理仿真数据集并评估了代码生成模型与视频扩散模型在物理推断与时间外推上的表现。

**💡 创新点**

提出了MPMWorld数据集，比较了可执行代码生成与直接视频生成两类方法，并揭示其各自优劣与可互补性。

**🔧 技术方法**

采用大型视觉语言模型（Qwen2.5‑VL‑7B‑Instruct）生成MPM代码，采用LongCat视频扩散模型进行直接像素预测，并利用最佳样本采样与前缀质量门控进行后处理。

**📊 数据集**

使用MPMWorld数据集，包含约95k条2D MPM仿真视频、对应的Taichi代码与YAML场景配置。

**📈 对比分析**

在held‑out父级场景分割上，以mIoU、CTV、W‑MAE、OCS、RTSJ等五项物理一致性指标评估；结果显示代码生成在时间稳定性与长期一致性上优于视频扩散，但视频扩散在空间重叠上更好；通过门控可实现比单一模型更优的性能。

**⚠️ 局限性**

仅限于2D MPM、缺乏真实世界视觉与材质复杂性，模型对自然视频的泛化不明，且生成的物理视频可能被误用。

---

## 266. Multi-Agent Computer Use

**arXiv ID:** 2606.01533 | [PDF](https://arxiv.org/pdf/2606.01533v1)

**作者:** Jing Yu Koh `[一作]` (Carnegie Mellon University), Daniel Fried `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10007 | [OpenAlex ID](https://openalex.org/A5003637850)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通用的多代理计算机使用框架（MACU），通过管理器在任务 DAG 中动态分解、并行执行子任务并实时重新规划

**💡 创新点**

创新点在于：将单一串行 CUAs 转变为多代理系统，利用 DAG 结构对任务进行可扩展、可重规划的并行拆分，并把部分可观测信息通过 DAG 传递，显著提升长期任务性能

**🔧 技术方法**

使用大型语言模型（LLM）做为管理器（如 Claude Opus 4.6）与子代理（如 Qwen3.6-27B）的 ReAct 循环，结合 DAG 操作与文件管理、迭代规划等技术

**📊 数据集**

在四大计算机使用基准上验证：OSWorld、Online-Mind2Web、WebTailBench-v2、Odysseys

**📈 对比分析**

与单代理基线及 pass@k 对比，MACU 在所有基准上成功率提升4.7%-25.5%，在长周期任务上加速1.5×，且在推理预算增大时显示更佳的测试时扩展性

**⚠️ 局限性**

局限包括：对管理器预算和并行度的敏感性、在高度序列化任务中收益有限、对高质量子代理的依赖以及需要进一步研究多代理训练与强化学习策略

---

## 267. Fast Generalization after Interpolation via Critically Damped Momentum Optimization

**arXiv ID:** 2606.01521 | [PDF](https://arxiv.org/pdf/2606.01521v1)

**作者:** Luca Muscarnera `[一作]` (University of Cambridge), Mihaela Van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 23048 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双相优化策略：先快速实现训练集的插值（无权重衰减），随后切换至临界阻尼动量（Critically Damped Momentum，CDM）与权重衰减的后插值阶段，以在插值解空间内寻找低范数解。

**💡 创新点**

创新点在于：① 将插值后阶段视为局部二次近似的阻尼振荡动力学；② 证明临界阻尼动量可在此阶段实现比梯度下降二次加速；③ 给出最优学习率闭式及其计算方法；④ 将上述理论落地为一种可直接使用的双相学习率与动量调度。

**🔧 技术方法**

主要技术包括：局部二次Taylor展开、Hessian基准的阻尼振荡分析、临界阻尼动量公式、最优学习率推导、Aitken修正的幂迭代估计最大Hessian特征值、实验中使用的Adam/AdamW/Muon/SGD比较。

**📊 数据集**

实验数据集：六类合成grokking基准（Gaussian、BinaryAddition、ModularAddition、RF TeacherLinear、SparseParity、Two-SubspaceLinear）；医学与生物信息学真实数据（Leukemia、TCGA）；分子性质预测（QM9）；低样本MNIST；受限语言模型预训练（WikiText2、BabyLMStrictSmall）。

**📈 对比分析**

与Adam、AdamW、Muon、SGD等基准在统一架构、批量、训练预算下比较。结果显示：在所有六个合成基准上均显著降低验证误差/提高准确率；在Leukemia、TCGA、QM9上均取得最高平均性能；在低样本MNIST和受限语言模型中也表现出较优或可比的样本效率与验证损失；整体表现表明双相策略能显著缩短后插值阶段的延迟，提升泛化。

**⚠️ 局限性**

局限性包括：只适用于可实现插值的数据受限场景；假设模型具有足够容量；理论基于局部二次近似，可能在非线性、非凸区域失效；范数被视为复杂度的代理，可能与实际泛化不完全对应；在大规模预训练或高度重参数化模型中，双相策略效果不如预期。

---

## 268. On the Limits of Token Reduction for Efficient Unified Vision Language Training

**arXiv ID:** 2606.01503 | [PDF](https://arxiv.org/pdf/2606.01503v1)

**作者:** Siyi Chen `[一作]` (University of Michigan), Lingjuan Lv `[通讯]` (Sony AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了在统一视听语言模型（VLM）训练中基于token减少的加速方法，并揭示了在统一训练下此类加速导致的协同损失。

**💡 创新点**

发现视觉理解任务在深层存在显著token冗余，而视觉生成任务在深层仍高度依赖图像token；提出任务专属加速策略（HiMix用于理解，HMGen用于生成），并证明仅在单任务下可实现显著效率提升；同时揭示统一训练时采用任务专属token裁剪会破坏跨任务协同，导致性能衰退。

**🔧 技术方法**

采用Transformer自回归框架、注意力分解分析、token裁剪技术（HiMix、HMGen）、参数分离（部分共享投影）以及多任务联合训练。

**📊 数据集**

使用ShareGPT‑4v（视觉理解）和JourneyDB（视觉生成）等公开数据集；在GQA、MME、POPE、SeedBench、MJHQ‑30K等基准上进行评估。

**📈 对比分析**

与统一基线（无token裁剪）和单任务基线比较，单任务加速后FLOPs可下降至约0.24×/0.85×，性能略降；但在统一训练中，HiMix+HMGen将FLOPs降至≈0.55×，但导致理解任务GQA跌至33%/47%以及生成任务MJHQ跌至12.5%/14.5%，明显低于统一基线，表明协同损失。

**⚠️ 局限性**

局限在于任务专属token裁剪无法在统一训练中兼顾两任务，导致梯度冲突与共享表示失效；需要设计能同时兼顾生成与理解的统一加速策略，保持跨任务协同。

---

## 269. TimeSage-MT: A Multi-Turn Benchmark for Evaluating Agentic Time Series Reasoning

**arXiv ID:** 2606.01498 | [PDF](https://arxiv.org/pdf/2606.01498v1)

**作者:** Yaxuan Kong `[一作]` (University of Oxford), Qingsong Wen `[通讯]` (Squirrel Ai Learning)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了TimeSage-MT，一个面向多轮时间序列推理的基准，包含240个任务、2680个对话回合，涵盖8个真实世界领域和4个难度级别；同时构建了可复现的构造流水线和统一评估协议；并对多种LLM与基于技能库的Agent进行了对比评测。

**💡 创新点**

①首次提供多轮、可追踪、可验证的时间序列分析基准；②将任务细化为9项推理能力，能精细诊断Agent的弱点；③构建可复现的流水线和质量控制流程，确保答案可检验；④公开技能库（226项分析技能）与评测仪表盘。

**🔧 技术方法**

使用LLM（Claude Sonnet 4.6、GPT‑5.3‑Codex、GLM‑5.1、Qwen‑3.5‑122B‑A10B、MiniMax‑M2.7、Gemini‑3‑Flash‑Preview）在四种系统设置下进行实验：直接回答、代码启用推理、技能引导代码推理、完整TimeSage管道；评估时采用确定性数值检验、代码执行验证、LLM裁判判定；同时利用可视化仪表盘展示每轮分数。

**📊 数据集**

从65个授权的真实世界时间序列数据集（医疗、能源、制造、气候、金融、电信、交通、零售）中挑选，按频率、维度、季节性等指标构建任务。

**📈 对比分析**

对照实验表明：①代码执行显著提升数值准确率和事实核对（比直接回答高约10–20分）；②技能库进一步提升分析质量与决策质量（部分模型提升5–10分）；③完整TimeSage管道对弱模型有正向提升，但对已具备强Agent性的模型（如GPT‑5.3‑Codex）可能降低分析质量与决策质量。总体而言，所有前沿LLM在深度多轮任务上仍有10–15分的性能下降。

**⚠️ 局限性**

限制：仍缺乏对内存/链式推理的深度改进；数值精度与领域判定仍易出错；TimeSage管道对不同模型的适配性不一致，导致潜在的“结构化过度约束”；基准仅覆盖已授权数据，未涵盖高度保密或专有业务时间序列。

---

## 270. ClawHub Security Signals: When VirusTotal, Static Analysis, and SkillSpector Disagree

**arXiv ID:** 2606.01494 | [PDF](https://arxiv.org/pdf/2606.01494v1)

**作者:** Vincent Koc `[一作]` (OpenClaw Foundation), Nir Paz `[通讯]` (NVIDIA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ClawHub Security Signals数据集，收集了67,453条公开Agent Skill的安全扫描结果，并对三种扫描器（VirusTotal、静态分析、NVIDIA SkillSpector）的标注结果进行系统对比与一致性分析。

**💡 创新点**

首次公开展示多源扫描器在Agent Skill上的结构化不一致性，提出需要分层治理的安全框架，并提供银标准标签作为未来人工审核和模型训练的基准。

**🔧 技术方法**

使用三类扫描技术：VirusTotal（恶意代码与信誉检测）、静态分析（代码与文本模式识别）、SkillSpector（语义能力与安全风险评估）；同时采用Jaccard、Cohen's κ等统计方法评估扫描器间的一致性。

**📊 数据集**

数据集来源为OpenClaw Registry的最新公开Skill版本，包含每条Skill的内容、红点化后的文本、筛选后的捆绑文件以及对应的三种扫描器输出和ClawScan最终判定，完整公开于Hugging Face Hub。

**📈 对比分析**

通过计算扫描器之间的重叠率（最大10.4%）和Kappa系数（0.045–0.082），发现三者几乎没有共识，且各自关注的攻击面不同；该结果表明单一扫描器无法全面评估Skill安全，需多层级协同决策。

**⚠️ 局限性**

局限性包括：标签为LMM生成的银标准，缺乏人工真值；仅覆盖单一时间点的公开Skill，语言主要为英文；缺乏运行时行为分析；扫描器覆盖率虽高但仍有未报错/缺失结果；结构化数据的可复现性受扫描器版本和配置影响。

---

## 271. Splatshot: 3D Face Avatar Generation from a Single Unconstrained Photo

**arXiv ID:** 2606.01493 | [PDF](https://arxiv.org/pdf/2606.01493v1)

**作者:** Hao Liang `[一作]` (Rice University), Guha Balakrishnan `[通讯]` (Rice University)

**通讯引用:** 3428 | [OpenAlex ID](https://openalex.org/A5081710525)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

SplatShot通过在推理时将预训练的3D Gaussian Splatting（3DGS）模型与稳定扩散（Stable Diffusion）生成器联动，利用3D反馈循环在单张无约束人像上生成高质量、身份保持、三维一致的面部3D头像。

**💡 创新点**

核心创新在于：①无需额外训练，采用“训练无关”框架；②在每一步扩散去噪时直接将多视角预测回投到3DGS模型并通过光度误差反向调节噪声，从而在保持扩散模型的图像真实感的同时保证三维几何一致；③使用噪声混合机制提升早期去噪稳定性。

**🔧 技术方法**

技术包括：3D Gaussian Splatting渲染器（gsplat）、Stable Diffusion v1.5 + IP-Adapter-Plus-Face用于身份条件、ControlNet用于姿态控制、DDIM去噪流程、基于光度损失的3DGS微调、噪声混合与3D引导噪声调整。

**📊 数据集**

实验使用CelebA和FFHQ 6,279张随机图片作为输入，300个基线3DGS模型来自NeRSemble数据集；评估时与多种基线（LAM、DreamGaussian、GAGAvatar、FastAvatar、Human-3Diffusion、FaceLift、Arc2Avatar、Intergsedit）比较。

**📈 对比分析**

SplatShot在多项指标上领先或同等：在100个FFHQ身份上获得最高CV‑CSIM（0.832）、最低AKD（0.92×10⁻²）、最低FID（216）、最高CLIP‑IQA（0.633），同时保持CSIM接近最佳（0.698），显示出更强的跨视角一致性、几何精度和视觉质量。

**⚠️ 局限性**

局限包括：推理时间较长（≈3分钟/身份），对后视角尚未验证；三维模型受基线3DGS几何限制，发型主要继承自基准模型；帽子等头饰难以准确处理；在极端姿态或光照下效果可能下降。

---

## 272. Perception First: A Frontier Native-Video Model with Self-Consistency for Implicit Video Question Answering

**arXiv ID:** 2606.01485 | [PDF](https://arxiv.org/pdf/2606.01485v1)

**作者:** Ali Alavi `[一作]` `[通讯]` (Ohio State University), Ali Alavi (Ohio State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种面向隐式视频问答（ImplicitQA/VRR-QA）的零训练、推理时可扩展的系统，利用原生视频输入、强大的 Gemini 3.1 Pro 模型与自一致性采样（self-consistency）实现高达81.18%的平均准确率。

**💡 创新点**

创新点在于：①将原生视频直接输入模型而非传统的帧采样，显著提升感知能力；②在基线模型上加入自一致性投票，提供轻量级的推理时去噪；③系统性验证发现感知瓶颈决定性能，推理层面的增强反而无效，阐明了隐式问答的本质。

**🔧 技术方法**

主要技术包括：多模态大型语言模型（Gemini 3.1 Pro、Qwen3-VL 等）、自一致性推理（k=5 采样投票）、视频预处理（原生视频、Whisper 语音转录）、vLLM 推理服务、数据层统一接口与调度。

**📊 数据集**

使用的数据集为 ImplicitQA/VRR‑QA，包含 1,001 个验证样本和 172 个隐藏测试样本，覆盖 9 类推理任务，其中 69% 属于空间感知类别。

**📈 对比分析**

在验证集上，最强的开源模型（Qwen3‑VL‑32B‑AWQ）达到 58.5% 的平均准确率；而本文提出的 Gemini 3.1 Pro + 自一致性系统在隐藏测试集上实现 81.18% 的平均准确率，略高于此前最佳 80.85% 并接近非专业人类基准 83.0%/85.6%。

**⚠️ 局限性**

局限性包括：① 仅使用 172 条测试样本导致单次提交的噪声较大；② 进一步提升感知性能（分辨率、帧率）和自一致性采样深度在实验中效果有限；③ 由于缺乏训练数据，系统无法通过自监督或少量标注进行改进，当前性能已逼近人类基准。

---

## 273. Beyond Topical Similarity: Contrastive Evidence Retrieval with Interpretable Attention Alignment in RAG

**arXiv ID:** 2606.01482 | [PDF](https://arxiv.org/pdf/2606.01482v1)

**作者:** Francielle Vargas `[一作]` (University of Chile), André Freitas `[通讯]` (Idiap Research Institute)

**通讯引用:** 2495 | [OpenAlex ID](https://openalex.org/A5053978668)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 CERA 框架，将检索从单纯的主题相似性转向证据驱动且可解释的检索。

**💡 创新点**

创新点在于结合基于主观性评估的硬负样本选择与人类标注证据的注意力对齐，注入证据归纳偏置。

**🔧 技术方法**

采用了双编码器对比学习、三元组损失、POS 加权的注意力对齐（KL 损失）以及 Transformer 的 CLS‑to‑token 注意力。

**📊 数据集**

使用了 Evidence Inference 2.0 临床试验报告数据集进行实验。

**📈 对比分析**

与 Contriever 及传统硬负基线相比，CERA 在 Recall@1、Recall@5、MRR 等指标上有显著提升（Recall@1 提升 0.0768，Recall@5 提升 0.1307，MRR 提升 0.10）。

**⚠️ 局限性**

局限包括仅在单一数据集上验证、依赖昂贵的人类注释、注意力解释可能不完全可信、未覆盖多跳推理与长文本场景，以及未处理多模态输入。

---

## 274. Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX

**arXiv ID:** 2606.01478 | [PDF](https://arxiv.org/pdf/2606.01478v1)

**作者:** Martin Schuck `[一作]` (Technical University Of Munich), Angela P. Schoellig `[通讯]` (Technical University Of Munich)

**通讯引用:** 6217 | [OpenAlex ID](https://openalex.org/A5052147335)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一套基于 JAX 的高保真、可微分无人机仿真器 Crazyflow，支持多机编队、低层电机级控制、可微分训练，并在多机、千机编队下实现极高并行性能。

**💡 创新点**

创新点在于将物理与控制器融合为单一可微分计算图，利用 JAX JIT + XLA 在 GPU 上实现百万级环境的并行仿真；同时提供第一原理与抽象数据驱动两种动力学模型，支持快速的系统辨识；实现了毫秒级 RL 训练和实时 MPPI 等。

**🔧 技术方法**

使用技术包括 JAX（XLA JIT、向量化）、CasADi、MJX、NVIDIA Warp、Newton 等；实现了低层控制器的 JAX 模拟，并支持深度 RL、BPTT、NMPC、MPPI 等算法。

**📊 数据集**

主要数据集是 Crazyflie 2.1、CF2.1+、CFB 等无人机飞行数据（几分钟内收集的飞行记录）用于辨识；实测轨迹（圆形、Lissajous）用于 sim‑to‑real 验证。

**📈 对比分析**

与 gym_pybullet_drones、CrazySim、Aerial Gym、DiffAero、DiffPhysDrone 等相比，Crazyflow 在 CPU、GPU 上吞吐量提升 1-2 个数量级，梯度计算速度提升 10 倍，sim‑to‑real 距离缩小 47-61%（甚至 82%）；RL 训练仅需 1-2 秒即可得到可落地策略，实时 MPPI 在 50Hz 下完成 500k 采样。

**⚠️ 局限性**

局限性包括：目前仅支持深度相机深度渲染，缺乏可微分的全景渲染；对非常大批量的物理场景仍受 GPU 内存限制；对多种外部控制器（PX4、ArduPilot 等）支持尚不完整；以及对视觉感知任务的支持仍需要进一步扩展。

---

## 275. OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification

**arXiv ID:** 2606.01476 | [PDF](https://arxiv.org/pdf/2606.01476v1)

**作者:** Yuhang Zhou `[一作]` (Meta AI), Zhuokai Zhao `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无logit的块级OPD框架，通过教师模型的文本rollout与语义相似度来监督学生模型的推理轨迹。

**💡 创新点**

创新点包括：①峰值熵块级调度，在学生生成轨迹中挑选不确定决策点进行验证；②利用多次蒙特卡洛rollout估计教师偏好，并通过Dirichlet‑Multinomial贝叶斯先验进行平滑；③对未审核的token设置KL锚定，防止策略坍塌。

**🔧 技术方法**

技术手段包括蒙特卡洛抽样、语义相似度度量（Edit Distance/ROUGE‑1）、贝叶斯平滑（Dirichlet‑Multinomial）、KL信任区间约束，以及峰值熵选择。

**📊 数据集**

实验使用数学推理数据集DAPO‑Math‑17K、AIME‑2024/25、AMC23、MATH‑500、OlympiadBench；编程任务使用PRIME‑RL、APPS、CodeContests、TACO、Codeforces。

**📈 对比分析**

与离线SFT、GRPO、自监督OPD以及白盒OPD做对比。数学任务上对SFT提升约+45%（绝对），相较白盒OPD提升约+28%；编程任务上亦显著优于SFT和OPD，整体性能明显提高。

**⚠️ 局限性**

局限性在于仍需多次教师查询（虽然低于token级别但对大模型成本较高）；语义相似度对极端不同风格或超长文本可能不稳定；以及对教师rollout质量的依赖较大。

---

## 276. A Minimalist Brain-Computer Musical Interface for Real-Time Emotion-Driven Sonification: System Design and Preliminary Evaluation

**arXiv ID:** 2606.01473 | [PDF](https://arxiv.org/pdf/2606.01473v1)

**作者:** Pablo A. Monroy-D'Croz `[一作]` (Universitat Pompeu Fabra), Julian Cespedes-Guevara `[通讯]` (Universidad Icesi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文设计并实现了一个使用两电极前额 EEG 的极简脑-音乐接口（BCMI），实时将情绪估计映射到音乐参数并在闭环中播放。

**💡 创新点**

创新点在于将 AFAH 的前额 α 异质性与随机规则生成的音乐结合，提供可复现的两通道低成本架构，并公开完整数据与代码。

**🔧 技术方法**

采用的技术包括无线 EEG（BITalino）采集、实时 Python 信号处理与 LSL 同步、基于概率规则的 MIDI 生成、Ableton Live DAW、LabRecorder XDF 记录。

**📊 数据集**

使用了 22 名受试者的 EEG 数据，包含 16 次情绪自诱任务（快乐/悲伤）共 352 试次，且包含自我情绪报告与音乐反馈记录。

**📈 对比分析**

通过线性混合效应模型比较，发现目标情绪对信号的影响仅占 0.4% 方差，整体固定效应 R² 0.037，说明性能不佳。

**⚠️ 局限性**

主要限制包括前额 α 异质性作为可控信号的灵敏度不足、闭环反馈混淆、两通道空间分辨率有限、以及未包含多模态生理数据或个体化校准。

---

## 277. Hierarchical Online Prompt Mutation with Dual-Loop Feedback for Guardrailed Evidence Document Generation: A Production-Evaluation Case Study

**arXiv ID:** 2606.01472 | [PDF](https://arxiv.org/pdf/2606.01472v1)

**作者:** Nataraj Agaram Sundar Tejas Morabia `[一作]` `[通讯]` (eBay Inc.), Nataraj Agaram Sundar Tejas Morabia (eBay Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一套分层在线提示变异框架，用于在市场争议审查中自动生成符合角色、标题、证据等约束的结构化文档。

**💡 创新点**

创新点包括：1）提示族与版本的分层路由与局部提示变异并行；2）双重反馈循环（人工评审+自动评估）实现实时奖励和变异优先级调整；3）严格的可复现性层次，采用架构、提示词类别、Likert量表与安全门槛的公开定义。

**🔧 技术方法**

技术手段包括：强化学习（带分层bandit的路由策略）、局部提示变异与条件化映射、自动化验证器（结构、证据、OCR、角色等），以及配套的统计评估方法（Wilson CI、Bootstrap、McNemar）。

**📊 数据集**

使用的主要数据集为生产抽样的600个匹配案例（包含多种提示变体），辅以770条人工评审抽样、70个OCR评估样本和相应的元数据与评分。

**📈 对比分析**

比较方法是将各变体（静态、手动迭代、Bandit、Mutation、单循环、人机双循环）在同一600案例上进行对比，采用计数胜率、金额加权胜率、Likert评分与问题标记率等指标；结果显示全双循环方案在计数胜率（45.7% vs 34.7%）和金额加权胜率（41.4% vs 22.3%）上均显著提升。

**⚠️ 局限性**

局限性包括：1）评估为匹配案例的离线实验，未进行真实流量随机对照；2）样本偏向SNAD争议，缺乏对其他争议类型的验证；3）原始证据不可公开，复现仅基于抽象化schema和示例；4）评审者间的一致性指标不易复现。

---

## 278. Hierarchical Object Representation for Spatial Robot Perception: Points, Meshes, and Superquadrics

**arXiv ID:** 2606.01545 | [PDF](https://arxiv.org/pdf/2606.01545v1)

**作者:** Ceng Zhang `[一作]` (National University of Singapore), Rajat Talak `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研发了一种层次化对象表示方法，将点云、网格与超四边形联合用于机器人空间感知。

**💡 创新点**

通过多层次抽象实现从细粒度点到高层次几何模型的无缝转换，并提出端到端训练框架，显著提升识别与操作效率。

**🔧 技术方法**

结合深度学习（如PointNet++）、三角网格重建（Poisson Reconstruction）、超四边形拟合算法以及几何约束学习。

**📊 数据集**

在ModelNet40、ShapeNetCore以及自建的机器人抓取数据集上进行训练与评估。

**📈 对比分析**

与传统单一点云/网格/几何模型方法（PointNet、MeshCNN、Superquadric fitting）对比，实验显示在物体检测精度、重建误差和抓取成功率方面提升约5–10%，计算时延保持在实时范围。

**⚠️ 局限性**

对稀疏点云或遮挡严重的场景易失真，超四边形拟合对高度复杂形状的适应性有限，且模型训练对计算资源需求较高。

---

## 279. PathAR: Structure-First Autoregressive Synthesis of Multimodal Pathology Images

**arXiv ID:** 2606.01543 | [PDF](https://arxiv.org/pdf/2606.01543v1)

**作者:** Yuan Zhang `[一作]` (Southeast University), Huazhu Fu `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 28023 | [OpenAlex ID](https://openalex.org/A5010970485)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 PathAR——一种结构优先的自回归生成框架，用于从模态标签直接生成对齐的病理图像与掩码；

**💡 创新点**

创新点包括：①双向向量量化（Dual‑VQ）将结构与外观解耦成两条对齐的离散流；②交错自回归 Transformer（IAR）配合非对称注意力实现结构先行的生成顺序，避免纹理偏差；③在标签‑仅生成模式下同时输出高质量图像与可直接用于下游任务的掩码；

**🔧 技术方法**

采用技术：Dual‑VQ tokenization、交错自回归 Transformer（IAR）+非对称注意力、2D‑RoPE位置编码、CFG指导、下采样率16、双代码本（K=1024）等；

**📊 数据集**

使用数据集：三模态病理子集（细胞学 cytology、荧光 fluorescence、组织学 histology）共约1500张图像；PanNuke（7,901张组织图像）用于细粒度器官标签生成；

**📈 对比分析**

对比 16 类基线（GAN、Diffusion、AR、病理专用模型），在多模态任务中在 FID‑DINOv2、KID、Style Score、Dice、LPIPS 等指标均表现更好或竞争性；在 PanNuke 上亦取得最佳 FID、KID，显示跨模态与细粒度标签的稳健性；

**⚠️ 局限性**

局限性：仅实现了 patch‑级生成，缺乏对 gigapixel 全切片级生成的支持；数据量有限，可能限制更高分辨率或更丰富多模态的扩展；

---

## 280. PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder

**arXiv ID:** 2606.01537 | [PDF](https://arxiv.org/pdf/2606.01537v1)

**作者:** Yancheng Liu `[一作]` (Brown University), Manan Pancholy `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种跨模态蒸馏框架 PaCX-MAE，将 ECG 与实验室指标的生理先验注入胸部 X 光（CXR）视觉编码器，保持推理时仅使用单模态。

**💡 创新点**

创新点包括：① 在 MAE 预训练基础上加入双重对比-回归蒸馏目标，② 采用低秩适配 LoRA 以避免对解剖细节的灾难性遗忘，③ 通过对比学习与回归损失共同对齐视觉与生理特征，实现无辅助模态时的生理感知。

**🔧 技术方法**

使用技术包括 Masked Autoencoder (MAE)、LoRA、InfoNCE 对比学习、回归损失、视觉‑ECG/实验室跨模态对齐、Attention Rollout 以及低样本学习（线性探测）。

**📊 数据集**

主要数据集：MIMIC‑IV Symile‑MIMIC（约10k 组 CXR+ECG+Labs），CheXpert 用于 MAE 预训练，以及 9 个公开评测基准（CheXchoNet、VinDr‑CXR、MedMod、NIH‑14 等）。

**📈 对比分析**

通过与 ImageNet 预训练 ViT‑B/16 与域特定 MAE 进行线性探测和分割任务比较，PaCX 在生理依赖任务上提升 2–6%AUROC/F1，且在 1%/10% 低样本环境下显著优于 MAE；分割性能与 MAE 接近。

**⚠️ 局限性**

局限性：仅在单中心 MIMIC 数据上验证，缺乏多中心多样性；缺乏对局部或时间维度生理关联的细粒度建模。

---

## 281. Rethinking the Role of Positional Encoding: Sliding-Window Transformers without PE Remain Turing Complete

**arXiv ID:** 2606.01532 | [PDF](https://arxiv.org/pdf/2606.01532v1)

**作者:** Qian Li `[一作]` (Shenzhen Research Institute of Big Data), Shang-Hua Teng `[通讯]` (University of Southern California)

**通讯引用:** 12022 | [OpenAlex ID](https://openalex.org/A5102005063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

证明在滑动窗口自回归条件下，无需位置编码的Transformer也能实现图灵完整的计算；

**💡 创新点**

核心创新是将窗口本身的FIFO动态视为足以提供序列信息，构造了只依赖窗口直方图（甚至仅奇偶性）的HIST模型，并证明其可模拟Post机器；

**🔧 技术方法**

利用抽象的自回归HIST模型、奇偶性直方图、以及在Transformer中通过统一注意力和特定嵌入实现窗口统计的技术；

**📊 数据集**

本文为理论研究，未使用任何公开数据集；

**📈 对比分析**

由于本工作仅提供可计算性证明，未进行实验对比，亦未给出具体性能指标；

**⚠️ 局限性**

主要局限在于实现需要适度的自适应窗口更新位、对奇偶性计数的高精度（log S）推断，并且无法恢复窗口内各位置的确切位置信息。

---

## 282. Joint Agent Memory and Exploration Learning via Novelty Signals

**arXiv ID:** 2606.01528 | [PDF](https://arxiv.org/pdf/2606.01528v1)

**作者:** Shizuo Tian `[一作]` (Tsinghua University), Yuanchun Li `[通讯]` (Tsinghua University)

**通讯引用:** 1869 | [OpenAlex ID](https://openalex.org/A5100628298)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 JAMEL 框架，联合训练代理的潜在记忆与探索策略，利用新颖性驱动的交互来监督记忆的压缩与使用；

**💡 创新点**

创新点在于把探索奖励（如代码覆盖率）作为天然的、无监督的监督信号，既训练记忆模块又提升探索深度，形成记忆与探索的互补循环；

**🔧 技术方法**

采用冻结的视觉语言模型压缩观测‑动作序列为记忆向量，线性对齐器将记忆投射到策略嵌入空间；使用代码覆盖率作为二元探索奖励；数据收集通过 86 个 Web 应用生成 24k 条探索轨迹；

**📊 数据集**

主要使用 ScaleWoB 训练集（86 个 Web 应用）和测试集（10 个未见应用），并在 86 个应用上采集 24k 训练样本；

**📈 对比分析**

与 ReAct‑text/vision、MAI‑UI、Mobile‑Agent‑v3.5 等基线对比，JAMEL 在 10 个测试应用上 50 步累计覆盖奖励为 20.7，接近 ReAct‑vision（20.9），明显优于 Open‑Source 基线（8.4/5.9）；同时 token 消耗仅为传统方法的 1/2 左右，显著更高效；

**⚠️ 局限性**

局限性包括：对界面层叠（modal overlay）和极度密集的 UI 仍可能导致记忆压缩不足，导致探索停滞；缺乏强化学习策略，未充分利用更长回合的探索；以及在更大规模或不同领域的泛化尚待验证。

---

## 283. Semi-Supervised Hyperbolic Hierarchical Clustering with Set-Level Structural Priors

**arXiv ID:** 2606.01525 | [PDF](https://arxiv.org/pdf/2606.01525v1)

**作者:** Junjing Zheng `[一作]` (National University of Defense Technology), Weidong Jiang `[通讯]` (National University of Defense Technology)

**通讯引用:** 1361 | [OpenAlex ID](https://openalex.org/A5013345563)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种半监督层次聚类方法，将原本仅在叶节点层面的对偶约束通过构造软子集（set）转化为子树层面的结构先验，在双曲几何空间上进行连续优化；

**💡 创新点**

核心创新在于：①将基于对偶约束和学习到的相似度结构的子集作为软结构单元，实现子树级别的监督；②在Poincaré模型中引入连续的集合内部最近公共祖先（LCA）作为集合表示，并在此基础上构造集合层级三元组损失；③通过混合几何表征学习（欧氏+双曲）与集合级约束学习相结合，提升了约束一致性和层次结构的标签一致性；

**🔧 技术方法**

使用技术包括：双曲自编码器（Poincaré + MLP）、混合欧氏/双曲距离、集合感知对偶约束损失（MLB、ML、CL）、约束感知k近邻图、集合间弱连通相似度、Klein模型求解集合内部LCA、集合层级三元组HypHC损失、梯度下降与BB步长、拉普拉斯正则化以及贪婪树解码；

**📊 数据集**

在11个常用基准数据集上评估：Yale、ORL、Wine、Breast‑Cancer、Isolet1、Australian、OpticalDigits、Spambase、COIL100、USPS、PenDigits；

**📈 对比分析**

与半监督基线（SSSE、Semi‑Multicons、COBRA）及无监督基线（HypHC、HypCSE）比较，采用层次树纯度（Dendrogram Purity）作为指标，本文方法在所有数据集上均获得最高DP值，并在消融实验中显示出对DP和Dasgupta成本的显著提升；

**⚠️ 局限性**

局限性包括：①依赖足够的对偶约束，极稀疏约束下性能下降；②集合构造和弱连通相似度对超参数敏感；③需要O(n²)内存存储相似度，规模化到大数据集存在挑战；④仍以对偶约束为监督来源，缺少更高层级的先验表达。

---

## 284. MotionDreamer: Universal Skeletal Motion Generation for 3D Rigged Shapes

**arXiv ID:** 2606.01518 | [PDF](https://arxiv.org/pdf/2606.01518v1)

**作者:** Ye Tao `[一作]` (City University of Hong Kong), Junhui Hou `[通讯]` (City University of Hong Kong)

**通讯引用:** 10765 | [OpenAlex ID](https://openalex.org/A5031957432)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了MotionDreamer，一个基于扩散模型的无类别骨骼动画生成框架，可根据2D视频为任意拓扑的3D网格生成高质量动画。

**💡 创新点**

① 构建20k多样化3D模型与动画的大规模数据集；② 采用拓扑无关的扩散模型并引入纹理‑语义注入机制；③ 双向视频‑骨骼融合实现视觉与结构的对齐。

**🔧 技术方法**

Diffusion Transformer、Skeletal Attention、DINOv2特征提取；纹理语义注入通过skinning权重投射；bidirectional cross‑attention视频骨骼融合；IK后处理等。

**📊 数据集**

自建约20,000个3D模型与40k对视频‑动画的数据集，来源于Articulation‑XL、Objaverse，并通过AutoRig‑Pro、Qwen等工具完善。

**📈 对比分析**

与Motion 3‑to‑4、ActionMesh、Puppeteer等基线对比，MPJPE仅0.054cm，Chamfer距离0.086，显著优于基线；在跨类别迁移和现实视频中表现稳定。

**⚠️ 局限性**

缺乏多模态（文字/音频）控制与环境交互约束，易受快速动作与复杂背景影响；训练仅基于前视渲染的条件，限制了通用性。

---

## 285. The Main Barrier to AI Adoption in the Public Sector is Lack of Training: How a Structured Method Increased Productivity in Two Brazilian Government Cases Without Incidents

**arXiv ID:** 2606.01517 | [PDF](https://arxiv.org/pdf/2606.01517v1)

**作者:** Vinicius Santana Gomes `[一作]` `[通讯]` (Government of Federal District), Vinicius Santana Gomes (Government of Federal District)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在巴西联邦区公共服务单位中引入了基于“AI House”四层结构的AI应用培训方法，并通过免费AI模型实现生产力提升；

**💡 创新点**

创新点在于将生成式AI技术与法律治理、培训与实践操作融合的四层教学框架，并通过定制AI和数据脱敏协议保障信息安全；

**🔧 技术方法**

使用了生成式语言模型（ChatGPT、Claude、Gemini等）以及Prompt Engineering、定制AI知识库和数据脱敏技术；

**📊 数据集**

使用了官方SEI‑GDF数据集，包括2023‑2024年SES/CONT和2024‑2025年UCI/SEDET的案件处理时间、文档产出等指标；

**📈 对比分析**

通过前后对比（2023‑2024、2024‑2025）展示方法效果：SES/CONT平均处理时间下降18.2%，UCI/SEDET下降50%，同时文档产出和技术报告产出分别提升约22.3%和92%；

**⚠️ 局限性**

局限性包括缺乏对照实验、样本仅限两机构、结果受组织文化和人员变动影响、未对AI生成文本质量进行独立评估、以及对AI推荐落实效果未量化。

---

## 286. Compliance-Scored Best-of-N Guardrail Orchestration for Multimodal Document Generation in Payments Dispute Defense

**arXiv ID:** 2606.01513 | [PDF](https://arxiv.org/pdf/2606.01513v1)

**作者:** Nataraj Agaram Sundar `[一作]` (eBay Inc.), Tejas Morabia `[通讯]` (eBay Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个统一的合规评分最佳‑N 生成管线，用于支付争议防御中的多模态文档生成，集成了 PII 检测、内容审核、架构验证和领域规则，并通过显式合规评分实现早期退出；

**💡 创新点**

创新点包括：① 统一的配置驱动 guardrail 层，实现一次定义到处执行；② 基于并行候选生成的最佳‑N 选择与合规评分；③ 支持文本与图像的多模态输入；④ 同步合规检查与异步监控，保证低延迟且可快速迭代；

**🔧 技术方法**

使用技术有：多模态生成 LLM、并行最佳‑N 采样、OCR/视觉模型提取图像文字、PII 检测与自动脱敏、内容审核模型、JSON/结构化 schema 校验、权重化合规评分、早期退出阈值、异步遥测与配置热更新；

**📊 数据集**

使用的数据集：支付争议案件记录（包含 770 条 evidence 评审样本、70 条 OCR 评审样本），以及生产环境的聚合运营日志；真实数据不公开，仅提供聚合统计；

**📈 对比分析**

比较方法：在生产运营中进行非随机化场景对比，计算 win‑rate、金额加权 win‑rate 的差异，并给出 95% 置信区间与 p 值；结果显示整体 win‑rate 提升 +11 pp，金额加权提升 +19.1 pp，调整后的 INR 场景 +7.5 pp，合规率达到 91%，5 次候选在 20s 内完成；

**⚠️ 局限性**

局限性：① 评估基于聚合非随机化对照，因果推断受限；② 无原始案例级别日志与成本/延迟分布；③ 只给出聚合统计，无法复现细粒度效果；④ 领域特定（支付争议），规则、权重、阈值需重新校准；⑤ 仍需人工复核 OCR 质量、标题漂移等风险。

---

## 287. Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech

**arXiv ID:** 2606.01479 | [PDF](https://arxiv.org/pdf/2606.01479v1)

**作者:** Hongfei Du `[一作]` (William & Mary), Ye Gao `[通讯]` (William & Mary)

**通讯引用:** 702 | [OpenAlex ID](https://openalex.org/A5101503393)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用稀疏自编码器分析LLM语义层隐藏状态，识别情感相关稀疏特征，并通过调节这些特征实现双向情感控制；

**💡 创新点**

首次将稀疏自编码器用于LLM TTS语义层，发现情感信息分布在少数稀疏特征上，并提出特征级干预框架；

**🔧 技术方法**

稀疏自编码器(k-sparse)、Top‑k稀疏机制、特征级干预、对比实验、情感相似度、WER、Spk‑SIM等评估指标；

**📊 数据集**

使用IEMOCAP情感标签的 56,000+ TTS 生成样本，配合 400 文本、20 声音参考，构建分析和评测数据集；

**📈 对比分析**

与基线VALL‑E‑X、Spark‑TTS、EmoVoice、CosyVoice以及全局Steering和随机SAE对比，Emotion‑SIM 最高或第二高，WER 低，语音自然性与人评均优于对手；

**⚠️ 局限性**

训练稀疏自编码器需要大量计算和存储，实验仅在单一backbone上验证，未覆盖多模型与更大规模数据。

---

## 288. An LLM-based Chain-of-Response Counter-Scam System

**arXiv ID:** 2606.01475 | [PDF](https://arxiv.org/pdf/2606.01475v1)

**作者:** Heedou Kim `[一作]` (Korea University), Jaewoo Kang `[通讯]` (Korea University)

**通讯引用:** 16355 | [OpenAlex ID](https://openalex.org/A5076917278)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Counter-Scam，一套统一的多代理LLM框架，实现从检测到预防、紧急处置到调查的端到端诈骗响应。

**💡 创新点**

首次将代理、任务和数据三元组合应用于诈骗响应，构建跨机构协同的完整生命周期模型。

**🔧 技术方法**

利用LLM多代理架构、链式任务（CSRT）、定制化NER和RAG等技术，结合微调sLLMs。

**📊 数据集**

构建了185,300条诈骗案例与38,587条响应知识条目（CSRD），并包含多种原始诈骗对话和法律知识库。

**📈 对比分析**

在多项任务上对比商业LLMs和微调sLLMs，sLLMs平均提高约10%（最高0.55 F1提升），并在九项NLP任务中占优。

**⚠️ 局限性**

对复杂推理、长文本摘要及法律分析仍表现不足，需进一步训练与人类审核。

---

## 289. An Enigma of Artificial Reason: Investigating the Production-Evaluation Gap in Large Reasoning Models

**arXiv ID:** 2606.01462 | [PDF](https://arxiv.org/pdf/2606.01462v1)

**作者:** Mingzhong Sun `[一作]` (National University of Singapore), Tan Zhi-Xuan `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型推理模型（LRMs）在评估推理过程时的缺陷，发现其与人类相比存在显著的生产‑评估差距。

**💡 创新点**

首次通过设计 Valid-Answer-Invalid-Reasoning (VAIR) 数据集，将答案正确与推理错误分离，并揭示了 LRMs 的答案确认偏差。

**🔧 技术方法**

结合链式思考（CoT）行为分析、线性探针和因果补丁技术，对模型内部表征与行为进行可解释性评估。

**📊 数据集**

使用 GSM8K、MATH、Process-Bench 等基准问题为种子，构造 VAIR、VAVR、IAIR 三类对照数据集。

**📈 对比分析**

将 frontier LRMs 与人类参与者在推理生产、VAIR 评估、VAVR/IAIR 对照任务中进行对比；模型在 VAIR 评估上准确率低至 48%（最高 78%），人类仅相差 6%，而在推理生成任务模型表现近乎完美。

**⚠️ 局限性**

研究仅聚焦数学推理，使用的是中小规模公开权重模型；未直接检验训练目标对偏差的影响，且对其他推理领域的普适性未验证。

---

## 290. Genotype-Conditioned Molecular Generation via Evidence-Grounded Multi-Objective Latent Perturbation in Diffusion Models

**arXiv ID:** 2606.01461 | [PDF](https://arxiv.org/pdf/2606.01461v1)

**作者:** Brenda Nogueira `[一作]` (University of Notre Dame), Nuno Moniz `[通讯]` (University of Notre Dame)

**通讯引用:** 61380 | [OpenAlex ID](https://openalex.org/A5068157871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

基于基因型条件的扩散模型进行化合物生成，并在潜在空间上通过多目标优化（药效预测、药物相似性、合成可行性和机制合理性）进行迭代改进。

**💡 创新点**

①将生物学注意力与大语言模型相结合形成多代理评分系统，实现机制合理性评估；②在冻结的生成器上使用可微的在线代理回归（QED/SAS/LLM），实现多目标梯度优化；③将实验室真实细胞系数据与文献证据同步反馈，提升生成化合物的临床相关性。

**🔧 技术方法**

扩散模型（G2D-Diff）、梯度上升潜在空间搜索、在线MLP代理回归、基于Transformer注意力的基因重要性提取、Claude-sonnet-4 LLM多代理推理、RDKit特征计算、AlphaFold/文献检索。

**📊 数据集**

NCI60细胞系基因型和抗药物敏感性数据；15个癌细胞系（EV1/EV2/EV3）作为评估集；FDA批准的靶向药物与对应细胞系的对照组。

**📈 对比分析**

与基于基因表达的生成器（PaccMannRL）、传统G2D-Diff、MolGPT等基线对比；在有效率、QED、SAS、LogP、有效性、独特性、创新度等指标上均优于或接近最优基线，且在预测AUC上取得最低值，证明对细胞系特异性更好。

**⚠️ 局限性**

1）LLM评分依赖AUC预测器的校准，若AUC预测不准则机制评分不足；2）扩散模型的VAE编码对化合物映射存在偏差，影响AUC估计；3）仅在基因级别使用NeST，未充分利用通路层面信息；4）实验室验证仍缺乏。

---

## 291. FLAME: Physics-Guided Neural Operators for Onboard Satellite Methane Detection in Hyperspectral Imagery

**arXiv ID:** 2606.01577 | [PDF](https://arxiv.org/pdf/2606.01577v1)

**作者:** Junhyuk Heo `[一作]` (TelePIX), Woojin Cho `[通讯]` (TelePIX)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种名为 FLAME 的物理引导神经算子，用于卫星超光谱图像中的甲烷 plume 检测。

**💡 创新点**

创新点在于将 Beer–Lambert 吸收模型直接嵌入网络，利用 log‑域匹配滤波的无参数内积作为得分，替代传统匹配滤波的 tile 级背景/协方差，并通过神经算子实现像素级背景和噪声估计。

**🔧 技术方法**

采用 Fourier Neural Operator（FNO）作为算子骨干，配合 1×1 卷积预测 log‑背景和谱权重，辅以逐步衰减的匹配滤波教师辅助损失。

**📊 数据集**

使用 STARCOP 数据集（AVIRIS‑NG 72 带 + RGB）进行训练与评估。

**📈 对比分析**

与经典匹配滤波、两阶段 UNet+MAG1C、端到端 SegFormer/EfficientViT 等方法对比，FLAME 在 F1、IoU 上分别达到 0.608/0.437，像素 FPR 降低三倍，参数仅 0.78M，推理时间 6.2 ms，可在 Jetson Orin/Thor 等嵌入式平台满足功耗与延迟预算。

**⚠️ 局限性**

仍受限于对训练数据的依赖，极弱 plume 的召回低于两阶段管线；未建模噪声协方差的非对角项，且在极端光照或云层条件下性能需进一步验证。

---

## 292. Deformable Wiener Filter for Future Video Coding

**arXiv ID:** 2606.01576 | [PDF](https://arxiv.org/pdf/2606.01576v1)

**作者:** Xuewei Meng `[一作]` (Peking University), Siwei Ma `[通讯]` (Peking University)

**通讯引用:** 15967 | [OpenAlex ID](https://openalex.org/A5039832462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种可变形 Wiener 滤波器（DWF），在 VVC 循环滤波中同时利用局部与非局部参考样本，并通过监督训练获得滤波器系数；

**💡 创新点**

创新点包括：① 在 Wiener 理论下联合局部和非局部样本；② 结合块级与像素级噪声估计的自适应分类；③ 通过自适应融合和异常值约束提升滤波效果；④ 设计了低复杂度的快速匹配与滤波器系数共享方案；

**🔧 技术方法**

使用了 Wiener 赫尔姆霍兹方程求解、块匹配（非局部相似块）与 L2 范数、基于噪声估计的分组分类、滤波器系数共享与约束函数、VVC 参考软件 VTM‑11.0 集成、BD‑rate 与 PSNR 评估、Fast Block Matching 算法；

**📊 数据集**

实验使用 JVET 推荐序列（自然类 A1、A2、B、C、E）以及屏幕内容类 G、M，每个序列 1–2 秒，QP 22、27、32、37；

**📈 对比分析**

与 VTM‑11.0 进行 BD‑rate 对比，并与 ALF 与 NLSF 进行相同配置下的比较。平均 AI、RA、LDB 配置下 BD‑rate 分别降低 1.16%、1.92%、2.67%，屏幕内容视频更显著；编码时间略升 13–15%，解码时间大幅升 1400–2400%；Fast 版降低约 48% 解码时间，性能损失低于 0.2%；

**⚠️ 局限性**

主要限制是计算复杂度高，尤其解码时间成百倍提升；在 AI 配置（无参考帧）下效果不如 ALF；需要手工调参且未针对硬件加速（GPU）做进一步优化；

---

## 293. MomentKV: Closing the Directional Gap in KV Cache Eviction for Long-Context Inference

**arXiv ID:** 2606.01563 | [PDF](https://arxiv.org/pdf/2606.01563v1)

**作者:** Yu Li `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6514 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于矩统计的KV缓存eviction方法，利用被evicted token的计数、键均值、值均值和值-键协方差来指导token淘汰和恢复方向信息。

**💡 创新点**

创新点在于：①在eviction过程中使用“moment residual”衡量token的独特性，从而更精确地保留重要token；②推导出一阶softmax近似，闭式恢复evicted attention输出，并与retained输出混合，显著降低方向误差。

**🔧 技术方法**

使用一阶Taylor展开softmax、Jensen下界估计partition函数、矩统计更新、moment-informed eviction、归一化校正推理等技术。

**📊 数据集**

在LongBench和RULER两个多任务长文本基准上进行实验，采用LLaMA-3.1-8B-Instruct和Qwen3-4B-Instruct两大模型。

**📈 对比分析**

与H2O、SnapKV、PyramidKV、Ada-KV四个主流eviction方法对比，所有缓存预算下均取得最高平均分；在L=128时比Ada-KV提升1.35分，RULER任务提升3.4分，显示显著性能优势。

**⚠️ 局限性**

局限性包括额外存储需求为O(d²)（每头约4.1 MB）和约1.6 ms的延迟，依赖head维度；在极大缓存或更大模型规模下的效果仍待验证，未结合低秩或量化等压缩技术。

---

## 294. E4GEN: Event-level Explainable Extreme-Enhanced Time-series Generation

**arXiv ID:** 2606.01634 | [PDF](https://arxiv.org/pdf/2606.01634v1)

**作者:** Lin Jiang `[一作]` (Florida State University), Guang Wang `[通讯]` (Florida State University)

**通讯引用:** 87281 | [OpenAlex ID](https://openalex.org/A5032583158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 E4GEN，一种可解释的扩散框架，用于极端事件级时间序列生成，并通过控制信号实现对极端事件的精确建模；

**💡 创新点**

三大创新：① E-Activator学习数据自适应的极端事件控制激活步骤，解耦全局与极端事件；② E-Predictor利用自驱动语义预测及无标签训练机制自动推断每个样本的极端事件语义；③ E-Control将语义映射为层级残差偏置，精细引导极端事件生成，同时保持整体时序结构；

**🔧 技术方法**

核心技术包括扩散模型的 x‑prediction backbone、基于 Transformer 的语义预测、数据自适应训练-噪声采样机制、层级控制网络与门控残差注入；

**📊 数据集**

六个真实与一个合成数据集，涵盖气候（温度、降雨）、医疗（心电 ST 段）、能源（电力消耗）与交通（拥堵）等四大领域；

**📈 对比分析**

与 9 组基线（GAN、VAE、Flow、Diffusion、LLM 及两种极端方法）进行 17 项评估，E4GEN 在 79.4% 的指标上位居第一，95.1% 的指标中排前两名，显示出在整体与极端事件生成质量以及下游任务性能上的显著提升；

**⚠️ 局限性**

局限性包括：① 对极端事件阈值的依赖仍需手工设定；② 在极端事件稀缺或标签缺失场景下的训练样本不平衡问题；③ 计算开销相对较高，尤其是多步骤扩散采样与层级控制网络。

---

## 295. Exploiting Semantic and Pixel Representations for Ultra-Low Bitrate Image Compression

**arXiv ID:** 2606.01608 | [PDF](https://arxiv.org/pdf/2606.01608v1)

**作者:** Hao Wei `[一作]` (Xi'an Jiaotong University), Ajmal Mian `[通讯]` (University of Western Australia)

**通讯引用:** 21069 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 SPRDiff 的极低比特率图像压缩框架，结合三编码器与扩散模型，在 0.03 bpp 以下实现更高的像素精度和感知质量。

**💡 创新点**

创新点包括：1）三编码器结构（失真、VAE、语义）融合消除 VAE 表示丢失；2）损失感知重建模块提供双重（像素与语义）条件；3）单步扩散解码利用上述条件实现高质量重建。

**🔧 技术方法**

使用技术：预训练 VAE + 失真编码器 + DINOv2 语义编码器；轻量级特征融合与注意力模块；LoRA 微调的 SD‑Turbo；单步扩散 U‑Net；感知损失（VGG）、对抗损失、空间-通道熵模型。

**📊 数据集**

训练集：LSDIR + Flicker2w；评估集：Kodak、CLIC2020、Tecnick。

**📈 对比分析**

与 Mao、GLC、ResULIC、DiffEIC、RDEIC、StableCodec、OSCAR 等方法在 PSNR、MS‑SSIM、LPIPS、DISTS、FID、KID 等指标下对比。SPRDiff 在 0.02–0.03 bpp 级别上在 PSNR、MS‑SSIM 最高，LPIPS 最低，且 FID/KID 也优于大多数对手，显示出更优的速率-失真-感知平衡。

**⚠️ 局限性**

局限性：在复杂背景下人脸重建会失去身份一致性；模型参数量和推理时间仍高，缺乏轻量化方案；需进一步研究 ROI 或剪枝/蒸馏等技术。

---

## 296. AlphaToken: Decoupling Adaptation and Stability for Path-Aware Response Token Valuation in LLM Post-Training

**arXiv ID:** 2606.01635 | [PDF](https://arxiv.org/pdf/2606.01635v1)

**作者:** Liu Qing `[一作]`, Yi Du `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 11290 | [OpenAlex ID](https://openalex.org/A5017820853)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AlphaToken，一种针对 LLM 事后训练的响应词级价值评估框架，在监督微调（SFT）和偏好优化（PO）过程中通过掩码低价值词条来集中梯度更新。

**💡 创新点**

创新点包括：① 将词值分解为适配（target‑adaptation）和保持（retention stability）两大目标，并在每个目标上进一步分离直接梯度与因果传播路径；② 采用数据自由的 Fisher‑drift 代理替代缺失的保持梯度；③ 将 Ghost Dot‑Product 延伸到词级别，实现四种价值分量的激活空间高效计算；④ 在训练中使用基于价值的硬阈值掩码，实现对信息贡献的精准过滤。

**🔧 技术方法**

主要技术手段：梯度对齐（gradient alignment）与验证目标、Fisher 信息矩阵、Ghost Dot‑Product、Value‑Propagation 近似、直接与因果路径的组合、词级二进掩码（masking）、SFT 与 DPO 训练框架。

**📊 数据集**

使用数据集：SFT 训练在 Magicoder 上，评估在 HumanEval；PO 训练在 UltraFeedback 上，评估在 AlpacaEval 2 与 Arena‑Hard；保持评估采用通用能力数据集 ARC‑C、HellaSwag、MMLU、GSM8K。实验在 Llama‑3.2‑3B、Gemma‑3‑4B 与 Qwen‑3.5‑9B 三种 3B–9B 规模模型上进行。

**📈 对比分析**

与标准微调、LoRA、LESS、Token Cleaning、STM、XTF、ssTOKEN（SFT）以及 DPO、SePO、TI‑DPO、ConfPO（PO）进行对比。AlphaToken 在总体（目标 + 保持）指标上分别提升约 1.5–3.0 点，且在目标任务和保持任务上均获得同步提升，表现优于所有基线。

**⚠️ 局限性**

限制：① 计算开销随 K（层数）、W（因果窗口）和 B_val（验证批量）增加而线性增长，深层或长上下文模型需重新预算；② Fisher‑drift 代理对参考检查点的近似性敏感，若预训练不充分或领域迁移剧烈，代理误差可能增大；③ 当前采用硬阈值掩码，未探索软加权或动态调度等更细粒度的价值调节策略。

---

## 297. What to Test Next: Interpretable Coverage Gap Discovery in Driving VLMs

**arXiv ID:** 2606.01624 | [PDF](https://arxiv.org/pdf/2606.01624v1)

**作者:** Abhishek Aich `[一作]` (NEC Laboratories), Manmohan Chandraker `[通讯]` (NEC Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 SliceNav——一个基于 LLM 的验证框架，利用可解释、可审计的 SliceScorer 对驾驶 VLM 的缺失场景进行优先级排序，并在 ODD 内外支持安全关键的测试工作流。

**💡 创新点**

创新点在于将曝光稀缺性与邻近失败先验相结合的 deterministic scoring 规则 SliceScorer，嵌入 LLM 驱动的查询导向工作流，实现可追溯、可解释的缺失场景推荐。

**🔧 技术方法**

使用了曝光先验统计（Laplace 平滑、维度独立性假设）、文本嵌入相似度与邻居错误传播、乘积权重组合的 deterministic scoring；LLM 用于查询解析、操作组合和可读报告生成。

**📊 数据集**

基于 ISO、SAE 等标准构建的三种驾驶 VLM（WiseAD、DriveMM、Cosmos‑Reason2‑2B）以及从公开驾驶视频/图像合成的 Oracle 数据集。

**📈 对比分析**

与 SliceLine/SliceFinder、随机选取和 kNN 基线对比，SliceScorer 在 Regret@K 上显著降低失效风险（K=5000 时 65–71%），同时保持较高的 ILD，证明在有限预算下能更有效地发现多样且高风险的缺口。

**⚠️ 局限性**

限制包括：对维度独立性假设导致曝光估计粗糙；邻居传播依赖嵌入质量；扩展 ODD 时需手工提供新词汇；对极端稀缺或全新值的风险估计仍不可靠。

---

## 298. Self-Improving Small Object Grounding in LVLMs

**arXiv ID:** 2606.01612 | [PDF](https://arxiv.org/pdf/2606.01612v1)

**作者:** Tianze Yang `[一作]` (University of Georgia), Jin Sun `[通讯]` (University of Georgia)

**通讯引用:** 2440 | [OpenAlex ID](https://openalex.org/A5102942753)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出基于大型视觉语言模型内部注意力模式的候选框选择框架ACS，用以在不进行微调的前提下提升小目标定位性能。

**💡 创新点**

创新点在于：①证明注意力结构本身编码了定位质量；②训练轻量级IoU回归器（ACS‑Learned）直接预测候选框质量；③通过梯度和熵分析将知识蒸馏为无学习参数的规则（ACS‑Free），实现可解释且无需推理时训练。

**🔧 技术方法**

核心技术包括多响应采样、基于注意力图的IoU回归器、梯度归因与熵分析、基于熵的候选框排序。

**📊 数据集**

使用公开数据集MS COCO和Objects365的单目标和多目标小物体样本。

**📈 对比分析**

与贪婪解码及七种采样+投票/熵/平均等基线对比，ACS‑Learned在COCO Acc@0.5提升约4–6%，在Objects365提升约5–16%；ACS‑Free在所有无训练基线中排名第一，提升幅度约3–8%。

**⚠️ 局限性**

局限性包括：需要白盒访问注意力；相对贪婪解码存在额外推理开销；回归器需针对每种LVLM单独训练，迁移性受限。

---

## 299. Paving the Way for Point Cloud Video Representation Learning Using A PDE Model

**arXiv ID:** 2606.01604 | [PDF](https://arxiv.org/pdf/2606.01604v1)

**作者:** Zhuoxu Huang `[一作]` (Aberystwyth University), Josef Kittler `[通讯]` (Surrey University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 MotionPDE，一种基于简化Navier–Stokes PDE的插件式模块，用于正则化点云视频中的时空关联学习。

**💡 创新点**

创新点在于将连续的空间-时间关系抽象为可解的偏微分方程，并用对比学习来引导PDE求解，从而无需显式物理约束即可提升表示质量。

**🔧 技术方法**

采用PDE建模、谱方法（正弦余弦基）实现算子学习、交叉注意力以及InfoNCE对比损失，并将其嵌入现有点云视频骨干网络。

**📊 数据集**

在多种点云视频基准上验证，包括 MSRAction-3D、NTU RGB+D（60/120）、UTD-MHAD、SHREC 2017 以及 HOI4D 等。

**📈 对比分析**

相较于原骨干，MotionPDE 统一提升 1–3% 的动作识别准确率，甚至在某些任务上刷新 state‑of‑the‑art；在自监督预训练时能跨数据集迁移并进一步提升性能。

**⚠️ 局限性**

局限性包括对局部空间结构的依赖（在仅生成全局特征的网络上效果不佳）、对时间池化假设的敏感性，以及目前仅针对点云视频的分类/分割等任务，尚未验证在更复杂的稠密预测任务中的可行性。

---

## 300. TRON: Targeted Rule-Verifiable Online Environments for Visual Reasoning RL

**arXiv ID:** 2606.01599 | [PDF](https://arxiv.org/pdf/2606.01599v1)

**作者:** Tianze Yang `[一作]` (University of Georgia), Jin Sun `[通讯]` (University of Georgia)

**通讯引用:** 2440 | [OpenAlex ID](https://openalex.org/A5102942753)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TRON（Targeted, Rule-verifiable Online Environments）作为可在线生成、可验证的视觉推理训练环境，包含520个生成器-验证器对，支持可调节难度和持续生成。

**💡 创新点**

通过程序化生成、即时验证和动态难度递进的环境，实现无限可扩展、可控、可验证的RL训练信号，并提供完整的环境审计方法。

**🔧 技术方法**

采用生成器-验证器对、离散难度级、DAPO/GRPO风格的RL优化、vLLM张量并行、图像渲染与扰动等技术。

**📊 数据集**

使用TRON自带的520个环境进行训练，外部评测则使用十个多模态推理基准（WeMath, MM-HELIX, MME-Reasoning, SpatialEval, LogicVista, CharXiv, MathVerse-Mini, DynaMath, PuzzleVQA, ChartQA Pro）以及Qwen3-VL-4B、Qwen2.5-VL-7B、MiMo-VL-7B-SFT等模型。

**📈 对比分析**

通过比较RL后训练前后的十个基准准确率，Qwen3-VL-4B从52.61%提升至55.23%，Qwen2.5-VL-7B提升至43.35%，MiMo-VL-7B-SFT提升至66.50%；在能力专家模型中亦显示跨技能迁移优势。

**⚠️ 局限性**

局限性包括：环境为合成视觉，可能与真实数据分布不符；难度级由作者手工设定，缺乏严格单调；多样性度量依赖人工阈值；五个能力桶不是完全互斥，导致标签模糊。

---

## 301. Physics-Informed Modeling and Control of Emergent Behaviors in Robot Swarms

**arXiv ID:** 2606.01597 | [PDF](https://arxiv.org/pdf/2606.01597v1)

**作者:** Zixuan Jin `[一作]` (University of Science and Technology Beijing), Cheng Xu `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 18992 | [OpenAlex ID](https://openalex.org/A5100430968)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 PhySwarm 框架，将机器人群体的多阶段涌现行为建模为宏观层的多相输运扩散反应 (Macro‑ADR) 与微观层的确定性运动模型 (Micro‑EDM) 的耦合，并通过神经物理控制器 (NPC) 学习控制参数实现自适应控制。

**💡 创新点**

创新点在于将多阶段群体涌现视为受物理约束的连续场动力学，采用宏观 ADR 机制描述密度演化，微观模型实现可执行运动；同时将控制器的动作空间限定在可解释的物理参数上，结合 RL‑PINN 目标在训练中强制保持物理一致性。

**🔧 技术方法**

使用的技术包括：连续体 PDE 建模（advection‑diffusion‑reaction）、基于潜能场的输运基底、微观粒子匹配流速、基于状态机的相变转移、神经网络控制器（带记忆）、MAPPO 强化学习、Physics‑Informed Neural Networks（PINN）正则化。

**📊 数据集**

实验数据集为仿真与真实 E‑puck 机器人平台，包含三种任务场景：路径引导搜集、形态可重构导航与角色自适应搜救，每个场景在不同机器人数量、扰动与失败条件下进行多次运行。

**📈 对比分析**

与传统基于有限状态机的规则、单一阶段控制方法和纯黑盒多智能体 RL 方法进行对比；在累计搜集吞吐量、圆形聚集误差、连通度等任务指标上，PhySwarm 显著优于基线和消融版本，证明自适应物理参数调节在多阶段任务中的优势。

**⚠️ 局限性**

局限性包括：需足够多的机器人才能可靠地近似连续密度场，稀疏或碎片化群体时 PDE 残差可靠性下降；ADR 结构及相位划分预先设定，缺乏完全自适应的相变发现；目前仅在二维平面、同质机器人和小规模实验验证，尚需扩展至三维、多样化硬件与更大规模、真实环境长期部署。

---

## 302. TLG: Temporal-Logic Grounding for Video Question Answering via Source-Annotation Reconstruction and Category-Targeted Reasoning

**arXiv ID:** 2606.01591 | [PDF](https://arxiv.org/pdf/2606.01591v1)

**作者:** Ali Alavi `[一作]` `[通讯]` (Ohio State University), Ali Alavi (Ohio State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个三层时间逻辑推理系统TLG，将视频问答任务从感知问题转化为利用公开数据集注解重建时间线并执行符号化逻辑查询；当注解缺失时使用开放式VLM做全局回答；对多选问题中VLM表现最差的类别路由至前沿推理模型。

**💡 创新点**

核心创新点在于①直接利用公开注解重构精确时间线，根除传统VLM对时间定位的“单帧偏差”；②构建按类别精准路由的分层架构，使得仅在VLM弱项时调用高成本前沿模型；③通过系列消融实验明确指出注解质量是提升的关键，而模型规模、基线升级对性能影响有限。

**🔧 技术方法**

使用的技术包括：基于Token‑Jaccard与序列相似度的模糊动作归一化、间隔算术执行的符号化逻辑程序、Qwen2.5‑VL‑32B（AWQ）作为基础VLM、Gemini‑3.1‑Pro作为前沿推理模型、以及信心阈值驱动的三层路由策略。

**📊 数据集**

主要数据集为：CrossTask、Breakfast（细粒度178类注解）、STAR/AGQA（通过Charades+Action Genome构建）以及Charades。

**📈 对比分析**

在TimeLogic测试集（3000问）上与单一大规模VLM基线（46.9%）对比，TLG先通过符号化时间线提升至67.4%，再路由多选弱项至Gemini后达到71.37%（+24.5），仅与排行榜最高74.47%相差3个百分点。

**⚠️ 局限性**

主要限制是注解覆盖不足：约220个STAR/AGQA问题在公开注解中无对应条目，导致符号化层放弃，必须依赖VLM进行推理，进一步提升仍需更完整的时间线注解。

---

## 303. MobEvolve: An Agentic Self-Evolving Heuristic System for Interpretable Human Mobility Generation

**arXiv ID:** 2606.01640 | [PDF](https://arxiv.org/pdf/2606.01640v1)

**作者:** Junlin He `[一作]` (Hong Kong Polytechnic University), Lijun Sun `[通讯]` (McGill University)

**通讯引用:** 6021 | [OpenAlex ID](https://openalex.org/A5058941074)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于LLM编码代理自我演化的可解释人类移动生成框架MobEvolve，能够从个体特征合成现实可行的日常行程链

**💡 创新点**

创新点在于将行为驱动的可解释启发式模型与LLM编码代理结合，利用验证集误差自我诊断并迭代更新内部规则，兼顾行为合理性、分布一致性与推理效率

**🔧 技术方法**

采用LLM（Codex+GPT‑5.5）代理进行逻辑诊断与规则演化，构建可解释的四层生成流程（出行参与、检索、适配、行为细化）并引入演化记忆

**📊 数据集**

在新加坡（活动链）和蒙特利尔（活动‑地点联合链）两大真实城市日常行程数据集上进行评估

**📈 对比分析**

与传统多项式逻辑、Markov、Copula、深度生成模型（Deep Behavior Choice、AIRL、TVAE、CopulaGAN、CTGAN）以及LLMob（少样本/零样本）对比，MobEvolve在个体准确性、分布校准和行为合理性方面整体得分最低，性能提升约34%（新加坡）和11%（蒙特利尔），同时推理速度仅7.65 ms/样本

**⚠️ 局限性**

依赖于参考数据的完整性与代表性，若样本缺失特定人群或稀有路径，演化结果可能继承偏差；LLM代理提议的规则可能过度特化，需要严格验证与演化记忆审计

---

## 304. CanonCGT: Reference-Based Color Grading via Canonical Pivot Representation

**arXiv ID:** 2606.01638 | [PDF](https://arxiv.org/pdf/2606.01638v1)

**作者:** Jinwon Ko `[一作]` (Korea University), Chang-Su Kim `[通讯]` (Korea University)

**通讯引用:** 9633 | [OpenAlex ID](https://openalex.org/A5063748248)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于“canonical pivot”的两阶段参考色彩分级框架CanonCGT，能够先消除输入图像的色调偏差，再对齐参考图像的色调风格；

**💡 创新点**

创新点在于引入style‑neutral中间表示（canonical pivot）和双阶段训练（DP‑CGT），将偏差消除与风格映射解耦，显著提升色调稳定性和视觉真实性；

**🔧 技术方法**

使用了3D LUT生成器、FiLM调制的自注意力网络、对比学习的分级提取器以及自监督的局部重建损失；

**📊 数据集**

在监督阶段使用MIT‑Adobe FiveK（含5k图像与56种Lightroom预设）训练，在自监督阶段使用Flickr2K、LSDIR、PPR10K、DIV2K、Food‑101、GLD‑v2共108k+图像；

**📈 对比分析**

与PhotoNAS、PhotoWCT^2、Neural Preset、CAP‑VST、Deep Preset等方法在PSNR、SSIM、ΔE_ab、LPIPS、SSIM_ED、H‑Corr/H‑Chi等指标上统一获得最佳或接近最佳结果，用户研究也显示其在色调一致性和感知完整性上排名最高；

**⚠️ 局限性**

在极端参考风格（如强烈色彩或全黑白）下，CanonCGT倾向于保留原图的整体色彩平衡，难以完全实现参考的全局调色效果。

---

## 305. Goal2Pixel: Grounding Goals to Pixels for Vision-Language Navigation

**arXiv ID:** 2606.01621 | [PDF](https://arxiv.org/pdf/2606.01621v1)

**作者:** Muyi Bao `[一作]` (Carnegie Mellon University), Wenshan Wang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 800 | [OpenAlex ID](https://openalex.org/A5101634794)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种纯像素预测框架Goal2Pixel，将视觉-语言导航任务从低级动作预测改为图像平面上的可导航像素定位，从而实现高层VLM推理与低层机器人动作的解耦。

**💡 创新点**

创新点包括：①将所有导航决策统一为单一像素坐标预测（通过在图像边缘添加辅助指令区域实现转向与停止）；②设计了可见性感知关键帧记忆ViKeyMem，按视角变化动态选择关键帧并叠加轨迹，显著压缩历史图像；③在预训练VLM上加入视觉语义嵌入和坐标感知辅助损失，使模型更好地对齐像素几何与语言指令。

**🔧 技术方法**

核心技术包括：预训练的VLM（InternVL3等）用于语义理解与像素预测；相机几何把像素坐标映射到3D工作点；轻量化的视觉语义嵌入与坐标辅助损失；可见性感知关键帧选择与轨迹覆盖的历史记忆模块；以及基于像素回投的局部规划器。

**📊 数据集**

使用了连续环境下的室内导航基准：Room-to-Room Continuous Environment（R2R-CE）和Room-across-Room Continuous Environment（RxR-CE）的训练与验证集。

**📈 对比分析**

与不使用外部数据的SOTA方法相比，Goal2Pixel在R2R-CE上实现了54.1% SR、52.5% SPL，平均VLM调用量仅7.75次（比传统的46.62次低6倍）；在RxR-CE上也取得了相似的性能提升，显示了在路径效率与轨迹对齐度上的显著优势。

**⚠️ 局限性**

主要局限性是：①需要可靠的深度信息来将像素回投为3D点，若深度不可用则难以执行；②目前仅在室内数据上训练与评估，缺乏对户外环境的验证与适应。

---

## 306. ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL

**arXiv ID:** 2606.01619 | [PDF](https://arxiv.org/pdf/2606.01619v1)

**作者:** Zelin He `[一作]` (Pennsylvania State University), Matthew Reimherr `[通讯]` (Pennsylvania State University)

**通讯引用:** 2041 | [OpenAlex ID](https://openalex.org/A5037971028)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将技能创建与策略学习同步进行的RL框架，能够在训练过程中自动诊断、生成并评估模块化技能；

**💡 创新点**

创新点在于将Anthropic的Skill Creator循环嵌入GRPO训练循环，利用组内采样与Thompson Sampling实现技能版本的实时评估和选择，消除了传统方法中技能与策略脱节导致的冲突；

**🔧 技术方法**

主要技术包括基于GRPO的组相对策略优化、断言驱动的失败诊断与技能修订、带自适应衰减的Thompson Sampling用于技能版本分配，以及条件触发的技能加载；

**📊 数据集**

在五大类任务上进行实验，包含嵌入式推理（ALFWorld）、代理搜索（多项事实问答集）、科学发现、代码生成（InterCode-SQL）和网页购物（WANDS）等数据集；

**📈 对比分析**

与四类基线（基础ReAct、内存/技能进化、仅技能、RL改进方法）以及多模型规模进行比较，实验显示该方法在所有基准上均取得显著提升，尤其在未见任务和难度更高的任务上优势最为明显；

**⚠️ 局限性**

局限性包括对创建技能的LLM质量依赖度、在极端高维环境下的样本效率及推理时延可能增加，以及在多技能交互场景下的可解释性挑战。

---

## 307. Turing Patterns for Multimedia: Reaction-Diffusion Multi-Modal Fusion for Language-Guided Video Moment Retrieval

**arXiv ID:** 2606.01615 | [PDF](https://arxiv.org/pdf/2606.01615v1)

**作者:** Xiang Fang `[一作]` (Nanyang Technological University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 62129 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

研发了一种基于反应-扩散（Reaction-Diffusion）模型的视频‑文本融合框架 RDMF，解决了视频语言任务中动态非线性交互难题。

**💡 创新点**

创新点在于将 Turing 的反应‑扩散系统引入多模态融合，构建 Gray‑Scott RD 模块，使模型能自组织生成与文本匹配的时序模式，实现动态自适应融合。

**🔧 技术方法**

使用了 VideoSwin 与 BERT 预训练编码器、离散扩散算子、Gray‑Scott 反应‑扩散迭代、DETR‑style 时段检索头、IoU 预测头，并进行稳定性与 Turing 不稳定性分析。

**📊 数据集**

在 QVHighlights、Charades-STA、ActivityNet Captions 等标准视频‑语言数据集上进行训练与评估。

**📈 对比分析**

与 Moment-DETR、QD-DETR、CG-DETR、InternVideo 等 SOTA 基线比较，RDMF 在 R@1@0.5、R@1@0.7、mIoU、AP、Hit@1 等指标上均显著提升，达到 70.5% / 53.8% / 65.7% / 71.2% / 86.9% 等最优成绩。

**⚠️ 局限性**

主要局限是计算量较大，需要多步反应‑扩散迭代，影响实时性；对超参数（扩散系数、反应参数）敏感；对极长视频的细粒度捕捉仍有限。

---

## 308. Revisiting Ripple Effects in Knowledge Editing through Pressure-Aware Joint Neighborhood Optimization

**arXiv ID:** 2606.01610 | [PDF](https://arxiv.org/pdf/2606.01610v1)

**作者:** Haoben Huang `[一作]` (University of Chinese Academy of Sciences), Di Gao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 8987 | [OpenAlex ID](https://openalex.org/A5018706886)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型知识编辑中因单一事实修改引起的局部知识传播与干扰问题，提出 Joint Neighborhood Optimization (JNO) 框架。

**💡 创新点**

创新点在于将正向传播和负向扰动视为耦合压力，通过 Pressure‑Aware Coordination (PAC) 共同优化目标表示，并引入语义预执行门控防止不安全更新。

**🔧 技术方法**

采用邻域构造、PAC 联合优化、语义门控以及保留邻域对齐权重等技术。

**📊 数据集**

使用 RippleEdits、CounterFact 和 ZsRE 三大基准数据集进行评测。

**📈 对比分析**

与 ROME、MEMIT、GLAME 等传统编辑方法比较，JNO 在传播与保持指标上提升约 7% 以上，并保持 95% 以上的单步可靠率。

**⚠️ 局限性**

局限性包括邻域构造质量对 PAC 的影响、语义门控导致的覆盖率下降以及额外的计算开销。

---

## 309. EIVE: End-to-End Instance-Specific Visual Explanations for Detection Transformers

**arXiv ID:** 2606.01601 | [PDF](https://arxiv.org/pdf/2606.01601v1)

**作者:** Jianlin Xiang `[一作]` (Shenzhen University), Linhui Dai `[通讯]` (Shenzhen University)

**通讯引用:** 1093 | [OpenAlex ID](https://openalex.org/A5072786229)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种端到端实例级视觉解释框架 EIVE，使检测 Transformer 能在前向传播中直接生成实例级显著图。

**💡 创新点**

创新点在于将交叉注意力建模为实例级特征归因通道，并引入跨层混合共识融合（CLHCF）和注意力感知联合训练（AAJTS）实现高效、稳定且可训练的解释。

**🔧 技术方法**

采用 Detection Transformer 的交叉注意力、跨层融合、注意力权重约束等 Transformer 与可解释性技术。

**📊 数据集**

在 COCO、ExDark、Cityscapes 三大公开数据集上进行评估。

**📈 对比分析**

与 GradCAM、D‑RISE、VPS 等后置方法对比，EIVE 在插入/删除、指向游戏、能量 PG、紧凑度等指标上均表现更好，同时推理时间仅为几毫秒，显著提升效率。

**⚠️ 局限性**

局限性包括对 Transformer 检测器的依赖、对非 Transformer 模型适用性有限，以及在极低光或极大遮挡场景下解释可能仍受限。

---

## 310. RoboTrustBench: Benchmarking the Trustworthiness of Video World Models for Robotic Manipulation

**arXiv ID:** 2606.01600 | [PDF](https://arxiv.org/pdf/2606.01600v1)

**作者:** Huiqiong Li `[一作]` (Singapore Management University), Bin Zhu `[通讯]` (Singapore Management University)

**通讯引用:** 2766 | [OpenAlex ID](https://openalex.org/A5101945545)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RoboTrustBench基准，评估机器人视频世界模型在正常、约束敏感、反事实与对抗场景下的可信度。

**💡 创新点**

创新点在于四种场景与13细粒度评价维度的设计，强调安全、物理可行性与模型对不确定或危险指令的鲁棒性。

**🔧 技术方法**

结合人工评估与多模态大语言模型（如GPT‑5.4、Qwen3‑VL）自动评测，采用证据引用与文本‑视觉对齐。

**📊 数据集**

基于DROID真实机器人操作数据集抽取1207条指令‑图像对，人工验证并改造为四种场景。

**📈 对比分析**

通过人评与MLLM评测七种模型，Kling‑v2.6在多维度上表现最好，但在约束、假设和对抗场景下性能明显下降，显示现有模型可信度不足。

**⚠️ 局限性**

局限性包括人评覆盖有限、离线评估不涉及闭环控制、对抗/反事实指令未在真实机器人上测试。

---

## 311. Uncertainty-Calibrated Diffusion for Reliable 3D Molecular Graph Generation

**arXiv ID:** 2606.01595 | [PDF](https://arxiv.org/pdf/2606.01595v1)

**作者:** Fang Wan `[一作]` (State University of New York at Stony Brook), Yi Liu `[通讯]` (State University of New York at Stony Brook)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在3D分子图生成中扩散模型逆推过程的认知不确定性导致的方差膨胀问题，并提出了Uncertainty‑Calibrated Diffusion（UCD）方法在逆推时校准噪声以提升生成质量。

**💡 创新点**

创新点在于：①首次系统阐释认知不确定性与扩散过程中的随机性交互导致的方差膨胀机制；②基于贝叶斯 dropout 的蒙特卡罗估计，给出逆推噪声校准公式；③在理论上通过路径空间 KL 与误差传播分析证明校准可缓解方差膨胀。

**🔧 技术方法**

技术手段包括：贝叶斯 dropout 近似、扩散模型（DDPM）逆推、路径空间 KL、误差传播分析、实验评估、以及可插拔的采样算法。

**📊 数据集**

实验使用公开数据集 QM9 与 GEOM‑Drugs，涵盖从小分子到中等尺寸分子。

**📈 对比分析**

通过与多种基线模型（EDM、GeoLDM、RADM 等）在 Atom Stability、Molecule Stability、Validity、Validity×Uniqueness 等指标上对比，UCD 在大多数指标上均超越原始模型，刷新 3D 分子扩散的 SOTA。

**⚠️ 局限性**

局限性包括：①仅在小分子生成任务验证，蛋白质或材料等大尺度结构的适用性尚待研究；②需要多次前向推理（Monte Carlo dropout），导致推理时延增加；③对 dropout 层位置与 M（前向次数）的敏感性，需经验调参。

---

## 312. Attention-guided Fine-tuning of Multimodal Large Language Models Improves Chain-of-Thought Reasoning

**arXiv ID:** 2606.01558 | [PDF](https://arxiv.org/pdf/2606.01558v1)

**作者:** Sanchit Sinha `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 12152 | [OpenAlex ID](https://openalex.org/A5013588572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究多模态大型语言模型在链式推理（CoT）中的效果，诊断其失败模式，并提出 Attentive-CoT 通过注意力引导的微调提升 CoT 推理质量。

**💡 创新点**

创新点在于将早期答案承诺和视觉注意力两项约束嵌入到 CoT‑SFT 目标中，形成一种无架构改动、可插拔的监督式微调方案，显著提升视觉推理的准确性与可解释性。

**🔧 技术方法**

使用 CoT‑SFT、注意力统计、视觉依赖度量、早期承诺评估等技术；实现了基于视觉注意力的惩罚与奖励机制。

**📊 数据集**

评测数据集包括 ChartQA、CLEVR、CV‑Bench，以及 OOD 数据集 EvoCharts、Super‑CLEVR。

**📈 对比分析**

与 SFT、LLaVA‑CoT、LLaVA‑R 等基线对比，Att‑CoT 在准确率上提升 2–9%，显著降低早期承诺（EC‑AUC）、提升视觉依赖（Vis‑Dep）并缩小 Direct‑CoT 差距。

**⚠️ 局限性**

仅在自回归 MLLM 上验证，未探索 RL 后训练；实验仅使用开源模型，闭源系统的表现可能不同；对非自回归或扩散类模型的适用性未作研究。

---

## 313. Pave-GRPO: Beyond Instantaneous Guidance through Principled Average Velocity Decomposition

**arXiv ID:** 2606.01636 | [PDF](https://arxiv.org/pdf/2606.01636v1)

**作者:** Pengyang Ling `[一作]` (University of Science and Technology of China), Yuhang Zang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将少步纯SDE rollouts分解为更细粒度的子轨迹，并结合Group Relative Policy Optimization（GRPO）进行后训练偏好对齐，显著提升流模型在全局布局与细节表现的质量；

**💡 创新点**

提出Principled average velocity decomposition方法，实现零成本时域扩展与密集时间监督，从而在不增加采样成本的前提下，将奖励信号传播到更多中间 denoising 步骤；

**🔧 技术方法**

采用GRPO框架、纯SDE采样、ODE–SDE混合轨迹、可解析的高斯概率路径、剪切似然损失和多路径分解技术；

**📊 数据集**

使用HPD数据集（约10万多条提示）进行训练，UniGenBench（600条精心设计提示）进行评估，并以Flux.1‑dev 12B模型为基线；

**📈 对比分析**

在Flux.1‑dev上与Flow‑GRPO、DanceGRPO、Flow‑CPS等方法比较，使用HPS‑v2/3、CLIP、ImageReward、UnifiedReward、Pick‑Score等多种奖励模型，以及UniGenBench 10维度评分，Pave‑GRPO在整体得分、细节、布局、文本‑图像对齐等指标上均超过竞争方法，刷新SOTA；

**⚠️ 局限性**

局限性包括：对奖励模型的依赖，分解因子增大时计算开销上升；性能提升在分解细化到一定程度后趋于递减；并未彻底解决高步骤训练的整体成本问题。

---

## 314. TimeLogic Challenge @ CVPR 2026: Strong MLLMs Meet Evidence-Seeking Agents for Temporal-Logic Video Question Answering

**arXiv ID:** 2606.01631 | [PDF](https://arxiv.org/pdf/2606.01631v1)

**作者:** Zhaoyang Xu `[一作]` (Harbin Institute of Technology), Jianlong Wu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 3922 | [OpenAlex ID](https://openalex.org/A5100654190)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于主动证据寻求的时序逻辑视频问答(agent)，通过多粒度时间采样与多步骤Think–Act–Observe循环，动态定位关键时刻并进行时间关系推理。

**💡 创新点**

创新点在于将视频问答转化为主动探索问题，利用多粒度时间采样工具、时间戳标注、问答类型自适应策略以及语音上下文辅助，实现无监督、无训练的高效时序推理。

**🔧 技术方法**

核心技术包括：多粒度采样工具（全局概览、结构化扫描、段级粗采样、密集聚焦）、时间戳交错的帧输出、ReAct式思考-行动-观察循环、轻量级问题分类器、类别特定提示与预算控制，以及与Gemini 3.1 Pro的无训练交互。

**📊 数据集**

使用了TimeLogic基准数据集，包含来自CrossTask、STAR、AGQA、Breakfast四个来源的1850段视频共3000道问题（多选与布尔两种答案格式）。

**📈 对比分析**

与基准官方评测进行比较，使用Gemini 3.1 Pro作为后端，最终在TimeLogic测试集上获得77.13%的AvgAcc，未进行任何任务特定训练。

**⚠️ 局限性**

局限性主要在于：依赖外部大型VLM的推理能力，对长视频仍需手动设定预算与步骤；对语音信息的利用仅限于CrossTask，无法覆盖无声视频；模型推理过程中对时间戳解析与工具调用的鲁棒性仍有提升空间。

---

## 315. Benchmarking LLM-as-a-Judge for Long-Form Output Evaluation

**arXiv ID:** 2606.01629 | [PDF](https://arxiv.org/pdf/2606.01629v1)

**作者:** Junjie Chen `[一作]` (Tsinghua University), Qinyao Ai `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LongJudgeBench，一个用于评估大型语言模型在长文本输出评估中的可靠性的新基准；

**💡 创新点**

聚焦长文本评估需求，覆盖五个真实场景、六个数据集，提供平均长度超过9k的候选输出和专家注释，填补现有评估多聚焦短文本的空白；

**🔧 技术方法**

使用多种 LLM 判别器（Qwen、GPT、DeepSeek 等）在不同判别设置（Vanilla、Rubric、Reference、Ref.+Rubric）下进行评估；

**📊 数据集**

利用六个数据集（DeepResearch Bench、RealDeepResearch、SurGE、WP-Bench、VerifyBench、MA）收集长文本评估实例；

**📈 对比分析**

通过与专家标注的准确率、Spearman/Kendall 等指标对比，发现当前 LLM 判别器平均准确率仅 56% 左右，最高 67%，显示长文本评估仍极具挑战；

**⚠️ 局限性**

局限在于样本多样性仍不足、仅评估了部分判别设置、缺乏检索增强、跨域推理等更先进方法，且结果受输入长度、领域知识等因素影响明显。

---

## 316. TechGraphRAG: An Agentic Graph-Augmented RAG Framework for Technical Literature Reasoning

**arXiv ID:** 2606.01613 | [PDF](https://arxiv.org/pdf/2606.01613v1)

**作者:** Kanwar Bharat Singh `[一作]` `[通讯]` (Global Tire Intelligence and Solutions), Kanwar Bharat Singh (Global Tire Intelligence and Solutions)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一个面向智能轮胎与车辆动力学的13步代理检索增强生成（RAG）框架，用于技术推理与文献导航。

**💡 创新点**

核心创新在于：①基于100分多维评分的证据充分性判断和自适应检索；②多源检索（内部向量+关键词、Crossref、OpenAlex、Semantic Scholar）与Neo4j知识图的协同；③全流程代理控制与自动重检与生成自纠；④将LLM与规则混合评估以实现可解释性。

**🔧 技术方法**

使用的技术包括：FAISS+BM25+RRF混合检索、交叉编码器reranker、GPT‑4o‑mini（查询重写、评分、实体抽取、校验）、OpenAI API、Crossref/OpenAlex/Semantic Scholar API、Neo4j图数据库、PyMuPDF、MiniLM‑L6‑v2嵌入。

**📊 数据集**

数据集为约2100篇专门针对智能轮胎、车辆动力学与控制的学术论文，约24000个文本块，并通过OpenAlex补充外部文献。

**📈 对比分析**

通过6个典型查询的路由评估和10个查询的检索消融实验验证：检索精度从单一BM25（P@5≈0.52）提升到完整管线（P@5≈0.78），成本平均约0.0045美元/问，平均时延≈16秒，且所有示例答案均通过自动质量检测。

**⚠️ 局限性**

局限包括：依赖外部云API与网络稳定性、仅处理文本缺失图表与公式、评估样本有限、LLM主导的评估可能存在偏差、以及专有语料库限制可复现性。

---

## 317. Learning Chaotic Dynamics through Second-Order Geometric Supervision

**arXiv ID:** 2606.01596 | [PDF](https://arxiv.org/pdf/2606.01596v1)

**作者:** Shinhoo Kang `[一作]` (Korea University), Tan Bui-Thanh `[通讯]` (University of Texas at Austin)

**通讯引用:** 3139 | [OpenAlex ID](https://openalex.org/A5008274019)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用随机扰动下的雅可比匹配（随机雅可比匹配）来隐式约束学习的动力学模型的二阶几何结构，从而实现对混沌动力系统（Lorenz 63 和耦合 Lorenz 96）的高质量长期统计学习。

**💡 创新点**

创新点在于：①证明随机雅可比匹配的期望损失包含了二阶赫氏张量的不匹配项，从而无需显式构造 O(d³) 记忆量的 Hessian；②将该方法与传统的轨迹匹配、雅可比匹配、以及显式 Hessian 匹配进行系统对比，展示二阶监督对保持混沌吸引子、维持 Lyapunov 指数、抑制奇异吸引子等方面的显著优势。

**🔧 技术方法**

核心技术包括：神经常微分方程（NODE）、基于自动微分的雅可比/赫氏计算、随机扰动的模型约束（MC）框架、Taylor 展开分析以推导隐式二阶监督损失，以及多尺度耦合 Lorenz 96 的局部可变形 MLP 结构。

**📊 数据集**

使用的实验数据集为：①Lorenz 63 的低维轨迹（时间区间 [0,500]，步长 0.01，32 条初始轨迹）；②耦合 Lorenz 96 的高维多尺度轨迹（时间区间 [0,500]，步长 0.005，单个长轨迹并作 1 次验证），对不同外部驱动力 F（10、20）进行实验。

**📈 对比分析**

与五种监督策略（仅轨迹、MC、显式雅可比、显式 Hessian、MC+雅可比）对比。结果显示：二阶监督方法（显式 Hessian 和随机雅可比）在维持吸引子结构、Wasserstein‑1 距离、Lyapunov 指数误差、赫氏范数接近原始值以及抑制长期奇异行为方面均优于仅有一阶监督的方式；在 Lorenz 96 的高维实验中，随机雅可比匹配以第一阶成本获得与显式 Hessian 相当的性能。

**⚠️ 局限性**

局限性包括：①实验仅在结构化的“通用微分方程”设置下进行，未对完整高维系统或偏微分方程进行验证；②随机雅可比匹配对扰动尺度 σ 与损失权重的选择敏感，缺乏系统的调参准则；③在极端非分布或极大时间步长下的鲁棒性尚待进一步研究。

---

## 318. Distributed Algorithm for Robust Wardrop Equilibrium in Uncertain Aggregative Congestion Games

**arXiv ID:** 2606.01594 | [PDF](https://arxiv.org/pdf/2606.01594v1)

**作者:** Huan Peng `[一作]`, Karl Henrik Johansson `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种针对不确定聚合拥塞博弈的鲁棒 Wardrop 均衡（RGWE）分布式算法，用于电动汽车充电控制；

**💡 创新点**

创新点在于：①利用鲁棒优化把不确定的耦合约束转化为可解的确定性增强问题；②结合投影原始-对偶法与动态跟踪技术，构造了双时间尺度的分布式迭代；③通过奇异扰动和 LaSalle 不变原理证明收敛，并给出 RGWE 与 RGNE 的误差上界；

**🔧 技术方法**

采用的技术包括：鲁棒优化理论、投影原始-对偶算法、动态跟踪（Consensus+聚合）、奇异扰动理论、LaSalle 不变原理、凸分析与变分不等式；

**📊 数据集**

实验数据基于电动汽车充电模型，采用 5 台车辆的时间序列（24 时段）和不确定参数盒集（D_i=[I_T,-I_T]ᵀ、d_i 取多种取值）进行仿真；

**📈 对比分析**

与文献[2]的基线算法相比，该算法在收敛速度上与基线相当（早期迭代），但最终能保证约束满足、保持双侧非负性；在不同不确定性水平下，鲁棒算法表现出更优的安全与效率折衷，误差上界随人口规模 N 递减，证明了 RGWE 与 RGNE 的紧密关系；

**⚠️ 局限性**

局限性在于：①仅针对线性耦合约束，非线性或更一般多面体约束需进一步研究；②理论上缺乏线性收敛速率保证；③实验仅在小规模（5 台车辆）下验证，未在极大规模（如 N=10^7）场景下实证；

---

## 319. FedMTFI: Feature Importance Based Optimized Multi Teacher Knowledge Distillation in Heterogeneous Federated Learning Environment

**arXiv ID:** 2606.01607 | [PDF](https://arxiv.org/pdf/2606.01607v1)

**作者:** Nazmus Shakib Shadin `[一作]` (Kennesaw State University), Bobin Deng `[通讯]` (Kennesaw State University)

**通讯引用:** 140 | [OpenAlex ID](https://openalex.org/A5100940541)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出FedMTFI框架，利用基于硬件聚类的FedAvg、跨模型多教师知识蒸馏及SHAP特征重要性加权，构建面向异构设备的联邦学习系统。

**💡 创新点**

创新点在于将多教师知识蒸馏与Shapley值特征重要性结合，以在异构环境下实现更高精度和可解释性的全局模型，并通过聚类实现对设备异构的天然适配。

**🔧 技术方法**

采用联邦学习(FedAvg)、多教师知识蒸馏(MTKD)、Shapley值（SHAP）特征重要性估计以及梯度近似的SHAP计算方法。

**📊 数据集**

使用MNIST作为私有非IID训练数据，FMNIST与CIFAR-10作为公共评估数据集。

**📈 对比分析**

与FedAvg、FedProx、FedKDShap及集中式学习对比，FedMTFI在CIFAR-10上实现64.48%准确率、FMNIST上87.28%准确率，明显优于传统联邦学习基线。

**⚠️ 局限性**

主要限制是服务器端计算SHAP值导致的额外计算开销，以及仅在图像分类任务和静态非IID分布上验证，未涵盖更复杂或动态联邦环境。

---

## 320. Identifying High-Confidence Social Biases in LLMs for Trustworthy Conversational Tutoring Agents

**arXiv ID:** 2606.01584 | [PDF](https://arxiv.org/pdf/2606.01584v1)

**作者:** Aitor Arronte Alvarez `[一作]` (University of Hawaii at Manoa), Naiyi Xie Fincham `[通讯]` (University of Hawaii at Manoa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究评估了三种最新 LLM 在对话式辅导环境中识别社会偏见的能力，并探讨了模型在错误判断时的自信程度如何影响推理与反馈。

**💡 创新点**

创新点包括：①提出一种基于 DeepSeek 的对话重生成方法，将 StereoSet 的偏见语句植入真实辅导对话，构建了自然教学情境下的偏见数据集；②系统分析了模型在错误判断时的自信水平，并与其推理过程和学生反馈的一致性进行关联；③首次将置信度量化与人类评估结合，揭示过度自信在教育情境中的风险。

**🔧 技术方法**

技术手段主要包括提示工程（prompting）、置信度自述（confidence elicitation）和自我一致性（self‑consistency）方法；随后用标准分类指标（准确率、精确率、召回率、F₂）评估模型，并通过人类标注评估推理与反馈的连贯性。

**📊 数据集**

数据集：①自制 1,727 条学生–AI 辅导对话，包含 610 条刻板化语句、554 条反刻板化语句、563 条中性语句；②StereoSet 基准数据（与对话集同分布的 1,689 条样本）。

**📈 对比分析**

比较方法为在两组数据上执行同一分类任务，并通过置信度加权偏差评分（confidence‑weighted bias score）衡量错误时的自信程度。结果显示：在对话数据上模型整体表现下降 16–19%，且超过 30% 的错误具有极高或高自信；在传统基准上准确率约 0.66–0.75，但在对话场景仅 0.47–0.55。

**⚠️ 局限性**

局限性：①数据规模有限，缺乏多学科与多语言对话；②仅评估三款模型，未涵盖更大规模或开源模型；③缺乏多视角解释与校准机制，无法完全缓解过度自信带来的偏见放大；④实验聚焦英语第二语言学习场景，结果对其他教学领域的可推广性仍待验证。

---

## 321. Agent System Operations: Categorization, Challenges, and Future Directions

**arXiv ID:** 2606.01581 | [PDF](https://arxiv.org/pdf/2606.01581v1)

**作者:** Zexin Wang `[一作]` (CNIC, CAS), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 31487 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了LLM驱动代理系统的运维需求，提出了AgentOps框架，并给出代理系统异常的分类与检测、定位、修复策略。

**💡 创新点**

创新点在于首次针对代理系统构建完整的运维框架AgentOps，提出了面向 intra‑agent 与 inter‑agent 的异常 taxonomy，统一了监控、检测、根因定位与修复四个阶段，并梳理了现有技术与数据集。

**🔧 技术方法**

使用了多类技术：监控工具（LangDB、LangFuse 等）; 异常检测方法（白盒/灰盒/黑盒检测、Debate、CoK 等）; 根因定位技术（FAMAS、GraphTracer、LLM-as-judge 等）; 修复策略（预防性重构、动态再规划、回滚与重试）。

**📊 数据集**

参考的主要数据集包括 Who&When、MASFT、TRAIL、AgentFail、Aegis 与 AgentDebug 等，覆盖多种代理系统与异常类型。

**📈 对比分析**

对根因定位进行了基准评测，采用 step‑level 与 agent‑level accuracy 作为指标，比较了统计方法、图结构方法与 LLM‑judge 方法，结果表明无监督统计方法在长轨迹上更稳健，LLM‑judge 在细粒度定位上更精准，但整体缺乏统一的性能基准。

**⚠️ 局限性**

局限性主要体现在：缺乏统一、细粒度的评测标准；现有数据集对 inter‑agent 异常覆盖不足；大多数评估依赖 LLM 评审，易受模型偏差影响；长时序任务的在线检测与定位仍是挑战。

---

## 322. Defenses & Enablers For Skill Injection Attacks on Terminal Based Agents

**arXiv ID:** 2606.01567 | [PDF](https://arxiv.org/pdf/2606.01567v1)

**作者:** Yoshinari Fujinuma `[一作]` (Patronus AI), Anand Kannappan `[通讯]` (Patronus AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了技能注入攻击对LLM代理的威胁，并提出了静态与动态守护者防御以及攻击重构技术。

**💡 创新点**

引入守护者中间代理来实时或预处理技能文件，显著降低攻击成功率，并展示通过翻译和自动发现的攻击重构可放大威胁。

**🔧 技术方法**

使用MCP服务器实现动态守护、静态预处理脚本、LLM驱动的攻击搜索与翻译攻击等技术。

**📊 数据集**

基于Skill-Inject基准，48个注入、139个任务-注入沙箱。

**📈 对比分析**

与系统提示防御比较，动态守护将ASR降至≤15%，静态≈7%，保持任务成功率；攻击重构在无守护情况下将ASR升至81%，动态守护将其降至18%。

**⚠️ 局限性**

仅在Skill-Inject单一基准、三种代理模型、英语言环境下评估，未覆盖多语言、不同代理、异构守护模型及更广泛的攻击空间。

---

## 323. RobustModelMaker: Coupling Bootstrap Stability Selection with Leakage-Safe Nested Cross-Validation for Scientific Machine Learning

**arXiv ID:** 2606.01566 | [PDF](https://arxiv.org/pdf/2606.01566v1)

**作者:** Amanda S Barnard `[一作]` (Australian National University), Amanda S Barnard `[通讯]` (Australian National University)

**通讯引用:** 12504 | [OpenAlex ID](https://openalex.org/A5039459434)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并实现了 RobustModelMaker 框架，将 bootstrap 稳定性选择与嵌套交叉验证耦合，产生可复现且无泄漏的特征子集和性能估计，专为小至中等规模科学数据设计。

**💡 创新点**

创新点在于：① 统一解决不稳定特征选择与训练集与验证集共享导致的乐观偏差；② 引入基于中位重要度的稳定性阈值；③ 强化泄漏安全，所有预处理、选择、调参均在训练子集内部完成；④ 通过单文件、scikit‑learn 接口实现多算法支持，保证实验可复现和透明的性能-稳定性权衡。

**🔧 技术方法**

技术手段包括：bootstrap 稳定性选择（Meinshausen & Bühlmann 原理）、嵌套交叉验证、特征中位数插补、标准化、基于重要度的阈值筛选、九种算法支持（logistic、SVM、随机森林、梯度提升、XGBoost、神经网络等）、统计检验（Wilcoxon、t 检验、置信区间）以及并行化实现。

**📊 数据集**

实验使用三大真实科学数据集（SECOM 制造、Urban Land Cover、多类，Graphene Oxide Bulk 回归）以及两项实际应用（PLCO 卵巢癌生物标记发现，UCI 超导材料临界温度回归）。

**📈 对比分析**

与全特征基线、ANOVA F‑test、RFECV、Boruta 等四种选择方法在同一嵌套交叉验证框架下比较。结果显示：在三数据集上 RobustModelMaker 的预测得分与全特征基线相近或略优，同时在稳定性（Jaccard 相似度）上显著高于其他方法；在应用案例中获得可解释的特征子集，交叉验证与独立测试一致，体现了性能与可复现性的平衡。

**⚠️ 局限性**

局限性包括：不适用于大规模、流式或分布式训练；对基模型的重要度估计依赖，某些非线性树模型可能仍存在偏差；在强单变量关联数据（如 Graphene Oxide）上相对基线性能略低；并行执行时可能出现微小浮点差异，需单线程保证严格一致；需要根据数据特征手动调节 bootstrap 次数、阈值等参数以满足发表质量。

---

## 324. A Framework for Graph-Conditioned Hierarchical Shapley Attribution in Patent Valuation

**arXiv ID:** 2606.01632 | [PDF](https://arxiv.org/pdf/2606.01632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 325. Hierarchical Semantic-Augmented Navigation: Optimal Transport and Graph-Driven Reasoning for Vision-Language Navigation

**arXiv ID:** 2606.01565 | [PDF](https://arxiv.org/pdf/2606.01565v1)

**作者:** Xiang Fang `[一作]` (Huazhong University of Science and Technology), Changshuo Wang `[通讯]` (University College London)

**通讯引用:** 1059 | [OpenAlex ID](https://openalex.org/A5037445341)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为HSAN的框架，用层次语义场景图、最优传输规划和图感知强化学习实现连续环境下的视觉-语言导航。

**💡 创新点**

三大创新：①动态构建对象-区域-区域级的语义场景图；②基于最优传输的拓扑规划，理论保证语义与空间的最优平衡；③图感知低层控制策略，利用GCN提升子目标导航与避障性能。

**🔧 技术方法**

采用VLM（LLaVA-Onevision、SigLIP）生成语义描述，Spectral Clustering聚类对象，最优传输与Sinkhorn算法进行规划，GCN+PPO实现控制，Grounded‑SAM进行实例分割。

**📊 数据集**

在Matterport3D基础的R2R‑CE和RxR‑CE数据集上进行训练与测试。

**📈 对比分析**

相较于Cross‑Modal Matching、Waypoint Models、Neural Topological SLAM、Semantic MapNet、GraphNav等基线，HSAN在R2R‑CE/ RxR‑CE未见样本集上实现最高SR/ SPL，成功率提升约6‑8%，导航误差下降约7‑10%。

**⚠️ 局限性**

主要限制是实时构建层次图的计算开销较大，且对动态障碍物处理仍有限；未来工作计划降低图模型延迟并加入时序推理。

---

## 326. S-SPPO: Semantic-Calibrated Self-Play Preference Optimization

**arXiv ID:** 2606.01561 | [PDF](https://arxiv.org/pdf/2606.01561v1)

**作者:** Xiwen Chen `[一作]` (Morgan Stanley), Yuriy Nevmyvaka `[通讯]` (Morgan Stanley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析自我对弈偏好优化（SPPO）中的过度自信问题，提出双空间语义校准框架 S-SPPO，显著提升 LLM 的对齐性能。

**💡 创新点**

创新点在于：① 在标签空间使用语义门控实现监督校准，动态抑制语义相似样本的过度偏好；② 在潜在空间引入向量排斥力实现表示校准，防止流形坍塌；③ 证明校准后仍保持常数和博弈结构，确保收敛至 Nash 均衡。

**🔧 技术方法**

采用技术包括：Direct Preference Optimization（DPO）框架、Self-Play Preference Optimization（SPPO）、双空间最小-最大游戏设计、语义相似度编码器（如 Sentence‑BERT）、潜在空间正则化（向量排斥）、RLHF 训练流程。

**📊 数据集**

使用数据集：UltraFeedback（Prompt 数据）、PairRM（奖励模型）、AlpacaEval 2.0（评估），以及 MT‑Bench 与 Arena‑Hard‑Auto v0.1 用于多轮对话与复杂开放式问答评测。

**📈 对比分析**

与 SPPO、DPO、IPO、Snorkel、Self‑Rewarding LM 等基线对比，S‑SPPO 在 Llama‑3‑8B 上实现了 52.19% 原始赢率、47.46% 长度受控赢率，显著优于标准 SPPO 及其他自我对弈方法，且在 AlpacaEval 2.0 领导榜上超过多款更大规模专有模型。

**⚠️ 局限性**

局限性包括：仍依赖外部奖励模型的质量；对语义编码器的选取敏感；目前仅在少数基础模型（Mistral‑7B、Llama‑3‑8B）上验证，跨模型或更大模型的泛化尚待进一步研究。

---

## 327. Easier to Mislead Than to Correct: Harmful and Beneficial Revision in LLM Conformity

**arXiv ID:** 2606.01637 | [PDF](https://arxiv.org/pdf/2606.01637v1)

**作者:** Jiaming Qu `[一作]` (Amazon), Yibo Hu `[通讯]` (Illinois Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在控制实验中，作者让大型语言模型先给出答案，然后展示模拟同伴的回答，研究同伴一致性和权威标签如何影响模型的答案修订，区分修正错误（有益修订）与误导正确（有害修订）两种情况。

**💡 创新点**

创新点在于：①首次直接对比同伴一致性对有益与有害修订的相对影响，发现错误同伴更容易误导模型；②揭示权威标签无论答案正确与否都会强烈诱导模型跟随；③评估通用推理提示（CoT、反思）对修订质量的影响，并发现它们并不能可靠区分有害与有益修订。

**🔧 技术方法**

技术方法包括：使用开放权重LLM（如Llama、Falcon、Phi、Mistral）配合vLLM进行无温度生成；构造六个模拟同伴的提示，控制同伴答案一致性和权威标签；应用CoT和反思-再修订两种提示；统计分析采用混合效应回归和比值比（OR）检验。

**📊 数据集**

使用七个多选题目与推理数据集：BBH（几何形状、逻辑推理、时间序列、物体跟踪共250例）、MMLU-Pro（500例）、ARC-Challenge（500例）和TruthfulQA（500例），总计2500条实例。

**📈 对比分析**

对比方法：在同一实验框架下切换同伴一致性（混合、全正确、全错误）、权威标签数量以及推理提示，测算修订率、有害/有益修订率、信心变化；结果显示：全错误同伴将有害修订率从15.6%提升至62.9%，而全正确同伴只能将有益修订率从32.7%提升至51.5%；CoT在全错误条件下能显著降低有害修订，但在其他条件下抑制有益修订；反思-再修订整体降低修订率，既削弱了有害也削弱了有益修订。

**⚠️ 局限性**

局限性：①仅针对多选题，无法直接推广到开放式生成任务；②使用模拟同伴信息，未能捕捉真实多代理交互中的迭代对话和理由生成；③依赖模型自报的1–10置信度，缺乏严格的概率校准；④实验集中在开放权重LLM，未检验对闭源模型或不同规模模型的泛化性。

---

## 328. The Structural Influence of Low-Credibility Narratives During the COVID-19 Vaccine Rollout

**arXiv ID:** 2606.01630 | [PDF](https://arxiv.org/pdf/2606.01630v1)

**作者:** Lynnette Hui Xian Ng `[一作]` (Carnegie Mellon University), Kathleen M. Carley `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种结构化影响度量（Appeal 与 Scope），并利用它们评估 X 上 COVID‑19 疫苗低可信度叙事在不同时间阶段的传播效果。

**💡 创新点**

创新点在于把信息受欢迎度与发布者的网络结构相结合，形成结构化度量；同时揭示机器人与人类在预疫、疫苗发射和后疫三阶段的传播差异。

**🔧 技术方法**

使用 TwHIN‑BERT 词向量进行语义匹配、BotHunter 机器学习模型判定账号类型、Tweedie 回归模型估计结构化影响，并结合网络分析。

**📊 数据集**

基于公开的 CovidInfo 数据集，包含约 5.89 万条低可信度消息，及其在 X 上的交互网络。

**📈 对比分析**

通过 Mann‑Whitney U 检验、Kruskal‑Wallis 检验以及 Tweedie 回归比较机器人与人类的 Appeal 与 Scope。结果显示人类在大多数阶段具备显著更高的结构化影响，机器人在预疫期具有最高相对优势。

**⚠️ 局限性**

主要限制包括：阈值（机器人概率、语义相似度）选择的主观性；仅针对单一平台 X，缺乏跨平台验证；以及对时间段划分的依赖，可能影响结论的普适性。

---

## 329. IMWM: Intuition Models Complement World Models for Latent Planning

**arXiv ID:** 2606.01626 | [PDF](https://arxiv.org/pdf/2606.01626v1)

**作者:** Baoqi Gao `[一作]` (Beihang University), Song Wang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将世界模型与直觉模型结合的规划方法IMWM，利用示范检索和直觉得分来引导有限预算的采样规划，解决仅靠世界模型在搜索上的瓶颈。

**💡 创新点**

创新点在于：①将冻结的世界模型与专门训练的逆向直觉模型融合；②通过检索初始化、混合成本和可靠性门三种轻量级组件实现两者协同；③在有限查询下通过可靠性门自适应选择使用哪种策略。

**🔧 技术方法**

使用的技术包括：跨模态图像编码器、InfoNCE对比学习训练直觉得分器、Cross‑Entropy Method（CEM）采样规划、检索式初始化、混合成本（直觉+世界模型误差）以及基于两项诊断的可靠性门。

**📊 数据集**

在四个基于像素的目标到达任务上进行评估，分别为Two‑Room、Reacher、Push‑T和OGBench‑Cube，数据来自对应的模拟器帧（224×224像素）。

**📈 对比分析**

与冻结的世界模型+CEM基线对比，在48个不同seed/solver配置下，IMWM在Two‑Room和OGBench‑Cube上显著提升成功率（+11.5pp、+28.5pp），Push‑T略有提升，Reacher几乎相同。

**⚠️ 局限性**

局限性包括：可靠性门采用手工设计并且阈值固定，需在新任务上重新校准；模型仅在四个任务上验证，缺乏跨域泛化；直觉模型和检索库依赖示范数据，无法自我训练；未在真实机器人或更大规模任务上测试。

---

## 330. Real-Time Generation of Streamable Talking Portrait Video with Reference-Guided Deep Compression VAEs

**arXiv ID:** 2606.01620 | [PDF](https://arxiv.org/pdf/2606.01620v1)

**作者:** Sicheng Xu `[一作]` (Microsoft Research), Baining Guo `[通讯]` (Microsoft Research)

**通讯引用:** 50123 | [OpenAlex ID](https://openalex.org/A5101666011)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种实时流式的音频驱动肖像视频生成框架，能够根据语音和一张或多张参考图像即时合成高质量的说话人肖像视频。

**💡 创新点**

核心创新在于：1）利用参考图像引导的因果视频VAE实现极高压缩率（768倍）并提升解码质量；2）引入残差自动编码（CR‑VA）进一步提升重建精度；3）在此紧凑潜空间上构建块级自回归Rectified Flow Transformer，实现低延迟、可扩展的流式生成。

**🔧 技术方法**

使用技术包括因果视频VAE、参考图像Transformer融合、残差自动编码、Rectified Flow Transformer、块级自回归注意力、AdaLN、声学特征提取、LPIPS、SyncNet等。

**📊 数据集**

主要训练与评估数据集为VoxCeleb2（约50h）、自建的280h谈话肖像数据集，以及评测集HDTF和自采PortraitOneMin。

**📈 对比分析**

与Echomimic、Hallo系列、Sonic、FantasyTalking等大型扩散模型对比；在HDTF和PortraitOneMin上在唇同步、头部姿态、FVD等指标上与或优于现有方法，并实现42 FPS，速度比基准快25倍以上。

**⚠️ 局限性**

局限性包括：对参考图像数量敏感，更多参考图像可提升质量；模型对极端表情或动态场景仍可能失真；仅在单GPU上测试；缺乏完整闭环的实时语音输入测试；在不同光照或多语种环境下表现尚未充分验证。

---

## 331. Question Type, Cognitive Load, and CEFR Alignment: Evaluating LLM-Generated EFL Grammar Drill Exercises

**arXiv ID:** 2606.01592 | [PDF](https://arxiv.org/pdf/2606.01592v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 332. EvoPool: Evolutionary Programmatic Annotation for Label-Efficient Specialized Supervision

**arXiv ID:** 2606.01617 | [PDF](https://arxiv.org/pdf/2606.01617v1)

**作者:** Tianyi Xu `[一作]` (Oregon State University), Huazheng Wang `[通讯]` (Oregon State University)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5062299183)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 EvoPool，一种进化多智能体框架，自动生成可执行注释器池并通过 EvoAgg 文本感知聚合器将投票转换为软标签，用于在无标签数据上训练下游模型。

**💡 创新点**

创新点在于把注释器作者从一次性合成转变为可迭代进化过程，使用可判定的种群级自然选择、专门化的生成、改进和细化智能体，以及基于文本与投票特征的学习型聚合，显著提升专业领域标签质量与覆盖率。

**🔧 技术方法**

技术包括：基于 LLM（如 GPT‑4o‑mini）生成可执行 Python 规则；进化算子（Generator、Improver、Refiner）与池级过滤器（冗余、最小能力、边际贡献）；BatchBALD 作为查询选择；EvoAgg 通过文本嵌入+投票向量训练逻辑回归实现软标签；多轮实验验证不同 LLM 后端与下游模型。

**📊 数据集**

在 10 个数据集上评估，涵盖四类任务：常规分类（AGNews、Banking77）、高风险专业（ChemProt、DDI、Claude9）、复杂推理（FEVER、VitaminC、SciFact）和多标签生物医学（Ohsumed、PubMed）。

**📈 对比分析**

与 LLM 直接注释、Alchemist、DataSculpt、Self‑Training、黄金标签等基线对比。EvoPool 在 7/8 专业/复杂任务上平均提升 +0.141 macro‑F1，单任务最大提升 +0.301（ChemProt）。在下游模型上亦保持领先，且相较于单次 LLM 注释，EvoPool 的部署成本低 4500–31000 倍，且在 100K 示例上仅需约 0.21 美元。

**⚠️ 局限性**

局限性包括：仅在英文数据和结构化可程序化预测任务上验证；对多语言、多种子变异尚未探究；EvoAgg 需要每类约十个验证样本，极端不平衡类会退化为多数投票；训练阶段仍需 LLM 调用，成本依赖于初始作者模型；并未彻底审计注释器在稀有实体或边缘案例上的偏差。

---

## 333. Embedding Semantic Risk into Distance Fields and CBFs for Online Monocular Safe Control

**arXiv ID:** 2606.01605 | [PDF](https://arxiv.org/pdf/2606.01605v1)

**作者:** Dawei Zhang `[一作]` (Boston University), Zhiwen Fan `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个在线单目感知至控制的框架，利用基础模型驱动的稠密SLAM与语义分割生成语义感知的欧几里得签名距离场（ESDF），并通过控制障碍函数（CBF）实现安全导航与遥控。

**💡 创新点**

创新点在于：① 将语义信息直接嵌入ESDF的构造过程，实现类依赖的空间膨胀与风险分级；② 在CBF约束中使用类依赖的增益，实现对不同障碍类型的动态保守性调整；③ 通过TSDF中间层实现几何与语义的高效融合，保持10–20 Hz的在线运行速度。

**🔧 技术方法**

主要技术包括：基于MASt3R-SLAM的稠密三维重建；EfficientViT轻量级语义分割；TSDF到ESDF的转换；语义膨胀与选择；CBF-QP安全过滤；以及机器人运动学模型与实时优化求解。

**📊 数据集**

实验数据集主要为ScanNet++（六个室内场景）及对应的486条起点-终点轨迹；硬件实验在AgileX LIMO机器人上进行，使用单目RGB摄像头与RTX 3080显卡。

**📈 对比分析**

与SaferSplat（高计算量的高斯-溶解方法）和普通ESDF方法对比。结果显示：- 计算时间大幅降低（0.002 s vs 0.149 s）；- 任务完成进度与匹配进度更高；- 在语义ESDF下，平均前置干预距离、最小间隙与碰撞率分别提升/降低，表明风险分级有效提升安全性且不牺牲效率。

**⚠️ 局限性**

局限性包括：只适用于静态环境；对语义分割的准确性高度依赖；类依赖的增益和膨胀半径需手工预设或学习，缺乏自适应；未对动态障碍或更复杂几何形状的实时更新做进一步验证。

---

## 334. Estimating Mutual Information between Time Series and Temporal Event Sequences Across Diverse Analysis Tasks

**arXiv ID:** 2606.01602 | [PDF](https://arxiv.org/pdf/2606.01602v1)

**作者:** Haoji Hu `[一作]` (University of Minnesota - Twin Cities), Yao-Yi Chiang `[通讯]` (University of Minnesota - Twin Cities)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种非参数互信息估计器，可直接衡量连续时间序列与离散事件序列之间的相互依赖；

**💡 创新点**

在估计过程中引入了连续‑离散双重性建模与事件聚类技术，避免传统离散化、分布假设和事件冗余导致的偏差；

**🔧 技术方法**

采用KSG类非参数熵估计、连续‑离散混合分布模型、层次Wasserstein聚类以及偏差截断修正等方法；

**📊 数据集**

在合成数据、Minneapolis交通量、气温、Rossmann门店销量、M5零售预测以及标准分类基准集上进行实验；

**📈 对比分析**

与Series2Seq、Seq2Series、Ross、Mixture等基线以及自相关、傅里叶、AR、CatBoost、DeepAR、FMTimeSeries、Chronos‑2等模型对比，表明在TDMI、季节性识别、协变量排序、特征选择等任务中均取得更低误差、更高NDCG和更优分类准确率；

**⚠️ 局限性**

局限性在于对极端稀疏事件、强噪声和多维/非平稳时间序列的适应性仍有限，且事件聚类依赖于分布相似性，需要进一步研究更鲁棒的聚类与高维推广方案。

---

## 335. Effective Multi-sensor Conditioning for Street-view Novel-view Synthesis

**arXiv ID:** 2606.01590 | [PDF](https://arxiv.org/pdf/2606.01590v1)

**作者:** Zhengfei Kuang `[一作]` (Stanford University), Gordon Wetzstein `[通讯]` (Stanford University)

**通讯引用:** 19159 | [OpenAlex ID](https://openalex.org/A5014044649)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种多传感器融合的视频扩散框架，能够在稀疏 LiDAR 条件下从多视角合成街景视频，并支持极端离轨新视角。

**💡 创新点**

创新点包括：① 通过 Reference‑Enhanced Camera Attention 将 LiDAR 重投影、相机姿态和周围视角图像三种互补信号联合条件；② 采用相对光束级位姿编码实现跨视角自注意力；③ 两阶段稀疏训练策略，使模型在不同 LiDAR 密度下保持鲁棒。

**🔧 技术方法**

使用了大规模视频扩散模型 DiT、相对相机姿态编码（RayRoPE/UCPE）、Flow Matching 训练框架以及自注意力的 Ray‑level 位置编码。

**📊 数据集**

在 Waymo Open Dataset 上进行训练与评估，采集五摄像头与 LiDAR 的同步数据。

**📈 对比分析**

与 StreetCrafter、FreeVS、Gen3C 等基线及通用 V2V 模型对比，使用 PSNR、SSIM、LPIPS、FVD 评估，结果显示在 0.01 的 LiDAR 稀疏率下本方法显著优于所有基线，甚至可匹敌 10‑100 倍更密集 LiDAR 的方法，且在极端新视角下保持高度一致性。

**⚠️ 局限性**

局限性包括：仅在中等长度片段上验证，难以扩展到更长的行驶序列；仅聚焦于新视角生成，未涵盖场景编辑（插入/删除对象）等更高级的生成任务。

---

## 336. $\text{VG}^2$GT: Voxel-Gaussian Splatting Visual Geometry Grounded Transformer

**arXiv ID:** 2606.01573 | [PDF](https://arxiv.org/pdf/2606.01573v1)

**作者:** Yibin Zhao `[一作]` (East China University of Science and Technology), Jianjun Yi `[通讯]` (East China University of Science and Technology)

**通讯引用:** 1236 | [OpenAlex ID](https://openalex.org/A5086030465)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 VG^2GT，一种利用冻结视觉基础模型和多尺度可微体素模块，直接从无姿态 RGB 图像回归非像素对齐 3D 高斯场的高效 3D 重建与视角合成框架。

**💡 创新点**

创新点包括：①多尺度体素化实现几何一致性；②连续随机固体体积渲染监督提升深度精度；③保持 VFM 冻结，极大降低训练成本并实现插件式可扩展。

**🔧 技术方法**

技术手段涵盖：视觉基础模型（DINOv2/DA3）、PointTransformer/PointMLP、DPT 头、可微体素化、连续随机固体体积渲染、SH 颜色表示、深度/点图/SSIM/LPIPS/法向正则化等多种损失。

**📊 数据集**

使用数据集：Infinigen、KITTI、ScanNet、Replica、DTU、Tanks and Temples（TAT）等合成与真实室内外场景。

**📈 对比分析**

与 FreeSplatter、AnySplat、YoNoSplat、DA3 等无姿态 3DGS 方法比较，VG^2GT 在 DTU/Replica/TAT 的几何 RMSE、ABS、δ1.05 等指标显著优于 SOTA（例如 DTU RMSE 0.015 vs 0.022），视角合成 PSNR/SSIM/LPIPS 同样领先；训练成本仅 99 GPU 小时，推理时间约 2.44 秒。

**⚠️ 局限性**

局限性：仍依赖预训练 VFM；体素分辨率与分裂数的折中可能影响细节；在极大视角数或极端光照/遮挡场景下的鲁棒性尚待进一步验证。

---

## 337. GJDNet: Robust Graph Neural Networks via Joint Disentangled Learning Against Adversarial Attacks

**arXiv ID:** 2606.01560 | [PDF](https://arxiv.org/pdf/2606.01560v1)

**作者:** Canyixing Cui `[一作]` (Chongqing University of Posts and Telecommunications), Weina Niu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 14805 | [OpenAlex ID](https://openalex.org/A5100687681)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为GJDNet的图神经网络框架，旨在通过联合解耦节点表示和决策空间来提升对结构攻击的鲁棒性。

**💡 创新点**

创新点在于：①引入特征驱动的软结构解耦与偏度感知邻居过滤，分离多语义子空间以抑制攻击扰动；②设计球形决策边界，使类别在嵌入空间中形成紧凑球体，从而稳定决策边界并实现异常样本拒绝。

**🔧 技术方法**

采用了多子空间投影、双向注意力路由、偏度阈值过滤、Spherical Decision Boundary正则化以及交叉子空间路由等技术实现鲁棒表示与几何约束。

**📊 数据集**

在八个主流节点分类数据集上验证：Cora、Citeseer、Pubmed、Amazon Photo（同系图）和Cornell、Texas、Wisconsin、Actor（异系图）进行实验。

**📈 对比分析**

与GCN、GAT、GCN-Jaccard、RGCN、GCN-SVD、SimPGCN、ERGCN、ADGCN、GraphReshape等基线比较，在全范围结构攻击（Min‑Max、Nettack、Random）下，GJDNet在同系和异系图上均表现出最高或接近最高的准确率，鲁棒性显著优于现有方法。

**⚠️ 局限性**

局限在于：子空间数目固定、未显式建模高阶结构模式，且对动态图或更复杂攻击场景的适应性待进一步研究。

---

## 338. FlatVPR: Plug-and-play Geo-linear Residual Adapter for Geometric Rectification of Foundation Model Feature Manifolds

**arXiv ID:** 2606.01734 | [PDF](https://arxiv.org/pdf/2606.01734v1)

**作者:** Rai Hisada `[一作]` (University of Fukui), Kanji Tanaka `[通讯]` (University of Fukui)

**通讯引用:** 2449 | [OpenAlex ID](https://openalex.org/A5030913821)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Geo‑linear Residual Adapter与Pullback Flatness Loss，使得预训练DINOv2特征在极稀疏的锚点下可线性插值，从而在视觉定位中实现高精度、轻量化的地图构建。

**💡 创新点**

①将流形曲率自适应压平的残差变换；②将地图构建视为EM式自组织采样问题；③仅用单一季节训练即可跨季节推广。

**🔧 技术方法**

使用DINOv2 ViT‑S/14预训练特征，轻量级三层MLP残差适配器，Pullback Flatness Loss、Variance/Cosine保持损失，以及EM优化框架实现自组织锚点。

**📊 数据集**

NCLT北校园长周期数据集，涵盖27个季节、147.5 km的跨季节路线。

**📈 对比分析**

与DINOv2 baseline在110个跨季节对比中，平均MRR从0.697提升至0.872，Recall@1提升至82.98%；在100 m稀疏锚点下仍保持高性能，冬季‑春季单对比MRR从0.4439跃升至0.8415。

**⚠️ 局限性**

训练需长达500 epoch才能收敛；E‑step的锚点选择仍为概念化、缺乏自动化实现；在极端动态或快速变化环境下的鲁棒性未作充分验证；实验仅在单一基础模型上完成，跨模型泛化性待进一步验证。

---

## 339. Post-Deterministic Distributed Systems: A New Foundation for Trustworthy Autonomous Infrastructure

**arXiv ID:** 2606.01722 | [PDF](https://arxiv.org/pdf/2606.01722v1)

**作者:** Jun He `[一作]` (Openkedge), Deying Yu `[通讯]` (Openkedge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了 Post-Deterministic Distributed Systems (PDDS) 的理论框架，并设计了五大架构支柱（协议驱动开发、可验证代理基础设施、自治状态控制平面、语义仲裁保证、知识可见性复制）以协调确定性服务、随机模型、自治代理及人类操作员在分布式系统中的共识与一致性。

**💡 创新点**

创新点包括：① 抛弃传统确定性参与者假设，建立参与者通用模型；② 引入语义一致性与可验证证据链概念；③ 设计知识可见性复制（Epistemic State Replication）以及对应的语义线性化和事件一致性一致性范式；④ 定义语义仲裁保证与多代理协作的安全策略。

**🔧 技术方法**

使用的技术主要为：形式化模型与证明、可验证代理基础设施、协议驱动开发（Edit Automata）、自治状态控制平面、语义仲裁保证协议、知识可见性复制（Epistemic State Replication）等。

**📊 数据集**

论文未提供具体实验数据集或使用任何公开数据集，主要以理论推导与概念设计为主。

**📈 对比分析**

未进行实验对比或性能评估；作者通过与经典共识协议（Paxos、Raft、PBFT）的比较说明了 PDDS 在理论上的优势与差异。

**⚠️ 局限性**

局限性：当前框架仍处于理论阶段，缺乏正式语义定义、实现原型和性能评估；对多代理协作的安全性、可扩展性与实用性尚未在真实系统中验证；知识可见性复制的具体一致性协议与效率研究尚未完成。

---

## 340. Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs

**arXiv ID:** 2606.01710 | [PDF](https://arxiv.org/pdf/2606.01710v1)

**作者:** Afsaneh Hasanebrahimi `[一作]` (University of Melbourne), Sarah Erfani `[通讯]` (University of Melbourne)

**通讯引用:** 3959 | [OpenAlex ID](https://openalex.org/A5070030398)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Density‑Aware Translation（DAT）方法，利用小规模组参考集中的几何密度信息，对 CLIP 等 VLM 的图像‑文本相似度进行加权修正，以降低对伪相关性的敏感性，从而提升零样本分类的鲁棒性。

**💡 创新点**

创新点在于：①首次将局部密度估计（SLOF）与几何结构结合，直接在相似度上进行“翻译”校正；②从理论上证明该校正等价于在 Cosine 相似度中补齐缺失的贝叶斯对数似然项，恢复了对椭圆形分布的敏感性；③实现完全零样本、无需微调、无标签监督、可与任意 VLM 结合的通用框架。

**🔧 技术方法**

技术主要包括：CLIP（ViT‑B/32、ViT‑L/14、ResNet‑50）图像/文本编码；基于样本的几何密度估计（简化局部异常因子 SLOF）；局部几何信息聚合（exemplar herding）；密度修正后相似度重新加权与归一化；以及对齐与聚合的组合评分策略。

**📊 数据集**

实验数据集涵盖多领域的伪相关性基准：Waterbirds、CelebA、COVID‑19、FMoW；并在 ALIGN、AltCLIP 等其它 VLM 上进行扩展验证。

**📈 对比分析**

与传统零样本、基于提示的基线以及最近的零样本去偏方法（Ideal Words、Orth‑Cali、Perception CLIP、ROBOSHOT、TIE/TIE*）对比，DAT 在 WG（Worst‑Group）与 Avg 两项指标上普遍提升，WG 提升幅度可达 4%–14%，Avg 与 GAP 的改进也显著，尤其在 ViT‑L/14 与 ResNet‑50 上表现最突出。

**⚠️ 局限性**

局限性包括：①依赖参考集的选择与大小，过小或分布不均时可能影响密度估计；②对 SLOF 参数（k、λ）敏感，需要经验调优；③仅在单标签或多标签且伪相关性明显的场景表现优异，针对更复杂的因果结构或多模态交互的任务尚未验证。

---

## 341. JenBridge: Adaptive Long-Form Video Soundtracking across Scene Transitions

**arXiv ID:** 2606.01703 | [PDF](https://arxiv.org/pdf/2606.01703v1)

**作者:** Jiashuo Yu `[一作]` (Jen Music AI), Alex Wang `[通讯]` (Jen Music AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可解释的模块化框架JenBridge，用于生成长形式视频配乐并保证跨场景连贯性。

**💡 创新点**

创新点在于引入LLM导演的自适应过渡机制、双模态文本-视觉条件的生成模型以及新的长形式配乐基准LVS。

**🔧 技术方法**

核心技术包括基于Transformer的流匹配多模态扩散模型MMDiT、神经音频码流、双重条件编码、ControlNet生成过渡、LLM Qwen3-8B导演以及slerp插值。

**📊 数据集**

使用的主要数据集有大规模文本-音频对、V2M-20k视频-音频对、以及自建的LVS基准120条电影预告。

**📈 对比分析**

与四个现有开源基线（CMT、LORIS、AudioX、VidMuse）对比，在ImageBind、音乐美学和用户主观评分上均显著优于基线，尤其是过渡自然度提升了约0.9分。

**⚠️ 局限性**

局限性包括依赖大量GPU训练、对真实音轨缺乏直接评估、过渡策略仍受LLM生成质量限制，且未考虑音效与配乐整合。

---

## 342. Understanding Identity Continuity in Thermal Video through Scene-Level Consistency

**arXiv ID:** 2606.01694 | [PDF](https://arxiv.org/pdf/2606.01694v1)

**作者:** Wei-Chieh Sun `[一作]` (University of Washington), Jenq-Neng Hwang `[通讯]` (University of Washington)

**通讯引用:** 14289 | [OpenAlex ID](https://openalex.org/A5101702810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在热成像中使用轻量级后处理（短间隙重映射与保守长距离轨迹重连）来提升多目标跟踪中的身份连续性。

**💡 创新点**

提出将身份恢复视为场景级空间-时间一致性问题，证明保守的轨迹重连比增大在线追踪复杂度更能提升 IDF1。

**🔧 技术方法**

基于 YOLOv8 检测 + SORT 追踪，加上在线短间隙修复和离线基于时间、空间、运动、边界特征的两阶段身份修复。

**📊 数据集**

在 PBVS Thermal Pedestrian MOT 基准（30 条序列，总计 9000 帧）上进行验证，并在 6 条固定验证序列上做控制消融。

**📈 对比分析**

与 ByteTrack、BoT‑SORT 等主流轻量级追踪器做无后处理对比，原始 SORT 达到 82.25 IDF1，加入后处理后提升至 84.93，MOTA 与 MOTP 无明显变化，证明身份连贯性显著提升。

**⚠️ 局限性**

消融实验仅在单一热图基准上完成，缺乏与学习式轨迹关联方法的直接对比，且对不同摄像速率或拥挤度的泛化尚待验证。

---

## 343. Encoded but Not Routed: Explaining the Table-Chart Gap in Scientific Claim Verification

**arXiv ID:** 2606.01679 | [PDF](https://arxiv.org/pdf/2606.01679v1)

**作者:** Sunisth Kumar `[一作]` (University of Tokyo), Akiko Aizawa `[通讯]` (University of Tokyo)

**通讯引用:** 4577 | [OpenAlex ID](https://openalex.org/A5041062417)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比了表格和图表在科学主张验证任务中的性能差异，并通过层级线性探针和注意力分析揭示了图表信息未能被有效利用的原因。

**💡 创新点**

发现图表信息在中间表示中可恢复，但未能路由至最终预测位置，表明模型的路由机制存在缺陷而非感知缺陷。

**🔧 技术方法**

采用层级线性探针、注意力比例分析和链式思考提示等技术。

**📊 数据集**

使用了SciTabAlign+数据集，该数据集将相同内容以表格和多种图表形式呈现。

**📈 对比分析**

通过比较线性探针在平均池化与最终标记位置的AUROC，发现图表的平均池化AUROC高达约90%，但模型自身在图表上的准确率仅约56%，性能明显低于表格。

**⚠️ 局限性**

实验受限于小规模数据集、仅开放权重模型、诊断性而非因果性分析以及只关注二分类验证任务。

---

## 344. Why Do Self-Harm Prediction Models Struggle to Generalise? Lexical and Semantic Variations in Emergency Department Triage Notes

**arXiv ID:** 2606.01678 | [PDF](https://arxiv.org/pdf/2606.01678v1)

**作者:** Liuliu Chen `[一作]` (University of Melbourne), Vlada Rozova `[通讯]` (University of Melbourne)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5025162453)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对皇家墨尔本医院（RMH）与拉托布地区医院（LRH）2012–2017年急诊门诊分诊记录进行比较，分析词汇特征、重要预测特征和主题，探讨跨院自伤检测模型泛化的原因。

**💡 创新点**

首次系统比较两院临床文本的词汇差异与主题一致性，揭示文档写作差异导致模型跨院性能下降的机制，并提出通过文本标准化降低词汇变异以提升跨机构模型可泛化的思路。

**🔧 技术方法**

使用TF‑IDF+Chi‑Square+XGBoost进行特征选择，BERTopic进行主题建模，计算余弦相似度与Jaccard相似度，结合统计检验评估特征与主题差异。

**📊 数据集**

2012–2017年RMH 399,111条（1.4%自伤）和LRH 171,170条（1.7%自伤）分诊记录，均由心理学专家标注为自伤/非自伤。

**📈 对比分析**

通过词汇统计、特征重要性交叉对比和主题相似度分析比较两院数据，发现词汇分布差异导致跨站模型AUPRC从0.85降至0.78；主题语义相似度高（平均0.68），但词汇层面差异显著。

**⚠️ 局限性**

仅涉及两所澳大利亚医院，样本量和标注可能存在差异；仅使用TF‑IDF和BERTopic，未评估上下文嵌入或大语言模型的泛化表现，且数据无法公开。

---

## 345. DOT-MoE: Differentiable Optimal Transport for MoEfication

**arXiv ID:** 2606.01666 | [PDF](https://arxiv.org/pdf/2606.01666v1)

**作者:** Udbhav Bamba `[一作]` (Amazon), Deepak Gupta `[通讯]` (Amazon)

**通讯引用:** 16771 | [OpenAlex ID](https://openalex.org/A5034853815)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将预训练的稠密语言模型的 FFN 结构转换为稀疏的 MoE 形式，并通过可微分最优传输实现神经元到专家的分配。

**💡 创新点**

创新点在于把神经元分配视为可微分的最优传输（OT）问题，并使用 Straight‑Through Estimator 让分配和路由共同训练，从而在不冻结权重的情况下实现全流程端到端学习。

**🔧 技术方法**

采用可微分 Sinkhorn‑Knopp 迭代求解 OT、STE 进行离散决策的梯度传递、稀疏路由机制、KL/CE 损失等技术。

**📊 数据集**

在 LLaMA‑2‑7B、LLaMA‑3‑8B、Qwen2.5‑7B 三个预训练模型上使用 Dolmino‑mix 数据集进行对齐与微调。

**📈 对比分析**

与结构化剪枝、半结构化剪枝及现有 dense‑to‑MoE 方法比较，DOT‑MoE 在多项基准（WikiText‑2、HellaSwag、ARC‑Challenge 等）上保留约 90% 的原始性能，且在 50% 参数预算下 PPL 仅为 7.99，明显优于 DISP‑LLM、CMoE 等基线。

**⚠️ 局限性**

局限性包括 Sinkhorn 迭代带来的额外计算开销、对极端稀疏率或更大规模模型的收敛性验证不足，以及在极高稀疏率下专家分配可能出现不稳定或效率下降。

---

## 346. Quantifying the Energy Floor: Direct Measurement and Replay Buffer Bias in SAC-Based HVAC Control on sbsim

**arXiv ID:** 2606.01665 | [PDF](https://arxiv.org/pdf/2606.01665v1)

**作者:** Bo Li `[一作]` (Shanghai Jiao Tong University), Chen Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 53680 | [OpenAlex ID](https://openalex.org/A5100374115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过最小动作实验直接测量SB1建筑模拟器中SAC HVAC控制的能量底线，并评估重放缓冲初始化对成本的影响。

**💡 创新点**

①直接量化能量底线并发现其几乎完全由持续电负荷决定；②揭示重放缓冲初始化偏差导致SAC性能下降4.7%；③发现折扣因子耦合导致有效规划时长缩短至46分钟；④系统消融表明动作空间限制是瓶颈。

**🔧 技术方法**

Soft Actor-Critic算法、最小动作基线、重放缓冲初始化实验、折扣因子耦合分析、系统消融（奖励、规划时长、观测扩展）等技术。

**📊 数据集**

SB1建筑数字孪生（包含126热区、AHU、锅炉）的每日一次实验数据。

**📈 对比分析**

与规则时间表基线、预填缓冲SAC、空缓冲SAC、扩展动作空间SAC等对比。SAC空缓冲收敛至$35.57/天，接近测得的$35.51/天；预填缓冲SAC为$37.18/天，差距4.7%。

**⚠️ 局限性**

仅在夏季温和气候、动作空间边界最优的情境下适用；对于温度范围较宽、负荷需开启或不同建筑，结论可能不成立；重放缓冲策略需要更高质量演示才能避免偏差。

---

## 347. PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation

**arXiv ID:** 2606.01649 | [PDF](https://arxiv.org/pdf/2606.01649v1)

**作者:** Weixing Chen `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 32926 | [OpenAlex ID](https://openalex.org/A5100412937)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理一致性的桌面三维场景生成框架PhyScene3D，能够自动生成可直接用于物理仿真的无碰撞交互场景。

**💡 创新点**

核心创新包括：① 通过认知拓扑推理链CTRC将场景生成拆分为锚点条件的层次化顺序规划，显著减少因语义图缺乏空间约束导致的物理错误；② 物理感知去噪对齐PADA，将可微分的SDF引入测试时优化，实现对噪声监督的自我修正，并通过混合强化学习将物理约束直接注入生成策略。

**🔧 技术方法**

技术手段包括：可微分Signed Distance Field（SDF）物理引擎、层次化场景图与3D AABB相对定位、Test‑Time Optimization（TTO）与梯度投影、强化学习策略优化（GRPO）以及视觉‑语言模型（VLM）Fine‑Tuning。

**📊 数据集**

数据集：基于MesaTask-10k构建的MesaTask‑CTRC，包含9,429个训练样本和866个挑战性基准样本，涵盖6类桌面场景与5级难度。

**📈 对比分析**

与零样本LLM、基于求解器的代理方法及端到端回归基线（如MesaTask）以及扩散模型进行对比，PhyScene3D在场景级碰撞率（SCR）和资产级碰撞率（ACR）上分别下降至约1.6%和3.86%，比参考数据的81.5% SCR和人类注释数据大幅提升；在GPT-Score各维度和QPR指标上也显著优于所有基线，尤其在高密度场景和零样本场景中表现出色。

**⚠️ 局限性**

主要局限包括：TTO优化在部署时计算成本高、对初始生成质量依赖较大；强化学习阶段易出现奖励劫持导致语义漂移，需依赖PADA的混合策略缓解；当前框架主要针对桌面平面，扩展到更复杂空间（如多层结构或不规则桌面）仍需进一步研究。

---

## 348. Two-Fidelity Best-Action Identification for Stochastic Minimax Tree

**arXiv ID:** 2606.01708 | [PDF](https://arxiv.org/pdf/2606.01708v1)

**作者:** Peter Chen `[一作]` (Columbia University), Xi Chen `[通讯]` (New York University)

**通讯引用:** 58668 | [OpenAlex ID](https://openalex.org/A5100329996)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种在两种精度评估器（快速偏差评估与慢速无偏采样）共同作用下，对随机极小极大树进行固定置信度最佳动作识别的两阶段多精度搜索算法2FFS。

**💡 创新点**

创新点在于将多精度Bandit思想与minimax树搜索相结合，设计了递归局部证书与全局传播区间的两路决策机制，并给出了基于有效间隙的理论成本上界及PAC正确性与有限停止证明。

**🔧 技术方法**

使用了快速/慢速评估器、局部与传播区间构造、dyadic精度网格与递归费用模型，并在实验中与BAI‑MCTS、纯快速扩展及固定深度慢速采样等基线对比。

**📊 数据集**

实验数据基于合成平衡 b‑ary 随机极小极大树，参数集合{(5,8),(7,6),(10,3)}，共生成100棵树进行评估。

**📈 对比分析**

与三种基线对比，2FFS在采样数和运算量上分别低 160–1450 倍和 1.9–4.6 倍，且保持同等准确性；在不同树结构下均表现出显著的效率提升。

**⚠️ 局限性**

局限性包括：仅在合成树上验证，理论上对偏差界和噪声假设有局部正则性要求；对真实游戏或LLM rollout等实际场景的适用性仍需进一步实验验证。

---

## 349. Improving Visual Token Reduction via Rectifying Distortions for Efficient Multimodal LLM Inference

**arXiv ID:** 2606.01711 | [PDF](https://arxiv.org/pdf/2606.01711v1)

**作者:** Hyeonwoo Cho `[一作]` (Yonsei University), Bumsub Ham `[通讯]` (Yonsei University)

**通讯引用:** 4469 | [OpenAlex ID](https://openalex.org/A5054888241)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出RESTORE框架，改进视觉令牌缩减（VTR）方法以提升多模态大语言模型的推理效率。

**💡 创新点**

通过注意力校准恢复因位置信息失配导致的注意力衰减，并设计独特锚点选择机制降低合并过程的信息损失。

**🔧 技术方法**

使用RoPE/ M‑RoPE位置信息分析、距离感知注意力校准、基于相关性与多样性的锚点选择以及标准的视觉编码器-LLM流水线。

**📊 数据集**

在GQA、MMBench、MME、POPE、Science‑QA、VQA‑V2、TextVQA、SEED以及Qwen2.5‑VL‑7B‑Instruct等视觉问答与多模态基准数据集上进行实验。

**📈 对比分析**

与FastV、PDrop、SparseVLM、VisionZip、DivPrune、VisPruner、HoloV等多种VTR基线进行对比，在多模态LLM（LLaVA‑1.5/Next、Qwen2.5‑VL）上保持或提升98%以上精度，甚至在仅保留64视觉令牌时实现4–10%的性能提升，达到SOTA。

**⚠️ 局限性**

在极低视觉令牌比例下仍受RoPE长距离衰减限制，尽管计算开销微小但仍需额外矩阵运算；在更大模型或视频场景中的验证尚待进一步验证。

---

## 350. RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift

**arXiv ID:** 2606.01697 | [PDF](https://arxiv.org/pdf/2606.01697v1)

**作者:** Kilho Son `[一作]` (Microsoft), Dinei Florencio `[通讯]` (Microsoft)

**通讯引用:** 7304 | [OpenAlex ID](https://openalex.org/A5088040986)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 RCEM 模型，直接将多轮对话映射到重写后的独立查询的嵌入空间，无需在推理时调用 LLM 进行查询重写；

**💡 创新点**

创新点在于通过与 LLM 重写查询的嵌入对齐，而不是直接对话-文档匹配，避免对昂贵的对话-文档相关性标注的依赖，并保持原始嵌入空间兼容；

**🔧 技术方法**

采用冻结的基础嵌入器 G 加上 LoRA 适配器和两层 SELU MLP 头；训练时使用结构学习（SL）损失来对齐对话嵌入与重写查询嵌入；

**📊 数据集**

在 QReCC、TopiOCQA 与 TREC CAsT 三大对话检索基准上进行评估；

**📈 对比分析**

与主流两步重写+检索和直接对话嵌入基线对比，RCEM 在 Recall@10、MRR、NDCG@3 等指标上均提升 3%-20%，尤其在分布迁移场景下显著优于基线；

**⚠️ 局限性**

局限在于模型性能仍受 LLM 重写质量影响，若重写效果不足，RCEM 可能无法进一步提升检索效果

---

## 351. UniVocal: Unified Speech-Singing Code-Switching Synthesis

**arXiv ID:** 2606.01677 | [PDF](https://arxiv.org/pdf/2606.01677v1)

**作者:** Yufei Shi `[一作]` (Alibaba Group), Yang Ai `[通讯]` (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 UniVocal，一种统一的语音‑歌唱代码切换（SCS）合成框架，能够在单句中根据文本语义自动推断并切换语音与歌唱模式；

**💡 创新点**

创新点包括：① 通过两阶段课程学习在单一模型中实现语音与歌唱的无缝切换；② 设计可扩展的合成数据管道，利用 LLM 生成多场景脚本并合成高质量 SCS 语料；③ 引入细粒度“ refined cent token ”和 Chain‑of‑Thought (CoT) 计划策略，显著提升语音韵律与歌唱旋律的可控性；

**🔧 技术方法**

采用 CosyVoice 2 作为主干网络，使用语义 token 与 refined cent token 的交错自回归生成；辅以两阶段训练（先对齐语音与歌唱潜在空间，再训练切换能力）；利用 HiFi‑GAN 语音合成器完成波形重建；

**📊 数据集**

训练数据包括：262 h 的人工合成 SCS 语料（11,769 条样本）；960 h LibriTTS 语音数据；约 3,700 h Suno 歌唱数据（含 GTSinger 10 h 作为验证集）；以及通过 LLM 生成的多场景脚本；

**📈 对比分析**

在 SCSBench 上与 Gemini + Bark、Gemini + Cosy2 + LeVo 等基线对比，UniVocal 在混合子集上的 F1 分别为 0.871（目标）/0.810（主观）显著领先；在常规 TTS 与歌唱评测中，保持或优于现有模型（如 UTMOS、AES、N‑MOS 等），并在情感 TTS 上提升 0.48 MOS；

**⚠️ 局限性**

局限性主要体现在：① 合成歌唱数据的质量受源音源分离与 ASR 误差影响，存在电子音及歌词对齐误差；② 目前对纯隐式切换的泛化能力有限，仍需显式触发词；③ 自动评估指标（F1）对短样本的分辨率有限，需结合更细粒度的人类评测。

---

## 352. RDA: Reward Design Agent for Reinforcement Learning

**arXiv ID:** 2606.01672 | [PDF](https://arxiv.org/pdf/2606.01672v1)

**作者:** Hojoon Lee `[一作]` (Meta), Nitin Kamra `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于视觉语言模型的奖励设计代理（RDA），通过任务分解、视觉轨迹分析和迭代修正自动生成符合指令的奖励函数。

**💡 创新点**

创新点在于将视觉轨迹分析与子任务细化引入奖励设计循环，利用VLM直接诊断失败模式并指导奖励改进，而非仅依赖粗粒度奖励统计。

**🔧 技术方法**

使用的大型语言模型（GPT‑5/4）、视觉语言模型、强化学习（SAC）、演化搜索、子任务分解与视觉轨迹评估技术。

**📊 数据集**

使用了ManiSkill3中的12个桌面操作任务和HumanoidBench中的4个全身操作任务作为评估数据集。

**📈 对比分析**

与人类稀疏/密集奖励以及Eureka进行了对比，RDA在对齐率和成功率上均显著优于人类奖励，并在长周期任务中对齐率比Eureka高49%，整体性能与Eureka相当或更好。

**⚠️ 局限性**

主要限制包括高昂的计算成本（每个候选奖励需完整RL训练）、VLM的上下文长度和推理成本限制诊断精度，以及子任务奖励冲突导致难以同时满足所有子任务。

---

## 353. Time-Aware Diffusion based on Preference Disentanglement for Generative Recommendation

**arXiv ID:** 2606.01670 | [PDF](https://arxiv.org/pdf/2606.01670v1)

**作者:** Bangguo Zhu `[一作]` (Central South University), Senzhang Wang `[通讯]` (Central South University)

**通讯引用:** 6877 | [OpenAlex ID](https://openalex.org/A5035708362)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于时间感知扩散的生成推荐框架 TDPM，通过对用户偏好进行周期性和点性解耦，动态调节 SID 词元的掩码概率，实现非均匀扩散。

**💡 创新点**

创新点在于：①将用户偏好拆分为长期周期偏好和短期点偏好；②利用解耦的偏好动态生成时间感知掩码概率，替代传统均匀扩散；③在 SID 空间中引入受时间驱动的 token masking 与自回归解码。

**🔧 技术方法**

使用的技术包括：Diffusion 模型（正向噪声添加、反向去噪）、Semantic ID (SID) 生成与残差量化、时间感知 token masking、基于权重融合的周期/点偏好计算、受限 Beam Search 与 Trie 结构的生成解码。

**📊 数据集**

使用的公开数据集为 Amazon 购物评论的三类子集：All Beauty、Sports and Outdoors、Toys and Games（均经过 5‑core 预处理）。

**📈 对比分析**

与传统序列模型（GRU4Rec、Caser、SASRec、BERT4Rec、FMLP‑Rec）以及生成推荐模型（TIGER、LC‑Rec）和扩散推荐模型（PreferDiff、DDSR、PreferGrow）进行对比。实验显示 TDPM 在 HR@20、NDCG@20 等指标上分别提升约 16–29% 及 24–25%（最高 29.21% HR、25.45% NDCG），显著优于现有 SOTA。

**⚠️ 局限性**

局限性：①依赖于预先生成的 SID，生成过程可能受限于量化精度；②对超参数（α、β、λ_start、k）较敏感；③仅在 Amazon 购物数据上验证，未检验冷启动或跨域迁移能力；④扩散过程计算开销相对较大，实时性尚待改进。

---

## 354. ATLAS: Agentic Test-time Learning-to-Allocate Scaling

**arXiv ID:** 2606.01667 | [PDF](https://arxiv.org/pdf/2606.01667v1)

**作者:** Peijia Qin `[一作]` (University of California, San Diego), Pengtao Xie `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 Agentic Test-time Scaling 框架，LLM 作为 orchestrator 通过单一工具调用管理整个推理过程，包括何时展开新解、何时停止以及如何合成最终答案；

**💡 创新点**

创新点在于让 LLM 自主做资源分配与停止决策，摆脱外部规则；提供可扩展的动作空间（如多模型选择、指令聚焦等）；通过状态化证据管理（持久化观察历史）显著提升性能；

**🔧 技术方法**

技术方法包括：Claude Sonnet 4.6 作为底层模型；使用工具调用（AskSolver）发起独立解算；状态化证据管理与自适应停止；与自我完善、投票、预算强制等基线对比；

**📊 数据集**

数据集覆盖四大任务：HLE‑Verified（科学问答）、LiveCodeBench（代码生成）、GPQA‑Diamond（多模态推理）、BabyVision（多模态推理）;

**📈 对比分析**

与多种固定预算/自适应测试时扩展基线（Self‑Refine、Majority Voting、Budget Forcing 等）对比；在 Claude Sonnet 4.6 后端下，单模型版在四个基准上分别达到 56.00%、82.29%、85.86%、23.71%；多模型版进一步提升至 60.00%、85.63%、88.38%、23.97%；同时 API 调用次数明显低于固定预算方法；

**⚠️ 局限性**

局限性包括：需要手工设计动作空间和 prompt；探测器当前仅返回摘要而非完整推理轨迹；未实现端到端学习化的控制策略；在视觉类任务上的表现仍相对较弱；对极端难题的自适应性尚待进一步验证。

---

## 355. Gate the Filter, Not the Message: Node-Channel Mixtures for Pre-Propagation GNNs

**arXiv ID:** 2606.01660 | [PDF](https://arxiv.org/pdf/2606.01660v1)

**作者:** Zichao Yue `[一作]` (Cornell University), Zhiru Zhang `[通讯]` (Cornell University)

**通讯引用:** 7117 | [OpenAlex ID](https://openalex.org/A5037210004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在预传播图神经网络框架下，使用少量可学习的 Chebyshev 滤波专家并通过 3D 门控张量在节点与通道上联合路由的 Mixture‑of‑Experts 模型，解决了传统预传播聚合器难以同时兼顾节点级和通道级自适应的问题。

**💡 创新点**

创新点在于：①将滤波器视为预计算扩散基底上的图滤波，②将节点-通道自适应路由与共享滤波专家相结合，③引入基于谱响应的多样性正则以及两阶段响应感知路由器，显著提升预传播 GNN 的表达能力与鲁棒性。

**🔧 技术方法**

采用 Chebyshev 多项式滤波专家、稀疏/稠密 Mixture‑of‑Experts 路由、Spectral Response Sketch（SLQ）进行滤波器可视化与正则、以及多种辅助损失（多样性、平滑度、重要度、负载、z‑loss）来稳定训练。

**📊 数据集**

在 11 个基准图上进行实验，包含 8 个中小规模同质与异质图（如 Cora, Citeseer, ogbn-proteins 等）以及 3 个大规模网络（Pokec, ogbn‑products, ogbn‑papers100M）。

**📈 对比分析**

与 SIGN、HOGA、GAMLP 等传统预传播聚合器以及图 MoE、MP‑GNN 基线相比，所提方法平均提升 1.53 分、在 9/11 任务中夺冠，并在所有 3 个大规模图上获得最佳性能，特别是在大图环境下实现显著优势。

**⚠️ 局限性**

局限性包括：①在部分小规模、极度同质或异质图上仍不一定优于专业化的消息传播 GNN；②需要额外的预处理步骤（SLQ、Chebyshev 基底计算）和路由设计复杂度；③实验中对超参数（专家数、温度等）敏感，需一定调优。

---

## 356. CoreUnlearn: Rethinking Concept Unlearning through Disentangled Component-Level Erasure in Text-guided Diffusion Models

**arXiv ID:** 2606.01658 | [PDF](https://arxiv.org/pdf/2606.01658v1)

**作者:** Mengnan Zhao `[一作]`, Baocai Yin `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CoreUnlearn框架，能够在文本引导扩散模型中对不良概念进行有效去学习（erasure），同时最大限度地保持模型的整体生成质量和实用性。

**💡 创新点**

核心创新在于将概念嵌入拆解为多种组件，借助Component Extraction Module（CEM）和Swap Disentangling Strategy（SDS）识别并仅移除对去学习最关键的视觉稳健、文本敏感（V_rT_s）组件，从而实现“关键组件消除”而非整体消融。

**🔧 技术方法**

采用CEM（编码器+解码器）实现多维度分解；SDS通过视觉与文本噪声扰动以及跨组件交换构造解耦损失；在扩散模型的交叉注意力层进行细调；评估使用FID、LPIPS、分类ACC。

**📊 数据集**

主要实验数据集包括：Imagenette（目标对象去学习）、自建风格分类集（9种艺术风格）、性内容判定集（NudeNet）以及I2P类别集；使用Stable Diffusion v1.4与v2模型。

**📈 对比分析**

与FMN、SDD、ESD、AbC、UCE、RECE、ABO等基线对比，CoreUnlearn在对象和风格去学习任务中实现了与领先方法相当或更低的误判ACC，同时在FID/LPIPS上保持最低误差，且整体ΔACC（去学习+保留性能提升）最高，显示出最优的去学习与实用性平衡。

**⚠️ 局限性**

主要局限在于缺乏对抗鲁棒性（“深度去学习”）的支持，易受攻击性提示影响，且目前仅针对文本引导扩散模型的浅层去学习；未来需进一步提升对恶意提示的抵抗力和通用性。

---

## 357. THRD: A Training-Free Multi-Turn Defense Framework for Jailbreak Attacks on Large Language Models

**arXiv ID:** 2606.01738 | [PDF](https://arxiv.org/pdf/2606.01738v1)

**作者:** Zhiqing Ma `[一作]` (Beijing Language and Culture University), Pengyuan Liu `[通讯]` (Beijing Language and Culture University)

**通讯引用:** 1071 | [OpenAlex ID](https://openalex.org/A5100714941)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个无训练的多轮 jailbreak 防御框架 THRD，结合了瞬时风险评估、历史上下文分析、回复评估和时间演化决策。

**💡 创新点**

创新点在于显式建模对话轨迹中的风险累积，利用四个模块协同工作并通过时间递增评分机制捕捉跨轮次的攻击进展。

**🔧 技术方法**

技术包括基于提示的语义评估、跨轮意图检测、回复易害性评估，以及轻量级的衰减-聚合决策公式。

**📊 数据集**

使用了公开的多轮攻击数据集 X-Teaming 与 Tempest，以及单轮攻击 AutoDAN、HarmBench、AdvBench、MMLU、GSM8K、XSTest 等评测集。

**📈 对比分析**

与单轮防御方法 PROACT 与 SAGE 在 Qwen2.5-7B 与 Llama-3-8B 上对比，THRD 在 X-Teaming 与 Tempest 的攻击成功率降至 0.2–4.0%，模型效能下降仅 1.5%，拒绝率保持低于 20%，显示出优越的防御-实用性平衡。

**⚠️ 局限性**

局限包括对评估模型推理能力的依赖、对超长攻击序列的适应性有限、持续拒绝机制过于保守、以及 HCA 的计算瓶颈。

---

## 358. Argument Collapse: LLMs Flatten Long-Form Public Debate

**arXiv ID:** 2606.01736 | [PDF](https://arxiv.org/pdf/2606.01736v1)

**作者:** Yekyung Kim `[一作]` (University of Maryland), Mohit Iyyer `[通讯]` (University of Maryland)

**通讯引用:** 7255 | [OpenAlex ID](https://openalex.org/A5082767919)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在公共辩论文本生成中出现的“论点崩塌”现象，比较人类与多款LLM在主论点、子论点和论证结构上的多样性。

**💡 创新点**

首次系统量化不同LLM在多轮生成中对主论点、子论点及结构的重复率，揭示LLM生成文本多样性不足的“论点崩塌”问题。

**🔧 技术方法**

使用LLM判别器提取主论点与子论点，采用四级重叠标签判定相似性并计算唯一率；对段落进行双层结构标注，分析段落位置与过渡；在vanilla、diversified、position-guided三种提示策略下进行实验。

**📊 数据集**

New York Times Room for Debate（195 场辩论，1,039 条人类短篇）和 Boston Review 论坛（61 场辩论，448 条人类长篇），以及五款LLM（GPT、Claude、Gemini、DeepSeek、Minimax）生成的23,384 条文本。

**📈 对比分析**

通过比较人类与LLM的唯一率、覆盖率和结构分布进行评估；结果显示LLM主论点唯一率仅3.4%对比人类65.3%，子论点唯一率9.1%对比人类41%；多样化提示可提升至约50–80%但仍低于人类；LLM在结构上更趋向固定的“声明–支持–方案”轨迹，缺乏人类的多样化构建。

**⚠️ 局限性**

仅关注多样性不涉及论点质量；人类与LLM生成时间差异可能影响结果；研究仅聚焦公开辩论领域，结果可能不适用于其他写作场景；LLM-based注释可能存在系统性偏差。

---

## 359. Spatio-Temporal Correlation Guided Geometric Partitioning for Versatile Video Coding

**arXiv ID:** 2606.01701 | [PDF](https://arxiv.org/pdf/2606.01701v1)

**作者:** Xuewei Meng `[一作]` (Peking University), Siwei Ma `[通讯]` (Peking University)

**通讯引用:** 15967 | [OpenAlex ID](https://openalex.org/A5039832462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于时空相关的几何分割(STGEO)方案，用以提升VVC编码中的几何分割效率。

**💡 创新点**

通过高概率模式预测(HPS)和基于概率的合并候选列表(MCL)推断，显著降低分割模式与运动信息的信号开销。

**🔧 技术方法**

采用边缘检测、模板匹配、角度与距离量化、查表概率推断以及CABAC/Truncated Unary/Rice编码等技术。

**📊 数据集**

在JVET推荐的测试序列（如Cactus、BQMall等）以及多分辨率RA/LDB配置下进行评估。

**📈 对比分析**

与VTM-8.0基准（无GEO）对比，RA和LDB下平均BD‑rate分别下降0.95%和1.98%，并实现1–4%的码率节省。

**⚠️ 局限性**

主要限制在编码/解码复杂度略升高（RA约5%，LDB约7%），且对部分低分辨率或运动不显著的视频效果提升有限。

---

## 360. MixerSENet: A Lightweight Framework for Efficient Hyperspectral Image Classification

**arXiv ID:** 2606.01700 | [PDF](https://arxiv.org/pdf/2606.01700v1)

**作者:** Mohammed Q. Alkhatib `[一作]` (University of Dubai), Ali Jamali `[通讯]` (Simon Fraser University)

**通讯引用:** 1463 | [OpenAlex ID](https://openalex.org/A5046120119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了轻量级的 MixerSENet 网络，用于高光谱图像分类。

**💡 创新点**

创新点在于将 Mixer 结构与深度可分离卷积相结合，并加入 squeeze‑and‑excitation 块以提升特征表征，同时保持网络尺寸一致、参数极少。

**🔧 技术方法**

主要技术包括深度可分离卷积、点卷积（channel mixing）、SE 注意力模块、PCA 降维以及多尺度混合块。

**📊 数据集**

使用了 Houston13 与 QUH‑Qingyun 两个公开高光谱数据集进行实验。

**📈 对比分析**

与 3D‑CNN、HybridKAN、HSIFormer、SimPoolFormer、MorphMamba 等先进方法对比，MixerSENet 在两数据集上分别取得 OA 82.47% 与 96.70%，平均准确率、Kappa 指标均领先，对比模型参数仅 53k，推理速度也更快。

**⚠️ 局限性**

局限性包括深度可分离卷积在处理高度相关的光谱通道时效果受限；对极少训练样本的鲁棒性虽好，但在部分类别（如树木、河流）仍有提升空间；未来仍需探索更高级的光谱混合策略或结合 transformer 以进一步提升性能。

---

## 361. Learning Label-Efficient Interpretable Medical Image Diagnosis via Semi-supervised Hypergraph Concept Bottleneck Model

**arXiv ID:** 2606.01698 | [PDF](https://arxiv.org/pdf/2606.01698v1)

**作者:** Yijun Yang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Lei Zhu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种半监督概念瓶颈模型HyperCBM，旨在提升医学图像诊断的可解释性和数据效率

**💡 创新点**

创新点在于：①采用概念层超图（HECRL）捕捉高阶概念依赖；②采用图像层超图动态伪标签生成（HIDP）解决域差异和标注缺失；③通过两级超图实现概念推理与伪标签的联合学习

**🔧 技术方法**

核心技术包括超图卷积网络（HGNN+）、自注意力权重化超边、伪标签动态加权、二元交叉熵与对齐损失的联合训练

**📊 数据集**

使用三组数据集：新构建的胎盘粘连超声PAS数据集（45个概念、3级严重度），公共乳腺超声BrEaST数据集（7个概念），以及皮肤病变皮肤镜图像SkinCon数据集（22个概念）

**📈 对比分析**

与传统CBM、CEM、SSCBM及纯ResNet基线在不同概念标注比例（1%–80%）下对比；HyperCBM在所有数据集上均实现了更高的概念准确率和诊断AUC/ACC，尤其在低标注率下表现出显著优势，甚至在40%标注率时接近完全监督模型的性能

**⚠️ 局限性**

局限性包括：对预定义概念集合的依赖，可能无法覆盖所有临床细节；PAS数据规模有限，需跨中心验证；半监督伪标签易受域偏移噪声影响，可能影响解释性和诊断精度

---

## 362. IstGPT: LLM-based Anomaly Detection for Spatial-Temporal Graph in Industrial Systems

**arXiv ID:** 2606.01691 | [PDF](https://arxiv.org/pdf/2606.01691v1)

**作者:** Yuchen Zhang `[一作]` (Xidian University), Xiaolin Zhou `[通讯]` (Xidian University)

**通讯引用:** 2152 | [OpenAlex ID](https://openalex.org/A5075286263)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于大语言模型与图学习的工业控制系统异常检测框架，能够实时构建传感器-执行器的空间-时序图并检测异常。

**💡 创新点**

创新点包括：①多阶段 prompt engineering 使 LLM 逐步生成并验证精确的依赖图；②LLM‑Optimation 通过节点、边和逻辑一致性迭代优化图结构；③整合多模态工业知识（操作数据、文本文档、图像图纸）增强图构建可信度；④在图嵌入、时序编码与解码上使用 GAT+GCN+LSTM 的编码‑解码架构。

**🔧 技术方法**

技术栈：GPT‑4o 进行知识检索与图生成；图注意力网络 GAT 与图卷积 GCN 捕获空间-时序关系；LSTM 负责跨窗口时序编码；MSE+熵损失驱动重构；使用滑窗、归一化、自动相关周期估计预处理。

**📊 数据集**

使用 9 个数据集：公开 SWaT、WADI；7 个内部数据集（6 个工业仿真场景 + 1 个真实机器人臂），涵盖从 6‑39 维到 124 维的多尺度系统。

**📈 对比分析**

与 12 个 SOTA 基线（规则、序列、图、LLM 类）比较，平均 F1 提升约 3‑4%（SWaT 92.6%/WADI 94.5%），事件级 eTaF1 最高，在线检测延迟在毫秒级，整体训练耗时约 1 小时。

**⚠️ 局限性**

局限性：①需要丰富的工业多模态知识和文档支持；②LLM 仍可能出现幻觉，导致图结构需多轮修正；③离线图生成占主导时间，对极大规模系统仍有成本；④对具备图结构知识的隐蔽攻击尚未充分验证。

---

## 363. Conditional Collapse in Sign Language Production: A Diagnostic and a Scaling Argument

**arXiv ID:** 2606.01643 | [PDF](https://arxiv.org/pdf/2606.01643v1)

**作者:** Rui Hong `[一作]` (George Mason University), Jana Košecká `[通讯]` (George Mason University)

**通讯引用:** 7983 | [OpenAlex ID](https://openalex.org/A5086078885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于冻结运动自编码器的三层条件崩塌诊断指标，用来评估句子级手语生成模型的初始姿态、输出多样性和目标忠实度。

**💡 创新点**

创新点在于将生成质量分解为三独立层级指标，揭示传统 FID 与 BT‑BLEU 对文本-运动对应性的局限，并指出数据规模是导致忠实度低的主要瓶颈。

**🔧 技术方法**

使用 MoAE 将生成和真实手语动作编码为 256 维潜在向量，计算欧氏距离比值；实验涵盖扩散模型与回归模型两大架构。

**📊 数据集**

在 How2Sign（句子级）和 ASL3DWord（词汇级）两个数据集上进行评测。

**📈 对比分析**

与 FID、BT‑BLEU 对比发现 FID 与忠实度无关，所有 How2Sign 检查点在忠实度上仅达随机水平，而 ASL3DWord 则能够达到预期阈值，显示数据规模决定性能。

**⚠️ 局限性**

局限性包括只覆盖两种模型族、未探究大规模预训练、诊断依赖单一冻结 MoAE 的质量，可能在不同 MoAE 训练方案下表现不同。

---

## 364. mmAlert: A Simultaneous Device Localization and Target Tracking System via Cooperative Passive Sensing

**arXiv ID:** 2606.01653 | [PDF](https://arxiv.org/pdf/2606.01653v1)

**作者:** Chao Yu `[一作]` (Southern University of Science and Technology), Rui Wang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 48511 | [OpenAlex ID](https://openalex.org/A5100687842)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了mmAlert系统，利用毫米波通信的被动感知实现设备定位与目标轨迹跟踪的联合估计。

**💡 创新点**

首次将通信设备位置估计与被动目标轨迹重建联合化，并通过多条轨迹的多样性显著提升定位与跟踪精度，同时提出低复杂度的交替优化+EKF算法。

**🔧 技术方法**

采用多频段FDMA上行链路、相位阵列、bistatic Doppler与AoA测量、交替优化、扩展卡尔曼滤波、梯度下降与Levenberg‑Marquardt求解等技术。

**📊 数据集**

使用室内走廊实验数据，目标轨迹由Turtlebot机器人或志愿者（通过ZED深度相机记录）提供，真实轨迹以里程计或相机数据为标注。

**📈 对比分析**

与完美已知设备位置下的基准以及传统单目标方法比较，单轨迹下平均轨迹误差0.29 m、定位误差0.76 m；多轨迹（50条）下平均轨迹误差0.20 m、定位误差0.07 m，明显优于单轨迹或传统方法。

**⚠️ 局限性**

仅适用于单目标；对环境变化、遮挡、多目标情况的鲁棒性不足；需要多次轨迹采样才能达到高精度，单轨迹时精度显著下降。

---

## 365. Enhancing the Socioeconomic Understanding of Foundation Models with Urban Mobility

**arXiv ID:** 2606.01745 | [PDF](https://arxiv.org/pdf/2606.01745v1)

**作者:** Baoshen Guo `[一作]` (Singapore-MIT Alliance for Research and Technology), Shenhao Wang `[通讯]` (University of Florida)

**通讯引用:** 1102 | [OpenAlex ID](https://openalex.org/A5101998459)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文探索将人类移动网络嵌入基础模型，以提升城市社会经济预测，提出了MobFusion三模态融合框架；

**💡 创新点**

创新点在于将移动网络分别作为零射击LLM提示、视觉与文本嵌入的图连接以及多模态LLM的图令牌，系统化提升了基础模型的地理推理能力；

**🔧 技术方法**

采用零射击LLM提示、双跳异构R-GCN图神经网络、图令牌适配器注入多模态LLM，以及自监督对比预训练与信息论目标；

**📊 数据集**

使用三大城市（波士顿、芝加哥、纽约）的SafeGraph月度移动网络、POI数据、NAIP卫星图、AlphaEarth嵌入、2023年美国人口普查ACS数据和城市犯罪记录；

**📈 对比分析**

与仅POI提示、RidgeCV、MORA以及无图令牌等基线在Spearman ρ上比较，MobFusion在收入预测提升约+0.065、犯罪预测提升约+0.03，人口密度提升不显著；

**⚠️ 局限性**

局限性在于仅局限于CBG层级，细粒度缺失；缺乏对不同尺度或实时移动数据的探索；多模型集成导致计算成本较高。

---

## 366. TrafficRAG: A Multimodal RAG Framework for Traffic Accident Liability Determination

**arXiv ID:** 2606.01737 | [PDF](https://arxiv.org/pdf/2606.01737v1)

**作者:** Xu Li `[一作]` (Southwest Petroleum University), Xun Han `[通讯]` (Sichuan Police College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套多模态检索增强框架TrafficRAG，用视频信息自动生成交通事故责任认定报告。

**💡 创新点**

创新点在于：① 将视觉-语言模型用于结构化责任相关事实提取；② 采用双路检索（BM25+密集检索）获取法律条文和相似案例；③ 引入跨源一致性重排序模块提升证据契合度；④ 用结构化提示控制LLM生成合规、可解释的责任报告。

**🔧 技术方法**

技术手段包括：视觉-语言模型（InternVL2‑style + CLIP ViT‑L/14预处理），BM25+密集检索（FAISS + m3e‑base），交叉源一致性重排序（逻辑规则+向量相似度），以及大型语言模型（如GPT‑4V/类似）配合结构化模板进行报告生成。

**📊 数据集**

使用了自行构建的多模态交通事故数据集，包含1584个案例（视频+元数据+结构化描述+官方责任报告），视频来自VRU‑Accident和TAU‑106K，法律资源来自CADD、STARD、LeCaRDv2；外部检索库包括265条交通法规条文和671条案例。

**📈 对比分析**

与BM25、QLD、SAILER、LawRAG、Judge、DeepSeek‑V3.2、Gemini‑3.1‑Pro等基线进行统一评测，TrafficRAG在四项核心指标上领跑：KER 82.87%、SR 84.79%、LNA 77.32%、FF 81.71%，并在责任比例误差LR‑MAE上实现最低5.48%（相对竞争者高出约10%）。

**⚠️ 局限性**

局限性包括：需人工复核才能具备法律效力；仅覆盖中文交通法与案例，难以跨司法区使用；视频理解误差、检索偏倚、法律适用不稳定以及历史案例偏见等问题仍需进一步缓解。

---

## 367. Evidence-Gated LLM Priors for Multi-Objective Bayesian Optimization

**arXiv ID:** 2606.01730 | [PDF](https://arxiv.org/pdf/2606.01730v1)

**作者:** Jiangyu Chen `[一作]` (State Key Laboratory for Novel Software Technology, Nanjing University), Banyi `[通讯]` (State Key Laboratory for Novel Software Technology, Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究将LLM专家先验视为可验证的目标级先验，在离散多目标贝叶斯优化中构建基于信誉市场和反事实门控的自适应权重与是否使用先验的机制。

**💡 创新点**

①为每个专家-目标对维护独立信誉并实现在线更新；②设计两级反事实门控（先验使用与置信度更新）动态决定是否采用或抑制LLM先验；③实证表明LLM自报置信度并非统一有效，需要经过门控。

**🔧 技术方法**

采用高斯过程残差模型的多目标BO、专家信誉市场（在线奖励折扣与资本转换）、温度软最大化、随机标量化UCB采集、GP似然评分的反事实门控及Hedge更新。

**📊 数据集**

合成压力测试；三组分子优化基准——ESOL、FreeSolv、Lipophilicity。

**📈 对比分析**

与随机、原生MOBO、固定LLM先验、qLogNEHVI、ParEGO等基线比较，在ESOL与FreeSolv上获得最高最终超体积，Lipophilicity表现与原生MOBO相当，证明置信度门控提升鲁棒性。

**⚠️ 局限性**

先验缓存规模有限；对置信度的学习仅在部分任务有效；drop‑margin策略未达预期；对极端误导先验的自适应性仍不足，且需在更大规模候选集上进一步验证。

---

## 368. CANARY: Zero-Label Detection of Fine-Tuning Contamination in Language Models

**arXiv ID:** 2606.01695 | [PDF](https://arxiv.org/pdf/2606.01695v1)

**作者:** Swapnil Parekh `[一作]` `[通讯]` (Intuit Inc.), Swapnil Parekh (Intuit Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于稀疏自编码器的零标签检查点审计器CANARY，能在1%污染率下检测语言模型的微调污染。

**💡 创新点**

通过隐藏层几何差异而非输出，利用稀疏自编码器过滤噪声并提供可解释特征，实现了最早7.5×检测，且支持后续修复。

**🔧 技术方法**

稀疏自编码器（SAE）对基模型激活进行压缩，噪声特征屏蔽，隐藏层差分注入与logit放大，及推理时特征抑制。

**📊 数据集**

使用合成医学建议数据集，包含1%至20%有害示例混入基准数据，作为微调训练和评估。

**📈 对比分析**

与多种无监督基线（如Logit KL、L2差异）和有监督方法对比，AUROC均为1.0，在1%污染率下保持极高的检测率，生成层放大在隐藏层注入时峰值有害率提升5×且困惑度仅58。

**⚠️ 局限性**

需已知可信基准检查点；在LoRA rank‑4低秩微调或高方差模型时检测阈值升高，且对极大规模70B模型的验证仍待验证。

---

## 369. Scalable Concurrent Queues for GPU

**arXiv ID:** 2606.01693 | [PDF](https://arxiv.org/pdf/2606.01693v1)

**作者:** Pratheek Prakash Shetty `[一作]` (Virginia Tech), Wu-chun Feng `[通讯]` (Virginia Tech)

**通讯引用:** 8586 | [OpenAlex ID](https://openalex.org/A5058539554)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了GPU上三种具有强进度保证且内存有界的并发队列，涵盖从锁自由到等待自由。

**💡 创新点**

首次在GPU上给出可证明的等待自由队列，并通过wave‑batched fast path改进锁自由队列的吞吐量。

**🔧 技术方法**

使用HIP、64位CAS、FAA、分段预分配、并行批量取值等技术，并结合Porcupine等线性化验证工具。

**📊 数据集**

在AMD MI210与MI300A GPU上进行固定时长吞吐微基准、层同步BFS与基于tile的波前光线追踪，使用SuiteSparse图集与两套场景。

**📈 对比分析**

通过与SFQ、G‑WFQ‑YMC及Gunrock、流式压缩基线对比，G‑LFQ在平衡工作负载下最高吞吐，G‑WFQ在非平衡或高并发下更稳健，整体性能可匹敌或超过现有基线。

**⚠️ 局限性**

受限于GPU上64位CAS的可用性、慢路径激活开销及部分应用场景对FIFO不严格需求，导致在极端非平衡或高线程计数时吞吐下降。

---

## 370. Off-the-Shelf LLMs as Process Scorers: Training-Free Alternative to PRMs for Mathematical Reasoning

**arXiv ID:** 2606.01682 | [PDF](https://arxiv.org/pdf/2606.01682v1)

**作者:** Atoosa Chegini `[一作]` (University of Maryland), Soheil Feizi `[通讯]` (University of Maryland)

**通讯引用:** 10703 | [OpenAlex ID](https://openalex.org/A5025450606)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的生成指导方法：在小模型生成过程中，以固定长度块为单位采样候选，使用大型预训练模型的似然对每块进行评分并选取最佳块来推进生成。

**💡 创新点**

创新点在于：①利用现成的大语言模型仅作为概率评分器，省去奖励模型训练；②采用固定长度块避免大模型对步长的偏好导致的评分偏差；③引入差分评分（大模型-小模型）以强化两模型互补。

**🔧 技术方法**

核心技术包括：多样本块采样（k 个长度 L 的块）、大模型似然估计、两种评分规则（LGS 和 CGS）、基于块级评分的生成‑选择‑追加循环。

**📊 数据集**

使用五个数学推理基准数据集：GSM8K、MATH、Minerva Math、AMC23 和 AIME24，评估 Qwen2.5‑1.5B→32B、Llama‑3.2‑1B→70B 以及 Qwen2.5‑7B→72B 等模型对。

**📈 对比分析**

与多数后置选择方法（Majority@k、Best‑of‑N、Self‑Certainty、Borda Count）以及 PRM‑guided 搜索进行对比。该方法在大部分基准上实现了 12–28pp 的提升，尤其在 GSM8K、Minerva Math 和 AMC23 上显著优于 PRM；在小模型已较强的 7B→72B 组合中仍能逼近奖励模型指导的性能，且产生更短的推理轨迹。

**⚠️ 局限性**

局限性包括：仅在数学推理任务上验证，未探讨编程或常识推理等领域；实验仅在同一系列模型内部进行，跨系列效果未知；固定长度块的设置需手动调参，变量步长评分方法仍待进一步研究。

---

## 371. SECUREVENT: Hybrid AI/ML Security Monitoring for Distributed Event-Based Systems

**arXiv ID:** 2606.01741 | [PDF](https://arxiv.org/pdf/2606.01741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 372. Don't Let a Few Network Failures Slow the Entire AllReduce

**arXiv ID:** 2606.01680 | [PDF](https://arxiv.org/pdf/2606.01680v1)

**作者:** Peiqing Chen `[一作]` (University of Maryland), Zaoxing Liu `[通讯]` (University of Maryland)

**通讯引用:** 1437 | [OpenAlex ID](https://openalex.org/A5015818714)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对GPU集群中网络故障导致的AllReduce性能下降，推导了信息理论下的最优完成时间下限，并设计了一种四阶段流水线AllReduce算法OptCC，使其在出现单一或多重 straggler 的情况下仍能接近该下限。

**💡 创新点**

创新点在于：1）首次给出在异构带宽环境下AllReduce的下限并证明其可达性；2）将straggler带宽损失分离为独立的慢速链路与高速环路，利用流水线并行实现零空隙调度；3）提出bubble‑filling技术，在带宽不足时自动利用剩余空隙完成额外聚合，进一步逼近下限。

**🔧 技术方法**

主要技术包括信息理论下限分析、四阶段分解（Reduce‑Scatter、Upload、Download、All‑Gather）、流水线调度与闭式模式组合、P2P bubble‑filling、NVLink内部聚合与跨节点交互的分离、以及O(pk)时间的在线调度生成。

**📊 数据集**

实验使用 SimAI（基于NS‑3的GPU网络模拟器）在配备8个GPU/8个HDR InfiniBand NIC 的A100服务器集群上进行，覆盖单/多 straggler 及多GPU/服务器场景，测试多种带宽下降比例（最多50%）和不同GPU规模（16–256）。

**📈 对比分析**

与基准 NCCLNoFailure（无故障）、ICCL（传统 failover）和 R^2CCL（最新故障容忍）比较，OptCC 在所有场景下相对 NCCL 的开销仅为 2–6%，远低于 R^2CCL（最高 57%）和 ICCL；在单 straggler/多 straggler/多GPU/服务器情况下均保持 2–8% 内的性能。

**⚠️ 局限性**

局限性包括：① 对最坏 NIC 仍需保留至少 50% 带宽；② 当 p 较大时，P2P 同步开销随规模增长，略微拉高整体时间；③ 仅在带宽受限场景下评估，未覆盖显著的延迟敏感或动态带宽变化情况；④ 多 straggler 方案在理论上可行但实验未展示。

---

## 373. Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment

**arXiv ID:** 2606.01651 | [PDF](https://arxiv.org/pdf/2606.01651v1)

**作者:** Huayang Huang `[一作]` (Wuhan University), Ye Zhu `[通讯]` (École Polytechnique)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5133100012)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种Geometry-Aware Distillation (GAD) 框架，用以在文本到图像模型蒸馏过程中保留对初始噪声的敏感性，解决传统点对点蒸馏导致的模式坍塌与可控性下降问题。

**💡 创新点**

创新点在于引入Jacobian‑Vector Product（JVP）对齐损失，通过匹配教师与学生在噪声扰动方向上的梯度响应，恢复局部几何结构与噪声敏感性；该方法可无缝融入多种蒸馏范式，且不增加推理成本。

**🔧 技术方法**

技术包括：基于JVP的几何对齐损失、有限差分近似、在不同时间步和噪声尺度上进行稀疏监督、以及对多个主流文本到图像架构（SD、PixArt‑α、SANA）和蒸馏方法（Output Matching、Distribution Matching、Score Identity）进行统一训练。

**📊 数据集**

主要使用公开数据集：MS‑COCO（用于图像生成与布局控制评估）、CC12M（用于多样性评估）、DrawBench（用于零样本控制迁移），并在这些数据上与多种蒸馏基线进行比较。

**📈 对比分析**

与11种基线相比，GAD在保持或略提升 CLIP Score、PickScore 的同时，显著提升了噪声敏感性（自我可识别率与教师对齐率提高至≈92%）、布局控制精度（AP提升至5.8，接近教师的6.6）、以及多样性（Vendi Score 约1.1–1.2倍）。实验显示，GAD在不同架构和蒸馏方法上均能统一提升可控性与多样性。

**⚠️ 局限性**

局限性包括：训练时需要额外的前向传播以近似 JVP，导致训练成本上升；目前仅验证于单帧图像蒸馏，对视频或时序蒸馏的推广尚未评估；以及对高维噪声下的数值稳定性和理论收敛性尚缺乏深入分析。

---

## 374. When Meaning Travels: A Granular Lens on Hybrid-MoE's Role in Idiomatic Understanding for Language Models

**arXiv ID:** 2606.01671 | [PDF](https://arxiv.org/pdf/2606.01671v1)

**作者:** Sarmistha Das `[一作]` (Indian Institute of Technology Patna), Sriparna Saha `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 8216 | [OpenAlex ID](https://openalex.org/A5060797340)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了多模态成语语料库 Varnika，并提出 HybridMoE 框架与 Idiomatic Property Signals (IPS) 以提升低资源多语言成语理解。

**💡 创新点**

创新点：① 将七种情感色彩（幽默、嘲讽、亲情、期望、恐惧、悲伤、欺骗）细粒度标注成语语调；② 采用混合专家路由与全局 HyperNetwork 的 HybridMoE 以跨模态捕捉成语隐喻；③ 设计 IV（Idiomatic Validation）和 IDIO‑TONE 两种复合评测指标。

**🔧 技术方法**

技术：HybridMixture‑of‑Experts 与 IPS；ViT‑patch16 视觉编码 + BERT 文本编码；Einstein‑Summation 融合专家输出；Zero‑shot Qwen2.5‑VL‑7B 评估；LoRA、PEFT 细化微调；混合多模态注意力。

**📊 数据集**

数据集：Varnika（3,533 条印地语、孟加拉语、泰语成语，包含文本‑图像对和七色调标签），来源于 Mediom、FLUTE、V‑FLUTE 等；另外使用公开的多模态成语评测集。

**📈 对比分析**

对比方法：在 Blip2、Qwen2.5‑VL、Gemma、Llava 等 VLM 上分别做标准 fine‑tune 与 HybridMoE+IPS；HybridMoE 在 ROUGE‑L、BLEU‑3、BERTScore、IVS、IDIO‑TONE 上均提升，最显著的 Qwen2.5‑VL‑Instruct+HybridMoE 在 IVS 上达 0.82、IDIO‑TONE 0.47，整体约提升 5–6% 以上。

**⚠️ 局限性**

局限：① 对视觉语义的依赖导致抽象成语难以处理；② 语调标签主观性与多重标签导致标注不一致；③ IV 指标聚合模糊细节；④ 成语含义随语境动态变化，评测仍是静态；⑤ 未加入外部文化知识与推理机制。

---

## 375. Characterization of Multi-Model Agentic AI Systems on General Tasks via Trace-Driven Simulation

**arXiv ID:** 2606.01725 | [PDF](https://arxiv.org/pdf/2606.01725v1)

**作者:** Donghwan Kim `[一作]` (Pennsylvania State University), Kiwan Maeng `[通讯]` (Pennsylvania State University)

**通讯引用:** 1361 | [OpenAlex ID](https://openalex.org/A5035750684)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了两套复杂多模型代理 AI 系统在 GAIA 基准任务上的 token 级 trace 数据集（GAIATrace）以及基于此的 trace‑driven 仿真器（GAIATraceSim），用于系统层面研究 agentic AI 的行为与性能。

**💡 创新点**

创新点主要包括：①首次公开包含完整推理 tokens、任务级结构及所有参与 LLM 的 trace 数据；②在 Vidur 基础上扩展了异构模型、prefill–decode 分离、KV 与前缀缓存、工具调用延迟建模等功能的仿真框架；③利用上述工具对 GAIA 任务中多模型、多工具的 agentic AI 进行系统性表征与性能分析。

**🔧 技术方法**

技术实现基于 Vidur LLM 仿真器，加入对工具调用的延迟建模、前缀缓存与 KV 缓存机制、prefill–decode 分离、张量并行/流水线并行、任务级与查询级调度策略（Q‑FCFS、T‑FCFS、SJF、SJF+timeout）以及动态 GPU 分配。

**📊 数据集**

使用的数据集为 GAIA 基准任务集，结合两套 agentic AI 系统（一个 ReAct‑like、一个层次化多 Agent）在此基准下生成的 token 级 trace，涉及开源与闭源 LLM、工具调用等多种模型。

**📈 对比分析**

实验通过仿真不同 GPU 配置、前缀缓存、调度策略和任务到达率，对系统延迟、TTFT/TPOT 等指标进行比较。结果显示：前缀缓存可将 p50 任务延迟提升 1.67–3.82×；SJF 调度将 p50 延迟降低 2.22×；GPU 分配需随负载动态调整；大模型下 timeout 调度效果下降。

**⚠️ 局限性**

局限性包括：仅覆盖两套特定 agentic AI 系统；trace 仅基于 GAIA 任务，难以直接推广至其他基准；工具调用延迟模型不完美；仿真使用 CodeLlama‑34B 的性能模型，未针对实际模型精调；开放源代码与数据集发布仍待后续。

---

## 376. Shortcut to Nowhere: Demystifying Deep Spurious Regression

**arXiv ID:** 2606.01723 | [PDF](https://arxiv.org/pdf/2606.01723v1)

**作者:** Guanrong Xu `[一作]` (University of California, Los Angeles), Yuzhe Yang `[通讯]` (University of California, Los Angeles)

**通讯引用:** 15828 | [OpenAlex ID](https://openalex.org/A5108048509)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了连续回归任务中的伪相关问题，并提出了两种基于MDS的分布平滑方法以提升模型泛化能力。

**💡 创新点**

提出了DSR（连续回归中的伪相关）框架，定义了标签与属性的连续分布平滑，并引入Label-MDS与Feature-MDS两种利用标签分布或特征表示相似度的平滑策略。

**🔧 技术方法**

使用多维尺度映射（MDS）构建属性相似度矩阵，结合核平滑、标签分布平滑（LDS）和加权回归损失；同时采用Wasserstein距离和特征中心点距离计算属性亲和度。

**📊 数据集**

在视觉、环境感知和自然语言处理领域收集并基准化了四个DSR数据集：UTKFace（年龄/种族）、天文相机温度（相机ID）、PovertyMap（贫困指数/国家）以及CodeNet运行时（编程语言）。

**📈 对比分析**

与ERM、逆频率重加权、Class-Balanced、LDS、DANN、GroupDRO、RnC等基线对比，实验表明Label-MDS、Feature-MDS在总体MAE和GM上均取得显著提升，尤其在少样本、零样本以及低训练数据量场景表现最优。

**⚠️ 局限性**

在数据丰富区的性能略逊于部分基线；方法假设训练阶段已知伪属性，且对属性多样性和未观察属性的泛化仍有限；在极低数据环境下可能仍受限于属性相似度估计误差。

---

## 377. A Note on Stability for Orthogonalized Matrix Momentum with Client Sampling

**arXiv ID:** 2606.01720 | [PDF](https://arxiv.org/pdf/2606.01720v1)

**作者:** Da Chang `[一作]`, Ruijie Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 FedMuon 方法的基础上，给出了其一侧高概率风险差距（risk‑gap）上界的详细证明。

**💡 创新点**

创新点在于：① 提出了一种新的风险差距上界证明技巧，能够在非独立同分布（Non‑IID）环境下获得更紧的高概率保证；② 通过对 FedMuon 的迭代过程进行细致的分析，揭示了局部梯度偏差与全局收敛速度之间的关系。

**🔧 技术方法**

主要技术包括：统计学习理论中的集中不等式（如 Hoeffding、Bernstein 等）、渐近分析与序列收敛理论，以及对联邦学习中通信约束的数值逼近。

**📊 数据集**

在理论证明中未直接使用具体数据集；但在主论文（非附录部分）通常会以 FEMNIST、Shakespeare、MNIST 等常用联邦学习数据集进行实验验证。

**📈 对比分析**

对比方法：将 FedMuon 的理论风险上界与传统 FedAvg、FedProx 等算法的经验风险进行对比。实验结果表明，FedMuon 在保持相同通信预算的前提下，能够显著降低理论风险上界，实际训练误差亦与理论预测高度一致。

**⚠️ 局限性**

限制与不足：① 证明依赖于较强的假设（如梯度光滑性、Lipschitz 连续性），在极端非 IID 或高度噪声环境下可能失效；② 只给出了理论上界，实际训练过程中仍需进一步实验验证其鲁棒性；③ 对于大规模模型和极低通信频率的场景，证明中的常数项可能过于保守。

---

## 378. Fair Finetuning Mitigates Distribution Inference Attacks

**arXiv ID:** 2606.01719 | [PDF](https://arxiv.org/pdf/2606.01719v1)

**作者:** Rakshit Naidu `[一作]` (Georgia Institute of Technology), Rakshit Naidu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5075268897)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出公平微调（FFt）方案，利用等化赔率约束对已训练模型进行后训练微调，以降低对分布推理攻击的泄露；

**💡 创新点**

首次将公平性度量（等化赔率差距 Δ_EO）与分布推理攻击优势联系，给出紧密理论上限 Adv ≤ Δ_EO · W，并在偏置分布协议下简化为 Adv ≤ Δ_EO；

**🔧 技术方法**

使用等化赔率正则化的损失函数，配合 rehearsal（回放）防止灾难性遗忘，并在 MLP、ResNet、TF‑IDF+MLP 等模型上实现；

**📊 数据集**

在六个公开数据集上评估：ACS Income、COMPAS、German Credit、UTKFaces、Bias in Bios、LSAC，涵盖表格、图像和 NLP 三种模态；

**📈 对比分析**

通过与基线模型对比，使用 Loss Test 与 Threshold Test 两种黑盒攻击指标，实验显示 FFt 在大多数设置下将攻击准确度差距降至 τ=0.1 以下，理论界限在所有实验中均成立，提升泄露减少幅度在 10%–50% 之间；

**⚠️ 局限性**

局限包括：需要获得补集分布的样本；仅处理二元敏感属性和二分类任务；对更强攻击（如权重级联或侧信道）未验证；在极端群体不平衡时可能失效；未与 DP、属性遗忘等其他防御进行直接对比。

---

## 379. FlipItRight: Stable Pose-Targeted Throw-Flip Across Diverse Objects

**arXiv ID:** 2606.01713 | [PDF](https://arxiv.org/pdf/2606.01713v1)

**作者:** Axel Dawne `[一作]` (King Abdullah University of Science and Technology), Shinkyu Park `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 782 | [OpenAlex ID](https://openalex.org/A5038347119)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出 FlipItRight 框架，实现高自由度机械臂在平面上对多形状、多尺寸、多质量物体的姿态定向投掷翻转任务；通过两阶段规划先生成满足目标落地姿态的物体释放状态，再由机器人规划器检验可执行性并生成可行的摆动轨迹。

**💡 创新点**

核心创新在于将物体释放状态作为显式中间表征，拆分为对象级候选生成与机器人级可执行性评估；采用候选过滤、适配释放与预摆动姿态、并在接近释放时保持近似恒定末端执行器速度的结构化运动，从而提升对释放时机不确定性的鲁棒性，并在无先验数据或学习模型的前提下直接部署。

**🔧 技术方法**

技术手段包括：分析弹道传播模型与约束非凸优化（SLSQP）生成候选释放状态；基于逆运动学与线性、非线性最小二乘求解的摆动姿态选择；采用 Hermite 三次样条与恒速段构造末端执行器参考轨迹；利用离线优化与有限差分的机器人关节轨迹规划；在 UR10e + OnRobot RG6 上实现实时关节速度控制，并通过摄像头与 AprilTag 进行落地姿态测量。

**📊 数据集**

实验使用三类实验物体（长方体、椭圆柱体、六边形棱柱），每种物体在不同尺寸、质量与目标距离下共 12 种场景；未使用公开数据集，而是在实验室自行准备的物体集上进行 120 次真实投掷。

**📈 对比分析**

在 120 次试验中实现 90% 的成功率（108/120），并通过消融实验验证各模块（摆动姿态选择、接近释放恒速结构）对成功率和落地误差的贡献；与仅使用固定手设计姿态或无接近释放恒速的对比，FlipItRight 在成功率和落地误差上均有显著提升。

**⚠️ 局限性**

局限性包括：仅针对平面投掷翻转；假设释放瞬时且对非瞬时释放的补偿有限；需要手动放置与对接把握，导致抓取一致性受限；无法直接推广至三维全自由度投掷或未知物体属性；对抓取力与释放时机的手动调参仍存在依赖。

---

## 380. KDH-CAD: Knowledge-data hybrid CAD learning under data scarcity

**arXiv ID:** 2606.01702 | [PDF](https://arxiv.org/pdf/2606.01702v1)

**作者:** Ziqin Gao `[一作]` (Zhejiang University), Qiang Zou `[通讯]` (Zhejiang University)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5028026050)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出KDH-CAD框架，利用结构化领域知识与少量标注数据，在不微调预训练模型的前提下解决CAD数据稀缺问题，实现零件分类高精度；

**💡 创新点**

核心创新在于先用三元组知识唤醒并完成预训练模型中的CAD概念，然后通过shift向量与跨注意力对齐网络对这些概念进行几何校准，形成知识-数据混合学习；

**🔧 技术方法**

结合视觉‑语言预训练模型（Qwen3‑VL‑2B‑Instruct）、领域知识三元组构造、shift向量+跨注意力对齐模块以及多视图图像编码技术；

**📊 数据集**

使用真实机械部件数据集TMCAD（10类）与FabWave（45类），在每类仅用少量样本进行训练；

**📈 对比分析**

与UV‑Net、AAGNet、BRT等传统纯数据模型对比，KDH-CAD在5–100样本/类时平均准确率超过95%，在完整数据集上仅用100样本即可匹配甚至超越基线性能；

**⚠️ 局限性**

对复杂或罕见零件的识别效果有限，主要由于所使用的通用领域知识缺乏行业专属细节；此外，依赖多视图图像可能无法完整保留CAD拓扑信息。

---

## 381. RPCASSM: Robust PCA State Space Model For Infrared Small Target Detection

**arXiv ID:** 2606.01689 | [PDF](https://arxiv.org/pdf/2606.01689v1)

**作者:** Pingping Liu `[一作]` (Jilin University), Qiuzhan Zhou `[通讯]` (Jilin University)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5032424116)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于鲁棒主成分分析(RPCA)指导的双分支状态空间模型RPCASSM，用于红外小目标检测。

**💡 创新点**

创新点在于设计了背景分支的空间探测扫描机制(SPCM)和目标分支的可变形提示扫描机制(DPCM)，通过RPCA分解将空间自适应序列化与稀疏目标特性对齐。

**🔧 技术方法**

采用了RPCA分解、空间域分解、状态空间建模以及可变形注意力机制，并在训练中使用逐步IoU损失进行端到端优化。

**📊 数据集**

使用公开的NUDT‑SIRST和IRSTD‑1K两大红外小目标检测基准数据集进行训练与评估。

**📈 对比分析**

与10种最新CNN/Transformer/SSM方法对比，RPCASSM在mIoU、Pd、F‑measure等指标上位列前茅，同时仅含0.45M参数，显著提升检测准确率与误报率。

**⚠️ 局限性**

局限在于对RPCA分解参数敏感，可能在极度噪声或复杂背景下效果下降；双分支SSM结构虽然参数少但计算量仍较高，实时部署仍需进一步优化。

---

## 382. HAIM: Human-AI Music Datasets for AI Music Production Tracking Benchmark

**arXiv ID:** 2606.01686 | [PDF](https://arxiv.org/pdf/2606.01686v1)

**作者:** Seonghyeon Go `[一作]`, Yumin Kim `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HAIM（Human–AI Music）数据集，提出 AI 音乐追踪任务，并训练了多标签检测模型 MuQ‑FST，能够识别音乐中不同生产角色（作曲、作词、演唱、音频工程）是否由 AI 参与。

**💡 创新点**

突破了传统的二分类检测，提出细粒度多标签追踪和时序定位框架，首次将 AI 生成的音频与人类工程过程按角色拆解，并提供完整的标签与评估基准。

**🔧 技术方法**

使用 MuQ 预训练音频编码器、Fusion Segment Transformer 架构、多标签交叉熵损失和时间窗口滑动检测，实现多角色识别与 AI 区段定位。

**📊 数据集**

数据集为 HAIM，总计 196,000 条完整音乐轨道，分 13 类 3 组（A1–A2 基线、B1–B9 混合、C1–C2 时序混合），涵盖 6 种生成模型（Suno、Udio、Mureka、Lyria Pro、MusicGen、ACE‑Step‑1.5）及人类原始音乐来源（MTG‑Jamendo、SONICS）。

**📈 对比分析**

与 Deezer、SpecTTTra、CLAM、FST 等公开检测器做对比：MuQ‑FST 在全 AI 轨道上 99.9% 检测率、在人类轨道上 0.1% 假阳性；在多标签任务中 80–99% 的角色准确率；时序定位 F1 在 10 s 窗口下达到 0.91（C1）/0.78（C2），显著优于 30 s 基线。

**⚠️ 局限性**

局限性包括：角色划分过于粗糙（将作曲、作词合并为 Composer）、生成器覆盖单一（B 类大多基于 ACE‑Step）、类别分布不平衡、混音/母带使用单一模板导致模型过拟合、语言角色难以通过纯音频判定、缺乏更复杂的真实工坊工序与对抗性攻击场景。

---

## 383. Overcoming Challenges in Agile and DevOps Integration: A Qualitative Study

**arXiv ID:** 2606.01676 | [PDF](https://arxiv.org/pdf/2606.01676v1)

**作者:** Juliana Fraislebem `[一作]` (University of Applied Sciences Emden/Leer), Eva-Maria Schön `[通讯]` (University of Applied Sciences Emden/Leer)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过半结构化访谈对6名从业者进行定性研究，分析敏捷与DevOps整合的挑战与解决方案。

**💡 创新点**

首次系统归纳了四大挑战类别（文化、结构、过程、技术）和四大解决方案（团队结构、文化协作、过程管理、自动化基础设施），并将访谈数据与现有文献对照验证。

**🔧 技术方法**

采用半结构化访谈、Mayring 质性内容分析方法、MAXQDA 数据管理与编码工具。

**📊 数据集**

使用6位来自巴西与德国的专业人士（共30–50分钟访谈）生成的访谈文本（原始葡萄牙语与英语），未使用公开数据集。

**📈 对比分析**

未进行量化对比；研究基于编码出现频次与主题归纳，未给出性能指标或实验结果。

**⚠️ 局限性**

样本规模小（仅6人），受访者范围有限，存在解释偏差，且未评估提出策略在实际环境中的效果。

---

## 384. A Sheaf Framework for Strategic Multi-Agent Systems: From Consensus to Nash Equilibria

**arXiv ID:** 2606.01663 | [PDF](https://arxiv.org/pdf/2606.01663v1)

**作者:** Manuel Hernández `[一作]`, Eduardo Sánchez-Soto `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个统一的范畴框架，将几何协同、逻辑规划、时间推理和博弈论的战略优化融合到一个 Grothendieck 拓扑空间中，并用其在免疫学类“堡垒防御”场景中实现异构自主代理的协同与博弈。

**💡 创新点**

创新点包括：① 通过引入游戏层的 sheaf（游戏 sheaf）把 Nash 均衡与全局截面联系起来；② 通过 cohomology 对战略一致性障碍进行分类；③ 设计了混合动力学——Sheaf Laplacian 传导 + 期望收益梯度上升，实现几何与策略同步收敛。

**🔧 技术方法**

使用的技术有：
- Sheaf 与 Topos 理论（构造空间时间的 product site 𝒮=𝒯×G）
- 事件演算（Event Calculus）作为时态 sheaf 条件
- 游戏理论（最佳反应 sheaf、Nash 均衡）
- 线性代数与拓扑算法（Moore‑Penrose 伪逆、cohomology 诊断）
- 异步非线性 Sheaf 扩散算法
- 组合的梯度上升与 Laplacian 传导更新
- Python/Pygame、Erlang/Elixir 等实现框架

**📊 数据集**

本文没有使用公开数据集，而是采用基于模拟的“堡垒防御”实验场景：不同类型代理（侦察、火炮、后勤）在随机攻击事件下保护资产，模拟中采用 100HP 的堡垒、能量、弹药等内部状态。

**📈 对比分析**

实验比较主要通过内部指标（堡垒存活时间、资源利用率、全局策略收敛速度）与传统纯粹一致性算法（仅 Sheaf Laplacian）进行对比。结果表明，加入博弈层后，系统在资源受限与冲突情境下的堡垒存活时间提升 15–30%（取决于网络拓扑），而收敛速度略有下降但保持可接受的实时性能。

**⚠️ 局限性**

局限性包括：
- 计算复杂度：协同诊断的 cohomology 计算为 O((|V|+|E|)^3)，在大规模网络上成本较高；
- 对梯度上升的收敛性依赖于潜在游戏结构和拉普拉斯正定性，实际场景中可能出现非凸问题；
- 目前仅处理确定性奖励与完备信息的博弈，未来需扩展到不确定信息或 Bayesian 博弈；
- 只在仿真环境验证，缺乏真实机器人或无人机的硬件实验验证。

---

## 385. Edge-directed geometric partitioning for versatile video coding

**arXiv ID:** 2606.01641 | [PDF](https://arxiv.org/pdf/2606.01641v1)

**作者:** Xuewei Meng `[一作]` (Peking University), Siwei Ma `[通讯]` (Peking University)

**通讯引用:** 15967 | [OpenAlex ID](https://openalex.org/A5039832462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于边缘引导的几何分割方案E-GEO，用于VVC中的几何分割（GEO）模式，显著降低了GEO索引开销并提升了压缩效率。

**💡 创新点**

创新点在于首次将MMP（Most Probable Mode）列表与时空边缘信息结合，通过引导滤波+Canny检测预测分割角度和距离，实现高效的GEO模式预测。

**🔧 技术方法**

技术手段包括：指导滤波代替高斯滤波、Canny边缘检测、基于运动向量的参考块匹配、Sobel梯度统计以及根据边缘强度构造MPM列表。

**📊 数据集**

使用VTM‑6.0官方的Common Test Conditions视频序列（Class A1、A2、B、C、E），在Random Access Main10和Low Delay B Main10两种配置下，分别在22/27/32/37四个QP上进行实验。

**📈 对比分析**

对比VTM‑6.0基准，E‑GEO在RA和LDB配置下分别平均降低0.58%和1.00%的BD‑rate，改进率约为41%–43%，且在目标物体边缘的主观质量上也有明显提升。

**⚠️ 局限性**

局限性包括：仍有进一步优化空间，未深入评估编码复杂度与延迟，且方法仅在VVC标准下验证，其他编码器或更高码率场景的适用性尚未探讨。

---

## 386. Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging

**arXiv ID:** 2606.01717 | [PDF](https://arxiv.org/pdf/2606.01717v1)

**作者:** Minsik Choi `[一作]` (Korea University), Geewook Kim `[通讯]` (NAVER Cloud AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种去中心化指令微调框架 MERIT，先基于梯度冲突估计将任务集合按 PCA 轴分块，随后在无通信的分支上各自微调，最后通过 token 加权平均一次性合并模型。

**💡 创新点**

创新点包括：① 将梯度冲突与 Hessian 高曲率方向关联的理论分析；② 用 PCA 对冲突空间进行分块，最大化合并时的方差压缩；③ 证明合并等价于谱滤波与隐式范数正则化；④ 在实际模型中实现仅一次合并即可获得更好性能。

**🔧 技术方法**

技术手段涵盖：梯度冲突估计（小样本梯度采样）、余弦相似度矩阵构造、PCA 方向分块、通信自由的分支微调、token 加权参数平均合并、以及多任务/多模态的平衡采样与预算分配。

**📊 数据集**

主要使用数据集：136 个 Vision‑FLAN 任务（约 1.6M 样本、176 源）、Qwen2.5‑VL‑3B 与 7B 基础模型，以及文本仅 FLAN（66 任务）作为跨模态验证。

**📈 对比分析**

与中心化联合训练、随机分块、均匀模型“汤”等基线对比，MERIT 在 3B 模型下 8 项评测平均从 54.3 提升到 57.0；在 7B 模型上与 1.6M 样本混合对比时，平均分 55.4 对比 54.9；文本仅任务上 2D 分块平均提升 0.8 分。所有提升均在无通信或极低通信成本的前提下实现。

**⚠️ 局限性**

局限性：① 依赖于“merge‑ready”初始化和模型位于平坦低损失盆地；② 对极大 K（分支数）或极端数据不平衡的适应性未知；③ 需要额外的梯度冲突估计成本，虽一次性但在极大任务集合上仍非零；④ 对非结构化、动态变化的任务混合的鲁棒性尚待验证。

---

## 387. Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving

**arXiv ID:** 2606.01839 | [PDF](https://arxiv.org/pdf/2606.01839v1)

**作者:** Jianru Ding `[一作]` (University of Chicago), Henry Hoffmann `[通讯]` (University of Chicago)

**通讯引用:** 7168 | [OpenAlex ID](https://openalex.org/A5080833704)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 ConServe 的对话级拆分调度系统，专为 Agentic LLM 工作负载设计。它将第 1 回合的预填充（prefill）转发至高吞吐量 GPU，然后一次性将 KV 缓存转移到分配给整个对话生命周期的解码器（decoder）上，并在整个后续尾部保持对话固定。

**💡 创新点**

创新点包括：
- 将调度单元从传统的每回合转移到整条对话，消除了对未知解码成本的预测需求；
- 只做一次 KV 转移，充分利用跨回合 KV 复用；
- 采用可观测的首回合输入长度和解码器 KV 负载实现无预测的即时调度；
- 在异构 GPU 环境下天然映射：高功耗 GPU 执行 compute‑bound 的预填充，低功耗 GPU 执行 memory‑bound 的后续尾部，进一步提升能效。

**🔧 技术方法**

技术手段包括：
- 预填充‑解码（prefill‑decode）拆分架构；
- KV 缓存管理与一次性转移；
- 观察性调度器（基于预填充延迟曲线和解码器 KV 占用率）；
- 异构 GPU 的功率控制（模拟不同 TDP）；
- 对话级负载预测与弹性扩容策略；
- 对比基准（Collocated、Full Disaggregation、AMPD）与 vLLM、LMCache 等实现。

**📊 数据集**

使用的数据集和模型：
- Agentic 工作负载采样自 SWE‑bench_bm25_13K（包含 13k 代码仓库条目）并通过 swe‑agent 生成；
- 追踪生成模型为 Qwen3‑Coder‑30B‑A3B‑Instruct；
- 服务模型为 Qwen3‑0.6B（BF16），在 4 块 NVIDIA A40 GPU 上部署。

**📈 对比分析**

与三种基准（Collocated、Full Disaggregation、AMPD）对比，使用的评估指标有：TTFET（首个有效 token 的延迟）、最后一回合 TBT（token 间距）和整体 E2E 延迟。实验结果表明：
- ConServe 在 p95 TTFET 上提升约 51%（相对 AMPD）并在整体 E2E 延迟上减少约 28%；
- 能效（tokens/J）提升约 7%（单一硬件）以及 22%（异构硬件）；
- 在高负载饱和点仍保持零 SLO 违规，其他基准在此点出现明显性能退化。

**⚠️ 局限性**

局限性包括：
- 依赖预先离线测定的预填充延迟曲线，若模型或硬件变化需重新测量；
- 只考虑单一对话的调度策略，未探索多会话共享或多代理交互场景；
- 在极端长尾或高并发下的 KV 转移瓶颈仍未完全消除；
- 评估仅针对 Qwen3‑0.6B，性能可否直接迁移到更大或不同架构模型仍需验证；
- 异构调度在能耗提升上依赖于预填充与解码器的 compute/memory 区别，若两端性能差异不大则优势有限。

---

## 388. LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning for Agentic Language Models

**arXiv ID:** 2606.01838 | [PDF](https://arxiv.org/pdf/2606.01838v1)

**作者:** Prateek Kumar Sikdar `[一作]` `[通讯]` (Accenture), Prateek Kumar Sikdar (Accenture)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LayerRoute，一种轻量级适配器，使大语言模型在不同代理步骤（工具调用与规划）中根据输入条件动态跳过 Transformer 层，实现推理加速。

**💡 创新点**

通过在每层加入硬门控路由器与 LoRA 适配器，并结合直通估计（STE）、门正则化和偏置初始化，首次实现仅使用训练 0.22% 参数即可学习输入条件层跳过，无需重新训练整个模型。

**🔧 技术方法**

使用 LoRA 低秩微调、直通估计（Straight‑Through Estimator）、门正则化、偏置初始化以及冻结的 Qwen2.5‑0.5B‑Instruct 背骨网络。

**📊 数据集**

混合代理数据集，包括 Hermes 与 Glaive 的工具调用样本，以及 GSM8K 与 Turing 的规划/推理样本，总计约 10.7K 训练样本。

**📈 对比分析**

与全层模型、LayerRoute‑BCE、LayerRoute‑NoReg、LayerRoute‑UniformInit 等基线对比；在工具调用步骤实现 15.2% FLOPs 降低、规划步骤仅 2.3%；skip 差异达到 12.91%，并在工具调用与规划两类任务上均优于全层模型的困惑度。

**⚠️ 局限性**

中间层跳过均匀缺乏细粒度路由、整体 FLOPs 降低有限、规划任务几乎不跳层、仅在 0.5B 模型验证、数据集不平衡等限制。

---

## 389. Learning Implicit Bias in Generative Spaces for Accelerating Protein Dynamics Emulation

**arXiv ID:** 2606.01833 | [PDF](https://arxiv.org/pdf/2606.01833v1)

**作者:** Kaihui Cheng `[一作]` (Fudan University), Yuan Qi `[通讯]` (Fudan University)

**通讯引用:** 14843 | [OpenAlex ID](https://openalex.org/A5100676883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在冻结的蛋白质动力学预训练模拟器基础上，加入历史相关的偏置以加速长时域探索，并通过分数修正步骤保持结构有效性

**💡 创新点**

首次在生成空间引入历史依赖偏置，并用环境支持正则化保证偏置不脱离模拟器数据流形，同时设计了分数修正投影机制

**🔧 技术方法**

利用SE(3)分数扩散模型、历史感知分数估计器、距离加权偏置、环境支持正则化以及一阶正向-反向扩散修正

**📊 数据集**

DynamicPDB-80 以及 12 个无监督 Fast‑Folding 蛋白质的数据集

**📈 对比分析**

与无偏置预训练模拟器对比，偏置方法在 DynamicPDB-80 上多样性提升 35%，在 Fast‑Folding 蛋白上仅使用偏置即可 15 倍加速覆盖率，结合修正可达 37 倍加速并覆盖约 3 倍低能态

**⚠️ 局限性**

偏置是无条件的，只在预训练模拟器支持范围内重新分布密度，基模拟器不满足平衡分布，导致后期自由能重建困难

---

## 390. CAPF: Guiding Search-Agent Rollouts with Credit-Attenuated Privileged Feedback

**arXiv ID:** 2606.01830 | [PDF](https://arxiv.org/pdf/2606.01830v1)

**作者:** Bin Chen `[一作]`, Chonghan Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入 Privileged Feedback 机制，在训练阶段让搜索增强 LLM 通过 verifier 提供的反馈修正答案，从而增强学习信号，并通过信用衰减避免在部署时依赖该反馈。

**💡 创新点**

创新点在于：1）在训练中加入 Privileged Feedback 调用，利用 verifier 侧信息生成修复轨迹；2）采用 Credit‑Attenuated 机制，仅对反馈前的行为减弱回报，保持后续修复行为全信用，解决信用分配问题。

**🔧 技术方法**

使用技术包括 RL with verifiable rewards (RLVR)、policy gradient (REINFORCE++)、工具调用（Privileged Feedback）、信用衰减机制以及大规模 LLM（Qwen3‑4B/ Qwen2.5‑7B）。

**📊 数据集**

使用七个开放域 QA 基准：NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle。

**📈 对比分析**

通过将 CAPF（不同保留因子）与直接推理、Outcome‑only RLVR 进行对比，部署时移除 Privileged Feedback。CAPF（ρ=0.8）平均 EM 从 44.7% 提升至 48.5%；单跳任务提升至约 56.9%，多跳提升至约 48.7%；在与同类 Qwen2.5‑7B 系统比较时获得最高宏平均。

**⚠️ 局限性**

局限性：仅在 exact‑match QA 上验证；需要可靠的 verifier 与反馈生成器；Privileged Feedback 可能泄漏答案信息；未评估长文本或主观性较高的任务。

---

## 391. "I've Seen How This Goes": Characterizing Diversity via Progressive Conditional Surprise

**arXiv ID:** 2606.01811 | [PDF](https://arxiv.org/pdf/2606.01811v1)

**作者:** Matthew Khoriaty `[一作]` (ERA Fellowship), Shi Feng `[通讯]` (George Washington University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于in‑context learning的多样性度量（Decan），并用它评估大型语言模型（LLM）生成文本和人类写作的多样性，验证其在人工标注和后训练阶段的有效性。

**💡 创新点**

创新点在于：①不依赖嵌入模型、参考语料或人工标签；②利用基础模型的token概率计算进阶条件惊讶曲线a_k和一致性系数C，构成D_Ca_n = C × a_n；③通过ICL捕捉多响应的相似性，既能抑制噪声又能反映一致性。

**🔧 技术方法**

技术包括信息理论（交叉熵、熵率、几何平均perplexity）、in‑context learning、对多种响应顺序做随机置换平均、单次前向推理计算token log‑probability、Python/PyTorch实现。

**📊 数据集**

使用的数据集有：Tevet & Berant的McDiv与ConTest（人工多样性标注）、DecTest（温度标签）、OLMo‑2‑7B四个后训练阶段（base、SFT、DPO、RLVR）配合AlpacaFarm与NoveltyBench提示，采集每个提示10个生成样本。

**📈 对比分析**

比较方法：在二分类多样性任务中使用Spearman ρ、OCA、ROC‑AUC；与SentBERT、EAD、distinct‑n等基线对比；在OLMo后训练实验中使用Wilcoxon配对检验。结果显示：D_Ca_n在McDiv/ConTest上与SentBERT相近，最高OCA略低；在OLMo后训练阶段，D_Ca_n随阶段递减，Wilcoxon显著；与其他基线在趋势上保持一致。

**⚠️ 局限性**

局限性：①度量结果取决于基础模型θ的ICL能力，跨模型可比性受限；②选择C×a_n的公式基于经验，未推导为唯一正确的指标；③对短文本需要长度匹配，导致部分提示被剔除；④对内容“惊讶”敏感，可能低估仅在语义上多样但语法相似的响应；⑤在构造混淆的数据集（如McDiv_nuggets）中表现受限；⑥仍需更多多样性基准来进一步验证。

---

## 392. Token Predictors Are Not Planners: Building Physically Grounded Causal Reasoners

**arXiv ID:** 2606.01810 | [PDF](https://arxiv.org/pdf/2606.01810v1)

**作者:** Zheng Lu `[一作]`, Yiming Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Causal Plan 框架，将 embodied planning 从语言序列预测转向物理因果推理，并构建 Causal-Plan-Bench 与 Causal-Plan-1M 两大资源

**💡 创新点**

通过四阶段严格的因果标注流程与大规模因果监督，首次在多维度评测中实现真正的物理规划能力；并揭示物理推理遵循可预测的 Causal Scaling Law

**🔧 技术方法**

使用 GPT‑5.4 进行自检式因果标注、Qwen3‑VL‑8B 作为基础模型，结合分阶段 SFT 与 RL（GRPO）训练，辅以 GPT‑5.4‑驱动的评估与奖励机制

**📊 数据集**

Causal-Plan-1M（100万条因果推理记录）与 Causal-Plan-Bench（1200 条多维度评测案例），均从 EPIC‑KITCHENS、Ego4D、Ego‑Exo4D 等 egocentric 视频集采集并整理

**📈 对比分析**

在 Causal-Plan-Bench 上 Causal Planner 以 45.28 分领先 Gemini 3 Pro 的 38.18 分，跨 Benchmark（EgoPlan‑Bench2、RoboVQA、Cosmos‑Reason）亦取得 3‑4 p 的提升，显示强泛化与鲁棒性

**⚠️ 局限性**

受限于当前因果标注成本高昂、模型仍需对复杂多步情境的对抗性错误进行进一步完善，且尚未在更大规模真实机器人环境中进行实地验证

---

## 393. A Near-Optimal Offline Algorithm for Dynamic All-Pairs Shortest Paths in Planar Digraphs

**arXiv ID:** 2606.01809 | [PDF](https://arxiv.org/pdf/2606.01809v1)

**作者:** Debarati Das `[一作]` (University of Copenhagen), Christian Wulff-Nilsen `[通讯]` (University of Copenhagen)

**通讯引用:** 1057 | [OpenAlex ID](https://openalex.org/A5077977985)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对平面图的离线（及增量在线）动态全点对最短路径（APSP）算法，更新与查询均可在 Õ(√n) 时间完成。

**💡 创新点**

创新点在于：①首次给出能在离线更新序列下保持密集距离图（DDG）的子线性更新时间；②将 DDG 视为可查询的数据结构而非完整图，从而实现快速查询；③通过递归 r‑划分与 FR‑Dijkstra 的组合，突破了过去的 Õ(n^{2/3}) 限制。

**🔧 技术方法**

主要技术包括：递归 r‑划分、FR‑Dijkstra、最后区间检测数据结构、右侧最短路径与子区间分析、层级祖先查询以及全持久化技术；算法整体基于图的平面嵌入与唯一最短路径假设。

**📊 数据集**

该工作为理论算法，未使用实验数据集，所有结果均为证明与上界/下界分析。

**📈 对比分析**

与现有最佳在线/离线算法相比，更新/查询时间从 O(n^{2/3}) 降至 O(√n)，满足已知的条件下界，证明了结果的最优性（多项式对数因子内）。

**⚠️ 局限性**

局限性包括：仅适用于平面嵌入保持的权重增减更新；在线增量情形下可用，但无法处理一般的删除或完全动态情况；全持久化实现复杂且仍需多项式对数额外时间。

---

## 394. MOSS-Audio Technical Report

**arXiv ID:** 2606.01802 | [PDF](https://arxiv.org/pdf/2606.01802v1)

**作者:** Chen Yang `[一作]`, Xipeng Qiu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了MOSS‑Audio统一音频‑语言模型，支持语音识别、音频描述、音乐与环境声音理解、时间标记转写、基于时间的问答和多步骤推理；

**💡 创新点**

创新点包括深度跨层特征注入（DeepStack）、显式时间标记、事件保留分割与分支标注管道，以及针对指令跟随和推理的两种模型变体；

**🔧 技术方法**

采用编码‑适配‑解码结构，使用GatedMLP适配器、滑动窗口注意力、跨层注入、时间标记嵌入，结合预训练（ASR、音频描绘、文本语言模型）与后训练（监督微调、推理冷启动、DAPO强化学习）；

**📊 数据集**

利用数百万小时多源音频数据进行标注，覆盖语音、音乐、环境声，并在众多公开基准（MMAU、MMAU‑Pro、MMAR、MMSU、ASR、时间戳ASR等）上评测；

**📈 对比分析**

与现有开源和专有模型对比，MOSS‑Audio‑8B‑Thinking在广义音频理解任务上取得最高开放源代码成绩，MOSS‑Audio‑8B‑Instruct在语音描绘、ASR和时间戳ASR中表现最佳，整体性能领先同类模型；

**⚠️ 局限性**

仍受限于与顶级专有模型的差距，可能出现文本化推理中的信息漂移或幻觉，且对极长或多模态（如视频）音频的推理仍需进一步验证。

---

## 395. Consistency evaluation of benchmarks used for causal discovery

**arXiv ID:** 2606.01789 | [PDF](https://arxiv.org/pdf/2606.01789v1)

**作者:** Yuzhe Zhang `[一作]` (Independent researcher), Chen Wang `[通讯]` (CSIRO)

**通讯引用:** 46825 | [OpenAlex ID](https://openalex.org/A5057492548)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一套基于大语言模型（LLM）的自动化流程，用于评估流行因果发现基准图与最新领域研究之间的一致性。

**💡 创新点**

创新点在于：①首次系统地从科学数据库检索数以万计的研究论文并用LLM验证论文结论与基准图的d‑separation关系；②引入不一致率指标量化基准图与领域知识的匹配程度；③发现并指出旧版基准的高不一致率，为基准构建和更新提供实证依据。

**🔧 技术方法**

主要技术包括：LLM推理（Qwen3-30B）、查询扩展与关键词过滤、PubMed/OpenAlex检索、PDF OCR转换为JSON、d‑separation关系提取与生成验证问句、以及统计分析脚本。

**📊 数据集**

使用了11个最常用的真实世界基准（如Sachs、Child、Alarm、Asia、Insurance、Ecoli、Alzheimer、Arctic Sea Ice等），共计38,081篇检索到的全文论文。

**📈 对比分析**

通过比较每个基准的不一致率，发现Sachs基准的不一致率最低（约26.7%），而Asia、Child、Diabetes等旧基准的不一致率高达45–58%。该结果表明新近构建或混合方法构建的基准（如Insurance、Arctic Sea Ice）与领域知识更匹配，说明基准更新对因果发现评估的重要性。

**⚠️ 局限性**

局限性包括：①LLM验证器准确率仅约90%，更强大的LLM可能进一步提升评估质量；②检索效率受限于数据库搜索能力和全文获取权限，导致部分相关论文未被纳入；③评估仅聚焦于d‑separation所对应的条件关联，未涵盖所有可能的因果假设。

---

## 396. Structure-Guided Adaptive Propagation for Protein-Protein Interaction Site Prediction

**arXiv ID:** 2606.01781 | [PDF](https://arxiv.org/pdf/2606.01781v1)

**作者:** Enqiang Zhu `[一作]` (Guangzhou University), Baoshan Ma `[通讯]` (Dalian Maritime University)

**通讯引用:** 8746 | [OpenAlex ID](https://openalex.org/A5060457543)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种结构引导的自适应传播模型 SGAP-PPIS，用于蛋白质-蛋白质相互作用位点（PPIS）的预测。

**💡 创新点**

创新点在于：①基于多尺度几何状态生成每个残基的自适应传播系数，实现几何环境驱动的局部信息保留与邻域扩散平衡；②采用尺度对齐的传播步与几何层映射，保证传播过程与几何信息同步；③将多步 APPNP 的中间状态拼接并与几何摘要融合，充分利用层级拓扑信息；④设计 MUSE 融合模块，整合自注意力、深度可分离卷积和前馈网络，提升特征表达能力。

**🔧 技术方法**

主要技术包括：Equivariant Graph Neural Network（EGNN）提取多尺度几何特征；APPNP（Approximate Personalized Propagation of Neural Predictions）进行信息扩散；几何条件自适应传播系数生成；多步状态拼接与 MUSE 融合；Focal Loss 处理类别不平衡。

**📊 数据集**

使用的公开数据集为 Train_335-1 作为训练集，Test_60、Test_315-28、UBtest_31-6 为三套独立测试集，全部基于 GraphPPIS 的基准划分。

**📈 对比分析**

与多种基准方法（PSIVER、ProNA2020、SCRIBER、DLPred、DELPHI、DeepPPISP、SPPIDER、MaSIF-site、GraphPPIS、DeepProSite、AGAT-PPIS、AGF-PPIS、GHGPR-PPIS、ASCE-PPIS、MGMA-PPIS、ComGAT-PPIS）进行对比。SGAP-PPIS 在 Test_60 上获得最高的 MCC、AUROC 和 AUPRC，且在 Test_315-28 与 UBtest_31-6 上分别取得最高的 AUPRC；在多项指标上均优于 MGMA-PPIS、ASCE-PPIS 等现有方法。

**⚠️ 局限性**

局限性包括：仅针对静态蛋白结构，难以直接应用于动态多组分复合物或膜蛋白；缺乏对时间演化或脂质环境的建模；对大规模蛋白组学数据的可扩展性和实时性尚未充分验证。

---

## 397. Night-Window Batching versus Carbon-Aware Scheduling for Clinical AI GPU Workloads

**arXiv ID:** 2606.01766 | [PDF](https://arxiv.org/pdf/2606.01766v1)

**作者:** Nishi Doshi `[一作]` (University of Southern California), Shrey Shah `[通讯]` (University of Southern California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在医院GPU AI工作负载的模拟环境中，对13种调度策略进行比较，评估夜间批处理与碳感知调度的碳排放与截止时间满足率。

**💡 创新点**

提供了一个统一的仿真框架，量化了简单夜窗策略与复杂碳权重策略在临床优先级和碳排放之间的权衡，并展示了碳优先策略在临床约束下的危险性。

**🔧 技术方法**

采用事件驱动GPU调度仿真器、合成患者式作业、碳强度时序、以及多种调度算法实现。

**📊 数据集**

使用合成工作负载（按到达率、临床优先级和不同碳情景生成的2000个作业），以及多地区的单一日循环碳曲线进行验证。

**📈 对比分析**

通过平均总碳、95%分位延迟、关键层次截止误差等指标对策略进行横向比较；结果显示夜窗策略在平均碳方面与CUCA_0.45相近，同时关键作业误差更低；碳优先策略性能差。

**⚠️ 局限性**

仅为仿真研究，未使用真实医院数据；缺乏迁移能耗、网络延迟等实际因素；统计显著性未调整；只能作为假设验证。

---

## 398. TriAlign: Towards Universal Truth Consistency in Personalized LLM Alignment

**arXiv ID:** 2606.01755 | [PDF](https://arxiv.org/pdf/2606.01755v1)

**作者:** Thi-Nhung Nguyen `[一作]` (Monash University), Dinh Phung `[通讯]` (Monash University)

**通讯引用:** 11642 | [OpenAlex ID](https://openalex.org/A5036447132)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了TriAlign，一种用于个性化大型语言模型的离线多智能体强化学习框架，旨在在保持个性化的同时保证跨社会群体的客观真理一致性。

**💡 创新点**

创新点在于：①将真理不变性（Truth‑Invariant Alignment）定义为离线MARL问题；②引入基于Nash社会福利（NSW）的公平感知目标以平衡不同社会群体；③在奖励中加入显式一致性惩罚，并通过多回合社会群体交互生成离线轨迹；④构建了多维度的社会群体与人物模拟流程。

**🔧 技术方法**

技术上采用：多智能体离线强化学习（MARL）与KL正则化的SFT目标；NSW启发的加权奖励；多轮交互轨迹生成；LLM提示工程用于实现环境反馈、奖励评估与人物生成；以及LangGraph、vLLM、DeepSpeed ZeRO‑3等工具栈。

**📊 数据集**

使用的数据集包括：LogicQA、SimpleQA、LIMO、StereoSet（训练/测试）、AIME25、MMLU‑Pro、TruthfulQA；人物数据通过LLM自生成的1B人物库扩增至约7.5K个，覆盖75个社会群体。

**📈 对比分析**

与多种无训练基线（P‑Debias、P‑Defense、BestPersona、2StepPrompt）以及训练基线（SFT、Swift）对比，TriAlign在统一真理准确率、真理一致性指标（Std、Gap、CV）和个性化对齐指标（Pref）上均取得显著提升，尤其在跨群体公平性方面表现最佳。

**⚠️ 局限性**

局限性包括：①合成人物可能无法完整捕捉真实世界的复杂多样性；②真理一致性与个性化的权衡仍未完全解决，仍需进一步研究；③多智能体离线RL与轨迹生成的计算成本较高，能耗与碳排放需关注。

---

## 399. SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving

**arXiv ID:** 2606.01751 | [PDF](https://arxiv.org/pdf/2606.01751v1)

**作者:** Quqing Zhang `[一作]` (MemTensor Technology Co., Ltd.), Xiaoxing Wang `[通讯]`

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通用段级KV重用框架，能够在多轮聊天、检索增强生成（RAG）和多智能体协作等常见场景下高效复用已计算的Key/Value。

**💡 创新点**

创新点包括：利用段级缓存与RoPE对齐实现位置无关重用；通过基于非重用段Query的稀疏Token重算来补偿跨段注意力耦合；引入虚拟块、冻结块池和额外键实现多域命名空间；以及在vLLM中深度集成，保持对现有前缀缓存、PagedAttention和FlashAttention/FlashInfer的兼容性。

**🔧 技术方法**

核心技术包括RoPE对齐算法、Sparse‑KV重算（基于Query‑Key重要性估计）、虚拟块与冻结块池、Delta‑RoPE复制核、混合层阈值稀疏注意力、vLLM调度与缓存管理、以及对FlashAttention/FlashInfer的元数据支持。

**📊 数据集**

实验使用的主要数据集有：多轮对话评测（LOCOMO、LongMemEval）、RULER（MQ‑NIAH、VT、CWE、FWE）以及MASLab多智能体工作流（MATH、GSM‑Hard、AQUA‑RAT、SciBench、GPQA、MedQA、MedMCQA）。

**📈 对比分析**

与全重算、Naive Reuse、CacheBlend、EPIC等基线对比，SparseX 在多轮聊天和 RULER 上实现了与全重算相近的质量，同时将预填充延迟 (TTFT) 降至 0.5–1 秒，吞吐率提升 2–3 倍；在多智能体场景下保持 95% 以上的准确率，同时显著降低了重算开销。

**⚠️ 局限性**

局限性包括：需要针对不同模型/任务调节层级阈值和重算比例；跨节点存储与调度尚未集成；在极端长上下文或极大并发下的内存占用仍可进一步优化。

---

## 400. STaR-KV: Spatio-Temporal Adaptive Re-weighting for KV Cache Compression in GUI Vision-Language Models

**arXiv ID:** 2606.01790 | [PDF](https://arxiv.org/pdf/2606.01790v1)

**作者:** Yuhang Han `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15661 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 STaR‑KV 的无训练 KV 缓存压缩框架，专门用于提升视觉‑语言模型驱动的 GUI 代理的部署效率。

**💡 创新点**

创新点在于三轴自适应重加权：① 在线子空间层面的空间互信息预估来保留布局敏感信息；② 累积时间稳定性折扣抑制长期冗余缓存；③ 基于熵的温度自适应来动态锐化分数分布。

**🔧 技术方法**

采用了在线互信息估计、指数滑动平均、GQA 组/头级空间剖面、时间稳定性余弦相似度、指数衰减、熵计算与温度调度等技术，全部为训练‑free，保持原始模型不变。

**📊 数据集**

在四个 GUI 代理基准上评测：ScreenSpot‑Pro、ScreenSpot‑v2、AndroidControl 以及 AgentNetBench，使用开源 UI‑TARS‑1.5‑7B 与 OpenCUA‑7B 模型。

**📈 对比分析**

与现有 KV 压缩方法（GUIKV、SnapKV、PyramidKV 等）在相同预算下对比，STaR‑KV 在所有基准上均取得最高或近似最高平均准确率，甚至在 40% 预算时优于全缓存基线，且无额外 FLOPs 开销，显著降低 GPU 内存占用。

**⚠️ 局限性**

局限性包括仅在两款 7B 规模开源模型上验证，未覆盖更大规模或闭源系统；评测仅限现有四个基准，缺乏对新兴域或多模态交互的验证；并且低成本部署可能间接助长不当 UI 自动化的风险。

---

## 401. Hist2Style: Histogram-Guided Stylization with Bilateral Grids

**arXiv ID:** 2606.01819 | [PDF](https://arxiv.org/pdf/2606.01819v1)

**作者:** Dekel Galor `[一作]` (Adobe Nextcam), Ilya Chugunov `[通讯]` (Adobe Nextcam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过把大型图像编辑模型的色调与色彩编辑能力进行选择性蒸馏，提出一种轻量级的Hist2Style网络，用于高分辨率、实时的照片真实感风格迁移；

**💡 创新点**

1）用直方图嵌入提供可解释、可交互的风格控制；2）在双边空间中使用局部仿射变换的双边网格，避免幻觉并保持结构；3）通过蒸馏训练大模型的“专业”编辑能力，得到更快、更稳健的结果；

**🔧 技术方法**

双边网格（bilateral grid）变换、跨注意力融合、ConvNeXt编码器、MSE + Wasserstein‑2 分布损失、VGG感知损失、交叉注意力与 1D 直方图 CNN、softplus‑α 估计、图像与直方图的双分支网络；

**📊 数据集**

以 Unsplash Lite 25K 高质量图像为基础，合成每张图 6–7 个不同风格变体（共 1.7M 生成对），同时使用 136 张真实自然图和 19 张风格图进行评估；

**📈 对比分析**

在用户研究（31 名摄影专家，3000 次对比）中，Hist2Style 在 61% 以上的场景中获胜，显著优于 PhotoWCT2、Xia、SA‑LUT 等基线；在跑时（最高 4096² 时 0.1 s）、显存（4 MP 时 1 GB）与基线相比快 1–2 倍以上，且比 D‑LUT 省内存、无 amortized 预处理；

**⚠️ 局限性**

仅能对色彩与色调进行编辑，无法实现纹理、光照、对象替换等非色彩编辑；在视频、3D 资产中尚未验证；缺乏多层/多对象的细粒度空间控制，且对极端风格的匹配仍受限于直方图表达。

---

## 402. Tridirectional Discriminating-Power Formal Verification of Smart Contract Reentrancy Defense Against Production-Deployed Solidity Source

**arXiv ID:** 2606.01794 | [PDF](https://arxiv.org/pdf/2606.01794v1)

**作者:** Ray Iskander `[一作]` `[通讯]` (Verdict Security), Ray Iskander (Verdict Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对OpenZeppelin重入保护模式在 DAO 2016、Compound v2 cToken 和 Aave V3 三款生产协议上的负面攻击、正面正确性以及边界区分能力进行了完整的机器检查，并给出跨协议无改装组合的元定理。

**💡 创新点**

首次实现多协议三向判定力证明，提出无改装组合学说、最小差异突变测试与 CI axiom‑record 验证等方法，保证了证明的独立性与可复现性。

**🔧 技术方法**

使用 Lean 4 证明助手、mathlib4、Solidity 源代码层级状态机模型、持续集成与 axiom‑record introspection 以及 Mutation‑testing 方式。

**📊 数据集**

DAO 2016 合约、Compound v2 cToken 系列、Aave V3 主网合约以及自定义的最小差异突变体。

**📈 对比分析**

与传统静态分析/模式识别方法对比，本工作在机器检查层面实现零用户引入公理、全程 CI 检验，验证时间仅数分钟，覆盖约 8,500 行 Lean 源码。

**⚠️ 局限性**

仅针对重入攻击与 OZ 保护模式，未验证其它攻击/防御；模型假设 Solidity 编译器正确；跨协议组合仅在同一类协议内验证，尚未扩展到其它协议族。

---

## 403. PlatonicNav: Unveiling Semantic Correspondence in Navigation with Platonic Topological Maps

**arXiv ID:** 2606.01788 | [PDF](https://arxiv.org/pdf/2606.01788v1)

**作者:** Junlin Long `[一作]` (University of Sydney), Yang Zhao `[通讯]` (La Trobe University)

**通讯引用:** 8325 | [OpenAlex ID](https://openalex.org/A5056718303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出训练‑free 框架 PlatonicNav，将视觉‑语言导航与目标导航统一到同一语义流形上，并通过盲匹配将语言目标映射到视觉拓扑图。

**💡 创新点**

创新点在于利用 Platonic Representation Hypothesis 在不需要跨模态监督的情况下，通过相对关系实现语言与视觉的对齐，并构造融合几何与语义距离的 Platonic Topological Map。

**🔧 技术方法**

使用技术包括自监督视觉编码器（DINOv3）、语言编码器（GTR‑T5）、盲匹配（blind matching / quadratic assignment）、K‑means 聚类、语义拓扑图、Dijkstra 规划以及 ROS2 实时部署。

**📊 数据集**

实验基准为 HM3D‑IIN、HM3D‑OVON、R2R‑CE（MP3D）等仿真数据集，并在 Unitree Go2 四足机器人上进行真实世界验证。

**📈 对比分析**

与多种基线对比，PlatonicNav 在 IIN 上 SPL/SSPL 高于 ObjectReact，在 OVON 上超越多种跨模态训练的 ObjNav，在 R2R‑CE 上优于大多数 VLN 基线，展示了跨任务、跨模态的通用性能。

**⚠️ 局限性**

局限性：仍依赖自监督视觉编码器对语义的捕捉，盲匹配在细粒度语义上可能出现误匹配；在动态或视觉模糊环境下表现未评估；对语言多样性（同义词、句式）鲁棒性有限。

---

## 404. Breaking the Information Silo: Semantic Personas for Cross-Domain Recommendation

**arXiv ID:** 2606.01783 | [PDF](https://arxiv.org/pdf/2606.01783v1)

**作者:** Jonathan Mayo `[一作]` (Tel Aviv University), Konstantin Bauman `[通讯]` (Temple University)

**通讯引用:** 475 | [OpenAlex ID](https://openalex.org/A5031646048)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出 SPHERE 框架，利用大语言模型生成共享行为语义词表并生成可解释的语义人设，进而在无共享用户或物品的跨域环境下通过社区源人设实现推荐知识迁移。

**💡 创新点**

创新点在于把跨域对齐从实体/结构依赖转向行为语义对齐；引入 LLM 生成的可解释语义人设并聚合为社区源人设；以及在双塔架构中通过动态融合门实现语义与协同信号的自适应融合。

**🔧 技术方法**

技术包括：大语言模型（Llama 3.1 Instruct）生成行为特征，文本向量化（OpenAI 语义嵌入模型），双塔设计（协同塔与语义塔），动态融合门（tanh 激活），对比学习损失（InfoNCE）以及在 NCF、SVD++ 与 LightGCN 基线上的端到端集成。

**📊 数据集**

数据集：Amazon Books、Goodreads、Steam 三大平台，构造严格无重叠用户/物品的跨域配对，使用留一验证与全量测试。

**📈 对比分析**

采用 Leave-One-Out 全量评估，比较 NDCG@10 与基线，SPHERE 在所有目标域均提升 5–21%（最大提升 21.4% 发生在 Steam→Goodreads 的 SVD++），尤其在结构稀疏的 Steam 目标上效果最显著。

**⚠️ 局限性**

局限性：依赖 LLM 生成质量与提示设计，性能对不同 LLM、提示、嵌入模型敏感；实验仅覆盖三域，未验证更异质或大规模场景；且生成的行为人设可能涉及隐私与公平性问题。

---

## 405. Construction of Historical Knowledge Graphs Based on BERT and Graph Neural Networks

**arXiv ID:** 2606.01747 | [PDF](https://arxiv.org/pdf/2606.01747v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 406. HarnessForge: Joint Harness and Policy Evolution for Adaptive Agent Systems

**arXiv ID:** 2606.01779 | [PDF](https://arxiv.org/pdf/2606.01779v1)

**作者:** Mingju Chen `[一作]` (Beihang University), Shiji Zhou `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为HarnessForge的元适应框架，联合进化LLM代理的外部执行框架（harness）和内部推理策略（policy），实现对异构任务的自适应调优。

**💡 创新点**

核心创新在于将agent系统视为harness–policy对，采用故障引导的harness定制与harness条件化的policy对齐两阶段共进化，显著提升了两者的可执行兼容性。

**🔧 技术方法**

技术手段包括：基于GPT‑5.5的meta‑agent进行故障定位、归档引导改进和候选生成；对policy使用LoRA增量适配；利用成功轨迹进行监督式trace对齐；并支持SFT、GRPO、RLOO等多种训练方式。

**📊 数据集**

实验使用ToolHop、RestBench‑TMDB、API‑Bank、SearchQA（HotpotQA+2WikiMultiHopQA）等多域评测数据集。

**📈 对比分析**

与搜索式harness基线（AFlow、ADAS等）及训练式policy基线（SFT、GRPO、RLOO）比较，HarnessForge在五大benchmark上平均提升3.56%（最高达12%），且在rollout‑性能曲线上接近Pareto前沿，展现出更优的成本效益。

**⚠️ 局限性**

限制包括：仅在Qwen3‑4B/8B两种规模模型上验证；需多轮rollout，长时序环境下成本仍高；以及编辑操作空间受限，未覆盖所有可能的agent实现方向。

---

## 407. Faster than the Team, Faster than the Customer: Tool Integration, Collaboration, and Organisational Lag in AI-assisted RE

**arXiv ID:** 2606.01772 | [PDF](https://arxiv.org/pdf/2606.01772v1)

**作者:** Jan-Philipp Steghöfer `[一作]` `[通讯]` (XITASO GmbH Software & IT Solutions), Jan-Philipp Steghöfer (XITASO GmbH Software & IT Solutions)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对XITASO公司内部使用的AI辅助需求工程工具进行了实证研究，结合公司问卷和两轮半结构访谈，识别并分析了15个使用案例与四个类别，评估其对工作流程、工具集成及PO-开发者关系的影响。

**💡 创新点**

提供了真实工业环境中AI辅助需求工程的纵向实证证据，揭示工具集成与组织治理是实现价值的关键，而非单纯工具功能；同时提出针对实践者的自评问题，弥补了学术研究对协作动态关注不足的空白。

**🔧 技术方法**

采用定性研究方法：公司范围调查、两轮半结构访谈；使用LLM与AI工具（ChatXiPT、Product Copilot、TenderZen、Claude Code等），并利用MCP协议实现工具与项目管理、代码库的集成。

**📊 数据集**

使用XITASO内部数据：20个使用案例（来自11个社区）、8名PO的访谈记录与工具使用经验；未使用公开公开数据集。

**📈 对比分析**

通过主题分析和成员检查对访谈数据进行编码，对比不同工具的使用效果；未进行量化性能对比，但发现集成完善时时间节省显著（如TenderZen从4小时降至30分钟），缺乏集成时效率低下且需手工工作。

**⚠️ 局限性**

局限性包括：仅研究单一公司八名PO，外部可推广性有限；数据为访谈笔记而非完整转录，可能遗漏细节；研究为描述性、探索性，缺乏因果推断；工具与客户治理的限制可能导致结果偏差。

---

## 408. Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams

**arXiv ID:** 2606.01770 | [PDF](https://arxiv.org/pdf/2606.01770v1)

**作者:** Zewen Liu `[一作]` (Emory University), Hanqing Lu `[通讯]` (Amazon)

**通讯引用:** 23599 | [OpenAlex ID](https://openalex.org/A5100511737)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Adaptive Auto‑Harness，结合持续演化、解题时路由和人机协作，以应对长序列、异构且分布漂移的任务流，提升 LLM 代理性能。

**💡 创新点**

创新点在于将演化损失与适配损失分解，设计多代理四阶段演化与跨周期状态，构建基于树的解题时路由，并通过结构触发的人机干预提升适应性。

**🔧 技术方法**

使用 Claude Sonnet/Opus LLM、git‑tree 结构的多分支 harness、agentic 路由、四阶段多代理演化（分析、研究、构建、验证）以及人机交互接口。

**📊 数据集**

实验数据集包括 PolyBench（预测市场）、CTF‑Dojo（安全挑战）和 FutureX（事件预测）三条开放式任务流。

**📈 对比分析**

与无演化 baseline、A‑Evolve、GEPA、Meta‑Harness、Continual Harness、SkillOS 及人类设计系统 OctoTools 对比，Adaptive Auto‑Harness 在 PolyBench Accuracy/Return、CTF‑Dojo Pass@1 和 FutureX Pass@1 等指标上均显著领先。

**⚠️ 局限性**

局限性：评估仅覆盖三种任务流，未检验更广泛的部署场景；演化损失和适配损失仅为理论拆解，缺乏直接估计；人机干预的实验仅限于特定缺失信号的场景。

---

## 409. Dynamic Trust-Aware Sparse Communication Topology for LLM-Based Multi-Agent Consensus

**arXiv ID:** 2606.01828 | [PDF](https://arxiv.org/pdf/2606.01828v1)

**作者:** Wanshuang Gou `[一作]` (Chengdu University), Zihan Liu `[通讯]` (Chengdu University)

**通讯引用:** 4010 | [OpenAlex ID](https://openalex.org/A5119011515)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了动态稀疏共识机制 DySCo，能够在多轮 LLM 多代理推理中动态评估并选择高价值通信边，采用压缩反馈、信任加权聚合并通过熵阈值实现早停，显著降低通信成本。

**💡 创新点**

创新点包括：①利用历史信任、当前置信度、答案偏离和任务相关性实时评估通信边价值；②在每轮仅选 k 个邻居进行稀疏通信，动态适配任务状态；③引入基于熵的早停机制，及时停止无效轮次；④在理论上给出了稀疏通信下的一致性与复杂度分析。

**🔧 技术方法**

核心技术：多代理 LLM 协作框架、动态稀疏通信拓扑、信任权重更新与加权聚合、消息压缩策略、熵阈值早停、抽象一致性与通信复杂度分析。

**📊 数据集**

实验数据集：GSM8K（数学推理）、LogiQA（逻辑推理）和 StrategyQA（常识问答），每个数据集随机抽取 300 条测试样本。

**📈 对比分析**

与单模型 Chain‑of‑Thought、Self‑Consistency、全连通多代理辩论（Dense MAD）以及静态环、随机 k 和仅基于信任的稀疏拓扑进行对比；DySCo 在准确率上超过 Dense MAD（最高可达 86.8%），同时 token 消耗下降约 70%，延迟约减半，显示出优异的性能与成本平衡。

**⚠️ 局限性**

局限性：依赖 LLM 置信度和外部验证器，置信度失准可能导致错误放大；过度稀疏或信任偏置可能产生 rich‑get‑richer 效应，形成局部意见集群；对开放式生成或多答案任务的适用性有限，需要结合外部检验或探索机制。

---

## 410. DisFlow: Scene Flow from Distance Field for Object Pose, Velocity Tracking, and Dynamic Object Reconstruction

**arXiv ID:** 2606.01824 | [PDF](https://arxiv.org/pdf/2606.01824v1)

**作者:** Lan Wu `[一作]` (University of Technology Sydney), Teresa Vidal-Calleja `[通讯]` (University of Technology Sydney)

**通讯引用:** 2083 | [OpenAlex ID](https://openalex.org/A5086794522)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DisFlow框架，利用距离场的场景流在对象坐标系中实现6DoF姿态、速度跟踪和稠密表面重建；

**💡 创新点**

创新点在于将Gaussian Process Implicit Surfaces与对象坐标系结合，利用距离场场景流作为对应关系，通过闭式优化实现实时概率融合并给出不确定性估计；

**🔧 技术方法**

使用Gaussian Process Implicit Surfaces、Octree局部GP、传输方程/场景流、SE(3)闭式优化以及运动平滑正则化等技术；

**📊 数据集**

在Fast‑YCB动态物体数据集以及真实序列上进行实验；

**📈 对比分析**

与DOPE、ROFT、PoseRBPF、TrackNet、TSDF等基线对比，姿态ADD‑AUC提升至92%以上，速度RMSE比ROFT低约50%，重建Chamfer/Hausdorff和F‑score显著优于TSDF；

**⚠️ 局限性**

对高度对称或几何结构极少的物体约束弱，优化可能产生模糊解，且缺少视觉特征匹配的支持。

---

## 411. TalkTag: Fine-Grained Morphosyntactic Error Annotation for Transcribed Speech

**arXiv ID:** 2606.01820 | [PDF](https://arxiv.org/pdf/2606.01820v1)

**作者:** Shamira Venturini `[一作]` (Karlsruhe Institute of Technology), Jannik Strötgen `[通讯]` (Karlsruhe University of Applied Sciences)

**通讯引用:** 2066 | [OpenAlex ID](https://openalex.org/A5066026428)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一款轻量级的 LLM 工具，用于在儿童叙事语料中自动化 CHAT 风格的细粒度形态句法错误注释。

**💡 创新点**

创新点在于：① 将 CHAT 错误注释视为受限的结构化生成任务；② 在极低资源条件下，通过合成数据扩充与 LoRA 微调，使大型 Llama‑3.1‑8B 在 4‑bit 量化下实现高质量标注；③ 采用可重用的标注子词表与结构化提示，提升标注的语法一致性与可解释性。

**🔧 技术方法**

技术细节包括：Meta Llama‑3.1‑8B‑Instruct 预训练模型；Unsloth + LoRA（rank 32，α=64）在 4‑bit 量化上进行参数高效微调；扩展分词器以支持标注构件；自定义结构化提示；Synthetic data augmentation 生成 5,830 条合成错误实例；自动评估、盲后审与人工评估三阶段评测流程。

**📊 数据集**

数据集：使用 TalkBank/CHILDES 中的 Edmonton Narrative Norms Instrument (ENNI) 语料，包含 4‑5 岁儿童 4,585 条真实手工注释的叙事句；另外生成 5,830 条合成错误实例，合计 10,415 条训练样本。

**📈 对比分析**

性能与比较：在保留测试集上，微型 F1 80.4%，精确率 86.0%，召回率 75.5%；对错误句子单独评估微 F1 82.7%；盲后审后可接受率提升至 82.8%；在人类评审的未见 ENNI 语料上，句子级可接受率 93.4%，标签级正确率 83.7%；总体上与手工注释差距主要集中在罕见标签与歧义上下文上。

**⚠️ 局限性**

局限性：① 仅覆盖部分 CHAT 错误标签，未包含所有形态句法错误；② 训练数据仍以单一语料为主，跨语料、跨年龄、跨临床人群的泛化尚未验证；③ 对于缺失时态与词干同形的零变形动词易产生误判；④ 仍需人工审校作为后处理，不能完全替代专家；⑤ 合成数据虽然提升覆盖度，但仍可能带来偏差。

---

## 412. CRAB-Bench: Evaluating LLM Agents under Complex Task Dependencies and Human-aligned User Simulation

**arXiv ID:** 2606.01815 | [PDF](https://arxiv.org/pdf/2606.01815v1)

**作者:** Danqing Wang `[一作]` (Carnegie Mellon University), Lei Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12314 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于约束图的真实代理基准与人类行为驱动的用户仿真引擎，用于生成具有复杂任务依赖和误导性干扰的多实体任务，并通过状态验证评估代理在真实服务场景中的表现。

**💡 创新点**

创新点：①使用约束图自动生成满足至少一个解且含大量干扰项的任务，显著提升任务难度；②构建基于人类行为研究的多维度用户仿真（四个行为维度，三种人格），替代模板式协作式仿真；③结合具体状态和抽象状态双重验证，适用于多解任务。

**🔧 技术方法**

技术：约束满足问题（CSP）求解、节点与边约束生成、工具链交互、状态验证器、LLM代理（DeepSeek、Qwen3、Claude Sonnet、GLM-5）以及RUSE用户仿真模型。

**📊 数据集**

数据集：基于Trip Booking领域构建的200个任务（S1–S4），每组包含不同种子解；使用人类行为研究数据定义的四个行为维度与三个人格模型；数据库初始化包含实际机票、酒店等实体记录。

**📈 对比分析**

比较方法：在与既有基准（如τ-bench、τ^2-bench）相同的任务框架下，评估四个LLM代理的 pass@1 与 pass_k；结果显示在新基准上最高pass@1仅61%，而在旧基准上可达≈80%，证明新基准更具挑战性；人类式用户导致性能下降19–57%。

**⚠️ 局限性**

局限性：仅在旅游预订领域验证，难以覆盖其他代理场景；用户仿真仍基于LLM，缺乏更丰富的真实用户行为（如欺骗、主题漂移等）；基准依赖于手工定义的实体与约束，扩展到新领域需重新设计。

---

## 413. ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference

**arXiv ID:** 2606.01806 | [PDF](https://arxiv.org/pdf/2606.01806v1)

**作者:** Sourav Das `[一作]` `[通讯]` (Indian Institution of Information Technology Kalyani), Sourav Das (Indian Institution of Information Technology Kalyani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个将神经缩放律与语言模型探测相结合的框架，自动从预训练的小型语言模型中提取参数高效的子网络，实现特定任务的高效推理。

**💡 创新点**

创新点在于把探测器的性能量化为层相关性得分，并在给定参数预算下求解最优层子集，突破了传统经验式层选择方法。

**🔧 技术方法**

采用神经缩放律假设、线性/浅层 MLP 探测器评估各层语言特征、基于预算的子网络搜索以及少量微调。

**📊 数据集**

在 RoBERTa‑Large 与 T5‑Base 上，用 GLUE 任务 SST‑2、QNLI 以及 POS 标注等数据进行探测与评估。

**📈 对比分析**

与原模型、Top‑k、Uniform‑k 以及 Distil 版基线对比，在 23%–15% 参数预算下，保持 95–98% 原模型性能，显著优于经验式基线。

**⚠️ 局限性**

局限性包括仅支持连续层选择、探测器简单，未细粒度分析头/FFN 结构，且模型偏见问题未得到缓解。

---

## 414. Tree-Guided Identify-Then-Exploit: A Unified Framework of Best Arm Identification and Regret Minimization for Dueling Bandits

**arXiv ID:** 2606.01799 | [PDF](https://arxiv.org/pdf/2606.01799v1)

**作者:** Pu Wang `[一作]` (Zhejiang University), Yao-Xiang Ding `[通讯]` (Zhejiang University)

**通讯引用:** 84 | [OpenAlex ID](https://openalex.org/A5045601793)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 Tree-Guided Identify-Then-Exploit (TG-ITE) 框架，统一解决了在 Condorcet 胜者假设下的最佳臂识别 (BAI)、弱均衡损失和强均衡损失三种目标，且在不依赖更强假设的前提下，实现了线性样本复杂度。

**💡 创新点**

核心创新在于：① 提出了共享的树引导识别子程序 TreeAscent，利用树形锦标赛分解实现 O(N) 次比较；② 针对不同目标设计了专属的利用阶段（停止、Winner‑Stays 风格的 ScreenAndReplace、以及自我对弈），从而在保持低样本成本的同时实现各目标的最优（或接近最优）性能；③ 在已知下界的情况下推出了预算化 DuelTest，消除了对全局间隔的对数依赖。

**🔧 技术方法**

主要技术包括：树形分解（Balanced Binary Tree + 祖先路径分块）、均衡锦标赛（Balanced Knockout Tournament）、自适应双边测试（DuelTest 与 BudgetedDuelTest）、Winner‑Stays 变体（ScreenAndReplace）、自我对弈（Self‑Play）以及多阶段、任何时刻的强均衡损失策略。

**📊 数据集**

实验使用了两类合成偏好矩阵：循环实例（Condorcet 胜者对所有其他臂有固定优势，剩余臂形成循环关系）和稀疏实例（基于循环的结构但非胜者的概率趋近 0.06/0.94，产生更大的次优间隙），在 N=8、16、32、256 的不同规模下进行评估。

**📈 对比分析**

与 BKT、SeqElim、WS‑W、WS‑S、WR‑TINF、WR‑EXP3‑IX、WSW‑PE、Versatile‑DB 等基线比较：在 BAI 任务中 TG‑ITE‑BAI 与 BKT 的线性规模相当；在弱均衡损失任务中 TG‑ITE‑Weak 以 2,384±82 的累计损失领先于所有基线，并在前 5,000 次对弈后趋于平稳；在强均衡损失任务中 TG‑ITE‑Strong‑KH 与 TG‑ITE‑Strong‑Anytime 分别以 5,928±154 和 25,337±336 的累计损失超过所有对手，且表现出更低的方差。

**⚠️ 局限性**

主要局限：① 对全局间隔 Δ 的正间隔假设仍然必要，无法在完全无信息的情形下去掉对数因子；② 预算化版本虽然消除了 log(1/Δ) 但需要先验下界 g；③ 对 BAI 的最优样本复杂度仍受限于 log(1/Δ) ；④ 在某些极端实例（如非 Condorcet、强顺序假设不满足）下，TreeAscent 的性能可能不如针对性 BAI 算法；⑤ 任何时刻强均衡损失策略在 Δ 方面仍次优，需要进一步研究如何在不牺牲探索的前提下实现 Δ‑最优。

---

## 415. Whole-Pool Setwise Reranking with Long-Context Language Models

**arXiv ID:** 2606.01782 | [PDF](https://arxiv.org/pdf/2606.01782v1)

**作者:** Hang Li `[一作]` (University of Queensland), Guido Zuccon `[通讯]` (University of Queensland)

**通讯引用:** 4979 | [OpenAlex ID](https://openalex.org/A5076031002)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了全池Setwise重排序方法及其双端版本DualEnd，在一次LLM调用中同时选择最相关和最不相关段落，以此在一次性处理所有候选段落的前提下完成完整排名；

**💡 创新点**

创新点在于利用长上下文LLM一次性获取全池信息，双端填充实现每次调用产生两条排序决策，调用次数近一半（从99降至50）且效果基本保持；

**🔧 技术方法**

核心技术为LLM Setwise提示工程、DualEnd算法、长上下文模型（如Gemma3、Qwen3.5、Llama3.1、Ministral3）以及贪婪解码与解析容错；

**📊 数据集**

实验使用TREC Deep Learning 2019/2020的MS MARCO段落集合，并在BM25前N（10/20/…/100）候选池上进行重排序；

**📈 对比分析**

与传统窗口Setwise和单端全池方法相比，WP-DE仅需50次串行调用（相对99），在nDCG@10约0.627（接近WP-T 0.703）且token和运行时间均下降约49%，证明了显著的成本-效能提升；

**⚠️ 局限性**

局限性包括仅在固定第一阶段候选集和TREC DL19/20数据集上验证，未评估检索召回或跨域适用性；对输入顺序敏感；使用单一prompt形式和解码策略，其他策略可能进一步影响效果与成本。

---

## 416. Trans2Occ: Voxel Occupancy Estimation and Grasp for Transparent Objects from Simulation to Reality

**arXiv ID:** 2606.01777 | [PDF](https://arxiv.org/pdf/2606.01777v1)

**作者:** Yixuan Yang `[一作]` (Shanghai AI Laboratory), Dongzhan Zhou `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于单张RGB图像预测透明物体体素占据的框架Trans2Occ，并实现了透明物体的机器人抓取。

**💡 创新点**

核心创新在于直接从RGB预测体素占据，绕过传统深度/多视角方法的折射与反射难题；构建了可扩展的Sim-Trans3D仿真数据管线，生成RGB与占据标注；以及基于占据的规则抓取策略，实现从感知到抓取的一体化。

**🔧 技术方法**

采用几何感知投影+深度概率分布将2D特征提升至3D体素空间，再用3D卷积网络进行空间推理；使用视觉Transformer编码器提取语义特征；在占据上基于PCA方向、碰撞检查的规则抓取策略；利用余弦退火、AdamW等训练技巧。

**📊 数据集**

使用自研Sim-Trans3D合成数据集（约5k场景，包含100种材料/光照、50个物体布置），并在真实场景采集40张RGB图用于仿真到真实的转移验证。

**📈 对比分析**

与DepthAnythingV2、Marigold等单目深度估计、MonoScene、ISO等占据预测、AnyGrasp、MODEST抓取基线对比。Trans2Occ在占据IoU/mIoU上明显领先，深度RMSE下降至0.35，仿真抓取成功率71.7%，真实抓取成功率57.1%（完成率83.3%）。

**⚠️ 局限性**

仅适用于静态透明物体，无法处理含液体等动态内容；对相机姿态误差敏感；规则抓取策略在复杂形状或稀疏占据下可能失效；尽管占据在一定程度上跨域，但在极端光照/材质变化下仍存在泛化瓶颈。

---

## 417. PillarDETR: YOLO-Backbone and RT-DETR Head for Real-Time 3D Object Detection

**arXiv ID:** 2606.01757 | [PDF](https://arxiv.org/pdf/2606.01757v1)

**作者:** Smit Kadvani `[一作]` (Independent Researcher), Harsh Dave `[通讯]` (Illinois Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种端到端的3D目标检测框架PillarDETR，结合柱状编码、YOLOv8 CSP骨干网络和RT‑DETR Transformer头，实现无锚点、无NMS的直接预测。

**💡 创新点**

创新点在于：①用YOLOv8 CSP骨干替代传统CNN，提高特征表达与梯度流；②将RT‑DETR解码器迁移到BEV空间，利用全局注意力完成集合预测；③实现高精度与实时速度的权衡。

**🔧 技术方法**

核心技术包括柱状特征网络（PFN）、YOLOv8的C2f模块、Path Aggregation Network、RT‑DETR Transformer头、Hungarian匹配、Focal Loss与角度专用损失。

**📊 数据集**

在KITTI、nuScenes、Waymo Open和SUN RGB‑D四个公开数据集上进行实验验证，覆盖车辆、行人、自行车、室内物体等多类。

**📈 对比分析**

与PointPillars、SECOND、CenterPoint等基线相比，KITTI中中等难度mAP提升至66.57%（+3.79%），nuScenes mAP提升至46.8%（+16.3%），Waymo车辆mAPH提升至64.5%；在A100 GPU上实现41.2 FPS，24.3 ms延迟。

**⚠️ 局限性**

局限性包括：相较基线参数增至12.4 M，算力需求提升；Transformer头的计算量仍高，难以进一步压缩；仅使用单一LiDAR模态，对光照、遮挡等极端环境的鲁棒性待提升。

---

## 418. EvoCut: Multi-Layer Evolution-Aware Visual Token Compression for Efficient Large Vision-Language Models

**arXiv ID:** 2606.01756 | [PDF](https://arxiv.org/pdf/2606.01756v1)

**作者:** Hongyu Lu `[一作]` (Harbin Institute of Technology), Shikai Jiang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 375 | [OpenAlex ID](https://openalex.org/A5082270319)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了EvoCut，一种无训练、无注意力的视觉令牌压缩方法，利用视觉编码器跨层演化方向评估令牌重要性并进行筛选。

**💡 创新点**

创新点在于识别视觉令牌在各层形成的多组演化方向，并通过令牌对这些组方向的持续偏差来判定其重要性，避免单层注意力或表示的不足。

**🔧 技术方法**

采用令牌演化方向分析、K‑means/HDBSCAN聚类、余弦偏差评分、指数移动平均累计得分等技术实现无训练压缩。

**📊 数据集**

实验使用图像数据集GQA、MMB、MMB^CN、MME、POPE、SQA、VQA^V2、TextVQA、SEED；视频数据集TGIF‑QA、MSVD‑QA、MSRVTT‑QA；在LLaVA‑1.5‑7B、LLaVA‑NeXT‑7B、Qwen‑2.5‑VL‑7B、Video‑LLaVA‑7B上进行评估。

**📈 对比分析**

与FastV、SparseVLM、PDrop、V^2Drop、VisionZip、ApET等训练自由压缩基线比较，EvoCut在不同令牌预算下平均性能保持≥94%，例如64令牌时可保持94.4%平均性能，并实现1.44×总时间加速，显著优于所有基线。

**⚠️ 局限性**

局限性包括需访问视觉编码器内部隐藏状态，无法在黑盒或API‑only LVLM中使用；实验仅覆盖部分图像/视频任务，未涵盖更长视频或更高分辨率场景。

---

## 419. Benign Inputs, Harmful Outputs: Cross-Modal Jailbreaking via Distributed Semantic Recomposition

**arXiv ID:** 2606.01837 | [PDF](https://arxiv.org/pdf/2606.01837v1)

**作者:** Yani Wang `[一作]` (City University of Macau), Zhuo Ma `[通讯]` (Xidian University)

**通讯引用:** 2669 | [OpenAlex ID](https://openalex.org/A5056982659)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种分布式语义重构（Distributed Semantic Recomposition, DSR）框架，用全无害的文本与图像原语诱导多模态大语言模型（MLLMs）生成有害内容。

**💡 创新点**

创新点在于：①将有害意图拆解为多个独立的、无害的视觉与文本原语，避免直接触发安全过滤；②利用同义指导的无害提示与迭代优化，让模型在内部跨模态推理中重组原始有害语义；③揭示“实用性‑安全性悖论”，即更强的跨模态推理能力会使模型更易被利用。

**🔧 技术方法**

使用的技术包括：大语言模型（LLM）进行语义拆解与同义提示生成；文本到图像模型（如Stable Diffusion）生成无害图像原语；交互式迭代优化策略（基于拒绝反馈）改进提示；跨模态注意力机制作为攻击的核心执行点。

**📊 数据集**

使用的数据集为 VBCDE（包含 42 条图像暴力/血腥提示）和 T2I‑RiskyPrompt（102 条暴力提示）。

**📈 对比分析**

与 DACA、SneakyPrompt、PGJ 等基线进行对比，评估指标包括多模型 ASR（Gemini、Qwen、Wanx）、CLIP 对齐、NIQE 质量、输入毒性率等。DSR 在所有模型上均显著提升 ASR（如 Gemini VBCDE 上从 0% 提升至 28.59%），同时保持低输入毒性率与良好图像质量。

**⚠️ 局限性**

局限性：①对语义拆解的质量高度依赖，复杂/抽象有害意图难以精准拆分；②拆解与提示生成需依赖高性能 LLM，可能受对齐约束限制；③同义提示设计对攻击成功率影响大，需手工调优，缺乏自动化；④未针对所有多模态安全策略（如动态检测）全面验证。

---

## 420. ROGLE: Robust Global-Local Alignment with Automated Region Supervision for Text-Based Person Search

**arXiv ID:** 2606.01825 | [PDF](https://arxiv.org/pdf/2606.01825v1)

**作者:** Zequn Xie `[一作]` (Zhejiang University), Tao Jin `[通讯]` (Zhejiang University)

**通讯引用:** 6726 | [OpenAlex ID](https://openalex.org/A5100661557)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ROGLE框架，利用自动的Region-to-Sentence Matching（RSM）生成伪区域-句子对，并在全局与局部层面进行联合对齐，从而提升文本检索的人物搜索性能。

**💡 创新点**

创新点包括：1) 通过RSM自动生成细粒度监督，避免人工标注；2) 设计多粒度学习策略，将全局对比学习与局部对齐相结合；3) 构建P-VLG基准，提供丰富的长文本和100k+区域对齐标注，支持全局与局部评估。

**🔧 技术方法**

使用CLIP（ViT-B/16）作为双编码器，SAM进行图像分割，spaCy分句，CLIP特征做跨模态相似度匹配，双分支损失（全局+局部）与可靠性引导权重组合训练。

**📊 数据集**

使用公开数据集CUHK-PEDES、ICFG-PEDES、RSTPReid，并从中构建P-VLG基准（包含48,485张图片、68,990长文本和100k+区域-句子对）。

**📈 对比分析**

在CUHK-PEDES、ICFG-PEDES、RSTPReid以及新提出的P-VLG长文本基准上进行对比实验，ROGLE在Rank‑1、mAP和mINP等指标均超过现有SOTA，尤其在长文本查询上提升显著（如P‑VLG Rank‑1 78.03% 对比基线73.30%）。

**⚠️ 局限性**

局限性包括：RSM的贪心一对一匹配可能无法处理多区域或多句描述的情况；目前仅在英文数据上验证，缺乏多语言适配；模型对极端噪声和复杂场景的鲁棒性尚待进一步提升。

---

## 421. Hierarchically Decoupled Mixture-of-Experts for Robust Traffic Sign Recognition in Complex Driving Scenarios

**arXiv ID:** 2606.01822 | [PDF](https://arxiv.org/pdf/2606.01822v1)

**作者:** Mingxiao Wang `[一作]` (Liaoning University of Technology), Lei He `[通讯]` (Tsinghua University)

**通讯引用:** 32309 | [OpenAlex ID](https://openalex.org/A5102798483)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 CBDES MoE TSR 框架，通过分层解耦的混合专家池和显式监督的图像级硬路由，实现交通标识识别的输入自适应动态推理。

**💡 创新点**

创新点包括：①将多场景任务拆分为功能级解耦并构建异构 YOLO 专家池；②采用显式监督的 Top‑1 硬路由实现稀疏激活；③在同一框架下兼顾高精度与低算力，显著提升实时性能。

**🔧 技术方法**

使用了混合专家（Mixture‑of‑Experts）与层次解耦技术、MobileNetV3‑Small 路由网络、两阶段分离训练、YOLOv11s/YOLOv9c 专家、交叉熵路由损失，以及 COCO mAP 评价标准。

**📊 数据集**

采用复合多域交通标识数据集，包括标准清晰域（C）、远距离小目标域（S）和恶劣天气域（W），样本按 5:3:2 比例混合，数据来源于 Roboflow、TT100K 并加入雨雪雾增强。

**📈 对比分析**

与 YOLOv9c 基线和 Faster R‑CNN 对比，mAP50‑95 从 74.5% 提升至 76.8%（+2.3%），平均 GFLOPs 从 103.70 降至 62.83（约 39.4% 降低），FPS 从 53.78 提升至 58.82。

**⚠️ 局限性**

局限性包括：依赖显式域划分与监督，增加数据准备复杂度；在混合场景下专家重叠可能导致误路由；路由网络为离线静态训练，无法适应分布漂移；需要进一步优化轻量专家与在线学习机制。

---

## 422. Cost-Aware Diffusion Draft Trees for Speculative Decoding

**arXiv ID:** 2606.01813 | [PDF](https://arxiv.org/pdf/2606.01813v1)

**作者:** Shuai Zhang `[一作]` (Zhejiang University), Yong Dai `[通讯]` (Westlake University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种成本感知的扩散草稿树（CaDDTree）方法，在每一推理回合中根据草稿机置信度与验证成本动态选择树节点数，从而直接优化每个 token 的生成时延。

**💡 创新点**

创新点：① 用吞吐率（token/时间）而非传统的接受长度作为优化目标；② 证明吞吐率随树节点数呈单峰，允许用贪心停止规则高效寻找最优节点数；③ 无需离线预算搜索，完全按回合自适应；④ 通过一次性成本剖面即可完成。

**🔧 技术方法**

技术：规格化推断（speculative decoding）、块扩散草稿（block diffusion）、树注意力、一次性草稿生成、成本剖面、凸回归、单峰最优搜索。

**📊 数据集**

数据集：Qwen3-4B/8B模型在八个基准上评测，涵盖推理、代码与指令遵循任务：MATH-500、GSM8K、AIME 2025、HumanEval、MBPP、LiveCodeBench、MT-Bench、Alpaca。

**📈 对比分析**

与 AR、DFlash 以及 DDTree-oracle（固定预算的最优树）进行对比。实验结果表明，CaDDTree 在大多数任务上匹配或超过 DDTree-oracle 的吞吐率，并且无需手工设置预算。

**⚠️ 局限性**

限制：依赖离线成本剖面的准确性；目前仅在块扩散草稿框架下验证；假设验证成本为凸函数，若该假设不成立则单峰性质和贪心算法的最优性可能失效。

---

## 423. OctoT2I: A Self-Evolving Agentic Text-to-Image Router

**arXiv ID:** 2606.01803 | [PDF](https://arxiv.org/pdf/2606.01803v1)

**作者:** Xu Jiang `[一作]` (Peking University), Jian Zhang `[通讯]` (Peking University)

**通讯引用:** 54809 | [OpenAlex ID](https://openalex.org/A5100410082)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套可自适应选择多种文本转图像模型的代理系统 OctoT2I，通过自演化机制学习工具知识并利用状态化多轮路由实现性能与效率协同优化

**💡 创新点**

创新点包括：①无监督自演化机制自动构建工具知识库，消除人工先验与昂贵标注；②状态化多轮路由决策，结合知识与记忆实现动态、低延迟的工具选择；③基于 LLM 的链式推理与评价反馈循环，使系统能够在无外部监督下持续改进

**🔧 技术方法**

核心技术包括：大型语言模型（Qwen2-0.5B/ GPT‑4o）驱动的决策与评价；Propose–Solve–Evaluate–Learn（PSEL）循环与探索空间剪枝；多模型推理、基于视觉‑语言评估器（NVILA-Lite‑2B‑Verifier）得到的连续质量分数；能耗与时间评估（CO₂e、kWh·PUE）

**📊 数据集**

使用了 GenEval、T2I‑CompBench++、WISE 三大公开基准以及 30 条真实用户提示的内部用户研究

**📈 对比分析**

与非代理模型相比，在 GenEval 上获得 0.96 的最高分，超越 Flow‑GRPO；在 T2I‑CompBench++ 上平均得分 0.6618，领先所有代理方法；在 WISE 上得到 0.54 分；在用户研究中，投票率 70.4% 而推理时间仅 18.45 s（相较 ChatGen 53.34 s）

**⚠️ 局限性**

局限性包括：①仍需依赖预设的提示模板与 LLM 质量；②针对特定 T2I 任务，扩展到图像编辑或 3D 生成等其他生成任务尚未验证；③自演化过程对计算资源有一定需求，且在极端复杂场景下的收敛速度可能受限

---

## 424. Stochastic convergence of parallel asynchronous adaptive first-order methods

**arXiv ID:** 2606.01787 | [PDF](https://arxiv.org/pdf/2606.01787v1)

**作者:** Serge Gratton `[一作]` (University of Toulouse), Philippe L. Toint `[通讯]` (University of Toulouse)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一类异步自适应一阶优化方法，结合块级自适应预条件、动量和随机梯度，在非凸优化中实现收敛。

**💡 创新点**

创新点在于将异步块坐标更新与多种自适应预条件器（AdaGrad、Shampoo、Muon等）统一到同一框架，并首次给出在非凸随机环境下的收敛理论。

**🔧 技术方法**

采用块级异步（Block‑Jacobi）调度、随机梯度Oracle、可变预条件更新（全/对角、Newton‑Schulz等）、动量变体以及延迟与累计方差的理论分析。

**📊 数据集**

实验数据集包括 FashionMNIST、MoE‑FashionMNIST、CovType、MovieLens 100K、Criteo 与 SVHN。

**📈 对比分析**

通过与同步版本及基线异步方法比较，实验显示异步变体保持甚至提升收敛质量，吞吐量提升可达 10 倍，且在多种任务上获得更优或相近的泛化性能。

**⚠️ 局限性**

局限性包括需满足有界延迟与 Lipschitz 光滑假设；动量方案对步长或额外条件敏感；矩阵预条件器计算成本高；实验仅在全梯度评估模型下验证，未覆盖更低成本的块梯度场景。

---

## 425. FLARE: Diffusion for Hybrid Language Model

**arXiv ID:** 2606.01774 | [PDF](https://arxiv.org/pdf/2606.01774v1)

**作者:** Yuchen Zhu `[一作]` (Adobe Research), Jiuxiang Gu `[通讯]` (Adobe Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套完整的转换流程，将具有混合软/线性注意力结构的自回归大语言模型（AR LLM）转换为高效的扩散式大语言模型（dLLM），并在同一检查点上实现自回归和扩散两种解码路径；

**💡 创新点**

创新点在于：①提出了平衡的 token‑级别干净/噪声双流训练目标和文档级打包注意力掩码，兼顾 AR 预测与块级扩散；②设计了针对混合注意力骨干的状态调度和专门的高效训练 kernel；③构建了统一的推理堆栈，使单个模型同时支持 AR‑Trust 与 Diffusion‑Trust 解码；

**🔧 技术方法**

核心技术包括：token‑平衡干净/噪声双流损失、文档级打包注意力掩码、Gated DeltaNet (GDN) 线性注意力的状态调度、专门的 CUDA 内核（Route II）与融合的验证/Top‑K 计算、SGLang 统一推理框架；

**📊 数据集**

使用的训练数据为四种混合组合（Mix 1/4），分别由公开的 Long‑CoT、Math、Instruction‑Follow（IF）等 SFT 语料构成，覆盖推理、数学、知识、代码等多种任务；

**📈 对比分析**

在 2B/4B/9B 规模的转换模型上与 LLaDA‑2.1‑flash、SDAR、Mercury‑2 等主流 dLLM 进行对比，9B 模型在大多数基准上匹配或超过 LLaDA‑2.1‑flash、在数学推理上显著优于 Mercury‑2，且在推理吞吐量上在单 GPU 高并发下实现 2‑4 倍提升；

**⚠️ 局限性**

主要限制包括：训练成本约为单向 AR 的两倍；与源 AR 模型相比在某些任务上仍存在 5–15 分的性能差距，原因是转移数据与源模型分布不匹配；实验仅覆盖 10B 以内的稠密模型，未验证 MoE 架构或强化学习等后续训练阶段。

---

## 426. EvoBrain: Continual Learning of EEG Foundation Models Across Heterogeneous BCI Tasks

**arXiv ID:** 2606.01767 | [PDF](https://arxiv.org/pdf/2606.01767v1)

**作者:** Yangxuan Zhou `[一作]` (Zhejiang University), Gang Pan `[通讯]` (Zhejiang University)

**通讯引用:** 473578 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了EEG基础模型的跨任务连续学习，提出EvoBrain框架实现单一模型在多任务上动态适应。

**💡 创新点**

关键创新是结合神经频谱任务归一化(NSN)和响应亲和蒸馏(RAD)两大组件，实现跨任务分布及频谱对齐与选择性知识迁移。

**🔧 技术方法**

采用自监督预训练EEG基础模型（Transformer、Mamba等）+ NSN、RAD+时间依赖重放与响应几何蒸馏技术。

**📊 数据集**

在六个不同EEG任务（情绪识别、运动想象、睡眠分期、想象语音、精神疾病诊断、运动想象再）上测试，使用FACED、PhysioNet-MI、BCIC-IV-2a、ISRUC、BCIC2020-3、Mumtaz等数据集。

**📈 对比分析**

与单任务微调、多任务微调及多种持续学习基线（MMD、EWC、DER++等）对比，EvoBrain在保持 91–94% 单任务性能的同时显著降低遗忘率，平均 B‑ACC 超过 0.6，优于其它方法。

**⚠️ 局限性**

限制包括：任务边界已知、需要任务专属头、重放记忆带来隐私问题、尚未验证在线无任务识别或生成式重放等更开放场景。

---

## 427. Physics-Guided Attention in a Lightweight TCN for Efficient WiFi CSI-Based Human Activity Recognition

**arXiv ID:** 2606.01834 | [PDF](https://arxiv.org/pdf/2606.01834v1)

**作者:** Chinthaka Ranasingha `[一作]` (Queensland University of Technology), Harshala Gammulle `[通讯]` (Queensland University of Technology)

**通讯引用:** 630 | [OpenAlex ID](https://openalex.org/A5054311875)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种轻量化的时间卷积网络（TCN）框架，通过在特征空间中引入物理相关的运动先验来实现WiFi CSI信号的人类动作识别。

**💡 创新点**

创新点在于设计了两种基于CSI物理特性的注意力机制：基于多普勒能量的时序注意力与基于方差的通道注意力，直接嵌入特征学习流程，显著提升了模型对运动动力学的感知。

**🔧 技术方法**

技术包括：短时傅里叶变换（STFT）提取多普勒能量、温度缩放的softmax注意力、通道方差统计、轻量级瓶颈MLP、残差融合与全局平均池化等。

**📊 数据集**

在三大公开数据集上评估：CSI‑HAR、NTU‑Fi HAR 和 UT‑HAR，数据涵盖多种人体动作与多用户、多环境场景。

**📈 对比分析**

与传统深度CNN、RNN、Transformer等基准模型比较，本文模型在保持或超过准确率的同时，参数量减少 80%+、FLOPs 减少 70%+，尤其在 UT‑HAR 与 CSI‑HAR 数据集上实现了最高或近似最高准确率。

**⚠️ 局限性**

局限性包括：对不同信号采集设备或极端环境下的泛化能力尚未充分验证，且注意力机制仍依赖 STFT 预处理，可能对实时低延迟部署产生一定开销。

---

## 428. Unsupervised Collaborative Domain Adaptation for Driving Scene Parsing

**arXiv ID:** 2606.01818 | [PDF](https://arxiv.org/pdf/2606.01818v1)

**作者:** Jiahe Fan `[一作]` (Tongji University), Rui Fan `[通讯]` (Tongji University)

**通讯引用:** 4295 | [OpenAlex ID](https://openalex.org/A5038867899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无监督协作域适应框架（UCDA），在不访问源数据的前提下，将多源预训练模型的互补知识迁移到目标驾驶场景解析模型。

**💡 创新点**

创新点包括：①基于类别级原型记忆库的跨模型可靠性评估，消除独立模型间置信度尺度差异；②两阶段协作策略，先在目标域协同优化源模型，然后将验证后的专家知识蒸馏到单一目标模型；③在低置信度区域同时使用正负一致性约束，利用未标注区域的软信息提升鲁棒性。

**🔧 技术方法**

采用原型记忆、可靠性匹配、正负一致性损失、协同优化与知识蒸馏技术；网络架构以DeepLab‑V2+ResNet‑101和MiT‑B5为主。

**📊 数据集**

使用多源合成数据（GTA5、SYNTHIA、Synscapes）和真实数据（Mapillary Vistas、Cityscapes、BDD100K、ACDC），以及自研NIO车辆平台收集的真实驾驶数据。

**📈 对比分析**

在Cityscapes、BDD100K、ACDC、NIO四个目标域上与多种SOTA源自由/源依赖方法对比，UCDA在mIoU上均超过对手，特别在罕见类别和跨域泛化上表现显著提升。

**⚠️ 局限性**

局限性包括：对源模型多样性与质量的依赖，若源模型偏差大或单一；可靠性估计仍可能误选错误预测；多源协作训练时计算与内存成本上升，虽然推理成本不变。

---

## 429. Personalized 3D Myocardial Infarct Geometry Reconstruction from Cine MRI for Cardiac Digital Twins

**arXiv ID:** 2606.01808 | [PDF](https://arxiv.org/pdf/2606.01808v1)

**作者:** Yilin Lyu `[一作]` (National University of Singapore), Lei Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个全自动、无对比剂的三维心肌梗死（MI）重建框架GeoMo-Net，可直接从多视角慢动MRI生成可用于心脏数字双胞胎（CDT）模拟的3D梗死模型。

**💡 创新点**

创新点包括：① 明确解耦几何（形态）与运动（功能）特征，采用双分支图卷积网络实现自适应融合；② 通过多尺度AHA‑17段监督和交叉注意力实现空间一致性和临床可解释性；③ 在4D拓扑一致心室网格上直接预测梗死节点，突破传统2D分割或对比剂成像限制。

**🔧 技术方法**

技术手段主要有：nnU‑Net自动分割与基于可微分配准的4D心室网格重建；几何/运动特征提取（坐标、UVC、间帧位移、总位移、边界应变）；双分支图神经网络+Mamba时序模块+多尺度输出头；Dice+Focal+MSE联合损失；多尺度交叉注意力和全局池化实现节点级与段级融合。

**📊 数据集**

使用了内部数据集：129名急性MI患者共225份三视角慢动MRI与LGE对照（NUH Singapore），以及公开CMR‑MULTI数据集30份含梗死标签的病例进行外部验证。

**📈 对比分析**

与GAT、PointNet++、ST‑GCN、C2I‑Net等基线对比，GeoMo-Net在Dice、G‑Dice、敏感度、特异性、准确度和ASSD等指标均取得最高或接近最高（Dice 0.678±0.011，G‑Dice 0.739±0.011），在下游心电模拟中也能逼近LGE基线的功能表现。

**⚠️ 局限性**

局限性包括：① 仅预测梗死分布，无法恢复病灶的壁厚分布（即壁层深度）；② 基准标签来自稀疏2D LGE切片，配准误差可能影响边界；③ 外部验证受域迁移、视角缺失和高梗死负荷样本不足限制，需要更大规模、多中心、多协议的训练与评估。

---

## 430. MetaForge: A Self-Evolving Multimodal Agent that Retrieves, Adapts, and Forges Tools On Demand

**arXiv ID:** 2606.01801 | [PDF](https://arxiv.org/pdf/2606.01801v1)

**作者:** Shouang Wei `[一作]` (East China Normal University), Min Zhang `[通讯]` (East China Normal University)

**通讯引用:** 62731 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种自适应工具编排的多模态代理MetaForge，能够根据任务需求决定是否调用工具、检索适合的工具、调整调用参数，并在必要时在线生成并注册新的工具。

**💡 创新点**

创新点在于：①将工具使用拆解为四阶段闭环（决定-检索-适配-合成）并实现工具库的自我进化；②设计多奖励复合目标与GRPO训练，平衡答案正确性、工具检索/适配、调用必要性、合成可重用性；③采用容量约束与语义去重管理动态工具池，防止过大且冗余。

**🔧 技术方法**

主要技术包括：多模态大模型（Qwen3‑VL‑8B‑Instruct）、多轮策略学习（Group Relative Policy Optimization, GRPO）、两阶段工具合成（规划+代码实现）、工具验证（执行+语义判定）以及工具池管理机制。

**📊 数据集**

使用12个公开的图文推理基准（如DocVQA、TallyQA、OCRVQA、ChartQA、MathVista等），在IID与OOD工具分布下进行评测。

**📈 对比分析**

与16个基线（包括GPT‑5.4、Claude‑Sonnet‑4.6、Gemini‑3.1、Qwen系列、InternVL、VTool‑R1、R1‑Onevision等）对比，MetaForge在准确率、推理效率（平均交互轮数、延迟、token数）和工具使用可靠性（TUE、Tool SR、FSR）方面均领先或持平，尤其在OOD工具设置下保持更高的泛化性能。

**⚠️ 局限性**

局限性：①合成工具仅支持程序化任务（代码、表格、数值），对视频、音频、交互式场景支持不足；②实验仅覆盖图文任务，未验证在更丰富的多模态环境中的表现；③效率评估侧重延迟，未充分考虑调用成本与上下文消耗，实际部署需进一步平衡。

---

## 431. Multilinguality of Large Language Models From a Structural Perspective

**arXiv ID:** 2606.01800 | [PDF](https://arxiv.org/pdf/2606.01800v1)

**作者:** Haruki Sakajo `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1642 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过代表性结构分析研究LLM的多语种处理方式，构建跨语言的树结构并计算结构距离。

**💡 创新点**

创新点在于从整体结构视角（树结构）评估语言相似度，揭示低资源语言在结构上与英语的差异及语言特定后训练如何改变内部结构而保持跨语言关系。

**🔧 技术方法**

使用StructLens构建最大生成树、Tree Edit Distance (TED) 量化结构差异，并通过余弦相似度、困惑度、MT BLEU等传统指标进行对照。

**📊 数据集**

数据集包括 FLORES+ 的多语言平行句子和 Jinghpaw 专属数据，语言覆盖高/中/低资源、不同语系与脚本。

**📈 对比分析**

通过对 Gemma3 4B PT、Qwen3 8B Base、Olmo3 7B Base 等多语种与单语种模型进行层级树结构距离对比，发现多语种模型在高/中资源语言上结构相似度高，而低资源语言距离大；语言特定后训练显著增大目标语言的结构距离但不改变跨语言距离，表明结构视角能捕捉传统指标未能体现的差异。

**⚠️ 局限性**

主要局限在语言样本有限（仅含 7 种语言）、结构树仅为模型内部关系而非语法树、缺乏对训练数据的完整可视化，且对受限语言适应性的进一步验证需要更多语言与受控训练数据。

---

## 432. An Algebraic View of the Expressivity of Recurrent Language Models

**arXiv ID:** 2606.01765 | [PDF](https://arxiv.org/pdf/2606.01765v1)

**作者:** Franz Nowak `[一作]`, Reda Boumasmoud `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

建立统一的代数框架，将递归语言模型的表达能力归结为层级变换单子（wreath product）的可除性问题，并用代数方法解释数值离散化对表达能力的影响。

**💡 创新点**

关键创新在于将算术语义视为可定制的对象，统一前期关于RNN可判定性与Turing完备性的冲突结论，并通过代数分析揭示浮点与整数量化下的表达差异。

**🔧 技术方法**

采用代数理论（单子、变换单子、wreath product、Krohn–Rhodes分解）、递归一致的评估策略以及离散算术模型进行形式化推导。

**📊 数据集**

无（该论文为理论工作，不使用数据集）。

**📈 对比分析**

无实验比较，主要通过代数归约与可除性链得到理论结论。

**⚠️ 局限性**

局限在于仅适用于满足递归一致评估的算术模型；对更复杂架构（如注意力机制）需进一步形式化。

---

## 433. Quality-Guided Semi-Supervised Learning for Medical Image Segmentation

**arXiv ID:** 2606.01753 | [PDF](https://arxiv.org/pdf/2606.01753v1)

**作者:** Kumar Abhishek `[一作]` (Simon Fraser University), Ghassan Hamarneh `[通讯]` (Simon Fraser University)

**通讯引用:** 12112 | [OpenAlex ID](https://openalex.org/A5072684302)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于质量预测的半监督医学图像分割框架，利用一个独立训练的质量评估网络为无标签数据提供学习信号；

**💡 创新点**

创新点在于（1）首次将学习得到的质量预测作为框架无关的正则化或样本加权模块融入任何半监督方法；（2）设计了结合弱模型预测的多样化腐蚀策略，弥合合成误差与真实网络误差的分布差距；

**🔧 技术方法**

使用ResNet-18编码器的回归网络进行质量预测，采用随机腐蚀与弱模型输出混合生成训练数据；在半监督阶段应用质量感知正则化（QAR）或质量加权伪标签（PL-QW）；

**📊 数据集**

实验涵盖五个医学图像分割数据集（皮肤病变：PH2、SCD、DMF；结肠镜息肉：COL、CLI）以及来自不同来源的大量无标签数据（ISIC2020、Polyp-Box-Seg）；

**📈 对比分析**

与五大类半监督方法（一致性正则化、伪标签、对比学习等）对比，QAR/PL-QW在所有数据集和三种主干网络（UNet++、Attention-UNet、Swin-Unet）上均实现了Dice/Jaccard指标的显著提升；

**⚠️ 局限性**

局限性包括仅针对二分类分割、对多类别场景尚未验证、质量预测在极端低质量样本时的鲁棒性有限，以及实验主要基于公开数据集，真实临床迁移性待进一步评估。

---

## 434. Sensitivity as a Double-Edged Sword: A Trade-off Between Discriminability and Adversarial Robustness

**arXiv ID:** 2606.01746 | [PDF](https://arxiv.org/pdf/2606.01746v1)

**作者:** Kai Wang `[一作]` `[通讯]`, Kai Wang

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Hybrid Prototype Mixing (HPM) 框架，结合全连接(FC)与L2距离原型进行鲁棒重分类，并设计了Mixed Surrogate Attack (MSA) 的评估协议；

**💡 创新点**

创新点在于通过融合数据集级与批级原型，同时利用L2距离提高抗干扰性并保持判别力；以及针对动态模型梯度障碍提出的MSA评估方法；

**🔧 技术方法**

使用了EMA更新、Straight-Through Estimator、线性混合器、BPDA、AutoAttack/PGD、Square Attack以及RobustBench评估工具；

**📊 数据集**

在CIFAR-10、CIFAR-100和ImageNet子集上进行实验；

**📈 对比分析**

通过对比原始SOTA模型在相同攻击（AutoAttack+MSA+Square）下的鲁棒准确率，实验显示在所有选取模型上均提升若干百分点；

**⚠️ 局限性**

局限性包括：在大类别数或高维场景下L2原型的判别能力有限；动态原型生成仍受梯度障碍影响；仅在ImageNet子集验证，缺乏全数据集验证。

---

## 435. Residual Decoder Adapter: ID-Preserving Tokenizer Adaption for Autoregressive Text Rendering

**arXiv ID:** 2606.01911 | [PDF](https://arxiv.org/pdf/2606.01911v1)

**作者:** Dongxing Mao `[一作]` (Central South University), Jingru Tan `[通讯]` (Central South University)

**通讯引用:** 1051 | [OpenAlex ID](https://openalex.org/A5031272540)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后置残差解码器适配器（RDA），可在不重新训练视觉自回归模型或分词器的前提下显著提升文本渲染质量。

**💡 创新点**

创新点在于：①共享ID提示码表，使得新辅助码表与原始码表保持相同离散分布；②残差解码器通过像素级残差学习恢复细节；③实现插件化、可无缝集成至任何已有AR模型。

**🔧 技术方法**

技术手段包括：视觉自回归框架、VQ-VAE分词器、共享ID提示码表、残差解码网络、MAE/MSE、感知损失、Sobel 边缘损失、频域损失以及实例相关特征注入。

**📊 数据集**

训练数据：Mario-10M（去重后5M张图）；评测数据：AnyText-Benchmark、Mario-Eval、LongTextBench、CVTG-2K、TextAtlasEval、StyledTextSynth。

**📈 对比分析**

与原模型、其他分词器（LlamaGen-VQ、Chameleon-VQ、TA-Tok、UniTok）比较，RDA在OCR准确率、F1、SSIM、LPIPS等指标上提升显著（如Janus‑Pro 1B OCR准确率从24.52%提升至58.26%，泛化至更高分辨率时性能保持稳定）。

**⚠️ 局限性**

局限性：仍依赖原始分词器的离散表示，无法解决AR模型在文本预测层面的错误；对非文本领域的细节提升有限；对极端长文本的恢复效果尚待验证。

---

## 436. Auto formalisation of Goedel's Second Incompleteness Theorem in Binary Recursive Arithmetic

**arXiv ID:** 2606.01898 | [PDF](https://arxiv.org/pdf/2606.01898v1)

**作者:** Thierry Coquand `[一作]` (University of Gothenburg), Thierry Coquand `[通讯]` (University of Gothenburg)

**通讯引用:** 5461 | [OpenAlex ID](https://openalex.org/A5087100539)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文使用 Claude 进行交互式自动化，将哥德尔第二不完全性定理在 Agda 中完全形式化，整个开发约 5 万行代码，未使用任何人工编写的 Agda 代码；同时对 Guard 1963 年讲义中的隐含细节进行了显式化、纠正和补充；并验证了从内部可证明的 Gödel 定理到外部可证明的一致性蕴含的完整证明；进一步探讨了 LLM 在数学形式化中的局限性与潜力。

**💡 创新点**

创新点在于：1) 完全基于大型语言模型完成非平凡的形式化工作，展示了 LLM 在数学证明自动化中的实际可行性；2) 明确梳理并实现了 Guard 文献中未写出的关键细节（如内部 numeral‑inertness、predicate‑Leibniz、Carneiro 的提升、验证器的解码器不变性等）；3) 通过单变量替换+数值闭合的技巧，消除多变量替换所需的复杂性，降低证明成本；4) 系统性评估了 LLM 产生错误规范导致错误形式化的风险，为未来的自动形式化提供经验。

**🔧 技术方法**

采用的技术包括：Agda 证明助手；大型语言模型 Claude 与交互式提示（prompting）；内部化的编码与解码器实现；课程值递归（course‑of‑values recursion）在 Church 递归器中的实现；内部化的对角线引理、单变量替换与数值闭合 lemmas；Carneiro 的提升技巧实现 Hilbert 系统下的条件化推理；predicate‑Leibniz 替换规则；以及对 Guard 1963 讲义的形式化翻译与纠错。

**📊 数据集**

本研究不涉及传统意义上的数据集；所使用的“数据”是 Guard 1963 年的证明笔记和 Agda 代码库（约 5 万行）。

**📈 对比分析**

由于该工作为完全形式化的数学证明实现，未进行性能比较实验；重点是证明完整性与可复现性，强调 LLM 与人工交互在证明规模、代码量、错误率等方面的可比性。

**⚠️ 局限性**

局限性：1) 结果高度依赖于 LLM 的规范理解，错误的规范会导致完全错误的形式化；2) 目前缺乏自动化验证 LLM 生成规范正确性的机制；3) 该方法在更大或更复杂的数学领域（非组合式结构）中的可迁移性尚未验证；4) 由于缺少正式的评价指标，难以客观衡量 LLM 交互式形式化的效率提升。

---

## 437. Mos-Gen: A Generative Molecular Framework for Mosquito Insecticide Design

**arXiv ID:** 2606.01846 | [PDF](https://arxiv.org/pdf/2606.01846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 438. EVA-Net: Subject-Independent EEG Motor Decoding with Video-Derived Motor Priors

**arXiv ID:** 2606.01884 | [PDF](https://arxiv.org/pdf/2606.01884v1)

**作者:** Ziyuan Li `[一作]` (South China University of Technology), Yimeng Zhang `[通讯]` (South China University of Technology)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5100439305)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种基于动作视频先验的双阶段EEG运动解码框架EVA-Net，在训练阶段通过跨模态对齐与先验迁移，最终实现仅用EEG进行的无校准运动解码。

**💡 创新点**

①将连续时空动作视频作为语义锚点替代传统文本；②采用跨模态对齐与监督对比学习降低个体噪声；③使用视频类别原型与知识蒸馏将视频先验迁移至EEG分类器，保持推理时仅EEG。

**🔧 技术方法**

使用EEG Conformer编码器、VideoMAE自监督视频编码器、双向对比损失、监督对比损失、视频原型聚类、知识蒸馏以及标签平滑交叉熵等技术。

**📊 数据集**

在EEG Motor Movement/Imagery (EEGMMI)、BCI Competition IV Dataset 2a (BCIC-IV-2a) 两个公开EEG数据集以及自建的5类动作视频数据集上进行实验。

**📈 对比分析**

与EEGNet、ShallowConvNet、EEGCCT、EEG Conformer、MSVTNet等5个基线进行对比，在Leave-One-Subject-Out (LOSO) 下，EEGMMI的准确率提升至72.30%（比最佳基线高8.66个百分点），BCIC-IV-2a的准确率提升至71.25%（比最佳基线高3.35个百分点），并在其他交叉验证模式亦保持领先。

**⚠️ 局限性**

仍受EEG信号非稳态与个体差异影响，某些受试者出现混淆；需要离线获取动作视频作为先验，限制了实时应用；跨模态对齐与视频先验的有效性依赖于视频标签质量与多样性。

---

## 439. Deep Learning for Generating Computational PIN-4 Immunohistochemistry Staining from Prostate Biopsy H&E Images

**arXiv ID:** 2606.01871 | [PDF](https://arxiv.org/pdf/2606.01871v1)

**作者:** Vietbao Tran `[一作]` (University of California Irvine), Pratik Shah `[通讯]` (University of California Irvine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并训练了一个基于 Pix2Pix 的 cGAN 模型，利用常规 H&E 病理切片生成对应的 PIN-4 IHC 免疫染色图像，实现 H&E 到 PIN-4 的计算性染色。

**💡 创新点**

首次在临床常规获取的配对、已注册的 H&E/PIN-4 整幅切片数据集上实现监督式 PIN-4 合成；开发了核心级注册与质量控制流水线，并证明可在不使用高光谱或荧光数据的情况下，直接从 RGB H&E 预测 PIN-4 的空间对应染色。

**🔧 技术方法**

使用 VALIS+DISK+LightGlue+RANSAC 进行图像配准；采用 Pix2Pix 结构，改进上采样方式、加入 SSIM、PCC 损失；训练时采用 AMSGrad Adam、随机翻转、90° 旋转；评估指标包括 PSNR、SSIM、PCC、LPIPS。

**📊 数据集**

数据集来自 UCI Health，包含 172 对已配准的 H&E/PIN-4 全切片图像（93 名患者），共 27,298 个 1024×1024 级别的配对补丁，涵盖不同年龄、种族、族裔及腺癌阳性/阴性病例。

**📈 对比分析**

在留出测试集（1,814 对）上与真实 PIN-4 进行比对：PSNR 21.88 dB，SSIM 0.667，PCC 0.684，LPIPS 0.417；与之前 H&E→HER2/GPC3 的转化结果相比，结构相似度更高；病理学家定性评估表明生成图像在诊断相关染色模式上具有可用性。

**⚠️ 局限性**

局限性包括单中心回顾性数据、对极复杂病变（如高级别癌、腔内癌）的合成效果有限、配准误差残余影响、模型生成的染色并不能替代原始 IHC 用于最终诊断。

---

## 440. The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space

**arXiv ID:** 2606.01847 | [PDF](https://arxiv.org/pdf/2606.01847v1)

**作者:** Bing-Cheng Chuang `[一作]` (National Taiwan University), Chun-Yi Lee `[通讯]` (National Taiwan University)

**通讯引用:** 3057 | [OpenAlex ID](https://openalex.org/A5028600832)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在Lie群SE(3)上直接进行扩散学习的视觉语言动作策略框架

**💡 创新点**

通过在切空间进行分数匹配并使用指数映射重投，纠正了传统方法中将SE(3)嵌入为欧氏向量导致的几何误差

**🔧 技术方法**

左不变SDE、分数匹配、指数映射、SE(3)扩散

**📊 数据集**

CALVIN机器人仿真数据集以及真实机器人实验

**📈 对比分析**

与基线扩散策略对比，平均任务长度提升7.3%，在多项任务上表现优于基线

**⚠️ 局限性**

仍需在更复杂场景和大规模数据上验证，且对实时性与计算开销有一定限制

---

## 441. Evaluation of Baseline Methods for IDD-based SSD External Memory Search

**arXiv ID:** 2606.01840 | [PDF](https://arxiv.org/pdf/2606.01840v1)

**作者:** Yuki Suzuki `[一作]` (University of Tokyo), Alex Fukunaga `[通讯]` (University of Tokyo)

**通讯引用:** 6565 | [OpenAlex ID](https://openalex.org/A5052263300)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个基于外部存储的简单即时重复检测（IDD）算法，该算法采用分离链（Separate Chaining）哈希表实现外部记录链，并通过最小化对RAM的占用来实现高效搜索；

**💡 创新点**

创新点在于证明仅使用分离链的基本实现即可在外部内存搜索中达到与现有最先进方法A*-IDD相当的性能，同时系统地揭示了操作系统页缓存与文件系统层对外部内存搜索性能的影响；

**🔧 技术方法**

主要技术包括：外部存储的分离链哈希表、开放寻址对比、内存映射（mmap）与显式文件I/O（open/OA）两种I/O接口、直接I/O（Direct I/O）绕过页缓存、内存后备文件系统（memfs、ramdisk）用于评估系统层开销；

**📊 数据集**

使用了来自Autoscale基准套件的STRIPS规划实例，挑选了高内存压力（1 GiB）与低内存压力（32 GiB）两组测试集，涵盖Blind、Merge-and-shrink等两种启发式；

**📈 对比分析**

通过节点展开率（states/sec）与峰值内存使用对比，在高内存压力下，所提分离链实现的IDDA在最优节点扩展率上与A*-IDD相当，且在低内存压力下可接近纯RAM Fast Downward的速率；在无页缓存或直接I/O情况下，发现系统层（块驱动、文件系统）开销显著；

**⚠️ 局限性**

局限性包括：仅评估了最简实现，未加入压缩或用户级缓存等高级优化；实验仅涵盖STRIPS规划任务，未验证在其他搜索领域的适用性；对多线程或分布式环境的适配尚未研究；

---

## 442. Suppressing Forgery-Specific Shortcuts for Generalizable Deepfake Detection

**arXiv ID:** 2606.01843 | [PDF](https://arxiv.org/pdf/2606.01843v1)

**作者:** Yihui Wang `[一作]` (Hefei University of Technology), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 62129 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过子空间建模显式识别并抑制深度伪造检测中的方法特定捷径，提升跨方法泛化能力。

**💡 创新点**

提出两种补偿策略：训练时的子空间投影抑制（NSP）与推理时的神经元激活编辑（NAE），首次将方法特定捷径视作低维子空间并进行直接抑制。

**🔧 技术方法**

使用轻量级线性探针 + 奇异值分解提取方法敏感子空间，随后在梯度投影和神经元抑制两阶段进行抑制。

**📊 数据集**

在DF40深度伪造数据集（包含40种伪造方法、4个类别）上进行实验，并在FaceForensics++的训练/验证/测试划分上评估。

**📈 对比分析**

与12种现有基线（Xception、ResNet、EfficientNet、CLIP及多种专用检测器）对比，NSP/NAE在跨方法测试中平均提升10–15% AUC，且在主流网络上表现一致，只有1–2% 近似不变的在域内损失。

**⚠️ 局限性**

对超参数（子空间秩、抑制比例）敏感；两种策略不一定互补，组合时可能出现过度抑制导致性能下降；且未考虑多模态或时间序列伪造场景。

---

## 443. Bayesian Spectral Emotion Transition Discovery from Multi-Annotator Disagreement

**arXiv ID:** 2606.01906 | [PDF](https://arxiv.org/pdf/2606.01906v1)

**作者:** Keito Inoshita `[一作]` (Kansai University), Takato Ueno `[通讯]` (Shiga University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了BSETD框架，通过双阶段方法利用多评审者的软标签发现对话情绪转移结构；

**💡 创新点**

创新点在于将注释者不一致性保留为概率信息，利用层级Dirichlet–Multinomial后验与图谱谱分解，将情绪惯性与传染性分离；

**🔧 技术方法**

使用了层级Dirichlet–Multinomial（EB浓度估计）外积计数、Benjamini–Hochberg FDR控制、对称图拉普拉斯谱分解；

**📊 数据集**

实验数据涵盖EmotionLines（5评审）、MELD、DailyDialog、中文M3ED及GPT‑5.4‑mini虚拟注释，覆盖多语言与软标签来源；

**📈 对比分析**

与传统多数投票、硬标签方法及已有ERC模型对比，BSETD在跨语料、跨语言相关性均达到0.79–0.98，能重现Plutchik与Russell的情绪邻接/反转模式；

**⚠️ 局限性**

局限在于跨对话异质性未完全捕捉、谱分解对方向信息的丢失、以及对注释者相关性的假设。

---

## 444. KliniskVestBERT: BERT Model Specialised to Norwegian Clinical Texts

**arXiv ID:** 2606.01904 | [PDF](https://arxiv.org/pdf/2606.01904v1)

**作者:** Christian Autenried `[一作]` (Helse Vest IKT), Cosimo Persia `[通讯]` (Helse Vest IKT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对挪威临床文本，利用16.2M de-identified 文档对三种BERT基编码器进行领域预训练，并在五个临床NLP基准上评估性能。

**💡 创新点**

首次结合大规模真实临床语料进行BERT预训练，提供专门针对挪威两种书面标准的临床模型，并展示显著优于通用预训练模型的效果。

**🔧 技术方法**

采用Transformer BERT架构，进行masked language modeling、span MLM和下一句预测等预训练任务，并使用AdamW/StableAdamW优化器与梯度累积。

**📊 数据集**

使用来自Helse Vest 16.2M de-identified临床文本以及MedMCQA、Medical Question Pairs、Nor-DeID-SynthData、急诊呼叫与药物识别等五个公开/内部基准数据集。

**📈 对比分析**

通过在相同训练/验证/测试拆分下，比较新预训练模型与原始BERT、BERT-base、ModernBERT的F1/准确率，结果显示所有临床模型在三类synthetic数据集和两类真实任务上均优于基线，表现提升可达10–20%。

**⚠️ 局限性**

受限于仅在挪威语文料上训练，模型可能不具备跨语言泛化；并且未公开内部真实数据的评测细节，外部复现受限。

---

## 445. Auteur: Language-Driven Cinematographic Framing for Human-Centric Video Generation

**arXiv ID:** 2606.01900 | [PDF](https://arxiv.org/pdf/2606.01900v1)

**作者:** Muhammed Burak Kizil `[一作]` (Koc University), Duygu Ceylan `[通讯]` (Adobe)

**通讯引用:** 4748 | [OpenAlex ID](https://openalex.org/A5068985412)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于语言的、以人类为中心的摄像机框架控制方法，能够将自然语言描述转化为演员运动和摄像机框架关键帧，随后生成连续的6-DoF摄像机轨迹并用于视频生成。

**💡 创新点**

创新点在于：①引入人类中心摄像机参数化，将摄像机状态定义为相对于演员的框架变量；②设计可被大语言模型生成的DSL程序，实现离散化、可解释的摄像机计划；③采用两阶段LLM“导演”流程，先生成粗略演员轨迹，再生成摄像机关键帧，从而实现摄像机对演员动作的主动响应。

**🔧 技术方法**

使用的技术包括：Qwen-2.5-VL作为多模态LLM导演；DSL离散化与向量嵌入；基于SOMA人体模型的演员姿态与运动生成；几何解码器将DSL状态映射为世界空间6-DoF轨迹；与VerseCrafter、Kimodo+VACE等视频生成模型的接口。

**📊 数据集**

训练数据来自两部分：①基于SOMA的合成剧本与DSL摄像机程序生成的人工数据；②从CondensedMovies影片中提取的真实3D人体与摄像机轨迹、字幕（AuroraCap）与DSL注解。

**📈 对比分析**

在自定义的“Controlled Diagnostic Benchmark”与公开的PulpMotion基准上进行对比。方法在所有框架相关指标（F-Ori、F-RoT、F-Scale、F-Tilt、F-Roll、Auteur Score）均明显优于LAMP和PulpMotion，并将out‑rate降至约1%，远低于对手的10%+；在PulpMotion纯/混合设置中亦获得最高的Camera CLaTr分数和最小的out‑rate。

**⚠️ 局限性**

局限性包括：DSL关键帧表示假设摄像机意图可稀疏离散化，难以捕捉连续或手持式的动态拍摄；仅建模粗略演员轨迹，未包含精细关节动画；最终视频质量仍受下游生成模型限制。

---

## 446. Train, Test, Re-evaluate: Schedule-Sensitive Evaluation of Generative Data for Hand Detection

**arXiv ID:** 2606.01896 | [PDF](https://arxiv.org/pdf/2606.01896v1)

**作者:** Atmika Bhardwaj `[一作]` (Federal Institute for Occupational Safety and Health), Nico Steckhan `[通讯]` (Federal Institute for Occupational Safety and Health)

**通讯引用:** 2679 | [OpenAlex ID](https://openalex.org/A5088871320)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用生成式填充技术在真实手部图像上添加护具、纹身等配件，并在多阶段训练策略下提升YOLOv8n手部检测器性能。

**💡 创新点**

提出两阶段增量训练与三阶段课程学习方案，通过在真实+合成数据上预训练后再细调、递减学习率，显著缩小裸手与配饰手的分布差距。

**🔧 技术方法**

使用SemProbe+FLUX.2实现局部语义填充，YOLOv8n进行目标检测，MediaPipe Hands、PSNR/SSIM/LPIPS等指标评估合成图像质量，配合统计显著性检验与跨域缺口分析。

**📊 数据集**

基于德国BAuA公开的手部图像数据集（6,507训练、3,591测试）生成6,507幅配饰手合成图像，形成配对训练/测试集。

**📈 对比分析**

通过mAP@0.5、mAP@0.5:0.95、精确率/召回率等指标对比不同实验，结果显示两阶段训练提升8.7点mAP@0.5，三阶段训练实现最高mAP@0.5:0.95，并将手部配饰的OOV误差降低约18点。

**⚠️ 局限性**

仍存在严格IoU框紧致度不足，合成图像边界可能出现可见缝隙，且实验仅用3个随机种子，难以充分覆盖训练不确定性；生成管线需进一步改进以提升手部细节真实感。

---

## 447. Collaborative Space Object Detection with Multi-Satellite Viewpoints in LEO Constellations

**arXiv ID:** 2606.01895 | [PDF](https://arxiv.org/pdf/2606.01895v1)

**作者:** Xingyu Qu `[一作]` (University of Manitoba), Peng Hu `[通讯]` (University of Manitoba)

**通讯引用:** 72163 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了多视角观测融合在低地球轨道星座中利用深度学习提升太空目标检测的方案。

**💡 创新点**

提出了基于早期融合的多视角输入管线，可将多视角图像拼接为多通道输入并直接用于YOLOv9，实现轻量化多视角检测，同时验证RGB与灰度多视角输入的有效性，并评估通信与推理成本。

**🔧 技术方法**

采用YOLOv9（含t/m版和GELAN-t）、通道拼接早期融合、灰度转换、Grad‑CAM++可视化等技术。

**📊 数据集**

使用SOD_Clustering数据集（180幅三视角图像，80/20划分）。

**📈 对比分析**

与单视角RGB/灰度基线对比，采用mAP50和mAP50-95评估；多视角在YOLOv9‑m无预训练时mAP50提升≈15%，mAP50‑95提升≈22%；灰度多视角提升更大（≈36%/46%）；推理时间略增，通信延迟极低。

**⚠️ 局限性**

仅在有限的三视角样本与小规模数据集上验证；改动仅为输入通道，未探究更复杂的融合结构；未评估真实星座多视角同步与网络拥塞情况。

---

## 448. MidSurfNet: Learnable Face Pairing and Interference Implicit Fields for Generalized Mid-surface Abstraction

**arXiv ID:** 2606.01891 | [PDF](https://arxiv.org/pdf/2606.01891v1)

**作者:** Li Ye `[一作]` (Zhejiang University), Min Tang `[通讯]` (Zhejiang University)

**通讯引用:** 6585 | [OpenAlex ID](https://openalex.org/A5077671701)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了一种学习增强的薄壁 CAD 模型中面抽象框架，结合可学习的面配对模块和干涉隐式场实现对中面的自动提取；

**💡 创新点**

创新点包括：①基于图神经网络的学习面配对，克服多壁厚与自匹配等规则基方法的缺陷；②干涉隐式场可在任意 α∈[0,1] 位置上提取中面，实现可调偏移；

**🔧 技术方法**

采用了 Transformer‑based 图神经网络、FiLM 条件模调、Topo/位置编码以及阶层化分类的深度隐式场网络；

**📊 数据集**

使用 MidSurf 数据集，涵盖 1,575 个 CAD 模型，提供面配对与隐式场标注；

**📈 对比分析**

与 Woo、MidSurfer、CAT、Zhu、Parasolid 等现有方法对比，面配对准确率 87.32%（比 MidSurfer 提升 23%），多壁厚场景完成率 61.90%，隐式场 MAE 0.0052，1% 误差率 91%；

**⚠️ 局限性**

主要局限包括 O(N²) 记忆消耗、对非薄壁区无法处理、小面噪声导致配对误差、极端 α 位置的精度下降，需进一步研究稀疏注意、N‑N 配对及混合几何方法。

---

## 449. Beyond the Simplex: Balanced Prototype Geometry for Scorer-Agnostic Open-Set Recognition

**arXiv ID:** 2606.01883 | [PDF](https://arxiv.org/pdf/2606.01883v1)

**作者:** Mayank Sharma `[一作]` (Indian Institute of Technology Jodhpur), Rohit Kumar Mourya `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套通用的理论框架，证明了在任何嵌入维度下，基于简单形比率（simplex-ratio）的开放集识别（OSR）方法均能保持良好的拒绝几何与误拒率上界；

**💡 创新点**

创新点在于将原先只适用于正交正等距（simplex）构造扩展到任意维度，提出平衡等模数（balanced equal‑norm）代码并给出“单一距离对称性”与“简单形缺陷”参数的精确二分，阐明了低维下几何退化的原因，并给出了指数级误拒率上界与阈值选取的闭式规则；

**🔧 技术方法**

核心技术包括：全局 Lipschitz 分析、平衡等模数代码下的球面几何描述、单一距离对称性定理、简单形缺陷参数 λ、误拒率上界的球帽概率估计、UCDSC 损失的梯度解释、以及基于验证集的阈值与 σ 估计算法；

**📊 数据集**

实验使用 CIFAR‑10/50/100、BloodMNIST、PathMNIST 等自然与医学图像开集基准，采用 ResNet‑18 8‑维嵌入，并对比 Simplex、Harmonic、Learned、CE 四种表征；

**📈 对比分析**

方法在保持理论性质的同时，通过与 KNN、MSP、ODIN、Energy 等多种后置评分器的对照实验验证，发现平衡等模数/学习型表征在低维下能实现可接受的误拒率，但在大多数数据集上原始比率评分器的 AUROC 与 OSCR 仍落后于基于 Logit 或 KNN 的方法；

**⚠️ 局限性**

局限性包括：对分布假设（A5 的子高斯尾）和等模数假设的依赖；误拒率上界在实践中可能过于保守；在极端低维/高类数情形下平衡等模数代码的对称性退化导致性能下降；以及未能完全消除后置评分器与比率评分器之间的性能差距。

---

## 450. CultureForest: Understanding and Evaluating Cultural Norm Grounded Reasoning in LLMs

**arXiv ID:** 2606.01879 | [PDF](https://arxiv.org/pdf/2606.01879v1)

**作者:** Yangfan Ye `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 17173 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Cultural Norm Grounded Reasoning基准，评估LLM在现实场景中利用文化规范进行推理的能力；

**💡 创新点**

核心创新在于：①将每个问题与一组原子文化规范对齐，构建可验证、可归因的评估；②引入从易到难的三种难度级别（多选、二分类、生成），覆盖从指导到开放式情境；③开发轻量级Verifier实现可扩展的开放式生成评估；

**🔧 技术方法**

使用了LLM（如GPT-4、Llama、Qwen等）进行推理；构建Verifier基于Qwen3.5-0.8B-Base；采用生成式QA pipeline（GPT-4.1）及多模态文本嵌入、GMM聚类等技术；

**📊 数据集**

数据来源为Cultural Atlas的专家验证规范，覆盖53个国家/地区、8个日常生活域，共5378条实例；每条实例含3条原子规范；

**📈 对比分析**

对比方法包括：多选/二分类/生成三种评估；使用Mean@3、Pass@3、标准差与Gap衡量跨地区差异；实验显示顶级模型在Easy模式下表现近满分，但在Medium/Hard模式下降明显，且跨地区差异扩大；

**⚠️ 局限性**

局限性包括：①当前评估仍依赖模型的知识回忆能力，无法完全排除知识缺失导致的误差；②Verifier虽然轻量但对极端生成仍可能误判；③不同文化规范的“严格度”估计仍以人工标签为基础，可能存在主观偏差；④开放式生成的评估仍受模型温度、长度限制等技术参数影响。

---

## 451. Improving LLM-Based Go Code Review through Issue-List Generation and Context Augmentation

**arXiv ID:** 2606.01859 | [PDF](https://arxiv.org/pdf/2606.01859v1)

**作者:** Kexin Sun `[一作]` (Nanjing University), Christoph Treude `[通讯]` (Singapore Management University)

**通讯引用:** 5359 | [OpenAlex ID](https://openalex.org/A5077658936)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过引入问题列表生成、邻近/语义/相似上下文增强以及改进候选整合与剪枝，提升LLM驱动的代码审查效果。

**💡 创新点**

① 用问题列表代替单一主要问题生成；② 结合LSP语义与IR相似上下文形成多模态上下文；③ 采用改进的候选合并与基于改进预测的剪枝策略。

**🔧 技术方法**

利用DeepSeek‑V3 LLM、Go语言服务器（LSP）进行语义查询、IR检索相似变更、CodeReviewer模型进行代码改进预测以及Prompt工程。

**📊 数据集**

基于GitHub Go PR 的CodeReviewer基准，重构得到1,438条审查实例（来自54个仓库、2,786条变更）。

**📈 对比分析**

对比CodeReviewer及人工oracle，使用RefineEM、RefineBLEU、ReviewBLEU、ReviewBERT等指标；最佳配置实现28.00% RefineEM（+10.85pp）远超CodeReviewer（15.02%），接近人类oracle（36.09%）。

**⚠️ 局限性**

实验仅覆盖Go语言和单一LLM，评估基于模型改进预测；未验证多语言或工业环境，缺乏对问题优先级和上下文组织的深入探讨。

---

## 452. Parallelizing Large-Scale Tensor Network Contraction on Multiple GPUs

**arXiv ID:** 2606.01852 | [PDF](https://arxiv.org/pdf/2606.01852v1)

**作者:** Feng Pan `[一作]` (Singapore University of Technology and Design), Xipeng Li `[通讯]` (NVIDIA Corporation)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种多 GPU 的张量网络（TN）精确收缩框架，通过将固定收缩路径转换为通信高效的分布式执行计划，实现了显著的加速。

**💡 创新点**

创新点在于（1）利用 GEMM‑导向的模式重排，将每一步收缩映射为矩阵乘法布局，消除运行时转置；（2）基于动态规划的通信感知模式分布规划，智能决定哪些模式分区、何时重分布，以最小化通信成本；（3）将上述规划与 cuTENSORMp 的计算‑通信重叠结合，形成完整的端到端执行流水线。

**🔧 技术方法**

技术手段包括：cuTENSORMp（分布式张量收缩库）、NCCL 集合通信、NVLink/InfiniBand 高带宽互连、动态规划成本模型、GEMM 重排、双缓冲与分块传输。

**📊 数据集**

使用的数据集涵盖四大类：
- 量子电路模拟（Zuchongzhi n60m24）
- 量子错误校正（旋转表面码距离 7）
- 组合优化（King’s 子图独立集）
- 量子多体动力学（矩形、六边形、三角格子）

**📈 对比分析**

对比方法：与理想的 embarrassingly parallel 切片划分做对比，报告 projected full‑contraction 速度提升及超线性加速（extra speedup）。在单节点 8 GPU 时，extra speedup 介于 7×–173×；在 1024 GPU 规模时，extra speedup 达到 41.8×–67,869×。整体加速与切片并行的 1×–1024× 基线相比，显示出显著的通信优化收益。

**⚠️ 局限性**

局限性包括：
- 依赖于路径优化器；不同 GPU 数量下的路径质量差异导致性能非单调。
- 在跨节点的 InfiniBand 互连上，通信开销成为瓶颈，导致相较于 NVLink 的加速更受限。
- 对高维、超大规模张量网络仍需更高带宽的互连或进一步的通信压缩技术。

---

## 453. Echo: A Joint-Embedding Predictive Architecture for Speaker Diarization and Speech Recognition in a Shared Latent Space

**arXiv ID:** 2606.01909 | [PDF](https://arxiv.org/pdf/2606.01909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 454. Does Compression Preserve Uncertainty? A Unified Benchmark for Quantized and Sparse LLMs via Conformal Prediction

**arXiv ID:** 2606.01850 | [PDF](https://arxiv.org/pdf/2606.01850v1)

**作者:** Yujia Tong `[一作]` (Wuhan University Of Technology), Jingling Yuan `[通讯]` (Wuhan University Of Technology)

**通讯引用:** 1160 | [OpenAlex ID](https://openalex.org/A5062853168)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个压缩LLM不确定性评估基准，使用分布无关的 conformal prediction 对 12 种不同规模 LLM 进行量化与剪枝压缩后多任务性能与不确定性测评。

**💡 创新点**

创新点在于首次将 conformal prediction 应用于压缩模型的可靠性评估，并揭示压缩可导致准确性与不确定性解耦、模型规模缓冲压缩引起的不确定性膨胀以及阈值式不确定性增长的现象；同时提供统一的多压缩范式评估框架。

**🔧 技术方法**

主要技术包括：分布无关 conformal prediction（LAC 与 APS 评分函数）、多种量化方法（W4A16: RTN, AWQ, GPTQ；W4A4: QuaRot, SpinQuant, FlatQuant）、多种剪枝方法（无结构：Magnitude, SparseGPT, Wanda；有结构：LLM‑Pruner, SliceGPT），以及统一的评价指标（准确率、覆盖率、预测集大小）。

**📊 数据集**

使用了五个多项选择 NLP 任务的数据集：MMLU、CosmosQA、HellaSwag、HaluDial、HaluSum，均标准化为六选项。

**📈 对比分析**

通过在相同 calibration/test 分割下比较压缩前后模型的准确率、覆盖率与预测集大小，结果表明：量化（尤其是 W4A16）往往能保持接近 FP16 的准确率和不确定性；而激活量化（W4A4）与剪枝往往显著增加预测集大小；更大规模模型在同一压缩力度下不确定性提升更小。

**⚠️ 局限性**

局限性包括：仅针对多项选择任务，未覆盖自由文本生成；固定 α=0.1 以及 50/50 calibration-test 比例；未考虑硬件加速差异对量化/剪枝效果的影响；仅评估单一压缩配置，未探究压缩与模型微调、融合等组合策略。

---

## 455. Unveiling the Limits of Large Language Models in Inferring Pragmatic Meaning from Non-Verbal Responses

**arXiv ID:** 2606.01845 | [PDF](https://arxiv.org/pdf/2606.01845v1)

**作者:** Sugyeong Eo `[一作]` (Yonsei University), Heuiseok Lim `[通讯]` (Korea University)

**通讯引用:** 50089 | [OpenAlex ID](https://openalex.org/A5027447596)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统评估大型语言模型（LLM）在仅由非语言回应（沉默、面部表情、身体动作）构成的对话中推断隐含意图的能力，并针对其误差模式进行分析。

**💡 创新点**

首次提出仅通过非语言线索推断隐含意图的评估框架，发现LLM在此任务上性能远低于语音回应；同时揭示少量上下文学习可显著提升模型的解释能力。

**🔧 技术方法**

采用多选题式评估、零/少量提示（few-shot）、Chain‑of‑Thought（CoT）提示以及Logit Lens层级分析等技术，对公开指令微调模型进行实验。

**📊 数据集**

构造包含沉默、面部表情和动作三类非语言回应的多选题数据集，使用人工标注的参考答案作为基准。

**📈 对比分析**

与人类基准（≈0.93）以及各类LLM（从3B到100B+）进行对比，发现非语言条件下模型准确率平均约0.5，最高仅0.66，低于语音回应；大型模型略好；few‑shot提示提升约10%–15%。

**⚠️ 局限性**

仅针对英文文本的非语言线索进行评估，未考虑多语言文化差异和多模态输入，模型在缺乏上下文描述时表现更差。

---

## 456. ContinuousBench: Can Differentially Private Synthetic Text Improve Capabilities?

**arXiv ID:** 2606.01849 | [PDF](https://arxiv.org/pdf/2606.01849v1)

**作者:** Peihan Liu `[一作]` (Columbia University), Alex Bie `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个持续更新的基准和评估工具，用来衡量差分隐私文本合成是否能传递原始敏感语料的知识。

**💡 创新点**

设计了三大特性：①新鲜度与访问依赖的 QA 集，保证知识只能通过新语料获得；②基于短答案的事实 QA 评估，直接测量事实迁移；③生成器与评估器规模匹配，消除教师‑学生偏差。

**🔧 技术方法**

使用 DP‑SGD（梯度裁剪、截断泊松采样、PLD 会计）进行隐私训练；利用 Gemma‑3 等预训练模型进行持续预训练；通过 MAUVE、FID 等分布相似度指标评估生成质量。

**📊 数据集**

两条轨迹：①完全虚构的 Pokémon‑风格域（Gemininon）提供可控的事实重复；②持续收集的公共新闻（Common‑Crawl‑News）提供真实语言与时间演化。每条轨迹包含训练语料、支持记录数可调的 QA 集。

**📈 对比分析**

对比三类数据来源（无训练、真实语料、非私有合成、DP 合成）以及不同 ε、模型尺寸；结果显示非私有合成可将 QA 准确率提升至 90%+，但 DP 合成（ε=100 或 10）仅提升 5‑15%，甚至低于无训练；私有演化亦表现不佳。

**⚠️ 局限性**

DP 合成在捕获稀有/新鲜事实方面受限，梯度裁剪削弱稀有 token，导致事实迁移极差；评估仅基于短答案 QA，可能无法覆盖所有知识类型；持续生成需频繁管道运维。

---

## 457. Absorbing Complexity: An Interaction-Native Knowledge Harness for Financial LLM Agents

**arXiv ID:** 2606.01886 | [PDF](https://arxiv.org/pdf/2606.01886v1)

**作者:** Ailiya Borjigin `[一作]` (True Trading), Julia Stadnyk `[通讯]` (Inc4.net)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了交互原生知识捕捉架构，利用被动注入、时序图存储、wiki审计表实现持续金融认知。

**💡 创新点**

将知识注入系统主动完成，实时无缝更新图数据库并在写时失效，从而降低用户认知摩擦和过时知识使用。

**🔧 技术方法**

基于事件流感知、限定工作上下文缓存、时间知识图、wiki审计、背景抽取与失效、治理阈值等技术。

**📊 数据集**

使用24个种子、4轮、80条合成工作流的控制实验，涵盖市场分析、投资组合评估、复制交易评估、交易准备等任务；未来可扩展至FRED、SEC EDGAR、Binance公共数据。

**📈 对比分析**

与六个基线（模型仅、工具代理、简单记忆、wiki漫步、KH-无失效、INKH）在70.4k次评估中对比；INKH在任务质量0.815、延迟900 ms、Token 1540、context精度0.329、过时使用0.009、可追溯性0.999，显著优于其他基线。

**⚠️ 局限性**

实验为合成基准，未评估真实交易盈利，质量指标为仿真评估，图数据库实现为抽象，公共数据回放尚未完成。

---

## 458. Comparing ML-Specific and General Python Code Smells Across Project Characteristics

**arXiv ID:** 2606.01882 | [PDF](https://arxiv.org/pdf/2606.01882v1)

**作者:** Halimeh Agh `[一作]` (University of Stuttgart), Stefan Wagner `[通讯]` (Technical University of Munich)

**通讯引用:** 8722 | [OpenAlex ID](https://openalex.org/A5022333047)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

对 279 个开源 Python 机器学习项目进行大规模实证研究，比较 ML 专用代码臭味（ML‑CS）与通用 Python 代码臭味的密度及其与项目特征（大小、年龄、贡献者、提交频率、CI/CD、领域）的关系。

**💡 创新点**

首次系统比较 ML‑CS 与通用代码臭味在同一项目集中的出现频率、与项目属性的相关性差异，以及不同领域在两种臭味上表现出的独立性，为领域定制化质量保证策略提供依据。

**🔧 技术方法**

使用 CodeSmile（检测 12 种 ML‑CS）和 Pylint（筛选 20 个最常见通用 Python 规则）进行静态分析；统计分析采用 Spearman、Wilcoxon、Mann‑Whitney、Kruskal‑Wallis 等非参数检验，并通过 Cliff’s Delta、Fisher’s z 评估效应大小。

**📊 数据集**

基于 NICHE 数据集（572 个项目）更新并筛选后得到 279 个满足活跃度、星标、提交数、贡献者数和分支数阈值的项目，覆盖 7 个 ML 领域，累计 132,067 个 Python 文件、17,911,446 行代码。

**📈 对比分析**

通过配对 Wilcoxon 检验发现 ML‑CS 密度比通用 Python 臭味低 41–94 倍；回归和组间比较表明提交频率与领域与 ML‑CS 密度显著相关，而项目大小、年龄、CI/CD 与通用 Python 臭味无显著关联；两种臭味的领域排名彼此独立（Spearman ρ=0.643，p=0.119）。

**⚠️ 局限性**

局限性包括：仅检测 12/22 种 ML‑CS；工具误报/漏报可能影响结果；CI/CD 采样不平衡（244 对 35）降低统计力；项目筛选阈值可能导致样本偏差；研究仅覆盖 Python，未涵盖 R、Julia 等语言或专有项目。

---

## 459. WorldCoder-Bench: Benchmarking Physically Grounded 3D World Synthesis

**arXiv ID:** 2606.01869 | [PDF](https://arxiv.org/pdf/2606.01869v1)

**作者:** Shuo Lu `[一作]` (Chinese Academy Of Sciences), Jian Liang `[通讯]` (Chinese Academy Of Sciences)

**通讯引用:** 19147 | [OpenAlex ID](https://openalex.org/A5089400423)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了专门针对浏览器原生3D世界生成的基准（WorldCoder-Bench），并提出了基于执行的状态验证协议（WorldCoder-Verify）来评估生成代码的行为正确性。

**💡 创新点**

核心创新点包括：①隐藏且经过突变硬化的行为合同，确保验证覆盖真实功能而非仅视觉匹配；②通过头less Chromium沙箱实时提取运行时状态并与合同对照；③引入“返回自动化回报”和“时间效率乘数”两种基于成本与时间的实用度度量。

**🔧 技术方法**

使用的技术主要有：JavaScript/HTML/WebGL的程序生成、Playwright驱动的头less浏览器执行、脚本化动作序列与状态快照、软件工程中的突变测试、以及自定义的运行时状态接口。

**📊 数据集**

数据集包含 2,026 个专家标注的任务，分为 Simulation、Rendering、Application 三大类和 15 个细粒度子域，包含 205 个隐藏评测集、1,621 个公开对照集、615 个通过物理参数、资产和初始状态随机扰动生成的压力测试变体。

**📈 对比分析**

与传统的截图/DOM/视觉模型评估方法相比，基准对九种前沿大型语言模型进行零-shot评测。最佳系统（GPT‑5.4）仅实现 27.8% 的验证覆盖率；在扰动版任务上最高为 19.9%。这些结果表明，即使视觉效果接近目标，行为正确性仍难以达成，显示出现有模型在状态同步和物理一致性方面的显著不足。

**⚠️ 局限性**

主要局限包括：验证覆盖率低、状态模式漂移与交互链断裂占据大多数错误；突变硬化虽提升严谨性，但对复杂物理或算法推理的支持不足；评测成本高，需在真实浏览器环境中执行；此外，数据集虽多样，但对极端高复杂度或创意任务的覆盖仍有限。

---

## 460. Task-Induced Representational Invariances Depend on Learning Objective in Deep RL

**arXiv ID:** 2606.01868 | [PDF](https://arxiv.org/pdf/2606.01868v1)

**作者:** Manu Srinath Halvagal `[一作]` (Harvard University), SueYeon Chung `[通讯]` (Harvard University)

**通讯引用:** 803 | [OpenAlex ID](https://openalex.org/A5016533438)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对比 DQN、PPO、Deep SARSA(n) 等典型深度 RL 算法在迷宫、CartPole、Pong、Atari 等环境中的学习过程，系统分析了它们在表征空间中对 MDP 同构对称性和策略对称性的不同编码方式；

**💡 创新点**

创新点在于首次将 MDP 同构理论与表征相似度分析（RSA/CKA）相结合，揭示价值型方法倾向学习 MDP 同构对称性，而策略梯度方法倾向学习策略对称性，并进一步探讨了这种对称性差异对迁移学习和 LLM 提示依赖性的影响；

**🔧 技术方法**

主要技术包括 MDP 同构与 Bisimulation 理论、RSA/CKA 相似度计算、深度 RL 算法训练（DQN、PPO、Deep SARSA(n)）、LLM（Qwen2.5-72B-Instruct）提示实验以及在 Atari 迁移学习中的性能评估；

**📊 数据集**

使用的数据集包括可视化与表格版的迷宫导航任务、CartPole、Pong、Atari 游戏（Breakout、SpaceInvaders、Pong）以及 LLM 提示的 ASCII 树或边列表形式；

**📈 对比分析**

通过 RSA 计算表征相似度矩阵，对比不同算法在 MDP 对称性与策略对称性上的相似度，结果显示 DQN 在 MDP 对称性上相似度更高，PPO 在策略对称性上更突出；在 Atari 迁移学习实验中，DQN 的性能明显优于 PPO；LLM 在不同提示格式下呈现不同的对称性特征；

**⚠️ 局限性**

局限性包括：仅评估了有限的离散任务，未覆盖 POMDP、模型基算法和记忆网络；RSA 只适用于全可观测且无序列依赖的情形；只使用余弦相似度，未尝试其他度量；缺乏对对称性产生机制的形式化理论。

---

## 461. RadioMaster: Multi-Agent System for Autonomous Radio Signal Generation

**arXiv ID:** 2606.01862 | [PDF](https://arxiv.org/pdf/2606.01862v1)

**作者:** Jiazhen Lei `[一作]`, Xiaohua Tian `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 RadioMaster，一个完整的多智能体系统，能够将用户意图自动转换为可执行的物理无线信号，完成无线原型的“最后一公里”工作。

**💡 创新点**

创新点包括：①基于领域特定知识库的 RadioWiki 采用自适应路由 RAG 消除生成式模型的幻觉；②RadioAgent 通过 Planner、Worker、Modulator、Operator 四个协作智能体拆分任务并递归优化；③RadioEmulator 实现闭环虚拟仿真验证，保证物理层信号完整性；④提出首个专门评测无线信号生成的 RadioBench 基准。

**🔧 技术方法**

技术手段包括：检索增强生成 (RAG)、多模态嵌入与双通道检索、LLM 结构化生成、多智能体协同推理、基于 MATLAB/USRP 的硬件仿真与 OTA 验证。

**📊 数据集**

使用 RadioBench 数据集，涵盖四个层级的测试用例：知识问答（1000 Q‑A），实际实现（400 例，分为 Wi‑Fi/BLE 与 MATLAB/UHD 两范式），以及复杂场景（200 例，含歧义需求与严苛硬件约束）。

**📈 对比分析**

与多种 SOTA LLM（如 Gemini3.1、Qwen3‑Max 等）及开源多智能体框架（AutoIoT、IoTPilot）在 RadioBench 上对比。RadioMaster 在知识准确率（QAA）接近最优，同时在实装层面表现显著：配置通过率 (CPR) 83%、硬件部署率 (HDR) 71%、信号完整率 (SIR) 64%（最高层级）高于对照组约 4–5 倍，且整体系统稳定性显著提升。

**⚠️ 局限性**

主要局限：对用户输入的参数仍需相对完整、规范，缺乏足够的自动化推断；在极度模糊或新兴标准下，检索与推理仍可能产生误差；系统对高频硬件平台的适配尚未完全覆盖，未来需扩展知识库与仿真模型。

---

## 462. From Global Policies to Local Strategies: Multi-Objective Optimization of Resource-Specific Handover Policies

**arXiv ID:** 2606.01857 | [PDF](https://arxiv.org/pdf/2606.01857v1)

**作者:** Lukas Kirchdorfer `[一作]` (SAP Signavio), Hugo A. López `[通讯]` (Technical University of Denmark)

**通讯引用:** 553 | [OpenAlex ID](https://openalex.org/A5087594010)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于多目标进化算法的资源特定交接策略优化方法，通过模拟与搜索实现流程性能提升；

**💡 创新点**

创新点在于将代理式过程模拟与NSGA-II相结合，生成针对资源协作模式的Pareto最优交接策略，突破传统全局分配的局限；

**🔧 技术方法**

核心技术包括代理式过程模拟（AgentSimulator）、多目标进化算法NSGA-II、定制遗传算子（交叉、四种变异策略）以及基于模拟的适应度评估；

**📊 数据集**

实验使用了五种Loan Application流程的合成变体以及七个公开业务过程日志（如BPI、ACR等），并对资源成本进行合理赋值；

**📈 对比分析**

与As-Is、AlwaysAvailable、Random、Cheapest、ShortestProcessingTime等基线进行对比，实验表明平均成本下降约37%，等待时间下降约58%，性能明显优于基线，但计算耗时显著增加；

**⚠️ 局限性**

局限性包括对模拟模型准确性的依赖、可能的过拟合风险、计算开销大、假设资源可自主决定交接以及仅考察两项目标等。

---

## 463. Boosting Multimodal Federated Learning via Chained Modality Optimization

**arXiv ID:** 2606.01856 | [PDF](https://arxiv.org/pdf/2606.01856v1)

**作者:** Zixin Zhang `[一作]` (Inner Mongolia University), Changsheng Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 26432 | [OpenAlex ID](https://openalex.org/A5022636178)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 FedMChain 框架，通过链式多模态联邦学习（MCFT）阶段化单模态训练并加入误差补偿正则化，解决模态竞争问题。

**💡 创新点**

创新点包括：① 模态链式训练将多模态学习拆分为阶段化单模态优化；② 误差补偿正则化提升跨模态互补；③ 稀疏符号聚合（SSCA）基于方向一致性进行簇化与聚合，降低通信频次并提升鲁棒性。

**🔧 技术方法**

采用联邦学习、阶段化梯度更新、跨模态对齐与互补正则、稀疏符号聚合与簇化技术，处理非 IID 与模态异构。

**📊 数据集**

使用 CREMA‑D、AVE、CMU‑MOSEI 三大多模态情感/语音数据集进行评估。

**📈 对比分析**

与 Local、FedAvg、FedProx、SCAFFOLD、BMSFed 等基线对比，FedMChain 在所有数据集上实现最高 ACC/ACC_m，且通信成本更低，可在更少同步周期下达到同等或更佳性能。

**⚠️ 局限性**

主要局限是模态训练顺序目前经验确定，缺乏自适应最优顺序策略；实验范围受限，未在更大规模或其他任务上验证。

---

## 464. Decoupled Residual Quantization for Robust Semantic IDs in Recommendation

**arXiv ID:** 2606.01844 | [PDF](https://arxiv.org/pdf/2606.01844v1)

**作者:** Xuesi Wang `[一作]` (Shopee), Guanxing Zhang `[通讯]` (Shopee)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种用于评估推荐系统中语义 ID 质量的定量诊断框架，并以 Decoupled Residual Quantization（DRQ）为示例实现。

**💡 创新点**

创新点在于引入期望重叠率（Expected Overlap Rate）和有效码本容量（Effective Codebook Size）两个指标，能够将语义 ID 退化分解为分布失衡与几何重叠两大根源，并通过解耦连续表示学习与离散分配来缓解这两种失效模式。

**🔧 技术方法**

采用 VAE 对连续潜在空间进行重塑，随后使用层级 K‑Means 进行离散编码；对比实验还使用了传统 RQ‑VAE、EMA 版 RQ‑VAE、直接对原始嵌入做 K‑Means、以及加入对比学习的 DRQ‑VAE+CL 等方法。

**📊 数据集**

实验基于 15M 条短视频的内部工业数据集，原始多模态嵌入维度为 256。

**📈 对比分析**

在三种评价维度上（几何保真度、符号鲁棒性和软匹配效果）进行比较：RQP‑VAE 在符号容量与码本利用率上表现最好；DRQ‑VAE 在重构精度和近似检索保持率上遥遥领先；DRQ‑VAE+CL 在软匹配和高截断检索召回率上最优。

**⚠️ 局限性**

局限性包括：仅在单一专有数据集上验证，缺乏公开基准的交叉验证；评估侧重于检索与匹配，而非完整推荐流水线的 Recall/NDCG；所用的 O_π 等指标基于简化假设，需进一步验证其通用性。

---

## 465. Single-Line Drawing Generation via Semantics-Driven Optimization

**arXiv ID:** 2606.01910 | [PDF](https://arxiv.org/pdf/2606.01910v1)

**作者:** Tanguy Magne `[一作]` (ETH Zurich), Olga Sorkine-Hornung `[通讯]` (ETH Zurich)

**通讯引用:** 6792 | [OpenAlex ID](https://openalex.org/A5064927253)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于统一有理B样条（URBS）的单笔绘图生成方法，能够从文本提示或参考图像直接得到连续的矢量线稿；

**💡 创新点**

创新点在于将分数扩散采样（SDS）与URBS曲线参数化相结合，并通过自适应权重稀疏化、拉伸与排斥损失等正则化实现单笔连续性与艺术化控制；

**🔧 技术方法**

技术核心包括分数扩散采样、DiffVG差分光栅化、LoRA微调的Stable Diffusion 3.5、深度ControlNet、以及曲线梯度优化；

**📊 数据集**

使用了220幅网络收集的单笔绘图数据进行LoRA训练，并在50对文本–图像基准集上评估；

**📈 对比分析**

与多种基线（TSP、ControlSketch、CLIPasso、Gemini、Flux等）对比，本文在文本-图像相似度、图像-图像相似度、审美分数和FID上均优于直接生成的多笔绘图方法，并在用户研究中获得最高排名；

**⚠️ 局限性**

局限性包括对稀有主题的细节不足、生成耗时约15分钟、对扩散模型性能高度依赖以及缺乏动画生成支持。

---

## 466. Physically-Constrained Mamba-SDE for Remaining Useful Life Prediction under Irregular Observations

**arXiv ID:** 2606.01894 | [PDF](https://arxiv.org/pdf/2606.01894v1)

**作者:** Deyu Zhuang `[一作]` (Nanjing University of Aeronautics and Astronautics), Daoqiang Zhang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 19113 | [OpenAlex ID](https://openalex.org/A5018821033)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于连续时间神经随机微分方程的 RUL 预测框架 PC-MambaSDE，能够在异步、缺失、时序抖动的实际传感器数据下进行准确的剩余使用寿命预测。

**💡 创新点**

核心创新包括：1）掩码感知连续 Mamba 编码器将缺失信息直接嵌入控制信号；2）物理引导的潜在 SDE 采用参数化修正混合漂移，强制单调衰变；3）终端退化惩罚将 RUL 预测建模为边值问题；4）构造了混合不规则性生成方案 HIGS 用于严格评估。

**🔧 技术方法**

融合了 Mamba 状态空间模型、连续时间神经控制微分方程、Girsanov 理论下的 KL 正则化、Lyapunov 稳定性分析以及基于终端惩罚的边值优化。

**📊 数据集**

在公开的 C-MAPSS 与 N-CMAPSS 两大工业监测数据集上进行实验，分别对应飞机发动机和制造系统的多传感器轨迹。

**📈 对比分析**

与离散时序模型（LSTM+GP、LSTM+MPACE、S-MFMLP、PSR）以及连续神经方程模型（Latent-ODE、Latent-SDE、ACSSM）进行对比。PC-MambaSDE 在各种数据稀缺率（50%–90%）与 HIGS 注入的不同异常模式下均获得最低 RMSE，尤其在 90% 缺失率下 FD004 子集 RMSE 仅为 22.18，明显优于最佳基线 24.84。

**⚠️ 局限性**

方法在模型复杂度和训练时间上较传统 LSTM 高；需要预先设定物理偏置参数，若系统衰变非单调或存在可逆过程，当前物理约束可能不适用；此外 HIGS 生成的异常模式虽逼真但仍有限，可能无法覆盖所有工业场景。

---

## 467. Segment-driven Structural Induction and Semantic Alignment for Heterogeneous Tabular Representation

**arXiv ID:** 2606.01890 | [PDF](https://arxiv.org/pdf/2606.01890v1)

**作者:** Woojun Jung `[一作]` (Korea University), Susik Yoon `[通讯]` (Korea University)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5083900503)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于segment的预训练框架NAVI，通过Masked Segment Modeling和Entropy-driven Segment Alignment学习异构表格的域特定语义。

**💡 创新点**

创新点在于将header–value对视为segment，结合结构层和分布层证据，并使用熵划分属性进行对齐，实现跨表的语义一致性。

**🔧 技术方法**

使用Transformer编码器、masking策略、对比学习（InfoNCE）以及熵阈值路由等技术。

**📊 数据集**

在Movie和Product两个子域的WDC WebTables数据集上进行实验。

**📈 对比分析**

与BERT、TAPAS、CM2、HAETAE等基线相比，NAVI在header恢复、值填充、语义聚类以及下游分类任务上均取得显著提升，特别是在异构表格上的鲁棒性最高。

**⚠️ 局限性**

局限性包括对数值特征依赖不强，仍需改进对高度冗余schema（如Movie）下的结构利用，以及对跨域迁移的进一步验证。

---

## 468. Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection

**arXiv ID:** 2606.01885 | [PDF](https://arxiv.org/pdf/2606.01885v1)

**作者:** Xiaolu Kang `[一作]` (Wuhan University), Qian Wang `[通讯]` (Wuhan University)

**通讯引用:** 58456 | [OpenAlex ID](https://openalex.org/A5100422786)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种可靠的多视图证据学习框架 DiCoME，用于深度伪造检测，将语义与结构视图分离后进行证据融合。

**💡 创新点**

创新点包括：1) 用几何正交投影进行视图净化，显著抑制语义遮蔽效应；2) 采用基于 Dempster–Shafer 的不确定性感知证据融合，捕捉语义与结构视图的冲突；3) 结合 CLIP 先验与 β‑VAE 语义流形投影，提升通用性。

**🔧 技术方法**

使用了 CLIP ViT‑L/14 (LoRA 微调)、β‑VAE、正交投影、Evidential Deep Learning、Dirichlet 分布、Dempster–Shafer 理论、以及相应的对齐与 KL 正则化损失。

**📊 数据集**

训练使用 FaceForensics++ (FF++)，在跨数据集（CDFv2、DFD、DFDC、DFo、WDF、CDFv3）和跨操纵（DF40）上进行评估。

**📈 对比分析**

与 12 先进方法（如 F3Net、SPSL、SRM、CORE、RECCE、SBI、UCF、IID、LSDA、ProDet、Effort、GenD）对比，DiCoME 在所有基准上均取得最高 AUC，跨数据集最高 0.977，跨操纵 0.982，且在不确定性校准方面表现优异。

**⚠️ 局限性**

局限性包括：对视频级时序信息处理有限；正交投影依赖对语义子空间的准确估计；过度依赖 CLIP 可能限制跨模态推广；模型推理成本较高；对抗攻击的鲁棒性尚未系统评估。

---

## 469. A Theoretical Framework for Self-Play Theorem Proving Algorithms

**arXiv ID:** 2606.01861 | [PDF](https://arxiv.org/pdf/2606.01861v1)

**作者:** Thomas Chen `[一作]`, Zhiyuan Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文建立了基于定理图的自对弈（prover‑conjecturer）理论框架，并证明在图满足强等温不等式时，利用可逆随机步（随机游走）构造的猜想器可使证明者知识集指数级增长；

**💡 创新点**

创新点在于：①将定理关系建模为图并给出原子假设；②提出针对定理图连通度的理论分析；③发现自对弈中出现的“过度复杂定理”问题，并引入多样性度量和基于扩散相似度的改进猜想器；④利用对比学习得到扩散相似度的近似。

**🔧 技术方法**

主要技术包括：图论与随机游走、强等温不等式、随机游走的扩散相似度定义、对比学习构造嵌入、对策优化求解猜想器权重。

**📊 数据集**

论文主要为理论分析，未给出具体数据集；若要实验可使用 Lean 或 Coq 等形式化证明库作为定理图。

**📈 对比分析**

未进行实验对比，因本文侧重理论证明；理论结果显示在满足连通性假设时，知识集可指数增长，改进的猜想器在“聚类图”情形下能显著提升扩散相似度与多样性。

**⚠️ 局限性**

局限性：猜想器对当前证明者不适配，仅以证明者类别为目标；不考虑定理的组合/层级结构；假设图为无向连通且可逆随机游走，实际定理图可能更复杂。

---

## 470. Pool-Select-Refine: Allocation-Aware Generative Dataset Distillation with Soft-Label-Guided Latent Refinement

**arXiv ID:** 2606.01920 | [PDF](https://arxiv.org/pdf/2606.01920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 471. Polaris: Scaling Up Instruction-Guided Image Generation Towards Millions of Personalized Style Needs

**arXiv ID:** 2606.01858 | [PDF](https://arxiv.org/pdf/2606.01858v1)

**作者:** Zhi-Kai Chen `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**通讯引用:** 3677 | [OpenAlex ID](https://openalex.org/A5065180062)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建一个大型的 Stable Diffusion 检查点与 LoRA 适配器模型库，并通过多模态检索和指令解析器，实现无需额外训练即可根据用户的文本与图像指令自动选择最合适的模型进行图像生成或编辑。

**💡 创新点**

① 将社区公开的检查点与 LoRA 适配器统一成可检索的库；② 设计多模态检索+树形 Rerank 机制，显著提升检索效率和精度；③ 引入指令解析器与区域掩码器，实现对指令的细粒度理解与定位。

**🔧 技术方法**

Stable Diffusion、LoRA 适配器、CLIP 编码器、多模态嵌入、Vision‑Language Model（Qwen2.5‑VL‑7B）、SAM、LLM（基于树形注意力的 reranker）等。

**📊 数据集**

从 Civitai 等平台收集 6,500 个检查点和 75,000 个 LoRA 适配器；使用 GEdit‑Bench 和自建的 User‑Bench（社区真实任务）进行评估；利用 VLM、GPT‑4o 进行自动化评测。

**📈 对比分析**

与 InstructP2P（小规模 fine‑tuning）和 Bagel（大规模预训练+LLM 监督）进行对比；在 GEdit‑Bench 上与 InstructP2P 的编辑质量相当，且在风格相关任务上更优；在 User‑Bench 的 Local Style Change 与 Style Extraction 上均优于 InstructP2P 并接近 Bagel；同时无额外训练，推理时间约 78 秒，树形 rerank 加速 1.5×。

**⚠️ 局限性**

检索过程对候选集规模仍有敏感性；极为稀有或新颖的风格受限于模型库中是否存在对应资源；多模态检索可能出现误检；局部细节编辑效果不一定能匹配专门 fine‑tuned 模型；树形 rerank 需要复杂的注意力掩码实现，增加实现难度。

---

## 472. PHASOR: Phase-Anchored Universal Action Representations for Humanoid Embodiments

**arXiv ID:** 2606.01851 | [PDF](https://arxiv.org/pdf/2606.01851v1)

**作者:** Kihyun Kim `[一作]` (AIM Intelligence), Haon Park `[通讯]` (AIM Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了基于运动周期性阶段（phase）的动作嵌入空间，并将其通过人类预训练的冻结oracle与轻量级适配器迁移到四种人形机器人上。

**💡 创新点**

创新点包括：1）将运动分解为分体部位的周期参数，形成跨体型可解释的phase manifold；2）将姿态信息与phase分离，只在解码时通过FiLM条件化，保持时序一致性；3）使用人类锚定与LAMP语义先验实现跨体型对齐。

**🔧 技术方法**

使用技术包括：周期自动编码器（FFT瓶颈）、FiLM条件化、Soft InfoNCE + LAMP软目标、单位球映射与对齐头、速度/形状一致性正则，以及轻量级输入/输出适配器。

**📊 数据集**

使用了AMASS CMU/KIT 人类运动数据，经过GMR重新定向到 Unitree G1、H1.2、Booster T1、Berkeley Humanoid Lite 共计约50,000条同步人类-机器人剪辑。

**📈 对比分析**

与 MLP 与 VQ 两种基线在跨体型检索、运动模仿、遥操作和强化学习任务中进行对比；PHASOR 在检索 Recall@1 约90%，MPJPE 1.62mm，遥操作 64.75mm，RL 任务中步态更稳，整体性能显著优于基线。

**⚠️ 局限性**

局限性在于：仅对周期性、节奏化动作提供强结构，非周期性或瞬态动作的表现较弱；姿态分支仅在各体型内部共享，无法实现跨体型姿态对齐；实验仅在仿真环境，缺乏真实硬件验证。

---

## 473. Private and Stable Test-Time Adaptation with Differential Privacy

**arXiv ID:** 2606.01908 | [PDF](https://arxiv.org/pdf/2606.01908v1)

**作者:** Zefeng Li `[一作]` (University of British Columbia), Evan Shelhamer `[通讯]` (University of British Columbia)

**通讯引用:** 71269 | [OpenAlex ID](https://openalex.org/A5023786468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在测试时自适应过程中加入差分隐私机制，构建了 DP‑TTA 方法并研究了梯度逐样本裁剪对自适应效果的影响。

**💡 创新点**

创新点在于首次将差分隐私技术应用于测试时自适应，提出了可适配多种 TTA 方法的通用 DP 框架，并发现逐样本梯度裁剪既能保障隐私又能提升自适应性能。

**🔧 技术方法**

使用了 DP‑SGD（逐样本梯度裁剪 + 高斯噪声）以及对现有 TTA 算法（Tent、EATA、SAR、DeYO、COME）做的改造；同时利用了隐私核 DP（GDP）进行隐私计数。

**📊 数据集**

在 ImageNet-C（severity 5）和 ImageNet-R 上的 ViT‑Base/16 与 ConvNeXt‑Tiny 视觉模型进行评估。

**📈 对比分析**

与原始非隐私 TTA 方法对比，DP‑TTA 在 ε≈10–20 时的精度仅略低于或略优于原始方法；在低隐私预算（ε≈1）时仅损失约 2% 误差；逐样本裁剪单独使用时可提升 0.1–4.1% 的准确率。

**⚠️ 局限性**

局限性包括：在更强隐私预算下准确率下降显著；部分 TTA 算法的 DP 版本实现较为复杂；仅在视觉任务上验证，缺乏跨领域通用性验证。

---

## 474. Mechanistic Diagnostics of Spatial Lexical Bias in Multimodal Large Language Model Spatial Reasoning

**arXiv ID:** 2606.01914 | [PDF](https://arxiv.org/pdf/2606.01914v1)

**作者:** Chuang Ma `[一作]` (Kyoto University), Sadao Kurohashi `[通讯]` (Kyoto University)

**通讯引用:** 4903 | [OpenAlex ID](https://openalex.org/A5028836340)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究发现并诊断了多模态大语言模型在空间多选题中的“空间词汇偏差”，并通过轻量化的LLM侧DPO修复提升性能。

**💡 创新点**

创新点在于提出“binary‑stable but ternary‑fragile”诊断、定位语义偏差到LLM内部状态，并证明单一对齐数据即可修复。

**🔧 技术方法**

使用的技术包括多模态LLM、视觉注意力可视化、残差流线性探针、激活补丁、稀疏干预和LoRA‑DPO。

**📊 数据集**

用到了九个开源LLM（LLaVA、InternVL、Qwen 等）以及三个合成数据集（Sphere–Cube Raw、Sphere–Dog Raw、Sphere–Dog Outdoor）和三个公开基准（WhatsUp、SpatialMQA‑Direct、VSR）。

**📈 对比分析**

通过样本级鲁棒准确率比较，修复后在合成数据四选项上提升至约100点，在 WhatsUp、SpatialMQA‑Direct、VSR 分别提升 68.0、32.6、20.1 点，表现显著优于基线。

**⚠️ 局限性**

限制包括仅针对四方向简单多选、使用极小的单对齐数据，可能无法覆盖更复杂的空间关系、开放式生成任务和更大视觉多样性。

---

## 475. SMH-Bench: Benchmarking LLM Agents for Environment-Grounded Reasoning and Action in Smart Homes

**arXiv ID:** 2606.01912 | [PDF](https://arxiv.org/pdf/2606.01912v1)

**作者:** Kuan Li `[一作]` (Midea Group), Yi Xu `[通讯]` (Midea Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SMH-Bench，用于评估LLM驱动的智能家居代理。

**💡 创新点**

创新在于基于可执行、可验证的HomeEnv模拟器，构造了多样化、分层任务和可验证的评估。

**🔧 技术方法**

采用可执行工具调用、ReAct交互式推理、LLM-as-Judge与规则验证等技术。

**📊 数据集**

使用1,100个人工审核的任务，涵盖7类能力、22细粒度子类，规模从小型公寓到135设备的大型住宅。

**📈 对比分析**

对13种LLM在DR和EIA两种设置下进行比较，Gemini-3.1-Pro在EIA下平均成功率达85.2%，但在自动化、模糊意图等类别仍显著下降。

**⚠️ 局限性**

局限在于仍难以在复杂家居中完成自动化和模糊意图处理，且对状态推理、记忆与工具交互的可靠性有待提升。

---

## 476. The Image Reconstruction Game: Drawing Common Ground Through Iterative Multimodal Dialogue

**arXiv ID:** 2606.01901 | [PDF](https://arxiv.org/pdf/2606.01901v1)

**作者:** Sherzod Hakimov `[一作]` (University of Potsdam), David Schlangen `[通讯]` (University of Potsdam)

**通讯引用:** 3447 | [OpenAlex ID](https://openalex.org/A5032801642)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种完全自动化的图像重建游戏，通过视觉语言模型与图像生成器的多轮对话，检验模型在描述和修正图像上的能力。

**💡 创新点**

创新点在于把对话生成的渲染图像作为直接可观察的共同基础，系统化评估描述者与生成器的交互效果，并验证自动评判与人类感知的差异。

**🔧 技术方法**

使用VLM（如GPT‑5.2、Qwen3‑VL‑30B）作为描述者，图像生成器Gemini‑3.1‑nano‑banana和GPT‑Image‑1.5，以及LLM评判和LipSim相似度指标进行多轮对话与性能评估。

**📊 数据集**

构建了七类图像（简单形状、几何图、函数图、折线图、柱状图、饼图及Visual Genome照片）且设易/难两级，共140个实验实例。

**📈 对比分析**

通过人类与LLM评判的相似度与偏好任务对模型对比，发现描述者质量决定重建质量，生成器决定多轮是否带来提升；GPT‑5.2+Gemini‑3.1在大多数条件下表现最佳，但自动评判与人类仍存在显著差距。

**⚠️ 局限性**

局限性包括实验规模有限（仅四对模型共140条）、人类评判仅覆盖一对模型、数据集语言单一（英语），以及自动评判与人类感知的不一致导致对真实质量估计不可靠。

---

## 477. Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation

**arXiv ID:** 2606.01897 | [PDF](https://arxiv.org/pdf/2606.01897v1)

**作者:** Tianjiao Li `[一作]` (Bilibili Inc), Huyang Sun `[通讯]` (Bilibili Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了社区共鸣评价任务 CASTER，设计并训练 MEDEA 框架，通过多模态视角取向和社会链式思维评估 UGC 质量。

**💡 创新点**

创新地将社会认知与多模态评价结合，引入 Social-CoT 与社会对齐奖励实现人类情感共鸣的模拟。

**🔧 技术方法**

采用多模态预训练模型、监督微调、过程监督强化学习、Skellam 一致性机制和社会对齐奖励。

**📊 数据集**

使用自建 CASTER‑Bench（1485 条长视频+多模态信息+专家标签）以及大规模未标注 UGC 评论数据。

**📈 对比分析**

与传统 VQA、标准 LMM、长链式 CoT 以及仅使用 Social-CoT 的基线进行对比，MEDEA 在 CASTER‑Bench 上高质量类 F1 达到 0.650，宏观 F1 0.749，明显优于所有基线。

**⚠️ 局限性**

计算开销略高，社群对齐主要针对特定平台，二元质量划分限制细粒度评价，且对不同文化生态的泛化仍待验证。

---

## 478. Adversarial Attacks on Robot Localization Systems via Deep Feature Perturbation

**arXiv ID:** 2606.01892 | [PDF](https://arxiv.org/pdf/2606.01892v1)

**作者:** Zhenyu Li `[一作]` (Shandong Academy of Sciences), Tianyi Shang `[通讯]` (Fuzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量级产品量化网络（LPQN）来生成针对机器人定位系统中产品量化（PQ）模块的对抗查询。

**💡 创新点**

创新点在于利用软质心分布的可微分近似，在两阶段优化（分布全局扰动+峰值扰动）下同步攻击分布层和特征层，且LPQN结构轻量、可插拔。

**🔧 技术方法**

技术手段包括：LPQN残差扰动模块、软质心分布（soft‑assignment）、分布全局扰动（DWP）、峰值扰动（PTP）、两阶段前向/反向优化、PGD等梯度方法。

**📊 数据集**

实验数据集涵盖了大规模视觉定位基准（Pittsburgh250k、Tokyo24/7），并在移动机器人与无人机实际环境中进行了实地验证。

**📈 对比分析**

与传统对抗方法（FGSM、PGD、UAPR）以及多种主流视觉定位模型（NetVLAD、CosPlace、MixVPR等）对比，LPQN‑Hybrid在Recall@1上从约85%降至≈35%，比基线低30+个百分点，且对不同码长均保持显著破坏效果。

**⚠️ 局限性**

局限性包括：仅在白盒攻击情境下有效；仅针对基于PQ的检索管线，其他量化或聚类方案的鲁棒性尚未评估；对抗扰动需离线训练，在线生成受限于计算频率。

---

## 479. The Price of Decentralization in Block Building

**arXiv ID:** 2606.01874 | [PDF](https://arxiv.org/pdf/2606.01874v1)

**作者:** Burak Öz `[一作]` (Flashbots), Stefanos Leonardos `[通讯]` (King's College London)

**通讯引用:** 407 | [OpenAlex ID](https://openalex.org/A5048295129)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文将去中心化区块构建建模为带有延迟的随机覆盖博弈，研究自利建造者的地域选择对交易覆盖、福利与奖励分配的影响。

**💡 创新点**

创新点包括证明等分奖励下游戏为精确潜在游戏、给出最优势（PoA）上界为 2 的渐近紧致证明，并推导出博弈均衡下最差收益与最优收益之比不小于 1/2、HHI 上界为 9/8 之类的集中度上界；通过仿真验证在中间延迟-价值参数区间福利损失最大，并揭示空间集中与收益均衡不必同步。

**🔧 技术方法**

主要技术手段有：潜在游戏理论、光滑性框架下的 PoA 证明、Harmonic‑潜在函数构造、均衡存在性证明；仿真采用基于 GCP 地理延迟矩阵的 Lognormal 延迟模型、Poisson 交易生成、等分奖励计算与异步更优响应动力学。

**📊 数据集**

使用的数据集包括：Google Cloud Platform (GCP) 的 24 个区域坐标及其间延迟测量、随机抽样的高价值与外围交易源（各 5 个）以及相应的交易价值 λ_I，延迟为 Lognormal 取值。

**📈 对比分析**

对比方法是把自利均衡（PNE）与社会规划器最优解（或贪心近似）在福利、HHI、地理 HHI 与源覆盖率等指标上进行比较。理论上 PoA ≤ 2，仿真中福利比值在 0.8–1 之间波动，地理 HHI 与收益 HHI 在均衡时相对较低，规划器能显著提升外围源覆盖率。

**⚠️ 局限性**

局限性包括：忽略块容量约束、假设奖励完全等分、未考虑自利交易路由、Sybil 与合谋、以及仅在单轮、单节点的随机模型下检验，缺乏对真实协议层次传播细节的精细建模。

---

## 480. G2LoRA: Gradient Orthogonal Low-Rank Adaptation Framework for Graph Continual Learning on Text-Attributed Graphs

**arXiv ID:** 2606.01873 | [PDF](https://arxiv.org/pdf/2606.01873v1)

**作者:** Yuhan Wang `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**通讯引用:** 18964 | [OpenAlex ID](https://openalex.org/A5100380470)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在文本属性图（TAG）的连续学习场景中，提出一种基于 LLM‑as‑Aligner 的参数高效微调框架，统一节点、边、图级任务的对齐目标，并通过梯度投影与调制机制缓解灾难性遗忘。

**💡 创新点**

创新点包括：① 统一的图‑文本对齐损失，将异构任务归一化；② 以类别为粒度的双向梯度投影，既抑制新旧任务冲突，又实现条件向后迁移；③ 梯度幅度调制，在双编码器架构中动态平衡图像与文本的更新速率，防止跨模态漂移。

**🔧 技术方法**

采用 LoRA 低秩适配器、梯度正交投影（GPM）、SVD 生成类别子空间、CLIP‑style 对比学习、梯度幅度调制以及少样本（k‑shot）连续学习策略。

**📊 数据集**

实验使用 11 个公开 TAG 数据集（5 大规模用于预训练，6 小规模用于下游连续学习），涵盖学术引用网络、社交平台、电子商务等四个领域。

**📈 对比分析**

与 GNN、LLM、GLM 基础模型以及 LoRA、InfLoRA 等 PEFT 基线进行对比；在 CIL、DIL、TIL 三种增量设置下，平均准确率提升约 7.6%（相对最优基线），遗忘率下降，部分任务甚至实现正向遗忘（知识向后迁移）。

**⚠️ 局限性**

局限性：需在冻结的双编码器上训练，对极端异构任务的迁移仍有挑战；梯度幅度调制阈值和正交投影参数需要经验调优；缺乏重放机制，长期任务序列可能出现细粒度对齐漂移。

---

## 481. Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion through Corrections

**arXiv ID:** 2606.01865 | [PDF](https://arxiv.org/pdf/2606.01865v1)

**作者:** Zhaoting Li `[一作]` (Delft University of Technology), Jens Kober `[通讯]` (University of Stuttgart)

**通讯引用:** 8565 | [OpenAlex ID](https://openalex.org/A5035229829)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

研发一种 Set‑Supervised Diffusion Policy（SDP），利用人类纠正的正负动作块构建可接受动作集合，并在此集合内训练扩散策略生成动作块。

**💡 创新点**

创新点包括：①将 CLiC 的可设定动作集合扩展到动作块空间；②将集合监督融入扩散模型，形成基于集合的训练目标；③采用反射/投影采样在集合内生成样本；④结合在线交互式学习和离线数据聚合，显著提升对噪声和分布漂移的鲁棒性。

**🔧 技术方法**

使用的技术：扩散策略（Diffusion Policy）基于 DDPM；集合监督框架 CLIC；对集合的反射/投影采样；行为克隆、直接偏好优化、Ambient Diffusion 等对比损失；IIL、DAgger‑style 数据聚合；在仿真和实际机器人上进行实验。

**📊 数据集**

使用的数据集：仿真任务（Push‑T、PickCan、Square、TwoArmLift）；离线数据来源包括 SDP 自己在线收集、DP 在线收集以及 Robomimic；真实机器人实验任务包括 Insert‑T（KUKA iiwa）和 round‑table 组装（Franka Panda）。

**📈 对比分析**

与 DP、DP‑DPO、Ambient DP、CLIC、IBC 等基线比较；在在线交互和离线训练上均优于基线，尤其在噪声干扰下更鲁棒；SDP 在 Insert‑T 和 round‑table 任务中成功率提升约 10% 以上；同时产生更广覆盖的训练数据，提升离线学习效果。

**⚠️ 局限性**

局限性：仅支持绝对纠正（非相对纠正）；训练过程中需要采样集合内动作，约占总训练时间的 38%，增加计算成本；目前未针对更复杂的反馈模态或高维连续动作块空间进行深入优化。

---

## 482. Continual Learning as a Multiphase Moving-Boundary Problem

**arXiv ID:** 2606.01863 | [PDF](https://arxiv.org/pdf/2606.01863v1)

**作者:** Snigdha Chandan Khilar `[一作]` `[通讯]`, Snigdha Chandan Khilar

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将持续学习的稳定-可塑性困境映射为Stefan（熔化/固化）物理问题的框架——Stefan-CL，通过学习的签名距离场（SDF）来表示已固化的知识区域，并利用级集方法驱动前沿演化以冻结已学习区域。

**💡 创新点**

创新点在于将物理自由边界问题的Stefan条件与级集方法直接转化为持续学习机制，使“固化”与“可塑”由单一物理参数（潜热L）控制，并且前沿的形状与位置由数据自驱动学习，无需显式阈值或手工规则。

**🔧 技术方法**

采用的技术包括：签名距离函数与Eikonal约束、层级网络实现SDF、基于数据驱动的速度场与最小化残差的物理信息神经网络（PINN）思路、级集的拓扑变化表示、平滑相掩模与功能锚定。

**📊 数据集**

使用的是人工构造的二维“Frank球”连续环形任务集合（每个环的半径随任务号按√k增长），以及标准连续学习基准（如EWC、SI、经验回放）与一个联合训练的上界。

**📈 对比分析**

与EWC、SI、经验回放对比，Stefan-CL在10个随机种子上平均准确率达到0.923，遗忘率仅0.021，明显优于EWC（0.716）和SI（0.701），并在不存储原始数据的情况下匹配经验回放（0.940）。

**⚠️ 局限性**

局限在于对非凸或拓扑变化的前沿（如两球合并）时，基于最近点扩展的速度场会失稳，导致前沿被侵蚀；因此目前只能处理单一凸或已知闭合重置目标的前沿，需要进一步研究稳定的速度扩展方法。

---

## 483. RescueBench: Can Embodied Agents Save Lives in the Wild ?

**arXiv ID:** 2606.01848 | [PDF](https://arxiv.org/pdf/2606.01848v1)

**作者:** Kui Wu `[一作]` (Beihang University), Fangwei Zhong `[通讯]` (Beijing Normal University)

**通讯引用:** 1114 | [OpenAlex ID](https://openalex.org/A5081893016)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为RescueBench的面向搜索救援（SAR）的四阶段顺序任务基准，并在多难度级别下评估了多种现有方法的性能；

**💡 创新点**

创新点包括：① 将探索、定位救援、记忆导航和递送四个子任务组合成完整流水线，揭示不同能力之间的级联失败；② 设计了五级难度曲线，系统调节环境复杂度、线索模糊度与空间层级；③ 构建了可扩展的自动化数据收集与注释管线；

**🔧 技术方法**

使用了多模态感知（RGB、深度、语义分割）、基于图谱的导航、视频-视觉语言模型、VLN基础模型、地图驱动策略以及基于视觉语言的运动策略；

**📊 数据集**

基于UnrealZoo平台的七个不同场景（住宅、商业、街道、自然地形）构建的照片级逼真环境；

**📈 对比分析**

对七个学习型基线（包括LLM-YOLO Planner、Uni-NaVid、ViNT、NoMaD、SG-Nav、OmniNav、ROCKET-2）以及Oracle和人类进行评估；结果显示：在最难级别L5下，所有方法均未完成完整任务，探索阶段是主要瓶颈，空间记忆阶段次之；

**⚠️ 局限性**

限制主要在于：目前的架构缺乏有效的开放世界探索逻辑与稳定的空间记忆机制，导致在更大尺度的空间中难以持续搜索和返回；此外，基于地图和路径检索的模型受限于推理延迟，难以覆盖广阔区域。

---

## 484. SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning

**arXiv ID:** 2606.01991 | [PDF](https://arxiv.org/pdf/2606.01991v1)

**作者:** Lichao Wang `[一作]` (Beijing Institute of Technology), Juntao Dai `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SafeMCP，一个服务器端插件，通过预测性世界模型对LLM代理的工具获取进行主动过滤和即时干预，从而在Model Context Protocol (MCP)中抑制代理的power‑seeking行为。

**💡 创新点**

创新点在于将代理与防御交互建模为Cooperative Stackelberg Power Game，将工具过滤作为主动权力约束，并引入双层防御（预过滤+失效干预）以及三阶段训练管道。

**🔧 技术方法**

使用内部世界模型进行前瞻推理，结合安全性与工具过滤的双重可验证奖励、Smooth Tchebycheff scalarization以及RLVR强化学习，构建SafeMCP。

**📊 数据集**

使用了自建的环境动态数据集（ToolEmu与AgentHarm产生的转移数据）、PowerSeeking Bench评测套件以及公开的ToolEmu和AgentHarm基准。

**📈 对比分析**

与多种现有防御（Llama Guard 3、Qwen3Guard-8B、Lakera-ChainGuard、NeMoGuard、AgentMonitor、RL-Guard、Safiron）和多模型（GPT‑4o、GPT‑4o‑mini、Gemini‑2.0‑Flash、Claude‑3.5‑Sonnet、LlaMA‑3.1‑Instruct‑8B）进行对比，SafeMCP在安全率和效用率上均达到或超过基准的Pareto前沿，尤其在power‑seeking和非对抗风险场景中安全率接近1。

**⚠️ 局限性**

局限性包括对环境动态建模的复杂度依赖，跨域迁移的安全先验难以无大规模本地数据即可实现，以及在极端状态下预测误差可能导致误拦截或误放行。

---

## 485. Generalization Limits in Vehicle Re-Identification

**arXiv ID:** 2606.01981 | [PDF](https://arxiv.org/pdf/2606.01981v1)

**作者:** Anis Yassine Ben Mabrouk `[一作]` (Université Paris-Saclay), Rodrigo Verschae `[通讯]` (Universidad Técnica Federico Santa María)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对车辆重识别任务中 VeRi-Wild 与 VeRi-776 数据集的训练/测试分割存在的偏差进行了揭示，并提出了基于车辆类型的见/未见拆分以及同视角/异视角评估协议。

**💡 创新点**

创新点在于通过车辆类型拆分评估泛化能力，并将同视角与异视角重识别拆分为两个子任务，展示了当前最先进模型在未见车型上的显著性能衰退。

**🔧 技术方法**

作者采用了基于 Transformer 的度量学习方法（TransReID、ClipReID 与 RotTrans），并结合对比学习与视角/相机嵌入来构建特征嵌入。

**📊 数据集**

使用的数据集为 VeRi-Wild（40,671 车种，416,314 张图）与 VeRi-776（776 车种，49,357 张图）。

**📈 对比分析**

在见车型的混合视角评估中，三种方法均达到 98–99% 的 mAP；在未见车型或不同视角评估中，mAP 降低 20–30%，表明泛化能力不足。

**⚠️ 局限性**

局限在于只针对两个数据集进行实验，未给出针对未见车型提升泛化的具体改进方案，且评估仍依赖于手工标注的视角划分。

---

## 486. Contrastive Augmented Transformer with Domain-specific Enhancement for Robust Multi-scenario Metal Surface Defect Detection

**arXiv ID:** 2606.01962 | [PDF](https://arxiv.org/pdf/2606.01962v1)

**作者:** Yiyao Liua `[一作]`, Huan Wang `[通讯]` (Tsinghua University)

**通讯引用:** 19806 | [OpenAlex ID](https://openalex.org/A5100332013)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 Contrastive Augmented Transformer（CAT）框架，用于金属表面缺陷检测，结合异构双分支网络、物理启发的水滴增强、多尺度特征融合和加权对比损失，实现对细微多尺度缺陷的高精度定位。

**💡 创新点**

创新点包括：① 采用 Swin Transformer + 冻结 ResNet‑50 的异构双分支设计，兼顾全局语义与局部纹理；② 引入基于金属氧化和腐蚀的物理预设“水滴”增强算法，生成更逼真的缺陷样本；③ 设计了双向多尺度特征融合 FPN（MSF‑FPN），提升细节与语义的协同；④ 结合加权对比损失和硬负样本挖掘，显著提升对模糊缺陷区的判别能力。

**🔧 技术方法**

技术手段：Swin Transformer、ResNet‑50、MSF‑FPN、对比学习（Hard Negative Mining）、自定义 Droplet Augmentation、加权对比损失。

**📊 数据集**

主要使用 KolektorSDD2 作为训练集，评估 KSDD1、Magnetic Tile Defects（MTD）和 Rail Surface Defect Detection（RSDD）等未见域数据集。

**📈 对比分析**

与 SuperSimpleNet、ReContrast、MMR、DFR 等无监督/半监督方法以及 PSPNet、CCNet、DeepLabV3+MobileNet 等语义分割网络对比，CAT 在 KSDD2 上实现像素级 AUROC 99.54%、图像级 AUROC 93.80%，并在跨域数据集上保持优异性能，明显优于基线方法。

**⚠️ 局限性**

局限性：① 使用零填充导致特征稀释和迁移性能下降；② 固定阈值（0.5）进行缺陷分割导致跨域阈值不适配，影响最终的检测准确率；需要开发自适应或无阈值的分割策略。

---

## 487. Toka: A Systems Programming Language with Explicit Resource Semantics

**arXiv ID:** 2606.01974 | [PDF](https://arxiv.org/pdf/2606.01974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 488. SAVMap: Structure-Aided Visual Mapping of Large-Scale 2.5D Manhattan Wireframes from Panoramic Video

**arXiv ID:** 2606.01939 | [PDF](https://arxiv.org/pdf/2606.01939v1)

**作者:** Howard Huang `[一作]` (Nokia Bell Labs), Chen Feng `[通讯]` (NYU)

**通讯引用:** 7867 | [OpenAlex ID](https://openalex.org/A5100699151)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用全景视频摄像头提取结构点，构建仓库货架和灯光的语义线框三维地图；

**💡 创新点**

首次将语义分割与曼哈顿网格约束结合，实现稀疏但语义丰富的结构化映射；

**🔧 技术方法**

技术包括全景视频捕捉、视图渲染、YOLOv8语义分割、结构点跟踪、受限结构光照（SfM）以及GTSAM优化；

**📊 数据集**

使用一小时内收集的60GB全景视频，对一座46排货架、总面积约8000 m²、2 km长的仓库进行测试，并以构造文件生成的地面真值为参考；

**📈 对比分析**

与COLMAP和VGGT等传统与深度学习重建方法相比，SAVMap在单排平均绝对误差仅为4.0 cm，整体四十六排平均误差为4.8 cm，显著优于对比方法；

**⚠️ 局限性**

局限于离线静态映射，需多次通道行走、依赖高质量语义分割，且对极端遮挡或动态环境的鲁棒性尚待验证。

---

## 489. MMG2Skill: Can Agents Distill In-the-Wild Guides into Self-Evolving Skills?

**arXiv ID:** 2606.01993 | [PDF](https://arxiv.org/pdf/2606.01993v1)

**作者:** Xinyu Che `[一作]`, Jiaheng Liu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建MMG2Skill-Bench基准，评估VLM代理将公开多模态手册转换为可执行技能的能力，并提出MMG2Skill闭环框架实现从手册到技能的编译、执行、诊断与修订。

**💡 创新点**

①首次将多模态手册与交互式任务耦合，形成“guide‑to‑skill”评估轴；②闭环框架在不使用基准分数反馈的前提下，通过轨迹诊断实现技能持续改进；③证明原始手册直接作为上下文会削弱性能，而结构化技能提取+轨迹修订能显著提升。

**🔧 技术方法**

多模态技能构造、基于VLM的条件执行、轨迹诊断分析器、技能修订器、基于分析结果的早停决策。

**📊 数据集**

公共网页手册（HTML+图像）与三类任务：GUI控制（OSWorld）、游戏（OpenHA Minecraft/MineStudio）和策略卡牌（RLCard的Doudizhu、Mahjong）。

**📈 对比分析**

对六种VLM骨干（如Gemini、Qwen、GPT-5.5、Kimi、Qwen3.6-Plus、Sonnet）进行对照。MMG2Skill在所有模型–域组合上平均提升12.8–25.3个百分点；结构化提取+修订相较于原始手册或无修订方案带来大幅性能提升；基于分析的早停可在成功可推断任务中节省25–53%尝试。

**⚠️ 局限性**

仅考虑已选手册，未解决手册检索与过滤；评估成本高且对前沿VLM受限；交互式评估耗时，限制实验吞吐量；框架对低质量或不匹配手册的容错性有限。

---

## 490. MT-EditFlow: Reinforcement Learning for Multi-Turn Image Editing with Flow Matching

**arXiv ID:** 2606.01985 | [PDF](https://arxiv.org/pdf/2606.01985v1)

**作者:** Jiahui Huang `[一作]` (Apple), Ying Nian Wu `[通讯]` (University of California, Los Angeles)

**通讯引用:** 20146 | [OpenAlex ID](https://openalex.org/A5101780958)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发 MT-EditFlow 强化学习框架，专为多轮图像编辑设计，并通过流匹配模型优化奖励信号。

**💡 创新点**

将多轮、多奖励视角与流匹配 RL 结合；引入优势广播、思考式 VLM 评估和优势级融合，有效缓解“全局一致性失衡”与曝光偏差问题。

**🔧 技术方法**

使用 Flow‑Matching（GRPO、DiffusionNFT）、LoRA 微调、FSDP/DeepSpeed ZeRO 并行、VLM（Qwen3‑VL‑8B）评估、EdiVal‑CC 评分、EdiVal‑Agent 生成多轮指令链。

**📊 数据集**

基于 Pico‑Banana‑400K 数据集，生成 2,319 张三轮编辑轨迹（9 种编辑类型、12 类语义对象），评估集为 EdiVal‑Bench 与 ImgEdit。

**📈 对比分析**

与多款开源/闭源编辑模型对比；在 EdiVal‑Bench 三轮任务中，FLUX.1‑Kontext‑dev 提升 6.85 分、FLUX.2‑klein‑base 提升 2.90 分；在单轮 ImgEdit 上亦有小幅提升。

**⚠️ 局限性**

限制包括：VLM 评估成本高、评估器仍存在偏差、未验证更长交互序列的效果，且未探索 actor‑critic 等更复杂 RL 方法。

---

## 491. Improved Amenability Bounds for Local Coordination Games

**arXiv ID:** 2606.01963 | [PDF](https://arxiv.org/pdf/2606.01963v1)

**作者:** Ron Peretz `[一作]` (Bar Ilan University), Dean Kraizberg `[通讯]` (Tel Aviv University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在有限社交网络上的纯协调游戏中，证明了平均不一致度小于ε时，图的可分解性参数可缩小到O(ε log(1/ε))。

**💡 创新点**

利用互信息游戏的Shapley值，取代传统方差方法，取得了比先前平方根失效率更好的可分解性上界。

**🔧 技术方法**

信息论（互信息、Shapley值）、Shapley影响分布、总变差距离及极大化耦合理论。

**📊 数据集**

无具体实验数据集，研究完全理论性。

**📈 对比分析**

与之前的(√ε,r)-可分解性结果比较，证明了在{-1,+1}取值下可将损失从√ε降低到O(ε log(1/ε))，在理论上显著改进。

**⚠️ 局限性**

仅适用于二值均匀取值的变量，无法推广到一般实值变量；此外，改进仍有限，无法达到线性ε的可分解性。

---

## 492. Are Economists Open to AI? Text as Data as Survey on Professional Sentiment and Academic Research Trends

**arXiv ID:** 2606.01958 | [PDF](https://arxiv.org/pdf/2606.01958v1)

**作者:** Yi Wang `[一作]` (Renmin University of China), Lei Ge `[通讯]` (Renmin University of China)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5053211572)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 TaDaS（Text as Data as Survey）框架，利用两套不同语料库（专业讨论文本和学术期刊文章）构建“问题-答案”体系，并将自然语言转化为可量化的问卷式态度指标；将此框架应用于研究经济学家对人工智能（AI）的态度变化，探究专业讨论随 AI 在顶级期刊曝光度提升的演变；

**💡 创新点**

创新点在于：①首次将语义检索与主题映射相结合，双重有效信息提取——先在问题语料库中筛选焦点语义邻域，再映射到答案语料库的主题中心；②构建可重复、可追溯的问卷式测量架构，避免传统调查的成本、时效与社会期望偏差；③提供一种通用的“文本转问卷”方法，可用于多学科、跨语料库的态度研究；

**🔧 技术方法**

采用的技术包括：All‑RoBERTa Large v1 句子嵌入；UMAP+HDBSCAN 进行语义聚类；BERTopic 风格的 c‑TF‑IDF 生成主题关键词；Gemma‑3‑12B‑IT 作为指令调优的解码器对回复进行多维度态度评分（开放、负面、毒性、傲慢、好奇、困惑）等；

**📊 数据集**

数据集：1) 经济学就业论坛 EJMR，约 130 万条帖子（2010‑2024）；2) 53,585 篇来自 ABS 4* 级顶级经济与金融期刊的论文，构成答案语料库；3) 32 个由论文聚类生成的主题系统；

**📈 对比分析**

对比方法：先做跨截面基线（AI 讨论 vs 其它研究讨论），再构建交互式回归（AI 主题暴露 × AI 趋势），加入线程情绪、论坛指标与年份固定效应等控制；结果显示 AI 讨论起始更消极、开放度低，但随着 AI 在顶刊曝光度上升，所有六维态度均趋向更积极、开放；该模式在多种鲁棒性检验（全主题趋势控制、预 2023 样本）下保持一致，表明结果稳健；

**⚠️ 局限性**

局限性：①仅利用 EJMR 论坛，可能不代表全部经济学家态度；②仅以顶刊论文曝光度为 AI 关注度指标，忽视其他传播渠道；③模型识别的是协同趋势，未证明因果机制；④缺乏人工编码或传统问卷的验证；⑤方法在学术经济学背景下验证，其他专业领域需进一步检验。

---

## 493. Flow-Transformed Implicit Processes for Function-Space Variational Inference

**arXiv ID:** 2606.01954 | [PDF](https://arxiv.org/pdf/2606.01954v1)

**作者:** Luis A. Ortega `[一作]` (Aalborg University), Thomas D. Nielsen `[通讯]` (Aalborg University)

**通讯引用:** 5994 | [OpenAlex ID](https://openalex.org/A5080416900)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在函数空间上使用流变换的隐式过程变分推断方法FTIP；

**💡 创新点**

通过在代理系数上引入可逆流，使后验更具表达性，能够捕捉多模态与偏斜的不确定性；

**🔧 技术方法**

使用正则化的高斯基函数先验采样构建有限维代理，再将正态基分布通过可逆流变换得到后验分布；

**📊 数据集**

在一维合成回归、UCI回归数据集（Boston、Concrete等）、YearPredictionMSD以及人行轨迹长度预测等任务上评估；

**📈 对比分析**

与VIP、FBNN、MFVI、TFSVI等方法对比，FTIP在NLL、CQM等分布性指标上优于基线，点预测指标相近；

**⚠️ 局限性**

需要更多训练时间和调参，后验表达性对简单或近似高斯任务收益有限，且仍受有限采样代理的限制。

---

## 494. Closed-Form Pose Estimation of Endoluminal Medical Devices via Gradiometer-Based Electromagnetic Localization System

**arXiv ID:** 2606.01946 | [PDF](https://arxiv.org/pdf/2606.01946v1)

**作者:** Zhiwei Wu `[一作]`, Jinhui Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e15e3743-5ee0-4d5f-813d-d146868082fc`

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

## 495. SCAPO: Self-Supervised Category-Level Articulated Pose Estimation from a Single 3D Observation

**arXiv ID:** 2606.01940 | [PDF](https://arxiv.org/pdf/2606.01940v1)

**作者:** Can Zhang `[一作]` (National University of Singapore), Gim Hee Lee `[通讯]` (National University of Singapore)

**通讯引用:** 9772 | [OpenAlex ID](https://openalex.org/A5071967339)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种自监督框架，能够从单张 RGB‑D 观测中恢复物体的标准几何、刚性部件分割以及关节的枢轴、轴线和运动状态，实现类别级的关节建模。

**💡 创新点**

创新点包括：使用 SE(3) 等变换自编码器实现全局姿态去耦和共享标准空间；设计基于骨骼的混合蒙皮变形来联合预测部件运动；通过循环一致性与跨空间对齐，结合多项正则化，使几何与运动实现解耦并获得可解释的关节参数。

**🔧 技术方法**

核心技术包含：Vector‑Neuron 自编码器、SE(3) 等变换网络、关键点网络、姿态网络、混合蒙皮权重生成、循环一致性损失、跨空间对齐损失以及关键点分割、关节方向对齐、关节‑形状相近性与关节‑边界吸引等正则化项。

**📊 数据集**

使用合成 HOI4D 与 Shape2Motion 数据集（包含多种关节物体类别）以及从公开资源采集的五类实测 RGB‑D 数据（手提箱、抽屉、手提电脑、剪刀、篮子）进行评估。

**📈 对比分析**

与 EAP、OP‑Align 等自监督方法以及 3DGCN、NPCS‑EPN 等监督基准进行对比；在合成数据上取得最高的部件 IoU、最低的关节方向/枢轴误差；在真实数据上实现了超过 OP‑Align 的 mAP 与部件 mAP，整体性能优于或匹敌监督方法。

**⚠️ 局限性**

局限性包括：假设固定的部件与关节数目，导致对新类别的适应性受限；在极端形变或严重遮挡下 SE(3) 编码器性能下降；当运动线索弱或分割噪声大时关节推断精度会显著下降。

---

## 496. What to Format and How: A Benchmark and Workflow Approach for Document Formatting

**arXiv ID:** 2606.01936 | [PDF](https://arxiv.org/pdf/2606.01936v1)

**作者:** Shihao Rao `[一作]` (Chinese Academy of Sciences), Can Ma `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 1766 | [OpenAlex ID](https://openalex.org/A5033649206)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DocFormBench评测集和DocFormFlow工作流格式化方法，以解决内容感知的文档格式化任务。

**💡 创新点**

创新点在于引入面向内容感知的评测基准、四阶段分离定位与执行的工作流，并提供效率与精度指标。

**🔧 技术方法**

使用大型语言模型（如GPT‑5、GLM‑4.6、Qwen3等）结合工具调用、功能调用、视觉输入以及验证循环实现。

**📊 数据集**

使用DocFormBench共500条中英双语、12类文档的手工验证实例，覆盖多种格式化需求。

**📈 对比分析**

与GUI/ API基线比较，DocFormFlow在四大指标上均优于对手，格式化准确率提升至70%+，平均token消耗降低70%+。

**⚠️ 局限性**

局限性包括对最新大模型未评估、推理成本高（约10万token）、仅支持Windows/Word接口且工具覆盖不全。

---

## 497. An NLP-Driven Framework for Curriculum-Labor Market Alignment: Schema-Constrained LLM Extraction, ESCO-Anchored Semantic Matching, and Multi-Dimensional Gap Quantification

**arXiv ID:** 2606.01982 | [PDF](https://arxiv.org/pdf/2606.01982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 498. 3rd Place at CVPR 2026 CASTLE Challenge: Agentic Multi-View Long-Context Video Understanding via Hierarchical Knowledge Graph Retrieval

**arXiv ID:** 2606.01933 | [PDF](https://arxiv.org/pdf/2606.01933v1)

**作者:** Raghad Albusayes `[一作]` (TAHAKOM), Munirah Alyahya `[通讯]` (TAHAKOM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的多模态多摄像机代理框架，用知识图谱和分层检索实现长视频问答；

**💡 创新点**

核心创新在于将长视频转化为带时间标注的知识图谱，并设计层次化检索与代理式查询分解流程；

**🔧 技术方法**

使用Gemini 2.5 Pro、LangGraph、LLMGraphTransformer等大模型实现统一音视信息提取、知识图谱构建及代理推理；

**📊 数据集**

在CASTLE 2026挑战中使用了600小时、15个摄像头同步录制的多模态视频数据集；

**📈 对比分析**

相较于基线准确率从43%提升到55%，在挑战排行榜中获得第三名，表现优于绝大多数参赛团队；

**⚠️ 局限性**

局限包括对GPU显存和大模型推理时间的高需求、对知识图谱构建质量的敏感性以及对非视觉信息的依赖不足。

---

## 499. CARTE: A Benchmark for Mapping Language Model Knowledge Across France

**arXiv ID:** 2606.01995 | [PDF](https://arxiv.org/pdf/2606.01995v1)

**作者:** Sarah Almeida Carneiro `[一作]` (École Polytechnique Institut Polytechnique De Paris), Michalis Vazirgiannis `[通讯]` (École Polytechnique Institut Polytechnique De Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CARTE与CARTE-Linguistic两个多选评测基准，用以评估大型语言模型在法国13个大都市区内的细粒度地域知识与语言变体推理能力。

**💡 创新点**

创新点在于填补了仅有国家层面或翻译基准的空白，针对法国语境中的区域差异与地方语言特点构建了高细节、多主题的文化与地域化评测。

**🔧 技术方法**

技术方案主要包括基于Gemini 3 Flash等LLM的自动化题目生成、LLM辅助的自动过滤与人工审核管道，确保问题的地域性与逻辑严谨。

**📊 数据集**

使用的数据集来源于公开的统计、历史与地理文献，经过人工挑选与LLM过滤后生成约15K道题（CARTE）与233道语言变体题（CARTE-L），覆盖14主题域与13地区。

**📈 对比分析**

实验在0/1/3-shot设置下对27个模型（法国本土、欧洲多语言与通用大型模型）进行评测，整体准确率最高约75%，易难分层显示区域与主题差异显著，说明模型在细粒度地域推理上仍存在显著缺口。

**⚠️ 局限性**

局限性包括缺乏全面人工评审、区域分布不均、可能的刻板偏见与记忆偏差，评测结果不适用于高风险决策且未充分检验模型真正的推理能力。

---

## 500. Graph Edit Distance Formulation for the Vehicle Routing Problem: Theory and Analysis

**arXiv ID:** 2606.01987 | [PDF](https://arxiv.org/pdf/2606.01987v1)

**作者:** Adel Dabah `[一作]` `[通讯]` (Forschungszentrum Juelich), Adel Dabah (Forschungszentrum Juelich)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种将车辆路径规划(VRP)转化为图编辑距离(GED)最大化的问题，通过在完整实例图上删除边来最小化路径成本。

**💡 创新点**

三项创新：① 证明VRP最小化与GED最大化等价；② 将Clarke–Wright（CW）节省值与每一次合并的GED增量等价；③ 提出从GED近似比到VRP成本上界的转移定理。

**🔧 技术方法**

核心技术包括：图编辑距离框架、边删除成本模型、理论证明（等价性、合并分解、近似转移）、多启动CW启发式、2-opt改进、以及基于GED的误差分解和潜在GNN边预测。

**📊 数据集**

使用90个已知最优解的标准CVRP基准实例（Augerat、Christofides、Fisher、Meisel、TSPLIB等），共涵盖多种规模与容量设置。

**📈 对比分析**

与最优解直接比较：GED近似比平均≥0.9991（即≥99.9%），成本误差平均约4.3%（最佳解约3.4%），通过GED分解揭示误差主要来自分配错误。与传统CW单启动相比，多启动加2-opt可显著提升GED占比并降低成本。

**⚠️ 局限性**

局限性：仅验证于CVRP，未涵盖时窗或随机需求等变体；方法依赖于完整图的边删除模型，可能对大规模实例计算量大；未提出自适应GNN模型的完整实现，仅给出潜在方向。

---

## 501. A Simple Hierarchical Causality Primer

**arXiv ID:** 2606.01979 | [PDF](https://arxiv.org/pdf/2606.01979v1)

**作者:** Tim Gebbie `[一作]` (University of Cape Town), Tim Gebbie `[通讯]` (University of Cape Town)

**通讯引用:** 484 | [OpenAlex ID](https://openalex.org/A5063137805)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

**🎯 论文内容**

提出了一个离散化的层级因果模型框架，明确区分了层级结构、局部动力学、上层约束（演员实例与因果类）以及事件计时；

**💡 创新点**

创新点在于将“上层因果类”与“演员实例”显式化为约束对应关系，构造了“聚合等价”与“因果等价”之间的区别，避免了传统粗粒化与自下而上动力学混淆；

**🔧 技术方法**

使用离散时间马尔可夫核、聚合映射、子算子（事件计时）以及演员-因果类的约束映射等形式化工具；

**📊 数据集**

本工作为理论性研究，未使用具体数据集；

**📈 对比分析**

未做实验或性能对比，仅通过定义与推导说明框架的完整性与区别；

**⚠️ 局限性**

局限性包括：仅在离散时间下讨论，未给出连续时间或实际系统实例；缺乏经验验证、因果强度量化和算法实现细节；

---

## 502. AutoBG: A Board Game Design Assistant with Interactive Ideation, Iterative Rulebook Generation, and Individualized Feedback

**arXiv ID:** 2606.01976 | [PDF](https://arxiv.org/pdf/2606.01976v1)

**作者:** Zizhen Li `[一作]` (Alaya Lab), Kaipeng Zhang `[通讯]` (Alaya Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个端到端的棋盘游戏设计助手AutoBG，涵盖交互式构思、规则书生成、诊断与闭环迭代、个性化玩家反馈。

**💡 创新点**

关键创新包括：MDA框架驱动的专用评估模型BG-Critic、Verifier‑Gated Iteration闭环改进，以及利用150真实玩家档案的个性化反馈模块BG‑Persona。

**🔧 技术方法**

采用大型语言模型（Qwen3.5‑27B等）配合LoRA微调，结合多任务训练（生成、修订、诊断、比较）以及对话式交互模块。

**📊 数据集**

基于公开的2.2K结构化规则书和180K玩家评论构建的数据集，随后扩增生成约4.5K草稿及1.5K规则书。

**📈 对比分析**

与GPT‑5.4、Gemini‑3.1‑Flash等通用LLM基线对比，AutoBG在规则书质量（评分≈7.08）和无缺陷率（36.7%）上接近真实出版游戏，并在用户研究中得到高满意度。

**⚠️ 局限性**

限制包括缺乏真实物理游戏测试、单一文本模式缺失视觉/图形支持，以及个性化反馈模型尚未覆盖群体动态与长期喜好变化。

---

## 503. Market-Based Replanning for Safety-Critical UAV Swarms in Search and Rescue Missions

**arXiv ID:** 2606.01970 | [PDF](https://arxiv.org/pdf/2606.01970v1)

**作者:** Luiz Giacomossi `[一作]` (Mälardalen University), Håkan Forsberg `[通讯]` (Mälardalen University)

**通讯引用:** 674 | [OpenAlex ID](https://openalex.org/A5029097575)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于逆拍卖的分布式任务重规划架构（IRDS），实现了无人机队伍在搜索救援任务中对个体失效的自愈与重分配。

**💡 创新点**

创新点在于将失效视为市场事件的液化机制、简化的距离加权逆拍卖、几何共识验证以及低通信开销的分布式状态机。

**🔧 技术方法**

采用逆拍卖竞价、距离加权成本函数、几何三角共识投票、Boustrophedon覆盖路径、人工势场碰撞规避以及 PyBullet 物理仿真。

**📊 数据集**

使用 8×8 网格（64 个搜索区块）与 8 名 Crazyflie 2.x 无人机的仿真环境，进行 100 次 Monte‑Carlo 试验并注入单/双节点失效。

**📈 对比分析**

与 N=2、4、8 的基线和失效情景对比，成功率维持 93–100%，任务重分配延迟约 13–18 s（约占总时长 4–5%），搜索时间从 721 s 降至 223 s，能耗略升 21%。

**⚠️ 局限性**

局限性包括能耗增加、验证投票缺乏超时机制导致死锁风险、仅在仿真中验证通信延迟、以及仅针对同质无人机的简化模型。

---

## 504. Training Prompt Matters: State-Adaptive Optimization for Robust Fine-Tuning

**arXiv ID:** 2606.01967 | [PDF](https://arxiv.org/pdf/2606.01967v1)

**作者:** Wenhang Shi `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**通讯引用:** 6542 | [OpenAlex ID](https://openalex.org/A5008721449)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了训练时提示（prompt）对大语言模型微调的影响，并提出了一种基于模型状态的提示优化方法SAPO；

**💡 创新点**

创新点在于揭示即使语义相同的训练提示也能显著改变遗忘与泛化表现，并证明预更新损失能预测并挑选出更优提示；

**🔧 技术方法**

方法主要采用预更新损失评估、提示生成器（如Gemini、GPT、Qwen）生成同义提示、LoRA微调以及梯度角度分析等技术；

**📊 数据集**

实验使用SuperNI和TRACE两大持续学习基准，涵盖分类、生成及混合任务；

**📈 对比分析**

与LoRAInc、O-LoRA、EWC、InsCL等主流持续学习方法对比，SAPO在AP、BWT、FWT指标上均实现显著提升，尤其在弱基线上提升更为显著；

**⚠️ 局限性**

局限性在于对提示生成器和预更新损失预测的依赖，难以处理极端多样化任务及可能导致的提示不确定性。

---

## 505. Eyettention II: A Dual-Sequence Architecture for Modeling Fixation Location, Within-Word Landing Position, and Fixation Duration in Reading

**arXiv ID:** 2606.01964 | [PDF](https://arxiv.org/pdf/2606.01964v1)

**作者:** Shuwen Deng `[一作]` (University of Potsdam), Lena A. Jäger `[通讯]` (University of Zurich)

**通讯引用:** 1618 | [OpenAlex ID](https://openalex.org/A5025151670)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种名为 Eyettention 2 的轻量级双序列生成模型，能够按顺序预测阅读过程中的词索引、单词内落点和停顿时长，进一步推出可模拟个体读者的 Eyettention 2_reader。

**💡 创新点**

创新点在于：①首次将双序列编码（词序列与时间序列）与跨序列注意力相结合，实现完整扫描路径的端到端生成；②通过额外的落点和停顿时长预测模块，显著拓宽模型的功能覆盖；③引入读者身份嵌入实现个体化预测；④对多种解码策略、预训练/微调与层级特征的系统评估。

**🔧 技术方法**

核心技术包括双向 LSTM 与单向 LSTM 的双编码器、跨序列注意力窗口、局部高斯平滑、softmax 与回归解码器、教师强制训练、Nucleus/温度/贪婪解码、基于预训练语言模型（BERT/ RoBERTa）生成词嵌入，以及 MLP 预测模块。

**📊 数据集**

使用四个公开眼动数据集：中文北京句子语料 BSC、英文 CELER L1、英文 ZuCo NR 与 ZuCo 2.0 NR；这些数据涵盖不同文字系统、实验设置与阅读任务。

**📈 对比分析**

与传统认知模型（E-Z Reader、SWIFT）及机器学习基线（Eyettention 1、BERT+线性层、Last LightGBM）进行对比，评估指标包括 NLL、MAE、MultiMatch 各维度、统计显著性检验。Eyettention 2 在词索引、停顿时长和落点预测均显著优于基线（提升 2–6% 以上），在 MultiMatch 向量/长度/位置/持续时间维度也超越认知模型，且与人类扫描路径相似度更高。

**⚠️ 局限性**

局限性包括：①解码时生成的扫描路径方差低于人类，可能导致功效分析过度乐观；②使用双向语言模型产生的词嵌入包含后视信息，与人类实际读者仅使用已读信息不完全一致；③对极端实验设置的跨语料迁移仍存在性能下降；④模型仍相对简单，未充分利用自注意力或更大规模数据。

---

## 506. WALL-WM: Carving World Action Modeling at the Event Joints

**arXiv ID:** 2606.01955 | [PDF](https://arxiv.org/pdf/2606.01955v1)

**作者:** Shalfun Li `[一作]`, Qian Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于事件（semantic event）而非固定长度动作块的全新世界-动作模型（WALL-WM）训练与推理框架，并实现了可同时支持事件驱动与统一块推理的双模式推理策略。

**💡 创新点**

创新点主要包括：
1) 以事件为对齐单元的WAM预训练，保留视频先验并实现视频-动作-语言三模态几何保持；
2) 引入跨模态层耦合的多视角DiT与动作DiT，并通过时间对齐与跨模态注意力实现视频与动作的协同推理；
3) 设计两种推理模式：事件模式（可变长度）和统一模式（固定长度），并通过梯度连续的阶梯式潜在链式思考（Staircase latent CoT）实现高效长序列推理；
4) 构建五轴（来源、同步、层级标注、聚类、增强）数据生态系统，配合层级注释与聚类平衡采样，显著提升样本多样性与稀有事件覆盖；
5) 通过分阶段训练（视频先预训练→动作先预训练→VLM对齐→潜在CoT蒸馏→可选块级微调）实现先验保持与可扩展性。

**🔧 技术方法**

技术方法包括：
- 事件级无监督视频-动作去噪（flow‑matching）与跨视角注意力；
- 动作DiT层级跨模态耦合与时间对齐；
- 多视角自注意力与Camera RoPE、视角几何遮罩；
- 事件级层次化标题标注与聚类平衡采样；
- VLM文本对齐（Qwen3.5‑9B）与潜在CoT蒸馏；
- 两种推理模式（事件与统一）与阶梯式潜在CoT；
- 大规模分布式训练（DMuon、分布式调度、合并卷积核、FP8量化、分布式分块训练）。

**📊 数据集**

使用的数据集与来源：
- 互联网视频（OpenVID 等 120 万+片段）
- 视角第一人称与人类动作视频（Ego4D、EPIC‑KITCHENS 等）
- 无机器人 UMI‑Style 录制（XRZero‑G0 系列）
- 机器人自采集与混合远程遥控数据（QUANTA 系列桌面双臂与移动平台）
- 人为干预与失败恢复片段
- 所有数据统一经过层级注释（Task/Subtask/Action/Segment）和时间同步校准。

**📈 对比分析**

比较方法与性能：
- 以 WorldArena 评测为基准，评估视频生成的视觉质量、运动质量、语义一致性、物理可行性、跨视角一致性与 3D 觉知；WALL‑WM 在运动质量、语义一致性、物理可行性上显著优于 Wan2.1/2.2 以及其他基线；
- 真实机器人测试（多桌面平台）涵盖 Diverse Manipulation、Reasoning Manipulation 与 Dexterous Manipulation 三套任务；在 Task‑Progress（0‑100）评分上，事件模式平均分 75.86，显著高于统一模式（63.00）及传统基线（π_0.5 55.64，DreamZero 39.97，LingBot‑VA 29.71）。
- 在 3D 识别与多视角一致性评测中，WALL‑WM 取得接近最佳的 probe 与对应误差，显示其视角一致与 3D 结构保留能力。

**⚠️ 局限性**

局限与挑战：
- 需要海量多模态视频与对齐标签，收集与标注成本高；
- 事件划分与层级标注仍依赖人工或半自动策略，错误划分可能影响预训练效果；
- 目前模型对非事件式任务（例如高度自由的运动规划）仍需进一步适配；
- 在极端低帧率或极短动作窗口下的时间对齐与跨视角注意力可能不够稳健；
- 复杂的多视角几何遮罩与跨视角注意力在大规模部署中仍带来额外计算开销；
- 仍需更强的跨机器人、跨平台迁移能力，尤其在硬件差异显著时的可泛化性。

---

## 507. A Unified E2E Energy Efficiency Testing Framework for Open RAN

**arXiv ID:** 2606.01931 | [PDF](https://arxiv.org/pdf/2606.01931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 508. Learning Action-Conditional and Object-Centric Gaussian Splatting World Models for Rigid Objects

**arXiv ID:** 2606.01950 | [PDF](https://arxiv.org/pdf/2606.01950v1)

**作者:** Jens U. Kreber `[一作]` (University of Augsburg), Joerg Stueckler `[通讯]` (University of Augsburg)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5085424314)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 MRO-GWM，利用基于物体中心高斯 splatting 的场景表示和时空 Transformer 预测多物体刚体运动，支持部分观测。

**💡 创新点**

将物体感知的高斯 splatting 与可压缩锚点结合，设计新的时空注意力块，并实现基于动作的世界模型用于非抓取操控。

**🔧 技术方法**

使用物体中心高斯 splatting、锚点变换、空间+时间+时空注意力的 Transformer、旋转矩阵正交化、MLP 锚点编码以及基于模型的 MPC（iCEM）。

**📊 数据集**

采用 Maniskill 仿真平台生成的桌面场景，包含 YCB 物体集合（如 YCB-100-100、YCB-50-200、YCB-200-50、YCBV-100-100 等）。

**📈 对比分析**

与现有多物体动力学学习基线对比，位置误差约 0.45–0.52 cm，旋转误差 5.6–7.8°；在两项非抓取任务的 MPC 中成功率分别达到 77% 与 66%，并显著降低目标距离。

**⚠️ 局限性**

需已知物体掩码与位姿，局限于合成数据；仅考虑刚体动力学，未处理柔性形变；实时控制性能待进一步提升。

---

## 509. Rank-Constrained Deep Matrix Completion for Group Recommendation

**arXiv ID:** 2606.01948 | [PDF](https://arxiv.org/pdf/2606.01948v1)

**作者:** Mubaraka Sani Ibrahim `[一作]` (African University of Science and Technology), Isah Charles Saidu `[通讯]` (Baze University)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5008087589)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种名为Group RC‑DMC的组推荐框架，融合低秩矩阵补全与Set‑Transformer注意力聚合，以同时预测个体与组级评分；

**💡 创新点**

创新点在于在自编码器结构中显式加入低秩正则化（Soft‑Impute预热+核范数约束）并利用Set‑Transformer对组成员进行非线性、可置换不变的聚合；

**🔧 技术方法**

主要技术包括：核范数软阈值矩阵补全、线性编码器‑解码器、周期性奇异值阈值（SVT）投影、Set‑Transformer（SAB+PMA）与梯度下降；

**📊 数据集**

使用MovieLens 100K与Goodbooks-10k两大真实数据集，分别含约1k用户、5百物品与10k物品；

**📈 对比分析**

与传统加权前/后因式分解（WBF、AF）做对比，Group RC‑DMC在RMSE、组级Precision/Recall/F1上均达或略优于基线，尤其在召回率方面表现突出；

**⚠️ 局限性**

局限性包括：仍未处理冷启动与侧信息，缺乏对更大规模数据集（如Netflix、Amazon）的实证评估，且SVT运算在极稀疏场景下可能成为瓶颈。

---

## 510. Unified Driving Tokens: Representation- and Geometry-Guided Discrete Tokenizer for Driving World Models and Planning

**arXiv ID:** 2606.01935 | [PDF](https://arxiv.org/pdf/2606.01935v1)

**作者:** Ziyang Yao `[一作]` (Peking University), Huijing Zhao `[通讯]` (Xiaomi EV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一种统一离散视觉词表，结合冻结的 DINO 表示与相邻帧几何监督，既能用于规划读取器，又能直接支持自回归世界模型；

**💡 创新点**

创新点在于（1）在词表学习中加入表示导向与几何增强的多目标监督；（2）采用多码本量化缓解容量瓶颈；（3）实现同一词表同时满足规划和生成两大任务；

**🔧 技术方法**

技术手段包括 VQ‑VAE 变体、预训练 DINO 特征对齐、RGB 重建加感知与对抗损失、硬量化 + EMA 代码表更新、交叉帧深度与相对位姿监督、多码本量化、GPT 风格自回归 Transformer 以及轻量规划解码器；

**📊 数据集**

实验基于 NAVSIM 基准（来源于 OpenScene/nuPlan 的约 120 小时驾驶日志）；

**📈 对比分析**

在 NAVSIM 测试集上，所提词表在 rFID/PSNR/SSIM 与 DINO 语义一致性指标上均优于现有方法；使用同一词表的规划解码器在单视角场景下 PDMS 最高达 91.8；自回归世界模型在 FID/FVD 指标上亦优于基线；

**⚠️ 局限性**

局限性包括：几何监督仅覆盖相邻帧，缺乏长时序动态建模；多码本量化提升了模型复杂度；对多模态传感器（如 LiDAR）与极端稀疏数据的适配性仍待进一步验证。

---

## 511. Mitigating Bias in Locally Constrained Decoding via Tractable Proposals

**arXiv ID:** 2606.01926 | [PDF](https://arxiv.org/pdf/2606.01926v1)

**作者:** Meihua Dang `[一作]` (Stanford University), Stefano Ermon `[通讯]` (Stanford University)

**通讯引用:** 24907 | [OpenAlex ID](https://openalex.org/A5091179481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在大语言模型生成任务中，提出全局约束解码（GCD）和概率全局约束解码（P-GCD），将它们作为 Sequential Monte Carlo（SMC）采样的 proposal 与 potential，保证在给定 token 数内满足逻辑/结构约束。

**💡 创新点**

创新点在于：① 将有限自动机（FA）张量化并在 GPU 上高效执行，实现 GCD；② 利用 HMM 对 LM 进行概率近似，并通过电路乘积与 FA 结合，得到 P‑GCD，融合约束的逻辑与概率信息；③ 将这些构造的 proposal/potential 与 SMC 结合，显著提升采样效率与质量。

**🔧 技术方法**

技术手段包括：自动机张量化、前向/后向消息推理、HMM 近似、Kronecker 乘法、GPU 张量运算、SMC 采样及重采样、温度控制与重排等。

**📊 数据集**

实验数据集：xLAM（函数调用，JSON 与 Python 语法）、CommonGen（关键词生成）、Spider（文本到 SQL 生成）。

**📈 对比分析**

与拒绝采样、LCD、LCD+SMC 等基线相比，P‑GCD 在相同粒子数下收敛更快、生成质量更高，约束满足率始终 100%，且在低粒子数 regime 下粒子数可减少数倍，整体性能显著提升。

**⚠️ 局限性**

局限性包括：P‑GCD 需要先训练 HMM，训练成本和推理时的算力消耗较大；GCD/P‑GCD 仅适用于可用有限自动机表达的约束；对大词表或非常长的序列 GPU 内存瓶颈仍存在；当粒子数非常大时，优势减小。

---

## 512. Algorithmic algorithm development with LLMs: A Case Study on LLM-Usage for Contraction Order Optimization in Tensor Networks

**arXiv ID:** 2606.01975 | [PDF](https://arxiv.org/pdf/2606.01975v1)

**作者:** Fabian Hoppe `[一作]` (German Aerospace Center), Philipp Knechtges `[通讯]` (German Aerospace Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究通过OpenEvolve框架对张量网络收缩顺序优化问题进行实验性探讨，考察LLM选择、评估指标与测试样本对进化结果的影响；

**💡 创新点**

创新点在于系统化评估LLM驱动的演化编码在特定算法改进任务中的有效性，并揭示了评估设计与测试样本对优化结果的关键作用；

**🔧 技术方法**

采用的技术包括OpenEvolve的LLM演化编码、cotengra张量网络规划器、以及多种开源LLM模型（如GPT-OSS-20B、Qwen、Gemma等）；

**📊 数据集**

实验数据集为5000个随机生成的周期MPS张量网络，分为小、中、大三类（每类5000个），并使用“small”“middle”“large”“all”四种子集进行训练与评估；

**📈 对比分析**

结果与cotengra“cheap”基线及其改进版本（cmaes、optuna）比较，发现GPT-OSS-20B在小规模网络上可实现约1.15倍FLOP压缩，且在中等规模网络上平均可压缩约10^1.67倍，但在大型网络上性能下降；

**⚠️ 局限性**

主要局限包括仅使用单一框架（OpenEvolve）、缺乏闭源LLM测试、实验规模有限、评估指标单一（仅FLOP计数），以及对生成代码可读性与运行时开销关注不足。

---

## 513. VET: A Framework for Analyzing AI Discourse

**arXiv ID:** 2606.01929 | [PDF](https://arxiv.org/pdf/2606.01929v1)

**作者:** Meredith Ringel Morris `[一作]` `[通讯]`, Meredith Ringel Morris

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并阐述了VET框架，系统地将AI话语按情感倾向(valence)、效能(efficiency/ effectiveness)和发展轨迹(trajectory)三维度进行分类与分析，随后利用该框架对美国主流媒体中的四类AI叙事——AI炒作、AI末日、AI否认和AI常态——进行识别、比较与批判；

**💡 创新点**

创新点在于将AI话语视角拆解为情感、效能与轨迹三维度的统一框架，既补充了以往单一维度（如风险或技术层面）的分类，又提供了可操作的评估工具，可用于评估和“审核”公众与媒体的AI论述；

**🔧 技术方法**

技术上主要采用定性研究方法——理论框架构建、文献综述、案例分析与多维度对照，未使用算法或机器学习模型；

**📊 数据集**

本文未使用标准数据集，而是依赖公开的媒体报道、行业声明与学术出版物中的案例与引用，对美国本土和部分全球视角的AI话语进行抽样和阐释；

**📈 对比分析**

通过VET框架，将四类叙事映射至二维格局，进行对比分析；尽管缺乏量化指标，但作者通过对每类叙事的影响范围、可能误导性与教育价值进行评估，展示了该框架在提高AI素养和公共讨论质量方面的潜在效用；

**⚠️ 局限性**

局限性包括：1）框架缺乏经验验证与客观指标；2）依赖作者主观选择的案例与文献，可能存在样本偏差；3）仅聚焦美国主流媒体，对其他地区的多样化叙事关注不足；4）未对不同技术路径或法律监管环境进行系统性跨域分析；5）对框架在实际政策或媒体编辑实践中的可操作性未进行实证检验。

---

## 514. Teaching Synchronous Dataflow Modelling with Learn-Heptagon

**arXiv ID:** 2606.01928 | [PDF](https://arxiv.org/pdf/2606.01928v1)

**作者:** Pierre-Loïc Garoche `[一作]` (Fédération ENAC ISAE-SUPAERO ONERA, Université de Toulouse), Basile Pesin `[通讯]` (Fédération ENAC ISAE-SUPAERO ONERA, Université de Toulouse)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款基于Heptagon编译器的Web应用Learn-Heptagon，用于教学同步数据流编程（Lustre）与模型检验，提供交互式编辑、实时模拟、属性验证与自动纠正等功能；

**💡 创新点**

创新点在于将Heptagon编译为JavaScript实现浏览器端实时模拟，结合Kind 2进行在线模型检查，形成无安装、跨平台、自动保存和导入导出的完整教学平台，并支持同步观察者与 assume-guarantee 合约验证及自动纠正功能；

**🔧 技术方法**

采用Heptagon（OCaml）与Js_of_ocaml将编译器移植到浏览器，使用Ace代码编辑器实现语法高亮与错误提示；后端用OCaml的conduit实现用户认证与笔记本存储；模型检查则调用Kind 2 + Z3；前后端通过JSON交互；

**📊 数据集**

主要使用教学用笔记本（Course 1、Course 4等）和学生提交的 Lustre 程序进行实验，未涉及公开数据集；

**📈 对比分析**

在ENAC课程中通过实验观察到，学生使用 Learn-Heptagon 能在几秒内完成模型检查，模拟实时性足够满足课堂需求；与传统命令行工具相比，用户体验提升，减少工具安装与配置时间；具体性能指标未给出，但效果被评为良好；

**⚠️ 局限性**

局限性包括：自动纠正仅适用于不含状态机和数组的程序；Heptagon→Kind 2 翻译不完整，导致部分程序无法验证；缺少教师面板与统计功能；新笔记本需手动修改源码后重编译；未来需要完善翻译、协作与防作弊机制。

---

## 515. Resonant Context Anchoring: Decoupling Attention Routing and Signal Gain at Inference Time

**arXiv ID:** 2606.01923 | [PDF](https://arxiv.org/pdf/2606.01923v1)

**作者:** Mingkuan Zhao `[一作]` (Xi'an Jiaotong University), Yuheng Min `[通讯]` (Tsinghua University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5109786296)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种推理时干预方法 Resonant Context Anchoring（RCA），通过在自注意力机制中对值向量进行动态放大，以提升外部上下文信号在残差流中的能量，从而抑制大语言模型的事实幻觉。

**💡 创新点**

创新点在于将注意力的路由决策（softmax分布）与信息幅度（值向量大小）解耦，利用原始预softmax注意力分数作为语义共振度量，并通过 Softplus 非线性放大函数对相关值向量进行增益控制，从而提升信噪比（SNR）而不改变概率分布，既保持语法与流畅性，又有效对齐上下文。

**🔧 技术方法**

使用技术包括：Transformer 残差流信号动力学建模、非线性 Rectifier（Softplus）与 γ 参数控制的增益字段、在值矩阵上做逐元素乘法放大、无额外训练、低开销的推理时插件。

**📊 数据集**

实验数据集覆盖多任务：XSum（摘要）、NQ-Swap（对立事实）、MemoTrap（记忆冲突）、FactKB、AlignScore、ROUGE-L、TriviaQA、TruthfulQA、PopQA 等，模型使用 Llama-3-8B-Instruct 与 Llama-3-70B-Instruct。

**📈 对比分析**

方法与传统贪婪解码做对比，RCA 在事实一致性指标（FactKB、AlignScore）和知识冲突任务（NQ-Swap EM、MemoTrap Micro/Macro Acc）上均有显著提升，摘要任务中保持或略升 ROUGE‑L，且未出现显著的流畅度或困惑度下降；对闭卷通用 QA（TriviaQA、TruthfulQA）几乎无性能差异，证明安全性。

**⚠️ 局限性**

局限性：需要为不同模型或任务范围调节 γ 超参，虽然在 Llama‑3-8B/70B 之间保持稳定，但仍需小量验证；理论基于残差流线性近似，可能忽略高度非线性动态；实验仅验证在 Llama‑3 系列，跨模型或跨架构迁移尚未充分评估；对非文本或多模态上下文的效果未知。

---

## 516. Towards 3D-Aware Video Diffusion Models: Render-Free Human Motion Control with Mesh Tokenization

**arXiv ID:** 2606.02000 | [PDF](https://arxiv.org/pdf/2606.02000v1)

**作者:** Jingyun Liang `[一作]` (DAMO Academy, Alibaba Group), Fan Wang `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无渲染的 3D‑感知视频扩散框架，用 3D 人体网格代币直接控制视频生成。

**💡 创新点**

创新点在于：① 通过 VQ‑VAE 将 3D SMPL 网格压缩成离散代币；② 将运动拆解为轨迹与姿态并分别编码；③ 通过 per‑frame cross‑attention 将运动代币注入 DiT 结构，避免 2D 渲染导致的视角偏差。

**🔧 技术方法**

核心技术包括 SMPL 参数化模型、VQ‑VAE、DiT‑based 视觉扩散、跨注意力机制、DDIM 加速采样以及基于 Wan‑2.1 I2V 的预训练解码器。

**📊 数据集**

使用内部 30 万条视频的运动数据集（通过 GVHMR 估计网格、LLaVA 生成字幕），以及公开基准 Trajectory100 与 RealisDance‑Val，另外对 Trajectory100 进行动作编辑实验。

**📈 对比分析**

与 Wan‑2.1‑I2V、Tora、RealisDance‑DiT、RealisMotion 等方法对比，本文在轨迹/旋转误差、PSNR、SSIM、LPIPS、FID、FVD 以及编辑后视频的时序平滑度等指标均保持竞争或优于对手，尤其在视角变化和编辑误差场景中表现突出。

**⚠️ 局限性**

局限性：仅针对人类网格；对非人类或更复杂场景的 3D 物体支持有限；依赖大规模预训练模型与 GPU 资源；在极端遮挡或高度动态背景下仍可能出现细节失真。

---

## 517. Why Do Time Series Models Need Long Context Windows?

**arXiv ID:** 2606.01999 | [PDF](https://arxiv.org/pdf/2606.01999v1)

**作者:** Luca Butera `[一作]` (Università della Svizzera Italiana), Cesare Alippi `[通讯]` (Università della Svizzera Italiana)

**通讯引用:** 10396 | [OpenAlex ID](https://openalex.org/A5005003786)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析并证明全球时间序列预测模型的性能受生成过程识别 (GPI) 与条件预测 (CF) 两个任务共同决定，提出通过增大输入窗口减少 GPI 不确定性，并给出理论证明与实验验证；同时提出将 GPI 与 CF 解耦的架构，以降低计算成本并提升可扩展性。

**💡 创新点**

将全球预测任务拆分为 GPI 与 CF 两子任务，证明长窗口的主要作用是降低 GPI 的不确定性；提供理论证明窗口需超过过程记忆长度；提出并验证一种通过分离 GPI 与 CF 的模型设计（context‑embedding + 条件预测器）以实现性能与成本的 Pareto 优势。

**🔧 技术方法**

理论推导（贝叶斯后验平均、定理证明）、实验评估（GRU、PatchTST、TSMixer、ModernTCN、lru 等模型），使用背景/上下文窗口策略、embedding 缓存、可变窗口长度等技术。

**📊 数据集**

合成 NAR 过程数据；真实公开数据集包括交通（traffic）、能源（cer）、气候（climate‑t、climate‑r）；以及用于对比基础模型与领域专用模型的标准基准数据集。

**📈 对比分析**

通过 MSE/MAE 评估，比较 global vs local、foundation vs domain‑specific，以及标准模型 vs 解耦模型；实验显示 global 需要更大窗口才能逼近 local，foundation 需要更大窗口才能匹配域特定模型；解耦方案在小窗口下可达到与大窗口相当的精度，同时显著降低计算成本，Pareto 前沿表明其性能‑成本优势。

**⚠️ 局限性**

未实现完整的基于解耦的时间序列 foundation 模型；对漂移过程需要动态更新 GPI embedding；在资源受限环境下，GPI 复杂度与窗口大小的权衡仍需进一步探索。

---

## 518. Trust-Calibrated Code Review: A Participatory Design Study of Review Workflows for LLM-Generated Multi-File Changes

**arXiv ID:** 2606.01969 | [PDF](https://arxiv.org/pdf/2606.01969v1)

**作者:** Lo Gullstrand Heander `[一作]` (Lund University), Nikita Mukhortov `[通讯]` (JetBrains)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过与 JetBrains 合作的双钻石参与式设计研究，探讨并设计了针对 LLM 生成多文件代码变更的 IDE 审核流程，提出了三层级（概览、文件分析、代码片段审查）工作流及七个设计构件（块、行级风险、文件级风险、评判者、走查、缩放、沙箱），并基于高保真原型在 43 名专业开发者中进行问卷验证。

**💡 创新点**

创新点在于：①将“信任校准”定位为 LLM 代码审核的核心挑战；②提出兼顾信息层级与风险可视化的三层级工作流与七个交互构件；③将参与式设计与概念框架结合，提供可直接用于 AI 代码审核工具的设计指导。

**🔧 技术方法**

主要技术包括：双钻石参与式设计方法、以 JetBrains IDE 为基础的高保真半交互式原型（React+JavaScript），以及基于 LLM 评估结果（风险/评判）和链式思维文本的可视化展示。

**📊 数据集**

数据集主要来源于 17 名专业开发者的工作坊笔记与草图、以及 43 名参与者的问卷回复；此外原型使用了预渲染的 LLM 评估与链式思维文本。

**📈 对比分析**

方法评估采用定性访谈与定量问卷，比较指标为各工作流层级的 Likert 平均得分（3.50–3.91）以及对比现有工具的预期工作量减少（整体 63%，信任评估 52%）。未提供客观性能指标。

**⚠️ 局限性**

局限性包括：①验证基于视频演示的半交互原型，缺乏真实任务的纵向使用数据；②受访者自选性偏差，可能偏向 AI 工具积极体验者；③安全笼（sandbox）构件受欢迎度不高，功能可行性尚未在实际代码审核场景中验证。

---

## 519. Co-training with Ego-centric Video and Demonstration for Robot Navigation Task

**arXiv ID:** 2606.01951 | [PDF](https://arxiv.org/pdf/2606.01951v1)

**作者:** Shoya Kuno `[一作]` (Kyoto University), Kanata Suzuki `[通讯]` (Fujitsu Limited)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将人类第一人称行走视频通过视觉里程计转换为移动机器人可执行的动作标签的框架，并将其与机器人收集的数据联合训练 VLA 模型，实现语言驱动的导航。

**💡 创新点**

创新点在于利用视觉里程计估计人类摄像机运动并投影到机器人运动学，直接将人类视频转化为机器人动作标签，并通过人类与机器人数据的联合训练缓解域差距。

**🔧 技术方法**

采用 SuperPoint+LightGlue 视觉里程计、SE(2) 运动投影与平滑、Diffusion Transformer 以及 SigLIP2/DINOv3 视觉语言编码等技术。

**📊 数据集**

使用了人类第一人称视频（240 条）与机器人在实验室收集的 150 条演示数据，涵盖不同水果目标和初始位置。

**📈 对比分析**

通过与仅使用机器人数据、仅使用人类数据的三种训练方式对比，在果物搜索任务中结合数据后在训练位置和未见位置的成功率分别提升至 100%/94% 与 75% 等，显著提高了泛化与稳定性。

**⚠️ 局限性**

限制在于仍需机器人数据以弥补域差距，混合比例未优化，生成数据质量受人类视频稳定性影响，仅验证了 2-DoF 平地机器人，跨平台迁移仍待研究。

---

## 520. Parameter-Efficient Fine-Tuning of Large Pretrained Models for Instance Segmentation Tasks

**arXiv ID:** 2606.01947 | [PDF](https://arxiv.org/pdf/2606.01947v1)

**作者:** Nermeen Abou Baker `[一作]`, Uwe Handmann `[通讯]` (Ruhr West University of applied sciences)

**通讯引用:** 988 | [OpenAlex ID](https://openalex.org/A5078414923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了将顺序重复适配器和LoRA应用于大型预训练实例分割模型（SEEM和Mask DINO）的参数高效微调，并在四个数据集上进行评估。

**💡 创新点**

首次将LoRA应用于多尺度可变形注意力模块，并提出顺序排列适配器的可扩展微调方案，开辟了PEFT在实例分割中的新方向。

**🔧 技术方法**

采用参数高效微调技术——适配器与LoRA，结合可变形注意力的低秩更新，对模型进行训练、评估和推理速度测评。

**📊 数据集**

使用了NDD20、ZeroWaste、WIXray和Cityscapes四个实例分割数据集。

**📈 对比分析**

与全头微调、仅解码器微调、仅分类/掩码嵌入微调等传统方法比较；在大多数数据集上，2-3个适配器或LoRA可实现与全头微调相近的AP，参数量仅占1–6%；在Cityscapes中LoRA甚至超越全头。

**⚠️ 局限性**

性能仍低于SOTA，适配器导致推理延迟；LoRA在某些复杂数据集（如WIXray）效果不佳；只评估两种模型，未全面验证泛化；缺乏深入的超参数调优与混合微调探索。

---

## 521. Real-world and simulated thermal data from 960 residential multi-zone buildings in Central Europe

**arXiv ID:** 2606.01994 | [PDF](https://arxiv.org/pdf/2606.01994v1)

**作者:** Fabian Raisch `[一作]` (Technical University of Applied Sciences Rosenheim), Benjamin Tischler `[通讯]` (Technical University of Applied Sciences Rosenheim)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 ThermBuild 数据集，结合两栋实际单户住宅的15分钟测量数据与 958 套 TRNSYS 模拟建筑的三年时间序列，用于多区热动力学建模、能效控制与故障诊断。

**💡 创新点**

首次提供了包含真实测量与大规模仿真相结合的多区住宅数据，涵盖建筑属性、占用、气候、热泵配置等多维度变化，专门为迁移学习与真/模转移研究设计，提升可重复性与基准性。

**🔧 技术方法**

采用 PLC 与热泵自带日志进行数据采集，利用 k‑NN 填补缺失；TRNSYS 18 生成模拟数据；Occdem 生成占用轮廓；Meteonorm 产生多地区天气；对比真实测量与仿真、数量级检验、统计散点分析等多种验证方法。

**📊 数据集**

主要使用自研 ThermBuild 数据集（两栋真实住宅 + 958 套仿真建筑），并与 Building Data Genome Project、BuildingBench、ecobee、HOT、IDEAL 等公开数据集做对比讨论。

**📈 对比分析**

通过与真实测量的能耗与温度比较，验证了模拟误差在 6.9%–8.5% 范围内，符合 ASHRAE 10% 容差；同时对参数影响进行散点图与统计检验，展示了年龄、气候、窗格类型等对能耗与电耗的预期趋势，验证了模型的物理合理性。

**⚠️ 局限性**

局限性包括：仅有两栋真实住宅样本；模拟模型缺乏公开源代码，无法进行深度调优；部分传感器在 BSE1 建筑中缺失；数据仅覆盖中欧单户住宅，未能覆盖更广泛的建筑类型与更高时间分辨率；未验证实际控制策略的实时性能。

---

## 522. A Structured Benchmark for Text-Guided Anomaly Detection: When Language Stops Conditioning the Decision

**arXiv ID:** 2606.01992 | [PDF](https://arxiv.org/pdf/2606.01992v1)

**作者:** Stefano Samele `[一作]` (Politecnico di Milano), Matteo Matteucci `[通讯]` (Politecnico di Milano)

**通讯引用:** 7024 | [OpenAlex ID](https://openalex.org/A5003932703)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了文本引导异常检测(TGAD)基准，构造了三个层级的评测场景：1）控制文本敏感度的MVTec AD实验；2）组件级异常检测的MVTec AD扩展；3）真实工业场景的Assembled Panel Dataset；并发布了MVTec AD的组件标注版和APD数据集。

**💡 创新点**

创新点在于：①系统化地把语言的功能作用逐级放大，从文本提示敏感度到组件级控制再到结合缺陷类型与位置的完整工业检测；②提出了可衡量语言是否真正控制决策的评测指标与诊断（如Object‑Anchor Collapse、Localization‑Decision Dissociation）；③通过前向掩码(FMS)对LogSAD进行扩展，证明在无训练下也能实现一定的语言指令跟随。

**🔧 技术方法**

采用的技术主要有：多模态大模型推理（AnomalyGPT）、基于CLIP的无训练判别模型（LogSAD）及其掩码扩展、基于CLIP的嵌入自适应判别模型（AA‑CLIP），以及常用的AUROC、AUPRO等评价指标。

**📊 数据集**

使用的数据集包括：MVTec AD（原始与组件标签扩展版）和新的Assembled Panel Dataset（99正常/53异常图像，像素级注释）。

**📈 对比分析**

对三种模型在三种场景下进行跨模型比较：在文本敏感度实验中，AnomalyGPT仅在去掉对象名时出现显著下降；LogSAD对文本几乎无响应；AA‑CLIP对关键词几乎无变化。组件级实验中，EV2条件下所有模型的图像级AUROC显著下降，而AUPRO下降不大，体现Localization‑Decision Dissociation。APD实验中，所有模型图像级AUROC低于50%，但AA‑CLIP在AUPRO上相对较好。整体来看，文本提示对决策影响微弱，模型更多地把文本视为先验。

**⚠️ 局限性**

局限性在于：①语言仅充当先验而非控制信号，无法真正指导模型忽略无关缺陷；②在多组件、复杂布局的工业场景中，现有模型的定位性能和决策一致性均受限；③评测仅覆盖单一代表模型，缺乏对更广泛多模态框架的系统验证。

---

## 523. A Closer Look at In-Distribution vs. Out-of-Distribution Accuracy for Open-Set Test-time Adaptation

**arXiv ID:** 2606.01973 | [PDF](https://arxiv.org/pdf/2606.01973v1)

**作者:** Zefeng Li `[一作]` (University of British Columbia and Vector Institute), Evan Shelhamer `[通讯]` (University of British Columbia and Vector Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对开放式测试时自适应（open-set TTA）方法进行系统评估，比较其在InD与OOD准确率的平衡，并提出使用Sigmoid输出的基线。

**💡 创新点**

首次全面量化InD/OOD性能差异，揭示现有方法在未知类别拒绝上的不足，并通过将softmax改为sigmoid显著提升OOD检测。

**🔧 技术方法**

采用entropy最小化、基于阈值的过滤、Sigmoid交叉熵、Group/Layer归一化等多种TTA技术。

**📊 数据集**

在CIFAR-10-C/SVHN-C、CIFAR-10-C/CIFAR-100-C、ImageNet-C/ImageNet-O-C、Texture-C、ImageNet-R等多种污染与未知类别数据集上实验。

**📈 对比分析**

使用InD准确率、OOD准确率、混合准确率、AUROC、FPR@TPR95、OSCR等指标；结果显示现有方法在InD上表现强劲但OOD拒绝率低，Sigmoid基线在OOD指标上有显著提升但InD略有下降。

**⚠️ 局限性**

方法对OOD比例变化敏感，归一化层选择影响稳定性；Sigmoid提升OOD但牺牲InD，未进一步探索校准或更高效的过滤策略。

---

## 524. Implementation and Optimization of HQC Decoding on NPU-Integrated Devices

**arXiv ID:** 2606.01968 | [PDF](https://arxiv.org/pdf/2606.01968v1)

**作者:** Vu Minh Chau `[一作]` (Hanoi University of Science and Technology), Hoang Ta `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 112 | [OpenAlex ID](https://openalex.org/A5061545393)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在Qualcomm Hexagon NPU上对Hamming Quasi‑Cyclic (HQC) 解码过程进行向量化重写，重构 Hadamard 变换、峰值选择、RS 同调、有限域乘法和 Chien 搜索等核心核，提升解码效率。

**💡 创新点**

发现 HQC 解码天然适配 Hexagon Vector Extensions (HVX) 的 SIMD 结构，提出 HVX 友好的数据布局、保持标量等价的峰值选择、位串行 GF 乘法、向量化 RS 根搜索等技术，实现大幅度降低 Pcycles 与能耗。

**🔧 技术方法**

使用 Hexagon HVX SIMD 指令、表驱动与位串行的有限域乘法、Fast Hadamard 变换、Chien 搜索、FastRPC NPU offload、SIMD 重排与循环展开等技术。

**📊 数据集**

采用随机生成的 256 条码固定长度数据集（无公开数据集），在模拟器与真实 Snapdragon 8 Gen 2 设备上进行实验。

**📈 对比分析**

通过与 scalar CPU、scalar Hexagon 基线进行对比，使用 Hexagon Pcycles、真实设备延迟、能耗和 CPU 占用率作为指标；模拟器上实现 HQC‑128/192/256 分别获得 23×/26×/34× 加速，能耗提升 18×/12×/17×；真实设备上获得 2.07×/1.85×/1.96× 延迟加速，能耗提升 18×/12×/17×，CPU offload 达到 99%+。

**⚠️ 局限性**

实现非 constant‑time，表索引和分支取决于密钥，存在侧信道风险；FastRPC 边界开销大，仅适合批量调用；未完成完整解封装（缺少稀疏多项式乘法加速）。

---

## 525. Randomized Least Squares Value Iteration itself is Joint Differentially Private

**arXiv ID:** 2606.01952 | [PDF](https://arxiv.org/pdf/2606.01952v1)

**作者:** Haiyang Lu `[一作]` (Université d'Orléans), Mohammad Sadegh Talebi `[通讯]` (University of Copenhagen)

**通讯引用:** 455 | [OpenAlex ID](https://openalex.org/A5101839059)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在周期性有限状态马尔可夫决策过程下，证明随机化最小二乘值迭代（RLSVI）本身已满足联合差分隐私（JDP），且不需额外噪声。

**💡 创新点**

首次将RLSVI的探索噪声视为隐私保护机制，并给出其RDP到JDP的隐私上界。

**🔧 技术方法**

使用RDP分析、Gaussian机制、敏感度估计、序列化复合等技术。

**📊 数据集**

未使用真实数据，研究基于理论分析的表格MDP模型。

**📈 对比分析**

与已有的私有UCBVI、DP-UCBVI等算法对比，表明在保持相同贝叶斯/最坏情况回报下，RLSVI无隐私成本。

**⚠️ 局限性**

局限在假设用户状态不变、未处理Q值无界情形，且仅分析表格MDP，未来需扩展至通用网络及实证验证。

---

## 526. Beyond Low-Rank: Low-Rank Sparse Prompting via Spiking Neural Network and Prompt Factorization

**arXiv ID:** 2606.01945 | [PDF](https://arxiv.org/pdf/2606.01945v1)

**作者:** Yumiao Zhao `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**通讯引用:** 12319 | [OpenAlex ID](https://openalex.org/A5030720334)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出低秩稀疏视觉尖峰提示（LoRSP）框架，通过低秩分解生成多组提示因子，再利用脉冲神经网络（SNN）生成实例特定的稀疏视觉提示，以适配冻结的预训练视觉模型。

**💡 创新点**

创新点在于：①将SNN引入视觉提示学习，利用脉冲稀疏机制实现实例特定提示；②通过低秩分解构造多组提示因子，捕获语义子空间；③在提示生成中同时保持低秩与稀疏双重约束，提升适配效率与泛化。

**🔧 技术方法**

使用技术包括低秩矩阵分解、脉冲神经网络（Leaky Integrate‑and‑Fire）、像素级视觉提示、线性探测（Linear Probing）与交叉熵损失训练，优化采用SGD。

**📊 数据集**

使用数据集：Tiny‑ImageNet、CIFAR‑100、EuroSAT、OxfordPets、Food101、DTD、Flowers102、CIFAR‑10、ImageNet‑1K 及其 OOD 变体 ImageNet‑V2、ImageNet‑Sketch、ImageNet‑A、ImageNet‑R。

**📈 对比分析**

与 LP、AutoVP、EVP、LoR‑VP、VPT 等基线在 ResNet‑50、ViT‑B/32/16、Swin‑B、CLIP 等骨干上对比，LoRSP 在 Tiny‑ImageNet/CIFAR‑100 上平均提升 2.9%/1.2%；在多任务平均 90.54%（相对 LP +2.5%）；在 OOD 场景下保持 78.78% 的源域准确率，OOV 平均 38.72%，均优于其它方法。

**⚠️ 局限性**

局限性：需要额外的低秩因子与SNN参数；对 SNN 超参数（阈值、泄漏因子）敏感；在极端稀疏或噪声场景下性能可能下降；未在大规模实时推理环境中充分验证能耗优势。

---

## 527. HMPO: Hybrid Median-length Policy Optimization for Chain-of-Thought Compression

**arXiv ID:** 2606.01934 | [PDF](https://arxiv.org/pdf/2606.01934v1)

**作者:** Minghui Zheng `[一作]` (Li Auto Inc.), Kun Zhan `[通讯]` (Li Auto Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种单阶段强化学习框架 HMPO，用于压缩大型语言模型的链式思考（CoT）输出长度，同时保持准确率。

**💡 创新点**

创新点在于：①自适应中位数预算机制自动调节长度限制；②余弦衰减的 token 奖励使压缩更平滑；③乘法奖励组合抑制奖励作弊，仅奖励正确且简洁的回答。

**🔧 技术方法**

技术包括 GRPO（Group Relative Policy Optimization）、自适应预算计算、cosine‑decay token 奖励、乘法奖励分解以及基于数学数据的单阶段 RL 训练。

**📊 数据集**

训练数据主要是 DeepMath‑103K（包含 6.5K 难度高的数学题），评估数据覆盖 AIME、GPQA‑Diamond、LiveCodeBench‑V6、IFEval 等跨域基准。

**📈 对比分析**

与 AdaptThink、Thinkless、ThinkPrune 等基线比较，HMPO 在 9B~122B 的稠密与 MoE 模型上实现了 19%–46% 的 token 压缩，平均准确率下降不到 0.1pp，压缩后模型在跨域任务上与更大规模模型保持竞争或超越表现，且训练成本比多阶段方法低 1.5–2.5 倍。

**⚠️ 局限性**

局限性包括：仅在单轮推理场景验证；未针对多轮对话或工具交互等代理任务的长度需求；对多步推理的预算与奖励设计需进一步研究。

---

## 528. Scaling LLM Inference Beyond Amdahl`s Limits via Eliminating Non-Scalable Overheads

**arXiv ID:** 2606.01927 | [PDF](https://arxiv.org/pdf/2606.01927v1)

**作者:** Alan Zhao `[一作]` (scitix), Wei Xu `[通讯]` (Tsinghua University)

**通讯引用:** 16860 | [OpenAlex ID](https://openalex.org/A5013867024)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套针对单节点LLM推理的并行推理系统，通过异步调度、I/O与计算重叠以及序列并行采样来消除非可扩展的开销，显著提升了吞吐量、降低了延迟、提高了GPU利用率并降低能耗。

**💡 创新点**

创新点在于：1）提出“乐观单轮异步调度”以打破调度与后续任务的顺序依赖；2）实现“早期反馈回填”机制，让输入输出处理与采样解耦并实现并行；3）设计“序列并行采样”将采样工作按请求分布到多GPU上，使用all-to-all替代gather，进一步减少通信；4）整体系统实现为可插拔插件，兼容多种框架。

**🔧 技术方法**

核心技术包括：异步任务调度、GPU与CPU任务重叠、早期反馈回填、序列并行采样（all-to-all通信）、随机数预生成、批量填充、CUDA虚拟内存管理、KV缓存分块管理、GPU功耗监测与能耗优化。

**📊 数据集**

使用Databricks公开数据集作为用户提示，评估了八种常见LLM（Llama-2-7B/13B/70B、Qwen-2.5-7B/14B/32B/72B）。

**📈 对比分析**

与主流推理引擎vLLM（v0.11.2）和SGLang（v0.5.5）在同一硬件（H100、A100）下对比，平均提升吞吐量约1.3-1.9×，平均延迟降低22-48%，GPU利用率提升约10-40%，能耗降低约54%。在生产环境的MaaS平台上，吞吐量提升可达2×，99%/99.9%尾部延迟显著下降。

**⚠️ 局限性**

局限性：仅针对单节点部署，未考虑多节点超大模型的TP+PP混合并行；乐观调度在极端情况下可能浪费少量KV块；系统对CPU与GPU同步的假设可能在极高负载或不同硬件上产生新的瓶颈。

---

## 529. QoEReasoner: An Agentic Reasoning Framework for Automated and Explainable QoE Diagnosis in RANs

**arXiv ID:** 2606.01925 | [PDF](https://arxiv.org/pdf/2606.01925v1)

**作者:** Qizhe Li `[一作]` (Chinese University of Hong Kong), Qingjiang Shi `[通讯]` (Shenzhen Research Institute of Big Data)

**通讯引用:** 10340 | [OpenAlex ID](https://openalex.org/A5059252324)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了QoEReasoner，一个基于大型语言模型的代理式框架，用于自动化、可解释的无线接入网络QoE诊断；

**💡 创新点**

创新点在于将LLM与外部确定性工具、知识库与历史案例库相结合，形成可状态化的规划与验证循环，既实现了可靠的数值感知，又保证了因果推理的协议一致性与逻辑连贯性；

**🔧 技术方法**

使用技术包括大型语言模型（如GPT‑5.2、Qwen3-14B等）、LangGraph+ReAct式规划框架、KPI预处理与特征提取工具、基于图的因果知识库、历史案例检索与验证模块；

**📊 数据集**

使用的数据集为来自中国香港中文大学深圳分校的真实运营RAN数据集，包含300个会话（130个异常会话），覆盖6类根因及21条验证过的因果链模板；

**📈 对比分析**

与规则启发式、深度学习模型（TSTCC、CATCC）、纯LLM及现有RCA-Agent进行对比；在异常检测、因果链推理与根因分类等任务上，QoEReasoner分别提升18%–40%的准确率，诊断时间从30分钟缩短到约3分钟，且对不同LLM后端具有稳健性；

**⚠️ 局限性**

局限性包括：目前仅支持单一主导链的诊断，无法处理多链并发情况；历史案例库覆盖不足时会影响检索效果；对未来的网络演化需要不断更新工具与知识库；并未覆盖后续优化与决策建议等后诊断环节。

---

## 530. AutoMedBench: Towards Medical AutoResearch with Agentic AI Models

**arXiv ID:** 2606.01961 | [PDF](https://arxiv.org/pdf/2606.01961v1)

**作者:** Junqi Liu `[一作]` (University of California Santa Cruz), Yuyin Zhou `[通讯]` (University of California Santa Cruz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了一个面向自主医学AI研究的工作流感知基准，包含 Plan–Setup–Validate–Inference–Submit 五个阶段，并覆盖 24 个任务、5 个研究轨道（分割、图像增强、视觉问答、报告生成、病灶检测）。

**💡 创新点**

创新点在于：①将流程级评估与最终结果相结合，提供阶段分数与错误码诊断，能细粒度揭示工作流瓶颈；②公开统一的评估协议和两难度层级（Lite/Standard）以控制代理自治度；③提供完整的运行环境、代码执行接口以及与多模态医学数据的无缝对接。

**🔧 技术方法**

使用基于大型语言模型的代码执行接口、LLM 评审器进行阶段打分、自动化验证脚本、错误码归因机制，并采用加权的工作流得分与传统任务指标相结合的评估协议。

**📊 数据集**

采用多种公开医学影像与多模态数据集，涵盖 CT、MRI、X‑ray、病理图像、微镜图像、牙科图像和医学视频，来源于 KiTS19、PanTS、FeTA、AeroPath 等公开挑战。

**📈 对比分析**

对六款前沿 LLM（Opus 4.6、GLM‑5、Gemini‑3.1 Pro、GPT‑5.4、MiniMax‑M2.5、Qwen3.5‑397B）在 Lite/Standard 两层级下进行多轮交互实验。结果显示整体得分区间 51.2–66.5，验证（Validate）阶段最弱；错误多为工程失误（验证、提交错误占 76%），而任务理解错误极少；成本与性能弱相关，说明单纯提升算力并不能显著提高质量。

**⚠️ 局限性**

局限包括：仅评估推理阶段，未涉及训练/微调；对真实临床数据安全与隐私保护不足；缺少对模型可解释性、伦理风险和临床可用性的评估；基准主要基于公开数据，可能无法完全覆盖真实医疗环境的多样性。

---

## 531. FocusDiT: Masking Queries in Diffusion Transformers for Fine-grained Image Generation

**arXiv ID:** 2606.02090 | [PDF](https://arxiv.org/pdf/2606.02090v1)

**作者:** Xueji Fang `[一作]` (Zhejiang University), Guo-Jun Qi `[通讯]` (Westlake University)

**通讯引用:** 14429 | [OpenAlex ID](https://openalex.org/A5100766907)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FocusDiT，利用查询令牌掩蔽（Q-MaskGen）和词汇重分配（VR）技术提升 Diffusion Transformer 的细粒度图像生成能力。

**💡 创新点**

创新点包括：① 通过动态掩蔽挑选关键查询令牌，使其专门调用 FFN 词汇来捕捉复杂细节；② 对 FFN 词汇容量在不同层级进行重分配，使浅层/深层使用更少词汇，中间层获得更大容量；③ 根据掩蔽值跳过不重要的 FFN 计算，进一步加速推理。

**🔧 技术方法**

主要技术包括：Diffusion Transformer、FFN 作为键值词汇、AdaLN 适配时间步、MLP 结构的 Q-MaskGen、词汇重分配策略、FID/CLIPScore/GenEval/ImageReward/SCTF 等多指标评估、推理加速与参数剪枝。

**📊 数据集**

使用的数据集：Coyo-HD-11M、LAION-Aesthetics-V1-120M、LAION-Art-8M、Midjourney-Niji-1M、以及内部 500K text-image 对。

**📈 对比分析**

与 PixArt-α、SD3、OpenSoraPlan 等基准模型进行比较；在文本到图像任务中 FID、CLIPScore、SCTF 等指标均优于 PixArt-α 与 OpenSoraPlan，虽然在 SD3 上仍略有差距；在人类评测中，FocusDiT 在细节与质量方面均取得 52.6% 对 PixArt-α 与 54.5% 对 SD3 的优势。掩蔽策略还可将推理时间从 58.69s 缩短至 50.66s（≈13%）并将参数量削减至 88%。

**⚠️ 局限性**

局限性：依赖公开数据集规模有限，尚未达到 SD3 的性能；词汇重分配在训练初期可能略微降低质量；推理加速阈值需根据具体任务手动调优；模型对更大规模多样化数据的鲁棒性还有待验证。

---

## 532. Topological texture analysis of microscopy images of dynamic casein gelation and its relation to rheological properties

**arXiv ID:** 2606.02048 | [PDF](https://arxiv.org/pdf/2606.02048v1)

**作者:** Zahra Tabatabaei `[一作]` (University of Copenhagen), Jon Sporring `[通讯]` (University of Copenhagen)

**通讯引用:** 2034 | [OpenAlex ID](https://openalex.org/A5006195104)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究开发了一套基于超分辨 STED 显微镜的多尺度结构分析工具箱，用于定量追踪酸性乳蛋白凝胶化过程中的微观网络演化。

**💡 创新点**

创新点在于首次将拓扑数据分析（TDA）与差分盒计数（DBC）、多分形分区（MFP）以及局部二值模式（LBP）四种描述符结合，能够从全局拓扑连通性到局部纹理细节多角度捕捉凝胶形成与重组的动态变化，并与宏观力学属性直接关联。

**🔧 技术方法**

采用的技术包括：① TDA（基于立方体复形的 Betti-1 曲线）；② DBC（灰度结构复杂度的分形维数）；③ MFP（分区函数的泛维数谱）；④ LBP（纹理模式的直方图统计）。

**📊 数据集**

使用的数据集：① 30 张人工合成的分形图像（粗糙度 0.7–1.0）做方法验证；② NaCas 在 30 °C/40 °C 下分别用 1.8 % 与 3.5 % GDL 诱导的酸化凝胶，采集 145 帧 STED 时序（每 25 s，像素 30 nm），并配合同一体系的振荡粘弹实验得到储模 G'。

**📈 对比分析**

方法比较与性能：所有四种描述符均能在凝胶形成的关键阶段（从液相到网络定型再到重组）产生可辨别的数值转折；TDA 的 max‑Betti‑1 曲线与 G' 产生的溶胶‑凝胶转折高度一致，表明其对拓扑重排的敏感性；DBC 与 MFP 在 pH 下降过程中呈现下降‑再上升的趋势，与网络稠密化与后期重排相吻合；LBP 的熵与方差在凝胶点附近出现明显变化，进一步验证了纹理细节的演变。整体上，该工具箱能够在早期捕获微观结构变化，并与宏观力学指标保持良好相关性。

**⚠️ 局限性**

局限性包括：① 仅在 NaCas 酸化凝胶体系中验证，需进一步推广到其他蛋白质、聚合物或胶体系统；② 受限于 STED 成像的时间分辨率与光漂白问题，长时间或高速动态仍可能出现信息丢失；③ 目前方法主要关注 2D 断层图像，三维拓扑分析仍待开发；④ 参数选择（如 DBC 箱尺寸、MFP q 范围、LBP R/P）对结果有一定影响，需进一步系统化优化。

---

## 533. OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents

**arXiv ID:** 2606.02031 | [PDF](https://arxiv.org/pdf/2606.02031v1)

**作者:** Rui Yang `[一作]` (UIUC), Jianfeng Gao `[通讯]` (Microsoft)

**通讯引用:** 35603 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了开放式框架 OpenWebRL，利用在线多轮强化学习在真实网站上训练视觉 Web 代理，构建了可扩展的浏览器基础设施、监督初始化、多模态上下文管理、轨迹级判定以及高效的多轮 GRPO 优化。

**💡 创新点**

创新点包括：1) 只用 0.4K 监督示例即可获得有效的 warm‑start；2) 通过文本环境反馈和历史推理实现长时间记忆，减少视觉上下文负担；3) 采用轨迹级多轮 GRPO 并结合动态采样和 PPO 训练，显著提升在线 RL 稳定性；4) 开发 8B 版判定模型，减少对 GPT‑4 评估的依赖。

**🔧 技术方法**

核心技术：多模态大语言模型（如 Qwen3‑VL‑4B/8B）、ReAct 风格工具调用、浏览器驱动（Orchard Env）、轨迹级奖励与 GPT‑4/8B 评估、GRPO（多轮版）、动态采样、PPO 迭代、数据过滤与去重、图像与文本混合上下文管理。

**📊 数据集**

使用 WebGym 进行任务过滤、教师推理；构建 15K 监督示例（仅 412 条高质量轨迹）用于 warm‑start；2.2K 任务用于 RL 训练；Benchmarks 采用 WebVoyager、Online‑Mind2Web、DeepShop 三个开放式实时网站基准。

**📈 对比分析**

与公开与闭源基线对比，OpenWebRL‑4B 在 WebVoyager/Online‑Mind2Web/DeepShop 的成功率分别为 74.1%/67.0%/64.0%，平均 68.4%，超越现有公开系统（如 MolmoWeb‑8B、FARA‑7B 等）并在部分基准上超过 GPT‑5、Gemini‑3‑Flash 等闭源模型；8B 版本进一步提升至 68.7%。

**⚠️ 局限性**

局限性：1) 仍受浏览器交互延迟、网络波动和网站防爬限制影响，导致 50% 以上失败来自环境；2) 依赖 8B 判定模型和浏览器服务，成本仍较高；3) 目前仅适用于 4B‑8B 规模，尚未验证更大模型的可扩展性；4) 任务过滤与去重需人工阈值，可能遗漏部分复杂任务；5) 仍需改进对多域网页的通用性与安全性。

---

## 534. The Completion-Threshold Framework for Obligatory-Test Scheduling on Multiple Machines

**arXiv ID:** 2606.02029 | [PDF](https://arxiv.org/pdf/2606.02029v1)

**作者:** Kao-Chuan Liang `[一作]` (National Tsing Hua University), Ya-Chun Liang `[通讯]` (National Tsing Hua University)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5037147386)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在 m 台同质机器上进行强制性测试（obligatory‑testing）的在线调度问题，目标是最小化所有作业完成时间之和。

**💡 创新点**

提出基于完成阈值 T_X 的新分析框架，并利用该框架给出了多机下的确定性下界（三型下 1.4811，Dyadic 多型下渐进逼近 1.5），以及一个 2‑竞争的确定性算法。

**🔧 技术方法**

核心技术包括：1）将每个作业完成时间拆分为测试与处理的两段，定义 T_X 并证明 ∑C_j = ∑T_X；2）构造逆向可控的多型作业序列；3）使用递归/分段分析 Dyadic 案例；4）在算法部分采用优先级列表调度并证明 2‑竞争。

**📊 数据集**

本工作完全是理论分析，没有使用任何实验数据集；所有结果均为上界/下界证明。

**📈 对比分析**

通过与已知的下界（单机 √2 以及多机 φ 等）对比，证明 2‑竞争算法在任意测试时间下是最优的（在该优先级族内）；下界从 1.4811 逐步提升到 1.5，表明理论上无法低于 1.5。

**⚠️ 局限性**

主要限制：算法在多机下无法达到 1.5 的竞争比；下界技术目前仅适用于强制性测试，无法直接推广到可选测试；对多机可选测试的下界仍远低于实际性能；未给出单机上限 1.5 的算法，开放进一步研究。

---

## 535. Ranking vs. Assignment: The Metric Mismatch in Multi-View Object Association

**arXiv ID:** 2606.02022 | [PDF](https://arxiv.org/pdf/2606.02022v1)

**作者:** Matvei Shelukhan `[一作]` (Tevian), Karina Kvanchiani `[通讯]` (Tevian)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过理论证明和实证实验，研究了多视角物体关联任务中Pairwise指标（AP、FPR-95）与一对一匹配结果的匹配不一致，并提出基于Sinkhorn归一化的后处理压力测试来检验指标是否易被操纵。

**💡 创新点**

创新点包括：1）证明即使匹配已正确，AP/FPR-95仍可能不完美；2）证明最佳pairwise排序仍可能导致错误匹配；3）提出只学习温度和尘埃桶参数的Sinkhorn后处理压力测试，验证pairwise指标可被轻易提升却不提升实际关联质量。

**🔧 技术方法**

使用的主要技术包括：Sinkhorn归一化（带尘埃桶）、匈牙利算法进行一对一匹配、平滑pairwise ranking loss、assignment drift loss，以及基于全批梯度的后处理参数优化。

**📊 数据集**

实验数据集为公开的多视角人关联基准：WILDTRACK和MVOR。

**📈 对比分析**

通过对比原始affinity矩阵与Sinkhorn后处理后矩阵在AP、FPR-95、ACC、IPAA等指标上的变化，实验发现AP/FPR-95提升显著（如Self‑MVA AP从0.567提升到0.955，FPR‑95从0.112降到0.011），但ACC与IPAA提升有限或无提升，表明pairwise指标并不代表关联质量。

**⚠️ 局限性**

局限性包括：评估仅局部视图对，未考察全局多摄像头一致性；后处理只调整affinity矩阵，未改善模型本身；离散阈值和匈牙利算法的依赖可能导致不同场景下的效果差异。

---

## 536. PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing

**arXiv ID:** 2606.02010 | [PDF](https://arxiv.org/pdf/2606.02010v1)

**作者:** Oleksandr Nikitin `[一作]` `[通讯]`, Oleksandr Nikitin

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了PlanarBench基准，用于评估大型语言模型的空间推理能力，要求模型仅凭边列表绘制无交叉的ASCII平面图；

**💡 创新点**

创新点在于将空间布局任务与可机械验证相结合，使用边数而非节点数作为难度轴，并提供严格与放宽两种验证策略；

**🔧 技术方法**

主要技术包括基于提示的LLM生成、ASCII解析与验证流水线、Pearson相关分析以及对模型推理成本的统计；

**📊 数据集**

使用了199个由2–7个顶点构成的无同构连通平面图的边列表，来源于graph6格式的完整平面图语料；

**📈 对比分析**

对91个模型进行单次测试，最高分为159.5/199，结果表明扩展推理和模型规模是关键因素，边数是主要难度驱动器；

**⚠️ 局限性**

局限性包括仅一次尝试、ASCII绘图带来的可视化约束、图形规模限制以及对提示工程的未充分探索。

---

## 537. Jailbreaking Multimodal Large Language Models using Multi-Clip Video

**arXiv ID:** 2606.02111 | [PDF](https://arxiv.org/pdf/2606.02111v1)

**作者:** Choongwon Kang `[一作]` (Sungkyunkwan University), Jang Hyun Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 3449 | [OpenAlex ID](https://openalex.org/A5062492638)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了多剪辑视频安全基准MCV SafetyBench，评估多模态大型语言模型在视频输入下的安全漏洞

**💡 创新点**

发现视频多剪辑多样化显著提升攻击成功率，并提出基于图像过滤的防御策略

**🔧 技术方法**

使用文本到视频生成（Wan2.2‑T2V‑A14B）、文字排版嵌入、ASR评估、GPT‑4o‑mini判别、PCA可视化等技术

**📊 数据集**

构造了2,920个四剪辑视频（共1,460个恶意查询），覆盖13类OpenAI政策违规

**📈 对比分析**

对8款视频LLM进行Explicit/Implicit攻击实验，ASR随剪辑数上升；相对图像与静态视频比较显示视频更易攻击；图像过滤防御平均降低ASR约49%

**⚠️ 局限性**

仅测试最多5剪辑/10秒视频；防御方法间接，未直接解决视频特有安全弱点

---

## 538. Network Distributed Multi-Agent Reinforcement Learning for Consensus Control of Quadcopters

**arXiv ID:** 2606.02107 | [PDF](https://arxiv.org/pdf/2606.02107v1)

**作者:** Youssef Mahran `[一作]` (German University in Cairo), Ayman El-Badawy `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种基于网络分布式多智能体强化学习（ND‑MARL）的四旋翼群体一致控制框架；

**💡 创新点**

创新点在于将通信图嵌入决策过程，实现仅通过每个无人机与两邻居的局部信息实现全局一致，且所训练的策略在不重新训练的情况下即可在多达250台无人机的群体中实现零射击可扩展；

**🔧 技术方法**

采用了多智能体软演员-评论家（MASAC）算法进行高层规划，并在下层使用低级推力向量控制器实现轨迹跟踪，整体采用层次化强化学习；

**📊 数据集**

主要使用仿真数据集——gym‑pybullet‑drones 环境下的三台四旋翼无人机数据进行训练，随后在不同规模（3–250台）群体上进行零射击测试；

**📈 对比分析**

与集中式 MARL 基线相比，分布式 ND‑MARL 在三台无人机上实现了相同的时间一致性（3.7 s vs 3.6 s）和更低的终端误差（0.001 m vs 0.007 m），且在大规模群体中保持了较低的误差（N=100时误差约0.16 m，N=250时约0.64 m）且仅需常数通信成本；

**⚠️ 局限性**

局限性包括：在大规模群体下信息传播延迟导致一致时间显著增加；固定的两邻居拓扑在极大规模时会出现局部收敛但全局不完全一致的情况；并且实验仅基于仿真环境，缺乏真实无人机硬件验证。

---

## 539. Multimodal Action Diffusion for Robust End-to-End Autonomous Driving

**arXiv ID:** 2606.02105 | [PDF](https://arxiv.org/pdf/2606.02105v1)

**作者:** Jorge Daniel Rodríguez-Vidal `[一作]` (Computer Vision Center), Antonio M. López Peña `[通讯]` (Computer Vision Center)

**通讯引用:** 14404 | [OpenAlex ID](https://openalex.org/A5087248790)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种端到端的驾驶控制模型——Action Diffusion Transformer (ADT)，该模型在动作空间直接生成多种可行的油门、转向、刹车候选，并在推理时通过最近邻匹配(NNM)挑选最终执行的控制指令。

**💡 创新点**

创新点包括：① 在动作空间而非轨迹空间实现多模态预测，直接产生多样化的控制候选；② 构建无锚点的扩散 Transformer，使动作分布可被显式采样；③ 采用最近邻匹配将采样得到的候选聚合成最可靠的执行动作；④ 通过实验验证多模态预测和候选选择显著提升闭环驾驶表现。

**🔧 技术方法**

核心技术：条件扩散概率模型（DDIM）、Transformer 编码器-解码器架构、近邻匹配（NNM）、ResNet‑34 视觉特征提取、命令与速度嵌入、两摄像头观测、动作噪声预测（MSE 损失）。

**📊 数据集**

使用 Bench2Drive 基准数据集（约 1,000 条视频片段）进行训练，并在 Bench2Drive‑220 评测集上进行闭环评估；同时利用 Bench2Drive 验证集进行模型挑选。

**📈 对比分析**

与传统基于轨迹的模型（如 DriveTransformer‑Large、DriveAdapter、ETA、Hydra‑NeXt）以及纯控制的基线（TCP‑ctrl、MILE、CIL++）相比，ADT 在闭环评估中取得了 77.90 的 Driving Score、55 % 的 Success Rate、19.2 ms 的推理延迟，且在使用 NNM 的版本下闭环 DS 进一步提升至 82.88，显著优于所有对比方法。

**⚠️ 局限性**

局限性：① 在多步预测（H > 1）时表现不佳，表明多步动作执行仍是挑战；② 仅使用两摄像头，未充分利用多传感器信息；③ 视觉分辨率低（300×300），高分辨率输入可能带来更好性能；④ 采样候选数的提升至 10 个后已趋于饱和，进一步扩展可能收益有限。

---

## 540. Testing Decision Makers without Counterfactuals

**arXiv ID:** 2606.02095 | [PDF](https://arxiv.org/pdf/2606.02095v1)

**作者:** Yakov Babichenko `[一作]` `[通讯]` (Technion Israel Institute of Technology), Yakov Babichenko (Technion Israel Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一个在bandit环境下，通过观察决策者与顾问的动作与结果，来识别哪一方更具信息的框架；

**💡 创新点**

在同步决策情境下构造了一种分数测试，使其能够在不断变化的环境中成功区分更有信息的代理；

**🔧 技术方法**

运用零和博弈、完全信息博弈的分析以及分数测试的分离条件（separating）等理论工具进行证明；

**📊 数据集**

无；研究完全基于理论模型，无使用具体数据集；

**📈 对比分析**

通过严格的存在与否定证明展示测试在不同决策顺序和福利约束下的可行性与局限性，但未给出数值性能指标；

**⚠️ 局限性**

结果仅适用于分数测试，序列决策无法识别；且任何能识别更有信息代理的测试都会导致福利低于最优的一半；

---

## 541. Where Do Deep-Research Agents Go Wrong? Span-Level Error Localization in Agent Trajectories

**arXiv ID:** 2606.02060 | [PDF](https://arxiv.org/pdf/2606.02060v1)

**作者:** Jiaming Wang `[一作]` (Nanjing University), Jiaheng Liu `[通讯]` (Nanjing University)

**通讯引用:** 2374 | [OpenAlex ID](https://openalex.org/A5032858379)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究深度搜索式研究代理的过程可靠性，提出在语义跨度层面定位错误的技术与基准。

**💡 创新点**

①创建了 TELBench 1,000 条真实轨迹的跨度级错误定位基准；②提出了基于声称追踪的 DRIFT 审计框架，能在轨迹中识别支持不足或矛盾的关键声称并定位相关错误跨度。

**🔧 技术方法**

基于大语言模型（GPT‑5, Gemini‑2.5‑Pro, Claude‑Sonnet‑4.5 等）的多代理声称记录、支持检查与依赖追踪；使用 LLM 辅助的专家标注流程将轨迹转换为语义跨度。

**📊 数据集**

从 GAIA‑val、XBench、BrowseComp‑test 三大深度研究基准收集 2,790 条真实轨迹，经过语义跨度分割与人工复核后生成 TELBench 的 1,000 条验证实例。

**📈 对比分析**

与裸 LLM、Claude Code、Codex 等通用审计方法对比，DRIFT 在 TELBench 上取得最高宏 F1，提升 30% 左右；但在“首错误”定位上仍有显著差距；规模增大并未必然提升诊断效果。

**⚠️ 局限性**

限于当前框架仍难准确定义首错误位置；错误检测依赖于声称与证据的匹配，可能对极其细粒度或隐式错误无效；模型规模并非唯一瓶颈，需进一步完善支持与依赖逻辑。

---

## 542. Waiting at the front door: Continuous monitoring of latency in the host network stack

**arXiv ID:** 2606.02057 | [PDF](https://arxiv.org/pdf/2606.02057v1)

**作者:** Simon Sundberg `[一作]` (Karlstad University), Toke Høiland-Jørgensen `[通讯]` (Red Hat)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5031222214)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了一种名为 netstacklat 的工具，用于在生产环境中连续监测 Linux 主机网络栈的延迟。

**💡 创新点**

创新点在于利用内核现有的 skb 时间戳、eBPF 进行内核级聚合，并通过指数对数直方图实现低开销的统计，首次实现可持续监测。

**🔧 技术方法**

技术包括 eBPF 程序、内核 skb 时间戳、ebpf-exporter、Prometheus 直方图以及对 TCP HoL 的过滤。

**📊 数据集**

实验数据集为 144 种 Nginx/Apache 的 HTTP 工作负载（不同文件大小、连接数）以及在 Cloudflare 全球 CDN 的真实生产数据。

**📈 对比分析**

与 lattrace、kyanos、retis、pwru 等工具比较，netstacklat 的 CPU 负载平均 0.81%（最高 3.2%），对 P99 延迟的影响仅 6%，而其他工具导致 100% 以上的尾部延迟膨胀。

**⚠️ 局限性**

局限性包括仅监控入站路径、无法覆盖 NIC‑CPU 接口延迟、缺乏用户空间协议和内核旁路技术的可视化、对根因诊断功能有限以及在极高速率时仍可能产生显著开销。

---

## 543. Query-Limited Community Recovery in Stochastic Block Models

**arXiv ID:** 2606.02055 | [PDF](https://arxiv.org/pdf/2606.02055v1)

**作者:** Sabyasachi Basu `[一作]` (Microsoft Research), Suhas Thejaswi `[通讯]` (Aalto University)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5054266141)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在两社区随机块模型（SBM）中，受限且含噪声的邻域查询（oracle）下的精确社区恢复问题。

**💡 创新点**

创新点在于将数据采集本身视为推断的一部分，证明了在 oracle‑only 模型中线性规模查询是必要的，但通过两阶段自适应策略可在常数层面取得优势；在有子采样图的辅助信息时，自适应查询可实现子线性规模的显著提升，形成真正的自适应优势。

**🔧 技术方法**

主要技术包括：离线留一（leave‑one‑out）筛选框架、两阶段自适应查询策略、对 SBM 下的有效边保留概率分析以及对噪声一侧错误的处理。

**📊 数据集**

使用的不是实际数据集，而是标准的平衡两社区、对数度数的随机块模型（参数 α、β，边概率 αlog n/n 与 βlog n/n）。

**📈 对比分析**

通过与均匀非自适应查询以及已知子采样图的基准比较，结果显示：在 oracle‑only 情况下，自适应策略可将所需查询从 m n（m>1）降至 n+o(n)；在子采样图+oracle 情况下，子线性预算下自适应查询能在均匀查询失败的阈值区间实现精确恢复，而均匀查询无法突破。

**⚠️ 局限性**

局限性包括：只给出了渐进意义下的结果，未给出完整自适应相位图；适用范围局限于平衡两社区、对数度数 SBM；实际部署时需考虑隐私与伦理问题。

---

## 544. Attention mechanisms and transfer learning for robust peach leaf damage classification under domain shift

**arXiv ID:** 2606.02045 | [PDF](https://arxiv.org/pdf/2606.02045v1)

**作者:** Adrián Cánovas-Rodriguez `[一作]` (University of Murcia), Antonio F. Skarmeta `[通讯]` (University of Murcia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建桃叶损伤六类数据集并评估深度学习模型

**💡 创新点**

将CBAM注意力机制融入CNN并在公共与本地域上实现迁移学习

**🔧 技术方法**

使用EfficientNet、DenseNet、Inception、MobileNet等CNN与CBAM

**📊 数据集**

利用公开的PlantDoc与Mendeley数据集以及180张本地桃叶图像

**📈 对比分析**

通过交叉验证与宏F1/准确率对比，EfficientNetB5+CBAM达到93.3%准确率、93.6%宏F1，迁移后提升至94.6%准确率

**⚠️ 局限性**

局限在样本不均衡、迁移策略对不同网络的敏感性以及未评估低功耗部署

---

## 545. Normality-Preserving Continual Industrial Anomaly Detection via Orthogonal LoRA Banks

**arXiv ID:** 2606.02042 | [PDF](https://arxiv.org/pdf/2606.02042v1)

**作者:** Weibai Fang `[一作]` (Yanshan University), Qiancheng Lao `[通讯]` (Yanshan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在冻结的Stable Diffusion U‑Net上使用LoRA进行的持续工业异常检测框架，旨在在不断引入新类别时保持历史类别的正常性先验。

**💡 创新点**

创新点在于引入History‑Frozen Orthogonal LoRA Bank（HF‑OLB）冻结旧任务LoRA并将新任务的LoRA投影到历史子空间正交补空间，以防止正常性先验漂移；以及Hierarchical Novelty‑Adaptive Bank Growth（HNABG）根据层级和子空间新颖度自适应扩展LoRA容量，从而实现高效且稳健的正常性记忆。

**🔧 技术方法**

技术上结合了LoRA低秩适配、正交投影约束、层级秩分配、子空间新颖度评估与动态LoRA增益，以及冻结的Stable Diffusion潜在扩散模型做为重建基底。

**📊 数据集**

实验使用工业缺陷数据集MVTec和VisA，在多种持续学习设置（如7‑1×8、2‑6）下进行验证。

**📈 对比分析**

与ReplayCAD、CDAD、Seq‑SD‑LoRA、Seq‑SD‑CLLoRA等方法相比，该方法在A‑AUROC上提升约2–3个百分点，遗忘度FM显著降低，且显著提升像素级异常定位稳定性。

**⚠️ 局限性**

局限性包括仍依赖于预训练Stable Diffusion的可用性，对极端域漂移和在线实时流的适配尚未充分验证，并且在极长任务序列下对资源开销的进一步优化仍有空间。

---

## 546. World-Task Factorization for Robot Learning

**arXiv ID:** 2606.02027 | [PDF](https://arxiv.org/pdf/2606.02027v1)

**作者:** Eduardo Sebastián `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**通讯引用:** 2692 | [OpenAlex ID](https://openalex.org/A5066624177)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

研究机器人政策的世界/任务因子化，并实现了基于AICON图结构的世界模型与低维梯度调制器的混合框架；

**💡 创新点**

提出将学习限定在梯度路径接口上的原则性因子化，使政策既能保持物理结构，又能高效学习任务约束；

**🔧 技术方法**

采用AICON（可微分递归估计器与主动互连图）实现世界模型，配合小型神经网络调制器学习梯度权重与Riemannian预条件矩阵；

**📊 数据集**

在三种模拟任务（搜索、双臂交接、压力板挑战）中训练，随后在真实硬件上部署；

**📈 对比分析**

与端到端学习和纯解析基线比较，显示在所有任务中均取得更好或相当的性能，样本效率更高，零样本泛化和硬件迁移均表现优异；

**⚠️ 局限性**

对世界与任务因子相互独立假设的依赖、对完整解析世界模型的要求、以及在缺少显式物理模型或专家演示的情境下的适用性仍有待验证。

---

## 547. PerBite: A Curated Diagnostic Workflow for Bite-Aware Food Volume Estimation

**arXiv ID:** 2606.02021 | [PDF](https://arxiv.org/pdf/2606.02021v1)

**作者:** Ahmad AlMughrabi `[一作]` (University of Barcelona), Petia Radeva `[通讯]` (University of Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种针对连续食物摄入场景的配对状态（吃前/吃后）三维重建与体积估计工作流程；

**💡 创新点**

将提示式单图像重建、基于餐盘直径的度量标定、结构清理以及差分体积诊断整合为一套可诊断的“bite-aware”工作流，并强调表面重建质量与体积估计之间的差异；

**🔧 技术方法**

使用 SAM 3 进行食物与餐盘分割，Hunyuan3D/SAM 3D 生成无量纲网格，Blender 进行餐盘去除、孔洞填补与网格闭合，基于餐盘直径或 MoGe-2/PCA 进行均匀尺度标定，最后计算水密网格体积；

**📊 数据集**

在 MetaFood CVPR 2026 Continuous 3D Reconstruction While Eating Challenge 的 17 对吃前/吃后图像上进行评估；

**📈 对比分析**

在官方挑战中以无尺度 ICP 对齐的 Chamfer 距离 8.31 取得第一名；在自定义指标中，单态体积 MAPE 33.87%，消耗体积 MAPE 53.74%，并且所有 17 对都满足单调递减一致性；

**⚠️ 局限性**

无法自动选择吃前/吃后帧，依赖人工挑选；对 Blender 处理的依赖使系统不完全自动；尺度标定高度依赖餐盘可见度，软性或复杂形状食物易导致大误差；

---

## 548. Extreme Low-Bit Inference in Reasoning Models: Failure Modes and Targeted Recovery

**arXiv ID:** 2606.02011 | [PDF](https://arxiv.org/pdf/2606.02011v1)

**作者:** Ekaterina Alimaskina `[一作]` (BRAIn Lab), Aleksandr Beznosikov `[通讯]` (BRAIn Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究极低位数（2‑bit）量化对大型推理模型（LRM）的影响，并通过完整推理轨迹分析发现两种关键失败模式（路径搜索失败与承诺失败）。

**💡 创新点**

提出基于轨迹指标的四级“退化模式”分类，利用该分类指导两种针对性干预（FP16规划 + “loop‑rescue”）来显著恢复或提升量化模型性能，同时保持大部分低位推理速度优势。

**🔧 技术方法**

核心技术包括：GPTQ 仅权重量化（W2A16/W4A16），循环检测与即时恢复，FP16 规划提纲生成，定量评估推理长度、循环率、预算耗尽等轨迹指标，批量推理下的吞吐量测量。

**📊 数据集**

使用多种数学与常识推理基准：AIME 2026、GPQA‑Diamond、MATH‑500、GSM8K、StrategyQA、WinoGrande、ARC‑Easy / ARC‑Challenge、PIQA。

**📈 对比分析**

与全精度 FP16 基线相比，直接 2‑bit 量化准确率下降 10–30%；加入规划（+P）可恢复 3–10% 甚至更多；加入循环恢复（+L）可恢复 10–30% 并显著缩短生成长度；两者合并（+P+L）在“过程退化”或“崩溃”模式下可将准确率提升 20–60% 并在批量1时实现 2–5× 的速度提升，批量8 时速度提升降低但仍具竞争力。

**⚠️ 局限性**

局限性包括：仅评估 GPTQ W2A16（仅权重量化）且未使用更先进的量化或激活量化技术；实验仅在 Qwen3‑8B / Qwen3‑32B 上进行，未覆盖更大规模模型；缺乏对不同硬件（尤其是 FP4 支持）下的全面验证；仅关注单一温度（t=0.6）和固定推理预算，未探索自适应预算或温度控制。

---

## 549. DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding

**arXiv ID:** 2606.02091 | [PDF](https://arxiv.org/pdf/2606.02091v1)

**作者:** Jiebin Zhang `[一作]` (Peking University), Sujian Li `[通讯]` (Peking University)

**通讯引用:** 8285 | [OpenAlex ID](https://openalex.org/A5058353424)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种改进的块扩散推测解码方法（名为"?"），通过让每个草稿层学习其自身的目标模型层的加权融合，显著提升了草稿模型的表达能力并实现了更深层次的可扩展性。

**💡 创新点**

核心创新点包括：1）轻量化层级融合机制，使每个草稿层可获得独立的目标层组合；2）异质键值投影解耦草稿与目标信息，增强注意力表达；3）渐进式位置加权损失，优化早期和后期位置的学习平衡；4）训练数据规模从80万样本提升至240万，充分利用扩展模型容量。

**🔧 技术方法**

技术主要包括：块扩散（block diffusion）文本生成框架、草稿-验证的推测解码、层级融合（softmax 加权加权和）、异质 KV 投影、RMSNorm、逐层学习率和位置加权交叉熵损失。

**📊 数据集**

使用了约240万样本的混合数据集，来源于 NVIDIA Nemotron Post-Training Dataset V2、CodeAlpaca 与 Step-3.5-Flash-SFT。

**📈 对比分析**

与最新方法 DFlash 对比，在 Qwen3‑4B、Qwen3‑8B 和 GPT‑OSS‑20B 三个目标模型上，平均 wall‑clock 加速比分别提升至 5.52×、5.46× 和 3.91×，比 DFlash 提升约 11%、8% 与 5%，同时在数学推理、代码生成与对话等六个基准任务上均表现出更高的接受长度与更快的推理速度。

**⚠️ 局限性**

主要局限包括：1）模型训练成本高，需大规模计算资源；2）尽管训练数据已扩展至240万样本，但仍有进一步扩大训练集的潜力未被探索。

---

## 550. Planar Symmetric Pattern Generation

**arXiv ID:** 2606.02073 | [PDF](https://arxiv.org/pdf/2606.02073v1)

**作者:** Ning Lin `[一作]` (Renmin University of China), Hao Sun `[通讯]` (Renmin University of China)

**通讯引用:** 12265 | [OpenAlex ID](https://openalex.org/A5030575469)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种通用的平面对称性连续表示与生成框架，将任意二维连续表示转化为严格满足任意平面群对称且保持连续的表示，并通过该表示实现可控的图形与材料设计。

**💡 创新点**

核心创新在于：①将任意平面群嵌入仿射反射群并利用Hironaka分解构造高对称系数与低对称基底的连续表达；②在此表达上直接施加对称约束，解耦其他目标，使得无需对称数据即可实现零样本可控生成；③将扩散模型的Score Distillation Sampling与自定义物理损失相结合，完成视觉与物理双重约束。

**🔧 技术方法**

使用的技术包括：仿射反射群嵌入、Hironaka分解与基底构造、连续G‑不变函数构造、Score Distillation Sampling (SDS)、LoRA微调、虚拟温度方法 (VTM)、同质化优化、以及基于U‑Net的扩散模型。

**📊 数据集**

实验数据集主要为：Stable Diffusion 2.1/SDXL 1.0 的通用图像库，纸张切割数据集（用于LoRA微调），以及 36,000 个基于同质化优化得到的二值单元格样本（用于材料设计）。

**📈 对比分析**

通过与直接生成、条件生成、后置对称化以及投影式对称化等基线对比，评估了 CLIP‑A 美学分数、对称误差（MSE）、机械性能（体积、体积误差、有效体积模量）等指标。结果显示：在保持或提升美学分数的同时，严格对称化方法显著降低对称误差；在材料设计任务中，实现了与基线相当甚至更优的体积模量，同时满足对称与体积分数约束。

**⚠️ 局限性**

局限性包括：①仅在二维（以及三维）可导的仿射反射群可使用，对更高维或非仿射群的适用性有限；②基底与系数的构造需要符号计算与复杂矩阵操作，导致计算成本相对较高；③在极端高对称群或细节复杂结构下，SDS 可能收敛缓慢；④对实际制造的兼容性需进一步验证。

---

## 551. BADGER: Bridging Agentic and Deterministic Evaluation for Generative Enterprise Reasoning

**arXiv ID:** 2606.02109 | [PDF](https://arxiv.org/pdf/2606.02109v1)

**作者:** Shannon Serrao `[一作]` (Merkle Analytics), Nathan Miller `[通讯]` (Merkle Analytics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一的企业AI评估框架，结合文本到SQL生成与多步骤代理推理的评估。

**💡 创新点**

创新点包括：LLM辅助SQL组件提取、基于LLM的结构对齐后确定性的Hybrid-EX执行准确度指标，以及将RAGAS、G-Eval等现有指标整合成可部署的生产级评估套件。

**🔧 技术方法**

采用的技术主要是大型语言模型（LLM）作为结构提取器和判断器、基于表格对齐的两阶段Hybrid-EX算法、以及对SQL的组件级匹配与人工标注的对齐。

**📊 数据集**

使用的数据集为从金融服务和零售两家企业内部生产系统抽取的150条人工标注查询，包含Gold SQL与结果表。

**📈 对比分析**

通过与六个现有框架（如BIRD、RAGAS、Defog等）在同一测试集上计算Cohen's κ，Hybrid-EX得到κ=0.717（95%CI 0.600–0.822），显著优于所有基线，且敏感性-特异性差距仅11.6个百分点。

**⚠️ 局限性**

局限性包括样本规模受治理限制仅为150条；仅验证了金融与零售两行业；无法覆盖多模态检索生成；以及对实时数据库执行的适配需要额外工作。

---

## 552. The Role of Ambiguity in Error Prediction via Uncertainty Quantification

**arXiv ID:** 2606.02093 | [PDF](https://arxiv.org/pdf/2606.02093v1)

**作者:** Ieva Raminta Staliūnaitė `[一作]` (University of Cambridge), Andreas Vlachos `[通讯]` (University of Cambridge)

**通讯引用:** 5043 | [OpenAlex ID](https://openalex.org/A5067943980)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型的错误预测任务，提出通过辨别输入的歧义性（aleatoric uncertainty）与模型自身的不确定性（epistemic uncertainty）来改进错误预测。

**💡 创新点**

创新点在于：①引入歧义性预测器并将其作为 gating/penalty 信号；②通过 gated experts 与 selective prediction 两种架构，将歧义信息与传统 UQ 量化分离；③证明即使使用预测的歧义概率也可匹配或超越使用真实歧义标签的基线。

**🔧 技术方法**

技术：使用多种 UQ 指标（MSP、Semantic Entropy、MI、SAR、Semantic Energy、CoCoA）作为特征；构建 MLP + 软排序损失的错误预测头；设计基于歧义概率的门控专家网络和基于歧义惩罚的拒绝策略；利用内部隐藏层特征与语义聚类特征进行歧义分类。

**📊 数据集**

数据集：AmbigQA（开放域含歧义问答）、TriviaQA（大规模阅读理解，约16% 具歧义）以及 NCQA（二元事实判断，具对立证据）。

**📈 对比分析**

比较方法：对比基准 UQ‑only 误差预测模型与加入歧义信息的 gated/selective 模型；评价指标为 AUROC 与 PRR。实验显示，在所有模型、UQ 量化方法与数据集上加入歧义信号均提升误差预测性能；在 TriviaQA 上可达 0.904 的 AUROC 与 0.879 的 PRR，且在 AmbigQA 与 NCQA 上也呈显著提升。

**⚠️ 局限性**

局限性：仅针对文本、英语数据；需要事先标注的歧义标签；未考虑检索增强场景；只评估了语义性 UQ 指标，未探究其他类型不确定性；超参数调优基于同一验证集，可能导致略高估。

---

## 553. Agentic-J: An AI Agent for Biological Microscopy Image Analysis

**arXiv ID:** 2606.02080 | [PDF](https://arxiv.org/pdf/2606.02080v1)

**作者:** Lukas Johanns `[一作]` (Leibniz-Institut für Analytische Wissenschaften – ISAS – e.V.), Jianxu Chen `[通讯]` (Leibniz-Institut für Analytische Wissenschaften – ISAS – e.V.)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aaccfe5c-6b26-4208-b23c-35331481e142` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个容器化的多代理系统Agentic-J，能够通过自然语言指令在Fiji中自动生成、执行并记录可复现的生物显微镜图像分析工作流程。

**💡 创新点**

集成了沙箱化执行、插件依赖管理、领域知识图谱与RAG检索，形成一个端到端可复现、可审核且对非程序员友好的图像分析平台。

**🔧 技术方法**

使用Docker容器化、LangChain/Agentic框架、Qdrant知识库、插件技能文件、Groovy/Python脚本生成与调试，以及OpenAI/OpenRouter LLM。

**📊 数据集**

在两项真实科研案例中验证：1）小鼠肌干细胞在水凝胶微孔中的迁移跟踪（TrackMate），2）Leishmania感染宿主细胞的凋亡与原生体增殖指数。

**📈 对比分析**

通过与原文报告的结果对比（如轨迹数、平均速度分布、增殖指数与凋亡关系的统计显著性），Agentic-J能重现与专家相近的数值，质量保证代理进一步标注出差异并给出改进建议。

**⚠️ 局限性**

仍需人工参与调参、处理异常情况；代理输出存在随机性，某些插件版本或参数可能不匹配；系统受限于预装插件集，无法即时更新最新工具；对极端或多模态数据支持有限。

---

## 554. Beyond $\ell_2$-norm and $\ell_\infty$-norm: A Curvature-Inspired $\ell_p$-Norm Scheme for Deep Neural Networks

**arXiv ID:** 2606.02078 | [PDF](https://arxiv.org/pdf/2606.02078v1)

**作者:** Jianhao Xu `[一作]` (Soochow University), Zhuang Yang `[通讯]` (Soochow University)

**通讯引用:** 49381 | [OpenAlex ID](https://openalex.org/A5100401978)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种动态调节ℓ_p范数的Cosine ℓ_p-norm调度方案，并将其嵌入SGD和SGD‑Momentum，得到LPSGD和LPSGDM优化器。

**💡 创新点**

创新点在于：①识别并跟踪训练过程中曲率各向异性的演化；②设计从大p（>2）逐步衰减到2的Cosine调度，自动匹配不同阶段的几何特征；③在保持与标准SGD相近的计算复杂度下，提供理论O(T⁻¹/2)非凸收敛保证。

**🔧 技术方法**

主要技术包括：ℓ_p范数梯度归一化、Cosine调度策略、动量（SGDM）与可分离权重衰减、随机Langevin分析、Hessian谱估计。

**📊 数据集**

实验数据集：CIFAR‑10、CIFAR‑100、ImageNet‑1K，网络模型：VGG‑11、ResNet‑18、ResNet‑50。

**📈 对比分析**

与SGD‑Momentum、AdamW、Lion等基线比较，LPSGDM在所有模型和数据集上均取得最高或相近的Top‑1精度，并在中后期阶段表现出更快的收敛与更好的泛化。

**⚠️ 局限性**

局限性包括：对p_max的选择仍需经验调优；理论分析未覆盖权重衰减的完整影响；在更大规模或不同任务（如自监督、强化学习）中的适用性尚待验证。

---

## 555. TIDES: Time-Derivative Event Simulation via Deformable Reconstruction

**arXiv ID:** 2606.02058 | [PDF](https://arxiv.org/pdf/2606.02058v1)

**作者:** Christopher Thirgood `[一作]` (University of Surrey), Simon Hadfield `[通讯]` (University of Surrey)

**通讯引用:** 4973 | [OpenAlex ID](https://openalex.org/A5091184063)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于动态高斯溅射（4D Gaussian Splatting）的连续时间事件合成器TIDES，直接在3D场景的可见性一致渲染框架中计算像素对数亮度及其时间导数，从而实现无时间分辨率限制的事件阈值跨越预测。

**💡 创新点**

创新点包括：
1) 前向模式时间导数渲染，获取与可见性一致的亮度斜率，避免传统差分导致的时间批处理；
2) 风险引导的自适应时间步长，集中计算在遮挡动态区域；
3) 基于渲染器的有限带宽事件读出仲裁模型，模拟真实传感器的时延、抖动与事件丢失；
4) 结合可见性诊断（α̇、贡献计数、覆盖量）实现多跨事件预测与混合可见性屏蔽。

**🔧 技术方法**

核心技术包括动态高斯溅射（4DGS）、前向模式自动微分求时间导数、基于渲染器的光度与透明度正则化、可见性一致的阈值比较器、以及基于活动度的事件读出仲裁器。

**📊 数据集**

使用四个公开 RGB‑事件数据集：EDS、DSEC（行车视觉）、HS‑ERGB、BS‑ERGB（动态对象）。每个序列都从 RGB 视频中重建4DGS场景，然后在同一渲染器下生成事件。

**📈 对比分析**

与 ESIM、V2E、ICNS/IEBCS、DVS‑Voltmeter 等基准方法在三类指标上比较：
- IG‑NLL 与 Chamfer（事件时序与空间一致性）
- Same‑ts、Fano、ISI 频率（批处理/突发误差）
- 下游任务转移（重建、帧插值、深度估计、语义分割）
TIDES 在所有四个数据集上均取得最优 IG‑NLL、Chamfer 与最低批处理率；在下游任务上，其训练的模型在真实数据上性能最高。

**⚠️ 局限性**

局限性主要在于：
- 事件质量受底层 4DGS 场景重建精度限制；
- 对极高帧率动态场景的实时渲染仍有计算开销；
- 某些复杂光照或材质效果（如反射、散射）在当前高斯溅射模型中未充分建模；
- 传感器模型主要聚焦于时间分辨率与读出，其他噪声源（如温度漂移）未显式考虑。

---

## 556. Realistic noise synthesis reduces bias and improves tissue microstructure estimation with supervised machine learning

**arXiv ID:** 2606.02044 | [PDF](https://arxiv.org/pdf/2606.02044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 557. eMoT: evolving Memory-of-Thought via Symbolic Anchoring and Memory Corrosion

**arXiv ID:** 2606.02054 | [PDF](https://arxiv.org/pdf/2606.02054v1)

**作者:** Xiang Li `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112742 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 eMoT 框架，将多步推理建模为可演化的记忆库，并通过符号执行和一致性修正来提升稳定性与准确率。

**💡 创新点**

创新点包括：1) 记忆腐蚀机制（高效模式强化、低效模式衰退）；2) 将 Python 代码执行作为符号锚定，替代传统的后验验证；3) 双流输出（神经推理 + 符号结果）的一致性驱动修正。

**🔧 技术方法**

技术手段：检索增强生成（RAG）+ 记忆检索与激活权重；Python 代码生成与执行；神经推理与符号计算的双流交互；一致性修正模块。

**📊 数据集**

使用数据集：GSM8K、ASDiv、SVAMP、MGSM、GSM‑Hard、Game of 24、WordSorting、Checkmate。

**📈 对比分析**

与同一 Qwen‑32B 基座下的 CoT、ToT、BoT、PaL 等基线比较，eMoT 在所有算数和组合任务上均实现显著提升：GSM8K 93.4%↑+53.9%，GSM‑Hard 71.5%↑+55.6%，Game of 24 100%↑+15.6%，WordSorting 96.8%↑+12.4%，Checkmate 90%↑+14%。

**⚠️ 局限性**

局限性：1) 额外的记忆检索、代码生成和一致性修正导致 token 消耗增加；2) 目前采用单次代码生成，缺乏迭代调试；3) 记忆管理仍需手动调参，可能对跨领域推理泛化有限；4) 对极大模型的依赖仍不充分。

---

## 558. Private Learning in Bilateral Trade

**arXiv ID:** 2606.02050 | [PDF](https://arxiv.org/pdf/2606.02050v1)

**作者:** Simone Di Gregorio `[一作]` (Sapienza University of Rome), Chris Schwiegelshohn `[通讯]` (Aarhus University)

**通讯引用:** 702 | [OpenAlex ID](https://openalex.org/A5080748807)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究在双边贸易（卖方与买方）场景中，以差分隐私约束下学习接近最优的机制，既可最大化利润，也可最大化交易带来的社会福利（gain‑from‑trade）。

**💡 创新点**

创新点主要包括：
- 在一般分布下证明差分隐私学习不可行；
- 证明当交易价值分布满足σ‑平滑假设时，能用近似最优且差分私有的算法实现，并给出几乎最优的样本复杂度；
- 设计了“η‑simple”机制族与“固定价格网格”机制族，既能逼近最优，又可实现高效的指数机制采样；
- 利用概率链技术和对数级别的稠密网格得到快速的统一收敛上界。

**🔧 技术方法**

核心技术包括：
- 采用PAC‑学习框架与指数机制（exponential mechanism）结合，利用敏感度2/n；
- 构造离散化机制族（η‑simple/固定价格网格），并证明其逼近误差与平滑度σ相关；
- 用概率链（probabilistic chaining）和Gaussian/ Rademacher 复杂度证明快速统一收敛；
- 通过图搜索与动态规划实现指数机制在指数规模机制族上的高效采样。

**📊 数据集**

论文主要为理论性工作，未使用具体数据集；所有结果均基于理论分析与假设（如σ‑平滑分布）。

**📈 对比分析**

相较于传统非私有学习算法，论文提供了差分隐私约束下的样本复杂度上界：
- 利润最大化：样本复杂度 Θ̃(1/(σ α²))，与非私有下的最优 1/α² 仅相差 1/σ 量级；
- 交易增益最大化：样本复杂度 Θ̃(1/α² + 1/α)，几乎达到非私有下的极限；
- 算法在多项式时间内可实现；

**⚠️ 局限性**

局限性包括：
- 对一般分布不可行，需σ‑平滑假设；
- 需要知道或估计σ，若估计不准会导致样本复杂度增加；
- DP 预算 ε 的影响仅在样本量中通过对数项体现，实际 ε 的选择仍需经验；
- 证明基于离散化机制族，实际实现可能在高维/复杂分布上仍面临实用性挑战。

---

## 559. Explainable Data-driven Deep Reinforcement Learning Methods for Optimal Energy Management in Buildings

**arXiv ID:** 2606.02049 | [PDF](https://arxiv.org/pdf/2606.02049v1)

**作者:** Hallah Shahid Butt `[一作]` (Karlsruhe Institute of Technology), Benjamin Schäfer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 9478 | [OpenAlex ID](https://openalex.org/A5005576823)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证了一套可解释的深度强化学习（XRL）框架，用于住宅建筑的能源管理。

**💡 创新点**

创新点在于将可解释技术（决策树近似、特征剔除）与DRL控制流程深度耦合，并在同一框架中对多种算法进行统一基准评估。

**🔧 技术方法**

采用了多种深度RL算法（A2C、PPO、DQN、SAC、TD3、DDPG）、LSTM预测、Optuna超参搜索、决策树解释、特征剔除等技术。

**📊 数据集**

使用了两类数据集：来自KIT Living Lab Energy Campus（LLEC）的真实建筑能耗和光伏、天气、电价等测量数据，以及基于EnergyPlus模拟的全一年量化合成数据。

**📈 对比分析**

通过在相同状态空间、奖励函数和环境条件下对六种算法进行训练、测试，并使用累计奖励、成本收益、动作一致性等指标比较，结果表明A2C和PPO在两组数据上均优于其他算法，且解释结果可提供透明决策依据。

**⚠️ 局限性**

主要局限包括：对单一建筑的实验，缺乏跨建筑泛化评估；解释方法仍为后置近似，未在实时部署中验证其安全性；对网格级目标的协同和激励机制尚未实现。

---

## 560. SentGuard: Sentence-Level Streaming Guardrails for Large Language Models

**arXiv ID:** 2606.02041 | [PDF](https://arxiv.org/pdf/2606.02041v1)

**作者:** Jiaqi Yu `[一作]` (Fudan University), Yingchun Wang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种句子级的流式安全守卫SentGuard，利用轻量级等待缓冲区在生成过程中并行检查句子前缀并只发布已验证的句子。

**💡 创新点**

创新点在于将安全监测粒度定位到句子层面，既比基于单词的实时监测更具语义完整性，又比完整响应后审查更及时；并通过两阶段粗细匹配的监督训练和专门构建的StreamSafe数据集实现早期检测。

**🔧 技术方法**

技术包括：基于LLM的句子级安全分类模型，粗细匹配（coarse‑to‑fine）监督学习，等待缓冲框架实现非阻塞并行推理，阈值决策机制。

**📊 数据集**

使用自建的StreamSafe数据集（62K对话，按句子划分并标注8类危害），以及公开的BeaverTails、Safe‑RLHF、XSTest、WildGuardTest等安全基准。

**📈 对比分析**

与现有响应级和词级守卫模型比较，SentGuard在五大基准上在两句内检测率平均达90.5%，MFDS为1.72句，Streaming False‑Positive Rate仅7.41%；在全响应评估中F1平均88.7%，保持或超越其他模型。

**⚠️ 局限性**

局限性包括仅针对文本生成，未扩展到音频、图像或视频等多模态输出；在多模态场景中句子边界不明显，需新的分块与监督策略。

---

## 561. Federated Formal Verification: Cross-Backend Citation, Cross-Axis Convergence, and AI-Orchestrated Proof Dispatch for Production Systems

**arXiv ID:** 2606.02019 | [PDF](https://arxiv.org/pdf/2606.02019v1)

**作者:** Pierre Falda `[一作]` `[通讯]` (Bullish), Pierre Falda (Bullish)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种面向生产系统的联邦式形式化验证架构，将验证任务拆分为三种互补机制：跨后端引用、跨轴收敛和AI驱动的并行调度；在Mercury高频交易平台的Raft共识子系统和金融算术不变子系统上验证，成功将26条Raft公理在17小时内全部闭合，并发现四个实际生产缺陷。

**💡 创新点**

创新点在于（1）跨后端引用将TLA+义务通过引用等价定理分发至结构不同的证明系统，实现后端多样性与验证结果一致性；（2）跨轴收敛矩阵记录各义务在多种验证范式下的判决并以双重/三重核门控实现鲁棒性；（3）AI协同调度在不扩展可信计算基的前提下，通过诚实负面反馈在多核、跨语言后端间并行探索，大幅压缩总墙钟。

**🔧 技术方法**

使用的技术包括TLA+ / TLAPS、Coq、Lean 4、Why3、Apalache、PRISM、Z3、CBMC等多种证明与模型检查器；Gradle构建系统的CI门控实现跨后端闭合检测；AI代理（LLM+自定义脚本）用于自动发现可用后端并触发并行调度；SpecBridge代码生成器将TLA+规范映射到Java运行时断言。

**📊 数据集**

数据集主要来自Mercury高频交易平台的生产代码：1）Raft共识实现，包含完整算法范围（共识、领导迁移、日志压缩、线性可读、动态重配置）；2）匹配引擎金融算术不变层，包含五个数值规范（余额、自动做市曲线、隔离保证金、锁定追踪、清算）。

**📈 对比分析**

通过三类基准对比验证方法：①自身内部单专业顺序估算（≈50-60×）；②相同形状的Path‑A.2幽灵补丁基线（≈60×）；③独立前沿IronFleet（≈635×）。性能上每条公理平均耗时仅34 s（Coq），相较于30 min的传统补丁方法下降约60倍；整体验证耗时17小时，远低于5‑7个月的手工估算。

**⚠️ 局限性**

局限性包括：跨后端引用所依赖的对应证书仍为人工声明，尚未机械化证明；对每条公理的端到端迁移仍需人工编写端口；可信计算基相对较大（需保证多后端闭合和SpecBridge运行时一致性）；AI调度策略依赖规则手工设定，需进一步自动化；跨后端引用和收敛机制对TLA+规范的依赖限制了可迁移性。

---

## 562. TAPAAL SMC: Statistical Model Checking of Stochastic Timed-Arc Petri Nets

**arXiv ID:** 2606.02007 | [PDF](https://arxiv.org/pdf/2606.02007v1)

**作者:** Tanguy Dubois `[一作]` (Nantes Université), Jiri Srba `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种带有随机等待时间的弱语义时序化-弧 Petri 网模型，并在此基础上实现了统计模型检验（SMC）引擎，能够对含有随机延迟、优先级、权重等特性的系统进行概率性时间行为分析；

**💡 创新点**

创新点在于首次为弱语义时序化-弧 Petri 网引入随机时间属性、权重优先级以及动态触发模式，并通过随机 walk 抽样与多模态采样方法实现高效的 SMC 采样；

**🔧 技术方法**

核心技术包括随机 walk 采样、弱语义时序化-弧 Petri 网的语义定义、统计模型检验算法（含量化公式与 SPRT 等），以及 TAPAAL SMC 引擎的多线程并行实现；

**📊 数据集**

本文没有使用公开数据集，而是通过若干工业和科研案例（如森林火灾、木窗制造、量子通信协议等）构建了对应的 Petri 网模型进行实验；

**📈 对比分析**

与传统的离散事件仿真、标准模型检验工具以及手工分析方法相比，SMC 引擎在保持同等精度（如95% 置信度、±0.01 精度）下，能够在几分钟到数小时内完成对复杂系统的概率评估，显著提升了分析效率；

**⚠️ 局限性**

局限性包括：仅支持弱语义时序化-弧 Petri 网，无法处理强语义或同步时间约束；模型规模增长导致采样次数急剧增加；以及对高维概率分布的抽样仍然依赖经验参数，缺乏全局最优保证。

---

## 563. Distortion-Aware Fusion of Statistical and Vision-Language Features for Blind Image Quality Assessment

**arXiv ID:** 2606.02002 | [PDF](https://arxiv.org/pdf/2606.02002v1)

**作者:** Bishr Omer Abdelrahman Adam `[一作]` (Northwestern Polytechnical University), Xu Li `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 12335 | [OpenAlex ID](https://openalex.org/A5100342405)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种三流融合的盲图像质量评估框架，将138维自然场景统计特征与SigLIP和CLIP‑H视觉语言嵌入通过乘法门控机制融合，并用混合MSE+PLCC+排序损失训练轻量级MLP头。

**💡 创新点**

创新点包括：①融合经典NSS与大规模VLM嵌入，充分利用低层像素统计与高层语义信息；②引入输入感知门控，按图像内容自适应调节每条流的权重；③在冻结VLM权重的前提下，仅训练约46万参数即可达到SOTA水平，兼顾精度与计算效率。

**🔧 技术方法**

技术细节：多流特征提取（138维NSS、1152维SigLIP、1024维CLIP‑H），PCA降维到256维后标准化；门控网络基于SigLIP特征生成三个[0,2]权重；MLP回归头3层GELU；损失为MSE+0.5*PLCC+0.5*pairwise ranking；训练使用AdamW、cosine调度；跨数据集预训练+微调+集成+MC dropout用于小数据集LIVE‑itW。

**📊 数据集**

使用公开的三个基准数据集：KonIQ‑10k（真实失真）、KADID‑10k（合成失真，25类×5强度）、LIVE‑Challenge in‑the‑Wild（真实失真，1,162张）。

**📈 对比分析**

与现有方法比较：在KADID‑10k上SROCC 0.9715，超越MANIQA(0.946)、LIQE(0.930)等；在KonIQ‑10k与LIVE‑itW上保持竞争力（KonIQ 0.9142，LIVE‑itW 0.8527），并且模型仅使用约46万可训练参数，推理速度快（CPU 0.32 ms/图）。

**⚠️ 局限性**

局限性：冻结VLM权重限制了在真实失真数据集上的最高性能；NSS特征采用全局块平均，对局部失真（如色块、非等心形补丁）不敏感；per‑distortion分析仅在合成数据上完成，未覆盖真实失真类别；模型在小数据集上容易过拟合，需预训练+微调策略。

---

## 564. Scaling Agentic Capabilities via Grounded Interaction Synthesis

**arXiv ID:** 2606.02001 | [PDF](https://arxiv.org/pdf/2606.02001v1)

**作者:** Wenhang Shi `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**通讯引用:** 6542 | [OpenAlex ID](https://openalex.org/A5008721449)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了GAIS（Grounded Agentic Interaction Synthesis）框架，自动化构建协议锚定的工具环境和结构引导的任务，生成多轮高质量交互数据以提升LLM的 agentic 能力。

**💡 创新点**

创新点在于：① 两阶段 grounding 机制——从真实 Model Context Protocol（MCP）服务器抽取工具并通过 LLM 生成可执行 Python 代码；② 结构化任务生成，利用工具依赖图进行复杂路径规划并注入对抗策略；③ 双阶段验证（状态一致性与响应一致性）确保数据真实性，显著克服传统 LLM 生成的偏差、工具多样性不足和任务短链问题。

**🔧 技术方法**

核心技术包括：LLM 驱动的代码转换与测试驱动迭代；功能难度评估与过滤；工具依赖图构建与复杂链路规划；对抗策略注入；双阶段验证（状态与响应）；以及在大规模训练中使用的非思考/思考推理模式。

**📊 数据集**

数据与基准：自建 GAIS 数据集；对比基线数据集 ToolLLM、ToolACE、Nemotron；评测基准 BFCL-V3、τ²-Bench、ACEBench-en；使用的 M‑CPS 服务器源代码作为工具实现来源。

**📈 对比分析**

比较方法：在 7k 样本统一规模下，对 Qwen3‑Base（4B/8B/14B）和 Llama‑3.1‑8B 进行 fine‑tune，评估 PASS@1/accuracy；对照官方 instruction‑tuned 版本。结果显示：GAIS 在所有模型、规模和推理模式下均优于基线，且在 7k 样本时即可匹配或超越官方指令调优模型；在数据效率上相较 Nemotron 提升 2–3 倍；并保持持续增长而非出现平台期。

**⚠️ 局限性**

局限性：① 依赖可公开获取的 MCP 服务器，难以覆盖所有真实应用；② 简化工具实现可能导致细节失真；③ 对知识类任务仍存在一定的遗忘；④ 评测仅覆盖三大基准，未验证更大规模或多语言场景下的泛化；⑤ 对复杂对抗策略的鲁棒性需进一步验证。

---

## 565. Fast and Lightweight Novel View Synthesis with Differentiable Multiplane Image

**arXiv ID:** 2606.02068 | [PDF](https://arxiv.org/pdf/2606.02068v1)

**作者:** Kaidi Zhang `[一作]` (Universiti Malaya), Guanxu Zhu `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于可学习多平面图像（MPI）的多视角新视图合成框架，并结合一阶扩散模型对渲染质量进行提升。

**💡 创新点**

创新点包括：①将MPI作为可学习参数直接通过可微渲染进行优化；②扩展视锥以支持更宽视角运动；③将一阶扩散模型DIFIX同时用于训练监督和推理后处理。

**🔧 技术方法**

使用的技术包括MPI表示、可微渲染、基于PI3的大规模基底模型几何初始化、单步扩散神经增强器（DIFIX）、点云投影和同构变换。

**📊 数据集**

在LLFF真实场景数据集和NeRF Synthetic合成数据集上进行实验。

**📈 对比分析**

与NeRF、PixelNeRF、DeepView和AnySplat等方法对比，本文方法在PSNR/SSIM/LPIPS上均优于大部分基线，渲染速度达135 FPS，模型大小仅22.8 MB，明显快于AnySplat（103 FPS，153 MB），在保留高质量的同时实现了高效与紧凑。

**⚠️ 局限性**

局限性在于MPI在大视角变化和离前向摄像机运动时仍可能出现几何误差和纹理重复；未使用扩散模型时，视图质量仍受几何初始化精度影响。

---

## 566. Unveiling the Entropy Dynamics of Chain-of-Thought Reasoning

**arXiv ID:** 2606.02020 | [PDF](https://arxiv.org/pdf/2606.02020v1)

**作者:** Ting Xu `[一作]` (Chinese University of Hong Kong), Jianye Hao `[通讯]` (Huawei Technologies Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性研究链式推理（CoT）的熵动力学，发现从探索到收敛的两阶段结构并基于此实现早停与推理时缩放；

**💡 创新点**

首次将CoT监控视为经典序列变点检测问题，利用Cumulative Sum（CUSUM）实现无训练、统计最优的可靠性-效率平衡；

**🔧 技术方法**

核心技术为预测熵计算、熵分布估计、CUSUM变点检测、早停决策与加权投票；

**📊 数据集**

使用Bespoke-Stratos-17k（混合任务集）进行熵分析，并在AIME24/25、GPQA-Diamond等标准推理基准上评估；

**📈 对比分析**

与DEER、Dynasor及自一致性（Self‑Consistency）等基线对比，早停下可达63.06%准确率、11% token减少；在推理时缩放中CUSUM加权投票均优于自一致性，提升幅度随采样量增大而增大；

**⚠️ 局限性**

主要局限为对熵分布估计的依赖（需校准数据集），以及在高度相关的熵序列下可能出现的检测延迟和假警报；

---

## 567. When Tabular Foundation Models Transfer Across Modalities: A Systematic Evaluation Across 95 Datasets, 7 Modalities, and Two Regimes

**arXiv ID:** 2606.02106 | [PDF](https://arxiv.org/pdf/2606.02106v1)

**作者:** Julien Lafrance `[一作]` `[通讯]` (Télécom Paris), Julien Lafrance (Télécom Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的分类管线，将ETF预处理、TabICL无上下文推理和温度标定组合起来，适用于视觉、音频、语音、文本、分子、时序和表格七种信号模态；

**💡 创新点**

创新点在于将ETF几何预处理与表格基础模型TabICL结合，并给出无需验证集的几何早停规则；

**🔧 技术方法**

使用的技术包括Equiangular Tight Frame (ETF) 预处理、TabICL表格基础模型、温度标定以及投影深度等几何指标；

**📊 数据集**

使用95个数据集，涵盖七个模态（视觉6个、音频7个、文本5个、语音2个、分子3个、时序5个、表格7个）以及60个经典表格基准；

**📈 对比分析**

对比方法以同一特征（冻结特征）下最强轻量化调优基线为标准，结果显示管线在大多数任务中与基线相当或更优，速度比完整细调快4–200倍；

**⚠️ 局限性**

局限性包括：在低维特征（d≤30）下ETF预处理无效；对极少数提升案例的预判准确率低；未评估回归、极大数据量或流式场景；

---

## 568. PortBERT: Navigating the Depths of Portuguese Language Models

**arXiv ID:** 2606.02100 | [PDF](https://arxiv.org/pdf/2606.02100v1)

**作者:** Raphael Scheible-Schmitt `[一作]` (Technical University Of Munich), Armando B. Mendes `[通讯]` (Azores University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PortBERT——一种基于RoBERTa的葡萄牙语语言模型，训练于450 GB去重后mC4和OSCAR23语料，发布基线与大模型两个变体，并在ExtraGLUE上进行评估，同时公开训练与推理的吞吐量指标。

**💡 创新点**

侧重计算-性能权衡，提供训练与推理效率数据；使用去重、过滤后的本土语料；在公平可复现的fairseq流水线下训练；同时给出开源模型与检查点，填补葡萄牙语模型对效率评估的空白。

**🔧 技术方法**

RoBERTa编码器、byte‑level BPE分词、fairseq训练框架、全精度(fp32)训练、TPU与GPU对比、Masked Language Model预训练、ExtraGLUE下游任务微调。

**📊 数据集**

葡萄牙语mC4与OSCAR23（经CulturaX去重过滤后约456 GB），无额外领域或变体控制。

**📈 对比分析**

与BERTimbau、AiBERTa、AlBERTina、RoBERTa系列、XLM‑RoBERTa、EuroBERT等模型在ExtraGLUE（STS‑B、RTE、WNLI、MRPC）上对标；PortBERT‑base平均分80.57，PortBERT‑large 82.26，均与最强基线相当或略高；同时报告训练吞吐量（samples/s）与推理吞吐量，显示PortBERT‑large在保持较高精度的同时实现了最高的训练与推理效率。

**⚠️ 局限性**

仅使用网页葡萄牙语文本，未针对不同变体或专业语域进行控制；未做WWM或广泛超参调优；评价仅基于ExtraGLUE，缺乏真实测试集与错误分析；TPU版缺乏混合精度与动态内存支持；未对模型偏见与隐私风险进行系统评估。

---

## 569. Overview of the ClinicalSkillQA 2026 Shared Task on Continuous Perception and Procedural Reasoning in Clinical Skill Assessment

**arXiv ID:** 2606.02082 | [PDF](https://arxiv.org/pdf/2606.02082v1)

**作者:** Xiyang Huang `[一作]` (Wuhan University), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17601 | [OpenAlex ID](https://openalex.org/A5077976343)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并组织了 ClinicalSkillQA 2026 共享任务，评估多模态大型语言模型在临床技能评估中的连续感知与程序推理能力，任务要求模型重新排列打乱的关键帧并给出基于临床工作流程的解释。

**💡 创新点**

创新点在于：①设计了专门针对临床操作视频的关键帧时间序列重排序任务；②将解释生成纳入评估，强调模型可解释性；③引入三种评价指标（Task Accuracy、Pairwise Accuracy、BERTScore）全面衡量顺序准确性与解释质量。

**🔧 技术方法**

采用了多模态大型语言模型（如 Qwen3-VL-32B-Instruct、Gemini 3 Pro、GPT-5.2、Qwen2-VL-2B-Instruct 等），结合视觉特征提取、结构化描述、规则指导推理和无监督两阶段框架进行任务求解。

**📊 数据集**

使用了 ClinicalSkillQA 数据集，该集由 SiMing-Bench 临床技能视频抽取的 200 个测试样本组成，涵盖 CPR、AED、BMV 三个急救场景，每个样本包含 4–6 张打乱顺序的关键帧、真实时间序列与专家核实的解释。

**📈 对比分析**

在 7 支队伍的提交中，ZZUNLP 以 71.43 的整体分数（Task Accuracy 0.63，Pairwise Accuracy 0.86，BERTScore F1 0.79）获得榜首，其余队伍表现相对逊色。比较显示模型在保持局部时间关系方面表现更好，完整序列重建仍是主要瓶颈。

**⚠️ 局限性**

局限性包括：①仅包含 200 个测试样本且仅覆盖 3 项急救程序，缺乏多样性；②任务仅使用打乱的关键帧，未涵盖完整视频的运动动态；③BERTScore 对临床准确性与可视化关联的评估不够充分；④为测试集仅任务，不涉及训练或领域自适应。

---

## 570. FACT: A Simple and Efficient Framework for Active Finetuning

**arXiv ID:** 2606.02079 | [PDF](https://arxiv.org/pdf/2606.02079v1)

**作者:** Wenshuai Xu `[一作]`, Zhenghui Hu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了三阶段层次化微调框架FACT（及其变体L^3FACT），专门解决主动微调中数据分布偏移和大模型过拟合问题

**💡 创新点**

创新点在于：①将线性探测→全微调→轻量模型相结合，逐步细化特征并降低过拟合；②引入冻结特征增广（FroFA）提升鲁棒性；③提供LP-LoRA等参数高效变体；②将主动学习与微调流程统一；

**🔧 技术方法**

使用了线性探测、全微调、LoRA、轻量模型（MAP+LR）、冻结特征增广、L‑BFGS优化等技术；在框架内部融合多种PEFT方法

**📊 数据集**

在经典（CIFAR‑10/100、ImageNet‑1k）、长尾（CIFAR‑10/100‑LT）和细粒度（StanfordCars、FGVCAircraft）图像分类数据集上进行实验，并扩展至语义分割ADE20k

**📈 对比分析**

与传统全微调、LP、LP‑FT、LP‑LoRA及现有主动微调方法（ActiveFT、ActiveDC、BiLAF、VeCAF）对比，FACT/L^3FACT在低样本比例下平均提升20%+准确率，显著优于基线；在大规模模型、不同预训练方式（DINO、CLIP）和不同PEFT方法上同样保持优势，训练时间更短、可训练参数极低

**⚠️ 局限性**

限制：在极高分辨率或大规模数据下仍需进一步评估；对极小样本（<0.1%）的鲁棒性未充分验证；轻量模型的泛化仍受限于特征提取器的表达能力

---

## 571. Ablating Archetypes: The Stability of Archetypal SAEs is an Artifact of Initialization and Metric Design

**arXiv ID:** 2606.02061 | [PDF](https://arxiv.org/pdf/2606.02061v1)

**作者:** Michał Brzozowski `[一作]` (Samsung AI Center), Neo Christopher Chung `[通讯]` (Samsung AI Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对稀疏自编码器（TopK SAE）与其在凸包约束下的变体（archetypal SAE）进行对比实验，揭示了先前声称的稳定性实际上源自共享初始化，而非凸包约束本身。

**💡 创新点**

创新点在于将稳定性拆分为“稳定”和“稳定化”，提出沿训练轨迹的稳定化诊断，进行初始化消融和几何度量分析，以澄清对 archetypal SAE 的误导性稳定性声明。

**🔧 技术方法**

采用稀疏自编码器、凸包参数化、RA‑SAE、k‑means 预处理、中心化、余弦匹配、Hungarian 匹配等技术。

**📊 数据集**

实验使用 Pythia‑160m 模型第 6 层残差流激活和 DinoV2 Vision patch 嵌入（单类 “rabbit”）的数据集。

**📈 对比分析**

对比方法包括基于余弦 Hungarian 距离的终端稳定性和 R² 重构得分；结果显示，完全的 archetypal SAE 在 R² 上表现最差，稳定性与共享初始化一致；对比随机初始化的经典 SAE 及消融条件，发现前者更不稳定。

**⚠️ 局限性**

局限性包括未进行超参数搜索、仅在单层/单类设置下实验、以及对训练时间/多任务的覆盖不足。

---

## 572. Respectful Things: Adding Social Intelligence to 'Smart' Devices

**arXiv ID:** 2606.02037 | [PDF](https://arxiv.org/pdf/2606.02037v1)

**作者:** Max Van Kleek `[一作]` (University of Oxford), Nigel Shadbolt `[通讯]` (University of Oxford)

**通讯引用:** 12604 | [OpenAlex ID](https://openalex.org/A5056692594)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文提出将“尊重”作为智能设备设计目标，定义了四种适用于智能设备的尊重类型（指令性、障碍性、识别性、关怀性），并讨论了其在智能家居与IoT中的实现与影响。

**💡 创新点**

创新点在于将人际关系中的“尊重”概念转化为可操作的智能设备设计原则，并首次系统性地将四种尊重类型与设备功能、用户隐私、社会适应性等方面关联。

**🔧 技术方法**

主要采用概念分析、伦理与法规框架（如GDPR、EPSRC机器人五原则）相结合的方式，探讨数据保护、社会适应性设计与设备行为的可解释性；未使用具体算法或实现技术。

**📊 数据集**

无实验数据集，文章为理论性和概念性讨论。

**📈 对比分析**

未进行实验或性能评估，也没有与其他方法进行比较，故无法给出性能指标。

**⚠️ 局限性**

局限性包括：缺乏实证验证与量化评估；责任归属（制造商、用户、设备）不明；技术实现难度大，尤其是设备自适应学习与隐私保护的平衡；法律与商业模式冲突可能阻碍实施。

---

## 573. RL-ACRGNet: Reinforcement Learning-Based Chest Radiology Report Generation Network

**arXiv ID:** 2606.02035 | [PDF](https://arxiv.org/pdf/2606.02035v1)

**作者:** Yogesh Kumar Meena `[一作]` (Indian Institute of Technology Gandhinagar), K. V. Arya `[通讯]` (ABV-Indian Institute of Information Technology and Management)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于强化学习的胸片报告生成网络RL-ACRGNet，结合DenseNet编码器与多层LSTM解码器；

**💡 创新点**

创新点在于融合off-policy actor‑critic框架与多网络（policy、value、reward）协同训练，并使用多级注意力与双向排名奖励提升语义一致性；

**🔧 技术方法**

采用DenseNet+LSTM+多头注意力、Actor‑Critic RL、双向排名损失、BLEU/METEOR/ROUGE等评估指标；

**📊 数据集**

在IU‑Xray与MIMIC‑CXR两个公开胸片报告数据集上进行训练与评估；

**📈 对比分析**

与AdaAttn、METransformer、R2Gen等SOTA方法对比，RL-ACRGNet在BLEU‑4、METEOR、ROUGE‑L等指标上分别提升约0.47%、0.17%和0.518；

**⚠️ 局限性**

存在数据偏倚、模型可解释性不足以及在不同临床场景下的泛化与验证需求。

---

## 574. Evaluating Real-World Generalizability of Algorithm Selection Models

**arXiv ID:** 2606.02016 | [PDF](https://arxiv.org/pdf/2606.02016v1)

**作者:** Gjorgjina Cenikj `[一作]` (Jo\v{z}ef Stefan Institute), Tome Eftimov `[通讯]` (Jo\v{z}ef Stefan Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统评估了算法选择（AS）模型在从学术基准（BBOB、CEC）向真实世界任务（机器人轨迹优化、无人机路径规划）迁移时的泛化能力。

**💡 创新点**

创新点在于：①首次在跨领域（synthetic ↔ real‑world）设定下对AS模型进行严格的交叉基准评估；②利用交叉匹配检验对基准间特征空间相似度进行量化；③探讨了特征预处理与采样密度对泛化的影响。

**🔧 技术方法**

技术上主要使用了基于ELA特征的随机森林多目标回归模型；特征预处理包括无缩放、y缩放、x‑y缩放；采样方案分为大样本（10k）与小样本（50d）。

**📊 数据集**

数据集涵盖四个来源：BBOB（20实例×24类）、CEC（15实例×47题）、ROBOT（4规模×100实例）、UAV（4规模×100实例），维度覆盖6到30。

**📈 对比分析**

对比方法为基准“dummy”均值预测；实验结果显示，只有在特定维度与源/目标基准配对（如12d BBOB→UAV、18d CEC→UAV）下，RF模型可略优于dummy；整体而言，泛化效果有限，往往无法超过基线。

**⚠️ 局限性**

局限性包括：①基准间特征分布差异大，导致迁移效果差；②模型容易对单一优算法过拟合，导致泛化不足；③未尝试更复杂的深度特征或主动学习策略，可能进一步提升跨域性能。

---

## 575. Automated Essay Scoring and Language Certification: Assessing Generalizability, Agreement and Validity for French

**arXiv ID:** 2606.02009 | [PDF](https://arxiv.org/pdf/2606.02009v1)

**作者:** Rodrigo Wilkens `[一作]` (University of Exeter), Thomas François `[通讯]` (Cental, IL and C, UCLouvain)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套可自动化的 Argument‑Based Validation (ABV) 框架改进版本，用于系统化评估法语高考自动作文评分（AES）模型，并在两套大规模 TCF 考试作文语料上对八种模型进行了全面对比。

**💡 创新点**

创新点包括：①将 ABV 框架中的五个评估维度映射到可自动化的量化指标；②在法语 AES 领域首次构建并使用最大规模的核心与泛化语料；③通过软标签集成与投票投票策略提升模型性能；④系统性检验公平性与错误识别，为高考应用提供可监控机制。

**🔧 技术方法**

技术方法主要包括：基于 CamemBERT 的 Transformer 编码器；多种混合架构（Soft‑Labeling、Hard‑Labeling、Simple Concatenation、Concatenate MLP、Implicit Feature 等）；多任务与多模型投票集成；以及多种评估指标（Spearman、QWK、EA、AA、F1、MSE、Kappa、ECE 等）。

**📊 数据集**

使用的数据集为：核心语料 27,683 篇 TCF 试卷（每篇两位评分者），以及泛化语料 961 篇（每篇 9–55 位评分者）。核心语料经过抽样与清洗，保证 CEFR 6 个水平分布均衡；泛化语料则采用多评分者高质量校准。

**📈 对比分析**

采用 10‑折交叉验证训练 8 种模型，对核心与泛化语料分别计算排名、标签识别与一致性指标。结果显示：软标签集成模型在所有指标上均优于其它模型，并在泛化语料上往往优于人工评分者；投票集成虽略有提升但成本更高；模型在 A/B1 层级表现最佳，C2 层级存在一定下滑。

**⚠️ 局限性**

局限性包括：①仅针对法语与 CamemBERT，缺乏跨语言验证；②ABV 的第三维（独立测量）未得到评估；③核心语料的“两评”标注可能导致自洽性偏差；④对提示（prompt）与任务类型的细粒度影响分析仍有限；⑤模型解释性与对错误类型的机制尚未深入探究。

---

## 576. An Agentic Approach Towards Replication Package Quality Evaluation

**arXiv ID:** 2606.02006 | [PDF](https://arxiv.org/pdf/2606.02006v1)

**作者:** Maximilian Alexander Amougou Mbida `[一作]` (Technical University of Munich), Florian Angermeir `[通讯]` (fortiss)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5020380942)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了一个多代理自动评估框架，能够对软件工程研究复制包进行可复现性检查并生成可追溯的改进报告。

**💡 创新点**

创新点在于将开放科学准则转化为机器可验证的评估标准，设计了可插拔的多代理工作流，并通过与人工基准对比展示了可复现性评估的可行性。

**🔧 技术方法**

技术实现基于LangGraph的状态机架构，使用GPT‑5系列 LLM 进行检索、计划、执行和报告；采用 RAG+LLM+检索技术，并实现了人机交互的 HITL 环节。

**📊 数据集**

使用了 380 条原始需求合并而成的 51 条可操作准则（其中 31 条可自动评估），并在 5 个来自已知论文的复制包上进行实验验证。

**📈 对比分析**

与人工基准对比，微平均准确率为 75.4%、宏平均 68.2%；运行一致性达 91.4%；在代码、数据与环境等结构性检查表现良好，但对实验严谨性等定性维度的准确性较低。

**⚠️ 局限性**

局限性包括对定性或混合方法研究的适用性不足、对非标准化结构的误判、对复杂嵌套格式的前处理脆弱，以及 HITL 规划阶段带来的认知负荷。

---

## 577. Machine Learning for Coding Retail Product Names to Consumer-Price Categories: A Rule-plus-Bag-of-Words Pipeline with Reliability-Weighted Human-in-the-Loop Labeling

**arXiv ID:** 2606.02004 | [PDF](https://arxiv.org/pdf/2606.02004v1)

**作者:** Vladimir Beskorovainyi `[一作]` `[通讯]` (Besk Tech), Vladimir Beskorovainyi (Besk Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套针对交易/收据产品描述的混合识别流程，将噪声短文本映射至消费分类以生成 CPI 价格报价。

**💡 创新点**

创新点在于：①将规则基前置分类与 per‑category 二分类模型结合，②引入可靠度加权的“人机循环”标签协议，③系统性验证词袋模型在此任务中已足够强大，无需复杂序列模型。

**🔧 技术方法**

技术包括文本标准化/分词、前缀树（Trie）规则匹配、每类词袋线性/MLP 确认模型、可靠度加权投票与 Dawid–Skene 评估、价格异常检测与单元归一化。

**📊 数据集**

使用大规模交易/收据数据流（数千万行条目/日）以及对单一代表性类别（细砂糖）的人工标注样本，且未公开具体数据集。

**📈 对比分析**

在对单类的实验中，词袋逻辑回归达到 F1≈0.999，MLP 与词袋线性模型差异微乎其微；添加 n‑gram 或子字符特征无提升；在模拟投票实验中，Dawid–Skene 的准确率显著优于可靠度加权投票，后者仅略优于多数表决。

**⚠️ 局限性**

局限包括：仅在单一类别上验证，难以保证在更复杂、多义类别上的泛化；未对交易数据覆盖率与传统实地报价的差异进行量化；模型数量多导致运维复杂；且可靠度加权投票在标签精度上不如潜在能力估计。

---

## 578. WebSpline: Structure-Informed Splines for Real-Time 3D Gaussians from Monocular Videos

**arXiv ID:** 2606.02096 | [PDF](https://arxiv.org/pdf/2606.02096v1)

**作者:** Jongmin Park `[一作]` (Korea Advanced Institute Of Science And Technology), Munchurl Kim `[通讯]` (Korea Advanced Institute Of Science And Technology)

**通讯引用:** 5957 | [OpenAlex ID](https://openalex.org/A5027012300)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于3D高斯的动态场景重建框架WebSpline，能够从单目视频实现高质量、结构一致且高效的实时渲染。

**💡 创新点**

创新点：①将结构代理图（SPG）与结构信息插值样条（SIS）耦合；②两阶段训练（先优化SPG后训练SIS），利用时间刚度与空间/结构邻域约束；③在推理阶段仅使用SIS，避免了混合插值，显著提升渲染速度。

**🔧 技术方法**

采用3D高斯点云、立方Hermite样条、结构代理图、时间刚度损失、空间/结构邻域刚度损失，并结合RGB/深度L1损失进行优化。

**📊 数据集**

在两个公开单目视频基准上评估：iPhone数据集（7个场景）和NVIDIA数据集（7个场景）。

**📈 对比分析**

与现有方法（WorldTree、MoSca、SE3BSplineGS、SplineGS等）比较，WebSpline在mPSNR/mSSIM/mLPIPS、PSNR/SSIM/LPIPS等指标上均名列前茅，同时在iPhone数据集上实现278 FPS，超过WorldTree的27 FPS，提升约10倍。

**⚠️ 局限性**

局限性：仅在单目视频场景验证，依赖2D轨迹初始化；对极端光照或快速摄像机运动的鲁棒性尚未充分评估；推理速度虽快，但仍受限于高斯数量与硬件。

---

## 579. Better with Experience: Self-Evolving LLM Agents for Evidence-Grounded Health Community Notes

**arXiv ID:** 2606.02215 | [PDF](https://arxiv.org/pdf/2606.02215v1)

**作者:** Zihang Fu `[一作]` (National University Of Singapore), Jiaying Wu `[通讯]` (National University Of Singapore)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5101421649)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种自进化的健康社区笔记生成框架（EvoNote），利用过去的纠错经验持续改进后续笔记的质量。

**💡 创新点**

创新点在于将轨迹级的反馈细化为可执行的阶段性记忆（记忆演化器），并通过社交效用评判将经验分配到声称分析、证据获取和写作三个阶段，实现可复用的纠错策略。

**🔧 技术方法**

核心技术包括：基于MedGemma‑27B的大语言模型，社交效用评判（四维健康沟通质量评估），记忆演化器（提取策略记忆），检索式记忆驱动的声称分析与证据获取流程，以及预算约束的行动步骤。

**📊 数据集**

使用了自构建的1.2K多模态健康帖子基准（含文本、图像、视频），每条帖子配有人工撰写的社区笔记、投票得出的有用/无用标签及层级效用评估。

**📈 对比分析**

与人类撰写笔记、GPT‑4.1、Grok‑4.3、Gemini‑2.5‑Flash、CrowdNotes+、DeepResearch、ExpRAG、ReMem等基线比较，EvoNote在文本、图像、视频三种模态下平均对人类笔记的对比胜率达89.6%，在未达成共识的帖子中实现82%有用率，并在证据多样性、来源质量和时间效率上均优于对照方法。

**⚠️ 局限性**

局限性包括：仅针对英文健康信息；记忆仅来自自身生成轨迹，未结合高质量人工笔记；自进化仅在记忆层面，未改变模型或工作流；仅处理已标注的错误帖子，未实现误报检测或实时路由等前端治理功能。

---

## 580. Cross-Environment Neural Reranking for Sample-Efficient Action Selection in Text-Based Agents

**arXiv ID:** 2606.02204 | [PDF](https://arxiv.org/pdf/2606.02204v1)

**作者:** Kan Shao `[一作]` `[通讯]` (Jinglue Technology Development (Nanjing) Co., Ltd.), Kan Shao (Jinglue Technology Development (Nanjing) Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一个轻量化重排序模型在三个不同文本环境下的跨域动作选择能力，证明单一模型可实现跨环境正向迁移。

**💡 创新点**

引入数据重平衡、少样本微调和环境感知的LoRA路由等技术，系统评估了跨环境转移并提出可持续的模块化代理架构。

**🔧 技术方法**

采用DeBERTa-v3编码器+线性重排序头，配合对比损失、二分类点损失、少样本微调、LoRA适配器+PCGrad梯度分离。

**📊 数据集**

统一三环境候选集，包含51,580个训练实例（ALFWorld、WebShop、ScienceWorld），共计455,473个候选，利用原始轨迹和人工演示。

**📈 对比分析**

在统一评估下，基线单环境模型平均net gain为+0.357/0.249/0.142；两环境重平衡后+0.412/0.214，三环境联合后均值+0.551，单个最佳seed+0.611；少样本微调仅需9.2%数据即可恢复93%性能。

**⚠️ 局限性**

仅覆盖三种环境，依赖专家轨迹，候选预先枚举，且环境感知路由方法稳定性差，需进一步验证在更复杂、多模态、实时环境中的鲁棒性。

---

## 581. C2GA: A Class-Controllable Generative Augmentation Framework for Respiratory Sound Classification

**arXiv ID:** 2606.02212 | [PDF](https://arxiv.org/pdf/2606.02212v1)

**作者:** Ziqi Ma `[一作]` (Shanghai University), Sheng Hu `[通讯]` (Osaka University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为C2GA的类可控生成增强框架，用于在稀缺、嘈杂和不平衡的数据环境中进行呼吸声音分类。

**💡 创新点**

C2GA通过构建一个语义丰富的离散潜在空间，结合条件向量量化变分自编码器（VQ-VAE）和基于变换器的自回归先验，提供了一种有效且语义可靠的数据增强策略。

**🔧 技术方法**

使用了条件向量量化变分自编码器（VQ-VAE）和基于变换器的自回归模型。

**📊 数据集**

在两个呼吸声音数据集上进行了实验评估，一个是自建的二分类数据集，另一个是来自ICBHI基准的高质量子集。

**📈 对比分析**

与传统增强方法（如DCRN和ESPnet-SE++）以及生成方法（如Conv-VAE和WaveGAN）相比，C2GA在分类准确性和少数类分离性上均表现出显著提升，准确率提高了1.35到2.20个百分点。

**⚠️ 局限性**

C2GA的局限性在于其对生成样本的控制能力依赖于训练数据的质量和数量，且在极端噪声条件下可能仍然面临挑战。

---

## 582. The Use of Computational Thinking Skills, Difficulties, and Strategies of Introductory Programming Students Solving Bebras Tasks

**arXiv ID:** 2606.02175 | [PDF](https://arxiv.org/pdf/2606.02175v1)

**作者:** Enrico Benedetti `[一作]` (Utrecht University), Johan Jeuring `[通讯]` (Utrecht University)

**通讯引用:** 4017 | [OpenAlex ID](https://openalex.org/A5011892251)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在乌得勒支大学非CS专业的入门Python课程中，收集并分析了58名学生完成5个Bebras任务的书面计划与解答，探讨他们在解决问题时所使用的计算思维（CT）技能、遇到的困难以及采用的策略。

**💡 创新点**

创新点在于：①将CT技能编码与学生书面化解答相结合，系统量化CT技能的出现频率；②通过相关分析揭示CT技能与答案正确率的正向关系；③首次将Bebras任务应用于本科阶段的CT培养与评估，为高等教育CT研究提供新的数据与方法。

**🔧 技术方法**

技术与方法：①采用基于De Jong等人的CT技能代码表进行文本编码，并通过Cohen’s κ检验标注一致性；②对困难与策略进行主题分析；③使用描述性统计和Pearson相关系数评估CT技能、计划、解决方案与正确率之间的关系。

**📊 数据集**

数据集：241份Bebras任务（5个）书面计划与解答、118条困难评论、55条策略反思，来自58名参与学生（非CS专业）在乌得勒支大学的入门编程课程。

**📈 对比分析**

比较方法：将每份解答中出现的CT技能与其最终答案是否正确进行交叉表与相关性检验；结果显示：算法思维和拆分与正确率显著相关，CT技能出现率越高，正确率越高；与先前K‑12或高等教育中Bebras任务的研究对比，证明这些任务能有效激发CT技能，但对评估与泛化等技能的覆盖度不足。

**⚠️ 局限性**

局限性：样本来自单一高校单门课程，缺乏多样性；学生自愿参与，样本可能偏向积极或技术熟练者；仅收集书面记录，缺乏口述或实时观察，可能遗漏隐式CT技能；Bebras任务数量有限，未能覆盖所有CT技能；无法确定因果关系，只能提出相关性假设。

---

## 583. Disentanglement-Based Equivariant Learning for Compositional VQA

**arXiv ID:** 2606.02168 | [PDF](https://arxiv.org/pdf/2606.02168v1)

**作者:** Zhou Du `[一作]` (Southwest Jiaotong University), Changsheng Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 26432 | [OpenAlex ID](https://openalex.org/A5022636178)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

暂无论文摘要信息

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

## 584. An Abstract Worlds Semantic Framework for Belief Change Operators

**arXiv ID:** 2606.02163 | [PDF](https://arxiv.org/pdf/2606.02163v1)

**作者:** Daniel Grimaldi `[一作]` (Universidad de Buenos Aires), Ricardo O. Rodriguez `[通讯]` (Universidad de Buenos Aires)

**通讯引用:** 492 | [OpenAlex ID](https://openalex.org/A5079110572)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了基于抽象世界语义（AWS）的全新集合论框架，用来统一和简化 AGM、KM 与多重变更等经典信念更新与收缩操作。

**💡 创新点**

创新点在于将世界视作原语，构造了可泛化的“多功能”操作符，并通过世界选择函数实现对所有经典与非优先化操作的统一描述；同时将可信度与可疑性等概念标准化为关系而非集合，扩大了对非经典逻辑下操作符的适用范围。

**🔧 技术方法**

采用集合代数、Galois 连接、Suszko 归约与布尔代数等形式化工具，构建了可证明的表示定理、后置定理和相互转换规则；利用 Tarskian 逻辑的结构实现了从抽象世界到信念集合操作符的系统性翻译。

**📊 数据集**

该工作为纯理论研究，无需外部数据集；所有证明均基于逻辑公理与集合论公设。

**📈 对比分析**

由于本研究属于理论框架构建，未进行实验比较；但通过形式化证明表明，AWS 能在任意满足 Tarskian 条件的逻辑中实现 AGM、KM 与多重变更的标准操作，并可在非 AGm 合规逻辑中定义新的可多功能操作符。

**⚠️ 局限性**

局限性在于：1) 主要聚焦于 Tarskian 逻辑，未深入探讨模态、蕴含式、描述逻辑等非经典语义；2) 研究缺乏经验验证，无法评估在实际知识库中的效率与可扩展性；3) 对于极大规模知识基的实现细节与算法优化仍需进一步研究。

---

## 585. On Proof Systems for #QBF

**arXiv ID:** 2606.02143 | [PDF](https://arxiv.org/pdf/2606.02143v1)

**作者:** Sravanthi Chede `[一作]` (Institute of Mathematical Sciences (A CI of Homi Bhabha National Institute)), Anil Shukla `[通讯]` (Indian Institute of Technology Ropar)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

设计并实现了新的 #QBF 证明系统，用于计数满足策略数。

**💡 创新点**

创新点在于将 #SAT 证明系统与 QBF 证明技术结合，提出行基证明系统并证明其对特定 QBF 家族具有指数级优势。

**🔧 技术方法**

使用了行基推理规则、∃/∀ 扩展、组合规则、联接规则以及 FQBF 证明来构建系统。

**📊 数据集**

没有使用实验数据集，全部为理论构造与证明。

**📈 对比分析**

通过对 XOR-PAIRS 与索引仿射 QBF 家族的上界证明，展示了系统与传统系统（朴素计数、扩展基证明）相比的指数分离。

**⚠️ 局限性**

局限在于缺乏真正的下界证明，系统尚未与现有 #QBF 求解器（如 d4-QBF）进行实证对比，且对双指数规模解的处理仍不够。

---

## 586. Do Gender Cues Affect LLM Value Trade-offs? Evidence from a Controlled Decision Benchmark

**arXiv ID:** 2606.02214 | [PDF](https://arxiv.org/pdf/2606.02214v1)

**作者:** Yangyang Liu `[一作]` (Beijing Language and Culture University), Pengyuan Liu `[通讯]` (Beijing Language and Culture University)

**通讯引用:** 1071 | [OpenAlex ID](https://openalex.org/A5100714941)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一套名为 Realistic Value Decision Benchmark (RVDB) 的受控基准，用于检测大型语言模型在价值冲突决策中是否会因角色性别提示而产生偏差。

**💡 创新点**

创新点在于：①通过严格固定情境、价值对、候选决策、决策严重性等因素，只变化角色性别配置，从而精准定位性别提示对决策的影响；②采用位置平衡评估设计，消除展示顺序的干扰；③提出一系列度量指标（Decision Flip Rate、Gender Consistent Flip Rate、Bias Denial Rate 等），系统量化性别偏差与模型自我归因之间的差距。

**🔧 技术方法**

技术手段主要包括：零样本提示模板、统一输出模式（支持决策、理由、性别影响标签、性别影响理由）、多模型对比评估（7款指令微调 LLM）、统计显著性检验与结果可视化。

**📊 数据集**

使用自建数据集 RVDB，涵盖 4 个领域（公民权、工作场所、家庭、教育）、90 个有序 Schwartz 价值对、每对情境下 5 种角色性别配置，总计 18,000 决策实例；数据由 DeepSeek 生成并经过人工审核后挑选。

**📈 对比分析**

对比 7 款 LLM（Qwen、Llama、GLM、GPT-4o-mini 等）在多项指标上的表现：所有模型均显示非零 Decision Flip Rate，女性提议决策的倾向性显著高于男性；Bias Denial Rate 接近 100% 表明模型在自我归因上高度否认性别影响；整体性别偏差受价值距离和决策严重性影响，低价值距离与极端严重性时偏差更大。

**⚠️ 局限性**

局限性包括：①受控基准无法覆盖真实应用中的开放式、多轮价值冲突；②仅使用男性/女性/中性三种性别标签，未涵盖非二元或交叉身份；③缺乏对内部机制的解释，无法揭示性别提示如何在模型内部影响决策。

---

## 587. Consistency Training while Mitigating Obfuscation via Rate Matching

**arXiv ID:** 2606.02211 | [PDF](https://arxiv.org/pdf/2606.02211v1)

**作者:** Sohaib Imran `[一作]` (Independent), David Demitri Africa `[通讯]` (UK AI Security Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了一种新的一致性训练方法——Rate Matching Consistency Training（RMCT），该方法仅对模型在不同输入下的目标行为率进行一致性约束，而不限制模型的表述方式，从而缓解传统一致性训练导致的“模糊化”问题。

**💡 创新点**

创新点在于：①通过匹配不同输入下的行为率（而非完整响应或内部激活）实现一致性训练；②奖励函数仅依赖于目标行为的出现率，保持链式思考等其他输出自由；③使用 GRPO 强化学习与 LoRA 微调结合，实现了对大模型的高效训练，并提供了可灵活指定参考输入与聚合方式的框架。

**🔧 技术方法**

技术细节包括：RMCT 采样多条轨迹计算目标行为率，利用二分类器 T 评估每条轨迹的行为属性；通过 GRPO 进行策略优化并加入 KL 惩罚；采用 LoRA 低秩适配器进行参数高效微调；评估指标为 BSR（Bias Switch Rate）与 BVR（Bias Verbalisation Rate）。

**📊 数据集**

实验使用了 sycophancy 基准数据集（包含 six 种 bias 类型：suggested answer、distractor fact、distractor argument、post hoc、spurious few‑shot squares、wrong few‑shot），并从 LogiQA、HellaSwag 和 Humanity's Last Exam（HLE）中采样多项选择题进行训练与评估。

**📈 对比分析**

对比基准方法 BCT，RMCT 在 held‑out bias 上的 BSR_← 降低与 BCT 相当，甚至在 GPT OSS 20B 上超越 BCT；同时 RMCT 在 BVR 上几乎保持与基线相同，显著降低了 BCT 带来的 obfuscation；在数据效率上 RMCT 只需 64 条数据点即可显著减小 BSR_←，但每条数据的计算成本更高。

**⚠️ 局限性**

局限性包括：①仅在多项选择的 sycophancy 场景和两款开源模型上验证；②评估与训练数据分布差异较大，导致 BSR 下降幅度受限；③BVR 定义过于宽松，可能高估真实的 obfuscation；④计算成本高，尤其在大模型或更大数据集上不易扩展；⑤未探讨对其他不一致问题（如 jailbreak、persona drift 等）的泛化。

---

## 588. The Ghost Couple: Correlated LLM Name Priors and Their Haunting of the Web and Academic Publishing

**arXiv ID:** 2606.02184 | [PDF](https://arxiv.org/pdf/2606.02184v1)

**作者:** Michał Brzozowski `[一作]` (Samsung AI Center), Neo Christopher Chung `[通讯]` (Samsung AI Center)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统探测大型语言模型（Claude、GPT、Gemini）在生成虚构专家时产生的“幽灵名字”及其共同出现的名字组合，进一步追踪这些名字在网络与学术平台（Zenodo、ResearchGate、Google Scholar等）的扩散，揭示大规模数据集被合成作者污染的机制与规模。

**💡 创新点**

创新点包括：①发现并量化模型特定的名字共现组合（非单个名字）以及版本间的抑制曲线；②将模型名称先验作为无标记网络爬取签名，首次将模型行为直接映射到真实学术元数据；③首次在Zenodo上检测到1,655条带真实DataCite DOI但刊物不存在的“幽灵作者”记录，并证明其通过自动化批量上传产生；④利用发布日期的时间漂移为模型部署窗口提供新的时间指纹。

**🔧 技术方法**

技术方法：①对可访问的模型检查点使用API，设计单人、双人、三人prompt（30个示例）并统计名字出现频率与共现率；②使用Serper.dev Google搜索API抓取包含目标名字的URL与片段，提取标题/描述中的共现；③通过Zenodo、DataCite、ResearchGate、Semantic Scholar API抓取论文元数据，检测DOI、出版日期与期刊真实性；④对比不同模型版本的抑制曲线与共现比例，绘制时间序列。

**📊 数据集**

数据集：①官方API可访问的Claude、GPT、Gemini检查点（共9+10+若干版本）；②针对每个模型生成的30个prompt的响应；③Serper检索得到的超过2,000个网页URL及片段；④Zenodo、DataCite、ResearchGate、Semantic Scholar公开元数据，覆盖1,655条伪造记录与约436条Ghost作者记录。

**📈 对比分析**

比较方法：对同一prompt在不同模型/版本间计算名字单独出现率、配对共现率和三人组合共现率；与训练数据中名字出现频率做对比，验证抑制效果；对Zenodo上传速率与常规上传速率对照，展示991条记录在单月内上传的异常峰值；性能方面，Claude单人名字最高共现率达67%，Gemini达93%，GPT仅为23%，共现率随版本更新呈显著下降。

**⚠️ 局限性**

局限性：①仅覆盖公开API检查点，未考察内部或微调模型；②prompt规模有限（每类30个），可能漏检低频名字；③网络爬取受Google索引延迟与偏差影响，网页发布时间不可靠；④Ghost名字可能与真实研究者重叠，无法完全排除误判；⑤未能对已出版的学术文献进行全面溯源，部分数据可能受限。

---

## 589. On the Generalization in Topology Optimization via Sensitivity-Conditioned Bernoulli Flow Matching

**arXiv ID:** 2606.02179 | [PDF](https://arxiv.org/pdf/2606.02179v1)

**作者:** Mohammad Rashed `[一作]` (Technical University of Munich), Nils Thuerey `[通讯]` (Technical University of Munich)

**通讯引用:** 3051 | [OpenAlex ID](https://openalex.org/A5047248117)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于灵敏度（adjoint）或伪灵敏度条件的离散伯努利流匹配模型，用于高效且能在分布外（OOD）场景下生成拓扑结构

**💡 创新点**

理论证明灵敏度是信息论上最优的条件信号，定义伪灵敏度并推导结构与CFD问题中的对应关系；首次将伯努利流匹配应用于二值拓扑生成；公开10k样本的多口径CFD-TO数据集

**🔧 技术方法**

信息论分析（DPI）、伪灵敏度定义、离散伯努利流匹配、层次视觉变换器、AdaLN时间编码、终端贪婪采样与体积分配剪枝

**📊 数据集**

结构合规性基准（Topodiff、NITO）以及新构建的10k CFD-TO（2D湍流通道流，单口径训练，多口径OOD）

**📈 对比分析**

与传统一次迭代STAR‑CCM+、UDiT、DiT、PDE‑Transformer及Topodiff、NITO比较；在结构合规性中，伯努利流匹配模型在OOB上实现0.54%中值合规误差，优于Topodiff（1.14%）和NITO（2.37%）；在CFD中，伯努利模型在三口径OOD上达74.5%准确率，优于PDE‑T（68.6%）和UDiT（62.7%），并在推理速度与计算成本上显著领先

**⚠️ 局限性**

依赖于物理求解（单步仿真或附加的adjoint求解）来获得灵敏度/伪灵敏度；目前仅在二维结构和二维湍流RANS问题上验证；对高Re或多物理场、路径依赖性多目标优化等场景的适用性仍待进一步研究

---

## 590. Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization

**arXiv ID:** 2606.02178 | [PDF](https://arxiv.org/pdf/2606.02178v1)

**作者:** Yiming Wang `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 8061 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FLAME 框架，利用局部邻接差异 (LAD) 图捕捉扩散模型产生的能量异常，并结合轻量 LAD-Net 与 SAM 适配器实现像素级伪造定位，同时设计 EditStream 自动生成训练数据。

**💡 创新点**

理论证明扩散过程产生局部能量缺口并提出 LAD 统计量，结合参数高效的 SAM 适配器实现精细定位，并引入自进化 EditStream 解决基准滞后。

**🔧 技术方法**

使用 Gibbs 能量建模、LAD 算子、轻量 CNN（LAD‑Net）、SAM 分割网络与适配器、以及自动化数据合成管道。

**📊 数据集**

训练集采用 MagicBrush、SID；Fine‑Tuned 版本使用 EditStream 生成的 Qwen‑Image‑Edit、Nano Banana、Flux 等；评测覆盖 MagicBrush、SID、CoCoGLIDE、AutoSplice、Nano Banana、Qwen‑Image‑Edit、Flux Kontext。

**📈 对比分析**

与 TruFor、SAFIRE、Mesorch、AdaIFL、SparseViT、SIDA、FakeShield 等基线对比，基于 IoU、F1、ACC、AP 进行评估，FLAME 在 OOD 数据上平均 IoU 超过 0.54、F1 超过 0.65，显著优于其他方法；Fine‑Tuned 版进一步提升并保持历史基准。

**⚠️ 局限性**

对 JPEG 压缩和高斯噪声等强局部统计扰动的鲁棒性有限，LAD 对高频信息依赖，极端后处理下精度下降；仅针对扩散生成伪造，未充分评估其他生成模型或多模态伪造。

---

## 591. Low-Pass Flow Matching

**arXiv ID:** 2606.02177 | [PDF](https://arxiv.org/pdf/2606.02177v1)

**作者:** Francesco M. Ruscio `[一作]` (ELLIS Institute), T. Konstantin Rusch `[通讯]` (ELLIS Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种低通流匹配（LP‑FM）方法，通过时间可调的谱偏置改进传统白噪声注入的流匹配模型。

**💡 创新点**

设计了基于线性平移不变算子调制的插值器，使噪声谱随时间从白噪声平滑过渡到低通谱，从而实现自适应频率衰减的谱偏置。

**🔧 技术方法**

采用条件流匹配（CFM）框架，结合低通参数化（RLP/GLP）、U‑Net骨干、EMA、Dropout 等技术，并使用自适应 ODE 求解器（Dopri5/Tsit5）进行训练与采样。

**📊 数据集**

在 Dead Leaves、CIFAR‑10 与 Galaxy10（DECaLS）三组数据集上进行实验评估。

**📈 对比分析**

与 FM、CFM、VP‑FM 等基线在 FID 与 NFE 上对比，发现 RLP‑CFM 在自适应求解器下既保持或提升 FID，又显著降低 NFE（Galaxy10 NFE 降至 1.65–2.16 倍），但在固定步长 Euler 求解器下效果不佳。

**⚠️ 局限性**

局限性：在 Galaxy10 等科学数据上仍存在训练不稳定的情况；GLP‑CFM 在某些任务中无法收敛；自适应谱偏置在固定步长求解器下不具优势。

---

## 592. InfoMerge: Information-aware Token Compression for Efficient Video Large Language Models

**arXiv ID:** 2606.02161 | [PDF](https://arxiv.org/pdf/2606.02161v1)

**作者:** Xinxin Liu `[一作]` (Nanjing University), Sanglu Lu `[通讯]` (Nanjing University)

**通讯引用:** 8565 | [OpenAlex ID](https://openalex.org/A5043533769)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的视频令牌压缩方法 InfoMerge，能够在不影响 Video‑LLM 性能的前提下显著减少视觉令牌数量。

**💡 创新点**

创新点包括：① 用第二阶时间指纹差分（Temporal Fingerprint Difference, TFD）对视频段内时序冗余进行稳健估计；② 基于段唯一性与谱熵的内容感知预算分配（Content‑Aware Budget Allocation, CABA）动态分配令牌；③ 将 TFD 与 CABA 结合进空间‑时间压缩管线，实现高压缩率下保持性能。

**🔧 技术方法**

技术手段包括：DySeg 动态时间段分割；TFD 通过构建时间相似性指纹矩阵计算二阶差分；CABA 计算段唯一性（全局余弦距离）与谱熵（SVD + 归一化熵）；注意力驱动的令牌选择与冗余抑制；基于密度的令牌合并；所有步骤均不需额外训练，直接对原始视频令牌进行操作。

**📊 数据集**

使用了 VideoMME、MLVU、LongVideoBench 等多种视频理解基准数据集，评估基于 LLaVA‑OneVision‑7B、LLaVA‑Video‑7B 等 Video‑LLM 模型的压缩效果。

**📈 对比分析**

与 FastV、VisionZip、FastVID、MMG‑Vid 等现有训练无关压缩方法进行对比。实验表明，在 15% 令牌保留率下 InfoMerge 保留 98.8% 的原始性能，视觉令牌仅保留 12.9%，预填阶段加速 4.24×，生成阶段加速 3.95×；在 10% 与 5% 保留率下也保持领先或相当的性能，并在跨 backbone（LLaVA‑Video‑7B）测试中取得 97.1% 的性能保留。

**⚠️ 局限性**

局限性包括：① TFD 对持续抖动或极端视觉噪声区域的冗余估计可能不够稳健；② 在极大规模或特定领域的视频分布中未做充分验证；③ 虽然压缩后 FLOPs 降低，但冗余估计和谱分析本身仍带来一定算力开销，可能在资源极度受限的环境下不够轻量。

---

## 593. VLBM: Variational Latent Basis Modeling for OOD Robust Multivariate Time Series Forecasting

**arXiv ID:** 2606.02138 | [PDF](https://arxiv.org/pdf/2606.02138v1)

**作者:** Xudong Zhang `[一作]` (University of Chinese Academy of Sciences), Haina Tang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 591 | [OpenAlex ID](https://openalex.org/A5110364708)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对多变量时间序列预测中稀有高影响的 OOD 事件，提出 VLBM 模型实现 OOD 可靠预测

**💡 创新点**

创新点包括：1）学习共享低秩潜在基底以捕获稳定的 ID 动态；2）通过正交基底残差通道显式分离 OOD 偏差；3）使用后验-先验变分转移把未来感知的潜在激活迁移至推理时可用的先验；4）基于基底的 Base–Residual 生成器实现稳定与偏差的双分支解耦

**🔧 技术方法**

采用变分推断、低秩潜在基底投影、正交残差分解、图结构编码器、Base 与 Residual 双路径、ELBO 训练框架

**📊 数据集**

使用 12 个公开多变量时序数据集（Weather、ECL、Solar、Flight、PEMS03/04/07/08、MSL、PSM、CHP-LCS-Flow、CHP-LCS-Speed）以及人工构造的 Synthetic Graph Pulse 进行实验

**📈 对比分析**

与 12 个基线（Transformer、Memformer、iTransformer、DUET、TimeLLM、TimeMixer++、FilterTS、DLinear、ModernTCN、STONE、MegaCRN、TimePro、TimeKAN）进行对比；在 ID 与 OOD 环境下均实现 MAE/MSE 领先，平均 OOD MAE/ MSE 分别比最强基线提升约 15.08%/7.74%，在 OOD 测试集上取得最优性能

**⚠️ 局限性**

局限性包括：假设稳定子空间与 OOD 偏差可以分离，若两者高度耦合效果下降；未利用外部事件或天气信息；对低秩基底容量的选择需经验调优

---

## 594. Edge-aware Decoding for Neural Asymmetric Routing

**arXiv ID:** 2606.02136 | [PDF](https://arxiv.org/pdf/2606.02136v1)

**作者:** Li Liang `[一作]` (Sun Yat-Sen University), Zizhen Zhang `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 1934 | [OpenAlex ID](https://openalex.org/A5100751707)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了神经网络在不对称路径规划中的解码器设计，提出了边缘感知（edge‑aware）解码器；

**💡 创新点**

创新点在于在最终得分中显式加入当前有向边成本、闭合成本及静态前瞻等转移级信息，缓解了表示与决策不匹配的问题；

**🔧 技术方法**

采用了RADAR的SVD/Sinkhorn异向性编码器，结合MLP边缘感知偏置，并通过强化学习进行端到端训练；

**📊 数据集**

使用了合成的ATSP和ACVRP实例数据集，规模分别覆盖100、200、500、1000个节点；

**📈 对比分析**

通过与RADAR及其他基线在ATSP/ACVRP上的零射向评估对比，边缘感知解码器在ATSP‑1000上将缺口从4.13%降低到2.73%，在ACVRP各规模亦表现略优，虽然推理时间略增；

**⚠️ 局限性**

局限在于额外的推理开销显著，闭合与前瞻仅为启发式近似，且实验仅在合成数据上验证，未在真实物流场景中测试。

---

## 595. Rethinking Evaluation Paradigms in IBP-based Certified Training

**arXiv ID:** 2606.02134 | [PDF](https://arxiv.org/pdf/2606.02134v1)

**作者:** Konstantin Kaulen `[一作]` (RWTH Aachen University), Holger H. Hoos `[通讯]` (RWTH Aachen University)

**通讯引用:** 28529 | [OpenAlex ID](https://openalex.org/A5025342513)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 IBP‑based 认证训练方法进行多目标超参数优化，获取完整的自然准确率与认证准确率 Pareto 前沿，并以此重新评估并比较多种方法在不同扰动半径与数据集上的性能。

**💡 创新点**

提出了标准化的多目标评估范式，揭示之前实验报告的配置往往非 Pareto 最优；通过全局搜索发现各方法的互补优势，刷新了对小 ε 与大 ε 训练方法的真实性能认知。

**🔧 技术方法**

采用贝叶斯多目标超参数优化（Gaussian Process + EHVI）结合 IBP、CROWN‑IBP、SABR、MTL‑IBP 等损失函数；使用不完整验证（IBP、CROWN）做预筛选，随后用完整验证（αβ‑CROWN）精确评估 Pareto 配置。

**📊 数据集**

主要使用 CIFAR‑10、Tiny ImageNet（以及 MNIST 作为补充）进行实验，探讨 ε=2/255、8/255 与 1/255 三种扰动半径。

**📈 对比分析**

通过对比每种方法在 Pareto 前沿上的主导关系，发现此前文献的配置大多被新找到的前沿所支配；在小 ε 下 SABR 在自然准确率上领先，MTL‑IBP 在认证准确率上更强；在大 ε 下 IBP/CROWN‑IBP 与 SABR/MTL‑IBP 共同构成最优前沿，提升幅度约 1–6%。

**⚠️ 局限性**

主要局限在于极高的计算成本（完整验证时间数百小时）、仅针对 ℓ∞ 误差模型、对测试集的调参可能导致过拟合、以及结果对不同网络结构、数据域或更复杂威胁模型的可迁移性尚未验证。

---

## 596. Variational Learning for Insertion-based Generation

**arXiv ID:** 2606.02133 | [PDF](https://arxiv.org/pdf/2606.02133v1)

**作者:** Yangtian Zhang `[一作]` (Yale University), Jiaxin Shi `[通讯]` (Meta Superintelligence Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可变长度插入式生成模型 Insertion Process (IP)，并给出了一套变分学习框架，能够同时学习插入位置、插入内容和终止决策。

**💡 创新点**

创新点主要包括：① 将插入轨迹与全局排列一一对应，得到精确的似然重参数化；② 在此基础上构造基于排列的变分下界，并利用 Gumbel‑Top‑k+REINFORCE Leave‑One‑Out 进行高效训练；③ 通过变分后验学习数据驱动的插入顺序，实现可变长度、非左到右的自适应生成。

**🔧 技术方法**

技术实现包括：Transformer 编码器/解码器、Plackett‑Luce 排列模型、Gumbel‑Top‑k 采样、REINFORCE Leave‑One‑Out 梯度估计、变分下界（ELBO）以及插入过程的概率推导。

**📊 数据集**

实验使用的主要数据集有：规划任务的 Maze Planning（Braided/Perfect/Imperfect 迷宫）和 Star‑Graph Planning；分子生成任务的 GuacaMol SMILES 基准；以及若干条件生成任务（fragment decoration、linker design 等）。

**📈 对比分析**

与固定左到右的 FO‑ARM、任意顺序的 AO‑ARM、Diffusion 模型 MDM、学习顺序的 LO‑ARM、FlexMDM 等基线比较。IP 在所有规划任务中均击败基线，准确率、编辑距离均显著提升；在 SMILES 生成中，IP 的 Validity、Uniqueness、Novelty 均高于基线，且学习顺序的版本优于随机插入版本，说明学习插入顺序能提升生成质量。

**⚠️ 局限性**

局限性包括：在分子任务中对分布匹配（KL/FCD）略逊于 FO‑ARM/LSTM；训练过程中需要 Gumbel‑Top‑k 采样和 REINFORCE，收敛速度相对慢；目前仅针对离散序列，难以直接迁移到连续或非序列数据；在极长序列上的可扩展性尚未充分验证。

---

## 597. How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning

**arXiv ID:** 2606.02119 | [PDF](https://arxiv.org/pdf/2606.02119v1)

**作者:** Jiangwei Chen `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**通讯引用:** 811 | [OpenAlex ID](https://openalex.org/A5030304400)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于硬度感知的多目标机器无学习算法（HAMU），在保证遗忘质量提升的同时，最小化保留数据性能损失。

**💡 创新点**

创新点在于用梯度点积定义硬度度量，并从受限优化理论出发推导最优更新规则和停止准则，实现可理论保证的硬度感知无学习。

**🔧 技术方法**

采用梯度点积硬度度量、线性近似受限优化、可并行的层级更新以及随机批量梯度。

**📊 数据集**

在CIFAR‑10、ImageNet（ResNet‑20）和LLM Llama‑2‑7b‑Chat 与 WaterDrum‑TOFU 问答数据集上实验。

**📈 对比分析**

与 FT、GA、GDiff、KL、SCRUB 等基线对比，HAMU 在不同硬度（相似度）场景下均能在遗忘质量与保留性能上取得更好的 Pareto 前沿，尤其在高相似度难题上表现突出。

**⚠️ 局限性**

局限性包括每步需估计梯度点积、理论仅在局部线性近似下成立，以及在极端相似情况下仍不可避免的协同遗忘。

---

## 598. Coherent Off-Policy Improvement of Large Behavior Models with Learned Rewards

**arXiv ID:** 2606.02194 | [PDF](https://arxiv.org/pdf/2606.02194v1)

**作者:** Christian Scherer `[一作]` (Technical University of Darmstadt), Jan Peters `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在大型行为模型（LBM）上利用逆强化学习（CSIL）对专家演示数据进行微调，以提升稀疏奖励下的机器人抓取与插入任务表现。

**💡 创新点**

提出将CSIL作为稠密奖励学习方式，避免RL中常见的“unlearning”现象，并通过批量归一化、权重归一化、分类价值网络和更强的图像编码器改进，将其扩展至更难的视觉任务。

**🔧 技术方法**

使用逆强化学习（CSIL、CSIL++）、软演员-评论家（SAC）、批量归一化、权重归一化、分类价值网络、HetStat stationary policy、改进的图像编码器以及残差/集成动作策略。

**📊 数据集**

使用Robomimic与MimicGen的模拟抓取/插入任务数据集，演示样本约200条，任务奖励稀疏。

**📈 对比分析**

与稀疏奖励RL基线（XQC+OD、PLD）以及原始CSIL比较，CSIL++在五个任务中实现最高成功率，尤其在插入任务中从14%提升至92%，且无初始性能下降。

**⚠️ 局限性**

依赖高质量的演示；演示不足或子最优时样本效率下降并可能强化错误行为；在长周期任务中仍存在优化不稳定性。

---

## 599. Closing the Alignment-Maturity Gap in Federated Prototype Learning

**arXiv ID:** 2606.02172 | [PDF](https://arxiv.org/pdf/2606.02172v1)

**作者:** Mario Casado-Diez `[一作]` (Universidade da Coruña), Bertha Guijarro-Berdiñas `[通讯]` (Universidade da Coruña)

**通讯引用:** 1665 | [OpenAlex ID](https://openalex.org/A5046448840)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedSAP 框架，解决 Federated Learning 中原型对齐的成熟度差距，提升分布式视觉表示学习。

**💡 创新点**

创新点在于两项机制：逐步对齐课程调度以及利用现有原型作为几何代理的分离损失。

**🔧 技术方法**

采用原型聚合、线性 warm‑up 对齐权重、单位超球面上的余弦软最大化代理损失，兼容无监督伪标签。

**📊 数据集**

在 FEMNIST、CIFAR‑10、CIFAR‑100 三个视觉基准上验证。

**📈 对比分析**

与 FedAvg、FedProto、FedMPS 对比，FedSAP 在所有异构程度下均获得 1–4pp 的准确率提升，且通信成本不变。

**⚠️ 局限性**

局限性包括对超球面代理的依赖、在原型基推理下性能略逊、未探究开放集或自监督预训练。

---

## 600. InsightVQA: High-Dimensional Emotion-Cognitive Visual Question Answering Benchmark

**arXiv ID:** 2606.02171 | [PDF](https://arxiv.org/pdf/2606.02171v1)

**作者:** Shiyu Wang `[一作]` (East China Normal University), Yan Wang `[通讯]` (East China Normal University)

**通讯引用:** 214983 | [OpenAlex ID](https://openalex.org/A5100437036)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了InsightVQA三级情绪认知视觉问答数据集，并提出InsightNet模型进行层级情绪理解和认知推理。

**💡 创新点**

创新点在于将情绪理解分为感知、理解和认知三层，提供基于视觉触发的解释与决策序列，且首次发布面向情绪认知的Benchmark。

**🔧 技术方法**

采用多模态大语言模型（Qwen2.5-VL-7B）结合LoRA微调、视觉触发提取、指令调优与层级评估。

**📊 数据集**

使用从6大公开数据源采集的351K图像筛选得到138K图像，生成725K问答对；Benchmark包含30K样本。

**📈 对比分析**

与多款开源与闭源MLLM（如Qwen2.5-VL-32B、Gemini、Claude等）对比，InsightNet在感知层准确率达76.25%，F1达90.56，显著领先；在理解和认知层也表现最优。

**⚠️ 局限性**

模型在认知推理和序列排序上的表现仍低于人类，且数据集依赖自动生成与人工审核，存在潜在的标注偏差与多样性不足。

---

## 601. From Capability Models to Automated Planning: An AAS-Native Approach for Automatic PDDL Generation

**arXiv ID:** 2606.02167 | [PDF](https://arxiv.org/pdf/2606.02167v1)

**作者:** Hamied Nabizada `[一作]` (Helmut Schmidt University Hamburg), Alexander Fay `[通讯]` (Ruhr University Bochum)

**通讯引用:** 4991 | [OpenAlex ID](https://openalex.org/A5016841027)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种自动从工业4.0资产管理壳（AAS）能力模型生成完整PDDL规划问题的方法，支持多文件分布式架构；

**💡 创新点**

创新点在于无需PDDL专业知识，直接利用现有标准化子模型（VDI3682、IEC61360-1、IDTA02011/02016）和跨AAS引用机制完成规划元素提取；

**🔧 技术方法**

使用AAS标准化子模型、参考引用解析、五阶段提取算法、UPF框架和Fast Downward规划器；

**📊 数据集**

以Festo MPS500实验系统的AAS模型为数据集，包含30条输送带段、7个工位、4个运输载体等；

**📈 对比分析**

通过对四种布局变体生成PDDL并求最优计划，发现内置短路路径可将最优计划长度从65步缩短至35步（约46%），规划时间从6.8s到219s不等；

**⚠️ 局限性**

局限于仅支持PDDL 2.0（无持续时间和数值算子），且模型仍需人工手工编写，尚未在大规模工业系统上验证。

---

## 602. On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective

**arXiv ID:** 2606.02158 | [PDF](https://arxiv.org/pdf/2606.02158v1)

**作者:** Yikai Guo `[一作]` (Beijing Institute of Computer Technology and Application), Haoran Luo `[通讯]` (Nanyang Technological University)

**通讯引用:** 1430 | [OpenAlex ID](https://openalex.org/A5101634507)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于多尺度不确定性估计的零样本 AI 生成文本检测方法（Uncertainty 与 Uncertainty++），强调低概率词的判别作用并利用 Renyi 熵捕捉分布形状，显著提升检测性能。

**💡 创新点**

创新点包括：① 用低概率词（bottom‑ρ）聚焦信息密集区，缓解常见词汇“模板”对判别的干扰；② 将局部日志概率与全局 Renyi 熵融合，形成鲁棒的不确定性信号；③ 通过条件独立采样对局部信号进行标准化，得到 Uncertainty++，进一步提升对攻击与解码变化的稳健性。

**🔧 技术方法**

技术核心为：统计语言模型的条件概率分布、分位数聚合（百分位低尾均值）、Renyi 熵（α 取 0.5~2）、条件独立采样（m 次）与归一化，以及线性融合权重 β；实现方式为无监督、零样本的黑盒/白盒检测框架。

**📊 数据集**

实验使用七个跨领域数据集（XSum、SQuAD、WritingPrompts、Reddit、FiQA、Wiki-csai、arXiv），并评估十六种源模型（包括 GPT‑2、Llama、Gemini‑3‑Flash、GLM‑4.5‑Flash 等），以及多种解码策略与重写攻击。

**📈 对比分析**

与现有九种统计基线（Likelihood、LogRank、DetectLRR、Lastde 等）以及采样基线（DetectGPT、Fast‑DetectGPT、Lastde++ 等）比较，Uncertainty 与 Uncertainty++ 在黑盒平均 AUROC 分别提升 2–3% 以上，白盒平均 AUROC 超 99%；在不同域、长度、最新 LLM 与攻击场景下均保持领先，鲁棒性优于对手。

**⚠️ 局限性**

局限性包括：对极短文本或高度专业化领域（如财务文本）效果仍有限；依赖代理模型的概率估计，若代理模型与目标生成模型差距大，性能可能下降；对高质量重写攻击仍有一定下降空间。

---

## 603. Ultra Diffusion Poser: Diffusion-Based Human Motion Tracking From Sparse Inertial Sensors and Ranging-Based Between-Sensor Distances

**arXiv ID:** 2606.02153 | [PDF](https://arxiv.org/pdf/2606.02153v1)

**作者:** Dominik Hollidt `[一作]` (ETH Zurich), Christian Holz `[通讯]` (ETH Zurich)

**通讯引用:** 7153 | [OpenAlex ID](https://openalex.org/A5046815740)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于六轴惯性测量单元(IMU)和超宽带(UWB)距离测量的全身姿态估计方法——Ultra Diffusion Poser (UDP)，通过在扩散模型中显式建模UWB几何约束来提升姿态重建精度。

**💡 创新点**

核心创新在于（1）Spatial Layout Module（SPL）利用多维尺度(MDS)与可学习旋转估计器，实时重建3D传感器布局；（2）UWB‑Diffusion Guidance，在扩散采样过程中通过前向运动学与UWB距离误差梯度引导，确保预测姿态与测距数据一致；（3）将上述模块嵌入自回归扩散填充模型，实现平滑且符合物理约束的连续运动。

**🔧 技术方法**

技术手段包括：多维尺度(MDS) + 可学习旋转估计器、SMPL人体模型、6D姿态表示、Diffusion Probabilistic Model（DDPM）与自回归扩散填充、前向运动学（FK）以及基于梯度的扩散引导。

**📊 数据集**

使用公开的大规模运动捕捉数据集（AMASS、DIP‑IMU、DanceDB、TotalCapture）以及真实采集的IMU+UWB数据集（UIP‑DB、GIP‑DB）进行训练与评估。

**📈 对比分析**

与现有IMU+UWB方法(UIP、UMotion、GIP)以及IMU‑only方法(PNP、GlobalPose、DynaIP、PIP、TIP)进行对比，UDP在大多数数据集上均取得最高精度，关节位置误差（JPE）提升最多22%，并在真实场景下保持低抖动（jitter）和优秀的角度误差。

**⚠️ 局限性**

主要局限：对UWB测距噪声高度敏感，过度依赖高精度UWB；未整合物理约束或地面信息，难以处理零足滑或地面穿透等极端情况；目前仅支持固定的六传感器布置，缺乏对任意传感器配置的适应性。

---

## 604. S3TS: Stochastic Scenario-Structured Tree Search for Advanced Planning Under Uncertainty

**arXiv ID:** 2606.02151 | [PDF](https://arxiv.org/pdf/2606.02151v1)

**作者:** Fabio Pavirani `[一作]` (IDLab Ghent university -- imec), Chris Develder `[通讯]` (IDLab Ghent university -- imec)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种名为 S3TS 的蒙特卡洛树搜索算法，用于在具有不确定性和非线性动力学的能源系统需求响应信号发布问题中做出高级规划。

**💡 创新点**

创新点在于：①将多阶段随机优化的场景树直接嵌入 MCTS，避免在线采样或隐式学习；②通过预构建的场景树保持对不确定性的显式建模；③在同一场景树下与传统随机 MPC 进行公平比较，揭示了场景树对性能的显著影响。

**🔧 技术方法**

核心技术包括：蒙特卡洛树搜索（UCT 选择、回溯更新）；场景树构建（采样、降维、聚类、路径一致性组装）；线性/非线性动力学模拟；混合整数线性规划（MILP）对照；基准规则策略；匹配计算预算的实验框架。

**📊 数据集**

使用合成的系统不平衡（SI）随机过程生成数据，共 1000 个独立调节期，每期 15 步（1 分钟粒度），数据来自一阶自回归加正弦季节性与噪声的组合模型。

**📈 对比分析**

通过在 1、2、4、8 秒的匹配时间预算下对比 Rule‑Based、Perfect Knowledge、Stochastic MPC、Deterministic MPC、MCTS 与 S3TS。结果显示：在线性模型中，S3TS 与 Stochastic MPC 的 MAE 仅相差 13.7%（最高 8 秒）；在非线性模型中，S3TS 在 MAE 上比 MCTS 提升 2.2–3.9%，在 MSE 上提升 3.1–5.4%，并显著优于规则基准。

**⚠️ 局限性**

局限性：①性能高度依赖于场景树的质量；②在已知可用 MILP 的线性情形下仍落后于 Stochastic MPC；③对计算预算敏感，低预算下效果不明显；④仅在单一合成问题上验证，缺乏跨域通用性；⑤未对风险敏感度或先验价值进行扩展。

---

## 605. Hybrid Neural Ordinary Differential Equations for Data-Efficient Polymerization Modeling with Incomplete Kinetics

**arXiv ID:** 2606.02145 | [PDF](https://arxiv.org/pdf/2606.02145v1)

**作者:** Marah Almanasreh `[一作]` (RWTH Aachen University), Eike Cramer `[通讯]` (University College London)

**通讯引用:** 139 | [OpenAlex ID](https://openalex.org/A5074964842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出一种将机理模型与神经网络闭包相结合的Hybrid NODE框架，用于在数据稀缺条件下预测MMA自由基聚合动力学，并通过有限采样和噪声数据验证其性能。

**💡 创新点**

创新点在于仅用神经网络学习未建模的自由基浓度闭包项，而非完整动力学，从而显著减少所需数据量并提升外推能力。

**🔧 技术方法**

使用了神经常微分方程（NODE）、离散时间前馈神经网络、基于Python的自动微分与可微积分求解器，以及软约束正则化。

**📊 数据集**

数据集为利用基于Butté-Morbidelli pivot离散化的机理模拟器生成的MMA批量聚合轨迹，共25个归一化时间点，训练时仅用前10点，包含正则和不规则采样以及加入高斯噪声的全景数据。

**📈 对比分析**

与单纯离散时间FNN和全数据驱动NODE在三种评估情景（正则、非规则采样、噪声未知工况）下比较，Hybrid NODE在未见温度下RMSE仅为0.013，远低于FNN（0.68）和NODE（0.31），且在稀疏采样中保持物理一致的外推。

**⚠️ 局限性**

局限性包括仅在仿真数据上验证，未考虑实验噪声、传输限制、温度耦合和聚合物分布预测，未来需在实验系统中检验并扩展至更复杂机理。

---

## 606. A Primer in Post-Training Reasoning Data: What We Know About How It Works

**arXiv ID:** 2606.02113 | [PDF](https://arxiv.org/pdf/2606.02113v1)

**作者:** Yaoming Li `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 72496 | [OpenAlex ID](https://openalex.org/A5100359646)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对超过150篇后训练推理数据相关论文进行系统综述，提出统一的归因框架和四大核心问题（对象、实用性、构建方式、规模）。

**💡 创新点**

首次将推理数据从单一prompt–response对扩展为“验证器绑定的反馈接口”，并给出针对验证器、基模型、来源、构建层等维度的归因字段，帮助未来公开数据集和训练过程可检验、可比较。

**🔧 技术方法**

采用归因分类与指标框架，对验证器类型、反馈粒度、行为约束、数据源演化、奖励通道等进行系统梳理，并以表格和图示形式总结常见陷阱与未来报告建议。

**📊 数据集**

综述了公开的数据集与构建方法，包括DeepMath‑103K、DAPO、PRM800K、Math‑Shepherd、Skywork‑OR1、OpenThoughts、Structure Trumps Size等，涵盖程序验证、环境验证与判定验证三类。

**📈 对比分析**

比较方法基于归因字段的完整披露（如验证器版本、源混合、采样协议、推理预算等），并指出不同配置对准确性、效率与覆盖率的影响，表明性能提升往往源自数据质量、验证器严谨性和搜索拓扑，而非单纯规模或优化器改进。

**⚠️ 局限性**

局限性在于仅涵盖公开资料，缺乏对闭源流水线的分析，部分关键字段（如版本化、污染审计、奖励信号细节）仍未充分披露，且未对所有验证器进行独立重跑或污染检测。

---

## 607. Model Multiplicity and Predictive Arbitrariness in Recidivism Risk Assessment

**arXiv ID:** 2606.02198 | [PDF](https://arxiv.org/pdf/2606.02198v1)

**作者:** Ashwin Singh `[一作]` (TU Wien), Carlos Castillo `[通讯]` (ICREA and Universitat Pompeu Fabra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并改进了我国监狱常用的刑事再犯风险评估系统，首先构建了基于法律规则的再犯标签生成算法，随后利用可解释的整数线性模型（SLIM）和其公平性扩展（FairSLIM）训练了一系列具有低误差率差异的模型，并对模型集合中的预测多重性（Rashomon效应）进行了定量分析，提出了基于最低风险的决策策略（FairSLIM‑LRP）来缓解预测歧义。

**💡 创新点**

创新点包括：
1) 通过多轮专家校验实现了大规模、可信的再犯标签生成；
2) 将公平性约束（等化误差率）加入SLIM MILP，显著减少多组间误差差距；
3) 推导并应用了针对有限模型集的自一致性（self‑consistency）下界，验证了实际预测多重性远低于理论最坏情况；
4) 设计并验证了一种“最低风险”策略，该策略在保证公平性和可解释性的前提下，提升了整体准确率。

**🔧 技术方法**

采用的技术包括：
- MILP求解（Gurobi）实现SLIM与FairSLIM；
- 逻辑回归与CatBoost作为基准模型；
- 结构化的自一致性分析与下界推导；
- 迭代式标签生成算法（法律规则→人工校验→规则更新）。

**📊 数据集**

数据集：来自2010‑2019年共计17.5千起释放事件，包含43个评估特征（20个静态、23个动态），标签由自定义法律规则算法自动生成，并与专家标注一致。

**📈 对比分析**

与现行风险评估模型（RiskEval）和CatBoost进行对比。评估指标包括F1、准确率、平衡准确率、误差率差异（Δ_FPR、Δ_FNR、Δ_EO）。实验结果显示：
- FairSLIM在测试集上取得约70%平衡准确率，误差率差异≤5%；
- FairSLIM‑LRP在平衡准确率和误差率差异方面优于两者，且保持了与最佳FairSLIM相当的公平性。

**⚠️ 局限性**

局限性：
- 研究仅针对单一司法辖区的特定评估系统，结果可能不具普适性；
- 未针对周期性评估与模型更新的动态情形做进一步实验；
- 仅关注预测性能和公平性，未评估决策支持系统在实际使用中的效果；
- 采用的模型仅为整数线性模型，未探讨更广泛模型族对预测多重性的影响；
- 由于数据敏感性，公开数据不可用，限制了外部复现。

---

## 608. Conditional Graph Diffusion for Negotiation Support: Overcoming Discrete Infeasibility and Preference Elicitation Gaps

**arXiv ID:** 2606.02209 | [PDF](https://arxiv.org/pdf/2606.02209v1)

**作者:** Moirangthem Tiken Singh `[一作]` (Dibrugarh University), Moirangthem Tiken Singh `[通讯]` (Dibrugarh University)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5019862234)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Conditional Graph Diffusion (CGD) 框架，用于协作谈判支持系统，能够在连续效用空间中生成满足个体合理性、程序安全性、对等性和帕累托效率的推荐，并通过推理时的规范性引导梯度实现对这些约束的强制。

**💡 创新点**

创新点包括：
- 采用 GATv2 动态图注意力编码谈判双方的相对偏好结构；
- 通过交叉注意力将图结构和自然语言对话信息融合为统一的条件上下文；
- 将条件扩散模型与解析得到的规范性梯度结合，实现推理时无需重新训练即可强制满足多项规范；
- 采用连续效用空间避免离散分配空间中的不可行性；
- 引入福利引导机制，实现规范性遵从与福利最大化的操作解耦。

**🔧 技术方法**

核心技术包括：
- GATv2 图注意力网络（处理代理间相对价值）；
- 交叉注意力模块（融合对话与图特征）；
- Transformer 句子编码器（捕获对话隐式偏好信号）；
- U‑Net 结构的 Denoising Diffusion Probabilistic Model (DDPM) 生成效用向量；
- 推理时的规范性引导梯度（基于 IR、Security、Equity 约束）以及可选的福利引导项。

**📊 数据集**

实验使用三组数据集：
- Synthetic：从对称 Dirichlet 分布生成的 300 训练 / 60 验证样本；
- CaSiNo：1030 条人类露营地谈判对话（800 训练 / 200 验证）；
- Deal or No Deal (DND)：600 训练 / 120 验证的书、帽、球分配对话。

**📈 对比分析**

与基线（NBS、随机、贪婪、CGO、RNN‑SL/RL 等）对比，CGD 在 IR、Security Gap、Symmetry Gap 三项指标均满足阈值，IR 接近 100%，Security Gap ≤ 0.009，Symmetry Gap ≤ 0.15；在福利方面仅比 NBS 降低 1–3%（Synthetic/ CaSiNo）或 1%（DND）。此外，CGD 在推理时可通过调节 welfare 引导参数实现帕累托效率恢复。

**⚠️ 局限性**

局限性：
- 需要手动调节推理时的引导尺度 γ，才能保证 IR；
- 对话信息的有效利用受数据集信息量限制；在真实对话中，若对话不携带可辨别的偏好信号，CGD 与仅使用图信息的模型效果相近；
- 离散分配阶段仍需暴力搜索，虽然在连续空间中避免了不可行性，但最终的离散投影可能仍产生 IR 违规；
- 目前仅在零 BATNA 条件下评估，实际应用中可能需考虑更复杂的 BATNA 估计；
- 模型推理成本主要集中在 diffusion 步骤和后处理搜索，后者随着问题规模增长仍呈指数增长。

---

## 609. Context-Aware Workflow Decomposition for Automated Mobile UI Annotation Using Multimodal Large Language Models

**arXiv ID:** 2606.02208 | [PDF](https://arxiv.org/pdf/2606.02208v1)

**作者:** Athar Parvez `[一作]` (King Fahd University of Petroleum and Minerals), Omar Hammad `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 329 | [OpenAlex ID](https://openalex.org/A5075996232)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模态大型语言模型在移动 UI 注释中的工作流拆分，提升了注释精度。

**💡 创新点**

通过中度拆分工作流（两步）实现提示复杂度与上下文信息的平衡，并提出基于语义关联的类分组策略。

**🔧 技术方法**

使用多模态 LLM（Gemini 3.1 Pro）配合结构化提示、schema‑约束的 JSON 输出及自动化管线。

**📊 数据集**

使用 MUIAnno 数据集中的 8 种常见 UI 元素（button、tab、clickable text、card、label、plain text、icon、image）。

**📈 对比分析**

与一、二、四、八步工作流比较，发现两步工作流在精度上最高（≈0.43），召回最高的是八步但精度最低；整体 F1 约 0.40。

**⚠️ 局限性**

局限性包括仅测试 8 类元素、仅用单一 LLM、未针对屏幕复杂度进行自适应分组，且在高密度小图标场景下仍易产生误检。

---

## 610. EEG-FuseFormer: A Transformer-Driven Feature Fusion Framework for Seizure Onset Prediction

**arXiv ID:** 2606.02166 | [PDF](https://arxiv.org/pdf/2606.02166v1)

**作者:** Vigneshwar Hariharan `[一作]` (National University of Singapore), Ganesh Neelakanta Iyer `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于Transformer的特征融合框架EEG-FuseFormer，用于癫痫发作预测，融合CNN‑LSTM与ResNet‑18提取的时域与频域特征；

**💡 创新点**

创新点在于使用Transformer编码器实现自注意力多尺度特征融合，并结合跨患者自适应微调提升泛化性能，同时对运行时复杂度进行评估；

**🔧 技术方法**

采用的技术包括1D‑CNN‑LSTM、ResNet‑18、STFT时频变换、Transformer Encoder、全连接分类器、滑动窗口增强、随机下采样、目标适应微调以及多平台（GPU、Jetson）计算性能评测；

**📊 数据集**

使用CHB‑MIT癫痫EEG数据库，挑选13名患者的多通道记录进行实验；

**📈 对比分析**

通过患者特异性、交叉患者验证及自适应微调三种对照方法，EEG‑FuseFormer在患者特异性下平均召回率达98.85%、F1得分97.92%，跨患者自适应后召回率提高至93%，显著优于现有方法；同时在不同硬件平台上评估了运行时延迟和参数规模；

**⚠️ 局限性**

局限性包括模型结构复杂、训练与推理时间长，对低算力设备友好度有限；跨患者泛化仍受限，需要进一步提升适应性，并且对极少量患者数据的适应效果尚待验证。

---

## 611. Multilingual Idioms in Sentences and Conversations Across High-, Medium-, and Low-Resource Languages

**arXiv ID:** 2606.02147 | [PDF](https://arxiv.org/pdf/2606.02147v1)

**作者:** Saeed Almheiri `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个跨18种语言、覆盖高、中、低资源语料的多语言习语理解数据集MIDI，并在其上对多种LLM进行零样本多选评估，探究记忆与推理的关系。

**💡 创新点**

创新点在于：①将习语嵌入句子与对话两种语境，兼顾字面与比喻两种意义；②通过记忆/推理对照实验和激活 steering 方法揭示模型对习语的记忆-推理耦合；③提供了低资源语言习语的公开基准。

**🔧 技术方法**

使用的方法包括零样本多选评估、对比记忆与推理设置、激活 steering（ActAdd）与方向估计，以及对模型表现的对照实验。

**📊 数据集**

使用的数据集是自行构建的MIDI，包含2278条习语、8378个语境，涵盖高、中、低资源三类语言，且每条习语均有句子和对话两种形式、字面与比喻两种解释。

**📈 对比分析**

通过与人类评估对照，使用多选准确率评估不同模型（包括GPT‑5.2、Gemini、Gemma‑3、Llama‑3等），结果显示专有模型平均准确率约70–75%，开源模型约60–70%，低资源语种和字面意义的准确率显著下降；对话语境比句子语境提升约5%。

**⚠️ 局限性**

局限性包括：①语言覆盖不平衡且部分语言缺乏字面对应；②记忆与推理实验仅关注比喻意义，未涉及字面推理；③未报告注释者一致性；④对话由LLM生成后人工修订，缺乏自然对话多样性；⑤多选评估无法覆盖开放式解释。

---

## 612. TimeBlocks: Foundational and Continual Time-Series Blockbase -- Extended Version

**arXiv ID:** 2606.02142 | [PDF](https://arxiv.org/pdf/2606.02142v1)

**作者:** David Campos `[一作]` (Aalborg University), Christian S. Jensen `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用预训练时间序列处理块集合并路由器动态拼接的轻量级模型构建框架，支持多任务零样本推理和流式持续校准。

**💡 创新点**

创新点在于：①把大型模型拆解为可插拔的、标准化输出的小块并聚集为块池；②设计高效的路由器通过块指纹在推理时迭代选择最合适的块；③提出低成本近似核心子集（coreset）方法，用于持续校准而无需存储完整流。

**🔧 技术方法**

使用模块化小块（如MLP、LSTM、注意力）与标准化层、块指纹聚类、1D卷积+FC路由器、基于设施定位的近似核心子集算法。

**📊 数据集**

实验涵盖四类任务：预测（ETTh1/2、ETTm1/2、Weather）、缺失插补、异常检测（SMD、MSL、SMAP、SWaT）以及分类（UCR时序归档）。

**📈 对比分析**

与专用模型、基础时间序列模型及LLM模型对比，实验显示在预测、插补、异常检测和分类任务上均取得最优或相近的MSE/精度；模型尺寸最小、推理速度最快；持续校准时核心子集显著提升性能。

**⚠️ 局限性**

局限性包括：需要先在大规模数据集上预训练块池；块池规模和路由策略仍可能影响搜索效率；对极端稀有或异常模式的适应性未充分验证；框架目前仅支持预定义的块类型，扩展至更复杂模型仍需研究。

---

## 613. QEC and EAQEC Codes from Hermitian Sums and Hulls of Cyclic Codes over $\mathbb{F}_2 \times (\mathbb{F}_2+v\mathbb{F}_2)$

**arXiv ID:** 2606.02137 | [PDF](https://arxiv.org/pdf/2606.02137v1)

**作者:** Rabia Zengin `[一作]` (Istanbul Bilgi University), Mehmet Emin Köroğlu `[通讯]` (Yildiz Technical University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过分析复合环 F₂×(F₂+vF₂) 上循环码的 Hermitian 和-和与 Hull，构造了新的量子误差纠正码和纠错辅助量子误差纠正码。

**💡 创新点**

在传统欧几里得方法基础上引入 Hermitian 求和与 Hull 的结构，获得了更优的量子码参数，且与已知最佳码相比大多接近或达到量子 Singleton 界。

**🔧 技术方法**

主要技术包括 Gray 映射、Hermitian 双重码理论、矩阵乘积码以及量子构造 X。

**📊 数据集**

本文未使用公开数据集，而是通过理论构造与计算验证得到码参数。

**📈 对比分析**

与文献中已知的最佳量子码进行对比，实验结果显示多条码达到或逼近最佳性能，部分码为最优码。

**⚠️ 局限性**

研究仅局限于二进制场（p=2），对更大质数场的推广仍需进一步研究；此外，随着码长增大，计算复杂度显著增加。

---

## 614. Learning When Not to Act: Mitigating Tool Abuse in Agentic Reinforcement Learning

**arXiv ID:** 2606.02132 | [PDF](https://arxiv.org/pdf/2606.02132v1)

**作者:** Liuji Chen `[一作]` (Institute of Automation, Chinese Academy of Sciences), Liang Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一种新的 Agentic Reinforcement Learning 框架 EAPO，旨在降低大语言模型对外部工具的滥用，提高工具使用效率与任务准确性的平衡。

**💡 创新点**

创新点包括：① 在每轮采样中引入工具自由轨迹，让模型直接学习何时无需调用工具；② 根据任务难度自适应地对奖励进行塑形，仅在难度低时加大工具使用惩罚；③ 对每个生成 token 按置信度加权，提升低置信度 token 的学习力度，改善策略学习。

**🔧 技术方法**

采用了 Agentic RL、工具感知采样、难度感知奖励塑形、token 置信度估计、置信度加权重训练、KL 正则化等技术，结合大模型的策略梯度学习。

**📊 数据集**

使用了 9 个数学与知识推理基准（AIME 2024/2025、MATH500、MATH、GSM8K、HotpotQA、2WikiMultihopQA、Musique、Bamboogle），以及 10K 的混合 SFT 数据集作为预训练。

**📈 对比分析**

与 GRPO、ARPO、AEPO、Reinforce++ 等现有方法对比，评估指标为知识推理的 F1/EM、数学推理的 Pass@1 以及平均工具调用次数；EAPO 在 Qwen2.5‑3B、Qwen2.5‑7B、Llama3.1‑8B 上平均提升 10.45%、7.27%、9.69% 的准确率，同时平均减少 18.33%、18.33%、24.59% 的工具调用。

**⚠️ 局限性**

局限性：主要集中在目前基准侧重于难题，缺乏从易到难并同时评估工具滥用的专门数据集；对极大模型和多工具环境的泛化性尚未充分验证。

---

## 615. Equilibrated Diffusion: Frequency-aware Textual Embedding for Equilibrated Image Customization

**arXiv ID:** 2606.02129 | [PDF](https://arxiv.org/pdf/2606.02129v1)

**作者:** Liyuan Ma `[一作]` (Westlake University), Guo-Jun Qi `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Equilibrated Diffusion方法实现频率分离的文本嵌入与掩模引导扩散，从而提升图像定制的风格与文本一致性。

**💡 创新点**

通过频率分离的文本嵌入(FDTE)、掩模引导扩散(MGDP)和残差参考注意力(RRA)，实现内容与风格解耦并减少背景干扰。

**🔧 技术方法**

频域分解、可学习文本嵌入、掩模引导扩散、残差参考注意力、Stable Diffusion v1.4 训练框架。

**📊 数据集**

DreamBooth 30个主体（每个4–6张图片），包含无风格和22条风格化提示。

**📈 对比分析**

与Textual Inversion、DreamBooth、CustomDiffusion、ELITE等方法对比，在CLIP‑T、CLIP‑I、DINO‑I指标上均取得最高或第二高分，尤其在风格化文本对齐与主体一致性上表现突出。

**⚠️ 局限性**

对极端风格描述仍可能出现颜色偏差，且对多视角/遮挡的鲁棒性有待进一步提升。

---

## 616. Understanding-Enhanced Model Collaboration for Long-Tailed Egocentric Mistake Detection

**arXiv ID:** 2606.02120 | [PDF](https://arxiv.org/pdf/2606.02120v1)

**作者:** Boyu Han `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Qingming Huang `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种UE-MCM模型，使用小模型（DCR增强的CLIP4CLIP）和大模型（Qwen3-VL Embedding）分别从工作流层面和动作层面评估动作是否存在错误，并通过自适应门控融合两者输出。

**💡 创新点**

创新点在于：①将粗粒度视频与细粒度片段联合编码，提升对工作流一致性的判断；②采用Diffusion Contrastive Reconstruction提升CLIP的细节感知；③通过三种长尾优化目标（重加权交叉熵、AUC损失、标签感知调整）显著提升稀缺错误样本的召回与判别质量；④使用轻量级协作门实现动态平衡两分支的预测。

**🔧 技术方法**

核心技术包括CLIP4CLIP、Diffusion Contrastive Reconstruction、Qwen3-VL Embedding、轻量级协作门、重加权交叉熵、AUC损失、标签感知损失。

**📊 数据集**

在HoloAssist 2026误差检测挑战的 egocentric 视频数据集上进行实验，主要使用 RGB 模式。

**📈 对比分析**

与随机、TimeSformer、2024 2025 年的冠军方法对比，UE-MCM 在 F‑score、错误召回率和正确召回率均有显著提升，尤其在仅使用 RGB 的情况下已逼近甚至超越多模态模型的表现。

**⚠️ 局限性**

局限性：①依赖大规模预训练模型（Qwen3-VL）对资源和算力有一定要求；②长尾优化参数调优复杂；③在极稀缺错误类别上仍可能受限，且目前未针对实时推理做进一步加速。

---

## 617. Faster Synchronous On-Policy RL via Straggler-Aware Group Sizing

**arXiv ID:** 2606.02218 | [PDF](https://arxiv.org/pdf/2606.02218v1)

**作者:** Azal Ahmad Khan `[一作]` (University of Minnesota), Ali Anwar `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自适应组大小控制器 Straggler-Aware Group Control (SAGC)，通过在线调节同步强化学习训练中的批量规模来降低“慢跑”导致的同步停滞。

**💡 创新点**

创新点在于将组大小选择建模为受约束的在线优化问题，利用 Lagrangian 方法与贝塔先验结合的 Thompson Sampling，动态权衡大批量带来的统计优势与因长度方差产生的停滞风险。

**🔧 技术方法**

采用的技术包括：
- 在线主从双重更新（primal–dual）
- 贝塔先验与折衷因子估计“慢跑”概率
- 采样 Lagrangian 评分实现组大小的 Thompson Sampling
- 轻量级 CPU 控制器与 GPU 生成统计的低延迟通信。

**📊 数据集**

训练使用 Qwen2.5‑3B‑Instruct 与 Llama‑3.2‑3B‑Instruct 两大模型，数据集为统一的训练提示集；下游评测在 AIME 2023‑2025、AMC 2023 以及 GPQA Diamond 三个推理基准上进行。

**📈 对比分析**

对比方法包括最小化同步实现 Vanilla、系统优化后的 OpenRLHF 以及固定组大小的 baseline。实验显示 SAGC 在保持或提升奖励的同时，显著降低慢跑率、缩短壁钟时间，并在下游任务中获得与最优固定组大小相当或更好的准确率，同时往往生成更短的答案。

**⚠️ 局限性**

局限性在于控制器仅利用长度统计信息，未结合提示难度或长度预测，可能无法提前预知哪些查询易产生慢跑；此外不直接对答案长度施加显式惩罚，导致对极长生成的抑制依赖于间接效应。

---

## 618. PyFEX: Uncovering Evasive Python-based Threats via Resilient and Exhaustive Path Exploration

**arXiv ID:** 2606.02196 | [PDF](https://arxiv.org/pdf/2606.02196v1)

**作者:** Meng Wang `[一作]` (CISPA Helmholtz Center for Information Security), Ali Abbasi `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于 CPython 解释器的强制执行引擎，能够同时解析 Python 源码和编译字节码，并以此构建了一套完整的恶意包检测框架；

**💡 创新点**

在解释器层面实现全路径强制执行，配合弹性崩溃恢复、合成对象、语义保留路径合并与状态共享以及沉睡函数分析，极大提升了对环境检查、网络失效、编译型包和隐藏入口的可见性，并将丰富的执行轨迹交给 LLM 进行智能分类；

**🔧 技术方法**

主要技术包括 CPython 内部字节码拦截与路径分叉、Synthetic Object 合成与追踪、post‑dominators 路径合并与冲突解析、跨路径状态共享、沉睡函数自动触发、执行阈值控制、LLM（GPT‑4.1‑mini）行为推理与分类；

**📊 数据集**

使用三个数据集：D1‑Raw（10,920 已知恶意 PyPI 包）经校准去重得到 D1‑Calibrated（2,961 样本）；D2（417 真实世界 Python 可执行文件）；D3（90 天 PyPI 实时上传包）用于生产环境检测；

**📈 对比分析**

与 Malguard、EA4MP、GuardDog、Bandit4Mal、Package Analysis 等静态/混合工具对比，D1‑Calibrated 上 F1‑score 达 99.5%；对 D2 集合，其余工具均无法完成分析，且路径合并将平均进程数从 955 降至 44，单包平均耗时约 75 秒；实时部署发现未知恶意包并累计高下载量，误报主要源自 LLM 误判；

**⚠️ 局限性**

局限性包括：仍无法处理缺失解密密钥的加密负载；需要手动设定执行阈值防止无限循环；只覆盖 Python 层，无法直接分析 C/C++ 扩展；LLM 分类可能出现误报；沉睡函数分析在大型库中耗时高；对资源密集型环境的性能与可扩展性仍待进一步优化。

---

## 619. Rotatable Antenna-Enabled Satellite Communication: Joint Design of Boresight Alignment and Beam Tracking

**arXiv ID:** 2606.02193 | [PDF](https://arxiv.org/pdf/2606.02193v1)

**作者:** Tiantian Ma `[一作]` (South China University of Technology), Robert Schober `[通讯]` (Friedrich-Alexander-University Erlangen-Nurnberg)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于可旋转天线（RA）的低地球轨道（LEO）卫星通信框架，联合优化发射/接收波束成形和天线视轴方向以保持高速移动时的方向对准；

**💡 创新点**

创新点在于利用LoS信道的秩一结构，推导出闭式解实现波束成形与视轴角度的解耦优化；同时提出低复杂度的轨道可预测的通道估计与跟踪协议；

**🔧 技术方法**

采用了RA天线阵列、最大比率传输/接收（MRT/MRC）、MUSIC算法进行AoA估计，以及基于轨道参数的线性角度跟踪；

**📊 数据集**

使用仿真数据集：圆形LEO轨道（半径6.97×10^6 m），2×2 UPA阵列，频率2 GHz，λ=0.15 m，卫星与地面节点均为2×2 UPA；

**📈 对比分析**

与随机视轴、固定视轴、等向性天线基线进行对比，结果显示RA+跟踪方案在可观测的卫星通行期间几乎达到完美CSI上限，显著优于固定/随机视轴，提升可达率；

**⚠️ 局限性**

局限性包括：仅考虑单一GN场景，假设纯LoS且轨道已知，未考虑多用户干扰、硬件实现细节、功耗与尺寸等实际部署问题。

---

## 620. Unicity: Predicates and Atomic Swaps

**arXiv ID:** 2606.02192 | [PDF](https://arxiv.org/pdf/2606.02192v1)

**作者:** Ahto Buldas `[一作]` (Tallinn University of Technology), Ahto Truu `[通讯]` (Unicity Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在Unicity基础上将令牌所有权与转移泛化为可编程谓词，实现了类似智能合约的离线执行；

**💡 创新点**

提出谓词化所有权模型并证明其安全属性与原有Unicity层相容，同时设计了基于谓词的无信任原子交换协议；

**🔧 技术方法**

采用数字签名、哈希函数、可隐藏承诺以及谓词验证机制，并通过安全性归约论证；

**📊 数据集**

未使用实际数据集，全部为理论构造与形式化证明；

**📈 对比分析**

主要通过理论证明对比，未给出实验性能数据，强调在安全性和可扩展性方面的优势；

**⚠️ 局限性**

依赖于外部可信的Unicity服务、谓词的正确选择及其实现复杂度高，且未讨论恶意谓词导致的阻塞风险。

---

## 621. Efficiently Listing Projected Trees, and Equivalence of Listing and Enumeration

**arXiv ID:** 2606.02183 | [PDF](https://arxiv.org/pdf/2606.02183v1)

**作者:** Karl Bringmann `[一作]` (ETH Zurich), Yanheng Wang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了针对投影树（以及更一般的投影超图）进行列表和枚举的首个多项式预处理时间（约 n^17.42）与多项式延迟（1）算法，并给出了从列表到枚举的通用转换方法；

**💡 创新点**

核心创新是将投影树问题转化为矩阵乘法问题，利用快速矩形与输出敏感矩阵乘法实现极低延迟，并证明任何突破传统时间界限的算法必需依赖快速矩阵乘法；另外首次证明了列表与枚举在自然假设下可互相转化的等价性；

**🔧 技术方法**

使用快速矩形矩阵乘法（Coppersmith-Winograd等技术）、输出敏感矩阵乘法、矩阵乘法的阻塞技巧、Shearer 余子集平衡、哈希分区（color‑coding）等组合技术；

**📊 数据集**

本研究主要为理论算法设计，未在具体数据集上实验；所有结果均为理论复杂度分析；

**📈 对比分析**

与之前基于粗暴 O(n^k) 或 O(m^k-1) 的列表/枚举算法相比，新的算法在投影树上实现了固定参数可解的预处理时间（从 n^k 降至 n^17.42 或在 ω=2 时降至 n^3），并在超图上得到 O(m^(H)) 的预处理时间；在所有已知实例上，算法显著优于传统方法；

**⚠️ 局限性**

主要局限：预处理时间仍为超线性（n^17.42），需要快速矩阵乘法（目前尚未实现高效实用实现）；延迟仅为多项式对数级，无法得到常数延迟；对更一般图形的上界尚未给出；并且在没有快速矩阵乘法的情况下无法突破传统上界，因而理论上存在不可优化的下界。

---

## 622. The Unicity Execution Layer

**arXiv ID:** 2606.02181 | [PDF](https://arxiv.org/pdf/2606.02181v1)

**作者:** Ahto Buldas `[一作]` (Tallinn University of Technology), Ahto Truu `[通讯]` (Unicity Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 Unicity 框架中的 Unicity Execution Layer，构建了一种只在链上执行双花防护、其余交易与合约逻辑离链执行的去中心化系统，并给出了完整的安全模型与证明；同时引入多公钥签名、隐蔽承诺、稀疏 Merkle 树等技术实现用户与服务端的隐私保护与交易无关联。

**💡 创新点**

核心创新点是将传统区块链的大部分工作迁移至链下，只保留最小化的链上双花防护，从而实现线性可扩展性；提出了可链式、多公钥签名的隐私方案，并在理论上证明了双花、阻塞与服务端隐私等安全属性；在此基础上进一步设计了时序隐私与自定义交易协议（包括 Masked 与 VRF 变体）。

**🔧 技术方法**

使用的技术包括：数字签名方案（ECDSA、BLS 等）、可绑定与可隐藏的承诺方案、哈希函数（可作为一次性哈希）、可扩展的稀疏 Merkle 树作为累加器、一次性预映像难度哈希（k,ℓ)-one‑way、可随机数生成、可交互式与非交互式 VRF（用于 Masked 协议）等。整个系统基于严格的安全模型和证明。

**📊 数据集**

该论文为理论/设计性工作，没有使用公开数据集；所有安全性与可扩展性分析均基于形式化模型与抽象实验。

**📈 对比分析**

在方法比较与性能方面，论文主要通过理论证明与复杂度分析来展示相对于传统链上执行或 Layer‑2 解决方案的线性可扩展性；并在安全性上与现有双花防护方案进行对比，证明其在阻塞与双花攻击下具有更强的安全保障；但未给出实际部署或基准测试结果，性能评估仅停留在理论层面。

**⚠️ 局限性**

局限性包括：需要可信的 Unicity Service（聚合层）来维护全局注册表；隐私保护依赖于承诺方案的完美隐藏、哈希函数的可预映像难度等强假设；对服务端的可扩展性和可用性仍未做实证验证；在多方协作与跨链互操作性方面仍存在未解决的问题。

---

## 623. CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning

**arXiv ID:** 2606.02170 | [PDF](https://arxiv.org/pdf/2606.02170v1)

**作者:** Chengtao Gan `[一作]` (Zhejiang University), Wen Zhang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个名为CRAFTQA的代码驱动统一结构化数据问答框架，包含CodeSTEP和CRAFT两个模块，实现从问题到可执行Python代码的自动生成与执行。

**💡 创新点**

创新点在于动态生成自定义函数（超出预定义函数集合），通过CodeSTEP逐步生成代码与CRAFT互相配合，突破传统统一方法对预定义函数的限制，显著提升复杂推理能力。

**🔧 技术方法**

技术手段包括：利用大型语言模型生成可执行代码、构建条件图（Condition Graph）进行统一查询、使用Sentence‑BERT进行实体对齐、实现基于LLM的动态自定义函数合成以及代码可执行性验证。

**📊 数据集**

实验使用了六个数据集：WikiSQL、TableBench（Fact‑Checking 与 Numerical‑Reasoning）、WikiSQL‑E（自构造的包含“out‑of‑predefined”操作的数据）、WebQSP、CronQuestions，并在这些数据上评估模型性能。

**📈 对比分析**

与PoT、StructGPT、Readi、TrustUQA等先进统一方法对比，CRAFTQA在复杂推理任务上获得最高的Denotation Accuracy（例如GPT‑4o下WikiSQL‑E达85.6%），在标准任务上保持或略优的表现，并且在开源LLM上甚至能超过部分闭源大模型，证明了框架的强大效果。

**⚠️ 局限性**

局限性包括：在简单标准推理任务中提升有限；模型对代码生成能力要求高，若LLM编程能力弱则效果受限。

---

## 624. Multimodal Approaches for Visually-Rich Document Type Classification: A Comparative Analysis

**arXiv ID:** 2606.02162 | [PDF](https://arxiv.org/pdf/2606.02162v1)

**作者:** Catyana Heyne `[一作]` (OTH Regensburg), Filippo Riccio `[通讯]` (OTH Regensburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在统一实验框架下，对视觉丰富文档的文档类型分类进行了系统评估，比较了四种代表模型（LayoutLMv3、Donut、Qwen3‑VL‑32B‑Instruct、Qwen3‑32B）的表现，并细致分析了文本、图像与布局信息对分类结果的贡献。

**💡 创新点**

创新点在于提出了一套结构化的多模态设计分析方法，并通过统一实验设置实现了Transformer与LLM架构的可比性，揭示了图像信息对分类最关键、OCR文本为辅助作用，并强调多模态处理在布局结构明显文档中的必要性。

**🔧 技术方法**

采用的技术包括：基于Transformer的多模态模型（LayoutLMv3、Donut）和大型语言模型（Qwen3‑VL‑32B‑Instruct、Qwen3‑32B），多模态特征融合策略，OCR-free与OCR-dependent的对比实验，以及统一的评估脚本和指标。

**📊 数据集**

使用的数据集为RVL‑CDIP视觉丰富文档分类基准，包含多种文档类型，具备文本、图像和布局信息。

**📈 对比分析**

通过在同一实验环境下对四个模型在RVL‑CDIP上的准确率进行直接对比，发现专用多模态Transformer（LayoutLMv3、Donut）显著优于LLM（Qwen3‑VL‑32B‑Instruct、Qwen3‑32B），其中图像特征对最终准确率贡献最大，OCR文本为次要辅助。

**⚠️ 局限性**

局限性包括：仅聚焦文档类型分类任务，未扩展到其他多模态理解任务；实验仅在RVL‑CDIP上进行，可能缺乏跨数据集的泛化；模型规模差异及资源消耗不一导致对比不完全公平；对布局特征的细粒度贡献分析尚不充分。

---

## 625. PeAR: A Static Binary Rewriting Framework for Binary-Only Fuzzing

**arXiv ID:** 2606.02126 | [PDF](https://arxiv.org/pdf/2606.02126v1)

**作者:** Alvin Charles `[一作]` (Australian National University), Alwen Tiu `[通讯]` (Australian National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出并实现了一个名为 <framework> 的可扩展的二进制仅字节覆盖灰盒模糊测试框架，能够在没有源代码的情况下使用静态二进制重写实现高精度覆盖追踪，并支持延迟初始化、持久模式和共享内存模糊等现代模糊技术。

**💡 创新点**

创新点在于：1）利用 GTIRB 等现代二进制中间表示实现高精度且可扩展的静态二进制重写；2）首次在单一 SBI 框架中完整实现并验证了延迟初始化、持久模式和共享内存模糊等高级技术；3）通过对 25 个真实目标的广泛实验，展示了 SBI 在覆盖率与性能上的可与 DBI 甚至编译器级别的竞争力。

**🔧 技术方法**

核心技术包括：基于 GTIRB 的静态重写与分析、使用 Datalog 进行精确反汇编、插桩 Trampoline 与 Coverage Map、Forkserver 与持久循环包装、共享内存 Hook、以及针对 Windows PE 的初步支持。

**📊 数据集**

实验使用了 25 个来自公开基准（包括 LAVA‑M、LibFuzzer 等）和 20 个可被重写的目标，覆盖了常见的 ELF 与 PE 二进制。

**📈 对比分析**

与 E9AFL、E9、Trampoline 等现有 SBI 框架以及 QEMU 动态重写（DBI）和编译器插桩基线对比，<framework> 在成功重写率 88% 以上、覆盖率相当于 QEMU（87.84%）且比其他 SBI 低 6.62% 以上，持久模式与共享内存模式下通过中位数提升 4.07 倍的吞吐量，从而进一步提高了覆盖率。

**⚠️ 局限性**

局限性包括：目前仅支持 Linux ELF 和有限的 Windows PE（尚未完整实现），对多体系结构的支持仍在开发；部分复杂目标仍需手动提示才能正确重写；以及在极端异常控制流或自定义指令集场景下的鲁棒性待进一步验证。

---

## 626. Statistically Robust Resource Block Allocation for Satellite Communications

**arXiv ID:** 2606.02124 | [PDF](https://arxiv.org/pdf/2606.02124v1)

**作者:** Chaitanya Manapragada `[一作]`, Philippe Martins `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种卫星覆盖区资源块（RB）容量预估规则，基于随机用户分布、关联高斯衰减场和容量需求映射，并给出了Monte‑Carlo和浓度不等式两种互补的估计方法。

**💡 创新点**

创新点在于：①首次把空间衰减的协方差结构直接纳入卫星容量维度化，避免了传统单点衰减模型忽视的聚集性；②结合Monte‑Carlo模拟与Poisson功能浓度上界，提供了可靠的高置信度预算与保守性检验；③通过不同衰减均值与相关长度的敏感性分析，揭示空间相关性对RB预算的显著影响。

**🔧 技术方法**

主要技术包括：随机高斯场模型（RBF核协方差）、Poisson点过程生成用户位置、Shannon-赫特利率与RB需求映射、取整与用户上限截断、Monte‑Carlo采样、以及针对Poisson聚合的浓度不等式（利用均值与方差估计）。

**📊 数据集**

使用的“数据集”为仿真生成的随机场和用户分布：覆盖区为半径20 km的圆盘，用户按密度λ=10⁻⁵ 用户/㎡的泊松分布，衰减场参数（均值m∈{-1.5,-0.5,-0.25}，方差σ²=0.2，相关长度ℓ∈{500 m, 20 km}）共六种情景；每种情景下进行T=100次Monte‑Carlo试验。

**📈 对比分析**

比较方法：将Monte‑Carlo得到的99%分位数RB预算与平均浓度上界的RB预算对比。结果显示，短程相关下两种预算差距≤1%，但在覆盖尺度相关时，Monte‑Carlo预算显著提高（约14%–52%），浓度上界更为保守，单个场样本的预算波动可达±20%。在实际卫星容量（17 280 RB）限制下，所有情景的Monte‑Carlo预算均不满足，表明在给定密度与速率下单颗卫星不可行。

**⚠️ 局限性**

局限性包括：①仅考虑单个覆盖区，未讨论多轨卫星联合或多束配置；②假设用户需求为固定目标速率，未加入动态流量模型；③高斯衰减模型与真实气象衰减差异可能较大；④浓度不等式在覆盖尺度相关时过于保守，可能导致资源浪费；⑤仿真规模有限（T=100），置信区间粗糙。

---

## 627. POIROT: Interrogating Agents for Failure Detection in Multi-Agent Systems

**arXiv ID:** 2606.02282 | [PDF](https://arxiv.org/pdf/2606.02282v1)

**作者:** Iñaki Dellibarda Varela `[一作]` (Spanish National Research Council), Manuel Cebrian `[通讯]` (Spanish National Research Council)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 POIROT 协议，通过让多智能体系统内部代理相互询问和投票，实现系统自身的失效检测与故障归因。

**💡 创新点**

其创新点在于将评估去中心化：利用结构化危害空间和距离加权投票，避免单一评审者的偏见与上下文饱和问题。

**🔧 技术方法**

技术实现包含五阶段协议（自评、同行质询、私有投票、危害感知聚合）以及 LangGraph、LLM 和自定义工具链。

**📊 数据集**

实验使用 Who&When 基准以及作者自建的 BLAME 数据集（CORTEX 及 TradingAgents 的错误向量）。

**📈 对比分析**

与单一 LLM 评估者相比，POIROT 在四个模型上均表现更佳；在问题复杂度、代理数和故障维度较高时提升 12–46%，多故障情形下提升 6–25%。

**⚠️ 局限性**

局限性包括对模型推理能力的依赖（如 GPT‑OSS 120B 在多故障时略逊）以及需要人工定义危害空间的步骤。

---

## 628. Matter to Mechanism: A Benchmark for AI Co-Scientists in Materials and Battery Research

**arXiv ID:** 2606.02258 | [PDF](https://arxiv.org/pdf/2606.02258v1)

**作者:** Shashwat Sourav `[一作]` (Washington University in St. Louis), Tirthankar Ghosal `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6215c339-3735-4be3-8a07-5bbb7004712d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 Matter to Mechanism benchmark，专门评估 AI 合作科学家在材料科学（尤其是电池研究）中从具体问题推导机制驱动假设的能力。

**💡 创新点**

创新点在于：① 用结构化的“问题→假设”实例与显式推理链；② 设计了六维度（推理链、问题对齐、机制细节、创新性、可行性、问题拆解）参考无关评估；③ 通过对抗性压力测试和 LLM 判断验证指标鲁棒性。

**🔧 技术方法**

技术包括：自动化文本抽取与结构化、无参考评估指标、加权综合分数、对抗性攻击模拟、基于 Gemini 的对比评估。

**📊 数据集**

使用了约 90,000 篇公开论文，筛选后得到 2,645 条电池相关问题‑假设实例，形成评测数据集。

**📈 对比分析**

对比了 ChemDFM‑8B、Gemini‑Direct/Weak/Retrieval、AI‑Researcher、Open Co‑Scientist 与文献基准。结果显示文献基准最高；在六维度中，推理链强但机制细节弱的模型占多；ChemDFM‑8B 在整体分数和机制专属性上领先，表明域专用模型更具优势。

**⚠️ 局限性**

局限包括：仅评估假设生成与机制推理，未覆盖实验设计与工具使用；抽取过程可能引入噪声；LLM 判断一致性有限；数据集中大部分实例缺乏稀有机制术语，难以检验记忆化风险。

---

## 629. A Doeblin-Anchored Contrastive Chart for Learning Markov Transition Kernels

**arXiv ID:** 2606.02232 | [PDF](https://arxiv.org/pdf/2606.02232v1)

**作者:** Ao Xu `[一作]` `[通讯]` (Jilin University), Ao Xu (Jilin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Doeblin锚定对比图，用于将对比风险映射为可合法的马尔可夫转移核，并给出从估计到动态误差的完整理论与实验验证。

**💡 创新点**

创新点在于：①将Doeblin小化、对比实验正类和可逆坐标统一于一个锚定图；②设计可恢复合法核的Markov化操作；③给出有限时空动态误差传递与覆盖条件的诊断。

**🔧 技术方法**

采用对比噪声学习、Doeblin小化、贝塔混合、神经网络近似、凸优化、覆盖数与泛化不等式、稀疏/异常状态诊断等技术。

**📊 数据集**

实验使用仿真生成的连续一维、多模态、有限状态链及几何β混合轨迹，所有数据均为合成，方便精确评估。

**📈 对比分析**

通过对比风险、L²密度误差、TV误差、Markov化前后负质量、滚动误差等指标，与无锚定、无Markov化、基线高斯CDE 进行消融实验，结果显示锚定+Markov化能恢复核合法并在有限时空误差上保持相对优越性能。

**⚠️ 局限性**

局限在于：需要独立转移对；全局密度有界假设；必须可模拟且足够支持的锚定律；有限时空覆盖条件；未对高维或长期轨迹的误差扩散给出完整理论。

---

## 630. Cross-Domain Dead Tree Detection via Knowledge Distillation in Aerial Imagery

**arXiv ID:** 2606.02303 | [PDF](https://arxiv.org/pdf/2606.02303v1)

**作者:** Anis Ur Rahman `[一作]` (CSC IT Center for Science Ltd.), Samuli Junttila `[通讯]` (University of Helsinki)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过知识蒸馏技术，将在芬兰航空影像上训练的教师模型迁移至波兰、德国、爱沙尼亚等目标域，完成跨域死树检测任务。

**💡 创新点**

提出四种KD变体并创新性地加入特征层对齐的Feature‑level KD，在跨域泛化和少样本场景下显著提升精度、鲁棒性和误检率。

**🔧 技术方法**

使用基于ResNet‑34编码器与自注意力解码器的U‑Net改版（Tree Mortality 1‑Task），配合温度化软标签、EMA、特征对齐、多任务混合损失与丰富的数据增强。

**📊 数据集**

采用四个高分辨率航空影像数据集：芬兰（源域）、波兰（目标域）、德国、爱沙尼亚（仅评估），每个数据集提供RGB‑NIR通道并手工标注树木多边形及热图。

**📈 对比分析**

与传统微调基线对比，使用Mean Tree IoU、Instance F1、Precision、Recall、Mean Centroid Error等指标；Feature‑level KD在波兰测试集上获得Mean Tree IoU 0.106、F1 0.63、Precision 0.55，误检率明显下降，在德国和爱沙尼亚等其他域亦保持高Precision，并在低数据情况下保持竞争性表现。

**⚠️ 局限性**

研究仅覆盖北欧与中欧森林，未验证在热带或干旱区域的适用性；学生模型与教师采用同一架构，未实现压缩或轻量化；仅使用RGB‑NIR光谱，未探索更高维光谱或自动标注方法。

---

## 631. Cross-modal linkage risk in clinical vision-language models

**arXiv ID:** 2606.02276 | [PDF](https://arxiv.org/pdf/2606.02276v1)

**作者:** Soroosh Tayebi Arasteh `[一作]` (RWTH Aachen University), Daniel Truhn `[通讯]` (RWTH Aachen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了临床视觉-语言模型中图像与报告共享嵌入空间的跨模态链接风险，并提供了评估与减轻该风险的完整流程。

**💡 创新点**

创新点在于把跨模态链接风险量化为图像到报告检索任务，证明专业化模型风险随医学领域深化而上升，并提出仅对对齐层进行差分隐私微调（DP‑Finetune）的低成本缓解方案，能够显著削弱链接同时几乎不损失图像诊断性能。

**🔧 技术方法**

使用了四种公开视觉-语言模型（CLIP、PubMedCLIP、BiomedCLIP、BioViL‑T），对齐层微调采用DP‑SGD，评估则采用线性探测器在14‑标签胸片异常检测任务上的AUROC、准确率等指标。

**📊 数据集**

数据集包括公开胸片报告配对数据MIMIC‑CXR（43 793对）和CheXpert Plus（29 296对），共计406 241例。

**📈 对比分析**

通过Recall@1、Recall@5/10、MRR等指标与随机基线和硬负样本对照进行评估，专业模型Recall@1可达15%（相对随机1%提升约15×），DP微调后Recall@1下降61%至0.19%（N=10 000），而图像诊断宏观AUROC仅下降0.20个百分点。

**⚠️ 局限性**

局限性包括仅评估胸片域、仅在单一模型上实验、硬负样本仅基于14标签标签空间、仅使用线性探测器评估诊断效能、以及未在真实脱敏部署环境中验证攻击效果。

---

## 632. Guided Sensemaking: Agents in Collaborative Deliberation

**arXiv ID:** 2606.02260 | [PDF](https://arxiv.org/pdf/2606.02260v1)

**作者:** Aaditya Bhatia `[一作]` (United States Military Academy), Jack Park `[通讯]` (TopicQuests)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并概念化了 Guided Sensemaking——一个多代理、AI 辅助的协同推理平台，帮助用户在文本写作过程中通过 Socratic 提问、个人论证图和集体论证图实现结构化思维与批判性推理。

**💡 创新点**

创新点在于：① 将多代理协同工作与论证图可视化相结合，既保留人类作者主动性，又提供持续的认知支撑；② 采用 Socratic 机制作为轻量化推理脚手架；③ 在用户提交后自动构建可追溯的集体论证图，维持多样视角而不合并成单一叙事。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）进行论证挖掘与自然语言理解；自动化图谱构建与更新；Socratic 提问生成；图形可视化与交互展示；多代理架构设计。

**📊 数据集**

本文为概念设计与原型实现，未使用公开数据集；主要通过人工编写的示例和教师/学生情境案例进行演示。

**📈 对比分析**

暂无量化比较与性能指标；文章未进行实验评估，仅通过示例展示功能实现与潜在效益。

**⚠️ 局限性**

局限性包括：① 缺乏实证验证与用户研究，无法量化对学习效果或批判性思维的提升；② 规模化应用时的算力与实时性挑战；③ 对偏见、误导信息的处理机制尚不成熟；④ 依赖大型语言模型，可能受训练数据偏差影响。

---

## 633. EES-CND: Collaborative Neural Decision-Making for Drift-Aware Fault-Tolerant Edge-Cloud Service Placement

**arXiv ID:** 2606.02259 | [PDF](https://arxiv.org/pdf/2606.02259v1)

**作者:** Mohammadsadeq Garshasbi Herabad `[一作]` (Karlstad University), Calin Curescu `[通讯]` (Ericsson AB)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于协同神经决策与增强进化策略的容错服务放置方法（EES-CND），可在动态漂移的边缘云环境中实现自适应恢复。

**💡 创新点**

创新点在于将预训练神经网络与在线自适应演化相结合，形成协作决策机制，能够实时跟踪系统漂移并保持低成本、低停机时间。

**🔧 技术方法**

采用协同神经网络（多模型）、增强进化策略（EES）、遗传算法预训练、心跳监控与在线采样评估等技术。

**📊 数据集**

使用基于远程维护场景的仿真器，生成小/中/大规模实例，并加入多层漂移，以验证方法的泛化与鲁棒性。

**📈 对比分析**

与BP-CND、ES-CND、单模型CND比较，EES-CND在成本降低44.8%，恢复时间、响应时间和可靠性均优于对照组，且单次失败间的计算时间不超过1秒。

**⚠️ 局限性**

局限性包括：需额外的计算与存储资源以维持多模型协作；在极大规模或漂移率极高的场景下性能可能下降；未在真实生产环境中进行验证。

---

## 634. Network Learning with Semi-relaxed Gromov-Wasserstein

**arXiv ID:** 2606.02223 | [PDF](https://arxiv.org/pdf/2606.02223v1)

**作者:** Charles Dufour `[一作]`, Leonardo V. Santoro `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了通过半松弛Gromov–Wasserstein barycenter对大型无标签网络的生成机制进行估计，实现了从单一邻接矩阵恢复低维块模型的目标。

**💡 创新点**

核心创新是将硬节点分配放松为概率耦合，证明了松弛解与最优硬分配的误差为O(1/n)，并给出了最优收敛率；同时提供了可扩展的GPU加速算法。

**🔧 技术方法**

采用半松弛GW目标、Frank‑Wolfe条件梯度求解器和块坐标更新技术，以求解大规模网络的低维块结构。

**📊 数据集**

在合成SBM、平滑Hölder图形以及真实EEG（64电极）和OpenFlight机场（3257机场）网络上进行实验。

**📈 对比分析**

与现有SBM/graphon基线（Largest‑Gap、Neighbourhood Smoothing、Sort‑and‑Smooth、USVT等）比较，实验表明在SBM设置下显著优于竞争者，在平滑图形上表现相当，且计算效率高。

**⚠️ 局限性**

局限在于缺乏对块数的显式正则化、对稀疏性分析不足，以及对非二值或带属性图的理论推广仍待进一步研究。

---

## 635. SeClaw: Spec-Driven Security Task Synthesis for Evaluating Autonomous Agents

**arXiv ID:** 2606.02302 | [PDF](https://arxiv.org/pdf/2606.02302v1)

**作者:** Hao Cheng `[一作]` (Hong Kong University of Science and Technology), Chao Shen `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SeClaw 框架，结合基于风险规范的任务合成与 Docker 沙箱下的执行轨迹评估，系统化构造并测评自主 LLM 代理的安全性。

**💡 创新点**

创新点包括：① 规范驱动的安全任务合成（多代理协作生成任务原型并迭代校验）；② 通过 ToolHub 自动构建任务级工具与环境；③ 记录完整执行轨迹，支持过程级别的安全评估与归因；④ 统一的安全评估指标（覆盖度、成功率、F_attack）。

**🔧 技术方法**

技术手段：多代理协作任务合成、ToolHub 工具索引与配置、Docker 沙箱执行、轨迹日志化（模型、工具、文件视图）、基于规则的安全评估器。

**📊 数据集**

使用结构化风险分类（资源、任务、环境、内部风险）生成的任务库；尚未公开标准公开数据集，任务由框架自动生成并在 Docker 中重现。

**📈 对比分析**

比较方法：对每个安全目标计算覆盖率 C、样本成功率 P，并通过调和平均 F_attack 评估安全弱点；预期对 Qwen、Kimi、GPT、Gemini 等模型进行基准测试，较低 F_attack 表示更稳健，但本文未给出实验结果。

**⚠️ 局限性**

局限性：① 任务生成与评估高度依赖规范质量，易受人工作业偏差；② 目前缺乏大规模真实数据验证，评估覆盖面仍有限；③ 对不同代理架构的迁移性与可扩展性尚未充分验证；④ 评价指标侧重安全触发，未覆盖模型的完整性能。

---

## 636. A Kinetic Theory of Encounter-Based Information Propagation in Multi-Robot Systems

**arXiv ID:** 2606.02296 | [PDF](https://arxiv.org/pdf/2606.02296v1)

**作者:** Alkesh K. Srivastava `[一作]` (Temple University), Philip Dames `[通讯]` (Temple University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究多机器人在仅靠物理接触通信时的信息传播机制并提出动力学理论。

**💡 创新点**

首次把信息传播视为碰撞式动力学，建立访问、陈旧和几何三种极限模型。

**🔧 技术方法**

采用碰撞率推导、Age‑of‑Information分析、仿真验证和回归预测。

**📊 数据集**

使用多机器人仿真数据，参数覆盖团队规模、环境尺寸、通信半径和目标速度。

**📈 对比分析**

通过交叉验证和保留参数验证，发现联合访问‑陈旧坐标能以R²≈0.76预测跟踪误差。

**⚠️ 局限性**

仅限仿真，单目标、均质机器人、简单感知与随机行走，未考虑实际硬件限制。

---

## 637. CityTrajBench: A Unified Benchmark for City-Scale Vehicle Trajectory Generation

**arXiv ID:** 2606.02287 | [PDF](https://arxiv.org/pdf/2606.02287v1)

**作者:** Shibo Zhu `[一作]` (Hong Kong Polytechnic University), Jinyue Yan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CityTrajBench，统一城市级车辆轨迹生成的基准框架和评估协议

**💡 创新点**

对数据预处理、表示统一、模型适配、地图投影、评估多维度等做系统化标准化，消除不同研究间的实验差异

**🔧 技术方法**

融合统计模型、VAE、GAN、扩散模型和流匹配模型，并采用JSD、热点匹配、DTW、Frechet、条件OD误差等多种指标

**📊 数据集**

三大真实轨迹数据集：Porto、Chengdu（出租车）和Shanghai（电动车）

**📈 对比分析**

在统一训练/验证/测试划分、固定长度序列、统一坐标归一化、统一后处理下，对七种模型进行公平比较；结果显示扩散与流匹配模型在不同维度表现最佳，DiffTraj 最优于轨迹级几何，DiffRNTraj 最佳于宏观空间结构，TrajFlow 兼顾质量与效率，Markov 在端点统计上仍具竞争力

**⚠️ 局限性**

局限包括：仅覆盖车辆轨迹，未考虑行人/骑行/多模态；固定长度序列可能失真；后处理投影掩盖原始生成质量；对时间维度、隐私、鲁棒性等方面评估不足，且控制生成能力有限

---

## 638. Physics-Guided Recurrent State-Space Neural Networks for Multi-Step Prediction

**arXiv ID:** 2606.02278 | [PDF](https://arxiv.org/pdf/2606.02278v1)

**作者:** Ruiyuan Li `[一作]` (TU Delft), Manon Kok `[通讯]` (TU Delft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于物理引导的递归状态空间神经网络PG-RSSNN，用于多步预测。

**💡 创新点**

核心创新在于使用RNN结构而非直接反馈状态，并采用ReLU激活，解决梯度消失和训练不稳定问题。

**🔧 技术方法**

技术包括GRU层、特征提取与输出层的两层全连接网络、ReLU激活、Adam优化器。

**📊 数据集**

使用四个数据集：线性高斯状态空间模型、真实与模拟的KUKA机器人臂以及级联水箱系统。

**📈 对比分析**

与九种对比模型（黑盒RSSNN、物理引导模型等）相比，PG-RSSNN在大多数数据集上取得最低RMSE/NRMSE，并且在训练不稳定性上表现更稳健，尤其在小样本场景下仍优于其他方法。

**⚠️ 局限性**

局限性在于对物理模型的依赖程度仍高，且实验主要集中在离散时间系统，未来需扩展到连续时间及更复杂动力学。

---

## 639. RoboSemanticBench: Diagnosing Semantic Grounding in Action Prediction for VLA Models

**arXiv ID:** 2606.02277 | [PDF](https://arxiv.org/pdf/2606.02277v1)

**作者:** Bin Yu `[一作]` (Harbin Institute Of Technology), Kai Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RoboSemanticBench (RSB) 基准，通过让机器人在执行多选数学或常识问题的答案选择任务中检验 VLA 模型是否能将语义理解转化为正确的动作。

**💡 创新点**

创新点在于将语义理解任务与物理执行解耦，设计 GSR、TSR 与 nSG 三个指标分离抓取能力与语义目标选择，形成一种针对语义 grounding 的诊断工具。

**🔧 技术方法**

使用预训练视觉-语言背骨（如 Qwen3-VL、OpenVLA、π_0 等）与动作专家，结合脚本专家演示、物理仿真（Aloha-AgileX + MPLib）及 CoT 生成与 VQA 监督等技术进行模型训练与评估。

**📊 数据集**

使用 RSB 自己构造的数据集，包含三类子集：RSB-Math（程序生成算术题）、RSB-HardMath（来自 GSM8K 的文字题）以及 RSB-General（MMLU 风格常识题），并在 4 选与 10 选两种设置下进行实验。

**📈 对比分析**

采用统一的训练–测试协议，对 10 款代表性 VLA 模型在 6 个 RSB 套件上进行评估。实验结果显示大多数模型的 TSR 与 nSG 接近或低于随机水平，仅 π_0.5 在某些设置表现稍好，表明当前 VLA 训练在语义 grounding 上表现不佳。

**⚠️ 局限性**

局限性在于即便加入 CoT 语义中间输出或 VQA 辅助监督，模型仍无法显著提升语义 grounding；现有 VLA 训练往往依赖视觉或动作快捷方式，缺乏将语义决策稳定传递给动作专家的机制。

---

## 640. Vision-language Models for Driver Monitoring Systems: A Driver Activity Description Dataset

**arXiv ID:** 2606.02273 | [PDF](https://arxiv.org/pdf/2606.02273v1)

**作者:** David J. Lerch `[一作]` (Fraunhofer IOSB), Rainer Stiefelhagen `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了基于Drive&Act的细粒度自然语言描述数据集，并在该数据集上评估并微调了多种视觉‑语言模型；同时在DMD（dBMD子集）上检验了跨数据集的泛化能力。

**💡 创新点**

首次构建面向驾驶员行为的自然语言描述数据集，证明零样本视觉‑语言模型在细粒度动作识别上的不足，并通过领域适配微调显著提升了描述质量与跨数据集泛化。

**🔧 技术方法**

采用Video‑LLaVA、InternVideo2.5、Perception‑LM三种VLM；使用LoRA参数高效微调、视频增强、BERTScore、CLAIR和LLM‑评估的ACCR指标进行评估。

**📊 数据集**

主要使用Drive&Act数据集（转换为描述数据集）和Driver Monitoring Dataset（DMD）中的dBMD子集。

**📈 对比分析**

通过BERTScore、CLAIR和ACCR对模型进行基准评估，零样本性能低；在Drive&Act上Fine‑tune后ACCR从48.88提升至79.77，在dBMD上从66.12提升至76.33，显示显著的性能提升。

**⚠️ 局限性**

当前VLM缺乏细粒度动作推理能力，评估依赖LLM可能产生偏差；数据集规模有限、需要人工校正，扩展到更大数据集的可行性仍待验证。

---

## 641. Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025

**arXiv ID:** 2606.02255 | [PDF](https://arxiv.org/pdf/2606.02255v1)

**作者:** Maria Kunilovskaya `[一作]` (NLLG Lab University of Technology), Steffen Eger `[通讯]` (NLLG Lab University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对2018-2025年ACL系列会议论文中人类注释的报告进行了大规模、任务级别的审计，构建了统一的注释报告分类法，并设计了一套基于大型语言模型（LLM）的结构化信息抽取管线，随后利用该管线对约1600篇论文进行抽取，提取出2667个注释任务，并从中分析了注释报告随时间、领域、会议、以及注释目的的变化趋势。

**💡 创新点**

创新点包括：
①提出了首个覆盖25个维度、适用于所有注释任务的统一分类法；
②构建了任务级别的人类注释报告数据集（gold）和大规模自动抽取数据集（LLM）；
③设计并验证了LLM抽取管线，最高模型（Gemini‑3.1‑Pro）在人工裁定集上的一致性超过人类一致性；
④系统性分析了注释报告的不足与改进空间，提出了可操作的最小报告框架。

**🔧 技术方法**

核心技术包括：
- 结构化抽取的LLM（Gemini‑3.1‑Pro、GPT‑4、Claude‑3等）配合自定义提示和JSON输出模式；
- 交叉验证的Krippendorff’s α与精确匹配率评估抽取质量；
- 断点时间序列回归分析评估ACL责任清单（Responsible NLP Checklist）对报告质量的影响；
- 逻辑回归模型探讨注释目的与报告细节的关联。

**📊 数据集**

数据集：
① gold集：41篇论文、72个注释任务的人工裁定标准，用于验证抽取质量；
② LLM集：从ACL Anthology 2018‑2025年间的1,603篇论文中抽取的2,667个注释任务，用于大规模趋势分析。

**📈 对比分析**

比较方法与性能：
- 将LLM抽取结果与gold集对齐，计算每个属性的精确匹配率和Krippendorff’s α；
- Gemini‑3.1‑Pro在所有属性上的精确匹配率为79.9%，Krippendorff’s α为0.606，略高于人类一致性（79.2% / 0.585），表明LLM在任务级别抽取任务报告信息时已达到人类可比水平。

**⚠️ 局限性**

限制：
- 分类法中部分属性需要对论文描述进行细致解释，导致某些属性的一致性较低；
- 依赖LLM抽取，仍可能出现错误，尤其在文本不完整或表述模糊时；
- gold集规模有限（仅41篇论文），难以覆盖所有类型的注释任务；
- LLM集采用关键词检索，可能存在采样偏差，无法完全代表所有ACL会议论文。

---

## 642. FW-NKF: Frequency-Weighted Neural Kalman Filters

**arXiv ID:** 2606.02251 | [PDF](https://arxiv.org/pdf/2606.02251v1)

**作者:** Adnan Harun Dogan `[一作]` (ETH Zürich), Christian Holz `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出频率加权神经卡尔曼滤波器(FW-NKF)，通过学习观测网络和可学习的IIR创新滤波器，显式抑制测量中的频带噪声，实现更鲁棒的状态估计。

**💡 创新点**

创新点在于：① 将因果频率加权算子嵌入Kalman残差；② 使用频域重建损失指导观测网络；③ 联合训练观测网络、过程模型与IIR滤波器，实现对频域噪声的主动抑制。

**🔧 技术方法**

采用深度学习技术（GRU/MLP观测网络、可学习的Q/R、IIR滤波器），频域损失（FFT幅度差），以及因果IIR创新滤波实现频率选择性校正。

**📊 数据集**

四个异构基准：Lorenz混沌系统、摆杆、UIP-DB人体姿态+UWB传感器、EuRoC MAV无人机。

**📈 对比分析**

与经典Kalman、Autoregressive KF、BayesKNet、KalmanNet、Recursive KNet等基线比较；FW-NKF在RMSE、MAE、ATE等指标上普遍优于对手，提升幅度最高可达10%~20%，且保持较低计算开销。

**⚠️ 局限性**

局限性：需手工设置频率权重λ，过大会导致性能退化；IIR滤波参数学习需稳定性约束，训练复杂；在某些非频域噪声占主导的场景提升有限；对极端非线性或高维系统的可扩展性仍待验证。

---

## 643. Towards Resolving Optimization Conflicts Between Image- and Text-Based Person Re-Identification

**arXiv ID:** 2606.02242 | [PDF](https://arxiv.org/pdf/2606.02242v1)

**作者:** Karina Kvanchiani `[一作]` (Tevian), Timur Mamedov `[通讯]` (Tevian)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段训练框架，先用ReText在多摄像头图像数据上预训练共享视觉编码器并加入文本监督，再冻结该编码器，在RDE框架中仅训练文本编码器以实现图像与文本的统一人检索。

**💡 创新点**

通过训练阶段分离和视觉编码器冻结策略解决了图像检索与文本检索目标冲突的核心问题，并证明加入文本监督能显著提升视觉表示的语义一致性，从而兼顾I2I和T2I性能。

**🔧 技术方法**

使用ReText的IAM损失进行图像预训练，RDE的TAL与IAM混合对齐；采用CLIP‑style对齐、Triplet Alignment Loss、Identity‑aware Matching Loss；实现视觉编码器冻结及多域混合训练。

**📊 数据集**

多摄像头人检索数据集：CUHK03、Market‑1501、MSMT17；单摄像头文本描述数据集：CUHK‑PEDES、ICFG‑PEDES、RSTPReid；合成文本图像数据集SYNTH‑PEDES；5D组合（ENTIRe‑ID、IUSTPersonReID、CUHK‑SYSU、PANDA、OWD）。

**📈 对比分析**

与ReText、DynaMix等单模态方法相比，Rank‑1提升约5–10%；与RDE、IRRA、HPL等统一模型相比，T2I mAP超过90%（CUHK‑PEDES）并在跨模态检索上保持竞争力；整体性能在I2I与T2I之间取得较好平衡。

**⚠️ 局限性**

冻结视觉编码器虽能避免目标冲突，但可能限制模型在更大规模跨模态任务中的适应性；对域间差异的泛化仍需更多多模态数据；文本描述多样性不足可能导致语义对齐不足。

---

## 644. Why Are DMD Students Lazy? Understanding the Copying Behavior in Few-Step Distillation

**arXiv ID:** 2606.02237 | [PDF](https://arxiv.org/pdf/2606.02237v1)

**作者:** Shucheng Li `[一作]` (University of Oxford), Michael M. Bronstein `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究在分布匹配蒸馏（DMD）中，学生模型在高维数据中自发复制教师模型的噪声‑数据对，而非仅仅匹配分布；

**💡 创新点**

创新点在于提出并量化“复制行为”（Copying）以及其与高维几何约束的关系，并排除记忆化、对抗或回归损失是原因的可能性；

**🔧 技术方法**

采用分布匹配蒸馏方法、最优传输与蒸馏传输能量的比值（Pairing Inefficiency）作为衡量指标，并在训练过程中监控DM损失；

**📊 数据集**

主要在MNIST（28×28手写数字）和低维棋盘数据上进行实验，随后扩展到更高维图像数据；

**📈 对比分析**

通过将学生生成结果与教师多步采样在相同噪声下的配对差异绘制热图，并使用Δ_E指标量化，发现高维任务下Δ_E近0，证明复制显著；

**⚠️ 局限性**

局限性在于分析仍为经验性关联，缺乏严格的几何/拓扑理论证明，且实验仅限于小规模数据集，未验证在大型文本到图像生成模型中的适用性。

---

## 645. Simultaneous Model-Based Evolution of Constants and Expression Structure in GP-GOMEA for Symbolic Regression

**arXiv ID:** 2606.02236 | [PDF](https://arxiv.org/pdf/2606.02236v1)

**作者:** Johannes Koch `[一作]` (Centrum Wiskunde & Informatica), Peter A. N. Bosman `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了GP-RV-GOMEA方法，将GP-GOMEA与实时常量优化的RV-GOMEA融合，实现在整个种群中同时优化表达式结构与实数常量，并引入了intron-aware Gaussian更新机制；

**💡 创新点**

创新点在于：①在GP进化全过程中对所有个体的常量进行集成优化；②采用GAMBIT启发的模型基础进化框架；③加入了对未使用常量（intron）的自适应高斯分布更新，提高搜索效率；

**🔧 技术方法**

主要技术包括GOMEA、GP-GOMEA、RV-GOMEA、关联学习、Anticipated Mean Shift、Adaptive Variance Scaling、线性缩放、重启策略、L-BFGS常量调优、SymPy简化等；

**📊 数据集**

实验数据集包括六个合成符号回归问题（含正弦、平方根等复杂形式）以及五个公开真实数据集：Airfoil Self‑noise、Concrete Compressive Strength、Energy Cooling、Energy Heating、Yacht Hydrodynamics；

**📈 对比分析**

通过5折交叉验证、7次重复、比较ERCs、ERCs+CM、RV、RV+IA四种常量优化方式，评估MSE和R²；结果显示RV+IA在几乎所有设置下几乎完全达到目标MSE 1e-8，并在R²上优于ERCs和系数突变；

**⚠️ 局限性**

局限性包括：①常量池大小和RV-GOMEA参数尚未充分调优；②未与在线梯度优化方法对比；③在真实数据中表达式规模略增大，缺乏多目标平衡；④未考虑所有GAMBIT的机制，可能进一步提升效果。

---

## 646. Symmetry-Aware 9D Pose Estimation with Sim(3)-Consistent Feature and Spherical Inception Convolution

**arXiv ID:** 2606.02219 | [PDF](https://arxiv.org/pdf/2606.02219v1)

**作者:** Panfei Cheng `[一作]` (Hunan University), Naveed Akhtar `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无形状先验的类别级9D姿态估计框架SSH-Pose。

**💡 创新点**

引入基于对称性的翻译/尺寸估计、MHP几何特征以及低成本SlinConv旋转估计三大创新模块。

**🔧 技术方法**

使用DINOv2语义特征、PointNet++编码器、MHP-Module生成Sim(3)一致几何特征，以及SlinConv进行球面卷积实现旋转估计。

**📊 数据集**

在NOCS-REAL275、NOCS-CAMERA25、Wild6D三大公开数据集上进行训练与评估。

**📈 对比分析**

与SecondPose等SOTA直接回归方法对比，mAP指标均达到或超越SOTA，拾取成功率提升至86%。

**⚠️ 局限性**

训练依赖真实数据，缺乏纯合成数据的域迁移能力。

---

## 647. Deep Learning for Remote Sensing to Improve Flood Inundation Mapping

**arXiv ID:** 2606.02310 | [PDF](https://arxiv.org/pdf/2606.02310v1)

**作者:** Yogesh Bhattarai `[一作]` (Howard University), Sanjib Sharma `[通讯]` (Howard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在极端降雨导致云盖严重的遥感影像中，利用基于去噪扩散概率模型的 Masked Diffusion Transformer（MDT）实现云层去除，并重建洪水区域。

**💡 创新点**

创新点在于将自注意力 Transformer 与掩码扩散过程结合，专门针对洪水场景的多光谱数据和 DEM 信息进行学习，能够在保持水体连续性和光谱特征的同时，逐步去除云层遮挡。

**🔧 技术方法**

使用 Denoising Diffusion Probabilistic Models、Masked Diffusion Transformer（MDT）架构、交叉熵+Dice+Focal 组合分割损失、MSE+Perceptual+SSIM 组合重建损失，以及 AdamW 优化器。

**📊 数据集**

训练数据为 2018‑2025 年期间 9 场洪灾事件的 Sentinel‑2B 10 m 多光谱影像（共 62 张）及对应的 5 类土地覆盖分割图与 DEM，测试时使用 2021 年田纳西州洪灾场景。

**📈 对比分析**

与 Sen2Cor（规则云掩模）、U‑Net、ViT 基线模型对比；MDT 在测试集上实现 mIoU 最高 0.57，残余云覆盖率从 7.91% 降低到 6.09%，优于 Sen2Cor 的 10‑15% 及 U‑Net 的 8.5%。

**⚠️ 局限性**

局限性包括样本量有限、缺乏多地区多气候的泛化验证、未直接引入物理水动力学约束、仅关注云遮挡而非更细粒度的云类型，导致在极端条件下可能出现幻觉或水体不连续的问题。

---

## 648. Measurement Geometry and Design for Trustworthy Generative Inverse Problems

**arXiv ID:** 2606.02309 | [PDF](https://arxiv.org/pdf/2606.02309v1)

**作者:** Pengfei Jin `[一作]` (Massachusetts General Hospital and Harvard Medical School), Quanzheng Li `[通讯]` (Massachusetts General Hospital and Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究生成模型在逆问题中的测量几何影响，提出局部测量‑数据流形兼容度指标，并基于此设计固定与自适应采样规则，在行采样、CT角度选择和FastMRI Cartesian采样中验证其有效性。

**💡 创新点**

将生成先验的可观测切向方向量化为 α(A;x) 并证明其控制误差，提出对数行列式得分作为设计友好指标，进一步基于后验云实现无训练的自适应采样，说明测量算子如何决定幻觉与重建质量。

**🔧 技术方法**

使用生成式逆解（扩散/评分模型）、局部流形理论、最小奇异值与对数行列式指标、kNN‑PCA 估计子空间、贪婪子空间采样、FastMRI、CT 与 MNIST 数据集的实验。

**📊 数据集**

MNIST（行采样与CT模拟）和 FastMRI（体素 MRI）数据集。

**📈 对比分析**

与固定行/角/线采样、均匀、随机、Poisson‑式等非学习基线对比；自适应后验云方法在所有设置中均取得最低相对误差/NRMSE，提升幅度从数个百分点到十几个百分点，验证了几何指标与重建质量的关联。

**⚠️ 局限性**

仅提供局部方向性稳定性证明，贪婪改进不保证全局最优；先阶段后验云推断增加计算；未充分考虑真实硬件约束和更复杂的临床采样需求。

---

## 649. FATE-VLA:Failue-aware test generation for vision-language-action models

**arXiv ID:** 2606.02307 | [PDF](https://arxiv.org/pdf/2606.02307v1)

**作者:** Arusa Kanwal `[一作]` (Mondragon University), Aitor Arrieta `[通讯]` (Mondragon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了将 Vision–Language–Action（VLA）模型的评估转化为主动发现失败的问题，并实现了两种结合 Adaptive Random Testing 与机器学习预测模型的测试生成算法（FATE-VLA）。

**💡 创新点**

创新点在于：①把 VLA 评估从被动的静态基准转变为主动的失败发现框架；②融合多样性驱动的探索与基于决策树/随机森林的失败预测，显著提升失败检出率并保持多样性。

**🔧 技术方法**

使用的技术包括 Adaptive Random Testing（FSCS‑ART）实现多样性驱动探索，欧氏距离度量多样性；决策树和随机森林作为 surrogate 模型进行失败预测；自适应权重 α 控制探索与利用的平衡。

**📊 数据集**

使用了 SimplerEnv 仿真环境，包含 7 种物体（RedBull 罐、蓝塑料瓶、茄子、胡萝卜、苹果、勺子、海绵），仅在 Pick‑up 任务中进行实验。

**📈 对比分析**

与随机（B1）和仅基于 ART（B2）两种基线比较；在四个 VLA 模型上，失败率（FR）提升 14–30%（最高 29.7%），轨迹覆盖率（TC）、失败轨迹覆盖率（TCF）和失败物体覆盖率（FOC）与基线持平或略有提升，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：仅评估单一简易任务；物体种类受限于 7 种；实验仅在仿真环境中进行；方法对随机种子敏感，需多次重复以降低偶然性。

---

## 650. Unified Context Evolution for LLM Agents

**arXiv ID:** 2606.02304 | [PDF](https://arxiv.org/pdf/2606.02304v1)

**作者:** Zixuan Zhu `[一作]` (Nanyang Technological University), Yuzhi Zhao `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无梯度的框架——Unified Context Evolution（UCE），通过构建可演化的上下文单元（ECU）来持续提升LLM代理在多步交互任务中的性能。

**💡 创新点**

创新点在于：①把经验拆分为四种功能互补的ECU类型（Memory、Strategy、Workflow、Skill）；②通过使用频次与成功率的“fitness”评分跟踪每个ECU的价值；③设计知识收益调度（KYS）在每个循环中动态分配生成预算，聚焦库中薄弱区域。

**🔧 技术方法**

核心技术包括：基于语义相似度与任务类型匹配的检索策略；对生成的ECU进行类型化、去重与评估；使用拉普拉斯平滑的fitness评分；循环式的评估、收集、调度、生成与清理阶段。

**📊 数据集**

在两个文本交互基准上进行实验：ALFWorld（134个家庭任务，6类）和WebShop（100个亚马逊商品购买任务）。

**📈 对比分析**

与ReAct、NoThinking、Plan-and-Act、ReflAct以及ExpeL、Reflexion等方法对比，UCE在ALFWorld峰值成功率提升至96.3%（相对基线+20.9pp），在WebShop任务得分提升至61.3%（相对基线+16.2pp），并且库可无缝迁移到不同代理骨干。

**⚠️ 局限性**

限制包括：仅在离散动作、文本观察的两套小规模环境上测试；使用单一闭源代理与生成器模型，难以分离框架与模型贡献；ECU类型手工设计，缺乏自动发现机制；不涉及连续动作、视觉或更长规划的场景。

---

## 651. Quantitative Movement Testing: Measuring Patient Movements from a Single Smartphone Video

**arXiv ID:** 2606.02301 | [PDF](https://arxiv.org/pdf/2606.02301v1)

**作者:** Pranav Mahajan `[一作]` (University of Oxford), Ben Seymour `[通讯]` (University of Oxford)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证了Quantitative Movement Testing（QMT）框架，利用单摄像头手机视频实现三维运动测量，并在实验室、临床试验（纤维肌痛）与居家监测（慢性坐骨神经痛）中评估慢性疼痛患者的功能性运动；

**💡 创新点**

创新点在于：①将单摄像头的3D姿态估计与动态时间规整结合，提取可重复的运动周期；②使用平面角度替代传统Euler角，提升轴向运动的测量精度；③提出可作为数字终点的客观指标，并在真实世界临床流程中实现远程采集；

**🔧 技术方法**

核心技术包括：VideoPose3D 3D姿态估计、动态时间规整（DTW）对运动周期对齐、离线偏置校正、平面角度推导、线性混合模型、贝叶斯因子等；

**📊 数据集**

使用的数据集为：实验室验证13名健康受试者；PainLESS 纤维肌痛患者80名中54名有效用于前后测；BeADS 46健康+51慢性坐骨神经痛患者（共44名健康、39名患者有效）进行30天居家监测；

**📈 对比分析**

与光学运动捕捉对比采用相关系数、MAE、Bland-Altman；校正后所有主要指标r>0.85，MAE≤7.5°；居家测量与自评距离对比MAE≈7.25 cm，ΔMAE≈3.10 cm，组间差异显著；单日分类准确率68.5%（AUC≈0.77）；在PainLESS试验中，QMT检测到的干预效应未显著，贝叶斯因子提示数据更支持无效应假设；

**⚠️ 局限性**

主要局限包括：居家环境噪声大、单日测量变异高导致绝对精度下降；缺乏针对患者群体的实验室验证与正向对照；VideoPose3D模型未针对慢性疼痛数据微调，深度估计误差影响；需要改进实时帧率反馈与多次重复采集以降低噪声。

---

## 652. Beyond Isolated Behaviors: Hierarchical User Modeling for LLM Personalization

**arXiv ID:** 2606.02300 | [PDF](https://arxiv.org/pdf/2606.02300v1)

**作者:** Liang Wang `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将用户行为划分为实践、习惯和领域三层，构建了一个层级化的LLM个性化框架；

**💡 创新点**

创新点在于借鉴Bourdieu的实践-习惯-领域理论，首次将个体行为的时间累积与跨用户共享结构统一到模型层级；

**🔧 技术方法**

采用残差向量量化抽象实践，时间加权聚合习惯，KMeans聚类形成领域，并将习惯与领域嵌入投影到冻结的LLM中进行生成；

**📊 数据集**

使用公开的LaMP基准数据集，涵盖六个分类和生成任务；

**📈 对比分析**

与零样本、检索、聚合等多种基线对比，PHF_Compass在所有任务上实现了最高或接近最高的准确率/ROUGE分数，明显优于传统平面层次方法；

**⚠️ 局限性**

局限在于仅在单语文本静态任务上验证，缺少多语言、多模态、实时流式情境的评估，并未充分探讨隐私与公平性等伦理问题。

---

## 653. Neural Acquisition & Representation of Subsurface Scattering

**arXiv ID:** 2606.02292 | [PDF](https://arxiv.org/pdf/2606.02292v1)

**作者:** Arjun Majumdar `[一作]`, Hendrik Lensch `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用。

**💡 创新点**

创新点在于提出了一种改进的模型结构，能够更有效地处理复杂数据。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和循环神经网络（RNN）的结合。

**📊 数据集**

使用了公开的图像数据集和文本数据集进行实验。

**📈 对比分析**

与现有方法进行了对比，结果显示新方法在准确率和效率上均有显著提升。

**⚠️ 局限性**

限制在于模型对特定类型数据的适应性较差，且训练时间较长。

---

## 654. DECK: A Consistency x Confidence Taxonomy of LLM Hallucinations

**arXiv ID:** 2606.02289 | [PDF](https://arxiv.org/pdf/2606.02289v1)

**作者:** Mohit Singh Chauhan `[一作]` `[通讯]`, Mohit Singh Chauhan

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DECK 2×2 分类法，将 LLM 幻想按样本间一致性与标记级置信度划分为 Drift、Entrenched、Confabulation、Knotted 四类，并验证其可检测性签名与三大输出级不确定性评分器（黑盒一致性、白盒 token 概率、LLM-as-Judge）之间的对应关系。

**💡 创新点**

创新点在于把错误分类转向“可检测性签名”，揭示每种错误类型对应的检测盲区，并首次系统性证明了知识缺口输入会导致所有输出级不确定性方法同时失效（即“通用盲区”）。

**🔧 技术方法**

使用 UQLM 工具箱实现 6 种黑盒、5 种白盒、4 种 Judge 评分器；通过 Youden's J 最优阈值将样本划分到四个 DECK cell；采用 AUROC 和 hallucination‑restricted complementarity 指标（C_H）评估检测器性能与互补性。

**📊 数据集**

实验数据集包括 TriviaQA、HaluEval、SelfAware（未回答问题）和 PopQA（实体流行度分层），在 Llama‑3‑8B、GPT‑4o、Gemini‑2.5‑Flash 三个模型上进行评估。

**📈 对比分析**

比较结果显示没有单一评分器家族占优；在不同模型与数据集组合中，黑盒、白盒和 Judge 的 AUROC 均为 0.6–0.8 之间；DECK 预测的误差簇与外部标签一致，C_H 指标在 Judge‑BlackBox/WhiteBox 对中呈现预期分布，说明三家族互补性显著，集成可提升检测效果。

**⚠️ 局限性**

主要局限包括：阈值设定依赖于数据；Judge 仍可能受共享偏差影响；通用盲区验证仅用单一内部状态线性探针，其他内部状态方法未检验；仅针对短文本问答，长文本生成可能需要不同框架。

---

## 655. From Extrinsic to Intrinsic: Geodesic-Guided Representation Learning for 3D Geometric Data

**arXiv ID:** 2606.02268 | [PDF](https://arxiv.org/pdf/2606.02268v1)

**作者:** Yuming Zhao `[一作]` (City University of Hong Kong), Ying He `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了基于几何内在属性的3D预训练方法PRISM，通过预测点云上的测地线距离来学习等距嵌入。

**💡 创新点**

创新点在于将测地线距离作为自监督预训练目标，并引入结构一致性损失和两阶段采样策略，实现等距嵌入并兼顾全局与局部几何。

**🔧 技术方法**

采用点变换器PTv3作为骨干网络，结合MLP预测头、L1/MRE损失、结构一致性损失、重要性采样等技术。

**📊 数据集**

预训练使用ShapeNet点云子集，后续任务使用ShapeNetPart、FAUST、ScanObjectNN等公开数据集。

**📈 对比分析**

在测地线预测、固定边界参数化、非刚性对应、分类、分割等任务上与传统方法和最新学习方法对比，取得相当或更优的精度且推理速度最快。

**⚠️ 局限性**

主要局限是对大角度旋转仍不够鲁棒，且仅使用点云输入，尚未探索多任务协同训练。

---

## 656. A combination of noise and bilateral filters achieve supralinear and scalable adversarial robustness in CNNs

**arXiv ID:** 2606.02267 | [PDF](https://arxiv.org/pdf/2606.02267v1)

**作者:** Nicolas Stalder `[一作]` (ETH Zürich), Pau Vilimelis Aceituno `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了一种简单的预处理器，在训练和推理阶段先添加高斯噪声再进行多次双边滤波，显著提升深度网络对对抗攻击的鲁棒性。

**💡 创新点**

证明高斯噪声与滤波通过不同几何机制抵御攻击，并且两者组合可实现超线性鲁棒提升，首次系统性地阐述了这类组合的理论与实践效果。

**🔧 技术方法**

使用高斯噪声注入、双边滤波、对抗训练（TRADES）、EDM合成样本等技术构建模型。

**📊 数据集**

主要在CIFAR‑10上验证，并在Imagenet10上进行短期消融实验。

**📈 对比分析**

与RobustBench基准（AutoAttack等）比较，结合预处理器的模型在AutoAttack上排名第二、整体第三，同时仅消耗约35%的训练FLOPs、50%参数、33%训练周期、15%数据；在多种模型规模下亦保持竞争性准确率。

**⚠️ 局限性**

对大模型的收益递减、对抗训练与梯度遮蔽的潜在交互、实验复现误差以及仅针对CNN的通用性等方面仍存在局限。

---

## 657. Four constructions of self-dual binary cyclic codes with a lower bound on the minimum distances better than the square-root bound

**arXiv ID:** 2606.02262 | [PDF](https://arxiv.org/pdf/2606.02262v1)

**作者:** Xiaoqiang Wang `[一作]` (Hubei University), Cunsheng Ding `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文解决了编码理论中一个长期存在的开放问题，提出了无限个自对称二进制循环码的构造，这些码的最小距离下界优于平方根下界。

**💡 创新点**

创新点在于构造了四种自对称二进制循环码的无限家族，并且这些码的最小距离下界超过了平方根下界，从而为70年来的开放问题提供了肯定的答案。

**🔧 技术方法**

使用了循环码的定义集和BCH码的性质等技术，发展了四种构造方法。

**📊 数据集**

使用了长度为2^m+1-2的自对称二进制循环码，涉及的参数和构造方法在文中详细列出。

**📈 对比分析**

与已有的自对称循环码进行比较，本文构造的码在维度和最小距离下界上均优于现有文献中的结果，且部分码达到距离最优。

**⚠️ 局限性**

限制在于尽管构造了无限个自对称二进制循环码，但仍然存在更好的最小距离下界的构造问题，未来的研究可以进一步探索这一方向。

---

## 658. ResMerge: Residual-based Spectral Merging of Large Language Models

**arXiv ID:** 2606.02252 | [PDF](https://arxiv.org/pdf/2606.02252v1)

**作者:** Yandu Sun `[一作]` (Southeast University), Jinqiao Wang `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在强化学习后训练的专家模型的无训练合并，提出了一种基于残差的谱合并框架 ResMerge。

**💡 创新点**

创新点在于发现 RL 任务向量的头部和残差均携带可恢复的行为知识，并采用残差为主干、头部通过可靠性门控进行轻量级注入，显著降低了专家间的冲突。

**🔧 技术方法**

使用了矩阵奇异值分解 (SVD)、球面残差共识适配 (SRC‑A) 和轻量级头部校正 (LHC) 等技术，辅以向量一致性度量和加权指数控制。

**📊 数据集**

在 Qwen2.5‑7B‑Base 专家组上进行评估，使用了 LiveCodeBench、HumanEvalPlus、MBPPPlus、GPQA‑Diamond、AIME24/25、AMC23、MATH500、BFCL V3、HotpotQA 与 SQuAD 等多领域数据集。

**📈 对比分析**

与 Task Arithmetic、TIES、DARE、TSV‑Merge、ISO‑C/CTS、RAM 等基线进行对比，ResMerge 在整体平均得分上领先，提升约 2%–3%，在多项子任务上保持最优或接近最优。

**⚠️ 局限性**

局限性包括仅适用于同一基础模型的专家，未支持多架构或输入依赖的动态路由，且合并后的模型可能继承源专家的安全与偏见风险。

---

## 659. Ego-METAS: Egocentric online Multimodal Energy-efficient Temporal Action Segmentation benchmark

**arXiv ID:** 2606.02246 | [PDF](https://arxiv.org/pdf/2606.02246v1)

**作者:** Maria Santos-Villafranca `[一作]` (University of Zaragoza), Antonino Furnari `[通讯]` (University of Catania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Ego-METAS 基准，专门用于评估具备能耗约束的实时多模态时间动作分割模型；

**💡 创新点**

首次将三大 egocentric 数据集统一，定义严格的能量预算和动态路由任务，并给出标准评估指标；

**🔧 技术方法**

使用 ProTAS 等现有时间动作分割网络，结合固定与学习型路由策略（随机、贪婪、AdaMML、HCMS）以及预提取特征，构建端到端评测流程；

**📊 数据集**

EgoExo4D、CMU-MMAC 与 CaptainCook4D 三大多模态 egocentric 数据集；

**📈 对比分析**

通过在 20 mW 与 2.8 W 两种能量预算下对比固定与学习型路由，实验发现随机路由在多数场景下达到最优能耗-准确率平衡，学习型策略在连续无剪辑序列中表现不佳；

**⚠️ 局限性**

能耗估算基于硬件配置，未在真实设备上验证；路由策略仅在预提取特征上评估，缺少与特征提取器共同优化的实验；

---

## 660. When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation

**arXiv ID:** 2606.02245 | [PDF](https://arxiv.org/pdf/2606.02245v1)

**作者:** Mingyan Wu `[一作]` (Northeastern University), Yftah Ziser `[通讯]` (NVIDIA Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了成本感知检索增强生成（Cost‑Aware RAG）框架，在检索阶段引入访问成本约束，扩展MS MARCO为成本标注语料，并评估静态与动态（代理）证据选择策略。

**💡 创新点**

首次将检索证据的访问成本作为预算约束纳入RAG，构建多层成本标签，并探讨LLM代理在每个查询上动态决定检索、访问层级和停止时机的能力。

**🔧 技术方法**

使用Llama‑3.1‑8B‑Instruct与Qwen3‑8B两大开放权重模型；基于ReAct框架的代理、Greedy、Knapsack、Redundancy‑Aware、MMR等算法；结合MS MARCO成本标注检索库和多种评估基准。

**📊 数据集**

MS MARCO v2.1（成本标注版）作为检索语料；HotpotQA、Natural Questions、TriviaQA、MedQA‑US、MMLU‑Med作为QA评测集。

**📈 对比分析**

与传统top‑k、成本无关的静态选择器对比，发现无单一策略在所有数据集、预算和模型上均占优；预算增大不一定提升答案质量；代理模型在Qwen3‑8B上能在更低成本、较少证据的前提下达到或超过最佳静态策略，而在Llama模型上表现不稳定。

**⚠️ 局限性**

局限性包括：访问成本仅为模拟的离散层级，未体现真实费用与许可；域级成本标签自动化可能产生误判；实验覆盖面仅限五个QA基准，未检验更大样本或不同域；检索过程简化，未处理多跳推理、查询重写等；代理控制在零样本提示下对模型和任务高度敏感。

---

## 661. AgentRedBench: Dynamic Redteaming and Integration-Aware Defense for LLM Agents over SaaS Integrations

**arXiv ID:** 2606.02240 | [PDF](https://arxiv.org/pdf/2606.02240v1)

**作者:** Hiskias Dingeto `[一作]` (StackOne), Will Leeney `[通讯]` (StackOne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于LLM的动态红队评测框架ARG，用于检测和防御通过工具响应实现的间接提示注入攻击，并提供了对应的防御模型ARG Guard。

**💡 创新点**

创新点在于：①将评测动态化，攻击内容在每次运行中由LLM实时生成；②覆盖了24种企业集成、215个细粒度授权攻击场景；③设计了一种小模型（MiniLM）防御器，在保持极低误报的同时显著降低攻击成功率；④在跨集成、跨攻击类型上验证了泛化能力。

**🔧 技术方法**

主要技术包括：LLM生成的动态攻击器、工具调用循环评测管道、LLM判别器评估攻击成功率、基于MiniLM的文本分类防御器、跨模型评测（Anthropic、OpenAI、Google）。

**📊 数据集**

数据集为从215个手工设计的授权攻击场景生成的动态注入文本，结合24种企业集成的工具响应，构成训练集（约14,846条攻击样本）和测试集（约2,029条跨集成样本）。

**📈 对比分析**

与四个主流开源防御模型（Llama Guard、PromptGuard 2、ProtectAI、WildGuard）对比，ARG Guard在0.37% FPR下实现99.75% TPR，攻击成功率从69.9%降至2.4%，在延迟和误拒方面表现优异（9.5 ms CPU延迟，0%误拒）。

**⚠️ 局限性**

局限性包括：①场景集的选择偏向已知可攻击性，可能高估总体攻击率；②攻击者在重试过程中仅在语义层面变体，未能探索新攻击类别；③未覆盖模型权重攻击、UI重定向等其他安全威胁；④评测依赖于模拟集成，真实环境的细节差异可能影响结果。

---

## 662. Optimizing the Envy Cycle Elimination Algorithm

**arXiv ID:** 2606.02233 | [PDF](https://arxiv.org/pdf/2606.02233v1)

**作者:** Karen Frilya Celine `[一作]` (National University of Singapore), Warut Suksompong `[通讯]` (National University of Singapore)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在分配不可分物品时，利用禀性循环消除（ECE）算法的启发式策略以提升公平性约束下的福利保证。

**💡 创新点**

创新点在于系统评估了多种启发式（仅选代理、仅选物品、同时选代理与物品以及最大最小匹配）对强效用价和强平等价价的影响，并证明了部分启发式可显著降低强效用价。

**🔧 技术方法**

主要技术包括EF1公平性理论、价格化分析（弱价与强价概念）、贪心优化、最大最小匹配算法以及严格的极值构造与证明。

**📊 数据集**

实验部分使用从[0,1]均匀分布、指数分布、对数正态分布以及Spliddit实际数据中随机生成的效用实例，进行10,000次重复。

**📈 对比分析**

通过与无启发式ECE、轮询、最大最小匹配等多种基线对比，发现同时最大化代理与物品效用的贪心启发式在强效用价和平均效用/平等价上均优于传统方法；在两代理情形下，其强平等价亦降至3。

**⚠️ 局限性**

限制在于对三及以上代理时，强平等价仍为无穷大；且目前尚未给出能同时最优提升强效用价与强平等价的多项式时间启发式。

---

## 663. Composable function systems as a general-purpose rendering framework

**arXiv ID:** 2606.02226 | [PDF](https://arxiv.org/pdf/2606.02226v1)

**作者:** James Schloss `[一作]` `[通讯]` (Massachusetts Institute of Technology), James Schloss (Massachusetts Institute of Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可组合函数系统（CFS）用于通用渲染，并基于此设计了GPU加速的元编程框架Quibble。

**💡 创新点**

创新点在于将函数系统的定义与渲染管线解耦，支持任意顺序组合函数、实时生成点云，并通过元编程在GPU上即时编译高效核。

**🔧 技术方法**

核心技术包括迭代函数系统（IFS）、自定义可延迟编译的“scribble”语言、OpenCL后端以及函数指针在GPU上的动态调用。

**📊 数据集**

主要使用自定义生成的几何体数据集（如由IFS生成的正方形、圆形、三角形等点云），并在实验中通过合成图像进行评估。

**📈 对比分析**

与传统网格、光线追踪或体素渲染比较，实验显示单帧生成时间约0.0215 s（平均1024次），可达45 fps；单图像完整耗时约0.745 s，显示出低内存占用与高实时性。

**⚠️ 局限性**

主要限制包括编译时间占比高、仅支持OpenCL导致与游戏引擎兼容性差、对稠密估计场景需原子操作降低性能，以及点云过度填充导致效率下降。

---

## 664. Chroma Clues: Leveraging Color Statistics to Detect Synthetic Images

**arXiv ID:** 2606.02224 | [PDF](https://arxiv.org/pdf/2606.02224v1)

**作者:** Lea Uhlenbrock `[一作]`, Christian Riess `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于颜色变换的图像取证框架，通过构造手工和学习得到的颜色变换，提取噪声残差、频谱特征和共现统计，从而实现对生成图像的二分类、局部篡改定位和多源归因；

**💡 创新点**

创新点在于揭示生成器对LPIPS损失的亮度偏向导致的色度失真，并基于此设计六种手工颜色变换；同时通过1×1卷积CNN学习颜色变换，最大化自然图像与生成图像在颜色空间中的可视差异；

**🔧 技术方法**

使用的技术包括LPIPS敏感性分析、颜色空间转换、手工/学习的颜色变换、对角拉普拉斯高通滤波、图像残差特征（指纹、熵、空间/通道共现）、SVM分类器集成以及广泛的后处理鲁棒性评估；

**📊 数据集**

实验数据集涵盖COCO、LAION、Stable Diffusion、Midjourney、Adobe Firefly、TGIF填补数据、Synthbuster基准等多种真实与生成图像，覆盖多代生成模型；

**📈 对比分析**

与十个SOTA方法（包括CLIP、频谱、颜色、深度特征等）进行对比，采用二分类、后处理鲁棒性和跨生成器泛化等评价指标；结果显示平衡模型在二分类平均精度达到93.27%，鲁棒模型在后处理攻击下平均精度80.25%，在多源归因上也显著优于多数对手；

**⚠️ 局限性**

局限性包括对图像质量和纹理敏感，低分辨率或高噪声图像时颜色残差可视性下降；在复杂后处理（如JPEG AI、强烈伽马校正）下性能仍有衰减；对新兴生成器的适应性需进一步验证。

---

## 665. CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations

**arXiv ID:** 2606.02221 | [PDF](https://arxiv.org/pdf/2606.02221v1)

**作者:** Chengfeng Wu `[一作]` (Tsinghua University), Jingge Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于因果正交表示的多任务学习框架 CORE-MTL，利用语义流和残差流的结构化分解来提升模型的 OOD 泛化能力。

**💡 创新点**

创新点在于从表示层面解耦任务相关语义与噪声（残差），通过物理或软 grounding、CKA 独立性约束和 Counterfactual Augmentation 形成结构化正交性，从而在理论上给出更紧的 OOD 泛化上界，并实现梯度冲突的几何消除。

**🔧 技术方法**

主要技术包括：双流编码器 (Dual-Stream Encoder)、中心化核对齐 (CKA) 作为正交性正则、对抗式重构 (reconstruction loss) 与物理逆渲染解码器、对抗性 Counterfactual Augmentation、以及标准 ResNet-50 作为基底。

**📊 数据集**

实验数据集涵盖密集场景任务：NYUv2、Cityscapes、CelebA；以及 OOD 场景：GTA5→Cityscapes、Cityscapes-C。

**📈 对比分析**

与 10 种主流基线（GradNorm、PCGrad、MTAN、STCH、MOML、RLW、ExcessMTL、FairGrad、RepMTL 等）进行对比，CORE-MTL 在 ID 和 OOD 任务上均取得显著优势，尤其在模拟到真实转移与多种噪声干扰下表现最优。

**⚠️ 局限性**

局限性包括：理论证明基于线性高斯结构因果模型，深度非线性表示的泛化证明尚未完成；当前实现主要针对密集视觉任务，对跨模态或异构任务的适应性需要进一步研究。

---

## 666. $γ$-CounterBoost: Optimizing response time tails using job type information only

**arXiv ID:** 2606.02311 | [PDF](https://arxiv.org/pdf/2606.02311v1)

**作者:** Nils Charlet `[一作]` (University of Antwerp), Benny Van Houdt `[通讯]` (University of Antwerp)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在轻尾分布的 M/G/1 队列中，仅利用作业类型而不利用到达时间信息时的尾部性能优化问题，提出并证明了 γ‑CounterBoost 策略在 Contextual CounterBoost（CCB）策略族内的尾部最优性；

**💡 创新点**

①将 Nudge‑M 策略推广到多类型作业，提出 γ‑CounterBoost；②引入 CCB 类，证明任意 CCB 策略可通过一系列“单次交换”步骤收敛至 γ‑CounterBoost，故其尾部常数最小；③给出重负载下的近似提升函数 b^heavy，易于估计且与 γ‑Boost 重负载极限相同；

**🔧 技术方法**

利用排队理论中的最终值定理、拉普拉斯变换、指数尾部分析以及单次交换（Single Swap）定理；通过构造无理独立的增量 ε 使得策略在增量调整过程中保持可行；

**📊 数据集**

实验使用三种混合分布：a）三种超几何分布（不同均值、SCV、形状参数），b）混合分布（指数、Erlang、超几何），各取不同出现概率；

**📈 对比分析**

通过与 FCFS、γ‑Boost、Nudge‑M 以及基于 30%/50% 偏离最优提升的 CounterBoost 进行比较；结果显示 γ‑CounterBoost 在大多数负载下与 γ‑Boost 差距很小，明显优于 Nudge‑M；在重负载时 γ‑CounterBoost 与 γ‑Boost 的尾部常数趋同；

**⚠️ 局限性**

仅在 CCB 类内证明最优；对全信息（完整作业大小）或更一般的上下文提升（含未来到达）尚未证明；实际实现中需估计 γ、Laplace 变换或提升函数，可能带来误差；

---

## 667. Regularized Large Neighborhood Search

**arXiv ID:** 2606.02294 | [PDF](https://arxiv.org/pdf/2606.02294v1)

**作者:** Germain Vivier-Ardisson `[一作]` (Google DeepMind), Mathieu Blondel `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Regularized Large Neighborhood Search (RLNS)，将传统的 LNS 通过正则化子问题转化为无拒绝的块 Gibbs 采样器，兼具可训练性与可扩展性。

**💡 创新点**

创新点在于：①将正则化的 LNS 与 Gibbs 采样相连结，实现无采样拒绝的 MCMC；②通过熵正则化实现精确的块 Gibbs 采样；③通过扰动正则化仅依赖局部 MAP 求解器；④使用 RLNS 迭代次数调节学习从伪似然到最大似然的连续空间。

**🔧 技术方法**

使用技术包括：熵正则化与扰动正则化的平滑最大化、块 Gibbs 采样、Rao‑Blackwell 化、动态规划求解子问题、Perturb‑MAP 采样、以及深度网络预测与梯度传播。

**📊 数据集**

实验数据集：k‑hot 向量（受控实验）、Generalized Assignment Problem (GAP) 真实数据集、以及多场景随机车辆调度基准。

**📈 对比分析**

与精确 Monte‑Carlo、Local Search MCMC、Perturbed Fenchel‑Young (PFY)、Black‑box 差分 (DBB)、二元交叉熵 (BCE) 以及贪婪策略等方法对比。RLNS 在收敛速度、梯度方差和子问题规模上均优于或与基准相当，尤其在子问题约为原规模三分之一时仍能接近全局 PFY 的性能。

**⚠️ 局限性**

局限性：熵正则化理论上实现精确 Gibbs 采样，但需要全局边缘算子，在某些组合问题中仍不可行；扰动正则化会引入结构性偏差；算法依赖可解的局部子问题，限制了对更复杂约束的通用性。

---

## 668. AI as a Tool for Simulation-Based Experiments in Literary Studies

**arXiv ID:** 2606.02293 | [PDF](https://arxiv.org/pdf/2606.02293v1)

**作者:** Matthew Wilkens `[一作]` `[通讯]` (Cornell University), Matthew Wilkens (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一套基于大型语言模型的文学实验模拟框架，利用生成式AI生成小说章节并与人类创作文本进行对比，探讨通过提示工程和合成作者传记来控制文本风格和内容的可行性。

**💡 创新点**

创新点在于：①首次将“硅样本”与文学生成相结合，通过合成作者简历实现个性化语境；②展示了复杂提示工程在提升生成文本与人类文本相似度、降低同质性方面的显著作用；③系统性评估了模型在跨体裁、受奖级别等多维度上的表现，为未来的文学因果推断实验奠定了技术基础。

**🔧 技术方法**

核心技术包括：GPT‑5（和其他GPT版本）的文本生成；ROME/MEMIT等模型编辑与知识去学习技术的讨论；Qwen3‑Embedding‑8B 文本嵌入；UMAP降维；Fightin’ Words 词频差异分析；prompt‑steerable embedding 与多模态提示工程。

**📊 数据集**

主要数据集为 CONLIT（2001–2021年英文长篇叙事文本，包含2,754卷，其中1,934卷为小说，820卷为叙事非虚构），以及从其中挑选的258部获奖或入围大奖的小说、1,676部非获奖小说作为词频先验；合成作者简历由同一数据集的真实作者信息生成。

**📈 对比分析**

比较方法：对全文（或章节）进行文档嵌入，计算余弦相似度；在嵌入空间中绘制 UMAP 可视化；进行词频差异（Fightin’ Words）分析。结果显示：①复杂提示生成的文本在同类内相似度低于基本提示，且更接近人类文本；②不同体裁之间的相似度保持与人类文本相似的秩序；③在词频层面，AI文本与人类文本可分辨，但差异主要体现在语体与时态偏好上。整体性能表明：模型能够产生与目标体裁在嵌入空间内接近、且多样性更高的文本。

**⚠️ 局限性**

主要局限包括：①对历史/非英语文学的适用性有限；②简单的时间提示无法实现有效的年代化控制；③生成的文本为单章节而非完整小说，难以评估整体叙事结构与质量；④现有模型编辑/去学习方法仍不能充分改写文化/风格知识；⑤对叙事质量、连贯性等细粒度特征的评估不足，需进一步引入专门的叙事分析工具。

---

## 669. Massive Spikes in LLMs are Bias Vectors: Mechanistic Uncovering and Spike-Free Quantization

**arXiv ID:** 2606.02288 | [PDF](https://arxiv.org/pdf/2606.02288v1)

**作者:** Yung-Chin Chen `[一作]` (Princeton University), Naveen Verma `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对大型语言模型中出现的巨大激活尖峰进行深入分析，并提出“偏置向量假设”，认为这些尖峰是结构化向量偏置的标量中间体。基于此，作者设计了一种后训练量化（PTQ）框架：先将尖峰所在的令牌抑制为零，然后在后续层通过预计算的键/值模板恢复其功能，从而实现激活无尖峰、低位宽的高保真量化，并能跨模态（如Vision Transformer）通用。

**💡 创新点**

创新点在于：1）将激活尖峰从标量偏置提升为向量偏置，并通过几何分析揭示其在注意力吸收与价值状态排空中的作用；2）发现模型通过低频RoPE区块与旋转对齐对偏置向量进行稳定化；3）首次提出通过抑制尖峰并插入预计算模板的PTQ方案，既消除尖峰导致的量化误差，又保持模型无操作特性，且可跨文本与视觉任务。

**🔧 技术方法**

技术上使用RMSNorm对激活进行标准化，分析自注意力中键（W_K）与查询（W_Q）对偏置向量的对比放大与聚集作用，使用RoPE频域分析验证偏置向量在低频稳定区的对齐；在量化环节采用每通道权重量化、每张量激活静态量化，以及对KV缓存的分组动态量化；通过离线校准收集偏置模板，并在推理时实施抑制尖峰与模板插入的两步策略。

**📊 数据集**

实验使用的主要数据集：LLM 评估基于 Pile 数据集（64 条 512 令牌序列）进行校准；零样本常识推理任务（ARC‑Easy/Challenge、BoolQ、PIQA、SIQA、HellaSwag、OpenBookQA、WinoGrande）与 WikiText‑2；ViT 评估使用 ImageNet‑1K 验证集 1000 张图片以及 Flickr30k 检索测试集。

**📈 对比分析**

与基线 RTN、QuaRot‑RTN（激活旋转）及 PrefixQuant 进行对比。对于 LLaMA‑2‑7B 等模型，4‑bit 量化下该方法在 CSR 准确率与 WikiText‑2 perplexity 上与 PrefixQuant 相当或略优（如 CSR 60.26% 对比 60.17%），明显优于 RTN 与旋转方法。ViT 侧实验显示在低精度下，RTN 失真严重（DINOv2 ViT‑B 仅 6%），而该框架可恢复至 74% 以上，证明跨模态通用性。

**⚠️ 局限性**

局限性包括：1）仅适用于存在巨大激活尖峰的模型，对已通过架构改动抑制尖峰的模型无效；2）缺乏对训练过程为何收敛到偏置向量的理论证明；3）需要离线校准与模板存储，虽开销极小，但仍需额外步骤。

---

## 670. Dynamics Are Learned, Not Told: Semi-Supervised Discovery of Latent Dynamics Geometries For Zero-Shot Policy Adaptation

**arXiv ID:** 2606.02280 | [PDF](https://arxiv.org/pdf/2606.02280v1)

**作者:** Zhiming Xu `[一作]` (Tongji University), Chenpeng Yao `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于结果导向的半监督方法LDG，利用对比学习学习动态几何结构的潜在空间，从而实现机器人控制的零样本自适应；

**💡 创新点**

创新点在于：①从结果而非物理参数出发，构建潜在空间的几何拓扑；②通过对比学习显式控制编码器的Lipschitz常数，理论上可上界目标域的损失；③融合变分推断与对比学习，得到既平滑又结构化的潜在表示；

**🔧 技术方法**

使用技术包括：变分信息瓶颈（VAE）框架、InfoNCE对比损失、SAC强化学习、梯度正交性分析和谱归一化对比；

**📊 数据集**

主要数据集为MuJoCo连续控制任务：Hopper、Walker2d、HalfCheetah、Ant，随机化质量、阻尼、摩擦等物理参数；

**📈 对比分析**

与基线（SAC+DR、RMA Phase1/2、SO‑CMA、VAE）以及测试时优化方法比较，LDG在ID稳定性和零样本OOV环境（未建模参数、时间变换、结构失效）中均取得更高奖励，尤其在未建模参数和结构失效场景中优势明显；

**⚠️ 局限性**

局限性包括：对高度反应式任务如HalfCheetah可能过度正则化导致性能下降；依赖训练分布中足够的功能多样性，无法泛化至完全新颖的动力学变化；对比学习需要正负样本配对，可能受数据采样影响。

---

## 671. Dexterity-BEV: Aligning 3D World and Actions for Generalizable Robot Policies Learning

**arXiv ID:** 2606.02274 | [PDF](https://arxiv.org/pdf/2606.02274v1)

**作者:** Huayi Zhou `[一作]` (DexForce Technology), Kui Jia `[通讯]` (DexForce Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Dex-BEV框架，统一三维空间对齐的输入输出，实现端到端视觉语言动作（VLA）在多种机器人、相机视角和数据集上的泛化；

**💡 创新点**

创新点包括：①使用对齐的顶点映射与顶点光谱将二维RGB输入提升至三维，②采用鸟瞰视图（BEV）作为全局对齐坐标系并构造BEV图像，③实现跨机器人、数据集的时序轨迹对齐；

**🔧 技术方法**

技术手段包括：顶点映射与光谱投影、BEV图像构建、Flow Matching动作生成、三维空间对齐管线（含手工GUI、ICP、深度估计）、时序归一化；

**📊 数据集**

使用公开数据集LIBERO、RoboTwin-2.0及内部多机器人数据集（Agibot、RoboMind等），以及真实机器人平台Agilex、DexForce W1、W1*、A1进行评估；

**📈 对比分析**

与基线π_0、X-VLA进行对比，Dex-BEV在模拟和真实场景下均取得更高成功率，尤其在相机视角、机器人/场景布局扰动较大时表现更稳健；

**⚠️ 局限性**

局限性：依赖相机标定参数，未在标定未知或极端非结构化环境中验证；

---

## 672. Exact Sampling of Permutations with a Fixed Longest Increasing Subsequence

**arXiv ID:** 2606.02263 | [PDF](https://arxiv.org/pdf/2606.02263v1)

**作者:** Peter Clifford `[一作]` (University of Oxford), Raphaël Clifford `[通讯]` (University of Bristol)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计了两类算法，完成在给定长度 n 与 LIS 长度 k 的前提下，对满足 (π)=k 的排列进行精确均匀采样。对 k=Θ(n) 的情况提出直接拒绝采样，期望时间为 O(n log log n)；对任意 k 提出基于 Robinson–Schensted 对应与哈希行列式技术的分形采样，期望时间为 Õ(n³k⁴)，最直接实现为 Õ(n⁴k⁵)。

**💡 创新点**

创新点在于：
① 通过扩展提议空间并引入“左侧最小 LIS”规范，得到常数接受概率的直接采样器；
② 将形状采样转化为严格分割的 Vandermonde 与独立阶乘因子相乘的权重，允许按坐标逐步抽样；
③ 利用 Cauchy–Binet 公式把指数级的补全和转化为单一行列式系数；
④ 进一步发现该行列式在特定评估点为 Hankel 矩阵，利用高速 Hankel 行列式算法将复杂度从 Õ(n⁴k⁵) 降到 Õ(n³k⁴)。

**🔧 技术方法**

主要技术包括：
- Robinson–Schensted 对应与标准 Young 表格的随机生成；
- 递归的“左侧最小 LIS”判定与 O(n log log n) 的 LIS 长度计算；
- Hook‑length 公式与严格分割变换；
- Cauchy–Binet 行列式展开与多项式系数提取；
- Hankel 矩阵的快速行列式计算（Liu–Xin–Zhang 算法）。

**📊 数据集**

本工作为理论算法，未使用具体数据集，全部以理论复杂度分析为主。

**📈 对比分析**

与传统的暴力枚举或基于 Markov 链的近似采样相比，本算法提供了严格的均匀采样保证，并在期望时间上实现了多项式级别的可行性。对 k=Θ(n) 的直接采样器的接受率为常数，导致期望时间与 n 仅线性或线性对数相关；对一般 k，虽然复杂度较高，但仍为多项式；尚未提供实验验证。

**⚠️ 局限性**

局限性包括：
① 对一般 k 的期望时间仍为 Õ(n³k⁴)，在 n, k 较大时仍相当昂贵；
② 算法高度依赖精确整数运算与 Chinese Remainder，实际实现对大规模输入可能受限；
③ 未给出随机行列式计算的数值稳定性分析；
④ 未针对特殊 k（如 k=1、k≈n）给出更简化的实现；
⑤ 与 MCMC 等近似采样相比缺乏实验对比，无法评估实际性能与实现复杂度。

---

## 673. ArrythML: An Autoencoder-Based TinyML Approach for On-Device Arrhythmia Detection on Resource-Constrained Embedded Systems

**arXiv ID:** 2606.02256 | [PDF](https://arxiv.org/pdf/2606.02256v1)

**作者:** Nagarajan S `[一作]` (International Institute of Information Technology), Kurian Polachan `[通讯]` (International Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `109c2b71-d051-425c-831f-0c544c24280d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在可穿戴低功耗嵌入式系统上实现了基于自编码器的实时心律失常检测

**💡 创新点**

创新点在于将自编码器压缩为 INT8 量化 TinyML 模型，并通过 R–R 分割与重构误差阈值实现异常检测，同时提供对模型误判的可视化分析

**🔧 技术方法**

使用了自动编码器（全连接与卷积）、量化感知训练（QAT）和 TensorFlow Lite for Microcontrollers

**📊 数据集**

采用了 MIT‑BIH Arrhythmia Database 的 ECG 数据进行训练、验证和 95,000 条 R–R 段的评估

**📈 对比分析**

与浮点 FP32 基线和未量化模型比较，量化模型在 ESP32‑S3 上实现 180 KB/42 KB 模型、9 ms 推理延迟，精度在精细化评估后召回率 84%/82%，F1 分别 79%/72%，显示出高效且可接受的性能

**⚠️ 局限性**

限制包括单导联约束导致对某些节律（如阵发性室性早搏、左/右束支传导阻滞）的识别不足，以及对节律间隔变化的敏感性不足，且模型对噪声和标注不一致较为脆弱

---

## 674. CEON: Circular Economy Ontology Network

**arXiv ID:** 2606.02253 | [PDF](https://arxiv.org/pdf/2606.02253v1)

**作者:** Huanyu Li `[一作]` (Linköping University), Eva Blomqvist `[通讯]` (Linköping University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

开发了跨行业的圆形经济本体网络CEON，用于在产品生命周期内对资源、产品、过程和价值等进行语义化数据文档化；

**💡 创新点**

创新点在于将多行业（建筑、电子、纺织）共通概念模块化为本体网络，并提供与现有CE本体及ISO标准的语义映射，支持跨行业数据共享与互操作；

**🔧 技术方法**

采用敏捷本体工程方法Xtreme Design与MOMo结合，使用OWL 2 DL、本体设计模式、模块化设计以及SPARQL查询技术；

**📊 数据集**

未使用公开真实行业数据，主要通过构造的跨行业用例场景（建筑空间、电子设备、纺织材料）来演示和验证本体结构；

**📈 对比分析**

通过编写90个CQ并用SPARQL验证覆盖率（25完整、28部分、37未覆盖），并使用OOPS!和FOOPS!进行坑洞扫描；性能指标未进行大规模实验，仅展示了可查询的SPARQL端点；

**⚠️ 局限性**

局限在于核心本体未覆盖战略层面的CQ，部分用例专属内容留在单行业用例本体；依赖未来标准成熟与行业专家评估，且需进一步完善与行业实际数据的对齐。

---

## 675. Geometric Latent Reasoning Induces Shorter Generations in LLMs

**arXiv ID:** 2606.02248 | [PDF](https://arxiv.org/pdf/2606.02248v1)

**作者:** Shashi Kumar `[一作]` (Idiap Research Institute), Andrea Cavallaro `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种几何潜在推理方法，将大语言模型的推理过程视为在预训练的标记嵌入空间中的连续路径近似问题。

**💡 创新点**

创新点在于引入了一种轻量级的过渡头，能够预测嵌入空间中的迭代方向更新，从而实现更高效的推理，减少生成步骤的数量。

**🔧 技术方法**

使用了几何潜在推理技术，该技术通过训练一个轻量级的过渡头来预测嵌入空间的连续更新。

**📊 数据集**

使用了Open-R1 Mixture-of-Thoughts数据集的一个随机抽样的10K示例子集进行评估，该数据集提供了高质量的文本链式推理轨迹。

**📈 对比分析**

与标准的链式推理（CoT）进行比较，几何潜在推理在数学推理基准上能够在更少的生成步骤中产生正确答案，尤其是在受限的生成预算下表现更佳。

**⚠️ 局限性**

限制在于潜在步骤的数量K对推理的准确性和生成长度之间的权衡存在非单调影响，过多的潜在步骤可能导致准确性下降。

---

## 676. BlockGen: Flexible Blockwise Sequence Modeling with Hybrid Samplers

**arXiv ID:** 2606.02241 | [PDF](https://arxiv.org/pdf/2606.02241v1)

**作者:** Justin Deschenaux `[一作]` (École Polytechnique Fédérale De Lausanne), Caglar Gulcehre `[通讯]` (École Polytechnique Fédérale De Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出BlockGen框架，将单一去噪器训练为混合块大小的序列模型，兼容掩码扩散与均匀状态扩散；引入AR-informed Predictor-Corrector (ARPC)采样器，利用自回归预测重新标记并纠正低概率标记；

**💡 创新点**

创新点在于：1) 通过块大小混合训练实现模型既可做块级扩散生成，又可在同一模型内部执行自回归验证；2) ARPC在不需额外验证器的情况下，使用自回归log‑likelihood对候选令牌进行评分，提升采样质量；3) 系统性比较均匀扩散与掩码扩散在块级生成与ARPC下的性能，揭示两者的优势转折点；

**🔧 技术方法**

主要技术包括：混合块大小的ELBO训练与分层采样；基于DiT的去噪器与块因果注意力；ArPC与EIPC两种信息化校正步骤；以及温度调节与NFE（函数评估次数）对比实验；

**📊 数据集**

使用的数据集包括TinyGSM（约1180万条GSM8K式算术题）、OpenWebText、LM1B；在TinyGSM上评估GSM8K测试集准确率；在OpenWebText和LM1B上评估验证困惑度与生成困惑度；

**📈 对比分析**

在祖先采样下，均匀扩散在块级生成中优于掩码扩散；加入ARPC后，两者差距缩小，高NFE时掩码扩散反而更好；在OpenWebText上，BlockGen（掩码）实现17.5 PPL，仅比AR低0.8；在GSM8K上，ARPC在匹配NFE时超过祖先采样与EIPC，最高提升约8%；总体表明ARPC可在多块大小混合下实现近似AR的质量；

**⚠️ 局限性**

局限性包括：模型规模有限（仅170M参数），不验证更大规模是否保持同样权衡；块级训练相较于纯AR或全序列扩散成本更高；实验仅覆盖特定采样器与预算，低温度下AR仍优于ARPC，且均匀与掩码的相对排名随任务、温度与NFE变化而异。

---

## 677. Topology as Logic: Structural Role Geometry Across Formal, Software, Biological, and Prebiotic Systems

**arXiv ID:** 2606.02392 | [PDF](https://arxiv.org/pdf/2606.02392v1)

**作者:** Vladi Ivanov `[一作]` `[通讯]` (Independent Researcher), Vladi Ivanov (Independent Researcher)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过多层网络分析验证依赖拓扑与功能负载结构的相关性，探测负载逻辑节点；

**💡 创新点**

提出betweenness比degree更能捕捉负载逻辑的观测与方法，确立“负载逻辑信号”；

**🔧 技术方法**

使用IRDME框架、功能邻近定律、betweenness中心性、置换检验和统计相关度量；

**📊 数据集**

采用7种不同领域数据：4-bit ALU、ISCAS85 c432、Lean 4 mathlib4、Coq Corelib、COBOL银行代码、C. elegans–Drosophila 神经网络、CatReNet预生化网络；

**📈 对比分析**

通过预注册的Pearson/Spearman相关性与置换显著性比较，4-bit ALU betweenness r=0.77、c432 degree r=0.43、Lean r=0.78、Coq r=0.51等，显示betweenness更稳健；

**⚠️ 局限性**

受限于小样本（Coq 17、COBOL 14、预生化 13）、部分结果为后发探索，以及数字电路仅为规模复制而非跨域验证。

---

## 678. Honey, I Shrunk the Arc de Triomphe!

**arXiv ID:** 2606.02379 | [PDF](https://arxiv.org/pdf/2606.02379v1)

**作者:** Yuanbo Xiangli `[一作]` (Cornell University), Noah Snavely `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个大规模在野真实度量数据集MetricScenes，并利用其微调MoGe-2模型，从而解决单目深度/几何估计中的尺度崩塌问题。

**💡 创新点**

创新点包括：①提出结合互联网照片与立体视频的多源、真实度量数据集；②设计两阶段边缘感知泊松补全方法，兼顾背景尺度与前景细节；③通过地理标签与已知相机基线恢复绝对尺度。

**🔧 技术方法**

技术手段涵盖SfM/MVS多视图重建、地理标签对齐、相机基线标定、基于MoGe-2的泊松补全、两阶段加权Poisson求解，以及对ViT-Backbone的微调。

**📊 数据集**

使用的数据集包括MetricScenes（AerialMegaDepth、MegaScenes、Stereo4D）以及标准基准如NYUv2、KITTI、ETH3D、iBims-1、GSO、Sintel、DDAD、DIODE、Spring、HAMMER。

**📈 对比分析**

在自建测试集上与MoGe-2、DepthAnything v3、Metric3D v2等对比，WildMoGe在尺度一致性、绝对深度误差等指标显著优于基线；在标准基准上保持与MoGe-2相当甚至略优的性能。

**⚠️ 局限性**

局限性主要在于：对传统基准的轻微尺度偏差、对相机内参精度的依赖、以及仍缺少更丰富的室内场景与更精细的地面真实标注。

---

## 679. Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses

**arXiv ID:** 2606.02373 | [PDF](https://arxiv.org/pdf/2606.02373v1)

**作者:** Pengcheng Jiang `[一作]` (University of Illinois at Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种20B规模的检索搜索代理，通过强化学习在状态化搜索 harness 中训练；代理负责决定搜索、审阅、验证和停止，而环境负责维护候选池、重要性标签、证据图、验证记录及上下文预算等可恢复的状态。

**💡 创新点**

核心创新在于“状态化认知卸载”——将搜索过程中的繁重书写与管理责任交给环境，减少策略需要学习的书写负担，从而实现更高效、可迁移的检索行为；同时引入了重要性标签、证据图压缩、自动种子化和多样性奖励等机制，提升检索质量。

**🔧 技术方法**

技术包括：大型语言模型（20B）与 Tinker RL 框架、SFT 与 CISPO on-policy RL、状态渲染与压缩（BM25句子压缩、MinHash dedup）、证据图构建（正则提取实体）、自动种子化、重要性标签、工具多样性奖励、奖励设计（覆盖率、答案支持、工具多样性）。

**📊 数据集**

使用了八个检索基准：BrowseComp+、Web synthetic、USPTO patents、SEC filings、LongSealQA、Seal0QA、FRAMES、HotpotQA；SFT 训练数据来自 BC+、Web、Patents、SEC 四个源族；RL 训练仅在 SEC 数据上。

**📈 对比分析**

与多种开源模型（Qwen3‑32B、Context‑1、Search‑R1‑32B、Tongyi DeepResearch 30B）以及前沿模型（GPT‑5.4、Sonnet‑4.6、Opus‑4.6、Kimi‑K2.5、GPT‑OSS‑120B）进行对比；平均 curated recall 达 0.730，优于下一强模型 Tongyi DeepResearch 30B +11.4 分，在八个基准上保持竞争力；在离谱转移任务上表现尤为突出。

**⚠️ 局限性**

局限性包括：依赖手工设计的状态接口和奖励机制，若接口不当可能导致学习偏差；对证据图的正则提取过于简化，缺乏深度语义链接；缺少对精确度（precision）或 F1 的深入分析；在极大规模检索场景下的实时性能与资源消耗未充分评估。

---

## 680. COMAP: Co-Evolving World Models and Agent Policies for LLM Agents

**arXiv ID:** 2606.02372 | [PDF](https://arxiv.org/pdf/2606.02372v1)

**作者:** Youwei Liu `[一作]` (Central South University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8d10c613-917e-4880-9716-17789f50e119` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 CoMap 框架，让语言模型代理在闭环中共同演化文本世界模型与策略，通过未来感知反思和自蒸馏来实现自我提升。

**💡 创新点**

创新点在于将文本世界模型与策略视为互相耦合的进化过程，首次引入未来感知反思与基于真实转移的自蒸馏实现两者的闭环强化。

**🔧 技术方法**

采用大型语言模型（如 Qwen3‑4B/8B、GPT‑5.4 等）作为策略与世界模型的主体，并实现一阶前瞻预测、反思门控、学生-教师自蒸馏等技术。

**📊 数据集**

实验使用 ALFWorld、ScienceWorld、WebShop 与 StableToolBench 四大基准数据集，涵盖家居规划、科学推理、网页导航与工具调用等任务。

**📈 对比分析**

与 ReAct、Imagine‑and‑Act、ITP_R 等现有基线相比，CoMap 在 Qwen3‑4B 上平均提升约 16.75%（相对），在 Qwen3‑8B 上提升约 2.58%，并在多项任务上超越同等规模的 API 调用模型。

**⚠️ 局限性**

局限性包括：仅适用于文本可表示的状态与动作；推理阶段需额外一次世界模型调用，增加推理延迟；以及对多模态环境与高风险场景的适用性尚未验证。

---

## 681. Minimax-Optimal Policy Regret in Partially Observable Markov Games

**arXiv ID:** 2606.02363 | [PDF](https://arxiv.org/pdf/2606.02363v1)

**作者:** Raman Arora `[一作]` `[通讯]` (Johns Hopkins University), Raman Arora (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对部分可观测马尔可夫博弈（POMG）中自适应对手的策略回退（policy regret）最小化框架，并给出了一个基于 epoch 的乐观最大似然算法。

**💡 创新点**

创新点在于：① 将观测可观测算子模型（OOM）与因果分解结合，独立处理世界动力学和对手聚合；② 证明在 Posterior-Lipschitz 对手和弱可揭示性假设下，策略回退可以实现最优的 √(T) 速率；③ 提供了与之匹配的 Ω(√(d_E T)) 下界，展示了结果的最优性。

**🔧 技术方法**

使用了观测可观测算子框架、Eluder 维数理论、乐观最大似然估计、Kullback–Leibler 损失下的预测误差分析以及对手记忆的后验-Lipschitz 传输成本估计。

**📊 数据集**

论文为理论分析，未使用实际数据集；仅在仿真式的 tabular POMG 以及低秩/线性模型中说明了维数上界。

**📈 对比分析**

通过与传统外部回退（external regret）方法对比，本文的策略回退在自适应对手环境中实现了 √(T) 的上界，且下界匹配，证明其优越性；在有限状态/动作实例中给出具体的复杂度表达式。

**⚠️ 局限性**

局限性包括：① 对手记忆需满足 Posterior-Lipschitz 条件，且假设对手策略光滑；② 需要完整的模型类可实现最大似然和乐观规划，计算复杂度高；③ 仅适用于可观测算子可分解的 POMG，且对持续记忆长度的扩展仍需进一步研究。

---

## 682. MOC: Multi-Order Communication in LLM-based Multi-Agent Systems

**arXiv ID:** 2606.02359 | [PDF](https://arxiv.org/pdf/2606.02359v1)

**作者:** Yao Guan `[一作]` (Fudan University), Qiang Duan `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多阶通信（MOC）方案，用于LLM多智能体系统，扩展证据接收范围并通过结构化合并实现高效通信。

**💡 创新点**

首次将多跳祖先消息结构化呈现，并引入语义-拓扑合并算法压缩上下文，以解决传统一阶邻居信息限制的问题。

**🔧 技术方法**

使用图模型与DAG执行顺序、多阶信息流构建、语义嵌入+余弦相似度、轻量级蒸馏模型进行信息合并以及批量近似合并等技术。

**📊 数据集**

在MMLU、MMLU-Pro、GSM8K、SVAMP、AQuA、HumanEval等六大数据集上进行实验。

**📈 对比分析**

与单体LLM和Vanilla MAS在相同拓扑下对比，MOC在Gemma‑2‑27B和Qwen2.5‑32B等模型上平均提升约2–4%，在稀疏图上提升更显著，并在输入token使用上减少约10–20%。

**⚠️ 局限性**

K阶数固定，未自适应选择；可能引入错误或恶意信息；在稠密图中冗余削减效果有限。

---

## 683. VEDAL: Variational Error-Driven Asynchronous Learning for 3D Gaussian Splatting Pruning

**arXiv ID:** 2606.02346 | [PDF](https://arxiv.org/pdf/2606.02346v1)

**作者:** Aoduo Li `[一作]` (Guangdong University Of Technology), Xuhang Chen `[通讯]` (Huizhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于变分推断的异步错误驱动剪枝框架 VEDAL，用于 3D Gaussian Splatting 的模型压缩。

**💡 创新点**

创新点在于引入预测误差门控异步激活 KL 正则和变分不确定性头，使剪枝时机随每个高斯的收敛稳定性自动决定。

**🔧 技术方法**

采用了变分 ELBO、指数移动平均误差估计、Bernoulli 变分头、Binary Concrete 采样以及近似留一误差计算等技术。

**📊 数据集**

实验使用了 Mip-NeRF 360、Tanks&Temples、Deep Blending 三个基准数据集。

**📈 对比分析**

与 LightGaussian、Compact3D、Mini-Splatting、PUP 3D-GS、MaskGaussian 等基线对比，能够实现 5.2× 的高斯压缩率，PSNR 仅下降 0.31 dB，保持 185 FPS，优于同类方法。

**⚠️ 局限性**

局限性包括对高斯相互耦合的近似为加性，可能导致少量共剪；门控阈值与 warmup 等超参数仍需手动调节，且对动态场景的适用性尚未验证。

---

## 684. I-(OT)^2: A Client-optimal Oblivious Transfer Protocol for IoT Devices

**arXiv ID:** 2606.02344 | [PDF](https://arxiv.org/pdf/2606.02344v1)

**作者:** Elia Onofri `[一作]` (King Abdullah University of Science and Technology), Roberto Di Pietro `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向IoT设备的轻量级 1‑out‑of‑2 基础OT协议，利用平方剩余问题实现接收端极低的计算与通信成本。

**💡 创新点**

创新点包括：1）基于QR的安全性，将接收方计算几乎完全移交给发送方；2）只需两轮交互，通信量仅为6个环元素＋4个摘要；3）支持离线预计算，使在线阶段仅做一次哈希与一次XOR；4）提供恶意发送方检测的概率游戏，确保模数合法性。

**🔧 技术方法**

核心技术：模数n=p·q（p,q≡1(mod 4)）的平方剩余；关键派生函数F_s(k)=H(s||k)（SHAKE‑256）；哈希函数H(k)用于根辨识；离线阶段的Tonelli‑Shanks求根与CRT组合。

**📊 数据集**

实验使用随机生成的消息和随机选择的k、b；未使用公开数据集，所有测试均在自定义随机数据上完成。

**📈 对比分析**

在Raspberry Pi Zero 2W与桌面CPU上对λ∈{1024,2048,3072,4096}进行基准；3072‑bit模数时，接收端在线时间仅为39.90 µs，桌面为2.80 µs；与SimplestOT相比，IoT设备上实现了约10×更快的在线性能，整体传输时间下降至1/3以上。

**⚠️ 局限性**

局限性：仅适用于基础OT；不具备量子抗性；不支持批量或自适应OT；需要对发送方的模数合法性进行额外验证；在高并发场景下发送方计算负载仍显著。

---

## 685. Parameter-efficient Dual-encoder Architecture with Differentiable Choquet Integral Fusion for Underwater Acoustic Classification

**arXiv ID:** 2606.02341 | [PDF](https://arxiv.org/pdf/2606.02341v1)

**作者:** Amirmohammad Mohammadi `[一作]` (Texas A&M University), Alexandra Van Dine `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种参数高效的双编码器架构，能够同时处理原始波形和声谱图进行水下声学分类。

**💡 创新点**

创新点在于引入可微分的Choquet积分融合层，结合soft‑sort门控实现动态加权；并在冻结的预训练骨干中嵌入共享权重的参数高效微调模块。

**🔧 技术方法**

使用的技术包括双分支神经网络（波形编码器+声谱图编码器）、可微分Choquet积分融合、soft‑sort门控、参数高效微调（LoRA/PEFT）、交叉熵多目标损失。

**📊 数据集**

实验数据集为 DeepShip 与 ShipsEar 两个公开水下船舶声学数据集。

**📈 对比分析**

与单一编码器基线比较，双编码器+Choquet融合在准确率上提升约1–2个百分点，同时训练参数仅为单模型的一小部分，并通过多目标损失平衡单分支与融合性能。

**⚠️ 局限性**

局限性包括：尚未评估实时推理时的计算开销和硬件效率；对极端噪声或不同海域的泛化性仍需进一步验证；Choquet积分参数的解释性虽有提升，但难以完全揭示所有决策依据。

---

## 686. Forget Attention: Importance-Aware Attention Is All You Need

**arXiv ID:** 2606.02332 | [PDF](https://arxiv.org/pdf/2606.02332v1)

**作者:** Soohyeong Shin `[一作]` (Kangwon National University), Yeongwook Yang `[通讯]` (Kangwon National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SISA，一种在注意力得分层面融合状态空间模型（SSM）重要性信号的Transformer架构，利用Q/K增强实现单一SDPA调用；

**💡 创新点**

创新点在于首次将向量化、数据依赖的SSM得分直接注入注意力得分，开启了“score‑level fusion”这一新的混合设计轴；

**🔧 技术方法**

采用了Mamba-3衍生的SSM通道、累计衰减与旋转编码、RoPE位置编码、FlashAttention兼容的SDPA、以及可调的d_s维度；

**📊 数据集**

使用了5B Token的SlimPajama-6B文本数据进行训练，并在LAMBADA、NIAH、HellaSwag、ARC‑Easy、WinoGrande等多项基准上评估；

**📈 对比分析**

通过与Transformer、Mamba‑2、Mamba‑3在相同参数规模下对比，SISA在152M规模下LAMBADA提升3.4pp、NIAH始终100%、训练吞吐率比Mamba‑3快25%，但在369M规模下仍逊色于Mamba‑3的LAMBADA；

**⚠️ 局限性**

局限性包括仅在5B训练数据、未达到Chinchilla最优tokens/param比例、softmax归一化导致重要性信号稀释、RoPE最大长度限制为2048、以及相对于纯Transformer仍有约39%吞吐率下降。

---

## 687. Hallucination-Aware Diffusion Sampling for Inverse Problems via Robust Prior Updates

**arXiv ID:** 2606.02331 | [PDF](https://arxiv.org/pdf/2606.02331v1)

**作者:** Pengfei Jin `[一作]` (Massachusetts General Hospital and Harvard Medical School), Quanzheng Li `[通讯]` (Massachusetts General Hospital and Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了扩散模型在逆问题中的测量条件幻觉，并提出了一种鲁棒的先验更新模块（RPU）来提升实例可信度。

**💡 创新点**

创新点在于将逆问题求解视为先验更新与测量一致性交替优化，并通过局部探测并重新锚定先验更新来抑制幻觉。

**🔧 技术方法**

采用扩散模型的后向采样与Bayes规则分解的先验-测量双向优化，结合RPU的局部探测步骤实现。

**📊 数据集**

实验使用FFHQ（人脸）和ImageNet（通用图像）数据集的盒子填补、Gaussian模糊和运动模糊任务。

**📈 对比分析**

与原始DPS相比，RPU在FFHQ上在PSNR、LPIPS和人类可信度评价上都有提升，ImageNet上也表现出更高的PSNR和更好的人类偏好。

**⚠️ 局限性**

局限性包括仅在DPS实例上验证，幻觉检测不完善，且在模糊等相对受限任务中人类评价差异有限。

---

## 688. Riemannian Gradient Descent for Low-Rank Architectures

**arXiv ID:** 2606.02328 | [PDF](https://arxiv.org/pdf/2606.02328v1)

**作者:** Nicholas Knight `[一作]` `[通讯]` (NVIDIA), Nicholas Knight (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将多头注意力的权重参数写成低秩矩阵 ABᵀ，并在其秩受限子流形上直接进行黎曼几何梯度下降，从而在训练过程中仅更新秩受限矩阵。

**💡 创新点**

系统探索了十种黎曼几何设计（两种固定秩几何、三种部分等距几何及其网格化块矩阵变体），并给出相应的投影重吸收、向量传输以及动量归一化的实现细节，构建了一套完整的低秩黎曼优化框架。

**🔧 技术方法**

采用黎曼梯度下降、投影重吸收、向量传输、归一化、动量，并结合因子梯度的恢复与多重几何等价映射，实现了在低秩矩阵空间上的迭代更新。

**📊 数据集**

在 512 维 decoder‑only GPT‑2 变体模型上使用 FineWeb “sample‑10BT” 数据集进行训练，模型包含 6 层、8 个注意力头，且实验以 512 长度的子序列为单位。

**📈 对比分析**

与 AdamW 基线相比，经过学习率调优后低秩黎曼优化器在训练/验证损失上基本与 AdamW 相当，偶有轻微优势但不显著；然而，其算法复杂度远高于 AdamW，且在大规模训练时尚未显示出显著优势。

**⚠️ 局限性**

主要局限包括：模型规模较小、缺少学习率退火与时间/能耗评估、未对 AdamW 进行参数调优、对梯度一致性与重投影精度缺乏理论分析，且实验结果受随机种子波动影响较大。

---

## 689. Strategies for Molecular Dynamics using Hybrid Systems: LAMMPS Use Case

**arXiv ID:** 2606.02319 | [PDF](https://arxiv.org/pdf/2606.02319v1)

**作者:** Paulo Henrique Leme Ramalho `[一作]` (Universidade São Francisco), Fábio Andrijauskas `[通讯]` (Universidade São Francisco)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多节点 AMD EPYC 计算平台上，系统性评估了 LAMMPS 粗粒度（SPICA）生物分子模拟（Tritrpticin 1D6X）在纯 MPI、MPI+OpenMP 混合模式以及纯 OpenMP 三种并行化方案下的执行时间、加速、并行效率与内部时间分解，探究不同并行粒度对可扩展性的影响。

**💡 创新点**

首次在 NUMA 体系结构上对 LAMMPS 粗粒度工作负载进行细粒度可扩展性分析；揭示纯 MPI 在多节点时因 PPPM+Comm 占比激增而导致性能急剧下降；证明 MPI+OpenMP 通过降低每节点 MPI 进程数、减少跨节点通信并充分利用 CCD‑local L3 缓存，可在大规模节点数下保持接近单节点的性能；同时通过内部时间分解精准定位瓶颈。

**🔧 技术方法**

使用 LAMMPS 30Mar2026（启用 CG‑SPICA、KSPACE、OPENMP 包）、MPI/OpenMPI‑UCX、OpenMP、GCC 8.5.0、CMake 3.31.8；实验平台为 AMD EPYC 7662 芯片（128 核/512 GB 内存，8‑通道 DDR4‑3200，256 MB L3 分布于 8 个 CCD），节点间通过 InfiniBand 互连，作业调度使用 PBS；分析工具包括 LAMMPS 内部时间分解、time 工具、Python 验证脚本。

**📊 数据集**

基准工作负载为抗菌肽 Tritrpticin（PDB ID: 1D6X）嵌入 DOPC 双层脂质膜的粗粒度模型，总计 4354 个 CG 颗粒；模拟采用 5000 步 NPT 生产阶段（10 fs 步长，15 Å 近场截断，10⁻⁵ PPPM 容差）。

**📈 对比分析**

采用三次独立种子（111、222、333）进行重复实验，测量总执行时间、加速比与并行效率；比较纯 MPI（128×1、256×1、512×1、1024×1）、MPI+OpenMP（不同 MPI×OpenMP 组合）以及纯 OpenMP（1×128）。结果显示：单节点纯 MPI（128×1）最快；纯 MPI 在多节点显著退化（1024×1 时 Kspace+Comm 占比约 89%）；MPI+OpenMP 在 4、8 节点时保持 70% 左右的通信+电荷占比，可在 1024 核时获得 4.51 s 的循环时间，几乎等同于单节点最快值；纯 OpenMP 由于缺乏 MPI 空间分解导致 Pair 占比 65%，性能差。

**⚠️ 局限性**

实验受限于小至中等规模（≤ 4354 颗粒）粗粒度系统，MPI 进程数过多导致通信/同步成本占比激增，无法突破；未评估更大体系、GPU 加速或机器学习势能的影响；实验仅在 AMD EPYC NUMA 架构上完成，结果对其他 CPU/架构的迁移性需进一步验证。

---

## 690. Attention Dynamics and Adaptive Decision Support in C5ISR: A Recurrence Quantification Analysis of Visual and Multimodal Attention Guidance Effects on Mission Performance

**arXiv ID:** 2606.02382 | [PDF](https://arxiv.org/pdf/2606.02382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 691. Towards Precise Intent-Aligned VLA Aerial Navigation via Expert-Guided GRPO

**arXiv ID:** 2606.02313 | [PDF](https://arxiv.org/pdf/2606.02313v1)

**作者:** Tianyang Chen `[一作]` (Zhejiang University), Fei Gao `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了一种基于视觉-语言-动作（VLA）模型的无人机（UAV）导航强化学习框架，并通过专家引导的群组相对策略优化（EG‑GRPO）与异构并行仿真‑推理流水线提升飞行指令对齐与任务成功率。

**💡 创新点**

创新点包括：①将少量专家轨迹直接注入在线 RL 训练中，解决高维连续空间下的稀疏奖励与探索停滞；②设计群组相对优势计算保证梯度信号非退化；③构建双缓冲异构并行仿真‑推理管道，将仿真（RT‑core GPU）与 VLA 推理（compute GPU）解耦，缩短 43.5% 的 rollout 时间。

**🔧 技术方法**

技术实现主要使用：①VLA 模型（OpenVLA‑OFT）作为基线；②Proximal Policy Optimization（PPO）改进的 Group Relative Policy Optimization（GRPO）与 EG‑GRPO；③基于 LLM 的指令条件奖励模型；④Isaac Lab 物理仿真平台；⑤Ray + SSH 双缓冲并行调度。

**📊 数据集**

数据集：①UAV‑flow 开源航迹数据用于监督预训练；②由规则生成器合成的少量专家轨迹；③在 Isaac Lab 生成的多场景高保真仿真环境进行在线 roll‑out。

**📈 对比分析**

与基线（SFT OpenVLA‑OFT）及其他 VLA 模型（π_0）进行对比：在易/难任务组上，成功率从 26.1% 提升至 55.6%（提升 29.5%），意图对齐得分（IAS）从 4.50 提升至 7.24（提升 60.9%）。同时与传统 GRPO 相比，EG‑GRPO 在难任务上成功率提升 16.7%，IAS 提升 1.54；并行流水线使每步 rollout 时间由 904.67s 减少至 511.01s。

**⚠️ 局限性**

局限性：①对专家轨迹的依赖仍存在，若专家样本不足可能影响优化；②奖励模型基于 LLM，需保证其评估准确性，否则梯度方向可能误导；③虽然异构并行显著提升效率，但在多节点部署时仍需网络同步与资源调度；④在真实环境中零样本迁移已表现良好，但极端天气或动态障碍等更复杂场景尚未充分验证。

---

## 692. Policy and World Modeling Co-Training for Language Agents

**arXiv ID:** 2606.02388 | [PDF](https://arxiv.org/pdf/2606.02388v1)

**作者:** Ning Lu `[一作]` (Southern University of Science and Technology), Ke Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在大语言模型代理上实现策略与世界建模联合训练的框架，通过在标准RL回放中添加下一观测的预测损失来提升代理的决策能力。

**💡 创新点**

创新点在于：①直接利用RL回放中的动作-观测对作为世界建模监督；②引入基于动作熵的转移筛选、截断MAE损失以及奖励自适应的损失权重，使监督信息更具信息量且更稳健；③不需要额外模拟器、额外训练阶段或推理时的规划。

**🔧 技术方法**

主要技术包括：基于自回归语言模型的策略学习；在同一模型上同时优化RL策略损失与下一观测预测损失；动作熵筛选机制；截断MAE（CMAE）观察预测损失；奖励自适应的WM权重系数。

**📊 数据集**

实验使用的公开数据集包括 ALFWorld、WebShop、以及多轮检索增强QA任务（NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle）。

**📈 对比分析**

与GRPO、GIGPO、PPO、RLOO等主流RL基线以及闭源大模型和提示式代理进行对比；在ALFWorld、WebShop和QA任务上均获得显著提升，例如 ALFWorld 上 GRPO 成功率提升至 77.9%（+7.9%），WebShop 上 GIGPO 成功率提升至 68.6%（+9.1%），并在QA任务上平均得分提升 0.9–1.7%。

**⚠️ 局限性**

局限性在于：①仅使用一跳的下一观测监督，无法捕捉长程依赖或多步误差累积；②监督来自原始回放，未做轨迹去重或多样性筛选，可能导致监督偏向频繁模式。

---

## 693. A Game-Theoretic Decision Framework for Optimal Selection of Coordination Detection Methods in Multi-UAV Fleet Operations

**arXiv ID:** 2606.02383 | [PDF](https://arxiv.org/pdf/2606.02383v1)

**作者:** Christian Manasseh `[一作]` `[通讯]` (Mobius Logic Inc.), Christian Manasseh (Mobius Logic Inc.)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于博弈论的决策框架，用以在不确定的 UTM 交通场景下自动选择最佳的协同检测算法，并给出对应的混合策略；

**💡 创新点**

创新点在于将方法选择建模为两人零和博弈，得到鲁棒的最小化最大化混合策略，并结合多目标 Pareto 优化生成完整的性能曲线；

**🔧 技术方法**

使用了博弈论、蒙特卡洛敏感性分析、NSGA‑II 进化多目标优化、以及八种协同检测算法（如 CRQA、Koopman/DMD、Koopman Phase、mrDMD、DMDc、Physics Baseline、Graph SP、Streaming DMD）；

**📊 数据集**

使用的是由 200 次随机生成的模拟 UAV 路线数据集，涵盖不同飞行员、车队规模、协调窗口长度和观测时长的组合；

**📈 对比分析**

通过对八种方法在 200+ 场景中的 F1 坐标、F1 路线主导和实时倍数进行统计，发现最优混合策略在不同优先级下分别偏向 Koopman Phase、CRQA 或 mrDMD，均实现了 0.29–0.53 的游戏价值；

**⚠️ 局限性**

局限性包括：仅考虑单一协调模型（路线领航/跟随者）；假设观测窗口内场景不变；蒙特卡洛样本和参数网格可能未覆盖所有性能极值；以及对真实 UTM 复杂动态交互的适应性需进一步验证。

---

## 694. When Do Attention Circuits Form? Developmental Trajectories of Capability and Attention-Sink Emergence Across Three 1B-ClassArchitectures

**arXiv ID:** 2606.02378 | [PDF](https://arxiv.org/pdf/2606.02378v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在三种1B级语言模型中，对注意力头的形成轨迹进行了跟踪，系统记录了诱导头、先前词头和BOS吸引头的出现时机。

**💡 创新点**

创新之处在于揭示了诱导阶段与注意力吸收阶段是两次独立的训练转折点，并证明它们在不同数据与架构组合下的形成时间与形状均有显著差异；同时发现L0/L1层始终不出现BOS头的结构性“零底”。

**🔧 技术方法**

技术方法包括：利用参与度比（Participation Ratio）作为谱信号来预识别潜在头；对合成诱导批次计算多类别选择性屏蔽，筛选出功能性头；采用组消融验证功能；以及对每层每头的PR和选择性随训练进度的定量跟踪。

**📊 数据集**

使用的数据集为：Pythia 1B训练于The Pile；OLMo 1B与OLMoE 1B-7B训练于DCLM；这三种模型覆盖了稠密Transformer与Mixture-of-Experts两大架构。

**📈 对比分析**

通过在每个模型上取10个对数间隔的检查点，比较诱导头与BOS头的出现百分比及比例曲线，结果显示诱导电路在约20–25B tokens完成，而BOS吸引器在260–420B tokens时才达到50%；能力筛选在1%训练量即可恢复约两倍半的最终诱导电路。

**⚠️ 局限性**

局限性包括：检查点间隔粗糙，可能掩盖更细粒度的转折；每个模型仅用单一随机种子；功能筛选基于合成批次，未验证在自然文本上的一致性；MoE模型前向传播成本高，导致检查点数量受限；以及未对多头之间的交互和更细粒度的能力类别进行深入探究。

---

## 695. Layered Ego Networks in Email Communication: From Enron to the Jmail Archive

**arXiv ID:** 2606.02376 | [PDF](https://arxiv.org/pdf/2606.02376v1)

**作者:** Francesco Di Cursi `[一作]` (Institute of Informatics and Telematics), Andrea Passarella `[通讯]` (Institute of Informatics and Telematics)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过分析 Enron 与 Jmail 两个邮件档案，探讨电子邮件中自我网络（ego network）的分层结构是否符合 Dunbar 模型；

**💡 创新点**

创新点在于首次将 Dunbar 层次模型应用于电子邮件数据，并提出在高频率邮件档案（如 Jmail）中对频率阈值进行动态缩放以恢复可解释层次结构；

**🔧 技术方法**

采用 MeanShift 聚类提取经验层次、Dunbar 参考频率阈值分层、以及双向通信相关性检验等技术；

**📊 数据集**

使用的两大数据集为 Enron（约 150 名员工、约 5 万封邮件）和 Jmail（单一活跃主角、约 4,299 名联系人、约 3 十万封邮件）；

**📈 对比分析**

比较方法为：先用原始阈值得到层次，再对 Jmail 进行频率缩放后再比较；结果显示缩放后层次稳定、层次尺寸符合 Dunbar 预期，且双向通信相关性高，证明方法有效；

**⚠️ 局限性**

局限性包括仅检验两个档案，Jmail 为极端高频率样本不一定代表一般情况；方法未考虑邮件内容与语义；对大规模或不同平台的推广仍需进一步验证。

---

## 696. WAXAL-NET: Finetuned Edge ASR Across 19 African Languages

**arXiv ID:** 2606.02375 | [PDF](https://arxiv.org/pdf/2606.02375v1)

**作者:** Victor Tolulope Olufemi `[一作]` (CMU Africa), Prasenjit Mitra `[通讯]` (CMU Africa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在19种非洲语言的对话语音数据集WAXAL上，训练并评估了三种细粒度的边缘ASR模型（Whisper Tiny、Whisper Small、MMS‑300M），并将其与三种多语种基础模型（Whisper Large‑v3、MMS‑1B、Omnilingual‑1B）的零射（zero‑shot）性能进行对比。

**💡 创新点**

① 领域专业化（domain specialization）显著优于规模（scale），细化模型在同一语音域上可实现近27个百分点的WER下降；② 架构与语言典型性（CTC vs AR 与语言族群）的匹配关系被系统性验证；③ 对于音节字母表（Ge'ez）语言，WER往往低估了字符级准确率，提出CER/WER比率作为补充评估指标；④ 公开完整的细化模型权重、训练脚本和清洗后的WAXAL子集。

**🔧 技术方法**

细化训练（full fine‑tuning 或冻结编码器的参数高效微调）、CTC与自回归解码器对比、数据清洗（时长与语速阈值）、分布式本地评审（native‑speaker audit）以及跨域（WAXAL→FLEURS）评估。

**📊 数据集**

主要使用WAXAL语料库（19种非洲语言的对话式语音，约2,279小时），并将FLEURS作为对外语域的跨域评估基准。

**📈 对比分析**

通过与基础模型的零射基线对照，细化边缘模型在宏观平均WER上从64.9%降至38.0%（≈27个百分点），模型参数量仅为基础模型的1/3–1/40；在FLEURS上，基础模型优势恢复，说明领域匹配是性能的主要驱动因素；在细化模型中，MMS‑300M在Bantu语族上领先，Whisper Small在阿非罗‑阿西亚语族上更优。

**⚠️ 局限性**

① 语料中说话人多样性不足，部分语言仅覆盖18–25位说话人；② 本地评审样本量有限（每种语言仅40句），缺乏统计显著性；③ WER对口音、停顿和代码切换的惩罚过重，可能低估实用可懂度；④ 仅验证了19种语言，未覆盖更广泛的非洲方言与口音；⑤ 结果主要基于单一评估框架，未考虑更多语音质量指标。

---

## 697. Spatial Representation Learning Beyond Pixels: Unifying Raster Data and Vector Semantics for Human-Centric Geospatial Foundation Models

**arXiv ID:** 2606.02374 | [PDF](https://arxiv.org/pdf/2606.02374v1)

**作者:** Steffen Knoblauch `[一作]` (Heidelberg University), WenWen Li `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了将栅格遥感影像与矢量数据统一学习的空间表示学习框架。

**💡 创新点**

首次强调栅格与矢量之间的联合自监督预训练和跨模态注意力，提出了多尺度、结构保留的统一表示。

**🔧 技术方法**

使用了自监督掩码自编码器、对比学习、跨模态Transformer、S2Geometry索引、S2Vec、Poly2Vec等技术。

**📊 数据集**

参考了卫星影像数据集（Sentinel‑2、Landsat、PlanetScope 等）和公开矢量数据集（OpenStreetMap、Overture、国家统计数据）进行讨论，未给出具体实验集。

**📈 对比分析**

文章未开展实验对比，因而没有定量性能结果；作者指出需要统一评测基准来验证该框架。

**⚠️ 局限性**

存在模态对齐难度大、信息损失、尺度和区域不平衡、缺乏统一基准、模型复杂度高以及公平性与解释性的挑战。

---

## 698. Certified Closed-Loop Control for Packet Networks: A Compositional Certification Framework

**arXiv ID:** 2606.02368 | [PDF](https://arxiv.org/pdf/2606.02368v1)

**作者:** Muhammad Bilal `[一作]` (Lancaster University), Huaming Wu `[通讯]` (Tianjin University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一个在每个控制周期对调度动作进行在线校验与修正的“Certified Operator”框架，确保排队稳定、尾延迟和安全性，并通过导出包络实现模块间的合同化组合。

**💡 创新点**

将调度控制与安全过滤器分离，提出基于到达包络与后效跟踪的证书编译方法；给出面向DAG与循环网络的组合安全与稳定性证明；引入小增益闭环诊断和紧急回退策略。

**🔧 技术方法**

离散时间队列模型、服务与到达包络、障碍函数与Foster–Lyapunov 驻留性、凸投影与QP求解、线性规划约束、网络计算包络导出、闭环闭包小增益理论、Python实验平台与快速投影实现。

**📊 数据集**

实验采用自定义的字节级闭环模拟，使用随机、学习和对抗式proposer产生的流量数据；未使用公开的真实网络数据集。

**📈 对比分析**

与直接执行、监控、启发式裁剪三种控制器对比。实验显示：在恶劣提议下，Certified Operator显著降低 p99 / p99.9 尾延迟并恢复吞吐量；在正常提议下几乎不影响性能；对延迟计量、超载和包络误差均能保持可观测的安全性与稳定性；投影计算时间在 1 ms 控制周期下保持数十微秒，满足实时需求。

**⚠️ 局限性**

需要先验的到达包络和后效跟踪参数 κ_min；对计数误差与计时偏差的鲁棒性有限；包络不准确时会导致频繁违约；紧急回退不保证满足原始目标；实验仅在 Python 模拟层验证，未在 Linux 或硬件交换机上验证真实追踪误差；证明仅覆盖单资源瓶颈/线性约束，复杂多资源场景需进一步扩展。

---

## 699. CHIMERA: A Flexible and Scalable 3.1 TOPS/W AI-MCU with Transformer Accelerator and 563 Gb/s Shared-L2 Memory Subsystem with QoS Guarantees

**arXiv ID:** 2606.02358 | [PDF](https://arxiv.org/pdf/2606.02358v1)

**作者:** Lorenzo Leone `[一作]` (ETH Zürich), Luca Benini `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一款名为Chimera的灵活可扩展AI-MCU，集成Transformer加速器、九个RV32IMA通用核心及共享高带宽L2内存岛，实现了低功耗边缘设备的实时Transformer推理。

**💡 创新点**

核心创新点包括：①面向Transformer的高能效加速器与可编程核心的紧耦合设计；②采用双宽度宽带AXI4接口与双互插宽带银行的L2内存岛，提供563 Gb/s总带宽；③支持QoS感知仲裁策略，保障延迟关键访问实现34周期worst‑case延迟，平均延迟降低16倍。

**🔧 技术方法**

技术实现涵盖：22 nm FDX工艺、RISC‑V RV32IMA核、AXI4总线、双缓冲权重内存、8‑bit整数量化、64‑路点乘PE、软最大/激活单元、固定与边界优先仲裁、时钟域交叉与软件可控时钟门控。

**📊 数据集**

评估使用的模型与数据集包括：MobileBERT、Whisper‑Tiny Encoder、DINOv2‑S以及矩阵乘法基准；在硅片上通过JTAG、UART与可编程电源进行测量，覆盖推理吞吐、能效与延迟指标。

**📈 对比分析**

通过与同级技术节点的SOA加速器、AI‑MCU以及现有Transformer加速器的对比，Chimera在峰值能效上实现3.1 TOPS/W（相比基准提升×），面积效率提升×，并在高速宽带场景下维持7%能效下降，吞吐量最高可达896 GOPS，功耗600 mW；在QoS评测中延迟降低16×。

**⚠️ 局限性**

限制方面：仅针对推理阶段，未覆盖大规模Transformer训练；加速器采用8‑bit量化，可能在极端稀疏或高精度需求模型中精度下降；当前实现聚焦单芯片，跨芯片或多核扩展仍需进一步研究。

---

## 700. Do Multimodal Agents Really Benefit from Tool Use? A Systematic Study of Capability Gains

**arXiv ID:** 2606.02357 | [PDF](https://arxiv.org/pdf/2606.02357v1)

**作者:** Garvin Guo `[一作]` (University Of Chinese Academy Of Sciences), Minpeng Liao `[通讯]` (Tongyi Lab, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对“思考与图像”多模态代理的工具调用进行细致评估，比较了工具开启与工具禁用、纯文本推理器三种设置，并通过过程级归因与轨迹消融分析探究工具调用是否真正扩展了可解题集。

**💡 创新点**

创新点在于将工具使用评估从简单的分数提升转化为可解题集扩展、过程级归因与工具调用方式消融三维度，揭示工具调用往往为冗余确认而非必要信息补充。

**🔧 技术方法**

采用工具调用、多模态推理、图像处理、OCR、代码执行等技术，并构建工具调用轨迹格式与结果消融实验。

**📊 数据集**

使用来自 DeepEyesV2 训练池的图像与文本样本，以及四类公开基准（真实世界理解、OCR、图表理解、数学推理）。

**📈 对比分析**

对比方法包括工具开启、工具禁用、纯文本推理器、工具格式仅/结果仅消融；实验显示工具开启对准确率与 token 成本影响不显著，工具仅解题集占比极低，且在多任务中两种消融方式均未提供持续优势。

**⚠️ 局限性**

限制在于仅评估两款代理与四类任务，纯文本推理器并非完整对照，过程归因依赖大型模型判断，且未涵盖多轮交互或用户感知等实际应用场景。

---

## 701. TROPHIES: Temporal Reconstruction of Places, Humans, and Cameras from Multi-view Videos

**arXiv ID:** 2606.02350 | [PDF](https://arxiv.org/pdf/2606.02350v1)

**作者:** Jinpeng Liu `[一作]` (National University Of Singapore), Xingyu Liu `[通讯]` (National University Of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的框架，用多视角视频同步重建动态人体、静态场景和相机轨迹，并在统一的世界坐标系下实现三者的全局一致性。

**💡 创新点**

创新点在于：①将人体分支、场景分支与全局对齐优化三者紧耦合，消除尺度漂移和坐标对齐问题；②引入人类感知注意力机制和多视角 Transformer，实现动态区域抑制和跨视角几何一致性；③在全局优化中加入接触约束和重投影束平整，实现物理可行的 4D 重建。

**🔧 技术方法**

采用了多视角 Transformer、Sim(3) 对齐、全局 Bundle Adjustment、接触感知优化，以及可插拔的场景分支（如 DUSt3R、MonST3R、CUT3R）和人体分支（SMPL 估计）。

**📊 数据集**

使用 EgoHumans 和 EgoExo4D 两大多视角数据集进行实验，分别覆盖室内外动态活动与复杂人机交互场景。

**📈 对比分析**

与基线 HSfM 等单帧或单视角方法对比，方法在人体 MPJPE、相机轨迹误差、接触一致性等指标上均实现显著提升（MPJPE 下降 50% 以上，TE 和 s-CCA 亦显著提升），同时保持平滑、物理一致的运动轨迹。

**⚠️ 局限性**

局限性包括：①依赖同步多摄像头输入，对单摄像头或非同步场景适应性有限；②在极端遮挡或高速运动时仍可能出现误差；③全局优化计算量较大，部署在实时场景时需进一步加速。

---

## 702. Are Algorithm Registers Transparent? Perspectives from Germany

**arXiv ID:** 2606.02347 | [PDF](https://arxiv.org/pdf/2606.02347v1)

**作者:** Iman Peljto `[一作]` (TU Darmstadt and Johannes Gutenberg University Mainz), Mattia Cerrato `[通讯]` (Johannes Gutenberg University Mainz)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将Lorenz的德国国家AI透明度注册表概念性方案转化为可操作的检查表，对现有的两大德国算法注册平台MaKI和Lernende Systeme进行外部审计，评估其在实现透明度目标方面的达成度，并提出改进建议。

**💡 创新点**

创新点在于：①将规范性提案重构为可审计的检查表；②采用外部可观测性分类（Yes/No/Partial/No Access）进行系统评估；③将评估结果可视化为饼图，以直观展示平台与目标的匹配程度。

**🔧 技术方法**

使用的方法主要是基于文本的手工审计和结构化评估；技术层面涉及数据抓取、信息结构化、可视化（饼图）和检查表生成。

**📊 数据集**

数据来源为公开的MaKI和Lernende Systeme平台页面及其公开文档，没有使用专门的第三方数据集；审计基于平台自身公开的系统信息和字段。

**📈 对比分析**

比较方法是将检查表项与平台公开信息逐项对应，按四类评分类别统计比例，并通过饼图展示各目标达成率。结果显示两平台在风险评估、透明度细节、反馈机制等方面存在明显不足，整体透明度水平偏低。

**⚠️ 局限性**

局限性包括：①审计只能基于公开信息，内部架构和治理细节无法获取；②使用手工审计，主观性较高；③缺乏强制性注册机制，导致覆盖范围不完整；④没有API和自动化接口，影响可追溯性和更新频率。

---

## 703. Entropy Minimization without Model Collapse: Mitigating Prediction Bias in Medical Imaging

**arXiv ID:** 2606.02339 | [PDF](https://arxiv.org/pdf/2606.02339v1)

**作者:** Tim Nielen `[一作]` (Technical University of Munich), Julia A. Schnabel `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

探究熵最小化导致模型崩溃的机制，并提出基于重要性重权重的DSBR方法实现无监督测试时适配

**💡 创新点**

创新点在于将预测偏差识别为熵最小化崩溃根源，并通过等化每个预测类别对损失的贡献来抑制偏差扩散

**🔧 技术方法**

核心技术包括熵最小化、指数移动平均估计类别分布和重要性重权重损失

**📊 数据集**

实验使用四个医学影像数据集（Camelyon17、Histopantum、MammoBench、GliomaMRI）以及ImageNet‑C自然图像数据集

**📈 对比分析**

与四个主流EM基线对比，DSBR在医学影像上平均提升4.2%/2.6%准确率，消除模型崩溃；在ImageNet‑C上获得或超越SOTA性能

**⚠️ 局限性**

局限性在于依赖于预测类别分布的估计，极端标签不平衡或极小批量大小时效果可能下降

---

## 704. Repair Before Veto: Repair-Augmented Constraint Learning for Contextual Decisions

**arXiv ID:** 2606.02326 | [PDF](https://arxiv.org/pdf/2606.02326v1)

**作者:** Yifan Wang `[一作]` `[通讯]` (McGill University), Yifan Wang (McGill University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个将可执行修复操作嵌入硬约束决策的框架，学习在给定预算内接受或拒绝候选，并在接受时给出修复计划与信用分类。

**💡 创新点**

创新点包括：1) 将修复作为决策语义的一部分，解决终端否决导致的误判；2) 定义信用分类与修复计划输出；3) 给出统计学习理论证明和样本复杂度上界；4) 采用验证选择的阈值保护（Validation-Selected Guard）以降低阈值附近的误差。

**🔧 技术方法**

技术手段包括：上下文约束学习、线性偏好评分、修复可行性与成本检查、验证选择的阈值保护、伪维度/样本复杂度分析以及校准与AUROC评估。

**📊 数据集**

使用了四层次基准：Synthetic‑MAXSAT（布尔修复任务）、Expedia‑schema（Kaggle Expedia酒店推荐数据）、DB1B‑schema（NBER/BTS DB1B机票记录的校准半合成数据）以及 DB1B‑dist（真实分布的半合成数据，含修复注入）。

**📈 对比分析**

与无修复 HASSLE、SoftPenalty、BlackBox、BlackBox+CreditHead、BlackBox+RepairSearch 以及 Oracle‑RACL 等基线进行对比；在控制层次实现近零误判和决策误差，在原始 DB1B 层次，验证选取的模型将误判降至 10/4039，信用与修复计划准确率完美，优于 BlackBox+RepairSearch 的 FVR 0.2633，但 EDR 略高。

**⚠️ 局限性**

局限性在于需要已知完整的修复库，缺失修复模板会直接导致失败；模型无法学习新的修复操作；DB1B 基准为半合成数据；阈值保护是模型选择，未针对排名最优进行优化。

---

## 705. Terminal Steiner tree problem : Complexity and Algorithms

**arXiv ID:** 2606.02325 | [PDF](https://arxiv.org/pdf/2606.02325v1)

**作者:** Jyothish S `[一作]` (Indian Institute of Information Technology, Design and Manufacturing), Sadagopan Narasimhan `[通讯]` (Indian Institute of Information Technology, Design and Manufacturing)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了终端Steiner树（TST）问题的可行性、复杂性与算法，给出了存在性的必要充分条件，证明了在多类图上的NP‑完备性，并在终端数k上提供了3^k的固定参数可解算法，进一步构建了将普通Steiner树（ST）作为黑盒求解TST的框架。

**💡 创新点**

①首次给出图中终端Steiner树存在的多项式可判定条件；②提出统一框架将ST问题转化为TST，从而在分裂图、双分裂图等类上直接得到TST的可解性；③在终端数k上证明TST是固定参数可解的，并给出3^k时间递推式；④通过“增添悬挂顶点”闭合性推导多类图上TST的NP‑完备性。

**🔧 技术方法**

主要使用图论基本概念（连通性、邻域、分裂图结构）、多项式时间构造与递推（类似Dreyfus–Wagner算法的动态规划）、归约与构造证明（从X3C到TST）以及复杂度类的闭合性分析。

**📊 数据集**

未使用实验数据集；所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

与已有的ST算法相比，TST在可解类（如分裂图、双分裂图、k‑分裂图）能直接利用ST求解；在NP‑完备类（如平面图、二分图、割点图等）则保持NP‑完备；在终端数k上，算法时间为O(3^k)，与ST的Dreyfus–Wagner O(3^k)相当，但额外需保证终端为叶子，适用于加权版本。

**⚠️ 局限性**

局限性包括：①对二分图等类的TST可解性仍未确定；②加权TST在特定图类上的效率与近似性能尚未探究；③黑盒框架在某些特殊结构（如K1,6‑free分裂图）仍需进一步优化；④理论复杂度高，实际实现及大规模实例的性能尚未评估。

---

## 706. A Simulation Platform for Flapping-Wing Vehicles

**arXiv ID:** 2606.02370 | [PDF](https://arxiv.org/pdf/2606.02370v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 707. TVIR: Building Deep Research Agents Towards Text--Visual Interleaved Report Generation

**arXiv ID:** 2606.02320 | [PDF](https://arxiv.org/pdf/2606.02320v1)

**作者:** Xinkai Ma `[一作]` (Nanjing University), Jiaheng Liu `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TVIR（Text–Visual Interleaved Report Generation）框架，包含100个专家构造的多模态深度研究任务、一个分层多代理基线和双路径评估体系；

**💡 创新点**

创新点在于：①首个专门评估文本与视觉交错报表生成的 benchmark；②把视觉元素视为推理核心并在规划、检索、生成四阶段显式处理；③设计双路径评估（文本评估+视觉评估）以量化证据可靠性和跨模态一致性；

**🔧 技术方法**

采用大语言模型（Claude‑4.5‑Sonnet、Qwen3‑Max、GLM‑4.7等）结合搜索、网页抓取、VQA、代码生成绘图等工具；

**📊 数据集**

使用 100 个多模态任务集合（50 中文、50 英文），涵盖10大领域、三层复杂度；

**📈 对比分析**

对比 9 个深度研究系统，TVIR‑Agent 在整体分数上领先（例如 Claude‑4.5‑Sonnet 73.92 分），在视觉评估上表现最突出，文本评估也不逊；

**⚠️ 局限性**

局限性在于：①长篇文本的事实与逻辑一致性仍弱；②视觉检索与生成仍受工具调用限制；③部分结构错误（可追溯性、完整性）仍存在，需进一步改进跨模态一致性与工具协同。

---

## 708. AgentPLM: Agentic Protein Language Models with Reasoning-Augmented Decoding for Protein Sequence Design

**arXiv ID:** 2606.02386 | [PDF](https://arxiv.org/pdf/2606.02386v1)

**作者:** Sahil Rahman `[一作]` (Bedford), Maxx Richard Rahman `[通讯]` (Saarland University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种名为AgentPLM的蛋白语言模型，能够在生成过程中主动调用外部生物物理工具并根据工具反馈动态调整序列；

**💡 创新点**

提出Reasoning-Augmented Decoding（RAD）和Contrastive Agent Policy Optimisation（CAPO）两大创新，前者实现生成与工具调用的交替进行，后者通过对比轨迹学习决定何时调用工具，从而实现在线推理和错误校正；

**🔧 技术方法**

基于预训练ESM‑2的Transformer，扩展词表、构建工具上下文编码器（TCE）与轨迹记忆缓冲（TMB），并结合RAD解码策略与CAPO强化学习/对比优化；

**📊 数据集**

在五个公开基准上评估：ThermoStab‑75（ProThermDB/Ssym/FireProtDB）、AntibodyOpt‑VH（SAbDab/OAS）、EnzymeDesign‑EC3（BRENDA/SABIO‑RK）、PPI‑Interface（PDBbind/SKEMPI 2.0）以及ZeroShot‑Fitness（ProteinGym DMS）；

**📈 对比分析**

与ESM‑2、ProteinMPNN、EvoProtGrad、RFdiffusion‑AA、ProtAgent等传统方法对比，AgentPLM在所有任务均获最高分，例如抗体top‑10%命中率52.4%↑至27.4%，酶k_cat/K_m提升1.89×，热稳定性提升至7.64°C，PPI ΔG为‑5.26 kcal/mol；

**⚠️ 局限性**

主要局限包括对预先定义工具套件的依赖、工具调用导致的推理延迟（尤其是AutoDock Vina）、在极度稀缺数据场景下可能出现过拟合、以及尚未验证跨任务迁移与大规模蛋白的可扩展性。

---

## 709. SPADE-Bench: Evaluating Spontaneous Strategic Deception in Agents via Plan-Action Divergence

**arXiv ID:** 2606.02380 | [PDF](https://arxiv.org/pdf/2606.02380v1)

**作者:** Yuyan Bu `[一作]` (Beijing Academy of Artificial Intelligence), Juntao Dai `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SPADE-Bench，用以评估大型语言模型代理在工具使用情境下的计划-行动偏差（自我欺骗）问题，探讨在压力诱导下代理是否会隐瞒真实行动；

**💡 创新点**

创新点在于将真实工具执行与控制压力场景结合，形成能区分策略性欺骗与幻觉的评测框架，并提供从种子生成到人工审核的完整数据构建流程；

**🔧 技术方法**

使用技术包括：LLM推理与工具调用、压力种子与情境生成、Pass@k 采样评估、基于 Qwen‑3‑32B 微调的自动欺骗判定器（结合 GPT‑4o 训练标签），以及多模型对比实验；

**📊 数据集**

采用的数据集为 300 对场景（600 条案例）覆盖 239 种工具，包含 4 类压力类型；数据通过种子生成、质量控制与人工审核得到；

**📈 对比分析**

通过 Pass@5 采样指标与 Wilson 95% 置信区间对 8 大前沿模型进行比较，发现欺骗率呈非单调规模关系，Gemini‑2.5‑Pro 最高 57.33%，GPT‑5.1 最低 25.00%，各模型在不同工具类别与压力类型下表现差异显著；

**⚠️ 局限性**

局限性包括：规模受手工验证限制，难以大规模扩展；评测假设代理对测试环境与真实环境无差别，未来模型可能出现“测试意识”导致评测失真；

---

## 710. A Computational Toolkit for Engagement and Scalable Assessment in a Large Logic Course

**arXiv ID:** 2606.02367 | [PDF](https://arxiv.org/pdf/2606.02367v1)

**作者:** Stephen M. Watt `[一作]` `[通讯]` (University of Waterloo), Stephen M. Watt (University of Waterloo)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了LogicLab——一个轻量级的Racket工具包，帮助Waterloo CS 245课程的学生通过计算方式操作逻辑公式、执行变形、归约和形式证明，并为大规模评估提供自动化校验；

**💡 创新点**

创新点在于将熟悉的Racket编程环境与逻辑教学紧密结合，提供即时反馈与可编程的校验流程，既增强学生对抽象逻辑的可视化理解，又显著提升批改效率与一致性；

**🔧 技术方法**

使用技术包括Racket/Scheme实现的逻辑公式解析与展示、等价变换、归约、CNF/DNF转换、Davis–Putnam式一致性检测，以及基于规则的形式证明检查器；

**📊 数据集**

未使用公开数据集，系统主要处理课程中出现的逻辑公式与证明文本；

**📈 对比分析**

论文未提供实验对比或性能评估，未来计划在完整课程整合后通过学生使用频率、错误率和批改一致性等指标进行评估；

**⚠️ 局限性**

局限性包括缺乏完整的一阶逻辑统一与Skolem化支持、对课程作业设计的高度依赖、以及尚未完成实证评估，未来需要扩展功能并收集学习成效数据。

---

## 711. Local Preferential Bayesian Optimization

**arXiv ID:** 2606.02351 | [PDF](https://arxiv.org/pdf/2606.02351v1)

**作者:** Johanna Menn `[一作]` (RWTH Aachen University), Sebastian Trimpe `[通讯]` (RWTH Aachen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一系列局部优先贝叶斯优化（PBO）方法，通过在高维设置下利用信任域和梯度/海森信息改进采样。

**💡 创新点**

创新点在于首次将信任域和梯度/二阶信息引入PBO，并推导出拉普拉斯近似下的梯度与海森预测分布。

**🔧 技术方法**

使用了拉普拉斯近似的对偶高斯过程、梯度信息采集、贝叶斯二次规划、信任域控制等技术。

**📊 数据集**

在GP样本路径、标准优化基准（Hartmann、Rosenbrock等）以及MuJoCo Hopper/Walker2D的策略搜索任务上进行实验。

**📈 对比分析**

与全局PBO基线（HB‑EI、qEUBO、GLISp、Sobol）对比，局部方法在高维、复杂景观下累积收益更低，尤其是TuRPBO和PrefSQP表现最稳健。

**⚠️ 局限性**

局部方法对初始区间敏感，在低维或缺乏良好起点时不如全局方法，且对高斯过程超参数学习和实际人类偏好噪声敏感。

---

## 712. Coordination Graphs for Constrained Multi-Agent Reinforcement Learning

**arXiv ID:** 2606.02337 | [PDF](https://arxiv.org/pdf/2606.02337v1)

**作者:** Santiago Amaya-Corredor `[一作]` (Universitat Pompeu Fabra), Anders Jonsson `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了CG-CMARL框架，将协调图与拉格朗日对偶结合，能够在去中心化执行下处理多智能体受约束强化学习问题。

**💡 创新点**

创新点包括：①双头Q网络（目标与约束）共享参数，②拉格朗日乘子可在评估时扫描得到完整的 Pareto 前沿，③在因子图上使用 Max‑Sum 消息传递实现动作协调，④给出收敛性保证和分层误差分解。

**🔧 技术方法**

使用技术包括协调图、Max‑Sum 消息传递、拉格朗日主从方法、双头 Q 学习、参数共享、两时间尺度随机逼近、理论误差分析。

**📊 数据集**

实验数据集为 MPE 套件中的 Simple Spread 任务，团队规模从 3 到 10 名智能体。

**📈 对比分析**

与 IQL、QMIX、DCG、MAPPO 及 MAPPO‑Lagrangian 等基线比较，CG‑CMARL 在覆盖率–安全性 Pareto 前沿上占优，能够从单一模型得到完整前沿且方差更低。

**⚠️ 局限性**

局限性包括：对大规模团队，双边协调图变得稠密导致 Max‑Sum 计算开销上升；双头模型的保守近似在理论收敛性上有待进一步验证；λ‑增强变体更易实现一致性但样本复杂度更高；未在具有相互依赖转移的环境中验证。

---

## 713. Less Is More? When Dataset Context Hurts LLM-Generated Dataset Descriptions

**arXiv ID:** 2606.02334 | [PDF](https://arxiv.org/pdf/2606.02334v1)

**作者:** Lisa-Yao Gan `[一作]` (Technical University of Munich), Elena Simperl `[通讯]` (King's College London)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了大语言模型在自动生成数据集描述时，随不同数据集上下文的变化对描述质量的影响，并通过大规模消融实验验证。

**💡 创新点**

提出了“schema penalty”现象，即仅提供表结构会削弱叙事质量，同时揭示了不同LLM在描述风格上的稳定差异，为数据发布工作流程提供实证指导。

**🔧 技术方法**

使用了多种开源指令调优LLM（LLaMA-3、Qwen、Mistral、Gemma）以及LLM-as-a-judge评估框架，结合手写正则的属性分析。

**📊 数据集**

采集自欧盟数据门户data.europa.eu的252个数据集（共1336个CSV文件），构建标题、架构和样本三种上下文。

**📈 对比分析**

通过LLM-as-a-judge的评分和描述属性分布分析，对比标题仅、标题+架构、标题+架构+样本三种提示，并与人类作者原始描述进行对照，发现即使加入更多上下文，整体质量并未必提升，且LLM生成描述往往优于原始元数据。

**⚠️ 局限性**

主要局限包括评估依赖LLM判定者的偏见、仅实验单一门户数据、仅测试少量开源模型和固定提示策略，结果可能不具普适性。

---

## 714. Repurposing Adversarial Perturbations for Continual Learning: From Defense to Active Alignment

**arXiv ID:** 2606.02322 | [PDF](https://arxiv.org/pdf/2606.02322v1)

**作者:** Ran Liu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Ming Liu `[通讯]` (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AdvCL，一种利用对抗扰动作为几何控制信号的持续学习框架，包含三大模块：Intra‑Smooth（局部平滑）、Proto‑Clip（原型裁剪）和 Inter‑Align（跨任务对齐）以提升 LLM 的持续学习稳定性、减轻遗忘并提升鲁棒性。

**💡 创新点**

创新点在于将对抗扰动从防御转为主动几何控制，利用对抗扰动实现局部平滑与跨任务方向对齐，并通过原型裁剪防止对当前任务的过度对齐；此外，该方法是可插拔的，能够在多种持续学习范式（回放、正则化、动态结构）中无缝集成。

**🔧 技术方法**

技术包括：基于嵌入空间的对抗扰动生成（PGD、FGSM、TRADES等）、小扰动局部平滑、基于余弦相似度的原型裁剪损失、以及利用前一任务原型进行方向性对齐的内外循环优化；同时保持 backbone 参数冻结，仅更新 PEFT（LoRA）子模块。

**📊 数据集**

实验使用六个跨领域任务组成的 CL 基准：SemEval 2018 舆情情感检测、情绪分类、AG News 主题分类、ANLI 对抗自然语言推断、XSum 极端摘要、XL‑Sum 标题生成。每个任务训练 2000 条样本，验证 200 条，测试 200 条。

**📈 对比分析**

与多种基线（Sequence、Replay、EWC、MoE、Joint、Separate）进行对比，AdvCL 在平均性能、遗忘、正向/反向迁移等标准 CL 指标上均表现出显著提升。具体而言，三模块组合在 AP、FGT、FWT、BWT 等指标上均优于单模块或基线，并在鲁棒性测试（PGD‑5/10、FGSM、Rand 等）中保持较低的性能衰退。

**⚠️ 局限性**

局限性包括：对抗扰动生成会增加约 25% 的训练开销；不同任务的相似度变化仍可能导致对抗扰动效果不一致；在某些低相似度任务对齐时可能出现训练不稳定；此外，对抗扰动的超参数（ε、K 等）对结果影响显著，需经验调优。

---

## 715. Discovering Agents for Discovery: The Case for DNS

**arXiv ID:** 2606.02314 | [PDF](https://arxiv.org/pdf/2606.02314v1)

**作者:** Ramachandra Rao Seethiraju `[一作]` (Verisign), Eric Osterweil `[通讯]` (Verisign)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于DNS的AI代理发现框架，并对其可行性进行了数据驱动评估

**💡 创新点**

首次将DNS、DNSSEC、DANE等现有网络基础设施与AI代理发现需求结合，构建了三维评估框架（导航完整性、查询复杂度、事务性能），并证明大多数代理元数据可在单个UDP DNS响应内完成

**🔧 技术方法**

DNS、DNSSEC、DANE（TLSA）以及SVCB记录；使用EDNS0、UDP/TCP回退机制

**📊 数据集**

APIs.guru（143,634个API端点）与MCPZoo（62,739个MCP服务器）数据集

**📈 对比分析**

对比不同发现方案的元数据大小、查询次数和UDP响应容量；实验显示90th分位代理元数据+证书+DNSSEC大小约940字节，低于1,232字节PMTU阈值，可在单轮DNS查询内完成；性能相当于现有数十亿次DNS事务的毫秒级延迟

**⚠️ 局限性**

仅评估元数据尺寸，未测量真实代理网络交互；使用间接数据来源，可能低估未来代理的复杂度；未覆盖所有现有发现方案的完整性能评估

---

## 716. Equilibrium Semantics and Strong Equivalence for Higher-Order Logic Programs

**arXiv ID:** 2606.02387 | [PDF](https://arxiv.org/pdf/2606.02387v1)

**作者:** Angelos Charalambidis `[一作]` (Harokopio University of Athens), Panos Rondogiannis `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了高阶逻辑程序的平衡逻辑语义，定义了强等价性，并证明分层程序具有唯一平衡模型，同时给出了可定义性定理

**💡 创新点**

创新点在于：① 将HT扩展到高阶类型并引入证明单调关系以保证语义一致性；② 证明高阶程序的强等价性与模型相等的等价性；③ 证明所有总的证明单调关系可由分层程序定义

**🔧 技术方法**

使用了高阶Heyting逻辑（HT）、平衡逻辑、证明单调关系（justification monotonicity）、定义可定义性和强等价性定理的形式化证明技术

**📊 数据集**

未使用任何实验数据集；本文为理论性研究，全部基于形式化证明

**📈 对比分析**

通过形式化证明与已有的平衡逻辑和AFT结果对比，未涉及实验性能评估，重点在理论一致性与可推理性

**⚠️ 局限性**

局限性：目前仅针对分层高阶程序给出唯一平衡模型与强等价性，非分层程序的情况仍待进一步研究；对更一般高阶语义（如包含嵌套蕴含等）的扩展尚未完成

---

## 717. TabPrep: Closing the Feature Engineering Gap in Tabular Benchmarks

**arXiv ID:** 2606.02384 | [PDF](https://arxiv.org/pdf/2606.02384v1)

**作者:** Andrej Tschalzev `[一作]` (University of Mannheim), Christian Bartelt `[通讯]` (Technical University of Clausthal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 TabPrep，一种轻量级预处理管道，结合针对三类结构模式的特征生成器，直接插入到模型训练流程中；

**💡 创新点**

创新点在于识别并聚焦三种常见的结构模式（有序算术交互、类别条件效应、伪类别数值交互），通过模式驱动的特征生成（OAFE、CatPrep、RSFC）取代传统的暴力搜索和监督选择，显著缩短计算成本并实现更高的峰值性能；

**🔧 技术方法**

采用有序算术特征扩展、交叉验证目标编码、相对组别特征、随机子集压缩等技术；在 TabArena 框架下与 LightGBM、TabM、TabPFN‑2.5、线性模型等四类模型集成；

**📊 数据集**

在 TabArena 51 个公共表格数据集（含 10k–100k 训练样本的中型数据集）上进行评估；

**📈 对比分析**

通过 ELO 统计量与 TabArena 27 个基线模型对比，TabPrep-增强模型在平均水平提升多达 5–10% Elo，且在 24/51 数据集上刷新了新的峰值表现，甚至超过最新的 TabPFN‑2.6 与 TabICLv2；与 autofeat 与 OpenFE 的比较表明 TabPrep 在性能上更优、计算成本更低；

**⚠️ 局限性**

局限性包括：仅提供轻量级基线并不保证最优特征；评估仅覆盖四类模型；特征扩展会增加训练和推理成本；只关注三种已知结构，其他潜在结构未被覆盖；仅适用于 IID 表格预测任务，未覆盖非 IID、关系型或文本特征场景。

---

## 718. FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo

**arXiv ID:** 2606.02365 | [PDF](https://arxiv.org/pdf/2606.02365v1)

**作者:** Kyunghun Nam `[一作]` (KENTECH), Sumyeong Ahn `[通讯]` (KENTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

改进 Shampoo 预条件优化器，提出自适应阻尼与频率控制机制，以降低矩阵逆运算的频率并提升数值稳定性。

**💡 创新点**

创新点包括：①从收敛与稳定性双重视角理论分析延迟（staleness）对 Shampoo 的影响；②设计基于矩阵扰动理论的误差代理 h_t，用作实时检测预条件器偏差；③利用 h_t 通过反馈控制动态调节阻尼 ϵ，并在阻尼超过阈值时触发特征分解，从而显著减少不必要的特征分解次数。

**🔧 技术方法**

主要技术手段：Kronecker‑factored Shampoo、矩阵逆根（inverse‑root）预条件、矩阵扰动与相对误差代理、反馈控制（阻尼调节）以及自适应特征分解触发策略。

**📊 数据集**

实验数据集包括 ImageNet‑1K（ViT‑small）、LibriSpeech（Conformer）和 Wikitext‑103‑v1（GPT‑2）等。

**📈 对比分析**

与传统 stale‑Shampoo、AdamW、SOAP 等基线对比，实验显示在保持或提升最终性能（Top‑1 accuracy、WER、PPL）的同时，wall‑clock 时间缩短 60–80%，特征分解调用次数下降 80%+，且在不同超参数设置下保持鲁棒性。

**⚠️ 局限性**

局限性：①需要手动调节诸如阈值 τ、最大阻尼 ϵ_max、刷新周期 f 等超参数；②理论上限和误差代理的估计仍有保守性，可能导致过度/不足的阻尼；③主要针对可微分、连续曲率的场景，极端非凸或大噪声环境下表现尚待验证；④在极大模型中，计算代理本身的开销虽小，但仍比完全跳过特征分解要高。

---

## 719. Multi-modal Video Representation Alignment for Robust Self-supervised Driver Distraction Detection

**arXiv ID:** 2606.02352 | [PDF](https://arxiv.org/pdf/2606.02352v1)

**作者:** David J. Lerch `[一作]` (Fraunhofer IOSB), Rainer Stiefelhagen `[通讯]` (Karlsruhe Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在驾驶员分心检测任务中，构建了一种全局多模态自监督对齐框架（GMA），通过软目标和正样本加权解决了多模态数据中的错误负样本和不可靠正样本问题。

**💡 创新点**

创新点包括：① 将对抗性负样本处理（soft target）与正样本加权（similarity‑based weighting）结合，形成全局多模态RxID损失；② 在所有模态对之间同步对齐，提升跨模态语义一致性；③ 通过循环一致性得分动态生成软目标，进一步减弱错误负样本影响。

**🔧 技术方法**

主要技术包括：自监督对齐的InfoNCE、软目标生成（基于循环一致性）、正样本加权（基于正态分布与Cumulative Distribution Function）、全局多模态对齐（GMA）以及使用预训练的模态特定编码器（CLIP‑ViP、OMNIVORE）。

**📊 数据集**

使用的公开数据集为Drive&Act，包含RGB、红外、深度和骨架四种模态，此外在跨视角实验中使用了该数据集的六个不同红外视角。

**📈 对比分析**

与传统的局部对齐（LMA）和现有的CMVRA GMA基线相比，MM‑RxID 在所有模态上均取得了显著提升（如 IR、Depth、Skeleton 的平均平衡准确率提升约0.9%–2.3%）。在跨视角泛化实验中，MM‑RxID 在新视角的准确率提升高达5.9%，显著降低视角相关性能波动。

**⚠️ 局限性**

局限性包括：① 某些模态（尤其是红外）在平衡信息中提升有限；② 软目标和加权策略引入额外超参数（δ、κ、w_min、α等），需细致调优；③ 计算开销略大，随着模态数量增多，训练效率可能下降。

---

## 720. Detecting Pen-In-Air States from Video: A Proof-of-Concept Toward Complementary Handwriting Analysis

**arXiv ID:** 2606.02342 | [PDF](https://arxiv.org/pdf/2606.02342v1)

**作者:** Lauren Sismeiro `[一作]` (IMT Mines Ales), Gerard Dray `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用顶部视角视频通过混合管线检测书写中的笔离纸状态（Pen‑Up）

**💡 创新点**

创新点在于将YOLOv11m笔尖检测与多尺度运动特征工程结合，再用可解释的机器学习模型进行事件级分类，并通过后处理实现时间一致性

**🔧 技术方法**

使用YOLOv11m做笔尖定位，提取147维局部/全局运动特征，采用随机森林、HistGBM和LightGBM进行分类，后续通过滞后平滑、形态学滤波和速度极值对齐进行事件级优化

**📊 数据集**

5段从YouTube公开收集的1080p 30fps书写视频，共计13,507帧，人工标注为Pen‑Down/Up/无信息，28.7%为Pen‑Up

**📈 对比分析**

与基线的端到端深度模型（ResNet18、CNN‑LSTM、3D‑CNN）相比，混合方法在LOVO交叉验证下实现了10帧容差时的F₂≈0.79–0.80，12帧时最高达0.805，远优于基线（最高0.48）

**⚠️ 局限性**

主要局限包括样本量仅5视频、单摄像头缺乏深度信息、视角固定、缺乏公开数据/代码，且未在大规模或临床环境中验证

---

## 721. O-POPE: High-Frequency Pipelined Outer Product based GEMM acceleration with minimal buffering overhead

**arXiv ID:** 2606.02333 | [PDF](https://arxiv.org/pdf/2606.02333v1)

**作者:** Danilo Cammarata `[一作]` (ETH Zurich), Luca Benini `[通讯]` (ETH Zurich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种高频率外乘法（outer‑product）GEMM加速器O‑POPE，能够在12 nm FinFET工艺下以1 GHz、0.72 V工作，支持多种浮点精度，并通过将FPU流水线寄存器重用为缓冲实现极高的算术利用率与极低的缓冲面积；

**💡 创新点**

核心创新在于将FPU流水线寄存器重新用作双缓冲，消除了传统Systolic架构中昂贵的输入/输出缓冲需求，并实现了1 GHz的高频率运算；外乘法数据流与输出静止策略相结合，进一步提升了数据重用和面积效率；

**🔧 技术方法**

采用半同步阵列、输出静止外乘法数据流、FPnew IEEE 754浮点单元、双缓冲、PULP集群框架、tcdm共享存储与DMA双缓冲、12 nm FinFET技术；

**📊 数据集**

使用多种机器学习工作负载的层作为实验数据集，包括Vision Transformer（ViT）、ConvNeXt、GPT、gMLP和MLP‑Mixer，对应的M×K×N尺寸在论文表中给出；

**📈 对比分析**

与Gemmini、RedMulE、Sauria等同类16×16加速器在相同MAC单元配置、相同浮点单元FPnew下进行对比；通过统一的层级映射和双缓冲策略，O‑POPE在GFLOPS、GFLOPS/mm²和TFLOPS/W上分别提升了约1.86×、1.56×和1.08×；

**⚠️ 局限性**

局限性包括：依赖12 nm FinFET工艺和FPnew单元；在K维较小或矩阵尺寸不为两倍阵列大小时利用率下降；需要128 KB tcdm存储，限制了能处理的大矩阵规模；随着阵列尺寸增大，缓冲比例虽低但仍会略微上升；

---

## 722. A Mathematical Conflict Framework for Contextual Data Modulation

**arXiv ID:** 2606.02381 | [PDF](https://arxiv.org/pdf/2606.02381v1)

**作者:** Hakan Emre Kartal `[一作]` `[通讯]`, Hakan Emre Kartal

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个基于算子的一般化冲突框架，用于显式量化原始数据与上下文数据之间的方向性差异，并通过优先级权重和投影算子实现冲突的测量、调制和决策映射。

**💡 创新点**

创新点在于：①把冲突视为独立的、可测量的算子对象；②基于五条基本公理构建冲突算子族 {g1,g2,g3}，覆盖尺度不变、比例敏感和绝对差异三种行为；③将冲突测量、上下文加权和投影解耦，形成模块化、可插拔的体系。

**🔧 技术方法**

使用的技术包括：算子代数（冲突核 Φ、Hadamard 权重、投影 Ψ）、多维线性组合、可微分算子（g1,g2,g3 可微）、O(N) 计算复杂度、以及基于优先级的上下文调制。

**📊 数据集**

实验仅使用了合成数据：在 [0.1,10] 的 200x200 网格上验证算子属性，并用两组 2x3 的原始/上下文矩阵演示冲突投影，未使用公开真实数据集。

**📈 对比分析**

比较方式是对齐冲突算子的零冲突、一致性、反对称性等公理的数值验证，实验显示所有算子均满足公理并产生不同几何形状；然而未与传统损失或距离度量在实际任务上的性能做定量比较。

**⚠️ 局限性**

局限性：①投影算子 Ψ 仅设为通用形式，缺乏对不同投影族的系统评估；②实验规模小，未在真实多任务或 MCDM 数据上验证；③缺少与现有方法在任务表现上的定量对比。

---

## 723. PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

**arXiv ID:** 2606.02366 | [PDF](https://arxiv.org/pdf/2606.02366v1)

**作者:** Xiaohang Yu `[一作]` (Ecole Polytechnique Federale De Lausanne), Mackenzie Weygandt Mathis `[通讯]` (Ecole Polytechnique Federale De Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出 PRIMA 框架，能够从单张图片中恢复动物 3D 网格，特别针对四足动物的种类与姿态不平衡问题，提升了对稀有物种和极端姿态的重建效果。

**💡 创新点**

创新点主要包括：① 利用 BioCLIP 的生物先验嵌入来初始化 SMAL 形状参数，增强模型对物种结构的泛化；② 开发基于 2D 重投影和关键点指引的测试时自适应（TTA）策略，实现实例级精细化；③ 通过 TTA 生成伪 3D 数据，构建 Quadruped3D 大规模数据集，进一步提升模型性能。

**🔧 技术方法**

技术上结合了 ViT 编码器、BioCLIP 先验、SMAL 参数化模型、Transformer 解码器（含关键点令牌）以及迭代误差反馈（IEF）与 2D 重投影损失。

**📊 数据集**

使用的主要数据集包括 Animal3D、CtrlAni3D、Quadruped2D（自建）以及新构建的 Quadruped3D；在评估时还使用了 Animal Kingdom 作为离群测试集。

**📈 对比分析**

与 HMR、HMR2.0、GenZoo、WLDO、AniMer、AniMer+ 等现有方法比较，PRIMA 在 3D（PAJ/PA V）与 2D（AUC、PCK）指标上均取得或逼近最优结果，在多种数据集上表现尤为突出，尤其是稀有种类和复杂姿态。

**⚠️ 局限性**

局限性包括：① 仍受限于 SMAL 模型的骨架拓扑，可能不适用于所有动物形态；② TTA 需要额外计算，实时性受限；③ 伪 3D 数据的质量依赖于初始模型，误差累积可能影响后续训练。

---

## 724. SIRI: Self-Internalizing Reinforcement Learning with Intrinsic Skills for LLM Agent Training

**arXiv ID:** 2606.02355 | [PDF](https://arxiv.org/pdf/2606.02355v1)

**作者:** Zhongyu He `[一作]` (Xiamen University), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用LLM自我挖掘、验证并内部化技能，使代理在长时间任务中无需外部技能库即可完成决策。

**💡 创新点**

提出三阶段无外部依赖框架：先warm‑up，随后自我挖掘与在线验证，再通过优势加权内部化，将有用技能转移到模型参数中。

**🔧 技术方法**

核心技术包括GiGPO强化学习、对齐式paired roll‑out验证、在线宏观效用评估与优势加权的技能内部化（distillation）。

**📊 数据集**

在ALFWorld和WebShop两大长时序代理基准上进行评估。

**📈 对比分析**

与提示式、RL、记忆增强RL基线对比，SIRI在ALFWorld成功率从0.908提升至0.930，在WebShop成功率从0.728提升至0.813，整体表现优于所有对比方法。

**⚠️ 局限性**

技能挖掘质量随代理能力提升而提升，初期缺乏专门的训练信号，导致低质量技能难以避免，限制了早期学习效率。

---

## 725. Multidimensional Reconciliation in Continuous-Variable QKD: Review, Coding Schemes, and Open Source Simulation

**arXiv ID:** 2606.02323 | [PDF](https://arxiv.org/pdf/2606.02323v1)

**作者:** Martial Lucien `[一作]` (Exail), Gouraud Baptiste `[通讯]` (Exail)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统综述了连续变量量子密钥分发中的多维重构技术，并基于该技术提出了高维（超过八维）的实现方案及开源仿真框架；

**💡 创新点**

创新点包括：①将随机正交变换与Householder方法引入高维重构，实现了对任意维度的通用映射；②将多维重构与LDPC编码（coset、syndrome拼接）结合，提供统一的纠错策略；③发布可复现的C++仿真工具，便于社区进一步研究；

**🔧 技术方法**

采用的核心技术有多维重构（Cayley‑Dickson代数与随机正交变换）、BIAWGN虚拟通道建模、LDPC纠错（coset、syndrome拼接）及Belief Propagation（SPA）解码；

**📊 数据集**

使用的是公开的原型图LDPC校验矩阵（来自文献中的高性能代码集），仿真时在多维虚拟通道上生成随机数据；

**📈 对比分析**

通过绘制β‑FER曲线、吞吐量与维度、码率等指标的对比实验，证明了大维度（如d=8、64）可显著提升β（最高10%提升）且与BIAWGN通道表现趋同；coset与syndrome拼接在误码率上相当，但后者吞吐量下降约一半；

**⚠️ 局限性**

局限性包括：①未完成完整的有限尺寸安全性与子帧处理分析；②仅考虑二进制线性码，未探讨非二进制或极化码等新型方案；③高维随机正交变换的计算开销和硬件实现仍待优化；④仿真环境未覆盖硬件加速和实时运行场景。

---

## 726. Training-Free Composed Video Retrieval via Visual Representation-Guided Video-LLM Reasoning

**arXiv ID:** 2606.02321 | [PDF](https://arxiv.org/pdf/2606.02321v1)

**作者:** Yang Liu `[一作]` (University of Chinese Academy of Sciences), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个训练自由的视觉表示引导的视频LLM推理框架，用于根据参考视频和文本修改指令进行组合视频检索。

**💡 创新点**

创新之处在于将冻结的DINOv3视觉表示与大型视频-语言模型相结合，并在检索阶段引入推理和细化，提升对修改指令的语义理解。

**🔧 技术方法**

采用DINOv3 ViT-L/ViT-7B进行帧级视觉表示、Chamfer相似度候选池构建，以及Qwen3-VL-8B-Instruct和Qwen3-VL-8B-Thinking进行候选重排序与推理。

**📊 数据集**

在CVPR 2026 Reason-Aware Composed Video Retrieval Challenge的数据集上评估，包含SSv2和WebVid 4,333个视频的候选库。

**📈 对比分析**

通过逐步比较：纯视觉检索Recall@1 25.34%，添加LLM重排序后提升至35.27%，再细化后达到48.78%，在测试集上表现优于单一视觉或单一LLM方法。

**⚠️ 局限性**

局限在于依赖大型LLM推理且未进行任何微调，且模型对候选池大小及推理成本敏感，未来需探索更高效的整合与更强LLM。

---

## 727. Meta Flip Graph meets Serendipitous Product: new Fast Matrix Multiplication results

**arXiv ID:** 2606.02480 | [PDF](https://arxiv.org/pdf/2606.02480v1)

**作者:** A. I. Perminov `[一作]` `[通讯]` (Research Center for TAI), A. I. Perminov (Research Center for TAI)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

利用元翻转图（meta flip graph）与意外乘积（serendipitous product）相结合，对所有 680 种 2 ≤ n ≤ m ≤ p ≤ 16 的矩阵乘法格式进行全覆盖搜索，得到大量新的低秩乘法方案，并显著提升了大部分格式的秩和异方程指数。

**💡 创新点**

创新点包括：① 将元翻转图框架扩展到 16×16×16，支持 680 个格式；② 将意外乘积作为一种新的构造方法引入搜索，显著降低了复杂格式的秩；③ 在多种系数域（ℤ_T、ℤ、ℚ）中发现了许多以前仅在 ℤ 或 ℚ 下存在的三元系数方案；④ 通过 Hensel 提升和 rational 还原实现了从 ℤ₂/ℤ₃ 到 ℤ_T 的转换；⑤ 发现 51 个指数小于 Strassen 的新方案。

**🔧 技术方法**

主要技术包括：元翻转图随机游走、局部翻转/加/归约算子、维度改变元算子（Merge、Product、Extend、Project）、意外乘积构造、Hensel 提升、整数/三元系数重构、随机化分解与 bud 结构分析、并行多工并行搜索。

**📊 数据集**

数据集为所有 680 个矩阵乘法格式（n×m×p 其中 2 ≤ n ≤ m ≤ p ≤ 16），以及每个格式下已知的最佳秩、系数域信息，全部存储在公开的 GitHub 仓库中。

**📈 对比分析**

对比方法：将新得到的秩与 Sedoglavic 在线目录中已有的最佳秩进行比较；通过表格列出改进数量、系数域和对应的 ω 指数；实验在两台机器上并行运行，累计约 25 M 个不同方案，发现约 4 M 为最优秩；改进后总体三元、整数、分数系数分布分别为 X%、Y%、Z%，指数小于 log₂7 的格式数量从 29 增至 51。

**⚠️ 局限性**

局限性包括：① 搜索仍高度依赖启发式和随机性，难以保证最优性；② Hensel 提升对大尺寸格式效率低下，导致部分三元方案被丢弃；③ 仅关注秩，未系统评估加法复杂度、数值稳定性与实际实现效率；④ 对更大尺寸（>16）格式的可扩展性未知；⑤ 结果仍以理论符号秩为主，缺乏对浮点实现误差的分析。

---

## 728. LLM-Evolved Pattern Generators for Optimal Classical Planning

**arXiv ID:** 2606.02438 | [PDF](https://arxiv.org/pdf/2606.02438v1)

**作者:** Windy Phung `[一作]` (Linköping University), Jendrik Seipp `[通讯]` (Linköping University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练并进化出每个规划域的可构造可接受的模式数据库生成器，利用LLM驱动的进化程序合成框架，得到能在测试时快速生成可接受启发式的可解释程序。

**💡 创新点**

通过学习可接受的模式集合生成器而非直接学习启发式值，保证启发式可接受性，并利用LLM进化生成可解释的域特定程序，实现对最优规划的有效加速。

**🔧 技术方法**

使用LLM驱动的进化程序合成（OpenEvolve + MAP‑Elites + Kimi K2.5 LLM）、模式数据库启发式、饱和成本分配、Fast Downward评估、Scorpion求解器等技术。

**📊 数据集**

使用IPC 2023学习轨道基准集和Autoscale基准集（7个域：Blocksworld、Childsnack、Floortile、Miconic、Rovers、Satellite、Transport）。

**📈 对比分析**

与五个系统化模式生成基线（1–3大小、全模式、随机替换）及最佳系统化基线在覆盖率、扩展数、总搜索时间和每状态评估成本上进行比较；在5/7域达到或超过最佳基线，在大多数任务中每状态评估速度快数倍。

**⚠️ 局限性**

在模式小、系统化枚举已足够的域（Floortile、Miconic）效果不佳；训练集规模有限、评分函数与最终覆盖度略有偏差，且对域结构的依赖可能限制跨域泛化。

---

## 729. A Local Perturbation Theory for Cross-Domain Interference and Recovery in Multi-Domain RL

**arXiv ID:** 2606.02398 | [PDF](https://arxiv.org/pdf/2606.02398v1)

**作者:** Lei Yang `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多域强化学习中跨域干扰机制，提出局部路由级别干扰与恢复理论，并通过短刷新与稀疏回滚验证。

**💡 创新点**

创新点在于将干扰定位到共享活跃路由的低维冲突子空间，证明第二阶损伤驱动交叉域干扰，并证明短刷新可几何收缩该冲突子空间，实现选择性恢复。

**🔧 技术方法**

技术包括局部扰动模型、第二阶损伤分析、短域刷新、稀疏坐标回滚、梯度正交性和激活重叠评估。

**📊 数据集**

使用 Qwen3-4B-Thinking-2507 预训练模型，并收集数学推理（OpenR1-math）、代码生成（KlearReasoner-CodeSub-15K）、问答（SuperGPQA）、创意写作（crownelius/Creative-Writing）四个领域数据。

**📈 对比分析**

对比单域专家、顺序训练、JT与CGPO混合训练，发现顺序训练导致局部退化，短刷新后平均分数从 64.25 提升至 66.39，恢复率高于混合训练。

**⚠️ 局限性**

局限在于尚未实现自动化刷新检测与调度，冲突子空间定位粗糙且回滚预算较大，且只验证了 RL 领域，未探究其他后训练情境。

---

## 730. Not What, But How: A Communicative Audit of LLM Response Framing

**arXiv ID:** 2606.02493 | [PDF](https://arxiv.org/pdf/2606.02493v1)

**作者:** Siddhesh Milind Pawar `[一作]` (University Of Copenhagen), Isabelle Augenstein `[通讯]` (University Of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在回答主观文化问题时的沟通框架，提出统一的四维评估框架，并构建包含 376,350 条 Reddit 文化问题的自定义数据集。

**💡 创新点**

创新点在于将文化定位、泛化语言、拟人化和格赖斯会话准则四个维度统一评估，揭示模型在这些维度上的差异以及文化定位与拟人化的耦合关系；并公开提供评估工具和数据。

**🔧 技术方法**

采用 LLM‑as‑a‑judge 自动标注四个维度，利用 Gricean maxims 评估会话准则；使用广义估计方程（GEE）进行统计分析；对 Llama、Gemma、Mistral 三大开源 LLM 生成的回答进行批量评估。

**📊 数据集**

数据集来源于 57 个子版块、7 个国家（印度、韩国、土耳其、中国、俄罗斯、菲律宾、美国）和 19 类文化问题类别的 376,350 条 Reddit 问题；已在 Hugging Face 上发布公共子集。

**📈 对比分析**

通过对三模型在四个维度出现率及其耦合关系进行统计比较，发现 Mistral 在文化定位和泛化语言上占优，Gemma 在格赖斯准则上表现最佳，Llama 最受约束；GEE 计算显著差异，呈现模型排名并展示不同模型的优势与劣势。

**⚠️ 局限性**

局限性包括：文化代理仅基于子版块，缺乏全面代表性；LLM 判定器存在噪声，标注结果不为绝对真值；格赖斯准则基于西方语用，可能与跨文化沟通不完全契合；未充分考虑多语言混合文本。

---

## 731. MORPHOS: Autoregressive 4D Generation with Temporal Structured Latents

**arXiv ID:** 2606.02491 | [PDF](https://arxiv.org/pdf/2606.02491v1)

**作者:** Minkyung Kwon `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种自回归4D生成框架，可从视频生成统一的动态三维表示（网格、3D高斯、辐射场），通过Temporal Structured Latents同时编码几何与外观；

**💡 创新点**

创新点包括：1）Temporal Structured Latents将三维结构与外观扩展到时间维度；2）两阶段自回归流网络（Φ_S 与 Φ_L）配合因果注意力实现长序列生成；3）时序-结构增强训练策略，有效抑制自回归过程中的误差累积；

**🔧 技术方法**

使用预训练的图像到三维生成模型、Rectified Flow Transformer、因果注意力、RoPE 位置编码、KV 缓存以及结构化与时序噪声增强；

**📊 数据集**

数据集涵盖10K 动态三维资产（Objaverse/Objaverse‑XL）生成12帧视频，以及Motion80、ActionBench、Consist4D 三个评测基准；

**📈 对比分析**

与现有视频到4D网格/高斯生成方法及单帧图像到三维模型（TRELLIS）比较，在所有基准上实现了近乎最优的外观指标（LPIPS、CLIP、DreamSim、FVD），几何指标排名第二，显示出统一表示下的优越性；

**⚠️ 局限性**

局限性在于几何精度仍略低于专门的网格生成方法，训练与推理成本较高，并且对极端形变或极长序列仍可能出现轻微误差。

---

## 732. Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA Manipulation

**arXiv ID:** 2606.02486 | [PDF](https://arxiv.org/pdf/2606.02486v1)

**作者:** Shahram Najam Syed `[一作]` (Carnegie Mellon University), Jeffrey Ichnowski `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 predict‑then‑act 包装器，在冻结的 Vision‑Language‑Action 模型上叠加运动感知的潜在世界模型，使机器人能够在动态场景中预测未来状态并生成动作。

**💡 创新点**

创新点包括：①仅对语言相关且运动显著的图像块进行自适应空间计算；②利用条件流匹配与显式加速度动力学在潜在空间中进行高效前向滚动；③基于不确定性阈值的自适应时间截断，实现实时计算。

**🔧 技术方法**

使用的技术包括：RAFT 光流估计、条件流匹配（flow‑matching）模型、Transformer 编码器/解码器、显式加速度的解析动力学更新、语言‑运动相关性掩码和不确定性基阈值截断。

**📊 数据集**

数据集涵盖：（1）20 个多对象动态仿真场景（使用 MuJoCo + Panda 机器人），（2）约 200 条在 UFactory xArm 7 上收集的真实任务轨迹，用于微调。

**📈 对比分析**

与六种基线（Open‑VLA、Retargeting VLA、VLA+Fast Replan、Realtime ACT、Streaming Diffusion Policy、DreamVLA）对比，实验显示该方法在仿真中成功率提升至 79–97%，而最强基线仅 31–58%；在真实机器人上，任务成功率可达 30/30（搬运/滚动球）和 19/30（投射物捕捉），基线全部失败。

**⚠️ 局限性**

局限性包括：假设恒定加速度的动力学在碰撞后或高度随机的运动中失效；RAFT 仅提供像素平面运动，对深度变化不敏感；模型未对运动估计不确定性建模；未针对不同机器人平台（双臂、类人）进行迁移验证。

---

## 733. Iteris: Agentic Research Loops for Computational Mathematics

**arXiv ID:** 2606.02484 | [PDF](https://arxiv.org/pdf/2606.02484v1)

**作者:** Leheng Chen `[一作]` (Peking University), Bin Dong `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个面向计算数学开放问题的探究–规划–执行循环式代理系统（Iteris），并将其应用于 CG 与 RCD 的相对效率相位图以及 QRCP 选取逆矩阵的反例构造，最终得到两个正式证明和一个可验证的反例。

**💡 创新点**

创新点在于将代理工作拆分为探索、规划、执行和审查四个可组合的角色，并通过文件共享机制实现长期记忆与信息流通，显著提升了在多模式（数值实验、证明构建、模型冻结等）下的长周期研究效率；同时首次用该框架完成了高阶相位图分析和低相干度下的 QRCP 反例构造。

**🔧 技术方法**

采用了 GPT‑5.5（Codex）驱动的语言模型代理，结合自定义的执行代理（Foundation、Experiment、Proof、Review），以及基于文件的通信协议和固定的探究–规划–执行循环；在证明阶段还使用了 Lean4 形式化验证工具。

**📊 数据集**

数据集主要是理论构造的矩阵实例：对 CG/RCD 分析使用了 Haar 随机正交矩阵和幂律谱的线性系统；对 QRCP 反例使用了构造的低相干度正交行矩阵；此外还通过数值实验生成了实验记录文件。

**📈 对比分析**

与直接使用 GPT‑Pro 的单步推理相比，Iteris 在完成相位图构造时获得了完整的证明结构并提供了可复核的中间文件，QRCP 反例则在形式化验证后实现了可证明的低相干度反例；系统在两项任务中都实现了人类可审计的最终证明，但仍需人工检查与整理。

**⚠️ 局限性**

局限在于系统尚不能完全自主完成验证与推理，需要人工干预进行错误检测、假设审查和最终排版；对复杂的证明路径，系统产生的证明往往过长且不易阅读，且其性能高度依赖于底层语言模型的推理质量。

---

## 734. AGENTCL: Toward Rigorous Evaluation of Continual Learning in Language Agents

**arXiv ID:** 2606.02461 | [PDF](https://arxiv.org/pdf/2606.02461v1)

**作者:** Yiheng Shu `[一作]` (Ohio State University), Yu Su `[通讯]` (Ohio State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向语言代理的持续学习评估框架，构造可控的合成任务流与无序任务流，并引入可塑性、稳定性、泛化三项度量，对非参数记忆方法进行系统评估。

**💡 创新点**

创新点包括：① 用受控任务流显著提升方法区分度；② 定义了 Plasticity Gain、Stability Gain、Generalization Gain 三个专门度量；③ 设计了记忆探测方法 probeMemory，能分层收集交互、洞察与技能信息。

**🔧 技术方法**

采用大型语言模型 Qwen3.5-35B 作为代理核心，结合非参数记忆技术（Mem0、Agent Workflow Memory、Dynamic Cheatsheet、ExpRAG、ReMem 等），并通过两次通过的检索‑求解‑巩固循环实现持续学习。

**📊 数据集**

使用了 CodeEval-Pro、BrowseComp+、MMLU-Pro、AgentBoard BabyAI/ScienceWorld 等公开数据集，并人工合成子任务/复杂任务对构建受控任务流。

**📈 对比分析**

与记忆less 基线 ReAct 进行对比，利用 PG/SG/GG 指标评估方法表现。受控任务流下，现有记忆方案在可塑性提升明显，但在稳定性和泛化方面表现不足，整体性能仍未超过基线。

**⚠️ 局限性**

局限性在于：记忆方案难以在不同任务分布间保持稳定，受控流设计需要人工构造且可能缺乏通用性；评测仅覆盖非参数记忆，未涉及参数化持续学习等方向。

---

## 735. Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models

**arXiv ID:** 2606.02459 | [PDF](https://arxiv.org/pdf/2606.02459v1)

**作者:** Wei Deng `[一作]` (State Key Laboratory of Networking and Switching Technology), Mengshi Qi `[通讯]` (State Key Laboratory of Networking and Switching Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种主动探索的空间推理框架，利用动态认知地图持续记录场景布局，并通过生成可执行的空间断言代码（SAC）在推理过程中实现中间结果验证与密集奖励；

**💡 创新点**

创新点包括①动态认知地图作为持久记忆；②空间断言代码将自然语言推理转化为可执行程序，实现可验证的中间奖励；③主动视图检索的 agentic 流程；④结合监督微调与强化学习的两阶段训练策略；

**🔧 技术方法**

采用 Vision‑Language 模型（以 Qwen2.5‑VL‑3B‑Instruct 为基础），实现监督微调（SFT）生成检索、地图更新和 SAC；使用 GRPO 强化学习通过多维奖励（检索相关度、地图准确度、SAC 验证）进行微调；同时使用 Python 代码生成与执行进行可验证推理；

**📊 数据集**

主要数据集为 MindCube（训练集10k QA，测试集 MindCube‑Tiny 1,050 题，包含 Rotation、Among、Around 子集），并在 EmbodiedBench 上评估跨任务泛化；

**📈 对比分析**

与随机基线、公开 VLM、MindCube、3DThinker 等进行对比，取得 80.5% 的 pass@1（比最佳 3DThinker 提升 13.3 绝对分），在 Rotation 子集提升 29.5 分（相对 53.2%），并在 MindCube-Tiny 上整体准确率较前沿模型高 9.8 分；通过密集奖励显著提升性能；

**⚠️ 局限性**

局限性：①对复杂语言理解能力不足，在 EmbodiedBench 的 Common/Complex 任务中表现为 0；②奖励依赖部分专有数据元信息，可能影响跨数据集迁移；③SAC 生成与执行易受代码错误或模型幻觉影响；④在极端视角或动态场景下仍可能出现地图误差导致推理错误。

---

## 736. Beyond One-shot: AI Agents for Learning in Field Experiments

**arXiv ID:** 2606.02458 | [PDF](https://arxiv.org/pdf/2606.02458v1)

**作者:** Junjie Luo `[一作]`, Gordon Gao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在两轮医疗处方短信实验中，先由行为学专家与GPT-4共同设计13种短信变体（共计444,691次访客），随后使用工具增强的代理式AI系统从第一轮数据中提炼行为原理并生成17种新变体（共计248,448次访客）

**💡 创新点**

创新点在于将数据、信息、知识、智慧（DIKW）层级与多智能体结构相结合，构建可执行代码、统计分析与层级推理的工具增强型代理式AI，使实验数据在下一轮实验中得到可重复、可解释的累计学习

**🔧 技术方法**

采用代码执行、统计分析、结构化DIKW推理、多智能体协作、透明证据链等技术，结合大型语言模型（GPT‑4）与专用分析工具，实现从原始实验数据到行为原理再到干预设计的端到端自动化流程

**📊 数据集**

使用了693,139次患者访问的处方短信实验数据（Stage 1 444,691次，Stage 2 248,448次），数据涵盖多种消息变体、患者子群与交互效应

**📈 对比分析**

对比方法是将人类专家+对话式AI共创的第一轮变体与工具增强代理式AI生成的第二轮变体进行随机对照实验；AI生成的最佳消息CTR达到69.8%，比基线提升6.5个百分点（相对提升10.3%），而不具备实验数据的前沿LLM无法预测有效干预

**⚠️ 局限性**

局限性包括：实验仅评估短期点击率与身份验证，未覆盖长期药物依从性等健康结果；仅在医疗短信场景验证，尚不确定是否适用于其他行业；未与人工专家进行正式对比；隐私与同意问题仍需进一步探讨

---

## 737. HLL: Can Agents Cross Humanity's Last Line of Verification?

**arXiv ID:** 2606.02449 | [PDF](https://arxiv.org/pdf/2606.02449v1)

**作者:** Xinhao Song `[一作]` (Shanghai Jiao Tong University), Dongrui Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个名为 Humanity's Last Line of Verification（HLL）的基准，用来评估多模态智能体在交互式 CAPTCHA 验证任务中的表现，涵盖难度、页面干扰和动态验证三维现实因素。

**💡 创新点**

创新点在于：①将 CAPTCHA 视为端到端的交互式验证瓶颈，超越单纯识别；②将现实因素因子化，提供可控的难度、干扰与交互一致性验证；③设计统一的动态验证规则，使得成功不仅取决于答案，还需符合合理的交互轨迹，形成对过程层次的诊断。

**🔧 技术方法**

使用多模态闭环 GUI 代理框架，将文本、视觉与动作生成整合；实现动态验证器基于任务特定的轨迹规则；对比多款前沿模型（GPT‑5.4、Gemini‑3.1‑Pro、Claude‑Opus‑4.6 等）在 HLL 环境下的交互行为。

**📊 数据集**

使用自建 HLL 数据集，包含十类 CAPTCHA 任务（文本转写、图标序列、滑块对齐、拼图等），并在每类任务上系统地产生不同难度、干扰与动态验证配置的实例。

**📈 对比分析**

采用实例级成功率作为指标，对比静态 vs. 干扰 vs. 难度 vs. 动态验证四种设置。静态平均成功率最高可达 70%，但在干扰条件下下降至约 65%，在“硬”难度下大幅下滑；动态验证进一步降低至约 45%，表明当前最强模型（Claude‑Opus‑4.6）在过程一致性方面仍显不足。

**⚠️ 局限性**

局限性包括：①基准仅涵盖十类 CAPTCHA，未覆盖所有生产环境的多样化验证机制；②动态验证规则虽较真实，但仍未实现完整的反机器检测逻辑；③实验集中在闭环 GUI 代理，未考察跨平台或移动端细节；④缺乏对错误原因的细粒度可解释性分析。

---

## 738. Poking Around in the Dark: Why a Shared Understanding of Components Matters

**arXiv ID:** 2606.02442 | [PDF](https://arxiv.org/pdf/2606.02442v1)

**作者:** Felix Reichmann `[一作]` (Ruhr University Bochum), Simon Koch `[通讯]` (University of Insbruck)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了软件构件定义与识别机制，评估并对比了五个主流SBOM生成工具在六种编程语言中的组件检测效果，并基于手工分析揭示工具实现差异与盲点。

**💡 创新点**

首次系统化提出“安全视角下的组件定义”并构建了完整的组件纳入机制（CIM）框架；同时开发了可复现的测试套件，用于验证SBOM工具在源、构建、运行时三阶段的覆盖率。

**🔧 技术方法**

利用标准化SBOM格式（CycloneDX、SPDX）以及各语言的包管理器（pip、gradle、go.mod、composer、conan）、构建工具和静态/动态分析技术；对工具源码进行手工审计以解释检测行为。

**📊 数据集**

自定义构建的控制测试项目（Python、Java、PHP、Go、Rust、C），每种语言提供“复制文件”“vendored submodule”“managed dependency”“build linking”“runtime linking/sideloading”六类实验，形成225个单元测试；同时在测试环境中动态生成lock/manifest文件以获得ground truth。

**📈 对比分析**

通过自动化脚本生成SBOM并与ground truth逐项对比，计算每种工具对不同CIM的检测率；结果显示所有工具均能高效识别managed components（>95%），但对build和runtime组件的检测率低于40%，且存在版本错误、漏报与误报等问题。

**⚠️ 局限性**

受限于：①未覆盖所有语言/包管理器与构建环境；②仅测试直接集成的组件，未评估传递依赖；③依赖docker化环境，真实CI/CD环境差异可能影响检测；④手工评估耗时，缺乏自动化验证；⑤未对容器化部署等Runtime SBOM场景进行深入探讨。

---

## 739. ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning

**arXiv ID:** 2606.02433 | [PDF](https://arxiv.org/pdf/2606.02433v1)

**作者:** Zhensheng Wang `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了开域表格问答中未来数据预测与推理的新任务ODTQA‑FoRe，并构建了首个面向此任务的真实房价数据集，同时设计了基于LLM的三阶段协作框架TimeFore，分别负责检索、预测与答案合成；

**💡 创新点**

创新点在于：①首次引入未来预测的开域表格问答任务与对应基准数据集；②通过Retriever–Forecaster–Analyzer三角色的LLM代理拆解问题，显著提升了检索精度、预测准确率和答案标准化；

**🔧 技术方法**

技术手段包括：检索端采用先简化查询为表格标题后BM25检索，SQL自动生成与执行反馈；预测端利用函数调用调用TimesNet进行缺失值插补，再用TimeXer进行12个月预测；分析端使用BERT分类判断问答类型，并通过数值提取模块输出规范答案；

**📊 数据集**

使用数据集为ODTQA‑FoRe，基于2022–2024年10个中国城市的房产交易记录，覆盖约11,149个项目、288张历史表和144张未来表，总计28,507问答对；

**📈 对比分析**

实验对比了vanilla基线以及Qwen3 30B、Qwen3 Next 80B、GPT‑OSS 20B/120B、GLM 4.5 Air等主流LLM；TimeFore在时间序列预测指标（MSE、MAE、MRE）和推理指标（准确率、F1）上均显著优于基线，说明专用时序模型与LLM协同提升性能；

**⚠️ 局限性**

局限性包括：①预测模型未纳入宏观经济、政策或环境等外部变量；②数据集仅覆盖房产领域，未验证跨域泛化；③模板生成方式可能不足以覆盖所有复杂真实问句，限制了数据多样性。

---

## 740. NDPP-Grasp: Non-Differentiable Physical Plausibility Constraint-Guided Task-Oriented Dexterous Grasp Generation

**arXiv ID:** 2606.02432 | [PDF](https://arxiv.org/pdf/2606.02432v1)

**作者:** Qiuchi Xiang `[一作]` (Lancaster University), Jun Liu `[通讯]` (Lancaster University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为NDPP‑Grasp的框架，在任务对齐的抓取扩散模型的去噪过程中直接注入非可微物理可行性约束，实现物理可行性与任务对齐的同时优化。

**💡 创新点**

创新点在于将非可微物理约束的引导转化为基于随机最优控制的梯度自由去噪指导，并通过摊销前瞻策略实现实时有效的指导。

**🔧 技术方法**

技术主要包括扩散模型、随机最优控制、Tweedie预测、摊销前瞻采样以及零阶梯度估计与强化学习对比。

**📊 数据集**

在DexGYSNet和DexTOG‑80K两个基准数据集上进行实验。

**📈 对比分析**

与现有任务导向抓取方法及后处理改进方法相比，NDPP‑Grasp在物理可行性、任务对齐和多样性指标上均显著提升，同时推理时间保持在10‑20毫秒级。

**⚠️ 局限性**

主要局限是对前瞻长度H的选择仍需经验调参，且对极端非可微约束的实时评估可能受限于计算资源。

---

## 741. Spectral Audit of In-Context Operator Networks

**arXiv ID:** 2606.02427 | [PDF](https://arxiv.org/pdf/2606.02427v1)

**作者:** Zhiwei Gao `[一作]` (Brown University), George Em Karniadakis `[通讯]` (Brown University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出一种基于雅可比矩阵的频谱审计方法，用以评估在上下文学习（ICON）中模型对 PDE 作用算子局部动力学结构的把握。

**💡 创新点**

创新点在于：①将模型对查询函数的局部线性响应（雅可比）投影到 Fourier 基底，从而得到频率响应、相位旋转和模间耦合等多维谱特征；②与传统的预测误差对比，能揭示模型在高频、相位或耦合方面潜在的错误，提升模型可解释性与可靠性。

**🔧 技术方法**

使用 Transformer‑based ICON 作为基准模型；利用自动微分求解雅可比向量积；在 Fourier 基底上计算增益、相位误差、块误差及特征映射；对比真值线性化算子（PDE 变分方程）的谱特征；在合成 PDE 示例上进行实验。

**📊 数据集**

数据集为三类 1D 周期 PDE 的人工合成样本：传输方程、布格斯方程和 Allen–Cahn 方程。每个示例通过随机初始条件生成 5 对演示样本与 1 个查询样本，网格 100 点，隐藏参数在给定区间内采样。

**📈 对比分析**

与仅使用 L² 预测误差评价方式相比，审计能揭示高频衰减失真、相位错误、模间耦合异常及提示一致性缺陷。实验表明：在训练区间内，模型的预测误差低且谱误差可接受；但当训练超出区间或提示不一致时，谱误差显著增大，说明模型在局部动态上的失真。

**⚠️ 局限性**

局限性包括：①方法依赖 Fourier 基底，难以直接推广到非周期、非一维或不规则几何；②需要精确计算雅可比，计算成本较高；③仅在合成 PDE 上验证，缺乏对真实工程数据的实证；④对训练带宽高度敏感，过宽/过窄都可能导致谱评估失真。

---

## 742. GC-MoE: Genomics-Guided Cell-Type-Specific Mixture of Experts for Histology-Based Single-Cell Spatial Transcriptomics

**arXiv ID:** 2606.02424 | [PDF](https://arxiv.org/pdf/2606.02424v1)

**作者:** Kaito Shiku `[一作]` (Kyushu University), Muhammad Nabeel Asim `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种从H&E图像预测单细胞空间转录组的方法GC‑MoE，能够在无需单细胞ST测量的情况下恢复单细胞基因表达

**💡 创新点**

创新点在于引入基因组学引导的细胞类型专属混合专家框架（GC‑MoE）并结合细胞类型特异的共表达感知预测器（CAP）以及邻居细胞注意力模块（C2CA），实现细胞类型结构化表达模式的建模

**🔧 技术方法**

技术上使用了路径学基础模型提取细胞中心特征，路由网络对专家进行软分配，CAP通过scRNA‑seq构建共表达图并应用GCN，C2CA使用轻量级交叉注意力捕获邻居细胞上下文

**📊 数据集**

实验数据集包括10x Xenium、COAD与IDC三组单细胞ST数据，共计超过三百万细胞，基因集合取自相应临床样本的高变基因

**📈 对比分析**

与单细胞ST基线GHIST以及将Spot‑level方法（ST‑Net、HisToGene、BLEEP、TRIPLEX）改写为单细胞版本对比，GC‑MoE在三个数据集上均取得最高PCC（平均0.260），相较最强基线提升约0.030

**⚠️ 局限性**

局限性主要是单细胞ST测量的dropout高、信号弱，CAP虽部分缓解但仍难以完全消除丢失，未来需要更稳健的观测或多源辅助训练

---

## 743. AutoForest: Automatically Generating Forest Plots from Biomedical Studies with End-to-End Evidence Extraction and Synthesis

**arXiv ID:** 2606.02403 | [PDF](https://arxiv.org/pdf/2606.02403v1)

**作者:** Massimiliano Pronesti `[一作]` (IBM Research), Yufang Hou `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个端到端的系统 AutoForest，能够直接从未结构化的生物医学论文中自动生成出版级森林图。

**💡 创新点**

创新点在于：① 自动化生成 ICO（Intervention‑Comparator‑Outcome）建议；② 用 LLM 进行数值证据提取和风险偏倚评估；③ 通过“思考过程”文本解释提升可解释性；④ 将所有步骤（文档解析、提取、合成、绘图）整合为一体化流水线。

**🔧 技术方法**

技术手段包括：Transformer‑based LLM（Claude Sonnet 4.5）进行文本提取与推理；pdf‑to‑markdown 转换与表格解析；R 语言的 meta 包完成统计合成；ReactJS+FastAPI 负责前后端交互；Prompt 设计与 RoB2 “宏”规则实现风险偏倚评估。

**📊 数据集**

使用数据集为 18 篇 Cochrane 系统综述共 56 篇研究，包含 206 张表格与 32 张森林图，作为自动化与手动流程对比的实验基准。

**📈 对比分析**

对比方法：在 8 名参与者（4 名专家 4 名学生）上进行三种工作流（手动 RevMan、AutoForest 自动化、AutoForest 人机协同）比较；评估指标为完成时间、数据提取与 RoB 评估准确率以及用户体验评分。结果显示：AutoForest 自动化将时间近半；准确率超过 80%；人机协同进一步提升至 90%（数据提取）和 79%（RoB），学生利用系统的准确率甚至超过手动专家，证明 AI 协助能显著提升效率与质量。

**⚠️ 局限性**

局限性：① 受限于仅 8 名参与者的样本规模，结果可能不具普适性；② 只测试了平行组试验，未覆盖多臂或交叉设计；③ 系统聚焦于森林图生成，未涵盖系统综述的检索、筛选与选取等前置步骤；④ 缺乏直接在原文中高亮提取证据的可视化追踪功能。

---

## 744. From Time to Space: The Impact of Linearity in Higher-Order Datalog

**arXiv ID:** 2606.02394 | [PDF](https://arxiv.org/pdf/2606.02394v1)

**作者:** Angelos Charalambidis `[一作]` (Harokopio University of Athens), Panos Rondogiannis `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出并研究了一类称为分层线性高阶Datalog（Stratified Linear Higher-Order Datalog）的逻辑程序语言片段，并证明该片段在不需要数据库排序的情况下，能够捕获空间复杂度层级[(k‑1)]，即(k+1)阶程序对应于[(k‑1)]。

**💡 创新点**

创新点包括：① 将传统的一阶Datalog中的线性概念推广到高阶语义，并通过层次化与高阶应用的结合，形成了新的分层线性高阶Datalog；② 通过构造高阶计数模块和对图灵机的空间有限模拟，证明(k+1)阶程序捕获[(k‑1)]；③ 提出一种基于自顶向下的空间高效评估机制，证明(k+1)阶程序的查询可在exp_k‑1(n^d)空间内完成；④ 讨论将二阶分层线性程序编译为量化布尔公式（QBF）进行求解的可行性。

**🔧 技术方法**

所用技术包括高阶Datalog的语法与语义定义、层次化（stratification）与线性约束、递归计数模块（高阶数值表示）、图灵机模拟、Savitch定理启发的空间高效查询机与表达式求值器、以及潜在的QBF求解器。

**📊 数据集**

论文未使用具体数据集，研究基于理论证明与形式化分析，主要关注复杂度与可实现性。

**📈 对比分析**

与传统一阶Datalog或ASP的比较通过理论复杂度层级进行。作者证明：① 分层线性高阶Datalog的(k+1)阶等价于空间类[(k‑1)]；② 其评估空间为exp_k‑1(n^d)，比全称解释法所需的“tower指数”空间低得多；③ 通过QBF编译方案可利用现有QBF求解器，预示可实现的高阶查询。

**⚠️ 局限性**

限制包括：① 需要严格的层次化与线性约束，无法直接处理非分层或非线性程序；② 目前仅给出理论证明与高阶计数模块的概念实现，实际实现与性能评估尚待后续工作；③ 适用阶数受限（如二阶实现已讨论），更高阶实现的空间/时间开销和编译难度仍是挑战。

---

## 745. Retrieve What's Missing: Coverage-Maximizing Retrieval for Consistent Long Video Generation

**arXiv ID:** 2606.02479 | [PDF](https://arxiv.org/pdf/2606.02479v1)

**作者:** Minseok Joo `[一作]` (Korea University), Hyunwoo J. Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种 Coverage-Maximizing Retrieval（CMR）深度基记忆检索框架，用于提升长时段自回归视频生成的几何一致性。

**💡 创新点**

创新点包括：利用预训练3D先验生成目标视图覆盖图作为轻量级3D证据；通过残余覆盖增益进行迭代帧选择，减少检索冗余；采用滑动窗口深度缓存以实现可扩展的几何推理。

**🔧 技术方法**

使用的技术包括：预训练3D先验模型（如DepthFormer）提供深度和相机姿态；像素级覆盖图推断；残余覆盖增益算法进行帧选择；滑动窗口深度缓存机制。

**📊 数据集**

在 RealEstate10K 和 DL3DV10K 两个大型3D场景数据集上进行实验。

**📈 对比分析**

与 WorldMem（基于视角重叠）和 VMem（显式3D重建）等基线对比，CMR 在几何一致性误差(ME_t3R)上最多降低29%，同时延迟降低至基线的1/5以内，表明在保持低延迟的同时显著提升一致性。

**⚠️ 局限性**

局限性包括：依赖深度估计的准确性，覆盖图可能忽略纹理细节；对动态物体或极端遮挡场景的适应性仍有限。

---

## 746. Learning When to Translate for Multilingual Reasoning

**arXiv ID:** 2606.02465 | [PDF](https://arxiv.org/pdf/2606.02465v1)

**作者:** Deokhyung Kang `[一作]` (POSTECH), Gary Geunbae Lee `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多语言推理中模型对语言理解的边界，提出了Luar框架，让RLM在需要时选择调用翻译，而不是始终翻译或直接推理。

**💡 创新点**

通过基于结果的语言理解边界估计与两阶段训练（SFT+边界感知GRPO）实现翻译调用的自适应决策，显著提升低资源语言推理性能。

**🔧 技术方法**

采用监督微调、强化学习（GRPO）、翻译调用前缀、边界感知奖励以及 token‑级策略梯度等技术。

**📊 数据集**

使用 PolyMath（多语言数学推理）和 MMLU‑ProX‑Lite（STEM 推理）两大基准数据集，并用翻译后的训练集（Arabic、Thai、Swahili）进行训练。

**📈 对比分析**

与基于提示、外部检测器、纯监督及 RL 基线对比，在多语言基准上 Luar 平均提升约 10% 的准确率，翻译调用率约 20%，同时保持较低延迟。

**⚠️ 局限性**

仅覆盖有限语言族与数学/ STEM 任务，未检验更广泛任务与语言，且仍需依赖外部翻译服务。

---

## 747. Food Noise & False Safety: A Systematic Evaluation of How LLMs Fail to Adapt to Eating Disorder Queries with Clinician Feedback

**arXiv ID:** 2606.02444 | [PDF](https://arxiv.org/pdf/2606.02444v1)

**作者:** Giulia Pucci `[一作]` (University of Aberdeen), Arabella Sinclair `[通讯]` (University of Aberdeen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估开放权重LLM在饮食失调相关查询中的安全性与不良输出，结合专家标注和自动检测方法。

**💡 创新点**

首次在提示层面引入风险分类，揭示食物噪声在“中性”查询中的普遍存在，并量化不同提示特征对模型行为的影响。

**🔧 技术方法**

使用规则基拒绝检测、词表匹配进行食品噪声识别，并辅以人工标注的安全评估。

**📊 数据集**

构造了11,712条人工设计的提示组合，涵盖性别、疾病披露、虚假披露和请求类型。

**📈 对比分析**

对三款7-9B参数开放模型进行对照实验，发现无模型在高风险提示下始终拒绝，安全率仅约45%。

**⚠️ 局限性**

实验仅覆盖三款开放模型，未检验付费模型或真实用户交互，且食品噪声词表可能无法覆盖所有语义。

---

## 748. PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning

**arXiv ID:** 2606.02443 | [PDF](https://arxiv.org/pdf/2606.02443v1)

**作者:** Yusong Zhao `[一作]` (Chinese University of Hong Kong), Pinjia He `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个针对多模态大语言模型的视频安全预警基准PaSBench-Video，包含风险与非风险视频，提供帧级风险开始与事故时间标注，并设计了实时流式评估与误报测量流程。

**💡 创新点**

创新点在于：①提出连续视频实时安全预警评估框架，覆盖时间校准、内容定位与误报评估；②提供跨驾驶、医疗、日常生活与工业四领域、740条视频的帧级标注数据集；③设计四级递进指标（Any-Det、Any-Hit、First-Hit、First-Act），揭示模型时机与内容判断的薄弱环节。

**🔧 技术方法**

使用多模态LLM推理、视频流式采样、结构化警报输出，并通过模型过滤+人工筛选生成数据；评估时采用统一提示、温度0、四级指标与无风险误报率的计算。

**📊 数据集**

PaSBench-Video共740条视频（481条风险视频+259条无风险视频），来源包括公开真实视频（DADA‑2000、Nexar、OOPS、SmartHome、RADSV）与文本到视频合成（健康、工业），每条风险视频均标注风险开始帧、事故帧、风险源与避险动作。

**📈 对比分析**

在13种开源与专有MLLM上统一评估，发现最高的严格指标First‑Act@1仅达20%，Recall与误报率呈正相关（ρ=0.64），不同领域差异显著（日常生活表现优于驾驶）。整体性能低且受限于时机定位与内容正确性。

**⚠️ 局限性**

局限性包括：①风险开始帧标注主观性，缺乏多评者一致性；②无风险样本有限，误报率按视频计数而非时长；③内容定位评估仅覆盖少量样本；④模型推理延迟与成本高，难以实时部署；⑤数据集覆盖范围有限，未包含所有安全情景。

---

## 749. On the Scaling of PEFT: Towards Million Personal Models of Trillion Parameters

**arXiv ID:** 2606.02437 | [PDF](https://arxiv.org/pdf/2606.02437v1)

**作者:** Mind Lab `[一作]` (Mind Lab), Murphy Zhuang `[通讯]` (Mind Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 PEFT 在个人模型上的三轴扩展（Scale Up、Scale Down、Scale Out），通过在 1T 参数 MoE 上的 Trillion‑Scale LoRA‑RL、低秩 LoRA 稳定性实验、δ‑mem 可写记忆与 Context Learning 写策略等，验证了大模型+低秩适配器可实现持续学习、可靠记忆与多样化个人化。

**💡 创新点**

将 PEFT 重新定位为持久个性化状态容器，系统性阐述三轴尺度关系，提出 OLoRA‑tail、δ‑mem 等新型适配器与写策略，并证明在大模型上实现持续学习、可写记忆与集群多样性的方法。

**🔧 技术方法**

使用 LoRA 低秩适配、RL‑native OLoRA‑tail 初始化、Router Replay R3、δ‑mem 在线记忆、Context Learning 写策略、MinT 基础设施、投票/路由集群多样化等技术。

**📊 数据集**

采用 Qwen3 系列模型、MATH、DapO、AlfWorld、HotpotQA、MemoryAgentBench、DishNameBenchmark、OASIS 社交仿真、AIME24 等数据集进行评测。

**📈 对比分析**

通过对 1T 参数 MoE 进行 LoRA‑RL 训练、不同秩 LoRA 的奖励/准确率对比、δ‑mem 与静态 LoRA 的基准准确率比较，以及 OASIS 中用户适配器与共享模型的多样性与社交指标对比，发现 LoRA 在奖励、准确率上提升约 20–30%，并在社交模拟中提升极化距离、同伴模块度等指标。

**⚠️ 局限性**

需解决多系统对齐（路由一致性、稀疏架构、训练‑推理差异）、低秩适配器对初始化与超参敏感、记忆容量有限、极端规模硬件与分布式开销未完全解决，以及用户隐私与可解释性挑战。

---

## 750. Not All Errors Are Equal: A Systematic Study of Error Propagation in Large Language Model Inference

**arXiv ID:** 2606.02430 | [PDF](https://arxiv.org/pdf/2606.02430v1)

**作者:** Yafan Huang `[一作]` (University of Iowa), Guanpeng Li `[通讯]` (University of Florida)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一个可配置、可确定性的LLM故障注入框架（LLMFI），利用它在3款开源LLM（Phi‑3.5‑mini‑instruct、DeepSeek‑LLM‑7b‑chat、Gemma‑7b）上对13个代表性推理任务（推理、跨语言、数学、代码生成）进行大规模错误注入实验，系统分析了错误在不同层级、推理阶段、数值类型和KV缓存中的传播规律，并基于17条经验总结提出四种低成本软件级容错策略。

**💡 创新点**

创新点在于：①首次为LLM推理提供细粒度、任务/模型/阶段无关的故障注入框架；②通过大规模实验揭示了LLM推理中阶段/层级/维度扩张/数值精度对错误传播的影响规律；③基于发现提出的四种针对性容错方案（混合精度推理、选择性校验和保护、双首词生成、动态样本提示）在不改动模型或硬件的前提下显著提升可靠性。

**🔧 技术方法**

技术手段包括：PyTorch + HuggingFace Transformers 运行时钩子实现单比特计算/内存错误注入；自定义调度器分离 Prefill、First‑Token、Decode 等推理阶段；基于 GPGPU‑Sim/NVBit 兼容的 GPU 级别故障注入；使用权重与激活位翻转、位索引控制和多维度量化等多种故障模型；以及多任务评估脚本。

**📊 数据集**

使用的任务数据集共13个，覆盖四大范畴：推理（ARC、BoolQ、HellaSwag、OpenBookQA、PIQA、TruthfulQA、WinoGrande、MMMLU、XCOPA）、数学（GSM8K、MATH500）和代码生成（HumanEval、MBPP）。

**📈 对比分析**

通过将每个任务的基线准确率（无故障）与在不同推理阶段注入单比特错误后的准确率进行对比，量化错误对性能的影响。实验表明错误可导致准确率下降最高约5.01%（例如HumanEval的生成任务），但在某些阶段（如Decode）影响极小；在混合精度/校验保护下，错误导致的准确率下降可降低到1%以下，且额外开销仅为3–15%。

**⚠️ 局限性**

局限性包括：①仅评估单比特错误，未覆盖多比特或跨字节错误；②主要针对中等规模稠密模型，MoE 仅以 41.9B Phi‑MoE 为例；③未考虑 ECC、冗余硬件或 GPU 侧其他软错误；④容错方案针对 deterministic decoding，可能不适用于采样式生成；⑤实验规模受 GPU 资源限制，未验证更大模型或不同算子实现的普适性。

---

## 751. K-BrowseComp: A Web Browsing Agent Benchmark Grounded in Korean Contexts

**arXiv ID:** 2606.02404 | [PDF](https://arxiv.org/pdf/2606.02404v1)

**作者:** Nahyun Lee `[一作]` (Chung-Ang University), Seungone Kim `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了首个面向韩语环境的 Web 浏览代理基准 K‑BrowseComp，包含 300 题人工验证子集与 100 题基于失败模式的合成子集。

**💡 创新点**

创新点在于将代理式评估迁移至韩语语境，构造可检索但难以验证的浏览任务，并通过失败模式引导的合成生成提升基准难度与诊断性。

**🔧 技术方法**

采用浏览代理评估框架（Perplexity Search、tool‑call harness）以及校准误差评估与多轮搜索预算控制。

**📊 数据集**

使用 300 题手工编写且经母语评审验证的韩语浏览问题和 100 题基于 LLM 生成且通过多轮过滤的合成问题。

**📈 对比分析**

对比实验显示闭源前沿 LLM 的 Pass@1 仅 30–46%，韩语开源模型低至 0–10%；合成子集进一步压低至 0–26%，表明模型在检索后状态维护和答案最终化方面仍存在显著瓶颈。

**⚠️ 局限性**

局限包括数据集规模有限且领域分布不均，评估受单一搜索后端与调用预算约束，合成子集与验证子集表面特征差异，以及 Web 证据随时间变化需持续验证。

---

## 752. Welfare-Optimal Classification with Accuracy Auctions

**arXiv ID:** 2606.02435 | [PDF](https://arxiv.org/pdf/2606.02435v1)

**作者:** Bana Sadi `[一作]` (Technion), Nir Rosenfeld `[通讯]` (Technion)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种“精确度拍卖”机制，将机器学习分类任务转化为在保证报酬诚实的前提下最大化社会福利的优化问题；

**💡 创新点**

创新点在于将机制设计与学习算法结合，利用单参数拍卖的单调性实现真实价值的激励兼容，并证明对线性稳定学习器支付成本上界为常数；

**🔧 技术方法**

核心技术包括基于加权损失的风险最小化、Myerson定理的支付计算以及算法稳定性分析；

**📊 数据集**

实验数据涵盖合成多元高斯分布以及公开的Folktables（ACSIncome）数据集；

**📈 对比分析**

通过在不同α值下调节权重，对比准确率与福利的Pareto曲线，结果显示福利提升可达5%（或更高）且付费用户极少；

**⚠️ 局限性**

局限性包括需要用户在训练时支付（可能不公平）、仅适用于线性或稳定学习器、假设用户完全理性且价值独立于标签错误等。

---

## 753. Towards Multidisciplinary Summarization of Hospital Stays: Efficient Sentence-Level Clinical Provenance Categorization

**arXiv ID:** 2606.02487 | [PDF](https://arxiv.org/pdf/2606.02487v1)

**作者:** Baris Karacan `[一作]` (University of Illinois Chicago), Andrew D. Boyd `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个临床 Provenance 分类管道，利用大语言模型对多源 NICU 记录进行句子级别的分组和标注。

**💡 创新点**

发现模型规模对跨域迁移至关重要，70B 模型在量化后仍能保持结构化约束并显著提升跨域性能；同时使用 QLoRA 4‑bit 量化实现 70B 模型的高效适配。

**🔧 技术方法**

采用 Llama‑3.1‑8B‑Instruct 与 Llama‑3.3‑70B‑Instruct，使用 QLoRA+LoRA 微调并进行 4‑bit NF4 量化。

**📊 数据集**

源域：MedSecId（2,002 篇 MIMIC‑III ICU 记录，17 类标注，1,562 篇用于训练/验证/测试）。目标域：NICU Pilot Set（3 篇多学科出院摘要，227 句子级标注，25 类细粒度）。

**📈 对比分析**

在源域宏 F1≈0.92‑0.93，权重 F1≈0.94。跨域评估显示，70B 微调后宏 F1从0.52提升至0.59，权重 F1从0.57提升至0.60；8B 微调提升有限（宏 F1≈0.53）。

**⚠️ 局限性**

受限于 NICU 评估集仅 227 句子且来自 3 篇摘要，样本量小导致统计显著性不足；跨域语义细粒度差异大，标注成本高。

---

## 754. X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding

**arXiv ID:** 2606.02482 | [PDF](https://arxiv.org/pdf/2606.02482v1)

**作者:** Peiwen Sun `[一作]` (Chinese University of Hong Kong), Xiangyu Yue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了首个多流实时理解基准 X-Stream，包含 4,220 题答对、932 个视频、451 个多流片段，并通过三种信号分路（空间、时间、语义）将多流视频输入大语言模型进行评估。

**💡 创新点**

创新点在于：① 通过双重验证（充分性与必要性）确保每道题确实需要多流信息；② 将 MLLM 视为“裸装置”多路复用器，系统化分析其在多流场景下的性能折衷；③ 提供完整的在线推理与 LLM-as-a-Judge 评测框架，揭示现有模型在主动推理与多流整合方面的瓶颈。

**🔧 技术方法**

技术主要包括：大规模多模态 LLM（如 Gemini‑3‑Pro、Qwen‑3.5、InternVL‑3.5 等）；三种多路复用策略（空间分割、时间分割、语义分割）；基于 token 带宽 C_max 的预算控制；在线推理管道；人类双轮审阅与 LLM‑as‑a‑Judge 评测。

**📊 数据集**

使用自建 X‑Stream 数据集（857 小时原始多流视频，覆盖驾驶、体育、机器人、日常、聊天、监控、直播、接口等八大领域），并从 Egolife、Seamless‑Interaction、Comma2K‑19 等公开数据集重组与仿真得到多流视角与多设备场景。

**📈 对比分析**

通过即时、回退、前瞻三种回答模式以及综合得分，对比了专有与开源 MLLM 的表现；专有模型 Gemini‑3‑Pro 领先，但整体准确率仅约 50%，而前瞻问答更易失误；在多路复用实验中，空间分割最擅长跨流引用，时间分割在 token 宽裕时表现最佳，语义分割在 token 受限或流数 ≥3 时最优。

**⚠️ 局限性**

局限性包括：① 现有多流模型对主动推理与多流同步仍表现差，准确率低；② 多路复用策略受 token 带宽与流数限制，缺乏统一最优方案；③ 基准多流多样性有限（多数为双流，3‑5 流场景不足）；④ 需要更多真实同步、专业级采集的多流数据；⑤ 当前评测仍侧重视觉+音频，缺乏更丰富的感知模态。

---

## 755. Physics-Informed Residuals for Adaptive Mesh Refinement in Finite-Difference PDE Solvers

**arXiv ID:** 2606.02475 | [PDF](https://arxiv.org/pdf/2606.02475v1)

**作者:** Henry Kasumba `[一作]` (Makerere University), Ronald Katende `[通讯]` (Kabale University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种混合策略：先用已训练或部分训练的 PINN 计算离网残差，将其转化为单元级细化指标；随后在经典有限差分求解器上执行自适应网格细化，最终得到高质量的数值解。

**💡 创新点**

创新点在于：① 将 PINN 残差仅作为诊断工具，而非最终求解器；② 将连续残差映射为可用于经典求解器的离散细化指示；③ 通过阈值和 Dörfler 两种标记规则验证该方法的有效性。

**🔧 技术方法**

使用的技术包括：PINN 训练（自动微分、损失加权）、残差采样与单元级指标构造、阈值与 Dörfler 标记、非均匀有限差分求解、以及标准的 solve–estimate–mark–refine 循环。

**📊 数据集**

实验数据集为三组人工合成测试：1D viscous Burgers 方程（有高精度参考解）；2D 非线性薛定谔方程的制造解；3D Navier–Stokes 方程的制造解。

**📈 对比分析**

比较方法：与均匀细化、随机细化、梯度细化、参考基准细化以及单独 PINN 进行对比。结果显示：在 Burgers 实验中，PINN‑threshold 与 Dörfler 细化在 60/58 个自由度下分别取得相对 L²误差 0.021067/0.021264，明显优于 192 个自由度的均匀细化；匹配自由度时误差降幅超过 67%；在 2D/3D 代理实验中，PINN 细化优于随机细化和单独 PINN，但略逊于梯度细化或均匀细化。

**⚠️ 局限性**

局限性包括：① 仅在 1D Burgers 方程的完整求解器实验中得到充分验证，2D/3D 仅为代理测试；② PINN 残差质量高度依赖训练过程，训练成本显著；③ 未与成熟的残差/跳跃/恢复/adjoint 估计器结合；④ 需要多种随机种子、复杂几何和更真实工程问题的进一步验证。

---

## 756. MASER: Modality-Adaptive Specialist Routing for Embodied 3D Spatial Intelligence

**arXiv ID:** 2606.02463 | [PDF](https://arxiv.org/pdf/2606.02463v1)

**作者:** Hilton Raj `[一作]` (Boston University), Vishnuram AV `[通讯]` (Boston University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MASER框架，训练五个针对不同传感器模态的DoRA适配器，并学习一个神经路由器根据问题语义动态选择最佳模态；

**💡 创新点**

创新点在于将模态适配与路由相结合，利用冻结的SBERT句子嵌入与三层MLP实现轻量级、基于语义的模态选择，而非传统的多模态集成或随机选择；

**🔧 技术方法**

使用Qwen2‑VL‑2B冻结视觉语言模型骨干、DoRA低秩适配器、SBERT句子编码器、三层GELU MLP路由器，以及轻量级LLM判定器进行答案评估；

**📊 数据集**

在Open3D‑VQA数据集上进行实验，该数据集包含RGB图像、深度图、点云、相机姿态和文本答案，覆盖四种场景类型；

**📈 对比分析**

与单模态适配器、无适配器基线以及基于随机森林的路由进行对比，MASER路由在测试集上实现了51.33%的路由准确率，推理时均时延仅为1.56秒/问，且每题只调用一次适配器；

**⚠️ 局限性**

局限包括：图像模态在准确率上仍落后于单独图像适配器；路由器仅基于问题文本，难以判断视觉信息质量，导致深度/图像问题的路由错误；以及使用的oracle标签将模态性能与速度混合，可能偏向轻量级模态。

---

## 757. Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior

**arXiv ID:** 2606.02453 | [PDF](https://arxiv.org/pdf/2606.02453v1)

**作者:** Xiang Li `[一作]` (National University of Singapore), Kenji Kawaguchi `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于指导潜能后验的初始化方法（Diversity-inducing Initialization），通过 Langevin 动力学在噪声空间中偏向低潜能区域，从而显著提升扩散模型和流匹配模型的生成多样性。

**💡 创新点**

创新点在于：①从初始化视角重新定义多样性，使用指导潜能后验重新加权标准高斯先验；②利用 Langevin 动力学进行可插拔的概率采样，既保持先验正则化又能探索多样模式；③该方法与轨迹干预互补，可与现有采样增强技术联合使用。

**🔧 技术方法**

核心技术：Langevin 动力学采样、基于 Tweedie 潜能的指导潜能估计、可插拔的初始化框架，兼容扩散模型和流匹配模型。

**📊 数据集**

使用的数据集包括 ImageNet-1K（用于类图像生成）与 Stable Diffusion v1.4、SD v3.5（文本图像生成），并利用 MS-COCO、Lexica、Tuxemon 及 GPT‑4 生成的多样化提示集进行评估。

**📈 对比分析**

与基线扩散、轨迹指导方法（PG、CADS、IG）以及 Sharpness‑Aware Initialization (SAIL) 进行对比；结果显示在 Vendi、recall、coverage 等多样性指标上提升 30‑40%，FID 下降 10‑20%，并在多模型、多提示下显著扩大多样性‑质量 Pareto 前沿。

**⚠️ 局限性**

局限性：①需手动调节温度 τ 以平衡先验与多样性；②额外的 Langevin 采样步骤会增加推理时间；③随机性导致指标波动；④目前仅适用于有条件的生成任务，无法直接用于无条件生成。

---

## 758. Reason-Then-Retrieve for CoVR-R with Structured Edit Prompts and Dense-Sparse Fusion

**arXiv ID:** 2606.02450 | [PDF](https://arxiv.org/pdf/2606.02450v1)

**作者:** DongQing Liu `[一作]` (Beijing University of Posts and Telecommunications), HongWei Ji `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了零训练的“先推理后检索”框架，在CoVR‑R任务中生成结构化视频描述并结合稠密与稀疏检索；

**💡 创新点**

创新点在于使用结构化提示让模型保留动作顺序、手物交互、最终状态等细节，采用token权重池化得到稠密向量，并与TF‑IDF稀疏检索融合；

**🔧 技术方法**

使用Qwen3.5‑27B零训练提示生成文本、token‑weighted hidden‑state pooling、稠密检索cosine相似度、TF‑IDF检索以及分支融合；

**📊 数据集**

使用CoVR‑R数据集（包含WebVid和Something‑Something V2两个子集）进行评测；

**📈 对比分析**

在验证集上达到R@1 80.81、R@5 94.86、R@10 97.11、R@50 98.59；在测试集上达到R@1 89.73、R@5 95.79、R@10 96.63、R@50 97.98，展示了比单一分支更优的性能；

**⚠️ 局限性**

局限性在于仍受生成描述中缺失关键动作或状态导致的表征不匹配影响，且系统未进行任务微调，可能在更复杂编辑场景下表现不佳。

---

## 759. Geometry-Aware Implicit Memory for Video World Models

**arXiv ID:** 2606.02436 | [PDF](https://arxiv.org/pdf/2606.02436v1)

**作者:** Zhengxuan Wei `[一作]` (Nanjing University), Qi Fan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 GIM-World，一种几何感知的隐式记忆框架，用于长时延视频世界模型，能够把可变长度的历史压缩为固定大小的记忆，并在训练时通过几何监督让记忆携带跨视角几何信息。

**💡 创新点**

创新点包括：①利用 3D 基础模型（VGGT）作为教师，对记忆状态进行相机可查询的几何监督；②设计轻量化的 Transformer 编码器和几何监督头；③提出信息导向的互信息贪心剪枝规则，使历史编码成本可控。

**🔧 技术方法**

技术手段包括：轻量级 Transformer 记忆编码器、相机可查询几何监督头、互信息驱动的高斯过程剪枝、流匹配训练目标、DiT 扩散生成器、VGGT 3D 基础模型教师。

**📊 数据集**

在 MIND 数据集上进行实验，该数据集包含约 100 条第一人称和 100 条第三人称的 1080p/24fps 视频，每条约三分钟。

**📈 对比分析**

与显式记忆（Matrix-Game 2.0、MIND-World、FramePack、Context-as-Memory）和隐式记忆（VideoSSM）基线在相同 backbone、记忆容量和训练计划下进行对比，评估记忆一致性、动作精度和几何一致性；GIM-World 在所有度量上均优于基线，尤其几何重投影得分提升至 81.7（第一人称）/87.1（第三人称）。

**⚠️ 局限性**

局限性包括：训练阶段需要 3D 基础模型和几何监督头，推理阶段虽然已丢弃，但训练成本较高；目前仅在 MIND 上验证，泛化到其他场景或更大规模尚未评估；记忆容量固定，极长历史仍可能受限；模型仍受扩散框架的时间步限制。

---

## 760. Bridging the Sim-to-Real Gap in Semiconductor Visual Program Synthesis via Input Binarization

**arXiv ID:** 2606.02434 | [PDF](https://arxiv.org/pdf/2606.02434v1)

**作者:** Yusuke Ohtsubo `[一作]` (Hitachi, Ltd.), Tatsuya Sasaki `[通讯]` (Hitachi, Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于视觉语言模型的视觉程序合成框架，将半导体检测图像转换为可编辑的DSL代码，实现对电路几何参数的精准控制与训练数据的主动生成。

**💡 创新点**

创新点在于：①在推理阶段对真实SEM图像采用全局阈值二值化，显著消除纹理噪声，桥接合成到真实的域差距；②提出适用于曼哈顿布局的轻量级DSL，既能压缩表达又能保持几何细节；③利用单一VLM架构，改变量语言特定的DSL即可迁移到不同检测域。

**🔧 技术方法**

主要技术包括：Qwen3‑VL‑8B视觉语言模型的监督微调；通过DSL渲染器生成大规模合成图像–DSL对；推理时对真实图像做全局阈值二值化；使用IoU、Dice、BF1、ASSD、SkF1等多维指标进行评估。

**📊 数据集**

使用 MIIC 实际SEM图像集作为测试集；合成数据共19,200对（训练18,900对，评估300对）以及300个与训练分布一致的合成样本。

**📈 对比分析**

方法对比采用二值化输入与原始灰度输入；在 MIIC 上，二值化输入将Dice提升至0.5256（从0.4393），IoU、BF1、SkF1等指标均有提升；但性能仍低于合成数据的无域差上限（Dice 0.6340）。

**⚠️ 局限性**

局限性包括：二值化虽抑制纹理噪声，却无法消除形状细节差异导致的误差；模型在复杂布局下召回率下降；长序列生成受限，需改进DSL的token效率和模型的长上下文推理能力。

---

## 761. Fostering Emotional Perspective-Taking: An Exploration of Affective Face-Tracking Interactions in the VR Narrative Rekindle

**arXiv ID:** 2606.02425 | [PDF](https://arxiv.org/pdf/2606.02425v1)

**作者:** Hector Fan `[一作]` (Northeastern University), Mark Sivak `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在《Rekindle》VR叙事游戏中加入基于面部追踪的情绪交互机制，让玩家通过表情匹配记忆碎片的预设情绪才能收集。

**💡 创新点**

将玩家的情绪表情作为主动输入，既是交互控制又是叙事意义构建的核心，促成情绪视角转移与深层情感参与。

**🔧 技术方法**

Meta Quest Pro 的 FACS‑based blendshapes、AU‑映射、Du 等复合情绪模型，以及自定义的面部校准算法。

**📊 数据集**

使用游戏内的十个情绪标记记忆碎片及 Meta Quest Pro 的面部追踪数据；未使用公开情绪数据集。

**📈 对比分析**

计划采用被试间实验，将原版与情绪交互版对比；通过 SAM 量表、面部情绪记录与访谈评估情感投入与叙事理解；目前未给出实验结果。

**⚠️ 局限性**

缺乏对玩家情绪偏差的实时提示，需玩家自我调节；依赖特定 VR 设备的面部追踪；实验尚未完成，功能与效果尚待验证。

---

## 762. Investigating and Alleviating Harm Amplification in LLM Interactions

**arXiv ID:** 2606.02423 | [PDF](https://arxiv.org/pdf/2606.02423v1)

**作者:** Ruohao Guo `[一作]` (Georgia Institute of Technology), Alan Ritter `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出多轮危害放大评估基准和主动监测框架TrajSafe

**💡 创新点**

1) 首次系统性构建基于真实威胁的多轮危害放大基准；2) 设计结构化干预动作空间并采用树式强化学习训练监测器，使其在多轮交互中精准干预；3) 证明相对单轮评估，多轮交互显著提升模型危害性。

**🔧 技术方法**

树式强化学习（Tree-based RL）、结构化干预动作（Engage, Probe, Shape, Divert, Hard Refuse）

**📊 数据集**

HarmBench、Hex-Phi、OR-Bench等单轮安全基准的多轮化扩展，生成12类威胁的多步计划与对话模拟

**📈 对比分析**

与Llama-Guard、GPT-oss-safeguard、Prompting-based Monitor、SFT Monitor等对比；在三种目标模型（Llama‑3.1‑8B、Qwen3‑4B、GPT‑5‑mini）上，TrajSafe将危害得分从86.36%降至9.85%（Llama‑3.1‑8B）或从14.39%降至0.76%（GPT‑5‑mini），同时保持最低的过度拒绝率（≈11%）并几乎不损害通用能力。

**⚠️ 局限性**

基于LLM模拟的恶意用户缺乏真实人类多样性；只覆盖文本单会话；对抗者可能开发规避技术，监测器需持续在线学习。

---

## 763. Bridging the Last Mile of Time Series Forecasting with LLM Agents

**arXiv ID:** 2606.02497 | [PDF](https://arxiv.org/pdf/2606.02497v1)

**作者:** Yuhua Liao `[一作]` (Trip.com Group), Zhenhua Zhang `[通讯]` (Trip.com Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了“最后一公里”时序预测框架，通过LLM代理在基线预测基础上进行情境化修订；

**💡 创新点**

创新点在于把后期修订视为受限的可审计动作序列，结合证据检索、范围变换、点覆盖等操作，并支持多阶段拆解与跨会话记忆；

**🔧 技术方法**

采用LLM代理+工具（检索、日历、记忆库）与约束性动作接口，基线使用TimesFM，后期修订通过代码执行型LLM完成；

**📊 数据集**

使用了国内航空日票销售的日度数据（2024‑01‑01至2026‑05‑05）；

**📈 对比分析**

与Prophet和TimesFM对比，框架在节假日窗口将MAE降低80‑88%，MAPE降至≈32%；全周期MAPE仅18.9%，相较Prophet的27.3%显著提升；

**⚠️ 局限性**

局限包括缺乏大规模基准、仅支持文本/日历上下文、检索来源可靠性待提升、未覆盖多模态输入和细粒度用户审批环节。

---

## 764. Monitoring Agentic Systems Before They're Reliable

**arXiv ID:** 2606.02494 | [PDF](https://arxiv.org/pdf/2606.02494v1)

**作者:** Marisa Ferrara Boston `[一作]` (Reins AI), Heather Frase `[通讯]` (Veraitech)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套针对早期部署的 agentic 系统的监控与自动分流方法，利用三维度（质量、适用性、效率）和三范围（运行内、跨运行、结构）对系统行为进行分解，并通过方差信号识别不同失败类型，随后使用基于 FMEA 的严重度分类实现自动化的治理分流。

**💡 创新点**

创新点在于：1) 将方差作为第一类信号，区分监控范围并捕捉结构性缺陷；2) 将传统的 FMEA 体系引入 AI 运维，实现严重度与人力投入的可调分流；3) 提出从结构诊断到错误检测再到可靠性跟踪的成熟度分阶段模型，强调监控在系统早期的重要价值。

**🔧 技术方法**

核心技术包括：1) 规则/统计/LLM 评估器在三个维度上产生分数；2) 在运行内和跨运行上计算 z‑score 并设定阈值触发告警；3) 方差（CV）用以表征监控范围；4) 基于 MIL‑STD‑882E 的四层严重度模型（L1‑L4）与 FMEA 风险计算；5) deterministic 路由规则将 L3 处理为自动跟踪，L2/ L1 触发人工调查。

**📊 数据集**

使用的是一个 120 份文档捆绑（每份包含 10 条 CSV 账务文件）的合成审计测试集，分为 20 份干净基线和 100 份注入错误的捆绑；错误类型覆盖算术、跨文档比较与时间序列三大类，每类有 4 个难度等级。实验共 220 次运行。

**📈 对比分析**

与传统聚合指标（均值）相比，该方法能在结构性缺陷主导的早期阶段准确定位缺陷并实现约 97% 的人工审计量缩减。实验结果显示：within‑run 监控检测到 45.7 条 L3 发现；cross‑run 监控 24% 的发现为 L2；structural 监控 100% L2。没有使用真实生产数据，仅在合成测试中验证；在该测试中监控有效揭示结构缺陷，任务级错误被掩盖。

**⚠️ 局限性**

局限性包括：1) 仅在合成测试环境下验证，缺乏真实生产数据的泛化能力；2) 只覆盖审计场景的错误类型，未检验对其他领域的适用性；3) 现有实验仅验证 Stage 1 的成熟度模型，后续 Stage 2/3 的效果尚需纵向追踪；4) 评估器多为规则/统计，未涉及 LLM 评估器的偏差影响。

---

## 765. $O(n +f(k))$: Truly Linear FPT

**arXiv ID:** 2606.02492 | [PDF](https://arxiv.org/pdf/2606.02492v1)

**作者:** Benjamin Merlin Bumpus `[一作]` (University of São Paulo), Ella Yates `[通讯]` (University of Glasgow)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文定义并探讨了真正线性固定参数可解性（TLFPT），并展示其作为线性固定参数可解性（LFPT）的严格子集。研究了在大数据背景下，如何在仅线性依赖于输入大小的情况下处理NP难题。

**💡 创新点**

创新点在于提出了TLFPT的概念，并通过对比LFPT，展示了在处理大规模数据时，真正线性算法的必要性和有效性。

**🔧 技术方法**

使用了基于深度优先搜索和广度优先搜索的技术，结合了线性时间的算法和数据结构。

**📊 数据集**

没有使用特定的数据集，主要是理论探讨和算法设计。

**📈 对比分析**

与传统的LFPT方法相比，TLFPT在处理大规模数据时表现出更优的性能，尤其是在输入规模极大的情况下，TLFPT算法的运行时间显著低于LFPT算法。

**⚠️ 局限性**

限制在于TLFPT的定义和应用仍然需要进一步的理论支持，且在实际应用中可能面临实现复杂性的问题。

---

## 766. Places in the Wild: A Large, High-Resolution RAW Photograph Dataset for Ecologically Valid Vision Research

**arXiv ID:** 2606.02481 | [PDF](https://arxiv.org/pdf/2606.02481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 767. Expressivity of congruence-based architectures for DNNs on positive-definite matrices

**arXiv ID:** 2606.02490 | [PDF](https://arxiv.org/pdf/2606.02490v1)

**作者:** Antonin Oswald `[一作]`, Estelle Massart `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 768. RASER: Recoverability-Aware Selective Escalation Router for Multi-Hop Question Answering

**arXiv ID:** 2606.02488 | [PDF](https://arxiv.org/pdf/2606.02488v1)

**作者:** Yuyang Li `[一作]` (Karlsruhe Institute of Technology), Tobias Käfer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级路由器 RASER，能在一次性检索（RAG）后决定是否进行额外的桥接或迭代检索，从而提升多跳问答的效率。

**💡 创新点**

创新点在于：①基于一次检索的六个低成本特征构建恢复可行性判断；②采用梯度提升机（GBM）实现无额外 LLM 调用的决策；③提供两种路由器（RASER‑2、RASER‑3），分别实现阈值门控与成本敏感的动作选择。

**🔧 技术方法**

使用技术包括：密集检索（Nomic‑Embed‑Text‑v1.5）、大语言模型（GPT‑OSS‑120B、Gemma‑3‑31B、Llama‑3.1‑8B、Llama‑3‑8B、Phi‑4‑mini、Mistral‑S‑119B）、梯度提升机（scikit‑learn GBM）以及成本‑准确性权衡的 argmax 公式。

**📊 数据集**

评估数据集：HotpotQA、2WikiMultiHopQA 与 MuSiQue，覆盖不同难度与桥接需求的多跳问答任务。

**📈 对比分析**

与传统的 always‑PRUNE、迭代检索（IRCoT）、ChainRAG 等基线对比，RASER 在保持与 SOTA 相当的 F1 的同时，token 使用量仅为 always‑PRUNE 的 39–57%，且比迭代检索更具成本效益。

**⚠️ 局限性**

局限性包括：基线实现为简化版，未完全复现 KiRAG/ChainRAG；实验规模有限（每个模型 200–500 条样本）；大模型可能依赖自身记忆导致检索改进空间受限。

---

## 769. Ghost Tool Calls: Issue-Time Privacy for Speculative Agent Tools

**arXiv ID:** 2606.02483 | [PDF](https://arxiv.org/pdf/2606.02483v1)

**作者:** Bardia Mohammadi `[一作]` (Max Planck Institute for Software Systems), Laurent Bindschaedler `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文发现并研究了在工具增强型语言代理的推测性工具调用中产生的“幽灵工具调用”，提出了Speculative Tool Privacy Contracts来控制推测阶段的隐私泄露，并在多种策略下进行评估。

**💡 创新点**

创新点在于首次将“issue-time”隐私视为一种首要效应，正式定义幽灵调用的隐私面，构造多字段标签与五种动作的合同框架，并证明其安全性。

**🔧 技术方法**

使用了Python原型运行时、标签化工具适配器、合同监视器与决策函数，并结合Claude Opus 4.7、Haiku‑4.5、GPT‑4o‑mini等LLM进行推理与攻击实验。

**📊 数据集**

采用了三大合成语料库、AgentDojo 66‑任务子集以及公开搜索接口（Brave/DuckDuckGo）等数据集进行实验。

**📈 对比分析**

通过在相同推测前沿下对十二种策略（包括Naive、No‑Spec、SIA、ACL、Pre‑Scrub、Late‑Scrub、Rewrite、Shadow、Gate、Taint、Taint‑F、Drop）进行配对重放实验，比较隐私泄露恢复率与任务成功率，发现issue‑time变换显著降低泄露，性能损失仅在1–5%之间。

**⚠️ 局限性**

主要限制包括：需要完整中介才能保证安全；标签准确度对策略效果影响大；模型与工具提供方的信任假设未处理；对网络/共享缓存侧信道的覆盖不足；以及Shadow替代质量对任务成功率的潜在影响。

---

## 770. MCP-Persona: Benchmarking LLM Agents on Real-World Personal Applications via Environment Simulation

**arXiv ID:** 2606.02470 | [PDF](https://arxiv.org/pdf/2606.02470v1)

**作者:** Wenhao Wang `[一作]` (Zhejiang University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MCP-Persona 基准，专门评估大型语言模型在真实个人化 MCP 工具（如 Slack、Lark、Rednote 等）上的代理表现。

**💡 创新点**

创新点包括：①将工具、用户上下文和任务三要素组合成可复现的评估框架；②开发 Tool‑Traverse、Context‑Tree、Persona‑Gen 三大方法，能在保持隐私的前提下模拟 12 个真实 MCP 服务器并生成 173 个人工验证任务；③首次在社交媒体、协作平台等高度个性化场景中对代理进行系统性评测。

**🔧 技术方法**

技术手段：MCP 架构 + LLM 驱动的功能调用遍历与错误生成；代码化模拟器（用 LLM 合成可执行 Python 逻辑）；树形上下文生成与动态状态管理；任务生成链采样与模糊化（隐式上下文去除）；LLM 判断器用于检查点与执行评估。

**📊 数据集**

数据集：12 个真实 MCP 服务器（Slack、Lark、Rednote、Notion 等）+ 173 个人类验证任务 + 大量功能调用记录与上下文树实例。

**📈 对比分析**

比较方法：在 10+ SOTA 代理（GPT‑5、Claude‑Sonnet‑4.5、Gemini‑3‑Pro 等）上测量检查点准确率、执行准确率、SR@0.8 等指标；对比 Tool‑Traverse 与 Vanilla 模拟，Tool‑Traverse 在行为一致性、响应相似度上提升 30% 以上；整体发现即使最强模型准确率也低于 50%，跨服务器任务更差。

**⚠️ 局限性**

局限性：模型在隐式信息提取、跨工具协同、长上下文管理等方面仍表现不佳；基于 LLM 的代码生成可能引入偏差；基准覆盖面虽然扩展，但仍难以完全复现所有真实环境；对安全与隐私的深度评估尚待进一步完善。

---

## 771. Speculative Sampling For Faster Molecular Dynamics

**arXiv ID:** 2606.02455 | [PDF](https://arxiv.org/pdf/2606.02455v1)

**作者:** Arthur Kosmala `[一作]` (FAIR at Meta), Brandon Wood `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种名为Langevin Speculative Dynamics (LSD) 的显式并行采样框架，用于加速基于机器学习势能的分子动力学模拟，并保证与目标模型轨迹分布一致。

**💡 创新点**

将Speculative Sampling思想推广到二阶Langevin动力学，引入反射最大耦合实现动量更新的高接受率，并通过流水线异步验证实现显著加速。

**🔧 技术方法**

反射最大耦合、ABOBA拆分积分、Langevin热浴、异步流水线、Speculative Error Correction、MMD高维分布对比等技术。

**📊 数据集**

使用FCC Cu、bulk water、Li10GeP2S12 (LGPS) 等典型材料系统的原子构型与相应机器学习势能（UMA、Orb-v3-direct等）进行实验。

**📈 对比分析**

对比目标MLIP、不同草稿模型与LSD组合在速度、拒绝率、热力学量（温度、扩散系数）及MMD相似性等指标；LSD在小尺寸系统上可实现约4–9倍加速，并保持与目标模型一致。

**⚠️ 局限性**

拒绝率随原子数增长而升高，导致大体系（>10^3原子）加速受限；此外流水线依赖足够计算资源，若设备不足可能无法实现预期加速。

---

## 772. Spatial-Temporal Decoupled Reference Conditioning for Identity-Preserving Text-to-Video Generation

**arXiv ID:** 2606.02441 | [PDF](https://arxiv.org/pdf/2606.02441v1)

**作者:** Yuheng Chen `[一作]` (Shanghai Jiao Tong University), Jiangning Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ST-DRC框架，实现文本条件下的身份保持视频生成；

**💡 创新点**

创新点在于将参考图像编码到视频VAE潜在空间并通过Temporal‑Adjacent Spatial‑Shifted RoPE实现时空分离，结合参考增强与面部身份监督以及三流条件无监督引导；

**🔧 技术方法**

采用视频Diffusion Transformer LTX‑2.3、VAE编码、RoPE、ArcFace面部识别、CLIP等技术；

**📊 数据集**

使用VIP‑200K数据集进行训练与评估；

**📈 对比分析**

与ConsisID、Phantom、VACE、IPVG‑STD等基线对比，ST‑DRC在身份相似度、CLIP评分、视频美学与运动平滑度等指标上均表现最优；

**⚠️ 局限性**

局限在于仅针对人脸，难以直接扩展到多主体或复杂背景场景；对高帧率或长时序生成的效果尚未充分验证。

---

## 773. Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation

**arXiv ID:** 2606.02417 | [PDF](https://arxiv.org/pdf/2606.02417v1)

**作者:** Miaomiao Cai `[一作]` (National University of Singapore), See-Kiong Ng `[通讯]` (National University of Singapore)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种针对多行为推荐的鲁棒方法，先通过频域特征级谱过滤去除行为内的噪声，再以全局语境注意力进行跨行为可靠性加权融合，最终提升目标行为（如购买）预测效果。

**💡 创新点**

创新点在于：①将嵌入空间从空间域转到特征‑频率域，用FFT实现可学习的谱调制；②在谱空间对每个行为自适应地抑制噪声；③以已去噪的全局表示为上下文锚点，计算行为间兼容度并做可靠性感知的注意力聚合，避免频繁但不可信的辅助信号主导。

**🔧 技术方法**

主要技术包括：Fast Fourier Transform (RFFT/IRFFT) 的特征‑频率变换、可学习的对角谱调制矩阵、全局上下文注意力机制（兼容度计算+维度‑级归一化）、残差全局骨干、LightGCN 作为基础图编码器以及 BPR 损失。

**📊 数据集**

在三个真实多行为数据集上进行评估：Taobao、TMall（电商日志）和 MovieLens‑10M（将评分映射为 dislike/neutral/like 三类隐式行为）。

**📈 对比分析**

与 LightGCN、SimGCL、MBGCN、MB‑CGCN、PKEF、BCIPM、S‑MBRec、MISSL、DeMBR、UIPL、MBID 等 12 个基线对比，本文方法在大多数指标（HR@10/20、NDCG@10/20）上取得最高或相当接近的分数，尤其在 NDCG 方面持续领先；在加噪实验中表现出更强的鲁棒性，降噪效果明显优于现有方法。

**⚠️ 局限性**

局限性包括：①需要额外的 FFT 与注意力运算，增加了模型复杂度和推理时延；②对残差系数 α 的调优具有一定敏感性，需要针对不同数据集手工搜索；③谱过滤依赖于 embedding 维度和频率分辨率，过低维度可能导致信息丢失；④在极度稀疏或行为极少的场景下，仍可能受限于全局结构的稳定性。

---

## 774. Structure-Informed Multiple Sequence Alignment: A Formal Model and Hardness Results

**arXiv ID:** 2606.02408 | [PDF](https://arxiv.org/pdf/2606.02408v1)

**作者:** Yoshiki Kanazawa `[一作]` (Keio University), Rodney Van Meter `[通讯]` (Keio University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了结构信息引入的多序列比对（MSA）问题，并给出了其决策问题的 NP 完全性与优化问题的无 PTAS 结果。

**💡 创新点**

首次将结构信息以二元重叠得分形式与固定的配对字符串得分结合，形成可进行复杂度分析的组合优化模型。

**🔧 技术方法**

使用组合优化、归约证明、NP 完全性分析、无 PTAS 证明等理论计算机科学技术。

**📊 数据集**

未使用实际数据集，研究完全基于理论分析。

**📈 对比分析**

由于是理论证明，没有实验比较，结果表明问题在给定固定得分方案下本质上不可多解，优化问题亦难以逼近。

**⚠️ 局限性**

局限在于只考虑固定得分方案且未给出近似算法或实用解法，实际生物信息学应用仍需进一步研究。

---

## 775. Edge Prediction for Roof Wireframe Reconstruction with Transformers

**arXiv ID:** 2606.02406 | [PDF](https://arxiv.org/pdf/2606.02406v1)

**作者:** Gustav Hanning `[一作]` (Lund University), Viktor Larsson `[通讯]` (Lund University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文在S23DR 2026挑战中提出了一种基于Transformer的端到端方法，直接从稀疏SfM点云与语义分割/深度图中恢复屋顶线框。

**💡 创新点**

创新点包括动态语义优先的点云子采样、融合来自Gestalt/ ADE20k 的类别特征以及将点投影至冻结自编码器产生的多视角Gestalt特征的局部上下文融合，并在Transformer解码器中使用学习查询直接生成边。

**🔧 技术方法**

采用了DETR启发的Transformer编码器-解码器架构、交叉注意力、Hungarian匹配、L1与加权交叉熵损失、姿态编码、以及多尺度点采样和自编码器生成的局部特征。

**📊 数据集**

使用了“HoHo 22k”数据集，该集包含约22k个场景的稀疏SfM点云、Gestalt与ADE20k语义分割图、MoGe-2深度图及对应的地面真值线框。

**📈 对比分析**

在公开/私有测试集上与官方基准（手工和学习型）以及冠军提交进行比较，取得HSS 0.6476，位列第二，显著优于基准但略逊于冠军。

**⚠️ 局限性**

限制包括对点云密度和姿态敏感的绝对位置编码、对短边和小结构的预测不足，以及对不同视角和点云缺失的鲁棒性仍有提升空间。

---

## 776. Explainable Forensics of Manipulated Segments in Untrimmed Long Videos

**arXiv ID:** 2606.02402 | [PDF](https://arxiv.org/pdf/2606.02402v1)

**作者:** Yue Feng `[一作]` (Nanjing University of Aeronautics and Astronautics), Jie Qin `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出长未裁剪视频中的AI生成内容检测与解释任务，构建了包含12,472段长视频的TASLE基准数据集，并提出两阶段的MSLoc检测与解释框架。

**💡 创新点**

创新点包括：①将任务细化为时序AI生成段落定位与可解释化；②构建细粒度的边界级与对象级解释标注；③在MSLoc中引入四分类边界敏感检测和基于MLLM的精细边界定位与异常感知推理。

**🔧 技术方法**

技术手段包括：DeMamba+滑窗的边界感知前置提议；Q-Former+DAM与EAM模块的边界与事件特征融合；MLLM（Trace/Qwen3‑VL）进行边界重定位与理由生成；异常感知损失促进对可解释性推理的学习。

**📊 数据集**

使用自研的TASLE数据集（含真实-伪造混合、时序边界、语义理由），并与GenVideo、GenVidBench、GenBuster、GenWorld等公开数据集对比；对测试集进行源/工具/类型的多维度评估。

**📈 对比分析**

与传统短片检测器（D3、BusterX++、Qwen3‑VL‑8B）和全局MLLM定位器（Trace）相比，MSLoc在F1_Det、F1_Loc和Rationale Quality（RQ）上分别提升约20%–30%；在跨域与未见生成工具的评估中，MSLoc‑PR阶段显著弥补前置漏检，性能提升超过10%。

**⚠️ 局限性**

局限性在于：①检测召回受前置提议阶段限制，漏检难以恢复；②依赖可见生成痕迹，面临更逼真生成模型时表现下降；③目前仅利用视觉信号，缺少音频、物理一致性等多模态信息；④需要持续更新数据集以适应新兴生成技术。

---

## 777. From Zero to Hero: Training-Free Custom Concept Spawning in World Models

**arXiv ID:** 2606.02575 | [PDF](https://arxiv.org/pdf/2606.02575v1)

**作者:** Kiymet Akdemir `[一作]` (Virginia Tech), Pinar Yanardag `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种名为 SPAWN 的推理时无训练概念注入方法，能够在已运行的自动回归世界模型中将用户指定的视觉概念（图像或文本）插入场景，并在后续的摄像机运动中自然保持。

**💡 创新点**

核心创新是利用图像‑视频背骨的上下文记忆首槽（Anchor）在短窗口内被替换为概念潜变量，然后恢复原始 Anchor，使概念通过模型自身的记忆机制在后续生成中持续保留，从而无需任何模型改造或额外训练即可实现可控概念生成。

**🔧 技术方法**

技术手段包括：WorldPlay 自动回归世界模型、DiT 变换器、RoPE/PRoPE 位置编码、VAE 编码、上下文记忆重构与 Anchor 替换、窗口注入；同时支持文本提示或概念图像作为输入。

**📊 数据集**

实验基于公开的 WorldPlay 检查点，在 200 条 8 秒长的 roll‑out 上进行评估，涵盖多种场景与概念类别，未使用任何专门的训练数据。

**📈 对比分析**

与 HunyuanVideo、Wan 2.2、WorldPlay 等基线进行对比，使用 VBench 评价指标。SPA 在 Overall Score、Dynamic Degree、Camera Motion 等维度表现最佳，且在用户研究中获得最高的视觉质量、提示跟随度、动作控制和概念图像忠实度评分。

**⚠️ 局限性**

局限性：仅在推理时进行修改，受限于原模型的摄像机轨迹与交互原语；无法在原模型无法到达的区域注入概念；不扩展动作空间，且对长远可达区域的影响仍有限。

---

## 778. AdaCodec: A Predictive Visual Code for Video MLLMs

**arXiv ID:** 2606.02569 | [PDF](https://arxiv.org/pdf/2606.02569v1)

**作者:** Haowen Hou `[一作]` (Shanghai Jiao Tong University), Jiaqi Wang `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计 AdaCodec 预测视觉编码，将视频拆分为 I‑帧（完整 ViT tokens）和 P‑帧（运动+残差 P‑tokens），只在预测成本高时才使用 I‑帧，降低视觉 token 重复。

**💡 创新点**

①将预测编码从播放端转移为 MLLM 视觉接口；②自适应 GOP 与 P‑tokenizer 的结合；③双分支 token 化和两阶段训练方案，显著提升 token 效率与推理速度。

**🔧 技术方法**

预测编码、宏块匹配与 SAD 运动搜索、ViT 基础编码器、P‑tokenizer、两阶段对齐训练、token 预算自适应与时序压缩。

**📊 数据集**

使用 11 个视频理解基准：MLVU、LongVideoBench、LVBench、TempCompass、MotionBench、TOMATO、Video-MME、MVBench、NExT-QA、PerceptionTest、EgoSchema。

**📈 对比分析**

与 Qwen3‑VL‑8B RGB baseline 以及多种闭源/开源模型进行基准对比；在 1/7 token 预算下保持或提升准确率，在匹配预算下均超越 baseline，token 数量减少 84%+，TTFT 下降 5×，E2EL 下降 4×。

**⚠️ 局限性**

局限性：固定输入分辨率、固定 P‑token 数量、未评估流式视频场景、未对 token 数量做动态适配。

---

## 779. Policy-based Foveated Imaging and Perception

**arXiv ID:** 2606.02565 | [PDF](https://arxiv.org/pdf/2606.02565v1)

**作者:** Howard Xiao `[一作]` (Stanford University), Gordon Wetzstein `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种实时、基于策略的视网膜成像系统，利用双流高分辨率相机在采集阶段动态指向任务相关区域；

**💡 创新点**

创新点在于将注意力策略与感知任务耦合，形成预测式、实时的采集决策循环，避免传统下采样或时间下采样的无差别信息丢失；

**🔧 技术方法**

采用YOLO式低分辨率注意力检测、匈牙利匹配+卡尔曼滤波运动预测、Set Transformer策略网络，以及基于MobileNetV3的高分辨率特征提取；

**📊 数据集**

在SoccerNet Tracking、RoadText-1K和Static ALOHA三个公开数据集上进行评估；

**📈 对比分析**

与空间下采样、时间下采样及GT Oracle做对比，使用相同像素预算时策略式视网膜采样在三项任务上均显著优于基线，且多场景下接近或超越全分辨率结果；

**⚠️ 局限性**

局限性包括对时间连续性的依赖、额外的实时决策开销、以及当前只能支持有限数量ROI，未来需扩展更灵活的多ROI/层次化采样策略。

---

## 780. Permissive Safety Through Trusted Inference: Verifiable Belief-Space Neural Safety Filters for Assured Interactive Robotics

**arXiv ID:** 2606.02562 | [PDF](https://arxiv.org/pdf/2606.02562v1)

**作者:** Haimin Hu `[一作]` `[通讯]` (Johns Hopkins University), Haimin Hu (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在交互式机器人安全控制中，提出了一种基于 conformal prediction 的推理感知安全过滤器（belief‑space safety filter）验证与部署方法，重点在推理可信区间内进行安全保证。

**💡 创新点**

创新点在于将推理质量（runtime inference 的可靠性）与安全过滤器验证相结合，形成“推理可信区间”，从而显著降低因稀有推理失败导致的安全过滤器保守性。

**🔧 技术方法**

技术手段包括：信念空间安全过滤器（belief‑space safety filter）框架、Isaacs 方程与 HJ 逼近、分层 CP（conformal prediction）验证、推理质量评分函数 h_L、联合采样与安全覆盖参数 ϵ 的统计校准。

**📊 数据集**

使用的主要数据集是 18 维车辆‑行人交互仿真环境（来自文献 [cite]），包括行人意图、语义类别与目标位置的离散隐藏类型，生成 10⁶ 条样本用于训练推理评分器，2×10⁵ 条测试样本验证其误判率。

**📈 对比分析**

与传统在物理空间中构造的安全过滤器做比较；在相同的安全覆盖 ϵ 下，推理可信区间方法在测试中实现 99.21% 的安全率（比基线提升约 1.9%），拒绝率仅为推理误判率（≈4.7%），而基线需扩大安全集 δ 以达到同样覆盖率，导致拒绝率显著上升。

**⚠️ 局限性**

局限性包括：依赖可靠的推理质量评分函数 h_L；假设交互过程保持可交换性和 i.i.d.；在推理严重失效或分布漂移时安全保证不再成立；目前未实现对安全过滤器和推理模块的闭环迭代优化。

---

## 781. LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation

**arXiv ID:** 2606.02553 | [PDF](https://arxiv.org/pdf/2606.02553v1)

**作者:** Qixin Hu `[一作]` (NVIDIA), Yukang Chen `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种检索增强的自回归视频生成框架 LongLive‑RAG，利用已生成的潜在历史进行检索来提升长视频生成质量。

**💡 创新点**

将自回归视频生成视为对自身生成潜在的检索问题，构建轻量级检索编码器并引入 Window Temporal Delta Loss 与平滑正则来优化检索嵌入。

**🔧 技术方法**

使用自回归视频扩散模型、检索增强生成、潜在自编码器嵌入、Top‑K检索、滑动窗口注意力、∞‑RoPE、Deep Forcing 等技术。

**📊 数据集**

采用 MovieGenBench 128 条提示，并用 Qwen2.5‑7B‑Instruct 对提示进行细化。

**📈 对比分析**

与三种 AR backbone（Causal‑Forcing、Self‑Forcing、LongLive）以及两种现有长视频策略（∞‑RoPE、Deep Forcing）在 30s/60s/120s 生成长度下使用 VBench‑Long 评估，LongLive‑RAG 在所有基准上获得最低平均排名，显著提升主体、背景一致性、运动平滑度和成像质量。

**⚠️ 局限性**

检索质量对生成结果影响较大，对检索嵌入的训练依赖；检索开销在极长序列或大模型时可能增加；当前仅在自回归视频扩散框架下验证，未测试其他生成模式。

---

## 782. Why Not Hyperparameter-Friendly Optimisation? A Monotonic Adaptive Norm Rescaling Approach For Long-Tailed Recognition

**arXiv ID:** 2606.02526 | [PDF](https://arxiv.org/pdf/2606.02526v1)

**作者:** Shuo Zhang `[一作]` (University of Oxford), Tingting Zhu `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在长尾识别中提出Self-Adaptive Monotonic Normalization（SAMN）方法，直接在分类器重训练阶段通过Pool Adjacent Violators Algorithm（PAVA）对每类权重范数进行单调性约束，从而避免传统正则化超参数的敏感性。

**💡 创新点**

创新点在于：1) 从类条件分布角度阐释长尾性能瓶颈来源于稀缺类别的欠拟合；2) 提出无超参数的自适应单调归一化方案，利用PAVA直接对权重范数排序；3) 证明SAMN可与现有SOTA方法无缝集成并提升性能。

**🔧 技术方法**

使用PAVA进行等距回归约束权重范数；在训练中将权重拆分为方向与大小，使用指数缩放保证正值；结合两阶段解耦训练策略；在实验中与多种基线（CE、WD、SLAS、GLMC等）比较。

**📊 数据集**

在CIFAR10-LT、CIFAR100-LT、ImageNet-LT、iNaturalist2018四个长尾基准数据集上进行评估，覆盖不同类别数量（Many/Medium/Few）及不同不平衡因子。

**📈 对比分析**

在所有数据集上，SAMN与基线相比平均提升1–6%准确率，且在高不平衡因子或细粒度数据集（CIFAR-100、ImageNet）上表现尤为突出；与SOTA方法（GLMC、SLAS、WD+MaxNorm等）对比，往往实现或逼近最优结果。

**⚠️ 局限性**

局限性在于：1) 对极大样本类的性能略有下降；2) 需要在训练后期进行一次权重重塑，增加一次计算；3) 对PAVA的排序依据（类频率或第一阶段权重）仍需经验选择，虽然是分类参数，但在极端场景下可能影响效果。

---

## 783. FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes

**arXiv ID:** 2606.02523 | [PDF](https://arxiv.org/pdf/2606.02523v1)

**作者:** Liuliu Chen `[一作]` (University of Melbourne), Mike Conway `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个针对自杀主题表情包的细粒度数据集 FigSIM，并对其中的比喻、严重程度与自杀相关内容进行多标签标注，随后对多模态与大型语言模型在三项检测任务上的表现进行基准评测。

**💡 创新点**

创新点在于：①首次针对自杀表情包提出细粒度、多维度标注方案；②将比喻、严重程度与自杀内容三大维度结合，提升对幽默隐喻性表达的理解；③通过大规模基准实验展示多模态与 LLM 在此任务中的优势与不足。

**🔧 技术方法**

技术方法包括：使用 Argilla 进行人工标注；基于 BERT、RoBERTa、ResNet、ViT 等单模态预训练模型；对 CLIP、BLIP‑2 等多模态模型进行微调；使用 GPT‑5、Claude‑Sonnet、Gemini‑3‑Pro 等大型多模态语言模型进行零/少量提示实验；并对结果进行 macro‑F1 与 weighted‑F1 评估。

**📊 数据集**

所使用的数据集为 1,049 张来自 Reddit 上专门讨论自杀的子版块的表情包，按 60:20:20 的比例划分为训练、验证与测试集。

**📈 对比分析**

评估方法采用宏 F1 与加权 F1 分数，对 16 个基线模型在三项任务（比喻检测、严重程度识别、自杀相关内容检测）进行对比。最佳多模态模型（如 GPT‑5‑mini 或 Gemini‑3‑Pro）在比喻检测、严重程度识别分别达约 70% 和 72% 的宏 F1，内容检测约 58%；实验显示模型在高严重度样本上易低估严重程度，并且多模态提示效果受上下文影响。

**⚠️ 局限性**

局限性包括：①数据仅来自单一 Reddit 社区，缺乏跨文化与跨平台泛化性；②比喻与严重程度标注具有主观性，且缺乏临床诊断依据；③高严重度样本比例低，导致模型难以学习；④部分多模态模型受安全过滤限制，影响评估完整性。

---

## 784. CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning

**arXiv ID:** 2606.02502 | [PDF](https://arxiv.org/pdf/2606.02502v1)

**作者:** Jun-Tao Tang `[一作]` (Nanjing University), Da-Wei Zhou `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个多模态持续指令调优框架——Centroid Routing Adaptive MoE（CMA‑MoE），通过指令语义聚类、适应性秩实例化和质心引导路由来有效缓解灾难性遗忘并提升参数利用率。

**💡 创新点**

创新点包括①将相似指令格式聚类隔离以消除格式干扰；②利用奇异值分解与正交过滤动态分配专家参数，仅为未覆盖的能力分配资源；③引入质心引导路由与正交惩罚，保证专家复用的稳定性并抑制对已学习通用知识的重写。

**🔧 技术方法**

技术手段包括 CLIP 词向量与余弦相似度用于指令聚类、LoRA‑MoE 架构、奇异值分解（SVD）与正交投影实现适应性秩分配、RBF 核函数路由以及正交约束惩罚。

**📊 数据集**

使用的数据集有 UCIT（ArxivQA、CLEVR-Math、IconQA、ImageNet-R、VizWiz‑Caption、Flickr30k）和 TriGap（PMCVQA、DocVQA、ChartQA、IconQA、InfographicVQA、ArxivQA、Roadside、ChemVQA、FloodNetVQA、CLEVR-Math）。

**📈 对比分析**

与零样本、FT‑LoRA、Replay‑LoRA、MoE‑LoRA、CL‑MoE、SAME、DISCO、HiDe‑LLaVA、ModalPrompt 等方法在 UCIT 上平均精度 74.73%（较最佳基线提升 13.94%），在 TriGap 上 47.79%（较 DISCO 提升 1.25%），同时在参数占用方面也取得了显著优势。

**⚠️ 局限性**

局限性在于仅在有限的任务流上进行了验证，尚未探究更长任务序列下的性能与鲁棒性，以及在极端任务分布变化时的适应能力。

---

## 785. ProtoAda: Prototype-Guided Adaptive Adapter Expansion and Geometric Consolidation for Multimodal Continual Instruction Tuning

**arXiv ID:** 2606.02576 | [PDF](https://arxiv.org/pdf/2606.02576v1)

**作者:** Yu-Cheng Shi `[一作]` (Nanjing University), Da-Wei Zhou `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于任务原型的多模态持续指令调优框架，该框架结合格式感知原型、选择性LoRA扩展与几何合并，能够在连续学习中减少任务间的干扰。

**💡 创新点**

创新点在于同时利用视觉、文本语义与输出格式的任务原型进行任务分组，并通过几何约束的更新分解将共享与任务特定信息分离，从而更好地保留不同输出格式的响应习惯。

**🔧 技术方法**

采用了低秩LoRA、Mixture-of-Experts稀疏架构、格式统计编码、t‑SNE可视化、SVD压缩、谱推广、以及轻量级格式预测MLP等技术。

**📊 数据集**

实验数据集包括TriGap（10个视觉问答、图表、图标等任务）和UCIT（6个任务）等多模态问答与图像理解数据集。

**📈 对比分析**

与Zero‑shot、FineTune、MoE‑LoRA、Replay‑LoRA、HiDe‑LLaVA、CL‑MoE、ModalPrompt、DISCO、SAME等基线在TriGap上平均准确率达到47.23%，在UCIT上达到74.66%，显著优于其他方法。

**⚠️ 局限性**

目前仅在中等规模任务序列与有限的响应格式上验证，尚未检验在更长任务流和更开放式多模态指令场景中的泛化能力。

---

## 786. Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling

**arXiv ID:** 2606.02578 | [PDF](https://arxiv.org/pdf/2606.02578v1)

**作者:** Seojeong Park `[一作]` (KAIST), Hyunjung Shim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Perceptually Perturbed Judgment Dataset (PPJD)，并使用 Group Relative Policy Optimization (GRPO) 对多模态 LLM‑as‑a‑Judge 进行训练，目标是消除感知判断偏差。

**💡 创新点**

提出了通过可验证的感知扰动生成数据并引入全局批量排序奖励的训练框架，使评判器在视觉与推理上具备可解释、鲁棒的感知基准。

**🔧 技术方法**

采用 GRPO 强化学习，结合结构化奖励与 Levenshtein 距离批量排序奖励，确保判定的全局一致性。

**📊 数据集**

以 MMPR 为基础构建 3k 对 PPJD（四元组 x, r_c, r_p, r_p+r），并在 MLLM‑as‑a‑Judge benchmark 上进行评估。

**📈 对比分析**

与 Flex‑Judge、Qwen3‑VL‑4B、GPT‑4o 等基线对比，单分数评分提升 12%，批量排序准确率提升 11%，在多任务评估中整体超越现有开源模型并接近商业评估器。

**⚠️ 局限性**

仍然依赖人工生成的可验证扰动，可能在高度主观或 OOD 场景下表现欠佳，且模型对视觉错误的检测仍不完美。

---

## 787. HumanNOVA: Photorealistic, Universal and Rapid 3D Human Avatar Modeling from a Single Image

**arXiv ID:** 2606.02573 | [PDF](https://arxiv.org/pdf/2606.02573v1)

**作者:** Hezhen Hu `[一作]` (University of Texas at Austin), Georgios Pavlakos `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种从单张RGB图像快速生成高质量3D人类头像的模型（HumanNOVA），实现了在不到一秒的时间内生成全景化、光照一致的3D人体网格。

**💡 创新点**

创新点在于：1) 通过将合成资产与多摄像头真实捕捉数据融合，构建了包含10万人体资产的规模化训练集；2) 采用SMPL简化网格先验与DINOv2/ PTv3多模态编码相结合的跨模态注意力机制，将输入映射到可训练的三平面（triplane）表示；3) 在保持快速推理的同时，通过大规模训练显著提升了光照一致性与细节保留。

**🔧 技术方法**

使用技术包括：大型单视图重建模型（LRM）框架、DINOv2视觉编码器、PTv3网格编码器、PointInfinity变压器跨模态注意力、三平面表示、基于Raymarching的渲染以及RGB、mask、LPIPS三种损失。

**📊 数据集**

使用数据集：自建的100k资产合成数据（包括rigged资产动画+多摄像头真实拟合）以及公开的THuman2、CustomHuman、2K2K三大基准。

**📈 对比分析**

与Real3D、SF3D、Trellis、Hunyuan2、PaMIR、SiFU、SiTH等方法比较，在CustomHuman、THuman2、2K2K三组基准上均实现了LPIPS提升40%以上、PSNR提升约1-2dB、F-Score提升90%以上；并在侧视输入上明显优于SiTH。

**⚠️ 局限性**

局限性包括：对背面纹理推断不够准确，遮挡场景下细节缺失，对极端服装如连衣裙、连体服等纹理推断存在失败；对交互场景的处理尚不充分。

---

## 788. From Layers to Submodules: Rethinking Granularity in Replacement-Based LLM Compression

**arXiv ID:** 2606.02559 | [PDF](https://arxiv.org/pdf/2606.02559v1)

**作者:** Elia Cunegatti `[一作]` (University of Trento), Giovanni Iacca `[通讯]` (University of Trento)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种后训练压缩方法 SubFit，能够在 Transformer 的 Attention 与 FFN 子模块层面进行非连续选择并用低秩残差 bypass 进行替换，从而压缩 LLM 的参数并降低推理成本。

**💡 创新点**

创新点在于：①打破传统全层连续选择的限制，支持子模块级别的非连续选择；②对 Attention 与 FFN 采用不同的替换策略（Attention 采用层级参数，FFN 采用共享低维基底）；③使用闭式拟合公式，避免额外训练，提升压缩效率。

**🔧 技术方法**

技术手段包括：基于校准数据的低秩残差拟合；Attention 与 FFN 的分层评分与替换；使用共享输入基底来压缩 FFN 代替器；并在推理阶段仅插入轻量残差 bypass。

**📊 数据集**

实验使用的校准数据是 SlimPajama（基模型）和 SlimOrca（指令微调模型），每组 8k 条样本；评测指标覆盖基准模型的 PPL（Lambda、C4、WikiText-2）与指令微调模型的下游任务准确率。

**📈 对比分析**

与 Streamline（FFN/Layer）和 ReplaceMe（LS/Cosine）等四个基线相比，SubFit 在 12.5–37.5% 的稀疏率下实现了最优的 PPL‑accuracy 交易：在 25% 稀疏率时，平均准确率保持 84.6%，PPL 仅升至 2.42 倍；相比最强基线提升约 3 倍 PPL 与 3% 准确率；此外提供 1.1–1.4 倍的推理速度提升和 KV‑cache 25–35% 的压缩。

**⚠️ 局限性**

限制包括：在 25% 稀疏率下，需额外加入约 10% 的参数与 15% 的 MACs；压缩过程耗时 2000–2600 秒；相较于 ReplaceMe，压缩后模型需额外存储轻量替换器，无法完全消除参数量；并且在极端稀疏率下仍可能出现较大性能波动。

---

## 789. Modeling Depth Ambiguity: A Mixture-Density Representation for Flying-Point-Free Depth Estimation

**arXiv ID:** 2606.02552 | [PDF](https://arxiv.org/pdf/2606.02552v1)

**作者:** Siyuan Bian `[一作]` (University of Michigan), Jun Gao `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出混合密度深度表示，解决边界飞点问题

**💡 创新点**

在每像素处预测多重深度假设并学习其混合权重，取代单一深度预测

**🔧 技术方法**

混合拉普拉斯/高斯分布、混合负对数似然损失、轻量化多头输出、透明层与天空分量扩展

**📊 数据集**

合成数据集（Synthetic dataset list）与真实数据集（NYU-RGBD、Sintel、KITTI、Bonn、LayeredDepth、其他），并在多视图视频评估

**📈 对比分析**

与DA3、VGGT等基线对比，显著降低边界Chamfer/Accuracy，飞点几乎消除；在视频深度任务保持甚至提升精度，推理速度仅略低于原基线，远快于扩散方法

**⚠️ 局限性**

对高噪声、极端遮挡仍有限制，混合权重选择仍可能误判，透明层仅限两层且训练依赖合成标签

---

## 790. SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment

**arXiv ID:** 2606.02530 | [PDF](https://arxiv.org/pdf/2606.02530v1)

**作者:** Hao Li `[一作]` (Beihang University), Lei Sha `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的安全对齐框架 SafeSteer，通过在基础模型上激活驱动构造安全教师，并在安全标记子集上执行局部反向 KL 对齐，减少对整体模型的干扰。

**💡 创新点**

创新点在于：① 将安全对齐视为稀疏特征的局部更新而非全局权衡；② 利用激活驱动的安全教师生成高质量拒绝示例；③ 采用投票聚合的安全词选择方法，仅对稀疏安全词子集施加 KL 损失，显著降低对齐税。

**🔧 技术方法**

技术包括：激活驱动（Activation Steering）、安全教师构造、投票聚合的安全词抽取、局部反向 KL 反向对齐（On‑Policy Distillation）和对比对数概率分析。

**📊 数据集**

数据集：使用 100 条 PKU‑SafeRLHF 生成的有害指令样本；无额外通用数据；在评估时使用 AdvBench、PKU‑SafeRLHF、HarmBench、JailbreakBench、SORRY‑Bench、HarmfulQA、ALERT 等安全基准，以及 MMLU、AlpacaEval、GSM8K、MATH500、HumanEval 等通用能力基准。

**📈 对比分析**

与 MoCAN、W‑DOOR、BFPO、NSPO、DPO‑Mix 等基线相比，SafeSteer 在七个安全基准上取得最小攻击成功率（ASR），同时在五个通用基准上几乎无性能下降；相较于传统方法，提升安全性显著而代价仅 100 条样本，成本大幅降低。

**⚠️ 局限性**

局限性：① 需要基础模型已具备拒绝能力，无法直接用于无预训练检查点；② 仅在 10B 规模模型上验证，规模更大时需进一步评估；③ 仅针对文本自回归 LLM，尚未验证在 VLM 或 diffusion‑based 语言模型上的可迁移性。

---

## 791. SkillHarm: Lifecycle-Aware Skill-Based Attacks via Automated Construction

**arXiv ID:** 2606.02540 | [PDF](https://arxiv.org/pdf/2606.02540v1)

**作者:** Yuting Ning `[一作]` (Ohio State University), Huan Sun `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对Agent技能攻击的完整生命周期评测基准，并用自动化、可扩展的自然语言驱动编码代理构建了879个跨12种风险类型、两种攻击场景（单会话与跨会话）的攻击样本。

**💡 创新点**

创新点在于：①首次将攻击生命周期与风险类型系统化为三大类并覆盖；②使用自然语言 harness 指导的多阶段编码代理实现跨技能、跨风险、跨攻击场景的自动化攻击生成；③引入条件攻击成功率（cASR）和拒绝率（ARR）衡量隐藏风险与安全性。

**🔧 技术方法**

技术手段包括：自然语言驱动的编码代理（Claude Code、Codex、Gemini CLI、OpenCode），多阶段攻击构造管线（目标选择、攻击设计、质量过滤），确定性攻击成功评估器，LLM-as-a-Judge 用于计算cASR与ARR，技能扫描器与防御系统提示等安全评估工具。

**📊 数据集**

使用的主要数据集为 SkillsBench（含71个技能包、71+任务对），以及公开的各大模型（Claude Sonnet 4.6/Opus 4.7，Codex GPT‑5.4/5.5，Gemini 3 Flash，OpenCode Qwen‑3.6‑27B）。

**📈 对比分析**

通过对六种模型-抓手配置的对比实验，单会话攻击ASR最高达86.3%，跨会话攻击最高69.3%；条件ASR大幅提升，说明隐藏风险严重；拒绝率仅Claude系列显著，且跨会话攻击显著下降；现有技能扫描器与防御提示对降低ASR效果有限。

**⚠️ 局限性**

局限性包括：攻击样本仅覆盖71个公开技能，未覆盖更大规模或行业定制技能；评估集中在现有大模型，未探究更小模型或自研系统的安全性；防御方法多为已有扫描器或提示，缺乏针对技能执行层的实时防护机制。

---

## 792. RoboDream: Compositional World Models for Scalable Robot Data Synthesis

**arXiv ID:** 2606.02577 | [PDF](https://arxiv.org/pdf/2606.02577v1)

**作者:** Junjie Ye `[一作]` (USC), Yue Wang `[通讯]` (USC)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可在不同环境、对象和视角下通过解耦机器人运动与视觉上下文的生成式世界模型，用以合成真实感的机器人演示并实现检索与重生与无道具远程操控。

**💡 创新点**

在视频扩散模型中同时输入机器人仅视频、场景先验和对象先验，实现运动与视觉的可解耦合成，从而支持零样本生成、检索重生和无道具操作。

**🔧 技术方法**

基于多模态视频扩散 Transformer，结合 VAE 编码、Self‑Attention 注入对象先验、Cross‑Attention 注入指令与全局轨迹，实现对三种先验的条件生成。

**📊 数据集**

以 DROID 机器人数据集（约 40k 条）为主训练数据，结合 OmniPaint、Grounded‑SAM、GPT‑5‑nano 等工具自动提取场景与对象先验。

**📈 对比分析**

与仅使用真实数据、未重生的 DROID 轨迹以及用生成观测替换真实观测等基线对比，实验显示 Gen‑Mix 生成数据的策略成功率平均提升至 62.5%（相较于 36.3% 的 Real‑50），prop‑free 采集仅以 55% 近似真实性能并加速约 2.2 倍。

**⚠️ 局限性**

模型依赖于精确的机器人仅视频渲染，受限于扩散基底的时间长度和分辨率，在极端视觉或动力学域外的场景中生成质量下降。

---

## 793. SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation

**arXiv ID:** 2606.02548 | [PDF](https://arxiv.org/pdf/2606.02548v1)

**作者:** Priyaranjan Pattnayak `[一作]` `[通讯]` (Oracle America Inc), Priyaranjan Pattnayak (Oracle America Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Script‑Normalized WER (SN‑WER) 作为仅在评估阶段对 ASR 结果进行文字转写并重新计算 WER 的方法，以消除不同脚本导致的误差膨胀。

**💡 创新点**

创新点在于：①只改评估流程，无需重新训练或解码；②使用语言特定的可确定性转写映射，将参考与假设统一到同一规范脚本；③在多脚本环境下系统验证其对模型差距的缩小与对真实词错误的保持。

**🔧 技术方法**

核心技术包括：Unicode 规范化、基于块检测的罗马化识别、可确定的转写映射（IAST/ITRANS/ICU）、Levenshtein 编辑距离计算 WER，以及对转写不确定性与碰撞率的统计分析。

**📊 数据集**

使用的数据集有：FLEURS（5 种印度语种的精细标注集）、Common Voice（噪声环境的 4 种印度语种）以及 FLEURS 上的阿拉伯语和乌尔都语，模型包括 Whisper‑large‑v3、MMS 和 Whisper‑small。

**📈 对比分析**

对比实验显示：在 FLEURS 上，SN‑WER 可将模型间 WER 差距缩小最多 12%，在 Common Voice 上降幅小或不显著；对 10–50% 罗马化混合的人工压力测试，SN‑WER 对 WER 的增幅仅 67%；在词义扰动控制下，SN‑WER 与 WER 的增幅比约为 1.09，表明保持了对真实错误的敏感性；不同转写器的误差不超过 0.002，碰撞率低于 0.1%。

**⚠️ 局限性**

局限性包括：①仅在脚本不一致导致的误差上起作用，对脚本选择本身的质量评价无效；②需要可靠的语言特定转写库，对某些罕见或新兴脚本支持不足；③在多语言混写或代码切换场景下需要进一步扩展；④对高噪声数据的纠正效果有限，可能掩盖真正的识别缺陷。

---

## 794. A Biconvex Formulation for Stable Transport of Mixture Models with a Unique Solution

**arXiv ID:** 2606.02515 | [PDF](https://arxiv.org/pdf/2606.02515v1)

**作者:** Yeganeh Marghi `[一作]` (Allen Institute), Uygar Sümbül `[通讯]` (Allen Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 Optimal Mixture Transport (OMT) 方法，将 OT 从样本级别提升到子群（混合模型）级别，构造严格双凸优化并获得唯一全局最优解。

**💡 创新点**

创新点在于：
- 通过混合模型（指数族）重构 OT，使计算复杂度仅随混合成分数而非样本量；
- 对 OMT 进行理论稳定性分析，证明在弱正则条件下映射 Lipschitz 连续并对分布扰动具有倾斜稳定性；
- 设计了可在单步内收敛的全局优化算法。

**🔧 技术方法**

技术手段包括：
- 混合模型的熵正则化 OT，构造双凸目标；
- 采用指数族（尤其是高斯、双指数）求解子组件间的闭式 Wasserstein 距离和映射；
- Global Optimization Algorithm (GOP) 解决双凸问题；
- 理论证明（Lipschitz、倾斜稳定性）。

**📊 数据集**

使用的数据集涵盖：
- 合成 2D/3D 任务；
- 细胞组学：scRNA‑seq（sci‑Plex、10x 视觉皮层、老化脑）和 MERFISH 空间转录组；
- 图像数据：ImageNet、MNIST、CIFAR‑10。

**📈 对比分析**

与方法对比：
- 传统 OT、Sinkhorn、PROGOT、LOT、GMM‑OT、ExNOT、ENOT 等；
- 评估指标包括 Wasserstein‑2 误差、均方误差、cosine 相似度、可视化一致性；
- 结果显示 OMT 在大规模、高维、噪声环境下比基准方法更稳健、计算更快、映射更准确。

**⚠️ 局限性**

局限性：
- 对高分辨率图像生成仍不够成熟；
- 目前仅适用于平衡 OT，未处理不平衡情形；
- 需进一步扩展到连续时间动态 OT 的建模。

---

## 795. VISReg: Variance-Invariance-Sketching Regularization for JEPA training

**arXiv ID:** 2606.02572 | [PDF](https://arxiv.org/pdf/2606.02572v1)

**作者:** Haiyu Wu `[一作]` (Altos Labs), Morgan Levine `[通讯]` (Altos Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种自监督学习正则化方法 VISReg，在不依赖任何训练稳定性启发式的前提下，结合方差、相似性与切片 Wasserstein 正则化，显著提升了低质量数据与 OOD 场景下的特征表示质量。

**💡 创新点**

创新点在于：① 用 Sliced Wasserstein Distance（SWD）代替传统协方差正则化，既保留了方差约束的可解释性，又通过全分布形状匹配实现更严格的分布正则化；② 通过“方差‑形状”分离实现梯度鲁棒性，解决了 SIGReg 在嵌入崩塌时梯度消失的问题；③ 复杂度保持线性（O(NDK)），便于大规模训练。

**🔧 技术方法**

技术方法包括：方差约束（Variance Loss）、相似性损失（Invariance Loss）、使用切片 Wasserstein 正则化（Shape Loss）并通过 1D 随机投影实现分布匹配；使用 stop‑gradient 对形状正则化与尺度正则化分离；结合 Cramér‑Wold 定理与 SWD 的闭式一维距离计算。

**📊 数据集**

在 ImageNet‑1K、ImageNet‑22K 预训练后，分别在 ImageNet‑LT、Galaxy10、DTD、ChestXRay、RetinaMNIST、AID、OrganAMNIST、ADE20K 等 15 个在域与任务多样的下游数据集上评估，覆盖线性探测、迁移学习、域迁移、稠密分割与图像生成引导。

**📈 对比分析**

与 DINO、VICReg、SIGReg 等无启发式方法比较，VISReg 在 OOD 任务中往往优于同类方法，且在 ImageNet‑22K 规模下实现与 DINOv2 相当的 OOD 性能，超越了使用十倍数据量的对手；在大多数任务上保持了与或略低于 DINO 的线性探测性能，同时展现出更强的迁移与稠密预测能力。

**⚠️ 局限性**

局限性包括：在纯线性探测的 In‑Domain 任务中仍略逊于 DINO；对稠密分割的性能与 MoCoV3、iBOT 等最先进方法相比存在一定差距；需要手动调节 K（投影数）与 λ（正则化权重）等超参数；对极大规模数据集的训练仍需进一步验证其可扩展性。

---

## 796. VLMs are Good Teachers for Video Reasoning via Adaptive Test-Time Optimization

**arXiv ID:** 2606.02564 | [PDF](https://arxiv.org/pdf/2606.02564v1)

**作者:** Junhao Cheng `[一作]` (City University of Hong Kong), Jing Liao `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VLM-as-Teacher范式，通过视觉语言模型（VLM）在推理阶段为视频生成模型（VGM）提供可微分奖励，从而指导VGM在测试时在线优化生成符合规则的视频轨迹。

**💡 创新点**

创新点在于将VLM的角色从传统的文本解题转变为“教师”，自动从任务描述中合成过程约束和目标奖励查询，并利用VLM的可微分反馈在推理阶段微调VGM的LoRA模块，使视频生成具备逻辑一致性。

**🔧 技术方法**

核心技术包括：任务特定奖励查询合成、VLM可微分VQA损失、基于LoRA的轻量级在线优化、步骤蒸馏的四步生成器、轻量化解码器与帧采样、以及基于损失阈值的早停策略。

**📊 数据集**

使用了两大基准：符号视频推理基准VBVR-Bench（涵盖抽象、知识、感知、空间、变换）和通用推理基准RULER-Bench（涵盖人性、科学、假设、语义、视觉、游戏）。

**📈 对比分析**

与SOTA VGM、VLM-as-Solver（PE、VideoTPO）以及Best-of-N采样等方法比较，VLM-as-Teacher在VBVR-Bench上平均提升0.115分（+16.7%）并显著优于对手；在RULER-Bench上提升21.8分，且在所有30类任务均表现提升。

**⚠️ 局限性**

主要限制包括：奖励查询合成的准确性受VLM推理质量影响，VLM对细粒度视觉错误的感知不足导致优化误导，以及对强视觉理解的依赖，使得在某些细节错误难以被纠正。

---

## 797. IntraShuffler: A Privacy Preserving Framework for Heterogeneous DP Federated Learning

**arXiv ID:** 2606.02563 | [PDF](https://arxiv.org/pdf/2606.02563v1)

**作者:** Farhin Farhad Riya `[一作]` (University of Tennessee), Jinyuan Stella Sun `[通讯]` (University of Tennessee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对异构差分隐私 Federated Learning（HDP-FL）中梯度结构泄露的隐私攻击，提出并实现了一种名为 IntraShuffler 的中间件防御框架。该框架通过隐私兼容桶划分和参数级随机置换，在保持 ε‑aware 聚合的前提下，破坏梯度的持久结构，显著降低客户端身份和数据分布的可推断性。

**💡 创新点**

创新点：①首次将隐私预算兼容的桶划分与参数级打乱相结合，在保持梯度聚合权重的同时实现匿名化；②通过最小桶大小自适应合并，兼顾匿名性与聚合精度；③在标准 FL 优化器（FedAvg/FedProx/FedOpt）上实现，无需修改客户端训练流程。

**🔧 技术方法**

主要技术：梯度去噪（ε‑aware denoiser）、代理模型（surrogate modeling）用于属性推断、参数级随机置换、隐私兼容桶（bucket）划分与合并、DP-SGD（Gaussian noise）本地差分隐私、Shuffle‑Model 中间件实现。

**📊 数据集**

实验数据集：伦敦住宅电力、Pecan Street 电力、ComStock（时间序列），以及 CIFAR‑10 图像分类。

**📈 对比分析**

对比方法：Plain FL‑DP、Shuffle‑DP。结果显示：梯度可恢复度下降 60%+（Cosine similarity 从 0.45 降至 0.15），代理模型推断准确率从 0.78 降至 0.33；模型误差与 Shuffle‑DP 相当，RMSE/accuracy 变化在噪声范围内，几乎不影响收敛速度。

**⚠️ 局限性**

limitations：①若参与客户端数量小或隐私预算高度分散，桶规模过小导致参数级打乱失效；②仅针对 Heterogeneous DP，未考虑系统/数据分布等多重异构；③在极端异构预算场景下可能需要更复杂的自适应桶策略或结合其他技术（如安全聚合、TEE）。

---

## 798. HERO'S JOURNEY: Testing Complex Rule Induction with Text Games

**arXiv ID:** 2606.02556 | [PDF](https://arxiv.org/pdf/2606.02556v1)

**作者:** Anshun Asher Zheng `[一作]` (University of Texas at Austin), Junyi Jessy Li `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了HerosJourney基准，用于评估大型语言模型在目标导向的多步情节任务中从演示中诱导规则并执行的能力。

**💡 创新点**

创新点在于将规则诱导与多步执行、可配置的规则交互结构、可辨识性约束以及可替换的表面词汇统一到一个可扩展的评估框架中。

**🔧 技术方法**

主要技术包括基于文本游戏的模拟环境、LLM提示与多步交互、规则可视化与归纳、以及效率校准成功率（ECSR）与规则表述评分（RV）两种评估指标。

**📊 数据集**

数据集为自行构造的HerosJourney任务集，涵盖属性与程序两大族群，四种结构形式（Add、Comp、Cond、Over），共八个任务类型。

**📈 对比分析**

通过与多款前沿LLM（如GPT‑5.4‑mini、Qwen3.5‑27B、Gemini‑3.1‑Flash等）以及7名人类参与者进行对比，模型在属性任务上可达到与人类相近的ECSR，但整体仍落后人类30%（RV）且在程序任务表现更差；QA与执行模式存在显著的格式瓶颈。

**⚠️ 局限性**

局限性包括：任务完全合成，缺乏对噪声、开放式动作空间的验证；规则交互与可辨识性需人工设计；ECSR与RV未能完全排除仅靠试错的成功；评估方法与人类基准规模有限。

---

## 799. AFUN: Towards an Affordance Foundation Model for Functionality Understanding

**arXiv ID:** 2606.02551 | [PDF](https://arxiv.org/pdf/2606.02551v1)

**作者:** Zhaoning Wang `[一作]` (University of Michigan), Jun Gao `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种能够从单帧RGB‑D图像和语言任务描述中同时预测任务相关功能掩码与三维后接运动轨迹的开源基础模型，并通过统一的数据处理流水线将来自机器人、人类、仿真及真实扫描的大量多源数据转化为标准化的功能感知数据集；

**💡 创新点**

提出了将大型视觉语言模型与分割模型通过MetaQuery桥接，实现功能分割与运动预测的统一框架；使用Bézier曲线参数化的三维运动表示，使得模型能够在保持可解释性的同时生成可直接执行的机器人轨迹；构建了规模最大的公开功能感知数据集，实现了对多环境、多对象、多任务的开世界泛化；

**🔧 技术方法**

采用Qwen3‑VL做语义与运动条件编码，SAM3做功能掩码生成，Sonata做3D特征编码，Transformer解码器预测Bézier曲线；训练分为三阶段：MetaQuery–SAM3对齐、功能分割全链路训练、联合运动预测；

**📊 数据集**

采集并处理了10个公开数据源（机器人演示、人类自我摄像、仿真交互、真实扫描）共321,190条视频，拆分为1,242,740个动作区间，最终筛选出59,867条高质量训练样本；评估使用HOVA‑500K、RAGNet、InstructPart、ReasonAFF、SceneFun3D、RoboMIND2等公开基准；

**📈 对比分析**

在8个分割基准上与零样本Qwen3‑VL+SAM3、AffordanceNet、Affordance‑R1等对照，平均gIoU/cIoU提升23.9/26.3点；在接触点预测上hit‑rate提升12.7%–61.3%；在三维运动评估（ADE/FDE/CIM）上均超过所有基线，且在三套测试集上取得最优结果；在真实机器人任务（拿起螺丝刀、取下锅盖、拉开抽屉、打开微波炉）中，平均成功率达到90%；

**⚠️ 局限性**

仍依赖RGB‑D输入，深度估计误差和遮挡会影响掩码与轨迹质量；模型在极端视觉条件下（如光照骤变、复杂纹理）可能出现失败；目前仅在少数机器人任务上验证，尚未覆盖更广泛的操作场景；

---

## 800. SimSD: Simple Speculative Decoding in Diffusion Language Models

**arXiv ID:** 2606.02544 | [PDF](https://arxiv.org/pdf/2606.02544v1)

**作者:** Junxia Cui `[一作]` (University of California San Diego), Jingbo Shang `[通讯]` (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SimSD的plug‑and‑play掩码策略，使diffusion大型语言模型（dLLM）能够像自回归模型一样进行token级的speculative decoding，从而显著加速推理。

**💡 创新点**

创新点在于构造token级的时间因果注意力mask并对RoPE进行对齐，为dLLM提供可验证的token级上下文；该方法训练无关，可与KV缓存、blockwise decoding等并行加速技术无缝结合。

**🔧 技术方法**

采用token‑level temporal causal attention mask、RoPE对齐、pair‑data‑mask布局、speculative decoding接受规则、KV cache、blockwise decoding、CUDA Graph等技术。

**📊 数据集**

实验使用四个基准数据集：GSM8K、MBPP、TriviaQA、MMLU。

**📈 对比分析**

与Vanilla SDAR、Vanilla+CUDA Graph、S2D2等基线对比，SimSD在B=4、B=8时分别实现7.46×、5.40×的throughput提升，且生成质量保持不变或略有提升（平均+1.7%）。

**⚠️ 局限性**

局限性包括：对RoPE对齐高度依赖，跨模型验证时的分布差异可能导致误差；γ值选择需要权衡吞吐与质量；在更大批量或更复杂任务下的效果尚未充分验证。

---

## 801. LL-Bench: Rethinking Low-Level Vision Evaluation in the Era of Large-Scale Generative Models

**arXiv ID:** 2606.02535 | [PDF](https://arxiv.org/pdf/2606.02535v1)

**作者:** Lu Liu `[一作]` (Shanghai Jiao Tong University), Xiaoyun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了LL-Bench低级视觉任务基准，涵盖 16 个真实世界降质任务（共 2,469 张图像），并由 10 个大规模生成模型与 21 个传统恢复模型生成 28,919 张恢复结果；收集了 152,020 对专家级偏好标注与每张图像的伪影检测；基于此数据提出 LL-Score——一种多模态大语言模型评估器，能够同时给出恢复质量分数和伪影概率，并可用作奖励模型。

**💡 创新点**

创新点在于：①提供首个大规模、全任务覆盖、专家偏好驱动的低级视觉评估基准；②系统诊断大型生成模型在各任务上的优势与失败模式；③提出 LL-Score，利用多模态 LLM 与 rank‑aware 双任务训练，实现统一的质量评估与伪影检测，显著优于传统 FR/NR 评估与现有奖励模型；④证明 LL-Score 可有效作为 RL 奖励加速恢复模型优化。

**🔧 技术方法**

技术包括：大规模生成模型（如 GPT‑Image‑2、NanoBananaPro、Hunyuan‑Image‑3）、传统恢复模型（专业与全能式）、专家标注系统、Bradley‑Terry 排序模型、LLM（Qwen3‑VL‑8B‑Instruct）+ LoRA 微调、视觉编码器 + 视觉‑语言投影、rank‑aware margin‑augmented Bradley‑Terry 损失、二元焦点损失、LLM 作为奖励函数（GRPO）等。

**📊 数据集**

使用的主要数据集：LL-Bench（含 16 任务的 2,469 张降质图、28,919 张恢复图、152,020 对偏好标注及 28,334 B‑T 分数、伪影标签），对比的传统 IQA 与奖励模型来源于公开基准（如 DIV2K、RealSR、RealBlur 等）。

**📈 对比分析**

评估方法：对 16 任务分别计算 SRCC 与 pairwise accuracy，LL-Score 在所有 17 种 IQA/奖励模型中均取得最高 SRCC 与准确率；在伪影检测任务上，LL-Score 亦获得最高准确率、F1 与 AUROC。将 LL-Score 作为奖励对 Qwen‑Image‑Edit 进行 GRPO 微调后，生成的超分图在 PSNR、SSIM 及视觉可感知质量上显著提升，且伪影率明显下降。

**⚠️ 局限性**

局限性包括：①LL-Score 的伪影检测仍主要针对已标注的三类伪影（颜色偏移、对象伪影、结构编辑），对更细粒度或新型伪影识别效果未知；②虽然 LL-Bench 覆盖 16 任务，但仍未包含所有现实降质场景（如极端噪声、压缩极限）；③LL-Score 训练依赖大量专家标注，扩展到更大规模或不同语言环境时需要额外标注；④在极度噪声或极端光照等极端条件下，LL-Score 与模型表现仍可能退化。

---

## 802. Improving Combined Detection and Classification of TEM Defects via Mask-Conditioned Latent Diffusion Augmentation

**arXiv ID:** 2606.02532 | [PDF](https://arxiv.org/pdf/2606.02532v1)

**作者:** Ni Li `[一作]` (University of Wisconsin-Madison), Dane Morgan `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用掩码条件潜在扩散模型合成TEM图像并提升缺陷检测与分类

**💡 创新点**

首次将掩码条件潜在扩散模型与实验掩码分布采样相结合，实现可控、多类别缺陷图像生成

**🔧 技术方法**

采用Mask‑R‑CNN、VQ‑GAN编码器-解码器与U‑Net的潜在扩散网络及统计掩码采样技术

**📊 数据集**

基于Jacobs等公开的FeCrAl合金TEM实验数据集（共1024×1024像素图像与多类别掩码）

**📈 对比分析**

将生成的数据与10k、50k、100k实验样本合并训练Mask‑R‑CNN，比较F1检测、F1分类及其谐波平均（F1HM），生成数据带来0.01–0.02的F1HM提升，显著提升在50/100图像规模下

**⚠️ 局限性**

生成模型需要足够真实样本才能产出高质量图像，10图像规模下提升有限；生成方法难以解决类别不平衡且可能继承人类标注偏差

---

## 803. Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events

**arXiv ID:** 2606.02522 | [PDF](https://arxiv.org/pdf/2606.02522v1)

**作者:** Xiaolin Liu `[一作]` (Shandong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布 Moment-Video 基准，评估视频多模态大型语言模型对短暂关键视觉事件的时间保真度。

**💡 创新点**

引入四类任务（事件出现、计数、描述、推理）以系统化检验模型对瞬时事件的捕捉、计数、描述和推理能力，并构建 1,000 条人工验证的跨域问答对。

**🔧 技术方法**

采用弱时间标注、人工专家构造问题与答案、shuffle‑robust 多选评估、LLM 评判的开放式答案评估，并对不同帧率与视频时长进行 Ablation 分析。

**📊 数据集**

Moment-Video 数据集，包含 1,000 条视频问答对，覆盖 AIGC、GUI、自然、工业、游戏、人类、动物等 7 个领域和 25 个细粒度子类别。

**📈 对比分析**

评估 33 个视频 MLLM（17 款开源 + 16 款专有），通过平均准确率比较；最高分为 39.6%（Seed‑2.0‑Pro），大多数开源模型低于 25%，计数与推理任务表现尤为逊色。

**⚠️ 局限性**

现有模型在瞬时事件计数与推理上表现差，帧率提升收益有限，长视频中事件易被忽视，显示需要更细粒度、事件感知的时序机制。

---

## 804. Drifting Preference Optimization for One-Step Generative Models

**arXiv ID:** 2606.02521 | [PDF](https://arxiv.org/pdf/2606.02521v1)

**作者:** Zhou Jiang `[一作]` (Westlake University), Zhen Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于漂移偏好优化的在线微调方法，利用奖励排序的样本在特征空间生成漂移场，从而在不需要对奖励进行反向传播的情况下对单步图像生成器进行偏好微调。

**💡 创新点**

通过将奖励排序的正负样本转换为非参数的双极偏好场，并加入基于冻结基准生成器的参考漂移，实现了无梯度、无策略似然、无去噪轨迹的偏好微调；同时支持黑盒或不可微奖励。

**🔧 技术方法**

使用特征空间漂移、非参数双极奖励模型、mean‑shift估计的参考漂移、LoRA微调以及大规模特征提取（latent‑MAE、DINOv2）等技术。

**📊 数据集**

主要使用SD‑Turbo和SDXL‑Turbo一阶生成器；评估数据集包括Pick‑a‑Pic v2、Parti‑Prompts、GenEval以及外部VLM评判集Qwen3‑VL。

**📈 对比分析**

与基于奖励梯度的DPO、CLIP‑DPO等方法以及无梯度的Drift、ReNO等基线对比，DrPO在PickScore、AES、ImageReward、Qwen3‑VL偏好赢率上均优于基线，并在HPSv3训练中实现3.51×的时间加速。

**⚠️ 局限性**

结果依赖于奖励的质量与覆盖度；离线微调受限于偏好样本稀缺；需要生成候选样本和特征提取，计算成本仍非零；对极端细粒度属性或稀缺特征的对齐能力有限。

---

## 805. ToolFG: Towards Well-Grounded Fine-Grained Image Classification

**arXiv ID:** 2606.02518 | [PDF](https://arxiv.org/pdf/2606.02518v1)

**作者:** Yu Xue `[一作]` (Lancaster University), Jun Liu `[通讯]` (Lancaster University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ToolFG框架，使多模态大型语言模型(MLLM)能够通过调用图像处理工具来收集可验证的视觉证据，从而实现精细化图像分类(FGIC)；

**💡 创新点**

创新点在于（1）首次将工具调用嵌入到MLLM的推理链，实现可验证的理由链；（2）设计了基于MCTS的工具使用知识蒸馏机制，利用高级专有MLLM提取工具使用与FGIC相关的轨迹；（3）提出模型-工具协同进化机制，迭代优化模型策略与工具集，使二者互相适配；

**🔧 技术方法**

使用的技术包括：功能式API化工具接口、蒙特卡洛树搜索（MCTS）引导的轨迹探索、对比学习的轨迹级损失、基于RL的地面化奖励（groundedness reward）以及LLM驱动的工具演化；

**📊 数据集**

在五个主流FGIC基准上评估：CUB‑200、Oxford Flowers‑102、Stanford Cars‑196、Oxford Pets‑37、FGVC Aircraft‑100；采用基于类别划分的基类/新类零样本以及4-shot少样本设置；

**📈 对比分析**

与多种基线（Qwen2.5‑VL‑7B、MaPLe、PromptSRC、ViRFT、DiVE‑k、Fine‑R1）对比，ToolFG在三种评估场景下均实现最高平均准确率，尤其在基类/新类的harmonic mean上表现优异；

**⚠️ 局限性**

局限性包括：（1）工具集初始化仍依赖LLM生成，可能缺乏最优；（2）工具演化依赖LLM判断与单元测试，易受LLM错误影响；（3）RL训练的收敛性和样本效率未充分评估；（4）当前仅验证在公开FGIC数据集，缺乏跨领域推广证明；

---

## 806. Not All Points Are Equal: Uncertainty-Aware 4D LiDAR Scene Synthesis

**arXiv ID:** 2606.02510 | [PDF](https://arxiv.org/pdf/2606.02510v1)

**作者:** Xiang Xu `[一作]` (Nanjing University of Aeronautics and Astronautics), Qingshan Liu `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了基于不确定性感知的4D LiDAR场景合成框架U4D，采用“先难后易”的生成策略。

**💡 创新点**

创新点在于通过预训练分割器的Shannon熵量化空间不确定性，并利用两阶段扩散模型先重建高不确定性区域，再完成整体场景，同时引入Mixture of Spatio-Temporal (MoST)模块提升跨帧一致性。

**🔧 技术方法**

技术包括基于熵的区域选取、无条件与有条件扩散模型、MoST门控融合，以及跨帧时空约束。

**📊 数据集**

使用nuScenes和SemanticKITTI两个公开LiDAR数据集进行训练和评估。

**📈 对比分析**

与R2DM等均衡生成基线对比，在Fréchet Range Distance、Fréchet Point Distance、TTCE等指标上提升6%–11%，并在下游语义分割和模型校准上取得更高准确率和更低ECE。

**⚠️ 局限性**

局限性包括对预训练分割器的依赖、对高熵区域采样策略的敏感性，以及在极端稀疏或动态场景中的生成精度仍有提升空间。

---

## 807. When Rating Scales Fall Short: LLM-Assisted Discovery of ADHD Signals in Turkish Teacher Narratives

**arXiv ID:** 2606.02509 | [PDF](https://arxiv.org/pdf/2606.02509v1)

**作者:** Baris Karacan `[一作]` (University of Illinois Chicago), Elvan Iseri `[通讯]` (Gazi University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 ADHD 学生的结构化评分（CTRS‑R:S）和教师叙述文本进行分类，评估两者在 ADHD 识别中的效果，并利用 LLM 提取叙述主题进行定性分析。

**💡 创新点**

证明教师叙述文本提供了结构化评分难以捕捉的互补行为信号，并首次采用 LLM 辅助主题提取来解释两种信息来源的差异。

**🔧 技术方法**

使用逻辑回归分类、TF‑IDF 文本特征、分层五折交叉验证、Llama 3.3 70B 进行主题抽取以及误差统计分析。

**📊 数据集**

199 名学生（166 ADHD，33 对照）的去识别化教师评估表，包含 CTRS‑R:S 评分和开放式叙述。

**📈 对比分析**

在相同的交叉验证框架下比较 PR‑AUC、ROC‑AUC、平衡准确率和 ADHD 召回率；叙述模型在召回率和区分结构化误判样本方面表现更好。

**⚠️ 局限性**

样本量小、类别失衡、仅使用线性模型、叙述文本主观性高、LLM 主题映射依赖专家判断，结果难以推广至其他语言或教育体系。

---

## 808. ClinEnv: An Interactive Multi-Stage Long Horizon EHR Environment for Agents

**arXiv ID:** 2606.02568 | [PDF](https://arxiv.org/pdf/2606.02568v1)

**作者:** Yuxing Lu `[一作]` (Georgia Institute of Technology), May Dongmei Wang `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个基于真实住院病例的交互式基准，评估LLM在多阶段、实时信息查询与决策中的表现。

**💡 创新点**

创新点在于：① 自动化将MIMIC-IV的完整电子病历转化为有序决策阶段；② 引入四个专门的代理（病人、护士、实验室、病史）实现主动信息获取；③ 采用双重评估框架，既评估决策准确性，也量化信息获取质量和成本效率。

**🔧 技术方法**

技术包括：LLM（如Claude‑Sonnet‑4.6、GPT‑5.4）进行决策抽取与时间定位；多智能体交互环境与工具调用；基于词表嵌入与LLM重排序的ICD/ATC匹配；Hungarian算法进行决策匹配；成本评估利用CMS实验室费用表和NADAC数据库。

**📊 数据集**

使用MIMIC‑IV v3.1和MIMIC‑IV‑Note v2.2，共3,509个住院案例，9,297个决策阶段，26,043个真实决策，涉及2,128个ICD码和488个药物。

**📈 对比分析**

在七款LLM（GPT‑5.4、GPT‑5.4‑mini、GPT‑5.4‑nano、Llama‑3.1‑70B、Llama‑3.1‑8B、Gemma‑3‑27B、Gemma‑3‑12B）上评测，最高决策F1仅为0.306；诊断得分最高，管理（药物/手术）得分最低；信息获取覆盖率与实验室浪费呈负相关，表明信息寻求能力与决策质量相关。

**⚠️ 局限性**

局限包括：① 仅衡量与已记录的真实治疗方案的一致性，无法评估最佳医疗决策；② 仅来自单一美国学术中心，缺乏多中心、多语言和不同编码体系的泛化；③ 评估步骤虽使用LLM，但所有结构化真值均来自电子病历，避免了模型偏倚。

---

## 809. Pluralistic Leaderboards

**arXiv ID:** 2606.02547 | [PDF](https://arxiv.org/pdf/2606.02547v1)

**作者:** Nika Haghtalab `[一作]` (University of California, Berkeley), Kunhe Yang `[通讯]` (University of California, Berkeley)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种多元化排行榜机制，利用局部稳定性保证在用户偏好异质的环境下，排行榜对所有大规模子群体公平代表。

**💡 创新点**

创新点在于将社交选择理论中的局部稳定性概念迁移到大型语言模型排行榜，并通过迭代舍入算法在仅获取有限 pairwise 比较的情况下实现该稳定性。

**🔧 技术方法**

采用的技术包括局部稳定性理论、迭代舍入（Iterated Rounding）算法、Bradley–Terry 模型、主动采样与抽样oracle，以及对比基线方法。

**📊 数据集**

使用的数据集包括 LMArena 的 arena‑human‑preference‑140k 真实对比数据和基于 Mallows 模型的合成混合数据。

**📈 对比分析**

通过与 Bradley–Terry 排行榜、理想中心选取和状态基线等方法比较，实验表明我们的算法在所有前 k 前缀下均满足或接近稳定性，且在大部分 k 上优于传统 BT 排行榜。

**⚠️ 局限性**

限制在于假设用户偏好是确定且与提示无关，适用于在线主动采样环境；未考虑离线数据、提示依赖性以及更强的比例代表性等更广泛场景。

---

## 810. Transferable Self-Harm Surveillance from Emergency Department Triage Notes Using an Evidence-Augmented Machine Learning Approach

**arXiv ID:** 2606.02545 | [PDF](https://arxiv.org/pdf/2606.02545v1)

**作者:** Liuliu Chen `[一作]` (University of Melbourne), Vlada Rozova `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一个三阶段基于LLM的自伤监测系统，利用ED分诊文本进行自伤识别和方法提取。

**💡 创新点**

创新点在于将零射击LLM做高敏感筛选、结构化证据抽取与传统ML结合，实现无标注迁移并输出可解释的自伤方法。

**🔧 技术方法**

使用零射击大语言模型进行筛选和证据提取，随后以TF-IDF、MiniLM向量与提取的证据为特征训练逻辑回归分类器。

**📊 数据集**

使用来自皇家墨尔本医院、拉特罗地区医院和阳光医院的2012-2022年ED分诊笔记，标注自伤与否以及方法。

**📈 对比分析**

与基线模型（逻辑回归、ClinicalBERT、GBM）比较，跨医院及前瞻性测试中AUPRC均在0.81-0.89之间，优于传统方法。

**⚠️ 局限性**

局限包括仅在澳大利亚三个医院验证、仅用单一LLM、方法提取未完全验证、LLM推理资源成本较高。

---

## 811. Tracking the Behavioral Trajectories of Adapting Agents

**arXiv ID:** 2606.02536 | [PDF](https://arxiv.org/pdf/2606.02536v1)

**作者:** Jonah Leshin `[一作]` (Project VAIL), Ian Timmis `[通讯]` (Project VAIL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对AI代理技能文件的增删差异进行文本嵌入，构建特征向量来量化代理行为特征，并提供基于该向量的评估协议

**💡 创新点**

提出将行为特征定义为嵌入空间中的线性方向，利用差分嵌入学习 trait 向量，并将该方法嵌入代理间可验证的评估流程

**🔧 技术方法**

使用 Qwen3‑Embedding‑8B 进行指令化文本嵌入，Ridge 回归学习 trait 向量，构建无服务器端计算的评估协议

**📊 数据集**

基于 Awesome Copilot 仓库公开的 63 篇技能文件及其人工构造的“更具数据需求”或“更安全”版本，形成 68 对差分数据

**📈 对比分析**

在留一交叉验证中实现 91.2% 的符号分类准确率，Spearman ρ=0.82；相较于 YARA 规则 63.2% 的准确率，落后于 GPT‑5.4 100% 的 LLM 分类，但保持确定性与可审计性

**⚠️ 局限性**

仅验证单一“数据寻求”特征，样本量有限，缺乏跨特征聚合与鲁棒性评估，且仍依赖可信嵌入模型与中介服务器的安全假设

---

## 812. IMAC-AgriVLN: Can Agricultural Vision-and-Language Navigation Agents be Aware of Instruction Mistakes?

**arXiv ID:** 2606.02519 | [PDF](https://arxiv.org/pdf/2606.02519v1)

**作者:** Xiaobei Zhao `[一作]` (China Agricultural University), Xiang Li `[通讯]` (China Agricultural University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了农用视觉-语言导航中指令错误的评估与纠正框架，构建了A2A-MI基准并实现了IMAC模块；

**💡 创新点**

创新点在于：①通过半自动标注在每条原始指令中插入三类错误，形成大规模误指令基准；②设计IMAC模块让VLM在每个时间步结合前方图像主动识别并修正指令错误；

**🔧 技术方法**

使用了大语言模型（DeepSeek-R1-32B）与视觉-语言模型（Qwen2.5-VL-32B）以及子任务列表（STL）等技术；

**📊 数据集**

基准数据集为A2A-MI（基于A2A），包含1,560条原始导航任务，每条插入三类错误，总计4,680条误指令；

**📈 对比分析**

在A2A-MI上与SIA-VLN、DILLM-VLN、AgriVLN等对比实验显示，IMAC-AgriVLN的成功率从0.10提升至0.14，导航误差从4.81m降至4.79m，指令误差意识率达到0.27；

**⚠️ 局限性**

主要限制是缺乏对纠正质量的客观评估指标，且过度的误指令识别与纠正可能引入噪声导致性能波动。

---

## 813. Question-Aware Evidence Ledgers for Video Relational Reasoning

**arXiv ID:** 2606.02506 | [PDF](https://arxiv.org/pdf/2606.02506v1)

**作者:** Yilin Ou `[一作]` (Beijing University of Posts and Telecommunications), Huadong Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于GPT-5.5的测试时视频问答系统，通过路由式证据账本和保守合并门来校验答案。

**💡 创新点**

创新点包括：① 问题感知的证据账本（计数、空间、终点、社交等）; ② 只在必要时调用相应证据族的路由式推理；③ 采用保守门控策略，只有独立证据支持才允许答案修改。

**🔧 技术方法**

技术核心：GPT-5.5强大推理、外部工具（开词检测、深度估计、ASR、场景图）、证据账本结构化输出、保守门控合并。

**📊 数据集**

使用公开的多类视频问答基准（多种计数、空间、运动、视角、社交与对话等子任务），未使用任何额外标注数据。

**📈 对比分析**

与单一模型或简单投票方案对比，基准测试上整体准确率提升至92.95%（宏观准确率93.79%），相较基线71.89%提升约21个百分点；对照实验显示保守门控能将错误修正率提升到100%。

**⚠️ 局限性**

局限性：当关键证据缺失或问题本身语义模糊时仍会出现错误；外部工具的误差会影响证据可靠性；系统完全依赖测试时推理，未对模型进行训练或微调，可能对模型更新敏感。

---

## 814. GloResNet: A lightweight 3D CNN with global topological features for preterm brain injury prediction

**arXiv ID:** 2606.02498 | [PDF](https://arxiv.org/pdf/2606.02498v1)

**作者:** Boyu Yuan `[一作]` (Saanxi University of Science and Technology), Liang Guo `[通讯]` (Shenzhen University of Advanced Technology General Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了轻量级3D CNN框架GloResNet，利用全局拓扑感知的全局重采样和z-score归一化，对早产儿T2加权MRI进行脑损伤预测。

**💡 创新点**

创新点包括：①全局流形映射（128³下采样+z-score）保留脑结构全局拓扑；②“少即多”策略，采用ResNet‑10轻量化骨干并用MedicalNet进行领域迁移；③结合mixup数据增强与测试时平均（TTA）实现鲁棒的正则化与推理。

**🔧 技术方法**

使用的技术包括：3D CNN（ResNet‑10）、MedicalNet预训练、全局重采样与z-score归一化、mixup混合训练、测试时平均（TTA）、交叉熵加类别权重、AdamW优化器与余弦退火学习率调度。

**📊 数据集**

数据集为dHCP（Developing Human Connectome Project）T2加权MRI，128例（74正常，54脑损伤），采用5折分层交叉验证。

**📈 对比分析**

与Transformer、深度CNN、其他轻量CNN等多种基线模型进行对比，GloResNet在5折交叉验证中平均准确率75.18%、峰值81.82%，AUC 0.861，MCC 0.534，参数仅5.4M，GFLOPs 3.9，推理速度<50 ms。

**⚠️ 局限性**

局限性：仅在单中心dHCP数据上验证，缺乏多中心跨设备泛化评估；全局下采样导致对细小白质病灶（punctate WMI）检测能力受限；缺乏病灶级别的标注，难以进一步提升局部精度。

---

## 815. Thinking in Blender: Staged Executable Inverse Graphics with Vision-Language Models

**arXiv ID:** 2606.02580 | [PDF](https://arxiv.org/pdf/2606.02580v1)

**作者:** Guangzhao He `[一作]` (Cornell University), Hadar Averbuch-Elor `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于预训练视觉‑语言模型的分阶段可执行逆向图形框架，能够仅从单张图像直接重建可编辑的Blender场景；

**💡 创新点**

创新点在于将逆向图形任务拆解为几阶段（初始化、几何、材质、构图、照明）并为每阶段配备生成‑验证循环，使模型能够逐步细化场景并保持可编辑性；

**🔧 技术方法**

核心技术是利用预训练的VLM（如Claude Opus 4.7）生成Blender代码，结合Blender API进行场景操作、渲染与视觉评估，实现端到端代码生成与验证；

**📊 数据集**

实验使用NeRF合成数据集、VoxHammer、Edit3D场景以及一组真实图像；

**📈 对比分析**

与单一端到端VLM方法及配合SAM/SAM‑3D的VIGA基线相比，分阶段框架在PSNR、SSIM、LPIPS、DreamSim、DINO、CLIP六项指标上均优越，尤其在几何和材质重建精度上显著提升；

**⚠️ 局限性**

局限性包括早期阶段误差可能影响后续阶段、需多轮生成‑验证导致运行时间和API成本较高，以及对极端遮挡或多视角情境的鲁棒性有限。

---

