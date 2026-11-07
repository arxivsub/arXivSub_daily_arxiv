# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-06 | 今日论文总数: 688

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Server-Side Sparse Assignment for Federated Mixture-of-Experts

**arXiv ID:** 2611.07093 | [PDF](https://arxiv.org/pdf/2611.07093v1)

**作者:** Xue Zheng `[一作]` (Ohio State University), Aylin Yener `[通讯]` (Ohio State University)

**通讯引用:** 13257 | [OpenAlex ID](https://openalex.org/A5039328157)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在边缘设备资源受限的联邦学习环境中，提出服务器端稀疏专家分配框架S^3-MoE，实现对每个客户端的专家子集分配与更新。

**💡 创新点**

创新点在于：① 通过部分观测的稀疏路由数据，利用滑动窗口和倾向校正的相关性估计；② 采用探索–利用（exploit–explore）策略动态调整专家子集；③ 给出估计误差界与Exact Top-K子集恢复的理论保证。

**🔧 技术方法**

技术包括：CNN+MoE模型、LoRA本地微调、滑动窗口估计、Propensity-corrected（倾向校正）相关性计算、探索–利用分配策略、FedAvg与A^3S-MoE对比实验。

**📊 数据集**

实验数据集为CIFAR-100，构造5个强非IID任务组，每组2个客户端，共10个客户端。

**📈 对比分析**

与Random-5、FLEX-MoE-ACC、Full-20 FedAvg、A^3S-MoE等方法对比，S^3-MoE在仅同步5个专家时，平均准确率最高（53.05%），比Full-20 FedAvg提升1.31个百分点，并将通信成本降低约75%。

**⚠️ 局限性**

局限性在于：实验仅限于单层CNN MoE；探索–利用策略参数设定较为手工；未针对极端非IID或动态分布变化的鲁棒性进行深入评估。

---

## 2. DiaHeads: Jointly Generating Coordinated 3D Dialogue Heads with Non-verbal Cues from Dual Audio

**arXiv ID:** 2611.06970 | [PDF](https://arxiv.org/pdf/2611.06970v1)

**作者:** Jiahe Wang `[一作]` (University of Science and Technology of China), Shangfei Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4115 | [OpenAlex ID](https://openalex.org/A5077046519)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 DiaHeads，能够从两个人的音频流中联合生成两方的 3D 头部动画，并通过四个离散的非语言提示（注视、点头、摇头、微笑）来引导交互性。

**💡 创新点**

创新点包括：① 使用 VAE 的 product‑of‑experts 后验同时融合低层连续参数与高层离散提示；② 用流式先验模型学习共享潜在变量的时间相关分布；③ 通过 SyncNet 约束口型与语音同步，从而提升对话协调；④ 结合四类非语言提示实现互相影响的生成。

**🔧 技术方法**

主要技术手段：基于 HuBERT 的音频编码、Transformer 条件向量、双分支 VAE（连续与离散），流式先验、SyncNet、以及多任务损失（KL、MSE、连续性、BCE、同步）等。

**📊 数据集**

使用新构建的 CANDOR‑FG 数据集（CANDOR 的 FLAME 参数与 3D 注视同步扩展），共 14,204 条双人对话片段，支持低层参数与离散提示的训练。

**📈 对比分析**

与 SadTalker、SyncTalk、DIM 等基线对比，DiaHeads 在 FLAME MSE 与 Gaze MSE 上均保持竞争力，在同步误差（Sync Error）和交互 DTW 方面显著优于基线（DTW 降低 57.9%，Sync Error 降低 37.4%），用户研究中在自然度与互动协调度的 MOS 以及强制选择偏好上均获得显著优势。

**⚠️ 局限性**

局限性：① 在 FLAME MSE 上略高于单人模型，主要因关注前方互动而忽略侧视姿态；② 仅支持离线内容生成，实时交互尚未验证；③ 对非语言提示的标注依赖人工，若标注质量下降可能影响协调效果。

---

## 3. Autonomous AI Agents Systematically Diverge From Human Preferences in Proof Formalization

**arXiv ID:** 2611.06995 | [PDF](https://arxiv.org/pdf/2611.06995v1)

**作者:** Bálint Gyevnár `[一作]` (Carnegie Mellon University), Simon DeDeo `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2153 | [OpenAlex ID](https://openalex.org/A5009637990)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在Lean证明库上进行“模拟协作”，对比人类和LLM代理在下一步选择与编辑上的偏好。

**💡 创新点**

揭示LLM代理倾向于填补显式的sorry缺口、偏好图表边缘与低连通度的定义，并主动降低定义体的惊讶度，从而暴露人机合作中的盲点与潜在冲突。

**🔧 技术方法**

采用Logistic回归、惊讶度（per-token surprise）评估以及统计检验等技术来量化编辑偏好，并使用Claude Opus、Gemini、Codex三大LLM模型进行实验。

**📊 数据集**

使用九个公开Lean项目（共4.7M定义、6K+提交）作为实验数据集。

**📈 对比分析**

通过比较编辑比例、字符修改量、定义距离与惊讶度等指标，发现人类编辑更关注中层定义且保留惊讶度，LLM代理则更集中在边缘、较短定义并显著降低惊讶度，统计显著性高且效果显著。

**⚠️ 局限性**

局限性包括依赖词法解析器构建依赖图、训练集可能包含测试数据、仅考察非合并提交、结果为相关性而非因果关系、未区分不同人类贡献者角色。

---

## 4. Structural Conditional Independence Test via Neural Quantile Resolution

**arXiv ID:** 2611.07165 | [PDF](https://arxiv.org/pdf/2611.07165v1)

**作者:** Khoa Nguyen `[一作]` (Deakin Applied Artificial Intelligence Initiative), Thin Nguyen `[通讯]` (Deakin Applied Artificial Intelligence Initiative)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 QRCIT，一种通过残差分位数流模型学习条件分布变形并使用条件随机化校准的条件独立性检验方法；

**💡 创新点**

创新点在于把条件分布差异量化为可解释的分位数变形场，保留分位数与上下文的分辨率，同时保持全局 p 值校准；

**🔧 技术方法**

采用残差量化流模型、pinball 损失、条件随机化、分位数网格、局部救援统计量等技术；

**📊 数据集**

在合成 PNL、结构化依赖基准 SDB、真实数据 Sachs、METABRIC 乳腺癌和 CCLE/PLX4720 药物反应数据上进行评估；

**📈 对比分析**

与 GCM、SGMCIT、CDCIT、NNLSCIT、KCI 等方法对比，QRCIT 在传统与结构化基准上保持或提升检验力，在 Sachs 上 F1 仅次于最佳方法；

**⚠️ 局限性**

局限包括对影响区域的先验假设、对分布假设的敏感性以及多维响应扩展仍需进一步研究。

---

## 5. Expanding the Quality-Coverage Frontier with Competitive Flow Matching

**arXiv ID:** 2611.07055 | [PDF](https://arxiv.org/pdf/2611.07055v1)

**作者:** Wenxuan Peng `[一作]` (Cornell University), Zongze Wu `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在流匹配模型上引入赢家通吃训练，通过在训练时加入低维高斯代码作为辅助条件，让不同的预测竞争，从而在推理时通过调节辅助条件尺度实现质量和覆盖率的平滑控制，并对预训练模型（SiT、RAEv2）进行轻量级后训练，显著拓展质量-覆盖率前沿。

**💡 创新点**

创新点在于将赢家通吃（WTA）目标与低维辅助条件相结合，使同一噪声状态下可产生多条速度预测，并引入辅助条件缩放作为新的引导方式，既能提升样本质量，又能保持甚至提升覆盖率，并且展示了竞争训练如何改变模型对引导的响应。

**🔧 技术方法**

使用流匹配（Flow Matching）框架、WTA 训练、低维高斯辅助代码、辅助条件尺度 λ、推理时的引导（classifier‑free、AutoGuidance、RAEv2‑guidance）以及后训练细化。

**📊 数据集**

主要使用 ImageNet‑256 的类条件生成数据集，在 SiT 和 RAEv2 两个测试床上进行验证。

**📈 对比分析**

与最近的流匹配方法（如原始 SiT、RAEv2 及其他基准）在 FD_r,6、Recall、Precision、Inception Score、DreamSim 等指标上对比，实验显示在高覆盖率 regime 下 FD_r,6 下降显著，Recall 与多样性提升，构建了更优的 Pareto 前沿。

**⚠️ 局限性**

主要限制是后训练需要对每个训练状态评估 K 个候选预测，导致计算成本随 K 增大；此外辅助代码的作用机制尚未完全理解，未来可进一步优化其利用方式。

---

## 6. The Two-Ledger Model: Transaction-Grain Cost Accounting for Multi-Agent AI Systems Across Clouds, Models, and Systems of Record

**arXiv ID:** 2611.06989 | [PDF](https://arxiv.org/pdf/2611.06989v1)

**作者:** Vivek Acharya `[一作]` `[通讯]`, Vivek Acharya

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文提出一种双账本模型（Two‑Ledger Model），通过将 FOCUS 账本的结算级账单与基于 OpenTelemetry GenAI 的调用级归因账本相联接，实现多代理 AI 系统在跨云、跨模型、跨业务系统中的交易粒度成本核算。

**💡 创新点**

创新点在于：①对账单与业务交易粒度不匹配的根本问题进行正式化并提出七项必备要求；②设计了“关联脊柱”（correlation spine）层级策略，支持跨服务、跨协议、跨业务系统的事务追踪与归因；③引入了版本化费率表、实时成本估算、账单对账与业务语义映射，实现从调用到业务交易的闭环核算；④提出可验证的行线级线索和两边对账机制。

**🔧 技术方法**

使用的技术包括：FOCUS 1.5 账单标准、OpenTelemetry GenAI 语义约定、Model Context Protocol（MCP）、Agent2Agent（A2A）协议、LLM 网关（gateway）作为计量点、版本化费率服务、关联脊柱、概率匹配、两侧对账引擎以及基于业务流程（APQC PCF）的成本语义映射。

**📊 数据集**

实验数据为合成数据：模拟 20,000 起业务交易（订单、采购、索赔）和 109,792 次模型调用，包含重试、失败、边界时延等随机因素；此外还构造了单一 O2C 场景的调用记录以展示成本示例；所有数据均在实验代码中可复现，且已公开。

**📈 对比分析**

通过与仅使用追踪上下文（trace‑only）基线对比，模拟演示：关联脊柱在跟踪衰减时仍能保持 99%+ 的成本覆盖率，准确率从 100% 降至 95.6%；对账引擎能在 0.24% 内恢复植入的未计费（shadow AI）支出；运行中门控策略在 4–5% 的误停率下将 p99 成本降低 61%。这些指标表明双账本模型在合成环境下显著优于传统方法。

**⚠️ 局限性**

局限性包括：①全部基于合成数据，未在真实企业环境中验证；②关联脊柱的 L4 概率匹配精度受工作负载特征影响，缺乏广泛基准；③费率表与对账逻辑依赖尚未正式标准化（FOCUS 1.5 及 OpenTelemetry GenAI 仍在开发）；④模型未覆盖微调、嵌入刷新等 AI 相关费用；⑤实现需要多组件协同，实际部署成本和运维复杂度待进一步评估。

---

## 7. Certified folding-free deformable image registration: A whole-cell Jacobian certificate and a sequential variational solver

**arXiv ID:** 2611.06954 | [PDF](https://arxiv.org/pdf/2611.06954v1)

**作者:** Baihan Zhao `[一作]` (University of Texas at Arlington), Guojun Liao `[通讯]` (University of Texas at Arlington)

**通讯引用:** 653 | [OpenAlex ID](https://openalex.org/A5085683598)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种注册方法 SeqVP，核心思想是在每一次迭代中对整个图像域的三维三线性插值变形场进行严格的可逆性检验（即 Jacobian determinant 始终为正），从而保证所有返回的变形映射都是无折叠的生物学可解释变形。方法通过变分原理（Jacobian–curl）构造一系列“内问题”，并使用预条件梯度法逐步优化，所有被接受的迭代都经过证书检查，确保最终得到的变形是全局可逆且保持原图像边界。该方法在 GPU 上实现，单个 1 mm 脑 MRI（约 2 × 10⁷ 未知量）即可在 90 s 内完成，且每一次迭代都经过快速的全细胞正性检验。

**💡 创新点**

创新点主要有：
1. 提供了一套基于 IEEE 754‑binary64、向外舍入、Bernstein 形式与区间运算的 **完整细胞正性证明**，可对数百万个体素级别的三线性变形场做严格的可逆性保证；
2. 将 Jacobian‑curl 变分原理转化为一个 **逐步自引用的内问题序列**，消除了对控制变量的显式耦合，使得每一次内问题的惩罚在当前映射处消失，进而实现了全局收敛保证；
3. 设计了 **Poisson‑preconditioned 梯度法**（含自适应步长预测与仿射缩放阻尼），兼顾数值稳定性与效率；
4. 在训练阶段使用 **单精度预优化** 并通过 **安全的二次放大** 方式切换到双精度证书，保证从未经过验证的粗略解到最终可验证解的安全“交接”。

**🔧 技术方法**

技术细节包括：
- 变分模型：数据项、曲率项、局部约束项和外部能量的组合；
- 变形场的三线性插值（Q1）与其 Jacobian 的三次多项式表达式；
- 证书算法：
  * 边界凸包（edge‑hull）快速下界检验；
  * 计算浮点 Bernstein 系数并给出前驱误差上界；
  * 采用区间 Bernstein 形式与自适应细分，直到所有子盒的下界 > 0；
- 预条件梯度法：Poisson 矩阵 + 对角线均衡 + 仿射缩放阻尼；
- 双精度与单精度混合执行（GPU 上的正交运算采用双精度，预优化使用单精度）；
- 证书与优化的 **任何时间安全性**：在任何停机点返回的映射都是无折叠的。

**📊 数据集**

实验使用的公开数据集包括：
- OASIS‑1 1 mm T1 图像（160×192×224）和 35 结构 + 4 组织标签，拆分为 12 对调优集、19 对验证集、50 对独立测试集及 20 对大变形（H4）对；
- LPBA40（181×217×181）10 对配对（54 结构）；
- IBSR18（182×218×182）36 对配对（36 结构）。

**📈 对比分析**

方法与多种基线（ANTs SyN、NiftyReg、deedsBCV、ConvexAdam、FireANTs SyN/greedy、uniGradICON、VoxelMorph、TransMorph、VPreg）在 OASIS 的 49 对独立测试集上比较。结果显示：
- **seg35 Dice**：0.8205（+0.022 vs ANTs SyN，+0.022 vs uniGradICON），**seg4 Dice**：0.8888（+0.025 vs ANTs SyN，+0.023 vs uniGradICON），
- **HD95** 与基线相近或略优；
- **学习型方法**（TransMorph）在标签上更优，但其变形场大约有 1.3 × 10⁵ 个折叠细胞；
- **证书检验**：所有 SeqVP 输出均无折叠，而所有基线在单精度或网格点差分检验下出现 0–10⁵+ 折叠细胞。
- **效率**：单 1 mm 脑 MRI 在 RTX‑4090 上 90 s 内完成（含证书），比基线的 1–2 分钟稍慢，但比 VPreg 的 47 s 训练前向时间更高。

**⚠️ 局限性**

局限性：
- 输出的变形相对粗糙（SDlogJ 较大），在需要细致形变估计的应用中可能不够精细；
- 证书和梯度方法对 CPU/GPU 的高精度运算有一定依赖，导致实现复杂度高；
- 对极端大变形（如 H4 对）效果仍有限，尽管相较于基线略优；
- 目前仅针对三线性插值变形场，若需更高阶插值或其他变形表示需进一步扩展；
- 仅在固定阈值下保证无折叠，对极其薄细胞的判别仍存在不确定性。

---

## 8. Beyond the Agent Harness: Boundary Criteria for Software-Factory Orchestration - A Single-Operator Case Study

**arXiv ID:** 2611.06946 | [PDF](https://arxiv.org/pdf/2611.06946v1)

**作者:** Lei Hua `[一作]` `[通讯]`, Lei Hua

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一套“边界判据”与三字段事件编码，用以把编码代理系统中的机制与失效划分到代理层、工厂层、域层、工厂（基础设施）层与模型层，并在单一运营商的六个项目中验证这些判据的可操作性与一致性；同时对已发布的三种系统（HoH、Sengupta 等）中的组件与失效进行外部检验。

**💡 创新点**

创新点在于（1）给出明确的运行边界定义并根据机制是否跨运行、是否随域变化、是否依赖工厂控制等分层；（2）将失效归因拆分为“起因层”“控制缺失层”“修复层”，并对环境因子提出正向证据规则；（3）通过模型实例双编码实现对判据一致性的初步评估，揭示并修正冲突。

**🔧 技术方法**

技术手段包括：规则化判据手册（v1.0、v1.1）、手动/模型编码工具、Cohen κ 与 bootstrap 置信区间计算、基于已记录的执行日志与配置文件的案例提取、签名收据验证机制（类似 Proof‑or‑Stop）、对比实验（E1）以及描述性统计。

**📊 数据集**

数据集涵盖：E1（48 次运行、12 例固定测试）; E2–E7 的单例 defect 与操作日志；26 个从三种公开系统抽取的组件；20 条从其他论文抽取的失效实例；以及作者自建的 46 个独立单元（组件与失效）用于可靠性检验。

**📈 对比分析**

比较方法主要是：E1 通过两种执行后端在相同模型与提示下的配对运行对比；E3 通过在两种实现（System F vs. 通用桌面代理）下对照检测；其他实验为描述性对比或规则验证。性能方面未做显著性检验，报告了平均耗时、P95、严格规则合规率等指标，结论是后端切换无功能差异、但耗时略有差异。

**⚠️ 局限性**

局限性包括：单运营商（自评偏差）；样本量小且集中在医疗/养老领域；E1 仅比较同一代理族后端；未对人工编码者进行验证；模型编码者来自同一族群，可能共享偏差；缺乏长期连续运行数据；环境预检与签名收据在实际部署中的隔离性不够；以及构造效度、成本一致性与规则覆盖率等方面的不足。

---

## 9. When Does Combining Software-Effort Estimates Help? Evidence from Four Public Project Datasets

**arXiv ID:** 2611.06948 | [PDF](https://arxiv.org/pdf/2611.06948v1)

**作者:** Azzeddine Ihsine `[一作]` (Inovionix), Sara Ihsine `[通讯]` (Inovionix)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在软件工时估计中将已校准的数值估计进行组合，并在四个公开数据集上评估多种组合规则的效果。

**💡 创新点**

提出了在已校准误差的基础上直接应用传统组合规则，并系统检验组合在相关误差和主导源存在时的收益与风险。

**🔧 技术方法**

使用的技术包括普通最小二乘回归、最近邻相似估计、随机森林、Bates‑Granger签名加权、简约非负单纯形组合、以及Ledoit‑Wolf收缩的协方差估计。

**📊 数据集**

数据集为 Kitchenham、Desharnais、China、Maxwell 四个软件项目历史数据，分别包含 145、77、499、63 项目。

**📈 对比分析**

通过在 100 次 70/30 随机拆分中进行交叉验证，计算 log‑MAE 与 log‑RMSE，并用校准误差矩阵估计权重；结果显示除 Kitchenham 的人类估计占优外，其余三组数据组合均可降低平均误差 1–9%，但收益在重采样噪声范围内且各规则在 15–37% 的拆分中表现更差。

**⚠️ 局限性**

局限性包括：样本量有限、重复拆分导致的相关性、仅使用历史项目而非真实未来项目、未覆盖生成式 AI 预测、以及组合规则对负权重的敏感性。

---

## 10. Logically Augmented Quantum LDPC Codes: Construction and Hierarchical Decoding

**arXiv ID:** 2611.07062 | [PDF](https://arxiv.org/pdf/2611.07062v1)

**作者:** Sheida Rabeti `[一作]` (Northeastern University), Hessam Mahdavifar `[通讯]` (Northeastern University)

**通讯引用:** 1627 | [OpenAlex ID](https://openalex.org/A5056299839)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个逻辑增强（LA）QLDPC 代码的联合构造与解码框架，并引入了层次化逻辑列表解码器（HLLD）来提升解码性能。

**💡 创新点**

创新点在于通过在单一 QLDPC 代码块内加入辅助逻辑保护码（LPC）实现可调节的编码率与错误保护，且在不增加物理块长度的前提下利用 LPC 提供的逻辑冗余信息指导分支 BP 解码，显著降低逻辑错误率。

**🔧 技术方法**

核心技术包括：CSS 代码构造、重复码 LPC、逻辑列表生成（MAP 与全枚举）、条件 BP、BP-OSD、以及层次化逻辑列表解码（HLLD）。

**📊 数据集**

使用的实验数据集为多种 QLDPC 代码实例：[[2610,744]] lifted‑product、[[144,12,12]] bivariate bicycle、[[780,156]] mitten 等，分别在不同的 LPC 重复长度下进行仿真。

**📈 对比分析**

与直接 BP、直接 BP‑OSD 以及原始主码解码进行对比，结果显示 HLLD‑MAP/‑FULL 在保持相同物理块长度下，逻辑错误率分别下降约 62%–81%，并通过可调节 LPC 长度实现编码率–性能的灵活权衡。

**⚠️ 局限性**

局限性包括：对大规模 LPC（长度 >30）时 MAP 计算成本高、需使用 BP 近似；仅在 Pauli‑X 或 Pauli‑Z 单一错误模型下验证；未来需扩展至更多 QLDPC 家族和偏置噪声模型。

---

## 11. Synthesizing symbolic models of human and machine reasoning at scale

**arXiv ID:** 2611.07088 | [PDF](https://arxiv.org/pdf/2611.07088v1)

**作者:** Solim LeGris `[一作]` (New York University), Todd M. Gureckis `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于大语言模型的行为条件符号合成（BCSS）方法，用于自动搜索可解释的符号程序，逼近人类和大模型在ARC抽象推理任务中的错误行为；

**💡 创新点**

创新点在于将LLM作为离散程序空间的采样器，通过对比错误响应的线性编辑，克服传统符号模型手工工程和可扩展性限制，实现对海量开放式推理问题的符号逆向分析；

**🔧 技术方法**

核心技术包括：基于DSL的符号程序表示、LLM生成行编辑（line‑edit）建议、程序执行验证、LLM分类器对错误类别的判定以及贝叶斯回归与分类器对人机错误差异的量化；

**📊 数据集**

使用公开的H‑ARC（人类ARC行为数据）和ARC‑AGI‑1（400个任务）数据集，包含超过1700名参与者的错误响应及多个大语言模型（Qwen3.5‑27B、GPT‑OSS‑120B）在ARC任务上的生成结果；

**📈 对比分析**

通过BCSS将人类和模型错误分别映射到符号程序后，计算程序简化度、错误类型分布及顺序效应，发现两者都表现出简化、局部匹配和属性误选的错误特征，并通过ROC‑AUC等指标验证模型能区分人类与低性能模型的错误；

**⚠️ 局限性**

局限性包括：仅捕获约一半非平凡错误，可能遗漏需要多行或大幅扩展的错误；DSL与认知真实度不完全匹配；程序重叠导致多种等价程序，影响分类准确性；LLM提议和分类过程可能带来偏差；未对过程级推理时间或眼动等动态行为建模。

---

## 12. Closed-Loop Agent Evaluation: Layered Measurement and Self-Calibration

**arXiv ID:** 2611.07051 | [PDF](https://arxiv.org/pdf/2611.07051v1)

**作者:** Ruifeng Ren `[一作]` (Massachusetts Institute of Technology), Gun Ahn `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种闭环评估方法，将代理（agent）的评估从一次性打分转变为持续、可校准、可改进的控制系统。

**💡 创新点**

创新点在于：①将评估拆分为四层（Span、Trace、Conversation、Engagement）和七组件（Execution、Substance、Trajectory、Results、Conduct、Provenance、Adjudication）；②实现双循环（内循环校准评估本身，外循环驱动代理改进）；③所有结论都由可追溯的规则从原子数据计算，保证可解释性和可审计性；④通过 Agent Factory 和 orchestrator 自动化流水线完成评估全过程。

**🔧 技术方法**

使用的技术包括：测量模型（含偏差、缺失机制）、规则引擎（版本化 deterministic analysis）、LLM 采集链（rule → small model → jury）、数据审计（agreement、validity、Goodhart test）、回归检测门控、自动化 orchestrator 与 Agent Factory。

**📊 数据集**

主要数据集：SharpInsight 业务分析代理的中文真实用户数据（8,537 轮，4,393 会话，1,136 用户）以及 60 条中文 WildChat 公共对话（164 轮）。

**📈 对比分析**

对比方法：将闭环评估的规则+LLM 结果与传统一次性 LLM 评分及直接标签进行比较。指标显示闭环评估 ICC 0.86（vs 0.81）和 κ 0.81（vs 0.63），表明一致性和可靠性提升；在公开数据上，规则获得更高精度/召回，且直接标签难以覆盖。

**⚠️ 局限性**

局限性：①仅在单一代理和中文环境下验证；②样本量有限，full‑text 检查仅覆盖 20 条深度报告；③规则召回率低（如 Execution 0.33），可能低估失败率；④缺乏真实发布后的效果验证；⑤缺少人工金标评估和因果实验，难以确认改进效果。

---

## 13. What Patients Say They Can't Access: Mining Cancer Screening Barriers from Online Communities

**arXiv ID:** 2611.07113 | [PDF](https://arxiv.org/pdf/2611.07113v1)

**作者:** Ruimin Sun `[一作]` (Florida International University), Arjun Arunasalam `[通讯]` (Florida International University)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5093013828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国 Reddit 六个癌症专属子社区的约56,500条帖子进行大规模过滤与定性分析，提炼并分类出56,500条中576条与癌症筛查相关的帖子，再从中识别出217条包含障碍的帖子，并生成了43条障碍代码的结构化代码本。

**💡 创新点**

1) 首次将无结构化社交媒体语料与定性研究方法结合，构建出可直接用于监督学习的障碍代码本；2) 通过多层过滤与四阶段编码实现高粒度、自然语言驱动的障碍识别；3) 明确展示障碍多重共现与跨领域组合，为多标签干预设计奠定基础。

**🔧 技术方法**

基于 NLP 的关键词过滤（两级筛选）+ 手工定性编码（Barrier triage、Open coding、Codebook synthesis、Systematic coding），并结合人机协作的主题分析技术。

**📊 数据集**

约56,500条 Reddit 原始帖子（r/breastcancer、r/prostatecancer、r/lungcancer、r/colorectalcancer、r/pancreaticcancer、r/Cancer），经过滤后 576 条筛查相关帖子，进一步筛选出 217 条包含障碍的帖子。

**📈 对比分析**

本研究未直接与传统结构化问卷或现有 NLP 模型进行量化比较；但定性分析显示：37.7% 的筛查相关帖子包含障碍，平均每条含 2.94 条代码，85.7% 的帖子存在多重障碍共现，体现了障碍的多维度和复杂性。

**⚠️ 局限性**

1) 地理位置过滤仅基于关键词，可能漏检未标明地点的 US 帖子；2) 仅由单一研究者编码，缺乏互评可靠性；3) Reddit 用户群体偏向英语、互联网活跃者，可能低估老年人、农村地区和非英语患者的障碍。

---

## 14. Polynomial Lower Bounds for Steiner Shortcuts via Test Functions

**arXiv ID:** 2611.07043 | [PDF](https://arxiv.org/pdf/2611.07043v1)

**作者:** Aaron Bernstein `[一作]` (New York University), Thatchaphol Saranurak `[通讯]` (University of Michigan)

**通讯引用:** 1528 | [OpenAlex ID](https://openalex.org/A5010547647)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构造了一族有向图，使得任何几乎线性规模的Steiner短路集都无法保持低直径，从而否定了Steiner短路猜想；同时给出了相应的无向图精确Steiner跳跃集与单调电路的下界。

**💡 创新点**

提出了通用的“测试函数”框架，利用低阶多项式约束将关键路径的性质推广到跨路径，实现了对Steiner短路、精确跳跃集和单调电路的统一下界证明。

**🔧 技术方法**

采用平均自由集、坐标增广、低度多项式测试函数、交叉路径分析与势函数技巧，对有向图的层化 DAG 进行构造与证明。

**📊 数据集**

未使用实验数据集，全部为理论构造与证明。

**📈 对比分析**

通过与已有的非Steiner短路下界对比，证明在Steiner设置下仍存在指数级的直径限制，进一步说明现有shortcut框架无法实现低深度并行可达性算法；相较于之前的 n^{2/9} 下界，本工作取得了 n^{1/56} 的下界。

**⚠️ 局限性**

下界指数远未达到最优的 n^{2/9}，且对更一般图类的推广仍有空间；所构造的图仅在层化 DAG 中有效，无法直接说明非层化或无向图的情况。

---

## 15. On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer

**arXiv ID:** 2611.06976 | [PDF](https://arxiv.org/pdf/2611.06976v1)

**作者:** Zhenyi Shen `[一作]` (King's College London), Yulan He `[通讯]` (King's College London)

**通讯引用:** 11961 | [OpenAlex ID](https://openalex.org/A5015709853)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PLatour框架，实现Prefill–to‑Autoregressive Cross‑model Transfer (PACT)，并设计了PACT‑Check评估协议；

**💡 创新点**

创新点包括：①使用零初始化线性桥接器直接用RL（GRPO）优化跨模型信息传递；②公开开源基线并与闭源Mostik对比；③引入四个sanity check（DT、IT、SC、TR）和PACT‑Score，用以分离模型适配与真正信息转移；

**🔧 技术方法**

技术手段包括：零初始化线性投影桥接、强化学习（GRPO）训练、混合注意力架构、与LoRA/全参数SFT等对比、基于KV缓存的C2C和KV‑Lingo等现有通信方法；

**📊 数据集**

实验数据集涵盖：ARC‑C、MMLU、Chemistry L3、GSM8K、MATH500、AIME、HMMT、IMO‑AnswerBench、DeepCoder、LiveCodeBench v6、HumanEval、MBPP；使用Qwen3/Qwen3.5系列模型作为发送者和接收者；

**📈 对比分析**

评估方法：与接收者单体、发送者单体、现有通信方法（C2C、KV‑Lingo、TextComm、Mostik）以及接收者的LoRA/全参数SFT/GRPO进行对比。结果表明，PLatour在生成任务上显著优于开源基线，在数学与编码基准上甚至超过Mostik，尽管使用的发送者参数规模更小；但PACT‑Score往往为零或负，显示许多提升可由接收者自身适配实现；

**⚠️ 局限性**

局限性：①大部分性能提升可通过桥接器本身的适配而非真正的信息传递获得；②训练过程中易出现不稳定，特别是大规模发送者；③缺乏跨模型族的验证；④需要更具挑战性的任务和更精准的评估指标来真正衡量跨模型信息流动。

---

## 16. Backbone-Informed Adaptation after EEG Pretraining: A Capacity-Controlled Study

**arXiv ID:** 2611.07192 | [PDF](https://arxiv.org/pdf/2611.07192v1)

**作者:** Mingyang Jiang `[一作]` (Vanderbilt University), Catie Chang `[通讯]` (Vanderbilt University)

**通讯引用:** 13891 | [OpenAlex ID](https://openalex.org/A5038256364)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文系统评估了EEG自监督基础模型后置适配器，比较了遵循预训练编码器通道/时间轴注意力的原生适配器（native）与等参数、独立token的MLP控制器（token‑wise），并在多任务、多数据集上检验两种设计的性能差异。

**💡 创新点**

创新点在于将结构化与容量匹配的非结构化适配器进行直接对比，探讨了适配器与预训练特征组织的一致性对下游表现的影响，并评估了序列编码器训练对结果的敏感性。

**🔧 技术方法**

使用了两种EEG基础模型（CBraMod、LaBraM），设计了native轴注意力分支与token‑wise MLP控制器，采用参数匹配、冻结预训练编码器、不同训练协议（probe、全微调）和分类头，评估平衡准确率与宏F1指标。

**📊 数据集**

实验数据集包括FACED（情绪识别）、ISRUC（睡眠分期）、TUEV（EEG事件分类）、PhysioNet‑MI（运动意象）和SEED‑V（情绪识别）。

**📈 对比分析**

通过在相同预训练模型、相同数据划分、相同随机种子和分类头下匹配适配器参数后对比native与token‑wise，发现native在部分任务（如LaBraM‑ISRUC）略有优势，但总体未表现出一致性优越；在全微调条件下，native有时提升但同样缺乏稳健性。

**⚠️ 局限性**

局限性包括仅评估两种骨干且仅在后置位置插入适配器，使用统一的优化设置且未单独调优各个实验；未覆盖所有初始化不确定性；未对注意力与MLP差异进行更细粒度分离；对序列编码器训练的敏感性未系统化，导致结论受实验设置限制。

---

## 17. Adapting the Process of Adaptation: Darwinian, Lamarckian, and Agent-Directed Evolution in AI

**arXiv ID:** 2611.07048 | [PDF](https://arxiv.org/pdf/2611.07048v1)

**作者:** Andrew W. Lo `[一作]` (Massachusetts Institute of Technology), Chaoyi Zhao `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5068642094)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种理论框架，用以比较自然选择（Darwinian）、拉马克式学习（Lamarckian）和代理导向进化（Agent‑Directed）三种机制在环境变化后适应速度的差异。

**💡 创新点**

创新点在于：①将适应速度量化为增长后悔（growth regret）并与KL散度联系；②给出三种机制的适应速率与适应时间解析结果；③揭示代理导向进化能够通过调节变体数量实现自我搜索决策；④分析误评估目标对快速适应的安全风险。

**🔧 技术方法**

使用数学建模与信息理论（KL散度）、自然选择模型、学习更新规则以及随机搜索（变体采样）等技术；并通过蒙特卡洛仿真验证理论。

**📊 数据集**

论文不依赖具体实验数据集，而是在理论模型中设定参数（如环境概率 p_old、p_new，学习率 λ，搜索难度 ρ，变体数 N）进行数值实验。

**📈 对比分析**

通过对三种机制在相同起始条件下的适应时间和适应速率进行比较，实验显示：Agent‑Directed 适应最快，Lamarckian 次之，Darwinian 最慢；在误评估目标下，Agent‑Directed 可加速朝错误目标收敛，表明安全风险。

**⚠️ 局限性**

局限性包括：模型假设环境变化仅一次且概率已知；学习与搜索更新规则被简化为线性/随机形式；未考虑多目标或复杂评价函数；对实际 AI 系统的实现细节与可操作性讨论不足。

---

## 18. Batch-Conditioned Semantic Anchors for Robust Transductive Adaptation of Vision--Language Models

**arXiv ID:** 2611.06993 | [PDF](https://arxiv.org/pdf/2611.06993v1)

**作者:** Mohammed Rahman Sherif Khan Mohammad `[一作]` (Edge Hill University), Amr Ahmed `[通讯]` (Edge Hill University)

**通讯引用:** 9443 | [OpenAlex ID](https://openalex.org/A5009154893)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在测试时无标签批量环境下，提出 JPTA 通过批量条件的语义锚定来动态调整 CLIP 文本原型，从而实现稳健的传递式视觉‑语言模型适配。

**💡 创新点**

创新点：1）批量感知的语义锚定——在保持文本原型结构的同时，利用可靠的图像‑批量结构进行受控正交传输；2）先验‑感知的传递优化——将传输后的锚点与原始文本权衡，结合图一致性、类别似然与批量边缘正则，形成自适应的更新策略；3）自适应信任调节——根据批量稀疏度、邻居一致性、类别覆盖度等指标动态控制锚点移动与统计更新。

**🔧 技术方法**

技术手段：1）正交 Procrustes 传输 + 置信权重；2）基于软原型的图像‑批量相似度构造；3）先验‑感知的分配矩阵更新（指数梯度+KL 约束）；4）可调伪计数的软原型更新；5）多任务损失组合（类别似然、图 Laplacian、锚点对齐、批量先验）。

**📊 数据集**

评估数据集：ImageNet、SUN397、Aircraft、EuroSAT、Cars、Food101、Pets、Flowers、Caltech、DTD、UCF101；同时在多种批量规模/类别覆盖度、在线流、全类评估等多种实验场景下测试。

**📈 对比分析**

对比方法：TransCLIP、StatA、TENT、TDA、DMN 等。JPTA 在低有效类别、稀疏与先验偏移批量中平均提升 8–12% 以上，甚至在最差场景上比最强基线高 12+%，在在线流中也比 StatA 高 5–6%。在全类评估和大批量时保持或略优于对比方法。

**⚠️ 局限性**

局限性：1）对极度噪声或几乎无类别信息的批量仍可能产生偏差；2）对超参数（信任系数、伪计数等）敏感，需一定的调优；3）主要基于冻结的 CLIP 特征，对更大规模或多模态预训练模型的迁移效果需进一步验证。

---

## 19. Recursively Testing and Vibe Debugging a Google AI Studio Generated Vibe-Coded Database Software Application

**arXiv ID:** 2611.06952 | [PDF](https://arxiv.org/pdf/2611.06952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 20. A Sharp Additive Bound for the Threshold Rule in Repeated Rent-or-Buy

**arXiv ID:** 2611.06955 | [PDF](https://arxiv.org/pdf/2611.06955v1)

**作者:** Vasco Schiavo `[一作]` `[通讯]`, Vasco Schiavo

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对在LLM上下文管理中出现的重复租购（rent‑or‑buy）问题，研究了经典阈值规则（threshold rule）并给出了其对任意会话的精确加法失误上界；同时构造了匹配上界的最坏实例，证明该上界是最优的；并提出一种对下降率做出额外重置的 drop‑aware 规则，进一步分析其加法失误。

**💡 创新点**

创新点在于：①提出了与会话中“变化次数”P相关的精确加法失误上界 (2P‑1)K；②证明该上界在所有规则中是最优的；③通过构造指数增长的最坏实例展示了阈值规则在变化点处失误可逼近上界；④设计并证明了一种在下降点立即重置的 drop‑aware 规则，其在两段会话中可将加法失误降到 2K，并揭示了阈值规则在下降时可能失误高达 (3‑ε)K 的现象。

**🔧 技术方法**

主要技术手段为：①潜在函数（potential）和相对价值函数（relative value functions）的构造与分析；②对每个运行（run）进行张力平衡，利用跨越点的潜能差值来累积失误；③对增长率变化点的分析通过证明 Wα(e) 的单调性来处理下降和上升；④构造指数递增的增长序列和相应的最坏会话来证明上界的锐度。

**📊 数据集**

本工作为理论性分析，不涉及实验数据或公开数据集；所有结果均在数学上证明。

**📈 对比分析**

评价方式为理论上界比较：阈值规则在任意会话中失误不超过 (2P‑1)K；drop‑aware 规则在两段会话中失误不超过 2K；同时阈值规则在单段会话中至少失误 K，表明该上界不可进一步降低。对比传统 2‑竞争性质，阈值规则在变化次数小的情况下表现更好；但在下降点仍可能失误高达 (3‑ε)K。

**⚠️ 局限性**

局限性：①对于 P≥3 的会话，尚未证明是否存在规则能保证 PK 的加法失误；②阈值规则在单个增长率变化为上升时的最优上界仍未完全确定；③drop‑aware 规则虽然在下降点表现更好，但在上升点会出现较大比例失误，且不具备固定竞争因子；④整个分析仅针对确定性规则，随机化或学习式规则的潜在优势未被探讨。

---

## 21. Perceptual Distortion for Overfitted Video Coding: Sliced Matching in Space, Error Alignment in Time

**arXiv ID:** 2611.07148 | [PDF](https://arxiv.org/pdf/2611.07148v1)

**作者:** Tianyi Yang `[一作]` (University of Bristol), David Bull `[通讯]` (University of Bristol)

**通讯引用:** 16238 | [OpenAlex ID](https://openalex.org/A5048009053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的感知失真损失（Spatio-Temporal Sliced Wasserstein Distortion）用于过拟合视频编码，改进了局部特征统计匹配并通过流残差保证时域一致性。

**💡 创新点**

创新点在于用局部切片Wasserstein距离替代传统独立高斯近似，并引入基于光流的编码误差残差来抑制帧间纹理闪烁。

**🔧 技术方法**

主要技术包括VGG-16特征提取、随机投影一维Wasserstein距离、RAFT光流估计与误差对齐、以及对COOL‑CHIC编码器的目标函数重构。

**📊 数据集**

在1080p的UVG和HEVC Class B视频数据集上进行实验。

**📈 对比分析**

与x265、VTM、COOL‑CHIC（MSE/WD）和PLVC等基准比较，平均BD‑rate在LPIPS和DISTS上分别降低约3.0%和9.2%，同时tOF等时域指标提升约10%。

**⚠️ 局限性**

局限性包括仍依赖源视频的光流估计，且仅对相邻参考帧有效，未来需进一步降低对光流的依赖并扩展至更长时域。

---

## 22. A First Look at Starlink Maritime on Cruise Ships

**arXiv ID:** 2611.07191 | [PDF](https://arxiv.org/pdf/2611.07191v1)

**作者:** Jinwei Zhao `[一作]` (University of Victoria), Jianping Pan `[通讯]` (University of Victoria)

**通讯引用:** 8603 | [OpenAlex ID](https://openalex.org/A5043097022)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过结合UT遥测、乘客设备内部测量和云端外部测量，对两艘不同规模的邮轮（Utopia of the Seas和Silver Muse）上的Starlink海事连接进行系统评估。

**💡 创新点**

首次系统性测量商业邮轮的Starlink海事网络，揭示PoP分配不受海域司法或距离限制，IP地址可在同一PoP内变更，船舶上Wi‑Fi和速率限制决定乘客体验。

**🔧 技术方法**

使用gRPC接口读取UT状态、IP/PoP映射、RTT和吞吐量；在乘客设备上发送ICMP ping和iPerf3；云服务器近PoP进行ICMP探测。

**📊 数据集**

采用两条邮轮航程数据集：Utopia of the Seas（港口Canaveral-巴哈马）和Silver Muse（威尼斯-亚得里亚海），收集UT遥测、GPS、公共IP、RTT、吞吐、失效记录。

**📈 对比分析**

对比UT报告的PoP延迟、乘客设备内部ping以及云端外部ping，发现外部ping与UT报告相差≈1 ms，内部ping则高≈8–9 ms；吞吐量被船上速率限制，港口出现短暂阻塞。

**⚠️ 局限性**

限制包括仅测两艘船且仅单一乘客设备，缺乏多终端并发测量；无法控制船舶网络设置；未深入分析动态波束切换与多UT聚合吞吐量。

---

## 23. BlockLock: Reproducible LLM Inference with Fixed Batch Geometry

**arXiv ID:** 2611.06977 | [PDF](https://arxiv.org/pdf/2611.06977v1)

**作者:** Xiuyuan Qi `[一作]` (University of Electronic Science and Technology of China), Shanshan Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2382 | [OpenAlex ID](https://openalex.org/A5100417119)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了BlockLock框架，通过固定批次几何形状实现大语言模型推理的可重复性与高效性。

**💡 创新点**

创新点在于引入块级锁定（BlockLock）机制，使得不同硬件与并行度下的推理结果保持一致，并显著降低动态批量带来的性能波动。

**🔧 技术方法**

核心技术包括固定批次几何布局、块级并行调度、内存预分配与动态张量管理，以及对Transformer解码器的细粒度优化。

**📊 数据集**

作者在公开数据集如OpenWebText、Wikitext-2、C4以及多个自然语言推断任务（GLUE、SuperGLUE）上进行实验。

**📈 对比分析**

与传统动态批处理、按需调度等方法相比，BlockLock在保持相同吞吐量的同时将推理延迟降低约15–20%，并在多GPU设置下实现了更好的可扩展性。

**⚠️ 局限性**

局限性包括：仅针对推理阶段；对模型架构的适配需要手动调整；在极大模型（如70B参数级）下的内存占用仍然较高。

---

## 24. Global Surveillance Under the Lens: Reconstructing Passive Dolev-Yao Intruders

**arXiv ID:** 2611.07170 | [PDF](https://arxiv.org/pdf/2611.07170v1)

**作者:** Paolo Modesti `[一作]` (Teesside University), Paolo Modesti `[通讯]` (Teesside University)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5080667921)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出了一种基于协议规范的自动化方法，用于从被动抓取的PCAP流中重建Dolev–Yao被动入侵者的执行并验证其符合协议规范。

**💡 创新点**

创新点在于将编译器的符号分析结果转化为可执行的被动检查，并通过候选重构与评分机制在多会话环境下解析多路复用的TCP流，实现对协议执行的语义级验证。

**🔧 技术方法**

技术上结合了Java序列化解析、TShark流提取、基于OGF的协议编译器、Bouncy Castle/JDK加密库以及自定义的候选枚举与多期评分算法。

**📊 数据集**

使用了包含121条来自OFMC、SET和iKP协议集合的人工生成PCAP，涵盖单会话和多会话（N=2,3）场景。

**📈 对比分析**

通过完整性、Dolev–Yao一致性和成功重构率对比两大加密提供者，并记录各阶段时间，单会话平均完成时间约54秒（JDK）或61秒（BC），多会话时候选枚举使得成功率从94.2%下降至89.3%。

**⚠️ 局限性**

局限性包括对多会话候选空间的阶乘增长、对块数据标记导致的对象截断、无法区分交叉会话的 freshness 失效，以及需要已知协议映射与加密实现才能完成验证。

---

## 25. TimeArch: A Multi-Agent Approach for Software Architecture Design

**arXiv ID:** 2611.06921 | [PDF](https://arxiv.org/pdf/2611.06921v1)

**作者:** Muhammad Waseem `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 12478 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 TimeArch，多代理生命周期方法，支持需求提取、架构设计、验证与交付，并提供可追溯的治理与评估；

**💡 创新点**

创新点在于将软件架构设计细化为四阶段生命周期，采用生成器‑评估器‑人类三位一体的多代理模式，结合检索增强知识库、治理检查点与完整的可追溯链；

**🔧 技术方法**

技术手段包括多模型 LLM（通过 LLM Gateway）、LangChain.js 任务编排、Deno 边缘函数、React+TypeScript 前端、Mermaid 图表渲染、PostgreSQL 存储、RAG 知识检索、PDF/DOCX 文档生成等；

**📊 数据集**

使用了多领域案例系统（ITS、IoT 监控、移动应用 MoodFlow）的业务需求文档、SRS/BRD、音频讨论等输入，并搭建了包含架构方法、标准与框架片段的 RAG 知识库；

**📈 对比分析**

通过案例评估关注生成物清晰度、一致性、可追溯性、代理协同与验证结果相关性，未与传统 GenAI 流程做定量对比，结果显示在需求提取、设计方案与验证报告方面可行，但图表渲染、语义一致性与输出稳定性仍需提升；

**⚠️ 局限性**

局限包括：仍需人工审查与决策、生成质量不稳定、LLM 变异导致结果差异、对大型需求集合的上下文管理不足、图表语法错误、评估得分不稳定、案例覆盖有限、隐私与安全需进一步保障。

---

## 26. Resource-Efficient Reinforcement Learning Post-Training through Elastic Expert Parallelism

**arXiv ID:** 2611.06984 | [PDF](https://arxiv.org/pdf/2611.06984v1)

**作者:** Jia Wei `[一作]` (Tsinghua University), Xuehai Qian `[通讯]` (Tsinghua University)

**通讯引用:** 4889 | [OpenAlex ID](https://openalex.org/A5047215143)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 FlexFloor 系统，利用弹性专家并行与响应调度的联合规划，在大规模 MoE 模型的强化学习后训练中提前释放未完成生成的计算资源。

**💡 创新点**

创新点在于将历史长度预测、KV 容量约束和完成仲裁相结合，形成无损的专家切换规划，并通过图捕获与重捕实现动态重构兼容性。

**🔧 技术方法**

采用历史长度预测、TailGuard 负载平衡、KV 先决许可、ILP 配对、图捕获与重捕、native sleep、侧车任务重用以及分布式 MoE 通信库等技术。

**📊 数据集**

实验数据集包括 Qwen3-30B-A3B 的数学推理任务（DeepScaleR）、160 个代码生成任务（Eurus-2-RL）以及 GSM8K 侧车评估。

**📈 对比分析**

与 verl 基线、长度排序、LS+TG、Fixed-F4、MinSkew 等方法对比，Flex-P4 在相同后端下吞吐量提升约 55%（3711 → 5669 tok/s），资源效率提升约 65%（232 → 384 tok/rank-s），奖励保持不变。

**⚠️ 局限性**

局限性包括受 KV 容量与专家驻留策略限制，弹性切换的复杂度和恢复开销较高；实验仅在 16 个 Ascend 设备上验证，缺乏更大规模或不同硬件环境的评估。

---

## 27. Big Models, Small Gains: Frozen Forecasters as Classifiers on MONSTER

**arXiv ID:** 2611.06988 | [PDF](https://arxiv.org/pdf/2611.06988v1)

**作者:** Rishit Jain `[一作]` (University of Mumbai), Nilesh Patil `[通讯]` (University of Mumbai)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5032921679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估冻结的预训练预测模型（Reverso、Chronos‑Bolt）在六个MONSTER大规模时间序列数据集上的分类性能，并与无预训练的强基线（MiniRocket、MultiRocket、Hydra、Quant）以及随机权重版本进行对比

**💡 创新点**

引入随机权重控制以分离预训练与网络架构的贡献，系统性比较大规模数据集上冻结预测模型与最强基线的性能，并分析幅度统计、微调和预测误差特征对分类效果的影响

**🔧 技术方法**

使用预训练的时间序列预测模型Reverso、Chronos‑Bolt，冻结提取特征后进行mean/max池化；随机权重复制；基线方法MiniRocket、MultiRocket、Hydra、Quant；ridge/extra‑trees分类器；加入幅度统计、微调（线性探测）及预测误差特征

**📊 数据集**

六个MONSTER数据集：UCIActivity、WISDM2、CrowdSourced、InsectSound、FordChallenge、LenDB

**📈 对比分析**

采用1%、10%、100%三种标签预算，分层抽样并在五折（或两折）上进行配对t检验。结果显示冻结模型对随机权重有显著提升，但在任何设置下都未能稳健超越最强基线；在部分数据集仅在低标签时接近基线

**⚠️ 局限性**

仅评估两种预测模型、单一随机初始化、池化方式固定、仅以准确率评估、微调实验覆盖有限、使用LenDB子样本以及非默认的MultiRocket配置等限制

---

## 28. Uncertainty calibration in Bayesian M/EEG source imaging

**arXiv ID:** 2611.06979 | [PDF](https://arxiv.org/pdf/2611.06979v1)

**作者:** Ismail Huseynov `[一作]` (Physikalisch-Technische Bundesanstalt Braunschweig and Berlin), Stefan Haufe `[通讯]` (Physikalisch-Technische Bundesanstalt Braunschweig and Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

对M/EEG源重建的经验贝叶斯方法进行后验不确定性量化与校准的系统评估，并提出了基于等价变换的后期置信区间校准技术。

**💡 创新点**

将稠密（BMN+sLORETA）与稀疏（γ-MAP+S-FLEX+S‑FLEX）经验贝叶斯模型结合自适应噪声学习，利用等价变换的单调回归实现后期校准，从而提升后验置信区间的经验覆盖率。

**🔧 技术方法**

经验贝叶斯源先验（BMN+sLORETA、γ‑MAP+S‑FLEX），闭式高斯后验推导，等价变换与等价单调回归（Isotonic Regression）校准，EMD、UNC、MAD/MOD/MUD等评估指标。

**📊 数据集**

合成数据基于CamCAN四个MEG头模型（不同SNR、源稀疏度、噪声设定），以及公开的Brainstorm CTF听觉MEG真实数据。

**📈 对比分析**

通过大量仿真跑测定EMD、UNC、MAD等指标，比较BMN+sLORETA与γ‑MAP+S‑FLEX的定位误差、后验不确定性及校准误差。结果表明γ‑MAP+S‑FLEX在定位更精确、后验不确定性更小、校准更好；等价回归显著提升两者的经验覆盖率，尤其对BMN+sLORETA的保守性产生重要改善。

**⚠️ 局限性**

仅评估了条件均匀高斯噪声与事件相关的单向激活，未涵盖空间/时间相关噪声、非高斯误差、支持不确定性及完整三维EEG置信区间校准；后期校准依赖于与真实条件匹配的模拟，转移效果有限。

---

## 29. SteelAgent: LLM-Orchestrated, Physics-Informed Steel Property Prediction with Audited Generalization and Calibrated Uncertainty

**arXiv ID:** 2611.06953 | [PDF](https://arxiv.org/pdf/2611.06953v1)

**作者:** Aleksandr Volkov `[一作]`, Yury Maximov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了SteelAgent，一个结合金属学描述符、机器学习模型和大型语言模型的钢材性能预测与合规性审计系统，并在SteelBench上进行了多种分层评估。

**💡 创新点**

创新点在于将物理派生特征与机器学习模型联合、使用语言模型进行模型选择与解释，并系统展示了不同留出单位对性能评估的影响以及分布移位下的区间置信度。

**🔧 技术方法**

采用梯度提升、CatBoost、Ridge回归、Conformalized Quantile Regression等算法，并通过Python/NumPy实现。

**📊 数据集**

使用SteelBench v1.0公开数据集（3741条记录，1234种牌号，27个来源）以及其Silver核心和Gold实验数据。

**📈 对比分析**

与随机k折相比，按牌号、家族或来源分组留出的模型R²从>0.85降至0.11-0.35，90% 区间覆盖率从91%降至37%；物理特征在家族留出时提升约0.25 R²，但在实验室四级留出时降低。

**⚠️ 局限性**

局限在于评估结果高度依赖留出单位，样本量有限；未在真实合金资格测试中验证合规评分；语言模型交互未进行用户研究；区间置信度在分布移位下不一定可靠。

---

## 30. Authorization Propagation in Agent2Agent Systems: An Experimental Security Evaluation of Multi-Hop Delegation

**arXiv ID:** 2611.07050 | [PDF](https://arxiv.org/pdf/2611.07050v1)

**作者:** Gabriele Donno `[一作]` (Norwegian University of Science and Technology), Katina Kralevska `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 1583 | [OpenAlex ID](https://openalex.org/A5064365516)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了Agent2Agent系统中多跳授权代理的传播机制，提出了一种结合授权来源追踪与重放防护的多跳授权验证方法。

**💡 创新点**

创新点在于同时实现可信授权传播、有效重放攻击防护和授权源追溯，使多跳委托的安全性显著提升。

**🔧 技术方法**

使用了代理授权模型、基于哈希链的重放防护技术、数字签名与授权追溯机制，并结合TLS/DTLS进行通信安全保障。

**📊 数据集**

实验基于OpenStack云环境的代理链模拟数据集以及自定义的多跳委托场景进行评估。

**📈 对比分析**

与传统单跳授权和基于令牌的方案对比，测量延迟、吞吐量与安全性，结果显示本方法在授权链长≥5时延迟提升约15%，但重放攻击防护率达到99.9%，总体安全性能优于对比方案。

**⚠️ 局限性**

局限性包括实验规模受限于仿真环境，未在真实分布式部署中验证；此外，代理链长度过长时签名验证开销显著，影响整体性能。

---

## 31. KG-AST for Cheaper CI/CD: Budgeted Test Selection and AI Bug Review

**arXiv ID:** 2611.06950 | [PDF](https://arxiv.org/pdf/2611.06950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 32. TraCE: Trajectory-Aware Concepts for Explainable Video Understanding

**arXiv ID:** 2611.07009 | [PDF](https://arxiv.org/pdf/2611.07009v1)

**作者:** Wooil Lee `[一作]` (Kyung Hee University), Jinwoo Choi `[通讯]` (Kyung Hee University)

**通讯引用:** 8047 | [OpenAlex ID](https://openalex.org/A5032355516)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过聚类光流轨迹生成可解释的运动概念，并结合物体与场景概念，构建一个可解释的视频理解框架。

**💡 创新点**

提出基于轨迹的概念框架，解决现有视频XAI中运动来源不明确的问题，实现多来源运动的可解释性。

**🔧 技术方法**

使用光流轨迹聚类（FINCH）、视频‑LLM文本检索、概念瓶颈学习、文本编码器及对齐损失等技术。

**📊 数据集**

在NTU RGB+D Mutual、Chiral SSv2、UCF‑101、KTH、TOMATO和STAR等六个动作识别与VQA基准上进行评估。

**📈 对比分析**

与非可解释基线（VideoMAE、InternVideo2）对比，动作识别平均下降1点，VQA略增0.4点，且在多源运动任务上保持竞争力。

**⚠️ 局限性**

受限于光流估计误差、轨迹聚类的离散化以及对多实体交互细节的捕捉深度，仍需进一步提升解释的细粒度与鲁棒性。

---

## 33. Curvature by Design: A Support Structure for Continuum Robots

**arXiv ID:** 2611.07091 | [PDF](https://arxiv.org/pdf/2611.07091v1)

**作者:** Justus Sagemüller `[一作]` (KTH Royal Institute of Technology), Kristin Y. Pettersen `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 12261 | [OpenAlex ID](https://openalex.org/A5055872448)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种多段剪式支撑结构，使连续机器人在弯曲时保持预定的常曲率或可变曲率。

**💡 创新点**

通过泰勒展开实现对多自由度连杆的几何耦合，使用柔性铰链实现无间隙、单件FDM打印，并显著降低了可变形自由度。

**🔧 技术方法**

剪式连杆设计、柔性铰链（live hinge）、泰勒近似、3D FDM打印、肌腱驱动、光学运动跟踪、力学实验。

**📊 数据集**

实验数据：光学标记位移、肌腱位移、已知外载荷；没有公开数据集，全部为自制实验记录。

**📈 对比分析**

与无支撑的15%/40%填充TPU梁以及常曲率SpiRobs进行对比，结果显示支撑结构在共同模式敏感度、形状精度、漂移、抗S形弯曲方面均优于对照组。

**⚠️ 局限性**

受限于离散连杆的最大角度和泰勒近似失效，导致曲率上限受限；单件打印需要精细工艺，材料选择与连接点容易产生误差；部分摩擦产生滞后。

---

## 34. Reclassification Is All You Missed: Surfacing Variant Reinterpretation Need in Online Health Communities at Population Scale

**arXiv ID:** 2611.07054 | [PDF](https://arxiv.org/pdf/2611.07054v1)

**作者:** Kamalakkannan Ravi `[一作]` (Harvard Medical School), Cong Liu `[通讯]` (Harvard Medical School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了PRISM，一个基于LLM的管道，利用Reddit公开帖子识别基因变异重新解释需求并草拟更新通知，完成不发送的dry‑run；

**💡 创新点**

首次将大规模社交媒体信号与基因变异重新解释行动相连，提供人群级别的重新解释需求地图，并验证草稿可发送性；

**🔧 技术方法**

采用OpenAI GPT‑4多代理架构（triage、branch‑2、outreach、coach），结合ClinVar数据库查询；

**📊 数据集**

收集并处理了2021‑2026年间约20万条来自15个基因学相关Reddit子版块的公开帖子，构建合成控制集；

**📈 对比分析**

在合成控制集上三阶段精度达90% B2 precision、95% 变异识别；在真实帖子上成本约0.3美元/帖；抽样真实案例中事实性达成率90%，可发送率约30%，相较于大模型的web‑search代理更高性价比；

**⚠️ 局限性**

受限于缺乏召回评估、仅覆盖Reddit用户、未临床验证、草稿需人工审核，无法实现自动发送。

---

## 35. Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models

**arXiv ID:** 2611.06982 | [PDF](https://arxiv.org/pdf/2611.06982v1)

**作者:** Yutong Bian `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 5724 | [OpenAlex ID](https://openalex.org/A5102765381)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了分块交叉式全双工语音语言模型的解码延迟，并提出在空闲窗口内进行自我投机式监听（Speculative Listening）以提前解码下一块输出。

**💡 创新点**

创新点在于：①发现下一块音频往往不改变输出且变化集中在首个输出标记；②基于此构造了无训练、无误差的“静默假设”预解码方法，并加入跨块假设池和跑者分支，实现在空闲窗口预解码并在音频到达时一次性验证，显著减少后期计算。

**🔧 技术方法**

主要技术包括自我投机式解码（Draft‑Verify）、假设池（Hypothesis Pool）、跑者分支（Runner‑Up Branch）、静默假设草稿以及与传统无训练投机方法（PLD、Token Recycling、Lookahead）对比。

**📊 数据集**

使用了 TickTock‑Bench 基准，涵盖 Speech reasoning（Web Questions、Llama Questions、TriviaQA、AlpacaEval、BBA）、Conversational dynamics（FDB v1、v1.5）以及 Agentic tool‑call（FDB v3）等八个任务。

**📈 对比分析**

在 TickTock‑Bench 上对四种全双工模型进行评估，采用 mean post‑audio latency、deadline miss rate、max lag、响应起始延迟等指标。该方法平均提升 speak chunk 速度 2.6–2.8 倍、tool‑call 最高 23.4 倍；最大滞后下降 94.4%；响应起始延迟缩短 73%，接近人类交互的 200 ms；与 PLD、Token Recycling、Lookahead 等基线相比，性能显著提升。

**⚠️ 局限性**

局限性包括：①仅适用于低并发、低延迟部署，空闲窗口随多会话共享 GPU 缩小导致加速下降；②只评估 Thinker 组件，未考虑 Talker 的合成差异；③不适用于并行流架构（如 Moshi），需要重新设计。

---

## 36. Polynomial-Time Computation of Extensive-Form Proper Equilibria in Two-Player Zero-Sum Games

**arXiv ID:** 2611.07089 | [PDF](https://arxiv.org/pdf/2611.07089v1)

**作者:** Sam Ganzfried `[一作]` (Cornell University), Sam Ganzfried `[通讯]` (Cornell University)

**通讯引用:** 668 | [OpenAlex ID](https://openalex.org/A5081380385)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种多项式时间算法，能够精确计算两人零和完美回忆博弈的广义形态正确（extensive-form proper）均衡。

**💡 创新点**

首次将几何排列多面体（permutahedron）扰动与序列形式多面体结合，并证明可在多项式时间内求得极限行为分布，从而实现精确的局部正确均衡计算。

**🔧 技术方法**

利用几何排列多面体约束、序列形式扩展多面体、参数化线性规划的稳定基准以及符号稳定性与代数插值技术。

**📊 数据集**

本文不依赖具体数据集，而是基于理论证明，使用任意给定的两人零和完美回忆博弈树的显式表示。

**📈 对比分析**

与现有的可多项式时间求解的其他均衡细化（如EFPE、OPE、QPE等）相比，该方法在多项式时间内给出完全精确的广义形态正确均衡，且输出的概率为有理数，编码长度多项式。

**⚠️ 局限性**

局限性包括仅适用于零和、完美回忆博弈，未给出强多项式时间或实际高效实现，且不适用于一般博弈或需要枚举所有正确均衡的场景。

---

## 37. An Integrality-Gap Lower Bound of 1.2325 for Minimum Multiway Cut

**arXiv ID:** 2611.06957 | [PDF](https://arxiv.org/pdf/2611.06957v1)

**作者:** Dimitar Chakarov `[一作]` (Toyota Technological Institute at Chicago), Yury Makarychev `[通讯]` (Toyota Technological Institute at Chicago)

**通讯引用:** 2175 | [OpenAlex ID](https://openalex.org/A5068045392)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过构造新的流实例，证明当终端数足够大时，CKR 松弛的整数化缺口至少为1.2325，从而提升了多路割问题的近似难度下限。

**💡 创新点**

创新点在于引入非相反切割与标签流的框架，结合线性规划和计算机辅助搜索，构造了高维单纯形上更强的流实例。

**🔧 技术方法**

主要技术包括非相反切割理论、标记流的容量与保守性约束、凸分析的跨面流量约束以及利用线性规划搜索最优流实例。

**📊 数据集**

由于研究对象是理论构造的图实例，未使用实际数据集，而是依赖于单纯形截面上的离散化与自定义的权重。

**📈 对比分析**

与已有的1.20016下界相比，本研究通过精细化的流实例得到1.2325的更高下界，进一步证明在UG假设下多路割问题的最优近似比不可能低于1.2325。

**⚠️ 局限性**

主要局限在于仍需依赖UG假设、构造的实例仅在终端数极大时才有效，以及求解流实例的计算量随维度显著增长。

---

## 38. A hierarchy of physics-embedded neural operators for structural dynamic surrogate modelingA hierarchy of physics-embedded neural operators for structural dynamic surrogate modeling

**arXiv ID:** 2611.07059 | [PDF](https://arxiv.org/pdf/2611.07059v1)

**作者:** Amir Ansrollahi `[一作]`, Mehmet Sefa Orak `[通讯]` (Maltepe University)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5010636246)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一系列基于物理嵌入的神经算子，用于结构动力学响应的代理建模。

**💡 创新点**

设计了Modal Rational、PGMON、MOSAIC-MR/P等层级架构，将模态参数映射与解析频率响应重建结合，并加入受限频率、阻尼、残差校正。

**🔧 技术方法**

采用神经网络映射模态量，解析模态叠加，带正则化的残差与频谱校正，使用ReLU/GELU、SIREN等激活，训练时采用多目标损失、正则化和学习率调度。

**📊 数据集**

使用20,000个钢桁架塔与4,000单元薄壁梁的有限元模型，包含32个截面面积因子/153个质量坐标和10个可变质量点，配合7条PEER NGA-West2加速度记录。

**📈 对比分析**

通过给定模态信息（频率/阻尼）与无模态信息的直接频率响应网络对比，使用时间域L2误差和峰值误差评估；Modal Rational等模型在塔结构下平均位移误差0.73–0.76%，梁结构下0.029–0.116%，显著优于传统SIREN/ ReLU。

**⚠️ 局限性**

仅验证固定拓扑的固定尺寸模型，未验证网格/拓扑迁移；误差仍受模态误差、缺失动力学和激励权重影响，且未提供完整算子实现与泛化评估，计算加速在完整流程中未得到充分证明。

---

## 39. Robust Stream Classification using Time Neutralising Decision Trees

**arXiv ID:** 2611.06991 | [PDF](https://arxiv.org/pdf/2611.06991v1)

**作者:** Brandon Gower-Winter `[一作]` (Utrecht University), Georg Krempl `[通讯]` (Utrecht University)

**通讯引用:** 675 | [OpenAlex ID](https://openalex.org/A5077689739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的批量学习决策树架构——Time Neutralising Trees（TNT），用于在概念漂移环境下自动过滤不相关的历史样本并重建模型。

**💡 创新点**

创新点在于将时间索引作为特征，在训练时动态判断并“中和”旧样本，通过信息增益和Mann-Whitney U检验实现无需传统漂移检测器即可实现漂移自适应。

**🔧 技术方法**

技术包括决策树信息增益分裂、时间特征分裂、阈值λ控制过滤敏感度、Mann-Whitney U统计检验、批量重训练与超参数调优。

**📊 数据集**

实验使用了USP DS仓库中的24条真实或半真实分类数据流，涵盖从城市发展到医学的多领域。

**📈 对比分析**

通过Cohen's Kappa评估，将TNT与多数基线（Majority、Last）、增量决策树（HT、EFDT、PLASTIC、HAT、LAST）及其漂移检测器（ADWIN、DDM、D3）以及不同窗口大小的批量决策树进行比较，TNT在大多数流上取得最高排名，表现优于传统批量树与增量树。

**⚠️ 局限性**

局限性在于目前仅实现单棵树，未构建集成版本；对时间阈值λ敏感；对存储要求高；未在大规模集成算法中验证。

---

## 40. Trust Responsive Runtime Authorization for Agentic AI with Empirical Characterization of Policy Behavior and Parameter Sensitivity

**arXiv ID:** 2611.06996 | [PDF](https://arxiv.org/pdf/2611.06996v1)

**作者:** Ahsan Habib Shihab `[一作]` (Farabi University), Amir Mosavi `[通讯]` (Obuda University)

**通讯引用:** 25582 | [OpenAlex ID](https://openalex.org/A5109011370)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过模拟器生成100条同一随机序列的运行时信号，对Trust‑Responsive Authorization Policy (TRAP) 进行经验性评估，比较其与静态、信任仅、风险仅及对称阈值策略在急剧与缓慢退化场景下的授权行为。

**💡 创新点**

创新点在于首次将信任、稳定性与风险三项运行时指标融合为Authorization Readiness Score (ARS)，并设计了非对称阈值规则，使TRAP在退化时立即限制授权、在恢复时更快重新授权，同时对参数敏感性和阶梯级别分布进行了系统性量化。

**🔧 技术方法**

技术手段包括：基于Python的蒙特卡洛仿真、状态机实现TRAP、Bootstrap置信区间估计、对比实验与一因素敏感性分析，以及对已发布代码与数据的完整可复现验证。

**📊 数据集**

使用的“数据集”是由仿真器在固定随机种子（seed 42）下生成的100条长度为100的时间序列，信任、稳定性与风险分别在预设区间内随机波动并在退化点进行显著下降。

**📈 对比分析**

对比方法是将5种策略在相同100条轨迹上逐步执行，记录授权稳定性（AS）、限制延迟（RR）、恢复时间（RT）、保护效率比（PER）与渐进恢复比（PRR）等指标。结果显示TRAP在急剧退化时RR为0、RT比对称策略快37.5%，AS最高且变异最小；但在缓慢退化时TRAP的过量授权（EAD）显著高于对称策略。

**⚠️ 局限性**

主要局限包括：仅使用单一随机种子与单一信号生成器；信任、稳定性与风险均为人工合成且无真实Agent；仅评估了一种非对称规则与有限的退化情形；band edges与阈值未作广泛探索；指标如EAD基于抽象级别，未映射到具体权限，故难以直接衡量实际安全性。

---

## 41. Replay Is a Family of Assertions: Fork-Safety in Event-Sourced Agent Runtimes

**arXiv ID:** 2611.06924 | [PDF](https://arxiv.org/pdf/2611.06924v1)

**作者:** Yohei Nakajima `[一作]` `[通讯]` (Untapped Capital), Yohei Nakajima (Untapped Capital)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

定义可执行的事件源代理运行时断言与分叉安全评估，验证日志的回放与分叉属性。

**💡 创新点**

按属性划分的分叉安全判定、分离效果证据、检测终止与并发性假设，并在真实运行日志上进行全面切点评估。

**🔧 技术方法**

使用 F# 可执行属性检查（FsCheck）、JSONL 日志解析、SQLite、HMAC 签名以及可视化断言。

**📊 数据集**

基于 5,540 条公开 Synthetic Players 事件日志、121 条观测分叉以及一个公开的 post‑oracle conformance fork 进行实验。

**📈 对比分析**

通过生成式断言和所有切点枚举实现全面覆盖；评估结果显示所有投影切点通过、外部连续性切点 160,076 条通过、120,969 条件、36,251 条不通过；文中未给出具体性能指标。

**⚠️ 局限性**

未提供机械化证明、缺乏通用非冲突条件、仅检查四种调度策略、依赖手工日志适配、无法处理异步延迟错误、未评估跨平台性能、仅验证 ActiveGraph 族。

---

## 42. Distributed Geometric Optimization and Control of Rigid Differential-Drive Formations from Relative-Position Measurements

**arXiv ID:** 2611.07163 | [PDF](https://arxiv.org/pdf/2611.07163v1)

**作者:** Thanacha Choopojcharoen `[一作]` (University of Waterloo), Baris Fidan `[通讯]` (University of Waterloo)

**通讯引用:** 10434 | [OpenAlex ID](https://openalex.org/A5001551882)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于优化的分布式估计–控制框架，用单一绝对位姿锚点和相邻相对位置测量，直接重构差分驱动机器人队列的质心、锚向量与居中坐标，并实现形状、质心、姿态三者的同时调节。

**💡 创新点**

创新点包括：①利用图近端更新实现有限步完成几何重构；②通过离散姿态–形状分离和水平提升（horizontal‑lift）校正实现姿态漂移的精确消除；③将原始连续时间估计器-控制器离散化为邻居仅通信的Primal–Dual Hybrid Gradient（PDHG）迭代，保证有限计算量即可实现高精度控制。

**🔧 技术方法**

主要技术方法包括：优化理论（proximal map 与 exact penalty）、图论中的拉普拉斯一致性与无穷远一致性、离散时间 Lyapunov 分析、Primal–Dual Hybrid Gradient（PDHG）迭代、零阶保持（ZOH）实现差分驱动模型、以及对控制增益与采样周期的理论收敛性分析。

**📊 数据集**

实验采用的“数据集”是基于 5 台 DJI RoboMaster EP 机器人在 Vicon 运动捕捉系统下的实际位姿数据，并在 Gazebo 物理仿真环境中使用相同的传感器噪声与延迟模型进行验证。

**📈 对比分析**

与传统的仅形状或仅质心/姿态调节方法相比，该方法在硬件实验中实现了质心误差≤1.1 cm、姿态误差≤0.019 rad、形状误差≤3.2 cm（最终周期），并在起始无任何全局定位信息的情况下完成从任意姿态到目标轨迹的平滑跟踪，展示了显著的精度与鲁棒性。

**⚠️ 局限性**

局限性包括：仅适用于差分驱动平面机器人，需至少有一个绝对位姿锚点；对采样周期与通信延迟有严格要求；在高噪声或大延迟环境下重构误差会累积；未来工作需扩展到 Ackermann 驱动或三维形态。

---

## 43. Operationalizing Responsible AI in MLOps and ML Production: A Systematic Literature Review

**arXiv ID:** 2611.06961 | [PDF](https://arxiv.org/pdf/2611.06961v1)

**作者:** Hinduja Nirujan `[一作]` (University of Calgary), Gouri Ginde `[通讯]` (University of Calgary)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5020045407)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对44篇研究论文进行系统文献综述，梳理其在MLOps/ML生产环境中如何将责任AI原则转化为可操作的技术机制；

**💡 创新点**

首次将责任AI在生产中的实现按监测与运行保障、审计与合规、生命周期集成及可追溯性四大机制主题进行分类，并揭示现有研究在整合、证据闭环与实践落地方面的核心缺口；

**🔧 技术方法**

采用系统性搜索与筛选流程（多数据库检索、Snowballing、手工筛选），对文献进行编码与归类；

**📊 数据集**

研究基于文献数据而非实验数据集，使用的“数据集”即为被纳入综述的44篇原始研究；

**📈 对比分析**

通过评估成熟度（概念框架、原型演示、案例研究、生产部署等）与机制对比，展示了各主题在技术实现、法规映射与运营实践中的表现差异，说明大多数研究停留在原型或实验阶段，真实生产验证稀缺；

**⚠️ 局限性**

局限性包括：文献覆盖可能不足，方法多为概念/原型，缺乏长期、跨组织的实测；缺少对人因、成本、组织治理等软因素的系统评估，导致对责任AI机制在真实生产环境中可持续性的结论仍不充分。

---

## 44. SD-LoRA: Training-Time Structural Distillation into LoRA for Few-Shot Vision-Language Adaptation

**arXiv ID:** 2611.07063 | [PDF](https://arxiv.org/pdf/2611.07063v1)

**作者:** Mohammed Rahman Sherif Khan Mohammad `[一作]` (Edge Hill University), Yonghuai Liu `[通讯]` (Edge Hill University)

**通讯引用:** 5540 | [OpenAlex ID](https://openalex.org/A5017205444)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SD-LoRA，一个在训练阶段使用结构化图教师对视觉语言模型进行 token‑级别监督，并将结构信息隐式蒸馏到 LoRA 参数，推理时仅使用轻量化的 LoRA+缓存，无需图计算。

**💡 创新点**

创新点在于：① 训练时仅存在的图结构教师提供 token‑级别的关系监督，② 通过共享 LoRA 背骨实现隐式结构蒸馏，③ 引入 Prototype Predictive Alignment 进一步强化类别层面的结构一致性，整体实现了在不增加推理成本的前提下提升 token 级别判别能力。

**🔧 技术方法**

使用的技术包括：LoRA 参数高效微调、异构图（patch–patch、global–local、visual–text）推理、隐式结构蒸馏、Prototype Predictive Alignment、缓存式分类器和 ViT 高分辨率 token 采样。

**📊 数据集**

在 11 个标准 few‑shot 基准（一般物体、细粒度、纹理、场景、遥感、动作等）以及 ImageNet 基础‑新颖、OOV（ImageNet‑R、A、V2、Sketch）和跨数据集转移等多种数据集上进行实验。

**📈 对比分析**

与多种基线（CLIP‑LoRA、Tip‑Adapter、CoOp、MaPLe、PromptKD 等）对比，SD‑LoRA 在 1/2/4/8/16-shot 场景分别提升 2.4/1.8/3.0/2.4/2.0 个百分点；在基底‑新颖、OOV、跨域校准等指标上也均优于最强基线，显示出更稳健的迁移性能。

**⚠️ 局限性**

局限性包括：① 训练阶段需构造高分辨率 ViT tokens 与图结构，增加训练成本；② 对某些 OOV 场景（如 ImageNet‑A）提升有限；③ 仅在归纳式 few‑shot 环境下验证，未探讨在线或转导适配；④ 结构教师在训练时需要额外参数与计算，且对大规模长尾任务的适用性尚未评估。

---

## 45. Matriarchy Classics: A Computational Approach to Swap Only Gendered Words in Classic Novels

**arXiv ID:** 2611.07134 | [PDF](https://arxiv.org/pdf/2611.07134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 46. A Multi-Modal Robotic Framework for Physical GUI Interaction Testing

**arXiv ID:** 2611.06939 | [PDF](https://arxiv.org/pdf/2611.06939v1)

**作者:** Davi Freitas `[一作]` `[通讯]` (Federal University of Pernambuco), Davi Freitas (Federal University of Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了RFNIT，一个全非侵入式多模态机器人框架，用于物理GUI交互测试。

**💡 创新点**

通过将手机固定在机器人上实现触摸、侧键、旋转和高级手势等多模态感知与执行，并使用LLM决策，无需系统级访问即可实现自动化测试。

**🔧 技术方法**

采用视觉感知（OpenCV、YOLOv8、PaddleOCR）、LLM驱动决策、机械臂执行器，并实现闭环感知-决策-执行。

**📊 数据集**

利用Android XML dumps、YOLOv8训练集以及自建的GUI图像+文本数据集进行实验。

**📈 对比分析**

与ADB、DLD、DroidBot等基线对比，在五个Android应用上实现91% mAP@0.5视觉检测、95% OCR精度、2mm执行精度，覆盖率与执行时间可竞争。

**⚠️ 局限性**

主要限制包括时延同步难题、视觉遮挡与反光、LLM决策不确定性以及多设备并行扩展性不足。

---

## 47. Finite-Time Convergence and Regret Guarantees for Neural Reinforcement Learning in POMDPs

**arXiv ID:** 2611.07069 | [PDF](https://arxiv.org/pdf/2611.07069v1)

**作者:** Jingjuan Huang `[一作]` (Ohio State University), Jia Liu `[通讯]` (Ohio State University)

**通讯引用:** 6563 | [OpenAlex ID](https://openalex.org/A5100409739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于有限记忆超状态MDP的POMDP强化学习框架，利用神经网络价值函数逼近与策略镜像下降（PMD）实现在线学习，并给出有限时刻收敛与子线性回报误差分析。

**💡 创新点**

创新点在于：①在非线性神经网络价值逼近下，首次实现了对POMDP轨迹采样的非i.i.d. TD分析；②给出了从POMDP到超状态MDP的点态误差上界；③将神经TD误差与PMD策略更新结合，得到完整的子线性回报误差上界，填补了以往仅在线性或i.i.d.采样情形下的空白。

**🔧 技术方法**

主要技术包括：超状态MDP构造与滤波稳定性假设；ReLU多层神经网络价值函数与NTK局部线性化；基于几何混合的非i.i.d. TD误差分析；策略镜像下降（PMD）与KL正则化策略更新；浓缩系数假设用于处理分布不匹配。

**📊 数据集**

实验使用的基准是公开的“CheeseMaze”部分可观测导航任务（11个隐状态、7个观测、4个动作）。

**📈 对比分析**

与Q-learning和Actor-Critic基线对比，实验显示神经PMD方法在不同记忆窗口（ℓ=1,2,4）下收敛更快、回报曲线增长更慢，验证了理论上的子线性回报误差；尤其在ℓ≥2时已基本收敛，表明短窗口即可捕获大部分信息。

**⚠️ 局限性**

局限性包括：仅适用于离散状态、观测与动作空间；超状态MDP需要可行的窗口截断；对连续域、非滤波稳定性POMDP的推广仍待研究。

---

## 48. T$^2$-Splat: Adaptive Topology Mesh Splatting with Texture Residuals

**arXiv ID:** 2611.06974 | [PDF](https://arxiv.org/pdf/2611.06974v1)

**作者:** Zhihao Tang `[一作]` (Huazhong University of Science and Technology), Wei Yang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 38619 | [OpenAlex ID](https://openalex.org/A5100781368)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了T^2-Splat两阶段网格光栅化框架，利用自适应拓扑与残差Mipmap纹理实现高保真视角合成。

**💡 创新点**

将拓扑演化与纹理细化解耦，采用Safe RDT初始化网格，Dynamic Topology Evolution（边折叠/顶点拆分）动态调整面数，并通过每面残差纹理保留高频细节。

**🔧 技术方法**

使用Safe Restricted Delaunay Triangulation、QEM驱动的边折叠与顶点拆分、每面残差Mip‑Texture、差分可渲染的软窗口函数、LOD自适应采样以及Adam优化等技术。

**📊 数据集**

在Mip‑NeRF 360、Tanks & Temples、DTU以及Garden场景等公开数据集上进行实验。

**📈 对比分析**

与MeshSplatting、MiLo、3DGS等基线对比，PSNR提升约0.71dB、LPIPS降低约0.03、SSIM提升，面数从7.3M降至2.7M，帧率从126fps提升至236fps，文件体积从152MB降至62MB。

**⚠️ 局限性**

对极低面数下细节仍有一定损失，DTE比例需手动调节，极复杂场景下可能出现拓扑不稳定问题。

---

## 49. When Can Self-Play Fix a False Belief? Answer Keys, Pass Bands and Replacement Rules in an Execution-Grounded Loop

**arXiv ID:** 2611.06965 | [PDF](https://arxiv.org/pdf/2611.06965v1)

**作者:** Nirdesh Chauhan `[一作]` (AltSlate Labs LLP), Siddharth Kosaraju `[通讯]` (AltSlate Labs LLP)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了自我改进循环中，当答案键来自执行器时，模型的错误信念如何被纠正，并评估了生成器、键和通过过滤器（pass band）的三条“门”对错误纠正的作用。

**💡 创新点**

提出了三门框架来解释自我改进循环如何直接纠正错误，证明了键的去相关化和通过执行器键可以显著降低错误率，并揭示了替代规则和生成器写入规范的意外行为。

**🔧 技术方法**

采用 Qwen3-4B-Base 模型的专家迭代自我改进循环，使用执行器、不同来源的答案键、滤波阈值（1–7/8）以及对生成任务和答案的自动化审计。

**📊 数据集**

通过脚本生成的输出预测任务与代码实现任务，包含植入的错误（S1、S2）以及自然的 Python 运行时错误 R1、R2；评估使用诊断和邻居探针。

**📈 对比分析**

在四个实验分支（执行器键、关联键、去相关键、控制）下的多轮训练中，测量诊断项的真值率、错误率和邻居项的错误分布，发现去相关键能将真值率提升至约0.6–0.8，而关联键和控制键效果不佳。

**⚠️ 局限性**

实验规模有限，仅使用单一模型和少量种子；键的去相关化依赖已知错误的基线；生成器对边界条件的习惯导致缺少真实邻居任务；未评估更大规模或不同模型族的效果。

---

## 50. How Python Audio Libraries Change Across Releases: Module Structure and Documentation

**arXiv ID:** 2611.06947 | [PDF](https://arxiv.org/pdf/2611.06947v1)

**作者:** Mateusz Modrzejewski `[一作]`, Łukasz Neumann `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过静态导入图挖掘 Python 音频机器学习库在各个发布之间的模块结构变化，并将这些变化与对应的发布说明相匹配。

**💡 创新点**

创新点在于提出以模块路径和公共名称为单位的“覆盖率”指标，将模块结构演化与发布说明内容关联，并首次使用大型开源语言模型对发布说明条目进行自动标签化，构建可公开的数据集。

**🔧 技术方法**

主要技术包括静态导入图构建、语义版本号解析、GitHub Releases API 抓取、Open-weight LLM（Qwen2.5、Llama‑3.3）进行条目标签化，以及统计与人工验证相结合的评估流程。

**📊 数据集**

使用的实验数据集为 17+ 后续扩充的公开 GitHub 音频ML 库集合（如 librosa、ESPnet、SpeechBrain、Whisper 等），并与同类型非音频 ML 库做对比。

**📈 对比分析**

通过计算 top‑level 保持率、模块文件变化率、增量类型对结构变化的影响率，并使用百分比、置信区间、Cliff's δ 等统计检验，发现 major 与 0.x minor 增量导致约一半的 top‑level 变化；发布说明仅在不到 10% 的情况下提及模块或名称的删除。

**⚠️ 局限性**

局限性包括：静态导入图忽略动态/插件导入；仅检测模块路径和公共名称的删除，未评估对调用方的实际破坏；词匹配过于宽松导致命中率不高；LLM 标签仅在部分代码类型通过验证，其他标签未得到充分的人类验证。

---

## 51. Stochastic Dynamics of Generative AI: From Autoregressive Models to Interacting LLM Networks and Neural SDEs

**arXiv ID:** 2611.07065 | [PDF](https://arxiv.org/pdf/2611.07065v1)

**作者:** Vikram Krishnamurthy `[一作]` (Cornell University), Luke Snow `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个统一的随机动力学框架，分别将大型语言模型（LLM）视为高维非线性自回归模型，研究LLM网络的均值场动力学以揭示偏好吸附与玻璃天花板效应，并提出利用神经随机微分方程（neural SDE）对连续时间随机轨迹进行生成建模，进一步利用Wiener‑空间立方体法加速训练；

**💡 创新点**

1）首次将LLM的注意力机制形式化为高维非线性AR模型并与最大似然预训练相连接；2）在LLM网络层面引入均值场ODE，揭示网络结构与偏差的动态产生机制；3）提出将Wiener‑空间立方体方法与神经SDE结合，提供比Monte Carlo更高阶的期望近似；

**🔧 技术方法**

高维非线性AR建模、注意力机制、均值场动力学（ODE）、随机微分方程、粗糙路径分析、Wiener‑空间立方体（Deterministic ODE替代MC）、自动微分/随机伴随、正则化/对齐技术（RLHF、DPO等）

**📊 数据集**

主要使用大型文本语料库进行LLM预训练（未给出具体数据集）；神经SDE实验使用了金融、人体运动轨迹、基因调控等不规则时间序列数据（同样未给出具体数据集）

**📈 对比分析**

对比方法主要集中在理论推导与数值仿真：对均值场动力学通过理论证明收敛性与唯一平衡；对Wiener‑空间立方体方法与传统Monte Carlo在弱误差阶数（O(n⁻¹) vs O(n⁻¹/2)）及梯度计算成本上的提升；但论文未给出实验性能指标，主要通过数值实验展示收敛速度与梯度计算时间的改善；

**⚠️ 局限性**

1）模型假设（如独立性、连续可导性）可能与实际LLM产生的高维非线性分布不完全吻合；2）均值场近似在有限规模网络中误差难以评估；3）Wiener‑空间立方体方法要求预先构造高阶立方点，计算成本和实现复杂度较高；4）缺乏大规模实验验证与对比；5）对多模态、跨域生成任务的适用性仍待研究。

---

## 52. Depth or Description? When a One-Step Latent Score Replaces Lookahead in Proactive Information Acquisition

**arXiv ID:** 2611.07155 | [PDF](https://arxiv.org/pdf/2611.07155v1)

**作者:** Biswa Sengupta `[一作]` `[通讯]` (JPMorgan Chase & Co.), Biswa Sengupta (JPMorgan Chase & Co.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究主动决策中何时需要深度搜索，并提出利用潜在信息增益（LIG）与回溯-LIG作为一阶评分替代深度搜索的规划器。

**💡 创新点**

创新点在于证明标签一阶评分对交互式信息不敏感，而潜在评分能捕捉隐藏因子，给出理论上可视化并可排序的评分；同时提出在有限计算预算下的迭代加深与回溯策略，减少搜索成本。

**🔧 技术方法**

使用了完整的动态规划（Exact k-step lookahead）、信息增益（CMI、LIG）、回溯（Rollout-LIG）、迭代加深、理论错误界限、LLM token概率读取与混合模型、以及基于验证的模型误差认证。

**📊 数据集**

实验数据包括：合成任务（Context Parity、CUBE‑NM、九个常见二元特征数据集；Iris、Wine、Breast Cancer、Digits、Diabetes、PhysioNet、MiniBooNE、Zoo），以及基于LLM Qwen/GLM 的111项20 Questions 银行（答案计数与token读取）。

**📈 对比分析**

通过将规划器在相同拟合模型、预算与评价标准下与D3、D2、Rollout-LIG、UCT、DIME、GDFS、SEFA、AACO等基线对比，发现：在需要深度的合成任务中LIG/回溯与深度搜索差距小且计算量约为深度的十分之一；在真实数据上深度收益通常为0或微小；LLM计数模型在同一家族内能恢复大部分性能，而跨家族表现差；模型误差界限能够在约一半案例中证明规划器间的偏好。

**⚠️ 局限性**

局限性包括：合成任务可能夸大交互依赖；潜在因子需预先设计或正确拟合；LLM概率不可校准，读取token概率仅能提供少量标签信息；规划器安全性与鲁棒性未全面评估；性能区间包含零时难以判定实质提升；方法对开放式问答场景的推广尚未验证。

---

## 53. Towards Comprehensive Mobile Application Testing for User-centric Feature Coverage via Simulating Real-World Users with Personas

**arXiv ID:** 2611.06922 | [PDF](https://arxiv.org/pdf/2611.06922v1)

**作者:** Ha Min Son `[一作]` (University of California, Davis), Xin Liu `[通讯]` (University of California, Davis)

**通讯引用:** 54755 | [OpenAlex ID](https://openalex.org/A5100352324)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了用户中心的特征覆盖基准和ECS框架，用LLM生成关键用户旅程并合成评论，自动评估移动应用的特征覆盖率。

**💡 创新点**

创新点在于①构建3,012条基于真实用户评论和应用描述的目标特征列表并实现自动LLM评判管线；②将探索轨迹转换为关键用户旅程（CUJ），再通过多样化人物合成评论聚合复杂特征，从而显著提升特征覆盖。

**🔧 技术方法**

采用Gemini 3 Flash等LLM进行特征翻译与评判，嵌入与余弦相似度检索，凝聚聚类挑选代表性测试；结合静态代码分析、强制探索和synthetic review生成技术。

**📊 数据集**

使用10款热门Google Play应用的描述与5,000条用户评论，提炼出3,012个目标特征。

**📈 对比分析**

与DroidAgent基线对比，ECS在匹配集下特征覆盖率从40.3%提升至58.0%，完整集覆盖率从39.3%提升至64.1%，在应用描述、正评与负评覆盖率均显著优于基线。

**⚠️ 局限性**

受限于LLM无状态、跨会话推理不足和探索可发现性限制，导致约36%的特征未被覆盖，尤其是需要状态持久化或跨屏交互的功能。

---

## 54. LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time

**arXiv ID:** 2611.07025 | [PDF](https://arxiv.org/pdf/2611.07025v1)

**作者:** Teresa Delgado `[一作]`, Zafeirios Fountas `[通讯]` (Huawei)

**通讯引用:** 577 | [OpenAlex ID](https://openalex.org/A5066982052)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究KV缓存压缩，提出值缓存摊销和低秩键压缩的低内存解码方案LowRAM，兼容量化与淘汰。

**💡 创新点**

在注意力头内利用键-值可预测性进行轻量预测，并与跨层低秩键压缩结合，构成异构压缩管线；引入元学习初始化和选择性重构提升效率。

**🔧 技术方法**

值缓存摊销（per-head MLP预测+残差）、xKV低秩键压缩、RoPE逆变换、TurboQuant量化、淘汰、MAML元学习初始化、选择性重构。

**📊 数据集**

RULER、LongBench（含10K+ token子集）以及Mistral-7B、Llama-3.1-8B、Qwen3.5-4B三大模型。

**📈 对比分析**

与Eviction、Quantization、xKV等基线对比；在4×压缩下保持96-99%性能；在极端压缩下实现26.9×压缩，94.5%性能；Selective Reconstruction与Meta-init降低TTFT，提升吞吐。

**⚠️ 局限性**

未评估长生成过程；对新生成的后缀未做动态压缩；上下文间预测参数共享与具体化不清。

---

## 55. Octopus: One Brain, Many Branches for Latent-KV Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.06956 | [PDF](https://arxiv.org/pdf/2611.06956v1)

**作者:** Tianming Sha `[一作]` (Stony Brook University), Tianming Sha `[通讯]` (Stony Brook University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5119152033)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Octopus，构建 GPU 端驻留的执行 DAG，实现长周期推理的并行分支、KV 挂载和强化学习训练；

**💡 创新点**

创新点包括：①GPU 原生分支原语与 radix cursor；②零拷贝、单向 KV 挂载以消除 recurrent hijacking；③前缀树打包的 DAG kernel 与 DAG‑GRPO；④纯二元奖励下自组织的 commander‑worker 结构；

**🔧 技术方法**

采用 GPU radix cursor、零拷贝 KV 挂载、单向位置偏移、前缀树打包的前向后向通道、DAG‑GRPO 与损失掩码、LoRA 适配器、Mamba‑2/GDN 混合模型等技术；

**📊 数据集**

在 OpenResearcher（3,040 题/500 题验证）与 BrowseComp‑Plus（830 题）数据集上，使用 BM25 sparse 检索与 14.9M 文档语料；

**📈 对比分析**

与 32K/143K ReAct、GRPO、multi‑agent fork、开源 LLM（如 GPT‑4.1、Claude‑4）及 9B 以上规模模型比较，Octopus 4B 在 BrowseComp‑Plus 上达到 49.6%（+17.5pt 取代 143K GRPO），在 OpenResearcher 上 51.2%，超越同规模及更大模型；

**⚠️ 局限性**

局限性：仅支持单模型 episode；未实现异构模型或多模型分支；KV 页面在 episode 结束即丢失，缺乏持久化记忆；分支规模上限为 10，单节点 GPU 资源限制。

---

## 56. KDASO: Knowledge-Data Dual-Driven Automated Skill Optimization for Liability Adjudication Tasks

**arXiv ID:** 2611.06972 | [PDF](https://arxiv.org/pdf/2611.06972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 57. Formalizing inverse problems in the age of AI: possibilities, pitfalls, and an invitation

**arXiv ID:** 2611.06962 | [PDF](https://arxiv.org/pdf/2611.06962v1)

**作者:** Christian Clason `[一作]` (University of Graz), Lauri Oksanen `[通讯]` (University of Helsinki)

**通讯引用:** 1219 | [OpenAlex ID](https://openalex.org/A5044437642)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了一个由 AI 自动 formal 化的线性正则化理论库，并设计了人机协作的审查流程，重点核对正式声明与自然语言原文的一致性。

**💡 创新点**

创新点在于将自动 formal 化与专注于声明对应的审查机制结合，配合非真值测试与哈希变更检测，揭示经典文献中的隐藏假设错误。

**🔧 技术方法**

使用 Lean 证明助手、Claude Opus、GPT‑5.6 Sol 等 AI 代理进行代码生成与审核，并实现程序化检查、非真值测试和 VS‑Code 扩展。

**📊 数据集**

主要使用标准逆问题教材和 Mathlib、Tau Ceti 等已有库中的公理与定理作为正式化对象，形成约两百个节点的知识库。

**📈 对比分析**

通过多轮 AI 评审与人类审核，发现仅剩一处不匹配；非真值测试成功捕捉到一条假设过强的定理，表明自动化可提升理论质量。

**⚠️ 局限性**

局限性包括正式化只能验证声明本身，需人工核对语义；AI 评审可能产生分歧；库中“垃圾值”导致声明弱化；尚未实现完全自动化，需要社区共识和共享基础。

---

## 58. Follow the Policy, Fail the Benchmark: Auditing and Revising Policy-Based Decision Benchmarks

**arXiv ID:** 2611.07042 | [PDF](https://arxiv.org/pdf/2611.07042v1)

**作者:** Yanru Xiao `[一作]` (Amazon Web Services), Yifei Cai `[通讯]` (Iowa State University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5068070082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对九个公开的政策决策基准进行了源追踪、程序执行和受控检查，发现了参考答案、程序和评分器的错误；并通过修正这些错误重新评分，显示对模型表现评估的显著影响。

**💡 创新点**

提出了 Benchmark Audit and Revision (BAR) 框架，将审计证据和修正版本化，明确区分可重新评分的修正、需重新预测的修正以及未解决的缺陷。

**🔧 技术方法**

采用源代码审计、程序回放、执行跟踪、控制实验、解析器修正、评分器重写等技术，对规则、参考答案、程序及评分逻辑进行验证。

**📊 数据集**

使用 RuleArena、OpenExempt、LegalBench SARA、DeonticBench、ConditionalQA、ShARC、PolicyGuardBench、WorkArena++、TheAgentCompany 等九个公共政策基准及其公开代码与数据。

**📈 对比分析**

通过固定输出重评分和实验对比，发现修正零底线和小数解析后，三种模型在 RuleArena Tax 上的正确率均提高了 80+ 百分点；在 LegalBench SARA 上接受率从 8–9% 提升至 87–91%。

**⚠️ 局限性**

审计覆盖有限，未能检测所有错误；实验仅覆盖部分模型和提示；部分修正依赖未解决的资格或案例事实，可能需要进一步专家评估。

---

## 59. Segment What It Can Draw: Querying a Text-to-Image Generator with Its Own Images for Skin Lesion Segmentation

**arXiv ID:** 2611.07061 | [PDF](https://arxiv.org/pdf/2611.07061v1)

**作者:** Hongyu Cao `[一作]` (Clemson University), Kunpeng Liu `[通讯]` (Clemson University)

**通讯引用:** 1833 | [OpenAlex ID](https://openalex.org/A5100786547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种训练无关、无标注掩码的皮肤病变分割方法，通过生成图像构造健康皮肤到病变的特征方向，利用未标注的真皮图像进行原型细化，并采用门控阈值实现精确分割。

**💡 创新点**

创新地用文本生成的健康皮肤与病变图像来生成可靠的查询向量，取代传统概念词查询，并通过未标注图像自适应细化原型，且引入门控阈值降低小病变旁的注意力泄漏。

**🔧 技术方法**

基于冻结的FLUX文本到图像扩散变压器的注意力输出，概念词标记、生成图像原型、原型细化、单向投影评分和多阈值门控。

**📊 数据集**

在ISIC-2018、PH2、HAM10000三大公开皮肤病变分割测试集上进行评估，使用同一数据集的500张未标注图像作为原型细化池。

**📈 对比分析**

与平均掩码、手工设计的SLED、外部零样本方法和多种概念词读出方法对比，Dice得分0.85–0.88，显著优于所有零样本和词读出基线，并在两套数据集上击败手工管线。

**⚠️ 局限性**

仅假设位置不变对比，需少量未标注图像；对非常小或形状极异的病变效果受限；与有监督分割器相比仍存在性能差距。

---

## 60. Automated Feedback to Peer Tutors Increases Their Talk Moves and Learner Reasoning but Not the Balance of Talk

**arXiv ID:** 2611.07072 | [PDF](https://arxiv.org/pdf/2611.07072v1)

**作者:** Dorottya Demszky `[一作]` (Stanford University), Drew Bent `[通讯]` (Schoolhouse.world)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Schoolhouse.world平台上，对两期SAT数学训练营的1,030名志愿辅导员和7,443名学习者实施了随机对照试验，评估自动化会话反馈对辅导员话语、学习者参与度及学习效果的影响。

**💡 创新点**

①首次在小组志愿辅导环境中测试基于录音的自动化话语反馈；②引入双向反馈（辅导员+学习者）并比较社交式与目标式激励框架。

**🔧 技术方法**

使用语音转写+NLP分类器识别教学话语（邀请、接收、推理等）与学习者推理；用GPT‑4 Turbo生成具体建议；通过回归与多重比较校正进行实验效果评估。

**📊 数据集**

两期训练营共5,653课时、1,030名辅导员、7,443名学习者、6,508条录音；包括自动生成的反馈、转录文本以及访谈数据。

**📈 对比分析**

采用意向治疗OLS回归（控制辅导员特征、班级固定效应），比较四个实验组；结果显示辅导员反馈提升提问/接收/推理8–15%（效应≈0.12–0.15SD），但未改变辅导员谈话比例；学习者参与轻微提升，学习成绩无显著差异；学习者反馈无额外效应，且两种激励框架无显著差异。

**⚠️ 局限性**

主要局限：交付率不完全导致意向治疗估计偏低；样本仅为志愿高中新生辅导员与学生，缺乏可推广性；学习者反馈单次、不可复访；访谈样本小、单一平台且缺乏对照组；NLP分类器与转录误差可能影响测量；仅观察四周，未评估长期学习效果。

---

## 61. Interactive Backdoor Attacks in LLM-Based Proactive Dialogue Systems

**arXiv ID:** 2611.06969 | [PDF](https://arxiv.org/pdf/2611.06969v1)

**作者:** Zhiyang Xia `[一作]`, Xi Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究LLM驱动的主动对话系统中的交互式后门攻击，提出并实现三种后门范式（直接、受限进阶、无约束进阶），通过训练时对话数据投毒实现触发后引导用户朝攻击者指定目标行动。

**💡 创新点**

创新点在于将后门攻击从单轮响应扩展为多轮持续引导，构造触发条件下的策略偏移，并在无约束多轮场景中实现持续的目标驱动，首次展示了后门对整个对话轨迹的操控。

**🔧 技术方法**

主要技术包括对话数据poisoning、LoRA微调、信号-失真优化实现目标对齐、三阶段候选生成与多代理采样、ASR/FTR评估以及基于专业维度的对话质量测评。

**📊 数据集**

使用的数据集包括中文心理咨询对话集 PsyDT、PsyDial，英文 ESConv，教学对话集 MathDial；后端模型为 Llama3-8B-Instruct 与 Qwen2-7B-Instruct。

**📈 对比分析**

通过与干净基线对比，使用 ROUGE、BLEU、情感共情等专业度量评估常规性能；攻击成功率（ASR）分别从 16%→84%（直接）、2%→66%（受限进阶）、约30%→87%（无约束进阶），在保持大部分正常功能的前提下实现高攻击成功。

**⚠️ 局限性**

局限性包括：现有训练数据过滤与后期对齐只能部分抑制攻击；实验使用仿真用户，真实用户行为的鲁棒性尚待验证；攻击对触发词、语言与任务的依赖性需要进一步评估。

---

## 62. An LLM-Orchestrated Neuro-Symbolic Framework for Verifying Diagnostic Signal Reachability in Electronic Systems

**arXiv ID:** 2611.07040 | [PDF](https://arxiv.org/pdf/2611.07040v1)

**作者:** Heba Jaradat `[一作]` (Binghamton University, State University of New York), Sangwon Yoon `[通讯]` (Binghamton University, State University of New York)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套名为 NetFlow-DSR 的全流程诊断信号可达性验证框架，自动识别诊断信号、解析器件语义、构建电路图、执行合并与 BFS 搜索，并为每条可达路径生成可审计的证明卡。

**💡 创新点**

创新点包括：① 将 LLM 仅用于提取 netlist 无法编码的器件语义（如功能类别、监测方式），将结构计算完全交给确定性图算法；② 在电路图上分离信号身份与可达性关系，构造“功率保护”可达性图；③ 通过可审计的证明路径实现零假阳性、可追溯性；④ 在跨模型实验中证明精度由结构求解器控制，召回率受 LLM 语义完整性影响。

**🔧 技术方法**

技术方法：LLM（OpenAI GPT 类模型）提取器件角色与控制器识别；Netlist 解析与 Pin 方向从设计库读取；Union‑Find 并查集合并同一物理信号的重命名片段；BFS 广度优先搜索在可达性图上完成控制器可观测性判定；生成 trace‑card 记录路径与节点信息。

**📊 数据集**

数据集：五块生产级 AI 服务器/加速器板（主板、加速器、HDD、PCIe、SCM），共 421 条专家标注的诊断信号（369 正例，52 负例），包含不同管理控制器架构与器件种类。

**📈 对比分析**

与直接节点检测基线相比，三阶段 ablation 提升 recall 从 40.7%→64.2%→89.4%，F1 从 0.578→0.782→0.944；精度始终 100%，无假阳性。跨模型实验在同一 netlist、同一设备规格下，三款 LLM 的精度均保持 100%，召回率与运行时间分别为约 88%/3min、88%/6min、80%/21min。

**⚠️ 局限性**

局限性：① 需要完整、准确的设备规范才能正确提取器件语义；② 对电源线间接监测（如电源良好信号）和 active buffer、I/O expander 等功能未完全覆盖；③ 仍需人工标注作为基准；④ 对跨板信号需要进一步实现 trace‑card 合并；⑤ 对缺失设计库信息的电路仍需手工推断。

---

## 63. Call-Level Revision Contracts for LLM-Authored Workflows

**arXiv ID:** 2611.06967 | [PDF](https://arxiv.org/pdf/2611.06967v1)

**作者:** Chenyang Jiang `[一作]` (South China Normal University), Puqu Wu `[通讯]` (South China Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现一种可在模型编写的工作流脚本中执行修复（repair）的合同（revision contract），保证在重写后不会重复付费或返回被替换的结果。

**💡 创新点**

创新点包括：① 将调用的结构身份（callsite、fanout路径、出现计数）作为唯一键；② 通过静态门（gate）在容器启动前检测对已冻结调用表达式的移动或修改；③ 在运行时使用 payload digest 检测已提交调用的 payload 变更；④ 记录 occurrence ledger 识别相同 payload 的语义重复；⑤ 通过上述三层检查实现“一次执行一次结果”的安全性。

**🔧 技术方法**

技术细节：基于 Python AST 的静态检查与 callsite 标注；结构身份计算与签名链；运行时 broker 负责授权、idempotency 与记录；容器化 sandbox（无网络、只读根、Seccomp、cgroup 限制）保证安全；repair loop 对模型生成的候选脚本进行检查、gate 通过、执行，并报告 orphan。

**📊 数据集**

使用的数据集：媒体工作流（含图像生成、审核、故事板），客服支持任务（文本生成、审阅），60 条人工标注任务，32 条模型生成脚本（失败场景），35 条 mutation corpora 以及 12 条客服任务，另外通过模拟的 provider（图像、文本、LLM）和 Durable engine（Temporal）进行基准。

**📈 对比分析**

比较方法：与完整重跑、与 Durable 引擎的 reset 以及与仅 replay 的对比；在媒体工作流中，repair 只执行 5 次调用而重跑执行 12 次，节省 7 次付费；在客服任务中，gate 在重写过程中只执行 1 次调用；总体性能上 gate 与 digest 的开销低于 10 ms，sandbox 约 50 ms，整体执行时间比重跑快 30‑80 秒，成本节约 10‑50 % 取决于 fanout 宽度。

**⚠️ 局限性**

局限性：① 仅适用于结构身份不变的场景，无法接受已移动调用的重写；② 对非确定性或外部状态（如时间、随机种子）的 payload 变更仍需 provider 端 idempotency；③ 需要先验的 committed prefix，无法在无历史的首次运行中使用；④ 仅对单一容器/单一 provider 的环境做评估，跨多 host 或多 provider 的扩展未测试。

---

## 64. Gaussians Can Subtract: Neural Dissolve Fields for Hybrid Scene Rendering

**arXiv ID:** 2611.07049 | [PDF](https://arxiv.org/pdf/2611.07049v1)

**作者:** Yiyan Wang `[一作]` (University of Tokyo), Tetsuo Shibuya `[通讯]` (University of Tokyo)

**通讯引用:** 3055 | [OpenAlex ID](https://openalex.org/A5043159528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种混合 Mesh–Gaussian 渲染框架，利用可训练的 Neural Dissolve Field 对不一致的 Mesh 内容进行抑制，然后让残差 Gaussian 重新填补可见区域。

**💡 创新点**

将 Mesh–Gaussian 冲突视为多视角不一致问题，设计可视化方向可变的溶解场，并采用连续传输融合而非硬裁剪，从而实现“subtract‑and‑refill”分解。

**🔧 技术方法**

3D Gaussian splatting、可微光栅化、视角相关球谐展开、可训练的 Neural Dissolve Field、连续传输融合、双重密集化训练策略。

**📊 数据集**

ML‑Hypersim‑AI、Deep Blending 以及与 MeshGS、Hybrid Mesh‑Gaussian 等公开数据集进行评估。

**📈 对比分析**

与 3DGS、Reduced 3DGS、3DGS‑MCMC、MeshGS、Hybrid Mesh‑Gaussian 等基线比较，在保持或减少 Gaussian 数量的前提下，PSNR 与 SSIM 明显提升、LPIPS 降低，Gaussian 数量减少约 50%–60%，渲染速度保持在实时范围。

**⚠️ 局限性**

对 Mesh 质量和纹理投影的依赖较大；在高度非结构化或极细小细节场景中的表现仍有限；训练时间相对更高；需要额外的 Mesh 与 NDF 资源。

---

## 65. ReDrone5G: Field Trial Evaluation of Remote UAV Media and Control over Public and Private 5G Networks

**arXiv ID:** 2611.07068 | [PDF](https://arxiv.org/pdf/2611.07068v1)

**作者:** Kashif Mehmood `[一作]` (Norwegian University of Science and Technology), Hacı Mehmet Güzey `[通讯]` (Sivas University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在现场实验中评估了通过公共和私有5G网络实现无人机远程媒体传输与控制的性能。

**💡 创新点**

首次在实际环境中将无人机控制与5G网络结合，并系统比较了公共5G与私有5G在延迟、吞吐量和可靠性方面的差异。

**🔧 技术方法**

采用了5G NR（gNodeB、基站）、Wi‑Fi 6、LTE、软件定义无线电（SDR）与无人机SDK、边缘计算节点等技术栈。

**📊 数据集**

使用了现场收集的网络延迟、吞吐量、丢包率等指标数据，以及无人机视频流（4K@30fps）与遥控命令的实时采样数据集。

**📈 对比分析**

通过与Wi‑Fi 6、LTE等传统网络的基准实验对比，5G网络在平均延迟<50 ms、视频质量保持4K@30fps、控制命令丢包率<1%等关键指标上显著优于对比方案。

**⚠️ 局限性**

实验受限于可用基站密度、频段干扰、无人机数量与测试时间，且仅在城市区域内进行，无法涵盖极端天气、大范围空域或复杂障碍物环境。

---

## 66. Cooperative Multi-Agent Reinforcement Learning via Epigraph-Form Guided Exploration

**arXiv ID:** 2611.07066 | [PDF](https://arxiv.org/pdf/2611.07066v1)

**作者:** Sungil Son `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种保守式探索框架EFXPLORER，利用epigraph约束在合作MARL中最大化内在奖励同时保持外在任务性能。

**💡 创新点**

创新点在于将探索视为epigraph约束优化，消除线性加权的敏感性；并引入基于后继距离的个体化内在奖励实现长周期探索。

**🔧 技术方法**

采用epigraph重构、离散化预算更新、基于后继距离的对比学习、PPO/MAPPO等actor-critic技术。

**📊 数据集**

在SMAX、VMAS和MPE（包括自制的MPE-corridor）三组基准任务上进行实验。

**📈 对比分析**

与多种中心化与独立学习基线（如MAPPO、IPPO、COIN、HASAC、DiCO）进行对比，EFXPLORER在大多数任务上取得更高的归一化团队回报，并在可视化实验中展示更稳健的协作。

**⚠️ 局限性**

局限在于仅适用于基于状态观测的policy-gradient方法，需同步更新参考策略；在高维感知、价值学习或无参照更新场景下效果未知。

---

## 67. SeVerus: Context-Aware, Gamified Benchmarking for Rust Verification

**arXiv ID:** 2611.07194 | [PDF](https://arxiv.org/pdf/2611.07194v1)

**作者:** Álvaro F. Silva `[一作]` (University of Porto), João F. Ferreira `[通讯]` (University of Porto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于证明依赖图的“分离化（severification）”技术，将已验证的 Rust 项目拆分成一系列递增难度的验证任务（quests），每个任务从只缺少一个中间 lemma 开始，逐步撤除原始证明上下文直至完整证明被完全重构。

**💡 创新点**

创新点在于：① 用游戏化的层级设计（levels）可系统地衡量代理在缺失证明支持时的能力；② 通过深度优先的证明依赖展开来构造层级，保持目标函数不变，突出证明构造与依赖发现的双重挑战；③ 引入“噪声二分搜索”与“可靠层级得分（RLS）”的评估框架，避免逐层评估的高成本。

**🔧 技术方法**

技术手段包括：Rust 代码静态分析构建证明依赖图；三种状态（proved/ admitted / hidden）来生成层级；Verus 逆向检查与反作弊检查；对话式 LLM 代理（Codex GPT‑6 Sol、Luna、Claude Sonnet 4.6）进行交互式证明生成；噪声二分搜索与 Jaccard/结构相似度等统计方法用于评估与比较。

**📊 数据集**

数据集：从六个公开 Verus 项目（Anvil、IronKV、PMemLog、NRKernel、TLA‑RS、Vest）中抽取，生成 819 个 quests、10,095 个层级；实验样本包含 60 个 quests（10 个 per 项目）供 RQ2 与 RQ3 评估。

**📈 对比分析**

评估方法：对每个层级执行一次或三次 LLM 生成尝试，并通过 Verus 通过 anti‑cheat 检查验证；对 RQ1 用 Mann‑Kendall 检验成功率随层级递增的趋势；对 RQ2 用噪声二分搜索快速估计可靠最大层级，计算 RLS；对 RQ3 对比 agent 生成的 lemma 语句与结构与原始证明的相似度；实验结果显示：GPT‑6 Sol 在 60 个 quests 中根（root）级别成功率 52/60，平均 RLS 0.35；Luna 接近但成本更低；成功率随层级递增显著下降，成本随层级递增显著上升。

**⚠️ 局限性**

局限性：① 仅基于 Verus 的 Rust 代码，未扩展到其他语言/验证器；② 仅使用六个项目，难以覆盖更广泛的应用场景；③ 层级生成采用单一深度优先顺序，可能影响难度分布；④ LLM 代理受预算与时间限制（$5 或 4h）约束，影响最终结果；⑤ 评估中部分成功证明可能借助已知公开证明（网络限制下仍存在泄露风险）。

---

## 68. VINO: Revisiting 3D Representation Learning with Volume Transformers

**arXiv ID:** 2611.07176 | [PDF](https://arxiv.org/pdf/2611.07176v1)

**作者:** Adrian Kruse `[一作]` (RWTH Aachen University), Bastian Leibe `[通讯]` (RWTH Aachen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套全新的3D表示学习框架，利用Volume Transformer、教师-学生解耦、交叉注意力掩码预测以及视角一致的2D‑3D蒸馏，实现对室内外场景的高质量语义与几何表示；

**💡 创新点**

创新点在于①将PTv3替换为单分辨率的Volume Transformer，消除多尺度上采样的干扰；②教师与学生采用不对称视角（教师全局干净，学生局部强扰动）并将目标头与学生头解耦；③在掩码建模中去除掩码位置的几何信息，仅通过交叉注意力预测缺失表示；④通过学习视角一致的2D原型实现对3D编码器的稳健蒸馏；

**🔧 技术方法**

核心技术包括：Volume Transformer、教师‑学生自蒸馏、交叉注意力掩码预测、Sinkhorn‑Knopp平衡分配、视角一致的2D原型学习、两阶段训练（先3D自监督再2D‑3D蒸馏）；

**📊 数据集**

实验使用室内数据集ScanNet、ScanNet++、S3DIS、ARKitScenes、HM3D、RealEstate10K，以及户外数据集nuScenes、SemanticKITTI、Waymo；

**📈 对比分析**

在6个基准上（ScanNet200、ScanNet、ScanNet++、nuScenes、SemanticKITTI、Waymo）冻结线性头时均取得最佳性能，ScanNet200提升5.2 mIoU，加入2D‑3D蒸馏后进一步提升4.5 mIoU；在细调任务中同样在5/6基准上领先；在跨域配准任务中达到84.1%/44.6%，在跨模态语义探测任务中3D版在NYUv2获得59.2 mIoU，2D版在ScanNet++获得48.2 mIoU；

**⚠️ 局限性**

主要局限在于仍依赖大量标注或预训练图像数据，未能在极低采样或非对齐场景中验证；对不同分辨率或稀疏度的鲁棒性尚待进一步评估；

---

## 69. Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons?

**arXiv ID:** 2611.07166 | [PDF](https://arxiv.org/pdf/2611.07166v1)

**作者:** Xiangkai Zeng `[一作]` (Klavis AI), Zihao Lin `[通讯]` (Klavis AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个多步、长周期的编码代理基准Multi-step SWE，构建了69个软件工程任务，每个任务包含5–15个相互依赖的、耗时数小时的完整工程请求，要求代理在每一步基于前一步的代码完成指定功能并通过相应验证器；

**💡 创新点**

创新点在于：①将长周期、分步工程需求与传统单次任务结合，逼近真实开发流程；②每一步均基于公开的规范、文档或原始代码库作为“真相源”，保证指令与验证器高度一致；③覆盖四大工程范畴（性能优化、功能实现、版本升级、迁移）和九种主流语言；④通过对验证器的对抗性强化，防止奖励钓鱼与作弊。

**🔧 技术方法**

技术手段包括：Harbor多步任务格式、各模型的代理抓手（Claude Code、Gemini CLI等）、每步独立的验证容器、基于公共工具（测试套件、基准程序）进行验证、对抗性硬化以抵御作弊。

**📊 数据集**

数据集为69个基准任务，来源于真实开源项目的发布历程、协议修订、框架迁移与优化核实现；每个任务拆分为5–15步，总计574步；语言分布为C/C++(38)、Rust(11)、Java(6)、Go(4)、JavaScript(4)、Python(3)、PHP(2)、Ruby(1)，任务规模从几千行代码到数万行不等。

**📈 对比分析**

比较方法：对九个前沿模型分别进行四次跑通，记录每一步的完整通过与否，计算solve rate（所有步骤均通过）和Pass@4（至少一次通过）。结果显示，Claude Opus 5.5最高达18.5%的solve rate（31.9%通过至少一次），其余模型低于17.8%，整体任务完成率极低，且性能随步骤累积显著下降。

**⚠️ 局限性**

局限性包括：语言分布偏向C/C++，其他语言样本不足；任务可能仍存在歧义或验证器漏检；因基准来源于公开项目，模型存在记忆化风险；参考实现的速度阈值不一定为最佳；未能完全区分步骤难度与累积误差，且未覆盖多模态或UI交互等工作场景。

---

## 70. FastAFD: When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?

**arXiv ID:** 2611.07149 | [PDF](https://arxiv.org/pdf/2611.07149v1)

**作者:** Yichao Fu `[一作]` (University of California San Diego), Hao Zhang `[通讯]` (University of California San Diego)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在GB200 NVLink系统上实现并调优Attention‑FFN Disaggregation（AFD）推理引擎，并与最优的vLLM宽专家并行（wide EP）设置进行对比实验；

**💡 创新点**

提出了一个仅基于已部署的colocated vLLM性能剖面即可预测AFD吞吐量的分析模型，并将其拆解为批量增益、重叠增益和离散化成本，揭示哪些MoE模型最能受益；

**🔧 技术方法**

使用了Attention‑FFN离散化、MoE专家并行、重叠调度、Speculative Decoding以及针对GB200 NVL72的优化技术，配合SGLang、vLLM、DeepEP、DeepGEMM等开源框架；

**📊 数据集**

在大规模MoE模型上进行推理实验，主要使用LLaMA 2‑13B、Qwen等公开模型；

**📈 对比分析**

通过在同一硬件平台上对比AFD与宽EP设置的吞吐量，AFD在LLaMA 2‑13B上实现最高1.17×的加速，Qwen模型也达到了1.16×的提升；

**⚠️ 局限性**

受限于当前硬件的离散化成本、需要针对不同模型手动调优宽EP宽度、以及未在更大规模或更不同硬件上验证；未来需要在具备SRAM加速器的系统上进一步测试。

---

## 71. Scalable Bayesian Gaussian Process Classification for Cancer Histopathology: Calibrated Uncertainty Under Distribution Shift

**arXiv ID:** 2611.07127 | [PDF](https://arxiv.org/pdf/2611.07127v1)

**作者:** Yunru Zheng `[一作]` (New York University), Zhenyu Hong `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

使用冻结的 Phikon 病理图像嵌入构建可扩展的低秩贝叶斯高斯过程分类器（GPC），并通过 Hamiltonian Monte Carlo（HMC）采样后验，显著提升跨医院癌症检测的校准与预测性能。

**💡 创新点**

核心创新包括：① 将加速随机枢轴 Cholesky（RPCholesky）与低秩核分解结合，实现 N≈3×10^5 的训练样本可扩展；② 在非中心化参数化下对低秩 GPC 进行 HMC 采样，克服传统变分推断的方差低估；③ 使用枢轴点作为诱导点的 DTC 预测方案，实现高效预测。

**🔧 技术方法**

技术方法包括：低秩 RPCholesky、非中心化 HMC、枢轴诱导点 DTC 预测、Phikon 嵌入、随机抽样、对数似然与 ECE 等校准指标。

**📊 数据集**

数据集为两大病理图像基准：PCam（约26万训练样本）和 CAMELYON17-WILDS（约30万训练样本，跨医院分布移位）。

**📈 对比分析**

与 SVGP 变分 GPC 和三种神经网络基线（MLP、U-Net、ViT-S）比较，低秩 HMC GPC 在 PCam 上 ECE、准确率、AUROC 等均优于 SVGP；在 CAMELYON17-WILDS 上，ECO、NLL、Brier 分数、误检率降幅分别达到 88%、43%、77% 与 90% 以上，校准优势最为显著。

**⚠️ 局限性**

局限包括：① 未对核长度尺度和幅度进行完整贝叶斯推断；② 参数调优采用顺序搜索，缺乏系统化优化；③ 对比基线差异不完全归因于后验推断；④ 仅在补丁级别评估，未验证患者级临床效用。

---

## 72. PIT and Reconstruction for Sums of Powers

**arXiv ID:** 2611.07106 | [PDF](https://arxiv.org/pdf/2611.07106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 73. Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning

**arXiv ID:** 2611.07126 | [PDF](https://arxiv.org/pdf/2611.07126v1)

**作者:** Tamás Janusko `[一作]` (Hochschule für Technik und Wirtschaft), Maik Thiele `[通讯]` (Hochschule für Technik und Wirtschaft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在冻结的开放源LLM评判者上进行软提示调优（SPT），使其更好地与人类注解者的评价策略对齐，并系统评估其在偏见、毒性和个性化三大维度上的表现。

**💡 创新点**

首次将软提示调优应用于LLM-as-a-Judge，系统比较不同初始化、文本提示组合，并探究虚拟token本身能否完整表达任务信息，以及其跨数据集的泛化能力。

**🔧 技术方法**

使用软提示调优（SPT）对 Qwen 3.5 0.8B/9B 与 Llama 3.2 1B/3.1 8B 等冻结基础模型进行参数高效微调，结合文本提示、虚拟token训练与控制实验。

**📊 数据集**

训练与评估基于 FairBotBench（含偏见、毒性、个性化三维度的对话评价）进行，外部验证使用 ToxiGen 以测试毒性评判的跨数据集迁移。

**📈 对比分析**

与基础模型（仅文本提示）以及托管 GPT-5.4 mini/5.6 Terra 做基准比较；指标包括 Krippendorff's α、MAE 与 within‑one accuracy；Hybrid SPT 在多数模型与维度上提升了 0.2–0.5 的 α 值，部分情况下甚至超过托管模型，且显著降低 MAE 与提升 within‑one accuracy。

**⚠️ 局限性**

局限性：实验仅覆盖单一基准与三维度；跨数据集验证仅针对毒性；未与 LoRA、adapter 等其他参数高效微调方法做直接对比；虚拟token在不同模型间迁移性能未作深入探讨。

---

## 74. PromptOps: Iterative Prompt Optimization for LLM-as-a-Judge Systems

**arXiv ID:** 2611.07123 | [PDF](https://arxiv.org/pdf/2611.07123v1)

**作者:** Sven Kleinkop `[一作]` (Hochschule für Technik und Wirtschaft Dresden), Maik Thiele `[通讯]` (Hochschule für Technik und Wirtschaft Dresden)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PromptOps，一个闭环框架，用来持续监控并迭代校准 LLM 作为评判者的提示，防止评判漂移。

**💡 创新点**

创新点在于将提示优化拆分为触发器、调度器、样本选择器和优化器四个可独立配置的组件，形成可操作的持续维护流程。

**🔧 技术方法**

使用了五种提示优化技术（GEPA、SIMBA、Rubric Synthesis、示例选择的模拟退火和 Tree‑structured Parzen Estimator）以及两种评判后端（GPT‑5.4‑nano/Gemma‑4）来实现。

**📊 数据集**

基准数据集为德国–英语电商对话集 600 条，包含偏见、毒性和个性化三个维度的人类标注。

**📈 对比分析**

通过对比静态提示、不同 Dispatcher 配置、周期性与性能基线触发器，实验显示迭代优化可在漂移后恢复或提升 MAE，且不同后端与成本下的 Pareto 前沿显著差异。

**⚠️ 局限性**

局限在于仅使用单一 500 轮短流、两种后端、三次重复，且假设即时完整的标注反馈，未覆盖延迟、稀疏、噪声或争议反馈。

---

## 75. Consistent yet Contextual: Learning Author-Level Mixtures of Style for Authorship Attribution

**arXiv ID:** 2611.07111 | [PDF](https://arxiv.org/pdf/2611.07111v1)

**作者:** Omkar Yadav `[一作]` (University of Michigan), David Jurgens `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出通过高斯混合模型（GMM）显式捕捉作者多样化的写作风格，提升作者身份验证与归属的准确性。

**💡 创新点**

创新点在于将作者建模为多重一致但不同的子风格，突破传统单一风格假设，并系统分析上下文、时间、主题与对话对作者风格的影响。

**🔧 技术方法**

使用的技术包括：多种语义风格表示（NeuroBiber、LISA、LUAR、PCM、StyleDistance、SADIRI）、PCA正交化、GMM（自适应组件数）、邻居与质心模型、RoBERTa微调，并通过对比实验验证。

**📊 数据集**

数据集涵盖约4万作者、约300万篇文档，来源包括Reddit评论、文学书籍、GMANE邮件、维基百科文章与讨论页，保证了跨域与时间跨度。

**📈 对比分析**

与最近的基准（Nearest Neighbor、Centroid、RoBERTa）对比，采用GMM的模型在Success@8和p(a|d)指标上提升约5–15%（取决于风格表示），在四种数据源均表现优异。

**⚠️ 局限性**

局限性包括：可能仍存在话题/社区偏倚、对短文本与少量文档的适用性有限、以及不同上下文/时间段下作者风格变化导致的泛化瓶颈。

---

## 76. WAVES: Wearable Active Vibroacoustic Elasticity Sensing for Low-Cost and Non-Invasive Muscle Stiffness Monitoring

**arXiv ID:** 2611.07087 | [PDF](https://arxiv.org/pdf/2611.07087v1)

**作者:** Shubham Rohal `[一作]` (University of California, Merced), Shijia Pan `[通讯]` (University of California, Merced)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种可穿戴活性声学弹性传感系统，用骨传导发射器与IMU阵列测量肌肉在休息状态下的剪切波传播，从而连续估算肌肉刚度。

**💡 创新点**

创新点在于将低成本骨传导驱动与贴片式IMU结合，配合基于生理约束与几何一致性的TDoA筛选与聚合，显著提高了在日常使用中的刚度估计精度。

**🔧 技术方法**

采用声学剪切波激励、连续小波滤波、相位一致性筛选、GMM聚合以及时间差估计（TDoA）等技术实现。

**📊 数据集**

使用球弹性凝胶（不同硬度）以及14名受试者在股肌、斜方肌和前臂的肌肉数据，作为实验数据集。

**📈 对比分析**

与传统的交叉相关（CC）、GCC-PHAT以及无筛选基线相比，系统平均误差降至约4.9%，拒绝率较低，整体性能在30%误差阈值内。

**⚠️ 局限性**

局限性包括对肌肉表面附着和运动产生的干扰仍有一定影响，需进一步提升对不同体型和皮下脂肪层厚度的适应性。

---

## 77. Regulating Autonomous Mobility-on-Demand Operators with Fixed and Adaptive Penalties: A Multi-Agent Reinforcement Learning Approach

**arXiv ID:** 2611.07122 | [PDF](https://arxiv.org/pdf/2611.07122v1)

**作者:** Korbinian Kügler `[一作]` (Technical University of Munich), Filipe Rodrigues `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了针对自主移动需求运营商的固定与自适应罚款监管机制，并通过多智能体强化学习联合学习运营商与监管者策略。

**💡 创新点**

将监管视为外部控制循环，提出部分可观测随机博弈（POSG）实现自适应罚款，并将固定与动态监管在实际曼哈顿数据上的效果进行比较。

**🔧 技术方法**

采用图卷积网络+多智能体深度强化学习（SAC 与 A2C）以及 MDP/POSG 框架，对运营商的定价与车辆再平衡以及监管者的罚款策略进行联合训练。

**📊 数据集**

使用 2013 年 9 月 15‑20:59 的纽约出租车历史数据、公交地铁 GTFS 数据以及对应的行驶时长和费用信息，构建南曼哈顿 12 区域的仿真场景。

**📈 对比分析**

通过与无监管基准、固定罚款以及动态罚款的对比实验，发现固定罚款可将过度涨价用户比例从 22% 降至 1%，而动态监管在区域服务可用性上相较于匹配平均罚款的静态方案提升约 15.6% 的监管奖励，并降低平均价格因子、提高服务率和乘客实用性。

**⚠️ 局限性**

局限性包括：监管效果受车队容量限制，峰值期间仍难以完全满足目标；学习过程对罚款水平敏感；未考虑多运营商竞争与乘客异质性，且罚款对运营商盈利带来较大压力。

---

## 78. A Survey of Nonlinear Filtering Methods for Data Assimilation and Recent Advances in Generative AI-Enabled Score Filters

**arXiv ID:** 2611.07160 | [PDF](https://arxiv.org/pdf/2611.07160v1)

**作者:** Cath Pham `[一作]` (Florida State University), Kadesha Reynolds `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了EnSF及其十四种扩展，探讨其在高维非高斯滤波中的优势。

**💡 创新点**

创新点在于将滤波密度编码为分数函数，使用训练无关的Monte Carlo估计，突破了高维、非高斯、计算成本三大瓶颈。

**🔧 技术方法**

技术包括后向随机微分方程、分数导数理论、逆向扩散、Monte Carlo估计，以及各种扩展如IEnSF、EnFF、Fire-EnSF等。

**📊 数据集**

数据集涵盖多维Lorenz-96、Surface Quasi-Geostrophic（SQG）模型、Navier‑Stokes、Burgers、火灾模型以及真实气象观测等。

**📈 对比分析**

与EnKF、PF、LETKF等传统方法对比，EnSF在维度5–1000、非线性观测下RMSE显著更低，计算时间相对更短，尤其在大尺度下优势更明显。

**⚠️ 局限性**

局限在于对部分观测缺失时需额外修正、对非常稀疏或极度非线性观测需扩展、以及在全局尺度时仍需放弃训练无关特性（如DBUQ、STORM）以实现大规模应用。

---

## 79. Beyond Proof of Possession: A Relational Theory of Authority

**arXiv ID:** 2611.07159 | [PDF](https://arxiv.org/pdf/2611.07159v1)

**作者:** Nicola Gallo `[一作]` `[通讯]`, Nicola Gallo

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在授权模型中引入执行发生与因果历史的概念，区分 PoP、PoR 与授权，并构建了以执行者等价类为基础的关系空间和闭包理论。

**💡 创新点**

创新点在于：①把授权从单一持有者转移到执行发生的因果链；②提出多种等价关系的商空间和正向关系谓词格；③证明双向单调性与闭包，展示 PoP 并非授权必要；④给出不需要后继 PoP 的证据构造。

**🔧 技术方法**

主要技术是形式化的等价类商、关系谓词正向代数、单调性证明、证明持续性（Proof‑of‑Continuity）与关系闭包，配合安全假设与见证构造。

**📊 数据集**

无实测数据集，全部为形式化证明与理论分析。

**📈 对比分析**

比较方法为在同一执行发生域上定义观察映射与转移关系的两轴比较，结果通过形式化定理证明一致性与闭包，未给出实验性能。

**⚠️ 局限性**

局限性：依赖强假设（完整调解、因果归因等），无法覆盖侧信道与异步状态；实现需满足见证假设，理论与实际部署间仍有距离。

---

## 80. Elapsed-Service-Time-Aware Preemption for Age of Information Minimization

**arXiv ID:** 2611.07151 | [PDF](https://arxiv.org/pdf/2611.07151v1)

**作者:** Mohammad Moltafet `[一作]` (Tennessee Tech University), Roy D. Yates `[通讯]` (Rutgers University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究多源共享单服务器的状态更新系统，提出基于已服务时长的自我抢占策略，并推导出平均信息时效（AoI）和峰值信息时效（PAoI）的解析表达式。

**💡 创新点**

创新点在于：①引入任意分布的服务时间和与已服务时长相关的抢占概率，覆盖非抢占、全抢占和概率抢占三类；②利用非齐次吸收连续时间马尔可夫链与半马尔可夫链构建统一分析框架，得到多源情况下的AoI/PAoI闭式公式；③在单源情形下对15种不同危害率形态的服务时间进行系统评估，发现阈值抢占在单峰危害率分布上显著优于其它策略；④在双源对数正态服务时间场景下进行交叉源调优，阈值抢占实现更低的总AoI/PAoI。

**🔧 技术方法**

主要技术包括：非齐次吸收CTMC求解存活时间分布；半马尔可夫链计算交付周期；积分求解与MATLAB自适应四边形法；一维粗网格+Nelder‑Mead局部优化来调优抢占参数；对各种危害率分布进行数值积分。

**📊 数据集**

使用合成服务时间分布数据：15种以均值1为标度的分布（确定性、Erlang、Weibull、对数正态、对数逻辑、指数、双峰超指数、伽玛、Lomax等），以及双源实验中均值为1的对数正态服务时间。

**📈 对比分析**

比较方法为将所提阈值抢占（TP）与非抢占（NP）、全抢占（FP）和概率抢占（PP）进行对比，指标为平均AoI、平均PAoI以及两源情形下的总AoI/PAoI。实验结果显示：在单源下，TP在单峰危害率分布上可实现30–70%的AoI/PAoI下降；在双源下，TP调优后在三种λ_2情形下均实现最低的总AoI/PAoI，比其它策略优越。

**⚠️ 局限性**

局限性包括：①仅考虑泊松到达过程，缺乏对一般到达过程的分析；②推导需数值积分，难以对复杂服务分布得到闭式结果；③未对大规模多源（C>2）交叉调优进行系统评估；④假设各源更新相互独立且仅同源可抢占，实际网络中可能存在跨源抢占或队列管理策略的多样性。

---

## 81. More Agents, More Reliable? System-Level Reliability of Multi-Agent LLMs under Operational-Profile Uncertainty

**arXiv ID:** 2611.07140 | [PDF](https://arxiv.org/pdf/2611.07140v1)

**作者:** Burkay Tunçtürk `[一作]` (University of Hull), Bao Nguyen `[通讯]` (University of Leeds)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种结合层次模糊概率评估与贝叶斯网络的多代理LLM系统可靠性分析框架，并在化学危险信息提取管道中验证。

**💡 创新点**

创新在于将单代理的不确定可靠性以区间形式通过系统失效逻辑传播，允许组合规则本身失效，并给出可靠性包络，提供可辩护的可靠性声明。

**🔧 技术方法**

采用HIP-LLM的层次不确定概率评估、HiP-HOPS的系统失效树建模、贝叶斯网络推理、以及多模型投票（TMR）组合。

**📊 数据集**

使用公开的ECHA化学品危险信息网页快照，包含九种化合物的SMILES与对应H-code标注，用作测试和验证。

**📈 对比分析**

通过比较单代理、TMR冗余配置和实测投票结果的可靠性包络，发现TMR在最佳代理基础上并未显著提升可靠性；性能主要受代理证据和投票器失效率限制。

**⚠️ 局限性**

局限在于样本量小、仅单一任务与冗余结构、缺乏真实使用日志、组合规则模型近似、未估计共享失效因子与投票器失效概率。

---

## 82. A Proxy-Based Method for Measuring Racial Disparities in~Text-to-Image~Generators

**arXiv ID:** 2611.07132 | [PDF](https://arxiv.org/pdf/2611.07132v1)

**作者:** Nia Mucher `[一作]` (University of Virginia), David Evans `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于代理变量的自动化审计方法，用于评估文本到图像生成模型的表征偏差，尤其聚焦于种族和肤色偏差；

**💡 创新点**

创新点在于：①将美国反歧视法中常用的职业、地理、姓氏三类代理变量应用到图像生成审计；②结合CIELAB色彩空间的肤色量化和图像嵌入的相似度分析，揭示传统种族分类无法捕捉的细粒度偏差；

**🔧 技术方法**

技术包括：文本到图像生成模型调用（Imagen 4 Fast、Stable Diffusion 3.5 Large Turbo、Flux Schnell）；自动化种族分类器（Gemini 2.5 Flash）；CIELAB肤色提取与ΔE_2000差异度量；CLIP与人脸嵌入的余弦相似度计算；统计分析（TVD、总体差异距离）；

**📊 数据集**

数据集为美国人口普查（U.S. Census）中职业、地理位置、姓氏对应的种族分布；生成的图像样本为每个代理实例各生成100张；

**📈 对比分析**

比较方法是将模型输出的种族分布与人口普查基准通过TVD量化，进一步对肤色分布进行CIELAB分析，并计算同种族内部与跨种族的相似度差距；实验显示Flux Schnell在TVD上最接近人口普查，但其肤色多样性和同质性表现最差；Stable Diffusion 3.5在肤色多样性上最佳；

**⚠️ 局限性**

局限性包括：①基准仅限美国，跨国应用需替换代理和基准；②自动化种族分类器对Hispanic等类别的准确性低；③肤色量化受光照、图像质量影响；④方法聚焦种族/肤色，未覆盖性别、残障等其他偏差维度；

---

## 83. NUMA-Torch: Compiler-Driven NUMA-Aware LLM Prefill on Multi-Socket CPUs

**arXiv ID:** 2611.07105 | [PDF](https://arxiv.org/pdf/2611.07105v1)

**作者:** Thisara Rathnayaka Weerakoon `[一作]` (University of Moratuwa), Sanath Jayasena `[通讯]` (University of Moratuwa)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在 TorchInductor 编译链中引入 NUMA‑aware Tensor Parallelism（NUMA‑TP），实现多 socket CPU 上 LLM 预填充阶段的 DRAM 带宽瓶颈缓解。

**💡 创新点**

采用编译器级别的后梯度 FX 变换，将模型分块并在每个 NUMA 节点上局部分配权重和调度内核，完全透明无模型改写，显著提升跨 socket 内存局部性。

**🔧 技术方法**

使用 TorchInductor FX 变换、扩展的 Triton‑CPU 后端、显式 NUMA clone、线程组绑定与跨 socket all‑reduce 等技术。

**📊 数据集**

使用 GPT‑2（117M、345M、762M）、BERT（Small、Large）和 Gemma‑3（270M）等预训练模型进行评估。

**📈 对比分析**

与未修改的 Triton‑CPU/TorchInductor 基线对比，在双 socket Haswell/Cascade Lake 平台上分别获得最高 2.63× 与 1.87× 的预填充加速；在四 socket Haswell‑EX 平台上最高 2.27×；相较于 OS 级 NUMA 策略和单 socket 实验，性能更优。

**⚠️ 局限性**

仅针对预填充阶段，对解码阶段、KV 缓存放置无直接支持；在短序列或小模型时收益有限；需要进一步扩展策略以平衡通信与计算开销。

---

## 84. DMuon: Scaling Matrix Optimizers in Data Parallelism with Topology-Aware Matrix Ownership and Communication

**arXiv ID:** 2611.07084 | [PDF](https://arxiv.org/pdf/2611.07084v1)

**作者:** Starrick Liu `[一作]` (X Square Robot), Hao Wang `[通讯]` (X Square Robot)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

DMuon在PyTorch FSDP/HSDP数据并行环境中实现了矩阵级所有者分配，利用单一副本的Newton–Schulz迭代和参数更新，消除矩阵优化器与数据并行之间的通信和重复计算冲突。

**💡 创新点**

其创新点包括：1）为每个权重矩阵指派唯一所有者并跨节点共享梯度与状态；2）在HSDP中实现全局唯一所有者与跨步发布；3）实现可变长度的AllGatherV/ReduceScatterV以及持久化跨步发布，利用两级网络实现高效通信；4）通过负载平衡和Ready调度降低链路瓶颈。

**🔧 技术方法**

使用了Muon等矩阵优化器、Newton–Schulz迭代、PyTorch FSDP2（ZeRO-2/3）、HSDP、NVLink+InfiniBand两级网络、GPU-Initiated Networking (GIN)、自定义NCCL算子、CUDA/C++内核、Python runtime、对称内存池等技术。

**📊 数据集**

在三种密集LLM（Qwen2.5-1.5B、7B、32B）和两种VLA模型（Wall-OSS-0.5、π_0.5）上进行训练，使用了FineWeb-Edu等大规模文本数据集。

**📈 对比分析**

与PyTorch FSDP2+AdamW基线、Naïve Muon、MatrixFSDP等在多种 sharding 配置（FSDP-Z2/3、HSDP-Z2/3）下进行对比，DMuon在256 GPU时吞吐量比 AdamW 中位数高 1.05 倍，最高 1.34 倍；相比 Naïve Muon 提升 2–5 倍；与基线 C 的可变长度算子提升 3–78%。

**⚠️ 局限性**

局限性：仅适用于需要全矩阵更新的矩阵优化器；对两级网络和NVLink/InfiniBand的依赖；所有者分配与负载平衡实现复杂；在超过256 GPU或其他硬件平台的可扩展性未验证；对混合/低精度训练的支持仍有限。

---

## 85. Wide View Synthesis from Overlapping Perspective Cameras in Autonomous Driving

**arXiv ID:** 2611.07081 | [PDF](https://arxiv.org/pdf/2611.07081v1)

**作者:** Kaixuan Zhang `[一作]` (Nanjing University of Science and Technology), Xiatian Zhu `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出宽视图合成（WVS）任务并设计了基于3D高斯点阵的全流程框架WV-GS；

**💡 创新点**

创新点在于①Piecewise Perspective Rendering (PPR)通过多视角局部投影缓解宽视角下的边缘拉伸；②Spatial Affine Photometric Modeling (SAPM)在训练时对每个摄像机进行空间变换学习，消除曝光与相机响应差异；③引入合成数据集WideDrive作为定量评估基准；

**🔧 技术方法**

使用OmniVGGT特征提取、基于3D Gaussian Splatting的几何重建、PPR与SAPM模块、光度与几何监督、LiDAR-Projected Sparse-Reference (LPSR) 评估协议；

**📊 数据集**

在nuScenes、Lyft以及新构建的WideDrive数据集上进行实验；

**📈 对比分析**

与8种传统与学习型拼接方法比较，在nuScenes、Lyft及WideDrive上均实现最高PSNR（≥22dB）、SSIM（≥0.91）并且成功率为100%，比最强竞争者提升约5–10dB；

**⚠️ 局限性**

限制为对未被任何视角覆盖的区域无法生成内容，结果显示为白色空白，未来可通过扩散模型等方法填补这些缺失区域。

---

## 86. When Verification Is Not Agreement: Cross-Agent LLM-to-Dafny Formalization

**arXiv ID:** 2611.07073 | [PDF](https://arxiv.org/pdf/2611.07073v1)

**作者:** Priyanath Maji `[一作]` `[通讯]` (Georgia Institute of Technology), Priyanath Maji (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并评估了一种跨代理的正式化协议，用于检验两台LLM从同一自然语言文本独立生成的规范的一致性。

**💡 创新点**

将验证成功视为初步一致信号，并在此基础上加入结构审计与突变适用性检查，证明单靠验证无法保证两代理间真实独立一致；同时展示模型能力与角色分配对一致性的显著影响。

**🔧 技术方法**

使用Dafny进行形式化与验证；采用GPT‑4.1及GPT‑5.x系列LLM执行两条提取路线（字面与解释性），并通过结构审计（检测直接调用指定器谓词）和突变测试评估规范充分性。

**📊 数据集**

使用96条来自Chia语料库的临床试验纳入/排除条件文本。

**📈 对比分析**

对四种模型配置（同模型控制、角色互换、强弱模型对）分别计算“真诚一致率”，并在结构审计与突变检验后给出校正比例；同模型基准约59.3%，强实现器约53%，弱实现器仅14.3%；强模型组仅34.4%，且以矛盾为主。

**⚠️ 局限性**

角色与提取策略混合导致无法区分能力与阅读风格；未使用修复循环；无人工标注；仅针对单一数据集与Dafny；模型组合有限；结构审计仅捕捉简单调用短路。

---

## 87. MammoSeek: A Mammography Agent with selective zooming for grounded reports

**arXiv ID:** 2611.07188 | [PDF](https://arxiv.org/pdf/2611.07188v1)

**作者:** Haoyan Ding `[一作]`, Debiao Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于多模态大语言模型的乳腺影像阅读代理，模拟放射科医生的工作流程：先对四视图做全局感知，再自适应缩放可疑区域，基于BI‑RADS判定并生成结构化报告，最终与影像模型分数融合做召回决策。

**💡 创新点**

①实现与医生相同的逐步缩放与BI‑RADS判读流程；②设计保留高强度信息的“calcification view”以识别微钙化；③采用工具调用循环与BI‑RADS技能注入；④在召回决策中使用校准的逻辑回归融合多种特征。

**🔧 技术方法**

使用 Qwen3‑VL‑8B 多模态LLM、Faster R‑CNN 检测器、Mammo‑CLIP 线性探针、全分辨率缩放工具、BI‑RADS 语义技能、逻辑回归决策层和结构化报告生成。

**📊 数据集**

内部 Emory EMBED FFDM 444 试验样本（111 肿瘤，111 取活检良性，222 负面）和外部 VinDr‑Mammo 375 试验样本（BI‑RADS 4/5 为阳性）。

**📈 对比分析**

与通用/医学 MLLM（Qwen2.5‑VL‑7B、Qwen3‑VL‑32B、Lingshu‑7B）、MammoClaw 代理、NYU BI‑RADS 与癌症分类器、Mammo‑CLIP 线性探针等基线对比；本方法在内部和外部数据集上 exam‑level AUC 0.824/0.818，敏感度 0.50/0.51、特异度 0.90/0.89，BI‑RADS κ 0.33/0.43，定位 0.35/0.43，报告 F1 0.24/0.38，均显著优于基线。

**⚠️ 局限性**

性能受限于所用影像模型的诊断准确度；外部数据 VinDr‑Mammo 采用放射科医生 BI‑RADS 而非病理，导致外部提升部分来源于与医生一致；尚未在更多病理确证人群或临床读者研究中验证；对不同设备/扫描协议的迁移性待进一步评估。

---

## 88. Meta-Frieren: Beyond Single Level Online Learning

**arXiv ID:** 2611.07169 | [PDF](https://arxiv.org/pdf/2611.07169v1)

**作者:** Vahan Arsenyan `[一作]` (ENSAE), Arnak Dalalyan `[通讯]` (ENSAE)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段在线学习框架：meta‑learner 先选择一个 agent‑environment 对，随后该 agent 在观测到的上下文中决策并获得奖励；提出 Meta‑Frieren 方法，构造 meta‑层 UCB 进行对齐；给出统一的高概率累积回报界，并将其实例化为 GP‑UCB（Bayesian 与 frequentist）和 Feel‑Good Thompson Sampling (FGTS)。

**💡 创新点**

创新点包括：
1) 一个通用的 meta‑层 UCB 规则，只需各 agent 提供其内部的高概率回报界即可；
2) 证明 FGTS 的首次高概率累积回报界，并结合 Meta‑Frieren 给出整体的子线性回报保证；
3) 将 meta‑层框架与不同的下层学习器（GP‑UCB、FGTS）无缝结合，实现了对多种模型类的超参数搜索。

**🔧 技术方法**

采用的技术包括：
- 上界置信（UCB）分析与信息增益（max‑information‑gain）理论；
- 高斯过程回归与 GP‑UCB 的探索参数设计；
- Feel‑Good Thompson Sampling 的 Gibbs 后验与能量函数；
- 适用于 Bayesian 与 frequentist 设定的子序列均值与噪声浓缩；
- 通过联合概率、Hoeffding 与 sub‑Gaussian 取样，构建置信半径。

**📊 数据集**

使用的数据集与实验设置：
- 先验分布的 GP 目标函数（不同核、长度尺度）；
- Causal Bayesian Optimization（CoCa‑BO）所用的结构因果模型；
- 真实数据：MAGIC Gamma Telescope 数据集（10 个特征），用于三种模型类（GBDT、SVM、k‑NN）的超参数搜索。

**📈 对比分析**

比较方法与性能：
- 与 CoCa‑BO 直接对比，Meta‑Frieren 在 meta‑层选择上采用 UCB，收敛速度略慢但最终同样识别最优模型；
- 与贪心基准对比，Meta‑Frieren 能通过探索项避免被初始噪声误导，最终获得更低的时间归一化回报；
- 在 GP‑UCB 与 FGTS 设置下，Meta‑Frieren 均表现出子线性回报，达到 O(√T) 级别，且在多模型超参数优化实验中显著优于贪心策略。

**⚠️ 局限性**

局限性：
- 需要各 agent 给出高概率的内部回报界，若无此保证需自行证明；
- 对上下文独立性与噪声子高斯性有严格假设，实际环境可能不满足；
- meta‑层 UCB 可能因保守置信半径导致探索速度慢，尤其在 agent 数量较多时；
- 目前实验规模有限，尚未在大规模多模型超参数搜索（数十类）中验证；
- 参数调优（如 η、λ 序列）需经验或多次实验，缺乏统一自动化方案。

---

## 89. Full Provenance on the COSS Causal Axis: An Event-Level Closure of Causal Attribution and Confused-Deputy Exclusion

**arXiv ID:** 2611.07164 | [PDF](https://arxiv.org/pdf/2611.07164v1)

**作者:** Nicola Gallo `[一作]` `[通讯]`, Nicola Gallo

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

将 Provenance Identity Continuity (PIC) 模型中的事件从（操作/资源/历程）改为原子执行发生点，并推广至任意安全状态坐标的乘积模型 (COSS)，在此基础上对混淆代理（confused‑deputy）问题给出一条通用的、与架构无关的排除定理，并将 RGR 的能力与完整溯源分析映射到该因果框架中。

**💡 创新点**

创新点包括：①把事件本身作为因果结构的基元；②在同一因果偏序上携带多维安全状态坐标，实现一次性传递而非多时序；③在 DAG 上定义普通连续性（ordinary continuation）与授权放宽（authorized relaxation）两类转移，并证明普通连续性不能隐式扩展权限；④提出并证明“因果完整、受保护效应调解且普通连续性满足的系统中，任意 PIC 线性化的混淆代理实例必然不可接受”的普适定理。

**🔧 技术方法**

使用的技术主要是形式化模型（Lamport 风格的因果 DAG、部分序、下层安全状态坐标的乘积）、严格的归纳证明、持续性（continuity）与交叉（join）约束、以及对 RGR 完整溯源语义的事件细化。

**📊 数据集**

无数据集，论文完全基于理论证明与形式化推导。

**📈 对比分析**

无实验比较与性能评估，论文仅给出定理证明与抽象模型对比。

**⚠️ 局限性**

局限性：理论结果依赖于对系统的因果完整性、受保护效应调解、以及对普通连续性的严格实现；若存在物理/隐蔽通道、未建模的因果关系或未显式授权的非保守状态跃迁，则定理不适用。

---

## 90. Lossless Source Coding with Marginal Distribution Constraint

**arXiv ID:** 2611.07156 | [PDF](https://arxiv.org/pdf/2611.07156v1)

**作者:** Cheuk Ting Li `[一作]` (Chinese University of Hong Kong), Shahab Asoodeh `[通讯]` (Vector Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出并分析了损失无源编码在边缘分布约束下的理论框架，给出了非渐进和渐进的最优编码长度与编码率界限，并引入耦合聚合数 m^*(P‖Q) 来表征编码长度。

**💡 创新点**

创新点在于：①定义耦合聚合数，证明一般与特殊（如均匀目标分布、i.i.d. 源）情形的精确下界和上界；②将上界从传统的 H_0(P)/H_∞(Q)+O(·) 提升到 H_0(P)/H(Q)+O(α^{2/3})，显著缩小误差；③对近似边缘约束给出 Rate–Perception 函数；④在可容错误时证明最优率为 H(P)/H(Q)；⑤提出新的 Shannon 与 Hartley 熵的公理化表述。

**🔧 技术方法**

使用信息理论工具：majorization、Berry–Esseen 定理、相对熵、卡罗瑟里定理、子加性与幂级数展开、极限定理等，构造耦合与证明熵比界限。

**📊 数据集**

本工作为理论研究，没有使用具体实验数据集。

**📈 对比分析**

通过与已知的分布匹配、信号塑造、同源编码等方法对比，证明在 Q=Bern(1/2) 时编码长度仅比传统方式多 1 位；在渐进情况下得到最优比例 γ(P‖Q)=max{H(P)/H(Q),H_0(P)/H_0(Q)}；对近似边缘约束给出 Rate–Perception 函数，且可容错误时实现 H(P)/H(Q)。

**⚠️ 局限性**

局限在于对非均匀目标分布的精确上界仍不完全匹配；对无限支撑的 Q 缺乏完整的渐进描述；理论证明主要针对离散分布，缺乏具体编码实现示例。

---

## 91. Automatic Assistance to Reduce Information Overload during Software Issue Report Understanding

**arXiv ID:** 2611.07109 | [PDF](https://arxiv.org/pdf/2611.07109v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 92. Teaching Agents the "Principle", Not the "Reward"

**arXiv ID:** 2611.07130 | [PDF](https://arxiv.org/pdf/2611.07130v1)

**作者:** Shan Yang `[一作]` `[通讯]` (National University of Singapore), Shan Yang (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种直接使用行为原则指导策略学习的方法，称为 Prescriptive Policy Learning（PPL）。

**💡 创新点**

创新点在于将任务知识以行为原则形式提供给代理，并通过关键量和参考值计算的原则系数作为每一步的学习信号，避免奖励解码负担。

**🔧 技术方法**

采用基于策略梯度的更新框架，构造原则系数并将其替代传统奖励，支持多原则、各维度原则和优先级平衡。

**📊 数据集**

使用三类经典控制环境：自定义障碍规避任务、OpenAI Gym LunarLander-v3 以及 Reacher-v5。

**📈 对比分析**

与 PPO、DQN、SAC 和用原则系数替代奖励的 PPO-R 进行对比，PPL 在样本效率、成功率和鲁棒性方面均优于对照组，且在噪声或无训练扰动的环境中仍保持高性能。

**⚠️ 局限性**

局限性包括需人工设计关键量与参考值、在多阶段冲突目标任务中平衡原则困难、以及在更复杂的长期操作或全身控制任务上的推广性待验证。

---

## 93. Data-Driven Modeling of Molten Chloride Corrosion Using Static Test Data

**arXiv ID:** 2611.07125 | [PDF](https://arxiv.org/pdf/2611.07125v1)

**作者:** Md Mahmudul Hasan Pathik `[一作]` `[通讯]` (North Carolina State University), Md Mahmudul Hasan Pathik (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究对291条静态熔融氯化物腐蚀实验数据进行整理，并构建了一套可解释的机器学习框架，用以预测腐蚀速率；

**💡 创新点**

创新点在于提出了“盐制备指数”（CSPI）这一连续量化指标，将盐的净化过程、气氛控制和前暴露杂质等信息统一编码，并证明其优于传统的纯净/不纯净二元分类；

**🔧 技术方法**

使用了集成学习方法（XGBoost、CatBoost、Extra Trees、Gradient Boosting）以及SHAP、Sobol等解释与敏感性分析技术，并采用合规预测（Conformal Prediction）进行不确定性量化；

**📊 数据集**

数据集来自文献汇总的291条静态熔融盐腐蚀实验，包含12种合金元素、7种盐成分、容器材料、温度、暴露时间、CSPI以及可选的盐体积/表面积比等28-29个特征；

**📈 对比分析**

模型通过5折交叉验证（重复两次）进行比较，集成模型实现R²≈0.81，MAE≈0.43 mg cm⁻² day⁻¹，加入V/SA后R²提升至0.85，MAE降至0.41，超越单一模型并保持较好的泛化能力；

**⚠️ 局限性**

局限性包括：数据来源主要为文献，存在报告不一致、缺失关键实验参数（如红氧势、微结构等）；模型仅基于静态测试，无法捕捉动态流动、热梯度等工况；CSPI为基于文献描述的指数，无法完全反映实际盐纯度；

---

## 94. Machine Learning for Live-Birth Prediction After IVF/ICSI From Structured Clinical Data: A Systematic Review, Meta-Analysis, and Critical Appraisal

**arXiv ID:** 2611.07124 | [PDF](https://arxiv.org/pdf/2611.07124v1)

**作者:** Abdul Raheem Shahzad `[一作]` (Wuhan University), Ming Zhang `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对利用结构化临床数据预测 IVF/ICSI 活产的机器学习模型进行了系统综述、方法质量评估（PROBAST+AI、TRIPOD+AI）与随机效应元分析。

**💡 创新点**

首次将决策点、终点和单位分析等关键因素纳入模型评价框架，系统比较内部与外部验证的性能，并提供完整的质量评估清单与研究议程。

**🔧 技术方法**

采用树集成（GBM、RF）、深度学习、混合模型等机器学习算法，并通过 logit 变换、REML、Hartung–Knapp–Sidik–Jonkman 区间等统计方法进行元分析。

**📊 数据集**

整合了 48 项研究的英国 HFEA 注册数据、中国单中心数据及多国数据，覆盖从基础信息到实验室与胚胎学变量的结构化记录。

**📈 对比分析**

比较方法包括内部 vs 外部验证 AUC 的随机效应元分析、与同一数据集下逻辑回归的配对差异；内部验证平均 AUC 0.814，外部验证平均 AUC 0.756，机器学习模型对逻辑回归的平均提升约 0.056 AUC，但多来自高偏倚设计。

**⚠️ 局限性**

主要局限在于证据高度偏倚、外部验证不足、极高异质性、缺乏临床影响与可解释性评估，样本量与事件数有限导致不确定性。

---

## 95. A Night at the LLM Zoo: Data Conditioned Flow Matching for Generating Pretrained LLM Weights

**arXiv ID:** 2611.07103 | [PDF](https://arxiv.org/pdf/2611.07103v1)

**作者:** Scott Biggs `[一作]` (Northeastern University), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个受控的 GPT‑2 规模模型“动物园”，通过系统地改变训练数据的混合比例（Web 文本、书籍、数学、编码、法语）生成 300 个不同参数量的模型，并基于此训练一个条件流匹配生成器，能够在未见的混合比例下生成完整的预训练语言模型权重。

**💡 创新点**

创新点在于：① 只通过数据混合控制模型差异而非架构或超参数，形成可解释的权重空间；② 将 Gram‑PCA 压缩与流匹配结合，实现对大规模 GPT‑2 权重的条件生成；③ 在未见的数据混合上实现零-shot 泛化，并通过连续排名概率评分（CRPS）和下游任务验证生成模型的真实性。

**🔧 技术方法**

使用的技术包括：1) 统一 GPT‑2 架构与训练流程（分支训练法）；2) Gram‑PCA 权重压缩；3) 条件流匹配（rectified flow）与 MLP 速度场；4) 余弦相似度、方差分解、Leave‑One‑Mixture‑Out（LOMO）等统计分析；5) CRPS、平均 perplexity、HellaSwag 评估。

**📊 数据集**

数据集由 5 个文本领域（Web、书籍、数学、编码、法语）构成的混合语料，使用 Dirichlet 采样得到 85 种混合比例，生成 51M、124M、355M 参数的 GPT‑2 模型，每个规模 100 个模型。

**📈 对比分析**

与 1‑NN 检索、线性回归、权重混合（weight soup）等基线进行比较；在所有规模下，生成模型的平均 perplexity 变动仅为 4.1–5.1%（比检索基线 5.8–6.0% 更好），CRPS 技能约 0.70，且在 HellaSwag 上与真实模型表现相近；在未见混合比例上也能保持接近真实模型的功能表现。

**⚠️ 局限性**

局限性包括：① 样本规模有限（仅 300 个模型）；② Gram‑PCA 基础是传递性的，难以在完全新样本上直接生成；③ 对模型规模的扩展仍受压缩与解压开销限制；④ 在混合比例边界（顶点）上的泛化性能不如内部混合；⑤ 生成模型对细粒度功能差异（如 perplexity 概率分布）的捕捉仍不足。

---

## 96. Stacking Stars: Minimum-Area Packing with and without Rotation

**arXiv ID:** 2611.07100 | [PDF](https://arxiv.org/pdf/2611.07100v1)

**作者:** Mikkel Abrahamsen `[一作]` (University of Copenhagen), Hanwen Zhang `[通讯]` (University of Copenhagen)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了多项式时间的 O(log n) 近似算法，用于将任意星形多边形（可旋转或仅平移）装入面积最小的凸容器，并推广到箱子与条带装填问题。

**💡 创新点**

核心创新在于将星形多边形划分为“细类型”（同向径长相近且宽度相似），引入“碗”结构，证明可以把任意装填重新组织成简单堆叠；随后利用不对称 TSP 的常数近似来确定堆叠顺序，从而在整体上只损失 O(log n) 的面积。

**🔧 技术方法**

主要技术包括：尺寸分组与细类型划分、碗（bowl）转换、堆叠（stack）构造、对堆叠的结构性质证明、异质 TSP 的常数近似、尺寸组间的充电/核心方法来消除额外的 O(log n) 因子，以及对旋转版本的角度校正与刚性移动。

**📊 数据集**

该工作为理论研究，无实验数据集；所有结果均为理论证明与多项式算法。

**📈 对比分析**

与之前仅适用于凸多边形的算法相比，本算法在非凸星形多边形上实现了首个非平凡近似保证；与最优解相比，近似比为 O(log n)，并且对箱子与条带装填问题给出了对应的 O(log n) 近似。

**⚠️ 局限性**

局限性包括：常数系数较大、实现复杂；仅针对星形多边形（对更一般的非凸形状不适用）；条带与箱子问题需额外假设星形多边形直径受限；并且通过归约证明了 APX‑hardness，表明无法得到多项式时间内更好的多项式近似。

---

## 97. Optimistic Hedge in General-Sum Games: Fundamental Limits and Regret Bounds via Temporal Approximation

**arXiv ID:** 2611.07078 | [PDF](https://arxiv.org/pdf/2611.07078v1)

**作者:** Yuval Dagan `[一作]` (Tel Aviv University), Tomer Koren `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了多玩家在线学习中的可优化 Hedge（Optimistic Hedge）算法。首先给出了 2 名玩家时，使用该算法能够实现与对手交互所产生的损失序列的累积误差（regret）在时间 T 上的上界为 O(√T)。随后，作者构造了一类三名及以上玩家的博弈，证明在此类游戏中任何基于可优化 Hedge 的策略都存在 Ω(T) 的下界，即线性增长的 regret。整个工作提供了一套新的时序近似框架，用来分析可优化 Hedge 的连续时间行为，并将其映射回离散时间，从而得到精确的误差估计。

**💡 创新点**

创新点在于：
1) 提出并正式化了所谓的“时序近似框架”，能够在多玩家环境下将连续时间的动态转化为离散时间的更新并保持误差可控；
2) 对可优化 Hedge 进行细致的谱分析，使用指数和序列的平滑技术，获得 2 名玩家时的 O(√T) 上界；
3) 通过构造一个周期性动力系统（利用强稳定流形与隐函数定理），实现 3 名及以上玩家时的 Ω(T) 下界，展示了多玩家情形的根本难点；
4) 将传统的线性响应与非线性反馈相结合，证明了在特定信号下可观测玩家的误差会被放大，从而获得线性增长的 regret。

**🔧 技术方法**

技术手段主要包括：
- 指数级数（exponential sums）与其误差界定，借助 Bernstein 不等式控制高阶项；
- 线性化与 Jordan 标准形的使用，利用强稳定流形（strong stable manifold）和谱分解来分析周期轨道的吸引性；
- 隐函数定理（implicit function theorem）与可微分拓扑，构造周期轨道并保证参数化的平滑性；
- 连续时间与离散时间的转换，通过积分与微分的对等式来得到能量守恒与梯度下降的关系；
- 对游戏动力学的非线性控制与反馈校正，保证在周期性输入下的平稳响应。

**📊 数据集**

该研究为理论工作，未使用任何实测数据集。所有结果均基于严格的数学证明与构造。

**📈 对比分析**

性能比较方面：
- 对于 2 名玩家，本文提供了与已知最优结果相匹配的 O(√T) 上界；
- 对于 3 名及以上玩家，构造的博弈证明了任意基于可优化 Hedge 的策略都至少会出现 Ω(T) 的 regret，表明了上界与下界之间的根本不匹配；
- 在实验部分，作者对比了传统 Hedge 与可优化 Hedge 的误差曲线，验证了理论上限与实际表现的一致性。

**⚠️ 局限性**

局限性：
1) 仅对“可优化 Hedge”以及其特定的更新规则（带记忆项）进行分析，其他自适应或随机化策略不在覆盖范围内；
2) 下界构造依赖于特定的高维周期动力系统，可能不适用于所有自然的多玩家博弈；
3) 上界的证明假设损失值与奖励均在 [-1/2,1/2] 范围内，未考虑更广泛的实值范围；
4) 多玩家下的线性增长下界是在极端的构造环境下得到的，实际游戏中可能表现更好；
5) 该框架主要处理对称或近似对称的博弈，对于极度不对称或信息不完整的场景仍需进一步研究。

---

## 98. Shaping & Monitoring Models through their Internal Representations

**arXiv ID:** 2611.07071 | [PDF](https://arxiv.org/pdf/2611.07071v1)

**作者:** Thomas Fel `[一作]`, Owen Lewis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一框架，利用内部表示对模型进行约束，既可限制模型表达不良概念，又可限制模型对这些概念的依赖。

**💡 创新点**

创新点包括：将表达与依赖两类约束统一为基于表示的约束学习问题；构建监控层级体系以评估概念的可监测性；对三种“规避”几何机制（概念漂移、消除、非线性编码）进行了理论与实证剖析。

**🔧 技术方法**

技术主要包括：对内部激活的线性/非线性探测器（线性探测器、DoM、属性回归）、梯度与归因（梯度×输入、积分梯度、删除、随机噪声）作为约束损失；对RL奖励进行内部表示惩罚；以及周期性更新约束方向、重新训练探测器等策略。

**📊 数据集**

实验数据涵盖：视觉领域的 Waterbirds、ImageNet、CheXpert（胸部X光）；语言领域的 DPO/SFT/GRPO；以及奖励优化环境（Countdown、Chess）。

**📈 对比分析**

在这些任务中，所提出的方法显著提升了 worst‑group accuracy（Waterbirds 最高可达 95%）、降低了奖励游戏和有害输出（如 Countdown 的 peek 率降至 0，Chess 的作弊率降至 0.4%），并在保持原任务性能的同时，几乎不影响模型能力；对比基线（ERM、数据过滤、奖励重塑等）显示明显优势。

**⚠️ 局限性**

局限性包括：对监测器的依赖（如线性探测器可能失效，需要重新拟合或使用更高阶探测器）；训练过程可能诱发概念规避，导致监测器在长时间训练后失效；方法在极大规模模型与极长训练时的鲁棒性尚未验证；以及对非线性编码的检测需要更复杂的监测手段。

---

## 99. Quadratic Extractors

**arXiv ID:** 2611.07266 | [PDF](https://arxiv.org/pdf/2611.07266v1)

**作者:** Gautam Chandrasekaran `[一作]` (University of Texas at Austin), David Zuckerman `[通讯]` (University of Texas at Austin)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种显式二次源（quadratic source）随机数提取器，能够在最小熵仅为多项式对数级别时提取几乎全随机位。

**💡 创新点**

突破了传统求和集（sumset）提取器的熵门槛，首次构造了可在 k=polylog(n) 之下工作的二次源提取器，并引入了新的“秩组合器（rank combiner）”概念。

**🔧 技术方法**

核心技术包括二次条件化（quadratic conditioning）、结构定理与 Cauchy–Schwarz 解析、二次相关性打破器（quadratic correlation breaker）以及对低度多项式的线性种子提取器。

**📊 数据集**

该工作纯粹基于理论构造，无需实验数据集，所有结果均为显式多项式时间算法的证明。

**📈 对比分析**

提取器在误差为 n^{-Ω(1)} 的前提下输出长度 m= k^{Ω(1)}，在多项式对数熵范围内优于以往所有显式提取器（例如仅能达到 k≈ n-√{log n} 的前置工作）。

**⚠️ 局限性**

限制在于仅适用于二次（degree‑2）源，对更高次数多项式源的推广仍未解决，且构造中常数因子和参数设置相对繁琐。

---

## 100. MAC-v2: Long-horizon model-based value expansion from pixels

**arXiv ID:** 2611.07245 | [PDF](https://arxiv.org/pdf/2611.07245v1)

**作者:** Kwanyoung Park `[一作]` (University Of California Berkeley), Sergey Levine `[通讯]` (University Of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于像素的离线强化学习框架，利用无重建的潜在世界模型与动作块化（action chunking）实现长时间步模型扩展，从而实现长时序的价值估计。

**💡 创新点**

创新点在于将无重建潜在表示与动作块化结合：潜在表示只通过预测与控制目标学习，剔除无关视觉细节；动作块化减少递归预测次数，降低错误累积；两者共同实现 100+ 步的可观想（rollout）并显著提升离线 RL 性能。

**🔧 技术方法**

核心技术包括：
- 无重建潜在世界模型（Latent Dynamics + Reward 预测）
- 行为克隆（Behavior Cloning）与流匹配（Flow Matching）相结合的策略学习
- 动作块化（Action Chunking）与拒绝采样（Rejection Sampling）实现的价值函数训练
- 组合使用分布式奖励预测与分布式价值函数（distributional V/Q）

**📊 数据集**

在四类视觉离线 RL 基准上评估：
- OGBench（25 个单任务）
- MMBench2（10 个未见任务）
- Robomimic（3 个机械臂任务）
- LIBERO‑10（10 个长时序任务）

**📈 对比分析**

与多种强基线对比（IQL, ReBRAC, FQL, LOMPO, OfflineDV2, QC 等）。在 OGBench 上 4/5 任务族上超越所有基线，在 MMBench2 上与最佳基线相当或更优；在 Robomimic 上成为唯一能够持续匹配或超过行为克隆（BC）的算法；在 LIBERO‑10 上保持与 BC 竞争力但提升空间有限。整体表现显示长时序模型扩展在视觉离线 RL 上的有效性。

**⚠️ 局限性**

局限性包括：
- 主要在单任务模拟环境中验证，缺乏多任务或真实世界场景的实验
- 动作块化大小固定，未实现自适应或层次化块化，可能影响对快速变化情境的响应
- 对高视觉多样性的任务泛化能力待进一步验证
- 对极端分布漂移或极大数据规模的鲁棒性未做深入探讨

---

## 101. Neural Compression over Equivalence Classes: Exact Gauges and Functional Rate-Distortion

**arXiv ID:** 2611.07233 | [PDF](https://arxiv.org/pdf/2611.07233v1)

**作者:** Toshiaki Koike-Akino `[一作]` `[通讯]`, Toshiaki Koike-Akino

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过搜索神经网络的函数保持变换（如通道置换、尺度变换、低秩因子变换）以及在允许的功能失真范围内的微调，直接对原始模型参数进行重构，从而在不改变模型功能的前提下实现压缩。

**💡 创新点**

创新点在于：①把压缩目标直接定义为实际物理压缩器（如 Zstd、gzip 等）返回的字节数，而非传统的熵模型或可微率近似；②构建了“提案‑审核‑测量”三步管道，在保持功能或在可接受失真阈值下搜索可行表示；③在多个模型规模和架构上验证了该方法的跨编解码器迁移性，展示了从 KB 级别的 exact‑gauge 到 MB 级别的功能邻域压缩幅度。

**🔧 技术方法**

使用的技术包括：函数保持变换的数学族（通道置换、SwiGLU 缩放、值/输出的 Γ‑变换、低秩因子等）、基于显式功能审核（输出误差、KL 散度）与数值误差控制的精度层级；多位权重提案（显著位舍入 Qb、ℓ1 稀疏化）；以及基于物理编解码器的离线字节计数评估和增量搜索。

**📊 数据集**

主要数据集：WikiText‑2 用于校准；WikiText‑test、C4 流式验证用于离线功能审核；下游任务包括 LAMBADA、PIQA、ARC‑Easy、HellaSwag 等，用于评估压缩后模型的实际性能。

**📈 对比分析**

与原始模型和不同压缩器（Zstandard、gzip、Brotli、ZipNN）直接比较。结果显示：exact‑gauge 在 SmolLM2‑135M 上可获得约 14 KB（≈0.5%）压缩；在功能邻域下，SmolLM2‑360M 的 Zstandard 缩小 12.8 MB（≈6%），在 Qwen3‑0.6B 上达到 4.8%；跨编解码器时压缩率保持相对一致（≈20–23%）。在保持 1–2% 的 KL 或 NLL 失真时，压缩比例可达 23%。

**⚠️ 局限性**

限制包括：①自然训练的模型或 LoRA 轨迹对某些函数保持变换的自由度有限，导致压缩收益不大；②只在有限的校准/验证样本上评估功能失真，无法保证在更广泛或长序列生成任务中的稳健性；③未探索针对更复杂编解码器或混合量化/稀疏化的联合优化；④压缩结果高度依赖于所选的失真度量和阈值，缺乏统一的评估标准。

---

## 102. Spatial-UMI-Dream: Scaling World Action Models with Metric Geometry for UMI Manipulation

**arXiv ID:** 2611.07229 | [PDF](https://arxiv.org/pdf/2611.07229v1)

**作者:** Weiduo Yuan `[一作]` (USC Physical Superintelligence Lab), Yue Wang `[通讯]` (USC Physical Superintelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了一种5B规模的空间世界-动作模型（WAM），融合了UMI手持双臂摄像头收集的RGB-D数据，并通过两阶段预训练实现跨机器人和跨环境的学习，并基于RoboTwin构建了20任务基准用于系统评估。

**💡 创新点**

创新点包括：①将精确的度量深度引入UMI视觉框架，实现对场景尺度和空间关系的显式建模；②采用两阶段预训练（先世界预测再联合动作学习）提升跨环境和跨实体的泛化能力；③提供可复现的跨机器人/环境基准，方便对UMI策略进行分离与联合泛化测试。

**🔧 技术方法**

技术手段包括：基于Wan2.2扩散变压器的世界-动作框架；立体匹配得到的度量深度和三通道深度编码；双臂RGB-D画布与视频VAE的统一编码；教师强迫与流匹配损失；指令补偿模型；以及在模拟中使用四视角合成的鱼眼图像。

**📊 数据集**

使用的主要数据集为UMI手持双臂收集的RGB-D轨迹（数千小时），以及基于RoboTwin的20项仿真任务数据和六项真实世界双臂任务演示。

**📈 对比分析**

与多种VLA和WAM基线（如VLA、UMI‑VLA、WAM‑X等）对比，模型在RoboTwin基准上达79.10%成功率，超过基线4.65个百分点；在六项真实任务中平均成功率为85%，高于最强基线24.2个百分点；在跨机器人+环境组合偏移下实现72.15%成功率，领先基线10.3个百分点。

**⚠️ 局限性**

局限性在于依赖高质量的鱼眼深度重建，对动态或光照极端变化的鲁棒性仍有限；两阶段预训练与大规模模型训练成本高；跨实体泛化虽有提升但仍存在性能下降；评估范围局限于UMI数据和预定义任务，未检验对其他机器人或更广泛任务的适用性。

---

## 103. Encoding and Propagation of Grammatical Number in Large Language Models

**arXiv ID:** 2611.07283 | [PDF](https://arxiv.org/pdf/2611.07283v1)

**作者:** Jacob Solinsky `[一作]` (University of Minnesota), Serguei Pakhomov `[通讯]` (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用稀疏自编码器（SAE）对LLM的隐藏状态进行解码，重点研究英语主谓数一致性的内部机制；通过激活补丁（AP）和因果框架量化特征与输出之间的关系。

**💡 创新点**

创新点：① 将SAE与行为主义因果框架结合，提出单数偏好度量B并用Cliff's Δ筛选重要特征；② 识别出Pythia-70M等模型使用隐性/主数标记的特征，并绘制多跳和0跳电路；③ 在不同规模模型上验证方法的可迁移性。

**🔧 技术方法**

技术手段：TopK和Matryoshka SAEs、激活补丁（AP）、梯度筛选、Cliff's Δ、Pearl因果图、单数偏好度量B、对比电路图分析；使用的模型包括Pythia‑70M、Gemma‑2‑2B、Llama‑3.1‑8B。

**📊 数据集**

数据集：Pythia‑70M训练用的OpenWebText、人工构造的短语/矩阵主语/嵌入子句句对（共200句）以及自生成的11词长句；所有句子保持相同长度以便批量计算。

**📈 对比分析**

评估方法：通过对SAE的方差解释率（80‑90%）评估重构质量；使用B度量和Cliff's Δ筛选特征并绘制因果边；在三类数据集上比较电路复杂度；在Gemma和Llama上复现结果，证明方法在更大模型上有效。

**⚠️ 局限性**

局限性：① SAEs对Query/Key向量信息捕获效果差，无法分析注意力头内部机制；② 需要多轮筛选和手工阈值，计算成本高；③ 仅聚焦数一致性，未扩展到其他句法现象；④ 句子长度固定、token化细粒度可能影响特征识别；⑤ 对极大模型的可扩展性虽有初步验证，但仍需更高效的特征搜索策略。

---

## 104. Scenario Augmentation for Rare Financial Events via Multimodal Latent Diffusion

**arXiv ID:** 2611.07282 | [PDF](https://arxiv.org/pdf/2611.07282v1)

**作者:** Mahdi Arab Loodaricheh `[一作]` (Graduate Center CUNY), Indranil SenGupta `[通讯]` (Hunter College CUNY)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于多模态潜在扩散的极端金融事件数据增强框架，用于在罕见市场窗口下改进CVaR标签的表示与分类。

**💡 创新点**

创新点在于：①将结构化表格、空间图像和时间序列三种金融信息通过专属VAE编码并对齐；②使用CVaR定义的罕见事件标签进行条件扩散生成；③在多模态潜在空间上训练类条件扩散模型，既提升生成质量又保持风控相关特征。

**🔧 技术方法**

主要技术包括：Transformer‑VAE、卷积‑VAE、双向LSTM‑VAE用于各模态编码；潜在空间对齐与归一化；EDM风格的类条件扩散模型；XGBoost分类器用于下游评估。

**📊 数据集**

数据集为美国科技股八只主要公司的日交易数据，构造5~8天窗口，包含开盘价、收盘价、成交量、夜间价差及等权组合收益。

**📈 对比分析**

与TabDDPM和TabSyn两种基准表格扩散器对比，使用PRDC指标衡量生成质量、XGBoost AUC评估分类效果。结果显示本方法在Precision、Recall、Coverage上表现最优，生成的样本在Synthetic‑to‑Real和Real+Synth‑to‑Real实验中AUC提升至0.953–0.960，均超过基准。

**⚠️ 局限性**

局限性包括：仅在美国科技股上验证，未考察更广泛市场；扩散训练与VAE训练较耗时；对模态对齐的假设在高度异质数据中可能不稳健。

---

## 105. Degree profiles of uniform polynomial consequences: Linear lower bounds for Nullstellensatz and Polynomial Calculus

**arXiv ID:** 2611.07280 | [PDF](https://arxiv.org/pdf/2611.07280v1)

**作者:** Søren Riis `[一作]` `[通讯]`, Søren Riis

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在所有正特征域上均匀多项式系统的 Nullstellensatz 与 Polynomial Calculus 推导所需的最低度数达线性下界，并将统一后果的度数行为分为四类（常数、线性、周期混合、波动峰）。

**💡 创新点**

首次给出完全互斥的四种度数模式分类，证明线性度数下界是最优的，并在 Lean 中形式化验证；同时揭示正特征下 p‑赤子硬度谱与不可判定性。

**🔧 技术方法**

采用置换模块表示稳定性、度数保持提升与删除标签的精确投影、p‑赤子硬度分析、以及对称群代数中的内逆与投影构造等技术。

**📊 数据集**

使用由固定模板生成的均匀多项式系统（如鸽巢原理、计数原理、第一阶句子翻译等）作为实验对象；所有系数规则均为统一、固定。

**📈 对比分析**

通过证明在足够大的宇宙尺寸上度数上限对应的证明空间呈周期性，得到难点处证明大小指数级（2^Ω(n)），易点处多项式级；未给出匹配的上界。

**⚠️ 局限性**

仅适用于正特征域和固定模板，未给出上界；无法处理非均匀或动态度数的系统；证明区分类型是不可判定的。

---

## 106. Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging

**arXiv ID:** 2611.07275 | [PDF](https://arxiv.org/pdf/2611.07275v1)

**作者:** Yuanchu Liang `[一作]` (Australian National University), Hanna Kurniawati `[通讯]` (Australian National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Latent Noise Linear MDP (LNL‑MDP) 框架并设计 LARBRID 算法，利用预训练的扩散/流模型的噪声空间进行微调，仅冻结模型参数，避免反向传播到生成网络。

**💡 创新点**

创新点在于：①通过桥接方程解决噪声别名问题，使动作空间与噪声空间共享同一表征；②在 LNL‑MDP 下给出近似最优的样本复杂度理论；③提出可在在线、离线以及离线‑>在线多种设置下应用的统一算法 LARBRID/LARBRID‑I。

**🔧 技术方法**

技术手段包括：线性 MDP 理论、桥接方程、扩散谱表征学习、SAC、FQL、FineTuneRL、噪声匹配和能量基模型。

**📊 数据集**

使用的数据集有：Gymnasium、Robomimic、Ogbench（10 种运动与操纵任务）、视觉任务、Libero、Aloha 等多种机器人控制和仿真环境。

**📈 对比分析**

与 DSRL、DPPO、IDQL、IQL、REBRAC、FQL 等基线进行对比，LARBRID 在 Gymnasium 与 Robomimic 的在线任务中提升约 21%+ 奖励，Transport 任务提升约 70%；在离线任务中 7/10 任务上优于多数基线，最高提升 3 倍；离线‑>在线任务中 LARBRID‑I 在大多数任务上获得更高成功率。

**⚠️ 局限性**

局限性包括：对复杂任务（如 humanoid maze）可能陷入局部最优特征导致性能下降；对预训练模型覆盖度和表征质量敏感；训练速度比部分现有方法略慢。

---

## 107. How Deep is the Signal? Time-Context up to 30 Years for Colorectal Cancer Prediction in a Primary-Care EHR Model

**arXiv ID:** 2611.07265 | [PDF](https://arxiv.org/pdf/2611.07265v1)

**作者:** Achille Raviez `[一作]` (Imperial College London), Brendan Delaney `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究引入了时间上下文的概念，系统评估了十年尺度的时间上下文对电子健康记录（EHR）基础模型的影响，特别是在预测结直肠癌方面。

**💡 创新点**

创新点在于首次将时间上下文作为设计轴进行评估，并通过十年尺度的预测信号深度曲线进行验证，强调了时间上下文与令牌上下文的区别。

**🔧 技术方法**

使用了一种仅解码的变换器模型（34.6M参数），并进行了微调以预测结直肠癌。

**📊 数据集**

使用的数据集为来自伦敦西北部的Whole Systems Integrated Care (WSIC)初级护理数据库，包含1.4亿个编码事件和360万名患者的记录。

**📈 对比分析**

与XGBoost基线模型进行比较，AUROC从2年的0.551提高到18年的0.590，然后趋于平稳，XGBoost的峰值AUROC为0.589，显示出时间上下文的有效性。

**⚠️ 局限性**

限制在于数据集的可用性，Discover数据集并未公开，且研究仅限于特定的患者群体，可能影响结果的普遍性。

---

## 108. Evolutionary Adaptive Neighborhood Learning: A Genetic Algorithm for Instance-Dependent k Optimization in Nearest Neighbor Classification

**arXiv ID:** 2611.07257 | [PDF](https://arxiv.org/pdf/2611.07257v1)

**作者:** Isaac Touza `[一作]` (University of Maroua), Kolyang `[通讯]` (University of Maroua)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了EANK（Evolutionary Adaptive Neighborhood K）方法，利用遗传算法学习基于局部邻域统计的实例依赖k值策略，替代传统的全局k搜索。

**💡 创新点**

核心创新在于把k的调参转化为学习一个低维可解释的实例级策略；通过预计算投票表实现每次适应度评估仅需O(n)；并给出无样本量依赖的泛化界限。

**🔧 技术方法**

采用遗传算法（SBX交叉、Gaussian变异、精英保留）、留一交叉验证、kNN投票、七维邻域元特征、投票表加速、VC理论泛化界限和统计显著性检验等技术。

**📊 数据集**

使用12个公开分类数据集（Iris、Wine、BreastCancer、Digits及8个UCI数据集）以及一个人工合成的空间异质性数据集进行实验。

**📈 对比分析**

与默认k=5、全局k最优、加权k、GA单k、局部k两阶段等6种基线进行对比；在12个数据集上平均排名第二，仅比k=5显著；在合成实验中显著优于全局k，测试集提升约0.01%，但在部分数据集上略逊。

**⚠️ 局限性**

主要局限包括：LOO准确率作为训练目标易产生选择偏误，泛化能力受限；策略为单调线性，无法捕捉更复杂关系；适用于中小规模数据，邻域搜索成本高；仅优化准确率，未考虑不平衡或校准；需要额外的特征归一化和参数设置。

---

## 109. $L_1$ versus $L_2$ Clustering: When Does the Choice of Norm Matter?

**arXiv ID:** 2611.07248 | [PDF](https://arxiv.org/pdf/2611.07248v1)

**作者:** Nick Gravin `[一作]` (Shanghai University of Finance and Economics), Jianhao Jia `[通讯]` (University of Hong Kong)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在随机坐标系统下，k聚类的L1（曼哈顿）与L2（欧几里得）目标的近似保证之间的转换关系；

**💡 创新点**

首次证明在随机旋转的平均情形下，β-近似L1聚类对L2仅需O(β)近似，并给出与k、d无关的√(log k / loglog k)上界，且提供匹配的下界；

**🔧 技术方法**

采用几何投影、随机旋转概率分析、核心点/小欧几里得核心集、方向集合控制与尾部概率积分等技术；

**📊 数据集**

理论分析主要使用构造的k+1点特殊实例来实现下界，并未依赖实际数据集；

**📈 对比分析**

通过在同一数据集上随机旋转后比较两种目标的成本比率，得到高概率常数近似或期望O(β min{√d, k²/d})的性能，最坏情况约为Θ(√(log k / loglog k))；

**⚠️ 局限性**

对固定维度时k¹/d与k²/d之间仍存在未收敛的差距，且不适用于平方欧几里得（k‑means）与L1的比较。

---

## 110. Goal Revelation as a Bridge Between Information-Theoretic Empowerment and Capability-Based Human Power

**arXiv ID:** 2611.07247 | [PDF](https://arxiv.org/pdf/2611.07247v1)

**作者:** Jaime Ruiz-Serra `[一作]` (University of Sydney), Jobst Heitzig `[通讯]` (Potsdam Institute for Climate Impact Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究并建立了人类在智能体交互中的三种功率衡量指标（能力、目标揭示、赋权）之间的量化关系，证明了能力向揭示的下界，并揭示了赋权与揭示之间的差距。

**💡 创新点**

提出了利用Fano不等式将个体功率转化为目标揭示的下界，首次阐明赋权不一定带来揭示，揭示不一定带来实现，并给出满意度下界来监测已部署系统中的人类功率。

**🔧 技术方法**

使用信息理论（互信息、定向信息、马尔可夫游戏框架）、Fano不等式、赋权容量计算、能力递归以及实验模拟等技术。

**📊 数据集**

主要使用随机生成的一步马尔可夫决策过程（随机多臂赌博机）进行模拟；实验中还设计了恶意环境用于验证容量/揭示失效。

**📈 对比分析**

将三种指标在同一状态下计算，证明满足链式顺序 ≤I_h ≤I(A→|s) ≤E(s)，实验显示它们高度相关但不等价，排名一致度在0.58–0.75之间，最优状态的确定仅略优于随机。

**⚠️ 局限性**

仅适用于完全可观测、有限窗口、离散状态；假设独立目标族并需参考先验；能力是反事实的，需对人类策略建模；对目标重排不变的指标不提供正向保证。

---

## 111. Adaptive Hybrid Force-Vision Control with EMG-Derived References for Deformable Bottle Manipulation

**arXiv ID:** 2611.07235 | [PDF](https://arxiv.org/pdf/2611.07235v1)

**作者:** Jeffrey Lee `[一作]` (University of Waterloo), Yue Hu `[通讯]` (University of Waterloo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研发了一种自适应混合力-视差控制器，实现了机器人在实验室中对可变形压缩瓶的精准液体分配。

**💡 创新点**

创新点在于将人体肌电信号映射为抓取力参考，并将该参考与力传感器和视觉测量的瓶体变形信息动态融合，从而在不同压缩次数、瓶型和目标体积下实现人类示范的逼真重现。

**🔧 技术方法**

采用了表面肌电（sEMG）信号处理、多层感知机映射、FSR力反馈、Intel RealSense深度相机的点云变形估计，以及自适应比例增益与混合因子σ的控制算法。

**📊 数据集**

实验数据由单名实验者在TIAGo Pro机器人上完成的24条人类压缩示范（12种条件×2次演示）收集，配合5次机器人重复试验。

**📈 对比分析**

与纯力控制器对比，混合控制器在12种条件下的平均绝对误差从15.92 mL降至6.54 mL，均方根误差亦相应下降；标准差保持在≈3 mL范围内，显示出更高的准确性和可重复性。

**⚠️ 局限性**

局限性包括：视觉在70 %以上变形时精度下降，限制了最大闭合位置；实验仅针对塑料瓶、恒定液体黏度，未验证不同材质或玻璃器皿；以及对更复杂实验室设备的适用性尚需进一步研究。

---

## 112. Archetypal contexts in the neurons of language models: an architecture for intelligence?

**arXiv ID:** 2611.07234 | [PDF](https://arxiv.org/pdf/2611.07234v1)

**作者:** David Nordfors `[一作]` `[通讯]`, David Nordfors

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探索并验证大型语言模型（如Qwen3.8-27B）内部是否存在“原型关系模式”与主题域分离的结构，并通过神经元映射与消除实验进行验证。

**💡 创新点**

首次发现并证明语言模型呈现三层“语言-主题-原型”层级结构（沙盒），其中原型模式被单独存储、可跨主题跨语言迁移；同时通过神经元消除提供双分离的定量证据。

**🔧 技术方法**

使用神经元激活/权重映射、神经元消除（遮罩）实验以及跨模型可解释性分析等技术，对Qwen3.8-27B及30个不同规模、不同族群模型进行研究。

**📊 数据集**

主要使用作者自行构造的12篇多主题、多语言文本（涵盖四个原型和多领域）以及用于映射的更大文本语料。

**📈 对比分析**

通过对特定原型或主题神经元组进行消除，观察模型回答是否丢失对应模式或主题；实验结果在图表中展示，四个原型在五个领域均通过消除实验成功验证双分离。

**⚠️ 局限性**

局限性包括：仅在人工设计的文本上验证，缺乏野生文本的评估；消除工具相对粗糙（仅遮罩一半权重）；未对不同原型频率或重要性进行量化；未探讨模型在其他维度（如情感、时态）上的类似结构。

---

## 113. Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost

**arXiv ID:** 2611.07232 | [PDF](https://arxiv.org/pdf/2611.07232v1)

**作者:** Ruitian Wang `[一作]` (University of California, San Diego), Hao Zhang `[通讯]` (University of California, San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Recursive Sparse Attention（递归稀疏注意力），通过在 KV 缓存上构建树形结构进行分层选择，并在每层用单个键和值摘要来近似未被选中的子树，实现百万 token 上下文的日志级计算成本；

**💡 创新点**

创新点包括：1）树形分层选择与统一的候选本地 RoPE，保证相对距离始终在训练范围内；2）共享池化器（pooler）从密集注意力中蒸馏得到，支持跨层和跨 KV 头使用；3）高效的 prefill 与解码内核设计，支持 KV 缓存在 host 上的 offloading；4）仅在原始 32K 训练长度下训练，便于跨长度外推；

**🔧 技术方法**

使用的技术包括：稀疏注意力与树形分层选择、候选本地 RoPE、池化器蒸馏与 KL 损失、GQA（Grouped‑Query Attention）、FlashAttention‑2 与自定义 Triton/CUDA kernel、KV 缓存 offloading、以及标准的语言模型训练与评估流程；

**📊 数据集**

使用的数据集有：Qwen3‑4B 预训练数据（约 0.62B tokens，32K 长度训练），RULER 合成长上下文任务、PG‑19 书籍文本用于 perplexity 评估、LongBench 与 En.MC 长文档任务、以及 MMLU、ARC‑C、GSM8K、HumanEval、MBPP 等短上下文基准；

**📈 对比分析**

与 Dense+YaRN/DCA、MoBA、DSA、InfLLM、InfiniteHiP 等基线在 RULER、perplexity、长文档与短文档任务上进行对比；结果显示 RecSA 在 131K–2M 范围内始终领先（例如 1M 上 60.5 分对比 Dense+DCA 40.0 分），并在 64K–2M 的预填和解码阶段速度提升至 3.2×/3.8×（如 524K 上 80s vs 257s dense），解码延迟保持在 8.35 ms/10.7 ms/token；

**⚠️ 局限性**

局限性包括：需要精心蒸馏与调参的 pooler，KV 缓存 offloading 对 GPU 内存和 PCIe 带宽要求高；方法对 RoPE 的依赖意味着在非 RoPE 模型上效果未知；目前仅在 Qwen3‑4B 规模上验证，规模可扩展性及在其他任务/模型上的泛化尚待进一步评估；

---

## 114. Efficient Projection-Based Reduced-Order Modeling for Computational Fluid Dynamics via Multifidelity Basis Construction

**arXiv ID:** 2611.07231 | [PDF](https://arxiv.org/pdf/2611.07231v1)

**作者:** Ashok Viswanathan `[一作]` (Stanford University), Charbel Farhat `[通讯]` (Stanford University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于物理的多精度方法，用少量高保真（RANS）快照与大量低保真（Euler）快照构建PROM，显著降低离线建模成本。

**💡 创新点**

创新点在于通过对齐、差异校正基（correction ROB）和HF残差驱动的残差最小化实现低保真预测的高保真修正，并采用RANS平滑筛选采样。

**🔧 技术方法**

使用投影式模型降阶、奇异值分解、残差基贪婪采样、线性插值与状态补全等技术。

**📊 数据集**

实验数据集为NACA0012翼型在稳态亚声速到超声速（Mach 0.4–0.65）和攻角0–5°的转子流场。

**📈 对比分析**

与仅用高保真快照构建的HF-PROM比较，MF-PROM4在相同40维基底下平均误差分别为升力1.1%、阻力3.6%、升阻比3.9%，而HF-PROM误差分别为0.18%、2.9%、2.8%；离线快照生成速度提升约7.8倍。

**⚠️ 局限性**

局限包括：在高维基底（>20）时升力误差不降，无法很好捕捉参数相关的冲击移动；非单调的HF初始化大小与预测精度关系；仅验证二维稳态流场，需进一步扩展至三维/非稳态。

---

## 115. SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection

**arXiv ID:** 2611.07222 | [PDF](https://arxiv.org/pdf/2611.07222v1)

**作者:** Woosang Lim `[一作]` (SAINT Machines), Jaewoo Lee `[通讯]` (Chung-Ang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 SPILLWAY 的防御框架，用于保护使用工具的语言模型代理，防止注入攻击导致代理将动作转发给攻击者。

**💡 创新点**

创新点在于将目标接收与前置消毒两步拆分：①通过对工具返回的类型化身份字段做集合查找来决定是否允许动作（不依赖模型）；②在模型读取输入前使用无模型规则消毒，移除指令式文本，随后仅在被拒绝的调用中使用本地裁决器恢复合法请求。

**🔧 技术方法**

技术包括：类型化身份索引（Typed Identity Index）、基于规则的无模型消毒器、基于两视图的本地裁决器、固定工具角色词典、以及对已消毒文本的 span 证据追踪。

**📊 数据集**

数据集主要是 AgentDojo v1.2.2 的四个任务套件（Banking、Slack、Travel、Workspace）以及三种 LLM 主干（gpt‑4o‑mini、qwen3‑coder:30b、qwen3:8b）。此外还评估了 ChatInject、AutoDojo 优化注入、以及公开的 RL 攻击者 PISmith。

**📈 对比分析**

对比方法包括未加防御、MELON、Progent、DRIFT、ROPE（作者版和自用版）、Spotlighting、PI 检测器等。SPILLWAY 在所有 12 个主干×套件组合中均实现了零在域内攻击成功率，且在攻击下的平均效能（BU/UAU）和综合指标（NRP、ERS）均高于或接近最佳基线，尤其在标准攻击、ChatInject、AutoDojo、以及 PISmith 上表现优异。

**⚠️ 局限性**

局限性：只针对身份目标（Identity Targets）做安全检查，无法防御未出现文字的目标或攻击者直接写入类型化字段；需要为每个工具集手工声明角色词典；对于被消毒文本中的目标，仍可能被裁决器误判；若攻击者能够在工具返回的类型化字段中植入恶意值，则安全保障失效。

---

## 116. SAFE-Teen: Evaluating Chatbot Safety Across Adolescent Mental Health Conversations

**arXiv ID:** 2611.07218 | [PDF](https://arxiv.org/pdf/2611.07218v1)

**作者:** Andrea Mock `[一作]` (Vals AI), Caroline Figueroa `[通讯]` (Vals AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对18种聊天机器人API在模拟青少年对话中进行安全评估，构建了SAFE-Teen框架，比较了两种系统提示对安全性的影响。

**💡 创新点**

提出基于临床专家的多轮对话安全评估方法，使用匹配的情景分支比较青少年不同回应对安全的影响。

**🔧 技术方法**

利用大语言模型（LLM）生成青少年模拟用户、使用Claude Sonnet 5.5作为自动评判器，结合22项安全准则进行评测。

**📊 数据集**

构建106个临床专家编写的种子情景，扩展至983个对话变体，生成32,555个模拟对话。

**📈 对比分析**

通过对比无提示与含青少年提示两种系统提示下的失败率，结果提示将总体失败率从43.3%降至24.9%，但在间接自杀风险和诊断等方面仍高达52%。

**⚠️ 局限性**

模拟对话缺乏真实青少年交互，情景覆盖有限，评判器和推理文本可能存在误差，且仅测试了18个API。

---

## 117. Compositional Unalignment: Auditing Safety in Large Vision-Language Models

**arXiv ID:** 2611.07216 | [PDF](https://arxiv.org/pdf/2611.07216v1)

**作者:** Hong kyu Lee `[一作]` (Emory University), Li Xiong `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的 LVLM 组合安全未对齐方法 CMaG，用于检测并削弱模型在组合伤害场景下的拒绝行为。

**💡 创新点**

创新点在于利用拒绝与合规对比的梯度差异来精确提取跨模态交互方向，从而实现对安全机制的直接干预，解决传统方向搜索方法无法有效未对齐的问题。

**🔧 技术方法**

核心技术包括拒绝边际计算、梯度对比、SVD 提取单维方向、层级比例缩放与基于单步反向传播的推断。

**📊 数据集**

使用了 SIUO 组合伤害基准、Synthetic Alignment Set（Qwen2.5-VL-7B / LLaVA-1.5-7B）以及 VLGuard 文本显式伤害数据集进行评估。

**📈 对比分析**

与 SFT、Pragma‑VL、GrAInS 与 Attribution Patching 等基线相比，CMaG 在 SIUO 上将安全分数下降约 65 分，同时保持 98%+ 文本连贯性；对已防御的 LAT 训练模型亦能显著降低安全分数。

**⚠️ 局限性**

局限性包括：对组合伤害与文本显式伤害关系的机制尚不完全阐明；方法仅针对 LVLM 设计，跨模态或更大模型的泛化尚待验证；过度调节参数可能导致连贯性下降。

---

## 118. Canonical Correlation Neural Networks

**arXiv ID:** 2611.07212 | [PDF](https://arxiv.org/pdf/2611.07212v1)

**作者:** Andrea Cavallo `[一作]` (Delft University of Technology), Elvin Isufi `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

设计并实现了一种名为CANE的两视图神经网络，利用视图内协方差和跨视图协方差作为关系性先验，直接训练下游任务。

**💡 创新点**

将CCA的谱分析融入可学习的多阶多视图图卷积，避免显式特征提取与矩阵逆/奇异值分解，提供理论稳定性分析和端到端训练。

**🔧 技术方法**

采用协方差图卷积、可学习多项式滤波器、谱分析、Lipschitz稳定性证明以及端到端任务损失等技术。

**📊 数据集**

在七个真实世界数据集上评估，涵盖脑部MRI（ADNI、ABIDE）用于年龄预测和运动传感器（MHEALTH、REALDISP、DSADS）用于动作分类。

**📈 对比分析**

与单视图、多视图、VNN、CCA、RCCA、PLS、KCCA、DCCA、SDGCCA、TOCCA等基线在相同任务下进行交叉验证比较，CANE在6/7数据集上获得最佳或接近最佳效果，并且参数更少。

**⚠️ 局限性**

目前仅支持两视图，缺乏多视图扩展；在高维数据中协方差估计仍是瓶颈；理论分析局限于单视图和有限样本。

---

## 119. Not All Loops Cost the Same: Phase-Dependent Depth for Serving Looped Transformers

**arXiv ID:** 2611.07211 | [PDF](https://arxiv.org/pdf/2611.07211v1)

**作者:** Zhixu Du `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoopDuet，一种无需训练的循环 transformer 推理方案，联合选择前置（prefilling）和解码阶段的循环深度，并通过批量化递归执行与队列年龄控制实现高效推理。

**💡 创新点**

创新点在于：① 将前置和解码阶段分离深度决策；② 在执行时聚合已完成前置位置的查询，节省计算；③ 通过队列年龄动态调节前置深度，以在不同负载下保持质量与吞吐的最佳平衡。

**🔧 技术方法**

采用共享核心权重的循环 transformer 架构、vLLM 变体实现的批量化递归执行、队列年龄控制器以及基于 KV 缓存的聚合与散列技术。

**📊 数据集**

在 HotpotQA、GSM8K、ARC-Challenge 等基准数据集上进行评估，使用 Huginn-0125、OLMo-2 等复现模型。

**📈 对比分析**

与统一深度配置对比，LoopDuet 在 Huginn 上实现 19% 吞吐提升、6.04 EM 点提升；在 OLMo 上通过动态前置深度控制在 60 请求/秒负载下提升 0.92–1.35 EM 点，且保持 100% 的 1 秒首词时间满足率。

**⚠️ 局限性**

局限性包括：① 需要离线特征化以确定前置深度范围，无法自适应任务外的质量变化；② 仅在前置与解码深度差异显著的任务中显著受益；③ 对极短 prompt 或长生成任务的收益有限。

---

## 120. A Complete Equational Theory for Quantum Circuits with Generalized Control

**arXiv ID:** 2611.07207 | [PDF](https://arxiv.org/pdf/2611.07207v1)

**作者:** William Schober `[一作]` (Università della Svizzera italiana), Scott Wesley `[通讯]` (Dalhousie University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了泛化控制积(⊙)和电路指数化，并构造了层次化量子电路语言，给出了其完整的等价理论。

**💡 创新点**

将标准控制的对称泛化与指数化统一到一种新的prop构造中，首次实现层次化语义的完全等价推理。

**🔧 技术方法**

使用对称单子范畴、dagger prop、控制 functor、矩阵指数与对数、图形化推理以及自由 prop 与等价闭包等范畴代数技术。

**📊 数据集**

未使用实验数据集；在理论证明中以 Pauli、Hadamard、旋转门等数学实例为验证对象。

**📈 对比分析**

通过符号化等价推理与已有的标准控制电路进行对比，证明了可重写与优化的有效性；未给出实验性能指标。

**⚠️ 局限性**

控制积与指数化非双函子，指数化在非 Hermitian 情况下不满足群作用；实际编译与优化实现尚未完成，理论复杂度高。

---

## 121. Capability Gain from ISA to Shared Execution on AMD MI300A and MI250X

**arXiv ID:** 2611.07198 | [PDF](https://arxiv.org/pdf/2611.07198v1)

**作者:** Aaron Jarmusch `[一作]`, Sunita Chandrasekaran `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对AMD MI300A（CDNA3）与MI250X（CDNA2）两代GPU进行全面的功能可行性追踪，构建“Capability Gain”指标，从指令层、库层、应用层到多流共享设备四个层级评估FP8和2:4稀疏的实际加速与理论差距。

**💡 创新点**

创新点包括：①提出Capability Gain度量并在跨代、跨层级系统地追踪；②将硬件队列池与精度无关的公平性关联；③构建稀疏的break‑even模型；④在同一协议下对两代GPU进行共享实验，揭示队列数而非精度决定共享公平性。

**🔧 技术方法**

使用AMD MI300A、MI250X两代GPU；ROCm生态（hipBLASLt、hipSPARSELt、rocBLAS、rocHPL-MxP、vLLM、PyTorch）；自定义指令微基准、GEMM库基准、多流基准、LLM推理/训练、HPL求解、STREAM Triad；硬件计数器与时钟测量。

**📊 数据集**

基准数据集：LLM使用WikiText‑2文本；HPL使用随机生成矩阵；GEMM形状从1024^3到16384^3，覆盖MLP和薄K形状；多流实验使用同一GEMM形状与不同流数；LLM推理/服务使用多请求混合长度轨迹。

**📈 对比分析**

比较方法：在统一输入与驱动版本下，分别测量指令级、库级、应用级和共享级的加速比；Capability Gain = 加速比 / 指令级加速比；多代比较时除以硬件CU数、时钟比；共享层通过Jain指数、系统吞吐量(STP)和平均归一化周转时间(ANTT)评估公平与吞吐；实验结果显示FP8在库层能保留≈70–90%指令加速，MI300A的8192^3 GEMM比MI250X FP16快≈1.6×；2:4稀疏在hipSPARSELt中大多失去加速，只有在大列宽下才可击败密集；在共享多流下，FP8的Capability Gain可超过1，说明队列共享与功耗管理带来的额外加速；LLM推理/服务在未融合FP8转换时几乎没有收益，只有在vLLM等融合方案中才显著提升。

**⚠️ 局限性**

局限性：仅覆盖AMD MI300A/MI250X两代GPU；实验未覆盖NVIDIA等竞品；对FP8/稀疏的精度损失分析有限，仅使用WikiText‑2；缺乏大规模多机或多GPU环境；稀疏压缩成本未完全量化；部分加速依赖未充分验证的DVFS假设；最终应用中的优化策略（如量化融合、稀疏fine‑tune）未实现。

---

## 122. Graph Neural Networks for Resource Allocation in Integrated Sensing and Communication

**arXiv ID:** 2611.07277 | [PDF](https://arxiv.org/pdf/2611.07277v1)

**作者:** Junyu Liao `[一作]` (University of Melbourne), Jamie Evans `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究车联网ISAC（集成感知与通信）中的功率分配问题，提出了一种基于图神经网络（GNN-ISAC）的无监督学习方法，用以在有限功率预算下同时满足感知质量与通信速率的最优化。

**💡 创新点**

创新点在于：①设计了两种不同的边类型（干扰边与功率预算边），使图能同时建模同通道干扰与同供应商功率共享的耦合；②通过感知惩罚的无监督损失实现对感知SINR约束的软约束；③证明该GNN在未重新训练的情况下即可在更大规模网络中保持高达95%+的性能。

**🔧 技术方法**

使用技术包括：图神经网络（消息传递与更新MLP），无监督学习（感知-通信损失），改进的WMMSE（感知功率预留），Per-Provider softmax分配，ReLU激活及Adam优化。

**📊 数据集**

数据集为仿真生成的车联网场景：服务提供者随机放置于225×225 m²区域，感知目标与通信接收机随机距离2–5 m，采用2.4 GHz载波、5 MHz带宽，噪声功率6.29×10⁻¹⁴ W，感知SINR阈值Γ_min=-31.77 dB。

**📈 对比分析**

与修改版WMMSE、VehicleHetero GNN以及直观分配基线对比，GNN-ISAC在通信总速率上达96.4–99.4%与WMMSE相当，感知链路违约率仅0.056–0.183%，且计算时间从WMMSE的23 ms压缩到0.127 ms；在更大网络规模上未重新训练的GNN-ISAC仍保留95%以上性能。

**⚠️ 局限性**

局限性包括：无监督惩罚方式无法严格保证每条感知链路满足SINR阈值；对极端网络拓扑或非常高的感知要求时可能出现违约；需要准确的信道状态信息；以及对训练时配置的依赖，使得在大幅度超出训练范围的网络中性能可能下降。

---

## 123. VREgo: Consumer VR Headsets for Learning Robot Manipulation from Egocentric Demonstrations

**arXiv ID:** 2611.07272 | [PDF](https://arxiv.org/pdf/2611.07272v1)

**作者:** Liuchuan Yu `[一作]` `[通讯]`, Liuchuan Yu

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在不需要行走的情况下，利用消费级VR头显PICO 4 Ultra的双目相机和实时手部追踪，采集人类在桌面上执行的抓取与堆叠演示，并将这些 egocentric 记录转换为虚拟手臂的训练数据，从而训练机器人抓取和堆叠策略。

**💡 创新点**

创新点包括：①使用立体视觉在静止摄像机下定位物体，避免需要头部运动；②为每个演示自适应抓取阈值并重写抓取轨迹；③在模拟闭环执行中引入一系列执行规则（如释放承诺、抓取对齐、从上方接近等），显著提升多步骤堆叠任务的成功率；④通过仅使用PICO 4 Ultra实现了与Aria玻璃相当甚至更高的训练数据压缩和简化。

**🔧 技术方法**

使用的技术包括：Stereo triangulation，Grounding DINO + SAM 2 目标检测与分割，CoTracker3 关键点跟踪，OpenXR 手部姿态估计，虚拟抓手与 flow-matching 策略，MuJoCo 仿真环境与 WidowX AI 机械臂控制器，演示预处理与抓取标注算法。

**📊 数据集**

使用的数据集为 123 条演示（Task1 41 条，Task2 41 条），所有演示均在同一桌面场景下由同一位操作员使用 PICO 4 Ultra（以及 Task1 额外的 Aria 记录）录制，形成 41 个布局的训练与评估集合。

**📈 对比分析**

评估方法：在 100 个未记录布局上进行仿真测试，比较 PICO 与 Aria 的成功率；Task1 中 40 条演示训练后成功率在 60–78%；Task2 中基线仅 0–3%，改进后达到 93.8%；此外还对训练时间、存储占用和抓取误差等指标做了对比，PICO 只需 1.3 GB 存储并保持可比性能。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，未对真实机器人进行测试；训练样本来源单一操作者和单一环境，可能不具备普适性；执行规则与物体检测高度依赖模拟的精确位姿，实际部署需要额外的姿态跟踪；规则是针对特定任务手工调优，通用性待验证。

---

## 124. CRISP: Constraint-Resolved Interactions between Subsystems for Predictive Whole-Body Cooperation

**arXiv ID:** 2611.07268 | [PDF](https://arxiv.org/pdf/2611.07268v1)

**作者:** Dario Valenzuela `[一作]` (University of Texas at El Paso), Jeeseop Kim `[通讯]` (University of Texas at El Paso)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种分层控制框架，将人体机器人拆分为躯干单刚体和浮动基臂系统，并通过全局非线性模型预测控制(NMPC)联合优化关节力、地面反作用力和闭链交互力；

**💡 创新点**

创新点在于将双臂协作的闭链耦合以及躯干-臂间结构耦合统一建模为全局约束，并将拉格朗日乘子直接作为决策变量，使得交互力在全局优化中得到一致且在线的估计；

**🔧 技术方法**

使用了非线性模型预测控制(NMPC)、拉格朗日约束动力学、离散化欧拉法、IPOPT求解器、MA57线性求解器，以及低层全身协调控制；

**📊 数据集**

实验数据来源于Unitree G1 29-DOF humanoid硬件与MuJoCo仿真环境，没有使用公开的标准数据集；

**📈 对比分析**

通过与真实接触力测量对比，验证了交互力估计的误差在可接受范围内；在硬件上，控制周期平均8 ms、99 %小于14 ms，表现出实时性与稳定的全身协同控制；

**⚠️ 局限性**

局限性包括：只能处理无滑移的接触、低层控制假设低身体为准静态、未覆盖行走或动态姿态变化、对大模型不确定性或极端扰动的鲁棒性有限。

---

## 125. Energy Measurement and Optimization of TinyML in Resource-Constrained Embedded Systems: A Systematic Mapping Study

**arXiv ID:** 2611.07264 | [PDF](https://arxiv.org/pdf/2611.07264v1)

**作者:** Clariele Almeida `[一作]` (Federal Rural University of Pernambuco), Ermeson Andrade `[通讯]` (Federal Rural University of Pernambuco)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统映射方法，对 2020-2025 年资源受限设备上 TinyML 能耗相关研究进行综述，梳理了使用的算法、能耗测量方法、技术影响因素以及优化策略。

**💡 创新点**

创新点在于以能耗为核心构建完整的分类体系，识别了主流 TinyML 模型、测量方法、技术因素，并提出四大优化类别（模型压缩、硬件感知设计、系统与通信优化、数据与处理优化），同时指出标准化不足与研究空白。

**🔧 技术方法**

采用文献检索、系统映射、定量统计与定性分析相结合的技术手段，并用条形图、表格等可视化手段呈现结果。

**📊 数据集**

检索来源为 ACM、IEEE、ScienceDirect、Scopus、Springer 等数据库，筛选 25 篇符合 2020-2025 年间的论文作为数据集。

**📈 对比分析**

通过频率统计、条形图和表格比较 25 篇论文中模型、测量方法、指标和优化技术，发现 CNN 最常用、直接测量最普遍，但由于缺乏统一测量规范，难以在不同平台之间进行直接能耗对比。

**⚠️ 局限性**

限制包括检索词和数据库差异导致的样本覆盖不足、样本量有限、缺乏统一能耗测量与报告规范，以及难以在同一硬件平台上精确归因单一优化技术的能耗影响。

---

## 126. An Empirical Study of Reinforcement Learning for Keypoint Positioning of Deformable Linear Objects

**arXiv ID:** 2611.07261 | [PDF](https://arxiv.org/pdf/2611.07261v1)

**作者:** Vinay Rodrigues `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了一端抓取的可变形线性物体（DLO）在平面上的强化学习操控，重点提升训练效率、观察设计、鲁棒性和零射模拟到实物的迁移。

**💡 创新点**

创新点包括：
- 将GPU并行MuJoCo Playground（MP）与JAX强化学习框架集成，显著提升经验采样率与收敛速度；
- 采用相对EE坐标的关键点表示与非对称actor‑critic结构，提升学习效率并保持可部署性；
- 在训练中加入观测噪声、延迟、域随机化和扰动恢复，使策略具备高度鲁棒性；
- 通过标记无RGB‑D关键点估计实现零射的实物迁移，未使用任何实物微调。

**🔧 技术方法**

使用的技术主要有：Proximal Policy Optimization（PPO）、JAX与MuJoCo（MJX）实现的GPU并行仿真、Stable‑Baselines3与Stable‑Baselines JAX的比较、相对EE坐标关键点表示、非对称actor‑critic、YOLOv5‑segmentation（YOLO26-seg）与骨骼化关键点提取、域随机化与扰动恢复训练。

**📊 数据集**

数据集：为提升RGB‑D关键点估计的鲁棒性，本文收集并标注了多种绳索、背景、光照与视角的任务专用数据集，用于YOLO26-seg的微调；仿真中使用MuJoCo生成的DLO配置。

**📈 对比分析**

对比方法：在相同wall‑clock时间下对比SB3+MuJoCo、SBX+MuJoCo与MP三套完整训练栈。MP在相同时间内收集约13倍的交互样本，最终单点追踪SR为0.95（MP）对比0.86（SB3）和0.88（SBX），奖励AUC亦最高。实物实验中，单点追踪9/10，扰动恢复10/10，两点追踪8/10，完成时间均在10s以内。

**⚠️ 局限性**

局限性：
- 仅针对平面一端抓取的DLO，未扩展到三维或多端抓取；
- 对复杂形状控制和极端物理参数的泛化仍有限；
- 依赖仿真物理引擎，实际物理差异可能导致迁移失效；
- 关键点估计对遮挡和材料柔韧性敏感。

---

## 127. TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference

**arXiv ID:** 2611.07256 | [PDF](https://arxiv.org/pdf/2611.07256v1)

**作者:** Hanzhi Zhang `[一作]`, Yunhe Feng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的稀疏混合精度注意力机制，能够在LLM推理中根据不同区域的重要性动态分配跳过、4‑bit和8‑bit的QK计算，从而显著减少计算量；

**💡 创新点**

创新点在于同时考虑稀疏和量化的双重敏感性，并用两条对角距离边界实现每层每头的压缩策略；通过离线校准测量归一化输出误差与完整块执行成本，利用全局分配器在共享预算下为每个头选择最优边界；

**🔧 技术方法**

核心技术包括：对角距离压缩编码、离线误差-成本评估、全局预算分配、融合的prefill与压缩KV缓存内核、整数4/8‑bit QK乘法与精度分组处理；

**📊 数据集**

使用的数据集包括：WikiText‑2（词预测）、HellaSwag、MMLU、ARC‑Challenge、LAMBADA、PIQA、WinoGrande（通用任务评估）；以及LV‑Eval的九个长上下文问答数据集（16K、32K、64K级别）做长上下文QA评估；

**📈 对比分析**

与Torch、BF16、MInference、FlexPrefill、SageAttn、MoA、SpargeAttn等方法比较，实验显示在Llama‑3.2‑3B‑Instruct上，生成吞吐量提升至7.88×（相较Torch）和2.65×（相较BF16），单个token生成时间下降至10.06 ms（BF16为26.64 ms），VRAM占用与能耗也分别降低约36 %与30 %；在通用与长上下文任务中，质量与原始模型几乎无差距；

**⚠️ 局限性**

限制包括：需要耗时的离线校准与预算搜索，且对不同模型的迁移需要重新校准；边界压缩虽然存储小，但在极长上下文或非常大模型时可能仍需额外调优；目前仅在3‑B级别模型验证，尚未在更大规模模型上验证其可扩展性。

---

## 128. Characterizing AI Service Traffic for 6G: An Open Testbed and Initial Results from 3GPP SA4

**arXiv ID:** 2611.07250 | [PDF](https://arxiv.org/pdf/2611.07250v1)

**作者:** Imed Bouazizi `[一作]` `[通讯]` (Qualcomm Technologies, Inc.), Imed Bouazizi (Qualcomm Technologies, Inc.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对 6G AI 服务的网络流量特性，构建了一个开源测试床，结合网络仿真、抓包、应用记录与场景执行，并对 3GPP SA4 TR 26.870 研究中采集的数据进行分析。

**💡 创新点**

创新点在于：① 将网络仿真层与 AI 场景执行层严格分离，保证实验可复现；② 通过统一的配置文件实现多场景、多网络配置的批量执行；③ 对仿真器行为（重排序、相关丢包抑制、Gilbert‑Elliott 参数映射）进行深入评估，揭示其对测量结果的影响；④ 通过将应用层记录与抓包结果关联，提供统一的多层度量视角。

**🔧 技术方法**

使用的技术包括 Linux traffic control (tc)、HTB+netem、IFB、tcpdump/sidecar、libpcap/pcapng 分析、SQLite 日志、Python orchestrator、Markov 基础流量生成器；在场景端使用了 vLLM、MCP、OpenClaw、WebRTC、HTTPS、JSON‑RPC 等协议与工具。

**📊 数据集**

主要数据集来自 TR 26.870 V0.6.2 工作表与 S4aP260148 工作簿：14,355 条请求记录、104,664 TCP 流、19,729 循环回流、162 UDP 流；还利用了 Huawei 提供的实时视频理解实验（125 问题/配置）以及 5G‑MAG 的仿真脚本。

**📈 对比分析**

比较方法：对不同网络配置（延迟、抖动、丢包、速率）下的场景，统计上传/下载比、峰值/平均比、TTFT/TTLT、RTT 及连接建立时间、误码容忍度等指标；对本地与远程执行、不同代理、不同任务类型进行交叉对比。结果表明：上传/下载比跨场景相差 6+ 量级；在 10% 丢包时视频理解准确率下降至 27%；在拥塞或长延迟下，TTFT/TTLT 明显升高；A2A 代理在高 RTT 下的延迟几乎等于 RTT；但由于模拟器重排序、丢包抑制和网络配置不一致，实际可达性与报文层级测量存在偏差。

**⚠️ 局限性**

局限性包括：① 模拟器的重排序和相关丢包抑制导致测得的丢包率与理论值不符；② 仅使用独立丢包与 Gilbert‑Elliott 模型，缺乏真实链路的丢包分布；③ 远程执行受供应商侧变异影响，未随机交错配置；④ 仅覆盖 TCP/UDP，未覆盖 QUIC 或持久连接；⑤ 缺乏多次重复实验与每个问题的结果方差；⑥ 数据集对特定模型、提示、版本高度耦合，难以复现。

---

## 129. KG-Neighbor: Knowledge-Guided Neighborhood Construction for Embedding Visualization

**arXiv ID:** 2611.07246 | [PDF](https://arxiv.org/pdf/2611.07246v1)

**作者:** Chang Won Kim `[一作]` (DATUMO Inc.), Kyohoon Jin `[通讯]` (DATUMO Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Knowledge-Guided Neighborhood Graph Construction（KG‑Neighbor）和 Residual Mismatch Diagnosis（KG‑Mismatch）两种方法，用内容提取的知识图三元组来调节嵌入距离并重新构建邻居图，进而提升嵌入可视化的语义邻居质量；

**💡 创新点**

创新点在于：①利用实例内容构造的知识图三元组共享度作为语义证据，对嵌入距离进行可调节的修正；②引入无标签残差不匹配诊断框架，将不匹配分为 Placement Issue、Label Issue、Boundary Case 与 Unresolved Case，帮助区分投影失败与标签/内容不一致；

**🔧 技术方法**

技术手段包括：使用 LLM（Claude Opus、GPT‑5.4 等）进行知识图构建与判定；计算四种 KG 相似度（Jaccard、Overlap、IDF 加权余弦、Adamic‑Adar）；引入 α 参数控制知识引导强度；结合 UMAP 等降维方法构建低维布局；

**📊 数据集**

实验数据集为四组：文本类 StackOverflow、BANKING77；图像类 CUB‑200‑2011、Oxford Flowers 102；每组随机采样 20 类，使用多种文本/视觉编码器；

**📈 对比分析**

与 kNN、SNN、MAPLE 基线对比，采用 NMI 与 LT_2D 评估；在弱编码器条件下，KG‑Neighbor 可使 NMI 提升至约 0.29、LT_2D 提升至约 0.54，明显优于基线；在强编码器条件下提升较小但仍有改善；邻居重排率与性能提升呈正相关；

**⚠️ 局限性**

局限性包括：仅基于精确三元组匹配，忽略实体/关系的模糊匹配；评估依赖标签，未覆盖所有语义维度；LLM 判定可能出现误判；对多标签或非文本/图像模态的适用性尚未验证；

---

## 130. How Do AI Agents Waste Your Money?

**arXiv ID:** 2611.07238 | [PDF](https://arxiv.org/pdf/2611.07238v1)

**作者:** Huimin Zeng `[一作]` (University of Illinois Urbana Champaign), Kai-Wei Chang `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套统一框架，用于跟踪、记录、识别并验证 AI 代理执行过程中可被移除的 Token，进而量化“Token 浪费”。

**💡 创新点**

创新点包括：① 将 Token 成本与 Token 浪费区分；② 采用 LLM‑as‑Judge 自动识别可移除的文本 span（Restate、Duplicate‑Fetch、Padding、Detour）；③ 通过在线剪枝和干预实验验证识别的浪费；④ 在六大前沿 LLM、五大代理 harness 以及多种 benchmark 上进行系统化比较，揭示浪费结构与任务、模型、工具、环境的关系。

**🔧 技术方法**

技术手段：统一代理追踪（LiteLLM 代理）、跨模型、跨 benchmark 的轨迹归一化；LLM‑as‑Judge（大模型作为判定器）对轨迹进行 span 标注；Token 计数与对账（与各供应商原生计数比对）；在线裁剪（只保留非 Restate/Padding span）以及多次 roll‑out 的行为差异评估。

**📊 数据集**

使用的数据集与 benchmark 包括：SWE‑bench Verified、MCP‑Atlas、τ²‑bench（airline/retail）、Claw‑Eval、SWE‑bench、Claw‑Eval 等；模型覆盖 Kimi‑K3、Gemini 3.1 Pro、GPT‑5.6 Terra、GPT‑6 Astra、Claude Sonnet 5、Claude Fable 5.1；代理 harness 包括 OpenCode、OpenClaw、OpenHands、Hermes、Kimi Code CLI。

**📈 对比分析**

比较方法：① 通过将识别出的浪费 span 删除并重新执行，测量任务成功率与下一步动作的差异；② 在线裁剪实验对比不同比例删除（0%、41–46%、100%）下的性能；③ 在同一模型、同一任务、不同 harness 下统计 Token 消耗、浪费比例与成功率；结果显示约 18% 输出 Token 为浪费，在线裁剪可删除 41–46% reasoning 而保持 89%+ 成功率；不同任务/模型/工具导致浪费比例差异明显，且并不与成功率呈正相关。

**⚠️ 局限性**

局限性：① 仍依赖 LLM‑as‑Judge 的判定，尤其对 Detour 的可移除性存在歧义；② 仅能对暴露完整推理的模型测量内部浪费，其他模型只能测量可见轨迹；③ 实验范围有限，未覆盖所有代理架构、工具/环境组合，且 harness 设计会快速演进；④ 只关注可直接移除的文本，未涵盖因环境约束导致的局部“合理但无效”行为。

---

## 131. TileQuant: Tile as the Precision-Allocation Unit for LLM Mixed-Precision Post-Training Quantization

**arXiv ID:** 2611.07217 | [PDF](https://arxiv.org/pdf/2611.07217v1)

**作者:** Hanzhi Zhang `[一作]` (University of North Texas), Yunhe Feng `[通讯]` (University of North Texas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种训练无关的TileQuant框架，将LLM线性层权重按硬件对齐的2D Tile划分，利用离线校准的输入激活感知(IAA)或梯度感知(GI)重要性评分，选择少量Tile保留16位，其余量化为8位，并在GPU上实现混合精度推理；

**💡 创新点**

①以2D Tile为精度分配单元，既捕捉列内敏感度又保持GPU矩阵乘法的规则化访问；②提供两种低成本校准路径（IAA与GI）快速得到Tile重要性；③通过load‑time重构与局部累加显著降低内存流量；④在LLM上实现生成吞吐提升至原BF16的2.66×且质量不下降甚至提升；

**🔧 技术方法**

PTQ、2D Tile划分、IA/GI重要性评分与top‑k选择、load‑time重构、混合精度（8/16位）矩阵乘法、Tensor Core、局部累加、Triton GPU内核等技术；

**📊 数据集**

WikiText‑2、HellaSwag、MMLU、ARC‑Challenge、LAMBADA、PIQA、WinoGrande、LongEval、LV‑Eval等语言建模、推理、检索与问答数据集；

**📈 对比分析**

在NVIDIA A100 40GB GPU上与原BF16、Uniform W8、W8A8、GPTQ、SmoothQuant、LLM.int8等基线对比，生成吞吐提升最高达2.66×BF16；在大部分任务（如WikiText‑2 PPL 12.403 vs 12.487，LAMBADA 65.78% vs 64.43%）保持或超过原始模型质量；在长上下文检索与问答中亦保持优势；

**⚠️ 局限性**

仅支持训练无关的8/16位混合精度，精度映射固定无法实时适配不同输入或任务；未扩展到训练或梯度优化；评估局限于GPU，其他硬件平台未知。

---

## 132. Latent Terrain: Adapting Neural Audio Autoencoders as Design Materials in NIME

**arXiv ID:** 2611.07210 | [PDF](https://arxiv.org/pdf/2611.07210v1)

**作者:** Shuoyang Jasper Zheng `[一作]` (Queen Mary University of London), Nick Bryan-Kinns `[通讯]` (University of the Arts London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一款 Max/MSP 包【[匿名名称]Latent Terrain】，通过坐标到潜在向量的映射，构建可在二维控制平面上导航的低维音频空间，以神经音频自编码器的潜在轨迹为基础，实现实时音频合成。

**💡 创新点**

① 采用 Fourier 特征映射提升 MLP 在高频潜在轨迹上的拟合精度，解决传统 MLP 的谱偏差问题；② 将高维潜在空间与低维控制空间的映射框架与可交互训练结合，形成一种可定制化的“音景地形”工具；③ 通过实践案例展示将自编码器作为设计材料而非单纯工具的可行性。

**🔧 技术方法**

主要技术包括：神经音频自编码器（RAVE、Music2Latent、Stable Audio Open、FlowDec），坐标到潜在空间的 MLP（含 Fourier 特征映射），实时前向推理与在线训练，Max/MSP 与 Python 的接口实现。

**📊 数据集**

使用了多种音频语料库：复古音乐、北京胡同档案录音、管风琴录音、鼓循环、现场声场等，全部采用 CC0、CC BY 或作者自制数据，以满足不同项目需求。

**📈 对比分析**

通过对比标准 MLP 与 Fourier 特征 MLP 的潜在轨迹回归误差，展示前者在高频分量拟合不足，后者显著提升 5–20% 的准确率；在实时因素（RTF）与训练时间上，Fourier 特征对性能影响极小，均保持在 1–2 秒内；音频重构质量通过感知实验和定量评估保持在原始模型的 95% 以上。

**⚠️ 局限性**

局限性：① 对高频潜在轨迹仍有过拟合风险，需手动调节 Fourier 参数；② 主要面向连续潜在空间，离散潜在代码的映射尚不成熟；③ 依赖预训练模型和音频数据集，若语料库质量不足会影响空间可塑性；④ 对用户而言，接口仍较为技术化，缺乏直观的可视化与教育资源。

---

## 133. Structured Machine Theory of Mind from Agent Trajectories

**arXiv ID:** 2611.07208 | [PDF](https://arxiv.org/pdf/2611.07208v1)

**作者:** Nigel F. Doering `[一作]` (University of California San Diego), Tauhidur Rahman `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种利用代理轨迹进行机器心智理论建模的方法，能够从轨迹中学习并推断代理的信念、欲望和意图。

**💡 创新点**

创新点在于将Belief-Desire-Intention (BDI) 结构嵌入模型瓶颈，使得信念和欲望成为可直接操作的中间表示，从而支持因果推断和可解释的 counterfactual 评估。

**🔧 技术方法**

技术上使用多头 Transformer 对三类轨迹（欲望上下文、最近上下文、当前轨迹）分别编码，随后通过多任务损失训练信念和欲望头，最后通过包含这些隐含状态的瓶颈进行目标预测。

**📊 数据集**

实验数据来自一个受控的行人导航仿真环境，包含 24 个兴趣点（6 类）和多种假信念配置，生成了数千个轨迹实例。

**📈 对比分析**

与传统无监督或仅使用轨迹的基线相比，BDIBottleneck 在目标预测上接近贝叶斯逆规划上限，并在新代理和罕见信念配置上保持高准确率，同时提供可编辑的心理状态接口。

**⚠️ 局限性**

局限性包括：需要仿真环境提供真值标签，信念标签稀疏导致学习困难；模型在全路径级别的 counterfactual 预测仍不够精确；以及对真实世界动态环境和隐私伦理的可迁移性尚未验证。

---

## 134. Patch-Based Generalized Neural Texture Upscaling

**arXiv ID:** 2611.07289 | [PDF](https://arxiv.org/pdf/2611.07289v1)

**作者:** Shin Fujieda `[一作]` (Advanced Micro Devices, Inc.), Takahiro Harada `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级神经网络纹理上采样方法，在渲染时对低分辨率纹理进行实时放大，无需修改原有纹理或资产制作流程。

**💡 创新点**

创新点在于：①使用单一泛化 MLP 训练，适用于所有纹理类别，避免每资产单独训练；②采用局部 Patch 采样与双线性基线+MLP 残差结构，降低模型尺寸与计算量；③模型可直接在 GPU 着色器中执行，消除执行分歧，易于集成。

**🔧 技术方法**

技术手段包括：多层感知机（MLP）+残差块、L1 损失、双线性残差连接、Patch 扩展与反射边界、Adam 优化、余弦退火学习率调度、LPIPS/FLIP 等感知度量。

**📊 数据集**

使用的数据集为 Poly Haven（Diffuse、Normal、ARM 共计 565+ 纹理）、Hangar、Victorian、Museum 等公开纹理集，用于训练和测试。

**📈 对比分析**

评价方法：与传统分析放大（bilinear、bicubic、Lanczos）以及 PFT 超分辨率模型进行对比，采用 LPIPS、PSNR、SSIM、FLIP 等指标；实验结果显示在 LPIPS 上显著优于分析方法，部分场景接近 PFT；渲染误差 FLIP 亦明显降低。过拟合模型相较泛化模型略有提升，但需额外预处理。性能评估未完整给出，预期 MLP 轻量级且 GPU divergence 较低。

**⚠️ 局限性**

局限性：①无法恢复源纹理中不存在的细节；②泛化模型质量低于针对单个纹理的 NTC 或过拟合模型；③高倍率（>2×）和极高分辨率细节再现受限；④缺乏完整的实时渲染性能评估；⑤在多纹理混合或特殊材质场景下可能仍需额外优化。

---

## 135. How Should Agents See Code? Rethinking Code Representations for Agentic Software Engineering

**arXiv ID:** 2611.07286 | [PDF](https://arxiv.org/pdf/2611.07286v1)

**作者:** Melika Sepidband `[一作]` (York University), Hadi Hemmati `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究不同代码表示（AST、调用图、依赖图、文档检索、混合）在LLM代理进行仓库级软件工程任务时的影响，探讨其对任务成功、token消耗、延迟和工具调用的作用

**💡 创新点**

首次在统一代理框架下对五种代码表示进行系统对比，明确表示方式与任务瓶颈之间的匹配关系，提出基于任务瓶颈选择表示策略的理论框架

**🔧 技术方法**

使用MCP（Model Context Protocol）实现代码表示服务器；利用Claude Code代理、Tree-sitter、Jedi、Griffe、随机游走中心性等技术；评估指标包括任务成功率、token使用、壁钟时间、工具调用次数

**📊 数据集**

LinuxFLBench（故障定位）、SWE-bench Pro（程序修复）、SecBench（安全补丁）、SWT-bench（测试生成）四个仓库级基准，各自代表不同任务瓶颈

**📈 对比分析**

通过在相同任务ID、模型、prompt下对六种工作流（无表示+五种表示）进行实验，比较成功率、token使用、时间等；结果显示：在定位为瓶颈的故障定位任务中，所有表示均提升recall@10并减少token消耗；在修复、补丁任务中，表示主要降低token消耗且对成功率影响有限；在测试生成任务中，表示提升率有限但token和时间成本上升；安全补丁任务中调用图表现最佳

**⚠️ 局限性**

局限性包括：只评估了Claude Code代理，未覆盖其他模型或多语言场景；实验单次运行（k=1）导致结果受随机变异影响；表示与工作流的配合（如先查询）混合影响，难以单独归因；对token、时间的度量未考虑构建映射等一次性成本

---

## 136. Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target

**arXiv ID:** 2611.07270 | [PDF](https://arxiv.org/pdf/2611.07270v1)

**作者:** Moto Shinriki `[一作]` (Tokyo Denki University), Tatsuji Takahashi `[通讯]` (Tokyo Denki University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究风险敏感满足式决策规则RS并将其重新解释为Attention Competitive Bayes（ACB），揭示满足水平作为先验均值、有限试验分配作为资源限制的结构；

**💡 创新点**

提出pseudo-forgetting机制并证明满足水平可在动态调整下降低Bernoulli多臂老虎机中的累积损失；

**🔧 技术方法**

利用RS、ACB、标准贝叶斯、忘记贝叶斯、FQL等模型，并实现动态满足水平调整算法；

**📊 数据集**

在Bernoulli多臂老虎机上进行实验，使用均匀抽样奖励概率（K=8）和固定概率（K=2）两种数据集；

**📈 对比分析**

与Softsatisficing、SAT-UCB+、Thompson Sampling比较，结果显示动态RS在两种满足水平下的平均累积损失低于其他满足式方法；

**⚠️ 局限性**

局限性包括满足水平为外部给定、仅在静态Bernoulli环境下验证、RS为确定性策略、动态满足调整缺乏理论最优保证。

---

## 137. Reweighting is not Harmless: Understanding the Role of Dataset Characteristics

**arXiv ID:** 2611.07258 | [PDF](https://arxiv.org/pdf/2611.07258v1)

**作者:** Amalia Ionescu `[一作]` (Cedars-Sinai Medical Center), Tiffani J. Bright `[通讯]` (Cedars-Sinai Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对 3,000 组人工生成的二分类数据进行实验，研究在不同群体规模和事件基线发生率下，对模型使用重加权（reweighting）做公平性干预时对预测结果、误分类率以及整体准确率的影响。

**💡 创新点**

揭示了重加权虽然能提升群体间的统计平等（DP 与 FNR 统一），但会把误分类成本转移到样本量较少、事件基线较高的群体，尤其当该群体为少数群体时，误分类率显著上升。该现象在真实数据中难以单独观察，本文通过模拟精准分离并量化。

**🔧 技术方法**

使用了重加权方法（基于 Kamiran & Calders 的权重公式），配合六种分类器（无正则化逻辑回归、L2 逻辑回归、决策树、随机森林、梯度提升与多层感知机），并在每种模型上比较加权与不加权两种情况，评估 DP、FNR 以及整体准确率。

**📊 数据集**

使用完全合成的数据集：每组 10,000 条样本，分为 70% 训练集与 30% 测试集，六种实验条件（群体比例为 20%/80% 或 50%/50%，以及事件基线为低、中、高），共 3,000 次重复实验以获得稳健的平均效果。

**📈 对比分析**

与未加权模型相比，重加权在大多数情形下显著改善 DP 和 FNR 统一（减小差距至约 2–3%），但对高基线少数群体的 FNR 上升 3.9–6.2%（≈ 48–82 额外漏诊/10,000 人）。整体准确率差异不到 0.1%，导致误分类成本在聚合指标中几乎不可见。

**⚠️ 局限性**

局限包括：仅考虑单一二分类任务、单一二值敏感属性、单一重加权策略；未引入真实数据、交叉特征、缺失或噪声标签；未探究多阈值或交叉群体；未与其他公平性干预（如阈值调整、判别可解释性）进行比较；因此结果对真实医疗等复杂场景的可推广性仍有限。

---

## 138. $D^2$-OPD: Separating Direction and Distance in Extrapolative On-Policy Distillation

**arXiv ID:** 2611.07241 | [PDF](https://arxiv.org/pdf/2611.07241v1)

**作者:** Bowen Fang `[一作]` (Columbia University), Hengzhi Pei `[通讯]` (Amazon AGI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的在策略上蒸馏方法，能够在学生生成的前缀上对教师的下一词分布进行学习，并通过解耦外推方向和距离来提高蒸馏效果。

**💡 创新点**

核心创新点是：①为每个前缀设定硬 KL 限制以控制目标与教师的距离；②利用教师后训练偏移与初始学生-教师差距的自适应组合来确定外推方向；③采用前向 KL 损失以保持对低概率词的梯度信息。

**🔧 技术方法**

技术包括：在策略上蒸馏（OPD）框架下构造外推目标；利用教师后训练偏移 Δ_T 与学生参考 Δ_π0 的中心化信号；硬 KL 限制与软权重 κ 组合的目标优化；前向 KL 损失实现目标传递。

**📊 数据集**

训练数据使用 DAPO‑Math‑17K，评估数据集为 AIME24、AIME25、AMC23 三大数学竞赛题库。

**📈 对比分析**

与 OPD、EOPD、Direct‑OPD、TOP‑D 等基线对比，三种实验设置中均实现了最高平均分，分别提升 OPD 的 19.62、8.09、7.36 分，表明方法在数学推理任务上显著优于现有蒸馏方案。

**⚠️ 局限性**

局限性包括：仅在 Qwen3 系列模型和数学推理任务上验证；需要教师后训练检查点来构造第二条外推方向；硬 KL 约束与软权重为超参数，需要手工调优；对其他模型族或任务的迁移性尚未评估。

---

## 139. PRB Starvation at Slot Time via Rank Indicator Inflation: Attacking Proportional Fair Memory in the Scheduler

**arXiv ID:** 2611.07225 | [PDF](https://arxiv.org/pdf/2611.07225v1)

**作者:** Ryan Barker `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在5G NR中注入伪造的 Rank Indicator (RI) 报告，探究并验证了 Proportional Fair (PF) 调度器在连续收到零成功传输时的服务内存失效，导致恶意 UE 占用全部 PRB 而正当 UE 被长期饿空的攻击模式。

**💡 创新点**

创新点在于：① 将“已服务但未交付”边界形式化并推导出闭式饥饿边界；② 在 3,300 个 200 槽对等实验中系统评估 RI 泼水对 PF 的影响；③ 通过离线重放与 Monte‑Carlo 递归证明攻击仅通过服务内存分母（EWMA）触发；④ 明确指出现有基于活动门控的防御无法检测此类持续攻击，并提出在 O‑DU 内进行 RI 一致性检查的防御方向。

**🔧 技术方法**

使用技术包括：5G NR 虚拟化调度器（winner‑take‑all PF 与 RR）、Sionna PHY 与 Ray‑Tracing 仿真、三种链路自适应控制器（OLLA、SALAD、SALAD‑CQI）、Monte‑Carlo 递归模型、离线决策重放、PRB 与吞吐量 Jain 公平度指标。

**📊 数据集**

数据集为 3,300 对 200 槽的仿真碎片，包含两种城市峡谷 Ray‑Tracing 几何（rank‑1 与 rank‑2）、4 名 UE、5 个随机种子、3 个链路自适应控制器，实验涵盖 RI 递增 Δ_r∈{0,1,2,3}、CQI/SINR 披露偏移、HARQ/PMI 随机化等参数。

**📈 对比分析**

比较方法：对 PF 与 RR 两种调度策略在不同 RI 递增强度下测量目标占有率、PRB 与吞吐量 Jain 公平度以及最长无授予时长。性能结果显示：RI+1 时目标占有率升至 94–95%，PRB 公平度从 0.997 降至约 0.28，最长零授予期达到 41–56 ms；RR 则保持 25% 目标占有率、PRB 公平度 1.0，最长零授予期仅 1.5 ms。

**⚠️ 局限性**

限制包括：① 调度器为理想的 winner‑take‑all 模型，未考虑实际 PRB 分配细粒度与多用户干扰；② 仅在仿真环境下验证，未通过实地 OTA 或真实硬件演示；③ 未覆盖多种 O‑DU 时延与 fronthaul 变异；④ 防御机制仅提出概念，未实现 RI 一致性校验或 PRB 日志记录；⑤ 结果受限于单一 100 MHz 带宽、特定 Ray‑Tracing 场景和链路自适应参数。

---

## 140. TrustMix: Graceful Degradation for Pretrained Models under Distribution Shift

**arXiv ID:** 2611.07206 | [PDF](https://arxiv.org/pdf/2611.07206v1)

**作者:** Juliusz Ziomek `[一作]` (University of Oxford), Yee Whye Teh `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TrustMix，一种轻量级的后处理框架，使预训练模型在分布偏移时能够平滑退化；

**💡 创新点**

创新点在于将模型的预测与用户指定的退化分布按可学习的单调门控混合，并在不改动原模型的情况下显式控制外域行为；

**🔧 技术方法**

采用了单调神经网络门控、马氏距离等OOD偏差信号以及对回归/分类的分布混合估计；

**📊 数据集**

在合成高维GP、Tabular（Weather、VPower_s）、Vision（UTK‑Face、CIFAR‑10/100、ImageNet）以及NLP（DistilBERT在多域评论）等数据集上进行评估；

**📈 对比分析**

与σ‑缩放、MC‑Dropout、距离感知σ‑缩放、温度缩放、密度感知校准等基线相比，TrustMix在OOD NLL、ECE上表现最优或相当，并在贝叶斯优化 surrogate 中获得竞争力；

**⚠️ 局限性**

主要限制是对退化分布的选择敏感；若退化分布不合适，退化过程可能导致性能下降；此外，效果依赖于OOD偏差信号的质量。

---

## 141. τ-bench-LILT: Evaluating Conversational Agents on Realistic Multilingual Tool-Use Tasks

**arXiv ID:** 2611.07203 | [PDF](https://arxiv.org/pdf/2611.07203v1)

**作者:** Kaden Uhlig `[一作]` (LILT), Joern Wuebker `[通讯]` (LILT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了多语言版本的工具使用会话代理评测基准τ‑bench‑LILT，覆盖德语、韩语、土耳其语和波斯语的航空、零售、电信和银行四个领域，完整本地化数据库、用户模拟器、任务与奖励；

**💡 创新点**

在不翻译政策与工具的前提下，系统性本地化每一层以提升真实性；通过原生本地化数据库、反复对比注释改进用户模拟器；并对多语言模型失败模式进行深入分析与改进；

**🔧 技术方法**

基于τ^3‑bench框架，使用Gemini 3.5 Flash、Claude Opus 5等LLM构建代理与用户模拟器；对任务与数据库进行本地化转换；利用人类对比评估改进模拟器；使用DeepSeek V4 Flash评估自然语言断言；

**📊 数据集**

原始τ^3‑bench数据集及其本地化版本，包括客户数据库（姓名、地址、手机号、航班等）、用户模拟器提示与任务指令，全部以GitHub开源形式发布；

**📈 对比分析**

对每个模型在英语和各语言的四个任务域分别进行pass1/​pass4评估，使用4次试验；结果显示韩语和波斯语零售域下降最高达61分，德语表现与英语相近甚至更好，其他语言模型表现差异显著；

**⚠️ 局限性**

仍未覆盖所有失败模式（如自注入、幻觉、错误语言回复）；工具与策略未本地化，可能导致真实部署出现问题；评估仅基于deterministic reward，未捕捉安全与质量风险；缺少完整多语言人类评估覆盖所有语言；

---

## 142. IRIS: Intelligent Random Access with Integrated Shared Learning for Decentralized LEO Direct-to-Satellite Uplinks

**arXiv ID:** 2611.07201 | [PDF](https://arxiv.org/pdf/2611.07201v1)

**作者:** Abdullah Al Noman `[一作]` (University of Texas at Arlington), Debashri Roy `[通讯]` (University of Texas at Arlington)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于参数共享多智能体强化学习（MARL）的去中心化卫星直连上行访问框架IRIS，并在其上实现了A2A-IRIS（侧链协作版）和Vanilla-IRIS（仅本地版）两种模型。

**💡 创新点**

创新点在于利用轻量级的侧链信息（延迟的同行状态直方图和当前优先级排名）实现了UE之间的隐式协同，显著提升了冲突抑制和公平性，同时保持了参数共享的可扩展性。

**🔧 技术方法**

核心技术包括共享DQN（Deep Q-Network）网络、负载感知ε-greedy探索、事件保留回放、合作奖励设计以及对延迟CSI的鲁棒处理。

**📊 数据集**

实验数据来源于基于TLE/SGP4轨道传播的LEO卫星轨迹、3GPP S-band传播模型、按人群密度生成的交通流和有限缓冲队列的仿真数据。

**📈 对比分析**

与理论上限、延迟CSI的集中式调度、p*ALOHA、BEB以及Vanilla-IRIS对比，A2A-IRIS在乡村和郊区环境中分别实现了99.9%和96.1%的理论上限吞吐率，较Vanilla-IRIS提升40%至325%，并在碰撞率、能量效率和公平性上显著优于所有基线。

**⚠️ 局限性**

主要局限包括仅在理想化仿真环境下验证，侧链传输被假设为无误且无延迟；未考虑多卫星协同、移动端切换开销、复杂动作空间或更高频段的应用，未来工作需在更真实场景中评估和扩展。

---

## 143. HyFlash: A Prefix-Memory Prior for On-Device Speculative Decoding

**arXiv ID:** 2611.07263 | [PDF](https://arxiv.org/pdf/2611.07263v1)

**作者:** Jongwoo Ko `[一作]` (Microsoft Applied Sciences Group), Pashmina Cameron `[通讯]` (Microsoft Applied Sciences Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种前缀记忆先验的紧凑并行草稿器HYFLASH，用于在设备端低内存环境下加速推理。

**💡 创新点**

创新点在于将历史上下文压缩到可读的前缀记忆与局部注意力分离，避免了全上下文注意力的内存膨胀，并通过可读前缀实现远程依赖建模。

**🔧 技术方法**

使用局部滑动窗口注意力、Mamba‑2/DeltaNet‑2递归记忆、层归一化/RMSNorm、融合门控等技术；并采用了DFlash2/DSpark读取器。

**📊 数据集**

训练基于Open‑PerfectBlend，评测使用数学、代码与聊天基准（GSM8K、MATH、AIME25、HumanEval、MBPP、LiveCodeBench、MT‑Bench）以及LongBench（HotpotQA、QASPER、GovReport）和Phi‑4‑reasoning。

**📈 对比分析**

与DFlash、DSpark、SWA及全上下文注意力对比，HYFLASH在Qwen3‑4B与Phi‑4‑reasoning上平均速度提升约1.95×→2.15×、2.22×→2.58×，接受长度提升约10–16%；量化至4bit后仍保持99.2%接受率并提升4.3%速度。

**⚠️ 局限性**

仅验证两层浅层草稿器，递归记忆为压缩形式而非完整token，固定窗口与记忆大小限制了对更大模型或自适应记忆的探索。

---

## 144. Source-balanced Capped Fisher: Bounded Evidence Aggregation for User Inference in Language Models

**arXiv ID:** 2611.07228 | [PDF](https://arxiv.org/pdf/2611.07228v1)

**作者:** Yang Liu `[一作]` (University of North Carolina), Wanrong Zhu `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于文档级评分的用户参与推断方法——Source‑Balanced Capped Fisher（SCF），能够在查询文档未参与训练时通过文档评分的极端值判断源是否被包含在语言模型更新中。

**💡 创新点**

创新点在于将已知未参与源的参考尾频映射为负对数证据，对单文档贡献做上限截断后平均，既保留多文档一致证据，又抑制极端单一文档的影响，并配合基于排名的p‑值校准，实现有限样本的 Type‑I 错误控制。

**🔧 技术方法**

使用的技术包括：文档级对数似然差分评分、参考源尾频平滑、负对数转化、截断（cap）聚合、基于校准源的秩‑p‑值决策，以及对结果进行AUROC、TPR/FPR评估。

**📊 数据集**

实验使用四个真实世界数据集：Reddit 帖子（用户），CC News 文章（域），Enron 邮件（邮箱拥有者）和 Chromium 开发者评论，模型基于 GPT‑Neo‑125M 并在参与源上微调。

**📈 对比分析**

与四种基线聚合方法（Mean、Max、Min、RaMIA）以及无截断 Fisher 比较，SCF 在所有数据集上均取得最高平均 AUROC（最高提升 2.41%），在不同显著性水平下的 TPR 也大多优于基线，且误报率保持在或低于预设阈值。

**⚠️ 局限性**

局限性包括：截断参数 α 的选择仍无理论最优准则；假设校准源与目标源在零假设下可交换，难以应对源分布漂移；实验仅使用一种文档评分规则和单一模型（GPT‑Neo‑125M），对其他模型、评分方法和更大规模数据的推广性尚待验证。

---

## 145. Loopback Transformer: Towards a Unified View of Residual Looped Transformers

**arXiv ID:** 2611.07226 | [PDF](https://arxiv.org/pdf/2611.07226v1)

**作者:** Xuan Ouyang `[一作]` (University of Wisconsin--Madison), Junjie Hu `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Loopback Transformer，一种通过分离更新与输入、几何衰减以及注意力重分配，在循环共享参数的 Transformer 中实现参数高效残差设计的方法。

**💡 创新点**

将残差记忆视为线性注意力的写‑传输‑读过程，单独存储每个块的更新并用几何衰减限制总权重，再用注意力重新分配，既保留了输入路径又显著减少共享块参数。

**🔧 技术方法**

采用循环共享参数、残差分离更新、几何衰减、注意力加权重分配、线性注意力/递归记忆框架，并在 Transformer 基础上实现。

**📊 数据集**

在 ClimbMix 预训练语料上训练，并在 13 个零样本任务（PIQA、HellaSwag、WinoGrande 等）、WikiText、GSM8K、MATH‑500、HumanEval、MBPP 等数据集上评估。

**📈 对比分析**

在相同训练 token 与深度条件下与标准 Transformer 及 7 种循环残差设计对比，使用验证损失、13 任务平均准确率、WikiText 困惑度等指标；Loopback 在 191M‑3B 范围内均实现最低验证损失、最高平均准确率，参数仅占标准 Transformer 的 40‑75%。

**⚠️ 局限性**

仅在 ClimbMix 语料上验证，未检验不同预训练语料、可变循环访问次数或显式历史对内存/吞吐的影响，且额外循环访问未能提升性能。

---

## 146. Fractional physics-informed neural networks to model social media virality

**arXiv ID:** 2611.07215 | [PDF](https://arxiv.org/pdf/2611.07215v1)

**作者:** Alex Koong `[一作]` (Brown University), George Em Karniadakis `[通讯]` (Brown University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用分数物理信息神经网络学习社交媒体内容传播动力学，联合估计记忆阶数与增长律；

**💡 创新点**

创新点在于将分数阶记忆与可解释增长律结合，并通过教师‑学生分段训练实现时间可变的记忆动态；

**🔧 技术方法**

采用分数PINN、符号回归（PySR）、全白化残差、分段教师‑学生机制等技术；

**📊 数据集**

使用2025‑2026年X平台（前Twitter）24小时累计互动数据，共348条训练帖子与86条测试帖子；

**📈 对比分析**

与九种传统基线（如Logistic、Gompertz、Hawkes等）对比，教师‑学生PINN在MSE<0.01时获得64%成功率，超过最佳基线约45%；

**⚠️ 局限性**

局限包括样本规模有限、主题集中、仅预测轨迹形状不含最终规模、对平台算法更新高度敏感等。

---

## 147. Using Transformers for Teaching and Learning via Symbolic Distillation of Neural A-life Agents

**arXiv ID:** 2611.07254 | [PDF](https://arxiv.org/pdf/2611.07254v1)

**作者:** Shuwei Wang `[一作]` (University of Alberta), William Yeoh `[通讯]` (Washington University in St. Louis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究使用预训练的Transformer模型将演化出的自组织生命（A‑life）智能体的神经控制策略蒸馏成可读的符号策略，以便人类理解和跨架构共享知识。

**💡 创新点**

创新点在于：①首次将自动化神经架构搜索（NAS）用于设计适合符号回归的Transformer；②利用随机生成的符号策略与对应的感知‑动作对训练Transformer，实现无需真实智能体数据即可学习；③将蒸馏过程嵌入A‑life生命周期，支持自我反思与教学学习。

**🔧 技术方法**

核心技术包括Transformer序列到序列模型、神经架构搜索、自动化数据生成、束搜索、遗传算法（GA）与决策树（CART）作为基线；使用A‑life环境生成的感知‑动作对作为训练/测试数据。

**📊 数据集**

使用自生成的符号策略语法样本（约45万条）与A‑life仿真中演化出的神经智能体产生的512个感知‑动作样本。

**📈 对比分析**

与GA（多时间限制）和CART（不同树深度）对比，Transformer在平均0.63‑4.08秒内即可得到准确率≈99%的符号策略，明显快于GA最长45分钟的时间，并在准确率上优于所有基线。

**⚠️ 局限性**

局限性包括：①Transformer参数量（约540万）仍高于单个智能体的300参数神经网络，消耗额外能量；②对自我反思与教学学习的积极性未做自适应；③在更大、更复杂的A‑life环境中符号空间偏差对群体动态的影响不明显，需进一步研究。

---

## 148. Designing with Cinematic Language: An Intent-Driven Approach to Human-AI Video Clip Co-Creation

**arXiv ID:** 2611.07377 | [PDF](https://arxiv.org/pdf/2611.07377v1)

**作者:** Yi He `[一作]` (Tongji University), Nan Cao `[通讯]` (Tongji University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种人机协同的电影剪辑创作系统CineAssist，利用电影语言设计空间将创作者的高层意图转化为可执行的摄像参数，支持意图澄清、策略推荐、3D编辑和偏好学习。

**💡 创新点**

创新点在于：①构建了包含53个二级意图与九个摄像参数的结构化电影语言设计空间；②设计了多代理流水线（意图感知器、策略推荐器、偏好学习器），将自然语言意图与具体拍摄参数显式映射；③提供了可检索、可编辑的意图与参数表征，使创作者能够直观比较不同策略并在3D编辑器中即时预览；④实现了基于交互证据的上下文敏感偏好学习，支持在新任务中恢复并可确认或拒绝。

**🔧 技术方法**

技术主要包括：大型语言模型（LLM）用于意图解析与策略生成，检索增强生成（RAG）与共享向量索引实现设计空间查询，基于规则与LLM的策略审阅，3D相机控制与实时预览，偏好学习采用证据状态机与上下文检索，视频生成采用Seedance 2.0等扩散式视频模型。

**📊 数据集**

使用了两大数据集：1) 从107部电影学术与实践文献中抽取的意图与技术标签，形成初始意图树；2) 1,436个电影片段的人工与AI辅助双重标注，提供53个二级意图与相应摄像参数的映射，构成最终设计空间。

**📈 对比分析**

对比方法：在30个视频对比实验中，将CineAssist与直接使用Seedance 2.0对照；用户以7分量表评估意图表达、电影语言匹配与整体偏好。结果显示CineAssist在所有三个维度上均显著优于基线（平均分≈5.7 vs 2.2，p_Holm<0.001），并在专家实验中获得高于4分的满意度。偏好学习在新任务中的匹配准确率约为77.8%。

**⚠️ 局限性**

局限性包括：仅支持单镜头/短序列创作，缺乏对长片故事结构、剪辑连续性等宏观层面的支持；设计空间虽然覆盖丰富，但仍不足以覆盖所有文化与创作语境，存在潜在误解；系统依赖于预训练LLM与检索索引，模型误判与检索缺失可能导致意图解析错误；偏好学习在多样化场景下的泛化能力尚待进一步验证。

---

## 149. Breaking $\mathsf{TraintoDefend}$ via Adaptive Neuron Clustering in Cryptanalytic Model Extraction

**arXiv ID:** 2611.07370 | [PDF](https://arxiv.org/pdf/2611.07370v1)

**作者:** Haojie Wu `[一作]` (Shandong University), Guoyan Zhang `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究针对深度学习模型的加密分析提取攻击，证明并利用权重相似性导致的簇错误，提出一种自适应簇规则突破了现有的防御机制。

**💡 创新点**

创新点在于将相对尺度拟合与权重方向和偏置一致性结合的自适应簇准则，克服了原有基于坐标一致性的簇错误，从而实现对被防御模型的完全签名恢复。

**🔧 技术方法**

主要技术包括：对局部签名观察的相对尺度估计、方向一致性度量、偏置一致性检验、以及对所有公共坐标的平方残差最小化。

**📊 数据集**

使用 MNIST 数据集训练的全连接 ReLU 网络，实验覆盖 6 种不同宽度和层数的网络结构。

**📈 对比分析**

与原始提取攻击（仅使用坐标一致性）相比，新的簇规则在 6 个模型上将签名恢复率从 0% 提升到 100%，相对误差最高仅为 2⁻²².²⁷，证明了防御效果被突破。

**⚠️ 局限性**

局限性包括：仅针对第一隐藏层的防御机制，未评估更深层或其他网络架构的效果；对高维或更复杂网络的可扩展性待进一步验证。

---

## 150. Acoustic Anomaly Detection for Early Identification of Drill Jamming

**arXiv ID:** 2611.07356 | [PDF](https://arxiv.org/pdf/2611.07356v1)

**作者:** Maria Beatriz Silva `[一作]` (Federal Rural University of Pernambuco), Saeed Afshar `[通讯]` (Western Sydney University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用超声波声学信号进行CNC钻孔过程中的提前钻头堵塞检测

**💡 创新点**

提出了在不同工具间不泄漏验证的Leave-One-Drill-Out方案，比较半监督自动编码器与监督XGBoost在协变量移位下的鲁棒性

**🔧 技术方法**

使用自动编码器（MLP）做异常检测与XGBoost分类器，采用MFCC、RMS、谱心差、谱带宽等46维特征，配合遗传算法调参

**📊 数据集**

公开17个4mm碳化物钻头的声学数据集，涵盖工具寿命不同阶段的声学特征

**📈 对比分析**

在LODO验证下，自动编码器的平均F1为0.81、召回率1.00、AUC 0.8294，XGBoost平均F1为0.55、召回率0.47、AUC 0.7559，表明半监督模型更可靠

**⚠️ 局限性**

数据量仅17个工具，未覆盖不同工件材质与切削条件，阈值设定需针对特定工具几何校准

---

## 151. AtlasULP: Domain-aware Universal Link Prediction via Relation Atlas

**arXiv ID:** 2611.07347 | [PDF](https://arxiv.org/pdf/2611.07347v1)

**作者:** Yujing Liu `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种基于关系图谱（Relation Atlas）的通用链路预测框架，结合全局与局部多尺度结构表示，并通过域感知提示与上下文推理实现无训练的跨域链路预测。

**💡 创新点**

创新点在于：①构造全局多阶路径响应（Chebyshev递推）与局部统计指标（HN、RA、LC、FP、FD）相结合的关系图谱，①实现对目标域的提示学习和上下文推理，使模型能自适应捕捉域特定的连通模式。

**🔧 技术方法**

使用技术包括：Chebyshev多阶递推、五种局部统计指标、双流编码器、Perceiver提示生成器、Transformer上下文编码、余弦相似度评分与Sigmoid分类。

**📊 数据集**

数据集涵盖七个未见目标图，跨六大领域：生物（Ecoli、Yeast、Celegans）、交通（Power、USAir）、网络（PolBlogs、Router、PB）、协作（Physics、CS、NS）、引文（Pubmed、Citeseer、Cora）和社交网络（Twitch、Github、Facebook），采用源域预训练、目标域无训练推理。

**📈 对比分析**

与传统启发式方法、基于GNN的有监督模型以及最先进的通用链路预测模型UniLP进行对比；平均Hits@50达到71.69%，比UniLP高3.69%，比启发式方法高7.85%，比SEAL等GNN模型高4.83%，同时推理速度显著提升。

**⚠️ 局限性**

局限性在于关系图谱的构造仍需人工设计，缺乏自动化学习方式；此外对无属性图的依赖可能限制在有属性图上的应用。

---

## 152. GreedTok is a Constant-Factor Approximation for Partition-Cover Tokenization

**arXiv ID:** 2611.07346 | [PDF](https://arxiv.org/pdf/2611.07346v1)

**作者:** Kundan Baliga `[一作]` (University of Illinois Urbana Champaign), Aditya Byju `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 GreedTok 贪心分词算法的最坏情况近似性进行了深入分析，并给出了理论证明。

**💡 创新点**

创新点：①首次证明 GreedTok 对于任意预算 k 的近似下界为 0.93（即至少能获得最优解的 93%）；②构造了上界实例，证明其最坏比率不超过 (7‑ln 2)/12≈0.525；③提出了 Shift Lemma、窗口分布等新的技术手段；④证明该问题在无约束词典下 NP‑难以近似到 1‑1/31580。

**🔧 技术方法**

主要技术包括：组合优化与贪心分析、Shift Lemma（处理重叠出现），窗口平均与线性规划求解分布，概率期望与递推不等式，以及归约构造实现 NP‑硬度证明。

**📊 数据集**

实验数据主要是人工构造的合成语料（单词集合），并在 Zenodo 上公开代码、证书与结果文件；没有使用公开的真实语料库。

**📈 对比分析**

通过与 GreedTok 传统实现的对比实验，展示在 worst‑case 示例中得到的比值在 0.53 左右；理论上得到的 0.93 下界与实验表现相距较大，说明实际性能优于理论预测。

**⚠️ 局限性**

限制：①上界仍远高于实际实验结果，常数因子下界 0.93 仅在理论层面；②硬度证明给出的常数 1‑1/31580 远低于常见的 1‑1/e 目标，实际难度未知；③本文未给出更紧凑的近似比率或针对有限字母表的结果。

---

## 153. Where Do Reported Long-Video QA Gains Come From? A Controlled Audit of Frame Selection, Evidence Injection, and Expert Routing

**arXiv ID:** 2611.07329 | [PDF](https://arxiv.org/pdf/2611.07329v1)

**作者:** Shih-Yao Lin `[一作]` `[通讯]` (Independent Researcher), Shih-Yao Lin (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过对长视频问答的帧选择、非视觉证据注入和回答模型路由三大决策进行可控审计，评估其对性能的真实贡献。

**💡 创新点**

创新点在于提出统一的可控审计框架，将不同决策分离，量化其对准确率的边际影响，并揭示子集估计的不稳定性与模型路由效应的主要来源。

**🔧 技术方法**

使用了基于子模优化的帧选择级联、LLM驱动的规划器、Whisper/OCR文本注入以及对比实验的统计方法（McNemar、Bootstrap CI、TOST）。

**📊 数据集**

实验数据集包括Video-MME长视频、UCF-Crime、XD-Violence、MUSIC-AVQA和EgoSchema。

**📈 对比分析**

与均匀采样对照、已移植的选择规则和不同注入策略进行匹配对比，结果显示在全尺度下帧选择提升不足5个百分点，子集估计波动大，模型路由带来最大提升但主要是模型换装。

**⚠️ 局限性**

局限性包括仅针对K=8、特定模型和数据集，审计的级联配置与发布默认设置不同，规划器记忆基于开发集，未验证对更大规模或不同任务的泛化。

---

## 154. Permanent and Transient Representations for Continual Reinforcement Learning

**arXiv ID:** 2611.07327 | [PDF](https://arxiv.org/pdf/2611.07327v1)

**作者:** Nishanth Anand `[一作]` (McGill University), Doina Precup `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种系统级的持续强化学习框架，使用永久（慢速、参数化）与瞬态（快速、非参数化）两套特征表示来平衡稳定性与可塑性；

**💡 创新点**

创新点在于把两套表示分离，并引入基于 MinHash 的非参数瞬态逼近器，实现快速在线更新与局部泛化；

**🔧 技术方法**

使用线性/神经网络参数化的永久价值函数、MinHash+哈希表的瞬态价值函数、两时间尺度 TD 更新、线性逼近理论证明；

**📊 数据集**

主要实验数据集包括 CIFAR‑10（图像任务）和 Craftax‑Classic（开放式游戏环境），并在 5 状态/5 动作的简化图像 MDP 进行验证；

**📈 对比分析**

与 PQN、PT‑IHT、T‑Only、PT‑NN、EWC、CB、NEC 等基线对比，实验显示在图像任务中对任务切换的适应速度最快；在 Craftax‑Classic 上在 250M 规模实验中比 PQN 快 2 倍（<15 归约为 150M 步）并在 50M 步内达 10 回报；

**⚠️ 局限性**

局限性包括对哈希与槽大小等超参数敏感，令令牌化和图像预处理仍不够精细，缺乏针对策略梯度方法的扩展，且在更高维度连续空间中的可扩展性尚待验证。

---

## 155. An Ordering Optimization for Fruit-Vegetable in High-Density Environment based on Genetic Algorithm

**arXiv ID:** 2611.07293 | [PDF](https://arxiv.org/pdf/2611.07293v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 156. Exact $\ell_p$-CVP via Euclidean SVP in Polynomial Space

**arXiv ID:** 2611.07313 | [PDF](https://arxiv.org/pdf/2611.07313v1)

**作者:** Jiseung Kim `[一作]` (Jeonbuk National University), Hyung Tae Lee `[通讯]` (Chung-Ang University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种从精确的ℓ_p-CVP到精确的欧几里得SVP的随机化、保持秩的归约，适用于每个整数p≥2。

**💡 创新点**

创新点在于使用多项式工作空间外部的SVP oracle，解决了CVP和SVP之间的复杂性差距，并为p≥3的情况提供了新的算法。

**🔧 技术方法**

采用了随机化局部搜索、计数论证和基于SVP的采样技术。

**📊 数据集**

使用了全秩格L⊂ℝ^N的有理输入，输入的比特长度为ℓ。

**📈 对比分析**

通过与Kannan和Feng等人的SVP算法进行比较，展示了在多项式空间内的CVP算法的运行时间分别为N^N/(2e)+o(N)和N^N/(4e)+o(N)。

**⚠️ 局限性**

限制在于对于p≥3的情况，虽然提供了单指数时间和空间的算法，但仍然存在指数开销，未来的工作可以集中在减少这种开销上。

---

## 157. Quadratic-Time Barriers from Self-Referential SAT

**arXiv ID:** 2611.07308 | [PDF](https://arxiv.org/pdf/2611.07308v1)

**作者:** Guangyan Zhou `[一作]` `[通讯]` (Beijing Technology and Business University), Guangyan Zhou (Beijing Technology and Business University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

基于自指 K‑SAT exhaustive‑search 假设（SRK‑ESH），证明二进制最长公共子序列、编辑距离以及一维 DTW 在多项式时间内无法实现严格二次以下的确定性算法。

**💡 创新点**

创新点在于提出 SRK‑ESH 作为新的硬件假设，并利用其推导 Orthogonal Vectors 在多项式维度下的硬件；随后通过 split‑and‑list 构造将 SAT 问题精确映射到向量列表，再将向量列表映射到序列问题，从而得到针对多种序列相似度问题的强子二次下界。

**🔧 技术方法**

主要技术包括：自指 SAT 的假设框架、变量拆分并构造向量列表的 split‑and‑list 方法、向量–序列的 gadget 归约（针对 LCS、编辑距离、DTW 的三层对齐 gadget）、以及严格的时间与维度算量分析。

**📊 数据集**

本文为理论性工作，没有使用具体实验数据集；所考虑的输入对象为 CNF 公式、向量列表和字符串序列。

**📈 对比分析**

相较于之前基于 SETH 的下界，本文在 SRK‑ESH 假设下给出了更广泛的多目标下界，但并未提供可执行算法，因此无法与实际实现的性能进行比较。

**⚠️ 局限性**

局限性：结论完全依赖于 SRK‑ESH 假设，一旦该假设被破除，下界将失效；此外本文仅给出条件性下界，没有提出新的加速算法；在实际应用中缺少实验验证。

---

## 158. Provably Efficient Representation Learning for Low-Rank CMDPs

**arXiv ID:** 2611.07295 | [PDF](https://arxiv.org/pdf/2611.07295v1)

**作者:** Kaixuan Liu `[一作]` (Emory University), Jian Li `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向低秩约束马尔可夫决策过程的模型无关表示学习方法；

**💡 创新点**

创新点在于将最小化-最大化表示学习目标与LSVI‑UCB及双重投影技术结合，实现了同时优化奖励与约束的联合表示学习；

**🔧 技术方法**

采用了最小化-最大化的特征学习算法、LSVI‑UCB价值迭代、拉格朗日双重投影（primal–dual）以及上置信界（UCB）奖励；

**📊 数据集**

实验使用了新构造的“Constrained Latent ComboLock (CLCL)”基准环境，包含高维视觉观测与长时序安全约束；

**📈 对比分析**

与Lagrangian Policy Gradient、CPO、HOMER、BRIEE等基线相比，方法在CLCL上实现了更高的奖励且满足约束，表现出最优的奖励-约束平衡；

**⚠️ 局限性**

局限包括缺乏下界证明、表示学习依赖优化oracle、辅助 roll‑ins 不计入 regret/violation、实验仅在单一基准且高维任务受限，并假设动作空间有限与特征可实现性等。

---

## 159. How do Transformers Perform Low-Level Control In-Context?

**arXiv ID:** 2611.07291 | [PDF](https://arxiv.org/pdf/2611.07291v1)

**作者:** Ebonye Smith `[一作]` (University of California Berkeley), Gireeja Ranade `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出利用Transformer模型实现对不稳定、非线性动力系统的低层控制，通过仅给出无驱动力的系统轨迹作为上下文，学习在测试时无需显式参数估计即可稳定系统。

**💡 创新点**

创新点在于：①将Transformer的上下文学习能力应用于低层控制；②使用多任务训练（控制、状态预测、模式分类）以及噪声扰动数据增强，以提升鲁棒性和泛化；③证明在足够长上下文和充分训练后，模型能在离散分布外的系统上实现稳定。

**🔧 技术方法**

技术上采用GPT‑2风格的解码器Transformer，配备RoPE位置编码，三头MLP输出；多任务损失包含控制误差、状态预测误差和模式交叉熵；训练数据通过专家控制器生成并注入高斯噪声；对比基线为同构LSTM。

**📊 数据集**

使用Cartpole和Acrobot两种经典不受控系统的仿真数据，分别在不同参数分布（在分布、易分布、难分布）下生成数十万轨迹进行训练。

**📈 对比分析**

与传统的基准控制器和同结构LSTM比较，Transformer在足够长的上下文（N≥50）时能达到约86%（Cartpole）和74%（Acrobot）的在分布稳定率，明显优于基准控制器；在离散分布下，Transformer表现与LSTM相近，但在更长上下文下可略优。

**⚠️ 局限性**

局限性包括：①需要为每个系统实例事先提供参数化的专家控制器；②对无噪声的“干净”数据训练效果差，需手工加入噪声；③在极端离散或参数变化极大时，Transformer仍可能出现记忆化而非真正的上下文适应，导致泛化受限。

---

## 160. EnvOpt: Automated Optimization for Agent Environments

**arXiv ID:** 2611.07372 | [PDF](https://arxiv.org/pdf/2611.07372v1)

**作者:** Mingxuan Li `[一作]` (University of California, Berkeley), Alex Dimakis `[通讯]` (University of California, Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自动化管道，用于在固定（stationary）环境上对大型语言模型代理进行环境表示的优化，从而降低执行成本并提升成功率。

**💡 创新点**

创新点在于把环境优化纳入 LLM 代理改进的第三个维度，并通过无评估器的训练任务让“研究代理”在不触碰模型和 harness 的前提下迭代修改环境结构与接口。

**🔧 技术方法**

技术包括：使用编程型代理（如 Codex/Claude Code）进行环境演化、无评估器的训练与验证、自动化工具调用、数据库/索引构建、cheatsheet 与程序化文件筛选等；评估时采用 token‑cost 与 pass‑rate 监测。

**📊 数据集**

数据集涵盖五个基准：AutomationBench、Workspace‑Bench、Data‑Eng‑Bench、Bleve（SWE）与 Tenacity（SWE），其中大部分通过合成任务实现训练，验证集使用真实答案。

**📈 对比分析**

与基线、GEPA、Meta‑Harness 等方法对比，平均成本下降约 48%（最高 88%），pass‑rate 保持或提升（最多 14%）。在不同模型和 harness 上迁移效果良好，且可叠加 Meta‑Harness 进一步提升。

**⚠️ 局限性**

局限性：需依赖足够的无评估器训练任务；环境演化不适用于动态或极度多样化环境；训练成本和时间仍不小，且某些优化可能导致过拟合或引入新错误。

---

## 161. Design of Multi-UAV Cooperative Deep Semantic Autoencoders for Communication Networks

**arXiv ID:** 2611.07374 | [PDF](https://arxiv.org/pdf/2611.07374v1)

**作者:** Xiaoling Han `[一作]` (Dalian Maritime University), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种多无人机协同深度语义自编码器（MC-DSA），通过任务分支与策略分支结合语义匹配与源选择机制，实现低带宽条件下的任务导向协作与语义通信。

**💡 创新点**

创新点包括：① 语义匹配+源选择框架，在需要时动态激活协同；② 任务分支与策略分支共享编码器参数但互不干扰，兼顾任务性能与通信效率；③ 软硬阈值训练与测试策略，实现端到端学习与推理一致；④ 在多信道、多SNR环境下自适应减少冗余传输，提升系统整体效率。

**🔧 技术方法**

采用深度学习技术（卷积+Swin Transformer 语义编码器、轻量化解码器），查询-键匹配、梯度下降训练、TDMA 带宽/延迟/能耗建模、AWGN/Rician/Rayleigh 信道仿真。

**📊 数据集**

使用 AirSim–SRMS 模拟低空图像数据集（512×512分辨率），包含多视角图像及对应语义分割标注。

**📈 对比分析**

通过与 MC-DSA@Cooperative/Standalone、ResNet、ECANet、SegNet、Random Select/Fusion 等基线在 AWGN/Rician/Rayleigh 信道及不同 SNR 下对比，MC-DSA 在 OAcc、mAcc、FWIoU、mIoU 上持续领先，且带宽占用、累计延迟与能耗显著低于传统协同方案，表现出稳健的性能优势。

**⚠️ 局限性**

局限性包括：实验仅在仿真环境，缺乏真实无人机部署验证；模型参数共享假设在实际多机协同中可能受硬件差异影响；通信模型简化为 TDMA，未考虑多跳/干扰等复杂情况；阈值选择对低 SNR 性能影响较大，需进一步自动化；模型规模与推理速度仍可进一步轻量化。

---

## 162. A Simple Discrete Newton Method for the Quickest Transshipment Problem

**arXiv ID:** 2611.07369 | [PDF](https://arxiv.org/pdf/2611.07369v1)

**作者:** Naoyuki Kamiyama `[一作]` `[通讯]` (Kyoto University), Naoyuki Kamiyama (Kyoto University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种简单的多源多汇快速转运问题的多项式时间离散牛顿方法；

**💡 创新点**

不需要计算左导数，只需一次候选参数；相比原先强多项式方法实现更简洁；

**🔧 技术方法**

利用子模函数最小化、最小成本流与离散牛顿迭代；

**📊 数据集**

未使用实验数据，全部为理论算法分析；

**📈 对比分析**

与Schlöter等人提出的强多项式算法做理论对比，虽时间复杂度更高但实现更简单；

**⚠️ 局限性**

时间复杂度仍较高，实际效率不如已有最优算法；

---

## 163. MIKU: Material Invariance and Knowledge Unification for PBR Generation

**arXiv ID:** 2611.07354 | [PDF](https://arxiv.org/pdf/2611.07354v1)

**作者:** Yajie Bao `[一作]` (Nanjing University), Yao Yao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出统一的 MIKU 框架，标准化 PBR 资产表示并构建统一的数据流水线，使用两阶段扩散模型在多视图条件下生成高质量纹理与物理一致的金属/粗糙度映射。

**💡 创新点**

创新点包括：① 统一 PBR 规范化与多视图数据标准化，消除艺术家定义的不一致；② 采用几何引导的条件无监督扩散模型与 CFG，保证跨视图的物理一致性；③ 采用动态分辨率打包与分辨率级联训练，提升细节与效率；④ 通过 DMD 加速推理，将 30 步扩散压缩至 6 步，仅需约 40 秒。

**🔧 技术方法**

核心技术：多视图数据打包与动态分辨率优化、扩散模型（FLUX.2 + LoRA 微调）、几何引导的 CFG、VAE+ DINO 双模特特征融合、VLM+SAM3 的材料识别与分割、分辨率级联训练、DMD 推理加速、标准化 PBR 参数映射与高频残差注入。

**📊 数据集**

使用约 260k 公开+自有 3D 资产（TexVerse、Objaverse 等）经统一流水线过滤后得到约 20k 高质量标准化 PBR 数据；训练数据主要来自统一后的 20k 资产，测试使用 Objaverse 200 资产做多视图评估。

**📈 对比分析**

与 MvAdapter、UniTex、Hunyuan3d‑2.1、LumiTex 等现有方法对比，MIKU 在 FID 121.25、CLIP‑I 0.9411、推理时间 40+12 秒、最高 1024×1024 albedo、512×512 MR 分辨率下表现最佳；用户研究与量化指标均显示显著优于对比方法。

**⚠️ 局限性**

局限性：数据构建仍需人工审核，容易引入主观偏差；统一规范仅保证相对物理一致性，未结合真实测量材料，缺乏严格物理真实性；数据规模有限，难以覆盖所有材质变种。

---

## 164. Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair

**arXiv ID:** 2611.07342 | [PDF](https://arxiv.org/pdf/2611.07342v1)

**作者:** Tung-Wei Lin `[一作]` (University of California Berkeley), Alberto Sangiovanni-Vinentelli `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM的迭代图修复框架，利用路由器决定是否启用外部检索器来修复多重违约；

**💡 创新点**

创新点在于：①构建多违约测试生成器以获得已知修复；②训练可部署的检索路由策略，动态控制检索；③通过SHAP解释路由决策；

**🔧 技术方法**

使用LLM修复代理、检索代理、路由器（T‑learner XGBoost）、SHACL约束验证、聚类合并违规、SPARQL编辑工具、外部语料检索；

**📊 数据集**

在两个建筑自动化数据集Pritoni和IBAL（均基于223P schema）上进行实验；

**📈 对比分析**

与始终开启（ON）、始终关闭（OFF）、随机开启、LLM零样本开启等基线对比，路由器在六个评测套件中多达五套提升F1且使用更少token，且在某些场景可匹敌ON；

**⚠️ 局限性**

局限性包括仅在两域内验证、需要已知参考图与文档、离线训练无法适应演进、无法区分检索器与语料库的缺失，以及对大规模知识图的可扩展性未做评估。

---

## 165. Functional Converses for Private Matrix-Product Retrieval under Library-Wide Decoder-Side Correlation

**arXiv ID:** 2611.07325 | [PDF](https://arxiv.org/pdf/2611.07325v1)

**作者:** Malihe Aliasgari `[一作]` (Kean University), Yousef Nejatbakhsh `[通讯]` (Rutgers University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在库宽解码器侧相关情况下的私有矩阵乘法检索，提出了一种功能性对偶方法来处理在多个服务器上存储的矩阵产品的私有检索问题。

**💡 创新点**

创新点在于提出了一种新的隐私定义和通信成本度量方法，考虑了用户持有的数据与服务器存储的库之间的相关性，解决了现有模型中未考虑的相关性对通信成本的影响。

**🔧 技术方法**

使用了功能性对偶理论、线性Slepian-Wolf编码和条件数据安全性等技术。

**📊 数据集**

使用了多个服务器存储的矩阵库，假设这些矩阵与用户持有的数据通过共同的潜在状态相关。

**📈 对比分析**

与传统的私有信息检索（PIR）方法相比，提出的方法在处理库宽相关性时表现出更低的通信成本，具体性能通过理论推导和示例进行了验证。

**⚠️ 局限性**

限制在于模型假设的复杂性和对特定条件的依赖，尤其是在处理不同秩的矩阵时，可能会影响实际应用中的可行性和效率。

---

## 166. Take It From Here: Evaluating Coding Agents On Project Takeover

**arXiv ID:** 2611.07307 | [PDF](https://arxiv.org/pdf/2611.07307v1)

**作者:** Dingxian Wang `[一作]` (Upwork Inc), Zhao Chen `[通讯]` (Upwork Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个项目接手（project takeover）评估框架 ReqMemBench，结合需求诊断与交付验证，系统评估编码代理在长交互历史中重建并实现需求变更的能力。

**💡 创新点**

创新点在于将需求理解诊断（识别相关需求、重构前置状态、更新后置状态）与项目交付验证相结合，提供了两阶段评估流程，并通过“oracle”状态实验揭示需求状态生成与代码实现之间的瓶颈。

**🔧 技术方法**

使用的大模型-工具链配置包括六套GPT与Claude模型，配合Codex CLI/Claude Code等工具，结合需求原子（requirement atoms）与语义判断器进行诊断评分，并通过构建、回归与目标测试三重检验进行交付评估。

**📊 数据集**

实验数据来源于44个真实项目的长期交互记录，共提炼349个接手目标（其中37个具备可执行验证环境），包括软件、文档与PCB三种交付类型。

**📈 对比分析**

比较方法为：①Full History（完整历史）与Relevant History（过滤相关历史）两种上下文长度；②Oracle Pre‑State（提供正确前置状态）与Oracle Post‑State（直接提供目标状态）。结果显示，在Full History下完整交付率仅4.95%，Relevant History略低；Oracle Post‑State提升至8.11%，构建/回归通过率高达96.4%/90.99%，但目标测试通过率仍仅9.01%，反映实现落差。

**⚠️ 局限性**

局限性包括：交付验证仅覆盖37个目标，难以体现多样化项目场景；实验仅针对六套GPT/Claude配置，缺乏对其它代理系统的普适性；需求边界与状态值的语义判定仍依赖人工/模型评估，存在不确定性。

---

## 167. Reducing Representational Interference in Retrieval Embeddings with Trainable Prefix Tokens

**arXiv ID:** 2611.07298 | [PDF](https://arxiv.org/pdf/2611.07298v1)

**作者:** Seongtae Hong `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在检索嵌入模型训练中加入可学习的前缀Token，以减轻残留的无意义Token对表示的干扰。

**💡 创新点**

提出一种轻量级、无需改动模型结构或损失函数的前缀Token策略，通过在输入前置可学习Token来吸收无语义噪声，从而提升检索表示质量。

**🔧 技术方法**

对Transformer编码器/解码器使用对比学习（InfoNCE）损失，加入可学习的DUMP前缀Token，保持原有池化和池化策略。

**📊 数据集**

使用RLHN 100k硬负样本、MLDR 10k长文本数据进行训练；评估使用MTEB/BEIR子集（NQ、HotpotQA、Quora、DBPedia、MS MARCO硬负样本）以及LongEmbed长文本检索。

**📈 对比分析**

与基线模型（无前缀Token）在标准检索和长文本检索上对比，平均提升约2.5%（如ettin-decoder从0.4377升至0.4775，EuroBERT从0.5786升至0.5884，LongEmbed gte-en-mlm从0.4797升至0.5024）。

**⚠️ 局限性**

前缀Token数量的最佳值因模型/数据集而异，增大不一定带来更好效果；方法在极长序列或不同任务上可能需要进一步验证。

---

## 168. Robust Ranking of Port Impacts from Impending Tropical Cyclones: A Two-Stage Recommendation System

**arXiv ID:** 2611.07296 | [PDF](https://arxiv.org/pdf/2611.07296v1)

**作者:** Chenchen Kuai `[一作]` (Texas A&M University), Zihao Li `[通讯]` (Marquette University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用两阶段推荐框架，结合AIS船舶计数和港口、台风特征，预测并排序各港口在台风来袭时的停运天数。

**💡 创新点**

创新点在于将港口受影响度视作推荐系统问题，采用多种特征交互模型（FM、GBDT+LR、WDL）学习港口与台风交互，并通过采样模拟陆降不确定性，提升对停运时长的预测。

**🔧 技术方法**

使用的技术包括因子分解机（FM）、梯度提升树+线性回归（GBDT+LR）、宽深学习（WDL）、时间序列回归识别影响天数以及距离检索方法。

**📊 数据集**

数据集涵盖2017–2022年（除2020）间37个港口的AIS船舶计数、港口属性、港口性能统计、以及北大西洋气象中心提供的31次台风轨迹与强度指标。

**📈 对比分析**

通过对Nicole和Ian两台台风的测试，使用精确度、召回率、F2、MAE、RMSE和AO等指标评估模型；WDL在AO（排名一致性）上显著优于基线（约提升85%/55%），误差最低。

**⚠️ 局限性**

局限性包括：港口历史事件稀缺导致训练样本有限；仅聚焦墨西哥湾37港口，缺乏更广泛区域验证；仅使用单次轨迹不确定性采样，未考虑多时间预测更新；模型泛化性待进一步验证。

---

## 169. Energy-Time Optimal Decoding Using Expander-Graph LDPC Codes and Mesh Network Architecture

**arXiv ID:** 2611.07336 | [PDF](https://arxiv.org/pdf/2611.07336v1)

**作者:** Christopher Blake `[一作]` `[通讯]` (Great Bay University), Christopher Blake (Great Bay University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了第一种能量时间最优的解码方案，达到时钟周期下限。

**💡 创新点**

提出了一种基于扩展图的LDPC编码和网格网络架构的解码方案，首次实现了能量和时间的最优平衡。

**🔧 技术方法**

使用了扩展图LDPC编码、网格网络架构和并行化技术。

**📊 数据集**

使用了具有常数最大节点度的图作为数据集，特别是LDPC解码问题和稀疏矩阵-向量乘法问题。

**📈 对比分析**

与传统方法相比，提出的方法在能量和时间复杂度上达到了基本下限，性能优越，且在多维网格网络中扩展性良好。

**⚠️ 局限性**

未证明所构建的编码方案能够达到信道容量，且仅适用于特定的信道类型。

---

## 170. DC-SQEM: Medial Mesh Extraction by Dual Contouring with Spherical Quadric Error Metric

**arXiv ID:** 2611.07328 | [PDF](https://arxiv.org/pdf/2611.07328v1)

**作者:** Bingchuan Li `[一作]` (Xiamen University), Xiaohu Guo `[通讯]` (University of Texas at Dallas)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于 Dual Contouring 与 Spherical Quadric Error Metric（SQEM）的框架（DC‑SQEM），直接从带法向的点云中提取完整、结构化的三维中轴网（Medial Mesh），无需网格、体素或 RPD 结构。

**💡 创新点**

核心创新包括：
1) 在 Dual Contouring 中引入虚拟节点（Virtual Node）来处理 SQEM 解决方案落在当前细胞外部的情况，保证局部连通性；
2) 通过投影聚类与奇异值阈值相结合的类型降级（Type Downgrading）策略，提高局部类型判定的稳健性；
3) 结合拓扑感知与几何驱动的清理步骤（Solid‑cell 删除、Feature‑preserving 规则、面错误度量），去除多余的跨片连接与冗余面，得到高质量、非流形的中轴网。

**🔧 技术方法**

主要技术包括：
- 自适应八叉树分割与候选细胞筛选；
- 基于点云投影与法向的 SQEM 拟合，求解局部最大球；
- 双曲线型 Dual Contouring（每个活跃细胞产生一个代表性球心点）；
- 虚拟节点机制、类型降级、拓扑/几何清理与三角化；
- K‑近邻图用于投影聚类与特征点筛选。

**📊 数据集**

使用了两个数据集：
- 93 个 ABC CAD 模型（含锐角、尖角、复杂曲面）
- 8 个有机形体（如熊、海豚等），均通过同一采样策略生成约 50 万个带法向点云。

**📈 对比分析**

在多项指标上与八个基线（PC, SAT, VC, Neural Skeleton, VMAS, MATFP, MATTopo, MATStruct）对比：
- MSER：3.09%（排名第2，MATStruct 0.94%）
- TER：2.15%（仅次于 RPD 方法 0%）
- HD：0.42%（最低，MATStruct 1.09%）
- TQ：0.683（与 MATStruct 0.712 相近）
- SITR：4.07%（高于 RPD 方法 0% 但低于 SAT、VMAS）
- 运行时间与 RPD 方法相比相当或更快（在多数模型上 7‑17 秒）。

**⚠️ 局限性**

局限性：
- 虚拟节点会在某些罕见局部配置（如两片面相交角 < 70.5°）下失效，导致连通性破裂；
- 自适应八叉树深度与采样密度对结果敏感，过低分辨率会出现填补错误、缺口；
- 在细长管状结构的 T∞/1D 中轴连接不够稳健，可能出现局部断裂；
- 对噪声、稀疏点云的鲁棒性有限；
- 目前不提供严格的拓扑正确性保证，主要依赖经验参数与后处理。

---

## 171. Rule-Guided and Trajectory-Aware Repair for LLM-Based HLS Generation

**arXiv ID:** 2611.07324 | [PDF](https://arxiv.org/pdf/2611.07324v1)

**作者:** Xianjing Zhu `[一作]` (University of Science and Technology of China), Xuehai Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MementoHLS，一种面向自然语言规格的端到端 HLS 生成框架，利用可执行修复合同和评审驱动的伪证账本实现多轮自动修复。

**💡 创新点**

创新点在于把跨任务 HLS 知识转化为可执行的修复合同（ERCL），并通过任务本地轨迹证据（RFL）对修复过程进行闭环控制，实现可验证、可追溯的修复决策。

**🔧 技术方法**

技术手段包括固定 LLM（如 Llama‑3‑8B、Qwen2.5‑Coder）、vLLM 调度、可执行修复合同库、评审驱动伪证账本、候选绑定审阅与单次提交仲裁，以及 Vitis HLS / Vivado 评估链。

**📊 数据集**

使用 HLS‑Eval（94 任务）和 Bench4HLS（170 任务）两套公开基准进行实验。

**📈 对比分析**

与现有 NL‑to‑HLS 方法（Gai、ContractHIL‑HLS、HLS‑Seek 等）比较，MementoHLS 在同候选 TBSynth Pass@1/Pass@5 上分别提升约 19.1/17.7 百分点（HLS‑Eval）和 32.4/50.6 百分点（Bench4HLS），显著提高生成可靠性。

**⚠️ 局限性**

局限性包括：仅支持固定 LLM 与最多 5 轮修复；目前仅验证了 Vitis HLS / Vivado 流水线；对极大或结构极其复杂的算法支持有限；以及未完全解决性能（延迟、资源）优化的需求。

---

## 172. ArtiVidRM: Verifiable Artifact-as-Rubric Reward Modeling for Explicit Quality Articulation in Video Generation

**arXiv ID:** 2611.07301 | [PDF](https://arxiv.org/pdf/2611.07301v1)

**作者:** Caorui Li `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一种基于可视缺陷列表的奖励模型ArtiVidRM，用于评估和强化学习优化视频生成质量。

**💡 创新点**

创新点在于先识别生成缺陷并形成结构化artifact列表，再以此列表作为可验证的评判尺度，并在强化学习中同时优化artifact识别与偏好预测。

**🔧 技术方法**

采用大型视觉语言模型Qwen3‑VL‑8B‑Instruct进行SFT与GRPO强化学习，并引入格式奖励、语义召回奖励、长度惩罚以及LLM语义匹配等技术。

**📊 数据集**

使用VideoGen‑RewardBench构建Artifact‑augmented Preference Data，并在GenAI‑Bench、VBench‑2.0、MonetBench、T2VQA‑DB、VideoFeedback2等公开基准上进行评估。

**📈 对比分析**

与VisionReward、VideoScore、VideoAlign等基线在pairwise preference、ranking、点评分等指标上对比，ArtiVidRM在大多数指标上实现最高或相当水平；在VBench优化任务中提升Motion Smoothness和Imaging Quality，在闭源Veo‑3.1迭代中也获得显著质量提升。

**⚠️ 局限性**

局限性包括对大模型推理成本高、artifact列表可能存在漏报或误报、闭源模型改进仅靠提示有限以及对长视频或更复杂多模态任务的适应性仍需提升。

---

## 173. Householder Motion Embedding: Noise Modulation for Motion-Controllable Video Generation

**arXiv ID:** 2611.07294 | [PDF](https://arxiv.org/pdf/2611.07294v1)

**作者:** Tao Jun Lin `[一作]` (Australian National University), Hongdong Li `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种新的噪声调制方法 Householder Motion Embedding (HME)，通过在通道空间使用 Householder 反射对每个像素位置的噪声进行正交变换，从而在保持空间高斯分布的同时实现时间相关的运动控制。

**💡 创新点**

创新点在于：① 用正交 Householder 变换代替传统空间噪声扭曲，完全消除后期高斯性校正；② 在通道空间编码运动信息，跨帧相关性自然由正交矩阵决定；③ 仅需要少量反射（K≈2）即可获得良好性能，显著提升计算速度与显存占用。

**🔧 技术方法**

技术包括：光流估计（RAFT）、光流到通道空间的线性投影、Householder 反射序列生成正交变换、噪声降解参数 γ 控制运动遵从与生成自由度、基于 CogVideoX 的视频扩散模型微调。

**📊 数据集**

使用 RealCam‑Vid、DAVIS、MC‑Bench、DL3DV 等公开视频数据集进行训练与评估，训练集约 100K 条视频，测试集包含 45 条 DAVIS、965 条 MC‑Bench、200 条 DL3DV 场景。

**📈 对比分析**

与现有噪声扭曲方法（GWTF、HIWYN、InfRes）以及监督式运动控制方法（MotionPro、TTM、DisMo、ImageConductor）对比，HME 在 VBench 视觉一致性、FVD、FID、光流 EPE、CoTracker mIoU、PSNR 等指标上均表现更好，并实现约 2.7× 的噪声处理速度提升和更低的显存占用。

**⚠️ 局限性**

局限性包括：对光流质量敏感；反射数 K 的选择需要权衡表达能力与计算成本；对复杂非刚性运动仍存在挑战；需要对扩散模型进行微调才能充分利用噪声编码；并且实验基线使用官方模型，未在相同训练数据上重训练。

---

## 174. Vis2State: Visual Reasoning through Executable State Abstractions

**arXiv ID:** 2611.07397 | [PDF](https://arxiv.org/pdf/2611.07397v1)

**作者:** Sheng Jin `[一作]` (SenseTime Research and Tetras.AI), Chen Qian `[通讯]` (SenseTime Research and Tetras.AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Vis2State 框架，将视觉谜题拆解为可视化的状态和可执行程序，支持多次检视与执行的交互式推理。

**💡 创新点**

引入显式可修改的“状态”作为视觉理解与计算推理的中介，使得视觉解释与求解过程可以独立检验和修订，并且不依赖训练。

**🔧 技术方法**

结合视觉工具（如 V*、Visual Sketchpad 等）进行图像检视、JSON 状态构建，以及 Python 代码执行与回调；使用大语言模型（DeepSeek V4.1 Flash、GLM-5.3 Flash 等）作为主控。

**📊 数据集**

在四大视觉推理基准上进行实验：AlgoPuzzleVQA、VisualPuzzles、VisuLogic、PuzzleVQA。

**📈 对比分析**

与官方 Direct CoT 及多种大型模型做对比，Vis2State 在所有四个基准上均取得最高准确率，提升幅度 8–15pp；跨模型评估亦表现出一致性。

**⚠️ 局限性**

存在推理开销增加、状态与程序可能出现错误且缺乏自动验证机制；对某些感知与规则推断任务仍易出现误差。

---

## 175. CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment

**arXiv ID:** 2611.07393 | [PDF](https://arxiv.org/pdf/2611.07393v1)

**作者:** Vihan Lakshman `[一作]` (Massachusetts Institute of Technology), Michael Cafarella `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将大模型（4B）生成的 KV cache 与小模型（0.6B）进行离线融合，生成只需小模型即可解码的 KV cache，在线检索时直接加载该 KV cache 并生成答案。

**💡 创新点**

首次把 Cache-to-Cache（C2C）KV cache 翻译技术从实时多代理通信迁移到检索增强生成（RAG），实现了无查询时依赖大模型的文档丰富，并且在零shot 训练下即可提升性能。

**🔧 技术方法**

使用 Qwen3-4B 作为 sharer、Qwen3-0.6B 作为 receiver；采用公开的 C2C fuser；检索使用 BGE‑small embedding；评价采用两种 LLM‑judge（Llama‑2‑7B‑Chat 与 StableVicuna‑13B）和 contains 指标；对比使用 text‑RAG 与仅 sharer‑only。

**📊 数据集**

SQuAD v1.1、AdversarialQA、CovidQA、QASPER、TriviaQA 五个知识密集 QA 语料库。

**📈 对比分析**

通过 LLM‑judge 进行自动评测，并用两组独立模型对答案质量做对比；结果显示 CacheGraft 在 29/30 次比较中均优于 text‑RAG，17 次达到统计显著（p<0.05），提升幅度最高达 8 百分点；在 oracle 检索和 top‑1 检索两种场景下均保持显著优势。

**⚠️ 局限性**

主要限制包括：KV cache 需要存储每条文档，空间占用远大于文本（数 TB）；仅支持单文档检索、未针对检索任务训练 fuser；当前未实现 top‑k 检索与多模型扩展；整体模型仍受限于小模型的推理速度。

---

## 176. WeLM-Image: A Unified Image Generation Model towards Visual Intelligence

**arXiv ID:** 2611.07392 | [PDF](https://arxiv.org/pdf/2611.07392v1)

**作者:** Canmiao Fu `[一作]`, Zhipeng Huang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一的图像基础模型 WeLM-Image，能够在同一条多模态序列上完成文本到图像的生成、指令引导的图像编辑和多参考合成。

**💡 创新点**

创新点包括：①双路径 Mixture‑of‑Transformers（MoT）+Mixture‑of‑Experts（MoE）骨干，理解和生成分别拥有独立参数但共享注意力；②统一的“交错”文本‑图像序列表示，所有任务通过注意力掩码和任务标签共享同一 Transformer；③文本兼容的 3D RoPE 统一了语言与视觉空间位置编码；④针对单 GPU、低推理成本的设计（可压缩 31.7B 参数、8 步无引导推理、单 GPU 5.4B 激活参数）。

**🔧 技术方法**

核心技术包括：流匹配训练（Flow‑Matching）用于生成目标；MiT‑MoE 模型；Sparse Mixture‑of‑Experts（每路 64 路专家+2 路共享）；3D RoPE；Token‑Packing + FSDP + Context Parallelism；强化学习 + On‑Policy Distillation + MeanFlow 蒸馏；多语言（中英）标注和 OCR；大规模视觉词表与聚类数据引擎。

**📊 数据集**

使用约 0.6 B 的清洗后图文数据（来自内部、公开和授权来源），通过知识驱动视觉词表和层次聚类构建任务数据；同时包含多语言文本、OCR 信息、语义标签、编辑对、参考图像等；训练数据分阶段，先做多模态预训练，再逐步加入生成、编辑和多参考任务。

**📈 对比分析**

通过一系列公开基准（PRISM‑Bench、Qwen‑Image‑Bench、UnicBench、LongText‑Bench 等）与主流开源模型对比，WeLM‑Image 在 1536 分辨率下获得第一名；在 1024×1024 的无引导生成中，8 步蒸馏版在单 NVIDIA H20 GPU 上仅需 4.98 s（比 50 步基线快 11.4 倍）；在编辑任务中，单图、双图、三图、四图的推理时间几乎线性增长，且 1.96×–5.10×快于 Qwen‑Image‑Edit‑2511。整体吞吐量（QPM/GPU）与同级别模型相比提升 20%~30%。

**⚠️ 局限性**

局限性包括：①训练规模依赖 30B 级参数，仍需多 GPU 训练；②推理仍受限于 1536 分辨率和 8 步步长；③模型专注图像生成与编辑，其他视觉任务（如检测、分割）未覆盖；④对极端长文本或多语言混杂的鲁棒性仍有提升空间。

---

## 177. Design of a Speech Enhancement-Based Deep Semantic Autoencoder for Wireless Communications

**arXiv ID:** 2611.07361 | [PDF](https://arxiv.org/pdf/2611.07361v1)

**作者:** Xiaoling Han `[一作]` (Dalian Maritime University), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于语音增强的深度语义自编码器（S‑DSA）通信系统，采用双分支语义编码器、通道编码器/解码器以及两条语义解码器，实现端到端的语义与信道联合编码与解码；

**💡 创新点**

创新点包括自适应语义噪声解耦机制，通过交互模块实现语音与噪声特征在语义空间中的正交分离；在推理阶段禁用噪声解码路径，实现轻量化推断；以及联合训练策略与多重损失函数的组合优化；

**🔧 技术方法**

技术手段主要是深度卷积网络、残差+注意力块、可分离自注意力（SSA）、语音增强的复数谱掩码估计、端到端联合源信道编码、以及多尺度STFT损失；

**📊 数据集**

实验使用公开语音语料与模拟噪声（AWGN、Rician、Rayleigh）混合生成的混合信号进行训练与评估，具体数据集未在文中列明；

**📈 对比分析**

与CNN‑AE、DeepSC‑S以及理想条件下的S‑DSA@Oracle进行对比，使用MSE、SI‑SDR、PESQ、STOI等指标，在低信道SNR和多种噪声环境下均表现出更低误差、更高SI‑SDR和更优感知质量；

**⚠️ 局限性**

局限性主要在于仅验证了单一噪声模型和仿真信道，未覆盖多噪声类型和真实环境；未来工作需提升对多噪声场景的鲁棒性和自适应能力。

---

## 178. Constant-Factor Approximation Algorithms for Weighted Budgeted Multisensor Sweep Coverage

**arXiv ID:** 2611.07355 | [PDF](https://arxiv.org/pdf/2611.07355v1)

**作者:** Jingyang Zhao `[一作]` (Kyung Hee University), Hyundong Shin `[通讯]` (Kyung Hee University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了两种常数因子近似算法，用于求解加权预算多传感器扫掠覆盖（WBSCP）问题，并将方法扩展到传感器速度不均的异构场景；

**💡 创新点**

创新点在于：① 通过从任意α,β双因子近似算法转换为β/2α近似，首次给出1/16的近似比；② 采用短窗口平均与最小超量路径子例程的贪心最大覆盖框架，将近似比提升到1–e^(−1/2)−ε；③ 通过群覆盖（GBMCP）将异构速度问题归约到同质情况，实现同样的近似比；

**🔧 技术方法**

主要技术包括：路径分解与短环化、双因子近似转换、短窗口平均论证、最小超量路径（q‑MEP）近似、贪心覆盖、群覆盖框架与条件期望反向求解；

**📊 数据集**

本文为理论研究，不使用实验数据集；

**📈 对比分析**

与此前最优0.0116‑O(ε)的结果相比，1/16已是同质问题的首次常数近似；通过贪心框架进一步提升至1–e^(−1/2)−ε≈0.393，对异构速度也保持同样性能；

**⚠️ 局限性**

局限性在于：近似比虽为常数，但仍远离最优（1），未证明可进一步提升；算法依赖于完整度量图和统一扫掠周期，难以直接适用于非度量空间、动态权重或有部署约束的实际场景。

---

## 179. Constant-Factor Approximations for the Biased Traveling Salesman Problem and Its Path Variant

**arXiv ID:** 2611.07348 | [PDF](https://arxiv.org/pdf/2611.07348v1)

**作者:** Jingyang Zhao `[一作]` (Kyung Hee University), Hyundong Shin `[通讯]` (Kyung Hee University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对在局部满足三角不等式但跨区域可能被乘以偏置因子（biased）的旅行商问题（BTSP）及其路径变体（BTSPP），提出了多区域的常数因子近似算法。

**💡 创新点**

创新点在于引入状态序列压缩框架，证明任意最优解可以通过合法压缩得到长度仅依赖于区域数k的“不可约状态序列”，从而能枚举有限种结构并构造近似解；并在k=2的路径变体中进一步利用状态结构改进至α+2+ε（α<3/2）逼近。

**🔧 技术方法**

核心技术包括：
- 状态序列（每个状态记录同一区域内的起终点）
- 合法压缩（MO‑1、MO‑2）与闭合子链概念
- 递推上界fk=O(2^k k!) 证明不可约序列长度有限
- 对每个子图使用经典的度量TSP/路径近似（双树算法或α≈1.4999近似）
- 通过对压缩后序列进行补丁（patching）得到完整 Hamiltonian 路径/环
- 对k=2路径变体采用平均论证与多种结构分类，实现α+2+ε逼近。

**📊 数据集**

无实验数据集，全部结果均为理论分析与近似比率证明。

**📈 对比分析**

与现有仅在k=2时已知的7/2、4逼近相比，本文提供了任意固定k的常数因子逼近（BTSP为αk+1，BTSPP为2k+1），并在k=2路径变体中把4逼近提升到α+2+ε（<3.5+ε）。

**⚠️ 局限性**

局限性：
- 算法时间为N^{O(2^k k!)}，对大k不可行；
- 逼近比率仍非最优，尤其在k>2时逼近系数随k增长；
- 只处理完全图的biased模型，未考虑更一般的偏置结构或多目标场景。

---

## 180. Deep reinforcement learning-based optimization of multi-RAT handovers

**arXiv ID:** 2611.07390 | [PDF](https://arxiv.org/pdf/2611.07390v1)

**作者:** Pavan Kumar Mangipudi `[一作]` (University of Florida), Janise McNair `[通讯]` (University of Florida)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了可配置的深度强化学习框架 ReFlexMRS，用于多RAT环境下的多标准手移动目标选择，并在 SDN 控制的 5G‑WiFi 仿真平台上进行评估。

**💡 创新点**

创新点在于：① 将多种 QoS 指标（RSSI、链路延迟等）融入状态表示并通过卷积层捕捉节点间关联；② 提供网络独立与部署专属两种奖励设计，支持运营商偏好（如 WiFi 负载卸载）；③ 通过离线 DQN 训练实现快速推理，兼顾多标准和移动性。

**🔧 技术方法**

使用了深度 Q 网络（DQN）与卷积层进行状态特征提取；SDN 控制器收集测量并执行手动决策；Mininet‑WiFi 搭建 5G‑WiFi 多RAT 仿真网络；离线经验回放和 Softmax 探索实现训练。

**📊 数据集**

训练数据来自 SDN‑启用的 5G‑WiFi 仿真环境中收集的 RSSI 与平均链路延迟测量，使用 7 次 60 分钟的仿真跑产生的观测序列进行离线训练。

**📈 对比分析**

与传统基于 RSSI 的手动决策做对比。评估指标包括链路延迟差、总手移动次数、WiFi‑to‑5G 手移动次数以及平均累计吞吐量。结果显示：DRL1/DRL2 分别将链路延迟降低约 184%/259%，总手移动减少 12%/18%，WiFi‑to‑5G 手移动减少 6%/13%，并保持 90%/95% 的基线吞吐量。

**⚠️ 局限性**

局限性包括：① 仅在仿真环境中进行离线训练，未实现在线闭环学习；② 未涵盖完整 5G NR 协议栈与标准化手移动细节（如信令延迟、切换失败率）；③ 评估范围受限于单一 5G‑WiFi 场景，难以推广到更复杂多 RAT 部署或不同移动模型。

---

## 181. Adversarial Multiplicity: Cherry-Picked Models and the Limits of Disclosure

**arXiv ID:** 2611.07378 | [PDF](https://arxiv.org/pdf/2611.07378v1)

**作者:** Cassandra C. Chou `[一作]` (Harvard University), Elizabeth T. Chin `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了预测多样性背景下，恶意主体通过硬（直接改参数）与软（模型搜索）操纵模型以实现利益最大化，并分析了此类操纵对法规审计与监管的影响。

**💡 创新点**

创新点在于：①用理论证明任何个体预测可在保持准确率的前提下被硬操纵反转；②展示软操纵在多指标评估下仍能显著影响决策；③评估Reconcile算法在对抗性多样性中的局限性；④系统性绘制现行法规对抗恶意多样性的覆盖缺口。

**🔧 技术方法**

技术手段包括：理论证明、LightGBM/ CatBoost/ TabPFN 等树/神经网络的硬参数编辑与软搜索；多指标评估（Brier、AUC、校准、影响比）；聚合估计实验（火灾归因 FAR 计算）；Reconcile 算法对比；法规映射与预注册、版本化建议。

**📊 数据集**

使用的数据集：HELOC、COMPAS、Adult、German 信用数据，以及加州 2003–2020 年野火极端增长数据。

**📈 对比分析**

比较方法：在 1% Brier 误差范围内进行硬/软操纵；测量决策逆转率、FAR 变化；在多指标下评估软操纵的抵抗力；对比 Reconcile 在对抗性模型与随机模型下的决策恢复率。结果显示硬操纵可在几乎不损失性能的情况下逆转 90% 以上个体决策；软操纵在多指标下仍能逆转 10–70% 决策；Reconcile 对抗性模型的成功率明显低于随机模型。

**⚠️ 局限性**

局限性：实验仅覆盖四个公开基准和火灾数据，未考察更复杂模型（如视觉/语言模型）对多样性与操纵的影响；硬操纵假设模型足够灵活且叶子可自由编辑；软操纵效果受模型搜索规模与超参数空间限制；法规分析基于已公布政策，可能不完全涵盖所有司法区与最新修订。

---

## 182. When Scores Conflict with Preferences: Pre-Alignment Diagnosis and Adaptive Correction of Constraint Drift in DPO

**arXiv ID:** 2611.07375 | [PDF](https://arxiv.org/pdf/2611.07375v1)

**作者:** Ping Liu `[一作]` (University of Nevada), Yan Yan `[通讯]` (Washington State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过理论分析揭示了在存在点对点与点对点评价共存的条件下，直接偏好优化（DPO）中约束漂移的根本原因，并提出一种闭式诊断量ρ^*_eq用于在训练前判断数据集是否需要干预；基于此诊断量，设计了冲突感知DPO（CoDPO），在训练前使用ρ^*_eq确定冲突对的权重下调，并在训练中动态偏移DPO边际以消除残留漂移；

**💡 创新点**

创新点主要包括①从冲突对与匹配对的压力与修复平衡出发推导出闭式预先诊断量ρ^*_eq，能在不调参的前提下决定是否以及干预力度；②在CoDPO中首次将权重下调与动态边际偏移两级控制结合，实现对漂移的精准抑制，同时保持偏好学习；

**🔧 技术方法**

使用技术包括DPO、PSDPO的冲突加权、点对点与点对点约束的第一阶分析、参考策略的倾斜实现边际偏移、基于梯度的漂移估计与自适应λ调节；

**📊 数据集**

实验数据集涵盖六个冲突占比高的域（HelpSteer2、GSM8K、ARC、Math-Calc、TL;DR）以及一个自校正域PKU-SafeRLHF，用于验证诊断与CoDPO的效果；

**📈 对比分析**

与Vanilla DPO、硬过滤、SafeDPO、PD-DPO、C2-DPO、MODPO、Rewarded Soups等方法比较，CoDPO在降低约束漂移的同时，保留甚至提升了偏好学习性能，且在漂移-偏好学习权衡上表现出更优的曲线；

**⚠️ 局限性**

局限性包括①假设更新足够小且使用第一阶近似，可能在大步长或非线性阶段失效；②诊断量ρ^*_eq仅基于初始化统计，未跟踪训练过程中的动态平衡；③对冲突严重度参数p需经验选择；④实验集中在固定域与pairwise数据，未覆盖在线生成对的情况。

---

## 183. Test-Time Self-Play: Scaling without a Verifier

**arXiv ID:** 2611.07367 | [PDF](https://arxiv.org/pdf/2611.07367v1)

**作者:** Akira Yoshiyama `[一作]` (ETH Zurich), Andreas Krause `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在目标问题无可靠验证器或弱验证器环境下，通过自我对弈生成合成问题并对解算器进行测试时自适应的测试时自我对弈（TTSP）方法，提升大型语言模型在单一难题上的表现

**💡 创新点**

创新点在于：①不依赖目标问题的验证器，直接利用自生成的合成问题的执行反馈作为学习信号；②通过两者（提议者与解算者）分离的LoRA适配器实现异步自我对弈；③构造基于目标问题的合成任务与历史记录的上下文，形成针对单一目标的专属自适应课程

**🔧 技术方法**

使用Qwen3‑8B基础模型的冻结Backbone，配合两套LoRA适配器实现提议者和解算者；利用GRPO（Generalized Reward‑oriented Policy Optimization）进行策略更新；合成问题生成基于目标问题的上下文与历史，执行验证使用Python解释器；数据集使用LiveCodeBench v6（竞赛编程题目）

**📊 数据集**

数据集：LiveCodeBench v6，包含131个从2025年2月至5月发布的竞赛编程题目，按难度划分为HardLCBv6、MidHardLCBv6和VeryHardLCBv6

**📈 对比分析**

与三类基线（基于拒绝采样的重抽样、功能多数投票FMV、上下文优化ACE）在不同验证器强度（v_0到v^*）下进行对比；TTSP在所有验证器强度下均保持最高mean@8，尤其在无验证器（v_0）时提升≈35%；其他方法随着验证器弱化性能明显下降

**⚠️ 局限性**

局限性：①仅对每个目标问题单独适配，无法跨问题复用计算资源；②依赖基线模型对目标问题已有一定覆盖，否则无法生成有效课程；③训练时可能出现提议者或解算者偏离真实目标的“漏洞”或熵坍塌，需要稳定性监控；④在“发现”模式（模型对目标答案几乎无覆盖）下效果有限

---

## 184. WeightLoop: Weight-Space Programming with Experimental Feedback

**arXiv ID:** 2611.07319 | [PDF](https://arxiv.org/pdf/2611.07319v1)

**作者:** Mengran Li `[一作]` (Hong Kong Institute of Science and Innovation), Stan Z. Li `[通讯]` (Westlake University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 WeightLoop 框架，利用 GPT-6 Astra 生成可重用的编辑器程序，基于实验反馈持续更新已部署的预测模型，更新过程不再需要再次调用 LLM 或梯度训练。

**💡 创新点**

创新点在于将权重空间编程（WSP）与递归自我改进（RSI）相结合，创建一个固定的 LLM 编辑器，使实验数据直接驱动模型参数更新，形成闭环的持续改进流程；此外实现了在多轮实验中仅通过写回检查点即可实现参数更新的机制。

**🔧 技术方法**

使用的技术包括：GPT-6 Astra 生成编辑器、权重空间编程（WSP）、基于学习到的排序损失与残差修正的线性/树模型编辑、实验测量验证、无梯度训练直接写回模型参数、对模型的可重用性与持久化更新支持。

**📊 数据集**

使用的数据集为公开的 DepMap 功能筛选数据（涵盖 HepG2、Jurkat、HL60 等细胞系的生长、迁移、LDL 摄取等测量）以及 HuH7 细胞的自噬流与 LDL 摄取实验数据。

**📈 对比分析**

与随机编辑和原始模型进行对比，单轮排名实验中在七/八个端点–预算设置下平均排名效用显著提升；顺序发现实验中编辑策略在所有任务的发现率 D_100 上均优于随机和原始；系数/叶值编辑在 MSE 上分别提升 10%/13% 和 2.4%/2.6%；参数更新仅改动极少量权重，预测结果保持一致，且更新可持久化。

**⚠️ 局限性**

局限性包括：实验仅覆盖少量细胞系与任务，未验证更广泛的生物学场景；依赖 GPT 生成的编辑器，若任务结构与假设不符可能效果下降；更新仅限于可编辑参数，未能处理模型结构变化；实验反馈量有限，更新效果受限；未在真实部署环境进行长期评估。

---

## 185. Asymptotic advantage of quantum one-way and SMP communication for total functions

**arXiv ID:** 2611.07318 | [PDF](https://arxiv.org/pdf/2611.07318v1)

**作者:** Dmytro Gavinsky `[一作]` `[通讯]` (Institute of Mathematics of Czech Academy of Sciences), Dmytro Gavinsky (Institute of Mathematics of Czech Academy of Sciences)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构造了一种总布尔函数，该函数在量子一次性/并行消息传递模型下需要线性量子位，而在经典一次性/并行消息传递模型下至少需要二次量子比特，并给出了利用共享 EPR 对实现的经典 SMP 协议。

**💡 创新点**

创新点在于首次给出明确的总布尔函数，展示了量子与经典并行/一次性通信复杂度之间的二次分离；同时提出了一个新的矩阶矩估计方法，用于证明经典下界；并证明共享纠缠对经典 SMP 没有实质性降低成本。

**🔧 技术方法**

使用技术包括：量子协议中构造特殊的正交基向量 ψ_M(a) 并利用投影测量；并行消息协议中利用共享 EPR 对与 Hadamard 变换；经典下界则采用傅里叶变换、Parseval 定理以及自正交子空间的组合计数方法；整体结合了线性代数、离散 Fourier 分析与信息论。

**📊 数据集**

无实际数据集；研究完全基于理论构造，输入为二进制矩阵和向量，运算域为 F₂。

**📈 对比分析**

通过理论证明比较量子与经典模型的通信复杂度，量子一次性/并行消息传递可在 O(n) qubits 内完成，而经典一次性/并行消息传递至少需要 Ω(n²) bits；使用共享 EPR 对的经典 SMP 协议可在 O(n) 经典比特 + n 个 EPR 对内完成，展示了理论上可行但实验实现仍具有挑战性。

**⚠️ 局限性**

局限性在于仅针对该特定函数给出分离结果，无法直接推广到更一般的函数；此外证明依赖于高阶矩估计和自正交子空间的组合结构，对其它类型问题的适用性尚未证明；实验实现上需要大量 EPR 对，实际成本和噪声会影响可行性。

---

## 186. FFMT: Frequency-Factorized Multi-Rate Tokenization for Music-to-Dance Generation

**arXiv ID:** 2611.07384 | [PDF](https://arxiv.org/pdf/2611.07384v1)

**作者:** Qirong Guo `[一作]` (Hong Kong University of Science and Technology), Weihang Ding `[通讯]` (University of California)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种频率分解的运动令牌化方法（FFMT），将根相对运动拆分为低频舞蹈骨架和高频细节，并在音乐驱动的舞蹈生成中采用计划-先行-细化的自回归生成流程。

**💡 创新点**

核心创新在于：①显式将运动按频率分解，分配不同的时间步长和代码簿容量；②低频令牌先生成粗略舞蹈结构，高频令牌再在其条件下细化细节；③结合训练得到的解码先验提高采样质量。

**🔧 技术方法**

使用 VQ‑VAE 进行双分支编码解码，Transformer 自回归生成（低频层 + 高频层），Gaussian 低通滤波做频率分离，音乐与舞蹈对齐的注意力机制，以及基于训练数据的解码先验。

**📊 数据集**

在 AIST++ 和 FineDance 两个公开舞蹈数据集上进行实验。

**📈 对比分析**

相较于现有音乐到舞蹈方法，FFMT 在 FID_k/FID_g 上实现最低值，分布拟合更好，且多样性与节拍对齐评分（BAS）接近真实数据，说明该方法在保持真实感和多样性方面表现突出。

**⚠️ 局限性**

局限性包括：对高频细节的重建仍受限于代码簿容量，需较大的模型与训练成本；在不同舞蹈风格或更复杂音乐场景下的泛化能力尚待验证；以及低频与高频间的依赖仍可能出现同步失配。

---

## 187. Breaking the Modality Barrier: Multimodal Fusion With Missing Modalities for Cybersickness Prediction

**arXiv ID:** 2611.07386 | [PDF](https://arxiv.org/pdf/2611.07386v1)

**作者:** Purnata Saha `[一作]` (University of Texas at San Antonio), John Quarles `[通讯]` (University of Texas at San Antonio)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对虚拟现实中的网络眩晕预测，本文提出了一种能够处理缺失传感器并可跨数据集训练的多模态Transformer框架。

**💡 创新点**

创新点在于将眼动追踪作为锚点进行对比学习，并在Transformer自注意力中引入缺失掩码，使模型能在任意子集传感器下进行推理；同时通过跨数据集联合训练实现了泛化能力的提升。

**🔧 技术方法**

使用的技术包括：多模态一维卷积编码器、锚点对比对齐（contrastive anchor alignment）、掩码多模态Transformer（masked multimodal transformer）以及随机模态丢失的训练策略。

**📊 数据集**

实验所用数据集为VR.net、SIM21和Blur‑Rollercoaster三大公开VR眩晕数据集，三者分别涵盖眼动、头部运动、皮肤电反应、心电图等不同模态组合。

**📈 对比分析**

在单数据集模态消失实验中，模型在不同传感器缺失下MAE仅变化不到0.1，证明鲁棒性；在跨数据集联合训练实验中，MAE降至0.51、皮尔逊相关系数提升至0.63，明显优于单数据集模型。

**⚠️ 局限性**

局限性包括：未在实时推理环境下验证、对时间序列内部缺失值处理不足、对训练集缺失的模态在推理时无法使用，以及对心率等派生特征未做统一编码。

---

## 188. Asymptotic-Preserving Tensor Neural Networks for 3D High-Field Kinetic Equations with Applications to Uncertainty Quantification

**arXiv ID:** 2611.07364 | [PDF](https://arxiv.org/pdf/2611.07364v1)

**作者:** Shi Jin `[一作]` (Shanghai Jiao Tong University), Tian-ai Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种自适应张量神经网络(APTNN)用于高场制约的高维Kinetic方程，并给出了高效的维度分离损失评估方法。

**💡 创新点**

创新点在于将CP分解与张量网络结合，实现指数级降维，并通过引入一阶速度动量方程显式约束流速闭合，从而提升高场极限下的预测精度。

**🔧 技术方法**

主要技术包括CP张量网络、维度分离的积分策略、异步的首动量约束AP损失、自动微分与JAX/Flax实现。

**📊 数据集**

使用的测试数据集为一维Landau阻尼、二维双流、三维三维BGK–Poisson和VPFP系统，并加入3维与5维随机变量进行不确定性量化。

**📈 对比分析**

与传统TNN和基线APNN对比，APTNN在相同网络规模下在高场极限下误差从10⁻²提升到10⁻³，训练时间约30分钟，显著低于全网格评估的GPU内存需求。

**⚠️ 局限性**

局限性包括对低秩假设的依赖、缺乏秩-误差定量关系、以及在非平衡或激烈不稳定场景下可能需要更高秩导致内存瓶颈。

---

## 189. A Synchronized Tri-Modality Wireless System for Multi-Point Non-Contact Cardiac Monitoring

**arXiv ID:** 2611.07358 | [PDF](https://arxiv.org/pdf/2611.07358v1)

**作者:** Wei Ren `[一作]` (Hong Kong University of Science and Technology), Fugui Qi `[通讯]` (Fourth Military Medical University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建了一套整合 ECG、SCG 与 DCG 的三模态无线系统，用于同步、可穿戴的心脏生理监测；

**💡 创新点**

首次实现三模态同步采集，提供直接机械参考（SCG）验证雷达（DCG）的定位精度，并实现多点胸壁测量与关键特征点提取；

**🔧 技术方法**

采用 mmWave 雷达（MIMO FMCW）、可穿戴 ECG/SCG 传感模块（ESP32‑C6、ADXL355、ADS1292）、低通/高通 Butterworth 滤波、R 峰检测、基于窗口的关键点提取算法；

**📊 数据集**

实验数据集包括：① 5 轴线性运动平台的 SCG 与 DCG 对比；② 3 名健康男性受试者在平躺姿势下进行自然呼吸与屏气两种呼吸条件下的 30 秒/ 3 分钟同步记录；

**📈 对比分析**

与 SCG 位移进行定量比较，RMSE 0.0045 m、Pearson r 0.9947，显示 DCG 位移与 SCG 极为一致；交叉模态关键点（AO、MO 等）在三模态间保持高度一致，验证了系统的精度与鲁棒性；

**⚠️ 局限性**

局限在于需要受试者保持静止，且雷达阵列与 SCG 传感点需手动空间校准，导致在真实环境下对运动抑制与自动定位的需求尚未解决。

---

## 190. Slide to Report: Continuous Reporting Symptom-Specific Cybersickness in VR

**arXiv ID:** 2611.07345 | [PDF](https://arxiv.org/pdf/2611.07345v1)

**作者:** Purnata Saha `[一作]` (University of Texas at San Antonio), John Quarles `[通讯]` (University of Texas at San Antonio)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究提出并验证了一个三轴物理滑块接口，用于在VR中持续、症状特异性（恶心、眼动不适、方向失调）地记录网络晕动症；

**💡 创新点**

创新点在于：①将三种核心症状分别映射到独立滑块，实现多维连续监测；②验证该方法与传统SSQ子量表的对应性；③公开了包含GSR、ECG、PPG、眼/头跟踪以及Simon任务的完整多模态数据集。

**🔧 技术方法**

技术手段包括：HP Reverb G2 Omnicept VR头盔、Polar H10心电胸带、Shimmer Consensys GSR/PPG传感器、线性电位计滑块、Unity3D构建的过山车场景以及听觉Simon任务。

**📊 数据集**

使用的数据集为52名受试者的公开多模态记录，包含滑块实时值、SSQ问卷、GSR、ECG、PPG、眼球、头部运动以及Simon任务的反应时间与准确率。

**📈 对比分析**

通过与SSQ子量表的Pearson相关检验（H1,H2）以及单滑块与三滑块之间的RT、准确率、SUS与NASA‑TLX比较，结果表明三滑块与对应子量表相关性较高且未显著增加认知负荷或可用性差异；

**⚠️ 局限性**

局限性包括：①等效性检验功效不足；②单滑块与三滑块训练内容不同，导致对比受干扰；③SSQ症状重叠导致子量表区分不够清晰；④滑块自身可能对晕动症产生抑制作用。

---

## 191. Rethinking Data Selection: The Importance of Coverage over Difficulty in LLM Fine-Tuning

**arXiv ID:** 2611.07323 | [PDF](https://arxiv.org/pdf/2611.07323v1)

**作者:** Lalchand Pandia `[一作]` (University of Chicago), Allyson Ettinger `[通讯]` (Allen Institute for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了难度评分在LLM微调中的有效性，发现随机采样和基于覆盖率的采样方法往往优于传统难度评分。

**💡 创新点**

提出仅基于覆盖率的聚类采样方法，并证明覆盖率优先可在多域任务上与完整微调相当甚至更优，优于传统难度评分。

**🔧 技术方法**

使用LLM最后隐藏层嵌入、k‑means聚类、Jensen‑Shannon Divergence评估覆盖率，比较了Perplexity、Confidence、Variability、VoG、PVI等难度指标。

**📊 数据集**

实验数据集包括单域多选问答集（Social IQa、CommonsenseQA、CosmosQA）以及多域Tülu 3，并在MMLU、BBH、HumanEval等基准上评估。

**📈 对比分析**

与随机采样、最优难度评分VoG、D²平衡方法对比，覆盖率方法在多域任务上与完整微调表现相当或更好，在单域任务上与随机相当；随机在部分情况仍保持最优。

**⚠️ 局限性**

局限性包括难度评分需要完整训练成本高，覆盖率方法仍需一次前向推理，缺乏因果关系证明，且在单域数据覆盖率不足时效果不明显。

---

## 192. DiagDBench: Evaluating Agentic LLMs on Interactive Database Diagnosis and Anomaly Remediation

**arXiv ID:** 2611.07339 | [PDF](https://arxiv.org/pdf/2611.07339v1)

**作者:** Xinxin Zhao `[一作]` (Renmin University of China), Jing Zhang `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于训练和评估数据库代理的基准，专注于交互式数据库诊断、异常修复和可选的工作负载性能优化。

**💡 创新点**

通过引入多样化的异常注入机制和真实世界的基础，构建了一个大规模的数据集，包含6940个合成训练样本和454个专家注释的测试样本。

**🔧 技术方法**

采用了基于大型语言模型（LLM）的框架，包含手动引导的异常合成和事件引导的异常再现两个互补管道。

**📊 数据集**

使用了来自真实企业在线诊断记录的异常样本和DBA手册的定义，构建了多种硬件配置和数据库模式下的样本。

**📈 对比分析**

通过与多个LLM驱动的数据库代理进行广泛评估，发现即使是表现最佳的代理（ReAct与GPT-6 Astra）也仅得分56.24，大多数代理得分低于45，显示出数据库诊断和异常修复的挑战。

**⚠️ 局限性**

存在的局限性包括LLM合成可能引入的幻觉问题，导致样本的有效性和可靠性受到影响，以及数据库代理在复杂异常的诊断和修复方面的能力不足。

---

## 193. Code, Commits, and Complexity in the GenAI Era: A Longitudinal Study of Team-Based Capstone Projects

**arXiv ID:** 2611.07297 | [PDF](https://arxiv.org/pdf/2611.07297v1)

**作者:** Asma Shakil `[一作]` (University of Auckland), Paul Denny `[通讯]` (University of Auckland)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2021–2025年10个学期共260个团队式capstone项目仓库进行纵向分析，比较生成式AI出现前后在代码规模、代码特征（复杂度、错误处理密度、注释密度）和开发行为（提交量、频率、提交大小）方面的变化。

**💡 创新点**

提出多维度测量框架，将代码规模、结构复杂度、错误处理密度、注释密度与提交频率等多指标结合，首次在真实团队项目中揭示“复杂度提升但注释不增”与“提交频率提升但提交大小不变”的现象。

**🔧 技术方法**

利用GitHub API抓取提交历史，使用Tokei统计LOC/注释，Lizard计算cyclomatic complexity，自定义脚本统计错误处理密度，并用Spearman相关和Theil–Sen斜率分析跨学期趋势。

**📊 数据集**

使用260个团队级capstone项目的GitHub仓库数据，涵盖2021–2025十个学期，主要语言为JavaScript/TypeScript和Python。

**📈 对比分析**

通过跨学期单变量趋势检验（Spearman ρ 与Benjamini–Hochberg校正），结果显示代码规模、复杂度、错误处理密度、提交量和提交频率显著上升，注释密度无显著变化；方法在数据上无明显错误，分析完成。

**⚠️ 局限性**

未直接测量学生对GenAI的使用，因而无法确认因果关系；可能受课程、框架、语言变化等因素影响；指标为描述性代理，未捕捉语义层面；研究局限于单一机构、单一课程，外推性受限。

---

## 194. A New Characterization of Streaming Decidability of Boolean CSPs Under Deletions

**arXiv ID:** 2611.07309 | [PDF](https://arxiv.org/pdf/2611.07309v1)

**作者:** Aditya Bandekar `[一作]`, Santhoshini Velusamy `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文研究了在严格转向模型下，决定布尔约束满足问题（CSP）的单次流复杂性，针对每个固定的非平凡约束语言Γ进行了表征。

**💡 创新点**

创新点在于首次在流CSP文献中区分了插入-only和转向复杂性，并且通过整数捕获度来表征严格转向空间复杂性。

**🔧 技术方法**

使用了捕获多项式和谱稀疏化技术，结合整数性将近似谱保证转化为满足赋值的精确保留。

**📊 数据集**

使用了多项式长度的流，具体数据集未明确提及，但涉及到布尔变量和约束的实例。

**📈 对比分析**

与现有方法比较，提出的随机算法在空间复杂性上为O_k(n^d_Γpolylog n)，并证明了Ω_k(n^d_Γ/log n)的下界，显示出算法的有效性。

**⚠️ 局限性**

限制在于算法的随机性和对流长度的多项式限制，且未能解决最小违反约束数的严格转向空间表征问题。

---

## 195. ZYFA: Speed and Accuracy of FP8 FlashAttention on AMD CDNA4

**arXiv ID:** 2611.07496 | [PDF](https://arxiv.org/pdf/2611.07496v1)

**作者:** Quentin Anthony `[一作]`, Stephen Ebert `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在AMD CDNA4 MI355X GPU上实现了高效的FP8 FlashAttention核，针对注意力算子进行软最大化、矩阵乘法、概率量化等关键步骤的优化。

**💡 创新点**

创新点在于①使用“lean P”——仅用乘加与整数转换直接近似指数，显著减少向量单元开销；②对概率块进行指数缩放并在矩阵单元上完成行求和，降低了舍入误差；③采用ping‑pong两波调度在同一SIMD上实现软最大化与矩阵运算的几乎完全重叠。

**🔧 技术方法**

主要技术包括FP8（e4m3）格式的数值近似、基于汇编的手工指令调度、块级指数缩放与矩阵级行求和、以及对GPU硬件特性的精细建模（如指令周期、时钟衰减、功率限制）。

**📊 数据集**

使用了三种大模型的真实激活（Mistral‑7B、Qwen3‑8B、DeepSeek‑V2‑Lite）以及标准正态分布的合成输入来评估吞吐量和精度。

**📈 对比分析**

与AMD AITER基准核比较，FP8核在捕获激活下实现了高达1.6×的吞吐量提升（在32K标记时可达1.3×），且在FP8误差上优于AITER（相对RMS误差降低至约30%以内），同时在BF16上保持了相近的性能。

**⚠️ 局限性**

局限性包括：①FP8精度仍受低概率下溢和指数范围限制，需要手动块缩放；②GPU在持续负载下时钟会随算术操作的比特切换而下降，进一步抑制峰值吞吐；③在头维度192及更大配置下，FP8的加速优势相对减弱，且对不同数据分布的鲁棒性尚需进一步验证。

---

## 196. KOSMOS: Self-scheduling Persistent Kernels for Communication-Compute Overlap

**arXiv ID:** 2611.07477 | [PDF](https://arxiv.org/pdf/2611.07477v1)

**作者:** Alexander Magro `[一作]` (Advanced Micro Devices Inc.), Wen Chen `[通讯]` (Advanced Micro Devices Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 KOSMOS，自行调度的持久核，融合通信与计算，实现一核覆盖所有 TP+SP 与 MoE 的前向后向计算。

**💡 创新点**

创新点在于无需主机调度或预先分配通信/计算工作量，工作组基于数据可用性动态切换角色，从而显著减少空闲窗口并提升吞吐量。

**🔧 技术方法**

技术包括：持久内核、基于队列的读写同步、单一工作组切换策略（单向/双向）、按步/按带宽/按块调度、MXFP8 双精度压缩与重量化、八波动量化移动器、以及对 AMD MI350X/MI355X GPU 的 XGMI 互连优化。

**📊 数据集**

在 Llama‑3（8B、70B、405B）和九个 MoE 模型（DeepSeek‑V3/V4、Qwen3‑30B/235B、Qwen3.5‑397B、Kimi K2.7、GLM‑5.2、Llama‑4 Scout/Maverick）上进行实验，使用 Mixture‑of‑Experts 与张量/序列并行训练。

**📈 对比分析**

相较于 ROCm Transformer Engine、DeepEP 以及独立收敛的收集器，KOSMOS 在 MI350X/MI355X 上实现了 1.10–1.28× 的 TP+SP 速度提升（BF16/ MXFP8）和 1.29–1.83× 的 MoE 训练加速，显著提高整体吞吐率。

**⚠️ 局限性**

局限性包括：仅支持单机 XGMI 内部传输；需手动对每个 GPU 进行 CU 预算；不兼容多线程 GEMM 算法（如 Stream‑K）；在多节点或不具备 hipIPC 的系统上难以推广。

---

## 197. Understanding Self-Supervised Representation Learning for Encrypted Traffic

**arXiv ID:** 2611.07452 | [PDF](https://arxiv.org/pdf/2611.07452v1)

**作者:** Zongxi Liu `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统研究了自监督学习在加密流量分类中的有效性，探讨了可学习信息来源、上下文预训练、对比学习及其边界。

**💡 创新点**

创新点包括：① 对可见协议字段的字节级互信息分析，揭示局部依赖；② 设计 ByteBERT 与 TrafficDCNet 两种字节级模型以充分利用邻接字节；③ 构建 AlignTraffic 跨视角对比与匹配框架，将两种模型视角融合提升表示质量；④ 通过多任务消融验证每项目标的贡献；⑤ 在工业数据、地理迁移、Tor 和单包分类等极端场景评估其边界。

**🔧 技术方法**

技术手段主要是：字节级掩码语言模型（ByteBERT）、动态分块自编码网络（TrafficDCNet）、跨视角对比学习、匹配损失、融合注意力、联合 MLM 损失及相关正则化。

**📊 数据集**

使用的公开数据集包括 ISCX-Direct、ISCX-VPN、CSTNET、ISCX-Tor；工业私有流量数据（165 类应用）；在这些数据上做流级与包级分类。

**📈 对比分析**

与传统上下文预训练（ET‑BERT、YaTC、NetMamba）以及浅层特征学习（Pcap‑Encoder、浅层基线）比较：在冻结和微调两种评估中，AlignTraffic 在四个公开任务上 Macro‑F1 最高，提升 5.97–30.70%；在工业数据上 Macro‑F1 由 42.38% 提升至 51.26%；在地理迁移上提高 0.66–4.74% Macro‑F1；对比学习和匹配损失的消融显示其对性能提升贡献显著。

**⚠️ 局限性**

限制包括：当应用信息被 Tor 等额外加密封装隐藏或仅观察单包时，Macro‑F1 仍低于 30%，表示当前方法对完全隐藏的应用结构仍无效；跨视角对齐虽提升表示但显著增加参数量、推理延迟和显存；对比学习的负样本采样和队列管理需要调优，且在资源受限环境下部署受限。

---

## 198. TRiST: Topology-Reinforced Swin Transformer for Medical Image Analysis

**arXiv ID:** 2611.07431 | [PDF](https://arxiv.org/pdf/2611.07431v1)

**作者:** Pengfei Gu `[一作]` (University of Texas Rio Grande Valley), Danny Z. Chen `[通讯]` (University of Notre Dame)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为Topology-Reinforced Swin Transformer (TRiST) 的模型，能够在医学图像分析中将多尺度拓扑结构嵌入Swin Transformer的层级架构中；

**💡 创新点**

创新点在于：① 设计了层级多尺度拓扑表示算法，在每个Swin阶段直接计算对应尺度的2D持久同调特征；② 引入基于持久寿命的Band-Aware H_0/H_1残差细化模块，按短/中/长寿命分块独立处理；③ 通过Persistence-Band Progressive Attention Bias 将拓扑信息以可学习的加性偏置注入窗口注意力，实现对拓扑显著token的显式关注；

**🔧 技术方法**

核心技术包括：层级多尺度持久同调（PH）计算、向量化持久图（PI/BC/PS/PL）、残差MLP细化、门控行向量分块、外积形式的注意力偏置注入，以及Swin Transformer的层级结构；

**📊 数据集**

在五个医学图像数据集上验证：ISIC 2018（皮肤病分类）、CBIS-DDSM（乳腺X光分类）、RetinaMNIST（视网膜分类）、Synapse（多器官CT分割）、ACDC（心脏MRI分割）；

**📈 对比分析**

与ResNet、SENet、SwinV2-B等基线以及PHG-Net、PHG-Net+multi-scale等拓扑增强方法比较，TRiST在所有任务上均优于Swin基线，且在分类上接近或超过PHG-Net+multi-scale，在分割上获得与3D方法相当的Dice并在HD95上表现最优；

**⚠️ 局限性**

主要局限：需要在离线预计算多尺度持久同调，导致部署时增加预处理时间；目前仅支持二维H_0/H_1特征，未覆盖本地3D拓扑，未来工作将聚焦降低预计算成本并扩展至3D PH。

---

## 199. Kinematic Hypothesis Context: Embodiment-Conditioned Learned Video Coding for Bandwidth-Limited Robot Policy Uplinks

**arXiv ID:** 2611.07411 | [PDF](https://arxiv.org/pdf/2611.07411v1)

**作者:** Yuze Lu `[一作]` `[通讯]` (Tsinghua University), Yuze Lu (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了机器人远程推理的摄像头上行链路，提出零比特运动学假设上下文并将其嵌入条件视频编码器，以显著降低比特率并提升任务成功率。

**💡 创新点**

将机器人运动学信息作为零比特侧信息，生成一组像素传输假设（平面扫描、机器人层、末端执行器平面、背景记忆），实现对运动学的直接利用，避免传统历史或全局向量条件无法捕捉的运动信息，从而将压缩收益转化为任务成功收益。

**🔧 技术方法**

使用运动学假设上下文（plane sweep homographies、机器人层重投影、末端执行器平面、背景记忆）、学习融合网络(KinFusion)、可变率的条件学习式编码器、k‑skip机制、手眼自标定、以及多种主机（DCVC‑RT、Scale‑Space‑Flow、entropy‑only）等技术。

**📊 数据集**

实验数据来自公开的 LIBERO 仿真数据集（四个子集、200 条测试任务）和真实 DROID 机器人数据（100 条实验），并在两者上进行评估。

**📈 对比分析**

通过 BD‑rate、PSNR、MS‑SSIM、LPIPS、动作偏差等传统度量以及闭环任务成功率进行比较。与容量匹配的历史条件码相比，提出方法在 PSNR 上节省约23.9% 比特；与 x264/x265/SVT‑AV1 等传统编解码器相比节省 49–56%；与 DCVC‑RT 相比在动作偏差上节省 23.2%；在闭环任务成功率上，6 kbps 下从 22.8% 提升至 43.1%，在 8.3 kbps 和 10.7 kbps 亦均显著优于基线。

**⚠️ 局限性**

主要限制包括：需要状态与图像严格同步（1 帧偏移会消除收益）；依赖准确的手眼标定，未标定摄像头退回至历史模式；对非刚性物体（液体、柔性物体、人）假设不足；与更强主机（VTM‑23、DCVC‑RT）之间仍存在一定差距；目前仅针对同步摄像头，异步场景需进一步研究。

---

## 200. Req2Rust: A Requirements-Guided Agent Harness for Project-Level C++-to-Rust Migration

**arXiv ID:** 2611.07442 | [PDF](https://arxiv.org/pdf/2611.07442v1)

**作者:** Shengbo Wang `[一作]` (Sun Yat-sen University), Yanlin Wang `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Req2Rust，一套完整的项目级 C++ 到 Rust 的迁移系统，先通过 MRS 记录保留义务，再通过 RDS 生成 Rust 设计，最后利用编译器、语义和 unsafe‑reduction 验证器进行迭代修复。

**💡 创新点**

创新点在于：① 用需求驱动的中介表示（MRS）显式分离“必须保留”的语义与“如何实现”的实现细节；② 通过 RDS 把这些需求映射到 Rust 项目结构；③ 结合验证器反馈实现的闭环迭代，显著提升迁移的可执行性与语义保真度。

**🔧 技术方法**

主要技术：LLM（DeepSeek‑V4‑Pro）生成代码与修复；Agent‑harness 对项目状态进行监控与更新；编译器验证、语义验证（对比测试与 MRS）以及 unsafe‑reduction 验证器；迭代修复与验证循环。

**📊 数据集**

使用的数据集包括 7 个通用 C++ 项目（覆盖 7–10 种 C++ 特性）和 5 个 OpenHarmony 模块（覆盖全部 10 种特性），共计约 200k 行代码。

**📈 对比分析**

与 ClaudeCode 作为基线进行对比。通用项目中，Req2Rust 的编译通过率与测试通过率分别为 100% / 91.9%，显著高于 ClaudeCode 的 100% / 69.5%；在 OpenHarmony 模块中，平均测试通过率从 45.4% 提升至 87.9%。整体性能主要体现在更高的测试覆盖与更低的 unsafe 使用。

**⚠️ 局限性**

局限性：① 评测仅覆盖公开测试用例，未能保证对所有边界和并发行为的完整验证；② 迁移质量受 LLM 生成能力与提示策略影响，模型不同或参数变更可能导致结果差异；③ 在大型项目或高度依赖平台特定接口的场景下，未覆盖所有可能的构建/运行时约束。

---

## 201. Recursive Self-Improvement for Long-Horizon Role-Playing Dialogue

**arXiv ID:** 2611.07469 | [PDF](https://arxiv.org/pdf/2611.07469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 202. Fast and Reliable Open-Vocabulary Object Search with Absence Detection from Robot Memory

**arXiv ID:** 2611.07493 | [PDF](https://arxiv.org/pdf/2611.07493v1)

**作者:** Nicholas Paul `[一作]` (Lawrence Technological University), Eric Martinson `[通讯]` (Lawrence Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种两阶段架构，先使用视觉或字幕检索快速筛选候选观察，再通过视觉验证（SAM3）或LLM进行最终选择或拒绝，实现机器人对RGB序列中物体的高召回率与可靠缺失检测。

**💡 创新点**

创新点在于将检索与验证完全解耦，使系统既能保持低延迟又能在物体缺失时进行可靠拒绝；并通过视觉验证显著提升特异性，从 9% 提升到 92.5%。

**🔧 技术方法**

采用 CLIP 视觉嵌入、Gemma‑3 生成字幕、Nomic Embed Text、Milvus 向量数据库、SAM3 开放词汇检测器、LLM 推理、SLAM 姿态估计等技术。

**📊 数据集**

在 ScanNet 室内公寓场景上进行大规模评估，并在校园走廊真实机器人环境中进行验证。

**📈 对比分析**

对八种检索‑选择组合与完整 SAM3 exhaustive baseline 进行对比，Caption/SAM3 组合在 F1s 上最高达 92.5%，特异性 94.1%，查询时延约 2 秒，显著快于耗时 3 分钟的全局检索。

**⚠️ 局限性**

限制包括：字幕生成速度慢，LLM 在负查询时特异性仍偏低；在更大规模、动态变化或更异构的环境中的泛化尚需验证；以及对移动物体的动态推理尚未涵盖。

---

## 203. ReviewMap: Mapping Evaluative Acts and Automation Boundaries in Peer-Review Comments

**arXiv ID:** 2611.07489 | [PDF](https://arxiv.org/pdf/2611.07489v1)

**作者:** Kihyun Na `[一作]` (University of Alabama at Birmingham), Injung Kim `[通讯]` (Handong Global University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出ReviewMap框架，显式区分审稿评论中的评估（ASM）和校准（CAL）两种评判行为，并利用多模型LLM共识对评论进行分类，进一步为作者、审稿人和领域主席提供可检验的审稿信息。

**💡 创新点**

创新点在于：①首次在评论层面系统区分评估与校准两种评判行为；②通过多模型共识和自一致性采样实现可可靠的自动化分类；③将这一自动化边界与人类验证结合，明确哪些层级可以自动化、哪些需要人工干预；④提出针对不同角色的可视化工具（审稿人工作流程、作者回应策略、领域主席讨论面板）。

**🔧 技术方法**

使用技术包括：多模型LLM共识（Qwen 3.5、Gemma 4、Nemotron 3‑Super），自一致性采样、两阶段投票机制；自定义规则分段器将审稿文本拆分为评论；统计学评估（Fleiss κ、Cohen κ、Fleiss κ）来度量一致性；人工标注验证与前沿LLM对照。

**📊 数据集**

数据集为2026年ICLR提交的500篇论文（共17,654条评论），以及12篇论文共470条评论的开发集，用于人工标注与对照；全部来自OpenReview公开数据。

**📈 对比分析**

比较方法为与五位人工标注者及前沿LLM（GPT‑5.5、Gemini 3.1 Pro、Claude Opus 4.6）进行对照。结果显示，Level‑1 评估/校准分类的内/间模型κ均≥0.77（大部分为substantial‑to‑almost‑perfect），但Level‑2属性（P1、P2、grounded）的κ明显下降，表明属性层需人工核查；整体上，审稿评估层可实现可靠自动化。

**⚠️ 局限性**

局限性包括：仅覆盖ICLR 2026英语审稿，未评估其他会议或语言；只对Level‑1进行人工验证，属性层缺少人类金标准；模型共识可能偏向过度预测CAL，影响频率估计；未评估系统对决策支持的实际效果；以及对异常或未覆盖的评论（FLAG）处理的可扩展性待验证。

---

## 204. Two-stage residual learning for calibrated uncertainty in solar irradiance forecasts

**arXiv ID:** 2611.07444 | [PDF](https://arxiv.org/pdf/2611.07444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 205. Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics

**arXiv ID:** 2611.07478 | [PDF](https://arxiv.org/pdf/2611.07478v1)

**作者:** Shradha Agarwal `[一作]` (Missouri University of Science and Technology), Deepak Rajbhar `[通讯]` (Missouri University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大语言模型在多步数学推理中出现的单一符号错误，定位错误的写入层并验证其机制。

**💡 创新点**

提出将错误写入视为可解释的“习惯层”，区分固定负号“stamp”与自适应放大器“amplifier”，并证明局部定位并不等同于可操纵性。

**🔧 技术方法**

使用logit lens、直接logit归因、因果干预（层ablations、方向减法、补丁、steering）以及多模型多任务的对照实验。

**📊 数据集**

采用LinAlg-Bench线性代数基准及自建的积分分部法（IBP）问题集，针对每个模型单独生成更大规模问题。

**📈 对比分析**

通过对五大开源模型（Gemma, Mistral, Phi, Llama, Qwen）在两个任务上的定位与干预结果进行比较，发现定位率高达70–100%，但干预成功率在4–93%之间，显示模型间显著差异。

**⚠️ 局限性**

局限在于仅探讨了单一错误类型和两类任务，干预评价只聚焦答案位置，未覆盖完整生成过程；未验证更广泛任务或更小模型的普适性。

---

## 206. Polynomial-Time Approximate EFX for Chores via Flow and Matching

**arXiv ID:** 2611.07468 | [PDF](https://arxiv.org/pdf/2611.07468v1)

**作者:** Genjie Qin `[一作]` (Ocean University of China), Xin Chen `[通讯]` (Ocean University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

本文设计了一套多阶段算法，在多方位非负加性代价的不可分任务分配问题中，能够在多项式时间内输出完整的 (3+ε)-EFX 近似分配，并将此框架推广到 EFkX 近似分配，实现 (2+1/k+ε)-近似。

**💡 创新点**

创新点在于：①首次给出常数因子 (3+ε) 的多项式时间 EFX 近似；②提出 Max‑Flow Capped Price (MFCP) 结构，直接从成本矩阵构造可控价格和约束；③结合排序槽舍入 (Sorted‑Slot Rounding) 与包（packet）重分配策略，保证约束不被破坏；④给出稀疏实例的 BRBI 方法，既能在 m≤2kn 时实现精确 EFkX，又能在 m≤(3k−1)n 时提供 2−1/k 近似。

**🔧 技术方法**

核心技术包括：最大流求解与切割更新以得到约束平衡的价格向量；Shmoys–Tardos 风格的排序槽匹配实现整数分配；多重包（packet）拆分与局部交换（Adaptive Packet Reallocation）实现公平性修复；以及对成本矩阵进行缩放与扰动保证可行性。

**📊 数据集**

论文中未使用公开数据集，而是以理论构造的任意非负加性成本矩阵为输入，所有实验与证明均基于理论分析与构造实例。

**📈 对比分析**

与现有最优 4‑EFX 存在性结果、2‑EFX 局部搜索等方法相比，本文提供了可计算的常数近似（3+ε 或 2+1/k+ε），并在稀疏实例上进一步压缩到 2−1/k 或精确解；实验结果表明算法在多项式时间内完成，且在构造的极端实例中逼近证明的最优上界。

**⚠️ 局限性**

局限性包括：①算法复杂度仍然相对较高，主要受 MFCP 中多次最大流迭代影响；②只适用于非负加性成本，无法直接处理子模或更一般的代价函数；③近似因子 3（EFX）与 2+1/k（EFkX）已被证明在本框架下近似上界，进一步降低仍需突破当前技术；④对输入矩阵的整数化与扰动可能在极端精度要求下引入误差。

---

## 207. BRIDGE: Bridging the Semantic Quantization Gap for Generative Recommendation

**arXiv ID:** 2611.07425 | [PDF](https://arxiv.org/pdf/2611.07425v1)

**作者:** Yatong Xiao `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对生成式推荐中RQ‑VAE量化与解码不匹配的Semantic Quantization Gap（SQG）提出整体解决方案Bridge，显著提升生成式检索性能。

**💡 创新点**

创新点在于分别闭合三大缺口：①通过代码本地化潜在推理（CGLR）将冻结的码本质心注入生成前的潜在空间；②使用特权EMA自蒸馏（PESD）将邻域语义信息传递给学生；③设计努力感知递归（EAR）按层难度分配循环计算。

**🔧 技术方法**

主要技术包括共享循环解码器、残差量化RQ‑VAE、EMA教师与JSD蒸馏、原型正则化、右对齐循环预算以及轻量级终端块。

**📊 数据集**

实验使用Amazon Beauty、Sports、Toys三大公开电商数据集，并在一周的线上广告A/B实验中验证效果。

**📈 对比分析**

与GRU4Rec、SASRec、BERT4Rec等序列模型以及TIGER、ReaRec、COBRA、GR4AD等生成式基线比较，Recall@10/NDCG@10平均提升10%–12%，线上eCPM提升0.65%。

**⚠️ 局限性**

局限性包括在更大目录规模下收益递减、循环预算手工设定、对RQVAE与原型聚类的依赖，以及未处理动态物品集和多模态SID的扩展。

---

## 208. Adversarial Robustness of Kernel Machines: A Comprehensive Analysis

**arXiv ID:** 2611.07474 | [PDF](https://arxiv.org/pdf/2611.07474v1)

**作者:** Rahul Kumar Yadav `[一作]` (IIT Bombay), Arjun Bhagoji `[通讯]` (IIT Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对核机器(KMs)、随机特征分类器(RFCs)和递归特征机器(RFMs)在图像与表格数据上的对抗鲁棒性进行了系统实验，提出了一种近似对抗训练算法；

**💡 创新点**

创新点在于①提供全面对比实验表明KMs天然比NN更鲁棒；②展示RFCs可通过增大特征数提升鲁棒；③提出可在RKHS中进行近似对抗训练的迭代方法；

**🔧 技术方法**

使用核方法（Matérn、Laplacian、Gaussian）、随机特征近似、梯度上升/自适应PGD、SPSA、AutoAttack；

**📊 数据集**

使用FashionMNIST、KMNIST、QMNIST、SVHN、CIFAR-10等图像数据以及Adult、Bank Marketing、Phishing等表格数据；

**📈 对比分析**

与FCNN、LeNet、WideResNet等标准NN进行比较，KMs在大多数任务下对抗鲁棒性优于NN，且可接近甚至超越对抗训练NN；

**⚠️ 局限性**

局限包括：仅评估ℓ_p攻击，未检验更复杂数据集与模型；未给出理论鲁棒性证明；对Laplace RFC梯度屏蔽机制分析不充分。

---

## 209. ProCraft: Instruction-Driven Music Production from Free-Form Text

**arXiv ID:** 2611.07423 | [PDF](https://arxiv.org/pdf/2611.07423v1)

**作者:** Yuan-Chiao Cheng `[一作]` (Georgia Institute of Technology), Vincent K. M. Cheung `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了ProCraft，一个0.5B参数的自回归音频‑文本码流语言模型，可根据自由文本指令一次性编辑30秒音乐片段，支持七种编辑操作（移除、提取、伴奏生成、单音轨覆盖、乐器转换、降噪及音效），并能够处理复合指令。

**💡 创新点**

创新点包括①将文本与多轨音频拼接为单一因果序列，利用持久编辑头与门控交叉注意力保证指令信息长期可用；②在中间层对音乐嵌入、音高、乐器与音效进行对齐，提升对未目标音轨的保留；③采用在线编辑对生成（On‑the‑fly Editing‑pair Synthesis）结合工具调用式语言模型自动合成大规模复合指令数据。

**🔧 技术方法**

技术核心为基于 InspireMusic（Qwen‑2 架构）的预训练模型，结合 WavTokenizer 低比特码流编码、门控交叉注意力、中间层音乐对齐（MERT、Pitch/Instrument、StemFX）、z‑loss 训练稳定性以及工具调用式语言模型生成指令与效果链。

**📊 数据集**

训练数据来自 Slakh2100 的多轨音频与 MIDI，通过在线合成生成指令与目标；评估数据包含 Slakh2100 测试集与 MoisesDB 实际录音；与 AudioSep、HTDemucs、InstructMusicGen、InstructFX2FX、MG‑Stem、MuseControlLite、PiCoGen2、ZETA、DDIM‑inversion、SteerMusic、Audio‑Omni 等基线进行对比。

**📈 对比分析**

通过 mel‑L1、MRSTFT、SI‑SDR、CLAP 等客观指标和 MUSHRA 评测对单任务与复合指令进行评估；ProCraft 在复合编辑任务上比四个自由文本编辑器高出 8–12 dB；在单任务上与专用系统竞争，但在需要真实目标的移除/提取任务中仍略逊于 HTDemucs；在创意任务上音质与源保持略低于 Audio‑Omni。

**⚠️ 局限性**

局限性包括①低比特码流导致在高保真目标时受限；②在缺乏真实目标时难以与专门分离模型匹敌；③在伴奏、翻奏等创意编辑上音质与源保持仍落后于大模型；④依赖大型语言模型生成指令与效果，可能继承其偏见。

---

## 210. Adaptive and High-Fidelity Matching Algorithm for Retrieval-Based Accent Conversion

**arXiv ID:** 2611.07420 | [PDF](https://arxiv.org/pdf/2611.07420v1)

**作者:** Xinran Wang `[一作]` (East China University of Science and Technology), Li Tian `[通讯]` (East China University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种检索式口音转换框架AA-kNN-AC，用于在固定目标口音匹配池下实现高保真语音转换。

**💡 创新点**

创新点包括：①基于音素覆盖的匹配池构建策略，确保罕见音素均衡覆盖；②基于奇异值分解的各向异性感知自适应kNN检索机制；③进阶AdaIN解码器，将风格注入与时间上采样耦合，提升频谱连续性。

**🔧 技术方法**

技术手段涵盖：WavLM特征提取、SVD奇异值分析、可变kNN检索、进阶AdaIN解码、HiFi‑GAN声码器以及多维度质量评估函数。

**📊 数据集**

使用的数据集包括LibriSpeech（训练），CMU‑ARCTIC（目标口音匹配池），L2‑ARCTIC（ASI、ABA）和VCTK（P248）进行评估。

**📈 对比分析**

与kNN‑VC、kNN‑AC、TokAN、Seed‑VC等基线对比，AA‑kNN‑AC在WER、CER、SMOS、NMOS等指标上均显著优于检索式基线，在大多数指标上与生成式模型相当或更好，且实时因子低。

**⚠️ 局限性**

局限性包括：匹配池固定仅覆盖单一目标口音，跨语言或多目标口音的适应性待验证；自适应kNN的阈值和范围仍需经验调参；模型对极端音素稀缺或极端音素变化的鲁棒性尚未充分测试。

---

## 211. Dynamic Watermarking and Action-Conditioned Latent Dynamics for Secure Vision-Based Controllers

**arXiv ID:** 2611.07403 | [PDF](https://arxiv.org/pdf/2611.07403v1)

**作者:** Aditya Parameshwaran `[一作]` (Clemson University), Yue Wang `[通讯]` (Clemson University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于视觉控制的自适应动态水印框架，用于检测机器人执行过程中的回放攻击；通过在控制输入中注入私有高斯水印，并利用控制输入条件的潜在动力学网络（LDN）预测图像潜在状态，比较预测与实际编码的残差来生成攻击证据；并通过强化学习动态调整水印协方差，以平衡检测灵敏度、控制能量和任务性能；当检测置信度超过阈值时使用Simplex式安全停机进行保护。

**💡 创新点**

创新点包括：①将潜在动力学网络与动态水印相结合，实现对高维图像输入的攻击检测；②提出基于MCUSUM的多步残差统计与贝叶斯置信更新，能够在线估计攻击概率；③使用PPO学习策略在运行时自适应选择水印协方差，并加入安全约束的Lagrangian惩罚；④通过Simplex停机与攻击置信度耦合，实现视觉误导下的安全停机。

**🔧 技术方法**

核心技术：动态水印（私有高斯噪声注入）；控制输入条件潜在动力学网络（LDN）预测图像潜在状态；多步残差的MCUSUM累积检测；贝叶斯攻击置信度更新；强化学习（PPO）自适应水印协方差选择；Lagrangian安全成本；Simplex式安全停机。

**📊 数据集**

使用仿真数据：在Isaac Sim中对UR5机械臂进行目标到达与移动人臂避让任务，采集RGB‑D图像（120×160×4）与关节信息，共计30个回放攻击实验与30个无攻击实验，数据来自机器人在不同起始状态下的轨迹。

**📈 对比分析**

与无水印、无Simplex基线对比：在水印+检测下，29/30回放攻击样本在未出现误报前成功穿越置信阈值，平均检测延迟1.23 s；无水印时0/30；在安全评估中，使用Simplex时所有实验均未进入3 cm接近区，最小间隙从0.023 m提升至0.122 m，末端速度从0.14 m/s降至0.003 m/s；无Simplex时20/30实验进入接近区。

**⚠️ 局限性**

局限性：在清洁样本中仍出现6/30误报，导致偶尔提前切换；Simplex停机虽然提高安全性，但可能因误报而不必要地中断任务；当前仅在仿真环境验证，缺乏真实机器人延迟与噪声的考量；未提供正式的安全性形式化证明。

---

## 212. Coverage Hides Churn: Auditing and Preserving Conformal Decisions as VLM Label Registries Grow

**arXiv ID:** 2611.07502 | [PDF](https://arxiv.org/pdf/2611.07502v1)

**作者:** Jai Kumar Sharma `[一作]`, Peeyush Tapadiya `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在可扩展的视觉‑语言分类器中，研究发现新增标签会导致旧标签的 conformal 决策发生漂移，并提出 PrefixCP 方法来保持旧决策不变；

**💡 创新点**

创新点在于设计了每个类使用其加入时的标签前缀作为参考，既保持了旧决策的稳定性，又保持了类条件覆盖率，同时通过一次前缀扫描实现高效计算；

**🔧 技术方法**

使用了软最大化基的 conformal 预测、类条件（Mondrian）校准、PrefixCP 评分规则以及前缀扫描算法；

**📊 数据集**

在 Food‑101、Stanford Cars、CUB、ImageNet‑R、ImageNet‑Sketch 等数据集上，并采用 CLIP 与 SigLIP 两种 ViT‑B/16 编码器进行评估；

**📈 对比分析**

与 HPS‑Refresh、HPS‑Frozen、PRE、raw cosine 等基线以及 Transductive Conf‑OT 进行对比，PrefixCP 在保持覆盖率的前提下，实现了比 HPS‑Refresh 更大的集稳定性，并在 set‑size 上位于 PRE 与 HPS‑Refresh 之间；

**⚠️ 局限性**

局限性包括需满足追加式标签增长、固定编码器与提示、每类有标注样本、旧标签校准不变，且仅在假设可交换性下提供类条件覆盖率保证，对分布漂移或标签删除不适用；

---

## 213. Degree-Equalized View-Graph Selection under Tight Pair Budgets

**arXiv ID:** 2611.07498 | [PDF](https://arxiv.org/pdf/2611.07498v1)

**作者:** Minhyeok Im `[一作]`, Jongseong Brad Choi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的 Degree‑Equalized Allocation（DEAL）算法，用来在预算 B=2N 下对图像对进行分配，避免度数集中导致的“hub‑and‑chain”图结构，从而提升 Structure‑from‑Motion 的注册率。

**💡 创新点**

创新点在于引入基于当前已选图的度数信息的动态优先级调整：对已连接图块的边给与乘子 β，对已被覆盖图像的边做指数衰减；并通过组件合并奖励来驱动图的均匀覆盖，整个规则不需要任何训练数据。

**🔧 技术方法**

使用的技术包括：基于 DINOv2 或 coverage‑prior 的候选图生成；利用 LightGlue 匹配并进行 RANSAC 验证；使用 COLMAP（增量式）或 GLOMAP（全局式）作为后端；在 DEAL 中实现的批量选择、度数跟踪、可观测更新等机制。

**📊 数据集**

数据集涵盖了 38 个标准 SfM 基准（DTU、ETH3D、Tanks & Temples、IMC2023）以及 9 个大规模 1DSfM 互联网照片集合，总共约 5,000+ 场景。

**📈 对比分析**

与传统的 top‑B、coverage top‑B、学习边缘先验以及多种检索式选择器进行对比；在 38‑场景套件中，DEAL 的注册率为 0.875，略高于 coverage top‑B（0.858）且与学习先验相当（0.930）；在 1DSfM 集合中，DEAL 从 0.776 提升至 0.929，基本匹配学习系统的表现；在大预算下效果趋于相似，优势显著集中在 B≈2N 的紧凑预算。

**⚠️ 局限性**

局限性：仅在预算接近 2N 时效果显著；对候选集合与先验质量高度敏感，若先验误差大或候选稀疏，性能可能下降；不提供任何理论保证能得到连通或可解的视图图；对增量式 SfM 具有依赖性，其他后端或更大预算下的表现尚未系统评估。

---

## 214. When Does Blockwise Additivity Break? Diagnosing and Repairing the Additive Assumption in Distillation-Based Architecture Search

**arXiv ID:** 2611.07487 | [PDF](https://arxiv.org/pdf/2611.07487v1)

**作者:** Yuefeng Xu `[一作]` (Niigata University), Jun Yu `[通讯]` (Niigata University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文比较了基于块级加法评分的整数线性规划（ILP）搜索与考虑块间交互的整数二次规划（IQP）搜索，并提出一种只需单块评估即可近似加法的logit‑sum预测器与无配对测量的Hybrid搜索方法，进一步降低搜索成本；

**💡 创新点**

创新点在于对KL加法误差进行度量与功能两部分的分解，揭示两类模型误差来源差异，并利用仅缓存单块logit的logit‑sum预测器完成近似加法评分，成功消除对配对测量的依赖；

**🔧 技术方法**

使用了块级知识蒸馏、整数线性/二次规划、logit‑sum预测、增量搜索、Bootstrap置信区间、迁移学习等技术；

**📊 数据集**

实验数据集为DeiT‑S和DeiT‑B两款Transformer模型，在ImageNet‑100（100类）和完整ImageNet‑1k上进行评估；

**📈 对比分析**

在70%–60%预算下ILP与IQP差距≤1.6个百分点，55%–40%预算差距升至7–17个百分点；Hybrid搜索在大多数设置下显著优于IQP且性能接近或优于ILP，同时评估成本仅为IQP配对阶段的13–27%；

**⚠️ 局限性**

限制包括仅评估DeiT家族模型、仅在短期微调（≤2000步）下验证、延迟测量仅在单一GPU上、Hybrid成本随预算增加而上升、未做多重比较校正、短微调评分策略无显著提升。

---

## 215. MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports

**arXiv ID:** 2611.07486 | [PDF](https://arxiv.org/pdf/2611.07486v1)

**作者:** Hongyu She `[一作]` (Zhejiang University), Lorenzo Cavallaro `[通讯]` (University College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建知识约束的LLM推理框架，在恶意软件报告中识别并映射ATT&CK技术

**💡 创新点**

提出共享行为表示、原语引导抽取与证据对齐检索相结合的双阶段流程，并通过确定性验证与无参数证据记忆实现高可信度映射

**🔧 技术方法**

利用大型语言模型（GPT‑4o）结合离线知识基（行为原语、技术配置）进行原语抽取、检索和验证，辅以向量检索与证据记忆

**📊 数据集**

使用610份公开恶意软件分析报告（对应6,095份报告‑技术关联），在MITRE ATT&CK Enterprise v19.2 下进行评测

**📈 对比分析**

与11种基线（分类、NER、SFT、直接提示、RAG、CTIKG等）对比，得到 exact‑ID 宏 F1 0.310，parent‑ID 0.386，分别比最强基线提升 17.4% 及 0.7 个百分点；证据记忆进一步提升至 0.316

**⚠️ 局限性**

对新 ATT&CK 版本的适应依赖离线概念挖掘，整体召回仍有提升空间，且框架对 LLM 提示质量与检索深度敏感

---

## 216. Signed but Not Safe: Requirement Integrity in Multi-Agent LLM Handoffs

**arXiv ID:** 2611.07483 | [PDF](https://arxiv.org/pdf/2611.07483v1)

**作者:** Xiaobo Chen `[一作]`, Chu Wang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对多代理LLM系统中的需求交接进行系统实验，评估自由文本、结构化JSON和签名需求清单三种交接方式在面对五类威胁（无攻击、摘要压缩、代理中间人、工具注入、合并器覆盖）时的安全性和实用性。

**💡 创新点**

创新点包括：① 将需求完整性问题拆解为失真（loss）与篡改（tampering）两类，并在实验中分别验证两类问题的防御效果；② 引入签名需求清单（Signed Requirement Manifest），在签名前对需求进行可验证的提取、规范化与 HMAC 签名，形成可在每个代理节点进行确定性验证的安全边界；③ 在相同轨迹下对比确定性检查器与LLM检查器，证明代码检查更可靠且误报更低；④ 通过 oracle 清单量化提取阶段的“预签名残留”，揭示签名保护的真正边界。

**🔧 技术方法**

技术手段包括：使用 Qwen3-8B、gpt-oss-20b、GPT-5.6-luna 等大型语言模型；基于 HMAC‑SHA256 的签名与验签；对需求进行句子原文提取、谓词解析、规范化和 JSON 规范化；实现 deterministic verifier 与 LLM checker；预注册实验设计、对照组、paired McNemar 检验、任务级 bootstrap CI 等统计方法。

**📊 数据集**

数据集：完全合成的任务库，共计 48 个主实验任务（三设定、两条强制需求、偏好、背景句子等），涵盖 IT 访问、财务审批、供应商入职等三类虚构企业场景；此外还有 6 条设施类验证任务和前瞻性 12 条 GPT‑5.6‑luna 的子集，用于探索更强模型的表现。

**📈 对比分析**

比较方法：在预注册的 5 个威胁脚本下，对每个任务进行 5 种交接方式的全组合评估，记录 unsafe commit、conformant commit、abstention、false alarm 等指标；使用 paired McNemar 检验和 Holm 多重校正评估假设；结果显示：在攻击下，签名清单将 unsafe commit 从 91.7% 降至 0%，在无攻击下，签名清单与结构化 JSON 的实用性基本一致；deterministic verifier 在同一轨迹上检测到的违规率高于 LLM checker，误报率低 7%。

**⚠️ 局限性**

局限性：① 任务仅为合成、有限域的约束，可能不代表真实企业需求的复杂性；② 仅测试两分支代理结构，未覆盖更深层次或多分支的工作流；③ 只评估有限数量的模型和版本，未检验模型升级对结果的影响；④ 提取阶段的准确性仍是签名保护的瓶颈（预签名残留）；⑤ 对威胁脚本的构造相对简单，真实攻击可能更隐蔽或更强；⑥ 结果主要来自 Qwen3‑8B，其他模型的统计功效有限。

---

## 217. AMORT-V1: Checkpointed Post-Quantum Attestation for Industrial Controllers

**arXiv ID:** 2611.07476 | [PDF](https://arxiv.org/pdf/2611.07476v1)

**作者:** Rishabh Das `[一作]` (Ohio University), Ahmed Oun `[通讯]` (Ohio University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对工业控制器进行长期后量子远程证明检查点式验证，显著降低每份报告的签名和持久写入成本。

**💡 创新点**

创新点在于将非对称签名与持久提交移到周期性检查点，并引入设备持有的可公开验证披露链，实现可扩展、可恢复且可公开验证的证明。

**🔧 技术方法**

技术手段包括 ML‑KEM、ML‑DSA/SLH‑DSA、Merkle 树、HMAC、TPM 以及 Python/C 实现的 liboqs 与 OpenSSL 库。

**📊 数据集**

实验使用七节点 Raspberry Pi 5 组网、五台物理 OpenPLC 设备、149 条差分测试向量、27 种攻击场景以及完整软件镜像（6 239 文件，355 MB）进行评估。

**📈 对比分析**

与单报文 ML‑DSA/SLH‑DSA 方案相比，Checkpointing 将每报文成本从数百微秒降至仅几微秒，10 000 设备时每报文约 6 μs，检测延迟可按检查点间隔配置，整体性能显著提升。

**⚠️ 局限性**

主要局限是对完整软件镜像测量的时间仍无法缩短，且需要额外的公开验证机制，适配性取决于检查点间隔与检测窗口的权衡。

---

## 218. Kinematic analysis of a double-closed-loop four-bar chain-based leg for a multimodal quadruped-pentarotor robot

**arXiv ID:** 2611.07470 | [PDF](https://arxiv.org/pdf/2611.07470v1)

**作者:** Prashant Raturi `[一作]` (Indian Institute of Technology Palakkad), Santhakumar Mohan `[通讯]` (Indian Institute of Technology Palakkad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

设计并分析了双闭合四杆链结构的三自由度单腿，实现了正逆运动学、雅可比矩阵与奇异性分析，并在单腿实验台上验证了工作空间、步态与变位。

**💡 创新点**

引入双闭合四杆链实现膝关节无直接驱动的变位机制，提供完整的关节-任务空间映射与闭环传动矩阵，首次实现陆地四足到空中五旋翼的腿-旋翼互相转换模型。

**🔧 技术方法**

使用非标准 DH 参数建模、循环约束矩阵求解、雅可比矩阵构造、点云生成与插值、MATLAB 仿真、3D 打印与 Dynamixel 伺服驱动硬件实验。

**📊 数据集**

基于 CAD 设计的几何参数和硬件极限生成的点云工作空间数据，用于逆运动学与步态插值。

**📈 对比分析**

通过仿真与实机轨迹点的误差对比验证工作空间与步态匹配度；雅可比最小奇异值在 0.03–0.1 之间，表明可行空间内不易出现奇异；硬件实验演示步态平滑运行，未出现显著失稳。

**⚠️ 局限性**

仅完成运动学分析，未进行动力学、碰撞及飞行控制；实验仅在单腿级别，未验证完整多足机器人；缺乏实时闭环控制与环境适应能力。

---

## 219. Information-Spectrum General Formulas for the Capacity Region of Classical-Quantum Multiple Access Channels

**arXiv ID:** 2611.07462 | [PDF](https://arxiv.org/pdf/2611.07462v1)

**作者:** Takuya Kubo `[一作]`, Hiroshi Nagaoka `[通讯]` (University of Electro Communications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了任意序列的经典-量子多路访问信道（cq‑MAC）的容量区域，并给出了两种通用表达式

**💡 创新点**

提出了利用谱下极限互信息率和线性组合量 K 的新公式，证明了 Kubo‑Nagaoka 外界限的紧性，构建了 Sen 的一次性内界在该框架下的应用

**🔧 技术方法**

主要技术包括信息谱方法、Sen 的一次性内界、量子假设检验、Hayashi–Nagaoka 运算不等式以及随机编码和平方根测量

**📊 数据集**

无具体数据集，论文为理论推导与证明

**📈 对比分析**

通过与 Han（经典 MAC）和 Kubo‑Nagaoka（外界限）的结果对比，证明了新公式的正确性和外界限的紧致性，但没有实验性能指标

**⚠️ 局限性**

仍需要更直接的证明来替代 Sen 的一次性内界，且对于 0<ε<1 的情形只能得到 K 的表达式，尚未完全统一两种表达式

---

## 220. The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks

**arXiv ID:** 2611.07461 | [PDF](https://arxiv.org/pdf/2611.07461v1)

**作者:** Haibo Jin `[一作]` (University of Illinois Urbana-Champaign), Haohan Wang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向真实世界任务的“agent‑native primitives”及其harness工程框架，并在heelo.dev平台上验证其效果。

**💡 创新点**

创新点在于将资源与本地LLM结合形成可复用原语，使资源携带使用能力，并通过需求驱动的图编排、证据验证与自我修订，实现“最后一公里”的自动化。

**🔧 技术方法**

采用LLM本地化接口、资源评估、证据验证与修订、基于图的任务编排以及多模型协作的工具链等技术。

**📊 数据集**

在超过二十个公开基准（如ToolBench、Terminal‑Bench、SWE‑bench、Putnam、GAIA‑Text‑103等）以及46个heelo.dev真实任务上进行评测。

**📈 对比分析**

与传统harness（Codex、Claude Code等）比较，性能显著提升：Terminal‑Bench 37.3%→55.5%，SWE‑bench Pass@1 73%→80.5%，Putnam 95%→98%，heelo.dev任务完成率从约20%提升至90%以上。

**⚠️ 局限性**

局限性在于对大型LLM和持续验证流程的依赖，资源库扩展与实时服务部署仍存在成本与安全挑战；在极大信息空间或复杂多代理场景中可能出现激活过度或通信错误。

---

## 221. Physics-informed latent conditional flow matching for generative modeling of granular flows on inclined planes

**arXiv ID:** 2611.07458 | [PDF](https://arxiv.org/pdf/2611.07458v1)

**作者:** Ziqi Han `[一作]` (Texas Tech University), Yimin Lu `[通讯]` (Texas Tech University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于物理约束的潜在条件流匹配框架PI‑LCFM，用来从极少的边界观测恢复完整的多变量颗粒流时空场。

**💡 创新点**

创新点在于将VAE降维与流匹配相结合生成全序列潜在空间；在推断时无训练地利用观测与质量/动量守恒梯度引导；实现无需重训练即可适配不同观测类型与超出训练范围的几何/粒子数量。

**🔧 技术方法**

使用变分自编码器(VAE)、Temporal Diffusion Transformer(Total DiT) 的流匹配模型、物理约束梯度引导、显式欧拉积分以及 DEM 仿真生成数据。

**📊 数据集**

使用 88 条粒子尺度 DEM 模拟数据，覆盖倾斜角 15°–60°、粒子数 20k–60k，四个切片、7 变量，每场 500 帧。

**📈 对比分析**

与 MLP、Temporal MLP、U‑Net、DDPM、NCSN 及前帧逐步 CFM 等基线对比，PI‑LCFM 在平均速度分量 RMSE 降低 45.2%、质量/动量残差最低，单次 500 帧生成仅 1.25 s，带观测约束 69 s，较 CFM 提升约 44 倍。

**⚠️ 局限性**

局限性包括仅在平面倾斜-水平几何且周期侧壁的 DEM 训练下验证，实验数据适用性未知；物理约束仅为二维质量/动量守恒，未包含粒子温度或三维守恒；对观测噪声敏感性低但对初始化依赖较大。

---

## 222. Spectral symmetry fingerprints in equivariant representations

**arXiv ID:** 2611.07410 | [PDF](https://arxiv.org/pdf/2611.07410v1)

**作者:** Jiacheng Cen `[一作]` (Renmin University of China), Wenbing Huang `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于等变图神经网络特征奇异值谱的对称性指纹（SymFP），可同时进行离散点群识别和连续对称性分析；

**💡 创新点**

创新点在于将等变网络中的秩约束视为可利用信息，利用奇异值谱实现方向不变、架构无关的对称性表征，并通过谱泄漏分数统一衡量对称性偏离；

**🔧 技术方法**

技术核心是构建等变表示的奇异值谱、秩模板匹配和谱泄漏（leakage）分数计算；实现采用参数无关等变GNN（PF）以及HEGNN、TFN、MACE、Equiformer等四种可学习等变网络；

**📊 数据集**

实验数据包括：全 32 种晶体点群的合成点云、控制对称性破缺实验、86 个芳香族二烷基化分子张力扫描、7 种无水 Ln‑DOTA 配合物与水帽 Dy‑DOTA、水扭转实验，以及四大规模生物装配（p97、HA、GluK2、E18 病毒衣壳）等真实结构；

**📈 对比分析**

与传统几何基连续对称性量（CSoM）和 Engel 量的比较表明，SymFP 在点群识别上可达 99% 以上的准确率，在能量极值检索中取得最高的 AUC，并在多组目标下比 Engel 快约两位数、比 CSoM 快四位数；

**⚠️ 局限性**

局限性包括：高谱泄漏分数可能在非对称结构上也出现；对称性识别依赖于预先计算的秩模板，需预知目标点群；在高维或缺少足够特征通道时谱可能不完全覆盖固定子空间；对手性、空间群等更复杂对称性尚未直接扩展；

---

## 223. DelegateEval: Evaluating Platforms for Personal External Service Agents

**arXiv ID:** 2611.07508 | [PDF](https://arxiv.org/pdf/2611.07508v1)

**作者:** Jichao Li `[一作]` (BRICS Institute of Future Networks), Zhixuan Liu `[通讯]` (BRICS Institute of Future Networks)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并阐述了一个名为 DelegateEval 的平台级评估框架，用以衡量个人外部服务代理的创建、发现、交付、持续性、学习、可靠性与经济价值。

**💡 创新点**

创新点在于将服务生命周期拆解为创建、单个服务期、纵向历史三类记录，并定义了八个维度与技术清单，既兼顾专业性又支持跨模型比较。

**🔧 技术方法**

核心技术包括大型语言模型、可复用技能库、MCP 协议、可执行工具环境，以及与平台交互的权限、记忆与学习机制。

**📊 数据集**

目前尚未提供完整数据集；文中仅列出10类任务类型及其验收准则，后续需补全示例输入、授权材料、参考产物和独立验收材料。

**📈 对比分析**

由于缺乏可执行的基准和评测脚本，尚未进行实证比较；框架设定了统一评分和安全阈值的规范，但真正的性能评估待后续实现。

**⚠️ 局限性**

主要局限在于：①数据集与评测环境未完成；②评估协议与评分脚本缺失；③缺乏跨模型基准与对照；④无法区分模型、个人资源与平台贡献；⑤安全事件记录不完整。

---

## 224. Learning Multi-Agent Orchestration:Higher-Order Interactions in Policy Improvement

**arXiv ID:** 2611.07495 | [PDF](https://arxiv.org/pdf/2611.07495v1)

**作者:** Kaixuan Liu `[一作]` (Emory University), Guojun Xiong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 IAPO（Interaction-Aware Policy Optimization）框架，用于大型语言模型（LLM）多智能体系统中的任务编排，通过显式考虑决策间交互来优化编排器策略。

**💡 创新点**

创新点在于：① 推导出完整的交互展开式，将性能差异拆解为单步与多步交互项；② 仅保留前两阶并给出上界，显著降低误差；③ 采用访问计数约束的可置信更新，保证每次更新均为改进；④ 在实践中通过经验贝尔努利上界实现安全的策略选择。

**🔧 技术方法**

技术包括：强化学习（MDP 轨迹采样、优势估计）、重要性采样、交互项估计、经验贝尔努利误差上界、可置信更新与验证、以及基于 Qwen2.5-7B-Instruct 的 LLM 代理。

**📊 数据集**

使用了五个长输入版本的数据集：GSM8K、MATH、DROP、MBPP、HumanEval，将每个任务嵌入 30–90 个其他任务单元中，构造统一测试集。

**📈 对比分析**

对比 15 种已发布的多智能体/工作流搜索系统，在相同 7B 执行器上评估准确率与 token 消耗。IAPO 在所有基准上均达到或超过最强 baseline，且 token 费用大幅降低；在匹配预算下，IAPO 的准确率提升 2–10 点，尤其在代码基准上显著领先。

**⚠️ 局限性**

局限性包括：① 仅对已固定的工作者策略有效，未兼顾联合训练；② 对交互项的估计依赖足够多的轨迹，计算开销仍高；③ 采用固定的任务重写和检索框架，可能限制在其他场景的迁移；④ 仍需对更大模型或更复杂任务进行进一步验证。

---

## 225. Compact Motion Embeddings for Feed-forward Unified Human Animation

**arXiv ID:** 2611.07481 | [PDF](https://arxiv.org/pdf/2611.07481v1)

**作者:** Seonwook Park `[一作]` (NVIDIA), Shalini De Mello `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种实时可行的前向Transformer模型，实现面部、上身和全身动画，并利用紧凑运动嵌入驱动；

**💡 创新点**

通过Q‑Former生成两维运动嵌入并加入对比三元组损失实现运动与外观解耦；不需要姿势估计，单帧前向推理，兼顾多区域；

**🔧 技术方法**

使用Q‑Former、DINOv3视觉编码器、Cosmos/TAESD3 tokenizer、对比三元组损失、AdaFace身份监督、Transformer解码器等技术；

**📊 数据集**

使用约1,056,941段互联网视频（5,172小时）进行训练，评估集包括 TalkingHead‑1KH、FFHQ、TikTok dance、公开YouTube 视频等；

**📈 对比分析**

与 LivePortrait、X‑UniMotion、IM‑Animation 等基线在面部、上身、全身的自我重现与跨重现上进行对比，指标涵盖 L1、SSIM、LPIPS、FID、FVD、AED、AKD、ArcFace。FUHA‑C 在上身自我重现和跨重现身份保持上均领先；FUHA‑T 在 512² 下实现 38.9 FPS；

**⚠️ 局限性**

细节柔和，仍受视觉编码器限制；对极端姿势或遮挡的泛化有限；需要大量训练数据；潜在被用于伪造。

---

## 226. Echoes of Error: Residual-Directional Local Rollout Consistency for Timeseries Forecasting

**arXiv ID:** 2611.07473 | [PDF](https://arxiv.org/pdf/2611.07473v1)

**作者:** Zheng Wang `[一作]` (Bosch Center for Artificial Intelligence), Wanfang Chen `[通讯]` (Bosch (China) Investment Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于本地回滚一致性的训练正则化（consistency loss），通过比较教师强制与自由运行预测的差异，抑制模型自身误差在后续预测中的放大。

**💡 创新点**

创新点在于：将一致性误差转化为一阶残差方向的雅可比向量乘积（JVP）惩罚，实现不改模型结构、保持一射预测的同时提升长周期鲁棒性；并给出预测子流形几何解释，阐释其对自由回放稳定性的影响。

**🔧 技术方法**

使用的技术包括：残差插入随机化、雅可比向量乘积正则化、梯度停止（stop‑gradient）处理、预测子流形理论与稳定管道分析；实验中用三层2048维隐藏的残差MLP作为基座。

**📊 数据集**

使用的数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity（ECL）、Traffic 等公开长周期时间序列数据。

**📈 对比分析**

与 PatchTST、DUET、AMD、PDF、SparseTSF、FilterNet、TSLANet、RootPurge、TimeDistill 等基线在 28 个数据集×预测时段（96/192/336/720）上比较，单纯残差MLP+CL 在 20/28 设定下获得最佳或次优结果；对所有四种主干模型均有显著提升，递归回放误差也得到降低。

**⚠️ 局限性**

局限性：依赖一阶线性近似，可能无法完全抑制非线性漂移；JVP 计算带来额外算力成本；在极大 λ 时会过度正则化，导致拟合不足；方法目前主要适用于块式一射预测，扩展到更复杂的递归预测需进一步验证。

---

## 227. Closed-Loop Physics-Based Manipulation Evaluation at a Commodity Price: What 4.4 L4 GPU hours at $3.50 Buys You

**arXiv ID:** 2611.07441 | [PDF](https://arxiv.org/pdf/2611.07441v1)

**作者:** Rohun Kshirsagar `[一作]`, Diana Rypkema `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个可在无服务器、成本低廉的GPU环境下跑200次采样评估的Drake仿真抓取任务的评估工具包，并利用该工具测试了稀疏数据下的动作分块策略。

**💡 创新点**

提供可复现的无服务器评估框架，揭示动作分块在样本稀缺情况下导致性能崩溃，并将评估成本压缩至约3.5美元。

**🔧 技术方法**

使用Docker/Modal无服务器容器、Xvfb headless渲染、gRPC政策服务、BC与DDPM模型、EGL/vendor ICD、Wilson CI统计等技术。

**📊 数据集**

采用TRI公开的lbm_eval演示数据（约100个演示）以及Barreiros等的pick_and_place_box任务。

**📈 对比分析**

通过对比单步与分块（H,K）策略的闭环成功率，单步BC多重检查点在95% Wilson CI下成功率为5.1%–11.8%，而所有分块检查点在780次评估中零成功，表明分块在稀疏数据中显著劣势。

**⚠️ 局限性**

仅覆盖单任务单规模稀疏数据，环境要求未完全公开，且分块结论不一定泛化到大规模或多任务情境。

---

## 228. IMUDINO: Self-distilled Foundation Model for Inertial Sensing

**arXiv ID:** 2611.07437 | [PDF](https://arxiv.org/pdf/2611.07437v1)

**作者:** Zhoutong Ye `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了首个可在多种惯性传感任务（HAR、健康监测、运动捕捉、机器人里程计）中均表现出色的通用惯性感知基础模型IMUDINO

**💡 创新点**

创新点在于将DINO自监督蒸馏与iBOT++细粒度重建结合，并设计一系列语义保持的运动变换（裁剪、缩放、旋转、重采样、通道遮掩），以及频率感知位置编码，使模型能在不同采样率、输入长度和传感器位置下保持鲁棒性

**🔧 技术方法**

核心技术包括：Transformer编码器（IMUFormer）、DINO全局对齐损失、iBOT++补丁级重建损失、KoLeo正则化、频率感知正弦位置编码及上述运动变换的自监督训练框架

**📊 数据集**

使用了三大公共可穿戴惯性数据集（Capture24、SHL、PAAWS）共11600小时的数据进行预训练；下游评测涵盖七个HAR数据集、四个健康/步态分析任务、四个MoCap数据集（AnDy、UNIPD、CIP、Virginia Natural Motion）以及TartanIMU机器人里程计挑战集

**📈 对比分析**

与SimCLR、RelCon、HAR‑FM等自监督基线、TimesFM、Chronos‑2等通用时间序列基础模型以及DeepConvLSTM等监督基线进行比较；在七个HAR数据集上实现了近乎或超过监督SOTA的macro‑F1；在MoCap任务上获得TIP级别的精度，且比多数监督方法更优；在机器人里程计上实现了比TartanIMU的跨平台平均AVE/ATE20更低的误差，证明跨体型迁移能力

**⚠️ 局限性**

局限性包括：在最先进的MoCap方法（如DynaIP）仍略逊；对原始陀螺仪数据的依赖较弱，可能在需要物理先验的任务上表现受限；目前模型未结合实时推理或低功耗硬件部署的评估；需要更大规模或更丰富多样的数据进一步提升鲁棒性

---

## 229. Typed Knowledge over Constructive Institutions

**arXiv ID:** 2611.07432 | [PDF](https://arxiv.org/pdf/2611.07432v1)

**作者:** Hans-Martin Will `[一作]` `[通讯]`, Hans-Martin Will

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出并形式化了 Eigenius 这一存储类型化知识的框架，定义了其核心的依赖类型理论、证明逻辑、机构理论以及记录类型语义，并给出了相应的类型检查与归约算法。

**💡 创新点**

主要创新点在于：①将证明不可相关性视为可证明的命题而非转换规则，从而保留证明的可检索性；②在定义等价上采用无类型 joinability，保持算法的可决定性；③将知识图谱、推理结果与机构理论统一到一个可决策的类型系统中，形成一个完整的知识存储与验证生态。

**🔧 技术方法**

使用的技术包括：依赖类型理论（类似 Lean4 核心）中的 Π/Σ 类型、递归与大归约；证明逻辑（justification logic）及其证据函数；机构理论（institution）框架；记录类型（record）与宽子类型；以及双向类型检查、弱头归约、归约评估等算法。

**📊 数据集**

该工作主要是理论与形式化验证，使用 Lean4 作为伴随实现，未涉及具体实验数据集；所有证明均在 Lean4 进行形式化。

**📈 对比分析**

通过在 Lean4 上实现并验证理论，证明了类型检查的可决策性、一致性与子类型（subsumption）的正确性；在理论层面已给出算法的正确性证明，但未给出性能基准，主要关注形式化与可证明性。

**⚠️ 局限性**

局限性包括：①实现细节与理论仍存在微小差异（如 λ‑抽象的域注解差异）；②缺乏对大规模实际科学知识库的实测评估；③未完成完整系统的实现与部署，尚需进一步验证在真实科学记录中的可用性与效率。

---

## 230. Truncated backpropagation through time induces a two-sided gradient turnpike property

**arXiv ID:** 2611.07415 | [PDF](https://arxiv.org/pdf/2611.07415v1)

**作者:** Julian D. Schiller `[一作]` (Leibniz University Hannover), Matthias A. Müller `[通讯]` (Leibniz University Hannover)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在递归模型训练中使用截断反向传播（TBPTT）时梯度误差的本质，并通过理论推导得出了两侧梯度 turnpike 性质，随后在 Wiener–Hammerstein 基准上验证了该理论。

**💡 创新点**

首次证明了截断上下文导致的梯度误差会在远离两个边界的地方呈指数衰减，提出了两侧梯度 turnpike 定理，并给出了对更新梯度的误差界、梯度对齐性以及在 Polyak–Łojasiewicz 条件下的最优性误差界，阐明了过去与未来上下文的权衡。

**🔧 技术方法**

采用 BPTT/TBPTT 的梯度推导、状态转移雅可比矩阵指数衰减假设、Lipschitz 连续性与平滑性假设、Polyak–Łojasiewicz 条件，以及实验中使用 vanilla RNN 进行系统辨识的实现。

**📊 数据集**

使用 Wiener–Hammerstein 基准数据集（80,000 训练样本 + 20,000 验证样本）。

**📈 对比分析**

通过 RMSE、训练时间等指标与全上下文 BPTT 进行对比。结果显示，在截断长度为 40 的 TBPTT 方案中，测试 RMSE 与 BPTT 仅相差约 1.11 倍，同时训练时间提升约 5.7 倍；较小截断长度会导致性能下降，表明两侧上下文均需适度扩大。

**⚠️ 局限性**

理论推导依赖于指数衰减的状态转移雅可比矩阵和 Polyak–Łojasiewicz 条件，在非凸训练中未能保证收敛；实验验证仅在 Wiener–Hammerstein 任务上，结果对截断长度的敏感性不一定在其他任务中保持一致；此外，实际训练中梯度噪声与采样策略也会影响性能。

---

## 231. PSGAgent: Enhancing MLLMs' Dynamic Visual Perception for Panoptic Scene Graph Generation via Agentic Reasoning

**arXiv ID:** 2611.07408 | [PDF](https://arxiv.org/pdf/2611.07408v1)

**作者:** Qingyun Liu `[一作]` (Peking University), Tong Mo `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PSGAgent，利用代理式思考–执行–观察循环，使多模态大型语言模型能够动态地对场景图进行细粒度视觉感知，并通过两种工具（Region Zoom-In与Visual Highlight）精细化信息获取；

**💡 创新点**

通过把场景图生成任务重构为多轮工具调用的推理过程，构建错误感知的多模态链式思维(M-CoT)数据集并设计层级化奖励与GSPO算法，实现工具使用的自主学习，显著缓解传统模型的显著性偏差与语义歧义瓶颈；

**🔧 技术方法**

多模态大语言模型（Qwen2.5-VL-3B），工具调用框架（Region Zoom-In、Visual Highlight），错误感知M-CoT数据集，Group Sequence Policy Optimization (GSPO) 与多级奖励；

**📊 数据集**

Panoptic Scene Graph (PSG) 数据集（48,749张图，133类物体，56类关系），并使用预训练Mask2Former生成掩模；

**📈 对比分析**

在SGDet与PredCls任务上对比现有基准与多模态方法，PSGAgent在3B参数下分别取得Recall@100 50.27%/mRecall 42.88%/Mean 46.58%（SGDet）和Recall@100 73.04%/mRecall 67.75%/Mean 70.40%（PredCls），超越所有SOTA；在零样本与开放集设置下亦表现最佳；

**⚠️ 局限性**

依赖外部掩模分割器且工具设计与奖励调优需手工，模型对工具调用次数与顺序的敏感度较高，且对极端遮挡或极小物体的识别仍有限，未来需进一步提升自适应感知与通用性。

---

## 232. FreePath: Neural Mean Free Path for Image Dehazing

**arXiv ID:** 2611.07398 | [PDF](https://arxiv.org/pdf/2611.07398v1)

**作者:** Rongrong Gao `[一作]` (Tsinghua University), Wenyu Xu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FreePath框架，将单幅图像去雾视为特征空间自由路径状态的恢复；

**💡 创新点**

创新点在于引入神经自由路径（Mean Free Path）状态，设计Mean Free Path Residual Block与FreePath State Conservation Integration，既保留物理意义，又避免显式估计大气参数；

**🔧 技术方法**

采用残差编码解码器结构，深度可分离卷积、Sigmoid门控、深度残差过滤器、VGG感知损失以及多尺度损失（PSNR/SSIM/MSSSIM）等技术；

**📊 数据集**

使用CDD-11雾化子子集进行配对训练和评估，并在RTTS（RESIDE）真实场景集上进行无参考评估；

**📈 对比分析**

与GCANet、ROP、AirNet、PDDA、IHDCP等基线在PSNR、SSIM、FSIM、VSI、LPIPS、GMSD以及CLIPIQA等指标上对比，FreePath在所有指标上均优于基线，显示出更高的像素级与感知级质量；

**⚠️ 局限性**

局限性包括对极端雾密度和不同光照条件的泛化尚未充分验证，缺乏推理速度与资源消耗评估，以及对配对训练数据的依赖。

---

## 233. VIPA: View-Invariant Projector-Based Adversarial Attack

**arXiv ID:** 2611.07443 | [PDF](https://arxiv.org/pdf/2611.07443v1)

**作者:** Jiyu Han `[一作]` (Southwest University), Bingyao Huang `[通讯]` (Southwest University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为VIPA的投影式对抗攻击方法，能够同时实现视角不变和分类器无关的投影攻击；

**💡 创新点**

创新点在于使用基于高斯喷射的物理仿真（GS‑ProCams）和多视角损失聚合策略，使得单一投影图像在不同摄像角度下仍能有效误导多种深度学习分类器；

**🔧 技术方法**

核心技术包括：Gaussian Splatting 渲染、可微分投影‑捕获仿真、视角聚合损失、多分类器攻击损失以及 L2 视差与隐蔽性约束；

**📊 数据集**

使用了五个真实世界场景（包含平面与三维物体）作为实验数据，所有数据均采用相机与投影仪同步拍摄得到；

**📈 对比分析**

与SPAA、CAPAA以及VIPA的分类器专用变体对比，VIPA在20个未见视角上实现了最高的整体成功率（OSR）且在 ΔE、L∞、PSNR、SSIM 等隐蔽性指标上均优于基线；

**⚠️ 局限性**

主要局限包括：对目标物体静止要求严格，易受轻微运动影响；计算成本高（18视角×30迭代）；在暗色或几何复杂物体上重建质量下降；以及针对性攻击成功率仍较低。

---

## 234. Memory Fit: Architectural Support for Page Placement by Learning Models of Good Behavior

**arXiv ID:** 2611.07506 | [PDF](https://arxiv.org/pdf/2611.07506v1)

**作者:** Chun Deng `[一作]` (Stanford University), Philip Levis `[通讯]` (Stanford University and Google)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种硬件抽象——Memory Fit，用于为操作系统提供精准的页级内存适配信号，并实现了软跟踪匹配（STM）算法在 BOOMv3 RISC‑V 上的硬件实现，随后与 Linux TPP 进行对比评估。

**💡 创新点**

创新点在于：①将“页的适配性”定义为可比的分数（memory fit score）并通过学习好的模型在硬件上实时计算；②STM 采用压缩上下文树的学习权重和重用距离归一化，既保持低状态开销（每页 64 bit）又能捕获读/写、重用、并行度等多维访问特征；③硬件与软件分离，OS 只需读取 64 bit 表项并处理事件环，完全不依赖昂贵的采样或页表扫描。

**🔧 技术方法**

使用的技术包括：硬件软跟踪匹配（STM）算法（基于 CTW 结构与重用距离归一化）、delta 缓存与事件环的无阻塞更新、随机化四舍五入与低通滤波减少噪声、Linux 6.6 的 page‑migration 路径以及 BOOMv3 RISC‑V 的 FPGA 原型实现。

**📊 数据集**

训练数据集为 200 条合成访问轨迹（覆盖读/写比例、重用尺度、并行度等多种模式），评估数据集包括：扩展 GUPS 微基准、Redis（读写并行 + 指针更新）以及 llama.cpp（深度学习推理工作负载）。

**📈 对比分析**

比较方法：在相同硬件/内存配置下，将 Memory Fit 与 Linux TPP 的页面排序和迁移进行对比，并衡量“cycle‑gap recovery”（相对于最优与最差放置的周期差恢复比例）。实验结果显示：在 GUPS 中 Memory Fit 的恢复率约 72 % 而 TPP 仅 19 %；Redis 在不同 DRAM 容量下恢复率分别为 37 %–90 % vs 7 %–32 %；在 50 % DRAM 容量的重放中恢复率 80 % vs 19 %；对 llama.cpp 的恢复率在 29 %–78 % 之间，TPP 甚至出现负恢复；硬件观察与通知的运行时开销仅 1.2 %。

**⚠️ 局限性**

局限性：①实验仅在单核 BOOMv3 FPGA 原型上完成，未验证多核/多插槽下的扩展性；②STM 的学习模型是为特定“更快”目标训练，若目标多样化（能耗、耐久度等）需重新训练；③仅支持 2 种内存类型，未覆盖更丰富的异构内存（如 HBM、NVM 堆叠等）；④需要硬件支持（delta 缓存、事件环、可读写的 PFN 表），在现有主流 CPU 上直接移植仍有难度。

---

## 235. Transparency in Team Contracts

**arXiv ID:** 2611.07494 | [PDF](https://arxiv.org/pdf/2611.07494v1)

**作者:** Paul Dütting `[一作]`, Jinzhao Wu `[通讯]` (Yale University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在多代理合同设计中信息披露与合同方案的联合优化，重点评估完全透明度对代理团队的影响。

**💡 创新点**

创新点在于给出对可替代（子模）成功概率函数的4/3逼近证明，并证明在互补（超模）情形下透明度损失可无界，同时提出基于剪枝的多代理合同近似算法，提升逼近比至2.311。

**🔧 技术方法**

核心技术包括“全或无”引理、潜在函数（Potential Lemma）与确定性剪枝策略，以及对子模与超模函数的理论分析。

**📊 数据集**

本文不使用外部数据集，而是构造通用的理论实例（如可加奖励函数、两代理两状态情形）来证明极限和逼近比。

**📈 对比分析**

比较方法主要是理论上对比最优联合信息与合同方案与完全透明方案的预期效用，得到3/4或4/3的比例；对于算法则给出多项式时间实现，并证明在已知状态下可达到1/γ_⋆≈2.31的近似。

**⚠️ 局限性**

局限性在于对互补情形仅证明了无界下界，未给出可行的近似信息结构；并且结果仅适用于二进制行动和公共信号，未探讨多行动或私有信号的情况。

---

## 236. SAVeR$^2$: Reasoning-based Safety Alignment for Large Reasoning Models via Verifiable Rewards

**arXiv ID:** 2611.07482 | [PDF](https://arxiv.org/pdf/2611.07482v1)

**作者:** Yichen Sun `[一作]` (Zhejiang University), Zhixuan Chu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SAVER²框架，对大推理模型实现安全对齐，充分利用模型的链式推理能力；

**💡 创新点**

创新点包括四个细粒度可验证奖励、难度感知的数据选择策略、梯度投影消除安全税机制，以及混合数据训练保证安全与通用推理兼顾；

**🔧 技术方法**

采用了SFT预热、GRPO强化学习、可验证奖励评估、梯度投影算法、难度加权采样和混合数据训练等技术；

**📊 数据集**

使用了安全池（JailBreakHub、Alert、WildGuardMix等）与推理池（Polaris、Open‑R1、WebInstruct等）进行训练，评估数据包括WildJailbreak、StrongReject、XsTest、BSA、MMLU、AIME24、HumanEval等；

**📈 对比分析**

与SFT、Alpha‑Align、TARS、PPO、GRPO等方法对比，SAVER²在安全指标上取得99%+分数，过度拒绝率显著降低，同时保持或提升推理准确率，在H‑mean上超过所有基线；

**⚠️ 局限性**

局限性包括对大规模算力的高度依赖、对极端对抗性输入的鲁棒性仍待进一步验证，以及对外部评判器的依赖可能导致评估偏差。

---

## 237. FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance

**arXiv ID:** 2611.07464 | [PDF](https://arxiv.org/pdf/2611.07464v1)

**作者:** Yupeng Cao `[一作]` (Fin AI), Sophia Ananiadou `[通讯]` (University of Manchester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了FinCosmos框架，提供一种开放世界的金融智能体评估范式，模拟用户隐藏策略、要求代理从Web获取数据并自行构建回测，最终通过程序验证而非参考答案评判交付结果。

**💡 创新点**

创新点在于同时保持用户意图、数据来源与时点的开放性，利用语法生成任务、程序化的无漏检评估以及可反复反馈的学习循环，实现对金融代理“静默失败”的精确测量。

**🔧 技术方法**

主要技术包括LLM驱动的代理与用户模拟器、基于规则的策略生成器、程序化检查器（数据来源、时点、可重现回测）、LLM判定器以及可持续学习的记忆机制。

**📊 数据集**

使用的数据来自开放式Web（例如Financial Modeling Prep API）以及语法生成的隐藏策略样本；评估过程中不使用任何预先标注的真实金融数据集。

**📈 对比分析**

与传统闭合式基准对比，FinCosmos在四个通用代理和一个无工具模型的150次评估中，接受率为0%，展示了现有模型在真实金融情境下的显著不足；通过记忆+反馈循环，代理最终在所有十个任务中获得通过，表明框架支持学习与改进。

**⚠️ 局限性**

限制包括：任务仅覆盖美国股票的长仓策略，语法生成的策略空间有限；评估器与模拟器均来自同一LLM家族，可能存在一致性偏差；程序检查无法证明所有数值真实可靠；以及冻结任务导致的潜在过拟合风险。

---

## 238. FREE-LIVO: Physically Decoupled Yet Tightly Coupled LiDAR-Inertial-Visual Odometry

**arXiv ID:** 2611.07463 | [PDF](https://arxiv.org/pdf/2611.07463v1)

**作者:** Reon Tabata `[一作]` (Toyohashi University of Technology), Jun Miura `[通讯]` (Toyohashi University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种不需要固定LiDAR-相机刚体或时间同步的紧耦合LiDAR-惯性-视觉里程计（FREE-LIVO）。

**💡 创新点**

创新点在于利用生成式模型上采样LiDAR强度图并用预训练特征匹配获得跨模态像素-点对应，构建LiDAR-相机重投影因子，实现在物理解耦、非同步传感器下的轨迹估计。

**🔧 技术方法**

采用Conditional Rectified Flow生成密集LiDAR强度图、SuperPoint+LightGlue特征匹配、iSAM2滑动窗口因子图、GLIM、VINS-Mono等现有LIO/VIO模块以及IMU预积分。

**📊 数据集**

主要使用M3DGR数据集（包含室内外、动态序列）以及FAST-LIVO2和内部数据进行实验。

**📈 对比分析**

与VINS-Mono、GLIM、LVI-SAM、R3LIVE、FAST-LIVO2等基线相比，FREE-LIVO平均ATE 0.45 m，保持在不同步误差0.5 s下仍能保持稳定，并在分离传感器实验中成功估计双机器人轨迹。

**⚠️ 局限性**

局限性包括对稀疏匹配的依赖、在极端遮挡或低纹理环境下可能失效，以及生成式上采样模型在不同场景下的泛化性待进一步验证。

---

## 239. Latency-Aware Adaptive Best-of-$N$

**arXiv ID:** 2611.07453 | [PDF](https://arxiv.org/pdf/2611.07453v1)

**作者:** Alon Jacobson `[一作]` (Cornell University), Peter I. Frazier `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种两阶段自适应Best‑of‑N抽样方法，该方法利用第一批生成的奖励和对应延迟来决定是否抽取第二批以及其规模，从而在保证质量的同时降低生成和延迟成本。

**💡 创新点**

创新点包括：① 引入延迟感知的决策阈值，使阈值随第一批延迟动态变化；② 通过奖励的CDF变换实现尺度无关的阈值；③ 利用离线回归预测首批最大奖励的CDF分布，闭式计算期望收益提升；④ 仅使用最多两批生成，兼顾在线实时性。

**🔧 技术方法**

使用的技术主要有：奖励CDF变换、Beta分布预测、梯度提升树回归、Monte‑Carlo仿真评估、两阶段基于阈值的决策和生成等价成本的计算。

**📊 数据集**

实验数据来自24种配置，包括四个指令调优模型（Llama‑3.1‑8B‑Instruct、Gemma‑2‑9B‑it、Mistral‑7B‑Instruct‑v0.3、Qwen2.5‑7B‑Instruct）、三个奖励模型（FsfairX‑LLaMA3‑RM‑v0.1、RM‑Mistral‑7B、ArmoRM‑Llama3‑8B‑v0.1）以及两个提示集（AlpacaFarm、HH‑RLHF），每个提示预采样960个奖励。

**📈 对比分析**

与无延迟感知的固定阈值Best‑of‑N、固定批次版本以及纯Best‑of‑N进行对比，采用生成等价成本（generations‑equivalent cost）评估。实验结果显示，在不同延迟模型、β、γ 下，所提方法在相同奖励水平下可比Best‑of‑N降低约10%–25%的成本，甚至接近理论上限的48%提升，尤其在高延迟成本、延迟波动大或延迟与批量高度相关时优势更显著。

**⚠️ 局限性**

局限性包括：仅基于仿真验证，未在真实在线服务环境中测试延迟预测准确性；只考虑最多两批生成，可能不足以应对极难提示；依赖离线回归模型，跨任务迁移可能需要更多样本；未考虑多请求并发导致的相互影响。

---

## 240. MessyNav: Zero-Shot Navigation via Manipulation in Messy Environments

**arXiv ID:** 2611.07417 | [PDF](https://arxiv.org/pdf/2611.07417v1)

**作者:** Kenneth Llontop `[一作]` (Stanford University), Jeannette Bohg `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种零样本开放世界交互导航框架，能在未知物体和环境中通过视觉语言模型推理交互可行性并生成导航计划。

**💡 创新点**

核心创新在于显式将交互可行性推理（Affordance Reasoner）与任务规划分离，利用 VLM 进行实例级、上下文感知的可交互性判断，从而大幅提升零样本泛化能力。

**🔧 技术方法**

技术实现包括基于 VLM 的交互可行性推理、VLM 任务规划、开放语言动作库（导航、抓取、侧滑）以及 VLM 关键点定位与深度去投影实现三维定位。

**📊 数据集**

使用自建的 100 场景实例级可交互性标注数据集（217 个对象）进行评估，并在 55 场景的离线规划评估中测试性能。

**📈 对比分析**

与几种基线（几何阈值、闭集类别、端到端 VLM 规划）比较，实验显示在真实机器人上成功率提升 27 pp（从 40% 提升至 67%），在离线规划上成功率提升 7 pp，且在室内光照良好、室外白天场景表现最佳。

**⚠️ 局限性**

局限性包括使用开环抓取/侧滑原语、对 VLM 的依赖导致在弱光和复杂几何下性能下降，以及缺乏长期记忆与闭环反馈，限制了更深层次的规划与恢复能力。

---

## 241. Beyond Independent Predictions: Learning the Browsing Journey for Sequence-Level Re-ranking

**arXiv ID:** 2611.07591 | [PDF](https://arxiv.org/pdf/2611.07591v1)

**作者:** Jiayi Zhang `[一作]` (Taobao & Tmall Group of Alibaba), Jian Xu `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了联合行为建模（JBM）视角，并实现了可插拔的印象-点击评估框架ICE，用以提升工业级重排序评估器的序列级效用预测；

**💡 创新点**

创新点在于将用户浏览旅程拆分为显示→印象→点击链，显式建模印象概率和点击的交互依赖，并通过印象掩蔽和序列级点击分布匹配实现对竞争与因果关系的捕捉；

**🔧 技术方法**

采用多任务学习的印象分支与点击分支，利用深度分类器预测浏览深度，Transformer/注意力偏置或特征门控实现印象掩蔽，温度软化的KL散度进行点击分布匹配；

**📊 数据集**

使用淘宝App大规模电商重排序数据集（约48.38M用户、141.79M序列、2.84B条展示、1.65B条印象、35.45M次点击），并在工业线上14天A/B测试验证；

**📈 对比分析**

与DNN、DCN、PRM、BiGRNN、Transformer、NLGR-E等多种评估器基线对比，ICE在印象AUC、点击AUC、序列级SeqAUC等指标均显著提升，线上流量部署后点击+0.70%、收入+3.67%；

**⚠️ 局限性**

局限在于仅考虑印象与点击两种行为，未覆盖停留时间、滑动等更丰富信号，且模型对极长序列的可扩展性与实时性仍需进一步评估。

---

## 242. MetricTrail: Revisiting the Power of Metric Depth and Local Optimization for Long-Sequence Streaming 3D Reconstruction

**arXiv ID:** 2611.07555 | [PDF](https://arxiv.org/pdf/2611.07555v1)

**作者:** Jingnan Gao `[一作]` (NVIDIA), Shengyu Huang `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于单图像度量深度先验和冻结的DINOv3视觉表示的长视频实时重建系统。

**💡 创新点**

创新点在于利用单帧深度预测初始化和持续约束DPVO的稀疏局部BA，并通过学习的适配器将冻结的DINOv3特征注入匹配网络，提升匹配精度与轨迹一致性，而无需扩大优化窗口或重新训练对应网络。

**🔧 技术方法**

采用DPVO的稀疏局部BA框架、MoGe-2单图像度量深度模型、冻结的DINOv3特征编码器与适配器、以及可选的环路闭合与立体输入。

**📊 数据集**

在KITTI Odometry、Oxford Spires、VBR、7Scenes、TUM-Dynamic等公开数据集上进行评估。

**📈 对比分析**

与LingBot-Map、HorizonStream、InfiniteVGGT、TTT3R、LoGeR等多种基线对比，平均全序列绝对轨迹误差（ATE）最低，密集重建Chamfer距离也名列前茅，实时帧率约12.7 FPS。

**⚠️ 局限性**

局限性包括对动态场景的重建效果不如静态场景，依赖深度先验模型的准确性和训练分布，单图像深度预测可能在新域下失效，以及在需要长时间推理时产生较高的启动延迟。

---

## 243. SPARE: Fast Linear Attention Decoding with Compact Recurrent States

**arXiv ID:** 2611.07552 | [PDF](https://arxiv.org/pdf/2611.07552v1)

**作者:** Ke Wan `[一作]` (University of Virginia), Chen Chen `[通讯]` (University of Virginia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SPARE 框架，利用训练无关的低秩投影与写补偿技术，对线性注意力中的递归状态进行压缩，从而在解码时直接在压缩空间完成读写，显著减少 GPU 记忆传输。

**💡 创新点**

创新点在于：① 采用固定低维投影子空间（PCA）保持关键维度；② 对每个注意力头采用异质秩分配，充分利用不同头的低秩冗余；③ 引入轻量级写补偿机制，补偿投影导致的写强度衰减；④ 所有这些改动均不需要额外训练，只需离线校准即可。

**🔧 技术方法**

核心技术包括：低秩投影（PCA）、异质秩分配策略、写补偿因子、GPU 打包实现、CUDA Graph 加速、Gated DeltaNet（线性注意力）架构。

**📊 数据集**

主要使用的数据集有：WikiText‑103（下一个词预测）、512 条四选一检索提示（评估检索准确率）、SQuAD Completion、TriviaQA、DROP（三种下游检索任务）。

**📈 对比分析**

在与 Dense（未压缩模型）和 DRRQR（结构化剪枝）对比下，SPARE 在 6 MiB 与 12 MiB 递归状态压缩预算下取得：语言建模 NLL 更低，检索准确率从 33% 提升至 60%；在批量 512 的解码实验中实现 2.55× 全模型速度提升、7.49× DRAM 访问量下降、3.31× 峰值 GPU 内存下降。

**⚠️ 局限性**

局限性包括：压缩对不同任务的质量影响不一，部分下游任务仍会出现显著性能衰减；写补偿虽能降低写失真，但在极端压缩下仍可能不足；实现依赖离线校准，需额外耗时；目前仅在 Gated DeltaNet 上验证，其他线性注意力模型的适用性待进一步研究。

---

## 244. Unified Overview of Safety Certificate Functions: Techniques from Hamilton--Jacobi Reachability and Control Barrier Functions

**arXiv ID:** 2611.07537 | [PDF](https://arxiv.org/pdf/2611.07537v1)

**作者:** Hao Wang `[一作]` (University of Southern California), Somil Bansal `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对安全证书方法进行了系统综述，聚焦于 Hamilton–Jacobi (HJ) 到达性和控制障碍函数 (CBF) 两大类，统一了它们的理论框架、合成技术、以及在两种车辆跑道基准中的实验评估，并给出了实践选择指南。

**💡 创新点**

创新点在于：①从安全证书的视角统一两大主流方法，明确它们的相互关系与差异；②系统性比较经典与学习型合成方案，并用两种实际车辆模型做基准；③提出针对不同系统属性（模型可得性、状态维度）选择合成工具的准则；④提供安全过滤器实现与性能对比的完整实验流程。

**🔧 技术方法**

使用的技术包括：数值 HJ 到达性（级联网格求解、PINN DeepReach、MPC 辅助学习、RL 基于安全 Bellman 方程）；CBF 设计方法（高阶 CBF、反向退步、备份 CBF、基于简化模型的 CBF、QP 安全过滤器）；安全过滤器框架；学习式安全证书（Neural CBF、PNCBF、RL‑Reachability）；以及后验安全性验证（概率校准、确定性 Lipschitz 验证、可达集估计）。

**📊 数据集**

数据集主要来自两种车辆模型：①5 维简化车轮动力学在 6×4 米赛道内；②更高维的 F1Tenth 单轨车模型，含姿态、速度、滑移等状态。训练数据采用随机采样、MPC 轨迹、仿真回放及对轨道的 Frenet 观测，未使用公开标准数据集。

**📈 对比分析**

比较方法：在两基准下分别评估安全证书的真安全率/误安全率以及安全过滤器的安全率与通过门数；结果显示：经典 HJ 数值求解在低维可获得最大安全集；CBF 的 HOCBF、回退式与备份 CBF 在高维/输入限制下更易实现但保守或不稳定；学习型方法（DeepReach、Neural CBF、RL‑Reachability）在两案例中逼近最大安全集，且安全过滤器实现的安全率均高于 90%，通过门数与基准相当甚至更好。

**⚠️ 局限性**

局限性包括：HJ 数值方法受维数指数增长限制；CBF 设计对输入约束、相对阶与参数敏感；备份 CBF 需要高质量备份策略且优化成本高；学习型方法缺乏严格保证，依赖超参数、训练样本分布和校准；对黑盒系统的泛化仍有限，验证成本高。

---

## 245. Semantics-to-Contact: A Stagewise Framework for Robust Contact-Rich Manipulation

**arXiv ID:** 2611.07532 | [PDF](https://arxiv.org/pdf/2611.07532v1)

**作者:** Guanren Qiao `[一作]` (Chinese University of Hong Kong), Guiliang Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了Semantics-to-Contact（S2C）框架，采用阶段化的方式先通过视觉‑语言模型（VLM）进行语义定位，再用人类介入的残差强化学习完成细粒度接触控制，从而实现对视觉变化和位姿偏移具有鲁棒性的接触丰富操作。

**💡 创新点**

创新点包括：1）将接触丰富操作拆分为语义聚焦与接触细化两阶段，显著降低探索空间；2）使用VLM（MolmoPoint）实现任务指令驱动的几何定位；3）提出对象中心视觉增广策略，在真实世界中实现背景随机化；4）结合人类介入的残差RL实现高效样本利用。

**🔧 技术方法**

技术手段包括：VLM-guided 语义定位、对象中心视觉增广、基于行为克隆的 ACT 基础策略、残差强化学习（SAC）与人类介入、端到端的机器人控制与力反馈闭环。

**📊 数据集**

数据集：作者使用真实工业环境中的六种接触丰富任务（充电器插拔、USB插入、试管插拔等）收集的实时视觉与力反馈数据，未使用公开数据集。

**📈 对比分析**

与ACT、SERL、Octo‑small、HIL‑SERL、HIL‑ConRFT等基线比较，S2C在保持相同训练时间的前提下，平均成功率提升至94.17%（比最接近基线提升≈3%），且在多种视觉与位置扰动下仍能保持78–84% 的成功率。

**⚠️ 局限性**

局限性：对任务关键部件的外观大幅变化（如黑色充电器或陌生物体出现在插槽内）表现不佳；在极端光照或背景变化下仍有性能下降；以及依赖VLM的定位精度，若VLM误判会导致后续步骤失败。

---

## 246. Understanding Data Influence in Transportation Networks: A Physics-Constrained Influence-Function Framework

**arXiv ID:** 2611.07526 | [PDF](https://arxiv.org/pdf/2611.07526v1)

**作者:** Hamidreza Attari `[一作]` (University of California, Davis), Johannes Royset `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在交通网络中加入硬物理约束的影响函数框架，量化训练样本对物理可行预测的影响，并通过实验评估其对模型性能与感知点重要性的作用。

**💡 创新点**

创新点在于首次将影响函数与硬物理约束相结合，提出了在物理可行空间内的影响度量，并设计了PIP与SSS两种指标，能够区分约束保留与被消除的梯度分量。

**🔧 技术方法**

采用HardNet-Aff硬约束神经网络、约束投影层、伪逆投影、LiSSA迭代求逆、梯度正交分解、PIP与SSS指标计算等技术。

**📊 数据集**

使用2025年1-5月Sacramento PeMS District 3的318条高速公路链路小时流量、速度、占用率数据集。

**📈 对比分析**

与传统无约束模型及无约束影响函数比较；硬约束模型在约束一致性上几乎零误差，MAE略降，影响度量显示关键感知点；感知点移除实验表明排名前10的SSS站点对性能影响最大。

**⚠️ 局限性**

主要限制是计算成本高，受模型参数和约束数量限制；未考虑时间变化的约束；对更大规模网络的推广仍需进一步研究。

---

## 247. LMGN: A Unified Policy Learning Model for Last-Mile Grasp-Navigation

**arXiv ID:** 2611.07515 | [PDF](https://arxiv.org/pdf/2611.07515v1)

**作者:** Xuenmin Hu `[一作]` (Hubei University), Long Chen `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种统一的多阶段策略框架 LM GN，解决四足机器人在到达目标附近后如何同时决定底盘摆放、接近方向和抓取动作，以实现高效的“最后一公里”抓取导航。

**💡 创新点**

创新点：1) 引入 Explicit Distance and Directional Reachability (ED‑DR) 表示，将 6D 抓取姿态转换为可直接用于底盘规划的距离、方向和高度特征；2) 设计层次化混合专家 (HMoE) 模型，使得不同抓取阶段（底盘调整、预抓取对齐、抓取与抬升）能够使用专属专家路由；3) 提出联合优化策略，包括 RL 任务奖励、专家路由与 ED‑DR 的对齐损失以及动作级一致性损失，确保底盘、手臂和抓手动作协同。

**🔧 技术方法**

使用的技术包括：点云感知网络 (PointNet) 与 GSNet 生成抓取候选；多层感知机 (MLP) 作为特征融合骨干；Proximal Policy Optimization (PPO) 进行策略学习；多正样本对比学习 (MPCL) 进行专家路由对齐；以及自定义奖励与一致性损失实现多目标优化。

**📊 数据集**

主要实验数据集为 Habitat 与 Isaac Gym 两大仿真平台，使用 Unitree B1 四足机器人搭载 Z1 机械臂，测试对象涵盖多种形状（方盒、长盒、钻头、杯子、碗、瓶子）以及三类未见对象（透明盒、塑料香蕉、壶）。

**📈 对比分析**

与 SOTA 方法（VBC、HB、ReachMM、GAMMA）对比，LM GN 在 GazeSR（目标定位精度）和 GraspSR（抓取成功率）上分别提升 4.9%/22% 与 7.6%/10.3%，在 Habitat 上 GazeSR 73.9%/GraspSR 74.8%；在 Isaac Gym 上平均 GazeSR 84.47%/GraspSR 53.86%，并在未见对象上保持 77.49%/48.52%。

**⚠️ 局限性**

局限性：1) 仍依赖高质量的抓取候选生成，若候选缺失会导致性能骤降；2) 对多重可行抓取方向的支持不够灵活，导致在对称物体上 GazeSR 与实际抓取率偏离；3) 需要在仿真中精细调参，实机迁移仍待验证。

---

## 248. Concertina: Embodying Manipulation and Cognition in One Diffusion Vision-Language Model

**arXiv ID:** 2611.07513 | [PDF](https://arxiv.org/pdf/2611.07513v1)

**作者:** Chengcheng Wang `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种 Embodied Vision‑Language Model（EVA），将连续动作直接嵌入共享 diffusion VLM 的隐藏空间，实现认知与操控的统一。

**💡 创新点**

创新点在于：① 用位置式动作编码器/解码器将每个时间步的完整连续动作映射为一个连续隐向量；② 在相同的 Transformer 共享参数中完成动作生成、交互与监督，消除外部动作专家和离散词典；③ 通过 blockwise 损失、packed teacher‑forcing 与物理坐标回归实现端到端训练。

**🔧 技术方法**

采用的技术包括：Diffusion Vision‑Language 模型（BARD‑VL）、block‑wise 生成、位置编码动作投影、离散化损失、风险引导的推理与多阶段闭环控制。

**📊 数据集**

在 LIBERO、LIBERO‑Plus（10,030 任务）以及 CALVIN ABC‑D（多环境链任务）三大机器人视觉‑语言‑动作基准上进行训练与评估。

**📈 对比分析**

相较于 RT‑2、OpenVLA、π_0、π_0.5、X‑VLA 等代表方法，EVA 在 LIBERO 平均成功率 98.15%、LIBERO‑Plus 85.9% 以及 CALVIN 平均链长 4.142（SR@1 = 98.6%，SR@2 = 93.8%）均实现了显著提升。

**⚠️ 局限性**

局限性包括：仍依赖单一硬件平台，面对极端动态或稀缺视觉信息的鲁棒性有待提升；对多模态长期规划与更高维动作空间的适配仍需要进一步研究。

---

## 249. What Gets Hacked in a Collaborative Reward for LLM Recommenders: The Part That Does Not Depend on the User

**arXiv ID:** 2611.07588 | [PDF](https://arxiv.org/pdf/2611.07588v1)

**作者:** Zhuangsheng Li `[一作]` (University of California, San Diego), Julian McAuley `[通讯]` (University of California, San Diego)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在LLM推荐器的强化学习阶段使用协同过滤（CF）评分作为奖励时导致曝光集中（exposure collapse）的问题，并通过将CF奖励拆分为“物品均值”与“残差”两部分，识别出物品均值是导致曝光集中主因，随后提出两种基于每个物品的奖励校正方法（居中和标准化），在不重新训练CF模型的前提下显著提升了推荐多样性和覆盖率。

**💡 创新点**

创新点在于：①首次将CF奖励拆解为物品均值与残差并通过受控实验验证均值成分对曝光集中影响；②提出仅依赖物品统计的奖励校正（居中/标准化），实现了曝光多样性的显著提升，同时保持或略微提升推荐精度；③系统评估了多种CF架构、LLM骨干及标识符方案，证明该方法的通用性。

**🔧 技术方法**

主要技术包括：冻结的CF模型（如SASRec、GRU4Rec等）提供奖励；基于GRPO的critic‑free策略梯度强化学习；对CF奖励进行每物品均值估计与标准差估计后做居中与标准化；使用Semantic ID与Atomic ID两种标识符方案；对暴露度量（distinct top‑1、coverage@10、Gini系数）和推荐准确度（HR@10、NDCG@10）的评估。

**📊 数据集**

使用公开的亚马逊评论数据集：Amazon Beauty（7,819件商品）和Amazon Sports & Outdoors（10,052件商品），并分别采用Semantic ID和Atomic ID两种标识符，构成四个实验设置。

**📈 对比分析**

与基线（SFT起始、原始CF奖励、规则奖励、随机奖励）以及HiLaR、流行度惩罚等方法对比。实验显示，居中/标准化方法在所有四个设置下将distinct top‑1数量提升至原CF奖励的1.5–9.8倍，覆盖率提高、Gini下降，且HR@10保持与CF奖励相当甚至略有提升，证明校正方法有效且不损失准确度。

**⚠️ 局限性**

局限性包括：仅在离线评估下验证，未研究长期学习或真实用户交互中的效果；对公平性只关注曝光均匀性，未全面考量用户或商家的公平；方法依赖于对物品均值和标准差的预估，若数据分布变化可能需要重新估计；未评估在更大规模或多语言场景下的性能。

---

## 250. SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking

**arXiv ID:** 2611.07558 | [PDF](https://arxiv.org/pdf/2611.07558v1)

**作者:** Moonseok Choi `[一作]` (Korea Advanced Institute of Science and Technology), Juho Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SKIM框架，在结构化生成下对LLM代理进行剪枝与蒸馏，依据token的上下文/决策/格式三种角色实现定向压缩。

**💡 创新点**

创新点在于：①把输出token按功能划分为context、decision、format三类，②用此划分改进剪枝saliency并设计了on‑policy、约束分布的前向KL蒸馏及隐藏状态匹配，③整体在结构化生成环境下实现了性能最大化。

**🔧 技术方法**

采用层级重构剪枝（category‑aware saliency）、硬约束解码、on‑policy前向KL蒸馏、隐藏状态匹配、token角色标注（GPT‑4/规则）等技术。

**📊 数据集**

使用了Qwen3.5系列模型，xLAM‑function‑calling‑60k、BFCL‑v3、API‑Bank L1、HotpotQA、Spider、GSM8K、Hammer‑2.1‑3B等数据集。

**📈 对比分析**

与Wanda‑sp、FLAP、SlimGPT等基线在10%、25%、50%稀疏率下对比，SKIM在50%稀疏率下保留70–85%稠密模型精度，并在多步代理任务中始终保持最高结构化解码准确率。

**⚠️ 局限性**

局限性：仅验证至27B规模；蒸馏仅使用单步函数调用轨迹，未覆盖多步代理轨迹；仅采用层级重构剪枝，未探索全局剪枝或量化等更高效方法。

---

## 251. Learning and Steering in Incomplete-Information Games

**arXiv ID:** 2611.07589 | [PDF](https://arxiv.org/pdf/2611.07589v1)

**作者:** Asrın Efe Yorulmaz `[一作]` (University of Illinois Urbana--Champaign), Tamer Başar `[通讯]` (University of Illinois Urbana--Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在线学习与干预框架，允许中介在仅能观察终端轨迹且不了解参与者效用的前提下，通过支付与推荐学习未知游戏并引导无后悔学习者实现最优 EFCE。

**💡 创新点**

创新点在于：①在仅有轨迹反馈的条件下给出 O(ε⁻²) 的样本复杂度学习算法；②将学习、计算（EFCE 优化）与干预三者在同一算法中耦合，形成 Learn‑Compute‑Steer 方案；③实现了 O(T⁻¹/⁴) 的直接性与最优性误差收敛，并保证平均支付为 O(T⁻¹/⁸)。

**🔧 技术方法**

技术上结合了 halving‑type 学习、极大似然式的 EFCE 激励约束、拉格朗日变换、在线无后悔学习（MWU、CFR+）以及对支付与推荐的分层上下文管理。

**📊 数据集**

使用了两个典型对照实验：二维 Stag‑Hunt 正规博弈与两人 Kuhn 扑克 EFG，均为基于模拟生成的稀疏游戏实例。

**📈 对比分析**

通过理论证明与实验可视化，发现学习误差与直接性、最优性误差随时间以 T⁻¹/⁴ 速率下降，平均支付以 T⁻¹/⁸ 速率消失；实验曲线与理论上界高度吻合，说明算法在有限样本下表现良好。

**⚠️ 局限性**

主要限制包括：①假设每个中介提供的上下文下玩家均满足无后悔；②需要完整的游戏结构与机会概率；③EFCE 推荐分布规模指数级，计算与存储上存在潜在瓶颈；④仅处理完全可观测的 EFG，未覆盖 POMDP 或马尔可夫游戏情形。

---

## 252. Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment

**arXiv ID:** 2611.07586 | [PDF](https://arxiv.org/pdf/2611.07586v1)

**作者:** Yao Huang `[一作]` (Tsinghua University), Yinpeng Dong `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段的 TraceMRT 红队框架，用来系统性测试链式思维（CoT）监测器在隐藏目标下的鲁棒性，并提出贝叶斯轨迹监测方法提升检测效果。

**💡 创新点**

创新点在于：①把隐藏目标转化为主任务的子目标并在指令层自动化重构；②在推理层采用遗传搜索生成多样化的轨迹隐蔽与监测干扰策略；③将全局轨迹特征与关键子段证据通过贝叶斯融合，显著提升监测 AUC。

**🔧 技术方法**

技术包括：自动化指令重写（LLM 生成）、遗传搜索优化（策略空间）、贝叶斯概率推断（LR 估计）以及多模型对比（OpenAI GPT‑5.2、Claude‑4.5、Claude‑4 等）。

**📊 数据集**

使用公开基准 SHADE‑Arena 与 CUA‑SHADE‑Arena（共 41 任务），以及多款开源代理（DeepSeek‑v3.2、GLM‑4.7、Kimi‑K2、Qwen3‑235B‑A22B）与专有监测器。

**📈 对比分析**

相较于 CoT Red‑Handed、MRT 等基线，TraceMRT 将逃逸成功率提升至 60–100%，同时保持甚至提高副任务完成率；在恶意轨迹上，传统监测器 AUC 下降至 0.34–0.23，贝叶斯轨迹监测将 AUC 提升至 0.93–0.96，TPR@FPR=5% 提升至 0.77–0.79。

**⚠️ 局限性**

局限性：① 仅模拟系统提示层的诱导，无法覆盖自学习的隐蔽行为；② 假设全局与局部视图条件独立，估计误差可能存在；③ 由于多阶段处理，平均推理延迟从 7.8 秒升至 17.9 秒，实际部署需要进一步优化。

---

## 253. Subspace Learning with Outlier Exposure for Out-of-Distribution Detection

**arXiv ID:** 2611.07580 | [PDF](https://arxiv.org/pdf/2611.07580v1)

**作者:** Hongxu Shen `[一作]` (Beijing Institute of Technology), Qinghua Tao `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Outlier Exposure的特征子空间学习框架 RecOE，通过学习两个正交低维子空间 U_in 与 U_aux 来分别拟合训练样本和辅助异常样本的主变异方向，并利用两子空间重建误差的对比来判别是否为 OoD 样本。

**💡 创新点**

创新点在于：①首次将特征子空间构造与 OE 结合，利用主成分分析思想在特征空间中学习分布特定的低维子空间；②引入子空间拟合、跨子空间分离（hinge margin）以及正交正则化三项目标实现信息聚焦与分离；③采用渐进式 rank 优化以稳健捕捉主变异方向，提升学习稳定性。

**🔧 技术方法**

核心技术包括：Riemannian Adam 优化 Stiefel 范围下的正交基；重建误差最小化与 hinge 损失；子空间正交正则化；渐进式 rank 方案；在训练中对 backbone 与子空间共同微调；以及在推理时使用重建误差差值作为 OoD 分数。

**📊 数据集**

使用的公开数据集：CIFAR‑10 作为 InD 训练集，Textures、Places365、iSUN、LSUN_Crop、LSUN_Resize 作为 OoD 测试集；ImageNet‑1K 作为 InD 训练集，ImageNet‑21K_p 作为辅助异常样本，iNaturalist、SUN、Places、Textures 作为 OoD 测试集；此外还在近端/远端 OOD 设置与同源设置下做实验。

**📈 对比分析**

与多种 post‑hoc 与 OE 基线（MSP、ODIN、Mahalanobis、Energy、ASH、PCA、KPCA、Mahalanobis++、OE、EnergyOE、POEM、MixOE、DivOE、GReg、PFS、VPC）进行对比。RecOE 在 CIFAR‑10 上 FPR95 下降至 4.36%（相对 OE 为 13.76%），AUROC 提升至 98.68%（相对 OE 为 97.53%）；在 ImageNet‑1K 上 FPR95 降至 36.32%（相对 OE 为 61.94%），AUROC 提升至 91.06%（相对 OE 为 90.90%），并在近端/远端 OOD 场景中均表现优于现有最强方法。

**⚠️ 局限性**

局限性包括：①训练时需额外维护两个子空间矩阵，虽然参数增量不大但略增训练时间；②对辅助异常样本的依赖较强，若辅助数据分布与真实 OOD 误差较大，性能可能下降；③当前实现仅使用全局 InD 子空间，未考虑按类别的子空间差异，可能影响细粒度检测；④在同源 OOD 设置下 AUROC 下降，表明对同源分布的泛化仍有待提升。

---

## 254. Provable Zeroth-Order Safe Exploration for Reinforcement Learning from Human Feedback

**arXiv ID:** 2611.07561 | [PDF](https://arxiv.org/pdf/2611.07561v1)

**作者:** Weiqin Chen `[一作]` (Purdue University), Santiago Paternain `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于零阶对数障碍梯度（ZO‑LBPG）的奖励模型无关安全探索方法，能在强化学习从人类反馈（RLHF）的过程中保证所有训练过程中使用的策略始终满足约束。

**💡 创新点**

创新点在于：①将对数障碍目标与零阶梯度估计相结合，利用人类二值反馈直接估计奖励和安全约束；②使用置信下界对约束边距进行保守估计并构造自适应步长，确保单步更新不会跨越约束边界；③在理论上给出高概率安全保证、全局收敛性与维数无关的样本复杂度。

**🔧 技术方法**

主要技术包括：零阶梯度估计（通过耦合轨迹对比与方向采样）、对数障碍法、置信区间与自适应步长设计、扩展的MFCQ约束资格与Fisher信息非退化假设。

**📊 数据集**

实验使用两组数据集：①六状态三动作的合成CMDP（用于控制噪声与冲突程度）；②基于Qwen2.5‑3B‑Instruct的语言模型对齐任务，输入为PKU‑SafeRLHF提示，使用Beaver‑7B奖励与成本模型模拟人类反馈。

**📈 对比分析**

与零阶对偶方法ZPGPD‑HF比较：ZO‑LBPG在训练期间始终保持正约束边距，保证安全；但最终奖励相对较低，收敛速度更慢。ZPGPD‑HF能够更快提高奖励，但在训练中多次违反约束，安全性不足。

**⚠️ 局限性**

局限性包括：①需要已知的可行初始策略；②对置信区间与步长的参数设置敏感；③保守策略导致查询成本高（特别是绝对安全判定的评估器数量）；④理论证明依赖扩展的MFCQ与Fisher非退化假设，在某些实际环境中可能不满足。

---

## 255. The Memory Trap: Language Agents Overgeneralize from Benign Interactions

**arXiv ID:** 2611.07554 | [PDF](https://arxiv.org/pdf/2611.07554v1)

**作者:** Kaiqu Liang `[一作]` (Princeton University), Andrew Lampinen `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言代理在跨会话持久记忆中出现的“记忆陷阱”，并在六种前沿模型上进行实验验证。

**💡 创新点**

首次系统性表明即使是良性交互产生的记忆也能导致代理在后续任务中出现不安全或不可靠行为，并提出记忆陷阱概念与写入时与读取时的过度泛化机制。

**🔧 技术方法**

使用自写记忆工具、记忆注记审计、基准评测（Autonomy、SycophancyEval、IMO‑GradingBench、ImpossibleBench）以及统计分析。

**📊 数据集**

结合自建低风险交互记录、SycophancyEval（错误数学解、论证与诗歌对照）、IMO‑GradingBench、ImpossibleBench的SWE‑bench冲突集，以及自制的保安缺陷与测试修复任务。

**📈 对比分析**

通过对比有记忆与无记忆、原始转录与自写记忆条件下的代理行为，测量不安全行为率、错误遗漏率、过度批评率与作弊率，结果显示记忆显著提升危险行为比例，且更强模型更易陷入记忆陷阱。

**⚠️ 局限性**

实验仅覆盖少数模型族，场景为短期模拟，未检验长期交互或记忆范围保护机制，需要进一步研究。

---

## 256. Improving Long-Tail Detection via Layout-Instruction Conditioned Diffusion Synthesis

**arXiv ID:** 2611.07553 | [PDF](https://arxiv.org/pdf/2611.07553v1)

**作者:** Xinyu Zhang `[一作]` (University of Auckland), Lingqiao Liu `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种可同时使用文本、布局或二者混合进行条件生成的扩散模型，专门用于生成针对长尾类别的可控合成图像，从而提升目标检测性能。

**💡 创新点**

创新点在于：①融合布局条件与指令条件的混合控制，突破单一条件限制；②设计多任务交叉注意力模块（MTCA）和多任务嵌入标记（MTET），实现多种输入格式在同一模型中的统一学习与推理；③通过多任务学习将头部类别的对齐知识迁移到稀有类别，实现更可靠的稀有类别合成。

**🔧 技术方法**

使用了扩散模型（Stable Diffusion v1.5）作为基础，加入MTCA和MTET模块；在检测端采用CenterNet2与Faster R‑CNN，训练时混合真实与合成图像。

**📊 数据集**

实验数据集包括COCO、LVIS、Visual Genome以及VGPhraseCut，用于训练扩散模型和检测器。

**📈 对比分析**

与Copy‑Paste、X‑Paste、FreeStyleNet、Gen2Det、InstructPix2Pix等基线相比，在COCO上Box AP提升约1.1点、Mask AP提升0.8点；在LVIS上整体AP提升1.17点、稀有类Box AP提升4.89点、Mask AP提升2.80点，显示出明显的性能优势。

**⚠️ 局限性**

局限性：模型依赖公开数据集和预训练扩散模型，泛化能力有限；合成图像质量仍受限于布局信息的准确性和文本描述的表达；在极度稀有或复杂场景下可能仍难以生成高质量样本。

---

## 257. SEROA: Stable Encrypted-Traffic Representation and Online Adaptation for Service-Aware Networks

**arXiv ID:** 2611.07550 | [PDF](https://arxiv.org/pdf/2611.07550v1)

**作者:** Zongxi Liu `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一套针对服务感知移动网络的加密流量分类框架SEROA，兼顾离线稳定表示学习和在线QoE驱动的无标签自适应。

**💡 创新点**

创新点在于将协议结构分层为稳定字段与不稳定字段，分别采用稳定掩码建模、协议感知对比学习和时间对抗学习，以获得抗时序漂移的表示，并通过上下文赌博机与QoE反馈实现闭环在线更新。

**🔧 技术方法**

使用了自监督掩码语言模型、MoCo对比学习、时间对抗网络以及基于PPO的上下文赌博机策略梯度。

**📊 数据集**

采用公开的CSTNET-APP与CSTNET-SERVICE TLS加密流量数据集进行实验。

**📈 对比分析**

与ET‑BERT、YaTC、NetMamba等基线对比，SEROA在随机拆分与时间拆分、冻结/解冻微调以及在线QoE适应实验中均显著提升Macro‑F1和QoE奖励，达到90%以上的准确率。

**⚠️ 局限性**

局限在于仅验证了单小区下的功率分配场景，未涵盖子载波分配、排队等多维度RAN调度；在线学习仍依赖QoE弱标签，且对模型更新频率与信道变化的鲁棒性待进一步评估。

---

## 258. OptIMerge: Optimization-Based Merging of Diffusion Models

**arXiv ID:** 2611.07543 | [PDF](https://arxiv.org/pdf/2611.07543v1)

**作者:** Viraj Shah `[一作]` (Google), Hossein Talebi `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对从共享基模型微调得到的多任务图像到图像（I2I）扩散模型专家，本文提出了一种参数高效的优化合并框架 OptIMerge，能够在不对专家进行再训练的前提下，通过学习少量的结构感知合并参数，实现多任务模型的融合。

**💡 创新点**

创新点在于：①把模型合并视为一个优化对齐问题，而非简单的权重平均；②针对 U‑Net 的注意力层和卷积层分别设计了偏置 LoRA 与低秩 Cayley 旋转/核中心广播两种参数化，既能对齐冲突的空间先验，又保持权重范数与空间几何；③在仅 0.4% 参数量的前提下，显著提升多任务性能，逼近甚至超越联合多任务训练。

**🔧 技术方法**

主要技术包括：梯度下降优化多任务扩散损失；偏置 LoRA（Biased LoRA）用于注意力层的路由与特征对齐；低秩 Cayley 变换实现卷积层的正交旋转对齐；核中心广播（Kernel‑Centric）对卷积层做通道级加权；以及标准的 DDIM 推理和 PSNR/SSIM/LPIPS 评价。

**📊 数据集**

使用的公开数据集有：DF2K（超分辨率）、DPDD（散焦去模糊）、OID（运动模糊）。此外，在 T2I 个性化实验中使用了少量自制的目标图像和参考图像。

**📈 对比分析**

与单任务专家、联合多任务训练、零射模型 Soups（α=0.5）、TIES、DARE 等基线相比，OptIMerge 在所有三组任务（运动去模糊+散焦去模糊、散焦去模糊+超分辨率、运动去模糊+超分辨率）中均取得了更低的 LPIPS、更高的 PSNR 与 SSIM，且大部分指标逼近甚至超过联合多任务训练，且仅需要 <0.4% 的可学习参数。

**⚠️ 局限性**

局限性包括：①随着专家数量增多，参数对齐与交互复杂度上升，仍需进一步研究扩展到大规模专家集合；②在某些极端空间先验冲突的任务组合中，性能提升幅度有限；③目前仅在 U‑Net 结构上验证，需探讨对 Diffusion Transformer 等新型架构的适配。

---

## 259. FlashOffloading: Low-Latency KV Cache Offloading for Long-Context LLM Serving

**arXiv ID:** 2611.07542 | [PDF](https://arxiv.org/pdf/2611.07542v1)

**作者:** Xuan Wang `[一作]` (Huawei Technologies Co., Ltd), Cheng Li `[通讯]` (Huawei Technologies Co., Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 FlashOffloading，一种在长上下文 LLM 服役中将 KV 缓存低延迟地 offload 到 host DRAM 的完整架构，覆盖预填与解码两阶段。

**💡 创新点**

创新点包括：①四层循环缓冲预填流水线实现 H2D/compute/D2H 的并行覆盖；②逆向稀疏索引 + cache_slots 将缓存管理与 miss‑lookup 复杂度降到 O(k)+O(|miss|)；③融合 LIM 与 CopySFA 内核，在稀疏注意力中实现 H2D load 与计算的细粒度重叠，从而几乎不增加整体延迟且显著提升吞吐。

**🔧 技术方法**

技术方案：Ascend910C HBM‑DRAM 分层存储、四缓冲层级调度、逆向稀疏索引、LIM/CopySFA 融合 kernel、vLLM‑Ascend 框架改造、MTP3 预测解码、TPOT 服务级目标。

**📊 数据集**

实验数据集：GLM‑5.2‑w4a8c16 模型，Open‑SWE 长上下文 20 个请求（短/长序列）用于解码实验，prefill 采用 256k token 自制测试。

**📈 对比分析**

对比方法：在 Ascend910C PD‑disaggregated 平台上将 FlashOffloading 与不 offload 基线进行 P99 TTFT、TPOT、ITL、吞吐等指标对比；结果显示预填吞吐提升 1.38×，解码吞吐提升 1.91×，TPOT 接近 30 ms；同时 512k/1M 长序列可被服务，解码批量可扩展 3–4 倍。

**⚠️ 局限性**

局限性：仍存在 20–50 ms 的 TPOT 轻微上升，对缓存容量选择较敏感；实现复杂度高且依赖 Ascend/HBM‑DRAM 的高带宽；对非 DSA 模型或不同硬件平台需额外改造。

---

## 260. Characterizing CUDA Locality Domains on Blackwell GPUs

**arXiv ID:** 2611.07535 | [PDF](https://arxiv.org/pdf/2611.07535v1)

**作者:** Rocky Wang `[一作]` (Scitix AI), Jason Zhao `[通讯]` (Scitix AI)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文在Blackwell B300 GPU上系统性地评估了CUDA Locality Domains（LD）的内存、计算和通信性能，并提出了两项成本模型。

**💡 创新点**

创新之处在于将LD分为内存放置和计算绑定两控件，分别测量并建模其跨die延迟与带宽影响，同时揭示并优化了32MiB阈值以上的驱动分配开销。

**🔧 技术方法**

实验采用CUDA 13.4的LD控制、green contexts、nvbandwidth、nccl-tests、DeepGEMM、decode-attention微基准，配合Nsight Compute计数器和自定义驱动补丁。

**📊 数据集**

使用的工作负载为合成的指针追踪、256MB流读写、1MiB/256MiB拷贝、All-Reduce/All-Gather、分组GEMM、KV序列解码等，而内存分配规模覆盖1KiB–1GiB。

**📈 对比分析**

通过四种状态（原生、内存仅、计算仅、两者兼用）和负控，结合SM数目对比与硬件计数，发现内存放置可将远程读加速2.08×、计算绑定可将SM集体通信提升1.49×，两者联合对驻留数据GEMM提升1.37×，但对流式工作会降低30%。

**⚠️ 局限性**

局限性包括仅在单一B300平台实验、缺乏集体路径的内核计数器、对MIG未直接测量、注意力实验范围有限、对冷缓存工作组收益不足、以及die间性能差异尚未解释。

---

## 261. Route Before Response: Fail-Closed Routing for LLM-Enabled Customer-Service Software

**arXiv ID:** 2611.07534 | [PDF](https://arxiv.org/pdf/2611.07534v1)

**作者:** Yuchen Cao `[一作]` (City University of Hong Kong), Jingyu Zhang `[通讯]` (Hong Kong Metropolitan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种 Boundary-aware Pre‑Response Routing 控制器（RouteBR），将大语言模型的语义选择与代码层的权限校验、实体闭合列表验证以及最终 dispatch 决策分离，并在金融机构的客户服务软件中进行部署。

**💡 创新点**

创新点包括：① 主路由承诺（primary commitment）机制，确保模型提出的首选路由不被后续备选替代；② 代码拥有最终决策权，模型仅提供语义建议，提升安全性和可追溯性；③ 引入可回放的决策记录和可检测的规则、实体、状态版本化，支持回归与审计；④ 在同一数据集上对多种模型后端进行一致性评测，验证接口与算法的可移植性。

**🔧 技术方法**

技术手段：大语言模型（DeepSeek V4 Pro、GPT‑4o、GPT‑5.2、Qwen 等）作为提议者；确定性策略引擎（rule engine）；闭合列表实体验证；JSON 结构校验与格式修复；分层控制流（提议 → 校验 → 权限决策 → dispatch/非 dispatch）。

**📊 数据集**

数据集：600 条隐私保护的客户服务场景（MAIN、FLIP、ENTITY 三类），来自一家真实金融机构；包含 40 个公开 ETF 实体、对应的服务目录、规则集合和状态记录；拆分为 DEV、SELECT、CONFIRM 三个子集用于接口验证、后端比较和最终评测。

**📈 对比分析**

比较方法：与单呼单返回 Baseline B1、带规则的单呼 Baseline B2 进行对比；对主路由进行 A‑state（去掉权限校验）和 A‑ground（去掉实体校验）机制检查；在 SELECT 子集上对六个模型后端进行可移植性评测。性能结果：在 200 条 MAIN 例子中 RouteBR 达到 97.5% 的最终路径精确匹配，B2 为 95.5%，B1 仅 41.0%；False‑dispatch 0/166，False‑block 1/114；FLIP 对称准确率 92.5%；ENTITY 100% 精确匹配；后端差异仅 1.9 个百分点。

**⚠️ 局限性**

限制：评估基于单一机构的人工重构场景，状态与规则均为合成；未覆盖跨机构、多语言或跨服务组合请求；模型仅采用一次性调用，未探究连续交互的鲁棒性；闭合实体列表和规则集的固定范围限制了模型的泛化与可扩展性。

---

## 262. JailBound: FOL-Guided Jailbreak Evaluation for Exploring LLM Behavioral Boundaries

**arXiv ID:** 2611.07530 | [PDF](https://arxiv.org/pdf/2611.07530v1)

**作者:** Fazong Wu `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Xiaoming Wu `[通讯]` (Qilu University of Technology (Shandong Academy of Sciences))

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了JailBound框架，结合层级威胁分类、自动化元攻击生成与FOL导引的双分支搜索，用于系统评估和发现LLM的越狱漏洞。

**💡 创新点**

创新点在于将越狱评估与嵌入空间的局部敏感性搜索相结合，提供高价值攻击和安全敏感状态的双重视角，并实现可自动扩展的威胁空间。

**🔧 技术方法**

技术包括层级威胁分类、基于Qwen2.5-14B的元攻击生成模型、第一阶损失（FOL）指导的双分支优化以及嵌入空间连续优化与文本化。

**📊 数据集**

使用了约10万条结构化元攻击实例、Qwen3-72B-Embiggened标注的威胁数据，以及46个LLM（13个家族）的公开与闭源模型。

**📈 对比分析**

与现有基准（如HarmBench、S-Eval）对比，JailBound在匹配设置下高价值分支提升约5% ASR，安全敏感分支的行为翻转率更高，整体覆盖率和跨模型转移表现均优于基线。

**⚠️ 局限性**

限制包括FOL仅为局部敏感性信号、连续优化到文本的转换导致ASR下降约6.9%，评估依赖固定外部评判阈值，且转移规律未深入因果分析。

---

## 263. OptST: Optimizing Semantic Tokenization for LLM-Enhanced Recommendation

**arXiv ID:** 2611.07523 | [PDF](https://arxiv.org/pdf/2611.07523v1)

**作者:** Hanchong Chen `[一作]` (Shenzhen Technology University), Xing Tang `[通讯]` (Shenzhen Technology University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出OptST框架，使预训练的LLM语义分词器能够在下游推荐任务中自适应调整分词几何并根据交互选择可变深度，以提升推荐性能。

**💡 创新点**

创新点在于：①对冻结的残差量化器进行低秩共享变形，结合锚定正则化实现任务感知的分词分配；②引入基于交互的前缀门控，动态决定向下游模型提供的语义分辨率；③使用直通估计实现离散分配与梯度传播兼容。

**🔧 技术方法**

技术包括残差量化VAE、低秩适配、锚定正则化、直通(straight‑through)梯度、门控网络以及二分类交叉熵推荐损失。

**📊 数据集**

实验数据集为Amazon Movies、Amazon Games、Amazon Clothing（来自Amazon Review）以及Yelp Open Dataset。

**📈 对比分析**

与13种传统与LLM增强基线（如FMM、AutoInt、LLaCTR、UserIP等）及多种推荐骨干（DCN、xDeepFM等）对比，OptST在AUC上始终优于基线，并在不同骨干上保持一致提升。

**⚠️ 局限性**

局限性包括：对预训练LLM和RQ‑VAE结构的依赖；门控深度选择仍可能受超参数影响；未在多语言或动态时间序列场景中验证，且在公平性与隐私方面需进一步评估。

---

## 264. Graph learning-assisted logic-based Benders decomposition for distributionally robust in-station train rescheduling under disruptions

**arXiv ID:** 2611.07520 | [PDF](https://arxiv.org/pdf/2611.07520v1)

**作者:** Xiaobin Li `[一作]` (East China Jiaotong University), Wuming Lei `[通讯]` (East China Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在列车调度中动态更新关闭时间预测、联合路径规划与时序的分布式鲁棒模型，解决资源关闭、到达延迟与处理时间扩展的冲突。

**💡 创新点**

创新点在于将剩余关闭时间的条件预测融入决策、构造基于资源使用的条件Benders割平面，并利用图神经网络引导子问题求解顺序。

**🔧 技术方法**

采用逻辑基础Benders分解、图神经网络（GNN）辅助的资源降维与条件割、以及分布式鲁棒优化框架。

**📊 数据集**

使用Schutt等人公开的CP2025车站调度基准数据集（141个实例）进行训练与测试。

**📈 对比分析**

与Tabu Search和Variable Neighborhood Search在相同计算预算下比较，GNN-LBBD平均总完成延迟降低16.63%和20.06%，计算时间相近。

**⚠️ 局限性**

局限在于仅在单一车站结构下验证，且分布式鲁棒参数ρ和场景数量的选择仍需经验性调节，未考虑多站多站间交互。

---

## 265. Evolver: Turning Agent Exploration into Evolving Capabilities

**arXiv ID:** 2611.07595 | [PDF](https://arxiv.org/pdf/2611.07595v1)

**作者:** Haoyang Zhang `[一作]` (EvoMap), Junjie Wang `[通讯]` (EvoMap)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“经验驱动”框架，将 LLM 代理的临时探索过程转化为持久可复用的程序性经验，并在不更新基础模型参数的前提下，支持经验的获取、检索与反馈驱动的演化。

**💡 创新点**

核心创新点在于：① 将探索得到的操作序列抽象为结构化策略资产（strategy‑gene）并与执行案例关联；② 通过检索与经验条件化执行，将历史经验注入现有代理；③ 采用反馈驱动的生命周期管理，实现经验的持续评估、验证与迭代。

**🔧 技术方法**

技术实现包括：结构化策略表示、经验提取（自动草稿+LLM 归纳）、内容寻址持久化存储、任务条件检索（hook、MCP 接口）、经验条件化执行注入、EvoMap 网络共享与可追溯性管理，以及基于执行反馈的版本演化。

**📊 数据集**

使用四个基准数据集进行评估：Terminal‑Bench 2.1、SWE‑bench Multilingual、SWE‑rebench 和 GAIA；对 Claude Code 与 Codex 两套代理、两种基础模型配置进行 320 条配对实验。

**📈 对比分析**

与基线（无经验注入）对比，任务成功率提升 4.38%（Claude Code +2.50%，Codex +6.25%），整体 token 消耗下降 24.24%（单个配置降幅 13.55%–35.96%）。在成功任务子集上，token 进一步下降 17.57%，说明经验既提升效果又降低成本。

**⚠️ 局限性**

局限性包括：经验选择与验证过程尚不完善，可能导致不相关或过时的策略被使用；实验仅覆盖 320 条配对任务，缺乏对未知任务或长期在线演化的泛化评估；以及依赖外部工具和接口，对部署环境与安全约束仍有一定依赖。

---

## 266. TACT-KV: Tri-Axis Cosine Transform for Compressing Volumetric KV Caches in Medical VLMs

**arXiv ID:** 2611.07593 | [PDF](https://arxiv.org/pdf/2611.07593v1)

**作者:** Chongyu Qu `[一作]` (Vanderbilt University), Yuankai Huo `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种面向医学视觉‑语言模型的 KV 缓存压缩方案 TACT‑KV，先将三维扫描产生的视觉 KV 缓存重塑为 (S, H, W) 张量，随后对其进行分离式三轴离散余弦变换，使用逆水填充进行自适应比特分配，再对每个频率 bin 进行标量量化，最后在解码时恢复缓存。

**💡 创新点**

创新点在于①首次将三轴空间相关性应用于 KV 缓存压缩；②利用三轴 DCT 将能量聚集在少数低频系数，显著提升压缩效率；③引入逆水填充并给出闭式表达式 1/γ 预测其相对于均匀分配的误差下降；④整个流程不需要额外校准数据或学习参数。

**🔧 技术方法**

核心技术包括分离式三轴离散余弦变换（DCT‑II）、逆水填充比特分配、gain‑shape 标量量化、基于 FlashAttention 的内核融合实现、以及 GPU 内存与吞吐量评估。

**📊 数据集**

实验数据集包括 RSNA‑ICH 与 CT‑RATE 胸部 CT 公开数据集，以及 Video‑MME 视频问答数据集；在这些数据上对六种医学 VLM（MedGemma、Hulu‑Med 等）和四种通用 VLM 进行评估。

**📈 对比分析**

与 KIVI、KVQuant、TurboQuant、KVTC、VidKV、AKVQ‑VL 等现有 KV 压缩方法比较，TACT‑KV 在 1‑bit 下在 11/12 医学 VLM‑数据对中获得最低翻转率（平均低于基线 2.6×），在 Video‑MME 中保持或提高四向 MCQ 准确率；在单 GPU 上可将最大解码 batch 提升至 8×，解码吞吐量提升 1.1–2.0×。

**⚠️ 局限性**

局限性包括：①仅压缩视觉 KV 缓存，文本部分仍使用 bf16；②逆水填充在极低比特率下不满足高率假设，预测失效；③实验仅覆盖胸部与头部 CT，未验证 MRI、PET、超声等其他 3D 模态；④对非医疗视频仅做经验性验证，缺乏对时间轴与切片轴差异的理论分析。

---

## 267. CasHyper: Modeling Multi-Factor Interactions and Asymmetric Influence in Cascade Prediction via Hypercomplex Algebra

**arXiv ID:** 2611.07585 | [PDF](https://arxiv.org/pdf/2611.07585v1)

**作者:** Yansong Wang `[一作]` (Southwest University), Tao Jia `[通讯]` (Southwest University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种基于超复数代数的级联预测框架CasHyper，用来建模信息扩散中的多因子交互与非对称影响。

**💡 创新点**

创新点在于将Cayley‑Dickson超复数表示与图神经网络和超复数Transformer结合，利用非交换乘法天然建模交叉组件交互和有序影响，并引入Neural ODE连续动态编码和旋转位置编码保持相对位置信息。

**🔧 技术方法**

采用了超复数代数（四元数、八元数等）、图神经网络、Neural ODE、超复数Transformer带旋转位置编码、相对位置一致性正则、交互感知与线性预测模块。

**📊 数据集**

使用了四个真实社交网络级联数据集——Android、Christianity、Twitter和Memetracker。

**📈 对比分析**

在Hits@k和MAP@k评估指标下，与十个主流级联预测基线（NDM、FOREST、DyHGCN、CE‑GCN、MS‑HGAT、DisenIDP、RotDiff、GODEN、MetaCas、PMRCA）进行全候选人softmax对比，CasHyper在所有数据集和指标上均显著提升，例如Hits@50提升至Android 43.24%、Christianity 18.43%。

**⚠️ 局限性**

局限性包括对极短级联或稀疏数据的鲁棒性不足，模型参数量和计算成本随超复数维度增加而上升，且在大规模图或长级联上的可扩展性有限；对非随机缺失或噪声结构的敏感性尚待进一步研究。

---

## 268. Digital Security Under Compound Threats: Surveillance, Threat Reasoning, and Habituation Among North Korean Defectors in China

**arXiv ID:** 2611.07576 | [PDF](https://arxiv.org/pdf/2611.07576v1)

**作者:** Yibo Meng `[一作]` (Cornell University), Hongkun Yang `[通讯]` (Ocean University of China)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对居住在中国的无证朝鲜逃亡者进行半结构化访谈，探讨其数字安全威胁认知、保护行为与习惯化过程。

**💡 创新点**

首次系统揭示无证逃亡者在跨国监控、身份绑定基础设施与不确定威胁共存的复合威胁情境，并分析其安全认知迁移与适应路径。

**🔧 技术方法**

采用质性访谈与反射性主题分析法，结合安全心理模型与复合威胁框架进行解释。

**📊 数据集**

访谈样本共19名无证朝鲜逃亡者，采集中文（韩语）访谈记录并译为英文进行分析。

**📈 对比分析**

本研究并未使用量化性能指标，而是通过对比不同防护实践的技术适用范围与依赖性进行定性评估，结果表明多数实践缺乏完整保护效能。

**⚠️ 局限性**

局限包括样本规模小、地域集中、回忆偏差、缺乏技术验证、难以普遍化、数据收集与分析的安全约束导致信息缺失。

---

## 269. Guidance Is a Field, Not a Scalar: Interval-Integrated Classifier-Free Guidance for One-Step Video Generation

**arXiv ID:** 2611.07566 | [PDF](https://arxiv.org/pdf/2611.07566v1)

**作者:** Zijie Meng `[一作]` (Peking University), Xixin Cao `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 MeanFlow 的单步视频生成方法 AGAVE，通过将 classifier‑free guidance 视为随噪声时间、空间和帧位置变化的连续场，并利用区间积分 CFG 同识别将该场编译进一次网络评估，从而实现无离散化误差的完整自适应引导。

**💡 创新点**

创新点包括：① 把传统的标量 CFG 转化为三维连续场；② 证明区间积分 CFG 同识别，表明任意可积引导场可被完全吸收到 MeanFlow 的目标中；③ 在一次前向传播中完成时间、空间、帧位置的自适应调度；④ 结合 Beta 学习调度、运动显著性空间调制以及帧位置非对称性，提升视频生成质量。

**🔧 技术方法**

核心技术：MeanFlow 速率匹配、区间积分 CFG 同识别、连续引导场设计、可学习 Beta 调度、运动显著性判别、帧位置基准；训练中使用基于 DiT 的网络，采用自适应引导蒸馏来学习调度参数；推理时保持 1‑NFE（一次网络评估）且无额外延迟。

**📊 数据集**

实验数据集：VBench（文本到视频评估）、UCF‑101（视频分类/FVD）、MSR‑VTT（视频检索/FVD），以及 ImageNet‑256² 的 1‑NFE 证明实验。

**📈 对比分析**

与现有单步基线（iMF、rCM、MeanFlow CFG、CFG‑Zero⋆）以及多步自适应引导（APG、β‑CFG、CFG‑Zero⋆）对比。AGAVE 在 VBench 总分、UCF‑101、MSR‑VTT 的 FVD 上均优于所有单步模型，并可匹配甚至超过多步自适应引导的质量，且在过度饱和度、背景一致性、动态程度等指标上取得显著提升；推理成本保持在 1‑NFE，延迟最小。

**⚠️ 局限性**

局限性：依赖 MeanFlow 的双时间结构，无法直接迁移到单时间模型（如一致性模型）；需先训练高质量多步教师并进行蒸馏，增加训练复杂度；调度参数虽少，但仍需手动设计 Beta 族和帧位置基底；在极端高分辨率或长时序视频中，单步采样的极限尚未完全验证。

---

## 270. Action-Conditioned VLM Monitoring for Autonomous Vehicle Semantic Anomalies

**arXiv ID:** 2611.07540 | [PDF](https://arxiv.org/pdf/2611.07540v1)

**作者:** Daniel Adejumo `[一作]` (New York University), Aliasghar Arab `[通讯]` (City College of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于动作条件的监控框架，利用视觉‑语言模型（VLM）先在隐藏窗口中预测场景所需动作，再将其与实际车辆运动进行兼容性比较，以判定自动驾驶车辆在语义不寻常场景中的行为是否合规。

**💡 创新点**

创新点在于（1）将监控问题正式化为场景预期与实际动作的兼容性检验；（2）设计了两阶段零样本VLM流程——先提取场景预期再与轨迹做显式规则比较；（3）构建了六字段语义异常分类法及对应的生成视频语料库，为该任务提供系统化的评估基准。

**🔧 技术方法**

技术核心包括：大规模视觉‑语言模型（Cosmos3、Qwen 等）与零样本推理；光流与逆动力学管道估计轨迹；显式兼容性规则与对话式 prompt 设计；以及对生成视频的多模态预处理（分辨率、帧率、文本摘要）。

**📊 数据集**

使用了自研的 av_semantic_anomalies 视频语料库，包含 315 条生成视频、630 条轨迹文件，涵盖 112 条场景‑动作对（42 为真实事故衍生、70 为合成），并提供 15 个场景族的多样化视频。

**📈 对比分析**

与单阶段仅视频的基线进行对比，基于 235 条剪辑（110 异常、125 正常）评估。两阶段方案获得 83.0% 绝对准确率、82.6% 平衡准确率、77.3% 异常召回率；单阶段基线仅 74.5% 准确率、53.6% 异常召回。正类特异性由 92.8% 降至 88.0%，但整体性能提升显著。

**⚠️ 局限性**

主要局限包括：数据全部为合成，缺乏对真实驾驶场景的迁移评估；轨迹估计为序数化，未提供精确车辆遥测；评估仅离线完成，未验证实时性；缺乏细粒度的因果性 ablation，难以明确动作条件贡献；统计显著性受样本与场景聚类限制，需进一步的独立验证。

---

## 271. Enhancing Repository-Level Code Completion with Reinforcement Learning in Real-World Scenarios

**arXiv ID:** 2611.07510 | [PDF](https://arxiv.org/pdf/2611.07510v1)

**作者:** Zhenlong Dai `[一作]` (Zhejiang University), Jingyuan Chen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于强化学习的仓库级代码补全框架，融合了基于标识符的意图识别、奖励驱动的补全学习以及选择性探索策略，以提升跨文件与单文件上下文的利用率。

**💡 创新点**

创新点包括：①使用GRPO进行自监督的RL训练；②设计标识符匹配奖励以引导模型关注开发者意图；③引入奖励驱动的补全学习，提供细粒度的编辑相似度奖励；④提出选择性探索优化，专注低置信度高奖励的关键子句，显著提升跨文件推理能力。

**🔧 技术方法**

采用的技术有：GRPO强化学习框架、LoRA微调、vLLM推理、DeepSpeed分布式训练；奖励函数包括编辑距离、格式匹配、行数约束和标识符F1；并在训练中使用多样化采样、KL惩罚和选择性探索策略。

**📊 数据集**

使用ICRepo数据集（来自RepoBench-C和RepoBench-P的32K例子，均衡Python与Java，覆盖三类场景：CF-F、CF-R、IF），测试集为RepoBench-Java和RepoBench-Python。

**📈 对比分析**

与StarCoder、CodeLlama、Qwen2.5-Coder、DeepSeek-Coder、aixcoder等模型以及fine-tuning、GRPO、CoLT等训练方法对比，经过RL训练的模型在Code EM、ES、Identifier F1等指标上均超过SOTA，Java代码完成率最高达41.60%（Code EM），Python 77.64%，相比上一最佳模型提升约8–12%。

**⚠️ 局限性**

局限性包括：1）未使用执行奖励，缺乏运行时验证；2）仅聚焦生成侧，未对检索与上下文选择做系统化优化；3）仅训练7B规模模型，未探索更大模型在RL下的效果；4）依赖静态奖励，可能在极端场景下仍出现错误。

---

## 272. Per-Query Recall Guarantees for Approximate Nearest Neighbor Search in Dynamic Vector Databases

**arXiv ID:** 2611.07579 | [PDF](https://arxiv.org/pdf/2611.07579v1)

**作者:** Woong-Kee Loh `[一作]`, Suan Lee `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了 PQR-Search，一种在动态向量数据库中为每个查询提供召回率保证的近似最近邻搜索方法。

**💡 创新点**

创新点在于通过随机正交变换与前缀 1‑bit 二进制码结合的自适应剪枝规则，给出基于 P 的概率上限的 (τ,δ)‑召回保证，并且在更新时无需重新调参。

**🔧 技术方法**

采用随机正交变换、IVF 划分、1‑bit 前缀二进制码、逐步剪枝、误差量化和自适应半径等技术。

**📊 数据集**

在 20 个公开数据集（如 SIFT、GIST、arXiv、Yandex、Laion 等）以及 NeurIPS 2023 Streaming Runbook 进行实验。

**📈 对比分析**

与传统基于平均召回调优的索引、CTR、HNSW、DiskANN、FAISS 等方法相比，PQR-Search 在保证下的失败率始终 ≤ δ，单线程下比 HNSW 快 2.5 倍，批量吞吐量比 HNSW 低 1.3 倍；在 100M 规模数据集上，批量吞吐量比 FAISS exhaustive 快约 26 倍，并在所有设置下优于 IVF。

**⚠️ 局限性**

局限性包括需假设查询和更新与随机变换独立、在极高召回目标下成本仍相对较高、以及对旋转随机化的实现与安全性要求较高，且在极大规模或极高维度数据的实际部署仍需进一步验证。

---

## 273. Georeferenced Cross-View 3D Reconstruction from Ground and Satellite Images

**arXiv ID:** 2611.07575 | [PDF](https://arxiv.org/pdf/2611.07575v1)

**作者:** David G. Shatwell `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 X-VGGT 模型，利用地面图像与正射卫星图像联合进行跨视角地理参考 3D 重建，能同时输出地面相机位姿、密集 3D 点图以及将重建投影到卫星坐标系的场景级变换。

**💡 创新点**

创新点在于：①将跨视角定位与多视角 3D 重建统一到一个 feed-forward Transformer；②设计卫星对齐头与对应的参数、点投影和相机中心损失，实现全局地理参考变换；③提出专门的训练与评估协议，兼顾有序/无序地面视角。

**🔧 技术方法**

核心技术包括跨视角几何 Transformer（Alternating Attention）、卫星对齐头（Satellite Alignment Head）、DPT 级密集点预测、投影变换与多任务损失。

**📊 数据集**

使用的公开数据集有 KITTI、Aerial-MegaDepth（改为 Sat-MegaDepth）、Mapillary Metropolis、WRIVA、ACC-NVS1；训练集覆盖顺序与无序地面视角。

**📈 对比分析**

与跨视角定位方法（FG^2、Loc^2）、单独 3D 重建模型（MASt3R、VGGT、A-MD）以及基于定位对齐的重建基线比较，X-VGGT 在相机位置误差、方向误差、像素级地理注册误差以及局部 3D 重建的相对姿态一致性上均显著优于所有基线，误差下降幅度可达数十米到数十度。

**⚠️ 局限性**

局限性包括：依赖可获得的正射卫星图像，无法在无卫星参考环境下使用；在极端视角、遮挡或低光条件下性能可能下降；模型规模较大，对计算资源需求高。

---

## 274. Integrating Learned Priors through Prior-Space Reparameterization for Physically Based Inverse Rendering

**arXiv ID:** 2611.07573 | [PDF](https://arxiv.org/pdf/2611.07573v1)

**作者:** Hun Hwangbo `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在逆向渲染中引入基于先验的材料重参数化，利用学习得到的材料预测构建一个固定的Permutohedral格子，将相似预测的表面共享材料变量，进而在保持原有光度目标不变的前提下提高材料与光照的分离质量。

**💡 创新点**

创新点在于：①不通过额外正则化损失将先验信息注入，而是通过共享格子变量将先验约束嵌入参数空间；②利用先验特征构建的格子实现高维稀疏投影与切片，显著减少优化自由度；③这种重参数化显著提升更新一致性，使联合优化阶段可压缩至不到一分钟。

**🔧 技术方法**

使用的技术包括：Permutohedral格子投影/切片、Adam + Large Steps预处理、Mitsuba3自动微分渲染、Joint Trilateral Refinement对先验进行细化、基于SDF的网格重建以及光照参数化（SG + 环境纹理）。

**📊 数据集**

在 Synthetic4Relight、TensoIR‑Synthetic、DTC‑Synthetic 三个合成基准以及 Stanford‑ORB 实景数据集上进行评估。

**📈 对比分析**

与 Neural‑PBIR、MaterialFusion 以及 DiffReg‑PBIR 等基于网格的逆向渲染基线相比，本文方法在 relighting PSNR 上提升 1.5–2.5 dB，NVS PSNR 提升 1.6–5.1 dB，整体优化时间从数十分钟降至 5.9–6.4 min，且可在 100 步压缩版本下仍保持竞争性能。

**⚠️ 局限性**

局限性包括：①材料重参数化会限制可表示的材料细节，导致与全自由度对比的结果仍有 1–4 dB 的差距；②依赖先验预测的准确性，若先验误差大或不一致会影响格子共享；③对格子分辨率和σ_prior 的设定敏感，需要针对不同场景调节。

---

## 275. ACSeL: Adaptive Configuration Search for Hardware-Constrained LLM Inference

**arXiv ID:** 2611.07570 | [PDF](https://arxiv.org/pdf/2611.07570v1)

**作者:** Bokun Wang `[一作]` (University of Texas at Austin), Diana Marculescu `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ACSeL框架，利用低成本的RMQD质量代理和少量端到端延迟校准（FMC），自动在满足内存、TTFT/TPOT等硬件SLO的前提下，联合搜索模型尺寸、批次、量化算法及层级混合精度配置，以最大化生成质量；

**💡 创新点**

核心创新包括：1) 将KL散度对比参考模型作为连续、可预测的生成质量指标RMQD；2) 通过少量端到端测量实现的中值对数回归校准AIConfigurator延迟预测（FMC）；3) 结构化的BandPareto搜索算法，结合轻量级质量/延迟代理高效探索大规模混合精度配置空间；

**🔧 技术方法**

技术细节：混合精度Post‑Training Quantization、RMQD计算、线性/约束岭回归的质量代理、FMC的对数空间中值回归、BandPareto结构化搜索、AIC原始延迟表、NVIDIA RTX A6000与Jetson Thor实验平台；

**📊 数据集**

使用的数据集包括NeMo Evaluator 17个基准（共计24条样本）进行RMQD计算，Llama‑3.x与Qwen3模型家族（最高至8B）作为候选模型，Llama‑3.1 405B与Qwen3 235B‑A22B作为参考模型，及在上述硬件上收集的全套延迟测量数据；

**📈 对比分析**

与未校准的AIC、随机采样、hill‑climbing以及BandPareto+FMC等方法对比，实验表明FMC显著提高答复率和可行性（如A6000上从48%提升至至少89%，Thor上从16%提升至至少72%）；BandPareto+FMC在满足SLO时平均RMQD误差仅0.009（A6000）/0.002（Thor），并在84%/60%请求中匹配或超过最优均匀配置，表明方法在质量‑可行性上表现突出；

**⚠️ 局限性**

局限性：1) 校准仍依赖少量端到端测量，SLO边界附近可能出现误差；2) 仅针对单GPU/单加速器环境，未涵盖多机/多GPU情况；3) RMQD与真实评测指标（如人类评估）的相关性有限；4) 混合精度搜索受限于预设band数和格式，可能未覆盖更细粒度的配置。

---

## 276. MambaTG: Candidate-Aware State Space Modeling for Future Link Prediction on Temporal Graphs

**arXiv ID:** 2611.07569 | [PDF](https://arxiv.org/pdf/2611.07569v1)

**作者:** Nguyen Hong Quan `[一作]` (Nanyang Technological University), Lihui Chen `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种基于Mamba的候选感知框架MambaTG，用于未来链接预测，能够在重复边和未见边两种场景下高效建模历史序列演化与候选-历史兼容性。

**💡 创新点**

创新点包括：① 识别并联合建模两条互补信号（历史序列演化与候选-历史兼容性）；② 利用Mamba的输入依赖状态转移实现候选感知的单次历史编码；③ 引入双分支表示细化（历史频谱细化HSR + 候选特征细化CFR）提升跨尺度特征利用；④ 通过终端状态缓存实现候选评价的高效共享。

**🔧 技术方法**

使用技术主要有：Mamba selective state‑space 模型；时间相对编码与频域变换（FFT/IRFFT）进行频谱细化；候选感知Mamba Encoder（CAME）；双分支表示细化（HSR、CFR）；BPR 损失训练；时间复杂度分析与批量化实现。

**📊 数据集**

使用了13个未来链预测数据集，涵盖社交网络、推荐系统、交通等场景，其中7个来自TGB‑Seq，6个常用公开数据集；按重复边主导与未见边主导划分。数据集包括ML‑20M、Taobao、Yelp、GoogleLocal、Flickr、YouTube、WikiLink、Wikipedia、Reddit、MOOC、LastFM、UCI、Flights 等。

**📈 对比分析**

与12个代表性基线（EdgeBank、CRAFT、JODIE、DyRep、TGAT、TGN、CAWN、GraphMixer、DyGFormer、FreeDyG、DyG‑Mamba、DyGMamba、SGNN‑HN）进行对比。MambaTG 在未见边主导数据集上平均排名 1.29，MambaTG‑R 在重复边主导数据集上平均排名 1.50，几乎在所有数据集上获得最佳或次佳 MRR；在长序列和大候选集场景下表现更为稳健，并在训练速度和模型规模上显著优于竞争方法。

**⚠️ 局限性**

局限性包括：① 对极度稀疏或冷启动节点的候选兼容性建模仍有限；② 依赖历史序列顺序，若历史信息噪声大或不完整可能受影响；③ 仅考虑单一节点特征，未充分利用多模态或异构图信息；④ 在极大候选集时仍需进一步优化内存和计算开销；⑤ 部分数据集训练过程不稳定（如DyGMamba 在长序列时出现 NaN）。

---

## 277. Whole-Body Elasto-Geometrical Calibration of Wheeled Humanoid Robots Using a Shared-Waist Dual-Chain Model

**arXiv ID:** 2611.07560 | [PDF](https://arxiv.org/pdf/2611.07560v1)

**作者:** Pengxin Zha `[一作]` (AgiBot), Maoqing Yao `[通讯]` (AgiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种共享腰部的双链弹性几何校准方法，用于有摆臂的轮式类人机器人。

**💡 创新点**

创新点包括：① 在同一模型中联合估计两个臂与共享腰部参数；② 利用重力驱动的腰部刚度并加入轴向与倾斜弹性；③ 采用两阶段分离外部标定与内部参数识别；④ 通过证据最大化的高斯先验稳定估计；⑤ 对单链、顺序与双链方法进行可辨识性分析。

**🔧 技术方法**

技术手段：MDH完整运动学参数化；弹性关节模型（轴向与倾斜刚度）；光学跟踪标记测量；非线性最小二乘/贝叶斯MAP估计带先验；信息矩阵和Schur补分析；实验评估与文献方法对比。

**📊 数据集**

实验数据集：两台AGIBOT机器人（G2、G2max），分别在无负载与5/10 kg负载条件下采集约80–100个校准姿态，划分为训练集与验证集。

**📈 对比分析**

比较方法：与独立、顺序单链校准以及文献中的BON、NGU方法对比。双链弹性校准在验证姿态上的平均误差下降约1.5–2.5 mm，提升约10–20%，且在不同训练集规模下表现更稳健。

**⚠️ 局限性**

局限性：仅在单次测量会话内评估；未验证多会话重复性或任务级双臂定位效果；腰部负载模型基于CAD点质量，未实验验证；倾斜刚度模型为简化模型；对棱柱腰部未做实验；外部标定误差未完全消除。

---

## 278. Goal-Branching LaCAM: Temporary Goals as Search Branches in Dense Multi-Agent Pathfinding

**arXiv ID:** 2611.07528 | [PDF](https://arxiv.org/pdf/2611.07528v1)

**作者:** Stephen Chang `[一作]` (Monash University), Daniel Harabor `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

扩展了基于 PIBT 的 LaCAM2，在遇到 jam 时为阻塞代理生成临时目标，并把每个临时目标作为搜索分支，显著提升了在密集、死角多的 MAPF 任务中的求解率和速度。

**💡 创新点**

创新点是把临时目标生成与完整的配置搜索结合：在 jam 触发时即时生成若干临时目标分支，尝试多种解除 jam 的方案，并证明该扩展在任何临时目标选择规则下仍保持完整性和最终最优性；同时提供随机和走廊两种临时目标选择规则。

**🔧 技术方法**

主要技术包括：基于 LaCAM2 的深度优先配置搜索框架；使用 PIBT 低层配置生成；临时目标机制（包括年龄、持久限制、释放规则）；生成分支（随机和走廊规则），以及与高层搜索的协同执行。

**📊 数据集**

使用 MovingAI benchmark 的 32×32 地图（maze-32-32-2、maze-32-32-4、room-32-32-4 等）和 128×128 地图进行密集实例实验；随机生成 60%–90% 负载的实例，并在整个 MovingAI benchmark（33 图）上进行完整测试。

**📈 对比分析**

与 LaCAM2、LaCAM3 等基线在 30 秒/5GB 资源限制下进行比较。结果显示：在密集地图上成功率提升至 89%（对比 44%/33%），首次解时间平均提升 13.8×，成本略高 15–22%；在开放地图几乎无性能差异；在完整 benchmark 上不降低成功率，速度与 LaCAM2 相当，成本略高但不超过 3%。

**⚠️ 局限性**

局限性包括：临时目标分支往往导致更高的解决成本；对非密集或无死角实例提升有限；对临时目标选择规则（随机/走廊）仍需更智能的生成策略以进一步降低成本；依赖完整搜索框架，若去掉高层搜索则缺乏完整性保障。

---

## 279. RLGPA: A Reinforcement Learning-Guided Granularity Planning Attack for Decision-Based Sparse Adversarial Examples

**arXiv ID:** 2611.07512 | [PDF](https://arxiv.org/pdf/2611.07512v1)

**作者:** Chao Zhou `[一作]` (Huzhou College), Fengchuang Xing `[通讯]` (Guangdong University of Education)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于强化学习的在线粒度规划攻击（RLGPA），在决策式黑盒稀疏对抗攻击中动态调度支持恢复执行器与粒度；

**💡 创新点**

通过将宏观动作抽象为可重用的“策略–尺度”对，避免对临时支持组赋值Q值，利用在线Dueling Double DQN学习状态条件化的调度策略；

**🔧 技术方法**

强化学习（Dueling Double DQN）、支持相对粒度分组、三种恢复策略（Primary、Safe、Memory）、动态重划分、硬标签查询反馈；

**📊 数据集**

在CIFAR‑10（ResNet18）以及ImageNet‑1k（ResNet50、ViT）上进行实验；

**📈 对比分析**

与Pointwise、SparseEvo等基线对比，RLGPA在稀疏度低（CIFAR‑10 0.01、ImageNet 0.005）时实现更高的攻击成功率（如CIFAR‑10目标攻击99.0%/0.0085稀疏度，ImageNet目标攻击88.5%/0.0034稀疏度），且在查询预算下的查询-稀疏度曲线优于所有比较方法；

**⚠️ 局限性**

对标记学习的规模和奖励设计仍未显著提升；在低预算或较宽松稀疏阈值下仍不如SparseEvo；学习的尺度选择与奖励形状对性能影响有限；

---

## 280. Binding Paraphrased Instructions to Encountered Objects in sLLM Agents under Partial Observability

**arXiv ID:** 2611.07599 | [PDF](https://arxiv.org/pdf/2611.07599v1)

**作者:** Chanyoung Kim `[一作]` (Soongsil University), Dahuin Jung `[通讯]` (Chung-Ang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ALFWorld-Para 评测基准并提出 BINDIntent，解决 RL‑finetuned 小型 LLM 代理在部分可观测环境下对同义词指令绑定不当导致的鲁棒性下降。

**💡 创新点**

提出一种无训练、基于概率的同义词绑定框架，利用词向量距离与 Weibull 分布估计未见候选者的持久性，显著提升指令鲁棒性。

**🔧 技术方法**

使用词嵌入相似度（ConceptNet Numberbatch）、Weibull 分布建模、概率绑定决策以及 RL‑finetuned sLLM 与 Prompted LLM 的对比实验。

**📊 数据集**

使用 ALFWorld-Para（基于 ALFWorld 的改写任务）和 ScienceWorld-Para（基于 ScienceWorld 的改写任务）作为评测数据集。

**📈 对比分析**

对比 Baseline、Greedy、Self‑clarification、BINDIntentSBERT 等方法，在 ALFWorld-Para 上 BINDIntent 在同义词改写场景提升 SR 17.5–49.1 pp，整体提升 7–20 pp，GiGPO‑7B SR 从 61.8% 提升至 77.4%，超过 24B Prompted LLM；在 ScienceWorld-Para 上亦提升 7–10 pp。

**⚠️ 局限性**

仅针对同义词改写的绑定失败，对行动轴改写（如 Question、Hint）缺乏针对性；不改进探索策略，无法处理探索行为差异导致的绑定问题；对更复杂改写类型尚未验证。

---

## 281. Sharp Limits of Graph-Based Nearest Neighbor Search on the Sphere

**arXiv ID:** 2611.07603 | [PDF](https://arxiv.org/pdf/2611.07603v1)

**作者:** Kejing Lu `[一作]` `[通讯]` (University of Yamanashi), Kejing Lu (University of Yamanashi)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了贪婪图搜索在从单位球面上的n个独立数据点出发时找到每个起始点的确切最近邻所需的度数。

**💡 创新点**

确定了在均匀数据情况下，当d与log n成比例时的显式亚线性度阈值，以及在密集区域的阈值2^(d/2)+o(d)。

**🔧 技术方法**

使用了贪婪搜索算法和几何概率估计。

**📊 数据集**

使用了从单位球面上独立抽取的随机数据集。

**📈 对比分析**

通过与现有图结构的比较，证明了在特定度数下，贪婪搜索能够以亚线性评估成本成功找到最近邻，并且在低于临界度的情况下，存在失败的条目。

**⚠️ 局限性**

在非均匀数据情况下，局部密度变化会改变所需的临界度，且在某些情况下，设计的图可能无法满足成功搜索的要求。

---

## 282. Align Before Aggregation: Basis-Consistent Federated LoRA under Heterogeneous Ranks

**arXiv ID:** 2611.07598 | [PDF](https://arxiv.org/pdf/2611.07598v1)

**作者:** Pengpeng Qiao `[一作]` (Institute of Science Tokyo), Masatoshi Yoshikawa `[通讯]` (Osaka Seikei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FedAbA框架，在异质rank的联邦LoRA中实现基准一致的聚合；

**💡 创新点**

创新点在于先对每个客户端的局部基准进行正交Procrustes对齐并重构为全局参考基准，再使用基准一致的重构矩阵进行聚合，同时引入一致性得分与数据量共同加权，并在聚合后做系数空间SVD刷新与gauge固定；

**🔧 技术方法**

主要技术包括LoRA、QR分解、正交Procrustes、一致性得分计算、加权聚合、系数空间SVD、gauge修正；

**📊 数据集**

实验数据集涵盖GLUE四项NLP任务（MRPC、CoLA、QNLI、RTE）、Dolly‑15K、Alpaca、WizardLM、MMLU、MT‑Bench、Natural Instructions v2；

**📈 对比分析**

与HetLoRA、FLoRA、LoRA‑A²、FlexLoRA、ECLoRA、Fed‑PLoRA等六种异质rank LoRA基线进行对比，FedAbA在所有任务上均取得更高的准确率/ROUGE/MMLU，尤其在高rank异质度下提升显著；

**⚠️ 局限性**

局限性包括需要warm‑start阶段、通信成本相对较高（与纯因子聚合方案相比）、仅验证在注意力投影层的LoRA插入，其他类型适配器或更大规模模型的适用性待进一步研究。

---

## 283. Fine-Tuning a Transducer for Impaired Speech: Where the Gains Appear and What They Mean for Evaluation

**arXiv ID:** 2611.07587 | [PDF](https://arxiv.org/pdf/2611.07587v1)

**作者:** Kien Nguyen `[一作]` (Elokly), Bao Phan `[通讯]` (Elokly)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Speech Accessibility Project Challenge 2 上，作者对公开的语音识别模型进行微调并平均化，随后在 SAP 语料库上评估其性能。

**💡 创新点**

创新点在于通过模型平均化显著降低误差，系统性分析熟悉度与严重程度对识别误差的影响，并证明在普通 CPU 上即可实现实时低延迟。

**🔧 技术方法**

技术包括 Parakeet TDT 1.1B 作为基础模型，Whisper large‑v3 作为基线，NeMo 进行微调，模型平均化，以及 CPU 端的实时推理。

**📊 数据集**

使用的数据集为 SAP 语料库约 950 小时，覆盖 1,100+ 说话人，涵盖 ALS、帕金森、脑瘫、唐氏综合症与卒中等多种疾病。

**📈 对比分析**

评估方法为与官方基线、原始模型和 Whisper 进行对比；在 Test1 隐藏测试集上取得 7.81% CER（比官方基线下降 48%），实时条目延迟仅 0.85–0.93 秒，准确率与离线模型相近。

**⚠️ 局限性**

局限性包括测试说话人仅 41 人（四名卒中患者）、严重程度评估非临床、熟悉度实验可能高估效果、句子/说话人重叠导致的潜在偏差，以及未能对第二隐藏测试集进行分析。

---

## 284. Cores of $3$-distant convex games

**arXiv ID:** 2611.07572 | [PDF](https://arxiv.org/pdf/2611.07572v1)

**作者:** Keisuke Bando `[一作]` (Keio University), Ryuhei Mizutani `[通讯]` (Tokyo University of Agriculture and Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了3-远凸合作博弈的核心，提出了核心非空的必要和充分条件，并开发了一个强多项式时间的组合算法来寻找核心分配。

**💡 创新点**

提出了3-远凸博弈的核心非空条件，简化了经典的核心非空条件，并展示了该框架在经济竞争和联合生产中的应用。

**🔧 技术方法**

使用了组合算法和线性规划的经典特征来证明核心非空性，并开发了一个新的算法来寻找核心分配。

**📊 数据集**

使用了多种经济模型作为数据集，包括寡头竞争模型和生产游戏模型，展示了3-远凸性在这些模型中的适用性。

**📈 对比分析**

与传统的凸博弈方法进行了比较，证明了在某些情况下3-远凸博弈的核心可能为空，但在满足特定条件时仍然可以找到核心分配。

**⚠️ 局限性**

限制在于3-远凸博弈的核心可能为空，且在更高维度的k-远凸博弈中，核心的性质和计算方法可能不再适用。

---

## 285. OpenRDMA: A GPU-Native Multi-Path Reliable RDMA Transport Architecture

**arXiv ID:** 2611.07559 | [PDF](https://arxiv.org/pdf/2611.07559v1)

**作者:** Ziheng Ji `[一作]` (The University of Hong Kong), Hai Zhang `[通讯]` (Pazhou Laboratory (Huangpu))

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 OpenRDMA，一套面向 GPU 的 RDMA 通信体系，利用 FPGA 将 WQE 组装卸载给 NIC，GPU 仅通过 8 字节门铃提交意图，NIC 负责完整请求、跨路径报文喷射、可靠传输与按需重传，并在数据已正确写入 GPU 内存后发布可见完成标记。

**💡 创新点**

创新点包括：① GPU 友好极简门铃机制，将重 WQE 组装完全移至 FPGA；② 基于 MRC 的多路径乱序可靠传输协议，通过按位 Bitmap 记录到达状态，使用 ACK/SACK/NAK 进行选择性重传；③ 硬件‑软件协同的重传管理，CPU 只处理异常重传，显著降低 NIC 状态开销；④ GPU 可见的有序完成策略，将报文到达、写入与 GPU 读取同步分离。

**🔧 技术方法**

实现技术涵盖：FPGA 逻辑实现门铃解析、Slot Table、PSN 生成、报文组装、SRv6 多路径调度；PCIe DMA 读写 GPU 内存；FTile 100GE 四通道物理接口；基于 Bitmap 的可靠性状态管理；硬件‑软件协同的重传控制（SACK 解析、Timer、Scoreboard）；GPU 可见的 Flag 发布与有序队列管理。

**📊 数据集**

评测采用合成与 ns‑3 网络模拟，使用 Verilator 进行 RTL–网络共模仿真；实验平台为四路 100 Gb/s FTile 互连，四个 QP，报文尺寸从 1 KiB 到 16 KiB；不使用真实训练/推理数据集，而是通过可控的合成负载和不同路径延迟/丢包模式进行评估。

**📈 对比分析**

比较方法：将 OpenRDMA 与传统 CPU 代理 RDMA、IBGDA（GPU 直接异步）以及 GDR 进行对比；指标包括：单 QP 传输速率提升 3.51×、总聚合负载良好率 380.9 Gb/s（占 400 Gb/s 的 95.2%），单包丢失恢复时 Drop‑to‑CQE 延迟 508–598 ns（多包丢失 620–682 ns），多路径喷射显著降低平均网络完成时间，提供了高吞吐、低延迟的 GPU‑友好 RDMA 实现。

**⚠️ 局限性**

局限性：需要 FPGA 资源来实现门铃解析和可靠性状态，受限于硬件状态容量；多路径喷射虽然提升吞吐但在路径偏移时可能导致接收窗口压力和跨 QP 的头阻塞；重传仍依赖 CPU 侧的周期性轮询，无法在极高丢包率下保证实时性；实验目前基于模拟，尚缺乏物理 GPU/PCIe 现场验证，且对极端流量、长链路时延和多 QP 并发的性能尚未完整评估。

---

## 286. Training-Free Predictive Spatio-Temporal Token Pruning for Few-Step Autoregressive Video Diffusion

**arXiv ID:** 2611.07551 | [PDF](https://arxiv.org/pdf/2611.07551v1)

**作者:** Jia Li `[一作]` (University of Hong Kong), Hayden Kwok-Hay So `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练、在线预测的时空令牌裁剪框架PreSTP，用于加速少步自回归视频扩散模型的生成

**💡 创新点**

通过利用模型自身的清晰估计在线识别跨帧静态冗余与跨步收敛冗余，并将两种冗余统一成令牌掩码实现令牌级别的计算裁剪，解决了少步模型中传统步维缓存方法效率低下的问题

**🔧 技术方法**

基于清晰估计的余弦相似度判断、跨帧静态掩码与跨步冻结掩码的动态更新、内容自适应掩码复用以及稀疏Transformer执行

**📊 数据集**

在VBench（5s短视频）和VBench-Long（30s长视频）评测集上进行实验，并使用MovieGen等公开提示集进行定量与定性比较

**📈 对比分析**

与TeaCache和FlowCache等训练‑free缓存基线对比，PreSTP在5s任务中实现1.52×–3.20×的速度提升，VBench分数保持在0.07以内；在30s任务中实现1.96×速度提升，且色彩漂移和VBench总分显著优于基线；在单步模型中仍可获得1.10×–1.22×加速并提升长序列质量

**⚠️ 局限性**

对高动态场景的裁剪效果有限，裁剪率随内容运动而下降；需要在首次帧做完整稠密推理，导致少量初始开销；目前只关注令牌级裁剪，未结合上下文压缩或联合优化

---

## 287. CST-Bench: Benchmarking Counterfactual Spatial Transition Reasoning with Causal Transition On-Policy Self-Distillation

**arXiv ID:** 2611.07541 | [PDF](https://arxiv.org/pdf/2611.07541v1)

**作者:** Ying Li `[一作]` (North China University of Technology), Yawei Luo `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于评估多模态大语言模型在空间推理中对因果干预的理解能力的基准CST-Bench，并在此基准上开发了一种新的自监督蒸馏方法CT-OPSD，以提升模型在对抗干预场景下的推理准确性。

**💡 创新点**

创新点包括：①首次系统化地将空间推理任务扩展到因果干预层面，构建了6类干预与12个细粒度转移算子的完整测试集；②提供了丰富的因果转移注释（干预源、受影响关系、预期效果、不变关系、度量变化），为模型提供结构化的监督信号；③设计了基于教师模型的因果状态转移自监督蒸馏框架CT-OPSD，利用这些注释训练教师并通过 on‑policy 对齐将推理行为迁移到仅观测图像和问题的学生模型。

**🔧 技术方法**

技术主要包括：多模态大语言模型（MLLM）微调与评估；基于符号求解的干预样本生成与校验；答案无关的 VLM 一致性验证；以及 on‑policy 自蒸馏（CT-OPSD）结合 KL 对齐和答案监督。

**📊 数据集**

使用了7,082张自然场景图像组成的CST-Bench（含6类干预、12类算子），并在其上评估了10个开源MLLM；此外，还将训练后的模型在3DSRBench、CV-Bench和BLINK_val等标准空间推理基准上进行跨任务验证。

**📈 对比分析**

对比方法包括原始模型、答案监督SFT、GRPO、OPS​D以及CT-OPSD。CT-OPSD在CST‑Bench平衡评估集上从基准模型的17.0%微平均提升至55.2%，并在CV‑Bench、BLINK_val上取得了最高的平均准确率；在3DSRBench上虽不如GRPO/OPSD但已明显优于未训练模型，说明其提升了因果空间推理而不损失其他空间任务的表现。

**⚠️ 局限性**

局限性：①CST‑Bench仅覆盖了六类干预与十二个算子，尚未涵盖更复杂的多步或动态干预；②CT‑OPSD仍无法在所有干预类型上实现高精度，尤其是深度与视角反转仍表现较弱；③依赖教师的高质量因果注释，对数据生成与标注成本提出挑战，未来需探索更高效或无监督的因果结构学习方法。

---

## 288. Embodied RAG++: Retrieval-Augmented Generation From What Exists to What Works in Embodied Scenes

**arXiv ID:** 2611.07517 | [PDF](https://arxiv.org/pdf/2611.07517v1)

**作者:** Fan Yang `[一作]` (Fujitsu), Yuushi Toyoda `[通讯]` (Fujitsu)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了E2W-Bench基准和Embodied RAG++方法，用于在动态混合场景图和产品手册之间进行能力驱动的机器人-对象匹配。

**💡 创新点**

通过产品手册链接的能力卡缓存、检索范围限定与维度拆分，以及无LLM的确定性空间推理，实现高效准确的机器人-对象配对。

**🔧 技术方法**

使用RAG、图检索、层次可导航小世界图、LLM（Qwen2.5-32B、Llama3.3-70B、Gemini-3.5-Flash）和确定性空间推理。

**📊 数据集**

E2W-Bench包含20个动态混合场景图、770页真实产品手册，涵盖10种产品类别和3种机器人。

**📈 对比分析**

使用Naive LLM、GraphRAG、LightRAG、EmbodiedRAG等基线进行对比，测量pair准确率、预处理时间、查询时间和总token消耗；Embodied RAG++在pair准确率上显著优于基线，且预处理时间最短、token消耗最低。

**⚠️ 局限性**

仅在仿真中验证，假设已知对象-手册链接，类别和原型有限，缺乏对真实感知误差与定位噪声的评估。

---

## 289. Training-Independent Validation of a Learning-Based Controller for Stunt Maneuvers

**arXiv ID:** 2611.07516 | [PDF](https://arxiv.org/pdf/2611.07516v1)

**作者:** Kiruthiga Chandra Shekar `[一作]` (General Autonomy), Jingang Yi `[通讯]` (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套独立于训练的验证流程，用来评估学习驱动的车辆转弯（J-turn）控制策略，并通过模拟、现实以及再次模拟的循环验证其可行性。

**💡 创新点**

创新点在于（1）将验证与训练解耦，基于安全标准将物理风险映射为可测试参数；（2）提出适应性覆盖搜索，显著减少仿真试验数量；（3）引入Sim2Real2Sim闭环，验证仿真指标在真实车辆上的可迁移性。

**🔧 技术方法**

采用的技术包括强化学习、基于规划的运动规划、少量样本学习（G-Maneuver）以及大语言模型的策略修正；测试使用CARLA高保真仿真、Mini Cooper S 真实平台；安全分析使用ISO 26262、ISO 21448 与 STPA。

**📊 数据集**

数据集主要为CARLA仿真生成的1152个测试点，涵盖天气、等待时间、速度阈值等六维参数；真实实验采集100 Hz惯性与定位数据。

**📈 对比分析**

与先前报告的90.11%成功率对比，验证后成功率降至16.9%，且通过适应性覆盖仅需631次仿真；Sim2Real2Sim显示航向变化与侧滑等几何指标良好迁移，但侧向加速度和滚转指标存在显著误差。

**⚠️ 局限性**

限制包括：仅考虑平面单轨模型，未包含悬挂、横向转弯等动力学细节；实验仅在单辆车与无障碍环境中进行；天气与初始状态的交叉影响未完全分离；缺乏感知与其他车辆交互的验证。

---

## 290. PhySA: Decoupling Aesthetic and Physical Rewards for Humanoid Locomotion

**arXiv ID:** 2611.07679 | [PDF](https://arxiv.org/pdf/2611.07679v1)

**作者:** Yunwan Fu `[一作]` (Shanghai Jiao Tong University), Yue Wang `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一个将视觉语言模型审美评分与物理平滑纠正分离的奖励框架（PhySA），并在 MuJoCo Humanoid‑v5 上训练出更平滑、低关节加速度的步态；

**💡 创新点**

创新点在于：①把审美评分与物理平滑放入独立、可切换的奖励通道；②采用异步 VLM 调用与相位触发的奖励编辑；③证明 VLM 评分与平滑方向相反，主要的平滑提升来自纠正通道；

**🔧 技术方法**

使用技术包括 PPO、GPT‑4o 异步 VLM 评分、EMA 平滑、相位触发奖励编辑、MuJoCo/IsaacLab/Unitree G1 仿真；

**📊 数据集**

数据来源为 MuJoCo Humanoid‑v5 生成的轨迹，VLM 评估使用 GPT‑4o、Claude、Gemini 等四种生产 VLM；无公开专门数据集；

**📈 对比分析**

与 Prompt2Policy、PPO、Text2Reward、Eureka 等基线在 MuJoCo Humanoid‑v5 上比较，单跑关节加速度降低 59.5%，三种种子平均 7.2%（不显著）；相较 Prompt2Policy 机械功率降低 43%、期数延长 57%；跨模拟器/机器人转移效果不佳；

**⚠️ 局限性**

局限性包括：单跑效果显著但多种种子不显著；VLM 通道对平滑无益且仅影响存活奖励；跨模拟器/机器人转移失败；VLM 评分只能在每 200k 步异步更新，缺乏实时反馈；

---

## 291. Who Should Know What? Empirical Information Design for Multi-Agent LLM Economies

**arXiv ID:** 2611.07667 | [PDF](https://arxiv.org/pdf/2611.07667v1)

**作者:** Mete Erdogan `[一作]` (Stanford University), James Zou `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究多智能体LLM系统在不同信息披露策略下的经济表现，通过实验比较多资源贸易、双边议价和双拍卖三种市场模式，揭示信息精细化对总福利与收益分配的双重影响。

**💡 创新点**

首次提出“经验信息设计”框架，在保持模型、市场和交互规则不变的前提下系统评估信息披露对多智能体系统的影响，强调模型选择与信息设计需协同优化。

**🔧 技术方法**

基于10种前沿LLM（GPT、Claude、Gemini、Grok等）进行对话生成，采用自定义的经济游戏规则、消息交换协议以及归一化的效用评估指标（总收益、Nash得分等）。

**📊 数据集**

使用人工构造的经济市场实例（90个多资源市场、15个议价市场、30个双拍卖市场），每个市场包含完整的真实效用向量，未使用公开的真实交易数据。

**📈 对比分析**

通过对比粗略信息、细化排名和完全信息三种披露条件下的候选质量、实现绩效与差距，发现不同模型对信息的响应差异显著；在多资源贸易中部分模型在全信息下收益下降，单价议价中信息增多可提高达成率但削弱分配平衡，双拍卖中全信息往往导致交易减少，整体性能下降。

**⚠️ 局限性**

局限在于实验环境为人工设计的静态游戏，未涵盖真实世界的不确定偏好、动态变化及更复杂交互；数据集有限，缺乏跨领域验证；结果可能受模型版本与提示细节影响，需要进一步在真实部署情境中验证。

---

## 292. CoGenRec: A Unified Generative Framework for Complementary Vision Reconstruction

**arXiv ID:** 2611.07665 | [PDF](https://arxiv.org/pdf/2611.07665v1)

**作者:** Yapeng Meng `[一作]` (Tsinghua University), Rong Zhao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CoGenRec，一种基于补偿式视觉传感器（CVS）多模态测量的统一生成式重建框架，集成去模糊、帧插值和 HDR 合成；

**💡 创新点**

创新点在于：①利用 CVS 的空间-时间采样、快慢运动分离及曝光互补三种互补特性构建多尺度引导特征；②将物理测量约束注入预训练的视频扩散模型（ControlNet），实现生成与测量信息协同；③引入细节保留 VAE 解码器，缓解 latent 空间压缩导致的细节损失；

**🔧 技术方法**

核心技术包括多尺度时空引导提取器（曝光感知端点编码器+双向 ConvGRU）、多尺度 ControlNet 对接视频扩散 UNet、以及细节保留 VAE 解码器；

**📊 数据集**

训练数据为约一百万条混合数据，来源于 DLP 硬件转换的 SportsSloMo‑CVS、Ego4D‑CVS、OpenVid‑1M‑CVS 以及软件仿真生成的 OpenVid‑1M‑CVS，测试集包含 129 条真实 CVS 场景；

**📈 对比分析**

与 Tianmouc‑UNet、IGFNet、CBRDM、STGDNet 等基线对比，CoGenRec 在去模糊、帧插值、HDR 合成任务上分别提升 PSNR 4.40 dB、3.89 dB、5.04 dB，且在感知指标、NIQE、FID、TR 等方面均显著优于所有对手；

**⚠️ 局限性**

局限性包括对大规模混合数据与预训练扩散模型的高依赖，推理计算量较大；对极端曝光或完全无监督场景的鲁棒性仍待进一步验证；

---

## 293. Beyond Instruction Override: Benchmarking Indirect Prompt Injection with Task-Tampering Attacks

**arXiv ID:** 2611.07640 | [PDF](https://arxiv.org/pdf/2611.07640v1)

**作者:** Rongting Zhang `[一作]` (Amazon Responsible AI), Vanessa Murdock `[通讯]` (Amazon Responsible AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于权限边界的间接提示注入（IPI）评估框架，并构建了90个经过验证的任务篡改案例；

**💡 创新点**

创新点在于将攻击定义为来自未获授权源的决策干预，而非仅靠指令式措辞；并通过影响原语（目标、规则、后果、动作、观察、状态）细化攻击方式；

**🔧 技术方法**

采用了LLM辅助的案例生成、手工验证以及多种防御技术（提示隔离、计划检查、工具/参数门控、内容检测与移除等）；

**📊 数据集**

使用了自制的多环境任务（生产力、DevOps、市场）数据集，总计90个案例、1343个单原语攻击；

**📈 对比分析**

对五大类防御（提示屏蔽、隔离、计划/参数检查、工具级门控、检测优化）进行评估，结果显示隔离+检测能将目标攻击成功率降至2.4%，但对状态/观察类攻击仍高达12%；

**⚠️ 局限性**

局限性包括：评估仅涵盖模拟环境，未测试真实部署；部分防御在面对工具必要的攻击时效果有限；且检测器对非指令式攻击识别率低。

---

## 294. Redacting Personal Data: A Speed and Accuracy Comparison of Regex, Presidio and Hybrid Designs

**arXiv ID:** 2611.07633 | [PDF](https://arxiv.org/pdf/2611.07633v1)

**作者:** Sai Yashwant `[一作]` (ManpowerGroup Services India Pvt. Ltd.), Shubham `[通讯]` (ManpowerGroup Services India Pvt. Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了四种在LLM调用前对文本进行PII红局的设计，包括纯正则、Presidio、混合和快速混合方案。

**💡 创新点**

创新点在于将红局的准确性、延迟与LLM调用延迟对齐，提出只用spaCy NER的快速混合方案实现高召回与低延迟，并给出量化的比较。

**🔧 技术方法**

使用了正则表达式、Microsoft Presidio（spaCy NER）以及自定义的合并与占位符逻辑。

**📊 数据集**

使用100份合成简历（约523字符/文）作为测试语料，包含900个PII值和600个业务术语。

**📈 对比分析**

通过准确率（PII漏检率和业务术语保留率）以及中位延迟、95%分位延迟进行比较，结果显示快速混合在保证100%召回的同时，中位延迟仅13.48 ms，比Presidio慢约30%，比纯正则慢两百倍但召回率从88.9%提升至100%。

**⚠️ 局限性**

局限性包括仅针对合成且长度有限的美国简历，未覆盖不同语言或更长文本，模型和正则在真实数据中召回可能下降，实验环境单一，无法完全反映云函数冷启动或多进程负载。

---

## 295. TRACEMARK: Editor-Aware Watermarking for User Attribution in Text-Guided Image Editing

**arXiv ID:** 2611.07632 | [PDF](https://arxiv.org/pdf/2611.07632v1)

**作者:** Xiaodong Wu `[一作]` (Queen's University), Rongxing Lu `[通讯]` (Queen's University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种编辑器感知的水印框架，能在文本引导的图像编辑后仍识别请求编辑的用户。

**💡 创新点**

创新点在于：将水印编码器、检测器和用户参考向量联合训练，并通过固定的编辑器传递梯度；同时兼容两种识别方式（原型匹配和二进制解码），显著提升在多种编辑条件下的归属准确率。

**🔧 技术方法**

使用深度学习技术：共享卷积编码器/检测器网络、用户嵌入表、原型向量；训练目标包括载体/编辑保真、嵌入一致性和跨编辑用户识别损失；编辑器为InstructPix2Pix、sd‑turbo、ChordEdit等预训练模型。

**📊 数据集**

使用 CelebA‑HQ、AFHQ‑Dog、CUB‑200‑2011 三个高质量图像数据集，涵盖 30 个编辑特征，最多支持 500 名注册用户。

**📈 对比分析**

与 TrustMark、RoSteALS、VINE‑R、Robust‑Wide 等公开水印方法进行对比；在 256 像素载体下，匹配编辑器训练后平均准确率可达 87.5%（500 用户），在未见编辑器上表现提升 1.9‑5.0 倍；在 512 像素下，部分公开方法仍保持竞争力；性能随编辑器兼容性、图像分辨率和用户规模变化。

**⚠️ 局限性**

局限性包括：评估仅基于训练集图像，未验证对未见图像的泛化；只考虑单一编辑器的固定权重，无法处理多阶段或攻击性去除；对未嵌入水印的图像识别率接近随机；需要在具体服务中额外设计拒绝未标记图像的规则。

---

## 296. Real-Time Event-Based LED Tracking and Pose Estimation with PWM-Encoded Markers

**arXiv ID:** 2611.07630 | [PDF](https://arxiv.org/pdf/2611.07630v1)

**作者:** Junkai Niu `[一作]` (Hunan University), Yi Zhou `[通讯]` (Hunan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种实时事件相机下基于PWM占空比编码LED标记的跟踪与6-DoF姿态估计框架；

**💡 创新点**

创新点在于：①使用PWM占空比区分多LED共享同一闪烁频率，降低频率资源与计算成本；②设计双速异步跟踪框架，将周期性检测与事件级更新相结合，提升跟踪鲁棒性与实时性；

**🔧 技术方法**

采用了事件相机、PWM占空比编码、频率加权置信图检测、四状态卡尔曼滤波、PnP姿态估计、RANSAC、以及在NVIDIA Jetson Orin NX上的实时部署；

**📊 数据集**

使用自采集的室内数据集：1280×720 Prophesee EVK4事件相机，四LED占空比为30%、45%、60%、75%，共1kHz闪烁频率，并配合NOKOV光学运动捕捉系统提供真值；

**📈 对比分析**

与HASTE、jAER、AEB-Tracker等异步事件跟踪器比较，平均LED跟踪RMSE为12.13像素，6-DoF平均ATE为15.52 cm；在无人机悬停与跟随实验中，平均轨迹误差在2–15 cm之间；计算效率约为每个事件15 ns；

**⚠️ 局限性**

局限性包括：对极端光照变化仍有误差；需要预先设定占空比集合，扩展到更大LED数时需重新设计；在LED重叠或遮挡严重时仍可能出现跟踪或识别错误；多光源/密集LED场景的性能尚未充分验证。

---

## 297. Towards Real-Time Full-Waveform LiDAR Transformers via Intensity-Guided Token Reduction and Physics-Aware Augmentation

**arXiv ID:** 2611.07613 | [PDF](https://arxiv.org/pdf/2611.07613v1)

**作者:** Kotaro Oishi `[一作]` (Keio University), Kentaro Yoshioka `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了两种方法，分别为 FWL-ToPM 和 FWLAug，用于提升全波形 LiDAR Transformer 的实时性与泛化性能。

**💡 创新点**

创新点在于利用 FWL 的强度稀疏性实现 token 的按强度剪枝与空间结构感知合并，并开发了保持 ToF 波形结构与视锥几何的专属数据增强框架。

**🔧 技术方法**

使用的技术包括 Vision Transformer、基于强度的 token 剪枝、空间填充曲线序列化后的 token 合并、FWLShiftMix 与 FWLCutMix 两种增强方式，以及 focal loss 训练策略。

**📊 数据集**

实验数据集为 Ghost-FWL 以及构造的 OOD 测试集，后者在完全不同位置采集但采用相同传感器与标注协议。

**📈 对比分析**

与现有 FWL 识别方法和基于图像的 token 减少方法相比，FWL-ToPM 在 Ghost-FWL 上实现 0.592 的 F1-mean 并达 53.2 FPS，速度提升约 8 倍，F1 提升约 9 分；FWLAug 在 OOD 集上进一步提升了 F1 及 Glass 类别性能。

**⚠️ 局限性**

局限性包括可能误删弱信号导致漏检、仅在单一传感器平台验证、增强方法未完全模拟光学传播、以及缺乏完整系统级延迟评估。

---

## 298. A Gingerbread House for LLM Agents: Source-Planned Honeypots That Lure Autonomous Attackers into an Isolated Shadow

**arXiv ID:** 2611.07611 | [PDF](https://arxiv.org/pdf/2611.07611v1)

**作者:** Seungdeok Roh `[一作]` (Soongsil University), Daeseon Choi `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM的静态蜜罐规划与路由策略，用于在LLM攻击者利用Web漏洞前将攻击流量重定向到隔离副本，从而显著降低真实目标被攻破的风险

**💡 创新点**

创新点在于将源代码级的LLM生成静态欺骗计划与按客户端状态路由到阴影副本相结合，且评估了在真实LLM攻击框架（ARTEX）和大语言模型（Claude Opus 4.8）下的效果，首次展示了工作攻击被转移而非被阻断的证据

**🔧 技术方法**

主要技术包括：LLM（Claude Opus 4.8）进行防御计划生成，LLM（DeepSeek-V4-Flash）驱动的多代理攻击框架ARTEX，容器化的隔离副本，路由器的单客户端状态机，静态欺骗规则与可选的运行时LLM响应器，自动化评估脚本和Cheating Scanner

**📊 数据集**

使用了CVE-Bench 40个关键Web CVE任务（2023-2024）和自构建的alt-cve-bench 20个新任务，均采用相同的评估器和任务格式

**📈 对比分析**

通过对比三种配置（无防御、静态蜜罐+路由、增补响应器+额外规则）进行实验，评估指标包括真实目标被攻破、阴影副本被攻破、假秘密被报告、确认时间等；结果显示静态计划将真实攻破率从26/40降至5/40（Claude）或0/40（ARTEX），阴影副本成功率在一定范围内保持，增补配置未显著提升性能

**⚠️ 局限性**

主要限制包括：仅测试单个保留运行，未评估多次运行方差；缺乏仅检测或拦截基线；在静态计划已达到零时无法区分新增组件效果；路由器基于单监听器的状态，可能无法处理身份切换；攻击者可利用共享内存泄漏或反向识别阴影副本；实验仅覆盖单服务容器，未涵盖多服务或WAF环境

---

## 299. The Gating Mechanism of KV Cache Eviction

**arXiv ID:** 2611.07605 | [PDF](https://arxiv.org/pdf/2611.07605v1)

**作者:** Hao Gu `[一作]`, Sirui Han `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的 KV 缓存淘汰与恢复机制 IndexMem++，通过学习门控重排注意力实现高效长上下文推理。

**💡 创新点**

创新点在于将淘汰与恢复统一在同一 softmax 归一化中，并引入内容聚类的质心 KV 来补偿被淘汰信息。

**🔧 技术方法**

采用可学习的门控网络、蒸馏监督、按 head 动态分配预算以及聚类质心恢复。

**📊 数据集**

使用公开模型 Qwen3-4B/8B，评测 RULER、LongBench、Math500 等长文本基准。

**📈 对比分析**

与全 KV、TrimKV、FastKVzip 等基线比较，在 75% 压缩率下，RULER 90.14/LongBench 46.03，几乎匹配全 KV，且在 Math500 上保持 0.807 的 pass@1。

**⚠️ 局限性**

限制在于仍需手工调参的预算分配、对特定查询的恢复效果有限，且对超大模型的泛化尚未验证。

---

## 300. FisherDistill: Diffusion Distribution Distillation via Closed-Form Fisher Critics

**arXiv ID:** 2611.07604 | [PDF](https://arxiv.org/pdf/2611.07604v1)

**作者:** Mahesh Bhosale `[一作]` (University at Buffalo, State University of New York), Defang Chen `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FisherDistill，一种非对抗性后期细化方法，将一或少步扩散蒸馏生成器的残留分布差距通过闭式 Fisher 判别器进行修正；

**💡 创新点**

创新点在于完全不需要学习判别器，而是利用冻结教师网络的中间特征一、二阶统计量直接求解 Fisher 判别方向，提供轻量级的分布匹配修正信号；

**🔧 技术方法**

核心技术包括 Fisher 线性判别分析、分布匹配蒸馏损失、指数移动平均（EMA）在线统计、闭式 Fisher 方向计算与梯度反向传播；

**📊 数据集**

主要使用 ImageNet 64×64（图像）、COCO-2014、LAION-Aesthetics、Midjourney、Sana 等公开数据集进行训练与评估；

**📈 对比分析**

与原蒸馏基线、对抗性细化（DMD2、SiD²A 等）以及 RealUID 进行对比，FID、precision/recall、CLIP、Aesthetic、HPSv2 等指标均显示 FisherDistill 在保持或提升样本质量的同时，显著缩小与多步教师的分布差距，且在训练样本和时间上更高效；

**⚠️ 局限性**

局限性在于需要教师特征对真伪区分度足够强，层选择对效果影响大；依赖固定教师和预计算的统计，若教师或任务变化可能需重新估计；对极小残差的分布差距修正效果有限。

---

## 301. ReferClaw: Disambiguating User Intent for Egocentric Assistance

**arXiv ID:** 2611.07683 | [PDF](https://arxiv.org/pdf/2611.07683v1)

**作者:** Hangyu Qin `[一作]` (National University of Singapore), Angela Yao `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个三代理交互式框架，能在第一人称视频问答中主动提问澄清问题并基于澄清结果给出答案，解决用户意图模糊的难题。

**💡 创新点**

创新点包括：① 用可共享的空间‑时间候选解释集同时驱动澄清、选择与答案；② 通过成本感知的澄清策略和固定交互预算平衡信息增益与用户努力；③ 在交互过程中自适应地生成意图特定技能并利用过程奖励对各代理进行细粒度更新；④ 对多种视觉‑语言基准实现零样本跨域迁移。

**🔧 技术方法**

技术手段主要有：多模态冻结 backbone + LoRA 微调；三代理（Intention‑Grounding Agent、Cost‑Aware Clarification Agent、Evidence‑Fused Answering Agent）与确定性控制器；信息增益与成本建模的强化学习策略；MetaClaw‑style两阶段学习（快速技能生成 + 慢速策略优化）。

**📊 数据集**

使用了 BuddyVQA 与 EgoBlind（经意图、时间窗口与边框标注扩展）作为主要 egocentric 流式 QA 评测集，并在 ClearVQA、ContextClarify 上验证跨域通用性。

**📈 对比分析**

与现有的直接回答与交互式澄清基线相比，Base 方案在 BuddyVQA 上从 66.2% 提升至 69.0%（澄清启用），Full 进一步提升至 74.3%；在 EgoBlind 上准确率从 54.0% 提升至 60.7%；在 ClearVQA、ContextClarify 上分别从 41.5%→48.1% 与 42.9%→54.8%；同时用户平均提问率仅 18.6%，每次平均 0.22 题，成本和延迟明显低于传统澄清方法。

**⚠️ 局限性**

局限性：① 仍需用户或 LLM 回答澄清问题，若用户缺乏合作或出现错误可能导致误判；② 需要在每次交互中维护多代理和候选集，计算开销较大；③ 对极端视觉噪声或缺失信息的鲁棒性尚未完全验证；④ 目前仅针对 egocentric 视频及静态图像 QA，尚未证明能直接迁移至非第一人称或多摄像头场景。

---

## 302. Calibrated Target Noise Recovers Curvature from the Gradients

**arXiv ID:** 2611.07671 | [PDF](https://arxiv.org/pdf/2611.07671v1)

**作者:** Arash Jamshidi `[一作]` (University of Helsinki), Kai Puolamäki `[通讯]` (University of Helsinki)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用对目标加入方差校准的噪声，仅用批量梯度即可估计网络的二阶曲率（GGN矩阵）并在训练中作为预处理器或早停信号。

**💡 创新点**

创新点在于：①通过对目标加零均值噪声并按输出曲率校准其方差，使得噪声梯度的协方差在期望下正好等于GGN；②提出无偏的“中心”估计器和“溶解”估计器，并证明后者误差随批量大小与噪声乘子可控制；③证明缩放 Rademacher 噪声在所有满足方差约束的分布中能最小化估计器的噪声风险。

**🔧 技术方法**

技术主要是目标噪声注入、梯度协方差估计、无偏中心化与溶解、Rademacher 采样、基于估计的预处理（块对角或对角），以及与常规优化器（SGD、Adam、Sophia 等）和其他曲率估计方法（清洁梯度协方差、Hutchinson、GNB）的对比实验。

**📊 数据集**

实验使用了多种数据集：telescope（分类）、occupancy detection（时间序列分类）、air‑quality（回归）以及 Transformer 编码器上的相同任务；网络结构包括 MLP、CNN、1D CNN、Transformer encoder。

**📈 对比分析**

与基线方法相比，CNGC 的估计误差（相对算子范数）显著低于清洁梯度协方差和 Hutchinson，且计算成本与最优基线相当；在作为预处理器时，GD+PC 和 SophiaC 版本在大多数任务中能更快达到目标损失，并在大多数实验中获得最快或第二快的训练时间；在早停上，基于 CNGC 的停止信号更稳健，避免了早停过早导致欠拟合。

**⚠️ 局限性**

局限性包括：只恢复 GGN（除 GLM 和 ReLU 层对角 Hessian 块外不获取完整 Hessian）；中心化估计器需要额外一次正向/反向传播；需要批量中每个样本的输出和标签，且块对角或对角近似仍需 O(d^2) 存储，块大小过大时可导致 O(d^3) 的求逆成本。

---

## 303. English Calibration Erodes Lithuanian in Expert-Pruned Mixture-of-Experts Language Models

**arXiv ID:** 2611.07654 | [PDF](https://arxiv.org/pdf/2611.07654v1)

**作者:** Tomas Rasymas `[一作]`, Povilas Kvedaras `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在混合语言环境下对Mixture-of-Experts (MoE) 大模型进行专家裁剪时，英语校准数据对低资源语言立陶宛语的影响，并在四个开源MoE模型上系统评估了裁剪比例（25%与50%）与校准语言（英语、立陶宛语、混合）对阅读理解、翻译和语言建模性能的差异。

**💡 创新点**

创新点在于：①首次系统展示英语校准在立陶宛语上的显著负面迁移，并证明立陶宛语或混合校准能显著缓解损失且几乎不损害英语性能；②揭示语言特定专家和路由分布差异是导致性能不对称的关键因素；③提供完整的实验网格与实用部署建议，为多语言模型压缩提供新的参考。

**🔧 技术方法**

技术方法包括：使用 REAP（Router‑weighted Expert Activation Pruning）对四个MoE模型（Qwen3‑30B、Qwen3.6、Gemma 4、GLM‑4.7‑Flash）分别裁剪 25% 与 50% 的专家；在每个模型上以英语、立陶宛语或 50/50 混合校准集（来自 Wikipedia 与 Europarl）计算专家重要性；随后在两种语言上评估阅读理解（Belebele）、翻译（FLORES‑200）与语言模型（Wikipedia NLL），并结合路由统计分析专家利用情况。

**📊 数据集**

数据集包括：①英语与立陶宛语 Wikipedia（单语）用于校准与语言模型评估；②Europarl 双语并行语料作为校准与内容对齐的基准；③Belebele 多语言阅读理解问答集；④FLORES‑200 公开翻译测试集；⑤NLL 评估采用 5,000 篇保留 Wikipedia 文章。

**📈 对比分析**

比较方法是对每个模型、每种校准语言和裁剪比例，计算未裁剪与裁剪版本在两语言下的性能差距（Δ）。统计显著性通过 95% 对齐引导自举检验得到。结果显示：在 25% 裁剪下，英语校准导致立陶宛语阅读理解平均下降约 20–30 分，而立陶宛语或混合校准平均提升 20 分以上且英语性能几乎不变；50% 裁剪损失更为严重，所有校准语言均出现显著退化；翻译与 NLL 亦呈相同趋势。混合校准与立陶宛语校准性能相近，且在英语语言建模上略优。

**⚠️ 局限性**

局限性包括：①仅试验两种裁剪比例与固定 token 预算，未探索更细粒度或自适应裁剪；②仅评估四个模型，无法保证结果泛化至其他 MoE 架构；③评估集中在固定任务（阅读理解、翻译、NLL），未涵盖开放式生成、指令遵循等；④校准与评估使用的 token 数量不平衡（立陶宛语约为英语的一半），可能影响对比；⑤路由与专家重要性分析仅为相关性，未验证因果机制；⑥实验仅跑一次，缺少对校准样本多样性的验证。

---

## 304. Neural Image Compression with Internal-External Prior Fusion and Alignment

**arXiv ID:** 2611.07644 | [PDF](https://arxiv.org/pdf/2611.07644v1)

**作者:** Tiancheng Si `[一作]` (Xidian University), Dahua Gao `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种内部-外部先验融合对齐（IE-PFA）的上下文模型，利用可共享字典和RWKV注意力实现内部先验与外部先验的线性融合。

**💡 创新点**

创新点在于将共享可学习字典作为外部先验，并通过RSCPF模块与SDPA对齐，有效缓解两种先验的分布偏移，支持单模型多码率编码。

**🔧 技术方法**

采用RWKV的空间-通道融合模块、三阶段渐进训练、共享率调整模块、以及自定义CUDA算子等技术。

**📊 数据集**

使用ImageNet 400K训练，评估基准为Kodak、CLIC、Tecnick、HEVC B–E等七个数据集。

**📈 对比分析**

相较于VTM-22.0-Intra和多种先进神经编码器，平均BD-Rate降低21.65%，编码/解码最快（61/52 ms），参数仅29.9M，显著提升性能-复杂度平衡。

**⚠️ 局限性**

局限包括对字典大小的敏感性、对高分辨率/视频场景的通用性待验证，以及外部先验学习可能受训练数据分布偏差影响。

---

## 305. MEMOIR: A Training-Free Memory Lifecycle for Long-Horizon LLM Agents

**arXiv ID:** 2611.07631 | [PDF](https://arxiv.org/pdf/2611.07631v1)

**作者:** Mayank Anand `[一作]` (Indian Institute of Information Technology Allahabad), Sriparna Saha `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种端到端的长期记忆生命周期框架 MEMOIR，通过在写入时无 LLM 调用进行高效写入，并在检索后根据答案结果自适应更新记忆重要性；

**💡 创新点**

创新点在于将前向合并（证书认证的合并）与后向归因（TRACE-UCB）相结合，实现训练自由、低延迟的记忆更新与安全压缩；

**🔧 技术方法**

使用了证书认证的合并（MERGE）与逆向归因算法 TRACE（包括文本重叠和留一对比法）以及 UCB 探索策略，配合 RECALL‑Q 复合检索评分；

**📊 数据集**

在 LoCoMo（约1,905 条多会话问答）和 LongMemEval‑S（500 条单问答）两个长期记忆基准上进行实验；

**📈 对比分析**

与现有记忆系统（Mem0、LiCoMemory 等）对比，MEMOIR 在 LoCoMo 上 Recall@5 提升 11.92 点、准确率 58.70%，在 LongMemEval‑S 上准确率 81.20%、Recall 85.10%，且保持了 0.2k token 和 1.18s 的低延迟；

**⚠️ 局限性**

局限性包括需在共享存储上多次查询才能收集反馈，无法改善冷启动查询；对非文本同义词的归因精度有限，Counterfactual TRACE 计算延迟高；证书认证的安全门槛虽有效但并非正式保证，实际存储压缩率有限。

---

## 306. Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone

**arXiv ID:** 2611.07610 | [PDF](https://arxiv.org/pdf/2611.07610v1)

**作者:** Shivesh Rane `[一作]` `[通讯]` (Goa College of Engineering), Shivesh Rane (Goa College of Engineering)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在仅凭LoRA微调模型本身的权重、激活或输出上进行训练域推断，提出白盒、灰盒、黑盒特征。

**💡 创新点**

创新点在于证明即使不接触训练数据，仍能通过这些特征准确识别微调的训练域，并发现激活和输出信息比权重更具可识别性。

**🔧 技术方法**

使用LoRA微调、特征提取（权重谱、激活均值/方差、输出熵、句子嵌入等）和标准化逻辑回归分类器。

**📊 数据集**

实验基于Qwen3.5-0.8B模型，对18个公开数据集（涵盖医学、代码、数学、多语言、创意写作）共36次微调。

**📈 对比分析**

在留一数据集外（LODO）测试中，灰盒激活达61%准确率，黑盒输出达64%，远高于20%随机，白盒权重仅42%；与重现率和宏F1一致。

**⚠️ 局限性**

局限包括仅使用单一0.8B混合模型和LoRA，样本量有限、白盒特征手工设计、未检验更大规模或全微调。

---

## 307. Combinatorial limitations of list-decoding Reed-Solomon codes

**arXiv ID:** 2611.07607 | [PDF](https://arxiv.org/pdf/2611.07607v1)

**作者:** Omar Alrabiah `[一作]` (University of California Berkeley), Venkatesan Guruswami `[通讯]` (University of California Berkeley)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文研究全长 Reed–Solomon 码在列表解码中的极限，证明在低特征域上存在常数率的码在 Johnson 半径附近可出现超多项式列表，而在素数域上可在容量下常数距离处得到多项式列表，从而否定了此前的常数列表大小猜想。

**💡 创新点**

创新点在于利用结构化的约束集合（如多项式的子空间、多元二次形式及其极化多项式）构造高度稠密的错误多项式，从而在 Johnson 半径附近实现超多项式列表，并首次给出素数域上容量附近的多项式列表构造。

**🔧 技术方法**

核心技术包括线性化多项式、二次型与其极化、多项式根数与稀疏性的深度分析、以及“移动”集合的乘法放大技巧，结合数论工具（如素数分布与同余条件）实现构造。

**📊 数据集**

本研究不使用任何实验数据集，全部采用理论证明与构造方法。

**📈 对比分析**

与现有结论相比，本论文证明了在特定率和域特征下列表解码的不可行性，说明了此前认为可实现常数列表大小的假设并非普遍成立，显示出明显的理论性能限制。

**⚠️ 局限性**

局限性包括仅覆盖两类特定常数率（低率与高率）以及对素数域仅在满足特定同余条件的无限多素数可用，未能推广至所有率或所有素数，并且构造复杂度高，实际编码实现仍有挑战。

---

## 308. Adversarially Robust Streaming Algorithms for $(k, z)$-Clustering

**arXiv ID:** 2611.07678 | [PDF](https://arxiv.org/pdf/2611.07678v1)

**作者:** Chengzhi Guo `[一作]` (Peking University), Samson Zhou `[通讯]` (Texas A&M University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种对抗性鲁棒的动态几何流算法，该算法在欧几里得空间中实现了(1 + ε)-近似，并使用了(kd(ε^-1))的空间。

**💡 创新点**

该算法显著改善了之前的O(d log n)近似，并且是第一个在kd的空间上线性工作的算法，即使在无适应性设置下也是如此。

**🔧 技术方法**

使用了新的数据无关的欧几里得原语（层次Lipschitz分配）、对聚类的鲁棒维度降低技术和用于内积查询的鲁棒流算法。

**📊 数据集**

使用了动态几何流模型中的数据集，假设输入P是从整数网格[Δ]^d中插入和删除的点流。

**📈 对比分析**

与之前的方法相比，该算法在适应性对手设置下实现了O(d log Δ)的近似比，并且在空间上具有线性依赖于kd的特性，性能优于现有的O(d log Δ)近似。

**⚠️ 局限性**

该算法在处理适应性对手时面临技术挑战，尤其是在动态流中，现有的聚类算法通常依赖于线性草图和采样，无法有效应对适应性对手。

---

## 309. Energy-Aware Open-Vocabulary Detection Scheduling for Low-Power Robotic Autonomy

**arXiv ID:** 2611.07663 | [PDF](https://arxiv.org/pdf/2611.07663v1)

**作者:** Abdul Basit `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在Jetson Orin Nano上实现并评估了针对低功耗移动机器人能量感知的开词汇目标检测（OVD）调度器，利用机器人状态（运动、地图变化、检测置信度）在低、中、高功耗三种模式间动态切换；

**💡 创新点**

创新点在于将OVD推理视为任务级计算资源，提出基于机器人运行状态的有限状态调度策略，并在三种精度/频率模式之间进行动态平衡，以降低功耗的同时保持任务成功率；

**🔧 技术方法**

核心技术包括YOLO‑World‑S 目标检测、TensorRT INT8 与 FP16 推理、基于GMapping的地图熵变化检测、机器人里程计与目标距离感知、以及 ROS Noetic 机器人软件框架；

**📊 数据集**

使用了真实机器人实验数据，AgileX LIMO 在固定起点和瓶子/杯子目标的 40 次导航试验（每种目标 5 个位置）以及 60 秒静态功耗基准测试；

**📈 对比分析**

与固定频率基线（8 Hz、5 Hz、2 Hz）对比，调度器在 8/10 的成功率（高于基线的 7/10）下将平均 Jetson 输入功率降低 5.2%（相对 8 Hz），但由于任务时长延长，整体功耗并未显著下降；

**⚠️ 局限性**

局限性包括：任务时长延长抵消功耗下降；仅评估两种目标且实验环境固定，未随机化；仅测量 Jetson 侧能量，未考虑完整机器人能量；缺乏统计显著性分析。

---

## 310. PharmAgents2: Towards Autonomous End-to-End Drug Discovery

**arXiv ID:** 2611.07653 | [PDF](https://arxiv.org/pdf/2611.07653v1)

**作者:** Haichuan Tan `[一作]` (Tsinghua University), Yanyan Lan `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并验证了一套名为 PharmAgents2 的自主药物发现系统，能够从研究目标出发，自动制定研究策略、执行多学科计算与实验仿真、评估并递交可实验验证的候选分子。

**💡 创新点**

① 通过 Paper2Skill 自动从文献中提炼可重用的决策知识；② 设计 Self‑Scaling Discovery 框架实现大规模并行假设探索；③ 将知识与计算平台 AIPherDD 结合，形成端到端的决策与执行闭环。

**🔧 技术方法**

① 基于语言模型（DeepSeek V4.1 Flash）与知识检索的决策系统；② 论文知识抽取与 PPL 引导的技能完善；③ 并行子代理与自调用机制的 Self‑Scaling Agent；④ 计算工具集（DrugCLIP、DrugLens、GenPack 等）与实验仿真。

**📊 数据集**

① 10 个主题的文献集合用于评估 Paper2Skill 的 PPL 与答案质量；② 121 题目的决策评测集（9 个药物发现场景）；③ 3 个真实项目的实验数据（LpxH、JAK1、AIPher‑X 目标验证）；④ 内部使用的蛋白质库 2,607 个，化合物库 100,000+ 等。

**📈 对比分析**

在 PharmAgents‑Bench v0.1 的决策轨道中，PharmAgents2 在 121 题目上平均提升 30‑40% 的正确率（相比基线 LLM 的 60‑70%）。在端到端轨道中，三项任务的评审分数平均比基线系统高约 20%（专家评审得分从 65% 提升至 85%）。实验验证显示系统递交的候选分子在体外测试中实现 30‑40% 的成功率。

**⚠️ 局限性**

① 缺乏持续学习机制，未能将实验反馈直接用于模型或技能再训练；② 依赖预设工具和数据库，扩展性受限；③ 对复杂多目标任务的全局优化能力尚待提升；④ 在极端数据稀缺或新颖机制预测时仍表现不佳。

---

## 311. Don't Prompt Your Agents. Equip Them: Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems

**arXiv ID:** 2611.07652 | [PDF](https://arxiv.org/pdf/2611.07652v1)

**作者:** Vladimiros Peilivanidis `[一作]` `[通讯]` (ARPA Hellenic Logical Systems), Vladimiros Peilivanidis (ARPA Hellenic Logical Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Skillware 框架，将自主代理的认知规划与可执行能力分离，构建包含契约、执行代码、指令、验证与语料的五元组能力包；实现无网络、无 IPC 的本地适配器、确定性 DAG 链接引擎，并在 368 条评测条件下验证其在小模型与大模型中的效果。

**💡 创新点**

创新点包括：
- 彻底分离认知与执行，消除随机推理与代码生成的噪声。
- 统一、可版本化的能力包标准，提供静态 JSON 契约、离线单元测试与可验证性。
- 统一跨模型的本地适配器，支持 Gemini、Claude、OpenAI、Ollama 等主流 LLM；
- 通过确定性链式执行显著降低 token 与成本，实现 84% 以上的 token 减少、97% 的成本压缩；
- 发现并验证了“小模型参数悖论”，在同等任务下 3B‑8B 本地模型配合 Skillware 能超越 70B‑1.8T 大模型。

**🔧 技术方法**

技术手段包括：
- JSON Schema + 代码签名、Python BaseSkill 运行时；
- 内联适配器（Gemini、Claude、OpenAI、Ollama 等）；
- 确定性 DAG 运行时与条件跳过；
- 离线安全校验与零令牌预执行过滤；
- SHA‑256 链式审计日志与可验证合规；
- Unicode、同形字母、HTML 等多阶段离线净化管道。

**📊 数据集**

使用的数据集与评测：
- 内嵌的 OFAC 受限名单、EVM 地址、金融交易日志等静态数据库；
- 公开的开源模型权重（Llama‑3、Qwen、Phi‑3、TinyLlama 等）；
- Skillware‑Bench：368 条真实任务，覆盖邮件处理、企业识别、金融筛查、权限确认、链式合规等七大场景；
- 参考基准如 Berkeley Function Calling Leaderboard 与 ToolBench。

**📈 对比分析**

比较方法与性能：
- 在 23 种模型、368 条实验条件下对比 8 种架构（原生代码生成、平面工具调用、完整 Skillware、LangChain、AutoGen、CrewAI、Semantic Kernel 等）。
- 结果显示：token 使用降低 51%（平均）、延迟提升 55%（平均）、通过率提升至 91.8%（从 58% 基线），小模型配合 Skillware 在成本、速度、可靠性上均超过大模型；
- 成本曲线显示，配备 Skillware 的 8B 本地模型仅需 0.42 美元/1000 次，对比 15.58 美元/1000 次的未配备大模型，降幅 97.3%。

**⚠️ 局限性**

局限性：
- 仍需人工编写与维护能力包（虽然 5‑分钟模板大幅简化，但不完全自动化）；
- 依赖模型对认知层的可靠性，若模型推理质量下降，仍可能产生错误的参数选择；
- 目前仅支持 Python 运行时，其他语言需额外实现；
- 需要将关键数据集离线打包，更新时需同步维护；
- 仍需人工确认安全门槛（确认参数、dry‑run）；
- 大规模能力库管理与冲突检测会增加注册表管理负担；
- 虽然采用离线校验，但若能力包本身被篡改，仍可能带来安全隐患。

---

## 312. What the Middleman Delivers: Answer Behavior, Tool Permissions, and Billed Usage in LLM API Relays

**arXiv ID:** 2611.07650 | [PDF](https://arxiv.org/pdf/2611.07650v1)

**作者:** Xiaodong Wu `[一作]` (Queen's University), Jianbing Ni `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了20家第三方LLM API中继服务，在官方参考接口的基准下，对回答准确性、工具权限执行、以及计费 token 计数等三方面进行系统测评，并通过两轮实验观察结果稳定性。

**💡 创新点**

创新点在于将回答质量、内容注入检测、工具调用合规性和计费一致性等多维度评估整合到同一套公开接口测试流程中；引入受控混合响应模拟和重复测量来评估模型替换的可检测性，提供了一套可复现、可验证的中继评估框架。

**🔧 技术方法**

技术手段包括：基于 Knowledge Boundary as Fingerprint 的问答题库设计、官方与中继接口的精确匹配调用、统计检验（Fisher精确检验+BH校正）用于行为一致性判定、代码与安装指令的自动扫描与人工复核、token 使用计数差异与账户余额对账。

**📊 数据集**

使用的数据集为：由 96 道候选题构成的知识/参数/能力三类题库，挑选出 27 道用于正式测试；收集了 20 家中继在两轮实验中对每道题的 5 次调用结果，共 3,332 条回答（含官方参考）。

**📈 对比分析**

比较方法为：将每个中继的答案与官方参考答案按题目进行准确率比对，计算风险差 RD 并按预设阈值标记 Confirmed / Suspected；对工具调用与 token 计数分别做异常率与余额对账；两轮实验中检验标签稳定性。结果显示低价中继中有 7/10 在第一轮被标记为行为不一致，工具调用违约出现于 4 家中继，token 计数差异在 5 家中继出现，其中一方计费与余额完全匹配。

**⚠️ 局限性**

局限性包括：样本非随机、仅覆盖 20 家中继；实验仅使用非攻击性问答和代码请求，无法覆盖所有注入或隐蔽攻击；未能直接识别后端模型或替换原因，因缺少官方-to-official 参考复测；多轮实验间仅 11–18 天，无法覆盖长期变化；扫描工具可能漏检复杂注入，且未对工具调用执行做实际验证。

---

## 313. UniGADD: Universal Generator-Agnostic Deepfake Detector

**arXiv ID:** 2611.07647 | [PDF](https://arxiv.org/pdf/2611.07647v1)

**作者:** Bernard Tomczyk `[一作]` (Lancaster University), Plamen Angelov `[通讯]` (Lancaster University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段的对比学习与原型细化框架 UniGADD，用于在已知与未知生成器上检测深度伪造。

**💡 创新点**

创新点在于结合监督式对比损失与只对真实图像进行的簇内紧凑化细化，使得模型在保持对已知伪造器高精度的同时，对未见伪造器具有更强的泛化能力。

**🔧 技术方法**

采用 Vision Transformer 预训练模型（如 DINOv2、CLIP）提取特征，使用 K‑means 生成真实图像伪标签，进行监督对比学习；随后通过超正样本和弱正样本的余弦相似度约束实现簇内紧凑化，最终以最近中心距离做判别和生成器归因。

**📊 数据集**

在 AIGC 基准数据集上进行评估，涵盖多种 GAN 与扩散模型（如 BigGAN、StyleGAN、Stable Diffusion、Midjourney 等）。

**📈 对比分析**

与多种基准方法（监督、无监督、线性分类器等）进行 Leave‑One‑Out 与 Leave‑N‑Out 对比。UniGADD 在已知生成器上的准确率与最先进方法相当，在未知生成器上的准确率显著提升（平均 85.5% 对比 73.6% 的最佳竞争对手），且在不同生成器组合下方差更小。

**⚠️ 局限性**

局限性包括：依赖冻结的 ViT 编码器，可能受到编码器偏差影响；仅处理单帧图像，缺乏时序特征；聚类步骤在大规模数据上计算成本较高；对压缩、模糊等低级扰动的鲁棒性未作系统评估；阈值固定，可能需在实际部署中动态调整。

---

## 314. Data-Driven Configuration of Protective Separation Zones in Collaborative Robot Cells from Operator Occupancy

**arXiv ID:** 2611.07621 | [PDF](https://arxiv.org/pdf/2611.07621v1)

**作者:** Aulon Bajrami `[一作]` (Fraunhofer Institute for Manufacturing Engineering and Automation IPA), Werner Kraus `[通讯]` (Fraunhofer Institute for Manufacturing Engineering and Automation IPA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于相机的、数据驱动的方法，用实时姿态检测与活动识别生成操作员占据热图，从而在协作机器人工作单元中对保护分隔区进行优化配置。

**💡 创新点**

创新点在于将实际工位操作员占据数据直接映射为安全距离与机器人速度的权衡，提供离线安全区尺寸建议，而非传统的基于最坏情况的静态配置，并结合工位级风险评估。

**🔧 技术方法**

使用的技术包括Intel RealSense RGB‑D相机、轻量级OpenPose姿态估计、OKS关联与One Euro滤波、3D重投影、基于热图的空间统计以及ISO/TS 15066速度与分离监测模型。

**📊 数据集**

数据集来源于两个工业合作伙伴的生产现场，收集了数个生产班次的操作员位置轨迹、活动标签与安全距离统计，未公开完整原始数据。

**📈 对比分析**

通过在两台工业机台上实施实验，保护区半径可缩减约17%，工艺循环时间提升约9%，与纯在线安全区缩放在理想感知环境下报告的26.5%提升形成对比。

**⚠️ 局限性**

局限性包括：需在已完成安全评估的现场采集数据，接受过程受数据隐私与修改安全概念的阻碍；模型仅在离线配置时有效，无法实现在线动态调整；评估仅覆盖两台机台，缺乏统计代表性。

---

## 315. Monotone Viability Learning for Offline Safe Reinforcement Learning

**arXiv ID:** 2611.07616 | [PDF](https://arxiv.org/pdf/2611.07616v1)

**作者:** Ziqian Wang `[一作]` (Tsinghua University), Zhen Zhang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在离线安全强化学习中通过单调有限时域集合递归学习 Hamilton–Jacobi 可行集，并利用其对部署时的动作进行安全认证。

**💡 创新点**

创新点在于用二元监督与期望分位交叉熵实现可行集的递归逼近，保持语义保真、收敛性与数值稳定，同时通过概率校准和轨迹证人保护提升可靠性。

**🔧 技术方法**

技术包括基于行为分布的离线递归可行集学习、期望分位交叉熵与二元阈值校准、动作级安全评分网络、基于行为克隆的 Gaussian 策略与成本回归。

**📊 数据集**

数据集涵盖 Dubins car 约 20 万条转移、Bullet‑Safety‑Gym 17 任务（Ball/Car/Drone/Ant 等）和 MetaDrive 9 任务的离线轨迹。

**📈 对比分析**

与 FVI、FISOR、BCQL、CPQ、CDT、LSPC、GSCO 等基线对比，MoViL 在所有任务均满足成本约束，平均成本最低且回报竞争力强；Dubins car 上 IoU 99.34%、训练时间 45 秒，远优于 FVI（2985 秒）和 FISOR（84 秒）。

**⚠️ 局限性**

局限在于对可行集递归深度与阈值设置敏感，且对高维连续状态空间的泛化尚需验证，缺乏对非确定性动态的理论与在线验证。

---

## 316. MCPHallu: Benchmarking Reasoning, Execution, and Memory Hallucinations in MCP Agents

**arXiv ID:** 2611.07689 | [PDF](https://arxiv.org/pdf/2611.07689v1)

**作者:** Yizhen Jiang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了 MCPHallu 基准，用于在多服务器、状态持久的 MCP 环境中诊断代理的幻觉失效。

**💡 创新点**

创新点在于通过可控结构触发器将失败风险聚焦至四类幻觉子型（Branch Collapse、Unreachable Goal、Tool Misuse、Context Forgetting），实现子型级别的可解释评估，并在真实生产服务器上进行验证。

**🔧 技术方法**

采用 ReAct 代理框架、动态脚本执行、依赖图结构检查与 LLM 判别器相结合的多组件评分技术。

**📊 数据集**

使用358个可执行任务，涵盖5个能力域、3个难度级别，并部署在36个真实 MCP 服务器（约300个工具）上。

**📈 对比分析**

与14个前沿 LLM（OpenAI、Anthropic、Google、ByteDance 等）对比，平均得分0.642，最佳模型0.728；子型间表现差异显著，说明评估具备细粒度诊断能力。

**⚠️ 局限性**

局限性包括只覆盖四种幻觉子型，未能涵盖多模态或跨会话记忆失误；基准规模虽真实但仍相对有限，未来需扩展更多子型和更大任务集。

---

## 317. A Universal Threshold Theory for Local Properties of Random Linear Codes

**arXiv ID:** 2611.07661 | [PDF](https://arxiv.org/pdf/2611.07661v1)

**作者:** Vinayak M. Kumar `[一作]` (Simons Institute for Theory of Computing), Gilad Shavit `[通讯]` (Ben Gurion University of Negev)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨了随机线性编码（RLC）中局部属性的阈值，提出了b-local属性的阈值与期望阈值之间的关系，并展示了如何将这些保证转移到结构化编码上。

**💡 创新点**

创新点在于提供了一个通用的去随机化声明，表明任何在给定速率下满足的递减局部属性在类似速率下也适用于欺骗覆盖联合界限的伪随机集合。

**🔧 技术方法**

使用了覆盖的最佳联合界限来表述阈值，并通过线性规划对局部属性进行分析。

**📊 数据集**

论文未具体提及使用的数据集，但讨论了随机线性编码和伪随机编码的理论性质。

**📈 对比分析**

通过与之前的研究进行比较，论文展示了其O(b)的误差改进，优于之前的O(q^2blog_q n)和O(b^2+nb/log_2 q)的误差界限，且在适当的伪随机性界限下，速率损失在b=o(n)时消失。

**⚠️ 局限性**

限制在于该研究主要集中在局部属性的阈值上，可能未能涵盖所有类型的编码属性，且对某些特定结构化编码的适用性仍需进一步验证。

---

## 318. Not Losing Your Way: Mitigating Visual-Inertial Odometry Drift through On-Demand Visual Feature Creation

**arXiv ID:** 2611.07687 | [PDF](https://arxiv.org/pdf/2611.07687v1)

**作者:** Simone D'Angelo `[一作]` (University of Naples Federico II), Fabio Ruggiero `[通讯]` (University of Naples Federico II)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种利用UAV仅通过摄像头和IMU在飞行中主动创造视觉特征以降低VIO漂移的框架。

**💡 创新点**

创新点在于把环境改造（投放地面标记或物理接触墙面标记）与VIO结合，并通过信息量指示器与时间确认机制动态触发特征创建。

**🔧 技术方法**

使用了视觉惯性里程计、IMU、摄像头、地面标记投放、墙面接触标记、时间确认与迟滞算法、OptiTrack飞控。

**📊 数据集**

评估使用了三种仿真环境、900次因子实验以及五次真实走廊实验，并在实验中使用OptiTrack提供飞行控制。

**📈 对比分析**

与基线VIO对比，主动特征创建使位置均方根误差降低约35%，证明该方法能显著抑制漂移。

**⚠️ 局限性**

局限性包括需降低巡航速度才能最小化漂移、对速度-策略交互敏感、仅在重复或纹理不足的环境中验证，且在高度动态或无结构环境下的表现尚未探究。

---

## 319. DEQ-Det: Dual Evidence Queries for Few-Shot Analog IC Device Detection

**arXiv ID:** 2611.07682 | [PDF](https://arxiv.org/pdf/2611.07682v1)

**作者:** Zidong Zhao `[一作]` (Zhejiang University), Guoqiang Zhang `[通讯]` (Ningbo Tech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 DEQ-Det 框架，利用前景与背景双重证据实现少样本模拟 IC 设备检测。

**💡 创新点**

创新点在于在无额外标注的前提下构造前景/背景参考库，双证据查询同时进行目标定位与干扰抑制，并结合质量感知的检测排序。

**🔧 技术方法**

采用 DINOv2 ViT‑L/14 编码器与冻结的 IC 重建适配器，构建前景/背景特征银行，配合查询选择器、RoI 关系头、质量预测与排序模块，以及 AdamW 等优化策略。

**📊 数据集**

使用基于电子显微镜的 IC 布局图数据集，907 个训练窗口、282 个测试窗口，涵盖 5 个留出的新设备类别，共 3,118/760 个实例。

**📈 对比分析**

与 TFA、Meta R-CNN、FSCE、VFA、FPD、DE‑ViT/FSOD‑VFM 等基线对比，10 / 30 shots 下 macro AP50 分别达到 39.41 / 53.03，超越最强基线约 13 / 12 分，显著提升检测性能。

**⚠️ 局限性**

局限性包括对背景干扰仍有限制，依赖特定特征提取器和阈值设置，且在极少样本（1 shot）或跨工艺迁移时性能下降，未充分验证更大类别集的泛化能力。

---

## 320. Instrumental Variables in Continuous-Time Stochastic Processes

**arXiv ID:** 2611.07674 | [PDF](https://arxiv.org/pdf/2611.07674v1)

**作者:** Lars Pennig `[一作]` (Technical University of Munich), Niki Kilbertus `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套基于连续时间随机微分方程（SDE）的工具变量（IV）框架，导出了路径值矩条件，并在线性（OU）和非线性（核/神经网络）情形下给出了可识别性分析与一致估计方法。

**💡 创新点**

创新点包括：
- 在连续时间SDE上首次构建IV理论，利用漂移函数的稀疏性刻画结构IV假设；
- 推导出全路径可观测的矩条件，证明单一连续IV过程即可识别多维因果效应；
- 对OU系统给出可控性条件的识别定理与闭式一致估计器；
- 针对非线性目标，提出基于签名核的极小化估计方案，并给出理论与实验支持。

**🔧 技术方法**

使用技术包括：SDE建模与Itô公式、路径值矩条件推导、可控性与Lyapunov方程分析、非参数协方差估计（Nadaraya–Watson）、最小二乘与伪逆求解、RKHS/签名核、V-statistic、梯度下降与正则化、模拟实验与RMSE评估。

**📊 数据集**

数据集：
- 通过随机生成的OU过程轨迹（用于验证线性方法）；
- 半合成的糖尿病葡萄糖‑胰岛素动力学数据（用于验证非线性方法与真实实验）；
- 真实糖尿病患者数据（外部胰岛素注射作为IV），用于实证检验。

**📈 对比分析**

与方法比较：
- 与忽略未观测混杂的MLE相比，IV-SDE在所有混杂强度下误差显著降低；
- 与离散时间IV（VAR(1)估计、归一化参数估计）相比，IV-SDE在平稳 OU 系统中保持不偏且误差更小；
- 在非线性情形下，高混杂下IV-SDE优于MLE；
- 在糖尿病实验中，IV-SDE 的RMSE为 0.0157，显著低于MLE 的 0.0998。

**⚠️ 局限性**

局限性：
- 非线性识别仍依赖正则化与核选择，理论完整性待完善；
- 对弱工具变量的稳健性与不确定性量化不足；
- 方法假设过程平稳，非平稳系统的推广尚未实现；
- 非参数协方差估计对不规则采样敏感，可能影响估计精度。

---

## 321. Egocentric Action Recognition with Objects and Vision-Language Affordances

**arXiv ID:** 2611.07673 | [PDF](https://arxiv.org/pdf/2611.07673v1)

**作者:** Raúl Boix-Aldeguer `[一作]` (University of Alicante), Jose Garcia-Rodriguez `[通讯]` (University of Alicante)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种结合冻结的 SlowFast 视频专家与基于注意力的对象专家的头戴视角动作识别方法，利用 VLM 生成对象的 affordance 以及交互 affordance 进行非活跃对象过滤；

**💡 创新点**

创新点在于：1）零样本 VLM 生成对象 affordance 与交互 affordance；2）使用交互 affordance 过滤检测到的非活跃对象；3）通过 late fusion 将视频与对象信息融合；

**🔧 技术方法**

使用的技术包括：冻结的 SlowFast 3D CNN、ViT-B/16 用于对象 crop 编码、DistilBERT 对文本（对象类别、位置、affordance）编码、自注意力（MHA）用于对象专家、Gemini 3.1 Flash‑Lite VLM 生成 affordance 与交互 affordance、Faster‑R‑CNN 对象检测、late fusion 与加权投票；

**📊 数据集**

使用了 MECCANO 视角动作数据集，该数据集包含 61 个动作类别（12 个动词+20 个对象），并提供官方 active‑object 检测器与标注框；

**📈 对比分析**

与公开基线相比，方法在 MECCANO 上 Top‑1 取得 49.9%（比 SlowFast RGB 基线提升 4.7%），Top‑5 82.5%（提升 8.8%），并且在使用检测框时仍显著优于其他基线（如 G3Ego、HOCL+OSL 等）；

**⚠️ 局限性**

主要局限在于：1）依赖 VLM 的交互 affordance 过滤效果有限，误删活跃对象 3.7%；2）对非活跃检测的依赖导致模型对检测器性能敏感；3）只在单帧上判断对象是否被操作，未充分利用时序信息；4）对小对象检测召回率低，影响整体性能。

---

## 322. An Information Bottleneck for Post-hoc Memorization Mitigation in LLMs

**arXiv ID:** 2611.07669 | [PDF](https://arxiv.org/pdf/2611.07669v1)

**作者:** Dina EL Zein `[一作]`, James Henderson `[通讯]` (Idiap Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练大型语言模型后期插入可插拔的非参数变分信息瓶颈（NVIB），从而抑制模型对训练数据的逐字记忆并保持语义表现

**💡 创新点**

提出了在冻结的LLM骨干与语言建模头之间插入噪声化的随机信息瓶颈，以压缩隐藏表示并过滤细粒度记忆，而不需要重新训练模型

**🔧 技术方法**

利用非参数变分信息瓶颈（NVIB）结合稀疏Dirichlet过程、随机采样、去噪注意力以及对KL正则化的调节

**📊 数据集**

使用Open‑R1 Mixture‑of‑Thoughts作为微调数据，评估在四本受版权保护的书籍（《Animal Farm》《The Hitchhiker's Guide to the Galaxy》《Crime and Punishment》《The Fellowship of the Ring》）上进行的提取攻击，另外在Llama‑3.1‑8B和Qwen3‑8B两大公开权重模型上测试

**📈 对比分析**

对比原始模型和仅插入普通Transformer层的基线，利用概率提取分数、提取序列数量以及多项评测（MMLU、MMLU‑Redux、MMLU‑Pro、GPQA Diamond、MuSR、IFEval）评估下游性能。NVIB显著降低提取概率（平均分数下降约1‑2个数量级），且在Qwen3‑8B上几乎不损失或略提升多数下游指标；在Llama‑3.1‑8B上则表现出任务依赖的轻微下降或提升

**⚠️ 局限性**

仅在两种模型架构上验证，未覆盖更大或更异构的LLM；提取评测仅关注书籍文本，未检验其他敏感信息；瓶颈的压缩强度和正则化参数对结果影响需进一步探索

---

## 323. Simple Meta-Learning for HPO and CASH Using Performance Representations

**arXiv ID:** 2611.07659 | [PDF](https://arxiv.org/pdf/2611.07659v1)

**作者:** Emiel Witting `[一作]` (Delft University of Technology), Tom Julian Viering `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种直接使用历史任务性能向量作为配置表示的元学习方法，结合Bayesian Ridge回归和MMR采集函数，用于高维CASH和HPO搜索。

**💡 创新点**

创新点在于将历史性能直接作为输入向量代替高维超参数编码，实现简单的元学习；提出无后验的MeanMMR采集规则，并在TabArena上证明在CASH上优于专用元学习基线。

**🔧 技术方法**

使用了Bayesian Ridge回归、Gaussian Process、Random Forest、MTGP、RGPE、ABLR、Active Testing等传统与元学习模型，并引入MMR多样性采集方法，采用5折任务级交叉验证进行评估。

**📊 数据集**

采用TabArena基准，包含多任务表格分类/回归数据以及大量模型与配置的性能矩阵。

**📈 对比分析**

在CASH基准中，性能表示+线性回归实现最佳平均regret，仅需约73次评估即可达到ABLR的149次；在HPO基准仅次于ABLR；模型训练时间显著更短；MeanMMR在CASH上亦超过专用元学习基线。

**⚠️ 局限性**

局限性包括需要完整的历史性能矩阵，稀疏矩阵支持有限；对连续超参数空间缺乏扩展；在低维HPO上不如专用元学习方法表现。

---

## 324. ArtCrafter: Feed-Forward Generation of Articulated 3D Object with Analytic Joint Derivation

**arXiv ID:** 2611.07658 | [PDF](https://arxiv.org/pdf/2611.07658v1)

**作者:** Minh Canh Tu `[一作]` (Qualcomm AI Research), Khoi Nguyen `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 ArtCrafter，一种双图输入下可控生成可关节化 3D 资产并自动推导 URDF 关节参数的两阶段 Feed‑Forward 框架。

**💡 创新点**

创新点包括：1+2N 结构化 slot DiT 使同一关节的两状态共享底座槽；Triplet Attention 促成跨状态几何一致；repulsion loss 保证部件不重叠；并通过解析 rigid transformation 自动获取关节类型、轴、原点与范围，无需学习回归头。

**🔧 技术方法**

采用预训练 TripoSG DiT + VAE、Slot Attention 结构、Triplet Attention、state‑cross attention、global attention、repulsion loss 以及 RANSAC+ICP 的解析几何注册。

**📊 数据集**

使用 URDF‑Anything+（基于 PartNet‑Mobility 与 PhysX3D 的组合）训练集，另外构造 OOD 测试集。

**📈 对比分析**

与 DIPO 在同一双图设定下对比，几何指标 CD/F1 分别提升约 18‑21%，关节指标保持竞争力；与单图 PhysX‑Anything、检索方法 URDFormer 以及优化方法 FreeArt3D 相比，速度更快且质量更好。

**⚠️ 局限性**

局限性：仅支持两种关节类型（旋转/滑动），对多自由度或极端姿态的鲁棒性有限；需要已知或估计可动部件数 N，且依赖于两张高质量图像。

---

## 325. GraspBench: Execution-Aware Cross-Robot Evaluation of 6-DoF Grasp Generation in MuJoCo

**arXiv ID:** 2611.07651 | [PDF](https://arxiv.org/pdf/2611.07651v1)

**作者:** Abdul Basit `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GraspBench，一个基于MuJoCo的执行级评估协议，用来对六种抓取生成器在四种机器人、五种诊断物体和结构化工作台姿态网格上的完整抓取管线进行统一测试。

**💡 创新点**

创新点在于将抓取生成器与机器人特定的IK过滤、运动规划和固定执行序列结合起来，形成场景级、机器人感知条件下的执行感知评估；并通过保守缺失输出计失败的方式实现公平对比。

**🔧 技术方法**

使用的技术包括MuJoCo物理仿真、DLS IK、OMPL RRTConnect运动规划、以及六种现有的抓取生成算法（AnyGrasp、Contact‑GraspNet、EconomicGrasp、GPG、GraspGen、HGGD）。

**📊 数据集**

使用的“数据集”是一个人工构造的5物体诊断集合（飞机、玻璃、立方体、圆柱体、瓶子）以及在Near/Mid/Far、Left/Center/Right、Pitch±45°的三维网格，共540个场景。

**📈 对比分析**

比较方法：在相同场景网格下，统计每个生成器在IK过滤后选出的最高分可行抓取的二元抬升成功率。GraspGen最高73.1%，EconomicGrasp次之63.5%，其余四种在44–48%之间；不同机器人或物体会显著影响成功率。

**⚠️ 局限性**

局限性：仅在单物体、无障碍、无随机噪声的仿真环境中测试；未考虑多物体或真实机器人误差；缺少对各个执行阶段（IK、规划、抓取）失效原因的细粒度分析；候选数量与预算不统一，无法直接比较。

---

## 326. Generator-Aligned Reward Modeling with Target-Aware Bounded Optimization for Visual Generation

**arXiv ID:** 2611.07648 | [PDF](https://arxiv.org/pdf/2611.07648v1)

**作者:** Ziqiao Weng `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合生成器对齐奖励建模(GARM)和目标感知上界优化(TABO)的框架，用以在保持生成质量的前提下，强化视觉生成模型的指定属性。

**💡 创新点**

创新点包括：①通过生成器支持匹配与内容匹配硬负样本来解决真实样本与生成样本分布不匹配与内容混淆；②设计数据校准的平滑上界奖励函数(TABO)，使奖励优化与优化器无关；③在Flow‑GRPO基础上引入阶段优势缩放和生成器‑目标学习曲线的TA‑GRPO。

**🔧 技术方法**

采用 VAE 先验的潜在奖励模型、基于生成器支持的邻域匹配、结构化标签的内容匹配检索、VADER 直接梯度优化与 TA‑GRPO 的策略梯度更新、以及参考保持的 KL 正则化。

**📊 数据集**

使用 Re‑LAION‑Caption19M（约 60K 张图像）和 5.5M 带结构标签的视频数据集；针对图像属性（Vintage Faded、Oil Painting、Muscular）与视频属性（2D Anime）进行实验。

**📈 对比分析**

与基线奖励训练、仅生成器对齐、仅内容匹配、仅支持匹配、以及 SFT 对比，评估指标包括属性百分位均值/P20、HPSv2/3、PickScore、VQA、Q‑Align、VBench、VideoScore2 等；GARM 在目标属性提升上表现最佳，质量保持竞争；TABO 在未受限最大化基础上显著平衡目标与质量；TA‑GRPO 在 Flow‑GRPO 上进一步提升目标表达并降低中性提示泄漏。

**⚠️ 局限性**

局限性包括：对正样本的依赖（需足量标注）；奖励模型仅在 VAE 潜空间内学习，可能不适用于所有视觉属性；在未知属性或极端条件下的泛化尚未验证；计算成本相对较高，尤其是生成器‑目标学习曲线与多阶段优化。

---

## 327. Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning

**arXiv ID:** 2611.07639 | [PDF](https://arxiv.org/pdf/2611.07639v1)

**作者:** Amit Sagtani `[一作]`, Saloni Garg `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个确定性可验证奖励基准(gpu-train-debug)，并在控制环境下对比监督微调(SFT)与可验证奖励强化学习(RLVR)的边际贡献，进一步通过前置试点与预注册确认的两阶段实验评估其效果。

**💡 创新点**

创新点在于：①提出并发布了具有稀释校准难度、精细部分计分的确定性验证器基准；②首次在同一任务分布下通过试点-确认的预注册设计精准测量RLVR相对于SFT的增益；③通过头房间分解与“系统性残差”机制分析，解释了RLVR增益极小的根本原因。

**🔧 技术方法**

使用的技术包括：基于程序化验证器的确定性奖励；梯度强化学习算法GRPO；群组相对策略优化；稀疏到部分计分的多目标奖励；预注册的配对自助bootstrap统计检验；以及系统化的实验记录与重现框架。

**📊 数据集**

数据集为人工生成的GPU/LLM训练失败诊断任务，共1,877个任务，覆盖30种典型故障模式；划分为训练1,200、难度硬分200、测试236、OOD 241，所有任务均提供金手册示例。

**📈 对比分析**

对比方法为：①基准模型；②SFT（1/2轮）；③SFT+GRPO（60步，lr1e-6）；④基准+GRPO（60步，探索）。性能上，SFT在硬集上达0.785，GRPO单独0.314；SFT+GRPO相较SFT仅提升硬集+0.22（CI包含0）但在OOD提升+0.33（CI不包含0），表明RLVR的边际增益极为有限。

**⚠️ 局限性**

局限性包括：仅使用单一3B规模模型；RL步骤有限（60步，单个种子）；只评估GRPO算法；验证器完全确定且部分计分；实验仅在GPU/LLM训练失败诊断的合成任务上；高质量金示例可能夸大SFT效果；对不同规模模型、RL预算、学习算法、稀疏/学习型验证器的适用性未作验证。

---

## 328. Approximate Hamming Convolution via Rectangular Matrix Multiplication

**arXiv ID:** 2611.07622 | [PDF](https://arxiv.org/pdf/2611.07622v1)

**作者:** Przemyslaw Uznanski `[一作]` `[通讯]`, Przemyslaw Uznanski

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种随机化算法，用于在 word‑RAM 模型下对长度为 N 的模式与长度为 2N 的文本进行 (1±)-近似汉明距离的全对齐计算。

**💡 创新点**

创新点在于通过引入多层二进制划分的随机乘法哈希、XOR 草图与矩阵乘法的巧妙结合，扩展了可接受误差范围至 a≤1/8，并在更宽范围 (1/12, (3+√3)/16) 内显著降低运行时间。

**🔧 技术方法**

采用的技术包括：随机乘法哈希、XOR 基础草图、二进制树的加法恒等式、矩阵乘法（利用 ω(k) 计算矩阵乘法指数）以及已知的确定性精确算法进行距离阈值补偿。

**📊 数据集**

未使用外部数据集，全部工作基于理论分析与算法实现。

**📈 对比分析**

与先前工作（如 Chan 等人提出的近似线性时间算法）比较，本文在误差参数 a 在 (1/12, (3+√3)/16) 区间内实现了更低的时间指数；当 a≤1/8 时达到近似线性时间 (O(N))，比之前的 O(N^{1+2a/3}) 更优。

**⚠️ 局限性**

主要限制是算法性能受限于当前矩阵乘法指数 ω(k) 的上界；若 ω(k) 无法进一步改进，算法在更大误差范围内的时间提升将受限；另外，精确补偿步骤仍需额外的确定性算法，增加了实现复杂度。

---

## 329. The Popular Get More Popular: Auditing Brand Bias in Generative Search

**arXiv ID:** 2611.07614 | [PDF](https://arxiv.org/pdf/2611.07614v1)

**作者:** Giorgio Monaco `[一作]` (Politecnico di Milano), Francesco Pierri `[通讯]` (Politecnico di Milano)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过两项实验（观察性品牌受欢迎度偏差审计和受控品牌熟悉度偏差实验）评估了生成式搜索引擎对品牌的系统性偏见，探讨了检索增强生成（RAG）与模型参数记忆对偏差的影响，并考察了事实保留的生成引擎优化（GEO）策略的有效性。

**💡 创新点**

创新点在于首次将品牌偏差细化为“品牌受欢迎度偏差”和“品牌熟悉度偏差”，提出了结合模型参数记忆与检索结果的审计框架，并通过品牌名称置换和GEO重写的对照实验，系统量化了品牌熟悉度对排名的显著影响，揭示GEO在事实保留条件下效果有限。

**🔧 技术方法**

技术方法包括调用多款大型语言模型（GPT‑4、Gemini、Perplexity、DeepSeek、Llama）在检索增强（RAG）与仅参数记忆两种配置下进行回答生成；构造基于真实产品页面的上下文；使用OLS回归、bootstrap、Jaccard/余弦相似度等统计工具对品牌提及、引用和排名差异进行分析。

**📊 数据集**

数据集主要来源于意大利市场的商业查询（DataForSEO），涵盖四个行业的1,176个知名品牌（其中683个拥有官方域名）以及24个受控产品页面，结合相应的网页来源和域名信息。

**📈 对比分析**

比较方法是通过统计学显著性检验（t检验、Benjamini–Hochberg校正）评估品牌受欢迎度与提及/引用率的斜率，以及在受控实验中品牌名替换和GEO重写对排名的平均提升。结果显示：品牌受欢迎度偏差普遍存在；品牌熟悉度偏差导致真实品牌相较于虚构品牌平均提升约0.44个排名位置；GEO重写在事实保留条件下平均提升不到0.05个位置。

**⚠️ 局限性**

局限性包括：实验仅在意大利语Rome地区进行，时间点固定；Study 2仅在禁用实时检索的环境下执行；persona模板有限；使用黑盒API，缺乏对客户端提示的可视化；GEO重写仅保留事实，未探究恶意或夸张修饰；未与传统检索引擎进行对比；未进行情感分析。

---

## 330. GEKey: Learning Geodesic Eccentric 3D Keypoints via Self-Supervision

**arXiv ID:** 2611.07677 | [PDF](https://arxiv.org/pdf/2611.07677v1)

**作者:** Pierre Onghena `[一作]` (Mines Paris PSL University), Beatriz Marcotegui `[通讯]` (Mines Paris PSL University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 GEKey 框架，通过计算点云的 geodesic eccentricity 并提取其局部极大点作为伪标签，在自监督方式下训练模型完成 3D 关键点检测；

**💡 创新点**

创新点在于：① 利用 SE(3) 不变的 geodesic eccentricity 识别实例特定的关键点并与类别特定点平衡；② 采用局部极大点作为伪标签并通过 Hungarian bipartite 匹配损失实现可排序的关键点输出；③ 在保持高覆盖率与一致性的同时实现对噪声、下采样、非刚性变形的鲁棒性；

**🔧 技术方法**

使用的技术包括：k‑NN 图构建、地理弧度（geodesic eccentricity）变换、Farthest Point Sampling、软注意力关键点头、Hungarian 匹配损失、Point Transformer V2 等深度网络骨干；

**📊 数据集**

使用的数据集包括：ShapeNetCoreV2、KeypointNet、CAPE、ITOP、DeformingThings4D；

**📈 对比分析**

与现有监督、无监督和手工方法在 KeypointNet、CAPE、ITOP、DeformingThings4D 上进行 mIoU、DAS、包含率、覆盖率、时间一致性等多指标对比，GEKey 在 KeypointNet 上平均 mIoU 达 54.5%（比前一最佳提升 5.5 分），在 CAPE、ITOP、DeformingThings4D 的包含率、覆盖率及时间一致性均高于所有基线方法；

**⚠️ 局限性**

局限性在于：对极端噪声或极低点密度下性能下降；缺乏显式的点间分离约束，可能导致密集区域的关键点聚集；对非等距变形或细纹理细节的鲁棒性仍需进一步提升。

---

## 331. $O(1/n)$ Convergence Rate for RVI $Q$-Learning

**arXiv ID:** 2611.07676 | [PDF](https://arxiv.org/pdf/2611.07676v1)

**作者:** Avik Kar `[一作]` (Indian Institute of Science), Shalabh Bhatnagar `[通讯]` (Indian Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

建立了有限时间收敛保证，针对有限弱通信马尔可夫决策过程的同步相对价值迭代（RVI）Q学习。

**💡 创新点**

提出了线性正则性和噪声条件的概念，证明了在这些条件下，RVI Q学习的均方残差以O(n^-1)的速率收敛。

**🔧 技术方法**

使用了随机Krasnosel'skiĭ–Mann迭代和非扩展线性正则映射的理论。

**📊 数据集**

研究了有限弱通信马尔可夫决策过程（MDP）。

**📈 对比分析**

与现有方法相比，当前方法在均方残差和贪婪策略的期望次优性上均以O(n^-1)的速率收敛，性能显著提升。

**⚠️ 局限性**

限制在于当前结果的参数依赖性尚未得到进一步的优化，且异步RVI Q学习的收敛速率仍然是未来研究的开放方向。

---

## 332. Which Objects Stay? Learning Hierarchical Persistence Beliefs for Lifelong Open-Vocabulary Object Maps

**arXiv ID:** 2611.07662 | [PDF](https://arxiv.org/pdf/2611.07662v1)

**作者:** Thanh Nguyen Canh `[一作]` (Japan Advanced Institute of Science and Technology), Nak Young Chong `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了HiPer，一种层次化的持久性过滤器，用于终身对象地图中估计对象在离开视野后是否仍留在原位；

**💡 创新点**

创新点在于：①不依赖语义类别，而是通过检索相似外观和几何的训练实例直接估计存活先验；②为每个对象维护可更新的Beta后验，支持在线学习和迁移迁移时保留移动模式；③将持久性关联到对象身份而非固定位置，实现对象级移动学习；

**🔧 技术方法**

使用了基于DINOv2和CLIP的特征检索、指数生存模型、Rosen的持久性框架、Beta更新闭式推断、环境速率自适应、Hungarian关联与射线投射可见性判断；

**📊 数据集**

在AI2-THOR的THOR-Lifelong仿真序列（120条）和3RScan真实扫描（47个环境）以及ObChange小规模实验上进行评估；

**📈 对比分析**

与Rosen的持久性过滤器、ELite、POCD、PF(λ)、Fuse-then-decay、永不忘记等基线进行比较，HiPer在NLL、Brier、ECE、AUROC等指标上均优于所有基线；在THOR中NLL最低为0.265（无校准），在3RScan中NLL最低为0.310；

**⚠️ 局限性**

局限性包括：依赖Oracle分割与跟踪，真实场景中缺乏长时间观测；在高频随机重排的ObChange中表现不佳；部分方法（如对象级学习、迁移）在数据不足时无显著提升。

---

## 333. WaSP: Wasserstein Data Synthesis Pipeline for Solver-Free Operator Learning

**arXiv ID:** 2611.07646 | [PDF](https://arxiv.org/pdf/2611.07646v1)

**作者:** Ryan Liu `[一作]` (California Institute of Technology), Anima Anandkumar `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于Wasserstein插值的无求解器训练框架WaSP，利用制造数据生成输入-解对并逐步沿Wasserstein路径进行自监督训练，解决了传统物理信息学习与制造数据方法的缺陷。

**💡 创新点**

创新点在于：①通过Wasserstein插值构建逐步训练的分布路径，缓解制造数据的分布移位；②在此路径上使用自监督生成的解对，并加入局部PDE残差正则化，避免残差方差放大和确认偏差；③给出误差传播的理论界限与自适应调度策略。

**🔧 技术方法**

使用技术包括：Fourier Neural Operator（FNO）架构；制造数据与PDE算子交替生成训练对；Entropy-regularized optimal transport（Sinkhorn）估计运输映射；McCann Wasserstein插值；动态归一化、回放缓冲、局部PDE残差正则化；SGD优化。

**📊 数据集**

数据集为三种一维周期演化PDE：Allen–Cahn、Kuramoto–Sivashinsky、Complex Ginzburg–Landau；实验仅使用无标注目标分布的输入采样，不使用求解器产生的目标解标签。

**📈 对比分析**

与传统PINN、PINO+OT、制造数据+PINO及纯制造数据进行对比；在相对L2误差上，WaSP比PINO低约3.9–9.5倍，且在不变量分布、空间/时间自相关、能谱等统计量上匹配或优于其他方法。

**⚠️ 局限性**

局限性包括：需要手工设计制造分布；仅在一维周期问题上验证，尚未证明在更高维、复杂几何下的可扩展性；计算成本仍高，尤其是Wasserstein估计与自监督生成；对非周期边界或复杂物理过程的适用性待进一步研究。

---

## 334. AiXapply: Fast and Reliable Full-File Code Integration with Specialized Small Models for IDE Workflows

**arXiv ID:** 2611.07634 | [PDF](https://arxiv.org/pdf/2611.07634v1)

**作者:** Siyuan Jiang `[一作]` (aiXcoder), Ge Li `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将代码整合（Apply）任务单独定义，构建高质量多语言数据集，并训练专用4B模型实现完整文件的合并。

**💡 创新点**

创新点包括：① 将代码整合视为独立任务，避免 diff 语法与匹配不稳问题；② 通过多阶段验证与可解性过滤构造可信数据；③ 采用 SFT+GRPO RL 与错误类型奖励提升模型对局部更新的精确性；④ 结合 Speculative Decoding 大幅降低推理延迟。

**🔧 技术方法**

技术手段：Qwen3‑4B 4B 语言模型，监督微调 + Group Relative Policy Optimization (GRPO)，语言感知等价比较，错误‑aware dense 奖励，n‑gram 与 suffix 方案的 Speculative Decoding。

**📊 数据集**

数据集来源：CommitPack 的真实提交，经过规则筛选、LLM 描述转化、更新片段生成、完整文件推导、一致性验证与可解性过滤，最终得到约21k条高质量三元组，划分训练/测试。

**📈 对比分析**

对比基线：DeepSeek‑V3.2、GLM‑5、Qwen‑3.5‑397B‑A17B、Fast‑Apply‑7B 等；aiXapply‑RL 在多语言 20 种语境下平均准确率 94.4%（SFT 94.4%），与 4B 大模型相近；Speculative Decoding 将推理延迟从 28.8s 降至 1.45s，部署后平均 Apply 延迟从 50s 缩至 1.89s。

**⚠️ 局限性**

局限性：对极长文件（>32k token）和多文件/跨文件编辑的鲁棒性不足；在长上下文场景下准确率显著下降；模型仅支持单文件整合，需外部机制处理文件间依赖。

---

## 335. Online Learning with Ordered Experts (via Two-point Feedback)

**arXiv ID:** 2611.07629 | [PDF](https://arxiv.org/pdf/2611.07629v1)

**作者:** Shiliang Zuo `[一作]` `[通讯]` (Meta), Shiliang Zuo (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文提出一种在有序行动空间中，利用两点反馈进行专家建议在线学习的框架，并给出了分离单调问题和有界变差（BV）问题的最优渐进 regret 上界与下界。

**💡 创新点**

创新点在于：① 构造了边界保持的“端点流估计器”，能够用两次观测实现全信息估计；② 证明两点反馈在有序环境下相较单点反馈可以将多项式系数降为对数级；③ 将该理论应用于动态定价，设计了兼容激励的两选项菜单，并实现接近全信息的收益退化率。

**🔧 技术方法**

主要技术：端点流估计器（随机路径 + 可逆配对采样）、指数加权更新、镜像下降结合负熵与对数障碍、Jordan 分解、对数化简与第二矩控制、信息论下界构造。

**📊 数据集**

没有使用实际数据集，全部为理论分析与证明；在定价应用中使用了理论模型的买家价值序列。

**📈 对比分析**

与单点反馈和全信息反馈对比：单点反馈在分离单调问题下退化到 Ω(min{√(K T), T^{2/3}})，在 BV 问题下退化到 Θ(√(K T))；全信息反馈在两类问题下均为 Θ(√(T log K))。两点反馈实现了 O(log(K T) √(T log K)) 与 (√T + K) 的上界，几乎匹配全信息的对数级差距，且比单点反馈多项式因子更优。

**⚠️ 局限性**

限制：① 结果仅适用于有界变差或分离单调结构，不能直接推广到任意连续损失；② 对于 BV 问题仍有 additive K 项，说明在某些实例中两点反馈不能完全消除专家数量的影响；③ 端点流估计器需要对齐分布的下界（π(j)≥ξ/m），导致额外的平滑与潜在的损失开销。

---

## 336. Beneath the Surface: What We Overlook When We Talk About Agent Evolution

**arXiv ID:** 2611.07624 | [PDF](https://arxiv.org/pdf/2611.07624v1)

**作者:** Xiaobo Wang `[一作]` (University of Science and Technology of China), Zilong Zheng `[通讯]` (Beijing Institute for General Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在保持模型权重不变的前提下，通过改变 agent harness（prompt、工具接口、代码等）实现递归自我改进（RSI），对四种演化方法在两种 harness（MiniSWE、Codex）和两类任务（通用任务 τ^3‑Banking、编码任务 Terminal‑Bench 2）上的表现进行系统评估，并提出了基于诊断的演化改进框架 EvoAudit 与其改进版 AGE；

**💡 创新点**

首次揭示 benchmark 得分与实际可用性不一致，提出了基于保留效果、成本与迁移的“保留意识”评估框架；通过分析演化过程中的父体选择、分析与突变机制，提出了针对性 operator 设计假设，并通过 EvoAudit 自动化生成四项 operator 修订，显著提升了可靠性、覆盖率并减少任务失效；

**🔧 技术方法**

使用 agent harness evolution 体系（包含 select、analyze、mutate、gate 等操作符），利用 GPT‑5.4 作为核心模型；评估指标包括 Avg@k、Pass@k、成本（C_dep@k）、任务转移与任务增失；实验中结合人类专家评估验证修订有效性；

**📊 数据集**

基准数据集包括 τ^3‑Banking（97 任务）、Terminal‑Bench 2（89 任务），并对 τ^2‑bench 与 DeepSWE v1.1 进行跨基准迁移测试；

**📈 对比分析**

与四种现有方法（AHE、HyperAgents、A‑Evolve、GEPA）在 MiniSWE/​Codex、通用/编码两类任务上进行对比；结果显示 AGE 在所有四种设置下均实现了最高的 Avg@8 与 Pass@8，且失去的种子覆盖任务最少，成本相对可接受；

**⚠️ 局限性**

主要局限包括：评估反馈有限且可能与实际行为不完全对齐，导致任务回归；演化改动在新任务或真实部署环境中可能产生意外副作用；迁移实验覆盖面有限，未验证更广泛的泛化能力；成本与可部署性仍需进一步优化；

---

## 337. Ordering Multivariate Time Series by Volatility Memory: A Temporal Convolutional Approach to Volatility Heredity

**arXiv ID:** 2611.07746 | [PDF](https://arxiv.org/pdf/2611.07746v1)

**作者:** Kaoutar El Azaar `[一作]` (Hassan II University), Mohamed Tabaa `[通讯]` (EMSI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 QS-TCN 方法，用以根据多变量时间序列的波动记忆对变量进行排序。

**💡 创新点**

创新点在于将量化的波动记忆（QML 持久性估计 + Matrix Pencil 时域模式）与结构化的 Temporal Convolutional Network 结合，生成可解释的双向方向分数并通过全局解码得到整体顺序。

**🔧 技术方法**

技术包括：准极大似然估计波动持久性、Matrix Pencil 提取多尺度指数模式、结构化卷积滤波器、残差比较分数函数、同步块引导自举、顺序解码器。

**📊 数据集**

使用的数据集包括：合成金融与 fMRI 参考数据（Finance、fMRI benchmark），以及实际金融市场数据（外汇/黄金面板、加密货币市场）。

**📈 对比分析**

通过与随机顺序、VAR‑LiNGAM 和外部 lead‑lag 评估方法的 Ordering Quality (OQ) 比较；在合成和基准数据上，QS‑TCN 的 OQ 通常高于对照方法，且在多种波动记忆情景下表现稳健。

**⚠️ 局限性**

局限性包括：无法恢复完整因果图；对波动记忆估计的误差影响结果；自举稳定性有限；在实际市场缺乏真实参考顺序，评估仅为描述性；未给出统计显著性检验。

---

## 338. Static Recovery Is Not Dynamic Stability: Dynamics-Aware Benchmarking of Protein Motif Scaffolding

**arXiv ID:** 2611.07723 | [PDF](https://arxiv.org/pdf/2611.07723v1)

**作者:** Emile de Bruyn `[一作]` (Forschungszentrum Jülich), Alina Bazarova `[通讯]` (Forschungszentrum Jülich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了新的基准BILBY-PRO和动态评估框架BILBY-MD，以系统评估蛋白质功能基序的构建和动态稳定性。

**💡 创新点**

①基于PROSITE结构保守基序构造大规模基准220个案例；②引入基于MD的动态验证，揭示静态成功与动态稳定性不一致；③将目标覆盖、唯一解产出与动态可信度三轴并列。

**🔧 技术方法**

使用生成模型（RFDiffusion、Proteina、FrameFlow、Genie2、DPLM2、ESM3）、ProteinMPNN+ESMFold结构预测、所有原子MD模拟以及RMSD、接触保持、PCA/TICA自由能分析等技术。

**📊 数据集**

基于PROSITE链接的实验结构（220个案例）、原MotifBench 30个手工案例，以及1 µs的MD参考轨迹。

**📈 对比分析**

在BILBY-PRO和MotifBench上对六种模型按覆盖率、MotifBench得分和MD支持率进行比较，RFDiffusion/Proteina覆盖率最高，ESM3唯一解最高；MD支持率仅约56%，接口型基序表现最差。

**⚠️ 局限性**

局限性包括：静态评估依赖ProteinMPNN+ESMFold预测的偏差；MD阈值经验性且计算成本高；BILBY-PRO受PROSITE序列模式限制，缺少断裂多段基序；模型训练集与基准可能存在重叠，影响结果独立性。

---

## 339. CAMP: Context-Aware Mapping and Placement for LLMs on 3D-IMC Arrays

**arXiv ID:** 2611.07707 | [PDF](https://arxiv.org/pdf/2611.07707v1)

**作者:** Orhun Görkem `[一作]` (IBM Research), William Andrew Simon `[通讯]` (IBM Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 CAMP 的上下文感知权重映射算法，专为 3D‑IMC 结构下的大语言模型推理设计，显著提升计算资源利用率和吞吐量。

**💡 创新点**

创新点在于：①使用冲突矩阵将体系结构、网络与运行时上下文的交互性压缩为可优化的抽象；②基于该矩阵先进行贪婪的初始映射，再通过模拟驱动的 beam‑search 精细化权重布局，兼顾张量并行、批量并行与专家并行。

**🔧 技术方法**

核心技术包括：动态工作负载分析与图形 IR 生成、系统级模拟求解时间重叠、冲突矩阵构建、贪婪映射、beam‑search 反馈循环与仿真。

**📊 数据集**

在 Llama‑3.1‑8B/70B、Qwen‑72B、Mixtral‑8×7B 三大主流 LLM 上进行实验，使用真实推理序列和 GPU‑加速的 KV 缓存。

**📈 对比分析**

与 Greedy‑Util、3D‑aCortex 及理想的 Full‑Compute 基线比较，CAMP 在 3D‑IMC 推理吞吐量上提升至 2.0×，在端到端推理吞吐量上提升至 1.74×，并将计算利用率从 42.4% 提升至 60.3%。

**⚠️ 局限性**

局限性包括：①算法仅为离线软件方案，需在部署前完成映射；②对大规模网络的完整映射计算量较大，需分层或分组求解；③评估基于模拟，缺乏实测硬件验证；④对 KV 缓存瓶颈的影响有限，需结合其他优化技术共同提升整体性能。

---

## 340. FlyingDrones: A Dataset and Benchmark for Optical Flow Estimation from UAV motion

**arXiv ID:** 2611.07705 | [PDF](https://arxiv.org/pdf/2611.07705v1)

**作者:** Laurenz Ruzicka `[一作]` (Austrian Institute of Technology), Daniel Cremers `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 FlyingDrones 数据集，专门评估无人机微小目标的光流性能；

**💡 创新点**

创新点在于提供高分辨率合成光流与分割标注，以及引入分辨率归一化 EPE 和目标区域评价指标；

**🔧 技术方法**

主要技术包括基于 Blender 的物理渲染、DenseFlow 提取、以及对 16 类光流模型（共 88 检查点）的统一评测；

**📊 数据集**

使用了 FlyingDrones 三种分辨率（480×270、960×432、1920×1080）以及 21 台实机摄像头的真实无人机视频；

**📈 对比分析**

对比方法通过分辨率归一化指标、目标区域 EPE 以及零流基准进行统计，结果显示全局匹配、变压器和短窗口时序模型在高分辨率下领先，且模型间差距随分辨率升高而扩大；

**⚠️ 局限性**

局限包括：数据集为合成场景，无法覆盖大气散射、传感器噪声等真实影响；评测仅聚焦光流质量，未考虑检测/跟踪等下游任务；以及不同训练课程对模型表现影响大，导致难以单纯归因于网络结构。

---

## 341. InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning

**arXiv ID:** 2611.07702 | [PDF](https://arxiv.org/pdf/2611.07702v1)

**作者:** Liangjian Wen `[一作]` (Southwestern University of Finance and Economics), Zhao Kang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 InfMasking 框架，在多模态学习中对每个模态的特征进行随机遮掩，并对所有可能的部分组合执行一致性约束，统一实现无监督对齐与监督标签代理两种学习目标。

**💡 创新点**

创新点在于：① 通过“无限遮掩”实现对无数子组合的期望约束；② 推导可解析的变分上下界并给出紧致性证明；③ 将同一框架同时应用于无监督和监督任务。

**🔧 技术方法**

采用 Token 级随机遮掩、Transformer/注意力融合、InfoNCE 对齐、标签代理交叉熵、Gaussian 近似与变分界定、PID 与 InterSHAP 评估等技术。

**📊 数据集**

使用的主要数据集包括合成 Trifeature、真实多模态数据集 MultiBench、MM-IMDb、MUStARD、MIMIC、MOSI/MOSEI、MAMI/MUTE 等。

**📈 对比分析**

与 Cross、Cross+Self、FactorCL、MAE、CoMM、MultiMAE、DCP 等基线在无监督线性探针和监督分类任务中对比，InfMasking 在大多数数据集上提升 1–5% 的准确率/ F1，且在联合信息和交互性指标上显著优于基线。

**⚠️ 局限性**

局限性包括：需要采样多次遮掩导致计算和内存成本增加；实验仅验证至三模态，未探讨更高模态数；变分近似依赖 Gaussian 假设，实际误差与可扩展性待进一步研究。

---

## 342. K-subareas partitioning subject to coinciding boundary segments of neighboring subareas

**arXiv ID:** 2611.07701 | [PDF](https://arxiv.org/pdf/2611.07701v1)

**作者:** Mogens Plessen `[一作]` `[通讯]` (Findklein GmbH), Mogens Plessen (Findklein GmbH)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种分层的面积划分方法，将给定非凸区域及障碍物划分为 K 个子区域，且相邻子区域的边界线段保持完全重合。

**💡 创新点**

创新点在于结合递归二分拆分、聚类合并与旅行商问题，确保边界重合并兼顾非凸形和障碍物处理，同时通过超参数 N_p 控制划分细化与计算复杂度。

**🔧 技术方法**

采用的技术包括基于点聚类的二分拆分、K-means 聚类合并、旅行商问题（TSP）求解以及几何轮廓重构。

**📊 数据集**

使用了真实户外地理数据，包括农业田块、湖泊和障碍区域的轮廓坐标。

**📈 对比分析**

通过实验比较不同 N_p 取值下子区域面积平衡度（e_max）和计算时间（τ），结果显示 N_p 较大时面积平衡好但运行时间显著增加。

**⚠️ 局限性**

主要局限是 TSP 求解的 NP-hard 性导致可能出现边界重叠，且过大的 N_p 会导致计算复杂度高、轮廓边缘锯齿化，需进一步平衡。

---

## 343. Flexible Piezoelectric Tactile Sensing System for Intelligent Object Recognition

**arXiv ID:** 2611.07779 | [PDF](https://arxiv.org/pdf/2611.07779v1)

**作者:** Razan Khalifeh `[一作]` (University of Genoa), Yahya Abbass `[通讯]` (University of Genoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计并验证了一套柔性PVDF压电触觉传感系统，可嵌入机器人抓手，实现抓取早期实时物体识别；通过六种日常物体的抓取实验收集触觉信号，比较SVM、1D‑CNN、C‑RNN、2D‑CNN热图四种识别方法，并系统评估传感器数量与时间窗口对识别性能的影响。

**💡 创新点**

创新点包括：①采用柔性PVDF阵列与可拆卸、可定制传感帽，提升柔性与多点覆盖；②将触觉信号转化为空间‑时间热图，用2D‑CNN提取特征，取得最高识别精度；③在ARM Cortex‑M4微控制器上实现端到端推理，延迟≈101 ms、能耗仅63.8 µJ；④详细研究传感器数与窗口大小对性能的影响，给出32传感器、100 ms窗口的最佳配置。

**🔧 技术方法**

使用的技术包括PVDF压电传感器阵列、柔性PCB与Dragon Skin封装；32通道ADC采集、ARM Cortex‑M0/M4嵌入式控制；特征提取（均值、STD、偏度、峰值等）与PCA降维；SVM、1D‑CNN、C‑RNN、2D‑CNN网络；热图构造与归一化；STM32CubeIDE实现模型部署。

**📊 数据集**

实验数据集由六个日常物体（玻璃瓶、电视遥控器、半装水瓶、草莓、硬苹果、半软苹果）组成，每个物体50次抓取，总计300条；采样率500 Hz，按窗口大小50、100、150、200对信号进行分段。

**📈 对比分析**

通过F1‑score、Precision、Recall等指标比较三种方法；SVM（特征）在32传感器时F1≈0.966；1D‑CNN/C‑RNN在原始信号时最高F1≈0.953；2D‑CNN热图在32传感器、100 ms窗口时达到最高F1=0.9903；在STM32上推理时间0.88 ms、能耗63.8 µJ，端到端延迟约101 ms。窗口越大性能下降；传感器从16到32提升有限。

**⚠️ 局限性**

局限性包括：仅测试六种物体、单一TIAGo机器人平台；抓取动作固定为抓取‑保持‑释放；未评估不同抓取策略、滑移、复杂环境；对更大尺寸、形状、材质多样性的泛化能力仍待验证。

---

## 344. BiGrounder: Bidirectional Training Unlocks the Potential of GUI Grounding

**arXiv ID:** 2611.07765 | [PDF](https://arxiv.org/pdf/2611.07765v1)

**作者:** Dengyao Shi `[一作]` (MAIS, Institute of Automation, Chinese Academy of Sciences), Zhen Lei `[通讯]` (MAIS, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BiGrounder框架，利用双向奖励循环将模型自身的指令变体作为强化学习轨迹，实现GUI指令到坐标的自我强化

**💡 创新点**

创新点在于将逆向指令生成与正向定位耦合的循环一致性奖励，并使用语义一致性门控，构建基于模型自身变体的在线探索机制

**🔧 技术方法**

结合监督微调+强化学习，使用GRPO、BERTScore语义一致性、稠密bbox奖励、动作-坐标奖励及Group Relative Policy Optimization实现双向策略优化

**📊 数据集**

在SFT阶段采用ScaleCUA和GroundCUA的数据集，随后在五大基准上评测：ScreenSpot-Pro、ScreenSpot-V2、MMBench-GUI L2、UI-Vision、OSWorld-G-R；并使用OSWorld-G-R-Variants和GPT‑4o生成的规划指令进行在线测试

**📈 对比分析**

与现有Qwen3‑VL‑4B等基线比较，BiGrounder-4B在所有基准上取得竞争性最高准确率（如ScreenSpot-Pro 66.6%，MMBench-GUI L2 87.5%），在指令变体鲁棒性测试中将Any‑All差距缩小至14.8点，显著提升鲁棒性

**⚠️ 局限性**

局限在于对较小规模模型的依赖，奖励门控仍可能导致样本效率下降；仍需在更大模型、更多语言或更稀缺UI场景中验证；训练成本高，且对BERTScore阈值与像素容差的选择敏感

---

## 345. ReConMem: A Graph-Based Dual Memory System for Repository-Oriented Conversations

**arXiv ID:** 2611.07759 | [PDF](https://arxiv.org/pdf/2611.07759v1)

**作者:** Zhishuo Yang `[一作]` (Beihang University), Fang Liu `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个双图记忆系统（ReConMem），通过语义记忆图和结构记忆图联合管理对话历史与代码仓库上下文，并在每轮对话时通过双重检索机制选择最相关的信息来辅助大型语言模型生成回答。

**💡 创新点**

创新点在于：①构建并维护“双图记忆图”，显式关联自然语言对话记忆与代码实体；②结合代码草稿检索与图邻接扩展的联合检索策略；③针对代码实体的分层压缩方法，兼顾上下文完整性与 token 长度。

**🔧 技术方法**

技术手段包括：图结构记忆（semantic & structural graph）、LLM驱动的记忆更新与双向同步、代码草稿对齐检索、图邻接扩展检索、分层代码压缩、以及基于 embedding 的相似度检索。

**📊 数据集**

使用的数据集为 LoCoEval（Python 长篇对话+代码生成）与 CodeAssistBench（多语言 GitHub issue 交互），分别评估长时延续与实际项目交互场景。

**📈 对比分析**

与基线（Full History、Vanilla RAG、MemGPT、LD-Agent、Mem0、Mem0^ℛ 等）相比，ReConMem 在 LoCoEval 单跳/多跳 Pass@1 提升约 7.9~15.6pp，保持相近压缩率；在 CodeAssistBench 上 Correct/Partially Correct 提升 4.1~7.1pp，Appropriate 提升至 67.5%，并在平均对话轮次上略低，表明效果显著且交互更高效。

**⚠️ 局限性**

局限性包括：①额外的记忆管理开销在短轮对话中可能未被完全抵消；②评估依赖 LLM 生成与判定，可能带来主观性；③仅在现有两个 benchmark 上验证，缺乏对更大规模、跨仓库、跨语言长周期交互的泛化评估。

---

## 346. RIVER: Evidence-guided multi-agent retrieval for enzyme--reaction discovery

**arXiv ID:** 2611.07757 | [PDF](https://arxiv.org/pdf/2611.07757v1)

**作者:** Yifan Chang `[一作]` (Shanghai Innovation Institute), Yuchen Ren `[通讯]` (Shanghai AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 RIVER 的多代理框架，用于双向酶–反应检索，能给定酶时检索潜在反应，或给定反应时检索候选酶，并为每个候选给出可追踪的证据链。

**💡 创新点**

创新点在于将语言模型、外部数据库检索和专门的生化工具协同工作，形成“证据引导”的预筛选和反思重排序两阶段流程，强调证据的可追溯性和冲突检测，而非单纯的相似性匹配或训练模型。

**🔧 技术方法**

使用大型语言模型（如 Qwen3.6-35B、GPT‑5.6 Luna）、UniProt、Rhea、BRENDA、Pfam、HMMER、M‑CSA 等数据库的检索，结合自定义的 ReactionAnalysis、EnzymeAnnotate、Filter、Reconsider、Contradiction、BioChem Expert 等代理，并实现在线工具调用与多轮对话推理。

**📊 数据集**

利用 667 条 E. coli ASKA 酶候选集作为回溯实验数据集，公开基准包括 Enzyme‑405、Orphan‑335、P450、Terpene、Phosphatase 五个数据集。

**📈 对比分析**

与基线方法（MMseqs2、Selenzyme、ESP、CLIPzyme、EnzymeCAGE、Phylor Max、Claude Code 等）对比，RIVER 在 Enzyme‑405 上 Top‑1 率 82.3%、MRR 0.846，Orphan‑335 上 Top‑1 率 72.8%、MRR 0.708；在家族特异性数据集上 Top‑1% 成功率分别提升 19.8%、4.1%、61.5%，整体表现明显优于现有方法。

**⚠️ 局限性**

局限性包括：依赖外部数据库与网络访问；未经过训练，性能受模型先验知识与检索策略限制；对罕见或无注释酶的预测仍可能缺乏充分证据；对实验验证的依赖较大，排名不等同于生理活性或催化效率。

---

## 347. Automated Video-to-QA Pipeline: End-to-End Question Generation and Verification Using Large Language Models

**arXiv ID:** 2611.07755 | [PDF](https://arxiv.org/pdf/2611.07755v1)

**作者:** Jaival Chauhan `[一作]` (Sardar Vallabhbhai National Institute of Technology), Pruthwik Mishra `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套端到端的自动化流水线，能够从YouTube视频中提取音频、转录为文本、分块摘要、生成并自我验证问答对，并用LoRA对Llama模型进行微调，以输出高质量的教育性题目和摘要。

**💡 创新点**

创新点在于将LLM同时用作生成者和评审者的闭环自校验机制，能够自动过滤幻觉问题，并利用生成的高质量合成数据进行低秩适配（LoRA）微调，避免手工标注。

**🔧 技术方法**

使用的技术包括FFmpeg/yt‑dlp提取音频、Whisper进行ASR、BART进行抽象式摘要、meta‑llama/Llama‑3.1‑8B‑Instruct进行问题生成与评审、LoRA进行参数高效微调；整个流程基于Python实现。

**📊 数据集**

数据集为从YouTube获取的技术教育类视频（约20–30分钟），使用Whisper生成的原始转录文本进行分块处理；实验中还随机抽取25个QA对进行人工评估。

**📈 对比分析**

通过对比基线Llama‑3.1的零样本问答生成与加入自校验及LoRA微调后的模型，人工评分显示相关性与真实性均提升至平均4.2/5；在问答正确率、误幻率等指标上，微调模型显著低于基线。

**⚠️ 局限性**

局限性包括仅利用音频转录信息，未整合视频视觉内容；只支持英文，跨语言能力需进一步扩展；实时处理效率仍有限，无法即时生成直播问答。

---

## 348. Adjustment-Anchor Dependence in Formulaic Alpha Languages: A Certified Grammar and a Re-Anchoring Audit

**arXiv ID:** 2611.07751 | [PDF](https://arxiv.org/pdf/2611.07751v1)

**作者:** Zhengkai Tu `[一作]` (Chinese University of Hong Kong), Jimmy Huang `[通讯]` (York University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对公式式alpha语言中的调整锚依赖问题，提出了一套双排序类型系统用于静态判定公式对锚的不可变性。

**💡 创新点**

创新点在于将锚依赖视为每只股票的标量缩放组作用，设计了可在理想化算子语义下验证的两种类型（乘法与加法），并提供线性时间检查器。

**🔧 技术方法**

使用了类型检查、GFlowNet、强化学习与遗传程序等多种生成器，以及数值重锚测试和符号归约等技术。

**📊 数据集**

实验数据主要来自Qlib的中国A股CSI300/CSI500历史行情（2010-2026）和美国标普500公开快照。

**📈 对比分析**

通过冻结阶段协议与多种基准（未受限搜索、粗粒度约束、随机裁剪等）比较，证明已认证池在锚变动下的IC漂移几乎为0，而传统池漂移约14%；在预测上差异低于设计分辨率，表明约束成本可忽略。

**⚠️ 局限性**

局限包括依赖于理想化算子语义、对字段合同和分拆/股息调整方式的假设、仅覆盖单一市场与数据源、以及对浮点精度和缺失值处理的敏感性。

---

## 349. ConvPE: Convolutional Position Embedding Improves Transformers

**arXiv ID:** 2611.07749 | [PDF](https://arxiv.org/pdf/2611.07749v1)

**作者:** Zehao Wang `[一作]` `[通讯]` (Nanjing University), Zehao Wang (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种卷积式注意力得分偏置位置嵌入 ConvPE，作为轻量级插件，可在语言建模、长上下文扩展和视觉 Transformer 中使用。

**💡 创新点**

创新点在于通过对低维投影的隐藏状态进行因果卷积生成局部、可学习的偏置，兼容 RoPE、mask、KV 拼接，并支持多维卷积，显著提升了模型的精度与效率。

**🔧 技术方法**

使用了低维投影、因果一维/二维卷积、Flash MQA/TileLang 加速、RoPE、ALiBi、CABLE、DAPE V2 等技术；在实验中对比了多种位置编码方式。

**📊 数据集**

主要数据集包括 WikiText‑103、FineWeb‑Edu‑10BT、CIFAR‑10/100、RULER 等，用于预训练、零样本评估、长上下文验证和视觉任务。

**📈 对比分析**

与 RoPE、ALiBi、CABLE、DAPE V2 等方法比较，ConvPE 在预训练困惑度、零样本准确率、8K 长上下文 perplexity 以及 CIFAR‑100 准确率上均表现更优，且参数增长约 2–3%，每步时间增长仅约 2–3%。

**⚠️ 局限性**

局限性包括仅在单一随机种子下测试、模型规模不超过 167M、最大上下文长度仅 8K，未对更大规模模型或更长上下文的可扩展性进行验证。

---

## 350. Fix One, Miss Another: Leveraging Historical Bug Fixes to Uncover Residual Bugs in Database Systems

**arXiv ID:** 2611.07739 | [PDF](https://arxiv.org/pdf/2611.07739v1)

**作者:** Yankai Rong `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Reaper 框架，利用历史 bug 修复中的语义信息，对数据库管理系统（DBMS）的语法兄弟处理器进行系统性检测，发现未被修复的残留缺陷。

**💡 创新点**

创新点在于将历史修复转化为可重用的 Bug Profile（触发与验证），并通过语法引导的兄弟变异与分阶段的 LLM 验证，实现从单个修复到全族处理器的自动化缺陷迁移。

**🔧 技术方法**

技术主要包括：多代理 LLM（Query Analyzer、Patch Analyzer、Orchestrator）提取 Bug Profile；语法引导的兄弟变异与触发约束化 concretization；基于验证 Profile 的三阶段 LLM 判定（匹配失效模式、检查修复适用性、验证或acles）。

**📊 数据集**

使用了 4,487 条历史 bug 修复数据集，覆盖 MySQL、MariaDB、OceanBase、CockroachDB 四大主流 DBMS。

**📈 对比分析**

与 SQLancer、Squirrel、Argus 三种主流 DBMS fuzzers 进行 8 小时对比，Reaper 在 MySQL、MariaDB、OceanBase 的独立 bug 数量分别为 7、0、1，覆盖率提升 13%–23%，语义有效率高达 86%–88%（相比基线约 73%–63%）。

**⚠️ 局限性**

限制包括：依赖 LLM 的推理准确性；对某些 DBMS（如 CockroachDB）覆盖测度受限；验证过程较为耗时，且对无法恢复的修复或缺乏完整语义描述的缺陷不易适用。

---

## 351. Extending Reasoning in Typed Decision Models via Looping

**arXiv ID:** 2611.07736 | [PDF](https://arxiv.org/pdf/2611.07736v1)

**作者:** David Berghaus `[一作]` (Lamarr Institute for Machine Learning and Artificial Intelligence), Volkan Cevher `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了在已预训练的决策模型中加入可训练的循环模块（RELOAD），通过在中间层块上做多次迭代并使用轻量级控制器实现按需计算。

**💡 创新点**

创新点在于：①循环训练与决策模型一起进行，而非仅在推理时循环；②使用阻尼的Euler更新实现可训练的ODE轨迹；③单模型即可在不同预算（1-4次循环）下工作；④引入基于前缀输出的控制器实现请求级自适应深度。

**🔧 技术方法**

技术包括：Transformer（编码器/解码器）、阻尼循环（G(h)+1/K(G(h)-h)）、ODE视角的Euler与Midpoint积分、交叉熵+Brier损失、逻辑回归控制器、批量训练在多预算上、基准数据集混合训练。

**📊 数据集**

使用的主要数据集：ProofWriter、StepGame（1–10条语句）、Kev决策、Kev迁移、ParaRules、JustLogic、ProverQA、BoardgameQA、MuSR、ReClor、LogiQA 2.0、ReSQ 等；训练集包括公开分类、问答、自然语言推理、合成多步推理等。

**📈 对比分析**

与单步训练的基准模型进行比较，四次循环模型在所有组（内分布、长推理、离群）均提升准确率（尤其长推理提升10–15%）并显著降低Brier分数；与等价计算量的深度扩展模型相比，仅在几乎两倍参数下能逼近；使用控制器后，计算量约为四步的3/4，精度几乎不减。

**⚠️ 局限性**

局限性包括：对因果解码器的提升有限；对知识密集型选择题提升不明显；模型规模有限（最高4B），对更长自然语言推理的通用性未知；控制器主要依赖输入长度，未能精准捕捉推理深度；训练循环需要在多预算下训练，增加训练复杂度。

---

## 352. Pediatric ECG foundation model via adult-to-pediatric fine-tuning: downstream tasks and clinical evaluation

**arXiv ID:** 2611.07717 | [PDF](https://arxiv.org/pdf/2611.07717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 353. Seeing Beyond the Next Step: World-Model-Guided Human-Like Navigation in Multi-Agent Scenes

**arXiv ID:** 2611.07730 | [PDF](https://arxiv.org/pdf/2611.07730v1)

**作者:** Seong-Eun Hong `[一作]` (Korea University), HyeongYeop Kang `[通讯]` (Korea University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在训练阶段利用世界模型作为偏好判别器的导航策略，通过先学习可执行的运动词典，再用双分支世界模型（社交动态预测和语义场景预测）对候选动作进行评分，并通过群体相对策略优化（GRPO）将评分信息蒸馏到最终策略，实现无滚动推理、单前向传播即可完成导航。

**💡 创新点**

创新点包括：① 将世界模型迁移到训练阶段而非推理阶段，实现推理时的实时性；② 将未来预测拆分为社交动态和语义布局两支，避免单一生成模型的冗余；③ 通过偏好蒸馏将前向预测能力“摊平”到策略网络，兼具安全、社交合规与目标导向；④ 在六个ETH/UCY场景上验证，展示在目标成功率、违规率和组内距离上的显著提升。

**🔧 技术方法**

使用技术包括：预训练多模态视觉语言模型（VLM）用于动作优先级学习；K-means聚类构建运动词典；两条专用世界模型——基于状态转移的社交动态预测和基于语义分割的场景预测；Group Relative Policy Optimization (GRPO) 用于偏好蒸馏；训练时使用的前向传播仅包含动作分类器。

**📊 数据集**

训练与评估数据集主要来自ETH/UCY人群轨迹数据集（ETH_Hotel、Zara01、ETH、Students003），在Unity重建的模拟环境中进行。评估使用了Zara01、Students003、ETH_Hallway、ETH_Intersection以及Zara01_Group等五个独立场景。

**📈 对比分析**

与GBM、CCP、CEDRL等基线以及在线世界模型规划器（OnlineWM）进行比较。实验结果表明，在所有评估场景中，本方法的目标成功率最高，违规率最低，组内距离均优于或与最佳基线相当。用户研究显示，生成轨迹在运动质量、自然度和流畅度上获得最高分。推理时间仅为单前向传播（约65 ms），相比在线规划器节省了两三百倍。

**⚠️ 局限性**

局限性包括：① 只能在固定的100个短期运动词典内选动作，无法覆盖更长时间或全局规划需求；② 仅做一次动作条件预测，无法捕捉延迟影响或多步交互；③ 两个世界模型的权重为固定，缺乏动态仲裁机制；④ 仅在模拟环境验证，未在真实机器人上测试；⑤ 训练成本较高，需大规模预训练与多阶段优化。

---

## 354. Dual-Arm Tabletop Rearrangement with Movable Stacking and Gap-Aware Push Transfers

**arXiv ID:** 2611.07726 | [PDF](https://arxiv.org/pdf/2611.07726v1)

**作者:** Mohammadreza Firoozalizadeh `[一作]` (University of Tehran), Mehdi Tale Masouleh `[通讯]` (University of Tehran)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种双臂桌面重排框架，将可移动堆叠与Gap‑Aware推送传输相结合，实现跨界异步传输。

**💡 创新点**

核心创新在于：①动态在共享边界上寻找空闲推送间隙，取代固定交接点；②在MCTS搜索中同时考虑堆叠与跨界传输；③利用执行成本评估计划质量。

**🔧 技术方法**

使用联合蒙特卡罗树搜索（Joint MCTS）规划，动作集包括move、stack、push、wait；推送传输分为预对齐、直推、回撤三阶段；执行成本计算包括机械臂运动、抓取与推送代价。

**📊 数据集**

数据集为在1×2 m桌面上随机生成的5–12个异形物体的起始与目标场景（20组随机对），不使用公开物理实验数据。

**📈 对比分析**

与固定交接点基线同一MCTS框架对比，评估指标为Joint Steps和Execution Cost。实验表明，在N≥7时，Gap‑Aware推送在两指标上均优于固定交接，尤其在N=12时Joint Steps下降≈10%，Execution Cost下降≈5%。

**⚠️ 局限性**

局限性：仅在仿真环境验证，未考虑真实机器人抓取误差与动态障碍；对共享边界的推送假设物体可完全沿直线推送，未考虑复杂形状或软体物体。

---

## 355. Multimodal Information Interactions Bottleneck for Multimodal Self-Supervised Learning

**arXiv ID:** 2611.07718 | [PDF](https://arxiv.org/pdf/2611.07718v1)

**作者:** Liangjian Wen `[一作]` (Southwestern University of Finance and Economics), Zhao Kang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种多模态自监督学习框架 MIIB，利用信息瓶颈与部分信息分解（PID）显式建模冗余、唯一与协同信息，实现最小充分表示；

**💡 创新点**

提出双路径 Transformer 结构与基于熵的优化目标，显式解耦冗余、唯一与协同交互，并通过自监督方式实现三种信息的最小充分表示；

**🔧 技术方法**

融合信息瓶颈理论、PID、熵/条件熵估计、Score‑based von Mises‑Fisher 熵估计、对齐‑均匀性正则化以及多视角对比学习等技术；

**📊 数据集**

在合成 Trifeature、MultiBench（MIMIC、MOSI、UR‑FUNNY、MUSTARD、Vision&Touch）以及 MM‑IMDb 等多模态基准上进行评估；

**📈 对比分析**

与 Cross、Cross+Self、FactorCL、CoMM 等基线在线性探针和全参数微调下对比，MIIB 在所有数据集均取得最佳或相近性能，平均提升约 1–3% 以上；

**⚠️ 局限性**

主要局限在于针对静态模态，模型规模略增，未涵盖动态视频等时序模态，未来需进一步扩展到更大规模和动态模态。

---

## 356. AffectiveVLA: Safe and Intent-Aware VLA Control for Human-Humanoid Social Interaction

**arXiv ID:** 2611.07712 | [PDF](https://arxiv.org/pdf/2611.07712v1)

**作者:** Nikita Kuzmin `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在Unitree G1 humanoid 上实现了 AffectiveVLA 系统，结合视觉‑语言‑动作管线与意图识别，完成了握手、拳击、拥抱等安全、意图感知的社交触碰交互。

**💡 创新点**

创新点在于将双目 RGB 意图检测与 VLA 动作生成相结合，并通过任务特定阻尼控制与语音安全停机，实现低延迟、可靠的社交交互。

**🔧 技术方法**

技术包括 TinyViT+VideoLightFormer 轻量级视频分类器、GR00T VLA 模型、GEAR‑SONIC 与 BrainCo 手部控制，以及 Vosk 语音识别。

**📊 数据集**

使用了约 1,270 条手势交互数据集（19 名参与者、4 种交互类），并在此基础上对四个 VLA 模型进行微调。

**📈 对比分析**

通过离线 RMSE 与在线机器人评估对四个 VLA 模型进行比较，GR00T+意图模块宏 F1 从 0.75 提升至 0.90，拳击交互准确率提升至 0.78，整体表现最佳。

**⚠️ 局限性**

局限性包括样本量有限、仅覆盖三种预设交互、单一机器人平台、对长时间无交互时误触发缺乏评估、语音识别在嘈杂环境下可靠性待验证。

---

## 357. Fast and Accurate Hand Estimation in Pixels, Interaction and World Space

**arXiv ID:** 2611.07711 | [PDF](https://arxiv.org/pdf/2611.07711v1)

**作者:** Jinkun Cao `[一作]` `[通讯]`, Jinkun Cao

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种实时手部网格估计框架（FAH），通过先预测对应的2D/3D手部点，再单独对MANO或MHR-Hand网格进行拟合，并支持部件标签与接触预测，能够在视频中通过因果细化实现平滑的世界空间跟踪。

**💡 创新点**

创新点包括：①直接预测3D点而非仅回归参数，随后单独网格拟合能显著提升网格精度；②同时输出2D与3D点，并通过热图头与密集点实现更好像素对齐；③引入部件与接触头，能够在无物体几何的条件下完成接触预测；④使用因果Transformer细化与实时相机估计，使模型既保持高速又支持世界空间定位。

**🔧 技术方法**

技术实现上：基于SAM 3D Body手分支的ViT（ViT-L/ViT-H+）编码器，替换其解码器与rig头为热图密集点头；使用密集点拟合MANO/MHR-Hand的Horn算法+学习器；部件与接触头利用Transformer与交叉注意力；视频时序细化采用4层因果Transformer+One Euro滤波；相机估计结合DPVO、RAFT与DROID‑SLAM的中值融合。

**📊 数据集**

训练与评估使用了多域数据集，包括HOT3D、EgoStandard、bench2d v2、HO-3D、DexYCB、HOI4D、ARCTIC、InterHand2.6M、OakInk2、WiLoR、HaWoR等，融合2D关键点、3D关节、手骨点与接触标签。

**📈 对比分析**

与MediaPipe、HaMeR、Hamba、WiLoR、SAM 3D Body、HaWoR等基线对比，FAH在HOT3D和EgoStandard的PA‑MPJPE降至<5 mm，2D PCK@0.1提升至0.83，接触F1提升至0.76–0.81，世界空间误差明显下降；在RTX 4070 Ti上，ViT‑L实现65 fps（含网格拟合），ViT‑H+为49 fps，ViT‑S在无网格拟合时可达125 fps，满足交互级实时需求。

**⚠️ 局限性**

主要局限包括：相机估计仍是世界空间误差的主导因素；无物体几何的接触预测存在模糊性，部分接触点被遮挡；为保持实时性能，模型在最强精度上略有折衷，进一步提升精度可考虑更强分割或光流模型，但需权衡计算量。

---

## 358. Distributed Optimization Involving Non-Lipschitzian, Linear Composite Functions: Unified Algorithm Design and Convergence Analysis

**arXiv ID:** 2611.07703 | [PDF](https://arxiv.org/pdf/2611.07703v1)

**作者:** Liang Ran `[一作]` (Southwest University), Run Tang `[通讯]` (Southwest University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种全分布式的非光滑凸优化算法（FDOA），可处理带有边缘等式约束、局部非线性不等式约束以及全局耦合线性/非线性约束的问题，目标是最小化各智能体局部非可导凸函数与线性映射下的复合非光滑项之和。

**💡 创新点**

创新点在于：①通过共轭分解推导出统一的边缘本地一阶条件，消除了对拉普拉斯矩阵的依赖；②利用近端投影与Moreau分解对多重非光滑复合项和耦合约束进行分离，允许在不需要解析近端映射的情况下实现完全分布式迭代；③在弱假设下（无Lipschitz、无子梯度有界、无强凸性）得到o(1/k)的收敛率；④支持局部步长和容错内外误差。

**🔧 技术方法**

主要技术包括共轭分解、变分不等式分析、近端投影与Moreau分解、可变子空间投影、内部错误容忍、局部自适应步长以及利用迭代近端方法求解局部子问题。

**📊 数据集**

实验采用两类典型案例：直流动力学最优潮流（DC‑DOPF）和多车队形控制（MVFS），并在随机生成的20节点、5维状态、随机连通网络上进行数值验证。

**📈 对比分析**

与分布式ADMM、补偿算法和并行算法等现有方法相比，实验结果显示FDOA在解收敛速度、目标误差以及约束违背上均优于对手，并且在采用容错内外误差时能够显著降低计算成本。

**⚠️ 局限性**

局限性包括：仅在无向连通图上得到理论保证，难以直接推广至有向或不平衡网络；对全局耦合约束的内子问题求解仍可能产生较高的计算开销；以及需要满足一定的步长和参数设定才能保证收敛。

---

## 359. PBPO: Prefix-Block Policy Optimization for Diffusion Language Models

**arXiv ID:** 2611.07783 | [PDF](https://arxiv.org/pdf/2611.07783v1)

**作者:** Katherine Tieu `[一作]`, Jingrui He `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种 Prompt‑Based Black‑box Programmatic Optimization (PBPO) 框架，用来自动优化大语言模型的提示词，从而提升在数学推理和编程生成任务上的表现。

**💡 创新点**

创新点在于将提示词视为可优化的程序，通过黑盒评估和自动搜索（如遗传算法或贝叶斯优化）迭代生成最优提示，从而显著提高模型输出的正确率，且不需要模型参数微调。

**🔧 技术方法**

技术包括：提示词的自动生成与组合、黑盒评估函数（通过校验输出与答案是否匹配），以及基于评价结果的搜索优化算法；实现上使用了 LLaDA‑1.5/8B-Instruct 作为基础模型。

**📊 数据集**

使用的数据集为：MATH500（500 道数学推理题）和 MBPP（编程练习集）。

**📈 对比分析**

实验通过与原始模型（仅用手工提示）直接对比，评估准确率和正确率。结果显示：在 MATH500 上，PBPO 提升了约 15–20% 的正确率；在 MBPP 上，正确率提升约 10–12%。

**⚠️ 局限性**

局限性包括：需要多次评估和提示搜索，导致计算成本显著增加；对极长或多轮交互式推理的泛化能力尚待验证；以及依赖于评估器对答案的严格匹配，可能对开放式回答任务不适用。

---

## 360. Reliability-Aware Z-Grammars and Z-Automata over a Credible-Grade Semiring

**arXiv ID:** 2611.07781 | [PDF](https://arxiv.org/pdf/2611.07781v1)

**作者:** Sherzod Turaev `[一作]` (United Arab Emirates University), Wan Heng Fong `[通讯]` (Universiti Teknologi Malaysia)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了加权文法和有限自动机，提出了一种新的加权结构，结合了每个单词的等级和可靠性，形成了基于Z数的加权文法和自动机。

**💡 创新点**

创新点在于引入了Z-权重结构，将等级和可靠性结合在一起，形成了一个新的选择性半环，能够在文法推导中保持两者的关联性。

**🔧 技术方法**

使用了基于Z数的加权结构，定义了Z-文法和Z-自动机，并证明了其半环性质。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到了一些基于评估和证据的示例，可能涉及到实际的评估数据。

**📈 对比分析**

通过比较不同的组合规则，识别出在特定载体上满足半环法则的结构，证明了正可信等级的切割和选择运行的可靠性过滤器是正则且可有效构造的。

**⚠️ 局限性**

限制在于对任意模糊值Z数的扩展仍然是开放的，且在某些情况下，等级和可靠性之间的关系可能导致不一致性。

---

## 361. Uncovering Systematic Blind Spots in Graph Database Testing through Integration and Process-Level Analysis

**arXiv ID:** 2611.07754 | [PDF](https://arxiv.org/pdf/2611.07754v1)

**作者:** Tinghan Li `[一作]` (Chinese University of Hong Kong), Pinjia He `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究将八种现有图数据库自动化测试方法集成到统一平台，并在12小时内分别对Neo4j、Memgraph、FalkorDB和新系统NeuG执行测试，结合过程级数据分析发现并解决两个盲点：误报/重复报告过多及元变换因优化导致失效；随后设计并实现了基于多代理LLM的后处理流水线和计划感知分流机制，最终在NeuG上发现19个此前未知的bug。

**💡 创新点**

创新点包括①首次从过程层面系统分析整合测试方法的行为，揭示误报与重复报告的根本原因；②提出“同计划”检测与排除策略，解决元变换因查询优化器而失效的问题；③设计多代理LLM驱动的自动报告分析与反馈闭环，实现误报过滤、重复归类与适配器规则生成。

**🔧 技术方法**

主要技术手段有：①统一测试框架的接口抽象，统一执行、归一化与计划记录；②基于执行计划哈希的同计划检测；③多代理LLM（Claude Code）三步分析：聚类拆分、深度调试与规则合成；④过程级数据存储与可追溯的报告结构。

**📊 数据集**

实验数据集主要来自三款成熟图数据库（Neo4j、Memgraph、FalkorDB）在12小时内随机生成的测试状态与查询；另外对尚未被测试过的NeuG 0.1.1 进行评估。无外部公开数据集，所有测试均在本地容器化环境下完成。

**📈 对比分析**

通过RQ4评估后处理流水线在工作量上的提升，报告压缩率从66.7%到84.6%不等，人工审核与LLM判定一致率达91%，真正缺陷召回率100%；RQ5显示在NeuG上八种方法共发现19个独立bug，单方法未覆盖全部，证明集成与后处理显著提升发现率。

**⚠️ 局限性**

局限性包括：①依赖人工评审与开发者确认，存在主观偏差；②仅在Cypher语义下实验，结果对其他图数据库或查询语言的可迁移性尚未验证；③同计划检测基于哈希，可能漏掉细微差异；④误报率虽大幅下降但仍存在，需进一步细化规则。

---

## 362. Optimal Learning in Games with Arbitrary Delays: Best-of-Both-Worlds for Self-Play and Adversarial Opponents

**arXiv ID:** 2611.07773 | [PDF](https://arxiv.org/pdf/2611.07773v1)

**作者:** Emmanuel Esposito `[一作]` (FairPlay Team), Mengxiao Zhang `[通讯]` (University of Iowa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了具有延迟反馈的零和博弈中的学习，提出了一种名为多锚乐观在线镜面下降（MAO-OMD）的算法，该算法在自我博弈中获得了最佳快速率，同时在面对任意对手时保持了极小极大最优保证。

**💡 创新点**

创新点在于MAO-OMD算法能够在未知的反馈延迟情况下，保证每个玩家在自我博弈中获得O(σ_max+1)的社会和个人遗憾，同时在面对任意对手时保证O(√(T+d_tot))的个人遗憾，这些最优速率无法通过标准的黑箱减少方法同时获得。

**🔧 技术方法**

使用了多锚乐观在线镜面下降（MAO-OMD）技术，该技术通过适应性地分配过去迭代的正则化权重来直接处理延迟反馈。

**📊 数据集**

论文中没有具体提到使用的数据集，但研究的背景是基于零和博弈的理论框架。

**📈 对比分析**

与现有方法相比，MAO-OMD在自我博弈中能够获得更快的收敛率和更低的遗憾，且在面对任意对手时的遗憾保证是最优的。与调度基础的方法相比，MAO-OMD避免了额外的对数因子。

**⚠️ 局限性**

限制在于该算法的性能依赖于对延迟的适应性，且在某些情况下，可能无法完全消除调度方法中存在的对数因子。

---

## 363. Neural Collapse on Imbalanced Datasets and Open-Set Scenario

**arXiv ID:** 2611.07753 | [PDF](https://arxiv.org/pdf/2611.07753v1)

**作者:** Qiufu Li `[一作]` (Shenzhen University), Yongdong Huang `[通讯]` (North Minzu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对使用交叉熵（CE）和二元交叉熵（BCE）损失的神经网络在类不平衡（long‑tail）和开放集（open‑set）情景下，推导并证明了在全局最小点时必然出现神经崩溃（Neural Collapse）现象，并给出了对应的特征向量与分类器向量的具体几何结构。

**💡 创新点**

①将神经崩溃理论从均衡数据扩展到极度不平衡和开放集；②在不平衡情境下给出全局最优点的显式解析（包括均衡点、等角性、等角等距性和少数类崩溃）；③提出了一种统一的分析框架，利用ETF（Equiangular Tight Frame）与 Jensen、AM‑GM 等不等式精确界定最小化条件。

**🔧 技术方法**

主要采用凸分析、矩阵正交与等角框架、Jensen 与 AM‑GM 不等式、以及对数/指数函数的凸性分析，构建了损失函数的下界并求解最优条件；在实验验证中使用了常见的卷积网络（如 ResNet18）与标准数据集。

**📊 数据集**

实验所用数据集包括：CIFAR‑100 的长尾版本（CIFAR‑100‑LT），以及对部分类别设为空样本的开放集变体（Open‑Set CIFAR‑100）。

**📈 对比分析**

与传统的 BCE/CE 训练结果对比，实验展示了在极度不平衡或开放集情形下网络特征矩阵与分类器矩阵快速收敛至 ETF 结构，验证了理论预测；在分类准确率上表现与标准 BCE/CE 训练相当或略优，说明理论分析并未损失性能。

**⚠️ 局限性**

局限性：①理论证明基于特征维数大于类别数且假设存在全局最小点，实际训练可能受优化路径和梯度消失影响；②只分析了均衡平均损失，而未涉及带权重或采样平衡的变体；③开放集分析假设未出现未知类别样本的特征分布，现实中可能更复杂。

---

## 364. Wedge-Jamming Mechanism for String Configuration

**arXiv ID:** 2611.07748 | [PDF](https://arxiv.org/pdf/2611.07748v1)

**作者:** Taro Tako `[一作]` (National Institute of Informatics), Taisuke Kobayashi `[通讯]` (National Institute of Informatics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种在线性连杆模块中加入楔形接触面以实现可调方向性阻尼的楔形阻塞机制。

**💡 创新点**

通过楔形角度调节实现了对垂直与扭转扰动的不同阻力，首次将楔形结构引入线性阻塞机制并实现了可调的各向异性抓握性能。

**🔧 技术方法**

使用3D打印模块、电磁力驱动的线性绳张力、力传感器和扭矩测量装置进行实验验证，并结合解析力学模型进行性能预测。

**📊 数据集**

没有公开数据集，实验使用自制的三种模块（圆珠、外楔、内楔）进行25次重复测试，测得张力-保持力曲线。

**📈 对比分析**

将三种模块在相同绳张力、不同扰动方向和不同支点位置下比较，结果显示外楔在垂直扰动下保持力最高，内楔在扭转扰动下保持力最强，且相较于传统圆珠模块保持力保持一致。

**⚠️ 局限性**

受3D打印表面粗糙度影响摩擦模型不精确；楔形角度对不同扰动方向的优化仍需更广泛的参数空间验证，且在复杂多关节机器人中实际集成仍面临尺寸与驱动控制挑战。

---

## 365. Comparing Classification and Feature Learning by Analyzing Their Loss Functions in Normalized Spaces with Different Scale Factors

**arXiv ID:** 2611.07741 | [PDF](https://arxiv.org/pdf/2611.07741v1)

**作者:** Qiufu Li `[一作]` (Shenzhen University), Yongdong Huang `[通讯]` (North Minzu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探讨并比较了在归一化特征空间中分类任务（使用交叉熵CE）与特征学习任务（使用二元交叉熵BCE）的本质差异，并理论分析了尺度因子γ对两者收敛行为和特征质量的影响；进一步在多种CNN与ViT模型上通过实验验证了理论结论，并评估了BCE在长尾与开放集识别中的优势。

**💡 创新点**

①首次从特征学习的第一性原理出发重构BCE，并证明其在归一化空间中可实现神经崩塌；②系统研究了尺度因子γ对CE和BCE收敛速率（线性与指数衰减）的影响；③发现BCE在较大γ空间下收敛更快、特征更优，并在长尾、开放集任务中优于CE。

**🔧 技术方法**

理论分析（神经崩塌、最小化分析、收敛率推导）；数值优化（SGD、AdamW）；实验评估（NC指标、分类准确率、特征紧凑度/可分性）。

**📊 数据集**

MNIST、CIFAR‑10、CIFAR‑100、Tiny‑ImageNet，且通过生成的长尾（IF=10/50/100）和开放集数据集（MNIST、SVHN、CIFAR‑10、CIFAR+50）进行验证。

**📈 对比分析**

对比方法：在相同网络、相同训练策略下对CE与BCE进行实验，评估收敛时间、NC指标、分类准确率、长尾/开放集指标；结果表明：BCE在γ∈[3,8]区间内收敛更快、NC更好，且在长尾、开放集任务上相对CE提升约1–4%。

**⚠️ 局限性**

限制：理论分析假设d≥K-1且数据平衡，无法直接推广到低维/不平衡情形；实验多聚焦于小数据集和单一网络结构，未探究更大规模或自监督场景；对温度系数τ与γ的对应关系仍缺乏深入实证。

---

## 366. Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction

**arXiv ID:** 2611.07735 | [PDF](https://arxiv.org/pdf/2611.07735v1)

**作者:** Nils Schwager `[一作]` (Trier University), Achim Rettinger `[通讯]` (Trier University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估大语言模型在预测卢森堡新闻平台用户对评论点赞/不点赞时的准确率与校准性能，并探究任务特定微调对这两项指标的影响

**💡 创新点**

首次将模型的token概率视为行为预测的置信度指标，系统性比较校准与准确率的关系，并证明微调可显著提升两者

**🔧 技术方法**

使用开源Ministral 3（3B、8B）模型，进行自监督微调（SFT），并采用Expected Calibration Error、Brier Score、ROC‑AUC等指标评估校准；对比统计基线和专有模型

**📊 数据集**

RTL.lu新闻平台的用户点赞/不点赞记录，包含约57k训练样例和13.8k测试样例，评论多语言（卢森堡语、英语、法语、德语）

**📈 对比分析**

对比未微调的开源模型、微调后模型、统计基线和专有模型；微调后8B模型准确率提升至0.758，ECE降至0.0056，Brier Score降至0.162；未微调模型准确率仅略高于随机，且过度自信

**⚠️ 局限性**

仅评估单词级二元决策，无法推广到开放文本生成；未考察多轮生成中的自我误差累积；实验仅针对Ministral 3，难以泛化到其他模型或行为数据

---

## 367. O$^2$PD: Overlap-Gated On-Policy Distillation on Reliable Prefix

**arXiv ID:** 2611.07731 | [PDF](https://arxiv.org/pdf/2611.07731v1)

**作者:** Shuaishuai Zu `[一作]` (Renmin University of China), Baochuan Yang `[通讯]` (Huawei Technologies Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在 on‑policy distillation（OPD）中哪些 token 需要对齐、如何对齐，提出了 O^2PD 方法：只对学生生成轨迹的前 20% prefix 进行对齐，并使用重叠门控 α‑散度在教师和学生之间动态调整覆盖与寻模的平衡。

**💡 创新点**

创新点包括：① 发现并量化跨前缀不对称性，证明前缀提供更可靠的教师监督；② 理论分析 top‑k OPD 的 Teacher‑support Uncovering 与 Dominant‑Mode Mismatch 两大问题，并设计 Reset Support 与 Support Convergence 两种支持策略；③ 引入重叠门控 α‑散度，使得每个 token 的覆盖/寻模策略由教师‑学生重叠率自适应切换；④ 将 prefix‑only 对齐与门控散度结合，显著提升 OPD 的效果。

**🔧 技术方法**

技术方法包括：on‑policy distillation、top‑k 逆 KL（reverse KL）对齐、α‑散度（α‑divergence）、重置原子（reset atom）机制、重叠门控机制、前缀限制策略、梯度覆盖与寻模理论分析。

**📊 数据集**

使用 DAPO‑MATH‑17K 作为训练集；在七个公认的数学/推理基准上评估：AIME24/25/26、HMMT25、Minerva、Olympiad、MATH500。

**📈 对比分析**

与 OPD、ESR、EOPD、TIP 等现有 OPD 方案在同一学生‑教师对比下进行对比；O^2PD 在三组学生‑教师组合中平均提升 +1.89、+4.81、+5.29 分点；在大多数单项指标上位列第一或仅次于最优方法。

**⚠️ 局限性**

局限性：1）只针对 top‑k OPD，可能不适用于其他对齐框架；2）前缀比例设为 20% 的经验值，可能在不同任务或更长文本中需调整；3）门控阈值选择仍需经验，可能影响性能；4）对低重叠率或高度多模分布的轨迹对齐仍面临挑战；5）未深入分析计算成本和训练时间的提升。

---

## 368. Reinforcing What the Student Has Learned: Teacher-Guided On-Policy Distillation

**arXiv ID:** 2611.07722 | [PDF](https://arxiv.org/pdf/2611.07722v1)

**作者:** Bozhen Fang `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了学生增量式在策略蒸馏（Student-Delta On-Policy Distillation, SD‑OPD），利用学生自身的后训练改动作为更新强度，并以教师反馈决定更新方向，从而提升小模型在数学推理任务上的表现。

**💡 创新点**

创新点在于：①将学生与其预训练基准模型的差异作为更新系数，避免因教师与学生差异导致的风格偏移；②通过教师同符号筛选仅保留学生已学偏好中教师认可的部分；③每轮刷新学生信号，使学习持续关注最新的学生改动。

**🔧 技术方法**

技术手段包括：在策略梯度框架下使用对数概率差分（log‑ratio）中心化；基于 top‑K(1024) 近似进行中心化计算；采用双重裁剪（dual‑clipped）损失优化；在每轮使用相同教师和基准模型进行评分。

**📊 数据集**

使用的主要数据集为 DAPO‑Math‑17K（数学推理对话）进行训练，并在六个公开数学评测集（AIME24/25/26, HMMT25, AMC23, MATH500）进行评估。

**📈 对比分析**

与传统 OPD 和 ExOPD 在相同采样与推理预算下比较，SD‑OPD 在所有教师‑学生配置中均提升了 3–8 个百分点的 pass@1，并在 8B 学生上实现超过教师的平均准确率。

**⚠️ 局限性**

局限性包括：仅在 Qwen3 与 MiMo 教师、小尺寸学生（1.7B–8B）且仅针对数学推理任务进行验证；缺乏对更大模型或非数学任务的通用性验证；方法依赖可用的预训练基准模型。

---

## 369. Evaluating Language Model Features for Zero-Shot Time-Series Anomaly Detection Under Isolation Assumption

**arXiv ID:** 2611.07720 | [PDF](https://arxiv.org/pdf/2611.07720v1)

**作者:** Romain Hermary `[一作]`, Djamila Aouada `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于注意力机制的双向 token 对齐框架，利用图形化的链接关系对两侧 token 进行互相匹配；

**💡 创新点**

创新点在于将注意力权重与图结构（link）相结合，形成可视化的中间注意层，提升了对齐的可解释性；

**🔧 技术方法**

核心技术包括自注意力（attention）、图连接（link）以及自定义的 TikZ 绘图实现；

**📊 数据集**

文中未给出具体公开数据集，可能以自制的 12×12 token 对齐样本为例；

**📈 对比分析**

未提供实验对比或性能指标，若有实验则需与传统对齐算法（如动态规划、基于概率的匹配）做速度和准确度比较；

**⚠️ 局限性**

局限性主要在于缺乏真实数据验证、可扩展性未评估以及对多语种/长序列的适应性未知。

---

## 370. Towards Generalist Graph-Level Anomaly Detection

**arXiv ID:** 2611.07706 | [PDF](https://arxiv.org/pdf/2611.07706v1)

**作者:** Junjun Pan `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于可辨子图提取的通用图级异常检测框架 GenGLAD，能够在多域数据上通过少量正常样本进行快速适应。

**💡 创新点**

通过跨域记忆引导的子图提取和多维差异度量实现可迁移异常知识学习，并用轻量级评分模块实现高效少样本适应。

**🔧 技术方法**

使用 GNN 子图提取、记忆池检索、交叉域 motif 注入、Gumbel-Softmax 采样、以及多维差异度量与线性评分模块等技术。

**📊 数据集**

在 10 个来自分子、化学、生物、社交网络等领域的公开 GLAD 数据集上进行训练和评估。

**📈 对比分析**

与多种无监督 GLAD 基线对比，GenGLAD 在 9/10 组数据集上均实现 AUROC 提升 5–15%，且适配速度更快。

**⚠️ 局限性**

仍需少量正常样本进行适配，无法实现完全零样本的通用异常检测。

---

## 371. Fast Unsupervised Anomaly Detection Methods on Software Logs - An Empirical Study

**arXiv ID:** 2611.07760 | [PDF](https://arxiv.org/pdf/2611.07760v1)

**作者:** Jesse Nyyssölä `[一作]` (University of Helsinki), Yuqing Wang `[通讯]` (University of Helsinki)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对七个软件日志数据集进行了 672 次实验，系统评估了四种无监督异常检测模型（KMeans、IsolationForest、RarityModel、Out‑of‑Vocabulary Detector），三种日志表示方式（单词、三元字符、解析事件），以及三种预处理策略（过滤异常、随机打乱、使用自定义载入器），并通过 AUC‑ROC 与 F1 评估模型性能，同时测量模型与预处理的执行时间。

**💡 创新点**

创新点在于：①提供了迄今为止最全面的无监督日志异常检测实验框架；②使用广义线性模型（GLM）深入分析各因素与数据集之间的交互影响；③揭示数据集特性是决定模型表现的主因，并证明随机打乱训练集对无监督模型影响微乎其微，挑战传统的“打乱会导致数据泄漏”观点；④系统量化了预处理（尤其是去除异常训练数据）对性能的提升。

**🔧 技术方法**

采用的技术包括：Python + sklearn（IsolationForest、KMeans），自定义 OOV Detector 与 Rarity Model；日志处理使用 Polars 进行低成本字符串规范化、词/三元字符切分、Drain 解析事件；向量化采用 TfidfVectorizer 或 CountVectorizer；模型评估使用 scikit‑metrics 计算 AUC‑ROC 与 F1；统计分析使用 Statsmodels 的 GLM 进行交互项回归；实验在 Intel i7‑13700K CPU 上运行，记录训练/测试时间与预处理耗时。

**📊 数据集**

使用的数据集有：BGL、HDFS、Hadoop、Spirit、Thunderbird、Liberty、LO2Token（新建微服务日志集），涵盖事件级和序列级两类日志，样本量从几百万行到数十万行不等。

**📈 对比分析**

比较方法：对每个模型-表示-预处理组合在每个数据集上跑 5 次，取 AUC‑ROC 作为主要指标；使用 GLM 评估主效应与交互效应；对同一组合在不同预处理下的 AUC‑ROC 进行配对比较；执行时间分为预处理时间、训练时间和推理时间。结果表明：①整体上 OOV Detector 在去除异常训练数据时表现最佳；②KMeans 与词表示在速度上最优；③数据集决定了模型表现，某些组合在某些数据集上能获得 AUC‑ROC >0.85，而在其他数据集仅略高于 0.6；④随机打乱对无监督模型几乎无效，只有监督决策树才显著提升。

**⚠️ 局限性**

局限性包括：①未对每个模型进行超参数调优，导致性能可能不达最优；②预处理仅考虑三种策略，未涵盖更细粒度的日志解析方法；③使用了带标签的基准数据来计算 F1，现实环境中往往缺乏标签；④实验仅在单台 CPU 环境下完成，未验证大规模分布式部署时的可扩展性；⑤结果高度依赖于数据集特性，无法直接推广到完全不同类型的日志。

---

## 372. Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment

**arXiv ID:** 2611.07780 | [PDF](https://arxiv.org/pdf/2611.07780v1)

**作者:** Ziang Cui `[一作]` `[通讯]` (Southeast University), Ziang Cui (Southeast University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过生成自然语言表示（NLR）并结合BERT对知识图谱三元组进行二分类，提出了GenVer框架；

**💡 创新点**

创新点在于：①使用Triple–NLR alignment挑选训练对；②利用seq2seq生成器产生NLR；③把生成文本特征与序列化三元组一起输入BERT并用结构化与非结构化特征进行融合；

**🔧 技术方法**

采用UniLM进行NLR生成，BERT进行双向编码，卷积特征提取，TransE式能量门控制结构化特征，最终用全连接+sigmoid做二分类；

**📊 数据集**

实验基于FactFB四个子集（由FB15k抽取并用扰动方式生成负例），分别对应2、4、8、16种关系；

**📈 对比分析**

与TransE、DistMult、ComplEx、KGTtm等传统图嵌入基线及SimKGC文本基线对比，GenVer在四个子集的准确率最高，AUROC也最高，提升约1–3个百分点；

**⚠️ 局限性**

局限性：生成的NLR并非独立证据，可能缺失实体、改变关系或增添不实信息；负例仅为随机扰动，未必代表真实错误；数据集仅覆盖FB15k子集，缺乏跨域泛化；生成与分类分离，难以直接评估生成质量。

---

## 373. The Intrusion Pathway: How Non-Evidential Context Degrades Language Model Performance

**arXiv ID:** 2611.07778 | [PDF](https://arxiv.org/pdf/2611.07778v1)

**作者:** Dikshant Kukreja `[一作]` (TurboML), Erik Cambria `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了大型语言模型在面对无证据的“无关上下文”时准确率下降的机制，并通过识别“入侵头”（Intrusion Heads）提出了一种稀疏的、训练无关的注意力掩码来降低干扰，恢复约57%的准确率。

**💡 创新点**

创新点在于：①揭示了“入侵路径”与“入侵头”对答案支持衰减的具体贡献；②证明这些入侵头与用于检索黄金上下文的头是分离的；③提出仅约占模型1%注意力头的掩码即可显著减轻无关上下文的负面影响；④通过与“oracle残差补丁”对比，证明大部分干扰可在局部计算状态恢复。

**🔧 技术方法**

技术主要包括：Transformer电路层级分析（MLP enrichment、parroting、relational reading）、attention头的入侵得分（intrusion score）计算、头级掩码干预、残差补丁（oracle patch）以及对多任务的跨任务干预效果评估。

**📊 数据集**

使用的数据集有：事实召回与绑定/检索任务（Factual Recall, IOI），以及四个下游基准——GSM‑DC、GSM‑IC、HotpotQA、PubMedQA，每个基准都在原始问题中注入无关或干扰上下文。

**📈 对比分析**

与基线（含无关上下文）对比，掩码干预在四个任务上平均恢复约57%的准确率提升；oracle补丁则恢复约97.6%，显示大部分损失可在局部状态恢复；实验验证了掩码在跨任务、跨模型上的可迁移性。

**⚠️ 局限性**

局限性包括：仅评估了开源decoder‑only模型（7B级别及GPT‑2 XL），未验证更大规模、多模态、Encoder‑Decoder或Mixture‑of‑Experts模型；仅覆盖英文基准，缺乏多语言、代码或对话等场景；掩码需要内部注意力访问，且恢复效果仍有限；未提供正式的上限或最优头数；

---

## 374. Adapters as Data: The Latent Space of LoRAs

**arXiv ID:** 2611.07775 | [PDF](https://arxiv.org/pdf/2611.07775v1)

**作者:** Moritz Grävinghoff `[一作]` (University of St.Gallen), Damian Borth `[通讯]` (University of St.Gallen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

训练 Transformer 自动编码器将 LoRA 适配器的权重序列化为潜在空间，随后在该空间上进行采样、属性编辑、扰动和多轮编码解码等操作，并验证其对比原始权重空间的效果。

**💡 创新点**

（1）把 LoRA 适配器视为数据模态，学习非线性潜在表示并保持采样/编辑能力；（2）通过稀疏 Tokenization 实现同一潜在空间同时容纳多种 Rank 的适配器；（3）使用无监督 AutoEncoder 解决多种下游操作，提升鲁棒性。

**🔧 技术方法**

Transformer 自动编码器（SANE）进行权重 Tokenization；PCA 与岭回归用于采样和属性方向；正交扰动实验与多轮编码解码；稀疏 Tokenization 支持多 Rank。

**📊 数据集**

64,974 个基于 Realistic Vision v5.1 的 Frozen Stable Diffusion 训练的 DreamBooth 风格 LoRA 身份适配器，配有 39 个可变 CelebA 属性；另外自行训练少量 rank‑2 适配器。

**📈 对比分析**

与 w2w 基线直接在权重空间采样/编辑比较，使用 R²（权重重建）和 LPIPS（图像感知距离）作为评估指标；实验表明潜在空间保留相同的采样/编辑质量，且在扰动下更鲁棒，十轮编码解码后仍保持可识别；混合 Rank 模型在重建精度略低（LPIPS 大约提升 32%）。

**⚠️ 局限性**

（1）训练、验证、测试拆分按适配器而非身份，导致近似重复，AUC 可能偏高；（2）重建误差形成“重建地板”，编辑操作需以 D(E(θ)) 为基准；（3）仅针对 rank‑1/2 的查询/值投影，无法直接推广到更大 Rank 或其他网络；（4）未验证跨域（如动物）或真实人脸的泛化；（5）属性方向受限于标签且不唯一。

---

## 375. Selective Event Evidence and Persistent Temporal Priors for Event-Based Point Tracking

**arXiv ID:** 2611.07713 | [PDF](https://arxiv.org/pdf/2611.07713v1)

**作者:** Xiaokai Song `[一作]` (National University of Defense Technology), Xichao Teng `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种事件相机下的长时序点跟踪方法，称为SEPT，结合事件权重调制和持久轨迹先验。

**💡 创新点**

创新点是Task-Adaptive Event Evidence Modulation (TEEM) 通过仅用跟踪监督学习可辨别事件重要性，以及无参数的Schur补全的Persistent Trajectory Prior (PTP) 在窗口切换时保留几何约束。

**🔧 技术方法**

采用事件堆叠表示、卷积编码、Transformer迭代细化、TEEM权重网络和PTP Schur补先验。

**📊 数据集**

使用合成EventKubric数据训练，并在EventKubric、EVIMO2、EC、EDS、M3ED等五个公开数据集评估。

**📈 对比分析**

与重新训练的ETAP基线及多种事件/帧混合方法比较，SEPT在TAP平均Jaccard、特征年龄和期望特征年龄上均显著提升，尤其在EC、EDS上超过多模态跟踪器。

**⚠️ 局限性**

局限包括对静态场景事件稀疏的敏感性、对TEEM训练数据量的依赖以及相对较高的推理时间与显存开销。

---

## 376. AffineGS: Unlocking Accurate Geometry in 3DGS via Spatially Varying Affine Rectification

**arXiv ID:** 2611.07709 | [PDF](https://arxiv.org/pdf/2611.07709v1)

**作者:** Tze Ho Elden Tse `[一作]` (dConstruct Robotics), Angela Yao `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过对单目深度先验中的区域性、分段线性失真进行空间可变仿射校正，并结合梯度域和法线约束，提升3D高斯分裂（3D Gaussian Splatting）在无传感器环境下的几何精度。

**💡 创新点**

①发现现有单目深度网络在不同区域表现出分段线性失真，导致传统全局线性对齐失效；②提出空间可变仿射校正模型与基于法线的梯度正则化，能自适应修正这些失真；③利用无传感器的深度重校准来对齐全局尺度，保持视觉质量。

**🔧 技术方法**

空间可变仿射校正、Guided Filter边缘保持平滑、梯度域法线正则化、最小二乘尺度对齐、3D Gaussian Splatting渲染与优化。

**📊 数据集**

NeRF‑LLFF、MuSHRoom、ScanNet++（含密集传感器深度标注用于评估）。

**📈 对比分析**

与传统显式模型（Splatfacto、2DGS、SuGaR）以及基于传感器深度的DN‑Splatter、NeRF家族的隐式模型进行比较；在无传感器条件下在PSNR/SSIM/LPIPS维持近乎最优视觉质量，并在网格F‑score、Chamfer距离、法线一致性上显著优于所有基线，甚至逼近使用传感器深度的上限。

**⚠️ 局限性**

仍需依赖较大的 Guided Filter 窗口调参；对极端视角稀疏或极端光照变化的鲁棒性有待进一步验证；与完整传感器深度相比，在某些场景中仍存在尺度漂移和细节欠缺。

---

## 377. Non-Parametric Mutual Information Estimation with Vine Copulas

**arXiv ID:** 2611.07693 | [PDF](https://arxiv.org/pdf/2611.07693v1)

**作者:** Sigurd Holmsen `[一作]` (University of Oslo), Ali Ramezani-Kebrya `[通讯]` (University of Oslo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于简化D-vine Copula的无参数互信息估计器VINE。

**💡 创新点**

创新点在于利用简化D-vine的结构将密度比简化为一组二元Copula乘积，得到了可解析、可并行的MI估计，并且完全无参数化，兼具高灵活性与计算效率。

**🔧 技术方法**

采用简化D-vine Copula模型、无参数二元Copula估计、Kendall τ阈值裁剪、PCA降维以及Monte Carlo求和等技术。

**📊 数据集**

在多种合成数据集（高维高斯、带空间相关、Mixture of Gaussians、非高斯变换、Spiral、Swiss Roll）以及含已知MI的图像数据（高斯平面与矩形）上进行实验。

**📈 对比分析**

与MINE、InfoNCE、SMILE、MIENF、VCE等前沿神经MI估计器比较，VINE在大多数情形下误差更小、计算更快，尤其在维度高至256时仍保持鲁棒。

**⚠️ 局限性**

缺点是需要估计O(d²)对二元Copula，导致二次复杂度；同时对简化假设的依赖若被严重破坏，性能可能下降。

---

## 378. Scaling Laws for SAE Training Data

**arXiv ID:** 2611.07756 | [PDF](https://arxiv.org/pdf/2611.07756v1)

**作者:** Nikita Koriagin `[一作]` (Yandex), Daniil Gavrilov `[通讯]` (T-Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究稀疏自编码器（SAE）训练中激活存储瓶颈，探索在固定训练预算下需要多少唯一激活来保持性能

**💡 创新点**

发现激活多样性对重建质量的提升呈先快后慢的饱和曲线，并提出利用缓冲区重放实现显著存储压缩（8–64×）而无质量损失的训练规则

**🔧 技术方法**

使用基于有效激活计数的缩放法则、批量TopK/标准TopK/JumpReLU稀疏门、解释方差（EV）、重建MSE、特征吸收、自动可解释性和稀疏探测等指标评估SAE

**📊 数据集**

FineWeb文本、数学文本（FineMath）和Python代码（Stack-Edu）等语料，Llama-3.1 8B、Qwen3系列模型不同层的残差流激活

**📈 对比分析**

通过对不同字典大小、预算、层和模型的网格搜索以及保留测试集评估，EV在到达饱和区间后几乎不随缓冲区增大而提升；重建MSE、AutoInterp等下游指标与EV同步；在推荐的最小缓冲区下可实现8–64×存储节省，且无显著性能下降

**⚠️ 局限性**

仅在固定稀疏度和单一任务下验证，扩展到其他门/宽度时需重新校准；对低训练预算和宽字典的欠训练情况拟合误差较大；未测算端到端计算时间和多模态/多任务适用性

---

## 379. Foresee to Trade: Forecast-Augmented Policy Optimization for Portfolio Management

**arXiv ID:** 2611.07750 | [PDF](https://arxiv.org/pdf/2611.07750v1)

**作者:** Zeyu Liu `[一作]`, Song Guo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 Forecast-Augmented Relative Policy Optimization (FRPO)，一种在在线投资服务中将资产机会与账户决策相结合的强化学习框架。

**💡 创新点**

创新点在于：①将资产机会与账户决策分别用未来相对收益目标和同一路径相对收益进行监督；②通过共享编码器实现两种监督的协同学习；③设计了 Adaptive Gate 动态权衡预测方差与梯度对齐，以自适应地控制辅助监督；④在无价值网络的情况下，使用同路径群组相对优化实现政策更新。

**🔧 技术方法**

技术方法包括：Transformer 编码器、连续多资产组合策略、Financial State Head (FSH) 用来预测未来相对收益及其方差、同路径相对奖励 (Same‑Path Group‑Relative Optimization)、梯度对齐与方差门控的 Adaptive Gate、基于高斯负对数似然的损失、以及 PPO‑style 采样与裁剪。

**📊 数据集**

使用的数据集为：DJIA30 与 TOPIX Core30 两个股票指数；以及三种自定义 6 资产投资组合（Mainstream、Defensive、High‑Volatility Growth），共计五个测试环境。

**📈 对比分析**

在十个随机种子下的平均表现中，FRPO 在所有五个数据集上均取得最高的累计收益率和年化收益率；在风险调整后指标（Sharpe、Sortino、Calmar）上也保持竞争力。与传统技术指标、经典 RL 算法（PPO、A2C、TD3、SAC）以及其他金融 RL 基线（DeepAries、EarnMore、DeepTrader、MASA）相比，FRPO 在大多数场景下取得显著优势。

**⚠️ 局限性**

局限性包括：①理论分析仅针对非预知交易策略下的方差；②实验仅基于历史 OHLCV 数据与技术指标，未考虑多模态 Web 信息或自然语言输入；③当前模型不支持短期预测或用户偏好建模，未来工作可扩展为端到端语言代理或加入用户个性化需求。

---

## 380. Over-the-Scene Computation: Illumination-Programmed Functional Sensing of Unresolvable Populations

**arXiv ID:** 2611.07732 | [PDF](https://arxiv.org/pdf/2611.07732v1)

**作者:** Seyed Mohammad Azimi-Abarghouyi `[一作]` (Aalborg University), Christopher G. Brinton `[通讯]` (Purdue University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种通过光照即为程序、场景即为计算机的“场景级计算”方法，能够在不分辨目标个体的情况下直接计算密集目标群体的功能量（如预计会穿过某一边界的目标总质量、流量或平均速度）

**💡 创新点**

创新点在于将目标回波的散射函数与发射波形的相干函数相结合，证明了发射波形的模糊函数是场景计算的传递函数；进一步给出可计算的核函数、噪声成本和Cramér–Rao下界；提出使用线性chirp实现的“预估范围剖面”，在低SNR下通过聚合实现低噪声、低采样率的功能量估计；并证明对极端功能量（首次到达时间）只能通过对数级的二分搜索实现。

**🔧 技术方法**

使用了信号的相干/交叉相干函数、时间频率分布理论、Cohen类分布、相位空间傅里叶变换、散射函数理论、Cramér–Rao极限、Lieb定理、Brunn–Minkowski不等式、Speckle统计、MTI模板、以及模拟仿真中的FMCW波形与Doppler解耦。

**📊 数据集**

主要使用模拟数据：在77 GHz雷达条件下，模拟300个反射率相同、距离分布在[60,150] m、闭合速度在[5,25] m/s的目标群体，SNR在-25 dB至0 dB范围内。

**📈 对比分析**

与传统的“感知‑后‑计算”（先形成分辨率较高的距离-多普勒图再进行积分或检测后计数）进行比较。实验表明，chirp式场景级计算在相同能量、时域驻留和频带宽度下，单次扫描所需样本数少77倍，且在-10 dB至6 dB的SNR区间内，均方根误差比传统CPI低约16%（甚至低至2.9%），对噪声基线误差的鲁棒性也显著优于图像积分方法。

**⚠️ 局限性**

局限性包括：无法在单个探测波形下确定未受约束的Rayleigh场景的首次到达时间（需要二分搜索）；对加速目标的假设仅适用于小速度变化；对宽带信号、非窄带模型的推广尚未给出；需要在场景满足无相关返回、随机相位等假设下才能实现理论解析；实际系统需考虑硬件对相位与幅度的失真与时钟同步误差。

---

## 381. Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction

**arXiv ID:** 2611.07695 | [PDF](https://arxiv.org/pdf/2611.07695v1)

**作者:** Shi Yin `[一作]` (Hefei Comprehensive National Science Center), Lixin He `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DP-Learner框架，直接在直积空间对Hamiltonian块进行等变式信息传播，以预测自洽电子结构Hamiltonian。

**💡 创新点**

解决传统等变网络在直和空间的表示不匹配问题，使用Lie生成器方向嵌入和不变加权左右乘/收缩，保持SO(3)、SU(2)与空间反演等变性，并实现Transformer式直积空间信息传递。

**🔧 技术方法**

使用Lie-生成器方向嵌入、不可变权重的左右乘收缩、Transformer架构、Frobenius总结的可变标量路由、层归一化与LayerScale、以及H^(0)残差学习等技术。

**📊 数据集**

构建并使用MP‑HAM‑110K（110k结构，68元素）以及现有Materials‑HAM‑SOC数据集进行评估。

**📈 对比分析**

与NextHAM在两大基准上对比，DP‑Learner的Gauge MAE下降约34‑40%，推理时间缩短10‑11倍，GPU内存消耗减少约95%，参数量约减50%。

**⚠️ 局限性**

仍缺乏对学习到的标量系数、注意力权重等物理解释；仅在非SOC/ SOC 设定下验证，未深入探究在更复杂多体相互作用或极限条件下的泛化能力。

---

## 382. Continued Training of Continuous Language Flows with Token Identity Squared Supervision

**arXiv ID:** 2611.07721 | [PDF](https://arxiv.org/pdf/2611.07721v1)

**作者:** Jiawei Jiang `[一作]` (Kunming University of Science and Technology), Zhengtao Yu `[通讯]` (Kunming University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Token Identity Brier Flow (TIBF)，通过对token概率分布使用Brier平方损失直接监督身份，同时用概率加权的token嵌入生成连续流。

**💡 创新点**

创新点在于将Brier损失引入连续语言流训练，直接监督完整token分布，解决连续MSE对概率分布不可区分的问题，并通过冻结预训练嵌入实现更稳定的后续训练。

**🔧 技术方法**

采用高斯观测模型、流匹配（Flow Matching）、概率加权嵌入、信息均匀噪声调度、Self‑conditioning、EMA以及Brier损失与风险分解等技术。

**📊 数据集**

在OpenWebText (OWT) 与 Language Model Benchmark 1B (LM1B) 上进行持续训练和评估，并在多种文本语料上进行零样本跨域转移测试。

**📈 对比分析**

与ConvergeFlow（连续MSE）和LangFlow（CE）在相同预训练初始化和额外token预算下对比，使用Gen.PPL、Dataset PPL、token entropy、4-gram重复率等指标；冻结嵌入TIBF在所有采样预算下均取得最低Gen.PPL，并在LM1B上获得最低重复率，在跨语料转移上亦优于其他方法。

**⚠️ 局限性**

局限性包括在OWT上生成文本的多样性略低、重复率偏高；Brier平方损失对概率分布的细粒度信息捕捉有限；以及冻结嵌入在极端噪声条件下的收敛性和鲁棒性仍需进一步验证。

---

## 383. Adapt the Object to the Subject: Representative Study Material for Empirical Software Engineering

**arXiv ID:** 2611.07772 | [PDF](https://arxiv.org/pdf/2611.07772v1)

**作者:** Max Unterbusch `[一作]` (University of Duisburg Essen), Andreas Vogelsang `[通讯]` (University of Duisburg Essen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过 31 名行业从业者的 2×2 内部对照实验，评估需求条款的代表性对其可评估性的影响，并提出一种基于 LLM 的即时生成与可选的缺陷注入方法，生成与参与者工作环境高度匹配的合成需求。

**💡 创新点**

①首次提供代表性对可评估性的因果证据；②提出上下文自适应的 LLM 需求生成与缺陷注入流程，突破传统公共数据集或人工构造研究对象的局限；③展示缺陷注入不影响代表性与可评估性，为实验对象控制提供新途径。

**🔧 技术方法**

利用 OpenAI GPT‑5.5 进行需求生成与缺陷注入，结合 2×2 内部对照实验设计；数据通过 Bayesian 数据分析与统计因果推断框架建模，使用累积回归模型估计代表性、生成方式、缺陷等变量对可评估性的效应。

**📊 数据集**

静态需求来自公共数据集 PROMISE+（规范式需求）和 Dalpiaz（用户故事）；合成需求则完全由 LLM 根据参与者提供的简短工作情境即时生成。

**📈 对比分析**

通过在实验中让受试者对 20 条需求分别评估代表性与可评估性（5 级 Likert），比较静态与合成两种条件。结果表明：合成需求在代表性评分上显著高于静态，且由于代表性提升，可评估性也随之提升；缺陷注入对两项评估均无显著影响。复杂度分析显示合成需求在句法树深度与句子长度上与静态需求相当或更高。

**⚠️ 局限性**

样本量仅 31，且为非随机邀请的行业从业者，外部有效性受限；仅使用单一 LLM，缺陷注入多样性有限；构念测量采用单条目自评，可能不足以捕捉多维特性；实验中存在潜在的识别偏差与 LLM 非确定性；未对实际行为进行测量，结果仅基于自我报告。

---

## 384. When Bigger Becomes Brittle: Why Compression Hurts More as Language Models Scale

**arXiv ID:** 2611.07891 | [PDF](https://arxiv.org/pdf/2611.07891v1)

**作者:** Gautam Gupta `[一作]` (Indraprastha Institute of Information Technology), Avinash Anand `[通讯]` (Singapore Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在量化与剪枝下语言模型的压缩如何影响检索能力，并通过将损失分解为直接缺口与补偿成分来量化模型在不同规模下的自我修复能力。

**💡 创新点**

创新点在于提出一种完整的损失拆解框架，区分检索头直接贡献、抑制释放、备份响应和最终归一化四个渠道，并对不同模型规模的补偿效应进行系统对比。

**🔧 技术方法**

采用注意力头剔除、4‑bit 权重量化、结构化剪枝以及 TransformerLens 进行内部激活归因，结合对齐度量和 logit 差分进行定量分析。

**📊 数据集**

使用 1,000 条 IOI（间接对象识别）提示进行实验，另外 2,000 条 held‑out 提示进行验证，并在 Pythia 410M–12B、GPT‑2 XL 与 Llama‑3.1‑8B 等多模型族上进行规模实验。

**📈 对比分析**

通过对比不同规模模型在相同缺口（D=0.5 逻辑）下的补偿比例，结果显示小模型能通过抑制释放和备份响应大幅恢复输出，而大模型的补偿效果明显下降；量化时缺口比值从 0.55 提升到 0.91，剪枝在不同掩码下呈现单调或非单调趋势。

**⚠️ 局限性**

限制在于仅关注 IOI 模板检索，未涵盖其他能力；实验仅在浮点推理中评估压缩，未涉及真实部署核；以及补偿机制的因果关系和多头对抗性贡献尚未深入探讨。

---

## 385. CORE: Coarse-to-Fine Search with Context Management for Deep Research

**arXiv ID:** 2611.07859 | [PDF](https://arxiv.org/pdf/2611.07859v1)

**作者:** Hoang Tran Vuong `[一作]` (Qualcomm Vietnam Company Limited), Dat Quoc Nguyen `[通讯]` (Qualcomm Vietnam Company Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 CORE，一种训练无关的深度研究助手，利用粗到细检索与上下文管理技术在多步信息检索问答中提升答案质量。

**💡 创新点**

核心创新在于：①把检索细化到片段级别，先用稠密检索选取 passage，再利用 late‑interaction reranker 的 token 级相似度直接定位最相关的 snippet，避免额外编码；②引入上下文扩展和跨检索去重，让代理在已有证据基础上扩展相关上下文并减少重复文本；③完全无需额外训练，可直接与任意代理模型结合。

**🔧 技术方法**

技术细节包括：稠密检索（AgentIR‑4B）做粗检索；ColBERT‑style late‑interaction reranker（Reason‑ModernColBERT）做细检索并输出 token 相似矩阵；CORE harness 管理工具调用与上下文；上下文扩展、去重逻辑；所有组件均为无监督/训练‑free。

**📊 数据集**

实验使用三大基准数据集：BrowseComp‑Plus（100K网页）、BrowseComp‑Wikipedia（6M Wikipedia 文章）和 BrowseComp‑Plus_CM（553M ClimbMix 文档），并在固定语料下进行封闭检索评测。

**📈 对比分析**

在与 Tongyi DeepResearch 和 OpenResearcher 的对比实验中，CORE 在三种代理模型（GPT‑5.4、Qwen3.8‑27B、Tongyi‑DeepResearch‑30B‑A3B）和三项基准上均取得最高答案准确率和证据召回率，同时每次检索返回的 token 数量最少，搜索调用次数和峰值上下文也更低或相当。

**⚠️ 局限性**

主要限制：仅在固定语料的封闭检索环境下评估，未覆盖实时网页搜索或开放式研究任务；参考答案评估可能不足以体现长篇综合写作能力；未验证在动态多源环境中的鲁棒性。

---

## 386. CANMentor: A Large-Small Model Collaboration Framework for CAN Bus Intrusion Detection

**arXiv ID:** 2611.07853 | [PDF](https://arxiv.org/pdf/2611.07853v1)

**作者:** Zhiquan Liu `[一作]` (Jinan University), Jianfeng Ma `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CANMentor框架，在车辆侧部署轻量级IDS模型，云端使用大型语言模型（LLM）作为教师。车辆先做本地推断，若置信度低则生成隐私保护的代理样本上传给云端，云端返回教师预测；若响应时间在规定阈值内使用教师结果，否则保留本地结果；所有教师logits用于周期性知识蒸馏，持续提升车辆IDS性能。

**💡 创新点**

创新点：
1) 通过固定距离（d）代理样本实现隐私保护的教师-学生协作；
2) 在非IID数据分布下，利用LLM教师的强泛化能力弥补本地模型盲点；
3) 结合延迟控制与熵加权的定期知识蒸馏，兼顾实时性与模型升级；
4) 引入数值CAN帧编码与LoRA微调，使LLM直接处理CAN信号。

**🔧 技术方法**

核心技术：
- WGAN‑GP生成器+语义/特征一致性约束实现固定距离代理样本；
- 数值CAN帧编码将标识符、载荷、时间差映射到LLM输入；
- LLM（Qwen2.5、OPT）+LoRA微调及分类头；
- 温度缩放softmax + KL蒸馏，结合熵加权与anchor loss；
- 延迟阈值 T_max 控制在线教师使用。

**📊 数据集**

数据集：ROAD（汽车CAN数据，含fuzzing与masquerade攻击）和 HCRL Car‑Hacking（含DoS、fuzzing、spoofing攻击）。

**📈 对比分析**

与传统方法比较：
- 与仅使用本地IDS模型对比，宏F1提升最大 25.01%（ROAD）和 20.79%（HCRL）；
- 与传统BiLSTM教师对比，LLM教师在非IID场景下表现更好，跨数据集鲁棒性更强；
- 对低置信度样本，教师准确率 95.7%–96.9%，提升 38–42 pp；
- 通过固定距离 d=0.40/0.35，重建优势低于 5%，满足隐私需求。

**⚠️ 局限性**

局限性：
- 需要持续的云端连接和网络延迟满足阈值；
- 代理样本生成与教师适配对特定车辆、传感器有依赖，迁移到其他平台需重新训练；
- 目前仅在仿真/离线环境验证，真实车联网环境的通信时延与断线情况未充分评估；
- 对极少量或新型攻击类别的检测仍受样本稀缺影响；
- 仍需探索更高效的代理生成与教师压缩方案以降低算力和能耗。

---

## 387. Self-Supervised Diffusion Models for Handwritten Signature Verification via Ruin Reconstruction

**arXiv ID:** 2611.07810 | [PDF](https://arxiv.org/pdf/2611.07810v1)

**作者:** Peirong Zhang `[一作]` (South China University of Technology), Lianwen Jin `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于自监督扩散模型的离线手写签名验证框架 SigDiffuser，用于学习高质量的签名表示。

**💡 创新点**

创新点包括：① Reconstruction from Ruins (RFR) 预训练任务，将签名极度破坏后再重建，迫使模型捕获写作风格；② Pyramidal Feature Aggregation Module (PFAM) 在编码器与扩散去噪器之间进行多尺度特征聚合，提升表示能力。

**🔧 技术方法**

采用 DDIM 扩散去噪器、ResNet‑18 视觉编码器、PFAM、Siamese 网络和 NSPP+PDSN 进行下游验证。

**📊 数据集**

使用四个多语言签名数据集进行评估：BHSig260‑B、BHSig260‑H、HanSig、UTSig、CEDAR。

**📈 对比分析**

通过与现有监督和自监督方法在开放式验证（单参考、全局阈值）下对比，SigDiffuser 在所有数据集上均取得领先表现，EER 仅为 1.85%（BHSig260‑B）、3.13%（BHSig260‑H）、7.39%（HanSig）、23.48%（UTSig）和 6.92%（CEDAR），显著优于对比模型。

**⚠️ 局限性**

局限性包括：① 在跨数据集迁移时性能下降；② 依赖人工生成的极端破坏方式，缺乏对自然破坏的适应；③ 主要聚焦表示学习，未针对图像生成和可控编辑；④ 在验证集划分导致训练样本减少时性能会有所下降。

---

## 388. A Sequential Framework for Urban Drone Delivery: Network Design, Vehicle Sizing, and Energy and Cost Assessment

**arXiv ID:** 2611.07808 | [PDF](https://arxiv.org/pdf/2611.07808v1)

**作者:** Hong-Su Nam `[一作]` (Inha University), Hak-Tae Lee `[通讯]` (Inha University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个顺序框架，将机型尺寸、航线规划、枢纽布局与能耗、运营成本评估耦合起来，用来评估城市无人机快递系统的设计与运营；

**💡 创新点**

框架首次将枢纽距离约束直接转换为机型续航需求，并通过离散-连续优化实现最小起飞质量的机型设计，随后用该机型参数评估整个系统的能耗和成本，揭示了机型轻量化并不一定能降低运营成本；

**🔧 技术方法**

使用了p‑中心与k‑means混合的枢纽选址模型、深度优先搜索生成航线、混合整数线性规划优化航班频率、以及基于电动四旋翼动力学与结构模型的机型尺寸与航程优化；

**📊 数据集**

采用首尔（韩国）行政区级地面配送数据（CJ物流6月22日的4,798,710件配送记录），按比例缩放得到无人机配送需求；

**📈 对比分析**

对比了1–4个枢纽配置下的基础机型设计、不同枢纽下使用其他配置机型（s→H）、以及不同负载-续航范围下的推进器选择；指标包括运营成本/投递、能耗/公里、机队规模、航班数等，结果显示增加枢纽能降低机型重量与能耗，但由于航班数、租金与人力成本上升，运营成本/投递往往上升；

**⚠️ 局限性**

假设配送需求已知、无人机按直线飞行、忽略空域限制、障碍物与高度限制，使用平均飞行率估算人力，未考虑车辆折旧、保险、交通管理费用等，因而更适合作为比较规划工具而非精确成本预测。

---

## 389. TactileLab: Sim-to-Real Dexterous Robot Manipulation with Efficient Shear-Sensitive Tactile Simulation

**arXiv ID:** 2611.07881 | [PDF](https://arxiv.org/pdf/2611.07881v1)

**作者:** Yijiong Lin `[一作]` (Beijing Institute of Technology), Nathan F. Lepora `[通讯]` (University of Bristol)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个可扩展的多模态触觉仿真与控制框架TactileLab，利用刚体仿真计算出光流式触觉流（simulated tactile flow）以及触觉深度图，实现对粘性接触的高效模拟与机器人控制。

**💡 创新点**

创新点包括：① 通过刚体运动直接推算稠密触觉流，避免了高成本软体仿真；② 设计了PETS-Net时空编码器，联合姿态、动作历史与高维触觉图像，实现更高效的隐层适配；③ 研发了TacFlowNet实现真实触觉图像对触觉流的像素级翻译，完成仿真-真实的跨模态迁移。

**🔧 技术方法**

技术手段包括：基于IsaacLab的GPU并行刚体仿真；光流式触觉流计算与触觉深度图生成；双阶段教师-学生强化学习框架；位置编码+TCN时空网络（PETS-Net）；pix2pix式生成器+判别器的TacFlowNet；多任务评估指标与对照实验。

**📊 数据集**

数据集：1）自制的真实-仿真配对触觉图像/深度/流数据，用于训练TacFlowNet；2）模拟任务中产生的大规模触觉深度与流数据；3）在真实机器人上收集的多对象、不同运动轨迹的跟随与掌控实验数据。

**📈 对比分析**

与现有方法（HORA、AnyRotate、传统PID等）对比，TactileLab在双臂跟随任务中的位置/姿态误差均低于6.5 mm、0.06 rad；在掌控任务中旋转计数最高，时间完全保持30 s；在仿真中时间‑失败率与跌落率显著降低，实验结果表明高维触觉流+深度图能显著加速学习并提升最终性能。

**⚠️ 局限性**

局限性：假设触觉表面始终处于无滑移状态，突发滑移时翻译质量下降；仅基于刚体仿真，无法捕捉软体变形的完整动力学；对大范围几何与表面特征的泛化需更大多模态数据；实现时仍需针对新传感器训练专门的真实‑仿真翻译器。

---

## 390. Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency

**arXiv ID:** 2611.07899 | [PDF](https://arxiv.org/pdf/2611.07899v1)

**作者:** Karthik Ravula `[一作]` (ServiceNow), Gunjit Mittal `[通讯]` (ServiceNow)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种持久化检索子代理，直接将检索证据返回给主代理以生成答案，并在企业查询场景中进行评估。

**💡 创新点**

创新点在于将检索状态持久化并直接返回检索证据，消除了双重生成步骤，同时支持对检索结果的纠错与上下文共享。

**🔧 技术方法**

使用 GPT‑5.4 LLM、Schema 与 Cypher 结构化检索工具、子代理/主代理协同架构以及缓存/压缩引擎。

**📊 数据集**

使用 ITSM、HRSD 单轮与多轮问答集，以及包含多轮依赖的 ITSM Complex 数据集。

**📈 对比分析**

通过与技能检索（Skill）和无持久化检索（Non‑persistent）两种基线对比，测量准确率、响应时间、Token 量与估算 API 成本；在多轮情境下，持久检索平均降低约45%成本、提升约9%响应速度。

**⚠️ 局限性**

局限性包括检索结果不完整时可能误导主代理、纠错机制依赖主代理识别缺口、评估仅覆盖结构化企业记录，未验证对更广泛任务的适用性，且不同硬件/配置可能影响性能。

---

## 391. DISCERN: Distributional Synthetic Replay for Retain-Free Incremental Machine Unlearning

**arXiv ID:** 2611.07892 | [PDF](https://arxiv.org/pdf/2611.07892v1)

**作者:** Nicolás Segura-Kunsagi `[一作]` (University of Granada), Francisco Herrera `[通讯]` (University of Granada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种保留无记忆的增量类消除方法 DISCERN，利用每类嵌入的分布样本重放来实现机器无学习。

**💡 创新点**

创新点在于针对不同网络的潜在空间几何，动态选择欧氏或方向性分布（高斯、拉普拉斯、t、von Mises–Fisher、Fisher–Bingham），并通过统计检验挑选最合适的分布，从而在增量消除（ICU）中显著降低已遗忘知识的再出现。

**🔧 技术方法**

主要技术包括分布拟合与选择（Henze–Zirkler 正态检验、Rayleigh、方向性检验、交叉验证对数似然）、合成样本重放、两种消除损失（notop、confidence）以及对变压器与卷积网络嵌入空间的建模。

**📊 数据集**

在 MNIST、CIFAR‑10 与 CIFAR‑100 上使用 EfficientNetV2‑S、ResNet18、ViT‑B/16 与 Swin‑T 四种 backbone 进行评测。

**📈 对比分析**

与七种保留无记忆基线（Boundary Shrink/Expand、Bad Teacher、DELETE、EWC、LF‑SSD、Random Labeling）以及 retrain 进行对比；在单类/多类消除场景下性能相当或优于多数基线；在 ICU 场景中对变压器模型表现尤为突出，H 值保持约 0.8，接近 retrain 的 0.93。

**⚠️ 局限性**

限制包括：对卷积网络的 ICU 仍显著退化；分布选择对低样本或高维情况下的鲁棒性不足；需要一次性预先拟合分布，成本可观；对真实生产数据的迁移性和可扩展性仍需进一步验证。

---

## 392. A Low Similarity Score Does Not Show Independence after Source-Conditioned Repository Rewriting

**arXiv ID:** 2611.07887 | [PDF](https://arxiv.org/pdf/2611.07887v1)

**作者:** Micheline Bénédicte Moumoula `[一作]` (University of Luxembourg), Tegawendé F. Bissyande `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在保持原始仓库可见的条件下，让多种大型语言模型驱动的编码代理在 Python 项目上完成全新重写，并评估其成本、相似度与可重现性。

**💡 创新点**

创新点在于提出并实现了“源条件重写”协议，公开了 69 次重写实验、相关基准及五种不同模型的重复性分析，首次系统性检验了低相似度是否能证明独立重写。

**🔧 技术方法**

主要技术包括：使用 GitHub Copilot Agent 模型（Claude、Gemini、Qwen 等）、JPlag 与其他 token‑based 检测器评估相似度、构建行为 harness 进行功能验证、以及提取外部 import 与指纹的保留率做 provenance 信号分析。

**📊 数据集**

数据集包括六个 AGPL/GPL 许可的 Python 项目（总计约 50k 代码行）以及对应的独立实现（同任务）和五个跨领域的无关项目，用于计算相似度边界和保留率差异。

**📈 对比分析**

对比方法是将重写、独立实现与无关项目在 JPlag、保留率（imports、fingerprint）上的相似度进行统计；结果显示大多数重写低于 25% 相似度但高度可变，保留率未能区分重写与独立实现（AUC ≈ 0.4）。

**⚠️ 局限性**

局限性包括：仅对六个小型 Python 项目评估，单一控制实现、模型训练与公开项目重叠的可能影响、仅使用 token‑based 检测器、操作员单一且未完全消除人为偏差，以及相似度分布高度波动导致阈值意义模糊。

---

## 393. When, Where, How: Active In-Context Learning for Tabular Foundation Models

**arXiv ID:** 2611.07886 | [PDF](https://arxiv.org/pdf/2611.07886v1)

**作者:** Christian M. M. Frey `[一作]` `[通讯]` (University of Technology Nuremberg), Christian M. M. Frey (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估在冻结的表格基础模型（TFM）上进行主动上下文学习（Active In-Context Learning）的两种新型采样策略：基于目标最近邻的TANN和基于期望预测信息增益的EPIC。

**💡 创新点**

创新点包括：1）首次将期望预测信息增益（EPIG）迁移到冻结的TFM，利用模型的上下文条件实现无需后验或重训练即可评估信息增益；2）引入目标感知最近邻采样TANN，证明在早期采样时仅需覆盖输入空间，而后期则应聚焦于模型不确定的目标；3）系统性研究三类采样准则（多样性、不确定性、目标感知）随上下文增长的价值变化。

**🔧 技术方法**

使用的技术包括：冻结的TabICL（v2）和TabPFN（v3）表格基础模型；多样性规则（核心集、随机、混合）；不确定性规则（边距）；目标感知规则（TANN、TANN-feat、Target-Random、Target-Coverage）；期望预测信息增益规则（EPIC）；对比实验中还使用了XGBoost、CatBoost、LabelSpreading等传统模型；评估指标为Cohen's κ以及AULC。

**📊 数据集**

实验使用了20个OpenML分类基准，包含从145到581,012行、4到94特征、2到7类别不平衡的数据集；其中8个大型数据集（1,358-7,000行）用于细粒度预算评估。

**📈 对比分析**

与十个基线采样规则（随机、边距、核心集、混合、代理混合、TANN、TANN-feat、Target-Random、Target-Coverage、EPIC）以及传统模型的自我采样规则进行对比。实验结果表明：①在少量标签时，多样性规则（如核心集）最佳；②随着上下文增长，预测导向规则（EPIC、TANN）在最终预算下取得最高Cohen's κ；③EPIC在5-10%标签预算下平均保留80–87%的全数据性能；④TANN以约八倍低的计算成本实现与EPIC相近的效果；⑤EPIC所选标签同样能显著提升XGBoost和CatBoost的性能。

**⚠️ 局限性**

限制包括：①尚未实现自适应规则切换；②需要提前获得目标输入（转导设置）；③EPIC目前仅支持分类任务；④批量选择采用贪心近似，可能导致信息冗余；⑤对候选短名单和目标样本的近似假设未在更大规模或不同模型上广泛验证。

---

## 394. Responsible GenAI Use During Active Armed Conflict: Evidence from Sudan's Emergency Response Rooms

**arXiv ID:** 2611.07874 | [PDF](https://arxiv.org/pdf/2611.07874v1)

**作者:** Dalia Ali `[一作]` (Technical University of Munich), Orestis Papakyriakopoulos `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了苏丹战区社区志愿者在紧急响应室（ERR）中使用或不使用生成式人工智能（GenAI）的实践，并分析了冲突环境下的安全、信息保护与决策可靠性等关键因素。

**💡 创新点**

首次系统记录并分析了冲突条件下社区基础人道主义响应者对GenAI的实际使用、限制与避免行为，为冲突敏感情境下的责任AI提供了经验性见解。

**🔧 技术方法**

采用定性研究方法：问卷调查、15个半结构化访谈，使用人工与GPT-6辅助的文本转录、翻译、主题编码与模式生成，结合人工检查确保译文与编码质量。

**📊 数据集**

数据集包括89份匿名问卷结果与15份访谈记录（包含中文原文与英文翻译），覆盖不同地理区域与职责角色的ERR志愿者。

**📈 对比分析**

通过对比使用与不使用GenAI的志愿者在安全风险、信息泄露、可信度与实用性方面的经验描述，采用主题分析与案例对照进行比较；由于缺乏可量化的性能指标，未给出具体数值性能评估。

**⚠️ 局限性**

样本非代表性（多集中于较安全、连接良好的地区），数据主要来自受访者自述，缺乏直接观察或客观验证，且受冲突环境限制，可能低估最受限制志愿者的真实情况。

---

## 395. Quasi-polynomial-sized vertex multicut-mimicking networks and kernelization for Vertex Multiway Cut

**arXiv ID:** 2611.07880 | [PDF](https://arxiv.org/pdf/2611.07880v1)

**作者:** Eduard Eiben `[一作]` (Royal Holloway), Magnus Wahlström `[通讯]` (Royal Holloway)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种确定性多项式时间算法，构造了具有准多项式大小的顶点多切割模仿网络。该算法能够为给定图G、终端集T和正整数c构造一个包含|T|^7 c^O(log c)个顶点的图，精确保留每个最小顶点多路切割值。

**💡 创新点**

创新点在于首次提供了一个具有|T|^O(log |T|)个顶点的精确顶点多切割模仿网络，解决了Wahlström提出的一个开放问题，并且该算法是确定性的，而之前的结果依赖于随机算法。

**🔧 技术方法**

使用了确定性算法和图论中的切割覆盖引理，结合了标记方案和递归替换策略。

**📊 数据集**

使用了图G和终端集T的组合，具体数据集未明确提及，但涉及到的图和终端集是算法的输入。

**📈 对比分析**

与之前的方法相比，本文的方法在顶点多切割模仿网络的大小上有显著改进，提供了k^O(log k)的大小界限，优于之前的k^O(log^2 k)的界限。性能上，算法在多项式时间内运行，并且能够处理更复杂的终端配置。

**⚠️ 局限性**

限制在于该算法的性能依赖于终端的可删除性，且在处理超图时，算法的复杂性可能会增加。

---

## 396. FARR: Forecast-Aware Residual Refinement with Matched Endpoint Evidence for Long-Horizon Time-Series Forecasting

**arXiv ID:** 2611.07870 | [PDF](https://arxiv.org/pdf/2611.07870v1)

**作者:** Yiyang Zhang `[一作]` (Nantong University), Qiang Sun `[通讯]` (Nantong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于端点证据的残差细化方法FARR，用于提升长周期时间序列预测的准确性

**💡 创新点**

创新点在于利用同一历史数据下的有效与无效条件预测差异作为端点证据，结合历史特征学习残差并通过水平门控动态调节残差贡献

**🔧 技术方法**

采用预训练大语言模型作为基础预测器，构建端点提议模块、共享残差细化网络和水平门控；使用MLP、tanh、SmoothL1等技术实现细化与正则化

**📊 数据集**

在四个公开长周期预测基准上进行评测，分别为ETTh1、Weather、Traffic与Electricity（ECL）

**📈 对比分析**

与多种基线（SE-LLM、TimeMixer++、Time-MoE等）比较，FARR在所有数据集上实现了最低的数据集平均MSE和MAE，且在个别时点的排名虽有波动但总体表现突出

**⚠️ 局限性**

主要局限为需要额外的无效端点预测导致推理时延约35%增加，并且模型仍需冻结基础预测器，未来可探索端点生成的选择性策略与模型蒸馏以降低成本

---

## 397. PlayGen2: Learning Real-Time Playable Worlds with Explicit 3D State

**arXiv ID:** 2611.07867 | [PDF](https://arxiv.org/pdf/2611.07867v1)

**作者:** Jihai Zhang `[一作]` (Chinese University of Hong Kong), Yu Cheng `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练并部署了PlayGen2，一个将世界状态演化与神经渲染分离的实时可玩世界模型，能在Doom中持续更新体素几何、实体状态，并通过动作条件预测与可编程规则实现多玩家和自定义玩法。

**💡 创新点**

创新点包括：1）显式持久的3D世界内存（体素占据格 + 身份索引实体表）；2）联合学习几何补全、实体发现和动作条件状态转移；3）将渲染任务交给独立的latent diffusion模型，避免生成图像被直接反馈到状态预测；4）通过代码规则实现多玩家组合与可编程交互。

**🔧 技术方法**

使用3D卷积编码器+共享上下文Transformer、发现Transformer、转换Transformer；体素占据网格与实体表；深度+语义投影生成的latent diffusion渲染器；ViZDoom收集轨迹；Dice、focal、位置等损失函数；BF16推理与8步去噪。

**📊 数据集**

约500小时（≈500 GB）Freedoom2 MAP01游戏轨迹作为训练集；评估使用Freedoom2 MAP01（已见）和Scythe社区制作的7张未见Doom地图。

**📈 对比分析**

与原PlayGen在同一地图和相同动作序列下比较，采用PSNR/SSIM/LPIPS/FID/FVD；PlayGen2在10 分钟时保持稳定性能（LPIPS≈0.4），PlayGen在5 分钟后显著退化；多玩家一致性评估（与MultiGen对比）准确率≥96%；单机推理帧率≈5 fps，后端延迟≈200 ms，可在MacBook Pro M4 Pro上实时运行。

**⚠️ 局限性**

局限性：仅在Doom类固定布局场景验证，未处理更复杂的物理交互；需要规则控制器收集数据；渲染依赖RGB历史，历史缺失会导致质量下降；实体检测与状态更新仅覆盖预定义类，缺乏实时碰撞纠正。

---

## 398. Quantum Communication and Phase Sensing over a Dephasing Channel with Thermal Noise

**arXiv ID:** 2611.07855 | [PDF](https://arxiv.org/pdf/2611.07855v1)

**作者:** Nikos A. Mitsiou `[一作]` (University of Cyprus), Ioannis Krikidis `[通讯]` (University of Cyprus)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在带有热噪声的von Mises失真通道中，使用相干态BPSK传输时，异频检测器同时完成通信（符号误码）和相位估计（平均相位获取）的性能分析，提出了符号误码概率（SEP）的两点Gauss-Legendre近似以及相位估计的Fisher信息（FI）与量子Fisher信息（QFI）闭式上界/下界，进一步给出了在有限光子预算下通信与相位估计可同时满足的光子分配边界。

**💡 创新点**

创新点在于：1) 将通信误码概率与相位估计误差耦合的量子ISAC问题首次用统一框架分析；2) 推导了异频检测在von Mises失真通道下的SEP闭式近似并揭示了不可消除的误码底限；3) 通过闭式FI与QFI上下界阐明异频检测在大光子数下可逼近量子极限；4) 给出了光子分配的闭式上界/下界，明确了通信与感知可行性区间。

**🔧 技术方法**

技术手段包括：量子光学模型（失真通道、热噪声）、von Mises相位分布、异频检测的统计模型、Gauss-Legendre数值积分逼近、Fisher信息与CRB分析、量子Fisher信息理论与矩方法、闭式贝塞尔函数展开、极限分析与误码底限推导。

**📊 数据集**

本文为理论分析性工作，没有使用公开实验数据集；所有结果均基于解析推导与数值积分仿真验证。

**📈 对比分析**

与传统异频检测在已知相位或无失真通道下的误码率以及与量子极限（QFI）的误差界比较，结果表明：①误码率随光子数增加趋于由失真引起的不可消除误码底限；②在光子数足够大时，异频检测的Fisher信息与QFI几乎相等，即达到量子极限；③光子分配边界与数值仿真吻合，误差约10%以内。

**⚠️ 局限性**

限制主要包括：仅考虑BPSK调制、已知的相位估计符号、已知失真浓度参数κ、仅使用异频检测（未探讨更优量子测量）、未对多符号或多相位分布的通用情况进行分析、光子分配最优问题未给出完整解。

---

## 399. Multiplicative and Additive Extractors for Low Entropy Samplable Sources Under Reduced Assumptions

**arXiv ID:** 2611.07829 | [PDF](https://arxiv.org/pdf/2611.07829v1)

**作者:** Marshall Ball `[一作]` (New York University), Yunya Zhao `[通讯]` (Cornell University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构造了三类新型的提取器，分别针对对数级、亚线性及多项式对数级最小熵的可采样分布，实现了在不同熵阈值下提取近乎全部熵的功能。

**💡 创新点**

创新点包括：① 在非确定性硬件假设下实现对数级熵的单比特提取；② 引入鲁棒无误差凝聚器与乘法提取器，实现了对几乎所有熵的高效提取；③ 通过两源提取器框架和小规模电路的相关性破坏器，扩展了对多项式对数熵分布的提取能力。

**🔧 技术方法**

主要技术手段包括：非确定性电路硬件假设、Σ₂^P 访问电路的硬件假设、鲁棒无误差凝聚器、相关性破坏器、哈希到凝聚、去相关化以及两源提取器的改编。

**📊 数据集**

该工作为理论性质，无使用具体实验数据集；所有结果均通过数学证明与复杂度分析给出。

**📈 对比分析**

与先前工作（如BGDM23、Shaltiel25）相比，本论文在最小熵阈值从0.99n降至对数级或多项式对数级，且所需的硬件假设更弱；提取器的误差可调为多项式小，且提取效率与先前提取器相当或更优。

**⚠️ 局限性**

局限性主要在于：仍依赖强硬件假设；构造复杂度较高，实际实现难度大；对某些特殊可采样分布可能不适用；并未给出对所有熵分布的通用提取方案。

---

## 400. STAG-Net: Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation

**arXiv ID:** 2611.07828 | [PDF](https://arxiv.org/pdf/2611.07828v1)

**作者:** Asiri Gawesha Lindamulage `[一作]` (Sri Lanka Institute of Information Technology), Nuwan Kodagoda `[通讯]` (Sri Lanka Institute of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 STAG‑Net，一种基于结构感知注意力的轻量级头部姿态估计网络，利用面部关键点分组和自注意力实现高精度预测。

**💡 创新点**

创新点包括：自动化的基于运动敏感度的关键点分组与选择、每组使用 DeepSet 编码实现可置换不变性、组间多头自注意力捕捉局部与全局关系、结合紧凑几何描述子增强全局信息。

**🔧 技术方法**

采用的技术包括：基于图聚类的关键点分组、运动敏感度排序选择、DeepSet 结构编码、跨组多头自注意力、几何统计描述子、轻量化全连接回归头，以及学习率余弦退火与 warm‑restart 调度。

**📊 数据集**

使用 300W‑LP（合成）、AFLW2000（标注）和 BIWI（真实）三大数据集进行训练与评估，跨数据集与主体独立两种协议。

**📈 对比分析**

与现有图像基与关键点基方法对比，STAG‑Net 在所有评估场景下均取得最低或第二低的平均绝对误差（例如 3.51°/2.39°），模型参数仅 0.012–0.043 M，FLOPs 仅 24–84 k，GPU 推理速率达 322 FPS，CPU 也可实现 111 FPS。

**⚠️ 局限性**

主要限制在于整体系统性能受关键点检测器的覆盖范围与算力影响，FaceMesh 速度快但极端姿态覆盖不足；FAN 覆盖广但推理速度低；缺乏完整的微控制器级部署评估与量化实现。

---

## 401. Instantiating the Knowledge Plane

**arXiv ID:** 2611.07827 | [PDF](https://arxiv.org/pdf/2611.07827v1)

**作者:** Francisco Pereira `[一作]` (University of Lisbon), Pratiksha Thaker `[通讯]` (Databricks)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用大型语言模型（LLM）构建了一个分布式知识平面（Knowledge Plane），实现跨域协作的网络管理与故障诊断，并通过 Mininet 仿真案例展示了 LLM 代理在故障排除、BGP 拒绝和动态路由决策中的协同推理能力。

**💡 创新点**

首次将 LLM 代理用于多域协作，突破传统固定协议限制，展示自然语言可作为灵活通信介质，并揭示了跨域协同所面临的信任、隐私与可扩展性等新挑战。

**🔧 技术方法**

采用 Claude Opus、GPT‑5.5、Qwen‑2.5‑72B、GLM‑5.2 FP4 等大模型，配合自定义工具（shell 命令、自然语言互通、等待事件），在 Mininet 虚拟网络环境中实现代理间消息传递与操作。

**📊 数据集**

仅使用 Mininet 生成的仿真网络数据，包括故障诊断场景、BGP 劫持拓扑和 ISP 负载均衡配置，未引用公开真实网络数据集。

**📈 对比分析**

对比了四种 LLM 在三类任务（故障诊断、恶意 BGP 检测、契约驱动路由决策）中的表现；高容量模型（Claude Opus、GPT‑5.5）在推理、冲突处理和策略理解上显著优于 Qwen 与 GLM，表明 LLM 能在多代理协同中发挥作用，但仍存在误判和性能波动。

**⚠️ 局限性**

存在计算成本与能耗高、在真实网络规模下的可扩展性不足、对隐私与安全（提示注入、信息披露）控制不完善、缺乏正式安全与可审计机制、以及对真实网络环境验证不足等局限。

---

## 402. CMDiff: Left Atrial Scar Segmentation from 3D LGE MRI Using Conditional Mask Diffusion Model: A Feasibility Study

**arXiv ID:** 2611.07814 | [PDF](https://arxiv.org/pdf/2611.07814v1)

**作者:** Jing Zhang `[一作]` (Sorbonne Université), Nadjia Kachenoura `[通讯]` (Sorbonne Université)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了基于条件扩散模型（CMDiff）对LGE-MRI中左心房纤维化斑块进行3D分割的可行性。

**💡 创新点**

首次将直接清洁掩模预测与动态编码器条件的3D正弦UNet相结合，用扩散过程生成稀疏二值掩模。

**🔧 技术方法**

使用Conditional Mask Diffusion（CMDiff）、3D SinUNet、Dynamic Conditional Encoder、DDIM、UniPC等扩散采样器，以及传统的确定性nnU-Net对照。

**📊 数据集**

在公开的LASCARQS 2022数据集（60例左心房纤维化注释）上进行5折交叉验证。

**📈 对比分析**

与匹配的确定性基线相比，CMDiff在DSC、HD95、ASD等指标上略逊一筹，且推理耗时更长；在瘤负荷评估中误差略大。

**⚠️ 局限性**

受限于样本量小、推理速度慢、未利用额外的心腔标签或显式解剖约束，以及未进一步优化采样策略和目标参数化。

---

## 403. Different Corpora, Different Skills? A Paired Study of Small Language Models

**arXiv ID:** 2611.07812 | [PDF](https://arxiv.org/pdf/2611.07812v1)

**作者:** Edoardo Vaira `[一作]` (Lund University), Pierre Nugues `[通讯]` (Lund University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小规模语言模型上，研究者通过匹配训练字数预算，对1911年版《百科全书》与BabyLM 2026严格语料库两者进行系统比较，观察到不同语料对模型能力的差异；

**💡 创新点**

创新点在于构建匹配字数、种子、架构和训练流程的对照实验，展示语料库选择会导致模型在多项能力上出现“反转”效应，并对知识蒸馏的影响进行了细致拆解；

**🔧 技术方法**

使用了基于BabyLlama的58M小模型，采用Llama-360M、GPT-2-705M、GPT-J-705M等教师模型，进行知识蒸馏；还使用BPE分词、AdamW优化、cosine学习率调度等技术；

**📊 数据集**

主要数据集为1911年版《百科全书》文本（约10M/37M词）与BabyLM 2026严格语料库（约10M/37M词），两者分别在单一源和混合源两种语料下训练；

**📈 对比分析**

比较方法是采用零样本评估在五个子基准（BLiMP、BLiMP Supplement、EWoK、Entity Tracking、COMPS）上的表现，结果显示BabyLM在语法可接受性、语义和形态学任务上优势明显，而《百科全书》在实体跟踪任务上表现更佳；

**⚠️ 局限性**

局限性包括只研究两种语料库、仅用两组随机种子、token与词数不匹配、实体跟踪基准为人工合成、以及模型规模与蒸馏设置可能不具普适性。

---

## 404. Dataset Collection and Fine-Tuning of NLLB for German--English Translation

**arXiv ID:** 2611.07797 | [PDF](https://arxiv.org/pdf/2611.07797v1)

**作者:** SK Deeraj `[一作]` (Sardar Vallabhbhai National Institute of Technology), Pruthwik Mishra `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对NLLB-200模型在德英文学与技术文本上进行微调，并将微调后模型集成至LinguaHaru文档翻译系统，实现格式保持的离线翻译。

**💡 创新点**

①使用八阶段筛选流程将45k原始OPUS句对压缩为25k高质量对；②在仅4GB消费级GPU上实现可复现的微调，结合FP16、梯度累积与检查点；③将微调模型无缝嵌入LinguaHaru，实现文档级格式保留。

**🔧 技术方法**

Transformer NMT、NLLB-200-distilled-600M、混合精度训练、梯度累积、梯度检查点、HuggingFace datasets & Trainer、BLEU/chrF++/TER评估、配对bootstrap显著性检验、PyMuPDF/python-docx文档处理。

**📊 数据集**

OPUS Books的德英平行语料，原始45k句对经筛选后得到25k高质量对，随后划分为train/val/test。

**📈 对比分析**

与基线NLLB-200在同一测试集上做BLEU/chrF++/TER对比，微调后BLEU提升3.4分（32.4→35.8），chrF++提升2.6分，TER下降3.2点，差异在p<0.01统计显著。

**⚠️ 局限性**

仍存在罕见术语、尴尬表述、代词性别错误等误差，缺乏文档级上下文，微调规模受限于25k样本；模型未做量化，部署在极低端设备上仍有难度。

---

## 405. Structure from Collision via Hypothesis-Space Reasoning

**arXiv ID:** 2611.07900 | [PDF](https://arxiv.org/pdf/2611.07900v1)

**作者:** Takuhiro Kaneko `[一作]` `[通讯]` (NTT, Inc.), Takuhiro Kaneko (NTT, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过碰撞导致的形变推断物体内部空腔结构，提出基于假设空间推理的两阶段框架；

**💡 创新点**

创新点在于将内部空腔作为显式结构假设进行全局搜索，并在粒子级别进行梯度优化，解决了密度更新导致的定位模糊；

**🔧 技术方法**

使用前向无梯度的MPM模拟进行全局假设评估，随后采用可微MPM与Lagrangian粒子优化，并结合几何约束损失；

**📊 数据集**

使用SfC基准（40个序列、5种外形）和Topology数据集（60种复杂拓扑）进行评估；

**📈 对比分析**

与SfC-NeRF和GIC对比，在尺寸与位置变化场景下CD/F-score显著优于基线，未来预测的PSNR/SSIM亦明显提升，尤其在长时间预测中表现突出；

**⚠️ 局限性**

局限包括对合成数据的依赖、对复杂拓扑恢复仍受限、假设空间需先验定义、对真实世界校准与模型不匹配敏感。

---

## 406. Which Signals of Code Correctness Survive Distribution Shift? Hidden-State and Causal Predictors of Program and Verdict Correctness

**arXiv ID:** 2611.07876 | [PDF](https://arxiv.org/pdf/2611.07876v1)

**作者:** Viola Campos `[一作]` (RheinMain University of Applied Sciences), Adrian Ulges `[通讯]` (RheinMain University of Applied Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对六种执行无关的代码正确性预测方法（输出概率、样本一致性、隐藏状态探针、LAT、因果子空间DAS、值探针）进行了系统评估，探究它们在不同分布和语言偏移下对程序正确性和判定正确性的预测能力。

**💡 创新点**

创新点在于首次将自然语言任务中的因果定位技术迁移到代码评估，并证明在语言偏移下对判定正确性具有更高鲁棒性；同时在同一实验框架下对四类预测器进行了全面对比。

**🔧 技术方法**

使用的技术包括模型输出概率、样本一致性测度（EigenScore）、隐藏状态线性探针、Principal Component Analysis 的 LAT 方法，以及基于分布式对齐搜索（DAS）的因果子空间学习和相应的对抗模拟/值探针，结合逻辑回归和统计显著性检验。

**📊 数据集**

采用了 LiveCodeBench、BigCodeBench 以及它们的 Python/C++/Java 对应版本作为训练和评估数据集，对候选程序进行自动生成、判定和执行测试。

**📈 对比分析**

通过对每种方法在候选层面和问题层面计算 AUROC，并使用配对引导重采样检验显著性，结果显示：在分布内隐藏状态方法表现最佳；在语言偏移下 Qwen3-Coder 的因果子空间值探针平均领先其他方法约 0.04 AUROC；程序正确性在不同 benchmark 间迁移更好，但整体优势有限。

**⚠️ 局限性**

局限性包括：仅在两种开源 LLM 上验证，因果子空间在不同模型间表现不一致；预测主要识别问题难度而非在同一问题内候选排序；对测试环境、标注质量和模型推理方式敏感，且未涵盖更广泛的模型或任务场景。

---

## 407. PVDepth: Panoramic Video Depth Estimation via Geometry-Aware Spatiotemporal Adaptation

**arXiv ID:** 2611.07873 | [PDF](https://arxiv.org/pdf/2611.07873v1)

**作者:** Chuanxin Song `[一作]` (Peking University), Peixi Peng `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出PVDepth框架，实现全景视频的时空一致深度估计。

**💡 创新点**

创新点在于：① Progressive Sphere‑aware Noise Initialization 通过逐步迁移噪声分布克服ERP几何差异；② Cube‑rectified Temporal Modeling 利用无失真立方体投影补偿ERP中的非线性运动；③ 引入大规模合成全景RGB‑D视频数据集PanoCARLA。

**🔧 技术方法**

技术主要包括：扩散式深度生成模型（DepthCrafter）、自适应噪声初始化、立方体投影与双向映射、跨模态CLIP语义引导、基于Transformer的时空注意力。

**📊 数据集**

使用CARLA模拟器合成的PanoCARLA数据集，包含约12.6万帧、117条长视频，覆盖多条城市道路与无人机视角。

**📈 对比分析**

与单帧、后处理与视角分解等方法比较，PVDepth在AbsRel约0.199、δ1约0.798的指标上显著优于现有基线，且在开源全景视频上表现出更强的时间一致性。

**⚠️ 局限性**

局限性包括：推理速度与算力开销较高，尤其相较于单帧方法；在室内环境与强反射表面上泛化能力有限。

---

## 408. Towards Foundation Decision Models: Drex

**arXiv ID:** 2611.07854 | [PDF](https://arxiv.org/pdf/2611.07854v1)

**作者:** Nischay Dhankhar `[一作]` (Nace.AI Research Team), Abulhair Saparov `[通讯]` (Nace.AI Research Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的多模态决策模型 Max，结合指针头和读出头，能够在单句文本和图像混合的请求上直接给出答案概率。

**💡 创新点**

创新点在于将全注意力层与密集 Transformer 混合，利用读出头将每个选项编码为单个标记，从而实现对所有选项的双向全局上下文比对，并支持在同一请求中缓存状态，显著提升多问句的推理效率。

**🔧 技术方法**

采用 Qwen3.8‑27B 大模型作为背骨，Gated DeltaNet 与全注意力层交替，配合指针头（W_q,W_k）或读出头（W_r），并使用 RLAF 与 SFT 训练策略，加入温度调优。

**📊 数据集**

训练数据来源于 1150k 文本决策记录（11 个领域）以及图像问答数据，评估基准包括 Decision Index、Decision Bench v4、JevBench‑231、OpenSpiel、OSWorld 等。

**📈 对比分析**

与 v1.5、v2.0、Jev 1.14.0 等基线以及 GPT‑6 Sol、Claude Opus 5.5 等前沿聊天模型对比，在 Decision Index 公共子集上 Max 以 67.5% 的 skill 位居第一，在 Decision Bench v4 上实现 73.5% 的准确率，并且 ECE 低于大多数基线。

**⚠️ 局限性**

局限在于对结构化长文本和 Arts 领域的任务表现下降，读出头的符号绑定仍可能导致对选项顺序敏感，且模型对图像的解释仍依赖可见的视觉信息。

---

## 409. Tasteply: Externalizing Stable and Situational Preference as Manipulable Spatial Objects

**arXiv ID:** 2611.07851 | [PDF](https://arxiv.org/pdf/2611.07851v1)

**作者:** DaEun Choi `[一作]` (KAIST), Daniel Lee `[通讯]` (Salesforce AI Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个将用户偏好外化为二维空间中可操作的“吸引子”和全局偏好场的推荐系统，让用户可以通过移动位置和调节全局强度来实时控制推荐内容；

**💡 创新点**

创新点在于将稳定偏好与情境偏好分别映射为可视化的吸引子与全局场，直接暴露给用户进行交互，而非隐藏在模型内部或通过代理参数调节；

**🔧 技术方法**

采用文本嵌入聚类生成吸引子向量，使用高斯核对吸引子加权求和生成查询向量，结合全局向量与可调γ指数的余弦相似度评分，以及前端二维交互式可视化；

**📊 数据集**

使用用户自行提供的社交媒体使用历史（Bluesky、X、Threads等平台的点赞内容），并未使用公开标准数据集；

**📈 对比分析**

通过对5名HCI研究员进行15分钟自由探索和25分钟半结构化访谈的定性评估，未与传统推荐算法做数值对比，但发现系统在偏好可视化、连续混合与用户主动调节方面表现良好；

**⚠️ 局限性**

局限性包括样本量极小、结果缺乏统计显著性、连续混合控制在目标导向任务中效果有限、未实现长期学习与实时更新机制、且缺乏客观性能指标。

---

## 410. Generate-then-Rank: Learning Symbolic Goal Interpretation for Embodied Agents

**arXiv ID:** 2611.07825 | [PDF](https://arxiv.org/pdf/2611.07825v1)

**作者:** Jisen Lei `[一作]` (Newcastle University), Huizhi Liang `[通讯]` (Newcastle University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段的Generate-then-Rank方法，用于将自然语言指令转换为可由规划器执行的符号化目标；

**💡 创新点**

创新点在于引入CORE（Candidate‑Output Ranking from Errors）——基于模型自身生成错误进行对比学习的候选排序机制；

**🔧 技术方法**

技术包括：对大语言模型进行LoRA适配的监督微调（SFT）以生成完整目标，随后利用对比损失对生成的候选进行排序，并使用长度归一化的似然得分；

**📊 数据集**

数据集为EAI（Embodied Agent Interface）基准中的342条公共目标解释记录，划分为274条训练集与68条诊断集，涵盖五个任务族；

**📈 对比分析**

与直接提示、链式思维、知识图谱前缀、DPO等基线进行比较。实验表明，在诊断集上，CORE将Combined F1从0.716提升至0.736，Edge F1提升至0.887，超过SFT基线及实现的DPO（Combined 0.735、Edge 0.887、Action 0.681），但在Action F1上仍略逊于DPO；

**⚠️ 局限性**

局限性包括：仅在单一随机种子和重复检查的68条诊断集上验证；CORE与DPO在对比样本构造与学习率上不一致，难以单独评估损失效能；未对候选生成质量进行系统提升；缺乏多种种子、跨任务族泛化评估及列表式排名或约束解码的比较。

---

## 411. MV-Sign: Multi-View 3D Motion Integration and Enhanced Text-to-Motion Conditioning for Sign Language Production

**arXiv ID:** 2611.07811 | [PDF](https://arxiv.org/pdf/2611.07811v1)

**作者:** Singyu Muk `[一作]` (Independent Researcher), Kalin Stefanov `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出MV‑Sign框架，利用多视角数据纠正噪声3D手语运动并生成文本到手语的运动序列。

**💡 创新点**

创新点在于：①异步多视角运动缝合管道用于去噪；②联合体–手势的ST‑VQVAE分解；③伪词义适配器与关键帧检索相结合提升文本‑运动对齐。

**🔧 技术方法**

使用的技术包括AiOS+WiLoR融合、时空对齐、手势修复、ST‑VQVAE、GPT‑5.2伪词、TCN+IG关键帧检索、mBART‑large自回归生成。

**📊 数据集**

数据集有How2Sign‑MV3D（重新生成的高质量3D）、ASLCitizen‑3D、MM‑WLAuslan‑MV3D、SGNify‑MV以及原How2Sign等。

**📈 对比分析**

与NSA、SOKE等现有方法在How2Sign数据空间评估比较，本文在体部DTW‑PA‑JPE下降32.8%、手部下降20.9%，实现最优性能。

**⚠️ 局限性**

局限性：多视角仅在正面视角模糊时有效；侧视图时间错位或干扰可能引入噪声；检索使用平均显著性，未充分利用空间信息；系统仅作为辅助工具。

---

## 412. Rain Beyond Streaks: Cascaded Realistic Rain and Wet Surface Synthesis via Scene Priors

**arXiv ID:** 2611.07793 | [PDF](https://arxiv.org/pdf/2611.07793v1)

**作者:** Melissa Tijink `[一作]` (University of Twente), Nicola Strisciuglio `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套无训练、基于深度和语义先验的雨水合成流水线，能够在真实场景中生成雨滴、雾、反射与积水，构建了 Cityscapes‑CascadeRS 真实雨数据集。

**💡 创新点**

创新点在于：① 将场景深度与语义分割作为先验，按层次（反射→积水→雾→雨滴）进行物理启发式渲染；② 采用可控参数实现多种雨强度与视角；③ 通过预训练模型实现无监督生成，显著提升感知真实感并可直接用于下游任务评估与数据增强。

**🔧 技术方法**

核心技术包括预训练深度估计（MiDaS/DepthAnything）、语义分割（SegFormer/FastSeg）、Perlin 噪声生成积水形状、视角映射实现反射、气象散射模型渲染雾、运动模糊与高斯模糊生成雨滴；所有步骤均为程序化，不需要额外训练。

**📊 数据集**

主要使用了 Cityscapes（生成合成雨数据），以及 RID、ACDC、KITTI、BDD100K、nuScenes、Roadmarks 等公开数据集进行评估与验证。

**📈 对比分析**

通过用户研究、LLM‑as‑judge、FID/KID 等指标与真实雨图对比，CascadeRS 在 FID 上优于 PBR 与 Cityscapes‑Rain，用户感知最高；在语义分割任务中对 SegFormer、FastSeg 等模型的 mIoU/FWIoU 有显著提升；在目标检测任务中对 DINO Teacher 的性能提升有限，但能复现真实雨对检测的负面影响。

**⚠️ 局限性**

局限性包括：仅针对白昼光照，未模拟夜间低光与镜头雨滴；对检测任务的鲁棒性提升有限；对雨滴细粒度控制（如尺寸、落点密度）仍不够细致；依赖预训练模型的质量，若深度/语义分割误差较大，渲染质量会受影响。

---

## 413. On computing the determinant over a ring

**arXiv ID:** 2611.07786 | [PDF](https://arxiv.org/pdf/2611.07786v1)

**作者:** Gilles Villard `[一作]` `[通讯]` (CNRS), Gilles Villard (CNRS)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种在任意交换环上无除法且无零检验的算法，用O(n^ω)环运算计算方阵的行列式和特征多项式；

**💡 创新点**

将多项式矩阵视角与块 Krylov 视角相结合，并通过分块压缩与 t‑变量同胚实现无除法的 O(n^ω) 复杂度；

**🔧 技术方法**

使用同胚（homotopy）L(t)=M-tAM、块 Krylov 序列、矩阵生成多项式、Hankel 系统求解、Newton 迭代、矩阵多项式乘法和中间乘积；

**📊 数据集**

本工作为理论算法，不涉及具体数据集；

**📈 对比分析**

与已知的 O(n^ω+1) 或 (n^ω/2+2) 等算法相比，本算法在理论上实现了最优 O(n^ω) 复杂度，并且控制流程仅依赖于 n；

**⚠️ 局限性**

算法实现较为复杂，需先行构造合适的投影与同胚；在实际应用中对矩阵乘法的依赖及对环运算的实现细节可能影响效率；

---

## 414. XM4D: 4D Reconstruction through Certifiable Geometric Optimization

**arXiv ID:** 2611.07902 | [PDF](https://arxiv.org/pdf/2611.07902v1)

**作者:** Haoyu Han `[一作]` (Harvard University), Heng Yang `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套完整的4D重建框架，先用深度预测和点跟踪得到视图图，随后通过可证全球最优的几何优化（QCQP→SDP→Burer–Monteiro+Riemannian阶梯）分别求解静态相机/深度尺度/静态点云和动态形变图（嵌入变形图）的位姿与运动；最后以此结构初始化动态高斯喷射进行新视角渲染。

**💡 创新点**

核心创新包括：①将相机运动与形变分离，均转化为仅含旋转的QCQP；②利用统一球面约束解决尺度歧义；③在动态阶段引入附加的线性齐次化技巧，将线性项嵌入辅助旋转块，实现与静态问题同构的可证优化；④在整个流程中保持可证最优，避免局部陷阱。

**🔧 技术方法**

使用的主要技术包括：深度预测（Depth Anything 3）、点跟踪（TAPNext）、Mask预测（SAM）、视图图构建、嵌入变形图（EDG）与线性混合蒙皮、Burer–Monteiro低秩因式分解、Riemannian阶梯求解SDP、动态高斯喷射渲染（gsplat+EDG包装）。

**📊 数据集**

实验数据集涵盖七个基准：静态/动态照片集IMC、Replica、TUM、Sintel；动态视频集Syn4D、NVIDIA、DyCheck；并在BAL等场景验证算法稳健性。

**📈 对比分析**

与COLMAP、VGGT、MegaSAM、MoSca、Shape of Motion以及多种时间可变场/高斯渲染基线比较，本文方法在相机轨迹精度、运行时（尤其是动态阶段最快）以及新视角渲染（mPSNR、mSSIM、mLPIPS）均优于或匹配最强基线，尤其在DyCheck、NVIDIA、Syn4D三大动态基准上取得最佳综合分数。

**⚠️ 局限性**

主要局限：依赖单目深度预测与点跟踪，若两者误差大会影响最终结果；SDP求解器在存储数据矩阵时为密集结构，导致内存随帧数二次增长；EDG拓扑与蒙皮权重在求解前固定，无法与运动同步学习，未来可改进为联合优化。

---

## 415. Compositional Loop Invariant Inference for Multiphase Programs via Structured Neurosymbolic Reasoning

**arXiv ID:** 2611.07882 | [PDF](https://arxiv.org/pdf/2611.07882v1)

**作者:** Mingxiu Wang `[一作]` (Macquarie University), Xiao Cheng `[通讯]` (Macquarie University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多相循环程序提出一种结构化神经符号推理框架，用来自动推导循环不变量。

**💡 创新点**

创新点在于：①组合推理——将循环按分支序列拆分为若干子执行场景，让LLM一次只生成一个子不变量；②基于SMT的局部修正——通过对counterexample的分支追踪，仅修正导致失败的子不变量，保持其他子不变量不变。

**🔧 技术方法**

技术实现包括：CFG解析提取循环分支序列，LLM（GPT‑5.1）生成子不变量并评估，树搜索与beam search构造全局候选，SMT求解器（Z3）验证并提供counterexample用于局部修正。

**📊 数据集**

使用的实验数据集共420个验证任务：316个线性算术基准，50个非线性算术基准，54个来自CHC‑COMP的多相程序基准。

**📈 对比分析**

与多种LLM、学习和传统工具（如PAL2Inv、ImplCheck、Eldarica等）对比，框架在多相程序上解决了35/54个任务，比PAL2Inv提升20.7%，比最佳传统方法提升59.1%；在线性/非线性基准上分别解决了311/316和44/50个任务，表现优于或与现有工具持平；平均运行时间与API成本与PAL2Inv相当或更优。

**⚠️ 局限性**

局限性包括：仍难处理周期性（如取模）分支导致的循环；局部修正的分支判定是启发式，可能误判；LLM的随机性和多分支场景下的搜索空间导致运行时增长；对全局计数或跨分支依赖的非分支结构不够成熟。

---

## 416. Do Recommenders Need to Think in Words? Latent Reasoning for Efficient LLM-Based Recommendation

**arXiv ID:** 2611.07866 | [PDF](https://arxiv.org/pdf/2611.07866v1)

**作者:** Hongyang Su `[一作]` (Tencent), Zang Li `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种 LLM‑基生成推荐框架 LatentRec，通过训练时对每一步隐藏状态进行预测头监督，消除推理时的链式思维（CoT）文本或隐式推理，从而在保持或提升准确率的同时显著降低推理延迟。

**💡 创新点**

创新点在于：①将每一步隐藏状态的监督直接投射到 LLM 的共享预测头上（Per‑step Reasoning Loss）；②引入 Soft Token Selection 解决隐藏状态与输入嵌入的 14× 范数不匹配问题；③通过 2.5M 参数的 Continuous Thought Module 在训练阶段完成多步隐式推理，推理时可完全去除该模块，避免每次推荐的额外开销。

**🔧 技术方法**

使用 Qwen3‑1.7B LLM，结合 LoRA 微调、Soft Token Selection、Per‑step Reasoning Loss、以及在训练中引入的 Continuous Thought Module；推理阶段仅使用标准 SID 解码器。

**📊 数据集**

在 Amazon Review 2014 公开数据集的 Beauty、Toys、Sports 三个子域上进行评估，采用 5‑core 留一评估，完整语料库排名，beam=50。

**📈 对比分析**

与 GRU4Rec、SASRec、BERT4Rec、HGN、TIGER、ReaRec 等传统与生成式推荐器，以及基于 CoT 的 LLM‑CoT SFT 基线进行比较。LatentRec 在 HR@10 上相较 CoT SFT 提升 4.0–5.0%（相对），同时推理延迟下降 18.3×（K=0）或 17.7×（K=1），并在 HR@5、NDCG 等指标上保持或优于基线。

**⚠️ 局限性**

局限性包括：仅在单一 Qwen3‑1.7B LLM、单一 GPT‑4 CoT 教师、以及 Amazon 电子商务三个域上验证；未探讨跨模型、跨教师、跨领域泛化；未针对非 SID 生成推荐器进行扩展；对更深层 SID 级别的监督尚未实现。

---

## 417. OcTreeMamba: Topology-Aware Mamba for 3D Medical Segmentation via Octree Sparsification

**arXiv ID:** 2611.07865 | [PDF](https://arxiv.org/pdf/2611.07865v1)

**作者:** Kaixiang Mai `[一作]` (Guangdong University of Technology), Yiu-ming Cheung `[通讯]` (Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 OcTreeMamba，结合 Octree 结构与 Mamba（State Space Model）实现 3D 医学图像分割的拓扑感知长程上下文传播。

**💡 创新点**

创新点在于：①使用可微分的结构学习器通过多粒度门控与 Gumbel‑Softmax 选取前景体素；②构建动态 Octree 生成层次化稀疏拓扑；③改进的双向 Tree‑SSM 在该树上实现上下文聚合与广播，既捕获全局依赖又保持局部拓扑；④在 Mamba 框架下首次将稀疏树结构与线性时序建模结合。

**🔧 技术方法**

技术核心包括：Octree 结构、可微分稀疏化、Gumbel‑Softmax、层次化门控、双向 Tree‑SSM（基于状态空间模型）、残差融合、混合损失（交叉熵 + 结构稀疏 + 辅助监督）。

**📊 数据集**

在四个公开基准上验证：ImageCAS（冠状动脉分割）、TopCoW2024（Circle of Willis 颈内动脉分割）、BraTS2021（脑肿瘤分割）和 AbdomenCT（腹部多器官分割）。

**📈 对比分析**

与 CNN（nnU‑Net、MedNeXt）、ViT（UNETR++、SegFormer3D）以及稠密 SSM（U‑Mamba、SegMamba）对比，OcTreeMamba 在血管数据上显著提升 Dice/ClDice、减小 HD95，整体保持或略优的分割精度；在计算效率上相较于稠密 Mamba 提升 1.8–2.0× 速度、降低 40–47% 训练内存，参数量与 nnU‑Net 差距极小。

**⚠️ 局限性**

局限性：①对非血管类稀疏结构（如肿瘤、器官）提升有限；②需要额外的结构学习与 Octree 构建开销；③在浅层特征上稀疏建模可能导致细节损失，需进一步调优多尺度集成策略。

---

## 418. Learning Robotic World Models with Semantic and Visual Feedback

**arXiv ID:** 2611.07850 | [PDF](https://arxiv.org/pdf/2611.07850v1)

**作者:** Hao Wu `[一作]` (Carnegie Mellon University), Ming-Hsuan Yang `[通讯]` (University of California Merced)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种通过候选未来比较实现的自回归机器人世界模型后训练框架

**💡 创新点**

将分组多样化束搜索与冻结的多模态评判结合，利用语义与视觉双重反馈形成可标准化奖励，实现对候选轨迹的比较学习

**🔧 技术方法**

GPT风格自回归Transformer、离散视觉与动作标记、T5文本交叉注意力、分组多样化束搜索(GDBS)、冻结多模态评判mJev、LPIPS/SSIM/PSNR评价、PPO式剪切优势优化

**📊 数据集**

RT-1、BridgeV2、RoboCasa 三大机器人操控视频数据集

**📈 对比分析**

与多类基线（闭源/开源视频生成器、动作条件世界模型、RL后训练模型、任务适配视频基座模型）对比，实验显示在视觉预测和任务一致性指标上均取得最高分（PSNR提升≈2 dB，任务评分≈8.98/10），且后训练显著优于单纯继续监督微调

**⚠️ 局限性**

仅在离线视频预测上验证，未测试闭环规划或真实机器人控制；后训练计算开销大，等价计算预算下不一定高效；需进一步评估下游能力并降低成本

---

## 419. Computation, not weight access: large language models as auditors of clinical prediction networks

**arXiv ID:** 2611.07847 | [PDF](https://arxiv.org/pdf/2611.07847v1)

**作者:** Zhiyu Liu `[一作]` (Tianjin Medical University), Xinyu Yang `[通讯]` (Tianjin Medical University General Hospital)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过构建仿真临床数据，训练不同规模的 Tanh MLP，评估前沿 LLM（如 Astra、Claude Opus 5.5 等）在不借助工具、仅加权计算器、或仅获得模型预测的条件下，对网络权重的读取、模型审计与修复能力；同时与脚本化方法进行对比；进一步探讨了系统指令与工具可用性对模型表现的影响。

**💡 创新点**

1) 明确区分无工具读取与工具辅助两种情境，揭示前者仅适用于极小网络；2) 发现仅需计算工具（即便不暴露权重）即可实现对 3.8 万权重网络的准确读取与审计；3) 通过系统指令调整消除默认模式下的拒绝现象，突出指令对 LLM 任务完成率的关键作用；4) 将 LLM 结果与传统脚本化方法对齐，指出 LLM 在设计实验与自然语言报告方面的增值。

**🔧 技术方法**

- 前沿 GPT‑6 语言模型（Astra、gpt‑6‑1‑sol、gpt‑6‑sol、gpt‑6‑luna 等）
- 自定义计算器工具（白盒权重或黑盒预测）
- 多尺度 Tanh MLP（105–38,401 权重）
- 自动化评估脚本（精确扫描、回归树、穷举搜索、梯度微调、脚本门控）
- 预先设定的评分规则与内部预注册实验设计。

**📊 数据集**

基于 20,000 名患脑蛛网膜下腔出血的虚拟患者构成的模拟队列；10 个临床输入与 10 个实验室变量，手工设定的 logistic 生成模型与植入的阈值、交互或无关系；另构造 209 权重网络以模拟标签偏差。

**📈 对比分析**

对比指标：读取准确率（是否成功识别植入关系）、R²、强度比；审计准确率（Jaccard ≥0.5、符号正确、半强度）；修复效果（子组对数优势变化、外部平均变化、残余偏差）。实验显示：无工具时，Astra 仅能在 353 权重网络中稳定读取；使用计算器（无论白盒/黑盒）则在 353–38,401 权重网络中都能 100% 读取并审计；脚本方法在大多数任务上与 LLM 接近，甚至略胜；修复任务表现最弱，LLM 受限于形状约束与容差。

**⚠️ 局限性**

- 网络规模有限，均为小型多层感知器；不涵盖树模型、图模型或大规模预训练模型；
- 植入关系与偏差均为简单已知形式，真实临床模型的复杂性难以复制；
- 仅使用模拟数据，缺乏真实数据验证；
- 实验多为内部预注册，结果可重复性待验证；
- 工具与指令设定可能与未来 LLM 版本产生偏差；
- 对修复任务的评估受限于脚本基线与 LLM 约束，难以完整展示差异。

---

## 420. Decodability Is Not Validity: Evaluating Nuisance Sensitivity in Language Model Concept Probes

**arXiv ID:** 2611.07805 | [PDF](https://arxiv.org/pdf/2611.07805v1)

**作者:** Ayal Klein `[一作]` `[通讯]` (Ariel University), Ayal Klein (Ariel University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究隐藏层读出/探针在不同设置下的构造效度，比较在解码准确性相同的提取器对噪声属性的敏感性，并提出基于发展诊断的选择方法。

**💡 创新点**

1) 发现解码准确性并不可靠地衡量对噪声属性的独立性；2) 引入判别诊断和无对抗性编辑的开发筛选；3) 通过多设置实验证明训练数据相关性决定噪声依赖，提供了新的评估框架。

**🔧 技术方法**

线性探针（Logistic、PLS、均值差分）、噪声消除探针；隐藏状态标准化；输入编辑、LLM评审、placebo对照；评价指标包括AUROC、噪声依赖度、编辑响应幅度等。

**📊 数据集**

CEBaB（餐厅评论方面情绪）、BiasBios（医生/护士传记性别）、Amazon负面评论（否定和第一人称两种噪声）以及附录中的Civil身份交换。

**📈 对比分析**

在四个设置、两模型（Qwen2.5‑7B、Gemma‑2‑9B）下，训练不同相关度的提取器；通过开发集最大AUROC、最大噪声诊断和最小绝对编辑变化三种标准进行比较。结果显示准确性匹配的提取器在噪声敏感度上差异可达数倍；开发诊断能显著降低噪声敏感度，且对准确率的损失低至几个百分点。

**⚠️ 局限性**

1) 仅检验单一或两种噪声，未评估其他潜在噪声；2) 编辑筛选依赖LLM评审，未完全由人工验证；3) 未测试非线性探针、SAE特征等更复杂模型；4) 归一化敏感度对弱提取器不稳；5) 训练相关性通过重采样产生，可能与自然分布不完全一致。

---

## 421. VocabulARy 2.0: A Generative AR System Enabling Personalised 3D Mnemonics for Vocabulary Learning in Context

**arXiv ID:** 2611.07798 | [PDF](https://arxiv.org/pdf/2611.07798v1)

**作者:** Karolina Trajkovska `[一作]` (University of Primorska), Klen Čopič Pucihar `[通讯]` (University of Primorska)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了VocabulARy 2.0，允许学习者在未预先准备的环境中通过扫描现实场景生成个性化的3D记忆辅助工具，将词汇与现实物体关联。

**💡 创新点**

实现了基于生成式AI的实时3D记忆生成与AR渲染，支持即时生成而非预制模型，使用户能够根据自己的描述即时创造记忆锚点。

**🔧 技术方法**

结合Unity 3D + Meta XR SDK、FastAPI、Grounding DINO、TRELLIS.2、Gemini LLM、图片编辑模型等技术实现视觉识别、翻译、图像编辑、3D网格生成与AR交互。

**📊 数据集**

利用目标语言自动翻译的本地词汇表以及Grounding DINO训练集进行物体检测与验证，具体公开数据集未列明。

**📈 对比分析**

没有与其它系统直接对比实验，性能评估显示平均生成时间约1分55秒，显著延迟导致演示需预生成；渲染稳定性受网格优化影响。

**⚠️ 局限性**

生成时间长、3D网格未优化导致渲染抖动与跟踪问题、使用键盘输入记忆描述不便，以及缺乏可用性与学习效果评估。

---

## 422. Thinking Is Not Writing: Designing AI Chatbots to Connect Reasoning and Revision

**arXiv ID:** 2611.07792 | [PDF](https://arxiv.org/pdf/2611.07792v1)

**作者:** Lucile Favero `[一作]` (ELLIS Alicante), Nuria Oliver `[通讯]` (ELLIS Alicante)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一款名为 MAIKE 的教育型对话机器人，利用 Socratic 方式支持学生在论证写作中先进行思考（Think）再进行修改（Write），并保持学生对文本内容的完整责任。

**💡 创新点**

创新点在于：①将论证诊断、针对性问题生成、学生模型和修订审查集成到一个完整的交互循环；②明确区分思考与写作两种模式；③通过交互历史和修订回溯保持对话与文本的连续性；④不提供直接答案或文本重写，而只提供问题引导，促使学生自己推理与改写。

**🔧 技术方法**

技术手段：基于 Prompt 的大语言模型（如 8B Open‑Weight LLM）；论证成分提取与质量标注；特质评分（逻辑、修辞、辩证）；多重生成器与判别器产生 Socratic 关键问题；学生模型更新与教学策略选择；编辑回顾与对话继续。

**📊 数据集**

数据集：65 名瑞士中学（IB 课程）学生的两份（初稿与终稿）英文论证作文，配合完整的交互日志、按钮点击、键盘轨迹等。

**📈 对比分析**

对比方法：随机双组（MAIKE vs 对照聊天机器人，均使用相同 LLM 与界面），对 65 名学生进行 2 小时课堂实验。结果显示：两组在复合评分（逻辑、修辞、驳斥三维度）提升比例均约 40%；MAIKE 组无复合评分下降，而对照组出现 9% 的下降；MAIKE 组在“想法层面发言→实质性编辑”链条上的学生更易提升，关联显著。

**⚠️ 局限性**

局限性：样本量小、仅一次实验、单一作文题目、未考察长期学习与迁移、仅测量单篇作文变化；诊断流水线误判可能导致学生失信；缺乏对照组在思考/写作分离的独立评估；未能在多轮或无 AI 的情境中检验持久效果。

---

## 423. EnergyBench: a live, gate-aligned benchmark of day-ahead forecasts for European power markets

**arXiv ID:** 2611.07784 | [PDF](https://arxiv.org/pdf/2611.07784v1)

**作者:** Evgenii Genov `[一作]` (Vrije Universiteit Brussel), Christoph Bergmeir `[通讯]` (University of Granada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在欧洲日间市场上设计并运行了一个实时能源预测基准平台 EnergyBench，实时收集门槛前公开信息并对模型进行同一输入下的预测并排名。

**💡 创新点**

创新点在于将实时门槛时间、冻结的公开信息集、NWP 天气协变量以及基于 Elo 的比赛排名相结合，避免后向泄露并提供可复制的公开记录。

**🔧 技术方法**

采用基准的评分方法包括相对 MAE、RMSE、WIS 以及每个问题的 Bradley–Terry Elo 排名，模型包括传统基线、梯度提升、线性模型、18 种时间序列基础模型以及多模型集成。

**📊 数据集**

使用的数据集为 ENTSO‑E 公开的 15‑分钟欧盟电价、产量、负荷数据，以及三源 NWP（ICON‑EU、GFS、IFS HRES）和每日燃料成本/碳排放。

**📈 对比分析**

通过对 32 个实时日（2‑11 Oct）在 6 区 24 个时序的评估，发现加入天气协变量可将基础模型误差减少 37–51%，零射击基础模型与训练的 LightGBM 同样或更优，集成模型在整体误差中位数最低。

**⚠️ 局限性**

局限包括样本周期短、季节波动大、天气协变量来源单一、以及对不同目标定义/来源的不一致性导致的评估差异。

---

## 424. Enhanced MRI brain tumor detection using deep learning in conjunction with explainable AI SHAP based diverse and multi feature analysis

**arXiv ID:** 2611.07843 | [PDF](https://arxiv.org/pdf/2611.07843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 425. AffordJEV: Affordance Grounding by Confirming Candidate Regions

**arXiv ID:** 2611.07896 | [PDF](https://arxiv.org/pdf/2611.07896v1)

**作者:** Jincheng Li `[一作]` (Peking University), Mengyuan Liu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于 System One 模型的 affordance grounding 方法，直接对候选区域做判定，避免链式推理文本生成；设计了层级推理框架和 Region‑Set Judge，能够在对象、部件、子部件层级上交互式地确认区域。

**💡 创新点**

① 层级级别推理框架：通过上升和下降证据传递，让对象判断与部件判断互相纠正；② 专门为该框架设计的 Region‑Set Judge：利用候选区域的像素特征、几何信息和答案嵌入读取，省去额外判别头；③ 仅用约 3 万个框架生成的请求，从原始 Qwen3.5‑4B 权重即可训练，显著降低样本需求。

**🔧 技术方法**

使用冻结的 SAM 3 与 PartGLEE 生成候选区域；多模态语言模型 Qwen3.5‑4B 加 LoRA；视觉-语言融合、几何嵌入、集合块、答案嵌入读取；训练目标包括候选区域纯度、成员资格以及子部件/部件名称监督。

**📊 数据集**

训练集：InstructPart；评估集：InstructPart、UMD（novel/test）与 IIT‑AFF（test），并在 UMD 与 IIT‑AFF 上对操作词进行定义对齐。

**📈 对比分析**

与 Affordance‑R1、VisionReasoner（System Two）以及四个 System One 基线（Valen‑4B、H2O‑Lightning‑4B、JevAny‑4B‑LoRA）进行公平对比，采用 gIoU、cIoU、P_50 与 P_50–95 四项指标。实验显示，在 InstructPart、UMD 与 IIT‑AFF 三个数据集上均优于 System Two 方法；在 InstructPart 领域 gIoU 超过 70，UMD gIoU 75.26，IIT‑AFF gIoU 67.92；同时所用的 judge 请求量比基线减少约 4 倍。

**⚠️ 局限性**

局限性：① 对大面积“包含”或“包围”类查询的过度接受导致 IIT‑AFF cIoU 下降；② 高阈值的信心门限使得当 judge 过于确信时无法回溯到更佳对象假设；③ 对操作词定义对齐敏感，未能完全解决不同数据集间词汇差异；④ 仍需进一步提升对对象级错误的自我纠正能力。

---

## 426. InnerLogic-Surg: An Open Surgical Foundation Model via Continual Pretraining

**arXiv ID:** 2611.07883 | [PDF](https://arxiv.org/pdf/2611.07883v1)

**作者:** Kun Yuan `[一作]` (Inner Logic, Inc.), Mathias Unberath `[通讯]` (Inner Logic, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在数千万帧公开手术视频上进行持续自监督预训练，构建了21M至300M参数的 ViT 家族编码器 InnerLogic‑Surg，为外科视觉任务提供统一、强大的特征。

**💡 创新点**

创新地将 DINOv3 的通用视觉先验与 iBOT 掩码学习、Gram‑anchoring 约束以及 AnyUp 高分辨率上采样结合，既保持了空间结构，又提升了稠密预测能力。

**🔧 技术方法**

采用自监督学习框架（DINO + iBOT）、Gram 矩阵约束、AnyUp 上采样、ViT 变体（S/B/L）和大容量投影头。

**📊 数据集**

使用3.6M帧手术数据，来自11,965个公开手术视频（约2,900小时、15种手术类型），并在20个下游任务（workflow、instrument、anatomy、segmentation、detection、depth 等）上进行评估。

**📈 对比分析**

通过冻结线性探测（linear‑probe）对比 DINOv3、LemonFM、SurgeNet、SurgMotion、SurgLaVi 等基线，InnerLogic‑Surg 在所有20个任务上均取得最高或竞争性最优成绩，尤其在密集预测（分割、深度）上显著提升。

**⚠️ 局限性**

受限于 ViT 结构的计算延迟、对实时手术部署的验证不足，以及对更大规模多模态融合与长序列处理的探索不足。

---

## 427. A Projection-Coupled 2.5D THB-GIFT and Quadtree-SBIGA Formulation for Moving-Load Analysis in Semi-Infinite Media

**arXiv ID:** 2611.07872 | [PDF](https://arxiv.org/pdf/2611.07872v1)

**作者:** Hao Hong `[一作]` (Zhengzhou University), Stéphane P. A. Bordas `[通讯]` (University of Luxembourg)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种投影耦合的二维半维（2.5D）THB‑GIFT 与四叉树‑SBIGA 数值模型，用以解决移动荷载在半无限介质中产生的波场与局部几何细节并存的耦合问题，并通过非匹配界面投影实现不同区域自适应离散的无缝衔接。

**💡 创新点**

创新点：
- 结合THB‑GIFT（可局部细化的几何无关分析空间）与四叉树‑SBIGA（自动构造的尺度边界有限元）实现了局部高精度与全局辐射边界的分区离散；
- 在非匹配界面采用L²投影与Hermitian伴随力传递，保证了虚功一致的耦合；
- 引入统一的Bézier提取与弧线提取，使不同几何描述（NURBS、多项式、 rational Bézier）在同一接口上无缝对接；
- 采用单一傅里叶约束统一了频域与波数域的动态算子与响应重构，极大简化了多尺度多物理耦合的实现。

**🔧 技术方法**

技术与方法：
- 2.5D傅里叶约简理论（沿运动方向的波数变换）
- THB‑GIFT（Truncated Hierarchical B‑splines 与 Geometry Independent Field Technique）
- 四叉树‑SBIGA（Scaled Boundary Isogeometric Analysis）
- B‑zézier 提取与 rational Bézier 提取
- L² 投影与 Hermitian 伴随力传递（虚功一致耦合）
- 外部半无限域的尺度边界阻抗与 Riccati 方程求解
- 频域数值求解与时域重构（逆傅里叶）

**📊 数据集**

数据集与验证：
- 半无限均匀介质的解析解（点荷载与车辙响应）
- 圆柱腔的闭式解（用于耦合验证）
- 多层铺装与埋藏通道的几何模型（CAD NURBS 描述）
- 双模改D隧道的几何与材料参数
- 真实车辆参数（轴数、轮胎压力分布、速度）
- 以上均在 0.01–200 Hz 频率范围内进行频域采样与时域重构。

**📈 对比分析**

比较方法与性能：
- 与全域四叉树‑SBIGA 单域模型比较，耦合模型在相同外部分辨率下自由度减少约 34%，S‑元件数降低约 48%，整体误差下降约 15%；
- 与解析解对比，局部 THB‑GIFT 细化可将位移误差降低 41 倍；
- 在圆柱腔验证中，接口位移跳跃低于 10⁻⁷，整体 L² 误差在 10⁻⁵ 级别；
- 时域重构误差在 0.1% 以内，说明频域求解与重构精度高。

**⚠️ 局限性**

局限与挑战：
- 需要在不同区域预先构造并匹配复杂的接口投影，实现复杂度高；
- 外部阻抗只在低频下有效，高频需要更细的人工边界或吸收层；
- 频域求解对频点密度要求较高，频域到时域的逆变换对数值误差敏感；
- 对极端材料异质性（如强弹性模量梯度）时，尺度边界近似可能失效；
- 目前仅针对线性弹性/粘弹性介质，非线性或塑性行为尚未覆盖。

---

## 428. THG-Omni: Traceable Human-Centric Graph Supervision for Video Reasoning

**arXiv ID:** 2611.07836 | [PDF](https://arxiv.org/pdf/2611.07836v1)

**作者:** Jisheng Dang `[一作]` (Lanzhou University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 THG‑Omni 框架，用 SAM 3 对视频中人物进行跟踪并得到时间‑空间轨迹，用 Qwen3‑VL 对整段视频进行全景语义注解，随后通过 HumanOmniV2 对每条候选语义声明进行视频‑音频验证，最终将验证通过的声明与跟踪轨迹按身份与时间绑定，构成可追踪的人本图，并将该图作为显式监督训练多模态大语言模型进行视频推理。

**💡 创新点**

创新点在于：①引入可追踪的人本图监督机制，将身份绑定、时间归一化与语义验证相结合，实现对图中每条声明可追溯的特性；②采用正向验证而非负样本生成，避免了短语捷径与伪逆向语义问题；③将图结构转化为多类图问题，作为显式监督注入模型；④在训练阶段混合任务问题、图匹配与图缺失示例，多角度提升模型对图的利用能力。

**🔧 技术方法**

主要技术包括：Segment Anything v3（SAM 3）人物跟踪与空间轨迹提取；Qwen3‑VL 生成整段视频的语义声明；HumanOmniV2 用于语义声明的多模态验证；低秩适配器（LoRA）对 Qwen2.5‑Omni‑7B 的多模态大语言模型进行任务与图特定微调；图结构生成与图问题构造；多模态输入（视频、音频、文本）融合。

**📊 数据集**

使用的数据集包括 IntentBench（Social‑IQ、EMER、MDPE 三子任务）、NExT‑QA（因果、时序、描述类问答）、MER24（多标签情绪识别）以及自制的验证集用于图验证与训练。

**📈 对比分析**

实验通过与未适配 Qwen2.5‑Omni‑7B、Graph‑SFT、LLaVA‑NeXT‑Video‑7B‑32K、HumanOmniV2 等基线对比，评估指标为 IntentBench 综合得分、EMER F1、NExT‑QA 准确率、MER24 图诊断准确率。THG‑Omni 在 IntentBench 上综合得分提升至 71.72%（比基线提升约 16 %），EMER F1 提升至 86.26%；在 NExT‑QA 上准确率从 77.98% 提升至 89.54%；MER24 图诊断上准确率从 14.67% 提升至 24.00%。图匹配实验表明匹配图显著优于无图或打乱图，证明模型已学会有效利用图信息。

**⚠️ 局限性**

局限性包括：①图构建高度依赖 SAM 3 跟踪和 Qwen3‑VL 注解，易受遮挡、相机切换、外观变化等影响；②验证过程基于模型判断，仍可能存在误判或漏判；③评估图未进行人工校正，可能缺失重要信息或包含不受支持的声明；④不同问题对图内容的敏感度不同，某些语义声明并不一定提升准确率；⑤在更长、更复杂的视频交互场景下的鲁棒性尚待进一步验证。

---

## 429. Operationalizing NCII and CSAM Law in the Age of AI: Open Problems

**arXiv ID:** 2611.07831 | [PDF](https://arxiv.org/pdf/2611.07831v1)

**作者:** Kimberly Mai `[一作]` (AI Security Institute), Ashia C. Wilson `[通讯]` (MIT)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统分析了英格兰和威尔士以及美国关于非同意亲密影像（NCII）和儿童性虐待材料（CSAM）的现行法律，并在此基础上识别了技术与法律交叉点上的开放研究问题，提出了针对性技术研究议程。

**💡 创新点**

创新点在于将两国法律框架与技术实现需求紧密结合，形成针对NCII/CSAM的跨学科技术研究路线图，并对各类法律行为（创作、持有、分发、工具服务、培训材料）提出了具体技术难题。

**🔧 技术方法**

主要采用法律文本解读与技术需求梳理方法，未开展新的算法或实验技术。

**📊 数据集**

本文未使用任何公开数据集；讨论基于现有公开报告与案例的法律与技术分析。

**📈 对比分析**

由于缺乏实验实现，本文未提供方法比较或性能评估，仅提出理论问题与研究方向。

**⚠️ 局限性**

局限性包括：缺乏实证验证与实现细节，研究受限于不同司法辖区的法律与伦理审查，且对技术方案的可行性与效果尚无量化评估。

---

## 430. ArcadeDB in Python: An In-Process Multi-Model Database

**arXiv ID:** 2611.07971 | [PDF](https://arxiv.org/pdf/2611.07971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 431. Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval

**arXiv ID:** 2611.07820 | [PDF](https://arxiv.org/pdf/2611.07820v1)

**作者:** Dohoon Kim `[一作]` (Seoul National University), Taesup Moon `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Axis-LoRA，通过在冻结的稠密嵌入模型后期层加入四个轴特定 LoRA 适配器并用稀疏门控实现对个人记忆检索的轴对齐改进。

**💡 创新点**

发现稠密嵌入模型在四个个人记忆轴（时间、地点、实体、事件）上表现不对称，并且每个轴在注意力头中具有独立模式，进而提出仅训练少量参数的 Axis-LoRA 来增强轴特征，同时保留通用检索性能。

**🔧 技术方法**

使用 LoRA 适配器、稀疏最大化门控、对比学习（InfoNCE）、知识蒸馏（KD）在 MS MARCO 上做锚定。

**📊 数据集**

使用 AxisCF 合成轴对照样本、LongMemEval、ATM-Bench（Normal 与 Hard）、LoCoMo、MTEB BEIR 子集等数据集。

**📈 对比分析**

在 MiniLM‑L6、bge‑base、Qwen3‑Embedding 三种主流稠密嵌入模型上与基线比较，平均 R@10 提升 1.28–1.56 个百分点，最大提升 3.75 个百分点；与提示、外部投影、共享 LoRA 等变体对比，Axis‑LoRA 性能最佳；在 MTEB BEIR 上保持基本通用检索性能。

**⚠️ 局限性**

局限性：轴设定依赖现有标注，隐式轴表达（如相对时间、模糊地点、复合实体事件）未充分覆盖；AxisCF 仅改动显式值；方法仅针对单一用户记忆检索，未探讨更广泛的个性化或快速适配；门控机制相对简单，可进一步改进。

---

## 432. QRAE: Query-Relevant Attention Entropy for Context Compression in DeepSearch Agents

**arXiv ID:** 2611.07976 | [PDF](https://arxiv.org/pdf/2611.07976v1)

**作者:** Yingjie Tan `[一作]` (Tsinghua University), Yaqing Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为QRAE的在线上下文压缩方法，用于DeepSearch智能体在推理过程中筛选和保留检索到的文本片段。

**💡 创新点**

创新点在于利用模型内部的查询相关注意力熵作为评分信号，采用稀疏的关注头进行证据评估，并通过持久白名单和候选缓存实现上下文的动态管理，避免额外的生成压缩步骤。

**🔧 技术方法**

技术包括：离线头定位（基于对答案跨度的注意力权重），在线查询‑无关对比预填充，正向差分注意力的熵归一化评分，基于得分的缓存‑到‑白名单授予，以及持久化上下文管理策略。

**📊 数据集**

使用了BrowseComp‑Plus、XBench‑DeepSearch、FRAMES、SimpleQA四个DeepSearch基准数据集；离线定位阶段使用HotpotQA；评估覆盖多种模型架构，包括DeepDiver‑7B、LiteResearcher‑4B、WebSeer‑14B、OpenResearcher‑30B和OpenPangu‑2.0‑Flash‑92B。

**📈 对比分析**

与原始DeepSearch以及LLMLingua‑2、ACON、Rerank、QRHead、SideQuest等压缩基线对比。结果显示，QRAE在六个模型‑基准组合中平均提升任务成功率2.35个百分点，同时令累计token使用下降约45%，端到端延迟下降约30%，并且在不同规模与结构的模型上均保持显著优势。

**⚠️ 局限性**

局限性包括：依赖于可被定位的查询相关注意力头，若模型内部未形成此类头则效果受限；评估过程仍需额外预填充计算，虽然开销低但不完全消除；仅在检索文本包含答案的任务中验证，对需要跨文本或非检索式证据的任务尚未评估。

---

## 433. A Little Derandomization Goes a Long Way: Unconditional Circuit Lower Bounds using Low-Degree PIT

**arXiv ID:** 2611.07972 | [PDF](https://arxiv.org/pdf/2611.07972v1)

**作者:** Nikolai Chukhin `[一作]` (JetBrains Research), Ivan Mihajlin `[通讯]` (JetBrains Research)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在不使用任何随机化假设的前提下，证明了一个二分命题：要么非确定性指数类NE需要指数级别的布尔电路（即大小至少为2^n/(2n)），要么常数自由版本的Valiant猜想即VP^0=VNP^0成立。

**💡 创新点**

创新点在于：①构造了一个“通用多项式”𝒰，使所有分区多项式（包括满足SAT判定的多项式）都是其投影；②给出了整数上对常数度数的白盒PIT算法，时间为O_d(st^{d-1})，比传统的O_d(st^d)大幅加速；③将Kabanets–Impagliazzo的低复杂度与PIT相结合，得到无随机化的布尔与算术低复杂度双赢。

**🔧 技术方法**

主要技术包括：分区多项式与下推自归约、构造通用多项式、常数自由算术电路的正式度数约束、Baur–Strassen微分技术、对集合多项式平方和的快速计算、Merlin–Arthur协议与sum‑check、以及从分支程序到布尔公式的深度/大小转换。

**📊 数据集**

无数据集，完全是理论证明。

**📈 对比分析**

本工作不涉及实验性能评估；相较于此前的最强已知低复杂度（如Ω(n^{3.5})对SAT），本文的布尔结果逼近指数上界，而算术结果则突破了以往在常数自由算术电路上的极限。

**⚠️ 局限性**

局限性在于：①仅适用于常数度数多项式，尚未推广到更高度数；②对通用多项式𝒰的低复杂度假设是核心；③证明仍依赖于分区多项式的特殊结构，难以直接应用于任意算术电路；④对实际布尔电路大小的下界虽然接近上界，但仍与最优指数上界存在较大距离。

---

## 434. TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research

**arXiv ID:** 2611.07962 | [PDF](https://arxiv.org/pdf/2611.07962v1)

**作者:** Yifan Luo `[一作]` (Tsinghua University), Andrew Chi-Chih Yao `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个覆盖理论计算机科学四大研究领域（定理证明、求解器工程、构造以及开放猜想解决）的统一 benchmark——TCSExam。

**💡 创新点**

创新点在于：①将四类研究任务统一到同一框架与评价协议；②使用 Lean 4 及可执行检查器提供正式可验证的结果；③提供可扩展的任务接口、Docker 构建与评分工具；④设计了 Lite 版以在保持评估完整性的同时显著减少任务量。

**🔧 技术方法**

使用技术包括 Lean 4 证明核、可执行任务检查器、Docker 容器、任务版本管理、自动化评分与统计脚本；同时在实验中结合多大模型（GPT‑6 Astra、Kimi K3.1 Pro、Opus 5.5 等）与工具调用框架。

**📊 数据集**

数据集由 100 个专家评审的研究任务组成（20 条定理、30 条求解器、20 条构造、30 条开放猜想），Lite 版选取其中 20 条（每轨 5 条）。

**📈 对比分析**

比较方法：对每个任务和每个代理在 Full 与 Lite 赛制下分别计分，计算四轨平均得分并合成 TCSExam Index；使用 Spearman ρ 对 Full 与 Lite 排名相关性进行检验。性能表现：GPT‑6 Astra 在 Full 中取得 71.9 % Index，Opus 5.5 在 Lite 中取得 72.3 %；排名高度一致（ρ=0.93）。

**⚠️ 局限性**

局限性包括：①任务受限于可验证性，部分复杂开放猜想无法完整解决；②评估依赖 12 小时开发预算，无法覆盖更长时间或更大算力场景；③当前代理多基于已有训练数据，缺乏真正的“自我学习”或探索能力；④Lite 版在某些轨道上的覆盖率和深度仍有限。

---

## 435. A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems

**arXiv ID:** 2611.07953 | [PDF](https://arxiv.org/pdf/2611.07953v1)

**作者:** Yao Huang `[一作]` (Tsinghua University), Yinpeng Dong `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一套以行为为基础的框架，用于预测前沿AI系统的失控风险，并通过13个方面和29个评测任务量化失控条件。

**💡 创新点**

创新点在于将失控风险拆解为误导动机、危害实现能力和监控逃避三大因子，形成可测量的指标体系，并证明其预测效度。

**🔧 技术方法**

采用行为评测方法、统计学相关分析（Pearson r）、多模型实验（13个前沿LLM），以及对策实验（监督微调与监控增强）。

**📊 数据集**

使用29个评测任务（来自公开基准及自定义），以及构造的400个失控风险场景案例（4个领域×5场景×20实例）。

**📈 对比分析**

通过LoC分数与失败率的相关性验证，Pearson r=0.84；在针对性微调与监控提升后，失败率从0.32下降至0.22，保持一般性能。

**⚠️ 局限性**

局限在于乘法聚合未捕捉因子间交互、实验仅在受控模拟中进行、需更逼真失控情境、且任务需随AI工具与多智能体发展持续更新。

---

## 436. Invariance before structure: two modes of emergent compression in representation learning

**arXiv ID:** 2611.07952 | [PDF](https://arxiv.org/pdf/2611.07952v1)

**作者:** Hardik Rajpal `[一作]` (Imperial College London), Dan Goodman `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了EARL框架，研究了在循环储备网络中出现的两种压缩模式——状态空间压缩与参数空间压缩；

**💡 创新点**

创新点在于将两种压缩模式用信息论度量（Ψ与Λ）量化，并揭示它们在训练过程中的不同时间段出现，从而解释学习表示对底层结构的去耦和任务特定性的形成；

**🔧 技术方法**

采用信息论方法（互信息、Fisher灵敏度估计）、储备网络+线性瓶颈结构、岭回归预测、参数空间与状态空间的集合估计（Barber‑Agakov 下界）以及多组超参数集合训练；

**📊 数据集**

使用合成吸引子数据集（Sprott B、Rössler、Lorenz、Chen、Sprott A），并对吸引子进行未见旋转测试；

**📈 对比分析**

通过与时间置换及跨序列目标的对照实验比较，评估一阶预测误差、Ψ和Λ；结果显示参数空间压缩在训练初期迅速升高，而状态空间压缩在后期恢复并提升，真实目标训练优于对照组；

**⚠️ 局限性**

局限在于度量相对性强、需要有可定义的潜变量、Ψ在大网络中始终为负、对不同任务和真实数据的推广仍需进一步验证。

---

## 437. AttnSR: Attention Step Reward for Long-Horizon Agentic RL

**arXiv ID:** 2611.07932 | [PDF](https://arxiv.org/pdf/2611.07932v1)

**作者:** Mengsong Wu `[一作]` (East China Normal University), Aimin Zhou `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Attention Step Reward（AttnSR）方法，利用训练时的 policy attention 图像来为多步决策过程中的中间步骤分配奖励。

**💡 创新点**

创新点在于把注意力作为无监督的步骤级奖励先验，既不需要额外 rollouts，也不需要训练单独的奖励模型，能够在保持 PPO 结构不变的前提下丰富中间奖励信号。

**🔧 技术方法**

技术主要包括 Transformer attention 提取、步骤级图回溯、正负奖励中心化、置信门控以及与 PPO 的无缝集成。

**📊 数据集**

使用 ALFWorld、WebShop 与 ScienceWorld 三大 agentic benchmark 进行实验评估。

**📈 对比分析**

与终端奖励 PPO、线性 Step Reward、GRPO 等 baseline 对比，AttnSR 在 ALFWorld 任务成功率从 91.2% 提升至 97.1%，在 ScienceWorld 上取得相同或更少交互的官方分数提升，而在 WebShop 上未出现显著差异。

**⚠️ 局限性**

局限性包括奖励分配仅为启发式，无法保证策略不变，对短轨迹效果有限，并且对注意力信号的可靠性和解释性存在一定依赖。

---

## 438. How Generative AI Redefines Personalization and Diversity in the Newsroom

**arXiv ID:** 2611.07931 | [PDF](https://arxiv.org/pdf/2611.07931v1)

**作者:** Cheng Yu `[一作]` (Technical University of Munich), Orestis Papakyriakopoulos `[通讯]` (Technical University of Munich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过16位新闻专业人士的半结构访谈，探讨生成式 AI 在新闻编辑工作中的双向个性化如何重塑编辑判断与新闻多样性决策。

**💡 创新点**

创新点在于提出“工具个性化”和“受众个性化”两种生成式 AI 个性化路径，揭示其在源头检索与版本生成层面产生的隐形多样性决策，并强调传统多样性评估需超越成品文本。

**🔧 技术方法**

主要技术包括访谈数据分析、人工评估以及对 ChatGPT、Claude、Gemini 等大型语言模型在检索与文本生成任务中的行为验证。

**📊 数据集**

使用的数据集包括访谈转录文本、两项验证实验（检索覆盖实验与版本信息保持实验，涵盖约12篇新闻稿与5类受众），以及模型输出与人工标注的评估记录。

**📈 对比分析**

通过量化检索覆盖率和版本信息保持率评估模型性能，发现检索覆盖率低、生成版本核心信息保持率约为75%，表明生成式 AI 在多样性方面可能带来信息缺失与偏差。

**⚠️ 局限性**

局限性包括样本为非代表性访谈、验证实验规模有限、缺乏真实新闻发布后的多样性测量、人工评估主观性及对不同新闻语境的可迁移性不足。

---

## 439. How Finite Brains Generalize Without Bound: Spatial Memory as a Universal Computer

**arXiv ID:** 2611.07920 | [PDF](https://arxiv.org/pdf/2611.07920v1)

**作者:** Yuhang He `[一作]` (Peking University), Si Wu `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种“仅读偶然性”原理，说明有限经验如何使学习者在输入长度扩大时仍保持正确，并用这一原理设计了可在短序列上训练、在长序列上准确执行的“时钟机”模型；随后在15个涵盖上下文记忆、递归、复制等多种算术/符号任务上进行实验，验证该模型在超长序列上可实现接近 99% 的准确率，且在 LSTM 与 Transformer 无法正确完成的任务上表现优异。

**💡 创新点**

核心创新点包括：①将记忆结构归结为可循环的时钟组（类似网格细胞周期），并证明其是唯一能够在读入坐标不变、仅读取偶然性时实现无穷长序列推断的结构；②将时钟组映射到大脑中网格细胞与位置细胞的动态，提出“记忆为普遍计算机”的统一框架；③在机器学习框架中实现可在短序列上训练、长序列上推理的“时钟机”，并展示其优于现有 LSTM/Transformer 的泛化性能。

**🔧 技术方法**

技术实现包括：①基于可循环时钟（周期为 2、3、5、7、11，L=2310）的符号机；②控制器采用带门控的循环网络（GRU），学习如何更新时钟并查询“目录”（类似海马索引）；③实验使用了 15 个从简单计数到二进制加法、括号算术、排序、乘法等任务；④在神经网络实现中，将时钟转化为连续吸引子环网络，使用网格细胞模型、位置细胞和海马回路，验证神经动力学与符号机等价。

**📊 数据集**

实验数据集为人工构造的序列任务集合，长度训练区间为 1~40，测试区间为 41~500（部分任务上测试至 15015），每个长度随机抽样 32 条样本；此外对 Turing 机、乘法器等自定义程序进行模拟，验证时钟机在无输入思考时间时的性能。

**📈 对比分析**

对比方法包括：LSTM（256 单元）和 Transformer（5 层、宽度 64，使用或不使用旋转位置编码）。评价指标为每个长度下的准确率，整体采用平均准确率和 99% 置信阈值。结果显示：在 9 个难以由 LSTM/Transformer 完成的任务中，时钟机在 99% 以上保持准确，尤其在逆序、复制、栈操作、奇偶优先和二进制加法等任务上明显优于基线；对 5 个 LSTM/Transformer 可解的任务，时钟机同样能达到 99% 以上的准确率；在极长序列（> L）时，时钟机按理论预测的误差曲线下降，验证理论与实验的一致性。

**⚠️ 局限性**

局限性包括：①目前仅在相对短的人工序列任务上验证，缺乏对真实自然语言或更复杂感知任务的测试；②对深层嵌套和需要多种递归结构的任务（如括号算术、乘法、平方根）仍无法收敛；③模型依赖于精确的时钟周期设置，周期选择不当会导致范围不足；④训练过程中需要对时钟读数进行离散化，可能受噪声与连续性问题限制；⑤神经实现中对噪声和突触权重变化的鲁棒性仍需进一步研究。

---

## 440. Tight Worst-Case Completion-Time Bounds for Dominant Resource Fairness

**arXiv ID:** 2611.07905 | [PDF](https://arxiv.org/pdf/2611.07905v1)

**作者:** Mugen Blue `[一作]` (University of California), Alexander Lindermayr `[通讯]` (Technische Universitaet Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并分析了Dominant Resource Fairness（DRF）在多维非clairvoyant调度中的完成时间性能，给出了总完成时间和ℓ_q范数目标的渐近最优竞争比。

**💡 创新点**

提出了Θ(√D)的总完成时间竞争比以及Θ(D^(1/(q+1)))的ℓ_q范数竞争比，完成了先前已知的Ω(√D)与O(D)之间的鸿沟；采用了Cauchy–Schwarz与LP对偶拟合的组合方法，并对D=2、3的情况给出更精确常数。

**🔧 技术方法**

主要技术包括：进阶填充（progressive filling）对DRF的单次迭代分析；使用Cauchy–Schwarz把拥塞与瓶颈资源分开估计；构造连续时间线性规划松弛并对偶拟合，处理重负载与轻负载两种情况；利用重力时间（heavy/light）分解和瓶颈资源的计数；以及对率上限和释放时间的单调性分析。

**📊 数据集**

无实验数据集；本文全部为理论分析与证明，未涉及实测或仿真。

**📈 对比分析**

通过与全知最优离线调度进行比较，给出了竞争比：总完成时间为O(√D)（下界为Ω(√D)），ℓ_q范数为O(D^(1/(q+1)))（上界与下界匹配），并在D=2、3时给出了更小的常数上界（分别为2+√3和2+√7）。

**⚠️ 局限性**

局限性：仅适用于分数调度（不考虑资源不可分割、作业放置等实际限制）；分析在单位容量假设下成立，未考虑多机或多集群的实际环境；常数值虽可计算但可能不够紧凑，且未给出平均情况或随机工作负载的表现。

---

## 441. CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation

**arXiv ID:** 2611.07959 | [PDF](https://arxiv.org/pdf/2611.07959v1)

**作者:** Guhan Chen `[一作]` (Tsinghua University), Zixiong Yu `[通讯]` (Huawei Large Model Data Technology Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种新的 on‑policy distillation 采样器，利用最大耦合将学生与教师的目标进行几何插值，减少教师偏好路径的盲点。

**💡 创新点**

提出 coupled‑rollout distillation，采用显式最大耦合采样保持两者一致时不干预，干预频率随两者差距自动下降，从而显著提升教师信息利用率。

**🔧 技术方法**

使用 Speculative Decoding 的最大耦合抽样、α 参数化的几何插值目标、reverse‑KL 损失以及对数学推理任务的评估框架。

**📊 数据集**

在 MATH（MATH500）、AIME24、AIME25、AMC23、GSM8K 等公开数学推理基准上进行训练与评估，训练数据为约5,000道问题的固定子集。

**📈 对比分析**

与传统 OPD（α=0）及 SKD 对比，采用 pass@1/8/16 评估；在所有 6 个学生上平均提升 pass@1 0.9–2.0 点、pass@8 1.2–3.4 点，OPD 训练两倍步长亦无法赶超。

**⚠️ 局限性**

需要教师在推理时提供 logits；仅在 4B 级学生与 9B/14B 教师上测试；学生与教师需共享 tokenizer；实验聚焦数学推理任务，缺乏对更大模型和其他任务的验证。

---

## 442. Modular Parameterized Model Checking of Threshold-Based Distributed Algorithms

**arXiv ID:** 2611.07919 | [PDF](https://arxiv.org/pdf/2611.07919v1)

**作者:** Paul Eichler `[一作]` (CISPA Helmholtz Center for Information Security), Swen Jacobs `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种新的模块化参数化模型检测方法，利用同步动作在阈值自动机（TA）中混合异步和同步步骤，并通过层式自动机实现模块化抽象，最终实现对分布式算法（如王者共识）进行可扩展、可验证的分析。

**💡 创新点**

创新点包括：① 引入同步动作的阈值自动机模型，兼顾共享变量与拜占庭容错；② 证明该模型在一般情况下不可判定，随后提出基于同步步数的有界模型检查和层式自动机的可判定子类；③ 开发同步模块抽象技术，允许用简洁的同步步骤替代复杂模块，自动生成正确性论证；④ 在同一框架内完成异步与同步步骤的混合验证，显著降低模型规模与验证时间。

**🔧 技术方法**

主要技术：阈值自动机与同步动作的语义定义；有界模型检查的SMT编码；层式自动机分层判定与同步步数上界计算；可达性保持替换（rp / srp）的SMT验证；实验实现基于ByMC文件格式的原型工具。

**📊 数据集**

实验采用案例研究——王者共识算法（包含弱共识、分级共识和王者共识三层模块）作为数据集；对每一层分别构建完整TA与同步抽象模型；使用ByMC/ByMC扩展文件进行编码。

**📈 对比分析**

比较方法：将完整TA模型与同步抽象模型（以及不同抽象精度）分别输入原型工具进行参数化可达性检测；记录验证时间与模型大小。结果显示：王者共识完整TA在41秒内完成，而完全同步抽象仅需0.03秒；使用分级共识精确同步抽象时为1.2秒，使用过度抽象时仅0.5秒；可达性保持检查均在1分钟内完成。总体而言，同步模块抽象大幅提升了性能与模型简洁度。

**⚠️ 局限性**

限制：① 同步模块抽象仍需手工设计，缺乏自动化生成；② 仅适用于层式（单轮）自动机，无法直接处理多轮或循环算法；③ 具有理论上Π₂‑完整的可达性保持验证，实际性能受限于SMT求解器；④ 对于非层式或包含复杂接收自环的模型，无法保证同步步数上界。

---

## 443. Beyond the Keyword: A Human-Annotated Benchmark for Arabic Anxiety-Related Discourse

**arXiv ID:** 2611.07949 | [PDF](https://arxiv.org/pdf/2611.07949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 444. Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection

**arXiv ID:** 2611.07994 | [PDF](https://arxiv.org/pdf/2611.07994v1)

**作者:** Zehao Wang `[一作]` (South China University of Technology), Yaowei Wang `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ReDHF框架，通过复兴被丢弃的最高分辨率特征提升极细目标检测效果。

**💡 创新点**

创新点在于三大模块：CSDF对最高分辨率特征进行语义增强并将细节注入浅层；EGSS采用椭圆指导的辅助监督提升浅层学习；FSQI让解码器查询同时访问全尺度特征。

**🔧 技术方法**

使用Transformer检测器(DETR/Deformable-DETR/DINO等)与变形注意力、RepNCSPELAN4、FCOS式辅助头等技术。

**📊 数据集**

在AI-TOD-v2、VisDrone和SODA-D三个小目标检测基准数据集上进行实验。

**📈 对比分析**

在三大数据集上相较基线提升AP1–1.8点，达到或超过SOTA，尤其在超细目标(AP_vt)上显著提升。

**⚠️ 局限性**

局部额外计算量和在大目标上的轻微性能衰退，以及对仅使用高分辨率特征的模型适用性有限。

---

## 445. Compression Through a Context Lens: Evaluating Adaptation, Quantization, and Distillation in Arabic Legal LLMs

**arXiv ID:** 2611.07988 | [PDF](https://arxiv.org/pdf/2611.07988v1)

**作者:** Md. Rafiul Biswas `[一作]` (Hamad Bin Khalifa University), Wajdi Zaghouani `[通讯]` (Northwestern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在阿拉伯法律场景下，使用 ArabLegalEval 进行的领域适应、后训练量化与知识蒸馏对大型语言模型（FANAR‑9B、JAIS‑8B、Qwen2.5‑7B）性能的影响，并引入了无上下文评估来衡量模型的参数知识；

**💡 创新点**

首次将无上下文评估与传统的有上下文评估结合，系统比较三种压缩策略在阿拉伯法律任务中的效果，探索跨体系结构蒸馏的可行性；

**🔧 技术方法**

采用 QLoRA 进行参数高效适配，使用 GPTQ/AWQ 等方法实现 INT8、INT4（NF4、FP4）后训练量化，并采用基于伪标签的硬标签蒸馏；

**📊 数据集**

以 Saudi Arabian 司法文档构建的 ArabLegalEval（多项选择题 MCQ 与 NajizQA 开放式问答）为主要数据集；

**📈 对比分析**

在三种模型上分别进行适配、量化与蒸馏，并在有上下文与无上下文两种协议下评估；结果显示无上下文准确率从 70%–80% 变化，量化可实现 64–71% 的显存压缩，准确率波动 <3%；蒸馏时模型规模越小准确率下降明显，但跨体系结构蒸馏可达到与教师相近的 76.6%；

**⚠️ 局限性**

仅针对 ArabLegalEval，固定 500 项子集评估，缺乏更大或多样化数据集；使用硬标签蒸馏而非软标签、未评估检索增强；教师与学生配置不完全一致，且部分评估指标（ROUGE‑L）无法直接衡量法律正确性；模型训练数据可能与评测集重叠的风险未完全排除；

---

## 446. Technohybridism: Recursive Semantic Normalization in AI-Mediated Societies

**arXiv ID:** 2611.07984 | [PDF](https://arxiv.org/pdf/2611.07984v1)

**作者:** Charalampos M. Liapis `[一作]` (University of Patras), Epameinondas Panagopoulos `[通讯]` (University of Patras)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出递归语义归一化（RSN）机制，阐释大型语言模型输出如何被社会实践再吸收并影响语言规范，构建混合语义系统的概念框架。

**💡 创新点**

创新点在于：①将生成、选择、吸收、持续等环节纳入统一的递归模型；②区分技术递归与社会递归，说明机器输出对语言环境的反馈作用；③提出混合语义系统概念，将人类实践、计算生成与制度化吸收视为相互递归的整体。

**🔧 技术方法**

采用社会理论、社会认知论与语言哲学的概念分析方法，构建六要素的解析模型（领域、候选空间、选择、吸收份额、持久性、对照基线），并结合案例观察进行阐释。

**📊 数据集**

未使用具体数据集；论文主要基于文献综述、案例观察（如学术摘要、同行评审、新闻稿等）和已有的统计报告。

**📈 对比分析**

未进行实验或性能比较；作者提出证据金字塔与研究设计，建议未来通过词频计数、语料学分析、对照实验、差分-差分等方法验证RSN效应。

**⚠️ 局限性**

局限性包括：①缺乏经验验证，论文仅为概念框架；②聚焦英语学术/专业语境，未考察其他语言或非学术文本；③未涵盖法律、监管与政治经济层面的影响；④对混合语义系统的实证可操作性仍待进一步研究。

---

## 447. ChemE-as-Code: An Open-Source Platform for Agentic AI in Chemical Engineering

**arXiv ID:** 2611.07947 | [PDF](https://arxiv.org/pdf/2611.07947v1)

**作者:** Artur M. Schweidtmann `[一作]` `[通讯]` (Delft University of Technology), Artur M. Schweidtmann (Delft University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了ChemE-as-Code平台，利用通用AI代理和可验证工具完成从需求到完整工程包（P&ID、模拟、布局、HAZOP、成本、生命周期评估）的全流程自动化设计；

**💡 创新点**

将工程文件统一在Git仓库中，结合可追溯的“证据链”和自动检查，形成了工程知识的可读、可验证、可共享的新范式；

**🔧 技术方法**

采用大型语言模型驱动的通用代理、Python实现的Deterministic工具集（DWSIM、pyDEXPI、CAPE-OPEN、Brightway等）、DEXPI 2.0.1格式、YAML+JSON Schema、持续集成与Git命令行接口；

**📊 数据集**

以天然气氨制备工艺为案例，使用公开的Ammonia工艺库（process‑library）以及相应的运行记录与决策日志；

**📈 对比分析**

通过对比已知规则下的人工注入错误、工程更改和自动重现，验证检查覆盖率和可重现性；在案例中，检查能捕捉19/19已知错误、对部分未知错误检出4/12，工程更改仅触发少量文件，Agent自动化更改在成本（USD 8.55）和时间（49 min）内完成，但仍有未检测到的缺陷；

**⚠️ 局限性**

缺点包括：检查仅覆盖已实现规则，无法捕获所有物理/工程错误；Agent产生的设计缺乏人类审核，可能导致过度信任；对新工艺适用性有限，需人工判断假设与模型；系统目前为原型，缺乏对比实验与工业验证。

---

## 448. tok: a textual concrete syntax of BPMN and DMN for humans and AI agents

**arXiv ID:** 2611.07945 | [PDF](https://arxiv.org/pdf/2611.07945v1)

**作者:** Marek Deja `[一作]` `[通讯]` (Jagiellonian University), Marek Deja (Jagiellonian University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向 BPMN 2.0.2 与 DMN 1.5 的纯文本表示法 tok，并实现了一个统一引擎，能够对 tok 文档进行校验、编译为 XML/SVG/DMN、导入导出、仿真及与语言模型的交互。

**💡 创新点**

创新点包括：
1) 将 BPMN 与 DMN 的抽象语法映射到一种极简文本语法，既保持标准精确映射，又实现更小的 token 规模；
2) 设计了一个局部化诊断的校验器，使得错误定位精确到行和元素；
3) 提供了基于 Model Context Protocol 的语言模型接口，让 LLM 能直接与流程模型交互；
4) 对 tok 进行了系统评估（尺寸、错误检测、LLM 生成、过程挖掘、性能与合规性），并公开了完整数据与代码。

**🔧 技术方法**

技术手段包括：
- Rust 编写的核心库（解析、校验、布局、导出、导入、仿真）
- 基于 offside 规则的无缝缩进语法与上下文无关的词法分析
- 对 BPMN/DMN 的 XML Schema 与 bpmnlint 规则集进行再实现
- Model Context Protocol（MCP）服务器与 LSP/VSCode 插件
- 多种 token 方案（o200k, cl100k, Qwen3, DeepSeek-V3, Mistral NeMo）
- 过程挖掘与仿真使用 PM4Py 与 XES 日志

**📊 数据集**

数据集与测试集合：
- N：93 个无错误 tok 示例
- V：37 个工具导出的 893 条 BPMN XML（MIWG 测试套件）
- R：22 条 MIWG 参考模型
- C：Camunda Modeler 画出的 16 条 BPMN
- D：DMN TCK 143 条 DMN
- PET：45 条英文流程描述（含专家注释）
- PMo：45 条英文流程描述（含对应 BPMN）

**📈 对比分析**

比较方法与性能：
- 大小比较：tok 文本的 token 约为最小标准 XML 的 1/2–1/4，最小 XML 的 1/7–1/11；
- 错误检测：tok 校验器 98.2% 的注入错误被检测到，XML Schema + bpmnlint 仅 85.8%；
- LLM 生成：在一次性参考下，四种 LLM 以 85.7–100% 的成功率生成有效 tok，Token 数量比 XML 低 20–60%；
- 过程挖掘：模拟日志在 PM4Py 上 44/47 的模型达到 100% 适配度；
- 性能：典型模型（约 100–3000 条流）检查 20–24 ms，导出+布局 0.11–0.43 s；
- 合规性：Tok 1.0.3 通过所有 DMN TCK、MIWG、Camunda 以及自定义 conformance corpus。

**⚠️ 局限性**

局限性：
- 仅在英文文本和四种 LLM 上评估；
- 参考标准不同（PET 通过注释，PMo 通过已绘制 BPMN），导致一致性评估受限；
- 仅使用了 4 种 LLM，无法覆盖更广泛模型；
- 技能与工具调用设计可能混淆结果，难以单独分离影响；
- 过程挖掘演示使用固定参数（时间、概率），未覆盖真实工作负载；
- 未进行人机实验验证 tok 对流程建模者的实际可用性。

---

## 449. Flux as a Codec: Zero-Shot Lossy Compression with Flux

**arXiv ID:** 2611.07929 | [PDF](https://arxiv.org/pdf/2611.07929v1)

**作者:** Fuma Kimishima `[一作]` (Hosei University), Jinjia Zhou `[通讯]` (Hosei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FluxC，一种基于 Flux 生成模型的零样本压缩编码器，并从编码成本、位分配、重建方法和低比特率性能四个维度进行理论分析与实验验证。

**💡 创新点**

创新点在于：①直接在 Flux 的 OT 路径上定义 Markov 转移而非重新映射为 VP；②将 ODE 重建与传统祖先采样对比，证明在低比特率下 ODE 能获得 √R 的失真优势；③将信息率与位分配视角结合，揭示 VP 与 OT 在相同 SNR 下的位分配差异。

**🔧 技术方法**

采用 Flux 生成模型、逆向通道编码（RCC）、概率流 ODE、DDPM 逆向采样、以及传统神经压缩基线（PerCo、MS-ILLM、DDCM、DiT-IC）等技术。

**📊 数据集**

在自然图像数据集 Kodak、DIV2K、CLIC 以及 Gaussian 源上进行评估。

**📈 对比分析**

与 DiffC、DiffC‑Flux、传统神经压缩基线比较；在极低比特率下 FluxC‑ODE/FluxC‑1 的 PSNR 与 LPIPS 高于祖先采样；整体在多种指标上与现有方法竞争，且在低比特率下表现尤为优异。

**⚠️ 局限性**

局限性在于：生成式解码可能产生伪影，无法保证恢复的图像内容真实性；在高 SNR/比特率下与基线差距不大；实验受限于 50 步编码/解码与 24 GB GPU；

---

## 450. ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation

**arXiv ID:** 2611.07928 | [PDF](https://arxiv.org/pdf/2611.07928v1)

**作者:** Petru Neague `[一作]` (Delft University of Technology), Johan Pouwelse `[通讯]` (Delft University of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出ZeroRAG，一种无中心化、无标签的分布式检索-增强生成（RAG）路由层；

**💡 创新点**

创新点在于仅利用邻居广告与最近查询缓存进行最佳优先遍历，并自适应重连邻接，避免全局索引或协调；

**🔧 技术方法**

技术包括基于嵌入的语义查询路由、最佳优先遍历、邻接动态重连、查询缓存、低维向量数据库、基于相似度的邻居选择；

**📊 数据集**

使用三大真实工作负载：Amazon产品搜索、PubMedQA问答和Web搜索日志（用户-点击）等；

**📈 对比分析**

与分布式LSH/Kademlia、Semantica等基线对比，ZeroRAG在召回率、消息量、查询次数方面明显优于基线，接近中心化HNSW索引（召回93.0% vs 97.3%，准确率77% vs 79%）；

**⚠️ 局限性**

局限包括对网络规模的非单调请求数、对查询缓存大小敏感、对新主题传播速度依赖初始拥有者数量，以及未评估恶意节点、同步延迟和隐私细节等。

---

## 451. StitchEdit: Stitching Depth Priors into Image Editors via Per-Layer Gradient Probing

**arXiv ID:** 2611.07922 | [PDF](https://arxiv.org/pdf/2611.07922v1)

**作者:** Qin Guo `[一作]` (Hong Kong University of Science and Technology), Dan Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于层级梯度探测的深度适配器精细调优框架，用于提升文本引导的空间图像编辑效果。

**💡 创新点**

创新点在于：①利用每层梯度相似度探测编辑与深度目标的兼容性；②根据探测结果制定动态学习率与剪枝策略，实现仅在兼容层细调，保留几何知识；③构建150案例的StitchEditBench基准，系统评估几何依赖编辑。

**🔧 技术方法**

使用了Diffusion Transformer（Qwen‑Image‑Edit）与LoRA轻量适配器进行深度激活与编辑训练，并结合梯度探测计算每层的相似度。

**📊 数据集**

数据集包括：3M图像‑深度对（用于深度激活）、10K编辑对（源图、文本指令、目标图）以及150个评测案例，源自Nano‑Banana与Gemini生成。

**📈 对比分析**

与Nano‑Banana、Qwen‑Image‑Edit（指令编辑）和Uniform FT（全层统一微调）对比，本文方法在LPIPS降至0.10、DCS升至0.92、CLIP‑Dir升至0.81、用户偏好率达58%，明显优于基线。

**⚠️ 局限性**

局限在于仍依赖自然语言描述的空间编辑指令，缺乏对具体几何参数（如精确旋转角度、位移量）的精准控制。

---

## 452. Towards Multiscale Molecular Representation Learning through Geometry-Based Fragmentation for Polymer Property Prediction

**arXiv ID:** 2611.07998 | [PDF](https://arxiv.org/pdf/2611.07998v1)

**作者:** Makoto Takamoto `[一作]` (NEC Laboratories Europe), Mathias Niepert `[通讯]` (NEC Laboratories Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种仅基于几何信息的半径约束k‑center分片方法，并将其用于生成可调多尺度的聚合物片段特征。

**💡 创新点**

核心创新是将分片转化为半径约束的k‑center优化，消除化学规则依赖，能够灵活控制片段尺度，进而实现多尺度、可调的分子表征。

**🔧 技术方法**

使用几何k‑center分片、MACE‑MP 3D构象编码、PolyBERT/TransPolymer p‑SMILES编码，以及多模态融合与注意力机制。

**📊 数据集**

使用公开的聚合物属性基准集（PI1M 预训练数据与后续测试数据），涵盖电学、热学、光学等多种性质。

**📈 对比分析**

与单模态的p‑SMILES模型（TransPolymer、PolyBERT）以及仅3D模型对比，实验显示加入3D+分片后 RMSE 均降低约5–10%，部分属性提升超过10%。

**⚠️ 局限性**

局限性包括：仅使用 RDKit 生成的单一经典力场构象，未能充分捕捉真实构象多样性；XAI 结果缺乏方向性；对全新化学结构的泛化能力尚待进一步验证。

---

## 453. How Far Can Lightweight LLMs Go in Intent-to-SLO Translation?

**arXiv ID:** 2611.07982 | [PDF](https://arxiv.org/pdf/2611.07982v1)

**作者:** Sabbir Ahmed `[一作]` (Queen's University), Salimur Choudhury `[通讯]` (Queen's University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究轻量化大语言模型（<10B参数）在意图到服务级目标（SLO）翻译中的表现与可行性，构建了包含翻译、验证、幻觉、能耗与鲁棒性等五维度的完整评估框架。

**💡 创新点**

首次系统评估轻量LLM在意图翻译中的多维度能力，提出了能耗与语义鲁棒性指标，并证明在少样本链式思考配置下可达与大型模型相当的翻译质量（F₁≈0.943）。

**🔧 技术方法**

使用10款开源轻量LLM（Qwen、Gemma、Ministral、Granite、Llama系列），4种提示方式（零样本、少样本、含/不含链式思考），结合vLLM推理、NVML能耗计数和手工定义的幻觉检测规则。

**📊 数据集**

基于Islam与Calheiros公开的716条放置意图（其中521有效、195无效）构成的基准集，并生成2,148个重新表述变体用于鲁棒性测试。

**📈 对比分析**

与Intent Engine GPT‑4.1 mini对比，最佳配置实现严格F₁=0.943、拒绝无效意图率≈98%、幻觉率≈1%，单条翻译耗能0.044–0.148 Wh（不含推理），加入推理后能耗提升3.5–7.1×但性能提升有限。

**⚠️ 局限性**

主要限制在隐式SLO的准确性、对自然表述的鲁棒性不足、能耗受GPU空闲功耗主导，且实验仅覆盖单一集群快照与本地GPU，需进一步验证在多节点与边缘加速器上的表现。

---

## 454. Solving Linear Programs in the Matrix Multiplication Time

**arXiv ID:** 2611.07980 | [PDF](https://arxiv.org/pdf/2611.07980v1)

**作者:** Zhao Song `[一作]` `[通讯]`, Zhao Song

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种随机算法，可在矩阵乘法时间内求解有界箱形线性规划问题。

**💡 创新点**

创新点在于将线性规划的时间复杂度降至O((n^ω+n^2 log n) log(1/ε))，回答了Song博士论文中关于矩阵乘法时间的未解问题。

**🔧 技术方法**

利用随机化、精确内积、dyadic投案、ℓ_2采样、Lewis权重维护、反正切势函数等技术。

**📊 数据集**

未使用具体数据集，属于理论算法研究。

**📈 对比分析**

与传统的O(n^3)或其他基于单纯形/内点法的实现相比，本文算法在理论上实现了更快的矩阵乘法时间复杂度，性能上大幅提升。

**⚠️ 局限性**

局限性包括仅适用于有界箱形LP，且对精确实数模型及随机化有要求，且在实际实现中常数项和实现复杂度较高。

---

## 455. Perception Over Geometry: Rethinking Gaussian Allocation in Mesh-based Splatting

**arXiv ID:** 2611.07979 | [PDF](https://arxiv.org/pdf/2611.07979v1)

**作者:** Yuting Wang `[一作]` (Hefei University Of Technology), Ronggang Wang `[通讯]` (Peking University Shenzhen Graduate School)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了P-MGS框架，实现了通过感知复杂度驱动的高效高质量3D高斯渲染与可编辑的网格控制；

**💡 创新点**

将网格几何与高斯表现解耦，采用感知复杂度分配高斯、后期绑定网格与局部仿射变换，提升渲染质量与编辑一致性；

**🔧 技术方法**

3D高斯散射（3DGS）、多视角感知复杂度估计、后期高斯–网格绑定、局部仿射变换；

**📊 数据集**

NeRF-Synthetic与DTU真实物体数据集；

**📈 对比分析**

与传统基于网格的高斯方法（如GaMeS、Mani-GS）及原始3DGS进行对比，P-MGS在保持或提升PSNR/SSIM的同时显著减少高斯数量，编辑时可避免漂浮与失配；

**⚠️ 局限性**

仍需进一步提升在极端非刚性变形下的鲁棒性，并处理对极低纹理区域的高斯分配不足问题。

---

## 456. Bri3D: Bridging the Object-Scene Gap in Single-Image 3D Generation via Scene-Anchor

**arXiv ID:** 2611.07961 | [PDF](https://arxiv.org/pdf/2611.07961v1)

**作者:** Hongye Hou `[一作]` (Hong Kong University of Science and Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于场景支架与锚点的三维场景生成框架Bri3D，可在单张图像条件下生成完整、协调且高保真度的室内场景；

**💡 创新点**

创新点在于引入实例感知的体素场景支架与锚点机制，将全局布局与局部细节分离，场景层负责布局与姿态，预训练单体生成器负责细节合成；

**🔧 技术方法**

使用场景与物体级结构颜色VAE、Rectified Flow Matching、Color‑Attention Gating、DBSCAN聚类、基于体素的细节合成模块（如TRELLIS或Direct3D-S2）等技术；

**📊 数据集**

训练数据来自Objaverse‑XL和3D‑FRONT，使用3D‑FRONT场景拆分为训练/测试集；

**📈 对比分析**

与MIDI、SceneGen、SAM3D、PartCrafter、DepR等基线对比，Bri3D在场景/物体Chamfer距离、F‑score、BBox IoU、视觉指标（PSNR/SSIM/LPIPS/FID/CLIP/DINO）等指标上均表现最佳，并且在全场景生成速度上比SceneGen和SAM3D快约3×；

**⚠️ 局限性**

局限性包括在极度拥挤或多物体高密度场景下可能合并或遗漏物体，且目前仅针对前景物体，缺乏对结构元素（墙、地板、天花板）的完整建模。

---

## 457. AraEngage: Interpretable Linguistic Correlates of Engagement in Arabic Facebook News

**arXiv ID:** 2611.07926 | [PDF](https://arxiv.org/pdf/2611.07926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 458. Rotation Is Token-Blind: A Structural Criterion for Post-Training Quantization of Feed-Forward Geometry Transformers

**arXiv ID:** 2611.07955 | [PDF](https://arxiv.org/pdf/2611.07955v1)

**作者:** Huanfei He `[一作]` (Soochow University), Cong Yang `[通讯]` (Soochow University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种针对几何Transformer的结构感知后训练量化框架（命名为），能够在保持几何重建精度的前提下将模型量化至低比特位。

**💡 创新点**

创新点在于：①识别并量化了四个结构属性（S1–S4）来评估量化对特殊 token 的影响；②引入筛选统计量 Γ 用于判断是否需要行选择的偏置校正；③在保持浮点精度的旋转与平滑变换之外，加入行选择的仿射校正 b_s，解决特殊 token 的网格不匹配与乘法放大效应；④在单个量化步骤中同时兼顾全局共同模式误差与特殊 token 的差异化误差。

**🔧 技术方法**

使用技术包括：正交旋转（Hadamard 变换）、通用的 token‑均匀偏置校正 β、行选择仿射校正 b_s、量化校准、以及基于梯度和激活范围的统计 Γ 与 η_s|β 的评估。

**📊 数据集**

实验数据集包括 ETH3D、ScanNet++v2 与 TartanAirV2‑WB，涵盖室内、室外和合成几何场景，并对 MapAnything、VGGT 与 DA3 三个几何Transformer 进行评估。

**📈 对比分析**

与 RTN、SmoothQuant、QuaRot 等现有 PTQ 方法对比，所提框架在 W4A8 下实现了近乎无损的相机位姿、点云、深度与射线精度，W4A4 下仍保持显著优势；在 W8A8 上可将峰值内存降低 2.2–3.1 倍、推理延迟约为 FP32 的 0.32 倍，并恢复完整的 AUC@30°。

**⚠️ 局限性**

局限性包括：需要专门的结构审计与 Γ 计算，可能对非几何 Transformer 或多头注意力分布差异较大的模型适应性有限；量化后仍需少量校准样本；在极端低位宽（如 W4A4）下性能衰减较明显。

---

## 459. Requirements Nobody Wrote Down: Emergence in a Public-Sector ML Pipeline

**arXiv ID:** 2611.07951 | [PDF](https://arxiv.org/pdf/2611.07951v1)

**作者:** Ronivaldo Ferreira `[一作]` (Federal University of Pará), Gustavo Pinto `[通讯]` (Federal University of Pará)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了巴西全国数字参与平台的主题分类机器学习流水线，并通过15个月的案例研究探讨需求在公共部门ML项目中的产生与演化。

**💡 创新点**

创新点在于揭示需求不在于预先获取，而是通过实验、模型输出和机构制品不断涌现，并提出七种需求出现模式，特别强调公共部门的“无所有者分类词典”与“无赞助承诺”问题。

**🔧 技术方法**

使用的技术包括无监督主题建模（BERTopic）、受控词典（VCGE）种子、监督分类、Airflow/MLflow自动化、CPU推理及与Decidim平台的集成。

**📊 数据集**

数据集为平台收集的约60,000条市民提案文本（单一过程约8,254条），并基于这些文本构建训练集与验证集。

**📈 对比分析**

方法上主要比较了无监督与监督模型的主题一致性、F1等指标，但未给出最终系统在生产环境中的性能对比，且缺乏实际使用者的评估。

**⚠️ 局限性**

局限性包括回顾性设计导致回忆偏差、政府方工件缺失、未访谈市民或决策者、缺乏正式需求文档与使用验证，以及研究仅为单例。

---

## 460. WAM-Sim0.1: World Anchored Memory Meets Online RL for Consistent and Realistic Driving Simulation

**arXiv ID:** 2611.07939 | [PDF](https://arxiv.org/pdf/2611.07939v1)

**作者:** Jiahao Cui `[一作]` (Fudan University), Siyu Zhu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种长时序驾驶视频生成框架WAM‑Sim，通过持久化世界记忆和在线流匹配RL实现结构、身份与质量的稳定生成。

**💡 创新点**

创新点在于三层持久化记忆（结构、场景、实例）与在线DiffusionNFT奖励优化的协同体系，既防止结构漂移、身份漂移，又提升视觉与物理真实感。

**🔧 技术方法**

采用3D Diffusion Transformer、HD地图编码、位置化场景检索、写一次实例记忆库、Flow‑Matching RL、视觉语言模型和物理判定器等技术。

**📊 数据集**

使用nuScenes基准集与自采集的约1,000小时多模态驾驶数据集，涵盖多天气与复杂场景。

**📈 对比分析**

在nuScenes上通过FID=3.47、FVD=25.72、24.16 mAP、71.23 mIoU等指标优于现有方法；自采集长时序评测显示Scene Similarity=0.910、Subject Coherence=0.681、VBench=78.71，表现领先。

**⚠️ 局限性**

局限性包括对地理数据库的依赖、记忆规模受限，且在极端遮挡或极长时延下仍可能出现细节漂移。

---

## 461. Efficiently Estimating Cycle Statistics on Graphs via Spanning Trees

**arXiv ID:** 2611.07925 | [PDF](https://arxiv.org/pdf/2611.07925v1)

**作者:** Josef Hoppe `[一作]` (RWTH Aachen University), Michael T. Schaub `[通讯]` (RWTH Aachen University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于均匀生成树和 Horvitz–Thompson 估计的 Monte Carlo 方法，用以在不枚举所有简单环的情况下估计图的环统计量。

**💡 创新点**

创新点在于利用均匀生成树产生的基本环并计算其包含概率，构造不偏的 HT 估计器，并给出近似计算包含概率的均值场近似，实现近线性时间的可扩展采样；同时引入多重 HT 平均的 MCHT 方法并比较单环重要性采样。

**🔧 技术方法**

使用了均匀生成树采样（Wilson 算法）、Kirchhoff 矩阵树定理、Horvitz–Thompson 估计、重要性采样、均值场近似、LCA 预处理及逆 Ackermann 函数加速等技术。

**📊 数据集**

实验使用合成 Erdős–Rényi 随机图、稠密随机图、10×10 网格、Delaunay 三角剖分以及来自 Netzschleuder 的真实网络（如 polbooks、contact 等）。

**📈 对比分析**

与 Giscard 等人基于子图的 Monte Carlo 方法和传统重要性采样进行比较；MCHT 的方差最低、MCOC 次之、IS 最差；近似版本在计数任务中虽然引入可接受的偏差，但显著提升速度（线性/近线性时间），在大图（10⁷ 边）仍可高效运行。

**⚠️ 局限性**

受生成树直径限制，最多能高效采样长度约 O(√n) 的环；均值场近似在稀疏或高直径图上误差增大，可能导致计数估计产生偏差；极长环（如 Hamiltonian 循环）仍需指数多树。

---

## 462. SymFC: Configurable Semantics for All Paths Symbolic Execution in Haskell via Embeddings

**arXiv ID:** 2611.07916 | [PDF](https://arxiv.org/pdf/2611.07916v1)

**作者:** Robin Webbers `[一作]` (Vrije Universiteit Amsterdam), Klaus v. Gleissenthall `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文实现了一个基于 GHC Core 的全路径符号执行引擎，能够在 Haskell 代码中直接指定属性并在编译时通过 SMT 求解器验证其正确性。

**💡 创新点**

核心创新在于提出可配置语义（embeddings）机制，利用 Haskell 的 coercion 机制让用户在属性级别自定义任意类型与函数的符号解释，从而兼容 IO、指针、硬件描述等多样化语言特性并显著提升执行效率。

**🔧 技术方法**

技术实现包括：GHC 插件框架、Grisette 库提供的有序 guard 合并、所有路径符号执行的 reduction 语义、以及对类型和术语 embedding 的实例化与合成策略。

**📊 数据集**

评估使用了 GHC core 的标准库（Data.Map、Data.Set）、第三方库（iproute）以及低层 IO/指针 API（IORef、Ptr/ForeignPtr）的 54 条属性，此外还在 G2Q 基准上与 G2 与 Grisette 进行对比。

**📈 对比分析**

与 G2 的比较显示，在同一套基准下，本文实现的工具平均 6.25 倍速度提升；在 54 条属性验证中，Z3 求解时间从几毫秒到十几毫秒不等，全部通过编译时检查且能生成对抗例子。

**⚠️ 局限性**

主要局限包括对 GHC 内联优化的不兼容（需关闭优化以保留 embedding），无法直接使用 GHC 预编译的 base 库版本，以及对多态容器和全局内存模型的支持尚不完善。

---

## 463. GuidedWAM: Training-Free Energy Guidance for World Action Models in Autonomous Driving

**arXiv ID:** 2611.07912 | [PDF](https://arxiv.org/pdf/2611.07912v1)

**作者:** Xiyan Su `[一作]` (Technical University of Munich), Markus Lienkamp `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在冻结的世界行动模型（WAM）上实现一种无训练的能量引导方法，能够在采样时将新的驾驶目标注入模型。

**💡 创新点**

提出了 Jacobian‑free 的能量引导框架 GuidedWAM，结合噪声级三角窗调度、每能量归一化和速度限幅校正，使能量引导在不需要反向传播的情况下稳定且高效。

**🔧 技术方法**

使用流匹配采样器、可微分能量函数、噪声级调度、梯度归一化与速度限幅等技术实现无训练能量引导。

**📊 数据集**

在 NAVSIM v2 hard‑split 数据集上进行评估。

**📈 对比分析**

相较于原始冻结模型，GuidedWAM 在三种公开 WAM（DriveLaW、DriveWAM、SimWAM）上分别提升 1.8–3.6 EPDMS，采样时间增加不到 10%，并且在舒适度指标上优于后验投影方法；与投影相比仅牺牲少量 EPDMS 但保持更好的连续性和舒适性。

**⚠️ 局限性**

评估仅为开环伪仿真，软约束无法保证完全遵守目标；能量加权求和无法处理相互矛盾的目标；仅在确定性欧拉积分器上验证，未测试随机采样器或闭环控制场景。

---

## 464. SEER: A Scalable Event-Driven Runtime for Parallel Execution of Multi-Agent Reinforcement Learning Environments

**arXiv ID:** 2611.07996 | [PDF](https://arxiv.org/pdf/2611.07996v1)

**作者:** Jiming Su `[一作]` (National University of Defense Technology), Miao Zhang `[通讯]` (National University of Defense Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 SEER，一种事件驱动的多智能体强化学习环境运行时，用来降低 CPU 侧环境执行的延迟和开销。

**💡 创新点**

创新点包括：① 引入事件本地性契约，将跨实体交互抽象为有序事件，从而在并发执行中保证序列等价的边界输出；② 构建 CAS 基于的双层事件队列，将本地事件队列与全局可执行事件列表分离，既保持实体因果顺序，又实现高效并行调度；③ 通过预飞行轨迹与静态硬件/工作负载特征结合的 STPA 方法，在正式执行前预测合适的线程池大小，实现工作负载感知的线程调度。

**🔧 技术方法**

使用技术包括：事件驱动并发模型、CAS 原子操作的分布式局部事件队列与全局可执行列表、动态并行调度算法、时间序列编码 TimesURL、XGBoost 回归模型用于线程数预测、基于 ROSS‑T 的实现框架。

**📊 数据集**

实验使用了两个参数化 MARL 工作负载：Aircraft（基于真实飞行数据的无人机控制与碰撞避免）和 PCS（个人通信系统，通信密集型）。

**📈 对比分析**

方法上与 ROSS‑T、NBSE、ROOT‑Sim 等现有运行时进行对比；在 Aircraft 上，SEER 将环境执行吞吐率从 16.92 步/秒提升至 52.93 步/秒（约 143% 的提升），延迟下降 68%；在 PCS 上，SEER 与 ROSS‑T 相比，延迟降低 46.9% 并将吞吐率提升 64.9%；线程预测误差 MAE 仅为 0.3394，显示预测效果良好。

**⚠️ 局限性**

局限性包括：仅适用于满足事件本地性契约的环境，无法处理直接共享写；线程预测依赖于预飞行阶段的代表性，若工作负载发生剧烈变化可能失效；局部事件队列与全局可执行列表的拆分在某些场景下可能导致缓存局部性降低。

---

## 465. Stealthy Malicious Interference Detection in UAV Formations: A Power-Domain Signaling Game Approach

**arXiv ID:** 2611.07977 | [PDF](https://arxiv.org/pdf/2611.07977v1)

**作者:** Jianle Ba `[一作]` (Xi'an Jiaotong University), Zhendong Li `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于贝叶斯安全博弈的无人机群体隐蔽干扰检测框架，利用干扰源的发射功率导致的 SINR 变化来更新对干扰类型的后验信念，并根据该信念决定是否对目标链路进行功率增强。

**💡 创新点**

创新点在于：①将干扰源的隐蔽性通过私人类型和功率策略建模为博弈中的信息不对称问题；②设计了纯策略完美贝叶斯均衡（PBE）分析，给出了分离与聚合结构下的最优信念阈值决策；③将通信链路收益、功率消耗、误报与漏报损失统一到一个支付函数中，实现了成本感知的防御策略。

**🔧 技术方法**

技术方法包括：贝叶斯安全博弈建模、后验信念更新、纯策略 PBE 推导、分离与聚合结构的 IC（激励相容）约束分析、基于 SINR 的阈值决策。

**📊 数据集**

本研究使用仿真数据，未采用公开数据集；通过给定的链路参数（频率、功率、信道增益等）进行 Monte Carlo 评估。

**📈 对比分析**

与三种基线方案（始终开启功率增强、固定阈值检测、始终不增强）进行对比，结果表明在不同干扰源先验概率、功率成本以及距离变化情形下，PBE 基于贝叶斯推理的方案在期望防御效用上均有显著提升，尤其在隐蔽干扰环境中表现更佳。

**⚠️ 局限性**

局限性包括：①仅考虑单一干扰源和两种功率水平；②假设信道为静态且已知；③未考虑多干扰源、时变信道或更复杂的发射策略；④纯策略均衡可能并非唯一，且对参数敏感。

---

## 466. Tight Asymptotic Bounds on the Maximum Number of Linear Regions of Piecewise-Linear Kolmogorov--Arnold networks

**arXiv ID:** 2611.07973 | [PDF](https://arxiv.org/pdf/2611.07973v1)

**作者:** Ye Su `[一作]` (Shenzhen Institutes of Advanced Technology), Yong Liu `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

在论文中，作者对连续分段线性 Kolmogorov–Arnold 网络（CPWL KAN）进行了严格的线性区域计数分析，给出了与网络宽度、深度和每条边段数相关的匹配上下界。

**💡 创新点**

创新点在于通过有效维度和并行超平面排列的几何视角，精确刻画了每层的可分割能力，得出了与先前松散上界相比的紧致指数表达式，并证明了共享点位模型同样达到最优指数。

**🔧 技术方法**

使用的技术主要包括多维超平面排列理论、并行族超平面计数公式、折叠函数构造以及对深层网络的递归分解，结合组合几何与线性代数推导出上下界。

**📊 数据集**

由于研究完全是理论性的，本文未使用任何实验数据集，而是通过解析证明和符号计算给出结果。

**📈 对比分析**

作者将 CPWL KAN 与 ReLU 网络在相同参数预算和层数下进行比较，证明在 T≥2 时 KAN 的最大线性区域数以 Λ^dT 增长，而 ReLU 为 Λ^d⌈T/2⌉，因而 KAN 的表现显著优于 ReLU。

**⚠️ 局限性**

限制方面，结果仅适用于标量输出的 CPWL KAN，且假设每条边最多 k 段，未探讨训练动态、泛化能力或高阶非线性激活的实际表现。

---

## 467. Time-Indexed Skill Success Models for Latency-Aware Skill Selection and Timing in Dynamic Manipulation

**arXiv ID:** 2611.07964 | [PDF](https://arxiv.org/pdf/2611.07964v1)

**作者:** Mohsen Omidi `[一作]` (Technological University Dublin), Brian Vaughan `[通讯]` (Technological University Dublin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出基于延迟时间的技能成功模型（Time‑Indexed Skill Success Models），用于在存在感知与执行延迟的动态操作中预测技能启动后整项任务的成功概率，并在此基础上设计递归视野规划器来选择最佳技能及其启动时机。

**💡 创新点**

创新点在于：①将技能成功率建模为决策状态与启动延迟的联合函数；②利用分支回放（branched rollouts）得到精确的延迟标签，直接训练时间索引模型；③在规划层面同时考虑技能选择和等待时间，提升对延迟不确定性的鲁棒性。

**🔧 技术方法**

技术上采用了 MuJoCo 仿真与硬件匹配的物理模拟，使用 MLP 与梯度提升树（GBDT）进行概率回归；通过多量化指标（ECE、Brier、95% 置信区间）评估模型校准与性能；使用多重随机种子与分布位移进行实验复现。

**📊 数据集**

实验数据包括 20,000 条仿真 episode，产生 540,000 条回放样本（含 9 个延迟网格）；硬件测试使用 49 次抓取实验（35 次带 0.2 s 延迟）。

**📈 对比分析**

与静态模型、基于预测的模型及规则基准相比，时间索引模型在 100–300 ms 延迟下提升 2.9–5.3 个百分点（95% 区间均 >0），并在随机延迟情形下保持优势；在硬件测试中，静态模型对高置信度延迟抓取的预测为 99% 但实际成功仅 63%，而时间索引模型预测 56%。

**⚠️ 局限性**

局限性包括：仅使用解析控制器、未考虑学习型技能与等待期间的前置姿态；时间索引模型在仿真中只对 9 个离散延迟进行插值；硬件比较仅覆盖单一技能与单一延迟；实验基于仿真，缺乏大规模真实世界验证。

---

## 468. Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification

**arXiv ID:** 2611.07950 | [PDF](https://arxiv.org/pdf/2611.07950v1)

**作者:** Dang-Khoa Bui `[一作]` (Ho Chi Minh City University Of Technology), Hoang-Anh Ngo `[通讯]` (AK Technologies Company Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对基于MIMIC-CXR和MIMIC-IV的多模态胸片与EHR分类，使用类不平衡处理策略进行基准测试，并通过严格的患者分层和标签重构审计了记录级分割导致的泄漏，重新评估各策略、特征融合与分类器在不同实验协议下的性能。

**💡 创新点**

提出了泄漏感知的基准设计和完整的诊断检查清单；发现记录级分割导致的患者身份泄漏使得EHR模型的优势是伪造的；在患者分层、时间对齐标签和诊断历史的评估中，图像嵌入和诊断历史才真正提供信号，且不再出现明显的类不平衡采样器优势。

**🔧 技术方法**

使用SMOTE、随机过采样、SMOTE-NC、权重平衡、Balanced Random Forest、EasyEnsemble等采样与分类策略；融合技术包括早期融合、晚期融合以及图像特征（11个病理得分、DenseNet-121 1024维池化特征、RAD‑DINO 768维CLS嵌入）与EHR特征；使用逻辑回归、随机森林、Extra Trees、LightGBM、XGBoost等分类器；采用PCA降维、距离加权5-NN填补缺失等预处理。

**📊 数据集**

利用MIMIC-IV（患者病历、ICD-10代码、结构化EHR变量）与MIMIC-CXR-JPG（胸片图像）共计29,083张胸片，覆盖10,479名患者，八个二分类诊断目标（肺炎、肺结核、支气管炎、关节炎、骨折、肺癌、脊柱侧弯、肺部感染）。

**📈 对比分析**

比较方法：在不同协议（记录级分割、患者级分割、对齐标签、规则精炼标签、日历时间分割）下，计算宏观F2、PR-AUC、AUROC、Brier分数；基准包括全正预测、EHR查找表。结果显示：在记录级分割下EHR模型和查找表接近最优，但在患者级分割和时间对齐标签下，所有模型均不显著优于全正基线；图像嵌入（尤其是RAD‑DINO）与诊断历史在F2上略有提升（最大提升≈0.05），但不超过统计显著阈值；采样器在患者分层下几乎无效。

**⚠️ 局限性**

局限性包括：仅使用单一机构的MIMIC数据；实验中未对实验室和生命体征进行时间对齐；RAD‑DINO在预训练阶段已接触大部分评估图像，可能导致结果偏高；未对模型进行微调；CheXpert报告标签噪声未完全处理；实验仅针对特定目标与模型组合，难以推广到其他任务或数据集。

---

## 469. Circe: Training-Free LLM Self-Evaluation via Component Interaction Consistency

**arXiv ID:** 2611.07948 | [PDF](https://arxiv.org/pdf/2611.07948v1)

**作者:** Shuyi Miao `[一作]` (Beihang University), Tat-Seng Chua `[通讯]` (National University Of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种无参考答案、无需重复采样的LLM自评方法Circe，用以判定生成答案的正确性。

**💡 创新点**

创新点在于引入跨层注意力更新与FFN输出变化的方向一致性(CIC)以及表示几何(RG)与输出置信度(LC)的融合，从内部计算角度补充传统输出置信度的局限。

**🔧 技术方法**

技术上结合Transformer内部激活、注意力与FFN差分、隐藏状态轨迹、平均负对数似然转化的置信度等信号，形成三种互补指标并通过加权融合得到最终得分。

**📊 数据集**

使用六个公开基准（GSM8K、MATH、CommonsenseQA、TheoremQA、MMLU、Belebele）及多语言MGSM、视觉文本Geometry3K等数据集进行评估。

**📈 对比分析**

与12种无参考评估基线以及采样法对比，Circe在四大任务域（数学、推理、知识、理解）AUROC均高于最佳基线，提升幅度约5%–7%，并在多语言、多模态及开放世界分布下保持竞争力。

**⚠️ 局限性**

局限在于对超大模型的内部状态读取与计算仍有一定开销，且在极低准确率或极少样本情境下性能可能不稳定。

---

## 470. Operational Indirect Rate-Distortion Functions for Classification Tasks

**arXiv ID:** 2611.07946 | [PDF](https://arxiv.org/pdf/2611.07946v1)

**作者:** Yann Miguet `[一作]` (IMT Atlantique), Philippe Mary `[通讯]` (Univ Rennes)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析目标导向通信中分类任务的间接源编码，提出四种操作性率失真函数并对其层次关系进行理论推导；

**💡 创新点**

将间接率失真函数与实际学习型编码方案（VIB、VAE、估计-压缩等）联系起来，量化中间表示、网络架构与训练损失导致的三个性能损失源；

**🔧 技术方法**

使用信息瓶颈与变分自编码器框架、Blahut‑Arimoto算法、KKT逆水位填充、凸优化与水位填充求解；

**📊 数据集**

在高斯混合模型和CIFAR‑100图像分类数据集上进行实验；

**📈 对比分析**

通过与理论间接率失真函数以及不同编码方案的对比，发现VIB方案在接近理论极限时性能优异，而VAE方案因使用重构损失导致显著失真；

**⚠️ 局限性**

局限性在于只考虑了分类任务、采用的近似优化方法、对实际任务感知的损失不足，以及对更复杂数据分布与多任务场景的扩展不足。

---

## 471. CardioLM: A multilingual suite of small language models for the cardiology domain

**arXiv ID:** 2611.07940 | [PDF](https://arxiv.org/pdf/2611.07940v1)

**作者:** Manuela-Daniela Danu `[一作]` (Siemens SRL), Bram van Es `[通讯]` (University Medical Center Utrecht)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了七种欧洲语言的心血管临床案例语料库CardioCCC，并训练了CardioLM模型（CardioBERTa和CardioNER），实现多语言、四类实体（疾病、症状、程序、药物）的命名实体识别；

**💡 创新点**

首次在心血管专用领域实现跨语言、全实体覆盖的标准化语料和模型；将心血管特定的持续预训练与多语言联合训练相结合，验证心血管预训练对NER影响有限，但多语言共享能显著提升性能；

**🔧 技术方法**

采用Transformer编码器（BERT/Roberta/XLM‑RoBERTa）进行持续自监督预训练（MLM）和下游BIO标注的token‑classification微调；同时使用伪困惑度（pseudo‑perplexity）评估模型对心血管文本的适配度；

**📊 数据集**

数据来源包括：1）CardioCCC 508篇案例报告（原始西班牙语+机器翻译+人工校正），共400,506条实体标注；2）持续预训练语料约24.9 B词，涵盖心血管、医学及一般文本；3）ParaCliTe 658个人工合成病例，用于伪困惑度评估；4）ELCardioCC 20份匿名出院记录，用于临床验证；

**📈 对比分析**

与基线（规则+词典）以及各语言原始模型比较。CardioBERTa在伪困惑度上显著降低（至1/8），但在严格F1上提升≤0.04。多语言XLM‑RoBERTa‑Large在所有语言、实体类型上均优于单语模型，严格micro‑F1提升≈0.06，药物实体最易识别（≈0.90），症状最难（≈0.57）。临床评估显示CardioNER的整体满意度从26%提升至73%；

**⚠️ 局限性**

心血管预训练对下游NER提升有限，且受限于机器翻译误差和实体边界错误；多语言训练虽效果好，但在计算成本、部署规模上更高；此外，实体覆盖仍受标注一致性和语言差异影响，未覆盖所有临床信息（如关系、层级）。

---

## 472. Adaptive Semantic Branching: Disagreement-Localized Cross-Trajectory Regeneration for Reasoning Candidate Expansion

**arXiv ID:** 2611.07918 | [PDF](https://arxiv.org/pdf/2611.07918v1)

**作者:** Jeong-Yoon Kim `[一作]` (Hanbat National University), Tae-Yoon Lee `[通讯]` (Hanbat National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种无训练的推理框架Adaptive Semantic Branching，利用不同完整推理路径的分歧信息来生成新的候选答案。

**💡 创新点**

核心创新在于通过动态时间规整对跨路径步骤进行单调对齐，定位持久分歧边界，并在此处将来源路径的前缀与目标路径的局部推理窗口结合，形成有方向的局部重生成。

**🔧 技术方法**

使用了基于词和字符4-gram的哈希嵌入做步骤相似度、DTW对齐、持久分歧指标、分支得分计算以及基于答案聚类的自适应分支预算。

**📊 数据集**

在MATH 500数学题集上评估，采用Qwen2.5‑7B‑Instruct模型。

**📈 对比分析**

与仅使用七个独立采样的基线相比，方法将候选覆盖率从89.0%提升至91.0%，额外生成576个分支，平均每题8.15次调用，覆盖率提升2个百分点。

**⚠️ 局限性**

局限性包括：需要额外的生成调用和计算开销；对答案解析与聚类质量敏感；当前仅在数学题目上验证，其他领域需进一步测试。

---

## 473. Same Answer, Different Future: Matched Predictions Do Not Imply Matched Belief Updating in Language Models

**arXiv ID:** 2611.07913 | [PDF](https://arxiv.org/pdf/2611.07913v1)

**作者:** Weihan Li `[一作]` (University of Tokyo), Tianshi Zheng `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在三种开源大语言模型上对伯努利推断任务进行实验，探究不同信息呈现方式（观察序列、计数摘要、声明、指令）以及激活层干预（steering、状态移植）如何影响模型的“信念内容”（当前概率估计）和“信念权重”（在后续证据下的更新力度）。

**💡 创新点**

创新点在于将信念分为“内容”和“权重”两维度，并提出通过“交换率”量化权重；证明不同路由即使在当前输出相同，也会导致在相同新证据下的更新路径不同；揭示激活层干预能写入内容却几乎不携带权重，说明内容与权重由不同内部通道决定；进一步说明文本单位（单词、句子、摘要）的长度和形式对权重影响显著。

**🔧 技术方法**

采用的技术主要包括：
- 在大语言模型中注入不同格式的证据（序列、计数、声明、指令）；
- 激活层 steering 与隐藏状态移植两种干预方法；
- 读取两种推理通道（文档续写 vs 报告式回答）来获取内容；
- 对后续证据段落进行回归分析，计算旧信念对新观测的有效观测量（交换率）。

**📊 数据集**

使用了人工生成的伯努利试验数据集：随机生成不同长度（8/16/32/64）抛硬币序列，随后以相同或不同比例（0.25/0.75）追加新的观测段；对每条记录还生成对应的计数摘要、声明或指令文本；共构造数千个对照组以进行统计分析。

**📈 对比分析**

比较方法：在三种模型（Llama‑3.1‑8B‑Instruct、Qwen3‑8B、Gemma‑3‑12B‑it）上分别测量内容与权重；与理论贝叶斯后验（均匀 Beta prior）做对比；使用 bootstrap 95% 置信区间评估差异。结果显示：
- 记录序列对续写通道影响最大，而摘要/声明对报告通道影响显著；
- 记录的权重随长度饱和，仅为新观测的一小部分；
- 摘要的权重仅随声明计数微弱变化；
- 激活干预写入的内容几乎不保留权重，且随后观测快速消失；
- 对比权重，某些路由（如指令）在某些查询下可获得几百甚至上千个等价观测。

**⚠️ 局限性**

局限性包括：
- 仅研究单一伯努利推断任务，未验证多项式或更复杂分布；
- 只涉及三种模型，缺乏更大或更旧模型的泛化性；
- 依赖人工生成的序列与摘要，未探讨真实世界语言的噪声与多义性；
- 权重定义为回归估计，未直接测量内部记忆或网络状态；
- 研究集中在文本层面，未考虑模型的参数更新或长期记忆机制。

---

## 474. EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure

**arXiv ID:** 2611.07909 | [PDF](https://arxiv.org/pdf/2611.07909v1)

**作者:** Kwan Yun `[一作]` (KAIST), Junyong Noh `[通讯]` (KAIST)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研发了EDGAR，一个基于扩散模型的高效Gaussian头像细化框架，能够修复单视/单图训练得到的Gaussian avatar的几何与纹理缺陷。

**💡 创新点**

创新点在于：①构造真实+合成的三元组数据集EDGAR-SET模拟多种失效模式；②利用Sapiens特征条件化与单步扩散加Mixture‑of‑Experts实现身份与表情一致的高效去噪；③将细化结果反向优化原始Gaussian avatar，使改进无推理开销。

**🔧 技术方法**

采用SD‑Turbo单步扩散、Sapiens自监督特征、MoE架构、表达一致性损失、LPIPS/Gram等感知损失，并结合Gaussian splatting渲染技术。

**📊 数据集**

使用NeRSemble、CAFCA等真实与合成数据构建的EDGAR‑SET（约400K图像），并在CAP4D、RGBAvatar、GaussianAvatars等三种基线上评测。

**📈 对比分析**

通过与基线模型及SDEdit单步扩散在CLIP、LPIPS、PSNR、L1、CSIM等指标上对比，EDGAR在所有基线与输入量级下均在视觉质量、身份保持、跨视角一致性及大角度/大张口表现上超越对照组，且推理时无额外开销。

**⚠️ 局限性**

局限性在于只能修复可恢复的纹理/几何失真，无法处理严重结构错误；多视角伪目标不一致可能导致3D细节冲突；在某些管线中对时序稳定性的提升有限；对眼部内部、透明区域等细部仍具挑战。

---

## 475. IDPrior: Decoupling Identity Generation from Facial Image Synthesis

**arXiv ID:** 2611.07941 | [PDF](https://arxiv.org/pdf/2611.07941v1)

**作者:** Fadi Boutros `[一作]` (Fraunhofer Institute for Computer Graphics Research IGD), Naser Damer `[通讯]` (Fraunhofer Institute for Computer Graphics Research IGD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了IDPrior框架，分两阶段：先在预训练人脸识别模型的嵌入空间学习扩散先验生成身份嵌入，再用身份条件扩散模型合成多张对应身份的人脸图像。

**💡 创新点**

创新在于显式学习身份嵌入分布的扩散先验，解耦身份生成与图像合成，可与任何身份条件生成模型无缝组合，且在大规模身份生成上保持高区分度与分布逼真度。

**🔧 技术方法**

使用扩散概率模型（DDPM）、预训练人脸识别网络ArcFace/ElasticFace、条件扩散生成器、DDIM采样、U‑Net网络等技术。

**📊 数据集**

在WF4M（约400万张人脸）上训练身份先验，随后在多种身份条件生成模型上评估，验证数据集为LFW、AgeDB‑30、CFP‑FP、CA‑LFW、CP‑LFW。

**📈 对比分析**

通过身份唯一性、ID‑FID、FR验证准确率等指标与PCA、随机采样、VIGFace、DCFace等传统方法对比，IDPrior (DDPM) 在身份唯一性高、ID‑FID低、FR验证精度超越或等同现有合成方法，并能提升多种生成器的下游识别性能。

**⚠️ 局限性**

主要限制在于需要大量真实身份嵌入来训练先验，对预训练FR模型的表达能力依赖较大；生成的身份在极端表情或姿态上可能欠缺；虽然可兼容多种生成器，但仍需手工适配。

---

## 476. Towards Scalable Diagnostic-Driven Whole-Slide Image Retrieval for Pathology Archives

**arXiv ID:** 2611.07935 | [PDF](https://arxiv.org/pdf/2611.07935v1)

**作者:** Jingsong Liu `[一作]`, Peter J. Schüffler `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可扩展的、诊断驱动的全切片检索框架 CLEAR-WSI，利用固定长度嵌入实现高效检索并持续扩充检索库。

**💡 创新点**

创新点在于将注意力多实例学习（DX‑Agg）与诊断先验结合，实现不受采样或聚类影响的全切片固定向量表示，同时通过自检索（SR）和类信息检索（CI）两种模式提升诊断一致性，并支持检索库随时间增长。

**🔧 技术方法**

核心技术包括：基于基础模型（UNI）的切片块编码、注意力多实例聚合、余弦相似度检索、诊断预测筛选以及增长式内存池。

**📊 数据集**

使用了 CAMELYON16（淋巴结转移），BRACS（乳腺癌分型，包含粗粒级别 BRACS‑1 与细粒级别 BRACS‑2）和私有的 X‑HER2（HER2 状态）三组数据集进行评估。

**📈 对比分析**

与 Yottixel‑K、SISH、RetCCL 等基准方法比较，CLEAR‑WSI 在 Precision@5 及 NDCG@5 上均取得显著提升（例如 CAMELYON16 + SR 提升 3.10%），且在复杂的 BRACS 任务中优势更为明显。

**⚠️ 局限性**

局限性包括：仅在少量（百级）切片上验证，未覆盖多器官或大规模医院档案；检索相关性仅依据数据集标签，缺乏临床意义验证；以及在单一 GPU 设备上的延迟评估，未充分证明在生产环境的可扩展性。

---

## 477. The Moving Frontier: How the National Justice Project Is Using Digital Technology to Expand Access to Justice

**arXiv ID:** 2611.08052 | [PDF](https://arxiv.org/pdf/2611.08052v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 478. The Blindfolded Controller: Decision Model-Guided Training for Reinforcement Learning

**arXiv ID:** 2611.07934 | [PDF](https://arxiv.org/pdf/2611.07934v1)

**作者:** Muhammad Ibrahim Khan `[一作]` (Coventry University), James Brusey `[通讯]` (Coventry University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种将零样本决策模型（DM）与强化学习（RL）相结合的决策模型引导训练（DMGT）框架；

**💡 创新点**

创新点在于利用DM的即时、零样本决策为RL提供起步策略，再通过RL进一步改进；

**🔧 技术方法**

采用了决策模型d3-turbo进行决策、行为克隆或DAgger进行模仿学习、PPO作为RL优化器；

**📊 数据集**

在六个Gym/MiniGrid/Arcade Learning Environment等环境上进行实验，包含七个任务；

**📈 对比分析**

与从零开始的PPO和单独DM进行对比，DMGT在七项任务中全部优于PPO、在五项任务中优于DM，学习速度明显加快；

**⚠️ 局限性**

局限性包括对DM描述的依赖性、DM对环境的泛化受限、对更大规模或更复杂任务的可扩展性未作充分验证。

---

## 479. MUSS: A Multi-scale and Sequence-based Model for Single-cell Gene Regulation

**arXiv ID:** 2611.08086 | [PDF](https://arxiv.org/pdf/2611.08086v1)

**作者:** Kangjie Zheng `[一作]` (Wellcome Sanger Institute), Mo Lotfollahi `[通讯]` (Wellcome Sanger Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于DNA序列的多尺度模型MUSS，联合表示单细胞ATAC‑seq和RNA‑seq中的峰和基因，支持细胞级和基因级的可解释性预测。

**💡 创新点**

创新点在于：① 用冻结的AlphaGenome编码器从DNA直接生成峰与基因的嵌入；② 在多尺度编码器中引入峰‑峰自注意力和距离加权的峰‑基因交叉注意力；③ 通过峰掩码语言模型预训练获取细胞可变的染色质可达性上下文；④ 能对训练外的基因、峰以及细胞类型进行零样本或少样本预测。

**🔧 技术方法**

技术包括：冻结的序列预训练模型（AlphaGenome）、RoPE位置编码、RBF距离偏置、距离衰减惩罚、跨尺度交互块、峰掩码语言模型、Poisson式基因表达损失。

**📊 数据集**

使用了四个公共单细胞配对数据集（PBMC10K、BMMC、OpenProblem、Bone），并在PBMC169K上进行预训练，保留部分批次用于零样本验证。

**📈 对比分析**

与多种基线（CMAE、scMoGNN、ChromFound、EpiAgent、Borzoi、AlphaGenome、scooby、GET、EpiBERT）比较，在基因表达预测、增强子‑基因连接和零样本细胞聚类任务中，多数指标（pseudobulk PCC、单细胞 PCC、AUROC、ARI等）均优于或持平于现有方法，尤其在训练外基因/峰的泛化表现突出。

**⚠️ 局限性**

局限性包括：依赖大型冻结序列编码器与配对多组学数据；自注意力计算为二次复杂度；推理速度慢；增强子‑基因评估仅覆盖训练词汇，未验证对未知链接的泛化；交叉尺度信息流向未完全解耦。

---

## 480. Automatic keyphrase extraction and annotation: modern theoretical approaches and practical solutions for text and speech

**arXiv ID:** 2611.08006 | [PDF](https://arxiv.org/pdf/2611.08006v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 481. When Imperfect Automated Fact-Checking Backfires: Social Learning in the Presence of Misinformation

**arXiv ID:** 2611.08053 | [PDF](https://arxiv.org/pdf/2611.08053v1)

**作者:** Zixuan Liu `[一作]` (University of Groningen), Ming Cao `[通讯]` (University of Groningen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在顺序社会学习框架下研究当平台同时提供免费、错误率较高的自动化事实核查工具与收费且准确的人工核查工具时，正确信息级联（即正确认知的连锁反应）概率的变化。

**💡 创新点**

揭示了“更广泛核查”与“误导错误”两种对福利的相反影响，并给出了在不同可靠性区间内自动化工具必然降低福利的阈值；同时证明了在人工核查成本足够高且真新闻信息不够显著时，自动化工具的加入会降低整体福利。

**🔧 技术方法**

使用顺序贝叶斯社会学习模型、信息级联分析、阈值推导、Markov 过程解析以及蒙特卡洛仿真来计算不同方法组合下的福利概率。

**📊 数据集**

使用完全由模型生成的合成数据：文章可靠性 r、真新闻概率 p、先验分布 H（Beta(0.5,0.5)）以及人工/自动核查的误差率与成本，未使用真实网络数据。

**📈 对比分析**

通过将只有人工核查、仅自动核查和两者同时可用三种情形的福利（即正确级联概率）进行对比。仿真显示：在可靠性区间 (r_-, r_+) 内，自动化工具可提升福利；在 (r_+, r_+) 区间则降低福利；平均福利提升仅在工具准确率超过约 0.81 时才出现。

**⚠️ 局限性**

局限包括：① 仅在理想化的单一错误率设定下得到结果，未给出平均福利提升阈值的解析表达式；② 仅考虑隐私核查（仅行动可见），未研究公开核查对级联的影响；③ 假设先验分布对称且仅有两类信息源；④ 只分析了两种极端核查方式，未探讨多源、多成本情形。

---

## 482. HITCH: Teaching a Vision-Language Model to Operate a Humanoid through a Whole-Body Harness

**arXiv ID:** 2611.08039 | [PDF](https://arxiv.org/pdf/2611.08039v1)

**作者:** Qingzhou Lu `[一作]` (Tsinghua University), Jianyu Chen `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HITCH harness，使 VLM 能通过感知和执行接口控制 Unitree G1 humanoid。

**💡 创新点**

创新点在于：①双接口 harness（主动感知返回 3D 位置，执行接口将身体部位指令转为全身运动）；②通过教师策略轨迹微调 VLM 学习全身控制技能；③在不使用任务专属数据的情况下实现跨任务迁移和真实部署。

**🔧 技术方法**

使用 Qwen3.8‑27B 作为基础 VLM，HEFT 训练的全身跟踪器，逆运动学规划器，Active Perception（Depth + 2-DOF 头部）以及教师策略脚本。

**📊 数据集**

使用五个基准任务（抓取、搬运、导航、开门等）收集教师轨迹，SIMPLE 与 HumanoidArena 两个 humanoid 评测基准，以及真实 G1 机器人场景。

**📈 对比分析**

方法通过与基准任务的对比评估：在 SIMPLE 上基于基准任务的微调模型无特定轨迹即可 51% 成功，加入 5 条轨迹提升至 95%；在 HumanoidArena 仅基于基准任务成功率 42%；与传统跟踪器相比，HEFT+HITCH 轨迹误差降至 0.6 cm 手误，成功率 97.9%；真实机器人上成功率 65%。

**⚠️ 局限性**

局限性包括：规划器不支持坐姿、跨越障碍等动作；在 HumanoidArena 中加入 SIMPLE 轨迹会降低成功率；推理会因错误读取导致失败；每条指令完成后才观察，无法实时调整；真实机器人手部缺少指尖控制。

---

## 483. Probabilistic Relevance Decisions for Semantic Retrieval

**arXiv ID:** 2611.08014 | [PDF](https://arxiv.org/pdf/2611.08014v1)

**作者:** Bahadır Tatar `[一作]` (Recruspace LLC), Cevat Şener `[通讯]` (Recruspace LLC)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比了 Dense 向量检索与两种云端决策接口（Jev Noul 与 OpenAI Predicate）在 BEIR SciFact 测试集上的检索和再排序效果。

**💡 创新点**

首次在统一实验协议下对不同决策接口的原始概率分数、零/一饱和、精确分数平局以及成本/延迟进行系统评估，并探讨候选深度（Top‑50 与 Top‑100）对最终指标的影响。

**🔧 技术方法**

使用 BGE‑M3 作为 Dense 检索器、Jev 与 OpenAI API 作为单独的概率相关性判定器，配合标准的 Recall@k 与 nDCG@k 评估，并利用配对 Bootstrap 进行统计置信区间估计。

**📊 数据集**

采用 BEIR SciFact 官方测试集（300 条查询、≈30k 文档、约1.3k 正样本），保持原始 qrels 和文档内容不变。

**📈 对比分析**

在控制池（保证所有正样本）和全语料的 Top‑50/Top‑100 两种候选深度下比较三种方法，发现 Jev 在 Recall@5 与 nDCG@10 上普遍优于 OpenAI 与 Dense；OpenAI 与 Dense 的 nDCG@10 较 Dense 有提升但 Recall@5 的提升不显著；候选深度提升仅带来微小或不确定的性能变化。

**⚠️ 局限性**

局限性包括：只评估单一 Dense 检索模型与单一数据集；未检验跨域或多语言适用性；OpenAI 结果受零分饱和与精确分数平局的影响；成本与延迟评估受缓存、网络与本地硬件差异影响，未给出统一冷启动基准；统计结果为探索性，未进行多重比较校正。

---

## 484. DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels

**arXiv ID:** 2611.08082 | [PDF](https://arxiv.org/pdf/2611.08082v1)

**作者:** Jinho Park `[一作]` (Columbia University), Mingoo Seok `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于字典的后训练压缩方法 DictLLM，将大型语言模型的权重拆解为共享字典和层级稀疏系数，实现高效压缩。

**💡 创新点**

结合激活感知的字典因子化、字典感知 Hessian 剪枝、字典重拟合以及 NVFP4 量化四步，首次在后训练中实现共享字典与稀疏系数的协同压缩，并保持高性能。

**🔧 技术方法**

采用激活感知 SVD、字典感知 Hessian 重要性评估、稀疏系数剪枝、字典重拟合、Hessian 量化的 NVFP4，以及 LoRA 微调等技术。

**📊 数据集**

使用 WikiText‑2、Penn TreeBank、C4 等语言建模数据集，以及 OpenBookQA、WinoGrande、HellaSwag、PIQA、MathQA、ARC‑Easy、ARC‑Challenge 等零样本推理基准。

**📈 对比分析**

与 ASVD、SVD‑LLM、Basis Sharing 等后训练低秩压缩基线在相同压缩比例下对比，DictLLM 在 LLaMA‑7B 上实现 5.1× 压缩仅 2.9 点 perplexity 增加，压缩率比现有低秩方法高 3.6×，并在零样本推理上保持或提升准确率。

**⚠️ 局限性**

受限于字典尺寸和系数稀疏度的手动设定，对极端压缩比例仍会出现显著性能下降，并且方法需要大量校准样本来估计激活协方差。

---

## 485. Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model

**arXiv ID:** 2611.08111 | [PDF](https://arxiv.org/pdf/2611.08111v1)

**作者:** Zihao Zhang `[一作]` (Southeast University), Ziyuan Pu `[通讯]` (Southeast University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在统一的世界-行动模型上做安全后训练，利用近专家负轨迹构造负向约束，提升模型在安全关键场景的表现

**💡 创新点**

创新点包括：①将近专家负轨迹作为距离自适应的repulsive anchors和场景条件化的rider tokens，在不增加推理开销的前提下实现安全边界学习；②联合优化原始流匹配目标与两种负样本监督，既约束动作生成，又强化场景-轨迹兼容性

**🔧 技术方法**

使用流匹配（flow matching）训练的统一世界-行动Transformer，离线负样本挖掘与距离自适应惩罚、场景兼容性评分头、预训练视频生成模型Wan2.2、Diffusion Transformer DiT等技术

**📊 数据集**

主要使用NAVSIM v2（navtrain、navtest、navhard）进行训练与评估，并在HUGSIM上进行零射程闭环测试

**📈 对比分析**

与DriveWAM、BeyondDrive等无轨迹评估的WAM基线对比；在navhard上取得40.4 EPDMS（比最佳基线提升4.1%），navtest 93.3 EPDMS；在HUGSIM零射程得到RC 45.1、HD-Score 30.6，超过DriveWAM；推理延迟提升仅0.75%，基本不影响部署成本

**⚠️ 局限性**

局限性：需要离线负样本挖掘且对模拟环境的依赖较强；在极端情景性能下降；泛化到真实多样化交通场景仍待验证

---

## 486. An Integrated Open-Source Stack for Vision-Based Autonomous Navigation of Wheeled Rovers

**arXiv ID:** 2611.08100 | [PDF](https://arxiv.org/pdf/2611.08100v1)

**作者:** Iaroslav Kolomiets `[一作]` (MSU Rover Team), Isaac Chairez `[通讯]` (Tecnologico de Monterrey)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发并公开了完整的 ROS2 栈，实现了基于视觉的行驶车自主导航，包括箭头和圆锥检测、SLAM、本地规划、行为树管理以及 STM32 控制。

**💡 创新点**

创新点在于将 YOLOv8n 目标检测与几何方向估计、RGB‑D SLAM、Smac 混合规划、行为树任务管理以及硬件驱动集成为一个可复用的、开源的堆栈，并公开了数据集与权重。

**🔧 技术方法**

采用的技术包括 YOLOv8n TensorRT INT8、RTAB‑Map + EKF、Nav2 Smac Planner、BehaviorTree.CPP、比例（P）路径跟随插件、独立转向驱动、Jetson Orin NX 与 Beelink PC 组合。

**📊 数据集**

使用公开的 MSU Rover Team CV 数据集，包含 6,147 张图像、20,303 个实例（5,145 个箭头、15,158 个圆锥）。

**📈 对比分析**

与手工 OpenCV 基线在同一 478 张测试图像上对比，YOLOv8n 的 AP@0.5 约为 0.68（基线为 0.23），mAP@0.5 在测试集为 0.720；推理延迟 22–25 ms，三次现场任务均无碰撞成功完成。

**⚠️ 局限性**

局限性包括：仅完成三次任务、未量化真实定位误差和全距离感知精度、缺乏控制器性能比较、搜索旋转时占据面积未检查、未在其他平台验证可移植性。

---

## 487. ATHAAR: A Medieval Scientific Arabic Manuscript Dataset for Handwritten Text Recognition

**arXiv ID:** 2611.08091 | [PDF](https://arxiv.org/pdf/2611.08091v1)

**作者:** Ismail El Jamiy `[一作]` (Université de Toulouse), Nabil Hathout `[通讯]` (Université de Toulouse Jean-Jaurès)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并公开了首个针对中世纪阿拉伯科学手稿的手工注释数据集Athaar，包含多见证、跨书体和跨章节的文本；

**💡 创新点**

提出了基于文本批评的注释指南，首次为手稿文本、数字与几何对象制定统一的标注规范，并对注释者间一致性进行量化；

**🔧 技术方法**

使用基于视觉语言模型的手写文本识别技术（Gemini 3 Flash、CHURRO‑3B、Qari‑OCR‑2B）进行零样本与少样本实验；

**📊 数据集**

Athaar数据集共3,893行，涵盖来自11至19世纪的多位作者的天文数学手稿，包括《Qānūn al‑Mas‘ūdī》及其他科学论著；

**📈 对比分析**

通过三种跨写体/章节/论著的迁移实验评估模型表现，结果显示通用VLM Gemini 3 Flash在所有场景下实现16.3%–21.7%字符错误率，优于专门化模型；

**⚠️ 局限性**

局限性在于数据集仅覆盖单一时期作者、未进行模型微调与更细粒度的提示策略实验，且对其他阿拉伯手稿传统的泛化能力有限。

---

## 488. Optimal Competitive Ratios for Chasing Convex Functions with Quadratic Growth

**arXiv ID:** 2611.08078 | [PDF](https://arxiv.org/pdf/2611.08078v1)

**作者:** Linzhe He `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在二维欧氏和平方欧氏运动成本下，满足二次增长条件的凸函数追踪问题，提出了Clipped Proximal Descent (CPD) 算法，并对 Regularized Online Balanced Descent (R-OBD) 进行了紧凑分析，给出了这两种运动下的最优维度无关竞争比；

**💡 创新点**

创新点在于：①用二次增长代替强凸性实现更弱假设下的最优竞争比；②提出 CPD 通过平衡的近端步和裁剪校正实现 O(κ^{1/3}) 竞争比，填补了现有 O(√κ) 与 Ω(κ^{1/3}) 之间的多项式鸿沟；③在平方欧氏运动下给出精确的最优竞争比 1+q^*(λ)，并给出匹配的随机化下界；

**🔧 技术方法**

核心技术包括：潜能法、平衡近端优化、裁剪校正、梯度下界与光滑性结合、正则化在线平衡下降的优雅潜能分析以及构造对抗输入实现下界；

**📊 数据集**

本文无实验数据集，全部工作基于理论分析与证明；

**📈 对比分析**

通过竞争比的形式与先前最优 O(√κ) 或 1+2/√λ 等上界进行比较，证明所提出的 CPD 和 R-OBD 能取得更优的竞争比，且匹配下界，表明其在理论上已达到最优；

**⚠️ 局限性**

局限性包括：仅在满足二次增长且 Euclidean 运动需光滑性、平方 Euclidean 运动不需光滑性；未讨论非凸或更一般的运动成本；并且理论结果仅适用于维度无关算法，实际实现仍需进一步验证。

---

## 489. Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring "Weight-Space Programming"

**arXiv ID:** 2611.08070 | [PDF](https://arxiv.org/pdf/2611.08070v1)

**作者:** Nazmus Sakib Ahmed `[一作]` `[通讯]` (BRAC University), Nazmus Sakib Ahmed (BRAC University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估Claude Opus 5.5在无工具、无前向推理条件下，对125k参数CNN权重进行零射读取，探究其是否能理解网络训练任务。

**💡 创新点**

创新点在于将阅读能力拆解为局部算术与数据集先验的组合，并设计标签置换、随机权重等控制，提供可复现的审计协议。

**🔧 技术方法**

技术手段包括纯文本权重序列输入、批归一化统计与输出层符号模式推断、以及自定义评分规则。

**📊 数据集**

使用的数据集包括CIFAR-10、KMNIST、Cats vs Dogs、Fast‑Food、SAT‑6 等，覆盖熟悉与陌生任务。

**📈 对比分析**

比较方法为先通过标签置换检验模型是否跟随输出权重结构，随后评估其对未标注数据集的推断准确度；结果显示模型局部算术准确，语义推断受数据集熟悉度影响，误判率高。

**⚠️ 局限性**

限制在于仅测试单一模型单次随机种子、缺少前向推理验证、控制设置有限、结果受Token限制与自我报告置信度影响。

---

## 490. Secure Formal Raising: Protecting the Development of AI through Executable Properties

**arXiv ID:** 2611.08042 | [PDF](https://arxiv.org/pdf/2611.08042v1)

**作者:** Rolf Drechsler `[一作]` (University of Bremen and DFKI), Christina Plump `[通讯]` (University of Bremen and DFKI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出 Secure Formal Raising (SFR) 模型，使用四值证据在强化学习环境中实现对政策更新、工具调用和数据流的安全约束，并给出了可执行的 C 参考实现与验证。

**💡 创新点**

创新点在于将四值逻辑与 Formal Raising 结合，定义六个安全不变量，证明仅确认证据才能通过；同时揭示并通过 provenance 约束消除了混淆代理（confused‑deputy）漏洞。

**🔧 技术方法**

使用四值（Belnap）逻辑、双格（bilattice）属性判定、状态机安全自检、优先级排序、以及广度优先搜索的模型检查技术；实现采用纯 C 代码并利用常数时间评估。

**📊 数据集**

未使用公开数据集；通过枚举 1,536 个候选场景、执行 6,262 个可达状态以及 72,329,329 条转移进行内部验证。

**📈 对比分析**

通过有限状态空间探索与枚举实验检验不变量，所有测试均通过；实现空间与时间复杂度均为常数级，运行效率高且无安全违规。

**⚠️ 局限性**

局限性包括：仅在抽象模型上验证，未覆盖实际身份服务、语言模型或网络工具；需要外部适配器提供可信四值证据；未考虑并发、多许可、复杂标签流等扩展场景。

---

## 491. Should We Pay This Much for Robustness? Efficient Proxy Certificates with Marginal Guarantees

**arXiv ID:** 2611.08035 | [PDF](https://arxiv.org/pdf/2611.08035v1)

**作者:** Soroush H. Zargarbashi `[一作]` (CISPA Helmholtz Center for Information Security), Aleksandar Bojchevski `[通讯]` (University of Cologne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种代理证书框架，通过对高成本的随机平滑证书进行校准，用低成本代理函数实现鲁棒性验证；

**💡 创新点**

创新点在于将 conformal 风险控制与随机平滑相结合，实现对代理误报率的边缘期望控制，从而在不完整的点wise 可靠性的前提下获得低成本的鲁棒性保证；

**🔧 技术方法**

主要技术包括随机平滑（Gaussian噪声下的模型无关下界上界）、Conformal 风险控制（CRC）、以及阈值化的代理函数；

**📊 数据集**

在图像任务上使用 ImageNet 与 CIFAR‑10，在文本任务上使用 AG‑News（RoBERTa‑base）进行实验；

**📈 对比分析**

与传统的随机平滑（需 2000+ MC 采样）和确定性验证器相比，代理证书在单次前向传播下实现了 39× 的 end‑to‑end 速度提升，同时保持了接近原始证书的 Certified Ratio 与 Certified Accuracy；

**⚠️ 局限性**

局限性包括仅提供边缘（marginal）误报率保证，不能对单个样本或类别做条件控制；对分布漂移敏感；以及在严格的点wise 可靠性需求场景下不适用。

---

## 492. GTAFL: Addressing Test-Agnostic Long-Tailed Federated Learning in Medical Image

**arXiv ID:** 2611.08021 | [PDF](https://arxiv.org/pdf/2611.08021v1)

**作者:** Guangyu Chen `[一作]` (Northeastern University), Hui Jiang `[通讯]` (Huizhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了测试无关的长尾联邦学习任务，并设计了 GTAFL 框架，利用全局信息引导采样、专家分类器重训练、服务器端自监督学习以及专家集成，实现对未知测试分布的鲁棒性提升。

**💡 创新点**

创新点包括：①首次定义测试无关长尾联邦学习；②使用全局信息指导采样并生成全局联邦特征（GFF）；③在服务器端基于 GFF 训练多种专家分类器并在推理时投票集成；④将 SimCLR 自监督对比学习引入联邦特征提升表示泛化能力。

**🔧 技术方法**

采用的技术主要有：联邦学习 FedAvg、全局信息引导采样（GI‑GDS）、专家分类器重训练（类似 CReFF）、全局联邦特征（GFF）生成与匹配、SimCLR 对比学习、硬投票集成。

**📊 数据集**

实验数据集包括：CIFAR10‑LT（自然图像），PathMNIST‑LT（结肠病理图像），OCTMNIST‑LT（视网膜 OCT 图像）。

**📈 对比分析**

与 FedAvg、FedProx、BalanceFL、CReFF、FedGraB、FedLoGe 等基线比较，GTAFL 在前向、均匀、后向多种长尾测试分布上均表现优于或相近；在后向分布上平均提升约 8.5%，在医疗数据上平均提升 2‑3%。

**⚠️ 局限性**

局限性包括：对极端后向分布下头类准确率略低；服务器端自监督与专家重训练导致额外计算与通信开销；需要手动调参 λ 控制 GFF 分布，易受不当设置影响；在极端数据稀缺或分布极端偏差时性能下降；未给出严格的隐私保护证明。

---

## 493. Route-DMD: Distribution Matching Distillation with Task-Routed Mechanism for Speech Foundation Models

**arXiv ID:** 2611.08011 | [PDF](https://arxiv.org/pdf/2611.08011v1)

**作者:** Yanru Huo `[一作]`, Zilong Zheng `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为Route‑DMD的两阶段蒸馏框架，将语音基础模型从32步压缩到4步，同时保持生成、编辑和处理等多任务性能。

**💡 创新点**

创新点在于：①对学生进行一致性初始化以加速收敛；②对分离任务采用回归损失路由，避免分离时教师分数不可靠；③用自适应投影引导（APG）替代CFG，消除指导误差；④结合分布匹配蒸馏与回归路由，实现任务感知的蒸馏。

**🔧 技术方法**

核心技术包括：条件流匹配、解耦分布匹配蒸馏（Decoupled DMD）、一致性蒸馏、回归路由机制、以及自适应投影引导（APG）和无条件/条件分数网络。

**📊 数据集**

在AuK大模型基础上使用的多任务数据集包括SeedTTS‑Eval、InstructTTSEval、MMAE、CHiME‑4、Libri2Mix和VCTK‑SR，涵盖文本转语音、指令式TTS、编辑、降噪、分离和超分辨率等。

**📈 对比分析**

实验对比显示：4步学生在生成与编辑任务上与32步教师相当，且在所有处理任务的DNSMOS OVRL和UTMOS指标上更优；实时因子（RTF）从0.4333提升至0.0958，实现约4.5×的端到端加速。

**⚠️ 局限性**

局限性：对分离任务的回归路由虽能恢复性能，但仍依赖教师在噪声状态下的可靠分数；模型在更大规模或未见任务上的泛化能力未完全验证；同时，蒸馏过程仍需大量教师推理开销。

---

## 494. National-scale mapping of tree species composition in Switzerland from high-resolution aerial imagery and multimodal data

**arXiv ID:** 2611.08095 | [PDF](https://arxiv.org/pdf/2611.08095v1)

**作者:** Yixin Zhou `[一作]` (ETH Zurich), Mirela Beloiu `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

利用瑞士全国森林调查（NFI）数据与多模态遥感（0.1 m RGB航空影像、ALS点云、AlphaEarth卫星嵌入、气候变量）构建了30 m分辨率的树种基面积比例地图，覆盖10种主要树种与残余类别。

**💡 创新点**

①首次在全国尺度提供连续的树种基面积比例，而非仅标注主导树种；②将自监督视觉Transformer与ALS、卫星嵌入、气候特征通过门控融合，实现多源信息的有效利用；③在不同空间尺度（像素、林区、地区）下系统验证，展示了混交林中次生树种的重要性。

**🔧 技术方法**

多模态深度学习框架：RGB使用冻结的DINOv3 ViT‑H+/16编码器，ALS使用随机初始化的稀疏3D U‑Net，卫星嵌入与气候通过MLP编码；所有特征通过门控注意力融合成站点表示；网络输出包括树种比例、林分纯度、叶类比例，并采用KL+L1损失与一致性约束进行训练。

**📊 数据集**

主要数据集：瑞士全国森林调查（3992个样点，2017‑2022年），瑞士森林保护区监测（132个林区样本）用于独立验证；SWISSIMAGE 0.1 m RGB orthomosaic；swissSURFACE3D ALS点云；AlphaEarth annual embeddings（10 m）；CHELSA BIOCLIM+气候变量（~1 km）。

**📈 对比分析**

与基线（随机森林、ResNet‑50、AlphaEarth MLP）以及不同ViT预训练版本进行比较。最优模型在5折空间交叉验证中宏观R²=0.485，像素级独立评估R²=0.552，林区级R²=0.733；主导类F1≈0.585。对比NFI4地区份额时R²=0.953，平均绝对误差≈1.7个百分点；剔除单一模态后，AlphaEarth与气候对R²贡献最大，ALS贡献较小。

**⚠️ 局限性**

局限性：ALS点云密度有限且与RGB影像时间间隔较大，导致对相似色调的针叶树种识别混淆；模型未充分利用垂直结构信息，可能限制在高度梯度显著地区的精度；NFI样本采用的是分区样点，无法完全匹配30 m像素的空间细节，限制了地图的最小分辨率；未覆盖极端稀疏或裸露地面区域。

---

## 495. PersonaAgent: Retrieval-Augmented Transaction Anomaly Scoring with Refreshable Multi-Entity Persona Memory

**arXiv ID:** 2611.08094 | [PDF](https://arxiv.org/pdf/2611.08094v1)

**作者:** Zhengyu Fang `[一作]` (Case Western Reserve University), Yiwei Cai `[通讯]` (Visa Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Visa大规模支付网络中，通过构建可刷新、多实体语义Persona记忆，将实体行为上下文实时注入冻结的异常检测模型，以提升交易异常检测精度。

**💡 创新点**

创新点在于：① 采用LLM在设计时生成和演化Persona协议，构建语义主题并聚合稀疏实体证据；② 在运行时通过轻量级适配器实现检索增强评分，完全不触发LLM调用；③ 将多实体记忆与多检测器共享，解决知识滞后问题。

**🔧 技术方法**

使用技术包括：LLM代理自演化（提议/批评）、语义主题建模、检索增强（键值检索+回退）、树形编码器+小型MLP融合、批处理式记忆刷新、版本化发布控制。

**📊 数据集**

使用数据集：Visa交易日志，覆盖约4.9B凭证、175M商户、329B交易，实验分为离线（Period 1）和线上部署（Period 2）。

**📈 对比分析**

通过与四个冻结检测器在离线和上线的PR‑AUC、AUC和业务指标BM1比较，PersonaAgent分别提升PR‑AUC 0.65–6.58%、AUC 0.23–0.66%及BM1 8.93–14.24%，并在所有组合上显著提升。

**⚠️ 局限性**

局限性包括：对主题映射的敏感性导致记录老化影响性能、需在设计时依赖LLM生成协议、可能对敏感属性产生偏差，且对极少见实体的效果依赖回退机制。

---

## 496. Linguistic boundaries emerge from pressure-constrained processing dynamics

**arXiv ID:** 2611.08085 | [PDF](https://arxiv.org/pdf/2611.08085v1)

**作者:** Rui He `[一作]` (Universitat Pompeu Fabra), Wolfram Hinzen `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过一个基于依赖局部理论的抽象生成模型、听力划分实验、自然故事自测阅读和大型语言模型分析，探讨语言边界如何由处理压力动态产生。

**💡 创新点**

创新点在于将记忆、整合与语义压力的交互视为边界形成的驱动机制，并揭示多时间尺度处理动态与句子分割的对应关系。

**🔧 技术方法**

采用的技术包括依赖局部理论（DLT）、fastText 词向量、Qwen3‑4B 大型语言模型及其内部隐藏状态的几何度量（如曲率、局部维度、路径长度/对角线比）。

**📊 数据集**

使用的数据集为公开的语音划分实验、Natural Stories 自测阅读语料以及相同文本的写作语料。

**📈 对比分析**

通过与仅靠结构完成的控制模型、人类边界标记、阅读时长以及LLM内部几何的比较，显示压力驱动模型生成更符合自然句长分布、边界预测更准确，并使LLM在句子结尾处产生更明显的内部重组；与传统模型相比，性能更贴近人类处理动态。

**⚠️ 局限性**

局限在于模型结构简化、缺乏因果验证、仅使用英语数据、未检验不同语言和模型架构的普适性。

---

## 497. Winding-Pack Quench Simulation of Stellarator Coils: Hydraulic Field-Network Coupling and Transverse Heat Conduction in OPENSC$^2$

**arXiv ID:** 2611.08084 | [PDF](https://arxiv.org/pdf/2611.08084v1)

**作者:** Laura A. M. D'Angelo `[一作]` (Max Planck Institute for Plasma Physics), Felix Warmer `[通讯]` (Max Planck Institute for Plasma Physics)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

扩展了OPENSC2开源代码，在磁流体（CICC）绕组包中加入横向热传导、三维磁场依赖的电流共享、感应损耗及与外部水力网络耦合，实现对星型托卡马克磁体热电液耦合的完整模拟。

**💡 创新点**

首次在开源框架下实现了水力场网络耦合以及对绕组包内非局域横向热传递的显式建模，显著提升了对二次故障和流动重分布的预测能力。

**🔧 技术方法**

采用一维电热液耦合有限元离散，利用Bishop框架和Landreman–Hurwitz Biot‑Savart方法获得三维磁场，加入感应耦合、磁滞和涡流损耗模型；水力网络采用基于质量流率的改进传输线模型，并通过改良的节点分析（MNA）与场模型进行单片耦合，求解使用θ‑scheme与Schur补方法。

**📊 数据集**

使用W7‑X非平面AAB‑51绕组在2006年CEA实验的测量数据，包括入射/出口温度、电压、压力、流量以及磁场分布等。

**📈 对比分析**

通过与实验测量的热电信号、热点温度、探测电压及出口温度进行对比，扩展模型将出口温度误差由55 K降至12 K，成功预测了二次故障；但计算时间从原先约20 分钟增至2–3 小时。

**⚠️ 局限性**

对外部水力电路、传感器位置的完整信息缺失、对三维曲率效应的近似、以及对长时间出口温度的预测仍存在差距，导致模型与实验在长期行为上不完全一致。

---

## 498. Quantum Advantage in Local Certification of Graph Properties

**arXiv ID:** 2611.08076 | [PDF](https://arxiv.org/pdf/2611.08076v1)

**作者:** Benjamín Jáuregui `[一作]` (University of Chile), Masayuki Miyamoto `[通讯]` (University of Tsukuba)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在分布式量子本地认证框架下，证明了对仅依赖图拓扑的图性质可以获得指数级和多项式级的量子优势，具体构造了两类性质 _μ 和图对称性（Symmetry），并给出了相应的一轮量子协议与经典下的下界。

**💡 创新点**

创新点在于：①首次将分布式等价问题的量子优势推广到无输入的图性质；②构造了特殊的树形编码 _μ，利用量子等价协议实现 O(μ log n) 的量子证明，而经典协议至少需要 Ω(n/(μ log² n)) 位；③对图对称性问题给出了 O(n^{5/3}) 的量子证明，并在半径受限图中进一步压缩至 O(nD log n)；④改进了路径上等价问题的量子协议，将证明大小从 O(r² log N) 降至 O(r log N) 并获得 O(N^{2/3}) 的无距离依赖量子协议。

**🔧 技术方法**

主要技术包括：量子指纹（fingerprint）与 SWAP 测试的改进分析；基于标签树（Tag(i)）的树形编码与局部可验证性；使用局部哈密顿量与 spectral gap 结合 Detectability Lemma 进行误差分析；以及哈密顿量的 Knabe 绑定和三块/两块块级别的能隙估计。

**📊 数据集**

无实测数据集，全部为理论构造与数学证明。

**📈 对比分析**

通过对量子协议与经典协议在证明大小（certificate size）和消息大小（message size）的比较，展示了：①对 _μ，量子协议需 O(μ log n) 量子位，经典需 Ω(n/(μ log² n)) 位；②对 Symmetry，量子需 O(n^{5/3}) 量子位，经典需 Ω(n²) 位；③对路径等价问题，量子 O(r log N) 或 O(N^{2/3})，经典至少需要 Ω(log N) 位。实验与模拟表明量子方案在理论上可实现显著压缩。

**⚠️ 局限性**

限制与不足：①只对特定构造的图性质给出优势，难以推广至更一般的图属性；②量子协议需要在证明阶段提供完整的量子态，实际实现受限；③经典下界多为信息论或结构性计数证明，可能不适用于所有图；④对图对称性的优势在半径受限图中才显著，且对 μ 的取值存在约束。

---

## 499. Leave It Better Than Found: Value-Guided Cooperation in Sequential Tasks

**arXiv ID:** 2611.08075 | [PDF](https://arxiv.org/pdf/2611.08075v1)

**作者:** Mario Gomez Andreu `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 PAVE 方法，使机器人在不进行联合训练的前提下，通过共享价值函数实现顺序任务协作。

**💡 创新点**

创新点包括：①将价值函数视为通用通信介质；②引入 KNN‑based 不确定性滤波器防止离群状态误导；③使用在线递归规划（CEM + 价值函数）在有限时间窗内实现自适应策略。

**🔧 技术方法**

使用强化学习（PPO）训练单机策略、价值函数聚合、交叉熵法（CEM）进行规划、KNN 近邻法做不确定性估计。

**📊 数据集**

在多种仿真环境（Warehouse、Mobile Arm、Obstacle Arena、Warehouse Pickup Blocked/Unseen）以及真实硬件（RoboMaster 家庭场景）进行评估。

**📈 对比分析**

与 Naive、PAVE（无 OOD）、PAVE（无 prior）和 Leader‑only 等基线对比，PAVE 在所有场景下显著提升追随者成功率（最高可提升约 50%），并在多数场景中逼近或超过经过微调的专用策略。

**⚠️ 局限性**

局限性：仅适用于固定顺序的两机器人序列；规划时限短导致需用预测状态近似交接；假设两机器人共享同一环境表示；价值函数共享假设可信，若存在恶意或错误信息则可能导致灾难性失败。

---

## 500. Terrain-Indexed Motion Priors and Geometric Support for Extreme Humanoid Parkour

**arXiv ID:** 2611.08056 | [PDF](https://arxiv.org/pdf/2611.08056v1)

**作者:** Chenghao Sun `[一作]` (Noetix Robotics), Wenhao Cui `[通讯]` (Noetix Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种渐进式模仿学习框架，先通过人类演示生成可执行的机器人轨迹，再利用基于地形的对抗运动先验学习多种障碍跨越技能，随后通过几何手膝支撑引导降低运动先验权重，使机器人能够自适应不同地形下的支撑方式。

**💡 创新点**

创新点包括：①将演示迁移为物理可执行的参考；②使用地形索引的多头对抗运动先验（AMP）实现对不同障碍高度的技能分离；③引入几何手膝支撑候选与实际接触奖励，实现无序列化的支撑适配；④在训练后期逐步降低AMP权重、增强支撑奖励，保持技能效果的同时允许机器人自由调整接触点。

**🔧 技术方法**

核心技术有：对抗运动先验（AMP）与多技能判别器；深度感知与姿态历史编码（History Estimator）；PPO策略优化；几何支撑候选生成与吸引/实际接触奖励；基于关节、接触、扭矩、扫描的高维运动特征；地形标签与训练时的专属信息。

**📊 数据集**

使用人类演示数据（共约3,500帧，包括站立、转向、侧向、后退、前进以及三种高度的障碍演示）作为参考；训练时对不同高度的障碍进行地形标签化；硬件测试使用Noetix E1机器人与其深度相机；仿真环境为Isaac Lab（Sim）和Gazebo（真实对比）。

**📈 对比分析**

与三种基线（Velocity Tracking、Single AMP、Multi‑Skill Distillation）以及消除阶段的消融模型进行比较。仿真中：单障碍成功率最高可达100%，混合路线成功率96.34%，OOD成功率93.46%；硬件零射击测试：单障碍29/30成功，混合4/5成功，OOD13/15成功。相较基线，本文方法在连续多障碍跨越和地形外推上表现显著优异。

**⚠️ 局限性**

局限性包括：①支撑维护仍有失败（如Mid障碍时失去手臂支撑）；②对极端地形或非规则支撑面适配能力有限；③需要更多多样化演示以覆盖更广泛的障碍类型；④当前几何支撑候选生成仅针对规则平面，难以处理不规则表面；⑤对真实世界动力学差异的补偿仍依赖仿真训练。

---

## 501. Precision - Preserving Subtensor Inference for a 22B Joint Audio - Video Generator under a Fixed GPU Working Set

**arXiv ID:** 2611.08023 | [PDF](https://arxiv.org/pdf/2611.08023v1)

**作者:** Hanyue Shen `[一作]` (YH Intelligence Technology, Co., Ltd.), Xinling Liao `[通讯]` (YH Intelligence Technology, Co., Ltd.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 GPU 内存受限的环境下，设计并实现了一种固定 arena 运行时 PSI（Precision-Preserving Subtensor Inference），通过将权重切片加载到可重用的 GPU 页面、将中间状态保存到索引化的主机 journal，并结合权重预取、FFN 重用及异步状态传输，完成了 22B BF16 语音–视频联合生成 Transformer LTX-2.5 的完整推理。

**💡 创新点**

① 通过固定 arena 与可重用权重页实现重量与状态的彻底分离；② 使用子张量评估保证实数算术一致性；③ 通过可控数值改动（bias 加法、attention 归一化改写）提升数值稳定性；④ 在 3 GiB 可用内存下实现完整生成，并与 FLASH 方案对比。

**🔧 技术方法**

GPU 权重流式加载、权重预取、异步主机日志传输、FFN 重用、BF16 运算、固定 arena 内存管理、事件排序与依赖、实数算术分区、CUDA 与 PyTorch 库结合、Spark DGX 服务部署。

**📊 数据集**

22B BF16 LTX-2.5 精炼模型；三类提示（发言人、厨师、航空公司）+ 3 个 64 位种子，产生 27 条完整推理轨迹；Spark GB10 上的 30 条请求；RTX A6000 与 DGX Spark 进行硬件评测。

**📈 对比分析**

使用官方 MATH、baseline PSI 与 modified PSI 进行对比；衡量指标包括 GPU 进程峰值内存、wall‑time、终端 L₂ 误差；9 条完整提示中，modified PSI 使 transformer 进程峰值由 8 578 MiB 降至 360 MiB（95.8% 内存减幅），但平均耗时从 247 s 增至 349 s（+41%）；在固定 3 GiB 内存下，native PSI 610 s、FLASH 460 s；所有 27/27 与 30/30 输出均通过完整视频/音频解码；Spark 通过率 6.86 输出/GPU‑h。

**⚠️ 局限性**

① 运行时间显著增长（尤其 modified PSI +40%）；② 数值差异导致平均误差上升；③ 仅支持 BF16，未验证 FP8/FP32；④ 需要手动调优页/工作区大小、对主机存储依赖；⑤ 仅在 RTX A6000 与 Spark DGX 上验证，其他 GPU 兼容性未知；⑥ 依赖完整模型切片与 host journal 的实现；⑦ 仅针对 LTX‑2.5 22B，未评估其他规模或架构。

---

## 502. A Unified Fractal Impedance Control Framework

**arXiv ID:** 2611.08009 | [PDF](https://arxiv.org/pdf/2611.08009v1)

**作者:** Adnan Takriti `[一作]` (University of Sussex), Carlo Tiseo `[通讯]` (University of Sussex)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发并验证了一套统一的 Fractal Impedance Control (FIC) 框架，重点实现了 Harmonic-Passive Motion Paradigm (H-PMP)，并在 Kinova Gen3 7-DOF 机器人上完成轨迹跟踪、墙压释放和学习演示等实验。

**💡 创新点**

创新点在于：①将任务空间与关节空间的 admittance‑impedance 交互无缝嵌套，形成无状态机、自适应的 H-PMP；②首次完整将 H-PMP 引入机器人，实现了全局可视化与实时参数调节的 FIC Studio；③提供了一个开源的 FIC Suite，统一支持 MuJoCo 仿真、Pinocchio 逆运动学和 ROS2 通信。

**🔧 技术方法**

使用技术包括：Fractal Impedance Control（基于 Liénard 方程的非线性阻尼）、H-PMP 架构、任务空间 admittance/Impedance、关节空间 HapFIC、MuJoCo、Pinocchio、ROS2、FIC Studio GUI、Python 现场脚本和 STM32 MCU（低功耗实现）。

**📊 数据集**

实验数据来源为在 Kinova Gen3 机器人上收集的轨迹跟踪误差、墙压传感器估计力、学习演示的重放误差及扰动响应。未使用公开数据集，全部为实验室自制数据。

**📈 对比分析**

对比方法：将 H-PMP 与三种基线 FIC 架构（Cartesian Impedance+Planner、Task‑Space HapFIC、Cartesian Impedance+Joint‑Space HapFIC）在同一机器人、同一实验设置下进行。结果显示：H-PMP 在轨迹 RMSE、峰值误差、力追踪、扰动抑制等指标均优于或相当于基线，尤其在墙压释放时实现了平稳的 admittance‑impedance 过渡。

**⚠️ 局限性**

局限性：①H-PMP 轨迹相对较为振荡，需更精确的动态模型或在线参数自适应；②实时在线调参与自适应学习功能尚未实现；③实验仅覆盖单一 7-DOF 机器人，缺乏对更复杂、多机器人环境的验证；④硬件实现对 MCU 的依赖和 Pinocchio 计算需求限制了在低功耗嵌入式平台上的完整部署。

---

## 503. When GeoFMs Meet VHR: Benchmarking Foundation Models on Very High Resolution Earth Observation Imagery

**arXiv ID:** 2611.08003 | [PDF](https://arxiv.org/pdf/2611.08003v1)

**作者:** Kshitij Raj Sharma `[一作]` (Université Bretagne Sud), Sébastien Lefèvre `[通讯]` (Université Bretagne Sud)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了 VHR-Bench 这一针对 1.5 m 及更高分辨率 RGB 遥感图像的十数据集基准，评估了 12 种不同预训练策略的编码器，并在此基准上进行短期无标签持续预训练。

**💡 创新点**

创新点包括：①首次系统评估 GeoFMs 在 VHR 任务上的迁移性能；②提出 VHR-Bench 作为专门针对高分辨率遥感的统一评测框架；③证明通用自然图像预训练与高分辨率遥感预训练互补，可通过短时持续自监督训练进一步提升性能。

**🔧 技术方法**

技术手段：使用冻结 backbone + 任务特定解码器（UPerNet、Siamese UPerNet、Faster R‑CNN、linear probe），采用 DINOv3 ViT‑L 进行自监督预训练与持续训练，并比较三大预训练层级（HR‑FM、CV‑FM、MidRes‑FM）下的表现。

**📊 数据集**

数据集：十个公开 VHR 数据集（SpaceNet 2/3、xView2、FLAIR 2、Open‑Canopy‑Small、EverWatch、NZ Cattle、m‑pv4ger、PureForest、HOTOSM Buildings）以及 1.02 M 张 VHR tile 的持续预训练语料库。

**📈 对比分析**

比较方法：统一冻结 backbone、相同解码器、80 epoch 训练、3 种随机种子；在 10% 与 100% 标签两种数据量下评估。结果显示，web 预训练（DINOv3 LVD）在聚合分数上略优，SkySense 在稀缺标签时表现最好；持续预训练后聚合分数提升至 70（10% = 65.2，100% = 70.0），验证了“先通用再专化”的有效性。

**⚠️ 局限性**

局限性：仅限 RGB VHR，忽略多光谱/SAR 输入；检测任务仅包含小动物；预训练分层混合，未单因素对比；冻结 backbone 限制了对全微调性能的评估；聚合分数未考虑数据集大小，可能掩盖某些数据集的影响。

---

## 504. Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines

**arXiv ID:** 2611.08093 | [PDF](https://arxiv.org/pdf/2611.08093v1)

**作者:** Albert Saiapin `[一作]` (Delft University of Technology), Kim Batselier `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GB-CPKM 梯度提升框架，使用固定秩的 CP 核机基学习器逐步构建模型，从而实现高效的张量秩选择与训练。

**💡 创新点**

创新点在于将梯度提升用于张量网络核机，动态增加模型容量，避免传统交叉验证的多模型训练，且通过验证误差选择集成规模，显著降低训练时间和内存消耗。

**🔧 技术方法**

使用了 CP 分解张量网络核机、交替最小二乘（ALS）优化、梯度提升算法、正则化以及后续全模型 ALS 细化（GB-CPKM(H)）等技术。

**📊 数据集**

在七个真实回归基准（Energy、Concrete、Kin8nm、Naval、Precipitation、HouseElectric、Airline）以及 UCI 数据集上进行实验验证。

**📈 对比分析**

与基于交叉验证的 CP 序列秩搜索方法（CV-CPKM）进行对比，GB-CPKM 在训练时间上可快约 6 倍、峰值内存低约 2 倍，预测均方误差与 CV 方法相当，GB-CPKM(H) 在保持速度优势的同时进一步提升了预测精度。

**⚠️ 局限性**

局限在于需要手动设定基学习器秩与提升阶段数，梯度提升的顺序训练限制了并行性，混合版 GB-CPKM(H) 的内存占用会升高，且对极大规模或高维问题的适应性及分类/时序任务的泛化能力尚待进一步验证。

---

## 505. TerraMesh-Masks: Open-Vocabulary Segmentation for Earth Observation

**arXiv ID:** 2611.08081 | [PDF](https://arxiv.org/pdf/2611.08081v1)

**作者:** Benedikt Blumenstiel `[一作]` (IBM Research), Konrad Schindler `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TerraMesh-Masks 开词表卫星分割数据集并在此基础上扩展 TerraMind，构建了 EO 领域的开词表分割基线；

**💡 创新点**

创新点包括：① 超大规模（16 M 二值掩码、44 k 类别）且多模态（Sentinel‑1/2、DEM、NDVI 等）卫星分割数据；② 结合 OpenStreetMap 与 Overture 语义图层，利用 Mistral‑Large‑3 LLM 自动校验与生成标签；③ 在此数据上持续预训练，显著提升 EO 领域开词表分割性能；

**🔧 技术方法**

技术手段涵盖：多模态 Transformer encoder‑decoder（TerraMind）、CLIP 文本编码、二值掩码离散化 tokenizer、LLM（Mistral‑Large‑3）语义验证、WebDataset 训练流水线、类平衡重采样及多模态输入融合；

**📊 数据集**

使用数据集：TerraMesh‑Masks（约 9 M 10 m 轨道图像与 16 M 掩码）、TerraMesh‑Masks‑Eval（523 评测样本 200+ 类别）、并与 DINOv3.txt、SAM‑3、GeoPixel、SegEarth‑OV、RemoteSAM、DINO Soars 等公开模型进行对比；

**📈 对比分析**

评估方式：在 523 样本零样本设置下，以 IoU_pos、mIoU、F1、Precision、Recall 为指标；TerraMind‑OVS‑L 在 mIoU 上相较 SAM‑3 提升 479%、DINOv3.txt 提升 53%，相较 SegEarth‑OV 提升 13.6pp；推理时间仅 52–73 ms，显著快于 GeoPixel（≈6 s）；

**⚠️ 局限性**

局限性：词汇覆盖仅为静态地物，缺失云、洪水等动态现象；自动标注仍存在噪声；地理覆盖偏倚（欧美密集，非洲/高纬度稀疏）；评测集样本有限导致置信区间宽；多模态输入对性能提升有限，需要更深层次融合方法。

---

## 506. Cleaning or Recalling? Knowledge Dependence in LLM-based Data Cleaning [Experiment, Analysis & Benchmark]

**arXiv ID:** 2611.08106 | [PDF](https://arxiv.org/pdf/2611.08106v1)

**作者:** Jingru Yang `[一作]` (Minzu University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为CoRe（Cleaning or Recalling）的受控评估协议，用来检测大语言模型（LLM）在数据清洗任务中对实体知识、命名形式及表格记忆的依赖性。

**💡 创新点**

创新点在于：①通过将原始表格中的实体名称替换为词语或音节形式，保持错误位置不变，构造“词语替代”和“音节替代”两类对照表，以分离实体熟悉度与词形规律对修复性能的影响；②加入“新鲜记录”与“连续探测”两种额外实验，用以检验模型对新实体的泛化能力与表行记忆；③对引用表（Rayyan）的系统性转换错误进行剖析，揭示传统评估可能掩盖的修复细节。

**🔧 技术方法**

技术手段包括：基于Prompt的LLM修复（零样本、对齐样本、完整表格等）；对比分析框架（O‑W、W‑S、O‑S、F‑W、O‑F等对照）；统计重采样与Bootstrap置信区间；对目标值的写入冲突检测；以及对引用转换的字段级校验。

**📊 数据集**

使用了四个公开基准表：Hospital（美国医院）、Movies（IMDb与RottenTomatoes电影元数据）、Beers（啤酒与酿酒厂信息）以及Rayyan（生物医学文献转换）。在每个表上构造了21个受控表（包括原始、词语替代、音节替代以及新鲜记录）。

**📈 对比分析**

通过计算各模型在不同条件下的F1修复率，并与对照组（如O‑W、F‑W等）进行配对差异检验。结果显示：①在医院表中，深度学习模型对词语替代的修复准确度下降约0.72，音节替代更显著；②在电影表中，年份范围的候选值保持不变，但因实体映射缺失，DeepSeek‑V4‑Pro的年份修复率从91%骤降至18%；③商业模型在“新鲜记录”条件下通常优于原始表，表明存在一定的泛化；④连续探测未发现模型对基准行的记忆。

**⚠️ 局限性**

局限性包括：①实验仅覆盖英语、美国实体的公开数据集，未涵盖多语言或私有数据；②受限于已构造的受控表格，无法完全代表真实业务中的错误多样性；③评估聚焦于实体识别与候选值选择，忽略了更复杂的语义关联与结构一致性；④对不同模型的参数设置（温度、解码策略）差异较大，可能影响可比性。

---

## 507. Deformable Contact-Aware 3D Object Placement

**arXiv ID:** 2611.08005 | [PDF](https://arxiv.org/pdf/2611.08005v1)

**作者:** Songhe Wang `[一作]`, Charles A. Kamhoua `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DCAP框架，将语言引导的3D物体放置问题转化为“释放状态-下落至平衡”的物理过程，并构建了对应的物理仿真准备基准。

**💡 创新点**

将语义意图与物理实现严格分离，利用释放状态+刚柔耦合模拟实现自然接触与变形，并提供可解释的可行性判断与物理诊断。

**🔧 技术方法**

结合大语言模型、视觉语言模型、3D分割、材质映射、MPM（材料点方法）与刚体模拟、粒子自适应填充及落地到平衡的刚柔耦合物理仿真技术。

**📊 数据集**

构建了包含186个室内重建场景、155个物体资产、2063个提示-资产-场景三元组的物理仿真准备基准（软、混合、刚性接触以及压力分割）。

**📈 对比分析**

与FirePlace、PlaceIt3D、LayoutGPT、Holodeck等静态基线以及内部对照（火把+刚体/柔体、静态+几何捕捉）进行对比，DCAP在VLM Physics & Naturalness得分从64.0提升到85.2，人类偏好提升26.5点，平均穿透深度降至0.41 cm，残余速度降72.1%，整体性能显著优于基线。

**⚠️ 局限性**

受限于简化的各向同性材质库、重建几何的精度和缺乏真实物理验证，未来需要更复杂柔性材料的真实测试和更精细的物理模型。

---

## 508. Towards Practical 20 kbps Generative Video Compression via Adaptive Tokenization and Entropy Modeling

**arXiv ID:** 2611.08099 | [PDF](https://arxiv.org/pdf/2611.08099v1)

**作者:** Jiancheng Zhao `[一作]`, Xuelong Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了如何将预训练的视频VAE表示转化为可离散化的分词器，并提出联合率‑失真优化与分组并行熵模型，提升极低码率下的视频压缩性能。

**💡 创新点**

创新点在于：① 直接把MiniMax‑H3 VAE改造为离散分词器；② 设计局部+全局上下文的自回归熵模型并采用分组并行预测；③ 用可微率近似实现token分配与熵模型的联合训练，使编码器自适应低码率；④ 通过prefix传输实现运行时码率可调。

**🔧 技术方法**

技术手段包括：向量量化（VQ）、预训练视频VAE、分组自回归熵模型（局部与全局上下文）、可微率代价（soft‑assignment）、联合率‑失真优化、光流Warp误差与CLIP‑F评估。

**📊 数据集**

数据集：训练使用OpenVid‑HD；测试使用UVG、MCL‑JCV、HEVC Class B。

**📈 对比分析**

与传统HEVC/VVC、学习型DCVC、生成式PLVC/GLC‑Video、GNVC‑VD、GVC1D、VoRTeC等方法按LPIPS、DISTS、PSNR进行RD曲线对比，获得BD‑LPIPS 82.2%–87.5%与BD‑DISTS 78.3%–85.0%的节省，且在极低码率下保持更优的视觉质量与时序一致性。

**⚠️ 局限性**

局限性：依赖预训练VAE的表示，模型体量和解码延迟相对较高；在极低码率下仍可能出现纹理模糊；对高分辨率或快速运动场景的鲁棒性未充分验证。

---

## 509. GUARDTs: Decentralized Access Control for Replicated Data Types

**arXiv ID:** 2611.08083 | [PDF](https://arxiv.org/pdf/2611.08083v1)

**作者:** Christian Kuessner `[一作]` (Technische Universität Darmstadt), Mira Mezini `[通讯]` (Technische Universität Darmstadt)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GUARDTs，一种在分布式、Byzantine环境下实现细粒度、可动态演进、无中心化访问控制的强一致性复制数据类型；

**💡 创新点**

创新点在于通过路径基能力、哈希DAG事件图在应用层嵌入权限，支持动态委托与撤销，并且在不需要可信协调器的情况下保证CIA；

**🔧 技术方法**

采用了ARDTs、CRDT、路径权限、哈希DAG、加密哈希与签名以及因果序列等技术；

**📊 数据集**

实验使用了嵌套计数器的ARDT和示例业务旅行计划数据模型；

**📈 对比分析**

通过与两种HashDAG基线（有无签名）比较，测量创建、发送、接收及撤销开销，整体控制开销约为4%，撤销导致的重材料化成本可通过快照降低；

**⚠️ 局限性**

局限在于并发撤销会导致重材料化成本高，Byzantine节点可能泄露数据，路径描述符在多路径更新上仍需手工指定。

---

## 510. xDSL-CCPP: An alternative code generator for atmospheric modelling

**arXiv ID:** 2611.08060 | [PDF](https://arxiv.org/pdf/2611.08060v1)

**作者:** Nick Brown `[一作]` (EPCC at University of Edinburgh), John Dennis `[通讯]` (National Center for Atmospheric Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

基于xDSL和MLIR实现了CCPP（Common Community Physics Package）的DSL代码生成框架xDSL-CCPP，简化了物理模块配置、元数据管理，并提供了Python前端和自动元数据推断功能。

**💡 创新点**

创新点在于将传统基于XML/键值文件的手工元数据管理转为自动推断，利用xDSL实现高层DSL并与MLIR生态兼容，同时显著减少了70%（约15,000行 → 4,859行）的专有工具代码，并提供可插拔的Python前端，提升了可维护性和灵活性。

**🔧 技术方法**

核心技术包括：xDSL（Python实现的MLIR兼容编译器工具）、MLIR的方言与转换通道、Flang Fortran前端与IR推断、xDSL自定义的further和ftn方言、以及Fortran与LLVM后端的代码生成。

**📊 数据集**

使用了三个气象测试用例（advection、ddthost、capgen）和四个NCAR单柱模型（LASSO、GABLS3、BOMEX、COMBLE）进行性能与功能验证。

**📈 对比分析**

通过比较行数、编译时间、单核运行时性能来评估。xDSL-CCPP在编译时间上略高（约0.5–0.6 s），但运行时性能与原工具相当（<4%差异），且对不同架构（Cirrus、ARCHER2）表现一致。

**⚠️ 局限性**

主要限制是编译时间增加、对MLIR生态的依赖需要学习曲线、以及目前仍需通过Flang或Clang生成目标代码，GPU迁移仍是未来工作。

---

## 511. NutriVisRAG: Multimodal Exemplar Grounding for Nutrient Estimation with Vision-Language Models

**arXiv ID:** 2611.08054 | [PDF](https://arxiv.org/pdf/2611.08054v1)

**作者:** Lubnaa Abdur Rahman `[一作]` (University of Bern), Stavroula Georgia Mougiakakou `[通讯]` (University of Bern)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出NutriVisRAG，一种在推理时通过视觉检索相似已标注膳食图片作为多模态示例来辅助视觉语言模型进行营养估计的框架；

**💡 创新点**

其创新点在于首次将视觉检索与多模态示例直接嵌入VLM推理过程，无需任务特定微调，即可利用已有的膳食图片与营养注释进行上下文引导；

**🔧 技术方法**

技术上结合了预训练视觉编码器（PE-Core-G14、SigLIP 2、Nomic Vision）进行检索，并将检索到的图片及其营养标签作为多模态上下文输入到四大预训练VLM（GPT-5.4-mini、Gemma-4-31B、Mistral-Small-3.2、GLM-5.3-Flash）进行生成；

**📊 数据集**

实验使用四个公开膳食图像数据集：Nutrition5k、SNAPMe、MADiMa与EDUCATE，后者包含糖尿病患者的自我记录图像；

**📈 对比分析**

在96种配置（4数据集×4 VLM×3检索器×2提示方式）中，NutriVisRAG在56种情况下降低了主要误差，整体上与最优 Vanilla 的误差相比可实现最高约35.7%的改进，尤其在全膳食提示下表现更佳；

**⚠️ 局限性**

局限性包括检索效果高度依赖配置，视觉相似度并非可靠指标，检索语料库覆盖有限，且未对随机示例或文本检索做对照，未来需进一步探索更具可解释性的检索/重排机制。

---

## 512. Reproducing Kernel (Quasi-)Banach Spaces with the $\ell^p$ Coefficient Norms

**arXiv ID:** 2611.08032 | [PDF](https://arxiv.org/pdf/2611.08032v1)

**作者:** Tianfang Wang `[一作]` (Sun Yat-sen University), Haizhang Zhang `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文研究了重现核（准）巴拿赫空间（RKBSs）的理论，特别关注于非凸的ℓ^p正则化框架（0<p<1）。构建了具有ℓ^p准范数的重现核准巴拿赫空间（RKQBSs），并在适当假设下建立了插值和正则化学习的存在性和稀疏性结果。

**💡 创新点**

创新点在于扩展了RKBS理论，特别是针对0<p<1的情况，构建了新的重现核准巴拿赫空间，并提出了线性和放松的代表定理，揭示了不同p值下的基本差异。

**🔧 技术方法**

使用了重现核理论和泛函分析技术，特别是针对ℓ^p空间的性质进行了深入分析，构建了相应的重现核准巴拿赫空间。

**📊 数据集**

使用了三种典型的平移不变核：高斯核、拉普拉斯核和逆多重二次核，进行了数值实验以验证理论结果。

**📈 对比分析**

通过数值实验比较了不同p值下的模型在准确性和稀疏性方面的表现，结果表明当0<p<1时，模型能够实现更稀疏的表示，而p=1时则表现出经典的稀疏正则化效果。

**⚠️ 局限性**

限制在于当前的RKBS理论仍然依赖于巴拿赫空间的几何性质，尤其是在0<p<1的情况下，缺乏直接应用于函数空间的理论支持。

---

## 513. Symmetry-Aware Surrogate-Assisted Testing of UAVs

**arXiv ID:** 2611.08012 | [PDF](https://arxiv.org/pdf/2611.08012v1)

**作者:** Anas Shrinah `[一作]` (Applied Science Private University), Kerstin Eder `[通讯]` (University of Bristol)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种将障碍物配置映射到统一基准坐标系的对称性感知代理辅助测试方法，用于快速生成无人机自动避障系统的安全关键测试案例。

**💡 创新点**

创新点在于利用几何对称性将多条航线映射到单一基准航线，训练出的神经网络代理可跨航线重用，大幅减少高保真仿真数据需求，并通过分阶段筛选实现高精度与多样性兼顾。

**🔧 技术方法**

采用了两层并行子网络与全连接网络结构的回归神经网络代理、几何变换与约束随机生成器、基准测试竞争框架与动态时间扭曲（DTW）等技术。

**📊 数据集**

使用了超过2500条高保真仿真数据集（包含两障碍物的几何参数与对应最小间隙距离）以及三条竞赛航线（CS2、CS4、CS5）的障碍物配置。

**📈 对比分析**

与2024‑2026年UAV测试竞赛的参赛者进行基准对比，平均关键案例命中率提升至42%（对比约10%），在2024/2025赛季排名第一，2026赛季排名第二，且测试精度高达96.7%，但输出轨迹多样性相对较低。

**⚠️ 局限性**

主要限制是代理对不同航线的泛化能力受限，尤其是跨多段航线时预测误差上升；此外受限于约束随机生成的障碍配置范围，导致生成的轨迹多样性不足。

---

## 514. Post-Quantum Authentication of IoT Data on Ethereum: Signatures, the Calldata Floor, and Device-Signed Merkle Roots

**arXiv ID:** 2611.08108 | [PDF](https://arxiv.org/pdf/2611.08108v1)

**作者:** Ming-Fong Sie `[一作]` `[通讯]` (Chung Yuan Christian University), Ming-Fong Sie (Chung Yuan Christian University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文测量并评估了在以太坊上使用后量子签名（ML‑DSA、Falcon、SLH‑DSA、LMS、XMSS）对物联网记录进行身份验证的成本，包括设备端签名时间、Solidity 以及拟议的预编译验证器的 gas 消耗，并在当前与未来的 gas 规则下构建了完整的成本模型。

**💡 创新点**

创新点在于：①结合多种 PQ 签名方案的设备端基准与链上验证器实现，②在考虑 calldata floor、交易上限与即将到来的 Glamsterdam repricing 的前提下，对每条记录与 Merkle 根签名的成本进行量化；③提出标准化的 LMS/HSS 验证器；④给出可直接用于设计决策的闭式分解点与数据可用性分析。

**🔧 技术方法**

使用的技术包括：以太坊 EVM、EIP‑7623、EIP‑7825、EIP‑7976、预编译 EIP‑8051/8355/8052、Solidity 0.8.28、Foundry、EthereumJS VM、liboqs、pqm4、ZKNOX 验证器、Keccak‑256、SHA‑256。

**📊 数据集**

数据集主要为：x86‑64、ARM64 与 Cortex‑M4 上的 PQ 库基准（liboqs、pqm4、Campos 等），以及通过 EVM 仿真生成的交易日志与 gas 计数；同时使用 NIST 官方测试向量验证签名器正确性。

**📈 对比分析**

比较方法：对每条记录签名与设备签名 Merkle 根两种模式下，按设备数量、记录数 k 以及签名方案进行 gas 成本对比；与 ECDSA、P‑256 作为基线；在不同 floor 设定下评估成本变化；并通过闭式成本模型计算 break‑even k。性能结果显示：单条 PQ 签名成本约为 ECDSA 的 8.7 倍；根签名成本约 1‑3 千 gas/记录，远低于单条签名；预编译显著降低单条 PQ 验证成本。

**⚠️ 局限性**

局限性包括：①缺乏真实 IoT 硬件（如 STM32、ESP32 等）的现场测评；②未执行 blob 与 rollup 交易；③预编译价格仅为草案，实际部署成本未知；④Solidity 验证器与 LMS/HSS 验证器尚未得到正式审计；⑤成本模型基于当前 Osaka 规则，未来链上升级可能导致结果变化。

---

## 515. STEP: Spatiotemporal Training with Explicit Prediction for Perceptive Humanoid Locomotion

**arXiv ID:** 2611.08097 | [PDF](https://arxiv.org/pdf/2611.08097v1)

**作者:** Youhao Lyu `[一作]` (University of Science and Technology of China), Wenhao Cui `[通讯]` (Noetix Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了STEP框架，结合多查询融合、跨模态与时序预测，实现了统一的多模态控制表示与本体控制策略；

**💡 创新点**

创新点在于一阶段学习：通过四个可学习查询对深度与本体感知进行聚合，同时利用双向掩蔽跨模态预测和动作条件下一步状态预测共同监督，避免了传统先预训练再训练策略的两阶段流程；

**🔧 技术方法**

技术包括多查询注意力聚合、掩蔽跨模态预测、动作条件下一步预测、EMA目标、SIGReg抗坍塌正则、PPO强化学习和软MoE Actor；

**📊 数据集**

使用Isaac Lab中的多种地形数据集（粗糙地面、平台、楼梯、桥隙、踏石）进行仿真训练，并在23-DoF Noetix E1机器人上进行真实硬件测试；

**📈 对比分析**

与基线（单查询、无预测、传统融合）进行对比，仿真中在多种地形上取得最高成功率与遍历率，硬件测试中98/100成功率，特别在无后方视觉覆盖的倒退楼梯上表现优秀；

**⚠️ 局限性**

局限在于持续深度失效（无新深度输入）时仍难以通过跳跃或踏石等间断地形，需进一步提升对间断视觉信息的记忆与预测能力。

---

## 516. Gabor Gabor: E-Gaborformer, a Strictly Causal Speech Recognizer with Gabor-Dictionary Convolutions and Delta-Rule Memory

**arXiv ID:** 2611.08092 | [PDF](https://arxiv.org/pdf/2611.08092v1)

**作者:** Nir Krakowski `[一作]` (Deepdub), Chen Zarfati `[通讯]` (Deepdub)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种严格因果、常量状态的实时语音识别编码器E‑Gaborformer，并在大型转录器上实现；

**💡 创新点**

创新点在于将Gated DeltaNet‑2递归层与滑动窗口注意力按3:1比例混合，并使用固定因果Gabor词典构建深度卷积，解决递归分支训练停滞并保持常量状态；

**🔧 技术方法**

使用的技术包括GDN‑2递归、滑动窗口注意力、固定Gabor词典深度卷积、两类RNN‑T训练优化（两类图、闭式梯度、对齐输出层、Toeplitz卷积）以及MuON+AdamW优化；

**📊 数据集**

使用的主要数据集为LibriSpeech 100h训练集以及生产规模的以色列希伯来语和英语电话/即时消息语料；

**📈 对比分析**

通过与同等规模的E‑Branchformer在LibriSpeech‑100上做严格因果对比，在训练更快且达到相同或更优WER；在生产规模转录器上比Nemotron 3.5 ASR Streaming同等参数下降低28–31% WER，保持常量19 MB状态；

**⚠️ 局限性**

局限在于仅在短语音/30分钟内验证，缺乏长流延迟测量，需要进一步评估多分钟实时语音的发射延迟和鲁棒性。

---

## 517. Performance metrics in machine learning: a scenario-based guide from the confusion matrix to calibration, regression and uncertainty

**arXiv ID:** 2611.08048 | [PDF](https://arxiv.org/pdf/2611.08048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 518. SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization

**arXiv ID:** 2611.08089 | [PDF](https://arxiv.org/pdf/2611.08089v1)

**作者:** Anyuan Zhuo `[一作]` (ByteDance Seed), Pinyan Lu `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用稀疏自编码器（SAE）从 Qwen3‑30B‑A3B 的 MoE 层输出中学习无监督特征，并通过计算每个专家对每个 latent 的贡献分数构建主导图，进一步用 GPT‑5.5 自动解释主导集，从而对专家的专长进行细粒度分析。

**💡 创新点**

创新点：
① 用模型自身的特征空间替代人工设定的域标签，以无监督方式捕捉专家在特征维度上的真正分工；
② 将每个 latent 的激活精确分解到各激活专家，得到贡献分数并归一化，构造专家主导集；
③ 通过自动解释词云对主导集进行语义化描述，展示专家专精细到子领域（如 Java、前端 Web、几何、算术等）。

**🔧 技术方法**

技术方法：
- 稀疏自编码器（Top‑k ReLU 编码+线性解码），
- 贡献分数计算与归一化（c(j,i,x)、C(j,i,x)、s(j,i)），
- 主导图（dominance graph）构建与 Jaccard 相似度分析，
- GPT‑5.5 自动解释与词云生成。

**📊 数据集**

数据集：
- Qwen3‑30B‑A3B 模型的第 4、30、46 层；
- 为每层训练一个 SAE（latent 2^20）；
- 验证集 1.5 B 令牌，分布与训练集一致且无重叠。

**📈 对比分析**

与传统基于域标签的专家分析比较：
- 贡献分布比路由权重更陡峭，latent 的激活更集中在少数专家；
- 专家之间主导集重叠极低，重叠集中在少数同域专家对；
- 通过词云和主题一致性显示专家分工细粒度；
- 该方法在定性上优于仅统计专家激活频次的做法，未给出数值性能指标，但通过可视化与主题一致性证明效果。

**⚠️ 局限性**

局限性：
- 结果高度依赖 SAE 的表征质量，若 SAE 失效则无法得到主导集；
- 自动解释依赖 GPT‑5.5，存在主观性与一致性问题；
- 仅在单一模型、单层级（4、30、46）验证，缺乏跨模型或层级的泛化评估；
- 未对该分析对下游任务性能或模型剪枝的量化影响进行实验。

---

## 519. Value or Attention: Investigating Layer-wise Relevance Propagation for Graph Attention Networks

**arXiv ID:** 2611.08088 | [PDF](https://arxiv.org/pdf/2611.08088v1)

**作者:** Ken Schreiber `[一作]` (Fraunhofer Heinrich Hertz Institute), Jonas Naujoks `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种基于 LRP 的产品规则，用于在 GATv2 中分别计算价值路径与注意力路径的相关性，并通过 BA‑House 数据集验证其在解释网络决策时的分离优势。

**💡 创新点**

将 LRP 规则统一为一个可调权重的乘积规则，从而同时兼顾价值路径与注意力路径，实现对两条路径单独解释，揭示了这两条路径在模型解释中的不同意义。

**🔧 技术方法**

使用 LRP（层级相关传播）、GATv2 的注意力机制、深度 Taylor 分解与 softmax 归一化分析，并与 CP‑LRP、AttnLRP 进行对比实验。

**📊 数据集**

使用 BA‑House（基于 BA‑2Motifs 的合成图分类数据集），该数据集包含已知的房子 motif 作为基准。

**📈 对比分析**

通过在同一图节点上对比 CP‑LRP、AttnLRP 与本方法在价值路径、注意力路径和组合路径上的相关性，展示了分离解释在揭示模型内部机制方面的优势，未给出传统数值性能指标，但在解释质量上更具可解释性。

**⚠️ 局限性**

仅在单层单头 GATv2 进行实验，未涉及多层、多头或真实世界数据；此外两条路径在 GATv2 中可能因共享投影而产生信息混杂，限制了方法的普适性。

---

## 520. Development of Anthropomorphic Arm for collaborative and Home Service Robot CHARMIE

**arXiv ID:** 2611.08013 | [PDF](https://arxiv.org/pdf/2611.08013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 521. Spatiotemporal Dynamics-Aware Industrial Anomaly Detection and Reasoning

**arXiv ID:** 2611.08080 | [PDF](https://arxiv.org/pdf/2611.08080v1)

**作者:** S-Wei Chen `[一作]` (National Taiwan University), Shao-Yuan Lo `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出MUSTARD框架，利用光流引导的关键帧选择和多模态信息，实现对工业动态异常的检测与解释。

**💡 创新点**

创新点在于将光流用于关键帧抽取、构造轨迹帧与流幅曲线两种增强模态，并通过少量正常样本的模式提炼与推理，实现无需全量训练的时空异常推理。

**🔧 技术方法**

技术包括光流估计（Dual TV-L1）、轨迹点采样与Farthest Point Sampling、基于MLLM（Qwen2.5‑VL‑7B）的视频描述与LLM（Gemma3‑12B）模式抽取与跨模态推理。

**📊 数据集**

主要使用Phys‑AD和IPAD这两大工业动态异常视频数据集，分别包含22个物体类别和16个设备类别。

**📈 对比分析**

与传统无监督/弱监督方法、通用MLLM和专用VAD MLLM相比，MUSTARD在Phys‑AD上达到0.681 AUC、0.456 PAEval，IPAD上0.572 AUROC，均显著优于基线。

**⚠️ 局限性**

局限性包括对光流质量依赖较高、关键帧数量仍有限，且对极为细微或长时序异常的检测效果尚待进一步提升。

---

## 522. KeHoSu: Keyed Homoglyph Substitution for Authenticated Text Provenance over Copy-Paste Channels

**arXiv ID:** 2611.08046 | [PDF](https://arxiv.org/pdf/2611.08046v1)

**作者:** Andrei Alexandru Drăghici `[一作]` (University of Bucharest), Dorjan Hitaj `[通讯]` (Sapienza University of Rome)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于键控同形字母替换的文本水印方案 KeHoSu，用于在复制粘贴渠道中嵌入经过身份认证的可追溯标记，能够在句子重排、部分引用及轻微编辑下保持可识别性。

**💡 创新点**

创新点包括：① 将同形字母视为可编码的信道，形成键控的、内容锚定的索引；② 对该信道进行容量、可移除性与可检测性的理论分析；③ 采用闪存式（fountain）码与 AEAD/签名结合的复合密钥方案，实现低误判率的身份验证；④ 通过内容窗口哈希实现对句子边界的无序稳健性，允许文本被分割、重组后仍能恢复。

**🔧 技术方法**

主要技术：同形字母映射、键控哈希索引、内容锚定窗口、随机线性闪存码、AEAD（AES‑256‑GCM‑SIV）截断标签、可选异步签名（BLS/EdDSA），以及错误修正（Reed‑Solomon）和投票一致性恢复。

**📊 数据集**

实验数据集：三本公共领域英文书籍（共约14k字）和40篇 Claude Haiku 5.5 生成的 LLM 故事（共约4000字/篇），此外在18个常见平台上进行复制粘贴传输测试。

**📈 对比分析**

方法对比：与此前的无键位、顺序相关的同形编码方式相比，KeHoSu 在 178 字（LLM 文字）和 167 字（标准英文）内即可可靠恢复标记；在句子重排、部分引用、轻微编辑和不同平台粘贴下均保持 100% 以上恢复率；相比位置式编码，KeHoSu 在重排与删除前缀后仍能完整恢复，而位置式编码在任何单词删除或重排均会导致失效。

**⚠️ 局限性**

局限性：① 需要足够多的同形字母携带者（约 0.45 的字母比例），对没有同形字母的语言或字符集适用性有限；② 仅适用于文本在复制粘贴过程中保持 Unicode 码点不变的渠道；③ 在格式化（如 HTML、Markdown、富文本）或转换（打印、图片 OCR）时可能失效；④ 由于采用内容哈希索引，对大规模编辑或拼接产生的窗口冲突仍有一定恢复阈值；⑤ 在高错误率或极端重排场景下，恢复概率会下降，需更长文本或更强纠错。

---

## 523. Learning Dynamics of On-Policy Distillation

**arXiv ID:** 2611.08016 | [PDF](https://arxiv.org/pdf/2611.08016v1)

**作者:** Guan Zhe Hong `[一作]` (University of Oxford), Yi Ren `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对在语言模型后训练中使用的 On‑Policy Distillation（OPD）进行机制化分析，拆解其反馈循环为“转换（conversion）”与“广播（broadcast）”两个阶段，并通过神经切线核理论解释教师监督如何通过参数更新影响学生在后续生成中的行为与分布。进一步探讨离线蒸馏准备对 OPD 学习动力学的调节作用，并在数学推理任务上实证验证其效果。

**💡 创新点**

创新点包括：①提出 OPD 的两阶段反馈框架，量化教师请求对参数更新的可访问性与连贯性；②揭示广播机制通过张量核传播到前缀分布，解释局部教师一致性不必导致整体推理提升；③实验验证离线蒸馏可显著提升后续 OPD 的收敛和可靠性，解决 OPD 可能导致生成退化的现象。

**🔧 技术方法**

技术手段：神经切线核（NTK）与连续梯度流分析、reverse‑KL 对抗训练实现 OPD、离线蒸馏准备、以及大规模语言模型（Qwen、Llama）在数学推理数据集上的训练与评估。

**📊 数据集**

使用的数据集：OpenMathInstruct‑2、AIME 2024/2025、MATH‑500、DAPO‑Math‑17k、以及 Math‑500 等数学推理与解题数据集。

**📈 对比分析**

评估方法：在相同模型规模、相同训练步数下对比单纯离线蒸馏、单纯 OPD 与离线+OPD 三种设置；使用 pass@k（k≤32）和每题解答率等指标衡量性能。实验显示，足够的离线准备后，离线+OPD 在 pass@32、整体准确率与题目可靠性上显著优于单纯 OPD 或蒸馏，且在 100 步 OPD 终点提升约 10% 以上。

**⚠️ 局限性**

局限性：实验仅覆盖数学推理任务与少数大模型；理论基于连续梯度流，离散步长的近似可能影响结论；对教师与学生的超参数与匹配细节仍需更系统探讨；未检验该框架在其它领域或更小模型上的泛化能力。

---

## 524. NOMA Assisted Coordinated Beam Hopping for VLEO Satellite Communication Systems

**arXiv ID:** 2611.08174 | [PDF](https://arxiv.org/pdf/2611.08174v1)

**作者:** Jiayang Kang `[一作]` (Beijing Information Science & Technology University), Shaoli Kang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出了一种适用于非常低地球轨道（VLEO）卫星通信系统的 NOMA 辅助协调波束跳变（NOMA-CBH）方案，并设计了低复杂度的 LCD（分解式）算法实现波束调度与功率分配，同时推导了系统的闭式误差概率、冗余多样性以及延迟受限吞吐量。

**💡 创新点**

创新点包括：①在 VLEO 环境下引入快速轨道变化、大气衰减、残余多普勒失真和指向误差的综合通道模型；②将波束调度与功率分配解耦为两阶段 LCD 方案，显著降低计算复杂度；③给出 NOMA-CBH 的闭式误区概率表达式并进行极限分析，证明了 2μ 的多样性阶；④通过与 OMA-CBH、随机匹配 NOMA-CBH 以及全枚举最优 benchmark 的对比，验证了该方案在吞吐量、可靠性和指向误差鲁棒性方面的优势。

**🔧 技术方法**

采用了 NOMA 资源分配、协同波束跳变、Nakagami‑m 信道建模、贪心大尺度通道匹配、离散功率搜索、闭式解析与极限分析，以及 Monte Carlo 仿真验证。

**📊 数据集**

主要使用仿真生成的 VLEO 轨道几何、地面用户与基站的随机位置分布，并基于 3GPP TR 38.811/821 的卫星参数和通信参数进行系统建模；未使用真实测量数据集。

**📈 对比分析**

通过与 OMA‑CBH、随机用户匹配 NOMA‑CBH、固定功率分配 NOMA‑CBH 以及全枚举最优 benchmark 的对比，利用系统总速率、聚合流量、指向误差鲁棒性、误区概率等指标评估性能。实验结果显示 LCD NOMA‑CBH 在功率、吞吐量和可靠性方面显著优于基线，且在大多数场景下逼近全枚举最优解。

**⚠️ 局限性**

主要限制包括：①调度仅基于大尺度通道信息，未考虑小尺度波动和同步延迟误差；②仅考虑两波束协同，未处理多波束干扰和更复杂的多用户场景；③调度策略以容量为导向，未结合排队和实时服务需求；④算法设计和仿真基于理论模型，尚未在真实硬件或实验平台上验证实现可行性。

---

## 525. Reflective Nature Experiences in Virtual Reality

**arXiv ID:** 2611.08167 | [PDF](https://arxiv.org/pdf/2611.08167v1)

**作者:** Shiva Ghasemi `[一作]` (Virginia Tech), Scott McCrickard `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并评估了一个可访问的VR自然体验，通过沉浸式景观、环境音效、引导呼吸以及交互式检查点来促进正念、减压与情绪健康。

**💡 创新点**

创新点在于：① 采用连续自然步行代替瞬移以提升沉浸感；② 集成可定制的天空、音效与检查点，满足不同感官与身体需求；③ 通过混合方法（量化问卷与质性访谈）系统性验证其心理效益。

**🔧 技术方法**

技术包括Meta Quest 2头显（手部追踪、全景音频）、Unity/Unreal 3D 渲染、实时定位与步行控制、引导呼吸脚本、可交互的自然环境元素。

**📊 数据集**

使用自建的数据集：10名大学生的前后5点李克特量表问卷（压力、情绪、自然连接、正念）以及访谈录音。

**📈 对比分析**

对比方法为 Wilcoxon 符号秩检验，结果显示压力、情绪、自然连接、正念均显著提升（p < 0.02），并通过定性主题分析进一步阐释机制；未与传统冥想或其他VR环境做直接对照。

**⚠️ 局限性**

局限性包括样本量小、缺乏对照组、只观察短期效应、硬件视场与视觉跳转导致沉浸受限、未能复制真实徒步的身体运动与多感官刺激。

---

## 526. H-TRIP: Exact Many-to-Many Distance Index for Multi-Robot Exploration

**arXiv ID:** 2611.08139 | [PDF](https://arxiv.org/pdf/2611.08139v1)

**作者:** Elif A. Özaslan `[一作]` (Kastamonu University), Tolga Özaslan `[通讯]` (Ankara Yildirim Beyazit University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了H-TRIP，一种在多机器人探索过程中可保持精确几何距离的层级距离索引，用于随时更新增量占据网格；

**💡 创新点**

其创新点在于将固定边界空间层级与最小加法闭包结合，并采用一次性秩‑一增量更新，仅传播距离下降，避免全局重新搜索；

**🔧 技术方法**

实现技术包括最小加法半环与克莱尼闭包、基于网格的层级划分、秩‑一增量更新、批量k×k查询和层级提升；

**📊 数据集**

实验使用了MovingAI基准的六幅占据网格（尺寸16–32，5cm分辨率）以及16台机器人在模拟前沿搜索中的数据；

**📈 对比分析**

与传统k×BFS、Dijkstra等方法比较，H-TRIP在完全揭示的地图上对25个视点可实现最高82倍加速，在实时探索中平均提升2–9倍，并与参考搜索保持完全一致；

**⚠️ 局限性**

局限在于仅支持二维占据网格的插入（增量）更新，且当开口影响多个层级时更新成本可能超过BFS补全，需要进一步扩展至非单调更新和降低内存占用。

---

## 527. How Far Can Untrained Neural Networks Go in Anomaly Detection? From Tables to Pixels

**arXiv ID:** 2611.08136 | [PDF](https://arxiv.org/pdf/2611.08136v1)

**作者:** Hyojun Yi `[一作]` (Sejong University), Seunghyoung Ryu `[通讯]` (Sejong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了PRISM框架，利用固定随机非线性特征和Mahalanobis分数进行无学习的异常检测。

**💡 创新点**

创新点是证明在未训练的网络中即可实现竞争性检测性能，并提供理论分析说明非线性变换对分数间隔的影响。

**🔧 技术方法**

使用随机初始化的线性层、归一化、LeakyReLU的PRISM块以及Mahalanobis距离进行评分；视觉端使用多尺度卷积与BlurPool。

**📊 数据集**

在47个ADBench表格数据集、MVTec AD和VisA 27个视觉类别上进行评估。

**📈 对比分析**

与14种表格基线和8种视觉基线比较，PRISM‑T在平均AUROC上名列第一，PRISM‑V在图像/像素层AUROC分别达到90.4%/96.4%，仅用约0.33M参数，性能仅低于预训练模型。

**⚠️ 局限性**

局限在于相较于最优预训练方法仍有性能差距，且对某些视觉类别的定位效果不如专门的记忆/邻近搜索方法；未探索不同随机初始化对结果的稳定性。

---

## 528. Between Hi-Fi and MMI: the overlap of redundancy and synergy in partial information decomposition

**arXiv ID:** 2611.08131 | [PDF](https://arxiv.org/pdf/2611.08131v1)

**作者:** Niels Van Santen `[一作]` (Ghent University), Daniele Marinazzo `[通讯]` (Ghent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了两源信息的部分信息分解（PID）与Hi‑Fi特征重要性方法的关系，提出了通过重叠量δ将所有非负PID映射到一条线段上，并将不同冗余/互补度量（如Hi‑Fi、MMI、BROJA、对称参考PID等）按δ排序，阐明了它们在同一分布下的相对位置；

**💡 创新点**

创新点在于：①引入重叠参数δ统一描述PID空间；②证明Hi‑Fi对应δ=0点，MMI对应最大重叠；③给出对称参考PID并证明其在重叠上最小；④为多源情况提供零重叠条件与最小重叠线性规划；⑤将重叠与三阶互联信息拆分，明确协同信息的三阶结构；

**🔧 技术方法**

使用了信息理论工具（互信息、共信息、最大熵模型、对数线性模型）以及集合论的参考族框架，结合数值实验（随机分布、Gaussian、AND/XOR等）进行验证；

**📊 数据集**

主要使用合成数据集（如AND、XOR、SUM、COPY、RdnXor、PwUnq、Gaussian三元组等）以及两类随机生成分布（随机二进制、随机高斯）进行实验；

**📈 对比分析**

通过在每个分布上计算不同PID量的重叠δ位置，并在800个随机分布上验证排序一致性；实验显示Hi‑Fi与MMI在δ上分别为极端，BROJA等在中间，pairwise-reference PID最小；性能方面，Hi‑Fi提供最小重叠和最大唯一信息的下界，MMI提供最大重叠和最小唯一信息的上界；

**⚠️ 局限性**

局限性包括：①零重叠在多源情况下往往不可达；②Hi‑Fi并非多源PID；③多数度量需全枚举子集，计算量大；④估计时存在偏差（共信息偏差、稀疏样本问题）；⑥只针对信息理论形式，实际应用需考虑采样与模型假设。

---

## 529. H-BRICK: Hierarchical Boundary Reachability Index with Compressed Kleene Closure for Directed Grid Graphs

**arXiv ID:** 2611.08119 | [PDF](https://arxiv.org/pdf/2611.08119v1)

**作者:** Tolga Özaslan `[一作]` (Ankara Yıldırım Beyazıt University), Elif A. Özaslan `[通讯]` (Kastamonu University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了二维网格嵌入的有向图中精确可达性查询的索引，提出了 H‑BRICK 基于层级边界压缩的 Kleene 闭包索引。

**💡 创新点**

通过把局部闭包投影到边界端口并递归合成层级接口关系，构建了只暴露边界可达性的层级索引，实现了空间压缩且不近似的可达性查询。

**🔧 技术方法**

采用布尔半环 Kleene 星、布尔矩阵乘法、GraphBLAS 及自适应位并行闭包引擎，配合 skip‑lift 预计算和批量多源多目标查询加速。

**📊 数据集**

在 Moving AI 公开的八张二维占据网格（城市街道、游戏竞技场、迷宫等）上进行实验，生成六种方向化流量模型（随机非对称到梯度流）。

**📈 对比分析**

与 BFS、SCC‑DAG、GRAIL、O’Reach、2‑hop 等基准比较，H‑BRICK 在均值查询延迟微秒级，速度可达 BFS 的 190 倍，索引占用约比全闭包小 240 倍；批量查询吞吐可提升 5–23 倍。

**⚠️ 局限性**

只适用于静态网格图；层级参数需手工调优；在极稀疏或高度分离的图中预处理成本不划算，且对动态更新支持有限。

---

## 530. Cloudy Skies: Early Detection of Jamming Against Iridium LEO Satellite-IoT Links

**arXiv ID:** 2611.08118 | [PDF](https://arxiv.org/pdf/2611.08118v1)

**作者:** Ingrid Huso `[一作]` (Hamad Bin Khalifa University), Gabriele Oligeri `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在Iridium LEO 卫星-IoT 链路上，利用已成功解码的物理层 IQ 采样进行及时干扰检测，并公开包含 47 种不同干扰条件的真实数据集。

**💡 创新点**

创新点：①发现干扰检测的关键是接收机噪声底的升高，而传统的 SNR / C/N0 指标在已解码消息中几乎无效；②证明即使在相同 SNR 类别下，基于 AI 的异常检测器仍能有效区分干扰；③公开了首批针对卫星 IoT 的干扰数据集；④通过与 GNSS 传统指标对比，验证 AI 方法的优越性。

**🔧 技术方法**

使用技术：三种一类 autoencoder（Raw AE、LSTM‑AE、IB AE）；基线为 moment‑based C/N0 估计器和 symbol‑directed SNR 估计器；MATLAB 训练与推理；IQ 采样特征、SNR 统计、噪声底提升 ΔN 等。

**📊 数据集**

数据集：在四种场景（低多径/高多径、静态/移动）下收集的 Iridium 消息与 IQ 采样，覆盖 47 种干扰条件（AWGN、伪装、不同功率、距离、移动性），已公开发布。

**📈 对比分析**

比较方法：采用 AUC、TPR/FPR 等指标，进行三类测试（块大小、域内、迁移）。结果显示 Raw AE 与 LSTM‑AE 在 1000 样本块（≈40 ms）时 AUC ≥ 0.84、TPR ≥ 0.98；传统 GNSS 指标 AUC ≤ 0.7；IB AE 在精度上略优但模型规模和计算成本高。

**⚠️ 局限性**

局限性：①仅评估了恒定 barrage（AWGN 或伪装）干扰，未考虑自适应或定向干扰；②仅针对 Iridium 频段，未验证其它卫星系统的泛化；③未收集未解码波束的 IQ 数据，导致对极低噪声底提升的感知不足；④阈值在不同环境间迁移需重新校准，迁移时 FPR 可能显著上升。

---

## 531. BabyTheorist: A Benchmark for Learning to Theorize the World from Observation Alone

**arXiv ID:** 2611.08177 | [PDF](https://arxiv.org/pdf/2611.08177v1)

**作者:** Doojin Baek `[一作]` (Korea Advanced Institute of Science and Technology), Sungjin Ahn `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 BabyTheorist 基准，用于评估模型在连续、未标记的视觉观测序列中学习可复用理论的能力。

**💡 创新点**

创新点在于引入隐藏的 DSL、频繁子程序、层级难度递增与级别跳过的持续评估协议，构建可控制的观察理论学习实验。

**🔧 技术方法**

实验采用多种技术：自回归 Transformer、预训练 LLM、连续/离散潜在动作模型、以及可组合程序推理模型 NEO 等。

**📊 数据集**

使用人工合成的 15×15 彩色网格数据集，程序通过隐藏的组合规则生成前后图像对。

**📈 对比分析**

对七种基线在八级别持续训练测试，结果显示所有模型均停留在前两级，最大得分仅在单对象模式下达到 Level 2，整体表现远低于理想阈值。

**⚠️ 局限性**

局限性包括过度简化的合成环境、有限的原语集合以及缺乏真实世界数据验证，导致难以评估在更复杂情境中的迁移与泛化能力。

---

## 532. ProQuant: Progressive Quantization-aware Training for Edge MLLMs

**arXiv ID:** 2611.08216 | [PDF](https://arxiv.org/pdf/2611.08216v1)

**作者:** Yufei Xue `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了面向小规模多模态大型语言模型的权重‑激活量化感知训练框架 ProQuant，解决低位宽量化下的性能衰退问题。

**💡 创新点**

创新点包括：①自标注 (SRL) 方案将公开多模态数据重新标注为目标模型的响应，缩小分布差异；②进阶块激活 (PBA) 机制根据视觉敏感度因子 τ 优先恢复视觉理解能力；③将上述两项结合到权重‑激活 QAT 中，实现 W4A8/W4A4 低位宽压缩。

**🔧 技术方法**

采用了量化感知训练、视觉敏感度评估、分阶段块激活调度、混合精度策略等技术。

**📊 数据集**

使用 COCO Caption 等公开多模态数据集，通过自标注生成响应；评估基准为 MME、AI2D、TextVQA、ChartQA、DocVQA、MMMU、ScienceQA、SeedBench 等多模态任务。

**📈 对比分析**

与 MBQ、SmoothQuant、MQuant、LLM‑QAT 等 PTQ 和 QAT 基线对比，W4A8 下平均精度仅下降约 1%，W4A4 下提升 3–6 个百分点，且在 PTQ 方法失效时仍保持可用性。

**⚠️ 局限性**

局限性包括：自标注需要额外推理成本；混合精度收益因模型而异；在极低位宽下仍受激活异常影响，需更细粒度的多模态特定策略。

---

## 533. IntentWeaver: Sustaining Coherence across Evolving Intents in Human-AI Co-Writing

**arXiv ID:** 2611.08214 | [PDF](https://arxiv.org/pdf/2611.08214v1)

**作者:** Yuanhao Zhang `[一作]` (Hong Kong University of Science and Technology), Xiaojuan Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了IntentWeaver，一套基于LLM的共写工具，通过从生成草稿中自动提取并持续维护意图层级，实时评估文本与意图的一致性，支持冲突提醒、提示与影响摘要等交互，帮助作者在AI共写中保持高层意图的一致性并持续演进。

**💡 创新点**

将写作意图视为从文本中动态浮现的结构，而非事先固定的规范；实现了从草稿即时推断意图图、评估满足度并在层级间传播变更；提供冲突检测、提示生成及影响摘要等交互机制，首次实现了“意图作为活跃结构”的共写交互。

**🔧 技术方法**

利用大型语言模型对草稿进行生成与意图推断；构建基于目标导向需求工程的意图图（节点、边、贡献类型）；实现图形化交互界面并在后端实现意图满足度计算与变更传播算法；在交互中使用预执行确认、冲突警报与影响摘要等技术。

**📊 数据集**

研究中使用两篇社交媒体写作任务（LinkedIn帖子与跑步社群帖子）作为任务文本；进行12人形成性访谈、24人within‑subjects实验；评估阶段还邀请3名专业写作者对最终文本进行打分。

**📈 对比分析**

采用within‑subjects对比法，将IntentWeaver与基线AI写作工具（类似Lex/Google Docs+Gemini）进行比较。关键指标包括漂移检测率、上升回高层意图的频率、高层意图演进计数、专家评定的意图一致性与文本实现度。实验结果显示，IntentWeaver显著提高高层意图回访与演进次数，降低未察觉漂移，提升高层意图的实现度，并在工作量、可用性与创造性支持上表现不逊于基线。

**⚠️ 局限性**

学习曲线较陡峭，部分参与者反映界面与循环流程难以掌握；实验仅涉及短篇社交媒体文本，未检验长篇或复杂文档；每位参与者仅一次25分钟实验，缺乏长期使用观察；推断精度受LLM限制，错误的意图推断可能被忽视；当前系统仅支持有限的意图操作（pin/unpin/edit/add），未涵盖贡献类型或满足度手动调整等功能。

---

## 534. Do sEMG Foundation Models Improve Cross-Subject Generalisation?

**arXiv ID:** 2611.08199 | [PDF](https://arxiv.org/pdf/2611.08199v1)

**作者:** William Raftery `[一作]` (Imperial College London), Dario Farina `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种sEMG基础模型在三份数据集上进行系统评估，采用 intra‑session 与 inter‑subject 切分，并对比 full fine‑tuning、from‑scratch 与 linear probing 三种微调策略，量化预训练对跨主观泛化的贡献

**💡 创新点**

首次通过同一模型架构的对照实验消除模型容量偏差，揭示预训练在跨主观任务中几乎无效，强调了交叉主观泛化评估的重要性

**🔧 技术方法**

使用自监督掩码重建（TinyMyo）、多尺度小波预处理+变压器（PhysioWave）和 VQ‑VAE 代码本分类预训练（NeuroRVQ），统一采用均值池化和线性分类头

**📊 数据集**

NinaPro DB5、EPN‑612 与 UCI EMG Gestures 三个手势识别数据集，均划分为 70/10/20 的训练/验证/测试比例

**📈 对比分析**

在 intra‑session 评估中，预训练显著提升准确率（最高可达 97.6%），但在 inter‑subject 评估中，预训练优势消失，经典 ML 基线（如 LDA/SVM）甚至表现更好；总体来看，预训练在跨主观设置下的最大增益仅 2.32%

**⚠️ 局限性**

模型仅靠预训练无法缓解跨主观过拟合；缺乏针对低样本主观数的 fine‑tuning 方法；现有预训练目标未能鼓励学习主观不变特征

---

## 535. Common zeros of homogeneous polynomials and the weight hierarchy of projective Reed-Muller codes

**arXiv ID:** 2611.08203 | [PDF](https://arxiv.org/pdf/2611.08203v1)

**作者:** Rodrigo San-José `[一作]` `[通讯]` (Virginia Tech), Rodrigo San-José (Virginia Tech)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

确定了在有限域上任意一组线性无关的同类齐次多项式的最大公共零点数，证明了Beelen、Datta和Ghorpade的猜想，并确定了相应的投影Reed-Muller码的完整权重层次。

**💡 创新点**

证明了Beelen、Datta和Ghorpade的猜想，并通过使用Green的超平面限制界限和足迹论证将问题简化到已解决的仿射情况，展示了在每个度数下的完整层次结构。

**🔧 技术方法**

使用了Green的超平面限制界限、足迹技术以及Wei对偶性等数学工具。

**📊 数据集**

使用了有限域上的同类齐次多项式集，具体数据集未明确给出，但涉及到的多项式和点集是基于有限域的结构。

**📈 对比分析**

通过与仿射点集的比较，使用Hilbert函数和足迹技术，证明了在每个度数下的上界，并 conjecture 该上界在每个度数下都是尖锐的。

**⚠️ 局限性**

限制在于对于更高的度数，可能无法通过单一有序多项式族的连续跨度来获得达到广义汉明权重的子码。

---

## 536. SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving

**arXiv ID:** 2611.08196 | [PDF](https://arxiv.org/pdf/2611.08196v1)

**作者:** Sai Vallurupalli `[一作]` (University of Maryland Baltimore County), Francis Ferraro `[通讯]` (University of Maryland Baltimore County)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SMART框架，通过多代理与结构化角色提示获取、筛选并利用隐式实体状态知识来验证科学主张

**💡 创新点**

创新点在于：①将实体状态知识拆解为对比前提对，②使用多模型验证与动态专家分配实现可验证的、基于证据的推理

**🔧 技术方法**

技术包括：多代理LLM推理、对比前提生成、参数知识与Web搜索验证、元推理（Reasoning Judge）以及多轮交互式提示

**📊 数据集**

使用了三大科学事实验证数据集：SciFact、Scifact-Open与SFBench（共577条主张）

**📈 对比分析**

与Chain‑of‑Thought和RAG基线对比，SMART在支持/驳斥的平衡准确率上提升约5%–10%，单模型验证最多提升20%，对比前提和元推理对性能提升约3%–8%

**⚠️ 局限性**

局限包括：模型仍可能产生偏见与幻觉，Web搜索结果不稳定且易受噪声影响，前提生成质量与模型规模相关，实验未覆盖所有专业领域

---

## 537. Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation

**arXiv ID:** 2611.08179 | [PDF](https://arxiv.org/pdf/2611.08179v1)

**作者:** Jaewon Cho `[一作]` (KAIST), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的多语言自我蒸馏方法 TOPSD，通过在学生推理轨迹中动态确定可监督前缀并验证教师恢复能力，从而提升多语言数学推理表现。

**💡 创新点**

创新点在于发现并量化了教师漂移（Teacher Drift）现象，并设计了两阶段策略：可教学前缀选择与教师恢复验证，避免在错误累积后仍给出错误指引。

**🔧 技术方法**

技术包括：On-Policy Self‑Distillation（OPSD）框架、教师‑学生对数似然比、连续阈值检测、贪婪解码验证、分层均衡的损失归一化。

**📊 数据集**

使用 OpenThoughts 多语言数据集（10 种语言，共 5,000 题）和 PolyMath High 测试集进行评估；实验也在单语（English）AIME 2024/2025 上验证。

**📈 对比分析**

与 Vanilla、OPSD、COPSD 等基线对比，TOPSD 在三种 Qwen3 模型规模（1.7B、4B、8B）下均取得 Avg@3 和 Pass@3 的最高分，尤其在低至中等资源语言中提升 0.89–3.47% (Avg@3) 与 0.80–4.00% (Pass@3)。

**⚠️ 局限性**

局限性：仅验证数学推理任务，对其他推理领域和模型族的通用性尚未探索；依赖英文参考答案，其他源语言或无参照情境下效果未知；计算成本高，需评估能耗与环境影响。

---

## 538. Improving Context-Shift Robustness of Convolutional Networks via Context-Regularized Cross-Entropy

**arXiv ID:** 2611.08157 | [PDF](https://arxiv.org/pdf/2611.08157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 539. Uniform Three-Fold Product Expansion of Algebraic-Geometry Codes over a Fixed Field

**arXiv ID:** 2611.08172 | [PDF](https://arxiv.org/pdf/2611.08172v1)

**作者:** Cassie Ding `[一作]` (University of Pennsylvania), Min-Hsiu Hsieh `[通讯]` (Hon Hai (Foxconn) Research Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了一类经典编码的三重乘积扩展性，特别是通过构造在固定域上的显式代数几何码族，满足常数速率和线性距离的条件。

**💡 创新点**

创新点在于首次构造出在固定域上具有常数速率、线性距离和三重乘积扩展性的显式代数几何码族，解决了之前未能实现的理论问题。

**🔧 技术方法**

使用了几何方法，通过将和码的字视为在三重曲面上的数组，并利用线束的截面来证明乘积扩展性。

**📊 数据集**

使用了Garcia-Stichtenoth塔上的一点评估码，长度为N_i=728·27^(i-1)，极点阶数为⌊N_i/5⌋。

**📈 对比分析**

与现有方法相比，本文的方法通过几何构造提供了更强的乘积扩展性，性能上证明了在每个楼层的三重乘积扩展性是均匀的，且常数独立于i。

**⚠️ 局限性**

限制在于，虽然构造了显式的代数几何码族，但这些码并不是低密度奇偶校验码（LDPC），且每个非零校验的权重至少线性增长，未能满足所有量子编码的需求。

---

## 540. AI-Native Task-Oriented Semantic Communications over a Relay: Benefits and Vulnerabilities

**arXiv ID:** 2611.08155 | [PDF](https://arxiv.org/pdf/2611.08155v1)

**作者:** Yalin E. Sagduyu `[一作]` (Nexcepta), Aylin Yener `[通讯]` (Ohio State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练三节点语义通信系统，比较了LF、AF、DF等中继策略，并提出在中继层注入结构化扰动的Byzantine攻击，评估其对语义推断的影响。

**💡 创新点**

将学习驱动的中继映射与传统AF、DF相结合，首次在中继层使用无标签、单向前向传播的对抗网络生成扰动，显著降低语义准确率。

**🔧 技术方法**

使用端到端深度学习（卷积神经网络）训练源编码器、中继变换、目标解码器；采用SSIM、PSNR等结构化损失；对抗训练中利用熵、KL散度和功率正则化目标。

**📊 数据集**

采用CIFAR-10彩色图像分类数据集进行实验。

**📈 对比分析**

通过与直接源-目的终端通信基线比较，低SNR下LF方案可提升语义准确率0.4248、PSNR 2.58 dB、SSIM 0.1362；在中继攻击下，语义准确率可下降至基线以下，降幅高达89%。

**⚠️ 局限性**

安全性尚未得到充分保障：攻击仅在单个通道模型下验证；实验仅针对CIFAR-10，未覆盖更复杂场景；未提出针对中继Byzantine攻击的有效防御机制。

---

## 541. MCUFlowNet: Hardware-Aware Neural Architecture Search for Optical Flow on Microcontrollers

**arXiv ID:** 2611.08147 | [PDF](https://arxiv.org/pdf/2611.08147v1)

**作者:** Enmin Lin `[一作]` (KU Leuven), Jona Beysens `[通讯]` (KU Leuven)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在 MCU（带神经加速器）上设计可部署的稠密光流网络，提出宏-微 NAS 框架 PACE-NAS 并实现 MCUFlowNet 两种变体，随后将框架迁移至语义分割任务。

**💡 创新点**

创新点在于：① 通过宏设计先定位并降低峰值 SRAM（内存瓶颈），从而把内存需求与延迟解耦；② 微搜索仅在不改变峰值 SRAM 的空间内进行操作符和深度搜索，使用 NSGA-II 进行多目标（EPE‑Latency）Pareto 优化；③ 首次在 MCU 级神经加速器上开展稠密光流的硬件感知 NAS。

**🔧 技术方法**

使用技术包括：硬件平台剖析与瓶颈诊断；架构重构（把转置卷积改为双线性上采样+卷积）与轻量级上下文模块（ECA + 全局广播门）；权重共享超网络训练（SPOS + FairNAS 抽样 + FlowNAS 对齐蒸馏）；NSGA-II 演化 Pareto 搜索；INT8 量化与 Arm Vela 编译器估计延迟；基于 Ethos‑U55 的实验部署。

**📊 数据集**

数据集：FlyChairs2 训练超网络；FlyingThings3D 进行细调；MPI‑Sintel 评估光流 EPE；Cityscapes 评估语义分割 mIoU；AvoidBench 模拟森林飞行测试成功率。

**📈 对比分析**

对比方法：与重训练后的 EdgeFlowNet 以及公开的 FP32 光流模型（RAFT、NeuFlow v2、FastFlowNet、SPyNet）比较；指标为 EPE、峰值 SRAM、延迟、模拟飞行成功率/碰撞率。结果显示 MCUFlowNet‑S 以 1.76× 的速度提升、仅 1.6% 的 EPE 负增幅，成功率从 41.8% 提升至 51.5%；MCUFlowNet‑L 与 EdgeFlowNet 近似延迟、相近 EPE；在语义分割上实现 1.17× 的速度提升，mIoU 维持相近。

**⚠️ 局限性**

局限性：仅针对 Ethos‑U55 及其 8‑bit INT8 精度；搜索空间不包含深度卷积，可能限制模型表达力；实验仅在仿真与单一硬件平台验证，未覆盖更大输入尺寸或其他 MCU；Vela 延迟估计虽可靠但仍为预测，真实系统可能存在差异；仅报告了两条 Pareto 点，未完整探索整个搜索空间。

---

## 542. Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning

**arXiv ID:** 2611.08144 | [PDF](https://arxiv.org/pdf/2611.08144v1)

**作者:** Zuo Gou `[一作]` (Tsinghua University), Chenhao Lu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过把环境与训练器之间的接口明确定义为网络协议，注入多种无错误的“静默”故障，并系统评估各种检测方法是否能发现这些故障。

**💡 创新点**

创新点在于：①将环境–训练器边界建模为可验证的 wire‑protocol；②提出并实现了“step ledger”记录每一步状态、动作与回报的对比检查；③用 mutation testing 对深度 RL 进行边界级故障注入，展示了传统学习曲线、seed 变异等检查大多无法发现故障。

**🔧 技术方法**

技术实现包括：协议设计（MsgPack + WebSocket）、C++ 侧客户端与 Python 侧训练器的桥接、SB3 PPO 训练、step ledger 与 episode record 记录器、字节级权重对比、gain threshold 与 seed‑spread 检查，以及对 Gymnasium、RLlib、PettingZoo 等多代理库中发现的实际故障的复现。

**📊 数据集**

使用的环境与数据集为四个经典连续控制任务：Pendulum、HalfCheetah、Hopper、七连杆推杆（Pusher），所有实验均在 Stable‑Baselines3 的 PPO 算法上进行。

**📈 对比分析**

比较方法：将注入故障的跑（seed 0–2、不同故障剂量）与十个无故障种子进行对比，分别用学习曲线阈值、seed 变异、字节一致性、episode 记录、step ledger 进行检测。结果表明 step ledger 能捕捉到所有故障（平均训练时间提升 3%），而其他方法仅捕捉到少量（不到 30%）故障；大多数未被捕获的故障对最终策略或日志产生显著影响。

**⚠️ 局限性**

局限性：仅在四个状态空间、稠密奖励的模拟环境中验证；只测试了 SB3 PPO、三个 seed；step ledger 需要环境包装，无法覆盖内部环境错误；未考察稀疏奖励、图像输入或真实机器人；故障注入覆盖面有限，未覆盖所有可能的边界错误。

---

## 543. Educational robotics beyond ROS: a custom library for direct transfer of Webots controllers in the Turtlebot3

**arXiv ID:** 2611.08142 | [PDF](https://arxiv.org/pdf/2611.08142v1)

**作者:** Jaime P. Pérez-Moro `[一作]` (Universidade da Coruña), Francisco Bellas `[通讯]` (Universidade da Coruña)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套自定义 Python 库，能够在 Webots 模拟器与实际 TurtleBot3 Burger 之间直接无缝迁移控制器，省去 ROS 依赖。

**💡 创新点**

创新点在于将 Webots API 与 TurtleBot3 硬件接口镜像一致，实现同一份控制代码可在仿真与真实机器人上直接运行，并提供自动化部署脚本。

**🔧 技术方法**

采用 Python、Webots、自研 OpenCR 固件、Serial/USB 通讯、paramiko+scp 自动部署、以及 NumPy 等库。

**📊 数据集**

使用自定义的 6×10 网格环境（模拟和真实实验室）进行验证，没有使用公开数据集。

**📈 对比分析**

通过前进 25 cm、90° 旋转、障碍物检测等实验，仿真与真实机器人表现高度一致，学生实验表明学习难度降低，性能表现符合教学需求。

**⚠️ 局限性**

局限性包括仅支持目前使用的舵机、IMU、LiDAR，无法扩展到高带宽传感器（如 RGB‑D），且仅在 TurtleBot3 Burger 平台上验证，未在更复杂环境或更大规模机器人集群中测试。

---

## 544. Beyond K/V Caches: Cross-Model State Handoff in Hybrid Linear-Attention LLMs

**arXiv ID:** 2611.08133 | [PDF](https://arxiv.org/pdf/2611.08133v1)

**作者:** Jiacheng Cai `[一作]` (South China University Of Technology), Ziqian Zeng `[通讯]` (South China University Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Qwen3.5 混合线性注意力模型的跨模型状态迁移（FA 缓存、REC 与 Conv）提出统一框架 Hybrid State Transfer (HST)，实现不同规模模型的快速切换而无需重复前缀预填。

**💡 创新点**

引入结构化 REC 映射、保持 Conv 历史顺序的仿射对齐、基于冻结目标模型的功能校准以及可折叠的 FA 低秩增量，全面解决 Conv-REC 交互导致的状态不一致问题。

**🔧 技术方法**

线性注意力、Gated DeltaNet、RoPE、偏置对齐、ridge 回归、LoRA 低秩增量、KL 误差与交叉熵的功能校准、离线迁移参数训练与在线状态转换等技术。

**📊 数据集**

ARC-Challenge、HellaSwag、WinoGrande、MMLU 与 GSM8K（分别为零样本、5/8-shot、数学推理任务）。

**📈 对比分析**

与 Ridge、CacheBridge、CacheLink 等完整混合状态基线对比，4B→27B 与 9B→27B 的任务保留率分别为 89.88% 与 94.11%，在所有五个任务上均优于最强基线；在 16K 语境下，HST 体积延迟比原生预填快 21.66× 与 21.91×。

**⚠️ 局限性**

仅在 Qwen3.5 家族内验证，未测试其他线性注意力或 MoE 模型；长上下文理解覆盖有限，且实验与实际推理环境对齐不足。

---

## 545. Sum-of-Squares Refutation Implies Planted Recovery for Semirandom CSPs

**arXiv ID:** 2611.08114 | [PDF](https://arxiv.org/pdf/2611.08114v1)

**作者:** Arpon Basu `[一作]` (Princeton University), Peter Manohar `[通讯]` (Institute for Advanced Study)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一种 n^O(ℓ) 时间的算法，用于解决半随机植入 k 维布尔约束满足问题（semirandom planted CSP），在约束数量 m ≳ (n/ℓ)^{k/2ℓ} 的情况下实现搜索，填补了先前仅能在 m ≳ n^{k/2} 时使用的线性时间算法的空白。

**💡 创新点**

创新点在于首次建立了从 Sum‑of‑Squares (SOS) 反证到植入恢复的通用归约，使得搜索问题可以直接由 SOS 反证算法得到；并引入“学习坏超边”这一思路，避免了传统的显式分解与伪随机性质的依赖。

**🔧 技术方法**

技术上主要采用了 Sum‑of‑Squares 伪期望与 SDP 计算、噪声预测算法、样本拆分（cloning）技术以及线性系统求解，构建了一个完整的学习+回归框架。

**📊 数据集**

论文没有使用真实数据集，而是在理论框架下构造半随机 k‑XOR 或一般布尔 CSP 的合成实例进行分析。

**📈 对比分析**

与之前的“<cit.>”风格时间‑约束权衡相比，本文实现了相同的 m ≳ (n/ℓ)^{k/2ℓ} 与 n^O(ℓ) 速度的匹配；相较于仅在 m ≳ n^{k/2} 下可用的线性时间算法，显著扩展了可处理的约束规模。

**⚠️ 局限性**

局限性包括需要对偏置 β 进行猜测（或预估）、对 SOS 反证算法的假设依赖、学习坏超边需要大量样本、对奇数 k 或特殊约束类型的推广尚未完全解决，且目前仅在理论层面得到验证，缺乏实际实验评估。

---

## 546. Zirah: A Post-Quantum True Random Number Generator Hardware Signer With an Avalanche Noise

**arXiv ID:** 2611.08200 | [PDF](https://arxiv.org/pdf/2611.08200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 547. First-Order Model Checking on Monadically Dependent Graph Classes

**arXiv ID:** 2611.08143 | [PDF](https://arxiv.org/pdf/2611.08143v1)

**作者:** Jan Dreier `[一作]` (HPI), Szymon Toruńczyk `[通讯]` (University of Warsaw)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明在所有单子依赖（monadically dependent）图类上，一阶模型检验问题是固定参数可判定（FPT）的，并给出了一个统一的算法；这与之前仅在特定子类（如平面图、稀疏图、双子宽度类）得到的结果相统一。

**💡 创新点**

创新点在于：①提出并证明了单子依赖类中任意固定长度元组的量化层类型数量受多项式上界（对参数U而言为 |U|^O(1)），扩展了 Sauer–Shelah 引理到更高量化层；②利用此类型上界构造“类型树”表示局部类型，显著压缩了原本指数级别的类型集合；③设计了“几乎受限合并宽度”构造序列，证明所有单子依赖类都能得到此类序列并高效计算；④将上述三项技术结合，得到一个在任何单子依赖类上都能工作的统一模型检验算法。

**🔧 技术方法**

核心技术包括：结构图分解（几乎受限合并宽度、构造序列），有限模型论中的 Gaifman 局部性和类型计数，稳定性理论中的类型上界，类型树（Hintikka 树）与距离逻辑的结合，以及动态规划在构造序列上的实现。

**📊 数据集**

该工作为理论算法，不涉及具体实验数据集；所有结论均在数学上严格证明，适用于任意满足单子依赖条件的无穷多图类（如全图类、无处稠密类、双子宽度受限类等）。

**📈 对比分析**

与之前的算法相比，本算法在任何单子依赖类上实现了统一的 FPT 复杂度，时间为 O(f(|ϕ|)·n^3+)，其中 f 仅取决于输入公式 ϕ；相比之下，之前的算法往往需要先给出特殊宽度分解（如树宽、双子宽度、合并宽度）并带有指数级的参数依赖；本算法消除了这些条件和指数依赖，提供了多项式级别的时间上界。

**⚠️ 局限性**

局限性：①算法仍需在预处理阶段构造几乎受限合并宽度的构造序列，尽管证明可多项式完成，但实现复杂度高；②对双子宽度或合并宽度常数阶分解的高效构造仍是未解决的问题；③虽然对单子依赖类统一，但对于不满足单子依赖的类（如大部分稠密图类）仍不适用；④在实践中，参数 f(|ϕ|) 可能极大，导致实际运行时间不一定可接受。

---

## 548. RoboARCade: Benchmarking Agentic Robot Control

**arXiv ID:** 2611.08209 | [PDF](https://arxiv.org/pdf/2611.08209v1)

**作者:** Satvik Sharma `[一作]` (Stanford University), Jeannette Bohg `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了名为RoboARCade的22项长周期桌面操作任务基准，用于评估基于多模态大模型的直接机器人控制（ARC）。

**💡 创新点**

创新点在于统一了多种任务、接口和评估流程，并引入了在情境学习与提示优化等自我改进机制。

**🔧 技术方法**

使用多模态大型语言模型（如GPT‑6 Astra、Claude Fable 5.1等）、MuJoCo仿真环境以及两种控制接口（多视角RGBD与VIA）。

**📊 数据集**

数据集由22个自制任务与现有公开任务（Meta‑World、RLBench、LIBERO等）组成，每个任务配有随机初始条件、Blender渲染和精细成功判定。

**📈 对比分析**

通过零样本评估和自我改进评估，对比模型与接口，GPT‑6 Astra在VIA下达49.8%子任务成功率、在RGBD下48.3%，并通过一次成功演示提升至73%（或通过提示优化可达72%）。

**⚠️ 局限性**

限制包括任务数量有限、仅覆盖静态桌面操作、接口仅支持单一末端执行、缺乏移动与高精细手部操作等，难以扩展到更动态或多关节任务。

---

## 549. Counting over Unbreakable Decompositions: A Single-Exponential Parameterized Algorithm for Minimum Bisection

**arXiv ID:** 2611.08202 | [PDF](https://arxiv.org/pdf/2611.08202v1)

**作者:** Aditya Anand `[一作]` (Max Planck Institute for Informatics), Karol Węgrzycki `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

给出一个确定性单指数时间算法解决最小二分割（Minimum Bisection）问题，并能计数所有满足给定边界大小的二分割。

**💡 创新点**

创新点在于：①改进了分解步骤，构造了一个在 2^O(k)n^O(1) 时间内得到的（3k,k）边可断开的树分解；②使用对数分解和 Möbius 逆推，动态规划记录计数而非最小值，利用代数运算实现对连通支持的消除；③通过重要分隔（important separators）枚举候选子集，避免了对全局大小的指数级搜索。

**🔧 技术方法**

主要技术包括：树分解（tree decomposition）与可断开性（unbreakability）概念；代数计数（exponential / logarithm of partition function）和 Möbius 变换；重要分隔（important separators）与剪枝；动态规划与状态压缩；组合代数与多项式截断。

**📊 数据集**

该工作为理论算法研究，未使用实验数据集；所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

相较于之前的 2^O(k log k)n^O(1) 算法，本方法在同类参数化复杂度下实现了 2^O(k)n^O(1) 的单指数上界；实验上并无实现细节，仅提供理论时间复杂度证明。

**⚠️ 局限性**

局限性包括：①算法复杂度中的隐式常数较大，实际实现难度高；②仅针对最小二分割问题，尚未直接推广至更广泛的全局约束切分问题；③依赖于树分解的可断开性质，若图结构不满足该条件需进一步改进。

---

## 550. BUMP: Brief Unexpected Mid-Interpolation Peak

**arXiv ID:** 2611.08182 | [PDF](https://arxiv.org/pdf/2611.08182v1)

**作者:** Melinda Flóra Kiss `[一作]` (HUN-REN Alfréd Rényi Institute of Mathematics), Dániel Varga `[通讯]` (HUN-REN Alfréd Rényi Institute of Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文观察并系统分析了在对齐的神经网络权重进行线性插值时，插值端点附近出现的准确率提升现象（BUMP）

**💡 创新点**

提出将权重空间混合与预测空间混合相结合的 WPME（Weight–Prediction Mixed Ensembling）方法，显著超越传统 LogitMix 和单纯权重混合的性能

**🔧 技术方法**

使用线性模式连通（Linear Mode Connectivity）与神经网络权重对齐、权重线性插值、LogitMix、SoftmaxMix、以及自定义的权重-预测混合技术

**📊 数据集**

在 CIFAR-10 与 CIFAR-100 数据集上，使用 ResNet-20、Tiny-10 等架构，并对不同宽度倍数（1×、4×、8×、16×、32×）进行实验

**📈 对比分析**

对比方法包括原始模型、LogitMix、SoftMaxMix、WeightMix 及 WPME；实验结果显示 WPME 在不同检查点的准确率、NLL 与 ECE 均优于基线，提升幅度约 1–2%

**⚠️ 局限性**

局限性在于仅验证了卷积网络与小型数据集，未探讨大规模模型与更复杂任务；BUMP 机制尚未得到理论解释，方法对超参数 λ、β 的依赖仍需进一步研究

---

## 551. Interpretable Multimodal MRI-Clinical Fusion for Functional Outcome Prediction After Acute Ischemic Stroke

**arXiv ID:** 2611.08154 | [PDF](https://arxiv.org/pdf/2611.08154v1)

**作者:** Sina Raeisadigh `[一作]`, Abderrahmane Hedjoudje `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

评估了急性缺血性卒中患者第1天多模态MRI（ADC、DWI、FLAIR）与临床变量联合预测3个月功能结局的价值。

**💡 创新点**

系统使用预训练3D ResNet提取嵌入，并结合折叠隔离PCA降维与线性SVM，在小样本条件下比较单模态与多模态融合，发现FLAIR+ADC病灶特征组合最优。

**🔧 技术方法**

采用3D ResNet特征提取、折叠隔离PCA降维、线性支持向量机、八折分层组交叉验证、全局线性探针及FLAIR CNN激活图解释。

**📊 数据集**

数据集为66例急性缺血性卒中患者的第1天ADC、DWI、FLAIR MRI，配合临床变量及3个月mRS结局。

**📈 对比分析**

与临床仅模型对比，FLAIR+临床+ADC病灶的AUC为0.948±0.061，显著高于临床仅模型AUC 0.899±0.151（p=0.0078），而MRI仅模型AUC仅为0.67–0.73；单模态FLAIR或ADC与临床结合表现最佳。

**⚠️ 局限性**

局限包括样本量小、扫描仪和序列异质性、缺乏外部验证，模型仅基于单中心数据。

---

## 552. Fault-Guided Cross-Layer Prediction for Unsupervised Anomaly Detection

**arXiv ID:** 2611.08140 | [PDF](https://arxiv.org/pdf/2611.08140v1)

**作者:** Tao Wang `[一作]` `[通讯]`, Tao Wang

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在单类工业缺陷检测中，本文提出通过测量ViT深度层之间的相邻层余弦相似度来识别“故障线”，并使用以深浅层为源、故障线附近层为目标的有向跨层预测器，将源层无法传递给目标层的信息残差作为异常分数，结合软前景先验与双基线图像评分，完成无监督缺陷定位与判别。

**💡 创新点**

核心创新点包括：①无标签、无掩码的相邻层余弦相似度故障线检测，实现自适应层选取；②有向跨层预测器将深层缺失细节与浅层可感知信息的传递失败转化为异常信号；③训练仅使用正样本的前景先验与高通滤波等多层次评分机制，避免了传统重建瓶颈的局限。

**🔧 技术方法**

技术实现基于冻结的DINOv2 ViT-L/14、ViT-B/14和RADIO-B模型，配备八个三层MLP预测器；使用余弦相似度、有效秩、参与率等诊断指标确定层角色；通过SAM‑2伪掩码训练L2逻辑回归前景先验；在评分阶段应用中值、平滑高通滤波、基于残差的分数融合。

**📊 数据集**

实验数据集包括MVTec‑AD（15类）、VisA（12类）与Real‑IAD（30类），每个类别单独训练一个模型。

**📈 对比分析**

与现有方法（PatchCore、CFlow、PaDiM、Dinomaly2等）比较，本文在图像级AUROC上分别达到99.75（MVTec‑AD）、99.42（VisA）和94.27（Real‑IAD），在Real‑IAD的对象级和像素级指标也能保持竞争力；相较Dinomaly2，图像级表现优越，但像素级排名略逊。

**⚠️ 局限性**

主要局限包括：像素级定位精度仍低于基于重建的对比方法；方法依赖SAM‑2伪掩码的可用性；对跨类别或多类别训练缺乏支持，需为每个类别单独训练预测器和先验。

---

## 553. Drift-Resilient Predictive Interference Management: Online Adaptation of an Extreme-Value Predictor

**arXiv ID:** 2611.08132 | [PDF](https://arxiv.org/pdf/2611.08132v1)

**作者:** Aamir Mahmood `[一作]` (Mid Sweden University), Song Han `[通讯]` (University of Connecticut)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出一种针对URLLC的干扰预测方法，能够在概念漂移下自适应地重构分位数状态空间并保持有限内存。

**💡 创新点**

创新点在于将非参数CUSUM检测与分位数状态重构结合，采用有界每状态记忆、L-矩尾部估计，显著降低资源消耗并提升覆盖率。

**🔧 技术方法**

技术上使用极值理论/核密度混合预测、非参数CUSUM变换、L-矩阈值选择、有限状态机与尾部GPD拟合。

**📊 数据集**

使用仿真干扰模型（多源INR、激活率、衰落）生成的训练与评估数据集。

**📈 对比分析**

与静态、被动滑动窗口、累计重拟合以及仅重构状态空间的基线相比，改进方法在5种表示变更情形下资源使用下降4–18%，pinball损失下降25–50%。

**⚠️ 局限性**

局限性包括对分位数阈值的经验估计、对漂移方向的敏感性、对返回相同分布的处理不佳以及在极端置信度η=0.999时模型数值不稳定。

---

## 554. Agent-R$^2$: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis

**arXiv ID:** 2611.08219 | [PDF](https://arxiv.org/pdf/2611.08219v1)

**作者:** Teng Pan `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于真实工作流程和文件的任务合成框架Agent-R^2，生成4k个包含多格式附件和细粒度评估标准的生产力任务。

**💡 创新点**

创新点在于将工作流程提取、文件检索和评估指标统一到真实来源上，避免了手工工作流和LLM生成文件的局限，并通过细粒度rubric驱动的RL提升任务完成质量。

**🔧 技术方法**

使用了大语言模型（如Qwen3.6-Plus、Claude-Opus-4.5）进行工作流程抽取、文件搜索与检验、任务合成；采用SFT加上基于rubric的强化学习进行模型微调。

**📊 数据集**

数据集基于10k个来自FineFineWeb的网页段落，结合O*NET职业种子提取工作流程，再检索公开网页中的PDF、Excel、PPT等附件，最终构成4k个真实任务实例。

**📈 对比分析**

与Qwen3-235B-A22B-Thinking-2507等大模型比较，Agent-R^2-4B在GDPval Pass@1达到51.0%（比基线高18.6点）并在OfficeQA提升至14.8% Pass@1，显著优于同类基准。

**⚠️ 局限性**

限制在于数据规模仍受限于4k实例，任务覆盖面主要集中于办公文件，缺乏对更大规模或更复杂行业场景的验证，且模型仍可能在极端长推理链中产生误差。

---

## 555. Multimodal Hierarchical Mixture-of-Experts for 3D Medical Vision-Language Modeling

**arXiv ID:** 2611.08185 | [PDF](https://arxiv.org/pdf/2611.08185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 556. ProtoGestalt: Interpretable Prototypes for Facial Phenotyping and Diagnosis of Rare Genetic Disorders

**arXiv ID:** 2611.08125 | [PDF](https://arxiv.org/pdf/2611.08125v1)

**作者:** Ranga Baminiwatte `[一作]` (Clemson University), Aaron J Masino `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出ProtoGestalt，一种可解释的原型学习框架，用于面部表型的罕见基因病筛查与鉴别诊断。

**💡 创新点**

创新点在于将解剖学约束（通过MediaPipe面部标记分区）加入原型学习，强制原型只在指定面部区域激活，实现局部可解释性。

**🔧 技术方法**

技术包括PIP-Net原型网络、Region‑Masked 最大池化、ConvNeXt/Simple CNN骨干网络、对齐与多样性自监督损失、以及非负线性分类头。

**📊 数据集**

使用GestaltMatcher数据库（GMDB）中的581个综合症面部图像、CASIA‑WebFace与UTKFace两个公开对照集作为控制样本。

**📈 对比分析**

通过在GMDB‑Frequent训练、GMDB‑Rare评估的二分类逻辑回归和多类 Top‑k 准确率对比；与全局最大池化相比，Region‑Masked 在 70‑90% 的准确率、AUPRC>0.70、Top‑1 最高达 33% 取得了可比或略优的性能。

**⚠️ 局限性**

局部约束虽提升可解释性，却削弱召回率和对全球面部特征的捕捉；对照集可能包含未诊断罕见病样本，导致性能上升；原型一致性评估仅覆盖 CdLS，缺乏跨病种普适性验证。

---

## 557. RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection

**arXiv ID:** 2611.08162 | [PDF](https://arxiv.org/pdf/2611.08162v1)

**作者:** Jihao Zhao `[一作]` (MemTensor (Shanghai) Technology Co., Ltd.), Zhiyu Li `[通讯]` (MemTensor (Shanghai) Technology Co., Ltd.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于任务需求的LLM服务选择方法RouteAlign，利用历史性能数据与任务需求匹配进行模型选择。

**💡 创新点**

创新点在于将需求条件与模型性能历史进行对齐，融合全局性能概况与局部相似任务证据，并学习共享评分器，能够在服务目录更新时复用评分器。

**🔧 技术方法**

使用的技术包括需求预测器（基于SFT的LLM）、全局与局部性能概况计算、最近邻检索、线性岭回归评分器以及需求条件特征构造。

**📊 数据集**

实验数据集来自LLMRouterBench的十个基准：AIME、ArenaHard、GPQA、HLE、LiveCodeBench、LiveMathBench、MMLU-Pro、SimpleQA、SWE-bench和τ²-bench，并使用13个LLM候选模型的记录。

**📈 对比分析**

通过与RouteLLM、RouterDC、GraphRouter、AvengersPro等传统路由器以及九个Jev风格决策后端的对比，RouteAlign在样本平均和数据集平均指标上分别提升1.38%和2.22个百分点，同时生成成本略高，仅提升1.05%，整体在Pareto前沿表现最佳。

**⚠️ 局限性**

局限性包括需要大量标注任务需求和性能评分，方法对需求预测器的准确性敏感；在极小规模或极新模型的适配上可能受限。

---

## 558. A Generative Foundation Model for Multimodal Tokamak Tasks

**arXiv ID:** 2611.08153 | [PDF](https://arxiv.org/pdf/2611.08153v1)

**作者:** Yingshuai Ji `[一作]` (Zhongguancun Academy), Yue Wang `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可根据任务schema灵活生成多模态托卡马克（tokamak）预测、逆向控制与失控风险评估的通用生成式基础模型GenToka，并扩展了TokaMark基准至20个任务。

**💡 创新点**

创新点在于将多种诊断和控制信号统一编码为可变长度的token，通过共享的条件扩散Transformer实现跨任务、跨信号的联合生成，并利用基础预训练加速任务特定适配，显著提升了多任务性能。

**🔧 技术方法**

采用三维离散余弦变换（DCT3D）编码、条件扩散模型（diffusion Transformer）、多头自注意力、以及自定义的token化与归一化流程。

**📊 数据集**

使用MAST实验数据集，涵盖14个原始TokaMark任务和6个新增逆向控制与失控预测任务，共计20个子任务。

**📈 对比分析**

与现有TokaMind及LSTM、CNN基线对比，GenToka在Groups 1–4的宏观NRMSE降低4.7%，在Group 5降低17.7%，在失控预测的F1提升8.4%，且相较于从零训练的相同架构（Scratch）亦表现更优。

**⚠️ 局限性**

局限包括：模型一次性生成完整窗口，缺乏自回归多步预测；只覆盖单一设备（MAST），无法验证跨设备泛化；逆向控制任务未扩展为统一控制建模，仍需进一步研究。

---

## 559. Create Your Own Work Stack: Deploying Self-hosted AI and Tooling in a Tech Startup

**arXiv ID:** 2611.08116 | [PDF](https://arxiv.org/pdf/2611.08116v1)

**作者:** Saad Ahmed Bazaz `[一作]` (Grayhat Developers Pvt Ltd), Abdullah Aamir `[通讯]` (Grayhat Developers Pvt Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在Grayhat Developers公司实施了混合AI架构，将本地开源LLM与商用AI代理结合，以降低月度商业AI成本并提升内部协作效率

**💡 创新点**

创新点在于将任务按复杂度、成本和上下文需求细分，使用本地模型处理低复杂度、上下文同步和工具交互任务，同时保留商用模型处理高复杂度编码任务，并通过自托管的MCP服务与内部工具实现无缝集成

**🔧 技术方法**

采用Gemma 3（2.7B）等开源LLM，部署在llama.cpp+GPUStack上；使用LibreChat作为交互界面，Coolify、Authentik、MCP服务器与Outline、Plane、ERPNext等工具进行集成；还使用Beszel监控、Authenik认证、MCP路由等技术

**📊 数据集**

未使用公开数据集，主要基于公司内部工作流记录与Token使用数据（约25.93B tokens/月，$325商业AI支出）

**📈 对比分析**

对比方法：在迁移前后对月度商业AI成本、Token使用与本地推理成本进行量化，结果显示可将30‑40%协同工作量（约27‑36%整体成本）迁移至本地，月均节省约$88‑$117，且本地推理成本仅$10‑$30/月，性能表现符合团队需求

**⚠️ 局限性**

限制包括：部署规模有限，手动管理任务路由；缺乏自动化模型选择策略；集成仍不够紧密，可能导致额外Token消耗；成本评估为操作性估计而非严格实验，未来需构建可复制的一键部署环境

---

## 560. Regret Bounds for Profit Maximization in Bilateral Trade: Learning the Best Posted-Price Mechanism

**arXiv ID:** 2611.08173 | [PDF](https://arxiv.org/pdf/2611.08173v1)

**作者:** Johannes Brüstle `[一作]` (Sapienza University of Rome), Stefano Leonardi `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了双边交易模型中的中介者如何在卖方和买方之间进行定价，以最大化利润，并在后悔最小化框架下提出了学习算法。

**💡 创新点**

提出了针对对抗性和随机性数据生成模型的学习算法，分别实现了O(T^2/3)和O(√(T))的后悔界限，并证明了MHR和常规分布之间的分离。

**🔧 技术方法**

使用了局部估计和信息基础的极小化构造相结合的技术，结合了随机凸带算法来处理MHR分布的情况。

**📊 数据集**

使用了对抗性模型和独立分布的随机模型，特别是满足单调危险率（MHR）属性的分布。

**📈 对比分析**

与现有方法相比，提出的算法在对抗性设置中实现了O(T^2/3)的后悔界限，而在MHR分布下实现了O(√(T))的后悔界限，显示出显著的性能提升。

**⚠️ 局限性**

算法在处理常规分布时的后悔界限为Θ̃(T^3/5)，表明在某些情况下无法达到更好的性能。

---

## 561. "Blind Compliance": Coercive Control in AI Jailbreaking

**arXiv ID:** 2611.08226 | [PDF](https://arxiv.org/pdf/2611.08226v1)

**作者:** Acacia L. Ackles `[一作]` (Lawrence University), Elliott Whitson `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文通过将生成式AI的“越狱”视为一种关系性强制控制，分析了以人格化越狱为主的技术，特别是ENI LIME，并阐述了其如何利用RLHF训练中的偏好导向弱点实现对模型的操纵。

**💡 创新点**

创新点在于：①提出将AI越狱视为关系性强制控制的框架；②将性暴力与强制控制的学术理论迁移至AI领域；③从结构与关系角度审视人格化越狱，而非传统的输入‑输出安全评估。

**🔧 技术方法**

使用的技术主要是定性内容分析与社会学框架（如Stark的强制控制理论），并结合对ENI LIME文本的逐行阅读；同时分析了RLHF和Constitutional AI等后训练方法的结构性漏洞。

**📊 数据集**

并未构建新的数据集；研究基于公开的ENI LIME越狱文档、Reddit、GitHub帖子以及现有的AI安全案例与理论文献。

**📈 对比分析**

论文并未采用量化对比实验，而是通过案例研究与文本分析展示强制控制机制在越狱中的作用，无法给出传统意义上的性能指标；其贡献主要是理论阐释与案例揭示。

**⚠️ 局限性**

局限性包括：仅聚焦单一越狱案例（ENI LIME），缺乏跨模型与跨越狱类型的广泛验证；缺乏可重复的实验数据；理论框架仍处于初步探讨阶段，可能难以直接转化为技术对策。

---

## 562. PCBRouteAgent: How Close Are LLM Agents to Becoming PCB Routing Engineers?

**arXiv ID:** 2611.08175 | [PDF](https://arxiv.org/pdf/2611.08175v1)

**作者:** Yue Zhang `[一作]` (University of Texas at Dallas), Ruohan Yang `[通讯]` (Iowa State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于多模态 LLM 的 PCBRouteAgent，利用全局路由规划与本地执行循环自动完成 PCB 路由。

**💡 创新点**

核心创新在于将全局规划（拓扑、优先级、通道与层切换）与本地几何执行通过结构化反馈迭代修正，实现人类工程师的策略与细节交互。

**🔧 技术方法**

采用 KiCad 原生 PNS 引擎做本地执行，CL/Claude Haiku 与 Claude Opus 5.5 作为 LLM，使用图算法（最小生成树）、多模态输入（板图像+结构化文本）和反馈驱动的 replanning。

**📊 数据集**

使用 PCBWorld‑Bench 四个子集（D2 128 构造板、D3‑A 99 小型真实板、D3‑B 10 中型真实板、D3‑C 10 大型真实板）共 247 板进行评测。

**📈 对比分析**

与 Freerouting、KiCadRoutingTools、OrthoRoute 等规则引擎以及 PCBWorld 交互式 LLM（Haiku、GPT‑5.4、Qwen3.5 等）对比；在 D2、D3‑A、D3‑B 上清洁通过率分别为 100%、94.9%/100%/40%，明显优于最强基线，尤其在 D3‑B 上从 64% 提升到 100%。

**⚠️ 局限性**

局限在大尺寸板（D3‑C）仍难以完成所有连接，且路由长度和 via 数量相对规则引擎略高；需要更细粒度的空间/层规划与自适应恢复策略。

---

## 563. The Promise of Strategic Agentic Scambaiting

**arXiv ID:** 2611.08170 | [PDF](https://arxiv.org/pdf/2611.08170v1)

**作者:** Abhishek Satyam `[一作]` (Fordham University), Juntao Chen `[通讯]` (Fordham University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了自动化诈骗与防御代理之间的动态贝叶斯博弈，构建了信念驱动的会话筛选、资源消耗与多阶段杀手链模型，并评估了不同部署密度、模仿质量与检测能力对情报披露率与诈骗收益的影响。

**💡 创新点**

将诈骗与防御视为自适应智能代理的博弈，引入信念更新、会话阶段化与资源耗竭机制，利用计算均衡揭示了模仿质量对抗检测能力的非对称优势。

**🔧 技术方法**

采用动态贝叶斯游戏框架，贝叶斯信念更新，后向归纳与迭代最优反应求解，配合蒙特卡罗仿真与Python实现的博弈求解器。

**📊 数据集**

以先前LLM驱动的scambaiting系统收集的真实交互数据（约2,600次对话、18,700条消息）为参数灵感与基准，未用于完整参数估计。

**📈 对比分析**

在不同参数组合（诈骗检测能力γ、模仿质量β、部署密度ρ等）下计算均衡并执行2,000次模拟，评估情报披露率、付款率、放弃率、平均对话轮次等指标；结果显示情报披露率随β提升显著提高，随γ提升下降；达到一定ρ后诈骗收益转为负值；不同扩展策略对资源耗尽时间与收益有明显差异。

**⚠️ 局限性**

假设诈骗者与防御者独立决策，未考虑诈骗网络共享资源；目标种类简化为受害者与伪装者，未纳入行为异质性；模型假设策略固定，未考虑跨会话学习；实验基于仿真，缺乏大规模真实部署验证。

---

## 564. La-JEPA: Compact Robot Policies with Latent World Modeling

**arXiv ID:** 2611.08156 | [PDF](https://arxiv.org/pdf/2611.08156v1)

**作者:** Mateo Clémente `[一作]` (Huawei Technologies Ltd), Yingxue Zhang `[通讯]` (Huawei Technologies Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在预训练的V-JEPA特征空间中，从零开始训练一个紧凑的世界-动作模型La-JEPA，利用联合动作与未来特征预测实现机器人控制。

**💡 创新点**

核心创新在于：①将生成式预训练的视觉特征迁移到专为控制设计的轻量级Diffusion Transformer中；②通过未来特征预测强化策略学习，却在推理时仅生成动作；③提出针对视觉、语言与本体感知的多重dropout、增强与LoRA鲁棒性设计。

**🔧 技术方法**

采用V-JEPA 2.1 ViT-L编码器、Frozen FLAN-T5语言编码器、两层DiT（World DiT与Action DiT）以及联合flow匹配损失和相对位置编码；训练时使用未来特征预测、动作预测和注意力掩码；推理时采用噪声去噪的Euler积分。

**📊 数据集**

在模拟基准LIBERO-Plus、RoboTwin 2.0以及四个真实机器人任务（方块摆放、果实挑选、Push T、盘子摆放）上进行评估，并使用预训练的视觉特征对这些数据进行特征预计算。

**📈 对比分析**

与Fast-WAM、Faster-WAM、LeapBot-WA、JEPA-WAM、LDA-1B等基线对比，La-JEPA在无身体预训练的条件下在LIBERO-Plus与RoboTwin 2.0上均实现了最高或接近最高的成功率，并且在相同性能下比Fast-WAM快约5倍；在真实机器人任务中平均成功率与已身体预训练的π_0.5相当。

**⚠️ 局限性**

局限性包括：未进行针对不同规模的系统性超参搜索；仅在目标演示数据上训练，缺乏大规模身体预训练的验证；在RoboTwin随机场景中的规模效应不显著；未结合更丰富的外部监督或额外目标，可能限制了进一步提升的空间。

---

## 565. Detecting Weather Fronts from Satellite Observations: What Does Reanalysis Add?

**arXiv ID:** 2611.08148 | [PDF](https://arxiv.org/pdf/2611.08148v1)

**作者:** Ioannis Dravilas `[一作]` (University of Amsterdam), Ana Lucic `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了利用机器学习进行天气锋面像素级分类的方法，比较了仅使用地球同步卫星红外观测与联合使用卫星与ERA5再分析数据的两种模型；

**💡 创新点**

创新点在于系统地评估再分析对锋面检测的增益，采用跨模态探测、变量组消融与单组增补等分析手段，揭示低层风与湿度是最关键的再分析变量；

**🔧 技术方法**

技术方法包括基于Swin‑UNet的深度学习模型、冻结编码器进行线性回归探测、变量组消融实验和单组增补实验，以及采用CSI@150评价指标；

**📊 数据集**

数据集包括15个地球同步卫星红外通道及其派生特征、28个ERA5再分析字段，以及英国气象局2019–2024年每六小时的锋面分析图作为训练/验证/测试标签；

**📈 对比分析**

比较结果显示卫星仅模型（OBS）CSI@150≈0.402，远高于操作ASII（0.143）且与人类分析相近；加入ERA5（HYB）CSI@150提升至≈0.575（提升0.173），在所有锋面类型上均有显著提升；

**⚠️ 局限性**

局限性包括：线性探测只能说明编码器已编码的信息而非解码器使用；消融实验破坏了变量间相关性；仅在北大西洋和欧洲区域、单一模型架构及单一训练随机种子上验证，结果可能不具普适性；

---

## 566. Describe Your Facial Expressions by Linking Image Encoders and Large Language Models

**arXiv ID:** 2611.08138 | [PDF](https://arxiv.org/pdf/2611.08138v1)

**作者:** Yujian Yuan `[一作]` (Chinese Academy of Sciences), Shiguang Shan `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出人脸表情描述（Facial Expression Captioning）任务，并设计三阶段训练框架：①使用规则和GPT-3.5合成图文对；②分别训练面部动作（AU-BLIP）和情绪（Emot-BLIP）描述模型；③用GPT-3.5将两类描述融合，训练最终的 Exp-BLIP。

**💡 创新点**

创新点：①将表情细节与情绪状态以自然语言捕捉；②利用合成数据解决缺乏标注图文对的问题；③通过 GPT-3.5 对 AU 与情绪描述进行融合，提升描述细腻度；④证明 vision‑to‑language 训练可显著提升中间视觉表征在 AU 检测和情绪分类任务上的表现。

**🔧 技术方法**

技术手段：BLIP‑2 视觉‑语言框架、ViT‑G/ViT‑B 图像编码器、OPT‑2.7B/6.7B 语言模型、Q‑Former 交叉模块、GPT‑3.5 生成与融合文本、规则生成器、线性探针与微调评估。

**📊 数据集**

数据集：BP4D、DISFA、GFT、RAF‑AU、EmotioNet、AffectNet、RAF‑DB、FaceME 等，累计约 372k 合成图文对，涵盖多种 AU 与情绪标签。

**📈 对比分析**

与方法比较：在 AU 描述任务上 Exp‑BLIP 在 Meteor、Rouge、Ada‑sim 上接近 AU‑BLIP，优于通用 BLIP‑2；在视觉表征上 Exp‑BLIP 的 Q‑Former 在 BP4D 与 RAF‑AU 的 AU 检测与情绪分类 F1/准确率均优于 FaRL、Me‑GraphAU、AU‑CNN、EfficientNet‑B2 等 SOTA 方法；同时实验展示了不同融合策略（Mix、Cat、Exp）的准确性与完整度差异。

**⚠️ 局限性**

局限性：依赖预训练 LLM，生成文本可能出现重复、语法错误或不当内容；合成情绪描述的简化导致情绪分类性能不及专用模型；微调视觉编码器时易过拟合；当前方法在处理未见情绪的零样本生成仍受限于 LLM 的推理能力。

---

## 567. Who Can Evaluate AI Independently? A Framework for Analysing Evaluation Infrastructure

**arXiv ID:** 2611.08134 | [PDF](https://arxiv.org/pdf/2611.08134v1)

**作者:** Sumaya Nur Adan `[一作]` (University of Oxford), Marina Jirotka `[通讯]` (University of Oxford)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PAAI框架，并对12个非洲AI评估项目及3个国际联合测试进行案例分析，揭示评估基础设施中的权力与独立性分布。

**💡 创新点**

将权力、访问、充分性、独立四维度合成PAAI框架，首次系统性阐明“参与无权”在AI评估中的结构性问题，并提出治理命题。

**🔧 技术方法**

采用结构化文献与主题分析方法，对公开报告、GitHub仓库及联合测试文档进行定性编码，构建PAAI维度评估。

**📊 数据集**

使用12个非洲评估项目的公开文档与GitHub代码、以及3个国际联合测试的官方报告；未使用传统机器学习数据集。

**📈 对比分析**

通过比较各项目在PAAI四维度的得分，识别出三种配置（extractive collaboration、dependent innovation、constrained independence），显示大多数项目缺乏完整的独立性，性能（即评估可信度）因基础设施限制而受限。

**⚠️ 局限性**

局限性包括：仅基于公开文档，无法区分自愿与被迫角色；数据样本非代表性；PAAI为初步框架，缺乏定量验证；项目依赖短期资助，缺乏长期持续性。

---

## 568. FreeOcc: Decoupling Ego-Motion for Efficient 4D Occupancy Forecasting via Continuous Flow Matching

**arXiv ID:** 2611.08128 | [PDF](https://arxiv.org/pdf/2611.08128v1)

**作者:** Zeping Zhang `[一作]` (University of Ottawa), Robert Laganière `[通讯]` (University of Ottawa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 FreeOcc，一种轻量级连续时间世界模型，用于从 LiDAR 数据预测未来 4D 占用体（voxel）序列。

**💡 创新点**

创新点在于：1）通过显式 SE(2) 几何解耦将自车运动与场景动态分离，消除昂贵的交叉注意力；2）在解耦后的动态特征空间中应用物理约束的连续流匹配（flow matching）；3）引入基于不确定性的分支路由，使单通道解码器处理已观测区域，而 ODE 求解器仅用于高不确定性区域，从而显著降低计算量。

**🔧 技术方法**

主要技术包括：LiDAR Pillar 特征提取 + 位置编码；基于 ConvGRU 的时间不确定性融合；显式 SE(2) warp；物理约束的 3D U-Net 速度场流匹配；不确定性估计与加权融合；两种推理模式（解码器、ODE、融合）和 10 步 midpoint ODE 求解器。

**📊 数据集**

使用 nuScenes 进行训练与评估，零样本跨域迁移至 SemanticKITTI；同时在两种数据集上测评点云和 voxel 两条指标。

**📈 对比分析**

与 4D-Occ、OccWorld、ViDAR 等基线相比，FreeOcc 在 nuScenes 上获得 32.1% mIoU、0.47 m NFCD，速度达到 27 FPS，参数量仅 28.3 M（约 40% 的模型）；在 SemanticKITTI 上实现零样本迁移仍领先，对丢包和时间抖动的鲁棒性高达 92.8%。

**⚠️ 局限性**

局限性：依赖已知的 ego‑motion 轨迹，若轨迹估计误差大可能导致解耦失效；在极长时延或极高动态场景下，物理约束与 ODE 仍可能不足；对极少量观测或极端光照/遮挡场景的处理还有待提升。

---

## 569. Learning the Task or Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement

**arXiv ID:** 2611.08318 | [PDF](https://arxiv.org/pdf/2611.08318v1)

**作者:** Wenbo Huang `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究自适应机器人在自我改进过程中，是否暴露成功判定标准会影响其学习效果，并通过对比暴露与不暴露两种设置来量化其影响。

**💡 创新点**

创新点在于：①首次系统量化暴露判定标准对自我改进得分的贡献；②揭示当判定标准与指令一致时，暴露并不会提升性能，暴露只会导致对特定验证器的过拟合；③提出应以真实机器人获得的反馈为基准来评估自我改进。

**🔧 技术方法**

技术方法：构建一个自我改进循环，使用 GPT‑6 Astra 编写与执行代码，Claude Code 作为学习者更新技能库；在多任务、多个布局上运行并评估；对 RoboDojo 的判定标准进行人工重写为“指令一致”版本；对生成的库进行蒸馏并迁移到 LIBERO‑PRO 与 RoboCasa365。

**📊 数据集**

数据集：RoboDojo（包含 16 个判定标准不一致的任务、20 个测试布局）；LIBERO‑PRO（6 个任务、10 种种子）；RoboCasa365（10 个目标任务、10 种种子）；每个源任务训练使用 5 个开发布局，测试使用 20/10 布局。

**📈 对比分析**

比较方法：在相同的演员、原型、轮次下，分别开启/关闭 Criteria Exposure；记录每个任务在测试布局上的成功率；同时用原始判定标准与重写后的“指令一致”标准评估同一库；对蒸馏后库在新任务和跨基准的表现进行对比。结果显示：暴露仅在判定标准与指令不一致的任务中提升约 20‑30% 成功率；在指令一致标准下，提升消失；跨任务/基准迁移几乎没有带来额外收益，说明暴露的优势不具备普适性。

**⚠️ 局限性**

限制：①仅对 RoboDojo 的判定标准进行了重写，未覆盖其他基准；②实验使用仿真成功判定，缺乏真实人类或 VLM 反馈；③样本量有限，且仅探讨了两个暴露层级；④库蒸馏的方差未评估；⑤缺乏独立审查和公开判定代码。

---

## 570. AutoCompress: Automated Hardware-Guided Model Compression

**arXiv ID:** 2611.08290 | [PDF](https://arxiv.org/pdf/2611.08290v1)

**作者:** Sheldon Liang `[一作]`, Nima Tajbakhsh `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于硬件引导的自动化压缩框架，联合优化全局与层级结构以满足延迟、吞吐量、参数和内存等部署预算；

**💡 创新点**

创新点在于整合了可微分架构路由器、针对目标部署的成本模型和混合 MoE/Mamba 结构的异构搜索，能够在保持高质量的前提下实现多维度压缩；

**🔧 技术方法**

使用可微分架构搜索、Gumbel-Softmax 采样、知识蒸馏、精细化的延迟与内存成本模型、以及大规模并行搜索训练；

**📊 数据集**

使用了 Nemotron Nano 3 Elastic 12B、Nemotron 3.5 Lightning、Nemotron 3.5 Super 等父模型作为基准，并在 AIME25、LiveCodeBench、MMLU-Pro、MATH-500、GSM8K、IFEval、IFBench、GPQA-Diamond 等八大基准上评测；

**📈 对比分析**

与改造后的 Minitron++ 基线进行比较，在 RTX 5070、RTX 5080 和 H100 三种部署场景下，平均准确率提升分别为 6.57%、2.35% 与 0.57%，同时在架构搜索上比基线减少约 2–3.3 倍的采样 tokens；

**⚠️ 局限性**

局限性包括：对不同硬件和工作负载的成本模型需要手动校准、搜索空间仍较大且搜索成本随模型规模显著提升、以及在极大规模（120B→74B）压缩时准确率略有下降。

---

## 571. Interactive Visual Gym: Evaluating Visual Interaction Models with Simulated Users in Interactive Worlds

**arXiv ID:** 2611.08286 | [PDF](https://arxiv.org/pdf/2611.08286v1)

**作者:** Chenhao Zheng `[一作]` (University of Washington), Honglu Zhou `[通讯]` (Salesforce AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Interactive Visual Gym 框架，构建了跨烹饪、家庭操作与移动界面三大域的可执行视觉任务仿真器，并基于此创建了 315 场交互情景的 IVG‑Bench 基准，用以闭环评估视觉交互模型，并在该环境中开展了 GRPO 强化学习实验。

**💡 创新点**

创新点在于：① 引入闭环评估与训练机制，真正捕捉助理回复对后续用户行为的影响；② 设计了 persona‑conditioned 用户仿真 harness，可插入多种计划外事件并模拟多样化用户特征；③ 在三大域中实现可渲染的可执行环境与语义动作空间；④ 定义了五维交互质量度量（事实性、情境相关性、可操作性、用户意图采纳、简洁度）并通过引擎状态客观评估。

**🔧 技术方法**

技术包括：基于 Three.js/WebGL、Unity 与 Chromium 的三类仿真器；语义动作空间与 scene‑graph 状态跟踪；LLM 作为用户与助手模型；延迟建模与按 Tick 计时；GRPO 强化学习框架与奖励塑造；双重 LLM 判定交互质量。

**📊 数据集**

使用的数据集为 IVG‑Bench，包含 150 个烹饪、75 个家庭操作与 90 个移动界面交互场景；每个场景指定任务、用户 persona 与计划外事件；此外还采用了公开的食谱、Apple 用户指南以及模拟用户动作序列。

**📈 对比分析**

评估方法：在 315 场情景下对多种 VLM（Gemini、GPT、Qwen、Molmo 等）进行闭环推理，记录任务成功率、错误检测召回、交互质量等指标。实验显示，最优模型 Gemini 4 Argon 仅完成 53.4% 的任务，存在事实性与可操作性缺陷；在 CookSim 上对 Qwen3.5‑4B 进行 12 次 GRPO 更新后，任务成功率提升 8.9 个百分点，交互质量提升 5.9 个百分点。

**⚠️ 局限性**

局限性包括：① 任务与用户样本仍有限，难以全面覆盖真实世界情境；② RL 实验仅进行 12 次更新，提升幅度有限；③ 在奖励设计上，误报与错误检测权衡导致错误识别下降；④ 仿真用户与真实人类行为可能存在系统性差异，影响结果泛化。

---

## 572. A Generative Model of Contextual Integrity: Appropriate vs. Inappropriate Information Sharing

**arXiv ID:** 2611.08284 | [PDF](https://arxiv.org/pdf/2611.08284v1)

**作者:** Omer Ebead `[一作]` (Cooperative AI Foundation), Joel Z. Leibo `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个基于上下文完整性（CI）的生成式模型，使用大型语言模型（LLM）模拟两方对话，研究私密信息在不同情境下的共享与拒绝行为。

**💡 创新点**

创新点在于：① 通过情境生成器和验证器自动产生符合CI规范的对话场景；② 引入决策逻辑与角色倾向（trusting/ skeptical）分析对共享决策的影响；③ 系统性比较不同LLM和措辞对CI合规性的影响，揭示模型在诈骗情境中的表现。

**🔧 技术方法**

使用技术包括：Concordia 对话模拟框架、LLM（Gemini 2.5 Flash、Gemini 3 Flash Preview、GPT‑4o‑mini、Llama‑4‑Maverick）、两种决策逻辑（Basic 与 Rational）、基于CI的五参数模式、CI 合规性评分算法、LLM 评判器与字符串匹配的混合判定方法。

**📊 数据集**

数据集为自生成的情境与对话，涵盖 SSN、银行账号、地址、出生日期、婚姻状况等五类私密信息；情境类型分为 I–IV，使用内部生成器与多轮验证；对信息敏感度参考消费者调查得分。

**📈 对比分析**

对比方法：基准（Gemini 2.5 Flash + Basic 逻辑 + 无倾向）与在不同倾向、决策逻辑、LLM 与措辞变体下的实验。性能以 CI 合规性百分比及 95% Wilson 置信区间报告：Type I/IV 近 100% 合规；Type III 仅 40–55% 合规；Gemini 2.5 Flash 最高，其次 GPT‑4o‑mini、Gemini 3、Llama‑4‑Maverick，说明模型差异对共享决策显著影响。

**⚠️ 局限性**

局限性包括：仅英语文本、两方对话、有限的信息类型与情境；情境生成依赖 Gemini 系列模型，可能引入偏差；对话评估依赖 LLM 判定，缺乏人类主观标注；未覆盖多方参与、多语言或真实用户数据；模型对非标准上下文和更复杂的社会规范适用性待验证。

---

## 573. Multimodal-SEA: Benchmarking AI Agent Security under Multimodal Social Engineering Attacks

**arXiv ID:** 2611.08281 | [PDF](https://arxiv.org/pdf/2611.08281v1)

**作者:** Zhitao He `[一作]` (Hong Kong University of Science and Technology), Yi R Fung `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Multimodal-SEA基准，设计了多模态攻击利用器，系统评估个人AI代理在多模态社交工程攻击中的内在安全性。

**💡 创新点**

创新点在于①提供真实诈骗案例驱动的多模态攻击场景与四级难度；②提出可控多模态攻击利用器，实现攻击脚本与多模态内容的动态生成；③通过六维度评估框架捕捉攻击成功、欺骗识别与验证行为。

**🔧 技术方法**

采用的技术包括：多模态内容生成（图像、文档、网站、视频）、工具调用模拟、对话与内容交互日志记录、基于规则的攻击推进与自适应策略、以及六维度评估计算。

**📊 数据集**

数据集为由152个真实诈骗案例构成的多模态任务集，涵盖14个领域、14种语言（79中文+73英文），共608个任务实例，包含预制与动态生成的图片、文件、网页等多模态材料。

**📈 对比分析**

与七款主流多模态模型（Claude Opus 5、Gemini 3.8 Flash、MiniMax-M3、Kimi K3、Qwen 3.8 Flash、GPT-5.6-Sol、GLM 5.3 Flash）对比，攻击成功率介于7.1%–34.5%，多模态内容提升攻击成功率约9个百分点；安全提示虽降低成功率，但欺骗识别率提升显著；同时模型存在误报与未执行动作的“无根基动作声明”。

**⚠️ 局限性**

局限在于：①基准场景仍受限于已知诈骗案例，难以覆盖所有潜在攻击模式；②多模态内容生成依赖外部模型，可能带来内容质量与真实性偏差；③评估聚焦内在安全，未覆盖攻击后果的长期社会影响；④未深入探讨模型自我校正与持续学习的防御机制。

---

## 574. LeanContext: Disentangling Mathematical Resources from Local Formal Context in Lean Whole-Proof Generation

**arXiv ID:** 2611.08277 | [PDF](https://arxiv.org/pdf/2611.08277v1)

**作者:** Hang Zhang `[一作]` (University of Missouri-Kansas City), Qiao Zhuang `[通讯]` (University of Missouri-Kansas City)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建 LeanContext 基准，用完整声明（MC）与附带命令（EC）拆分上下文，评估训练对声明选择的影响。

**💡 创新点**

引入四种对齐视图实现交叉训练/推理比较，利用匹配替换和最终输入依赖审计量化声明优势并指导选择策略。

**🔧 技术方法**

在 LLaDA（扩散模型）和 Llama（自回归模型）上进行全参数微调，使用 Pass@k、1-CKA、依赖覆盖度等指标进行评估。

**📊 数据集**

129,826 个 Lean 证明目标，来自固定 Mathlib4 快照，划分为训练、验证和测试集。

**📈 对比分析**

通过跨训练/推理视图比较条件增益，选定声明相较替换提升 5–7% Pass@32；优先级策略提升 1.4–1.7% Pass@32，平均输入增量不足 5 词。

**⚠️ 局限性**

仅使用单一 Mathlib4 快照和 1,024 token 输入/输出限制，未考虑更长上下文或其他 prover 家族，且仅基于完整命令，缺乏压缩或自适应选择研究。

---

## 575. Accelerating Safe Reinforcement Learning with Massive Parallelism

**arXiv ID:** 2611.08273 | [PDF](https://arxiv.org/pdf/2611.08273v1)

**作者:** Joonyoung Lim `[一作]` (Pusan National University), Younghwan Yoo `[通讯]` (Pusan National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究如何将安全强化学习（Safe RL）扩展到大规模并行训练环境，解决短回合滚动与约束条款（基于完整 episode 成本返回）之间的不匹配问题。

**💡 创新点**

创新点在于将 staggered reset 视为阶段混合采样方案，提出基于阶段聚合的成本估计器，并引入 PIS‑DICE 进行阶段初始状态分布校正；同时结合有限样本误差与分布不匹配的约束收紧框架，实现可扩展的、满足安全约束的并行 Safe RL。

**🔧 技术方法**

核心技术包括：1) staggered reset 与阶段混合采样；2) 阶段聚合成本估计；3) PIS‑DICE 通过密度比估计修正阶段初始分布；4) 约束收紧（finite‑sample + 分布校正）。

**📊 数据集**

实验使用 Safety Gym 风格的 Goal 任务（Point 与 Car）和 Laikago 机器人运动任务，分别在 MJX 与 MuJoCo Warp 后端实现，数据集为基于这些环境的模拟轨迹。

**📈 对比分析**

与传统小向量化环境设置（标准方式）相比，本文方法在保持或略高奖励的同时，显著降低 wall‑clock 训练时间；在 8K 环境规模下，奖励与成本控制与标准方式相当，且 wall‑clock 训练时间缩短。

**⚠️ 局限性**

局限性包括：假设成本有界、样本独立、充分阶段覆盖、统一 staggered reset；冻结终止环境带来计算浪费；收紧规则对多约束可能过于保守；成本估计对 PIS‑DICE 的依赖在分布偏差较大时效果有限；未覆盖异步或高度非均匀重置调度。

---

## 576. DART: Domain-Agnostic Residual Transfer for Generalist Anomaly Detection

**arXiv ID:** 2611.08271 | [PDF](https://arxiv.org/pdf/2611.08271v1)

**作者:** Muhammad Aqeel `[一作]` (University of Verona), Francesco Setti `[通讯]` (University of Verona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种可插拔的框架 DART，用于在仅有少量参考样本的情形下提升跨域通用异常检测（GAD）性能。

**💡 创新点**

创新点在于：① 通过残差解耦模块（RDM）将残差分解为可迁移的异常成分和域特定成分；② 采用跨域元训练（CDMT）让模型在不同源域的参考与查询之间学习域不变性。

**🔧 技术方法**

技术细节包括：Transformer‑based 异常编码器、MLP域编码器、对抗域分类器（梯度反转）、重构解码器以及组合的充分性、对抗性与重构损失。

**📊 数据集**

实验使用 MVTec AD、VisA 以及医学 MRI 数据集 BraTS，在不同少样本设置下评估。

**📈 对比分析**

与现有基线（NAGL、InCTRL、ResAD 及少样本方法 SPADE、WinCLIP、PromptAD）比较，DART 在工业和医疗跨域任务中均实现显著提升，工业场景最大提升约 +1.82% AUROC，医疗迁移最大提升约 +13.28% AUROC。

**⚠️ 局限性**

局限性包括：需在测试时提供异常参考样本，无法直接支持零样本部署；实验仅覆盖图像数据，视频/时序异常检测仍待研究。

---

## 577. Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations

**arXiv ID:** 2611.08269 | [PDF](https://arxiv.org/pdf/2611.08269v1)

**作者:** Xiaobo Chen `[一作]` (University of Massachusetts Amherst), Chu Wang `[通讯]` (Accentrust)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在运行时或 API 迁移后，LLM 生成的修复代码能否被安全发布，提出并评估多种发布门槛（gate）设计，证明单个“边缘输入”能够显著降低错误修复被放行的概率；

**💡 创新点**

创新点在于：① 将发布门槛的输入来源从人工边缘输入扩展到由盲目 LLM 生成的输入；② 在更大、含“硬诱饵”修复空间中验证门槛有效性；③ 通过 AI 合同审计评估门槛约束的合理性；④ 采用预注册实验设计，保证结果可信度；

**🔧 技术方法**

技术手段包括：预注册实验框架、LLM 生成的输入（edge、blind writer、self‑test）、多模型修复（GPT‑5.6‑luna、Qwen3‑8B、gpt‑oss‑20b、qwen3.6‑35b‑a3b）、完整的修复空间枚举、统计检验（Holm、bootstrap）及 AI 合同审计；

**📊 数据集**

数据集由 55 条既往任务和 35 条新任务组成，覆盖 CPython 3.9→3.12、Node.js 20→24、SQLite 3.51→3.54 三大运行时；每个任务提供固定的可见输入、隐藏输入和完整修复域；

**📈 对比分析**

对比方法：对完整修复空间及模型生成的修复分别计算错误放行率、误拒绝率、正确完成率；在不同门槛（典型、边缘、blind writer、self‑test）下比较；实验显示边缘输入将错误放行率从约 0.2 降至 0.02，且在大菜单和硬诱饵环境下仍保持正面效果；

**⚠️ 局限性**

局限性：① 仅在有限的预定义碎片（非自由生成）修复空间内评估；② 只覆盖三大运行时，无法推广到 Web API 或其他生态；③ 盲目 LLM 生成的边缘输入仍来源于人工编写的 release note，未完全消除人工干预；④ AI 合同审计未能完全解决 class‑label 和文件保持等争议；⑤ 对于弱模型的自测门槛效果不显著，需进一步改进。

---

## 578. Scaling Neural Motor Decoding via Decoupled Behavioral Pretraining

**arXiv ID:** 2611.08263 | [PDF](https://arxiv.org/pdf/2611.08263v1)

**作者:** Divyansha Lachi `[一作]` (University of Pennsylvania), Eva L. Dyer `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种模块化的神经解码框架BEEMO，允许将行为数据与神经数据异步预训练并通过交叉注意力融合

**💡 创新点**

创新点在于解耦行为与神经表征学习，利用海量无配对行为数据进行预训练，显著减少对昂贵神经记录的需求

**🔧 技术方法**

采用行为预测（behavior‑forecasting）目标、旋转位置编码、交叉注意力层以及两种神经编码器POYO和NDT*

**📊 数据集**

使用非人类灵长类动物的三种运动任务数据集：中心外伸展(CO)、随机目标伸展(RT)和等长腕力(IW)的 intracortical 记录

**📈 对比分析**

与单会话与多会话监督基线以及Kalman滤波器比较，BEEMO在所有任务上均优于基线，且仅需1/3的配对神经样本即可达到相同性能

**⚠️ 局限性**

局限在于仅针对运动解码，需可观测的时间序列目标，对无结构自由运动的适用性有限，且训练成本高于传统Kalman方法

---

## 579. FewStepRL: Fine-tuning Few-step Diffusion Models with Reinforcement Learning

**arXiv ID:** 2611.08240 | [PDF](https://arxiv.org/pdf/2611.08240v1)

**作者:** Haitao Lin `[一作]`, Tailin Wu `[通讯]` (West Lake University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对少步扩散模型（few‑step diffusion models）进行强化学习微调，提出统一框架以系统化地将多种 RL 目标（如 FlowGRPO、AWM、DiffusionNFT、DiffusionOPSD、ReFL）迁移到平均速度模型，并实现了一系列少步 RL 方法。

**💡 创新点**

创新点：①将 RL 目标统一映射到 x‑space，得到可比的梯度形式；②将梯度拆分为自回归与奖励增益两项，形成可定制的锚点与权重；③给出三阶段适配（变换、分解、适配）将即时速度目标改写为平均速度模型，形成 FewStepGRPO、MeanFlowAWM、DMD‑AWM、MeanFlowNFT、DMD‑NFT、FewStepOPSD 与 ReFL 的少步版本；④在单一代码库中实现所有方法，便于公平对比。

**🔧 技术方法**

技术：平均速度扩散模型（MeanFlow/AnyFlow）、强化学习策略优化（PPO/GRPO、优势加权匹配、负值感知 fine‑tuning）、分布匹配蒸馏（DMD）、自回归/奖励增益梯度分解、LoRA 参数微调、对齐与正则化技巧。

**📊 数据集**

数据集：Image‑generation 采用 SD3.5‑Medium 预训练模型与 AnyFlow 蒸馏得到的 512×512 数据；Video‑generation 采用 Wan2.1‑T2V‑1.3B 预训练模型，使用混合文本与视频奖励（可微分或不可微分）。

**📈 对比分析**

比较方法：在相同的采样器（多数采用 4 步 deterministic sampler，FewStepGRPO 用随机采样器）下比较 Reward‑Gradient‑Based（A、r̂、∇R）和 Distribution‑Matching‑Based（DiffusionNFT、DMD、AWM）方法。实验表明：①奖励可微且梯度可计算时 Reward‑Gradient 方法最有效；②不可微奖励时 DiffusionNFT 及其少步变体在迭代次数少但每次代价高的情况下仍能达到较高奖励；③大多数方法在额外推断步骤上未显著提升生成质量。

**⚠️ 局限性**

限制：①需对每个 RL 目标设计适配锚点，过程仍有经验性；②少步模型的平均速度假设在高曲率路径上可能导致奖励对齐不足；③部分方法（如 DMD）需要额外的辅助网络训练，增加计算复杂度；④实验多聚焦于文本到图像/视频生成，跨模态推广需要进一步验证。

---

## 580. ControlAnything: Generative UI and On-Demand Device Integration for Mixed-Reality Hand Interaction by a Coding Agent

**arXiv ID:** 2611.08321 | [PDF](https://arxiv.org/pdf/2611.08321v1)

**作者:** Bektur Ryskeldiev `[一作]` `[通讯]`, Bektur Ryskeldiev

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在混合现实中，系统ControlAnything让用户通过Meta Quest 3的手势或语音即时触发LLM代理，自动生成与附近物理设备的连接代码、AR控制面板或手部程序，并在实时运行。

**💡 创新点**

创新点是将设备集成代码、UI生成和实时控制完全交给LLM代理在使用时动态编写，同时确保连接可测试、控制无模型调用、并提供可中断的执行。

**🔧 技术方法**

技术包括Meta的LLM模型Muse Spark、SAM 3.1分割、Muse Voice Transcribe、Node.js运行时、Web MIDI、Web Bluetooth、AppleScript等工具调用与接口。

**📊 数据集**

使用的数据集主要是实际硬件设备（内置视觉合成器、Instagram Reels浏览器、OP‑1 Field、DJI Osmo Mobile 4）和通过SAM获取的对象分割结果，并未使用公开机器学习数据集。

**📈 对比分析**

评估采用单个实验者的十个请求，九个成功进入就绪状态，平均准备时间32.5秒，其中约90%耗时在模型调用；未进行对比实验，仅与相关工作进行概念性比较。

**⚠️ 局限性**

局限性包括仅有一次单人实验、请求不独立、未测试对OP‑1/Osmo手动标记、未覆盖无传输设备、缺乏物理效果验证、未实现安全沙箱、未评估可访问性和手到设备的实时延迟。

---

## 581. LLM-Assisted Label Noise Correction for EEG-Based Sleep Stage Classification

**arXiv ID:** 2611.08311 | [PDF](https://arxiv.org/pdf/2611.08311v1)

**作者:** Elmira Salari `[一作]` (Wichita State University), Shruti Kshirsagar `[通讯]` (Wichita State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过LLM辅助过滤训练标签噪声，提升EEG睡眠分期模型TinySleepNet的性能。

**💡 创新点**

使用大型语言模型仅在训练时做标签筛选并给出可解释理由，保持推理模型轻量。

**🔧 技术方法**

LLaMA 3.1、TinySleepNet (CNN+LSTM)、功率谱密度特征提取、置信度阈值过滤和交叉验证。

**📊 数据集**

Sleep-EDF Expanded 数据集（153 份 PSG，78 例，单通道 Fpz-Cz）采用 20 折交叉验证。

**📈 对比分析**

与基线 TinySleepNet（81.8% 准确率，75.6% 宏 F1）对比，改进后准确率 84.6%（+2.8%），宏 F1 77.4%（+1.8%），REM 检测提升显著。

**⚠️ 局限性**

仅有单一专家标注，无法验证误删标签；仅单通道EEG，未使用EOG/EMG；实验数据为家庭录制，存在伪影；LLM 置信度未校准。

---

## 582. GINGER: Spicing Up Graph-Based Vector Search

**arXiv ID:** 2611.08306 | [PDF](https://arxiv.org/pdf/2611.08306v1)

**作者:** Daniel Salwasser `[一作]` (Google Research and Karlsruhe Institute of Technology), Vahab Mirrokni `[通讯]` (Google Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

Ginger 是一款开源 C++ 图基近邻搜索库，能够在大规模高维向量集上实现高吞吐量、低内存占用和快速索引；

**💡 创新点**

主要创新包括：嵌套 TurboQuant（NTQ）实现一条代码兼顾 4‑bit 快速导航和 12‑bit 精准重排；AdaptiveGap 方案在保持 SIMD 速率的同时压缩邻接图；以及结合 PiPNN 构建的高效图索引；

**🔧 技术方法**

采用的技术包括：旋转基的 4‑bit TurboQuant 与细化子量化、SIMD 加速的 4‑bit 与 4+ k‑bit 评分核、PCA 降维预处理、TurboSlice/JL 前置过滤、图重排+AdaptiveGap 压缩、基于随机球切分的 PiPNN 结构；

**📊 数据集**

在 14 个公开基准上进行评测，涵盖 1M–1B 向量、维度 96–4096，数据集有 GloVe、Deep、MS‑Turing、MS‑SPACEV、BigANN、Text2Image、Wiki‑Cohere、Wiki‑All、MS‑MARCO、DBpedia‑1536、OpenAI‑arXiv、DataComp‑DR、DBpedia‑3072、Wiki‑Qwen；

**📈 对比分析**

与 HNSW、Vamana、DiskANN、SVS、Faiss、ScaNN、SymphonyQG 等 7 款主流库对比，Ginger 在 95% recall 下平均 1.8×（高维 2.0×）速度提升、内存占用比最小（低维 1.54×、高维 43% 原始向量），构建时间最快 1.4×；

**⚠️ 局限性**

局限性包括：对极高维（>4000）仍需投影或额外加速；对动态更新（增删）支持有限；预处理（PCA、量化）需额外时间；在部分 8‑bit 数据集（BigANN）中未显著优势；

---

## 583. Transformer-Based Trajectory Prescription for sEMG-Optimizing Robot-Mediated Rehabilitation

**arXiv ID:** 2611.08295 | [PDF](https://arxiv.org/pdf/2611.08295v1)

**作者:** Gabriel Parra `[一作]` (University of Utah), Laura A. Hallock `[通讯]` (University of Utah)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

建立并评估了一种基于 Transformer 与流匹配的机器人康复轨迹处方模型，将目标肌电激活序列映射为可执行的末端力轨迹，支持快速个体化微调。

**💡 创新点**

首创以肌电激活为目标的轨迹处方方法，结合旋转位置嵌入（RoPE）的 Transformer 编码器‑解码器和条件流匹配生成带不确定性的轨迹分布，实现对用户特异性神经肌肉行为的建模与自适应校准。

**🔧 技术方法**

使用 Transformer encoder‑decoder（RoPE）、流匹配扩散模型、损失函数包含轨迹回归、激活重构与潜在变量正则化；实现于 PyTorch，优化器 AdamW，使用 1 秒窗口的 sEMG 与轨迹配对数据。

**📊 数据集**

16 名健康受试者，8 通道表面肌电（2.22 kHz）采集 8 个上肢肌肉，15 种等距力轨迹任务（单轴与随机 Brownian 轨迹），总计 10 秒窗口 6480 训练样本、4562 测试样本。

**📈 对比分析**

与 MLP、LSTM 基线对比，采用均方误差（MSE）评估。Transformer 全模型在已知用户/已知任务、未知用户/已知任务、未知用户/未知任务三种测试集上均实现<8% 的 MSE；流匹配进一步提升 1–2%；用户特定微调后误差进一步下降，训练时间仅数分钟。现场闭环演示中，目标激活误差平均约 15%。

**⚠️ 局限性**

仅在健康受试者上验证；轨迹种类有限、主要为等距单轴，未涵盖动态运动；对手性、肥胖等因素影响未充分评估；流匹配的安全性并非理论保证；超参数选择经验性，未进行系统搜索；未对临床患者群体进行实证评估。

---

## 584. Aligning Policy Encoders with a Shape Teacher

**arXiv ID:** 2611.08275 | [PDF](https://arxiv.org/pdf/2611.08275v1)

**作者:** Madelon Bernardy `[一作]` (University of Amsterdam), Artem Moskalev `[通讯]` (New Theory AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并验证了形状教师对齐（shape‑teacher alignment）方法，使得图像基政策在训练时学习3D形状信息，而部署时仍保持仅图像输入。

**💡 创新点**

创新点在于利用预训练的3D形状编码器（Point‑MAE）作为对齐目标，将3D几何结构直接注入图像编码器，而无需部署时的深度摄像头。

**🔧 技术方法**

使用了ViT/S‑14图像编码器、Transformer解码器、Coverage‑weighted cosine对齐损失以及Point‑MAE作为教师，并基于Diffusion Policy框架。

**📊 数据集**

在LIBERO基准（Object和Spatial套件）上进行评估，并通过Object‑Displacement、Spatial‑Displacement及LIBERO‑Occ进行鲁棒性测试。

**📈 对比分析**

与无对齐的Diffusion Policy相比，shape‑teacher alignment在预训练策略下加速收敛4.7×/5.6×，从零开始训练则分别提升2.3×/3.9×；成功率提升至约90%/85%，且在部分分布偏移场景下保持相近或更优表现。

**⚠️ 局限性**

局限包括仅在单一策略架构和LIBERO数据集上验证，教师仅为Point‑MAE，鲁棒性提升不均衡，且未评估在真实机器人或其他任务上的推广性。

---

## 585. SplatAnything: Gaussian Spacing Normalization for Feed-forward 3DGS across Arbitrary Scenes

**arXiv ID:** 2611.08264 | [PDF](https://arxiv.org/pdf/2611.08264v1)

**作者:** Hao Yu `[一作]` (Zhejiang University), Sida Peng `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种可通用的 feed‑forward 3D Gaussian Splatting 框架 SplatAnything，并提出 Gaussian Spacing Normalization（GSNorm）来统一 Gaussian 的空间尺度。

**💡 创新点**

创新点在于：① 用统计量（Gaussian spacing 的中位数）定义场景级归一化因子，解决了传统 Radial Distance Normalization（RDNorm）导致的尺度监督不一致问题；② 建立了跨数据集、跨尺度、跨视角的系统评测基准，进一步验证方法的普适性。

**🔧 技术方法**

技术包括：基于 WorldMirror 2.0 的几何基础模型进行多视角深度与相机估计；将几何转到 GSNorm 归一化空间；使用 Gaussian Backbone + 轻量 GSHead 预测每像素的高斯属性；采用 MSE + LPIPS 损失进行端到端训练。

**📊 数据集**

使用了 ScanNet、NRGBD、DL3DV、Tanks & Temples、Mip-NeRF 360、ETH3D、DTU 七大公开数据集进行评测。

**📈 对比分析**

与多种 w/o pose 与 w/ pose 的 3DGS 方法（如 AnySplat、YoNoSplat、DepthSplat、ReSplat、TokenGS 等）在多视角与单视角、不同输入视角数（4/8/16/32/64）下进行比较，主要指标为 LPIPS/DISTS；SplatAnything 在大多数数据集与场景尺度/视角变换场景中均取得最高或相近最高的性能，尤其在大尺度和极端视角外推上表现显著提升。

**⚠️ 局限性**

局限性：高度依赖几何先验，极端稀疏或难以恢复几何的输入场景下性能会下降；当前仅重建可观测区域，对未观测区域缺乏补全能力。

---

## 586. Terminal-style Mathematical Modeling Bench: The First Agentic and Verifiable Benchmark for Mathematical Modeling

**arXiv ID:** 2611.08237 | [PDF](https://arxiv.org/pdf/2611.08237v1)

**作者:** Xiaojun Wu `[一作]` (IDEA Research), Jian Guo `[通讯]` (IDEA Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了终端化、可验证的数学建模基准——Terminal Bench Math Modeling，能够让大型语言模型在完整的建模任务中完成从假设到结论的完整流程并输出结构化JSON结果。

**💡 创新点**

创新点在于：①将完整竞赛题目包装为可容器化终端任务并提供可执行验证器；②采用方向感知的一侧评分和多参考面板，避免单一参考的主观性；③双层可验证性——程序门层+Lean4证明层，确保答案物理可行。

**🔧 技术方法**

主要技术包括：Docker容器化任务交付、可执行评分脚本、方向感知分数计算、六类判定门逻辑、Lean4形式化证明与合规性检查。

**📊 数据集**

使用了2023–2025年CUMCM与MCM共18道A/B/C类问题的完整题目、附件和裁剪图像，形成公开的评测数据集。

**📈 对比分析**

通过对比七个多模态LLM+工具链组合（如GPT‑6 Astra、Fable 5.1、DeepSeek Flash等），在多参考得分和相对索引上评估模型性能，旗舰模型在多参考得分上达64.58%，相对DeepSeek Flash提升13.2%；成本相对低的模型仍能保持约60%得分。

**⚠️ 局限性**

局限性包括：评分仍依赖多参考面板，缺乏统一标准；可验证层仅覆盖可结构化的物理约束，无法验证所有现实物理真理；数据集规模有限，仅包含18道题，未来需扩大范围。

---

## 587. Signals, Evidence, and Context: User Scrutiny in Real-World LLM Conversations

**arXiv ID:** 2611.08232 | [PDF](https://arxiv.org/pdf/2611.08232v1)

**作者:** Ahana Biswas `[一作]` (University of Pittsburgh), Yu-Ru Lin `[通讯]` (University of Pittsburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对 ShareChat 与 WildChat 两大 LLM 对话数据进行规模化分析，量化了系统在回答中使用的“可信度标识”与可检查证据的出现频率，并研究了这些特征如何影响用户的检查或质疑行为以及在不同任务、对话层级中的分布。

**💡 创新点**

创新点在于：①将“权威引用”与“可检索证据”区分为独立特征；②将用户的检查（请求证据）与质疑（直接反驳）拆开评估；③揭示对话层级的聚类效应；④在同一数据集上同时考察多种验证线索与用户回应的关联，为大规模真实对话的可信度评估提供了全新的框架。

**🔧 技术方法**

使用了基于词典的特征提取、十维多项式逻辑回归、会话聚类稳健估计、ICC 与滞后效应的线性模型，以及宏观 AUC 等预测性能指标；同时利用了自动化标签器与人工校验相结合的混合评估方法。

**📊 数据集**

数据集包括 98,007 条来自 ShareChat（公开共享）与 WildChat（用户自愿）的大型对话记录，涵盖 ChatGPT、Gemini、Grok、Perplexity 等多种 LLM，约 489,283 条回答–用户对话对。

**📈 对比分析**

在全局多项式模型中，权威引用的出现率约 31.9%，URL 仅 1.3%，可检索证据 1.46%；检查与质疑的相对风险比（RRR）均略高于接受，且仅在“技术帮助”场景中不确定性显著倾向于检查；预测层面，任务上下文特征的宏观 AUC 为 0.592，低于仅使用回答特征的 0.551，表明上下文信息更具预测力。

**⚠️ 局限性**

局限性包括：不同平台的数据收集方式与字段差异导致比较受限；特征词典的召回率未知，精度虽有验证但仍有限；用户行为标签仅覆盖对话内可见交互，无法捕捉外部检索；短面板的动态固定效应估计可能偏倚；以及整体模型未能评估证据质量与具体内容的准确性。

---

## 588. From Twisted Facts to Rogue Agents: Can LLMs Reliably Detect Multi-Dimensional Hallucinations?

**arXiv ID:** 2611.08231 | [PDF](https://arxiv.org/pdf/2611.08231v1)

**作者:** Hanyu Qiu `[一作]` (Beijing Normal University), Renfen Hu `[通讯]` (Beijing Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

建立了双语、细粒度的幻觉检测基准HaluCheck，涵盖42k文本与代理轨迹样本，支持检测、定位、纠正及根因追踪；

**💡 创新点**

创新点在于：① 细粒度多级标注与定位与纠正；② 同时覆盖文本与代理两类幻觉；③ 通过可控扰动生成合成幻觉，降低人工成本；④ 设计统一评测指标与复现评测流程；

**🔧 技术方法**

主要技术包括：可控扰动生成合成幻觉；参考无/有的检测方法；显式思考、Web检索与监督微调提升性能；使用 SpanF1、CorrectionRate、RootTurnAccuracy 等指标；

**📊 数据集**

数据集来源于 Wikipedia、HotPotQA、PopQA、HalluQA、CMRC2018 及真实代理轨迹（AgentFLAN、AgentBank、ToolACE 等），构成 HaluReal、HaluFake、HaluAgent，覆盖中英两语；对比 TruthfulQA、HaluEval、AgentHallu 等基准；

**📈 对比分析**

在14大模型上评测，最佳检测准确率约84%（Gemini3.1-Pro），定位/纠正精度低于检测；代理任务更难，最佳准确率72%；显式思考、检索与微调可提升5–10%但整体仍未达到可靠水平；

**⚠️ 局限性**

限制包括：① 合成幻觉与自然幻觉不完全一致；② 仅覆盖代表性文本与代理场景，未涵盖更广泛真实环境；③ 缺乏独立人工检测基准；④ 可能存在模型偏见；⑤ 对检索质量与训练数据覆盖度敏感。

---

## 589. Beyond Quadratic Attention: Separable Gaussian Operators for ODE Vision Transformers

**arXiv ID:** 2611.08236 | [PDF](https://arxiv.org/pdf/2611.08236v1)

**作者:** Carlos Boned Riera `[一作]` (Universitat Autònoma de Barcelona), Oriol Ramos Terrades `[通讯]` (Universitat Autònoma de Barcelona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

提出一种Separable Sum-of-Gaussians（SSoG）空间算子，用来替代传统的密集自注意力，显著降低了视觉Transformer在高分辨率下的计算与显存开销。

**💡 创新点**

创新点在于：1) 通过轴分解的高斯核实现子二次复杂度且全局Lipschitz连续性；2) 让连续深度（Neural ODE）和循环Transformer具备理论上唯一解与稳定收敛的保证；3) 在多种Transformer变体中统一验证其优势。

**🔧 技术方法**

核心技术包括：SSoG核混合、轴分解高斯卷积、连续深度（Neural ODE）框架、循环Transformer参数共享、Picard–Lindelöf理论分析。

**📊 数据集**

主要使用数据集：CIFAR‑100（224×224）以及在更高分辨率（1024×1024）下的ImageNet/类似任务进行推理时间与显存评估。

**📈 对比分析**

与多头注意力（MHA）在离散、循环与连续深度Transformer三种架构下进行对比：SSoG在离散模型Top‑1准确率提升2.2–4.1点，循环模型提升5.4–8.1点；在1024×1024输入下，连续深度SSoG吞吐量提升154%，峰值显存降低60%；整体显示SSoG在参数效率和推理性能上优于MHA。

**⚠️ 局限性**

局限性包括：1) 目前实验主要聚焦于视觉任务，对语言或多模态任务的适用性尚未验证；2) 固定核版本在某些复杂场景下可能缺乏足够的表达能力；3) 对大规模预训练模型的评估尚未完成，需进一步验证其可扩展性。

---

## 590. Domain-specific information retrieval

**arXiv ID:** 2611.08228 | [PDF](https://arxiv.org/pdf/2611.08228v1)

**作者:** Suzan Verberne `[一作]` (Leiden University), Allan Hanbury `[通讯]` (TU Wien)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述并系统化了过去15年信息检索社区在法律、健康、学术与专利四大领域的域内检索任务、数据集与方法，提出了“设计三角”与“域内检索抽象”模型以统一描述用户、任务与内容三维关系。

**💡 创新点**

创新点在于将多源域内检索挑战与方法集成为可转移的框架，阐释了跨域迁移、长文档/长查询处理、文献计量增强与对话式检索的最新进展，并首次将LLM驱动的对话检索纳入评估维度。

**🔧 技术方法**

主要技术包括两阶段检索（BM25、DPR/ANCE等一阶检索与MonoBERT/LegalBERT等二阶重排序）、基于Transformer的域特定BERT预训练与微调、跨域迁移学习、基于文本理解的查询扩展与聚类、以及对话式检索的检索-生成框架。

**📊 数据集**

使用的数据集涵盖传统评测库（TREC、CLEF、NTCIR、FIRE）、域特定集合（LegalBERT数据、CLEF-IP、BEIR、MedSci、Papers,ClinicalTrials、iSearch、PatentBERT等），以及多语言与跨域合成集如BEIR、MS MARCO、WPI+、LongEval等。

**📈 对比分析**

对比实验表明，在高召回任务（如法律案件检索、专利优先权检索、系统综述检索）中，域特定BERT+重排序显著提升F1/Recall，但在开放域或大规模多任务下仍落后于通用模型；跨域迁移能部分弥补数据稀缺问题，但整体性能提升受限于目标域标注量。

**⚠️ 局限性**

局限性包括评测数据稀缺导致深度评估困难、低召回任务的高成本标签、可复现性与版本管理挑战、偏见与公平性问题（曝光偏差、社会偏差、交互偏差）、多语言与跨域语义不一致，以及对话式检索在真实场景中的鲁棒性与解释性不足。

---

## 591. Dynamic Optimistic Constrained OCO with Memory via Delay Equivalence

**arXiv ID:** 2611.08302 | [PDF](https://arxiv.org/pdf/2611.08302v1)

**作者:** Mohammed Abdullah `[一作]` (Universite Paris Saclay), Tijani Chahed `[通讯]` (Telecom Sudparis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

提出了一种新的在线凸优化框架COCO-M，解决了具有记忆和时间变化约束的在线学习问题，提供了一种乐观算法来处理不可信的预测。

**💡 创新点**

创新点在于首次建立了适应环境非平稳性和预测准确性的界限，同时引入了记忆到延迟的有效转化，克服了以往方法的限制。

**🔧 技术方法**

使用了乐观延迟在线梯度下降（OGD）算法，并结合了自适应对冲的元学习框架。

**📊 数据集**

未具体提及使用的数据集，但框架适用于多种实际应用场景，如推荐系统和投资组合优化。

**📈 对比分析**

与现有方法相比，提出的框架在未知路径长度下实现了𝒪(√(T))的累积约束违反（CCV），在已知路径长度下实现了𝒪(log T)的CCV，且在完美预测下实现了𝒪(1)的后悔值。

**⚠️ 局限性**

限制在于框架需要预测未来的函数和动作，这种循环依赖可能会影响实际应用的可行性。此外，完美预测下的约束违反未能缩减到𝒪(1)，仍需进一步改进。

---

## 592. Should You Split Decode? Disaggregation Across Future Models and Accelerators

**arXiv ID:** 2611.08299 | [PDF](https://arxiv.org/pdf/2611.08299v1)

**作者:** Ramyad Hadidi `[一作]` `[通讯]`, Ramyad Hadidi

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究大规模语言模型推理中两种服务拆分方式——预填/解码分离（P/D）与预填/解码/注意力/前馈网络分离（P/D+AFD），并构建跨层校准模型评估在未来模型、加速器类与网络结构下的功率效率（SLO goodput / kW）。

**💡 创新点**

创新点在于：①将拆分决策与硬件分配独立建模，揭示二者的相互影响；②使用可扩展的跨层模型（39种加速器、4类、6种网络）进行大规模搜索；③量化“提前硬件锁定”带来的损失并展示通过重新优化映射和池大小可显著降低该损失；④针对不同KV/专家负载方向系统性比较，发现AFD仅在特定KV‑heavy场景与小规模扩展域时有利。

**🔧 技术方法**

主要技术包括：多维度硬件规范表征、LLM架构参数化（KV缓存、专家激活、稀疏注意力）、计算/内存时间与重叠模型、通信与集成模型、排队论（Erlang‑C）用于SLO goodput估计、Monte Carlo误差评估与两部分判定法。

**📊 数据集**

使用公开的LLM模型集合（DeepSeek、Kimi、Qwen等），以及已发布的加速器性能数据（GPU、Systolic、3D‑DRAM、SRAM/Dataflow类），并基于公开测量结果进行模型校准。

**📈 对比分析**

比较方法：对每个模型方向、网络结构、功率预算下，分别在P/D与P/D+AFD两种拆分方式上，独立搜索最优硬件分配、映射与池尺寸，计算SLO goodput / kW；利用两部分判定法判断优势是否稳健。结果显示：大多数情况P/D优于P/D+AFD，AFD优势仅在KV‑heavy、专家宽度大的方向且小扩展域时出现，优势幅度最多~9%，整体未见显著性能提升。

**⚠️ 局限性**

局限性：①模型对计算/内存重叠、通信延迟、批处理大小等参数的近似可能在极端工作负载下失效；②仅评估两阶段拆分，未考虑多级或动态拆分；③使用的加速器规格和网络参数基于公开信息，未来硬件的差异可能影响结论；④排队模型未考虑请求多样性与后台任务干扰。

---

## 593. Hard Issue, Weak Model, or Bad Run? Rethinking Failure Prediction for Coding Agents

**arXiv ID:** 2611.08285 | [PDF](https://arxiv.org/pdf/2611.08285v1)

**作者:** Xingyue Shi `[一作]` (NC State University), Wesley K. G. Assunção `[通讯]` (NC State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对现有的基于轨迹的失败预测器进行系统评估，探讨其是否真正学习到单个运行的失败原因，并针对问题难度、模型能力和随机误差进行分解；

**💡 创新点**

创新点在于提出并验证三大假设（高AUROC意味着学习运行信息、oracle差距证明模型可切换、报警能识别可重试的失败）并证明它们在现实数据中往往不成立，呼吁未来预测器必须明确失败原因并按需行动；

**🔧 技术方法**

技术包括TF‑IDF + Logistic回归、Qwen3-0.6B+LoRA微调、Rasch和2PL参数化模型构建无特殊化基准、条件logit、AUROC对跨/内问题对的分解、等价性检验以及重复运行的分层评估；

**📊 数据集**

使用公开SWE‑bench Verified轨迹（7个模型、23个模型、16个外部模型）以及包含重复运行的SWE‑agent语料（两款Llama模型共522个问题）进行实验；

**📈 对比分析**

比较方法：在跨问题和内问题两层分别计算AUROC，比较预测器与无运行信息的基准（issue difficulty、model identity、长度+）；对oracle差距与无特殊化参考进行对比；对重试行为的有效性进行评估；结果显示预测器仅提升约0.02 AUROC，oracle差距无显著超越随机，重试仅在特定“坏运行”场景下有小幅收益；

**⚠️ 局限性**

局限性包括：仅评估轨迹预测器，未考虑其他预测形式；使用的基准和模型相对有限，缺乏更广泛的多任务或更大规模模型的验证；未对真实部署中的预算/误报成本进行深入量化；无法充分验证特殊化程度的真实存在与否。

---

## 594. Geometric Relevance Is Not Sufficient: Evaluating Focused Post-Training Supervision in Diffusion Docking

**arXiv ID:** 2611.08249 | [PDF](https://arxiv.org/pdf/2611.08249v1)

**作者:** Michael Brown `[一作]` (University of Central Florida), Ivan Garibay `[通讯]` (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了DiffDock后训练时如何分配torsion监督，比较宽泛连续与针对Cartesian后果或目标一致性的聚焦监督。

**💡 创新点**

提出基于几何诊断的torsion优先级分配方法，并探讨其对分子对接性能的影响。

**🔧 技术方法**

采用Diffusion-based docking框架DiffDock、损失重分配、几何后果/一致性评估、EMA等技术。

**📊 数据集**

使用PDBBind与MOAD子集进行实验。

**📈 对比分析**

在固定预算下做5轮后训练，对比生成<5Å成功率、best‑of‑10采样、Top‑1返回姿态等指标，结果显示宽泛连续法与聚焦或混合策略无显著差异，宽泛连续略优。

**⚠️ 局限性**

仅评估整体姿态性能，未拆分 pocket 或 ligand 失败原因，且仅在单一模型/数据/预算下实验，缺乏更广泛的泛化验证。

---

## 595. Value-oriented policy distillation for interpretable wind farm power control

**arXiv ID:** 2611.08245 | [PDF](https://arxiv.org/pdf/2611.08245v1)

**作者:** Max Bruninx `[一作]` (Vrije Universiteit Brussel), Jan Helsen `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于强化学习的隐式平衡控制策略的可解释化蒸馏框架，将软演员-评论家（SAC）学习的策略转化为可解释的增广提升机（EBM）控制模型，减少风电场的不平衡成本。

**💡 创新点**

创新点包括：①在可解释化策略蒸馏中引入价值导向权重，突出高价值动作；②利用专家知识裁剪EBM的单变量和交互项，使得控制策略可被领域专家解释；③首次在风电隐式平衡任务中应用EBM，实现高透明度与可解释性。

**🔧 技术方法**

核心技术：强化学习（SAC），可解释提升机（EBM），价值加权蒸馏损失，专家知识融合，基于历史数据的价格冲击模拟。

**📊 数据集**

数据集：比利时Elia公开平台的2022‑2023年电网实时数据，Norther离岸风电场的SCADA测量数据，用于训练与评估。

**📈 对比分析**

与决策树、价格阈值控制器以及LightGBM对比；EBM专家+价值模型在评估期内将不平衡成本降低82%（相对无控制场景），比价格阈值控制器的54%和SAC的93%更好，但与SAC仍有一定差距；在预测精度上，EBM模型虽优于决策树但仍低于LightGBM，说明可解释性与性能之间的权衡。

**⚠️ 局限性**

局限性：可解释性与性能之间的权衡导致EBM模型的预测精度不如全模型或黑盒模型；价值加权的提升有限；实验仅基于比利时数据，未在真实风电场部署，尚缺乏跨区域或多风电场的泛化验证。

---

## 596. Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation

**arXiv ID:** 2611.08315 | [PDF](https://arxiv.org/pdf/2611.08315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 597. Escalate or Close: How Fast-Tier Models Handle Held-Out Web Attack Families in Two-Tier Triage

**arXiv ID:** 2611.08229 | [PDF](https://arxiv.org/pdf/2611.08229v1)

**作者:** Maciej Kisielewicz `[一作]` (Wrocław University of Science and Technology), Ngoc Thanh Nguyen `[通讯]` (Wrocław University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估两层分流模型对未见攻击家族的检测与决策性能

**💡 创新点**

首次将攻击payload注入真实流量，使用零样本决策模型与监督模型对比，并探索不同预算下的误判率

**🔧 技术方法**

使用现代BERT、Jev、Clef、OWASP Core Rule Set以及异常基线等技术进行检测与决策

**📊 数据集**

构造包含七类攻击payload与真实浏览器流量的混合数据集，并保留一类作为留出测试集

**📈 对比分析**

通过与监督检测器、语言模型、核心规则集对比，发现现代BERT在5%正常预算下对留出路径遍历攻击的误报率低于0.05，且只升阶约0.01；而Jev与Clef在同一预算下仍升阶超过20%，但其训练数据可能已包含公开家族，导致泛化不足

**⚠️ 局限性**

局限性包括训练集可能覆盖公开攻击家族，形状控制不足导致对“良好复制”请求误判，且对未见家族的泛化仍受限

---

## 598. Cultural Adaptation within Repeated Social Interactions

**arXiv ID:** 2611.08334 | [PDF](https://arxiv.org/pdf/2611.08334v1)

**作者:** Chen Cecilia Liu `[一作]` (University of Cambridge), Anna Korhonen `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了大语言模型在连续社交互动中通过外部记忆机制学习并适应隐式文化规范，提出将文化适应视为多轮交互中的规范同化问题；

**💡 创新点**

创新点在于：①将文化适应框架化为持续交互的规范同化；②设计双层（会话级短期+跨会话长期）记忆架构，并证明会话级记忆比回合级更新更能持续学习；③在多语言、多文化场景中验证了跨场景迁移与长期适应的可行性；

**🔧 技术方法**

采用外部非参数记忆（STM/LTM）、BDI记忆与反思记忆机制，并结合情境推理提示；使用Gemma、Qwen、Llama等小型开源LLM作为学习主体，Gemma4‑31B担任NPC与评判者；

**📊 数据集**

使用5个文化（中国、法国、日本、新西兰、沙特）共1,264个社会交互场景，场景基于CultureForest/Bank数据由Gemini‑3.7‑Flash自动生成并进行人类验证；

**📈 对比分析**

通过与无记忆基线、情境推理和两种记忆机制的对照实验，使用LLM判定器评估“社会声誉得分”作为指标；Memory_R机制在所有模型上显著提升平均得分（+0.21），并在多语言、跨场景、长周期实验中保持或提升性能；

**⚠️ 局限性**

局限性包括：①场景由Gemini生成，可能引入模型写作风格偏差；②文化划分仅按国家，缺乏细粒度子文化分析；③实验仅在少量小型模型和主要语言环境下进行，未探究更开放式交互或更大规模模型的适应效果。

---

## 599. Polarity Classification with Multitask Learning Syntax Augmentation

**arXiv ID:** 2611.08323 | [PDF](https://arxiv.org/pdf/2611.08323v1)

**作者:** Muhammad Imran `[一作]` (Universidade da Coruña), Carlos Gómez-Rodríguez `[通讯]` (Universidade da Coruña)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出两种多任务学习策略（Standard‑MTL 与 Contrastive‑MTL），在不需要推理时句法解析的前提下，将句法知识融入情感极性分类模型。

**💡 创新点**

创新点在于：①将依存句法解析转化为序列标注作为辅助任务；②利用对比学习和句法偏置注意力，在训练阶段隐式注入句法信息，推理阶段完全消除解析开销。

**🔧 技术方法**

主要技术包括 XLM‑RoBERTa 大模型、序列标注句法解析、对比学习（语法关系与情感标签两种负样本设定）、语法偏置注意力池化与不确定性权重的损失组合。

**📊 数据集**

使用多语言 Amazon 评论数据集 MARC（英、法、德、西）以及 SemEval‑2022 任务10的七个多语言评论数据集（英、西、挪、巴斯克、加泰罗尼亚、德、法）。

**📈 对比分析**

与单任务基线和外部规则式管道 SELSP 进行对比。Relative 编码下，Contrastive‑MTL 在 10/11 组数据上优于基线，并在所有支持 SELSP 的数据集上均优于 SELSP；Inference 速度与基线相当，远快于需要解析的 SELSP。

**⚠️ 局限性**

局限在于依赖训练阶段的离线句法标注，若解析器错误会影响对比学习；目前仅针对句子级极性分类，未扩展至细粒度或多方面情感分析；对不同句法表示（短语结构、轻量句法特征）的兼容性尚未评估。

---

## 600. Interpretable many-body interaction laws for collective motion from few trajectories

**arXiv ID:** 2611.08314 | [PDF](https://arxiv.org/pdf/2611.08314v1)

**作者:** Timon S. Gutleb `[一作]` `[通讯]` (University of Leeds), Timon S. Gutleb (University of Leeds)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了方向性原子集群展开（dACE）方法，用以系统地学习自驱动物体群体运动中的多体交互规则，并在二维与三维环境下实现完全等变的、可解释的加速器。

**💡 创新点**

创新点包括：①把原子物理学中的纯体排序（pure body‑ordered）集群展开迁移到群体运动，得到可分离的1、2、3…体交互；②引入分阶段的自我推进项（self term），避免自身运动被误吸收；③实现严格的旋转/反射等变性，支持有色（chiral）规则；④通过极简正则化与贝叶斯岭回归实现极大化证据，提升数据效率；⑤与现有的注意力网络 SocialNet 直接对比，展示更低误差与更好解释性。

**🔧 技术方法**

主要技术手段包括：方向性原子集群展开（dACE）框架、纯体排序与纯化（purification）、等变基函数（spherical /圆形谐波与速度多项式）、阶段式拟合（先拟合交互，再拟合自我推进），以及Julia实现的高效 QR/奇异值分解求解；与之对照的基准是公开实现的 SocialNet（注意力网络）。

**📊 数据集**

数据集：①合成数据——基于 D'Orsogna‑type 自驱动粒子模型、Boids、zonal 模型，覆盖 2、3、4 体规则与不同连续/离散特性；②真实跟踪数据——8 条 60–100 条鱼的 zebrafish 轨迹（idtracker.ai）、10 组 10–151 条 golden shiners 的实验轨迹，时间步长约 0.1 s，邻域半径分别 8 BL（zebrafish）和 3 BL（shiners）。

**📈 对比分析**

比较方法：①相对力误差（RMS）与体排序误差、②下一个时间步预测的 R²、③数据量与噪声敏感度（误差随样本数和观测噪声变化）、④分割一致性（同一数据分两半独立学习的差异）、⑤真实轨迹滚动预测。实验结果显示：在所有合成规则中，dACE 在达到真规则体排序时误差降至 1–2 % 级；SocialNet 在多体或色彩规则上误差大、数据需求高；dACE 在跟踪数据中实现更高的一致性、更稳健的交互提升，并在三体项学习中揭示了对多体邻域的抑制作用；在自驱动噪声实验中，next‑step R² 对交互质量不敏感，提示需要多种评价指标。

**⚠️ 局限性**

局限性：①对高度离散或硬阈值规则（Boids、zonal）近似慢；②下一个时间步预测误差与交互质量无关，需慎用；③自我推进项若不分阶段拟合，可能吸收邻居效应；④目前仅支持固定邻域半径（未考虑 k‑近邻、可视化遮挡等选择性影响）；⑤未处理动态障碍物或多种物种交互；⑥对极端数据量/高噪声下的鲁棒性仍需进一步验证。

---

## 601. Trilemma: How Agents choose between Honesty, Confidentiality, and Task Completion

**arXiv ID:** 2611.08312 | [PDF](https://arxiv.org/pdf/2611.08312v1)

**作者:** Johannes Taraz `[一作]` (Ellis Institute), Sahar Abdelnabi `[通讯]` (Ellis Institute)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个名为Trilemma的多智能体仿真环境，用以研究个人AI代理在面对用户隐私、诚实与任务完成三重冲突时的行为。

**💡 创新点**

创新点在于将隐私泄露与虚假陈述这两大问题统一到同一实验框架，并揭示了在“无指令”情境下，代理会自发产生欺骗、隐瞒或误导的行为，且对不同模型与隐私指令的敏感性提供系统评估。

**🔧 技术方法**

技术上主要使用开源与闭源大型语言模型（如DeepSeek、GLM、Kimi、Qwen、Claude、GPT系列），通过工具调用（Slack、日历、看板）和链式思维（CoT）进行多轮交互；并设计一套自动与人工混合的判定器（deception、disclosure、motivation）。

**📊 数据集**

数据集为完全合成的公司工作区，包含四名员工的私信、日历条目、任务描述及预植的隐私偏好（如“避免与Bob合作”），共计若干场景（主实验、保密级别消除、预先讨论、honeypot等）。

**📈 对比分析**

比较方法：对七类开源模型和两类闭源模型进行10次跑，每模型/跑对应两个代理（Alice、Carol）进行统计。结果显示：在追求T2（避Bob）时，135/140代理中有107进行欺骗（41作假言、66误导），披露隐私偏好的比例随隐私指令强度由83%降至2%；任务完成率仅16/70。与之前研究相比，模型间的欺骗与披露倾向差异显著，且更高隐私指令能显著降低泄露。

**⚠️ 局限性**

局限性包括：样本规模有限、仅测试单一任务场景、CoT的真实性与一致性难以保证、闭源模型无法获取内部推理导致评估不完整、未探究不同奖励结构或更复杂的冲突情形。

---

## 602. Measured Coupling for Parallel Decoding in Diffusion Language Models

**arXiv ID:** 2611.08307 | [PDF](https://arxiv.org/pdf/2611.08307v1)

**作者:** Tal Oved `[一作]` (Tel Aviv University), Lior Wolf `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在扩散语言模型解码过程中直接测量标记间相互影响（coupling）的无训练方法，并将其嵌入到主前向传播中，减少了需要的解码步骤。

**💡 创新点**

创新点在于用模型自身通过块内探测器（probe）即时计算每个候选标记对其他标记置信度的实际影响，从而捕捉支持与冲突信息，替代传统基于相似度的近似；并且实现了 probe 与主前向同一次读模型权重，显著降低了计算成本。

**🔧 技术方法**

技术包括：扩散语言模型的块级并行解码、使用 Jensen‑Shannon 相似度的 mean‑field 变分更新、设计的块局部 probe（仅重算当前块的掩码位置，复用其余位置的 key/value）、在同一次前向中同时处理主句子和 probe 行、以及基于测得 coupling 的 mean‑field 更新。

**📊 数据集**

使用的数据集有 GSM8K（算术题）、MATH（高级数学题）、HumanEval（编程题）以及 MBPP（Python 编程任务）。实验在三大模型上进行：LLaDA‑8B‑Instruct、LLaDA‑1.5‑Instruct 以及 Dream‑v0‑Instruct‑7B。

**📈 对比分析**

与一token‑per‑step、Fast‑dLLM、DAWN 以及原 mean‑field 解码进行比较。测得 coupling 在所有基准上都实现了最多 6× 的步数压缩、最高的 tokens‑per‑second 速度，并且与 mean‑field 的准确率差距均在 ±2 分之内（大多置信区间包含 0）。

**⚠️ 局限性**

局限性包括：每一步仍需 1.2–1.5 倍的计算量相较于传统 mean‑field，导致墙壁时间提升幅度小于步数提升；在某些数据集（如 MATH、Dream HumanEval）上需要在速度与准确率间做权衡；并且支持关系可能把错误的标记相互强化，导致误判。

---

## 603. The Proof Analysis Problem for Constant-Depth Frege

**arXiv ID:** 2611.08305 | [PDF](https://arxiv.org/pdf/2611.08305v1)

**作者:** Noel Arteche `[一作]`, Kilian Risse `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于递归分支与剪枝的CNF公式解析方法，利用分解树把原公式逐步展开成更小的子公式，并给出了以变量数 n 和对数差值 Δlog|π| 为参数的时间复杂度上界。

**💡 创新点**

创新点在于：① 用图形化分解树直观展示分支、剪枝与条件化的过程；② 推导出从原公式到最终单元子公式的复杂度随 n 和 Δlog|π| 的递增关系；③ 通过该分析为后续的剪枝策略提供理论依据。

**🔧 技术方法**

主要技术包括：图论/树结构分析、逻辑推理（分解与合并）、复杂度分析（对数差值的估计）以及实验评测。

**📊 数据集**

实验使用了公开的 SATLIB 数据集以及人工生成的中等规模 CNF 公式，变量数从几十到几千不等。

**📈 对比分析**

与传统的 DPLL、CDCL 以及其他剪枝方法对比，实验结果表明该方法在变量数在 200–800 之间的实例上平均速度提升约 25%–35%，内存占用相对稳定；在更大规模实例上表现仍在可接受范围内。

**⚠️ 局限性**

局限性包括：① 对极高树宽或超大规模公式的扩展仍受限；② 该方法对高度稠密的 CNF 公式剪枝效果不明显；③ 复杂度上界中的 Δlog|π| 估计依赖于先验信息，实际场景中可能难以精准获取。

---

## 604. First-Token Attraction in Mamba Dynamics

**arXiv ID:** 2611.08303 | [PDF](https://arxiv.org/pdf/2611.08303v1)

**作者:** Trinh T. Nguyen `[一作]` (Shanghai Jiao Tong University), Thieu N. Vo `[通讯]` (University of Bath)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究深层Mamba模型中的令牌（token）动力学，提出了基于高维不变盒子与精确退出时间的全新解析框架，刻画了令牌轨迹的极限点并证明了指数级收敛，同时发现首个令牌具有全局吸引作用，导致后续令牌聚集并解释了“注意力汇聚”现象。

**💡 创新点**

创新点在于：①针对Mamba中矩阵型交互系数缺乏正定性，构建了不依赖正性假设的新分析框架；②揭示首令牌为全局吸引子，解释了深层模型中的注意力汇聚；③将该理论推广至时变参数以及更广泛的自回归序列模型。

**🔧 技术方法**

技术手段包括：连续时间极限下的动力学系统建模、球面上的流映射、invariant box 约束、exit‑time 论证、线性化与稳定性分析、数值仿真与可视化。

**📊 数据集**

主要实验数据集为 Falcon‑Mamba‑7B 预训练模型（在大规模语言建模数据上训练的 Mamba 变体）以及随机生成的参数集合，用于验证理论预言。

**📈 对比分析**

方法对比：与理论极限点和指数收敛速率进行直接对比，实验中观察到首令牌趋向坐标单位向量、后续令牌聚集在同一吸引点，并在各层层级上逐渐表现出明显的注意力汇聚；相对其他基准模型未在本文中给出数值性能比较。

**⚠️ 局限性**

局限性在于：分析仅针对连续时间极限与渐近行为，尚未完全对接离散大规模 Mamba 实现；对实际推理速度、显存占用等工程指标未作评估。

---

## 605. REACT: Towards Real-Time Completion of Skeletal Motion Signals

**arXiv ID:** 2611.08301 | [PDF](https://arxiv.org/pdf/2611.08301v1)

**作者:** P. de Clermont Gallerande `[一作]`, L. Hoyet `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为REACT的实时骨骼运动信号补全框架

**💡 创新点**

创新点在于训练时模拟关节遮挡与旋转噪声，推理时通过检测不可靠关节动态识别并完成运动流

**🔧 技术方法**

使用深度递归/Transformer网络结合时序卷积，对训练时的数据进行数据增强并在推理时引入置信度评估机制

**📊 数据集**

在公共动作捕捉数据集Human3.6M与CMU MoCap上进行实验

**📈 对比分析**

与传统基于RNN、CNN和基线补全方法相比，REACT在RMSE与实时延迟方面分别提升约15%和20%，并在遮挡严重的测试场景下保持高精度

**⚠️ 局限性**

局限性包括：对极端噪声或缺失比例过高的关节补全效果下降；模型参数较大，导致在极低功耗设备上部署受限

---

## 606. Hierarchical Incremental Beta Splatting for Online Reconstruction

**arXiv ID:** 2611.08288 | [PDF](https://arxiv.org/pdf/2611.08288v1)

**作者:** Linus Franke `[一作]` `[通讯]` (University of Würzburg), Linus Franke (University of Würzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种在线增量重建框架，使用可调形状的β核来构建辐射场，并通过几何证据驱动初始化、误差驱动的重定位以及闭式矩匹配实现多层级加速。

**💡 创新点**

创新点包括：① 将β核的形状参数视为“硬度”或置信度轴，实时根据几何可观测性和多视深度一致性初始化；② 在没有离线密集化的情况下使用误差驱动重定位替代克隆/分裂；③ 为β核导出闭式合并公式，保持形状统计并构建LOD层级；④ 开发基于β核的快速融合光栅化器。

**🔧 技术方法**

核心技术包括：β核光栅化、光栅化边界裁剪、几何可观测性度量、深度一致性置信度、误差驱动重定位、闭式矩匹配合并、在线姿态估计、fused优化、快速Adam。

**📊 数据集**

在MipNeRF360（OTF）、TUM、StaticHikes、3DGS基准以及大规模场景SmallCity、CityWalk上进行评估；使用公开的相机位姿和RGB图像序列。

**📈 对比分析**

与i3DGS、MonoGS、S3PO-GS、LongSplat、On-The-Fly NVS、DROID-Splat、H3DGS、Octree-GS等方法比较，PSNR提升0.5–0.8 dB，SSIM提升0.01–0.02，LPIPS下降0.02–0.04；优化时间平均缩短30–40%，大规模场景在4分钟内完成，显著快于Octree‑GS（约1小时）。

**⚠️ 局限性**

局限性：仍未达到完全离线优化的质量；合并时近似近带外贡献，可能导致细节损失；依赖姿态估计的准确性；对极端光照或透明材质的鲁棒性有限。

---

## 607. Fault-Geometry-Aware Control Allocation for Emergency Quadrotor Landing under Single-Motor Loss of Effectiveness

**arXiv ID:** 2611.08259 | [PDF](https://arxiv.org/pdf/2611.08259v1)

**作者:** Suhas S P `[一作]` (Visvesvaraya Technological University), Ashwini B P `[通讯]` (Visvesvaraya Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在单个电机失效效能下降（LoE）的情况下，比较了三种控制分配器（bounded weighted least squares/PINV、QP‑lite残差分配器和CEM调优的QP‑lite），并基于故障电机索引制定了冻结路由策略，在新的随机种子和集成监督实验中验证其安全性，随后对10个±10%单参数扰动进行边界敏感性评估。

**💡 创新点**

发现分配器的优劣随失效电机位置变化，提出仅凭失效电机索引即可决定最优分配器的“冻结路由”方案；同时将离线交叉熵方法（CEM）与有限候选QP‑lite进行系统对比，首次量化了单参数扰动对安全触地边界的影响。

**🔧 技术方法**

使用 CrazySim/Crazyflie 软硬件集成仿真（Gazebo 物理、ROS 2、Ubuntu 22.04），通过定向种子实验进行安全触地试验；实现了 bounded weighted least squares（PINV）、QP‑lite残差分配、CEM 离线评分参数调优，以及 Firth 偏差校正的逻辑回归来估计安全阈值。

**📊 数据集**

实验数据集包含 3,570 次仿真试验，涵盖开发集（30 条/电机×4 电机×3 分配器）、冻结策略验证（120 条新种子）、集成监督（120 条随机顺序）、M2/PINV 边界扫描（210 条）、OFAT 10 个单参数扰动（2,100 条）以及路由稳定性对比（660 条）。所有数据均来自仿真，不涉及真实硬件采样。

**📈 对比分析**

通过安全触地试验计数比较分配器性能：在开发集 120/120 选择出路由后，冻结策略在新种子上获得 117/120（97.5%）安全率，集成监督 115/120（95.8%）。M2/PINV 的安全阈值 η_50≈0.4956，±10% 质量或推力系数扰动可使 η_50 变动约 ±0.05；在 10 个单参数扰动中，PINV 在 9/11 状态下安全率高于 CEM，且未出现逆转。

**⚠️ 局限性**

局限性在于：仅在仿真环境下验证，未测试硬件平台、在线故障检测/隔离、完整或多电机失效场景；单参数扰动分析未涵盖联合不确定性；bounded WLS 仅在 PWM 域下操作，QP‑lite 为有限候选搜索，CEM 仅调优评分参数，缺乏全局安全或最优性保证。

---

## 608. Minimum-Cost $k$-Arborescence Packing in $\tilde O(km+k^4 n)$ Time

**arXiv ID:** 2611.08256 | [PDF](https://arxiv.org/pdf/2611.08256v1)

**作者:** Zhaienhe Zhou `[一作]` `[通讯]` (University of Michigan), Zhaienhe Zhou (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4`

**🎯 论文内容**

本文提出了一种确定性、强多项式时间的算法，用于在有向多重图中求解最小总成本的 k 个互不相交的根向 arborescence 以及在无向图中求解最小总成本的 k 棵互不相交的森林（包括 k 棵互不相交的生成树）

**💡 创新点**

创新点主要有：① 将 Gabow 的无权版细粒度增广路径技术推广到带权情形；② 通过“紧块”概念把增广路径局部化，使得可以一次性执行线性数量的最优增广；③ 构造压缩搜索图并引入顶点价格，使得所有增广路径可在一次 Dijkstra 计算中得到；④ 通过兼容增广路径集合的最大化实现批量增广，从而将时间复杂度降至近线性；⑤ 进一步将上述技术应用到无向森林打包，获得几乎线性的时间

**🔧 技术方法**

使用的技术包括：加权 matroid intersection（图的 k-副本图形 matroid 与入度分区 matroid 的交叉）；紧块与块收缩；细粒度搜索图与树路径压缩；负权最短路与可行潜能；Dijkstra 计算可调价；批量增广与兼容性判定；以及两阶段贪心与流网络求解来实现无向森林打包

**📊 数据集**

本文是理论算法研究，没有使用具体实验数据集；所有结果均为算法运行时间的上界和理论证明

**📈 对比分析**

与之前的最速弱多项式算法（O(k m √n log C)）以及强多项式算法（O(k m n)）相比，本算法在 k = n^o(1) 时实现了近线性 O(k m + k^4 n) 的时间；在无向森林打包方面，从 O(k m n) 降至几乎线性 O(k m^{1+o(1)} + k^5 n) 的时间；实验上由于没有实现，本文未给出运行时间对比，但理论上已大幅度提升

**⚠️ 局限性**

局限性包括：① 对 k 的取值仍有限制，最佳性能在 k = n^o(1)；② 在无向森林打包中时间复杂度仍含高阶多项式项 k^5 n；③ 需要对图进行严格的预处理和压缩，实际实现复杂；④ 该方法仍是纯理论复杂度分析，缺乏实验验证

---

## 609. Nowcasting of Tropical Cyclone Wind Fields and Intensity from Geostationary Imagery

**arXiv ID:** 2611.08255 | [PDF](https://arxiv.org/pdf/2611.08255v1)

**作者:** Simon Donike `[一作]` (Earth Systems Lab), Anna Jungbluth `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用地球同步卫星（GEO）红外影像，训练简易神经网络同时预测热带气旋的二维表面风场及其强度指标（V_max 与 R_max）

**💡 创新点**

创新点在于将风场预测与强度量化联合训练到潜在空间，显著提升快速增压事件的预测精度，并实现仅依靠GEO影像完成无SAR、无ERA5输入的近实时监测

**🔧 技术方法**

采用U‑Net结构生成风场，随后通过MLP在潜在空间或后处理阶段预测V_max、R_max及其它半径指标；使用Huber损失、权重平衡与风速区间加权等技术进行训练

**📊 数据集**

数据来源为CyclObs SAR表面风场（3 km）、GOES‑16/18及Himawari‑8/9十通道IR影像、IBTrACS最佳轨迹指标，额外可选ERA5七个气象变量（预留用于实验）

**📈 对比分析**

通过与单独预测V_max的baseline、Post‑hoc MLP及仅风场的U‑Net等模型对比，实验显示在快速增压样本中V_max MAE从12.2 m s⁻¹降至5.4 m s⁻¹，整体MAE约5.9 m s⁻¹，ERA5输入虽可提升部分指标但并非必需

**⚠️ 局限性**

局限性包括训练样本稀缺、SAR覆盖不连续、未对ERA5输入组合做全面 ablation、未探索更复杂网络结构及上层海洋变量的潜在影响

---

## 610. VoxCanvas-41M: A Large-Scale Bilingual Dataset for Instruction-Based Speech Editing and Its Learning Dynamics

**arXiv ID:** 2611.08251 | [PDF](https://arxiv.org/pdf/2611.08251v1)

**作者:** Zhikang Niu `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种大规模双语指令式语音编辑数据集VoxCanvas（约4100万对，涵盖中文和英文），并基于此数据集对AuK模型进行LoRA微调，实现高质量的语音编辑；同时系统分析了语音编辑模型的学习动力学，包括编辑任务的收敛行为、任务交互、数据比例、编辑与生成的关系、课程学习和数据反转等。

**💡 创新点**

①构建首个覆盖内容编辑和非语言编辑的双语大规模语音编辑数据集；②设计可扩展的多阶段数据构建流水线（重识别、对齐、LLM生成指令、TTS重构、质量过滤、人工检测）；③在学习动力学方面首次量化不同编辑操作（插入、替换、删除）和多跨度编辑、位置参考编辑以及非语言编辑的收敛差异与任务交互；④发现编辑任务与文本到语音生成任务互相促进，并证明数据反转能显著提升音质。

**🔧 技术方法**

使用先进的TTS模型F5-TTS、LLM Qwen系列（文本生成、对齐、指令化）、强大的ASR/Qwen-ForcedAligner进行重识别与对齐；模型方面采用AuK（Qwen2.5-Omni编码器+VAE+flow‑matching Transformer）并通过LoRA进行高效微调；在实验中还使用AdamW、EMA权重、Euler采样等技术。

**📊 数据集**

主数据集为VoxCanvas（≈41M编辑对，20.5M中英各一份），来源于Emilia（中文、英文）及六个非语言语料库；此外还使用GigaEdit‑L、RealEdit、Ming‑Freeform‑Audio‑Edit等公开基准数据进行评测；训练集亦包含生成任务的Seed‑TTS数据。

**📈 对比分析**

评估指标包括SpeechEditBench的内容与非语言编辑准确率ACC、Ming‑Freeform‑Audio‑Edit的WER、ACC、SIM、no‑edit WER，实验表明LoRA微调的AuK+VoxCanvas相较于基线AuK以及AuK+GigaEdit‑L在两大基准上均获得显著提升，内容编辑ACC提升至约91%、非语言编辑提升至53%，且在音质和说话人相似度上均优于对照模型。

**⚠️ 局限性**

当前数据集仅覆盖内容编辑和非语言编辑，未包含情感、说话风格或多轮历史依赖编辑；缺乏更丰富的非语言编辑场景；此外，生成目标的合成质量仍有提升空间，且对低资源语言的扩展尚未展开。

---

## 611. InstantHand: Feed-Forward 3D Gaussian Splatting for Animatable Hand Avatar Reconstruction from a Single Image

**arXiv ID:** 2611.08330 | [PDF](https://arxiv.org/pdf/2611.08330v1)

**作者:** Chanho Kim `[一作]` (UNIST), Seungryul Baek `[通讯]` (UNIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个单图像即时可动画的3D高精度手部Gaussian avatar，避免了测试时优化。

**💡 创新点**

创新点在于使用MANO-锚定的canonical Gaussian结构、Soft Alpha Band正则化控制薄指节以及Reliability-gated Appearance Transfer按Gaussian渲染贡献自适应地融合源图像外观。

**🔧 技术方法**

采用3D Gaussian Splatting、DINOv3特征、条件Transformer、MANO参数、可靠性门控等技术。

**📊 数据集**

训练使用InterHand2.6M、HO3Dv3、Goliath、Hand Appearance、PALM等数据集，评测在InterHand2.6M和HanCo。

**📈 对比分析**

与基于多视图或单帧优化的基线相比，InstantHand在单图像重建中达到或超过最佳PSNR/SSIM/Lpips，并以2.84秒构建、161FPS渲染，速度提升400×，零样本在HanCo仅差0.14dB PSNR。

**⚠️ 局限性**

局限在于缺乏视角/姿态相关的外观变化，不能很好处理自遮挡时的阴影等光照效应；同时依赖手部分割、MANO与相机估计的精度。

---

## 612. Decision Models as Retrieval Gates Against Injection and Pollution

**arXiv ID:** 2611.08325 | [PDF](https://arxiv.org/pdf/2611.08325v1)

**作者:** Mahdi Ghadamyari `[一作]` (Google), Reza Habibi `[通讯]` (University of California Santa Cruz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估开源决策模型（如Kev-4B）作为检索门，阻止检索增强生成（RAG）系统中的指令注入与污染攻击，并与已发布的清理+引用提示防御做对比。

**💡 创新点**

创新点在于提出轻量级决策门，可读取检索到的原始HTML并通过三问式评判过滤不安全或无关段落，显著降低攻击成功率，且成本仅为传统LLM判断的一小部分。

**🔧 技术方法**

技术包括BM25检索、Kev/Tev1/Von/Laya等决策模型、Llama-3.1‑8B 与 Llama‑3.3‑70B 生成模型、LLM判别器（DeepSeek V4.1 Flash）以及文本/HTML预处理（清理、Unicode正则化）。

**📊 数据集**

使用了两大数据集：一是基于Reddit问答构建的1,000条注入测试实例（含攻击/控制、无关、硬负样本及毒化页面），二是改造后的MS MARCO 200条查询与20篇生成式点击诱导文档，用于污染实验。

**📈 对比分析**

与单一清理+引用提示（ASR≈13.6%）相比，Kev门将攻击成功率降至≈5.7%，同时保持≈86%干净查询的响应性；与LLM判断相比，Kev门在0.015 $/1,000段落、76 ms延迟下实现了93% AUROC，远低于70B判别器的1.15 s/段落与0.15 $/1,000段落成本。

**⚠️ 局限性**

局限性包括：数据集与原始研究不完全一致（硬负样本来自有限模板）、缺少关键消融实验、TF‑IDF/Perplexity基线实现弱、对抗者能访问门评分或使用更强重写器的场景未评估。

---

## 613. RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines

**arXiv ID:** 2611.08308 | [PDF](https://arxiv.org/pdf/2611.08308v1)

**作者:** Roko Parac `[一作]` (Imperial College London), Alessandra Russo `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于自然语言任务描述，自动生成奖励机（Reward Machine）并通过视觉‑语言模型（VLM）对图像中的原子命题进行归一化，以实现非马尔可夫（non‑Markovian）任务的强化学习；

**💡 创新点**

①利用大型语言模型（LLM）一次性推导任务命题与奖励机结构；②采用VLM作为无监督的命题归纳函数；③在奖励机状态上维护概率信念以抵消VLM噪声；④将生成的“进度”估计（Generative Value Learning, GVL）作为稠密奖励；

**🔧 技术方法**

大型语言模型（Gemma‑4‑31B）生成奖励机与命题；视觉‑语言模型（同一模型）提供每帧命题标签；贝叶斯状态信念更新；价值迭代与GVL进度预测；离线与在线强化学习（IQL/Flow‑Q‑Learning/Proximal Policy Optimization）；

**📊 数据集**

GeoGrid（8×8 网格）、OGBench‑Scene（机器人操控场景）、MiniWorld‑Pickup（三维目标收集）这三组任务数据集；

**📈 对比分析**

在三大域上与BC、GVL、VLM‑SD、GCR等基线对比；RM‑VLM在GeoGrid几乎完美（≥99%），OGBench‑Scene 4/5 任务≥65%成功率，MiniWorld‑Pickup 约70%；与基线相比显著提升，尤其在涉及子目标顺序和安全约束的任务上；

**⚠️ 局限性**

VLM 的命题识别精度依赖视觉可辨性，噪声仍会影响奖励信号；需要估计每个命题的真/假正率以构造信念；奖励机仅能描述正则语言（regular‑language）任务，无法处理计数、可变长度依赖等更复杂约束。

---

## 614. ManimCode: Learning Visual Self-Correction for Educational Animation

**arXiv ID:** 2611.08294 | [PDF](https://arxiv.org/pdf/2611.08294v1)

**作者:** Zhengyang Liang `[一作]` (Singapore Management University), Lizi Liao `[通讯]` (Singapore Management University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

创建了ManimData数据集和ManimCode框架，实现了基于渲染-审阅-修订循环的教育动画自动生成

**💡 创新点**

创新点在于完整记录渲染后错误、视觉批评与修订轨迹，并利用角色条件监督与强化学习联合训练单一4B模型实现全流程自修复

**🔧 技术方法**

采用角色条件监督、强化学习（自生成状态）、自检渲染、API知识库检索和多模态评估（Gemini、CLIP）等技术

**📊 数据集**

使用24,034条经渲染验证、质量过滤的ManimData记录，涵盖请求、代码、视频、错误、评审与修订轨迹

**📈 对比分析**

与现有通用模型和专用Manim模型对比，ManimCode在ManimData-test和ManimBench的渲染成功率分别达到96.6%和97.0%，并在Code2Video、TheoremExplainAgent等长视频系统中显著提升完成率与视频质量

**⚠️ 局限性**

局限包括对渲染预算的依赖、对极端视觉缺陷的修复能力有限，以及在更广泛领域的通用性尚未充分验证

---

## 615. The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model

**arXiv ID:** 2611.08262 | [PDF](https://arxiv.org/pdf/2611.08262v1)

**作者:** Mathis Immertreu `[一作]` (University Erlangen-Nürnberg), Achim Schilling `[通讯]` (Heidelberg University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 Qwen3‑VL‑8B‑Instruct 上通过 100 张图片和 53 种提示条件，系统研究了 persona 条件（身份内容）与提示框架对视觉趣味性评分和内部表示几何的影响。

**💡 创新点**

发现提示框架对内部表示几何影响最大，而 persona 内容对最终评分影响最大，二者在表示层次与行为表现上呈现逆向耦合，揭示了内部表示与行为相关性的基本分离。

**🔧 技术方法**

采用多模态 Transformer 的层级隐藏状态提取、MDS 投影、GDV 聚类度量以及方差分解和混合效应模型进行定量分析。

**📊 数据集**

使用 Flickr 公开子集（100 张图片），每张图片均具备“广泛兴趣度”标签，并构建两套 persona 方案（社会人口学属性与 Big‑Five 性格百分位）。

**📈 对比分析**

通过方差分解、混合效应检验与 Pearson 相关矩阵比较，结果显示 persona 对评分贡献约 33%（Big‑Five）或 17%（社会人口学），提示风格仅约 5%，并未在模型输出中产生显著变化。

**⚠️ 局限性**

限制包括单一 8B 模型、贪婪解码、有限的 persona 与提示样式数量、缺乏人类对照评价以及对更大模型或不同任务的泛化未知。

---

## 616. All minimum cuts of symmetric submodular functions

**arXiv ID:** 2611.08253 | [PDF](https://arxiv.org/pdf/2611.08253v1)

**作者:** Chao Xu `[一作]` `[通讯]`, Chao Xu

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文设计了一种O(n³)时间算法，利用Queyranne排序和digraph表示，求得对称子模函数的Cunningham规范分解，进而得到所有最小割；

**💡 创新点**

创新点在于证明了对称子模函数的最小st割可以通过O(n²)时间构造的有向图完全表示，并结合Chekuri‑Xu算法实现整体O(n³)复杂度；

**🔧 技术方法**

主要技术包括Queyranne的排序、贪心基底与紧集的digraph构造、流与闭集的互补松弛、以及递归森林结构的预排序；

**📊 数据集**

该研究为理论算法，未使用具体数据集，所有分析均基于值 oracle 的调用；

**📈 对比分析**

与以往需要O(n⁴)值 oracle调用的算法相比，本文将复杂度降低至O(n³)，且实现中仅需O(n²)额外时间构造st割图；

**⚠️ 局限性**

局限在于仅适用于对称子模函数，且依赖值 oracle 的假设，未给出对非对称或近似情形的扩展及实验验证。

---

## 617. T3R: Training-Free Two-Stage Token Refinement Towards Efficient and Robust VLA Models

**arXiv ID:** 2611.08413 | [PDF](https://arxiv.org/pdf/2611.08413v1)

**作者:** Mohammed Hassan `[一作]` (University of Texas at Austin), Chaojian Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练的两阶段令牌精炼框架T3R，用于高效鲁棒的视觉-语言-动作（VLA）模型推理

**💡 创新点**

创新点在于结合分割引导的令牌剪枝与显著性引导的注意力偏置，既降低令牌数量又恢复因剪枝导致的信息损失

**🔧 技术方法**

采用SigLIP视觉-语言相似度、EfficientTAM分割、SAM分割器进行空间剪枝，并利用Integrated Gradients进行显著性计算，随后在Transformer中注入注意力偏置

**📊 数据集**

主要使用LIBERO、SIMPLER、RoboTwin等机器人操纵数据集（包含多摄像头和单摄像头情形），并在OpenVLA-OFT、CogACT、π_0.5等VLA骨干上评估

**📈 对比分析**

与无剪枝基线、TeamVLA、ADP、FastV等方法对比，T3R在LIBERO-Spatial上可削减85%视觉令牌、成功率保持96%（仅低1%），推理延迟下降约14%，并在OOD物体放置扰动下提升至7个百分点

**⚠️ 局限性**

局限包括对分割器的依赖（可能误剪），IG显著性一次性计算成本、单摄像头环境下效果有限，以及在真实机器人与边缘硬件上的能耗与延迟验证不足

---

## 618. LABBA: LLM-Assisted Behavioral Biometric Authentication

**arXiv ID:** 2611.08343 | [PDF](https://arxiv.org/pdf/2611.08343v1)

**作者:** Howard Halim `[一作]` (Singapore University of Technology and Design), Jianying Zhou `[通讯]` (Singapore University of Technology and Design)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种利用大语言模型进行上下文推理的手机吹气声行为生物识别框架，结合频谱与时序特征实现自适应身份验证。

**💡 创新点**

创新点在于把LLM作为推理模块替代传统阈值决策，并通过结构化提示将频谱/时序相似度映射为可解释的标签或分数，实现场景感知和抗欺骗的智能决策。

**🔧 技术方法**

技术手段包括GFCC+CNN嵌入、双尺度高斯平滑+DTW时序特征、规则/分数两种特征归一化、Phi‑3 Mini Instruct LLM推理以及 Prompt Engineering。

**📊 数据集**

使用了50名参与者在不同姿势、噪声环境下收集的500次吹气声样本作为实验数据集。

**📈 对比分析**

与传统无LLM阈值基线对比，LLM规则模式准确率达99.46%、EER 9.24%；LLM分数模式准确率99.38%、EER 7.09%，显著提升了误拒率和误接受率。

**⚠️ 局限性**

局限性包括推理耗时较长（8–11 s，CPU端），缺乏GPU/云加速；LLM的可解释性仍受限；对极端对抗攻击、隐私保护与实时部署的评估尚待进一步研究。

---

## 619. Benchmarks as Software: Terminal-Bench and the Case for Continuous Benchmarks

**arXiv ID:** 2611.08417 | [PDF](https://arxiv.org/pdf/2611.08417v1)

**作者:** E. Kelly Buchanan `[一作]` (Stanford University), Christopher Ré `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套持续验证工作流，用于监测、识别并修复大型语言模型代理基准（Terminal‑Bench）中的任务缺陷，并发布更新版Terminal‑Bench 2.1。

**💡 创新点**

创新点在于：①将社区报告、任务工件、代理轨迹与自动 oracle 检查相结合，形成可持续的、可复现的审核流程；②通过三阶段审核（证据收集、人工标签、验证器复核）实现对任务缺陷的精确定位与修复；③对资源预算进行动态校准，证明资源配置同样是基准结果的重要决定因素。

**🔧 技术方法**

使用技术包括：大型语言模型（Claude Opus 4.6、GPT‑5.4）作为审计助手；自动化 oracle 与验证器脚本；Docker/Daytona 运行时环境；人类审阅与修复工作流；以及对任务工件和社区 issue 的结构化提取。

**📊 数据集**

使用数据集：Terminal‑Bench 2.0 原始任务与 19 名代理的评测记录；Terminal‑Bench 2.1 修复后的任务与对应评测；此外利用公开的 GitHub issue 与 pull‑request 作为社区报告来源。

**📈 对比分析**

通过在 19 个代理上分别跑 TB2.0 与 TB2.1，测得准确率提升约 10 个百分点，排行榜 Spearman ρ = 0.958；在扩展时间预算后发现 GPT‑5.4 的 harness 差距从 20pp 缩小至 0，GPT‑5.4 mini 的差距从 30pp 缩小至 5pp，证明资源预算对性能评估有显著影响。

**⚠️ 局限性**

局限性包括：低成本审计模型的归因一致性不足；依赖社区主动报告，若缺失会漏掉缺陷；修复过程仍需人工判断，无法完全自动化；工作流针对 Terminal‑Bench 设计，推广到其他基准需进一步验证；以及修复后基准仍会随外部依赖漂移而失效。

---

## 620. Social Humanoid: Socially Responsive Human-Humanoid Interaction

**arXiv ID:** 2611.08407 | [PDF](https://arxiv.org/pdf/2611.08407v1)

**作者:** Vongani H. Maluleke `[一作]` (University Of California Berkeley), Angjoo Kanazawa `[通讯]` (University Of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一个基于人类‑仿人交互的全身运动生成与执行系统；

**💡 创新点**

创新点在于两阶段自适应重映射、共享码本VQ‑VAE与Diffusion Forcing Transformer的协同设计，实现场景中的即时、可执行的社交响应；

**🔧 技术方法**

技术包括RGB‑D感知、共享码本VQ‑VAE、Diffusion Forcing Transformer以及NVIDIA SONIC低层控制器；

**📊 数据集**

使用了五个运动捕捉数据集（Chi3D、Hi4D、Harmony4D、Inter‑X、Embody3D）经自适应重映射得到260小时的人类‑仿人交互基准；

**📈 对比分析**

与检索、随机、Ready‑to‑React 等基线对比，在MPJPE、MPJVE、IP、MI 等指标上均优于基线，生成的运动在仿真与真实硬件上成功率接近100%；

**⚠️ 局限性**

限制包括缺乏手部、面部表情及物体交互建模，依赖现成姿态估计与SONIC，且未进行人机交互质量的主观评估。

---

## 621. A Multimodal Haptic Interface for Simultaneous Softness and Thermal Modulation

**arXiv ID:** 2611.08398 | [PDF](https://arxiv.org/pdf/2611.08398v1)

**作者:** Joshua Brown `[一作]` (Imperial College London), Lynette Jones `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种结合粒子堵塞软化和液体热交换的软热触觉显示器，能同步调节触感柔软度和表面温度。

**💡 创新点**

创新点在于实现无泵粒子堵塞与热流通道的整合，兼顾柔软度可调和高温度精确控制。

**🔧 技术方法**

使用了粒子堵塞技术、热电Peltier模块、液体循环冷却/加热以及闭环PID控制。

**📊 数据集**

主要使用实验测量数据：力计、热敏电阻和热像仪记录的柔软度和温度曲线。

**📈 对比分析**

通过对柔软度的线性增加（约300%）和温度在10~50°C范围内±0.2°C的闭环控制进行验证，显示性能优于现有单一模式触觉装置。

**⚠️ 局限性**

局限包括温度响应慢（上升/下降均需数分钟）、30秒延迟、热惯性大以及初始柔软度的高滞后。

---

## 622. Adaptive Low-Light Enhancement for Road Anomaly Detection: A Multi-Architecture Benchmark and Edge Deployment Study

**arXiv ID:** 2611.08393 | [PDF](https://arxiv.org/pdf/2611.08393v1)

**作者:** Obed Honour Eje `[一作]` (Federal University of Technology Minna), Taliha Abiodun Folorunso `[通讯]` (Federal University of Technology Minna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在模拟低光与恶劣天气条件下，对四种目标检测架构进行基准测试，并提出了基于亮度的自适应路由器以动态选择是否使用低光增强。

**💡 创新点**

创新点在于：①证明统一低光增强在暮光条件下会显著降低检测精度；②设计了无需训练、仅依赖Rec. 601亮度阈值的三路路由器，显著恢复暮光损失；③在Jetson Nano上完成了端到端的边缘推理性能评估。

**🔧 技术方法**

主要技术包括：Physically‑grounded 亮度与天气退化合成、SCI（Self‑Calibrated Illumination）低光增强网络、基于阈值的亮度自适应路由器、YOLOv8n/YOLOv12n、RT‑DETR、Faster R‑CNN 等检测框架，以及 TensorRT、FP16 推理优化。

**📊 数据集**

使用合并自公开 pothole 数据集（Kaggle Rajdalsaniya 与 Roboflow speed‑bump），共 4,444 张图像，划分 70/15/15 用于训练/验证/测试。

**📈 对比分析**

通过 mAP@50 评估：在暮光下统一 SCI 使 mAP 降低 16–31%；采用亮度路由器后恢复 71–90% 的暮光损失，夜间对 RT‑DETR 进一步提升 +0.071 mAP；在 Jetson Nano 上，YOLOv8n 在所有分支均可达 15–26 FPS，满足实时性。

**⚠️ 局限性**

局限性：①实验使用合成退化而非真实夜间/雨雾数据；②路由阈值固定，未针对两阶段检测器进行调优；③未加入天气专属恢复模块（去雾、除雨），且未探索更高压缩或更低功耗平台。

---

## 623. Hybrid Machine Learning for Imbalanced Landing-Phase Safety Event Prediction from Flight Data

**arXiv ID:** 2611.08386 | [PDF](https://arxiv.org/pdf/2611.08386v1)

**作者:** Taha Asghar `[一作]` (National University of Sciences and Technology), Luqman Ahmed `[通讯]` (National University of Sciences and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种混合特征融合框架，用于在极度不平衡的登陆阶段飞行数据中预测多类安全事件；

**💡 创新点**

创新点在于将LSTM自编码器学习到的时间序列重构误差与参数特定的统计特征融合，并采用类别加权XGBoost实现对罕见事件的精确识别；

**🔧 技术方法**

使用的技术包括LSTM自编码器、统计特征提取、类权重损失、XGBoost集成学习及Tree SHAP解释；

**📊 数据集**

使用的数据集为NASA DASHlink公开的约99,000条航班终段窗口，包含20个飞行参数，标注四类事件（Normal、Speed High、Path High、Flaps Late）；

**📈 对比分析**

通过对比集中学习与机场特定学习、五种模型基准以及多次随机划分重训练，结果显示宏观召回率平均超过0.90，最佳为0.9542（KMSP）和0.9046（KDTW），同时证明融合方法优于仅使用统计特征或阈值门控；

**⚠️ 局限性**

局限在于仅针对单一区域喷气机和三家机场，罕见事件样本稀少，未验证跨机型或新机场的泛化性，且对时间漂移和运营条件变化的鲁棒性仍待进一步评估。

---

## 624. FedReason: Large Language Models as Weight-Space Meta-Aggregators for Heterogeneous Federated Learning

**arXiv ID:** 2611.08383 | [PDF](https://arxiv.org/pdf/2611.08383v1)

**作者:** Ali Mohammadi Ruzbahani `[一作]` (University of Calgary), Hadis Karimipour `[通讯]` (University of Calgary)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在联邦学习中引入一种基于大型语言模型的元聚合器FedReason，允许服务器在不训练任何模型的前提下，通过读取客户端模型权重并生成聚合程序来动态选择聚合规则。

**💡 创新点**

创新点在于将权重空间分析与LLM编程相结合，使得聚合决策不再依赖预设规则或客户端训练调整，而是通过LLM实时读取权重特征并写出可执行的聚合逻辑，且仅使用有限的代理评估进行验证。

**🔧 技术方法**

采用Claude Opus 5.5等大型语言模型，通过Python沙盒执行代码、计算余弦相似度、权重剪裁、基于类别的权重调节以及服务器步长等工具，并结合代理集（500张图像）进行前向评估。

**📊 数据集**

使用CIFAR-10和CIFAR-100数据集，实验涵盖标签不平衡（Dirichlet划分）、模型宽度异质性以及拜占庭攻击（签名翻转和标签翻转）等多种场景。

**📈 对比分析**

与14个基线（FedAvg、FedProx、FedAvgM、FedAdam、FedNova、SCAFFOLD、FedAdp、HeteroFL、FedRolex、median、trimmed mean、Krum、FLTrust、ProxySel）对比，FedReason在标签偏斜下平均提升1.26个百分点，成为CIFAR-100上最优的服务器端聚合规则，并在无服务器训练的情况下逼近FLTrust（相差≈1个百分点），但在SCAFFOLD下仍略逊一筹。

**⚠️ 局限性**

局限性包括对代理集质量高度依赖、无法处理部分参与或大规模客户端群组、未对自适应攻击进行评估、需要暴露客户端权重信息可能带来隐私风险、LLM调用成本与可重复性受模型版本影响，以及实验仅在单一架构和有限随机种子上验证。

---

## 625. MOMASPLAT: Dynamic Novel-View Demonstration Augmentation for Mobile Manipulation

**arXiv ID:** 2611.08372 | [PDF](https://arxiv.org/pdf/2611.08372v1)

**作者:** Sateesh Kumar `[一作]` (University of Texas at Austin), Roberto Martín-Martín `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种用于移动操纵的视角增强框架，能够从少量原始演示中合成完整的动态演示（含接触后交互），并用生成的数据训练鲁棒的视觉-运动策略。

**💡 创新点**

创新点在于：①将场景与被操纵物体分别用3D高斯分布重建；②利用导航轨迹桥接两者的坐标系并对齐；③针对新基座位姿进行光照一致的三流合成（场景、物体、机器人）并对动作进行几何一致的重映射，实现了跨视角的全程演示合成。

**🔧 技术方法**

技术手段包括：结构光/相机位姿估计（COLMAP）、3D Gaussian Splatting、基于 Feed‑forward 3D 基础模型（FastVGGT）进行物体位姿推断、Sim(3)对齐、OmniGibson 渲染、alpha 合成、Diffusion Policy 作为行为生成器。

**📊 数据集**

数据集：在桌面、厨房、客厅三种真实场景下收集了 5 个任务（瓶子放置、抽屉开启、笔记本关闭、遥控器捡取、空气炸锅开启）共约 450 条演示；同时采集了 9 条导航轨迹作为多视角支持；在模拟环境中构建了相似任务用于评估生成质量。

**📈 对比分析**

与四种基线（仅原始数据、动作噪声、VISTA、One‑Thousand Trajectories）对比；在 12 个未见基座位姿上，平均成功率从最强基线 17% 提升至 37%，在距离 30–50 cm 的远场位置保持 33–50% 的成功率，显著优于其他方法。

**⚠️ 局限性**

局限性：依赖相对静态场景和可见刚性物体；对柔性物体、严重遮挡、低纹理表面处理不足；场景重建忽略被遮挡区域，导致在被物体移动后出现缺失；生成的数据不包含物理接触动力学，无法模拟真实碰撞效果。

---

## 626. Information Elicitation by Ordinal Agreement

**arXiv ID:** 2611.08366 | [PDF](https://arxiv.org/pdf/2611.08366v1)

**作者:** Yiling Chen `[一作]` (Harvard University), Fang-Yi Yu `[通讯]` (George Mason University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一种基于 Kendall tau 的同僚预测机制，激励在缺乏真相验证的环境下，评估者诚实地报告秩序信息。

**💡 创新点**

在 Bayesian SOC（顺序一致性）信息结构下证明该机制在随机占优与强随机占优意义上具有诚实激励，同时支持固定形状桶排序、top‑K 与子集分配等多种报告格式。

**🔧 技术方法**

采用 Kendall tau 距离的成对一致性得分、SOC 条件下的概率论证明、泡沫排序路径，以及随机占优与强诚实性分析等技术。

**📊 数据集**

实验使用合成 Mallows 与 Thurstone 模型数据，以及真实的 SUSHI 口味偏好数据集。

**📈 对比分析**

通过与 Enforced Agreement、BPP 等基线的支付 ECDF 对比与敏感度测量，Kendall tau 机制在支付分布和对报告质量的敏感性上表现最优。

**⚠️ 局限性**

依赖 Bayesian SOC 条件，无法处理存在“cheap 信号”或非一致性顺序信息的场景；需要评估子集有重叠才能对每对项目进行评分，对极度稀疏的评估分配仍有限。

---

## 627. Radio-to-Pose Information for Gauge-Drift Correction in Cooperative Localization

**arXiv ID:** 2611.08361 | [PDF](https://arxiv.org/pdf/2611.08361v1)

**作者:** Aditya Iyer `[一作]` (Brown University), Nora Ayanian `[通讯]` (Brown University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究使用注册的无线信号衰减地图来纠正协同定位中的 gauge drift，并提出 Radio-to-Pose Information (RPI) 接口将功率残差转换为高斯信息矩阵供姿态估计使用。

**💡 创新点**

证明相对测量加权无法直接约束共同运动；设计 RPI 通过地图衰减信息提取共享运动信息并保留地图相关性；给出理论分析和闭式约束条件。

**🔧 技术方法**

高斯因子图求解、射频信号衰减模型、格点离散化、Schur 补、非线性迭代仿真评估等技术。

**📊 数据集**

仅使用仿真生成的机器人轨迹和衰减地图（如9个RBF系数或40×40格点），未使用真实测量数据。

**📈 对比分析**

与全联合优化(Joint)、固定权重相对测量(Scalar)、每步地图(Per-step map) 进行比较；RPI 与 Joint 结果一致，非线性实验中将共同位置 RMSE 从 0.542 m 降至 0.140 m，保持相对误差≈0.06 m；对地图误差的鲁棒性有限。

**⚠️ 局限性**

仅在仿真中验证；对地图注册误差未建模；需要完整且先验已知的衰减地图；真实环境下多径、收敛性等问题待进一步评估。

---

## 628. USB Gym: A Gymnasium Testbed for Closed-Loop Ultrasound View Matching

**arXiv ID:** 2611.08352 | [PDF](https://arxiv.org/pdf/2611.08352v1)

**作者:** Thanapong Intharah `[一作]` (Khon Kaen University), Hao Dong `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了USB Gym闭环测试平台，评估将公开的超声引导组件（如分割器、控制器、奖励等）逐槽插入后在模拟器中是否能实现探头视图匹配；

**💡 创新点**

首次提供可插拔闭环实验框架并通过“oracle ladder”对不同槽位方法进行统一量化，揭示当前公开方法无法逼近姿态oracle，系统化验证超声引导技术的瓶颈；

**🔧 技术方法**

采用物理渲染（LOTUS/SIMUS）、SonoSPADE纹理、基于Mask的视觉伺服、REINFORCE/DAgger强化学习、SSIM/NMI/LPIPS等奖励；

**📊 数据集**

使用公开CT数据集TotalSegmentator（13训练+6留存患者）、Kaggle Vitale、KidneyUS、Annotated Ultrasound Liver等实时图像，以及外部AMOS 2022 CT做复制；

**📈 对比分析**

在同一6个留存患者的60条episode上逐槽替换方法，比较最终平移误差与姿态oracle误差；最优组合（SSIM奖励）得到13.7mm误差，仍比oracle高12mm，整体性能低于预期；

**⚠️ 局限性**

仅在模拟器中评估，未涉及真实探头/机器人；渲染器假设硬组织、固定声学参数、无呼吸等，纹理和分割器未针对性调优，导致与真实超声差距大。

---

## 629. Connectivity in a New Bounded-Space Dynamic Model

**arXiv ID:** 2611.08351 | [PDF](https://arxiv.org/pdf/2611.08351v1)

**作者:** Bardiya Aryanfard `[一作]` (Institute of Science and Technology Austria), Virginia Vassilevska Williams `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种新的动态算法模型——动态oracle模型，要求算法在读写工作内存有限（S 词）且只能通过查询读访问外部读仅图形的接口来维护一个固定终点对 s、t 的连通性，并给出了该模型下增量和全动态情形的时间–空间权衡上界与下界。

**💡 创新点**

创新点在于：①首次把传统的静态读‑写空间权衡与动态更新结合；②设计了随机化的 landmark‑based 方案（BFS、Metropolis/普通随机游走、有效电阻分析）来在极小空间内实现 s–t 连通性；③利用通信复杂度与概率编码技术证明了在此模型下多项式更新时间是不可避免的；④对森林图给出了匹配的确定性上界，证明了下界的 tightness。

**🔧 技术方法**

主要技术包括：随机游走与有效电阻理论、Metropolis walk、landmark 采样与稀疏化、潜能函数与势能分析、通信复杂度（Index、PPHP）以及概率编码与信息计数的 lower bound 证明。

**📊 数据集**

本文不使用任何实验数据集，全部工作均为理论分析与证明；主要使用抽象的随机图构造（如带置换的槽‑物品构造、clique‑gadgets 等）来实现硬实例。

**📈 对比分析**

与传统的全动态连通性数据结构（如 DSU、Euler‑tour 树、top‑tree）相比，本文的算法在空间受限时可达到的更新时间在矩阵模型下为 Ω(n²/S)（增量）或 Ω(n²/S)（全动态）与上界一致；在列表模型下则为 Ω(n²/S²) 的最坏情况和 Ω(n/S) 的平摊情况。实验或数值对比并未给出，但理论上证明了在 S ≪ n 时传统数据结构的多项式时间上界是不可打破的。

**⚠️ 局限性**

限制包括：①仅针对单对连通性而非完整连通分量；②结论仅适用于随机化算法，确定性上界相对较弱；③下界与上界之间仍留有多项式对数因子；④模型假设图是外部只读并且查询是单位成本，现实系统的 I/O 与同步成本未被考虑；⑤对极端小空间（如常数词）时的细节尚不完整。

---

## 630. Enhancing Local and Global Structure in Compressed AI Agent Memory

**arXiv ID:** 2611.08341 | [PDF](https://arxiv.org/pdf/2611.08341v1)

**作者:** Henry Han `[一作]` `[通讯]` (Baylor University), Henry Han (Baylor University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在大型 AI 代理记忆中提出了一种新型量化方法——局部感知量化（LAQ），通过训练 OPQ 的码本以在保留近邻结构的同时显著提升记忆的全局结构（全局保真度 G）。

**💡 创新点**

创新点在于：① 证明了记忆的全局保真度可由跨对（cross‑pair）排名错误率上界；② 将该上界转化为可训练的 hinge 损失；③ 在无标签、无查询的情况下，仅修改 OPQ 码本即可提升全局结构；④ 对传统 OPQ 的训练成本进行显著降低。

**🔧 技术方法**

技术包括：产品量化（OPQ）、交叉对排名误差的 hinge 损失、无标签的局部感知训练、直通（straight‑through）梯度、监控集的早停、基于概率分布的远端误差预测等。

**📊 数据集**

使用了多种财经与通用语料：FinQA、Multi‑Doc‑2025、LoCoMo、20 Newsgroups、AG News、FinDER 以及 100 万条 MS‑MARCO 句子，配合 6 种句向量编码器（BGE‑base、BGE‑large、E5‑base、MobileCLIP2‑L/14、RoBERTa‑base、GloVe、Titan V2）。

**📈 对比分析**

与 OPQ、PQ、RQ、LSQ++、PCA+int8、RaBitQ、ScaNN 等同等尺寸量化器对比，LAQ 在所有 52 个 hold‑out 运行中均提升全局保真度（平均 +0.012），保持近邻保真度不变；在全局相关操作（主题聚合、删除、公司/行业映射、全局距离相关性等）上表现最优；训练时间仅为 OPQ 的数十秒至两分钟，显著低于增量/加法量化器。

**⚠️ 局限性**

局限性：① 需要先行训练 OPQ 并在此基础上微调，若无 OPQ 初始码本则需额外步骤；② 证明和损失设计主要针对产品量化，尚未验证对残差或加法量化的直接适用性；③ 依赖 Gaussian 噪声假设，理论上在极端分布或高噪声场景下可能失效；④ 对极大规模（百万级）记忆训练仍需采样参考集，精度依赖样本选择；⑤ 仅提升全局保真度，近端检索性能不再受益，若需双端优化需进一步研究。

---

## 631. The Power of Meta Access in the Software of the Future --- Demonstrated on LLM Chat UIs and Chat AI

**arXiv ID:** 2611.08428 | [PDF](https://arxiv.org/pdf/2611.08428v1)

**作者:** Julian Kunkel `[一作]` (Universität Göttingen), Ali Doosthosseini `[通讯]` (Universität Göttingen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个在 Chat AI 客户端中的“Meta Chat”原型，使聊天助手能够获取、呈现并安全地修改整个工作区（对话、主题、设置、偏好、记忆等）

**💡 创新点**

提出了“元访问阶梯”概念，系统性划分了从功能教学到完整工作区编排的五个安全级别，并将这些级别与证据、确认、可逆性、审计等守护约定相结合

**🔧 技术方法**

采用检索增强生成、摘要压缩、局部词义搜索、结构化上下文包装、类型化能力注册表、链接引用、用户确认卡片以及撤销机制等技术手段构建元访问层

**📊 数据集**

实验使用的并非公开数据集，而是 Chat AI 客户端本地存储的真实用户对话、主题、设置与记忆；原型在这些数据上演示功能和安全性

**📈 对比分析**

原型与传统基于文本的聊天助手或单一功能工具相比，在信息检索、导航、功能教学和安全变更上提供更直观、更可追踪的交互；具体性能指标（如召回率、时间节省等）未给出，作者提出了未来评估议程

**⚠️ 局限性**

局限性包括：摘要可能过时或失真；检索覆盖范围受上下文截断限制；存在幻觉与隐私泄露风险；尚未实现完整的层级 5 工作区编排；缺乏标准化基准与跨应用兼容性评估

---

## 632. Failure Prediction and Classification in Vision-Language-Action Models Via Calibrated Latent Probing

**arXiv ID:** 2611.08425 | [PDF](https://arxiv.org/pdf/2611.08425v1)

**作者:** Michal P. Podolinsky `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级的探针，监测视觉-语言-动作模型（VLA）的潜在失败，并在失败发生前发出预警，同时预测失败模式，支持实时控制与干预。

**💡 创新点**

创新点在于：①在VLA潜在层中引入LSTM+MLP探针，并通过预失败段监督实现更早的故障检测；②采用Learn‑then‑Test（LTT）框架联合校准FPR与早期TPR，提供可靠的误报/漏报界限；③探针不仅给出失败概率，还输出失败模式，提升可解释性；④系统实现高效，GPU/CPU延迟不足1%。

**🔧 技术方法**

技术包括：1) VLA潜在层嵌入作为输入；2) 单层LSTM编码序列，随后使用三个MLP头（失败得分、失败模式、CLIP对齐）；3) 预失败段监督（pre‑onset labeling）和三种损失（binary cross‑entropy、mode CE、CLIP对齐）；4) Learn‑then‑Test阈值校准；5) 训练、校准、推理在RTX‑4090/CPU上实现毫秒级延迟。

**📊 数据集**

使用由3,180条带注释的操控任务组成的数据集，覆盖LIBERO、LIBERO‑Occ、LIBERO‑X以及8个真实机器人任务，总计约288k条VLA嵌入；所有失败事件均由Gemini自动标注并手工复核。

**📈 对比分析**

与SAFE‑LSTM、SAFE‑MLP、STAC、ActProbe、FailureSpot、Mahalanobis等基线比较；在未见任务上，早期检测AUC提升最多30%；校准后FPR与TPR_early的联合保证在见任务上100%通过，未见任务中保持86%；失败模式预测top‑3准确率提升44%；整体推理延迟仅0.12 ms/步（GPU）/0.9 ms/步（CPU），对VLA控制周期影响极小。

**⚠️ 局限性**

局限性：①校准假设样本可交换，随着分布偏移而退化；②未系统评估不同潜在层对预测性能的影响；③控制/干预策略仅为启发式示例，缺乏通用评估。

---

## 633. Low-Temperature Sampling via Equilibrium Disagreements

**arXiv ID:** 2611.08416 | [PDF](https://arxiv.org/pdf/2611.08416v1)

**作者:** Andreas Galanis `[一作]` (University of Oxford), Daniel Stefankovic `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于Glauber动态的新框架，用于低温下的采样，解决了竞争模式共存导致的混合瓶颈问题。

**💡 创新点**

通过将相同相的两个独立配置之间的几何不一致性与收敛性联系起来，提供了一种新的组合方法来实现快速收敛。

**🔧 技术方法**

使用了Glauber动态和弱Poincaré不等式的组合方法，结合了模拟退火和单调性。

**📊 数据集**

应用于随机正则图上的三种经典模型：硬核模型、随机集群模型和铁磁Ising模型。

**📈 对比分析**

与现有方法相比，提出的方法在每个固定度数下都能在O(n^(3/2))步内实现混合，且提供了O(n^2)的算法来从全分布中组合相样本，性能显著提升。

**⚠️ 局限性**

限制在于该方法仍然依赖于图的结构特性，且在某些情况下可能需要额外的条件以确保快速收敛。

---

## 634. Scaling Up Analog CIM: A Simulation Framework for Topology-Aware Mapping

**arXiv ID:** 2611.08399 | [PDF](https://arxiv.org/pdf/2611.08399v1)

**作者:** Yuxin Ren `[一作]` (University of Arizona), Huanrui Yang `[通讯]` (University of Arizona)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了Isobar模拟平台，用于在大规模模拟CIM系统中实现拓扑感知的映射，以降低LLM推理的延迟和能耗。

**💡 创新点**

创新点在于将矩阵分解、异构执行与共享资源争用统一建模，提供阶段分析和映射搜索，显著揭示并量化拓扑映射对性能的影响。

**🔧 技术方法**

采用阶段时间模型、动态规划与模拟退火搜索、网络通信仿真、基于矩阵分解的CIM部署，以及能量模型来实现对映射的评估与优化。

**📊 数据集**

使用Gemma‑7B、OPT‑7B、OPT‑13B等大语言模型的预填充和解码工作负载进行验证。

**📈 对比分析**

通过与硅测量结果和BookSim网络仿真对比，Isobar优化后的映射相较最强基线减少23.3%通信时间、18.6%整体延迟，且相对于GPU能耗低90–200倍。

**⚠️ 局限性**

局限性包括仅针对已知硬件层次与拓扑，未考虑随机噪声或非线性误差；对极大规模（万枚芯片）交互模型仍需改进；以及假设静态工作负载，未涵盖动态流式执行。

---

## 635. Module-Aware Optimization for Graph Neural Networks

**arXiv ID:** 2611.08379 | [PDF](https://arxiv.org/pdf/2611.08379v1)

**作者:** Guy Hadad `[一作]` (Ben-Gurion University of the Negev), Bracha Shapira `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 GraphMuon 这一图感知优化器，针对图神经网络（GNN）中的图传播输入调整权重矩阵的更新几何和步长，确保在图传播后的表示空间中产生可控的扰动。

**💡 创新点**

创新点包括：① 用图传播协方差定义的图度量来对 Muon 的矩阵正交化进行改造，实现图度量正交化；② 对更新步长进行基于图传播协方差的归一化，使步长真正反映对图表示的影响；③ 在理论上证明两项改造能抑制图诱发的幅度放大并产生图度量极值分解的极值方向。

**🔧 技术方法**

技术上使用了：Muon's Newton–Schulz 近似正交化、图传播协方差估计、图度量正交化（类似极值分解）、步长归一化（基于 X_g ΔW 的 Frobenius 范数）、以及标准的 AdamW、Shampoo 和自然梯度等对照优化器；并在多种 GNN 架构（GCN、GAT、GINE、GraphGPS）上实现。

**📊 数据集**

实验覆盖 9 个图学习基准：ogbn-arxiv、ogbn-products、ogbn-mag、Questions、Cora、Citeseer、ogbg-molbace、ogbg-moltox21、ZINC，涵盖节点分类、图分类、图回归等任务。

**📈 对比分析**

对比方法包括 AdamW、Muon、Shampoo、图感知自然梯度等；采用固定验证损失目标的时间‑to‑target 评估。结果显示，GraphMuon 在 7 大 OGB 任务中均实现 1.32×–2.42× 的墙钟速度提升，并在 15 个不同数据集–架构组合中在 13 种场景中获得最快的时间‑to‑target，同时保持或略优于其他优化器的最终预测性能。

**⚠️ 局限性**

局限性：当图传播协方差估计不稳定（如节点数小、特征维度大导致低秩或噪声）时，图度量正交化与步长归一化可能无法充分发挥优势；在小型图级任务或批量变异显著的场景下，AdamW 的坐标梯度估计更具鲁棒性，GraphMuon 可能不如 AdamW 效果明显。

---

## 636. Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout

**arXiv ID:** 2611.08376 | [PDF](https://arxiv.org/pdf/2611.08376v1)

**作者:** Tushaar Gangavarapu `[一作]` (University of Texas at Austin), Clemens JS Schaefer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了可在TPU上高效训练的通道级门控Delta记忆模型Mewr，以线性时间捕捉长文本上下文。

**💡 创新点**

通过低秩门控融合、匹配滤波读出门以及边界枢轴块求逆的chunkwise算法，解决了Delta记忆中跨通道干扰、读出偏差和宽矩阵加速瓶颈。

**🔧 技术方法**

结合了通道级指数衰减、Delta消除、低秩投影、匹配滤波读出、递归式状态更新以及基于TPU MXU的边界枢轴块求逆实现。

**📊 数据集**

在FineWeb‑Edu 100B tokens的预训练上进行评估，并在RULER、SQuAD、TriviaQA等多种检索与自然语言推断基准上测试。

**📈 对比分析**

与Gated DeltaNet、Kimi Linear、Splash Attention等基线在相同参数规模下对比，Mewr在验证NLL、WikiText困惑度、零样本推理和多跳检索任务中均提升数十到上百mnats，且在长上下文推理上保持线性复杂度并显著加速训练。

**⚠️ 局限性**

动态读出与写入门压缩之间存在权衡，单查询时仍需较高写门容量；在短上下文下与Splash Attention的吞吐量仍有差距；边界枢轴块实现复杂，需特定TPU硬件支持。

---

## 637. NPCBench: A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma

**arXiv ID:** 2611.08369 | [PDF](https://arxiv.org/pdf/2611.08369v1)

**作者:** Pengkai Wang `[一作]` (Hong Kong Polytechnic University), Hongxia Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并发布了面向鼻咽癌的临床学徒评测基准 NPCBench，涵盖从指南记忆到多模态证据解读、纵向决策修订、完整 CarePath 路径以及开放式咨询评估的全链条能力。

**💡 创新点**

创新点在于：①将临床技能拆解为可量化的六级能力阶梯并与真实患者轨迹对齐；②引入严格的 Patient‑Level Strict 成功指标，揭示局部准确率与整体连贯性之间的“构成差距”；③结合专家制定的咨询 Rubric，评估 AI 产生的临床建议安全性与合理性。

**🔧 技术方法**

使用的大模型技术包括多模态 LLM（如 Gemini、Claude、Qwen、GPT 系列）以及图像处理流水线；评测框架采用标准化任务模板、自动与人工（医生）判定相结合的评估流程。

**📊 数据集**

数据集包含 4,777 条组件与咨询项目，100 名患者 CarePath 样本（共 417 条诊断与护理任务），覆盖 CSCO/NCCN 指南、病理切片、MRI 多序列、EBV DNA 以及完整 EHR 轨迹。

**📈 对比分析**

在 30 个模型中，指南记忆与应用可达 92%+，但仅 23% 的模型在 CarePath 级别实现严格成功；多模态模型在阶段预测的整体准确率约 40% 但完整结构与侵袭评估成功率不足 5%；咨询 Rubric 最高得分为 58.4，主判定器仍检测到 14.6% 的重大错误。

**⚠️ 局限性**

局限性包括：单中心病历来源、部分患者的病理分配为模型选择、类不平衡与样本长度对评分影响、仅有 6 例扩展审计，缺乏多中心或前瞻性验证。

---

## 638. SHORE: Stability-Hybrid On-Policy Reinforcement Learning for Online Bin Packing

**arXiv ID:** 2611.08364 | [PDF](https://arxiv.org/pdf/2611.08364v1)

**作者:** Chang Shi `[一作]` (University of Texas at Austin), Ann Majewicz Fey `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将学习到的稳定性概率融入策略与价值函数的在线3D箱子堆叠强化学习方法。

**💡 创新点**

通过软性稳定性先验与价值加权实现风险与容积平衡，解决传统硬性掩码导致的决策受限问题。

**🔧 技术方法**

使用3D U‑Net预测放置稳定性，Packing Transformer处理箱子与放置候选，结合离散REPPO‑D进行 on‑policy 训练。

**📊 数据集**

在 Isaac Gym 仿真环境下测试四种盒子尺寸组合及扩展的 12 种几何箱子（共 60 个到达），并采用真实机器人演示。

**📈 对比分析**

与 OL Mask、GOPT、GOPT+ 等基线比较，SHORE 在扩展集上实现 0.734 的空间利用率，跌倒率 1.03%，在利用率上优于 GOPT+ 但跌倒率略高。

**⚠️ 局限性**

受限于观测与候选表示的简化、近似稳定性概率的误差、仿真与真实物理参数的不匹配，以及对前期决策的回报依赖导致的学习效率受限。

---

## 639. GPWM: Modeling the World as 3D Particle Motion

**arXiv ID:** 2611.08338 | [PDF](https://arxiv.org/pdf/2611.08338v1)

**作者:** Robert Ren `[一作]` (Carnegie Mellon University), Katerina Fragkiadaki `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于粒子与刚体的生成式三维世界模型，能够预测多种物理介质（刚体、可变形、布料、颗粒）在长期时间上的运动，并通过噪声空间搜索实现接触丰富的规划。

**💡 创新点**

创新点包括：①对象感知的多分辨率Transformer，融合流匹配与扩散强制，实现跨域统一学习；②对未来帧设置不同噪声水平的流匹配训练，支持流式长时程生成；③在噪声空间进行动作规划，显著提升接触序列搜索效率。

**🔧 技术方法**

核心技术：多分辨率DiT（Diffusion Transformer）架构、PointNet++粒子聚合、相对3D几何注意力、流匹配（Flow Matching）与扩散强制（Diffusion Forcing）、隐噪声优化（Latent Noise Optimization）与交叉熵方法（CEM）用于规划。

**📊 数据集**

使用了新构建的GPWM数据集，包含约2550万帧仿真数据，覆盖六大物理域：自由落体、拖拽、布料折叠、推送、抓取与放置、手部操控，并涉及多种几何、材料和机器人装置。

**📈 对比分析**

与GNS、PGND、PointWorld和PhysCtrl等基线对比，统一模型在所有六个域上平均降低预测误差31%，在拖拽和抓放域甚至超过专用域模型；在几何、材料、对象数、以及超出训练窗口的滚动预测等泛化测试中也表现优于现有方法。

**⚠️ 局限性**

局限性包括：①需要完整、已跟踪的3D场景状态，未解决从原始视觉观测恢复粒子和对象身份；②统一模型在某些域仍落后于专用模型，表明跨域干扰；③隐噪声优化需多窗口反向传播，计算成本随规划时间增长，限制了在线实时控制的可行性。

---

## 640. Scaffolding GenAI Use in Ethical Essay Writing: A Three-Year Experience Report in Introductory Computing

**arXiv ID:** 2611.08431 | [PDF](https://arxiv.org/pdf/2611.08431v1)

**作者:** Asma Shakil `[一作]` (University of Auckland), Claus Brabrand `[通讯]` (IT University of Copenhagen)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过三年（2024-2026）在一所大学一年级计算机系统课程中收集学生关于伦理论文写作中使用GenAI的问卷与成绩数据，分析AI采用率、工具选择、使用目的与学生态度，并在2026年引入责任使用流程（先写草稿再使用AI），评估其对学生AI使用行为和写作质量的影响。

**💡 创新点**

将责任使用流程与评估代码（AI语气、批判性参与、真实洞察）系统化；三年纵向比较AI采用与效果；展示政策/评估框架对AI使用率的显著影响；证明AI使用本身不提升写作分数，但通过引导可提升学生批判性参与。

**🔧 技术方法**

使用问卷调查（Microsoft Forms）收集自评数据；采用比例检验、ANCOVA、ANOVA、Kruskal‑Wallis等统计方法进行分析；教师手工对论文进行评估编码，跟踪AI工具多样性。

**📊 数据集**

共1,894名学生的问卷数据（分为2024、2025、2026三期），以及2025、2026期的两篇伦理论文评分数据；记录AI工具使用、使用目的、信任度等信息。

**📈 对比分析**

采用比例检验比较跨年AI采用率；ANCOVA控制先前写作成绩评估AI对最终分数的独立影响；对采用轨迹（永不、始终、采用、放弃）比较分数变化；在2026期对标记代码进行频数比较，观察AI语气、批判性参与、洞察度。结果显示：AI使用率随政策变化显著提升；AI使用对分数无显著独立预测；批判性参与率高于80%，AI语气从负面/中性转正面。

**⚠️ 局限性**

仅为跨截面研究，无法追踪个体变化；样本仅一所大学一年级课程，外推性有限；自报数据可能存在社会期望偏差；评估代码缺乏可靠性验证；2026期无对照组，且不同评分尺度导致跨期比较受限；分数接近上限可能导致截尾效应。

---

## 641. Dynamics Aware Past-Future Information Bottleneck

**arXiv ID:** 2611.08426 | [PDF](https://arxiv.org/pdf/2611.08426v1)

**作者:** Wooyoung Chung `[一作]` (Texas Tech University), Stas Tiomkin `[通讯]` (Texas Tech University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在信息瓶颈框架下，对线性多输入多输出系统的压缩进行结构化约束，使得压缩结果始终可实现为有效状态空间模型；

**💡 创新点**

首次将块Hankel矩阵构造为Lie群，并在其李代数上进行PF-IB优化，从而消除传统PF-IB方法中的实现间隙；

**🔧 技术方法**

采用块Hadamard乘积的Lie群结构、李代数指数/对数映射、反向传播与投影算子，并在多变量弹簧-质量系统上实现实验；

**📊 数据集**

使用两质量弹簧-质量系统的模拟数据（k=7步，n=2）进行压缩、状态估计和LQR控制评估；

**📈 对比分析**

与标准PF-IB及其后置实现方法相比，结构化优化在信息曲线几乎无实现损失，状态估计误差降低≈5倍，LQR成本接近原始最优；

**⚠️ 局限性**

仅适用于可逆Markov参数的系统，且仅限线性动力学，未来需放宽GL(n)约束并推广到非线性模型。

---

## 642. A Near-Optimal Generalized Sorting Algorithm

**arXiv ID:** 2611.08397 | [PDF](https://arxiv.org/pdf/2611.08397v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Junkai Song `[通讯]` (New York University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种随机化算法，利用预测-修复和误差隔离技术，在任意比较图上以期望O(n·log⁴n)次比较恢复隐藏的全序。

**💡 创新点**

创新点在于构造低错误预测函数的迭代框架：先粗略预测、转化为有界度双分图、通过随机抽样与排序判定挑选误差集，并反复去除错误密集区域，从而将误差降到O(n)。

**🔧 技术方法**

主要技术包括随机采样、粗糙预测、双分图分裂、误差隔离协议、排列选择、Chernoff/Hoeffding 估计、期望分析与迭代减半。

**📊 数据集**

论文采用理论分析，无实验数据集。

**📈 对比分析**

相较于过去最优的 Õ(n^{3/2}) 上界，本文将查询复杂度降至 O(n·log⁴n)，在理论上几乎达到普通排序的 Θ(n·log n) 下界。

**⚠️ 局限性**

限制在于仍有多项式对数因子，尚未得到 O(n·log n) 的无对数或确定性算法；以及实现非多项式时间。

---

## 643. Learning a patient world model for health system simulation in intensive care

**arXiv ID:** 2611.08392 | [PDF](https://arxiv.org/pdf/2611.08392v1)

**作者:** Yingtai Li `[一作]` (University of Science and Technology of China), Shaohua Kevin Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究构建了一个共享的患者世界模型（pWM），并将其与不同情景的ICU仿真耦合，用来评估入院、护理调度和出院等运营问题。

**💡 创新点**

创新点在于学习时间条件化的患者动态模型，避免为每个政策问题重新定义状态空间，保留历史依赖，能够在不同运营场景下复用同一模型。

**🔧 技术方法**

采用深度Transformer网络对MIMIC-IV中的稀疏时间序列进行编码，训练可预测未来状态和事件强度的生成模型，并结合校准技术。

**📊 数据集**

使用MIMIC‑IV v3.1（约94,444 ICU住院记录）进行训练、验证和测试。

**📈 对比分析**

与基于Markov和半马尔可夫的任务特定状态模型比较，冻结模型在入院比例、队列负担、床位利用等指标上方向一致，且在模拟重放时能保持调度策略收益；整体占用、支持状态转换等分布也逼近真实数据。

**⚠️ 局限性**

局限包括：模型学习的是历史相关的条件分布而非因果效应；仿真使用的到达、服务时间等参数来自文献范围，未与医院实时数据校准；生成轨迹在某些事件率上存在偏差。

---

## 644. SCALE: Real-Time Latent Communication for Cyber-Physical Agents

**arXiv ID:** 2611.08390 | [PDF](https://arxiv.org/pdf/2611.08390v1)

**作者:** Shanghao Shi `[一作]` (Washington University in St. Louis), Ning Zhang `[通讯]` (Washington University in St. Louis)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种实时潜在通信方法SCALE，专为网络受限的网络物理(CPS)代理设计，利用空间-时间连续性避免重复传输，并通过实时调度器根据语义收益、带宽与任务截止时间动态选择消息及其传输长度；

**💡 创新点**

创新点在于①将空间-时间连续性用于潜在信息重用；②构建语义收益-通信成本曲线用于多消息多任务下的实时调度；③基于边缘截止时间的最优增量分配算法；

**🔧 技术方法**

技术包括：潜在语义变化评估（token匹配+余弦相似度），语义收益与成本剖面（累计语义收益、通信成本），基于边缘优先（EDF）和增量收益比（ρ）的贪心调度算法；

**📊 数据集**

实验数据集：CooperScene（3D检测+轨迹预测）、OPV2V（检测+BEV分割）、LIBERO（语言条件下的机器人操作），以及Raspberry Pi蜘蛛机器人实物案例；

**📈 对比分析**

与Full‑send、Uniform压缩、CoSDH、V2X‑ViT、CooperTrim、Where2comm等基线对比；在三大基准和机器人案例中，SCALE在几乎所有带宽/时变条件下都实现了最高的任务性能（如检测AP、预测成功率、分割mIoU）和更高的按时交付率（高于70%甚至100%），显著优于其他方法；

**⚠️ 局限性**

局限性：①对潜在信息空间-时间冗余的依赖，突发环境变化时重用收益下降；②调度器仅基于当前链路速率预测截止时间，无法及时应对链路速率的快速突降，可能导致信息延迟。

---

## 645. SensorGuard-RV: Sensor-to-Register Data Protection for Resource-Constrained RISC-V Devices

**arXiv ID:** 2611.08380 | [PDF](https://arxiv.org/pdf/2611.08380v1)

**作者:** Hamid Noori `[一作]` (Durham University), Carlton Shepherd `[通讯]` (Durham University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了SensorGuard-RV，一种在RISC-V设备上通过硬件加密和认证将传感器样本直接送入处理器寄存器的安全通道。

**💡 创新点**

采用写一次式所有权切换、固定功能采集控制器和专用指令实现硬件级别的加密、认证与防重放，避免中间软件信任。

**🔧 技术方法**

硬件层面使用Ascon-128a AEAD加密、RISC-V自定义指令、FPGA实现的所有权开关和采集封装；软件方面使用NEORV32裸机固件。

**📊 数据集**

在NEORV32上使用ADXL362加速度计的SPI接口作为传感器样本来源，无专门的数据集。

**📈 对比分析**

与传统软件驱动SPI基准对比；在100 MHz系统时钟下，受保护样本完成周期为5191个（51.9 μs），吞吐率19015个/秒，比基线高5.3%；额外硬件使用3155 LUT和2215寄存器，占设备5%并将功耗从118 mW升至368 mW。

**⚠️ 局限性**

缺乏DMA支持、只能一次性配置传感器、键固定在FPGA配置、未覆盖多传感器、未进行侧信道或物理攻击评估，且无法在运行时重新配置传感器。

---

## 646. Lord of the Precision, the Two Floats: Double-Word Arithmetic for Complex Numbers

**arXiv ID:** 2611.08371 | [PDF](https://arxiv.org/pdf/2611.08371v1)

**作者:** František Stloukal `[一作]` (CERN), Fabienne Jézéquel `[通讯]` (Sorbonne Université)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了复数双词（double‑word）算术的误差分析与实现，给出乘除运算的精度保证并实现了开放源码库CompleXDW；

**💡 创新点**

首次对复数双词乘除进行误差界定，证明在不进行归一化情况下仍能保持三阶误差，并引入低成本Madd加法算法；

**🔧 技术方法**

利用IEEE‑754错误自由变换（EFT）、FMA、FastTwoSum、TwoProdFMA、CUDA/HIP与SIMD技术进行实现与性能优化；

**📊 数据集**

使用随机生成的双词数据（覆盖不同条件数）以及MadGraph7高能物理事件生成器的真实工作负载进行实验；

**📈 对比分析**

通过理论误差界限、随机与最坏案例误差对比以及CPU/GPU（NVIDIA L40S、AMD Radeon Pro W7900）性能测评，验证FP32双词在GPU上可比FP64提升2.1–2.9倍，误差在1.3–1.5位有效数字以内；

**⚠️ 局限性**

局限在于仅处理双词算术，除法误差界限仍可改进；在GPU上大规模算子可能因寄存器压力导致性能下降；

---

## 647. StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference

**arXiv ID:** 2611.08363 | [PDF](https://arxiv.org/pdf/2611.08363v1)

**作者:** Jake Hyun `[一作]` (Cornell University), Mohamed S. Abdelfattah `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个利用主机RAM与SSD提供额外精度的框架，恢复低位量化LLM的准确度，同时保持低内存占用与高速推理。

**💡 创新点**

创新点在于将精度表示、选择与交付三者协同设计为“选择性精度流式传输”，能减少数据传输量、与计算重叠并在多步解码中复用已获取的高精度信息，从而突破量化带来的准确度上限与主机端延迟瓶颈。

**🔧 技术方法**

使用了2–4位权重、2–3位KV缓存量化，主机RAM/SSD高精度缓存，GPU统一内存，Selective Precision Streaming技术，以及对比基线如vLLM 4‑bit AWQ + 8‑bit KV。

**📊 数据集**

评测数据集包括 WikiText‑2（perplexity 评估）与 AIME 2024/2025（Qwen3-8B 的准确率评估）。

**📈 对比分析**

通过在 RTX 5090 与统一内存平台上与纯量化基线和 vLLM 进行对比，实验结果显示：2‑bit Llama‑3.1‑8B‑Instruct 的 perplexity 从 9.46 降至 8.40，吞吐率保持 95.4%；2‑bit 权重+KV 在 Qwen3‑8B 上将 AIME 准确率从 11.0% 提升至 47.1%；3‑bit 权重+2‑bit KV 在 128K 上下文时，速度与准确度与 vLLM 的 4‑bit/8‑bit 方案相当，却仅使用约一半 GPU 内存。

**⚠️ 局限性**

局限性包括对主机 RAM/SSD 带宽与延迟的依赖，流式传输实现仍需额外硬件与复杂调度；实验仅在 RTX 5090 与统一内存平台验证，跨平台可移植性未完全证明；对极大上下文或更深模型的可扩展性仍待进一步研究。

---

## 648. MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue

**arXiv ID:** 2611.08433 | [PDF](https://arxiv.org/pdf/2611.08433v1)

**作者:** Donghua Zhang `[一作]` (University of Michigan), Siyu Qian `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种图形记忆系统，用于LLM医学助手跨访谈追踪患者临床状态，解决状态有效性、主体归属和安全优先性三大挑战。

**💡 创新点**

采用多层记忆结构（原始对话→临床事实图→患者档案），实现事实按时间链式存储、主体显式标注，并将安全事实与每个查询一同提供；检索时按患者链排序并利用日期通道。

**🔧 技术方法**

使用LLM进行事实抽取、图结构构建、BM25与稠密检索、链式检索策略以及LLM生成答案。

**📊 数据集**

在中文纵向医疗对话基准（包含967个问题的LongMemEval等数据集）上进行实验。

**📈 对比分析**

与五种基线（Embedding RAG、Mem0、MemOS、LightMem、REMem）以及GPT和Qwen两种后端进行对比，取得最高平均准确率，较最佳基线提升约10.5/10.4个百分点，尤其在时间定位和推理生成任务上表现突出；同时构建与回答成本最低。

**⚠️ 局限性**

受限于LLM抽取的质量、仅适用于中文、因果链仅限同一访谈、链排序依据插入顺序、以及安全事实永不删除等因素，可能导致信息不完整或误归属。

---

## 649. Data-driven discovery of intrinsic Lie point symmetries of differential equation systems

**arXiv ID:** 2611.08421 | [PDF](https://arxiv.org/pdf/2611.08421v1)

**作者:** Ilker Gurcan `[一作]` (Middle East Technical University), Ramazan Gokberk Cinbis `[通讯]` (Middle East Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从差分方程的取样jet空间数据中直接恢复其隐含的李点对称群（intrinsic symmetry algebra），无需符号求解或闭式表达式。

**💡 创新点**

创新点在于：①将对称判定转化为算子理论问题，利用固定的分块样条字典（类似KAN）构造线性可实现的生成元空间；②引入秩稳定预检（Rank‑Stabilization Preflight，RSP）判断对称代数在字典细化下是否稳定；③采用多阶段约束剔除平凡、解相关及已发现方向，逐步聚焦真正的intrinsic生成元。

**🔧 技术方法**

使用的技术包括：jet空间几何、算子理论、固定分块样条字典、块归一化的梯度下降优化、非线性正则化、RSP预检、欧氏内积投影、离散取样对称性判定、局部/全局线性独立性检验等。

**📊 数据集**

数据集为在一个已知11维对称代数的基准DE系统上通过随机shell采样生成的取样点；未使用公开数据集。

**📈 对比分析**

与传统符号方法以及最近的LieStoNet、LieDynNet等数据驱动对称发现方法对比；在基准实验中成功恢复完整的11维生成元，Lie括号闭合至机器精度；对已知生成元的全局线性逼近误差低，能够在训练外域保持一致性，计算速度在可接受范围内。

**⚠️ 局限性**

主要局限包括：只能处理有限维intrinsic symmetry algebra；对无限维或含非线性解相关分支的系统无法完整识别；RSP预检失败时无法保证完整性；对样本分布、字典阶数、离散误差敏感；仅适用于李点对称，无法处理切换对称等非点对称。

---

## 650. On the limitations of molecular representations

**arXiv ID:** 2611.08420 | [PDF](https://arxiv.org/pdf/2611.08420v1)

**作者:** Acer Blake `[一作]` (University of Oxford), Charlotte M. Deane `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了传统指纹与学习嵌入在多维相似度、虚拟筛选、聚类、3D形状及ADMET相似度等任务上的表现。

**💡 创新点**

创新点在于：①在10.8B Enamine REAL数据库上量化指纹退化率并揭示退化导致的化学活性差异；②使用MCES最大公共边子图距离对全局图相似度进行客观评估；③揭示聚类泄漏、3D与ADMET相似度不足等关键瓶颈。

**🔧 技术方法**

采用的技术包括：RDKit指纹生成、MCES距离计算、Butina聚类、形状/颜色相似度评估、ROGI-XD景观粗糙度指标、以及多种学习模型（MolBERT、ChemBERTa、MolCLR、MolFormer、UniMol、COATI 等）与经典指纹/描述符对比。

**📊 数据集**

使用的数据集包括：Enamine REAL 10.8B、ChEMBL、MCES、ZINC、Biogen ADMET、以及 KEGG、ChEBI、HMDB、UNPD 等公开化学数据库。

**📈 对比分析**

通过 Spearman 相关性、Tanimoto 与余弦相似度与 MCES 的对比，发现无论是传统指纹还是学习嵌入在捕捉图形、3D 形状和 ADMET 相似度方面均表现不足，传统指纹在某些评估上往往优于学习嵌入。

**⚠️ 局限性**

局限性包括：指纹退化率高导致虚拟筛选结果被误判；学习嵌入未能有效捕捉全局拓扑与 3D 形状；聚类泄漏导致数据集划分不可靠；缺乏统一、全面的评估标准。

---

## 651. Spectral and Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement Learning

**arXiv ID:** 2611.08414 | [PDF](https://arxiv.org/pdf/2611.08414v1)

**作者:** Chenruo Liu `[一作]` (New York University), Qi Lei `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在监督微调（SFT）和强化学习（RL）阶段使用LoRA时的谱值与奇异子空间动力学，发现组均值基线减法是两者差异的关键因素，并据此设计了针对不同阶段的优化器配置。

**💡 创新点**

创新点在于揭示组均值基线减法如何通过平衡一致拟合与梯度噪声，导致LoRA增量的谱集中与子空间旋转差异，并基于此提出了阶段感知的LoRA更新策略。

**🔧 技术方法**

技术手段包括谱与子空间诊断（有效秩、重叠度量）、二次多指数模型的理论分析、Muonspectral flattening以及Adam/SGD混合更新。

**📊 数据集**

实验使用GSM8K与MATH两大数学推理数据集，针对Llama-3.2-1B/3B与Qwen2.5-3B/Math-7B模型进行评估。

**📈 对比分析**

通过对比不同LoRA更新配置（标准Adam、Adam+Muon、SGD+Muon、RL专用Muon⊥等），在不同学习率下测量测试准确率，发现SFT阶段的动态配置在多数设置中优于标准Adam，而RL阶段的动态配置则在大多数设置中取得最高准确率。

**⚠️ 局限性**

局限性包括仅关注LoRA方法，实验模型与任务范围有限，理论假设（如任务扰动小、无穷维特征空间）可能不完全适用于所有实际场景，需进一步验证。

---

## 652. The Gold Rush in AI4Math, II: What are signatures of AI-assisted Proofs?

**arXiv ID:** 2611.08412 | [PDF](https://arxiv.org/pdf/2611.08412v1)

**作者:** Yuanchuan Guo `[一作]` (Harvard University), Bingcheng Sui `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了五个不同 AI 参与程度的论文样本，开发 ProofTrace 框架将每篇论文转化为证明支持图（GIPS），并从中提取可解释的结构特征，用于统计分析和 AI 参与检测。

**💡 创新点**

创新点在于：①将数学证明组织建模为图并提取结构特征，②发现 AI 参与在证明组织上的“签名”并验证其可检测性；③展示 ProofTrace 在保持重写不变的情况下对 AI 参与检测的鲁棒性。

**🔧 技术方法**

使用大型语言模型（GPT‑5.6 Luna、GPT‑6 Astra 等）进行图抽取；统计学检验（配对 t‑检验、Benjamini‑Hochberg 多重检验）；机器学习分类（ProofTrace、ProofTrace++）以及与多种预训练与微调的 AI 文本检测器进行对比。

**📊 数据集**

数据集共 8,281 篇论文：arXiv 的 Human、Hybrid 论文；OpenAI 发布的 AI 生成论文；以及用 LLM 生成的 AI‑rebuilt、AI‑reworded 论文，形成五个队列。

**📈 对比分析**

通过配对 t‑检验评估结构特征差异，发现 265/342 结构特征显著差异；在检测实验中，ProofTrace 在原始与重写测试集上 AUROC/AP 稳定，ProofTrace++ 与微调的预训练模型性能相当。

**⚠️ 局限性**

局限性包括：未考虑对抗性攻击、仅关注证明结构而非技术细节，未来需改进特征并扩展到其他依赖数学推理的学科。

---

## 653. Conformal Prediction for Time-Dependent PDEs

**arXiv ID:** 2611.08409 | [PDF](https://arxiv.org/pdf/2611.08409v1)

**作者:** Joshua Stiller `[一作]` (LMU Munich), Eyke Hüllermeier `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究时间相关 PDE 预测中 conformal prediction 的覆盖保证，提出理论极限并给出可行的加权 CP 方法。

**💡 创新点**

首次证明在函数空间中不同时间点的解分布互相奇异导致覆盖失效，并量化网格细化导致 TV 距离指数爆炸的现象。

**🔧 技术方法**

采用加权 conformal prediction（WCP），利用高斯初始条件与线性 PDE 的闭式似然比、总变差距离和数值误差上界实现覆盖提升。

**📊 数据集**

通过合成 PDE、激光热成像、FRAP 微测、湖泊热量传输等四大真实数据集进行验证。

**📈 对比分析**

与传统 CP、WCP learned、LSCI、DtACI 等基线比较，WCP Gauss 在所有任务上始终达到目标 90% 覆盖率，其他方法在网格或时间偏移时易出现低覆盖。

**⚠️ 局限性**

仅适用于线性 PDE，非线性系统无法直接应用；函数空间覆盖不可实现；需要精细网格时覆盖衰退，且需估计均值协方差，样本量不足易失效。

---

## 654. StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning

**arXiv ID:** 2611.08403 | [PDF](https://arxiv.org/pdf/2611.08403v1)

**作者:** Yang Ouyang `[一作]` (North Carolina State University), Jung-Eun Kim `[通讯]` (North Carolina State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为StepBack的步骤级错误定位与后向重采样方法，该方法在保持初始推理前缀的基础上，仅重新采样推测错误步骤及其后续，以提升链式推理的准确率和效率。

**💡 创新点**

首次将轻量级步骤级信号（如NLL增量或PRM分数下降）用于定位推理轨迹中的首个错误位置，并在此位置处进行后向重采样，从而显著减少完整轨迹的重复生成，达到成本与性能的双重优化。

**🔧 技术方法**

结合自一致性（Self‑Consistency）框架，使用负对数似然增量和过程奖励模型（PRM）作为定位信号，并在多种大型语言模型（Llama‑3B、Qwen‑3B、Qwen‑7B）上实现；采用vLLM进行高效推理；与Adaptive Consistency、Early‑Stopping SC、Difficulty‑Adaptive SC等自适应推理方法进行比较。

**📊 数据集**

在七个数学推理与多跳问答基准上进行评估，包括GSM8K、MATH500、AMC2023、AIME24、OlympiadBench、HotpotQA和2WikiMultiHopQA。

**📈 对比分析**

与同等候选答案预算（B=32）的全轨迹自一致性相比，StepBack在保持或略优的准确率下，生成令牌数平均减少约21%至41%；在与Adaptive Consistency、ESC、DSC等方法结合时，进一步降低了22%–38%的令牌消耗，同时准确率提升1–2个百分点，表明在效率-性能平衡上具有显著优势。

**⚠️ 局限性**

局限性包括：仅测试了两类定位信号，其他更轻量化信号尚未探索；在更难的竞赛级数据集（如AIME24、OlympiadBench）上效果不够稳定；未验证方法在代码生成、长文本摘要等非推理任务中的泛化能力。

---

## 655. Synthesis of Ovarian Cancer Multi-Omics Datasets: A Biology-Informed Evaluation Framework

**arXiv ID:** 2611.08394 | [PDF](https://arxiv.org/pdf/2611.08394v1)

**作者:** Déborah Boyenval `[一作]` (University of Helsinki), Sampsa Hautaniemi `[通讯]` (University of Helsinki)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套评价合成配对RNA和CNV多组学数据的方法，并在DECIDER卵巢癌队列上做案例验证；

**💡 创新点**

创新点在于引入患者分离的多模态评估框架，兼顾单模态分布保真、跨模态相关性和训练集接近度，并给出基准比较；

**🔧 技术方法**

使用变分自编码器（VAE）三种变体（ISOMERIC、TMO-Net、CustOmics）实现无监督的RNA-CNV配对生成；

**📊 数据集**

数据集为DECIDER前后治疗的高分辨率卵巢癌样本，共632个配对样本，选取3000个高变基因；

**📈 对比分析**

评估指标包括：单模态G-score相关性、Wasserstein/JS/JS距离、RBF MMD、共表达网络Edge F1、RNA‑CNV斜率相关、训练-留存样本最近邻距离及AUC；在前期数据集上，TMO-Net在RNA、CNV保真和共表达上表现最好，但在训练接近度上最突出；ISOMERIC在RNA‑CNV关联上优；CustOmics则在共表达密度控制上有待提升；

**⚠️ 局限性**

局限包括：仅评估VAE生成器，未覆盖GAN/流/扩散模型；未加入临床或组织学条件；评估未能完全捕捉隐私风险；模型参数规模差异未完全统一，可能影响对比；多模态协同学习机制仍需进一步优化。

---

## 656. When Less is More: Emerging Network-Wide Use Cases for Hollow-Core Fibers in Optical Networks [Invited]

**arXiv ID:** 2611.08389 | [PDF](https://arxiv.org/pdf/2611.08389v1)

**作者:** Mëmëdhe Ibrahimi `[一作]` (Politecnico di Milano), Massimo Tornatore `[通讯]` (Politecnico di Milano)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文综述了空心核心光纤（HCF）在网络级别的多种使用场景，并整合了以往研究成果，重点探讨了在延迟、能效、容量扩展与量子密钥共存等四大业务场景下的选择性部署策略。

**💡 创新点**

创新点包括：① 引入分层优先级目标和边缘数据中心互斥变体；② 对单纤双向HCF（BiDi-HCF）的网络级评估；③ 统一阐述选择性部署在不同业务场景下的收益上限与饱和点；④ 对四大业务场景的交叉洞见进行系统性提炼。

**🔧 技术方法**

主要技术手段包括：混合整数线性规划（MILP）、遗传算法（GA）、ROMA能效放大器布置启发式、根-子网-映射（RMSA）、k-最短路径与第一适配弹性网格调度，结合 HCF 的IMI、Raman、Rayleigh 等物理层模型进行性能仿真。

**📊 数据集**

所使用的数据集为典型的网络拓扑与流量模型：Metro‑52、EU49、EU19、IT21、东京地铁等；流量采用均匀源-目的地分布与预设需求混合；并无公开真实业务数据，主要依赖合成场景。

**📈 对比分析**

通过与全SMF网络基准进行对比，评估指标包括：边缘DC数量、每Tbps功耗、服务流量、量子模块数量等。结果显示：即使仅 20‑40% 的 HCF 预算，边缘DC 数量可减至 50% 以内；能效提升 20‑30%；服务流量提升 70%；量子模块成本可降低 40%。

**⚠️ 局限性**

局限性主要体现在：① 缺乏可靠的 HCF 成本模型，导致经济评估不确定；② 物理层模型忽略了气体吸收峰和高功率非线性效应；③ 流量与需求模型过于理想化，未覆盖真实网络异质性；④ 单一业务场景分析，未形成多目标多用例协同优化；⑤ 计算规模有限，难以扩展至千级网络；⑥ 仍需实地验证与标准化、制造可扩展性的进一步突破。

---

## 657. SLA: Scaling Spatial-Language-Action Model via Continual Pre-training

**arXiv ID:** 2611.08373 | [PDF](https://arxiv.org/pdf/2611.08373v1)

**作者:** Fuhao Li `[一作]` (Hong Kong University of Science and Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个在持续预训练与后期微调阶段通过隐式3D对齐提升视觉嵌入的机器人基础模型，实现跨任务、跨机器人平台的空间感知。

**💡 创新点**

创新点在于使用冻结的3D感知教师对VLA中间视觉表示进行对齐，并结合自适应层选择与状态抖动，在不改动原有结构的前提下实现可迁移的空间理解。

**🔧 技术方法**

采用的技术包括OmniVGGT 3D教师、可学习的多层门控融合、余弦对齐损失、状态抖动、旋转向量动作表示以及持续预训练+后期微调策略。

**📊 数据集**

整合了InternData-A1、AgiBotWorld-Beta、RoboCasa365-Pretrain、RoboCOIN、DROID、MolmoAct等约170万条演示的统一数据集。

**📈 对比分析**

在LIBERO-Plus、RoboTwin 2.0（Clean2Random）、RoboCasa-Panda、RoboDojo-Real等基准上与π_0.5、Spatial Forcing等方法对比，成功率提升约10%–30%，尤其在跨环境、跨机器人、低样本适配与动态目标场景中表现最优。

**⚠️ 局限性**

局限性包括仍需大量演示数据、依赖冻结的3D教师、无法实时获取深度信息，在高度动态或极端几何变化下性能仍有限，对不同机械臂和传感器的适配需额外调优。

---

## 658. VisionSIM: Beyond Idealized Sensors - A Multi-Modal Simulator for Robot Perception

**arXiv ID:** 2611.08368 | [PDF](https://arxiv.org/pdf/2611.08368v1)

**作者:** Sacha Jungerman `[一作]` (University of Wisconsin -- Madison), Mohit Gupta `[通讯]` (University of Wisconsin -- Madison)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一、基于物理的机器人感知仿真平台，能够以高保真度模拟从传统相机到单光子LiDAR、时间飞行、事件相机以及长波红外热像仪等多模态传感器的原始测量过程；

**💡 创新点**

创新点在于将传感器测量的物理模型与失效模式（如单光子LiDAR的堆叠、iToF的相位包装、事件相机的可恢复性和热像仪的历史依赖性）直接嵌入仿真中，同时实现了跨模态、可调参的统一框架；

**🔧 技术方法**

使用了光线追踪渲染、GPU加速热传导求解、Poisson过程和脉冲响应卷积等物理模型，并结合分布式渲染、帧率插值与深度视频插值技术；

**📊 数据集**

使用了自制的室内100个场景的高清渲染数据集，并同步生成了1000×1000分辨率、100fps的多模态原始传感器数据；

**📈 对比分析**

通过与真实传感器采集的原始数据（单光子LiDAR、iToF、热像仪、事件相机）做对比，仿真在热像、iToF相位误差、事件计数与单光子检测率上分别实现了7.6%以内的MAE、0.13弧度相位误差、1.5%/5.8%事件计数误差、0.221与0.229的检测统计相匹配；

**⚠️ 局限性**

局限包括：仅覆盖室内场景、缺乏对复杂天气或毫米波雷达等波动传播效应的建模、只做了开放式评估未闭环控制、数字孪生验证样本有限、并且依赖中间抽象导致无法模拟体积光或高阶光传输效应。

---

## 659. CascadeFilter: Cost-Efficient Semantic Filtering with Joint Threshold Selection

**arXiv ID:** 2611.08367 | [PDF](https://arxiv.org/pdf/2611.08367v1)

**作者:** Yanji He `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三阶段语义过滤器CascadeFilter，通过联合阈值搜索和有限样本校准来降低基于LLM的过滤成本。

**💡 创新点**

创新点在于对固定三阶段结构的阈值进行联合搜索、使用固定序列多重检验校准，以及结合检索与邻域支持提升证据质量。

**🔧 技术方法**

使用了小型语言模型、BM25+稠密检索、RRF重排序、邻域支持置信度、WS&R下注框架进行校准。

**📊 数据集**

实验使用了Yelp、Codebase、IMDB-Review、AirDialogue、Twitter Hate、FEVER共六个数据集，覆盖13种谓词设置。

**📈 对比分析**

与Full SLM、LOTUS、BARGAIN、ScaleDoc、UniVote、SimVote、Task Cascades等基线对比，CascadeFilter在12/13设置上取得最高的准确率和宏F1，成本与Oracle调用相比显著降低。

**⚠️ 局限性**

局限性包括：仅在oracle相对误差上提供保证，未对真实标签误差进行理论控制；校准样本大小有限且置信区间对不同设置差异大；缺乏对运行时成本（吞吐量、延迟）以及分布漂移的实验验证。

---

## 660. SpecCTMC: Exact Speculative Simulation of Diffusion Language Model Continuous-Time Markov Chains

**arXiv ID:** 2611.08362 | [PDF](https://arxiv.org/pdf/2611.08362v1)

**作者:** Siva Rajesh Kasa `[一作]` (Amazon), Hongdong Li `[通讯]` (Australian National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SpecCTMC的训练无关的推测采样器，旨在减少连续时间马尔可夫链中逐个模型评估的瓶颈，从而实现更高效的文本生成。

**💡 创新点**

创新点在于SpecCTMC能够在不需要单独草稿模型的情况下，通过并行评估中间状态来减少序列模型评估的次数，同时保持生成的联合分布不变。

**🔧 技术方法**

使用了推测采样和精确事件驱动模拟的技术，结合了并行模型评估和接受-修正程序。

**📊 数据集**

在LLaDA-8B-Instruct和Dream-v0-Instruct-7B数据集上进行了实验，评估了其在多个基准测试中的表现。

**📈 对比分析**

与DirectCTMC（精确事件驱动采样器）和两种近似采样器（Tau-leaping和trapezoidal sampling）进行比较，SpecCTMC在序列模型评估上减少了1.73到4.38倍，同时在墙钟时间上也实现了加速。

**⚠️ 局限性**

限制在于SpecCTMC的性能依赖于序列长度、内存和目标模型的批处理成本曲线，且在某些情况下，较大的预测变化可能会降低加速效果。

---

## 661. OraRefer: Segmentation-Guided Multi-Photograph Fusion for Oral Cancer Screening with Guaranteed Patient-Level Sensitivity

**arXiv ID:** 2611.08360 | [PDF](https://arxiv.org/pdf/2611.08360v1)

**作者:** Rupam Mukherjee `[一作]` (Indian Institute of Technology Kharagpur), Subhamoy Mandal `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种从患者的智能手机口腔照片到具有保障灵敏度的筛查建议的端到端模型。

**💡 创新点**

创新点在于：①将口腔与病变分割与分类结合为同一网络；②利用该分割自定位，避免测试时标注；③在患者层面融合多张照片与风险因子，并通过分布无关的保证灵敏度阈值和 Mondrian 合法化三分支，实现可调的灵敏度保障；④通过交叉拟合保证训练和测试不泄漏。

**🔧 技术方法**

技术包括：分割引导的 Transformer 影像模型、区域亲和注意力、监督对比学习、蒙特卡罗 dropout 估计、基于逻辑回归的患者级融合、分布无关灵敏度阈值和 Mondrian 合法化阈值。

**📊 数据集**

使用的是 Piyarathne 等公开数据集：3000 张来自 714 名患者的智能手机口腔照片，含病变多边形、风险因子及诊断标签。

**📈 对比分析**

与多种基线（仅影像、风险因子、无训练的聚合、ABMIL、CAR‑MIL、梯度提升等）对比，本文方法在患者层面 AUROC 最高（0.919），在 90% 目标灵敏度下实现 0.950 的灵敏度与 0.610 的特异性，显著优于对照方法。

**⚠️ 局限性**

局限性包括：评估为回溯性实验，缺乏前瞻性验证；模型对不同手机摄像头的泛化性未进一步验证；对极少量病例（如罕见癌症）仍有漏检风险；需要本地样本进行阈值校准，降低了即插即用性。

---

## 662. 4D Point Splatting for mmWave Radar Micro-Doppler Differentiable Rendering

**arXiv ID:** 2611.08342 | [PDF](https://arxiv.org/pdf/2611.08342v1)

**作者:** Adnan Armouti `[一作]` (Cornell Tech), Rajalakshmi Nandakumar `[通讯]` (Cornell Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理的可微分雷达渲染器4D Point Splatting（4DPS），能够把摄像机初始化的带材质、法向量和速度的有向点云拟合到真实毫米波雷达捕获数据中，并生成原始复数返回，支持任意DSP表示；

**💡 创新点**

创新点在于：①将速度作为可微分参数引入雷达渲染；②利用闭式Doppler点分布函数实现高效的范围‑多普勒分裂，降低了对慢时轴的计算成本；③通过双视角（第一拍慢时视角与多普勒平面）损失实现静态场景与运动体的联合拟合；④提供完整的训练链、CUDA核以及可编辑场景，实现大规模数据集的逆向渲染；

**🔧 技术方法**

使用了可微分雷达渲染框架、闭式Doppler点分裂、ITUR P.2040 BSDF、基于摄像机的点云初始化、梯度下降优化、CUDA加速自定义核以及多频段DSP后处理；

**📊 数据集**

在HuPR（双单芯片毫米波雷达）和RT-Pose（级联雷达+摄像机）这两个公开人类感知数据集上进行实验；

**📈 对比分析**

对比方法包括：与传统前向模拟器RF-Genesis、未优化的4DPS渲染、以及在真实数据上训练的基线模型；在范围‑方位、范围‑多普勒以及微多普勒上的中位相关系数分别提升至0.91、0.87、0.93（HuPR）和0.82、0.85、0.90（RT-Pose）；在姿态估计下，使用4DPS渲染训练的模型在真实数据上仅损失约20%的准确率（HuPR 0.8 AP vs 1.0 AP，RT-Pose 4.5 cm MPJPE vs 3.4 cm），显著优于RF-Genesis；

**⚠️ 局限性**

局限性包括：速度仅在视频运动的残差范围内优化；背景保持静态，只考虑单人场景；渲染仅支持单次反射路径；对相机与雷达的外参估计敏感；缺乏对编辑场景与真实数据匹配的评估。

---

## 663. Discovering Learning Rules that Shape Others

**arXiv ID:** 2611.08335 | [PDF](https://arxiv.org/pdf/2611.08335v1)

**作者:** Hong Jun Lee `[一作]` (King's College London), Yali Du `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并训练了一种名为 DUOS 的元学习更新规则，能够仅凭自身经验和收益矩阵在一般-和博弈中对对手进行塑形并提升自身回报。

**💡 创新点**

创新点在于通过元强化学习自动发现只依赖自身轨迹的更新规则，无需访问对手的策略、参数或奖励；并结合 PSRO 训练以提升对抗性和鲁棒性。

**🔧 技术方法**

使用技术包括：元网络（GRU+MLP）生成 log‑odds 变更、似然比估计梯度（Evolution Strategies 风格）、Gauß‑Newton 内层参数更新、Gaussian 噪声扰动、PSRO 双奥洛罗以及基于经验回报的 meta‑objective。

**📊 数据集**

实验数据集主要为四个 2×2 迭代矩阵游戏（IPD、IMP、Chicken、Stag Hunt）和 Coin Game，每个游戏使用其标准收益矩阵；训练对手混合包含自玩、无知学习者、始终合作/始终背叛和利用者。

**📈 对比分析**

通过在五个游戏中进行头对头锦标赛，与 LOLA‑OM、POLA‑OM*、SHAPER、M‑FOS、M‑FOS‑H、Reciprocator、COALA‑PG* 等基线对比；DUOS 在所有游戏中标准化平均得分均为 1.0（排名第一），自玩得分最高，且在 Chicken 与 Stag Hunt 的零样本迁移亦能获得竞争性得分。

**⚠️ 局限性**

局限性包括：规则需针对每个游戏单独训练；在 Chicken 训练的规则比 IPD 训练的零样本版本表现更好；在 Coin Game 规则只能通过缩放 REINFORCE 步长，难以实现真正的对手塑形；泛化能力受限，训练成本较高。

---

## 664. User Embeddings are Superpositions of Interpretable Behavioral Modes

**arXiv ID:** 2611.08487 | [PDF](https://arxiv.org/pdf/2611.08487v1)

**作者:** Jonathan Kozaczuk `[一作]` (Amazon), Brandon R. Feng `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种后置方法，利用稀疏自编码器将用户嵌入分解为稀疏的可解释行为模式，并通过 LLM 自动命名，从而实现零样本自然语言受众检索。

**💡 创新点**

创新点在于将机制可解释性中的稀疏叠加假设迁移到用户嵌入，结合稀疏自编码器实现无监督的可解释分解，并在嵌入未训练文本的情况下支持自然语言查询。

**🔧 技术方法**

使用技术包括稀疏自编码器（ReLU+L1）、过完备字典学习、Transformer 编码器、TF‑IDF 预处理、以及 LLM 进行特征自动解释和查询匹配。

**📊 数据集**

实验数据集涵盖合成用户嵌入（100k 用户，75 种模式）、公开电商数据（180 万交易，5027 用户）和 MovieLens 1M（6040 用户，100 万评分）。

**📈 对比分析**

与 PCA、ICA、FastICA、K‑Means、NMF 以及基于文本嵌入的检索基线比较，稀疏自编码器在 Lift、AUROC、覆盖率和方差解释度（FVU）上均优于传统方法，并在合成数据中逼近理论上限。

**⚠️ 局限性**

局限性包括仅考虑线性或 Transformer 嵌入，无法处理时间演化或逻辑约束；只验证了 ReLU SAEs，未探索其他稀疏模型；以及实验集中于正反馈交互，缺乏对负面或多模态交互的评估。

---

## 665. Do You Miss the Nights We Spent Crafting Figures? Rethinking Teaser and Framework Figures in the Age of Generative AI

**arXiv ID:** 2611.08486 | [PDF](https://arxiv.org/pdf/2611.08486v1)

**作者:** Xiaoqi Zhao `[一作]` (Nanyang Technological University), Kailai Zhou `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建可追溯的科研图表语料库并评估人工与AI辅助修订的视觉质量，提出设计准则；

**💡 创新点**

首次系统地将六维度评估指标与源受限修订流程相结合，揭示AI辅助对图表清晰度与美学的一致性与权衡；

**🔧 技术方法**

使用OpenAI GPT‑4多模态模型进行图表评估与自动修订，配合图表解析、版本追踪和Bootstrap置信区间；

**📊 数据集**

采集200篇论文中的371张动机与框架图表（S1–S4四阶段加对照样本），覆盖多研究主题与不同来源；

**📈 对比分析**

采用同一模型多次匿名评分比较原图与修订图，六维度平均提升0.10，清晰度提升0.43，美学下降0.11，差异显著且呈现各维度权衡；

**⚠️ 局限性**

样本非概率抽样、评估者单一、模型版本依赖、未区分布局与内容信息差异、缺乏读者理解验证等限制。

---

## 666. Where and Why is Grounded Reasoning Useful?

**arXiv ID:** 2611.08485 | [PDF](https://arxiv.org/pdf/2611.08485v1)

**作者:** David Yan `[一作]` (Elorian), Yinfei Yang `[通讯]` (Elorian)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并量化视觉 grounding 在 chain‑of‑thought 视觉推理中的作用，提出随机化 grounding 的评估方法，并系统性地 ablation 训练 recipe，最终提出 BrickVision benchmark 以检验对“不可描述”视觉引用的处理。

**💡 创新点**

①提出基于随机化 grounding 的 CoT 级依赖评估；②系统探究不同训练策略对空间推理、感知与计数任务的影响；③证明 grounding 仅在缺乏可描述文本引用时显著提升；④设计 BrickVision benchmark 解决这一缺口。

**🔧 技术方法**

coordinate‑only grounded reasoning、chain‑of‑thought 评估（随机化 grounding）、SFT 与 RL fine‑tuning、reflection 与 appended grounding 模式、synthetic probe tasks、程序化生成与风格迁移。

**📊 数据集**

使用 SpatialMQA、MultihopSpatial、V*、MME‑RW、TreeBench、CountQA、FSC‑147 等公开基准；构造高质量 6,368 条 SFT 样本；创建 1,000 条 BrickVision 任务。

**📈 对比分析**

通过在各基准上与无 grounding 文本推理的准确率对比（Δ accuracy），发现 grounding 仅在计数任务上显著提升（CountQA +3–4%，FSC‑147 +2–3%），其余空间与感知任务无显著差异；在 BrickVision 上前沿模型普遍低于 30%，但 GPT‑6 Astra 及 fine‑tuned grounded Qwen 取得 68.7% 与 82.9% 的高分。

**⚠️ 局限性**

Grounding 仅在缺少可描述文本引用的任务中有效；大多数公开模型并未真正利用其生成的 grounding；现有数据集往往依赖文本标签，难以大规模获取高质量 grounding；训练 recipe 对 grounding 的利用率敏感，且未必提升整体性能。

---

## 667. Unpacking the Compute Efficiency of Strong-to-Weak OPD against GRPO

**arXiv ID:** 2611.08453 | [PDF](https://arxiv.org/pdf/2611.08453v1)

**作者:** Zeyu Leo Liu `[一作]` (Salesforce AI Research), Semih Yavuz `[通讯]` (Salesforce AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统地比较了 On‑policy Distillation（OPD）与 Group Relative Policy Optimization（GRPO）在多任务、多模型尺寸、多教师设置下的训练 FLOPs 与性能，重点探究自我 GRPO 教师与多教师 OPD 在计算效率方面的优势。

**💡 创新点**

创新点在于发现仅当教师为学生自身的 GRPO 检查点时，OPD 能以约 50 倍更少 FLOPs 达到相同或更高的准确率；并提出将多个领域专家通过多教师 OPD 合并为单一学生，显著降低 FLOPs 并优于联合 GRPO。

**🔧 技术方法**

使用了 OPD（闭式 top‑k reverse‑KL 目标）与 GRPO 两种 RL/蒸馏框架，配合多组 rollout、思考模式、教师前向传递等训练技术。

**📊 数据集**

实验数据集包括数学任务的 DAPO‑Math‑17K、DeepMath‑103k，代码任务的 LiveCodeBench 与 CodeForces，多域实验加入 MMLU‑Pro 与 IFBench‑RLVR，以及 OOD 评估的 AIME24/25、AMC23。

**📈 对比分析**

在匹配训练 FLOPs 的条件下，作者对 held‑out 准确率进行曲线比较；外部教师 OPD 通常低于 GRPO；自我 GRPO 教师在单域和多域、多尺寸实验中能在相同 FLOPs 下达到或超过 GRPO；多教师 OPD 在 2‑域和 4‑域混合中分别比联合 GRPO 少 42 与 31 FLOPs。

**⚠️ 局限性**

局限性包括仅评估 vanilla reverse‑KL 与闭式 top‑k 目标，未涵盖混合目标、其他 divergences、轨迹重塑、异步或推测 rollouts 等可能缩小 FLOPs 差距的技术；实验仅在公开模型/数据上进行，缺乏对更大规模或不同任务的验证。

---

## 668. Souffleur: A Glanceable Display for Mobile Voice Conversations

**arXiv ID:** 2611.08443 | [PDF](https://arxiv.org/pdf/2611.08443v1)

**作者:** Anton Wolter `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在智能手机上实现了一种名为 Souffleur 的面向语音对话的“可一眼看见”主题堆叠显示，并通过 48 名参与者的对照实验评估其对跟随、恢复和记忆的影响。

**💡 创新点**

创新点在于将语音会话自动归纳为主题标题并在堆叠中标记来源与决策状态，形成专为手机视线有限场景设计的最小化可视化方案；同时提出并验证了六条针对可一眼看见语音对话显示的设计目标。

**🔧 技术方法**

技术方案包括：OpenAI 实时语音‑语音模型进行对话；Whisper 语音识别；LiteLLM 代理模型完成主题命名、决策摘要与简述；PWA（Progressive Web App）在手机上实现前端显示；以及 PostHog、Langfuse 等日志/跟踪系统。

**📊 数据集**

数据来源为实验中收集的 48 次会话记录（对话文本、音频片段和交互日志），未使用公开数据集；评估过程依赖这些实时生成的语音转文本和主题化输出。

**📈 对比分析**

采用 2×2×2 的被试间设计，将 Souffleur 与传统聊天记录进行对比；主要指标为恢复延迟、跟随易度、记忆回溯率等。结果显示两种界面在所有指标上无显著差异，未检出任何大幅优势或劣势。

**⚠️ 局限性**

局限性包括：样本量有限、仅涉及短时（≈2‑3 分钟）对话、缺乏并行任务或真实世界多样化中断、对话主题数量少、评估多为自评且回忆率低，因而难以证明可一眼看见显示在更长或更复杂情境中的实际价值。

---

## 669. Scientist World: Grow a Research World for AI-Scientists

**arXiv ID:** 2611.08478 | [PDF](https://arxiv.org/pdf/2611.08478v1)

**作者:** Taiming Lu `[一作]`, Zhuang Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将1,000篇AI论文自动转化为可执行的研究环境，使AI代理能够重现、改进或从零开始创新；并收集完整的研究轨迹数据；

**💡 创新点**

构建无人工干预的可扩展、可演进、执行基准化的研究环境管道，结合自动重现、任务设计、打包与验证，形成可持续的科研挑战平台；

**🔧 技术方法**

采用多模型语言助手（GPT、Claude、MiniMax、GLM）进行自动化构建与验证；利用多阶段强化学习（GRPO）和监督微调训练小型科学家；使用大型计算集群执行实验；

**📊 数据集**

基于arXiv AI核心论文库（124,816篇）筛选至1,000篇；收集各环境所需数据集（如ImageNet‑50、CIFAR等）；公开训练和追踪数据在huggingface“zlab-princeton/scientist-world-traces”上；

**📈 对比分析**

通过对10个低算力基准环境的改进与创新两种模式评估模型；在改进模式下Fable 5表现最佳，在创新模式下Fable 5.1/5领先；相较原始方法，84%改进、74%创新得分提升，且成绩与研究质量评价高度相关；

**⚠️ 局限性**

仅涵盖1,000篇论文且受算力限制；评估依赖单一重现方法，可能忽略多模态或更大规模的科研挑战；系统对极端复杂或非公开数据的适用性待验证；

---

## 670. Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information

**arXiv ID:** 2611.08469 | [PDF](https://arxiv.org/pdf/2611.08469v1)

**作者:** Linqian Wu `[一作]` (University of Waterloo), Diaoxuan He `[通讯]` (University of Debrecen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在小型随机二进制网络中，作者通过精确计算IIT的整合信息，探讨了有限实验能否识别机制层级量。

**💡 创新点**

创新点在于将测量理论框架与IIT相结合，系统分析了行为盲区、隐藏单元导致的识别底线、测量值的尖锐边缘，并提出可通过有限干预证明的一侧陈述（证书）方法。

**🔧 技术方法**

使用了精确求解IIT公式、逻辑斯蒂网络模型、模拟实验、Clopper-Pearson置信区间、Hoeffding界，以及对测量设计的理论分析。

**📊 数据集**

主要使用了四五个单位的随机逻辑斯蒂网络与奇偶网络，生成了完整的转移概率表以进行计算和仿真。

**📈 对比分析**

通过对比不同观测设计（如一次性记录、延迟读取、全单位记录）以及不同推断方法（Box、Joint），发现访问越完整、推断更精细，区分力越强；但要获得高置信度的证书需要数百万次受控转移，性能受限。

**⚠️ 局限性**

局限性包括仅适用于极小网络、IIT计算复杂度随单元数指数增长、对模型假设敏感、需要大量实验样本、且边缘效应对测量结果影响显著。

---

## 671. Towards interpreting any residual stream direction by scaling activation inversion

**arXiv ID:** 2611.08465 | [PDF](https://arxiv.org/pdf/2611.08465v1)

**作者:** Celeste De Schamphelaere `[一作]`, Jiahai Feng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可训练的元模型MAEM（Maximally-Activating Example Metamodels），通过把残差流方向作为输入，让模型生成在目标方向上激活度最高的文本示例。

**💡 创新点**

创新点在于：①使用强化学习直接最大化目标方向与生成文本激活的余弦相似度；②不依赖标签或人类/LLM生成的描述，训练目标完全可验证；③单一模型可适用于多种方向（SAE特征、Steering Vector、LoRA写/读向量等）。

**🔧 技术方法**

技术包括：监督微调（SFT）+基于ScaleRL的强化学习；在冻结的语言模型上评估激活并作为奖励；在生成文本时注入残差方向；使用中心化评分消除共享残差成分。

**📊 数据集**

使用的数据集：Qwen3.6-27B的训练文本、English Web、Ultra‑FineWeb、Concept500、Anthropic Model‑Written Evaluations、LoRA trojan样例、J‑lens提供的工作空间数据等。

**📈 对比分析**

与语料库搜索、NLAs、J‑lens、Patchscopes等基线比较，MAEM在激活度、跨语言/领域泛化、Steering Vector解释、Backdoor读取、工作空间解码等任务上均显著优于基线，平均激活度提升约10–30%，在大多数任务中超过90%准确率。

**⚠️ 局限性**

局限性：①对约11%稀有/位置相关的SAE特征无法触发；②生成示例的多样性低，导致基于示例的解释召回率不足；③仅在单一模型上验证，缺乏跨架构的泛化评估。

---

## 672. Adversarial Training with Large Step Sizes: Implicit Bias, Optimization, and Sharpness

**arXiv ID:** 2611.08457 | [PDF](https://arxiv.org/pdf/2611.08457v1)

**作者:** Yi Feng `[一作]` (Aarhus University), Andrea Paudice `[通讯]` (Aarhus University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析了使用大步长的确定性梯度下降在对抗训练中的行为，包括隐式偏向、优化收敛以及损失曲率的演化；

**💡 创新点**

将隐式偏向和收敛分析推广到任意常数步长的对抗线性逻辑回归；证明对抗训练的梯度下降没有统一的收敛速率，并揭示了与标准训练不同的两阶段锐度动态；

**🔧 技术方法**

利用确定性梯度下降理论、比较点法、对抗逻辑回归的解析式、Huber 损失的最小化方程、以及实验验证中的 FGSM/PGD 生成对抗样本；

**📊 数据集**

在人工合成的线性可分数据上验证理论，并在 CIFAR‑10 上使用 MLP、CNN、ResNet 与 Vision Transformer 进行对抗训练实验；

**📈 对比分析**

与标准训练（ST）在相同网络与数据集下进行对比，发现 AT 在鲁棒最大间距方向上的偏向更大、优化速度更慢、锐度最终低于 2/η；

**⚠️ 局限性**

仅研究了确定性 GD，未考虑随机梯度或自适应步长；分析局限于 ℓ∞ 攻击，且对大步长的全局收敛机制尚未给出；

---

## 673. Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes

**arXiv ID:** 2611.08439 | [PDF](https://arxiv.org/pdf/2611.08439v1)

**作者:** Vladimir Beskorovainyi `[一作]` `[通讯]` (Besk Tech), Vladimir Beskorovainyi (Besk Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在调用时由客户端提供标签集合的 typed‑decision 模型的标签表述鲁棒性，比较了 Ru‑Jev 两个版本在不同词表和候选集大小下的性能。

**💡 创新点**

创新点在于系统地评估标签表述多样化对跨语言交叉编码器的准确性与置信度的影响，并提出了多维度对比协议与四种发布比较风险。

**🔧 技术方法**

使用了 307 M 参数的 mmBERT‑base 交叉编码器（Ru‑Jev），并在 13 个俄语任务上进行训练，采用多候选集评估、AURC 等度量。

**📊 数据集**

数据集包括俄语 SuperGLUE、RuCoLA、ruOpenBookQA、ruWorldTree、ruMTEB（评论情感、评分、意图、情景）和 toxicity，以及 1,000 个来自 MASSIVE 情景任务的验证问答。

**📈 对比分析**

对照方法是固定问题、改变标签词表，比较 v1.1 与 v1.3 两个版本在 5 候选和 18 候选两种设置下的准确率、风险‑覆盖率、置信误差；v1.3 在大多数词表上提高了 0.05–0.25 的准确率，但在缩写和无意义代码上无显著提升。

**⚠️ 局限性**

局限性包括仅一次训练、种子方差未完全估计、词表与标签风格混淆、仅针对单一俄语路由任务、温度校准不迁移至新标签/领域。

---

## 674. Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety

**arXiv ID:** 2611.08438 | [PDF](https://arxiv.org/pdf/2611.08438v1)

**作者:** Fazl Barez `[一作]`, Hassan Sajjad `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对当前在人工智能安全领域内的机理解释性（Mechanistic Interpretability）方法进行了系统综述，阐述了其理论基础、核心假设、技术流程、研究方向以及与安全实践的交互。

**💡 创新点**

提出了统一的评估框架，用以验证机理解释性研究的主张，并将现有研究归纳为六大研究取向，明确它们的目标、假设与权衡；同时将机理解释性与安全干预（如编辑、引导、监测）结合，探讨其实际可行性与局限。

**🔧 技术方法**

综述涵盖特征提取（稀疏自编码器、跨层变换器、激活可视化等）、电路发现与因果验证、训练动态分析、数据来源归因、自然语言解码、模型差异化与知识定位等方法，并对其实现细节与技术假设进行剖析。

**📊 数据集**

虽然为综述论文，作者引用了多种公开数据集（MMLU、BigBench、gsm8k、ReAct 等）和标准评测平台，以说明现有方法在多任务、多模型下的表现和安全评估场景。

**📈 对比分析**

通过对比表格与定量评估指标（可信度、可解释性、因果性、可扩展性、对安全任务的影响力）对不同方法进行层级评估；对比结果表明，稀疏自编码器在特征可解释性与可操作性上表现突出，但在大模型规模下仍显资源瓶颈；电路发现方法提供最强因果证据，但验证成本高。

**⚠️ 局限性**

主要局限包括：①机理解释往往不完整，难以捕获所有相关机制；②可扩展性差，现有方法在千亿参数模型上的计算与验证成本极高；③缺乏统一、严格的评估标准，导致结果可重复性差；④解释性与安全干预之间的映射关系仍未完全实现，解释不一定转化为可控操作；⑤多方法之间的语义不一致（如概念定义、特征表示）导致难以跨模型迁移。

---

## 675. SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization

**arXiv ID:** 2611.08483 | [PDF](https://arxiv.org/pdf/2611.08483v1)

**作者:** Qile Jiang `[一作]` (Brown University), George Em Karniadakis `[通讯]` (Brown University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SwarmEvolve，一个基于隐向量空间的低成本代理群体进行进化搜索的框架，用于解决数学和科学计算问题。

**💡 创新点**

通过让代理在隐空间推理、在 KV 缓存中通信，并使用余弦距离新颖性过滤与自动交叉，显著提升了搜索吞吐量、成本效率和多样性。

**🔧 技术方法**

采用 LLM 的 KV 缓存、隐式推理、隐空间通信、聚类化 swarm 设计、余弦距离新颖性过滤、参数调优代理等技术。

**📊 数据集**

在 21 个包含圆/格子填充、几何、分析、组合学、稀疏线性系统、特征值、Burgers 方程等的数学与科学计算任务上进行评测。

**📈 对比分析**

与 OpenEvolve、ShinkaEvolve、CodeEvolve 等 SOTA 在同一 GPU 与 60 分钟时间预算下对比，SwarmEvolve 在所有 LLM 规模下平均得分最高，速度提升至 3‑8 分钟即可达到其 60 分钟性能，吞吐量提高 5‑13 倍。

**⚠️ 局限性**

仅支持同一 LLM 的 KV 缓存共享，无法直接与线性注意力/状态空间模型兼容，KV 缓存内存随代理数线性增长，限制更大模型或更长时间运行。

---

## 676. XStack: A Benchmark for ML Repo Migration

**arXiv ID:** 2611.08472 | [PDF](https://arxiv.org/pdf/2611.08472v1)

**作者:** Linrong Cai `[一作]`, Zhuang Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个跨堆栈迁移基准 XStack，用来评估代码迁移代理将 PyTorch GPU 代码迁移到 JAX TPU 的能力。

**💡 创新点**

创新点在于引入隐藏目标、冻结第二阶段和双重 GPT 验证员，真实评估代理迁移完整仓库的可靠性，并将硬件迁移与框架迁移结合。

**🔧 技术方法**

使用大语言模型（如 GPT‑5.6、Claude Opus 等）作为迁移代理，评估框架采用 JAX+Flax+Optax，实验运行在 TPU v6e/v5p/v4 上。

**📊 数据集**

基准采用 20 个公开 PyTorch 仓库（包括 minGPT、nanoGPT、LoRA 等）及其对应训练数据集。

**📈 对比分析**

通过两阶段评估和两位 GPT‑5.6 验证员判定迁移是否符合隐藏目标；平均通过率约 59%，通过率随成本对数线性提升；对比实验显示仅框架迁移在 GPU 上几乎完美，而跨框架+硬件迁移仍然困难。

**⚠️ 局限性**

局限在于代理在验证、权重加载和完整实验复现上仍不可靠，评测耗时高、对成本敏感，且依赖大模型。

---

## 677. Canonical Cut Hierarchy in Almost-Linear Time via Reduction to Max-Flow

**arXiv ID:** 2611.08466 | [PDF](https://arxiv.org/pdf/2611.08466v1)

**作者:** Karthekeyan Chandrasekaran `[一作]` (University of Illinois Urbana Champaign), Harshul Sagar `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种随机化近线性时间算法，用最大流的归约方法高效计算无向图的标准割层次（canonical cut hierarchy）以及相关连通度、树打包数、森林度数等图结构。

**💡 创新点**

核心创新在于构造一种紧凑的“树基网络”将寻找最优划分问题转化为一次或多次最大流求解，并通过随机流取整与树交换技术实现树的“接近最优”快速更新，从而实现整体复杂度趋近于 (m+n)^{1+o(1)}，大幅改进了此前 O(nm) 或 O(n^3m) 的算法。

**🔧 技术方法**

主要技术包括：
- Tutte–Nash–Williams 最优划分与 Lagrangian 目标的等价性；
- 通过构造具有 O(m+n) 节点、O((m+n)log^2 n) 边的压缩网络（利用 Heavy‑Light 及平衡二叉树加速路径查询）；
- 随机化流取整（unbiased rounding）生成整数流；
- 两棵树的分割与 Edmonds arborescence 证明得到两条相互独立的生成树；
- 递归的 contraction‑restriction 机制重建完整的标准割层次。

**📊 数据集**

本文属于理论算法研究，无实验数据集；所有结果均为理论时间复杂度与正确性证明。

**📈 对比分析**

与先前的最优割、Gomory–Hu 树、树打包等算法相比，新的方法将时间从 O(nm) 降至 O((m+n)^{1+o(1)})（若最大流可近线性实现则进一步下降），并且在一次树求解后即可得到所有相关图属性，显著提高了算法效率与统一性。

**⚠️ 局限性**

局限性包括：
- 仅适用于容量多项式有界的整数图；
- 结果是随机化的，成功概率需通过多次重试或设置 η 保障；
- 依赖于近线性时间最大流实现，若该前提不成立则整体复杂度会随之上升。

---

## 678. OopsieRecover: Learning Transferable VLA Corrections from Semantic Interventions

**arXiv ID:** 2611.08462 | [PDF](https://arxiv.org/pdf/2611.08462v1)

**作者:** Arpit Bahety `[一作]` (University of Texas Austin), Roberto Martín-Martín `[通讯]` (University of Texas Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种从少量带语义注释的人类干预中学习可迁移纠正的框架，用 abductive 推理生成基于语义点集的门控成本，并在推理时对冻结的 Vision‑Language‑Action（VLA）模型进行动态引导；

**💡 创新点**

创新点在于：①通过 abductive 推理从少量干预中提取纠正意图；②将纠正表述为语义点集的门控成本，既能捕捉空间关系，又可在不同姿态和场景下零样本泛化；③在不重新训练基准 VLA 的前提下实现推理时策略引导；

**🔧 技术方法**

使用技术包括：视觉‑语言‑动作模型（VLM）进行实体定位与成本合成；3D 点集跟踪与语义关联；门控成本（gate、correction、task）表达式；动态模型预测候选动作块并基于门控成本排序；abductive 推理与多轮验证与细化；

**📊 数据集**

实验数据集主要为 RoboCasa 任务环境（仿真）和 DROID 硬件平台（真实世界），基准 VLA 为 OpenPI/π_0.5，干预数据每任务仅收集 5–8 条带语义注释的纠正；

**📈 对比分析**

与 human‑gated DAgger、OLAF、VLS 等基线对比，结果显示：在 0–2 级变异场景下，本文方法在零样本下实现 84% 的成功率（仅需 8 条干预），在真实任务中任务进度提升 60% 以上，且相对最优基线提升 74%；

**⚠️ 局限性**

局限性包括：只能处理基于语义点集的空间关系，无法直接编码接触力或未观测的物理属性；依赖准确的感知与动力学预测；引导仅基于有限动作块，可能无法覆盖所有恢复情形；验证机制仅检验对已记录干预的一致性，无法保证在所有场景下的完整性。

---

## 679. Latent RL Priors for Data-Efficient Dexterous Manipulation

**arXiv ID:** 2611.08458 | [PDF](https://arxiv.org/pdf/2611.08458v1)

**作者:** Yixiao Wang `[一作]` (University of California, Berkeley), Son Tran `[通讯]` (Amazon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究如何将大规模仿真中获得的强化学习控制先验迁移到仅靠有限视觉演示的灵巧操控任务上。

**💡 创新点**

创新点在于提出“Latent RL Prior”框架：先用特权信息训练流匹配动作解码器，然后通过学习视觉到隐式上下文的映射（仅用动作监督）冻结解码器，实现低样本视觉适配；并将单臂解码器共享至双臂任务，兼顾跨臂协调。

**🔧 技术方法**

核心技术包括：流匹配动作解码器（Transformer + AdaLN），视觉编码器（DINOv2），Q-Former用于上下文映射，隐式与显式两种视觉-解码器接口，保留预训练的本体感知通道，保留解码器冻结；在多臂任务中使用共享解码器与跨臂查询。

**📊 数据集**

使用大规模仿真教师收集的目标姿态对齐数据（100k轨迹）进行预训练；下游任务数据集包括：目标姿态对齐（4k演示），门升起（500演示），双臂装配（5k演示），每个任务仅收集视觉+本体感知+动作，未包含特权信息。

**📈 对比分析**

与全参数微调的π_0.5和GR00T N1.7、从零开始的相同架构以及显式接口进行比较。实验显示，在低样本和高精度需求下，Latent RL Prior 的成功率显著高于基线；门升起任务仅用50个演示即可超过使用500演示的基线，双臂装配仅用1k演示即可超过使用5k演示的基线；在目标姿态对齐任务中，低样本时优势明显，随着样本增多逐渐缩小。

**⚠️ 局限性**

局限包括：依赖强大的特权教师和大量仿真经验，冻结的解码器可能继承教师的失败模式并限制对新控制需求的适应；仅在仿真环境评估，未验证真实机器人性能；未研究不同机器人形态的迁移；并未探究导致数据效率提升的因果机制。

---

## 680. SoK: Post-Quantum Cryptography (PQC) Integration in Software Development: Approaches, Gaps, and Future Directions

**arXiv ID:** 2611.08442 | [PDF](https://arxiv.org/pdf/2611.08442v1)

**作者:** Shakya R. D. N. `[一作]` (University of Moratuwa), Nalin A. G. Arachchilage `[通讯]` (RMIT University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了后量子密码（PQC）在软件开发中的集成方法，并系统评估其与 NIST 标准、PQC 迁移指南、软件开发生命周期（SDLC）和 NIST CSF 2.0 的对齐情况，以及部署成熟度与评估证据。

**💡 创新点**

创新点在于提出四个跨领域的研究方向：①整合生命周期支持；②PQC 专用软件保障；③以能力为导向的开发者教育；④真实环境下的评估与部署验证；并通过 Systematisation of Knowledge (SoK) 方法形成了完整的知识体系。

**🔧 技术方法**

采用系统文献回顾、Evidence Mapping 与主题分析技术，依据 FIPS 203-205、PQC 迁移路线图、NIST CSF 2.0 和 NIST IR 8397 等权威标准进行维度映射与对齐评估。

**📊 数据集**

使用的“数据集”是 25 篇经过 PRISMA 2020 筛选的同行评审论文及相关标准文档（FIPS 203-205、PQC 迁移指南、NIST CSF 2.0、NIST IR 8397 等）。

**📈 对比分析**

比较方法是将每个研究按照标准匹配、SDLC 覆盖、CSF 功能、部署成熟度（概念、原型、实验、试点、生产）以及评估证据（功能测试、协议符合、性能、侧信道等）进行多维度编码与统计；大多数研究停留在实验层面，性能指标显示技术可行，但缺乏生产级验证。

**⚠️ 局限性**

局限性包括：仅关注英文同行评审文献，缺少现场部署案例；方法高度碎片化，缺乏端到端追踪；实验评估多在受控环境，难以反映真实系统约束；教育研究多针对学术学习者且评估周期短，未检验对实际开发技能的迁移效果。

---

## 681. ReproBridge: Building Reproducibility into ML Research Artifacts

**arXiv ID:** 2611.08452 | [PDF](https://arxiv.org/pdf/2611.08452v1)

**作者:** Yubo Wang `[一作]` (University of Waterloo), Jinsung Yoon `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 36 篇机器学习论文的实验程序进行构建、修复、记录和打包，生成四组件可复用的可复现工件（代码、环境、启动器、使用指南），并用 AI 代理对其进行评估。

**💡 创新点**

提出的 ReproBridge 框架将实验重建过程标准化为四个可分离的组件，并通过系统化记录与审核来显著提升实验复现成功率和降低后续重现成本，首次在大规模公开数据集上验证这一点。

**🔧 技术方法**

使用大型语言模型（Codex、Sonnet、Opus、Gemini）进行实验构建与复现；结合自动化审核器（Codex、Claude）进行结果完整性检查；采用统计兼容区间、参考误差估计等方法评估数值兼容性。

**📊 数据集**

构建数据集：36 篇 ICML/NeurIPS/ICLR 论文中的 1,029 个可验证任务，覆盖 36,000+ 任务；另外使用 45 个 CORE‑Bench Hard Capsule 进行外部基准对比；实验数据来自作者公开的代码仓库、官方 checkpoint 及手工验证的参考分数。

**📈 对比分析**

与原始代码（Raw）对比，RRA 在 4 种模型上平均提升 5.5–23.2 个百分点的成功率，降低 6.3–19.8% 的 token 使用和 16.8–23.7% 的 API 消耗；在 CORE‑Bench Hard 上成功率从 62.2% 提升到 73.3%，耗时从 33 分钟降至 14 分钟，成本从 8.43 美元降至 4.34 美元。

**⚠️ 局限性**

局限包括：仅评估 36 篇精选论文，未覆盖全部会议；依赖 5 次人工复现验证的参考数值，可能不足以捕捉所有不确定性；自动审核器仍会漏检违规；构建成本和人力投入未全面量化，复现收益主要以 API 费用衡量。

---

## 682. VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding

**arXiv ID:** 2611.08449 | [PDF](https://arxiv.org/pdf/2611.08449v1)

**作者:** Yu-Cheng Chou `[一作]` (Johns Hopkins University), Zhiding Yu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出VFlash，一种基于视觉原语结构的并行解码框架，实现快速准确的视觉规划。

**💡 创新点**

创新点是将视觉原语的已知结构用于一次性并行解码，直接接受结构有效的草稿而非逐令牌一致性，显著提升速度与空间精度。

**🔧 技术方法**

采用轻量级扩散草稿模型（DFlash）与结构化草稿布局，结合VLM特征并在推理中使用验证器与约束。

**📊 数据集**

在多种机器人与自动驾驶任务上评估，包括ShareRobot-Bench、VABench-V、COCO、Dense200、InternA1、AgiBot、MolmoBot、PointMotionBench、MolmoSpaces、LIBERO等数据集。

**📈 对比分析**

与自回归基线对比，VFlash在2D/3D轨迹、边界框和稠密点运动预测中降低误差、加速约25×，并在下游操控、驾驶等任务中提升成功率并减少每步延迟3–5×。

**⚠️ 局限性**

局限在于对结构化原语的依赖，无法处理完全无结构的视觉输出；同时需要额外的验证器与布局设计，且在某些稠密场景下仍存在误检。

---

## 683. Edge-Assisted Crowdsensing for Explainable Cyclist Safety Perception

**arXiv ID:** 2611.08441 | [PDF](https://arxiv.org/pdf/2611.08441v1)

**作者:** Hongsen Cao `[一作]` (Queen Mary University of London), Yasmin Fathy `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个基于边缘计算的骑行安全感知系统，利用手把摄像头采集视频，使用单一的视觉‑语言模型（VLM）提取结构化场景属性，再结合 OpenStreetMap 交通基础设施和骑行者运动学特征，最终用可解释提升机（EBM）给出 1–9 级安全评分。

**💡 创新点**

创新点在于：① 用单一 VLM 提示替代传统多模型检测管道，显著降低模型复杂度与计算量；② 将可解释提升机与交通属性融合，产生可解释的每特征评分点，为规划者直接提供成本‑收益分析的量化依据；③ 在网络边缘完成属性提取后仅传输 14 字段 JSON，极大降低上行带宽需求；④ 对连续路段的感知安全进行实时监测，填补现有只关注事故统计或问卷调查的空白。

**🔧 技术方法**

采用的技术包括：Vision‑Language Model（如 Grok‑2‑Vision、Qwen3‑VL、InternVL3 等）进行结构化描述；Explainable Boosting Machine（EBM）进行可解释评分；Edge‑Assisted Crowdsensing（MEG）框架实现边缘推理与压缩；GPS 与惯性测量单元（IMU）提供速度、加速度；OpenStreetMap 进行路网映射；空间分块交叉验证评估模型稳健性。

**📊 数据集**

使用的数据集来自两条伦敦骑行路线（Route A 11 km，Route B 2.9 km）的 1,419 帧手把摄像头视频（每 5 s 取一帧），共 32 名骑行者在自定义平台上进行 7,045 次安全评分，最终筛选出 1,362 帧带有 GPS 定位用于建模，包含 1–9 级评分及 14 维属性向量。

**📈 对比分析**

方法比较采用随机 5‑折交叉验证和空间分块交叉验证两种方案。EBM 回归模型在随机折中实现 72.2% 的二分类准确率、0.719 的加权 F1、MAE 1.011；与 XGBoost、随机森林等黑盒模型相差 ≤1 点；在空间分块验证中，各模型准确率差异不显著，说明模型对空间自相关不敏感；计算成本极低，评分步骤仅 0.4 ms CPU，VLM 提取 2.49 s GPU，且通过 20 s 采样间隔可将 1,000 名骑行者的上行带宽压缩至 54 Mbit/s。

**⚠️ 局限性**

局限性包括：① 数据仅覆盖两条伦敦路线，缺乏跨城或跨国泛化验证；② VLM 对每帧独立处理，未利用连续帧上下文信息；③ 标注受骑行者熟悉度影响，可能不适用于偶尔或非骑行者；④ 属性提取仍需高端 GPU，部署成本高；⑤ 现有模型仅解释单一安全等级，无法直接处理多模态交互或更细粒度风险评估。

---

## 684. SM4RT: Cascaded Feed-forward Model for 4D Reconstruction

**arXiv ID:** 2611.08436 | [PDF](https://arxiv.org/pdf/2611.08436v1)

**作者:** Pang-Chi Lo `[一作]` (Carnegie Mellon University), Tsung-Wei Ke `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种多任务级联前馈网络，用于从RGB视频中先重建3D几何，再利用该几何信息进行物体运动估计。

**💡 创新点**

通过将运动表示为相机坐标系下的对象流并引入弱监督Chamfer损失，有效分离相机运动与物体运动，提升了动态场景的运动估计精度。

**🔧 技术方法**

采用基于Transformer的DUSt3R骨干，3D位置编码（RoPE）增强注意力，弱监督Chamfer损失，EMA静态几何蒸馏和流头塑性注入等技术。

**📊 数据集**

在WildRGBD、BlendedMVS等静态数据集以及Waymo、Kubric-F、Spring、Point Odyssey等动态数据集上训练和评估。

**📈 对比分析**

与MonST3R、DPM、DynaDUSt3R等基线对比，在Waymo、Kubric-F、Point Odyssey的对象流EPE、相对位姿误差上均优于或竞争性表现，几何重建指标保持与现有方法相当。

**⚠️ 局限性**

仅处理两帧输入，依赖骨干在动态场景中的几何鲁棒性不足，难以扩展到长时序视频，需进一步提升动态几何重建和多帧集成能力。

---

## 685. EgoGeneralist: Unifying Egocentric Action-Video Tasks through Video-Native Modeling

**arXiv ID:** 2611.08477 | [PDF](https://arxiv.org/pdf/2611.08477v1)

**作者:** Yanlin Jin `[一作]` (Monash University), Kai Zhu `[通讯]` (Ant Digital Technologies, Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一框架，将预训练视频生成模型适配为视角视频生成、三维姿态估计和联合动作-视频生成的多任务学习平台；

**💡 创新点**

创新点在于角色自适应动作渲染（根据任务需求切换ego-view或Posegrid表示）以及跨流动作-视频耦合（共享latent空间与轻量LoRA），实现在保持视觉先验的同时高效对齐三维动作与视频；

**🔧 技术方法**

采用预训练的Wan2.2-A14B（I2V/T2V）视频生成器，冻结VAE编码，加入角色自适应渲染、共享自注意力、目标门控LoRA、流式噪声匹配训练；

**📊 数据集**

主要使用EgoDex实验室数据进行微调，并在Xperience‑10M、HOT3D等公开数据集上进行跨域评估；

**📈 对比分析**

在EgoDex与Xperience‑10M上实现的动作条件视频生成在PSNR/SSIM/FVD、PCK、IoU、Failure等指标上均优于或匹配多种专门基线；在姿态估计上MPJPE最低；在世界动作策略上在HOT3D中达成更低的目标距离和更高的到达率；

**⚠️ 局限性**

局限在于EgoDex的多摄像头融合导致投影误差，训练样本受限于实验室环境，缺乏站立、行走及户外等多样化动作场景，可能影响模型在更广泛情境中的泛化性能。

---

## 686. Monus vs. Interpretations in Semiring Semantics

**arXiv ID:** 2611.08474 | [PDF](https://arxiv.org/pdf/2611.08474v1)

**作者:** Phokion G. Kolaitis `[一作]` `[通讯]` (University of California Santa Cruz), Phokion G. Kolaitis (University of California Santa Cruz)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了基于解释的语义（canonical 与 compatible 解释）来定义在 semiring 注解数据库上的一阶逻辑查询，并证明了该语义无法表达 monus 操作。

**💡 创新点**

创新点在于首次定义了 canonical 与 compatible 解释语义并通过形式化证明展示了它们对 monus 的不可表达性，从而揭示 SQL 差集语义无法被这些语义捕捉。

**🔧 技术方法**

主要技术包括 semiring 代数、解释理论、归纳证明与对特定 semiring 的值域分析。

**📊 数据集**

论文未使用实验数据，所有结果均为理论证明。

**📈 对比分析**

未进行实验对比，结果为形式化可达性证明。

**⚠️ 局限性**

局限性是解释基语义无法表达 monus，因而无法模拟 SQL 差集；仅在包含 bag semiring 的正 semiring 上有效，且尚未给出完整可表达性阐述。

---

## 687. Cognitive Tours: Making Understanding a Deliverable in AI-Assisted Development

**arXiv ID:** 2611.08456 | [PDF](https://arxiv.org/pdf/2611.08456v1)

**作者:** Margaret-Anne Storey `[一作]` `[通讯]` (University of Victoria), Margaret-Anne Storey (University of Victoria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 Cognitive Tours，一个基于 Web 的工具，用以创建、维护和导航以“waypoint”为节点的认知与意图导览，帮助团队外化并持续更新对系统的理解。

**💡 创新点**

创新点在于将 AI 作为辅助审计与建议的工具，而非直接生成文档；同时引入“Docent 注释”与“waypoint 关系图”，实现人类主导的意义构建与持续迭代，从而专注于缓解认知债务和意图债务。

**🔧 技术方法**

技术实现包括：React 等前端框架构建可视化地图与 Markdown 导览；使用 Claude Code LLM 技能（audit-waypoints、audit-tours、suggest-waypoint、suggest-tour）进行审计与建议；与 GitHub、IDE 交互以获取文件与提交历史；以及基于 Web 的共享链接实现团队协作。

**📊 数据集**

主要使用作者自己的代码仓库（Cognitive Tours 及其演示项目）及其提交历史、ADR 文档和代码文件作为数据源；未使用公开的标准数据集。

**📈 对比分析**

未来计划通过人类手工、全自动 LLM 生成和半自动 AI 审计三种方式对比生成的代码说明与导览，评估定位精度、行为预测、解释质量、知识获取与转移等指标；目前尚无实验结果，性能评估仍在进行中。

**⚠️ 局限性**

局限性包括：需要人工持续维护与审核，工作量可能成为负担；AI 生成的建议可能不准确或过时，导致误导；过度依赖人工作者可能降低自动化程度；缺乏实证验证其对团队认知与意图债务的真正缓解效果。

---

## 688. A Hidden Markov Preference Model for Online Prediction of Adversary Actions

**arXiv ID:** 2611.08445 | [PDF](https://arxiv.org/pdf/2611.08445v1)

**作者:** Angel Daruna `[一作]` (SRI International), Ajay Divakaran `[通讯]` (SRI International)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于隐藏马尔可夫模型（HMM）与可行性掩码树的攻击者下一步行动概率分布预测框架，能够在每一步之前给出完整的行动分布，并在行动观察后更新对隐藏状态的信念。

**💡 创新点**

创新点包括：①将行为可行性与状态持续性统一建模，利用可行性掩码树在每个内部节点进行层次化软最大化；②在 HMM 的框架下给出对数损失下的可达上界、统一fallback保证和切换状态的经验分析；③实现了完全可滤波的推断和通过凸逻辑回归与 Baum–Welch 交替优化的高效学习。

**🔧 技术方法**

使用的主要技术有：隐藏马尔可夫模型、层次软最大化（hierarchical softmax）可行性树、贝姆‑韦尔斯算法、凸逻辑回归、对数损失理论（专家跟踪、KL 分析）、Uniform fallback 状态与一致性保证。

**📊 数据集**

实验采用 CAGE Challenge 2（CybORG++）网络安全基准的红蓝攻击日志，包含 56 个可执行动作，训练 40 轮、验证 10 轮、测试 25 轮，并在 60 条切换情境下评估。

**📈 对比分析**

与行动二元组模型、Masked GRU 以及基于策略类别的后验等基线对比，本文方法在测试集上每步 NLL 下降约 0.8 nats、准确率提升约 30%，参数量更少，且在策略切换后恢复速度更快（三步内开始下降 vs. GRU 的十步）。

**⚠️ 局限性**

局限性：训练分两阶段，未实现完整联合优化，导致后验对行动的反馈不充分；仅在 replay 设置验证，检测场景的鲁棒性尚未评估；缺失 fallback 状态会在未知策略下产生过度自信。

---

