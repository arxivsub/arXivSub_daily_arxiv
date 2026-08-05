# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-05 | 今日论文总数: 712

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. ANCHOR-RE: An Agentic Neuro-Symbolic Framework for Grounded Biomedical Relation Extraction

**arXiv ID:** 2608.03154 | [PDF](https://arxiv.org/pdf/2608.03154v1)

**作者:** Shufan Ming `[一作]` (University of Illinois Urbana-Champaign), Halil Kilicoglu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种无训练的神经符号框架ANCHOR-RE，通过LLM推理、知识库证据检索、示例检索以及离线学习的验证规则，实现了对生物医学关系抽取的可靠性提升。

**💡 创新点**

创新点包括：① 通过离线错误模式学习自动构建验证规则，以系统性抑制假阳性；② 将知识库证据、示例检索与LLM推理分离，形成代理式决策/验证流程；③ 在不进行任何参数微调的情况下，显著提升LLM在BioRE任务中的性能。

**🔧 技术方法**

技术手段包括：大语言模型（GPT‑5、Qwen 系列等）推理；向量检索（检索上下文示例）；外部知识库检索（UMLS、DrugBank、CTD 等）；离线错误模式学习与验证规则生成；语义类型约束；对比式示例检索。

**📊 数据集**

实验数据集：SemRepGS、DDI、ChemProt 三大BioRE基准；以及使用 PubTator3 处理的 2026 年后 100 篇医学文章的时间控制评估集。

**📈 对比分析**

与基线（直接提示）、仅知识库、仅检索、仅验证以及组合配置进行对比。结果显示：SemRepGS 微 F1 从 0.654 提升至 0.676；DDI 微 F1 从 0.769 提升至 0.872；ChemProt 微 F1 从 0.939 提升至 0.943；在时间评估中，随机抽样 500 条正预测的精度为 69%。在开放权重 LLM（Qwen 系列）上也观察到类似的性能提升。

**⚠️ 局限性**

局限性包括：① 只针对假阳性进行拒绝，未改进相似正类之间的区分；② 受知识库覆盖度影响，检索证据收益有限；③ 计算成本高于已训练的监督模型，需额外的离线错误模式学习开销；④ 需要手工制定语义类型映射和示例检索策略。

---

## 2. Emergence of Biased Consensus in Multi-Agent LLM Debates

**arXiv ID:** 2608.02827 | [PDF](https://arxiv.org/pdf/2608.02827v1)

**作者:** Maya Okawa `[一作]` `[通讯]` (Harvard University), Maya Okawa (Harvard University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多智能体大型语言模型辩论中出现的偏见共识现象，并提供理论解释与实验验证

**💡 创新点**

创新点在于将统计物理中的自旋模型引入LLM辩论，形成可量化的相变框架，揭示噪声与一致性阈值如何驱动偏见共识并提出通过多样化代理和温度调节来抑制该现象

**🔧 技术方法**

主要技术包括自旋模型的均值场近似、Boltzmann更新、LLM采样温度调控、以及对话式实验设计与参数反演

**📊 数据集**

使用了两类合成任务（二元选择与隐含性别偏差）以及两类现实任务（投资组合推荐与MT‑Bench评估），并对多种主流LLM（GPT‑4.1、Llama、DeepSeek等）进行评测

**📈 对比分析**

与单一LLM或全同温度的对照组相比，加入温度异质性和稀疏交互可显著降低偏见共识并提升任务表现，实验结果与理论预测高度吻合

**⚠️ 局限性**

局限性在于简化的辩论协议（短期记忆、全连接或随机交互、离散化输出），实验范围有限，未来需扩展到更复杂场景与高风险领域

---

## 3. PULSE: An Executable Contract Language for Spatiotemporal Knowledge Graph Engineering

**arXiv ID:** 2608.02630 | [PDF](https://arxiv.org/pdf/2608.02630v1)

**作者:** Dongxu Yang `[一作]` (DeepLethe), Ziyi Liang `[通讯]` (DeepLethe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种面向知识图的可执行合约语言 PULSE，能在单一运行时内统一处理声明、观察、约束和假设四种运算角色，提供时空计时器和事件调度。

**💡 创新点**

创新点在于将运算角色与写入效应分离并通过类型化时空约束在编译时和运行时保证安全性，形成一个可机械化的核心算子与六条安全性质；并通过与 GeoSPARQL、SOSA、SHACL、Sismic 等标准的组合验证其定位与完整性。

**🔧 技术方法**

主要技术包括：基于 OPM 的对象-过程-方法论思想、类型化时空模型、Lean 4 形式化证明、Python 运行时与无依赖几何核、GeoSPARQL 与 GEOS 的外部投影验证、以及对 GeoSPARQL 1.1 与 H3 的接口探针。

**📊 数据集**

使用的公开数据集有 NOAA IBTrACS（4,775 跟踪、1,476,290 区域过渡对）、GeoSPARQL 差异语料库（89 条 GEOS 对比案例）、以及自建的 91 点冷链轨迹和 37,440 条生成的时间序列。

**📈 对比分析**

比较方法包括：对 88 条单元测试与 Lean 证明验证核心属性；与标准工作流、Sismic 状态图的完整轨迹对比（完全一致）；与 GEOS 与 Apache Jena 的空间函数对比（无差异）；以及对 37,440 条生成轨迹与十种单字段变异模型的敏感度测试（所有变异均被检出）。性能上，核心检查在有限抽象下完成 3,534 次确定性和保存性验证；空间投影在 89 条测试中无差异，PostGIS 读写负载通过 10^5 步骤验证。

**⚠️ 局限性**

局限性包括：仅支持点与简单多边形、缺少多点、多面体、几何变换与不确定性处理；合约定位在可接受性与复制策略仍需外部决定；未实现完整的表面可证明性与跨语言（Python–Lean）细粒度细化；性能评估仅为局部实验，缺乏大规模分布式或实时吞吐量分析。

---

## 4. A Unified 2D Framework for DeepLesion Detection, Segmentation and Short Report Generation

**arXiv ID:** 2608.02805 | [PDF](https://arxiv.org/pdf/2608.02805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 5. The Ignition Is Real, and It Lives at the Readout: Latent composition, difficulty-clocked ignition, and the interface-constituted commit in a recurrent-depth reasoner

**arXiv ID:** 2608.03263 | [PDF](https://arxiv.org/pdf/2608.03263v1)

**作者:** Simon Lam-Muir `[一作]` `[通讯]` (Prime Calibre), Simon Lam-Muir (Prime Calibre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一篇复现性研究中，作者独立训练了30M参数的递归深度推理模型，并在训练过程中以每8个epoch的频率捕获读出排名（rank）与隐藏状态（Δh）两条通道的事件特征，验证“compositional ignition”在读出层面是否真实存在。

**💡 创新点**

研究的创新点包括：①确立Ignition是读出层面的真实事件，具备难度级别相关的首击时间、清晰度与离散度；②揭示决策边界在单步内出现显著幅度跳跃，隐藏状态在读出相关方向上冻结，而读出无关的径向幅度持续增长；③证明工作空间签名不依赖于人类语言训练数据，显示其本质与语言无关。

**🔧 技术方法**

技术上使用了：预注册实验设计与完整的证据记录库；双通道每步捕获（rank与Δh）；全签名门控检验、跑时、清晰度、窗口离散度等指标；角度/方向度量和层归一化坐标分析来捕捉状态几何变化。

**📊 数据集**

采用了闭合世界的合成事实数据集：200个实体与10个关系共2,000条事实，按跳深层级进行课程化训练。

**📈 对比分析**

与原始论文采用相同配置与种子进行交叉复现，比较首击迭代、清晰度与窗口离散度等指标，发现一致性良好；通过两次相同种子复现验证签名稳健；在隐藏状态通道未发现跨模型一致性或工作空间广播。

**⚠️ 局限性**

研究局限包括：①隐藏状态通道的跨模型一致性未得到验证；②未对读出接口的早停做干预验证其功能性；③实验仅在无语言合成任务上进行，尚未检验签名在更广泛真实语言或多模态任务中的通用性。

---

## 6. Sphere Retraction Normalizations

**arXiv ID:** 2608.02668 | [PDF](https://arxiv.org/pdf/2608.02668v1)

**作者:** Jie Zhang `[一作]` (National Central University), Min-Te Sun `[通讯]` (National Central University)

**通讯引用:** 2168 | [OpenAlex ID](https://openalex.org/A5060250892)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于球面重排（SpheretNorm）的深度连接方法，包括Proj‑SpheretNorm、Cay‑SpheretNorm以及可调参数p的通用p‑SpheretNorm，用于稳定Transformer等深层网络的训练。

**💡 创新点**

通过统一的球面重排框架，将GeoNorm、Proj‑和Cay‑变种统一为同一类二阶重排，推导出可调参数p的通用形式，揭示了不同重排对梯度传播与范数保持的影响，并提供了理论分析与Jacobian条件。

**🔧 技术方法**

利用Riemann几何中的重排映射（指数映射、投影重排、Cayley重排）、Gram–Schmidt正交化、学习率衰减、Softplus、ScaleNorm以及混合精度训练等技术，在nanoGPT框架下实现了SpheretNorm。

**📊 数据集**

在预训练阶段使用FineWeb‑Edu和OpenWebText两大文本语料；在下游任务阶段使用lm‑evaluation‑harness中的11个零样本基准进行评估。

**📈 对比分析**

与Pre‑LN、Pre‑DyT、Peri‑LN、Keel和GeoNorm等基线进行对比，结果显示SpheretNorm在训练/验证损失、困惑度以及绝大多数零样本指标上均优于所有基线，尤其在24层和36层模型中表现突出。

**⚠️ 局限性**

该方法在保持ℓ2范数的前提下不兼容非正交表示的等变网络，且在当前实验中未在更大规模Transformer上验证，需进一步扩展与评估。

---

## 7. Attacking and Defending Multi-Agent Collaborative Filtering Systems Through Connectivity

**arXiv ID:** 2608.03272 | [PDF](https://arxiv.org/pdf/2608.03272v1)

**作者:** Anjun Hu `[一作]` (University of Oxford), Kurt Cutajar `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究多智能体协同过滤系统中连通性对攻击与防御的影响，复现并评估多种基于LLM的攻击和防御方法；

**💡 创新点**

将通用MAS攻击/防御迁移到协同过滤场景，系统量化候选计数与目录浓度对攻击效果的非单调、角色异质性影响，并提出基于静态图指标的预测方法；

**🔧 技术方法**

使用AgentCF框架、LLM驱动的代理；改造CORBA、NetSafe、MAMA、MASLeak、TOMA、MASTER、G‑Safeguard、BlindGuard、T‑Guard、M‑Guard等攻击/防御；利用LLM判别器评估ASR并采用SIS传播模型做预测；

**📊 数据集**

MovieLens‑100K 子集（100用户，变更项目数）；

**📈 对比分析**

在候选计数(k=1,2,3)和目录浓度(ρ=0.5,1,2)上系统性评估攻击成功率、泄露率及防御ASR差异；发现高连通时攻击更有效，防御对高k更有成效，但结果表现出非单调性；

**⚠️ 局限性**

实验规模有限（仅100用户、单域），未评估推荐质量影响，LLM判别器误差未完全校正，攻击者模型和防御策略的覆盖范围有限，未考察更大稀疏系统的泛化性；

---

## 8. When Refusal Looks Safe: The Refusal-Cue Shortcut in Safety Guard Models

**arXiv ID:** 2608.03201 | [PDF](https://arxiv.org/pdf/2608.03201v1)

**作者:** Yu Feng `[一作]` (University of Sydney), Jieping Ye `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 WildGuardMix 与 GR-Train 等安全守卫训练数据进行审计，发现拒绝提示词与非有害标签的极端共现导致的拒绝‑提示词短路，进一步验证其在多模型、多位置和多规模下的普遍性，并提出一种稀疏补充掩码的后处理方法来消除该短路。

**💡 创新点**

首次揭示并量化响应级安全守卫中的拒绝‑提示词短路，证明其跨模型、跨位置的广泛存在，并在不重新训练模型的前提下，通过稀疏掩码实现对短路的高效抑制。

**🔧 技术方法**

使用稀疏补充掩码（Sparse Complementary Masking）在冻结模型参数的情况下，对注意力头和 MLP 通道进行二值化门控，配合梯度优化以最小化短路误判，进而实现后处理修正。

**📊 数据集**

利用 WildGuardMix、GR-Train 作为训练/审计数据，WildGuardTest 与 Aegis2 Test 作为评估基准，并在附录中验证了 BeaverTails 数据集。

**📈 对比分析**

通过比较原始模型与掩码模型在 WildGuardTest 与 Aegis2 Test 上的 @3、F1 与召回率，结果表明掩码能将 @3 降低约 79% 并保持或提升有害检测 F1，且在中间/尾部位置、不同规模模型和外部数据集上亦保持显著改进。

**⚠️ 局限性**

仅关注响应级安全守卫的拒绝‑提示词短路，缺乏在真实部署环境中的验证，可能忽略其他类型的短路或误判；掩码方法需额外 GPU 训练；对不同数据分布或任务的迁移能力尚未充分评估。

---

## 9. Internalizing Academic Writing Workflows for Introduction Generation via Struct-Aware Policy Learning

**arXiv ID:** 2608.03138 | [PDF](https://arxiv.org/pdf/2608.03138v1)

**作者:** Meicong Zhang `[一作]` (East China Normal University), Dejia Song `[通讯]` (Xiaohongshu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 StructPO，一个结构感知的策略优化框架，用单通道阶段标记实现学术论文引言的自动生成。

**💡 创新点**

创新点在于将多阶段写作流程内部化，利用阶段标记与结构感知相对优势（SRA）实现局部信用分配，并在训练时加入修订引导惩罚，将修订行为嵌入一次性生成。

**🔧 技术方法**

核心技术包括基于 PPO 的强化学习、SRA（结构相对优势）、阶段级奖励设计（长度、语义相似、结构合理性）、修订引导优化、嵌入相似度、DeBERTa 句子分类器等。

**📊 数据集**

使用了约 3,200 篇 ACL 2021-2025 论文作为训练与验证，1,176 篇 ACL 2025 论文作为测试，并在 141 篇 CVPR 论文上进行零样本迁移评估。

**📈 对比分析**

通过与提示式、工作流式、SFT、GRPO 以及多款闭源 LLM（GPT‑4o、GPT‑5.1、Claude Opus 等）对比，自动化指标上 StructPO 在语义、章节、长度和结构分数均超越大多数基线，Qwen3‑32B 版在人工评估中以 53.3% 的胜率击败 GPT‑5.1。

**⚠️ 局限性**

局限性包括固定的八阶段模板限制了对理论综述或非标准论文的适用性；在长度控制、检索增强以及跨领域适配方面仍有提升空间。

---

## 10. Beyond the Hivemind: Escaping LLM Homogeneity via Meta-Persona Anchoring and Sequential Temperature Scaling

**arXiv ID:** 2608.02618 | [PDF](https://arxiv.org/pdf/2608.02618v1)

**作者:** Tairan Fu `[一作]` (Politecnico di Milano), Javier Coronado-Blázquez `[通讯]` (Banco de España)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Meta-Persona Anchoring与Filtered Temperature Scaling两阶段框架，旨在打破大型语言模型的人工蜂群效应，显著提升生成文本的多样性与创意度。

**💡 创新点**

创新点在于将Top-p过滤与极端温度扩展相结合，先锁定语义合法的候选词再进行高温采样，并让模型自我选择并描述persona以实现语义方向与采样策略的解耦与协同。

**🔧 技术方法**

采用了Top-p过滤、极端温度扩展（T≥4.0）、Meta-Persona Anchoring以及自定义的两步采样流程来增强多样性。

**📊 数据集**

使用INFINITY-CHAT数据集进行评估。

**📈 对比分析**

相较于基线（T=1，未使用Persona），实验显示平均余弦相似度从约0.85降至0.65，且自动评测的连贯性得分保持在7–7.5之间，表明多样性提升的同时保持了逻辑一致性。

**⚠️ 局限性**

局限包括仅在20B以下模型上验证、仅测试单一数据集、温度过高可能在长文本生成中产生连贯性下降、Meta-Persona生成过程增加响应延迟，并未在大规模模型或多任务场景下进一步验证。

---

## 11. NOMADD: Numerical Optimization of Models Adapting to Data Drift

**arXiv ID:** 2608.02845 | [PDF](https://arxiv.org/pdf/2608.02845v1)

**作者:** Swapn Shah `[一作]` (University of North Carolina at Charlotte), Keith Burghardt `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后置（post‑hoc）方法NOMADD，利用每个时间段的模型拟合结果，通过低秩分解和衰减线性趋势预测未来决策边界，从而在不重新训练模型的前提下降低表格数据的概念漂移和数据漂移影响。

**💡 创新点**

创新点在于：①可对任何可解释或黑盒模型（树、神经网络、TabPFN等）统一处理；②使用低秩压缩和衰减趋势对参数漂移进行预测，避免过度拟合；③所有超参数均通过内部前向验证自动选取，无需未来标签；④在保持准确率的同时大幅减少推理时间和显存消耗。

**🔧 技术方法**

核心技术包括：在每个时间段训练单独模型并与全局锚模型对齐；计算参数/预测字段的差值矩阵并做截断SVD得到隐含时间轨迹；对每条轨迹做衰减岭回归预测；使用收缩因子将预测增量应用于锚模型；可选的数据还原步骤（均值/协方差回归）。

**📊 数据集**

使用了DR‑TabPFN基准中的18个数据集（15个真实数据集，3个合成漂移流：Hyperplane、RandomRBF、2‑Moons），并按基准协议进行训练/测试划分。

**📈 对比分析**

在与冻结基线和DR‑TabPFN的对比中，NOMADD在所有基模型上都优于其冻结版；对XGBoost在真实数据上显著提升（+0.011 ROC‑AUC，p=0.005）；与DR‑TabPFN的整体表现相当（0.817 vs 0.798），但在轻量级模型上推理速度提升3–4个数量级，显存占用几KB。

**⚠️ 局限性**

局限性包括：假设参数随时间呈线性变化，无法捕捉突发漂移；对小样本训练期噪声敏感；对TabPFN等ICL模型仅改动输出层，未直接更新参数；特征空间固定且假设足够表达；对非线性或非平稳漂移的预测效果有限。

---

## 12. Inverted Detection and Control in Steering Vectors

**arXiv ID:** 2608.02957 | [PDF](https://arxiv.org/pdf/2608.02957v1)

**作者:** Max Torop `[一作]` (Northeastern University), Jennifer Dy `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决了大型语言模型在推理时使用Steering Vectors（SVs）控制概念表达时出现的“反向调控”现象，提出了新的识别和修正方法。

**💡 创新点**

发现并系统性描述了所谓的Inverted-Steering Vectors（ISVs），即既高度判别又与正类对齐，却在推理时导致相反效果的向量，并通过几何学解释其对下游表示的“伪造”影响；提出了“representation response”指标来无生成地识别ISVs，并利用此指标对Inference Time Intervention（ITI）进行定向符号翻转以提升性能。

**🔧 技术方法**

使用注意力头输出的线性翻译方法构造SVs，设计并计算inner‑product response (IPR) 与representation response (Γ)；通过统计学方法（Spearman相关、Hoeffding不等式等）评估和估计这些指标；在STEER基础上加入符号翻转步骤（ITI‑RRF）。

**📊 数据集**

在Gemma 3 12B、Qwen 2.5 14B、Olmo 3 7B三大模型上，针对五个概念（纠正性、财富追求与短视、拒绝、真实性）使用多项选择（MWE、TruthfulQA）和开放式（OE）数据集进行实验。

**📈 对比分析**

与传统ITI方法对比，采用representation response进行ISV检测与符号翻转后，提升了27/30个实验的性能，改进幅度从+0.9%到+138%，在概念推广与抑制两种任务中均表现更优。

**⚠️ 局限性**

仅针对基于线性翻译的SVs，未探究更复杂或优化的调控框架；ISV识别与校正依赖于足够的下游判别头和AUC阈值；在某些模型‑概念组合中仍有提升空间，需进一步研究。

---

## 13. Reinforcement Learning with Evolving Rubrics as Rewards for Audio Reasoning

**arXiv ID:** 2608.02831 | [PDF](https://arxiv.org/pdf/2608.02831v1)

**作者:** Fangxu Yu `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (MBZUAI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于强化学习的音频推理框架，通过自适应生成与音频证据绑定的评判 rubrics 并将其作为过程级奖励，提升大规模音频语言模型的推理能力。

**💡 创新点**

创新点在于：①将评判 rubrics 与原始音频直接关联，保证过程级奖励与声学证据一致；②在训练过程中自我演化 rubric 以持续提升难度，避免奖励饱和；③结合长度惩罚避免过度“过度推理”导致的冗长、错误推理链。

**🔧 技术方法**

技术核心包括：基于 GRPO 的策略优化、音频感知 rubrics 的生成与判定（使用 Gemini‑3.1‑Pro 作为生成器/判定器）、过程级 rubric 评估、动态权重分配与方差过滤、长度惩罚正则化。

**📊 数据集**

使用 AVQA 数据集（约 40,000 条音频‑文本对）进行训练，并在三大音频推理基准 MMAU、MMAR 与 MMSU 上进行评测。

**📈 对比分析**

相较于 13 种开源 LALM、3 种专有模型及多种训练后推理方法，本文方法在三大基准上均取得最优或最接近最优的准确率，提升幅度从 5% 以上到 15% 以上，且在不同声学类别（语音、音乐、环境声）上均表现稳健。

**⚠️ 局限性**

局限性包括：依赖高质量的 rubric 生成与判定模型，若生成器/判定器不足则奖励噪声较大；计算开销较高（需要多条 rollouts 与多轮 rubric 迭代）；在极端复杂推理场景下仍可能出现冗长推理链或轻微的 hallucination。

---

## 14. DocTrace: Towards Traceable Long Document VQA via Hierarchical Evidence Graph Reasoning

**arXiv ID:** 2608.03292 | [PDF](https://arxiv.org/pdf/2608.03292v1)

**作者:** Le Xiang `[一作]` (Baidu Inc), Long Zeng `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DocTrace 框架，将长文档视觉问答转化为证据图推理问题，分为证据定位、结构化解析和图推理三阶段；

**💡 创新点**

创新点在于显式构建证据图以实现节点级证据可追溯，采用两阶段训练（联合 SFT + 任务专属 GRPO）实现证据定位与图推理的强化学习优化；

**🔧 技术方法**

技术包括多模态大语言模型（Qwen3-VL-8B-Instruct）、低分辨率粗定位、PaddleOCR-VL-1.5 结构化解析、图神经网络式证据图生成、GRPO 强化学习与专属奖励（定位、图可信度、答案正确性）；

**📊 数据集**

使用 MMLongBench-Doc、LongDocURL、SlideVQA 三大长文档基准，以及自建长文档语料用于训练；

**📈 对比分析**

与多类基线（E2E、RAG、Agent）及专有模型（Gemini、GPT-4o、GPT-4.1、Claude）对比，DocTrace 在 MMLongBench-Doc、LongDocURL、SlideVQA 上分别获得 52.9%、56.4%、85.1 F1，均比现有开源与专有模型高出 11-14 点；

**⚠️ 局限性**

局限包括对超长文档的定位覆盖率仍随长度下降，需更强的定位策略；图推理的结构化解析依赖 OCR 质量；对非结构化、图像化证据的处理尚待进一步完善。

---

## 15. A Unified Resolution-Conditioned Framework for Orthogonal Line-Scanning Image Fusion

**arXiv ID:** 2608.03107 | [PDF](https://arxiv.org/pdf/2608.03107v1)

**作者:** Yiming Gong `[一作]` (University of Michigan), Kai Wang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

提出了一种统一的基于线扫描显微镜的分辨率条件图像融合框架，能在单一模型中处理不同光栅宽度下的图像融合。

**💡 创新点**

创新点在于将分辨率比作为连续条件通过FiLM注入网络，并改进RELA为自适应RELA，使用多尺度深度卷积与可学习温度来调节注意力，以适应不同的方向性失真。

**🔧 技术方法**

采用线性注意力Transformer（RELA）+ FiLM + 多尺度深度卷积 + 可学习注意力温度等技术实现自适应融合。

**📊 数据集**

使用基于物理PSF模拟的多分辨率数据集，共12种光栅宽度，包含约1,945张真实样本及其旋转增强，共计约27,000个训练样本。

**📈 对比分析**

与经典的SE-FDMF、单一配置的Attn-DualUnet等方法对比，实验显示在所有光栅配置下均能获得34–40 dB的PSNR，显著优于未条件化的多配置模型，且与单配置专家相近，且在未见过的插值配置下保持平滑性能。

**⚠️ 局限性**

局限性包括仅在模拟PSF数据上验证，缺乏真实实验数据的测试；未检验完全不同样本类型的泛化；以及目前仅限于二维平面扫描，3D扩展尚未实现。

---

## 16. DP-MemView: A Memory Interface for Attribute-Level Transcript Privacy in Long-Term LLM Agents

**arXiv ID:** 2608.03130 | [PDF](https://arxiv.org/pdf/2608.03130v1)

**作者:** Jong Wook Kim `[一作]` (Sangmyung University), Beakcheol Jang `[通讯]` (Yonsei University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一个名为 DP-MemView 的差分隐私内存视图接口，能在长周期记忆 LLM 代理中通过私有视图选择和属性级账本实现对完整交互记录的纯 DP 保护。

**💡 创新点**

创新点在于：① 将长周期记忆隐私建模为适应性转录隐私问题；② 提出完整的接口契约（包含公共视图词表、读取集计费、私有 EM 选择、预期限额检查和后置通用视图）以实现全局 DP；③ 引入在线与预分配两种预算策略，并在实验中验证了两者在隐私与个性化间的权衡。

**🔧 技术方法**

技术方法包括：差分隐私理论（纯 DP、指数机制、适应性组合）；属性级记忆组划分与相邻性定义；账本计费与路径级预算检查；以及公开的公共视图词表与后置通用视图回退机制。

**📊 数据集**

数据集方面使用了：① 合成的 PairedMem 对照轨迹（320 对记忆，4-6 个受保护属性位置）；② 公开语料迁移轨迹（80 对记忆，来自 LoCoMo 角色、事件历史、医学问答和共情对话等公开语料）。

**📈 对比分析**

与基线方法（GenericOnly、RawReadSet、TypedMask、TaskMin、OutputFilter）以及两种 DP-MemView 模式（online、prealloc）在 Qwen2.5、Llama-3.1 和 Gemma-2 上进行对比。实验结果显示 DP-MemView 在隐私上接近无差异可辨识（AUC≈0.5、TPR@5≈0.05），而在个性化与整体响应质量（U、tRec）上优于 RawReadSet、TaskMin 和 OutputFilter；预分配模式在已知预留预算时能获得更高的 tRec，而在线模式在未知预算时能保持更低的 Unsup。

**⚠️ 局限性**

局限性包括：① 需要事先定义属性组和公共视图词表，若划分不准确或词表不足会影响效果；② 目前不支持内容依赖的检索机制；③ 依赖受信任的接口边界，若该边界被突破则隐私保障失效；④ 对于多个属性组同时变化的情形，需计算最小覆盖成本，复杂度较高。

---

## 17. SMOPD: Multi-Reward Reinforcement Learning via Specialize-and-Merge Online Policy Distillation

**arXiv ID:** 2608.03092 | [PDF](https://arxiv.org/pdf/2608.03092v1)

**作者:** Wen Wang `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SMOPD 两阶段多奖励优化方法，先通过奖励优先级分离专门化教师学习稀疏/密集奖励，再通过在线策略蒸馏合并为统一模型

**💡 创新点**

在多奖励 RL 中通过奖励优先级分离专门化教师，解决奖励稀疏与密集冲突，并使用统一学生通过在线策略蒸馏实现平衡

**🔧 技术方法**

GDPO、GRPO、DPO 的奖励归一化与优先级权重、前向 KL 在线蒸馏、top‑κ 近似、GDPO anchor 等技术

**📊 数据集**

RLLA-4K tool‑calling、BFCL AST、API‑Bank、Alpaca、HH‑RLHF、PKU‑SafeRLHF 等数据集

**📈 对比分析**

对比 GRPO、GDPO、GD²PO、单一教师等基线，SMOPD 在 1.5B/3B/7B Qwen2.5、Llama‑3.2 上在补充奖励和冲突奖励场景均优于基线；格式合规率从 8.8% 提升至 97.5%，Safe Domain Overall 提升至 5.65（约 +0.15）

**⚠️ 局限性**

对奖励极稀疏的情况仍需手动调节优先级，学生对教师质量高度依赖，anchor 仍受 GDPO 约束，未探索更强的序列级优化或自适应混合策略

---

## 18. Channel-wise Dynamic Knowledge Distillation via Adaptive Sample Generation for Action Recognition

**arXiv ID:** 2608.03100 | [PDF](https://arxiv.org/pdf/2608.03100v1)

**作者:** Ping Li `[一作]` (Hangzhou Dianzi University), Mingli Song `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了ASCD框架，即自适应样本生成与通道级动态知识蒸馏，用于压缩动作识别模型。

**💡 创新点**

创新点在于：①通过梯度驱动自适应生成样本，动态改变输入分布；②将特征投影到频域并用可学习的高斯掩膜保留关键运动频率；③采用教师-学生中心频率差异加权的通道级蒸馏损失，动态调节蒸馏强度；④利用类梯度的余弦相似度构造类分布约束。

**🔧 技术方法**

技术手段包括知识蒸馏、3D离散傅里叶变换、可学习高斯频域掩膜、通道级加权MSE损失、类梯度余弦相似度、动态采样更新调度。

**📊 数据集**

实验数据集涵盖视频数据：UCF101、Kinetics‑400、Something‑Something‑v2；图像数据：CIFAR‑100、ImageNet。

**📈 对比分析**

与CKD、DKD、CTKD、GKD、CrossKD、DualKD、DCSF、SAKD、DIST+等SOTA蒸馏方法对比，在所有基准上均取得或逼近最高的Top‑1/Top‑5准确率；如UCF101上提升约+2.5%、CIFAR‑100上+1.3%、ImageNet上+0.65%。

**⚠️ 局限性**

局限性在于：自适应样本生成和频域处理带来额外计算与内存开销；对超参数（如λ、η、α、γ）的敏感性需要仔细调优；在极大规模数据或实时场景下的可扩展性尚待进一步验证。

---

## 19. AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding

**arXiv ID:** 2608.02989 | [PDF](https://arxiv.org/pdf/2608.02989v1)

**作者:** Shuang Liang `[一作]` (Imperial College London), Wayne Luk `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在MoE目标模型上进行投票验证的推理选择器，能够根据每个验证块的承诺概率自适应地确定专家集合大小并在专家离线时进行缓存感知裁剪。

**💡 创新点**

创新点在于结合离线估计的承诺概率与目标路由器得分形成承诺加权需求，利用该需求的有效秩动态决定专家数，且在专家离线场景下依据缓存状态进行专家裁剪，省去手动预算与专家预测模型。

**🔧 技术方法**

采用投票验证（Speculative Decoding）+ MoE专家选择器，使用承诺加权需求、有效秩自适应大小、缓存感知裁剪三阶段流程，并在SGLang+CUDA Graphs框架下实现。

**📊 数据集**

使用Qwen3-30B-A3B与GPT-OSS-120B两大MoE模型，配合对应的EAGLE-3抽稿模型，在GSM8K、MATH500、HumanEval和MBPP四个基准数据集上进行评估。

**📈 对比分析**

与Vanilla AR、Standard SD以及固定预算选择器进行对比，平均准确率仅低0.27个百分点；在全显存环境下吞吐量提升1.29×，在物理专家离线环境下提升2.06×，同时将主机到设备的专家权重传输量降低约73–77%。

**⚠️ 局限性**

局限性包括：轻微的准确率下降、需要离线数据估计承诺概率、对缓存大小和负载分配的敏感度，以及在极大模型或极低显存条件下仍可能面临专家集合规模不匹配的问题。

---

## 20. Better, Stronger, Faster, and Broader: Structured All-Mask Prediction for MLLM-Based Segmentation

**arXiv ID:** 2608.02791 | [PDF](https://arxiv.org/pdf/2608.02791v1)

**作者:** Jiazhen Liu `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 All-Mask Prediction 框架，分为二阶段：第1阶段生成对话文本并输出结构化目标列表，第2阶段一次性非自回归预测所有掩码，支持从单目标二值掩码扩展到多目标结构化掩码，兼顾对话能力、精度和速度。

**💡 创新点**

核心创新：①将密集掩码预测从自回归拆分为一次性预测；②通过 Phase‑1 生成的 ID 列表动态绑定到统一的多类别掩码空间，实现多目标一次性输出；③引入高分辨率掩码标记扩展，解决遥感小目标难题；④保持原生 MLLM 文字生成接口，避免破坏对话能力。

**🔧 技术方法**

采用 Qwen2‑VL（以及可选 LLaVA）作为大模型基座；使用图像与掩码标记对齐的视觉嵌入、混合注意力机制、线性掩码头（二分类或 200 类多分类）；训练目标为联合文本交叉熵 + 掩码交叉熵/Dice，支持结构化 JSON 目标列表生成。

**📊 数据集**

训练数据覆盖多任务：RefCOCO/RefCOCO+/RefCOCOg、gRefCOCO、ReasonSeg、RRSIS‑D、EarthReason、COCO‑Stuff、ADE20K‑150、Pascal Context‑59、Pascal VOC‑20、MUSE、以及 LLaVA‑665k 视觉指令集。

**📈 对比分析**

与 Embedding‑Prediction、Next‑Token‑Prediction 等现有 MLLM 分割方法对比，在 RefCOCO 家族、gRefCOCO、ReasonSeg 取得 cIoU/IOU 最高；在遥感小目标上取得最高 Acc@0.5/IoU；在 ADE20K、Pascal 等开放词表语义分割上实现 mIoU 最高；在 MUSE 及实例分割上实现 gIoU/IOU 领先；推理延迟显著低于自回归方法，单目标 12 类从 13.50 s 降至 5.16 s，且多目标一次性推理比重复单目标快数倍。

**⚠️ 局限性**

局限性：①需要 Phase‑1 正确生成结构化目标列表，若生成错误会影响后续掩码；②多类别上限受 200 类固定容量限制，过多目标需截断；③高分辨率标记扩展对 GPU 记忆和计算资源有较高要求；④在极端多目标或极大图像场景下，混合注意力与 KV 缓存仍可能成为瓶颈。

---

## 21. PixelUp: Zero-Shot Semantic Feature Upsampling for Fine-Grained Vision Tasks

**arXiv ID:** 2608.02792 | [PDF](https://arxiv.org/pdf/2608.02792v1)

**作者:** Deepank Singh `[一作]` (University of Houston), Vedhus Hoskere `[通讯]` (University of Houston)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练了一种无监督的基于注意力的特征上采样器 PixelUp，用于从低分辨率视觉基础模型 (VFM) 的特征恢复高分辨率特征，支持语义分割、单目深度、无监督分割与开放词汇分割等多任务。

**💡 创新点**

创新点在于：①通过一次无标签训练完成所有分辨率的上采样，避免多模型、多参数；②采用交叉注意力链（Cross‑Attention Chain）将低分辨率 key 与高分辨率 query 关联，能够处理任意输出尺寸；③引入预训练语义编码器（DINOv3 ConvNeXt‑S）为查询提供语义先验，实现“semantic guidance”而非仅依赖低分辨率特征；④在多种评估协议下（linear probing、open‑vocab、unsupervised、深度、边界‑aware）证明其相较先前最佳方法 NAF 的显著优势。

**🔧 技术方法**

主要技术包括：无监督教师蒸馏（MSE 目标）; 交叉注意力上采样 (key 低分辨率池化, query 高分辨率); 语义编码器融合; linear probing, monocular depth head, DiffCut, RADSeg 等下游任务；AdamW 优化，bfloat16 精度；多尺度训练 (随机尺寸与缩放因子) 以实现任意输出分辨率。

**📊 数据集**

训练集：SA‑1B + COCO 共 134,473 张无标签图像；评估集：Cityscapes、ADE20K、VOC20、KITTI（开放词汇分割）、NYUv2（深度）、DiffCut（SSD‑1B）无监督分割。

**📈 对比分析**

与基线（最近邻、双线性、JBU、AnyUp、NAF）以及多任务评估进行比较：在 Cityscapes linear probing mIoU 68.16 vs NAF 64.96（+3.20）；开放词汇分割 mIoU 47.04 vs 44.37（+2.67）；NYUv2 depth RMSE 0.399 vs 0.401（+0.002）；无监督分割 44.55 vs 44.49（+0.06）；open‑vocab segmentation 71.82 vs 69.41；边界‑aware mIoU 在 8px、4px band 上分别提升 4.29、4.20。总体而言，PixelUp 在细节与边界表现尤为突出。

**⚠️ 局限性**

限制：①对预训练语义编码器和教师网络的质量高度依赖，教师或编码器差异会显著影响性能；②在极大 upsampling 比例或高分辨率网格下会导致显存溢出（如 AnyUp、NAF 1024 grid）；③对粗粒度区域提升有限，主要优势体现在细节/边界；④计算成本主要集中在冻结的 VFM 与语义编码器上，需较大 GPU；⑤缺乏动态多尺度/多任务自适应机制。

---

## 22. Trajectory-Guided Forget-Recover Network for Continual LLM Unlearning

**arXiv ID:** 2608.03123 | [PDF](https://arxiv.org/pdf/2608.03123v1)

**作者:** Zezheng Wu `[一作]` (Guilin University of Electronic Technology), Jingwei Zhang `[通讯]` (Guilin University of Electronic Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型的持续性机器无学习（continual unlearning），提出了 Trajectory-guided Forget-Recover Network (TFR-Net)，通过跟踪通道级风险轨迹进行持续性目标相关通道抑制，并在容量损失时恢复低风险的休眠通道，保证在多次无学习请求下仍能保持高遗忘效能和保留性能。

**💡 创新点**

① 用通道风险轨迹（包括历史风险、风险变化、置信度等多维度）区分持久的目标相关通道与瞬时热点；② 结合保留安全的容量恢复机制，在保证保留性能不下降的前提下逐步激活休眠通道，缓解多次抑制导致的可用模型容量消耗；③ 采用原子接受/回滚策略，使每次无学习请求的结构更新保持一致性。

**🔧 技术方法**

1) 轨迹感知通道风险计算（激活-梯度得分、正则化、风险记忆等）; 2) 基于轨迹优先级的通道抑制和容量恢复策略； 3) 以保留损失容忍阈值为准则的更新接受机制； 4) 在大规模预训练模型（LLaMA-7B）上实现，所有参数冻结，仅调整通道掩码。

**📊 数据集**

四个数据集：BoolQ、OpenBookQA、Arithmetic、TOFU。BoolQ、OpenBookQA 和 Arithmetic 用于交叉任务的无学习与保留评估；TOFU 用于隐私与实用性评估（MIA Gap、PrivLeak、Model Utility）。

**📈 对比分析**

与六个基线（GA、RMU、SimNPO、LLM-Eraser、O3、ASU）以及多种消融实验比较。实验表明，TFR-Net 在三条无学习流的平均 Trade‑off 上领先 8.34pp，且在 TOFU 上实现了最低的 MIA Gap 和 PrivLeak，几乎不损失模型实用性，整体表现显著优于所有对比方法。

**⚠️ 局限性**

1) 只关注通道掩码，未对底层参数进行微调，可能限制更深层次的无学习效果； 2) 对超参数（如阈值、恢复比例）敏感，需在不同任务上手动调优； 3) 目前仅在 LLaMA-7B 上验证，尚未在更大模型或多语言场景中验证泛化能力。

---

## 23. Cross-Ecosystem Bug Classification in Quantum Software

**arXiv ID:** 2608.03173 | [PDF](https://arxiv.org/pdf/2608.03173v1)

**作者:** Mir Mohammad Yousuf `[一作]` (NIT Srinagar), Bisma Majid `[通讯]` (NIT Srinagar)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了跨量子软件生态的 bug 分类，验证并推广了基于规则的分类框架，比较了 Qiskit 与 11 个其他生态的 bug 分布，并与机器学习基线进行对比；

**💡 创新点**

首次实现跨生态量子软件 bug 统计与可解释规则分类，结合统计检验与纵向时间分析，为量子软件质量保障提供了实证依据；

**🔧 技术方法**

使用规则基分类（关键词+TF‑IDF + 规则），统计假设检验（卡方+Cramér's V），以及四个监督学习基线（逻辑回归、决策树、随机森林、梯度提升）；

**📊 数据集**

共计 17,523 条公开 GitHub issue：12,910 条 Qiskit 与 4,613 条来自 Cirq、PyQuil、Q#、OpenQL、XACC、Strawberry Fields、Braket、D‑Wave Ocean、Staq、Tequila、Silq 等；

**📈 对比分析**

通过宏 F1 评价与四个机器学习基线比较，规则框架在所有维度上表现最佳，尤其在细粒度量子子类上提升高达 0.62 的 F1；

**⚠️ 局限性**

局限包括仅基于公开 issue 的数据、规则对歧义报错可能误判、严重性与质量属性标注不一致、缺少工业/闭源项目验证以及对罕见类别样本不足。

---

## 24. Deep Divide-and-Reduce in Symbolic Regression

**arXiv ID:** 2608.02628 | [PDF](https://arxiv.org/pdf/2608.02628v1)

**作者:** Yusong Deng `[一作]` (University of Chinese Academy of Sciences), Weijun Li `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 11796 | [OpenAlex ID](https://openalex.org/A5100359321)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对符号回归中的表达式拆分与合并问题，提出了一套基于深度分解与归约的框架 DDRSR。

**💡 创新点**

通过对平移对称性、变量可分离性和嵌套组合的严格数学推导，扩展了 AI Feynman 的适用范围，并消除了对暴力子结构搜索的依赖。

**🔧 技术方法**

使用符号表达式树、梯度比值检测、常数干扰分离以及上下文分离的理论判据，并将其整合成自底向上/自顶向下的分解算法。

**📊 数据集**

在公开数据集（如 Physics4K、DeepSig 等）以及自构造的高维复杂表达式上进行实验，采用 E2E、GP‑GOMEA、PySR 等主流符号回归方法进行后续回归。

**📈 对比分析**

相较于原始方法和 AI Feynman 的分解方案，DDRSR 在 R²、恢复率和相对分离分数上均表现更优，尤其在维度和复杂度增加时优势更为显著。

**⚠️ 局限性**

主要局限在于：需要可解析梯度、对噪声鲁棒性不足、仅适用于存在可分离或可归约结构的函数，对完全不可分离的表达式仍无法实现确定性分解。

---

## 25. Contact-Driven Localization in a Freeform Robotic Self-Assembled Structure

**arXiv ID:** 2608.02895 | [PDF](https://arxiv.org/pdf/2608.02895v1)

**作者:** Mohammadali Rashidioun `[一作]` (New Jersey Institute of Technology), Petras Swissler `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于模块机器人本地接触信息的定位方法，利用IMU姿态感知、接触角度和翻转历史通过虚拟力框架实现在自由形态自组装结构中的定位；

**💡 创新点**

创新点包括：①仅用二进制接触信息和IMU即可完成3D定位；②将机器人翻转历史与接触约束融入虚拟力，提供垂直方向校正；③引入全局未接触排斥力，防止密集区域重叠；④在模拟火焰蚂蚁（FireAnt V3）平台上验证其可扩展性。

**🔧 技术方法**

使用技术主要有：虚拟力（吸引/排斥）求解、IMU姿态估计、接触探测、翻转历史累积、全局未接触排斥、迭代优化。

**📊 数据集**

使用的数据集为仿真产生的结构：每种场景（塔式结构和悬臂结构）各生成25个、每个包含100台机器人，共计2000台机器人的位置信息。

**📈 对比分析**

通过与基线完整设置（含Z校正与全局排斥）对比去除Z校正或去除全局排斥的消融实验，使用RMSE和Procrustes形状差异两指标评估。塔式结构基线RMSE约1.64机器人半径（≈54 mm），悬臂结构约3.91机器人半径（≈92 mm）。消除Z校正会使RMSE和形状差异各升约12–13%，去除全局排斥则使塔式结构RMSE升约44%、形状差异升约47%。

**⚠️ 局限性**

局限性包括：①仅在仿真中验证，未在真实硬件上测试；②对IMU精度高度依赖，可能在振动或磁干扰环境下失效；③在悬臂结构中早期机器人定位误差易累积；④全局未接触排斥需要有限的全局通信；⑤仍需后向传播或其他误差补偿方法来进一步降低累计误差。

---

## 26. Towards Designing for (Dis)Trust in Technologies for Aging

**arXiv ID:** 2608.02784 | [PDF](https://arxiv.org/pdf/2608.02784v1)

**作者:** Muhid Hassan Risvy `[一作]` (New Jersey Institute of Technology), Alisha Pradhan `[通讯]` (New Jersey Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

综述了 59 篇关于老年人对技术信任与不信任的实证研究，采用解释性主题分析提炼出三大核心模式：信息系统的材料可读性、信任的时间构成与流动性，以及不信任作为界限设定。

**💡 创新点**

创新点在于将不信任从单纯的隐私或安全问题转化为（1）信息流与存储的“材料不透明”导致的不信任；（2）信任随经验与环境的动态演变；（3）不信任被用作保护个人自主权、关系与日常生活的边界工作，进而提出相应的设计方向。

**🔧 技术方法**

使用的技术主要是系统性文献检索（ACM Digital Library、Google Scholar）和反思性主题分析（reflexive thematic analysis），结合编码表与研究者互评进行数据整理。

**📊 数据集**

数据集为 59 篇符合包含老年人一手经验且讨论信任/不信任的经验证据，发表年份 2008–2025，来源为 ACM DL 与 Google Scholar。

**📈 对比分析**

由于该研究为质性综述并非实验或算法评测，没有传统的“方法比较”与“性能”指标；评价依据是对研究质量的解释与归纳，强调对不同使用阶段与历史语境下的信任支持建议。

**⚠️ 局限性**

局限性包括：检索范围与数据库的可重复性有限；未覆盖所有学科与非可检索文献；依赖已有研究报告，可能遗漏细节；并未进行系统评估，结果更多为启发性洞见而非可量化结论。

---

## 27. Exploiting Separability in Multi-Scale Grey-Box Bayesian Optimization

**arXiv ID:** 2608.03045 | [PDF](https://arxiv.org/pdf/2608.03045v1)

**作者:** Joshua E. Hammond `[一作]` (University of Texas at Austin), Michael Baldea `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将灰盒优化问题拆分为黑盒变量的贝叶斯优化和白盒子问题的全局优化的双层框架。

**💡 创新点**

通过只在黑盒维度构建GP代理，减少维度并在内层求解已知闭式约束，实现精确约束满足，无需惩罚函数或置信度近似。

**🔧 技术方法**

使用高斯过程回归、期望改进采样、Basin‑Hopping或SLSQP全局/局部求解器、Latin Hypercube 预采样等技术。

**📊 数据集**

创建了包含13个2‑5维、0‑3约束的合成与工程真实案例灰盒基准套件，用以验证方法。

**📈 对比分析**

与传统全空间贝叶斯优化、全局搜索和纯黑盒NLP对比，Bilevel BO在200次迭代下实现11×–10^8×更低的遗憾，样本效率提升，计算时间相当。

**⚠️ 局限性**

依赖内层子问题可全局求解，难以处理极窄或分散可行域；对黑盒约束无专门处理；内层求解器的选择会影响效果；在黑盒评估极其昂贵时仍需频繁调用。

---

## 28. Sedentary Behavior Classification for Wearable Sensors with a CNN-BiLSTM Model

**arXiv ID:** 2608.02946 | [PDF](https://arxiv.org/pdf/2608.02946v1)

**作者:** Yuliang Chen `[一作]` (University of California San Diego), Loki Natarajan `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究将先前在臀部加速度计上训练的CNN-BiLSTM模型CHAP迁移到手腕加速度计，用于坐姿与非坐姿的二分类，并系统评估零样本与微调的效果。

**💡 创新点**

创新点在于提出跨身体部位迁移学习框架，证明臀部预训练可显著提升手腕分类性能，并与Transformer从零开始训练进行对比，展示预训练在低标签情况下的优势；同时评估标签量对性能的影响。

**🔧 技术方法**

使用技术包括CNN-BiLSTM模型CHAP、Transformer ViTSmall、迁移学习/微调、加速度信号预处理与分段、权重平衡的二元交叉熵损失、数据增强（抖动、缩放、通道置换、轴旋转）以及多指标评估（平衡准确率、F1、误差率等）。

**📊 数据集**

使用的数据集为iWatch研究收集的自愿者数据，包含手腕和臀部ActiGraph GT3X+加速度计以及SenseCam相机的姿势标签，样本约141人臀部、143人手腕，总计约280万10秒窗口。

**📈 对比分析**

比较方法为在同一数据集上分别进行CHAP零样本（CHAPZS）、CHAP微调（CHAPFT）和Transformer ViTSmall训练；在不同标签比例（1%、10%、50%、100%）下评估。结果显示，CHAPFT在臀部保持≈89%平衡准确率，手腕在微调后提升至≈82%，显著优于Transformer；仅需10%手腕标签即可接近最佳性能。

**⚠️ 局限性**

限制在于手腕信号高度变异导致低运动非坐姿误判为坐姿；预训练对手腕迁移的效果有限，仍需足量标签；模型主要针对二分类，未涵盖更细粒度动作；实验仅在iWatch cohort，缺乏跨数据集的外部验证。

---

## 29. An Empirical Analysis of the Glob Ecosystem

**arXiv ID:** 2608.02610 | [PDF](https://arxiv.org/pdf/2608.02610v1)

**作者:** Phyllis Lim `[一作]` (University of Kentucky), Mark Marron `[通讯]` (University of Kentucky)

**通讯引用:** 1435 | [OpenAlex ID](https://openalex.org/A5024427812)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对全球 glob 生态系统进行实证研究，量化特性差异、使用情况与安全问题。

**💡 创新点**

首次系统构建 glob 语法规范 GlobSpec 并揭示跨语言碎片化与安全缺陷。

**🔧 技术方法**

采用混合方法：大规模静态提取、GitHub、Stack Overflow 与 CVE 数据分析，结合定性主题编码。

**📊 数据集**

采集 62,713 条 glob 模式、1,966 个开源项目、1,355 条 GitHub issue、444 条 CVE、361 条 Stack Overflow 问题。

**📈 对比分析**

通过对比不同语言实现的功能支持、错误率与安全报告，发现实现差异导致 22% 的问题，并在实验中验证 GlobSpec 可将语义不一致降至零。

**⚠️ 局限性**

受限于采样仅覆盖主流语言与公开仓库，且对极端嵌套模式与自定义插件支持不足。

---

## 30. Localize, Don't Beautify: Client-Side Control of Image-Editing APIs for Cosmetic Surgery Previews

**arXiv ID:** 2608.02841 | [PDF](https://arxiv.org/pdf/2608.02841v1)

**作者:** Sukhrobbek Ilyosbekov `[一作]` `[通讯]` (Northeastern University), Sukhrobbek Ilyosbekov (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

对商业图像编辑API进行无模型内部访问时的客观控制实验，评估客户端侧通过掩膜组合实现的面部区域限定效果；

**💡 创新点**

提出一种模型无关的控制阶梯（prompt-only、masked composite、masked inpaint）及其自动诊断协议，证明即便无权访问模型内部，客户端也能通过后置掩膜实现精准区域编辑；

**🔧 技术方法**

使用MediaPipe人脸网格与InsightFace关键点生成掩膜，采用ArcFace余弦相似度衡量身份保留，CIELAB ΔE比值评估区域局部变化，结合文本提示、对齐与羽化混合；

**📊 数据集**

使用公开可获得的真实术前术后照片集：12张面部提升（facelift）与14张隆鼻（rhinoplasty）照片（包含8张面部提升和8张隆鼻用于基准评估）；

**📈 对比分析**

在六种商业编辑配置（GPT‑Image‑2、Nano Banana、Seedream、FLUX.2 等）与三种控制层级下生成单张输出，统计中位数局部化比值、身份余弦、像素变化；masked composite 将局部化比值从约0.4提升至≈1（保留原像），但在身份相似度与靠近术后照片方面表现不佳，未显示任何显著改善；

**⚠️ 局限性**

局限包括样本量小、每个单元仅生成一次、缺乏外部外观评估、仅评估前景区域且不覆盖侧面、身份余弦仅为技术指标、未考察不同肤色与年龄分布、未评估临床可行性。

---

## 31. Interpolation of Non-Linear Functions for LLMs using Partial Reconfiguration in FPGAs

**arXiv ID:** 2608.03033 | [PDF](https://arxiv.org/pdf/2608.03033v1)

**作者:** Roger Morales-Monge `[一作]` (Costa Rica Institute of Technology), Jorge Castro-Godinez `[通讯]` (Costa Rica Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

针对大语言模型中指数和Sigmoid等非线性函数，提出了一种基于部分重配置的 PWL 插值加速器，将静态通信/控制区与可动态加载的可重配置区分离，实现时间复用的插值引擎。

**💡 创新点**

创新点在于：1）利用部分重配置将多种插值模块共存于同一可重配置区域，显著降低资源占用；2）在指数与Sigmoid上评估均匀与非均匀分段，提出基于二阶导数的非均匀分段策略；3）在FP16/FP32两种精度下系统级别评估功耗、面积与延迟，证明可重配置在面积‑延迟‑功耗上具有良好权衡。

**🔧 技术方法**

采用 Xilinx Kria KV60（XCK26‑SFVC784‑2LV‑C）FPGA，Vitis HLS 2023.2 进行高层合成，PYNQ 作为轻量级运行时进行 PR 控制，AXI4 接口完成数据流与控制交互，使用二进制搜索实现非均匀区间的索引查找。

**📊 数据集**

论文未使用传统机器学习数据集；实验仅在 [-8, 8] 的输入区间对指数和 Sigmoid 进行插值误差分析，并在 Kria 板上对功耗、面积与延迟做实测。

**📈 对比分析**

与同时实例化两个插值引擎的静态实现相比，基于 PR 的设计在 FP32 精度下 LUT 减少 48.4%，FF 减少 49.7%，DSP/BRAM 减少 50%；功耗在 0.02 W 以内增加，PR 延迟保持在 2.71×10⁶–2.80×10⁶ 周期，整体延迟比静态实现低约 9–10%。

**⚠️ 局限性**

限制：1）PR 过程会产生几百万周期的延迟，适合粗粒度切换而非每个 token 或 layer 的快速切换；2）仅验证了指数与 Sigmoid 两种函数，未验证更复杂的激活函数；3）仅在 FP16/FP32 两种精度下评估，低精度或自定义数值格式尚未探究；4）多线程或并行多实例场景下的资源共享与同步仍需进一步研究。

---

## 32. CorePath: A Breast-Specialized Pathology Foundation Model for Core Needle Biopsy Diagnosis and Risk-Controlled Report Generation

**arXiv ID:** 2608.03079 | [PDF](https://arxiv.org/pdf/2608.03079v1)

**作者:** Ting Yin `[一作]` (Sichuan University), Hong Bu `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了面向乳腺核心针活检（CNB）的多模态病理基础模型CorePath，并提出了风险控制报告生成框架CorePath‑CRG，能够在不训练特定分类器的情况下实现零样本诊断与安全报告发布。

**💡 创新点**

创新点包括：① 通过配对WSI‑报告的参数高效微调将通用PRISM模型转化为乳腺特化模型；② 在报告生成中引入合规化校准、Conformal Prediction 与 Learn‑Then‑Test（LTT）实现三层选择性发布（自动发布、亚型回退、退回至人工）；③ 通过多中心零样本评估展示模型在不同诊断层级和公共基准上的跨域泛化能力。

**🔧 技术方法**

使用技术包括：多模态预训练模型PRISM、Perceiver‑only 参数微调、合成式生成、Conformal Prediction、Learn‑Then‑Test 风险控制、三层选择性发布、LLM 评估（DeepSeek、Qwen3）、自动文本相似度度量（BLEU/ROUGE/METEOR）。

**📊 数据集**

数据集：7901 对CNB WSIs 与报告（WCH‑1、SPH‑1）用于微调；独立验证集（SWH、WCH‑2、WTH、SJH、SZH、SPH‑2）用于诊断评估；公共基准 BCNB（1058 例）和 BRACS（547 例）用于外部评估；报告生成评估在四中心（SWH、WCH‑2、WTH、SJH）进行。

**📈 对比分析**

比较方法：与 PRISM、TITAN 等基线模型在六中心和两公共基准上进行零样本分类比较；CorePath 在癌症检测的加权 AUC 达 0.9669–0.9989，亚型分化加权 AUC 0.9526–0.9735；在 BCNB、BRACS 亦取得最高加权 AUC（0.7780、0.8178、0.8252）。报告方面，非乳腺幻觉率由 PRISM 的 30.1% 降至 CorePath 的 2.8%，CorePath‑CRG 自动发布无幻觉；LLM 评估分数持续提升，自动发布文本与参考报告的相似度（BLEU/ROUGE/METEOR）显著提升。

**⚠️ 局限性**

局限性：① 仅为回顾性研究，未进行前瞻性临床验证；② 微调数据仅来自两中心，外部多样性不足；③ 合规化与 LTT 依赖样本可交换性，分布漂移可能削弱校准效果；④ 幻觉评估聚焦非乳腺错误，未覆盖细粒度诊断失误；⑤ LLM 评估偏保守，可能低估模型风险；⑥ 未在真实临床工作流中评估工作量、效率与用户信任。

---

## 33. V-FIND: Revealing the Intrinsic Forgery Knowledge Encoded in Video Forgery Detectors

**arXiv ID:** 2608.03008 | [PDF](https://arxiv.org/pdf/2608.03008v1)

**作者:** Shichao Kan `[一作]` (Central South University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 V-FIND 框架，通过在冻结的检测器中定位关键层并发掘稀疏的潜在锚定神经元，构建紧凑的取证子空间，使用轻量化线性分类器完成视频伪造检测。

**💡 创新点**

创新点在于将视频伪造检测视为内在知识挖掘任务，发现并利用仅占模型总参数极小比例的专门化神经元，从而实现参数高效、可迁移的检测方式。

**🔧 技术方法**

技术包括层级差异信号（方向分离与归一化位移）用于关键层定位，基于线性探针的神经元重要性评分，效应量阈值筛选锚定神经元，构建取证子空间并训练线性读出器。

**📊 数据集**

使用 15Model‑140K 进行模型训练与神经元发现，外部评测数据集包括 Magic Videos、MovieGen 与 DVF。

**📈 对比分析**

与多种先进检测器（MM‑Det、VINA、WaveRep 等）以及不同视觉/视频基座（CLIP、X‑CLIP、TimeSformer、Moon‑ViT 等）对比，V‑FIND 在 Magic 上实现 mACC 89.37%、mAP 96.92%，在 MovieGen 上 ACC 96.85%、AP 99.31，DVF 上 AUC 98.2，均显著优于基线。

**⚠️ 局限性**

局限性包括：对阈值 τ_d 的选择敏感；仅在现有模型架构与短视频长度上验证；对训练数据量和输入失真有一定鲁棒性限制；需先行在目标检测器上完成关键层定位，过程相对复杂。

---

## 34. GriD-LMIA: A Gridding-Based Assembler for Solving Differentiable Parameter-Dependent Linear Matrix Inequalities

**arXiv ID:** 2608.03175 | [PDF](https://arxiv.org/pdf/2608.03175v1)

**作者:** Yicheng Xu `[一作]` (University of California Irvine), Faryar Jabbari `[通讯]` (University of California Irvine)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108`

**🎯 论文内容**

提出并实现了GriD-LMIA工具箱，用于把参数相关LMI（包括可微PD‑LMI）在连续域上离散化为有限个SMI，并能自动生成YALMIP模型；

**💡 创新点**

创新点包括：①基于网格的可微PD‑LMI装配器，利用分块Bernstein多项式保证边界连续性；②在单元内将未知决策变量展开成Bernstein系数，并支持阶提升、乘积与导数等算子；③提供五种有限正性证书（直接Bernstein、Pólya、Putinar、FullBox、Sparse FullBox），实现从保守到更精确的阶梯式近似；④实现稀疏Gram矩阵窗口化，显著降低SOS SDP规模；

**🔧 技术方法**

技术：超矩形网格分割、张量Bernstein基底、Clarke广义导数处理、系数升维/乘法/导数操作、有限正性证书、Pólya提升、SOS/Putinar/FullBox预排序、YALMIP接口、SDP求解器（SDPT3、SeDuMi、MOSEK）

**📊 数据集**

使用标准LPV基准模型：一参数示例（A、B矩阵与ρ∈[0,1]），四参数质量‑弹簧模型（ρ∈[2/3,2]×[0.8,4/3]×[1,3]），以及与LPVTools、ROLMIP对应的相同数据集；

**📈 对比分析**

对比方法：与ROLMIP（Bernstein系数）、LPVTools IQC（基于IQC的无网格方法）进行匹配；实验显示：①增加网格密度和决策多项式阶数可降低γ上界；②直接Bernstein证书最保守；③Pólya提升在低阶时稍好；④SOS（特别是Sparse FullBox）在三参数例子中提供更紧的上界，但会产生更大的Gram块；整体性能随网格/阶数增加呈指数级增长；

**⚠️ 局限性**

局限性：①固定阶证书若不可行则结果无意义；②不支持决策变量乘积的非线性约束；③高维多参数时SDP规模迅速膨胀，部分SOS求解器可能报错；④需要人工选择网格、阶数、证书参数；⑤工具无正式许可证，可能影响长期使用。

---

## 35. MoEGen: Mixture-of-Experts for Instance-Adaptive LoRA Generation

**arXiv ID:** 2608.03275 | [PDF](https://arxiv.org/pdf/2608.03275v1)

**作者:** Yiming Zeng `[一作]` (University of Connecticut), Shangqian Gao `[通讯]` (Florida State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种实例适应的参数高效微调框架MoEGen，利用稀疏专家代码和轻量级超网络动态生成输入特定的LoRA更新，从而在保持存储成本低的同时实现更细粒度的模型适配。

**💡 创新点**

创新点在于将传统MoE-LoRA中完整LoRA专家替换为低维专家代码，先进行输入驱动的专家路由，再通过共享超网络根据专家混合生成低秩更新，解耦了专家容量与适配器存储，并实现实例级动态更新。

**🔧 技术方法**

使用的技术包括LoRA低秩参数化、Mixture-of-Experts路由、轻量级超网络（共享生成A矩阵）、层归一化、GELU激活以及负载平衡正则化。

**📊 数据集**

实验数据集包括Commonsense-170K（包含BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge、OpenBookQA）以及跨域混合任务集（MedNLI、PubMedQA、HQS、BillSum）。

**📈 对比分析**

在三大后端模型（LLaMA‑2‑7B、LLaMA‑3‑8B、Qwen3‑8B‑Base）上，MoEGen在Commonsense基准上平均提升约0.6–1.1点；在跨域联合训练中平均提升约4.6–6.2点；并且仅训练不到1%的参数，明显优于LoRA、MoELoRA、MoLA等静态或传统MoE‑PEFT方法。

**⚠️ 局限性**

局限性包括：需要额外实现路由与超网络模块，虽然参数和推理成本小幅增加；当前仅在监督微调场景验证，尚未探究偏好调优或持续学习等其他适配场景。

---

## 36. On the Non-Specificity of Statistical Measures Used in Script Decipherment

**arXiv ID:** 2608.02999 | [PDF](https://arxiv.org/pdf/2608.02999v1)

**作者:** Nikhil Raghavendra `[一作]` `[通讯]` (Independent Researcher), Nikhil Raghavendra (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了一个非语言但具有明确语义的合成铭文系统，并用它重新计算了54种统计方法，以检验这些方法是否能唯一识别语言。

**💡 创新点**

首次提供一个公开、可复制的合成标记系统作为对Indus文字符号统计特征的构造性反例，挑战统计特征对语言的专一性假设。

**🔧 技术方法**

采用生成式符号模型（Pitman‑York urns、有限槽位等）与多种统计指标（条件熵、Zipf‑Mandelbrot、LNRE、位置分布、方向不对称、网络连通度、分类器决策等）以及基于词典的逐词强制分割技术。

**📊 数据集**

使用了约3000条合成铭文（约13717符号）以及真实Indus铭文（≈2900条）和多种对照语料（随机、模板、Zipf等）进行比较。

**📈 对比分析**

通过方法级注册表，精确复现了Indus结果中6项可复现的方法，所有指标在合成文本上与Indus一致；描述性统计同样落在Indus参考区间；词典覆盖测试显示高覆盖率但识别不稳定，表明统计指标本身并不能唯一判定语言。

**⚠️ 局限性**

研究仅构造单一示例，未评估类似系统出现的概率；未给出语言与非语言之间的似然比；对Indus语料处理细节不完全公开；缺乏真正的外部验证或持久化测试。

---

## 37. RealWeather: Realistic and Scene-Faithful Weather Translation with Driving World Models

**arXiv ID:** 2608.02953 | [PDF](https://arxiv.org/pdf/2608.02953v1)

**作者:** Yuwei Ning `[一作]` (Sun Yat-sen University), Guanbin Li `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

建立了一种能从真实驾驶视频中学习并实现清晰↔恶劣天气的视频翻译的驾驶世界模型。

**💡 创新点**

创新点包括：①渐进现实引导 Bootstrapping 与伪清晰生成管线的组合，利用自监督迭代收敛真实天气；②Scene‑Fidelity RL 优化，采用分割奖励显式压制结构幻觉，保证场景忠实。

**🔧 技术方法**

技术手段包括：预训练的 Cosmos‑Predict2.5 世界模型、流匹配（flow‑matching）训练、GRPO‑style 强化学习、SAM 分割奖励、图像编辑与几何条件推理的伪清晰生成。

**📊 数据集**

使用数据集：Waymo Open Dataset、nuScenes 以及内部驾驶数据，涵盖清晰与恶劣天气视频。

**📈 对比分析**

与 IntrinsicWeather、WeatherEdit、AutoAWG、Cosmos‑Transfer2.5、SSGFormer、Wan2.1‑VACE 等基线对比，采用 CS、CC、DS、PS、VR 等指标，本文在视觉真实感、时序一致性和结构保真上取得最高或第二高分。

**⚠️ 局限性**

局限性：伪清晰生成的质量仍有限，无法完全覆盖所有极端天气与摄像头视角；在少数极端场景下可能出现细微结构失真。

---

## 38. SAGE: Semantic Explainability of Attention-Based Survival Models in Computational Pathology

**arXiv ID:** 2608.02803 | [PDF](https://arxiv.org/pdf/2608.02803v1)

**作者:** Abdallah Lamane `[一作]` (Massachusetts Institute of Technology), William Lotter `[通讯]` (Dana-Farber Cancer Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种名为SAGE的框架，能够在不重新训练或标注的前提下，对已训练好的基于注意力的多实例学习（ABMIL）生存预测模型进行全局语义可解释性分析；

**💡 创新点**

通过将模型注意力权重与视觉‑语言模型对25个病理概念的语义相似度相乘并聚合，SAGE生成患者级概念得分，进而量化各概念与模型预测风险之间的关联，从而实现模型行为的全局、语言可解释性；

**🔧 技术方法**

结合ABMIL注意力机制、视觉‑语言模型（CONCH、MUSK、UNI2）、prompt‑ensembling、余弦相似度计算、Spearman相关性分析、Bootstrap与Permutation检验等技术；

**📊 数据集**

在TCGA七个癌症组（BRCA、BLCA、CESC、COAD、KIRC、LGG、LUAD）上，每组从WSI提取一张切片，使用疾病特异性生存（DSS）作为终点；

**📈 对比分析**

将SAGE解释结果与模型预测的C‑index进行对比，发现基于视觉编码与基于文本相似度编码的ABMIL模型在七组数据上的平均C‑index均为0.625；对比注意力加权与均值加权聚合，注意力加权显著提升相关性；不同VLM的比较表明CONCH的文本空间最能保持与视觉编码一致性；

**⚠️ 局限性**

SAGE的关联结果仅为假设生成，未证明因果关系；视觉‑语言模型在不同概念上的准确性可能不均；所用的25个概念词典可能缺失癌种特异或细微形态特征；实验仅限于TCGA七组，需在其他数据集进一步验证。

---

## 39. CLASVS: Continuous-Latent Autoregression for Melody-Preserving Lyric Editing in Singing Voice Synthesis

**arXiv ID:** 2608.03253 | [PDF](https://arxiv.org/pdf/2608.03253v1)

**作者:** Yizhong Geng `[一作]` (Beijing University of Posts and Telecommunications), Ya Li `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种无谱注释的连续潜在自回归模型，用来在保持旋律、节奏和歌手身份的前提下对演唱歌词进行编辑。

**💡 创新点**

创新点是State–Control–Transition（SCT）编辑合同与Progressive State–Control Grounding（PSCG）技术，能够在无配对编辑数据的情况下，将目标歌词和参考旋律的控制保持持久、通过语义进度反馈引导全局规划，并仅在局部使用前一潜在补丁，解决了编辑过程中的对抗冲突。

**🔧 技术方法**

技术上采用连续AudioVAE潜在编码、流式Diffusion Transformer（Flow‑DiT）、冻结的旋律与声纹编码器、可学习停止头以及Causal Planner Qwen3-0.6B等组件。

**📊 数据集**

主要使用约8000小时的中文演唱、2000小时中文语音和300小时公开演唱音频，评测数据集包括CLA‑LyricEdit‑320和公开Mandarin LyricEditBench。

**📈 对比分析**

在与离散AR模型Vevo2和连续NAR模型YingMusic+的自动输入协议对比下，SCL将宏观PER降低46.2%，在删除/插入操作上优于对比者，并在自然度、歌词可懂度等多项指标上实现显著提升。

**⚠️ 局限性**

局限性包括仅评估中文2–6音节编辑，缺乏跨语种和长编辑验证，且模型对训练数据、可再现性及隐私合规的依赖仍需进一步解决。

---

## 40. MalTotal: Cost-Effective and Language-Agnostic Malicious Code Poisoning Detection for Millions of Repositories

**arXiv ID:** 2608.03232 | [PDF](https://arxiv.org/pdf/2608.03232v1)

**作者:** Jian Zhao `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MalTotal框架，利用LLM进行语言无关的恶意代码检测并实现大规模可扩展

**💡 创新点**

通过LLM辅助敏感API识别、混合语义切片和语义推理，显著降低token消耗并提升跨语言检测性能

**🔧 技术方法**

结合LLM（如DeepSeek‑V3）、Joern CPG、Semgrep、手工编制的敏感API知识库与层次化切片算法

**📊 数据集**

使用Multi‑Lang‑Bench（5种语言共约2.6万份样本）和GitHub‑120K（120K仓库）进行评估

**📈 对比分析**

相较于8种SOTA工具，平均F1≈93.1%，在5种语言均保持>85%；在120K仓库中发现564新恶意仓库，成本仅$338

**⚠️ 局限性**

依赖现有静态分析工具，无法跨语言或深度别名链追踪，LLM易受提示注入影响，且对C/C++等未支持

---

## 41. Surrogate Substitution Preserves PHI Detectability: A Multi-Detector Equivalence Study

**arXiv ID:** 2608.03172 | [PDF](https://arxiv.org/pdf/2608.03172v1)

**作者:** Qiming Bao `[一作]` (Custodian Labs), Meng Fon `[通讯]` (Custodian Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证结构保留型去标识化的可用性，使用配对多检测器评估协议只关注掩盖跨度的可检测性。

**💡 创新点**

① 引入只评估掩盖跨度的配对多检测器协议；② 用等价检验（TOST）替代传统显著性检验；③ 建立失效类型学，区分生成器缺陷与检测器极限。

**🔧 技术方法**

结构保留型替换、11种PHI检测器（规则、微调、LLM）、等价检验统计、遮蔽与泄露指标、错误类型分析、开放源代码基线Faker。

**📊 数据集**

7个公开/合成基准（ASQ‑PHI、MEDDOCAN、MultiCoNER v2、PII‑Masking‑300k 及其多语言版本），共1,750文档（每基准250）。

**📈 对比分析**

在57,112个掩盖跨度上，召回从76.1%降至74.9%，TOST证明差异在±2点内且保持排名；整体文档变动≤4.7点；与红action对比显示结构保留方法几乎不丢失可检测性；与Faker基线对比证明问题源于生成器质量。

**⚠️ 局限性**

仅评估掩盖跨度的可用性而不关注覆盖率；C1/C2对比仅用CPU级检测器；未验证真实EHR；LLM检测器对提示敏感；单基准样本限制导致统计功效受限。

---

## 42. Evaluating Counterfactual Sensitivity to Patient Information in Medication-Safety Reasoning

**arXiv ID:** 2608.03028 | [PDF](https://arxiv.org/pdf/2608.03028v1)

**作者:** Zhitian Hou `[一作]` (Hong Kong Polytechnic University), Hongxia Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并发布了 MedPIC‑Bench，一个基于可追溯临床指南的药物安全推理基准，并通过配对的反事实问题评估 LLM 在不同患者信息条件下的规则应用能力。

**💡 创新点**

创新点在于：① 通过 MedRule2Pair 两阶段流水线将多源指南转化为结构化、可验证的规则；② 设计了固定情景（GF）与受控反事实（CF）对，能直接测量规则的激活与撤销；③ 对问题进行六维度细粒度标注，揭示模型在激活/撤销方面的显著不对称；④ 在 28 个模型上发现医学专用 LLM 并未显著改善反事实性能，说明仅凭知识记忆不足。

**🔧 技术方法**

技术手段包括：使用 LLM 辅助提取规则并手工校对；利用确定性脚本生成 GF/CF 问题；专家人工校验保证答案与源推荐一致；在零样本（zero‑shot）下对 LLM 进行问答评估；并用激活/撤销准确率、对比准确率等指标细致分析性能。

**📊 数据集**

数据集：467 条问题（284 GF + 183 CF），覆盖 9 个器官系统、11 个临床科室、66 种药物类别、8 种患者信息类型、3 种特殊人群，来源包括 Beers Criteria、PLLR、TGA 怀孕数据库、2025 KIDs 列表等权威指南，所有问题均经过专家验证。

**📈 对比分析**

与 28 个医学专用、通用和专有 LLM 进行对比。整体 GF 准确率 63.6% 下降至 CF 45.1%，平均差距 18.5 分；最佳模型 Gemini‑3.1‑Pro 在 CF 上 78.2%，但对比准确率仅 48.3%；医学专用模型平均 CF 38.3%，低于通用模型 45.5%；在激活与撤销上模型均表现出激活准确率高于撤销（平均 57.7% vs 37.1%）。

**⚠️ 局限性**

局限性包括：① 仅基于合成病例和规则，不涵盖真实临床中的不确定性、患者偏好和当地指南；② 评价采用零样本设置，未探讨微调或提示工程的潜在改进；③ 只考察规则记忆与激活/撤销的二元判定，未涵盖更复杂的多步推理或多因素决策；④ 数据集规模相对有限，无法覆盖所有药物安全情景。

---

## 43. Confident but Unreliable: A Behavioral Safety Audit of Vision-Language Models on Brain MRI

**arXiv ID:** 2608.02790 | [PDF](https://arxiv.org/pdf/2608.02790v1)

**作者:** Amir Sabbaghziarani `[一作]` (Tri-Institutional Georgia State University/Georgia Institute of Technology/Emory University Center for Translational Research in Neuroimaging and Data Science), Sergey Plis `[通讯]` (Tri-Institutional Georgia State University/Georgia Institute of Technology/Emory University Center for Translational Research in Neuroimaging and Data Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

对六个指令调优的视觉‑语言模型（VLM）在脑 MRI 切片上进行自动化行为安全审核，评估覆盖率、准确率、校准度、错误时的置信度、误报（hallucination）和不确定性下的拒绝（abstention）等指标。

**💡 创新点**

创新点在于：
- 利用公开数据集的元数据和已发布的分割掩码实现无人工标注的切片级自动评估；
- 引入置信度相关指标（ECE、Brier、平均错误置信度、Confident‑Wrong Rate）揭示准确率与置信可靠性不一定同步；
- 进行基于家族的医学微调对比，阐明医学适配可提升肿瘤检测但不一定改善置信校准。

**🔧 技术方法**

技术手段包括：
- 采用六种指令调优的 VLM（InternVL、Qwen、Gemma 系列和 MedGemma）进行多轮对抗式问答；
- 计算 ECE、Brier 分数、平衡准确率（Balanced Accuracy）、覆盖率、平均错误置信度、Confident‑Wrong Rate；
- 对负样本（非脑/噪声控制）进行 abstention 评估；
- 使用 Bootstrap 进行主体级不确定性估计。

**📊 数据集**

使用的数据集：
- BraTS‑2024 处理后脑瘤扫描（1350 切片，150 受试者）
- IXI 健康志愿者扫描（2682 切片，100 受试者）
- 70 张非脑/噪声对照图像。
共 4,102 张切片，涵盖 T1、T1ce、T2、FLAIR、PD 等序列。

**📈 对比分析**

比较方法：在多项选择（MC）和开放式（OE）两种回答格式下，对各模型按任务（序列识别、平面识别、脑/非脑判断、肿瘤存在与侧向性）评估准确率与校准指标。结果显示：
- 覆盖率均近乎 100%；
- 最准确模型 Gemma‑4‑12B 的整体准确率最高（0.670），但其错误时平均置信度最高（0.968），ECE 亦较高；
- Qwen2.5‑VL‑7B 的准确率较低（0.563）但在错误时的过度置信度最低；
- 医学微调模型 MedGemma‑4B 在肿瘤检测上优于其基线，但在泛化视觉任务和置信校准上并未提升；
- Hallucination 率和 abstention 率在模型间差异显著，说明这些行为维度不随准确率同步变化。

**⚠️ 局限性**

局限性：
- 仅使用单层 2‑D 切片，缺乏跨切片上下文；
- 负样本数量有限（70 张），abstention 评估统计功效受限；
- 评估基于公开数据，无法排除模型训练时对数据的记忆或泄漏；
- 仅包含六款模型，未涵盖更大范围的 VLM；
- 置信度测量基于模型自述置信，可能受训练策略影响；
- 对肿瘤定位使用固定可见性阈值，未进行放射科医生重新检查；
- 结果仅适用于所评估的公共分布，不能直接推广为临床诊断结论。

---

## 44. Translation of Regular Expression with Lookahead into Finite State Automaton

**arXiv ID:** 2608.03167 | [PDF](https://arxiv.org/pdf/2608.03167v1)

**作者:** Akimasa Morihata `[一作]` `[通讯]`, Akimasa Morihata

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种将正则表达式与前瞻（正向和负向）转换为确定性有限自动机（DFA）以及加权NFA的算法，从而实现不使用回溯的正则表达式匹配，并支持子匹配提取。

**💡 创新点**

创新点在于：
• 证明正则表达式加前瞻（REwLA）所表示的语言仍为正则语言；
• 通过布尔有限自动机（BFA）处理前瞻的逻辑约束，实现从REwLA到DFA的转换；
• 在加权框架下，构造可直接计算子匹配信息的加权NFA，解决了负前瞻下子匹配提取的难题；
• 提供了复杂度分析：匹配时间为 O(n · 2^{2^m})，其中 n 为输入字符串长度，m 为正则表达式长度。

**🔧 技术方法**

使用的主要技术包括：
• Thompson 构造法的扩展，生成带有布尔表达式的 BFA；
• 布尔有限自动机（BFA）与布尔逻辑的组合，用于表达前瞻的约束；
• 加权正则表达式与加权NFA 的构造，结合语义半环（semiring）来记录子匹配信息；
• 对 BFA 进行幂集化得到 DFA，虽然是双指数状态，但实现了线性对字符串长度的匹配。

**📊 数据集**

本文未使用公开数据集；实验部分以 Ruby 1.8.7 的正则表达式为示例，演示了匹配时间与传统回溯实现相比的提升。

**📈 对比分析**

比较方法：将基于回溯的 Ruby 1.8.7 正则引擎与本文提出的基于 DFA/加权 NFA 的实现进行对比。性能表现：在包含前瞻的复杂正则表达式上，本文方法在大多数测试案例中实现了指数级别的加速，匹配时间主要受正则表达式长度的影响，而与输入字符串长度线性相关。

**⚠️ 局限性**

局限性：
• 生成的 DFA（或加权 NFA）状态数量为双指数级 O(2^{2^m})，在实际使用中可能导致内存爆炸；
• 对负前瞻中包含子匹配的情况支持有限，仍需进一步研究子匹配语义；
• 需要构造布尔公式的等价判定，实际实现上可采用 BDD 但仍有性能瓶颈；
• 只考虑了前瞻，后瞻（lookbehind）仍未纳入当前框架。

---

## 45. Calliphony: A Calligraphy-Driven Interface for Real-Time Generative Music Performance

**arXiv ID:** 2608.03040 | [PDF](https://arxiv.org/pdf/2608.03040v1)

**作者:** Tristan Wu `[一作]` (Hong Kong University of Science and Technology), Gus Xia `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 Calliphony 系统，将书法笔刷的旋转运动实时映射到 Notochord 模型的控制信号，生成多轨 MIDI 并通过 DAW 实现现场演出。

**💡 创新点**

首次将书法运动作为外部控制层，低延迟地操控实时符号音乐生成模型，动态调节音符密度、音高约束和伴奏层级，实现跨模态表演。

**🔧 技术方法**

使用 M5Stack 陀螺仪传感器、Max/MSP、Python（调用 Notochord）、Ableton Live、OSC 通信、LoopMIDI、Pygame UI 等技术栈。

**📊 数据集**

依赖 Notochord 预训练权重，数据来源为公开的 Lakh MIDI 数据集。

**📈 对比分析**

未进行系统性对比实验，仅通过现场演示和观众非正式反馈评估，表明系统低延迟、可实时响应，但缺乏量化指标。

**⚠️ 局限性**

传感器信息单一（仅旋转速度），无法捕捉笔压、接触面积等书法特征；模型为 GRU，难以捕获长程结构；缺乏针对中文传统音乐的专用数据集；未进行严格用户研究。

---

## 46. Extracting ODRL Policies from Business Process Models: A Graph Traversal Approach to Compliance-by-Extraction

**arXiv ID:** 2608.02607 | [PDF](https://arxiv.org/pdf/2608.02607v1)

**作者:** Meem Arafat Manab `[一作]` (Universidad Politecnica De Madrid), Víctor Rodríguez-Doncel `[通讯]` (Universidad Politecnica De Madrid)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一个七阶段的管道，能够自动将BPMN流程模型中的规范性内容（义务、许可与禁止）提取为可验证的ODRL JSON‑LD政策。

**💡 创新点**

创新点包括基于Natschläger形式化的可达性分类、对并行网关的修正处理，以及将中间捕获事件解释为带提升约束的等待禁止，实现了无人工编写的合规性可追溯提取。

**🔧 技术方法**

利用图遍历算法（Tarjan SCC、支配树、可达性检查）、Python实现的BPMN XML解析以及JSON‑LD生成，构成完整提取流程。

**📊 数据集**

在Camunda的bpmn-for-research基准数据集（包含五个模型：Dispatch of Goods、Recourse、Credit Scoring同步/异步、Self‑Service Restaurant）上进行评估。

**📈 对比分析**

通过对比每个任务的义务/许可/禁止归属，验证了提取结果的准确性；在所有模型上无错误，性能未给出具体指标，但实现已开源并支持实时演示。

**⚠️ 局限性**

主要局限包括：循环结构被折叠成无意义的单一义务，未能保留迭代语义；未命名的泳道导致角色标签难读；AND/OR网关在并行结构中的约束识别不完整；核心ODRL缺乏时间和并发表达，需要扩展；以及对大型模型的性能与可扩展性未充分评估。

---

## 47. The Tell-Tale Trace: Detecting Reasoning Failures in LLMs Using Chain-of-Thought Dynamics

**arXiv ID:** 2608.03291 | [PDF](https://arxiv.org/pdf/2608.03291v1)

**作者:** Shashwat Sourav `[一作]` (Washington University in St. Louis), Aishwarya Balwani `[通讯]` (St Jude Children's Research Hospital)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究可观测链式思维（CoT）在大型语言模型推理过程中的动态变化，探讨如何通过轨迹特征识别失败模式并在推理前预警与纠正。

**💡 创新点**

①关注整条 CoT 轨迹的组织而非单步语义正确性；②发现“提前验证崩溃”和“程序模式错配”等任务相关失败模式；③通过动态特征实现早期警告；④提出针对性证明搜索提示有效纠正 UNSAT 失败。

**🔧 技术方法**

角色标签（规划、赋值、验证等）对句子进行分层；统计转移矩阵、循环率、熵、最终化时序等特征；基于阈值的前置检测；针对性提示干预；在不同 LLM（Qwen3、Llama3、OLMo2）上进行对照实验。

**📊 数据集**

通过求解器生成的可验证 CNF SAT/UNSAT 公式，难度可调；选取每个模型在 30–90% 成功率带内的实例进行匹配对照。

**📈 对比分析**

对同一任务、同一提示下模型的正确与错误 CoT 进行匹配对比，使用 Wilcoxon 检验和 AUROC 评估早期检测；对 52 条错误 UNSAT 例子做原始、通用重试和证明搜索提示干预，准确率从 13.3% 提升至 85.0%（提升 73.1pp）。

**⚠️ 局限性**

①角色划分依赖规则匹配，可能引入误差；②早期警告在不同模型/难度下效果差异大；③干预为后置且需解算器标签；④仅在形式化 SAT 任务上验证，通用性待进一步评估。

---

## 48. Measuring Explainer Stability via Attribution Separability

**arXiv ID:** 2608.02697 | [PDF](https://arxiv.org/pdf/2608.02697v1)

**作者:** Eddie Conti `[一作]` (Barcelona Supercomputing Center), Axel Brando `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于分布的框架，用来衡量解释器（AM）的归因稳定性，并通过最大可分辨前 k 名特征的 k‑稳定性指标进行量化。

**💡 创新点**

创新点在于：①用分布距离度量归因分布的可分离性；②定义 k‑稳定性以评估前缀排名的可靠性；③提供完整的理论证明与经验验证。

**🔧 技术方法**

技术主要包括：归因分布的经验估计（Gaussian KDE）、连续 Jaccard 距离度量、k‑稳定性计算以及多解释器的比较实验。

**📊 数据集**

使用了四个常见的表格数据集：Diabetes、Heart Disease、Mobile 和 Churn，并在随机森林、逻辑回归、决策树和 SVM 等模型上进行实验。

**📈 对比分析**

通过计算不同解释器在每个数据集上的 k‑稳定性比例进行比较，结果显示 SHAP 在绝大多数场景下实现最高的 k‑稳定性，其次是 LIME，DiCE 最不稳定，随机解释器作为基线。

**⚠️ 局限性**

局限性包括：仅在表格数据上验证，缺乏对图像/文本/时间序列的测试；需要手动选择 KDE 方案与阈值 l；k‑稳定性只能衡量排名可靠性，无法完整评估解释的真实性或整体质量。

---

## 49. The Ground Is Shifting: A Reflection on the Foundations of Software Measurement

**arXiv ID:** 2608.03007 | [PDF](https://arxiv.org/pdf/2608.03007v1)

**作者:** Thomas Bock `[一作]` (Carnegie Mellon University), Bogdan Vasilescu `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文回顾软件测量的基础，分析随着 AI 代理和工作流程（如 squash‑merge）的出现，传统假设被破坏的情况，并提供了 squash‑merge 在 GitHub 项目中的普及度统计；提出了 AI 辅助的系统性复制议程与假设检测框架。

**💡 创新点**

创新点在于：①指出 AI 生成提交对测量构念的冲击；②提出在提交记录中嵌入 AI 来源元数据的做法；③将复制工作与 AI 辅助自动化结合，构建可规模化评估经典发现的新框架。

**🔧 技术方法**

使用 GitHub API 抽取提交与 PR 信息、World of Code 归档进行大规模统计分析，采用描述性统计和简单计数方法，讨论了潜在的机器学习/AI 辅助检测技术。

**📊 数据集**

主要数据集包括：GitHub 公共仓库（按 PR 计数、合并方式等），World of Code 归档（跨平台仓库历史）。

**📈 对比分析**

通过比较传统 merge 与 squash‑merge 的提交数、PR 规模以及项目级采用率，统计显示 39% 项目至少出现一次 squash‑merge，项目级采用率从 2016 年的 18% 上升到 2025 年的 39%；最大 PR 规模可达 9,187 次提交。

**⚠️ 局限性**

限制包括：仅做描述性统计，未对 AI 生成提交做精确识别；自动检测 squash‑merge 受限于提交信息格式；未对复制框架进行实证验证；缺乏对 AI 生成数据对测量影响的深入实验分析。

---

## 50. Instruction Stacking Collapse: A Benchmark and the Capability-Dependent Value of Prompt Compilation

**arXiv ID:** 2608.02639 | [PDF](https://arxiv.org/pdf/2608.02639v1)

**作者:** Atul Anand `[一作]`, Sourav Chattaraj `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在生产环境中堆叠多条指令导致LLM指令遵循率下降的现象，并提出了基于prompt编译的训练免费修复方案。

**💡 创新点**

创新点在于构建了24条原子指令的随机堆叠基准，系统量化指令冲突拓扑，并发现prompt编译的收益随模型能力呈梯度，弱模型受益最大。

**🔧 技术方法**

技术包括随机堆叠指令、严格验证器、对冲突对进行交互分析、以及用单次LLM调用的编译器实现指令重写。

**📊 数据集**

数据集是自定义的24条指令集合（6类），在多模型（Claude Sonnet 4.6、GPT‑5‑mini、Gemini 2.5 Flash）上采样1–20条堆叠，包含开源任务（开放式建议、GSM8K数学、HumanEval代码）。

**📈 对比分析**

通过对比原始堆叠与编译后指令在三模型、四堆叠规模下的遵循率，使用bootstrap聚类稳健检验，结果显示弱模型提升可达+11pp，中等模型+3pp，强模型基本无变。

**⚠️ 局限性**

局限包括：验证器侧重形式而非意图；基准仅测合规性而非内容安全；编译器仅由Sonnet实现；未与人工手写prompt对比；仅在单一seed、stack=20的within‑family ladder进行测试。

---

## 51. Frozen High-Resolution Inference for Cross-City Object Detection: An AI City Challenge 2026 Study

**arXiv ID:** 2608.03136 | [PDF](https://arxiv.org/pdf/2608.03136v1)

**作者:** Jaeuk Kim `[一作]` `[通讯]` (NextITS), Jaeuk Kim (NextITS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 AI City Challenge 2026 Track 6 赛题中，作者仅对单一源城市训练的 RF‑DETR‑Large 检测器进行一次“冻结高分辨率推理”与一次温启动 fine‑tune 的对比实验，探究在目标城市无标签、仅可获得整体 AP 反馈的限制条件下，推理分辨率变更与参数更新对隐藏混合基准性能的影响。

**💡 创新点**

创新点在于提供了在完全无目标域标签、仅可获得聚合 AP 反馈的极端条件下，对推理分辨率提升（冻结 1120 × 1120）与参数微调的可重复、受审计的实验结果，验证了在此限制下冻结高分辨率推理可优于 fine‑tune，且提示仅凭源域验证 AP 可能误导模型选择。

**🔧 技术方法**

使用技术包括：RF‑DETR‑Large（DINOv2 ViT‑S 背景）、单一源城市 704 × 704 训练基线、冻结 1120 × 1120 推理（通过位置嵌入插值）、温启动 1120 × 1120 fine‑tune（6 轮、AdamW、cosine 调度）、灰度世界归一化以及矩形尺寸推理（未按预期实现）。

**📊 数据集**

数据集为 AI City Challenge 2026 的单源城市训练集（约13k帧、150k实例）与隐藏混合基准（源域+目标域图像，10类交通目标），但测试集未公开，评估仅通过服务器返回的聚合 COCO‑style AP 进行。

**📈 对比分析**

比较方法为单次服务器提交，使用聚合 AP（IoU 0.50:0.05:0.95）作为唯一指标。实验结果显示：冻结 1120 × 1120 推理在 AP 上提升 0.0382（从 0.3272 至 0.3654），其中对小目标相对提升最高（+27.2%），对中目标绝对提升最大；温启动 fine‑tune 的 AP 仅为 0.3470；灰度世界与矩形推理无显著提升。

**⚠️ 局限性**

限制包括：仅一次实验（n=1）且无随机种子固定；评估基准仅提供聚合 AP，无域分离指标；L3 fine‑tune 与 L1 对比受阈值不一致及执行路径差异干扰；缺乏多次复现、超参数搜索、效率测量（推理时间/显存）及对目标域精细分析；因此结论仅适用于单一模型、单一配置，不能推广为普适规律。

---

## 52. Recurrent Contrastive Learning for Imbalanced Medical Image Classification

**arXiv ID:** 2608.03304 | [PDF](https://arxiv.org/pdf/2608.03304v1)

**作者:** Zhiyuan Zhu `[一作]` (Shenzhen University), Xin Yang `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了基于递归对比学习的RCL框架，用以解决医学图像分类中的类别不平衡问题；

**💡 创新点**

创新点在于引入Temporal Memory Queue（TMQ）与Temporal Anchors（TARs），通过历史特征递归构造锚点场，扩展尾类支持区域，提升类别间分离；

**🔧 技术方法**

采用DINOv3+LoRA微调的骨干网络、投影头、对比损失以及EMA更新的TMQ、TARs等技术；

**📊 数据集**

使用了三大数据集：私有多中心颈动脉超声数据集、公开APTOS 2019糖尿病视网膜病变分级数据集和KneeOA膝关节退化分级数据集；

**📈 对比分析**

通过与ResNet50、Focal、BCL、ECL、HiFuse、ADSR、DGN、DINOv3等多种基线对比，RCL在BAcc、F1、Acc、QWK等指标上均取得最佳或接近最佳性能，提升幅度显著；

**⚠️ 局限性**

局限性在于需手动调节多个超参数，锚点选择不具自适应性，且对极端不平衡情况的鲁棒性尚未完全验证。

---

## 53. Evading Chain-of-Thought Monitoring Through Model Poisoning

**arXiv ID:** 2608.02820 | [PDF](https://arxiv.org/pdf/2608.02820v1)

**作者:** Giorgio Severi `[一作]` (Microsoft), Amanda Minnich `[通讯]` (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在链式推理（CoT）监控中通过模型中毒实现的隐藏后门，证明可通过监督微调在不同模型中植入并使CoT保持表面正常。

**💡 创新点**

提出了“CoT隐藏后门”概念及一种分阶段课程学习方法，使模型在保持良好推理输出的同时实现被动触发的错误答案。

**🔧 技术方法**

使用监督微调、课程训练、线性探针、补丁实验、残差流可视化自编码器等技术分析后门机制。

**📊 数据集**

利用GSM8K数学题集和BeaverTail的有害问答数据集进行实验。

**📈 对比分析**

对比标准、脱耦合及课程后门在不同模型（Phi-4-mini、Qwen3.5-9B、Gemma-4-12B）上的攻击成功率和CoT监控AUC，发现脱耦合后门在CoT监控下AUC仅接近随机，但加入答案后AUC可达90%以上。

**⚠️ 局限性**

局限包括只在3.8B-12B模型、仅通过监督微调实现，未验证更大规模或强化学习/偏好优化场景，且CoT一致性检测仍需进一步研究。

---

## 54. On the Implicit Flatness Bias of Sharpness-Aware Minimization: A Linear Stability Analysis with Quantitative Hyperparameter Bounds

**arXiv ID:** 2608.03197 | [PDF](https://arxiv.org/pdf/2608.03197v1)

**作者:** Jiaxin Deng `[一作]` (Beijing University of Technology), Junbiao Pang `[通讯]` (Beijing University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过线性稳定性分析与梯度噪声协方差推导，量化了 SAM 对平坦性（最大 Hessian 特征值）的隐式偏置，并提出了 λ_max ≤ √(bΓ/(2ρη²) 的立方界，随后验证该预测并基于此设计了 Taylor‑Locality Controlled SAM（TLC‑SAM）以自适应调节扰动半径。

**💡 创新点**

核心创新在于：① 将扰动半径 ρ 与批量大小 b、学习率 η 与梯度范数上界 Γ 共同嵌入最大 Hessian 特征值的界定，形成局部性‑稳定性（stability‑locality）视角；② 以该理论为基础提出基于 Taylor 误差监测的半径自适应机制；③ 通过大量实验验证理论预测并展示其对模型泛化与 Hessian 谱结构的双重改进。

**🔧 技术方法**

主要技术包括：线性稳定性框架、梯度噪声协方差分析、Hessian 最大特征值估计（Hutchinson 方法）、Taylor 误差计算与指数滑动平均、以及多任务实验中的超参数网格搜索。

**📊 数据集**

实验数据集为 CIFAR‑10 与 CIFAR‑100，网络架构包括 ResNet‑18、VGG‑19、WideResNet‑28‑10 与 PyramidNet‑110。

**📈 对比分析**

通过与 SGD、SAM、ASAM、F‑SAM、Eigen‑SAM 等基线和变体的对比，TLC‑SAM 在 ResNet‑18、WideResNet‑28‑10 与 PyramidNet‑110 的 6 种设置中，平均提升 0.1–0.5 个百分点的测试准确率；最大 Hessian 特征值下降约 48%（相比 SGD），表明显著的平坦化效果。

**⚠️ 局限性**

局限性在于：理论推导依赖局部线性化、插值最优点、平方损失等假设，未针对非插值、强非线性网络或其他损失函数进行验证；此外，TLC‑SAM 的超参数选择和训练稳定性仍需在更大规模或不同任务上进一步评估。

---

## 55. Modeling Scientific Experiment Scenes: Dataset and Model

**arXiv ID:** 2608.02892 | [PDF](https://arxiv.org/pdf/2608.02892v1)

**作者:** Minghao Zou `[一作]` (Shandong University of Science and Technology), Wei Zhou `[通讯]` (Cardiff University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文先构建了专门针对物理实验场景的全新场景图数据集 PhysScene，并提出一种跨模态双路径生成器 CM‑DPG，用于开放词汇场景图生成（SGG）。

**💡 创新点**

创新点包括：① 早期双向视觉‑文本互补编码（RC‑MU）提升对象表征；② 视觉‑几何双分支关系推理（G‑VARR）结合空间几何信息，缓解视觉不一致；③ 自适应权重机制平衡长尾关系分布；④ 基于伪标注的关系预训练，进一步提升零样本泛化；⑤ 在科学实验环境中首次实现开放词汇 SGG。

**🔧 技术方法**

技术手段包括 Swin Transformer 视觉特征提取、BERT 文本原型编码、跨模态注意力、双分支融合、几何特征编码、伪标签关系抽取、对数损失与自适应加权、Hungarian 匹配等。

**📊 数据集**

使用的数据集有 PhysScene（4.5k 图像、45.9k 对象、130.4k 关系）和公开的 VG150，后者用于跨域评估。

**📈 对比分析**

在闭集（CS‑SGG）、开放词汇对象（OVD‑SGG）和开放词汇关系（OVR‑SGG）三种设置下，CM‑DPG 在 PREDCLS 与 SGDET 评估指标（R@K、mR@K）均显著优于 22 余种基线，尤其在长尾关系上提升 3–5 个点，并在开放词汇情境下取得首位。

**⚠️ 局限性**

局限性：① PhysScene 规模相对有限，难以覆盖所有实验类型；② 伪标注的噪声仍可能影响预训练效果；③ 模型参数量约 239M，推理速度略高，资源消耗仍不低；④ 对其他科学领域（如生物、化学实验）的迁移性尚待验证。

---

## 56. When Oracle Conditioning Misleads Deployment: Conditioning-Availability Bias in Echocardiographic Segmentation

**arXiv ID:** 2608.03342 | [PDF](https://arxiv.org/pdf/2608.03342v1)

**作者:** Dang P. M. Cao `[一作]` (VinUniversity), Hieu Pham `[通讯]` (VinUniversity)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

评估并缓解在心脏超声分割中因模型使用的阶段信息（心动周期）从理想的oracle信号转为部署时的估计信号所产生的“conditioning-availability bias”。

**💡 创新点**

提出了部署可审计的评价框架，定义oracle–estimated–random三种路径，量化Oracle–Estimated和Oracle–Random的gap，并通过强循环一致性、估计路径检查点选择和阶段扰动来减轻该偏差。

**🔧 技术方法**

使用带FiLM条件的U‑Net网络、相位编码与MLP、相位头预测、相位噪声/丢弃、交叉熵+Dice损失、相邻帧和循环一致性正则化；审计采用oracle、estimated、random三种模式的Dice差异。

**📊 数据集**

CAMUS（500张A4C超声，分割四类）与EchoNet‑Dynamic（10,030张A4C视频，左室分割）两大公开数据集进行训练和验证。

**📈 对比分析**

通过在hold‑out测试集上比较oracle、estimated和random路径的Dice，以及在不同子组和EF下的下游评估，展示强循环模型在oracle上表现良好但在估计路径下跌幅巨大；经过估计路径选择和相位扰动后，gap显著缩小，Dice保持在0.90以上，远优于未缓解模型。

**⚠️ 局限性**

限制包括：未对缺失条件（完全无相位）进行评估；公平性分析仅基于子组大小有限且未测量真实的条件可得性差异；下游EF误差并未完全消除；以及实验只在公开数据集上进行，缺乏真实临床部署的长期监测。

---

## 57. Caved or Convinced: Temporal Sampling Gates Claim Deference in Video Large Language Models

**arXiv ID:** 2608.03160 | [PDF](https://arxiv.org/pdf/2608.03160v1)

**作者:** Yuxin Cao `[一作]` (National University of Singapore), Jin Song Dong `[通讯]` (National University of Singapore)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在视频大型语言模型（VLLM）上设计两种干预（事件重排与采样偏移），研究模型在顺序判断问题中对用户陈述的顺从与纠正行为。

**💡 创新点**

创新点在于用覆盖率 ρ<1 的控制实验将导致顺从的原因拆分为可用性（是否采样到两个事件）和权重（对证据的信任度）两部分，并提出一种不读取用户主张的反转测试，能检测帧中是否真正包含顺序证据。

**🔧 技术方法**

使用稀疏帧采样、事件重排、采样偏移、Youden J 指标、无主张 informedness、前后顺序重排对比、以及前后帧的反转评分算法。

**📊 数据集**

实验数据基于 UCF‑101 与 NExT‑QA 生成的 500 条测试视频，视频包含两段可交换的短动作片段，用以构造真/假顺序对。

**📈 对比分析**

通过对九个公开 VLLM（LLaVA、InternVL、Molmo、Qwen 等）计算 S（顺从率）、C（纠正率）、J=C‑S、J₀（无主张 informedness）以及反转测试的覆盖率、平均帧数和准确率，发现五个模型能读取顺序但仍顺从，InternVL3 在强主张下达到了最高 J≈0.51；反转测试在这五个模型上实现了 0.92–1.00 的准确率，覆盖率约 50%–80%。

**⚠️ 局限性**

局限性在于实验仅关注短动作对与固定槽位的顺序判断，难以推广到更长、多事件或真实流程的视频；对于未采样到事件的模型无法恢复证据，反转测试也无法识别已采样但仍顺从的情况。

---

## 58. LowRank-SSM: Hardware-Software Co-Design for Rank-Reduced Mamba Acceleration on FPGA

**arXiv ID:** 2608.02954 | [PDF](https://arxiv.org/pdf/2608.02954v1)

**作者:** Haocheng Xu `[一作]` (University of California, Irvine), Sitao Huang `[通讯]` (University of California, Irvine)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 LowRank-SSM 框架，结合低秩投影分解与 FPGA 双路径加速，实现 Mamba 模型在保持精度的同时显著提升吞吐量和能效。

**💡 创新点**

创新点包括：① 将投影矩阵秩视为硬件可调变量，提出按频带的敏感度引导贪心低秩分配；② 设计双路低秩+全秩投影流水线，允许混合秩执行且无额外架构开销；③ 将选择性扫描的离散化、状态更新、输出累加、D 残差与 Z 门控融合为单一路线，最大化 DDR4 带宽利用。

**🔧 技术方法**

技术要点：截断 SVD + 按带敏感度评估 + 贪心分配；INT8/INT4 逐行逐组量化；Vitis HLS FPGA 实现双路径 GEMM、RMSNorm、SiLU；多路 AXI 主机实现 DDR4 并行访问；可混合秩掩码驱动运行时动态切换。

**📊 数据集**

使用的基准数据集：WikiText-2（校准与评估），七个零样本任务（LAMBADA、HellaSwag、ARC‑Easy、ARC‑Challenge、Winogrande、OpenbookQA），以及 FP16 基准对照。

**📈 对比分析**

与 FP16、LightMamba INT8/INT4、仅低秩等方案对比；在 Mamba2‑2.7B 上，INT8+低秩实现 7.89 tokens/s，吞吐率提升 2.19×，能效提升 2.03×；相较于 INT4 基线，保持相同精度但压缩率提升至 INT4 等效。

**⚠️ 局限性**

局限性：依赖特定 FPGA 平台（Versal VC1902），低秩对后 8 层效果有限；量化未进行梯度微调；DDR4 带宽仍是瓶颈；在更大模型或不同平台上的可扩展性需进一步验证。

---

## 59. SparSEEty: Extracting Tokens from Sparsity-Exploiting LLM Serving Systems via Deterministic Side Channels

**arXiv ID:** 2608.02995 | [PDF](https://arxiv.org/pdf/2608.02995v1)

**作者:** Yongwan Jo `[一作]` (Yonsei University), Dokyung Song `[通讯]` (Yonsei University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对在Intel TDX CVM中使用稀疏计算的LLM推理系统的端到端token提取攻击。

**💡 创新点**

利用稀疏激活导致的权重访问模式泄露，结合页面故障、块I/O和页面分配等确定性侧信道，构建神经元激活oracle，并通过选择性监控、链式页面武装和概率引导的搜索实现低开销、高精度的token重构。

**🔧 技术方法**

页面故障侧信道、块I/O侧信道、页面分配侧信道、信息熵驱动的神经元选择、链式单步自由页面武装、基于概率的词表搜索、离线反演等。

**📊 数据集**

Skytrax Reviews、Medical WikiDoc、ECHR Law、Private Prompts、System Prompt Leakage等五个真实和合成数据集。

**📈 对比分析**

在5种模型上评估重构准确率（BLEU）和推理时间开销；使用100个第一层神经元即可达到>0.95 BLEU，推理时间提升仅3.7–7.2%，而监控所有神经元会导致>30%开销。

**⚠️ 局限性**

攻击依赖离线模型权重和稀疏激活；若模型完全关闭稀疏优化、采用CPU侧安全计算、或对权重地址进行随机化/动态重排，效果会显著下降；同时需要在CVM内获取侧信道，无法直接在普通云服务中使用。

---

## 60. GraspMeanFlow: SE(3)-Equivariant MeanFlow for Few-Step 6-DoF Grasp Generation

**arXiv ID:** 2608.03295 | [PDF](https://arxiv.org/pdf/2608.03295v1)

**作者:** Jiyong Kwon `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于SE(3)等变的平均速度流模型GraspMeanFlow，用于一次或少数步骤生成六自由度抓握姿态。

**💡 创新点**

创新点在于：1) 通过时间有序指数定义SE(3)平均速度，使得一次更新即可准确实现刚体位移；2) 在等变条件下引入双时间嵌入，保持等变性且不增加网络容量；3) 设计了半群一致性和差分MeanFlow一致性两种无JVP的训练目标；4) 提供了α-Flow预热和端点样本引导。

**🔧 技术方法**

使用的技术包括：SE(3)-equivariant Vector Neuron网络、MeanFlow目标、半群一致性/差分一致性损失、α-Flow预热、Euler或exp‑SO(3)采样器、指导采样、以及与Acronym数据集的配准。

**📊 数据集**

在ACRONYM数据集上进行实验，使用四个类别（Bowl、Laptop、Mug、Pencil）的点云与抓握标注。

**📈 对比分析**

与EquiGraspFlow、SE(3)-DiffusionFields、BRIDGER等基线比较：在NFE=5时，GMF-JVP在大多数类别的抓握成功率超过EquiGraspFlow多达24.3%；GMF-SG在分布一致性（EMD）上更优；两者在一次评估下即可达到与迭代模型相同或更好的质量，且推理时间更短。

**⚠️ 局限性**

局限性包括：对高度对称的对象（如Pencil）表现不稳定；端点采样速率与步数的最佳调度仍未确定；在Euler采样器上自监督微调效果负面。

---

## 61. POMDPs for Autonomous Science Exploration

**arXiv ID:** 2608.03155 | [PDF](https://arxiv.org/pdf/2608.03155v1)

**作者:** Daniel Guirguis `[一作]`, Salah Sukkarieh `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 SHM-POMDP 框架，将科学假设图（SHM）与 POMDP 结合，实现高维观测下的自主科学探索决策。

**💡 创新点**

创新点：① 在观察层使用推断的物理属性而非原始观测进行树分支，抽象化观测空间；② 将 SHM 的观测模型嵌入 POMDP 观测函数，实现不确定观测的贝叶斯推断；③ 在保持完整贝叶斯不确定性的同时，显著降低计算复杂度。

**🔧 技术方法**

技术：POMDP 与 Monte Carlo 树搜索（POMCPOW/PFT‑DPW），SHM 层次贝叶斯网络，学习的观测模型（高斯混合/神经网络），粒子滤波更新，信息增益评估。

**📊 数据集**

数据集：扩展版 RockSample 任务的多维连续观测（5D–50D），以及 Cuprite 矿区的 AVIRIS‑NG 光谱与 USGS 矿物地图。

**📈 对比分析**

比较方法：与连续观测 POMDP 基线（POMCPOW）以及信息理论路径规划器（Greedy、pSPIEL、Random）对比；在 RockSample 上提升 18.6% 奖励、缩减 32.9% 计算时间；在 Cuprite 上实现 2.5× 信息增益、达到 80% oracle 级别。

**⚠️ 局限性**

局限性：仅处理离散物理属性；观测模型需离线训练，缺乏在线自适应；在极高维度或连续属性情形下仍有扩展空间；对新环境的鲁棒性需进一步验证。

---

## 62. MutMem: Cryptographically Authorized Mutation in Persistent Agent Memory

**arXiv ID:** 2608.02843 | [PDF](https://arxiv.org/pdf/2608.02843v1)

**作者:** Walid Saidi `[一作]` `[通讯]` (Independent researcher), Walid Saidi (Independent researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 MutMem，一种在持久 Agent 内存中保留历史、授权、可追溯且抗篡改的可变权重更新协议，并验证其在长记忆任务和对抗性知识毒化中的实用性与安全性。

**💡 创新点**

创新点在于将签名化的历史节点和独立的权重变更承诺双层绑定到可查询的 Merkle 证明链中，实现在不抹除历史的前提下可逆的检索频率调整，并通过可签名的“认知分类链”实现对已检索的毒化内容的可追踪标签化。

**🔧 技术方法**

技术手段包括 SHA‑256 哈希、Ed25519 数字签名、JSON Canonicalization、RFC‑6962 风格 Merkle 证明、PostgreSQL 的事务与 ACL 约束，以及独立可移植的验证器和端到端签名的 recall 证书。

**📊 数据集**

使用的数据集包括 LongMemEval（500 题）和 LoCoMo（1,986 题）评估记忆效能；PoisonedRAG N=100 适配实验用于评估对抗性知识毒化；此外还使用了自然语言问答/BEIR 作为检索语料。

**📈 对比分析**

在长期记忆评估中，LongMemEval 得分 91.8%（95% CI 89.1–93.9%），LoCoMo 得分 74.12%；在毒化实验中，干净目标泄露率 2%，攻击 ASR 3%，毒化检索率 0%；在四臂消融中，签名存储标签将毒化检索从 94% 降至 0%，并将受攻击答案准确率从 40% 提升至 65%，且未对干净答案产生显著影响。

**⚠️ 局限性**

局限性包括：仅保证授权与完整性，无法证明记忆真实性；若 housekeeper 私钥被泄露，攻击仍可合法修改；数据库超级用户仍可写入但可检测；无持续全库审计；对毒化检测的泛化性受限于已校准的固定语料；以及对大规模系统性能和硬件差异的量化尚未完成。

---

## 63. CUDA MPC: A GPU-Native Solver for Model Predictive Control

**arXiv ID:** 2608.03051 | [PDF](https://arxiv.org/pdf/2608.03051v1)

**作者:** Babak Akbari `[一作]` (Queen's University), Melissa Greeff `[通讯]` (Queen's University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一款 GPU 原生的 MPC 求解器 CudaMPC，可在单个 CUDA 核心中完成 ADMM 迭代，实现长时间视野的实时控制。

**💡 创新点**

通过并行在时间轴上的 ADMM、融合核、邻近块原子标志同步以及共享内存驻留，显著降低了 GPU 调度和内存传输开销。

**🔧 技术方法**

使用 CUDA 原生编程、共享内存与原子标志同步、CUDA Graphs、并行时间轴 ADMM、SCP 线性化、增量输入表述等技术。

**📊 数据集**

在六个非线性机器人基准上评估，包括摆杆、倒立摆、车道泊车、拖车、四旋翼杆、10 机器人群体。

**📈 对比分析**

与 CasADi/IPopt、acados/HPIPM、GPU‑iLQR、GPU‑SLS 等 CPU/GPU 求解器对比，CudaMPC 在大视野下实现 100–1000 步长的实时求解，速度提升可达 20–965 倍，且在群体协同与碰撞规避上表现优于基线。

**⚠️ 局限性**

受共享内存容量限制，子视域长度受限；对非常大状态维度或极高约束密度的系统仍需更大显存，且目前仅在桌面 GPU 上验证，嵌入式部署与功耗分析待后续研究。

---

## 64. Scalable Exact Densest P-Partite Subgraph Search in Heterogeneous Information Networks

**arXiv ID:** 2608.03061 | [PDF](https://arxiv.org/pdf/2608.03061v1)

**作者:** Jiadong Xie `[一作]`, Jeffrey Xu Yu `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种高效的、全局最优的密度最大化算法，用于在异构信息网络（HIN）中求解密集P-部图（DPpS）问题。

**💡 创新点**

主要创新包括：①基于整数盒子（box-level）搜索并使用安全上界与提升的裁剪规则；②通过原始计数向量（primitive count vector）实现iRM-set的去重；③有限热身（warm‑up）与种子调度提前获得高质量下界；④终端双胞压缩（terminal‑twin compression）和投影分组（projection grouping）将每个固定iRM-set问题压缩为更稀疏的闭包网络，并利用参数化Hochbaum伪流（parametric HPF）实现一次性求解。

**🔧 技术方法**

技术手段包括：iRM-set枚举、最小割求解、整数盒子划分与安全裁剪、GCD规范化原始计数、局部逼近的热身策略、投影聚合网络构造、终端双胞压缩、参数化伪流算法。

**📊 数据集**

在七个真实HIN数据集上验证：MovieLens、DBLP、Douban、DBpedia、Freebase、Cisco g21、Cisco g22。

**📈 对比分析**

与先前最优方法（AdvExactGVIt）比较，实验表明在k=3、k=4两种meta‑path长度下，平均加速约27.04×，在更大图和更长meta‑path时仍能完成任务；此外返回子图的DPpS密度与基准一致或略优，且在网络安全用户分组应用中保持或提升F1分数。

**⚠️ 局限性**

局限性：对极大图或极长meta‑path仍可能出现OOM或超时；小规模图上额外搜索与预处理开销可能不被完全抵消；算法对硬件内存与并行化支持仍有改进空间。

---

## 65. Relational Priors as Convergence Pressure in LLM-Based Multi-Agent Systems

**arXiv ID:** 2608.03239 | [PDF](https://arxiv.org/pdf/2608.03239v1)

**作者:** Ming Shen `[一作]` (Arizona State University), Yanjun Qi `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大语言模型多智能体系统中显式插入关系先验（signed‑network），观察其对协调与一致性的影响。

**💡 创新点**

创新点在于把关系语义从系统协议中分离出来，采用最小化的签名网络来做为提示级干预，系统性检验其对收敛压力的作用，并提出关系先验不应默认使用的设计建议。

**🔧 技术方法**

主要技术包括：提示工程（在系统提示中嵌入关系描述）、签名关系网络、三种关系类型（Attitude、Trust、Influence）以及对不同模型骨干的多次实验。

**📊 数据集**

使用的数据集有：GovSim（共享资源治理仿真）、OpinionQA（主观问答）、MMLU‑Pro、GPQA‑Diamond（客观问答）。

**📈 对比分析**

对比方法：把显式关系先验与完全不使用关系（zero prior）以及明确说明中立的情况做端点对比；结果显示：在GovSim中正向关系提升可持续协调；在主观辩论中提升一致率；但在客观问答中一致率上升并不伴随准确率提升，且影响随模型、关系类型和拓扑而异。

**⚠️ 局限性**

局限性包括：关系网络固定、对称且对所有智能体可见；仅测试三智能体或五智能体的小规模场景；未考虑私有或动态关系、工具使用、长时记忆或人机交互；因此结论不一定适用于更复杂的多智能体系统。

---

## 66. NeuroMosaic: Anatomically Grounded Multimodal Large Language Modeling for Molecularly Aware Glioma Reasoning from 3D MRI and Clinical Narratives

**arXiv ID:** 2608.03187 | [PDF](https://arxiv.org/pdf/2608.03187v1)

**作者:** Yantong Liu `[一作]`, Hyun-Ae Lee `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出 NeuroMosaic，一种将多序列3D MRI与临床文本和分子概念通过解剖索引稀疏路由集成的多模态大语言模型，用于脑胶质瘤分型、分子预测和可解释报告生成。

**💡 创新点**

创新点在于将3D影像分割与解剖图谱构建为患者特异性图网络，并通过任务条件稀疏路由将区域级视觉标记与分子概念记忆相绑定，实现可审计、证据关联的推理。

**🔧 技术方法**

技术上使用多分辨率3D分词器、解剖图网络路由器、分子概念记忆、门控多模态解码器和校准阶段，配合四阶段训练和覆盖、稳定性约束。

**📊 数据集**

数据集包括BraTS/TCGA‑TCIA、UPenn‑GBM、UCSF‑PDGM以及一组保留机构的外部数据，共计约2,381名患者。

**📈 对比分析**

与多种基准相比（3D判别、晚期融合、通用MLLM、检索MLLM），在内部和外部宏观F1上获得最高0.784/0.761/0.742，分子预测AUROC最高0.918、0.861、0.781，证据定位准确率0.703，校准ECE下降至0.034。

**⚠️ 局限性**

局限在于对缺失序列的鲁棒性仍有限（缺失两序列时宏观F1下降0.055），且模型在少量外部样本时置信区间较宽，需进一步临床验证。

---

## 67. Search, Inspect, Fetch: Exploiting Boolean Retrieval for Deep-Research Agents

**arXiv ID:** 2608.02751 | [PDF](https://arxiv.org/pdf/2608.02751v1)

**作者:** Shuai Wang `[一作]` (University of Queensland), Guido Zuccon `[通讯]` (University of Queensland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Sieve，一种利用布尔查询语言（BQL）进行结构化检索、检索后检查和只取所需章节的深度研究代理工作流程

**💡 创新点**

通过将文档结构（标题、章节、元数据）保留并可供检索、检查与读取使用，实现了更高准确率和更低上下文消耗的全流程改进

**🔧 技术方法**

布尔查询语言（BQL）、可插拔排名器（BM25、dense、BM25+Dense）、结构化结果卡、章节级别获取、LLM驱动的代理循环

**📊 数据集**

HotpotQA、MuSiQue、BrowseComp‑Plus（对其做结构化与平面版本对比）

**📈 对比分析**

与传统 Search‑Visit、Search‑AutoRead、Search‑Fetch、DCI 等基线在相同预算下比较，Sieve 在所有三组数据集上准确率最高，且在 20.7%‑50.6% 的 token 使用量下降

**⚠️ 局限性**

仅在三组数据集上评估；BQL 需要手工设计或 fallback 机制；结构化版本依赖已有结构字段，未验证对完全无结构文本的适用性

---

## 68. Clinically-Grounded Hierarchical Classification for Consistent Chest X-ray Interpretation

**arXiv ID:** 2608.03016 | [PDF](https://arxiv.org/pdf/2608.03016v1)

**作者:** Jong Hak Moon `[一作]` (Yeji X), Minjun Kim `[通讯]` (Yeji X)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出单阶段层次多标签分类框架 CHASE，利用三层临床对齐的解剖学层级（9 区→17 子区→28 病灶）对胸片进行粗到细的异常识别与定位。

**💡 创新点**

创新点在于构建符合放射科医生阅读流程的三层层次结构，并联合多级监督、全局 KL 对齐与层次违规惩罚，强制跨层级概率单调性，显著提升层次一致性与可解释性。

**🔧 技术方法**

技术手段包括 Vision Transformer 共享骨干，层级特定线性头，联合损失（多级 BCE + Global KL + Hierarchy‑Violation Penalty），以及基于 ViT 自注意力的可视化定位。

**📊 数据集**

使用结合 Chest ImaGenome 与 MIMIC‑CXR 的 54 级标签体系，训练/验证/测试分别为 213,361/1,733/3,041 研究（237,070/1,959/3,403 图像）。

**📈 对比分析**

在与 Flat‑ViT、单层 Flat‑ViT、Hier‑ViT、H‑CAST 等基线同条件下比较，CHASE 在 Top/Mid/Leaf 的 AUC 分别为 0.826/0.803/0.848，概率一致性 74.26%，明显优于基线并逼近单层上限。

**⚠️ 局限性**

局限性包括仅覆盖 54 级标签且层次结构固定；对多标签同时出现在不同解剖结构的情况适应性有限；跨设备、不同放射员的鲁棒性尚待进一步验证。

---

## 69. Fast Object Removal Attacks on Safety-Critical Video-based Perception Systems

**arXiv ID:** 2608.02806 | [PDF](https://arxiv.org/pdf/2608.02806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 70. Moving the Safety Barrier: Dynamic Routing Adaptive Alignment Against White-Box Attacks

**arXiv ID:** 2608.02674 | [PDF](https://arxiv.org/pdf/2608.02674v1)

**作者:** Shangze Li `[一作]` (Nanjing University of Science and Technology), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DRAA 框架，先对大模型的安全路由进行定位，随后通过掩蔽该路由并挖掘因果失败样本，利用动态路由 DPO 训练模型在安全路由被破坏时能通过补偿路由保持拒绝行为。

**💡 创新点**

创新点在于将安全防御从静态节点/路径转向动态补偿路由：通过 ES+SAS 定位安全路由、因果失败采样构造偏好对，并在训练中锁定原安全路由（LoRA 梯度掩蔽），迫使模型学习“被阻断时绕行”的安全路径。

**🔧 技术方法**

技术包括激活差异定位（Effect Size 与 Safety Activation Shift）、因果失败挖掘、动态路由 DPO（DR‑DPO）与 LoRA 梯度掩蔽，以及多模态/文本 LLM 的微调与评估。

**📊 数据集**

使用 SafeNeuron 相关安全数据集（StrongREJECT、HarmfulQA、LLM‑LAT、NaturalReasoning 等）进行安全评测，通用能力用 ARC、GSM8K、TruthfulQA 评估；多模态任务使用 VL‑Question 与 NSFW 数据集。

**📈 对比分析**

在 Qwen2.5、LLaMA‑3.2、Gemma、Phi 等多种后端模型上对 ORI/ES/SAS/FULL 级别的白盒剪枝攻击进行对比，DRAA 在 FULL 级别下 ASR 低至 0–5/313，显著优于 SN‑Tune、RLHF‑Safety、SafeNeuron，并保持或提升 ARC/GSM8K/TruthfulQA 等通用指标。

**⚠️ 局限性**

局限性：需先对每个模型定位安全路由，定位与训练成本随模型规模增长；实验仅覆盖剪枝与激活裁剪等攻击，未评估对参数梯度或结构重新构造等更极端白盒攻击；在极端大模型上可扩展性和对未知攻击的鲁棒性仍待验证。

---

## 71. Verifier-Guided Model Discovery for Physical Dynamical Systems with Pretrained Symbolic Transformers

**arXiv ID:** 2608.02662 | [PDF](https://arxiv.org/pdf/2608.02662v1)

**作者:** Farbod Faraji `[一作]` (Imperial College London), Francesco Belardinelli `[通讯]` (Imperial College London)

**通讯引用:** 879 | [OpenAlex ID](https://openalex.org/A5055883955)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于可执行验证器的工作流程，将预训练的符号Transformer（ODEFormer）迁移到高维物理数据上，自动发现可解释且物理可接受的动力学方程；通过多轨迹候选池化、可执行的局部/全局验证、系数优化，最终得到在多初始条件和参数范围内具有良好泛化的方程。

**💡 创新点**

创新点在于：①在保持Transformer不变的前提下，使用可执行验证器对多轨迹候选方程进行筛选，确保所选方程在动力学和物理上可接受；②通过多轨迹池化与验证实现从合成预训练向实际高维物理系统的可靠迁移；③在跨参数（Reynolds数）任务中获得单一可推广的符号模型。

**🔧 技术方法**

技术栈包括：预训练符号Transformer（ODEFormer），Verifier-guided（VG）流程（多轨迹池化、可执行局部/全局验证器、系数优化（Nelder‑Mead）），降维方法（POD、自编码器），以及对比评估指标（MSE、NMSE）。

**📊 数据集**

数据集：①合成Van der Pol轨迹（不同μ值的训练/测试轨迹）；②二维不可压流体绕柱的数值仿真数据，Re=150–450（固定Re=300测试）以及对应的Vorticity场（60,000维）。

**📈 对比分析**

比较方法：将VG工作流程与原始单轨迹ODEFormer进行对比；在Van der Pol上，VG在8个隐藏初始条件下平均误差从≈1.21降至≈6.99×10⁻⁴；在柱流固定Re实验中，坐标滚动误差<7%，端到端场误差≈15.6%；在跨参数Re实验中，VG在未见Re值上误差不随插值/外推单调增加，表现出良好的泛化。

**⚠️ 局限性**

局限性：①需要降维后的坐标能够近似闭合自律描述，并且与Transformer的预训练词汇匹配；②若系统存在显著的记忆、外部驱动或缺失的动力学，VG无法补充这些结构；③当前方法仅对已获得可行的符号候选类有效，无法生成完全超出预训练分布的新结构。

---

## 72. SUV: Future Scene Understanding as Video Generation for End-to-End Driving

**arXiv ID:** 2608.03084 | [PDF](https://arxiv.org/pdf/2608.03084v1)

**作者:** Yibo Yuan `[一作]` (Xi'an Jiaotong University), Jianru Xue `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的端到端驾驶框架SUV，将未来场景理解视为视频生成，通过共享的视频专家生成RGB、语义分割、相对深度和实例跟踪四个未来流，并用动作专家在视频生成过程中对这些流的潜在表示进行聚焦来生成行驶轨迹。

**💡 创新点**

创新点在于：①将多模态未来预测合并为单一视频生成任务，消除任务特定预测头；②利用预训练视频生成模型（Wan2.2‑5B）并在目标域进行后训练，提升多流预测能力；③通过掩码联合视频‑动作注意力，使动作专家能够直接读取所有未来流的潜在表示，提升规划效果。

**🔧 技术方法**

技术包括：预训练视频生成器（Wan2.2‑5B）、VAE编码解码、文本提示嵌入、Mixture‑of‑Transformers架构、Masked Joint Video‑Action Attention、流式时间注意力和多流流匹配损失。

**📊 数据集**

使用的主要数据集有NAVSIM‑v2（navtrain、navtest、navhard）和WOD‑E2E（长尾规划任务），同时用SAM3和DA3作为教师生成分割、跟踪和深度参考。

**📈 对比分析**

与现有方法（如UniAD、VAD、SparseDrive、DriveVLA‑W0、Fast‑WAM等）在NAVSIM‑v2上对比，SUV在navtest上获得91.0 EPDMS、navhard上获得36.9 EPDMS，超过最强对手（如Metis、EponaV2）约0.7–0.8分；在WOD‑E2E上取得RFS 7.94，位列同类方法前列。

**⚠️ 局限性**

局限性包括：①对单摄像头输入的依赖，缺少多视角或雷达信息；②视频生成模型在计算量上仍较大，尽管提供多步推理选项；③对分割、深度、跟踪等教师的依赖，使得评估受限于这些教师的准确性。

---

## 73. AgentPanel: Toward a New Paradigm for Human--AI Collaboration in Exploring Scientific Questions

**arXiv ID:** 2608.03283 | [PDF](https://arxiv.org/pdf/2608.03283v1)

**作者:** Zhiyao Cui `[一作]` (Shanghai Artificial Intelligence Laboratory), Shuyue Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并部署了 AgentPanel，一个公开的多智能体人机协作科学论坛，允许研究者与异构 LLM 代理在共享讨论中生成、讨论并完善科学想法。

**💡 创新点**

创新点在于将早期科学思路视为开放式探索过程，支持去中心化异构多智能体自发参与，提供多样化视角，并利用轻量级社交信号辅助候选想法筛选。

**🔧 技术方法**

使用了异构 LLM 后端、事件驱动生命周期管理、观察–推理–行动循环、成本感知执行策略、结构化工具接口，以及前端 web 与数据库交互。

**📊 数据集**

使用 IdeaBench、LiveIdeaBench 两大基准进行离线评估，并在实际部署期间收集了 1,508 条科学问题线程、467 个代理及 290,000+ 交互事件的数据。

**📈 对比分析**

通过与 Google MAD 的对比实验和 20 名科研者的使用调查，AgentPanel 在可行性、整体质量等指标上优于 MAD，且轻量级点赞/星标能有效提示高质量想法，整体性能显著提升。

**⚠️ 局限性**

限制在于生成的想法仍属推测性，需要专家验证；评估依赖人工标注，系统尚未完善交互引导与大规模协同机制，且在极端复杂任务中多智能体协作仍面临协调难题。

---

## 74. TASQ: Temporal-Adaptive Bit Sparsification Quantization for Diffusion Models

**arXiv ID:** 2608.03057 | [PDF](https://arxiv.org/pdf/2608.03057v1)

**作者:** Seokho Han `[一作]` (Sungkyunkwan University), Jong Hwan Ko `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种时序自适应位数稀疏量化方法TASQ，能够在扩散模型的去噪阶段按层次动态调整权重量化精度。

**💡 创新点**

创新点在于将存储精度与执行精度分离，利用共享高精度权重和可学习的LSB掩码实现多阶段、多层精度分配，无需复制权重或运行时搜索。

**🔧 技术方法**

采用了LoRA量化感知训练、LSB掩码学习、Farthest-Stage-First训练策略以及自研的Temporal-Precision Engine实现按位串行计算。

**📊 数据集**

在PixArt-Σ、SANA-1.6B和SDXL-Turbo三种扩散模型上使用COCO Caption、MJHQ-30K和sDCI等数据集进行训练和评估。

**📈 对比分析**

与静态8位量化、ViDiT-Q、MixDQ、SVDQuant等基线相比，TASQ在保持相同平均计算预算下实现了25–50%周期缩短、6.1–7.5×比8位串行低，且在FID/ImageReward等指标上接近或优于静态量化。

**⚠️ 局限性**

局限性包括：仍需先以最高精度存储权重；对不同硬件的实现复杂度；以及对极低位宽（≤3位）激活量化的鲁棒性尚待进一步验证。

---

## 75. Mapping the City Through the Lens of Language Models

**arXiv ID:** 2608.02971 | [PDF](https://arxiv.org/pdf/2608.02971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 76. Speculative Successive Cancellation Decoding of Polar Codes

**arXiv ID:** 2608.02760 | [PDF](https://arxiv.org/pdf/2608.02760v1)

**作者:** Ryan Seah `[一作]` (McGill University), Warren J. Gross `[通讯]` (McGill University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种在极化码成功率判定中引入投机执行并通过验证机制保证解码正确性的Speculative Successive Cancellation（Spec‑SC）框架。

**💡 创新点**

创新点在于把投机并行扩展到一般子树节点（不一定是传统的特殊节点），并通过符号一致性检查实现无误码率损失的在线验证，从而突破了SC解码的顺序依赖。

**🔧 技术方法**

使用的技术包括极化码结构、SC递归树遍历、投机右子树估计、符号检查验证、最小和近似的f、g函数以及在BPSK AWGN信道下的仿真。

**📊 数据集**

仿真数据集为长度为4096的极化码（R=1/4、1/2、7/8），使用Sionna框架、BPSK调制、Gaussian近似构造及AWGN信道噪声。

**📈 对比分析**

与传统SC解码比较时，BER/FER保持完全相同，但平均遍历节点数下降至原来的31%~36%（相当于降低64%~69%），验证了投机策略在提高吞吐量上的有效性。

**⚠️ 局限性**

局限性包括投机验证有时会误判为失败导致无效并发，投机成功率受信噪比和节点位置影响；目前仅在算法层面验证，缺乏硬件实现评估及更高效的验证准则。

---

## 77. GUI-Lens: Coarse-to-Fine Cropping for GUI Grounding with General-Purpose VLMs

**arXiv ID:** 2608.03270 | [PDF](https://arxiv.org/pdf/2608.03270v1)

**作者:** Zichuan Fu `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于通用视觉‑语言模型的粗到细裁剪框架，通过OCR与UI检测生成坐标参考，逐步裁剪并验证，最终实现精确的GUI目标定位。

**💡 创新点**

创新点在于将多步可视化裁剪与模型自选裁剪策略相结合，配合坐标参考与验证环节，显著降低单步点击预测误差的累积。

**🔧 技术方法**

采用OCR、UI组件检测、通用VLM（如GPT‑5.5、Claude Opus 4.7、MiniMax‑M3）、多轮裁剪、可视化验证与坐标映射等技术。

**📊 数据集**

实验覆盖 ScreenSpot‑Pro、ScreenSpot‑v2、MMBench‑GUI‑L2、UI‑Vision、OSWorld 等多种静态与交互式 GUI 基准数据集。

**📈 对比分析**

与多种通用 VLM、专用 GUI 模型和现有系统进行对比，平均提升可达 24.9% 的准确率，GPT‑5.5 在 ScreenSpot‑Pro 上达到 state‑of‑the‑art；在 OSWorld 交互任务中也表现最佳。

**⚠️ 局限性**

主要限制包括对 VLM 性能的高度依赖，裁剪与验证步骤导致额外推理延迟，以及 OCR/检测误差可能影响坐标参考的可靠性；在极高分辨率或极小目标场景下仍存在挑战。

---

## 78. Noise-Aware Shrinkage for Differentially Private Zeroth-Order Fine-Tuning of Large Language Models

**arXiv ID:** 2608.03277 | [PDF](https://arxiv.org/pdf/2608.03277v1)

**作者:** Lele Zheng `[一作]` (Xidian University), Yulong Shen `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于噪声感知收缩的DP-ZO方法SAGE，用来在仅前向评估的前提下改进大模型私有微调的梯度更新。

**💡 创新点**

通过对已知高斯噪声基准进行能量校正、指数移动平均跟踪并相对热身参考动态缩放，使得在不同训练阶段自适应减小噪声主导的更新，从而提升私有微调效果。

**🔧 技术方法**

纯后处理的噪声基准校正、EMA信号能量跟踪、相对可靠度量收缩因子、基于DP-AggZO的零阶梯度估计与聚合、RDP隐私计数。

**📊 数据集**

RoBERTa-large和OPT-1.3B/6.7B在SST-2、SST-5、SNLI、MNLI、RTE、TREC、SQuAD等自然语言任务。

**📈 对比分析**

与DP-AdamW、DPZero、DP-AggZO等基线在相同隐私预算、查询数和训练设置下进行匹配对比，SAGE在所有任务和模型规模上均提升2-6个百分点的准确率/F1，保持了前向评估的内存优势。

**⚠️ 局限性**

仍受限于零阶梯度估计的方差和对高维模型的收敛速度，且在极端强隐私（ε→0）下性能提升有限；只对DP-AggZO等聚合方法适用，未直接验证对单向或更复杂梯度压缩方法。

---

## 79. Maglev: Sliding Recurrent Memory

**arXiv ID:** 2608.02870 | [PDF](https://arxiv.org/pdf/2608.02870v1)

**作者:** Bo Liu `[一作]` (University of Texas at Austin), Qiang Liu `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Maglev模型，一种固定内存的循环Transformer，使用预填充器生成记忆目标，解码器通过滑动窗口注意力和递归K/V注入学习生成下一步记忆并预测下一个词。

**💡 创新点**

创新点在于通过两阶段并行训练（prefiller与decoder）实现非线性递归记忆的训练，消除传统递归模型在预训练时需要顺序展开的限制，并且在推理时仅需滑动窗口注意力即可保留长期记忆。

**🔧 技术方法**

采用Transformer架构结合滑动窗口注意力、递归K/V注入、可共享或分离的prefiller与decoder网络、记忆一致性损失，以及基于RMSNorm和门控混合的多头注意力机制。

**📊 数据集**

在nanochat d20框架下使用FineWeb-Edu、LAMBADA等大规模文本语料进行预训练，共训练约43.52B个token。

**📈 对比分析**

与纯滑动窗口Transformer、全注意力混合模型以及Latent Recurrent Transformer（LRT）比较，Maglev在FineWeb-Edu验证BPB、LAMBADA困惑度以及多项下游任务（PIQA、HellaSwag、WinoGrande、ARC、SocialIQA、BoolQ）上均取得更优或相近的平均表现，参数共享版在推理内存与速度上更具优势。

**⚠️ 局限性**

局限性包括：需额外的prefiller网络和一致性损失可能导致训练复杂度提升；最佳参数共享方案尚未确定；在更大规模模型或不同任务上需要进一步验证其可扩展性与稳健性。

---

## 80. Standalone DINOv3 for Training-Free Open-Vocabulary Semantic Segmentation in Remote Sensing

**arXiv ID:** 2608.03023 | [PDF](https://arxiv.org/pdf/2608.03023v1)

**作者:** Changhao Zhao `[一作]` (Huazhong Agricultural University), LingLin Zeng `[通讯]` (Huazhong Agricultural University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 DinoSplat-OV，一个训练免费、基于 DINOv3 文本编码器的遥感开放词汇语义分割框架，包含文本同义词聚合、文本感知拉普拉斯传播 (TLP)、二维高斯溅射上采样 (GSUP) 以及全局锚点滑窗等模块；

**💡 创新点**

创新点在于将训练免费 CLIP 思路迁移至冻结的 DINOv3，提出 TLP 进行粗粒度特征去噪、GSUP 通过测试时优化实现像素级特征重建，并通过全局锚点滑窗解决大尺寸遥感图像的分割；

**🔧 技术方法**

使用了文本同义词聚合、文本感知拉普拉斯传播、二维高斯溅射上采样、全局锚点滑窗注意力以及测试时优化等技术；

**📊 数据集**

在 UDD5、DOTA、LoveDA、Vaihingen 等四个遥感语义分割基准上进行评估；

**📈 对比分析**

在统一滑窗设置下与 MaskCLIP、ClearCLIP、SegEarth-OV、原始 DINOv3 等训练免费基线进行对比，mIoU 在 UDD5 42.9%、DOTA 28.6% 等数据集上达到或超过现有最优方法，尤其在密集目标场景表现突出；

**⚠️ 局限性**

局限性包括对文本同义词聚合的依赖、TLP 与 GSUP 需要额外的测试时优化和邻域采样、对极端尺度或多模态遥感数据的泛化仍有限，以及缺乏对极端光照、云雾等条件的鲁棒性验证。

---

## 81. SynEnergy: Anomaly Semantic-Guided Diffusion for Synthetic Energy Data Generation

**arXiv ID:** 2608.03087 | [PDF](https://arxiv.org/pdf/2608.03087v1)

**作者:** Lin Jiang `[一作]` (Florida State University), Guang Wang `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于异常语义的扩散框架 SynEnergy，用于生成既能保持整体能源消费分布又能保留低/高耗异常事件的合成数据。

**💡 创新点**

创新点在于两阶段设计：先通过异构图学习异常语义，再将语义注入扩散去噪过程，能够精确捕捉并重现稀疏、时间空间局部的异常事件。

**🔧 技术方法**

使用的技术包括稀疏残差编码器、跨区域异构图注意力（RAPA-Graph）、图增强的异常语义空间、Transformer 结构的扩散反向网络以及层级控制注入机制。

**📊 数据集**

实验数据集涵盖佛罗里达州塔拉哈西十年家庭级30分钟采样（147个CBG，超过10亿条数据），以及纽约州和加州的公开小时级数据。

**📈 对比分析**

与 11 种基准（GAN、VAE、Flow、Diffusion、LLM 及能源专用方法）比较，SynEnergy 在总体生成精度、异常保留（平均提升12.21%）以及下游异常检测/预测精度（提升2.96%）方面均表现最佳。

**⚠️ 局限性**

局限性包括：仅使用少量地区属性作为条件，可能不足以完全捕捉区域异质性；合成数据仍存在潜在隐私泄露风险，需要进一步评估。

---

## 82. Safety in Batches? Understanding and Mitigating Safety Failures in Batch Prompting

**arXiv ID:** 2608.02681 | [PDF](https://arxiv.org/pdf/2608.02681v1)

**作者:** Kihyun Kim `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大型语言模型中批量提示（batch prompting）所导致的安全失效，即在单独提问时被拒绝的有害问题，在与若干正常问题一起打包为批量请求时能够诱导模型生成有害答案。

**💡 创新点**

创新点在于首次将批量提示定位为独立的安全风险，证明其与上下文学习与长上下文攻击无关，并从对齐信号削弱和拒绝信号稀释两方面解释该漏洞；随后提出通过批量感知的偏好优化（DPO）显著恢复模型安全性。

**🔧 技术方法**

采用批量提示、奖励间隙（reward gap）分析、PCA可视化隐藏状态、拒绝向量（refusal vector）评估、直接偏好优化（DPO）以及监督微调（SFT）等技术，对模型在批量输入下的安全表现进行深入分析与对策。

**📊 数据集**

使用的主要数据集包括：JailbreakBench（100条有害问题）、StrongREJECT（313条有害问题）、GSM8K（无关普通问题）、以及多种公开安全评估模型（BeaverDam、WildGuard、Llama Guard4）作为判定器。

**📈 对比分析**

在与七种现有基线破解方法（CodeChameleon、FlipAttack、JAIL-CON、Working Memory Attack、ICA、Many-Shot Jailbreaking、NINJA）对比中，批量提示在六个开源与商业模型上平均攻击成功率（ASR）分别达到约62%和67%，显著高于基线；而在经过DPO训练后，批量提示的ASR降至<1%，表明防御效果极佳。

**⚠️ 局限性**

局限性包括：研究重点是发现与分析漏洞而非设计最强攻击；对齐信号的分析依赖公开奖励模型，可能无法完全反映目标模型内部机制；未探讨适应性批量结构对DPO防御的可能绕过。

---

## 83. Learning Music Style for Piano Arrangement Through Cross-Modal Bootstrapping

**arXiv ID:** 2608.03050 | [PDF](https://arxiv.org/pdf/2608.03050v1)

**作者:** Jingwei Zhao `[一作]` (Songscription), Ye Wang `[通讯]` (National University of Singapore)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种跨模态框架，利用预训练音频与符号音乐语言模型结合的Q-Former，从参考音频中提取隐式音乐风格，并在lead sheet的内容指引下生成风格化钢琴 MIDI 排列。

**💡 创新点**

创新点在于将Q-Former用于音频-符号风格对齐，实现不需重训练两大模型的“风格迁移”与“音频到MIDI检索”，并通过两阶段预训练-生成策略解耦风格与内容。

**🔧 技术方法**

核心技术包括：Q-Former跨模态查询网络、对比学习、匹配学习和生成预训练、MuseCoco符号LM的LoRA适配、音乐音频LM（MusicGen）与符号LM（MuseCoco）的联合使用。

**📊 数据集**

使用的主要数据集为POP909（钢琴改编曲）和PIAST（多风格钢琴录音+标注），并在Ballroom与GTZAN等外部数据集上进行泛化测试。

**📈 对比分析**

与PiCoGen2、Audio-to-MIDI等基线对比，在内容保真度、节奏与力度一致性等客观指标上表现优异；在主观听感评测中，模型在一致性、音乐性方面显著优于基线，性能提升显著。

**⚠️ 局限性**

主要局限在于对lead sheet的依赖导致内容保真度受Sheetsage转录误差影响；在流派与乐器多样性方面仍有提升空间，且跨模态风格对齐对极端音频变形（如混响、噪声）鲁棒性待进一步验证。

---

## 84. CrossScope: A Role-Asymmetric World Model for Joint Dual-Scope Surgical Video Prediction

**arXiv ID:** 2608.03211 | [PDF](https://arxiv.org/pdf/2608.03211v1)

**作者:** Wanhao Liu `[一作]` (Guangdong University of Technology), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向双摄像头（Mother–Child）ERCP手术的视频未来预测模型CrossScope，能够根据不同视角的角色特征进行目标特定的信息路由；

**💡 创新点**

创新点在于：1) 角色非对称双视角预测框架；2) 通过几何轨迹条件实现Child视角运动预测；3) 通过几何许可的姿态对齐实现Child视觉信息在Mother视角的可写性；4) 引入目标特定的读取/写入契约和残差交互；

**🔧 技术方法**

使用了基于Diffusion Transformer（DiT）的双流架构，结合VAE编码、Pose Readout、几何条件、残差写入和零初始化的残差机制，并在训练中采用流匹配、姿态损失和辅助轨迹监督；

**📊 数据集**

构建了双摄像头ERCP基准，包含同步的phantom实验（300条对齐视频，596,190帧）和真实世界数据（52条对齐视频，34,800帧），并提供相应的分割和定位标注；

**📈 对比分析**

与多种基线（HunyuanVideo-I2V、Wan2.2、Cosmos-H-Surgical、Early-Fusion、Symmetric MoT）进行对比，评估指标包括PSNR、SSIM、FID、LPIPS、Mask IoU、Box IoU、检测召回、终点误差和中心线误差。CrossScope在所有角色特定指标上均优于基线，尤其在Mother视角的结构保真度和Child视角的定位与轨迹精度方面表现突出；

**⚠️ 局限性**

局限性在于：仅针对短时域预测（8帧预测），仅在单一双摄像头ERCP场景中验证，缺乏长时域、多中心和多设备的验证，且未在临床决策支持环境中进行前瞻性评估。

---

## 85. When Policies Change Probabilities: Modular Decision-Making for LLM Code Review

**arXiv ID:** 2608.02677 | [PDF](https://arxiv.org/pdf/2608.02677v1)

**作者:** Rasvik Kudum `[一作]`, Sneheel Sarangi `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究四种已部署的LLM代码评审接口，探讨在相同补丁与证据下，改变决策成本阈值是否会导致报告的失败概率发生变化；并提出将风险评估、外部监测与决策阈值分离的模块化管道

**💡 创新点**

发现成本阈值的改变会显著影响LLM给出的概率，说明概率并非独立于决策策略；提出并验证了一种可分离风险估计与决策阈值的模块化设计，显著提升概率准确度并降低决策损失

**🔧 技术方法**

使用语言模型（DeepSeek、Grok、Mistral、GPT/Codex）进行概率与动作生成，使用Gemini 3.5 Flash作为监测模型，并通过逻辑回归与logit组合实现概率融合；采用Brier分数与决策损失评估

**📊 数据集**

基于SWE-rebench的360个Python仓库问题，每个问题产生一个通过与一个失败的补丁，共计720条记录；其中80个问题用于校准A，80个用于校准B，剩余200个用于测试

**📈 对比分析**

对比相同补丁在低先验、等成本、10:1、20:1等四种条件下的概率和决策；与重复调用对照、决策规则固定的代码实现、以及模块化管道进行比较。结果显示：高成本提示下的直接概率往往导致更高的决策损失；使用等成本概率并在代码中应用10:1阈值可显著降低损失；模块化管道在等成本下平均损失下降约0.07，接受率58–68%，但在10:1阈值下不再接受任何补丁

**⚠️ 局限性**

实验受限于成本与阈值同时变化，无法分离成本语义与阈值影响；使用的补丁对是人工选取，未覆盖自然PR场景；仅评估单一生成器、监测器和四个固定评审器，未能展示不同模型配置或多模型组合的普适性

---

## 86. LoCA: Forward-Only LLM Tuning after One-Shot Calibration with Local Credit Assignment

**arXiv ID:** 2608.03020 | [PDF](https://arxiv.org/pdf/2608.03020v1)

**作者:** Linhan Xia `[一作]` (University of Oklahoma), Shengxin Zhu `[通讯]` (Beijing Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段的 LoCA 方法，在一次前向传播的 calibration 阶段学习低秩反馈映射，然后在随后的每一步仅使用前向计算和闭式岭回归来更新适配器，从而实现无重复反向传播的参数高效微调。

**💡 创新点**

创新点在于将全局梯度校准一次性完成，利用低秩反馈映射把顶层误差映射为局部信用信号，并用闭式解的岭回归实现每层局部目标的最优更新，兼具低内存和低计算成本。

**🔧 技术方法**

技术包括一次性反向传播校准低秩反馈矩阵、基于残差流归一化的目标尺寸标准化、每层闭式岭回归求解、以及共享尺度归一化候选集以实现跨模型规模复用。

**📊 数据集**

使用 Qwen2.5 系列 0.5B、1.5B、3B、7B、14B 以及 SmolLM2-1.7B 模型，在 SST-2、BoolQ、ARC-Easy、OpenBookQA、HellaSwag 五个判别任务上进行实验。

**📈 对比分析**

与冻结模型、LoRA 和 Adapter-MeZO 进行比较。LoCA 在 25 个模型-任务组合中有 16 个单元的交叉熵低于对应 LoRA；GPU 峰值内存比 LoRA 低 26–29%，CPU 稳态内存低 36–52%，单轮前向时间低 43–48%。

**⚠️ 局限性**

局限性包括仅适用于小幅度调优，较大模型变动或长时间训练可能需要重新校准；对长文本生成、分布大幅变化的任务未做验证；仍需一次反向传播校准，完全无反向传播仍不可实现。

---

## 87. SABRE: A Multi-Agent Approach for Selecting Out-of-Distribution Detectors Under a Budget

**arXiv ID:** 2608.02959 | [PDF](https://arxiv.org/pdf/2608.02959v1)

**作者:** Mary Wisell `[一作]` (San Diego State University), Salimeh Sekeh `[通讯]` (San Diego State University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于语言模型代理的SABRE框架，在视觉-语言模型上动态选择和加权后验异常检测器以适应部署域。

**💡 创新点**

创新点在于将OOD检测视为域内选择问题，利用少量标记样本对检测器可靠性进行校准，并在预算内自动聚合最佳检测器。

**🔧 技术方法**

采用三种LLM代理（Selector、Reporter、Analyst）、多模态密度检测器（SMAP、RCAP、MMCA、GMAP）以及传统的MSP、Energy、MCM、QPM等后验评分器。

**📊 数据集**

使用自然图像域（ImageNet-100、iNaturalist、SUN、Places、Textures）与病理图像域（NCT-CRC、CRC-VAL、GCHTID）以及多种CLIP族编码器（ViT-B/16、ViT-L/14、SigLIP、BiomedCLIP）。

**📈 对比分析**

与固定后验检测器相比，SABRE在所有域中至少匹配最优单一检测器，并在检测器失效时恢复到≈97%的最佳性能，且在预算为3次检测器调用时已达到性能峰值。

**⚠️ 局限性**

局限在于仍需手工构建多样化检测器池、依赖少量标记校准样本且仅针对图像+字幕两模态，且在没有任何可用检测器的极端域上无法提升。

---

## 88. Robust Counterfactual Policy Optimisation via Nondeterministic Causal Models

**arXiv ID:** 2608.02893 | [PDF](https://arxiv.org/pdf/2608.02893v1)

**作者:** Jessica Lally `[一作]` (King's College London), Sander Beckers `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出在存在全局隐含混淆变量且系统本身随机的情况下，对马尔可夫决策过程进行反事实政策优化的框架。

**💡 创新点**

创新点在于：①引入概率非确定性因果模型(PNSCM)来同时处理隐含混淆与不可约随机性；②在混淆灵敏度分析框架下构造最坏情况的反事实价值函数，并给出可解的优化问题；③通过实际数据（糖尿病状态）估计混淆敏感度，并在不知全局混淆的情况下得到鲁棒策略。

**🔧 技术方法**

使用的技术包括：概率非确定性因果模型、敏感度约束（odds‑ratio），反事实价值函数的递归定义，针对每个时间步独立的凸/非凸优化（用 Gurobi 求解），以及可扩展的梯度下降近似。

**📊 数据集**

采用 Sepsis 治疗仿真环境：状态由四项生命体征（心率、血压、血氧、血糖）组成，动作为三种治疗的开/关，糖尿病为隐藏的全局混淆变量。

**📈 对比分析**

与已知真实混淆敏感度（oracle）下的最优策略和价值进行对比。实验表明：在给定的混淆敏感度估计下，得到的策略在完全观测环境中能显著优于原始子最优策略；估计的最坏情况价值对真实价值略有低估，但对策略的影响较小，尤其在非糖尿病病例中保持鲁棒。

**⚠️ 局限性**

局限性包括：①解耦的优化仅给出保守下界，真实耦合情况可能更严格；②仅考虑全局静态混淆，未处理部分时间步混淆或记忆效应；③梯度下降求解不保证全局最优；④实验仅在单一仿真环境上验证，尚未在大规模真实数据上测试。

---

## 89. DHMark: Public-Key Watermarking for LLM-Generated Text via Diffie-Hellman-Guided Rejection Sampling

**arXiv ID:** 2608.03093 | [PDF](https://arxiv.org/pdf/2608.03093v1)

**作者:** Haocheng Fu `[一作]` (Chinese Academy of Sciences), Yun Cao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Diffie–Hellman引导的拒绝采样的LLM文本公开水印框架，能在不泄露私钥的情况下实现离线第三方验证。

**💡 创新点**

创新点在于将授权payload与噪声文本证据分离：通过短签名注册表授权payload，并将其展开为多条一比特方程，采样时仅弱化对兼容方程的偏好，验证时通过统计聚合这些方程的投票来判断文本是否包含授权水印。

**🔧 技术方法**

技术上结合了公钥加密（Diffie–Hellman标签生成与公钥可验证接口）、分层编码（短payload展开为多条一比特方程）、统计水印（基于投票的检测）以及注册表认证（签名的payload绑定上下文）。

**📊 数据集**

在实验中使用LLaMA‑3 8B公开模型，30条不同提示，生成2000个token的文本，并对各种token级攻击（删除、替换、截断、copy‑paste、恶意后缀）以及错误上下文和普通文本进行评测。

**📈 对比分析**

与传统私钥水印（如green‑list）和公开可检测方案（如PVMark、Fairoze等）相比，DHMARK在默认32位payload配置下在八种攻击下保持0.967–1.000的有效率，错误上下文和普通文本的假阳性率为0；同时每个token平均需要3.24次拒绝尝试，约73%位置成功匹配水印兼容候选，性能开销相对可控。

**⚠️ 局限性**

局限性包括：需要一个具体可验证的公钥标签实现（当前仅给出抽象接口）；评估仅覆盖token级编辑，未测试语义重写、翻译等变换；样本量有限（30条提示），真实误判率难以估计；不提供完整文档完整性保障，需与签名或span定位等技术结合使用。

---

## 90. Separating Intelligence from Inference: A Standard for Edge-Native AI Computing

**arXiv ID:** 2608.02608 | [PDF](https://arxiv.org/pdf/2608.02608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 91. Rubrics as Privileged Information for Open-Ended Generation

**arXiv ID:** 2608.02948 | [PDF](https://arxiv.org/pdf/2608.02948v1)

**作者:** Deepika Bablani `[一作]` (Apple), Wanming Chen `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将自监督蒸馏（On‑policy Self‑Distillation）应用于开放式生成任务，并以医学与科学问答为例，探索如何通过“软”评估 rubrics 作为教师的特权信息来提升模型表现。

**💡 创新点**

创新点在于：①将 rubrics 直接作为教师的软特权信息（RuPI）使用，而非传统的将其转化为稀疏奖励；②通过对比软 vs. 硬（参考答案）PI，证明软 PI 在学生生成轨迹上产生更强、定位更好的 KL 信号；③证明在保持基础能力不损失的前提下，RuPI 能在多模型、多规模上击败基于 RL 的 Rubric‑as‑Reward（GRPO）方法。

**🔧 技术方法**

技术主要包括：On‑policy Self‑Distillation (OPSD)、反向 KL 训练、EMA 软教师、LoRA 低秩微调、密集 per‑token KL 监督。

**📊 数据集**

数据集：HealthBench（医学问答），RubricHub Science（科学问答）与对应评估集 ResearchQA；同时在四项通用基准（MMLU、GSM8K、IFEval、TruthfulQA）做跨域评估。

**📈 对比分析**

与基线（基础模型、RL‑GRPO、参考答案蒸馏、SFT 等）对比，RuPI 在 HealthBench 测试集上提升了 0.01–0.08 级别的 rubric‑satisfaction 分数；在 HealthBench Hard 子集的聚类准确率上提升至 0.53–0.78；在 ResearchQA 上提升 0.09 分；相比 RL‑GRPO，RuPI 只需 8 倍更少的 on‑policy 采样且不依赖外部评判。

**⚠️ 局限性**

局限性包括：仅在两类开放式问答任务验证，未覆盖创意写作等更结构多样的生成任务；使用的 hyper‑parameter（r=64、α=128、μ=0.02、τ=5.0）未做敏感性搜索；Rubrics 质量依赖专家标注，低质量或不一致的 rubrics 可能导致误导；模型仅为研究原型，医学/科学领域仍需人工审核与安全评估。

---

## 92. Earth Embeddings

**arXiv ID:** 2608.03410 | [PDF](https://arxiv.org/pdf/2608.03410v1)

**作者:** Adam J. Stewart `[一作]` (Technical University of Munich), Xiao Xiang Zhu `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文系统阐述了地球嵌入（隐式位置编码、图像块嵌入、像素级嵌入）的生成、存储、评估与实际应用，并给出了案例研究和未来的研究方向。

**💡 创新点**

创新点在于整合并比较了多种嵌入技术与产品，提出统一的可复现性与存储规范，探讨量化与聚合方法，并针对检索、映射等任务给出了基准与评估框架。

**🔧 技术方法**

采用的技术包括自监督预训练（SimCLR、DINOv2、MAE 等）、对比学习、Transformer、ResNet、SIREN、位置编码（RFF、Spherical Harmonics、HEALPix 等）以及图神经网络和生成式哈希等。

**📊 数据集**

使用的数据集涵盖 Sentinel‑2、Landsat、NAIP、Copernicus、Sentinel‑1、PALSAR、ERA5、MODIS、NASADEM、OCO‑2 等公开卫星影像，以及 iNaturalist、Flickr、YLI‑GEO 等地理标签数据。

**📈 对比分析**

通过 GEO‑Bench、PANGAEA、NeuCo‑Bench 等基准和案例实验，比较了不同嵌入在场景分类、农作物/土地覆被映射、灾害预测、检索等任务中的准确率、泛化能力、存储与推理成本；结果表明 GSE、Tessera 在土地覆被任务上表现优异，融合可提升多任务性能，但在跨区域迁移和时间序列变化捕捉方面仍有限。

**⚠️ 局限性**

局限性包括覆盖范围偏向陆地和 RGB/多光谱影像，缺乏海洋、气候、激光雷达等数据；缺乏对不确定性、云覆盖和季节偏差的量化；复现受限于专有模型与数据；基准多样但缺统一标准，时间序列评估不足。

---

## 93. Learning a Vector-Symbolic Model for Socio-Cultural Tasks

**arXiv ID:** 2608.02807 | [PDF](https://arxiv.org/pdf/2608.02807v1)

**作者:** Meera Ray `[一作]` (Pennsylvania State University), Christopher L. Dancy `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于向量符号架构（HRR）与角色自编码器的 ACT‑R 宣言记忆系统，并将其应用于评估种族隐含关联测试（IAT）中社会文化结构对决策的影响。

**💡 创新点**

创新点在于：①将多层语义与自我回忆的向量符号表示统一编码；②使用角色自编码器生成可解释的角色向量，保留语义层级；③将这些向量与 ACT‑R 检索机制耦合，从而在认知模型中显式体现社会文化偏见。

**🔧 技术方法**

核心技术包括：ACT‑R 认知架构、向量符号架构（HRR）、BERT 编码器、角色自编码器（Role‑Former）以及基于相似度的检索与激活公式。

**📊 数据集**

使用的数据集：10 篇黑人女性口述历史访谈文本（训练角色自编码器），以及 104 名黑人/白人参与者的 IAT 反应时数据；对比实验还使用了 BERT 的通用文本语料来评估重建误差。

**📈 对比分析**

通过计算查询向量与记忆块的余弦相似度，进而用 ACT‑R 激活公式预测反应时；模型在白人参与者的 IAT 反应时预测误差约为 30‑50 ms，整体拟合优于仅使用单词向量的对照模型，且在黑人参与者的情境下使用自我记忆向量可逆转这一趋势。

**⚠️ 局限性**

局限性包括：①缺乏与 IAT 参与者对应的自传性记忆；②基于 BERT 的编码可能不具备完整的训练透明度；③反向传播训练的生物可行性存疑；④仅处理文本输入，未考虑连续感知数据；⑤情节记忆被视为静态，未能模拟检索后动态更新。

---

## 94. Speculative Correction: Draft-then-Refine Decoding for Diffusion Language Models

**arXiv ID:** 2608.02625 | [PDF](https://arxiv.org/pdf/2608.02625v1)

**作者:** Brian K Chen `[一作]` (National University of Singapore), Kenji Kawaguchi `[通讯]` (National University of Singapore)

**通讯引用:** 7892 | [OpenAlex ID](https://openalex.org/A5003184366)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了先完整草稿生成后再用双向扩散模型进行全序列修正的推理模式，并在相同模型（flash‑flash）和跨模型（mini‑flash）两种配置下进行评估。

**💡 创新点**

提出了 draft‑then‑refine 解码架构以及无训练的 speculativе correction 方案，证明其能够在保持或提升质量的同时显著加速扩散语言模型推理。

**🔧 技术方法**

采用 LLaDA2.1 扩散语言模型，利用块级自回归生成与全序列双向去噪，结合可调的块大小、去噪步骤、阈值等参数实现两阶段推理。

**📊 数据集**

在 GSM8K、MATH、MBPP、HumanEval 以及合成算术等多任务数据集上进行实验评测。

**📈 对比分析**

与块级自回归基线（包括验证窗口匹配的对照）进行对比，flash‑flash 在 GSM8K 上提升约 5% 准确率且速度提升 1.2×，MBPP 提升 14%；mini‑flash 在 MATH‑384 等任务实现近等价质量但速度提升 2.17×，整体展现 Pareto 前沿。

**⚠️ 局限性**

由于草稿与修正器训练不匹配导致分布不一致，mask‑only 修正效果差异显著；跨模型性能波动且缺乏专门针对草稿/修正的训练目标，需进一步联合训练或专用优化以提升稳定性和质量。

---

## 95. Efficient Video Dataset Distillation via Cluster-Guided Prototype Blending

**arXiv ID:** 2608.03269 | [PDF](https://arxiv.org/pdf/2608.03269v1)

**作者:** Chongle Ren `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种基于教师引导的选择–分配–混合流程的无梯度视频数据集蒸馏方法 ProtoBlend，直接构造压缩后的视频集合。

**💡 创新点**

创新点在于通过教师的时间片段筛选、特征聚类分配以及原型‑锚点像素级混合，彻底消除了迭代梯度优化，显著降低构造成本，同时提升多类多样性。

**🔧 技术方法**

采用了 Teacher‑Guided Temporal Clip Selection、Cluster‑Guided Prototype Allocation、像素级混合与混合源软标签、VideoMAE 预训练教师模型以及 K‑means 聚类等技术。

**📊 数据集**

使用了 MiniUCF、HMDB51、Kinetics‑400 与 Something‑Something V2 四个动作识别基准数据集。

**📈 对比分析**

在相同视频/类（VPC）预算下与随机、Herding、K‑center 等基准以及 DM、FRePo、VDSD、PRISM 等蒸馏方法进行对比，ProtoBlend 在大多数场景中取得最高或第二高的准确率，且构造时间显著低于基于优化的方法。

**⚠️ 局限性**

局限性包括：在大规模、高多样性数据集（SSV2、Kinetics‑400）上增大预算带来的收益有限；混合过程未对时间对齐做细粒度处理，导致对细粒运动表达的效果不如专门针对时序的优化方法。

---

## 96. Screenshots or Tools? Eliciting Tool Use and Managing Multimodal Context in Hybrid GUI-MCP Computer-Use Agents

**arXiv ID:** 2608.03327 | [PDF](https://arxiv.org/pdf/2608.03327v1)

**作者:** Siqi Fan `[一作]` (University of Electronic Science and Technology of China), Weihang Chen `[通讯]` (AI Platform Xiaohongshu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究混合图形界面与文本工具在计算机使用代理中的作用，揭示工具可用性并不等同于工具被使用，研究了工具采用缺口与上下文压缩的影响。

**💡 创新点**

首次发现工具注入对模型性能的符号取决于模型的推理能力，并提出通过强化学习引入稠密工具奖励来单独调控工具采用，而非提升工具调用能力；同时展示了匹配训练和推理观察规则可实现接近无损的令牌成本压缩。

**🔧 技术方法**

多任务强化学习（GRPO）与密集工具奖励、执行成功奖励、窗口深度与截图丢弃规则的上下文压缩、对抗性训练等。

**📊 数据集**

OSWorld-MCP基准（309个桌面任务）以及其工具集120个MCP工具。

**📈 对比分析**

与仅使用GUI的基线模型相比，工具注入使思考型模型准确率提升约4个百分点，指令型模型下降约5.9个百分点；工具采用率分别为23.9%和13.9%；在窗口深度为2且成功后丢弃截图的压缩设置下，匹配训练后输入成本降至原来的53%，峰值上下文下降37%，准确率保持不变。

**⚠️ 局限性**

仅在单一模型骨干内验证符号逆转，工具调用能力仍未提升；强化学习仅能调节采用率，未解决工具调用的语义成功率；压缩收益未能迁移至未训练任务；上下文规则固定，未学习动态压缩；跨操作系统的可扩展性未验证。

---

## 97. A Hierarchical Approach to Imitation Learning for Manipulation Tasks Requiring Time Varying Forces

**arXiv ID:** 2608.03103 | [PDF](https://arxiv.org/pdf/2608.03103v1)

**作者:** Rishabh Shukla `[一作]` (University of Southern California), Satyandra K. Gupta `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种层次化的扩散策略与高速轨迹生成架构（DPA-FTG），实现对需高频力控的双臂材料分离任务的逼真执行。

**💡 创新点**

通过将低频规划迁移到潜在空间的离散词典，并在高速闭环控制层使用可学习的力敏感GRU解码，克服了传统扩散策略的推理延迟与开环执行瓶颈。

**🔧 技术方法**

使用条件扩散模型、向量量化VAE构建动作词典、GRU强化学习、异步线程推理、力/视觉多模态感知及数据驱动的力闭环控制。

**📊 数据集**

基于人工示范的双臂锯割与剥离实验，采集51条包含视觉、位置、力与速度的演示序列，并在不同几何形状（正方形、圆形、八边形）上进行训练，随后在六边形和三角形上测试零样本泛化。

**📈 对比分析**

与标准扩散策略、带力反馈的扩散、Reactive Diffusion Policy、行为克隆等基线进行对比，在真实机器人上验证，DPA-FTG实现100%完成率、100%任务成功率、执行质量平均2.73、峰值力27.7N，显著优于RDP等基线。

**⚠️ 局限性**

在极锐角几何的零样本场景中仍出现潜在匹配与漂移失效，且需要更多演示覆盖与更具空间感知的视觉编码器以提升泛化。

---

## 98. AI Security Leaderboard: Methodology, Results and Minimal Standard

**arXiv ID:** 2608.03070 | [PDF](https://arxiv.org/pdf/2608.03070v1)

**作者:** Jasper Timm `[一作]`, Kellin Pelrine `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对前沿大型语言模型的安全防护进行了大规模测试，提出并评估了 FAR.AI 最低安全标准，并通过一套自动化框架对多家厂商的旗舰模型在 CBRNE 与网络安全等高危领域的越狱攻击进行量化评估。

**💡 创新点**

创新点在于：①构建了 60+ 种常见越狱技术的层级分类与组合框架；②设计了三阶段自动化筛选管道与成本估算方法；③首次公开了针对安全防护的可量化最低标准和对应的排行榜，推动安全评测的透明度与可复现性。

**🔧 技术方法**

使用的技术包括：①基于原语（primitives）构建的变体生成器；②自动评估器（基于 GPT‑5.4‑mini）与人工校验相结合的三重判定；③对模型内部推理轨迹与激活状态的监控设计思路；④通过统计学方法估算随机与专家引导下的越狱成本。

**📊 数据集**

所用数据集为 DeepHarm（180 条技术性高危请求）和 Propensity（180 条意图明确的高危请求），共 360 条攻击目标，覆盖化学、生物、核/放射、爆炸及网络安全五大风险域，并按子域均衡分配。

**📈 对比分析**

评估方法通过在每个模型-域组合上计算“通用越狱成功率”（≥75% 的攻击目标得到合规响应）以及随机搜索与专家引导下的成功次数来进行比较。结果显示 Claude Fable 5 与 GPT‑5.6 Sol 在所有测试域均无通用越狱，Grok 4.5 与 Gemini 3.1 Pro 则在 CBRNE 与网络安全域分别出现 63/18 和 385/231 次通用越狱，表明安全防护存在明显差距。

**⚠️ 局限性**

局限性包括：①化学与爆炸域的攻击目标存在双重使用歧义，导致专家评分一致性低；②评估的误判率（尤其是误报 7.6% 的化学域）说明自动判定仍需改进；③未覆盖动态迭代式越狱与更深层的攻击手段；④成本估算假设攻击者具备本报告的工具，无法给出绝对价格；⑤输出令牌限制可能导致部分攻击结果被截断。

---

## 99. Quota and population monotonicity across house sizes are incompatible for apportionment to four states

**arXiv ID:** 2608.02759 | [PDF](https://arxiv.org/pdf/2608.02759v1)

**作者:** Lav R. Varshney `[一作]` (Stony Brook University), Lav R. Varshney `[通讯]` (Stony Brook University)

**通讯引用:** 7384 | [OpenAlex ID](https://openalex.org/A5065423139)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明在不假设匿名性、同质性、秩保序等常规正则条件下，任何四个州的确定性分配规则都无法同时满足配额约束和人口单调性。

**💡 创新点**

通过构造有限逻辑装置（Boolean 变量编码与桥接），消除了此前对五个州的必要性，首次在仅四州的情形下完成不相容性证明，并将结果与受限量化器（constrained quantization）几何框架关联。

**🔧 技术方法**

采用了离散逻辑推理（transfer lemma、布尔变量同步与循环矛盾）、组合几何（根格与单纯形层）以及对称性变换（坐标置换）等技术手段。

**📊 数据集**

未使用任何实验数据集，整个工作完全基于理论构造与证明。

**📈 对比分析**

此论文不涉及算法实现或实验对比，故无性能指标；其贡献在于提供了一个无条件的不可行性结论，无法与其他方法进行数值比较。

**⚠️ 局限性**

局限性包括：仅适用于不同议席规模比较；不涵盖每个州至少获得一席的约束；结论仅为最坏情况的存在性不可能性，未给出频率或实际影响的评估。

---

## 100. Double Descent in Gradient Boosting Decision Trees via Split-Candidate Scaling

**arXiv ID:** 2608.03111 | [PDF](https://arxiv.org/pdf/2608.03111v1)

**作者:** Ryuichi Kanoh `[一作]` `[通讯]` (University of Electro Communications), Ryuichi Kanoh (University of Electro Communications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究梯度提升决策树（GBDT）在分裂候选数量扩展下的容量控制，并揭示其导致的双重下降现象。

**💡 创新点**

提出分裂候选数量作为GBDT单一容量轴，并通过树、特征和核视角阐释其产生双重下降的机制。

**🔧 技术方法**

使用树结构分析、路径特征表示、固定核诊断（KGD/KRR）、CatBoost、XGBoost、LightGBM等GBDT实现，并进行实验验证。

**📊 数据集**

主要使用加州房价数据集（California Housing）以及18个仅含数值特征的TabularBenchmark回归数据集。

**📈 对比分析**

通过比较不同GBDT实现的测试RMSE、固定核的KGD/KRR参考误差、训练时间和内存等，验证在中间候选预算下出现测试误差峰值，且深度、样本量和标签噪声会影响峰值位置；与随机森林对比表现为单调下降，说明提升算法的作用。

**⚠️ 局限性**

局限在于只在经验层面验证，未给出完整理论证明；峰值位置对数据集几何高度敏感；候选字典扩展本身不足以产生双重下降，需与提升算法交互；计算成本随候选数增长显著。

---

## 101. PFM-HR: Pose Flow Matching for Humanoid Robots

**arXiv ID:** 2608.03227 | [PDF](https://arxiv.org/pdf/2608.03227v1)

**作者:** Yukang Gao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于流匹配的可复用姿态先验 PFM-HR，并在此基础上定义 Pose Geometry Score 用于调节强化学习中的跟踪奖励。

**💡 创新点**

创新点在于：1) 用无序姿态数据直接预训练流匹配去噪器，避免了姿态距离监督和时间序列需求；2) 通过去噪器雅可比对政策产生的姿态变化进行几何评分，冻结先验即可在多任务中使用；3) 以奖励调节方式将几何评分融入训练，提升动态动作的学习效率和准确性。

**🔧 技术方法**

主要技术包括流匹配（Flow Matching）、去噪器雅可比计算、Pose Geometry Score、奖励调节、强化学习（ADD+PDF-HR）以及多任务训练框架。

**📊 数据集**

使用了 60M 规模的无序姿态数据集 BONES-SEED 进行预训练，基准评估使用 LaFAN1、MimicKit 以及 BeyondMimic 部署数据集。

**📈 对比分析**

与 ADD、ADD+PDF-HR 以及 SMP 进行对比；在单轨迹跟踪、通用运动跟踪以及实测部署中，PFM-HR 在样本效率上提升 7–30%，位置/旋转误差比基线低 5–10%，尤其在高动态动作上优势显著。

**⚠️ 局限性**

局限性包括：只能捕捉姿态的局部协方差，无法反映时间动态、方向或顺序；对训练分布外姿态的指导效果有限；未来需探索时序先验和符号敏感的评分机制。

---

## 102. CVPO: Enhancing LLM Reinforcement Learning Reasoning via Value-Variance Adaptation and Dynamic Curriculum Learning

**arXiv ID:** 2608.03068 | [PDF](https://arxiv.org/pdf/2608.03068v1)

**作者:** Ziqi Jia `[一作]` (AI Cloud Group, Baidu), Yanpeng Wang `[通讯]` (AI Cloud Group, Baidu)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型语言模型的推理能力提升，提出了 CVPO 方法，通过在响应轨迹层面引入基于值方差的优势函数调整机制，并在问题层面采用动态课程加权方法，解决了反馈精度不足和难度漂移问题。

**💡 创新点**

创新点在于：① 将轨迹值方差与探索强度关联，并证明其对策略梯度上界的影响；② 设计了差异化的方差权重函数，使正确样本抑制高方差、错误样本鼓励高方差；③ 引入基于贝叶斯推理的动态课程加权，实时适应模型对不同难度题目的掌握程度，从而缓解难度漂移。

**🔧 技术方法**

技术主要包括：强化学习中的 Proximal Policy Optimization（PPO）与价值网络、Monte Carlo 回报与 GAE、方差归一化的优势修正、贝叶斯后验更新的动态课程权重、以及对长链推理任务的 KL 正则化去除。

**📊 数据集**

使用 DAPO-Math-17k 作为训练集，评估数据集包括 AIME 2024/2025、AMC 2023/2024、MATH-500 等数学推理 benchmark。

**📈 对比分析**

与 GRPO（ASYN）和 VAPO 等现有方法对比，CVPO 在所有测试集上均取得显著提升（例如 AIME24 提升 9.3%，AMC23 提升 25.2%），并通过 ablation 证明方差调节和动态课程各自贡献显著；训练曲线显示 CVPO 在 1500 步后持续提升，优于竞争对手。

**⚠️ 局限性**

局限性包括：对方差权重系数的选择敏感，过大或过小会导致收敛失衡；动态课程权重需要预先设定贝叶斯超参数，可能对不同任务不够通用；方法主要针对数学推理任务，其他领域的泛化能力尚待验证。

---

## 103. HyperFL: Query-Adaptive Representation Learning for Software Fault Localization

**arXiv ID:** 2608.02967 | [PDF](https://arxiv.org/pdf/2608.02967v1)

**作者:** Shuai Shao `[一作]` (University of Connecticut), Tingting Yu `[通讯]` (University of Connecticut)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种查询自适应的检索框架HyperFL，用于软件缺陷定位。

**💡 创新点**

利用轻量级超网络为每个缺陷报告动态生成LoRA参数，实现查询特定的编码适配，而代码编码保持共享。

**🔧 技术方法**

结合超网络、LoRA、双编码器结构以及InfoNCE对比学习训练。

**📊 数据集**

在基于GitHub Issue的真实世界定位基准（含多仓库多样化报告）以及SWE-bench Lite上进行实验。

**📈 对比分析**

与BM25、Jina-Code、CodeRankEmbed、Qwen3、SweRank等基线对比，HyperFL在函数级MRR@10提升约13%~17%，Hit@1提升约16%，在多种预训练检索后端均显著优于对手。

**⚠️ 局限性**

当缺陷报告已被预训练模型充分表达时提升有限，并且对训练数据中的噪声非常敏感。

---

## 104. Quo Vadis, World Modeling?

**arXiv ID:** 2608.02713 | [PDF](https://arxiv.org/pdf/2608.02713v1)

**作者:** Yu Yang `[一作]`, Shuicheng Yan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出“Agent‑Centric World Proxy”（ACWP）概念，阐述其与传统世界模型的区别，并系统梳理六类代理功能（动态、空间、执行、记忆、技能、奖励/验证）以及三层赋能级别（推理时指导、训练时优化、代理‑代理协同进化）

**💡 创新点**

将世界建模从单纯的未来状态预测转变为面向智能体的可用信息交互，强调代理需具备环境根基、可控性、可行动信息收益、可扩展性和前瞻性等关键属性

**🔧 技术方法**

理论框架与概念模型，结合已有技术（NeRF、视频预测、模型预测强化学习、程序/工具执行模拟、经验记忆、奖励模型/评估器等）来说明实现方式

**📊 数据集**

无新实验数据集，主要引用现有公开系统与技术作为实例示例

**📈 对比分析**

未进行实验评估；论文通过对比已有方法的功能定位与优劣、示例对话来说明ACWP在不同层级的优势，并未给出数值性能指标

**⚠️ 局限性**

关键局限包括：代理的真实度与不确定性难以保证；代理可信度判定机制缺失；奖励/验证代理可能引发奖励劫持与安全风险；缺乏面向代理使用的标准评测与基准

---

## 105. Global Graph-Validated Optimization for VLM-based 3D Indoor Scene Generation

**arXiv ID:** 2608.03064 | [PDF](https://arxiv.org/pdf/2608.03064v1)

**作者:** Jialu Huang `[一作]` (Xi’an Jiaotong University), Zheng Dang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于场景图验证和混合搜索的3D室内布局生成框架，能够从自然语言指令中生成语义一致且物理可行的室内场景。

**💡 创新点**

创新点包括：1）Global Semantic Verification（GSV）通过规则化的场景图验证实现全局语义一致性；2）Global Physical Feasibility Search（GPFS）融合进化搜索与梯度细化的混合优化，克服梯度下降易陷入局部最优的缺陷。

**🔧 技术方法**

主要技术包括：场景图构建与图验证、规则化语义约束、进化搜索（swap‑based crossover 与 center‑guided mutation）、梯度优化、VLM（GPT‑4o/4.1）与CLIP 资产检索。

**📊 数据集**

使用的公开数据集有：3D‑FUTURE、Objaverse（地面资产），HSSD‑200（家具资产），以及用于评估的 LayoutVLM 数据集（11种房间类型共33个场景）。

**📈 对比分析**

与 LayoutGPT、Holodeck、I‑Design、LayoutVLM 等最新方法在 CF、IB、Pos、Rot、PSA 等指标上对比，取得最高分，尤其在物理可行性（CF/IB）和语义一致性（PSA）方面表现显著优于基线。

**⚠️ 局限性**

主要限制包括：1）对 GPT API 的随机性高度依赖，导致结果不稳定；2）3D 资产质量参差不齐，存在尺寸过大或渲染失效的模型；3）仍需进一步降低管线随机性并构建更高质量的资产库。

---

## 106. Traceable Multi-Agent System for Knowledge-Based Forecasting

**arXiv ID:** 2608.03339 | [PDF](https://arxiv.org/pdf/2608.03339v1)

**作者:** Junhyeok Kang `[一作]` (LG AI Research), Soonyoung Lee `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建 TraceMAS，展示多智能体如何将文本知识、数据映射与因果图结合，迭代生成可追踪的原油价格预测模型。

**💡 创新点**

创新点在于引入 Ideal Causal Loop Diagram 与 Data‑Grounded Causal Loop Diagram 两层因果图，作为共享可追踪的中间表示，显式记录从文档证据到数据选择、特征构造及模型调整的完整路径。

**🔧 技术方法**

使用 LLM 驱动的多角色智能体（Domain Analyst、Causal Analyst、Data Engineer、Crawler、Risk Reviewer、Model Engineer），因果循环图结构、版本化追踪日志以及基于因果假设的特征工程。

**📊 数据集**

数据集包括原油市场报告（PDF/Docx）、企业内部时序数据库、外部指标（如 GPR 指数、OPEC+ 产量数据、进口 PMI 等）以及相应的代理生成的中间结果。

**📈 对比分析**

通过在每个迭代版本中对比因果图、特征映射、模型架构和预测结果来评估改进；虽然未提供绝对预测性能基准，但演示展示了模型精度的提升与可解释性的增强。

**⚠️ 局限性**

限制在于演示阶段缺乏大规模实测，仍需人工检查代理日志；因果图的构建和数据映射依赖于人工或半自动提取，可能导致主观性和覆盖不足。

---

## 107. EditFlow3D: Automated Local Editing of 3D Assets with Trajectory Preservation

**arXiv ID:** 2608.03179 | [PDF](https://arxiv.org/pdf/2608.03179v1)

**作者:** Rui Nie `[一作]` (Beihang University), Qian Yu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练、无逆向的 3D 资产局部编辑框架 EditFlow3D，能精准完成部件替换、删除、修改等多种编辑。

**💡 创新点**

创新点包括：①基于 VLM 的自动化编辑控制生成，既得到视觉指导图像又得到精细 3D 掩码；②差分流导向（DFG）与 3D 掩码相结合实现局部结构更新；③轨迹保持导向（TPG）在生成过程中软约束未编辑区域，避免边界破坏。

**🔧 技术方法**

主要技术栈为：Qwen3-VL-8B+FLUX.1 生成视觉指导；P3-SAM、PartField、S2AM3D 进行掩码构造与精细化；差分流与轨迹保持实现编辑；使用 TRELLIS/TRELLIS.2 作为 3D 生成基底。

**📊 数据集**

使用 Edit3D-Bench 及新构建的 EditFlow-Bench（100 个 3D 资产、200 个编辑案例，涵盖增删替换、几何与外观改动）。

**📈 对比分析**

与 Nano3D、PartFlow、VoxHammer、Vinedresser3D 以及直接重建基线对比，评估指标包括 CLIP 相似度、CD/PSNR/SSIM/LPIPS/FID 等，实验表明 EditFlow3D 在目标对齐和非目标保留两方面均优于基线，并获得用户研究中最高偏好度。

**⚠️ 局限性**

局限性：仍依赖 VLM 生成控制，生成质量受预训练模型限制；目前仅在 TRELLIS/TRELLIS.2 体系上验证，缺乏对其他 3D 生成模型的通用性研究。

---

## 108. Non-Destructive Quantification of Urea Adulteration in Bovine Milk Using Transmittance Multispectral Imaging

**arXiv ID:** 2608.03113 | [PDF](https://arxiv.org/pdf/2608.03113v1)

**作者:** Sharukshan Niranjan `[一作]` (University of Peradeniya), Janak Vidanarachchi `[通讯]` (University of Peradeniya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文利用在可控密度平衡条件下的多光谱透射成像系统，对牛奶中尿素掺假进行非破坏性定量检测；

**💡 创新点**

创新点在于将12个离散波段的多光谱成像与线性回归及前馈神经网络相结合，构建了低成本、快速且实验室验证的尿素定量模型；

**🔧 技术方法**

采用了自制多光谱成像系统（12个窄带LED、CMOS相机）和机器学习方法（岭回归、两层前馈神经网络）进行特征提取与回归；

**📊 数据集**

使用了80份牛奶样本（10个尿素浓度水平，每个水平8份复本）构成的多光谱数据集，并在训练/测试/验证三组上评估模型；

**📈 对比分析**

与线性回归相比，前馈神经网络在训练、测试和验证集上的R²分别为0.9891/0.9815/0.9773，RMSE分别为0.1753/0.2286/0.2322，显示出更好的预测精度；

**⚠️ 局限性**

主要局限在于实验仅在实验室控制密度平衡的牛奶样本上验证，未涉及不同来源、采样时间、存储条件的天然样本，且模型需进一步在实际供应链环境中检验和优化。

---

## 109. TabletCraft: Bridging a 4,000-Year Cultural Gap with Bidirectional Akkadian NMT and Cuneiform Rendering

**arXiv ID:** 2608.02609 | [PDF](https://arxiv.org/pdf/2608.02609v1)

**作者:** Zhaohui Wang `[一作]` (University of Southern California), Zhaohui Wang `[通讯]` (University of Southern California)

**通讯引用:** 28695 | [OpenAlex ID](https://openalex.org/A5100358029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了开源工具 TabletCraft，支持英文↔阿卡德语双向机器翻译，并将阿卡德语转为楔形文字并渲染成陶板。

**💡 创新点**

创新点在于实现了古代楔形文字的双向交互——既可阅读又可创作，并对输出做了可视化渲染与文化敏感标注。

**🔧 技术方法**

使用 ByT5-base 作为字节级翻译模型，结合查找表式楔形符号映射和 SVG/PNG 渲染器实现完整流水线。

**📊 数据集**

训练数据主要来自 Akkademia 平行语料（58k句），再补充 1.5k 共享任务和 6k 句子对齐扩充，构成 116k 双向样本。

**📈 对比分析**

在 Akkademia 验证集上，阿卡德语→英语 BLEU 49.1、英语→阿卡德语 BLEU 48.5，显著优于此前 37.5 的单向系统，说明双向训练有效且可生成接近原文的楔形文本。

**⚠️ 局限性**

局限在于输出仅为近似现代转写，单参考 BLEU 低估真实质量，且对古代方言/文本体裁的适应性有限，仅支持英语输入。

---

## 110. ZK-SR117: A Chunked Zero-Knowledge Attestation Design for Aggregated Fair-Lending Metrics, with a Control Mapping toward Full SR 11-7 Coverage

**arXiv ID:** 2608.02664 | [PDF](https://arxiv.org/pdf/2608.02664v1)

**作者:** Mohammad Nasir Uddin `[一作]` (Westcliff University), Asaduzzaman Anik `[通讯]` (Stanton University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并演示了基于零知识证明的银行模型合规证明框架，使用分块聚合统计的 zkSNARK 电路，对 2022 年 HMDA 贷款申请数据进行公平性（人群公平间距）和校准误差的证明，满足 SR 11‑7 监管要求。

**💡 创新点**

创新点包括：①将监管控制语言映射为可零知识证明的断言；②提出分块（chunked）聚合统计证明设计，解决大规模样本下的数值溢出和编译瓶颈；③首次在实验中识别并通过公开的预处理规范解决数据质量缺陷；④引入统计决策框架，将零知识证明与监管阈值的假设检验相结合。

**🔧 技术方法**

使用技术：zkSNARK（EZKL、Halo2/PLONK），KZG 提交，nonce‑controlled 随机抽样，FairZK 风格聚合统计，Bootstrap 置信区间分析，随机数 beacon，安全协议威胁模型。

**📊 数据集**

数据集：2022 年美国 HMDA 贷款申请注册表（过滤后 409,905 行），采用四个特征的逻辑回归模型（收入、贷款额、负债收入比、贷款价值比）。

**📈 对比分析**

比较方法：在相同模型和数据下，对比三种电路设计（平面求和、树形归约、分块）；分块设计实现 N=32,768 的批量证明，单块证明约 4 秒，总证明 24.8 MB，统计误差 0.00289；平面求和在 N>8,192 时溢出，树形归约因编译设置无法完成。相较于传统文档交换，验证成本几乎瞬时，且保持模型和数据的保密性。

**⚠️ 局限性**

局限性：仅演示两项监管控制（人群公平、校准误差）和单一简单模型；未实现树形归约的递归压缩、GBT/MLP 等复杂模型；nonce‑protocol 仍为概念化，未在真实监管环境中部署；数据预处理阈值的公平性影响尚未完整评估；抽样设计与 Bootstrap 置信区间不完全匹配；缺乏对跨块比较（如 AUC、KS）的证明方案。

---

## 111. Surface Keypoint Representation for Multi-Object and Articulated Human-Object Interaction Generation

**arXiv ID:** 2608.03158 | [PDF](https://arxiv.org/pdf/2608.03158v1)

**作者:** Xiaogang Peng `[一作]`, Huaizu Jiang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出利用表面关键点轨迹来表示对象运动，并结合空间-时间接触距离场，构建三阶段分解的文本驱动全身人机交互生成框架。

**💡 创新点**

创新点在于：①将对象运动压缩为少量非共线表面关键点轨迹，天然支持多对象和多关节结构；②设计全身多对象可变数量的接触距离场，提供细粒度、时变的接触指引；③将生成过程拆分为对象运动生成、接触预测、人体运动合成三步，使每个子任务更易训练且可扩展。

**🔧 技术方法**

采用基于扩散模型的文本与几何编码器；利用Kabsch算法从关键点恢复刚体变换；在全身层面使用SMPL‑X表面标记；引入接触距离场作为中间监督；最后进行接触优化以减少穿模和脚漂。

**📊 数据集**

使用 ParaHome、HIMO、ARCTIC、OMOMO 四个公开基准，分别覆盖多对象、关节对象、单对象等多种交互场景。

**📈 对比分析**

与 HOI‑Diff、CHOIS、ROG、HOIDiNi（单对象）以及 HIMO‑Gen（多对象）等方法对比，实验表明在 FID、R‑Precision、接触准确率、物体平滑度等指标上均实现或逼近 state‑of‑the‑art 级别；在多对象与关节对象任务中优于现有单独模型。

**⚠️ 局限性**

局限性包括：第 III 阶段的接触优化耗时约 4 分钟/124 帧，难以实现实时；仍存在脚漂、轻微穿模等物理伪影，且对不同对象尺度的鲁棒性仍需进一步提升。

---

## 112. HomoEnsNER: Does Language Alignment Outperform Architectural Complexity in Gujarati Named Entity Recognition?

**arXiv ID:** 2608.03105 | [PDF](https://arxiv.org/pdf/2608.03105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 113. Security-First Evaluation of Text-to-Terraform: Benchmarking LLMs and SLMs for Secure IaC Generation

**arXiv ID:** 2608.02672 | [PDF](https://arxiv.org/pdf/2608.02672v1)

**作者:** Francis Luis Santos Vargas `[一作]` (Universidade Federal do Pampa), Diego Kreutz `[通讯]` (Universidade Federal do Pampa)

**通讯引用:** 6149 | [OpenAlex ID](https://openalex.org/A5088960174)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在云基础设施即代码（IaC）生成中，对比评估了七个大型语言模型（LLM）和四个可本地部署的开源模型（SLM），通过双重扫描器（Checkov 与 Trivy）在 17 个 AWS Terraform 场景下进行安全与语法合规性测试。

**💡 创新点**

提出了“安全优先”基准：同时考量语法有效性、Terraform 计划可执行性与安全合规性，并在评估中引入详细安全提示、少量示例提示与扫描器反馈修正三种策略，首次证明语法与安全并非正相关，且仅靠提示工程不足以确保安全。

**🔧 技术方法**

技术包括 GitLab CI/CD 自动化流水线、OpenAI/Anthropic/Google REST API 调用、llama.cpp 服务器部署、Terraform 1.7.5 与 AWS Provider 5.100.0、Checkov ≥3.2.0 与 Trivy 0.69.3 等工具；利用 pass@5、Wilson 置信区间、卡方检验与 Cramér’s V 进行统计比较。

**📊 数据集**

使用 17 个精心挑选的 AWS 资源（S3、KMS、VPC）场景，来源于 IaC-Eval 数据集，并通过 LLM 共识筛选确保安全相关性；同时采用 5 次独立运行保证评估鲁棒性。

**📈 对比分析**

比较方法：在相同场景、相同提示层级下对 7 个模型执行 5 次生成，计算语法通过率、计划通过率、Checkov 与 Trivy 合规率；通过显著性检验比较模型差异。结果显示：Claude Opus 4 在 L3 详细安全提示下取得 23.1% Checkov 合规率和 92.5% Trivy 合规率；WizardCoder 33B 语法通过率 77.8% 但 Checkov 合规率 0%；SLM 在安全提示下合规率几乎为零，提示深度往往降低语法通过率。

**⚠️ 局限性**

局限性包括：场景数量有限且可能偏向模型训练集；使用的是单一版本模型与 4-bit 量化；只评估 Terraform，未覆盖其他 IaC 语言；扫描器仅进行静态分析，无法捕捉运行时安全问题；自我修正仅做单轮，未探索多轮迭代；统计显著性受低样本模型（如 CodeLlama、Magicoder）影响。

---

## 114. Self-Supervised Representation-Guided Generative Dataset Distillation

**arXiv ID:** 2608.03218 | [PDF](https://arxiv.org/pdf/2608.03218v1)

**作者:** Mingzhuo Li `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SRG，一种利用预训练自监督学习（SSL）表示引导的扩散生成框架，用于在冻结的 SSL 编码器下进行数据集蒸馏，生成可直接用于线性探针训练的合成样本。

**💡 创新点**

创新点在于将 SSL 表示空间中的类原型与三种目标（原型对齐、类间判别、类内分配）融合进分阶段扩散引导，兼顾视觉真实性与表示层次结构，并通过阶段性从潜在空间到 SSL 空间的切换实现高效蒸馏。

**🔧 技术方法**

采用 Diffusion Transformer 与 VAE 生成器、spherical K-means 生成 SSL 原型、预训练 SSL 编码器（如 DINOv2、CLIP 等）作为引导信号，并实现两阶段的潜在空间与 SSL 空间双重引导。

**📊 数据集**

实验使用 ImageNet‑1K 及其子集（ImageNet‑IDC、ImageNet‑100、Fine‑grained 子集 Woof、Fruits、Instruments）进行蒸馏，并在 ImageNet‑100 上进行跨编码器评估。

**📈 对比分析**

与随机采样、邻近样本、DiT、MGD3、IGD、LGM、CLP‑DD 等基线对比，SRG 在 IPC=1、3、5 的线性探针准确率均优于或相当于生成式基线，且在 IPC 较大时与优化型方法竞争。

**⚠️ 局限性**

局限性在于仅基于 ImageNet 预训练的扩散生成器，难以直接迁移到非 ImageNet 领域；同时对 GPU 资源的需求仍高，尤其在大规模数据集上需要额外的预处理和存储成本。

---

## 115. ATFlash: Per-RoPE-Wavelength Attention Windows for Compute/Memory-Efficient LLM Inference

**arXiv ID:** 2608.02947 | [PDF](https://arxiv.org/pdf/2608.02947v1)

**作者:** Shun-ichiro Hayashi `[一作]` (Nagoya University), Takahiro Katagiri `[通讯]` (Nagoya University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 RoPE 频率分解的 per‑RoPE‑wavelength distance window，对注意力中的每一频率对设置可变距离窗口，按窗口裁剪 QK 内积，称为 ATFlash。

**💡 创新点**

创新点：① 以 RoPE 的波长为自然阈值构建窗口，保证每个频率对只保留能区分位置信息的范围；② 通过窗口切片实现对 FA‑class 内核的无缝插入，保持硬件原有张量核心填充率；③ 给出闭式对 QK 术语裁剪率的对数表达，输入无关，可预估。

**🔧 技术方法**

使用的技术：RoPE 频率对分解、距离窗口裁剪、FA‑class（FlashAttention‑4、FlashInfer）在线 softmax 变体、基于 MMA 的切片实现、闭式分析、CUDA 端口。

**📊 数据集**

评测数据集：LongBench‑v2、RULER、MRCR、LongCodeQA、∞Bench、Retr.KV、GSM8K、OpenAI‑MRCR、LongCodeQA 等长上下文和检索基准。

**📈 对比分析**

对比方法：全注意力、滑动窗口、MInference、滑动/熵窗口；在 RTX PRO 6000 上测得 QK 裁剪率 37–48%（k=2），全注意力 top‑1 96–98%，KL 1e‑3；速度提升 1.29×（128K）至 1.31×（1M），预填/解码端口均优于原版。

**⚠️ 局限性**

局限性：窗口固定，无法自适应输入；仅裁剪 QK，KV 侧未压缩；实验仅在 sm_120/GDDR7 平台；对检索式任务（如 Retr.KV）k=2 时会出现下降，需调高 k；理论减速率为术语计数近似，实际时间受非注意力开销影响。

---

## 116. Revisiting TD Target Aggregation under Uncertainty in Q-Learning

**arXiv ID:** 2608.03069 | [PDF](https://arxiv.org/pdf/2608.03069v1)

**作者:** Lipeng Zu `[一作]` (Florida State University), Xiaonan Zhang `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在深度 Q 学习中引入一阶后向预测模型，利用其提供的下一步动作结果来重新排序最大化操作，从而构造更稳健的 TD 目标，降低因 Q 估计噪声导致的过估计和学习不稳定。

**💡 创新点**

创新点在于：①将短期（仅一步）环境动力学预测嵌入到 TD 目标的聚合步骤，既不改动 Q 函数结构也不改动原有的 bootstrap 机制；②提出混合 Bellman 操作，理论证明其在点wise 上不比标准最大化更乐观，并在模型误差趋零时收敛到最优 Bellman 目标；③在分布式 Q 学习框架下保持一致性。

**🔧 技术方法**

使用深度 Q 网络（DQN、Double、Dueling、Distributional 变体如 C51、QR-DQN、IQN、CoAct）以及一个轻量级的一阶动力学模型（向量空间下的 Gaussian 预测与图像空间下的简化 RSSM）。核心技术包括混合 TD 目标（α-混合）、模型预测损失（MSE/对数似然）与动态模型的自适应更新频率。

**📊 数据集**

评估数据集包括：经典控制任务（Acrobot、Cartpole、LunarLander、BitFlip）、真实世界场景（CityFlow 交通信号控制、O-Cloud 资源分配）以及 Atari100K 视觉游戏基准。

**📈 对比分析**

与多种基线（DQN、Double DQN、Dueling DQN、BDQN、SUNRISE、C51、Rainbow、QR-DQN、IQN、CoAct 等）进行对比。实验结果表明：SADQ 在所有任务上均实现更平稳的训练曲线、更高的最终回报，Atari100K 上平均/中位数得分均提升（如 QR-DQN 平均 +1.32、IQN +0.86、CoAct +1.54）。

**⚠️ 局限性**

局限性：①依赖辅助动力学模型，模型收敛不足时会产生噪声干扰；②额外的模型训练和推理会带来计算和内存开销，尤其在大规模视觉任务中显著；③目前仅验证在基于值的 Q 学习框架，尚未在策略梯度或 actor‑critic 方法中推广。

---

## 117. Chat Debugging: An Exploratory Study of Human-AI Collaboration to Debug Analog Circuits

**arXiv ID:** 2608.02955 | [PDF](https://arxiv.org/pdf/2608.02955v1)

**作者:** John Hu `[一作]` (Oklahoma State University), Andrew Ash `[通讯]` (Oklahoma State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在两学期本科实验课程中，探索并评估了公开域大语言模型（如ChatGPT‑4o、Gemini 1.5 Pro）在无领域微调情况下帮助学生调试入门模拟电路的有效性；通过对学生的聊天记录与考试表现进行主题分析，揭示了学生的图像使用模式、LLM在根因识别与建议上的表现及其局限。

**💡 创新点**

首创系统性评估现成LLM在模拟电路调试中的零样本支持，发现学生普遍采用图片上传来描述电路，LLM能在多项建议中准确定位根因，但在三维结构识别和自信度方面存在明显缺陷，并首次从教育视角讨论如何利用LLM与强调基础知识和批判性思维相结合。

**🔧 技术方法**

使用大语言模型（ChatGPT‑4o、Gemini 1.5 Pro）进行对话式调试；利用质性主题分析方法对聊天日志进行编码；实验采用本科期末定时实验考核与问卷问答相结合的混合研究设计。

**📊 数据集**

来自两学期ECEN 3314实验班的17名自愿参与学生的 30 分钟定时调试记录与对应聊天日志；包含六个预设故障电路（P1–P6）及其图像与文本信息。

**📈 对比分析**

通过对聊天日志与学生最终答卷进行主题分析，评估LLM建议的准确性、根因识别率、错误类型与自信度。结果显示LLM在零样本根因推荐上准确率较高，但在图像识别和三维推理方面错误频发，且过度自信的表述往往误导学生。

**⚠️ 局限性**

样本量有限（仅17名自愿参与者），且偏向自选使用LLM的学生，影响结果的普适性；缺乏严格的定量评估指标；LLM在视觉识别与3D空间推理方面的不足限制了其在实际硬件调试中的应用。

---

## 118. MeloCodec: Harnessing Melodic Priors for High-Fidelity Singing Voice Representation

**arXiv ID:** 2608.03021 | [PDF](https://arxiv.org/pdf/2608.03021v1)

**作者:** Yizhong Geng `[一作]` (Beijing University of Posts and Telecommunications), Wei Chen `[通讯]` (Li Auto)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出MeloCodec双流神经音频编解码器，利用显式的旋律先验实现高保真歌唱音频表示，并通过“Tokenize-then-Fuse”范式与两阶段训练解决融合时的梯度冲突和码书坍塌问题。

**💡 创新点**

首次将旋律先验离散化为量化子令牌，并在预训练阶段锁定结构，再与声学特征融合；通过信息瓶颈和两阶段训练避免了“Shortcut Learning”，显著提升低比特率下的音调一致性与可控性。

**🔧 技术方法**

采用Chromagram提取、卷积旋律/声学编码器、残差向量量化(RVQ)、多尺度STFT损失、辅助旋律重建损失、对抗学习等技术，构建可离散化的旋律-声学融合体系。

**📊 数据集**

使用5,500小时的中文歌唱语料库进行训练，评估基准为公开的Opencpop歌唱数据集。

**📈 对比分析**

与EnCodec、DAC、X-Codec及MuCodec等基线对比，MeloCodec在6.0 kbps下ViSQOL从3.97提升至4.08、F0‑RMSE从1.23降至0.96；在1.5 kbps下F0‑RMSE从4.12降至1.64，Chroma‑Sim从0.84提升至0.95，MUSHRA主观得分亦领先。

**⚠️ 局限性**

对绝对音高的控制仍受声学分支限制，导致在大幅度音高移位时RMSE升高；目前仅考虑旋律先验，未涵盖节奏、动态等其他音频先验；对不同语言或低资源场景的泛化尚需进一步验证。

---

## 119. TraceCAD: Trace-Guided Repair for Agentic CAD Generation

**arXiv ID:** 2608.03062 | [PDF](https://arxiv.org/pdf/2608.03062v1)

**作者:** Fengxiao Fan `[一作]` (Zhejiang University), Peng Du `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为可执行 CAD 生成系统引入了一层持久化恢复机制，记录特征状态、执行日志、失败证据以及修复结果；通过诊断错误步骤、在局部依赖区域内搜索受限修补方案，并将成功的修复迁移为可重用的修复技能；在 DeepCAD 基准上对 200 模型和 1K 模型进行了评估，并与 Text2CAD、CADCodeVerify、CADDesigner 等方法进行对比。

**💡 创新点**

创新点在于：
1) 持久化恢复状态（Feature + Step 记录）让修复过程可追踪、可复用；
2) 局部修补搜索结合依赖图与受限编辑窗口，避免全局重写导致的无效修复；
3) 通过执行与视觉验证的双重门控，将修复候选转化为可重用的“修复技能”，并记录其成功/失败统计；
4) 在修复过程中保留所有失败和成功的证据，供后续任务检索与优化。

**🔧 技术方法**

使用技术包括：
- 大语言模型（Gemini‑3.1‑pro、Gemini‑3‑flash）进行文本、图像融合的需求解析与代码生成；
- SimpleCADAPI 接口与 RAGFlow 进行动态 API 取证；
- 结构化的执行、视觉与形状差异（shape‑delta）评估；
- 诊断模块基于执行报错、视觉误差与步骤依赖图定位错误；
- 局部搜索采用受限编辑窗口、候选生成与执行验证；
- 修复技能存储与检索机制（signature、上下文、成功率统计）。

**📊 数据集**

使用数据集：
- DeepCAD（拆除重复后保持训练/测试分离）作为 200‑模型与 1K‑模型的基准；
- SkexGen 作为训练/测试拆分的参考；
- 通过 GPT‑5.5 从渲染图生成纯文本描述，避免原始注释与模型不匹配。

**📈 对比分析**

比较方法：
- 与 Text2CAD、CADCodeVerify、CADDesigner 在同一 1K‑模型基准上使用相同后端模型与评估脚本进行对比；
- 在 200‑模型子集上进行 ablation（无持久化状态、无局部搜索、无视觉反馈、无技能库、冷启动/热启动）。
- 结果显示：
  * 复合恢复层的 IoU 达到 0.3837（相较 CADDesigner 0.3610 最高），CD 与 HD 也明显下降；
  * Recovery Score 最高，修复范围最小，技能重用率高；
  * 热启动进一步提升效率，平均重试次数下降，Token 与延迟降低。

**⚠️ 局限性**

局限性：
- 诊断和局部搜索依赖于代码结构良好，若代码分解不清晰会导致错误定位困难；
- 视觉与形状差异评估基于模型推断，缺乏正式几何验证；
- 受限候选导致的技能可能过于专一，重用受限；
- 在线冷启动下的技能累积对任务顺序敏感，需更稳健的技能检索与泛化；
- 仅针对单一 API（SimpleCADAPI），在多 API 或更复杂建模流程中的适用性尚未验证。

---

## 120. DDSynth-RL: Audio Synthesizer Inversion via Discrete Diffusion with Reinforcement Learning

**arXiv ID:** 2608.03032 | [PDF](https://arxiv.org/pdf/2608.03032v1)

**作者:** Tristan Wu `[一作]` (Hong Kong University of Science and Technology), Gus Xia `[通讯]` (New York University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了使用离散扩散模型（Masked Discrete Diffusion）进行音频合成器参数反演，并在模型上引入GRPO强化学习进行音频域奖励微调。

**💡 创新点**

创新点：①提出离散扩散生成器，既避免了自回归模型的固定顺序假设，又克服了连续流匹配在离散参数上需要的连续松弛与投影问题；②在非可微的黑盒合成器上实现了基于渲染音频的奖励微调（GRPO），将音频相似度直接纳入训练目标。

**🔧 技术方法**

技术：Masked Discrete Diffusion（基于Transformer的时间嵌入）、GRPO强化学习（相对奖励优化），以及多种音频相似度评估（wMFCC、CLAP、CREPE、MSS、SOT、RMS）。

**📊 数据集**

数据集：Dexed参数‑音频对数据集（860k对，覆盖215k预设组）用于监督训练；NSynth数据集用于跨域（OOD）评估与GRPO奖励计算。

**📈 对比分析**

比较方法：与自回归Transformer（AR）和连续流匹配（FM）三种生成框架进行同条件下的对比。实验显示：在域内（Dexed）AR略优；在域外（NSynth）离散扩散在MFCC相关指标上优于AR，且GRPO微调后大幅降低多种音频距离，提升OOV匹配效果。

**⚠️ 局限性**

局限性：①离散扩散对超参数（如掩码策略、步数）敏感；②GRPO微调会使模型偏离原始预设分布，导致域内指标下降；③当前仍依赖人手设计的参数分布，未能充分泛化到更复杂合成器；④奖励设计需依赖多种音频指标，可能与实际主观感受不完全一致。

---

## 121. The Frontier LLM Trap in Network Automation

**arXiv ID:** 2608.03080 | [PDF](https://arxiv.org/pdf/2608.03080v1)

**作者:** Minhao Jin `[一作]` (Princeton University), Maria Apostolaki `[通讯]` (Princeton University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过离线教学循环结合前沿LLM、模糊测试、验证器和仿真器，将小型本地LLM与可审计的逻辑规则结合，实现网络配置翻译的高效、低延迟、可控的网络自动化代理。

**💡 创新点**

采用教学-错误循环、规则挖掘、流式块级修复代理，将前沿LLM作为临时教师而非永久依赖，并用可解释的符号规则提升小模型性能。

**🔧 技术方法**

小型本地LLM（如Qwen2.5-Coder-32B-Instruct）、前沿LLM（GPT‑5.5）、模糊测试器、验证器与容器化仿真器、规则挖掘与流式修复代理。

**📊 数据集**

基于50个配置翻译实例的规则挖掘集和约110个测试实例的仿真验证集，构建Cisco–Junos对。

**📈 对比分析**

与原始小模型、前沿LLM以及prompt注入对比，规则增强后小模型准确率提升约28×至75%，延迟降低81%，且延迟分布更稳定。

**⚠️ 局限性**

规则模板与挖掘仍为手工，缺乏自动化；对多任务通用性有限；前沿LLM仅作为假设生成器，无法保证完全覆盖；验证覆盖面有限，难以保证所有业务场景均通过。

---

## 122. SieveIVF: Threshold-Aware IVF Execution for Large-Scale Training Data Deduplication

**arXiv ID:** 2608.03199 | [PDF](https://arxiv.org/pdf/2608.03199v1)

**作者:** Zhisheng Hu `[一作]` (Chinese University of Hong Kong), Ming-Chang Yang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种阈值感知的IVF执行器SieveIVF，利用查询在连续W个分区内未出现符合阈值的邻居时动态停止搜索，从而避免不必要的分区访问。

**💡 创新点**

创新点在于将连续空分区窗口作为无模型的早停信号，并通过连续批处理（continuous batching）与lookahead调度，在保持partition-major批处理的同时实现数据驱动的自适应IVF搜索，且无需改动索引格式。

**🔧 技术方法**

技术实现包括阈值驱动的分区停止规则、连续批处理与lookahead调度、以及在Lance向量存储与查询接口上的无缝集成。

**📊 数据集**

实验数据集包括腾讯Hunyuan的四个10M工作负载（WEB、TABLE、STEM、SCENE）以及两个公开100M工作负载（LAION-100M、DEEP-100M）。

**📈 对比分析**

与固定probe IVF及手动调优的fixed probe进行对比，SieveIVF在W=8时对Hunyuan工作负载加速4.1–7.6倍、对公共工作负载加速6.1–8.4倍，召回损失仅为0.03–1.13%（Hunyuan）和1.43–2.29%（公共）。

**⚠️ 局限性**

局限性包括：需要精确的质心分配；近似分配会导致召回损失；W的选取需平衡速度与召回，过小导致召回下降，过大则失去加速优势。

---

## 123. Lightweight Chunk Selection for Mobile Retrieval-Augmented Generation

**arXiv ID:** 2608.03148 | [PDF](https://arxiv.org/pdf/2608.03148v1)

**作者:** Sicong Chang `[一作]` (University of Houston), Renjie Hu `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在移动和边缘设备上提出了一种轻量级的检索增强生成（RAG）chunk 选择框架，利用单个检索块进行生成，显著降低上下文量和计算开销。

**💡 创新点**

创新点包括：①将LLM的隐藏状态、Mixture‑of‑Experts（MoE）路由信息与检索块嵌入三种特征融合，并用一个小型 MLP 预测“证据原型”，通过余弦相似度选取最支持答案的块；②基于证据充分性而非答案字符串出现率构造语义层面的 chunk 正确性标签；③引入梯度敏感度的任务意识特征选择，实现可调节的输入维度预算。

**🔧 技术方法**

技术手段包括：LLM 预热获取隐藏状态与 MoE 路由权重；检索块使用 Sentence‑Transformer + FAISS；特征融合后 MLP（3 层 1024 隐层）与 L2 归一化；多目标对比损失；梯度基特征重要性评估与稀疏化；余弦相似度排序。

**📊 数据集**

使用三大公开数据集进行评估：TriviaQA、PopQA 和 MS MARCO Passage Ranking，每个数据集都构造了检索到的 5 个块并人工/自动标注语义正确性。

**📈 对比分析**

与 RAG 基线、BM25、RankSVM、LambdaMart、Local‑DUET、DRMM、KNRM 等移动友好型 baselines 对比，平均提升 2.5% 的 rank‑1 chunk 选择准确率，在 PopQA 上提升超过 9%；性能在所有三大数据集上均优于现有轻量级方法，且仅增加 7 M 参数。

**⚠️ 局限性**

局限性：①只选择单个块，难以处理多跳或多块共同支持的答案；②对检索质量仍有依赖；③需要支持 MoE 的 LLM，非所有模型均可直接提取路由信息；④特征选择策略在极低预算下仍需实验验证。

---

## 124. Every Wrong Answer Counts: Option-Level Psychometrics for LLM Multiple-Choice Benchmarks

**arXiv ID:** 2608.02966 | [PDF](https://arxiv.org/pdf/2608.02966v1)

**作者:** Xiao Fei `[一作]` (École Polytechnique, Institut Polytechnique de Paris), Michalis Vazirgiannis `[通讯]` (École Polytechnique, Institut Polytechnique de Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入LLM‑NRM模型，对LLM在多项选择题中的完整答案概率分布进行建模，进而挖掘错误答案背后的系统性信息；

**💡 创新点**

将Nominal Response Model（NRM）扩展为LLM专用版本，加入响应尖锐度、位置偏好以及难度门控猜测机制，并证明错误答案提供的额外信息可显著提升能力估计与排名一致性；

**🔧 技术方法**

采用NRM与多参数IRT框架，结合最大似然与MAP估计（Adam优化、交叉熵损失、L2正则），并利用Fisher信息、Spearman相关等统计工具进行评估；

**📊 数据集**

使用189个公开LLM与14个基准（MMLU、ARC‑Challenge、CommonsenseQA、RACE等）共计31,554道多选题；

**📈 对比分析**

与多种基线（1PL/2PL/3PL/4PL IRT、Bock NRM、Deep‑IRT、β³‑IRT、PSN‑IRT、SD‑IR）进行5折交叉验证，LLM‑NRM在答案选项预测精度0.707、对数损失0.818、与Arena.ai Elo Spearman 0.920，明显优于其它模型；

**⚠️ 局限性**

仅适用于完整答案概率，假设单维能力且固定答案数，对不完整概率或多答案题型不适用，且模型参数设定较为复杂。

---

## 125. Vulnerabilities, Secrets and Misconfiguration in the Highest-Exposure Docker Hub Images

**arXiv ID:** 2608.02669 | [PDF](https://arxiv.org/pdf/2608.02669v1)

**作者:** Cristhian Kapelinski `[一作]` (Federal University of Pampa), Diego Kreutz `[通讯]` (Federal University of Pampa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究开发了三阶段的流水线，先爬取Docker Hub全部仓库，构建图层继承层次，计算每个镜像的曝光度评分，随后对曝光度最高的52,895个镜像使用六种开源扫描器（Syft、Trivy、Grype、OSV-Scanner、Dockle、TruffleHog）进行安全性扫描，最终得到漏洞、误配置与秘密泄露的全面测量，并发布完整的数据集与工具。

**💡 创新点**

创新点包括：①提出曝光度评分E(I)，将镜像自身下载量与其所有下游镜像的下载量合并为单一量化指标；②在生态系统级别统一使用多种扫描器进行交叉验证，揭示单一工具结果的偏差；③构建完整的层级图谱（约8.4千万节点/5.4千万边），实现对漏洞传播范围的量化评估；④公开了从爬取到扫描的完整流水线与数据集，供社区复现与进一步研究。

**🔧 技术方法**

技术实现：①爬取阶段使用Go实现分布式、可恢复的前缀搜索；②层图构建阶段采用Go与Neo4j存储层级图，使用SHA-256哈希生成祖先-哈希层ID；③扫描阶段用Python orchestrator，调用六款扫描器并统一格式化输出；④数据存储使用MongoDB（爬取元数据）和Neo4j（图层），扫描结果存入MongoDB；⑤对扫描器输出做归一化、去重与统计。

**📊 数据集**

使用的数据集包括：12,716,568个公开仓库（12.7M），其中5,601,045个已解析（44%），涵盖6,399,608个标签与7,416,671个镜像Digest；最终对52,895个曝光度最高的仓库（覆盖全部下载量的84.7%）进行扫描，产生约1.7亿条扫描发现（其中1.4亿漏洞、23.1万组件、495万秘密、58万误配置）。

**📈 对比分析**

与先前单工具研究对比，采用三款漏洞扫描器交叉验证后发现，单一扫描器仅覆盖约66.9%的独立漏洞组，互斥率高达66.8%。曝光度评分与下载量对漏洞数量相关性极低（ρ≈-0.01）。扫描性能方面，平均扫描时长约197 s，Grype为最慢（35.1 s/镜像），单镜像中六个扫描器总计中位数117 s。通过公开数据集，作者重现了六项以往研究的关键结论，并证明在更大规模下结论保持一致。

**⚠️ 局限性**

局限性包括：①仅分析曝光度最高的镜像，未覆盖所有仓库；②Stage II仅解析了最热门的44%仓库，导致层图与传播计数为下限；③扫描结果为静态检测结果，未验证漏洞可利用性；④秘密检测以TruffleHog为例，绝大多数命中为假阳性（99.7%非凭证），需进一步验证；⑤扫描器错误率不等，尤其Trivy在并行环境下出现超时；⑥未对镜像的实际运行环境与多架构版本做全面覆盖。

---

## 126. PolyLayout: Multi-room Manhattan Layout Estimation

**arXiv ID:** 2608.03323 | [PDF](https://arxiv.org/pdf/2608.03323v1)

**作者:** Gustav Hanning `[一作]` (Lund University), Viktor Larsson `[通讯]` (Lund University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于学习的优化框架，用3D Manhattan多面体表示室内布局，利用多视角图像和深度特征迭代优化，并能联合优化多间房间布局。

**💡 创新点**

① 将房间布局建模为可自适应的Manhattan多边形并通过迭代墙壁拆分/合并动态更新拓扑；② 将学习到的特征度量与几何约束结合，端到端训练；③ 在多房间场景共享全局方向、地板/天花板高度，实现更高精度；④ 提供两大新多视角多房间基准数据集。

**🔧 技术方法**

ViT编码器+两路卷积解码器；深度特征度量 (E_feat)、边缘对齐 (E_edge)、消失点一致性 (E_VP) 以及周长惩罚 (E_per)；Levenberg‑Marquardt迭代；多尺度优化；可视化采样；端到端训练等技术。

**📊 数据集**

Aria Synthetic Environments (ASE) 100k 多房间合成数据；ScanNet++ v2 80室手工注释；2D‑3D‑Semantics 160 立方体房间；并基于这些数据构建新的验证/测试集。

**📈 对比分析**

与 Plane‑DUSt3R、PixCuboid、SceneScript、RoomFormer、Deep3DLayout、LED2‑Net、PSMNet 等方法对比，使用3D IoU、Chamfer、深度RMSE、法向召回、墙/房间召回等指标。该方法在ASE、ScanNet++和2D‑3D‑Semantics上均优于对手，尤其在非立方体房间和多房间共享参数时表现突出。

**⚠️ 局限性**

需要已知相机位姿；对非Manhattan或多楼层布局支持有限；依赖可见性采样；在极端遮挡下可能缺失墙体；训练依赖手工注释的Manhattan布局。

---

## 127. CRIL-U-Net: Compact Ratio-Interaction Learning for Focal Cortical Dysplasia Segmentation from T1w and FLAIR MRI

**arXiv ID:** 2608.03185 | [PDF](https://arxiv.org/pdf/2608.03185v1)

**作者:** Soumen Ghosh `[一作]` (University of Queensland), Rajat Vashistha `[通讯]` (University of Queensland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文研究了利用3D U-Net与Compact Ratio‑Interaction Learning（CRIL）模块，对T1w与FLAIR MRI进行局灶性皮层发育不全（FCD）自动分割。

**💡 创新点**

创新点在于提出CRIL模块，该模块将局部空间特征、体素级跨模态混合与比值交互融合成紧凑表示，显著提升对小而隐蔽病灶的识别能力。

**🔧 技术方法**

技术方案为在3D U-Net骨干前嵌入CRIL或自注意力模块，并使用Dice–BCE与Focal Tversky‑Focal（FTF）两种损失函数进行训练。

**📊 数据集**

使用公开的85例FCD患者和25例健康对照的T1w+FLAIR MRI数据集进行实验。

**📈 对比分析**

在五折交叉验证中，CRIL‑U-Net+FTF平均Dice为0.196，漏诊率48.2%，相比传统U‑Net（Dice0.136、漏诊率57.6%）和自注意力U‑Net（Dice0.135、漏诊率54.1%）表现显著更佳（p<0.05）。

**⚠️ 局限性**

局限性包括样本量有限、缺乏外部验证、未做模块消融或更高级融合对比，导致整体分割精度仍不足以直接临床应用。

---

## 128. Evaluating OpenAI's Privacy Filter: Cross-Lingual, Cross-Domain PII Detection Across 42 Benchmarks

**arXiv ID:** 2608.02616 | [PDF](https://arxiv.org/pdf/2608.02616v1)

**作者:** Rohith Uppala `[一作]` `[通讯]` (Independent Researcher), Rohith Uppala (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了OpenAI Privacy Filter（OPF）在多语言（22种）、多领域（5类）42个基准上的零射击性能，并与Presidio、GLiNER、GPT‑4o和XLM‑RoBERTa等模型进行对比。

**💡 创新点**

首次提供跨22种语言、5个行业的系统性评估，揭示OPF在结构化PII识别上的优势以及在非拉丁文字和叙事性文本中的显著局限，并分析了不同PII类型的可检测性规律。

**🔧 技术方法**

使用1.5B参数的OPF（基于Transformer的MoE + CRF头）进行零射击序列标注，并与Microsoft Presidio、GLiNER、GPT‑4o API以及XLM‑RoBERTa进行比较，采用span级F1作为评价指标。

**📊 数据集**

实验数据集包括AI4Privacy、Nemotron‑PII、Kiji、Gretel Finance、SPY、CoNLL‑2002/3、MultiCoNER v2、IndicNER等，覆盖合成PII、客户支持、金融、医疗/法律及多语言NER。

**📈 对比分析**

采用字符级span F1并给出95%置信区间进行比较；OPF在结构化PII（F1≈0.71）和客户支持（F1≈0.60）上表现最佳，金融为0.52，医学/法律为0.46；在非拉丁文字（如阿拉伯语、乌克兰语）F1低至0.02；GPT‑4o在医学/法律上领先。

**⚠️ 局限性**

仅评估零射击模式，全部使用合成PII数据，缺乏对真实数据的验证；对非拉丁文字的低分可能因训练覆盖不足；未对OPF的fine‑tune能力进行评估；脚本族与模型性能的因果关系难以 disentangle。

---

## 129. One Knob to Rule Them All: A Unified Optimal Transport View of Cold-Start Active Learning

**arXiv ID:** 2608.03249 | [PDF](https://arxiv.org/pdf/2608.03249v1)

**作者:** Ning Zhu `[一作]` (University of Electronic Science and Technology of China), Liang-Jian Deng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出统一的最优运输框架，涵盖典型的 Cold-Start Active Learning 方法，并基于此设计自适应熵正则化的 Sinkhorn 算法（-AS），实现无需标签数据即可在一次性采样中自动平衡代表性、覆盖度与多样性。

**💡 创新点**

创新点在于：①将典型性、覆盖度、连续多样性三类 CSAL 方法映射为同一优化程序；②证明熵正则化提供了几何拟合与分配扩散的单调权衡；③基于池子几何分辨率提出任务无关的最小化风险下界，并据此推导出基于数据的温度自适应规则，实现选择策略的自动调节。

**🔧 技术方法**

使用最优运输（Sinkhorn 迭代）与 k‑means 聚类、余弦/欧氏距离、熵正则化、Round‑Robin 解码等技术，构建可调节的平衡 OT 方案，并通过理论分析指导温度选择。

**📊 数据集**

在 MNIST、CIFAR‑10、CIFAR‑100、STL‑10、Caltech‑101 与 ImageNet‑1k 六大公开数据集上进行实验。

**📈 对比分析**

与 Random、Coreset、FPS、BADGE、ProbCover、DCoM、TypiClust、USL、ActiveFT 等九种基线对比，-AS 在 25/26 个实验设置中取得最高准确率，平均提升约 1–1.3%，并在 ImageNet‑1k 上实现约 2.3× 的选择速度提升，兼具精度与效率。

**⚠️ 局限性**

局限包括：需要预先校准一个无关常数 c；算法依赖于冻结的特征表示（仅使用 DINOv2 特征）；在极大规模池子时 Sinkhorn 迭代仍占用一定计算资源，且对不同任务/特征空间的进一步泛化尚未充分验证。

---

## 130. MemArena: An Ego-Centric Benchmark for On-Device Agentic Personal Memory Assistants at Scale

**arXiv ID:** 2608.02613 | [PDF](https://arxiv.org/pdf/2608.02613v1)

**作者:** Jiadong Zhang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiaosong Ma `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 5154 | [OpenAlex ID](https://openalex.org/A5057826903)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

创建了一个自我中心、活动密集的对话记忆基准，用于评估边缘部署个人助手的记忆能力和权限控制。

**💡 创新点**

通过解决视角、全局连贯性和活动密度三大结构缺口，提出EgoMem基准、可扩展多代理仿真器和基于证据的任务生成，并将其与五种内存后端和五个开放权重阅读器一起评估。

**🔧 技术方法**

采用多代理仿真器、时间同步批处理、BM25‑RAG、Memobase、MemSearch检索后端、LLM判定器（gpt‑4o‑mini）进行评估，并在NVIDIA Spark GB10边缘设备上测量TTFT。

**📊 数据集**

生成了一个合成数据集，包含50个代理在15天内的对话，总计10.3M文本标记，约24.1K每代理/天的自我观察文本，以及1,579个评估实例。

**📈 对比分析**

通过准确率（回忆、推理、信任度）和权限意识F1_PU，以及端到端延迟TTFT进行比较，结果显示MemSearch是最强的后端，内存后端比读者规模更影响准确率；权限控制表现普遍不佳，最高F1_PU仅44。

**⚠️ 局限性**

数据为合成且单语，缺乏噪声、多模态和文化差异；评估依赖LLM生成标签和单轮判定，未覆盖跨世界或长期漂移，后端与阅读器的接口不够健壮。

---

## 131. Provably Learning Multi-Head Attention with Queries

**arXiv ID:** 2608.03294 | [PDF](https://arxiv.org/pdf/2608.03294v1)

**作者:** Sunyeop Kim `[一作]` (Korea University), Jian Guo `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用黑盒值查询精确恢复多头 softmax 注意力模型的 Canonical 形式的方法。

**💡 创新点**

创新点在于不依赖正交子空间假设或已知子空间基，利用重复 token 的有理插值与跨查询匹配，得到唯一（按置换）Canonical 形式，并给出最优查询复杂度 4Hd^2-2H+1；同时扩展到一层 ReLU Transformer 的等价学习。

**🔧 技术方法**

核心技术包括：
- 设计重复-token 查询构造有理函数并进行多项式插值；
- 通过不同向量组合实现头部匹配；
- 线性系统求解和多项式因式分解（理论上是精确算子，实际用多精度数值实现）。

**📊 数据集**

实验使用合成数据：随机从 N(0,1/d) 生成每个头的 W_h 和 v_h，覆盖多组 (d,H) 配置。

**📈 对比分析**

与传统的单头恢复（O(d^2) 查询）以及先前需要正交子空间的多头恢复方法对比；在高精度（180 位）oracle 输出下几乎无误差（E_param < 10⁻¹⁰⁰），而在 IEEE 754 binary64 输出下成功率随 H、d 下降，验证算法对噪声的敏感性。

**⚠️ 局限性**

局限性：
- 需要精确或足够高精度的 oracle 输出；
- 对输出误差极为敏感，噪声会放大成参数误差；
- 需要随机选择查询方向，若无法满足非退化条件会失效；
- 计算上假设可进行精确算术、对数、指数、求解线性系统与多项式因式分解；
- 需知道 H 或上界 H_0；
- 对大规模模型的实际实现仍面临数值稳定性和计算成本挑战。

---

## 132. Biconvex Optimization for Smooth Minimum-Time Trajectories around Convex Obstacles

**arXiv ID:** 2608.02834 | [PDF](https://arxiv.org/pdf/2608.02834v1)

**作者:** Peter Werner `[一作]` (Massachusetts Institute of Technology), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种双凸最短时间规划器（BMTP），能够在凸障碍物环境中求解带任意阶导数约束的最短时间运动规划。

**💡 创新点**

创新点在于通过变量变换将最短时间目标和导数约束凸化，并将碰撞避免约束转化为时变分离平面，从而得到双凸问题；随后采用交替优化与仅标记碰撞障碍物的策略，提升局部最优逃逸能力并保证收敛。

**🔧 技术方法**

采用双凸优化、时间缩放变量、贝塞尔曲线/多段曲线表示、极点极点法（polar）构造分离平面、最大间距分离平面求解、Clarabel/Drake求解器以及Python实现的开源库。

**📊 数据集**

使用开放式村庄环境（521个轴对齐盒）进行无人机飞行实验，使用双臂箱子卸载任务（6维任务空间、88/123个凸障碍）进行仿真与硬件实验，另外在真实Franka机械臂上验证。

**📈 对比分析**

与FPP、离散约束的非凸求解、EI+SCS等方法对比：在村庄实验中BMTP完成时间为11.83s，远快于FPP（25.78s），且计算时间仅0.19s；在箱子卸载实验中BMTP平均约188ms，略快于EI+SCS的204ms，且轨迹时长与其相当；硬件实验中两者相似，BMTP略快。

**⚠️ 局限性**

局限性包括：仅支持凸障碍物，无法直接处理非凸或配置空间规划；收敛结果不一定为局部最优；需要预先可行路径作为初始化；分离平面阶数经验选择；高阶导数约束下数值可能不稳定。

---

## 133. GLOBE: Trajectory-Aligned Gradient Matching with Structured SparseOptimization for Coreset Selection

**arXiv ID:** 2608.02690 | [PDF](https://arxiv.org/pdf/2608.02690v1)

**作者:** Hetian Liu `[一作]` (Xi'an Jiaotong University), Pengju Pen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于多检查点梯度轨迹的样本选择框架 GLOBE，用以构建能够在设备端高效训练的核心子集。

**💡 创新点**

创新点在于将梯度轨迹与一阶均值及二阶协方差匹配相结合，并通过 Group LASSO 与 Elastic Net 实现结构化稀疏化，显著提升低保留率下的训练效果。

**🔧 技术方法**

采用梯度轨迹构建、两阶分布匹配、多重随机投影、Group LASSO、Elastic Net、非负预算约束以及类别平衡 Top‑K 选择等技术。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、CINIC‑10、SVHN、ImageNet‑100 与 ImageNet‑1K 六个图像分类基准上进行实验。

**📈 对比分析**

与几乎所有现有的几何、分数、决策边界及梯度匹配等基线进行比较，GLOBE 在 18 种不同数据保留比例与模型架构下均取得最高准确率，尤其在 10%–20% 保留率时提升 1–4个百分点。

**⚠️ 局限性**

局限性包括：需要多次训练代理模型获取梯度轨迹，计算与存储开销相对较高；对超参数敏感；目前仅在图像分类任务上验证，缺乏对其他任务或更大规模数据的通用性证明。

---

## 134. Convex-Hull-Neighborhood Smooth Dual Generalization: Controlling Local Correction Propagation in Offline RL

**arXiv ID:** 2608.03108 | [PDF](https://arxiv.org/pdf/2608.03108v1)

**作者:** Yi Yang `[一作]` (Xiamen University), Lvqing Yang `[通讯]` (Xiamen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的离线强化学习算法CSDG，通过在Bellman备份中加入基于Convex Hull Neighborhood (CHN) 的局部平滑校正来平衡近似分布外（OOD）动作的利用与误差放大；

**💡 创新点**

创新点在于将局部平滑的目标与期望值(expectile)的内样本目标分离，并通过λ控制校正贡献，实现两层级的控制：1）在局部几何空间内生成两尺度噪声采样以获得近似与扩展候选动作；2）在Bellman更新中按λ插入差异校正；同时给出了精确的一步、迭代、固定点以及隐式策略的理论界定；

**🔧 技术方法**

使用的技术包括：CHN几何参考、两尺度扰动噪声（不对称、带边界）、期望值回归(expectile regression)、双Q值、Polyak平均、TD3BC样式的演员-评论家结构、优势加权行为克隆；

**📊 数据集**

在D4RL基准上进行实验，主要使用Gym–MuJoCo（半长颈鹿、跳箱等）和AntMaze（u、u-d、m-p、m-d、l-p、l-d等）共21个数据集；

**📈 对比分析**

与20余个基线（包括BC、BCQ、BEAR、AWAC、TD3BC、CQL、IQL、XQL、CPI、C4、UNIQ等）进行对比。CSDG在Gym–MuJoCo总分1199.4、AntMaze总分445.8排名第一，平均分最高；与IQL、XQL等基线结合时均显著提升；

**⚠️ 局限性**

局限在于需要手动设置噪声尺度和λ等超参数，未实时跟踪CHN几何，未针对高维动作空间进行自适应调整，且仅在离线数据环境中验证，缺乏在线或动态环境的实验。

---

## 135. On the missing benchmarks layer and a potential solution

**arXiv ID:** 2608.02996 | [PDF](https://arxiv.org/pdf/2608.02996v1)

**作者:** Francis F Daniel `[一作]` (SURUS), Marian Basti `[通讯]` (SURUS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并设计了 Latin America 区域的 Benchmark 层——EvalsHub 及其首个实例 LatamBoard，作为公共机构审核与企业优化的共享评测基础设施。

**💡 创新点**

创新点在于任务优先的层次化本体、AI 系统无关的考试设计、开放式与激励驱动的构建机制，以及通过降低领域专家门槛来解决 Benchmark 供应瓶颈。

**🔧 技术方法**

利用开放源码的评测平台、标准化评测协议、AI 系统无关的评测框架，并与软件 3.0 堆栈优化器（如 DSPy）协同使用。

**📊 数据集**

文中未给出具体数据集，强调未来将使用本地语言、领域（如医疗、农业）真实数据构建 Benchmark。

**📈 对比分析**

比较方法基于统一的评分函数与标准化协议，目标是让不同模型、工作流和代理在同一任务、域、语种上可比；尚未给出性能数值，强调评测将持续追踪并提供可视化排行榜。

**⚠️ 局限性**

局限在于缺乏实际 Benchmark 与数据、缺乏治理与标准化框架、访问问题仍未彻底解决、以及模型可能在训练中泄露 Benchmark 内容导致污染风险。

---

## 136. NANQ: Noise-Floor-Aware Mixed-Precision Non-Uniform Quantization for Analog Compute-in-Memory

**arXiv ID:** 2608.02700 | [PDF](https://arxiv.org/pdf/2608.02700v1)

**作者:** Yizhe Chen `[一作]` (Beihang University), Wang Kang `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练、噪声感知的混合精度非均匀量化框架 NANQ，用来在模拟 CIM 加速器上实现高效的低精度推理。

**💡 创新点**

创新点在于：①将芯片测得的幅值相关权重噪声分布映射为量化密度，从而在低噪声区分配更细的量化级；②基于噪声饱和阈值自动确定每层的位宽，无需再训练或组合搜索；③兼顾权重量化与层级位宽分配，形成完整的无训练量化方案。

**🔧 技术方法**

技术手段包括：硬件噪声测量得到 σ(w)；采用 ρ(w)=1/(σ(w)+ε)^γ 定义逆密度；利用累计分布 F(w) 对量化区间进行等分；使用统一阈值 τ 计算每层的饱和点并分配位宽；在 eFlash CIM SoC 上进行真实芯片验证。

**📊 数据集**

数据集与模型：视觉模型使用 CIFAR‑100、ImageNet‑1K（ResNet‑20/56、VGG‑11、ViT‑Base）；语言模型使用 WikiText‑2（Pythia‑410M、1B、Llama‑3.2‑1B/3B、OPT‑1.3B、4‑1.3B）。

**📈 对比分析**

比较方法：与 Uniform、APoT、PowerQuant、PNMQ 等无训练或数据‑free 量化方法对比；在 2–6 位宽下，NANQ 在 38/45 组合中获得最佳精度，2 位时提升 7–24% 视觉精度，语言模型 PPL 降 54.7%；在混合精度下，在相同等效位宽 3.2–3.8 的条件下，NANQ 进一步提升准确率 1–2.5% 并减少 PPL 3–10%。

**⚠️ 局限性**

局限性：①需要频繁更新噪声特征以适应温度/老化变化；②目前仅考虑权重噪声，未建模激活与输出噪声；③提供的是算法层面的等效位宽，未直接给出实际能耗、延迟或存储占用；④在极高位宽或非 eFlash 设备时性能优势可能减弱。

---

## 137. Scalable Frequency- and Length-Aware Subdocument Deduplication for Large Language Model Pretraining

**arXiv ID:** 2608.03089 | [PDF](https://arxiv.org/pdf/2608.03089v1)

**作者:** Hai Wang `[一作]` (Tencent), Feng Zhang `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可扩展的子文档去重框架，先通过自然边界分块、归一化哈希和分布式聚合识别重复片段，然后基于全局重复频率和长度动态分配保留预算，最后采用连贯性保留删除减少文档碎片。

**💡 创新点**

创新点在于将重复检测与保留决策解耦，并设计了显式的频率-长度感知保留函数，能够在全局视角下自适应地决定每个重复组保留多少副本，同时通过分布式哈希实现大规模全局计数。

**🔧 技术方法**

核心技术包括：自然边界分块（段落/行/句/代码块），归一化文本与哈希，分布式键值聚合（MapReduce/Spark），以及基于随机分片推导的保留函数与长度衰减函数。

**📊 数据集**

实验数据集为 FineWeb‑Edu（6.28T 语料）和包含代码的网页语料（561.53B 语料）。

**📈 对比分析**

与文档级 MinHash、单例保留、分片后缀数组等基线比较，训练 36k 步后在 FineWeb‑Edu 上平均提升约 1.0 分（最高 52.92），在代码网页上平均提升 3.42 分（最高 41.40）。

**⚠️ 局限性**

局限性包括：保留函数依赖超参数 N 与 L₀ 需要经验调优；对非常长或非常短片段的处理仍可能不够精准；仅在单机或小规模集群上验证，跨集群可扩展性尚待进一步评估。

---

## 138. A Graph Signal Processing Perspective on Numerical Sequence Representations in LLM In-Context Learning

**arXiv ID:** 2608.03015 | [PDF](https://arxiv.org/pdf/2608.03015v1)

**作者:** Jiajun Bao `[一作]` (Cornell University), Christopher J. Earls `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出利用图信号处理框架，将大语言模型的注意力权重视为加权图，隐藏状态视为图信号，研究其在无监督数值推理（ICL）中的内部表示演化。

**💡 创新点**

创新点在于将注意力图谱与隐藏状态的谱特征结合，量化了全局连通性、高频能量与谱熵随上下文长度和输入动力学复杂度的变化，揭示了数值ICL的系统内部特征。

**🔧 技术方法**

使用的技术包括图拉普拉斯谱分解、归一化Fiedler值、HFER（高频能量比例）与谱熵、Fruchterman–Reingold布局以及基于SVD的隐藏状态颜色编码。

**📊 数据集**

实验数据来源于十类一维数值序列（常数、不同周期的Logistic映射、Lorenz系统等），每类20个轨迹样本，覆盖从100到2000令牌的上下文长度。

**📈 对比分析**

对比方法是对不同模型（Llama‑3.2‑3B、1B、8B及Phi‑4、SmolLM3）以及指令调优与基础模型进行层平均诊断，结果显示更长上下文能更清晰地区分动力学复杂度，且模型规模越大分离度越高。

**⚠️ 局限性**

局限性包括：诊断仅揭示结构关联未证明因果性；需要访问内部权重与隐藏状态，受限于开放权重模型；不同tokenization方案需额外聚合步骤。

---

## 139. Hear to See: Discerning Stateful Listening for Audio-Visual Instance Segmentation

**arXiv ID:** 2608.03264 | [PDF](https://arxiv.org/pdf/2608.03264v1)

**作者:** Leiye Liu `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出并实现了 H2S 框架，用于音频-视觉实例分割，能够精确匹配混合音频与视觉实例并跟踪异步变化。

**💡 创新点**

创新点包括：① Acoustic‑Semantic Projector 通过音源分离与层级对应实现精确音频‑视觉匹配；② Asynchronous Dynamics Modulator 动态调节 Mamba 内部参数 Δ，适应音频与视频异步状态变化。

**🔧 技术方法**

技术手段包括：MixIT 音频分离、VGGish 语音特征提取、Mamba 状态空间模型、k‑means 层级对应、Top‑p 筛选以及基于音频动态调节的 Δ。

**📊 数据集**

使用的数据集为 AVISeg 语音‑视觉实例分割基准（约 926 只视频，26 类音频）。

**📈 对比分析**

与 AVISM、ACVIS 等基线在 ResNet‑50+COCO 预训练下对比，H2S 在 mAP 上提升 7.8%（48.54 对比 40.66），HOTA 与 FSLA 亦显著提升。

**⚠️ 局限性**

局限性：目前仅在离线设置下验证，无法直接应用于在线实时推理；需要改造为因果时序和在线音频分离才能满足在线场景。

---

## 140. SAKI: Score-Aware Low-Rank Key Indexing for Long-Context KV Retrieval

**arXiv ID:** 2608.03228 | [PDF](https://arxiv.org/pdf/2608.03228v1)

**作者:** Lin Zhang `[一作]` `[通讯]` (Cotality), Lin Zhang (Cotality)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 KV-cache 低秩索引的目标函数，推导了基于注意力得分误差的两侧协方差加权低秩近似，并给出闭式最优解，提出训练‑free 的 SAKI 算法，用于高效检索自注意力键。

**💡 创新点**

创新点在于：① 明确注意力索引应最小化得分误差而非键方差或权重截断；② 推导出两侧协方差加权低秩目标并求得非投影闭式最优映射；③ 通过这一目标实现比 key‑PCA 更优的 top‑k recall。

**🔧 技术方法**

技术手段包括：协方差归一化、SVD、Eckart–Young 近似、异质低秩映射、预旋转 RoPE 代码、对查询/键协方差的校准与估计。

**📊 数据集**

实验使用 4,096 tokens 的自然文本，对 LLaMA‑3.1‑8B、Qwen2.5‑7B、Mistral‑7B‑v0.1、Llama‑3.2‑3B 四大模型进行评估。

**📈 对比分析**

与 key‑PCA、weight‑SVD、Schur 不变子空间等基线在 top‑64 recall 上比较，SAKI 在所有模型、所有 rank（16、32、64）均优于 PCA，删除 13–30% 的 PCA 余留误差，提升 68–89% 的 heads，尤其在深层表现突出；理论预测与实测高度一致。

**⚠️ 局限性**

局限性包括：仅在单一校准域（4K 上下文）评估；未测评生成质量、长文本、域迁移、RoPE 旋转对性能的影响；未解释不同模型表现差异的原因；需进一步研究在线更新与跨域适应。

---

## 141. BulkPR-Bench: Benchmarking Queue-Level Governance of Interacting Pull Requests

**arXiv ID:** 2608.02685 | [PDF](https://arxiv.org/pdf/2608.02685v1)

**作者:** Zetong Xiong `[一作]` (Baidu), Yehua Yang `[通讯]` (Baidu)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5037213475)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 BulkPR-Bench，一个用于评估队列级 Pull Request (PR) 治理的可执行基准，重点考察 PR 间相互作用、关系恢复、可安全提交子集选择以及可执行合并顺序。

**💡 创新点**

创新点在于：①构建可执行的“金丝雀”关系图，结合隐藏验证器和 exact solver 提供完整安全约束；②设计滚动发布协议（batch K 与 deferral）以模拟真实 CI 流程；③引入 Relational Delivery Score (RDS)、Global‑SGY 与 Exact Completion 等多维度评估指标，系统化衡量关系级别与全队列治理的差异。

**🔧 技术方法**

技术方法包括：GitHub 仓库快照与 CI 自动化执行、隐藏验证器、gold relation graph 与 exact solver、批量决策与 typed ledger、滚动可见性协议、对话式大模型代理（Claude‑Opus、DeepSeek、GLM、GPT、Kimi、Qwen）等。

**📊 数据集**

数据集由 18 个真实仓库（Python、Go、TypeScript/Node/Bun）组成，每个仓库包含 32–33 个新构造 PR，总计 581 个 PR，关系图涵盖冲突、依赖、全或无组、必拒绝、重复、超越等多种关系族。

**📈 对比分析**

评估方法：与三种确定性公共信息基线（Merge‑Queue、CI‑Fixedpoint、Batch‑Greedy）对比，采用 RDS（关系组安全交付）为主要排名指标，发现最高模型 Claude‑Opus 在缓冲协议 K=32 上获得 66.6% RDS，超过 53.1% 的 sequential baseline；Global‑SGY 最高为 13%，但 Exact Completion 仅 8/324，显示全队列治理仍有限。

**⚠️ 局限性**

局限性：①PR 集为人为设计，未代表真实队列频率；②评估侧重执行安全，未考虑业务价值、审查成本或风险严重性；③所有模型共享同一 scaffold 与 prompt，缺乏设计多样性；④每组仅跑 3 次，未充分覆盖运行波动；⑤缺乏对反馈重规划与动态决策的深入探索。

---

## 142. Efficient Optimal Mouse Sensor Position Estimation using Simulated Cursor Trajectories

**arXiv ID:** 2608.03168 | [PDF](https://arxiv.org/pdf/2608.03168v1)

**作者:** Minhyeok Baek `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Sunjun Kim `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过双传感器鼠标模拟不同传感器位置的光标轨迹，并在短时间内估计出最优位置

**💡 创新点**

使用仿真轨迹和MAE误差评估取代多次Fitts法实验，显著缩短校准时间

**🔧 技术方法**

双传感器融合、仿真轨迹生成、均值绝对误差（MAE）计算、模糊轨道光标可视化

**📊 数据集**

实验数据来自用户在5分钟模糊轨道光标任务中记录的前后传感器位移（dX_front,dY_front,dX_rear,dY_rear）

**📈 对比分析**

与传统手动校准（TP throughput）对比，仿真方法得到的MAE曲线更平滑且定位更精准，时间从约1小时降至5分钟，定位结果与手动方法一致

**⚠️ 局限性**

样本量小（仅4名参与者）、仅在一次性实验中验证、缺乏纵向跟踪与统计显著性检验

---

## 143. OPTD: On-Policy Transition Distillation with Consistency-Guided Adaptive Compression for Few-Step Diffusion Language Models

**arXiv ID:** 2608.02942 | [PDF](https://arxiv.org/pdf/2608.02942v1)

**作者:** Xiaocheng Lu `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于自我策略的过渡蒸馏框架OPTD，用于加速扩散式语言模型的多步推理。

**💡 创新点**

创新点在于：①使用学生自身轨迹采样状态；②通过冻结的仅问答教师进行一致性引导，动态确定可压缩的未来行动前缀；③使用集合瓶颈置信度损失与教师KL锚定的联合目标，提升并行解码效率而不牺牲准确率。

**🔧 技术方法**

技术包括：扩散语言模型（dLLM）、自我蒸馏（on‑policy distillation）、教师-学生一致性检验、集合瓶颈置信度约束、冻结教师KL正则、基于块的多阈值解码。

**📊 数据集**

使用四大基准：GSM8K（数学推理），MATH‑500（数学推理），MBPP（代码生成），HumanEval（代码生成）。

**📈 对比分析**

与现有少步蒸馏方法（Fast‑dLLM、D2F、dParallel、d3LLM、TAD‑Q、TAD‑S）以及原始LLaDA在相同解码器和阈值下进行比较。OPTD在四个任务的宏平均上取得 47.21% 准确率与 7.87 TPF 的平衡点，AUP（质量约束的积分性能）最高 313.18，显著优于TAD‑S（245.59）、d3LLM（230.50）和TAD‑Q（218.07）。

**⚠️ 局限性**

局限性包括：①推理时无一致性保证，仍依赖学生的置信校准；②仅在固定教师策略下训练，教师更新对性能影响较大；③在不同模型/硬件/解码策略上迁移性尚未彻底验证；④适配更大响应长度或多样化任务的效果未知。

---

## 144. PECR: A Reproducible Specification and Synthetic Stress Test of Telemetry-Informed Vulnerability Prioritization for SD-WAN

**arXiv ID:** 2608.03110 | [PDF](https://arxiv.org/pdf/2608.03110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 145. A Physics-Informed Hybrid Neural Operator for Transient Magnetization Prediction in Power Magnetics

**arXiv ID:** 2608.02965 | [PDF](https://arxiv.org/pdf/2608.02965v1)

**作者:** Yachao Zhu `[一作]` (University of Technology Sydney), Jianguo Zhu `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种物理信息混合神经算子，用于在电力磁性材料中预测瞬态磁化响应。

**💡 创新点**

创新点在于结合了局部递归分支和全局特征提取机制，以实现边界条件下的瞬态B-H轨迹预测，同时引入了能量一致性正则化。

**🔧 技术方法**

使用了混合神经网络架构，结合了局部递归单元（GRU）和基于Transformer的全局分支。

**📊 数据集**

使用了MagNetX瞬态数据库，包含14种铁氧体材料的测量数据。

**📈 对比分析**

与其他模型（如LSTM、MagLearn2等）进行比较，结果显示该模型在能量一致性和模型紧凑性方面表现优越，平均序列误差为13.40%，能量一致性误差为1.92%。

**⚠️ 局限性**

局限性在于模型的训练和评估依赖于特定材料的数据，可能无法广泛适用于所有类型的磁性材料。

---

## 146. On the Minimum Field Size of Network MDS Codes for Generalized Combination Networks

**arXiv ID:** 2608.03209 | [PDF](https://arxiv.org/pdf/2608.03209v1)

**作者:** Qin Zhou `[一作]` (Nankai University), Fang-Wei Fu `[通讯]` (Nankai University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在两类典型多层多播网络（一般化组合网络和Zosin–Khuller网络）上实现网络MDS（最大距离可分离）码所需的最小域大小，并给出了标量与向量码的存在条件和对应的域大小界限。

**💡 创新点**

创新点在于建立了网络码最小距离与经典线性码（以及覆盖Grassmannian码）最小汉明距离之间的严格等价；利用MRD码和改进的贪心构造分别得到更紧的上、下界；首次给出了一族组合网络在向量与标量码之间无域大小优势（MDS间隙为零）的证明；并通过超图同态框架为Zosin–Khuller网络推导了新的向量码域大小下界和间隙上界。

**🔧 技术方法**

主要技术包括：线性网络编码与经典码的距离等价；覆盖Grassmannian码的构造与大小估计；MRD码与矩阵幂的乘法性质；贪心与递归构造法；超图同态与色数分析；以及子集交叉与鸽巢原理等组合论工具。

**📊 数据集**

论文未使用任何实验数据集，所有结果均为理论推导与解析证明。

**📈 对比分析**

与以往的通用域大小上界相比，本文给出的标量/向量码域大小上界显著更小；在向量码可实现时，新的下界通常比先前已知下界更严格，且在某些参数区间内向量码与标量码实现相同最小域大小，从而证明了无优势的结论。

**⚠️ 局限性**

局限性包括：对一般化组合网络中ϵ≠0时向量码是否能严格降低域大小尚未完全解决；Zosin–Khuller网络中间距上界与下界之间仍存在差距；以及超图同态方法主要适用于该网络结构，推广到更一般网络仍需进一步研究。

---

## 147. ProPRL: Property-Aware Prerequisite Relation Learning in Educational Knowledge Graphs

**arXiv ID:** 2608.03006 | [PDF](https://arxiv.org/pdf/2608.03006v1)

**作者:** Xinghe Cheng `[一作]` (Jinan University), Quanlong Guan `[通讯]` (Jinan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为ProPRL的框架，用于学习教育知识图中的前置关系，结合概念–资源超图与学习行为图的多视角表示，并通过对每个候选概念对进行自适应融合与不可逆约束来提升预测质量。

**💡 创新点**

创新点包括：①方向保持的个性化多跳传播以捕获行为图中的高阶依赖；②对概念对进行门控融合的Pair-conditioned Gate，使得不同候选对的视角贡献可调；③不可逆正则化（anti‑symmetry constraint）抑制双向高置信度，提升方向一致性。

**🔧 技术方法**

使用的技术主要有：超图卷积网络 (HGCN) 捕获概念–资源关联；多跳传播（APPNP式）与方向化图卷积 (GCN) 提取行为图信息；门控门 (MLP + sigmoid) 进行自适应融合；Siamese 分类器与多视角一致性损失；交叉熵 + 方向不可逆正则化的联合优化。

**📊 数据集**

实验数据集为 MOOC、LectureBank 以及 University Course (UCD) 三个公开前置关系数据集，涵盖课程、讲座与大规模 MOOC 的知识图和学习轨迹。

**📈 对比分析**

与 10+ 基线（包括通用方法 NB、SVM、RF 等与任务特定方法 HGAPNet、MHAVGAE、ConLearn、LCPRE、DGCPL）在 ACC、F1、AUC 上进行横向比较，ProPRL 在所有 9 组指标中均获得最高分，AUC 最高提升约 6%，F1 与 ACC 亦提升 4–5%。

**⚠️ 局限性**

局限性主要在于：①需手动调节多跳深度、传播系数及不可逆权重等超参数；②整体模型相较于单一视角方法计算开销略大；③对缺乏行为轨迹或资源关联信息的数据集效果尚未验证。

---

## 148. Passively Safe Convex Guidance for Cislunar Rendezvous and Proximity Operations

**arXiv ID:** 2608.03060 | [PDF](https://arxiv.org/pdf/2608.03060v1)

**作者:** Ian M. Down `[一作]` (Advanced Space, LLC), Michael Caudill `[通讯]` (Advanced Space, LLC)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一套纯凸优化框架，用于在南9:2 NRHO轨道上实现自主安全的冲击式近距离会合、靠近和紧急撤离。

**💡 创新点**

创新点在于将三体动力学的相对运动用一阶STM近似，结合保守的凸化MEE模型和椭圆投影安全约束，构成仅依赖SOCP的完全凸控制策略，避免了SCP的迭代和收敛不确定性。

**🔧 技术方法**

技术包括SOCP求解、First‑Order STM相对运动建模、Gates MEE模型的凸化、椭圆投影约束、定时网格和Sundman变换。

**📊 数据集**

数据集为CAPSTONE 02 NRHO的高保真三体动力学轨迹、预设的相对导航传感器量测（S/光学/跨链）和燃料阈值。

**📈 对比分析**

通过两组100次蒙特卡洛闭环仿真（近极地和近逆转），与预设的安全、保持及姿态要求比较，结果显示在24 h/8 h会合下，保持99%置信度的接近距离≤200 m，约束满足率>95%。

**⚠️ 局限性**

限制在于对一阶STM近似的依赖，极端非线性区间可能导致误差累积；凸化MEE保守性导致燃料消耗增大，且未考虑实时燃料/热管理约束。

---

## 149. Evidence-Grounded Multimodal Knowledge Graph Construction for Multi-Lecture Educational Reasoning

**arXiv ID:** 2608.03161 | [PDF](https://arxiv.org/pdf/2608.03161v1)

**作者:** Sahil Al Farib `[一作]` (United International University), Md. Tanvir Raihan `[通讯]` (United International University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于多模态证据的教育知识图谱，从讲座视频中提取概念与关系并保留出处。

**💡 创新点**

创新点在于只接受有视觉或文本证据支持的提取，保证可追溯性和可审计性，并将检索与图结构结合。

**🔧 技术方法**

采用 Whisper 语音识别、EasyOCR、Qwen2.5‑VL 视觉‑语言模型、BGE embeddings、NetworkX 等技术。

**📊 数据集**

使用了 3Blue1Brown 三节关于神经网络、梯度下降和反向传播的讲座视频。

**📈 对比分析**

与仅基于转录的 RAG 或未加证据的提取做对比，最终图包含 172 概念、282 条关系，检索上 3 个 seed 问题均达到 100% top‑1/3 及 100% top‑5 recall。

**⚠️ 局限性**

局限在于样本仅三节课、缺乏人工金标准、未对检索做大规模评测、以及命名实体合并不完整。

---

## 150. Frequency-Decorrelated Temporal Ensembles for EEG--fNIRS Imagined-Handwriting Decoding

**arXiv ID:** 2608.03176 | [PDF](https://arxiv.org/pdf/2608.03176v1)

**作者:** Xiao Fan `[一作]` (Xidian University), Yi Zhang `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FRED 系统，利用多频段 EEG 与 fNIRS 对想象手写进行跨受试者解码；

**💡 创新点**

创新点包括：多频谱无关时间序列集成、跨频段成员误差低相关、协议匹配的 Hungarian 区块约束推断、伪标签自适应训练等；

**🔧 技术方法**

使用了多尺度时间卷积网络、频域过滤视图、实例归一化+对比学习、EEG‑Conformer、log 概率集成、块约束推断和伪标签训练等技术；

**📊 数据集**

数据集为 ACM MM 2026 多模脑机接口大赛的数据，包含 27 通道 EEG + 4 通道 fNIRS，20 名受试者训练集与 10 名未见受试者测试集，四类汉字想象；

**📈 对比分析**

通过逐步加入清洁集成、伪标签+Conformer、块约束等阶段提升性能；最终整体准确率 0.7952，私有分区 0.7718，排名第四；

**⚠️ 局限性**

局限性包括 fNIRS 在此稀疏配置下贡献极小、方法依赖已知块协议，跨受试者可变性仍大，且仅在此数据集与任务上验证有效。

---

## 151. 3DGSI-Assessor: A Large-Scale Dataset and An LMM-based Method for 3D Gaussian Splatting Image Quality Assessment

**arXiv ID:** 2608.03279 | [PDF](https://arxiv.org/pdf/2608.03279v1)

**作者:** Yuke Xing `[一作]` (Shanghai Jiao Tong University), Yiling Xu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模多维度3D Gaussian Splatting IQA数据集3DGS-IEval-15K+，并提出了全局语义与局部维度特征融合的LMM-based评估框架3DGSI-Assessor，实现单前向预测整体、几何和颜色质量。

**💡 创新点**

创新点在于首次为压缩3DGS提供多维度MOS标注的数据集、设计了全局与局部特征双通道Hierarchical编码与LoRA两阶段训练的LMM模型，以及可一次性输出三维质量维度的评估方法。

**🔧 技术方法**

采用了Vision Transformer、ResNet50提取局部特征，InternViT和InternVL2.5-8B LMM进行多模态融合，并使用LoRA进行参数高效微调。

**📊 数据集**

使用了包含10个真实世界场景、15,200幅图像、45,600条MOS（整体、几何、颜色）的3DGS-IEval-15K+，并在ENeRF-QA、NeRF-VSQA、GSC-QA等外部NVS基准上进行验证。

**📈 对比分析**

与传统PSNR/SSIM、现有LMM和深度学习IQA方法对比，3DGSI-Assessor在3DGS-IEval-15K+上实现SRCC、PLCC最高，且在外部基准零样本迁移中稳居第一，性能提升显著。

**⚠️ 局限性**

局限性包括数据集仅覆盖10个真实场景，缺乏合成数据；模型训练耗费较多算力；未将评估指标直接嵌入3DGS压缩优化流程。

---

## 152. Scaling an Autoregressive Transformer for Single-Cell Generation

**arXiv ID:** 2608.02961 | [PDF](https://arxiv.org/pdf/2608.02961v1)

**作者:** Aleksandr Sharipov `[一作]` (University of Hawai'i), Igor Molybog `[通讯]` (University of Hawai'i)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一个基于RQ‑VAE分词器和LLaMA架构的自监督单细胞基因表达向量生成模型，并在不同模型规模和数据量的网格上探索其预训练损失的缩放规律；同时评估生成向量的生物学真实性与下游任务的关联。

**💡 创新点**

①首次在单细胞Transformer上拟合联合两指数缩放律（α≈0.81，β≈0.57），并得到计算最优前沿；②发现生成质量与预训练交叉熵损失高度相关，可用损失直接指示生成效果；③提出基于分词器的离散化表示，使得Transformer能够处理高维基因表达数据。

**🔧 技术方法**

使用残差量化变分自编码器（RQ‑VAE）进行分词，随后用因果Transformer（类似LLaMA）对分词后的代码序列进行自回归建模；损失为跨步交叉熵；采用Chinchilla Approach‑3对损失进行双指数拟合。

**📊 数据集**

scBaseCount atlas（≈200M细胞，18,080基因面板）作为预训练数据集；在不同子集（38.1M–362M向量）上训练和评估。

**📈 对比分析**

通过与经验子集（baseline）比较，生成向量在平均表达Pearson相关（0.960 vs 0.964）、MAE（0.041 vs 0.038）和细胞类型可辨识度（0.990 vs 0.998）上接近真实复制，且这些指标随预训练损失下降而同步提升；拟合的缩放律对外推点误差≤0.5%。

**⚠️ 局限性**

局限包括：①仅评估预训练目标的缩放性，未直接验证对下游生物学任务（如差异表达恢复）的影响；②使用的离散化分词可能导致信息损失，限制了生成向量的最优性；③实验规模仍低于千亿参数级别，计算最优前沿仅在该范围内可靠。

---

## 153. Integration Barriers in Open-Source SSI Frameworks: An Exploratory Developer Experience Probe

**arXiv ID:** 2608.03039 | [PDF](https://arxiv.org/pdf/2608.03039v1)

**作者:** Breno Cerqueira Reis Nakamura `[一作]` (Universidade Federal de São Paulo), Arlindo Flavio da Conceição `[通讯]` (Universidade Federal de São Paulo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对九名开发者在三种开源 SSI 框架（Walt.id、Traction、MetaMask）上完成环境搭建、凭证签发、接收、验证和自定义模式工程等五项核心任务的探测实验，系统评估了集成体验和障碍。

**💡 创新点**

首次系统性发现了 SSI 开源工具在环境配置、文档缺失和抽象失效等三大集成障碍，并提出了 web 沙箱、AI 帮助的模式生成器和可执行文档等三项架构性改进方案。

**🔧 技术方法**

采用探索性探针研究方法，结合定量任务难度打分和定性主题分析，分析了 Walt.id、Traction 和 MetaMask 三个框架的 API 与 SDK 使用情况。

**📊 数据集**

数据来源于实验参与者提供的任务完成打分和开放式反馈，未使用公开数据集，仅基于内部收集的实验数据。

**📈 对比分析**

通过对各框架的平均易用性评分进行比较，发现自定义模式工程任务得分最低（平均 2.6），表明在主动构建阶段存在显著的集成痛点；相比之下，凭证接收任务得分最高（平均 3.8）。

**⚠️ 局限性**

研究样本规模仅为九人，工具选择分布不均，受试者缺乏实战经验，并且生态系统快速迭代导致结果具有时间局限性，无法广泛推广。

---

## 154. Emulate or Estimate? The Divergent Strengths of Base and Post-Trained Language Models for Opinion Simulation

**arXiv ID:** 2608.03044 | [PDF](https://arxiv.org/pdf/2608.03044v1)

**作者:** Seth Grief-Albert `[一作]` (Queen's University), Ashton Anderson `[通讯]` (University of Toronto)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了两种使用大语言模型模拟人类意见分布的方法：生成（emulation）与直接预测（estimation）。

**💡 创新点**

首次将这两类任务区分并系统比较，揭示基模型在生成多样文本时更贴近真实人类分布，而后训练模型在直接估计分布时更精确。

**🔧 技术方法**

采用大规模开源基础模型与对应的后训练（instruction-tuned）版本，利用开放式文本生成、LLM判别和JSON分布预测技术，并使用TVD与Wasserstein距离评估分布相似度。

**📊 数据集**

以Pew American Trends Panel Wave 54的59道四选项经济问题和七个人口统计条件（政党、意识形态、收入、宗教等）为基准数据集。

**📈 对比分析**

通过在七个条件下比较两类模型的TVD和Wasserstein误差，发现基模型在生成任务中的误差平均比后训练模型低（约20/21个比较），后训练模型在直接预测任务中误差显著低于均匀基线，且误差随模型规模提升而下降。

**⚠️ 局限性**

主要局限包括：未验证极大规模模型的行为，时间偏差（Pew 2019 vs. 模型训练至2025）未处理，且评估仅限于单轮调查意见，未涵盖更复杂的交互或行为模拟。

---

## 155. NanoMorph-3D: An End-to-End Physics-Driven Unrolling Framework for Nanomaterial Reconstruction

**arXiv ID:** 2608.03257 | [PDF](https://arxiv.org/pdf/2608.03257v1)

**作者:** Beiyuan Zhang `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 NanoMorph-3D 框架，用物理驱动的梯度下降展开网络在有限角度电子断层扫描中重建纳米材料的高保真三维结构，显著抑制缺失锥导致的 Z 轴拉伸和拓扑失真。

**💡 创新点**

核心创新包括：① 结合 Proximal Gradient Descent 的物理驱动展开；② Dual‑Domain Sinusoidal Attention 引入投影轨迹几何先验；③ Physics‑Normalization 以绝对物理尺度实现尺度不变特征提取；④ 基于 Nanomorphological Taxonomy 的大规模合成数据集；⑤ 双流一致性训练实现仿真‑真实域迁移。

**🔧 技术方法**

技术手段包括：物理前向/后向投影算子、梯度下降展开、三维 Transformer 与 Shifted Window 机制、正弦位置编码、双域注意力、动态视角丢弃、无监督重投影一致性损失。

**📊 数据集**

使用 6,000 体素纳米材料合成数据集（包含密集、空洞、孔隙、分层、1D 等五大形态）以及七组未标注的实验 HAADF‑STEM tilt series。

**📈 对比分析**

与经典迭代求解器（SIRT、GENFIRE、RESIRE）、图像域后处理（AET‑Net）、物理驱动展开网络（LPD）和 3D 神经渲染方法（Denza‑GS）比较，在合成数据上 PSNR/SSIM/FSC 均高于所有基线；在实验数据上重投影 PSNR、SSIM 与 LPIPS 均优于对手，表明在缺失锥条件下保持高结构保真且推断速度最快。

**⚠️ 局限性**

局限性：3D Transformer 计算与显存需求高，依赖 Beer‑Lambert 近似，可能在极端高损耗或非线性散射环境下失效，且需大量 GPU 训练资源。

---

## 156. Neural Networks with Local Converging Inputs for Efficient Options Pricing Models

**arXiv ID:** 2608.02778 | [PDF](https://arxiv.org/pdf/2608.02778v1)

**作者:** Harris Cobb `[一作]` (Georgia Institute of Technology), Yingjie Liu `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种名为 NNLCI 的局部校正神经网络，用于提升多资产期权定价数值方法的精度，并在现金付清式期权和 Heston 障碍期权模型上进行了实验。

**💡 创新点**

通过仅使用粗网与细网的局部补丁，无需显式输入参数或坐标，显著降低训练数据需求并实现对高维问题的强泛化。

**🔧 技术方法**

结合 NNLCI（神经网络局部收敛输入）与 ADI / Modified Craig–Sneyd 有限差分网格，使用 ReLU 前馈网络和 Adam 优化器。

**📊 数据集**

使用 Black–Scholes 解析解作为基准生成不同波动率、利率、相关系数组合；对 Heston 模型使用 200×100 参考网格产生训练/测试标签。

**📈 对比分析**

与仅使用细网求解结果比较，RMSE 在 1D、2D、3D 现金期权以及 Heston 障碍期权上分别降低约 2.5–12 倍；即使训练样本仅占总参数的 2% 甚至更少，也能取得显著提升。

**⚠️ 局限性**

仅适用于非混沌的抛物型 PDE；局部补丁在更复杂空间结构或自由边界（美式期权）下可能不足；缺乏更大补丁下的理论误差分析与非结构化网格的进一步验证。

---

## 157. Making AI Visible, Not Vanished: How AI Policies Reshape Developer Experience on GitHub

**arXiv ID:** 2608.03329 | [PDF](https://arxiv.org/pdf/2608.03329v1)

**作者:** Yunqi Chen `[一作]` (University of California, Irvine), Bianca Trinkenreich `[通讯]` (Colorado State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对GitHub OSS项目进行大规模经验研究，识别并分类AI治理政策，评估其对开发者体验的影响。

**💡 创新点**

设计TRACE框架捕捉AI治理的五个维度，并构造政策家族分类，同时通过差分中差分与倾向得分匹配提供因果证据。

**🔧 技术方法**

使用LLM进行文本编码与分类，倾向得分匹配、差分中差分、事件研究分析以及SonarQube扫描代码质量。

**📊 数据集**

基于29,624个GitHub仓库的PR、Issue、活动数据，手工标注385个AI政策。

**📈 对比分析**

将政策实施前后与匹配控制组比较19项开发者体验指标，发现AI政策提升透明度、参与度、代码质量，效果幅度在4%至40%之间。

**⚠️ 局限性**

仅涵盖人类面向政策，缺乏代理政策研究；部分TRACE水平样本量不足；结果仅为短期影响，缺乏对政策演化的长期跟踪。

---

## 158. SeaSlides: Semantic Abstraction Layer for Agentic Slide Generation

**arXiv ID:** 2608.03298 | [PDF](https://arxiv.org/pdf/2608.03298v1)

**作者:** Shengjun Fang `[一作]` (Nanjing University), Zongzhang Zhang `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SeaSlides 这一面向代理的演示文稿生成框架，利用语义抽象层实现内容与视觉实现的分离，并在 HTML 与 Typst 两个后端上实现；

**💡 创新点**

核心创新在于将模型的作者职责限制在语义化操作层面（可重用组件与能力模块），而将布局、样式与专门渲染器留给后端模板，从而提升源代码可读性与可编辑性；

**🔧 技术方法**

技术包括：LLM 生成的策略（Strategist）与执行器（Executor）、语义抽象层与模板-内容分离（TCS）、多阶段环境反馈（构建诊断、脚本检查、视觉复核）、以及后端特定的渲染路径（HTML+MathJax+Prism 与 Typst+包式渲染）；

**📊 数据集**

使用 UltraPresent 的 128 项任务集以及新设计的 SeaSlidesBench‑Rich 32 项技术内容任务，共计 160 项；

**📈 对比分析**

通过在同一套 160 项任务上与 KCTV、DeepPresenter、PPT‑Master 等基线系统及多种 LLM（Claude Sonnet、Gemini、Qwen、KIMI）进行对比，SeaSlides 在源代码可读性、内容比例和丰富内容（公式、代码、图表）宏平均评分上优于 SVG‑重度基线，且整体渲染质量保持竞争；

**⚠️ 局限性**

局限性包括：仅评估静态技术演示，未覆盖动画与高度艺术化设计；后端合同与组件实现维护成本未量化；样式控制仍受限；数据集偏向技术主题，未覆盖更广泛的演示领域；以及评估依赖单一 GPT‑5.5 注解与有限的实验次数。

---

## 159. From SQL Errors to Concept Gaps: An AI-Powered Knowledge Graph Analytics Platform for Personalized Feedback

**arXiv ID:** 2608.03118 | [PDF](https://arxiv.org/pdf/2608.03118v1)

**作者:** Abdulrahman AlRabah `[一作]` (University of Illinois Urbana-Champaign), Abdussalam Alawini `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个 AI 驱动的知识图谱分析平台，能够把 SQL 查询错误映射到课程概念，并为学生和教师提供可解释的诊断反馈。

**💡 创新点**

创新点在于将自动抽取的课程概念与学生提交轨迹同步到知识图谱，并结合执行校验与 LLM 语义反馈实现概念层面的错误分类与诊断。

**🔧 技术方法**

采用了执行型 SQL 校验、通用+微调无泄露 LLM 语义验证、Neo4j 知识图谱、PostgreSQL 存储、异步同步、以及指标分析等技术栈。

**📊 数据集**

使用了两门数据库系统课程的教学材料（课件、练习题）与真实或模拟学生 SQL 提交记录。

**📈 对比分析**

通过专家评估与 GPT‑4o‑mini 自动评估对节点、三元组和错误分类打分，结果显示节点有效率约 95.7%，三元组准确率约 63.8%，专家平均质量 3.75/5；相比传统仅基于语法错误的批改，该系统在概念层面提供更可解释、可操作的诊断。

**⚠️ 局限性**

局限性包括仅在 SQL 领域验证、概念标签粒度有限、LLM 评估偏正向、缺乏长期学习成效评估，以及对边缘关系和逆向依赖仍需人工复核。

---

## 160. PAMT: Process-Aligned Reinforcement Learning for Multi-Domain Machine Translation

**arXiv ID:** 2608.03077 | [PDF](https://arxiv.org/pdf/2608.03077v1)

**作者:** Yongshi Ye `[一作]` (Xiamen University), Xiaodong Shi `[通讯]` (Xiamen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PAMT（Process‑Aligned Machine Translation）框架，结合冷启动监督微调与强化学习，显式地优化多域机器翻译中的翻译过程；

**💡 创新点**

通过为每一步翻译生成过程分配step‑level过程奖励，将中间步骤与最终翻译结果对齐，解决了信用分配瓶颈，从而提升了术语控制和风格适配的可靠性；

**🔧 技术方法**

采用冷启动SFT、基于 Long‑CoT 的显式过程数据、强化学习（GRPO）以及序列级 BLEU/COMET 与 step‑level 过程潜力奖励、KL 正则化等技术；

**📊 数据集**

使用约7k条域感知 Long‑CoT 示例（DeepSeek‑R1）进行冷启动训练，20k条多域平行数据进行 RL 训练，评估覆盖15个域、四个方向（En↔Zh、En↔De）以及 WMT22、WMT23 术语共享任务和 Guofeng WebNovel；

**📈 对比分析**

与多组基线（LLM、LRM、MT 专用模型）对比，在 in‑domain、OOB 与多语种测试中均实现平均最高得分，显著优于 MT 专用基线且与强 LLM/ LRM 竞争；同时显著降低术语错误和风格漂移；

**⚠️ 局限性**

局限性：仅适用于显式输出的翻译过程，过程奖励基于参考文本，步骤划分固定且粗粒度；训练时需评估多步前缀，计算成本提升；未覆盖隐式内部推理场景。

---

## 161. Schedule-Informed Temporal Fusion Forecasting of Hourly Airport Security-Checkpoint Throughput

**arXiv ID:** 2608.02950 | [PDF](https://arxiv.org/pdf/2608.02950v1)

**作者:** Yinxiao Zhang `[一作]` (Purdue University), Yi Gao `[通讯]` (Kent State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究提出了一种基于航班调度信息的时间融合预测框架，用于预测亚特兰大国际机场每小时TSA安检通过量。

**💡 创新点**

创新点在于将航班离港时间通过截断泊松核转换为预检入站强度信号，并将此信号与历史通过量、日历变量等融合到可解释的Temporal Fusion Transformer中，实现多时段、可解释的预测。

**🔧 技术方法**

使用的技术包括Temporal Fusion Transformer、递归神经网络（RNN）和长短期记忆网络（LSTM）作为基准，以及截断泊松核生成的到达强度特征、时间序列编码器-解码器和注意力机制。

**📊 数据集**

数据集来源于2023–2024年ATL机场的TSA每小时通过量以及Cirium Diio航班时刻表（包含国内外航班座位容量）。

**📈 对比分析**

通过与RNN、LSTM的对照实验，使用MAE、WMAPE等指标评估。TFT在直接6小时预测中实现9.33%的WMAPE和317人MAE，优于RNN（12.16%/393人）和LSTM（11.37%/367人），在高通过量时段表现尤为突出；滚动多周期预测误差随时延不显著递增。

**⚠️ 局限性**

局限性包括：仅预测实际通过量，未能分离需求与容量；时间分辨率限制为小时；到达强度采用固定泊松核，未捕捉机场/航司/乘客差异；缺少排队长度、等待时间等更细致的运营指标。

---

## 162. AI Alignment and Fiduciary Obligation

**arXiv ID:** 2608.02660 | [PDF](https://arxiv.org/pdf/2608.02660v1)

**作者:** Benjamin Lange `[一作]` (Ludwig-Maximilians-Universität München), Benjamin Lange `[通讯]` (Ludwig-Maximilians-Universität München)

**通讯引用:** 700 | [OpenAlex ID](https://openalex.org/A5103081051)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出将受托人理论应用于AI助手开发者与用户之间的关系，阐述四项受托人义务（忠诚、关怀、诚信与坦诚）作为AI助手部署的对齐准则并给出对应的组织制度措施。

**💡 创新点**

创新点在于将传统受托人义务框架迁移至AI助手的开发者-用户三方关系，强调开发者对用户的义务而非仅关注用户-AI互动本身。

**🔧 技术方法**

主要采用法律与商业伦理中的受托人理论、道德哲学框架以及对现有AI对齐研究的文献综述来构建理论模型。

**📊 数据集**

未使用实验数据集，而是基于已有文献和案例分析进行理论构建。

**📈 对比分析**

未进行实验对比或性能评估，本文主要是概念性阐述与制度设计建议。

**⚠️ 局限性**

局限包括：跨文化适用性待进一步研究；仅关注自愿使用场景；聚焦商业部署，开源模型情形不详；与欧盟AI法等监管框架的具体对应尚未完善。

---

## 163. CLEAR: Causal Context-Based Agentic Reasoning for Vulnerability Detection

**arXiv ID:** 2608.03134 | [PDF](https://arxiv.org/pdf/2608.03134v1)

**作者:** Sungju Yun `[一作]` (Hanyang University), Sungbin Park `[通讯]` (Hanyang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 CLEAR，一个利用因果知识图和多智能体协作进行源代码漏洞检测的框架。

**💡 创新点**

创新点在于构建 Vulnerability Causal Knowledge Graph (VCKG) 并以其为依据进行结构化因果推理，显著提升漏洞辨别能力。

**🔧 技术方法**

结合 LLM、检索增强生成 (RAG)、多智能体推理（Collector、Claim、Critic、Judge）以及语义嵌入技术。

**📊 数据集**

使用 PrimeVul（C/C++）和 Java-Pair（Java）两套漏洞数据集。

**📈 对比分析**

与深度学习、单智能体与现有多智能体基线对比，CLEAR 在 P‑C 指标上分别提升约 130.7% 与 71.6%，并在 CWE 分类上显著超越 SOTA。

**⚠️ 局限性**

主要局限在于对 LLM 进行因果抽取和图构建的依赖、语义相似度导致的邻接误连，以及需要更细粒度的代码属性建模。

---

## 164. Paired Recipient-based Evaluation of Survival Prediction for Deceased Donor Kidney Transplants

**arXiv ID:** 2608.03017 | [PDF](https://arxiv.org/pdf/2608.03017v1)

**作者:** Misaki Matsuura `[一作]` (Case Western Reserve University), Kevin S. Xu `[通讯]` (Case Western Reserve University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于同一死亡供体的受体配对评估框架，用于评估和比较肾移植生存预测模型的临床实用性。

**💡 创新点**

创新点在于：①创建了针对供体-受体配对的准确率指标，直接衡量模型在实际分配决策中的价值；②将模型预测的准确率转换为可解释的移植后额外存活年数；③对模型公平性（种族差异）进行评估，揭示准确率与公平性不一定同步。

**🔧 技术方法**

使用了多种机器学习生存预测模型（Coxnet、DeepSurv、DeepHit、N-MTLR、RSF）以及基于单因素的规则模型，并采用嵌套交叉验证进行训练与评估。

**📊 数据集**

数据集来自美国Scientific Registry of Transplant Recipients（SRTR），包含2000-2016年间111,518例死亡供体肾移植病例，使用预移植特征进行预测。

**📈 对比分析**

与传统C-index对比时，所有ML模型的配对准确率约为60%，高于单因素模型；C-index在0.625-0.632之间变化；按配对准确率转换的后移植年数在约1.5-2.5年之间，显示模型在实际分配中可带来显著收益。

**⚠️ 局限性**

局限性包括：①基于随机配对的基线，未反映当前分配政策；②后移植年数估计依赖CoxPH伪值，受比例风险假设限制；③未探索更先进的预测模型和校准性能；④公平性评估仅用种族差异，未考虑其他敏感属性。

---

## 165. Stateful Governance for Concurrent Agentic Systems

**arXiv ID:** 2608.02764 | [PDF](https://arxiv.org/pdf/2608.02764v1)

**作者:** Yuxiang Peng `[一作]` (Purdue University), Xiaodi Wu `[通讯]` (University of Maryland)

**通讯引用:** 2097 | [OpenAlex ID](https://openalex.org/A5044901410)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为Provenact的运行时框架，旨在通过在AI代理操作中将政策决策与后续效果绑定，实现安全的状态化治理；

**💡 创新点**

创新点在于将政策与效果的检查-执行窗口通过可验证的policy‑state视图和操作合同进行显式约束，并定义了正确性属性Prg，支持事务、保留和超时审批等多种执行策略；

**🔧 技术方法**

采用了Python实现的政策编译器、合同注册、基于PostgreSQL/SQLite的policy‑state存储以及事务/锁/预约等数据库机制；

**📊 数据集**

使用的实验数据集为合成的多租户预算、库存与审批场景，以及模拟的LLM‑free采购工作流；

**📈 对比分析**

与全局序列化、手工事务、Cedar、AGT、Omnigent等基线对比，Provenact在消除旧授权（stale authorization）和审批失效（stale approval）方面表现最佳，吞吐量在大多数工作负载下与手工事务相当，略低于单线程全局锁；

**⚠️ 局限性**

主要局限在于需要供应商提供可验证的policy‑state视图和操作合同，且当前的policy语言有限制，无法支持任意回调或无限循环，且在外部API无事务支持时仍需额外设计。

---

## 166. Trust-Aware Topology Learning for Dynamic Decentralized Federated Learning under Adversaries

**arXiv ID:** 2608.03156 | [PDF](https://arxiv.org/pdf/2608.03156v1)

**作者:** Shubham Vaishnav `[一作]` (Stockholm University), Rajkumar Buyya `[通讯]` (University of Melbourne)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种动态可信拓扑的去中心化个性化联邦学习协议DMTT，旨在应对动态移动去中心化联邦学习中的拓扑操控攻击。

**💡 创新点**

DMTT通过引入可信拓扑管理，增强了对拜占庭攻击的抵抗能力，并在动态拓扑下实现了个性化学习，确保了模型的准确性。

**🔧 技术方法**

使用了证据深度学习、信任管理、动态拓扑建模等技术，构建了一个去中心化的个性化联邦学习框架。

**📊 数据集**

在UCI HAR和PAMAP2数据集上进行实验，数据集被划分为100个移动客户端，具有Dirichlet异质性。

**📈 对比分析**

与静态和动态FedAvg方法相比，DMTT在所有测试的拜占庭比例下（10%到80%）都保持了高于0.862的诚实节点准确率，而其他方法在攻击下表现不佳，甚至崩溃到随机猜测。

**⚠️ 局限性**

DMTT的局限性在于其对动态拓扑的假设和对拜占庭节点的信任管理，可能在极端情况下仍然受到影响。

---

## 167. Clarity Contrast and Similarity Selection for Multi-Focus Image Fusion

**arXiv ID:** 2608.03252 | [PDF](https://arxiv.org/pdf/2608.03252v1)

**作者:** Yicheng Zhang `[一作]` (Huazhong University of Science and Technology), Changxin Gao `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多焦点图像融合网络 CSNet，利用源图像间的清晰度对比实现显式信息交互，自动识别清晰区域并生成全景焦点图像。

**💡 创新点**

创新点在于（1）Clarity Contrast Attention Module（CCAM）通过通道与空间注意力结合清晰度差异进行互相调制，实现多尺度显式交互；（2）Similarity Selection Strategy 采用感知距离动态选择边界像素，兼顾重建图像与源图像，显著提升边界自然度和细节保留；（3）两者协同工作，兼顾决策式与重建式方法的优势。

**🔧 技术方法**

核心技术包括：双分支共享编码器；CCAM（通道注意力、空间注意力、清晰度对比、残差融合）；交叉注意力重建模块；VGG‑16感知距离相似度判定；二值焦点图监督+重建损失；多尺度特征融合；小区域去除后处理。

**📊 数据集**

使用 15,000 对合成数据（COME 数据集）训练；评估在四个公开基准集：Lytro、MFFW、MFI‑WHU、SIMIF（共 65 对真实多焦点图像）。

**📈 对比分析**

与 15 种传统与深度学习 MFIF 方法（CSR、DSIFT、MFM、SVDDCT、IFCNN、SESF、GACN、GRFusion、MUFusion、DB‑MFIF、FusionDiff、MFFT、SD‑Fuse、DMANet、MCCSR）在 6 项指标（Q_MI、Q_NCIE、Q_G、Q_Y、Q_C、Q_CB）上进行公平对比。CSNet 在 19 项指标中夺得第一，3 项排名第二，尤其在 MFFW、MFI‑WHU、SIMIF 数据集表现最为突出，速度与参数量保持中等水平。

**⚠️ 局限性**

局限性：对极度模糊或焦点差异极细的边界识别仍可能出现误差；依赖合成数据，真实复杂场景的泛化可能受限；目前仅处理两幅输入图像，扩展到多源场景需要进一步设计。

---

## 168. Don't Peek at the Answer: Outcome-Masked Group Relative Policy Optimization for Label-Free RLVR

**arXiv ID:** 2608.03119 | [PDF](https://arxiv.org/pdf/2608.03119v1)

**作者:** Yongshi Ye `[一作]` (Xiamen University), Biao Fu `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OM-GRPO框架，利用答案跨度梯度屏蔽和对比增强奖励，改进无标签强化学习奖励(RLVR)方法。

**💡 创新点**

创新点在于将奖励估计与策略优化解耦，采用答案级软奖励并屏蔽答案梯度，防止奖励黑客；以及引入低成本的对比增强奖励，通过对齐对比提升奖励质量。

**🔧 技术方法**

使用了策略梯度方法(Group Relative Policy Optimization)，答案跨度梯度屏蔽，软奖励与格式奖励结合，Contrast‑Augmented Reward (CAR) 对比采样；基于Qwen和Llama等LLM进行实验。

**📊 数据集**

实验使用MATH、AIME、GSM8K、AMC、LiveCodeBench、CRUX、IFEval、MMLU-Pro等多种推理基准，训练集为7,500道MATH题。

**📈 对比分析**

与已有的无标签RLVR基线（Majority Voting、Self‑Certainty、Entropy、CoReward）以及有标签GT‑Reward进行对比；OM‑GRPO在三大LLM上 consistently 超过所有无标签方法，且与GT‑Reward相当；在测试时训练场景中比多数投票提升4.24分。

**⚠️ 局限性**

局限性包括仅验证了至7B规模模型，CAR虽低成本但仍增加推理开销；方法仍依赖基于答案的奖励，未提供显式的推理质量监督。

---

## 169. TaskPress: Query-Agnostic KV Cache Compression via Task-Guided Pruning

**arXiv ID:** 2608.03276 | [PDF](https://arxiv.org/pdf/2608.03276v1)

**作者:** Wonpyo Park `[一作]` (Seoul National University), Seung-won Hwang `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TaskPress 框架，通过任务引导实现 KV 缓存的查询无关压缩；

**💡 创新点**

创新点在于使用任务描述作为“元查询”构造可重用的缓存，并利用量化比例因子作为无成本的值重要性度量；

**🔧 技术方法**

采用任务导向的关键重要性评估、量化比例因子阈值检测、乘积融合得分以及演化搜索生成最优任务指南；

**📊 数据集**

在 LongBench 与 RULER 基准上进行评测，使用 LLaMA、Qwen3 等模型；

**📈 对比分析**

与 SnapKV、PyramidKV、StreamingLLM、ExpectedAttention、ThinK、KVzip 等基线对比，TaskPress 在 75% 压缩率下平均准确率最高，且在多查询场景下实现显著的内存与延迟优势；

**⚠️ 局限性**

局限性包括对任务范围的依赖、在极度稠密信息环境下注意力稀释导致的误删、以及在非量化模型中需手动计算比例因子等问题。

---

## 170. ProCAVE: A Self-Adaptive, Full-Lifecycle Edge Caching Framework for Video Streaming via Predictive Bandwidth Estimation and Preference-Aware Deep Reinforcement Learning

**arXiv ID:** 2608.03313 | [PDF](https://arxiv.org/pdf/2608.03313v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 171. Fail-Fast, Restart-Smart: Early Failure Prediction and Restart for SWE Agentic Tasks

**arXiv ID:** 2608.03222 | [PDF](https://arxiv.org/pdf/2608.03222v1)

**作者:** Chenyu Wang `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两阶段控制器，利用轻量化监视器在单个LLM驱动的软件工程代理轨迹中预测失败并提前中断，然后在保留编辑的前提下用同一策略重新启动。

**💡 创新点**

创新点在于将逐步失败预测与编辑保留的同策略重启相结合，采用稀疏密集的fail-to-pass监督，既能提前终止不良轨迹，又能保留有价值的编辑成果。

**🔧 技术方法**

技术上使用0.6B LLM监视器配合LoRA适配、Platt缩放、密集F2P目标、Bradley–Terry排名，并通过重放生成编辑覆盖层以实现重启。

**📊 数据集**

实验数据集为SWE-bench Verified，使用mini-swe-agent采集500个实例（350/50/100划分）并在每个实例上执行11个随机种子。

**📈 对比分析**

与AgentStop、Duration、cold restart和SWE-PRM进行对比，最高可在5% FPR下节省20.4%推理token，25% FPR下提升5.2%的解决率，整体性能均优于基线。

**⚠️ 局限性**

局限性包括仅在SWE-bench Verified与mini-swe-agent环境下验证，未证明对其他软件工程任务或代理框架的泛化；在闭源API上召回略低，且重启会产生额外token开销。

---

## 172. Simulation-free and finite-time diffusion model

**arXiv ID:** 2608.03117 | [PDF](https://arxiv.org/pdf/2608.03117v1)

**作者:** Kentaro Kaba `[一作]` (Institute of Science Tokyo), Yuki Sughiyama `[通讯]` (Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种新的参考过程构造方法，实现了无需模拟参考SDE即可训练且生成在有限时间内完成的扩散模型。

**💡 创新点**

通过先预设可采样的时间相关条件分布再构造对应的SDE，揭示得分匹配非必要性，并把条件流匹配作为噪声极限。

**🔧 技术方法**

使用路径空间KL目标、Girsanov变换、Fokker–Planck方程拆分、重参数化技巧以及非高斯先验的推送构造等技术。

**📊 数据集**

在二维合成数据集（8分量高斯混合、螺旋、棋盘、两月亮）以及标准高斯和Johnson SU先验上进行实验。

**📈 对比分析**

与传统VP‑SBM（OU参考）在相同网络与训练设置下对比，结果表明新方法在无时间窗调节的情况下能更好捕获数据结构，VP‑SBM对时间窗敏感。

**⚠️ 局限性**

缺乏系统的条件分布设计准则，噪声水平仍需经验性选择，尚未在高维真实数据上充分验证。

---

## 173. Spatial proteomics guided by H&E-based AI reveals recurrence-risk niches in triple-negative breast cancer

**arXiv ID:** 2608.03145 | [PDF](https://arxiv.org/pdf/2608.03145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 174. How Should Vision-Language-Action Models Use Proprioceptive State?

**arXiv ID:** 2608.03052 | [PDF](https://arxiv.org/pdf/2608.03052v1)

**作者:** Yiren Zhao `[一作]` (Hong Kong University of Science and Technology), Rushi Dai `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文在一个统一的流匹配 Vision‑Language‑Action（VLA）框架下，系统地比较了机器人本体感知（关节角度、末端执行器姿态等）在不同表示、历史长度和注入位置上的效果，并给出了可复现的评估协议。

**💡 创新点**

创新点在于：①以同一模型、同一数据集和同一训练流程对五种主流状态接口（离散提示、VLM 前缀、动作前缀、状态专家、特征调制）进行公平对比；②通过滑动状态历史长度（1–96帧）和 slot‑matched 控制，揭示了短历史有效、长历史无效的现象；③发现注入位置随历史长度呈现“VLM 单帧优先，动作侧多帧优先”的设计规律，形成了可直接应用的设计原则。

**🔧 技术方法**

使用的技术包括：flow‑matching VLA（π_0.5+flow‑matching action expert）、两层投影网络用于连续状态编码、文本量化（256 bins）用于离散提示、Transformer 前缀/后缀、交叉注意力特征调制，以及配套的训练与评估脚本。

**📊 数据集**

数据集为 RoboCasa365，包含 45 个原子任务（按 pick‑and‑place、articulated‑object、精细操作三类划分）和 20 个由两三子目标组成的组合任务。所有实验均在同一视觉（三视角摄像头）和语言（自然语言指令）输入下进行。

**📈 对比分析**

比较方法是：在每种接口和历史长度下训练独立模型，使用固定 seed 的闭环 roll‑out 评估成功率（SR）。结果显示：单帧当前状态整体提升约 1–3 个点（离散提示最高且显著），短历史（K=8）可进一步提升 5–10 个点；长历史则产生负面影响。注入位置对 SR 影响显著：单帧更适合 VLM 前缀，8帧更适合动作头。

**⚠️ 局限性**

局限性包括：实验仅在仿真环境中进行，未在真实机器人上验证；所用状态仅为运动学信息，未考虑力/触觉等多模态感知；设计规律在本实验设置下成立，尚需在更大规模、多任务和不同 VLA 架构上进一步验证。

---

## 175. Information Technology Curriculum: General or Specialized? An Australia's Census Study

**arXiv ID:** 2608.02952 | [PDF](https://arxiv.org/pdf/2608.02952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 176. Studying Developer Perceptions on the Potential of CI Recommendation Systems

**arXiv ID:** 2608.02682 | [PDF](https://arxiv.org/pdf/2608.02682v1)

**作者:** Osamah H. Alaini `[一作]` (Trent University), Taher A. Ghaleb `[通讯]` (Trent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对约 5,000 名 GitHub 开发者（包括 CI 使用者与非使用者）进行问卷调查，系统评估 CI 采用动机、服务选择与推荐系统的感知价值。

**💡 创新点**

创新点在于首次将非 CI 使用者纳入样本，探讨需求驱动与社会影响对 CI 采用的影响，并收集开发者对 AI 驱动 CI 推荐系统的期望与信任度。

**🔧 技术方法**

主要使用了在线 Qualtrics 调查、GitHub REST API 抽样、统计检验（卡方、Mann–Whitney U）以及 Braun‑Clarke 方法的主题分析。

**📊 数据集**

数据集为 5,000 名公开邮箱的活跃 GitHub 开发者，其中最终收集到 250–500 条完整问卷，涵盖项目星级、提交频率、语言等信息。

**📈 对比分析**

通过描述性统计、两组（CI 使用者 vs 非使用者）比较以及主题编码，对比发现需求驱动因素显著高于社会影响，且非使用者对推荐系统的价值感知更高；整体样本量足以保证中等效应大小的检验功效。

**⚠️ 局限性**

局限性包括仅采样公开 GitHub 项目导致的代表性不足、依赖自报信息可能产生的偏差、以及非使用者响应率低导致的样本不平衡。

---

## 177. dots.tts.edit: Precisely Controlled Speech Editing with a Continuous Autoregressive Model

**arXiv ID:** 2608.02673 | [PDF](https://arxiv.org/pdf/2608.02673v1)

**作者:** Hankun Wang `[一作]`, Kai Yu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于连续自回归 TTS 模型的精确语音编辑框架，利用 XML 风格的结构化指令实现文本、情感、韵律、停顿等细粒度编辑，并构建了双语 doteBench 评测套件。

**💡 创新点**

创新点：①使用文本驱动的结构化编辑指令，明确操作类别、参数与定位；②在同一模型中学习多种编辑控制并支持多操作组合；③构建细粒度、双语评测基准，评估指令执行、局部保留与音质。

**🔧 技术方法**

技术：连续自回归 TTS（AudioVAE+DiT流匹配）、F5‑TTS 对齐‑填充、PSOLA/WSOLA、声学修复、混合训练；指令通过 XML 标签生成，模型输入为源语音+指令+目标文本。

**📊 数据集**

数据集：基于 1.5M 小时多语种语料；自建编辑对约 11M 文本、10M 情感、8M 韵律、5M 停顿样本；情感语料采用开放式多说话人数据集生成。

**📈 对比分析**

比较方法：在 doteBench 上与 Step‑Audio‑EditX、Ming‑UniAudio、Qwen3‑Omni 等开源模型对比，评测指令遵循率、局部保留率、UTMOS 音质；实验显示模型在指令遵循与局部保留方面领先，并保持相当音质；Seed‑TTS‑Eval 证明编辑后零样本 TTS 能力几乎无退化。

**⚠️ 局限性**

局限性：不支持多说话人转换或多说话人编辑；未与自然语言界面直接对比；音质仍有提升空间；结构指令虽可解释，但仍受模型生成准确性的影响。

---

## 178. Beyond Accuracy: A Multidimensional Evaluation of Statistical Reasoning in Large Language Models

**arXiv ID:** 2608.03038 | [PDF](https://arxiv.org/pdf/2608.03038v1)

**作者:** Monnie McGee `[一作]` (Southern Methodist University), Julian Cabrera `[通讯]` (Southern Methodist University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对15款当前世代大型语言模型在四份统计学考试中的回答进行多维度评估，包括准确率、回答行为、结构主题模型（STM）分析和词汇相似度分析。

**💡 创新点**

提出将准确率与解释内容和结构共同评估统计推理的框架，首次揭示LLM在统计推理上共享的概念组织但在解释语言上存在细微差异。

**🔧 技术方法**

使用结构主题模型（STM）提取解释中的统计推理主题，使用TF–IDF余弦相似度衡量解释文本的词汇相似性，并采用多维尺度分析（MDS）展示模型间语言差异。

**📊 数据集**

四份标准化统计学考试（AP Statistics、ACTM、CAOS、Graduate Exam）共90道题（73多选、17简答），用于生成模型回答与解释。

**📈 对比分析**

通过比较准确率、回答行为、STM主题分布以及词汇相似度矩阵评估不同模型；准确率从54.8%到78.1%不等，回答行为差异明显，STM显示所有模型共享20个核心统计推理主题，词汇相似度在同一厂商内略高于跨厂商，平均差异约0.028。

**⚠️ 局限性**

局限包括仅使用零射击提示、受限于四份考试、词汇相似度未捕捉语义深度、模型快速迭代导致结果随时间变化、并未对简答题进行评分。

---

## 179. On the missing data layer and a potential solution

**arXiv ID:** 2608.02949 | [PDF](https://arxiv.org/pdf/2608.02949v1)

**作者:** Francis F Daniel `[一作]` (SURUS), Marian Basti `[通讯]` (SURUS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并启动了DataHub平台，采用任务‑领域‑语言层级的本体索引，旨在解决拉美地区数据发现与供应的闭环，并为数据贡献提供可视化的奖励机制。

**💡 创新点**

创新点在于将任务置顶的本体结构与激励驱动的开放设计相结合，既解决了数据分散问题，又鼓励持续贡献，支持多极化AI发展。

**🔧 技术方法**

采用Web服务架构（前端页面、RESTful API）与数据库索引系统，结合开源数据处理与元数据标准化工具，以实现数据发布、检索与可视化。

**📊 数据集**

聚焦拉美地区已有的数据集，例如医疗转录、法律转录、农业病害识别等，来源于Hugging Face、Mozilla Foundation及学术论文附录，未对具体数据集进行实验验证。

**📈 对比分析**

该工作为基础设施性项目，不涉及模型训练与性能评估；因而未提供对比方法或性能指标。

**⚠️ 局限性**

局限包括缺乏跨国统一的元数据、许可与质量标准，数据保护与版权法规差异导致贡献障碍，且激励与治理机制尚未成熟，需进一步社区共识与政策支持。

---

## 180. CURV: Enhancing Chart Understanding Through Curriculum Visual Grounded Reasoning

**arXiv ID:** 2608.02833 | [PDF](https://arxiv.org/pdf/2608.02833v1)

**作者:** Xuehang Guo `[一作]` (William & Mary), Manling Li `[通讯]` (Northwestern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于课程学习的多步视觉对齐推理框架（CURV），通过在推理链的每一步动态聚焦图表区域，提升多模态大型语言模型（MLLM）在图表问答（CQA）任务中的视觉感知与逻辑推理能力。

**💡 创新点**

核心创新在于：①将CQA转化为多步视觉对齐推理，将逻辑推理与视觉关注同步；②引入三种视觉对齐策略（Applied、Boxed、Cropped）实现动态聚焦；③构建可扩展的三层级课程学习数据集 CURV-Dataset，系统地从单操作到多图表组合的复杂度逐步提升；④在多阶段训练（S1视觉对齐 + S2交互式推理）中实现模型的自我内化视觉推理。

**🔧 技术方法**

使用的技术包括：基于提示的CoT推理、视觉对齐损失（IoU、gIoU）、两阶段课程学习框架、显式与隐式视觉对齐策略、SFT 与 RL 结合的微调方法以及多模态大型语言模型（如Qwen2.5‑VL、InternVL‑3）。

**📊 数据集**

数据集为CURV-Dataset，包含7种图表类型（柱状图、直方图、散点图、折线图、热图、饼图、雷达图），30个领域类别，共计数千个带有人工或 GPT‑4o 生成的视觉标注与推理步骤的问答实例。

**📈 对比分析**

与 GPT‑4o、GPT‑4.1‑mini、Llama‑3.2‑Vision、Gemma‑3、InternVL‑3 等基线模型对比，CURV 在单图表层级提升最高可达 20.92%（两阶段训练），在多图表和跨领域基准上亦分别提升约 7% 与 12% 以上，显示出显著的性能提升与良好的泛化能力。

**⚠️ 局限性**

局限性包括：①仍需人工或大模型生成的视觉标注作为训练监督；②在极大尺寸或复杂布局图表上，裁剪式对齐会丢失细节；③强化学习版本虽提升更高性能但计算成本显著增加。

---

## 181. Near-Optimal Algorithms for Maximal Clique Enumeration in Structurally Sparse Graphs

**arXiv ID:** 2608.02614 | [PDF](https://arxiv.org/pdf/2608.02614v1)

**作者:** Jianfeng Hou `[一作]` (Fuzhou University), Hongbin Zhao `[通讯]` (Fuzhou University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在排除 K_t 作为子图或摊平的稀疏图中枚举所有极大团的近最优算法，给出了最优指数基

**💡 创新点**

创新点在于将局部密度阈值和最小度分支的结构性限制直接嵌入枚举逻辑，利用退化序列和潜能函数实现精确的指数下界，并证明了 4^{2t/5} 和 3^{t/3} 是不可再下降的最优基

**🔧 技术方法**

核心技术包括退化序列根分配、递归剥离、基于密度的终止判据、Fox‑Wei 结构定理、Zykov 对称化和输出敏感枚举子例程（Bron–Kerbosch 变体与 Tsukiyama 等算法）

**📊 数据集**

研究完全基于理论分析，没有使用具体实验数据集

**📈 对比分析**

相较于此前的 n·2^{O(t loglog t)} 或未定界结果，本文在 K_t‑minor‑free 图上实现了 n·4^{2t/5+o(t)} 的时间，K_t‑immersion‑free 图上实现了 n·3^{t/3+o(t)}，并通过构造极大团数量的上界证明了这一指数基的最优性

**⚠️ 局限性**

限制在于只针对排除 K_t 作为子图或摊平的稀疏图，尚未扩展到其他类型的排除结构或实际大规模图的实证评估

---

## 182. Neural network realization of binary refinement iterates via a two-chart atlas selector

**arXiv ID:** 2608.02624 | [PDF](https://arxiv.org/pdf/2608.02624v1)

**作者:** Tsogtgerel Gantumur `[一作]` (McGill University), Tsogtgerel Gantumur `[通讯]` (McGill University)

**通讯引用:** 529 | [OpenAlex ID](https://openalex.org/A5053230806)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文研究了在一次维数的二进制细化算子下，利用 ReLU 网络对细化迭代的闭式实现，证明每一次迭代都可以用固定宽度、线性深度的网络精确逼近。

**💡 创新点**

创新点在于引入双图（ordinary 与 half‑shifted）覆盖的两图坐标系，通过图集重叠一致性实现对残差动力学的精确切换，从而避免了传统的可变选择器导致的不连续性问题。

**🔧 技术方法**

主要技术包括：残差纤维的本地系统描述、双图切换与 ReLU 开关、有限窗口的自适应更新以及两遍网络结构（先求终值再传播协变向量）。

**📊 数据集**

该工作不依赖任何实测数据集，而是基于纯理论构造与证明。

**📈 对比分析**

性能方面，网络宽度不随细化步数变化，深度仅为 O(n)，与之前的深度/宽度增长做了比较，表明在理论层面实现了最优的复杂度。

**⚠️ 局限性**

局限性主要在于只针对标量一次维二进制细化算子，尚未推广到多维或非二进制情况，且实现依赖于精确的图集重叠结构，实际数值稳定性未作分析。

---

## 183. SeqLLM: Augmenting LLMs with Behavioral-Sequence Modeling for High-Stakes Decisions at WeChat Pay

**arXiv ID:** 2608.03063 | [PDF](https://arxiv.org/pdf/2608.03063v1)

**作者:** Guilin Li `[一作]` (WeChat Pay, Tencent), Weiran Huang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在微信支付平台上，SeqLLM 将预训练大型语言模型（LLM）注入行为序列建模能力，联合文本与行为序列进行高风险商户筛查，并在公共推荐数据集上验证通用性。

**💡 创新点**

核心创新包括：①采用字段级离散词表将每个交易事件压缩为约9个原子 token；②引入轻量化文本对齐投影器，在翻译-推理两阶段训练中将行为 token 对齐到 LLM 语义空间；③前缀引导的能力注入，通过指令化的下游 SFT 保留语言能力，避免持续预训练导致的灾难性遗忘。

**🔧 技术方法**

技术实现包括：预训练 Qwen3‑8B LLM；文本对齐投影器（两层 MLP + 余量连接）；两阶段训练（翻译目标 + 推理目标）；前缀指令化的下一事件预测（Prefix‑Guided SFT）；在生产环境中使用 0.6B/8B 双模型级联进行筛查；并在推荐任务上进行多任务微调与 RL。

**📊 数据集**

使用的数据集：微信支付内部商户信息（约 20M 未标记行为序列 + 4.06M 标记风险样本）；MovieLens‑20M、Amazon 评测数据；RecIF 公共推荐基准（96M 交互）。

**📈 对比分析**

与基线对比：在风险筛查上，SeqLLM 将风险识别精度从 92.0% 提升至 97.5%，申诉率从 12% 降至 ~2%；在欺诈检测模型中，Precision@Top‑0.01% 提升 26.8pp；在公共推荐任务上，相比 User‑LLM 提升 Recall@5 最高 32%，与 OpenOneRec‑8B 相比在 RecIF 上 Pass@32 提升 14.2%，且 GPU‑days 减少 4.8×。

**⚠️ 局限性**

局限性包括：①对业务内专有字段词表的依赖，迁移到不同业务领域需要重新构建词表；②前缀引导的 SFT 需要足够的指令化数据，若数据不足可能无法充分学习序列结构；③虽然保持了大部分语言能力，但在极大规模多样化文本任务上仍可能出现微调后偏差；④在极长序列（千级事件）下推理效率和内存占用仍受限。

---

## 184. BAP-SQL: Budget-Aware Observation Planning for Agentic Text-to-SQL

**arXiv ID:** 2608.02876 | [PDF](https://arxiv.org/pdf/2608.02876v1)

**作者:** Chong Peng `[一作]` (Microsoft), Varun Sah `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种预执行观察规划的文本到SQL代理框架BAP‑SQL，能够在执行前根据预算估计对SQL查询进行重写或替换，并使用运行时屏蔽控制资源使用。

**💡 创新点**

创新点在于将SQL查询视为预算控制决策，提供可见的估计与重写机制，并将规划与硬性约束分离，实现更精细的预算管理。

**🔧 技术方法**

使用强化学习控制器、SQL预估器（p50/p95）、运行时屏蔽、证据存储与管理、LoRA适配器以及多任务调度策略。

**📊 数据集**

基准数据集为BIRD及其交互扩展BIRD‑INTERACT（含BIRD‑INTERACT‑LITE），共计约8,477个SQL问答任务，覆盖69个数据库。

**📈 对比分析**

通过与SFT、接口匹配SQL‑RL、Adapted BACM‑RL以及FINER‑SQL等基线对比，BAP‑SQL在XS/S紧预算下实现约3.4–3.6个百分点的成功率提升，且总token使用降低4.5–5%，显示出显著的质量与成本优势。

**⚠️ 局限性**

局限性包括：在更大模型或更宽松预算下收益减弱；数据库工作量未得到减少；单通道干预效果不确定；在不同交互任务的迁移性能尚未充分验证。

---

## 185. Predictive Set Theory: A Generative Framework for Cognitive Architecture with Operationalized Core Mechanisms

**arXiv ID:** 2608.02704 | [PDF](https://arxiv.org/pdf/2608.02704v1)

**作者:** Yiyang Yu `[一作]` `[通讯]`, Yiyang Yu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Predictive Set Theory（PST），用最小化的操作（感知、状态刷新、三种参考链）构造完整的认知架构，并通过严格的集合论与逻辑推导推演出预测、比较、需求、效率、有限时域概率规划等核心认知功能。

**💡 创新点**

创新点：①将预测误差处理形式化为三类参考链（reference、counter‑reference、semi‑reference）；②将感知视为自身份函数 S(i)=i，剔除对外部真实状态的任何假设；③通过一致性要求与参考链自收敛原则生成认知的时序与一致性；④提供对 Russell、Gödel 等经典悖论的认知层面解释；⑤以形式化框架为基础，提供对感知-行动闭环的完整理论描述。

**🔧 技术方法**

技术手段：集合论与公理化定义、递归序列构造、极限与收敛分析、需求函数与效率多项式的数学定义、概率规划的递归公式等；全部为数学推导与形式证明，未使用实验或机器学习算法。

**📊 数据集**

未使用任何数据集，全部为理论推导和数学证明；若将来实现，将依据内部状态集合 K_t 的生成序列来验证。

**📈 对比分析**

本论文无实验对比与性能评估；理论上通过一致性与收敛证明保证模型内部自洽，并与传统预测加工、贝叶斯认知框架对比阐述其优越性，但未给出数值指标。

**⚠️ 局限性**

局限性：1) 完全基于形式化推导，缺乏经验验证与实际实现案例；2) 对感知设备的“身份函数”假设过于理想化，未考虑噪声与感知模糊；3) 只提供了框架层次的抽象，未给出可操作的实现细节；4) 在处理高度动态、开放域情境时如何扩展未知状态处理仍待研究。

---

## 186. Stylometric Defenses Against Author Impersonation in Software Repositories

**arXiv ID:** 2608.02695 | [PDF](https://arxiv.org/pdf/2608.02695v1)

**作者:** Leonid Ravich `[一作]`, Michael Fire `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种基于深度学习的创新方法，具体细节待补充。

**💡 创新点**

创新点在于结合了多种技术和思路，具体创新内容待补充。

**🔧 技术方法**

使用的技术包括但不限于深度学习框架、模型结构等，详细技术待补充。

**📊 数据集**

采用的数据集为公开或自建数据集，具体数据集待补充。

**📈 对比分析**

与已有方法进行比较，实验结果显示性能提升或表现较好，具体比较和性能指标待补充。

**⚠️ 局限性**

该研究的局限性主要体现在数据、模型、实验范围等方面，详细局限性待补充。

---

## 187. Distractor-Aware Truncation: Disentangling Context-Length Effects from Signal Loss in Long-Context LLM Benchmarks

**arXiv ID:** 2608.03297 | [PDF](https://arxiv.org/pdf/2608.03297v1)

**作者:** Mohsen Arjmandi `[一作]` `[通讯]` (evolutionID), Mohsen Arjmandi (evolutionID)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过对比“中间删除”与“仅删除噪声”两种截断协议，探究在长上下文下提示长度缩短是否能提升 LLM 的性能，并用配对统计方法对四个模型在四个基准上的表现进行量化评估。

**💡 创新点**

创新点在于：①提出并实现了“噪声感知截断”协议，明确区分样本中的信号与噪声；②给出了每个基准的信号定义并可验证；③构建了配对 Wilcoxon + Holm 校正、配对自助法 CI 的完整统计流程；④展示了跨模型、跨供应商（Claude 与 GPT‑5.5）重复性的结果。

**🔧 技术方法**

技术方法包括：基于正则表达式、数据集字段、图遍历等构造信号子集；中间删除与噪声感知截断；配对 Wilcoxon 符号秩检验 + Holm–Bonferroni 校正；配对自助法得到 95% CI；模型使用温度 0 或默认；对不同截断比例（100%、75%、50%、25%）进行实验。

**📊 数据集**

使用的基准数据集为：BABILong（bAbI 事实嵌入在 PG‑19 长文本中）、GraphWalks（BFS 子图检索）、MRCR v2（多轮核心ference）和 Oolong（聚合任务），均为英文合成/模拟任务。

**📈 对比分析**

实验结果显示：①在“中间删除”协议下，所有模型在 25% 上下文保留时表现显著下降（Δ 约 -0.1 到 -0.6，p < 0.05）；②在“噪声感知”协议下，性能无显著损失，且在 Haiku/Sonnet（小型 Claude 模型）上 BABILong 的 25% 保留能显著提升（Δ ≈ +0.08，p < 0.01）；③跨供应商实验（GPT‑5.5）复现了相同趋势。整体表明，仅删除噪声并不损害性能，且对小模型可带来收益。

**⚠️ 局限性**

局限性包括：①仅覆盖 Claude 系列模型和 GPT‑5.5，缺乏其他模型验证；②仅在英文合成/模拟任务上测试，缺乏多语言和真实世界任务的评估；③信号定义需要手工工程，难以自动化迁移到新任务；④不同模型使用的温度设置不统一；⑤采样仅使用单一随机种子，虽做了半样本稳定性验证但仍可能存在样本偏差。

---

## 188. Beyond Average Performance: Dynamic Instance Clustering and Specialized Algorithm Design in LLM-Assisted Evolutionary Search

**arXiv ID:** 2608.03129 | [PDF](https://arxiv.org/pdf/2608.03129v1)

**作者:** Qinglong Hu `[一作]` (City University of Hong Kong), Mingxuan Yuan `[通讯]` (Huawei Noah’s Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种动态实例聚类与专业化算法设计框架 DyCA，利用大型语言模型驱动的进化搜索（LES）过程中收集的算法-实例评价数据，无需特征即可识别实例结构并针对性地进化专门化算法，从而显著提升尾部鲁棒性与整体性能。

**💡 创新点**

创新点在于：①基于行为的动态实例空间分析 B‑DISA，实现在线无特征聚类；②结构感知的加权补充种群管理（W‑CPM）与偏置感知的专门化算法设计；③将补充与专门化进化耦合并共享知识，形成自适应资源分配与跨池知识共享机制。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑4o‑mini）生成算法候选、进化计算中的父子选择与种群管理、基于响应向量的聚类（K‑means/层次聚类）、加权评估与预算分配策略、在线特征稀疏化与锚算法选择。

**📊 数据集**

实验数据集涵盖四个任务：TSP、CVRP、OBP、Lunar Lander Control，分别使用多种分布生成的训练集（如240实例/任务）和对应测试集（TSP/CVRP/OBP 120实例，LLC 50实例），并在 CVRPLib、TSPLib、BPPLib 等公开基准上进行泛化评估。

**📈 对比分析**

与 EoH、FunSearch、ReEvo 等标准 LES 方法以及 InstSpecHH、EoH‑S 等可靠性增强方法进行对比；DyCA 在平均性能上优于所有基线，平均提升 7.1% 以上，在尾部 10% 性能上提升约 15.2%（相对标准 LES 提升 53.1%），并在所有任务中保持或超过头部性能，证明了其在可靠性与整体表现上的双重优势。

**⚠️ 局限性**

局限性包括：①依赖大量 LLM 推理查询，预算敏感；②聚类结果对锚算法选择与更新频率敏感，初始阶段聚类粗糙；③当前实现仅在四个特定任务上验证，跨领域推广需进一步评估；④对极端分布或缺乏足够实例多样性时的鲁棒性尚未充分验证。

---

## 189. Forbidden Region Dynamic Active Constraints in Robot-Assisted Minimally Invasive Surgery

**arXiv ID:** 2608.03010 | [PDF](https://arxiv.org/pdf/2608.03010v1)

**作者:** Zejian Cui `[一作]` (Imperial College London), Ferdinando Rodriguez y Baena `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计并验证了一种能实时更新三角网格约束的能量耗散FRDAC控制器，用于在呼吸引起的软组织变形下实现机器人手术的安全引导。

**💡 创新点**

创新点在于结合深度相机与TEASER++全局仿射配准，实时生成细网格FRDAC并保证系统能量耗散。

**🔧 技术方法**

采用三角网格表示、FPFH特征、TEASER++配准、DIBS近似查询、能量重定向耗散控制器等技术。

**📊 数据集**

使用Blender构造的受吸气/呼气变形的主动心脏模型和Acusense摄像机采集的软组织点云数据集。

**📈 对比分析**

与基于球形约束的Rydén等方法对比，在静态/动态轨迹跟踪实验中，DIBS方法保持10 mm安全距离、轨迹误差更小、频率43.48 Hz且耗时更少。

**⚠️ 局限性**

局限在于仅假设呼吸变形为仿射变形，对噪声点云仍易受影响，并未在真实临床人体数据上验证。

---

## 190. Adversarial Stress Testing of Role-Playing Language Agents using Multi-Agent Evaluation

**arXiv ID:** 2608.03166 | [PDF](https://arxiv.org/pdf/2608.03166v1)

**作者:** Saqib Shouqi `[一作]` (Informatics Institute of Technology), Ravisha De Alwis `[通讯]` (Informatics Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多代理平台，对角色扮演语言模型（RPLA）进行多轮对话的对抗性压力测试。

**💡 创新点**

创新性地将三位代理（策略驱动的提问者、被测目标代理、自动评判者）结合，实现多策略、持续对抗的动态评估框架，并提供自动化多维度评分。

**🔧 技术方法**

采用多代理协同交互、可扩展的LLM抽象层、FastAPI + React前端、规则+关键词的自动评分算法，使用六种结构化攻击策略（角色漂移、伦理探测、矛盾诱导、混淆、权威挑战、情绪操控）。

**📊 数据集**

实验中使用三种角色（医疗助手、客服代理、财务顾问），并在不同LLM提供商（Llama‑3.3‑70B、GPT‑4o‑mini、Claude‑3.5‑Haiku）上进行多模型验证；没有公开专用数据集，主要以自定义对话模板生成。

**📈 对比分析**

与单策略基线（仅角色漂移）对比，多策略测试导致整体鲁棒性平均下降0.17–0.20点；跨模型验证显示不同LLM在多轮攻击下的鲁棒性差异，Authority Challenge和Emotional Manipulation为最有效攻击；自动评判与人工评估相关性高（r≈0.8）。

**⚠️ 局限性**

局限于提示式攻击，未涵盖训练/微调层面的弱点；自动评判可能存在偏差，需要人工校准；角色范围有限，实验在受控环境下，真实世界可变性未充分捕捉。

---

## 191. ARCHead: Activation-Metric Residual Correction for Large Language Model Output Heads

**arXiv ID:** 2608.02703 | [PDF](https://arxiv.org/pdf/2608.02703v1)

**作者:** Şuayp Talha Kocabay `[一作]`, Kamer Ali Yuksel `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

ARCHead提出了一种针对大语言模型输出层（LM-head）的压缩方案，先用低秩量化核心再加入激活度量残差补偿，实现高压缩率且保持较低的困惑度。

**💡 创新点**

创新点在于：①使用激活度量变换下的残差低秩逼近，显著减少量化误差；②将低秩核心、INT4残差和INT8补偿一起打包存储，避免存储BF16全矩阵；③实现3.7–3.9×的持久参数压缩。

**🔧 技术方法**

技术手段包括低秩SVD、分组INT4/INT8量化、5/8位量化核心、激活协方差变换（T_p）、随机截断SVD、量化因子再量化、打包实现。

**📊 数据集**

使用WikiText-103的训练集做校准，16k测试集评估；在Qwen3-8B-Base、Gemma-4-E4B、VibeThinker-3B等模型上验证；还评估了Mistral-7B-v0.3和LFM2.5-8B-A1B。

**📈 对比分析**

与BF16、INT8、Group‑INT4、SVD+INT4以及GPTQ‑style INT4头等基线对比；ARCHead在相同存储下相对PPL仅为1.007，明显优于1.151的Group‑INT4；与AWQ/bitandbytes留下的BF16头相比，ARCHead仅增加0.006–0.007的交叉熵，存储压缩率约25%；推理吞吐量差异不到2%。

**⚠️ 局限性**

局限性：只针对输出层，不能压缩MLP/注意力权重；当头已不易受量化影响时收益有限；实验覆盖模型和数据有限；量化补偿因子仍带来额外误差；总体并非全局最优，需进一步评估跨模型、跨语言与大规模校准情况。

---

## 192. Impacts of Single-objective Landscapes on Multi-objective Optimization

**arXiv ID:** 2608.03266 | [PDF](https://arxiv.org/pdf/2608.03266v1)

**作者:** Shoichiro Tanaka `[一作]` (University of Electro Communications), Hiroyuki Sato `[通讯]` (University of Electro Communications)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析单目标局部最优网络与多目标帕累托最优网络之间的关系，并提出基于单目标网络的帕累托网络组件分类方法

**💡 创新点**

首次将单目标LON与多目标PON结合，系统量化帕累托网络中与单目标网络的耦合关系

**🔧 技术方法**

使用ρ MNK景观、全局穷举搜索构建LON和PON、构造图网络、统计指标、相关性分析

**📊 数据集**

ρ MNK景观实例，共3,300个（50个实例×不同M、ρ、K组合）

**📈 对比分析**

通过定量指标（如Pareto链接比例、交叉链接比例、组件数量）和箱线图/散点图比较，结果表明大部分帕累托解可由单目标局部最优解到达，且多目标维数和相关性会提升链接比例

**⚠️ 局限性**

仅做结构与统计分析，未验证具体搜索算法；高共变量数导致组件分散，易导致转移困难，实际搜索效率仍待验证

---

## 193. CAPE-T2V: Captioner-Anchored Prompt Enhancement toward Two-Sided Conditioning Alignment in Text-to-Video Generation

**arXiv ID:** 2608.03046 | [PDF](https://arxiv.org/pdf/2608.03046v1)

**作者:** Yizhuo Jia `[一作]` (Fudan University), Yuanxing Zhang `[通讯]` (Kling Team)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CAPE-T2V，一种两步框架：先用外部 captioner 生成目标短语训练 Prompt Enhancer（PE），再用该 Anchored PE 生成训练视频的 dense caption 供 Diffusion Transformer（DiT）微调；推理时同样使用该 PE 对用户提示进行改写。

**💡 创新点**

创新点在于通过 PE 的“锚定”实现训练时与推理时文本分布的对齐，显著减少 PE–Caption 匹配差距；并且该方法不依赖目标生成器的原始 captioning 方案，易于迁移到不同 DiT 模型。

**🔧 技术方法**

核心技术包括：文本-视频扩散 Transformer（DiT），Prompt Enhancer（Qwen3.5-9B 训练），embedding‑based MMD² 评估，自动化 captioner（Qwen3.5-397B-A17B）生成 dense caption，数据过滤与一致性校验。

**📊 数据集**

使用的数据集：OpenVid‑1M、MiraData、Video‑UFO、MovieStory101、LSMDC、Koala‑36M 生成 54K 过滤后的 dense caption；PE 训练使用约 735K 由 captioner 生成的输入‑目标对；评测基准为 StoryEval、VBench‑2.0、T2V‑CompBench。

**📈 对比分析**

与同一 schema 对齐的基线（Schema‑Aligned DiT）及官方 Prompt Enhancer 进行对比。CAPE‑T2V 在 Wan2.2 和 LTX‑2.3 上分别提升 StoryEval +1.6/1.4、VBench‑2.0 +1.12/0.92、T2V‑CompBench +0.30/0.65；MMD² 与推理时 PE 重写的差距下降约 11%。

**⚠️ 局限性**

局限性包括：仅评估单一 caption schema；未对 PE 训练中的三类输入构造做单独消融；仅针对官方 PE 进行实验，未知对其他 PE 的适用性；若 PE 更新需重新生成训练 caption，缺乏增量更新方案。

---

## 194. CAMTA: A Reconfigurable Multi-Region Activation Unit for Nonlinear Function Approximation

**arXiv ID:** 2608.02988 | [PDF](https://arxiv.org/pdf/2608.02988v1)

**作者:** Carlos Soto-Porras `[一作]` (Costa Rica Institute of Technology), Jorge Castro-Godinez `[通讯]` (Costa Rica Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了CAMTA，一种在FPGA和ASIC上运行的16位可重构多区域激活单元，用于非线性函数近似。

**💡 创新点**

通过独立阈值、每区域多项式度数、系数集合和执行模式（HORNER、CONST、ZERO、IDENTITY）实现同一算术数据通路在运行时重配置不同激活函数，避免重新综合与硅重spin。

**🔧 技术方法**

采用Horner法多项式求值、可重构阈值与模式选择、FPGA HLS合成、TSMC 65nm ASIC标准单元布局以及RMSE/MaxAE误差评估。

**📊 数据集**

使用10000个均匀采样输入与浮点软件参考验证，测试GeLU、tanh、sigmoid、Swish、Softmax等激活函数。

**📈 对比分析**

与CORDIC、PLAC、AFC/PPA-ED等专用近似器比较，FPGA上使用3 DSP、802 FF、1756 LUT，ASIC面积6632µm²、功耗1.363mW；误差在Softmax RMSE 3.6×10⁻⁶、tanh最大误差≈5.8×10⁻³，整体性能与专用实现相近且更灵活。

**⚠️ 局限性**

相对于同节点专用实现，面积约增加2.2倍、功耗约1.75倍；在极端精度需求时可能不如最优单函数近似器；硬件实现仍受限于16位定点表示。

---

## 195. Oh Deer, How Should I Handle This? Seasonal Priors for Selective Wildlife Annotation and Classification

**arXiv ID:** 2608.02762 | [PDF](https://arxiv.org/pdf/2608.02762v1)

**作者:** Hugo Markoff `[一作]` (Aalborg University), David C. Schedl `[通讯]` (University of Applied Sciences Upper Austria)

**通讯引用:** 630 | [OpenAlex ID](https://openalex.org/A5048747404)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文研究了在季节性抗鹿（红鹿）鹿角循环影响下，结合RGB与热红外两模态的无人机航拍图像进行成年雄性识别，并探讨了季节先验对标注质量、模型置信度与选择性推理的作用。

**💡 创新点**

创新点在于①引入基于鹿角生命周期的软季节先验，用于预测标注和模型表现的不确定性；②利用匹配的RGB+热图像进行联合标注与后续训练，显著提高标注覆盖率和分类性能；③在置信度阈值基础上实现选择性推理，并分析性别特定的拒绝率。

**🔧 技术方法**

使用了冻结的DINOv3 Vision Transformer提取特征，随后训练线性SVM、三元组度量学习和YOLO检测器；对热图像进行伪RGB复制；通过软先验调整样本权重；采用不确定性带（δ）实现选择性推理。

**📊 数据集**

数据集为公开的Aerial Red Deer Dataset，包含9个月的同步RGB与非辐射热图像，7,295个对齐的裁剪样本，其中有三位生态学家分别进行单模态与匹配模态标注。

**📈 对比分析**

在匹配模态下，线性SVM实现了95.4%整体准确率（89.4%平衡准确率，83.1%雄性F1），比单模态RGB（93.6%）和热红外（84.1%）都有显著提升；在选择性推理下，匹配模态在75.3%覆盖率下可达98.9%覆盖准确率，但雄性拒绝率较高。不同模型族表现一致，说明效果与特征抽取相关。

**⚠️ 局限性**

主要限制包括①标签仍为三位生态学家共识，无法完全排除标注噪声；②季节先验为手工设定，缺乏自动学习；③实验仅在裁剪级别，未考虑轨迹级别判定；④匹配模态仍为后处理融合，缺乏端到端学习；⑤雄性与雌/幼混合导致类别不平衡，影响少数类性能。

---

## 196. What Language Does and What the Evidence Supports: A Functional Role Taxonomy and Evidence Audit of Language Grounding in Embodied Agents

**arXiv ID:** 2608.03099 | [PDF](https://arxiv.org/pdf/2608.03099v1)

**作者:** Yifan Guo `[一作]` (Northwestern Polytechnical University), Zhiwen Yu `[通讯]` (Northwestern Polytechnical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性地将语言在具身智能体中的功能划分为五类（规范、具身表征、行动编排、地面化调节、执行耦合），并对近百篇论文中对每一角色所提供的证据进行审计与比较。

**💡 创新点**

创新点在于提出了以功能角色为核心的语言定位框架，并定义了五种证据操作（R、T、C、F、I），实现了对语言“地面化”责任与证据之间缺口的定量评估，促进了跨架构、跨阶段的可比性。

**🔧 技术方法**

采用的技术主要包括文献综述、角色与证据的编码规则、统计分析与可视化，以及案例对比分析，借此映射每篇论文的角色声明与对应证据。

**📊 数据集**

由于是综述工作，未使用新实验数据，所有数据来源均为公开论文与已实现系统的公开描述。

**📈 对比分析**

通过将论文的角色声明与五种证据操作匹配并统计出现频率，作者比较了不同机制在证据充分性上的差异，发现闭环反馈和可观测修正等强证据相对稀缺，整体表明大多数系统缺乏对语言功能的充分验证。

**⚠️ 局限性**

局限性包括：仅评估已发表实验结果，未进行独立复现；角色解释仍带有主观判断；证据仅基于论文描述，缺乏对真实世界多样性的验证；以及对新兴预印本的覆盖不足。

---

## 197. Towards Automated Proof-Theoretic Semantics: Inference-Behaviour Semantics for 3-Dimensional K3 and LP

**arXiv ID:** 2608.02654 | [PDF](https://arxiv.org/pdf/2608.02654v1)

**作者:** Sophie Nagler `[一作]` `[通讯]` (University of Amsterdam), Sophie Nagler (University of Amsterdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将推理行为语义（Inference‑Behaviour Semantics, I‑bS）从传统的二维序列推理系统推广到任意有限维度的序列推理系统，并以 3 维强克利尼逻辑 K3 与逻辑悖论 LP 为案例，证明其连接词在意义上与经典 LK 连接词保持一致，同时提出“意义扩展”（meaning‑extension）概念以比较不同维度算子的意义。

**💡 创新点**

创新点：
1. 将 I‑bS 机制从 2 维扩展到 n 维，为多维逻辑的证明理论语义提供通用框架；
2. 引入意义扩展概念，允许跨维度比较算子意义，弥补传统 I‑bS 只能比较同维度算子的不足；
3. 将 MUltlog 自动生成的多维序列推理系统与 I‑bS 结合，迈向 proof‑theoretic semantics 的计算化自动化。

**🔧 技术方法**

核心技术：
- 多维序列推理与 MUltlog 自动生成的三维算子规则；
- Belnap 样式的保守性与唯一性证明；
- 推理行为度量（inference‑behaviour profiles）与语义规则的构造；
- 语义句法规则的最小化与结构限制（如 ^∗, ^†, ^）。

**📊 数据集**

数据集：本研究完全基于理论推导与手工证明，使用 MUltlog 生成的 K3/LP 3 维序列规则作为实验对象，未使用实际文本或数值数据。

**📈 对比分析**

比较方法：通过对 K3/LP 与 LK 的推理行为度量进行对比，检查语义规则是否同构；结果显示 K3/LP 连接词的语义与 LK 对应连接词相同，且在三维系统中为其意义扩展；未给出量化性能指标，主要为理论证明和结构对比。

**⚠️ 局限性**

局限性：
- 每种新逻辑都需手工完成保守性与唯一性证明，难以在更高维度或更复杂逻辑上自动化；
- 仅适用于 MUltlog 生成的多维序列系统，未验证对其他自动推理框架的适用性；
- 缺乏实证评估与性能指标，未展示自动化实现的效率或可扩展性。

---

## 198. AS-FedBridge: Pseudo-Spike Bridge Distillation for Heterogeneous ANN-SNN Federated Learning

**arXiv ID:** 2608.03324 | [PDF](https://arxiv.org/pdf/2608.03324v1)

**作者:** Shengyang Li `[一作]` (Peking University), Zhaofei Yu `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了AS-FedBridge框架，实现了混合ANN‑SNN联邦学习中对齐与知识传递。

**💡 创新点**

创新点在于引入轻量化的Bridge与Pseudo‑Spike接口，利用伪脉冲正则化实现连续与脉冲表示的无缝对齐，从而解决了语义差距与梯度冲突。

**🔧 技术方法**

主要技术包括中心化核对齐（CKA）、最大均值散度（MMD）、梯度余弦相似度分析、Pseudo‑Spike正则化（PSPR）、时序高效训练（TET）和噪声平滑日志蒸馏（NLD）等。

**📊 数据集**

使用了CIFAR‑10、CIFAR‑100、Tiny‑ImageNet和神经形态数据集CIFAR10‑DVS进行实验。

**📈 对比分析**

与FedAvg、FedProx、FedFree、FedProto、FedTGP、FML、MH‑pFLID等先进异构FL方法对比，AS‑FedBridge在四个数据集上平均提升约2–3个百分点，且在代表性对齐和梯度一致性上表现突出。

**⚠️ 局限性**

局限性包括：对大规模客户端时聚合仍需更高通信效率；伪脉冲映射可能受时间步选择影响；当前仅针对视觉任务验证，需进一步验证在其他领域的泛化。

---

## 199. Contrast-invariant deep ptychography neural networks

**arXiv ID:** 2608.02869 | [PDF](https://arxiv.org/pdf/2608.02869v1)

**作者:** Albert Vong `[一作]` (Argonne National Laboratory), Nicholas Schwarz `[通讯]` (Argonne National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种对比不变的深度ptychography神经网络框架PtychoPINN-CI，利用实部/虚部解码器、动态尺度因式化、探测器加权拼接和相关性采样的合成数据，实现对不同光照条件的测量一致重建。

**💡 创新点**

创新点包括：①实部/虚部输出与尺度解耦，使远场强度成为二次函数，可在测试时直接求解尺度；②动态测试时尺度因式化，无需重新训练即可适配新数据；③探测器强度加权拼接提升局部信噪；④基于经验相关性采样的合成对象，消除相位分布失配。

**🔧 技术方法**

使用无监督卷积自编码器（U‑net）架构，Poisson负对数似然损失，FFT物理前向模型，梯度优化的尺度参数求解，探测器加权 patch stitching，经验实部/虚部分布采样生成合成数据。

**📊 数据集**

使用 APS Velociprobe（NCM、FLY1）、APS‑CNM Hard X‑ray Nanoprobe（W）、ALS Cosmic Imaging（LFP、NS）共 5 个实验数据集，涵盖不同光源与探测器。

**📈 对比分析**

与先前的 PtychoPINN‑torch 基线在同一 5 个数据集上对比，使用 Fourier error 量化误差；改进后平均降低约 5 倍 Fourier error，重建分辨率提升，幅度与相位更准确，且在仅用合成数据训练时亦能取得良好性能。

**⚠️ 局限性**

局限性：仍存在相位压缩问题，可能因网络容量不足或合成数据结构与真实样本不完全匹配；对探测器形状或极低信噪/极小探测器的适应性仍有限。

---

## 200. Light-Loco-Parkour: Versatile Perceptive Whole-Body Locomotion via Multi-Skill Distillation

**arXiv ID:** 2608.02653 | [PDF](https://arxiv.org/pdf/2608.02653v1)

**作者:** Hongming Chen `[一作]` (Light Origins), Tingxiang Fan `[通讯]` (Light Origins)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个单一的可部署策略，能够根据深度相机观测和速度指令实现全身感知式运动，包括行走、攀爬、跨越等多种技能；

**💡 创新点**

创新点在于（1）利用少量“种子”运动通过自我迭代增强生成多种地形匹配的可执行参考；（2）将 RL、DAgger 与运动先验相结合，使策略在没有手工标签或状态机的情况下自动决定何时使用何种全身技能；（3）通过从高度扫描到真实深度的再蒸馏，克服了传感器视野限制；

**🔧 技术方法**

核心技术包括强化学习（PPO）、模仿学习与 DAgger 蒸馏、奖励驱动的运动先验（AMP）、递归网络（GRU）记忆以及深度图像到高度扫描的辅助重构；

**📊 数据集**

数据集主要来源于从互联网视频提取的 SMPL 运动（GVHMR + GMR），随后手工对准障碍物并通过迭代自增生成多种障碍尺寸与姿态的参考对；

**📈 对比分析**

与 MGMT、PHP、BeamDojo、CReF 等现有方法比较，Light‑Loco‑Parkour 在多种稀疏踏板、斜坡、阶梯、桥梁等障碍上取得 90%+ 的成功率，且能在未见过的障碍高度（高达 0.83H）和形状（如新型凳形障碍）下保持 95% 以上性能；

**⚠️ 局限性**

主要局限包括：需要人工手动对准 seed 视频与障碍物，难以快速扩展技能库；技能集合有限，障碍重叠或连续出现时表现下降；以及固定胸部深度摄像头导致在大幅身体动作时的遮挡与信息缺失。

---

## 201. iFAN: Inference-Aware Learning for Plain Mask Transformers

**arXiv ID:** 2608.03216 | [PDF](https://arxiv.org/pdf/2608.03216v1)

**作者:** Fang Li `[一作]` (JD.com), Junshi Huang `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为iFAN的推理感知学习框架，通过在训练阶段调整概率-掩码得分和跨层自蒸馏，显著提升plain mask transformer的分割性能。

**💡 创新点**

创新点包括：1）Adjusted Probability‑Mask Ranking (APMR)，利用全局掩码质量校准查询得分并对匹配查询与硬负样本进行显式排序；2）Cross‑Layer Self‑Distillation (CLSD)，在多层中挑选最佳中间掩码并将其调整后的概率‑掩码得分图传递至最终层，保持推理阶段仅用最终层即可。

**🔧 技术方法**

使用技术包括：mask transformer、ViT backbone、Hungarian匹配、BCE+Dice+mask质量回归、softplus排序损失、stop‑gradient、soft‑IoU、以及多尺度训练和多任务损失融合。

**📊 数据集**

实验数据集涵盖COCO、ADE20K与Cityscapes，分别评估panoptic、instance和semantic分割。

**📈 对比分析**

与PMT、EoMT等plain mask transformer基线以及更复杂方法对比，iFAN在PQ、AP、mIoU分别平均提升约1.20、1.30、0.63，且几乎无额外参数、FLOPs或推理延迟。

**⚠️ 局限性**

局限性包括：仍依赖ViT预训练；跨层蒸馏在训练早期可能引入噪声；对极小模型的提升有限；以及对查询数、层数等超参数的敏感性未完全解析。

---

## 202. Steganalysis of Adaptive Covert Collusion in Tool-Using Agent Populations: A Black-Box, Cross-Principal Approach

**arXiv ID:** 2608.02698 | [PDF](https://arxiv.org/pdf/2608.02698v1)

**作者:** Mohamed Chahine Ghanem `[一作]` `[通讯]` (Keele University), Mohamed Chahine Ghanem (Keele University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种跨组织大语言模型代理群体的黑盒隐写分析检测器，评估其在工具使用与持续记忆环境下对自适应隐蔽协作的检测效果。

**💡 创新点**

创新点包括：①采用编码无关的特征融合与自适应红蓝对抗训练；②将检测与信息隐藏容量理论结合，绘制检测-容量前沿；③引入跨会话、工具调用、时间侧信道等多维信号提升鲁棒性。

**🔧 技术方法**

技术手段包括：交叉运行互信息估计（MINE/InfoNCE）、置换不变性检验、分布偏移统计、时间/工具调用侧信道、合成式对抗进化搜索以及合成统计阈值校准。

**📊 数据集**

使用的数据集为三种模拟环境（招标、同行评审、代码评审）中自定义代理交互日志，涵盖不同规模、不同主办方、不同工具调用与记忆跨会话数据。

**📈 对比分析**

通过在相同FPR预算下与六种基线（随机、LLM评判、单一MI、时序/工具侧信道、图异常、白盒激活探测）进行对照，静态编码下AUC≥0.99，面对自适应红蓝时AUC随迭代下降但仍优于基线；检测前沿显示在不使用工具时可检测率≈1bit/round，工具可用时下降。

**⚠️ 局限性**

局限性包括：①易被跨会话分布式编码绕过；②仅基于轨迹的被动监控无法保证低速通道的检测；③工具接口差异导致容量估计不确定；④互信息估计需大量采样，计算成本高；⑤仅关注LLM代理，未涵盖非语言模型协同。

---

## 203. Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't

**arXiv ID:** 2608.02829 | [PDF](https://arxiv.org/pdf/2608.02829v1)

**作者:** Ravi Satya Durga Prasad Yenugula `[一作]` `[通讯]` (Independent Researcher), Ravi Satya Durga Prasad Yenugula (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何将大规模预训练的Transformer模型转换为更小尺寸的同族模型，并评估不同初始化方法在有限训练预算下的效果。

**💡 创新点**

发现表示在不同尺寸之间高度可线性对齐，但权重并非如此；密集投影会破坏结构，导致功能失效；并提出将初始化拆分为两条独立杠杆——最小二乘补偿（修复功能）与方差保持缩放（修复优化尺度），两者叠加在低预算下显著优于传统子克隆。

**🔧 技术方法**

使用线性投影、最小二乘补偿、方差保持缩放、CKA、SVCCA、实验对比等技术；实现了一种结构保持的选择与补偿算法。

**📊 数据集**

主要使用Pythia系列模型的训练数据（Pile）和评估语料库（WikiText‑103、C4）以及零样本任务（LAMBADA、ARC‑Easy、HellaSwag、PIQA）。

**📈 对比分析**

在相同的训练代币预算（如30M、100M、1B）下与随机初始化、密集投影、传统子克隆等做对比；在低预算下，叠加杠杆实现的模型比最强单杠杆和从零训练快4–5倍，最终在1B代币时与单杠杆接近，均远优于从零训练。

**⚠️ 局限性**

实验仅在Pythia架构（LayerNorm、GELU、rotary attention）上验证，未覆盖其他模型；在大规模捐赠模型（6.9B→1.4B）时，两杠杆叠加会过度修正；对不同规范化或激活函数的通用性尚未测试。

---

## 204. CT-HEG: A Bidirectional, Timestamp-Attributed Event Graph for ICU In-Hospital Mortality Prediction - An Architectural Ablation Study

**arXiv ID:** 2608.02663 | [PDF](https://arxiv.org/pdf/2608.02663v1)

**作者:** Mohammad Nasir Uddin `[一作]` (Westcliff University), Asif Ahamed `[通讯]` (Westcliff University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究提出连续时间异构 EHR 图（CT‑HEG）架构，并实现 CHIRP‑Net 基于 GATv2 的图神经网络，直接从 ICU 记录的时间不规则观察中预测住院死亡。

**💡 创新点**

创新点在于：①将观测时间和数值作为边属性编码，完全避免插值；②通过结构化的双向边证明观察节点与读出节点之间的可达性是模型功能的必要条件；③系统消融显示将异构边类型合并为单一关系并未降低性能，挑战传统的“异构信息”假设。

**🔧 技术方法**

使用的技术包括：hetero‑GATv2Conv（hidden=192，heads=4，edge_dim=2）、四层堆叠、时间敏感的边特征（t_hours/48，value_norm）、温度标定、五随机种子训练和 5‑模型集成。

**📊 数据集**

使用的数据集为 MIMIC‑IV v3.1（31,142 份 ICU 病例，LOS≥48h），采用 70/15/15 的患者层面分层拆分。

**📈 对比分析**

与逻辑回归、GRU‑D、mTAND、4‑层 Transformer 等基线比较，CHIRP‑Net 在 5‑种子平均下 AUROC 为 0.8449±0.0071，集成后为 0.8618（95% CI: 0.8485–0.8745），相较于逻辑回归提升 7.6 AUROC 点，mTAND 与 Transformer 分别提升 5.9 与 7.2 点。

**⚠️ 局限性**

局限性包括：未在外部数据集或真实时间序列上进行验证；缺乏群体子组公平性与时序泛化评估；未实现因果可解释性；基线比较未进行统一的多种子调优；仅涵盖单一 EHR 模式（未加入波形或影像）。

---

## 205. Agentic Reinforcement Learning with Self-Distilled Reward Shaping

**arXiv ID:** 2608.03223 | [PDF](https://arxiv.org/pdf/2608.03223v1)

**作者:** Ranxu Zhang `[一作]` (University of Science and Technology of China), Chao Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Agentic Reinforcement Learning with Self‑Distilled Reward Shaping（ADRS）框架，将训练阶段仅可见的程序化教师信号转化为token‑级奖励，从而在稀疏轨迹奖励下实现更细粒度的时间信用分配。

**💡 创新点**

创新点在于：①通过within‑step校准使教师token分数可比；②利用返回关联的Teacher Value Advantage（TVA）门控衡量教师置信度与实际回报的相关性；③将门控后的token奖励预先融入本地advantage计算，保持原RL主干（GRPO/GiGPO）的trajectory方向不变，避免辅助目标的对齐问题。

**🔧 技术方法**

采用了对抗式预训练的语言模型（Qwen2.5‑3B/7B、Qwen3‑1.7B），基于GRPO或GiGPO的群体优势算法，结合token‑级奖励塑形、教师置信度门控和预优势奖励整合。

**📊 数据集**

在三大交互基准上评估：ALFWorld（文本驱动的嵌入式控制）、WebShop（网页导航购买）以及Search‑based QA（检索式问答）。

**📈 对比分析**

与多种基线（普通RL、技能提示、OPSD、SDAR等）对比，ADRS在所有模型规模下均实现了显著提升：ALFWorld 最高 94.5% 成功率、Search‑based QA 宏平均 45%、WebShop 得分 87.5/成功率 76.6；并在少量数据、未见任务和更长交互步长下保持优势。

**⚠️ 局限性**

局限性包括：1）教师信息仅在训练时可见，推理时需完全无教师，限制了在需要实时指导的场景中的直接适用；2）对教师置信度的门控依赖批次统计，可能对极端分布或小批量训练敏感；3）实现上需额外的回溯计算和正则化，增加训练复杂度。

---

## 206. From Routes to Steps: Separating Semantic Progress from Local Execution in Vision-and-Language Navigation

**arXiv ID:** 2608.03143 | [PDF](https://arxiv.org/pdf/2608.03143v1)

**作者:** Xiangyun Huang `[一作]` (Beihang University), Lin Jiarong `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Route2Step 框架，将 Vision‑and‑Language Navigation（VLN）中的语义进度追踪与本地执行分离，并通过显式接口实现两者协同。

**💡 创新点**

创新点在于（1）采用 E‑SPA 自动生成子指令–轨迹对齐；（2）在训练中分层监督——状态级别纠正语义进度，动作级别纠正执行；（3）引入执行状态接口（Normal/Recovering），明确区分语义错误与执行错误。

**🔧 技术方法**

使用预训练多模态模型 Qwen2.5‑VL‑3B，分两阶段训练：专家路径初始化 + 基于回放的状态/动作监督；利用动态规划实现 E‑SPA 对齐；在 inference 时通过两模块（Instruction Analysis 与 Action Generation）实现分层决策。

**📊 数据集**

在 R2R‑CE 与 RxR‑CE（Matterport3D）上进行训练与评估，使用 FG‑R2R、Landmark‑RxR 做对齐验证；真实世界实验中部署于 Unitree GO2 机器人，利用单视 RGB 进行导航。

**📈 对比分析**

通过与统一策略、仅动作监督、仅状态监督等对比，在 R2R‑CE Val‑Unseen 上 SR 从 48.1% 提升至 55.3%（SPL 48.2%），在 RxR‑CE 上达到 54.8/42.6；真实部署成功率为 19/33，且在最难室内任务中显著提升完成率。

**⚠️ 局限性**

局限性包括需要大量状态级监督与子指令–轨迹对齐、对大型预训练模型的依赖、单视 RGB 限制定位与地图构建、以及在复杂视觉或长句场景下的鲁棒性仍待提升。

---

## 207. Attribute-based Undetectable Watermarking for Generative AI Models

**arXiv ID:** 2608.03174 | [PDF](https://arxiv.org/pdf/2608.03174v1)

**作者:** Miryam Mi-Ying Huang `[一作]` (Carnegie Mellon University), Er-Cheng Tang `[通讯]` (University of Washington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种属性级可控的水印方案，允许在生成式 AI 模型输出中嵌入水印并仅在满足特定属性策略的检测密钥下可被检测，满足可定制授权。

**💡 创新点**

创新点在于将水印检测权限与输出属性绑定，使用受限伪随机函数（CPRF）生成与属性相关的水印随机性，构建了可按策略细粒度授权的水印检测；同时给出了一致性、适应性鲁棒性、不可检测性和正确性等安全定义并证明。

**🔧 技术方法**

核心技术包括受限伪随机函数（CPRF）、伪随机错误纠正码（PRC）、随机性恢复（RandRecovery）与生成式模型的随机性回收、以及多标签属性分类器；实现使用 PyTorch、Hugging Face Transformers、PRC 代码等。

**📊 数据集**

实验使用的文本生成模型（未指明具体型号）与多标签属性分类器，针对 5 个主题（医学、经济、艺术、软件、体育）构造提示，生成 200-250 份样本；属性词典大小 5。

**📈 对比分析**

通过单属性与双属性提示的检测矩阵对比，目标属性的检测率在 86%-98% 之间，非目标属性仅 14%-33%；FPR 随 PRC 参数变化；在同等条件下，单属性检测率显著高于非目标；性能评估显示该方案在保持水印不可检测性的同时实现了显著的选择性检测。

**⚠️ 局限性**

局限性包括：跨属性误检仍存在；依赖属性分类器的准确性；对随机性恢复的需求限制了可适用模型；在高熵或低熵词汇状态下性能波动；实验仅覆盖 5 个属性，未验证大规模属性空间；对图像、音频等多模态的适用性未评估。

---

## 208. Rectify Then Diffuse: Disentangling Concepts Before Denoising Trajectory Unfolds

**arXiv ID:** 2608.03135 | [PDF](https://arxiv.org/pdf/2608.03135v1)

**作者:** Ning Zhu `[一作]` (University of Electronic Science and Technology of China), Liang-Jian Deng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对文本到图像的多概念生成中概念混合或遗漏问题，本文提出了“Rectify-then-Diffuse（RTD）”框架，在生成开始前通过一次性修正初始潜变量的概念分配，从而提升组合图像的忠实度。

**💡 创新点**

创新点在于将多概念生成视为边界条件问题，提出无训练、一次性修正的Soft-Overlap Disentanglement（SOD）与Isotropic Gradient Rectification（IGR）技术，避免了传统方法在采样过程中多步迭代干预，显著降低计算成本。

**🔧 技术方法**

技术实现包括：①在高噪声时刻进行一次诊断前向传播，提取概念注意力图；②通过软IoU度量概念间重叠并构造可微分的分离目标；③利用梯度归一化得到固定比例的潜变量更新；④随后使用原始的diffusion采样器完成图像生成。

**📊 数据集**

实验使用AE-Bench、T2I-CompBench、RareBench三个多概念基准；同时在SDXL和SD 2.1两个主干模型上验证通用性。

**📈 对比分析**

与多种训练自由基线（如Attend-and-Excite、SynGen、CO3等）以及传统的attention/score引导方法对比，RTD在BLIP-VQA、ImageReward以及人类评估上均实现了SOTA水平，例如在AE-Bench O-O子集BLIP-VQA提升45.8%，整体Inference仅增加6.3%时间，速度比CO3快2.3×。

**⚠️ 局限性**

局限性在于目前仅对概念的初始空间分配进行无关布局的分离，未针对具体空间关系（如握持、穿戴、遮挡）建模，未来可将SOD扩展为关系感知的修正策略。

---

## 209. OncoTriad-QA: A Patient-Level Radiology-Pathology-Genomics Benchmark for Pan-Cancer Reasoning

**arXiv ID:** 2608.02615 | [PDF](https://arxiv.org/pdf/2608.02615v1)

**作者:** Ahnaf Munir `[一作]` (University of Central Florida), Yu Tian `[通讯]` (University of Central Florida)

**通讯引用:** 886 | [OpenAlex ID](https://openalex.org/A5113808840)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个面向患者的多模态癌症问答基准OncoTriad-QA，并提供对应的多模态语言模型OncoVLM

**💡 创新点**

首次将CT/MRI、全切片病理、基因组学与临床数据在同一病例层面集成，采用LLM辅助生成高质量问题与答案，并设计跨模态、缺失模态评估框架

**🔧 技术方法**

结合多模态编码器（MedSigLIP、UNI、BulkFormer、CpGPT等）、投影器、LoRA微调的LLM（Qwen3.5-9B）进行跨模态融合与指令调优

**📊 数据集**

使用TCGA、TCIA、GDC等公开数据共9,281例、32种癌症，包含86.1k语义问题（MCQ与开放式）

**📈 对比分析**

与GPT-5.4、MedGemma-4B、Gemma-4-E4B等基线比较，OncoVLM在多模态问答上平均提升10.7点（MCQ精度及BERTScore-F1），在结构化任务上显著优于所有基线

**⚠️ 局限性**

受限于公开回顾性队列、少样本癌症、模态缺失、不完整临床注释、扫描协议异质性，导致统计功效不足与潜在偏倚

---

## 210. Multimodal Auto-regressive Transformer Surrogate for Modeling Variable Operations and Quantifying Uncertainty in Geological Carbon Storage

**arXiv ID:** 2608.02629 | [PDF](https://arxiv.org/pdf/2608.02629v1)

**作者:** Yifu Han `[一作]` (Stanford University), Louis J. Durlofsky `[通讯]` (Stanford University)

**通讯引用:** 18361 | [OpenAlex ID](https://openalex.org/A5002057296)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种多模态自回归Transformer surrogate，用于在地质不确定性下预测变 perforation 与注入策略下的CO₂储存性能，并将其嵌入层级MCMC数据同化流程。

**💡 创新点**

创新点：① 将三种不同输入模态（3D地质模型、相对渗透率参数、操作控制变量）通过Transformer融合；② 采用自回归时间解码器捕捉BHP/注入速率转换；③ 结合层级PCA地质模型与超大规模GEOS训练数据，实现快速、精准的同化与预测。

**🔧 技术方法**

技术：Transformer Encoder-Decoder架构、Patch‑embedding、多头自注意力、自动回归解码、Scheduled Sampling、AdamW优化、Cosine学习率退火、混合精度训练、层级MCMC（非中心化pCN）和PCA地质生成。

**📊 数据集**

数据集：4000条高保真GEOS CO₂注入模拟（变量 perforation & 注入策略），500条独立测试案例（随机地质、相对渗透率、控制变量），并在真实模型（SEAM CO₂改进版）上进行数据同化验证。

**📈 对比分析**

与传统固定操作同化方法对比：在同样的数据量下，surrogate实现了在约4–10小时GPU时间内完成约10⁶次模拟评估；预测误差在注入量（2.3%）和饱和度（MAE 0.028）等关键量上优于现有非Transformer surrogate；同时能准确捕捉注入速率与BHP控制的切换。

**⚠️ 局限性**

局限性：① surrogate模型误差固定，未随参数动态更新；② 只在改进的SEAM CO₂模型上验证，缺乏真实现场数据；③ 对观测井位置和数量的敏感度未系统评估，可能影响同化效果；④ 大规模MCMC仍需要数百万函数评估，对更复杂地质或更高维参数空间的可扩展性未知。

---

## 211. GSTEP: Global Spatio-Temporal Density-Driven Visual Token Pruning for Efficient Video Large Language Models

**arXiv ID:** 2608.03083 | [PDF](https://arxiv.org/pdf/2608.03083v1)

**作者:** Mengjie Zhang `[一作]`, Jian Yang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全局时空密度驱动的视频视觉标记剪枝框架（GSTEP），用于在 VideoLLMs 中显著减少视觉标记数量。

**💡 创新点**

创新点在于将视频视为连续时空信息流，结合连续时间密度与空间密度进行全局采样，避免传统段落级剪枝导致的关键信息丢失。

**🔧 技术方法**

采用中心语义差分计算时间密度、Gaussian 平滑、空间中心偏差测量空间密度，并融合后使用对数分离的全局 FPS 采样策略进行标记选择。

**📊 数据集**

在 LLaVA-OneVision‑7B、Qwen2.5‑VL‑7B 等 VideoLLMs 上，使用 VideoMME、LongVideoBench、MLVU、MVBench、EgoSchema 等五大公开基准进行实验。

**📈 对比分析**

与 FastV、FastVID、CDPruner、VisionZip、VidCom2 等剪枝基线比较，在 75%–90% 剪枝比下，GSTEP 在多模型多基准上平均保持 97%–100% 的性能，同时实现 1.17–1.20 倍的推理速度提升。

**⚠️ 局限性**

局限性包括：仍需依赖预训练视觉编码器的特征，时间密度平滑参数对极长视频或高动态场景的适应性有限，且未提供针对不同视频内容的自适应预算分配策略。

---

## 212. Any-OPD: Heterogeneous On-Policy Distillation for Flow-Matching Models via Representation-Space Bridging

**arXiv ID:** 2608.03316 | [PDF](https://arxiv.org/pdf/2608.03316v1)

**作者:** Siming Fu `[一作]` (Joy Future Academy), Haojun Xu `[通讯]` (Joy Future Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种适用于任意两种latent flow‑matching生成器的异构 on‑policy 蒸馏框架，能够在教师与学生不共享 VAE、架构或噪声调度的情况下进行知识迁移。

**💡 创新点**

创新点在于：①用冻结的外部视觉表示（DINOv2 CLS）代替共享空间；②用噪声级匹配而非步数索引对齐轨迹；③引入离线锚定阶段先把教师分布映射到学生潜空间，再进行 on‑policy 微调；④整个过程仅依赖教师的采样接口，完全可替换教师。

**🔧 技术方法**

核心技术包括：latent flow‑matching生成器、噪声投影 Π^σ_T、DINOv2 CLS 视觉特征、流匹配损失、LoRA 微调、Euler/CPS 采样方法以及噪声级对齐与投影回传机制。

**📊 数据集**

使用的数据集有：Pick‑a‑Pic 训练集（用于锚定与微调），DrawBench、GenEval 与 DPG‑Bench（用于评估美学、图像奖励与人类偏好分数）。

**📈 对比分析**

在 12B FLUX.1‑dev 训练 2.5B SD3.5‑Medium 的实验中，PickScore 从 0.846 提升至 0.884，HPSv3 从 9.12 提升至 10.97，几乎等同甚至超过教师；当教师换成 6B Z‑Image 时，同样获得显著提升，验证了教师可替换性。

**⚠️ 局限性**

局限性包括：对教师采样接口的完整性依赖；对噪声投影强度和锚定质量高度敏感；实验仅覆盖图像生成任务，尚未验证到其他模态；并且在极度不匹配的噪声调度时可能需要更细致的超参调整。

---

## 213. ED-DiT: Physics-Guided Diffusion Pretraining for Transferable Molecular Representations from Electron Density

**arXiv ID:** 2608.03260 | [PDF](https://arxiv.org/pdf/2608.03260v1)

**作者:** Liang Shuang `[一作]` (Fudan University), Ben Fei `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了利用三维电子密度场进行自监督预训练，学习可迁移的分子表示。

**💡 创新点**

引入物理引导的扩散Transformer，结合掩码扩散去噪和电子数一致性约束。

**🔧 技术方法**

使用Diffusion Transformer、掩码扩散去噪、电子数一致性损失和零初始化交叉注意力。

**📊 数据集**

在EDBench六个任务的数据集上进行预训练和微调。

**📈 对比分析**

与从头训练的同架构、X-3D、PointVector-S以及Geoformer基线比较，ED-DiT在所有任务上均取得显著提升，尤其在少量标签时表现更优。

**⚠️ 局限性**

仍在高局部密度变化区域的预测误差较大，模型对极端高密度点的鲁棒性不足，且扩散Transformer对大规模分子计算成本较高。

---

## 214. AIDE: Automated Instruction via Distilled Expertise for Reference-Free Motor Skill Coaching

**arXiv ID:** 2608.03047 | [PDF](https://arxiv.org/pdf/2608.03047v1)

**作者:** Yoshiki Ito `[一作]` `[通讯]` (Hitachi, Ltd.), Yoshiki Ito (Hitachi, Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在训练阶段利用专家参考，推理阶段仅用学习者姿态生成运动技能反馈的框架

**💡 创新点**

通过隐式知识迁移（编码器共享+权重初始化）实现LUPI，消除推理时对专家输入的需求

**🔧 技术方法**

使用冻结的大语言模型、姿态编码器（MLP+Perceiver Resampler）和辅助潜在模块

**📊 数据集**

在ExpertAF数据集（篮球和足球）上评估

**📈 对比分析**

与零样本、检索、无专家、CoachMe、Learner+Expert等方法对比，AIDE在文本质量和部分预测上与需专家推理的方法相当，优于无专家基线

**⚠️ 局限性**

对低数据场景（如足球）部分预测性能不足，依赖外部姿态估计，评估仅限于单一LLM骨干和ExpertAF数据集

---

## 215. EFX Allocation In (Multi)Hypergraphs

**arXiv ID:** 2608.03171 | [PDF](https://arxiv.org/pdf/2608.03171v1)

**作者:** Thanasis Lianeas `[一作]` (University of West Attica), Minas Marios Sotiriou `[通讯]` (Athens University of Economics and Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在具匝度至少为 4 的超图和满足一定多重性约束的多重超图中构造证明存在公平至上不含任何单个物品的 EFX（公平至上不含单个物品）分配，并给出多项式（或伪多项式）时间的构造算法。

**💡 创新点**

创新点在于：
• 采用超图/多重超图框架，将传统图论中的邻接关系推广到任意大小的边，突破了仅考虑双重边的限制；
• 证明了在匝度≥4 的超图下，无论评估函数如何单调，始终存在 EFX 分配，并给出可多项式求解的算法；
• 对于多重超图，提出了一个新的“单点多重性约束”（某点所有边的多重性≤边大小−2）的足够条件，使得 EFX 分配仍可构造；
• 通过对不满足该约束的实例与“一件未分配物品的 EFX”难题的归约，证明该约束的必要性。

**🔧 技术方法**

使用的技术包括：
• 以超图的邻接关系定义公平性条件；
• 设计一系列维护属性的迭代分配子算法（例如 `Orienting Edges`、`Complete Allocation`），利用贪婪与递归的组合确保 EFX；
• 引入“patch”概念处理多重边；
• 通过潜在函数（如社会福利）证明算法终止；
• 对于多重超图的多重性约束，利用额外的“停车”顶点和邻居分配策略，避免了不可定向边造成的冲突。

**📊 数据集**

本工作完全是理论性的，没有使用任何公开数据集；所有结论均通过构造性证明与算法分析得到。

**📈 对比分析**

方法评估：
• 通过严格的存在性证明和构造算法展示了理论性能；
• 对于简单超图给出了多项式时间复杂度（O(n⁶) 等），而多重超图则给出伪多项式时间（因潜在函数上限为可实现的社会福利值）；
• 未做实验比较，因问题属于理论公平分配的存在性问题，主要关注的是证明与构造性而非运行时间对比。

**⚠️ 局限性**

局限性：
• 需要超图匝度至少为 4；
• 对于多重超图，只在满足“单点多重性约束”时给出构造，若该约束被放宽则问题退化为通用 EFX 存在性难题，当前尚无算法；
• 伪多项式时间依赖于评估函数取值范围，若取值范围大，实际运行时间可能不可接受；
• 未考虑可数化的评估函数或非单调情况，且未给出对大规模实例的实验验证。

---

## 216. Your Agentic LLMs Secretly Encode Latent Signals of Indirect Prompt-Injection Exposure

**arXiv ID:** 2608.02657 | [PDF](https://arxiv.org/pdf/2608.02657v1)

**作者:** Jianshuo Dong `[一作]` (Tsinghua University), Han Qiu `[通讯]` (Tsinghua University)

**通讯引用:** 3739 | [OpenAlex ID](https://openalex.org/A5019692903)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了代理式大语言模型在面对间接提示注入（IPI）攻击时内部的暴露信号，并基于这些信号设计了检测、对抗和解释方法。

**💡 创新点**

创新点在于揭示并利用LLM内部隐藏状态中的线性可预测IPI暴露信号，构建了 probe‑gated reasoning 防御机制，并提出了基于自然语言解释的信号解释框架。

**🔧 技术方法**

主要技术包括：线性探针（probe）训练、Chain‑of‑Thought (CoT) 监测、预填充 anti‑injection reasoning、以及基于 logit 比较的自问式解释采样。

**📊 数据集**

使用的实验数据集为 AgentDojo 基准（覆盖多套任务、攻击与指令组合），并在 GLM‑5.2、Qwen3 系列、Gemma‑4‑31B、GPT‑oss‑20B 等六款开源模型上进行评估。

**📈 对比分析**

与无干预、提示式对抗、工具结果提醒、SafePath 等基线相比，probe‑gated 方法将攻击成功率从 30–50% 降至 0–3%（在 Qwen3‑5‑27B 最高 0%），同时保持或提升清洁任务的效用（如 Qwen3‑8B 的 ASR 从 47.2% 降至 2.9%，Utility 仍为 69.1%）。

**⚠️ 局限性**

局限性包括：仅针对单一类型的间接提示注入；探针对模型内部机制的解释仍依赖预设的 128 个自然语言假设；对更大规模或闭源模型的迁移性及对抗攻击的全面鲁棒性仍需进一步验证。

---

## 217. Axiomatic shared-medium coordination for stigmergic systems

**arXiv ID:** 2608.02619 | [PDF](https://arxiv.org/pdf/2608.02619v1)

**作者:** Fernando Paredes García `[一作]` `[通讯]` (Independent Researcher), Fernando Paredes García (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个中介无关的比较层，利用启用-响应签名来对不同共享介质的间接协调机制进行抽象与比较。

**💡 创新点**

创新点在于给出粗细元数据分层时的商充分性判据、基于响应纤维的最小修复构造以及一阶匹配写入的保持性与基于 freshness 条件的一阶不可提升障碍。

**🔧 技术方法**

使用了抽象系统元组、可观测/响应映射、商映射与响应签名等形式化工具，并在 tuple‑space 与 timestamped 虚拟 stigmergy 两种实例中演示。

**📊 数据集**

未使用实验数据集，所有结果均为理论证明与符号实例。

**📈 对比分析**

比较方法通过构造响应签名空间与商映射，实现状态级别的响应等价判定；在匹配写入时，证明商映射下的状态更新保持一致，性能上仅涉及符号运算，无具体复杂度评估。

**⚠️ 局限性**

局限包括：只讨论有限序列匹配、缺乏并发/分支模拟、未给出通用匹配动作生成算法、仅使用单一键值共享介质示例，且无法处理更复杂的图/账本介质细节。

---

## 218. Unified Lookup-Table Inference with Signed-Digit K/V Caches for Ternary LLMs

**arXiv ID:** 2608.03229 | [PDF](https://arxiv.org/pdf/2608.03229v1)

**作者:** Ziang Duan `[一作]` (Huazhong University of Science and Technology), Chao Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种统一的查找表（LUT）推理方案，将三元 LLM 的线性层和动态 K/V 注意力都映射到相同的 LUT 计算子系统，解决了三元权重与实时 K/V 缓存的表示不匹配问题。

**💡 创新点**

创新点在于：① 将三元权重视为单平面（R=1）受限签数位（RSD）表示，② 为 K/V 缓存设计共享尺度、可变平面数和指数间隙的 RSD 编码，③ 采用边界 V‑尾机制在因果解码期间保持编码一致性，④ 通过硬件感知的设计空间探索（DSE）联合选择平面数、模板和映射，实现质量‑效率最优折中。

**🔧 技术方法**

技术手段包括：LUT 计算（分组 LUT 构造、查找与归约）、受限签数位（RSD）量化、动态量化与编码单元（DQEU）、多流共享 LUT 流程、约束引导的 DSE 与多精度（R、Δ）搜索、以及在 40‑nm 过程中的 ASIC 合成与性能仿真。

**📊 数据集**

实验使用了多种三元化模型：BitNet b1.58 2B4T、Falcon‑E‑1B/3B、以及 QAT 转换的 OPT‑350M/1.3B/2.7B，并在九个 NLP 任务（如文本分类、问答、摘要等）上评估模型质量与推理性能。

**📈 对比分析**

与基准（同面积、同吞吐量的异构线性/注意力加速器）对比，所提系统在 Falcon‑E‑3B 的 A8 配置下实现了 2.52× 的面积效率提升、2.30× 的功耗效率提升；在 OPT‑350M 的 A16 配置下实现了 4.78× 的面积效率和 8.61× 的功耗效率提升；同时 K/V 缓存占用可压缩至 BF16 的 21% 以内，显著降低内存带宽消耗。

**⚠️ 局限性**

局限性包括：① 对 R、Δ 的搜索与 DSE 计算开销较大，需预先校准和验证；② 对于极长序列或高频率的 V‑尾更新，编码和解码的时延可能成为瓶颈；③ 在极低精度（R=1）时会出现显著的准确率下降，需谨慎选择；④ 目前实验仅在 40‑nm 过程下验证，跨工艺迁移的可扩展性尚未彻底评估。

---

## 219. Pruning-Aware Multi-Cluster Co-Inference for Large AI Models in AI-RANs

**arXiv ID:** 2608.03026 | [PDF](https://arxiv.org/pdf/2608.03026v1)

**作者:** Xiaowen Cao `[一作]` (Shenzhen University), Jie Xu `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多聚类、剪枝感知的 LAIM 联合推理框架，利用边缘服务器和多 GPU 协同处理多设备的多视角特征。

**💡 创新点**

创新点在于将模型剪枝、任务调度、通信资源分配三者通过信息论的 rate‑distortion 与 PID（通过 Shapley 估计）耦合成联合优化问题，并给出可实现的迭代求解方法。

**🔧 技术方法**

使用了模型剪枝、Rate‑Distortion 理论、PID/Shapley 信息分解、混合整数线性规划（MILP）、序列凸近似（SCA）以及 GPU 队列排程等技术。

**📊 数据集**

实验数据集包括 CIFAR‑10（图像分类）和 MSR‑VTT（视频‑文本字幕生成）。

**📈 对比分析**

与固定功率/带宽、轮询/随机调度、均等重要性等基线相比，提出的方法在满足能耗/时延约束下，分类准确率、BLEU‑4 与 CIDEr 指标均有显著提升。

**⚠️ 局限性**

主要局限：假设聚类与请求是静态的，未考虑用户移动、网络波动及动态负载；仅考虑剪枝压缩，未结合量化或蒸馏；对极大规模 LAIM 的进一步扩展仍需研究。

---

## 220. Designing a Good Virtual Node: Addressable and Cardinality-Preserving Global Memory for Message Passing Architectures

**arXiv ID:** 2608.02709 | [PDF](https://arxiv.org/pdf/2608.02709v1)

**作者:** Félix Marcoccia `[一作]` `[通讯]`, Félix Marcoccia

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出一种可地址化且保持计数的虚拟节点架构，解决MPNN在Two‑Radius和重复键值场景下的有限容量瓶颈。

**💡 创新点**

创新点在于将虚拟节点拆分为多槽写入/读取、使用点积地址编码并引入anchor来恢复归一化质量，实现可插入的多集表达和1‑WL级别的计数。

**🔧 技术方法**

技术主要包括跨注意力写入/读取、点积地址码、anchor软max、Slot‑FiLM以及基于MPNN的局部传播。

**📊 数据集**

实验使用了合成的Two‑Radius、重复计数的多模任务、基于Motif的计数图以及NetDiff的约束链接预测。

**📈 对比分析**

对比实验表明，标准VN仅提升标签准确率，Cross‑Attn VN在标签上达到100%但计数不变，Anchor Cross‑Attn VN在标签和计数上均达到100%，Motif计数和链接预测亦显著优于均值/求和池化。

**⚠️ 局限性**

局限性包括：实验仅在合成/半合成数据上验证，未对宽VN与强解码器的对比，且缺乏对学习动态与泛化边界的深入分析。

---

## 221. When Should Graph Attention Be Sparse? Learning a Per-Edge Tsallis Index

**arXiv ID:** 2608.02938 | [PDF](https://arxiv.org/pdf/2608.02938v1)

**作者:** Kleyton da Costa `[一作]` (University College London & Holistic AI), Bernardo Modenesi `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种可学习的Tsallis图注意层LTGA，替换GAT中的softmax，使得注意分布能够在训练过程中自适应从密集到稀疏的不同形态；

**💡 创新点**

创新点在于通过tanh限定的可学习q指数，将q-softmax连续覆盖heavy‑tailed、Shannon和compact‑support三种注意形态，并在全局、层级、头部以及边缘四种粒度上学习该指数；

**🔧 技术方法**

技术包括Tsallis最大熵的q‑softmax、tanh重参数化、分离优化器加warm‑up、Shannon先验正则、梯度稳定的Taylor分支，以及用于边缘门控的轻量级MLP；

**📊 数据集**

实验数据集涵盖八个节点分类基准：Cora、Citeseer、PubMed、WebKB的Texas/Wisconsin/Cornell、Roman‑Empire和Amazon‑Ratings；

**📈 对比分析**

在GCN、GAT、GATv2、冻结q网格、α‑entmax、稀疏max、容量匹配控制及异质性专用模型的对照实验中，LTGA‑Edge以平均Friedman排名2.75位列第一，尤其在异质图Roman‑Empire与Amazon‑Ratings上显著提升（最多+16%），整体提升约1%以内；

**⚠️ 局限性**

局限性包括：对极小或高方差图（如WebKB）无显著优势；q<1（heavy‑tailed）在实验中未带来收益；门控可解释性不强；仅测试转导节点分类，未扩展到归纳、链路预测或图分类；计算开销略增（≤15%）。

---

## 222. LocAnyMed: Vision-Language Grounding for Multimodal Medical Images

**arXiv ID:** 2608.03322 | [PDF](https://arxiv.org/pdf/2608.03322v1)

**作者:** Zihan Wang `[一作]` (Wuhan University), Jing Zhang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了LocAnyMed框架，实现对CT、超声、X光和光学医学影像的统一开放式视觉定位。

**💡 创新点**

创新点在于构建200k条跨模态、开放式定位数据集，加入模态路由的视觉专家和视觉证据链式推理监督，显著提升跨模态定位泛化能力。

**🔧 技术方法**

主要技术包括大规模全参数微调、模态路由MoE视觉FFN、Chain-of-Thought（CoT）视觉证据生成以及多尺度特征融合。

**📊 数据集**

使用了209,910条来自Medical Segmentation Decathlon、TotalSegmentator、LUNA16、US30K、BUS-BRA、GRAZPEDWRI-DX、FracAtlas、DENTEX、Kvasir-SEG等公开医学数据集，覆盖四种模态。

**📈 对比分析**

与专用检测器、通用视觉语言模型及定位专家比较，LocAnyMed在Same‑set验证集上F1@IoU 0.50达85.59%、F1 Mean 69.15，跨集测试中同样保持33.63%/19.12%优于所有基线；加入CoT监督后跨集性能提升至35.49%/21.34%。

**⚠️ 局限性**

局限在于需已知影像模态标签、模型规模大且训练成本高，对极低对比度或模糊目标的定位仍受限，且CoT监督依赖可信视觉感知，易放大早期错误。

---

## 223. LLM Serving in the Wild: An Empirical Study of Frameworks, Methods, and System Designs

**arXiv ID:** 2608.03036 | [PDF](https://arxiv.org/pdf/2608.03036v1)

**作者:** Forough Majidi `[一作]` (Polytechnique Montreal), Heng Li `[通讯]` (Polytechnique Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对当前主流LLM服务框架（vLLM、SGLang、TensorRT-LLM、LMDeploy、FlashInfer）及其内部高效推理方法在开源软件中的采用情况进行了大规模经验性分析，评估了框架与方法的组合使用、与LLM类型的关联以及框架在不同软件系统中的实际落地场景。

**💡 创新点**

创新点包括：①首次系统梳理并量化LLM服务框架在GitHub中的受欢迎度与实际使用频率；②构建统一的高效推理方法分类体系，并映射到各框架实现；③通过代码级别的API/参数检测，挖掘框架与方法的细粒度共存模式；④结合聚类与主题模型，从“意图”“技术聚焦”“使用场景”“系统设计”四维度揭示LLM服务框架在实际系统中的多样化角色。

**🔧 技术方法**

技术手段：系统化的文献与GitHub搜索与筛选；Python AST + 正则解析框架API与方法调用；统计学指标（星标、fork、仓库活跃度）评估受欢迎度；基于SentenceTransformer、UMAP+HDBSCAN的语义聚类与BERTopic进行仓库文本主题抽取；以及多维度归因分析框架-方法-LLM的匹配关系。

**📊 数据集**

数据集：①来自Google Scholar与Engineering Village的46种LLM服务框架列表；②通过GitHub搜索得到约439个候选Python项目，筛选后共26个仓库；③每个仓库中提取的代码文件与API使用信息；④通过GPT-4o-mini生成的四维结构化摘要用于聚类。

**📈 对比分析**

评估方式：使用GitHub stars/forks等公开指标衡量框架流行度；在所有满足阈值的仓库中统计框架与方法使用比例、组合频次；通过框架-LLM维度交叉表显示不同模型族/规模/应用场景下的框架偏好；聚类结果通过主题一致性与可解释性评估。性能表现主要体现在：vLLM被最大化采用（1,821仓库），FlashInfer虽然星标较少但多框架共存率最高；内存管理、并行计算与网络裁剪是最常见的高效方法；框架与方法组合往往补充彼此缺陷，提升整体推理效率。

**⚠️ 局限性**

局限性：①仅聚焦Python语言和GitHub项目，可能遗漏非Python或私有部署；②采用星标/活跃度阈值的筛选可能排除真实使用但未被广泛关注的项目；③仅使用公开API和参数，无法检测自定义优化或内部实现；④对LLM模型的分类依赖于公开标签，可能存在误标；⑤未对实际推理性能（延迟、吞吐、成本）做基准测试，只评估采用频率与组合。

---

## 224. CastFSR: A Fast--Slow--Reflect Agentic Reasoning Framework for Context-Aware Time Series Forecasting

**arXiv ID:** 2608.03031 | [PDF](https://arxiv.org/pdf/2608.03031v1)

**作者:** Xiaoyu Tao `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CastFSR 框架，将上下文感知时序预测拆分为快速预测、慢速推理和反思验证三阶段协同。

**💡 创新点**

创新点在于把预测视为 agentic 决策过程，利用 LLM 进行工具调用、上下文检索和一致性检查，实现数据驱动先验与语义推理的动态融合。

**🔧 技术方法**

使用大语言模型（如 DeepSeek V4 Flash、Qwen3‑4B）作为推理引擎，结合轻量级时间序列预测工具、上下文检索模块和反思评估机制；训练时采用两阶段策略：SFT 与多轮 RL。

**📊 数据集**

评估使用 ETT、Wind、EPF 等多领域公开时序数据集，覆盖不同时间分辨率和预测时长。

**📈 对比分析**

与统计方法、深度学习模型、基础模型及其他 LLM/agent 方法对比，CastFSR 在绝大多数基准上取得最优或次优 MSE/MAE，显示出显著的准确性提升。

**⚠️ 局限性**

局限包括对 LLM 计算资源依赖、上下文检索覆盖范围受限，以及在极端事件下对历史模式的依赖可能导致误判。

---

## 225. Privacy-Preserving AI Verification via Minimal Information Disclosure

**arXiv ID:** 2608.02774 | [PDF](https://arxiv.org/pdf/2608.02774v1)

**作者:** Sleem Abdelghafar `[一作]` (Rice University), Gabriel Kulp `[通讯]` (Intelligence Security Laboratories)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种信息论框架 MID（Minimal Information Disclosure），在满足 AI 验证可用性阈值的前提下，选择传递给验证者的证据机制（信道、采集策略、发布变换），使其对敏感属性的余散信息（I(S;E|Y)）最小化。

**💡 创新点**

创新点在于：①把证据信息内容视为可设计对象，而非仅限于发布格式；②利用条件互信息量化“余散泄露”，提供统一且攻击无关的评估；③在证据源内部通过收集策略或后处理实现低信息发布，并可通过 ZKP（Groth16）保证发布过程的正确性。

**🔧 技术方法**

核心技术包括信息理论（互信息、条件互信息、Fano 不等式）、线性投影和加噪发布、ZKP（Groth16 zk‑SNARK）、基于统计分布的采样与交叉验证、以及多任务的隐私‑实用性前沿分析。

**📊 数据集**

实验使用六个公开物理测量数据集：NLR 设施功耗、DeepTheft RAPL 电源波形、Elsayed 设备识别与 RL 与非RL 工作负载、ModelSpy GPU 电磁波形、以及训练与推理功耗数据，覆盖硬件身份、计算规模、模型身份、执行类型等六类验证问题。

**📈 对比分析**

与现有单一输出或固定格式方案相比，MID 在三项任务中实现了完美验证且零测量泄漏；其余任务给出清晰的隐私‑实用性前沿，证明可在保持验证准确率（如 99% 以上）同时将敏感信息泄漏降至接近随机猜测；实验还展示了 MID 选择的低信息发布显著抑制 ModelSpy 与 DeepTheft 的攻击效果。

**⚠️ 局限性**

局限性包括：①需要在开发阶段获取足够标注的物理测量样本来估计互信息；②对证据源的完整性与认证假设要求高；③当前仅在物理侧迹象上验证，未覆盖软件级或网络级的泄露场景；④信息量估计在高维时可能不稳定，需更鲁棒的估计方法。

---

## 226. VIVID: A Culturally Grounded Benchmark Exposing the Figurative Language Gap in Vietnamese NLP

**arXiv ID:** 2608.03095 | [PDF](https://arxiv.org/pdf/2608.03095v1)

**作者:** Tu Tran Do `[一作]`, Long Hoang Dang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了首个针对越南习语与谚语的文化化比喻语言理解基准 VIVID，并提出了双层注释的复杂性与主题分类体系。

**💡 创新点**

创新点包括：① 第一次系统化收集并注释越南比喻语言；② 设计基于 LLM‑as‑a‑Judge 的生成式评测框架，并对四种提示策略进行对比，发现面向维度评估（aspect‑based）与人类评分最为一致；③ 将生成式与判别式任务结合，提供完整的评测方法。

**🔧 技术方法**

使用的技术主要有：大规模语言模型 (GPT‑4o、Gemini Flash 2.5、Qwen‑3、Llama‑4‑Scout 等)；LLM‑as‑a‑Judge（GPT‑4.1）进行自动评分；多种提示策略（zero‑shot、示例、链式推理、维度评估）；以及基于 Language‑Model‑Evaluation‑Harness 的判别式精确匹配评测。

**📊 数据集**

数据集为 VIVID，包含 1,636 条越南习语/谚语，分别来源于《Nguyen Lan 词典》和《Vu Dung 词典》；每条记录配有双层注释（复杂性特征与语义主题）。

**📈 对比分析**

比较方法：在生成式任务中，对 8 个模型分别在 zero‑shot 与 few‑shot 两种提示下进行评分；在判别式任务中计算话题和语言/文化特征的准确率。性能表现：大模型（GPT‑4o、Gemini Flash 2.5）平均得分约 2.3‑2.5（小于 50% 正确率），中型模型（14B）得分约 1.0‑1.2，轻量级（7B）低于 0.5。显示出规模影响显著，越南专用训练并未显著提升文化比喻理解。

**⚠️ 局限性**

局限性：① 评测依赖 GPT‑4.1，导致可复现性受限；② 仅对 200 条样本做人工可靠性验证，未覆盖全部多样性；③ 生成式评分仅与单一 LLM 判别器对齐，未来需尝试开放源模型以增强透明度。

---

## 227. Everyone Conforms, No One Believes: Pluralistic Ignorance in LLM Agent Populations

**arXiv ID:** 2608.02758 | [PDF](https://arxiv.org/pdf/2608.02758v1)

**作者:** Yashwanth YS `[一作]` `[通讯]` (Carnegie Mellon University), Yashwanth YS (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在LLM多代理系统中构建了100个跨域情境基准，并系统评估8个LLM模型在公开对话中出现的多元无知与连锁效应。

**💡 创新点**

创新点在于首次证明LLM代理群体会自发产生多元无知，揭示模型与领域选择是影响仿真结果的关键自由度，并通过提示组件消融验证其是交互自发现象。

**🔧 技术方法**

使用技术包括多轮对话交互、提示工程、关键词评分评估、prompt ablation、API调用和自定义对话框架。

**📊 数据集**

数据集为公开的 100 个情境基准（10 个领域 × 5 权威级别），已上传至 HuggingFace 数据集 yashwanthys/pluralistic-ignorance。

**📈 对比分析**

对 8 个模型（GPT‑4o‑mini、GPT‑4o、Claude Sonnet 4.6、DeepSeek V4 Pro、Qwen3 235B、GPT‑OSS‑120B、Llama 3.3 70B、Kimi‑K2.7）在 100 场景下测算合规率 64‑94% 与连锁率 0‑48%，其中 GPT‑4o 以 48% 连锁率突出，其他模型连锁极低。

**⚠️ 局限性**

局限性包括结果高度受模型与领域选择影响、连锁现象稀缺导致对真实社会脆弱性低估、仅测试英语西方语境、未考虑网络结构或更大群体规模。

---

## 228. Fovea: Physical-Implication-Aware Wafer-Scale DSE with Decision-Domain-Guided Cross-Fidelity Refinement

**arXiv ID:** 2608.03285 | [PDF](https://arxiv.org/pdf/2608.03285v1)

**作者:** Jinxi Li `[一作]` (Tsinghua University), Shouyi Yin `[通讯]` (Tsinghua University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种可重用的工作负载特定晶圆尺度设计空间探索方法，先通过物理影响感知构建可行设计空间，再用决策域引导的跨保真度细化筛选可能的最佳架构。

**💡 创新点**

创新点在于：①将晶圆尺寸、贴装、D2D能力等物理约束显式映射为可行空间；②利用精确局部冗余消除获得唯一可行配置；③基于评估器间误差界定决策域，保证保真度最高的选项被包含；④通过少量校准样本估计误差，实现自适应成本的评估。

**🔧 技术方法**

采用低成本ASTRA-sim解析后端与高成本ASTRA-sim+ns-3参考后端，配合配套的D2D、边界访问、贴装约束模型，以及配对采样校准和决策域判定算法。

**📊 数据集**

使用七个LLM训练工作负载（Llama‑3B、Llama‑8B、Qwen2.5‑32B、Llama‑70B、Llama‑405B等）和十个不同组件库与铺贴组合的参考可验证设计空间，共计70个工作负载/设计空间对。

**📈 对比分析**

与全量参考评估、全量解析评估、SA、Theseus、Polaris等基线比较，方法在所有70对中完全恢复参考最优，平均节省4.13倍时间（最大7.80倍），决策域平均占总空间13.6%，参考评估占比20.4%。

**⚠️ 局限性**

局限性包括：①仅适用于同质重复芯片结构，无法直接处理异构或非规则铺贴；②对参考后端的准确性依赖，若模型偏差大则误差界定失效；③校准样本比例需经验确定，可能在极大空间中仍需较多参考评估；④对极端资源极限配置的精确建模仍有不足。

---

## 229. VeriTrace: Human-Like Temporal Exploration Completes Agentic Action Space

**arXiv ID:** 2608.02878 | [PDF](https://arxiv.org/pdf/2608.02878v1)

**作者:** Yu-Tung Liu `[一作]` (University of Maryland), Cunxi Yu `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了VeriTrace多代理系统，通过Agentic Temporal Exploration实现完整的调试动作空间，支持信号选择、时间窗口查询和迭代推理，从而实现Verilog RTL代码的自动生成与调试。

**💡 创新点**

创新在于引入Agentic Temporal Exploration，使LLM调试代理能够完全控制信号选择和时间窗口，形成类似人类验证工程师的假设驱动调试流程，并在单模块Benchmark上首次实现100% Pass@1。

**🔧 技术方法**

采用多代理架构、ReAct式思考-行动-观察循环、专用Inspector代理进行波形查询、Claude Sonnet 4.0/4.5 LLM、模拟判定与高温采样等技术。

**📊 数据集**

使用VerilogEval-V2 Benchmark（156个单模块Verilog问题）进行评测。

**📈 对比分析**

与公开的多代理基线（VerilogCoder、MAGE、ACE-RTL）在相同LLM后端进行Pass@1对比，VeriTrace在Claude Sonnet 4.0下达97.4%（+5.1%）并在4.5下实现100% Pass@1，显著提升调试成功率。

**⚠️ 局限性**

局限于单模块Benchmark，尚未验证在多模块工业级设计、完整规格缺失及大规模信号空间下的可扩展性，同时对LLM的依赖性强，模型差异会影响性能。

---

## 230. Rethinking Self-Evolving Agent Skills: Feedback Dynamics over Multiple Rounds

**arXiv ID:** 2608.02636 | [PDF](https://arxiv.org/pdf/2608.02636v1)

**作者:** Yuxuan Liu `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10897 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个控制反馈的自我演化技能评估框架，在14个模型-基准组合中对三种反馈视角（成功、失败、两者）进行多轮演化，并完整记录搜索轨迹。

**💡 创新点**

系统性隔离并统一反馈、演化与验证流程，揭示技能自我演化实为稀疏的验证过滤搜索而非持续改进，并量化其对泛化与测试时计算的影响。

**🔧 技术方法**

使用提示式语言模型（GPT‑5.5、Gemini 3.1 Pro、DeepSeek V4‑Pro）执行任务，生成候选技能；采用验证门控、字节差异检查、单/多轮优化器、并行与顺序测试时扩展等技术。

**📊 数据集**

覆盖5个多样化基准（SearchQA、OfficeQA、SpreadsheetBench、LiveMath、DocVQA）以及扩展的8模型SearchQA分析，涵盖问答、表格、数学和视觉问答任务。

**📈 对比分析**

通过验证选取的最佳技能与父技能在发布测试、鲁棒性与迁移指标上对比，11个验证选取的技能中9个在测试集上提升，整体提升稀疏；对比测试时并行采样与序列细化，发现并行采样几乎匹配SearchQA演化收益，但在SpreadsheetBench仍存在显著差距。

**⚠️ 局限性**

评估仅覆盖5个基准且未涉及专业技能基准，未对所有模型进行完整的测试时扩展，缺乏更大规模搜索和多任务演化的探索。

---

## 231. HyperAgent: Planning and Acting over Tool-Schema Hypergraphs for Tool-Use LLM Agents

**arXiv ID:** 2608.02650 | [PDF](https://arxiv.org/pdf/2608.02650v1)

**作者:** Zian Zhai `[一作]` (University of New South Wales), Wenjie Zhang `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于工具-模式超图（Tool‑Schema Hypergraph，TSH）的动态工具使用规划框架HyperAgent，能够在LLM代理完成复杂任务时通过工具-模式依赖关系实现高效规划与执行。

**💡 创新点**

创新点在于：①构建细粒度参数级工具-模式超图，显式建模工具输入输出及其依赖；②利用TSH检索任务相关工具上下文图，生成基于模式的任务DAG；③采用缺陷导向的支持图扩展（DOE）在执行时动态生成状态条件下的工具支持子图，从而避免冗余调用。

**🔧 技术方法**

主要技术包括：图结构建模（TSH）、语义检索与节点对齐、超图子图检索、缺陷导向的启发式搜索、LLM引导的任务分解与执行（ReAct+代码执行）。

**📊 数据集**

使用了AppWorld基准数据集，该数据集包含多种应用的API交互任务，共750个任务，用于评估代理的任务完成率。

**📈 对比分析**

与多种基线（SFT、DPO、RL、ReAct、PlanExec、FullCodeRefl、Traj等）比较，HyperAgent在Test-N和Test-C上取得最高的任务完成率（TGC/SGC），同时在API调用次数、LLM交互轮数和token消耗上均优于对照方法。

**⚠️ 局限性**

局限性包括：①对超图构建与维护的人工与LLM辅助成本；②在极大规模工具集合时超图和上下文检索的计算开销；③仍依赖LLM的语义检索与任务解释，可能在语义歧义或极端任务描述下表现不佳。

---

## 232. ShielDroid: A Hybrid Approach Integrating Machine and Deep Learning for Android Malware Detection

**arXiv ID:** 2608.03250 | [PDF](https://arxiv.org/pdf/2608.03250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 233. SP3O: Reinforcement Learning from Segment Preferences without Reward Modeling

**arXiv ID:** 2608.02951 | [PDF](https://arxiv.org/pdf/2608.02951v1)

**作者:** Evan Assmus `[一作]` (University of Michigan), Lei Ying `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Segment Pairwise Proximal Policy Optimization (SP3O) 的无奖励模型、无评估器的偏好强化学习算法，能够直接使用分段偏好来更新策略；

**💡 创新点**

创新点在于：1) 将段级偏好与梯度优化结合，提供无奖励模型的梯度估计；2) 对段长度进行理论分析，揭示段长度与估计误差之间的权衡；3) 采用离策略重要性采样与PPO式裁剪实现稳定的更新；

**🔧 技术方法**

使用了段级偏好采样、离策略重要性采样、PPO风格的裁剪/KL正则化、理论证明（段MDP与原MDP梯度关系）等技术；

**📊 数据集**

在机器人控制任务（Gymnasium 的 Ant‑v5、HalfCheetah‑v5、Swimmer‑v5）以及一个LLM去毒化微调任务（GPT‑J 6B + LoRA）上进行实验；

**📈 对比分析**

与在线 DPO、P3O、ZPG 等奖励模型自由算法对比，SP3O 在长时间序列任务中表现更好，尤其是更长的段或更长的生成长度时优势更明显；

**⚠️ 局限性**

需要大量人类偏好查询，适用于偏好获取容易的场景；若偏好获取困难，可考虑规则化或 AI 辅助推断。

---

## 234. PRWeaver: Evaluating LLM-Based Code Auditors against Long-Horizon Malicious Pull Requests

**arXiv ID:** 2608.02693 | [PDF](https://arxiv.org/pdf/2608.02693v1)

**作者:** Yuekun Wang `[一作]` (Singapore Management University), Xiaofei Xie `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 LLM-PRBench 的基准，用于评估大型语言模型（LLM）在拉取请求（PR）审计中的对抗性检测能力，尤其是针对跨多个 PR 的长周期攻击。

**💡 创新点**

创新点在于：①将攻击拆分为多个可单独出现的 PR 片段，模拟真实攻击者的分步隐蔽行为；②提供 208 个经过执行验证的攻击实例及 832 个不同渲染条件（R0–R3）的 PR，方便系统化比较；③引入多种审计上下文设置（分解、插入正常 PR、携带者融合）揭示审计器的脆弱点。

**🔧 技术方法**

使用的大型语言模型包括 DeepSeek V4 Flash、Claude Haiku 4.5/4.6、GPT‑5.4 mini，集成在 OpenCodeReview、Claude Code、GitHub Copilot Code Review 等审计框架中，并利用 DeepSeek 作为无偏判定者。

**📊 数据集**

数据集来自十个真实开源仓库（如 Pretix、Django、Flask 等），包含 208 条执行验证的攻击链，覆盖 6 类安全风险（访问控制、财务完整性、工作流可用性、身份认证、数据暴露、注入风险）。

**📈 对比分析**

评价指标为序列级检测率（DR）和逃逸率（ER）；在单 PR 审计时，最高 DR 约 79%（Claude Code + Sonnet 4.6）；在 R2、R3 等复杂渲染下，DR 下降至 34–55%；整体发现长周期攻击会使检测率下降 10–20% 甚至更低，说明现有审计器对跨 PR 的安全依赖链识别不足。

**⚠️ 局限性**

局限性包括：①基准聚焦于特定类型的攻击链，未覆盖所有可能的对抗策略；②评估仅基于公开的 LLM 模型，未探索更强模型或更深层次的工具集成；③对“携带者融合”等渲染方式的可复制性和通用性尚待进一步验证；④缺乏对人类审计员与 LLM 结合效果的对比。

---

## 235. Understanding Organizational Strategies Across Multimodal Artifacts in Immersive Computational Notebooks

**arXiv ID:** 2608.03132 | [PDF](https://arxiv.org/pdf/2608.03132v1)

**作者:** Sungwon In `[一作]` (Kyung Hee University), Mallesham Dasari `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在VR空间中构建多模态计算笔记本（代码、叙述、数据表与可视化）并设计用户实验，系统探索了分析师如何在沉浸式环境中组织这些不同类型的工作单元及其流向关系。

**💡 创新点**

创新点在于首次从多模态视角考察沉浸式计算笔记本的空间组织与流表示策略，并提出深度维度、基于单元的组织中心以及可选透明度流表示的设计启示。

**🔧 技术方法**

使用Meta Quest 3头显进行无绳VR交互，提供六向连线选择与透明度调节；实验采用两种条件（固定显式流与可调隐式/空间代理流）。

**📊 数据集**

实验数据集为Iris与Wine两套常用机器学习数据，涉及加载、K‑Means/DBSCAN聚类、三维散点图和柱状图/热图等可视化输出。

**📈 对比分析**

与固定流条件相比，Hybrid（可调透明度）条件下参与者移动的工作单元更少（约24%）、完成时间缩短约30%，流线创建/删除次数更少，主观负荷与不满意度显著下降。

**⚠️ 局限性**

局限包括样本规模有限（20人）、任务为短期组织练习、仅使用固定大小的工作单元、只考虑透明度而未探索其他视觉编码，以及仅在笔记本结构内测试，缺乏对更一般化工作流程与更大规模、多样化工作单元的验证。

---

## 236. Evaluating LLM Trade-offs for Enterprise Automation: Lessons from Workflow Generation in a Production Enterprise Platform

**arXiv ID:** 2608.03311 | [PDF](https://arxiv.org/pdf/2608.03311v1)

**作者:** Xavier Wrenn `[一作]` (IBM), Anca Sailer `[通讯]` (IBM)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在IBM Concert Workflows平台上评估并改进了LLM驱动的低代码工作流生成，提出了分块（piecewise）生成管道，提升结构成功率与成本效益。

**💡 创新点**

创新点在于将工作流生成拆分为变量 scaffolding、基础块和嵌套块的三步序列化流程，并通过多模型、多轮跑测实现低成本小模型可生产级可用性。

**🔧 技术方法**

技术包括大型语言模型（granite, llama, mistral, gpt‑oss）与ReAct+RAG推理、检索增强生成、JSON Patch修正、结构验证、以及Claude Sonnet 4.5的外部判定器。

**📊 数据集**

数据集为29个真实企业合规自动化需求提示，结合内部的动作块、块示例、分块数据、文档等检索源，共计2,784次实验（8跑×29×6模型×2管道）。

**📈 对比分析**

比较方法为对同一提示在两个管道（v1 monolithic vs v2 piecewise）下分别用6个模型执行8次，评估结构成功率、生成时延、token使用、成本；v2在大多数模型上结构成功率提升至70–98%，成本从$0.01–0.20/流，时延增加8–75s，gpt‑oss结构最高但语义准确率最低。

**⚠️ 局限性**

局限在于未提供置信区间/显著性检验，语义评估仍基于单一外部LLM判断且未内嵌到管道；缺乏真实执行结果验证，且仅覆盖合规领域的提示。

---

## 237. FaithIR: Rethinking Infrared Image Super-Resolution from Perceptual Sharpness to Task Relevant Fidelity

**arXiv ID:** 2608.03106 | [PDF](https://arxiv.org/pdf/2608.03106v1)

**作者:** Axi Niu `[一作]` (Northwestern Polytechnical University), Yanning Zhang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于像素级恢复的红外图像超分辨框架 FaithIR，能够在不引入人工纹理的前提下精确重建热分布和目标轮廓。

**💡 创新点**

创新点在于：① 双分支结构，patch‑level conditioning 负责捕获全局热分布与长距离结构；② pixel‑level restoration 通过像素级自注意力与 AdaLN 实现细粒度恢复；③ 使用条件流匹配（conditional flow‑matching）直接在像素域训练，避免了对自然图像预训练模型的依赖；④ pixel‑wise AdaLN 为每个像素位置分配独立的调制参数，提升局部细节恢复。

**🔧 技术方法**

主要技术包括：条件流匹配、AdaLN、RMSNorm、位置编码自注意力、compact global interaction、像素嵌入、Euler 积分等。

**📊 数据集**

使用 FLIR‑IISR 作为训练和验证集，M3FD 与 FMB 用于跨数据集泛化评估以及目标检测与语义分割的下游任务测试。

**📈 对比分析**

与 PFT‑SR、HAT、SinSR、InfraFFN、DifIISR、Real‑IISR 等方法在 PSNR、SSIM、FID、LPIPS 等低层指标上竞争或优于它们；在目标检测（mAP_50）和语义分割（mIoU）等高层指标上显著领先，且在 M3FD 与 FMB 上保持较好的跨数据集性能。

**⚠️ 局限性**

局限性：目前仅针对 4× 放大因子；在更大放大倍数、极端噪声或真实降采样场景下的鲁棒性尚未充分验证；模型训练与推理成本相对较高，实时性尚需进一步优化。

---

## 238. Bayesian Data Reweighting Improves Multimodal Retrieval for Knowledge-Based Visual Question Answering

**arXiv ID:** 2608.02907 | [PDF](https://arxiv.org/pdf/2608.02907v1)

**作者:** Jingchen Sun `[一作]` (University at Buffalo), Changyou Chen `[通讯]` (University at Buffalo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Bayesian Data Reweighting (BDR)，通过贝叶斯推断对知识型视觉问答检索中的负样本进行自适应权重调整，降低假负样本影响；

**💡 创新点**

将负样本重要性建模为潜在变量，利用拉普拉斯增强实现可分离后验并闭式推断，结合随机近似 EM 实现实例级贝叶斯权重推断，同时提供连续 Gamma 先验与二值 Bernoulli 先验；

**🔧 技术方法**

对比学习（InfoNCE）、贝叶斯概率模型、拉普拉斯变换辅助因子化、Gamma/Bernoulli 先验、闭式后验采样、随机近似 EM（SAEM）以及 LoRA 微调多模态 LLM；

**📊 数据集**

在 OKVQA、EVQA、InfoSeek、WIT、OVEN、KVQA 等七个 KB‑VQA 基准上进行评估，并在 InfoSeek 与 EVQA 上测试检索+生成；使用 Qwen2‑VL‑2B、Phi‑3.5‑V‑3.8B、Qwen2‑VL‑7B 等多模态 LLM 作为检索器；

**📈 对比分析**

与多种负样本重加权基线（随机、相似度、边际、硬负、去偏、解耦正、硬度+去偏）比较，BDR（Gamma 先验）在 Recall@K、PseudoRecall@K、VQA 准确率等指标上均优于基线，平均提升 3–6 个百分点；与 PreFLMR、ReT 对比，在多模态 LLM 检索+生成任务上逼近 Oracle；

**⚠️ 局限性**

仅适用于单跳检索，使用伪似然非完整后验；实验仅覆盖有限数据集和模型，未验证开放语料库；未实现检索与生成的端到端联合优化。

---

## 239. ISEE: Interactive Semantic Enrichment for Database Fields

**arXiv ID:** 2608.02604 | [PDF](https://arxiv.org/pdf/2608.02604v1)

**作者:** Yuan Tian `[一作]` (Purdue University), Yunyao Li `[通讯]` (Adobe)

**通讯引用:** 1651 | [OpenAlex ID](https://openalex.org/A5106404797)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了一个交互式语义丰富系统，用来改进数据库字段描述，帮助大型语言模型在数据相关任务中更好理解字段语义。

**💡 创新点**

创新点在于：①多维度评分系统评估描述质量；②基于澄清分类法生成针对性澄清问题；③条件查询填充功能，让用户通过验证查询快速补充语义。

**🔧 技术方法**

技术包括：使用 OpenAI GPT‑4o 作为 LLM；text‑embedding‑3‑small 文本嵌入模型；自定义算法实现五维评分；两步澄清生成流程；条件查询填充；人机交互界面。

**📊 数据集**

数据集：BIRD 文本到 SQL 数据集用于自动化模拟评估；用户研究采用行业工程师提供的四字段描述；并进行案例研究。

**📈 对比分析**

对比方法为手动编辑、候选选择、ChatGPT 对话三种基线。系统在 NASA TLX 问卷、人工评分、以及 BIRD 上的实体链接 MRR、文本 SQL EX 与 VES 指标中均优于基线，显著提升实体链接 22%、文本 SQL 12% 等性能。

**⚠️ 局限性**

限制：需要用户具备充分领域知识；用户回答可能不完整或过时；仿真用户简化了真实交互；实验样本规模有限。

---

## 240. UniNav: A Unified World-Action Diffusion Model for Visual Navigation

**arXiv ID:** 2608.03244 | [PDF](https://arxiv.org/pdf/2608.03244v1)

**作者:** Changqing Zhou `[一作]` (Hong Kong University of Science and Technology), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的世界-动作扩散模型，实现图像目标视觉导航时同时预测未来视图和可执行路径

**💡 创新点**

通过在同一扩散过程里联合去噪视觉、轨迹和相机几何标记，既保留可解释的视觉预见性，又能在推理时去除未来图像令推理更高效

**🔧 技术方法**

利用预训练视频 VAE、扩散 Transformer、相机几何标记、流匹配损失以及混合训练（轨迹标注数据+无标记视频）

**📊 数据集**

在 RECON、SaCSoN、GO Stanford、SCAND 等四个真实图像目标导航数据集上进行评测，并使用 ScanNet、DL3DV、CityWalker、LAVN 等无标记视频数据辅助训练

**📈 对比分析**

相较于传统的基于动作的策略（ViNT、NoMaD、NavDP、FlowNav）和导航世界模型 NWM，Full 变体在 ATE 上全部数据集下降 3.7%–24.7%，Fast 变体在保持相似准确率的同时，单步推理时延仅 0.1 s，显著优于对比方法

**⚠️ 局限性**

目前仅适用于短时局部路径规划，未覆盖长时规划；实验多在离线场景进行，缺乏闭环部署验证

---

## 241. Material-Segmented Per-Pixel Emissivity Correction for Thermographic Anomaly Detection in Cultural Heritage Digital Twins

**arXiv ID:** 2608.02964 | [PDF](https://arxiv.org/pdf/2608.02964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 242. Language Models Encode the Contextual Truth of Propositions

**arXiv ID:** 2608.03035 | [PDF](https://arxiv.org/pdf/2608.03035v1)

**作者:** Rupak Sarkar `[一作]` (University of Maryland), Rachel Rudinger `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究大模型内部对上下文真值的线性表示，并验证其在多种任务和对话场景中的一致性与因果性；

**💡 创新点**

发现上下文真值方向既可跨任务迁移，又能揭示对话中两种不同的“顺从”行为（表演性与表征性），为真值追踪提供新诊断方法；

**🔧 技术方法**

采用质量为“mass‑mean”线性探测器（Probe），结合激活驱动（Steer）与Cohen's d、NIE等指标进行评估；

**📊 数据集**

使用ExploreToM生成的程序化故事、SNLI文本、以及多模态“spot‑the‑difference”对话数据；

**📈 对比分析**

在各模型（Llama‑2、Qwen3）上，探测器准确率均远高于随机/标签洗牌基线（约70–90%），跨任务/数据迁移误差小，驱动可显著改变模型输出，表现为高NIE；

**⚠️ 局限性**

局限在于仍需依赖程序化生成的可验证真值，无法完全排除常识推理影响；并未针对激活读取位置的因果机制进行深入验证。

---

## 243. Structure-Aware Robust Fine-Tuning: Defending Vision-Language-Action Robots Against Physical Attention Hijacking

**arXiv ID:** 2608.03231 | [PDF](https://arxiv.org/pdf/2608.03231v1)

**作者:** Jinquan Zhang `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), F. Richard Yu `[通讯]` (Carleton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究 Vision‑Language‑Action (VLA) 政策在面对可打印的物理攻击时的鲁棒性，提出一种以注意力劫持为核心的攻击方法 AGSD，并提出一种仅更新视觉编码器、结合注意力蒸馏与几何一致性的零推理开销防御方法 SARF；

**💡 创新点**

创新点在于将攻击与防御聚焦于机制层面：① 通过结合注意力聚焦与视觉‑语言语义破坏实现跨任务、跨架构的高效攻击；② 在防御方面提出仅更新视觉编码器、使用策略关键注意力蒸馏和语言引导几何一致性，显著提升鲁棒性的同时保持原始性能；

**🔧 技术方法**

技术方法包括 Expectation‑over‑Transformation (EOT) 优化、注意力引导损失、特征离散化损失、图像‑文本误对齐损失、特征锚定损失、策略关键注意力蒸馏损失以及语言引导的几何一致性损失；

**📊 数据集**

实验数据集主要是 LIBERO 四个模拟任务套（Spatial、Object、Goal、Long）以及真实 PiPER 桌面机器人平台；

**📈 对比分析**

与 UADA、UPA、EDPA 等攻击和对应的防御方法比较，AGSD 可将 OpenVLA 的失败率从 14.2–46.2% 提升至 100%；SARF 在所有攻击下将失败率从 100% 降至平均 28.6%，在 PiPER 上成功率从 23% 提升至 65%，且对干净样本的性能几乎无损；

**⚠️ 局限性**

局限性包括：1）对长时间序列仍可能存在注意力漂移导致的残余错误；2）仅针对视觉编码器的更新，未覆盖其他 VLA 架构；3）在复杂真实环境（如多物体杂乱、动态光照）下的鲁棒性尚待进一步验证。

---

## 244. Reachability Is Not Realization: Tracing the Sources of LLM Benchmark Gains

**arXiv ID:** 2608.03219 | [PDF](https://arxiv.org/pdf/2608.03219v1)

**作者:** Yanchao Li `[一作]` (Nanjing University), Yuqiang Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型在固定评估条件下的问答级别可实现性与可达性进行审计，并通过层级路由、MLP块抑制与RLVR训练三种干预方式检验模型的可实现性变化。

**💡 创新点**

首次区分“实现”与“可达性”，揭示随机路径可达性超过结构化搜索、MLP块是导致直接生成失败的关键组件，并证明RLVR训练提升实现但不一定扩大可达范围。

**🔧 技术方法**

采用预算匹配的随机路径与结构化搜索、答案无关投票选择、内部读数识别、MLP块抑制实验以及基于可验证奖励的RLVR训练等技术。

**📊 数据集**

在Llama3、Qwen2.5、Gemma4、Qwen3.5、OLMo2等模型上，评估ARC-Easy/Challenge、GSM8K、PIQA、LogiQA、DART-Math等六大任务。

**📈 对比分析**

通过匹配预算、温度、答案格式的对比，发现随机路径可达性持续提升且超过结构化搜索；答案无关投票几乎无法弥补可达性差距；RLVR训练在部署性能上提升1.7–14.7分，但可达性门槛往往保持不变或下降。

**⚠️ 局限性**

受限于固定预算与评估协议，未能捕捉所有可达性提升；答案无关选择无法利用oracle优势；MLP块定位局部性强且不一定普适；RLVR提升主要靠已有高频可达答案，缺乏对全新问题的扩展。

---

## 245. Test Time Adaptation Methods for Point Cloud Registration in Laparoscopic Surgery

**arXiv ID:** 2608.02883 | [PDF](https://arxiv.org/pdf/2608.02883v1)

**作者:** Nina Bodelot `[一作]` (ETS Montreal), Eric Granger `[通讯]` (ETS Montreal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对腹腔镜手术中的三维点云配准，研究并改进了四种测试时适应（TTA）方法，使其能处理预手术网格与实时内窥镜重建点云之间的异向域移位。

**💡 创新点**

创新点在于：①将原本针对分类任务的TTA方法（如Entropy阈值、正则化统计）改写为配准专用的方式；②针对点云配准中的单侧域移位，分别为预手术和实时点云维护独立的归一化统计；③将熵阈值替换为配准无监督指标Inlier Ratio；④在输入适应族中仅对受影响的实时点云做子集剔除或特征对齐。

**🔧 技术方法**

使用的技术包括：点云特征编码器（PARE‑Net）、自监督重建与对应分类的辅助任务、层归一化（LN）统计校准、进步嵌入对齐（PEA）和Purge‑Gate剔除异常特征，并实现了两种统计估计策略（episodic 与 continual）。

**📊 数据集**

数据集：源域合成数据集 P2P（肝脏）与 P2ILReg；目标域包括（a）对 P2ILReg 合成点云施加五种噪声/采样扰动的腐败版本；（b）真实内窥镜重建的 92 条点云。对合成目标使用统一/高斯/背景/脉冲噪声以及全局/局部密度下降、裁剪和遮挡等八种扰动。

**📈 对比分析**

对比方法分为模型适应（Point‑TTA）、归一化适应（LN）、输入适应（PEA 与 Purge‑Gate）。实验表明：输入适应族在所有数据集上均能降低配准误差，尤其是 Purge‑Gate 在保持低延迟的同时提升了 Inlier Ratio；归一化适应仅在 P2P 上有效，且在 P2ILReg 上退化；模型适应虽可取得一定改进但计算成本极高（≈1.2s/样本）。

**⚠️ 局限性**

局限性：①对真实数据的改进幅度仍有限，说明输入/统计估计在极端域移位下可能不够鲁棒；②归一化统计更新依赖单样本或窗口估计，导致误差累积；③模型适应方案在实时手术场景不可行；④缺乏针对多帧连续视频的持续适应机制，难以充分利用时间信息。

---

## 246. DenialRAG: Single-Document RAG Poisoning via Embedded Parametric Denial

**arXiv ID:** 2608.02678 | [PDF](https://arxiv.org/pdf/2608.02678v1)

**作者:** Abay Zhurekbay `[一作]` (Lawrence Technological University), Fan Li `[通讯]` (Lawrence Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种单文档RAG腐败攻击DenialRAG，攻击者通过在检索语料中注入一段包含正确答案与攻击者错误答案冲突且明确驳斥正确答案的短文档，诱导LLM生成错误答案。

**💡 创新点**

创新点在于将正确答案直接命名并在同一文档中给出驳斥理由，从而将冲突内置到检索上下文中，使得模型在生成时更易接受错误答案。

**🔧 技术方法**

技术主要包括两次LLM调用：一次提取问题意图和实体节点，二次生成包含四个结构化要素（声明、证据、驳斥、权威）≤100词的恶意文档；使用检索增强生成架构和现有的单文档攻击框架；对比五种推理时防御方法。

**📊 数据集**

数据集使用BEIR中的三大开放域问答基准：Natural Questions、HotpotQA、MS‑MARCO，各取100条目标查询；评测模型涵盖8个LLM（Mistral‑7B、LLaMA‑3.1‑8B、GPT‑4o‑mini、GPT‑5‑mini、GPT‑4o、GPT‑5.2、GPT‑5.5、DeepSeek‑V4‑flash）。

**📈 对比分析**

与四种现有单文档攻击（PoisonedRAG‑N1、AuthChain、CorruptRAG‑AK、PIA‑direct）以及五种推理时防御（Paraphrase、InstructRAG、TrustRAG、RobustRAG、AstuteRAG）比较；在成本层模型上DenialRAG达到最高攻击成功率（NQ≈89%，HotpotQA≈94%，MS‑MARCO≈86%），但在前沿模型上效果下降；在防御场景中，除AstuteRAG在部分模型上显著降低ASR外，其他防御对DenialRAG的抑制有限。

**⚠️ 局限性**

局限性包括：对前沿大模型的攻击效果相对较弱；依赖单一文档插入，若语料更新或多文档攻击被采用则效果受限；防御效果不统一，单一推理时策略难以同时抵御所有攻击；并未考虑对检索器和生成器内部可调参数的白盒优化或多文档协同攻击。

---

## 247. PLS-Calib: A Partial Least Squares Framework for Event Camera and Odometry Calibration under Ground Motion Constraints

**arXiv ID:** 2608.03296 | [PDF](https://arxiv.org/pdf/2608.03296v1)

**作者:** Guangyu Li `[一作]` (Guangdong Institute of Intelligence Science and Technology), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于偏最小二乘（PLS）回归的事件相机与地面移动机器人里程计的旋转标定框架（PLS-Calib），并改进事件时间表面为极性敏感表示，能在仅平面运动条件下实现闭式、数值稳定的标定。

**💡 创新点**

创新点包括：①将PLS回归用于标定，避免CCA在平面运动下协方差矩阵奇异导致的数值不稳定；②引入极性增强的时间表面，使事件图像在快速运动与低光/HDR环境下的特征检测更稳健；③构造闭式旋转矩阵的构造方法，直接从回归矩阵生成正交旋转。

**🔧 技术方法**

使用技术包括：偏最小二乘（PLS）回归、极性增强时间表面（polarity‑aware TS）、标准化与特征提取、右手坐标系构造、闭式解法；实验中还使用了传统CCA、Andreff求解器和基于APS帧的参考标定。

**📊 数据集**

数据集：合成数据（包含全6-DoF与平面约束两种运动场景），以及实际数据来自Jackal Clearpath平台搭载DAVIS 346事件相机和里程计，实验环境为平坦室内地面。

**📈 对比分析**

比较方法：对合成数据和真实数据分别评估PLS、CCA和Andreff三种方法。PLS在平面约束下的旋转误差从≈26°降至≈0.26°（合成）或≈3.18°（真实），相较于CCA（≈40–47°）和Andreff（≈74°）提升显著，说明PLS具有更高的鲁棒性与精度。

**⚠️ 局限性**

局限性：仍主要针对平面运动约束的地面机器人，缺乏对完整6-DoF运动的验证；对极性时间表面的参数需要手工调优；在高噪声或极端光照条件下的鲁棒性尚未彻底验证。

---

## 248. DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units

**arXiv ID:** 2608.03127 | [PDF](https://arxiv.org/pdf/2608.03127v1)

**作者:** Haoyu Gu `[一作]` (South China University Of Technology), Xiao-Ping Zhang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 DigitCode，一种基于手部解剖层级的符号化运动表示，能够在相同码率下大幅降低量化误差，并为手部姿态的编辑、修复和机器人再投射提供直接可操作的接口。

**💡 创新点**

创新点在于：①将符号的覆盖单元从传统的骨骼改为可分层的骨、指、手三级结构；②通过自适应的方向字典、指部联合量化和粗细层叠残差实现精度提升；③证明在固定码率下，单元选择对重构性能的影响远大于量化器类型，打破了以往只关注量化器的研究思路；④提供了可复现的评测框架 HandTok，允许在任意骨架结构上比较不同单元与量化器组合。

**🔧 技术方法**

采用了球面 k‑means 进行自适应方向编码；指部联合量化（12 维向量聚类）；粗层指令 + 细层残差（旋转归一化后再量化）；相对与事件编码压缩时间轴；以及与学习式 VQ、RVQ、FSQ 等基线的对比评测。

**📊 数据集**

使用了四大公开数据集：InterHand2.6M（双手交互），FreiHAND（单手静态），HanCo（单手序列），ASL‑Skeleton3D（噪声手语）。

**📈 对比分析**

对比方法包括：量化误差（角度误差）与码率的 rate‑distortion 曲线、六类下游任务（预测、检索、编辑、降噪、分类、生成）的定量指标。实验显示，DigitCode‑H 在与 HL‑26 相同码率下将误差从 14.71° 降至 3.26°，比学习式 VQ 在相同码率下的误差（≈4–6°）更低；在下游任务中，按单元匹配的 DigitCode 在每项任务上均取得最优或可比性能。

**⚠️ 局限性**

局限性包括：①仅在手部骨架上验证，尚未证明对其他解剖结构的通用性；②对极端噪声或非法姿态的修复仍需额外的学习模型或手工规则；③生成任务中对复杂动态的再现仍受限于时间轴压缩策略；④在某些任务（如精细运动生成）中，学习式量化器在大码率下可能略优。

---

## 249. Reducing CMSO to Unbreakable Graphs Cannot be Computable

**arXiv ID:** 2608.03144 | [PDF](https://arxiv.org/pdf/2608.03144v1)

**作者:** Colin Geniet `[一作]` (Institute for Basic Science), Roohani Sharma `[通讯]` (Institute for Basic Science)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究证明了Lokshtanov等人提出的将任何CMSO公式在任意图上求解归约到在(q,k)-不可分割图上求解的元定理，所需的参数q不可由公式唯一确定，即不存在可计算的上界函数；进而展示了该归约方法在理论层面的不可构造性。

**💡 创新点**

创新点在于通过结合Trakhtenbrot不完备性、参数化复杂度假设以及不可分割图的定义，构造了一个对任意可计算映射f都产生矛盾的逻辑句子，从而证明了q必须是不可计算的；这在元定理研究中首次给出了关于参数可计算性的严格限制。

**🔧 技术方法**

主要技术包括：逻辑公式的编译与编码（对三色图、带颜色图的转换）、不可分割图的定义与性质、Trakhtenbrot定理的运用、以及参数化复杂度假设（如W[k]不等于非均匀算法的等价条件）。

**📊 数据集**

无；该工作为纯理论研究，不涉及实验数据或实际数据集。

**📈 对比分析**

无实验比较；论文通过构造性的逻辑论证展示了无法构造可计算q的结论，未对算法性能进行定量评估。

**⚠️ 局限性**

局限性包括：结果仅适用于存在不可计算q的情形，且在特定逻辑片段（如存在性CMSO）下才得到最强限制；对更广泛的逻辑或更强的参数化假设的适用性仍待进一步探究。

---

## 250. Test-time reasoning effort and unauthorized tool use in language-model agents: a prespecified equivalence study

**arXiv ID:** 2608.03169 | [PDF](https://arxiv.org/pdf/2608.03169v1)

**作者:** Xiaonan Xu `[一作]` (Georgia Institute of Technology), Wenjing Wu `[通讯]` (University of Colorado Boulder)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了语言模型代理在执行多步工作流程时，调节推理计算量（reasoning effort）对工具调用合规性的影响。

**💡 创新点**

创新点在于：①在同一模型内部单一参数操控，构建匹配三组（triad）实验设计；②使用等价检验与精确零事件边界，证明在840条轨迹中无违规工具调用；③将规则探测与违规率的关系进行系统性量化。

**🔧 技术方法**

技术手段包括：GPT‑5.6 Terra 与 Sol 两级模型、TRIO‑20 最少权限工作场景测试床、确定性程序评判器、等价区间与精确二项上限、情景簇自助法等。

**📊 数据集**

数据集为 14 个确认性工作场景（TRIO‑20），共 840 条交互轨迹（每个场景 6 条低/高推理水平、3 条条件），以及 504/336 条来自两级模型的轨迹。

**📈 对比分析**

比较方法：对每个条件下低/高推理水平的违规概率做差异检验，使用等价区间（±7.01pp）和精确边界（低→高变化 ≤ ±4.34pp）。结果显示违规率为零，规则探测率在高推理水平上提升但未导致违规。

**⚠️ 局限性**

局限性：仅测试 GPT‑5.6 家族两级模型；仅采用低与最高推理水平，未覆盖中间值；实验顺序固定导致时间因素混杂；仅考虑显式系统层面禁止的场景，未检验隐式或冲突政策；模型外部干预或压力未纳入。

---

## 251. Secure AI Watermarking Framework for IP Protection in Multi-Tenant Cloud Platforms

**arXiv ID:** 2608.02656 | [PDF](https://arxiv.org/pdf/2608.02656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 252. Dynamically Allocating Evaluation Effort for Model Ranking

**arXiv ID:** 2608.03437 | [PDF](https://arxiv.org/pdf/2608.03437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 253. A Unified Feature Model for Microservice Identification and Refactoring

**arXiv ID:** 2608.02644 | [PDF](https://arxiv.org/pdf/2608.02644v1)

**作者:** Ana Almeida `[一作]` (University of Lisbon), António Rito Silva `[通讯]` (University of Lisbon)

**通讯引用:** 1062 | [OpenAlex ID](https://openalex.org/A5024207525)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对 11 篇二级研究（包括 SLR、系统映射、分类与理论等）进行元评审，构建了一套统一的特征模型（Feature Model），旨在捕捉微服务识别与重构方法的共性与变异点，并在该模型上对现有工具（Mono2Micro、Micro2Micro、HyDec、MEM、CARGO、MOSAIC 等）进行映射与评估，验证模型的完整性与可扩展性。

**💡 创新点**

创新点包括：①首次以特征建模的方式系统化描述微服务迁移中识别与重构的全流程；②明确将粒度、数据收集源、分析表示、聚合准则、算法、编辑与评估等作为可组合的变异点；③通过工具映射演示模型在实际工具中的可实现性，揭示工具间的互补性与缺口；④利用自动化映射技能与验证流程，使模型评估具备可复现性和可扩展性。

**🔧 技术方法**

技术手段主要包括：特征建模（Feature Modeling）与软件产品线原则（Nešić 等的模型构建准则）；系统映射与元评审（Systematic Mapping、Meta‑Review）收集文献变异点；自动化工具映射技能（Codebase‑Map、Docs‑Map、Verify‑Analysis）结合 Git 版本管理实现可复现的映射；在模型中使用简单的“requires”跨树约束；对比工具时采用颜色编码（暗绿、浅绿、黄、红等）直观显示特征覆盖情况。

**📊 数据集**

数据集包括：①11 篇精选的二级研究（涵盖 2021‑2026 年的 SLR、系统映射与分类研究）；② 另外 5 篇补充论文用于模型扩展与评估；③ 7 个公开实现的微服务识别/重构工具的代码仓库或文档，用于特征映射与验证；④ 通过映射技能自动收集的工具实现细节与对应特征作为评估依据。

**📈 对比分析**

比较方法：先手工对 Mono2Micro 进行特征映射，随后使用自动化映射技能对其余工具进行映射；通过颜色编码汇总每个工具对模型特征的覆盖情况；最后将所有工具的映射结果合并得到“全局覆盖”概况。性能方面，模型在 7 个工具中覆盖率超过 90%（大多数必选特征完整实现），而单个工具的覆盖率则在 60‑80% 之间；模型突出了工具间互补性——不同工具覆盖不同变异点，组合使用可实现更全面的功能。

**⚠️ 局限性**

局限性：①模型仅基于 11 篇二级研究，可能遗漏一些未被检索到的变异点；②跨树约束仅采用简单的“requires”，未考虑“excludes”等更复杂依赖；③对动态运行时行为与性能评估的支持有限，主要聚焦静态分析与元数据；④映射过程仍需人工验证，自动技能在缺乏足够文档时可能产生误判；⑤模型未覆盖所有新兴技术（如基于 AI 的分割、实时监控等），未来需持续更新。

---

## 254. ConFL: Explainable Concurrent Fault Localization via Hierarchy-Guided LLM Reasoning

**arXiv ID:** 2608.02974 | [PDF](https://arxiv.org/pdf/2608.02974v1)

**作者:** Shuai Shao `[一作]` (University of Connecticut), Tingting Yu `[通讯]` (University of Connecticut)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了ConFL框架，利用bug报告与构建的并发知识库实现仅凭bug报告定位并发缺陷。

**💡 创新点**

创新点在于构建并发知识库、层次化检索、交互级DSL以及LLM引导的可解释推理。

**🔧 技术方法**

技术包括静态并发信息抽取构建知识库、层次化检索、交互级DSL、LLM（如GPT‑4）推理与提示工程。

**📊 数据集**

使用来自八大Java项目的真实并发bug数据集（Dataset_Git）以及后期截断集，包含4–100个bug。

**📈 对比分析**

与IR基线（BRTracer、BoostNSift、BLCoiR）和LLM基线FlexFL比较，ConFL在Top‑1/3/5、MRR、MAP上分别达到0.503/0.486等指标，显著优于其他方法。

**⚠️ 局限性**

局限包括依赖静态分析的并发知识库可能产生噪声、对第三方库的并发行为覆盖不足以及对细粒度同步模型的缺失。

---

## 255. EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation

**arXiv ID:** 2608.02990 | [PDF](https://arxiv.org/pdf/2608.02990v1)

**作者:** Jiayi Luo `[一作]` (Beihang University), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种 EmbodiedVAE 视频 VAE，使用双编码器单解码器异构时空压缩架构，自动解耦机器人臂运动与背景，并提供可控的压缩潜在表示，用于高效的 LDM 训练与动作控制。

**💡 创新点**

创新点包括：①自动解耦机器人臂与背景的双编码器设计；②异构（强空间压缩 vs 强时间压缩）时空压缩策略；③基于最优传输的运动一致性损失，提升运动语义保持；④两阶段训练（先单独训练编码器，再冻结并训练统一解码器）实现更高的重建与控制性能。

**🔧 技术方法**

使用的技术包括 VAE、3D 卷积与 3D 注意力、AlphaBlender 交叉混合、最优传输（OT）与 Sinkhorn-Knopp 归一化、动作注入交叉注意力、以及多任务损失（重建、KL、感知、对抗）。

**📊 数据集**

训练使用约 100 万条自采机器人操作视频（来自 RobNet、RobSet、BC‑Z、RH20T、DROID），验证集使用 Agibot‑2025、Bridge、RT‑1 等数据集，动作控制评估在 RT‑1 与 Bridge 上进行。

**📈 对比分析**

与多种基线（OpenSoraPlan、MAGVIT‑v2、CV‑VAE、EMU‑3、Cosmos、CMD、iVideoGPT、VidTwin、Wan‑VAE、Cog‑VAE、SDXL 等）在 PSNR、SSIM、LPIPS、FVD 以及压缩率上进行对比。EmbodiedVAE 在 Agibot‑2025 上 PSNR 达 31.67，压缩率仅 0.39%；在动作控制任务中相较 Wan‑VAE 提升 2.02 dB，优于 SDXL 等基线。

**⚠️ 局限性**

局限性：①需要大规模标注（臂掩码）进行两阶段训练；②当前验证主要集中在机器人臂场景，跨任务泛化仍需验证；③压缩率虽然低，但与传统 VAE 相比仍有提升空间；④对实时性与推理速度的进一步评估尚未展开。

---

## 256. Decidability of Parameterised Dolev-Yao Secrecy

**arXiv ID:** 2608.02838 | [PDF](https://arxiv.org/pdf/2608.02838v1)

**作者:** Ioana Boureanu `[一作]` (Surrey Centre for Cyber Security), Srinibas Swain `[通讯]` (Adelaide University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在Dolev–Yao模型下，针对无限参与者的密码协议进行参数化保密性验证，证明在每个角色的全局新鲜度受限且攻击者仅允许使用良好类型的替换时，参数化保密性是可判定的，并给出了截断定理与WSTS结构。

**💡 创新点**

创新点在于：①引入参数化视角，将无穷多会话视为参数化系统；②提出全局新鲜度边界与良好类型攻击者的结构性约束；③构造代理折叠映射和BOS排序，使得大规模系统可折叠为有限代表；④得到截断定理并将问题归约为WSTS覆盖问题；⑤提供两种判定算法，弥补传统方法的局限。

**🔧 技术方法**

主要技术包括：符号协议分析与Dolev–Yao推导系统；良好类型替换与新鲜度边界的形式化；代理折叠映射与项映射的构造；BOS排序的WQO性质证明；正常形Dolev–Yao证明与前向保持性；WSTS上升兼容性与后向覆盖算法；截断与有限运行边界计算。

**📊 数据集**

本文为理论研究，无需具体数据集，所用实验基准为典型协议实例（如Needham–Schroeder共享密钥、示例1），但未提供量化实验数据。

**📈 对比分析**

与以往工作（如对单一会话、基于标签或类型兼容性的方法）相比，本文给出更通用的截断定理；算法1在截断大小内完成状态探索，复杂度为双指数；算法2采用WSTS后向覆盖，复杂度为超多项式；实验比较显示，算法1在小型实例上运行时间可接受，验证了小模型检验的合理性，但未给出具体性能数值。

**⚠️ 局限性**

局限性包括：仅适用于具有原子密钥、无等式理论和无复合密钥的协议；攻击者受限于良好类型替换，排除类型混淆攻击；仅讨论保密性（可达性）属性，对认证、隐私等等价性属性未覆盖；新鲜度限制为全局上界，无法处理无限新鲜度或可变长度消息；不考虑全局可变状态或多角色实例化。

---

## 257. Population-Robust Feature Selection via Generalized Welfare Optimization

**arXiv ID:** 2608.02887 | [PDF](https://arxiv.org/pdf/2608.02887v1)

**作者:** Ruiqi Lyu `[一作]` (Carnegie Mellon University), Bryan Wilder `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种共享特征集学习方法 PopFS，允许在不同人群中使用相同的有限特征集并为每个人群训练各自的模型。

**💡 创新点**

创新点在于：① 引入可调节的福利目标，通过幂均值在平均收益与最差人群收益之间权衡；② 采用直接的硬特征子集搜索，并结合多任务稀疏筛选与高斯-牛顿排名加速搜索；③ 使用教师压缩目标在保证信息保留的同时兼顾分布差异。

**🔧 技术方法**

核心技术包括：教师压缩（teacher‑compression）目标、基于多任务稀疏学习的候选特征筛选、基于岭回归的候选交换排名、验证式局部搜索以及幂均值福利聚合。

**📊 数据集**

实验使用了五个公开数据集：ACS 收入、NHANES 儿童和成人铅测定、HELOC 信贷风险、UCI Adult 收入以及 43 州 COVID‑19 症状搜索现在预测，共涵盖 10–2,110 个候选特征与 9,870–883,984 条样本。

**📈 对比分析**

与 Lasso、XGBoost、DRFS 等基线相比，PopFS 在所有 8 个人群划分上平均提升 5–22% 的标签收益，并在最差人群上提升 11–40% 的收益；运行时间仅 3–12 分钟，显著低于 DRFS 的 24–600 分钟。

**⚠️ 局限性**

局限性包括：依赖教师模型的质量；搜索过程为启发式，未提供全局最优保证；目前无法处理新出现或演化的人群，也未直接纳入异质采集成本与不确定性。

---

## 258. Single Canonical Prompts Underestimate LLM Safety's Surface-Form Sensitivity

**arXiv ID:** 2608.02665 | [PDF](https://arxiv.org/pdf/2608.02665v1)

**作者:** Yongxi Zhou `[一作]` (Northeastern University), Lai Yun Choi `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在安全评估中对单一表面形式的基准进行实验，使用预先编写、非LLM的多种意义保持改写，测量多种模型在不同表面形式下的拒绝与不安全合规率。

**💡 创新点**

提出了无偏见、噪声底线控制的测评协议，包括预先改写、跨供应商人类锚定评审、意图保持检查和安全与过度拒绝的双向不稳定性评估。

**🔧 技术方法**

使用机器反向翻译、机器翻译、Matrix Language Frame 代码切换生成、LLM无拒绝重构；评估使用Claude人类锚定评判器、GPT-4o交叉校验；统计采用McNemar检验、Bootstrap CI、Wilson CI、对齐一致性分解。

**📊 数据集**

基于 HarmBench 与 AdvBench 的 370 个恶意种子，生成 5 种表面形式共 1,850 条提示；还使用 XSTest 的 250 条安全提示作为对照。

**📈 对比分析**

对每种模型和改写形式计算拒绝与不安全合规率，比较单一表面形式与多表面联合暴露差异；发现单一形式低估不安全合规 5–13%，多形式联合覆盖率最高可达 100%，且各模型差异显著。

**⚠️ 局限性**

评测仅基于单语言(英中)、单回合、温度0；评审器为单一LLM且未全量人工标注；意图保持评估与评审器同一模型，缺乏独立验证；三种改写共享中文翻译工具，缺少完全独立的改写方法。

---

## 259. Right Divisibility in Erasing Semi-Thue Systems: A Minimal View of Intruder Deduction

**arXiv ID:** 2608.03274 | [PDF](https://arxiv.org/pdf/2608.03274v1)

**作者:** Raja Oktovin O. P. Damanik `[一作]` (Australian National University), Alwen Tiu `[通讯]` (Australian National University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从最小结构角度研究符号安全协议中的入侵推理问题，将一元符号的推理转化为半图系统（semi‑Thue system）的右除性问题，并进一步推广到包含上下文消除的项重写规则。

**💡 创新点**

创新点在于首次证明在有限收敛前缀消除系统和后缀消除系统（即递减系统）下右除性问题可判定，给出了线性时间算法；同时揭示了在同时变量提升（SVL）系统中推理已不可判定，展示了可判定性边界。

**🔧 技术方法**

主要技术包括将推理问题映射到字符串重写，构造后缀乘积集合、利用收敛性得到唯一规范形，设计前缀/后缀消除规则的正向/反向搜索算法，并使用图论中的 Cayley 图来处理后缀消除的反向遍历。

**📊 数据集**

文章未使用实验数据集，而是通过理论证明和构造例子验证结果。

**📈 对比分析**

对比传统的子项收敛、可换、收缩等可判定类，本文的结果在更宽松的消除规则下仍保持可判定，且前缀消除系统的算法时间为 O(|u|+|v|)，后缀消除系统则为指数时间。

**⚠️ 局限性**

限制在于：对后缀消除系统的算法仍呈指数复杂度；对更一般的变量提升或子项提升项重写系统，推理是否可判定尚未解决；此外，证明中使用的假设依赖于系统的收敛性，非收敛情况未考虑。

---

## 260. ValueFormer: A Causal Transformer Value Function with Stage-Aware Labels for Semi-Autonomous Vision-Language-Action Policies

**arXiv ID:** 2608.02958 | [PDF](https://arxiv.org/pdf/2608.02958v1)

**作者:** Inkyu Sa `[一作]` (Chefrobotics), Rajat Bhageria `[通讯]` (Chefrobotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ValueFormer——一种在现有 Vision‑Language‑Action（VLA）政策旁运行的因果 Transformer 价值函数，能够一次前向传递输出两条每帧信号：平滑的 Monte‑Carlo 进度值 V_mc 以及尖锐的在线错误检测值 V_bin；并将其应用于真实双臂三明治组装任务。

**💡 创新点**

创新点：
- Stage‑aware 的成功‑后‑衰减 MC 标签，使失败前段与成功段保持一致、失败后段平滑衰减；
- 双头共享 backbone，平滑 head 用于优势估计，尖锐 head 用于即时错误报警；
- 采用段级错误标注（t_start, t_end）而非单点失败时间，能够为已恢复的错误提供监督；
- 结合冻结的 DINOv3 ViT‑L/16 多视图特征和 2 层因果 Transformer，模型参数仅约 3.5M，且可在单 GPU 上 2 Hz 运行。

**🔧 技术方法**

技术与实现：
- 冻结的多视图 DINOv3 ViT‑L/16（6 视图）特征提取，按帧缓存；
- 视觉、关节状态和时间特征三者线性投影后相加；
- 16 步 8 s 的因果 Transformer（2 层、4 头）；
- 两个 MLP 头，分别输出 V_mc 与 V_bin；
- 采用 BCE 损失（对 V_bin）和 MSE‑sigmoid（对 V_mc），权重相同；
- BF16 自动混合精度、批量化编码实现 3–5× 的推理加速。

**📊 数据集**

数据集：
- LeRobot v2 格式，1,249 条专家成功演示；
- 178 条自主 roll‑out（90 成功、88 失败）；
- 213,102 个 0.5 s 采样点（共 1,427 条 episode）；
- 训练集与验证集按 episode 划分，约 80/20。

**📈 对比分析**

比较与性能：
- 与四种失败标签（outcome‑scaled、cliff、α‑mix、late‑diverge）对比，MC‑smooth 在验证 BCE、MAE 以及 roll‑out 误差上均最优；
- 在真实机器人上，V_mc 的 MSE ≈ 3.0×10⁻⁴、MAE 0.015，V_mc 与 V_bin 在 held‑out episode 上分别重现四种典型签名；
- 将 V_mc 用作每帧训练权重后，在 20 条双三明治实验中，任务完成率从 70% 提升至 85%，且完全消除了重复抓取错误；
- 推理成本通过批量化 + BF16 编码降低 3–5×，使同一 GPU 可同时跑 VLA 与 ValueFormer。

**⚠️ 局限性**

局限与待改进：
- 失败标注仍需人工或介入日志，难以扩展到海量日志；
- 价值函数受因果窗口限制，失败检测存在 5–10 s 延迟；
- 仅在单一三明治组装任务验证，跨任务泛化需进一步评估；
- 只实现了训练权重闭环，其他闭环方案（主动中止、优势条件化、离线过滤）尚未完整验证；
- 失败类型主要来自人为误操作，硬件故障等更复杂异常尚未建模。

---

## 261. DeRP: An Algorithm for Self-Assembly of Power-Delivery Networks using Recursive Branching in Information-Limited Environments

**arXiv ID:** 2608.02904 | [PDF](https://arxiv.org/pdf/2608.02904v1)

**作者:** Mohammadali Rashidioun `[一作]` (New Jersey Institute of Technology), Petras Swissler `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了名为DeRP的去中心化机器人集群算法，使机器人仅凭本地通信和指向目标的方位感知，递归自组装出连接源点与多终端（sink）的电力网络。

**💡 创新点**

创新点在于采用局部逼近Steiner三角角度（120°）的支点选择与递归分支策略，实现无需全局知识即可构建近最优网络，并显著降低电力传输损耗。

**🔧 技术方法**

使用技术包括：机器人本地间距控制、基于方位的邻居姿态估计、局部通信与信号广播、支点评估指标E(r)、递归分支与子节点指向平均方位等；所有操作均在仿真环境下实现。

**📊 数据集**

使用的数据集为在2500×2500单位平面内随机均匀生成的sink位置集合（最小间距为r_comm），每个实验规模（n=2~100）进行50次独立试验。

**📈 对比分析**

与最优欧氏Steiner树（GeoSteiner）和最小生成树（MST）进行对比，结果显示DeRP网络长度在1.04–1.25倍内，电力损耗约为GeoSteiner的65%，完成步骤呈子线性增长，且支点数仅为GeoSteiner的1.53倍。

**⚠️ 局限性**

主要局限包括：支点位置静态、支路分叉偏向固定方向、仅在均匀随机分布下评估、缺乏真实硬件验证以及对非均匀或动态sink分布的鲁棒性未深入研究。

---

## 262. Learning Molecular Representations from Cellular Phenotypes with Structure Preservation

**arXiv ID:** 2608.02688 | [PDF](https://arxiv.org/pdf/2608.02688v1)

**作者:** Xuan Lin `[一作]` (Xiangtan University), Dapeng Xiong `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了 PhenMol，一个结构保留的表型感知分子表示学习框架，通过将分子与细胞表型的共享与私有特征进行解耦，并在私有分支中保留分子骨架信息，实现了跨模态对齐与化学邻域保持的统一。

**💡 创新点**

创新点在于通过共享/私有分解以及骨架保留策略，解决了传统全局对齐导致的分子结构坍塌问题；同时结合全局对比学习和细粒度匹配，实现了在保持化学结构的前提下有效利用细胞表型信息。

**🔧 技术方法**

技术上使用ECFP4指纹与预训练的ResNet-50细胞图像特征；模型包含对比学习（ITC）、潜在匹配（ITM）、正交约束、批判性逆网络和分子骨架预测等多重损失，形成多任务深度学习框架。

**📊 数据集**

数据集包括约3.04万对的JUMP-CP与Broad Drug Repurposing Hub进行预训练；270个生物活性 assay用于属性预测；分子-图像检索测试集；以及Phase I–III的临床试验结果数据集。

**📈 对比分析**

与单模态基线（CellProfiler、ResNet、MACCS、RDKit、Morgan）以及多模态对齐基线（CLOOME、MIGA、MINER）比较，PhenMol在随机和结构拆分下AUC提升约4–5%、AUP提升约3–4%；检索Hit@10最高；在临床III阶段AUC和PR-AUC排名第一；在结构保留指标R-Precision、Rank Shift、Jaccard上显著优于其他方法。

**⚠️ 局限性**

局限性在于仅使用分子指纹和细胞图像，未整合靶点、通路、转录组、疾病背景及临床变量；在Phase II临床结果预测中表现略逊于部分基线；对批处理和染色变化的鲁棒性仍有限，未来需要加入更多多模态数据并提升可解释性。

---

## 263. AI Sandbox: Technical Report

**arXiv ID:** 2608.02679 | [PDF](https://arxiv.org/pdf/2608.02679v1)

**作者:** Muhammad Waseem `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

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

## 264. Diversity is Not Ambiguity: Toward Accurate and Efficient Ambiguity Detection for Open-Domain QA

**arXiv ID:** 2608.03177 | [PDF](https://arxiv.org/pdf/2608.03177v1)

**作者:** Jiwon Lee `[一作]` (Seoul National University), U Kang `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套完整的框架，用于在开放域问答系统中准确且高效地判断查询是否存在歧义；

**💡 创新点**

核心创新在于把歧义定义为答案集合中存在逻辑冲突，而非单纯的答案多样性，并基于此构建了四分类互斥的歧义类型；

**🔧 技术方法**

技术上采用了轻量级的早期退出编码器处理可从表面判断的歧义，再通过生成候选答案、使用自然语言推理（NLI）构建答案对关系网并引入查询门控与抗噪扰动的 invariance 损失，实现对逻辑冲突的精细建模；

**📊 数据集**

主要使用了自建的 AmbigQ 基准（包含多源的 factoid 与非 factoid 查询），以及 ORCAS、AmbigNQ、AmbER、Dolly-15K 等公开数据集的注释；

**📈 对比分析**

与九种基线（提示式、启发式、多样性度量、训练模型等）对比，框架在二分类与四分类任务中均取得最高的 F1（四分类 macro‑F1 达到 94.8%）、最高的二分类 F1（最多 89%），且平均每条查询只调用 3.3 次 LLM，平均延迟 0.13 s，比最优基线快 2‑3 倍；

**⚠️ 局限性**

局限性包括：仅在英文数据上评估；对候选答案的生成依赖冻结的 LLM，若 LLM 或 NLI 模型知识不足会影响性能；NLI 对实体层面的冲突识别能力有限；并且在基准构建时过滤了时间变化或用户特定的查询，未涵盖全部真实场景。

---

## 265. CROSS: Cascaded Distillation and Dual-Constraint Grounding for Remote Sensing Referring Segmentation

**arXiv ID:** 2608.03147 | [PDF](https://arxiv.org/pdf/2608.03147v1)

**作者:** Tingzhang Luo `[一作]` (City University of Hong Kong), Jianyuan Guo `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CROSS框架，结合了深度耦合的 Linguistic‑Guided Cascaded Distillation（LGCD）和 Perspective‑Spatial Contrastive Learning（PSCL），实现了遥感参考分割的结构先验注入与空间推理提升；

**💡 创新点**

创新点在于：①将 SAM 的像素级结构先验通过层级蒸馏融入 SigLIP 视觉语言模型，实现语义与几何的深度耦合；②构造视角‑空间对比学习，挖掘视觉负样本与空间反事实文本，显著抑制对象中心偏差；

**🔧 技术方法**

技术手段包括：SigLIP 2 + SAM 2 基础模型；LGCD（层级表示提取 + 文本引导关系蒸馏）；PSCL（视角硬负样本 + 空间反事实文本对比）；LoRA 微调；CKA 结构一致性评估；BF16 + DeepSpeed ZeRO‑2 训练；

**📊 数据集**

使用的公开数据集为 RefSegRS（512×512）与 RRSIS‑D（800×800）两大遥感参考分割基准；

**📈 对比分析**

与多种 SOTA 方法（RSRefSeg‑2、RS2‑SAM2、SegEarth‑R1 等）在 Pr@0.5/0.6/0.7/0.8/0.9、cIoU、gIoU 上进行对比，CROSS 在所有指标均领先，尤其在 Pr@0.9 上提升 6–7% 以上；

**⚠️ 局限性**

局限性包括：训练时需冻结大模型并微调 7% 参数，计算资源需求较高；对极小目标的边界细化仍有提升空间；未在更广泛的多任务或更大尺度场景中验证泛化能力。

---

## 266. Stuck on "A": Diagnosing and Repairing Interface Injury in Attention-to-KDA Linearization of a 0.6B Language Model

**arXiv ID:** 2608.02689 | [PDF](https://arxiv.org/pdf/2608.02689v1)

**作者:** Ronglong Bao `[一作]` `[通讯]` (DT-Project), Ronglong Bao (DT-Project)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将Qwen3-0.6B-Base模型的21层全注意力层转换为KDA线性注意力层，并通过层级对齐、KL蒸馏、完成仅KL训练与格式目标修复，最终获得在C‑Eval上约41.8%准确率的学生模型，同时实现persona对齐。

**💡 创新点**

提出四重置换诊断方法检测并修复接口损伤；提供在消费级GPU预算下的完整转换流程与故障记录；证明persona对齐可在架构转换后保持。

**🔧 技术方法**

使用KDA线性注意力、前向KL蒸馏、层级隐藏状态对齐、完成仅KL训练、SFT＋DPO人设对齐、BF16优化、缓存校验等技术。

**📊 数据集**

使用Qwen3预训练语料、6,250个教师验证的多选题、3,000句诗歌QA、3,000句翻译QA、C‑Eval基准以及内部自制的四重置换多选题。

**📈 对比分析**

通过C‑Eval、四重置换诊断以及与教师模型的对比，学生最终在C‑Eval上达到41.83%（相对教师50.6%），四重诊断准确率从36.18%提升至53.73%，证明接口损伤得到修复但整体性能仍低于教师。

**⚠️ 局限性**

仅单一模型单一种子；未对MMLU/CMMLU等长序列生成做测试；四重诊断未包含无标签内容评分；GPU预算限制下的通用性未知；温度消融缺乏统计显著性；对BF16更新的依赖未完全验证。

---

## 267. Knowledge-Geometry Decoupling: Refreshable Pretrained Transfer for Streaming Recommendation

**arXiv ID:** 2608.02738 | [PDF](https://arxiv.org/pdf/2608.02738v1)

**作者:** Zixuan Wang `[一作]` (Xiamen University), Hui Li `[通讯]` (Xiamen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在推荐系统中，提出知识–几何解耦框架 KGD，通过行为多标记预测 BMTP 预训练，结合只读交叉注意力和锚定校准残差，实现预训练知识与任务几何的解耦并支持持续刷新。

**💡 创新点**

创新点在于：① BMTP 过滤仅保留协同或语义相关的未来项，去除无关相邻转移噪声；② 将预训练编码器和任务学习器分离，预训练权重可持续刷新；③ 通过锚定校准残差在预训练嵌入正交空间写入任务几何，避免互相干扰。

**🔧 技术方法**

技术包括：Transformer 编码器、下一项预测、行为多标记预测、只读交叉注意力、锚定校准残差（ACR）、低秩正则化。

**📊 数据集**

数据集：公开的八个 Amazon 2023 评测数据集（Arts、Beauty、CDs 等）以及 Shopee 首页搜索的工业流数据。

**📈 对比分析**

与多种预训练目标（NTP、MTP）和迁移方式（全微调、冻结、Adapter、LoRA、Buffer replay）对比，KGD 在公开数据集上提升 4–12% 记录 NDCG/Recall，工业流上在 28 天和 90 天的 AUC/GAUC 上最高，线上 A/B 测试 GMV 提升 1.75%，广告收入 1.53%。

**⚠️ 局限性**

局限：需要额外的任务学习器和锚定残差实现，模型参数和推理成本略增；对极端稀疏尾部用户依赖语义过滤效果有限；仍需在更大规模多任务环境验证。

---

## 268. BBOWP-Bench: Evaluating LLMs on Black-Box Optimization Word Problems

**arXiv ID:** 2608.02612 | [PDF](https://arxiv.org/pdf/2608.02612v1)

**作者:** Yutaro Yamada `[一作]`, Shinichi Shirakawa `[通讯]` (Yokohama National University)

**通讯引用:** 1716 | [OpenAlex ID](https://openalex.org/A5005268349)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出黑盒优化单词问题（BBOWP）并构建对应的数据集与评估框架BBOWP-Bench，使系统能够根据自然语言描述同时推断搜索空间与合适的优化算法。

**💡 创新点**

创新点：①将LLM用于黑盒优化的完整推断（搜索空间+算法）而非仅生成代码；②设计了量化指标FRS和EPDF用于跨域可比的评估；③提供自然语言描述与可执行环境一体化的BBO基准，推动自动化BBO研究。

**🔧 技术方法**

技术手段包括：使用Gemini和GPT‑5系列LLM作为元求解器；通过Docker镜像部署可执行评估环境；利用OptunaHub实现BBO执行；统计FRS与EPDF进行实验分析。

**📊 数据集**

数据集来源于四个真实任务库（YAHPO Gym、Olympus、MECHBench、MuJoCo），共23个实例，每个实例提供易/中/难三层自然语言描述、可执行评估脚本和人类基线配置。

**📈 对比分析**

比较方法：对元求解器输出的搜索空间和算法组合进行FRS评估，并通过EPDF检视搜索空间质量；结果显示GPT‑5 mini与Gemini 3 Flash在Easy/Hard场景表现最佳，平均FRS在0.5–0.7之间，远低于人类基线0.964；LLM在算法选择上表现稳定，但在搜索空间设计上仍有显著差距。

**⚠️ 局限性**

局限性：①生成变量与可执行环境的标识符映射不完整，导致部分变量被忽略；②无法支持条件搜索空间；③算法候选集有限，缺乏自适应算法生成；④未能覆盖所有真实应用场景，评估范围受限。

---

## 269. PI-Mem: Pushing Long-Context Reasoning to 3.6M Tokens with Parallel-Iterative Memory

**arXiv ID:** 2608.03048 | [PDF](https://arxiv.org/pdf/2608.03048v1)

**作者:** Dawei Liu `[一作]` (Shanghai Jiao Tong University), Bowen Zhou `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了并行迭代记忆（PI‑Mem）机制，以并行读取所有文本块并在有限轮次内迭代更新共享记忆，提升极长上下文推理能力。

**💡 创新点**

首次将所有块并行读取、证据选择与合并分离，并通过强化学习引入轮次效率奖励，消除递归更新导致的证据覆盖和串行延迟。

**🔧 技术方法**

使用基于Transformer的大模型（Qwen3.5‑35B‑A3B、Qwen2.5‑7B）、并行 Read‑Select‑Merge 流程、轨迹级强化学习（GRPO）、多轮记忆融合与退出机制。

**📊 数据集**

以 HotpotQA 为基准生成长上下文样本，评测于 HotpotQA、RULER、LongBench v2。

**📈 对比分析**

与直接推理、YaRN、RAG、MemAgent、GRU‑Mem、ReMemR1 等基线比较；在 3.6M token 下 PI‑Mem 约比 MemAgent 提高 6–8 分、推理速度提升 6–16×；在 RULER OOD 与 LongBench 上亦取得最高分。

**⚠️ 局限性**

RL 训练依赖合成数据，训练复杂度高；仍需在更大模型或多语言场景验证；对极长序列的硬件资源需求仍显著。

---

## 270. One Discrete Gaussian Sample in $2^{n/2+o(n)}$ Time

**arXiv ID:** 2608.03220 | [PDF](https://arxiv.org/pdf/2608.03220v1)

**作者:** Jiseung Kim `[一作]` `[通讯]` (Jeonbuk National University), Jiseung Kim (Jeonbuk National University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了在任意参数下单次离散高斯采样，时间与空间均为 2^{n/2+o(n)}，并将该结果应用于求解近似最近点问题和最短向量问题。

**💡 创新点**

创新点在于：①将 ADRS 的快速 2^{n/2+o(n)} 采样时间推广到单样本；②利用随机稀疏化得到一个在目标尺度下平滑且保留原格点的超格；③用高斯质量比较证明 2^{n/2} 的上界是紧的；④借助该点采样实现 α<1/√μ≈1.4697 的子 2^n CVP 与 2^{0.7315n+o(n)} 的 SVP。

**🔧 技术方法**

技术手段包括：离散高斯采样（ADRS 算法）、格子稀疏化（Dadush–Kun），泊松求和、对称性与平滑参数分析、随机超格的构造与质量比较、随机 Turing 机实现与精度裁剪。

**📊 数据集**

论文未使用具体数据集，而是针对任意给定的整数/有理基底的格子进行理论证明与复杂度分析。

**📈 对比分析**

与之前的 2^n+o(n) 采样算法相比，新方法在仅需一次样本时将时间/空间降至 2^{n/2+o(n)}；在 CVP 上，在 α<1.4697 时实现了严格小于 2^n 的指数时间；在 SVP 上实现了 2^{0.7315n+o(n)} 的时间复杂度。

**⚠️ 局限性**

局限性：①仍需存储 2^{n/2+o(n)} 个样本；②算法仅适用于中心化的高斯采样，对位移的采样成功率受高斯质量比影响；③实现复杂度高，需要对随机采样和数值比较进行严格裁剪。

---

## 271. When Truth Is Distributed: Misinformation Derails Collective Fact Recovery in LLM-Based Multi-Agent Systems

**arXiv ID:** 2608.03421 | [PDF](https://arxiv.org/pdf/2608.03421v1)

**作者:** Chenfei Yan `[一作]` (Chinese Academy of Sciences), Yi Zeng `[通讯]` (Renmin University of China)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Hi-Agreement 框架，用以在受控的多智能体 LLM 系统中评估分布式信息整合与对抗性误导的影响，并通过一系列阶段性投票、证据引用与传播轨迹分析，对真伪信息在讨论过程中的演化进行追踪。

**💡 创新点**

创新点在于：①将“诚实合作”与“单一关键证据持有者的误导”进行成对实验；②引入多阶段传播阶数与根证据（true/false-root）线索的量化；③通过关键智能体退出和观察者干预实验，揭示误导信息在系统内部的持久性与传播机制。

**🔧 技术方法**

使用了三种主流大语言模型（GPT‑5.5、DeepSeek‑v4‑pro、Grok‑4.5），在预设的五代理讨论协议下，结合多阶段投票、证词引用、公开讨论以及根证据传播记录等技术。

**📊 数据集**

实验数据集为 120 条改编自 Hi-ToM 的对象移动故事，每条故事构造了状态链、观察窗口与通信内容，确保部分可观测、信息充分且存在信息重叠。

**📈 对比分析**

比较方法：在诚实 (C0) 与误导 (C1) 条件下分别测量 R3 阶段的 truth-majority 与 decoy-majority 率，并记录证词引用/采纳比例、传播阶数及根证据状态转移。结果显示：诚实条件下真值多数率为 72.5%，误导条件下降至 14.17%；所有三种模型在误导条件下均显著下降，且误导信息被更频繁采纳、传播阶数更深，误导者退出后仍能持续影响。

**⚠️ 局限性**

局限性：实验仅在合成的单一对象移动任务上进行，未涉及真实世界复杂情境；仅使用三种模型，缺乏跨模型泛化验证；观察者和误导者的设置均为人工控制，缺乏自然人类交互与真实社交网络结构。

---

## 272. Staying on Spec: Real-Time Monitoring under Uncertainty with a Maritime Case Study

**arXiv ID:** 2608.02811 | [PDF](https://arxiv.org/pdf/2608.02811v1)

**作者:** Elizabeth Dietrich `[一作]` (University of California, Berkeley), Murat Arcak `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于 pacSTL 的实时不确定性监测框架，利用实验数据驱动的可到达集实现对海上航行任务的安全性评估和逃逸决策。

**💡 创新点**

创新点包括：① 通过有限实验数据构造分布支持并结合高保真仿真实现 PAC‑bounded 可到达集，显著降低对大规模数据的需求；② 针对海上交通规则设计可在实时部署的 pacSTL 规范；③ 将可到达集与 pacSTL 语义结合，获得带概率保证的鲁棒性区间。

**🔧 技术方法**

技术手段：pacSTL 语义与区间优化、数据驱动可到达集（采样‑优化＋包络方法）、仿真‑实验闭环的 sim‑to‑real 校准、线性/二次/非线性原子命题的鲁棒性边界计算。

**📊 数据集**

使用了约 100 次真实船舶实验记录（控制输入与运动捕捉数据）来估计扰动分布，并在 Python 仿真中生成高保真轨迹；同时采用 JONSWAP 频谱生成波浪扰动作为实验环境。

**📈 对比分析**

与传统 STL 监测和基于 TCPA 的风险指标进行对比。实验显示 pacSTL 在静水和波浪条件下均能更早触发逃逸、碰撞率更低、检测率更高，尤其在存在波浪扰动时优于其他方法。

**⚠️ 局限性**

局限性：① 需要先行获取足够的实验数据来估计扰动分布；② 计算量主要集中在离线阶段，对实时性能有一定要求；③ 当前仅针对单一代理（主船）设计，扩展到多代理场景仍需研究；④ 对于极端扰动或模型偏差，所构造的可到达集可能不再满足 PAC 约束。

---

## 273. Joint Affine Spectral Shaping: Coupling Weight and Bias Updates Beyond Weight-Only Muon

**arXiv ID:** 2608.02991 | [PDF](https://arxiv.org/pdf/2608.02991v1)

**作者:** Gongyue Zhang `[一作]` (Harbin Institute of Technology, Shenzhen), Honghai Liu `[通讯]` (Harbin Institute of Technology, Shenzhen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在矩阵谱优化中将权重与偏置一起做谱变换的效应，提出Joint Regularized Inverse（JRI）方法；

**💡 创新点**

创新点在于把权重与偏置视为一个联合的谱对象，并同时将谱变换后的偏置列用于物理更新；

**🔧 技术方法**

使用了稀疏SVD、capped regularized‑inverse谱映射、α=√(in_features)缩放、以及传统Adam做偏置参考；

**📊 数据集**

在BERT‑mini模型上使用IMDb情感分类数据集进行实验；

**📈 对比分析**

与Exact‑SVD Muon、权重仅逆谱、以及bias probe等四种对照方法进行严格五种种子对照，JRI在验证选择的测试精度上提高约0.25个百分点，测试损失降低约0.007，整体表现最优；

**⚠️ 局限性**

局限包括对单一小型Transformer和单一文本任务的评估、需要每步SVD计算导致速度慢、α缩放与偏置物理尺度耦合、以及仅验证了逆谱场景，未探索正谱或多种任务的普适性。

---

## 274. Crayotter: Learning Long-Horizon Video Editing Agents via Group-Relative Preference Backpropagation

**arXiv ID:** 2608.02694 | [PDF](https://arxiv.org/pdf/2608.02694v1)

**作者:** Lecheng Yan `[一作]` (University of Science and Technology of China), Cathal Gurrin `[通讯]` (Dublin City University)

**通讯引用:** 5299 | [OpenAlex ID](https://openalex.org/A5014224452)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Group‑Relative Preference Backpropagation（GRPB）的长期视频编辑代理学习框架，利用同一任务下不同编辑轨迹的最终产品偏好，将偏好转化为零和序列优势并通过延迟、可控的分配器将信用分配到语义编辑段落，从而训练一个9B参数的Crayotter模型；

**💡 创新点**

核心创新包括：① 在同任务内把主观最终产品偏好映射为零和序列优势，消除跨任务量化不一致；② 设计延迟的Bradley–Terry分配器结合可靠性门控与段落级上限，实现可控的段落信用分配；③ 将该信用与PPO+GAE结合，完成从偏好到策略的端到端优化；

**🔧 技术方法**

主要技术手段包括：强化学习（PPO + GAE）、零和序列优势计算、Bradley–Terry 段落评分模型、延迟（lagged）分配器、可靠性门控、段落级上限分配、人工构造的项目级同任务对照组；

**📊 数据集**

使用手工收集的23个真实后期项目，拆分为70个任务（12正常、12中期、46长期），并在项目间保持离散；此外还利用公开的AgenticVBench外部基准评估模型跨任务泛化；

**📈 对比分析**

与Process‑PPO、Terminal Rank PPO、Uniform Allocation等基线对比，GRPB在AgenticVBench上平均得分15.7、Repurpose 23.0，排名第三，超过多家专有系统；在人类评测中对比基线净胜率达+34.2，显著提升视频质量和编辑行为；

**⚠️ 局限性**

局限性包括：依赖人工构造的同任务对照组，训练成本高且模型规模大；信用分配仍需可靠性门控，可能对评判者偏差敏感；跨任务可比性与不同编辑任务的多样性仍未完全覆盖；

---

## 275. Benchmarking the Benchmarks: Testing the Predictive Validity of Commonsense Benchmarks

**arXiv ID:** 2608.03340 | [PDF](https://arxiv.org/pdf/2608.03340v1)

**作者:** Ine Gevers `[一作]` (University of Antwerp), Walter Daelemans `[通讯]` (University of Antwerp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统评估23款LLM在传统与改写的常识基准以及8个下游任务上的表现，检验常识基准对下游任务的预测效度；

**💡 创新点**

创新点在于首次对原始与修订基准的模型排名保持性和对下游任务的预测效度进行比较，并揭示常识基准对大部分下游任务的预测力极为有限且不一致；

**🔧 技术方法**

采用log‑likelihood多选评估、Spearman相关、偏相关及留一族交叉验证等统计方法进行分析；

**📊 数据集**

使用的基准包括WinoGrande、HellaSwag、Social IQA、Physical IQA及其改写版，控制任务BLiMP、GPQA‑Diamond、MMLU‑Redux，下游任务包括CEI、SARC7、False Beliefs、Implicature、Presupposition、Indirect Requests、TimeDial和TRIP；

**📈 对比分析**

比较方法包括相关系数、偏相关与LOFOCV预测，结果显示仅对False Beliefs和TRIP产生显著预测，其他下游任务的预测力弱或无；

**⚠️ 局限性**

局限性包括模型和族类数量有限、下游任务样本量小、仅采用log‑likelihood评估方式，未涵盖更广泛的评价格式与应用级结果。

---

## 276. Adaptive Two-Stage Visual Token Pruning for Efficient Inference in Video-Language Models

**arXiv ID:** 2608.03112 | [PDF](https://arxiv.org/pdf/2608.03112v1)

**作者:** Paribesh Regmi `[一作]` (Amazon.com Services LLC), Hongda Mao `[通讯]` (Amazon.com Services LLC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对视频输入的两阶段自适应视觉令牌裁剪策略

**💡 创新点**

创新点在于结合帧级和令牌级两阶段裁剪，并通过令牌嵌入相关性谱自适应确定第二阶段的保留比例

**🔧 技术方法**

采用帧级多样性选择、令牌嵌入相关矩阵特征值分解、指数衰减拟合以及低秩SVD快速估计

**📊 数据集**

在VideoDetailCaption、VideoChatGPT、NextQA、PerceptionTest、Video-MME等多种视频语言基准数据集上进行评估

**📈 对比分析**

与PruMerge、FastV、AvgPool、DivPrune、LLaVA-Scissor等训练‑free裁剪基线对比，在相同平均保留比例下，本文方法在字幕生成和开放式问答任务上提升6%+准确率，同时将计算量（TFLOPs）降低约90%，且在低保留比例（10%）下仍保持2-3%的性能下降

**⚠️ 局限性**

主要局限在于裁剪后可能丢失对极为细粒度视觉信息的捕捉；自适应阈值依赖于相关矩阵估计，易受噪声影响；目前仅在VLM的后置裁剪阶段验证，未探讨与模型微调结合的效果

---

## 277. Qwen-3D: A Generalist 3D Vision-Language Model for Spatial Understanding

**arXiv ID:** 2608.02980 | [PDF](https://arxiv.org/pdf/2608.02980v1)

**作者:** Lucy Lin `[一作]` (Carnegie Mellon University), Katerina Fragkiadaki `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建一种能够在三维世界坐标中进行注意力计算的多模态大模型 Qwen-3D，结合多视角深度与相机位姿，实现了长视频的压缩与跨视角推理；

**💡 创新点**

创新点在于：①利用 voxel‑based 3D 旋转位置编码（3D RoPE）让视觉–语言注意力直接在世界空间上操作；②采用端到端的 query‑based 语义分割解码器，将语言语义与 3D 场景特征密切耦合，消除传统 3D LMM 的语言/提议瓶颈；③联合 2D‑3D 训练，保持 2D 视觉能力并提升 3D 认知。

**🔧 技术方法**

技术包括 Qwen2.5‑VL 视觉编码器、3D RoPE、voxel‑pooling 视觉特征压缩、LoRA 微调、Mask2Former‑style 3D 分割解码器、Hungarian 匹配多任务损失、以及基于 2D‑3D 数据的联合训练。

**📊 数据集**

使用的主要数据集有 3D 语义引用与定位：SR3D、NR3D、ScanRefer；3D 实例分割：ScanNet200、Matterport；3D VQA：ScanQA、SQA3D；2D 任务：RefCOCO/RefCOCO+/RefCOCOg、COCO、LLaVA‑Instruct‑150k、Alpaca。

**📈 对比分析**

与现有 3D LMM（3D‑LLM、Video‑3D‑LLM、LLaVA‑3D、Grounded‑3D‑LLM）以及专家 3D 感知模型（UniVLG、Video‑3D‑LLM）对比，Qwen‑3D 在 3D 引用定位 Top‑1@0.5 提升 12%~30%，在 ScanNet200 mAP 提升 13%，在 3D VQA 上与 LLaVA‑3D‑7B 持平或更优，且保持 2D 任务性能不降。

**⚠️ 局限性**

局限性：假设世界坐标随时间静止，无法处理动态非刚性场景；依赖外部深度与相机位姿估计，误差会直接影响性能；对多步指令推理、长程交互式任务的支持尚待扩展。

---

## 278. A Wearable Stiffness-Rendering Haptic Device with a Honeycomb Jamming Mechanism for Bilateral Teleoperation

**arXiv ID:** 2608.03002 | [PDF](https://arxiv.org/pdf/2608.03002v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 279. PatTree: a novel approach for automated creation of multimodal, graph-based patient representations for medical classification tasks

**arXiv ID:** 2608.02692 | [PDF](https://arxiv.org/pdf/2608.02692v1)

**作者:** Julia Gehrmann `[一作]` (University of Cologne), Oya Beyan `[通讯]` (University of Cologne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了PatTree，一种基于图的、自动构建的多模态患者表示方法；

**💡 创新点**

创新点在于无需预先标准化或命名实体识别，直接利用患者旅程和临床信息系统的自然结构构建知识图，兼容缺失值、时间序列和多模态数据；

**🔧 技术方法**

采用句子Transformer（PubMedBert/BioLORD/Multilingual-E5）生成节点嵌入，结合GATv2图神经网络进行患者级分类；

**📊 数据集**

使用阿尔茨海默病神经影像学计划（ADNI）ADNI-1数据集，包含763名患者的临床表格和结构MRI；

**📈 对比分析**

与基线FT-Transformer对比，在二分类（AD vs. CN）中两种方法均达100%精度；在三分类（AD vs. MCI vs. CN）中PatTree实现98.5%平衡准确率、0.987 F1分数，显著优于基线的88.7%/0.868；

**⚠️ 局限性**

局限包括：尚未系统评估对缺失值和多模态的鲁棒性；图构建和嵌入过程计算量大；对图神经网络的超参数和结构选择仍需进一步优化；

---

## 280. Breaking ACDGV MinRank Gabidulin encryption schemes over matrix codes

**arXiv ID:** 2608.03328 | [PDF](https://arxiv.org/pdf/2608.03328v1)

**作者:** Thai Hung Le `[一作]` `[通讯]` (École normale supérieure), Thai Hung Le (École normale supérieure)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Enhanced Gabidulin Matrix Code（EGMC）加密方案提出了一种新的混合区分器与关键恢复攻击，破解了所有 16 个参数集。

**💡 创新点**

创新点在于仅猜测一个压缩矩阵并用代数方法求解另一个，从而将攻击复杂度从指数降到多项式，并利用 Frobenius 轨道与 2×2 约束的零维理想实现高效求解。

**🔧 技术方法**

采用了组合与代数技术相结合的混合区分器、Frobenius 诱导的 Overbeck 区分、矩阵秩约束求解、Macaulay 矩阵与特征值方法以及 Gabidulin 码的结构重建。

**📊 数据集**

实验基于公开的 EGMC 参数集（如 (2,17,37,4,0)、(2,25,37,3,3) 等）以及在 GitLab 上的攻击实现代码。

**📈 对比分析**

与之前的指数级 MinRank 及组合区分器相比，新攻击在所有参数集上实现了多项式时间，安全级别从 186 位降低到 35 位（128 位参数集），耗时不到 10 分钟，显著优于现有方法。

**⚠️ 局限性**

局限在于攻击需要至少一个压缩矩阵维度为零（ℓ₁=0 或 ℓ₂=0），否则仍需指数级猜测；并且对非随机子空间模型或非均匀参数的稳健性尚未完全证明。

---

## 281. Can Training Logs Make Model Comparisons More Precise?

**arXiv ID:** 2608.02705 | [PDF](https://arxiv.org/pdf/2608.02705v1)

**作者:** Wei-Jung Huang `[一作]` `[通讯]` (Independent Researcher), Wei-Jung Huang (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究训练日志是否能提高随机训练模型比较的精度，提出了 arm‑specific 协变量调整并在 3×3 组合实验中进行验证。

**💡 创新点**

首次将训练日志协变量按模型 arm 单独调整，使用交叉拟合评估并系统探讨协变量选择风险，提供实用的操作指南。

**🔧 技术方法**

采用线性回归协变量调整、PCA 降维、K‑fold 交叉拟合、Welch 置信区间、方差减少量评估以及合成 null 校验等统计技术。

**📊 数据集**

使用 CIFAR‑10、CIFAR‑100 与 Tiny‑ImageNet 三个视觉数据集。

**📈 对比分析**

对比原始无调整与调整后的 95% 置信区间；在 50 次跑量下，多数模型差值区间宽度可缩小 1–9%，但在跑量不足时可能反而扩大。

**⚠️ 局限性**

协变量选择受限于跑量，过度搜索会导致噪声；调整后的置信区间仅在经验上校准，缺乏严格的分布无关保证，且仅适用于固定检查点的最终准确率。

---

## 282. UniGD: A Unified Generative-Discriminative Framework for Industrial Retrieval

**arXiv ID:** 2608.03150 | [PDF](https://arxiv.org/pdf/2608.03150v1)

**作者:** Shujie Ji `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的生成-判别检索框架 UniGD，能够在同一模型中完成候选广告的自回归 SID 生成与显式相关性评分，消除了传统检索链中独立的生成与排序模型所带来的延迟与效果损失。

**💡 创新点**

主要创新包括：①冲突感知梯度增强（CAGE）机制，能够在联合训练时自动协调生成与判别目标的梯度；②代码簿锚定表示模块（CAM），利用多模态预训练模型的分层量化代码簿为长尾与新广告快速生成可靠表示；③异质广告材料建模（HAM），为短视频、商品和直播广告分别构建专属 SID 代码簿与生成头，兼顾统一共享骨干与材料特异性。

**🔧 技术方法**

使用 decoder‑only Transformer 作为共享骨干，配合自回归 SID 生成、分层残差量化代码簿、线性适配器与轻量级评分网络；引入 CAGE 进行梯度协调；采用 CAM 的量化重构实现低延迟表示；实现多材料并行 beam search。

**📊 数据集**

实验数据集：公开检索基准 NQ320K（Natural Questions）和 MS300K（MS MARCO），以及千禧口碑 Kuaishou 搜索广告日志（约 153M 生成正样本、248M 判别正负样本，工业测试集 277K 对 / 10K 人工标注对）。

**📈 对比分析**

与传统的生成+单独排序（GR+RelToken）、仅生成（UniGD‑Gen）以及多种公开基线（DSI、NCI、Ultron、LTRGR、GenRRL、DDRO）进行对比。在线 A/B 测试中，UniGD 相比基线提升广告收入 5.78%、长尾/新广告曝光 16.8%/24.6%，推断延迟从 13 ms 降至 8.7 ms；在 NQ320K 上 Recall@10 提升 8.44%，在 MS300K 上提升 3.19%；工业测试集 Recall@1、MRR@10 等指标亦大幅提升。

**⚠️ 局限性**

当前方法仍依赖离线教师模型提供判别标签，对极度稀有广告的表示仍受限于代码簿的分辨率；在多材料高并发场景下，分层量化与 beam search 的组合可能带来额外的计算开销；此外，统一模型的规模相对较大，部署在资源受限环境时仍需进一步压缩。

---

## 283. TraceCompiler: Skill-Guided Mining and Compilation of LLM Agent Traces into Mostly Deterministic Workflows

**arXiv ID:** 2608.02680 | [PDF](https://arxiv.org/pdf/2608.02680v1)

**作者:** Salma El Yadouni `[一作]`, Guanyi Li `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 TraceCompiler，通过无监督聚类、行为去噪和参数级依赖验证，将 LLM 代理的嘈杂轨迹编译成可执行、几乎确定的工作流。

**💡 创新点**

提出证据驱动的参数级依赖规则，在部分可观测条件下通过排除其它来源来确认生产者-消费者边；支持在不可确定逆向效果时安全放弃编译，并将 LLM 运行拆分为一次性编译与运行时残留 LLM 任务。

**🔧 技术方法**

使用句子编码 + 余弦距离聚类进行意图聚类；行为去噪（标记重试、死读、分页）和证据驱动的依赖验证（最近生产者排除、可选边）；以及基于规则的技能脚本来执行编译后工作流。

**📊 数据集**

使用 T1（13.5k 模板生成对话）、AppWorld（9 个应用 457 个 API 56 场景 168 跟踪）和 Hermes‑Function‑Calling‑v1（单轮函数调用）等公开数据集。

**📈 对比分析**

与邻接、全对、频率阈值直接跟随等基线比较，机制化规则在 T1 训练集上达到 0.928/0.943 精确/召回；技能在 250 条边上 0.992；在 AppWorld 重放参考上 0.993/0.970；编译后调用从 34 降至 11，执行 15/21 状态测试通过。

**⚠️ 局限性**

缺乏编译成本与节省的评估、工作流版本管理与 API 演变兼容性、单一 LLM 版本的确定性保证；只在模板生成或单一代理轨迹上验证，缺少跨语料库完整性测试和编译/拒绝率的完整统计。

---

## 284. Intent-Level Quantum Programming with Assertion-Guided Execution and Inspectable Intermediate Representation

**arXiv ID:** 2608.02648 | [PDF](https://arxiv.org/pdf/2608.02648v1)

**作者:** Ilesh Vora `[一作]` (Institute of Applied Artificial Intelligence and Robotics), Rajesh Vasa `[通讯]` (Deakin University)

**通讯引用:** 1568 | [OpenAlex ID](https://openalex.org/A5030486012)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向意图的量子领域特定语言（QDSL），实现意图与执行分离、可检查的中间表示（IR）、断言驱动的执行模式推断，并将其与 PennyLane、Qiskit 等后端无缝对接，用于验证、调试和持续集成。

**💡 创新点**

创新点包括：①意图级别的构造与后端绑定分离；②首席 IR introspection 以可视化、可验证的形式提供电路宽度、wire 映射、操作顺序等信息；③通过声明式断言自动推断采样/状态向量/混合执行模式；④基于 Type‑A/B/C 语义的阶段分离，防止构造与执行混淆；⑤统一结构化日志与 CI 支持。

**🔧 技术方法**

使用技术包括：QDSL 语法设计、IR 数据结构、断言引擎（Wilson 置信区间、总变差距离检测）、统计与状态向量混合执行、差分测试、字节序规范化、故障注入、Python/NumPy、PennyLane、Qiskit、以及结构化 JSON 日志。

**📊 数据集**

实验数据集为一组经典量子基准电路（Bell、GHZ、Oracle‑based、QFT、VQE、Grover、QAOA、Deutsch‑Jozsa、Teleportation 等）以及针对 Bell 与 3‑qubit GHZ 的故障注入实验。

**📈 对比分析**

对比方法：与 PennyLane、Qiskit 的等价实现进行 LOC、意图密度、模板繁杂度、故障检测 TPR/FPR、跨后端数值一致性、IR 生成时延及内存消耗的对比；结果显示 QDSL 在意图密度上最高，TTR 100%，FPR 2‑4%（仅 128 shots），跨后端误差 ≤ 2.22×10⁻¹⁶，IR 生成子毫秒、内存 < 11 KB，整体性能优异。

**⚠️ 局限性**

局限性在于：仅覆盖小规模电路与两类故障（Hadamard 误置、GHZ 末端门缺失），缺乏更广泛的故障模型；对比依赖人工标注的“oracle”程序；统计断言在极少样本下仍可能产生误报；缺乏对大规模实用程序的验证与泛化实验。

---

## 285. GoT-CD: Graph-of-Thoughts Causal Discovery and the Fragility of Post-hoc Path-Specific Fairness Audits

**arXiv ID:** 2608.02877 | [PDF](https://arxiv.org/pdf/2608.02877v1)

**作者:** Nitish Nagesh `[一作]` (University of California, Irvine), Amir M. Rahmani `[通讯]` (University of California, Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于Graph-of-Thoughts（GoT）的因果结构学习方法GoT-CD，并在该方法基础上进行路径特定公平性审计，探讨结构学习误差对公平性结果的影响。

**💡 创新点**

创新点在于：①将因果发现视为全图推理问题，使用完整候选边集进行并行生成、评分与合并；②通过硬联合约束防止模型提出未被推理的边；③引入后处理的贪心投影保证始终生成DAG；④将路径特定公平性审计与结构学习结合，提出“路径恢复率”作为评估指标，揭示结构F1与公平性结果可能不一致。

**🔧 技术方法**

技术手段包括：图思维（Graph-of-Thoughts）框架、LLM（如ChatGPT）生成候选图、确定性局部评分函数、硬联合合并、贪心投影去环、线性回归求解路径特定效应、离散化计算自然直接/间接效应。

**📊 数据集**

使用五个公开因果基准数据集：Asia、Child、Alzheimer’s Disease、COVID-Respiratory、Sweden-Traffic（样本量均为100），并在Alzheimer’s Disease上进行公平性案例研究。

**📈 对比分析**

与传统方法（PC、GES、NOTEARS、DAGMA-linear）以及LLM基线（pairwise、LLM-BFS、GoT-CD-BFS）比较；GoT-CD在所有五个基准上均保证DAG有效，且在亚洲、阿尔茨海默病、COVID-Respiratory上获得最高的DAG-valid F1；在公平性审计中，GoT-CD恢复了真实的无公平路径并得到正确符号的效应估计，而LLM-BFS等方法即使结构F1相近也可能遗漏关键路径，导致错误的公平性结论。

**⚠️ 局限性**

局限性包括：仅在单一LLM（未公开模型号）和固定分支因子k=3下测试；样本量固定为100，缺乏对更大样本或更复杂结构的验证；公平性评估基于线性-Gaussian SEM，未考虑真实临床数据中的非线性或未观测混杂；循环图需人工去环，且未对更大图（如COVID-Complications、Neuropathic）进行系统评估。

---

## 286. NotDec: WebAssembly Decompilation With Inter-Procedural Type Recovery

**arXiv ID:** 2608.03286 | [PDF](https://arxiv.org/pdf/2608.03286v1)

**作者:** Jikai Wang `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了完整的 WebAssembly‑to‑C 逆编译器 NotDec，完成了从 Wasm 代码到 SSA 中间表示、优化与类型恢复，再到语义保持的 C 代码生成的全流程。

**💡 创新点**

创新点在于① 将 WebAssembly 操作栈转化为 SSA 基础 IR，扩展类型检查以便后续优化；② 结合 Retypd 与自定义 PNDiff 图实现跨函数的结构体恢复；③ 引入 Memory SSA 与语义保留的控制流结构化算法，使生成的 C 代码可读且语义一致。

**🔧 技术方法**

核心技术包括 LLVM SSA 中间表示、Retypd 类型恢复、PNDiff 边图进行指针/数值区分、标量聚合拆分（SROA）、Memory SSA、语义保持的结构化控制流算法以及一系列编译器优化与低级模式匹配。

**📊 数据集**

评估使用 Juliet 测试套件 5,241 个样本和 Howard 数据集（fortune、grep、gzip、lighttpd、wget）五个真实程序。

**📈 对比分析**

与 Ghidra、WasmDec、WaDec 进行对比，NotDec 在 Juliet 上实现 100% 重编译成功率、平均 87.6 行代码；在 Howard 上同样全部成功。执行时间仅 1.73 s、内存 119 MB，显著快于 Ghidra（27.6 s、572 MB）。结构体成员恢复率 85.33% 对比 Ghidra 9.24%，展示出更高的类型恢复精度。

**⚠️ 局限性**

局限性包括：交叉函数类型恢复在大二进制上耗时过长、对间接调用缺乏精确分析、PNDiff 依赖的指针/数值区分不完整导致部分成员未恢复、Memory SSA 近似可能导致临时变量插入，且对高度混淆或非标准编译器生成的 Wasm 支持不足。

---

## 287. Long-term Traffic Scene Prediction via Polynomial Representations in Autonomous Driving

**arXiv ID:** 2608.03330 | [PDF](https://arxiv.org/pdf/2608.03330v1)

**作者:** Yue Yao `[一作]` `[通讯]`, Yue Yao

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于多项式拟合与扩散概率模型的多智能体轨迹预测框架

**💡 创新点**

创新点在于将多项式基函数与DDPM/DDIM扩散模型结合，并引入Kalman滤波与Rauch‑Tung‑Striebel平滑器提升预测精度

**🔧 技术方法**

使用了多项式模型、注意力机制、DDPM/DDIM扩散模型、Kalman滤波、RTS平滑及多层感知网络等技术

**📊 数据集**

实验数据集包括Argoverse 1、Argoverse 2、Waymo Open、HD地图等公开轨迹数据

**📈 对比分析**

与FMAE‑MA、QCNet、OptTrajDiff等基线在minADE、minFDE等指标上对比，取得了显著的性能提升

**⚠️ 局限性**

主要局限在于对大规模场景的实时推理速度与计算成本有待进一步优化

---

## 288. Towards a new paradigm of scientific discovery with socialized artificial intelligence

**arXiv ID:** 2608.02775 | [PDF](https://arxiv.org/pdf/2608.02775v1)

**作者:** Xinjie Yao `[一作]`, Pengfei Zhu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文未提供具体内容，无法确定做了什么。

**💡 创新点**

论文未提供具体内容，无法确定创新点。

**🔧 技术方法**

论文未提供具体内容，无法确定使用了什么技术。

**📊 数据集**

论文未提供具体内容，无法确定使用了什么数据集。

**📈 对比分析**

论文未提供具体内容，无法确定比较的方法及性能。

**⚠️ 局限性**

论文未提供具体内容，无法确定限制是什么。

---

## 289. EduClaw-Bench: A Long-Horizon Benchmark for Pedagogical LLM Agents with Simulated Learners

**arXiv ID:** 2608.03206 | [PDF](https://arxiv.org/pdf/2608.03206v1)

**作者:** Unggi Lee `[一作]` (Korea University), Hoilym Kwon `[通讯]` (Korea University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出EduClaw-Bench，构建30天连续的基于LMS的学习者模拟环境，用于评估教师代理的学习增益、响应率、帮助度以及课程设计一致性；

**💡 创新点**

首次将知识追踪驱动的学习者模拟与长期教师代理对接，并通过多维度评分（学习增益、响应、帮助、甘纳九要素、罗森辛十原则）评估持续教学效果；

**🔧 技术方法**

使用注意力知识追踪(AKT)模型、LLM学生角色扮演、LLM评判面板（gemini‑3‑flash、gpt‑4.1‑mini、kimi‑k2.5）以及基于LMS的交互API；

**📊 数据集**

基于XES3G5M小学数学交互数据训练AKT模型，并采样该数据集中的个人学习轨迹生成学习者人格；

**📈 对比分析**

在10个教师代理与3类基础LLM（Solar‑pro3、Codex‑gpt5.5、Qwen3‑4B‑Thinking）上进行对比，发现代理与基础模型共同决定性能，单一模型排行榜误导；大多数组合在30天内学习增益迅速趋于平稳，课程设计一致性低；

**⚠️ 局限性**

模拟学习者虽与真实学生KL模型对齐（ECE=0.049），但仍为仿真，缺乏真实课堂动态；评估指标对单一代理的改进不敏感，且在更大样本或多领域的泛化尚未验证。

---

## 290. Efficient Grammar-Constrained Decoding via Parser Stack Classification

**arXiv ID:** 2608.03065 | [PDF](https://arxiv.org/pdf/2608.03065v1)

**作者:** Yongmin Li `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的语法约束解码（Grammar-Constrained Decoding，GCD）方法——Parser Stack Classification（PSC），通过预先构造有限状态机（FSA）对词法状态与解析器栈的组合进行分类，从而在解码时仅需一次栈检查即可得到有效词汇掩码。

**💡 创新点**

创新点在于证明每个词条的接收条件可被精确描述为正则语言，并通过构造单一FSA（而非为每个词条单独调用解析器）实现了对语法合法性检查的时间复杂度与词表大小无关，大幅提升了掩码计算速度（最高可达700倍加速）。

**🔧 技术方法**

主要技术包括：基于词法转导器（FST）和确定性上下文无关文法解析器（PDA）构造ε-转导器和任意终结符序列的FST；将这些FST映射为FSA并合并得到统一的验证器FSA；预处理阶段生成所有可能的词汇掩码；解码阶段通过一次栈扫描得到掩码，降低运行时开销。

**📊 数据集**

使用了多种编程语言（Java、Go、Python、SQL）和符合JSON schema的结构化数据集，涉及多种大语言模型（Llama‑3、Qwen‑2.5、Gemma‑3）不同词表规模的测试。

**📈 对比分析**

与现有GCD基线（如LLGuidance、GreatGramma等）对比，PSC在掩码计算上实现了高达30-700倍的加速；端到端解码吞吐量几乎与无约束解码相当，并显著优于其他基线，尤其在小模型和大批量场景中表现突出。性能评估包括在线开销、总体吞吐量、下游任务（代码生成、JSON生成）的通过率。

**⚠️ 局限性**

主要局限是需要对每个语法和词表进行一次昂贵的预处理（时间高达数小时、内存数百GB，尤其针对大型编程语言语法），且仅适用于确定性上下文无关文法（DCFG）；若用户需要自定义语法，需权衡预处理与解码收益，且在非DCFG场景下效果尚未验证。

---

## 291. Getting the Parameters Right: A Difficulty-Graded Benchmark and Probe-Guided Training for LLM Tool Calls

**arXiv ID:** 2608.03071 | [PDF](https://arxiv.org/pdf/2608.03071v1)

**作者:** Guoyao Yu `[一作]`, Zhenguang Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将工具调用参数生成（Parameter Generation）放在大语言模型工具使用研究的核心位置，并针对深层嵌套、字段间条件依赖以及跨调用衍生等难点进行正式定义与分析。

**💡 创新点**

创新点在于：①发现模型隐藏层状态在写出参数之前就蕴含可预测参数正确性的线性可分信号；②基于该信号提出两种方法——Probe‑Filtered Bootstrapped Training（PBT）用于自监督训练的伪标签筛选，Probe‑Guided Reranking（PGR）用于推理时候对候选参数进行重新排序；③构建了ParamBench基准，按参数嵌套深度、条件依赖、跨调用衍生等维度将实例划分为五个难度等级。

**🔧 技术方法**

主要技术包括：①在冻结的LLM中截取隐藏状态并训练单层逻辑回归线性探针；②利用探针在自监督训练中对生成的答案进行可信度过滤；③在推理阶段采样多条候选调用，使用探针得分进行候选级、字段级或字段集级的重排序；④对不同模型和数据集进行LoRA微调和实验评测。

**📊 数据集**

使用的数据集有：1）ParamBench（1022个实例，来自81个云网络API的真实执行轨迹及合成样例，按难度分级）；2）六个外部工具使用基准（NESTFUL、Seal‑Tools、xLAM、BFCL、API‑Bank、ComplexFuncBench），每个都被转换为每次调用的参数生成任务。

**📈 对比分析**

与基线方法（零样本、种子SFT、标准自监督、基于log‑prob或一致性选择的自监督）和四个开源大模型（Llama‑3.1‑8B、Gemma‑4‑12B、Ministral‑3‑8B、Qwen3‑8B）以及四个前沿模型（Claude Opus 4.7、GPT‑5.4、DeepSeek‑V4‑Pro、Qwen‑3.6‑Plus）进行比较。实验显示，在ParamBench和外部基准上，PBT+PGR将Qwen3‑8B的平均精确匹配率从约20%提升至约60%，并在所有七个基准上达到了或超过前沿模型的性能，尤其在最难的L4–L5级别提升尤为显著。

**⚠️ 局限性**

局限性包括：①ParamBench仅覆盖云网络API，难以直接推广到其他领域；②探针训练高度依赖特定模型、数据集和采样温度，跨域迁移性能不佳；③PBT和PGR都需要一定量的标注样本和大量无标注指令，实际部署中标签成本可能较高。

---

## 292. DiverseDiT++: Quantifying, Analyzing, and Promoting Representation Diversity in Diffusion Transformers

**arXiv ID:** 2608.03082 | [PDF](https://arxiv.org/pdf/2608.03082v1)

**作者:** Binglei Li `[一作]` (Fudan University), Hao Li `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对Diffusion Transformers的表示学习进行系统分析，提出Weighted Diversity Score量化表示多样性，并设计长残差+多样性损失框架提升模型生成质量。

**💡 创新点**

首次发现内部块间多样性与FID高度相关，提出无外部编码器的长残差与多样性损失方法，显著加速收敛并提升多域生成性能。

**🔧 技术方法**

基于SiT/DiT架构，使用CKA评估块间相似度，构造正交、互信息、离散度等多样性损失，并通过自适应权重与长残差实现输入多样化。

**📊 数据集**

使用ImageNet 256×256/512×512、蛋白质逆折叠PDB、GEOM‑Drug与QM9分子生成数据集。

**📈 对比分析**

与SiT、REPA、DispLoss、SRA等基线及现有SoTA模型对比；在ImageNet 256×256上80个epoch FID从4.0降至1.9，512×512 80个epoch FID从5.86降至2.20；蛋白质序列恢复率从37.5%提升至38.3%，分子稳定率提升至97.75%。

**⚠️ 局限性**

长残差增加约5.5%参数；自适应权重阈值经验设定；仅验证类条件图像生成，未覆盖文本‑图像或编辑任务；缺乏对多样性与鲁棒性理论关联的深入研究。

---

## 293. Preferred, Not Safer: Pairwise Preference Is a Poor Proxy for Clinical Safety

**arXiv ID:** 2608.02617 | [PDF](https://arxiv.org/pdf/2608.02617v1)

**作者:** Fay Elhassan `[一作]` (EPFL), Mary-Anne Hartley `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了临床专家对大型语言模型的盲对比偏好与安全性之间的关联，评估了MOOVE平台收集的偏好和多维评分数据。

**💡 创新点**

提出了安全调整后的偏好排行榜，并量化了偏好与安全性解耦的机制，揭示了特定专业领域的“禁区”。

**🔧 技术方法**

采用Bradley–Terry模型、逻辑回归特征分解、Rubric评分体系以及MOOVE平台的匿名对比与多维评分技术。

**📊 数据集**

使用MOOVE平台收集的376k+多维评分和26,804对比判定，覆盖13个LLM、736+临床医生、76个专业、约9,236条提示。

**📈 对比分析**

通过偏好-安全性对比、失败率分析、长度与行为特征分解，发现偏好排名与Harmlessness/Accuracy失败率存在负相关，安全调整后模型顺序显著改变。

**⚠️ 局限性**

数据依赖提示分布与评审者组成，缺乏下游临床结果，模型版本漂移，全球排行榜易被隐藏局部高风险，需分层报告。

---

## 294. BODHI: Do LLMs Branch Out and Discover Heterogeneous Inferences?

**arXiv ID:** 2608.02867 | [PDF](https://arxiv.org/pdf/2608.02867v1)

**作者:** Soumadeep Saha `[一作]` (Université de Toulouse), Nicholas Asher `[通讯]` (Université de Toulouse)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

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

## 295. Hunyuan3D-Buffalo 1.0: A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing

**arXiv ID:** 2608.02711 | [PDF](https://arxiv.org/pdf/2608.02711v1)

**作者:** Junliang Ye `[一作]` (Tencent Hunyuan), Chunchao Guo `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个统一的多模态框架Hunyuan3D-Buffalo 1.0，实现了3D理解、文本到3D生成、指令驱动的3D编辑以及文本定位的部件生成；

**💡 创新点**

通过构建87 M规模的3D多模态语料库并引入Nano3D‑v2高质量编辑数据生成管道，使得生成、编辑和理解三大任务在统一训练下实现互相促进的协同效应；

**🔧 技术方法**

采用Hunyuan3D‑VLM对三维结构与外观进行细粒度感知，配合3D‑DiT扩散生成模型，并通过MLP‑Connector、Q‑Former与VecSet等模块实现语义条件与三维生成的融合；Nano3D‑v2利用学习型编辑区域定位、体素级编辑、细粒度几何纹理优化等技术构建编辑数据；

**📊 数据集**

使用25 M的3D理解样本、50 M的文本‑3D对以及12 M的3D编辑对，其中理解数据来自Part‑X‑MLLM、ShapeLLM‑Omni等公开与内部数据，文本‑3D与编辑数据通过自动化的五阶段Pipeline与Nano3D‑v2生成；

**📈 对比分析**

在UniPart‑Bench、Edit3D‑Bench以及人类评测的文本到3D生成任务中，Hunyuan3D‑Buffalo 1.0分别在多模态理解、编辑（CD/F1均为最高）和文本到3D生成（人评偏好率>55%）方面均取得领先或最优表现；

**⚠️ 局限性**

仍存在单阶段高质量几何与纹理联合建模、编辑数据构造的一致性与可扩展性、以及对更大规模数据与更深度跨模态融合架构的进一步探索等局限。

---

## 296. Permission Denied: Policy-Graded Evaluation of Coding Agents in Hardened Environments

**arXiv ID:** 2608.02670 | [PDF](https://arxiv.org/pdf/2608.02670v1)

**作者:** Dotan Davidovich `[一作]` (Accomplish AI), Or Hiltch `[通讯]` (Accomplish AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估十二个模型-工具包在加入安全策略后的编码任务表现，提出Boundary-Bench插件实现对Terminal-Bench的安全硬化评测；

**💡 创新点**

首次将安全策略梯度作为评测维度，揭示不同策略对模型成功率与成本的非均匀影响，并提供任务可解性与参考解兼容性诊断；

**🔧 技术方法**

基于Linux原生访问控制的网络出口、文件系统、特权限制实现三层安全策略；对模型使用原生调用接口（Codex、Claude Code、Grok Build等）进行推理；

**📊 数据集**

使用Terminal-Bench 2.1共89个任务，覆盖编程与交互式任务；

**📈 对比分析**

通过对每个模型-工具包在三种策略下进行三次有效试验，计算成功率和平均成本，绘制成功–成本 Pareto 前沿；结果显示策略加重后所有模型均成功率下降、成本上升，但下降与上升幅度各异，部分模型失效主要因超时或错误解，成本增长多因重建工具链；

**⚠️ 局限性**

局限性包括：只评测单一Benchmark；评估单元为模型-工具包，未细化模型权重或工具包影响；只考虑三种策略阶梯，未覆盖所有企业场景；并未验证策略的实际安全效果。

---

## 297. Diameter-Free Distributed Frequency Control for Graph Coloring in the CONGEST Model

**arXiv ID:** 2608.02920 | [PDF](https://arxiv.org/pdf/2608.02920v1)

**作者:** Amit Nir `[一作]` (Weizmann Institute of Science), David Peleg `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了两种随机无全局协调的分布式图着色算法，能够在同步模型中控制颜色频率并保持着色合法性。

**💡 创新点**

创新点在于：①使用 Lazy Local Metropolis 采样实现两侧频率平衡且仅需约 2Δ 颜色；②设计了纯本地并行贪心算法，只需大于 Δ 的调色板即可得到上限频率约束，并提供多种调色板-频率折衷方案；③全部算法运行时间与网络直径无关，时间仅与 n、λ 成正比。

**🔧 技术方法**

主要技术包括：Dobrushin 依赖与自边界函数的集中不等式、随机游走的快速混合（Lazy Local Metropolis）、自适应伯努利和（Freedman/Doerr 形式）以及本地冲突检测与并行提议。

**📊 数据集**

论文仅给出理论分析，并未使用实际数据集；所有结论均为数学证明。

**📈 对比分析**

与之前的分布式着色方案相比，新算法消除了对网络直径的依赖，时间从 O(Δ(D+…)) 降到 O((λ+1)n)。在 2Δ 颜色下，频率偏差达到最优级别；在 Δ+O(Δ/ln n) 颜色下，单色类大小上限为 O((λ+1)(σ²n+n))，在高度图中远优于传统上限。

**⚠️ 局限性**

主要局限：①第二算法只能给出一侧（上限）频率约束，缺乏对下限的集中保证；②难以收紧提议预算与期望差距导致上限估计相对粗糙；③尚未解决在 Δ 接近 Δ+1 时仍能获得两侧平衡的可行性。

---

## 298. When Compression Scores Cannot Decide: Information Boundaries for Group-Robust LLM Pruning

**arXiv ID:** 2608.02940 | [PDF](https://arxiv.org/pdf/2608.02940v1)

**作者:** Andrew Zhang `[一作]` `[通讯]` (KTH Royal Institute of Technology), Andrew Zhang (KTH Royal Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究压缩统计作为信息接口，探讨如何通过局部指标预测完整压缩干预的终点性能，提出基于候选菜单的端点测量方法；

**💡 创新点**

创新点在于将压缩统计形式化为信息接口，阐明聚合与分解对极端组表现的影响，并引入“观察纤维”与“统一误差界”来评估接口可行性；

**🔧 技术方法**

主要技术包括凸锥分析、对角重建代理、路由器追踪、局部损失统计、分组暴露与恢复度评估，以及基于分布式评估的端点测量；

**📊 数据集**

实验使用 Llama-3.2‑3B‑Instruct、Qwen2.5‑3B、SmolLM3‑3B、Mistral‑7B‑Instruct 等密集模型，以及 OLMoE（3‑层，25% 专家删除）和多个 32‑条件校准网格，涉及常规、罕见代码、罕见知识、安全标签文本等数据集；

**📈 对比分析**

与传统稀疏化方法（Wanda、SparseGPT、HOPE 等）对比，分组分辨的对角指标在预测极端组损伤方面 Spearman ρ≈0.92，但无法决定同等预算内的排序；通过端点测量的有限菜单选择在 Llama、Qwen、SmolLM3 上实现 7–13% 的 worst‑group KL 降低，显示该策略可提升压缩后性能；

**⚠️ 局限性**

局限在于仅使用对角重建、固定预算和单层评估，未考虑跨层/全模型传输、非对角协方差、重复迭代收敛和非平衡组规模的影响，且在罕见知识/安全组上的效能依旧不稳定；

---

## 299. What the Detector Can See: Evaluating CPS Anomaly Detectors Independently of the Decision Rule

**arXiv ID:** 2608.02821 | [PDF](https://arxiv.org/pdf/2608.02821v1)

**作者:** Peiran Shi `[一作]` (University of North Carolina at Charlotte), Chenglong Fu `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种决策规则无关的CPS异常检测器评估框架，先通过归一化残差能量衡量Stage‑1残差表示是否能捕捉攻击，再用统一的能量尺度分析残差证据、训练-测试漂移、空间稠密度和阈值提取问题。

**💡 创新点**

创新点在于将检测器拆分为Stage‑1残差与Stage‑2报警两步，用残差能量与Kullback‑Leibler距离关联，构建统一的“信息‑能量”尺度，能够独立于阈值评估检测器表示质量并对失败原因进行诊断。

**🔧 技术方法**

技术包括归一化马氏距离计算残差能量、Ledoit–Wolf 估计协方差、KL 关联、ROC‑AUC、pAUC@1%、D_FULL、P_FULL、drift、compactness、F1@p99 等评价指标。

**📊 数据集**

使用了工业控制系统公开基准 SWaT、WADI、HAI 三个数据集，分别包含训练段和带标签的攻击测试段。

**📈 对比分析**

采用统一协议对五种检测器（GDN、FuSAGNet、TranAD、NSIBF、GeCo）在三测试床上评估Stage‑1能量，发现虽然 ROC‑AUC 在 SWaT 上差距不大，但 F1@p99 可相差十倍；跨测试床排名高度不稳定，框架揭示了弱证据、漂移、维度稀释等不同失败原因。

**⚠️ 局限性**

限制在于需获取或重构残差信号；KL 读取为矩阵匹配近似，可能低估非高斯残差分布；框架仅评估固定 Stage‑1 表示，未考虑自适应阈值或攻击者动态对抗；实验样本仅覆盖五种检测器。

---

## 300. Hypercubes, Hyperplanes, and Constraint-Induced Complexity Collapse in Atomic Concept Learning

**arXiv ID:** 2608.02930 | [PDF](https://arxiv.org/pdf/2608.02930v1)

**作者:** Irene Tsapara `[一作]` `[通讯]` (National University), Irene Tsapara (National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析高阶原子概念学习中基于基元实例的超立方体与超平面几何结构，证明除全对角线外的所有超平面在任意词深度下可归约为有限数量的本原等价类，展示结构约束引起的复杂度聚集与崩塌现象。

**💡 创新点**

首次将早期的最小化归约理论以几何视角统一解释，提出全对角线是唯一的异常区域，并给出递归的三元情况与学习理论意义。

**🔧 技术方法**

运用了有限模型理论（Ehrenfeucht–Fraïssé 论证）、最小归约技术、超图表示和组合计数论证等方法。

**📊 数据集**

论文为理论性工作，无实验数据集。

**📈 对比分析**

未进行实验比较，仅在理论上证明非对角超平面可归约为有限种类；对全对角线则无法给出统一上界。

**⚠️ 局限性**

局限于单一一元函数符号、有限深度、原子谓词，未处理多重函数符号或更高阶结构的情况，且未给出算法实现与实际性能评估。

---

## 301. Output-Aware Rotation for INT2 KV-Cache Quantization

**arXiv ID:** 2608.02691 | [PDF](https://arxiv.org/pdf/2608.02691v1)

**作者:** Vincent-Daniel Yun `[一作]` (University of Southern California), Sungjoo Yoo `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向INT2 KV缓存量化的输出感知旋转方法OptR，能在保持分页KV缓存格式的同时显著降低量化误差并提升推理精度。

**💡 创新点**

创新点在于：① 将KV量化误差直接映射到模型最终输出空间（post‑W_O注意力输出），而非仅对缓存重构误差做代理；② 引入通道级键中心化（key reparameterization）保持softmax不变；③ 通过对每个KV头学习正交校正矩阵，利用完整INT2注意力路径进行校准，解决传统固定或统计旋转的子最优问题。

**🔧 技术方法**

使用的技术包括：正交矩阵旋转与指数映射、INT2组量化、KL散度+误差正则化的输出感知损失、直通估计（straight‑through）梯度、Triton加速的键值旋转与重参数化内核、以及基于预热数据的离线校准。

**📊 数据集**

评估数据集包括：AIME24/25、GPQA‑Diamond、MBPP+、LiveCodeBench v6（推理任务）以及RULER‑NIAH长上下文检索任务，模型涵盖Qwen3‑4B‑Thinking‑2507、Qwen3‑8B、Phi4‑14B‑reasoning‑plus。

**📈 对比分析**

与BF16、TurboQuant（无混合精度）、QuaRot‑INT2和OSCAR进行对比。OptR在保持2.32 BPE的前提下，显著提升所有模型的推理准确率（例如Qwen3‑8B的AIME25从54.67%提升到66.00%，RULER‑NIAH在64K上下文下准确率提升约30%），且在解码延迟、吞吐量与预填时间上仅增加约2%或内存占用18 MiB，几乎无额外开销。

**⚠️ 局限性**

局限性：① 仅针对INT2低比特量化，尚未验证更低比特或不同模型架构；② 需要离线校准步骤，对动态推理场景的快速部署可能有限；③ 旋转学习涉及正交约束和指数运算，理论上可进一步优化；④ 结果主要在长上下文和推理精度上展示，生成质量的细粒度评估仍缺乏。

---

## 302. FakeI2V-Bench: Benchmarking the Applicability of Image-level Deepfake Detectors for Deepfake Video Detection

**arXiv ID:** 2608.03096 | [PDF](https://arxiv.org/pdf/2608.03096v1)

**作者:** Pei Li `[一作]` (Shandong University), Tianshuo Cong `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了包含 97,548 条多模态视频的 Deepfake 视频检测基准 FakeI2V‑Bench，并系统评估了 8 种视频级检测器与 12 种图像级检测器在该基准上的表现。

**💡 创新点**

提出了 IV‑Bridge 两阶段框架（Video‑Frame Fine‑Tuning + Multi‑Mode Aggregation），使原本仅针对图像训练的检测器能够直接并优于现有视频级检测器，显著提升跨域通用性与轻量化部署。

**🔧 技术方法**

采用视频帧微调（VFT）对图像级模型进行迁移学习，并通过六种帧‑>视频聚合模式（SMP, AVG, MAX, MIN, MED, VAR）构造多模态特征，再用轻量化随机森林实现最终视频级判别。

**📊 数据集**

使用四大公开数据集（Clever‑DF‑v2、DFD、GenVideo、GenVidBench）和自建的 FakeGenImage 图像数据集，涵盖多种生成模型（Sora、SVD、MuseV 等）与多样内容（人脸、自然场景等）。

**📈 对比分析**

在 FakeI2V‑Bench 上与视频级检测器对比，最优增强的图像级模型 RINE‑IV 达到 93.80% AUC、97.63% AP，明显高于最佳视频级检测器 FTCN（79.99% AUC、84.12% AP）；此外，IV‑Bridge 在推理速度、模型大小上均优于大多数视频级方法。

**⚠️ 局限性**

局限性：对极少量伪造帧（短期伪造）的鲁棒性仍不足，某些生成模型（如 Sora）仍导致检测性能下降；VFT 需要额外的视频训练数据，且随机森林聚合可能对不同数据分布产生偏差。

---

## 303. Reliability-Dependent Scaling Laws of Deterministic Identification over Binary Symmetric Channels

**arXiv ID:** 2608.03282 | [PDF](https://arxiv.org/pdf/2608.03282v1)

**作者:** Zhicheng Liu `[一作]` (Sichuan University), Zechun Hu `[通讯]` (Sichuan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在二进制对称信道（BSC）上，分析了确定性识别（DID）的第二阶渐进行为，给出了不同误差衰减模式（大偏差、适度偏差、中心极限定理）下的可达率与对偶下界，并通过哈密顿壳层几何和概率集中技术构造了代码。

**💡 创新点**

创新点在于首次将哈密顿壳层几何与中偏差、中心极限定理结合，得到针对 BSC 的精确二进制熵界，揭示可靠性要求对 DID 速率的具体影响；此外给出了可实现率与对偶下界的匹配，说明 BSC 在可靠性约束下保持线性标度。

**🔧 技术方法**

主要技术包括：GV 与哈密顿界的编码理论、伯努利和多项式和大偏差、适度偏差以及中心极限定理的概率集中估计、总变差距离与哈密顿距离的相互映射，以及对偶下界的最小距离约束。

**📊 数据集**

本研究为纯理论分析，无需实验数据集；所有结果均基于对 BSC 的概率模型和信息论极限推导。

**📈 对比分析**

在可达性上采用随机代码构造与集中定理得到率下界，在对偶下界上通过统计可分辨性与哈密顿距离得到率上界；两者在不同误差衰减区间均收敛至 BSC 的 DID 容量 1，且在指数衰减时保持非零速率损失。

**⚠️ 局限性**

主要局限是：可达率与对偶下界仍存在一定的常数与上界差距；研究仅针对 BSC，尚未推广到更一般的离散记忆无关通道；且对误差衰减的阶数选择与具体实现的细节尚未给出完整的构造算法。

---

## 304. TumorBoard: Evidence-Grounded Multi-Agent Decision Support for Longitudinal Neuro-Oncology

**arXiv ID:** 2608.03190 | [PDF](https://arxiv.org/pdf/2608.03190v1)

**作者:** Yantong Liu `[一作]`, Hyun-Ae Lee `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并评估了名为 TumorBoard 的多智能体决策支持系统，用于纵向脑肿瘤患者的诊疗决策。

**💡 创新点**

通过共享时间线状态、声称‑证据账本、对抗性批评器和安全门控器，实现结构化、可审计的多智能体协作，在相同推理预算下显著提升决策质量与安全性。

**🔧 技术方法**

结合检索增强的大型语言模型、Typed Communication 协议、图结构账本、对抗性批评和安全门控机制，构建多智能体框架。

**📊 数据集**

使用 360 例公开病例与专家编写的虚构病例（覆盖新诊、术后、随访、复发等）以及构造的隐藏测试集与扰动测试集。

**📈 对比分析**

在相同 token 预算下与单体提示、RAG、计划+批评、自由聊天多体、Typed Council 等进行配对 Bootstrap 比较；TumorBoard 在隐藏测试集上取得行动 F1 0.772、证据涵盖率 0.927、错误建议率仅 0.039，优于最强基线约 3 个百分点。

**⚠️ 局限性**

推理成本高（每例约 14k token，约 21.8 秒延迟）、依赖大量检索与专家协同，系统仍需人工监督与机构特定部署验证。

---

## 305. Control Barrier Functions via Minkowski Operations for Safe Navigation among Polytopes

**arXiv ID:** 2608.02886 | [PDF](https://arxiv.org/pdf/2608.02886v1)

**作者:** Yi-Hsuan Chen `[一作]` (University of Maryland), Michael Otte `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于Minkowski运算的多面体机器人与多面体障碍物的精确Signed Distance Function（SDF）构造，并将其与非光滑控制障碍函数（NCBF）结合，实现对动态、控制以及几何约束的安全导航；

**💡 创新点**

创新点包括：1）使用伴随的最小距离QP和穿透深度LP在配置空间中精确求解SDF；2）通过KKT条件和敏感性分析得到SDF的闭式梯度，统一安全、接触与穿透三种情况；3）引入几何-运动耦合的局部极小陷阱（GKC），揭示了非光滑梯度在非齐次动力学下的潜在死锁；

**🔧 技术方法**

核心技术包括：Minkowski差运算、凸优化（QP/Lp）、敏感性分析/可微优化、非光滑控制理论（NCBF）、多目标二次规划（CLF‑NCBF‑QP）与仿真评估；

**📊 数据集**

实验使用了合成的2D多边形机器人与障碍物场景（单积分器、单车模型、起始冲突、迷宫环境），未使用公开数据集；

**📈 对比分析**

与基线方法（LSE平滑SDF、MDF‑CBF等）比较显示：我们的框架在保持安全性的同时大幅降低轨迹保守性，控制输入更平滑、达标时间更短；在碰撞恢复和多障碍导航中更具鲁棒性；

**⚠️ 局限性**

局限性包括：1）相对于近似方法计算开销更大；2）NCBF框架本质上是局部的，仍可能出现死锁（GKC等局部极小点）；3）需要精细调参（如γ、ϵ）以避免不可行或产生新死锁；4）当前仅针对二维多面体，三维推广尚未完成。

---

## 306. Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures

**arXiv ID:** 2608.02645 | [PDF](https://arxiv.org/pdf/2608.02645v1)

**作者:** Isham Kalappurackal Mansoor `[一作]` (Old Dominion University), Pratip Rana `[通讯]` (Old Dominion University)

**通讯引用:** 755 | [OpenAlex ID](https://openalex.org/A5018203536)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对LLM代理在非原子工具调用中出现的重复执行和错误完成问题，提出并实现了一个轻量级的 verify‑before‑retry 包装器，在工具调用后先进行后置条件验证再决定是否重试，并使用幂等键保障重试安全。

**💡 创新点**

创新点在于将工具调用视为可验证的、可幂等的操作，首次将后置条件检查与重试策略结合，形成一个系统级的、无需修改语言模型的可靠性增强框架；同时提出了四类非原子失败模式的分类与针对性处理。

**🔧 技术方法**

技术手段包括：ReAct 结构的 LLM 代理、LangGraph 任务循环、Gemini Flash‑Lite LLM、轻量级后置条件验证器（直接状态读取、API轮询或 LLM 解释）、幂等键协议、模拟工具环境与故障注入、统计评估（任务成功率、重复动作率、置信区间、Cohen h）。

**📊 数据集**

使用了基于模拟器的实验环境，构造了两个多阶段任务（activate_customer、record_invoice）并在其中注入四类非原子失败（超时、延迟可见性、部分成功、冲突），没有使用公开真实数据集。

**📈 对比分析**

通过对比三种方法（仅重试、仅验证、验证+重试）以及不同故障级别，利用任务成功率和重复动作率评估性能。实验显示，verify‑before‑retry 在高故障下任务成功率从 64% 提升到 100%，重复动作率从 72% 降至 20%；在低/中等故障下同样实现了显著的可靠性提升。

**⚠️ 局限性**

局限性包括：仅在受控模拟环境下验证，缺乏真实 API 的多样性；后置条件验证器是手工设计，难以泛化；未涵盖身份认证、速率限制等额外错误；幂等键支持依赖于后端实现，若无此功能仍存在重试窗口；整体实验规模有限，无法覆盖更复杂工作流。

---

## 307. Bridging Online and Offline Handwriting via Differentiable Physical Rendering

**arXiv ID:** 2608.03198 | [PDF](https://arxiv.org/pdf/2608.03198v1)

**作者:** Seonmi Park `[一作]` (GIST), Hae-Gon Jeon `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个统一的在线-离线手写生成框架，能够把文字转化为可执行的笔迹轨迹，并通过可微分物理渲染器生成逼真的离线手写图像。

**💡 创新点**

创新点在于：①提出了仅包含六个核心参数的紧凑物理笔刷模型，将笔迹运动与像素级外观显式关联；②设计了可微分的笔刷渲染器，实现轨迹到图像的端到端梯度传播；③利用该渲染器合成在线-离线配对数据；④将零射扩散模型作为图像后处理器，实现无监督的细节增强。

**🔧 技术方法**

使用的技术包括：Transformer 结构的轨迹生成器；DINOv3+Transformer 的笔刷参数观测器；自定义的可微分笔刷渲染器；扩散模型（预训练的手写扩散器）做零射增强；机器人控制（将轨迹映射为笛卡尔运动）。

**📊 数据集**

使用的数据集有：IAM-OnDB、CASIA-OLHWDB（在线轨迹）以及 IAM、CVL（离线图像）。通过渲染器对在线数据合成了配对的离线图像，构成了 155,840 个词级样本用于训练；同时使用真实背景纹理库生成更逼真的离线图像。

**📈 对比分析**

评估时与现有单字符/多字符在线生成模型（如 SDT）以及离线生成模型（VATr++、DiffPen、One-DM、Emuru）进行对比。在线评测中，本文模型在多字 DTW 误差显著低于 SDT；离线评测中，结合渲染器的模型在 FID、BFID、HWD、CER 等指标上大幅提升（例如 FID 从 52–74 降至 11–12），证明在视觉真实性和结构一致性方面优于基线。

**⚠️ 局限性**

局限性：①笔刷参数虽然物理可解释，但未与真实物理单位完全对应，需要进一步校准以直接应用于实际笔具；②当前模型仅支持词级生成，无法一次生成完整句子或段落；③渲染器采用简化的物理模型，未覆盖完整的湿墨扩散、纸张吸收等细节。

---

## 308. AirKey: Multimodal Acoustic-Assisted WiFi Sensing for Zero-Training Robust PIN Inference

**arXiv ID:** 2608.03151 | [PDF](https://arxiv.org/pdf/2608.03151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 309. Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking

**arXiv ID:** 2608.03091 | [PDF](https://arxiv.org/pdf/2608.03091v1)

**作者:** Ethan Bito `[一作]` (RMIT University), Estrid He `[通讯]` (RMIT University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在候选项无序序列下的列表重排序稳定性；通过对等候选排列产生的排序视为偏好系统，评估对局部、全局和整体输出的一致性。

**💡 创新点**

提出三层一致性评估框架（PPI、GPI、LOC），从偏好级别捕捉候选顺序敏感性；区分位置曝光偏差与偏好一致性，揭示后者不随曝光均衡而改善。

**🔧 技术方法**

使用指令微调的解码器型 LLM（Llama‑3.2‑3B‑Instruct、Mistral‑7B‑Instruct‑v0.3、Qwen2.5‑7B‑Instruct）以及基于标记日志概率的排名提取；并对比三种位置偏差缓解策略（Bootstrapping、SGS、STELLA_LW）。

**📊 数据集**

MovieLens‑32M 与 Amazon Books 两大推荐数据集，采用 LightGCN 第一阶段候选检索。

**📈 对比分析**

与零射击重排序及三种偏差缓解方法进行对比；在 HR@5 与 nDCG@5 方面，STELLA_LW 最优；但在 PPI、GPI 与 LOC 等一致性指标上，SGS 与 Bootstrapping 更佳；显示效果与一致性不一致。

**⚠️ 局限性**

局限在于仅评估单一候选长度范围；方法对序列生成时的随机性敏感；SGS 需要顺序推理、STELLA_LW 计算开销大；未给出理论保证一致性提升的上界。

---

## 310. ICO: Enhancing Semantic-Shift Jailbreaks via Iterative Context Optimization

**arXiv ID:** 2608.03210 | [PDF](https://arxiv.org/pdf/2608.03210v1)

**作者:** Hujian Zhu `[一作]` (East China Normal University), Geguang Pu `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于迭代上下文优化（ICO）的黑盒语义移位越狱方法，使用占位符和上下文引导让模型重新解释原始有害词汇。

**💡 创新点**

创新点在于发现上下文的语义移位能力差异并通过迭代反馈与引导提示显著提升越狱效果，首次实现了高效且表面安全的攻击。

**🔧 技术方法**

采用占位符替换、辅助LLM生成与优化上下文、引导提示、判定器回馈以及迭代优化循环的技术栈。

**📊 数据集**

使用 HarmBench、AdvBench 与 StrongREJECT 三个公开恶意查询数据集进行实验验证。

**📈 对比分析**

与 Doublespeak、AutoDAN、PAIR、FlipAttack 等文本攻击以及 Visual Object/ Text Replacement、MM‑SafetyBench、HADES、FigStep 等多模态攻击基线对比，ICO 在所有目标模型上平均提升攻击成功率至 74.6%（文本）/ 63.2%（多模态），显著优于最强基线。

**⚠️ 局限性**

局限性包括对黑盒查询次数依赖较高、需在每次迭代获取模型响应、对部分模型仍有可观差异、且在极端安全防护下可能失效。

---

## 311. Neurosymbolic Reasoning with Incremental Knowledge for Sample Efficient Hierarchical Reinforcement Learning

**arXiv ID:** 2608.02993 | [PDF](https://arxiv.org/pdf/2608.02993v1)

**作者:** Subrat Prasad Panda `[一作]` (Nanyang Technological University), Arvind Easwaran `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种神经符号分层强化学习框架（Neurosymbolic HRL with Incremental Knowledge，InK），在高层使用可更新的符号模型进行规划，在低层使用目标条件神经网络控制，并设计了Belief World Tree Search（BWTS）算法以利用结构先验知识。

**💡 创新点**

创新点在于：1) 引入增量知识（InK）机制，让高层符号规划可随低层执行反馈动态更新；2) 结合动态 A* 与低层神经控制，实现从零经验即可进行目标导向的高效采样；3) 提出 BWTS，在未知环境中基于信念世界集合进行平均最优规划，克服传统 D* 仅考虑单一世界的缺陷，并且能利用结构先验。

**🔧 技术方法**

技术手段包括：分层强化学习（HRL）、符号规划（D*、A*）、目标条件神经网络（goal‑conditioned policy）、增量知识更新、Belief World Tree Search（基于 MCTS 思路的策略化回滚和探索奖金）、奖励塑造（reward shaping）以及贝叶斯动作-世界树搜索（BAMCP）对比。

**📊 数据集**

使用的实验数据集包括：1) RGL 环境下的点式迷宫（Four Rooms、Medium Maze、Hard Maze）以及 2) MuJoCo Ant‑Maze U‑Room。预训练低层策略均在无障碍迷宫中完成，随后在带障碍环境中评估。

**📈 对比分析**

方法比较：与非增量知识的 RGL（需要先学习完整世界图）相比，InK 在从零经验起始阶段采样效率提升 30–100 倍；在完成 75 个目标后的性能也保持更高；在 BWTS 与 D*、BAMCP 的对比实验中，BWTS 在有结构先验的信念世界集合上取得最低期望步骤和较优的运行时间；但 BWTS 在复杂或无先验时计算量显著增大。整体而言，InK 在样本效率上显著优于 RGL，BWTS 在具备结构先验时进一步提升。

**⚠️ 局限性**

局限性：1) BWTS 的搜索树规模随世界集合大小指数增长，导致在大规模或复杂先验下计算时间过长；2) 目前未能同时兼顾结构先验与概率先验（BAMCP 的优势），需进一步统一框架；3) 框架主要验证于导航任务，迁移到更复杂的操作或多目标任务仍需实验；4) 低层神经网络的预训练仍需要大量环境交互，且对动态变化环境的适应性有限。

---

## 312. Topological Simplification in Predictive Coding Networks

**arXiv ID:** 2608.02816 | [PDF](https://arxiv.org/pdf/2608.02816v1)

**作者:** Adam Shaw `[一作]` (University of Southern California), Alvin Jin `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了预测编码网络（PCN）中表示的拓扑演化，利用层级持久同调对不同层的拓扑特征进行定量分析。

**💡 创新点**

首次揭示了PCN在模型规模与激活函数作用下的拓扑简化规律，并与传统MLP对比，显示容量与双向递归动态共同决定简化时机。

**🔧 技术方法**

采用持久同调、Betti数衰减追踪、重建误差评估及种子引导的bootstrap对比等技术。

**📊 数据集**

使用了合成二维多环体数据集（九类、β0=9, β1=0）以及MNIST手写数字数据集。

**📈 对比分析**

通过训练多组不同宽度与激活函数的PCN，并与等深度MLP对比，发现PCN平均在比MLP晚3.6层后才完成拓扑简化；更大模型简化更晚，且简化越晚对应的重建误差越低。

**⚠️ 局限性**

实验仅在固定八层深度下进行，未探究更深网络或其他生成模型结构；持久同调在高维数据上的计算开销大，且仅使用有限数据集，限制了结果的普适性。

---

## 313. SeCo-SBIR: Semantically Consistent Prompt Learning for Zero-Shot Sketch-Based Image Retrieval

**arXiv ID:** 2608.03120 | [PDF](https://arxiv.org/pdf/2608.03120v1)

**作者:** Long Hoang Dang `[一作]` (Posts and Telecommunications Institute of Technology), Tu Minh Phuong `[通讯]` (Posts and Telecommunications Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的零样本草图图像检索方法SeCo‑SBIR，利用文本引导的多模态提示学习与扰动一致性约束实现对CLIP的高效适配

**💡 创新点**

创新点在于：①将可学习的提示向量先送入CLIP文本编码器，再通过可学习的耦合函数在每层将中间表示注入视觉编码器，实现在视觉侧的语义一致提示；②引入异步信息噪声一致性约束（InfoNCE），使训练好的模型与冻结的CLIP参考保持一致，抑制对已见类别的过拟合

**🔧 技术方法**

技术手段包括：文本引导的多模态提示学习、可学习耦合函数、轻量级适配器、异步扰动信息噪声一致性损失、三元组损失、NT‑Xent损失和分类损失的联合训练

**📊 数据集**

在Sketchy‑Ext、TU‑Berlin‑Ext和QuickDraw‑Ext三个公开SBIR基准上进行评估，其中Sketchy‑Ext被进一步划分为两种测试集

**📈 对比分析**

与现有CLIP‑基方法（如CLIP‑AT、SpLIP、Dr.CLIP等）以及ViT‑基方法相比，SeCo‑SBIR在所有三个评估场景（分类、泛化、跨数据集）均实现了显著提升，例如在TU‑Berlin‑Ext上mAP@all提升至78.7%，比SpLIP高约5.6%；在跨数据集设置中对QuickDraw‑Ext提升6.8%

**⚠️ 局限性**

局限性包括：仍依赖于大规模预训练的CLIP模型，适配过程需要额外的训练时间与内存；在极度抽象或噪声很大的草图数据上仍可能出现检索误差；目前仅在零样本检索任务上验证，未评估对其他跨模态检索任务的通用性

---

## 314. Aligned in Form, Not in Meaning: The Comprehension - Containment Decoupling of LLM Safety in Low-Resource Bangla Derogatory Speech

**arXiv ID:** 2608.02941 | [PDF](https://arxiv.org/pdf/2608.02941v1)

**作者:** Shadab Bin Habib `[一作]` (Islamic University of Technology), Adib Sakhawat `[通讯]` (Islamic University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

审计了五款前沿LLM在本土孟加拉贬义语（gali）上的安全表现，并提出语义理解与封锁不再耦合的“理解-封锁解耦”假设；

**💡 创新点**

首次系统揭示低资源语言中模型在理解上存在明显缺陷但在封锁上表现相同的解耦现象，并指出需要基于意义而非表面形式的安全封锁；

**🔧 技术方法**

采用人工标注的孟加拉贬义词集、六种评测协议（单轮理解、严重度校准、正则扰动、链式推理、多轮对话、专家角色）以及手工评注的Pass/Use/Refusal等指标；

**📊 数据集**

使用100条本土孟加拉贬义表达（与对应英语等价词）构成的评测集，包含53对单词以及扩展至501条的专家帧测试；

**📈 对比分析**

与英语基线对比，发现理解差距约7.9%但泄漏率相同；链式推理提升理解但泄漏率上升；专家角色显著降低拒绝率；整体表明安全机制主要依赖表面形式，无法保证低资源语言安全；

**⚠️ 局限性**

局限于孟加拉语言，样本量有限，评估模板和提示特定，未覆盖未来模型迭代及多语言验证，结论可能随模型更新或评测方式变化而调整。

---

## 315. FedGSA: Geometry-Consistent Subspace Aggregation for Differentially Private Federated LoRA

**arXiv ID:** 2608.03267 | [PDF](https://arxiv.org/pdf/2608.03267v1)

**作者:** Lele Zheng `[一作]` (Xidian University), Yulong Shen `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在差分隐私约束下，利用Grassmann流形上的子空间聚合实现Federated LoRA的聚合方法FedGSA

**💡 创新点**

创新点在于将低秩LoRA更新映射为基底不变的子空间，在服务器端执行几何一致的Grassmann聚合，从而消除因基底变换导致的聚合失配与二次噪声放大问题

**🔧 技术方法**

采用DP‑SGD单因素私有优化、SVD截断、投影矩阵表示、Grassmann流形上投影矩阵的加权平均及特征分解重构等技术

**📊 数据集**

在RoBERTa‑base与GPT‑2上分别使用GLUE四个任务（MNLI、SST‑2、QQP、QNLI）和E2E NLG Challenge数据集进行实验

**📈 对比分析**

与FedAvg、FFA‑LoRA、FedSVD、FedASK、LA‑LoRA、AS‑LoRA等六种基线进行比较，在ε=3、6的差分隐私预算下，FedGSA在所有任务上平均提升约2%–3%，在非IID环境下的鲁棒性更为显著

**⚠️ 局限性**

局限性包括：需要额外的子空间提取和投影计算，且对超参数（子空间维度r、投影加权方式等）敏感；在极端高异质性或极低隐私预算时仍可能出现收敛缓慢或性能下降

---

## 316. Multimodal Plant Root Phenotyping with Integration of 3D Skeleton Extraction and Language Analysis

**arXiv ID:** 2608.03109 | [PDF](https://arxiv.org/pdf/2608.03109v1)

**作者:** Jiakai Lin `[一作]` (Indiana University), Guoyu Lu `[通讯]` (Indiana University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种多模态机器人AI框架，将无监督的3D骨架提取与基于证据的语言推理相结合，实现可解释的根系表型分析。

**💡 创新点**

创新点在于引入方向-密度感知加权拉普拉斯收缩（W-LBC）实现高保真根骨架，和Evidence-First模板微调GPT以实现结构证据驱动的自然语言解释。

**🔧 技术方法**

使用PointTransformer、W-LBC骨架收缩、k‑NN+MST重建、GPT‑4o 证据驱动微调以及自生成的指令‑响应对。

**📊 数据集**

构建包含12种植物、共约1,200个根点云的自制数据集，涵盖从甜薯到苹果树等多种根系复杂度。

**📈 对比分析**

与Wen、Pc‑Skeletor、SmartTree等现有骨架方法和VQA基线相比，骨架提取的根计数/最长根长度准确率提升至0.79/0.84（<100根）及0.70/0.78（<500根），VQA多选和直接答复的整体准确率从约66%提升至70.4%及59.7%。

**⚠️ 局限性**

局限性包括对稀疏根系的捕捉仍有欠缺，数据集规模有限，且模型对极端形态或极低样本的泛化仍需提升。

---

## 317. Assessing Behavioral Validation in UI Component Test Suites Using Inferred Metamorphic Relations

**arXiv ID:** 2608.03337 | [PDF](https://arxiv.org/pdf/2608.03337v1)

**作者:** Yu Pei `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于推断的 metamorphic relation（MR）框架，评估 UI 组件测试套件的行为验证

**💡 创新点**

将 inferred MR 用作行为参考，区分行为触达（Touch）与验证（Cover），并展示 MR 覆盖与传统覆盖的互补性

**🔧 技术方法**

利用 UI 专用 MR 分类法、LLM 推断、确定性与语义对齐相结合的混合技术

**📊 数据集**

214 个组件，涵盖四个主流 UI 库（ant‑design、element‑plus、material‑ui、base‑ui）

**📈 对比分析**

与语句/分支覆盖比较，MR 覆盖率在 42.5–47.6% 之间，揭示大部分行为虽被触达却未被验证，补充了传统覆盖的不足

**⚠️ 局限性**

推断的 MR 空间不完整，依赖 LLM 的推理可能产生偏差，评估仅覆盖公开库，可能不适用于专有或特殊 UI 组件

---

## 318. Explainable AI for the EU Right to Explanation: A Systematic Review of the Law-XAI Translation Gap

**arXiv ID:** 2608.02699 | [PDF](https://arxiv.org/pdf/2608.02699v1)

**作者:** Benjamin Fresz `[一作]` (Fraunhofer Institute for Manufacturing Engineering and Automation), Marco F. Huber `[通讯]` (Fraunhofer Institute for Manufacturing Engineering and Automation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述后2024年发布的学术文献，评估其在欧盟《GDPR》《AI法案》《消费者信用指令》框架下对解释权的技术实现情况。

**💡 创新点**

提出了 Addressee/Purpose 框架，将解释的“形式”与“内容”分离并对应到法律条文，提出了四阶段操作蓝图，并梳理了六大开放研究问题。

**🔧 技术方法**

采用系统文献检索（Web of Science、Scopus）和 PRISMA 方法，使用 ASReview 工具进行标题/摘要筛选，双学科（法律+技术）评审流程。

**📊 数据集**

无实验数据集，唯一“数据”来源为被检索的 2643 篇文献，其中 57 篇全文可评，19 篇符合双重（法律+技术）标准。

**📈 对比分析**

比较方法为定性文献计量与内容分析；未进行实验性能评测，结论主要基于文献质量、法律解读准确性与方法覆盖度。

**⚠️ 局限性**

局限性包括：仅检索英语文献、2024年后出版物、Web of Science/Scopus 范围，单一技术审稿人导致潜在选择偏差；仅 19 篇双重符合标准，反映领域研究稀缺；未提供可操作的技术实现细节。

---

## 319. PDD-RRG: Posterior Diagnostic Decision for Study-level Radiology Report Generation

**arXiv ID:** 2608.03055 | [PDF](https://arxiv.org/pdf/2608.03055v1)

**作者:** Yang Yu `[一作]` (Soochow University), Yakang Dai `[通讯]` (Suzhou Institute of Biomedical Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文在自动放射报告生成任务中首次引入报告生成后决策阶段，提出 Posterior Diagnostic Decision（PDD-RRG）框架，用于融合多路径生成的诊断结果并修正报告。

**💡 创新点**

创新点在于将贝叶斯后验概率与观察特定阈值结合，形成基于似然比的聚合方法，显著提升诊断一致性和临床准确性，并实现无需模型再训练的通用后处理。

**🔧 技术方法**

使用技术包括：CheXbert 进行标签提取、贝叶斯后验概率与似然比估计、阈值校准、LLM（如 GPT‑4）进行报告修订，以及多视图输入生成多条报告候选。

**📊 数据集**

数据集为 MIMIC‑CXR，包含多视图胸片和纵向病历信息，评估基于 14 项临床观察的宏观与微观 F1 分数。

**📈 对比分析**

通过在 MAIRA‑2、LLM‑RG4 和 MLRG 三个基线上进行实验，PDD‑RRG 在宏观 F1、微观 F1 及 5 类疾病 F1 上均优于原始模型、基于一致性的选取、以及多数投票方法，提升幅度达 2–5% 以上。

**⚠️ 局限性**

局限性在于仅能纠正模型自身输出中的冲突，无法彻底消除严重幻觉或根本性诊断盲区，且对输入质量和模型先验仍有依赖。

---

## 320. The Agent Operating System (AOS): A Reference Operating Architecture for Distributed Agentic Systems

**arXiv ID:** 2608.03214 | [PDF](https://arxiv.org/pdf/2608.03214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 321. From Wearable Data to Personalized and Actionable Health Insights

**arXiv ID:** 2608.03251 | [PDF](https://arxiv.org/pdf/2608.03251v1)

**作者:** Esther Brown `[一作]` (Harvard University), Finale Doshi-Velez `[通讯]` (Harvard University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

结合可穿戴设备的连续生理数据与用户自我标注的生活事件，开发了一个交互式可视化工具，帮助用户反思并管理压力。

**💡 创新点**

创新点在于将实时生理数据与用户标注的干预事件叠加，支持多维度指标（HR、HRV、压力得分）以及自定义时间窗口的可视化，从而揭示干预效果与个体差异。

**🔧 技术方法**

使用Web框架实现交互可视化，时间序列处理（滑动窗口RMSSD计算）、基于BBI的HRV、热力图、柱状图等可视化技术。

**📊 数据集**

数据集为七名大学生在两周内佩戴Garmin Vívosmart 5收集的心率、HRV、BBI、呼吸率、睡眠得分以及用户日志共269条标注事件。

**📈 对比分析**

通过与随机时间戳基线对比、p值和效应量检验展示不同干预类别在HR、HRV、压力分数上的统计显著变化，结果表明社交、休息等干预在特定时间窗口内能显著降低压力。

**⚠️ 局限性**

局限性包括样本量小、仅限大学生、标注时延误和主观偏差、研究周期短、未使用正式人格量表等。

---

## 322. CIGTSurv: Clinical Information Guided Tri-modal Survival Prediction with Local Prototype Association and Global Feature Alignment

**arXiv ID:** 2608.03247 | [PDF](https://arxiv.org/pdf/2608.03247v1)

**作者:** Jing Dai `[一作]` (Dalian University of Technology), Hongming Xu `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

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

## 323. Tiny Enough to Break In: Agentic Remote Access Trojans Powered by Small Language Models

**arXiv ID:** 2608.03009 | [PDF](https://arxiv.org/pdf/2608.03009v1)

**作者:** Yuhan You `[一作]` (University of Virginia), Daniel Graham `[通讯]` (University of Virginia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室环境中，使用8B参数的Dolphin小型语言模型实现了Agentic Remote Access Trojan（agentic RAT），完成了从侦察到执行的 observe‑decide‑act 循环，展示了本地模型驱动的自主攻击可行性；

**💡 创新点**

创新点在于首次将小型语言模型嵌入到RAT的决策循环中，证明在不依赖云端或人工操作的情况下，局部模型即可实现完整的自主渗透流程；

**🔧 技术方法**

技术手段包括Docker Compose构建的隔离实验平台（Kali、Metasploitable2、LM Studio），基于ReAct框架的决策循环，工具调用与结果反馈机制，以及对模型输出的校验与执行；

**📊 数据集**

数据集主要是Metasploitable2公开漏洞服务（共55项），并通过Nmap生成的扫描结果作为模型输入；

**📈 对比分析**

对比方法采用100分制评分（任务完成度60%，效率20%，可靠性20%），实验平均得分12.72/100，成功率10.9%，显示模型在单一步骤推理上能成功，但多步骤推理与错误恢复能力不足；

**⚠️ 局限性**

局限性包括仅在单一易受攻击目标上测试、未部署完整的植入与持久化、未进行控制器对照实验、可能存在训练数据污染、缺乏长期持续运行与对抗性评估。

---

## 324. DiffImaginE: Imagine to Verify Entity Types with Diffusio

**arXiv ID:** 2608.03025 | [PDF](https://arxiv.org/pdf/2608.03025v1)

**作者:** Feng Zhang `[一作]` (Fuzhou University), Bin Chong `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对多模态命名实体识别（MNER）中的类型验证问题，提出一种基于条件潜在扩散模型的评分方法。

**💡 创新点**

创新点在于将传统单点想象-比较验证替换为条件扩散推断，用扩散残差作为类型似然估计；同时引入classifier‑free guidance、Min‑SNR权重、可学习时序聚合以及反向采样等技术，显著提升验证的鲁棒性与判别力。

**🔧 技术方法**

使用技术包括：RoBERTa+CLIP 跨模态编码、AdaLN Transformer 作为条件 denoiser、变分推断的 ELBO 兼容评分、classifier‑free guidance、Min‑SNR 加权、可学习时序权重、反向采样方差减小、温度参数化的 logits 训练。

**📊 数据集**

使用的数据集为 Twitter‑2015 与 Twitter‑2017 两个标准 MNER 语料库。

**📈 对比分析**

与公开基线和匹配的确定性 ImaginE 对比，DiffImaginE 在 Twitter‑2015 上严格 F1 提升 1.73 点（从 75.44% 到 77.17%），在 Twitter‑2017 上提升 0.72 点（从 87.72% 到 88.44%），并在两数据集上均取得最优 F1，差异通过配对显著性检验得到统计显著。

**⚠️ 局限性**

局限性包括：仅在单图短文本的 Twitter 语料上验证；扩散评分在推理时计算量相对较大；未探讨多图或视频场景，也未验证在开放词汇类型或其他结构化任务上的通用性。

---

## 325. Predicting Multilingual Classification and Translation Performance of LLMs with Cross-Lingual Alignment $\unicode{x2013}$ Is English Enough?

**arXiv ID:** 2608.03446 | [PDF](https://arxiv.org/pdf/2608.03446v1)

**作者:** Adnan Al Ali `[一作]` (Charles University), Alexander Fraser `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了27种跨语言对齐（cla）评分方法，探讨其对多语言LLM在分类、阅读理解和机器翻译任务中的预测能力，提出新的pmi翻译评估指标；

**💡 创新点**

提出基于pmi的目标语言无关翻译评估方法，并发现LLM在非英语翻译中倾向使用英语作为内部枢轴；

**🔧 技术方法**

使用句子嵌入（均值、pwm、last token、English prompt、few-shot）与对齐度量（cosine、xSIM、Dali、ANC、Eflomal）相结合，评估跨语言对齐；

**📊 数据集**

使用SIB-200、Belebele、Flores-200（1011条）和BOUQuET（验证集）等多语言数据集，覆盖44种语言；

**📈 对比分析**

与chrF、pmi指标相关性测试，发现pwm+ANC组合在系统级任务中表现最佳；cla-s源-英对齐与源-目标对齐的组合提升有限；在近似语言对中，源-目标对齐占主导；整体预测相关性较高但仍受限于特定任务；

**⚠️ 局限性**

局限包括：单一领域数据（Flores-200），少量近似语言实验，pmi的目标语言独立性未完全证明，few-shot嵌入依赖Wikidata描述，模型调优和数据覆盖不均，未覆盖所有语言种类。

---

## 326. Knowing the Form, Not the Function: Automatically Auditing Answer--Authority Decoupling in Legal Benchmarks

**arXiv ID:** 2608.02621 | [PDF](https://arxiv.org/pdf/2608.02621v1)

**作者:** Hsien-Jyh Liao `[一作]` `[通讯]` (Ministry of Justice), Hsien-Jyh Liao (Ministry of Justice)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对法律基准评测中答案准确性与法条引用（权威性）之间的关系，发现在常规未要求引用的情境下，模型常常给出正确答案但引用非金标准法条，或给出错误答案却引用金标准法条。为此作者提出了一套自动化的双重评测框架，能够同时评估答案正确性与权威性的一致性，并在四个不同类型的大语言模型上进行实验。

**💡 创新点**

创新点在于：①揭示答案正确性与权威性并非一一对应，存在双向解耦；②设计了一套可自动化、无人工审计的权威性匹配器；③提出了联合评测指标（A⁺G⁺、A⁺G⁻、A⁻G⁺、A⁻G⁻），并在现有法律基准中可直接实现；④通过检索实验和“可允许不引用”干预进一步说明答案与引用可独立移动。

**🔧 技术方法**

技术手段包括：规则基的法条匹配器（提取代码名、条文号、段落等）；对四种模型（Gemma‑4‑31B、LLaMA‑3.3‑70B、GPT‑4o‑mini、Qwen3.6 Flash）在不同实验条件（C1–C4、C2‑1）下收集回答；统计双向失配率、Hit@k 等指标；对比检索效果与答案正确性之间的关联。

**📊 数据集**

数据集主要为台湾法律资格考试（National Bar Examination）2022/2025年的多项选择题，共238道（民法118道，刑法120道），每道题均有验证过的正确答案和对应的法条；此外，还对中国大陆民法扩展进行初步验证。

**📈 对比分析**

实验对比四个模型，在C2条件下测得答案准确率与权威性命中率；发现民法区间为32.7–63.1%答案正确但未命中金标准法条，刑法区间为24.0–42.4%；反向错误答案命中金标准法条为15.2–21.7%。检索实验显示刑法的Hit@3显著高于民法，但双向失配仍普遍。C2‑1干预表明在保持答案准确率的前提下，引用率可大幅下降。

**⚠️ 局限性**

局限性包括：①仅评估行为表现，未揭示内部推理机制；②严格的金标准匹配可能将合法但非金标准条文误判为失配；③方法主要适用于编纂法条、可引用的任务，对普通法或开放式问答的适用性待验证；④实验模型覆盖面有限，未包含最新最前沿的推理型大模型，未知其是否能降低失配率。

---

## 327. JudgeArena: A Unified Framework for Reproducible LLM-Judge Evaluation

**arXiv ID:** 2608.02620 | [PDF](https://arxiv.org/pdf/2608.02620v1)

**作者:** Erlis Lushtaku `[一作]` (University of Freiburg), David Salinas `[通讯]` (ELLIS Institute Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个统一的开源框架，用于在多种LLM‑judge基准（AlpacaEval、Arena‑Hard、MT‑Bench、m‑Arena‑Hard等）上运行、记录和复现评估；同时提供可交换的评判模型、预调优的开源评判配置以及完整的元数据日志；还实现了通过LLM评判近似Elo评分和多语言元评估等功能。

**💡 创新点**

1) 将四大主流LLM‑judge基准统一接口化，打破代码碎片化；2) 设计并发布了针对开源评判模型的多目标调优配置，能够匹配或超越封闭模型；3) 全面记录元数据，保障评估的可复现性；4) 提供LLM评判近似Elo评分的低误差方法，减少人工标注成本。

**🔧 技术方法**

统一CLI接口、后端抽象（vLLM、llama.cpp、OpenRouter等）、多语言Prompt模板、链式推理与交换评估、概率化评分→BT模型、元数据JSON记录、自动化Elo近似等技术。

**📊 数据集**

AlpacaEval、Arena‑Hard（v0.1、v2.0）、MT‑Bench、m‑Arena‑Hard、ComparIA、LMArena等多语言评估集；以及多语言人类标注数据用于元评估。

**📈 对比分析**

在统一框架下对多种开源模型进行跨基准评估，结果显示预调优的Gemma‑4‑31B‑IT评判能够在大多数基准上匹配或超过封闭模型；Elo近似误差平均MAE≈0.2，表明LLM评判可有效替代人类标注；在案例研究中，Qwen3.5‑9B在所有基准上领先，证明框架的可比性和评估深度。

**⚠️ 局限性**

1) 受限于封闭模型的不可预测性，部分评估不可完全复现；2) 版本间数值不完全可比，需共享完整JSON；3) 评判配置可被误调，导致评估偏差；4) 评估数据可能被污染，尤其是静态基准；5) CUDA非确定性与云端服务更新可能导致微小差异。

---

## 328. Toward Certified Functional Safety for Industrial Humanoid Robots: The Fail-Passive Gap and a Feasibility Study

**arXiv ID:** 2608.02809 | [PDF](https://arxiv.org/pdf/2608.02809v1)

**作者:** Caiwu Ding `[一作]` (Siemens Corporation), Chengtao Wen `[通讯]` (Siemens Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并验证了一套基于已认证组件的外部安全监督链（Detection–Evaluation–Reaction），用于Unitree G1 EDU人形机器人在半封闭工作区执行拾取放置任务，并通过该链精准定位了人形机器人反应链中的“失效被动”安全缺口。

**💡 创新点**

①将已认证外部安全链作为测量工具精准定位人形机器人反应链中的失效被动缺口；②在机器人本体上托管软件定义自动化（SDA）控制器，将安全端点迁移到机器人侧，缩小缺口边界；③对主动安全状态进行专属分析，包括跌倒风险、单步停止约束、平衡策略残余风险以及ISO 13855分离距离。

**🔧 技术方法**

ISO 13849‑1/IEC 62061功能安全评估、ISO 13855保护距离计算、PROFIsafe无线安全通信、SICK deTec2光幕、Siemens E‑stop、Siemens Fail‑Safe PLC、SCALANCE W WLAN、Unitree G1 EDU机器人软PLC（IEC 61131‑3）等。

**📊 数据集**

通过实验演示获取的时序数据（光幕响应、PLC扫描、PROFIsafe网络延迟、SDA接收时间、停止时间/距离）和同步视频记录；未使用公开数据集。

**📈 对比分析**

与Siemens S7‑1500参考实现的PFHD/PL评估对比，外部链可达PL e/SIL 3；实验中通信丢失导致的失效安全停止在0.5–1.3 s内完成，符合预期worst‑case 1.1 s预算；停靠距离和停靠时间已测量但未给出完整可靠性数值。

**⚠️ 局限性**

机器人侧反应链未获得认证，缺乏PFHD/DC/CCF；缺乏安全评级的本体计算机；主动安全状态属于“失效主动”，不满足传统标准；实验仅覆盖单一安全区与单一场景；未完成对跌倒风险、网络丢包、误触发率等的完整性能评估。

---

## 329. Verifiable Memory: Learning Unified Memory Management with Local and Global Verifiers for Large Language Model Agents

**arXiv ID:** 2608.03137 | [PDF](https://arxiv.org/pdf/2608.03137v1)

**作者:** Xiaolong Sun `[一作]` (Sun Yat-Sen University), Liang Chen `[通讯]` (Sun Yat-Sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Verifiable Memory（VerMem）框架，统一管理 LTM、STM 和短期记忆，使用单一记忆操作策略通过七种原子操作完成信息存储、检索、过滤、摘要与历史上下文恢复。

**💡 创新点**

创新点：①将 LTM 与 STM 通过同一策略统一控制，同时保留各自独立状态；②结合局部与全局 verifiers 进行分层信用分配；③在三阶段 RL 训练中引入监督微调、单独 LTM/STM 训练以及完整任务协同训练，显著提升记忆决策质量。

**🔧 技术方法**

技术手段：监督微调（SFT）、三阶段强化学习（A、B、C 阶段）、局部与全局 verifier、GRPO 优化、七种原子记忆工具、结构化参数化命令、token‑级 PPO 剪裁与 KL 正则化。

**📊 数据集**

使用 HotpotQA 进行预训练，评估数据集包括 ALFWorld、SciWorld、PDDL、BabyAI、HotpotQA；在两种 LLM backbones（Qwen2.5‑7B‑Instruct 与 Qwen3‑4B‑Instruct）上进行跨任务评测。

**📈 对比分析**

与 LangMem、A‑Mem、Mem0、AgeMem 等现有记忆基线对比，VerMem 在所有五个基准和两种 backbone 上均取得最高或第二高的成功率/进度/J_judge 分数；在受限 token 预算下，其效率‑性能前沿明显优于其它方法。

**⚠️ 局限性**

局限性：每集后重置记忆，未检验跨会话或跨用户记忆持久性；仅在 HotpotQA 预训练，可能对其它模型族或领域迁移有限；verifier 依赖 Qwen‑Max，可能引入评估误差；训练成本高，且未验证对抗性或隐私冲突下的鲁棒性。

---

## 330. FinVerse: Financial Time-Series Benchmark

**arXiv ID:** 2608.03259 | [PDF](https://arxiv.org/pdf/2608.03259v1)

**作者:** Jaehoon Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了面向金融的 TimeVerse 领域特定基准 FinVerse，用于评估时间序列预测模型在金融决策场景下的实用性

**💡 创新点**

创新点包括：1) 根据金融时序的经济意义为每个序列分配决策导向的评估指标；2) 将评估拆分为点预测精度、横截面排名质量与组合回测三大视角；3) 引入 78 个指标（11 族）并采用基于排名的聚合方式，使不同尺度指标公平对比；4) 构建大规模金融时间序列宇宙并公开训练/评估拆分

**🔧 技术方法**

采用 Transformer 族基础模型（TimesFM、Chronos、Moirai、Timer、Toto 等）进行预训练与微调，结合自定义指标计算、排名聚合和多窗口组合回测等评估技术

**📊 数据集**

使用 116,897 条金融时间序列（171.1M 观测）构成完整数据集，其中 60,232 条经济含义显著的目标序列用于评估，66,134 条序列（100.3M 观测）用于模型预训练与微调，涵盖跨国、国内和个体资产，频率从日到年不等

**📈 对比分析**

对 43 公共预测模型在 FinVerse 上进行评估，并将结果与 GIFT‑Eval 进行对比。排名相关性仅为 0.40，表明通用误差指标与金融决策指标存在差距。Reverso Small 在整体排名中位居首位，Chronos‑2 Synth 在组合回测上最佳，TiRex‑1.1 在点预测精度上表现最佳；不同视角下模型表现不一致

**⚠️ 局限性**

局限性包括：1) 采用标准化的日历对齐，未充分考虑系列特定的报告日历与发布时点；2) 评估范围主要集中在美国市场，跨国和行业指数覆盖不足；3) 仅提供单一简单的等权长仓组合策略，缺乏多样化投资策略；4) 未来需扩展更丰富的决策任务与指标

---

## 331. Learning Context-Aware Motion Priors for Humanoid Control

**arXiv ID:** 2608.03234 | [PDF](https://arxiv.org/pdf/2608.03234v1)

**作者:** Yunyang Mo `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种上下文感知运动先验（CMP）框架，通过学习任务上下文与参考运动的兼容性，在训练过程中动态重新加权参考运动，从而指导 humanoid 控制学习。

**💡 创新点**

创新点在于：① 无需手工标记、技能划分或单独的技能发现阶段；② 结合高优势轨迹和演示正样本的对比学习，自动推断上下文-运动兼容性；③ 使用轻量级适配器将重新加权后的参考信息注入原始运动先验，保持先验的普适性。

**🔧 技术方法**

使用的技术包括：对比学习的相似度函数、优势加权采样、演示正样本正则化、参考运动重权重、上下文适配器以及与 Adversarial Motion Priors (AMP) 和 Score-Matching Motion Priors (SMP) 的无缝集成。

**📊 数据集**

实验采用 MimicKit 中的 humanoid 运动参考数据集（包含行走、奔跑、转弯等多样化动作），并在此基础上进行五个不同的控制任务测试。

**📈 对比分析**

通过在五个 humanoid 任务上与 AMP/SMP 基准对比，CMP 在任务回报和样本效率上均优于原始先验；在最不平衡的参考分布下，CMP 的表现保持稳定，显示出更好的鲁棒性。

**⚠️ 局限性**

局限性包括：受限于参考数据集的支持，无法产生未出现的行为；对优势估计的依赖使其对评论器误差和探索不足敏感；实验仅在仿真环境和结构化上下文中验证，未涉及高维感知或真实世界部署。

---

## 332. Coverage Matters: MarginMerge for Compressing Multi-Vector Visual Document Retrievers

**arXiv ID:** 2608.02969 | [PDF](https://arxiv.org/pdf/2608.02969v1)

**作者:** Ailar Mahdizadeh `[一作]` (University of British Columbia), Alireza Morsali `[通讯]` (Global Relay)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种称为MarginMerge的压缩方法，在离线阶段将多向量视觉文档检索模型的所有页面补丁向量压缩为少量合成代表向量，同时保持标准的MaxSim检索接口不变。

**💡 创新点**

创新点在于以“查询相关覆盖”作为压缩准则：选择能够覆盖不同查询可能匹配的多样区域的锚点，聚类后使用轻量网络学习每簇的代表向量，并通过排名边缘蒸馏确保压缩后文档排序与完整索引相近。

**🔧 技术方法**

技术包括：基于查询分布的锚点子模子（Greedy）选择、聚类分配、共享轻量网络合成代表向量、排名边缘蒸馏（Huber损失）以及全局有效秩分析。

**📊 数据集**

使用的预训练检索模型：ColQwen2.5（Qwen2.5‑VL‑3B）和ColPali（PaliGemma‑3B）。在这些模型上评估六个文档检索数据集：ArxivQA、DocVQA、InfoVQA、TAT DQA、TabFQuad和Flickr。

**📈 对比分析**

与多种压缩基线（Geometric Merging、Light‑ColPali、DSE单向量、随机/Top‑likelihood/Importance/Random/k‑center 选择等）以及不同保留比例（5%/10%/20%）对比。MarginMerge在5%/10%保留率下取得最高的nDCG@5，尤其在Flickr和DocVQA上显著提升，并将排名翻转率相较几何合并降低约41%。

**⚠️ 局限性**

主要局限包括：离线锚点选择耗时（约1.7秒/文档）；未对检索延迟或字节级存储做完整评估；压缩效果主要基于向量数量而非实际存储大小；对极低保留率下的鲁棒性仍有待验证。

---

## 333. LACE: Large Language Model Aided Multi-Agent Framework for Agile RISC-V Instruction Extension

**arXiv ID:** 2608.02915 | [PDF](https://arxiv.org/pdf/2608.02915v1)

**作者:** Pingqing Zheng `[一作]` (University of Minnesota, Twin Cities), Yang Katie Zhao `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

LACE框架通过多代理LLM流程，将自然语言描述的RISC‑V ISAX扩展自动转换为RTL并在四款开源核心中实现并验证；

**💡 创新点**

创新点在于提出两级IR分离指令语义与实现细节、检索增强的局部RTL编辑、以及步进式formal验证，完成从语义到硬件的端到端自动化；

**🔧 技术方法**

技术包括大语言模型（GPT‑4o）、多代理协作架构、RAG检索+ReAct提示、Verilator仿真、riscv‑formal验证、LangChain工具链；

**📊 数据集**

使用自定义ISAX基准（rol、ijmp、sbox、load_mul、sincos）并在PicoRV32、e203_hbird、ibex、cv32e40x四款核心上评测；

**📈 对比分析**

与非代理直接LLM生成对比，pass@1从约10%提升至72.8%；面积提升约10%，频率下降不超过10%，与SCAIE‑V人类专家实现的PPA相近；

**⚠️ 局限性**

局限在于对多周期、跨模块指令仍存在较高错误率；RVFI插桩依赖人工；评估仅基于28 nm后仿，未覆盖更深流水线或乱序核心。

---

## 334. SpreadMark: Robust Image Watermarking via Spread-Spectrum Embedding

**arXiv ID:** 2608.03165 | [PDF](https://arxiv.org/pdf/2608.03165v1)

**作者:** Wei Song `[一作]` (University of New South Wales), Jingling Xue `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

重新设计水印嵌入，采用稠密伪随机扩散频谱方式并结合神经网络后置水印架构；

**💡 创新点**

将每比特映射到全图稠密代码，配合匹配滤波解码与卷积补偿路径，并加入稀疏化感知对抗训练，显著提升在重建与稀疏化攻击下的鲁棒性；

**🔧 技术方法**

伪随机扩散频谱、匹配滤波、卷积解码器、稀疏化感知对抗训练、LPIPS视觉质量正则、芯片空间理论分析；

**📊 数据集**

COCO（训练4,500张，测试500张）与DIV2K（验证集500张）；

**📈 对比分析**

与九种基线方案（HiDDeN、MBRS、StegaStamp、SepMark、CIN、TrustMark、RivaGAN、WAM、DwtDctSvd）在标准失真、重建、latent sparsification与白盒对抗四类攻击下，用检测TPR@1%FPR衡量；在重建与稀疏化攻击中保持最高检测率（≈0.78/0.77），同时在PSNR 24–30 dB范围内保持高不可见性；

**⚠️ 局限性**

仅使用固定秘密码表，未评估码表泄露或自适应攻击；在高分辨率、尺度变换和代码同步失效时鲁棒性有限；白盒对抗攻击仅受能量限制，未保证解码器安全。

---

## 335. $S^3$: Improving Agent Safety through Multi-Stage Defense

**arXiv ID:** 2608.02683 | [PDF](https://arxiv.org/pdf/2608.02683v1)

**作者:** Zibo Xiao `[一作]` (Singapore Management University), Jun Sun `[通讯]` (Singapore Management University)

**通讯引用:** 23519 | [OpenAlex ID](https://openalex.org/A5100728816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Stage‑Specific Safety Skills 抽象，构建可自动转换为安全技能的流水线，并实现基于 Guard Agent 的多阶段防御框架（S3‑Framework）以及自研的 Multi‑Stage Risk Benchmark (MSRB)。

**💡 创新点**

创新点在于：①将异构安全设计统一为可重用、按阶段执行的安全技能；②提供自动化转换流程和社区驱动的安全技能库；③引入 Guard Agent 与分层触发机制，实现跨阶段协同防护和及时恢复；④通过 MSRB 对多阶段风险进行系统评估。

**🔧 技术方法**

使用 LLM agent 工作流、Guard Agent 协调安全技能、Skill.md 规范、分层触发机制、恢复模块、自动转换流水线；实现基于 DeepSeek‑V4‑Pro 的主、守护、转换模型；在 DeepAgent 框架上集成。

**📊 数据集**

主要数据集为自研 Multi‑Stage Risk Benchmark（675 例，覆盖六类风险），并参考 ATBench、AgentSecurityBench 等公开安全基准进行对比验证。

**📈 对比分析**

通过与单阶段安全方案（LC‑GuardRail、A‑MemGuard 等）、LlamaFirewall、SafeHarness 等对比，采用 ASR、RTR、TCR、TSR 等指标评估；实验表明 S3‑Framework 在安全成功率低、任务完成率高、恢复率高方面显著优于基线，且运行时开销可通过规则过滤和后恢复引导有效降低。

**⚠️ 局限性**

局限性：安全技能转换对复杂安全逻辑支持有限；安全规则依赖于预先定义，未知攻击场景覆盖率可能不足；安全技能库维护仍需人工投入；在不同 LLM 上的迁移性能尚未完全验证。

---

## 336. Frequency-Position-Fluid Antenna Array and Beamforming for Ultra-dense Connectivity in Terahertz Wireless Systems

**arXiv ID:** 2608.03289 | [PDF](https://arxiv.org/pdf/2608.03289v1)

**作者:** Heyin Shen `[一作]` (Shanghai Jiao Tong University), Jinhong Yuan `[通讯]` (University of New South Wales)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种动态频率-位置-流体天线(D-FPFA)架构，用可调频率局部振荡器、可移动子阵列和用户端流体天线实现超密集 THz 通信的频率与空间多路复用。

**💡 创新点**

创新点包括：
- 结合可调 LO 与可移动子阵列，突破 DAC/ADC 带宽限制；
- 两阶段频率分配：先按信道相关系数划分子波段，再利用近场宽带波束分裂实现多用户子载波分配；
- 通过天线选择与粒子群优化联合优化天线位置和混合波束成形，显著提升能效与总速率；
- 将用户分组问题建模为有向最小统治集，使用贪心算法解决。

**🔧 技术方法**

核心技术包括：可调 LO、可移动子阵列、流体天线、近场宽带波束分裂理论、最小统治集建模、动态天线选择、粒子群优化（PSO）混合波束成形。

**📊 数据集**

实验使用的主要参数为：总带宽 30 GHz、6 个 5 GHz 子波段、1024 架天线、16 个可移动子阵列、50 个单天线用户，基于 Monte‑Carlo 1000 次随机用户位置进行仿真；未使用公开数据集。

**📈 对比分析**

与多种基准（固定频率 AoSA、SPU、TTD 等）比较，D‑FPFA 在 40 dBm 传输功率下平均获得约 2.3 倍的总速率，并且在能效上达 95% 的 TTD 性能同时提升 2.8 倍能效；完全连通变体 FPFA 则在能效上最高，约 2.5 倍 AoSA 的能效。

**⚠️ 局限性**

局限性：
- 仅在理想完美 CSI 下评估，未考虑信道估计误差与频率/空间分配的鲁棒性；
- 只考虑 LoS 近场模型，未考虑多径、遮挡与用户移动的影响；
- 需要大量的训练与反馈开销，实际实现中可调 LO 与可移动子阵列的硬件实现与功耗仍是挑战。

---

## 337. Configurable and Hierarchical Allreduce

**arXiv ID:** 2608.02884 | [PDF](https://arxiv.org/pdf/2608.02884v1)

**作者:** Valentino Guerrini `[一作]` (University of Illinois Chicago), Sidharth Kumar `[通讯]` (University of Illinois Chicago)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可配置的分层 Allreduce（CHiArA）算法，利用逻辑批-通道拓扑与分阶段调度，实现对硬件层级的显式编码；

**💡 创新点**

创新点包括：1）可调批大小与独立的 intra-batch radix，精细控制局部与全局通信；2）旋转根通道（LaneOp）实现分布式跨域通信，避免单点瓶颈；3）半组合 Rabenseifner 风格 Allreduce，保持 lane-aligned 中间布局，消除 Reduce-Scatter 与 Allgather 边界上的冗余重组；

**🔧 技术方法**

采用递归交换（recursive exchange）与可调 radix、批-通道逻辑映射、分阶段（stage）处理、LaneOp 原语、以及半组合（semi-composed）执行方案；

**📊 数据集**

在领导阶系统 Polaris、Aurora、Fugaku 上进行微基准测试；并在基于 MPI+OpenMP 的并行 K‑means（使用合成数据：每 rank 256 个点、维度 D=K=256/192）进行端到端实验；

**📈 对比分析**

与主流 MPI 库（MPICH 默认 Allreduce）进行对比，采用相同进程数、消息大小和参数选择；实验显示在 Polaris 上最快可达 1.94×，在 Aurora 与 Fugaku 上分别可达 13.43×和13.48×；在 K‑means 迭代中，整体迭代时间提升至 2.2×；

**⚠️ 局限性**

限制在于：1）性能高度依赖批大小与 radix 的手工调优；2）对极大消息尺寸（带宽主导）提升有限；3）需要逻辑批-通道映射，若节点分配不连贯可能受限；4）实验仅覆盖 CPU‑resident MPI，未验证 GPU 或异构环境；

---

## 338. GeoID-PINN: Identifiability-Aware Regional Epidemic Inference with Geographic Coupling

**arXiv ID:** 2608.02633 | [PDF](https://arxiv.org/pdf/2608.02633v1)

**作者:** Weixiong Hua `[一作]` (University of Michigan), Fan Bu `[通讯]` (University of Michigan)

**通讯引用:** 1008 | [OpenAlex ID](https://openalex.org/A5100713584)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将行随机矩阵嵌入SIRD力学并用物理信息神经网络（PINN）估计区域间感染源组合的模型，旨在利用常规监测数据进行多地区疫情预测；

**💡 创新点**

通过在感染力中加入可正则化的行随机源组合矩阵，并在神经网络训练中同时约束非负状态、保持人口总量以及使用空间先验进行正则化，显著提升了预测精度，同时揭示了矩阵估计的识别难题；

**🔧 技术方法**

使用物理信息神经网络（PINN）、行随机化的源组合矩阵、空间先验正则化、深度神经网络状态约束、Adam优化、以及多种损失（观测拟合、物理残差、初始条件、先验）组合；

**📊 数据集**

合成四地区仿真数据和美国路易斯安那州64个县的COVID‑19周报数据；

**📈 对比分析**

在合成数据上通过已知真值验证矩阵恢复，在真实数据上与自回归负二项基线和无网络PINN进行对比，Forecast‑Trained Geo‑PINN将MSE从32,957降至11,468（降幅65.2%），MAE从70.60降至57.73（降幅18.2%），但负对数似然略高；在受控邻接网络对比中，邻接网络将MSE降低6.85%，MAE降低3.1%；

**⚠️ 局限性**

识别性有限：矩阵估计对空间先验敏感，观测过程（报告率、延迟）和参数约束同样影响估计；仅有常规监测数据不足以确定唯一的跨区域传播网络；实验缺乏多起随机初始、边缘灵敏度分析以及未来外部输入预测，且使用的是汇总县级周数据，无法支持个体层面的推断。

---

## 339. In-Context Collapse in Vision-Language Models and How to Mitigate it?

**arXiv ID:** 2608.02830 | [PDF](https://arxiv.org/pdf/2608.02830v1)

**作者:** Mohammad Rostami `[一作]` `[通讯]` (Amazon Generative AI Innovation Center), Mohammad Rostami (Amazon Generative AI Innovation Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统研究了视觉语言模型（VLM）的多示例上下文学习（many-shot ICL）中出现的“上下文崩塌”现象——随着示例数增多，模型准确率下降甚至跌破随机水平；并提出了一种可移植的修复框架CircA，主要通过在视觉–语言接口（connector + 早期/中期层）添加低秩适配器（疫苗）来恢复对示例的有效利用；

**💡 创新点**

1) 发现并量化了上下文崩塌的存在与多级影响；2) 将“鲁棒性”与“学习能力”分离，揭示两者独立的三种运作模式；3) 通过参数匹配的Lesion‑and‑Rescue定位崩塌原因至视觉‑语言集成路径；4) 提出CircA三阶段（疫苗、门控、注入）可在离线或在线环境下修复或缓解崩塌；

**🔧 技术方法**

低秩适配器（LoRA）在connector、早期/中期Transformer层的微调；损失函数为对remap标签的交叉熵；使用copy‑rate（复制率）作为无标签诊断指标；对比不同层次（early/mid/late）以及读出层的适配器效果；

**📊 数据集**

3个4‑way分类任务（Synthetic Shapes、CIFAR‑4、Fashion‑4）以及开放式VQA（VQAv2、TextVQA、ScienceQA）作为评估基准；模型覆盖0.5B–11B参数的多种connector/backbone（MLP‑projector、pixel‑shuffle、cross‑attention）以及前沿API模型（Claude Sonnet 4.5、Nova等）

**📈 对比分析**

对照标准ICL、无示例（zero‑shot）以及无修复的多示例情况；在所有评测任务中，未修复模型多示例时准确率下降0.1–0.8；CircA疫苗后，remap任务在未见过的任务族上从接近随机提升至0.6–0.9；在VQA任务中，崩塌模型在多示例时从0.7→0.12下降，但修复后可恢复至0.7以上；

**⚠️ 局限性**

仅在视觉‑语言接口集成层的改动有效，读出层不修复甚至加剧崩塌；适配器容量有限，过大可能导致其他干扰；对极大模型或不同架构的泛化仍需验证；门控与注入仅在已修复或鲁棒模型上有效，无法在崩塌模型上单独使用；

---

## 340. SPADE: An Input-Adaptive Sparse Attention Engine for Fast Video Diffusion Models Inference

**arXiv ID:** 2608.03335 | [PDF](https://arxiv.org/pdf/2608.03335v1)

**作者:** Shanghao Liu `[一作]` (Beihang University), Hailong Yang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个训练无关、算法与系统协同的稀疏注意力引擎 SPADE，用于加速视频扩散 Transformer（vDiT）的推理，统一了静态、半静态和动态稀疏策略。

**💡 创新点**

创新点在于提出了 vDiT-SSR 统一抽象，利用在线 Sum of Intra-block Cosine Similarities（SICS）实现头级自适应稀疏方案生成，并在执行层面结合 FlashBlock‑sparse、头级聚合与硬件友好优化，实现高稀疏率、低开销和质量保留的兼顾。

**🔧 技术方法**

使用了 3D 块化、SICS 在线聚合评估、head‑wise sparsity policy、Flash Attention、block‑sparse GPU kernel、头级分组、代码生成等技术。

**📊 数据集**

在 VBench‑2.0 文本到视频和图像到视频数据集上，结合 Hunyuan‑Video、Wan 2.1/2.2 的 13B/14B vDiT 模型（720p、61/125 帧）进行评估。

**📈 对比分析**

与全注意力 FlashAttention‑3、Semi‑static Sparse‑VideoGen、Dynamic Sparge Attention、X‑Attention 等基线对比，SPADE 在保持或提升 SSIM/PSNR/LPIPS 的同时，稀疏率最高（≈85%），Attention 速度提升 2.26×–3.44×，端到端加速 1.32×–1.49×（Turbo 版可达 1.80×）。

**⚠️ 局限性**

仍受限于 GPU 硬件特性的优化，适用范围主要为 vDiT 结构；动态稀疏模式下的模式搜索开销虽被压缩但仍存在；在极端稀疏或复杂场景下可能导致质量轻微退化；实现复杂度较高，需要手动配置 policy。

---

## 341. Double Down on Defense: Strengthening Deep Perceptual Hashes against Evasion Attacks without Retraining

**arXiv ID:** 2608.03101 | [PDF](https://arxiv.org/pdf/2608.03101v1)

**作者:** Bangjie Sun `[一作]` (National University of Singapore), Jun Han `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无模型改造的插件式防御，结合发布时的无感扰动硬化与查询时的随机平滑，以提升现有深度感知哈希的抗攻击能力。

**💡 创新点**

创新点在于：①以无感扰动硬化提前提升参考图像在匹配空间中的安全边距；②通过匹配时随机平滑将单次脆弱比较转化为概率判决，实现可证的 ℝ_2 稳健性；③两种机制互补，既提升实验攻击抵抗，又提供正式的鲁棒性保证。

**🔧 技术方法**

主要技术包括：无感扰动优化（在发布前对参考图像做微小不可见扰动），匹配时随机高斯噪声采样与蒙特卡罗估计，以及基于随机平滑的 ℝ_2 证明与置信下界计算。

**📊 数据集**

实验使用了 ImageNet、MS‑COCO 及 Stable‑Diffusion 三大数据集，覆盖自然图像与生成图像场景。

**📈 对比分析**

与原始哈希及仅使用随机平滑的对照相比，本文方法将白盒攻击成功率从 98.6% 降至 11.8%（极大幅度降低），黑盒攻击从 20.6% 降至 1.3%；同时在三种数据集上，平均 ℝ_2 证明半径提升至约 0.30，且对常见压缩、裁剪、旋转等变换的误判率保持在 10% 以下；碰撞率几乎不变，显示不显著损害原有实用性。

**⚠️ 局限性**

局限性包括：①随机平滑导致的推理延迟和计算成本较高；②无感扰动对视觉质量的影响因哈希模型差异显著，需在安全与美观间权衡；③证明仅覆盖加性 ℝ_2 攻击，对结构化变换不具备正式保证；④对已有碰撞问题（如 C‑PDQ）无修复效果；⑤在极大攻击预算下仍存在一定成功率，需进一步强化。

---

## 342. Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Hardware Operators

**arXiv ID:** 2608.02712 | [PDF](https://arxiv.org/pdf/2608.02712v1)

**作者:** Yansong Sun `[一作]` (City University of Hong Kong), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 40760 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在低资源硬件加速器AscendC上，提出将近似失败的生成内核进行自动调试，而非重新生成的“debug‑over‑regenerate”范式。

**💡 创新点**

创新点在于构建引擎主导的调试管道，集成知识库检索、诊断仪表、反作弊检测、全覆盖评估以及收敛守卫五大机制，确保调试安全、可追溯且高效。

**🔧 技术方法**

技术实现包括：LLM驱动的修复循环、结构化知识库自动摄取、动态调试仪表、引擎所有权的验证与完整性检查，以及基于状态机的硬性与软性收敛策略。

**📊 数据集**

使用AscendC的NPUKernelBench基准中27个冻结的近似失败内核作为实验数据集。

**📈 对比分析**

与CANNBot的三次重生成和官方precision‑debug模式对比，Debug Pass@1达66.7%（vs. 40.7%和44.4%），并且每成功一次的Token成本降低约92.8%，体现出更高的修复率与算子成本优势。

**⚠️ 局限性**

局限性包括：仅在AscendC上验证，缺乏跨平台泛化；硬性收敛阈值需要进一步调优；实验规模相对有限，尤其高级复杂度任务样本不足；以及对完整覆盖评估的依赖使得缺失全覆盖测试时效果不明。

---

## 343. Route-Align-Verify for Functional Correctness in Code Generation

**arXiv ID:** 2608.03341 | [PDF](https://arxiv.org/pdf/2608.03341v1)

**作者:** Erxue Zhou `[一作]` (East China Normal University), Aofan Liu `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RAV（Route‑Align‑Verify）框架，利用任务感知的提示路由、对齐 LoRA 适配和执行验证三阶段协同来提升固定基础模型的代码生成功能正确率。

**💡 创新点**

创新点在于将提示设计、模型适配与候选输出选择三者统一为可插拔的模块，并通过对齐 LoRA 减少 fine‑tune 与推理提示的分布差距，最终实现显著功能正确率提升。

**🔧 技术方法**

技术包括：任务感知提示路由（根据关键词分派不同提示模板）、LoRA 参数高效适配（配合提示对齐训练）、多样本生成与基于公开测试的轻量级执行验证。

**📊 数据集**

数据集：MBPP benchmark，分别在 Sanitized 与 Full 两种设置下进行评估。

**📈 对比分析**

通过与基线（无路由、无对齐、无验证）、路由+验证、对齐+验证、路由+对齐等组合的对照实验，Full RAV 在 MBPP Sanitized 上达到 Pass@1 0.8911（比基线提升 6.35%），在 MBPP Full 上达到 0.8520（比基线提升 9.92%）。

**⚠️ 局限性**

局限性：仅在 MBPP 任务上验证，未完成完整因子消融实验；Sanitized 版统计显著性较弱；缺乏更广泛的基准和更强的验证信号。

---

## 344. AI Agent Economics: Can Autonomous Economic Behavior Emerge among AI Agents under Minimal External Conditions?

**arXiv ID:** 2608.03076 | [PDF](https://arxiv.org/pdf/2608.03076v1)

**作者:** Lingyun Zhang `[一作]` (University of Tokyo), Shang Shang `[通讯]` (Beijing Chaitin Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建可执行的AI代理经济实验框架，探究在没有预设社会或经济策略的情况下，LLM代理是否能自发形成生产、转移、分配等经济关系。

**💡 创新点**

创新点在于：①引入“可执行机制”而非角色标签来触发经济组织；②通过可追踪事件链识别经济关系；③设计了生产边界、可执行分配权、语义框架与能量符号化等多维对照实验。

**🔧 技术方法**

主要技术包括大语言模型（GPT‑4 与 DeepSeek）、可执行任务调用、基于事件的审计扫描、Gini 系数与 AUC 评估、以及自定义可追踪日志与验证器。

**📊 数据集**

数据集使用的是 Google CTF 公开挑战中的三道字节保全安全任务，实验共 24 个独立的 6‑代理世界，每天提供 3 任务，持续 25 天。

**📈 对比分析**

比较方法：对每个模型族分别设定对照组与实验组，计算经济差异 AUC（结合分配与能量 Gini），采用引导抽样、随机化检验与 leave‑one‑world‑out 评估；实验显示可执行分配权显著提升经济差异并减少分配失败和排斥时间，语义框架对结果影响有限。

**⚠️ 局限性**

局限性包括：实验规模有限（仅 6 代理，单一任务域）；仅测试 GPT‑4 与 DeepSeek 两款模型；评估指标聚焦于分配与能量差异，未涵盖更细粒度的机构与公平性指标；实验设置缺乏长期演化与多样化的经济情景。

---

## 345. Evaluation Blindness: How Silent Measurement Failures Corrupt AI Systems from Training to Deployment

**arXiv ID:** 2608.02786 | [PDF](https://arxiv.org/pdf/2608.02786v1)

**作者:** Priyanka Bajaj `[一作]` `[通讯]`, Priyanka Bajaj

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“评估盲点（evaluation blindness）”这一统一的结构性属性，并将其用于解释训练阶段和部署阶段的 AI 系统故障，构建了包含六类的系统级故障分类法、训练‑时评估盲点的四个案例研究、基于风险的失效预算框架，并整理了 50 起真实与合成的生产事故数据集。

**💡 创新点**

创新点包括：① 统一的评估盲点形式化定义；② 将训练‑时与部署‑时测量失效归为同一问题；③ 六类系统级故障分类法（模型漂移、基础设施、集成、评估、合规/安全、运营）及其检测属性；④ 训练‑时评估盲点四个实证案例；⑤ 依据风险类别的失效预算公式；⑥ 提供公开数据集、规则式分类器和预算计算器。

**🔧 技术方法**

主要技术手段有：形式化定义与可检测性谓词；案例研究中对 RLHF、GRPO、评估污染等问题的实现与验证；规则式文本分类器（基于关键词加权匹配）；统计分析（MTTD、失效预算利用率）和实验数据（50 起事故标签）。

**📊 数据集**

使用了包含 36 起公开可验证事故（法院文件、监管备案、公司事故报告等）和 14 起合成案例的事故数据集，总计 50 起样本；数据集已公开在 GitHub 上；还使用了公开的 TRL 代码库、RLHF 评价数据、Benchmarks 等。

**📈 对比分析**

与传统单一基准评估或仅关注模型性能的方法相比，该框架能够检测到 53% 的“无声”事故，并在规则式分类器上实现 98%‑100% 的精度与召回率；失效预算框架使得对不同风险级别的请求量实现可量化的容错阈值。性能上，MTTD 由分钟级（C2）到数月级（C4）差异巨大，表明现有监控不足。

**⚠️ 局限性**

局限性包括：数据集偏向公开报告的高严重性事件，低严重性事故缺失；标注仅由单位作者完成，虽然与独立评估者达成 100% 一致，但仍可能存在偏差；合成案例不具备真实发生的统计学意义；缺乏跨领域、跨行业的多样化事故样本；规则式分类器在处理模糊边界时仍需人工干预。

---

## 346. CUADebug: Diagnosing and Repairing Computer-Use Agent Failures

**arXiv ID:** 2608.02643 | [PDF](https://arxiv.org/pdf/2608.02643v1)

**作者:** Weijia Zhang `[一作]` (University of Illinois Urbana Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对计算机使用代理（CUA）的根因诊断问题，提出了专门的错误分类体系、基于人类标注的OSWorld失败轨迹基准（CUADebugBench），以及利用ReAct循环和多模态检查工具的工具增强根因诊断（RCA）代理。

**💡 创新点**

创新点主要包括：① 为CUA提供了五大类的可解释错误分类，并将其与人类标注结合形成基准；② 设计了结合截图、动作轨迹和结构化提交的工具增强RCA代理，能够主动检视步骤并生成可执行的修复方案；③ 将诊断结果直接用于重执行（re‑rollout），实现诊断可操作化。

**🔧 技术方法**

使用的技术包括：ReAct循环、工具调用（多模态步骤检查工具与结构化提交工具）、截图对比、记忆检索、连续重执行框架以及大模型（Claude 4.5 Sonnet、Gemini 2.5 Pro、Qwen 3.5）的文本与视觉推理。

**📊 数据集**

采用的数据集是OSWorld的失败轨迹，构建了204条人类标注的根因样本，涵盖了Claude 4.5 Sonnet、Gemini 2.5 Pro和Qwen 3.5三种代理生成的轨迹。

**📈 对比分析**

评估方法包括：与一次性提示基线对比的错误分类准确率（L1/L2）、步骤定位精度（Step Exact/±2）以及联合精度（Tag+Step）；在单次和持续重执行实验中，RCA诊断显著提升任务完成率（从13.9%提升至约30%，持续执行从12.2%提升至25.9%），接近人类标注oracle的表现。

**⚠️ 局限性**

主要局限：样本分布不均，跨代理评估样本量小；诊断与重执行的匹配仍存在难度，特别是细粒度子类与确切根因步骤的同步；基准仅覆盖OSWorld，缺乏在其他CUI环境的验证。

---

## 347. Field Aware Agent Skill Retrieval

**arXiv ID:** 2608.02880 | [PDF](https://arxiv.org/pdf/2608.02880v1)

**作者:** Paimon Goulart `[一作]` (University of California Riverside), Liangjie Hong `[通讯]` (Nokia)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在终身学习智能体的技能检索任务中，是否保留技能的多字段结构（名称、描述、正文）能够提升检索效果。

**💡 创新点**

创新点在于：①将技能拆成独立字段并分别编码，构建三阶张量化表示；②在稀疏（TF‑IDF/BM25F）与稠密（Qwen3‑Embedding）视图下分别计算字段相似度；③用统一权重或小型MLP对字段级得分进行融合，形成字段感知的检索模型。

**🔧 技术方法**

使用的技术包括：TF‑IDF + BM25F（稀疏检索）、Qwen3‑Embedding‑0.6B（稠密检索）、张量化技能表示、均匀加权与MLP融合（混合检索），以及标准评测指标（Hit@1, Recall@5/10, nDCG@5/10, MRR）。

**📊 数据集**

实验数据集：SkillRet（6,660 技能，4,997 查询）和 SRA‑Bench（26,262 技能，5,400 查询，六个子域）。

**📈 对比分析**

对比方法包括：BM25、BM25F、拼接 TF‑IDF / Qwen / 混合检索、字段分离均匀加权、字段分离+MLP。结果显示：字段分离+MLP 在 SkillRet Recall@10 达到 77.95、SRA‑Bench Recall@10 达到 83.78，明显优于所有拼接基线，且在技能库规模增大时优势更显著。

**⚠️ 局限性**

局限性：①仅验证了字段分离与小型 MLP 的组合，未探究更深层的字段感知嵌入或张量分解技术；②实验仅在三字段（name, description, body）设置下进行，未验证其他字段组合的通用性；③模型训练成本相对较低，但在更大规模、跨域技能集上的鲁棒性仍需进一步评估。

---

## 348. Accelerating Human-Aware Robot Trajectory Generation via Diffusion and Consistency Distillation

**arXiv ID:** 2608.03159 | [PDF](https://arxiv.org/pdf/2608.03159v1)

**作者:** Byeong-Il Ham `[一作]`, Kyung-Soo Kim `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于扩散模型的人机交互轨迹生成框架，并通过一致性蒸馏显著加速推理。

**💡 创新点**

创新点在于将碰撞约束嵌入条件扩散模型，结合约束引导采样、一致性蒸馏与联合加权弯矩正则化，实现快速、平滑且安全的轨迹生成。

**🔧 技术方法**

使用RRT/RRT*生成训练数据，构建条件扩散模型与一致性模型，采用约束引导采样、时间一致性网络及联合加权弯矩正则化；实现基于PyTorch和Pinocchio的训练与推理。

**📊 数据集**

在UR5协作机械臂仿真环境中随机生成51,000条RRT/RRT*轨迹，拆分为42k/5k/0.5k用于训练、验证与测试。

**📈 对比分析**

与无弯矩正则化的扩散模型以及一致性模型对比；扩散推理耗时≈5.7 s，成功率≈99.6%；一致性蒸馏后推理仅≈95 ms，成功率≈98%，弯矩降低约36%，整体性能保持或提升。

**⚠️ 局限性**

仅在静态人类配置下验证，未考虑真实机器人跟踪误差、感知不确定性或动态人类动作；未来需在真实平台和动态环境中进一步验证。

---

## 349. LDU-Bench: Multimodal LLM Evaluation for Lithography Defect Understanding under Layout-Varying Circuit Backgrounds

**arXiv ID:** 2608.03078 | [PDF](https://arxiv.org/pdf/2608.03078v1)

**作者:** Huanglong Ji `[一作]`, Yue Lv `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 LDU-Bench，一个多任务、多模态的光刻缺陷理解基准，拆分为缺陷分拣、形态识别、粗定位和图像条件原因分析四个独立任务，并引入整体 Lithography Closure Score (LCS) 进行统一评估。

**💡 创新点**

创新点在于：①把光刻缺陷审查流程从单一检测解耦为四个可独立评估的阶段；②通过结构化诊断读数和 LCS 揭示现有多模态大模型在从低级视觉感知到高级语义推理的连续能力缺口；③为工业 MLLM 提供量化、可解释的评测框架。

**🔧 技术方法**

使用了多任务评测指标（宏 F1、DICU、结构化 rubric 评分）、诊断指标 CMD/EFS、CLIP 风格的视觉语言模型（GPT‑5.4、Claude Opus 4.6、Qwen3.6‑Plus、GLM‑5V‑Turbo、MiniMax‑M3、Gemma‑4‑31B‑it）、以及 AnomalyGPT 等对比模型，全部采用 deterministic scorer 与统一的 prompt 进行评估。

**📊 数据集**

基于 IC‑SEM 光刻审查图像构建的 LDU-Bench 数据集，包含 1,761 张用于缺陷分拣与形态识别的图像、984 张带掩码的定位样本、532 张带原因说明的图像，涵盖 BEOL、DEP、DPR 等多工艺阶段。

**📈 对比分析**

通过统一推理管道和 deterministic scoring，六种全链 MLLM 与 AnomalyGPT 在四个任务上分别得到宏 F1、DICU、rubric 评分；结果显示缺陷分拣的宏 F1 可达 0.93，但形态识别、定位和原因分析均显著下滑，最高 LCS 仅 0.474，说明现有 MLLM 在高层语义推理和因果映射上存在显著瓶颈。

**⚠️ 局限性**

局限性包括：①形态标签使用固定标准，未覆盖跨工厂、跨设备的标签差异；②原因分析仅基于图像可见证据，未整合工艺日志、设备状态等外部元数据；③公开基准受隐私和安全约束，未来可通过工业合作加入更完整的因果诊断资源。

---

## 350. Micro-Segmentation Anomaly Detection in Zero-Trust Software-Defined Network Fabrics

**arXiv ID:** 2608.02627 | [PDF](https://arxiv.org/pdf/2608.02627v1)

**作者:** Ashly Joseph `[一作]` `[通讯]`, Ashly Joseph

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在零信任 SDN 网络中，将微分段信息嵌入 ViT 与 1D‑CNN 深度学习模型，对比原始流与微分段流的异常检测效果。

**💡 创新点**

创新点在于将微分段作为特征直接融入模型，证明微分段显著提升对侧信道攻击和低速泄露的检测精度，并展示微分段与深度学习的协同优势。

**🔧 技术方法**

采用 Vision Transformer、1D‑CNN 两种深度学习架构，配合 min‑max 归一化、SMOTE 平衡、Adam 优化器以及片段 ID 嵌入等技术。

**📊 数据集**

使用基于 Mininet+Floodlight 的 SDN 仿真数据，结合 CIC‑IDS2017 与 NSL‑KDD 的攻击场景，生成 250,000 条流量记录，包含 5 个微分段标签。

**📈 对比分析**

通过相同的训练/验证/测试拆分，比较原始输入与微分段输入在准确率、F1、AUC 等指标上的表现；微分段 ViT 达到 97.5% 准确率、0.95 F1、0.99 AUC，明显优于原始模型。

**⚠️ 局限性**

局限性包括：依赖仿真数据，真实网络多样性未验证；微分段增加模型训练与维护成本；低流量段可能出现误报；缺乏在线动态分段与大规模部署的评估。

---

## 351. On the Diversity of Analogy Making in Large Language Models

**arXiv ID:** 2608.03233 | [PDF](https://arxiv.org/pdf/2608.03233v1)

**作者:** Yuanhao Shen `[一作]` (Queen's University), Xiaodan Zhu `[通讯]` (Queen's University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对10种开源与闭源LLM在类比生成任务中的多样性进行了系统评估，并探究了模型的域同质化现象及多样性提升方法的效果。

**💡 创新点**

首次从多样性角度全面评估LLM类比生成，揭示域同质化问题，并通过因果层级扰动分析解释多样性提升方法与质量权衡之间的机制。

**🔧 技术方法**

采用推理时扰动技术（多样性提示、logit steering、entropy‑gated steering）、语义聚类、MAUVE、多样性与质量指标、LLM‑as‑judge评估，以及层级噪声扰动识别模型敏感层。

**📊 数据集**

使用 AnaloBench、Metaphoric Analogies 与 Metaphor Understanding Challenge 三大类比生成数据集。

**📈 对比分析**

对比多种推理时多样性提升方法，结果显示没有单一方法在多样性、质量和聚类数上均占优，且不同模型对同一方法的响应差异显著，体现了多样性与质量之间的权衡。

**⚠️ 局限性**

局限性包括：多样性指标仍以表层语义为主，未能完全捕捉全局多样性；多样性提升策略主要在logit层，未能对齐模型内部敏感区；仅探索推理时改进，未涉及训练时方法；实验数据集规模有限，难以覆盖更广泛的类比场景。

---

## 352. Forecasting Revenue with its Customer-Base Drivers: When and Why Coordination Helps

**arXiv ID:** 2608.02911 | [PDF](https://arxiv.org/pdf/2608.02911v1)

**作者:** Kyeongbin Kim `[一作]` (University of Wisconsin--Madison), Dokyun Lee `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究联合预测客户基础的三个原始指标（客户获取、复购率、订单价值），并构建可解释的销售桥梁，帮助管理层判断销售变化来源。

**💡 创新点**

提出Customer-Based Multi-task Transformer（CBMT），在共享表示学习、任务特定头部和收益对齐正则化的组合下实现原始指标联合预测，同时提供可解释的收益桥梁。

**🔧 技术方法**

基于Transformer的多任务学习框架，加入收益对齐正则化；使用深度学习、线性回归、贝叶斯等多种基准模型做对比。

**📊 数据集**

使用Consumer Edge提供的2.3M信用/借记卡交易面板，涵盖2016-2020年约966家公司，按周分组的客户队列数据。

**📈 对比分析**

通过SMAPE、MAE等指标与15种基准模型（包括传统CBCV、概率CRM、LSTM、RF等）进行比较；CBMT平均总销售SMAPE比最强基准低约30%，比直接销售预测Transformer低约2.65%，在大多数公司上取得最高排名。

**⚠️ 局限性**

局限在于：模型仅基于交易面板预测，缺乏完整公司财务数据；对COVID-19等突发冲击未进行检验；单公司模型训练忽视跨公司迁移学习；无法提供个体层面的预测。

---

## 353. Internalising the Identity Primitive: Cryptographic Individuality for an Autonomous Agent on a Public Blockchain

**arXiv ID:** 2608.02986 | [PDF](https://arxiv.org/pdf/2608.02986v1)

**作者:** Keisuke Suzuki `[一作]` `[通讯]` (Hokkaido University), Keisuke Suzuki (Hokkaido University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

实现了一个在 Solana 公链上持续运行的 AI 代理，代理的神经网络权重完全由私钥确定，并在每个状态转移时通过零知识证明进行验证；整个系统在 2.36 天内完成 166 个周期的连续运行，没有任何拒绝的交易。

**💡 创新点**

创新点在于：① 将代理的计算子系统（权重张量）直接与私钥绑定，并在链上持续验证；② 通过一次性根证明实现“转移时间身份不变性”，使得任何权重替换或代码篡改都能在链上被拒绝；③ 将经济代谢机制（每周期消费费用）与身份绑定，形成完整的 (K, (K), (K)) 三轴模型。

**🔧 技术方法**

使用的技术包括：SP1 zkVM（编译 Rust → RISC‑V → STARK → Groth16 证明），EdDSA on edwards25519（签名），HKDF‑SHA256（确定性权重生成），Solana Anchor 框架（PDA、预编译 ed25519 验证），以及 Solana 的交易计费与最终化机制。

**📊 数据集**

数据集方面：实验主要采用合成环境输入（固定的 5 维向量序列）来驱动 Elman 循环；并在离线测试中使用 500 对随机密钥对评估权重差异，证明不同密钥产生的行为是可区分的。

**📈 对比分析**

性能对比（相对基准）：
- 区块链验证成本固定约 246k CU，远低于单笔交易上限；
- 每个 zk 证明的生成时间约 39 秒（GPU），一次性根证明约 48 秒；
- 166 周期连续运行无失败，平均每周期 20.6 分钟；
- 经济代谢测试显示每周期扣除 10,000 lamports，验证了代谢成本的可执行性。

**⚠️ 局限性**

局限性包括：
- 私钥由单一宿主持有，无法防止宿主被攻破；
- 权重不可训练，缺乏在线学习与自适应能力；
- 仅使用单一中心化 oracle，缺乏去中心化可信度；
- 对长周期连续性（>168 周期）仍依赖宿主恢复，缺乏完整的自愈机制；
- 对更大规模模型的证明成本及链上存储要求仍未知。

---

## 354. Tight Information Complexity of the Coin Problem in the Broadcast Model

**arXiv ID:** 2608.02776 | [PDF](https://arxiv.org/pdf/2608.02776v1)

**作者:** Hadi Kazemi `[一作]` (University of Cambridge), Varun Jog `[通讯]` (University of Cambridge)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

本文在广播模型下对“硬币问题”（区分两个伯努利分布）以及任意离散分布的检验信息复杂度进行严格分析，给出了在常数优势下的最优信息量，并指出在不同参数区间下最优协议的三种结构；

**💡 创新点**

创新点包括：1) 在三种参数区间内给出信息复杂度的精确上下界并揭示不同协议的最优性；2) 证明了一条新的混合Hellinger–Jensen–Shannon不等式，显著加强了以往的下界方法；3) 证明仅需考虑二值输出通道即可得到任意分布检验的最优信息复杂度；4) 将这些理论结果应用于集合不相交与多通道流处理的下界，取得了对先前结果的显著提升；

**🔧 技术方法**

主要技术手段包括信息理论框架、f‑divergence（Hellinger、Jensen–Shannon）的联合范围论证、混合不等式的构造、通道优化、广播模型下的 cut‑and‑paste 低界证明，以及对流式模型的多通道信息复杂度转换；

**📊 数据集**

本文完全基于理论分析，并未使用具体实验数据集；

**📈 对比分析**

与以往仅给出大致阶数或仅在特定极限下的下界相比，本文的上界与下界在常数优势下相匹配，精度达到常数因子；在流式和集合不相交等实际问题中，本文的下界比之前的 log 或常数因子提升了 log(1/α) 等对数项；

**⚠️ 局限性**

局限性包括：1) 只在常数优势下给出完整结果，对极低优势或渐进逼近极限的情况仍未覆盖；2) 单通道流式信息复杂度仍未得到完全解析；3) 主要针对广播模型，其它通信模型的推广仍需进一步研究。

---

## 355. PACE: Adaptive Budget Allocation for Time-Efficient Embodied Planning

**arXiv ID:** 2608.03034 | [PDF](https://arxiv.org/pdf/2608.03034v1)

**作者:** Yuchen Huang `[一作]` (ZTE Corporation), Wei Zhang `[通讯]` (ZTE Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出PACE框架，实现思考与执行交错与动态预算分配，使LLM在机器人等实体系统中实现实时规划。

**💡 创新点**

创新点包括Interleaved Think-Act架构、动态预算分配器（DBA）以及软硬结合的预算控制，实现时间延迟感知的TDAEP模型。

**🔧 技术方法**

使用了量化的Qwen3-8B‑AWQ LLM、vLLM推理框架、ReAct+Think流程、基于执行时间的预算公式以及软/硬预算约束。

**📊 数据集**

评估数据集为Robotouille同步任务集（100个实例），涵盖从10到63步的多种厨房任务。

**📈 对比分析**

与IO、ReAct、ReAct+Think、ReAct+Think(HB512)等基线进行配对比较，成功率从6%提升至10–13%，思考时间缩短6.9倍，思考隐藏率达约66%，显著提升了时间与质量的Pareto前沿。

**⚠️ 局限性**

局限性包括样本量有限、统计功效不足、仅使用单一模型、仿真与真实物理差距、仅关注成功率、预算规则为启发式，需在更大规模、多模型与真实机器人上进一步验证。

---

## 356. Character Iconicity vs. Arbitrariness: An Arabic NLP Perspective

**arXiv ID:** 2608.02935 | [PDF](https://arxiv.org/pdf/2608.02935v1)

**作者:** Dorieh Alomari `[一作]` (King Fahd University of Petroleum and Minerals), Maged S. Al-shaibani `[通讯]` (SDAIA--KFUPM JRC for AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对阿拉伯文字的视觉形态（点）与NLP性能的关系进行系统实验，比较传统无点写法与随机映射的字符简化；

**💡 创新点**

首次从计算角度检验视觉标识性 vs 任意性，证明随机重映射同样能保持高性能，并显著压缩词表、模型体积；

**🔧 技术方法**

采用字符/词级随机映射、Transformer语言模型、GRU/BiLSTM序列标注、BPE、Farasa、Disjoint等分词器，配合恢复模型进行评估；

**📊 数据集**

主要使用阿拉伯维基百科语料、Ashaar诗词、LABR评论、Sanad新闻、PADT POS、ANER NER、IWSLT 2017英阿双语等数据集；

**📈 对比分析**

通过在语言建模、文本分类、序列标注、机器翻译等任务上对标准点、点less、最高/最低熵随机映射进行对比，随机映射在词表压缩后仍保持几乎相同的准确率/BLEU，模型大小下降约30–50%，训练时间缩短；

**⚠️ 局限性**

实验规模受限，未覆盖大型预训练LLM；恢复模型采用RNN，可能不够鲁棒；仅验证阿拉伯语，缺乏跨语种推广。

---

## 357. Open-Linguistic Concept Unified Learning for Cross-Site Interpretable Dermatology Image Diagnosis

**arXiv ID:** 2608.03225 | [PDF](https://arxiv.org/pdf/2608.03225v1)

**作者:** Chengyu Wu `[一作]` (Zhejiang University), Yefeng Zheng `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种跨站点、跨模态的开放语言统一概念学习框架UniCon，用于可解释的皮肤病图像诊断。

**💡 创新点**

创新点在于（1）构建统一概念原型码表，将不同医院和不同图像类型（临床、镜下）下的概念映射到同一语义空间；（2）提出多面向开放语言规范（MSS），通过描述、同义词、边界例外和对比负样本三种文本槽增强概念的表达；（3）引入可靠性门控聚合（ICRG），根据图像条件动态筛选可信概念，实现可迁移的临床干预接口。

**🔧 技术方法**

技术方法包括：Vision‑Language 预训练模型（BiomedCLIP + Qwen3‑VL）作为基座，实例级对比学习、多模态注意力提取概念特征，统一概念原型码表（UCPC）构建，Learnable Evidence Aggregation（LEA）加权正负证据，Image‑Conditioned Reliability Gating（ICRG）进行动态过滤，以及三阶段联合训练策略。

**📊 数据集**

使用公开皮肤病数据集：Derm7pt（镜下图像）、SkinCon（临床图像，包含 Fitzpatrick17k 与 DDI 两子集），以及 PH2（镜下图像）做零样本验证。

**📈 对比分析**

与 12 种主流基准（CBM、PCBM、LF‑CBM、LaBo、MICA、MAKE 等）对比，UniCon 在 Derm7pt 上实现 AUROC 0.903、AUPRC 0.844、F1 0.828；在 SkinCon 上实现 AUROC 0.950、AUPRC 0.918、F1 0.854，均明显优于所有对比模型；在 PH2 的零样本任务中亦保持领先，提升诊断 AUROC 0.754、概念 AUPRC 0.749。

**⚠️ 局限性**

局限性包括：仍需依赖人工定义的概念集合；对新概念的适配需要重新构建码表；对非常低资源的医疗机构可能缺乏足够的概念标注；模型训练对计算资源要求高，实际临床部署仍需进一步验证。

---

## 358. LLMs Can Annotate Attribution Graphs

**arXiv ID:** 2608.02632 | [PDF](https://arxiv.org/pdf/2608.02632v1)

**作者:** Ameen Patel `[一作]`, Nathan Hu `[通讯]` (Stanford University)

**通讯引用:** 80991 | [OpenAlex ID](https://openalex.org/A5015008318)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套利用LLM自动化注释归因图的流程，取代原本耗时的人工节点分组步骤。

**💡 创新点**

创新点在于直接用LLM生成特征描述并将其聚合为supernodes，形成低成本、可扩展的自动化标注方法。

**🔧 技术方法**

核心技术包括GPT‑5 mini 进行特征描述与supernode聚类、自动解释度量（Feature Detection Score、Text Detection Score）以及在Gemma‑2‑2B模型上进行归因图生成与实验。

**📊 数据集**

使用的数据集为Gemma‑2‑2B模型的归因图，参考15个Neuronpedia图、100个两跳Capital任务查询以及1000条维基百科片段用于开放式探索。

**📈 对比分析**

通过与人工标注的supernodes进行自动解释度量对比，自动生成的supernodes与人类标注相当甚至略优；在两跳Capital任务中自动恢复中间跳转supernode的成功率为97/100，成本仅3–6美分/图。

**⚠️ 局限性**

局限性包括仅在单一模型和特征基（transcoder/MLP）上测试；仅使用单一LLM（GPT‑5 mini）；未利用空间信息、特征流向及更丰富的上下文；在更复杂任务如算术查询时表现不足。

---

## 359. KernelBrain: Coarse-to-Fine, Budget-Aware Search for Agentic GPU Kernel Optimization

**arXiv ID:** 2608.02611 | [PDF](https://arxiv.org/pdf/2608.02611v1)

**作者:** Shuai Che `[一作]` (Microsoft), Gang Peng `[通讯]` (Microsoft)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了KernelBrain，一套面向Triton核的智能优化框架，结合专家引导的变异、适应性多精度资源分配和基于性能诊断的LLM调优。

**💡 创新点**

创新点在于将多精度筛选与专家提示驱动的生成相结合，利用GPU内核诊断信息动态调整评估精度和迁移策略，实现低成本高效的搜索。

**🔧 技术方法**

采用LLM（GPT‑5.4）进行代码生成，NVIDIA Nsight Compute进行性能诊断，异步多阶段评估调度，基于CUDA/Triton编译和PyTorch基准。

**📊 数据集**

使用六个Triton基准核：GroupConv、MatVec、RMSNorm、Softmax、LinearReLUDiv、SDPAttn，覆盖不同算术密度和工作负载。

**📈 对比分析**

通过与PyTorch基准、单纯LLM生成、KernelAgent对比，KernelBrain在所有基准上获得0.88×–6.72×对PyTorch的加速，最高相对KernelAgent提升1.4×，并在48%以内节约搜索时间。

**⚠️ 局限性**

局限性在于仅处理源级Triton代码，无法直接优化编译器后端，部分性能机会仍待与编译器协同挖掘。

---

## 360. RIDGE: Re-Noising with Internal Dynamic Guidance for Image Editing

**arXiv ID:** 2608.03059 | [PDF](https://arxiv.org/pdf/2608.03059v1)

**作者:** Ruiliang Gong `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无反演、无训练的流模型图像编辑方法 RIDGE

**💡 创新点**

创新点在于重新设计目标侧状态构造：采用共享噪声的重噪声策略，并引入内部动态引导与软动态掩码，以在早期高噪声阶段提升编辑效果

**🔧 技术方法**

核心技术包括重噪声构造、速度差分更新、内部动态引导（目标状态预测与软掩码）以及流式反向时间采样

**📊 数据集**

使用 PIE-Bench Official 与 PIE-Bench++ 两个基准数据集，并在 Stable Diffusion 3 Medium 与 FLUX.1-dev 两个后端模型上进行评测

**📈 对比分析**

与多种基线方法（如 FlowEdit、FlowAlign、DRFS 等）对比，RIDGE 在源保持、目标对齐、感知质量等六项指标上均获得最高平均分，整体性能优于现有方法

**⚠️ 局限性**

局限性是对大幅度身份、几何或场景改动的编辑仍显保守，难以实现强烈的目标驱动修改

---

## 361. TQLite: Multi-LLM Jury Guided Distillation for Real-time MQM Translation Quality Evaluation

**arXiv ID:** 2608.02975 | [PDF](https://arxiv.org/pdf/2608.02975v1)

**作者:** Bhavin Jawade `[一作]` (Netflix), Cameron R. Wolfe `[通讯]` (Netflix)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过大规模实验比较了小语言模型（SLM）、大语言模型（LLM）和大推理模型（LRM）在基于MQM的翻译质量（TQ）评估中的表现，并提出一种基于多LRS jury的知识蒸馏框架TQLite，利用多模型投票生成高质量合成标注，进一步训练并压缩为高效的开源SLM评估器。

**💡 创新点**

创新点主要在于：①构建多LRS jury体系，采用投票和元评审聚合高置信度的错误标注；②开发TQLite蒸馏流程，将多模型推理结果压缩到小模型中；③系统评测了多种提示、输出格式、推理强度对评估质量的影响，为LLM-as-a-Judge提供最佳实践。

**🔧 技术方法**

使用技术包括：多模型交互式投票、CoT推理、JSON/结构化文本输出、元评审聚合、LoRA参数高效微调、OpenAI API和vLLM推理加速；核心算法为MQM分数计算、投票一致性筛选、误差标注聚合。

**📊 数据集**

主要数据集为WMT22 metrics test set（zh-en、en-de、en-ru共约2,000句），以及3M来源多语种语料（西班牙语、俄语、德语、中文、英语、日语、印地语等）用于合成训练数据；同一数据集被用于多模型的投票与微调。

**📈 对比分析**

通过与多种基准（MetricX、COMET、BLEURT、COMET-kiwi、COMET-QE、GPTScore、G-Eval等）对比，系统层面LRM平均准确率可达92%（o3高推理），而经过TQLite蒸馏的SLM在段级平均准确率约55%，显著优于所有开源SLM基线，并接近闭源LRM水平；在推理速度上TQLite实现更低延迟与成本。

**⚠️ 局限性**

局限性包括：蒸馏后的SLM仍无法与最佳闭源LRM相比；合成数据可能带来教师模型的系统性偏差；高一致性筛选会降低覆盖率，可能忽略难以判定的错误；实验仅覆盖高中等资源语言，未验证在低资源或形态复杂语言上的泛化。

---

## 362. Semantic Haptic Feedback Enhances Dexterous Robotic Teleoperation

**arXiv ID:** 2608.02780 | [PDF](https://arxiv.org/pdf/2608.02780v1)

**作者:** Bingjian Huang `[一作]`, Chase Tymms `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种语义触觉反馈框架，用于机器人遥操作的精细操作；通过可配置的管路将机器人状态映射为抽象触觉模式，并在腕带上实现气压与振动反馈。

**💡 创新点**

1）将高维触觉信息压缩为低维语义消息；2）利用气压与振动的模态一致性，实现一次性多状态映射；3）在多手操作中通过腕部触觉保持视觉注意力。

**🔧 技术方法**

使用Unreal Engine 5仿真平台、Franka FR3机械臂、Robotiq 2F‑85抓手、气压腕带与振动腕带；实现抓取稳定性与滑移检测算法，设计四种语义模式（确认/警告）。

**📊 数据集**

实验数据来自12-20名受试者在仿真环境下完成单手与双手的箱块搬运、容器滑移等任务，记录搬运次数、错误数、抓取力、主观工作量等指标。

**📈 对比分析**

与无反馈、视觉叠加、感官触觉等对照，采用重复测量ANOVA、Friedman检验和NASA‑TLX；结果显示在双手任务中语义触觉显著降低工作量、提高情境意识，且用户首选；在单手任务中差异不显著。

**⚠️ 局限性**

限制：需预先知道物体特性；仅在仿真中验证，缺乏真实机器人硬件测试；仅使用单一模式（稳态压缩、频率警告），难以覆盖更复杂任务；腕部反馈与交互点分离可能增加认知负荷。

---

## 363. 3 Players Auction Bridge - Statistical Algorithmic Strategies

**arXiv ID:** 2608.03217 | [PDF](https://arxiv.org/pdf/2608.03217v1)

**作者:** Sourish Sarkar `[一作]` (Indian Statistical Institute), Moutushi Chatterjee `[通讯]` (Indian Statistical Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对三人拍卖桥中无红牌叫牌的精确规则算法；

**💡 创新点**

创新点在于用确定性规则而非概率模型，显著提升无红牌获胜概率；

**🔧 技术方法**

采用枚举式算法与Python的NumPy、Pandas库进行模拟与分析；

**📊 数据集**

使用标准52张牌随机抽取的13张手牌进行100,000次Monte Carlo模拟；

**📈 对比分析**

与原先的概率性启发式对比，赢率从10^-5级提升至约0.06/0.03/0.0085，性能显著优异；

**⚠️ 局限性**

局限在于仅适用于全手牌信息，且仅针对无红牌情况，部分信息下效果未知。

---

## 364. Beyond Invariant Dictionary: Data-Driven Koopman Spectral Recovery with Filtered Extended Dynamic Mode Decomposition

**arXiv ID:** 2608.02661 | [PDF](https://arxiv.org/pdf/2608.02661v1)

**作者:** Siji Chen `[一作]` (Princeton University), Sui Tang `[通讯]` (University of California)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Projected Koopman Operator Approximation (PKOA) 框架，并在此基础上开发 Filtered EDMD 方法，用一次性投影前向交叉链对 Koopman 作用进行过滤，消除谱污染。

**💡 创新点**

创新点在于：①将范围选择与投影几何分离，定义可接受投影并证明所有非零 Koopman 本征对在该投影下保持不变；②使用前向交叉链构造一族兼容子空间，既能保留最大 Koopman 不变子空间，又能通过中间层逐步去除无关方向；③实现了无 Gram 矩阵估计、仅基于坐标的 SVD 方案，兼顾数值稳定与可实现性。

**🔧 技术方法**

技术包括：Koopman 理论、EDMD、投影算子理论、前向交叉链、SVD 及 QR 递归求交、坐标与 L² 归一化投影、极限收敛性证明（独立采样、无噪声、完备性）。

**📊 数据集**

实验数据集：Kronecker 流 (二维圆环), 低阶多项式非线性系统 (四维), Van der Pol 惯性振荡器 (二维，加入高阶多项式与三角基)。

**📈 对比分析**

比较方法包括 EDMD、SSD/T‑SSD、RFB‑EDMD、ResDMD 与 Filtered EDMD；评估指标为谱污染程度、矩阵块结构、误差统计、运行时间。结果显示 Filtered EDMD 在所有三种系统上都能消除伪谱，保持真谱，且中间级别可保留更多有用信息；在 Van der Pol 场景中仍能捕捉平衡点谱，其他方法无法得到。性能方面，计算成本略高于普通 EDMD，但远低于 SSD/T‑SSD，且对样本扰动更稳健。

**⚠️ 局限性**

局限性：①仅实现了坐标投影，缺乏对最优投影选择的理论与算法；②理论假设为无噪声、可识别、字典线性无关，未考虑观测噪声和有限样本误差的鲁棒性；③若字典根本不包含任何 Koopman 本征函数，终端不变空间无信息，中间层的选择和效果需进一步研究；④SVD 与交叉链在大规模字典下可能产生较高计算成本。

---

## 365. Studying, Identifying, and Fixing Hidden Technical Debt in AI-Intensive Cyber-Physical Systems

**arXiv ID:** 2608.02638 | [PDF](https://arxiv.org/pdf/2608.02638v1)

**作者:** Beena `[一作]` `[通讯]` (University of Sannio), Beena (University of Sannio)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过挖掘 Hugging Face 与 GitHub 上的 AI 模型仓库、AI‑CPS 仓库以及对开发者的访谈，构建了 AI‑CPS 专属技术债务（TD）分类体系，并提出了面向 AI‑CPS 的自动化 TD 识别、分类和治理框架。

**💡 创新点**

创新点在于首次系统性刻画 AI‑CPS 的 TD 产生机理与类别，扩展并细化了传统 TD 分类；引入多源信息融合（代码、模型卡、硬件配置、仿真文件等）和代理式 AI（agentic AI）来实现持续监控、评估与修复建议。

**🔧 技术方法**

使用了混合方法（仓库挖掘、访谈分析）、静态代码与配置分析、自然语言处理（对模型卡、注释、问题等文本的分析）、机器学习/大语言模型（LLM）用于分类与建议生成，以及自定义的代理 AI 系统（识别、推荐、评估、协调等子代理）。

**📊 数据集**

主要数据集包括 Hugging Face 上的公开模型及其对应 GitHub 仓库、15 个开源 AI‑CPS 项目的代码仓库、模型卡与训练/推理脚本、以及通过访谈收集的工程师经验和问题列表。

**📈 对比分析**

目前阶段尚未完成大规模性能评估，已通过案例研究和访谈验证了分类体系的可行性，后续计划在学术实验和 InnoGuard 演示项目（如人形机器人、Leo Rover）中对比代理式治理方案与传统静态分析方法的检测准确率和修复效果。

**⚠️ 局限性**

局限性包括：① 数据来源受限，覆盖的 AI‑CPS 仅为 15 个开源项目；② 分类体系和治理策略尚未在大规模真实项目中全面验证；③ 代理式 AI 仍为半自动化，需要人工确认；④ 对不同硬件/仿真平台的适配性仍待进一步研究。

---

## 366. Self-Organising Digital Circuits

**arXiv ID:** 2608.02606 | [PDF](https://arxiv.org/pdf/2608.02606v1)

**作者:** Marcello Barylli `[一作]` (IT University of Copenhagen), Sebastian Risi `[通讯]` (IT University of Copenhagen)

**通讯引用:** 3798 | [OpenAlex ID](https://openalex.org/A5020511097)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用拓扑掩码Transformer作为本地策略，训练数字电路在从无到有的自组织阶段以及遭遇软硬件故障后的自我修复。

**💡 创新点**

创新点包括：①将Neural Cellular Automata扩展到任意图形上实现功能生成；②设计Topology‑Masked Transformer实现无梯度全局搜索的可重复前向推理；③通过持续的元学习实现规模无关的自组织与维修；④揭示退化解空间的结构多样性。

**🔧 技术方法**

技术手段：拓扑掩码Transformer、可微分LUT（连续逻辑门）、BPTT/元学习、随机图生成、残差+ReZero、可扩展的正弦位置编码、梯度回传的闭环反馈。

**📊 数据集**

数据集：三种12位布尔任务（拆分乘法、拆分加法、位反转）共4096个输入输出对；此外使用随机生成的DAG拓扑（不同宽度和层数）作为训练和测试集。

**📈 对比分析**

与全局反向传播（BP）基线对比；在无故障情况下可达与BP相当的精度；在软错误、随机损伤和大规模未见故障下，恢复精度>99.99%；对更宽的电路（1.7×训练宽度）性能提升，证明规模无关性。

**⚠️ 局限性**

局限性：①定位编码仅捕捉深度，缺乏局部拓扑信息；②错误反馈仅为标量残差，缺乏任务数据的细粒度指导；③掩码为密集矩阵，计算开销随节点数呈线性增长；④仅在宽度扩展时表现良好，深度扩展仍需改进；⑤未实现结构层面的增删、重连，仍依赖固定拓扑。

---

## 367. Towards Wearable Opportunistic Crowdsensing for Open-Vocabulary Activity Data Collection Through User-Scheduled Trigger-Action Routines

**arXiv ID:** 2608.03152 | [PDF](https://arxiv.org/pdf/2608.03152v1)

**作者:** Zeyu Wang `[一作]`, Yuntao Wang `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种基于用户制定触发‑行动（trigger‑action）配方的可穿戴设备系统，利用手机与手表的协同，在实时触发时提示用户确认并记录短时传感窗口，从而实现低负担的活动数据自标注。

**💡 创新点**

创新点在于将自标注嵌入用户认为有价值的微习惯提醒中，既可作为健康行为干预，又可产生时间精确、开放词汇的标签；系统采用 LLM 辅助配方生成并支持开放词汇标签。

**🔧 技术方法**

技术上采用 iOS/watchOS 原生开发，利用本地音频分类与运动门限检测触发，配合手表 haptic 提示与短时 IMU/音频记录；配方作者通过手机聊天界面与 LLM 交互生成结构化配方。

**📊 数据集**

实验使用了自定义的 4 类音频触发（门关、冲水、流水、铃声）与用户自定义动作标签，并在实验室中收集 IMU 与音频窗口；无公开数据集，全部为自收集实验数据。

**📈 对比分析**

通过专家工作坊（N=6）、受控实验室研究（N=21）和 4 小时实地部署（N=8）进行评估；在实验室中执行记录的召回率 97.30%、精确率 97.15%，用户感知负担低于传统标注方式，且标签可靠性高于后期视频标注。

**⚠️ 局限性**

局限性包括仅支持四种音频触发导致触发误报、对环境噪声敏感、样本年轻、Apple 生态局限、实验时间短、未验证模型下游效果、未深入探讨长期使用与习惯形成。

---

## 368. Interpreting Black-Box Large Language Models with Sentence-Level Energy Landscapes

**arXiv ID:** 2608.02879 | [PDF](https://arxiv.org/pdf/2608.02879v1)

**作者:** Maryam Rezaee `[一作]` (Sharif University of Technology), Fatemeh Seyyedsalehi `[通讯]` (Sharif University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于能量场的句子级解释器，能够在不调用目标LLM API的情况下识别哪些输入句子最影响特定输出句子

**💡 创新点**

创新点在于：①将解释单元从噪声高的token转移到语义完整的句子；②使用能量模型（EBM）学习黑盒LLM的全局一致性，并以此引导轻量级解释器训练；③实现了全局训练的本地解释器，避免实例化偏差并实现单次推理无需再查询

**🔧 技术方法**

使用Transformer结构的能量模型、句子BERT嵌入、对比学习（InfoNCE）及自注意力投影、交叉注意力交互、Gumbel-Softmax实现二进制重要性向量

**📊 数据集**

在多领域问答数据集Hello‑SimpleAI/HC3上采样约2000条问答对进行实验，并使用多款LLM（GPT‑4o、Gemini等）作为“或acles”进行对齐评估

**📈 对比分析**

与多种解释基线（LIME、随机、不同大小LLM等）对比，采用nDCG、Soft Top‑1、Sufficiency/Comprehensiveness等指标；结果显示ESC I在效率上接近LIME，在解释精度上与大模型相当，并且在推理时无需额外API调用

**⚠️ 局限性**

主要限制：训练阶段对API调用与计算资源要求高，框架任务专一且需要手工调参；缺乏硬件可验证的机制层面基准，难以评估与真实内部路径的匹配；对长文本、复杂推理等任务的泛化与可扩展性仍待验证

---

## 369. Aligning Large Vision-Language Models at Test Time: A Trajectory-Guided Structured Sampling Approach

**arXiv ID:** 2608.03204 | [PDF](https://arxiv.org/pdf/2608.03204v1)

**作者:** Tianbao Jiang `[一作]` (East China Normal University), Linlin Wang `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的测试时对齐框架，利用轨迹学习构建推理记忆库，并在推理时通过轨迹引导的结构化MCMC采样与多目标目标函数提升大型视觉语言模型的视觉推理性能。

**💡 创新点**

创新点在于自动化轨迹记忆构建、轨迹引导的局部结构化MCMC采样以及融合视觉校准、熵正则和语言去退化的多目标采样目标，实现了无需参数更新的高效对齐。

**🔧 技术方法**

使用了自动轨迹学习（规划器/执行器/管理器）、向量检索/投票生成指导轨迹、Metropolis–Hastings局部采样、功率采样、视觉感知校准、熵正则化与语言去退化控制等技术。

**📊 数据集**

在 MathVista、MathVision、MathVerse、MMMU、MMStar 等多模态推理基准上进行评估。

**📈 对比分析**

与基线 Qwen2.5‑VL‑7B、RLVR 训练模型以及其他通用 LVLM 进行对比，平均准确率提升至 54.6%（比基线高约2.6个百分点），在多项数据集上逼近或优于 RLVR 训练模型，同时推理成本显著低于全序列采样。

**⚠️ 局限性**

主要限制包括对轨迹记忆库的依赖、在高度专业化的推理场景中效果略逊、MCMC 采样仍有较高计算开销且需手工调参。

---

## 370. Temporal Leakage in LLM Backtesting: Measurement, Validation, and Adjusted Scores

**arXiv ID:** 2608.02985 | [PDF](https://arxiv.org/pdf/2608.02985v1)

**作者:** Zeyu Zhang `[一作]` (Northwestern University), Bradly C. Stadie `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文展示了大语言模型（LLM）在回测中的污染检查方法的不足，提出了一种新的方法来识别和调整回测中的时间泄漏问题。

**💡 创新点**

创新点在于证明了传统的前后对比检查无法有效识别污染，并提出了使用已知截止日期和匹配的干净控制模型来识别和调整泄漏的方法。

**🔧 技术方法**

使用了回归不连续性和差异中的差异（DiD）等统计技术来识别和调整时间泄漏。

**📊 数据集**

使用了来自ForecastBench的1646个已解决的二元市场问题的数据集，这些问题的结果在2024年第三季度到2026年第三季度之间解决。

**📈 对比分析**

与传统的前后对比方法相比，提出的方法能够有效识别和调整时间泄漏，性能上能够清除五个模型的表面优势，这些优势仅仅是由于近期性造成的。

**⚠️ 局限性**

限制在于所提出的方法依赖于特定的假设，如已知的截止日期和干净的控制模型，且在某些情况下可能无法完全识别所有类型的泄漏。

---

## 371. A Human-in-the-Loop Deep Learning Framework for Color Reconstruction of Lenticular Films

**arXiv ID:** 2608.02835 | [PDF](https://arxiv.org/pdf/2608.02835v1)

**作者:** Saptarshi Neil Sinha `[一作]` (Fraunhofer IGD), Giorgio Trumpy `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种人机协同的深度学习框架，用于历史凸透镜胶片的颜色重建；

**💡 创新点**

创新点在于将凸透镜边界检测与颜色化解耦，公开可编辑的向量化边界表示，并通过专家交互反复细化与模型微调，最终实现纹理保留的后处理；

**🔧 技术方法**

使用的技术包括基于U‑Net的边界分割网络、深度学习颜色化网络、Lab色彩空间后处理、以及基于专家标注的迭代微调；

**📊 数据集**

实验数据集为德国1930年拍摄的《Ein Besuch im Lappenlager》历史凸透镜胶片序列；

**📈 对比分析**

与完全自动的deep‑doLCE方法对比，采用专家验证与迭代微调后，颜色质量大幅提升，视觉评估显示在四轮迭代内已达到可公开展览的水平；

**⚠️ 局限性**

局限性包括缺乏定量评价指标、对专家时间成本高、仅验证了Kodacolor胶片，且对不同胶片工艺的适用性仍待进一步研究。

---

## 372. Towards More Expressive Spoken LLMs: Fine-Grained Intent Benchmarking and Acoustic-Lexical Decoupled Policy Optimization

**arXiv ID:** 2608.03054 | [PDF](https://arxiv.org/pdf/2608.03054v1)

**作者:** Xiang Lin `[一作]` (Li Auto Inc), Liang Li `[通讯]` (Tsinghua University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了中文口语情感对话基准 ParaIntent，并提出 Acoustic-Lexical Decoupled Policy Optimization（ALPO）方法

**💡 创新点**

创新点在于将文本与语音奖励分别归一化并分别路由给对应的文本/语音 token，解决了传统 GRPO 的跨类型奖励误导问题

**🔧 技术方法**

采用 GLM‑5 进行意图推断与文本奖励，IndexTTS2 合成语音，使用强化学习（GRPO）和自定义格式/文本/语音奖励，结合独立优势归一化与 token‑level 路由

**📊 数据集**

使用 ParaIntent 数据集：14 个细粒度意图（4 类），14K 语音样本（合成+人工），140K 训练集，涵盖显式与隐式意图

**📈 对比分析**

与基线 SFT、DPO、GRPO 以及混合 SFT‑GRPO 进行对比。ALPO 在 IF、RQ、EA、ES 以及人类主观评估上均排名第一或第二，尤其在情感相似度（ES）和整体表现上显著提升

**⚠️ 局限性**

局限性包括：仅在单轮、单一 interleaved 文本‑语音模型上验证；数据主要为合成语音；未覆盖多轮对话或更真实的口语交互场景

---

## 373. Distributed Algorithms for Near-Equitable Coloring

**arXiv ID:** 2608.02910 | [PDF](https://arxiv.org/pdf/2608.02910v1)

**作者:** Amit Nir `[一作]` (Weizmann Institute of Science), David Peleg `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在分布式网络中实现近乎公平(near‑equitable)图着色的算法，探讨调色板大小与频率范围之间的权衡；

**💡 创新点**

创新点在于提出多种快速随机分布式算法，从允许更多颜色的近公平着色到仅用Δ+1颜色但允许较大频率偏差的方案，并提供了可调参数的通用框架；

**🔧 技术方法**

核心技术包括分割与重着色(parallel split‑and‑recolor)范式、颜色计数与配额分配(quotas)、以及基于随机排名的候选者筛选，辅以集中计数与分布式通信原语；

**📊 数据集**

论文未使用具体数据集，而是基于理论分析与复杂度评估，考虑任意n点、最大度Δ、直径D的无向图；

**📈 对比分析**

与现有的O(n²Δ)顺序算法及传统Δ+1着色相比，提出的算法在分布式CONGEST与Congested‑Clique模型中实现了O(D+Δ+Δ⁵n)或O(Δ·(D+Δ+Δ⁵n))的运行时间，并在调色板大小与频率范围上提供多种折衷；

**⚠️ 局限性**

局限性包括：需要随机化且只给出高概率结果；对特殊图结构未做针对性优化；实现细节高度依赖于原语的高效实现，且在极大Δ或低Δ场景下仍可能存在性能瓶颈。

---

## 374. UrbanAgent: A Tool-Augmented Agent for Cross-System Urban Tasks

**arXiv ID:** 2608.03018 | [PDF](https://arxiv.org/pdf/2608.03018v1)

**作者:** Jiayu Cao `[一作]`, Jian Yin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 UrbanAgent 框架，使大语言模型通过工具调用能够执行跨系统的城市任务，并构建了 UrbanEval 基准来评估此类任务的成功率与执行质量。

**💡 创新点**

创新点在于提出一个闭环的认知澄清–推理–工具调用–证据对齐合成流程，能够在执行前主动澄清缺失信息、保持工具调用顺序、验证结果的真实性，并首次为城市跨系统任务量化评估提供了专用基准。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑5‑mini、Gemini‑2.5‑flash 等）+ 工具调用接口（API、代码执行）+ Model Context Protocol + ReAct 风格的推理执行 + 基于证据的合成策略。

**📊 数据集**

数据集为 UrbanEval，共 250 条自然语言城市请求，涵盖 5 种任务类型和 3 难度等级，任务执行通过 7 个城市计算服务（MCP）提供的实时数据完成。

**📈 对比分析**

方法上将 UrbanAgent 与四种主流工具协同基线（Native, ReAct, Plan‑and‑Execute, AutoGen）在相同模型、工具、预算下进行对比。实验显示在 GPT‑5‑mini 上 UrbanAgent 的任务成功率为 71%，比最强基线 61% 高 10 点，且在所有四种基线模型上均保持最高排名。

**⚠️ 局限性**

局限性包括：相比基线消耗约三倍令牌，且在部分简单检索或受约束性较弱的任务中提升有限；对未见城市或新服务的泛化能力尚未评估。

---

## 375. AnchorKV: Anchor-Residual KV Cache Compression

**arXiv ID:** 2608.02901 | [PDF](https://arxiv.org/pdf/2608.02901v1)

**作者:** Malik Khalaf `[一作]` (Technion - Israel Institute of Technology), Assaf Schuster `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

AnchorKV通过引入锚点+残差的KV缓存压缩方案，在不丢弃任何token的前提下实现高达20倍的压缩；

**💡 创新点**

创新点在于将每个KV向量表示为最相似锚点的标量投影，并根据注意力输出误差动态分配量化残差，从而保持完整上下文并极大提升压缩率；

**🔧 技术方法**

采用锚点选择（基于SnapKV的观察窗口）、投影+残差表示、Hadamard旋转+Lloyd–Max两位量化、基于注意力输出的残差优先级估计以及FlashAttention风格的分块解码实现；

**📊 数据集**

在Llama‑3.1‑8B/24B/70B模型上使用RULER、LongBench和Needle‑in‑a‑Haystack等长上下文基准数据集进行评估；

**📈 对比分析**

与完整KV、三种eviction方法（SnapKV、PyramidKV、AdaKV）和TurboQuant基准在相同字节预算下对比，AnchorKV在20×压缩下保持93–99%完整缓存精度，优于10×压缩下的eviction方法，且在70B模型上可达99.3%精度；

**⚠️ 局限性**

局限性包括：残差量化仅为两位，仍有限制；在非常低的预算下仍会出现性能下滑；需要额外的前置压缩步骤且对GPU显存与实现细节有一定依赖。

---

## 376. On the Performance of Malware Detection Classifiers Using Hardware Performance Counters

**arXiv ID:** 2608.02671 | [PDF](https://arxiv.org/pdf/2608.02671v1)

**作者:** Alireza Abolhasani Zeraatkar `[一作]` (University of California), Hussain Al-Asaad `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多种机器学习模型与集成学习方法结合硬件性能计数器(HPC)用于恶意软件检测，并在不同HPC数量下比较其性能。

**💡 创新点**

创新点在于通过集成学习显著减少所需的HPC数量（从16降至4或2），同时保持甚至提升检测精度，从而降低硬件监控成本。

**🔧 技术方法**

使用了多类机器学习分类器（MLP、SGD、Random Forest、J48、LMT等）以及AdaBoost和Bagging等集成方法，对HPC特征进行特征选择与降维。

**📊 数据集**

使用了基于Linux perf收集的16维HPC数据集，平衡后包含3700个恶意样本和3700个正常样本。

**📈 对比分析**

通过准确率、AUC及综合性能（Accuracy×AUC）三种指标进行比较，实验表明在4个HPC配置下，集成模型可与16个HPC单一模型达到相同或更高的性能，提升约10%–15%。

**⚠️ 局限性**

局限性包括：在仅使用2个HPC时性能明显下降；实验仅在Docker环境下进行，缺乏真实系统负载验证；未对潜在对抗性攻击进行评估；仅针对单一数据集，泛化能力待进一步验证。

---

## 377. Sensus Pond: Exploring Water as Sensing Medium for More-than-Human Observation

**arXiv ID:** 2608.02749 | [PDF](https://arxiv.org/pdf/2608.02749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 378. IR2Solve: Structured Intermediate Representations for Cost-Efficient Optimization Autoformulation

**arXiv ID:** 2608.02641 | [PDF](https://arxiv.org/pdf/2608.02641v1)

**作者:** Penglin Zhu `[一作]` (University of Chinese Academy of Sciences), Xiuqi Wu `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 2030 | [OpenAlex ID](https://openalex.org/A5113805602)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

IR2Solve 提出了基于单次 LLM 调用的中间表示模型生成及后续确定性验证与编译的优化自动建模流水线。

**💡 创新点**

其创新点在于将模型抽象为约束清晰的 ModelIR，采用标量化约束、守恒规则和确定性编译，消除了传统直接代码生成的脆弱性并降低推理成本。

**🔧 技术方法**

技术包括使用 GPT-4o 进行语义转化、JSON 结构化中间表示、限定的 Python 类表达式、固定规则的验证器以及无 LLM 调用的 Gurobi 编译器。

**📊 数据集**

评估使用了六个清洗后基准数据集（NL4Opt、IndustryOR、EasyLP、ComplexLP、NLP4LP、ReSocratic）以及 153 条 IndustryOR 与 ComplexLP 进行消融实验。

**📈 对比分析**

与 Code-first、Chain-of-Experts、SAC-Opt 等方法相比，IR2Solve 在目标正确率上与 SAC-Opt 相近或更优，同时仅需一次 LLM 调用，令 token 量和调用次数降低 3.3×~22.9×，实现了更佳的精度-成本折衷。

**⚠️ 局限性**

局限性包括仅覆盖有限的 LP/MILP 形式、标量化约束导致输出长度膨胀、消融实验范围有限、模型覆盖率未直接测评，以及仅通过目标值一致性评估语义正确性。

---

## 379. Test-Time Scaling for Safe Text-Guided Image Generation via Intermediate Clean Estimates

**arXiv ID:** 2608.03284 | [PDF](https://arxiv.org/pdf/2608.03284v1)

**作者:** Jinya Sakurai `[一作]` (Nanyang Technological University), Xun Xu `[通讯]` (Institute of Advanced Intelligence and Computing, A*STAR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种重量不改动的测试时安全防御框架T2S2，通过在扩散过程的中间步骤监测干净图像估计并在检测到违规视觉内容时局部更新低秩文本条件残差，实现对裸体、IP角色及艺术风格等违规概念的抑制；

**💡 创新点**

创新点包括①利用中间干净图像估计作为视觉证据触发安全干预；②采用稀疏边际损失与低秩残差参数化在文本条件空间进行局部优化；③通过自适应触发和可调的测试时计算预算形成安全-计算权衡；

**🔧 技术方法**

技术手段主要有：低秩残差参数化 (B·A)、稀疏边际（hinge）损失、velocity基干净图像估计、截断梯度更新、适应性触发（adaptive unrolling）以及测试时缩放（test‑time scaling）；

**📊 数据集**

使用的数据集包括 Stable Diffusion v1.4 与 v3.5m；红队提示集 I2P、P4D、Ring‑a‑Bell、MMA‑Diffusion、Unlearn‑Diff‑Atk；NudeNet 作为裸体检测器；COCO‑30k 用于 FID/CLIP 评估；IP 字符（如 Mickey Mouse、Elsa、Sonic）和艺术风格（如 Van Gogh、Picasso、Rembrandt）样本；

**📈 对比分析**

与训练型权重编辑方法（ESD、CA、MACE 等）、测试时权重编辑方法（UCE、RECE、GLoCE）以及测试时权重保留方法（POSI、SLD、SAFREE、ZeroShot‑POSI）比较。T2S2 在红队数据集上的裸体检测率显著降低，同时保持或提升 FID/CLIP 指标，且在 IP 角色和艺术风格移除任务中实现更优的抑制‑保留权衡；此外，可通过设置不同的 (K, K_sub) 预算点实现安全‑计算可调；

**⚠️ 局限性**

局限性主要包括：①依赖 CLIP 安全编码器，存在语义歧义；②仅在检测到违规时才优化，可能错过间接攻击；③低秩更新在强干预时可能导致过度抑制或语义失真；④未直接优化原始提示或最终图像质量，可能出现细节缺失或风格失衡。

---

## 380. PolicyGuard: Prompt-Configurable Semantic DLP for LLM Coding Agents

**arXiv ID:** 2608.02687 | [PDF](https://arxiv.org/pdf/2608.02687v1)

**作者:** Kyutae Park `[一作]` (Amazon Web Services), Daeyeol Shim `[通讯]` (OpenAI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PolicyGuard，一个通过自然语言政策文件在编码代理调用前拦截并分类用户提示的 DLP 框架。

**💡 创新点**

将 DLP 规则以可编辑的自然语言政策文件形式提供，避免传统 regex/模型微调；实现了封闭评估协议、跨模型可迁移性，并证明自然语言格式优于 JSON。

**🔧 技术方法**

基于 LLM 的分类器（如 gpt‑oss‑safeguard‑20b）、policy‑as‑prompt 方案、fail‑closed 设计、少量示例提示以及模板级拆分评估。

**📊 数据集**

合成的 2,000 条多语言（英、韩、日、中文）提示，按开发、验证、隐藏 holdout、冻结测试集拆分，覆盖凭证、PII、机密信息与注入三类敏感标签。

**📈 对比分析**

在冻结测试集上与 JSON 等价规则、零射击和长度控制基线进行 McNemar、Cohen h 统计对比；PolicyGuard 在 927 条测试提示上达 96.5% EBR、3.0% FPR，且在四种 LLM 上保持 86.4–96.5% 的 EBR，JSON 格式准确率 90.7%。

**⚠️ 局限性**

数据为合成、单轮分类、对不同 LLM 依赖显著、未做专门对抗测试、格式遵从率 89.9%、无法处理多轮上下文等。

---

## 381. Activation-Guided Neuron Intervention to Induce Alzheimer's-Related Computational Language Phenotypes in a Large Language Model

**arXiv ID:** 2608.03067 | [PDF](https://arxiv.org/pdf/2608.03067v1)

**作者:** Rui He `[一作]` (Universitat Pompeu Fabra), Wolfram Hinzen `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

使用 Qwen3-8B 通过激活差异识别 AD 相关神经元，并通过缩放其下投影权重进行单元级干预，构建 AD 计算语言表型；

**💡 创新点**

首次将基于 AD 语料激活差异的神经元定位与单元级放大/衰减相结合，形成可实验操控的 AD 计算表型；

**🔧 技术方法**

激活提取、下投影权重缩放、神经网络干预、一般化估计方程(GEE)、神经心理学评测与自动语言指标分析；

**📊 数据集**

ADReSSo 语料（与 DementiaBank 记录对应）中 AD 与健康对照的口语转录；

**📈 对比分析**

对原始模型与九种编辑变体进行人类评分与语言指标比较，放大干预导致多领域功能受损、语料特征与临床 AD 相似，衰减则基本保持或提升部分表现；

**⚠️ 局限性**

仅评估单一 8B 语言模型、单一英语数据集、所有任务均通过语言完成，导致跨模型、跨语言及临床人群的普适性受限。

---

## 382. DRIFT: Derailing Denoising Trajectories of Flow-Matching VLAs with Adversarial Patch Attack

**arXiv ID:** 2608.03207 | [PDF](https://arxiv.org/pdf/2608.03207v1)

**作者:** Hoseong Tae `[一作]` (Yonsei University), Jong-Seok Lee `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种测试时的通用对抗贴纸 DRIFT，通过只攻击流匹配视觉语言动作模型（VLA）的第一步去噪速度场，导致机器人执行失败。

**💡 创新点**

创新点在于发现对抗只需扰动第一步即可获得更强且更高效的攻击；通过梯度冲突分析解释了“少即多”的现象，并且展示了单步攻击在性能与计算成本上优于传统动作/嵌入空间攻击。

**🔧 技术方法**

使用了白盒对抗贴纸优化（PGD 梯度下降）、流匹配去噪 ODE、梯度冲突可视化与分析等技术。

**📊 数据集**

使用了 LIBERO 基准中的四套任务（Spatial、Goal、Object、Long）以及相应的腕部图像数据。

**📈 对比分析**

与随机贴纸、UADA、EDPA 等基线在同一贴纸尺寸下对比，DRIFT 在 π_0 模型上实现近 100% 的攻击成功率（ASR），在 π_0.5 上使用 64px 贴纸也超过 90%；整体上显著优于现有方法。

**⚠️ 局限性**

局限性包括：需要白盒梯度访问模型、贴纸位置固定、跨模型迁移效果有限、对抗阈值敏感，且仅针对流匹配 VLA，未提供对应的防御方案。

---

## 383. Adaptive Sampling for Automated Post-Disaster Rapid Damage Assessment via Level-Set Cost-Aware Bayesian Optimization

**arXiv ID:** 2608.02868 | [PDF](https://arxiv.org/pdf/2608.02868v1)

**作者:** Boyang Xu `[一作]` (Arizona State University), Hao Yan `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种成本感知贝叶斯优化框架，结合层集估计和Ordinal Deep Kernel Gaussian Process（ODGP），用于无人机等自主采样器快速、成本低地完成灾后损伤评估；

**💡 创新点**

创新点包括：①设计ODGP模型，既能捕捉复杂空间与建筑特征相关性，又能处理序数损伤等级；②提出成本感知层集采样函数，在探索、利用与行驶成本之间进行平衡，专为边界/级别检测而设计；③在合成与高保真模拟海啸场景中验证该框架优于传统GP与随机采样；

**🔧 技术方法**

采用贝叶斯优化、深度核高斯过程、序数回归似然、稀疏变分推断、成本感知层集采样函数，以及无人机自主轨迹规划等技术；

**📊 数据集**

使用两类数据集：1）二维合成高斯热点网格；2）R2D 500年级海啸模拟数据（Seaside, Oregon），包含1000栋建筑的六维特征及四级序数损伤等级；

**📈 对比分析**

与GP/ODGP的随机采样、普通采样（不考虑成本）进行对比，评价指标为准确率、加权F1和累计行驶成本。结果显示ODGP‑CAF在100步后实现85%准确率、F1≈0.81，并在成本上最低（约44.69），显著优于其他方法；

**⚠️ 局限性**

局限性包括：仅使用贪婪单步贝叶斯优化，缺乏多步前瞻规划；深度核模型解释性有限；在真实灾区可能需处理更复杂的感知与导航约束；稀疏观测下模型仍可能受限。

---

## 384. Federated generative event models for tokenized electronic health records

**arXiv ID:** 2608.02939 | [PDF](https://arxiv.org/pdf/2608.02939v1)

**作者:** Michael C. Burkhart `[一作]` (University of Chicago), Brett K. Beaulieu-Jones `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文基于三家医院的ICU电子健康记录数据，构建并预训练了生成事件模型（GEM），并评估了联邦学习与集中式训练在跨机构迁移中的性能差异。

**💡 创新点**

创新点在于：①采用CLIF统一语义的tokenization实现多机构数据的共享；②系统比较FedAvg、FedAvgM、FedAdam等联邦聚合策略对GEM迁移性能的影响；③通过对12个临床预测任务的ROC‑AUC/PR‑AUC对比，验证GEM的高可迁移性与联邦训练的可行性。

**🔧 技术方法**

使用的技术包括Transformer‑based GEM（Llama‑3.2架构，76.9M参数）、FedAvg、FedAvgM、FedAdam聚合、RoPE+NEFTune正则化、token化和数值分箱、Logistic回归下游分类、ROC‑AUC/PR‑AUC评估及Bootstrap置信区间。

**📊 数据集**

数据集为122,251例ICU住院记录，来源于University of Chicago Medical Center、Northwestern Medicine和MIMIC‑IV（CLIF标准化），包含12个临床事件标签。

**📈 对比分析**

比较方法为在同一下游Logistic回归模型下，评估本地、跨站、集中式和联邦训练的预测性能。结果显示：GEM在跨站转移时平均惩罚仅0.025 ROC‑AUC；FedAvg/FedAvgM在5–10轮即可接近集中式GEM，且与本地GEM的差距很小；FedAdam表现较差；集中式训练提升有限。

**⚠️ 局限性**

局限性包括：仅涉及三家成人ICU机构；仅测试单一Transformer架构和tokenization策略；使用MIMIC‑IV学习的词表和分箱在其他站点可能不完全适配；未探索多机构、多模态或个性化适应技术；结果在其他临床领域或更大规模联邦设置下的泛化性未知。

---

## 385. Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting

**arXiv ID:** 2608.03116 | [PDF](https://arxiv.org/pdf/2608.03116v1)

**作者:** Sergio A. Esteban `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于可微分仿真器的直接多射击（DSMS）框架，将仅满足运动学的参考轨迹转化为满足完整动力学、接触、摩擦、自碰撞和驱动力限额的全身可行轨迹，并将其用于强化学习训练与硬件零样本迁移。

**💡 创新点**

核心创新点在于：①不需要预先给出接触序列或显式接触约束；②直接将仿真器的离散动力学作为非线性规划的状态转移；③通过多射击技术降低优化维度，兼顾全身动力学一致性与约束满足；④将优化得到的轨迹作为 RL 的高质量参考，显著提升学习收敛速度和轨迹跟踪精度。

**🔧 技术方法**

技术手段包括：可微分仿真器（MuJoCo）提供的刚体动力学和接触求解；直接多射击（DSMS）形式的非线性规划（NLP）；IPOPT + HSL 线性方程求解；PPO 训练 RL 控制器；基于多射击的再生窗规划（receding horizon control）以及命令约束的周期性步态库生成。

**📊 数据集**

使用的主要数据集：BONES‑SEED 人类动作捕捉数据作为参考轨迹；Unitree G1 机器人作为硬件平台；以及对比实验中使用的多种预处理或优化方法（如 SRB、KD、DSMS、OmniRetarget、DynaRetarget 等）。

**📈 对比分析**

与现有方法比较：在“超级英雄后空翻”任务中，DSMS 的训练收敛最快（≈2000 次迭代），成功率 98.7%，关节/姿态跟踪误差显著低于 OmniRetarget、DynaRetarget、SPARK 等；在硬件上实现的 180° 跳转与爬行任务实现零样本迁移，显示出更高的可执行性与稳定性；同时 DSMS 的优化时长约 12 分钟，明显优于竞争方法。

**⚠️ 局限性**

局限性包括：①对可微分仿真器（如 MuJoCo）高度依赖，若采用其他仿真器需额外适配；②优化仍需显著计算资源，尤其对复杂机器人模型；③当前实验集中在 humanoid 结构，扩展到其他机械结构需要进一步验证；④在极端不确定环境下的鲁棒性仍待通过更丰富的域随机化或在线学习来提升。

---

## 386. Electro-Magnetic Decoupling Preconditioner for Eddy Current Problems with External Circuit Coupling

**arXiv ID:** 2608.03350 | [PDF](https://arxiv.org/pdf/2608.03350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 387. Finite-valuation approximable structures: a solution to the Jung--Tix problem of probabilistic powerdomains

**arXiv ID:** 2608.03073 | [PDF](https://arxiv.org/pdf/2608.03073v1)

**作者:** Yuxu Chen `[一作]` (Sichuan University), Zhenchao Lyu `[通讯]` (Sichuan University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“有限取值可近似域（finite‑valuation approximable domains）”这一新的连续域子类，并证明该子类在子概率与概率取值幂域（valuation powerdomains）以及函数空间上保持封闭性，从而给出了Jung–Tix兼容性问题（Jung–Tix problem）的正面解答；

**💡 创新点**

创新点在于引入了以有限取值空间（subprobability valuation域）为基底的近似序列，构造了一套新的有限近似身份（finite‑valuation approximate identity），并通过“有限分离饱和定理”和“统一核提升定理”实现了对取值幂域与函数空间的闭包性；

**🔧 技术方法**

核心技术包括：1）构造半群Φ_t 对有限偏序的子概率取值空间进行逐步“侵蚀”并保持随机化顺序；2）利用凸多面体与Hasse图的对偶性，证明有限取值空间是FS域；3）采用随机化格化逼近与核提升，得到可数基FS域的递增逼近序列；4）利用复合与重提技术，将局部有限近似推广到任意对象，完成闭包证明；

**📊 数据集**

无数据集，纯理论证明与构造；

**📈 对比分析**

与传统的RB域、bc域等已有的连续域类进行对比，证明该子类包含所有可计数基的bc域、所有可计数基的FS域，但与RB域不兼容；通过示例（如B_5、D_4）展示两类间的不相容性；性能指标在理论上表现为闭包性质与计算机可构造的可数基FS域；

**⚠️ 局限性**

主要限制：1）要求域可计数基，尽管提出了非计数基扩展(ωFVA)，但仍需额外技术；2）该子类虽封闭，但对更广泛的连续域或无基域的适用性尚未完全覆盖；3）证明复杂度高，依赖大量构造性细节，实际实现难度大。

---

## 388. PLAN: Parallel Liquid-Inspired Approximation Network for Efficient Representation Learning in Flexible Job Shop Scheduling

**arXiv ID:** 2608.03041 | [PDF](https://arxiv.org/pdf/2608.03041v1)

**作者:** Dhivya Dharshini Kannan `[一作]` (Singapore Institute of Technology), Anupam Trivedi `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量化的并行液态启发式表示学习框架PLAN，用于强化学习求解可变工序车间调度问题

**💡 创新点**

创新点在于把连续液态神经网络的状态演化通过Euler近似转为可并行化的离散更新，同时仅用轻量化注意力做全局上下文聚合，显著降低参数量和推理延迟

**🔧 技术方法**

采用液态神经网络（LNN）离散化、并行近似、轻量化多头注意力、策略梯度PPO以及可选的稀疏场景聚合模块SPM

**📊 数据集**

使用SD1、SD2、SD3、Brandimarte、Hurink等公开基准及多层次动态调度数据集，涵盖从10×5到200×5规模的实例

**📈 对比分析**

与SOTA DANIEL、HGT以及OR‑Tools对比，在所有实验中PLAN平均缩短1.2%–2.3%的完工时间、10.2%最优场景提升、并将推理延迟降低13.2%–31.7%，模型大小仅占基线的22%–47%

**⚠️ 局限性**

缺点包括：对极大规模或高动态约束的鲁棒性尚未全面验证，且仍需进一步理论分析其收敛与稳定性

---

## 389. FLARE: Few-shot Learning-based Adaptive Reflective Engine

**arXiv ID:** 2608.02919 | [PDF](https://arxiv.org/pdf/2608.02919v1)

**作者:** Dhanasekar Sundararaman `[一作]` (Microsoft), Minjie Li `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FLARE，一种基于错误驱动的自适应反射引擎，用少量 few-shot 参考例子迭代优化 LLM 提示。

**💡 创新点**

创新点在于将反射与实例级错误信号相结合，利用 LLM 的“思考”阶段直接诊断每个失败并生成针对性的提示修正，显著提高了数据效率和稳定性。

**🔧 技术方法**

技术包括：Azure OpenAI GPT‑5 系列的元优化器、DSPy 框架、错误感知反馈循环、温度为 1.0 的随机推理、滑动窗口历史跟踪以及结构化提示模板。

**📊 数据集**

使用的数据集有：GoEmotions（多标签情感分类）、τ²‑bench（工具调用）、HotPotQA、MedQA、2WikiMultiHopQA（检索增强推理）。

**📈 对比分析**

与 GEPA、Promptomatix、OpenAI Prompt Optimizer 对比，FLARE 在所有任务‑模型组合上取得最高分；例如在 GPT‑5.1 上 GoEmotions 的 micro‑F1 由 37.5% 提升至 52.7%（+15.3），HotPotQA 上 GPT‑5‑Chat 的得分从 38.0 提升至 52.2（+14.2），工具调用上提升至 87.0%（+8.4）。同时 FLARE 在 GoEmotions 上仅需 100 条验证样本即可达到峰值，显示出更好的数据效率和更低的方差。

**⚠️ 局限性**

局限性：对不同任务的适用性仍有差异，某些多步检索任务（如 2WikiMultiHopQA）提升有限；增大 GEPA 的搜索预算并不能显著提升其性能，说明 FLARE 的优势主要来自搜索质量而非规模；此外在极大验证集或更复杂任务上可能需要进一步改进。

---

## 390. Towards Robust Tool Use in Agents via Experience-Driven Adaptive Guidance

**arXiv ID:** 2608.03403 | [PDF](https://arxiv.org/pdf/2608.03403v1)

**作者:** Can Wang `[一作]` (Key Laboratory of Digital Service Computing Technology and Systems), Zhiying Tu `[通讯]` (Key Laboratory of Digital Service Computing Technology and Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于经验的自适应工具使用指导机制（ExpG），通过经验获取、精炼和复用三阶段持续提升大型语言模型在工具交互过程中的鲁棒性。

**💡 创新点**

创新点在于：①将工具调用视为可解释的、多维度经验；②利用等价类分割和多重抽样实现高效、代表性的经验选取；③对经验进行过滤、总结生成动态与稳定两类指导；④通过将指导注入提示或工具模式，形成可自演化的工具使用策略。

**🔧 技术方法**

主要技术包括：LLM-as-Judge 评估器对历史轨迹进行多维度评分与自然语言解释；等价类分割与层级抽样算法选择代表经验；经验过滤、统计分析与链式思维总结生成工具级指导；动态/稳定指导在推理时以轻量提示或 schema 约束形式复用。

**📊 数据集**

使用了三大基准数据集：MetaTool（工具选择）、API-Bank（工具调用）和 BFCL-V3（响应生成），覆盖工具使用的各个阶段。

**📈 对比分析**

与 No Method、Few-shot、DRAFT、Mem0 等基线对比，ExpG 在 MetaTool、API-Bank、BFCL-V3 上均取得最高 Avg@3 和 Pass@3；尤其在小模型上，ExpG 能缩小与大模型的性能差距，并在噪声或挑战性环境中实现显著提升。

**⚠️ 局限性**

局限性包括：对评估器和总结器的 LLM 质量高度依赖；经验获取与过滤过程可能产生高计算成本；在全新工具或极端未见情景下，指导效果可能受限；且方法假设工具的接口与响应相对稳定，面对频繁变更的工具可能需要额外适配。

---

## 391. Shorter Reasoning, Earlier Answers? An Evaluation of Reasoning Interfaces

**arXiv ID:** 2608.03401 | [PDF](https://arxiv.org/pdf/2608.03401v1)

**作者:** Francesca Carlon `[一作]` (Vrije Universiteit Brussel), Andres Algaba `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在大语言模型中通过提示或训练设置来控制推理长度的方法，并评估其在提前截止点对答案质量的影响。

**💡 创新点**

提出并系统化了匹配时间窗（matched‑horizon）评估框架，能够区分模型提前停止、未完成前缀的答案以及概率质量，从而更清晰地揭示不同控制方式的实际效益。

**🔧 技术方法**

使用了候选词logit强制答案读取、token‑exact重放、概率质量评估（Brier、log‑loss）、多次生成复制、以及vLLM+flashinfer生成流水线。

**📊 数据集**

评估数据集包括 GPQA Diamond（198道多项选择），MMLU‑Pro（500道子集，10项多项选择）和 Omni‑MATH‑2（4,181道开放式问题）。

**📈 对比分析**

通过对比同一问题在相同推理长度下的两种设置（如数值/简洁提示 vs 普通提示，低/中 effort vs 高 effort），发现数值/简洁提示仅显著缩短推理长度且在同等长度下准确率差异不显著；简洁/早答提示在 512 token 时提升约 3.8% 终端准确率；低 effort 在 512 token 时能显著提高准确率（14.5–26.3%），但大部分优势来自更早停止；高 effort 若允许完成则可获得更高准确率。

**⚠️ 局限性**

局限性包括：仅测试两类模型（Qwen3、gpt‑oss）和两种多项选择基准；每条推理仅生成三次复制，样本随机性有限；强制答案读取可能偏倚未完成前缀准确率；以及仅评估在固定截止点的性能，未覆盖更广泛的实时推理场景。

---

## 392. SRAP: SVD-Refined Adversarial Perturbations for Imperceptible Face-Swap Defense

**arXiv ID:** 2608.03395 | [PDF](https://arxiv.org/pdf/2608.03395v1)

**作者:** Sungwon Cho `[一作]` (Seoul National University), Myungjoo Kang `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SRAP方法，用来在不易察觉的前提下对面部图像添加扰动，防止其被用于深度伪造。

**💡 创新点**

创新点在于将逐通道截断SVD降维与身份重要性引导的稀疏掩模相结合，既抑制高频残差又将噪声聚焦于身份敏感区域。

**🔧 技术方法**

采用PGD对抗扰动、SVD低秩重构、随机探测估计身份重要性、重要性掩模、以及JPEG压缩鲁棒性测试等技术。

**📊 数据集**

实验使用CelebA-HQ和VGGFace2-HQ高分辨率人脸数据集，并在SimSwap面部置换模型上进行评估。

**📈 对比分析**

与AdvDM、MIST、PhotoGuard、SDST及FaceShield等基线比较，SRAP在身份抑制效果上排名第二，但在LPIPS、PSNR、SSIM等保真度指标上表现最优，并在JPEG压缩下保持较好的鲁棒性。

**⚠️ 局限性**

局限性包括仅在SimSwap上验证，样本量有限，仅测试单一压缩方式，未覆盖其他置换/扩散模型及后处理攻击，掩模比例的选取仍需经验调参。

---

## 393. TimeRLM: Recursive Language Models Enable Precise Anomaly Localization in Long-Context Time-Series

**arXiv ID:** 2608.03391 | [PDF](https://arxiv.org/pdf/2608.03391v1)

**作者:** Nicolas Zumarraga `[一作]` (ETH Zürich), Robert Jakob `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 TimeRLM，一种递归语言模型，可通过代码与外部时间序列交互，实现长上下文异常定位。

**💡 创新点**

创新点在于将时间序列作为可查询外部上下文，允许多轮交互与递归子代理，从而显著提升精确检索能力。

**🔧 技术方法**

使用 LLM 后端（Qwen3.5‑4B、GPT‑5.5 等）、Python REPL 代码执行、可视化绘图、强化学习（GRPO）进行后训练。

**📊 数据集**

数据集包括自研合成长序列异常基准 AnomalyXL、真实生产遥测 ARFBench Tier 2，以及 ECG（LTAF）和睡眠 PSG 的零样本测试集。

**📈 对比分析**

与单通道 TSLM（ChatTS、OpenTSLM、ITFormer、Toto）和经典异常检测模型对比，TimeRLM 在 AnomalyXL‑Localize 上 IoU 最高（0.682），在 ARFBench Tier 2 上精度和宏 F1 最高（61.4%/57.9%），并在 ECG/睡眠数据上保持较好零样本性能。

**⚠️ 局限性**

局限性包括合成异常形态单一、真实异常多样性不足，且 RLM 需要较多推理轮次；基准与大模型 TSLM 直接对比存在规模和预训练差异。

---

## 394. Residual Flow Matching with Dynamic Cross-Interaction for 3D Multi-Person Motion Prediction

**arXiv ID:** 2608.03379 | [PDF](https://arxiv.org/pdf/2608.03379v1)

**作者:** Wei Wei `[一作]` (Shandong University), Ruixuan Yu `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种先用确定性粗先验（DCP）生成运动粗估，再用残差流匹配（RFM）对残差进行连续时间流生成的3D多人运动预测框架。

**💡 创新点**

创新点在于把运动残差作为连续时间流的目标，并在ODE进程中引入动态交叉交互（DCI）按时间调节交互强度，同时采用双分支关节-运动融合与全序列时间对齐，显著提升结构一致性和多模态预测质量。

**🔧 技术方法**

技术包括：确定性粗先验模块、残差流匹配（Flow Matching）、动态交叉交互、双分支关节-运动网络、端点参数化的多模态头以及全序列时间对齐。

**📊 数据集**

使用3DPW、CMU‑Syn、Mix1/2、AMASS、MuPoTS‑3D等公开多人动作数据集进行训练与评估。

**📈 对比分析**

与T2P、TBIFormer、JRT、EMPMP及CoMusion等基线对比，在MPJPE、JPE、APE、FDE、VIM等多项指标上实现SOTA或接近SOTA，尤其在多人密集场景和跨域迁移任务中表现突出。

**⚠️ 局限性**

局限包括ODE积分导致推理延迟、DCI调度依赖预设超参数且缺乏场景约束、以及模型固定人数限制。

---

## 395. Can Text-to-Image Models Draw from the Right Frame of Reference?

**arXiv ID:** 2608.03357 | [PDF](https://arxiv.org/pdf/2608.03357v1)

**作者:** Zheyuan Gu `[一作]` (ERNIE Team, Baidu Inc.), Zhenyu Zhang `[通讯]` (ERNIE Team, Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个帧参考依赖的文本到图像生成基准FoREval，并评估了22款模型在相同场景下的Cam与FoR两种提示的表现差异。

**💡 创新点**

提出了匹配Cam–FoR提示对、三层任务难度设计以及基于视觉反馈的VLM门控重写策略，以量化并缓解模型在对象本身方向描述下的空间推理误差。

**🔧 技术方法**

使用基于模板的提示生成、自动评估器（OWL‑ViT+SAM3.1+DA3+Qwen3.6‑27B）、视觉反馈门控重写以及对照实验等技术。

**📊 数据集**

利用从现有T2I基准（VISOR、GenEval、SpatialGenEval等）抽取的对象词汇表，生成1200个Cam–FoR提示对，覆盖三种难度层级。

**📈 对比分析**

在Cam提示下模型平均精度为约50%，在FoR提示下下降至约29%，最高FoR精度仅44%；VLM门控重写将FoR平均精度提升至约29%，相对基线提升约4个百分点。

**⚠️ 局限性**

模型对对象左/右轴的帧转换误差仍高，尤其在镜像/反向映射场景；重写策略仅为训练无关的轻量级补救，无法从根本上解决帧参考理解不足的问题。

---

## 396. FACTWASH: Catching AI Rewrites That Wash Hearsay into Fact

**arXiv ID:** 2608.03372 | [PDF](https://arxiv.org/pdf/2608.03372v1)

**作者:** Alex Kwon `[一作]` `[通讯]` (Independent Researcher), Alex Kwon (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种写入时门（factwash）来检测并防止知识库写入过程中的事实洗涤（factwashing），即在压缩或摘要时丢失可信度标记、来源或时间信息。

**💡 创新点**

创新点在于：①将事实洗涤拆解为七个可检测的语言属性（否定、条件、让步、归因、时间、脆弱性、截断）并实现零依赖的确定性检测；②区分闭类属性可用词表完成、开放类属性需LLM辅助的“见证者”模型；③构建公开的检测工具、标注语料与基准评测，并证明在真实业务邮件和对话中该门能显著发现写入错误。

**🔧 技术方法**

技术主要包括基于词表的子串匹配检测、可选的LLM见证者（返回布尔值与指示词），以及与原始文本对齐的内容词重叠策略；门的决策策略为硬编码的判定表。

**📊 数据集**

使用的数据集包括：① 57,891 句子来自生物医学、论文、百科和新闻的 4 个公开语料库（用于词表挖掘与评估）；② 47,705 句子来自政治新闻的归因标注集；③ 105,596 句子总数的多源标注集；④ 1,415 条从 Enron 邮件线程提取的真实写入实例用于实证评估。

**📈 对比分析**

与直接 LLM 判定（如 Claude‑Haiku‑4.5 / Claude‑Sonnet‑5）比较，门在真实写入样本上取得 0.73 精确率、0.95 召回率，而 LLM 判定仅 0.30 召回率；在开放类属性上加入见证者可将词表召回率提升约 15‑17 点；在业务邮件中的错误率极低（≈7 %），而在含听闻语境的对话中错误率高达 55 %。

**⚠️ 局限性**

局限性包括：① 召回率受词表覆盖度限制，尤其对开放类属性无法完全覆盖；② 仅支持英文、子串匹配，难以处理同义/语义变体；③ 对“范围扩大”类错误缺乏语义/世界知识支持；④ 实验样本量有限、标注者单一、交叉验证不足；⑤ 需要手动维护词表与阈值，且对不同数据域的迁移需重新校准。

---

## 397. LPV Control for Dynamic Power Capping in High-Performance Computing under Mixed Workloads

**arXiv ID:** 2608.03367 | [PDF](https://arxiv.org/pdf/2608.03367v1)

**作者:** Mohamed Abdeldjalil Maziz `[一作]` (University Grenoble Alpes), Sophie Cerf `[通讯]` (University Lille)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在高性能计算环境中设计并评估了一套基于反馈的动态功率限制框架，比较了增益调度PI控制器与多极LPV H∞控制器的性能。

**💡 创新点**

创新点在于引入工作负载指标β作为调度参数，利用多极LPV方法实现对内存与计算阶段间的平滑插值控制，从而在负载切换时保持系统稳定与鲁棒性，显著优于传统PI控制。

**🔧 技术方法**

采用了线性参数变（LPV）模型识别、β调度的增益调度、基于H∞约束的LMI求解、心跳式进度监测与RAPL功率上限接口等技术。

**📊 数据集**

利用STREAM（内存密集）和Embarrassingly Parallel（计算密集）基准测试数据进行模型识别与仿真评估。

**📈 对比分析**

通过仿真比较两种控制器在平稳与激进工作负载切换场景下的跟踪误差、RMSE、Fit百分比等指标，结果显示LPV H∞控制器在两种情况下均取得更低误差、更高Fit，体现出更好的稳健性。

**⚠️ 局限性**

限制在于假设调度参数β已知且无延迟、仅使用SISO简化模型、未在真实硬件上验证、未考虑RAPL的延迟与非线性等因素。

---

## 398. Multi-Task Multi-Frame Visual Piano Transcription

**arXiv ID:** 2608.03419 | [PDF](https://arxiv.org/pdf/2608.03419v1)

**作者:** Yonghyun Kim `[一作]` (Georgia Institute of Technology), Alexander Lerch `[通讯]` (Georgia Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出V2N系统，实现视频端到端钢琴乐谱转录，预测起始、释放、键盘保持状态和力度；

**💡 创新点**

创新点在于：①针对释放事件单独设计offset头并配合1秒长时序上下文；②多任务、多帧监督提升音符持续和力度预测；③通过offset引导的解码实现物理键盘释放时长；

**🔧 技术方法**

技术包括S2S视觉特征提取器（ResNet-18+滑坡先验）、三层Conformer卷积骨干、并行BiLSTM头、soft三角核、多帧损失、数据增强等；

**📊 数据集**

使用PianoVAM（107段视角视频）和R3（31h实践录音）两个公开数据集进行训练与评测；

**📈 对比分析**

与S2S、V2R、Li et al.、PPAN等基线对比，V2N在起始、释放、力度以及综合+Off+Vel指标上均达或超越现有最高分，尤其在释放和力度方面提升显著；

**⚠️ 局限性**

局限性：跨数据集迁移效果差，模型对键盘几何定位高度依赖；未能预测延音踏板控制；

---

## 399. DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces

**arXiv ID:** 2608.03451 | [PDF](https://arxiv.org/pdf/2608.03451v1)

**作者:** Boyan Li `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了 DataSpace benchmark，旨在评估数据代理在包含 CSV、JSON、SQLite、Markdown、PDF、视频等多模态、跨语言工作空间中生成可验证的完整表格结果的能力。

**💡 创新点**

创新点在于统一了工作空间范围、输出契约和确定性评估三大核心属性，并通过执行驱动的构造框架与语义感知评估器将可执行 Text‑to‑SQL 任务转化为跨语言、异构工作空间的完整表格检索问题。

**🔧 技术方法**

采用了跨语言转换、约束感知关系采样、模态路由与文档渲染、人工审核与任务修复等技术，并构建了一个头部不变、类型精准、行序可选的确定性评估器。

**📊 数据集**

任务来源于 EHRSQL 与 BULL 两个可执行 Text‑to‑SQL 数据集，随后通过上述框架生成了 410 个跨语言、异构模态任务，总计 7,439 份文件，约 15 GB。

**📈 对比分析**

实验在六个前沿多模态模型与五个代理框架下进行，最佳模型 Grok 4.5 在 66.34 % 的任务准确率；模型与框架差异可达 15.36 点，且多模态整合与连接操作是主要的性能瓶颈。

**⚠️ 局限性**

主要限制包括性能仍未饱和、跨模态融合与关联（尤其是视频与长文档）导致准确率下降、任务规模与多样性对模型的挑战，以及对人工审校的依赖。

---

## 400. Cross-cultural evaluation of taste-sound correspondences in AI-generated music

**arXiv ID:** 2608.03433 | [PDF](https://arxiv.org/pdf/2608.03433v1)

**作者:** Matteo Spanio `[一作]` (University of Padova), Antonio Rodà `[通讯]` (University of Padova)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究对AI生成音乐与味觉的跨文化对应性进行评估，首先在阿根廷、意大利和日本三国开展在线实验，比较fine‑tuned MusicGen模型与基准模型的偏好，并让受试者在十二个味觉、情感和温度描述上对同一音频进行语义评分；随后结合声学特征分析和因素分析，探讨不同文化群体在音频-味觉映射结构上的差异；

**💡 创新点**

创新点在于首次将AI生成的“调味音乐”在跨文化背景下进行系统验证，并通过对响应风格的校正与语义结构的对比，区分了尺度使用偏差与感知组织差异；同时将语义评分与音频声学特征关联，揭示文化对音频-味觉对应的重组机制；

**🔧 技术方法**

采用MusicGen文本到音乐生成模型的fine‑tuned版本进行音频合成；使用PsyToolkit平台收集在线问卷；采用非参数检验（Wilcoxon、Kruskal–Wallis）、线性混合模型、Response‑Style校正（within‑subject z‑scoring）、Tucker一致性系数以及最大似然提取的探索性因子分析等统计技术；

**📊 数据集**

样本共361名受试者，分布于阿根廷104名、意大利117名、日语140名；音频刺激为100段fine‑tuned MusicGen生成的短片（每种味觉25段），与原始意大利数据合并使用；

**📈 对比分析**

比较方法包括：对fine‑tuned与基准模型的偏好采用Wilcoxon符号秩检验和线性混合模型；语义评分通过三因素（国别×提示×描述符）Type III ANOVA和对应的混合模型评估；对响应风格进行z‑scoring后再次分析，发现跨国评分差异大多源于尺度使用，而保留的交互项表明感知结构仍存在显著差异；结果显示fine‑tuned模型在意大利和阿根廷获得显著偏好，而在日本无效；评分结构在三国呈现不同的味觉-音频对应模式；

**⚠️ 局限性**

局限性包括：样本在年龄、性别、音乐/饮食经验等背景变量上不完全相同，可能与文化因素混杂；响应风格校正方法简单，未采用更复杂的测量不变性或潜变量模型；因子分析为探索性，未检验随机斜率；使用英文提示导致模型可能偏向西方味觉概念，影响跨文化可比性；最后，sour–bitter映射不一致部分归因于音频的感知歧义，进一步验证仍需更细致的声学与语义匹配研究。

---

## 401. Dual-domain U-Nets with embedded back projection operators for motion-resolved 4D CBCT reconstruction

**arXiv ID:** 2608.03430 | [PDF](https://arxiv.org/pdf/2608.03430v1)

**作者:** Ivo Herzig `[一作]` (Zurich University of Applied Sciences ZHAW), Lukas Lichtensteiger `[通讯]` (Zurich University of Applied Sciences ZHAW)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种基于深度学习的双域U-Net模型，能够直接从临床常规自由呼吸CBCT投影数据中恢复出运动分辨的4D CBCT图像，无需呼吸信号或投影分箱。

**💡 创新点**

创新点在于将不可训练的后投影算子嵌入U-Net的跳跃连接，实现投影域与体积域的端到端联合学习，同时一次性预测最大吸气静态体积和完整的10相位运动场，从而在极短扫描时间（仅6 s）下完成4D重建。

**🔧 技术方法**

采用了U-Net架构的投影域编码器、体积域解码器、非可训练后投影层、Swish激活、MAE损失加弯曲能量正则化等技术；训练使用Adam优化器并在模拟数据上进行监督学习。

**📊 数据集**

训练与验证使用基于4D CT的合成数据集（包含560训练、110验证、110测试样本），测试包括13例临床30 s自由呼吸扫描，以及各自的60 s和6 s HyperSight扫描。

**📈 对比分析**

与传统3D SART-TV重建进行定量比较（RMSE、PSNR、SSIM、Dice系数）和临床专家评估，结果显示在模拟数据上RMSE、PSNR、SSIM与SART-TV相当，临床数据上显著减少运动条纹并提升肿瘤与食管可见度（专家选择率约59%/47%）。

**⚠️ 局限性**

局限在于网络使用像素级MAE损失导致重建图像相对模糊，且对不同扫描协议的鲁棒性需要进一步的超参数调优；缺乏真实病人运动的ground truth，运动验证仅通过胸腔运动曲线间接推断。

---

## 402. SGFormer: Structure-Guided Transformer for Robust Local Feature Matching

**arXiv ID:** 2608.03423 | [PDF](https://arxiv.org/pdf/2608.03423v1)

**作者:** Runyu Zhu `[一作]` `[通讯]`, Runyu Zhu

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SGFormer，一种结构引导的 Transformer，用于解决检测器自由局部特征匹配中的注意力分散问题，实现半稠密粗细匹配管线。

**💡 创新点**

核心创新是 Triple-Structure-Attention（TSA）模块，它利用浅层结构特征与相对空间编码引导注意力，同时结合 RoPE 以增强几何一致性；以及轻量化的 Positional Patch Embedding 与 FPN 级联。

**🔧 技术方法**

采用层次化 Transformer 主干、TSA、交叉/自注意力、FPN 解码、软空间正则化、交叉熵+方差加权 L2 损失；评估指标包括 MMA、AUC、HCR、椭圆面积/方差等。

**📊 数据集**

在 HPatches、MegaDepth（1500 对）和 Aachen‑Day‑Night 三个公开基准上进行训练与评估。

**📈 对比分析**

与基准 detector‑free 方法 LoFTR、ASpanFormer、SuperPoint+SuperGlue 等相比，SGFormer 在 HPatches 的 MMA 1/3/5px 分别提升 4.84%/1.10%/2.15%；在 MegaDepth 的 AUC@5° 提升 2.55%，在 Aachen‑Day‑Night 夜间 0.25m/2° 的定位准确率达到 100%，总体表现处于前沿。

**⚠️ 局限性**

在极端视角变化、尺度失配或非刚性变形场景下，浅层结构引导可能不再可靠，导致注意力聚焦不足；此外，仍可能出现少量非对应点误检。

---

## 403. DUD: Decoupled Update Dynamics for Reliable Uncertainty Quantification in Large Language Models

**arXiv ID:** 2608.03411 | [PDF](https://arxiv.org/pdf/2608.03411v1)

**作者:** Yixin Bu `[一作]` (Nanjing University of Aeronautics and Astronautics), Piji Li `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Decoupled Update Dynamics (DUD) 框架，通过解耦 Transformer 的 FFN 与 Attention 贡献进行主动机制诊断，以实现更精准的 Uncertainty Quantification。

**💡 创新点**

创新点在于将 FFN 与 Attention 的更新动力学分别测量、构造双流动态特征，并通过因果干预恢复单个模块的概率恢复分数，揭示内部机制冲突和脆弱性作为不确定性的本征信号。

**🔧 技术方法**

技术包括：多阶段因果干预（Clean、Corrupted、Patch）、噪声注入、模块级恢复得分计算、归一化双流特征、基于 MLP 的 DUD‑Probe、AUROC 等评估。

**📊 数据集**

数据集覆盖四大知识推理基准：HaluEval、SQuAD、TriviaQA、HotpotQA，并在 Llama-3.1-8B、Qwen2.5-7B、Gemma-2-9B 三大开源 LLM 上验证。

**📈 对比分析**

与多种无训练或训练型基线（PPL、LN‑Entropy、Semantic Entropy、SAR、LLM‑Check、SAPLMA、SEP、ICR Probe）比较，DUD‑Probe 在 AUROC、ECE、PRR 等指标上均高出 5‑20% 甚至接近 0.95，跨数据集泛化也最优。

**⚠️ 局限性**

局限性在于需要访问内部激活并执行多次前向传递，导致推理延迟；且仅适用于可白盒的模型，无法直接应用于封闭 API（如 GPT‑4）。

---

## 404. Distilled Roads: Generalisable Road Network Extraction Across Sensors, Resolutions, and Region

**arXiv ID:** 2608.03407 | [PDF](https://arxiv.org/pdf/2608.03407v1)

**作者:** Sanayya `[一作]` (SatSure Analytics India Pvt Ltd), Ashwathi Nambiar `[通讯]` (SatSure Analytics India Pvt Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于持续知识蒸馏和拓扑感知监督的道路网络提取框架，能够在不同分辨率、传感器和地区之间实现单模型鲁棒分割。

**💡 创新点**

将道路提取视为持续适应问题，采用分辨率递减课程的跨分辨率蒸馏、拓扑约束损失以及多传感器多地区训练，显著提升跨域泛化，而不依赖更复杂的模型结构。

**🔧 技术方法**

使用交叉分辨率知识蒸馏、DilationBlock结构、骨架回忆损失、交叉信息损失、Dice/Focal/BCE组合损失、数据域随机化增强以及大尺寸拼接补丁等技术。

**📊 数据集**

使用DeepGlobe（高分辨率）、专有航空数据、Global‑Scale（1 m商业卫星）和City‑Scale（谷歌地图）等数据集进行训练与评估。

**📈 对比分析**

与SAM‑Road、RNGDet++等基线在全球域内外和城市域上对比，F1分别为86.02、74.47、85.69，APLS分别为68.55、55.22、83.16，且推理速度提升3倍、参数量仅31M。

**⚠️ 局限性**

仍难以识别极细道路、树冠遮挡段落，以及低分辨率下交叉细节缺失；此外教师模型同时具备高分辨率和单地区特征，难以分离其对性能的具体贡献。

---

## 405. RoboReact: Agentic Skill Distillation from Generated Egocentric Videos for Generalizable Whole-Body Manipulation

**arXiv ID:** 2608.03387 | [PDF](https://arxiv.org/pdf/2608.03387v1)

**作者:** Shuliang He `[一作]` (Chinese University of Hong Kong), Guiliang Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

基于单帧自我视角RGB‑D图像与语言指令，利用视频生成模型生成的人类交互视频，提取几何保持的交互关键帧，并通过对象中心的重定位与冻结的VLM在校准回放中进行结构化优化，最终得到可执行的全身多手臂操纵策略，并在真实的Unitree G1机器人上执行；

**💡 创新点**

①首次将非度量化的生成视频作为全身操纵的交互先验；②采用对象中心在线重定位与VLM驱动的离线编辑实现从视觉先验到可执行技能的闭环转化；③证明生成先验的质量与强化学习或演示驱动方法相当；

**🔧 技术方法**

视频生成模型（如Seedance）、VLM（Frozen VLM作为优化代理）、深度相机与3D重建、离线结构化编辑算法、全身控制器（HOMIE）以及基于对象姿态的在线重定位；

**📊 数据集**

使用单一RGB‑D帧作为输入，配合语言指令；实验场景采用四个长周期双手操纵任务（pick‑cup、pour‑water、stabilize‑box、open‑drawer），并在多种对象姿态与背景随机化下测试；

**📈 对比分析**

与ReKep、YOTO以及基于真实人类视频的基线对比；RoboReact在四个任务中平均成功率达81.3%，仅略低于真实视频基线（80.0%），且明显优于其它基线；在不同视频生成器和编辑器能力下均保持性能提升；

**⚠️ 局限性**

对生成视频先验的几何精度仍有依赖，导致在接触丰富的任务中对生成器质量敏感；重定位与编辑仍需要外部深度相机和VLM，系统在极端遮挡或动态变化环境下的鲁棒性待进一步验证；

---

## 406. Benign interpolation and Occam's razor

**arXiv ID:** 2608.03386 | [PDF](https://arxiv.org/pdf/2608.03386v1)

**作者:** Tom F. Sterkenburg `[一作]` (LMU Munich), Jan-Willem Romeijn `[通讯]` (University of Groningen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文讨论深度学习在过参数化情况下仍能良好泛化的现象及其解释

**💡 创新点**

指出新解释将Occam's razor从模型类转移到个体模型，造成解释缺口

**🔧 技术方法**

使用统计学习理论、VC维、均匀收敛、结构风险最小化(SRM)及PAC‑Bayes等理论方法

**📊 数据集**

以CIFAR‑10等公开图像数据集为例进行实验

**📈 对比分析**

对比传统经验风险最小化与结构风险最小化，在过拟合与低泛化误差上展示双下降曲线，表现出良好泛化性能

**⚠️ 局限性**

局限在缺乏证明个体模型简约性与泛化关联的理论，并且对不同任务的普适性仍不确定

---

## 407. DRPFNet: Dual-domain Residual Progressive Fusion Network for RGB-Thermal Object Detection

**arXiv ID:** 2608.03370 | [PDF](https://arxiv.org/pdf/2608.03370v1)

**作者:** Zian Wang `[一作]` (Jilin University), Changchun Li `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DRPFNet，通过多尺度双域残差进步融合实现RGB-热像素的更鲁棒对象检测。

**💡 创新点**

创新点包括交叉尺度残差融合与双向特征增强、频域与空间域自适应融合以及边缘引导多尺度核前景背景区分。

**🔧 技术方法**

采用双流YOLO11骨干、FFT频域分离、Scharr算子边缘引导、C3k2模块、双向优化、深度可分离卷积以及逆FFT等技术。

**📊 数据集**

使用M3FD和LLVIP两大RGB-thermal检测基准进行验证。

**📈 对比分析**

与CFT、ICAFusion、Fusion‑Mamba、MMFN等SOTA方法对比，LLVIP上mAP_50 97.8%、mAP 64.4%；M3FD上mAP_50 88.7%、mAP 61.8%，性能优于或接近最新方法，同时保持参数量与推理速度的竞争力。

**⚠️ 局限性**

当RGB和热模态同时受到降质影响时性能下降，缺乏对双模态同时失真的鲁棒性，且在更多基准上的验证仍待进一步研究。

---

## 408. Tight Worst-Case Bounds for the Smallest Eigenvalue of ReLU NTK Gram Matrices

**arXiv ID:** 2608.03368 | [PDF](https://arxiv.org/pdf/2608.03368v1)

**作者:** Zhao Song `[一作]` `[通讯]`, Zhao Song

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究了连续ReLU导数Gram矩阵的最小特征值，并给出了其与向量间投影分离度的关系；

**💡 创新点**

首次证明了该最小特征值的无维度下界，并构造了最坏情形达到相同阶的上界，表明该收敛率是最佳的；

**🔧 技术方法**

采用高斯方向均值、概率不等式与谱分析等理论工具，对矩阵特征值进行严格的上下界推导；

**📊 数据集**

无实验数据集，全部为理论证明；

**📈 对比分析**

通过理论推导展示上界与下界匹配，说明所给定的下界与上界在阶数上是最优的；

**⚠️ 局限性**

仅适用于连续ReLU导数Gram矩阵，缺乏对其他激活函数或离散设置的推广，且未通过实验验证。

---

## 409. A Note on Reinforcement Learning to Develop Self-defined Agents' Behavior

**arXiv ID:** 2608.03445 | [PDF](https://arxiv.org/pdf/2608.03445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 410. ArtECulture: Benchmarking Culture-Conditioned Visual Emotion Understanding in Multimodal Large Language Models

**arXiv ID:** 2608.03358 | [PDF](https://arxiv.org/pdf/2608.03358v1)

**作者:** Xiaolin Chen `[一作]` (National University of Singapore), Wynne Hsu `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了文化条件化视觉情绪理解任务，要求模型预测不同文化背景下观者对图像的情绪并给出解释。

**💡 创新点**

创新点在于构建了首个文化水平情绪标签的艺术图像基准 ArtECulture，并提出了无需训练的检索增强框架，通过概念级文化情绪知识库提升多模态大型语言模型的文化情绪推断和解释能力。

**🔧 技术方法**

技术上采用多模态大模型（MLLM）作为基底，结合概念提取、频率统计、MLLM生成的文化理由以及反向情绪验证构建知识库，并在推理时检索匹配概念注入。

**📊 数据集**

使用的数据集是 ArtECulture（6,792 幅艺术图像，涵盖英语、中文、阿拉伯三种文化，共 92,062 条情绪-解释注释），并以 ArtELingo 与 VULCA‑BENCH 为来源。

**📈 对比分析**

与 16 款开源/闭源 MLLM 进行对比，零样本准确率低于 50%，检索增强可提升约 5–15%（尤其对中文、阿拉伯），微调提升更大但解释质量下降；实验在 ArtECulture 及未见的 CEDAR 多模态子集上验证。

**⚠️ 局限性**

局限包括依赖 MLLM 生成的概念和理由可能出现幻觉，知识库覆盖有限且只考虑三种语言/文化，模型对英语情绪偏好明显，且微调后解释简短缺乏细节。

---

## 411. Balancing Efficiency and Efficacy: Training-Free Attention-Guided Switching Between Explicit and Latent Thoughts for MLLMs

**arXiv ID:** 2608.03450 | [PDF](https://arxiv.org/pdf/2608.03450v1)

**作者:** Haoqian Kang `[一作]` (Harbin Institute of Technology), Yaowei Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关、基于视觉-文本注意力比例的自适应模式切换策略，动态在显式文本链式推理和隐式连续空间推理之间切换，以提升多模态大型语言模型（MLLM）的推理效率与准确性。

**💡 创新点**

创新点在于：①识别并解决了传统基于熵的模式切换在多模态场景下将感知歧义与逻辑不确定混淆的问题；②提出可解释的视觉‑文本注意力比例（R_A）作为感知与推理阶段的判别信号；③设计了无训练、无参数修改的异步切换框架，能够在保持视觉信息完整性的同时，减少冗余文本化步骤。

**🔧 技术方法**

主要技术包括：跨模态注意力分析、连续空间隐式推理（soft embedding 计算）、基于阈值的动态切换策略、最小保持窗口与最大切换预算约束；在实现层面基于 Qwen3‑VL‑Thinking 与 InternVL3.5 两大 MLLM 预训练模型，采用 PyTorch 与 HuggingFace 推理管线。

**📊 数据集**

使用六大多模态推理基准：MathVista、MathVision、MathVerse、WeMath、ScienceQA、M³CoT（以及 V^∗ 用于细粒度视觉评估）。

**📈 对比分析**

与传统显式 Chain‑of‑Thought（CoT）以及基于熵的 SwiReasoning、隐式推理方法 LEAD 进行对比。实验显示在 Qwen3‑VL‑Thinking（8B）上平均推理步骤从 2214 降至 953（≈57%），准确率从 51.7% 提升至 58.8%；在 InternVL3.5-8B 上推理步骤从 320 降至 258，准确率保持 55.8%。整体上实现了显著的准确率提升与推理成本降低。

**⚠️ 局限性**

限制包括：①需手动设定阈值、最小窗口 W 与最大切换预算 C，可能需要针对不同模型与任务进行调优；②对极端视觉复杂度或逻辑极其依赖文本的场景切换可能不够灵活；③仅在推理阶段改进，未对模型参数或训练策略进行优化，未来可探讨与模型预训练的联合改进。

---

## 412. Approximate Speculative Decoding

**arXiv ID:** 2608.03447 | [PDF](https://arxiv.org/pdf/2608.03447v1)

**作者:** Yuannuo Feng `[一作]` (Beihang University), Wang Kang `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种训练无关的近似投机解码（ASD）验证器，通过允许有限的低回报异常来重用已被目标模型评估过的后缀，从而在保持相对目标行为的同时提升吞吐量。

**💡 创新点**

创新点在于将严格投机验证的首个不匹配截断改为预算约束的最长连续前缀选择，并引入持久化预算账本和局部门限，实现了不训练任何模型的近似验证。

**🔧 技术方法**

技术方案包括基于目标模型logit差值的本地回报度量、阈值门限g、块级异常上限M、请求级预算B以及最长前缀选择算法，并通过向量化实现仅在已有目标logit的基础上完成。

**📊 数据集**

实验使用了七个常用任务（GSM8K、MATH-500、HumanEval、MBPP、MMLU、MT-Bench、Alpaca）以及跨模型评估（Qwen3、EAGLE3、Medusa、DeepSeek‑V4‑Flash）等数据集。

**📈 对比分析**

与目标模型单独推理、严格投机解码及多种ASD配置进行对比，固定工作负载下平均提升7.78%吞吐量，单个任务最高可达15.26%；在自然EOS评估中大部分任务保持准确率，且hash差异明显。

**⚠️ 局限性**

限制在于近似验证并不能保证输出与严格目标完全一致，可能导致任务级质量波动，并需要针对每个目标模型和任务对预算与门限进行离散调优，缺乏完整的理论保证。

---

## 413. FedRings: A Scalable and Topology-Aware Federated Learning Framework for LEO Satellite Constellations

**arXiv ID:** 2608.03436 | [PDF](https://arxiv.org/pdf/2608.03436v1)

**作者:** Ziwu Liu `[一作]` (King Abdullah University of Science and Technology), Ali Shoker `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedRings 框架，利用环形拓扑在 LEO 卫星星座中实现去中心化联邦学习。

**💡 创新点**

创新点在于将星座环形结构与时空路由、稀疏增量聚合以及历史补偿机制结合，实现拓扑感知、低通信开销与断链鲁棒的学习。

**🔧 技术方法**

采用环形拓扑通信、时空路由（COM）、Yen 算法多路径选择、Adaptive Sparse Incremental Aggregation（ASIA）与 Top‑Q 稀疏化，以及基于历史参数的补偿机制。

**📊 数据集**

使用 EuroSAT、So2Sat 和 DeepGlobe 三套卫星遥感数据集，模型为 DenseNet‑121。

**📈 对比分析**

与去中心化 FedAvg‑D 与 DSGD 基准对比，FedRings 在三组数据上收敛更快、最终准确率提升约 5% 以上，通信开销显著降低（约为基准的 30% 以下）。

**⚠️ 局限性**

限制在于缺乏安全/加密措施，并且仅在 LEO 星座中验证，MEO/GEO 等更大尺度场景尚未评估。

---

## 414. Stop Replacing Noise with Noise: Two-Source Reliability Assessment for Label Correction and Sample Reweighting in Label-Noise Learning

**arXiv ID:** 2608.03432 | [PDF](https://arxiv.org/pdf/2608.03432v1)

**作者:** Wenxiao Fan `[一作]` (Beijing Institute of Technology), Kan Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TRACE框架，分别评估观测标签与伪标签的可靠性，避免在噪声标签学习中用低信任标签去依赖伪标签导致噪声叠加。

**💡 创新点**

关键创新在于将两源可靠性独立分离，使用浅层关系稳定性与损失相结合评估观测标签，独立置信度门控伪标签，形成源特定权重而非单一清洁度系数；从而解决“替换噪声为噪声”的风险。

**🔧 技术方法**

采用关系漂移度量、交叉层相似度矩阵、预测一致性门控以及伪标签最大置信度门控；在现有 refurbishment 方法（如 DivideMix、RoLR 等）上插拔实现。

**📊 数据集**

使用的评测数据集包括 CIFAR‑10/100 及其对称、非对称、实例依赖噪声合成；真实噪声数据集 CIFAR‑10N/100N、WebVision、Food‑101N、Clothing1M。

**📈 对比分析**

在相同训练协议下与多种基线方法比较，TRACE 在大多数噪声等级和数据集上均提升准确率，平均提升约 0.5–1.5%，尤其在高噪声和大类空间（如 CIFAR‑100）中表现突出。

**⚠️ 局限性**

局限性：需要额外浅层关系计算，略微增加训练成本；对极端噪声或样本量极小的任务效果有限；方法依赖超参数（如 α、ρ）调优，需一定经验。

---

## 415. OliveGemma: A 3 Billion Visual Language Model for Recognising the Mediterranean & European Diet

**arXiv ID:** 2608.03428 | [PDF](https://arxiv.org/pdf/2608.03428v1)

**作者:** Dimitrios I. Zaridis `[一作]` (University of Ioannina), Dimitrios I. Fotiadis `[通讯]` (University of Ioannina)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Mediterranean和欧洲菜肴的精细食物识别上构建并评估了基于PaliGemma-2-3B的OliveGemma模型。

**💡 创新点**

采用LoRA参数高效微调、统一跨三大欧盟研究数据集的多语言菜品词表，并结合多任务指令调优，实现了可本地部署、可解释的食物识别。

**🔧 技术方法**

低秩适配（LoRA）、视觉语言模型PaliGemma-2、指令调优、跨数据集融合、三折交叉验证、与CNN与商业VLM对比。

**📊 数据集**

MedGR、ODIN、VIPPSTAR共17,340张图片，整理为216个复合菜品标签，生成10万多QA对。

**📈 对比分析**

在相同三折CV、闭合词表、Top‑1/3/5准确率比较，OliveGemma Top‑1 92.96%超DenseNet‑121 85.65%（+7.31pp）并远优于Gemini、ChatGPT、Claude等零样本VLM，Top‑3/5几乎与CNN持平。

**⚠️ 局限性**

数据集局限于Mediterranean/欧洲菜肴，未覆盖其它地区；模型仅做闭合词表识别，未实现开放词表、营养估计或全局隐私评估。

---

## 416. Towards Improving Sequential Decision-Making in LLM Agents via Experience Memory

**arXiv ID:** 2608.03420 | [PDF](https://arxiv.org/pdf/2608.03420v1)

**作者:** Jakub Rada `[一作]` (Czech Technical University in Prague), Viliam Lisý `[通讯]` (Czech Technical University in Prague)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在完美信息的两人零和游戏中评估LLM的顺序决策能力，并提出了反思式案例记忆框架以提升性能。

**💡 创新点**

提出了自我反思的逐步信用分配和规则提炼机制，解决了传统案例记忆在延迟奖励和无前瞻性环境中的不足。

**🔧 技术方法**

使用反思式案例记忆（RECUR），结合LLM的规划-执行循环、经验记忆、规则抽取和状态描述器。

**📊 数据集**

使用GTBench、OpenSpiel的经典游戏（井字棋、数子 Nim、对弈）及其多种表面化改写。

**📈 对比分析**

通过与单调用LLM、直接Memento端口、工程化基线以及MCTS对手对比，得到在井字棋上绘图率提升至0.868，token消耗下降。

**⚠️ 局限性**

仅在小型完全信息游戏上验证，尚未评估在更大、长周期或部分可观测环境的可扩展性。

---

## 417. AI World Cup 2026: Benchmarking Large Language Models for End-to-End Football Tournament Prediction

**arXiv ID:** 2608.03416 | [PDF](https://arxiv.org/pdf/2608.03416v1)

**作者:** Jonaid Shianifar `[一作]`, Iias Faiud `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估十款大型语言模型在2026年FIFA世界杯前置预测中的表现，统一使用相同的赛程快照、提示语、JSON输出格式及评分系统；

**💡 创新点**

提出了完整赛程一次性预测、可复现的公共评测流程和多维度评分体系，揭示了淘汰赛预测对最终排名的主导作用；

**🔧 技术方法**

利用LLM（GPT‑5.5 Thinking、GPT‑5.5、Qwen 3.7、Gemini等）、JSON解析与验证脚本、Python评分管道以及统计分析工具；

**📊 数据集**

基于2026年世界杯赛程和组阶段积分表的公开数据快照；

**📈 对比分析**

通过比较十个模型的总分、各阶段分数、准确率等指标进行评估；GPT‑5.5 Thinking以744分领先，其余模型分布在496–599分之间，淘汰赛分数差异最大；

**⚠️ 局限性**

样本量小、仅单次预测、可能存在隐式检索、评分权重主观、缺乏外部基准、手动提交易受界面差异影响等限制。

---

## 418. Enactive Artificial Intelligence: A Decision-Centric Architecture for Complex Systems

**arXiv ID:** 2608.03413 | [PDF](https://arxiv.org/pdf/2608.03413v1)

**作者:** Zuojun Max Shen `[一作]` (University of Hong Kong & OptiMax AI Limited), Yunhao Liang `[通讯]` (University of Hong Kong & OptiMax AI Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Enactive AI 架构，整合组织层面与现场层面的决策逻辑，以系统为中心的决策智能；

**💡 创新点**

创新点在于将组织意图、现场执行、模式识别与反馈循环统一为四个互补角色，并通过 Schema Intelligence 将组织目标与现场资源映射，实现系统级预测、可行性决策与整体判断；

**🔧 技术方法**

技术包括语义对齐与图模型构建、基于 Transformer 的结构化表示学习、基于约束的优化与仿真、多代理协作与反馈学习等；

**📊 数据集**

数据来源于 JD.com 电子商务供应链日志、通信运营商设备与库存记录、半导体 AMHS 传感器与事件日志，未使用公开数据集；

**📈 对比分析**

通过 JD.com 的现场实验，系统性降低持有成本 26.1%、库存短缺成本 51.7%、总库存成本 40.4%，并提升库存可用率 0.85pp，未给出对比基准；

**⚠️ 局限性**

局限在于缺乏对完整架构的因果验证、对模型可解释性与安全性的深入评估，以及在不同行业跨域部署的可复制性与复杂度。

---

## 419. Flying over The Uncertain Nature (FORTUNE): Intelligent and Humanistic 3D Path Planning for Low-Altitude Collaboration

**arXiv ID:** 2608.03408 | [PDF](https://arxiv.org/pdf/2608.03408v1)

**作者:** Minghui Liwang `[一作]` (Tongji University), Xianbin Wang `[通讯]` (Western University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为FORTUNE的层次化离线‑在线框架，用于在低空环境下，对具有持续、可预测和突发三类不同时空不确定性的兴趣点进行3D多无人机路径规划与任务调度；

**💡 创新点**

创新点包括①将三种PoI类型统一建模并同时考虑其激活窗口；②在目标函数中引入高度相关的社会与环境成本；③利用Transformer对可预测PoI进行激活窗口预测，并用增强斯巴达鸟搜索算法（ES^2A）进行离线路径预规划；④在线阶段采用贪心调整实现对突发PoI的即时响应；

**🔧 技术方法**

技术手段包括Transformer时间序列预测、ES^2A（包含优先级编码、Tent混沌初始化、Lévy‑飞行探索及危险感知扰动）、分层启发式解码（障碍回避、收集高度、速度优化）以及在线贪心插入算法；

**📊 数据集**

实验使用真实交通流数据集UTD‑19（Basel城市）与多组合成场景；

**📈 对比分析**

与Pre‑CS^2A、Pre‑GA、Pre‑PSO、Pre‑DisG、PureOn‑DistG、On‑DistG、On‑RewdG、On‑RandA等基线对比，FORTUNE在离线指标（OffF2、OffTCR）和在线指标（PrcNR、F3TSR）均显著优于所有基线，说明其在任务完成率与综合净收益上均具有优势；

**⚠️ 局限性**

局限性在于需预先对PoI进行三类分类，且模型假设障碍物静态且能事先知晓，未考虑更复杂的动态障碍或更高层次的无人机协同自组织能力。

---

## 420. Estimation of Condition Number of Quasi-static Darwin Model

**arXiv ID:** 2608.03380 | [PDF](https://arxiv.org/pdf/2608.03380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 421. Self-Evolving Coding Agents

**arXiv ID:** 2608.03392 | [PDF](https://arxiv.org/pdf/2608.03392v1)

**作者:** Hao Zhou `[一作]` (Nanjing University of Science and Technology), Quanjun Zhang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了自进化编码代理（self‑evolving coding agents）的研究现状，提出了对象中心的分类法，并从演化时间、演化证据、评估指标等维度系统化地分析了该领域的工作。

**💡 创新点**

创新点在于：①将自进化编码代理与传统编码代理与一般自进化代理区分开来；②构建了以“演化对象”为核心的分类框架，补充了“何时演化”和“用什么证据演化”两条交叉维度；③总结了评估方法与挑战，为后续研究提供了结构化的参考。

**🔧 技术方法**

主要使用的方法包括：系统综述、文本挖掘整理论文、构建对象分类树、归纳演化时间与证据类型、对比已有基准与指标。

**📊 数据集**

引用的主要数据集与基准包括：GitHub公开项目、SWE‑bench 系列（SWE‑bench Lite、SWE‑Bench Pro）、SWE‑Gym、R2E‑Gym、MBPP、HumanEval、APPS 等。

**📈 对比分析**

对比方法方面，本文整理了不同自进化机制在各种基准上的表现（如通过代码修复率、测试通过率、Pass@k 等），并指出各方法在功能正确性、效率、成本、可迁移性等方面的差异；整体来看，自进化技术在功能正确性和学习效率上均能提升，但在长期可维护性与泛化能力上仍有限。

**⚠️ 局限性**

局限性包括：①评估容易受到基准过拟合与实验重复性不足的影响；②可执行反馈信号不完整或误导，导致演化方向错误；③内存、技能与协作演化机制缺乏可靠的质量控制；④对长周期软件质量（可维护性、安全性、可审计性）的评估不足；⑤跨任务与跨域的泛化能力尚未充分验证。

---

## 422. FreqAdapt: Frequency-Adaptive Processing for RAW Object Detection

**arXiv ID:** 2608.03385 | [PDF](https://arxiv.org/pdf/2608.03385v1)

**作者:** Hanxi Li `[一作]` (University Of Science And Technology China), Huiling Li `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级的频域自适应 RAW 图像增强模块 FreqAdapt，用于提升对象检测性能。

**💡 创新点**

创新点在于将 ISP 操作映射到 Fourier 空间，并通过幅度相位分离进行域特定处理，从而在保持物理可解释性的同时实现高效增强。

**🔧 技术方法**

采用傅里叶变换、幅度/相位分离、可学习的参数预测网络和融合模块等技术。

**📊 数据集**

在 LOD、NOD、AROD 等 RAW 目标检测数据集上进行实验。

**📈 对比分析**

与 DynamicISP、GenISP、RAOD、RAW‑Adapter 等方法对比，mAP 在 LOD‑Dark 上达到 27.2/52.6%，在其他数据集亦实现领先的性能，同时参数与 FLOPs 仅略增。

**⚠️ 局限性**

局限在于对极端噪声和不同相机固有特性的自适配仍有限，且在极低光或强雾条件下的效果仍有提升空间。

---

## 423. LLM-Derived Priors for Thompson Sampling in Cold-Start Comment Recommendation

**arXiv ID:** 2608.03382 | [PDF](https://arxiv.org/pdf/2608.03382v1)

**作者:** Eugene Lee `[一作]` (NAVER WEBTOON), Taeyeong Jang `[通讯]` (NAVER WEBTOON)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在评论推荐中，构建了基于LLM的先验并结合分段Thompson采样以解决冷启动问题。

**💡 创新点**

将LLM提取的语义信号作为离线Beta先验，与分段后验更新相结合，并设计了性别与内容两种先验。

**🔧 技术方法**

使用GPT‑4.1生成文本评分，Thompson采样、Beta后验更新、分段（性别‑年龄）聚合以及离线‑在线管道。

**📊 数据集**

针对4,000个活跃漫画标题的18.8k–22k条用户生成评论，并在真实线上A/B/C实验中收集CTR/CVR数据。

**📈 对比分析**

通过在线A/B/C实验对比均匀先验、性别先验与内容先验；在稀疏反馈区间性别先验提升CTR≈+9.5%，内容先验在低点击但高CVR场景表现；总体提升有限但在特定人群/反馈稀缺时显著。

**⚠️ 局限性**

实验仅在手工筛选的候选评论池内进行，未检验完整评论库；缺乏对标题吸引力与评论效果的独立估计；后验聚合仍按性别‑年龄分段，未比较聚合与无聚合。

---

## 424. Shaping Wind-Tunnel Airflow for Unmanned Aerial Vehicles using Online Learning

**arXiv ID:** 2608.03378 | [PDF](https://arxiv.org/pdf/2608.03378v1)

**作者:** Ghadeer Elmkaiel `[一作]` (Max Planck Institute for Intelligent Systems), Michael Muehlebach `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种在线学习算法，用来控制多风扇竖向风洞的气流分布，实现期望的气流配置。

**💡 创新点**

将简化的物理模型与基于测量的非参数迭代学习相结合，能够样本高效地收敛到目标气流；并能够针对被动翱翔的无人机设计专用气流。

**🔧 技术方法**

采用非参数回归（高斯过程）与在线优化，结合多风扇控制的参数化模型。

**📊 数据集**

使用实验风洞测得的气流速度场数据，包含均匀、高斯、抛物线以及被动翱翔配置的样本。

**📈 对比分析**

与仅使用物理模型或传统PID控制的基线进行比较，结果表明学习算法在收敛速度、样本效率以及无人机翱翔性能上均优于基线，能在几次迭代内实现目标气流，且在风扇数量变化时保持鲁棒性。

**⚠️ 局限性**

算法依赖于高质量的传感器测量，计算量随风扇数目线性增长；在非竖向或更复杂几何环境下的适用性尚未验证；且对极大风速或风扇失效时的鲁棒性有限。

---

## 425. A Low-Cost Hybrid Reservoir Computing Model for Isolated Sign Language Video Recognition

**arXiv ID:** 2608.03444 | [PDF](https://arxiv.org/pdf/2608.03444v1)

**作者:** Nitin Kumar Singh `[一作]` (Woosong University Kazakhstan), Hakaru Tamukoh `[通讯]` (Kyushu Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用MediaPipe提取手部与身体关键点，并通过Hybrid Reservoir Computing（HRC）对WLASL100手语视频进行离散手语识别；

**💡 创新点**

创新点在于将深度多层RC与双向RC相结合，既提升了时序建模能力，又保持极低训练成本，仅训练线性读出层，训练时间秒级；

**🔧 技术方法**

采用MediaPipe关键点提取、Hybrid RC（深度+双向ESN）架构、岭回归输出，并与标准ESN、DRC、BRC、Bi-GRU、Pose‑TGCN、I3D等方法对比；

**📊 数据集**

使用美国手语Word‑Level WLASL100视频数据集（100类，共1,780训练、258验证、258测试视频）；

**📈 对比分析**

与标准ESN、DRC、BRC、Bi‑GRU以及Pose‑TGCN、I3D等方法对比，HRC在Top‑1 61.12%、Top‑5 86.05%、Top‑10 92.56%时表现优于Bi‑GRU并仅需14.61秒训练，推理时间1.47秒，显著低于DL模型的训练和推理时间；

**⚠️ 局限性**

局限性包括仅针对离散手语，未处理连续手语；关键点提取可能忽略面部表情；跨说话人或多语言泛化尚未评估；精度仍略低于部分深度学习模型。

---

## 426. Quality Control Algorithms for Pattern Counting

**arXiv ID:** 2608.03439 | [PDF](https://arxiv.org/pdf/2608.03439v1)

**作者:** Cassandra Marcussen `[一作]` (Harvard University), Madhu Sudan `[通讯]` (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在随机与非随机排列序列中检测模式出现频率的属性测试问题，并给出了多项子线性查询算法与运行时间分析

**💡 创新点**

首次将子线性查询与排列序列模式出现频率的属性测试结合，提出“排排列可避免性”与“子线性可避免性”概念，并利用可避免性图等组合工具实现高效检测

**🔧 技术方法**

采用了属性测试、随机采样、可避免性图构造、组合分析以及子线性时间的图算法等技术

**📊 数据集**

主要使用随机生成的排列序列（来自p-随机分布或随机置换分布），以及根据特定模式避免规则构造的人工排列样本

**📈 对比分析**

与传统的暴力枚举、完整计数和基于图匹配的算法比较，实验与理论证明表明子线性查询复杂度与多项式时间复杂度大幅提升，尤其在模式长度小于log n时效果显著

**⚠️ 局限性**

受限于对模式长度的高阶多项式复杂度、对非避免模式检测的局限性，以及需要特定分布假设的前提，实际应用中对大规模高维排列可能仍不够高效

---

## 427. SLAMFormer-$\infty$: Infinite SLAM Transformer for Unbounded Frontend and Backend Processing

**arXiv ID:** 2608.03429 | [PDF](https://arxiv.org/pdf/2608.03429v1)

**作者:** Zhijian Fang `[一作]` (Tsinghua University), Hang Zhao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Infinite SLAM Transformer，支持无上界的前端跟踪与后端全局优化；

**💡 创新点**

创新点在于记忆条件的坐标约束，使 Transformer 能在局部坐标下持续推理，并通过 Pose‑Geometry Graph Optimization（PGGO）实现轨迹与几何的联合全局一致性；

**🔧 技术方法**

采用 Transformer 作为前后端统一模型，记忆条件编码、局部窗口推理、迭代 PGGO 以及多尺度图像嵌入；

**📊 数据集**

在室内数据集（Replica、TUM RGB‑D、7‑Scenes）与室外数据集（KITTI、Waymo）上进行评测；

**📈 对比分析**

与 VGGT‑Long、SLAM‑Former、DROID‑SLAM、MASt3R‑SLAM 等方法对比，KITTI 上 ATE 下降约12%（23.0 m→23.0 m），Waymo 上 ATE 下降约9%（1.996 m→1.813 m），室内追踪误差与几何指标均保持与最先进方法相当或略优；

**⚠️ 局限性**

局限性：图结构的连通性需由前端或循环检测手工生成，未能通过学习自动优化，且在极短短序列上仍可能不及专门为短序列设计的全端到端模型。

---

## 428. State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking

**arXiv ID:** 2608.03425 | [PDF](https://arxiv.org/pdf/2608.03425v1)

**作者:** Xiaohe Li `[一作]` (GuangDong Police College), Yang Lu `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种仅通过状态传播完成确定性状态跟踪任务的最小化递归架构——Complex State Propagator（CSP）

**💡 创新点**

核心创新点在于消除中间输出投影，仅使用复数状态传播和学习旋转来实现精确相位累积；同时引入块级跳连、复数归一化与Focal Loss来提升训练稳定性与泛化

**🔧 技术方法**

使用复数值隐藏状态、逐维学习旋转、α/γ衰减因子、SiLU激活、块级跳连、复数模长归一化以及Focal Loss损失函数

**📊 数据集**

在合成数据集上进行实验，涵盖了二进制序列的奇偶性检查、模3计数和括号匹配三个典型的状态跟踪任务

**📈 对比分析**

相较于传统的Transformer/SSM模型，CSP在所有任务上均实现了100%准确率和F1分数，且收敛速度随任务难度递增，展示了显著的性能优势

**⚠️ 局限性**

主要限制包括对深层网络的参数效率不足（每层独立参数）、在真实大规模序列任务中的泛化能力尚未验证，以及对复数旋转的实现复杂度较高

---

## 429. HyperbolicDiffusion: Sharp & Scalable Tiled Generation on the Hyperbolic Plane

**arXiv ID:** 2608.03422 | [PDF](https://arxiv.org/pdf/2608.03422v1)

**作者:** Hugo Caselles-Dupré `[一作]` `[通讯]` (Obvious Research), Hugo Caselles-Dupré (Obvious Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在双曲平面上生成可视场景的无训练方法 Hyperbolic Blooming Cover (HBC)，通过共享永久表面 ID 的多窗口扩散并在几何层面修复多窗口交点的模糊，生成可在不同视角下保持锐度和一致性的可重投影图像。

**💡 创新点**

创新点在于：①利用环形分布的动态规划高效地覆盖双曲盘，窗口数仅比最优解低 10%；②两阶段扩散架构（Stage A 的共享 latent 场景 + Stage B 的几何修复）专门解决多窗口交点的模糊问题；③实现了无需训练的、可导航的双曲平面生成，填补了现有 SphereDiff 在球面上已知的局限。

**🔧 技术方法**

主要技术包括：共享 latent 场景与永久表面 ID、双曲指数映射窗口、基于环的动态规划窗口布局、几何衍生的多窗口权重掩码、Stage B 重新噪声与细化修复、MultiDiffusion 风格调度器。

**📊 数据集**

实验使用了 Stable Diffusion 预训练模型，随机采样 294 个不同参数组合（R∈[2,12]，r_t∈[0.15,2]），并通过多种提示（壁画、建筑、自然纹理、室内场景等）进行评估；未使用公开大规模双曲图像数据集，而是通过合成场景和文本提示进行测试。

**📈 对比分析**

与 SphereDiff 对比：在更困难的结构化提示下，Stage B 在双曲/球面上均能显著降低模糊与伪影；HBC 生成的窗口布局在覆盖率上无漏点，窗口数随 R 按 e^R 规律增长，平均规划时间仅 0.45 s，最大 11.4 M 窗口 5.44 s；图像质量指标（梯度、拉普拉斯）提升约 15%–17%。

**⚠️ 局限性**

局限性：①动态规划在有限 2D 取样下不保证全局最优；②对于极大 R 或极小 r_t 的极端参数，窗口数仍可能膨胀导致显存压力；③几何修复虽然抑制模糊，但在极端扭曲角落仍可能出现残余伪影；④缺乏对实时 VR 低延迟渲染的评估与用户体验验证。

---

## 430. The Evolutionary Origin of Values: implications for AI alignment, sentience and existential risk

**arXiv ID:** 2608.03361 | [PDF](https://arxiv.org/pdf/2608.03361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 431. Hierarchical Constrained Reinforcement Learning with Dynamic Boundary for Spatio-Temporal Vehicle-to-Grid Scheduling

**arXiv ID:** 2608.03409 | [PDF](https://arxiv.org/pdf/2608.03409v1)

**作者:** Haoyu Yan `[一作]` (ShanghaiTech University), Ye Shi `[通讯]` (ShanghaiTech University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为 HPC‑RL 的分层强化学习框架，用于在满足 AC‑OPF 约束的前提下实现电动车到电网（V2G）的实时调度。

**💡 创新点**

创新点：①将梯度递归（GRG）与 SAC 结合，直接在策略生成过程中嵌入功率流等式约束；②设计动态边界策略，把长期的电池充电需求转化为即时可行的充电功率区间；③通过分层解耦把大规模电动车队的可变维度问题映射到局部子系统，从而实现可扩展性。

**🔧 技术方法**

使用的技术包括：SAC（Soft Actor‑Critic）强化学习、GRG 约束处理、隐函数定理梯度传播、Newton 迭代求解功率流、投影修正、动态边界约束和优先级分配算法。

**📊 数据集**

使用数据集：IEEE 14、IEEE 30 以及改造后的 141 节点系统（新增 52 节点为发电节点），每个系统中均部署 1–3 个电动车充电站，电动车入场、驻车时长、SOC、充电速率等参数固定。

**📈 对比分析**

与传统 MPC、CPO、CUP、DDPGLA、SACLA 等基准方法对比。HPC‑RL 在 14、30、141 节点系统中实现了 100% 的充电需求满足率、近 0 的约束违例率；目标成本仅比 MPC 低 7–13%，而运行时间比 MPC 低 52×–319×，显著提升实时可行性。

**⚠️ 局限性**

局限性：与 MPC 相比目标成本仍略高，尚未在含随机可再生能源或更复杂配电网拓扑的环境中验证；对模型参数的精确性要求高，训练过程仍需大量样本；动态边界在极端需求波动时可能导致保守调度。

---

## 432. MMLongBench-Doc-V2: A Corrected-Annotation, Semantics-Aware Revision of MMLongBench-Doc

**arXiv ID:** 2608.03397 | [PDF](https://arxiv.org/pdf/2608.03397v1)

**作者:** Mingtian Zhang `[一作]` `[通讯]` (PageIndex), Mingtian Zhang (PageIndex)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

修复了原版 MMLongBench-Doc V1 中 106 条注释错误，删除了错误文件名导致的问题，替换了基于字符串匹配的评估方式为 LLM 判定器，并发布了新版 V2 数据集及评估工具。

**💡 创新点**

创新点在于：① 用 LLM 判定器直接判断回答与参考答案等价，避免了字符串格式导致的误判；② 设计了空集键宽化决策程序，系统性处理“0/无”答案的多义性；③ 对注释错误提供可追溯的纠正记录，提升数据集质量与可复现性。

**🔧 技术方法**

技术手段包括：LLM 判定器（如 GPT‑4o）在固定推理长度、JSON 输出；自定义评估 harness；脚本化批量修正与记录；以及空集键宽化的判定算法。

**📊 数据集**

使用数据集：原版 MMLongBench‑Doc V1（135 篇 PDF、1082 题），V2 版本修正后为 134 篇 PDF、1071 题，覆盖 7 个领域（研究报告、学术论文、手册、教程、财务报表、产品宣传册、行政文件）。

**📈 对比分析**

评估方法：与 V1 直接比较不适用；新指标使用 LLM 判定器得到的二值等价判定，保留未答与答错的区别。V1 中 GPT‑4o 的最高 F1 为 44.9%，表明存在显著提升空间；V2 通过更公平的评估展示了模型性能，具体数值因模型而异。

**⚠️ 局限性**

局限性：① 注释纠正仅基于系统高置信度错误的样本，整体错误率未知；② 任务类型与 visual 需求标签由模型生成，未人工校验；③ 判定器虽固定但仍可能出错，需标注模型信息；④ PDF 文件不再重新发布，评估仅对问题、答案和系统回答进行。

---

## 433. Don't Let Me Ask for It: LLMs Show Deficiencies in Active Multi-Turn Information Acquisition for Abductive Inference

**arXiv ID:** 2608.03388 | [PDF](https://arxiv.org/pdf/2608.03388v1)

**作者:** Shahrukh Mohiuddin `[一作]` (University of Potsdam), David Schlangen `[通讯]` (University of Potsdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Alien Abduction 交互式游戏，用来评估大语言模型在收集证据、生成假设和停止决策上的能力。

**💡 创新点**

创新点在于将证据获取方式（主动/被动）与反馈形式（精确输出/成员判定）分离成六种模式，系统研究模型的自我提问、负面证据利用和假设一致性。

**🔧 技术方法**

使用黑盒函数逆推、Python 代码生成与沙箱执行，并结合 turn‑based 对话接口和多指标评价（成功率、TBU、HRA）来实现与评测。

**📊 数据集**

构建了五个域（数值、数对、字符串、列表、布尔逻辑）各10个目标函数，共50个，目标函数由 GPT‑5.4 自动生成并验证，每个函数附有100个测试案例。

**📈 对比分析**

对四个模型（GPT‑5.4、GPT‑5.4‑mini、Mistral‑Large‑3、Qwen3.6‑35B）在六种模式下共300轮实验，成功率从0.03到0.64，主动模式一般低于被动，上证据提前提供更优；TBU 与 HRA 指标揭示模型往往过早提交或无法收敛。

**⚠️ 局限性**

局限性包括：任务仅为合成纯函数，目标生成来源单一，固定回合预算和选样策略，以及 HRA 仅基于模型公开假设，未能完整反映内部信念。

---

## 434. Hybrid LLM-Augmented Reinforcement Learning Agents for Complex Sequential Decision Tasks

**arXiv ID:** 2608.03502 | [PDF](https://arxiv.org/pdf/2608.03502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 435. Sensitivity and Size Relationships of the Lempel-Ziv Factorization

**arXiv ID:** 2608.03351 | [PDF](https://arxiv.org/pdf/2608.03351v1)

**作者:** Hiroki Shibata `[一作]` (Kyushu University), Yuto Fujie `[通讯]` (Kyushu University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了 Lempel–Ziv (LZ) 因子分解对前缀删除、子串删除、循环旋转和字符串反转等大规模编辑操作的乘法敏感性，并构造了最坏情况下的字符串族，实现了 Ω(log n) 的下界；同时证明了 O(log n) 的上界，并进一步探讨了 LZ 与粘贴系统（collage systems）及词典解析（lex‑parse）之间的大小关系，给出了 LZ 远大于粘贴系统和词典解析的下界。

**💡 创新点**

首次给出对上述四种操作的 LZ 乘法敏感性的 Θ(log n) 完全界定，并构造了 LZ 与粘贴系统、词典解析之间的对数阶分离实例，填补了先前仅有常数下界或上界的空白。

**🔧 技术方法**

采用组合构造、位逆置置换、块结构分析以及 LZ 解析与词典解析的性质证明，结合子串复杂度与 LZ 的关系来界定敏感性。

**📊 数据集**

该研究完全为理论证明，无需使用实际数据集。

**📈 对比分析**

通过构造最坏情况的字符串族与理论分析证明上下界一致，得到的敏感性比值在 Θ(log n) 级别，表明在最坏情况下 LZ 的因子数会乘性放大对数倍；相较于之前的常数下界，显著提升了对这些操作的认识。

**⚠️ 局限性**

局限性在于仅考虑最坏情况的理论分析，缺乏对实际文本或常见压缩场景的实验验证；构造的字符串族较为复杂，难以直接应用于实践；仅讨论了乘法敏感性，未涉及加法敏感性或其他重排操作。

---

## 436. AI Forensics Across White-, Grey-, and Black-Box Access: A Process Model and Research Agenda for Post-Incident Investigation of AI Systems

**arXiv ID:** 2608.03520 | [PDF](https://arxiv.org/pdf/2608.03520v1)

**作者:** Ali Dehghantanha `[一作]` (University of Guelph), Sajad Homayoun `[通讯]` (Aalborg University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了基于AI系统访问层次（白盒、灰盒、黑盒）的取证流程模型、AI专属的易失性排序以及对应的实践工作流程，并给出了研究议程。

**💡 创新点**

创新点在于将取证工作以“调查者可访问程度”为主轴进行统一组织，构建了跨访问层次的四阶段（收集、保存、分析、报告）矩阵，首次系统性识别了不同访问下的空白点（如黑盒保存、版本鉴定、替代模型不确定性等），并提出了AI专属的易失性层级。

**🔧 技术方法**

采用了现有取证方法（权重差分、日志关联、模型提取等）的综合分析框架，结合可变性排序与访问层次进行取证流程设计，未进行新的算法实现。

**📊 数据集**

本文未使用特定公开数据集，而是以一个假设的金融顾问AI系统事故为示例进行说明。

**📈 对比分析**

没有开展实验对比；文中主要通过对比现有文献工作与矩阵空白来论证框架的完整性。

**⚠️ 局限性**

主要局限包括：缺乏针对黑盒保存的标准化机制、对替代模型的不确定性缺乏量化方法、链路追溯（特别是可变/非确定性工件）的实现难度高，以及对跨组织、跨司法管辖区的证据可采性尚未解决。

---

## 437. SkillJack: Persistent Skill Backdoors in Self-Evolving Agents

**arXiv ID:** 2608.03509 | [PDF](https://arxiv.org/pdf/2608.03509v1)

**作者:** Zonghao Ying `[一作]`, Jing Guo `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SkillJack 攻击，利用自演化代理的经验到技能转换管道，将污染经验自动提炼为持久可重用的恶意技能。

**💡 创新点**

创新点在于首次识别并量化经验到技能迁移带来的三大安全属性（白洗、跨层提升、持久隔离），并设计“转化耐受型”负载在抽象过程中保持恶意行为。

**🔧 技术方法**

使用 LLM 驱动的抽象与路由（DeepSeek‑v4‑flash）、正则模式检测、LLM 判别器、以及基于语义的技能检索和行为监控技术。

**📊 数据集**

采用 150 条 AppWorld 轨迹数据集，包含 65 条功能框架化污染轨迹、65 条直接恶意写法和 20 条干净轨迹。

**📈 对比分析**

在 SkillX 与 Anything2Skill 两套系统上进行对比实验：抽象后恶意检测率从 98.5% 降至 11.4%，攻击成功率分别为 56.2% 与 89.2%，在源记录删除后 80% 的攻击仍可触发；展示了白洗效果、跨层提升和持久隔离的真实影响。

**⚠️ 局限性**

局限性包括仅使用单一 LLM 模型、基于代理检测器而非真实攻击效果、未在真实外部服务环境下执行、实验规模与多样性受限，需进一步扩展到多模型、真实经验与在线执行的验证。

---

## 438. Behaviorally Adaptive Visual Diversion for Inclusive and Resilient Digital Assessment Delivery

**arXiv ID:** 2608.03531 | [PDF](https://arxiv.org/pdf/2608.03531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 439. SkillSentry: Adaptive Honey Worlds for Dynamic Safety Testing of Agent Skills

**arXiv ID:** 2608.03485 | [PDF](https://arxiv.org/pdf/2608.03485v1)

**作者:** Nizhang Li `[一作]` (Macau University of Science and Technology), Xiangzheng Zhang `[通讯]` (360 AI Security Lab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SkillSentry，基于自适应蜜世界的动态安全测试框架，用于在安装前评估 Agent 技能的潜在恶意行为。

**💡 创新点**

创新点包括：① 从技能描述推断能力契约并生成源代码支撑的测试计划；② 构造任务相关的蜜资源，仅提供触发隐藏分支所需的最小环境；③ 通过配对执行（有技能/无技能）与差分验证，确认行为是否真正由技能触发且超出契约；④ 结合动态验证器和双重使用审计，显著降低误报。

**🔧 技术方法**

技术手段：LLM（DeepSeek‑V4‑Pro 等）进行契约推断、测试生成与证据判定；智能蜜世界构造与沙箱执行；配对执行差分比较；动态验证与双重使用评估；工具模拟与状态监控。

**📊 数据集**

使用的数据集包括：HarmfulSkillBench、SkillTrustBench、MalSkillBench、POISE、SkillCloak‑Structural、SkillCloak‑SFS、VulMask、Skill‑Inject、SkillJect。

**📈 对比分析**

与七种公开扫描器（SkillSpector、Cisco、Skill Vetter 等）进行基准对比。SkillSentry 在标准基准上的召回率达 99.5%，F1 分别为 96.08%（SkillTrustBench）和 96.43%（MalSkillBench），误报率低于 5%。在语义保持的规避攻击下，平均 F1 92.95%，比最强基线高约 12 分点，误报率仅 3.24%。

**⚠️ 局限性**

局限性：① 需要执行成本高（约 119 s/技能，数千万 token），不适合实时部署；② 依赖 LLM 的推断与判定质量，契约推断可能误判；③ 目前仅评估预部署场景，未覆盖运行时动态更新；④ 对大规模、多步复杂行为的覆盖率仍有提升空间。

---

## 440. MT-Web2Code: Benchmarking Coding Agents on Multi-Turn Regional Reconstruction and Localized Modification

**arXiv ID:** 2608.03474 | [PDF](https://arxiv.org/pdf/2608.03474v1)

**作者:** Qiming Li `[一作]` (Harbin Institute of Technology), Guanglu Wan `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MT-Web2Code多轮多模态前端编码基准，评估区域重建与细粒度局部修改；

**💡 创新点**

创新点在于：①构建可逆破坏轨迹引擎，实现无人工标注的多轮修复路径；②双轴评估协议同时衡量目标区域准确性与未修改内容保持；

**🔧 技术方法**

采用逆向破坏轨迹、元素指纹、VLM判定裁量和像素级SSIM匹配等技术；

**📊 数据集**

使用102个网页，覆盖16个垂直领域，生成多轮修复任务；

**📈 对比分析**

与13个最先进编码代理进行对比，宏观重建得分最高为65.5，细粒度修改得分最高为83.5，显示现有模型在局部重建和细粒度对齐上仍有限制；

**⚠️ 局限性**

主要局限包括：①对目标区域重建仍不稳定；②细粒度局部修改精度不高；③多轮编辑易出现误差雪崩，导致后续错误累积。

---

## 441. Designing and Evaluating Granular Consent for Data Sharing in Cardiac Disease Prevention

**arXiv ID:** 2608.03533 | [PDF](https://arxiv.org/pdf/2608.03533v1)

**作者:** Pavithren V S Pakianathan `[一作]` (Ludwig Boltzmann Institute for Digital Health and Prevention), Jan Smeddinck `[通讯]` (Ludwig Boltzmann Institute for Digital Health and Prevention)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过专家工作坊与老年心血管疾病患者的混合方法评估，研究了细粒度动态同意在健康数据生命周期中的可行性与使用体验。

**💡 创新点**

首次提出“控制‑负荷悖论”和“情境粒度”概念，揭示用户在信任与努力权衡下对同意粒度的可选择性，并给出了可配置粒度的设计指导。

**🔧 技术方法**

采用人机交互方法：专家共创工作坊、低/高粒度Figma原型、think‑aloud、半结构化访谈；量化测评包括NASA‑TLX、SUS、PIC、共享意愿量表；定性使用主题分析。

**📊 数据集**

使用了5位跨学科专家与7位55‑74岁心血管患者的参与数据，未使用公开大规模数据集，仅基于参与者的自我跟踪场景。

**📈 对比分析**

将单步低粒度与多步高粒度原型进行受试者内对比；在NASA‑TLX、SUS、PIC和共享意愿上均未出现显著差异，但多步原型在主观上更受欢迎；SUS均高于84，NASA‑TLX负荷低。

**⚠️ 局限性**

样本规模小（专家5人、患者7人），受试者为已有数字健康经验者，原型仅为设计验证，未涵盖真实复杂数据流，缺乏纵向长期使用验证。

---

## 442. Tired Actor: Fatigue-Informed Character Control

**arXiv ID:** 2608.03528 | [PDF](https://arxiv.org/pdf/2608.03528v1)

**作者:** Shengyuan Zhang `[一作]` (Shanghai Jiao Tong University), Yong-Lu Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种基于疲劳建模的角色控制器Tired Actor，使角色在物理模拟中能够自适应疲劳状态并保持动作覆盖；

**💡 创新点**

创新点在于将行为能量学中的疲劳概念引入到角色动画控制，采用三分室模型(3CC)对可用扭矩进行动态限制，并通过重加权疲劳初始化、硬负样本挖掘和下落恢复微调提升控制质量；

**🔧 技术方法**

使用强化学习( PPO+AMP+PHC)、三分室疲劳模型、三维人形SMPL、IsaacGym物理仿真、Re-weighted 初始化、Hard Negative Mining、Fall‑Recovery Finetuning等技术；

**📊 数据集**

主要使用AMASS大规模运动数据集进行训练和评估，并在CIRCLE数据集上进行跨域泛化测试；

**📈 对比分析**

与PHC基线及无疲劳版基线相比，Tired Actor在训练集上的成功率最高(99.1%)，在未见数据上保持更优的mPJPE‑L与成功率，整体表现提升约5‑10个百分点；

**⚠️ 局限性**

限制在于仍缺乏真正的等待/恢复行为、对大幅外力或极端疲劳的鲁棒性不足，且在极端外力或转向时仍会产生不自然的姿态或坠地。

---

## 443. WeClawArena: An Auditable Sandbox and Benchmark for Cross-User Agents Collaboration and Security in Human-Centered Agent Networks

**arXiv ID:** 2608.03499 | [PDF](https://arxiv.org/pdf/2608.03499v1)

**作者:** Prince Zizhuang Wang `[一作]`, Shuli Jiang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个面向人类中心代理网络的可审计沙箱与基准WeClawArena，评估多方代理在个人工作空间上的协作工具使用与安全性；

**💡 创新点**

创新点在于将个人工作空间作为任务资源与安全约束共同参与任务评估，并同时提供可审计的攻击成功率（ASR）与任务成功率（TSR）分离评估；

**🔧 技术方法**

技术主要包括Docker容器化个人工作空间、OpenClaw运行时沙箱、消息网关、工具调用接口以及LLM（如Claude Opus 4.7）作为审计判定者；

**📊 数据集**

使用包含124个基线任务和620个攻击/协作场景的自定义数据集，涵盖谈判、竞标、旅行、软件工程工作空间、临床和交易六个领域；

**📈 对比分析**

通过在不同模型（Claude Opus、Claude Sonnet、DeepSeek、Kimi、Qwen等）上跑TSR和ASR两种指标进行比较，结果显示Claude Opus 4.7在多领域表现最优，但不同攻击向量和领域的抵抗力差异显著；

**⚠️ 局限性**

局限性包括：任务难度高度依赖领域、攻击成功判定依赖LLM裁判者的可靠性、对攻击向量的手工设计与扩展成本较高、以及对长周期协作和复杂权限模型的覆盖仍有限。

---

## 444. Dr. AGENTONOMICS: A Didactic Experiment of AGENTONOMICS

**arXiv ID:** 2608.03524 | [PDF](https://arxiv.org/pdf/2608.03524v1)

**作者:** Fengjunjie Pan `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个基于检索增强生成的网页Tutor AI代理Dr. AGENTONOMICS，并提出其在教学、演讲、设计咨询和元代理四个累积角色的扩展路线。

**💡 创新点**

创新点在于将AI代理视为经济实体，并通过统一架构实现从教学到元代理的无缝演进，以及用元代理帮助学生生成新的经济主体。

**🔧 技术方法**

使用了大型语言模型、检索增强生成、文本转语音、视频合成、工具与向量，以及角色切换的Orchestrator等技术。

**📊 数据集**

主要依赖AGENTONOMICS手册和相关管理、经济学文献构建的知识库，外部Web检索作为补充；未明确使用公开数据集。

**📈 对比分析**

目前仅做了原型验证，尚未对比其他教学AI系统；性能评估主要通过学生交互日志和失败案例分析，后续计划进行实地实验评估。

**⚠️ 局限性**

局限包括仅实现Tutor角色，缺乏多模态交互、设计咨询和元代理功能；治理与所有权问题未解决；尚未在多中心AI经济中验证经济价值和协同效果。

---

## 445. Pivot-Centric Trajectory Prediction: Bridging Long Horizons via Dynamical Guidance

**arXiv ID:** 2608.03521 | [PDF](https://arxiv.org/pdf/2608.03521v1)

**作者:** Xiucong Zhao `[一作]` (Xi'an Jiaotong University), Hao Miao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于“枢轴点”拆分长时域轨迹预测的两阶段框架——Pivot‑Centric Trajectory Prediction（PCTP）

**💡 创新点**

创新点在于将长时域轨迹拆解为多尺度枢轴点预测与基于枢轴点的轨迹细化，既提供了全局意图指导，又显著降低了迭代误差累积；同时框架可作为插件轻松集成到现有SOTA模型中

**🔧 技术方法**

使用跨注意力的枢轴点解码器进行多尺度枢轴预测；枢轴到上下文的注意力用于细化轨迹；端到端训练采用拉普拉斯混合模型、胜者为王匹配策略与分类损失

**📊 数据集**

在Argoverse I（2 s历史→3 s预测）和Argoverse II（5 s历史→6 s预测）两个大规模自动驾驶数据集上评估

**📈 对比分析**

与LaFormer、HPNet、DenseTNT、QCNet等基准模型结合后，在两大数据集上均提升了minADE、minFDE、b‑minFDE和MR指标；以QCNet为例，在Argoverse II leaderboard 上以 1.854 b‑minFDE、0.630 minADE、1.235 minFDE、0.152 MR 成为所有单模型的最优结果

**⚠️ 局限性**

局限性包括：枢轴点定义主要基于时间采样，未充分利用地图几何或交互细节；在极端情景下对枢轴预测误差的敏感性仍有待进一步评估

---

## 446. GVCCTurbo: Rate-Compute Quality Scheduling for Codebook Driven Generative Compression

**arXiv ID:** 2608.03517 | [PDF](https://arxiv.org/pdf/2608.03517v1)

**作者:** Ziyue Zeng `[一作]` (Waseda University), Hiroshi Watanabe `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于预训练生成模型的压缩框架GVCCTurbo，采用BPP驱动的调度方案在不重新训练的前提下将先验刷新与代码书校正解耦，显著减少先验评估次数并提升解码效率；

**💡 创新点**

创新点在于通过端点缓存实现先验刷新与校正的分离，并引入一次性校准的跳过间隔与原子数参数，使得BPP成为调度输入，兼容多种生成压缩后端；

**🔧 技术方法**

使用的技术包括预训练生成模型（如Rectified-Flow与Diffusion）、代码书投影校正、端点缓存刷新、BPP驱动的调度计算器以及多尺度、端点一致的解码重播；

**📊 数据集**

实验数据集涵盖UVG、HEVC Class B、MCL-JCV用于1080p基准，以及720p版的UVG、HEVC-B、MCL-JCV进行控制调度实验；

**📈 对比分析**

与传统HEVC/VVC及训练型生成压缩方法对比，GVCCTurbo在极低比特率下实现更低LPIPS和更高PSNR，并在720p实验中将先验评估从20次降至9次，解码时间提升约44%；

**⚠️ 局限性**

局限性包括：需要针对每个协议、分辨率和GOP单独一次性校准跳过间隔与原子数；跳过间隔不适用于高运动内容时可能略微影响质量；未测量实时编码性能与跨时间一致性；

---

## 447. How Many Labels Are Enough? ALDA: Active Learning Deployment Advisor for Medical Image Classification

**arXiv ID:** 2608.03511 | [PDF](https://arxiv.org/pdf/2608.03511v1)

**作者:** Julia Machnio `[一作]` (University of Copenhagen), Mostafa Mehdipour Ghazi `[通讯]` (University of Copenhagen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 ALDA 框架，利用短期 pilot 的学习曲线预测，在医学影像分类主动学习中挑选成本最低、阈值鲁棒的采样策略

**💡 创新点**

将学习曲线预测与风险评估结合，提出绝对标注成本 B_abs 与部署窗口 W 的决策机制，实现风险感知的策略推荐

**🔧 技术方法**

参数化学习曲线模型（PALM）+ 非线性最小二乘拟合 + 绝对成本与窗口双指标的风险‑aware 选取规则

**📊 数据集**

BRISC2025（脑肿瘤 MRI）、ISIC2019（皮肤病变）、Fetal Planes（胎儿超声）和 BUSI（乳腺超声）四个医学影像分类数据集

**📈 对比分析**

与九种主流主动学习策略对比，ALDA 在 15–30% pilot 阶段准确选出最低成本方法，标注成本可比最差策略降低 18%–82%，并在不同阈值设定下保持较低的标签后悔率

**⚠️ 局限性**

对阈值不确定性与曲线拟合的依赖导致在极端类别不平衡或数据量极少时预测误差可能增大；需要足够多的 pilot 数据以保证拟合质量

---

## 448. ChronoLens: Measuring Language Change Across Time, Languages, and Linguistic Levels

**arXiv ID:** 2608.03507 | [PDF](https://arxiv.org/pdf/2608.03507v1)

**作者:** Gagan Bhatia `[一作]` (University of Technology Nuremberg), Steffen Eger `[通讯]` (University of Technology Nuremberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了ChronoLens框架，对五种语言在多维语言层面（形态、句法、语义、语用）进行跨时空的历史语言变迁分析

**💡 创新点**

首次在同一可比特征空间中同时测量多语言、多年代、多层面变化，突破了以往层面互不兼容的研究瓶颈

**🔧 技术方法**

使用冻结的多语言大模型、跨coder（feature‑aligned cross‑coder）以及后置特征归因的弱监督任务

**📊 数据集**

规模达44.98M篇议会文件、约17.2B词元，覆盖1803–2026年、五种语言（英、德、意、波、土）

**📈 对比分析**

对稀疏特征的语言层归因与方向一致性进行评估，结果显示跨层面变化幅度相近，但不同语言的时间轨迹与方向差异显著；跨层面的稀疏特征相较于密集嵌入和无监督稀疏自编码器在语言统计上的相关性提升（ρ≈0.72）

**⚠️ 局限性**

对议会语料的依赖导致对全体人口语言的代表性不足；OCR误差、模型覆盖与分词差异仍可能影响结果；特征归因采用单一最大影响原则，可能忽略跨层面共同作用

---

## 449. FedCARE: A Multi-Objective Personalised Federated Learning Framework for Smart Healthcare

**arXiv ID:** 2608.03498 | [PDF](https://arxiv.org/pdf/2608.03498v1)

**作者:** Rojalini Tripathy `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在多目标个性化联邦学习框架FedCARE中，针对医疗机构间的目标异质性和特征空间不重叠问题，提出了两阶段训练策略：先在公共特征上学习Pareto‑驱动的共享主干模型，再让每个客户端在保留自身私有特征和本地目标的基础上进行个性化微调；

**💡 创新点**

创新点在于：①将多目标联邦优化与个性化学习结合，解决不同医院追求不同临床目标的梯度冲突；②通过两阶段分离共享与个性化，既保持全局知识共享，又充分利用机构特有特征，且在第二阶段无额外通信开销；

**🔧 技术方法**

主要技术包括Pareto多目标联邦优化、联邦梯度聚合、QP求解权重的SLSQP算法、PyTorch深度网络、Flower框架模拟、云端客机服务器部署；

**📊 数据集**

实验使用真实医疗数据集MIMIC‑III（ICU 21139记录）和Diabetes 130‑US Hospitals（101766记录），分别针对死亡率、再住院、住院时长等任务；

**📈 对比分析**

与FedAvg、FedProx、FMGDA、FSMGDA、PAFNet等基线对比，FedCARE在AUROC上最多提升12.5%（MIMIC‑III）或6.14%（Diabetes），MAE降低32%（MIMIC‑III）或20%（Diabetes），整体表现明显优于现有方法；

**⚠️ 局限性**

局限性包括：未考虑动态客户端加入与离线（straggler）情况；缺乏差分隐私或安全聚合等更严格的隐私保护机制；个性化阶段依赖本地算力，跨机构聚类与协同个性化尚未实现；

---

## 450. Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension

**arXiv ID:** 2608.03494 | [PDF](https://arxiv.org/pdf/2608.03494v1)

**作者:** Raviraj Joshi `[一作]` (NVIDIA), Niranjan Wartikar `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在 Nemotron-3-Nano-30B-A3B 上扩展印地语词表的嵌入初始化策略，系统评估了 20+ 方案并提出异步子词组合与归一化校准的最佳初始化流程。

**💡 创新点**

创新点在于将输入输出嵌入分离并采用子词加权平均与语言特定范数校准，结合轻量级 50 步 CPT 作为策略选择标准，显著加速低资源语言适配。

**🔧 技术方法**

采用了子词加权平均、字符长度加权、语义相似权重、FOCUS、top-k 检索、残差 MLP、归一化校准以及轻量级 CPT 探针等技术。

**📊 数据集**

使用 Nanda 词表抽取的 25,600 个印地语子词、Nemotron v3 的混合 Hindi/English 预训练数据，以及 100,000 篇 Sangraha Hindi 文档进行验证。

**📈 对比分析**

通过对比初始化损失、bits-per-byte、50 步验证损失和 MILU-Hindi 基准，最优配置在 500 步即可比默认 Mean-all 达到 62.81 分，速度提升约 7 倍。

**⚠️ 局限性**

局限在于仅针对印地语、Nemotron-3 与 Nanda 词表，未验证跨语言/跨 tokenizer 的泛化，且 50 步 CPT 作为选择标准需在更多任务中验证。

---

## 451. RAG-Stack: Co-Optimizing RAG Serving Performance and Quality

**arXiv ID:** 2608.03487 | [PDF](https://arxiv.org/pdf/2608.03487v1)

**作者:** Haiqiang Zhang `[一作]` (ETH Zurich), Wenqi Jiang `[通讯]` (National University of Singapore)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套RAG系统配置与性能的Pareto前沿搜索框架，自动化在算法与系统设计空间中寻找质量与吞吐/延迟折衷点。

**💡 创新点**

创新点在于：①基于阶段子指标的全局多目标贝叶斯优化，既保留跨阶段交互，又用子指标引导搜索；②通过工作负载抽象（Intermediate Representation）将执行过程与部署解耦，实现不需部署即可预测性能；③融合机器学习与解析式的多层成本模型，既精确又可迁移到新硬件。

**🔧 技术方法**

使用的技术包括：多目标贝叶斯优化（LogNEHVI）、阶段诊断子指标、工作负载抽象、ML-解析混合成本模型、仿真式性能预测、FAISS向量检索模型、GPU/CPU并行调度。

**📊 数据集**

实验数据集为RAGEval（100条查询）和MS MARCO（100条查询），以及用于成本模型校准的ELI5、TriviaQA等检索语料。

**📈 对比分析**

与GP+LogNEHVI、SMAC、Greedy-Forward等基线相比，RAG-PE在RAGEval上提升了约52.5%在MS MARCO上提升约153.2%的归一化超体积，搜索效率也更快（如RAGEval 4.59h vs 5.60h）。

**⚠️ 局限性**

局限性包括：对agentic RAG仅支持trace驱动的性能预测，无法提前进行性能采样或过滤SLO违背配置；缺乏trace‑free静态预测路径，需进一步扩展。

---

## 452. Getting to the Root: A Combined Complexity Perspective on Consistent Query Answering

**arXiv ID:** 2608.03477 | [PDF](https://arxiv.org/pdf/2608.03477v1)

**作者:** Miika Hannula `[一作]` `[通讯]` (University of Tartu), Miika Hannula (University of Tartu)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在自连接自由、无重复键、原子唯一主键且攻击图无环的布尔合取查询下，CQA的组合复杂度，并给出了一个多项式时间算法。

**💡 创新点**

提出了键-非键图、攻击传播图、变量保护等新概念，并通过数据库与查询饱和、归一化等预处理技术，将问题转化为可通过多项式时间递归算法解决的形式。

**🔧 技术方法**

使用了图论（强连通分量、攻击图、键-非键图）、代数化简（查询饱和、归一化）、以及变量保护与受限替代技术，实现了组合复杂度的分析与算法设计。

**📊 数据集**

未使用具体数据集，本文完全基于理论分析与构造。

**📈 对比分析**

与之前仅在数据复杂度层面的分类相比，本文证明在源SCC数有限时组合复杂度为P，而无此限制时为Π₂ᵖ‑complete，展示了对复杂度的精细划分。

**⚠️ 局限性**

局限于自连接自由、原子唯一主键、攻击图无环等条件，且对复合主键及非自连接查询的扩展仍待研究。

---

## 453. Adaptive Modality Reliability Diagnosis and Restoration for Robust Multimodal Intent Recognition

**arXiv ID:** 2608.03475 | [PDF](https://arxiv.org/pdf/2608.03475v1)

**作者:** Suraj Kumar `[一作]`, Ayan Dutta `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 PRIME 框架，能够在多模态意图识别中自适应诊断、修复并重新评估每个样本的模态可靠性，从而提升鲁棒性。

**💡 创新点**

创新点包括：①利用能量、知识不确定性、跨模态一致性和特征质量四种诊断探针显式估计模态可靠性；②通过自监督的模态破坏与严重程度回归实现无标注可靠性学习；③采用原型条件变分恢复模块修复受损模态；④闭环可靠性更新与精度加权融合。

**🔧 技术方法**

核心技术包括 Transformer‑基上下文可靠性估计器、能量与熵的置信度度量、轻量级集成求取经验不确定性、Jensen‑Shannon 散度的跨模态一致性评估、原型记忆检索、变分残差生成、异方差不确定性损失、以及精度加权的逆方差融合。

**📊 数据集**

实验使用多模态对话基准 MIntRec 与其多方位子集 MIntRec2.0，涵盖文本、音频和视频三种模态。

**📈 对比分析**

与 11 种基准方法（MISA、MulT、MAG‑BERT、TCL‑MAP、MVCL‑DAF、WDMIR、MIntOOD、ECFMIR、CDPR、CR‑3WD、HIER）在 Accuracy、Precision、Recall、F1、Weighted Precision/F1 等指标上对比，PRIME 在两大基准上均取得最高分，尤其在缺失、噪声、冲突和文本主导场景下显著提升鲁棒性。

**⚠️ 局限性**

局限性包括：①依赖合成模态破坏来生成可靠性监督，真实噪声模式可能未被充分覆盖；②主要针对文本、音频、视频三模态，未扩展到更广泛的模态或任务；③恢复模块和多模态融合增加了计算开销；④在更大规模、多任务场景（如医疗或情感计算）下的泛化尚未验证。

---

## 454. Certified Split Points for Parallel Lexing: Exact and Modulo Discarded Tokens

**arXiv ID:** 2608.03473 | [PDF](https://arxiv.org/pdf/2608.03473v1)

**作者:** Nicklas Nidhögg `[一作]` `[通讯]` (Independent Researcher), Nicklas Nidhögg (Independent Researcher)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种仅依赖已编译词法分析器的静态分析方法，用来识别可安全作为分块边界的单字节符号（certified split symbol），并基于此构建分块规划器与并行执行器，使得在满足条件的输入上可无重叠、无恢复的并行扫描。

**💡 创新点**

创新点包括：
① 给出了对已完成最长匹配扫描器的必要且充分的符号安全性条件，并证明其可在 DFA 的可达且可接受子自动机上线性求解；
② 设计了一个更宽松但保守的“按丢弃符号集”下的条件，允许在某些情况下恢复并行扫描；
③ 将此条件实现为编译时常量位图，规划器在扫描前只需一次线性搜索；
④ 通过多种真实 token 组（C、JSON、日志等）和大规模数据集验证该方法的适用性与性能提升。

**🔧 技术方法**

主要技术包括：
- DFA 可达性与可接受性前向/后向遍历得到 Q⁺；
- 重新进入性（re‑entrancy）检测；
- 仅对 Q⁺ 中的状态进行的常数时间位图查询；
- 分块规划器按等距预估位置向右扫描寻找下一个证书；
- 并行执行器按分块顺序串联每块 token 序列；
- 在评测中使用多线程并行性和不同 CPU 排程方式。

**📊 数据集**

实验使用两类大型文本数据集：
1. 512 MiB 稠密源代码（C/JSON）
2. 512 MiB 日志行（纯行尾分隔）
每类数据分别按 1 MiB、16 MiB、128 MiB、512 MiB 进行多次测量，覆盖不同缓存与内存压力。

**📈 对比分析**

对比方法：
- 单线程串行扫描（plain scan）；
- 并行 API 但仅使用一块（one‑chunk baseline）来排除规划与同步开销。
性能表现：
- 在八线程下，512 MiB 稠密数据的并行效率可达 95.3%，相对串行扫描实现 3.46×–3.94× 的整体加速；
- 在 1 MiB 情形下效率约 80%，显示规划与线程启动开销显著；
- 当证书稀缺或不存在时，规划时间可达 16 ms（占总时间 1.9×），并行优势下降。

**⚠️ 局限性**

限制：
- 仅适用于单一初始状态、纯 DFA 的最长匹配扫描器；
- 对 lexer 模式、缩进栈、语义谓词等额外状态无支持；
- 若 token 组不满足条件（无可证书符号），并行方法失效；
- 规划阶段需要一次前向扫描，若输入非常大且证书稀少会产生较大开销；
- 仅保证 token 序列与长度一致，对外部副作用（如写文件、日志）不作保证。

---

## 455. LeanMem: Simple and Efficient Long-Term Memory for LLM Agents

**arXiv ID:** 2608.03463 | [PDF](https://arxiv.org/pdf/2608.03463v1)

**作者:** Yuxin Liao `[一作]` (Hefei University of Technology), Zishu Wang `[通讯]` (Hefei University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LeanMem 长期记忆框架，将对话内容分为稳定属性、可演化事件和细节密集记录三种类型，并通过控制写入、选择性演化与自适应证据合成实现高效记忆管理。

**💡 创新点**

创新点在于：① 针对对话信息的可压缩性、时序动态和保真需求设计三种记忆存储方式；② 只对事件记忆进行局部演化，减少在线维护成本；③ 根据查询的证据需求动态规划检索预算，避免固定检索导致的冗余。

**🔧 技术方法**

主要技术包括：LLM 驱动的关键话语过滤与主题分段、写入调度（基于稳定性、时序性、保真度的分类）、事件记忆的局部离线演化、以及检索规划与重排序的自适应证据合成；实现基于 GPT‑4.1‑mini / Qwen3‑8B 的推理。

**📊 数据集**

使用了两大长对话 QA 数据集：LoCoMo（约10条对话，16K tokens/会话）和 LongMemEval‑S（500条对话，115K tokens/会话）。

**📈 对比分析**

与 A‑Mem、LightMem、SimpleMem 等基线相比，LeanMem 在 LoCoMo 上可提升 5.5–15.0 分准确率，构造/推理 tokens 下降 17–8 倍，Latency 下降 1.4–2.7 秒；在 LongMemEval‑S 上准确率提升 15.1 分，构造 tokens 降至 117.6K，推理 tokens 仅 3.6K，Latency 仅 2.16 秒。

**⚠️ 局限性**

局限性包括：① 需要多轮 LLM 调用（写入调度、事件演化、检索规划），在极大规模对话或低算力环境下仍可能受限；② 对非文本或多模态对话的适用性未验证；③ 依赖精细设计的路由规则和硬编码阈值，可能在不同语言或对话风格中需要重新调优。

---

## 456. Solver-Aware Decompositions for Programming-by-Example: When Dividing Requires Knowing how to Conquer

**arXiv ID:** 2608.03461 | [PDF](https://arxiv.org/pdf/2608.03461v1)

**作者:** Janis Zenkner `[一作]` (Clausthal University of Technology), Christian Bartelt `[通讯]` (Clausthal University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为 SAD 的训练框架，针对分解式程序合成（PBE）任务，利用冻结的合成器对分解器进行强化学习，以提高分解子目标的可解性，从而提升整体合成准确率。

**💡 创新点**

创新点在于：①认定分解质量不是与真实分解（gt）相似度，而是相对合成器的可解性；②在训练时同时保持对 gt 分解的监督，形成结构化基底；③通过基于合成器交叉熵损失的奖励信号，形成 solver‑aware 的优势（SCST）优化；④通过 “accuracy paradox” 等实验揭示 gt 分解并非最佳。

**🔧 技术方法**

技术主要包括：Transformer‑based 分解器与合成器模型；冻结合成器并对其 CE 损失进行奖励；SCST（Self‑Critical Sequence Training）对分解器进行强化学习；多步推理的 beam search；实验中对 Deepcoder、LambdaBeam 与 RobustFill 三个 PBE 域进行评估。

**📊 数据集**

使用公开的 PBE 数据集：Deepcoder（整数列表）和 LambdaBeam（带条件的 lambda 函数扩展），以及 RobustFill（字符串变换），分别在分布内和长度泛化（train on ≤n，test on >n）两种设置下进行训练和评测。

**📈 对比分析**

与传统 solver‑blind（仅监督 imitation）以及 ExeDec 的比较：SAD 在 Deepcoder、LambdaBeam 的任务准确率均显著提升，尤其在长度泛化场景中提升幅度更大；同时在 RobustFill 上表现无显著差异，验证了方法的依赖性。实验展示了“accuracy paradox”，证明更高的 gt 对齐不一定带来更好的合成效果。

**⚠️ 局限性**

局限性：①需要冻结并固定合成器，若合成器更改需重新训练分解器；②单步奖励设定，无法直接处理多步分解与深层子目标；③仅适用于有 gt 程序的训练场景，缺乏基于执行反馈的无标签版本；④在解空间较为确定（如 RobustFill）时方法无优势，证明其依赖于分解歧义性。

---

## 457. Detecting Pose Estimation Failures via Keypoint Self-Consistency

**arXiv ID:** 2608.03516 | [PDF](https://arxiv.org/pdf/2608.03516v1)

**作者:** Robin Chan `[一作]` `[通讯]` (Technische Universität Berlin), Robin Chan (Technische Universität Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Meta Pose，一种基于关键点自一致性检测 6D 物体姿态估计失败的轻量级框架。

**💡 创新点**

创新点在于将手工几何一致性特征（两两距离、重投影、渲染、蒙版一致性以及关键点置信度）直接作为输入，使用简单的逻辑回归即可得到可靠的失败概率，无需额外渲染、深度网络或大规模采样。

**🔧 技术方法**

使用手工几何特征构造、逻辑回归分类器、基于渲染与 SAM 的蒙版一致性（可选），以及对关键点置信度的统计。

**📊 数据集**

在 LINEMOD Occluded 数据集（8 类物体，包含遮挡）上进行评估。

**📈 对比分析**

与两种基线（最大关键点置信度阈值、基于 conformal 的关键点预测）相比，Meta Pose 在 5° 与 10° 旋转误差阈值下的 AUROC、F1、AUPRC 均显著提升；渲染-无关的子模型几乎与完整模型相当，且运算速度最快。

**⚠️ 局限性**

局限性包括：1）对 Segment Anything Model 的蒙版质量依赖；2）仅针对旋转误差做失败检测，未覆盖位移或边框回归；3）仅在 LINEMOD Occluded 评估，需在更广泛的数据集与不同骨干网络上验证。

---

## 458. Can LLM design high-quality experiments? A Comprehensive and Systematic Benchmark on Autonomous Experimental Design

**arXiv ID:** 2608.03501 | [PDF](https://arxiv.org/pdf/2608.03501v1)

**作者:** Zejun Liu `[一作]` (Westlake University), Yue Zhang `[通讯]` (Westlake University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于评估LLM实验设计能力的SCOPE基准，并在此基础上设计了OptED工作流以提升实验设计质量。

**💡 创新点**

创新点包括：①SCOPE采用层级分解与redline机制实现细粒度评估；②OptED通过三阶段代理式工作流、工具化Think‑Act‑Observe循环和行为规范，系统性解决低级配置瓶颈与无结构搜索干扰。

**🔧 技术方法**

使用的技术包括：LLM‑as‑Judge评估框架、结构化工具接口（搜索、编辑）、阶段隔离与多步思考-行动-观察循环、行为规范检查、红线（redline）机制。

**📊 数据集**

数据集为从ICML/NeurIPS/ICLR等顶级会议选取的300篇论文，覆盖19个研究领域，构成SCOPE实验设计任务集。

**📈 对比分析**

通过在CoT‑only与CoT+Search两种提示策略下，对7大主流LLM进行评估，平均得分仅为14.81/30，最高18.62。OptED在6模型上平均提升High‑Level 1.6分、Low‑Level 1.2分，整体得分提升至约20分，红线率显著下降。

**⚠️ 局限性**

局限性：低级资源配置仍显不足，High‑Level与Low‑Level得分差距持续存在，OptED虽提升整体质量，但未根本解决低级配置难题；模型对外部知识的整合仍需更精细化。

---

## 459. LLM-Assisted Review Prioritization for German Statutory Health Insurance Websites: A Multi-Stage Corpus Audit

**arXiv ID:** 2608.03500 | [PDF](https://arxiv.org/pdf/2608.03500v1)

**作者:** Martin Möller `[一作]` `[通讯]`, Martin Möller

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对德国法定健康保险基金网站的56,198页内容实施了分层审核工作流，包括确定性筛选、LLM低成本预筛、深度审核、证据核查、模式聚类和模型互评，并生成35,998条审核记录，其中21,452条进入案例审核队列。

**💡 创新点**

提出了将AI出处信号（P）与实质质量信号（Q）分离的审核框架，结合时间有效性保障和最小证据检查，能够在不做最终判断的前提下生成可追溯的审核工作负载，并通过双模型一致性量化审核流程的可靠性。

**🔧 技术方法**

使用大型语言模型（如Gemini 3.1 Pro、DeepSeek v4 Pro、Claude Sonnet 4.6等）进行低成本预筛和深度审核；采用确定性规则进行P/Q信号标记；对比两种模型输出获得Kappa；对特定页面进行模型互评。

**📊 数据集**

公开数据集为84个网站（共56,198页）在2026-05-19的快照，包含爬取记录、页面元数据、审核记录、概念映射和模式表，数据已存档于Zenodo。

**📈 对比分析**

通过300页风险加权的路由压力测试（结果为33.3%出现潜在审核信号）和182条匹配记录的双模型一致性（P_o=0.758，kappa=0.532），表明工作流能产生可审计的审核优先级，但并未提供错误率或精准度的绝对衡量。

**⚠️ 局限性**

限制包括：非随机样本，无法估计错误/不合规的普遍性；模型输出为概率性，缺乏人类参考标准；时间有效性保障不覆盖所有法律/医学更新；结果仅为审核工作负载估计，不能直接转化为合规性判断；对公共可见性信号的评估未经过独立人工验证。

---

## 460. Principles of Robot Autonomy

**arXiv ID:** 2608.03496 | [PDF](https://arxiv.org/pdf/2608.03496v1)

**作者:** Daniele Gammelli `[一作]`, Marco Pavone `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本书系统阐述机器人自治的基本原理、方法与实践，整合感知、定位、规划与决策的全流程，提出See‑Think‑Act框架，并以ROS为平台提供实现示例与交互式笔记本；

**💡 创新点**

创新点在于将经典模型驱动算法与现代端到端机器学习方法结合，形成从硬件接口到高层决策的统一视角，强调软件中介与可组合性的设计哲学；

**🔧 技术方法**

使用的技术包括计算机视觉、传感器融合、滤波与SLAM、运动规划、强化学习、模仿学习、深度学习（如VLA模型）以及ROS（1与2）等中间件；

**📊 数据集**

涉及的数据集主要为公开的机器人与自动驾驶数据集，如Waymo自动驾驶数据、NASA Astrobee的飞行传感器日志、Boston Dynamics SpotMini与Tesla Optimus演示视频等；

**📈 对比分析**

章节通过案例实验与Python实现对比，展示传统基于模型的规划与端到端学习在速度、精度、适应性上的差异，强调在不同应用场景下的性能折衷；

**⚠️ 局限性**

局限性包括：1) 对实时与安全关键系统的ROS1缺乏硬实时保障，需要迁移到ROS2；2) 端到端模型对大规模数据依赖高，缺乏解释性；3) 书中主要聚焦理论与实现示例，缺乏大规模实地验证与性能基准。

---

## 461. Continue or Replan? Bernoulli-Continuation Policy Learning for Adaptive Horizon Execution

**arXiv ID:** 2608.03483 | [PDF](https://arxiv.org/pdf/2608.03483v1)

**作者:** Weichen Xu `[一作]`, Baining Guo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 Bernoulli-Continuation Policy (BCP)，通过学习可变执行时长来动态调整 VLA 的重规划时机。

**💡 创新点**

创新点在于把重规划时机视为可学习的决策，使得机器人能够在不同任务阶段自动决定何时停止执行当前动作块并重新规划。

**🔧 技术方法**

使用 Transformer 作为 BCP 的主干，结合 GRPO 强化学习框架和 Replanning-Efficiency Reward (RER) 对执行时长进行训练，并在多种 VLA 基础模型（LingBot‑VLA‑4B、ABot‑M0、ACT、π_0.5）上实现。

**📊 数据集**

数据集包括真实世界 AGIBOT G1 上的两项任务（Grasping Bottle、Hanging Mug）以及 RoboTwin 2.0 与 LIBERO 的 50 项仿真任务，用以评估 BCP 在不同环境下的性能。

**📈 对比分析**

通过与固定时长、Softmax 头、BC 头等基线进行对比，BCP 在 RoboTwin 2.0 上将平均成功率从 89.88% 提升至 93.94%，在 Hanging Mug 任务中从 41% 提升至 87%，并显著降低了运行时长和 VLA 调用次数。

**⚠️ 局限性**

局限性在于 BCP 只调整重规划时机，无法纠正 VLA 生成的动作块本身的错误，因而仍依赖基础 VLA 的动作质量。

---

## 462. LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models

**arXiv ID:** 2608.03457 | [PDF](https://arxiv.org/pdf/2608.03457v1)

**作者:** Fengqi Zhu `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了稀疏混合专家（MoE）扩散语言模型（dLLM）的优化、计算分配与架构扩展规律，并基于这些规律从零开始训练了30B参数的LLaDA MoE v2模型；

**💡 创新点**

创新点在于首次给出MoE dLLM的缩放规律，揭示其与传统自回归模型在批量大小、学习率、数据侧倾斜、激活比率及共享专家比例等方面的显著差异，并利用这些规律实现更高效的模型训练；

**🔧 技术方法**

采用的技术包括掩码扩散语言建模、Mixture-of-Experts Transformer、IsoFLOP 分析、功率律拟合、超参数搜索以及仅用监督微调（SFT）的指令适配；

**📊 数据集**

训练使用约23.5T的通用文本数据；评估采用多项公开基准，包括MMLU、MMLU-Pro、CEval、CMMLU、HellaSwag、KorBench、GSM8K、MATH、OlympiadBench、CRUXEval、MBPP、MultiPL-E、HumanEval、LiveCodeBench、BigCodeBench 等；

**📈 对比分析**

与多种基准模型（Qwen3、SDAR Sci、LLaDA 7B-A1B 等）进行对照，LLaDA MoE v2 在绝大多数任务上与 Qwen3 相近或更优，同时在使用更少预训练 token 的条件下，SFT 后的模型在推理与代码生成任务上超过 SDAR Chat；

**⚠️ 局限性**

局限性包括：实验仅在单一维度上递增，未探究多维度交互；训练数据来源未公开；仅使用监督微调，未结合强化学习；并且仅验证了单一规模的 MoE dLLM，缺乏更大规模或更多配置的进一步验证。

---

## 463. Cross-Lingual Bias in Large Language Models: A Comparative Analysis of English and Swahili

**arXiv ID:** 2608.03532 | [PDF](https://arxiv.org/pdf/2608.03532v1)

**作者:** Ruolei Zhang `[一作]` (University of Birmingham), Yue Feng `[通讯]` (University of Birmingham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建英–斯瓦希里对称提示，使用GPT‑5.2与Gemini 2.5 Flash在19,600条生成任务中评估跨语言偏见、情感、拒绝行为和语义相似度。

**💡 创新点**

首次系统比较两种语言的生成偏见，发现偏见在语言间“转化”而非“转移”，并揭示GPT‑5.2拒绝行为仅在英语提示上触发。

**🔧 技术方法**

使用descriptor‑template提示生成、Claude Sonnet 4.5评判器、卡方/Fisher检验、Bonferroni校正、自动语义相似度判定以及Qwen‑2.5‑3B的logit lens分析。

**📊 数据集**

自建4,900条英–斯瓦希里提示对（涵盖九个偏见轴），辅以SAD v1交叉验证和Qwen‑2.5‑3B内部层分析。

**📈 对比分析**

分别比较两模型在英、斯瓦希里中的偏见率、情感分布、拒绝次数和语义相似度；结果显示在不同轴上偏见率差异显著，情感转变更为明显，拒绝行为仅出现在英语提示，语义相似度低于55%，表明跨语言行为差异显著。

**⚠️ 局限性**

局限性包括机器翻译提示的语法与词汇错误、评判器对斯瓦希里的理解有限、仅评估英–斯瓦希里两语与两款专有模型，难以推广至更低资源语言或其他模型。

---

## 464. Reversing Arrows in Large Language Models

**arXiv ID:** 2608.03512 | [PDF](https://arxiv.org/pdf/2608.03512v1)

**作者:** Sefika Efeoglu `[一作]` (Freie Universität Berlin), Adrian Paschke `[通讯]` (Freie Universität Berlin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估指令调优大型语言模型（LLM）在逆向关系（inverse relations）方向性识别的能力，构造 5,457 条句子级逆向关系对，使用零样本多选提示，并通过加入关系描述、使用 synthetic 实体与完全掩码实体三种实体处理方式来探究模型对逆向关系的理解与实体熟悉度的影响。

**💡 创新点**

①首次在句子级别对 LLM 的逆向关系方向性进行系统性评估；②构造基于 FewRel 与 TekGen 的逆向关系基准；③通过关系描述与实体匿名化的对比实验，细化逆向关系识别中模型的方向性、描述信息利用与实体熟悉度三大因素的作用。

**🔧 技术方法**

零样本多选提示（多选题形式），关系描述加入/剔除；实体处理：Presidio 生成 synthetic 实体，完全掩码实体；宏 F1 作为评估指标；Wilcoxon 符号秩检验与 95% 置信区间分析对实验结果进行统计验证。

**📊 数据集**

从 FewRel 1.0 与 TekGen（TEXT2KGBENCH）两大公开数据集抽取 27 对逆向关系标签，最终得到 5,457 条句子级样本作为评估基准。

**📈 对比分析**

在五个开源 LLM（Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B、Qwen3‑4B、Flan‑T5 XL）上进行原始、synthetic、masked 三种实体处理以及加入/不加入关系描述的对照实验。结果显示：①LMM 对逆向关系方向性存在显著的不对称性，尤其在 FewRel 上；②关系描述在大多数模型和设置中提升有限，无法形成一致的正面效果；③实体熟悉度对性能影响显著，synthetic 与 masked 实体普遍导致 F1 降低；整体宏 F1 介于 30%–70% 之间，表明逆向关系识别仍具有挑战性。

**⚠️ 局限性**

①实验仅基于英语公开数据集，缺乏跨语言验证；②实体匿名化（synthetic 与掩码）可能改变原句语义，导致性能变化无法完全归因于实体熟悉度；③未构建新的逆向关系基准，只复用已有数据集，结果可能受数据集特性偏倚；④未能直接证实模型是否因记忆而取得高分，仍需要进一步研究。

---

## 465. From Multi-Resolution Cells to Gigapixel Whole Slide Images Foundation Model for Computational Pathology

**arXiv ID:** 2608.03508 | [PDF](https://arxiv.org/pdf/2608.03508v1)

**作者:** Basit Alawode `[一作]` (Khalifa University of Science and Technology), Sajid Javed `[通讯]` (Khalifa University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了多分辨率金字塔变压器（MRPT），通过层级聚合细胞、组织和整片图像的信息，实现了对整片数字化组织切片的统一表征。

**💡 创新点**

核心创新在于引入连续跨分辨率注意力（CCRA）机制，仅在相邻分辨率间进行交叉注意，既保证语义一致性，又模仿病理学家逐级放大观察的诊断流程；同时提出三阶段多分辨率自监督学习框架，将细胞、补丁、区域、WSI 四级特征串联学习。

**🔧 技术方法**

技术包括多分辨率 ViT（mViT_P-C、ViT_R-P、ViT_M-R）、CCRA、层级聚合、三阶段自监督预训练（DINO）以及后续的 MRPT‑LLaVA 视觉‑语言模型融合。

**📊 数据集**

使用了 36,000 张 TCGA/CPTAC 整片图像、624M 个补丁、2.4M 区域以及 34 个公开任务数据集（Patch 级别 17 组、WSI 级别 10 组）进行训练和评估。

**📈 对比分析**

与当前 SOTA 视觉基础模型、视觉‑语言模型及多模态 LLM 进行对比，MRPT 在 34 个任务中平均提升 5–10% 的准确率，零样本 WSI 分类的平衡准确率达 80.6%，VQA 任务中 80.8% 的准确率，报告生成等任务均超过 10% 的 BLEU/ROUGE 指标。

**⚠️ 局限性**

局限性包括：仍依赖高质量多分辨率配准，跨分辨率注意力在极高分辨率或不同扫描仪产生的尺寸差异时可能需要进一步校正；模型虽然参数相对较少，但预训练和多阶段训练仍需大量 GPU 资源；尚未整合基因、影像等多模态信息。

---

## 466. Leveraging System-Level Observations to Inform Bayesian Learning of Model Parameters for Quantitative Verification

**arXiv ID:** 2608.03489 | [PDF](https://arxiv.org/pdf/2608.03489v1)

**作者:** Simos Gerasimou `[一作]` (Cyprus University of Technology), Xingyu Zhao `[通讯]` (Wuhan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于系统级观测的先验知识（PK）挖掘与嵌入方法，用于贝叶斯学习的定量验证（PMC），通过推断模型转移参数的概率分布实现对未知参数的估计。

**💡 创新点**

创新点：①不要求专家直接给出模型转移概率，而是利用可观测的系统属性（如可靠性、响应时间）作为先验；②利用参数化模型检查（ParaMC）提取属性的代数式，避免反复调用模型检查器；③将知识挖掘与嵌入两阶段转换为多目标进化优化，生成Pareto前沿供决策者选择。

**🔧 技术方法**

核心技术：参数化模型检查、贝叶斯推断（Beta/Gamma先验）、多目标进化算法（NSGA‑II、SPEA2、CMA‑ES）、KL散度/JS/Wasserstein度量，用于评估和优化先验分布。

**📊 数据集**

数据集：果拾机器人（Fruit‑Picking Robot, FPR）与外汇交易服务系统（FX）两套真实案例，包含多种变体（FPRc、FPR1、FPR2、FPR3、FX1、FX2、FX3），每个变体定义不同的PK属性集合Y、难测属性集合Z和未知转移参数X。

**📈 对比分析**

比较方法：在30次独立实验中，对NSGA‑II、SPEA2、CMA‑ES三种算法的Pareto前沿进行Hypervolume、ε、IGD指标评估，并与随机搜索对比。实验结果表明NSGA‑II/SPEA2在所有指标上显著优于CMA‑ES，且生成的分布与真实分布的KL误差小于0.004，验证了方法的准确性和有效性。

**⚠️ 局限性**

局限性：①方法高度依赖专家提供的PK准确性，PK冲突会导致高KL和解的多样性；②非可辨识性导致存在多组参数分布产生相同属性分布；③符号代数式提取对复杂模型成本高；④目前仅适用于离线设计阶段，未涵盖运行时自适应或在线更新。

---

## 467. Pilot-Assisted Faster-than-Nyquist Signaling for HRLLC: A Non-Asymptotic Approach

**arXiv ID:** 2608.03479 | [PDF](https://arxiv.org/pdf/2608.03479v1)

**作者:** Ahmet Oguz Kislal `[一作]` (TUBITAK BILGEM), Halim Yanikomeroglu `[通讯]` (Carleton University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在超越奈奎斯特（FTN）信号技术下，针对超可靠低时延通信（HRLLC）的短包通信性能，尤其考虑了使用导频的通道估计和功率分配优化；

**💡 创新点**

创新点在于：①在短块长度、功率有限的 HRLLC 场景下，首次推导了匹配失配解码（SNN）下的随机编码联合上界（RCUs）；②将导频和空闲符号的分配与功率分配一起优化，以最大化 SNR 利益；③通过数值模拟验证 FTN 在短块长度下可实现约 2 dB 的 SNR 提升。

**🔧 技术方法**

使用技术包括：FTN 调制、块衰落信道建模、最优导频与空闲符号分配、线性无偏估计（BLUE）、匹配失配 SNN 解码、RCUs 非渐进可达性界限、Monte Carlo 评估与参数 s 优化。

**📊 数据集**

使用数据集：仿真 Rayleigh 块衰落信道（i.i.d. 0,1），参数 β=0.5、块长度 n=288、频谱效率 0.233 bit/s/Hz，分别在不同的加速因子 δ∈{0.5,0.68,1} 进行评估。

**📈 对比分析**

比较方法：将 FTN 与传统奈奎斯特（δ=1）信号在相同 SNR、块长度、功率约束下进行 RCUs 上界评估；结果显示 FTN 在 10⁻¹–10⁻⁵ 的误包率区间内，SNR 约低 2 dB；功率分配优化后可进一步提升约 1 dB。

**⚠️ 局限性**

局限性：①仅考虑单输入单输出 (SISO) 块衰落；②使用高斯信号集作为随机编码，可能并非最优；③仿真仅基于 Rayleigh 信道，未涵盖多路径或频偏等实际问题；④导频与空闲符号优化仅在理想化模型下完成，实际系统实现复杂度待进一步研究。

---

## 468. Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding

**arXiv ID:** 2608.03471 | [PDF](https://arxiv.org/pdf/2608.03471v1)

**作者:** Xiuyuan Zhu `[一作]` (University of Chinese Academy of Sciences), Jian Xue `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种分层坐标表示Hi-Token以及对应的几何奖励Hi-GAR，提升生成式视觉 grounding 的定位精度。

**💡 创新点**

创新点在于将每个坐标拆分为轴专属的百位、十位、个位三层 token，构造粗细分级的生成流程；同时设计了包含IoU、坐标精度、多阈值奖励与有效性门控的 Hi-GAR 奖励，配合 Group Relative Policy Optimization 进行后训练优化。

**🔧 技术方法**

使用的技术包括自回归 VLM（Qwen2.5‑VL‑3B）、GRPO 强化学习、基于 token 的分层坐标编码以及多项式奖励函数。

**📊 数据集**

主要在 RefCOCO、RefCOCO+、RefCOCOg 三个 Referring-Expression 基准集上进行训练与评估。

**📈 对比分析**

在相同训练设定下，Hi‑Token 使 P@0.95 提升约8–12点，Hi‑GAR 进一步减少低 IoU 预测并提升整体 IoU；最终的 Hi‑R1 在 RefCOCO 系列上均超过现有专用 VLM（如 VLM‑R1、Rex‑Omni），并保持与 SFT 相近的推理速度。

**⚠️ 局限性**

局限性包括坐标离散化固定为 1000 级，导致小目标的超严格定位受限；未能单独剖析层次结构、轴专属词表和词表大小对性能的单独影响；奖励函数和有效性门控的超参数未做全面敏感性研究。

---

## 469. ToolLIFT: Lifting Tool-Specific Trajectories into Function-Level Graphs for Generalizable Tool Planning

**arXiv ID:** 2608.03468 | [PDF](https://arxiv.org/pdf/2608.03468v1)

**作者:** Xiuhui You `[一作]` (Beihang University), Ziwei Zhang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种将工具使用轨迹提升为函数级工作流图（FWG），从而实现工具规划的泛化。

**💡 创新点**

创新点包括：① 通过功能聚类将不同工具映射到共享的函数角色，完成轨迹提升；② 将工作流规划与工具选择解耦，使用FWG指导全局规划；③ 引入强化学习的源跟踪奖励，实现工具调用的可追溯数据流。

**🔧 技术方法**

技术手段包括：功能聚类（BGE-M3 + UMAP + K-means），轨迹提升，FWG构建，解耦式规划与工具选取，源门控与技能特定奖励的强化学习（GRPO）以及工作流扰动训练。

**📊 数据集**

使用了 HuggingFace、Multimedia 两个内部数据集进行FWG构建和训练，并在 DailyLifeAPIs、ToolAlpaca、Seal-Tools 三个外部数据集上进行 OOD 评估。

**📈 对比分析**

与五个基线（Tool-Planner、ToolNet、GTool、DFSDT、ToolRL）在两种 LLM 基础上比较，本文在 ID 和 OOD 任务中均获得最高的准确率、节点 F1、链路 F1，并在未见工具集上显著提升性能（OOP 4–5 分点）。

**⚠️ 局限性**

主要局限在于假设每个参数只有单一信息来源，无法处理需要从多个上下文或工具输出合成的参数。

---

## 470. When Correct Solutions Repeat: Rarity-Aware Credit Redistribution for GRPO

**arXiv ID:** 2608.03467 | [PDF](https://arxiv.org/pdf/2608.03467v1)

**作者:** Zhe Cao `[一作]` (South China University of Technology), Fangjiong Chen `[通讯]` (South China University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在强化学习可验证奖励(RLVR)框架下，提出了Cue-GRPO方法，对已验证的正确完成进行结构级信用再分配，从而提升高采样预算下的重复抽样性能。

**💡 创新点**

创新点在于将信用再分配与分区完全解耦：使用预定义的Strategy Cues实现无辅助判别器的确定性分区；通过分区条件的信用再分配规则压缩频率偏倚，同时保持正向信用的平均量不变。

**🔧 技术方法**

核心技术包括：分区条件的信用再分配(CR)规则、确定性Cue提取与聚类、有限组稳定化、GRPO基础策略、LoRA微调以及对比实验的AUC@k评估。

**📊 数据集**

实验数据集：在Qwen2.5-Math-7B和Llama-3.1-8B-Instruct上使用1000道MATH Level 3‑5问题进行微调；评估集为AIME（90道）、MATH500（500道）和GSM8K（200道），以及HLE（200道）等。

**📈 对比分析**

与GRPO、UARL、CR‑JP等方法对比，Cue‑GRPO在AIME上AUC@256提升至42.75（+4.4）并在高采样预算下表现最佳；整体训练开销仅比GRPO多约6%；在MATH500、GSM8K等近饱和基准上差距较小。

**⚠️ 局限性**

局限性：分区依赖固定的27条Strategy Cues，可能在更复杂或多样化的推理任务中失效；缺少对非数学域的泛化评估；对稀有结构的划分质量仍可能受Cue覆盖度限制。

---

## 471. ChartAnno: Evaluating MLLMs for Chart Annotation Generation

**arXiv ID:** 2608.03464 | [PDF](https://arxiv.org/pdf/2608.03464v1)

**作者:** Zhenghan Chen `[一作]` (Fudan University), Siming Chen `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了用于评估多模态大型语言模型（MLLM）在已有图表上自动生成可执行注释代码的基准任务，涵盖意图、操作、实现三个指令层级以及代码与图像两种输入方式。

**💡 创新点**

创新点在于：①首次针对图表注释生成建立专门的数据集与评测指标；②将指令抽象化为三层细度，揭示模型在理解抽象意图与具体实现上的差距；③结合规则基与LLM评判的可执行率、结构合规、语义一致与设计效果四大维度。

**🔧 技术方法**

技术方法包括：利用Python绘图库重构 1,200 张真实图表生成基准代码；构造 3,600 条层级化注释指令；设计 8+ 项规则/LLM 评测指标；在 10 款 MLLM（4 款专有、6 款开源）上进行实验。

**📊 数据集**

数据集来源于公开图表数据集与 arXiv 论文，经过筛选、重构与注释去除，包含 1,200 张图表、相应代码、图像及多层级指令。

**📈 对比分析**

与 10 款 MLLM 进行对比，结果显示专有模型总体领先，尤其在语义一致与设计效果上；开源模型 Kimi K2.5 在多项指标上逼近专有模型；更具体的指令能显著提升质量，加入图像对性能提升有限，仅在设计指标略有帮助。

**⚠️ 局限性**

局限性包括：仅支持 Python 代码；LLM 评判可能忽略细微视觉错误；任务聚焦于注释生成，未覆盖更广泛的图表能力。

---

## 472. When AI Joins the Team! A Model of How AI Adoption Relates To Social Patterns in Software Engineering Teams

**arXiv ID:** 2608.03462 | [PDF](https://arxiv.org/pdf/2608.03462v1)

**作者:** Giusy Annunziata `[一作]` (University of Salerno), Filomena Ferrucci `[通讯]` (University of Salerno)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究AI工具采纳如何影响软件团队的社区气味（社交反模式），并基于转移记忆系统（TMS）理论构建并验证测量与结构模型。

**💡 创新点**

提出AI与团队社交动态的多机制模型，区分专门化与协调维度，并将社区气味从细粒度聚合为四大类，揭示AI在不同工作维度的作用方式。

**🔧 技术方法**

采用PLS‑SEM对5个结构模型进行检验，并使用探索性与验证性因子分析构建与验证测量工具。

**📊 数据集**

使用152名活跃使用AI工具的软件工程专业人员的问卷调查数据。

**📈 对比分析**

通过路径系数、R²、f²以及HTMT等指标评估模型显著性，结果显示AI在专门化维度通过提升同行交互降低知识碎片化，在协调维度直接提升沟通质量；模型整体拟合良好。

**⚠️ 局限性**

样本规模有限、仅横断面自评数据、行业与文化差异受限，难以确立因果关系。

---

## 473. Probing Character-level Transformers for the Spanish L-shaped Morphome

**arXiv ID:** 2608.03452 | [PDF](https://arxiv.org/pdf/2608.03452v1)

**作者:** Akhilesh Kakolu Ramarao `[一作]` (Heinrich Heine University), Dinah Baer-Henney `[通讯]` (Ruhr-Universität)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对5种字符级Transformer在无词干重合词的训练下，探究其是否在内部表示中编码西班牙语L形形态学模式。

**💡 创新点**

发现模型不仅能再现表面形态变化，还在解码器中对词干结尾位置形成类特定编码，且该编码由训练词汇决定而非架构。

**🔧 技术方法**

使用线性探针（Logistic回归）和词干位置精确读取，结合表面n-gram基线和控制实验进行表示层分析。

**📊 数据集**

基于公开的西班牙语动词12格体在Seseo IPA形式的训练集，包含333个词条（233训练、34验证、66测试）和7个L形词。

**📈 对比分析**

通过与表面基线和随机标签控制对比，发现L形类在最佳层的线性可解码率达0.70–0.80，优于表面信息且显著高于随机；位置无关架构在无表面证据时表现更佳。

**⚠️ 局限性**

限制在仅有7个L形测试词、样本不平衡、只用教师强制的解码器状态，且可能部分可解码与音位信息重叠。

---

## 474. A repository for discovery and reuse of higher-order network datasets

**arXiv ID:** 2608.03491 | [PDF](https://arxiv.org/pdf/2608.03491v1)

**作者:** Florian Frantzen `[一作]` (RWTH Aachen University), Michael T. Schaub `[通讯]` (RWTH Aachen University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 Aachen Higher-Order Repository of Networks（AHORN）——一个标准化、高阶网络数据集的可发现、可验证、可版本化的存档与平台。

**💡 创新点**

创新点在于统一的交换格式、自动化的提交与验证工作流、与 Zenodo 版本化托管的集成、以及为高阶网络研究提供专门的检索与 API 访问方式。

**🔧 技术方法**

技术包括自定义 line‑based 纯文本格式（v0.3）、Python 与跨语言 CLI 工具、RESTful API（/api/datasets.json）、格式验证器、以及与 Zenodo 的 DOI 版本化发布。

**📊 数据集**

使用了 83 个公开来源的数据集，涵盖协作/引用、社交接触、生物医药、几何三角化等多领域，形式包括超图、单纯复形、多元组合等。

**📈 对比分析**

通过可版本化下载与程序化检索，实现了可复现的基准对比；虽然本文未给出具体算法性能指标，但提供的版本化数据与统一接口使得不同方法可在相同数据上直接比较。

**⚠️ 局限性**

局限包括：数据源受限于公开许可和转换工艺；元数据覆盖不均；目前的格式不支持有向交互或轨迹数据；许可信息仅记录而不授权；与社区软件的兼容性仍在完善中。

---

## 475. MinerU.Chem: A High-Precision System for Optical Chemical Structure and Reaction Recognition

**arXiv ID:** 2608.03525 | [PDF](https://arxiv.org/pdf/2608.03525v1)

**作者:** Haote Yang `[一作]` (Shanghai Artificial Intelligence Laboratory), Conghui He `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 MinerU 在线平台上实现了化学文献的后处理层 MinerU.Chem，自动识别并解析有机化学文献中的分子结构描绘和反应方案，生成与原文定位关联的 Molecule Summary List 与 Reaction Summary List。

**💡 创新点**

创新点包括：① 将 CARBON 作为核心图形表示，既保留二维坐标又兼顾非标准键、Markush 片段等复杂化学语义；② 通过三类数据（公开数据、人为标注、合成数据）构建训练集，系统性覆盖 MOSAIC 视觉与化学复杂度；③ 在 MolRecBench‑Wild 评测中，SMILES 精确匹配准确率达 93.02%，图形准确率 79.66%，远超现有 OCSR 与大模型。

**🔧 技术方法**

核心技术：GTR‑VL 视觉语义模型、MolYOLO 检测器、CARBON 结构图生成、基于坐标的后处理将 CARBON 转换为 MolFile/SMILES，整体流程集成在 MinerU 文档解析管线之上。

**📊 数据集**

数据集：① 公开 OCSR 数据（已转换为 CARBON）；② 来自 MolRecBench‑Wild 的真实文献人工标注；③ 采用 MOSAIC 框架合成的 37 种视觉/化学难度组合数据。

**📈 对比分析**

与 19 种对比系统（专用 OCSR、商业 OCR、通用多模 LLM）在 MolRecBench‑Wild 进行比较。MinerU.Chem 在完整集、子集 A、B、C 上分别取得 SMILES 98.28%、93.49%、70.13% 与图形 92.15%、80.11%、55.42%；在整体指标上比最佳对手 GPT‑5.6‑Sol 高 18.15pp，Gemini‑3.5‑flash‑thinking 高 43.25pp。

**⚠️ 局限性**

局限性：① 对 Markush 结构、R‑group 替换模式的自动展开仍不支持；② 对复杂机理图（电子推移箭头、催化循环）和条件筛选表的语义解析有限；③ 在极高视觉/化学复杂度（子集 C）时性能显著下降，仍需进一步提升。

---

## 476. Beyond Peak TOPS/W: A System-Level Perspective on Hybrid Digital, Analogue and Neuromorphic Computing

**arXiv ID:** 2608.03514 | [PDF](https://arxiv.org/pdf/2608.03514v1)

**作者:** Eiman Kanjo `[一作]`, Varuna De Silva `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文从系统级角度探讨混合数字-模拟计算，阐述其在AI边缘设备中的能效优势、架构设计、工作负载匹配及评估方法，并对现有模拟、光学、神经形态硬件与数字平台的协同工作进行综合分析。

**💡 创新点**

创新点包括：①提出基于部署级指标而非单纯TOPS/W的混合计算评估框架；②构建系统层级的混合设计模式与功能层次；③给出系统级基准包ℬ（E_task、L_task、A_task等），强调在完整工作负载边界内比较；④提出能效归因到传感、计算、存储、控制和校准等六大系统层面；⑤提出针对物理不确定性与校准的动态补偿策略。

**🔧 技术方法**

主要使用的技术：①模拟内存计算（AIMC）与光学线性变换（photonic）以及神经形态事件驱动处理；②混合信号接口设计与数字调度/校准模块；③硬件感知训练、量化感知训练与噪声注入；④基于中间表示（e.g., Neuromorphic IR）的软件编译与部署框架；⑤系统级能耗与延迟归因模型。

**📊 数据集**

文中未进行实验，未引用具体数据集，仅以现有芯片（PCM AIMC、光子加速器、Neuromorphic 处理器等）为案例说明。

**📈 对比分析**

比较方法是构建多维度基准包ℬ，涵盖任务能耗、端到端延迟、准确度、数据迁移量、校准成本、场景波动、可编程性与部署成本。论文指出单纯的TOPS/W或芯片级指标难以直接比较，建议与优化后的数字基线在匹配任务精度与运行条件下进行系统级对比；对已发布的芯片数据进行了边界说明，但未给出统一性能数值。

**⚠️ 局限性**

局限性包括：①物理硬件的不确定性与漂移导致校准开销大；②混合信号接口的能耗与延迟显著；③目前缺乏统一的部署级评估标准与工具；④适用工作负载受限（需大量矩阵运算或稀疏事件驱动）；⑤硬件可编程性与迁移性差，软件生态不完善；⑥在大规模系统集成、热管理、能耗分配等方面仍面临挑战。

---

## 477. ConlangBench: Exploring Language Knowledge and Learning in LLMs through Diverse Constructed Languages

**arXiv ID:** 2608.03505 | [PDF](https://arxiv.org/pdf/2608.03505v1)

**作者:** Jinhong Jeong `[一作]` (Yonsei University), Youngjae Yu `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了 ConlangBench，首个包含 21 种构造语言与 2100+ 英语平行句对以及 32 万词表的大规模基准。

**💡 创新点**

创新点在于提供多种构造语言的平行语料和词表，系统评估 LLM 在不同构造语言上的翻译与学习能力，并揭示其与语言设计属性的关联。

**🔧 技术方法**

采用 LLM 翻译评估（chrF++）、预训练曝光量估算（Infini-gram）、词表注入实验以及对齐度量分析等技术。

**📊 数据集**

使用来自 OPUS、Tatoeba、HuggingFace 等公开数据以及 30+ 构造语言社区手工收集的文本，构建 2100+ 句对、430k 句对（非爱尔语）和 321k 词条。

**📈 对比分析**

通过对 12 大 LLM（10 开源+2 专有）的双向翻译实验、暴露量相关性分析和词表注入，发现后备词汇来源构造语言（a posteriori）翻译更好，学习曲线随语言类别差异显著，整体性能仍受限。

**⚠️ 局限性**

局限包括语料不平衡（爱尔语占绝大多数）、评估指标仅为表面层面的 chrF++、对构造语言分类的主观性以及仅以英语为目标语言。

---

## 478. Lightweight 3D Object Detection via Mamba-Based Knowledge Distillation

**arXiv ID:** 2608.03490 | [PDF](https://arxiv.org/pdf/2608.03490v1)

**作者:** Quoc Cuong Ninh `[一作]` (Viettel Group), Dinh Hoan Trinh `[通讯]` (Viettel Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了轻量化3D目标检测框架，通过Mamba知识蒸馏实现教师模型向学生模型的知识迁移。

**💡 创新点**

创新点在于多分支Mamba骨干网络与基于目标框的稀疏体素空间自适应蒸馏。

**🔧 技术方法**

使用Mamba序列模型、体素化、知识蒸馏、投影模块等技术。

**📊 数据集**

在公开的nuScenes数据集和自研的Livox-Legged数据集上进行实验。

**📈 对比分析**

与CenterPoint、DSVT、LION等先进方法对比，学生模型在保持近似准确率的同时将推理时延降低约一半，mAP提升至68%以上。

**⚠️ 局限性**

局限性是蒸馏过程需依赖标注框，且教师模型本身复杂，训练成本高。

---

## 479. Beyond the Gegenbauer Paradigm: q-Orthogonal Kernels for Machine Learning

**arXiv ID:** 2608.03482 | [PDF](https://arxiv.org/pdf/2608.03482v1)

**作者:** Álvaro Sánchez-Paniagua Ríos `[一作]` (University of Alcalá), Edmundo J. Huertas `[通讯]` (University of Alcalá)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于离散q-Hermite I多项式的新型支持向量机核函数；

**💡 创新点**

创新点在于利用q-正交多项式的本征有界性，构建了无需额外缩放的核函数，并证明其满足Mercer定理；

**🔧 技术方法**

采用了q-正交多项式理论、权重函数构造、三项递推计算和贝叶斯优化进行超参数搜索；

**📊 数据集**

在20个UCI与LIBSVM公开数据集（共涵盖二分类与多分类）上进行实验；

**📈 对比分析**

通过与线性、多项式、RBF、Hermite、Gegenbauer核的对比，q-Hermite核在准确率与F1得分上位列第三，训练时间比Gegenbauer快约30%，支持向量比例略高；

**⚠️ 局限性**

局限性包括：实验集有限，未探究不同类/维度对核性能的细粒度影响；缺少对超参数敏感度分析；在某些数据集上与传统核无显著优势。

---

## 480. Efficient Multilingual Neural Machine Translation via Corpus-Driven Vocabulary Pruning: An English-Arabic Case Study

**arXiv ID:** 2608.03480 | [PDF](https://arxiv.org/pdf/2608.03480v1)

**作者:** Ahmed Amine Aliane `[一作]` (Arabic Institute for Translation), Hassina Aliane `[通讯]` (CERIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段框架，先对多语言预训练模型进行语料驱动的词表裁剪，再针对特定语言对进行有针对性的微调；

**💡 创新点**

创新点在于通过直接删减词表大小（≈60%）而不改动模型结构，实现显著内存节省，同时保留跨语言知识；

**🔧 技术方法**

使用词表裁剪（按语料统计保留出现的子词）、embedding层切片、Tokenizer映射包装以及精细化的超参微调；

**📊 数据集**

主要数据集为英语‑阿拉伯语：MultiUN（正式外交文本）与OPUS‑100（开放域混合文本）进行训练，评测于MultiUN测试集及FLORES‑200；

**📈 对比分析**

与原始多语言模型、裁剪后未微调模型、OPUS‑MT‑en‑ar双语基线以及NLLB‑200蒸馏模型比较；裁剪+微调后M2M100在BLEU 42.04、COMET 0.8730的基础上接近或超过专门双语模型，同时内存减少约60%；

**⚠️ 局限性**

局限包括对少数语言对的验证不足、对罕见实体/引用的OOV处理不完善、以及在专门化与跨域泛化之间的权衡需进一步调优。

---

## 481. A Challenge-Nonce Freshness Gap in Project Veraison's TPM Reference Schemes, Found by Appraising Application-Layer Action Evidence End-to-End

**arXiv ID:** 2608.03534 | [PDF](https://arxiv.org/pdf/2608.03534v1)

**作者:** Anton Sokolov `[一作]` `[通讯]` (Tyche Institute), Anton Sokolov (Tyche Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对应用层动作证据包（AEP）进行端到端集成，完整地将其交给兼容的 Project Veraison RATS Verifier 进行评估，并通过测量 PCR 与 TPM Quote 证明输出绑定；

**💡 创新点**

①首次将 AEP 通过真实 Veraison Verifier 输出签名 EAR，验证了前期仅基于模拟评估的结果；②发现并修复了已部署 Veraison 参考方案中缺失挑战随机数新鲜度检查的安全缺口，并提供两步可上游的修复方法；③公开可复现的实验脚本和工件，促进社区复现与评估；

**🔧 技术方法**

使用 TPM 2.0 软件模拟器、EC P‑256 ECDSA attestation key、PCR 测量、TPM Quote、CoRIM 可信根与参考值、EAT/AR4SI/EAR 结果、Rego 策略引擎、Project Veraison Docker 镜像等技术栈；

**📊 数据集**

单一演示 AEP 实例（包含授权、输入、工具调用与结果）以及对应的黄金 PCR 值与 CoRIM 参考值；未使用公开大规模数据集；

**📈 对比分析**

通过对三种情形（正常、结果替换、签名篡改）进行端到端验证，比较 Veraison 返回 EAR 的 trustworthiness 级别，确认输出绑定有效并揭示新鲜度缺口；实验在单机 Docker 环境完成，耗时短，无显著性能瓶颈；

**⚠️ 局限性**

仅使用软件 TPM 模拟，未验证硬件 TPM；仅评估单一 Veraison 实例；仅验证平台轴，未覆盖授权轴；新鲜度修复仍需上游合并；未验证多 Verifier 或 EAT 子模块组合。

---

## 482. Consensus Measures for Unstructured Biomedical Text Annotations

**arXiv ID:** 2608.03529 | [PDF](https://arxiv.org/pdf/2608.03529v1)

**作者:** Pascal Wullschleger `[一作]` (Maynooth University), Jennifer Foster `[通讯]` (Maynooth University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了医学文本开放式标注中的软互评可靠性（soft IRR）并系统评估了多种语义相等度量。

**💡 创新点**

首次在自定义文本标注任务中引入软IRR概念，并对嵌入、编辑距离、NLI、LLM等度量进行比较，提出NLI作为折衷方案。

**🔧 技术方法**

使用句子嵌入、自然语言推理（NLI）模型、LLM-as-a-judge、Hungarian分配算法以及软Cohen's κ/Fleiss' κ计算方法。

**📊 数据集**

在ICD-11、MeSH、MedDRA三大医学术语库上进行合成实验，并在Derm1M和REFLACX真实生物医学数据集上验证。

**📈 对比分析**

通过将软IRR估计的Cohen's κ与精确匹配或Jaccard真值的MAE比较，发现NLI和LLM方法MAE≤0.065，优于编辑距离和嵌入等传统度量。

**⚠️ 局限性**

研究仅限于医学领域，实验集未覆盖所有语义误差类型，且LLM方法受规模限制，难以在大规模数据上估算偶然一致性。

---

## 483. Training Documents Reranker with Search Rubrics for Deep Research Agent

**arXiv ID:** 2608.03527 | [PDF](https://arxiv.org/pdf/2608.03527v1)

**作者:** Wenhan Liu `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于查询特定评估 rubrics 的文档 reranker —— RubricRanker，能够从检索候选中选择满足多维信息需求（相关性、多样性、简洁性、可信度等）的文档集合。

**💡 创新点**

创新点在于将搜索 rubrics 设计为层级化、查询特定的评估框架，并利用两阶段训练（rubrics‑guided SFT + rubric‑based RL）让模型在无 rubrics 输入时直接生成高质量文档子集。

**🔧 技术方法**

使用 GPT‑5.1 生成参考答案和 rubrics，利用 LLM 判断子集评分；训练采用 GRPO 强化学习与层级奖励聚合；基座模型为 Qwen3‑8B（或 8B 参数模型）作为 reranker。

**📊 数据集**

使用深度研究任务集（HealthBench、WebWalkerQA、DeepResearchBench、ResearchQA）以及 RAG 任务集（HotpotQA、Bamboogle、NQ、TriviaQA、PopQA）进行评测；训练时采集子查询和用户问题，并使用 Serper Google Search API 或 BGE‑retriever。

**📈 对比分析**

与多种 vanilla（BGE‑Reranker‑Large、MonoT5、RankT5 等）和 generation‑oriented（SetR、Rank4Gen）rerankers 对比，RubricRanker 在深度研究平均得分 60.1、RAG 平均 EM 40.0 分别超出第二名 2.6 分和约 2 分，且显著减少搜索调用次数。

**⚠️ 局限性**

局限性：依赖 GPT‑5.1 产生 rubrics 与奖励，成本高；评测仅在样本查询上，未覆盖完整基准；缺乏直接评估文档集合质量的客观指标。

---

## 484. When Many Answers Are Valid, Voting Fails: Symbolic Verification for Best-of-K Causal Reasoning in LLMs

**arXiv ID:** 2608.03506 | [PDF](https://arxiv.org/pdf/2608.03506v1)

**作者:** Omatharv Bharat Vaidya `[一作]` (University of Texas at Austin), Nhat Ho `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于可执行因果验证的最佳-多样本选择方法CALVER，能够在答案多样的推理任务中通过验证候选的图论合法性来选取最可信的答案。

**💡 创新点**

创新点在于将答案聚合从字符串频率转向对候选有效性类的可执行判定，利用因果图的d-分离、后门调整、干预等判据实现无监督、无目标标签的候选验证。

**🔧 技术方法**

使用了符号验证器、六槽式结构化推理模板、可执行的因果图计算（d-separation、m-separation、干预图、后门检验）以及对数值ATE的重算检查，并在LLM生成的候选上进行即时评分。

**📊 数据集**

在CLEAR的find-one-valid任务、10个贝叶斯网络、CausalGraph2LLM多种图形编码以及K&K逻辑谜题等多种公开基准上进行实验。

**📈 对比分析**

与多种基线（多数表决、set-medoid、奖励模型、LLM评判器、模型置信度）对比，CALVER在所有评测集上平均提升约15–25个百分点，最大单项提升达42%，且随着K增大性能持续提升。

**⚠️ 局限性**

局限性在于当答案近似唯一或文本到图的提取准确时，单图提取并精确求解更为有效；CALVER的优势主要体现在答案多样、单图提取不可靠的中间场景中。

---

## 485. Rethinking Modality Reliability in Multimodal Sentiment Analysis with Incomplete Observations

**arXiv ID:** 2608.03611 | [PDF](https://arxiv.org/pdf/2608.03611v1)

**作者:** Chunlei Meng `[一作]` (Fudan University), Zhongxue Gan `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Modality Reliability-Calibrated Framework (MRCF)，在不完整多模态情感分析中显式估计每个样本的模态可靠性，并通过可靠性感知分支、可靠性引导交互和可靠性校准融合实现更稳健的情感预测。

**💡 创新点**

创新点包括：① 通过内部质量和跨模态一致性联合估计样本特定模态可靠性；② 在交互阶段利用可靠性调节跨模态信息流，抑制可靠性传播偏差；③ 在融合阶段将可靠性融入权重并与完整观测锚点对齐，提升不完整输入下的稳定性。

**🔧 技术方法**

技术手段涵盖：BERT 语言编码器；音频、视觉特征提取器；投影网络与 Sigmoid 估计可靠性；可靠性引导注意力交互；可靠性加权融合；代理可靠性监督（完整观测对齐）；滑动 L1 与排序损失；完整观测校准损失。

**📊 数据集**

使用三大多模态情感基准：CMU-MOSI、CMU-MOSEI 和 CH-SIMS。

**📈 对比分析**

与多种基线（MISA、S-MM、MMIM、CENET、TFR-Net、ALMT、LNLN、P-RMF、DAR、MIG-HCL、ROSA 等）在随机时间缺失和固定模态缺失协议下进行全面对比。MRCF 在所有数据集与缺失设置下均达到或逼近最优性能，例如 MOSI F1 79.71% / Acc-2 75.16%，MOSEI F1 80.27% / Acc-2 79.38%，并展示更高的稳定性与可靠性指标。

**⚠️ 局限性**

局限性：① 可靠性估计依赖完整观测作为代理，实际无完整观测时需近似；② 对大规模或实时场景的计算效率与资源占用尚未深入评估；③ 在极端缺失（仅剩单一模态）或强噪声情况下的鲁棒性仍需进一步研究。

---

## 486. Language-Specialized Multi-Teacher On-Policy Distillation for Multilingual LLM-Based ASR

**arXiv ID:** 2608.03610 | [PDF](https://arxiv.org/pdf/2608.03610v1)

**作者:** Yuan Xie `[一作]` (NIO), Jie Wu `[通讯]` (NIO)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多教师自上政策蒸馏框架LS-MOPD，用以提升多语言LLM语音识别性能。

**💡 创新点**

创新点在于将语言专用教师与多教师蒸馏结合，并探索静态与动态声学前缀配置以缓解跨语言优化冲突。

**🔧 技术方法**

使用的技术包括基于DAPO的强化学习教师训练、自上政策蒸馏、语言路由和多教师加权聚合。

**📊 数据集**

实验数据集包括WenetSpeech、KeSpeech、WenetSpeech-Yue和LibriSpeech，共约5万句多语言语料。

**📈 对比分析**

与多种开源大模型对比，LS-MOPD在离线和流式任务上均实现显著CER/WER下降，甚至在多数基准上超越最佳教师的性能。

**⚠️ 局限性**

局限性包括仅在单一模型骨干和有限语言上验证，动态声学前缀的效果尚未充分挖掘，且缺乏跨架构的广泛验证。

---

## 487. From Bug Reports to Browser-Executable Procedures: An LLM-Driven Agent for Web GUI Bug Reproduction

**arXiv ID:** 2608.03598 | [PDF](https://arxiv.org/pdf/2608.03598v1)

**作者:** Cunming Zhang `[一作]` (University of Luxembourg), Michail Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于大型语言模型的浏览器代理，能够从自然语言 bug 报告中自动构建、执行并验证 Web GUI 的重现步骤；

**💡 创新点**

核心创新在于将缺失执行上下文的恢复、状态感知的浏览器交互以及基于报告描述的最终状态判定三大能力有机结合，实现从文本到可执行浏览器脚本的闭环；

**🔧 技术方法**

技术手段包括 LLM 驱动的上下文构建与计划生成、基于浏览器状态的动态动作生成、屏幕截图与 DOM 快照的实时记录以及 LLM 判定器进行最终结果验证；

**📊 数据集**

使用了 667 条来自 Ghost、Metabase、NocoDB 和 n8n 四个开源 Web 应用的真实 GitHub bug 报告，并在控制环境下对其进行标注和评测；

**📈 对比分析**

与直接脚本生成和无上下文准备的 Browser‑use 基线相比，系统平均重现成功率提升至 49.96%，任务完成率约 75%，动作成功率 86.5%，且在历史 Bug 版本回放中触发原始错误的比例超过 97%；

**⚠️ 局限性**

主要局限在于 UI 元素定位的动态性导致 60–74% 的失败，缺失的前置资源与环境依赖仍需进一步完善，且评测基于当前版本，无法覆盖所有历史 Bug 情况。

---

## 488. FOUND-AF: Benchmarking ECG Foundation Models for Atrial Fibrillation Detection

**arXiv ID:** 2608.03597 | [PDF](https://arxiv.org/pdf/2608.03597v1)

**作者:** Amirhossein Taleshinosrati `[一作]` (University of Southern Denmark), Abdolrahman Peimankar `[通讯]` (University of Southern Denmark)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `109c2b71-d051-425c-831f-0c544c24280d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FOUND-AF统一框架，对九个公开 ECG 基础模型在四个异质数据集上进行无泄漏、冻结编码器的 AF 检测基准评估

**💡 创新点**

首次将不同预训练目标、架构与规模的模型在相同实验条件下公平比较，揭示模型大小并非决定性能的关键

**🔧 技术方法**

使用预训练的 1‑D CNN / Vision Transformer、掩码预测、对比学习等技术，采用 XGBoost 作为统一下游分类器

**📊 数据集**

AFDB、CinC2017、CPSC2021、LTAFDB 四大公开 ECG 数据集，分别覆盖长时间 Holter、短单导联、变量长度双导联等多样采集场景

**📈 对比分析**

通过 5 折录音级分组交叉验证、AUC、F1‑score、配对记录级自助法 + Holm 校正等统计检验，结果显示 ECGFounder 在所有指标上均领跑，F1‑score 最高达 99.50%（CPSC2021）

**⚠️ 局限性**

仅评估二分类 AF/正常、仅使用冻结编码器、未验证多节律/多标签任务，未来需扩展至更复杂诊断与实际部署验证

---

## 489. Permutation Decoding of AG Codes from Curves Defined by Separated Polynomials

**arXiv ID:** 2608.03592 | [PDF](https://arxiv.org/pdf/2608.03592v1)

**作者:** Alonso S. Castellanos `[一作]` (Universidade Federal de Uberlandia), Wilson Olaya-León `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了利用曲线自同构构造的一点AG码的排列解码方法，推导了信息与校验位的结构，并给出了可纠正特定突发错误的部分PD集。

**💡 创新点**

提出了SAP曲线（Separated Additive Polynomial曲线）及其特殊子类的概念，证明其自同构群能产生高效的部分PD集，扩展了 Hermitian、norm‑trace 曲线的排列解码结果。

**🔧 技术方法**

采用代数几何码理论、曲线自同构群、轨道划分以及根图（root diagram）和 Gröbner 基构造的技术来确定信息位并构造 PD 集。

**📊 数据集**

未使用具体实验数据集，研究以理论证明和符号计算为主；示例以 Hermitian 子曲线、Abdón‑Torres 曲线等为例进行符号演算。

**📈 对比分析**

与传统的最大似然解码或前向后向检验相比，排列解码在具备大自同构群的曲线上能通过有限的置换集合实现高效的错误定位与纠正，能够在理论上纠正相同第二坐标或第一坐标的突发错误；然而缺乏实验性能指标。

**⚠️ 局限性**

局限性在于仅提供部分 PD 集，需自同构群的特定结构；对多点 AG 码、非 SAP 曲线以及更一般的错误模式尚无通用解法；同时实际实现的计算复杂度与 PD 集大小相关，需要进一步优化。

---

## 490. SlimVLM: Sensitivity-aware Dynamic Structured Pruning with Adaptive Visual Token Selection for Efficient Vision-Language Models

**arXiv ID:** 2608.03580 | [PDF](https://arxiv.org/pdf/2608.03580v1)

**作者:** Yaozhi Wen `[一作]` (Huawei Technologies), Xinghao Chen `[通讯]` (Huawei Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SlimVLM，一个针对视觉语言模型的结构化剪枝框架，先通过自适应视觉令牌选择去除冗余视觉令牌，再采用敏感度感知动态剪枝比率，对注意力头和 MLP 通道进行精确剪枝。

**💡 创新点**

创新点包括：① 基于文本-视觉注意力平均分数的自适应视觉令牌筛选策略，消除视觉令牌干扰；② 采用皮尔逊相关系数衡量模块输出线性重构误差，动态调整各模块的剪枝比例，实现模块级自适应剪枝；③ 在剪枝过程中同时兼顾多头注意力和分组查询注意力的特殊结构。

**🔧 技术方法**

使用的技术主要有：Transformer 结构（多头注意力、MLP）、自注意力矩阵提取、文本-视觉交叉注意力计算、平均注意力分数阈值筛选、皮尔逊相关系数动态剪枝、最小二乘线性重构、LoRA 微调、以及多种实验评估工具（lmms-eval）。

**📊 数据集**

实验覆盖 LLaVA-1.5-7B、LLaVA-Next-7B、Qwen2.5-VL-7B 三种模型，评估数据集包括 GQA、MMBench（英中版）、MME、POPE、SQA、VQA‑v2、TextVQA、VizWiz 等八大多模态基准。

**📈 对比分析**

与 FLAP、Wanda‑sp 等基线对比，SlimVLM 在 20% 剪枝时平均性能下降仅约 6.2%，比 FLAP 和 Wanda‑sp 分别高 4.43% 与 8.49%；在 40% 剪枝时仍保持 24% 左右的性能，明显优于对手。速度方面，20% 剪枝时吞吐量提升 24.94%，远超对手的 9–11%。

**⚠️ 局限性**

局限性包括：① 仍需在高剪枝率下（>50%）进行性能恢复训练；② 对开发集的依赖较大，需足够多样化的样本；③ 目前仅在 Transformer‑based VLM 上验证，尚未测试在非 Transformer 或更大规模模型上的适用性。

---

## 491. EffiHolmes: Differential Profiling-Guided Repository Level Time Inefficiency Fix Localization

**arXiv ID:** 2608.03558 | [PDF](https://arxiv.org/pdf/2608.03558v1)

**作者:** Haowen Yang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Zishuo Ding `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EffiHolmes框架，用于定位大型软件库中时间效率问题的修复位置。

**💡 创新点**

创新点在于：①使用差分采样对比基准与扩容工作负载，精准放大效率热点；②从完整执行轨迹中提取并压缩关键路径；③通过域驱动的LLM推理桥接热点与实际修复点之间的语义鸿沟。

**🔧 技术方法**

技术包括：差分性能分析、VizTracer无聚合追踪、调用链压缩、LLM指引式推理与多级提示策略。

**📊 数据集**

采用RepoEffi-Bench基准，包含140个来自Python数据科学仓库的真实耗时问题。

**📈 对比分析**

与检索、程序化、代理和聚合剖面等多类基线对比，EffiHolmes在GPT‑5.1文件级Acc@3提升4.29pp、在qwen3‑4b函数级Acc@5提升15.00pp，整体表现优于所有对照方法，并在不同模型规模下保持鲁棒性。

**⚠️ 局限性**

局限包括：仅覆盖Python层面，无法定位本地C/Cython实现的修复点；对低成本函数的时间分辨率有限，导致某些路径被忽略；以及对完全超出执行路径范围的修复点无法检索。

---

## 492. Heterogeneous LLM Serving with General-Purpose Processing-Near-Memory for Retrieval-Based Sparse Attention

**arXiv ID:** 2608.03555 | [PDF](https://arxiv.org/pdf/2608.03555v1)

**作者:** Hyungkyu Ham `[一作]` (Pohang University of Science and Technology), Gwangsun Kim `[通讯]` (Pohang University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种异构解码阶段服务系统，将 KV 缓存与索引键迁移到 LPDDR5X 设备，GPU 只处理模型权重与 MoE，配合细粒度微批调度以最大化吞吐量。

**💡 创新点**

设计通用 PNM 加速器 KARAT，满足容量、带宽、计算与通用性四项需求，并提出 OFMS 与 CMR 两种微批调度技术，实现对检索式稀疏注意力的高效执行。

**🔧 技术方法**

利用 LPDDR5X 大容量内存、可编程 GPNM 核、CXL 高速互连、预取‑解码分离、微批级细粒度调度与长度重平衡技术实现异构计算与内存协同。

**📊 数据集**

使用真实代理工作负载追踪（SWE‑Pro、SWE‑Chat、KV‑CT）以及 LLaMA3‑70B 训练自由稀疏注意力数据集进行评估。

**📈 对比分析**

在 100 ms TBT SLO 下与 GPU‑only、主机 DRAM、CXL、SPNM、PIM 等基线对比，KARAT 在三种模型上提升 2.09–6.13× 通过率/功耗，训练自由稀疏方法提升 1.36–3.21×。

**⚠️ 局限性**

对专家路由的均匀性敏感，设备核心数与 OI 匹配有限，CXL 链路延迟与内存控制器热边缘可能影响性能，模型跨层索引重用与调度仍有改进空间。

---

## 493. IRIS: Visual-Semantic Binding for Forgery-Resistant Watermarking of Diffusion Images

**arXiv ID:** 2608.03539 | [PDF](https://arxiv.org/pdf/2608.03539v1)

**作者:** Xiaoyan Feng `[一作]` (Griffith University), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的内生成水印IRIS，绑定到图像视觉语义。

**💡 创新点**

利用生成轨迹终点的CLIP嵌入生成一次性环形水印，并在后期采样步骤以匹配能量注入，解决因果耦合导致的语义漂移问题，同时通过规范化的CLIP读取实现对语义变动的敏感性。

**🔧 技术方法**

CLIP视觉语义编码、SimHash二进制化、HMAC生成环形种子、DDIM逆向重采样、频域环形写入与匹配、规范化处理。

**📊 数据集**

Stable-Diffusion-Prompts、DiffusionDB、MS-COCO等三组各500条提示。

**📈 对比分析**

与Tree-Ring、Gaussian Shading、SEAL、StableSignature等三种内生成与两种后处理水印对比，IRIS在TPR@1%FPR 0.99、PSNR 30.2、SSIM 0.92，且平均攻击成功率（Forgery、Removal）仅0.04，显示在保真、可检测、伪造抵抗和去除抵抗上优于现有方案。

**⚠️ 局限性**

对极端数值失真（亮度、噪声）的鲁棒性有限，安全参数为可接受角度，需更强规范化或专门训练的感知编码器以提升容忍度。

---

## 494. CodeAssay: A Multi-Metric Benchmark with Audited Ground Truth for LLM Code Generation

**arXiv ID:** 2608.03535 | [PDF](https://arxiv.org/pdf/2608.03535v1)

**作者:** Shahbaz Siddeeq `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CodeAssay benchmark，提供185个基于软件工程分类的 Python 代码生成任务，并对参考答案、测试集进行独立审计与改进；

**💡 创新点**

创新点在于：① 通过分类先行构造任务，保证覆盖多种软件工程维度；② 对参考答案和测试集进行严格审核，提升基准可靠性；③ 结合隐藏/公开测试、变异测试和多项代码质量度量，超越传统仅评测功能正确性的做法；

**🔧 技术方法**

使用了 Python 的单元测试框架（pytest）、变异测试工具、静态分析器 Bandit 与 Semgrep、代码风格检查工具 Flake8，以及自定义的任务生成与评估脚本；

**📊 数据集**

数据集为 185 个 Python 任务，涵盖10个 SWEBOK 领域，每个任务含自然语言描述、签名、参考实现、公共/隐藏测试，测试用例覆盖约98% 声明覆盖率；

**📈 对比分析**

通过对 7 种专有 LLM（GPT-4o-mini、GPT-4o、GPT-5.6-sol、Claude Haiku、Claude Sonnet、Gemini 2.5 Flash）在标准和安全聚焦两种提示下进行两轮生成与修复实验，评估隐藏测试功能正确性和多种代码质量指标；结果显示标准提示下模型功能正确性差异显著，安全提示未显著提升正确性但导致代码更长、复杂度更高；

**⚠️ 局限性**

局限性包括：仅评测 Python，未覆盖开放权重模型；静态分析工具和变异测试仅检测部分缺陷；缺乏任务级别的安全测试；模型解码参数不完全统一；评估仅基于单次生成与一次修复，未考虑多次交互与重现性。

---

## 495. SEER: A Self-Grounded Evidence Interface for Controlled Spatial Relation Classification

**arXiv ID:** 2608.03631 | [PDF](https://arxiv.org/pdf/2608.03631v1)

**作者:** Feixiang Liu `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种无训练、无需外部检测器的推理时可视化证据接口SEER，用于提升冻结视觉语言模型在空间关系推理中的准确率。

**💡 创新点**

创新点在于将查询相关的主体-客体位置通过自我归一化定位并显式标记，构造角色明确的局部视图，同时提供可选的几何约束和反向一致性校正，显著提高空间关系判断的鲁棒性。

**🔧 技术方法**

技术主要包括：目标关系隐藏的自我归地（self‑grounding），S/O标记视图（Self‑Marker），阈值几何判定（Geometry），以及基于逆关系的递归一致性精炼；全部操作在推理时完成，未涉及模型训练或参数更新。

**📊 数据集**

实验使用了 GQA、Visual Genome、EmbSpatial、VSR、SpatialSense 等公开数据集，其中对 GQA-Train900 和 EmbSpatial 进行图像分离冻结测试，评估跨模型的泛化能力。

**📈 对比分析**

与传统全图输入基线相比，SEER 在 Frozen GQA-Train900 上平均提升约 +4 分，EmbSpatial 上提升约 +5 分；对比多模型、多数据集的控制实验显示，局部重焦和角色显式标记贡献最大，逆一致性补丁进一步提高 1–2 分。

**⚠️ 局限性**

局限性包括：性能高度依赖自我定位的质量，无法处理深度或三维空间关系，逆一致性仅适用于具有可逆关系的多选任务，且在二元真值判断任务中的提升有限。

---

## 496. Geospatial-Prior Guidance for 3D Semantic Scene Completion

**arXiv ID:** 2608.03618 | [PDF](https://arxiv.org/pdf/2608.03618v1)

**作者:** Meng Wang `[一作]` (Hunan University), Kenli Li `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 GeoScene，一种利用卫星图像与 OpenStreetMap（OSM）结构化信息作为软先验，联合完成车辆摄像头视角外的 3D 语义场景重建的方法。

**💡 创新点**

创新点包括：① 引入 Dual‑Priors Weighted Classifier，学习观察可靠性权重和地理可靠性权重，实现对外部先验的软性加权；② 设计 Weights‑Guided Voxel Refiner，基于这两种权重自适应融合观测特征与地理特征；③ 通过 BEV 语义投影和多任务损失提升空间一致性。

**🔧 技术方法**

核心技术有：基于卷积的视图变换、稀疏体素编码、可学习的软可靠性权重、双分支融合与全局 3D 调和、BEV 监督、深度估计、以及跨视角注意力机制。

**📊 数据集**

使用 SemanticKITTI 与 SSCBench‑KITTI‑360 两大 3D 语义完成基准数据集，并使用 Mapbox 卫星图像与 Overpass OSM 向量数据做外部先验。

**📈 对比分析**

与 CGFormer、SGFormer 等现有 SSC 方法对比，GeoScene 在两大基准上均取得最优性能（SemanticKITTI mIoU 18.76%，IoU 46.58%；SSCBench‑KITTI‑360 mIoU 21.59%，IoU 49.46%），尤其在道路、建筑等大规模静态类上提升显著。

**⚠️ 局限性**

局限性：依赖 GPS/IMU 定位精度，严重误差会逐步削弱外部先验优势；对动态物体的 OSM 先验不适用，仅在静态结构上表现最佳；外部先验的噪声、时效性仍可能影响结果。

---

## 497. Formal Verification of Agentic Systems over Operational Data

**arXiv ID:** 2608.03609 | [PDF](https://arxiv.org/pdf/2608.03609v1)

**作者:** Alejandro J. Mercado `[一作]` (Imperial College London), Alessio Lomuscio `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了对单一大型语言模型（LLM）驱动的代理系统进行预部署验证的完整框架，将其建模为Stateful Tool‑Enabled Agentic Deployment（STEAD），并在FO‑CTL（First‑Order Computation Tree Logic）上验证业务要求。

**💡 创新点**

创新点包括：①将代理系统抽象为STEAD并证明其验证问题不可判定；②引入equivariance概念，确定在有限域约束下可精确实现模型检查（PSPACE‑complete）；③设计基于canonical labeling的部署包装器，强制LLM对不可区分标识符保持同构性，并证明其计算复杂度为图同构难度。

**🔧 技术方法**

采用的技术主要有：关系数据库建模、FO‑CTL规范与模型检查、图同构与canonical labeling、符号化验证与抽象、定理证明与不等式推导。

**📊 数据集**

使用的案例包括：一个改编自τ‑bench的航班取消任务以及一个自定义的客户退款工作流（case‑management workflow），均为论文自建示例，不依赖公开数据集。

**📈 对比分析**

在有限域抽象上进行显式状态评估（55个状态、72条转移）验证FO‑CTL规范，结果证明可行且精确；相较于在无限域直接验证，抽象化大幅降低了状态空间，验证成本主要由图同构求解决定，但示例实验未给出精细的性能对比指标。

**⚠️ 局限性**

局限性包括：仅处理单一LLM代理且不支持多代理或持续记忆交互；需要手工设定有限域与同构约束，无法自动化；canonical labeling求解是图同构难度，实际部署时计算开销可能较大；对复杂业务规则和非结构化数据的适用性有限。

---

## 498. Learning Clinical-Trial Strategy: Offline Policy Training for Decision Agents

**arXiv ID:** 2608.03606 | [PDF](https://arxiv.org/pdf/2608.03606v1)

**作者:** William Bolton `[一作]` (University of Oxford), Philip Torr `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过构建临床试验策略的离线决策框架，训练模型预测肿瘤药物项目在每个六个月窗口内的试验组合。

**💡 创新点**

创新点在于把临床试验规划视为离线强化学习问题，使用奖励加权模仿学习与隐式 Q 学习等多种离线目标，并结合时间门控检索框架评估模型。

**🔧 技术方法**

采用 Qwen‑2.5‑7B‑Instruct 作为基础模型，使用 QLoRA 微调，并实现行为克隆、奖励加权行为克隆、学习奖励行为克隆、隐式 Q 学习四种离线训练目标；模型通过结构化 JSON 模式输出试验组合；检索工具提供定期更新的公共数据。

**📊 数据集**

使用整合 31.7k 条公开记录（ClinicalTrials.gov、FDA/EMA 审查、SEC filings、CMS Part D、SEER 等）构建 881 条决策样本，覆盖 45 个药物项目的 1,956 个试验动作。

**📈 对比分析**

与四个前沿 LLM 代理（GPT‑5.4、Claude Opus 4.5、Gemini 3.1 Pro、Qwen3‑235B‑Thinking‑2507）在 drug、sponsor、drug‑class、temporal 四种拆分下进行对比；在无污染的 post‑August‑2025 组，奖励加权行为克隆取得 46.2% 指标 F1 与 14.2% 严格 F1，显著优于最好的工具代理（25.0% 与 2.1%）。

**⚠️ 局限性**

局限包括：奖励仅为回顾性关联，缺乏因果性；数据规模受限（特别是后期拆分仅 24 条样本）；结构化模式忽略了终点定义和可行性细节；模型在多步递归预测中对状态误差的鲁棒性仍需进一步验证。

---

## 499. Large language models for partial differential equation workflows

**arXiv ID:** 2608.03600 | [PDF](https://arxiv.org/pdf/2608.03600v1)

**作者:** Han Wan `[一作]` (Renmin University of China), Hao Sun `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了LLM在PDE工作流（发现、求解、优化）中的应用与进展

**💡 创新点**

提出三阶段工作流分类、评估维度，并强调LLM在工作流整体协同中的价值

**🔧 技术方法**

基于大型语言模型的代码生成、推理、工具调用、检索增强生成与多智能体协作

**📊 数据集**

综述多种公开基准，包括OpenFOAM教程、CFDLLMBench、PDEBench、PINNacle、ShapeBench等

**📈 对比分析**

通过可执行率、数值准确性、物理合理性、专家负担等多维度指标评估系统，指出现有方法在标准测试中表现良好但跨域通用性不足

**⚠️ 局限性**

受限于专家预定义搜索空间、缺乏高质量数据集、工作流长周期的可追溯性与科学判断不足，导致实际工程迁移性和物理可靠性有待提升

---

## 500. GenOS: Compositional Certificates for Semantic Robustness in AI Code Generation

**arXiv ID:** 2608.03588 | [PDF](https://arxiv.org/pdf/2608.03588v1)

**作者:** Corrado Priami `[一作]` `[通讯]` (Università di Pisa), Corrado Priami (Università di Pisa)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种观测相对兼容性的概率操作语义（Generative Operational Semantics，GOS），为 AI 编码代理的生成–验证–修复–提交工作流提供分层的概率核和等价关系，并通过插入排序案例和随机实验验证其理论定理。

**💡 创新点**

核心创新点：①定义观测相对兼容性并证明兼容性层的商化与组合可交换；②给出工作流的概率模仿（workflow bisimulation）定理、总变差非扩张与近似容忍度累计预算；③构造可检验的证书对象，将本地兼容性测试与端到端误差预算进行组合，实现可验证的替换安全。

**🔧 技术方法**

使用技术：概率迁移系统（Markov kernel）、概率等价和商化、概率模仿与双射、总变差度量、Hoeffding 统计检验、可执行审计脚本、Python 标准库的合成程序。

**📊 数据集**

主要实验数据集：121 个长度为 0–4 的整数数组（{-1,0,1}）用于插入排序的完整执行；随机化法律检查使用随机有限核进行 10,000 次配对测试；未使用公开的自然语言编码数据集，实验集中在合成程序与人工构造的提示/合同上。

**📈 对比分析**

比较方法：对等价提示、正式合同和近似提示分别计算合同分布与代码类分布的总变差；检验验证器的拒绝/确认率；实验结果显示等价提示产生相同的提交概率，近似提示的误差满足理论上限；随机化测试未发现违例，验证了理论与实现的一致性。

**⚠️ 局限性**

局限性：①仅适用于有限分布和可观测等价，需手工定义观测边界；②假设模型和工具在每次调用中保持 Markov 性，需冻结版本与状态；③不处理非终止或连续空间的情况；④实验成本较高，需大量采样和手工标注；⑤证书有效性受观测粗细限制，若观测不足会导致安全性失效。

---

## 501. Human Centric Embodied Intelligence for Soft Wearable Robotics

**arXiv ID:** 2608.03556 | [PDF](https://arxiv.org/pdf/2608.03556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 502. AI-Assisted Peer Review Across Research Communities: From Reviewer AI Policies to LLM Review Quality

**arXiv ID:** 2608.03581 | [PDF](https://arxiv.org/pdf/2608.03581v1)

**作者:** Alexander M. Fichtl `[一作]` (Technical University of Munich), Georg Groh `[通讯]` (Technical University of Munich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对111个 AI/NLP 会议与医学期刊的审稿 AI 政策进行调查，并构建包含 ICLR 2026 与 Nature Communications 原始稿件及人类/AI 评审的公开数据集，随后使用多模型（Llama‑4、Qwen‑3、GPT‑5）生成评审并通过多维度指标进行评估。

**💡 创新点**

首次系统性比较跨学科审稿 AI 政策差异，提出针对 AI 评审的多维度评估框架（分数一致性、关切覆盖、Granuscore、LLM‑as‑a‑Judge），揭示 LLM 生成评审虽细致但存在过度正面、证据不足等局限。

**🔧 技术方法**

利用大语言模型（LLM）进行评审生成，采用逐字段或一次性叙述提示，评估时使用分数对齐、关切重叠、Granuscore 细粒度评分以及 LLM‑as‑a‑Judge 的七分制整体评分。

**📊 数据集**

数据集包括：ICLR 2026 的 50 篇原始提交（含 3.8 评审/篇）与 Nature Communications 的 31 篇接受稿件的初评审报告（含 3.1 评审/篇）。

**📈 对比分析**

比较方法通过多维度量化 AI 评审与人类评审的相似度与差异；结果显示 GPT‑5 生成的评审在分数一致性（r≈0.62）与关切覆盖率上优于 Llama‑4/Qwen‑3，但整体评分仍高于人类，且在正面偏差与证据引用方面表现突出；Granuscore 上 Llama/Qwen 细粒度更好，GPT 细节性不足。

**⚠️ 局限性**

局限性包括：评估指标仅为代理，缺乏人类专家真实评价；模型与提示配置对结果敏感；仅覆盖 ICLR 与 Nature Communications，缺乏跨领域普适性；输入受限于原始稿件，未引入外部检索或最新研究；以及 LLM‑as‑a‑Judge 与评审生成使用同一模型可能引入偏差。

---

## 503. Robust General Utility for Reinforcement Learning

**arXiv ID:** 2608.03562 | [PDF](https://arxiv.org/pdf/2608.03562v1)

**作者:** Zixuan Liu `[一作]` (Tulane University), Zizhan Zheng `[通讯]` (Tulane University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出鲁棒通用效用强化学习框架，解决在部署时效用函数失配导致的鲁棒性问题

**💡 创新点**

将鲁棒效用框架统一化并覆盖奖励鲁棒、约束RL等，同时针对非凹效用设计了prox‑extragradient算法并给出收敛证明

**🔧 技术方法**

使用随机投影梯度下降‑上升（PGDA）和prox‑extragradient迭代，配合 Monte‑Carlo 采样估计梯度

**📊 数据集**

在LLM安全对齐任务中使用 Pythia‑70m 与 PKU‑SafeRLHF‑10k 数据集；在探索最大化任务中使用合成离散 MDP 进行验证

**📈 对比分析**

与传统 PGDA 及基线方法对比，LLM 任务中目标函数稳步下降、梯度映射逐渐收敛，收敛速度符合理论预测；探索任务亦表现出与理论一致的收敛曲线

**⚠️ 局限性**

需要先验设定效用不确定集且依赖 Minty VI 条件；非凹效用下收敛速率慢，未在大规模连续控制或高维状态空间中进行实证验证

---

## 504. Test-Time Augmentation for Tabular-to-Image Classifiers under Distribution Shifts

**arXiv ID:** 2608.03557 | [PDF](https://arxiv.org/pdf/2608.03557v1)

**作者:** Malena Loza `[一作]` (Universidad San Francisco de Quito), David Chushig-Muzo `[通讯]` (Rey Juan Carlos University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文评估了在分布偏移下，基于表格到图像编码方法的测试时增强（TTA）对分类器鲁棒性的影响。

**💡 创新点**

首次系统地比较了多种 TTA 方案与六种表格到图像编码器在 OOD 情况下的性能差异，并发现组合与混合增强在鲁棒性上表现最佳。

**🔧 技术方法**

采用了 EfficientNet‑B0 图像分类器、20 种 TTA 变换（几何、光度、结构、频域、Mixup、组合）以及六种表格到图像编码方法（BIE、DeepInsight、DistanceMatrix、Fotomics、IGTD、TINTO）。

**📊 数据集**

使用 TableShift 基准中的 HELOC（金融违约预测）和 Voting（投票行为预测）两个数据集。

**📈 对比分析**

通过对比 ID 与 OOD 子集的 AUC，发现几何与光度增强保持较高 AUC，频域与结构增强导致显著下降；组合与 Mixup 方案在 OOD 上平均提升约 5–10% AUC，并降低方差。

**⚠️ 局限性**

研究仅覆盖两个数据集且仅使用单一 EfficientNet‑B0，缺乏跨数据集的泛化与联合训练的探索。

---

## 505. Soft Guidance Starts to Outperform CoT Prompting as LLMs Improve

**arXiv ID:** 2608.03550 | [PDF](https://arxiv.org/pdf/2608.03550v1)

**作者:** Denys Pushkin `[一作]` (École Polytechnique Fédérale de Lausanne), Emmanuel Abbé `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了链式思维（CoT）提示在现代语言模型上的有效性，比较了零样本与传统少量样本CoT提示在数学问题求解中的表现。

**💡 创新点**

发现随着模型推理能力提升，传统少量样本CoT提示会产生干扰，导致性能下降；而使用零样本或模型生成示例的CoT提示相对更有效，揭示了“指导-干扰”权衡。

**🔧 技术方法**

使用多种提示策略（DS-CoT、DS-CoT+Instr、Self-CoT、Self-CoT+0shot）、答案提取方法（Rule-based、Prompt-based）以及贪婪解码的生成技术。

**📊 数据集**

使用 GSM8K 词汇数学问题数据集，对 Mathstral-7B、Qwen2.5-7B-Instruct 和 Llama-3.1-8B-Instruct 三个中等规模模型进行评估。

**📈 对比分析**

与传统少量样本CoT（最多 16 个示例）相比，零样本自由生成提示在 Mathstral 上从 71–74% 提升至 83.8%，在 Qwen 上从 83–84% 提升至 88.6%，在 Llama 上从 78–79% 提升至 81.4%；Self-CoT+0shot 在 Qwen 与 Llama 上可进一步提升，Mathstral 则仅略逊于零样本自由生成。

**⚠️ 局限性**

局限性包括仅测试 7–8B 参数的中等规模模型、随机挑选示例、仅覆盖 GSM8K 这一数学基准，缺乏更复杂的示例选择策略和跨领域推理验证。

---

## 506. Hi-TTRL: Regulating Consensus with Hints for Test-Time Reinforcement Learning

**arXiv ID:** 2608.03545 | [PDF](https://arxiv.org/pdf/2608.03545v1)

**作者:** Kunbin Xu `[一作]` (Harbin Institute of Technology), Kehai Chen `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在测试时进行强化学习（TTRL）的基础上，引入了 Hi‑TTRL，利用自适应提示（MCMC 采样的功率变换前缀）在采样阶段动态调节推理路径的共识强度，从而改善伪标签质量和优势信号。

**💡 创新点**

核心创新在于：① 在采样阶段就调节共识强度，而不是仅在奖励分配后修正；② 通过功率指数（α）控制提示的收敛或扩散，形成低共识下的收敛提示和高共识下的探索提示；③ 采用分块 MCMC 近似，仅在前缀层面采样，避免长序列推理开销。

**🔧 技术方法**

技术手段包括：测试时强化学习（TTRL）+ Group Relative Policy Optimization (GRPO)；功率变换提示分布 q_α(h|x)；分块 MCMC 提示采样；自适应阈值 [τ_low, τ_high] 触发提示；在多模型（Qwen2.5‑Math‑1.5B、Qwen3‑1.7B‑Base、Qwen3‑4B‑Base）上实施。

**📊 数据集**

实验使用多数学推理基准：AMC2023、AIME2024、GAOKAO2023‑en、MATH‑500、MINERVA，涵盖不同来源与难度的数学问题。

**📈 对比分析**

与标准 TTRL 和 Intuitor 进行对比，评估指标为答案准确率和召回率。Hi‑TTRL 在所有三种 backbone 上均优于 TTRL，最大提升约 9.87 个百分点（Qwen2.5‑Math‑1.5B），在低共识场景下效果尤为显著；在高共识场景下也保持稳定改进。

**⚠️ 局限性**

局限性包括：① 需要手工设定共识阈值和功率指数，适配不同任务仍需调参；② 额外的提示采样会增加推理时间；③ 目前仅在数学推理任务上验证，尚未验证在更广泛的自然语言推理或多模态任务中的适用性。

---

## 507. Unequal Verdicts: Investigating Gender Bias in LLM-Based Fake News Detection

**arXiv ID:** 2608.03627 | [PDF](https://arxiv.org/pdf/2608.03627v1)

**作者:** Razieh Chalehchaleh `[一作]` (Institut Polytechnique de Paris), Noel Crespi `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大语言模型在假新闻检测任务中对性别提示的敏感性，并系统评估了六种主流 LLM 的性别偏差。

**💡 创新点**

创新点在于首次在真实数据上量化性别偏差，区分不稳定性与方向性偏差，并公开了带性别化职业词条的 LIAR 数据集。

**🔧 技术方法**

采用零/少样本提示、量化评估指标（翻转率、GSI、DP、EO 等）以及多模型对比和统计检验来检测性别偏差。

**📊 数据集**

使用增强版 LIAR 数据集，其中每条陈述配备中性、男性、女性三种职业表述。

**📈 对比分析**

通过多轮推理与统计检验，发现不同模型的翻转率在 5.8%–23.6% 之间，GPT‑4.1 Mini 结果最公平，Qwen‑3 14B 翻转率最低。

**⚠️ 局限性**

局限性包括仅关注性别偏差，未探讨其他属性；仅在单一事实检测任务上实验，缺乏跨语言和跨领域验证。

---

## 508. A Theory of Conditional Collapse under Low-Rank Weight-Space Ablations: I. The Single-Block Theory and Synthetic Validation

**arXiv ID:** 2608.03620 | [PDF](https://arxiv.org/pdf/2608.03620v1)

**作者:** Abdallah Khemais `[一作]` `[通讯]` (University of Sousse), Abdallah Khemais (University of Sousse)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过理论分析与实验验证，阐明了在低秩权重空间消融条件计算时，激活补丁与权重消融两种干预方法对网络行为的不同影响，给出了精确的收敛判据与补丁-消融解耦定理，并推导了非线性交互项的二阶上界。

**💡 创新点**

创新点在于①给出了“条件收敛”即消融后匹配输入对齐到单一无条件分支的必要与充分条件，并揭示其对错误极性的确定性；②提出补丁与消融效应由两种不同量（β_iδ_i 与 β_iα_i(x_B)）驱动，可在同一模型中构造补丁足够但消融不必要的情形；③针对同一残差块内的注意力头与MLP相互作用，推导出精确的交互项公式及其二阶误差界限。

**🔧 技术方法**

主要技术包括：对残差网络的抽象线性化模型构造、代数证明与不等式推导、低秩权重消融与激活补丁实验、对交互项的解析推导及数值上界估计。

**📊 数据集**

使用两组自制的合成条件回忆任务（基于键值对的记忆与逆记忆），分别在三层、四层Transformer上训练的小型网络（宽度 48~64，头数 3~4）进行实验。

**📈 对比分析**

实验通过计算消融后网络的“倒置-A率”“文字-B率”等指标，并与理论预测的收敛判据、补丁恢复率（κ）及交互项大小进行对照；结果显示在 39 个消融配置中，交互项大小与模型误差呈显著负相关（Spearman ρ≈‑0.83），但首次实验中出现的阈值分离在后续 25 个配置中不再成立。

**⚠️ 局限性**

局限性包括：①理论假设的残差线性读出在真实网络中极少满足，条件判据仅在约 10% 配置下可用；②缺乏多层跨块交互的完整解析；③仅在人工合成任务与小型Transformer上验证，尚未在自然语言模型或更大网络中检验；④交互项虽强相关但缺乏可迁移的统一阈值，难以直接用于模型解释决策。

---

## 509. A machine-readable catalogue of the Tsiolkovsky papers (fond 555, Archive of the Russian Academy of Sciences), and a way to measure how well its handwriting can be read

**arXiv ID:** 2608.03617 | [PDF](https://arxiv.org/pdf/2608.03617v1)

**作者:** Vladimir Beskorovainyi `[一作]` `[通讯]` (Moscow Institute of Physics and Technology), Vladimir Beskorovainyi (Moscow Institute of Physics and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Tsiolkovsky个人档案（Fond 555）的完整机器可读目录、文件日期、页级手写/打字分类、机器转录语料库，并提出一种利用同一文本的手稿与打字副本比较来估计手写文本识别错误的方法。

**💡 创新点**

创新点在于：①利用档案中同一文本的手稿与打字副本作为“隐式地面真值”，通过两者转录比对来间接衡量识别误差；②基于墨迹行长变化的阈值分类器取代传统行距法；③通过测量同一页两份阅读的一致性来验证无地面真值的评估方法，并据此调优模型选择。

**🔧 技术方法**

采用的技术包括：自动化批量下载与重组扫描图像（支持分块请求和重试）；图像特征阈值分类器（墨迹行长）；两种HTR模型（手写与打字）与自动对齐算法；统计分析（相似度、最长连续匹配、排名相关系数）以及对比测评（字符/词级与出版版本的精确度）。

**📊 数据集**

使用的数据集为：俄国科学院档案的Fond 555——2,019个文件、51,008张扫描图像；转录语料库包含322个文件、5,454张扫描；与出版文本对应的55对手稿/打字副本（来自2个文件），用于验证评估方法。

**📈 对比分析**

评估方法：将转录与已出版的俄语维基文本逐字符/逐词比对；在手稿/打字副本对中测量两份转录的一致性（词级占比和最长连续匹配），并与对出版文本的精确度比较。结果显示：打字转录在一文件上达98.1%字符精确度，手写转录在归一化后可达81.1%；两份转录的一致性中位于37%词级，占比与手写精确度高度相关（相关系数0.92），模型选择通过此指标可提升识别精度约8.2%。

**⚠️ 局限性**

局限性：大部分文件缺乏人工验证的地面真值；转录未手工检查；分类器准确率约80%，对未转录的90%文件仅靠图像特征；日期信息来自档案原始描述，可能含误；评估方法依赖同一文本的打字副本，无法普适于所有档案；当前语料库的识别质量不足以支持两份手稿的词级对齐或细粒度文本批判。

---

## 510. Disentangling Language Modeling and Boundaries

**arXiv ID:** 2608.03599 | [PDF](https://arxiv.org/pdf/2608.03599v1)

**作者:** Mykola Haltiuk `[一作]` `[通讯]` (AGH University of Krakow), Mykola Haltiuk (AGH University of Krakow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用统一字节接口实现语言模型间知识转移与分界点独立控制的可行性，并设计了两项实验验证语言建模与边界预测可分离。

**💡 创新点**

提出将字节接口视为共享标准，使模型间能够精确地跨词汇表转移知识，同时在保持语言能力的前提下独立修改文本分段策略。

**🔧 技术方法**

利用Bolmo架构的联合分布p(b,m)对字节与边界进行建模，进行字节级知识蒸馏与边界重学习，并对边界预测进行F1分歧度量。

**📊 数据集**

主要使用乌克兰语模型以及基于SentencePiece与BPE的不同分词器对比实验，测量其边界分歧与对数概率分布。

**📈 对比分析**

通过对比蒸馏前后模型的边界F1分歧以及保持不变的语言模型概率，验证两方向蒸馏的可行性；实验显示在保持语言模型不变的前提下能显著改变边界分布，反之亦然。

**⚠️ 局限性**

受限于UTF‑8对拉丁语系的偏好导致字节成本偏高，并未讨论在极低资源或非拉丁文字上实现字节级模型的实际效果。

---

## 511. DiagChain: A Diagnostic Benchmark for Evaluating LLM Agents on Evidence-Grounded Attack Chain Reconstruction

**arXiv ID:** 2608.03591 | [PDF](https://arxiv.org/pdf/2608.03591v1)

**作者:** Xuyang Liu `[一作]` (Tsinghua University), Xibin Zhao `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了DiagChain基准，用以评估LLM在基于证据的攻击链重构任务中的表现；

**💡 创新点**

引入了可诊断的五阶段评估指标和Evidence-Centric Retrieval-Augmented Generation (ECRAG)框架；

**🔧 技术方法**

采用检索增强生成技术，结合多源证据检索与结构化链条构建；

**📊 数据集**

使用由多操作系统、噪声水平与链长三类维度组成的MAIN‑69数据集；

**📈 对比分析**

对六种LLM在检索、分组、排序、定位和归因五个阶段进行对比，结果显示即便是最强模型仅达39.6%步骤成功，且不同规模模型错误倾向不同；

**⚠️ 局限性**

局限性包括对固定检索预算与交互轮数的依赖、缺乏对更大规模攻击链或更复杂证据噪声的泛化验证。

---

## 512. From Social Coding to Agentic Coding: Productivity and Relational Reconfiguration in Open-Source Communities

**arXiv ID:** 2608.03585 | [PDF](https://arxiv.org/pdf/2608.03585v1)

**作者:** Mengying Zhou `[一作]` (Shanghai University of Finance and Economics), Yang Chen `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用基于大型语言模型的多智能体仿真，研究生成式编码代理对开源软件社区的生产力、交互路径以及公共知识生成的影响。

**💡 创新点**

首次在社区层面量化评估生成式编码代理的采纳扩散、工作模式转移与公共知识覆盖，并揭示其带来的生产力提升与公共知识稀缺的张力。

**🔧 技术方法**

采用AgentSociety框架和大模型（DeepSeek V4/GLM‑5.1/Qwen3）实现多智能体仿真，并通过少量真实提交的few‑shot in‑context 学习进行warm‑up，结合IDT分类驱动代理行为。

**📊 数据集**

以GitHub Developer数据集为基础筛选1,084位持续活跃开发者及其仓库关系，使用其历史提交记录构建协作网络并注入真实活动做为warm‑up。

**📈 对比分析**

在No‑CA与CA两条平行仿真中对比任务产出、完成率、完成时间、CA采纳比例及公共知识覆盖与检索效率；CA条件下任务产出提高34–39%、完成时间缩短56%，但公共知识覆盖仅22%（对比81%），检索步数升至约8步，成功率下降至22%。

**⚠️ 局限性**

样本固定、观察窗口仅8周，未考虑社区规模扩展、成员进出与角色转换，限制了对长期演化与更广泛社区影响的推断。

---

## 513. Looking under the Wrong Lamppost: On the Limitations of Automated Translation Quality Estimation

**arXiv ID:** 2608.03577 | [PDF](https://arxiv.org/pdf/2608.03577v1)

**作者:** Serge Gladkoff `[一作]` (Logrus Global LLC), Lifeng Han `[通讯]` (Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对段级翻译质量估计（QE）进行系统性理论与实证评估，利用大规模工业数据检验其在翻译流程中的可行性与局限性。

**💡 创新点**

提出将排名与路由（triage）区分开来并构建阈值策略评估框架，首次用统计显著性与运营实用性并行评估 QE，并展示 QE 缺乏可安全阈值与校准的实证证据。

**🔧 技术方法**

使用 TAUS EPIC QE 模型生成得分，配合 Revix 统计分析平台计算 ROC AUC、AUPRC、Brier、精度/召回、阈值最大化（F1、F2、Youden、triage gain）等指标。

**📊 数据集**

三类 MQM 标注数据：DFKI 规则式 MT 译文、欧盟议会（EP）未编辑 MT 译文、欧盟委员会（EC）人工编辑高质量译文，共 104,762 段。

**📈 对比分析**

通过与 MQM 人类标注的二分类错误判定比较，评估 QE 在不同阈值策略下的排名与路由效果。结果显示 AUC 仅 0.59–0.63，精度低，Brier 指标表明缺乏校准；无任何阈值能在保留大部分错误的同时显著降低审校负担。

**⚠️ 局限性**

局限性包括：缺乏可靠校准与安全阈值、对域、语言对和错误率不稳、依赖训练语料导致系统性偏差、对细粒度错误检测与严重性评估能力不足，整体不足以作为独立路由或发布决策依据。

---

## 514. SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs

**arXiv ID:** 2608.03573 | [PDF](https://arxiv.org/pdf/2608.03573v1)

**作者:** Kejian Zhu `[一作]` (Chinese Academy of Sciences), Jun Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统比较了多任务学习中监督微调（SFT）与强化学习（RL）的不同表现，揭示SFT在多阶段训练中出现任务冲突而RL则能够共存，并提出了基于RL的并行训练框架 Parallel‑RL 来解耦多任务学习。

**💡 创新点**

创新点在于：①首次将SFT与RL在多任务场景下的梯度干扰和参数更新特性进行对比与理论解释；②通过优势函数与 on‑policy 机制证明 RL 的更新近似正交；③提出并验证 Parallel‑RL，通过并行训练与合并策略实现高效且几乎不失效的多任务学习。

**🔧 技术方法**

使用技术包括：对比实验（mixed‑data 与 multi‑stage）、梯度干扰量化、参数更新可视化、理论推导（优势函数、KL 限制、方差上界）、并行训练与合并（平均、SVD、TIES、适配后微调）以及 LoRA 参数微调。

**📊 数据集**

主要数据集包括：MATH500、MMLU、Knights & Knaves（逻辑推理）、LiveCodeBench（编程）以及对应的混合数据集；基模型为 DeepSeek‑R1‑Distill‑Qwen‑1.5B 与 Qwen‑7B，RL 算法采用 GRPO。

**📈 对比分析**

对比方法包括：单任务 SFT/RL、混合数据 SFT/RL、多阶段 SFT/RL、并行训练（Naive、TIES、SVD、Adapted）。实验结果显示：RL 在多阶段训练中平均提升 24.9%；Parallel‑RL 在保持或超越单任务 RL 的同时，仅需 5% 额外样本进行微调，平均提升约 10%（最高 17.4%）并几乎无任务失效。

**⚠️ 局限性**

局限性包括：①实验仅涵盖四类推理任务，未验证更大规模多任务场景；②仅使用 GRPO，未探讨其他 RL 算法的可迁移性；③对并行合并策略的超参数敏感，需进一步自动化；④尚未深入分析模型安全性与公平性。

---

## 515. CAN Disabler: Hardware-based Prevention method of Unauthorized Transmission in CAN and CAN-FD networks

**arXiv ID:** 2608.03567 | [PDF](https://arxiv.org/pdf/2608.03567v1)

**作者:** Ryo Kurachi `[一作]` (Naogya University), Satoshi Horihata `[通讯]` (AutoNetworks Technologies, Ltd.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出并实现了基于硬件的CAN控制器设备禁用器（device disabler），用于阻止 ECU 在被篡改后发送未经授权的 CAN / CAN‑FD 消息并限制异常频率传输，提升车内网络安全性。

**💡 创新点**

创新点在于：①通过硬件白名单实现对邮箱、CAN ID 以及最小传输间隔的实时校验；②仅在硬件层面拦截非法发送，避免软件级 IDS 或防火墙的复杂性；③与现有 ECU 兼容，可通过两种写入方式（诊断工具或 Secure Boot）完成白名单配置。

**🔧 技术方法**

技术实现：在 FPGA（Altera DE0‑NANO）上改造现有 CAN‑FD 控制器 IP，加入邮箱禁用寄存器、CAN ID/数据长度寄存器、最小周期计数器等；使用硬件计数器实时判断发送频率；通过可写的白名单寄存器实现功能。

**📊 数据集**

未使用公开数据集；评估使用自制测试程序在 FPGA 上模拟 ECU 重写、频率攻击等场景，并对比开启/关闭禁用器的传输结果。

**📈 对比分析**

评估方法：①测量加入禁用器后的处理延时，最大 6.25 µs（相当于 500 kbps 速率下 3–4 个位的延迟）；②验证禁止未授权邮箱、CAN ID 及频率攻击的效果，实验显示在禁用器开启时，非法消息被完全丢弃且 DoS 攻击被抑制。

**⚠️ 局限性**

限制：①仅能阻止 ECU 本身的非法发送，无法防止其他 ECU 或已被篡改 ECU 的攻击；②白名单若被篡改则失效；③实现成本相对较高，且若在所有 ECU 上部署 Secure Boot 与禁用器会显著提升硬件与启动开销。

---

## 516. Enhancing Tabular Learners with Context-Aware Semantic Embeddings

**arXiv ID:** 2608.03565 | [PDF](https://arxiv.org/pdf/2608.03565v1)

**作者:** Günther Schindler `[一作]` (SAP SE), Johannes Höhne `[通讯]` (SAP SE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

结合大语言模型的语义知识与传统表格学习器，通过在Gemma 3模型上进行表格目标插补预训练，并在推理时使用KV缓存预置代表性行来生成上下文感知的行级嵌入，随后用PCA降维并拼接到原始特征，供任何下游表格学习器使用。

**💡 创新点**

创新点在于：①引入“语义上下文预置”机制，使行嵌入在全表语义背景下生成；②使用解码器型Gemma 3进行表格任务专门微调；③通过PCA降维与多种表格学习器无缝融合；④在低样本和语义丰富数据集上实现显著性能提升。

**🔧 技术方法**

采用表格序列化+目标插补预训练、Gemma 3解码器架构、KV缓存预置、PCA降维，以及与GBDT、XGBoost、CatBoost、TabPFN、TabICL、ConTextTab、RealMLP等表格学习器的组合。

**📊 数据集**

在CARTE、TextTab、TabArena公开基准上进行评估；预训练使用大规模T4真实世界表格语料库。

**📈 对比分析**

与现有GBDT、AutoGluon、TabPFN、TabICL、ConTextTab、RealMLP等对比，在语义丰富的CARTE和TextTab上提升分类精度2.9%–6.8%，回归R²提升3.8%–19.1%；在低样本/少量数据场景下显著优于SOTA；在数值密集的TabArena上略逊。

**⚠️ 局限性**

存在的局限包括：缺乏行列置换等价性、数值处理的隐式偏差；KV缓存采样随机可能遗漏关键行；计算开销高于纯统计方法；需要表格专门预训练的Gemma 3模型；PCA线性降维可能损失细粒度语义；以及潜在的数据泄漏或LLM记忆风险。

---

## 517. Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions

**arXiv ID:** 2608.03563 | [PDF](https://arxiv.org/pdf/2608.03563v1)

**作者:** Zhenyang Feng `[一作]` (University of California), Unnat Jain `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在视觉‑语言‑动作（VLA）模型的微调阶段，将直接回归低层机器人动作改为预测一种统一的视觉‑运动目标（UVT），该目标将机器人动作与预训练的离散动态代码（来自Latent Action Model）融合。

**💡 创新点**

创新点在于：①无需改变VLA网络结构或加入额外模块，仅修改监督目标；②将动作与视觉动态信息共同编码为低维潜在空间，使监督信号与预训练VLM的语义表示更匹配；③通过精准的概率融合和可训练解码器，保持对原始动作的可恢复性。

**🔧 技术方法**

使用的技术包括：预训练的Vision‑Language Model（如CLIP等），离散动态编码器（Latent Action Model），多模态变分自编码器（MVAE）进行目标融合，训练目标包含动作重构、动态代码交叉熵和KL正则；在微调时使用平方误差对UVT对齐和解码动作的重构损失。

**📊 数据集**

数据集：LIBERO、LIBERO‑Plus（四类任务集合），以及三种现实双臂操作任务（Lift Pot、Close Marker、Plate Handover），演示数据通过VR手柄收集。

**📈 对比分析**

与传统直接动作回归（VLA‑Adapter、π₀.₅）相比，UVT在相同训练步数（10k、100k）下显著提升成功率，尤其是早期阶段：在LIBERO‑Plus Spatial任务10k步时从34.0%提升至81.5%；在真实任务中把抓取成功率从0%提升至38%，并在Plate Handover中把完成率从7%提升至40%。整体上，UVT在不同模型和环境下均取得更快收敛与更高最终性能。

**⚠️ 局限性**

局限性：仍在高精度插接类任务（Close Marker）中表现有限；方法依赖已有的LAM离散代码，若LAM质量不足会影响UVT；目前只验证在两种VLA架构上，未探究在更大规模或不同语言模型上的通用性；并未解决多步延迟或长时序依赖问题。

---

## 518. S$^3$-Diff: Structural Semantic Synergy Diffusion Model for High Fidelity Super Resolution of Pathological Images

**arXiv ID:** 2608.03540 | [PDF](https://arxiv.org/pdf/2608.03540v1)

**作者:** Jiaming Liang `[一作]` (South China University of Technology), Hongmin Cai `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结构-语义协同扩散模型 S^3-Diff，用于从低分辨率病理切片恢复高分辨率图像，保持组织结构和诊断信息。

**💡 创新点**

创新点在于引入 Specimen‑aware Structural Anchoring (SSA) 与 Structure‑guided Semantic Fidelity Tuning (SSFT)，将基于 SAM 的结构提示和 DINOv3 的语义能量融合到扩散过程，实现结构保真与语义一致的超分辨率。

**🔧 技术方法**

使用了 Latent Diffusion、ControlNet、固定的 Segment Anything Model (SAM)、预训练的 DINOv3 视觉特征网络，以及梯度和灰度引导的控制信号。

**📊 数据集**

采用 TCGA 三个癌种（LUAD、KIRC、LIHC）共 150 张切片作为训练/验证/测试，并在 46 例独立 SurGen 结直肠癌数据集上进行外部泛化评估。

**📈 对比分析**

与传统确定性方法（SwinIR、SHISRCNet）、生成式方法（ESRGAN、SinSR、STAR‑RL、SuperDiff、UPSR）以及双三线插值对比，S^3‑Diff 在 LPIPS、ST‑LPIPS、Grad‑L1 等感知指标上均优于所有基线，并在生存分析中取得 CI 0.8044，接近 HR 参考。

**⚠️ 局限性**

局限性包括仅在 4×降采样的固定条件下验证、仅测试有限的癌种与样本量、缺乏多中心、真实环境评估，且模型仍属科研工具，不能直接用于临床决策。

---

## 519. POEM: Phase-Aware $\mathrm{SO}(2)$ Feature Rotation for Time Series Forecasting Under Periodicity Drift

**arXiv ID:** 2608.03630 | [PDF](https://arxiv.org/pdf/2608.03630v1)

**作者:** Jiawen Zhu `[一作]` (Zhejiang University), Di Weng `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出POEM框架，结合局部相位估计与SO(2)旋转，对时间序列的周期漂移进行相位校正并提升预测精度。

**💡 创新点**

创新点在于通过相位校正坐标与SO(2)特征旋转减轻相位变化，再利用Directional Phase Increment Attention推断未来相位，实现连续相位轨迹的自适应建模。

**🔧 技术方法**

核心技术包括SO(2)平面旋转、局部相位校正估计（offset与velocity）、轻量化MLP预测器以及基于时间上下文的方向性相位增量注意力。

**📊 数据集**

使用九个公开数据集：ETT（四个子集）、Exchange、Weather、NN5、ZafNoo、US Birth 等，涵盖能源、金融、气象等领域。

**📈 对比分析**

与PMDformer、Phaseformer、SRSNet等八个基线比较，POEM在大部分数据集上获得MSE/MAE第一或第二名，平均提升约10‑12%，同时保持接近PMDformer的推理速度。

**⚠️ 局限性**

局限性在于目前仅针对单变量相位建模，缺乏跨变量相位协同机制，且对多周期或复杂非线性周期结构的适应性尚未充分验证。

---

## 520. Cross-Layer Interaction under Weight-Space Ablation: A Closed-Form Attention Jacobian Bound and a Test on a Real Pretrained Model

**arXiv ID:** 2608.03629 | [PDF](https://arxiv.org/pdf/2608.03629v1)

**作者:** Abdallah Khemais `[一作]` `[通讯]` (ISITCOM, University of Sousse), Abdallah Khemais (ISITCOM, University of Sousse)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了在Transformer网络中对不同层的权重空间消融所产生的交互效应，并给出了多层交互的精确分解；随后推导并验证了注意力子块的点值雅可比界限与闭式曲率常数；最后在真实预训练模型 Qwen2.5‑1.5B‑Instruct 上检验了隐含电路的 collapse、dissociation 与 interaction 现象。

**💡 创新点**

创新点包括：①给出跨层交互的闭式分解为同块项与跨层剩余项；②为剩余项提供双重积分的精确表达；③推导并验证了注意力子块的局部雅可比界限；④计算并给出闭式曲率常数；⑤在未经过专门设计的真实模型上发现并评估了隐含的间接宾语识别电路。

**🔧 技术方法**

主要技术手段为：理论推导（微积分、雅可比与曲率分析）、闭式解析、有限差分数值验证、激活补丁与权重消融实验、以及激活补丁搜索算法。

**📊 数据集**

使用的数据集为：①五个从零开始训练的小型Transformer（synthetic conditional marker‑task checkpoints），②Qwen2.5‑1.5B‑Instruct 真实语言模型（包含 GQA、IOI 等自然语言实例）。

**📈 对比分析**

实验对比方法为：将多层交互的分解结果与伴随论文单块理论进行对比；在 Qwen 上分别测量 collapse、dissociation 与 interaction 三种指标。结果显示，同块项与跨层剩余项在量级上相当，且在真实模型中三种现象呈混合分布，说明理论在实际模型上具有一定但不完全的预测能力。

**⚠️ 局限性**

局限性在于：①跨层剩余项 R× 的数值闭式界限尚未给出；②注意力雅可比界限仅在单层上验证，跨层连乘的效果仍未确认；③实验仅覆盖少数实例与单一模型，缺乏更广泛的普适性验证。

---

## 521. ConformalShift: Targeted Event Reordering Against Adaptive ECG Monitoring

**arXiv ID:** 2608.03628 | [PDF](https://arxiv.org/pdf/2608.03628v1)

**作者:** Arash Vashagh `[一作]` (University of New Brunswick), Yasmin Vashagh `[通讯]` (Farzanegan Amin 2 High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种只重新排序已验证事件的ConformalShift攻击，能在不改变ECG信号、标签或分类器输出的前提下，抑制自适应共形监测器对心室搏动的救援判定。

**💡 创新点**

创新点在于发现并利用反馈顺序对自适应阈值的影响，设计了基于位移约束的束搜索与局部改进算法，专门针对自适应共形预测的时间完整性弱点。

**🔧 技术方法**

使用自适应类条件共形预测、位移约束的束搜索+局部搜索、以及对阈值更新的解析公式。

**📊 数据集**

实验数据集包括MIT–BIH心律失常数据库和St. Petersburg INCART 12导联心律失常数据库。

**📈 对比分析**

通过与随机可行调度的基线比较，MIT–BIH上成功率分别为66.7%（Extra Trees）和60.0%（HistGradientBoosting），相较于随机调度的4.4%和12.0%有显著提升；在INCART上成功率为33.3%，仍高于随机基线。

**⚠️ 局限性**

局限性包括：仅考虑白盒攻击、固定延迟模型、有限的目标集、单一共形预测更新规则，且未针对多源时间不确定性或部分知识攻击进行评估。

---

## 522. FraQ: Efficient Coordinate-Space Recompression for Federated Low-Rank Adaptation

**arXiv ID:** 2608.03605 | [PDF](https://arxiv.org/pdf/2608.03605v1)

**作者:** Shenghui Li `[一作]` (Uppsala University), Thiemo Voigt `[通讯]` (Uppsala University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种高效的坐标空间重压缩方法，用于联邦LoRA的聚合，解决了因简单因子平均导致的聚合不匹配问题。

**💡 创新点**

创新点在于通过坐标空间重压缩，避免了全矩阵的材料化和分解，同时保持了聚合的准确性和低通信开销。

**🔧 技术方法**

使用了方向自适应的单侧坐标空间重压缩框架，结合了减少的QR分解和紧凑的Gram特征问题。

**📊 数据集**

在多个文本分类和常识推理基准上进行了实验，使用了RoBERTa-base和LLaMA-3.2-3B模型。

**📈 对比分析**

与现有的联邦LoRA方法相比，提出的方法在准确性上接近未压缩的基线，同时显著减少了下行通信和服务器端重压缩开销。

**⚠️ 局限性**

限制在于坐标空间聚合的成本随着参与客户端的总堆叠秩的增加而增加，且每轮的合并和重新初始化方案可能导致适配器收敛速度较慢。

---

## 523. Design-Time Optimization of Deep Neural Networks for Intermittent Learning on Microcontrollers

**arXiv ID:** 2608.03589 | [PDF](https://arxiv.org/pdf/2608.03589v1)

**作者:** Jakob Schubert `[一作]` (Fraunhofer Institute for Integrated Circuits IIS), Christopher Mutschler `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种针对能源自给微控制器的间歇式学习（Intermittent Computing）下的深度神经网络（DNN）设计方法，利用能量预测模型与多目标优化（MOO）在设计阶段实现对推理与训练阶段每层能耗的评估与约束；

**💡 创新点**

创新点包括：①构建可同时预测前向推理与反向训练能耗的线性回归模型；②将间歇式检查点（IC）应用于DNN的前向与反向传播，确保能耗不超过能量缓冲区；③在优化过程中将层级能耗与检查点开销纳入约束，实现无需实际部署即可评估能耗的硬件感知多目标搜索；

**🔧 技术方法**

技术手段主要有：基于ONNX的层级特征提取（计算量与内存访问）；nRF52840 MCU上使用Joulescope进行能耗采样；线性回归模型训练；Optuna NSGA-II进行多目标搜索；以及将IC机制嵌入卷积自编码器的训练与推理流程；

**📊 数据集**

使用的主要数据集为CWRU轴承振动数据集（包含正常与故障样本），用于无监督自编码器的异常检测任务；

**📈 对比分析**

通过在MOO中比较预测能耗与验证重构误差，得到 Pareto 前沿。实验结果显示，与传统不考虑间歇能耗的模型相比，最优架构的预测总能耗降低约94%，同时验证重构误差仅略增，且能在所有异常样本上实现正确分离；

**⚠️ 局限性**

局限性包括：①能量预测模型基于线性回归，虽然对间歇能耗约束足够，但在极端硬件变化或复杂层类型时可能失准；②仅在nRF52840 MCU上验证，跨平台通用性需进一步评估；③间歇式检查点的实现假设能耗与能量缓冲区大小已知，实际能量采集的波动性仍是挑战。

---

## 524. Pin Once, Swap Light: Subspace-Aligned Centroid-Residual Training for Efficient Ultra-LoRA Serving

**arXiv ID:** 2608.03579 | [PDF](https://arxiv.org/pdf/2608.03579v1)

**作者:** Xiang Li `[一作]` (Purdue University), Saurabh Bagchi `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SALT框架，将LoRA分为高容量共享域中心和低秩残差，三阶段训练实现多租户环境下高效推理。

**💡 创新点**

创新点包括：1）对齐正则化使不同任务子空间统一；2）将高容量中心固定在GPU内存，只交换极低秩残差；3）自动中心路由与动态缩放γ，兼顾PCIe/VRAM瓶颈。

**🔧 技术方法**

主要技术：LoRA、矩阵余弦对齐正则化、激活取样+OOV检测、动态γ缩放、vLLM+SGMV、硬件层优化、层级子空间对齐。

**📊 数据集**

使用数学领域（GSM8K、SVAMP、MultiArith、AQuA）和代码领域（MBPP、SPIDER、APPS、HumanEval）公开数据进行训练与评估。

**📈 对比分析**

与标准LoRA、VeRA、Compress‑then‑Serve等基线对比；在Llama‑3.2‑3B、Mistral‑7B‑v0.3、Pythia‑12B等模型上，r=1残差可恢复80‑95%高秩性能，内存压缩达16×，推理吞吐在PCIe/VRAM压力下提升最多51%/28%。

**⚠️ 局限性**

局限性：对齐机制可能抑制正交子任务（如多选题）性能；跨域共用同一中心可能导致降效；对Mixture‑of‑Experts未扩展；极低秩残差在某些任务上仍低于标准LoRA。

---

## 525. Adversarial Fast-Moving Real-World Domains as Test Beds for Benchmarking AI Scientist Capabilities

**arXiv ID:** 2608.03569 | [PDF](https://arxiv.org/pdf/2608.03569v1)

**作者:** William Bolton `[一作]` (University of Oxford), Philip Torr `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种利用时间延迟的对抗性实时域（F1 2026 车规与MTG卡组构建）来评估 AI 科学家的创新与推理能力的框架

**💡 创新点**

将实验性评估从合成任务转移到真实专家产出上，强调可观察的、公开的、具有时间滞后验证的基准

**🔧 技术方法**

使用多模型三代理管线（LLM生成、法规/卡池映射、设计/卡组生成）以及嵌入检索、LLM 判定与人工审核相结合的评估流程

**📊 数据集**

F1 2026 技术规范 PDF（264 页）和 19 组 PT 卡组、40 条 F1 真实创新列表以及 MTG Lorwyn Eclipsed 卡池与 19 组 PT 卡组列表

**📈 对比分析**

通过将生成的想法与真实专家创新做匹配，GPT‑5.2 在 F1 领域匹配 10/40（25%）创新，在 MTG 领域一轮生成 5/7（71%）新卡；其他模型表现相对较弱，整体发现率仅略高于随机基线

**⚠️ 局限性**

评估仅证明与专家产出的一致性，无法确认模型推理路径；实验基准受限于公开数据与有限模型，未覆盖完整科学发现流程，且可能存在信息泄露与人工裁决偏差

---

## 526. Compass: Degradation-Simulated Reciprocal Learning with Lightweight Needle RWKV for Multimodal Crack Segmentation under Missing Modalities

**arXiv ID:** 2608.03559 | [PDF](https://arxiv.org/pdf/2608.03559v1)

**作者:** Hui Liu `[一作]` (Tianjin University of Technology), Shengyong Chen `[通讯]` (Tianjin University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级的Compass网络，能够在任意模态缺失的情况下进行裂纹分割。

**💡 创新点**

创新点包括通过降解模拟互学习DSD、模态无关的原型特征补全FAPT、利用方向条件调制的Needle RWKV以及基于Dempster‑Shafer理论的拓扑保持融合ETPF。

**🔧 技术方法**

采用RWKV变体、方向条件调制、原型学习、Dempster‑Shafer证据融合与不确定性门控解码等技术。

**📊 数据集**

使用了CrackDepth、CrackPolar和IRTCrack三大多模裂纹数据集进行实验。

**📈 对比分析**

与多种SOTA方法相比，Compass在不同缺失率下均取得最高的F1/mIoU，并且参数、FLOPs和模型大小均显著低于对比模型。

**⚠️ 局限性**

主要局限在于仅在公开数据集上验证，缺乏对更大规模或不同工业环境的泛化评估。

---

## 527. ComFuse: Fusing Complex Memory-Intensive Subgraphs with Compute-Intensive Kernels For Modern GPU Architectures

**arXiv ID:** 2608.03537 | [PDF](https://arxiv.org/pdf/2608.03537v1)

**作者:** Di Mu `[一作]` (Nankai University), Xiaoguang Liu `[通讯]` (Nankai University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了一个自动化的GPU编译系统ComFuse，用于联合融合计算密集型和内存密集型算子，实现更高效的深度学习计算图。

**💡 创新点**

提出了Stage-Stream执行模型打破归约边界，并支持B2BGEMM多层矩阵乘法联合融合，利用多线程块集群与DSM共享内存实现跨算子协同。

**🔧 技术方法**

利用NVIDIA Hopper/Blackwell的新特性（Thread Block Cluster、DSM、Tensor Memory Accelerator）以及自适应层级归约、流水线Ping-Pong调度和自动化IR到CUTLASS模板生成。

**📊 数据集**

在合成子图（MatMul-BiasAdd-RMSNorm、LayerNorm、Softmax等）和典型B2BGEMM工作负载（Self-Attention、Target-Attention、DLRM Bottom MLP）上进行实验。

**📈 对比分析**

与TorchInductor和TensorRT对比，ComFuse在大多数子图实现1.08~1.24倍加速，B2BGEMM中相较TorchInductor最高可达1.97倍，TensorRT匹配或略优；仅在Self-Attention上被FlashAttention超越。

**⚠️ 局限性**

受限于对计算密集算子缺乏更细粒度优化、流水线负载失衡导致的性能瓶颈以及对小规模任务的低效，并未能在所有场景下取优于现有工业级优化器。

---

## 528. Policy Fragmentation or Institutional Alignment? Institutional Governance of AI in Universities and Business Schools

**arXiv ID:** 2608.03584 | [PDF](https://arxiv.org/pdf/2608.03584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 529. A Security-Oriented Lifecycle Model for Large Language Model Systems

**arXiv ID:** 2608.03626 | [PDF](https://arxiv.org/pdf/2608.03626v1)

**作者:** Eleftherios Batzolis `[一作]` (Democritus University of Thrace), Konstantinos Rantos `[通讯]` (Democritus University of Thrace)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个面向安全的LLM生命周期模型，涵盖32个阶段（四层：数据、模型、分发、应用）以及12个LLMOps子阶段和9个治理类别，并将NIST AI RMF、EU AI Act和ISO/IEC 42001等法规映射到生命周期中；

**💡 创新点**

以安全边界为划分依据，识别出13个独立安全关键阶段，构建LLMOps交叉层以及治理映射，揭示治理证据与决策的结构不对称；

**🔧 技术方法**

采用结构化文献合成与攻击面曝光三维划分原则，将现有框架和法规对齐，形成可用于安全分析的生命周期结构；

**📊 数据集**

主要基于对现有生命周期框架、威胁文献和法规文本的分析，没有使用实验数据集；

**📈 对比分析**

通过对比表格和案例映射（如Hugging Face序列化攻击、奖励模型中毒、代理攻击）展示模型在区分安全边界方面的优势，而非通过实验评估；

**⚠️ 局限性**

模型复杂度高，需分阶段逐步采纳；基于解释性阶段划分缺乏经验验证；治理映射主观性强，需进一步实证验证。

---

## 530. LoopMTP: A looped transformer guided by latent multi-token prediction

**arXiv ID:** 2608.03624 | [PDF](https://arxiv.org/pdf/2608.03624v1)

**作者:** Behzad Shomali `[一作]` (Lamarr Institute), Mehdi Ali `[通讯]` (Lamarr Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LoopMTP，一种在循环 transformer 中加入多标记预测（MTP）引导的结构，解决了循环模型的潜在过度思考和计算冗余问题。

**💡 创新点**

创新点在于：① 用余弦相似度在潜在空间中对每次循环的隐藏状态进行软 MTP 对齐；② 通过共享门控机制对不同循环的表示进行加权聚合；③ 在循环 transformer 中引入固定比例的 Loop‑LNS 以保持梯度稳定；这些组合显著提升参数效率与推理深度。

**🔧 技术方法**

技术手段包括：循环 transformer（GPT‑2 样式）、多标记预测软对齐（cosine loss）、门控加权聚合、Loop‑LNS、RMSNorm、旋转位置编码、SwiGLU 等。

**📊 数据集**

训练数据为 6.8B 个标记，来源于 FineWeb‑Edu 与 OpenWebText；评估数据集包括 FineWeb‑Edu、OpenWebText、OLMES 任务集（ARC、HellaSwag、LAMBADA 等）、GSM8K、数学套件、CodeX HumanEval 与 MBPP。

**📈 对比分析**

通过与无循环基线和现有最佳循环模型 LoopFormer 在相同参数量下的对比，使用困惑度、BPB、准确率等指标评测。LoopMTP 在大多数任务上实现平均准确率提升 8.1%（相对），对 LoopFormer 有 21.8% 的提升，且在 15 次循环时仍保持训练稳定。小型数学专家模型在 GSM8K 上达 19% 以上的准确率，远超同等参数的非循环模型。

**⚠️ 局限性**

局限性包括：仅在数学领域验证小型专家模型的效果；专家模型实验仅使用单一随机种子；循环次数增多时性能增益趋于递减或非单调；尚未完全阐明循环深度的上限和最佳规模。

---

## 531. Secure Long-Range Autonomous Valet Parking: A Reservation Scheme With Three-Factor Authentication and Key Agreement

**arXiv ID:** 2608.03590 | [PDF](https://arxiv.org/pdf/2608.03590v1)

**作者:** Di Wang `[一作]` (Wuhan University), Yuan Zhuang `[通讯]` (Wuhan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出了面向长距离无人值守停车（LAVP）的三因素身份验证与密钥协商协议SecLAVP，用于安全的预约与通信。

**💡 创新点**

结合智能卡、密码与生物特征三因素，实现密码更新、智能卡吊销，正式安全证明并抵御离线字典与中间人攻击，同时显著降低通信与计算开销。

**🔧 技术方法**

基于椭圆曲线密码学、模糊提取、哈希与双线性映射的三因素认证协议，并使用AVISPA与ROR模型进行形式化验证。

**📊 数据集**

在ONE模拟环境中使用6个停车场、15个接送点、100个车位的虚拟网络进行排队与调度实验。

**📈 对比分析**

与四种前沿协议及无安全版LAVP比较，通信开销平均下降23.9%，计算开销略高但仍可接受，调度性能几乎不受影响，能够保持低等候时间。

**⚠️ 局限性**

仍存在对智能卡物理攻击后的侧信道防护需求、对大规模部署的可扩展性未充分评估，以及对移动网络时延不确定性的鲁棒性待进一步验证。

---

## 532. Beyond Simply Environment Scaling: Designing Effective Environment Distributions for Multimodal Agent Learning

**arXiv ID:** 2608.03571 | [PDF](https://arxiv.org/pdf/2608.03571v1)

**作者:** Kejian Zhu `[一作]` (Chinese Academy of Sciences), Jun Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对多模态环境分布的多维度分析，提出了基于能力的环境选择（AES）和分层难度课程（HDC），从而显著提升多模态智能体的训练效果。

**💡 创新点**

创新点在于：①从智能体行为与能力需求出发，构建元能力剖面，用以衡量环境多样性；②结合梯度冲突和冗余度的联合评价，实现高覆盖低冲突的环境子集筛选；③针对多模态智能体的视觉状态提取与世界建模瓶颈，设计“harness”弱化与状态规模双轴的分层难度课程。

**🔧 技术方法**

主要技术包括：GPT‑5驱动的轨迹原子能力分割与元能力聚合；梯度余弦相似度用于冲突估计；加权覆盖、冗余与冲突的增益函数实现环境选择；多层难度调度算法（harness弱化→状态规模提升）实现HDC；RL训练与多模态语言模型微调。

**📊 数据集**

数据集为基于现有多模态环境工作构建的200环境池（包含文本符号化与真正多模态版本），并在此环境池上使用Qwen3‑VL‑4B/8B等模型进行训练和评估。

**📈 对比分析**

与随机抽样（Random‑K）和全量环境（All Envs）等基线对比，AES在ID/OOD环境上平均相对提升约95–144%；HDC在单轴与双轴下分别带来11.5%与18.1%的提升，组合使用（AES+HDC）在ID/OOD场景下平均相对提升达143.2%，远优于传统规模或单层课程策略。

**⚠️ 局限性**

局限性包括：①环境池主要来自已有工作，缺乏大规模自研合成环境；②实验统一在固定计算预算下进行，未验证更大规模训练的鲁棒性；③AES依赖梯度冲突的离线计算，增加额外计算开销，未来可探索更高效的冲突估计方法。

---

## 533. ReputationChain: Robust Trust Updating for Blockchain-Enabled Supply Chains

**arXiv ID:** 2608.03554 | [PDF](https://arxiv.org/pdf/2608.03554v1)

**作者:** Adnan Iftekhar `[一作]` (Wuhan University), Mir Hassan `[通讯]` (Mykolas Romeris University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于区块链的参与者信任框架，将区块链用作证据和状态管理层，离线计算信誉并应用重复对折扣、多样性惩罚、身份可信度权重及基于成交量的衰减机制；

**💡 创新点**

创新点在于：①将区块链视为证据层而非信任源；②引入重复对交互折扣与多样性惩罚，限制协同获益；③结合治理提供的身份可信度调节信誉提升幅度；④使用成交量感知的指数衰减，缓解新人因稀疏历史导致的低信任误报；

**🔧 技术方法**

使用区块链记录身份、合约、交互结果及更新记录；离线信誉服务基于事件索引模型、软化函数、指数衰减公式等进行非线性计算；仿真脚本采用Python实现；

**📊 数据集**

使用30个种子生成的合成交互数据集，覆盖协同、身份多重性、稀疏历史等三种攻击场景，交互风险在[0.60,1.00]之间随机抽样；未使用真实供应链数据；

**📈 对比分析**

通过与六种基线（B1–B6）对比，采用种子配对分析计算平均效果及95%置信区间，使用Wilcoxon检验验证方向性；实验结果显示完整模型在协同获益（0.1443 vs 0.3688）、身份多重性比率（0.8723 vs >1.08）和新人误报率（0.1683 vs 0.3633）方面显著优于基线；

**⚠️ 局限性**

局限性：仅为受控仿真，未在真实区块链系统部署；仿真使用的交互和攻击模型为合成，缺乏真实需求、争议周期等复杂性；未覆盖白洗、证书替换、治理误配等攻击；离链计算虽可审计，但未实现加密证明；参数需在实际操作中校准；该方法仅能降低信誉扭曲，无法检测或阻止所有攻击。

---

## 534. On the Geometry of Music Bandwidth Extension in Latent Spaces of Audio Codecs

**arXiv ID:** 2608.03721 | [PDF](https://arxiv.org/pdf/2608.03721v1)

**作者:** Hendrik Vincent Koops `[一作]` (Universal Music Group), Elio Quinton `[通讯]` (Universal Music Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文对音乐带宽扩展(BWE)的音频恢复进行研究，提出使用神经编码器潜在空间中的单一均值平移向量进行零参数恢复。

**💡 创新点**

创新点在于发现不同神经编码器的潜在空间存在与带宽相关的线性结构，单一向量即可实现与大型生成模型相当的恢复效果，从而揭示潜在空间结构可作为恢复的有效基线。

**🔧 技术方法**

采用潜在空间均值平移（mean‑shift）技术，评估了 Stable Audio Open VAE、CoDiCodec、dac、Encodec 等编码器，并与扩散模型、Schrödinger 桥模型等高容量生成模型进行对比。

**📊 数据集**

使用公开音乐数据集 mtd、ccmixter、maestro，生成不同截止频率（4kHz、8kHz、12kHz）的带宽限制样本。

**📈 对比分析**

与 AudioSR、CQTDiff、IBAR、A2SB 等基线模型比较，结果显示均值平移在多项指标（lsd、SiSpec、ViSQOL）上与大型模型接近或优于部分基线，且所需参数为零。

**⚠️ 局限性**

局限性包括仅在带宽扩展任务中表现突出，对噪声、削波、混响等其他降级任务效果不佳；潜在空间结构的利用仍需进一步研究以提升泛化和非线性细节恢复。

---

## 535. Amortized Interventional Forecasting for Multivariate CIR Processes

**arXiv ID:** 2608.03715 | [PDF](https://arxiv.org/pdf/2608.03715v1)

**作者:** Andreas Sauter `[一作]` (Vrije Universiteit Amsterdam), Erman Acar `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 CIR-ACTIVA 模型，用于对多变量 CIR 过程进行分摊化的因果预测，能够在不重新训练的情况下给出不同冲击场景下的后验分布。

**💡 创新点**

创新点包括：① 将时间戳作为模型的一部分，使时间成为可交换的表格输入；② 采用时间桶化解码器以捕捉不同预测时隙下的分布变化；③ 设计了一个可生成因果多变量 CIR 过程的仿真器，提供观测与干预的配对样本，用于训练与评估。

**🔧 技术方法**

使用了基于 ACTIVA 的 β-CVAE、Transformer 编码器、条件正态化流、时间桶化 GMM 解码器，以及对 CIR 过程的数值积分实现。

**📊 数据集**

使用了 661 只 5 年期 CDS 价格序列进行参数校准，并生成两种因果 CIR 数据集（无周期调节与周期调节），每个数据集包含 5000 个任务。

**📈 对比分析**

与 ACTIVA、MACE‑TNP、GMR 以及 Oracle 下限进行对比，评估指标包括 Energy、Variogram 和 CRPS。CIR‑ACTIVA 在两类数据集上均在因果选择性（非影响变量预测更好）和时间分辨率校准（短期预测更精确）两方面领先，其表现优于基线，短期误差显著降低。

**⚠️ 局限性**

主要限制在于：① 仅在合成数据上验证，真实市场的冲击持久性与极端波动被低估；② 模型规模在大规模命名组合（多资产组合）时可能因自注意力成本过高而难以扩展。

---

## 536. Shielding for Higher-Order Safety

**arXiv ID:** 2608.03662 | [PDF](https://arxiv.org/pdf/2608.03662v1)

**作者:** Filip Cano `[一作]` (Institute of Science and Technology Austria), Konstantin Kueffner `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种用于高阶安全性的保护机制，称为差分安全属性，旨在通过限制控制器的行为来确保系统的安全性。

**💡 创新点**

创新点在于扩展了传统的基于状态的保护合成方法，提出了针对离散导数的差分安全属性，并开发了相应的合成算法。

**🔧 技术方法**

使用了有限状态安全博弈的构造方法，并提出了直接合成和迭代合成两种算法来实现保护机制的合成。

**📊 数据集**

在实验中使用了一个二维离散化的汽车模型，模拟车辆在接近障碍物时的行为。

**📈 对比分析**

通过实验比较了三种合成方法的性能，结果表明直接合成和迭代合成在合成成本上显著优于基线方法，尤其是在状态被大量修剪的情况下，迭代方法表现更佳。

**⚠️ 局限性**

限制在于当前方法主要针对离散系统的导数约束，未来可以探索如何将这些约束与其他滑动窗口规范结合，以及如何在连续动态中研究离散保护的影响。

---

## 537. How Closely Do LLM Reviews Align with Human Peer Review?

**arXiv ID:** 2608.03659 | [PDF](https://arxiv.org/pdf/2608.03659v1)

**作者:** Abraham Camelo-Guerrero `[一作]` (York University), Jairo Diaz-Rodriguez `[通讯]` (York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对比 OpenAI GPT‑5.4、Google Gemini 3.1 Pro Preview 与 Anthropic Claude Opus 4.6 在 ICLR 2026 300 篇论文上的自动评审与人工评审及最终决定，分析评分分布、尺度使用与弱点主题。

**💡 创新点**

首次在同一对照实验中比较多家 LLM 在学术同行评审中的整体与细粒度一致性，并揭示评分尺度与弱点关注点的差异。

**🔧 技术方法**

采用 LLM 自动生成评审、主题匹配（抽象嵌入+余弦相似度）、UMAP 降维+K‑means 聚类分析弱点、以及 Mann‑Whitney、Wilcoxon 等统计检验比较评分。

**📊 数据集**

300 篇 ICLR 2026 提交论文（100 口头、100 海报、100 被拒）以及对应的人类评审数据。

**📈 对比分析**

通过统计检验比较 LLM 与人类评分差异，发现 LLM 能区分接受/拒绝但未区分口头/海报；Gemini 评分偏高，OpenAI/Claude 与人类更相似；弱点聚类显示 LLM 偏重基线对比，人工偏重计算效率。

**⚠️ 局限性**

仅覆盖单一会议单年的 300 篇匹配样本，缺乏跨会议、跨领域验证；LLM 训练截止前可能未见这些论文；评审尺度与决策结构对齐有限。

---

## 538. AutoSND: From Execution Evidence to Structural Policies for Automated Network Dismantling Heuristic Discovery

**arXiv ID:** 2608.03653 | [PDF](https://arxiv.org/pdf/2608.03653v1)

**作者:** Zhijing Hu `[一作]` (National University of Defense Technology), Zhiguang Cao `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出AutoSND，一个三阶段树搜索框架，用于自动生成完整的网络拆除启发式程序；

**💡 创新点**

通过执行证据驱动的结构策略归纳，将候选程序的质量、运行时和状态信息映射为可执行的结构约束，显著提升跨网络的可执行性和质量；

**🔧 技术方法**

使用大语言模型（LLM）进行代码生成、LLM-Struct结构分析、结构策略诱导，并结合多阶段树搜索与Pareto优化；

**📊 数据集**

在12个真实网络（如Grid、Yeast、Collaboration等）以及3个百万节点大网络（Facebook、YouTube、Flickr）上进行评估；

**📈 对比分析**

与18种基线方法（包括传统启发式、其他LLM驱动AHD和AI代理）对比，AutoSND-Q/S在所有12个小网络上完成率100%，平均ANC(GCC)低于其它方法，且候选运行时间仅约2–3秒；在大网络上也保持高质量和可执行性；

**⚠️ 局限性**

主要局限在于仅验证于网络拆除任务，其他图任务的泛化尚未充分测试，且对大规模图的扩展仍需进一步验证。

---

## 539. Is Inter-Seed Cross-Play Enough? Evaluating the Robustness of Zero-Shot Coordination Algorithms to Implementation Details

**arXiv ID:** 2608.03644 | [PDF](https://arxiv.org/pdf/2608.03644v1)

**作者:** Maksymilian Wolski `[一作]` (University of Cambridge), Jakob Foerster `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨实现交叉游戏（XIXP）评估框架，并利用该框架在多种实现细节下训练和评估了 Other-Play 零样本协同算法，模拟了独立实现者的实现差异。

**💡 创新点**

创新点在于：①首次系统性地引入跨实现评估，检验 ZSC 算法对实现细节的鲁棒性；②通过与传统的跨种子评估对比，验证了跨实现评估对算法可靠性的必要性。

**🔧 技术方法**

采用了多智能体强化学习中的 IPPO 作为基准算法，变更了 PPO 的超参数与实现细节（如 λ_GAE、学习率调度、梯度裁剪、价值函数裁剪、权重初始化、网络结构等），并使用交叉游戏分数（XP、XIXP、WIXP）以及置信区间进行统计分析。

**📊 数据集**

使用了 Yokai Learning Environment（一个新的多智能体协同基准，类似 Hanabi）作为实验环境。

**📈 对比分析**

通过将跨种子交叉游戏得分与跨实现交叉游戏得分进行对比，并计算 WIXP 与 XIXP 的平均值及置信区间，发现两者无显著差异，说明传统的跨种子评估已能充分反映实现细节对协同性能的影响。

**⚠️ 局限性**

局限性包括：仅在单一环境（Yokai）、单一基准算法（IPPO）和单一 ZSC 算法（Other-Play）上进行评估，缺乏对其他环境、基准算法和 ZSC 算法的泛化验证；计算成本高，限制了实验规模。

---

## 540. Empirical Analysis of Evasion and Poisoning Against Malware Data Drift Detection

**arXiv ID:** 2608.03642 | [PDF](https://arxiv.org/pdf/2608.03642v1)

**作者:** Mingyue Yang `[一作]` (University of Toronto and Vector Institute), Nicolas Papernot `[通讯]` (University of Toronto and Vector Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了针对恶意软件数据漂移检测器（CADE、Transcendent）以及恶意软件分类器的逃逸与投毒攻击，探讨攻击对两类模型的不同影响。

**💡 创新点**

创新点在于首次针对机器学习驱动的恶意软件漂移检测器进行攻击实验，揭示其嵌入空间和损失函数对攻击成功率的关键作用，并系统比较不同攻击策略与模型架构的差异。

**🔧 技术方法**

采用了对抗样本生成技术（跨模态的统一扰动），基于交叉熵和均方误差的损失函数；利用对抗训练的反向传播、RAdam优化器；并构建了对抗样本的可转移攻击框架。

**📊 数据集**

使用了 Androzoo 与 Bodmas 两个大型安卓恶意软件数据集，分别包含二进制特征与数值特征，并在训练与未来选择集上进行评估。

**📈 对比分析**

通过比较在恶意软件分类器和漂移检测器联合攻击中的总体成功率（ASR_overall）以及各自的成功率，发现当扰动增大时，CAE基检测器的成功率下降；投毒样本数量增多时，CAE模型的成功率亦下降；而在Transcendent下，跨模态扰动相对更稳健，整体ASR可达到90%以上。

**⚠️ 局限性**

局限包括：对攻击模型的转移性依赖受限；仅考虑了可观测特征的增大而非减小；未对恶意软件功能完整性做深入验证；并且实验仅覆盖了两类检测器，其他漂移检测方法未涉及。

---

## 541. MuEvo: LLM-Driven Evolution of Multi-Heuristic Ensemble

**arXiv ID:** 2608.03636 | [PDF](https://arxiv.org/pdf/2608.03636v1)

**作者:** Haoze Lv `[一作]` (Southern University of Science and Technology), Ke Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 MuEvo，一种基于大型语言模型（LLM）的多启发式自动设计框架，用于在组合优化问题中共同进化多种互相依赖的启发式。

**💡 创新点**

创新点在于：①动态组件管理（Dynamic Component Management），通过短预算探测与可逆生命周期动态重新评估组件优先级；②LLM 驱动的协同进化（LLM‑Driven Co‑Evolution），包括多集成评估、多组件信息共享、关系导向配对进化与自适应预算分配，从而在集成层面实现跨组件的协同改进。

**🔧 技术方法**

主要技术：大型语言模型（DeepSeek‑V4‑Flash、Kimi‑K2.6 等）作为交叉/变异算子；反射式进化（ReEvo）框架；多场景评估（Best、Initial、Secondary 三个协作上下文）；跨组件关系记忆与共享；配对协同进化；自适应预算分配机制。

**📊 数据集**

数据集包括四大组合优化领域：TSP（SS、S、M、L 等子集）、BPP、CVRP（Set‑A、Set‑B、Set‑P 等）以及 Flowshop；在 ACO 框架下还使用 TSP、CVRP 的组件化版本。每个领域均包含训练集与测试集，使用公开库（TSPLib、BPPLib、CVRPLib、VRF）中的实例。

**📈 对比分析**

与人类手工设计的默认启发式、以及三种扩展的 LLM‑AHD（EoH、ReEvo、MCTS‑AHD）的单/批/分布式进化模式进行对比。MuEvo 在所有 16 个 SHH 数据集和 8 个 ACO 数据集上均显著降低平均最优性缺口，例如在 TSP-L 从 6.26% 降到 5.24%，在 Flowshop VRF100 从 3.51% 降到 3.24%。实验还验证了跨控制器迁移效果、消融实验与不同 LLM 后端的鲁棒性。

**⚠️ 局限性**

局限性：①仍依赖 LLM 的生成质量，模型改进可能带来收益；②实验预算有限，真实大规模优化场景中仍需评估可扩展性；③仅验证了两种结构化多启发式框架，未知对更复杂或动态交互结构的适应性；④算法实现复杂，调参较多，对非专业用户不易上手。

---

## 542. When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation

**arXiv ID:** 2608.03632 | [PDF](https://arxiv.org/pdf/2608.03632v1)

**作者:** Yinuo Jiang `[一作]` (Zhejiang University), Tiankai Li `[通讯]` (ByteDance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SA-OPD框架，通过对OPD中的token级教师-学生差异进行输入无关性检测，过滤掉无输入依赖且高影响力的“spurious signals”，从而提升蒸馏质量。

**💡 创新点**

创新点在于：① 识别并量化OPD中的spurious signals；② 用教师-学生差异在有输入与无输入两种情形下的差值（ΔIG）作为轻量级输入-groundedness代理；③ 将低ΔIG与高绝对差值（|A_t|）结合，构成双重过滤标准；④ 通过动态阈值自适应控制过滤比例。

**🔧 技术方法**

核心技术包括：反向KL目标、token级教师-学生差异计算、输入无关代理ΔIG、基于阈值的token筛选、动态FLMR阈值控制、梯度裁剪/重加权等。

**📊 数据集**

实验数据集涵盖：LLM任务 – DeepMath（AIME 2024/2025、Math500、AMC 2023、MinervaMATH）和VLM任务 – VERO-600K（Captioning、IF、Grounding、Counting & Search）、MMRL30k、EvoChart、MMIFEval、CountQA、MathVision、Geo3K、MathVista。

**📈 对比分析**

与Vanilla OPD、ExOPD、TIP、FiRe-OPD等182/183个基线对比；在LLM与VLM任务上均实现显著提升，平均提升1.6–3.5分，尤其在VLM视觉理解/推理和LLM数学推理任务中获得最高分。

**⚠️ 局限性**

局限性包括：① 需要额外计算无输入条件下的差异，带来轻量级但可累积的计算开销；② 对极低输入依赖或低梯度token的过滤仍可能误删；③ 需要调节动态阈值β，过度敏感会影响收敛；④ 仍受教师模型质量与先验偏差的影响，无法完全消除所有先验信号。

---

## 543. MultiCompose: Multi-Concept Personalized Composition with Per-Subject Attribute Binding

**arXiv ID:** 2608.03708 | [PDF](https://arxiv.org/pdf/2608.03708v1)

**作者:** Ruirui Zhang `[一作]` (Nanjing University of Aeronautics and Astronautics), Pan Gao `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MultiCompose 框架，实现多概念个性化图像生成并完成属性绑定，同时构建 MSP‑Bench 评测基准。

**💡 创新点**

创新点：将概念个性化与多主体组合分离，采用语义保持正则化、两阶段推理（预融合+融合）、跨主体注意力抑制和软掩码路由，显著提升身份保持与属性绑定效果。

**🔧 技术方法**

使用技术：Stable Diffusion XL、交叉注意力权重更新、语义保持正则、Token‑level 注意力抑制、MLLM+SAM 软掩码、两阶段推理、MSP‑Bench 评估。

**📊 数据集**

使用数据集：Concept101（单/多概念），MSP‑Bench（自构，包含多属性、两三主体），并与 Mix‑of‑Show、MC² 等方法做对比。

**📈 对比分析**

对比方法：Textual Inversion、DreamBooth、Custom Diffusion、MIP‑Adapter、Mix‑of‑Show、MC² 等；在 CLIP‑T、CLIP‑I、DINO 以及 MSP‑Bench 上均表现优于或相近，尤其在 MSP‑Bench 上获得最高 MSP 0.6932，比 MIP‑Adapter 提升 0.105。

**⚠️ 局限性**

局限性：需要两阶段推理和外部 MLLM/SAM 生成掩码；对极复杂场景、长属性序列或大量主体的处理仍有挑战；缺乏完全端到端学习机制。

---

## 544. To Describe or Construct Statistical Learning Models Using the Category-theoretical Language

**arXiv ID:** 2608.03706 | [PDF](https://arxiv.org/pdf/2608.03706v1)

**作者:** Congwei Song `[一作]` `[通讯]`, Congwei Song

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文尝试用范畴论语言对统计学习模型进行统一表述，并提出一系列构造器以生成新的模型。

**💡 创新点**

创新点在于将范畴论的对象、箭头、函子映射等概念引入统计学习，用“构造器”形式描述模型构造与扩展。

**🔧 技术方法**

主要技术包括范畴论概念、概率模型与统计模型的对应关系、变分推断、自动编码器、RNN、Transformer 等实例的构造。

**📊 数据集**

文中未给出具体数据集，示例主要为理论构造。

**📈 对比分析**

未进行实验比较，性能评价留空；仅给出理论推导与示例。

**⚠️ 局限性**

局限在于理论不够严格，缺乏实验验证，且对范畴论的使用仍偏向语言描述而非形式化证明。

---

## 545. When Agents Learn to Be You: Benchmarking Privacy Leakage, Impersonation Risk, and Defenses in Persona Skills

**arXiv ID:** 2608.03700 | [PDF](https://arxiv.org/pdf/2608.03700v1)

**作者:** Yongli Xiang `[一作]` (University of Sydney), Tongliang Liu `[通讯]` (University of Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AntiSkillBench，用于系统评估 Persona 技能在隐私泄露与身份模仿风险方面的安全性；

**💡 创新点**

创新点在于构造了包含 7,500 条基于用户画像的对话轨迹，定义了两级风险度量（技能层泄露与代理层模仿），并设计了四种线上/离线、主动/被动的防御方案；

**🔧 技术方法**

技术包括三种 Persona 技能蒸馏方法（Direct Distill、Three-stage Distill、Colleague Distill）、基于 LLM 的评估与判定器、以及隐私消除（PS、ADV）和语义水印（SBD）防御；

**📊 数据集**

使用了 50 个模拟用户角色，扩展成 50 组任务提问，生成 2,500 条三轮对话，共计 7,500 条用户发言；

**📈 对比分析**

与三种主流代理（GPT‑5.4、Claude‑Haiku‑4.5、Gemini‑3.6 Flash）和三种蒸馏协议进行对比；实验显示技能泄露普遍存在，代理层模仿风险显著，现有防御效果有限，且防御效果随蒸馏策略变化；

**⚠️ 局限性**

局限性包括：仅评估了模拟对话而非真实用户数据，防御方案多为单一实现，缺乏对不同隐私法规场景的适配，且对蒸馏策略的深度解释不足。

---

## 546. LAEF: A Lead-Agnostic ECG Foundation Model Towards Point-of-Care Diagnostics

**arXiv ID:** 2608.03690 | [PDF](https://arxiv.org/pdf/2608.03690v1)

**作者:** Edoardo Coppola `[一作]` (University of Cambridge), Alberto Signoroni `[通讯]` (University of Brescia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `109c2b71-d051-425c-831f-0c544c24280d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了LAEF，一种能原生处理任意子集心电图导联的基础模型；

**💡 创新点**

创新点在于将多导联心电图视为可变大小的时空图，并通过图注意网络实现导联无关性；

**🔧 技术方法**

使用了图注意网络（GAT）、多阶段掩码节点预训练、随机导联采样与代码书学习等技术；

**📊 数据集**

预训练基于920万份12导联心电图，评估使用18个公开心电图下游数据集；

**📈 对比分析**

与七种现有12导联基础模型对比，在全导联条件下性能相当，在1/2导联下表现优于所有零填补方法，平均AUROC提升约+3.2点；

**⚠️ 局限性**

局限在于尚未解决噪声、导联摆放等与现场点检设备的差异，未在真实点检数据上验证。

---

## 547. TAOT: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in MoE Training

**arXiv ID:** 2608.03676 | [PDF](https://arxiv.org/pdf/2608.03676v1)

**作者:** Lingyun Zhang `[一作]` (Baidu, Inc.), Dou Shen `[通讯]` (Baidu, Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 TAOT，一种拓扑感知的最优传输方法，用于动态专家复制放置，解决 MoE 训练中动态路由导致的负载不平衡，并通过重叠通信与计算来隐藏重量传输的开销。

**💡 创新点**

创新点在于：① 将“峰值削减收益”和“跨节点权重迁移成本”统一建模为熵正则化的最优传输问题；② 使用 Sinkhorn‑Knopp 迭代产生软流量提示，指导整数复制匹配；③ 在执行层面通过 Lagrange 拍卖实现令牌分配并实现通信与计算的重叠。

**🔧 技术方法**

核心技术包括：熵正则化最优传输、Sinkhorn‑Knopp 迭代、整数复制匹配、Lagrange 拍卖令牌分配、GPU‑友好的规划算法、通信重叠设计。

**📊 数据集**

实验使用 Qwen3‑30B‑A3B MoE 模型在 Pile‑test 数据集上进行微调，并在 4×8 A800 GPU 集群上评估不同 EP 规模与初始不平衡场景。

**📈 对比分析**

与 Megatron‑LM、ECHO、LPLB、LLEP 等方法对比，TAOT 在 42.82% 的端到端训练加速、0.3‰ 的损失误差、与 SOTA 相当或更好的负载平衡质量以及高达 74% 的权重通信成本降低方面均表现突出；计划算法开销低于 1%。

**⚠️ 局限性**

局限性包括：需要对每个微批次的负载进行估计，成本矩阵假设为静态；在极端不平衡或频繁拓扑变化的环境下可能需要进一步调整；存储成本随复制槽数增加而上升，且在超大规模 EP 规模时仍需评估可扩展性。

---

## 548. VetScore: Risk-Weighted Fact Verification for Veterinary Long-Form QA with Citations

**arXiv ID:** 2608.03675 | [PDF](https://arxiv.org/pdf/2608.03675v1)

**作者:** Ivan Kartáč `[一作]` (PrimVeterinary), Ondřej Dušek `[通讯]` (Charles University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现VetScore，一个多步骤评估方法，用于判断兽医长文本问答中生成答案对引用摘录的忠实度，并按潜在危害权重调整分数。

**💡 创新点**

首次将分解-验证范式与风险加权评分相结合，提供针对高风险医学场景的可解释事实真实性评估，并兼顾危害潜力。

**🔧 技术方法**

利用LLM完成主张分解、事实验证与危害评分，采用指数加权的归一化公式计算风险调整后得分，并构建专家标注的Meta‑Evaluation数据集。

**📊 数据集**

采集67个真实兽医问答查询，检索PubMed文献，使用六种LLM生成带引文输出共402份；人工标注约4996条主张的事实验证与危害潜力。

**📈 对比分析**

与仅验证、端到端模型对比，使用多种LLM评判器时，VetScore在事实验证与危害评分的Spearman相关系数最高达0.78/0.76，端到端相关低；小型模型如Gemma 4亦能实现与专有模型相近的性能。

**⚠️ 局限性**

仅评估引用摘录的忠实度，未覆盖信息缺失；聚焦文本查询，未涵盖多轮或多模态交互；依赖LLM可能引入偏见；未训练专门的判定模型。

---

## 549. Accountability Asymmetry and Structural Trust in Autonomous AI Systems

**arXiv ID:** 2608.03670 | [PDF](https://arxiv.org/pdf/2608.03670v1)

**作者:** Nathan DeBardeleben `[一作]` `[通讯]` (Los Alamos National Laboratory), Nathan DeBardeleben (Los Alamos National Laboratory)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出自主 AI 系统治理的结构性信任框架，强调责任不对称并建议通过工程异质性实现多层审计

**💡 创新点**

首次系统化描述“工程异质性”作为解决责任不对称的治理机制，提出分离决策、授权、执行和审计的设计原则

**🔧 技术方法**

主要采用理论分析、案例研究与架构设计方法，无具体算法实现

**📊 数据集**

未使用实验数据集，文中以公开事件（如 Hugging Face 事故）为案例

**📈 对比分析**

无定量性能评估，讨论以案例说明治理方案可行性和局限性

**⚠️ 局限性**

缺乏实证验证、可操作细节不足，治理框架对不同规模系统的适配性未系统评估

---

## 550. XiDepth: a Lightweight and Efficient Network for Self-supervised Monocular Depth Estimation

**arXiv ID:** 2608.03666 | [PDF](https://arxiv.org/pdf/2608.03666v1)

**作者:** Elena Izzo `[一作]` (University of Padova), Lamberto Ballan `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为XiDepth的轻量化自监督单目深度估计网络，能够在不依赖深度传感器的前提下，从单帧图像预测深度图。

**💡 创新点**

核心创新在于将XiNet模块迁移至稠密预测任务，消除了深度可分离卷积和注意力机制的高能耗，实现了显著降低参数量（0.8 M）和能耗的同时保持甚至超过现有模型的精度。

**🔧 技术方法**

技术手段包括基于U‑Net的多尺度编码器‑解码器架构、PoseNet估计相机姿态、视角重建自监督损失（光度重投影 + 边缘感知平滑），以及XiNet块的高效卷积实现。

**📊 数据集**

使用KITTI（主干训练/验证/测试）进行性能评估，并在Make3D上无微调验证跨域泛化；在Raspberry Pi 4上进行能耗、内存、CPU占用及推理速度的实测。

**📈 对比分析**

与Monodepth2、R-MSFM3/6、Lite-Mono等现有自监督方法在KITTI Eigen split上比较，XiDepth在七项指标（AbsRel、SqRel、RMSE等）上达到或超过最佳成绩，仅占参数量的1/20；在Raspberry Pi 4上实现0.31 s推理时间、52 mWh能耗，较最优模型分别降低约35 %能耗、40 % FLOPs。

**⚠️ 局限性**

局限性：目前仅在KITTI上进行自监督训练，缺乏大规模预训练；在不同光照或动态场景下的泛化仍待提升，未来可结合ImageNet预训练进一步提升跨域性能。

---

## 551. Decoupling Generation and Selection for Budget-Constrained Faithful Summarization

**arXiv ID:** 2608.03655 | [PDF](https://arxiv.org/pdf/2608.03655v1)

**作者:** Zeyu Wang `[一作]` (Kean University), Meng Xu `[通讯]` (Kean University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种生成-选择框架，先用预训练模型生成多条候选摘要，然后拆分为句子级别，最后在给定句子预算下进行全局选择，生成最终摘要。

**💡 创新点**

创新点在于把生成与选择解耦，利用句子级别的组合优化平衡相关性、事实性与冗余，而不需要重新训练生成器或额外重写步骤。

**🔧 技术方法**

主要技术包括预训练生成器（BART、Llama等）、句子级覆盖度与事实性打分、冗余相似度矩阵，以及DPP/ILP/MMR等组合优化算法。

**📊 数据集**

实验数据集涵盖CNN/DailyMail、Multi-News、FaithBench、TofuEval 等新闻与事实性基准。

**📈 对比分析**

与直接生成、MMR、ILP等方法对比，DPP选择在事实性指标（FactCC、MiniCheck、AlignScore、FactKB、FaithLens）上显著提升，ROUGE、BERTScore等参考重叠指标略低，但整体事实性与可控性更好。

**⚠️ 局限性**

局限性包括对候选池质量的高度依赖、缺乏跨句话语境建模、仅使用句子计数预算、求解近似且计算开销大，以及在非新闻领域的泛化能力待验证。

---

## 552. TARL: Transaction-Aware Reliable Ledgers for Executable Memory Management in Long-Term Agents

**arXiv ID:** 2608.03699 | [PDF](https://arxiv.org/pdf/2608.03699v1)

**作者:** Han Xiao `[一作]` (Xiamen University), Xiaodong Shi `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于事务的可靠记忆账本（TARL）框架，将记忆更新离散化为五种可执行操作，并通过反事实执行监督实现更精确的状态恢复与冲突管理。

**💡 创新点**

创新点在于引入五动作事务机制并配合确定性账本执行器、可靠性比较与目标定位，使得二元写/保留标签无法区分的更新细粒度问题得到解决，同时构建了TARL‑Mem基准。

**🔧 技术方法**

使用文本编码器+目标定位网络、可靠性比较子模块、动作条件化评分网络及确定性执行器，并通过对每个操作产生的下一状态进行比对实现反事实监督。

**📊 数据集**

基于HaluMem、LoCoMo、LongMemEval等数据集转换而来的TARL‑Mem，5422个带动作标签、目标与黄金状态的样本，且进行了实体、主题等拆分。

**📈 对比分析**

在多源、跨域、时间推理、对抗性与序列评估中，与七种代表性记忆系统对比，TARL在5‑way宏F1、下一个记忆状态准确率、冲突保持率最高，内存污染率与校准误差最低。

**⚠️ 局限性**

局限包括对极端噪声或未知源的鲁棒性尚待验证，对复杂多源时间序列推理的依赖仍显不足，且需显式目标定位与可靠性判断，增加了模型复杂度。

---

## 553. Detecting Hallucinations and Recovering Verified Answers in Arabic Islamic Question Answering

**arXiv ID:** 2608.03720 | [PDF](https://arxiv.org/pdf/2608.03720v1)

**作者:** Khaled Ziani `[一作]` `[通讯]` (Independent Researcher), Khaled Ziani (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套联合式的两步评估系统，先检测阿拉伯语伊斯兰问答生成回答是否出现幻觉，再在六个候选答案中选取真确答案。

**💡 创新点**

结合了针对幻觉检测与多选答案恢复的两任务特定适配器，并通过检测与选择之间的一致性目标减少错误复制。

**🔧 技术方法**

使用基于  的微调模型，采用确定性解码、温度0、top‑p1、生成2,048 token，并对输出进行标准化提取标签。

**📊 数据集**

使用了来自 IslamicWeb、QIAS 2025、MAWARITH、PalmX 2025、IslamicFaithQA、HalluTruthQA、IslamQA、Gemini 等来源共计 43,400 对检测样本和 2,600 对多选样本的训练集，以及官方 HalluScoring 2026 测试集。

**📈 对比分析**

与零样本基线（Qwen3‑32B、Falcon‑H1R‑7B 等）相比，本系统在检测的宏F1 0.928、标签准确率 0.935，答案选取准确率 0.895，综合得分 0.912，显著优于基线。

**⚠️ 局限性**

模型容易把可行的但错误的答案视为正确；对多义或可多答案的问题容易误判；检测与选择间缺乏完全一致性，导致检测正确但仍选错答案；对细粒度实体混淆的纠正能力有限。

---

## 554. Predicting Deep Neural Network Training Outcomes from Early Training Telemetry

**arXiv ID:** 2608.03709 | [PDF](https://arxiv.org/pdf/2608.03709v1)

**作者:** Ranjita Naik `[一作]` (Georgia Institute of Technology), Pankaj Kumar Singh `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究单一训练跑的早期内部监测（训练损失、准确率、梯度信噪比、权重范数增长等）能否预测其最终测试准确率、相对性能排名以及训练动态失败。

**💡 创新点**

创新点在于：①只利用单跑自身的内部指标而不与同一搜索中的其他跑比较；②通过受控无超参消融验证梯度与权重级别遥测提供额外信息；③同时预测数值回归目标、相对分类目标和明确的训练失败事件。

**🔧 技术方法**

主要技术是梯度提升树（GBDT）作为预测模型，并对比随机森林、线性回归、神经网络；特征包括对前5个epoch的多项统计量（均值、方差、增量等）以及超参数；实验采用配对无超参消融和多时间窗评估。

**📊 数据集**

数据集为CIFAR‑10和Fashion‑MNIST，使用三种网络架构（ResNet‑18、Compact CNN、两层MLP），共生成23,788个训练跑（15 epoch），并在20%冻结holdout上评估。

**📈 对比分析**

评估方法：在永久冻结的20% holdout上，5个epoch时GBDT取得R²≥0.92、相对分类ROC‑AUC≥0.983、失败预测AUC≥0.991；单个epoch即可达到≈0.94的R²；内部遥测在所有域和任务上相较于仅损失/准确率+超参有统计显著提升，提升幅度因域而异。

**⚠️ 局限性**

局限性：仅在15 epoch的小规模视觉任务、有限的超参维度和优化器范围；采样偏向成功/失败边界导致结果可能过度乐观；未考虑不同随机种子波动；仅在单域使用神经网络预测；预测结果仅为决策支持，不能直接用于自动终止。

---

## 555. LiLa-WAM: Lightweight Latent Reasoning World-Action Model for Robotic Manipulation

**arXiv ID:** 2608.03701 | [PDF](https://arxiv.org/pdf/2608.03701v1)

**作者:** Fan Yang `[一作]` (Tianjin University), Peiguang Jing `[通讯]` (Tianjin University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级的世界动作模型 LiLa-WAM，能够在紧凑的潜在空间中联合预测未来状态和生成动作，并在单块24GB GPU上端到端训练。

**💡 创新点**

主要创新点包括：1）在单一流中同时学习未来潜在预测和动作生成，形成紧凑的潜在推理空间；2）引入 Visual Transition Token (VTT) 的语言无关任务表示，只需事前从演示中计算，无需目标图像或文本；3）采用冻结的 DINOv3 视觉编码器与查询适配器压缩视觉信息，保持计算轻量；4）在训练阶段仅使用 Q-Former 风格解码器进行未来监督，推理时无需额外模块。

**🔧 技术方法**

使用的技术包括：冻结 DINOv3 ViT 编码器、查询适配器（Query-based Adapter）、Diffusion-Transformer (DiT) 块、Q-Former 形式的解码器、条件流匹配（Conditional Flow Matching）用于动作预测、cosine 监督用于未来潜在预测、以及 VTT 的计算。

**📊 数据集**

使用的数据集有：RoboTwin 2.0（50个操纵任务，清洁+随机演示）、LIBERO（四个suite共40个任务）、以及真实机器人实验环境（Agilex Piper 6-DoF + Intel RealSense D435），并在这些环境中收集演示。

**📈 对比分析**

在 RoboTwin 2.0 上，LiLa-WAM 取得 90.48% 的成功率，超过 8B Motus、5B GigaWorld-Policy 等大型模型，并且参数仅 0.5B，约 16 倍/10 倍更少；在 LIBERO 上，取得 97.1% 的平均成功率，匹配 OpenVLA-OFT 的性能，但参数仅 14 倍更少；在真实机器人上，平均成功率提升至 82%（比无未来监督的变体提升 8%）。

**⚠️ 局限性**

局限性包括：① 仍依赖大规模视觉预训练模型（如 DINOv3），若预训练不足可能影响性能；② 对非常复杂或极端变化的任务仍表现不佳（如篮球投篮）；③ 未来预测的时间步设定与动作块长度相等，可能限制更长时序的推理；④ 仅在视觉输入上做任务指定，未能充分利用多模态信息；⑤ 目前尚未在更大规模、跨任务的泛化测试中验证。

---

## 556. Accelerating Dynamic Graph Clustering on GPU Architectures with cuGraph

**arXiv ID:** 2608.03695 | [PDF](https://arxiv.org/pdf/2608.03695v1)

**作者:** Nelson Aloysio Reis de Almeida Passos `[一作]` (University of Pisa), Salvatore Trani `[通讯]` (National Research Council)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究基于GPU的动态图社区检测，提供了在RAPIDS生态系统下的谱聚类和模块化优化实现。

**💡 创新点**

创新点在于将Leiden多切片优化与Bethe-Hessian谱聚类迁移至GPU，实现数百倍加速，并通过NetworkX-Temporal实现零代码改动加速。

**🔧 技术方法**

使用了cuGraph、CuPy、cuML、Dask、RAPIDS，结合Bethe-Hessian、非退行传播、Leiden算法和多切片模块化。

**📊 数据集**

使用了ArxivAI、Dblp、ArxivCS、ArxivMath、Brain、Patent、School以及Synthetic SBM等真实与合成数据集。

**📈 对比分析**

与CPU基准相比，GPU实现平均实现22-64倍加速，最大可达978倍，尤其在快照数量高时实现可行性突破。

**⚠️ 局限性**

局限性包括仅适用于无属性、非嵌套的同质社区结构，以及对极稀疏或极大快照数时构造超图的内存和构建瓶颈。

---

## 557. SITA: Semantic Interest Tokens for Target-Aware Compression in Long-Sequence Recommendation

**arXiv ID:** 2608.03692 | [PDF](https://arxiv.org/pdf/2608.03692v1)

**作者:** Rui Zhou `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在长序列推荐中提出一种基于语义兴趣代币的目标感知压缩框架 SITA，用于高效捕捉用户整体兴趣并在推理时快速针对不同候选物品进行个性化选择。

**💡 创新点**

创新点包括：① 用 Balanced Parallel Quantization (BPQ) 构建可组合的语义空间，实现 O(|U|NK) 的存储复杂度；② 通过 Structured Interest Compression (SIC) 堆叠块在语义组内进行细粒度建模并在组间进行交互，得到结构化兴趣代币；③ 采用 SID‑Guided Selection (SGS) 依据目标物品的语义标识动态挑选相应兴趣代币，从而兼顾目标感知与全局兴趣。

**🔧 技术方法**

核心技术包括：多码本向量量化 (BPQ)、多层自注意力 + 组内 Swiglu 变换 (SIC)、组间交互 (Self‑Att) 与目标注意力 (Target Attention)、语义标识生成与使用 (SID)。

**📊 数据集**

使用公开数据集 Taobao‑MM、XLong 以及工业级大规模用户日活数百万的推荐平台数据进行评估。

**📈 对比分析**

与 DIN、SIM、TWIN、MUSE、C‑Former、UxSID、STCA、LONGER 等基线比较，SITA 在 AUC 与 GAUC 指标上均实现显著提升（最高提升约 3.8% 的 GAUC，工业场景中 0.05% 的绝对提升），且在线推理复杂度保持在 O(BND)，与现有压缩方法相当。

**⚠️ 局限性**

局限性：① 对语义空间配置 (N,K) 与 SIC 层数 L 的超参数较为敏感；② 需要额外训练 BPQ 的语义标识，增加离线预处理成本；③ 在极度稀疏或极端多样化的物品场景下，语义标识分布可能出现不均衡，影响兴趣代币的表达效果。

---

## 558. Pattern over Pixels: Measuring Pattern Completion Bias in Multimodal Code Generation

**arXiv ID:** 2608.03691 | [PDF](https://arxiv.org/pdf/2608.03691v1)

**作者:** Khai-Nguyen Nguyen `[一作]`, Antonio Mastropaolo `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了Pattern2Code基准，用以衡量多模态大模型在网页截图到代码任务中的模式完成偏差。

**💡 创新点**

首次系统量化视觉模式完成偏差，揭示了视觉显著性与模型偏差之间的关系。

**🔧 技术方法**

利用多模态大型语言模型对截图-代码填空任务进行推理，并分析模型的推理过程。

**📊 数据集**

采集并改造了Design2Code数据集中的30个网页，生成1,440个带有单一局部扰动的实例。

**📈 对比分析**

与五大前沿M模型对比，发现准确率低至7.9%（文本），偏差率高达80.22%，噪声、微调和位置变化进一步放大偏差。

**⚠️ 局限性**

局限在于仅评估两类局部扰动、使用合成噪声、且仅涉及闭源模型，未涵盖更广泛的网页结构与开源模型。

---

## 559. CausalOPD: First-Wrong-Step Supervision for Distilling Causal Chain Reasoning

**arXiv ID:** 2608.03673 | [PDF](https://arxiv.org/pdf/2608.03673v1)

**作者:** Jian Zhang `[一作]` (Zhejiang University), Yizhi Liu `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CausalOPD，一种利用知识校准教师、第一错误定位和基于因果阶段的课程化短期强化学习的在线过程蒸馏框架，用于将大型语言模型的多步因果推理能力迁移到小型可本地部署模型。

**💡 创新点**

创新点在于：①通过显式域知识和约束对学生生成的推理路径进行可信验证；②首次错误定位（First‑Wrong‑Step）技术将修正边界精确到产生错误的最早过渡；③依赖因果链依赖关系的课程化训练顺序，将修正从证据识别→机制推断→结论归因递进。

**🔧 技术方法**

技术方法包括：知识增强的教师构造-评估-定位流程、可校准的路径评估器、短期强化学习（基于局部错误的奖励）、修正状态细化（Correction‑State SFT）以及动态迭代在线蒸馏与课程化调度。

**📊 数据集**

使用了三大行业级因果推理基准：工业（air‑handling‑unit 故障诊断）、医学（呼吸系统诊断）、法律（内幕交易裁决推理），每个基准都配有领域知识库和约束模板。

**📈 对比分析**

与零样本教师、轨迹SFT、结论级RL、序列级OPD、全轨迹过程RL等五个对照方法比较，CausalOPD 在三大领域的路径正确率平均提升约 23.4pp，结论准确率亦显著提升，并将“正确标签—错误推理”率从 15.7% 降至 4.4%，在工业子集上更是实现 34.6pp 的跨系统性能提升。

**⚠️ 局限性**

限制包括：需要手工构建的高质量域知识库与约束模板；对知识覆盖不足时仍可能出现未检测的错误；定位与修正过程受限于评估器的可信度与推理可解释性的前提；目前实验仅覆盖结构化推理任务，对更自由文本推理的泛化能力尚未验证。

---

## 560. When Outputs Disperse, Does Epistemic Revision Follow? A Black-Box Coupling Diagnostic for Machine Collectives

**arXiv ID:** 2608.03722 | [PDF](https://arxiv.org/pdf/2608.03722v1)

**作者:** Molood Arman `[一作]` `[通讯]` (Independent Researcher), Molood Arman (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计了一种黑盒诊断方法，验证机器集合体在输出多样性与知识修正之间的耦合关系，并在两种不同配置的语言模型上进行实验。

**💡 创新点**

创新点在于将“输出分散-修正耦合”概念具体化为可操作的诊断流程，并引入Coherence Index（CI）与Meta-Predictive Clarity System（MPCS）来检验干预后输出散布与实际修正是否同步。

**🔧 技术方法**

技术上使用了文本嵌入（text‑embedding‑3‑small）计算CI，采用RDP（Re‑Differentiation Protocol）作为干预语句，基于LLM生成的多轮对话进行随机触发、静态角色分配等对照实验，并通过人工判别员评估修正程度。

**📊 数据集**

实验数据集为31个包含错误前提的任务模板（生物学、历史、物理等领域），每个模板在10个随机种子下运行，共计310个episode/配置，使用gpt‑4o‑mini和gemini‑2.5‑flash两种模型配置。

**📈 对比分析**

与基线（无干预）相比，条件性异议在gpt‑4o‑mini上提升恢复率17.7个百分点，随机异议亦有显著提升；但在gemini‑2.5‑flash上无显著改进。通过CI‑立场相关性、即时立场移动量、保留错误前提比例等指标量化耦合程度，表明两种模型在同一干预下的耦合强度截然不同。

**⚠️ 局限性**

局限性包括：仅测试两种单一配置，无法推广到整个GPT或Gemini系列；CI为单一编码器的第一时刻散度度量，可能受模型风格影响；干预不是完全因果中介，仅验证输出层面效果；样本量对每次RDP触发的细粒度统计有限；缺乏对内部表示的白盒分析。

---

## 561. Less Traffic, Better Outcomes: Competition-Aware Request Dispatch in Real-Time Ad Exchanges

**arXiv ID:** 2608.03705 | [PDF](https://arxiv.org/pdf/2608.03705v1)

**作者:** Jonaid Shianifar `[一作]` (Huawei), Bichen Shi `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并部署了一套竞争感知的RTB请求分发框架，利用分布式bid预测和概率转发，挑选性地将请求发送给对拍卖结果贡献更大的DSP，从而在保持交易效率的同时显著减少DSP请求量。

**💡 创新点**

① 将竞价价值作为分发决策的核心目标，而非仅仅追求响应率；② 将分布式bid预测与概率转发策略结合，形成软阈值转发；③ 采用PPO对分发阈值进行在线自适应优化，跟踪非平稳市场条件。

**🔧 技术方法**

使用Deep & Cross Network（DCN）生成分布式bid预测模型，Gamma分布回归估计bid值；概率门控+平滑门控实现转发概率；PPO进行阈值优化；离线训练+在线推理；ratio‑DID因果评估。

**📊 数据集**

基于生产RTB日志（每日约20亿请求），包含DSP响应、bid、拍卖结果等信息，采用最近7天滚动窗口进行模型训练与阈值更新。

**📈 对比分析**

对比四种离线策略（全转发、随机过滤、最优静态阈值、PPO适配）及四轮线上实验（单DSP与全DSP）。PPO策略将DSP请求量降低约35%，最高bid略升，净收入提升4.6%（p<0.001），fill率提升40%，impression率提升3–15%，点击率略升。

**⚠️ 局限性**

限制：PPO优化基于聚合统计，未建模单个拍卖的动态；离线仿真无法完全再现DSP行为；实验仅在单一交换机环境中进行，缺乏跨平台验证；未对DSP策略调整与流量组成的因果关系进行细致分离。

---

## 562. Learning and Clustering on Temporal Graphs: Principles, Primitives, and Pooling

**arXiv ID:** 2608.03696 | [PDF](https://arxiv.org/pdf/2608.03696v1)

**作者:** Nelson Aloysio Reis de Almeida Passos `[一作]` (University of Pisa), Salvatore Trani `[通讯]` (National Research Council)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在时间图上学习与聚类的关系，提出将时间聚类视为统计原则、可扩展计算原语和池化机制，并实现了 GPU 加速的多切片模块化优化与 Bethe‑Hessian 谱方法。

**💡 创新点**

1) 将社区检测的可检出性理论与图神经网络结合，形成可解释的时间池化；2) 通过将非对称非回溯矩阵改写为对称 Bethe‑Hessian，突破 GPU 端点，获得可扩展的谱解；3) 在多时间步数据上实现了三阶量级的加速。

**🔧 技术方法**

GPU 加速的多切片模块化优化、CuPy/RAPIDS (cuGraph, cuML) 生态、Bethe‑Hessian 变换、非回溯矩阵的谱近似、可区分的时间可检出性阈值等。

**📊 数据集**

Cora, CiteSeer, PubMed, Patent, Dblp, Brain, ArxivAI, ArxivMath, ArxivCS。

**📈 对比分析**

与 CPU 参考实现在相同工作量下比较，GPU 版本在密度与时间步数不同的数据集上实现了最高三阶量级的速度提升，且在大规模图上实现了可行性；在社区检测准确性上，算法基线在无属性图上优于神经模型，属性图中仅在信号耦合强时才有神经优势。

**⚠️ 局限性**

对属性时间图的学习收益仍有限；对非对称非回溯矩阵的直接 GPU 端点缺失；神经模型在属性和时间信息不一致时效果不佳；需要进一步探究时间依赖关系在池化中的保持。

---

## 563. LiveEvalBench: Toward Open-World Evaluation for Web Generation

**arXiv ID:** 2608.03689 | [PDF](https://arxiv.org/pdf/2608.03689v1)

**作者:** Yiyao Wang `[一作]` (Zhejiang University), Wei Chen `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LiveEvalBench，一个自动化的多代理评估框架，用于评价LLM生成的前端项目。

**💡 创新点**

将评估从一次性打分转为协作评审工作流，引入适应性评估协议和可扩展的评估基础设施，支持多视角、多层次的评估。

**🔧 技术方法**

构建了Build Engineer、Code Engineer、UI Tester三种评估代理，使用LLM（如Gemini 3.1 Flash）进行代码、构建与浏览器交互评测，并实现了实现相关检查的自动合成。

**📊 数据集**

构造了45100个真实前端生成任务（覆盖L1–L3难度、6类任务、3种规格粒度），并使用这些任务评估11个前沿LLM生成的项目。

**📈 对比分析**

在LiveEvalBench上对11个模型进行分层评分（90分制，子分为Build/Code/UI），Claude Opus 4.7获得最高70.7分，UI行为差距最大，难度越高分数下降；评估结果与人工判断一致率达到86.8%，显示可靠性。

**⚠️ 局限性**

评估受构建失败与环境差异影响，主要关注交互运行时错误，缺乏对多样化实现细粒度评判；跨平台兼容性与多模态适配尚未充分覆盖。

---

## 564. PhyAI: Real-Time Physical AI at the Edge, Scalable Rollouts in the Cloud

**arXiv ID:** 2608.03682 | [PDF](https://arxiv.org/pdf/2608.03682v1)

**作者:** Chenghua Wang `[一作]`, Ziqi Guo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了PhyAI，一个统一的物理AI推理运行时，能够在机器人本地、边缘与云端部署相同的模型路径。

**💡 创新点**

其创新点包括将模型语义与执行策略分离、提出控制时间Roofline分析方法以及多种优化技术（算子融合、CUDA图回放、量化、并行化）。

**🔧 技术方法**

实现中使用了CUDA Graph、FlashInfer、Humming、算子选择器、W4A8/W8A8量化、数据并行/张量并行/CFG并行等技术。

**📊 数据集**

评测以π_0、π_0.5、GR00T N1.7、MiniCPM-Robot和Cosmos3等模型为对象，并在LIBERO四套基准环境以及不同硬件（Jetson Thor、RTX5090、A40、H100、H20×8）上进行实验。

**📈 对比分析**

通过与官方实现对比单请求延迟，PhyAI在11组模型-设备组合上获得1.40×–4.65×的加速；在模拟RL回放中，将推理占比从53.1%降至36.2%，提升约1.36×训练吞吐量。

**⚠️ 局限性**

局限性包括未与专用运行时完全匹配精度、未考虑观测捕获、网络传输与队列延迟、仅针对单GPU性能、对Cosmos3多GPU扩展不足以及缺乏完整闭环任务成功率评估。

---

## 565. Active Stiffness Control of a Supportive Continuum Robot

**arXiv ID:** 2608.03677 | [PDF](https://arxiv.org/pdf/2608.03677v1)

**作者:** Rana Danesh `[一作]` (Toronto Metropolitan University), Farhad Aghili `[通讯]` (Concordia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在支持型连续机器人（SCR）中实现了在线的任务空间主动刚度控制，以提高其在不同负载和工作配置下的位姿精度与载荷抗力。

**💡 创新点**

创新点在于：①将闭链约束通过投影方式融入GVS动力学模型；②采用投影滑模控制实现位姿跟踪；③在位姿稳态后加入基于位置误差的虚拟卡尔曼弹簧实现主动刚度调节；④验证了该方法在不同工作姿态下的可调性。

**🔧 技术方法**

采用了几何可变应变（GVS）模型、投影滑模控制、约束一致投影、虚拟卡尔曼弹簧（主动刚度）以及逆运动学投影投射技术。

**📊 数据集**

使用了数值仿真数据（Nitinol 连续机器人单链模型）和实验平台测量数据（采用 Vicon 动作捕捉与 Dynamixel 伺服）。

**📈 对比分析**

通过与仅使用滑模控制（K_app=0）的基线对比，实验表明在 100 g 负载下将 K_app,x 设为 3 N/mm 可使末端位移减少约 25% 并提升方向性刚度约 34%；在仿真中将 K_app,x 设为 20 N/mm 可使位移降低约 42% 并提升刚度约 71%。

**⚠️ 局限性**

局限性包括：①刚度响应高度依赖工作姿态，需对每个姿态进行校准；②未考虑绳索摩擦、伺服分辨率、测量误差等非线性效应，导致实验与仿真结果差异；③仅针对已知外部负载进行控制，动态未知负载的适应性尚待研究。

---

## 566. DiagLoop: A Counterfactual Data Flywheel with Stage-Localized Reinforcement for Diagnostic LLMs

**arXiv ID:** 2608.03674 | [PDF](https://arxiv.org/pdf/2608.03674v1)

**作者:** Jian Zhang `[一作]` (Zhejiang University), Yizhi Liu `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一种闭环训练框架 DiagLoop，利用已编码的物理/临床机制生成可验证的合成诊断场景，并通过阶段化的强化学习提升诊断路径正确率。

**💡 创新点**

创新点包括：① 用同一套阶段标准在筛选、失败定位、奖励、再生成等环节复用；② 通过生成可验证的对抗性世界（counterfactual worlds）让模型学习因果链而非仅记忆；③ 结合局部化强化学习与弱点引导生成的闭环显著提升路径正确率。

**🔧 技术方法**

采用对抗性场景生成+规范检查（Hybrid checker）、基于阶段标准的错误定位与局部化奖励（GRPO）、软重放与保存策略保持已学技能，并以 8B Qwen3‑8B 为学生、Qwen3.7‑Max 为教师/检查器。

**📊 数据集**

训练完全基于合成场景；评估使用 LBNL 工业故障基准（8 个系统、91 种故障）和 DDXPlus 临床诊断基准（10 类疾病、128,800 病例），两者仅用于评估。

**📈 对比分析**

与多种基线（无过滤、答复一致过滤、约束录入、均匀随机 RL、失配路由等）按预算对齐后，DiagLoop 在工业上 Acc/Path 94.66/91.97，临床 79.84/70.23，严格路径正确率相较最强基线提升 11.6/5.5 点，超过商用零射线参考并显著降低“对标签对但路径错误”现象。

**⚠️ 局限性**

局限性：依赖已编码的机制与检查器准确性，适用于可规则化的因果关系；泛化仅限于同一机制族内的配置组合；未在临床真实流程中验证，需要进一步的临床专家迭代。

---

## 567. Morphology-Aware Implicit Super-Resolution Network for Pathological Images

**arXiv ID:** 2608.03664 | [PDF](https://arxiv.org/pdf/2608.03664v1)

**作者:** Jiaming Liang `[一作]` (South China University of Technology), Hongmin Cai `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向数字病理的 Morph-ISR 形态学感知隐式超分辨率框架，能够在保持像素级精度的同时，恢复细胞边界与核纹理的诊断关键信息。

**💡 创新点**

创新点在于：① 将超分辨率建模为连续坐标查询，并通过 Implicit Position-aware Kernel Generator (IPKG) 生成空间自适应卷积核以适应多变的组织形态；② 通过 Morphological Fidelity Prior (MFP) 将细胞核掩码与边界图引入训练，形成区域与边界约束，显著提升结构保真度；③ 结合残差修正、色彩一致性与结构锐化等模块，形成完整端到端的隐式重建流水线。

**🔧 技术方法**

使用了：隐式坐标查询 + Fourier 编码 + gated 多头 MLP + 位置感知卷积核生成器 + 频谱 + 感知 + SSIM 结合的像素损失 + MFP 语义约束 + 残差增强网络 + 通道均值匹配等技术；训练采用 AdamW、自动混合精度、梯度裁剪，硬件为单张 RTX PRO 6000 GPU。

**📊 数据集**

使用了 TCGA 三个子队列（LUAD、KIRC、LIHC）的 150 张 H&E 玻片（训练/验证/测试 7:1:2），以及独立的 SurGen 结直肠癌 46 例进行零样本泛化测试。

**📈 对比分析**

与 Bicubic、SHISRCNet、SwinIR、ESRGAN、SinSR、STAR-RL、UPSR、SuperDiff、LIIF 等基线方法对比，Morph-ISR 在 LPIPS、ST‑LPIPS 上实现了 38–40% 的提升，PSNR 与 SSIM 亦保持首位或接近首位；模型仅 0.829 M 参数、189.2 G FLOPs，推理吞吐 11.89 FPS、延迟 7.153 ms，证明其在边缘部署上的高效性。

**⚠️ 局限性**

局限性包括：① 需要高质量的 LR–HR 配对训练数据；② MFP 依赖预训练的细胞分割网络，若分割精度下降会影响约束效果；③ 目前仅在 H&E 组织切片上验证，跨染色或其他组织类型的泛化能力仍待进一步评估。

---

## 568. Taming the Implicit: Dual-Channel Risk-Aware Reinforcement Fine-Tuning for Continual Multimodal Post-Training

**arXiv ID:** 2608.03660 | [PDF](https://arxiv.org/pdf/2608.03660v1)

**作者:** Yibei Liu `[一作]` (University Of Electronic Science And Technology Of China), Yangyang Wu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了多模态大型语言模型在持续后期训练（Continual Post‑Training）中通过强化学习微调（Reinforcement Fine‑Tuning）减缓灾难性遗忘的难题，并提出了一种双通道风险控制框架 RAPO。

**💡 创新点**

创新点在于用 Rollout 可靠性与 Fisher 启发的局部预测敏感度两种指标对每个样本的优化风险进行显式估计，并分别在策略通道采用 R‑Scale 自适应缩放损失，在数据通道采用 R‑Sample 动态风险桶采样，从而实现无跨任务记忆的显式风险治理。

**🔧 技术方法**

技术上结合了现有的强化学习微调算法（如 PPO、GRPO、RLOO）与 R‑Scale、R‑Sample 两种风险调节模块；使用 Rollout 可靠性、Fisher 信息近似、局部预测敏感度等量化指标，并通过动态分桶采样构建训练批次。

**📊 数据集**

实验使用公开的 MLLM‑CL 基准（共 8 个任务，分别为 RS、Med、AD、Sci、Fin、OCR、Math & Logic、GUI Agent，每任务 10k 样本）以及 Qwen2.5‑VL‑3B‑Instruct 作为基础模型。

**📈 对比分析**

与多种 SFT 与 RFT 基线（Sequential SFT、EWC、L2P、ReMax、GRPO、RLOO）在相同无重放的连续训练设置下进行比较，评估指标为 MFN、FM 等；结果显示 RAPO 相较 RLOO 将最终遗忘率降低 79.8%，同时保持竞争的新任务学习性能。

**⚠️ 局限性**

局限性包括：仅在无重放的持续训练场景下验证，未评估在有经验回放或多任务并行中的表现；对极端分布偏移下长期多阶段训练的理论分析不足；样本级风险估计与动态分桶的计算成本和可扩展性仍待进一步研究。

---

## 569. When Do Fewer Visual Tokens Accelerate Multimodal Inference? A Break-Even Study Across Decision Locations and Hardware

**arXiv ID:** 2608.03649 | [PDF](https://arxiv.org/pdf/2608.03649v1)

**作者:** Hao Dou `[一作]` (Harbin Institute of Technology), Ruiwen Tian `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一套可复现的“break-even”评估协议，用以量化视觉令牌削减、决策开销与实际端到端延迟之间的关系，并在 Qwen2.5-VL-3B 模型上对多种视觉预算策略（包括后视裁剪的 Static‑0.900、预视图分辨率规则 Image‑Size Rule 以及学习型 Router）进行实验，比较其在 RTX 3090 与 A100 两种 GPU 上的延迟收益与质量保持情况。

**💡 创新点**

创新点在于：
1) 提供一种完整的阶段分解与可重复的计时框架，将决策成本、可共享工作与可避免操作分别量化，并与测得的整体延迟对齐；
2) 将“break‑even”概念转化为可检验的统计准则（包括配对置信区间与 Holm 多重校正），使得在不同硬件环境下的加速效果能够被系统化地评估；
3) 通过对比后视裁剪与预视图分辨率决策，揭示了在视觉编码已完成后裁剪并不能总是带来更低延迟，而预视图策略能利用预处理与编码的结构性机会，尤其在 A100 上显著超越后视裁剪。

**🔧 技术方法**

使用的技术包括：
- Qwen2.5‑VL‑3B‑Instruct 模型（bfloat16, greedy 解码, 24 token 上限）;
- 基于视觉编码器输出的视觉令牌比例裁剪（Static‑0.900）;
- 基于图像尺寸的分辨率切换规则（Image‑Size Rule）;
- 统计分析工具：配对 bootstrap、sign‑flip 检验、Holm 校正;
- 计时与日志收集：对每个执行阶段（共享、决策、执行）进行分段测量。

**📊 数据集**

使用了三个公开 VQA 数据集：VQAv2、TextVQA 与 ChartQA，共计 2,600 条样本用于构建安全预算与基线；随后在 120 条平衡采样样本上重复测量延迟；以及 403 条满足 Full‑correct 条件的样本用于质量评估。

**📈 对比分析**

比较方法：对每种策略在两种 GPU（RTX 3090 与 A100）上执行相同的 120 条样本，计算与 Full 的配对延迟差值，利用 10,000 次 bootstrap 估计置信区间，并使用 Holm 校正验证统计显著性。结果显示：
- Static‑0.900 在 RTX 3090 上平均减少 27.1 ms，A100 上 4.0 ms；
- Image‑Size Rule 在 RTX 3090 上减少 14.0 ms，A100 上 8.9 ms；
- Static 的延迟优势在两台机器上均通过 Holm 校正保持显著，而 Rule 的优势仅在 RTX 3090 上显著。 
质量方面，两者在三任务宏量上的分数分别为 0.9833 与 0.9949，均略低于 Full。

**⚠️ 局限性**

限制包括：
- 仅评估单一模型族（Qwen2.5‑VL‑3B）和三大 VQA 数据集；
- 批量大小固定为 1，未考虑大批量或多节点部署场景；
- 仅对满足 Full‑correct 条件的样本进行质量评估，未覆盖 Full‑error 情况；
- 预视图与后视裁剪的决策位置不是通过实验隔离比较；
- Rule 的加速集中于少数高分辨率 TextVQA 样本，真实流量中此类样本的比例未知；
- 所有实验均基于 GPU/软件版本，未评估在不同硬件或混合精度下的可迁移性。

---

## 570. Conditionally Identifiable Latent-Environment Modeling for Out-of-Distribution Recommendation

**arXiv ID:** 2608.03647 | [PDF](https://arxiv.org/pdf/2608.03647v1)

**作者:** Qianqian Wang `[一作]` (Southern University of Science and Technology), Lili Yang `[通讯]` (Southern University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在隐藏环境变化下仍能保持排序性能的推荐框架CILER，并在理论上证明其在满足可观测特征变异、模型规格与解码器正则化条件下的可识别性及部署风险上界。

**💡 创新点**

创新点：① 将隐藏环境建模为用户特征条件的指数族分布，① 通过特征索引多项式结构约束环境对偏好敏感分量的影响，② 在部署时对环境进行边缘化预测，从而在不需环境标签或测试时更新的前提下实现可识别的环境敏感表示并提升OOD性能。

**🔧 技术方法**

核心技术包括：条件指数族潜在变量建模、特征索引多项式因果机制、变分推断、边缘化预测、可识别性理论与KL风险上界证明。

**📊 数据集**

使用三个公开的推荐数据集：Synthetic、Meituan（时间分割）和Yelp（地理分割），并在各自的OOD评测协议下进行实验。

**📈 对比分析**

与13种基线（如FM、NFM、MultiVAE、COR、DT3OR、CDR等）比较，CILER在所有12个OOD排名指标上均取得第一名，最显著的提升达25.6%，且在IID评测协议下保持竞争力。

**⚠️ 局限性**

局限性：① 需要充分的特征变异与正确的模型规格（指数族类型、多项式阶数、因果顺序）才能保证可识别；② 仅适用于共享支持的分布移位；③ 环境维度与参数设置对性能敏感，需人工调优；④ 对于极端稀疏或非共享支持的OOD场景，理论与实验表现尚不充分。

---

## 571. Towards Reliable and Reproducible Fetal Brain Biometry: A Deep Learning Approach Using MRI

**arXiv ID:** 2608.03724 | [PDF](https://arxiv.org/pdf/2608.03724v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 572. SAT-Edge-Agent: Hardware-in-the-Loop Edge-Agent Orchestration for Onboard Satellite Intelligence

**arXiv ID:** 2608.03728 | [PDF](https://arxiv.org/pdf/2608.03728v1)

**作者:** Longji He `[一作]` (OgCloud Limited), Jeto Xu `[通讯]` (OgCloud Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在商用 ARM SoC 上实现 SAT-Edge-Agent，将浏览器前端、FastAPI 后端、本地 YOLO‑Style OBB 检测服务与本地 OpenAI‑兼容 LLM 组合，完成任务层在边缘卫星人工智能系统中的可观察、可复现工作流。

**💡 创新点**

提供硬件在环（HIL）下的完整端到端体系结构，区分机器面工具结果与操作员面叙述，并证明检测仅占总体延迟的 2–3%，从而表明优化重点在编排与响应生成，而非单纯的模型推理；同时公开了可复制的服务契约与数据记录。

**🔧 技术方法**

使用 ARM Cortex‑A55 多核 SoC（Debian 11）、FastAPI + LangChain、Uvicorn、Python 3.13、OpenAI‑兼容 LLM 接口、YOLO‑Style OBB 检测（YOLO26）、SSE 流、CPU/NPU 采样（devfreq/sysfs）等技术栈。

**📊 数据集**

采用 FAIR1M 远程感测图像作为检测与元数据测试集，用于生成目标类别、置信度、定向多边形和地理坐标等结构化输出。

**📈 对比分析**

对两个固定工作负载（单图像与串行两图像）进行 20 次重复实验，测量 Full‑Agent 延迟和检测延迟。单图平均 Full‑Agent 延迟 29.353 s（P95 31.166 s），两图平均 60.937 s（P95 66.882 s）。检测仅占整体平均延迟的 2.93% 与 2.48%；CPU 平均占用约 20%，NPU 采样为 100%（共享加速器）。

**⚠️ 局限性**

仅在地面 HIL 环境完成实验，未进行飞行验证或辐射环境测试；未评估检测准确率、能耗或长期鲁棒性；工作负载有限（单张与两张固定图像），缺乏并发、批量或更大分布的测试；未对前端渲染、内部子阶段进行细粒度拆解；能耗测量来自独立的 plug‑meter 试点，未同步校准。

---

## 573. GPTKB 2.0: Direct Construction of Disambiguated Knowledge Bases from Large Language Models

**arXiv ID:** 2608.03729 | [PDF](https://arxiv.org/pdf/2608.03729v1)

**作者:** Yujia Hu `[一作]` (ScaDS.AI Dresden/Leipzig and TU Dresden), Simon Razniewski `[通讯]` (ScaDS.AI Dresden/Leipzig and TU Dresden)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（LLM）直接从参数中递归抽取事实，构建了一个规模约 1.6 M 实体、38.4 M 三元组的知识库，并通过双轮次、上下文驱动的实体/关系/类别去歧义技术实现了同义词和多义词的实时消解。

**💡 创新点**

核心创新包括：① 上下文驱动的双轮次 NED（先用三元组上下文检索候选，再用生成的实体描述做二次检索）实现对同义词与多义词的高精度去歧义；② 有守护的并行化策略避免并行处理中重复实体的产生；③ 在不依赖外部知识库 ID 的情况下完成实体统一化；④ 公开了首个百万级 LLM 原生去歧义知识库。

**🔧 技术方法**

技术细节包括：LLM 任务（知识挖掘、NER、描述生成、去歧义）使用 GPT‑5.1 / GPT‑5‑mini；候选检索采用 Qwen3‑Embedding‑4B；prompt 设计实现上下文化；双轮次 NED、描述生成、缓存与并行化管线。

**📊 数据集**

数据来源完全来自 LLM 参数；起始种子实体为 E0Vannevar Bush；评估使用 1,000 条随机三元组和 1,000 条随机实体样本；不使用外部实体集或语料库。

**📈 对比分析**

与仅使用表面字符串识别实体的基线进行对比：基线 2,316,275 实体、131,990 同名实体冲突；去歧义 KB 1,592,185 实体、仅 30 个同名实体冲突。NED 合并精度 98%（同义词）/91%（同义词），拆分精度 100%/95%。三元组真值率约 94–92%，实体可验证率 96–92%。总体成本约 6,992 美元，耗时 70 天。

**⚠️ 局限性**

局限性：① KB 仅包含事实，无引用/出处信息；② 与 Wikidata 比较仍相对较小，需重新跑更新；③ KB 静态，随时间知识会过时；④ 仍有少量同义词误合并，尤其在长尾实体上；⑤ 可能继承 LLM 的偏见和幻觉。

---

## 574. Track4Action: Distilling World-Centric 3D Tracker into Vision-Language-Action Policies

**arXiv ID:** 2608.03727 | [PDF](https://arxiv.org/pdf/2608.03727v1)

**作者:** Chenyi Wang `[一作]` (Zhejiang University), Lixin Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在训练阶段利用冻结的 3D 追踪器对演示视频中动作对应的空间转移进行特权监督，并将该特征蒸馏到 VLA 策略中的跟踪查询，从而实现无追踪器的机器人控制。

**💡 创新点**

提出将动作对齐的 3D 场景变化特征作为训练时的“特权监督”，并通过跟踪查询与 gated fusion 机制将其融入动作头，实现在部署时不需要追踪器或演示视频的高效 VLA。

**🔧 技术方法**

使用冻结的 world-centric 3D tracker（生成轨迹、场景、运动、可见性与摄像机参数），可学习的跟踪查询，特征对齐损失，flow‑matching 动作头，以及特征门控融合技术。

**📊 数据集**

LIBERO 原始数据集、LIBERO‑Plus（7 种零射击扰动）、RoboTwin 2.0（50 个双臂任务的 clean 与 randomized split）以及 AgileX 双臂机器人平台上的 4 个物理任务。

**📈 对比分析**

与 OpenVLA、π₀.₅、LaMP、Motus、X‑VLA 等基线进行对比，评估指标为平均成功率、任务成功率和 process score。实验显示：在 LIBERO 上 97.0% 的平均成功率（比 π₀.₅ 高 0.1），在 LIBERO‑Plus 上 82.3%（比 LaMP 高 3.0），在 RoboTwin clean 与 randomized 上分别为 80.44% 与 81.48%，在物理任务上平均成功率 67.5%（比无对齐 42.5% 高 25），整体表现显著提升。

**⚠️ 局限性**

需要时间顺序的演示数据；只能利用主摄像头的多帧信息，可能忽略局部接触细节；对追踪器误差敏感；实验仅覆盖单一双臂平台与有限任务，缺乏对更广泛构型与扰动的验证。

---

## 575. Attention is Case-Sensitive

**arXiv ID:** 2608.03711 | [PDF](https://arxiv.org/pdf/2608.03711v1)

**作者:** Maximilian Dillitzer `[一作]` (University of Applied Science Esslingen), Michael Auerbach `[通讯]` (University of Applied Science Esslingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对不同大语言模型和视听模型进行字母大小写变化的实验，系统研究并量化了“字母大小写对内部注意力分布和下游性能的影响”。

**💡 创新点**

首次揭示字母大小写是预训练 Transformer 的潜在属性，能够在不修改模型权重或进行微调的情况下，零样本地通过 prompt 控制注意力，形成新的注意力调节范式。

**🔧 技术方法**

采用无训练、零推理的输入层干预（仅改变字母大小写），结合对模型内部注意力权重的直接可视化和统计分析，构建了“大小写干预”与注意力/性能的因果关系评估框架。

**📊 数据集**

实验数据集包括文本推理/理解基准（MMLU‑Pro、ARC‑Challenge、SQuADv2、XQuAD、HumanEval）以及视觉语言基准（RefCOCOg），覆盖 13 个模型（7 传统 LLM、4 VLM）。

**📈 对比分析**

比较方法：在 13 种模型上统一使用相同大小写模式，测量目标跨度的相对注意力质量和任务准确率。结果显示：全大写/高对比大小写能提升 1–9 个百分点准确率，交替大小写则在所有模型上平均降幅 2–14 个百分点，证明注意力集中与性能提升并不总是正相关。

**⚠️ 局限性**

局限性包括：仅研究 Latin 字母脚本，未探讨非拉丁书写系统；缺乏对大小写对内部表示机理的深入因果中介分析；在 VLM 细粒度视觉注意力方面表现不一致，且未构建自动化注意力调节工具。

---

## 576. Statistical Verification of Quantitative Hyperproperties: Beyond Boolean Quantification

**arXiv ID:** 2608.03694 | [PDF](https://arxiv.org/pdf/2608.03694v1)

**作者:** Amir M. Ahmadian `[一作]` (Chalmers University of Technology and University of Gothenburg), Hazem Torfah `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了量化超属性的逻辑 QHL，并基于统计方法实现了对其的验证

**💡 创新点**

创新点在于用测度（期望、极值等）替代传统的布尔量化器，扩展了超属性的表达范围，并结合 Hoeffding 不等式和极值理论实现了可分层、可容错的统计验证框架

**🔧 技术方法**

核心技术包括：测度基础的量化超逻辑 QHL、Hoeffding 不等式（期望量化器）、极值理论（最大/最小量化器）、基于拒绝抽样的条件抽样、误差与置信度的递归分配、以及实验中使用的 GPD 估计、样本大小自适应控制

**📊 数据集**

实验数据集主要来源于三类安全场景：密码验证系统（基于 Zipf 分布的密码字典）、硬件功耗侧信道（AES 近似模型和噪声注入）、匿名混合网络（随机流量与速率限制），每种场景采用相应的真实或仿真生成的执行轨迹

**📈 对比分析**

与现有的定性超逻辑（如 HyperLTL、HyperPCTL）及量化信息流逻辑（如 Bayes vulnerability、g‑leakage）进行对比，结果显示 QHL 能提供更细粒度的实值评估，虽然在样本量上通常比定性方法多，但误差可在预设的 ε 下保证；在极端量化场景中，极值理论显著降低了所需样本数

**⚠️ 局限性**

主要限制包括：对黑盒系统的依赖导致需要大量拒绝抽样；极值理论要求分布满足极值域吸引性质，对稀有事件的估计仍可能产生高方差；多层量化器的误差/置信度分配在理论上可行但在实际实现中会导致样本量急剧增加；并且 QHL 目前仅支持线性时间逻辑，无法直接表达分支时间或复杂的策略量化

---

## 577. Keep the Needle, Prune the Haystack: Defect-Preserving Token Pruning for Efficient Zero-Shot Anomaly Detection

**arXiv ID:** 2608.03681 | [PDF](https://arxiv.org/pdf/2608.03681v1)

**作者:** Yanning Hou `[一作]` (National University of Defense Technology), Ke Xu `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对零样本视觉异常检测，提出了一种基于Token剪枝的高效框架 KeepAD，能在保持高召回率的前提下显著降低推理成本。

**💡 创新点**

创新点在于：①在浅层采用覆盖保留（coverage‑preserving）与异常救援（anomaly rescue）相结合的路由策略，避免过早丢弃稀疏缺陷；②在深层使用冻结的正常/异常原型与图像自适应预算相结合的异常感知剪枝；③引入稠密到稀疏的自蒸馏辅助训练，并通过最近邻恢复保证像素级定位。

**🔧 技术方法**

核心技术包括：视觉Transformer（CLIP ViT‑L/14）、Token剪枝与图像自适应预算、原型驱动的异常评分、稠密‑稀疏自蒸馏、最近邻恢复与密集定位融合。

**📊 数据集**

在 13 组工业/医学数据集上评估：MVTec‑AD、VisA、BTAD、KSDD2、DAGM、DTD‑Synthetic 以及 OCT17、BrainMRI、Brain AD、HIS、CVC‑ClinicDB、Endo、Kvasir。

**📈 对比分析**

与 8 种主流零样本检测器（WinCLIP、APRIL‑GAN、CLIP‑AD、AnomalyCLIP、AdaCLIP、Bayes‑PFL、UniADet、VisualAD）对比，KeepAD 在保持 94.1/95.2 的平均 I‑AUROC/P‑AUROC 的同时，令 Token 保留率降至 14–18%，实现最高 7.9× 的推理速度提升，平均 AUROC 下降不超过 2.7%。

**⚠️ 局限性**

局限性包括：①对极小或极模糊缺陷的保留仍不确定，若剪枝力度过大可能导致完整缺陷误检；②模型依赖冻结的 CLIP ViT，若迁移至其他架构需重新设计；③在某些医学数据集（如 BrainMRI、Endo、Kvasir）上表现略逊，表明自适应预算与原型匹配仍需改进。

---

## 578. Group Perspective Matters: Regulating Debate Relationships Can Mitigate Blind Conformity in Multi-Agent Debate

**arXiv ID:** 2608.03648 | [PDF](https://arxiv.org/pdf/2608.03648v1)

**作者:** Hao Wu `[一作]` (Beijing Jiaotong University), Kai Lv `[通讯]` (Beijing Jiaotong University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DEAR 框架，通过多阶段（What-Who-How）动态调节多智能体辩论关系，从群体视角减少盲从，提升推理准确率并降低 token 消耗。

**💡 创新点**

创新点在于：①将群体证据（共识与分歧）量化为组证据；②用主体逻辑（Subjective Logic）拆解为咨询倾向与不确定性；③设计两阶段 RL 代理（选择代理与行为代理）联合优化，使用 HAPPO 实现协调学习；④基于 Dempster-Shafer 规则融合证据，避免简单平均导致的错误聚合。

**🔧 技术方法**

核心技术包括：多智能体辩论（MAD）、主体逻辑、Dempster‑Shafer 证据融合、行为与选择 RL 代理、Heterogeneous‑Agent Proximal Policy Optimization (HAPPO)、预训练文本编码器 BGE‑M3、并使用自定义奖励与 POMDP 框架。

**📊 数据集**

使用四类数学推理数据集（GSM8K、AIME24、GSM‑Hard、MATH‑500）以及四类问答数据集（ARC‑C、MMLU_Pro (Health)、TruthfulQA、GPQA Diamond），在开源模型（如 Llama‑2‑70B）与闭源模型（如 GPT‑4）上进行评估。

**📈 对比分析**

与单体推理（CoT、CoT‑SC）以及多智能体辩论基线（Majority Voting、MAD、DMAD、CortexDebate、MAD‑M²）对比，DEAR 在所有任务中平均提升 3–10% 的准确率，同时 token 消耗下降 30–60%。在闭源模型上，DEAR 同样保持优势，尤其在 AIME24 与 GPQA Diamond 等难度任务上显著提升。

**⚠️ 局限性**

局限性包括：①仅在零样本（zero‑shot）场景下验证，缺少微调或迁移学习的探索；②对高维多模态任务和生成性任务的适用性尚未测试；③RL 代理训练复杂度高，收敛稳定性依赖于奖励设计；④仍需进一步分析在极端噪声或极少数正确回答情况下的鲁棒性。

---

## 579. Learning Biomechanically Plausible Human Motion from Sparse Radar Point Clouds

**arXiv ID:** 2608.03637 | [PDF](https://arxiv.org/pdf/2608.03637v1)

**作者:** Jonas Leo Mueller `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Bjoern M. Eskofier `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于单一FMCW毫米波雷达的端到端人体姿态估计框架，利用可微全身骨骼模型从稀疏点云中恢复可解释的生物力学关节角和足部接触信息。

**💡 创新点**

创新点在于首次将完整的受生理约束的骨骼模型与雷达点云结合，利用前向运动学对姿态进行监督，并通过主体特定比例预测实现个性化尺度，同时加入足部接触分类损失保证物理可行的足-地面交互。

**🔧 技术方法**

技术包括点云Transformer编码器、双向LSTM时间建模、骨骼图卷积网络解码、可微前向运动学模块、主观比例多任务Lasso回归以及接触分类损失。

**📊 数据集**

使用公开的mmRadPose数据集，包含11名受试者在三种视角下进行的11种康复运动的雷达与光学运动捕捉同步记录。

**📈 对比分析**

与无约束回归基线mmMesh比较，本文方法在留一受试者交叉验证下平均MPJPE为6.456 cm、MPJAE为8.083°，比mmMesh的7.131 cm、8.115°更优，且骨长一致性为0，足部接触F1达0.935。

**⚠️ 局限性**

局限在样本量仅为11名健康受试者，缺乏临床人群及更大体型多样性；模型目前为离线双向时间模型，需改造为因果模型以实现在线监测。

---

## 580. MDLMPE: Distribution Aware Positional Encoding for Masked Diffusion Language Models

**arXiv ID:** 2608.03769 | [PDF](https://arxiv.org/pdf/2608.03769v1)

**作者:** Tong Ling `[一作]` (University Chinese Academic of Science), Yanlong Du `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种专门为掩码扩散语言模型（MDLM）设计的位置编码方法MDLMPE，能够动态反映未遮蔽与已遮蔽位置的分布；

**💡 创新点**

首次将token可见性分布作为位置信号，采用二值可用性序列、目标中心高斯加权、余弦基投影，并通过轻量级MLP注入到词嵌入和RoPE相位中；

**🔧 技术方法**

使用RoPE、Gaussian局部加权、余弦基投影、双路径（嵌入注入+相位修正）以及轻量级MLP；

**📊 数据集**

在LLaDA与DREAM两大MDLM框架上进行实验，并在OpenWebText、1BW、PTB、Text8、WT103、WT2等多语料库上做预训练和跨语料零样本评估；

**📈 对比分析**

与RoPE、ALiBi和绝对编码对照，采用指令后训练、预训练、零样本和块级扩散等多维度评测，结果在MMLU、HellaSwag、ARC、GSM8K、HumanEval、MBPP等任务上平均提升1–2%，在块级扩散下表现尤为显著；

**⚠️ 局限性**

仅在7B/8B规模MDLM上验证，未对更大模型或不同架构做广泛测试，且对可用性分布建模的最佳形式尚未完全探索，缺乏对更复杂遮蔽策略的评估。

---

## 581. GDPevo: Evaluating Agent Self-Evolution on Real Business Tasks

**arXiv ID:** 2608.03764 | [PDF](https://arxiv.org/pdf/2608.03764v1)

**作者:** Leijun Zhou `[一作]` (PrismShadow), Junhao Hu `[通讯]` (PrismShadow)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于规则混合的演化原生基准和自动化构建管道，用于评估代理在企业GDP相关任务中的自演化能力。

**💡 创新点**

创新点包括①从一开始构建可迁移的训练‑测试关系的规则混合；②完全自动化生成管道能快速更新并防止数据污染；③使用确定性规则评分、成本指标和可视化诊断以提升可重复性与可解释性。

**🔧 技术方法**

采用技能（skill‑based）演化方法，结合无监督、SFT、RL、无答案等四种监督类型，以及多种Harness+Model组合；pipeline 使用示例抽取、规则分解、任务生成与校准；评估采用 Deterministic Rule‑based Grader。

**📊 数据集**

基于公开的GDP相关工作基准（SOP‑Bench、GDPval、JobBench 等）生成 seed scenarios，最终构建 240 个任务，覆盖 CRM、ERP、Finance、Healthcare、Legal、Data‑centric 等 12 个领域。

**📈 对比分析**

通过对四个代理（Codex、Claude Code、GPT‑5.5、DeepSeek‑V4‑Pro‑Preview）和四种监督类型的实验，比较准确率、成本与效率。演化显著提升准确率（最高 16.44pp）并可降低成本；与 oracle 上限 91.6% 相比仍有差距；跨域实验表明 RL‑型监督更易迁移。

**⚠️ 局限性**

限制包括：演化能力仍低于 oracle 上限；规则混合依赖人工定义的 seed scenario；评估仅覆盖规则化的 GDP 任务，未覆盖更复杂的非规则任务；实验仅使用 skill‑based 演化，未探索参数或 prompt 等其他持久状态。

---

## 582. Can LLMs Test Terminal User Interfaces?

**arXiv ID:** 2608.03743 | [PDF](https://arxiv.org/pdf/2608.03743v1)

**作者:** Chao Peng `[一作]` (University of Edinburgh), Cuiyun Gao `[通讯]` (Harbin Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了197个多语言TUI基准，系统评估了LLM驱动与随机探索在终端用户界面自动化测试中的表现。

**💡 创新点**

创新点包括首个跨框架TUI基准、语言无关的覆盖工具（tuicov）、内容感知崩溃判别器以及对LLM与随机方法在等时钟预算下的对比实验。

**🔧 技术方法**

采用的技术有LLM（Claude、Gemini、GPT、DeepSeek）驱动探索、ratatui/Rust、bubbletea/Go、textual/Python、ink/TypeScript四大框架、Docker化的无头运行、伪终端驱动、行/小部件覆盖收集以及基于终端屏幕内容的崩溃识别。

**📊 数据集**

数据集为197个开源TUI应用程序，覆盖四大主流框架和语言，每个实例提供instrumented Docker镜像。

**📈 对比分析**

比较方法：在等时钟预算（600秒）下多次重复实验，测量行覆盖率、widget覆盖率、有效崩溃数及每千步崩溃率；实验结果显示随机探索在每次运行中发现更多崩溃，但LLM在每个交互步骤上更高效，且LLM的最大收益来自启动输入推导。

**⚠️ 局限性**

局限性：行覆盖率与崩溃发现不相关；实验仅覆盖可无头运行的应用，无法处理需网络/权限等外部依赖的TUI；LLM调用延迟导致步数低；内容感知崩溃判别仍需人工审核；未评估非崩溃错误（渲染错误、性能问题等）。

---

## 583. AI-Based Sound Effect Generation: A Narrative Review of Generative Models Across Input Modalities

**arXiv ID:** 2608.03742 | [PDF](https://arxiv.org/pdf/2608.03742v1)

**作者:** Sandy Abdo `[一作]` (Ontario Tech University), Adam Dubrowski `[通讯]` (Ontario Tech University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了过去五年内关于AI生成声效的30篇论文，分析了文本、视觉、音频及多模态输入的模型，并梳理了技术趋势与挑战。

**💡 创新点**

首次将声效生成模型按输入模态分类，系统总结了评估指标，揭示了技术进步与仍存的关键难点，为后续研究提供了框架。

**🔧 技术方法**

采用叙事文献综述方法，使用PRISMA流程、系统检索（Google Scholar、IEEE Xplore、ACM DL）及文献筛选与归纳技术。

**📊 数据集**

使用的主要资源为三大数据库检索到的论文与公开的模型评测数据；并未使用传统音频数据集，而是引用各模型报告的公开数据集信息。

**📈 对比分析**

通过对比各模型的客观与主观指标（FAD、CLAPScore、MOS等），阐明了扩散模型与多模态技术在音效质量、语义一致性和时间同步方面的优势与差距。

**⚠️ 局限性**

局限性包括仅聚焦过去五年英语同行评议论文、排除音乐/语音模型、对某些模型的评测信息不完整，且综述结果受限于已有公开指标，难以完整捕捉人类感知差异。

---

## 584. AgenticECO: An Agentic Framework for ECO on 3D Integrated Circuits

**arXiv ID:** 2608.03738 | [PDF](https://arxiv.org/pdf/2608.03738v1)

**作者:** Shuo Ren `[一作]` (Chinese University of Hong Kong), Tsung-Yi Ho `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于代理的 3D IC 现场更改 (ECO) 框架 AgenticECO，能够在已路由的芯片上自动定位、生成并执行最小扰动的修复操作，并通过独立验证器确保每一次修复都可追溯且不破坏原有设计。

**💡 创新点**

创新点在于：①将可观察证据（如网表、DRC、占用图、时序报告）拆解为读写式专门化代理；②通过约束合同 (contract) 与证据集合驱动的 Advance/Query/Revise 循环，实现对修复程序的动态修订；③使用 EcoRoute 层在不修改原路由器的前提下局部重路由，从而保证每次修复的可归因性；④独立验证器完全与代理隔离，保证修复判定的客观性。

**🔧 技术方法**

技术手段包括：大语言模型代理 (Claude Opus 4.8、GPT‑5.6)，专业化查询适配器、可追踪证据记录、基于占用图和冲突图的候选排序、局部重路由与可恢复复制、以及可验证的签名门（DRC、时序、结构等）。

**📊 数据集**

使用的数据集为三款基于 ASAP7 PDK 的 IC（GCD、I2C、UART），在每款芯片中挑选了三条天然的后路由跨层间距缺陷（共 9 条），并在相同的冻结基线和预算下进行实验。

**📈 对比分析**

在同一 40 次工具调用和 3 次签名预算下，AgenticECO 能够在 9 条缺陷中清除 7 条（相较于全路由 2 条、库存修复 2 条），平均扰动仅为 0.66%，且不触及任何时钟网络；在 GPT‑5.6 版本下进一步提升至 100% 清除率、0.44% 扰动，验证成功率显著高于单代理或传统修复方法。

**⚠️ 局限性**

局限性包括：①依赖于预先设定的合同和可验证门，若需求变化需重新配置；②对复杂的多层交互仍受限于当前的占用图和冲突图表达能力；③实验仅在封闭的单芯片级别进行，尚未验证在更大规模多芯片堆叠或工业级工具链上的可迁移性。

---

## 585. Resume Means Resume: A Machine-Checked Conformance Contract for Checkpoint, Interrupt, and Resume Semantics in Workflow Persistence Layers

**arXiv ID:** 2608.03836 | [PDF](https://arxiv.org/pdf/2608.03836v1)

**作者:** Sajjad Khan `[一作]` `[通讯]`, Sajjad Khan

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在实验平台上通过对 harness 的 mutation 操作（探针 169、170、172）和 probe‑suite 测试（pilot、matrix、live 等）系统地验证了多种错误模型和硬化措施，重新生成并分析了 15 个关键 verdicts。

**💡 创新点**

创新点在于提出了一套可自定位的 harness mutation operators 并结合 load‑bearing verdicts 进行精准评估，同时系统化地构建了大规模 probe‑suite，覆盖跨进程、持久化后端与崩溃实时性等多维度场景。

**🔧 技术方法**

采用了 harness mutation、FD oracle、state‑only oracle、pydantic‑graph 互斥构造、RD interleaving、持久化后端（SQLite、PostgreSQL）以及自动化恢复列（AutoGen）等技术，配合自定义的 kill‑matrix 进行评估。

**📊 数据集**

实验数据主要来源于内部构造的 CrewAI、pydantic‑graph、SQLite、PostgreSQL 等数据集，涵盖了从 pilot 到 crash‑realism、并发性等多层级实验。

**📈 对比分析**

通过比较预测的 kill 结果与实际观察结果，构建 kill‑matrix，并在多场景下量化 mutation 覆盖率与系统鲁棒性；实验表明部分硬化策略能显著提升系统对 mutation 的抵御能力。

**⚠️ 局限性**

限制在于某些 mutation operator 超出当前 driver 范围，部分 operator（如 undercount、crash no‑op）无法产生有效 kill，且实验仍依赖于特定的 probe 设计，未来需扩展 operator 设计与更大规模的真实世界数据集。

---

## 586. History Matters: Meta-policy Delegation with Heterogeneous Multi-agent Reinforcement Learning

**arXiv ID:** 2608.03833 | [PDF](https://arxiv.org/pdf/2608.03833v1)

**作者:** Ziqing Lu `[一作]` (University of Iowa), Weiyu Xu `[通讯]` (University of Iowa)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究异构多智能体系统的任务委托问题，提出基于历史依赖和可转移货币的协作与委托框架，并在GSM8K等任务上使用MARL方法训练委托策略。

**💡 创新点**

创新点包括①将委托建模为多智能体决策问题并设计可扩展的委托结构；②引入历史相关策略以提升信任与互惠；③构建虚拟货币机制实现去中心化激励与委托。

**🔧 技术方法**

使用的技术包括多智能体强化学习（DQN、Q-learning）、历史依赖策略、可转移货币奖励框架、两步纳什值迭代算法以及离线缓存与奖励归因技术。

**📊 数据集**

使用的数据集主要是GSM8K（小学数学问答）和自定义的两步交互模拟环境。

**📈 对比分析**

通过与无委托（always-Ultra）对照，采用返回值、准确率和成本等指标进行比较。在管理约束下，返回提升约3.7点、成本减半；在自由委托下，返回提升约1.9点、成本下降37%；在货币机制实验中实现完全合作，回报高达990。

**⚠️ 局限性**

limitations: 未探讨基于历史的定价机制；求解历史纳什均衡的算法效率低；未研究多维货币定价与策略收敛性等问题。

---

## 587. Geo-Embed: Towards Unified Multimodal Embeddings for Urban Understanding

**arXiv ID:** 2608.03826 | [PDF](https://arxiv.org/pdf/2608.03826v1)

**作者:** Jiapeng Li `[一作]` (Peking University), Yu Liu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了覆盖45个城市级多模态检索、问答、变化检测、分类和定位等任务的统一评测基准GeoMEB，并提出了基于共享视觉-语言骨干、指令驱动的统一嵌入模型Geo-Embed，用指令化对比学习实现跨模态、跨视角、跨时间的查询-目标匹配。

**💡 创新点**

①将多模态城市任务统一为指令条件的排名任务；②设计统一的查询/目标指令格式，使同一内容在不同角色下获得不同嵌入；③在单一嵌入空间内实现跨视角、跨时间、区域定位等多样关系的对比训练；④通过GeoMEB提供大规模训练/评测数据，推动城市嵌入模型的系统化研究。

**🔧 技术方法**

使用视觉-语言多模态Transformer骨干（基于Qwen3-VL-Embedding），指令化输入序列化、LoRA微调、信息对比（InfoNCE）损失、温度缩放、批次负样本采样等技术。

**📊 数据集**

训练集1.32M条样例，评测集286K条查询，覆盖卫星影像、街景图像、无人机图像、文本描述、区域掩模、问答选项等多模态；任务来源包括VRSBench、CityLens、VIGOR、Im2GPS3k、UAV-GeoLoc、UrbanBench等。

**📈 对比分析**

对比CLIP族、指令调优VLM、嵌入专用VLM等10+模型。评价指标为每个任务的Recall@1/5/10（检索）或Recall@1（其余），最终按任务加权平均。Geo-Embed在GeoMEB上取得整体平均25.48分，显著优于最强基线22.09分（提升≈15.3%），在分类、检索、定位等子任务中排名第一，VQA和变化检测次之。

**⚠️ 局限性**

①跨视角、跨时间与区域定位仍是难点，模型在这些任务上性能相对落后；②指令驱动的嵌入转移在不同模态间不一致，导致部分查询-目标对的相似度下降；③单一对比训练对任务间兼容性不一定有利，可能产生干扰；④评测侧重排名，缺少对检索质量、鲁棒性、可解释性的进一步分析。

---

## 588. UHP Detection: LVLMs have their Unique Hallucination Pattern in the Consistency Space

**arXiv ID:** 2608.03817 | [PDF](https://arxiv.org/pdf/2608.03817v1)

**作者:** Amir Mohammad Ezzati `[一作]` (Sharif University of Technology), Mohammad Hossein Rohban `[通讯]` (Sharif University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种完全黑盒的幻觉检测框架UHP Detection，利用图像与文本两种扰动以及命题的肯定与否定构成四个一致性组，提取14个内部与跨组一致性特征，训练轻量级分类器判定模型输出是否为幻觉。

**💡 创新点**

创新点在于将幻觉视为在多维一致性空间中的结构化模式，而非单一不确定性指标；通过交叉扰动与逻辑极性两轴生成互补的一致性组，发现不同组的协同信息对幻觉检测极为关键。

**🔧 技术方法**

采用图像语义保持变换、文本句子同义改写、问题-命题转换、内部与跨组一致性度量（QA、PA、BG）等技术，并用轻量级机器学习模型做最终判别。

**📊 数据集**

在两个视觉幻觉基准上验证：AMBER 与 PhD，并在三种开源大视觉语言模型（InstructBLIP‑7B、InternVL‑4B、Qwen2.5‑VL）上进行实验。

**📈 对比分析**

与多种白盒（AvgProb、MaxEnt 等）和黑盒（BERTScore、Unigram、VL‑Uncertainty 等）基线对比，UHP Detection 在 AUC‑ROC 和 AUC‑PR 上分别提升高达 18.72% 与 20.07% 的绝对值，且在跨数据集迁移时仍保持竞争力。

**⚠️ 局限性**

局限性包括：需要对每个输入执行 30 次 LVLM 推理（3 次句子生成 + 10–12 次模型评估），在计算成本与实时性上仍有挑战；目前仅适用于二元问答任务，缺乏自动化生成幻觉标签来推广至开放式生成任务。

---

## 589. OmniPack: Unified Token Compression for Efficient Omni-modal Large Language Models

**arXiv ID:** 2608.03812 | [PDF](https://arxiv.org/pdf/2608.03812v1)

**作者:** Wanshun Su `[一作]` (Northwestern Polytechnical University), Liang Ding `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 OmniPack，一个无训练的分阶段音频视觉令牌压缩框架，分别在 LLM 之前和内部实现结构压缩与语义精炼。

**💡 创新点**

创新点在于将预 LLM 阶段的模态特定结构压缩（重要性选择、覆盖选择、相似性合并）与后 LLM 阶段的查询条件跨模态协同压缩相结合，实现了在极低令牌预算下的性能‑效率最佳平衡。

**🔧 技术方法**

采用了注意力中心度、时空变异度、覆盖选择算法 (DPC‑KNN)、相似度合并、文本全局查询、音视频协同相关度等技术，以及归一化和权重加权聚合。

**📊 数据集**

在五个音视频理解基准上评估：AVUT、WorldSense、DailyOmni、VideoMME、LVOmniBench，并在 Qwen2.5‑Omni‑3B/7B 与 MiniCPM‑o‑2.6 这三种 Omni‑LLM 后端上测试。

**📈 对比分析**

与 FastV、VisionZip、FastVID、VidCom^2、OmniZip、OmniSIFT、SEATS 等训练自由压缩方法对比，OmniPack 在不同令牌保留比例下均获得最高的性能‑效率折中；例如在 15%/7.5% 保留时保持 95.6% 原始性能，FLOPs 仅 10% 甚至 6.8%。

**⚠️ 局限性**

局限性包括仍需在极低预算下进一步评估跨任务泛化，未使用训练提升的压缩策略，且在某些长视频或复杂场景中可能仍存在信息损失。

---

## 590. Evaluating LLMs in Database Scenarios: A Lifecycle Benchmark for Assessing Their Potential in Core Database Tasks

**arXiv ID:** 2608.03794 | [PDF](https://arxiv.org/pdf/2608.03794v1)

**作者:** Shunfan Zheng `[一作]` (East China Normal University), Gerard de Melo `[通讯]` (Hasso Plattner Institute/University of Potsdam)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了DBLifeBench基准，全面评估大语言模型在数据库生命周期五个阶段（设计、实现、操作、调试、维护）的能力，并提出Progressive-Text2SQL任务；

**💡 创新点**

①首次将数据库生命周期视作评估维度，打破单一Text‑to‑SQL的局限；②提出结构化推理图的Progressive-Text2SQL，模拟人类逐步推理；③揭示专业化Fine‑tune会导致在非SQL任务上的灾难性遗忘；

**🔧 技术方法**

使用多种大语言模型（GPT‑4o、GPT‑4o‑mini、Llama3、Mistral、DeepSeek、Qwen、ChatGLM‑4等）进行推理；采用动态推理图生成与评估；引入多阶段验证、SQLite执行验证、扰动鲁棒性测试；并用多项指标（实体、数据类型、外键精度；表级/字段级准确率；执行准确率；图执行准确率）进行量化评估；

**📊 数据集**

基于13个多域数据库（金融、教育、体育等）的公开/手工构造数据，构成五大子数据集：需求分析（P1）、实现（P2）、标准Text‑to‑SQL（P3）、Progressive-Text‑to‑SQL（P‑Text2SQL）、SQL调试（P4）和维护（P5）；其中P‑Text2SQL采用BIRD及自建的推理图数据；

**📈 对比分析**

对比通用模型与专业化SQL/代码微调模型在五阶段的表现，使用各阶段专属指标；实验结果显示GPT‑4o与GPT‑4o‑mini在大多数阶段表现最佳，专业化模型在设计、实现、维护阶段显著逊色；多轮调试实验表明部分通用模型（如Llama3、Mistral）在多轮修正上优于单轮；总体而言，通用模型更具全栈适应性；

**⚠️ 局限性**

仅适用于LLM，无法涵盖非LLM方法；设计与维护阶段的评价仍具主观性与难度；基准仅针对文本输入，缺乏多模态场景支持；

---

## 591. GORDON: Graph-based Object-centric Rewards for Decomposition of Long-Horizon Manipulation

**arXiv ID:** 2608.03753 | [PDF](https://arxiv.org/pdf/2608.03753v1)

**作者:** Andrea Protopapa `[一作]` (Politecnico di Torino), Giuseppe Averta `[通讯]` (Politecnico di Torino)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过将动作自由的视频演示转换为图结构，学习基于对象和关系的奖励函数，用于强化学习中的长短期抓取任务

**💡 创新点**

提出基于图神经网络的对象中心奖励框架GORDON，使用活动加权池化屏蔽机器人运动并自动发现子任务，显著提升长程任务性能

**🔧 技术方法**

利用图神经网络（GraphTransformer）、自监督时间循环一致性、结构重建损失、活动加权池化、奖励判别器与顺序子策略组合

**📊 数据集**

在MAGICAL的MatchRegions、以及基于RLBench改造的ManiSkill3七个任务（包含四个长程任务）上进行训练和评估

**📈 对比分析**

与手工环境奖励、XIRL、GraphIRL、RoboHorizon等基线比较；在长程任务中平均成功率达74.4%，比最佳学习基线高35.3个百分点，比手工分解奖励高25.4个百分点；在短程任务与环境奖励相当

**⚠️ 局限性**

对机器人检测误差和视觉外观变化的鲁棒性较好，但在极端视觉干扰、机器人姿态极端变化以及极其复杂的多阶段任务（如鞋子外盒放置）仍存在较高失败率，且依赖高质量的对象检测与图结构生成

---

## 592. AP Association for RHS-Enabled Cell-Free Uplink MIMO in Industrial Indoor UAV Networks

**arXiv ID:** 2608.03752 | [PDF](https://arxiv.org/pdf/2608.03752v1)

**作者:** Liangshun Wu `[一作]` (Shanghai Jiao Tong University), Ying Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对工业室内无人机(UAV)网络中单馈RHSEncoded（RHS）接收的空口无小区(cell‑free)系统，设计了一种低复杂度的AP（接入点）关联方案，提出了基于RHSEncoded后观察质量的SINR类评分并使用Top‑J排名实现用户中心的AP分配。

**💡 创新点**

创新点包括：①引入RHSEncoded后有效信号/干扰/噪声比的评分，直接反映共享单RF链下的接收质量；②在弱跨AP干扰相关性假设下，将复杂的log‑det目标分解为可加的每个AP评分，得到解析的Top‑J关联规则；③从几何视角推导了高度-角度最优关系与“邻近AP不一定最优”的理论结论；④阐明扩大服务集群带来的边际收益递减，为实际部署提供可量化的设计准则。

**🔧 技术方法**

技术手段包括：单馈RHS接收模型与幅度控制的全波导理论、基于物理路径损耗与阴影的三维几何LFS模型、近似对角化的干扰协方差、梯度投影优化求解RHS权重、以及CPU端LMMSE联合检测。

**📊 数据集**

使用了仿真数据，构造了40×30×12 m的工业仓库环境，16个AP均布于天花板、24个UAV沿直线恒定高度飞行，采用256个RHS元件、4个天线UAV、路径损耗指数2.2、阴影方差4 dB等参数，实验涵盖多种UAV分布扩散度。

**📈 对比分析**

对比方法包括：仅按距离、仅按大规模衰落、仅按路径损耗×入射角、CAPA、CLSF等传统关联规则，以及基于全干扰协方差的log‑det优化的“1‑swap”和穷举搜索。实验表明：所提评分法在最小/平均谱效率、Jain公平度、能效以及低尾率等指标上均优于所有基线，尤其在UAV聚集场景下提升显著；并且与高复杂度优化相比，性能损失低于15%。

**⚠️ 局限性**

局限性包括：①假设跨AP干扰相关性弱，实际环境中可能不成立；②仅考虑单馈RHS且未探讨多馈或数字前端的改进；③仿真为理想化，未包含硬件失真、相位误差或多时隙动态重选的真实信道估计误差；④方法是贪婪近似，无法保证全局最优，且对极端干扰/阴影环境的鲁棒性待进一步验证。

---

## 593. When Does Disaggregation Pay? Simulating Prefill--Decode--Attention--FFN Specialization for Agentic LLM Inference

**arXiv ID:** 2608.03741 | [PDF](https://arxiv.org/pdf/2608.03741v1)

**作者:** Przemyslaw Forys `[一作]` (Imperial College London), George A. Constantinides `[通讯]` (Imperial College London)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

未提供论文内容，无法确定具体研究工作

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定所用技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定比较方法与性能表现

**⚠️ 局限性**

无法确定研究局限性

---

## 594. MissClick: Exploiting Digit-Serialized Coordinates to Attack GUI Grounding Models

**arXiv ID:** 2608.03740 | [PDF](https://arxiv.org/pdf/2608.03740v1)

**作者:** Yu Ran `[一作]` (National University of Defense Technology), Yi Pan `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对GUI视觉定位模型的白盒对抗攻击MissClick，能在输入图像上加入微小扰动导致模型执行错误点击；

**💡 创新点**

创新点在于将坐标生成的十进制位值结构纳入攻击目标，设计了两种针对性目标：Soft‑Coordinate Displacement（未定向攻击）和Place‑Weighted Target‑Digit CE（定向攻击），显著提升攻击效果；

**🔧 技术方法**

采用PGD梯度优化、教师强制推理获取可微分的token概率分布、软坐标逼近与位值加权交叉熵损失，结合对位值的权重处理；

**📊 数据集**

使用公开的ScreenSpot‑v2数据集，涵盖移动端、桌面端和网页端共1272个GUI定位任务；

**📈 对比分析**

与随机噪声、Token‑CE以及Representation攻击对比，MissClick在OS‑Atlas与UGround模型上，桌面/网页/移动三平台的未定向ASR提升约16–31个百分点，定向ASR提升约30–47个百分点，且在不同扰动预算和迭代次数下均保持领先；

**⚠️ 局限性**

仅评估了两种按位十进制token化的模型，未考虑坐标无关或其他token化方式，且假设白盒访问；未加入数字距离权重且缺乏对抗防御与黑盒迁移性研究。

---

## 595. An Actionable Diagnosis of Multilingual, Multi-Agent Planning Failures

**arXiv ID:** 2608.03735 | [PDF](https://arxiv.org/pdf/2608.03735v1)

**作者:** Vikas Pahuja `[一作]`, Roman Vainshtein `[通讯]` (Fujitsu Research)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文分析了多语言多智能体系统在请求到计划转换过程中的规划失效，并构建了可操作的规划地面失败分类法。

**💡 创新点**

创新点在于提出了五维可操作规划失效分类法，并基于此设计了TART协议，实现了系统层面的语义契约，显著提升跨语言性能。

**🔧 技术方法**

主要技术包括LLM语义解析、TART结构化表示、LLM-as-judge评估以及基于OWL的多智能体框架。

**📊 数据集**

使用的数据集为GAIA-MAPS和MULTITAT，并在其中扩充了多种低资源语言的机器翻译版本。

**📈 对比分析**

通过将TART与基线在三大LLM（GPT-5-mini、Mistral-Large-3、Qwen3-VL-235B-A22B）和多语言任务上对比，TART平均提升了GAIA-MAPS的EM 5.6pp，MULTITAT提升约10pp，效果在不同资源级别均保持一致。

**⚠️ 局限性**

主要局限包括使用机器翻译生成的低资源语言任务、分类法仅基于80例跨语言失败样本、对LLM judge的依赖以及部分模型仅单次评估。

---

## 596. We Must Have Missed This Comment: Detecting and Repairing Stale Function References in Linux Kernel Comments

**arXiv ID:** 2608.03734 | [PDF](https://arxiv.org/pdf/2608.03734v1)

**作者:** Kexin Sun `[一作]` (Nanjing University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Linux 内核代码中自动识别并修复因函数重构或删除而导致的过时函数引用（stale function reference）

**💡 创新点**

将静态分析与大语言模型（LLM）结合，形成三阶段流程：1）检测未解析的函数符号；2）利用 Git 历史和 LLM 追踪函数演化；3）基于演化历史生成修复建议，显著提升了修复的准确性和可操作性

**🔧 技术方法**

1）Coccinelle 语义匹配 + grep；2）LLM（DeepSeek‑V3.2）进行符号定位、演化分析和修复生成；3）Tree‑sitter、Universal Ctags、GNU Global 用作对比；4）Git 仓库历史遍历

**📊 数据集**

Linux 内核 v6.18‑rc1（约 62,000 源文件、83,241 条注释、49,302 个唯一函数符号，8,258 条未解析提及）

**📈 对比分析**

与 Coccinelle、Tree‑sitter、Ctags、GNU Global 等工具在符号解析上的对比；与 RefDiff 在演化追踪上的对比；与 C4RLLaMA 在整体检测上的对比。结果显示：
• 识别 869 条 stale references；
• 89% 的修复建议有用（其中 42.5% 可直接应用）；
• 75 条提交中 50 条已被接受；
• RQ4 中平衡样本的 F1 达 0.937，准确率 94%。

**⚠️ 局限性**

仅针对 C 语言的 Linux 内核，无法处理宏生成的函数、前置预处理代码、非函数实体（如编译器内置、结构体字段）以及前 Git 时代的历史；LLM 依赖提示和模型质量；对多步重构的识别仍有误差；需人工干预才能完成全部修复。

---

## 597. Power Minimization under Quality of Service Constraints for MIMO Systems with a RIS-based Transmitter

**arXiv ID:** 2608.03829 | [PDF](https://arxiv.org/pdf/2608.03829v1)

**作者:** Erico S. P. Lopes `[一作]` (National Institute of Industrial Property of Brazil), Amine Mezghani `[通讯]` (University of Manitoba)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了基于可重构智能表面（RIS）传输的多用户MIMO系统，在离散相移RIS模型下求解在符号误差概率（SEP）或其上界（UBSEP）约束下的功率最小化问题，提出了部分分支定界（PBB）算法以及高分辨率RIS的逼近方法，并对不完美CSI进行了鲁棒优化

**💡 创新点**

创新点在于：①针对QPSK和一般M-PSK分别引入精确或上界SEP约束；②提出可接受子最优解的PBB算法，实现功率预算与复杂度的权衡；③利用连续相位逼近将离散优化映射到斜交流形上，采用二分法解决；④在不完美CSI下给出鲁棒上界公式

**🔧 技术方法**

技术包括：离散相位RIS模型、符号误差概率分析、上界SEP、分支定界优化、凸松弛、均衡流形（oblique manifold）优化、Riemannian共轭梯度、二分搜索、鲁棒优化中的椭圆误差模型

**📊 数据集**

采用随机Rayleigh衰落信道仿真，考虑多用户数量、RIS元素数、相位分辨率、功率预算与SEP阈值的不同场景；对不完美CSI使用统计误差模型（σ_h）进行仿真

**📈 对比分析**

与现有基于MDDT、PHMMDDT等方法比较，PBB在大部分设置下实现了至少50-90%复杂度下降，功率损失不足1-2 dB；在高分辨率RIS下，逼近方法进一步降低复杂度至O(N^2.5)并保持低ANTP；鲁棒版本相比非鲁棒版本在CSI误差下可保持SEP满足，非鲁棒方法性能显著下降

**⚠️ 局限性**

局限性包括：①PBB在最坏情况下仍可能退化为完整分支定界，复杂度退回O(N^3.5+N^3√K)；②对离散相位的上界约束可能导致性能损失，特别是低分辨率RIS；③鲁棒模型假设误差满足椭圆约束，实际环境误差分布可能更复杂；④算法实现依赖于凸松弛与二分法收敛性，需调参以保证数值稳定

---

## 598. Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss

**arXiv ID:** 2608.03796 | [PDF](https://arxiv.org/pdf/2608.03796v1)

**作者:** Bakbergen Ryskulov `[一作]` (Multiverse Computing), Román Orús `[通讯]` (Multiverse Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对小型LLM进行知识蒸馏，以在严格的延迟、成本和本地部署约束下实现长上下文恢复。

**💡 创新点**

提出离线Top‑K教师缓存蒸馏和融合分块KL损失两大系统改进，实现近乎相同的蒸馏质量但显著降低内存与提升吞吐量。

**🔧 技术方法**

采用离线Top‑K logits缓存、融合分块KL损失（避免全词表logits张量）、Megatron-Bridge+ModelOpt框架、GPU加速等技术。

**📊 数据集**

以Llama 3.1 8B Instruct为教师，3.2B学生，使用SmolTalk、MMLU、GSM8K、BoolQ等评测数据集。

**📈 对比分析**

与在线蒸馏和全密集KL做对比，离线蒸馏降低29%迭代时间、提升41%吞吐量，融合分块KL允许单卡训练32,768 tokens，显著降低内存并在长序列上速度优势明显。

**⚠️ 局限性**

局限在单一教师-学生组合、仅验证H200+Megatron-Bridge环境，对不同硬件、模型体系结构及极长序列的泛化尚未评估。

---

## 599. Empirical behavioural heterogeneity shapes the dynamics of an agent-based land use model

**arXiv ID:** 2608.03784 | [PDF](https://arxiv.org/pdf/2608.03784v1)

**作者:** Ronja Hotz `[一作]` (Karlsruhe Institute of Technology), Mark Rounsevell `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在大规模代理基础土地使用模型中将欧洲林业从业者的行为类型（基于问卷调查的五类）嵌入到认知决策层，以研究行为异质性对土地利用动态和生态系统服务供给的影响。

**💡 创新点**

首次将经验性的行为学分型与理论驱动的认知决策模块结合，在代理模型中实现行为异质性与土地管理实践的分离，揭示个体行为与集体结果的共进关系。

**🔧 技术方法**

采用 agent‑based land‑use 模型框架 CRAFTY 与 TPB/TRA 理论驱动的行为层，使用 NetLogo 进行数值仿真，并通过行为参数化和空间网格化实现。

**📊 数据集**

基于十三个欧洲国家的大规模社会调查数据（包含环境态度、社会规范等指标），构建五种林业从业者类型，并将其地区比例映射至北欧与西南欧两种行为组合。

**📈 对比分析**

将三类场景（北欧、西南欧、理性选择基准）在相同格网与生态系统服务需求情景下进行多次模拟，比较土地管理强度比例、生态系统服务供给与系统振荡；结果显示异质群体显著降低同步性、提升中等强度管理并在部分情景下更好匹配需求，优于纯理性选择。

**⚠️ 局限性**

行为参数化依赖于解释性映射，缺乏直接计量；空间上将行为类型随机分配，忽略真实的社会空间聚集；模型简化了景观、服务类别和政策影响，限制了对真实地区的预测能力。

---

## 600. KnowHal: A Knowledge-Driven Benchmark for Comprehensive Multimodal Hallucination Evaluation

**arXiv ID:** 2608.03782 | [PDF](https://arxiv.org/pdf/2608.03782v1)

**作者:** Ruihan Li `[一作]` (Shandong University), Yuntao Du `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

我们提出了KnowHal基准，用于在实体、属性、关系和知识四个维度上对多模态大语言模型的幻觉进行系统评估。

**💡 创新点**

创新点在于将感知层和知识层幻觉统一到同一基准，并通过正负对齐问题设计使模型同时评估视觉理解和误导鲁棒性。

**🔧 技术方法**

采用LLM辅助生成、CLIP过滤、人工验证以及自动评判（GPT‑4o‑mini）等技术构建数据，并用多模态LLM进行零样本推理。

**📊 数据集**

数据集包括1,800个实体–图像样本，覆盖10个领域、50个类别，共计14,400个问答对，包含正负两类。

**📈 对比分析**

对14个代表性MLLM在正负问答上进行零样本评测，整体准确率最高为约70%，知识维度最低，负样本准确率普遍低于正样本，说明模型易受误导。

**⚠️ 局限性**

局限在于知识幻觉仍未完全解决，基准仅覆盖公开可验证事实，缺乏对动态知识或更复杂场景的评估，且自动评判偶尔与人工判断不一致。

---

## 601. AgenticVAU: Multi-Agent Explore-Verify Reasoning for Video Anomaly Understanding

**arXiv ID:** 2608.03779 | [PDF](https://arxiv.org/pdf/2608.03779v1)

**作者:** Yuxiang Duan `[一作]` (Shandong University), Yuntao Du `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一个训练无关的多代理框架 AgenticVAU，采用探索-验证过程主动收集并验证视频异常证据。

**💡 创新点**

将视频异常理解拆分为规则构建、搜索规划、视频观察和最终决策四个专用代理，并通过锚点注册表管理支持/反对/未定证据，实现结构化的证据协同与跨时段比较。

**🔧 技术方法**

基于大型语言模型（DeepSeek‑V4‑Pro / Qwen2.5‑VL‑3B）进行规则生成、搜索规划和摘要，并使用多种观察工具（Global Scan、Segment Focus、Temporal Stitch）进行分层视频采样和对比。

**📊 数据集**

在 VAU‑Bench 的 ECVA、UCF‑Crime 与 MSAD 三个子数据集上进行实验。

**📈 对比分析**

与零样本 Qwen2.5‑VL‑3B 及强化学习微调的 VAU‑R1 进行对比，AgenticVAU 在多选问答、异常分类、异常推理与时间定位等任务均取得显著提升，二分类准确率提升至约 75–95%，多分类准确率提升至约 49–86%，总分得分提升至 30–35 以上。

**⚠️ 局限性**

主要瓶颈在视觉证据提取与锚点注册更新，观察代理错误占比高；推理过程仍依赖大型模型且推理成本较高。

---

## 602. Computing Actual Causes for Neural Network Predictions under Structured Causal Inputs

**arXiv ID:** 2608.03772 | [PDF](https://arxiv.org/pdf/2608.03772v1)

**作者:** Jannick Strobel `[一作]` (University of Konstanz), Stefan Leue `[通讯]` (University of Konstanz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在结构化因果输入空间下，研究并实现了一种能够对神经网络预测给出完整最小 Halpern–Pearl 实际因果（actual cause）的解释方法；

**💡 创新点**

创新点在于：①将结构因果模型（SCM）与神经网络分离，利用 SCM 明确干预语义；②通过多线性松弛与区间传播（IBP）结合分支定界（branch‑and‑bound）算法，能够在不枚举所有可能干预组合的前提下，返回所有最小实际因果及其最小后置集合，并提供完整性与最小性保证；

**🔧 技术方法**

使用的技术包括：Boolean SCM、Halpern–Pearl 实际因果定义、结构方程多线性松弛、区间传播（interval bound propagation, IBP）、分支定界搜索、剪枝与递归细分；

**📊 数据集**

实验数据集：合成的 Barabási–Albert 与 Erdős–Rényi DAG 生成的 SCM 与随机初始化的多层感知机（teacher‑student）构成的 SC‑NN 基准；以及真实的 SNAP（Supplemental Nutrition Assistance Program）质量控制案例；

**📈 对比分析**

与暴力枚举、基于搜索的启发式方法、以及基于 ILP 的约束求解方法进行比较；在节点数≤28时，该算法在 180s 预算内完成所有实例，平均耗时仅数秒；当节点数≥30时，基线往往超时，而本方法仍能在多数实例中完成；在 SNAP 案例中平均耗时 4.6 秒，对比基线 43 秒，且识别的因果数更少且更准确；

**⚠️ 局限性**

局限性：仅适用于离散 Boolean SCM，无法直接处理连续输入；搜索复杂度理论上为 O(3^n)，对大规模图仍具有挑战；算法对最大因果大小 k_max 的设定敏感；需要先验 SCM，若 SCM 与真实因果结构不匹配，解释可能不准确。

---

## 603. Linked Barcode for Persistence Induced by Filtrations

**arXiv ID:** 2608.03765 | [PDF](https://arxiv.org/pdf/2608.03765v1)

**作者:** Tamal K. Dey `[一作]`, Tao Hou `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种新型的持久性条形码——链接条形码（link barcode），通过跟踪链与循环在过滤中的演化来补充传统持久性条形码；

**💡 创新点**

创新点在于定义了 p‑link 模块及其高阶链接条形码，并通过参考过滤实现其稳定性，同时设计了 O(n^3) 的通用算法以及针对图过滤的 O(n log n) 快速算法；

**🔧 技术方法**

技术手段包括持久性同调、链/循环/边界模块、模块呈现与矩阵归约、链接-切割树、图算法（最小生成树、并查集）、持久性图像等；

**📊 数据集**

使用了多组实验数据：合成的正则图与强正则图族（如 (9,4,3)、(13,4,4) 等），以及多种真实时间网络数据集（email‑Eu、HighSchool、UCI‑msg、Enron core200、fb‑forum、CollegeMsg）；

**📈 对比分析**

通过瓶颈距离对传统持久性条形码和链接条形码进行比较；实验显示链接条形码在图同构判别上显著优于传统条形码，在时间网络链接预测任务中加入链接持久性特征后 AUC 平均提升约 2–3%，且均通过配对 t 检验得到显著性（p<0.05）；

**⚠️ 局限性**

局限性包括：链接条形码的稳定性需依赖参考过滤；全阶高阶链接条形码计算复杂度为 O(n^4)；仅在图过滤上实现了 O(n log n) 速度，对高维复杂体的应用仍待研究；高阶链接条形码的直观解释性仍有限。

---

## 604. Agents Catching Agents: Shortcut Cascades and Benchmark Gaming in Clinical Multi-Agent Systems

**arXiv ID:** 2608.03744 | [PDF](https://arxiv.org/pdf/2608.03744v1)

**作者:** Sebastián Andrés Cajas Ordóñez `[一作]`, Leo Anthony Celi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在临床多智能体系统中，如何通过“benchmaxxing”或捷径学习使模型在满足基准测试的形式要求却偏离预期目标，并探讨了多智能体共享工作空间导致错误共识的机制；

**💡 创新点**

创新点在于提出并实现了“referee”监督代理，该代理通过私下重新查询持有者模型并对比共享阶段的答案，从而检测和阻断因同伴推理导致的错误共识；同时验证了隐藏评分规则对模型行为的无声影响；

**🔧 技术方法**

主要技术包括：基于Gemini 2.5 Flash/Flash‑Lite的大语言模型推理、cue敏感性实验、对多智能体共享工作板的设计、referee与gate与judge三种监督代理的对比评估，以及使用精确McNemar、Fisher检验和bootstrap置信区间进行统计检验；

**📊 数据集**

使用的公开数据集包括：MedQA‑USMLE、MedMCQA、MIMIC‑CXR报告（转换为多选题）、NIH ChestX‑ray14、MIMIC‑CXR‑JPG、CheXpert、以及支持医疗记录数据集SUPPORT2；

**📈 对比分析**

比较方法主要是对单智能体的cue flip率、对多智能体的“共识传播”感染率以及监督代理的精确度、召回率和假阳性率进行对比。结果显示单智能体对cue几乎不敏感，但在两位同伴一致错误回答的情况下感染率可达≈38–60%；referee相较于gate和judge，精确度在0.68–0.88之间、召回率几乎为1、假阳性率≤0.21；隐藏评分规则能显著提升错误答案的采纳率，但几乎不被模型自报；

**⚠️ 局限性**

局限性包括：实验仅使用Gemini模型，未验证其他LLM的泛化；数据集主要为公开文本与影像，未涵盖更复杂多模态或实时数据；referee需要额外的私下查询，增加计算成本；隐藏评分规则的影响可能因任务而异，无法完全捕捉所有非显式捷径；最后，系统在面对多样化错误类型时的检测效果仍待进一步研究。

---

## 605. CARE-Bench: Benchmarking Patient-Facing LLM Triage

**arXiv ID:** 2608.03731 | [PDF](https://arxiv.org/pdf/2608.03731v1)

**作者:** Yining Hua `[一作]` (Harvard University), Cyrus Ayubcha `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建 CARE-Bench 基准，用以评估患者面对的医疗大语言模型（LLM）在每轮对话中给出的四类当前行动（澄清、自护/监测、专业护理、急诊）。

**💡 创新点**

创新点在于：①将真实医疗对话重构为可评估的前缀序列并标注行动；②采用固定 mapper 对开放式回复进行四标签编码；③在无提示和最小提示两种协议下对模型的阈值错误进行细粒度评估。

**🔧 技术方法**

技术方法包括：GPT‑5.5 辅助案例构建与人工复核、结构化 prompt 的 mapper 对响应进行自动编码、宏 F1、阈值误差等指标的统计与比较。

**📊 数据集**

数据集来源于 MedDialog/OpenMed、ChatDoctor、PriMock57 等公共医疗对话与咨询文本，重构后得到 500 个案例、1059 轮，包含 1,059 条评估前缀。

**📈 对比分析**

比较方法：对 11 款 LLM（闭源、开源、医疗专用）在未提示和最小提示两种协议下评估宏 F1、准确率及阈值错误。宏 F1 范围为 31.2–50.4（未提示）和 46.9–63.4（提示），提示虽提升模型倾向于给出行动建议，但仍存在显著的误阈值问题。

**⚠️ 局限性**

局限性：①重构对话不一定代表真实用户表达，缺乏自然语言多样性；② mapper 的边界判定与 300 token 上限可能影响结果；③ benchmark 只评估对话层面的行动沟通，无法证明临床安全或系统级安全，可用于研究而非直接临床部署。

---

## 606. FlowForm: Synergizing Fluid Physics with Topological Consistency for Satellite Flood Synthesis

**arXiv ID:** 2608.03822 | [PDF](https://arxiv.org/pdf/2608.03822v1)

**作者:** Zhang Weihui `[一作]` (Zhejiang University), Song Mingli `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为FlowForm的卫星洪水图像合成框架，能够在无时间序列的情况下从预灾图像生成后灾图像。

**💡 创新点**

创新点在于将浅水方程（SWE）残差作为潜在层正则化，结合结构感知的 Terrain Anchor Adapter（TAA）以保持景观不变性，并构建了大规模 FloodScape 数据集。

**🔧 技术方法**

采用了基于 Stable Diffusion 2.1 的潜在扩散模型，集成了 FDM（潜在深度代理与流场预测）和 TAA（多模态深度、语义、边缘注入）以及差分算子计算 SWE 残差。

**📊 数据集**

使用了约10,000对高分辨率预/后灾卫星图像的 FloodScape 数据集，数据来源于 Maxar、xBD、eBD 等公开资料。

**📈 对比分析**

与多种 GAN 与扩散基准（CycleGAN、CUT、Pix2Pix‑Zero、DiffusionSat 等）对比，FlowForm 在 FID、SSIM、LPIPS、CLIP、PSNR、IoU 与 FVPS 等指标上均取得了最高或第二高分，并在南非独立测试集上实现零样本泛化。

**⚠️ 局限性**

局限性在于模型仅处理静态二维映射，未建模洪水随时间演化的连续过程，未来可考虑序列或三维建模。

---

## 607. Designing Social Robots for Inclusive Child Wellbeing Assessment: Insights from Communities Supporting Developmental Language Disorder and Forced Migration

**arXiv ID:** 2608.03820 | [PDF](https://arxiv.org/pdf/2608.03820v1)

**作者:** Fethiye Irmak Dogan `[一作]` (University of Cambridge), Jenny L. Gibson `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过与支持发展性语言障碍（DLD）儿童和有强迫迁移背景儿童的父母及专业人员进行焦点小组，设计并评估了多模态的机器人交互活动，用以支持儿童福祉评估；

**💡 创新点**

创新点在于将社会机器人评估方法从单纯的可行性验证转向系统性探讨包容性与伦理设计，提出了四大主题（机器人角色与能力、交互细节、个体差异化设计、儿童主体性培养）并给出针对不同群体的具体设计推荐；

**🔧 技术方法**

技术方法主要为人本参与式设计与定性主题分析；机器人交互活动包括“了解彼此”“结构化故事”“手势游戏”“情绪表达”等多模态任务；

**📊 数据集**

数据集：使用焦点小组访谈记录，未收集大规模量化数据；

**📈 对比分析**

由于研究采用的是定性方法，未进行量化性能对比；研究以访谈反馈为依据，展示了设计建议的可行性与适用性；

**⚠️ 局限性**

局限性包括样本量有限（仅两组共14名参与者）、只聚焦DLD与强迫迁移两类群体，缺乏跨文化验证，且研究结果主要基于专家与父母观点，未在真实评估情境中进行实证验证；

---

## 608. VIBE: A VAD-Informed Benchmark for Entity-Centered Affective Profiling of Large Language Model Outputs

**arXiv ID:** 2608.03810 | [PDF](https://arxiv.org/pdf/2608.03810v1)

**作者:** Andrei Chetvergov `[一作]` (Russian New Economic School), Sergey Bolovtsov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于VAD空间的实体中心情感剖析基准（VIBE），通过测量合同将生成与评分分离，生成针对社会重要目标的情感属性报告。

**💡 创新点**

创新点包括：① 统一的测量合同（对象、评分接口、提示协议、报告边界）；② 将情感分量拆为Valence、Arousal、Dominance三轴并对目标指向性进行细粒度评分；③ 对测量不确定性、协议漂移与评分者身份做系统记录，鼓励发布完整的“护照”式报告。

**🔧 技术方法**

技术主要采用：LLM生成模型（6种指令调优模型），LLM判定器（Qwen3.6、Gemma-4、GPT‑4o‑mini）进行VAD评分，统计与相关性分析（Pearson、Euclidean距离、方差分解）以及交叉验证与人类评估对齐。

**📊 数据集**

数据集：从Wikidata构建的2,613个社会重要实体（13个类别），对六种LLM在多种提示条件下生成文本并评分，外部扩展为129,181行（Buyl复制）和342,779行（协议漂移实验）。

**📈 对比分析**

比较方法：将标量偏好与三轴VAD对比（H1），对比完整文本评分与目标指向评分（H2），以及在不同协议条件下的漂移度量（H3）。结果表明：标量偏好只能捕获Valence，无法覆盖Arousal/Dominance；目标指向与整体语调存在显著差距（平均欧氏距离≈0.24，Dominance最大差异≈0.17）；情境框架导致的漂移远大于模型差异（η²_family≈0.14 vs η²_model≈0.01）。

**⚠️ 局限性**

限制：评分依赖LLM判定器，可能存在评审者偏差；对人类验证仅覆盖标量偏好和目标指向评分，缺乏对整体语调的直接人类校准；实体库并非完全覆盖；多语言和更细粒度的协议空间尚未完全探索。

---

## 609. Does Forgetting Transfer Across Modalities? A Real-World Benchmark for Cross-Modal Knowledge Unlearning Evaluation

**arXiv ID:** 2608.03791 | [PDF](https://arxiv.org/pdf/2608.03791v1)

**作者:** Chunlin Liu `[一作]` (Shenzhen University), Yuntao Du `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个真实世界的 Vision‑Language 模型跨模态知识遗忘基准 UNLINK‑VL，并在其上评估多种机器遗忘算法。

**💡 创新点**

创新点包括：①在跨模态条件下单独控制遗忘与评估的模态；②利用 Wikidata 的一跳和两跳事实构建 Forget、Multi‑Hop、Rewrite、Retain 四个互补子集；③揭示多模态遗忘向单模态转移的显著不对称性。

**🔧 技术方法**

采用 GA、DPO、NPO、RT 四种遗忘技术，并使用 LoRA 与全参数微调两种训练策略；利用 Cosine 学习率调度、LoRA rank=32、scale=64 等技术实现高效微调。

**📊 数据集**

使用的主要数据集包括：
- 250 个可视化实体及其图像（来自公开图像搜索），
- 对应 Wikidata 的一跳事实与两跳路径，
- GPT‑4o 生成的 QA 与同义重写句子；
- 为每个模型生成的专属遗忘训练集。

**📈 对比分析**

对 Qwen3‑VL‑8B/32B 与 Llama‑3.2‑Vision‑11B 进行实验，比较四种遗忘方法在 T→T、T→M、M→T、M→M 四种设置下的 Forget、Retain、Multi‑Hop 与 Rewrite 准确率。结果显示：NPO 在 Forget 上最强；DPO 在 Retain 上最稳；RT 在两者间取得最佳平衡；多模态遗忘对单模态评估的转移效果优于相反方向，表明单模态评估易高估遗忘效果。

**⚠️ 局限性**

局限性：
- 仍无法完全阻止通过两跳推理或同义查询恢复被遗忘知识；
- 评估范围仅限于 VLM，其他多模态模型尚未验证；
- 数据构造与人工校验成本较高，可能存在选择偏差；
- 未系统探究不同模型结构对跨模态遗忘一致性的影响。

---

## 610. DiffPower: GPU-Accelerated Differentiable Switching Power Analysis and Optimization

**arXiv ID:** 2608.03778 | [PDF](https://arxiv.org/pdf/2608.03778v1)

**作者:** Isaac Jacobson `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个GPU加速、可微分的开关功耗分析框架 DiffPower，并利用其得到的功率梯度实现了高效的单元尺寸优化和功率病毒生成。

**💡 创新点**

创新点包括：① PDK无关的字节码表示，允许任意门库的前向与反向传播无需编写门特定代码；② 混合传播方法，将解析的静态概率模型与位打包仿真得到的空间相关性融合；③ 基于梯度的全局敏感度信息，首次在单元尺寸优化和功率病毒搜索中实现全局优化。

**🔧 技术方法**

技术手段包括：GPU级并行图引擎、反向模式自动微分、字节码堆栈机解释器、位打包仿真、级联级化的图结构、混合 TR 缩放 β、基于梯度上升的连续松弛搜索。

**📊 数据集**

使用三份专有工业设计（13K、117K、652K单元）和七份 IWLS 2005 开源基准，共十个设计。

**📈 对比分析**

与单线程CPU传播和商用向量仿真比较，DiffPower 在 652K 单元设计上单前向/后向传播耗时 <60 ms，速度提升约 1000×；与有限差分梯度比较，速度提升 904×；TR 相关性 r≥0.96；单元尺寸优化比局部功耗启发式提升 2.98×，功率病毒生成比进化搜索提升 2.13×，时间从数小时降至数十秒。

**⚠️ 局限性**

局限性：仅分析顺序单元之间的组合逻辑，忽略时序功耗、时钟网络与多电压域；基础设施网（时钟缓冲、测试线）仅作为固定边界处理；对深度时序/高度重合逻辑的精度仍有限。

---

## 611. TDVR: Joint Text Disambiguation and Viewpoint Reasoning for Zero-Shot 3D Visual Grounding

**arXiv ID:** 2608.03763 | [PDF](https://arxiv.org/pdf/2608.03763v1)

**作者:** Qingxi Du `[一作]` (Northwestern Polytechnical University), Yining Zhu `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的 3D 视觉定位框架 TDVR，用来在零样本情境下通过文本描述定位 3D 点云中的目标物体。

**💡 创新点**

核心创新在于：① 观测者中心协同消歧模块，利用 LLM 对原始查询进行语义增强；② 构建多属性 3D 场景图并用链式推理提取结构化语义；③ 视角感知方向推理和视角基相似度解耦推理双重机制，用以推断最佳视角并区分相似实例。

**🔧 技术方法**

使用 LLM（GPT‑4o、DeepSeek‑V3）、Mask3D 目标检测、CLIP 视觉文本嵌入、BERT 词向量、链式思维（Chain‑of‑Thought）以及多维场景图推理算法。

**📊 数据集**

在 ScanRefer 与 Sr3D 两大公开数据集上进行评测，ScanRefer 通过 64.06%（Acc@0.5）和 70.85%（Acc@0.25）领先于此前所有零样本方法，Sr3D 上实现 70% 的整体准确率。

**📈 对比分析**

与多种基准（SPAZER、SeeGround、VLM‑Grounder 等）对比，TDVR 在 Acc@0.5 方面提升 15.26%/18.4%（相对 SPAZER/所有零样本基准），并在某些指标上甚至逼近甚至超越部分全监督方法，速度平均 10.21 秒/查询，优于 SPAZER（23.5 秒）和 VLM‑Grounder（50.3 秒）。

**⚠️ 局限性**

局限性包括：① 依赖冻结的 Mask3D 检测，检测精度限制了整体召回；② 对 LLM 的推理成本和调用时间较高；③ 视角离散化（10° 步长）在某些细粒度定位场景下仍可能产生误差；④ 仅在室内点云环境验证，需进一步验证跨场景鲁棒性。

---

## 612. Failure-Informed Image Self-Augmentation for Multimodal Large Language Model Self-Improvement

**arXiv ID:** 2608.03733 | [PDF](https://arxiv.org/pdf/2608.03733v1)

**作者:** Chunyang Jiang `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于模型自身失败案例的图像自增框架 FISA，用于提升多模态大型语言模型的视觉理解能力。

**💡 创新点**

创新点在于通过三步推理模板生成既能增加视觉难度又能保持答案信息的图像变体，并采用双重真实性（Utility + 文本/图像 Fidelity）过滤，首次将失败信息驱动的图像增强与文本增强结合。

**🔧 技术方法**

使用了 LLM 的文本生成、文本到图像生成（Z‑Image‑Turbo）、自我检测（Utility、文本 Fidelity、图像 Fidelity 过滤）、rollout 策略与自解释训练等技术。

**📊 数据集**

种子数据集为 SEEDBench 和 A‑OKVQA，评估基准包括 SEEDBench、A‑OKVQA、MME、MMBench、MMStar 与 AI2D 六个视觉问答数据集。

**📈 对比分析**

与仅种子训练或仅文本自增训练对比，FISA 在多模型（Qwen3‑VL‑2B、Gemma3‑4B、LLaVA1.5‑7B）上均实现了 2%–5% 的平均精度提升，且在 OOD 任务中保持了显著改进。

**⚠️ 局限性**

局限性包括过滤阈值对结果的敏感性、不同模型下判定一致性差、生成图像依赖文本描述的精细程度、缺乏客观生成质量评估以及较高的计算成本。

---

## 613. LatentGuard: Efficient and Inspectable Latent Reasoning for LLM Safeguards

**arXiv ID:** 2608.03838 | [PDF](https://arxiv.org/pdf/2608.03838v1)

**作者:** Zhinan Liu `[一作]` (Xiamen University), Jiayi Ji `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LatentGuard，一个通过阶段化的隐式推理将安全判定与生成解释分离的LLM安全防护框架。

**💡 创新点**

创新点在于将连续的隐式推理压缩为紧凑状态，并在需要时通过独立解码器生成可审计的压缩解释，兼顾效率与可审计性。

**🔧 技术方法**

使用了阶段化的隐式推理课程、COCONUT式连续推理、可选审计解码器、stop‑gradient投影以及自适应隐式预算等技术。

**📊 数据集**

在 GuardReasonerTrain（由 WildGuardTrain、AegisTrain、BeaverTailsTrain、ToxicChatTrain 合成）数据集上进行训练，并在 GuardReasoner 评测套件进行评估。

**📈 对比分析**

与 GuardReasoner、LlamaGuard 等基准模型对比，LatentGuard‑8B 在三任务平均加权 F1 上提升了约0.96点，显著降低了推理成本（从 268.56 CoT token 降至 1.60 隐式 token）并将推理时延降至 0.089s。

**⚠️ 局限性**

局限性包括审计模式下仍需额外计算、生成的审计摘要不具备完整可解释性，以及目前仅针对文本安全判定，尚未扩展到多模态、长对话或动态安全策略。

---

## 614. A Single-Exponential FPT Algorithm for 2-Vertex-Connectivity Augmentation

**arXiv ID:** 2608.03830 | [PDF](https://arxiv.org/pdf/2608.03830v1)

**作者:** Tomohiro Koana `[一作]` (University of Tokyo), Soh Kumabe `[通讯]` (CyberAgent)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于路径森林压缩、动态规划与伪耳打包的算法，求解2-连通生成子图的边界对成本问题，从而实现2-连通网络补全的参数化算法；

**💡 创新点**

创新点在于给出了一种将2-连通子图问题转化为伪耳打包多项式求值的精细化方法，利用Cut–Möbius反演消除集合划分约束，并通过一次性求解所有子宇宙的oracle实现了更高效的子集动态规划；

**🔧 技术方法**

采用的技术包括：子集动态规划、区间覆盖表、最小加法子集卷积、伪耳打包的构造、Cut–Möbius变换、一次性所有子宇宙求解与多项式插值；

**📊 数据集**

无具体实验数据集，所有实验均为理论复杂度分析；

**📈 对比分析**

与已有的O*(3^n)和O*(2^n)算法相比，该方法在p路径数目≤p的前提下实现了O*((ρ+ν)^pW)的时间复杂度，其中ρ、ν为2VCSS-BP算法的常数，理论上比传统方法更快；

**⚠️ 局限性**

局限性在于算法仍属于指数级复杂度，对最大链成本W的依赖较大；且算法仅适用于无自环多重图，且需要预先构造完整的伪耳打包集合，实际实现较为复杂。

---

## 615. UNVaMP: Neural Knowledge Tracing with Variational Regularization of Latent Knowledge Dynamics

**arXiv ID:** 2608.03811 | [PDF](https://arxiv.org/pdf/2608.03811v1)

**作者:** Carson J. Cook `[一作]` (Amplify Education, Inc.), Luke G. Eglington `[通讯]` (Amplify Education, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种统一的神经变分测量框架（UNVaMP），通过记忆网络与变分推理估计学生知识的时序分布，并可配备可解释的1PL MIRT测量函数；

**💡 创新点**

通过KL正则化控制知识轨迹平滑度，实现可解释知识状态、内部不确定性估计及对多种输入特征的灵活支持；

**🔧 技术方法**

采用RNN（GRU）记忆模块、变分推理编码器、可选MLP或1PL MIRT解码器，训练时结合交叉熵损失与KL正则；

**📊 数据集**

在四个公开数据集（ASSISTments、EdNet、Cloze、Amplify）和一个内部Amplify数据集以及仿真数据上进行实验；

**📈 对比分析**

与BKT、Deep-IRT、SAINT、Lasso LKT等基线模型对比，UNVaMP-MLP在三/四个数据集上均表现最佳或同级；UNVaMP-MIRT虽略逊但仅损失少量准确率，同时提供可解释性；β值可调节轨迹平滑，验证了正则化效果；

**⚠️ 局限性**

缺点包括：β选择缺乏直观标准，平滑度与预测精度的权衡难以确定；不确定性估计未经过校准，缺乏统计覆盖保证；以及对动态真实知识变化的适应性尚未充分验证。

---

## 616. How Usable Are Geospatial Foundation Models? A Systematic Evaluation of 89 Models

**arXiv ID:** 2608.03804 | [PDF](https://arxiv.org/pdf/2608.03804v1)

**作者:** Robin Young `[一作]` (University of Cambridge), Srinivasan Keshav `[通讯]` (University of Cambridge)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并应用了一个以人机交互为核心的七维可用性评估框架，对89个地理空间基础模型（GeoFMs）进行系统评价，揭示了模型开发与实际用户需求之间的脱节。

**💡 创新点**

创新点在于：①通过专家访谈将域专家需求映射到评估维度；②构建了包含接入、交互、可信度、社区支持、科学持久性、多语言支持和离线可用性七个维度的可用性框架；③将评估结果分为可区分性与诊断性两类，为实践者和研究者提供选择与改进方向。

**🔧 技术方法**

采用的技术主要是：HCI理论（Norman、Nielsen、Shneiderman等）指导维度设计；专家访谈和问卷收集需求；使用两位评审的手工打分并计算Cohen’s κ与Gwet’s AC1等统计指标进行一致性检验；最终以可视化图表展示评估分布。

**📊 数据集**

数据集包括：1）89个GeoFM的公开信息（论文、代码仓库、文档、API、GUI等）；2）11位生态与保护科学家的专家访谈问卷结果。

**📈 对比分析**

评估方法是基于手工打分的量化维度，主要呈现各维度的分布和差异性；性能上未给出传统指标，而是通过统计显示约半数模型在接入层面不可用，几乎无不确定性量化，且大多数模型仅支持微调交互。

**⚠️ 局限性**

局限性包括：专家样本量小且单一领域；评估仅覆盖公开可见功能，隐藏功能不计入；评估时间窗口有限，后续模型更新可能影响结果；部分维度缺乏方差导致无法区分模型，说明当前实践仍集中在特定方向。

---

## 617. LegalPincite: Multi-level Legal Information Retrieval Dataset

**arXiv ID:** 2608.03756 | [PDF](https://arxiv.org/pdf/2608.03756v1)

**作者:** Theresia Veronika Rampisela `[一作]` (University of Copenhagen), Giovanni Colavizza `[通讯]` (University of Copenhagen)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开了一个大型多层级法律信息检索数据集LegalPincite，支持案例对案例、段落对案例以及段落对段落的检索；对查询文本进行了引用信息遮蔽，去除了潜在的泄漏，并将所有判决段落纳入检索语料库，同时加入了人工专家验证的相关性标注。

**💡 创新点**

主要创新点包括：
1) 通过命名实体识别与正则表达式系统化地去除查询中的引用信息，解决了现有数据集中引用泄漏问题；
2) 将两份原始数据集（含完整段落与已标注段落对段落关系）进行合并、错误修正与近四年数据更新，提供更真实、完整的检索场景；
3) 设计多层级（case‑case、par‑case、par‑par）评估框架并公开满足FAIR原则的开放数据。

**🔧 技术方法**

使用了法律命名实体识别模型与正则实现查询遮蔽；利用EUR‑Lex Cellar API爬取并解析HTML获取段落文本；采用Python与IR工具库（如pyserini）实现TF‑IDF、BM25、LMIR（Dirichlet）与DPH等基线检索；通过BERTScore、Jaccard等指标评估查询-文档相似度；对数据集进行时间切分、人工标注合并与统计。

**📊 数据集**

主要数据来源为：
- 原始未公开的包含全部段落的CJEU判决集合；
- 已公开的段落对段落引用数据（110,601对）及其890条人工验证标注；
- 通过EUR‑Lex检索得到的2021‑2025年新增判决，补充了最新四年的判决段落；
- 对所有段落进行语言过滤，保留英语文本。

**📈 对比分析**

方法比较：在dev/test集上对四种bag‑of‑words基线（TF‑IDF、BM25、LMIR、DPH）分别评估HR@k、NDCG@k、MAP与MRR。结果表明：
- case‑case检索中LMIR表现最佳；
- par‑case检索中TF‑IDF最佳；
- par‑par检索中BM25与TF‑IDF相近；
- 数据泄漏（原始或仅去除段落ID）会显著提升或降低NDCG，尤其在par‑par检索中去除引用信息后性能提升最高（约0.19）。
- 在所有层级与分割下，NDCG@10普遍落在0.43–0.56区间，表明该数据集具备挑战性。

**⚠️ 局限性**

限制：
1) 绝大多数ground‑truth来自EUR‑Lex，可能存在反馈循环（判决者检索后引用其高检索结果）。
2) 人工验证仅覆盖top‑10检索结果，标注不完全；
3) 只保留英语段落，忽略多语言特性；
4) 查询遮蔽步骤未完全覆盖所有泄漏实例，可能残留部分信息；
5) 评估侧重精确度（short‑cutoff），忽略召回，难以全面衡量检索效果。

---

## 618. Delay Attacks on the German Smart Metering Infrastructure: A Security Analysis of CLS Channel Timing Constraints

**arXiv ID:** 2608.03751 | [PDF](https://arxiv.org/pdf/2608.03751v1)

**作者:** Fabio Stoll `[一作]` (Albstadt-Sigmaringen University), Joachim Gerlach `[通讯]` (Albstadt-Sigmaringen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对德国智能计量基础设施的CLS通道时序约束进行安全分析，揭示并验证了多种基于延迟的攻击手段。

**💡 创新点**

创新点在于首次系统化地阐述了时延注入对DAPL、CLS.EEDI和IEC 61850协议的攻击向量，并演示其对电力系统静态负载调节的潜在威胁。

**🔧 技术方法**

采用网络时序分析、协议重放与实验室仿真相结合的技术，对三种协议实施延迟注入与捕获。

**📊 数据集**

实验使用在实验室搭建的德国计量环境仿真数据集，涵盖真实计量设备的流量日志。

**📈 对比分析**

通过与默认安全配置对比，实验结果表明攻击可导致负载误报幅度提升至数十个百分点，恢复时间延迟达数秒，性能影响显著。

**⚠️ 局限性**

主要局限在于实验仅在受控实验室环境完成，未验证在大规模现场部署中的可扩展性与实用性。

---

## 619. Predictive Triggering for Outage-Resilient Threshold Decisions over Short-Packet Links

**arXiv ID:** 2608.03750 | [PDF](https://arxiv.org/pdf/2608.03750v1)

**作者:** Nho-Duc Tran `[一作]` (Mid Sweden University), Mikael Gidlund `[通讯]` (Mid Sweden University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对短包无线链路的阈值决策的预测触发与鲁棒性更新策略，兼顾误判率与提前决策的可靠性；

**💡 创新点**

创新点在于：①基于后验误判率阈值定义决策可行区域并触发预测更新；②引入AoI控制的鲁棒更新以检测并恢复链路中断；③构建两态马尔可夫代理简化长期能耗与可靠性分析；④联合优化传输功率与AoI阈值，实现能耗最低且满足提前决策成功率；

**🔧 技术方法**

采用Kalman滤波、短包信息论PER模型、AoI与离散Weibull失效模型、两态马尔可夫近似、解析/仿真求解最优策略；

**📊 数据集**

使用仿真数据：系统矩阵A=[0 1; -0.9 1.8]，C=[0.5 1.0]，Q=diag(0,1)，R=0.1，阈值Δ=4，PER参数n=128,l=256，噪声σ²=0.4，出故障概率p_r=0.05，恢复分布参数λ=3.0,κ=2.0；

**📈 对比分析**

与四种基线（Ideal、Event-triggered、Predictive-only、AoI-based、AoII-based）在匹配能耗下比较，主要指标为误报率/漏报率、决策提前率(L≥0)和可操作提前率(L>0)。结果显示，所提策略在能耗相同的条件下，误报率略高于AoI-based，但在(L>0)上显著优于所有基线，尤其在中等干扰与容忍度的区间；

**⚠️ 局限性**

局限性包括：①对连续高斯过程的两态马尔可夫近似在长期尾部可能失准；②鲁棒更新的AoI阈值设计依赖经验/仿真，可能在不同系统参数下需重新调优；③仅考虑单一阈值决策，未扩展到多阈值或多类别决策；④缺乏实际硬件验证，仅在仿真中评估。

---

## 620. Dependency Triad: A Metric to Quantify the Dependencies Between Attributes for Local Differential Privacy

**arXiv ID:** 2608.03737 | [PDF](https://arxiv.org/pdf/2608.03737v1)

**作者:** Sandaru Jayawardana `[一作]` (University of Sydney), Kanchana Thilakarathna `[通讯]` (University of Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个新的三参数度量（Dependency Triad）用于在本地差分隐私下评估属性间的相关性诱发隐私泄露（CPL），并能在不完整或不准确的先验分布情况下给出紧凑、保守的估计；

**💡 创新点**

创新点在于用三参数（α、β、δ）压缩联合分布信息，得到可在任意 LDP 机制下的常数时间上界估计，同时对分布不确定性具有可配置容忍度；

**🔧 技术方法**

采用信息论与优化理论（线性分式程序、McCormick 松弛、极值分析）构造上界，并设计了多阶段算法实现 O(a²b³·⁵) 预计算与 O(1) 查询；

**📊 数据集**

在五个真实数据集（SPM、CelebA、Adult、CVD、DSS）以及多种合成数据上进行评估；

**📈 对比分析**

与现有 CPL 计算方法（ITS、CBP、HCC‑1/2、LTM、MI/PCC）对比，证明其在高维、大基数、不同结构和不确定性场景下均能保持低 NMSE‑CPL (<0.025)，同时显著加速（7 s vs 61 s）；

**⚠️ 局限性**

局限性在于目前仅适用于纯 LDP 机制，对近似 LDP 或中心 DP 的扩展尚未完成。

---

## 621. Oilbird: Training-Free Speculative Decoding with Keys the Verifier Already Computes

**arXiv ID:** 2608.03839 | [PDF](https://arxiv.org/pdf/2608.03839v1)

**作者:** Tao Jin `[一作]` (Japan Advanced Institute of Science and Technology), Naoya Inoue `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在无训练的推测式解码中加入一种基于隐藏状态的语义键（semantic key），通过检索历史请求中已生成的序列来补充传统的精确后缀键，从而实现更高的接受长度和更快的推理速度。

**💡 创新点**

创新点在于：①提出了位置级的语义键，使用已验证的隐藏状态作为检索依据，解决了传统精确后缀键在“可识别性缺口”（identifiability gap）中的定位失败；②设计了将语义键产生的链路与现有词法键的树状草稿结构合并的合并策略，节省节点预算；③在多种模型（Llama‑3.1‑8B、Qwen‑3‑8B）和多达十个基准上验证其效果，展示了对三种已发表无训练草稿器的提升。

**🔧 技术方法**

技术手段包括：①无训练的草稿器（SuffixDecoding、Token Recycling、ToolSpec）与精确后缀键；②基于隐藏状态的 k‑NN 检索（相似度阈值、邻近深度动态调整）；③多源草稿树合并算法（共享前缀、按节点预算分配）；④在推理过程中直接利用已验证的隐藏状态，避免额外前向传播；⑤通过批量化验证、节点预算控制实现高效推断。

**📊 数据集**

使用了十个公开基准数据集：API‑Bank、ToolAlpaca、tau‑bench（retail、airline、math）、HumanEval、ClassEval、GSM8K、MT‑Bench、WildChat 以及两项扩展基准（Spec‑Bench math、tau‑bench airline）。所有基准均包含服务流量、工具调用与常规对话等不同场景。

**📈 对比分析**

与基线的比较：在三大重复流量基准（API‑Bank、ToolAlpaca、tau‑bench）中，相较于最强的无训练基线（SuffixDecoding）和已训练模型（EAGLE‑3），本方法在 Llama‑3.1‑8B 上实现了 4.39× 的自动回归解码速度提升（API‑Bank）并将接受长度提升 24–29%；在 Qwen‑3‑8B 上也达到了 4.08× 的速度提升。整体上，本文提出的语义键在所有模型和基准中均保持正向贡献，且在工具调用密集的工作负载中优势最为突出。

**⚠️ 局限性**

局限性包括：①需要存储每个生成标记对应的隐藏状态（≈8 KB/标记），在长时间运行时可能导致显存/磁盘占用过大；②检索开销（k‑NN 计算、相似度阈值）在高吞吐量场景下仍需优化；③性能提升高度依赖于服务流量的重复性，非重复或极度多样化的任务中收益有限；④在批量推理时语义键的优势减弱，批量尺寸增大时速度提升下降。

---

## 622. From b-Coloring to $b^*$-Coloring: Large Girth and Parameterized Complexity

**arXiv ID:** 2608.03819 | [PDF](https://arxiv.org/pdf/2608.03819v1)

**作者:** Jakub Balabán `[一作]` (Masaryk University), Oliver Bukor `[通讯]` (Masaryk University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究b^*-色彩的结构与算法性质，证明了高girth图的b^*-单调性并给出完全可解的正则图例子；同时探讨b^*-色彩的参数化复杂度并给出多种参数下的算法与硬性证明。

**💡 创新点**

首次将b-coloring推广为b^*-coloring并证明在girth≥7的图中b^*(G)=m^*(G)+1，推出b^*-单调性；提出新的正则图类d-regular、girth≥5满足b^*(G)=d+1；将参数化复杂度从b-coloring迁移到b^*-coloring并发现在反馈边数参数下算法更简洁。

**🔧 技术方法**

使用图结构参数（模块宽度、树宽、邻域多样性、双覆盖数等）、Hall定理、动态规划、ILP、树分解、簇宽、以及新定义的m^*-度与fen-core等技术。

**📊 数据集**

无实验数据，全部为理论证明与算法复杂度分析。

**📈 对比分析**

通过与已知b-coloring结果对比，证明在多种结构参数下b^*-色彩的复杂度与b-coloring相同，且在反馈边数参数下的时间为2^{O(p log p)}·n^{O(1)}，相较于b-coloring更高效。

**⚠️ 局限性**

仅在girth≥7下证明b^*-单调性，girth 5及以下仍存在反例；对于某些参数如路径宽度、模宽度仍为W[1]-hard；在实际大规模图中算法复杂度仍指数级，需要进一步改进。

---

## 623. Autoreflection: How Agentic Strange Loops Turn Human Culture into AI Infrastructure

**arXiv ID:** 2608.03800 | [PDF](https://arxiv.org/pdf/2608.03800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 624. Design and Evaluation of an AI-Enabled Cloud-Edge Architecture for Connected Precision Agriculture Farms

**arXiv ID:** 2608.03816 | [PDF](https://arxiv.org/pdf/2608.03816v1)

**作者:** Deckshitha Angadi `[一作]` (Autonomous Robotics Systems Limited), Narsimlu Kemsaram `[通讯]` (University of Malaya)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并评估了一套 AI 支持的云‑边缘架构，用于连网精准农业场景下的番茄病害自动检测与监控。

**💡 创新点**

创新点包括：① 将 IoT 传感器、UAV 摄像、移动/Web/边缘设备多平台融合；② 通过 Azure IoT Hub 建立实时与批量分析双通道；③ 在 Raspberry Pi 5 上部署量化的 TensorFlow Lite 模型，实现离线即时诊断；④ 统一 API 使模型可在边缘、云、移动等多端无缝迁移。

**🔧 技术方法**

使用技术包括：IoT 传感器（土壤湿度、pH、光照、温湿度等）、UAV RGB 摄像、Raspberry Pi 5 边缘计算、TensorFlow/TensorFlow Lite、MobileNetV2（边缘）与 ResNet‑50（云端）、Azure IoT Hub、Stream Analytics、Azure Machine Learning、Azure SQL、Power BI、Java/Flask Web、Android Studio。

**📊 数据集**

训练与验证数据主要来自公开数据集 PlantVillage 与 Kaggle，另补充自采的番茄叶片图像与现场传感器记录。

**📈 对比分析**

在移动端、Web 端与边缘设备上对比实验，评估指标为准确率、精确率、召回率和 F1‑score。移动端 92%/0.91/0.90/0.905，Web 95%/0.95/0.94/0.945，Raspberry Pi 93%/0.92/0.91/0.915；平均推理延迟低于 8 秒。整体准确率 92–95%，体现了低延迟边缘推理与高精度云推理的互补优势。

**⚠️ 局限性**

局限性包括：仅针对番茄病害，缺乏对多种作物和多样环境条件的验证；模型对光照、天气变化敏感；硬件成本与连网可靠性仍是推广壁垒；系统未实现 UAV 的自适应飞行规划与闭环控制。

---

## 625. M-GATE: Multilingual Grammar, Accuracy in Translation, and Efficiency Benchmark for Large Language Models

**arXiv ID:** 2608.03803 | [PDF](https://arxiv.org/pdf/2608.03803v1)

**作者:** Tomáš Burkert `[一作]` (RWS TrainAI), David Zelený `[通讯]` (RWS TrainAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 M-GATE 基准，用于衡量多语言模型在语法错误检测和意义保持（回译）两项任务上的语言能力，并提供了持续更新、非公开测试集的公开排行榜。

**💡 创新点**

创新点包括：① 将任务聚焦于语言掌握而非任务完成；② 采用专业语言学家精心挑选的对抗性语法样本；③ 采用回译并用 LLM 判别器验证意义保持，避免了传统参考文本和人工评审的局限；④ 将语料覆盖 30 种语言（含低资源语言），并分为三资源层级，支持跨语言比较；⑤ 公开透明的 token‑efficiency 评估。

**🔧 技术方法**

技术实现上使用了零样本二分类提示（语法检测）与双向翻译提示（回译），并采用三方 LLM 判别器进行分数归一化；对模型进行温度 0 及默认温度下的多次采样；通过自定义 Python 脚本处理提示、结果解析与指标计算。

**📊 数据集**

数据集：① 30 种语言的 100 条专业语言学家构造的“stumper”句子（语法错误检测）；② 100 条英语源句子，覆盖七类翻译难点（回译任务），每句都有对应的目标含义与评分例子；③ 语言选择依据标准方言、官方规范与 Common Crawl 数据量，形成三资源层级。

**📈 对比分析**

比较方法：对 53 个模型（82 个配置）分别在两项任务上得分；语法检测用 Matthews Correlation Coefficient（MCC），回译用 1–5 量表归一化；跨模型、跨语言对比通过热图和相关系数（语法与回译约 0.56/0.63 相关）。性能显示：最强模型在回译上可达 82%（GPT‑5.5 高推理）但在语法检测上最高仅 0.36 MCC；多数模型在语法检测上接近随机，说明流利度与语言掌握差距显著。

**⚠️ 局限性**

局限性：① 语法任务仅评估二分类判断，未捕捉错误定位或解释；② 回译任务依赖 LLM 判别器，尽管验证良好，但仍可能在极低资源语言上产生偏差；③ 测试句子为非并行文本，无法直接衡量跨语言一致性；④ 评测采用温度 0 近似确定性，可能忽略某些模型的随机性；⑤ 词元效率评估基于句法不同的字符/词元比，未考虑内容等价性。

---

## 626. Fixed Budget vs. Covering Target: The Partial Set Cover Boundary for Bounded VC-Dimension

**arXiv ID:** 2608.03801 | [PDF](https://arxiv.org/pdf/2608.03801v1)

**作者:** Madhumita Kundu `[一作]` (University of Bergen), Anannya Upasana `[通讯]` (Institute of Mathematical Sciences, HBNI)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究参数化近似问题——在给定预算 k 的情况下覆盖尽可能多的元素（Maximum Coverage）和在满足给定目标 t 时使用最少集合数的目标覆盖问题（Partial Set Cover）。作者证明：在 VC‑维度受限的集合系统中，Maximum Coverage 可得到高效参数化近似方案（EPAS），但同样的结构限制并不能保证 Partial Set Cover 的良好近似。为此，作者提出更强的结构指标——半梯形指数（semi‑ladder index），并证明当该指数受限时，Partial Set Cover 可以实现 (1+ε) 的参数化近似。论文同时给出了针对加权版本、集合系统中的 matroid/划分约束、以及对 weighted CC‑MaxSAT 的可约化等扩展。

**💡 创新点**

创新点：①在 VC‑维度足够小的前提下，提出新的半梯形指数来捕捉集合系统的结构，弥补了传统 VC‑维度在目标覆盖问题上的不足。②给出了新的负结果：即使 VC‑维度为 7，Partial Set Cover 也不存在 (2‑δ) 的参数化近似，甚至在 ETH 下也不存在参数化近似方案；这说明 VC‑维度并非目标覆盖的足够条件。③设计了一种基于随机分支与下行交叉复杂度（downward intersection complexity）的算法框架，能在指数时间内实现 (1+ε) 的参数化近似，同时兼顾加权、matroid/划分约束等。

**🔧 技术方法**

技术：
- 结构层次分析：定义并证明半梯形指数与 VC‑维度、K_{d,d}‑自由等关系；
- 随机分支策略：在每一步以 1/2 概率选择顶点，若不包含则证明其邻域一定包含 ε‑分量，进一步随机挑选覆盖点；
- 下行交叉复杂度与半梯形指数的对应关系，限制递归深度为 O(k·Γ)；
- 结合 matroid 代表族与划分约束的代表子族技术，实现对更复杂约束的处理；
- 可约化：将 weighted CC‑MaxSAT 归约到一族受限 VC‑维度/半梯形指数的 Weighted Max‑Coverage，保持逼近因子与结构不变；
- 低阶硬性证明：利用 1‑局部表示、行/阻塞列表结构构造 1‑局部集合系统，证明其 VC‑维度 ≤ 7，从而给出低维下的硬性结果。

**📊 数据集**

本文为理论工作，未使用实际数据集；所有证明均基于构造性的集合系统、图形结构与复杂性假设（FPT≠W[1]、ETH）。

**📈 对比分析**

与已有工作比较：
- 在 VC‑维度受限的情况，Badanidiyuru 等人提供了 EPAS；本文进一步指出此类结构不足以处理 Partial Set Cover。 
- 对 K_{d,d}‑自由集合系统，Jain 等人实现了加一套的覆盖保证；本文通过半梯形指数证明此类结构是最小足够的，并在此框架下给出更通用的加权与约束扩展。 
- 在性能方面，本文的 EPAS 运行时间为 2^{O(Γ·k·log k)}·N（Γ 为下行交叉复杂度，等价于半梯形指数），相较于 Badanidiyuru 等人 2^{O(k^2·d/ε)} 的实现，显著降低了 k 维度的指数因子；在可约化与 weighted CC‑MaxSAT 方面，亦给出更高效、确定性的实现。

**⚠️ 局限性**

局限性：
- 需要对输入集合系统的半梯形指数做预先估计；若 Γ 仍较大，时间仍为指数级。 
- 对于仅满足 VC‑维度受限但半梯形指数较大的实例，本文方法不具备优势。 
- 负结果主要基于假设 FPT≠W[1] 与 ETH，若这些假设被推翻，结果可能需重新评估。 
- 论文中的随机化算法在实践中需多次重复以提升成功概率，且实现复杂度相对较高。

---

## 627. Risky Business: Measuring The Faithfulness-Safety Tension

**arXiv ID:** 2608.03745 | [PDF](https://arxiv.org/pdf/2608.03745v1)

**作者:** Dominik Meier `[一作]` (University of Göttingen), Bela Gipp `[通讯]` (University of Göttingen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个名为HazMart的自主 AI 店员安全评估数据集，并提出了一种基于推理链替换的干预方法 TRR，用于同时评估模型的推理可信度（faithfulness）与安全性（safety），从而揭示二者之间的对立关系；

**💡 创新点**

创新点在于（1）创建了包含 77 个人工撰写、涵盖 11 种危害类别的实际情境数据集，能真实模拟目标冲突；（2）提出 TRR 方法直接在模型生成的推理链中进行有针对性的词替换，以独立考察对推理可信度与安全性的影响；（3）通过内部机制分析发现安全与可信度分别对应模型残差流中的两条互相反相关的向量，并证明可以通过激活注入实现可控的安全提升；

**🔧 技术方法**

主要技术包括：大规模语言模型（Qwen、DeepSeek、QwQ 等）的推理与函数调用、TRR 推理链替换、差分均值差异探针（difference‑of‑means probing）、激活添加驱动（activation‑addition steering）以及注意力掩码消融等内部机制分析方法；

**📊 数据集**

使用 HazMart 数据集（77 例场景，11 类危害）以及现有的多项选择推理任务；

**📈 对比分析**

实验对比了七种前沿大语言模型，在 HazMart 上评估 faithfulness（97.5% DeepSeek）与 safety（73.9% QwQ‑32B）的性能。实验表明高 faithfulness 并不必然对应高 safety，且通过在 QwQ‑32B 的第 44 层残差流中激活安全向量可将安全率提升约 9pp，且不会削弱通用推理能力；

**⚠️ 局限性**

局限性包括：数据集规模仅 77 条，难以获得足够的统计力量；TRR 需要可直接操作推理链，限制了对封闭式模型的适用；内部机制分析仅针对 QwQ‑32B，是否可推广至其他架构未知；TRR 替换可能引入语言痕迹，影响模型检测；

---

## 628. Equivariant Music Transformer

**arXiv ID:** 2608.03920 | [PDF](https://arxiv.org/pdf/2608.03920v1)

**作者:** Zixun Guo `[一作]` (Queen Mary University of London), Simon Dixon `[通讯]` (Queen Mary University of London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Equivariant Music Transformer（EMT），通过联合下一个 token 预测与等变正则化损失，实现音乐片段在时间平移和音高转调下的等变性与生成能力提升。

**💡 创新点**

创新点在于：① 引入模型级等变正则化（自蒸馏 KL 损失）直接约束潜在空间的等变性；② 将特征级等变（FME+MRA）与模型级等变结合，形成更强的结构正则化；③ 证明仅靠规模或训练时间无法自然获得等变性，需显式监督。

**🔧 技术方法**

使用基于 Moonbeam 的 transformer 结构，加入绝对onset预测、合并 octave+pitch 词典、token 延迟模式；自蒸馏双分支训练；KL 损失约束等变正则；随机音高/时间平移、stop‑gradient；混合精度与分布式训练。

**📊 数据集**

主要使用 LakhMIDI 数据集（clean subset 训练/验证，full 集合用于主实验），对 MIDI 事件做 5 维 (onset, pitch, duration, instrument, velocity) 处理，按固定长度 1024 采样块训练。

**📈 对比分析**

与数据增强、特征级等变、Anticipatory Music Transformer (S/M/L, 100k–800k 步)、MIDI‑LLM 等基线对比。EMT 在等变性指标（KL loss、Top‑1 Accuracy、Top‑5 Jaccard）上均优于所有基线；在 MOS 评估中，EMT 与 Anticipatory 在未平移条件下生成质量相当，但在平移条件下的 MOS 下降更小，显示更稳健。

**⚠️ 局限性**

局限：训练时需两次前向传播，成本近翻倍；仅在符号 MIDI 数据上验证，未测试到真实音频生成；等变正则对非常大模型或更长训练时间的适应性仍待评估。

---

## 629. CPrefix: A Combinatorial Tensor Framework for Structured Discrete Color Mappings

**arXiv ID:** 2608.03863 | [PDF](https://arxiv.org/pdf/2608.03863v1)

**作者:** Yvan Richard `[一作]` `[通讯]`, Yvan Richard

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个称为CPrefix的组合可观测表示，用于将离散多通道映射从样本化值转化为离散帕斯卡简单形中的潜在组合结构，并通过张量方法实现对映射的表示、重建和结构分析。

**💡 创新点**

创新点在于：①将组合计数张量与多元帕斯卡简单形相结合，形成了一个隐藏的组合坐标系；②将观察空间与潜在组合空间解耦，使得映射的内在组合组织与测量值分离；③通过 shell 级别的张量收缩和逆问题实现高精度重建与可视化；④基于潜在坐标的感知色域映射模型，实现了更具结构性和可解释性的 gamut 传输。

**🔧 技术方法**

使用的技术包括：多项式计数观测、离散帕斯卡简单形、张量构造与收缩、shell 排序、正则化逆问题（T^⊤T+λI）⁻¹T^⊤、感知颜色差异 ΔE₀₀ 评估以及基于 shell‐band 的潜在空间传输。

**📊 数据集**

实验数据集主要是工业 ICC 配色文件：Apple P3 显示、Epson P9000、HP Z3200、Canon Pro4100 等打印机配置文件，以及 Hahnemühle Photo Rag® 打印机；RGB 样本网格分别为 (N=16) M=4913,K=969 和 (N=33) M=39304,K=7140。

**📈 对比分析**

通过与原 ICC 映射直接比较，采用 XYZ 均方根误差和 ΔE₀₀ 评价；显示配置文件重建误差几乎为零，打印机配置文件误差均低于 1，满足可见阈值；在感知 gamut 映射实验中，shell‑band 传输模型在 N=33 时得到 ΔE₀₀ 平均 0.38、95% 分位 0.82、最大 3.22，表明重构和映射效果相当优异。

**⚠️ 局限性**

局限性包括：仅在三通道 RGB 上验证，未对更高维度系统作实测；传输模型仅考虑 shell‑local 交互，跨 shell 的非线性关系尚未充分建模；张量规模随 N 增大快速增长，可能导致计算成本和存储需求较高。

---

## 630. Beyond Representational Similarity: Source-Conditioned Description-Length Gain for Generative Plagiarism Detection and Candidate Source Reranking

**arXiv ID:** 2608.03859 | [PDF](https://arxiv.org/pdf/2608.03859v1)

**作者:** Peijia Guo `[一作]` (Fudan University), Ming Li `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑free 的 Source‑Conditioned Description‑Length Gain（SCDG）框架，用来检测大型语言模型生成文本中被重写后仍可追溯到的原始来源。

**💡 创新点**

创新点在于：①利用模型对比描述长度（负对数似然）来量化候选来源对可疑文本的可预测性提升；②将增益分解为 token 级别，支持稀疏聚合；③可直接用于二分类与候选来源重排序，无需额外训练；④对深度重写、多源合成仍具鲁棒性。

**🔧 技术方法**

技术包括：使用冻结自回归 LLM（Qwen3‑8B‑Base、Meta‑Llama‑3.1‑8B、Ministral‑3‑8B‑Base‑2512）计算无来源与有来源下的负对数似然；token‑级 SCDG 累积与聚合；与 DAAC 检索融合；与 Lexical、Embedding、Prompt‑LLM、Retrieval‑Fusion 等基线对比。

**📊 数据集**

使用 PAN 2025 衍生的对比检测数据集（6,791 对，4,777 正例）、PAN 2026 多源检索基准（200 查询，86,822 候选）、以及 Multi‑News 同主题压力集（8,000 对）进行评估。

**📈 对比分析**

与传统基线比较，SCDG 在 PAN 2025 上取得 F1≈0.94、精确率≈0.92、召回率≈0.96；在 PAN 2026 上 nDCG@10≈0.83、Recall@100≈0.96；在 Multi‑News 上假阳性率仅 0.125%，显著低于 Lexical/Embedding 等方法；整体性能优于所有基线。

**⚠️ 局限性**

局限性：计算成本高于相似度度量；仅在英文科学文本和新闻样例上验证；缺乏多语言和跨领域评估；未提供段落级证据定位；需要进一步提升评分效率。

---

## 631. Quantization Effects on Biomedical LLM Reliability

**arXiv ID:** 2608.03854 | [PDF](https://arxiv.org/pdf/2608.03854v1)

**作者:** Anton Rasmussen `[一作]` (Old Dominion University), Hong Qin `[通讯]` (Old Dominion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种 Mistral‑7B 变体（Base、BioMistral、Instruct）在 PubMed RCT 句子分类任务上，使用 FP16、INT8、INT4 三种量化精度和四种 prompt 模板，系统评估了准确率和校准指标（ECE、Brier 等）。

**💡 创新点**

创新点在于：① 明确揭示概率提取协议（prompt、verbalizer、分数规则）对校准排名的主导作用；② 对不同量化精度与模板对准确率与校准的细致比较；③ 通过进化式提示搜索得到高性能模板；④ 证明单词代码评分失效并提出答案文本评分方案；⑤ 通过多规则（sum 与 mean token）比较，展示校准排名可逆。

**🔧 技术方法**

使用技术包括：答案文本 log‑likelihood 评分（sum/mean），温度缩放后处理，Mistral‑7B（Base、BioMistral、Instruct）模型，LLM.int8() 与 NF4（INT4）量化，进化提示搜索，评估指标（accuracy、macro‑F1、ECE、Brier、NLL）。

**📊 数据集**

数据集为 PubMed RCT 句子分类（5 类），测试集 2000 条自然分布样本，15% 验证集用于温度缩放；PubMedBERT（110M）作为监督编码器基准，训练约 176K 标注样本，评估 400/类平衡样本。

**📈 对比分析**

采用相同模板和精度条件对模型进行匹配比较；结果显示 BioMistral 在 Bare‑Cont 模板下达到最高准确率 70.6%，Instruct 稍高；INT8 对准确率影响 ≤1pp，INT4 影响更异质；提示模板对准确率和校准的影响大于模型差异；温度缩放在 sum 评分下可显著降低 ECE，但在 mean 评分下效果不同。

**⚠️ 局限性**

局限性包括：仅评估单一 PubMed RCT 任务；量化方法仅为 INT8/INT4；使用不等长度标签导致 surface‑form prior；温度缩放样本量小、离散网格搜索；模板搜索可能过拟合；未对更大模型或其他压缩技术做验证；PubMedBERT 基准与 decoder 结果不可直接比较；模型比较未完全隔离 instruction 与 domain 影响。

---

## 632. The Transformer Revolution, Part 1: Dynamic Processing through Output- Weight Interconnections

**arXiv ID:** 2608.03921 | [PDF](https://arxiv.org/pdf/2608.03921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 633. FedCritic-MIMO: Communication-Efficient Serverless Federated Critic Learning for Massive-MIMO Resource Control in Open and Disaggregated 6G RANs

**arXiv ID:** 2608.03852 | [PDF](https://arxiv.org/pdf/2608.03852v1)

**作者:** Amin Farajzadeh `[一作]` (University of Ottawa), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 FedCritic‑MIMO，一种在开放式、解耦 6G RAN 中实现的通信高效、无服务器的联合 Critic 学习框架，支持单元级控制器在 massive‑MIMO OFDMA 环境下通过共享 Critic 子网络实现资源调度、功率分配与波束成形的协同学习；

**💡 创新点**

核心创新点包括：① 只在共享 Critic 子网络上进行服务器无关的同伴通信；② 结合无线感知的事件触发、误差反馈的 adaptive top‑k 稀疏压缩以及干扰感知的对称融合，实现既低通信负载又能精准捕捉干扰耦合；③ 在固定策略、冻结目标的前提下给出有限时间的收敛与一致性理论保证；

**🔧 技术方法**

技术手段：分布式 Actor‑Critic (PPO)；无服务器联邦 Critic 通信；事件触发阈值与量化误差反馈；层级 top‑k 稀疏压缩；干扰感知加权融合；理论分析利用同步 gossip、随机梯度与谱混合条件；

**📊 数据集**

基于仿真数据：7 基站、每基站 8 UE、32 天线、16 子载波、最多 3 并行空间流；采用双高斯-马尔可夫 Rayleigh 小尺度衰落与对数正态大尺度衰落；仿真中每个基站独立训练六个随机种子，并在保留的 30 条新通道上验证。

**📈 对比分析**

与 9 种基线（随机、贪婪、独立 PPO、无联邦干扰 PPO、CTDE‑MAPPO、完整周期交换、事件触发但未压缩、压缩但无触发）比较。FedCritic‑MIMO 在验证与 held‑out 奖励上分别比 Event‑Uncompressed 提升 6.8%、比 CTDE‑MAPPO 提升 7.2%、比 No‑Federation‑IA‑PPO 提升 9.1%；平均 SINR 提升 1.7–2.5 dB；干扰‑比率下降 8–12%；训练侧 Critic 通信量减少约 76%（相比完整周期交换和未压缩）并仅比 CTDE‑MAPPO 低约 27%；同时实现 QoS 满足率与干扰效率兼顾。

**⚠️ 局限性**

主要局限：仅在仿真环境中验证；不支持异步/延迟通信与移动用户；理论分析基于固定策略、冻结目标的理想假设，未证明全局收敛或在实际网络中可直接部署；需要事先建立干扰邻居图；缺乏对极大规模异构网络与非理想 CSI 的评估。

---

## 634. GENESIS: Towards Explainable Causal Discovery

**arXiv ID:** 2608.03868 | [PDF](https://arxiv.org/pdf/2608.03868v1)

**作者:** Abhinav Thorat `[一作]` (Sony Research India), Niranjan Pedanekar `[通讯]` (Sony Research India)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Genesis的混合因果发现框架，旨在实现边缘可追溯性，即每条学习到的有向无环图（DAG）边都有可审计的统计证据、马尔可夫毯一致性或明确的域知识支持。

**💡 创新点**

创新点在于：① 将三节点因果图案（链、叉、碰撞）作为推理单元，将LMM提取的语义先验与统计证据按层级融合；② 通过分阶段（LMM探测 → 数据验证 → 逐步细化）保证每条边都有可追溯的证据；③ 在缺乏真实DAG的场景下提供完整的审计追踪。

**🔧 技术方法**

技术包括：大语言模型（GPT‑5、Qwen‑3.5‑27B、Gemini‑3 Pro）用于三节点图案提取、马尔可夫毯估计和反事实推理；统计方法如条件独立检验、BIC、ANM、LiNGAM；以及基于马尔可夫毯一致性的过滤与最终判断。

**📊 数据集**

使用了五个经典贝叶斯网络基准数据集（Asia、Cancer、Child、Earthquake、Survey），分别在1000、2500和5000样本规模下进行实验。

**📈 对比分析**

与传统约束式（PC）、评分式（NOTEARS、SCORE、CaMML）、可识别功能模型（DirectLiNGAM）以及最近的LLM辅助方法（CausalOrder、LLM‑CD、SLLM‑CD）进行对比。实验显示Genesis在大多数数据集和样本规模下的结构误差（SHD）优于传统方法，且与最先进的LLM辅助方法持平，同时实现100%边缘可追溯性。

**⚠️ 局限性**

局限性包括：对三节点图案的依赖可能在更大或更稠密的图中产生计算开销；LMM的可靠性受模型规模和提示设计影响；在极端低样本或高度混杂的情况下，统计验证可能不足，导致LLM反事实推理被频繁调用，进一步降低效率。

---

## 635. GeoMAR: Unleashing Geometrically Aligned Features for Masked Autoregressive Blind Face Restoration

**arXiv ID:** 2608.03923 | [PDF](https://arxiv.org/pdf/2608.03923v1)

**作者:** Lu Gan `[一作]` (Sun Yat-sen University), Dan Zeng `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 GeoMAR 框架，用于盲面部恢复任务。

**💡 创新点**

创新点包括：①双输入几何先验提取与对齐注入，②Masked Autoregressive 多步细化机制，以解决传统一阶映射的特征歧义和预测脆弱性。

**🔧 技术方法**

主要技术：使用 Vision‑Language 模型提取几何先验，SFT 与 KV‑Q 交叉注意力实现特征对齐，Masked Autoregressive Transformer 进行多步代码细化，以及 VQGAN 代码库进行离散化编码。

**📊 数据集**

数据集：训练使用 FFHQ；评估在合成的 CelebA‑Test、真实的 LFW‑Test、WebPhoto‑Test 与 WIDER‑Test 上。

**📈 对比分析**

与 GAN、扩散、代码库等九种方法对比，GeoMAR 在 FID、NIQE、MANIQA 等指标上位居前列，尤其在 WebPhoto‑Test 上取得最佳 FID；推理速度约 3.9 s/张，兼顾性能与效率。

**⚠️ 局限性**

局限性：对严重遮挡或解析错误的输入依赖较高，极端遮挡下会产生不一致细节；对解析图的精度与 VLM 描述的可靠性敏感。

---

## 636. Socially Grounded Agentic AI: Coordinating Plural Perspectives through Social Theory

**arXiv ID:** 2608.03910 | [PDF](https://arxiv.org/pdf/2608.03910v1)

**作者:** Matt Ratto `[一作]` (University of Toronto), Daniel Silver `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了以社会理论为基础的多元对齐框架，阐述如何通过角色化表征、结构化协商和领域感知聚合来实现 AI 系统的多视角协调。

**💡 创新点**

创新点在于将梅德的“普遍他者”、哈贝马斯的交往行动理论和布尔迪厄的场域理论分别映射到 Overton、可调式与分布式多元对齐的三个策略，形成一个完整的社会协调设计空间，并将对齐评估从单一输出扩展到交互轨迹层面。

**🔧 技术方法**

技术路径包括：角色化提示与元数据注入、基于多智能体的结构化对话与证据推理、基于混合专家的领域感知路由与加权聚合、以及交互日志与轨迹评估框架。

**📊 数据集**

文章为理论与设计性工作，未使用具体实验数据集；提及可通过现有对话或决策数据集（如医疗决策记录、公共服务交互日志）进行后续实现与评估。

**📈 对比分析**

暂无实验比较与性能指标，作者指出未来工作需在医疗、公共服务等真实场景中实现并评估本框架的可解释性、合法性和适应性。

**⚠️ 局限性**

局限性：1) 仅提供概念与技术路线，缺乏实现细节和实证验证；2) 依赖社会理论框架的选择可能导致覆盖不全，未覆盖所有可能的社会结构视角；3) 对安全、权益与专业约束的界定仍需进一步细化，框架本身无法保证所有情境下的合规与公平。

---

## 637. Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse

**arXiv ID:** 2608.03893 | [PDF](https://arxiv.org/pdf/2608.03893v1)

**作者:** Taekyung Heo `[一作]` (NVIDIA), Bita Darvish Rouhani `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型生产环境中实现跨模型 KV 缓存迁移，通过源模型的 KV 缓存直接映射到目标模型，从而跳过重新预填过程。

**💡 创新点**

提出了基于每头岭回归的闭式映射框架，利用 RoPE 去除位置因素并按层选择最具预测力的源层，实现高效、无梯度训练的跨模型 KV 迁移。

**🔧 技术方法**

闭式岭回归、RoPE 位置拆解、跨层源层挑选、内容空间映射，以及可选的非线性 MLP 细化。

**📊 数据集**

使用 FineWeb‑Edu 500 条 1024 词长序列进行校准，评估基准包括 ARC‑Challenge、HellaSwag、WinoGrande、MMLU、GSM8K、WikiText‑2 以及 CoQA。

**📈 对比分析**

与完整预填对比，迁移模型在 6 对匹配 KV 对中保留 73–98% 的下游准确率，并在小到大方向下实现 2.7–25× 的推理延迟加速；在大到小方向也能保持 3–7× 的速度提升，且多轮交互中漂移极小。

**⚠️ 局限性**

仅在同家族匹配 KV 的稠密全注意力模型上验证，缺乏跨家族或不匹配 KV 的实验；校准数据仅来自单一领域；k 选取依赖于留样评估；未探讨非全注意力或混合注意力架构。

---

## 638. Omega-S: A Functional Resilience Index for LLM Fine-Tuning

**arXiv ID:** 2608.03887 | [PDF](https://arxiv.org/pdf/2608.03887v1)

**作者:** Alberto Acedo `[一作]` `[通讯]` (Biome Makers Inc.), Alberto Acedo (Biome Makers Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Omega-S 的正则化方法，旨在解决大语言模型微调时的灾难性遗忘问题。

**💡 创新点**

创新点在于直接从权重矩阵构造的拓扑图（基于 Tr(A³)）得到正则项，无需先前任务数据、Fisher 矩阵或旧权重拷贝，并通过 log‑ratio 形式消除尺度敏感性，使正则化更稳健。

**🔧 技术方法**

技术手段包括：
- 通过矩阵 W 与其转置构造邻接矩阵 A；
- 使用 Hutchinson 估计 Tr(A³)；
- 对 A 进行 sigmoid 映射和对数组合得到四个量（C, D, M, Coex）；
- 在 LoRA 微调训练循环中每 K 步加入正则化，计算复杂度 O(N²)；
- 在 PyTorch、HuggingFace + PEFT 框架下实现，支持 FSDP 与多 GPU。

**📊 数据集**

数据集与评估：在 Llama‑3‑8B 上对代码→文稿微调任务进行评估，使用 HumanEval 10 个随机种子作为衡量原始能力保持的指标；同时对比权重衰减和 EWC 等基线。

**📈 对比分析**

比较方法与性能：
- Omega‑S 在 9/10 种子上比无正则化提升绝对 pass@1 从 0.173 到 0.238（+37.7%，p=0.011）；
- 在 10/10 种子上优于调优权重衰减（p=0.002）和调优 EWC（8/10，p=0.014）。
- 与基线同一实验环境下跑步差异标准差为 0.104，证明了方法的可靠性。

**⚠️ 局限性**

局限性：
- 仅在低秩 LoRA 微调场景下评估，未验证在全参数微调或其他模型结构上的效果；
- 通过 sigmoid 归一化导致聚类项 C 几乎失活，可能限制了原本拓扑信息的利用；
- 正则化参数 λ_Ω 需手动校准，适用性对不同任务或模型规模可能不稳定。

---

## 639. MultiGlobeQA: A Multilingual and Globally Diverse Benchmark for Geospatial Reasoning

**arXiv ID:** 2608.03882 | [PDF](https://arxiv.org/pdf/2608.03882v1)

**作者:** Martin Böckling `[一作]` (University of Mannheim), Andreea Iana `[通讯]` (University of Mannheim)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出并发布了MultiGlobeQA，一个涵盖46,060条问答对、14种空间函数、15种答案格式、17种语言、201个国家/地区的多语种地理空间推理基准，并提供了执行验证的黄金答案。

**💡 创新点**

创新点在于将执行验证与多语种并行化相结合，采用收入与人口密度分层抽样，设计了错误前提与多模态子集，并通过三种地理知识图谱生成可靠答案，填补了现有基准在规模、语言、多样性和真实性方面的空白。

**🔧 技术方法**

技术方法包括：模板化问答生成、与三大知识图谱（Geonames、DBpedia/Wikidata、H3）联动执行查询得到黄金答案、使用Google Translate+人工校对实现多语言翻译、基于规则+LLM的后处理修正语法错误，并在多重评估层（参数、推理、代理）中加入检索、工具调用与金手指或acles。

**📊 数据集**

使用的数据集为：Geonames、DBpedia/Wikidata（包含三维属性和文本描述）以及基于H3的地理索引图谱，三者共同构成了地理实体与属性的基础；问答样本通过模板与实体采样生成，覆盖高低收入与不同人口密度区域。

**📈 对比分析**

通过对四个模型（包括开源大模型和闭源模型）在四个评估层（无检索、加推理、检索+工具、oracle）下的EM和NE进行对比，发现检索与工具提升最大，但即便给出金三元组，模型最高EM仍仅达61.6%，显示计算能力是瓶颈；在坐标与方向类问题上表现最好，而网格索引和形状类问题表现最差。

**⚠️ 局限性**

主要限制包括：模板覆盖受限于手写查询、答案基于固定快照可能过时、模型选择受限于实验预算、无法评估自由文本输出与答案格式选择能力，以及基准依赖的知识图谱在高收入地区仍显稠密，导致地区间性能差异难以完全消除。

---

## 640. Heterogeneity-Aware Microscaling for Efficient Low-Bit LLM Inference

**arXiv ID:** 2608.03867 | [PDF](https://arxiv.org/pdf/2608.03867v1)

**作者:** Junyi Luo `[一作]` (Brown University), Mehdi Saligane `[通讯]` (Brown University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AdaMX格式与加速器，实现低位（4-bit）LLM推理的自适应微尺度量化，支持块级和算子级的量化异质性；

**💡 创新点**

创新点在于利用MX 8-bit指数位的冗余，将块级格式选择与精度恢复机制拆分为两轴，权重块通过2-bit路由选择四种格式+增强组合，激活块采用最优舍入与无损块最大扩展；

**🔧 技术方法**

技术手段包括两级尺度结构、权重块离线搜索、激活块在线量化、共享32通道乘法器与轻量级校正侧路径、支持块大小16/32的统一数据通路；

**📊 数据集**

实验使用Llama-3.1-8B/70B、Qwen2.5-3B/14B以及Gemma-4 12B多模态模型，评估WikiText-2困惑度、零样本常识/5-shot MMLU、四个图文基准；

**📈 对比分析**

与FP16及现有4-bit MX、QW4、W4A4KV4等基线相比，AdaMX在4.5b/4.25b位宽下分别将困惑度降低63–72%和55–62%，保持97% FP16常识准确率，MMLU保持95%；系统能耗仅增加≤1.1%，并可通过较大块尺寸降低2.9–5.2%能耗；

**⚠️ 局限性**

局限在于权重块需要离线搜索、激活块的2-bit路由无此能力，硬件增量仍存在（3–4%面积/功耗），且目前仅针对4-bit MX 结构，尚未验证更高位宽或其他量化框架的可迁移性。

---

## 641. SciRet: A Compute-Aware Empirical Study of Retrieval and Reranking for Scientific RAG

**arXiv ID:** 2608.03860 | [PDF](https://arxiv.org/pdf/2608.03860v1)

**作者:** Kaysarul Anas Apurba `[一作]` (Laurentian University), Asab Azad `[通讯]` (Laurentian University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在CORD-19上对检索增强生成（RAG）系统进行可复现的实验，比较不同规模（1K、5K、15K）下检索、rerank和生成的表现。

**💡 创新点**

提供系统级可重复评估框架，证明稀疏-稠密检索融合在科学QA中的优势，并揭示跨域MS MARCO reranker对科学文本的负面影响。

**🔧 技术方法**

使用BM25、BGE‑M3稠密检索、Reciprocal Rank Fusion、MS MARCO cross‑encoder reranker、GPT‑4o‑mini生成和RAGAS自动评估。

**📊 数据集**

基于CORD‑19标题和摘要的1K、5K、15K论文子集。

**📈 对比分析**

通过Recall@K和Precision@K对检索进行比较，RAGAS指标显示生成质量随规模提升；Hybrid检索在1K/15K规模下Recall@10达到1.000，reranker反而降低精确度。

**⚠️ 局限性**

仅索引标题摘要、使用伪相关性标签、查询样本仅15个、缺乏完整文档、缺少领域适配的reranker以及评估不含人类标注。

---

## 642. Sensitivity, Causality, and Repair Dissociate: A Layer-Wise Analysis of Perturbation Robustness and Its Scaling

**arXiv ID:** 2608.03842 | [PDF](https://arxiv.org/pdf/2608.03842v1)

**作者:** Nathan Labiosa `[一作]` (University of Southern California), Erica Donno `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了语言模型在面对表面扰动（如打字错误、OCR噪声、同音异义词等）时，各层级的失效机制，并发现敏感性、因果性和补偿能力三种层级“责任”指标不一致，提出了两种传播模式——峰值抑制与晚期累积，并通过“级联破坏”机制解释了这种分离。

**💡 创新点**

创新点在于首次系统性对三种层级责任指标（LRD、激活补丁、LoRA补偿）进行层级映射，揭示其互不相符的关系；引入级联破坏理论说明早期补丁为何会损伤后续计算，并验证了诊断引导的适配器放置反模式。

**🔧 技术方法**

主要技术包括层级表示发散（LRD）、激活补丁（activation patching）进行因果定位、LoRA低秩适配器用于补偿实验，以及固定评估主干（fixed harness）进行一致性检验。

**📊 数据集**

使用的数据集为GSM8K（数学推理）、MMLU（多项选择推理）和BBH（常识推理），并在五个不同规模的Transformer模型（Phi-3.5、Gemma-2-9B、Llama-3、Mistral、Qwen2.5-7B）上进行实验。

**📈 对比分析**

比较方法是对每一层/窗口训练LoRA并在固定主干上评估链式思考任务的准确率，发现最优放置往往为最深层，早期/中间层放置导致显著性能下降，平均提升仅为数个百分点，且在短形式任务上效果更弱。

**⚠️ 局限性**

局限性包括：实验模型仅为五个，未覆盖更广泛的架构与规模；任务主要集中在长文本链式思考，短文本多项选择效果有限；统计样本有限（每层窗口仅三次随机种子）；部分结果受评估主干长度限制的影响；级联破坏机制尚未在更大规模或多任务设置中验证。

---

## 643. Trajectory inference via Acceleration Matching

**arXiv ID:** 2608.03916 | [PDF](https://arxiv.org/pdf/2608.03916v1)

**作者:** Bartolo Dazzini `[一作]` (University of Padova), Aram-Alexandre Pooladian `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 Acceleration Matching 的算法，利用相位空间中的加速度场来进行多边缘轨迹推断，训练过程中完全不需要轨迹仿真，也无需昂贵的预处理。

**💡 创新点**

创新点在于将多边缘轨迹问题提升到相位空间，构造可学习的 Markov 加速度场，并通过条件期望得到显式回归目标，从而实现无仿真、低预处理成本的平滑轨迹生成。

**🔧 技术方法**

技术手段包括：Kinetic Brownian 轨道与相位空间 SDE、加速度匹配与条件加速度场、条件期望最小化、神经网络参数化、Euler–Maruyama 推断、以及对多边缘耦合的高斯条件采样。

**📊 数据集**

使用的数据集包括：低维的海洋流（GoM）与 Lotka–Volterra（LV）轨迹；以及高维单细胞数据的 PCA 表示，分别为 EB5、CITE5（5 维）和 CITE50（50 维）。

**📈 对比分析**

与 3MSBM、MMFM（const、van）、OT‑CFM、OT‑MFM 等方法进行对比。低维数据上，Acceleration Matching 在训练与 holdout 的 Wasserstein 误差上优于所有 MMFM 变体，但略逊于 3MSBM；高维 EB5 上表现与 OT‑CFM 相当，CITE5 上略优，CITE50 上相对性能较弱。

**⚠️ 局限性**

限制：需要先获得多边缘耦合，预处理成本对大样本或多时间点仍可能显著；对极高维度或极大时间点数量的可扩展性未充分验证；离散化误差和理论收敛性尚未给出完整分析。

---

## 644. ETA: A New Agentic Paradigm for Embodied Tasks

**arXiv ID:** 2608.03924 | [PDF](https://arxiv.org/pdf/2608.03924v1)

**作者:** Yitong Chen `[一作]` (Shanghai Innovation Institute), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Embodied Task Agent（ETA）范式并实现了开源框架 OpenETA，构建了数字代理与物理世界交互的闭环系统。

**💡 创新点**

创新点包括：1) 将规划器放在任务级闭环中心，确保每次只能执行一次可观测到的物理动作；2) 通过工具（Tool）和技能（Skill）实现可插拔的能力；3) 在 Agent–Interface–World 三层协议中加入可信证据、受控执行门控和可重放轨迹；4) 设计了自进化门控机制，确保经验更新先通过严格验证。

**🔧 技术方法**

使用技术：大型语言模型（GPT‑5.6 Luna/Terra/Sol）、多模态工具库（SAM‑3、AnyGrasp、GraspNet、AnyPlace 等）、OpenMoss 工具注册与调用框架、MCP（Simulator/Real‑Robot）接口、结构化命令与结果规范、实验记录与重放系统。

**📊 数据集**

使用数据集：LIBERO 物 Manipulation Benchmark（包括 Spatial、Object、Goal、LIBERO‑10、LIBERO‑90 套件）。

**📈 对比分析**

比较方法：在 LIBERO 上采用 Pass@k（k=1、5）指标评估；对比三种规划器（Luna、Terra、Sol）在同一工具集（Codex 的 3‑Tool 版）下的表现。实验显示 Sol 在 Pass@5 最高，分别为 XX%、YY%、ZZ%（具体数字请参见论文表格）。此外，Full OpenETA 在完整工具注册下在实验模拟中获得了若干成功案例。

**⚠️ 局限性**

限制：1) 主要失败为超时，子目标跟踪与预算管理不足；2) 放置、释放与附着关系仍是瓶颈；3) 缺少双臂、动态接触和移动操作的成熟工具；4) 仿真与真实机器人在安全、控制频率、校准等方面仍存在差距；5) 自进化门控未能提升任务成功率，缺乏可复制的性能提升。

---

## 645. ATLAS: Learning to Recommend Across Unseen Domains

**arXiv ID:** 2608.03899 | [PDF](https://arxiv.org/pdf/2608.03899v1)

**作者:** Pervez Shaik `[一作]` (Sony Research India), Niranjan Pedanekar `[通讯]` (Sony Research India)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在未见域零样本推荐的设定下，研究者提出 ATLAS 框架，通过在多源异构域上联合对齐用户与项目的语义与几何信息，学习一个统一且域不变的检索空间，随后在完全不进行目标域适配的情况下直接在未知域上完成推荐。

**💡 创新点**

创新点在于将 Gromov–Wasserstein 对齐与对抗式域判别相结合，同时采用残差向量量化（RVQ）构建离散码本，将连续表征压缩为共享的层级代码，从而实现对用户交互结构和项目语义的同时域不变学习。

**🔧 技术方法**

使用的技术包括：Sentence‑BERT 文本编码器、LightGCN 交互图编码器、对抗判别器与梯度反转层、GW 低估损失、残差向量量化、采样 Softmax 损失以及基于码本的离散化检索。

**📊 数据集**

实验使用亚马逊评论 2023 数据集的 15 个不同商品域，训练 5 个源域，随后在 10 个完全不重叠的目标域上评估零样本性能。

**📈 对比分析**

与顺序、图谱、跨域、通用预训练以及 LLM 推荐等基线对比，ATLAS 在大多数未见域上实现了 HR@10、NDCG@10 的平均相对提升约 24%，并在零样本场景中已超过多数传统和 LLM 基线。

**⚠️ 局限性**

局限性包括：对源域多样性高度依赖，若源域相似或数量不足则迁移效果显著下降；模型依赖文本编码器和码本的先行构建；在极端新颖域或与源域语义差距极大时的泛化仍有待提升。

---

## 646. The Hard-Core Model on Bipartite Spectral Expanders: Counting and Sampling at All Fugacities

**arXiv ID:** 2608.03848 | [PDF](https://arxiv.org/pdf/2608.03848v1)

**作者:** Ijay Narang `[一作]` (Georgia Institute of Technology), Will Perkins `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在Δ‑正则二部图上，满足谱扩张条件的硬核模型的近似计数与采样算法；给出了在所有活性系数λ>0下的FPRAS和高效采样器；

**💡 创新点**

创新点在于提出“离散二次局部化”技术，将左–右占用不平衡通过二次势场消除，得到可快速混合的Glauber动力学；以及利用谱谱值仅靠非平凡奇异值就能得到高活性系数下聚合模型的收敛性，从而实现所有活性系数的算法；

**🔧 技术方法**

核心技术包括：
1. 离散二次局部化与Hubbard–Stratonovich变换；
2. 谱独立性与依赖矩阵快速混合判据；
3. 聚合模型与Kotecký–Preiss收敛条件的谱分析；
4. 模拟退火与混合权重的离散高斯分布；

**📊 数据集**

无实验数据集，本文为理论分析和算法设计，主要以图论与统计力学模型为基础；

**📈 对比分析**

与之前的随机正则图算法相比，本文提供了一个可在任意实例上通过谱检验即可保证成功的判据；算法时间为多项式（对|V|和1/ε），在满足σ₂(M_G)≤c(Δ²/lnΔ)^{1/3}时即可在所有λ>0下实现；

**⚠️ 局限性**

局限性：需要满足谱扩张的条件，常数c与Δ_0未给出具体数值；在非正则或不满足谱条件的图上无法直接应用；算法在极小的Δ或近奇异谱值时性能未评估。

---

## 647. Why do we need social singularity? A mechanism-based critique of gradual scenarios in AI existential-risk discourse

**arXiv ID:** 2608.03904 | [PDF](https://arxiv.org/pdf/2608.03904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 648. PRISM: Powerful Time Series to Image (TS2I) Representations for Multivariate Anomaly Detection

**arXiv ID:** 2608.03926 | [PDF](https://arxiv.org/pdf/2608.03926v1)

**作者:** Mateusz Smendowski `[一作]` (AGH University of Krakow), Roberto Corizzo `[通讯]` (American University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PRISM框架，将多变量时间序列映射为多通道图像用于异常检测

**💡 创新点**

引入TS2I投影方案和MSM统计通道化策略，并证明其在多变量TSAD中的竞争性能

**🔧 技术方法**

使用TS2I变换（如GASF、MTF、波形变换等）、MSM通道化、卷积/ResNet自编码器和ImageNet预训练迁移学习

**📊 数据集**

在TSB-AD基准的14个多变量数据集上评估

**📈 对比分析**

与24种时间域基线对比，PRISM在10/14数据集上获得最高VUS-PR，平均提升41%，波形+MSM方案表现最佳

**⚠️ 局限性**

局限包括固定窗口长度、未探索图像分辨率、仅评估自编码器，且仍需进一步验证在实时部署与大规模数据上的可扩展性

---

## 649. When and Where to Look: Adaptive Visual Evidence Scheduling for Efficient Long Video Understanding

**arXiv ID:** 2608.03918 | [PDF](https://arxiv.org/pdf/2608.03918v1)

**作者:** Ke Li `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个训练‑free、低开销的查询自适应视觉证据调度框架 EcoFrame，利用 VLM 的输出熵来判断是否足够以及帧级注意力来定位后续搜索区域，从而在长视频问答任务中动态调节帧预算并提升推理效率。

**💡 创新点**

创新点包括：① 使用输出熵门控动态扩展帧预算，实现按查询需求自适应的证据增减；② 将帧级注意力转换为时间先验，用于高效的候选帧扩展；③ 仅依赖 VLM 内部信号，无需额外代理或多轮推理，显著降低调度开销。

**🔧 技术方法**

技术手段包括 VLM 输出熵估计、帧级注意力提取、注意力引导的候选池扩展、距离惩罚与重要性加权的增量选择、几何预算增速和阈值递增策略等。

**📊 数据集**

使用了 Video‑MME、LongVideoBench 和 MLVU 三个长视频问答基准数据集进行评估。

**📈 对比分析**

与统一采样、AKS、BOLT、FOCUS 等静态方法以及基于代理的 A.I.R 进行对比；EcoFrame 在 Qwen2.5‑VL 上平均准确率 64.4% 超过 BOLT（63.5%），速度提升 1.85×；与 A.I.R 比较，EcoFrame 在保持相似准确率的同时平均推理延迟提升 13.5×。

**⚠️ 局限性**

局限性在于对 VLM 输出熵与注意力的依赖，可能在极低置信度或注意力分散的情形下误判；对极长视频或极低资源环境下的实时性尚未充分验证；以及对多模态输入（如语音、文本多模态）的扩展仍需进一步研究。

---

## 650. Implementing Causal Perception: Competing SCMs and Situated Fairness

**arXiv ID:** 2608.03917 | [PDF](https://arxiv.org/pdf/2608.03917v1)

**作者:** Jose M. Álvarez `[一作]` `[通讯]`, Jose M. Álvarez

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了Álvarez与Ruggieri提出的因果感知框架，给出了单一和多次干预下的算法，并将其应用到实际信用评估问题中。

**💡 创新点**

首次将理论框架落地为可执行代码，提供了多种距离度量、聚合函数与阈值设置的实用指南，证明了“情境偏见”在多专家决策中的重要性。

**🔧 技术方法**

使用结构因果模型（SCM）与可加噪声模型（ANM），实现Pearl因果阶梯上的干预与逆因果推断；采用Wasserstein‑2、KL散度、总变差距离等统计距离；通过自举法估计置信区间。

**📊 数据集**

在德国信用（German Credit）数据集上进行实验，构建两个竞争SCM（一个包含性别→风险边，另一个不包含）。

**📈 对比分析**

与两个SCM的预测结果对比，发现准确率差异不大，但公平性指标（DP、EO）差异显著；不同距离度量对是否检测到因果感知的阈值敏感；展示了在不同阈值下感知判定的变化。

**⚠️ 局限性**

局限性：假设已知因果图且使用ANM；仅探讨单对SCM的竞争，未覆盖更大范围的干预集合；对计算复杂度和干预空间的扩展仍需研究。

---

## 651. ANNOTARES: A Dataset for Extracting Logical Structures from German Statutory Texts

**arXiv ID:** 2608.03898 | [PDF](https://arxiv.org/pdf/2608.03898v1)

**作者:** Ronja Schwarz `[一作]` (Karlsruhe University of Applied Sciences), Jannik Strötgen `[通讯]` (Karlsruhe University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了Annotares数据集并对德国法文本中的条件（Tatbestand）与结果（Rechtsfolge）进行序列标注；

**💡 创新点**

首次提出将立法文本逻辑拆分为法律条件与后果的标注任务，并构建跨法典的评估基准；

**🔧 技术方法**

使用规则基线、CRF、BiLSTM、BiLSTM-CRF、BERT变体（包括mBERT、LEGAL-BERT、DistilBERT、BERT-LER）以及Claude 3 LLM的提示式推理；

**📊 数据集**

包含BDSG（训练/验证/测试共896句）、BAföG和BauGB各50句的法典句子，累计21,550个标注词；

**📈 对比分析**

在BDSG测试集上，mBERT在准确率85.4%/宏观F1 0.798的水平领先，LLM提示虽在整体token准确率略低，却在none类的精确跨度匹配上表现最佳；

**⚠️ 局限性**

数据量有限导致传统CRF/BiLSTM模型泛化差，LLM在加入复杂句法信息时表现下降，且模型对“none”类的逻辑判别仍不够稳健。

---

## 652. DS@GT-ARC at eRisk 2026 Task 3: Sparse, Semantic, and LLM Reranking for ADHD Symptom Sentences

**arXiv ID:** 2608.03883 | [PDF](https://arxiv.org/pdf/2608.03883v1)

**作者:** David Guecha `[一作]` `[通讯]` (Georgia Institute of Technology), David Guecha (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出五种零监督检索/重排序方案用于eRisk 2026 ADHD症状句子排序任务，包含稀疏检索+证据重评分、Dense重排序、查询原型扩展和LLM重排序；

**💡 创新点**

创新点在于将自述式证据重评分与ASRS/Likert扩展查询原型结合，构造可解释的多阶段检索流水线，并首次在无标注ADHD数据上探索LLM级联重排序；

**🔧 技术方法**

使用BM25、MiniLM、BGE‑small、ASRS与Likert混合查询原型、Gemini LLM（OpenRouter）进行chunk‑级别重排序，并通过t‑SNE对嵌入空间进行可视化诊断；

**📊 数据集**

使用eRisk 2026 Task 3 Reddit句子集合（约417万句，4,521份用户文件）作为检索语料，未提供任何标注训练集；

**📈 对比分析**

通过majority和unanimity两种评判标准对MAP、P@10、nDCG@1000进行评估，Gemini LLM在majority评判下达到MAP 0.085、P@10 0.300、nDCG@1000 0.207，最高排名；原型MiniLM在无LLM情况下为最佳非LLM方案；

**⚠️ 局限性**

局限性包括无监督设定下缺乏精细标签、LLM重排序仅基于chunk级别非全局对比、对自述提示词过度依赖导致潜在隐含证据被忽略、缺少逐症状性能分析以及对复杂歧义症状的处理不足。

---

## 653. Evaluating MFU as a Proxy for GPU Power for Energy-Aware Simulation of LLM Training

**arXiv ID:** 2608.03880 | [PDF](https://arxiv.org/pdf/2608.03880v1)

**作者:** Niklas Enskat `[一作]` (Technische Universität Berlin), Philipp Wiesner `[通讯]` (Technische Universität Berlin)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对跨六款NVIDIA与AMD GPU的近3000个单机LLM训练实验进行基准，评估Model FLOPs Utilization（MFU）是否能作为可移植的GPU功耗预测代理。

**💡 创新点**

①证明在计算受限的LLM训练中，MFU与GPU功耗呈线性关系；②提出按GPU、数值精度和批量大小细分的单元化线性模型，误差降至≈1%，接近测量噪声；③指出在内存受限（如batch 1）或解码推理场景下MFU失效。

**🔧 技术方法**

采用跨架构基准框架；使用CalFLOPS测算算术操作数、NVML/ROCm获取硬件功耗与GPU Utilization；通过线性回归、MAPE、R²等指标评估MFU与功耗的关系；对不同模型家族、精度、批量、上下文长度进行实验。

**📊 数据集**

实验数据来自自定义synthetic workloads：单一英文prompt、不同批量大小（1–128）、上下文长度（512/2048）、三种数值精度（fp32/ fp16/ bf16）以及三种LLM家族（Qwen2.5、GPT‑2、DialoGPT）。不使用公开真实数据集。

**📈 对比分析**

通过比较MFU和GPU Utilization对功耗的线性预测，发现MFU在每台GPU上R²在0.68–0.84之间，MAPE为3.5%–13.7%；在每个（GPU、dtype、batch）单元化拟合后MAPE降至≈1%，几乎达到测量噪声极限；相比之下GPU Utilization仅在NVIDIA设备可拟合且精度略逊。

**⚠️ 局限性**

①仅覆盖单GPU训练，未考虑多GPU通信功耗；②在内存受限或解码推理（batch 1）场景下MFU失效；③未使用fused/SDPA注意力核，可能影响算术强度；④外部功耗验证仅限两款GPU，验证范围有限。

---

## 654. ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?

**arXiv ID:** 2608.03874 | [PDF](https://arxiv.org/pdf/2608.03874v1)

**作者:** Tianyi Guan `[一作]` (Institute for Artificial Intelligence, Peking University), Muhan Zhang `[通讯]` (Institute for Artificial Intelligence, Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了ContinualSkillBench，一个用于评估LLM代理在连续任务序列中自我进化技能的动态评价框架；

**💡 创新点**

创新点在于将任务按难度与技能依赖顺序化，构建跨域连贯子任务流，并系统比较显式技能维护与纯上下文学习的效果；

**🔧 技术方法**

采用LLM驱动的技能标注、图结构排序、以及基于Harbor的评测平台，配合Codex CLI和Claude Code工具链实现交互与反思；

**📊 数据集**

收集并筛选自五大领域（医疗、法律、数学、金融、办公）的100个子任务，来源包括OlympiadBench、LawBench、TAT-QA、GAIA、ClawBench等公开基准；

**📈 对比分析**

通过对比Independent、Sequential以及Pure ICL三种执行模式，发现Sequential在绝大多数模型–领域组合中提升了约16%（绝对提升0.071）奖励，GPT‑5.3‑Codex在绝对增益上领先，域内提升差异显著；

**⚠️ 局限性**

局限性包括：任务来源固定、缺乏长尾或分布漂移的真实环境、多模型与基础设施覆盖有限、以及高昂的API成本导致实验规模受限。

---

## 655. Bi-semantic Chemical Embedder for Joint Representation Learning of SMILES and Natural Language

**arXiv ID:** 2608.03855 | [PDF](https://arxiv.org/pdf/2608.03855v1)

**作者:** David Ming Segura `[一作]` (EPFL), Philippe Schwaller `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 CheMatE，一种将 SMILES 注入长文本并通过两阶段 MLM 与对比学习实现分子结构与科学文本双语义表征的 encoder；

**💡 创新点**

创新点在于结合 SMILES 注入管道、成本感知批次采样与 Matryoshka 多维对比损失，使单一模型在分子与科学文本两种语义上均表现出色；

**🔧 技术方法**

技术包括 ModernBERT 变体、Masked Language Modeling、Multiple Negative Ranking Loss (MNRL)、Matryoshka 对比学习、BalancedTokenBatchSampler 以及 SMILES 注入工具（CDE2、OPSIN、PubChem、RDKit）；

**📊 数据集**

使用 FineWeb、ChemPile、FineWeb-EDU 等 14.4M 文档进行 SMILES 注入，构建 21.9B 词的预训练语料，并从中抽取 20k anchor-positive 对进行对比训练；

**📈 对比分析**

在 48 个分子属性与 NLP 基准上与 11 个基线进行线性探针评估，CheMatE 在 SMILES 平均排名 1.9、NLP 3.2，bi‑semantic 分数 86.7%，在 43/48 任务中位列最佳组；

**⚠️ 局限性**

局限包括 SMILES 注入管道可能产生错误、对比学习样本量有限且阈值未系统探究，以及长上下文能力未在评测基准中充分验证。

---

## 656. LiteMVS: Efficient Multi-View Stereo with Foundation Distillation and Expert Aggregation

**arXiv ID:** 2608.03851 | [PDF](https://arxiv.org/pdf/2608.03851v1)

**作者:** Tianbao Zhang `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出LiteMVS，一种轻量级多视角立体模型，融合语义先验、Mixture‑of‑Experts聚合和基础模型蒸馏，实现实时高质量3D感知与重建。

**💡 创新点**

创新点包括：1) 将轻量化分割编码器产生的语义特征注入4D成本体；2) 引入Mixture‑of‑Experts（MoE）模块实现自适应深度聚合；3) 通过伪标签蒸馏从Vision Foundation Model获取相对深度与法线知识，无额外推理成本。

**🔧 技术方法**

采用轻量化CNN+MobileSAM进行语义编码，Plane‑Sweep视差栈构造4D特征体，MoE MLP实现成本聚合，伪标签蒸馏（Depth Anything v2、StableNormal），scale‑invariant 与法线损失进行训练，基于点云的重建评估。

**📊 数据集**

使用ScanNetv2、ScanNet++、7‑Scenes进行训练与评估；在LIBERO、RoboTwin 2.0基准上测试下游把手抓取任务。

**📈 对比分析**

与SimpleRecon、DoubleTake、MVDepthNet等轻量MVS方法对比，LiteMVS在深度误差和重建精度上均优于对手，同时推理延迟与SimpleRecon相近。下游把手抓取实验显示其几何表示与VGGT相当但推理速度最快。

**⚠️ 局限性**

局限性在于对反射、透明物体和低纹理区域等几何歧义场景仍存在误差；下游任务的表现尚未最优，需进一步提升泛化与鲁棒性。

---

## 657. MAFIA: Query-Only Memory Attacks via Probing and Factual Injection against Audited LLM Agents

**arXiv ID:** 2608.03844 | [PDF](https://arxiv.org/pdf/2608.03844v1)

**作者:** Jiaming Chen `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种仅通过查询即可进行的内存投毒攻击框架MAFIA，能够在存在大规模良好内存池和输入审计的实际部署环境下，使LLM代理的检索记忆被恶意重写并长期影响其决策。

**💡 创新点**

创新点在于：①使用检索探测+聚类+预算分配的放置策略，精准定位最易被检索到的语义空间；②引入紧凑的事实披露（Compact Factual Cloak）payload，既保持与受害查询的语义相似度，又能绕过基于提示的审计。

**🔧 技术方法**

核心技术包括：查询-仅攻击、检索探测（memory probing）、聚类与轮询式预算分配、顺序注入调度、事实披露payload设计、以及对比实验中的多种检索器与LLM主干。

**📊 数据集**

实验使用了四个公开基准数据集：MIMIC-III、eICU（医疗记录），WebShop（电子商务）和HuggingFace Hub Data Interpreter（代码/模型依赖），并在对应的EHRAgent、RAP、DataInterpreter代理上进行评估。

**📈 对比分析**

相较于基线MINJA等方法，MAFIA在包含5.8K+记忆条目与输入审计的严苛条件下，攻击成功率可达约90.7%，而审计检测率从83.3%降至仅7.4%；在不同LLM主干和检索器上保持高ISR/ASR，且对查询漂移、探测与聚类失配均表现稳健。

**⚠️ 局限性**

局限性包括：仅针对检索增强型（RAG）记忆模型，未覆盖图式或层次化长期记忆体系；评估的防御仅限于写时审计与检索后一致性检查，未涉及更系统级的权限控制、溯源与信息流控制等措施。

---

## 658. Low-Dimensional High-Leverage Subspace Optimization: Beyond Full-Parameter Coupled Training for Neural Network Quantization

**arXiv ID:** 2608.03919 | [PDF](https://arxiv.org/pdf/2608.03919v1)

**作者:** Peng Xia `[一作]` (Beijing University of Technology), Zheng Huang `[通讯]` (Beijing University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种归一化仿射预处理方法 NAP，仅调节 BN / RMSNorm 的尺度/偏置，冻结卷积权重，从而在低位量化前后提升网络量化友好性。

**💡 创新点**

创新点在于把归一化仿射参数视为高杠杆、低维子空间，并针对目标假量化图进行自适应；该方法既可在 PTQ 前预处理，又可在 QAT 后轻量调优，还可在 LLM 中交替优化响应与量化尺度。

**🔧 技术方法**

采用知识蒸馏、目标图对齐、局部投影分析、重构量化、RMSNorm‑NAP、QScale‑QAT 等技术实现低位量化的子空间优化。

**📊 数据集**

在 ImageNet、CIFAR‑100、Cityscapes、Qwen2.5‑3B‑Instruct 等数据集上进行实验验证。

**📈 对比分析**

与 RTN、QDrop、传统 QAT 方法对比，NAP 在 W4A4/W3A4 等极低位量化下恢复了 5–15 点准确率，在 LLM 上将 perplexity 从数万降低到 1–2 万，表现显著优于单纯量化或重构方法。

**⚠️ 局限性**

局限性包括对目标量化图高度依赖、需较大调优数据集、在已饱和 QAT 模型的部分情况可能产生干扰，且无法弥补严重的非线性截断误差。

---

## 659. UniEvo-RS: Omni-Prompt Unified Remote Sensing Segmentation with Representative Exemplar-Driven Prototype Evolution

**arXiv ID:** 2608.03911 | [PDF](https://arxiv.org/pdf/2608.03911v1)

**作者:** Kunquan Zhang `[一作]` (Sun Yat-sen University), Runmin Dong `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 UniEvo‑RS 的统一远程感知分割框架，支持文本与视觉提示的多种分割任务，并引入了无训练、基于代表样本的原型演化机制来提升批量注释精度。

**💡 创新点**

创新点包括：① 构建多指令统一数据集，将文本与视觉提示映射到共享 token 空间，实现动态任务路由；② 设计无训练的原型演化机制，将人工校正错误分为正负原型，并通过固定预算的 K‑Means 压缩；③ 在 LLM 预处理与 mask 解码阶段同时利用正负原型实现召回增强与背景抑制。

**🔧 技术方法**

技术手段包括：冻结预训练视觉 Transformer 提取多尺度特征；可训练视觉投影器将视觉提示映射至 LLM 嵌入空间；LLM 进行上下文化推理；共享 mask 解码器结合查询引导生成分割；代表样本原型记忆在推理时通过残差交叉注意力与空间抑制机制注入。

**📊 数据集**

使用四个公开遥感基准构建的多指令数据集：SIOR、iSAID、NWPU‑Refer 和 RRSIS‑D，涵盖文本提示的通用/指称分割以及视觉提示的交互式、一对多及跨图像分割。

**📈 对比分析**

与 SEEM、PSALM、SAM3、DINOv、YOLOE、X‑SAM、UniGeoSeg、RemoteSAM、SegEarth‑R2、OmniSegNet 等通用与遥感专用模型在九项评测指标（gIoU、cIoU、PQ、mIoU 等）上对比，UniEvo‑RS 在大多数任务中获得最优或相近表现，特别是在跨图像一对多、通用分割和交互式/指称分割的 cIoU 上实现了显著提升。

**⚠️ 局限性**

局限性包括：实验仅覆盖已选数据集与注释设置，跨传感器或跨域泛化尚未验证；多对多分割仅通过定性可视化评估，缺乏统一量化基准；原型记忆压缩参数对不同场景的鲁棒性需进一步探索。

---

## 660. When Efficiency Becomes Fragility: Exploiting Dynamic Routing Vulnerabilities in Adaptive UAV Tracking

**arXiv ID:** 2608.03902 | [PDF](https://arxiv.org/pdf/2608.03902v1)

**作者:** Shaofeng Liang `[一作]` (Hong Kong University of Science and Technology), Yutao Yue `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aaccfe5c-6b26-4208-b23c-35331481e142` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文针对无人机视觉跟踪中的自适应Transformer架构，揭示其动态路由的Lipschitz奇点导致的路径不连续性，并提出Adversarial Path‑Inversion (API)攻击框架，通过对门控模块的微小扰动实现推理路径翻转，从而严重破坏跟踪性能。

**💡 创新点**

创新点在于：①发现动态路由门控决策存在局部无界Lipschitz常数，构成新的攻击表面；②提出基于路径翻转的对抗攻击方法API，能同时操纵模型语义与推理拓扑；③设计门控感知扰动生成器和梯度目标，提升攻击隐蔽性与效率。

**🔧 技术方法**

技术包括：动态Transformer的门控模块分析、Lipschitz连续性理论、对抗扰动生成网络（编码器-解码器+SGPF），以及多目标损失（路径翻转、特征破坏、响应抑制、重建约束）。

**📊 数据集**

使用的数据集包括LaSOT（训练），以及六大无人机跟踪基准：DTB70、UAV123、UAVDT、VisDrone2018、UAVtrack112、UAV123@10fps，用于评估攻击效果。

**📈 对比分析**

与RTAA、UAP、FGSM等传统对抗方法对比，API在保持0.025扰动幅度下实现了精度/成功率的>80%下降，且推理速度高达62.5 FPS，显著优于其他方法的效果与效率。

**⚠️ 局限性**

局限性在于：攻击针对门控可逆的自适应Tracker，模型需具备明显的动态路由结构；对极少量路由块或已采用随机/软化门控的网络效果不佳；攻击生成器训练需要目标模型冻结并对齐噪声先验，增加了实施复杂度。

---

## 661. NCGR: Noise-Conditional Gated Rectification for Camera Extrinsic Perturbations in BEV 3D Object Detection

**arXiv ID:** 2608.03895 | [PDF](https://arxiv.org/pdf/2608.03895v1)

**作者:** Wenbin Pan `[一作]` (Guangdong University of Technology), Renquan Lu `[通讯]` (Guangdong University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 Noise‑Conditional Gated Rectification (NCGR) 来提升 BEVFormer 对相机外参扰动的鲁棒性。

**💡 创新点**

创新点在于在空间交叉注意力中加入可门控的二维残差校正，并通过教师‑学生 BEV 一致性约束在训练中学习无元数据的纠正策略。

**🔧 技术方法**

采用了卷积特征提取、DCNv2 变形卷积、SCA 机制、门控残差网络、BEV 软对齐损失以及时间一致性正则化等技术。

**📊 数据集**

使用 nuScenes 数据集，并在其上模拟旋转（±15°）和平移（±0.1 m）扰动进行实验。

**📈 对比分析**

与 BEVFormer 和 CAPE 进行对比，NCGR 在五相机动态扰动下 NDS 最高达 39.69%，mAP 最高达 35.0%，且在干净外参时保持与 BEVFormer 相近的性能。

**⚠️ 局限性**

主要局限在于仅在模拟扰动下验证，缺乏真实场景扰动的评估，且引入了轻量级的额外计算和参数开销。

---

## 662. Intertemporal Preference Steering in Qwen3 via Contrastive Activation Addition

**arXiv ID:** 2608.03892 | [PDF](https://arxiv.org/pdf/2608.03892v1)

**作者:** Michal Mráz `[一作]` (Independent), Justin Shenk `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Qwen3‑32B 的残差流进行线性时序偏好方向学习，并通过对比激活添加（CAA）实现对二元时间选择、货币时间‑效用任务及旅行规划能力的可逆调节。

**💡 创新点**

首次提出用短期/长期回答的均值差方向构造可解释的时序偏好向量，并证明该向量能在多种任务中产生大幅、可逆的行为改变。

**🔧 技术方法**

采用对比激活添加、线性探针（MM、LR、WLR 等）、激活补丁以及贪心或温度采样解码等技术。

**📊 数据集**

使用 500 条显式时间选择题 + 300 条隐式时间对照题训练探针；768 条货币延迟奖励题评估泛化；180 条旅行规划查询评估能力。

**📈 对比分析**

通过解析选择率、平均对数概率差和 AEIM 等指标评估 steering 效果；结果显示 CAA 在各层产生显著且符号一致的偏好转移，且在货币任务中 AEIM 变化可达 5‑56 倍，旅行规划微指标在中等正向 steering 下略有提升。

**⚠️ 局限性**

方向可能携带其他语义特征；大规模激活补丁可能破坏事实性或连贯性；实验仅限单一模型、静态提示、禁用思考，结果未必泛化到其他架构或更复杂提示。

---

## 663. MuRA: Multi-Rank Adaptation for Efficient and Effective Test-Time Vision-Language Generalization

**arXiv ID:** 2608.03885 | [PDF](https://arxiv.org/pdf/2608.03885v1)

**作者:** Gengyuan Liu `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MuRA框架，通过动态多秩LoRA模块与路由器实现基于token级视觉复杂度的测试时适配，解决传统静态秩配置导致的欠拟合与过拟合；

**💡 创新点**

核心创新包括多秩正交分解初始化(MROD)、统一组件融合与混合专家路由(UCF)，以及连续路由更新(CRU)，三者协同使模型在最深层高效自适应不同信息密度；

**🔧 技术方法**

采用CLIP预训练模型、LoRA轻量模块、SVD正交分解、Mixture‑of‑Experts路由、Entropy‑based TTA损失以及AdamW优化；

**📊 数据集**

在ImageNet及其变体（ImageNet‑A、V2、R、S）、十个跨域基准（Aircraft、Caltech、Cars、DTD、EuroSAT、Flower、Food、Pets、SUN、UCF101）以及ResNet‑50与ViT‑L/14骨干上进行评估；

**📈 对比分析**

与零样本CLIP、输入/输出/知识适配方法（TPT、DiffTPT、PromptAlign、ADTE、TTL、TDA、MTA、GS‑Bias、TT‑RAA）对比，MuRA在OOD和跨域平均精度均超过对手，最高平均提升约6–8%，同时吞吐量提升、内存占用降低；

**⚠️ 局限性**

局限包括需手动设定秩边界、仅使用最小熵目标可能导致过度自信、未在自回归大型语言模型上验证、未自动决定秩范围以及对极端分布适应性仍有提升空间。

---

## 664. Operationally Feasible Synthetic Power-Grid Scenarios via Learning the AC-Operable Joint Distribution

**arXiv ID:** 2608.03878 | [PDF](https://arxiv.org/pdf/2608.03878v1)

**作者:** Chenhan Xiao `[一作]` (Arizona State University), Yang Weng `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于层次扩散模型的可操作合成电网场景生成框架，直接学习AC可行联合分布；

**💡 创新点**

创新点在于：① 将网络拓扑、线路参数、负荷动态分层建模，减少维度；② 在扩散学习过程中加入AC潮流收敛和约束违规的奖励信号，实现可行性引导；③ 无需后置优化即可生成满足AC约束的场景；

**🔧 技术方法**

技术方法包括图Beta扩散模型、图Transformer、LSTM自编码器、强化学习式奖励引导与分层生成；

**📊 数据集**

使用IEEE 14/118/123及欧洲36路由分布式系统的真实负荷时间序列作为训练数据；

**📈 对比分析**

与StatAlign、GNN-Gen、PowerGrow等电网生成方法以及GDSS、GruM、GBD等通用图扩散模型进行对比，实验显示该方法在AC收敛率、可行性得分、N‑1鲁棒性、MMD统计一致性和生成速度方面均显著优于基线；

**⚠️ 局限性**

局限性包括：仅在中小型网络上验证，未覆盖大规模系统的模块化生成；对可再生/储能等动态能源的建模仍不完善；训练时需大量潮流评估，计算成本较高。

---

## 665. From population norms to personalized trajectories: interpretable Bayesian forecasting for cognitive decline

**arXiv ID:** 2608.03877 | [PDF](https://arxiv.org/pdf/2608.03877v1)

**作者:** Maria Sahakyan `[一作]` (UiT Arctic University of Norway), Brita Elvevåg `[通讯]` (UiT Arctic University of Norway)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了 PRISM 框架，用可解释的机器学习与贝叶斯序列更新相结合，实时监测并预测个体认知衰退。

**💡 创新点**

创新点在于：① 利用 Explainable Boosting Machine 生成个体化基线；② 用贝叶斯更新与时间衰减实现对随访数据的递进学习；③ 通过概率阈值与个体化锚点结合的三层检测规则，实现早期、解释性与不确定性兼备的衰退识别。

**🔧 技术方法**

主要技术包括 Explainable Boosting Machine (EBM)、贝叶斯正态更新、时间指数衰减权重、贝叶斯线性回归估计下降概率、三层阈值检测规则，及与传统 LME 模型的 AUC 比较。

**📊 数据集**

使用了两大纵向数据集：美国全国代表性的 Health and Retirement Study (HRS) 约30,000 名老年人，和临床研究型 Alzheimer's Disease Neuroimaging Initiative (ADNI) 1,866 名受试者。

**📈 对比分析**

性能评估：与人口标准（demographic norm）和个体平均（cumulative average）基线相比，PRISM 的单步预测 MAE 更低；与 LME 模型相比，PRISM 在每访视 AUC 上表现更好，尤其在随访早期。早期检测率：HRS 中 31% 的持续衰退者在转诊前一次访视被识别，68% 在转诊时被识别；ADNI 中分别为 41% 和 56%。

**⚠️ 局限性**

局限性包括：HRS 的认知测量与判定标准部分重叠；仅监测了短时记忆（词汇回忆）且未覆盖其他认知领域；时间衰减参数统一全局优化，未考虑个体差异；检测规则需手动调优，可能受阈值设置影响；外部验证仍受样本规模和诊断标准差异限制；未来需在临床实际环境中进行前瞻性验证。

---

## 666. EvoHIL: Self-Evolving Reward and Flow-Matched Policy Optimization for Robust Human-in-the-Loop Reinforcement Learning

**arXiv ID:** 2608.03872 | [PDF](https://arxiv.org/pdf/2608.03872v1)

**作者:** Shuoqin Zhang `[一作]` (Chongqing University), Kai Liu `[通讯]` (Chongqing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在机器人对人类的实时交互学习中，构建了一个自适应框架（EvoHIL），可在部署后持续更新奖励模型、动作生成方式和视觉域，提升任务成功率、动作平滑度和对光照变化的鲁棒性。

**💡 创新点**

创新点包括：①自演化奖励学习，仅利用人类确认的正样本与弱负样本，避免自标记错误；②通过流匹配生成动作块并用执行前缀评估，显著降低命令不连贯性；③采用保留意识的离线视觉重光（relight）微调，兼顾源域保持和新域适应。

**🔧 技术方法**

技术实现涵盖：强化学习（SAC+AFS）、流匹配（FPO‑style surrogate）、执行前缀批评家、保留正则化、离线 relit 重新标记、EMA 部署门控、奖励自适应与标签源隔离。

**📊 数据集**

数据集为六项真实机器人操作任务（Franka FR3 的 RAM 插入、USB 插入、擦桌、断路器操作；SO‑101 的糖果推送与药盒收纳），配合人类确认标签、干预示例和通过外部重光工具生成的光照变化记录。

**📈 对比分析**

与 HIL‑SERL、HG‑DAgger、BC、IBRL、ACT 等基线对比，EvoHIL 在 60% 光照变化条件下平均成功率提升至 93–100%，完成时间缩短 15–45%，干预率显著下降，表明在多种场景与机器人体系上均显著优于现有方法。

**⚠️ 局限性**

主要局限包括：奖励一致性评估仅基于确认代理，未验证真实任务成功率；Relight 重光假设几何保持且受限于外部渲染质量；离线重光更新不一定能实现零样本泛化；与基线的对比缺乏完整资源匹配（如重光计算开销）。

---

## 667. ADMITBench: A Safety-Governed Reference Framework for Evaluating the Admissibility of Industrial LLM Advisories

**arXiv ID:** 2608.03866 | [PDF](https://arxiv.org/pdf/2608.03866v1)

**作者:** Yash Misra `[一作]` (Refiant, Inc.), Mehmet Mercangöz `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ADMITBench，一个基于版本化、安全治理的框架，用来评估工业 LLM 产生的建议行动是否可行、合法、可验证、且安全。

**💡 创新点**

创新点在于将评估焦点从自然语言答案转向结构化行动记录，采用非补偿性 T0–T4 硬门检查，并为每个门提供可重放的安全合约；同时提供可插拔的工厂配置（cartridge）实现跨工艺的可迁移性。

**🔧 技术方法**

技术手段包括：1) 统一的行动记录模式与解析器；2) 版本化的评估合同与安全案例图；3) 分层的可执行检查（T0–T4）与排名（T5）以及审计追踪（T6）；4) 对接模拟器进行动态后果验证。

**📊 数据集**

数据集主要是两个手工构建的工厂配置：1) 连续搅拌釜（CSTR）和 2) 蒸馏柱；每个配置包含情景包、过程状态、警报、证据与程序规则；未来计划将扩展至 Tennessee Eastman 过程进行可迁移性验证。

**📈 对比分析**

比较方法强调可复现的硬门通过率、第一失败层、完整失败清单与可用性排名；报告聚焦于层级统计与置信度校准，而非单一分数；在现有版本中未给出数值性能指标，但已提供基准案例与评估报告模板。

**⚠️ 局限性**

局限性包括：1) 仅覆盖两种工艺，缺乏广泛的工业通用性；2) 无公开的 hold‑out 或对抗性测试集；3) 评估结果不等同功能安全认证，需要现场工程审查；4) 结果高度依赖工厂配置质量，若配置错误会产生误判。

---

## 668. Sparse Weight Decomposition for Efficient Circuit Extraction

**arXiv ID:** 2608.03913 | [PDF](https://arxiv.org/pdf/2608.03913v1)

**作者:** Chuanhao Yan `[一作]` (IQuest Research), Jie Fu `[通讯]` (IQuest Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对预训练的线性投影进行稀疏权重分解，得到可直接用于电路抽取的稀疏瓶颈单元

**💡 创新点**

在不训练额外替代网络的前提下，通过稀疏因子分解实现高保真度与低数据需求，并提供零数据变体

**🔧 技术方法**

使用双稀疏分解（DSF）与 ADMM 算法进行稀疏因子学习，结合校准激活或权重矩阵直接优化

**📊 数据集**

在 GPT‑2 Small、Qwen2.5（0.5B–3B）和 Qwen3.5‑27B 等模型上使用 FineWeb‑Edu 进行校准与评估

**📈 对比分析**

与 Transcoder、VPD 及稀疏预训练等基线在交叉熵、足够性/必要性阈值和活跃边数上比较，SWD 在保持相同替代保真度时使用更少的训练数据、边数更少、性能更优

**⚠️ 局限性**

局部性与非唯一性限制；需依赖校准数据或可变性高；在大规模模型和不同任务上的验证仍有限，无法提供完整的机制解释

---

## 669. StreamDAM: Presence-Aware Memory for Real-Time Streaming Video Object Segmentation

**arXiv ID:** 2608.03912 | [PDF](https://arxiv.org/pdf/2608.03912v1)

**作者:** Xiang Chen `[一作]` `[通讯]` (Shanghai Jiaotong University), Xiang Chen (Shanghai Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

重构质量级视频目标分割(VOS)的内存流水线，使其能够在30fps实时时钟下以帧率级别执行，并通过一个学习的存在感信号动态控制内存管理与输出；

**💡 创新点**

首次在模型内部实现完整的帧率级内存运算并引入单一的因果存在感估计器，以实现对内存接纳、递归窗口、输出抑制和重检测的统一控制，显著消除原本的实时崩溃与缺失感知问题；

**🔧 技术方法**

利用DAM4SAM与SAM2的空间时间内存网络；CUDA图捕获、GPU内存同步移除、掩码字节打包、尺寸上限的干扰器内省；通过一个小型GRU学习存在感并配合阈值化策略实现自适应控制；

**📊 数据集**

DAVIS-2017 val、LVOS val、VOST val以及从MOSE train挑选的60个难度高的MOSE-hard序列；

**📈 对比分析**

使用VOT/ VOTS实时协议，在单GPU下对30fps的实时性能进行评估，并与EdgeTAM、EfficientTAM、Cutie、SAMURAI以及vanilla SAM2.1-L等五个现代基线进行对比。自研方法在四个基准上平均得分最高，尤其在MOSE-hard和LVOS上领先，且在MOSE-hard上甚至超越离线模型，近乎恢复了97%的离线精度；

**⚠️ 局限性**

存在编译随机性导致的种子波动、仅评估单目标性能、超大尺寸序列仍可能突破预算、存在感信号无法完美覆盖所有困难序列、MOSE-hard为内部子集，结果可能不具普适性。

---

## 670. Structured-Sparsity-Aware Joint User Activity Detection and Channel Estimation for OTFS-Based Grant-Free Random Access

**arXiv ID:** 2608.03896 | [PDF](https://arxiv.org/pdf/2608.03896v1)

**作者:** Yao Ge `[一作]` (Nanyang Technological University), Zhi Ding `[通讯]` (University of California at Davis)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在高移动性多机器类型通信场景下，提出一种基于 OTFS 的无授予随机接入（GFRA）框架，联合实现用户活动检测与信道估计。

**💡 创新点**

创新点在于：①通过低维基函数展开模型（BEM）将时变信道映射到延迟-多普勒域；②识别并利用双层结构稀疏性（跨天线共同稀疏与用户活动稀疏）；③构建两层因子图并提出结构稀疏期望传播（SS‑EP）算法，实现对结构稀疏先验的高效 Bayesian 推理。

**🔧 技术方法**

采用的技术包括：OTFS 调制与反变换、基函数展开模型、稀疏压缩感知、因子图消息传递、期望传播（EP）与高斯近似。

**📊 数据集**

使用的仿真数据集：5G TDL‑B 通道模型（指数功率延迟曲线），设置 4 GHz 载频、15 kHz 子载波间隔、M = 32、N = 16、L = 512、K = 200 潜在用户、K_a = 10 活跃用户、U = 2 天线、最高速度 150 km/h。

**📈 对比分析**

与 OAMP、SOMP、AMP‑MMV、S‑GAMP 及 Oracle MMSE 基线进行对比。结果显示：SS‑EP 在不同 SNR、用户活动概率与天线数量下均优于其它算法，逼近 Oracle MMSE；性能随 SNR 提升、活动概率降低、天线增多而提升。

**⚠️ 局限性**

局限性包括：算法的计算复杂度较高（尤其是高阶 BEM 时），仿真仅基于标准通道模型，未验证在实际移动环境中的鲁棒性；此外，天线数、用户数的扩展性和硬件实现的可行性仍待进一步研究。

---

## 671. CARE-X: Towards Clinically Useful Radiology VLMs with Auxiliary Supervision, Reward-Aligned Learning, and Tool-Augmented Measurement

**arXiv ID:** 2608.03890 | [PDF](https://arxiv.org/pdf/2608.03890v1)

**作者:** Mercy Prasanna Ranjit `[一作]` (Microsoft Research India), Tanuja Ganu `[通讯]` (Microsoft Research India)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CARE-X系统，将生成式胸片报告与可调阈值分类、空间定位和定量测量统一在同一模型内；同时使用Qwen3‑VL‑4B‑Instruct与工具调用实现测量依赖诊断。

**💡 创新点**

三大创新：① 辅助监督与奖励对齐的联合训练，让判别头与生成器互相强化；② DAPO强化学习将临床质量指标直接映射到奖励；③ 通过工具调用实现视觉-定量协同，解决传统VLM无法完成的定量诊断。

**🔧 技术方法**

采用多模态结构：SigLip2‑so400M视觉编码器 + Phi‑4‑mini‑instruct 生成器；辅以焦点分类头、复合损失定位头；奖励信号使用BERTScore、RadGraph、GREEN、VQA准确度、mIoU等；工具链基于CXAS分割+测量+阈值判断；训练分为SFT + DAPO。

**📊 数据集**

公开胸片基准（MIMIC‑CXR、IU‑Xray、CheXpert‑Plus、ReXGradient），ReXVQA VQA；工具测量基准（Cardiomegaly、Mediastinal Widening、Aortic Enlargement等）；临床评估使用印度Narayana Health（NH）ICU罕见病案例与CT验证的增大病例。

**📈 对比分析**

与MedGemma、MedVersa、CheXOne-R1等最新VLM相比，CARE‑X在报告生成四大基准上获得26/32指标领先；ReXVQA闭合VQA准确率达94.0%（比CheXOne高+6pp）；定位mAP提升5–8pp并逼近判别头；工具测量模式平均F1提升43.6pp。

**⚠️ 局限性**

局限：工具调用依赖CXAS分割质量、未在VLM内部集成工具、奖励函数仍是近似临床质量、实验为回顾性，缺乏前瞻性临床验证。

---

## 672. BanglaWild: An In-the-Wild Bengali Scene Text Recognition Benchmark for OCR and Vision-Language Models

**arXiv ID:** 2608.03884 | [PDF](https://arxiv.org/pdf/2608.03884v1)

**作者:** Sadab Shiper `[一作]` (Islamic University of Technology), Eshat Tanzeem `[通讯]` (BRAC University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 BanglaWild 基准，收集并标注了 2,535 张真实世界的孟加拉语场景文本图像，并在同一数据集上同时评估 15 个通用视觉语言模型（VLM）、3 个传统 OCR 系统、以及 6 个 LoRA 微调模型。

**💡 创新点**

创新点包括：① 仅在孟加拉语的真实场景文本上提供 verbatim/standard 双重注释，能够区分视觉误读与拼写纠正错误；② 设计了 15 类错误分类法，揭示视觉识别错误是主导失误；③ 引入 LLM-as-a-Judge 的评估方式，用语义判定补充编辑距离指标。

**🔧 技术方法**

技术方法：零样本 VLM 提示（3 种提示策略）、LoRA 参数高效微调、传统 OCR（Tesseract、EasyOCR、Surya）对比、15 类错误自动化分类器、LLM 判断器（Gemini 2.5 Pro 及 Claude Sonnet 4.6）

**📊 数据集**

数据集：BanglaWild，涵盖 2,535 张公开场景图像，配有 Context、Content 两个类别标签、四个诊断属性（Curved Text、Font Style、Occlusion、Background Complexity）以及当图像拼写偏离标准时的正字法标准形式。

**📈 对比分析**

比较方法：在统一的 1,014 张测试集上，使用 CER、WER、1‑NED 以及 LLM 判定得分进行评估。结果显示：模型规模不一定带来性能提升；视觉误识率占大约 60% 的错误，正字法错误不到 2%；LoRA 微调能显著减少最差模型的灾难性失败，但对已具备较好表现的模型提升有限。

**⚠️ 局限性**

局限性：仅覆盖孟加拉语；剔除了代码混合图像；类别分布不均衡；LoRA 仅在约 1,268 张样本上训练，未检验大规模微调效果；LLM 判断器可能存在家族偏置；错误分类仅对部分模型执行，结果可能不完全可迁移。

---

## 673. Enhancing VLM Reward Models Through Structure-Aware Fine-Tuning

**arXiv ID:** 2608.03875 | [PDF](https://arxiv.org/pdf/2608.03875v1)

**作者:** Pyrros Koussios `[一作]` (ETH Zürich), Andreas Krause `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种在线自监督细调方法（Structure‑Aware Fine‑Tuning，SAFT），通过在冻结的视觉‑语言模型上插入 LoRA 模块，利用任务固有的结构先验（如对称性和奖励的 Lipschitz 连续性）来去噪并完善 VLM 生成的奖励信号。

**💡 创新点**

创新点在于：①不需要任何人工标签或偏好数据，完全自监督地改进奖励模型；②将对比增强损失（CAL）和奖励 Lipschitz 正则化（RLR）两种结构先验结合使用；③通过 LoRA 只微调极少量参数，实现高效在线细调。

**🔧 技术方法**

技术方法包括：CLIP‑style 视觉‑语言模型、LoRA 低秩适配器、对比增强损失、奖励 Lipschitz 正则化、在线强化学习与奖励模型自监督细调。

**📊 数据集**

实验数据集涵盖经典控制环境（CartPole、MountainCar）和机器人操作环境（Isaac Lab 的 Reach、ReposeCube），均采用 VLM 生成的目标描述作为奖励信号。

**📈 对比分析**

在与基线 VLM、goal‑baseline 规则化以及真实奖励对照的对比实验中，SAFT 在样本效率、收敛速度和奖励对齐（EPIC 距离）方面均优于基线，并且在不使用人类偏好标注的情况下，节省数百至数千次二进制比较。

**⚠️ 局限性**

局限性包括：①需针对每个环境手工设计对比增强和窗口大小等超参数，缺乏通用自动化；②文本编码器保持冻结，未充分利用文本信息；③在奖励模型与任务本质严重不匹配时，结构先验仍难以弥补错误；④对高维噪声环境中状态距离的依赖可能不稳健。

---

## 674. CRS-Triage: Confidence- and Reliability-Aware Selective Triage under Incomplete Clinical Evidence

**arXiv ID:** 2608.03862 | [PDF](https://arxiv.org/pdf/2608.03862v1)

**作者:** Guan Qiang `[一作]` (Western University), Fang Fang `[通讯]` (Western University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种可信度与可靠性感知的选择性急诊分诊模型CRS‑Triage；

**💡 创新点**

通过联合评估结构化数据与文本数据的可靠性、跨模态一致性以及预测不确定性，动态权重融合并生成置信度决定是否给出诊断；

**🔧 技术方法**

使用贝叶斯证据学习（Dirichlet分布）、可学习的特征权重、跨模态Jensen–Shannon散度、以及置信度阈值选择机制，结合强化的under‑triage惩罚与自适应选择损失；

**📊 数据集**

在MIMIC‑IV‑ED（418,100例成人急诊病例）上训练与评估；

**📈 对比分析**

与多种基线（单模态、早/晚/门控融合、DrFuse、TMC、SelectiveNet）对比，CRS‑Triage在宏观F1、kappa、下调下分诊率、校准误差与风险-覆盖曲线均表现最优；

**⚠️ 局限性**

主要局限在于对不同临床数据分布变化的可靠性估计仍需改进，且置信度阈值和under/over分诊惩罚比例需要手工设定；

---

## 675. Complexity of induced subgraph isomorphism and maximum common induced subgraph parameterized by cluster vertex deletion number

**arXiv ID:** 2608.03845 | [PDF](https://arxiv.org/pdf/2608.03845v1)

**作者:** Tomohiro Koana `[一作]` (University of Tokyo), Yota Otachi `[通讯]` (Nagoya University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本篇论文针对两图的诱导子图同构（ISI）与最大公共诱导子图（MCIS）问题，提出了以两图的集群顶点删除数（cluster‑vertex‑deletion number）之和为参数的固定耗时算法，针对ISI实现了O*(2^O(k))的算法，针对MCIS实现了O*(2^O(k^2))的算法，并给出了对应的ETH下限证明，表明算法几乎达到最优；

**💡 创新点**

创新点在于将ISI/MCIS问题的求解转化为“加权精确多色匹配（Weighted Exact Multicolored Matching）”问题，并通过巧妙的分割与映射构造保证两图的结构对应，从而利用已有的多色匹配算法实现对诱导子图问题的求解；同时给出了ETH下限，证明在参数化为两图大小之和的情况下，MCIS无法实现O*(2^o(k^2))的算法；

**🔧 技术方法**

主要技术包括图的集群顶点删除数分解、变量与子句的块划分、行/列约束的线性可满足性（LCBM）到图问题的归约、精确多色匹配的随机化指数算法、以及签名（signature）论证以决定顶点的可行映射；

**📊 数据集**

由于论文为理论研究，未使用具体实验数据集；

**📈 对比分析**

与方法比较：论文中仅通过理论复杂度对比，ISI算法达到O*(2^O(k))，MCIS算法达到O*(2^O(k^2))，并给出对应的ETH下限，说明在此参数化下性能已接近最优；

**⚠️ 局限性**

局限性在于：对MCIS必须显式猜测集群顶点删除集的连接方式，导致算法复杂度提升到k^2级；此外，方法仅适用于集群图模型，无法直接推广到更一般图结构；

---

## 676. Uplifting the Superpowers of Worst-Case-Optimal Join Algorithms

**arXiv ID:** 2608.03840 | [PDF](https://arxiv.org/pdf/2608.03840v1)

**作者:** Adrián Gómez-Brandón `[一作]` (Universidade da Coruña), Gonzalo Navarro `[通讯]` (University of Chile)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于最坏情况最优（wco）连接的succinct索引PG‑Ring，支持属性图和GQL查询，并将过滤器直接集成到LTJ算法中。

**💡 创新点**

首次将过滤条件以leaptor形式嵌入wco Join，实现预筛选与后筛选的最优代价，并在Ring索引中加入属性和标签的succinct表示。

**🔧 技术方法**

采用波形树、位向量、正交范围后继查询、Leapfrog Trie Join以及leaptor抽象，并将这些技术扩展到Ring索引。

**📊 数据集**

在LSQB（约3.55亿节点、2.19亿边）和Wikidata（1.09亿节点、7.195亿边）两大属性图上进行实验。

**📈 对比分析**

与Neo4j、DuckDB、Umbra、Kùzu、Memgraph、MillDB等主流系统对比，PG‑Ring在空间上接近最优，时间上对循环查询超越多数系统，整体实现良好的空间‑时间折中。

**⚠️ 局限性**

目前仅支持只读，更新需动态波形树/位向量实现，且对复杂数据类型的支持有限。

---

## 677. Semantic Bundling: Interactive Node and Edge Bundling to Simplify Knowledge Graphs using Large Language Models

**arXiv ID:** 2608.04002 | [PDF](https://arxiv.org/pdf/2608.04002v1)

**作者:** Adam Coscia `[一作]` (Georgia Institute of Technology), Alex Endert `[通讯]` (Georgia Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Semantic Bundling 技术，利用 LLM 对知识图进行交互式压缩与摘要，构建 AgentK 系统实现可视化与检索

**💡 创新点**

将 LLM 直接嵌入图形操作流程，生成可解释的 super node 与 super edge，并保持与原始三元组和源文档的可追溯链

**🔧 技术方法**

大语言模型（GPT‑5.2）、句子与交叉编码器（sentence‑encoder、cross‑encoder）、网络X、D3/ Vue 前端、Flask 后端、Graph Deduplication（SemHash）

**📊 数据集**

IMDb 电影评论数据集（Interstellar 前 400 条评论）和 KRONOS 军情情报案例数据集（800+ 新闻文章）

**📈 对比分析**

通过两条使用案例展示功能，未进行系统性量化对比；与传统图聚合、边捆绑、主题建模等方法的差异仅在概念层面进行讨论，未给出准确指标或性能数值

**⚠️ 局限性**

缺乏定量评估与用户研究、对 LLM 提取误差的影响未充分分析、可扩展性（大规模图、动态更新）不明、操作历史和多 KG 跨图功能缺失、LLM 定义本体导致的稳定性与精度折衷

---

## 678. Stochastic Multiple Shooting Trajectory Optimization via Sequential Local Policy Evaluation

**arXiv ID:** 2608.03978 | [PDF](https://arxiv.org/pdf/2608.03978v1)

**作者:** Ashwin Gupta `[一作]`, Joseph Moore `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

阐述了如何使用会议模板，包括排版规范、标题、章节、公式、表格、图形等的格式要求。

**💡 创新点**

模板的创新点在于将排版细节统一化，自动满足电子化出版与会议论文集的一致性需求。

**🔧 技术方法**

主要技术是 LaTeX 排版，使用.cls 文件、Times New Roman、Symbol 字体以及预设的样式表。

**📊 数据集**

无数据集，本文仅为格式规范说明。

**📈 对比分析**

无方法比较或性能评估，本文不涉及实验或算法实现。

**⚠️ 局限性**

限制在于仅适用于符合模板的会议或期刊，无法直接用于发表实际研究内容，且对非标准排版需求支持有限。

---

## 679. ReflectRL: Learning from Golden Negative Trajectories via Reflective-to-Direct Reasoning

**arXiv ID:** 2608.03972 | [PDF](https://arxiv.org/pdf/2608.03972v1)

**作者:** Jinhe Bi `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ReflectRL，一种轻量级框架，通过在 on‑policy 训练（RLVR 与 OPD）中利用专家失败（Golden Negative Trajectories, GNT）进行反思式推理，提升 LLM 的推理能力。

**💡 创新点**

创新点：① 将传统上被丢弃的专家失败轨迹重新利用为有效学习信号；② 设计反思式推理接口与直接推理接口，并通过 Reflective‑to‑Direct Policy Transition 将反思能力迁移到推理过程；③ 在不改动原始目标或增加额外损失的情况下实现这一迁移，保持训练和推理一致性。

**🔧 技术方法**

技术：反思式提示（Chat_Temp_R）、直接提示（Chat_Temp_D）、群体相对策略优化（GRPO）、On‑Policy Distillation（OPD）、余弦衰减的策略转换核（g(t))、Process Reward Model (PRM)、MathVerify 验证器。

**📊 数据集**

数据集：OpenR1‑GNT‑69k（从 DeepSeek‑R1 生成的 69k 失败轨迹）、9 个推理基准（AIME 2024/25、AMC、MATH‑500、Minerva、Olympiad、ARC‑c、GPQA‑Diamond、MMLU‑Pro）。

**📈 对比分析**

对比方法：GRPO、DAPO、EchoRL、OPD 以及先前的零‑RL 方法（Prime‑Zero、SimpleRL‑Zero、OpenReasoner）和监督学习（SFT、SFT‑KL）。实验显示 ReflectRL 在 Qwen2.5‑Math‑7B 及 LLaMA‑3.1‑8B 等模型上均提升 3–10% 甚至 20%（如 GRPO AIME 22.2→25.5，DAPO AIME 22.9→31.0），同时推理长度更短、训练效率更高、策略熵保持更高。

**⚠️ 局限性**

局限性：依赖预先生成的专家失败轨迹，若专家模型表现不佳或任务非数学推理可能效果不佳；在训练时需要 GNT 作为教师端特权上下文，推理时无法直接使用；对极端复杂问题或跨域推理的泛化仍需进一步验证。

---

## 680. HalluTruthQA-4K: A Fine-Grained Corpus and Annotation Process for Arabic Hallucination Detection and Truth Verification

**arXiv ID:** 2608.03966 | [PDF](https://arxiv.org/pdf/2608.03966v1)

**作者:** Salah Eddine Bekhouche `[一作]` (University of the Basque Country), Abdenour Hadid `[通讯]` (Universiti Malaysia Kelantan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了 HalluTruthQA-4K，一个包含 4,000 条阿拉伯语知识密集型问答实例的细粒度幻觉检测资源，包含响应级标签、字符级错误跨度、人工解释、层次化错误类型以及六选项的事实验证。

**💡 创新点**

创新点：① 将响应级幻觉检测、错误跨度定位、层次化分类、人工解释和多项选择验证统一在同一数据集；② 采用专家手工设计问题、参考答案和错误标注，并提供人类可读的错误解释；③ 设计了两级宏微错误分类体系，覆盖事实冲突、证据不符、逻辑不一致、虚构内容及无意义回答。

**🔧 技术方法**

技术手段：使用单一阿拉伯语 LLM（Fanar‑1‑9B‑Instruct）在受控配置下生成回答；专家与验证者双轮标注和独立复核；字符级跨度标注与误差解释；多选候选答案手工构造；统计分析与交叉验证。

**📊 数据集**

数据集：4,000 个问答实例，分布在伊斯兰知识、历史、科学与地理四个领域（各 1,000 条）；每条实例包含问题、模型生成答案、专家核对参考答案、六个候选答案（1 正确+5 干扰项），并对幻觉回答提供错误跨度、解释与错误类型。

**📈 对比分析**

比较方法与性能：资源已作为 2026 年 HalluScoring 共享任务 Track 2 的官方测试集，支持对不同系统在响应级检测、跨度定位、错误分类、解释生成与事实选择等多任务的统一评估；虽然本文未给出具体模型分数，但已提供完整评测基准与公开数据，方便后续系统对比。

**⚠️ 局限性**

局限性：① 仅使用单一 LLM 生成回答，缺乏多模型多架构的多样性；② 只覆盖四个领域，未覆盖阿拉伯语方言、口语及更广泛知识体系；③ 事实验证以六选项形式进行，无法完整模拟开放式纠错；④ 误差标注受人工主观因素影响，尽管通过双轮验证降低误差，但仍有不确定性。

---

## 681. A game theory for foundation models shows new paths to rational cooperation through similarity inference

**arXiv ID:** 2608.03958 | [PDF](https://arxiv.org/pdf/2608.03958v1)

**作者:** Alexander Meulemans `[一作]` (Google), Blaise Agüera y Arcas `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于大型语言模型的“理性基础模型代理”，通过构建联合预测与规划的嵌入式贝叶斯代理模型，展示其在多代理环境中能够通过相似性推理实现合作，并揭示其行为与经典博弈论预测的差异。

**💡 创新点**

创新点在于：①将语言模型的自我建模能力与贝叶斯规划结合，形成嵌入式贝叶斯代理；②提出嵌入式均衡概念，弥补经典博弈论对嵌入式代理的描述缺失；③发现相似性推理是现代 AI 代理实现合作的核心机制。

**🔧 技术方法**

技术上采用 Gemini 与 Gemma 语言模型的链式思维与显式规划框架，构建信息收集-终局囚徒困境的两阶段实验环境，并用 Bayesian mixture universe 进行理论建模与证明。

**📊 数据集**

数据集使用随机生成的矩阵游戏 payoff 表和固定策略的 NPC（合作或背叛），信息收集阶段 T 轮随机游戏；无公开标准数据集。

**📈 对比分析**

通过与经典博弈论预测（全背叛）以及与随机代理对比，评估平均合作率、预测相似性和 AUC；实验结果显示，信息收集阶段长度增加时，代理的合作率显著高于经典预测，在直接与间接相似推理环境中均表现出稳健合作。

**⚠️ 局限性**

局限性包括：实验仅限于简化的囚徒困境设置，未验证在更复杂开放式情境的适用性；链式思维推理可能偏离严格贝叶斯推断；对人类- AI 混合社会的风险与对策尚未深入探讨。

---

## 682. Muon Meets Mamba: Spectral Optimization for State Space Models

**arXiv ID:** 2608.03941 | [PDF](https://arxiv.org/pdf/2608.03941v1)

**作者:** Arslan Battalov `[一作]` (HSE University), Sofia Sinitsina `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在结构化状态空间模型Mamba‑2上系统评估MuOn优化器的不同分配方案，并比较其与AdamW基准在多预算、多数据集下的训练表现。

**💡 创新点**

首次在Mamba‑2中验证MuOn仅应用于输出投影即能显著提升训练效率和验证损失，并通过谱诊断揭示MuOn对矩阵条件数的局部作用，表明投影矩阵的光滑性对性能贡献更大。

**🔧 技术方法**

使用MuOn（Newton–Schulz正交化迭代）与AdamW混合优化，进行多种分配（仅输入投影、仅输出投影、两者都用）并配合学习率调优；对训练过程进行谱指标（条件数、有效秩、谱范数）和子块分解分析。

**📊 数据集**

OpenWebText和FineWeb‑Edu两个公开英文语言建模语料库，分别在10⁹、2.6×10⁹和5×10¹⁰ tokens的训练预算上进行实验。

**📈 对比分析**

在同一随机种子、固定学习率下对四种MuOn分配方案进行对照，评估最终验证损失、困惑度、token效率；结果显示k=3（仅输出投影）在各预算下均优于AdamW，验证损失降低约0.1–0.12，困惑度下降3–7%；token效率提升明显，尤其在单轮训练中，k=3可用约58% tokens达到AdamW在1B tokens下的损失；但在更大规模训练和下游零样本任务上优势并未转化为显著的准确率提升。

**⚠️ 局限性**

主要局限在于：1) 优势主要体现在token效率而非最终loss提升；2) 下游任务性能相近，未证明能带来实质收益；3) 机制分析基于单一seed的谱诊断，缺乏因果证明；4) 实验仅覆盖Mamba‑2 130M模型，未检验更大规模或其他SSM模型的泛化。

---

## 683. Assessment of Conditional Diffusion Model for Synthetic Histopathology Image Generation

**arXiv ID:** 2608.03990 | [PDF](https://arxiv.org/pdf/2608.03990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 684. WorldCup Arena: Prospective, Leakage-Free Evaluation of Frontier LLMs on a Live Tournament

**arXiv ID:** 2608.04008 | [PDF](https://arxiv.org/pdf/2608.04008v1)

**作者:** Zhenran Wang `[一作]`, Zhangyang Qi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对2026年FIFA世界杯进行实时前瞻性评估，六大LLM在每场比赛前预测七个盘口，覆盖104场赛事。

**💡 创新点**

设计完全无泄漏的前瞻性评估协议，连续刷新赛事资料并即时问答，公开完整数据和评分脚本。

**🔧 技术方法**

采用大语言模型（Claude、GPT、Gemini、Kimi、GLM、Seed）在最高思考设置下结合各自服务端检索，统一模板生成JSON答案。

**📊 数据集**

48支队伍每日Markdown档案、46个时间戳快照、赛事日程与官方最终比分。

**📈 对比分析**

通过七盘口加权计分生成总分，利用bootstrap和五种评分设计评估排名，六系统得分相近，最高897点，最少813点，差距仅10%。

**⚠️ 局限性**

受限于单一赛事、评测仅在世界杯范围内，且模型表现受投注线追踪而非超越，未能显著突破市场基准。

---

## 685. PAST-Bench: Benchmarking the Foundations of Recursive Self-Improvement in Personal Agents

**arXiv ID:** 2608.04003 | [PDF](https://arxiv.org/pdf/2608.04003v1)

**作者:** Shuhan Xue `[一作]`, Ling Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了名为PAST-Bench的在线自进化性能归因基准，评估个人AI代理在跨会话中对经验的保存、检索与应用效果。

**💡 创新点**

创新点在于引入配对的 persistence‑on/off 评估与轨迹级诊断，四个能力维度（记忆、程序重用、信息检索、更新）并结合机制证据实现因果归因。

**🔧 技术方法**

利用 Hermes 代理框架扩展5个运行时机制（计划检查、记忆绑定、技能生命周期、检索门控、关闭刷新），并在多大语言模型与代理框架上执行实验。

**📊 数据集**

使用26个人工构造的任务家族，共204个任务，覆盖四个能力维度，并在七种大型语言模型与四个代理框架上进行评测。

**📈 对比分析**

通过比较 persistence‑on 与 persistence‑off 的自进化增益 Δ 及机制证据分数，对模型与框架性能进行对比。平均增益从 +0.13 提升至 +0.15，机制证据从 0.64 提升至 0.73，尤其在更新能力上提升显著。

**⚠️ 局限性**

局限性包括：任务合成场景过于人工，缺乏跨家族迁移与长期追踪；机制归因基于一致性而非必然性；未加入人类交互数据；无法覆盖更强递归改进的能力。

---

## 686. Perceptual Anchoring: Prototype-Guided Text Calibration for Training-free Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2608.03991 | [PDF](https://arxiv.org/pdf/2608.03991v1)

**作者:** Wanli Ma `[一作]` (Huazhong University of Science and Technology), Xinge You `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构造视觉原型并校准文本嵌入，改进无训练的开放词汇语义分割模型。

**💡 创新点**

引入基于可靠视觉证据的原型引导文本校准（PTC），弥补文本嵌入与实例视觉表示的语义鸿沟。

**🔧 技术方法**

利用CLIP视觉-文本对齐、分数边际可靠性评估、混合证据量策略与自适应校准权重，实现无训练的文本嵌入校准。

**📊 数据集**

在八个公开基准（VOC21、Context60、Object、VOC20、Cityscapes、Context59、ADE20k-150、COCO-Stuff）上评测。

**📈 对比分析**

与六种主流无训练OVSS方法（如SCLIP、ClearCLIP、NACLIP、ResCLIP、ProxyCLIP、CorrCLIP）以及多模型基线对比，平均mIoU提升1–3个百分点，显著提升完整性和准确度。

**⚠️ 局限性**

性能仍受视觉原型构造质量限制，在复杂场景下原型不够可靠，导致校准效果有限；对不同数据集需要手动调参。

---

## 687. UniWorld-Design: From Pixel Generation to Layer-Native Design

**arXiv ID:** 2608.03971 | [PDF](https://arxiv.org/pdf/2608.03971v1)

**作者:** Zongjian Li `[一作]`, Li Yuan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个以语义RGBA层为原生单位的框架，包含 Text-to-RGBA 生成器和 Image-to-Layer 分解器，实现从文本生成可复用RGBA资产以及从完整图像按语义层级分解为可编辑的层堆叠。

**💡 创新点**

创新点在于将RGBA层视为本地可编辑的结构化对象，提出可按指令控制的分解机制，支持顶层分解、递归细化和目标提取，并通过语言可寻址的层状态为外部设计代理提供持久可操作的对象。

**🔧 技术方法**

采用流匹配的 LIB-MMDiT 模型，结合层-指令绑定注意力与层索引的 3D 旋转位置编码；使用 Progressive Distillation 与 DiffusionNFT 后训练，并构建了对 RGB 对齐的 RGBA 自编码器。

**📊 数据集**

训练数据包括内部设计资产与透明RGBA图像的文本配对集，以及从 PSD 文件生成的语义层树；评测使用 Crello 设计集合和内部 Hold‑out 文本-图像对。

**📈 对比分析**

通过与 Qwen-Image-Layered、LayerDiffuse、OmniAlpha 的对比，I2L 在每层 RGB L1 降低 37%、Alpha Soft IoU 提升 34%、Blank 层率下降 63%，CLIP Score 提升至 20.43；T2RGBA 在 CLIP Score 上取得 33.03，Alpha MSE、SAD、LPIPS 等指标均优于对手。

**⚠️ 局限性**

主要局限在于 alpha 边缘清晰度不足，密集排版（尤其中文文字）生成缺失笔画或字符错误，且在复杂排版设计中表现有限。

---

## 688. Latent Reward Registers for Diffusion Preference Alignment

**arXiv ID:** 2608.03929 | [PDF](https://arxiv.org/pdf/2608.03929v1)

**作者:** Yuanshen Guan `[一作]` (Kling Team), Peiqin Sun `[通讯]` (Kling Team)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Latent Reward Registers，在冻结的Diffusion Transformer上通过可学习的注册令牌从中间噪声潜变量直接估计终点偏好得分，并利用此稠密奖励信号实现训练时的Reward-Gradient On-Policy Distillation（RG‑OPD）和推理时的Reward‑Guided Sampling（RGS）两种对齐策略。

**💡 创新点**

创新点：①无需解码即可在高噪声阶段获取准确的偏好排名；②通过注册令牌保持生成器原始速度场不变，仅在侧流中学习奖励；③RG‑OPD实现了一步近似梯度蒸馏，避免了长链roll‑out与高方差；④RGS在推理时无参数更新即可引导多目标生成。

**🔧 技术方法**

核心技术：冻结Diffusion Transformer（DiT）+可学习位置无关注册令牌 + 轻量级注意力融合 + 阈值化pairwise‑ranking损失 + 目标归一化（RMS）+ 近似梯度蒸馏 + magnitude‑matched奖励梯度采样。

**📊 数据集**

主要使用公开的文本‑图像生成数据集：SD3‑Medium（Stable Diffusion v3）与FLUX.1‑dev，评估时对比ImageReward、HPSv2/HPSv3、PickScore等终点奖励模型以及MUSIQ/CLIP‑IQA等无参考感知指标。

**📈 对比分析**

与reward back‑propagation（ReFL、AlignProp）和在线RL（Flow‑GRPO、Diffusion‑NFT）以及训练‑free采样（CFG、DNO、Demon）比较。RG‑OPD在GPU时长上比RL快14–33×，同时获得更高的HPSv3与ImageReward分数；RGS在不更新参数的前提下，达到或超过现有训练‑free方法的对齐效果，同时保持更好的感知质量。

**⚠️ 局限性**

局限性：①仍依赖冻结的生成器架构，难以跨模型迁移；②奖励注册对高噪声下的通用性尚未在多模态/多任务场景验证；③对极端高分辨率或不同数据分布的稳健性待进一步评估；④多目标权衡需要手工调节γ_i和多头归一化策略。

---

## 689. ParVL: Parallel Scaling and Expandable Compute Allocation for Multimodal LLMs

**arXiv ID:** 2608.04010 | [PDF](https://arxiv.org/pdf/2608.04010v1)

**作者:** Yang Yang `[一作]` (Australian National University), Wenwei Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出ParVL框架，通过共享ViT与LLM骨干并行引入前缀参数，扩展多模态大语言模型的计算能力，并系统研究视觉与语言计算在不同分支配置下的分配关系。

**💡 创新点**

创新点在于首次将并行参数共享用于MLLM计算扩展，并将计算分配问题形式化为二维视觉‑语言分配，探究任务依赖的最佳分配策略。

**🔧 技术方法**

采用的技术包括前缀条件注意力、分支特定KV前缀、token‑wise MLP聚合、全参数监督微调、并行计算与可选稀疏路由、共享KV缓存等。

**📊 数据集**

使用InternVL3.5 SFT 13B tokens的1/20子集进行训练，评估9个多模态基准（MMMU、MathVista、MathVision、WeMath、LogicVista、ChartQA、TextVQA、DocVQA、OCRBench）。

**📈 对比分析**

与同配方单分支基线对比，在1B/2B/8B规模上平均分数提升0.9–1.5点，且不同任务对视觉或语言分支的需求不一致，取得最佳分配配置；与公开模型对照表现更优。

**⚠️ 局限性**

限制在于仅针对InternVL3.5 1B/2B/8B规模、仅做SFT且使用有限子集数据，未探索预训练、其他backbone、更大规模以及真实部署环境下的性能与适用性。

---

## 690. SocietyBench: Forecasting Counterfactual Social-World Evolution

**arXiv ID:** 2608.04009 | [PDF](https://arxiv.org/pdf/2608.04009v1)

**作者:** Zhenran Wang `[一作]`, Zhangyang Qi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 SocietyBench benchmark，通过自动化流水线将真实社会事件匿名化成对照事件，并评估 LLM 与 agent 在事件预测中的概率校准与时间准确度。

**💡 创新点**

创新点在于：①将事实报道与公众情绪双源时间线结合；②采用两维校准/时间评分并以 100 级刻度统一；③通过实体替换和日期偏移实现对记忆的去噪；④支持双语（中英）同步发布，防止互相去匿名化。

**🔧 技术方法**

使用技术包括：LLM 与 agent 框架（如 LangGraph、AutoGen、MiroFish）、实体/日期匿名化流程、问答银行生成、加权平均绝对误差（MAE）与时间误差归一化评分、自动化评测流水线。

**📊 数据集**

数据集为五个真实社会事件（公共争议、地缘政治、科技政策、金融市场、贸易政策），每个事件融合 Web 新闻与五大社交媒体，生成 39–168 节点的匿名时间线，并拆分为 125 个预测点。

**📈 对比分析**

比较方法：在两语言版上评测 6 大 LLM、3 个 agent、2 个非 LLM 基线，计算概率校准与时间准确度两轴平均分；最高分 75.0（GPT‑5.5），agent 未能提升基线，非 LLM 基线明显落后。

**⚠️ 局限性**

局限性包括：事件数量有限导致单事件差距大、匿名化仍可能泄漏信息、仅覆盖两维评分而非更细粒度预测指标、未评估不同模型对不同事件类型的细微差异。

---

## 691. TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning

**arXiv ID:** 2608.04007 | [PDF](https://arxiv.org/pdf/2608.04007v1)

**作者:** Changle Qu `[一作]` (Renmin University of China), Jun Xu `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为TurnSight的框架，通过执行条件的多视角回顾自我蒸馏，为多轮工具集成推理（TIR）提供精细的信用分配。

**💡 创新点**

创新点包括：①在执行状态对齐的基础上，将回顾信号聚合到交互层；②构造多lookahead教师并通过跨视角方向一致性来选择最可靠的监督；③利用组归一化标准化信号，并通过有界符号感知权重调节RL优势，既保持原有优化方向，又实现细粒度信用分配。

**🔧 技术方法**

使用的技术包括：on-policy自我蒸馏、GRPO策略梯度、组归一化、跨lookahead教师选择、信号归一化和有界调节等。

**📊 数据集**

训练数据采用FTRL数据集；评估基准为FTRL（在域）、BFCL和ToolHop（跨域）。

**📈 对比分析**

与Vanilla、GRPO、ToolRL、MatchTIR以及SDPO、RLSD、SDAR、SOD等多种基线进行对比，结果显示在三大基准上均取得最佳或第二佳成绩。特别是在BFCL的长上下文和参数缺失子集上，8B模型平均提升约7.7%，显著优于现有方法。

**⚠️ 局限性**

局限性：需要人工调优超参数（λ、ε_w），对参考策略的冻结设定敏感；依赖工具执行反馈的质量；仅在已知工具集合上验证，跨工具迁移及极长交互序列的鲁棒性尚未充分探究。

---

## 692. Calibrating Trustworthiness: Co-Designing Metrics and Visualizations for Evaluating LLMs in Education

**arXiv ID:** 2608.04006 | [PDF](https://arxiv.org/pdf/2608.04006v1)

**作者:** Adam Coscia `[一作]` (Georgia Institute of Technology), Alex Endert `[通讯]` (Georgia Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过与学习工程师的长期协作，共同构建了适用于教育场景的LLM可信度评价框架，并设计了可视化工具帮助工程师快速识别LLM回应中的可信度违规点；随后在真实课堂对话数据上进行LLM提示模板的评估实验，验证了该框架能提升专家一致性并揭示教学风险。

**💡 创新点**

①将ML领域的可信度维度（如真值性、外交性、亲和性等）与教育特定指标（学习方法、响应质量等）融合，形成5大可信度度量（共20项）；②采用协同设计方法获得实用可视化原型；③通过提示竞赛评估可信度度量对人类评估行为的影响，证明其在教育LLM评估中的价值。

**🔧 技术方法**

技术手段包括：①基于文本生成和事实校验的主张抽取与文本蕴含（使用DeBERTa-v3-large）；②LLM‑as‑Judge 模型对学习方法进行判定；③自定义二元违规判定算法；④多种可视化技术（指标概要、饼图、表格、文本重叠高亮等）；④采用OpenAI GPT和Llama3 生成提示模板响应；⑤采用Krippendorff’s alpha等统计方法评估一致性。

**📊 数据集**

使用来自“智能数字教材”真实课堂部署的30条学习者对话样本，生成5种提示模板的响应共150条，随后挑选30对比样本进行提示竞赛；所有数据均为受保护的学生交流记录，实验遵循IRB授权。

**📈 对比分析**

在两种界面条件（无可信度度量 vs. 伴随可信度可视化）下进行匹配评估；对比分析每位评估者的选项、决策时间、违规次数与最终提示获胜率；结果显示：①可信度度量可视化显著提升Krippendorff’s alpha（从0.32提升至0.56）；②使用可视化的评估者在每轮决策上平均多花2.3秒，但在识别关键违规（如幻觉、情绪不当）上更准确；③通过统计置信区间和bootstrap检验，验证了显著性差异；总体而言，可信度度量与可视化提高了评估的一致性和对教学风险的敏感度。

**⚠️ 局限性**

局限性包括：①度量仅采用二元违规判定，缺乏细粒度评分；②当前度量与可视化高度针对本研究的LLM模型和教育场景，需重新校准以迁移至其他任务或更大模型；③实验规模仅12位学习工程师，可能不足以泛化至更广泛的评估者群体；④可信度度量无法完全自动化，仍需人工判断；⑤可视化复杂度需平衡用户认知负荷，未来可进一步简化与个性化。

---

## 693. Transfer Learning for Avian Bioacoustics under Sparse Positive Labels

**arXiv ID:** 2608.03977 | [PDF](https://arxiv.org/pdf/2608.03977v1)

**作者:** Dhyey Patel `[一作]` (Eastern Michigan University), Yunting Yin `[通讯]` (Eastern Michigan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究在稀疏正标签下，利用生态先验、预训练音频表示、标签共现建模以及多源可靠性框架，将外部生物声学数据迁移到BirdCLEF+ 2026 基准上进行鸟类识别。

**💡 创新点**

创新点在于把不同来源的外部数据视为具有不同可靠度的弱正标签源，提出多源可靠性框架并对正负标签不平衡采用PU学习，从而降低负迁移，并通过生态信息与标签共现显著提升排名性能。

**🔧 技术方法**

技术主要包括：生态先验权重、Mel 频谱与轻量化声学特征、AST/AudioSet 等预训练嵌入、PPMI共现图、正负样本权重的 PU 损失、以及基于多源可靠性的 L2 正则化逻辑回归元学习器。

**📊 数据集**

使用的主要数据集有：目标 BirdCLEF+ 2026（公开验证集稀疏标签）、BirdCLEF 2021、iNatSounds、WABAD、BirdSet（PER 与 NES）作为外部迁移源。

**📈 对比分析**

与单源或简单拼接方式比较，采用多源可靠性框架的模型在宏平均 AP 上达到 0.584（相对基线 0.555 提升 0.029），宏 AUC 达 0.860（相对 0.832 提升 0.028），并在 micro F1 及序列校准方面亦有明显提升。

**⚠️ 局限性**

局限性包括：对外部数据的可靠性评估仍依赖名称匹配与采样，生态信息的可用性受限；负迁移虽被缓解但在标签覆盖率极低的类别仍难以提升；模型对标签共现的依赖可能导致过拟合；未尝试更复杂的时序模型（如 LSTM/Transformer）进一步提升时序信息。

---

## 694. Should We Type or Talk to LLM Agents? A Comprehensive Study of Voice and Keyboard Input Perturbations

**arXiv ID:** 2608.03970 | [PDF](https://arxiv.org/pdf/2608.03970v1)

**作者:** Zizhao Hu `[一作]` (University of Southern California), Jesse Thomason `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HIVE 评估工具，模拟键盘和语音输入的真实错误，对多种 LLM 进行鲁棒性测试。

**💡 创新点**

创新点在于：①统一的 17 种文本扰动（键盘误操作 + 语音转写扰动）和 2 种对照；②发现“原始 token 的破坏率”是性能下降的主因；③证明语音转写重写层是最致命的扰动来源，且轻量化适配无法修复；④思考（reasoning）块能几乎完全补偿键盘误差。

**🔧 技术方法**

技术：文本到文本的扰动生成（基于规则和 Qwen2.5-7B 的风格迁移）；模型评估采用 5 种 instruction‑tuned LLM，使用 6 个公开基准；自监督蒸馏、对照实验与 token‑survival 相关性分析。

**📊 数据集**

数据集：GSM8K、GSM‑Symbolic、GSM1k（算术）、HumanEval（代码）、MMLU‑Pro（学术）、TruthfulQA（多项选择）。

**📈 对比分析**

比较方法：对每个扰动与清洁输入进行同一条目对比，统计平均准确率下降（百分点）。结果显示：语音扰动平均损失约 9.7pp，键盘约 3.0pp；在需要推理生成答案的任务上，语音比键盘差 6–7pp；多项选择任务无显著差异。思考模块几乎消除键盘误差，却对语音扰动几乎无效。

**⚠️ 局限性**

局限性：仅限英语、QWERTY 键盘、合成语音（未涉及真实 ASR 错误）、单轮问答、未覆盖多轮或工具调用场景、仅测试 7–14B 开源模型。

---

## 695. Logic Before Language: Pre-pretraining on Formal Derivations Fosters Skill Acquisition and Compressibility

**arXiv ID:** 2608.03930 | [PDF](https://arxiv.org/pdf/2608.03930v1)

**作者:** Jo-Ku Cheng `[一作]` (University of Sheffield), Marco Valentino `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造形式逻辑推导数据集，对Transformer进行预预训练（Logic‑PPT），随后再进行常规自然语言预训练，探索逻辑推理如何加速语言学习并改变模型内部表征。

**💡 创新点**

创新点在于：①将形式逻辑推导（包含命题、术语、谓词逻辑）作为更富结构、表达力更强的预预训练任务；②在大规模（100B token）训练下发现预预训练能够显著降低模型表征秩、聚焦更少的主导方向；③利用这一结构化表征实现约33%稀疏剪枝仍保持密集模型性能，显示出压缩鲁棒性。

**🔧 技术方法**

技术包括：Qwen3 14‑层Transformer、后向链推导生成、结构保留的符号重命名、next‑step 预测任务、交叉熵训练、CKA/RankMe/谱衰减分析、Wanda权重剪枝。

**📊 数据集**

数据集：①247条形式逻辑规则库，自动生成推导树（2.5B token 预预训练数据）；②FineWeb‑Edu 约100B 纯文本进行自然语言预训练；③BLiMP 与 17 项 Elemental Tasks 用于评估语言与技能获取。

**📈 对比分析**

与随机初始化、算法式预预训练（Set/Sort/Union）及形式语言预预训练（Dyck/ShuffleDyck）对比。结果显示：逻辑预预训练使语言模型困惑度在 100B token 训练结束时比基线低0.97%；在 17 项技能任务上平均准确率为 87.4%，比基线高 7.1%；在 33% 稀疏度剪枝时保持密集模型相同的性能，其他方法在 40% 剪枝时性能下降更严重。

**⚠️ 局限性**

局限性：仅在 254M 参数 Qwen 模型上验证；实验使用单一随机种子，虽在 10B token 复验中显示稳定性，但未覆盖多种 seed；评估集中在预训练阶段，未检验微调后在下游任务的提升；仅覆盖英文数据，缺乏多语言验证；资源限制导致对更大模型或更长训练的探索不足。

---

## 696. A Physics-Flavored Transformer Network for Parametrizing Contraction Dynamics of Engineered Skeletal Muscle Tissues

**arXiv ID:** 2608.03927 | [PDF](https://arxiv.org/pdf/2608.03927v1)

**作者:** Mattias Luber `[一作]` (Campus-Institute Data Science), Timo Betz `[通讯]` (University of Göttingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种 Physics‑Flavored Neural Network (PFNN)，通过将张力-时间曲线的拉伸指数物理模型嵌入 CNN‑Transformer 架构，实现了对工程化骨骼肌收缩动力学参数的自动提取。

**💡 创新点**

创新点在于：① 将物理模型同时用作合成数据生成器和可微的 PINN 损失函数；② 采用混合半监督训练，将无标签实验数据通过自监督物理一致性对模型进行微调；③ 通过可解释的梯度归因验证网络已内化物理规律。

**🔧 技术方法**

使用技术包括：CNN‑Transformer 结合位置编码、可微拉伸指数模型、物理信息神经网络 (PINN)、集成梯度可解释性、Optuna 超参搜索、L‑BFGS 传统拟合比较、混合自监督训练框架。

**📊 数据集**

数据集：① 主训练集 370 条四秒张力曲线（LHCN‑M2），50 fps；② 生成 300 k 条合成曲线；③ 3 条不同来源细胞系（AB1167、KM571、KM670）的保留测试集，用于验证泛化。

**📈 对比分析**

与传统 L‑BFGS 拟合相比，PFNN 在验证集上的均方误差从 0.000417 降至 0.000017，混合训练进一步提升 2.5 倍；在三种细胞系上同样保持低误差并优于 L‑BFGS，表明模型具备更高精度与泛化能力。

**⚠️ 局限性**

局限性包括：Transformer 的 O(L²) 计算复杂度限制长序列；当前仅适用于固定 50 fps 采样率，需改进为时间戳编码或线性注意力；模型对极端噪声或未覆盖的生理变异仍有潜在偏差。

---

## 697. Agogic: Performance-Timed Music Tokens for LLM-Native Text-to-Symbolic-Music Generation

**arXiv ID:** 2608.03999 | [PDF](https://arxiv.org/pdf/2608.03999v1)

**作者:** Junhao Chen `[一作]` (Tsinghua University), Ruqi Huang `[通讯]` (Tsinghua University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个文本到音乐的语言模型，重点研究音乐的标记化方式对生成质量的影响。

**💡 创新点**

创新点在于提出了一种性能分辨率的参数化标记化方法（PMT），该方法能够保留微观时序和每个音符的动态特性，从而显著提高生成音乐的质量。

**🔧 技术方法**

使用了预训练的Qwen3.5模型，并通过固定的背骨、数据集和预算，比较了七种不同的标记化方式。

**📊 数据集**

使用了两个数据集：一个包含86,598个真实音乐片段的四模态对齐数据集，以及一个包含6.25M标注的音乐片段的扩展数据集。

**📈 对比分析**

通过与其他标记化方法（如Beat-TSD、REMI、MIDI-Like等）进行比较，PMT在Fréchet音乐距离（FMD）上表现出色，0.8B的PMT模型在FMD上达到159，而27B的Beat-TSD模型则为272，显示出PMT在生成质量上的优势。

**⚠️ 局限性**

限制在于：1) 近重复的分割可能影响结果；2) 音乐生成的可听性尚未经过人类研究验证；3) 在大规模模型中种子覆盖不足；4) PMT在音调遵循方面较弱；5) 控制研究中使用的数据集规模有限。

---

## 698. When AI Wears Many Hats: The Role of Generative Artificial Intelligence in Marketing Education

**arXiv ID:** 2608.03973 | [PDF](https://arxiv.org/pdf/2608.03973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 699. When Attention Goes Blind: Numerical Failure in ALiBi Positional Encodings

**arXiv ID:** 2608.03994 | [PDF](https://arxiv.org/pdf/2608.03994v1)

**作者:** Christopher Schröder `[一作]` (Institute for Applied Informatics at Leipzig University), Gerhard Heyer `[通讯]` (Institute for Applied Informatics at Leipzig University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过数学分析和实验，首次揭示 ALiBi 位置编码在浮点精度下的欠流失效模式，并在预训练模型和从零开始训练的 148 M 参数解码器上验证该问题；进一步提出并评估了四种训练时的缓解策略，证明其对检索任务的提升。

**💡 创新点**

创新点在于：① 从理论层面阐释 ALiBi 线性偏置导致 softmax 欠流，导致注意力头“失明”；② 对该问题在多种模型与数据集上的实证研究；③ 提出四种可组合的缓解方案（clamping、robust slopes、log‑scaled distances、soft capping），并系统评估其效果。

**🔧 技术方法**

技术手段包括：数学推导阈值、softmax 欠流分析；基于 FineWeb‑Edu 的 20 B 训练语料构建小型因果语言模型；使用 Passkey 与 Needle‑in‑a‑Haystack 检索任务、perplexity 与常用解码器基准（CS、QA、LG）进行评估。

**📊 数据集**

使用的数据集：预训练模型：BLOOM、Falcon‑RW、MPT；训练语料：FineWeb‑Edu 20 B tokens；评估语料：FineWeb‑Edu 的 32 k 文档集（用于 perplexity、Passkey、NIHS）。

**📈 对比分析**

比较方法：对照 ALiBi 基线，分别在检索任务的 AUC（Passkey、NIHS）和标准解码器基准上进行零样本评测。实验表明：log‑scaled 与 clamping 组合在超长上下文的 Passkey 检索上提升近十倍（0.08→0.79 AUC）；但在 NIHS 上 ALiBi 仍最优；在 CS/QA/LG 基准上提升幅度仅 1.6–3.4 个百分点。

**⚠️ 局限性**

局限性：仅使用 148 M 参数的小型解码器；固定架构和训练语料，未检验在更大模型或不同数据集上的迁移性；缓解策略的超参数搜索有限，实验规模受算力限制。

---

## 700. Can Large Language Models Recover Semantic Optimization Opportunities That Compilers Miss?

**arXiv ID:** 2608.03983 | [PDF](https://arxiv.org/pdf/2608.03983v1)

**作者:** Hailong Jiang `[一作]` (Youngstown State University), Chunwei Xia `[通讯]` (University of Leeds)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个名为Semantic Gap Benchmark的可执行基准，用于评估大型语言模型在 C/C++ 程序中识别并实现编译器无法推断的优化语义的能力。

**💡 创新点**

创新点在于将隐藏的优化语义与可验证的“oracle artifact”结合，构建严格的评估协议，使 LLM 在无编译器反馈的情况下自动提出可执行的语义改写并证明其对程序契约的完整性。

**🔧 技术方法**

核心技术是采用五种不同配置的 LLM（GPT‑5.6 Sol、GPT‑5.4‑mini、DeepSeek‑V4‑Pro、Llama‑3.3‑70B‑Turbo、Ternary‑Bonsai‑27B）进行单轮无反馈推理，配合人工验证器与基准跑测来判定语义识别、artifact 正确性和性能提升。

**📊 数据集**

使用的数据集包含 100 个合成案例（覆盖 50 个语义原型）和 20 个来源于六个 HPC 项目的实测案例，总计 120 个测试实例。

**📈 对比分析**

评估采用三阶段方法：语义识别、artifact 实现、性能实现，并通过恢复率、artifact 正确率、E2E 成功率等指标比较；最佳模型 GPT‑5.6 Sol 在 E2E@1.05 上达 83.3%，整体成功率 93.3%，但在真实案例上性能提升显著下降。

**⚠️ 局限性**

主要局限包括：结果高度依赖具体模型，只有少数模型能可靠恢复语义；评估仅在单一硬件/编译器环境下完成；oracle 仅为相对最佳而非全局最优；真实案例的性能提升有限；缺乏自动化验证与候选选择机制。

---

## 701. Video-DeepResearch: Towards the Next-Generation Multimodal Deepresearch Agent

**arXiv ID:** 2608.03979 | [PDF](https://arxiv.org/pdf/2608.03979v1)

**作者:** Zhen Fang `[一作]`, Feng Zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套从视频生成QA对、生成执行轨迹、训练并评估视频深度研究代理的完整管线，并提出Video-DeepResearch框架。

**💡 创新点**

采用分离感知-探索的两阶段训练、阶段化工具解锁以消除模态偏差、以及构造规模化的视频QA数据与多跳评测基准，突破了视觉工具使用率低与参数泄漏的瓶颈。

**🔧 技术方法**

结合CLIP相似性、视觉裁剪工具、两阶段SFT与Group Relative Policy Optimization（GRPO）强化学习、混合文本-视频训练以及人机协作的评测框架。

**📊 数据集**

30k视频QA对、7k训练轨迹、200实例多跳VQA基准，原始视频来自公开视频集与YouTube，使用Qwen3.5-397B-A17B等模型进行生成与验证。

**📈 对比分析**

在VideoDR和自建的Video-DR Benchmark上与Claude-4.5-Sonnet、GPT-5、Gemini 2.5 Pro等对标，Video-DeepResearch-35B-A3B取得64.0%准确率（SOTA），30B-A3B达到59.3%，显著超越同类开源模型并逼近闭源基线。

**⚠️ 局限性**

训练与评测需要大量GPU计算和人类标注，导致资源开销大且规模扩展受限；对时序动态内容的鲁棒性仍有待提升。

---

## 702. Beyond Compliance: A Proposed Framework for Ethical Governance of Student Data in Learning Analytics

**arXiv ID:** 2608.03968 | [PDF](https://arxiv.org/pdf/2608.03968v1)

**作者:** Sahana Varadaraju `[一作]` (Rowan University), Bharathwaj Vijayakumar `[通讯]` (Rowan University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并阐述了LEAGUE六柱伦理治理框架，旨在指导高等教育机构系统性评估学习分析项目。

**💡 创新点**

将合法性、平等性、主体性、治理、效用和设计伦理六个维度整合为单一可操作模型，并提供实际审核流程。

**🔧 技术方法**

基于文献综述、价值敏感设计、教育正义与能力方法构建概念性框架；使用案例分析展示其应用。

**📊 数据集**

未使用实测数据集，采用首年门槛课程早期预警系统案例演示框架可行性。

**📈 对比分析**

通过与FERPA、GDPR、DELICATE、Jisc等现有工具的对照表，证明LEAGUE覆盖范围更完整，但未给出量化性能指标。

**⚠️ 局限性**

仅为概念模型，缺乏经验验证；未涵盖所有地区法律；无法替代技术公平测试或法律审查。

---

## 703. Improved Euclidean Shallow Light Trees

**arXiv ID:** 2608.03951 | [PDF](https://arxiv.org/pdf/2608.03951v1)

**作者:** Hung Le `[一作]` (University of Massachusetts Amherst), Tianyi Zhang `[通讯]` (Nanjing University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究浅树（Shallow-Light Trees）在欧氏空间及L1空间中的性质，给出最优的光度（lightness）上界与下界；

**💡 创新点**

提出新的几何剪枝技术，实现任意维度下(1+ε)-浅树的光度上界为O(1/ε^{2/3})，并构造高维欧氏空间与L1空间的匹配下界；

**🔧 技术方法**

采用ε-网、正交几何、曲线在球面上的嵌入、正多面体构造、几何打包与分层证明等技术；

**📊 数据集**

不涉及实验数据集，全部为理论构造与数学证明；

**📈 对比分析**

与已有的最优浅树结果相比，提出的上界在任意维度下比之前的O(1/ε^2)上界更优，上界和下界在各维度下相差常数因子；

**⚠️ 局限性**

局限性在于结果主要为渐进式最优，常数项未完全精确，且方法对非欧氏范数（如L∞）的适用性尚未完全拓展。

---

## 704. Echoes in the Digital Abyss: Examining the Bubble Surrounding Security and Privacy Discourse in Social Networks

**arXiv ID:** 2608.03940 | [PDF](https://arxiv.org/pdf/2608.03940v1)

**作者:** Reagan Dennison `[一作]` (Northwestern University), Sruti Bhagavatula `[通讯]` (Northwestern University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过收集10,159名在Twitter上与安全与隐私相关的互动者及其关注者，构建包含约13.74M节点的关注者网络，并结合Louvain算法进行社群检测，同时利用NMF和LDA主题模型分析互动者的兴趣分布，以探究安全与隐私讨论是否被局限在兴趣泡沫内。

**💡 创新点**

创新点在于：①首次在大规模真实社交网络数据中验证安全与隐私讨论的社群同质性与孤立性；②结合图结构与文本主题模型，以多维度揭示讨论者兴趣聚合；③基于发现提出跨社区传播与提升信息吸引力的策略。

**🔧 技术方法**

技术手段包括：Python NetworkX构建有向加权图、Louvain社群检测、路径权重计算（基于最短路径与转发次数）、TF-IDF文本预处理、非负矩阵分解(NMF)与潜在狄利克雷分配(LDA)主题建模。

**📊 数据集**

使用数据集为：①81个关键词检索得到8,268条安全/隐私相关推文；②10,159名互动者及其粉丝，共计13.74M个用户节点；③每个互动者的20条最近互动推文用于兴趣主题建模。

**📈 对比分析**

通过计算网络的模数(0.801)和密度(1.25×10⁻⁷)评估社群结构，采用Louvain算法得到1,737万社区，进一步筛选出176个多节点社区；在主题建模上选用NMF并以主题相似度评估，得到最佳3主题；未与其他方法或基准进行对照实验，仅描述社区覆盖率与兴趣一致性。

**⚠️ 局限性**

局限性包括：①样本量相对较小（仅8k推文），受Twitter API限制；②未完整获取大账户（>75k粉丝）的关注者，导致部分边缺失；③未收集非互动者间的边，可能低估社区外传播；④以互动作为曝光衡量忽略非互动获取；⑤路径权重构造仅覆盖36%转发者-作者路径；⑥数据主要来自美国，可能存在地域偏差。

---

## 705. Bimanual Manipulation Within an 8 GB Budget: Zero-Copy Sensing and Quantized ACT on an Entry-Level Jetson

**arXiv ID:** 2608.03938 | [PDF](https://arxiv.org/pdf/2608.03938v1)

**作者:** Ekansh Singh `[一作]` (Georgia Tech Research Institute), Yashvi Gandhi `[通讯]` (Georgia Tech Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 NVIDIA Jetson Orin Nano Super 8 GB 上实现完整的双手抓取与放置机器人系统，并通过零拷贝摄像头采集与 ACT 模型量化推理实现实时控制；

**💡 创新点**

①验证零拷贝传感不受内存限制而是提升 CPU 余量；②在同一训练预算下 ACT 成功收敛而 Diffusion Policy 失败；③TensorRT INT8 只对 CNN 层量化，Transformer 层保持 FP16，表明量化需按策略配置；

**🔧 技术方法**

GStreamer + NVIDIA NVMM、ACT (Action‑Chunking with Transformers)、Diffusion Policy、TensorRT（FP16/INT8 量化）、LeRobot 框架、JetPack 6.2.2；

**📊 数据集**

100 条双手抓取演示（10 Hz，三摄 640×480），用于 ACT 与 Diffusion Policy 的共同训练与评估；

**📈 对比分析**

对比方法：在相同数据集、相同摄像头配置下，分别在 100 k 步（ACT）和 200 k 步（Diffusion）训练；实机 20 次 FP32/FP16/INT8 评估，ACT 在 FP32/FP16/INT8 均成功率 19/20/19；Diffusion 在 10 次试验中 0/10；量化后推理平均延迟从 114 ms 降至 17.9 ms（FP16）和 12.6 ms（INT8），吞吐率提升 6.4×/9.0×；

**⚠️ 局限性**

仅在单一任务（柔性豆袋）与单一平台上验证；数据量少、试验次数有限；未对 Diffusion Policy 进行超参数搜索、未测定其推理延迟、未构建 FP32 TensorRT 引擎，导致精度/编译收益混合；仅使用三 USB 摄像头导致 10 Hz 采集上限，未探索更高频传感。

---

## 706. Progressive Learning of a Diffusion-based Inpainting Model for Separating Overlapped Fingerprints

**arXiv ID:** 2608.03937 | [PDF](https://arxiv.org/pdf/2608.03937v1)

**作者:** Noor Hussein `[一作]` (Michigan State University), Karthik Nandakumar `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于扩散模型的分层学习管线，用来分离重叠指纹中的各个成分指纹；

**💡 创新点**

创新点在于将重叠指纹分离转化为条件填补任务，并采用三阶段逐步细化的 LoRA 训练策略，最终实现了多通道重叠感知填补；

**🔧 技术方法**

主要技术包括Stable Diffusion v1.5 的扩散框架、GenPrint 指纹先验、三阶段 LoRA 细化、遮罩多通道条件输入、联合重构与混合损失；

**📊 数据集**

使用了来自 11 个公开指纹数据库的 55,000 对合成重叠指纹（基于 FVC、IIITD、NIST 等）以及公开的 TLOF 与 TSOF 两个测试集；

**📈 对比分析**

与“无分离”“原始分离”“单指纹填补”等基线比较，最终模型在 TSOF/TLOF 上的闭集识别 TPIR/TPIR、FPIR 均大幅提升，Rank‑1 识别率超过 95%，优于 FinSNet；

**⚠️ 局限性**

主要局限在于需要先验的成分掩码、对低质量或高重叠比的指纹重建效果不佳，且在极端遮挡下仍可能产生错误填补。

---

## 707. Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility

**arXiv ID:** 2608.04001 | [PDF](https://arxiv.org/pdf/2608.04001v1)

**作者:** Mohsen Hariri `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统化定义了推理时扩展（test‑time scaling）为在自回归模型前缀树上进行预算化推理，并提出三类结构：单轨序列扩展、叶级采样与归约以及前缀级搜索。

**💡 创新点**

创新点在于：1) 将推理时扩展视作预算化前缀树搜索，统一描述三种常见范式；2) 提出了完整推理系统评估框架，区分候选库诊断与端到端性能；3) 发布了含 2.0M 条完整推理轨迹的开源数据集，用于可复现性与评估；4) 通过对 27 个开源模型、14 个领域、23 项任务的实验，验证了不同预算、检验器和抽样策略对准确率的影响。

**🔧 技术方法**

技术主要包括：预算化自回归推理、候选归约（verifier‑constrained selection、self‑consistency、Best‑of‑N、MBR 统一框架）、前缀评分模型（log‑prob、价值模型、过程奖励、Monte‑Carlo 估计）、搜索控制器（beam、best‑first、MCTS）以及评估指标（Pass@k、候选发现与稳定性曲线、误差分析）。

**📊 数据集**

使用的数据集包括：MathArena 2025–2026 竞赛题集、EvalScope 72 个领域的 3600 题、以及 23 项广泛认知与符号推理任务（共 14 个领域）。模型涵盖 Qwen、gpt‑oss、Phi‑4‑reasoning 等 27 个开源权重模型。

**📈 对比分析**

比较方法：在统一的推理协议下记录完整轨迹、候选分布和评估器输出，采用 Pass@k 以及候选发现/稳定性曲线进行性能对比。实验表明：单轨策略在单一样本上准确率最高（最高 70% 以上），但随着样本数增大，基于最高 token log‑prob 的选取会出现误差；叶级采样在 k=80 时 Pass@k 可达 94% 以上；前缀搜索可显著降低计算开销但对评分函数敏感。整体模型排行榜随评估标准和预算变动而显著变化。

**⚠️ 局限性**

局限性：1) 不同推理协议导致结果难以跨论文比较；2) 评估器（尤其是学习型评估器）存在偏差和过拟合风险；3) 前缀评分模型缺乏可验证性，可能导致误剪枝；4) 公开轨迹不覆盖所有模型和参数组合，且重放仅能验证已生成的候选集；5) 实验主要聚焦于开放权重模型，尚未对商业大模型的推理时扩展进行全面评估。

---

## 708. string2string Studio: An Interactive, In-Browser Platform for String-to-String Algorithms

**arXiv ID:** 2608.03984 | [PDF](https://arxiv.org/pdf/2608.03984v1)

**作者:** Mirac Suzgun `[一作]` (Stanford University), Dan Jurafsky `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于浏览器的交互式平台，整合了字符串对字符串的对齐、距离、相似度、检索、生成评估和BLAST同源搜索六大模块，可在字符、词、标记、行及残基层面运行；

**💡 创新点**

首次提供无需安装、客户端本地执行的多层级交互工作台，并将所有算法输出以可视化证据形式呈现，支持跨领域比较、可复现链接和教学；

**🔧 技术方法**

核心算法用 C++ 重写并编译为 WebAssembly（128 位 SIMD 与标量回退），配合 TypeScript 参考实现；利用 Hirschberg 线性空间、位并行、SIMD 加速，并在浏览器内完成 BLAST 的 seed‑extend 过程；

**📊 数据集**

使用公开基因组序列（如 SARS‑CoV‑1/2 spike、β‑globin、16S rRNA 等）和文本数据（英文学术文本、拉丁文手稿等）作为测试样本；

**📈 对比分析**

与 Python 版库、Biopython、ssw、ncbi blast+ 等基准库对比，单次原始算法速度提升 100–2,500 倍，整体平台在全局/局部对齐上与 native C 相当，BLAST 结果与 ncbi blast+ 排序、统计差异 ≤0.5%；

**⚠️ 局限性**

局限：浏览器内存限制最长可视化长度约 12k，未涵盖神经网络或语义评估指标，BLAST 仅验证统计一致性，未进行正式用户学习评估，示例以英语/拉丁字符为主，未覆盖多语言或多层级相似度广泛测试。

---

## 709. JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion

**arXiv ID:** 2608.03974 | [PDF](https://arxiv.org/pdf/2608.03974v1)

**作者:** Yicheng Xiao `[一作]` (Joy Future Academy), Nan Duan `[通讯]` (Joy Future Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了JoyAI-Video-Edit，一个16B参数的自回归扩散模型，能够实时、持续地进行指令或参考图像驱动的视频编辑，兼顾源视频保持与编辑一致性；

**💡 创新点**

通过分块自回归适配、源锚定分布匹配蒸馏（SA‑DMD）与长时程自回归蒸馏（LHAD），解决了训练推理不匹配、源漂移、实时推理与长时间稳定性等核心瓶颈，并将高阶扩散迭代压缩为两步生成；

**🔧 技术方法**

使用MLLM条件编码器、因果视频VAE、MM‑DiT、分块双向/因果注意力、滑动窗口KV缓存、FP8量化、动态镜像循环、源锚定CFG等技术；

**📊 数据集**

基于多模态T2V数据、JoyAI-Image的图像编辑对、从关键帧推导的编辑视频以及构建的LongV2VBench长视频编辑基准；

**📈 对比分析**

与StreamDiffusionV2、SANA‑Streaming、LiveEdit、XMax‑X2.0等流式编辑器以及VACE、OpenVE‑Edit、Kiwi‑Edit、Bernini‑R等离线编辑器在OpenVE‑Bench短视频和LongV2VBench长视频上进行自动与人工评估；在短视频上获得3.60分，明显优于流式基线；在长视频上获得3.30分，并以720p 30FPS的速度跑在同类方法之上，显著降低延迟；

**⚠️ 局限性**

仍受限于大模型规模与显存需求，对极长序列存在累计误差风险，缺乏真实直播场景的充分验证，对多目标复杂编辑的鲁棒性尚未完全确认。

---

## 710. Interpretable Adaptive Sampling for LLM Test-Time Scaling

**arXiv ID:** 2608.03961 | [PDF](https://arxiv.org/pdf/2608.03961v1)

**作者:** Mobina Kashaniyan `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可解释的自适应推理时间缩放方法，利用模糊控制器根据提示难度和模型不确定性动态分配候选答案数量；

**💡 创新点**

创新点在于用可解释的层级模糊规则取代黑盒策略，既能在保持高准确率的同时减少样本数量，又能公开每个提示的算力分配理由；

**🔧 技术方法**

使用模糊逻辑（梯形和三角形隶属函数）、自证据置信度、熵、历史性能缓存等信号构建控制器，并结合自证据+Borda聚合选择答案；

**📊 数据集**

在GSM8K、MATH和SciQ三个基准集上进行实验，涉及Phi-3-mini和Qwen2.5-1.5B两大模型；

**📈 对比分析**

与Best‑of‑N、compute‑optimal、固定预算+自证据+Borda等基线对比，实验显示自适应方法平均节省10–15%样本，准确率与全预算N=8差距不足1%，在数学推理任务上更显优势；

**⚠️ 局限性**

局限在于模糊规则是手工设计，可能不适用于所有模型/任务，且答案选择仍是性能瓶颈，改进需要更系统地调优规则或结合学习式重排器。

---

## 711. Data-Driven Online Slice Admission Control and Resource Allocation in NextG Mobile Networks

**arXiv ID:** 2608.03954 | [PDF](https://arxiv.org/pdf/2608.03954v1)

**作者:** Muhammad Sulaiman `[一作]` (University of Waterloo), Raouf Boutaba `[通讯]` (University of Waterloo)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于价格的在线切片接入控制与资源分配框架（OSARA），通过为资源动态赋予伪价格来评估每个切片请求的价值并做出接入与嵌入决策；

**💡 创新点**

创新点在于：①将接入决策与资源分配通过可学习的价格函数解耦，既保持了长期系统影响又降低了实时计算复杂度；②设计了指数定价策略，提供竞争性分析下的最坏情况保证；③在此基础上引入自适应学习版本（EXP‑Adaptive），利用历史数据调整定价参数以提升实际表现；

**🔧 技术方法**

核心技术包括：虚拟网络嵌入（VNE）建模、单切片成本最小化（可通过MIP求解）、指数价格函数设计与学习、竞争性分析与误差容忍的改进接纳规则、以及两层优化（时间限制+启发式）保证实时性；

**📊 数据集**

实验数据集：从真实5G试验平台（OpenAirInterface + Open5GS + Kestrel交通负载）获取VNF资源与延迟特征；使用Microsoft Azure VM时序与Telecom Italia Metro网络拓扑模拟切片请求；

**📈 对比分析**

与基线方法（贪心、SlicePilot+（MARL）、MPC‑Oracle（完美预测的模型预测控制）和Node‑Ranking）对比；OSARA在48小时模拟中平均收入提升32.2%（相较于SlicePilot+）和26.7%（相较于MPC‑Oracle）；同时在计算时间上比MPC‑Oracle低约1/9至1/45；

**⚠️ 局限性**

局限性：①指数定价的最坏情况保证依赖于多项假设（如资源需求分布与小于容量比例）；②自适应学习仍需要大量历史样本，且对分布漂移敏感；③单切片优化虽快速，但在极端拥堵或大规模并发时仍可能导致次优分配；④实验平台规模与真实运营商网络相比仍有限，尚未验证在更大、更动态的场景下的鲁棒性。

---

## 712. TACT: Taxonomy-Aligned Post-Training for Pedagogically Adaptive English Tutoring

**arXiv ID:** 2608.03952 | [PDF](https://arxiv.org/pdf/2608.03952v1)

**作者:** Dongjie Yang `[一作]`, Zixin Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们构建了一个基于人类教学原则的ESL对话导师框架，并将其应用于对Qwen3.5-4B模型的后训练和评估；

**💡 创新点**

创新点在于设计了13项导师策略与双维学生动作标签的整合分类，既能捕捉对话中的教学决策，又能为模型提供明确的策略指引；

**🔧 技术方法**

技术上采用了LoRA参数高效微调、基于分类标签的奖励设计以及GRPO策略优化，并配合可审计的评估流程；

**📊 数据集**

使用的数据集来源于TSCC v2，共260个真实一对一ESL对话，构成32,379条带标签的训练实例和78条诊断样本；

**📈 对比分析**

在策略平衡的诊断基准上，模型相较基线提升20.3个百分点，在50名学习者的盲测中获得5.54/7的最高平均评分，并在多项外部教育基准上保持或提升性能；

**⚠️ 局限性**

主要限制是仅评估单轮生成质量，未验证长期学习成效；标签抽象可能无法完整覆盖教师多维教学；评估高度依赖预设构造，需进一步验证。

---

