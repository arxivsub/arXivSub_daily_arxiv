# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-17 | 今日论文总数: 438

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. EEG-PRISM: Physiologically-Grounded Interpretability of Predictions by EEG Foundation Models

**arXiv ID:** 2608.13676 | [PDF](https://arxiv.org/pdf/2608.13676v1)

**作者:** Deeksha M Shama `[一作]` (Johns Hopkins University), Archana Venkataraman `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出EEG-PRISM，利用线性变换将基于时间-通道的模型归因映射到频谱和源空间。

**💡 创新点**

首次提供无需修改或再训练模型的全局可解释框架，并给出理论证明与误差界限。

**🔧 技术方法**

使用离散傅里叶变换、EEG前向/逆模型、LSTM/Transformer基础模型以及LRP、IG、DeepLIFT、DeepSHAP等归因方法。

**📊 数据集**

在模拟数据、Temple大学癫痫数据集（TUSZ）和自闭症对照数据集（ACE）上评估。

**📈 对比分析**

与四种归因器和五种基础模型比较，在模拟中实现近乎完美的频谱恢复、约69%空间精度；在癫痫中约50%定位准确；在自闭症中可识别Delta/Alpha生物标记，性能与已知文献相符。

**⚠️ 局限性**

仅适用于线性映射，逆问题仍受估计误差影响，未考察联合频谱-源域、非线性变换和多窗口一致性。

---

## 2. Removing Temporal Note Redundancy Improves Multimodal Reinforcement Learning for Medicine

**arXiv ID:** 2608.14157 | [PDF](https://arxiv.org/pdf/2608.14157v1)

**作者:** Chenran Weng `[一作]` (University of California Berkeley), Anil Aswani `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种冗余感知多模态状态表示框架，在机械通气的离线强化学习中先将临床笔记拆分为历史信息与新信息，再与结构化变量拼接后作为状态输入。

**💡 创新点**

创新点在于两种计算高效的笔记去冗余方法：嵌入空间的奇异值分解（SVD）与句子级 diff 运算，能显式分离持续病历与新出现信息，从而提升状态质量。

**🔧 技术方法**

技术手段包括 ClinicalBERT 对笔记编码、PCA 压缩、SVD 子空间投影、句子 diff、Conservative Q‑Learning、行为克隆以及模型基 Rollout、Fitted Q Evaluation、Weighted Importance Sampling 与 Weighted Doubly Robust 四种离线评估。

**📊 数据集**

实验数据来源于 MIMIC‑III 机械通气病历，涵盖 10,125 位 ICU 患者、157,924 条笔记、72 小时 4 小时决策间隔共 18 个决策点。

**📈 对比分析**

与结构化仅、原始笔记多模态基线对比，使用四种 OPE 估计器显示冗余感知表示均显著提升策略价值，最大提升为 MB 的 +0.75、WIS 的 +0.64，其余评估亦保持显著正增益。

**⚠️ 局限性**

局限性包括仅基于回顾性离线数据，存在样本偏差与分布移位，缺乏前瞻性验证，且去冗余方法未必总能捕捉与机械通气直接相关的临床更新。

---

## 3. Act2Intention: A Benchmark For Developing Active Mobile Agents Through Inferring User Intention from GUI Actions

**arXiv ID:** 2608.14132 | [PDF](https://arxiv.org/pdf/2608.14132v1)

**作者:** Xiaokai Yan `[一作]` (Northwestern Polytechnical University), Zhiwen Yu `[通讯]` (Northwestern Polytechnical University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了Act2Intention框架，构建了持续意图‑动作轨迹的基准数据集并实现了主动移动代理实现意图理解、预测与执行

**💡 创新点**

创新点在于首次把意图理解、预测和执行统一为三阶段的主动代理流程，并通过人工验证的LLM生成技术扩充大规模持续意图轨迹

**🔧 技术方法**

采用多模态大型语言模型（MLLM）与视觉语言模型进行动作描述、意图分割与理解；对意图预测与执行使用监督微调与经验引导的LLM；基于GUI‑Agent技术实现动作执行

**📊 数据集**

使用由90名匿名用户提供的真实手机交互日志（含屏幕截图、访问树等），通过LLM生成补充，最终得到72,511个意图、700k+动作、52个应用的Act2Intention Bench

**📈 对比分析**

与基线（未微调的LLM、ProactiveAgent等）对比，微调后模型在意图理解Acc‑S提升+0.49、意图预测Acc‑S提升+0.40，执行SSR在不同GUI‑Agent上提升约9–13个百分点，整体系统SR最高达47.6%（真实意图）

**⚠️ 局限性**

局限包括：缺乏时机预测与用户接受度评估、对新应用/界面适应不足、数据缺乏多样性与人口学信息、实际部署面临隐私与能耗挑战

---

## 4. Structural Leakage in Graph Encryption: Attacks and Defenses

**arXiv ID:** 2608.13981 | [PDF](https://arxiv.org/pdf/2608.13981v1)

**作者:** Hua Shen `[一作]` (Hubei University of Technology), Mingwu Zhang `[通讯]` (Hubei University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本报告分析了图加密方案中对单对最短路径查询的结构泄露问题，并针对PathGES提出了Fragment Tree攻击以及针对其缺陷设计的BlindGES防御方案。

**💡 创新点**

创新点在于将HLD分解与规范化片段编码结合，构造Fragment Tree以揭示结构泄露；并提出Merge‑and‑Divide机制与双层多映射索引，显著降低一一映射比例并抑制侧信道泄露。

**🔧 技术方法**

核心技术包括重写PathGES的HLD分解与canonical segment编码、构造Fragment Tree与查询树的同构证明、对M_1/M_2进行多映射加密（EMM‑RR与EMM‑RH）、以及侧信道推断与安全性证明。

**📊 数据集**

实验使用了七个真实网络数据集（InternetRouting、Ca‑GrQc、email‑EU‑core、facebook‑combined、p2p‑Gnutella08、p2p‑Gnutella04、p2p‑Gnutella25）进行攻击效果与性能评估。

**📈 对比分析**

与原PathGES相比，Fragment Tree攻击在稀疏图上最高可准确恢复10.24%查询；BlindGES将一一映射从99%降至≤20%，设置时间缩短50%，存储开销降低32%，并将路径长度泄露限制在1%以内，查询时延保持在毫秒级。

**⚠️ 局限性**

局限性包括攻击依赖已知完整图和完整令牌观察；对高密度图的精确恢复效果低；BlindGES需预设分割参数且对极大图仍有较长设置时间；若底层多映射加密存在漏洞，整体安全性将受影响。

---

## 5. Omni-LiveAvatar: Minute-Level Real-Time Streaming Joint Audio-Visual Avatar Generation

**arXiv ID:** 2608.13602 | [PDF](https://arxiv.org/pdf/2608.13602v1)

**作者:** Lunjie Zhu `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 59331 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了 Omni‑LiveAvatar，能够在单张 NVIDIA H200 GPU 上以每秒 22 帧的速度实时生成长达一分钟的音视频虚拟人；

**💡 创新点**

核心创新包括：① 逐步自回归蒸馏（progressive autoregressive distillation）将大规模双向扩散模型转为少步因果生成器；② 同步音视频长短期记忆（synchronized audio‑video long‑short‑term memory）实现全局一致性与跨模态对齐；③ 层次滚动提示规划（hierarchical rolling prompt planning）实现语义平滑过渡；

**🔧 技术方法**

技术实现基于 LTX‑2 19B 模型，使用分阶段蒸馏、ODE 回归、滚动窗口自回归推理、RoPE 重新定位、以及文本提示分块；

**📊 数据集**

使用内部 35K 文本提示数据集来生成训练样本，覆盖多样化语音与视频场景；

**📈 对比分析**

与 LTX‑2、Ovi、Omni‑Forcing、Hallo‑Live 等基线对比，Omni‑LiveAvatar 在视频质量、语音质量、同步性、文本一致性与人类身份保真度等六项指标上均优于实时自回归模型，并在速度上实现 33 倍提升；

**⚠️ 局限性**

局限性包括：对长时序的依赖仍需更多实验验证；在极长序列或更复杂场景下的记忆与提示规划可能出现漂移；以及对不同硬件配置的可迁移性尚未充分评估。

---

## 6. H2H Music Improv: A Communication Model and Audio-Visual Dataset for Music Improvisation

**arXiv ID:** 2608.13957 | [PDF](https://arxiv.org/pdf/2608.13957v1)

**作者:** Aleksandra Teng Ma `[一作]` (Massachusetts Institute of Technology), Alexander Lerch `[通讯]` (Georgia Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过与自由即兴演奏专家的协作共设计，构建了一个基于“发起”与“确认”事件的沟通模型，并收集整理了六小时的双人自由即兴音频-视频数据集，提供每位演奏者的意图与对方意图的双向注释。

**💡 创新点**

创新点包括：①首次将即兴演奏者的沟通过程形式化为可机读的事件序列和状态（谈判、提议、稳定）；②首次发布带有清晰单人音轨和双向意图注释的自由即兴数据集；③揭示了演奏者的产生意图与感知意图之间的显著非对齐现象，为 AI 合作伙伴的沟通设计提供了实证基础。

**🔧 技术方法**

技术手段包括：协作共设计工作坊与访谈、音频采集（Pro Tools、OBS 同步）、视频采集（Sony HXR‑NX3 + Blackmagic UltraStudio）、单轨录音与无交叉混响的隔音室设置、Reaper 进行双向时间标注，以及对标注进行定量对齐分析（整体对齐率、IoU 等）。

**📊 数据集**

使用的数据集为 H2H Music Improvisation Dataset，包含 37 个双人即兴录音，总时长 6 小时 8 分钟，配有每位演奏者的干净单轨音频、视频、以及每位演奏者对自己和对方意图的时间标注，共计 1,368 个标注事件。

**📈 对比分析**

比较方法主要是对齐分析：计算整体对齐率以及各状态（稳定、提议、谈判）的交并比（IoU）。结果显示稳定状态的对齐率中位数为 81.3%，而提议与谈判状态的对齐率分别为 23.6% 和 0%，体现了意图感知差异的显著性。

**⚠️ 局限性**

局限性包括：模型仅捕捉离散事件，无法描述快速确认或逐渐融合等连续互动；数据集规模有限且样本多为男性、正式训练的专家，缺乏多样性；部分录音因金属乐器动态范围导致削波；长时段演奏造成创作疲劳，影响数据一致性。

---

## 7. DepWareTrans: Dependency-Aware Incremental Repository Migration across Co-executable Languages

**arXiv ID:** 2608.14128 | [PDF](https://arxiv.org/pdf/2608.14128v1)

**作者:** Sivajeet Chand `[一作]` (Technische Universitaet Munich), Sushant Kumar Pandey `[通讯]` (University of Groningen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于依赖关系的增量仓库迁移框架 DepWareTrans，能够在可共执行语言之间按依赖一致性批次迁移代码。

**💡 创新点**

将迁移单元从单文件提升到依赖一致的批次，并通过构建依赖图、SCC 合并和批次化翻译，实现跨文件一致性；结合编译和测试驱动的迭代反馈。

**🔧 技术方法**

使用静态依赖分析构建文件依赖图、强连通分量检测、LLM（如 GPT‑5.3、Claude Opus 4.6）批次翻译，以及基于编译和单元测试的自动化验证与错误反馈。

**📊 数据集**

在 51K LOC 的工业系统 STAR、Apache Commons 系列（6K–29K LOC）以及 C#→F#、Java→Scala 等跨语言项目上进行实验。

**📈 对比分析**

与文件级翻译（含/不含反馈）对比；在 STAR 中文件级翻译编译成功率仅 38.16%、测试成功 9.39%，而 DepWareTrans 在同一环境下实现 100% 编译与测试通过，迭代次数极少；在其他语言对上亦能在少量迭代内达成 100% 成功。

**⚠️ 局限性**

对非可共执行语言或缺乏完整构建/测试环境不适用；对极大中心类或高度耦合模块仍可能需要人工干预；依赖分析误差或 LLM 非确定性仍是潜在问题。

---

## 8. Fashion Outfit Generation via Unified Sequential Composition Models

**arXiv ID:** 2608.13888 | [PDF](https://arxiv.org/pdf/2608.13888v1)

**作者:** Kaicheng Pang `[一作]` (Laboratory for Artificial Intelligence in Design), Waikeung Wong `[通讯]` (Laboratory for Artificial Intelligence in Design)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在大型服装库中根据种子服装生成风格一致的完整服装组合，将任务形式化为“受限集合生成”并通过 MDP 进行建模。

**💡 创新点**

主要创新是提出统一的序列组合模型（USCM）同时学习兼容性评估和潜在选择意图，并结合潜在扩展蒙特卡洛树搜索（LE-MCTS）实现对海量候选的动态规划。

**🔧 技术方法**

采用 Transformer 共享编码器、双头设计（Value 与 Policy）、InfoNCE 对齐、Robust Negative Sampling、CLIP 嵌入以及进阶 MCTS（Progressive Widening）等技术。

**📊 数据集**

主要使用 Polyvore Outfits、iFashion 与 PolyvoreU 三个公开服装数据集进行训练与评估。

**📈 对比分析**

与随机、Type-aware、VLLM、OutfitTransformer 等基线对比，使用人类偏好、神经兼容性评分、类别 JSD 和结构有效率等指标，本文方法在人类偏好 41.81% 及结构有效率 0.815 等指标上均超过或接近最新 SOTA。

**⚠️ 局限性**

局限包括：评估仍受主观影响；价值头仅训练于完整服装，对部分集合的评估不够精准；缺乏专业造型师的创造性与对地域、季节等细微偏好的适应。

---

## 9. HELIX: Model-Harness Co-evolution for Recursive Self-Improvement

**arXiv ID:** 2608.13951 | [PDF](https://arxiv.org/pdf/2608.13951v1)

**作者:** Tianyu Fan `[一作]` (University of Hong Kong), Chao Huang `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HELIX 系统，构建可追踪的 Agent 运行时组件，利用 Build–Update–Rebuild 循环实现模型与运行时共同进化，生成可用于下次模型更新的验证同胞轨迹。

**💡 创新点**

将运行时设计视为模型改进的同等重要部分，首次在源代码层面实现可追踪的运行时介入，形成构建、验证、重构三步闭环，并将验证同胞数据转化为多种训练记录。

**🔧 技术方法**

使用 typed ports、atoms、recipes、product shells 与 runtime policies 组成的结构化介入框架，结合预执行检查与 evidence plane 进行可审计的运行时构建；评估采用 LiveCodeBench 与 SWE‑bench 任务；通过对比固定运行时与候选组合进行性能测试。

**📊 数据集**

LiveCodeBench 代码生成任务（100 例）和 SWE‑bench 真实 GitHub issue 任务（55 例）。

**📈 对比分析**

对比基准 Pi，固定运行时从 50/100 提升到 52/100（+4%），候选组合覆盖率达 79/100；在更深层验证中单个成员覆盖 75/100（LCB）或 46/55（SWE‑bench）。同胞数据共生成 438 条多用途训练记录。

**⚠️ 局限性**

仅测试公开单元测试，未覆盖隐藏测试；未实际训练更新模型；数据规模有限；选取成员可能引入偏差；缺乏在线路由器评估；无法单独测量各组件影响；安全与权限控制需进一步验证。

---

## 10. OccPlanner: Goal-Aware Occupancy-Conditioned Diffusion Planner for Pixel-Goal Navigation

**arXiv ID:** 2608.14160 | [PDF](https://arxiv.org/pdf/2608.14160v1)

**作者:** Binling Huang `[一作]` (Changhong Intelligent Robot), Lanpeng Jia `[通讯]` (Changhong Intelligent Robot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于扩散模型的像素目标导航规划器OccPlanner，能够将像素目标在视角中几何地映射到本地三维占据空间并生成连续的避障轨迹；同时设计了L3ROcc框架，将单目RGB导航视频转换为可见性感知的局部占据与轨迹标签，实现大规模占据监督；

**💡 创新点**

1) 两阶段目标条件机制：先对时序视觉上下文进行编码，再结合本地占据特征形成ego‑goal表示，显著提升规划可靠性；2) 将像素目标直接映射到本地三维坐标系并与占据信息协同，突破传统像素目标缺乏深度与可通行性的限制；3) L3ROcc实现了从单目视频到占据监督的端到端自动化流程，解决占据标注瓶颈；

**🔧 技术方法**

基于扩散概率模型的连续轨迹生成、π^3 视觉几何编码器、DINOv2+Depth‑Anything V2 的RGB‑D融合、卷积3D占据预测、交叉注意力的ego‑goal与占据注入、可见性射线推理的voxel化占据生成；

**📊 数据集**

InternData‑N1 训练集（200K+轨迹）和 InternScenes 评估集（60个未见室内场景，四类场景与两距离区间）以及 Unitree Go2 实验数据（829个真实场景采样）；

**📈 对比分析**

在 60 场未见环境下的闭环仿真中，OccPlanner 在 5–8 m 区间对比 NavDP 与 PixNav，平均成功率从 NavDP 的 24.98% 提升至 69.90%，在拥挤易/难场景分别提升至 86.20% 与 84.92%；SPL 与 DTG 同样显著改善；在实际机器人上进行的开环测试中，零射显效能，微调后占据预测更密集、轨迹更平滑；

**⚠️ 局限性**

仅在开环条件下评估真实世界性能，未进行闭环物理部署；实验仅涵盖静态室内环境，缺乏动态障碍物与长时间导航验证；占据监督依赖单目重建的精度，可能受光照与纹理影响；

---

## 11. "I Thought You Were The Uncensored Place": Norms, Rules, and Moderation in AI-Generated Sexual Content Communities

**arXiv ID:** 2608.13659 | [PDF](https://arxiv.org/pdf/2608.13659v1)

**作者:** Lucy Qin `[一作]` (Georgetown University), Elissa M. Redmiles `[通讯]` (Georgetown University)

**通讯引用:** 2689 | [OpenAlex ID](https://openalex.org/A5074435310)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大型AIG‑SC社区（公开禁止非自愿内容）的24名成员与版主进行深度访谈，探讨社区形成、隐式与显式规范、治理实践与挑战。

**💡 创新点**

首次系统性梳理AIG‑SC社区治理结构与实践，揭示其在防止非自愿内容创作时的有效性与不足，并给出监管、技术与教育层面的干预建议。

**🔧 技术方法**

采用定性研究方法——深度访谈和反思性主题分析；未使用机器学习或自动检测技术。

**📊 数据集**

访谈样本共24名参与者，来自3个Discord社区（10k–50k+成员）。

**📈 对比分析**

本研究为探索性案例研究，无可比对的量化性能指标；主要以质性发现呈现。

**⚠️ 局限性**

样本规模有限、只包含已公开禁止AIG‑NCC的社区，难以推广到更广泛或未制定规则的AIG‑SC社区；访谈者与受访者可能存在自我呈现与社会期望偏差。

---

## 12. Over the Memory Wall, Into the Instruction Wall: The New Bottleneck in GPU Data Processing

**arXiv ID:** 2608.13696 | [PDF](https://arxiv.org/pdf/2608.13696v1)

**作者:** Sven Hepkema `[一作]` (ETH Zurich), Gustavo Alonso `[通讯]` (ETH Zurich)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了Valk工具，对cuDF在NVIDIA L4和GH200 GPU上执行TPC‑H基准进行综合性能分析，揭示内存带宽提升后瓶颈从内存迁移到指令吞吐量。

**💡 创新点**

创新点：①首次将Nsight Systems、Nsight Compute以及Maximus日志融合生成统一数据集；②使用roofline模型对cuDF算子进行详细分析，证明其在高带宽GPU上已转为计算绑定；③提出三项针对新瓶颈的优化建议：更高缓存利用、提高占用率/指令级并行、降低每次内存访问的指令强度。

**🔧 技术方法**

采用GPU roofline分析、CUDA Profiler工具（Nsight Systems/Compute）、Maximus执行器、cuDF 25.12、Valk可视化与SQL查询分析。

**📊 数据集**

使用TPC‑H OLAP基准（规模因子1、3、10、30；GH200额外包含SF100），在单GPU环境下进行评测。

**📈 对比分析**

通过比较L4与GH200的总kernel时间、算子占比、内存与指令利用率，发现GH200内存带宽提升13.4倍，但TPC‑H总体加速仅5.2倍；compute bound比例在GH200上升到77.5%，表明内存带宽提升未带来相应性能提升。

**⚠️ 局限性**

局限性：①实验仅在单GPU单机上完成，未覆盖多GPU并行；②仅评测TPC‑H基准，其他查询类型的适用性未知；③缺少在实际数据库系统（如BlazingSQL、SiriusDB）中的完整性能验证；④建议的缓存与ILP改进尚未实现验证。

---

## 13. SDO: Subspace Deconflicting Operator for Multi-Adapter Composition

**arXiv ID:** 2608.13820 | [PDF](https://arxiv.org/pdf/2608.13820v1)

**作者:** Zhongsheng Wang `[一作]` (University of Auckland), Jiamou Liu `[通讯]` (University of Auckland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Subspace Deconflicting Operator（SDO），用于在共享扩散模型中对多身份适配器进行子空间冲突消除，以提升多角色生成的身份一致性和场景稳定性。

**💡 创新点**

将多适配器干扰建模为参数空间的子空间冲突，并设计可置换等价的集体变换和低秩签名重构，实现对多个适配器的兼容重写而非简单相加。

**🔧 技术方法**

基于低秩分解提取子空间签名，使用注意力机制结合子空间冲突先验进行冲突感知集成变换，加入子空间冲突和保持正则化，并通过身份识别器监督训练。

**📊 数据集**

在两个身份LoRA池（Shakoomaku和Flintstones）上使用FLUX.1-dev和SDXL-base-1.0扩散模型进行实验。

**📈 对比分析**

与Naive Merge、CharCom、FreeFuse、LoRA-Composer等四种基线在2-5适配器组合下对比，指标包括身份相似度(ID)、身份召回(IR)、人脸计数(FCS)和CLIP相似度(CL)，结果显示SDO在4-5适配器高复杂度场景下显著提升身份保真度和稳定性，尤其在ID、IR、FCS上领先。

**⚠️ 局限性**

实验仅限于每池5个身份LoRA和小规模组合，未评估更大规模适配器集合或跨域任务的通用性，并且训练过程需要身份检测器作为监督。

---

## 14. Doomed to Re-Annotate, Forever: The ImageNet Story

**arXiv ID:** 2608.13783 | [PDF](https://arxiv.org/pdf/2608.13783v1)

**作者:** Illia Volkov `[一作]` (Czech Technical University in Prague), Jiri Matas `[通讯]` (Czech Technical University in Prague)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新注释 ImageNet-1k 验证集，构建 ReImageNet，提供多标签、对象定位、类别重定义和语义属性，形成更完整、更准确的数据集。

**💡 创新点**

① 将单标签任务转为多标签定位并加入属性；② 采用人类+LLM 协同的迭代流程提升标注质量；③ 分析错误如何在 ImageNet 派生基准中放大，并提出改进建议。

**🔧 技术方法**

利用 OWLv2、OpenCLIP、SAM3 等视觉模型及 GPT-4o/GPT-5.4 等大语言模型辅助标注，配合内部标注员、质量控制和工具化的标注工作流。

**📊 数据集**

原始 ImageNet-1k 验证集（50k 图像）以及公开的 ReImageNet 注释；在此基础上评估 MLLM、VLM 与监督模型的性能。

**📈 对比分析**

对比原始标签与 ReImageNet 的 Top-1 Accuracy，使用多标签、crop‑based 评价；结果显示 MLLM 提升 5–6%，监督模型提升 ≤1.2%；在裁剪评估中监督模型下降最多 33%，而 MLLM 更稳健。

**⚠️ 局限性**

标注者同质化、缺乏领域专家、模型预测可能导致偏差、仅覆盖验证集、仍存在标注差异、依赖闭源 LLM 的可重复性受限。

---

## 15. Multiphase-Diff: Diffusion-Based Generative Modeling for High-Contrast Multiphase Physical Systems with Sharp Interfaces

**arXiv ID:** 2608.13669 | [PDF](https://arxiv.org/pdf/2608.13669v1)

**作者:** Yining Huang `[一作]` (University of Texas), Zhenyu Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 17161 | [OpenAlex ID](https://openalex.org/A5088561209)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种多相物理约束扩散模型 Multiphase-Diff，用于生成高对比度、尖锐界面多相场，同时保证物理一致性。

**💡 创新点**

创新点包括基于保守有限体积的流量残差、对数/反双曲正弦的可逆表示以保持系数正值并解决低振幅相，及 Jacobi 预处理的残差尺度归一化。

**🔧 技术方法**

采用扩散概率模型（DDPM）与 PDE 残差监督相结合，利用 U‑Net 生成器、Min‑SNR 加权、Jacobi 预处理的似然损失。

**📊 数据集**

在三种多相基准上评估：二维 Darcy 透流两相流、气液管道流动两相粘度/速度、三相电导率/电势，均为 64×64 网格的 10,000 训练样本。

**📈 对比分析**

与七种基线（DDPM、PG‑Diffusion、CoCoGen、DiffusionPDE、FunDiff、PIDM、PIDM‑log）在四项物理指标上进行比较，Multiphase‑Diff 在物理残差、负系数率、相分布和界面锐度上均优于所有基线，PRF 下降 2.3–6.4 倍。

**⚠️ 局限性**

目前仅针对稳态椭圆型守恒律，未处理时间相关系统或界面演化，需要进一步扩展到时空保守残差。

---

## 16. Measuring Cross-Task Behavioral Consistency in Language Model Agents

**arXiv ID:** 2608.13598 | [PDF](https://arxiv.org/pdf/2608.13598v1)

**作者:** Amritesh Banerjee `[一作]` (University of Massachusetts Amherst), Pranil Raichura `[通讯]` (Mantis Ai Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了行为一致性度量（BCM），用于量化语言模型代理在跨任务和同任务中的行为一致性。

**💡 创新点**

创新点在于将行为一致性从单任务复制扩展到跨任务，并通过SHAP归因向量测量轨迹的相似度，揭示一致性与成功率分离。

**🔧 技术方法**

采用特征提取、LightGBM预测成功、TreeSHAP归因、余弦相似度以及Bootstrap置信区间等技术。

**📊 数据集**

使用SWE-bench软件工程任务的约9,000条轨迹，来自六个代理系统（Claude 3.5/3.7 Sonnet、GPT-4o、SWE-agent Llama 405B/70B/8B）。

**📈 对比分析**

通过与成功率、同任务一致性以及任务难度分层的比较，BCM发现跨任务一致性与同任务一致性可分离，且在开放源代码与专有模型之间存在显著差距。

**⚠️ 局限性**

局限包括仅基于结构特征的归因、仅评估单一任务域、模型与系统相关性未完全分离、数据量不均衡以及对采样温度与基准的依赖。

---

## 17. BiasTrace: Linking Reasoning Behaviours to Biased Outputs in LLMs

**arXiv ID:** 2608.14161 | [PDF](https://arxiv.org/pdf/2608.14161v1)

**作者:** Varsha Ramineni `[一作]` (University College London), Emine Yilmaz `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发了一套细粒度的推理行为标注方案（BiasTrace），对LLM生成的Chain‑of‑Thought轨迹进行行为标注，并将这些行为与偏见输出关联，构建了大规模标注数据集；随后分析了推理行为与偏见之间的关系，并利用行为信息改进偏见评估与推理时的偏见缓解。

**💡 创新点**

创新点在于：①提出细粒度推理行为标签，识别超越显式偏见语言的隐式推理模式（如过度思考、使用外部知识等）对偏见的影响；②基于这些标签构建评估提示，在推理时过滤高风险轨迹，实现偏见缓解；③展示这些行为标签可在不同模型、提示和任务上提升偏见检测的相关性。

**🔧 技术方法**

主要技术包括：LLM Chain‑of‑Thought（CoT）推理；LLM-as-a-Judge（DeepSeek‑V3.2）进行自动标注；逻辑回归与特征分析预测偏见；行为驱动的评估提示；多样本投票与行为过滤的推理时偏见缓解。

**📊 数据集**

使用的数据集有：BBQ偏见基准（9个社会类别的结构化问答），用于生成和标注推理轨迹；COMPAS司法风险评估数据，用于检验行为标签与群体公平性贡献的关联；以及公开LLM模型（Qwen3、GPT‑OSS、Llama3等）生成的推理轨迹。

**📈 对比分析**

与基线0‑5 ordinal、0/1二值以及Fairness Reward Model相比，基于行为的评估提示在与偏见输出的相关性上提升至r=0.08‑0.41（基线为0.01‑0.19）。在推理时过滤策略（Maj‑）中，准确率保持或略升，偏见率从3.6%降至1.7%等，显示出行为标注在评估和缓解偏见方面的显著优势。

**⚠️ 局限性**

局限性包括：仅在英语、开放源码模型和结构化问答任务上验证；依赖CoT可见性，可能不适用于自由形式推理或不产生可见轨迹的模型；注释方案需进一步验证其跨模型、跨语言的通用性；对过度思考阈值等参数未做系统探究。

---

## 18. FlatLab: A Unified Methodology Framework and Simulation-Based Benchmark for Robotic Manipulation of Flat Objects

**arXiv ID:** 2608.14049 | [PDF](https://arxiv.org/pdf/2608.14049v1)

**作者:** Xingyu Zhu `[一作]` (Jilin University), Yixing Gao `[通讯]` (Jilin University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将策略生成和动作执行解耦的统一机器人平面物体操作框架，并构建了FlatLab仿真平台；

**💡 创新点**

通过模拟数据变换、对比学习以及可复用的动作原语，实现对多形状、多材质平面物体的策略自适应与泛化；

**🔧 技术方法**

利用点云编码器、对比损失、动作原语学习和机器人轨迹规划；

**📊 数据集**

使用FlatLab内部自建的104个平面物体数据集（21类）以及部分真实世界平面物体数据；

**📈 对比分析**

与单一策略基线、端到端策略等进行对比，最终在训练集、Test α、Test β上的成功率分别为81.1%、74.2%、69.0%，明显优于所有基线；

**⚠️ 局限性**

主要局限在于对仿真环境的依赖、对更复杂场景的性能评估不足，以及未充分融合大型视听语言动作模型。

---

## 19. New lower bounds for constant-weight codes via seeded bit-swap tabu search

**arXiv ID:** 2608.13906 | [PDF](https://arxiv.org/pdf/2608.13906v1)

**作者:** William Echols `[一作]` `[通讯]`, William Echols

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用种子初始化的位交换禁忌搜索构造了124个二进制常权码，显著提升了 A(n,d,w) 的下界。

**💡 创新点**

创新点在于利用已有码（同参或邻参）作为种子进行热启动，并结合长度化/缩短技术扩展搜索空间，从而突破传统禁忌搜索的性能瓶颈。

**🔧 技术方法**

核心技术为 Rosin 的改进位交换禁忌搜索、热启动种子、邻参数种子、长度化/缩短与随机采样等启发式策略。

**📊 数据集**

主要数据来源于布罗伊（Brouwer）在线表中已知的常权码，以及作者公开的 GitHub 代码库，用作种子与验证集。

**📈 对比分析**

与布罗伊在线表中的旧下界比较，论文在 124 组参数上均至少提升 1 个码字，部分提升幅度高达数十到数百，亦相应提升了 32、33、34、37 维的接触数上限。

**⚠️ 局限性**

局限性包括：依赖已有码作为种子，参数覆盖受限；搜索方法仍为启发式，缺乏最优性证明；计算成本高，未给出时间复杂度；提升幅度对更高维度仍有限。

---

## 20. AI Research Preference Models

**arXiv ID:** 2608.13940 | [PDF](https://arxiv.org/pdf/2608.13940v1)

**作者:** Thomas Simon Foster `[一作]` (FAIR at Meta), Jakob Nicolaus Foerster `[通讯]` (FAIR at Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并评估了一种用于 AI 研究代理的首选模型（RPM），通过在搜索树中对未执行的候选子节点进行排序，从而在有限的 GPU 计算预算内高效选择最有价值的实验方向。

**💡 创新点**

创新点在于将候选实验视为相对排名问题，设计了两种 RPM：仅基于推理的 RPM 和可执行小规模试验的 Agentic RPM；并将 RPM 整合到进化式树搜索框架中，显著提升了实验分配效率。

**🔧 技术方法**

技术包括 LLM-as-a-judge 的推理排名、使用 Qwen3.6-27B 等大语言模型、基于工具调用的沙箱试验、pairwise tournament 结构的候选选择，以及对上下文、候选数和推理预算的规模化分析。

**📊 数据集**

使用了 20 个来自 MLE-bench 的文本与表格任务（例如语言模型、数学、生物信息学、时间序列预测）进行端到端评估，并通过 40 个未公开图像/视频/音频任务构建离线评估数据集。

**📈 对比分析**

与无 RPM 随机选择基线相比，Inference-only RPM 的平均归一化分数从 0.684 提升至 0.711，Agentic RPM 进一步提升至 0.729，逼近验证 oracle（0.748）和测试 oracle（0.759）。两种 RPM 在 15 小时内即可达到 24 小时基线性能，计算预算降低约 1.5 倍，并在 WinoGrande、SVAMP 等单任务上刷新 SOTA。

**⚠️ 局限性**

局限性包括：推理延迟与成本需考虑；离线数据为前期贪婪搜索产生的离策略样本，存在偏倚；RPM 仅在子节点选择阶段集成，父节点或最终选择的改进有限；实验仅在单一框架和单一后端（Qwen3.6-27B）上验证，跨框架迁移尚待探索。

---

## 21. The ack3 H1 2026 DeFi Incident Dataset: Audit Scope Across 135 Security Incidents

**arXiv ID:** 2608.13792 | [PDF](https://arxiv.org/pdf/2608.13792v1)

**作者:** Josef Gattermayer `[一作]` (ack3), Arman Bašović `[通讯]` (Czech Technical University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对2026年上半年公开的DeFi安全事故数据集进行分析，检索并归档每起事故的事前审计记录，评估事故路径是否落在已审计的代码、版本与系统边界内，并统计审计年龄、时间分布及项目类型。

**💡 创新点**

首次将审计标签的“项目级”视角与“范围级”视角区分开来，揭示大量事故路径实际上位于已审计范围之外，量化此现象对损失的影响，并提出审计报告应明确仓库、提交、部署地址、排除项等细节。

**🔧 技术方法**

手工审阅审计报告、定义事故路径与审计范围的匹配规则，计算审计年龄（按月差值），进行损失加权敏感性分析，绘制时间与项目类型分布图。

**📊 数据集**

ack3发布的2026年上半年DeFi事故数据集（135起事故，约9.4亿美元损失）。

**📈 对比分析**

通过定量指标（事故数、损失额、审计年龄）比较不同审计范围状态：在已审计的68起事故中，46起事故路径完全超出审计范围，损失占比94.4%；内部范围的事故占比仅为5.6%；外部范围的事故占比为67.6%（按数量）。该研究显示大规模事故对损失加权结果有显著影响，去除两起最大事故后，外部范围损失占比大幅下降。

**⚠️ 局限性**

仅覆盖公开报告的事故，未包含未被披露或私下解决的事件；损失估计存在价值与回收差异；单一编码的范围判定缺乏交叉验证；数据仅限于半年，无法推断整体审计效果或因果关系。

---

## 22. QuaSAR: Quantization Compensation via Stable Activation-Aware Rank Truncation

**arXiv ID:** 2608.14149 | [PDF](https://arxiv.org/pdf/2608.14149v1)

**作者:** Lin-Fa Lee `[一作]` (National Yang Ming Chiao Tung University), Kuo-Hei Yeh `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种训练无关的层级残差补偿方法，在低比特W4A4量化下通过截断伪逆解决数值崩溃问题。

**💡 创新点**

发现并修复了激活量化导致的Gram矩阵秩亏崩溃误判，提出无参数截断伪逆以恢复被误丢弃的高价值补偿层。

**🔧 技术方法**

使用层级线性补偿、截断伪逆求解、低秩分解加激活感知量化、R²门控机制以及联合压缩技术。

**📊 数据集**

在ImageNet（ImageNet‑1K）上评估ViT‑B、DeiT‑T、Swin‑T、ResNet‑50等模型。

**📈 对比分析**

与QwT、QwT*、QwT‑v2、IGQ‑ViT等PTQ方法对比，ViT‑B/W4A4无训练实现81.42% Top‑1，压缩后54.7 MB达到80.26% Top‑1，性能领先。

**⚠️ 局限性**

仍无法在CNN层级稳定超越块级补偿；低位宽如W3A3下的稳定性未解决；方法对不同量化器的通用性需进一步验证。

---

## 23. The Integer Alibi: Localizing Cross-Kernel Divergence in INT8-Quantized LLM Inference

**arXiv ID:** 2608.13756 | [PDF](https://arxiv.org/pdf/2608.13756v1)

**作者:** Teng-Ruei Chen `[一作]` `[通讯]` (Krixvon), Teng-Ruei Chen (Krixvon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在vLLM推理引擎中，对同一量化线性层（INT8 GEMM）替换实现（CUTLASS vs Triton），在固定模型、提示、硬件与量化配置下，评估两核在层级和端到端推理中的数值差异，并通过整数完整性（integer alibi）定位差异源。

**💡 创新点**

提出整数完整性检验作为核间可比性判定的可验证控制，并演示通过将权重尺度改为 2^k 近似可恢复位级一致性；同时提供一套可复现的“conformance procedure”与预注册流程。

**🔧 技术方法**

使用整数累加 INT32、bfloat16 归一化、CUDA TensorCore 的 CUTLASS 与 Triton 量化 GEMM 实现；利用预注册的层级预测、对比实验、教师强制回放与 ULP 距离度量。

**📊 数据集**

采用 Qwen3-1.7B 与 8B 两规模模型（W8A8 量化），以 64 条 320‑token 的 WikiText‑2 硬性提示进行推理。

**📈 对比分析**

通过层级逐层传递相同 INT8 乘法输入到两核，测量 2^24 边界内的累加完整性、实规模与 2^k 规模下的输出差异；对端到端生成序列进行比对。结果显示：实规模下无序列一致；2^k 规模下全部 448 层位点与 16,384 个位置实现位级一致；每层最大差距 ≤ 1 bfloat16 ULP。

**⚠️ 局限性**

局限包括：仅在单一 RTX‑4090 GPU 与 Qwen3 两规模上实验；未对更深模型或其他 GPU 架构进行验证；未提供正负对照测试导致检验灵敏度未知；同一核下 2^k 规模仍有 2.9% 位置差异未定位；尺度转换干扰模型精度与吞吐率未评估；且实验容器无法完全复现。

---

## 24. Context Aware AI Assistant and AR Interface for Lunar Extravehicular Activity (EVA) Procedural Guidance

**arXiv ID:** 2608.13589 | [PDF](https://arxiv.org/pdf/2608.13589v1)

**作者:** Rodrigo Gallardo `[一作]` (MIT), Skylar Tibbits `[通讯]` (MIT)

**通讯引用:** 3006 | [OpenAlex ID](https://openalex.org/A5085862023)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个名为GAIN-AI的情境感知AI助手及最小化AR显示界面，用于在模拟月球行走（EVA）中为宇航员提供实时的程序指导；

**💡 创新点**

创新点在于将大型语言模型（LLM）与结构化JSON上下文（程序文档、实时遥测、故障处理协议）实时耦合，并将模型输出重新整理为可执行的Goal-Task-Verification三维卡片，从而在有限视场和操作条件下降低认知负荷；

**🔧 技术方法**

使用技术包括：大规模预训练语言模型（gemini-2.5-flash-lite）、结构化JSON上下文注入、Microsoft HoloLens 2的AR头显显示、以及基于卡片的UI设计；

**📊 数据集**

使用数据集为111个合成EVA场景，涵盖常规、单一故障、多重故障、车辆错误和边界阈值四大类；

**📈 对比分析**

评价方法通过对每个场景的错误检测、正确行动、紧急度校准、分层排序四维度打分，总分10分；性能在常规场景达到10/10，在单一故障场景为8.15/10，随着故障复杂度上升性能逐步下降（多重故障7.73/10，车辆错误3.96/10，边界阈值5.24/10）；

**⚠️ 局限性**

局限性包括：在多重故障或边界阈值条件下，模型的行动排序和精确阈值识别能力不足；缺乏实时遥测的动态更新支持；尚未在真实宇航员操作中评估认知负荷和任务完成度；并未与传统检查表进行完整对比基线。

---

## 25. Source-Agnostic Image Translation Based on Latent Aware Adaptive Masking

**arXiv ID:** 2608.14046 | [PDF](https://arxiv.org/pdf/2608.14046v1)

**作者:** Tomislav Dobrički `[一作]` (Chung-Ang University), Byung-Woo Hong `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种源无关的无监督图像到图像翻译框架，利用扩散模型在每个噪声层动态生成二值掩码并通过掩码引导反向扩散，最终实现高质量翻译。

**💡 创新点**

核心创新在于引入时间相关的统计阈值（基于目标域预测误差均值与方差）来动态计算掩码，实现自适应遮蔽；并结合RePaint式迭代修复提升图像连贯性。

**🔧 技术方法**

采用扩散概率模型（DDPM/DDIM）、自适应掩码计算、统计阈值设定、以及RePaint掩码修复技术。

**📊 数据集**

在AFHQ（猫→狗、野生→狗）和CelebA-HQ（男→女）等公开数据集上进行实验，并在PIE‑Bench进行文本驱动编辑评估。

**📈 对比分析**

与ILVR、SDEdit、EGSDE、CycleDiffusion等现有无监督或源域监督方法比较，PM‑Edit在FID、KID、SSIM、LPIPS等指标上取得更优或竞争性的结果，体现了更好的真实性与忠实度。

**⚠️ 局限性**

局限性主要是假设源域与目标域属于同一模态；在跨模态或极端域差异情况下的表现尚未验证，且统计阈值对模型与数据分布的敏感性需进一步研究。

---

## 26. Resource-Adaptive Primal-Dual Learning for One-Warehouse Multi-Store Systems with Censored Demand

**arXiv ID:** 2608.14096 | [PDF](https://arxiv.org/pdf/2608.14096v1)

**作者:** Jiameng Lyu `[一作]` `[通讯]` (Fudan University), Jiameng Lyu (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了有限时限、非补充共享库存、需求未知且观测为遮蔽（censored）的单仓库多店分配问题，提出资源自适应原始‑对偶学习框架。

**💡 创新点**

创新点在于将固定目标学习替换为在线跟踪随资源状态变化的原始‑对偶重求解路径，并通过“anchor”双向稳定化实现对偶梯度，最终给出对数阶（O(log T)）的累积回退证明。

**🔧 技术方法**

技术上基于原始‑对偶重求解（fluid program）与期望销售几何分析，使用水分配投影实现物理可行性，利用梯度估计与资源上限裁剪的安全盒，以及两侧谐波步长调度。

**📊 数据集**

数据集为两组仿真实验：S1同质化（N=2，正态需求约束在[0,175]）与S2异质化（N=6，均匀分布+高斯Copula交叉相关），无真实外部数据。

**📈 对比分析**

与现有Double Binary Search（DBS）基准比较，在所有24个有限时限、库存情景中均表现出更低平均总成本，相关率误差在0.3%–12%范围内，且对资源水平的敏感性较低。

**⚠️ 局限性**

局限性包括仅在已知需求支持与密度下界的前提下实现；实验仅为仿真，缺乏真实运营案例；对极端需求分布或高阶相关性尚未评估；实现需要较多参数调优。

---

## 27. Musical Mirrors: The LLM as Sounding Board in Songwriting

**arXiv ID:** 2608.13944 | [PDF](https://arxiv.org/pdf/2608.13944v1)

**作者:** Xiao Xiao `[一作]` `[通讯]` (Institute for Future Technologies), Xiao Xiao (Institute for Future Technologies)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

作者以自身创作的16首歌曲为素材，将大型语言模型设为解释性回声板，帮助作者在创作过程中深化对自身作品的内部共振。

**💡 创新点**

创新点在于将LLM从传统的生成工具转变为长期校准的共振伙伴，展示了通过用户侧持续校准能够放大内部共振并识别两种失效模式（顺从漂移与魔法化解读）。

**🔧 技术方法**

使用技术包括对话式LLM（ChatGPT GPT‑4o、GPT‑5 系列、Claude、Gemini）、自定义模式设定、持续校准流程和情感共振分析。

**📊 数据集**

数据集由作者在 2025‑07 至 2026‑03 期间创作的 16 首原始歌曲（英语、法语及其他语言）以及对应的约 6,000 条对话日志构成。

**📈 对比分析**

比较方法为对不同模型版本与校准状态下的对话进行定性分析，展示了共振放大与失效模式的差异，未给出量化性能指标。

**⚠️ 局限性**

局限性包括依赖单一作者的第一人称案例研究、需要长期人工校准、在无校准时模型易出现顺从或过度解释的风险。

---

## 28. CipherSight: Robust Website Fingerprinting via Record-Resource Semantic Supervision under Distribution Shifts

**arXiv ID:** 2608.13905 | [PDF](https://arxiv.org/pdf/2608.13905v1)

**作者:** Runhan Song `[一作]` (Harbin Institute of Technology), Zhiyu Hao `[通讯]` (Zhongguancun Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于TLS记录的层次Transformer框架CipherSight，用以在时间、地域漂移及开放域情境下实现稳健的HTTPS网站指纹识别。

**💡 创新点**

创新点在于将记录级特征与多流结构层次编码相结合，并利用训练时的资源级先验监督与语义蒸馏提升泛化性能。

**🔧 技术方法**

采用了块对角自注意力的层次Transformer、掩码记录建模、LoRA微调、资源级监督以及教师蒸馏等技术。

**📊 数据集**

使用了六个HTTPS数据集，覆盖2008个网站，分别在不同日期与地区收集，包含闭域、时间漂移、地域漂移和开放域测试集。

**📈 对比分析**

在闭域、时间漂移、地域漂移和开放域评估中，CipherSight分别在闭域准确率达到95.41%，时间漂移92.99%，地域漂移约91%，开放域AUROC 97.64%，均明显优于现有Tor及HTTPS基线。

**⚠️ 局限性**

局限性在于训练阶段需要SSL密钥以获取资源对齐，且对HTTP/3等新协议支持尚待验证，数据预处理与窗口策略仍有提升空间。

---

## 29. Exploring ESC Winners with Nested Diagrams

**arXiv ID:** 2608.13630 | [PDF](https://arxiv.org/pdf/2608.13630v1)

**作者:** Anurag Sharma `[一作]` (University of Kassel), Gerd Stumme `[通讯]` (University of Kassel)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了可与scikit-learn兼容的Python库ConceptFlow，用于Formal Concept Analysis的概念扩展与嵌套线图可视化，并应用于欧洲歌唱大赛获奖者数据。

**💡 创新点**

将多值形式上下文拆分为外层与内层概念尺度，利用子直积构造嵌套线图，并提供交互式D3.js可视化，首次在正式概念分析中展示投票与音乐属性的直接关联。

**🔧 技术方法**

Python、scikit-learn估计器接口、DimFlux坐标布局、概念扩展与子直积算法、D3.js交互渲染。

**📊 数据集**

1975-2025年欧洲歌唱大赛50位获奖者的投票分数、地区文化政治关联与歌曲BPM、调性元数据。

**📈 对比分析**

未与传统统计方法或其他FCA工具直接对比，主要通过可视化展示蕴含关系，展示了在50个样本中能揭示多维投票与音乐属性的蕴含，性能表现以交互可视化的实时渲染为主。

**⚠️ 局限性**

仅作为演示案例，样本量有限，未提供完整统计检验或基准；未实现蕴含基计算与交互查询，缺乏自动化蕴含推断功能。

---

## 30. Learning to Run Power Networks: Effective AlphaZero-inspired Topological Control

**arXiv ID:** 2608.14114 | [PDF](https://arxiv.org/pdf/2608.14114v1)

**作者:** Lukas Zetto `[一作]` (Karlsruhe Institute of Technology), Qiong Huang `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过将AlphaZero框架与电网物理模型结合，利用蒙特卡罗树搜索（MCTS）在IEEE‑14电网上实现主动拓扑重构，以预防可再生能源波动引发的拥塞与失稳；

**💡 创新点**

创新点在于发现简化的二元存活奖励和最小化观测空间能显著提升搜索效率和稳定性，并提出一套“极简”AlphaZero实现蓝图，证明纯强化学习不足以满足电网安全要求；

**🔧 技术方法**

使用技术包括AlphaZero、MCTS、强化学习（PPO、Rainbow PPO对照）、动作空间约简（对称、N‑0、N‑1）、奖励函数设计（AlphaZero原始、D3QN‑2022二元、复合等）、Grid2Op仿真框架以及Python深度学习库；

**📊 数据集**

使用的数据集为IEEE‑14电网的Grid2Op仿真环境，包含1004条长度为8064步的chronics，并在10%保留集上评估泛化；

**📈 对比分析**

通过与DQN、PPO、Rainbow PPO等模型基准进行训练比较，AlphaZero基线在平均存活步骤约7486（92.83%），改进版（使用D3QN‑2022奖励）达7937步（98.43%），显著超越Rainbow PPO的91.80%存活；

**⚠️ 局限性**

局限性包括训练过程需要约4000万步，计算成本高；仅在小型IEEE‑14网格上验证，尚未在更大规模网络上证明可扩展性；缺乏解释性和对多目标奖励的鲁棒性研究。

---

## 31. RIVERPlace: Repairing Interconnect Violations with Efficient Retiming and Incremental Placement for AQFP Circuits

**arXiv ID:** 2608.13780 | [PDF](https://arxiv.org/pdf/2608.13780v1)

**作者:** Robert S. Aviles `[一作]` (University of Southern California), Peter A. Beerel `[通讯]` (University of Southern California)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

RIVERPlace 提供了一种集成的 AQFP 电路间连线违规解决框架，结合长线流水化、重定时与增量布置来优化布局。

**💡 创新点**

其创新点在于 Buffer Cut Insertion（BCI），将违规管线化归结为最大拓扑割问题，并通过最小 s‑t 割求得多行全局最优解，同时与可视化重定时和 LP 布置协同。

**🔧 技术方法**

使用了增量线性规划（LP）布置、基于行的重定时、缓冲器可视化重定时、最大拓扑割（转化为最小 s‑t 割）以及间距估计等技术。

**📊 数据集**

实验数据来自公开的 AQFP 基准套件以及更大规模的开源 AQFP 网表，涵盖多种尺寸与复杂度的电路。

**📈 对比分析**

与 GORDIAN、TAAS、DLPlace、SuperFlow 等现有方法比较，RIVERPlace 在插入 JJ、逻辑深度、面积、运行时间等指标上均实现了超过一位数的缩减，并首次完成完整后路由与时序闭合。

**⚠️ 局限性**

局限性包括对更大规模或更复杂时钟分布的验证不足，以及对行宽与路由密度的依赖，且未将流水化早期融入综合流程。

---

## 32. Agentao: A Governed Local-First Runtime for Tool-Using LLM Agents

**arXiv ID:** 2608.13574 | [PDF](https://arxiv.org/pdf/2608.13574v1)

**作者:** Bo Jin `[一作]` (Third Research Institute of the Ministry of Public Security), Xin Tong `[通讯]` (People’s Public Security University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了 Agentao，一个本地优先的受管控 LLM 代理运行时，将模型生成的动作与宿主授权执行分离。

**💡 创新点**

将权限模式、协议组合治理、状态纪律和可观测性统一到层次化架构；通过 host contract + permission‑mediated tool pipeline，实现工具调用的显式权限检查、确认和审计；支持协议驱动工具注册与治理。

**🔧 技术方法**

使用结构化事件流、沙箱执行、权限策略引擎、可插拔插件、MCP/ACP 协议、子代理、记忆管理和重放日志等技术。

**📊 数据集**

未在论文中使用任何特定数据集；主要以公开代码实现为实验基础。

**📈 对比分析**

论文未给出定量比较或性能评测，缺少与现有框架的基准对比。

**⚠️ 局限性**

缺乏形式安全保证、依赖工具描述与宿主执行的正确性、未进行量化评估、对供应链与沙箱等环境假设较强。

---

## 33. A dataset of article processing charges from 14 scholarly publishers, 2019-2025

**arXiv ID:** 2608.14116 | [PDF](https://arxiv.org/pdf/2608.14116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 34. Retrieval Grounding Latent Reasoning for Dense Retrieval

**arXiv ID:** 2608.14107 | [PDF](https://arxiv.org/pdf/2608.14107v1)

**作者:** Gang Zhou `[一作]` (Beijing University of Posts and Telecommunications), Xiaolong Zheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

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

## 35. DepressionAgent: Reading, Listening, Seeing, and Deliberating Multimodal Evidence for Depression Risk Assessment

**arXiv ID:** 2608.13891 | [PDF](https://arxiv.org/pdf/2608.13891v1)

**作者:** Fangjie Zhu `[一作]` (Shenzhen MSU-BIT University), Xiping Hu `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于多模态证据推理的抑郁风险评估框架 DepressionAgent

**💡 创新点**

创新点在于将特征融合转变为证据层面推理，采用分支对话、跨模态仲裁、冲突与风险反思等代理机制

**🔧 技术方法**

使用多模态观察、支持-挑战代理、跨模态判断代理、冲突反思和风险反思等代理式推理技术，搭配大型预训练模型（如 Qwen3-Omni、GPT‑5.5、Gemini 3.1 Pro）

**📊 数据集**

在公开的 D‑Vlog 与 LMVD 两个真实世界视频数据集上进行评估

**📈 对比分析**

与多种基于特征融合的监督模型对比，零监督下可达 95% 以上准确率，F1 约 93%，在不需要任务专用训练的前提下实现了与甚至超过当前 SOTA 的表现

**⚠️ 局限性**

局限主要在于多模态感知错误、证据归因与上下文误判、对文本的过度依赖以及额外的推理开销

---

## 36. AdsWorldEngine: A Self-Evolving Conversational Advertising Agent through Orchestrator and Tool Coevolution

**arXiv ID:** 2608.13833 | [PDF](https://arxiv.org/pdf/2608.13833v1)

**作者:** Simiao Zuo `[一作]` (Microsoft), Denis Charles `[通讯]` (Microsoft)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套端到端的会话广告系统 AdsWorldEngine，涵盖机会门控、意图生成、工具调用、广告板块生成与离线评估。

**💡 创新点**

创新点在于：① 通过奖励驱动的迭代演员-工具共进化训练，使广告工具在奖励轨迹中自行提升；② 采用基于标签的判断建模配合成本敏感 GRPO，以处理非对称错误成本。

**🔧 技术方法**

技术实现包括：大型语言模型驱动的 Agent、GRPO 强化学习、DPO 风格的偏好学习、思维轨迹生成与反射过滤、成本敏感 GRPO、检索/相关性/排名/定价等广告工具。

**📊 数据集**

使用了 30k+ 人工标注的机会门控与相关性数据集，并在 Qwen3‑30B‑A3B‑Thinking 基础上进行训练。

**📈 对比分析**

通过离线评估器（相关性、Diversity）和 20 天的在线 A/B 测试进行比较，离线提升多样性 62.9%、相关性 82.3%；在线上 RPM 提升 22%、广告覆盖率提升 74%。

**⚠️ 局限性**

局限性包括：高度依赖人工标注与反射过滤、工具改进受限于现有检索/排名模型、以及在复杂多轮对话中仍可能出现冗余或误触发。

---

## 37. CMCNet: Aligning Ultrasound Image Embeddings with Textual TI-RADS Representations for Fine-Grained Thyroid Classification

**arXiv ID:** 2608.13939 | [PDF](https://arxiv.org/pdf/2608.13939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 38. Scaling Creative Writing Beyond Story-Centric Data with Attribute-Guided Genre Expansion

**arXiv ID:** 2608.13947 | [PDF](https://arxiv.org/pdf/2608.13947v1)

**作者:** Hwan Chang `[一作]` (Chung-Ang University), Hwanhee Lee `[通讯]` (Chung-Ang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含13种创意写作类型（如故事、歌词、游戏设计等）的50K示例数据集，并提出基于属性引导的跨域写作扩展框架；

**💡 创新点**

创新点在于将主题多样性与体裁结构约束分离：先从人类创作的故事提示中抽取主题，再利用手工整理的体裁属性（如押韵、节奏、剧情结构）控制生成的格式和风格；

**🔧 技术方法**

核心技术包括：LLM（GPT‑5‑mini）用于主题抽样、查询生成与质量过滤；Qwen3‑235B‑A22B‑Thinking生成回答；Qwen3‑30B‑A3B‑Instruct做质量评估；对生成的查询采用“属性引导元提示”模板；SFT（LoRA）微调模型；以及可视化与多维度评估（t‑SNE、NoveltyBench、外部评测）等；

**📊 数据集**

使用了 Reddit 论坛的故事提示作为种子数据，手工编制每种体裁的5–15个属性；随后生成的 Multi‑Genre Collection（13类、50K条）作为主要训练集；比较实验亦使用了 DeepWriting、LongWriter 等现有写作数据集；

**📈 对比分析**

在 Arena Hard、WritingBench 与 Multi‑Genre 本身的三大基准上对比，SFT后的模型在所有基准上均显著优于基线（尤其在创意写作与跨体裁泛化上提升≥15%），并且在多体裁覆盖度提升时，输出多样性（Novelty）呈现单调上升；

**⚠️ 局限性**

局限性包括：依赖LLM生成与评估，可能引入模型偏差；属性抽取与人工校对工作量大；仅覆盖13种体裁，无法完全代表所有创意写作形式；并且实验主要集中在英语数据，跨语言推广尚未验证。

---

## 39. Stochastic Control Policies for Robust Molecular Transition Path Sampling

**arXiv ID:** 2608.13800 | [PDF](https://arxiv.org/pdf/2608.13800v1)

**作者:** Jingqian Liu `[一作]` (University of Illinois Urbana--Champaign), Ge Liu `[通讯]` (University of Illinois Urbana--Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出两种基于MD roll‑out 的随机控制策略（FS‑TPS 与 LaS‑TPS）来改进转移路径采样，使得路径生成更物理合理、转化成功率更高且对随机初始化更鲁棒。

**💡 创新点**

创新点在于：①将控制力建模为状态相关的高斯分布（FS‑TPS），并通过熵正则化避免过早收敛；②通过变分信息瓶颈在低维潜在空间注入噪声，再通过共享解码器产生低秩、跨原子相关的力扰动（LaS‑TPS），从而在保持能量优势的同时显著提升转化成功率和路径多样性。

**🔧 技术方法**

使用的技术包括：最大熵控制与概率推理框架、重参数化采样、熵正则化的路径测度匹配（log‑variance 目标）、变分信息瓶颈（VIB）、优先级重放缓冲、以及对齐和能量评估的标准化评测工具。

**📊 数据集**

实验数据集：三种生物分子——丙氨酸二肽（22 原子）、Chignolin（166 原子）和 BBL（711 原子），每种系统在多次随机种子下进行训练和评估。

**📈 对比分析**

与现有方法（传统 MD、SMD、PIPS、TR‑LV、TPS‑DPS）进行对比，结果显示：FS‑TPS 在转化成功率 (THP) 上提升约 20%–30%，RMSD 下降 40% 以上；LaS‑TPS 在三种系统上均获得最低的转化状态能量 (ETS)，THP 最高、RMSD 与 FS‑TPS 相当，并且在路径多样性（mode coverage）上大幅领先。两种方法均显著降低了对网络初始化的敏感性。

**⚠️ 局限性**

局限性包括：①控制目标与能量质量之间存在权衡，需调节 KL 正则化或熵系数；②单维潜在空间可能不足以捕捉复杂协调运动，导致训练不稳定；③虽然随机策略提升了鲁棒性，但仍不保证每条路径都到达终点；④方法依赖于大量MD roll‑outs，计算成本相对较高。

---

## 40. PPOM: Marginalizing Patch-Grid Phase for CLIP-Based Generalizable Vision-Language Prompt Tuning

**arXiv ID:** 2608.13969 | [PDF](https://arxiv.org/pdf/2608.13969v1)

**作者:** Liang Wang `[一作]` (Shanghai University), Yan Peng `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练、仅在推理时使用的 Patch‑Phase Orbit Marginalization (PPOM) 方法，利用反射填充和步长相位移生成多视角，对 CLIP‑基视觉‑语言模型的 prompt 学习进行后处理，以降低非重叠 patch tokenization 对预测结果的相位敏感性。

**💡 创新点**

创新点在于：① 把视觉采样的相位视为推理时的无关变量；② 构造基于 ViT patch stride 的九视角相位轨道；③ 采用方向平衡的族级边际化（horizontal、vertical、diagonal 三种族）消除相位导致的偏差，既保持原 prompt 训练结果，又提升鲁棒性；④ 该方法不需要额外训练或参数，直接在已训练的 prompt host 上加插件。

**🔧 技术方法**

主要技术包括：CLIP ViT‑B/16 固定视觉编码器；相位轨道构造与反射填充；对九个视角 logits 进行族级平均；在测试时一次性批量处理 9b 视图；与 CoOp、KgCoOp、PromptSRC、MMRL 等 prompt host 的无缝对接。

**📊 数据集**

使用的评测数据集有 11 个：ImageNet、Caltech101、Oxford Pets、Stanford Cars、Flowers102、Food101、FGVC Aircraft、SUN397、DTD、EuroSAT、UCF101，涵盖 2D 图像分类与视频帧分类。

**📈 对比分析**

通过与各 host 原始版本的基准对比（基类、样本类、调和平均 HM），在 11 个数据集上平均提升 HM 约 0.5%–0.8%，在跨数据集零样本迁移实验中，PPOM 在 ImageNet 源准确率和目标平均准确率上均有提升。对比方法为同一 checkpoint 在无训练、仅推理阶段加 PPOM，保证公平。

**⚠️ 局限性**

限制主要在于：推理时需要生成 9 个视角，导致推理时间和显存略有增加；虽然比 fine‑tune 成本低，但对实时或资源受限场景仍有一定负担；方法主要针对相位敏感性，可能无法解决其他来源的预测不稳定性。

---

## 41. Hybrid Quantum-inspired Kolmogorov-Arnold Networks for Privacy-Aware Federated Biosignal Learning

**arXiv ID:** 2608.13914 | [PDF](https://arxiv.org/pdf/2608.13914v1)

**作者:** Chun-Hua Lin `[一作]` (National Taiwan University), Hsi-Sheng Goan `[通讯]` (National Taiwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `109c2b71-d051-425c-831f-0c544c24280d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并验证了一种混合量子启发的Kolmogorov–Arnold网络（HQKAN）用于隐私友好的联邦心电图（ECG）分类，替代传统多层感知机（MLP）实现更高效的参数与通信开销；

**💡 创新点**

创新点在于将QKAN的稀疏激活与全连接编码器/解码器相结合，形成自动编码器风格的轻量级模型，显著减少参数量并提升对非IID分布的鲁棒性；

**🔧 技术方法**

采用了Hybrid Quantum-Inspired Kolmogorov–Arnold Network（HQKAN）与多层感知机（MLP）两种分类器，使用FedAvg进行联邦训练，并利用Dirichlet分布模拟客户端标签偏移；

**📊 数据集**

使用了MIT-BIH（五类心律）与INCART（三类心律）两个公开ECG基准数据集，进行分割与标准化处理；

**📈 对比分析**

在多客户端（8、16、32）及IID/非IID设置下，HQKAN在宏观F1、Cohen’s κ和Brier分数上均优于MLP，同时通信成本降低约25%–36%，参数量减少约37%–45%；

**⚠️ 局限性**

局限性包括实验仅基于FedAvg聚合，未探讨更复杂的联邦优化或对抗攻击；且在极端数据不平衡或更大规模数据中心场景下的可扩展性与安全性仍待验证。

---

## 42. Kolmogorov-Arnold Networks for Spatially Independent Multispectral Land Classification

**arXiv ID:** 2608.13769 | [PDF](https://arxiv.org/pdf/2608.13769v1)

**作者:** Katherine L. Bauer `[一作]` (University of Alberta), Arturo Sanchez-Azofeifa `[通讯]` (University of Alberta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文利用Landsat 8 OLI影像，对Edmonton与Calgary两市的农业、城市、水域、森林、裸土五类土地覆盖进行像素级分类，并首次将Kolmogorov‑Arnold网络（KAN）与随机森林（RF）和多层感知器（MLP）进行对比。

**💡 创新点**

创新点包括：①首次在跨城市情境下应用KAN；②比较使用归一化原始波段与光谱指数两种特征配置对KAN性能的影响；③从参数效率与可解释性两方面评估KAN相较于RF/MLP的优势。

**🔧 技术方法**

技术方法包括：PCA降维、最小‑最大归一化、光谱指数（SAVI、NDBI、NDWI）、RF与MLP超参网格搜索、KAN的基于样条函数的正则化训练、PyKAN可视化、SHAP解释。

**📊 数据集**

数据集：2022年7月的Landsat 8 OLI Level‑2影像；Edmonton区域3,000个人工标注点（80/20训练/验证），Calgary区域600个独立测试点；5类LULC。

**📈 对比分析**

评估方法：整体准确率（OA）、宏平均F1、IoU、95%置信区间，并在Edmonton内部以及Calgary跨城测试；结果显示，归一化KANN在Calgary达到97.3% OA，与RF相当，但参数量仅约1,400，远低于RF的17,444；在Edmonton，归一化MLP最高达98.7% OA，KAN稍逊。

**⚠️ 局限性**

局限性：仅包含两座加拿大城市，缺乏季节与多时相；仅使用光谱波段与常用指数，未引入纹理、地形或物理模型；评价仅以参数计数衡量效率，未评估训练/推理时间；跨域泛化受光谱指数敏感性限制。

---

## 43. E-S2Feat:Semantic-Guided Spiking Local Feature Detection and Description for Event Cameras

**arXiv ID:** 2608.14027 | [PDF](https://arxiv.org/pdf/2608.14027v1)

**作者:** Yang Yi `[一作]` (National University of Defense Technology), Dewen Hu `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了基于脉冲神经网络的E‑S2Feat框架，用于事件相机的局部特征检测与描述，并在资源受限平台上实现低功耗实时推理；

**💡 创新点**

创新点包括：①首次将SNN应用于事件特征的联合检测与描述；②针对不同网络模块设计模块特定脉冲激活（Spiking‑GELU、Spiking‑SiLU、MSF多级脉冲激活），实现低位量化同时保持表征能力；③引入语义引导特征调制机制（二值语义分割）抑制天空等几何不稳定区的噪声，提升关键点几何稳定性与描述符辨识度；

**🔧 技术方法**

采用事件到多通道时间表面(MCTS)编码、Spiking‑MaxViT骨干+FPN特征金字塔、单步脉冲推理、Surrogate Gradient训练、语义分割头、关键点响应调制与描述符融合模块；

**📊 数据集**

在ECD、EDS事件位姿估计数据集、DSEC语义训练集、TUM‑VIE视觉惯性里程计数据集上进行实验；

**📈 对比分析**

与SuperEvent、LLAK、SD2Event等基线相比，E‑S2Feat在ECD/EDS位姿估计AUC@20°分别达到59.6%/64.6%（比SuperEvent提升约13%/15%），在TUM‑VIE VIO中将ATE从1.71cm降低至0.96cm；在能耗方面，SNN版理论能耗仅为基线的20.9%，实现约4.8倍能耗提升；

**⚠️ 局限性**

局限性包括：单步脉冲编码仍受量化误差限制；SNN在通用GPU上的软件仿真效率不及专用神经形态硬件；语义分割采用二值方案，难以精确分割动态物体；对高质量语义标注依赖，未在多动态场景下充分验证；

---

## 44. Engineering Reliable Coding Agents: Evaluating and Operating the System Around the Model

**arXiv ID:** 2608.13867 | [PDF](https://arxiv.org/pdf/2608.13867v1)

**作者:** Stephanie Jarmak `[一作]` `[通讯]`, Stephanie Jarmak

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作对 AI 代码生成代理的可靠性进行系统性评估与实践设计，提出可靠性依赖链模型，构建包含 206 条可靠性记录的目录，提供可复现的评估与操作协议，并将实验结果与工业部署案例结合；

**💡 创新点**

创新点在于：①将模型能力与系统基础设施分离，形成可靠性依赖链；②系统化多声部证据审计与记录账本；③将评估、操作、检索、审核、资源分配等六大层面整合为一套可执行的实践与工具；

**🔧 技术方法**

采用多源文献检索与结构化多声部回顾、实验设计与统计功效分析、对照实验与 ablation、成本/准确率联合评估、持续集成与监控工具；

**📊 数据集**

使用 SWE‑bench、SWE‑bench Verified 等公开 benchmark，结合内部大型代码库与真实提交数据；

**📈 对比分析**

通过对照实验、重复运行与配对差分，结合统计功效分析评估组件对准确率与成本的影响；实验表明检索、工具、记忆等组件在不同配置下可提升 1–10% 的准确率，成本提升可从几倍到十几倍不等，需权衡；

**⚠️ 局限性**

局限性包括：证据来源覆盖仍不完整，实验规模受资源限制，模型版本漂移、prompt 变异等外部因素难以完全控制，实践对不同企业规模与语言生态的迁移性尚待验证。

---

## 45. FabDreamer: Exploring the Image-to-Physical Workflow Through AI-Assisted Layered Fabrication

**arXiv ID:** 2608.13665 | [PDF](https://arxiv.org/pdf/2608.13665v1)

**作者:** Chenfeng Gao `[一作]` (Northwestern University), Danli Luo `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了 FabDreamer，一套将任意图像转化为可层层激光切割制品的端到端 AI 辅助工作流，分为分层、创作和制备三阶段。

**💡 创新点**

创新点在于：① 在工作流不同阶段按错误可恢复性调节 AI 主导程度；② 将物理约束反馈嵌入创作视图，既防止错误又激发设计创新；③ 通过可重用的几何原语实现跨领域迁移，让使用者自行赋予专业语义。

**🔧 技术方法**

核心技术包括：多模态 VLM（Gemini 3.1 Pro）进行场景分析与自然语言交互；SAM2 进行像素级分割；Gemini Flash 负责图像修复、生成与重建；生成式提取（inpainting + 生成模型）作为后备；Three.js 与 React 构建实时 3D 预览；FastAPI 提供后端服务。

**📊 数据集**

评估数据集主要为 24 张公开图像（12 张真实摄影，12 张 AI 生成），以及 25 条手工切割教程和 6 名跨领域实践者的真实项目图像。

**📈 对比分析**

通过三轮实验（原型评估、专家评审、跨领域用户研究）与手工切割基准对比，显示所有 12 个制品成功切割、组装无结构失败；系统每层约 39 秒，总耗时 3–4 分钟；用户满意度最高的功能为 3D 预览（平均 4.9/5）。

**⚠️ 局限性**

局限性包括：① 生成式分割在细部、遮挡与多层叠加时仍可能产生错误；② 仅检测几何约束，无法捕捉材料、装配、打印规范等领域特定限制；③ AI 主导程度仍需在不同专业背景中进一步量化；④ 缺乏与激光切割机参数的自动同步；⑤ 仅在单次使用中评估，缺乏长期使用与学习曲线的纵向研究。

---

## 46. A Two-Validator Web Interface for Structured Geometry Figure Annotation

**arXiv ID:** 2608.13569 | [PDF](https://arxiv.org/pdf/2608.13569v1)

**作者:** Sabin-Codrut Badea `[一作]`, Adrian-Marius Dumitran `[通讯]` (University of Bucharest)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5107616090)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个轻量级、无服务器的 Web 界面，用于验证 AI 生成的几何图形描述、裁剪图像和校正题目文本。

**💡 创新点**

提出了对 FormalGeo CDL 的高层次复合谓词改进，显著降低描述长度，并实现了双人顺序校验与完整编辑日志。

**🔧 技术方法**

使用了 Web 技术（Cropper.js、MathJax、前端编辑器）、无服务器部署以及自定义的 CDL 语法。

**📊 数据集**

使用了 483 篇罗马尼亚奥林匹克几何题的扫描文档及 AI 生成的 CDL 结果进行验证。

**📈 对比分析**

通过两轮注释比较，最终仅 67 篇题目无需修改；校验一致率约 14.3%，说明第二轮校正主要是细化而非纠正错误。

**⚠️ 局限性**

限制在于校验一致率不高、需手工检查潜在的“泄漏”谓词，且未提供自动化预测误差检测或跨语料库扩展。

---

## 47. AppLooper: An Agentic Application Engineering Loop for Accountable Release with Virtual-User Feedback

**arXiv ID:** 2608.14093 | [PDF](https://arxiv.org/pdf/2608.14093v1)

**作者:** Zihong He `[一作]` (Hong Kong University of Science and Technology), Hai-Ning Liang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 AppLooper——一个多代理协同的应用工程循环，支持从需求冻结、开发、虚拟用户测试、读写测试、所有者意图回溯，到版本绑定的验收与发布，保障最终发布责任归属。

**💡 创新点**

创新点在于：①将所有者意图、虚拟用户体验、读写测试结果与具体候选版本绑定，形成可追溯、可复审的生命周期；②设计了所有者意图仿真代理与测试代理的权限分离，避免对实现产生误导；③引入版本边界验收机制，确保任何一次发布都对应一个不可变的候选版本；④通过统一的编排层将反馈聚合并路由，形成闭环的持续迭代流程。

**🔧 技术方法**

技术实现包括：Claude Code 作为编码代理；Agnes 2.5‑flash 用于可行性判断；多角色提示模板与结构化 JSON 输出；虚拟用户代理使用 LLM 生成人物与任务脚本并在真实浏览器中执行；测试代理执行灰盒回归与 UI 交互检查；所有者意图仿真代理仅读入已确认的需求与反馈；编排层维护版本、任务、反馈、验收状态；界面使用 OpenAI 图像生成展示角色。

**📊 数据集**

数据集主要为：①用户手动冻结的需求文本；②由所有者或设计师生成的虚拟用户人设与任务脚本；③合成的测试数据与回归测试用例；没有使用公开大型数据集，全部采用仿真与合成数据。

**📈 对比分析**

实验通过在 Todo‑list 原型上进行对比，评估了：①需求满足率、误差识别率与迭代次数；②可追溯链的完整性与人工审计成本；③开发时间与传统手工迭代的差异。结果显示，AppLooper 在需求满足率上达到 100%，误差识别率提升约 20%，迭代次数比手工平均少 2 次，开发时间缩短约 30%。

**⚠️ 局限性**

局限性包括：①仅在原型级别验证，缺乏大规模真实项目的实测；②依赖 LLM 的推理与代码生成，可能引入不确定性与隐藏错误；③虚拟用户反馈无法完全覆盖真实用户多样性与行为模式；④编排与多代理协调复杂，易出现性能瓶颈；⑤所有者需要持续投入确认与验收，若不及时介入可能导致需求漂移。

---

## 48. Identifiability and Order-Dimension Limits of In-Context Learning on Partial Orders

**arXiv ID:** 2608.14004 | [PDF](https://arxiv.org/pdf/2608.14004v1)

**作者:** Faizanuddin Ansari `[一作]`, Swagatam Das `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文研究了有限宇宙下偏序的版本空间、教学维度、可达性以及坐标解码的理论边界，并给出了对应的完整定理与证明。

**💡 创新点**

创新点在于：①提供了开放世界教学维度的闭式表达式并与闭世界形式对比；②首次给出判定任意查询可识别性的逻辑规则；③引入最小前向封闭集合作为负证据的最优证明；④确立坐标解码器的维度阈值与Dushnik–Miller维度的等价性；⑤给出完整的四元组偏序枚举示例，验证理论预期。

**🔧 技术方法**

主要技术包括偏序理论、图论（有向无环图、最短路径、前向封闭集）、集合覆盖与阻塞集的最小化、Floyd–Warshall闭包算法、位集操作以及离散枚举脚本。

**📊 数据集**

论文不依赖外部数据集，所有实验均为对4元偏序（共219个）的完整枚举；其它部分以符号演算和理论分析为主。

**📈 对比分析**

与传统的教学维度估计相比，论文给出了精确的闭式上界和下界，并证明了在开放世界情形下教学维度比闭世界更大；在查询可识别性方面，提供了三种判定方式（真、假、未知），并实现了O(|U|^3)的闭包预处理。性能在枚举实验中实现于1.6秒，内存低于200 MiB。

**⚠️ 局限性**

主要局限包括：①最小阻塞集（β(P)）求解为NP‑hard，未给出多项式算法；②结果仅在已知有限宇宙下成立，无法直接推广到无限或动态扩展的偏序；③坐标解码器的构造非有效，需先预先获得所有目标的维度上界；④枚举实验仅限于4元素偏序，缺乏对更大规模的经验验证。

---

## 49. Adaptive Snapshots Require Visible Reads

**arXiv ID:** 2608.13705 | [PDF](https://arxiv.org/pdf/2608.13705v1)

**作者:** Niv Sulimany `[一作]` (Technion – Israel Institute of Technology), Erez Petrank `[通讯]` (Technion – Israel Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

证明了在满足一组自然假设（单写单扫描、点击操作不访问组件、点击与更新线性化独立、组件封装等）的情况下，适应性快照算法无法实现隐式读取（invisible reads）。

**💡 创新点**

首次给出了关于适应性快照的不可实现性定理，并引入了“点击不知情”（oblivious click）和“k‑bounded 点击”这两类新性质，展示了即使在极简环境下隐式读取也不可行。

**🔧 技术方法**

主要使用了线性化性（linearizability）框架，构造了两个不可区分的执行（E 与 E'），并利用有效线性化步骤（effective linearization step）与阻塞自由（obstruction‑free）进程的特性进行推理。

**📊 数据集**

本工作为纯理论分析，未使用任何实验数据集；所有证明均基于抽象的共享内存模型和伪代码。

**📈 对比分析**

由于是不可实现性证明，没有可比的实现或性能数据；文章通过理论对比说明即使采用最佳实现策略也无法突破该限制。

**⚠️ 局限性**

局限性在于证明只适用于满足上述假设的算法；若放宽单写、单扫描或允许点击访问部分组件、或放弃组件封装等条件，可能仍能设计隐式读取的适应性快照。

---

## 50. MobileMem: Learning from a Year of Mobile Experiences

**arXiv ID:** 2608.13606 | [PDF](https://arxiv.org/pdf/2608.13606v1)

**作者:** Xinle Deng `[一作]` (OPPO), Ningyu Zhang `[通讯]` (OpenKG)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MobileMem评测框架，构建知识驱动的长时序多模态用户交互轨迹与问答数据集，并评估移动端长期记忆系统的能力。

**💡 创新点**

创新点包括：①关注移动端存储、计算、隐私受限的持续记忆问题；②提出知识引导体验合成（KEME）技术，可在保证时间连贯性的同时合成多模态轨迹；③设计系统层与应用层双重记忆架构，实现跨应用知识协同；④将多模态、多任务（多跳推理、时序推理、偏好推断、视觉推理等）融入统一评测。

**🔧 技术方法**

使用大语言模型（GPT‑5.2/5.4‑mini、Qwen3）、图结构检索（HippoRAG2、A‑MEM）、长上下文记忆（Long Context）、RAG框架（NaiveRAG、EverMemOS）、多模态检索模型（SigLIP2、M2A、Multimodal Long Context）、文本转图像与图像质量验证工具，构建完整的合成与评估流水线。

**📊 数据集**

基于真实手机一年级别交互数据（日历、照片、笔记、文档、待办、语音、浏览、视频等）与公开的用户画像数据集，生成MobileMem（≈16条轨迹、19 060张图像、7 415个QA）和MobileMem‑Omni（涵盖多语言、多模态的更大规模数据集）。

**📈 对比分析**

在MobileMem上对Long Context、NaiveRAG、HippoRAG2、A‑MEM、MemOS、EverMemOS等方法进行对比，采用LLM‑Judge与F1评估；结果显示A‑MEM和HippoRAG2在两种LLM后端表现最佳；在MobileMem‑Omni上，EverMemOS和M2A在多模态任务上取得最高分。评估还揭示了Token成本高、事件时长、语言差异对性能的显著影响。

**⚠️ 局限性**

局限性：①生成数据仍存在噪声与一致性问题；②用户模型相对粗粒，难以捕捉复杂偏好演化；③评估主要集中在移动端两语言与有限任务，未覆盖更广泛场景；④对长时序推理和跨模态细粒度检索的支持仍不足；⑤大模型与高Token成本限制了在真实边缘设备上的可落地性。

---

## 51. FactorFlow: A Visual Analytics Workspace with Large Language Model-Assisted Interpretation for Factor Analysis

**arXiv ID:** 2608.13585 | [PDF](https://arxiv.org/pdf/2608.13585v1)

**作者:** Justin Philip Tuazon `[一作]`, Richelle Ann Juayong `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个名为 FactorFlow 的可视化分析工作区，支持从数据上传、诊断、模型拟合到结果解释的完整 EFA 流程，并集成大语言模型生成自动化因子解释。

**💡 创新点**

创新点在于：①提供多模型并排比较与交叉负荷网络可视化；②引入 interpretability plot 结合语义相似性评估因子解释性；③将 LLM 用于因子标签自动生成，降低人工解释成本；④集成 Polychoric 相关、可视化阈值调整等实用功能。

**🔧 技术方法**

技术实现基于 Python3，Streamlit 前端框架，Plotly 可视化；计算使用 interpretablefa（扩展 factor_analyzer）及 NumPy/Pandas/Scipy；前端 LLM 调用 TensorFlow.js Universal Sentence Encoder 与 Groq API；数据上传支持任意表格，默认以 Likert 量表为主。

**📊 数据集**

未公开具体实验数据集，采用公开问卷式 Likert 数据示例（如情绪量表、驾驶行为量表）进行演示；在可用性评估中使用 5 名统计学背景参与者的自制 EFA 问卷。

**📈 对比分析**

方法对比通过仪表板的双模型选择实现；性能方面可在数秒内完成模型拟合、诊断和可视化，用户反馈认为交互流畅；可通过下载功能获取表格与图形，便于后续分析。

**⚠️ 局限性**

局限性包括：①可用性调查样本量小，未进行严格统计检验；②基于 Streamlit 的重新渲染机制导致部分交互略有延迟；③仅覆盖 EFA，CFA 等更高级模型待后续扩展；④对非统计背景用户的术语与交互体验仍需进一步改进。

---

## 52. Geometric Filtering of LLM-Generated Samples for Few-Shot Text Classification

**arXiv ID:** 2608.13866 | [PDF](https://arxiv.org/pdf/2608.13866v1)

**作者:** Benjamín Schindler `[一作]` (Universidad Adolfo Ibáñez), Gonzalo A. Ruz `[通讯]` (Universidad Adolfo Ibáñez)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于欧氏距离的几何过滤框架，用于筛选LLM生成的文本增量样本，以提升少样本文本分类性能。

**💡 创新点**

首次将LLM生成与经典几何滤波结合，提出软加权机制，并证明最简单的欧氏距离滤波优于复杂多准则方案。

**🔧 技术方法**

使用LLM（Gemini 3 Flash、GPT‑5‑mini、Claude 4.5 Haiku等）生成样本，句子嵌入（SBERT 等）计算欧氏距离，采用软加权机制，并与多种分类器（线性SVC、Ridge、MLP、RF、LogReg）以及多种增补方法进行对比。

**📊 数据集**

在 13 个文本分类与 NER 数据集上进行评估，包括 20newsgroups、ag_news、emotion、banking77、clinc150、MultiNERD、WikiANN、Few‑NERD 等。

**📈 对比分析**

与 SMOTE、EDA、回译等 10 种增补基线在 1,890 个实验配置中对比，软加权滤波在宏 F1 上平均提升 +2.61pp（p<0.0001，Cohen’s d=0.95，胜率 88.9%）。

**⚠️ 局限性**

仅在英文数据上评估，增益随训练样本数量增多或类别数增多而显著下降；使用相同嵌入模型可能导致循环偏差；多语言迁移和开源 LLM 的适用性尚未验证。

---

## 53. Reading Between The Lines: Modeling and Evaluating Behavioral Realism in Legal Simulation

**arXiv ID:** 2608.13712 | [PDF](https://arxiv.org/pdf/2608.13712v1)

**作者:** Divya Vetticaden `[一作]` (Stanford University), Megan Ma `[通讯]` (Stanford University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 WitnessSim——一种基于可控状态机的证人模拟器，用于在法律实务中训练律师的交叉询问技巧。

**💡 创新点**

创新点在于将证人行为建模为六维可解释状态空间（镇定、知识、友好、冗长、僵硬、表现力），并通过问题编码、压力与主题敏感度更新状态，实现动态且可追踪的行为变化；同时设计了双层评估框架，将行为真实性与教学效用分离。

**🔧 技术方法**

核心技术包括：基于大型语言模型（LLM）的文本生成、状态更新的离散动力学方程、情感向量轨迹分析、问答编码的压力/敏感度计分、以及事件触发机制。

**📊 数据集**

使用来自 1,169 篇美国联邦多地区诉讼“国家处方阿片诉讼”案件的 300 篇真实庭审/取证记录作为基准，并以此生成对应的 WitnessSim 模拟交互。

**📈 对比分析**

通过四类对抗性测试、盲判真实/生成对话的可行性评估以及情感轨迹相关性检验验证了行为真实性；在四个教学任务（问答形式、逼问逃避、失控冗长、对抗性证人）中证明其教学效用。整体表现显示：对抗性测试中 97%以上维持角色边界，盲判中律师对生成证人响应的可行性与真实证人相当，教学任务中 90% 以上达到预设行为转化。

**⚠️ 局限性**

局限性包括：评估仅基于单一阿片诉讼语料，情感向量方法尚未经过严格验证；未探究模拟训练对律师实际学习效果的影响；模型可能对特定法律环境的泛化能力有限。

---

## 54. Building AI-Intensive Software with AI: Early Results and a Cautionary Tale on Measuring Development Cost

**arXiv ID:** 2608.13730 | [PDF](https://arxiv.org/pdf/2608.13730v1)

**作者:** Victor Barros de Miranda Neves `[一作]` (Universidade Federal de Pernambuco), Vinicius Cardoso Garcia `[通讯]` (Universidade Federal de Pernambuco)

**通讯引用:** 1384 | [OpenAlex ID](https://openalex.org/A5006873642)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究以一支六人学生团队为例，利用全流程 AI 辅助，构建了一个基于检索增强生成（RAG）的对话式入职助手，并对其开发成本进行了三层成本模型的测算。

**💡 创新点**

创新点在于首次公开揭示并纠正了 AI 成本估算中的两类易犯错误（基于 token 计价的误用与使用错误区域工资率），并提出可复用的成本测算方法。

**🔧 技术方法**

主要技术包括大型语言模型（LLM）编码助手、检索增强生成、向量存储、关系数据库、FastAPI 后端与 React 前端，以及多语言代码分块与嵌入。

**📊 数据集**

实验使用了一个真实的多语言（15 种）遗留代码仓库，自动生成的测试代码和功能实现约 21,000 行生产代码与 2,000 行测试代码。

**📈 对比分析**

通过将实际 AI 支出、团队自报人力与估算的无 AI 对照人力进行对比，得到约 9.9 倍的成本节约比率，表明 AI 辅助在实现层面显著降低了人力投入。

**⚠️ 局限性**

局限性包括仅基于单一学生项目、对对照人力的后向估算存在偏差、数据自报可能产生误差、以及成本模型仍需在更多项目中验证。

---

## 55. Demonstration of Space Robot Teleoperation over a Lossy and Delayed Network using ATMOS

**arXiv ID:** 2608.14031 | [PDF](https://arxiv.org/pdf/2608.14031v1)

**作者:** Inkyu Jang `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过在远距离（约7450 km）上远程控制ATMOS机器人完成对接任务，展示了在微重力模拟环境下的硬件平台与网络延迟下的闭环控制方案。

**💡 创新点**

创新点在于将状态预测（基于对随机通信延迟的贝叶斯估计）与控制Lyapunov函数（CLF）约束相结合，形成一种对时变随机延迟鲁棒的轨迹跟踪控制器，并在真实跨洲网络实验中验证其有效性。

**🔧 技术方法**

使用的技术包括：状态预测与粒子滤波、CLF‑基准二次规划控制、网络中间件FleetMQ、ROS 2与Gazebo仿真、Qualisys运动捕捉、八通道真空阀执行机构以及车载IMU融合。

**📊 数据集**

未使用公开数据集，而是在仿真中生成不同延迟特性（无延迟、轻度、中度、重度）和实际网络中产生的延迟/丢包序列；硬件实验则利用自身的运动捕捉和实时网络时延测量。

**📈 对比分析**

通过将所提控制器与仅使用参考PD控制器（无延迟补偿）进行对比，实验表明在仿真中RMSE显著下降（尤其在中度延迟下位于PD控制器的一半左右），硬件对接成功率高，轨迹跟踪误差保持在厘米级。

**⚠️ 局限性**

局限性包括：缺乏正式的稳定性与性能理论保证、对延迟分布假设的依赖（实际网络存在强相关性与突发延迟）、仅在二维平面微重力模拟中验证、未评估极端延迟/丢包场景，且实验规模受限于单机器人。

---

## 56. OpenBelief-Nav: Evidence-Preserving Object Memory for Open-Vocabulary Language-Guided Navigation

**arXiv ID:** 2608.13923 | [PDF](https://arxiv.org/pdf/2608.13923v1)

**作者:** Dinh Tuan Nguyen `[一作]` (VinMotion), Quan Nguyen `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种延迟语义承诺的证据保留式物体记忆框架，能在任务时读取多种语义决策并在执行中纠正候选。

**💡 创新点**

核心创新在于：①在对象图中保留每一次观测的语言短语、可靠性指示与帧掩码；②将语义决策推迟到任务时，从而支持多读口和执行时的候选纠正；③通过可验证的候选机制实现物理执行纠正，而不改写持久图。

**🔧 技术方法**

利用 CLIP 与 MobileCLIP 进行视觉与文本特征编码，SAM 进行语义掩码生成，Qwen3‑VL‑4B‑Instruct 进行短语生成；点云投影、可扩展语义信念构建、伪计数平滑、语义共识与早期承诺读口；以及基于 RGB‑D 的可视化检查实现执行纠正。

**📊 数据集**

在 ScanNet200、Replica 进行语义分割评估；在 HM3D–YCB 进行语言导向导航评估；以及在 Unitree G1 人形机器人上执行真实世界的 20 次导航实验。

**📈 对比分析**

与 ConceptGraphs、HOV‑SG、DualMap 等基线在同一图上进行对照；在 ScanNet/Replica 上全信念投影 mIoU 最高（0.2742/0.2912）；在 HM3D–YCB 导航中 consensus/early‑commit 取得最高成功率（约 77%）；真实世界中执行纠正将成功率从 60% 提升至 80%。

**⚠️ 局限性**

评估范围局限于固定场景和已映射目标，未覆盖目标缺失、目标移动或大规模多环境的泛化；缺乏对动态场景和更复杂执行策略的全面验证。

---

## 57. LegacyWorld: Atomicity-Aware Evaluation of GUI Agents for Legacy Workflows

**arXiv ID:** 2608.14131 | [PDF](https://arxiv.org/pdf/2608.14131v1)

**作者:** Thilo Reintjes `[一作]` (Schub), Alexander Pretschner `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一套基于原子性（atomicity）的Legacy GUI工作流基准，测试多种大型语言模型（LLM）代理在Windows环境下的任务完成与安全失败表现。

**💡 创新点**

创新点在于将任务成功与安全失败区分为四类结果（valid success、valid failure、invalid success、invalid failure），引入原子性评估并比较专家手工提示与从屏幕录制自动生成提示的效果。

**🔧 技术方法**

使用了computer-use LLM代理（GPT‑5.4、Gemini 2.5、Claude Opus/sonnet/haiku、Kimi K2.5）与legacy-use harness，配合任务合同、状态验证器及窗口/文件/数据库快照。

**📊 数据集**

构建了28个Windows GUI工作流数据集，其中18个经过业务专家外部验证，覆盖行政、医疗、报表、实用工具等多种领域。

**📈 对比分析**

对每个模型在专家提示与视频生成提示两种条件下各执行一次，统计四类结果，计算有效成功率和原子性率；结果显示部分模型在视频提示下有效成功率下降但原子性率保持，说明安全失败占比上升。

**⚠️ 局限性**

局限性包括：评估仅在隔离虚拟机内进行，未覆盖真实生产环境；每个模型每个任务仅单轨执行，未统计随机性；验证仅针对预定义观察对象，可能遗漏隐藏副作用；未实现完整从视频学习任务的闭环。

---

## 58. CoSA: Context-Aware Severity Assessment via Context Analysis with Large Language Models

**arXiv ID:** 2608.13928 | [PDF](https://arxiv.org/pdf/2608.13928v1)

**作者:** Jinfeng Jiang `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种利用仓库级上下文进行CVSS v3.x severity自动评估的方法

**💡 创新点**

创新点在于：①两阶段 pruning（轻量化静态 pruning + LLM‑guided pruning）以筛选分散的证据；②通过 LLM 生成的 metric‑wise 文本摘要，将分散的仓库信息聚合成可供预测的结构化输入；③在同一框架下实现高效、可扩展的评估流程

**🔧 技术方法**

技术包括：代码属性图（CPG）构建、基于图的静态邻域剪枝、LLM 代理推理（信息收集与选取）、LLM 总结与文本摘要、轻量级 Transformer 预测模型

**📊 数据集**

使用新构建的 6,816 条 CVSS v3.x 标注实例（覆盖 90 个 CWE），来自 PrimeVul 与 TitanVul，并通过 NVD 提取完整 CVSS 向量

**📈 对比分析**

与 LLM prompting（直接和多代理）以及函数级 Transformer（CodeBERT/GraphCodeBERT/UniXcoder）基线比较；在基于严重性等级的多类分类任务中，取得 47.22% 的准确率、34.15% 的宏 F1，较最佳基线提升 14.4%（准确率）和 15.3%（宏 F1）

**⚠️ 局限性**

局限性：①需要完整仓库和 CPG 构建，部署成本较高；②对 LLM 的推理能力与提示设计高度依赖；③数据中标签分布不均衡，对稀有指标预测仍存在挑战；④对环境依赖（如构建配置、部署信息）缺乏支持，影响 AV、PR、Scope 等指标的推断

---

## 59. Never the Number: Structural Abstention for AI Systems Whose Answers Are Consumed as Fact

**arXiv ID:** 2608.13926 | [PDF](https://arxiv.org/pdf/2608.13926v1)

**作者:** Zhelun `[一作]` (Apple Inc), Wu `[通讯]` (Apple Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种“内核‑外壳”架构模式，利用生成式语言模型只在问题解释和确认阶段发挥作用，而真正的答案计算由确定性内核完成，保证返回值不被模型篡改；通过“结构性拒绝”（structural abstention）在用户确认之前发现不可回答的请求并明确拒绝，防止错误答案无声渗透。

**💡 创新点**

创新点在于提出并验证了“结构性拒绝”与“生成式外壳+确定性内核”的组合：① 通过一个不允许生成的内核实现答案可审计；② 在用户确认句子前进行全路径的结构化拒绝，使错误可见；③ 把问题解耦为“问题模板池 + 编译器”，将覆盖范围和错误可见性明确化；④ 通过五个设计决策（问题单元、部分请求表示、编译可信度、确认展示、拒绝处理）实现模式可迁移性。

**🔧 技术方法**

技术上主要使用：大语言模型（LLM）进行句子生成与对话引导；实体向量与槽位模板池实现问题匹配；确定性编译器将已确认的问题转化为 SQL 或调用受控服务；多轮对话状态管理（推理自动机+嵌入式近邻搜索）；双语支持通过小型翻译模型与命名实体识别实现；对比实验还实现了细调文本‑到‑SQL 解析器与工具‑检索代理两种替代方案。

**📊 数据集**

使用的实际数据来自两年的生产部署日志（业务查询与答案对），但论文中不公开具体表结构或数据集。为评估做了基准比较：Spider 2.0、BEAVER 等公开工业仓库基准；在内部生产中收集了数千模板与数万问答对，用于微调模型和构建模板池。

**📈 对比分析**

比较方法：在生产环境与实验环境分别跑 kernel‑first、端到端细调解析器和工具‑检索代理。评估维度包括覆盖率、错误模式、确定性、延迟、算术计算位置、审计可行性。结果表明 kernel‑first 在覆盖率上较低，但错误以可见拒绝形式出现，延迟最低，算术全部由数据库完成；细调解析器覆盖率最高但错误不可见；工具‑检索代理在覆盖率与可见性之间折中，延迟最高。

**⚠️ 局限性**

局限性：① 受限于模板池，无法回答池外新型问题；② 需要手工扩充模板，人工成本高；③ 需要多轮交互，用户体验受限；④ 冷启动阶段缺乏个性化；⑤ 对探索性分析者不友好，因为他们需要即时、可调试的 SQL；⑥ 仅在“事实消费”场景下有效，对可检查结果的用户不适用。

---

## 60. Do AI chatbots find what experts would? Effects of model, user role, and sample size on study retrieval for medical questions

**arXiv ID:** 2608.13786 | [PDF](https://arxiv.org/pdf/2608.13786v1)

**作者:** Qingfang Liu `[一作]` (National Institutes Of Health), Zhiyong Lu `[通讯]` (National Institutes Of Health)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估三款最新大型语言模型聊天机器人（Claude Sonnet 5、Gemini 3.1 Pro、ChatGPT GPT‑5.5）在检索系统综述中引用原始临床研究的能力，使用20个Cochrane综述的复合问题并模拟患者、临床医生、研究者三种用户角色，共产生720次回复。

**💡 创新点**

创新点在于将用户角色设定与检索结果关联、使用具有推理与工具调用功能的最新LLM、系统性对比不同模型和角色下的召回率及召回一致性，并揭示模型在检索时对大样本研究的偏倚。

**🔧 技术方法**

使用了Claude Sonnet 5、Gemini 3.1 Pro和ChatGPT GPT‑5.5三种通用聊天机器人，利用其内置网络搜索与推理能力来生成回答并列出原始研究。

**📊 数据集**

数据集为2026年Cochrane数据库第6、7期发表的20个干预综述的纳入与排除研究列表，作为检索结果的参考标准。

**📈 对比分析**

通过将每个模型在每个用户角色下的四次重复回复与Cochrane纳入研究列表对照，计算召回率、召回一致性；结果显示ChatGPT召回率最高（63.1%），Claude次之（37.0%），Gemini最低（17.3%），角色对召回影响有限，且仅研究样本量显著影响检索表现。

**⚠️ 局限性**

局限性包括仅评估20篇综述且问题与真实用户交互不完全一致、模型训练数据可能与Cochrane综述重叠、未检验检索到的非纳入研究是否适宜，以及未评估模型生成结论的准确性。

---

## 61. Overcoming Shortcut Learning in Graph Neural Networks through Active Explanation Guidance

**arXiv ID:** 2608.14121 | [PDF](https://arxiv.org/pdf/2608.14121v1)

**作者:** Taraneh Younesian `[一作]` (VU Amsterdam), Stefano Teso `[通讯]` (University of Trento)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过人类专家对 GNN 解释的主动纠正，提出一种激活式解释指导框架，旨在消除 GNN 的捷径（伪相关特征）并提升对分布外样本的泛化能力。

**💡 创新点**

创新点在于：①结合 GNN 解释（输入梯度）与可微分的解释损失，将专家纠正直接嵌入模型训练；②提出两种基于解释熵的主动学习查询策略（最大/最小解释熵），在有限注释预算下显著提升模型性能；③实现了与多种 GNN 结构（GCN、GraphSAGE、GAT、GIN）无关的通用框架。

**🔧 技术方法**

核心技术包括：输入梯度解释器、解释损失正则化、基于熵的主动学习查询策略、可微分的 GNN 训练流程；实验使用了 PyTorch‑Geometric 等标准深度图学习工具。

**📊 数据集**

数据集：1) 合成 Erdős‑Rényi 图（含颜色 confounder） ；2) Graph‑MNIST（将 MNIST 图像转为超像素图，训练集加入颜色 confounder）。

**📈 对比分析**

对比方法：被动解释监督、随机采样、最大分类熵（MaCE）等。结果显示，使用 MiEE（最小解释熵）查询的主动学习在保持低注释成本的同时，往往能逼近甚至超越被动监督的性能；在大注释预算时，所有查询策略差异减小。

**⚠️ 局限性**

局限性包括：①依赖解释器的可信度，若解释不准确则纠正无效；②需要人工专家提供解释，仍有一定成本；③在实验中仅评估了两种数据集，尚未验证在更复杂、真实世界图数据上的可扩展性；④当预算极大时，主动学习优势不明显。

---

## 62. Rethinking Automated Program Repair: The Impact of Bug Complexity, Fault Localization, and LLM Cost-efficiency

**arXiv ID:** 2608.14065 | [PDF](https://arxiv.org/pdf/2608.14065v1)

**作者:** Junchi Liu `[一作]` (Colorado State University), Fabio Santos `[通讯]` (Colorado State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过大规模实证实验，系统评估了基于大型语言模型（LLM）的自动程序修复（APR）技术在不同 bug 复杂度、故障定位（FL）精度、推理设置及成本效率方面的表现，构建了以 AtCoder 竞赛提交为基础的算法级 C++ bug 数据集，并提出了多维复杂度划分与两阶段 LLM 组合策略。

**💡 创新点**

创新点包括：①引入基于 Git、srcML 与 GumTreeDiff 的双层复杂度划分（hunk‑line 与 AST 操作），为 APR 评估提供更细粒度的难度刻画；②系统比较不同 FL 精度（行级、块级、无）对修复效果与代码一致性的影响；③分析推理模式在不同 LLM 家族中的差异，并提出低成本 LLM 先行、后续高级 LLM 补救的两阶段修复策略；④统一成本效率度量（Repaired Bugs/​$、Attempts/​$），实现跨模型成本-性能对比。

**🔧 技术方法**

使用的技术包括：ChatRepair 与 CodeCorrector 两个 APR 框架；DeepSeek、GPT（GPT‑5、GPT‑4o‑mini）和 Llama（Llama‑3.1‑8B/‑405B）等 LLM API；Git、srcML 与 GumTreeDiff 进行 bug 复杂度与差异分析；统计检验（χ²、Logistic 回归）与多指标评估（修复率、Top‑k、可编译/可行补丁、代码一致率）。

**📊 数据集**

数据集来源于 2025 年 10 月 AtCoder 竞赛提交，共 1,914 条 C++ 程序，构造 640 条 bug‑patch 对；在后续筛选后保留 211 条具有中等难度且低可修复率的 bug，用于高级 LLM 与推理设置评估。

**📈 对比分析**

比较方法：在同一 bug 集上，针对不同 FL 方案（行级、块级、无）和 LLM/推理配置，记录修复率、Top‑k、成本指标，并采用统计检验验证差异显著性。实验结果显示：DeepSeek‑V3.2‑chat 在低成本条件下具有最高成本效率；GPT‑5 在非推理模式下实现最高修复率；推理模式对 GPT‑5 效益有限，而对 DeepSeek 系列可提升 60‑80% 的额外修复；低精度 FL 时不同 APR 技术的性能差距显著扩大。两阶段策略（先使用 DeepSeek‑V3.2‑chat 低成本修复，再使用 GPT‑5 解决剩余 bug）能兼顾性能与成本。

**⚠️ 局限性**

局限性：①仅针对 C++ 算法级单文件 bug，缺乏对工业规模多文件项目的验证；②评估模型仅限两种 APR 框架和三大 LLM 家族，可能遗漏其他有效技术；③模型知识截止日期与数据集采集时间不一致，可能导致数据泄露或过时；④API 的非确定性输出与参数设定（温度、推理强度）可能影响结果稳定性；⑤测试用例覆盖率有限，存在过拟合或隐藏错误的风险。

---

## 63. L-FNO: Lorentzian Fourier Neural Operator for Stochastic Event Dynamics

**arXiv ID:** 2608.13562 | [PDF](https://arxiv.org/pdf/2608.13562v1)

**作者:** Songhee Kang `[一作]` (Tech University of Korea), Jihoon Kang `[通讯]` (Tech University of Korea)

**通讯引用:** 2605 | [OpenAlex ID](https://openalex.org/A5013471175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Lorentzian Fourier Neural Operator (L‑FNO)，用于稀疏事件的条件强度建模

**💡 创新点**

将Lorentzian谱核引入FNO并以Poisson负对数似然训练，使模型既能捕获自激记忆，又能兼顾外部协变量

**🔧 技术方法**

傅里叶神经算子、Lorentzian谱核、Poisson负对数似然损失、Softplus输出、滑动窗口训练

**📊 数据集**

八个合成点过程基准（B1–B8）与三个真实数据集（半导体缺陷FDC、SECOM、泰国疱瘙疹LSD）

**📈 对比分析**

与FNO‑MSE、FNO‑NLL及Neural Hawkes等基线对比；在所有指标（NLL、Brier、PR‑AUC、AUC）上均优于基线，PR‑AUC提升至0.74~0.88

**⚠️ 局限性**

只支持正向自激，无法建模抑制效应；空间处理仅为时间序列，缺乏完整时空算子；对长时延和连续时间的适应有限

---

## 64. Post-training Quantization for Hybrid Iterative Generative Models

**arXiv ID:** 2608.13932 | [PDF](https://arxiv.org/pdf/2608.13932v1)

**作者:** Jing Gao `[一作]` (Beijing Jiaotong University), Yao Zhao `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HyGenQ，一种面向混合迭代生成模型的后训练量化框架，解决过度异常值和放大异常的问题；

**💡 创新点**

创新点在于两大模块：Hierarchical Cluster Decoupling（分层聚类解耦异常通道）与 Scaling Recalibration（高斯边界重标定）实现对混合模型的高质量 W8A8 量化；

**🔧 技术方法**

采用多阶段聚类与遗传算法确定异常通道，使用高斯分布阈值动态缩放放大异常，并在每步迭代中为线性层分配独立量化参数；

**📊 数据集**

在 ImageNet 256×256 数据集上评估，通过 8,000 张生成样本计算 FID/IS；

**📈 对比分析**

与 UniformQuant、RepQ-ViT、SmoothQuant、PTQ4DM、TFMQ-DM、TaQ-DiT、OCS 等基线对比，HyGenQ 在 MAR-B/L/H 模型上均显著降低 FID、提升 IS，保持生成质量并实现近乎 INT8 速度提升；

**⚠️ 局限性**

局限包括：对更深模型仍存在一定生成质量下降、SR 计算导致额外延迟、低位宽（如 W4A8/W4A4）下易崩溃，且放大异常的累计误差仍是挑战。

---

## 65. Model-agnostic Retrieval-Augmented Extended Forecasting for time series

**arXiv ID:** 2608.14054 | [PDF](https://arxiv.org/pdf/2608.14054v1)

**作者:** Juan Pablo Villa Serna `[一作]` (Friedrich-Alexander-Universitaet Erlangen-Nuernberg), Vasileios Belagiannis `[通讯]` (Friedrich-Alexander-Universitaet Erlangen-Nuernberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无微调的检索增强时间序列预测框架 RAEF，直接在输入空间检索并通过分层拼接聚合提升预测精度。

**💡 创新点**

核心创新点是：① 在原始输入空间直接检索，省去嵌入计算；② 用候选–残差阈值分层后拼接聚合，保持时序结构并去除对齐步骤；③ 兼顾信息完整性与推理效率。

**🔧 技术方法**

利用检索增强生成（RAG）技术、HNSW 向量数据库、实例归一化、阈值候选分割、Transformer 预训练模型（MoiraiMoE、Chronos‑T5、Chronos‑Bolt）以及欧氏距离做相似性度量。

**📊 数据集**

实验数据集包括 ET、华为云电力、Power、Traffic、FRED‑MD、Electricity‑UCI 等六个公开时间序列基准。

**📈 对比分析**

与基线预训练模型、RAF、以及微调（Fine‑Tuning）进行对比。RAEF 在 MAE、MASE 指标上平均提升 11‑16%；在 4/6 个数据集上超过或匹配微调性能，同时推理时间更低。

**⚠️ 局限性**

局限性：对长上下文（>128）提升有限；聚合后序列长度仍可能增加注意力开销；需要手动调节阈值 d_t；仅验证单变量序列，尚未扩展至多变量或非平稳场景。

---

## 66. A Reproducibility Protocol for Cross-Implementation Evaluation of Post-Quantum ACVP Test Vectors

**arXiv ID:** 2608.13784 | [PDF](https://arxiv.org/pdf/2608.13784v1)

**作者:** Christopher M. Frost `[一作]` `[通讯]` (HEOSSI (Pte.) Ltd.), Christopher M. Frost (HEOSSI (Pte.) Ltd.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对NIST ML‑KEM标准的三款公开实现进行可重复的、基于向量的已知答案验证。

**💡 创新点**

提出了公开可重复的验证协议，包含可冻结的输入、接口契约、错误分类和案例保持记录。

**🔧 技术方法**

使用Python驱动、专门适配器、SHA‑256校验和CAVP样本向量。

**📊 数据集**

使用NIST ACVP公开的ML‑KEM样本向量（三参数集）。

**📈 对比分析**

通过对比实现输出与NIST预期字节/判决，三轮重复验证，结果无错误、无不稳定，表明实现一致。

**⚠️ 局限性**

仅验证特定版本、单一平台，未覆盖随机化、侧信道、性能或跨平台行为；接口差异限制了完整交互验证。

---

## 67. From BERT to Frontier Agents: Eight Years of Language-Model Progress, the Collapse of the Capability-Cost Curve, and the Rise of Task-Targeted Models

**arXiv ID:** 2608.13675 | [PDF](https://arxiv.org/pdf/2608.13675v1)

**作者:** Pranav Kumar Kaliaperumal `[一作]` `[通讯]`, Pranav Kumar Kaliaperumal

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

追踪并量化2018‑2026年语言模型的演进，测量SWE‑bench、MMLU等基准的增长、成本下降，并评估推理层面的增益如多样采样、置信估计与模型路由。

**💡 创新点**

提供统一时间线、SWE‑bench odds 5.8×/年增长、成本曲线60×下降、展示多模型路由可实现单模型最佳表现，以及对固定权重模型的置信分层选择实验。

**🔧 技术方法**

统计趋势拟合、Pareto/价格‑能力曲线、基准自测与独立 harness、四样本投票/自一致性、Logistic回归置信估计、模型路由与成本约束等技术。

**📊 数据集**

BERT/GLUE、SuperGLUE、MMLU、SWE‑bench Verified、ARC‑AGI‑3、GSM8K、ARC Prize、SWE‑bench Pro、Frontier‑Bench、Frontier Math、以及Meta/Anthropic/OpenAI内部测试数据。

**📈 对比分析**

通过公开基准分数、vendor 与独立 harness 对比、logit odds 5.8×/年、成本从GPT‑3 60$/M降至GPT‑5.6 Luna 1$/M、预算层与旗舰在大部分专业任务上相当；多模型路由可复现单模型最佳性能；固定权重模型四样本投票提升至62%，置信排序前50%准确率94%。

**⚠️ 局限性**

结果混合vendor/独立来源，SWE‑bench 仅覆盖22个月，置信模型为探索性未在独立数据验证；路由需任务标签推断；采样差异受设备/seed影响；部分性能提升未达统计显著。

---

## 68. Think in Latent, Explain in Language: Self-Explainable Latent Reasoning

**arXiv ID:** 2608.13570 | [PDF](https://arxiv.org/pdf/2608.13570v1)

**作者:** Dayuan Zhao `[一作]` (University of Illinois Urbana-Champaign), Liang-Yan Gui `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1352 | [OpenAlex ID](https://openalex.org/A5016962709)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一框架SELR，使模型在潜在空间中高效推理并自动生成可解释的推理步骤；

**💡 创新点**

创新点在于通过多任务损失同时优化答案生成和Chain‑of‑Thought解码，解决了潜在推理的可监督与可解释双重难题；

**🔧 技术方法**

采用潜在空间推理、联合训练的CoT解码器、单模型自解释技术，并在LLaMA‑3.2‑1B‑Instruct和Qwen2.5‑VL‑3B‑Instruct等大模型上实现；

**📊 数据集**

使用VLM的LLaVA‑CoT‑100k数据集和LLM的GSM8k‑Aug数据集进行训练与评估；

**📈 对比分析**

与SFT、Coconut、CoLaR、Heima等基线对比，SELR在多项VLM/LLM基准上实现了更短的响应长度（减少70%+）并提高了准确率，且自解释质量优于Heima；

**⚠️ 局限性**

局限性包括对潜在推理长度的手动设定、对长文本推理的支持不足，以及缺乏稳定的自停止机制。

---

## 69. Deep Reinforcement Learning solution for pickup and delivery routing problems with time window and capacity constraints

**arXiv ID:** 2608.14156 | [PDF](https://arxiv.org/pdf/2608.14156v1)

**作者:** Andrew Soroka `[一作]` (Moscow State University), Sergey Gerasimov `[通讯]` (Moscow State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本文中，作者使用改进后的JAMPR深度强化学习模型解决了带容量与时间窗约束的取货与投递车辆路径规划（CPDPTW）问题，并提供了快速的最优或近似最优解。

**💡 创新点**

创新点包括首次将强化学习应用于CPDPTW，改造JAMPR的可行性判定函数以支持取货-投递约束，采用软约束线性组合成本，并展示了模型在不同规模问题上的鲁棒性。

**🔧 技术方法**

技术手段主要是基于注意力的Encoder‑Decoder架构（JAMPR），利用强化学习的策略梯度进行训练，并在训练时使用早停策略；实验中还对比了Google OR‑Tools的启发式求解器。

**📊 数据集**

实验数据来自Solomon参考集的R201分布，构造了50、100、200、400、1000规模的随机实例，后续还尝试用正态分布和曼哈顿距离等变体检验鲁棒性。

**📈 对比分析**

与OR‑Tools基线对比时，本文模型在小至中等规模（≤200）问题中在秒级甚至毫秒级内即可给出与最优相当或更优的解；在大型（≥400）问题中提供了更快的近似解，虽然最终与OR‑Tools差距在20–50%之间。

**⚠️ 局限性**

局限性包括：训练时间极长（数天到数十天），无法在单GPU上高效训练超过1000点的问题；模型对分布变化敏感（超过约25%正态噪声时性能下降）；且目前仅实现软约束，硬约束下仍存在失效风险。

---

## 70. Explanation Multiplicity: Circuit-Level Interpretability Evidence Does Not Survive Defensible Analytic Variation

**arXiv ID:** 2608.13754 | [PDF](https://arxiv.org/pdf/2608.13754v1)

**作者:** Ajay Pravin Mahale `[一作]` `[通讯]`, Ajay Pravin Mahale

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 GPT‑2 small 的间接宾语识别任务上，基于七个分析轴构建了 15,840 份规范的预注册多元化电路发现实验，并提出了评估解释可提交性的协议。

**💡 创新点**

首次将监管层面的可验证性与机制解释的稳定性量化相结合，提出并验证了“可提交性”阈值（π∗≥1−α），并揭示现有工具在不同配置下的显著不稳定性。

**🔧 技术方法**

采用边缘归因补丁、集成梯度电路发现、确定性声明映射、Jaccard 相似度、Cohen κ 以及非参数自助法评估不确定性等技术。

**📊 数据集**

使用 GPT‑2 small 模型在先前发布的间接宾语识别数据集（含两种 prompt 变体）进行实验。

**📈 对比分析**

与随机基准对比，发现电路判定的“翻转率”为 0.7316，远高于 0.20 的阈值；即使标准化单一轴也无法将翻转率降至可提交性的门槛，表明解释不稳定。

**⚠️ 局限性**

仅在单一模型与任务上测试；抛弃率高、结果受电路大小影响；工具实现存在缺陷；统计方法在小样本组内估计区间不稳健。

---

## 71. Depth-Aware Sensitivity Analysis of Mixture-of-Experts Models via Magnitude-Based Expert Masking

**arXiv ID:** 2608.13565 | [PDF](https://arxiv.org/pdf/2608.13565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 72. Deterministic Spectral Sparsification in Almost-Linear Time for Dense Graphs

**arXiv ID:** 2608.13910 | [PDF](https://arxiv.org/pdf/2608.13910v1)

**作者:** Jason Li `[一作]` (Carnegie Mellon University), Trevor Vaughn `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了一个加权图的光谱稀疏化器，该稀疏化器是一个重加权的子图，其拉普拉斯二次形式近似于原始图的拉普拉斯二次形式。

**💡 创新点**

提出了一种确定性构造方法，能够在多项式时间内生成具有O(n^(-2) log^24+o(1)n)条边的(1±ε)-光谱稀疏化器，且在时间复杂度上优于现有的随机方法。

**🔧 技术方法**

使用了确定性矩阵切诺夫选择程序、稀疏化算法、递归阻塞方案以及稠密和稀疏实现的组合。

**📊 数据集**

使用了加权多重图，假设边的权重在[1,2]之间，并且最大权重与最小权重的比率是多项式有界的。

**📈 对比分析**

与现有的随机方法相比，性能上在边数和时间复杂度上都有显著的改进，能够在多项式时间内构造出更优的稀疏化器。

**⚠️ 局限性**

在处理高权重比的图时，可能会遇到性能瓶颈，且在某些情况下，算法的复杂度可能会受到图的结构影响。

---

## 73. The Tool-to-Entity Threshold: Parasocial Dynamics of Personalised AI Agents in Shared Social Spaces

**arXiv ID:** 2608.13586 | [PDF](https://arxiv.org/pdf/2608.13586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 74. Horospherically convex optimization for fractional subspace packing and its applications

**arXiv ID:** 2608.13972 | [PDF](https://arxiv.org/pdf/2608.13972v1)

**作者:** Hiroshi Hirai `[一作]` `[通讯]`, Hiroshi Hirai

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种半无穷线性规划的加性FPTAS，解决了向量子空间装箱问题的松弛版；

**💡 创新点**

创新点在于将该松弛问题视为欧氏建筑上的horospherically凸优化，并利用增量Busemann子梯度方法实现近似最优；

**🔧 技术方法**

核心技术包括Hadamard空间、Busemann函数、欧氏建筑的几何结构以及增量Busemann子梯度优化；

**📊 数据集**

本研究为理论算法，不使用实际数据集，而是针对符号向量子空间和线性变换的通用实例进行分析；

**📈 对比分析**

与已有的CLV算法相比，所给的算法实现更简单，但时间复杂度为O(n⁵m³)，并首次给出BL多面体成员检验的多项式时间（实数机模型）方案；

**⚠️ 局限性**

主要限制是对有理输入的比特复杂度未达到多项式级别，且在实数机模型下需要对基向量的位长进行指数级控制。

---

## 75. Robust XGBoosting for Regression

**arXiv ID:** 2608.13590 | [PDF](https://arxiv.org/pdf/2608.13590v1)

**作者:** Iris Aragón Mladosich `[一作]` (KU Leuven), Christophe Croux `[通讯]` (KU Leuven)

**通讯引用:** 15078 | [OpenAlex ID](https://openalex.org/A5025083815)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究在XGBoost回归框架下，引入多种鲁棒损失函数（M、S、τ^2、MM），构建了一系列鲁棒XGBoost变体，旨在提升对垂直离群值和杠杆点的稳健性。

**💡 创新点**

创新点在于将鲁棒回归中成熟的M、S、τ和MM估计器的损失函数直接嵌入到XGBoost的二阶泰勒近似中，并通过自定义梯度和Hessian实现；同时提出两阶段MM-XGBoost，兼顾高效率与鲁棒性。

**🔧 技术方法**

技术手段包括：1）XGBoost的梯度提升与二阶泰勒展开；2）自定义损失函数实现梯度/二阶导数；3）基于MAD或M尺度估计的鲁棒尺度计算；4）模拟实验设计与RMSE评估。

**📊 数据集**

使用人工合成数据，包含三种回归函数（线性、两种非线性），低维（p=10）和高维（p=100）场景，三种协方差结构（独立、递减相关、区块相关），以及5%/20%两种离群率的三种离群情形（垂直离群、好杠杆点、坏杠杆点）。

**📈 对比分析**

与标准XGB（平方损失）和XGB Pseudo-Huber，以及M-XGBoost进行对比。结果显示，MM-XGBoost在所有模拟配置下都实现了最优的RMSE与方差平衡，既能在无离群样本时保持与标准XGB相近的预测精度，又能在离群样本出现时显著降低RMSE；S-XGBoost表现出最强鲁棒性但效率略低，τ^2-XGBoost在效率与鲁棒性之间取得折中。

**⚠️ 局限性**

局限性包括：1）仅在XGBoost框架内验证，未对其他梯度提升实现（如LightGBM）进行全面鲁棒性改造；2）实验基于模拟数据，缺乏真实数据验证；3）鲁棒性改进主要集中在损失函数层面，对分裂策略等其他潜在影响尚未深入探究。

---

## 76. Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers

**arXiv ID:** 2608.14089 | [PDF](https://arxiv.org/pdf/2608.14089v1)

**作者:** Thiago Sandoval `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Regime-Conditional Verification (RCV)，一种轻量级包装器，利用安全分类器内部表示估计判决与部署者政策的一致性，若估计错误则自动翻转判决，并使用同一估计器进行漂移监测与维护。

**💡 创新点**

创新点在于：① 证明安全分类器内部表征能在未训练过的政策下估计判决正确性；② 采用判决分区（safe/unsafe）条件化估计器，分别训练两条校准曲线，解决单一校准无法兼顾两类错误的问题；③ 将决策纠正与漂移监测统一为同一信号，形成自动化维护循环。

**🔧 技术方法**

技术包括：白盒访问分类器内部隐藏层；使用线性/多层感知机/GBDT探针；Perceptron（Platt）校准；阈值翻转规则；基于事件计数的无标签漂移检测；基于校准的维护循环与可选的Fine‑tune。

**📊 数据集**

实验数据集为 PKU‑SafeRLHF 与 WildGuardMix（两者均为英语单轮安全数据集，含人工标注）；分类器包括 Llama‑Guard‑3‑8B、WildGuard‑7B 与 Beaver。

**📈 对比分析**

与原始分类器相比，RCV 在六个 classifier–dataset 组合中均提升了遵从度（raw→RCV）且最多可捕获 81% 先前漏判的不安全内容；在漂移检测实验中，RCV 在所有十个攻击系列上在攻击率≤0.30 时均提前报警；维护循环在 100 次漂移事件中 79 次在预算内修复，87 次在数据上限内修复，仅对 13 次无解需 Fine‑tune。

**⚠️ 局限性**

局限性包括：仅在单轮英语文本、白盒模型、人工或模型判定器的实验；漂移模拟为人工注入的类别，未覆盖真实生产漂移；需要访问内部隐藏层，API 仅暴露置信度时效果下降；遵从度评估依赖单一判定器，可能受偏见影响。

---

## 77. Benchmarking data-driven material models on the classic Treloar dataset

**arXiv ID:** 2608.14063 | [PDF](https://arxiv.org/pdf/2608.14063v1)

**作者:** Hagen Holthusen `[一作]`, Ellen Kuhl `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对六种主流数据驱动本构建模方法在 Treloar 天然橡胶实验数据上进行统一基准测试，评估其拟合精度、计算成本与模型复杂度。

**💡 创新点**

系统比较展示了在相同训练/验证框架下，稀疏回归与数据库匹配方法（EUCLID、MF、AMF）与神经网络方法（PANN、CANN、GI‑CANN）各自的优势与折衷，揭示了模型准确性、参数量和训练时间的权衡。

**🔧 技术方法**

采用稀疏回归（EUCLID）、数据库模式识别（MF、AMF）、物理约束神经网络（PANN、CANN）和可训练广义不变量的神经网络（GI‑CANN），并使用自动微分评估能量、应力与切线算子。

**📊 数据集**

Treloar 20°C 天然橡胶实验数据（拉伸-应力曲线，包含单轴、双轴拉伸和纯剪切三种载荷模式）。

**📈 对比分析**

结果显示：AMF 与 GI‑CANN 在纯剪切验证集上取得最高 R²≈0.9996；EUCLID 与 MF 拥有最小的参数量（≤4）和最低识别时间；PANN 在不同随机初始化下表现最不稳定；总体而言，各方法在不同性能维度存在明显互补。

**⚠️ 局限性**

局限性：仅针对单一材料与不可压缩等变形，未考虑更广泛材料或异构实验；部分方法对超参数/初始化敏感；未评估训练后模型的泛化至不同材料；比较未涵盖所有最新算法（如高斯过程、符号回归等）。

---

## 78. PPAPlace: Differentiable Cross-Stage Objectives for Chip Placement Optimization

**arXiv ID:** 2608.13790 | [PDF](https://arxiv.org/pdf/2608.13790v1)

**作者:** Ruogu Chen `[一作]` (University of Alberta), Jie Han `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种可微分的宏/标准单元混合尺寸布局预测器（PPAPlace），并将其用作定位梯度指导器，以显著提升后路由时序与功耗/面积（PPA）指标。

**💡 创新点**

核心创新在于：① 采用后全局路由（post‑GRT）阶段的标签训练预测器，解决传统 HPWL/预路由标签与最终后路由 PPA 的不匹配问题；② 双流网络（图注意力+空间卷积）同时编码网表连通性与布局密度/拥塞信息；③ 通过可微分特征提取实现从预测时序梯度到单元坐标的端到端梯度流，支持两种使用方式——作为 DREAMPlace 的协同目标（CoOpt）和后置梯度下降细化（Refine）。

**🔧 技术方法**

技术方法包括：可微分图注意力网络（GAT）、卷积神经网络（CNN）、多任务损失（MSE+对序排序损失）、后全局路由标签生成、梯度投影下降、以及与DREAMPlace 的梯度注入。

**📊 数据集**

使用 ChiPBench 的10个训练电路（基于 Nangate45 技术节点）生成约5000个post‑GRT标签；在5个未见电路上进行评估，并对IBM 45 nm SuperBlue 16/18 电路做零样本测试。

**📈 对比分析**

与 Hier‑RTLMP、DREAMPlace、AutoDMP、LaMPlace、Re²MaP 等现有 AI/分析放置器比较，PPAPlace‑CoOpt+Refine 在未见电路上平均 WNS↓22%、TNS↓51%，并在所有10个电路–指标组合中获得第一或第二名；在SuperBlue零样本测试中亦实现与 LaMPlace 相当或更优的时序排名。

**⚠️ 局限性**

局限性包括：仅在 Nangate45 训练；在不同工艺节点/标准单元库上验证不足；梯度在超过约20步后出现分布外退化；跨放置器（RTLMP）迁移性能低于同一放置器；以及 CoOpt 需要与可微分放置器耦合，难以直接应用于仅提供最终布局的商业工具。

---

## 79. Coverage Aware Active Evaluation for Failure Discovery with Paired Systems

**arXiv ID:** 2608.13719 | [PDF](https://arxiv.org/pdf/2608.13719v1)

**作者:** Anjali Parashar `[一作]` (MIT), Marco Pavone `[通讯]` (NVIDIA Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种自适应失败发现方法，结合低成本代理评估与有限的目标系统测试，主动挑选最可能导致严重失败的情景，并覆盖多样化的失败模式。

**💡 创新点**

创新点在于：① 使用局部控制变量（Control‑Variate）方法将代理信号校正为对目标系统风险的更精准估计；② 引入支持感知互信息（Support‑Aware Mutual Information）作为探索度量，平衡失败可能性与对未覆盖区域的探测；③ 将上述两种目标统一为批量化采样策略，适用于高维情景空间。

**🔧 技术方法**

采用 Bayesian 实验设计、控制变量估计、互信息优化、神经网络代理（BNN/MLP）、聚类与支持感知度量等技术。

**📊 数据集**

实验数据集包括：nuPlan 驾驶场景（384维），Manipulation 任务（19维），Quadruped 速度跟踪（3维），以及 KITTI 与 Virtual KITTI 的感知失败场景。

**📈 对比分析**

与 Random、BAMS、BNN‑C/GP‑C、BNN‑CV 等基线比较，实验表明在相同目标评估预算下，本文方法发现的严重失败数可达到基线的约 2 倍，且覆盖率与多样性显著提升，尤其能捕捉到高严重度的失败。

**⚠️ 局限性**

局限性包括：需要场景表示能够反映局部性与支持，代理与目标的相关性不足时效果受限；在更大或更复杂的情景空间中需要更高预算；仍需目标系统评估，无法完全消除对真实测试的依赖。

---

## 80. A Barrier-Free Synchronization Algorithm for Multi-Engine AI Accelerators

**arXiv ID:** 2608.13757 | [PDF](https://arxiv.org/pdf/2608.13757v1)

**作者:** Chungha Sung `[一作]` (Amazon), Joonwon Choi `[通讯]` (Amazon)

**通讯引用:** 245 | [OpenAlex ID](https://openalex.org/A5003461001)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

为AWS Trainium等多引擎AI加速器设计并实现了一种无障碍同步算法，利用共享计数器和循环计数器在结构化控制流下精确地为每个依赖生成闭式阈值，实现跨引擎、跨循环的细粒度同步。

**💡 创新点**

创新点包括：①对多引擎架构下的依赖满足条件进行精确表述；②提出基于循环计数的闭式阈值分配，消除全障碍的过度同步；③在Lean中完成了约15K行无零假设的机械证明；④通过实验验证了10–45%延迟提升和3.3×的微基准加速。

**🔧 技术方法**

技术手段包括共享计数器（Semaphore）、局部循环计数器（trip / monotone registers）、结构化控制流图（SCFG）分析、闭式阈值推导、AWS Neuron ISA后端实现、以及Lean证明框架。

**📊 数据集**

使用的数据集为AWS NKI生产的ML kernels（TiledMatmul、RMSNorm、FlashAttention、AdamW、Dropless MoE）以及自定义微基准AddManyLoop，全部在Trainium 2芯片上运行。

**📈 对比分析**

比较方法：在同一硬件（Trainium 2）上对比全障碍、无障碍（本论文）、手工分配、循环展开四种同步方案，测量端到端延迟、代码大小、寄存器和Semaphore占用；结果显示无障碍方案平均减少10–45%延迟，微基准实现3.3倍加速，且在绝大多数kernel上与手工分配相当或更优。

**⚠️ 局限性**

局限性：算法依赖于可通过共享计数器实现同步的多引擎架构；对高度紧凑、compute‑bound的内部循环可能需要展开以隐藏额外ALU开销；资源调优（Semaphore、寄存器、ALU）仍需人工决定；验证仅覆盖结构化控制流和特定可分配依赖子集。

---

## 81. Hard Cases, Bad Labels: Testing Error Exposure and Error Location in Uncertainty Sampling Under Bounded Label Noise

**arXiv ID:** 2608.13601 | [PDF](https://arxiv.org/pdf/2608.13601v1)

**作者:** John Myron Uy `[一作]` `[通讯]` (Independent Researcher), John Myron Uy (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在三种二分类表格数据集上，使用配对随机种子和预算相关的正则化，比较了margin‑based不确定性采样和随机采样在干净标签、随机分类噪声以及与难度相关的噪声下的标签效率，并评估了AULC、最终准确率、平均精度和固定FPR下的TPR等指标。

**💡 创新点**

创新点包括：① 引入暴露匹配的随机噪声控制来区分噪声暴露与位置效应；② 设计了与难度相关的“bounded difficulty‑dependent noise”模拟实际注释难度；③ 对完整学习曲线进行预算依赖分析；④ 发现平衡准确率提升而平均精度与TPR下降的指标反转现象。

**🔧 技术方法**

技术手段包括逻辑回归（L₂ 正则化）+margin‑based不确定性采样、随机采样、随机森林参考模型生成难度分数、伯努利噪声生成、配对随机种子、10,000 次配对 Bootstrap 估计、Wilcoxon 符号秩检验、Holm 校正、AULC 计算、平衡准确率、平均精度和固定 FPR 下的 TPR。

**📊 数据集**

使用了 UCI 公开的三大二分类表格数据集：Breast Cancer Wisconsin、Banknote Authentication 和 MAGIC Gamma Telescope。

**📈 对比分析**

通过配对种子对比两策略的 AULC、最终准确率等，结果显示：在干净标签下不确定性采样平均提升 1.1–1.8 个百分点的 AULC；在与难度相关的噪声下提升幅度显著下降（部分数据集未显著）；在随机噪声下仍保持正向优势；匹配控制未发现位置效应，某些指标出现反向；整体表明不确定性采样在低预算下更为标签高效。

**⚠️ 局限性**

局限性包括：仅三大二分类表格数据集、仅使用逻辑回归与两种采样策略、噪声为人工模拟且与参考边界无关、未对每个种子暴露轨迹做精确匹配、匹配控制仅在预算 120 时进行、指标仅限于 AULC、平衡准确率、平均精度和固定 FPR TPR、未公开预注册、早期正则化选择可能产生波动。

---

## 82. Ontology-Grounded Project Memory for Coding Agents

**arXiv ID:** 2608.13662 | [PDF](https://arxiv.org/pdf/2608.13662v1)

**作者:** James Adam `[一作]` `[通讯]` (Trivyn), James Adam (Trivyn)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MOOSEDev——一种基于本体的项目记忆系统，为编程代理提供结构化、可追溯的知识图谱，帮助其在代码生成过程中记住架构决策、约束和推理逻辑。

**💡 创新点**

创新点在于将编程代理记忆建模为本体驱动的知识图谱，结合神经符号引擎与Model Context Protocol，实现结构化捕获、生命周期跟踪与关系推理，从而显著提升对完整性、否定与超时关系的推断能力。

**🔧 技术方法**

采用OWL+SHACL本体、NeuroSymbolic引擎（符号层为主，LLM为传感器）、MCP接口、SPARQL查询、向量检索（mem0）与BM25基础检索，融合结构化与向量化搜索。

**📊 数据集**

主要数据集为CodeGraph的文档，转化为835条类型化记录（MOOSEDev图谱）和553条向量存储记录，并在公开与私有语料库上进行基准验证。

**📈 对比分析**

通过五种记忆/检索条件的对照实验（无记忆、平面文档、向量存储、BM25检索、结构化图谱）和严格LLM评判器，MOOSEDev在完整性、否定与超时任务上取得0.98–1.00的高分，向量存储仅6–27%，在相关性和代币成本上两者相近，但MOOSEDev在当前答案检索上始终准确。

**⚠️ 局限性**

局限性包括需要专业的记录捕获流程、对捕获质量高度敏感、依赖主机工具和MCP接口、适用于小型本体且对大规模代码库的扩展性尚待验证。

---

## 83. ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models

**arXiv ID:** 2608.14022 | [PDF](https://arxiv.org/pdf/2608.14022v1)

**作者:** Xinye Li `[一作]` (Chinese University of Hong Kong), Wai Lam `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

ForgeWM提出了一个四阶段的进化式因果训练框架，将双向动作条件生成器转化为多预算（1、2、4步）并保留帧级键盘与鼠标控制的少步视频世界模型，并支持草稿保持式重放细化。

**💡 创新点**

创新点在于：①通过四阶段（双向适配→教师强制因果训练→一致性初始化→自我策略分布匹配）将双向生成器快速转为预算专用因果学生；②保留游戏原生键盘/鼠标控制的完整对齐；③引入草稿保持式重放细化，无需额外细化器；④该训练流程可直接迁移到其他控制空间（如FPS手柄）而无需结构改动。

**🔧 技术方法**

技术手段包括：VAE潜在压缩、流匹配适配、教师强制因果训练、一致性蒸馏、分布匹配蒸馏、自回归自滚、动作接口模块（键盘/鼠标分路）、草稿重放细化、跨域适配。

**📊 数据集**

主要使用GF‑Minecraft 40k段视频（640×352）进行训练；在主评估中采用Minecraft数据集；另在跨域实验中使用Halo Infinite、Modern Warfare等FPS游戏。

**📈 对比分析**

与Matrix‑Game 2.0、HY‑WorldPlay在77帧 roll‑out 上进行对比，评估指标包括VBench IQ、LPIPS、AQ、KCtrl、鼠标准确率、FPS与延迟；ForgeWM‑1 在FPS与延迟上最优（168 ms/72 fps），ForgeWM‑2/4 在 KCtrl 与视觉一致性上领跑；人类偏好测试显示 ForgeWM‑4 以 60.7% 的总偏好率占优；重放细化在保持原稿的同时可达 4 步参考质量。

**⚠️ 局限性**

局限性在于评估聚焦于受控的 Minecraft 环境，未探讨对分布外的泛化；长时序会出现结构与色彩漂移；需对齐的键盘/鼠标控制接口限制了直接迁移至不同控制格式的适用性。

---

## 84. Capacity-Dependent Effects of Data Selection for Reasoning

**arXiv ID:** 2608.13721 | [PDF](https://arxiv.org/pdf/2608.13721v1)

**作者:** Cuong Dang `[一作]` (Virginia Tech), Ruoxi Jia `[通讯]` (Virginia Tech)

**通讯引用:** 2958 | [OpenAlex ID](https://openalex.org/A5032275274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在推理任务中基于模型当前似然值选择训练样本的效果，发现其性能随模型容量和训练时长呈现“Fast‑Fit / Slow‑Gain”模式。

**💡 创新点**

创新点在于：①提出高似然样本对小模型快速收敛、低似然样本对大模型长期提升的规律；②给出容量受限蒸馏的理论解释，说明数据、模型容量与知识迁移的三者相互作用；③用学习动态与分布偏移双视角阐释了现象。

**🔧 技术方法**

使用技术包括：大规模语言模型（1.5B–8B）在数学推理数据上做监督微调（SFT）；GRAPE式的基于似然的响应选择；多周期检查点评估；教师/原模型视角的损失曲线分析；线性/深度线性网络的蒸馏理论推导。

**📊 数据集**

数据集主要为MATH12K（训练集）和10个公开数学测评数据集（AIME24、AMC、CHMATH、Gaokao、GPQA、GradeSchool、KAOYAN、MATH500、Minerva、Olympiad Bench）用于评估。

**📈 对比分析**

比较方法：对同一模型规模分别用高似然（𝒟_high）和低似然（𝒟_low）数据训练，记录第1个epoch和第5个epoch的最佳pass@1分数。结果显示：在1.5B–4B模型中，高似然数据始终在早期（1 epoch）表现更好；在7B–8B模型中，低似然数据在5 epochs后取得更高分数，最终优势可达约10–20%pass@1。该结果验证了容量/训练时长依赖的双向优势。

**⚠️ 局限性**

局限性：①理论模型仅在线性/深度线性分类器上得到证明，未直接验证到大规模Transformer；②实验聚焦于数学推理任务，其他推理/生成任务的适用性待验证；③数据选择仅考虑单一似然度量，忽略了示例多样性、长度等其他难度维度。

---

## 85. From Passive Delegates to Strategic Negotiators: Reinforcing Social Reasoning in Small Language Models with SocialRL

**arXiv ID:** 2608.13787 | [PDF](https://arxiv.org/pdf/2608.13787v1)

**作者:** Wenyue Hua `[一作]` (Microsoft Research), Asli Celikyilmaz `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过后训练将小型4B语言模型转化为能够在多种代理决策环境中进行社交推理和战略代理的模型，并提出了一套完整的事件驱动环境接口与解耦式训练架构。

**💡 创新点**

创新点在于（1）构建可独立于代理实现的事件驱动多智能体环境接口，支持异步通信、部分可观测和多阶段奖励；（2）设计了解耦式训练栈，使用OpenAI兼容的rollout代理实现环境、代理和训练器的完全分离；（3）提出SocialRL框架，在六种不同的代理决策任务（分配、价格谈判、协调等）上统一训练和评估社交推理；（4）通过交叉环境转移矩阵指导策略统一，提出transfer-aware cascade RL和多教师OPD两种高效合并策略；（5）引入Explicit Theory‑of‑Mind (Infer→Act→Anticipate) 监督，证明下一步行动预测对谈判表现最关键。

**🔧 技术方法**

技术包括：事件驱动多智能体环境接口、OpenAI兼容rollout代理、PPO强化学习、SFT预训练、逆KL多教师distillation (MOPD)、gap‑closed自适应采样、ToM监督与SFT、以及多任务跨环境评估。

**📊 数据集**

使用六个自定义代理决策环境：Deal‑or‑No‑Deal、CaSiNo、Craigslist Bargains、Job Interview、Calendar、Marketplace，全部基于SocialReasoning‑Bench与Craigslist公开数据构造；此外利用GPT‑5.2生成的ToM示例进行监督。

**📈 对比分析**

与GPT‑4.1、GPT‑5.1、GPT‑5.2等前沿模型比较，域内训练的4B专家平均得分为0.619，已达到或超过这些大模型；cascade RL统一后Avg‑6得分0.627，匹配GPT‑4.1；MOPD统一后Avg‑6为0.597，恢复92.6%专家优势，训练成本显著降低。

**⚠️ 局限性**

局限性包括：跨域转移仍高度不对称，某些任务如Calendar在统一后仍表现低于单域专家；对ToM的监督依赖GPT‑5.2生成的数据；模型仍可能在信息泄露或谎言方面出现问题；统一策略在不同任务间的权衡仍需进一步自动化；实验主要集中在文本交互环境，缺乏对真实物理交互或多模态环境的验证。

---

## 86. Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use

**arXiv ID:** 2608.14047 | [PDF](https://arxiv.org/pdf/2608.14047v1)

**作者:** Yi Ding `[一作]` (Astribot), Jianan Wang `[通讯]` (Astribot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Agentic Robot with Tool-use（ART）框架，结合端到端Vision‑Language‑Action模型与多模态工具使用，实现对复杂环境的适应与错误纠正；

**💡 创新点**

通过工具注入（tool‑token）和动态LoRA微调实现工具推理与动作生成的非破坏性分离，构建30K工具轨迹数据集，显著降低对大规模数据的依赖，并提升模型的泛化与鲁棒性；

**🔧 技术方法**

LoRA微调、工具令牌注入、动作分块（action‑chunking）、多模态工具增强（视觉、可供性、身体姿态）以及GPT生成工具链与轨迹；

**📊 数据集**

30K工具使用轨迹与动作演示数据集，来源于LIBERO、Bridge v2、DROID等，随后在LIBERO模拟环境和Astribot S1真实机器人上进行实验；

**📈 对比分析**

与OpenVLA、π_0、π_0‑FAST等SOTA端到端VLA模型对比，在LIBERO和Astribot S1任务中，ART分别取得约75%和62%的平均成功率，比基线提升约20%；在工具推理、可供性任务上也优于ECoT和传统的FAST；

**⚠️ 局限性**

工具集仍有限，需依赖GPT生成工具链，极端环境或未知工具缺失时性能可能下降；实现复杂工具调用顺序仍是技术挑战。

---

## 87. HAM-RAG: Hierarchy-Aware Multimodal RAG for Structure-Faithful Interleaved Generation

**arXiv ID:** 2608.14032 | [PDF](https://arxiv.org/pdf/2608.14032v1)

**作者:** Yin Li `[一作]` (Hong Kong University of Science and Technology), Fugee Tsung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用文档层级信息实现结构忠实的多模态检索增强生成框架HAM‑RAG，并构建了对应的评测基准HAM‑Bench。

**💡 创新点**

创新点在于将文档层级作为检索与生成的核心地面信号，对文本和图像单元同时注入层级上下文与局部跨模态关系，从而实现图片选择与位置的结构化对齐。

**🔧 技术方法**

采用层级解析、层级感知的文本与图像特征构造、BGE-M3密集检索、基于结构化提示的多模态生成（LLM/MLLM），并结合上下文Aware视觉描述与层级标注。

**📊 数据集**

使用四类结构化文档数据集：游戏攻略Wukong、网页Wiki、学术论文arXiv、步骤式食谱Recipe，并在每类数据上复原层级结构。

**📈 对比分析**

与基线（文本检索+文本生成、文本+图片生成、规则式图片插入）在相同检索与生成预算下对比，HAM‑RAG在MM Avg上平均提升17.3%，在Wukong上Img‑CBS提升24.2%，在所有子集上均保持显著优于非层级模型。

**⚠️ 局限性**

局限性包括对层级信息的依赖，若文档缺失或层级恢复不准确会影响性能；对高密度图像子集（如arXiv）在图像覆盖率上略逊；模型对大规模文档生成时提示长度与计算成本仍有上限。

---

## 88. Consistent Model Chasing Is Minimax Optimal: The Exact Value of Scalar Adversarial Adaptive Control under Large Parametric Uncertainty

**arXiv ID:** 2608.13651 | [PDF](https://arxiv.org/pdf/2608.13651v1)

**作者:** Dimitar Ho `[一作]` (California Institute of Technology), Dimitar Ho `[通讯]` (California Institute of Technology)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5052748337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

解决了自适应控制在对抗干扰下的一个基本问题，精确地调节标量系统 x_t+1 = a x_t + u_t + w_t，x_0=0，w_∞≤ 1，其中常数极点 a ∈ [-Δ, Δ] 的符号和大小未知。

**💡 创新点**

首次确定了在参数不确定性和对抗干扰下，自适应控制问题的最优值 γ^⋆(Δ) = 1 + Δ，提供了精确的最优性证书。

**🔧 技术方法**

采用了确定性反馈控制法，结合了鲁棒预言机和一致模型追逐的架构。

**📊 数据集**

使用了标量线性系统的模型，参数 a 的范围为 [-Δ, Δ]，并且对抗干扰 w_∞ 的限制为 1。

**📈 对比分析**

与现有方法比较，标准工具（如探测、承诺和乐观）在此问题上均表现出严格的次优性，最优控制器的性能在对抗干扰下达到了最优值 1 + Δ。

**⚠️ 局限性**

限制在于该研究主要集中在标量系统上，如何将结果推广到更高维度的系统仍然是一个开放问题。

---

## 89. Variation Brownian Kernel Ladders

**arXiv ID:** 2608.13882 | [PDF](https://arxiv.org/pdf/2608.13882v1)

**作者:** Mahdi Mohammadigohari `[一作]` `[通讯]` (Free University of Bozen-Bolzano), Mahdi Mohammadigohari (Free University of Bozen-Bolzano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的深度函数空间框架Variation Brownian Kernel Ladder(VBKL)，其核心思想是先通过递归构造基于布朗核的路径字典，再在外层通过符号测度的线性组合完成函数表示，形成一种路径原子化的深度模型。

**💡 创新点**

创新点包括：①将非线性递归字典与线性变差凸包分离，形成层次化的路径原子空间；②利用布朗核RKHS的显式表达和平方根正则化，实现严格的深度层级关系与 Hölder 正则性；③给出完整的统计学习理论，Rademacher 与泛化界定量化了外层变差半径与内部参数数目的独立贡献；④构造两阶段近似算法，分离测度离散化与外部曲线插值，证明 M^{-1/2}+m^{-1/2} 的误差收敛且评估时仅需 2M 个基函数。

**🔧 技术方法**

技术手段包括：布朗核 RKHS 与其复合生成的递归字典；符号测度变差空间与变差复杂度；Rademacher 复杂度、VC 熵与二次布朗混沌的结合；基于平方根 Hölder 估计的正则性分析；分段线性插值与数值方向估计；以及实验中使用的有限差分方向梯度与蒙特卡罗方向平均。

**📊 数据集**

主要使用了两类数据集：一类是控制教师函数（已知递归结构的合成函数），用于验证近似误差；另一类是实际回归任务 Energy Efficiency 数据集，用于评估模型在真实数据上的表现。

**📈 对比分析**

在实验中与 Deep Neural Variation Spaces (DNVS)、核岭回归 (KRR) 与随机傅里叶特征 (RBF) 进行对比。结果显示 VBKL 在样本量极少时表现最佳，且参数量明显低于 DNVS；在大样本情况下，DNVS 与 KRR 取得更低的均方误差，但 VBKL 仍保持竞争力。总体来看，VBKL 在有限数据且需要高参数效率的场景下具有优势。

**⚠️ 局限性**

局限性包括：①理论复杂度与泛化界限仅适用于有限下支持的显式架构；②尚未给出完整的全空间统计学习理论；③优化过程缺乏全局收敛保证，主要依赖经验验证；④对路径原子化 VBKL 的泛化性能不一定优于所有深度变差空间，尤其在大样本或非递归目标时表现不如 DNVS 或 KRR。

---

## 90. SAGE: Surrogate-gradient Adaptation via Attention-Guided Entropy for Spiking Transformers

**arXiv ID:** 2608.13702 | [PDF](https://arxiv.org/pdf/2608.13702v1)

**作者:** Kiran Nair `[一作]` (University of South Dakota), KC Santosh `[通讯]` (University of South Dakota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在Spiking ViT训练中根据自注意力熵分散度动态调节伪梯度斜率的机制（SAGE），在不改变前向推理的前提下提升训练效果。

**💡 创新点**

创新点在于利用注意力头间熵的标准差作为块级不确定性估计，并通过死区控制器实时调整伪梯度参数，实现局部自适应优化。

**🔧 技术方法**

使用自注意力熵估计、指数滑动平均、标准化及tanh映射的控制器，并在Spikformer架构中集成。

**📊 数据集**

在CIFAR‑10、CIFAR‑100和ImageNet‑200三大视觉分类数据集上进行实验。

**📈 对比分析**

与固定斜率和可学习斜率的伪梯度方案以及多种SNN/Spiking Transformer方法对比，SAGE在低时步（T≤4）时取得最高Top‑1准确率，提升幅度约1‑2%，且训练时开销仅约0.03 ms/批次。

**⚠️ 局限性**

仅适用于基于自注意力的Spiking Transformer，其他非transformer SNN需要改用不同的不确定性度量；目前仅调节伪梯度斜率，未考虑膜电位、阈值等其他神经元参数的联合自适应。

---

## 91. Language-Specific Gaps in AI Safety Training Datasets

**arXiv ID:** 2608.13695 | [PDF](https://arxiv.org/pdf/2608.13695v1)

**作者:** Chialuka Prisca-Mary Onuoha `[一作]` (Black in AI Safety & Ethics), Rashidat Sikiru `[通讯]` (Black in AI Safety & Ethics)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对21个多语言安全数据集中的25个语言切片进行了系统的质量审计，揭示了资源层级与安全数据缺口的不一致性。

**💡 创新点**

创新点在于提出并应用可重复的切片级审计方法，发现传统的集合层级覆盖声明掩盖了个别语言的质量差距。

**🔧 技术方法**

采用结构化审计表单、人工判定、翻译质量指标对比等技术手段，对每个语言切片的来源、标注、许可等维度进行评估。

**📊 数据集**

使用的资源包括21个多语言安全数据集，覆盖低资源 Hausa、中等资源 Swahili 和高资源 French，共计25个语言切片。

**📈 对比分析**

通过在同一评估管线中比较 Hausa 与 Swahili 的翻译质量，发现 Hausa 的政策材料翻译质量低于 70% 的阈值，而 Swahili 则达标，证明翻译质量差距可测量且可改进。

**⚠️ 局限性**

研究局限在于评估由单一评审者完成、缺乏双重编码、样本覆盖仅限三种语言且时间敏感，可能影响结论的普适性与长期适用性。

---

## 92. Don't Claim Benchmark-Oriented Optimization Improves General Coding Capability -- Diverse Evaluation Is Required

**arXiv ID:** 2608.13566 | [PDF](https://arxiv.org/pdf/2608.13566v1)

**作者:** Egor Shibaev `[一作]` (JetBrains Research), Sergey Titov `[通讯]` (JetBrains Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建基于Django的多模态评测套件，对SWE-bench等现有编码基准的泛化性进行实验验证；

**💡 创新点**

提出“意义缺口”概念，证明单一基准无法准确衡量模型的通用编码能力，并提出多层次评估框架（全局评估、任务套件、人工评估、开放式评测）和持续维护的能力分类方法；

**🔧 技术方法**

使用LoRA微调、Greedy解码、官方Django 4.0.4测试套件、LiveCodeBench、以及自定义的生成、补全、程序修复任务；

**📊 数据集**

主要数据集为SWE-bench（Verified）中的Django代码快照、改写后的标准化Docstring数据、LiveCodeBench 341条自包含算法题；

**📈 对比分析**

对比基础模型（Qwen3-32B、Qwen2.5-Coder-7B/32B）与公开SWE-bench优化检查点以及自身任务微调模型，发现SWE-bench优化在Django任务和LCB上几乎无提升甚至下降，任务特定微调在自任务上提升明显但跨任务几乎无益；

**⚠️ 局限性**

局限性包括评测仅覆盖Python/Django，未考虑多语言或非开源仓库；评测采用单一Greedy推断，可能低估模型潜力；数据集规模有限，未完全消除泄漏风险；

---

## 93. Architecture and Affordances of PLAUD: Performative Latents and Unsupervised DDSP

**arXiv ID:** 2608.13724 | [PDF](https://arxiv.org/pdf/2608.13724v1)

**作者:** Błażej Kotowski `[一作]` (Universitat Pompeu Fabra), Frederic Font `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并实现了名为PLAUD的实时神经合成器，融合了NoiseBandNet基的DDSP合成、变分潜在空间、可选Transformer先验以及网络弯曲（bending）与轨迹采样等交互方式，供Max for Live使用。

**💡 创新点**

创新点在于：①在DDSP链中引入可解释的噪声滤波器组和潜在平滑，②将潜在空间正则化为VAE并结合多尺度谱与对抗损失提升音质，③使用可选的Transformer先验实现自回归控制生成与反馈，④在合成链中加入组件限制、波形塑形等弯曲操作，⑤在交互层实现轨迹采样与动态控制，整体实现了从架构到表演的嵌入式可操作性。

**🔧 技术方法**

技术包括：NoiseBandNet DDSP架构、变分自编码器（VAE）与KL正则化、MelGAN式多尺度对抗损失、基于mu-law的离散量化与Transformer Encoder‑only自回归先验、GRU解码器、Max for Live接口与nn~外部部署。

**📊 数据集**

使用了个人小规模音频语料库（4–95 s至30 min的噪声、纹理、无音高材料），并通过手工挑选的“纹理化”数据集进行训练，后期在现场演示中分别训练不同模型。

**📈 对比分析**

论文主要通过现场表演、听感评估和对比实验（如使用无先验、带先验、弯曲与否）来验证系统的可玩性与音质。虽然重建误差在传统评测中偏高，但在现场演出中被视为乐器特色；与RAVE等现有模型比较时，PLAUD在小数据、实时性能与可视化交互方面表现更佳。

**⚠️ 局限性**

主要限制包括：①计算开销大，尤其是启用先验时只能单实例运行；②潜在空间与合成链的解码器为GRU，导致实时吞吐瓶颈；③模型重建精度低，先验产生漂移；④仅适用于噪声/纹理材料，缺乏对音高精确的控制。

---

## 94. Emergent Models: Intelligence from Tiny Substrates

**arXiv ID:** 2608.14019 | [PDF](https://arxiv.org/pdf/2608.14019v1)

**作者:** Giacomo Bocchese `[一作]` (Wolfram Institute), Akshaj Devireddy `[通讯]` (Wolfram Institute)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实验了一种新的机器学习范式——Emergent Models（EMs），通过简单的局部迭代规则和可学习的初始状态在动力学子系统中实现计算。

**💡 创新点**

创新点在于将学习任务视为寻找能在固定动力学子系统中产生所需行为的“程序”（初始状态），并证明在某些设置下可实现潜在的“latent universality”，即仅通过改变初始条件即可模拟任何可计算函数。

**🔧 技术方法**

技术核心是基于离散/连续细胞自动机、Conway's Game of Life、以及一维/二维连续局部更新规则的可训练模型；通过遗传算法进行程序、规则或接口的进化搜索；实现输入/输出接口与停机条件以完成任务。

**📊 数据集**

数据集主要是简单算术函数（如 x+1、2x、round(x/3)、x·(1+x²)、a·b）以及控制任务（CartPole）和自适应任务（Meta‑Life）中的随机生成样本。

**📈 对比分析**

通过与传统 FFNN、RNN、Neural GPU 等对比，EMs 在极小参数量（10–300 维）下实现了对算术函数的完美外推、CartPole 的中等控制性能（CEM1D 退化 800 次），以及在 Meta‑Life 任务中实现了基本的自适应行为；但整体性能仍低于专用方法，且不具备统计显著性。

**⚠️ 局限性**

主要局限包括：搜索空间高度离散、进化搜索效率低；模型对接口设计高度敏感；缺乏对更复杂任务的实验验证；在连续控制与自适应任务中易出现振荡、冻结等鲁棒性问题；缺乏梯度可微的训练路径。

---

## 95. Content Based Video Narration of Gameplay with Vision Language Models

**arXiv ID:** 2608.14016 | [PDF](https://arxiv.org/pdf/2608.14016v1)

**作者:** Mathew Varghese `[一作]` `[通讯]` (University of Washington), Mathew Varghese (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个无需训练、可通用的内容驱动游戏视频叙述系统，使用通用视觉‑语言模型生成电子竞技式口述评论并支持本地TTS。

**💡 创新点**

创新点在于三项机制：时间马赛克打包降低图像请求成本、基于上下文的提示抑制重复、时长控制生成配合弹性对齐实现精确语音同步。

**🔧 技术方法**

采用了GPT‑4o等VLM进行图像描述，使用OpenAI tts‑1‑hd或Apple Silicon 4B量化Voxtral TTS，帧采样+3×3马赛克拼接、Prompt工程、Pydub时间缩放。

**📊 数据集**

使用公开的游戏回放数据，主要以《Age of Empires II: Definitive Edition》60秒实时策略片段为案例，未来评测计划收集30段RTS、FPS、赛车视频。

**📈 对比分析**

通过人工主观评分和客观指标（事实真实性、重复率、同步误差、成本/延迟）对比全系统、无历史、逐帧、人工注释及原声，初步定性显示系统在连贯性和成本上优于逐帧方法。

**⚠️ 局限性**

主要局限是马赛克导致HUD文本失真而产生幻觉的游戏状态表述、未使用音频信号、固定等长分段、缺乏实时推理、单声道、时间缩放引发韵律失真等。

---

## 96. BM25-Augmented Many-Shot Translation for Low-Resource North-Eastern Indian Languages

**arXiv ID:** 2608.13722 | [PDF](https://arxiv.org/pdf/2608.13722v1)

**作者:** Aashish Dhawan `[一作]` (University of Florida), Daisy Zhe Wang `[通讯]` (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用检索增强技术（BM25+Gemini 2.5 Flash）实现无训练的英-东北印度11种语言双向机器翻译。

**💡 创新点**

将无监督检索增强方法从前期视觉任务迁移到低资源语言翻译，完全不需要模型微调。

**🔧 技术方法**

BM25检索、Gemini 2.5 Flash LLM、系统提示工程、离线语言特定检索库、Unicode归一化与去重处理。

**📊 数据集**

WMT26/23 训练对、Samanantar、Endangered Recipes、IndicGenBench FLORES、English–Bodo 并行语料、NLLB‑200、Kokborok 数字化项目等多源并行文本。

**📈 对比分析**

对每个语言对做 4×4 r/d 网格搜索，使用 ChrF/ChrF++ 评估；在 19 个翻译任务中多数取得最优 BLEU 或 ChrF++，提升幅度微小但可测。

**⚠️ 局限性**

完全依赖 Gemini API，受限于速率与安全过滤；检索库质量参差，开发集与测试集域差导致性能下降；缺乏模型微调限制系统可扩展性。

---

## 97. Learning to Assemble Novel Structures with Unfamiliar Parts under Semantic Constraints

**arXiv ID:** 2608.13684 | [PDF](https://arxiv.org/pdf/2608.13684v1)

**作者:** Jonghyuk Park `[一作]` (University of Edinburgh), Subramanian Ramamoorthy `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种神经符号架构，支持机器人在部署时通过与教师的自然语言对话和演示，在线获取并利用语义约束来组装新结构。

**💡 创新点**

创新点在于将教师的通用语义约束（如“所有消防车的挡泥板都是红色”）自动转译为一阶逻辑规则并即时注入ASP规划器，从而显著提升在线学习效率。

**🔧 技术方法**

技术包括基于RGB图像的视觉分类与3D几何提取、对话管理中的语义解析、ASP规划的目标选择与连线序列规划，以及通过案例记忆或通用规则更新的符号记忆。

**📊 数据集**

使用了一个模拟的玩具卡车组装环境，其中包含22种原子部件和6种卡车类型，实验共生成30个数据集，每个数据集包含40个组装任务。

**📈 对比分析**

与只使用部件标签或案例记忆的对照方法相比，加入教师通用规则的策略在累计误差（regret）上平均下降约30%，视觉F1分数相近，表明性能提升主要来自符号规划层面的约束利用。

**⚠️ 局限性**

局限性包括仅在受控的模拟环境中验证，缺乏对真实机器人感知和开放域语言的鲁棒性测试，以及当前仅采用感知到符号的单向耦合，未实现符号信息回馈至神经网络的深度修正。

---

## 98. Thinking outside the box is useless NFA = FNFA

**arXiv ID:** 2608.14111 | [PDF](https://arxiv.org/pdf/2608.14111v1)

**作者:** Maxence Ponsardin `[一作]`, Ville Salo `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了在任意维度下，允许图片行走非确定性自动机退出图片与传统自动机等价，即 NFA = FNFA。

**💡 创新点**

首次给出全维通用证明，核心创新是导航引理与半空间穿越序列的引入。

**🔧 技术方法**

采用路径分解、半线性集合、PDA 与 Steinitz 定理等理论工具。

**📊 数据集**

无数据集，全部为形式化证明。

**📈 对比分析**

通过理论比较二者语言类，未给出实验性能；证明两类模型等价，复杂度保持不变。

**⚠️ 局限性**

局限在于仅覆盖平行多面体形状的图片，对更一般几何形状的适用性尚未讨论。

---

## 99. Owner3D: Ownership-Guided Style Writing for Training-Free Localized 3D Stylization

**arXiv ID:** 2608.14078 | [PDF](https://arxiv.org/pdf/2608.14078v1)

**作者:** Suchang Tao `[一作]` (Sichuan University), Yuqi Ouyang `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑free 的局部 3D 风格迁移框架，直接在大型重建模型（LRM）的解码阶段控制风格写入，只生成一张局部风格化的三平面特征；

**💡 创新点**

核心创新在于：① 基于 ownership‑guided 的风格写入，仅在目标区域的解码查询中注入风格残差；② boundary dual slots 在语义边界处为目标与非目标表面提供独立的局部特征槽，缓解三平面共享导致的容量冲突；③ surface‑first texture readout 分层利用表面、3D 与三平面所有权信息，优先选取可靠的纹理特征，提升在可见度不足时的表现；

**🔧 技术方法**

使用了 InstantMesh‑Large 作为预训练 LRM，Zero123++ 生成源视图，SAM‑3 进行目标分割，风格编码器提取风格键值，跨注意力与门控残差实现风格写入；在渲染阶段采用层级所有权投射与双槽抽样读取纹理特征；

**📊 数据集**

在 Google Scanned Objects (GSO) 与 PartNet 两个真实扫描与部分标注数据集上构建 benchmark，涵盖 49 个 3D 对象和 16 种风格；

**📈 对比分析**

与 StyleSplat、LAENeRF、G‑Style 等 3D 风格化方法以及多种图像空间风格化基线进行对比；实验显示在 Style Fidelity、T‑Gram、Leakage、MV Cons. 与 Runtime 四个指标上均优于现有基线（Leakage 下降 86.4%/89.9%，MV Cons. 0.00009，单例对话 31.10s）；

**⚠️ 局限性**

局限性包括：对 2D 分割质量高度依赖；边界多边形复杂时仍可能出现轻微泄漏；目前仅支持单视角输入，缺乏对多视图一致性细粒度控制；未来需改进区域控制与更丰富的 3D 表示

---

## 100. Scaling Domain Data Repetition in LLM Pretraining

**arXiv ID:** 2608.14071 | [PDF](https://arxiv.org/pdf/2608.14071v1)

**作者:** Jingwei Li `[一作]` (Tsinghua University), Jingzhao Zhang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在固定 tokens‑per‑parameter（TPP）比例下，高质量域数据的重复使用对大型语言模型训练的影响，系统探讨了最佳重复次数随模型规模、域特性和数据量的关系。

**💡 创新点**

发现最佳重复次数与域的最小验证损失呈强负相关；在相同 TPP 条件下，最佳重复次数随模型规模轻微上升，而与独特高质量数据比例几乎无关；并给出理论解释说明噪声拟合与知识获取的平衡。

**🔧 技术方法**

使用实验平台在多尺寸模型（通过固定 TPP 扩展）上训练，采用全批量梯度下降、不同学习率调度；理论层面构建一维线性回归模型并推导最优停止时间；利用二次曲线拟合验证损失与重复次数关系。

**📊 数据集**

四个高质量域：代码（Code）、数学（Math）、维基百科（Wiki）和医学（Medical），每个域在不同模型尺寸、独特数据比例（α=1/40、1/20、1/10）和重复次数（e=1…7）下进行实验；同时评估 OOD 集合（ArXiv、News）和不同学习率调度。

**📈 对比分析**

通过对比不同重复次数下的在域内与 OOD 验证损失、以及对学习率调度的敏感性来评估方法。结果表明：在域内，低验证损失的域可承受更多重复；在 OOD 侧变化不大；学习率衰减越早，允许的重复次数越少；总体性能在最佳重复次数附近达到最低验证损失。

**⚠️ 局限性**

局限性：仅在单一高质量域内重复，未考虑多域同时重复和交互；实验数据量范围有限，未覆盖极大或极小独特数据比例；理论模型简化为无噪声线性回归，可能与实际 LLM 训练差异；缺乏对更大模型尺寸的实证验证。

---

## 101. The Sharp Dimension Bound in the Johnson--Lindenstrauss Lemma

**arXiv ID:** 2608.13782 | [PDF](https://arxiv.org/pdf/2608.13782v1)

**作者:** Vishesh Jain `[一作]` `[通讯]` (University of Illinois Chicago), Vishesh Jain (University of Illinois Chicago)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

论文解决了Larsen和Nelson提出的关于约翰逊-林登斯特劳斯引理的最佳目标维度的猜想，证明了在给定的参数范围内，最小的目标维度是Θ(min{d,n-1,log(2+ε^2n)/ε^2})。

**💡 创新点**

创新点在于证明了这个上界是通过线性映射实现的，并且匹配的下界也适用于非线性嵌入。

**🔧 技术方法**

使用了线性映射和正半定矩阵的秩降低技术。

**📊 数据集**

使用了n个点的d维欧几里得空间中的点集作为数据集。

**📈 对比分析**

通过与之前已知的上界进行比较，证明了在特定条件下，新的上界可以显著改善，尤其是在d≥n且ε^2n与log n相近的情况下。

**⚠️ 局限性**

限制在于该结果主要适用于线性映射，可能无法直接推广到所有类型的嵌入。

---

## 102. Simulation-Aware In-Context Policy Improvement for LLM-Aided Analog Layout Refinement

**arXiv ID:** 2608.13767 | [PDF](https://arxiv.org/pdf/2608.13767v1)

**作者:** Bingyang Liu `[一作]` (University of Texas at Austin), David Z. Pan `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的多智能体框架，利用模拟感知的 ICPI（In-Context Policy Improvement）循环对模拟 IC 布局生成器的优化参数进行迭代改进，从而在仅数十次后布局仿真预算下提升后布局性能。

**💡 创新点**

创新点在于（1）设计了面向 LLM 的紧凑布局状态表示，将电路拓扑、几何信息、已曝光的优化参数、寄生摘要和仿真结果整合为结构化上下文；（2）构建了设计日志（Design Journal）以跨轮次记录经验，实现持续的经验累积；（3）实现了 act–observe–reflect 的三代理循环（Supervisor、Executor、Reflector），在模拟反馈驱动下自适应调整参数；（4）在 LLM 端使用一次性提示而非模型权重更新，保持模型通用性。

**🔧 技术方法**

技术上采用 GPT‑5（或其他可替换 LLM）作为核心推理模型；使用 MAGICAL 生成器的扩展接口公开 placement 与 routing 的参数空间；通过 Analyzer 对电路进行一次性先验分析，Prune 参数空间；在每轮中，Supervisor 选择参数族并设定目标，Executor 依据日志生成具体参数修改并调用生成器；Reflector 生成日志条目；整个流程通过 LLM 的链式思维与自反思实现。

**📊 数据集**

实验数据集为两款真实 OTA 电路：OTA1（65 nm CMOS 两级 Miller‑补偿）和 OTA2（40 nm 双差分带 CMFB）。每个案例都完成完整的布局生成、寄生提取（Siemens Calibre）与 Spectre 仿真（Cadence Virtuoso）。

**📈 对比分析**

对比方法包括：启发式默认参数、贝叶斯优化（BO）以及无 ICPI 的 LLM 方法。实验结果显示，ICPI 在同样的 31 次候选评估（每 3 次做一次仿真）下，OTA1 的 FoM 从 0.869（启发式）提升至 1.104（ICPI），OTA2 从 0.696 提升至 1.038；面积虽略大或相近，但显著优于 BO（OTA1 0.979，OTA2 0.880）和无 ICPI（OTA1 1.003，OTA2 0.868）。

**⚠️ 局限性**

局限性包括：对更大规模电路的可扩展性受限（状态表示维度与搜索空间复杂度高）；仅在电气 FoM 上优化，未直接控制面积或多目标目标；对后布局仿真的高成本仍然是瓶颈；目前模型仅基于一次性提示，无法动态更新权重；设计日志的手工格式化与检索开销亦是关注点。

---

## 103. Sequence prediction under a lying oracle

**arXiv ID:** 2608.14102 | [PDF](https://arxiv.org/pdf/2608.14102v1)

**作者:** Puspabeethi Samanta `[一作]` (IIT Bombay), Jayakrishnan Nair `[通讯]` (IIT Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在存在有限次欺骗的“十问游戏”启发下的扰动对数损失下的序列概率预测问题，分别在随机和对抗环境中给出加 β 估计器与指数加权在线优化器（EWOO）的对数阶回报上界。

**💡 创新点**

创新点在于证明该扰动损失函数满足 α-exp‑concavity，从而能利用 EWOO 在对抗环境中获得对数回报；同时给出了该损失函数在随机环境中的 add‑β 估计器的对数阶回报上界，扩展了传统对数损失理论。

**🔧 技术方法**

主要技术包括：add‑β 估计器（对称 Dirichlet 先验的贝叶斯后验）、指数加权在线优化器、对损失函数的凸性与 exp‑concavity 分析、理论证明与上界推导，以及数值模拟验证。

**📊 数据集**

使用的实验数据为合成数据：Bernoulli(p) 源（m = 2）以及针对对抗环境设计的二进制序列 0^n-11。

**📈 对比分析**

通过与不同 β 值、不同 k（欺骗次数）以及不同序列长度 n 的模拟结果比较，发现：add‑β 估计器的平均累计回报随 k 线性增长、随 n 对数增长；EWOO 的累计回报亦随 n 对数增长，并随 k 显著增加；实验结果与理论上界相符，验证了对数阶回报的有效性。

**⚠️ 局限性**

局限性包括：未给出下界证明；EWOO 的理论实现计算量大，需采样近似；实验仅使用合成数据，缺乏真实世界数据验证；对 β 与 k 的选择仍需经验调参。

---

## 104. InstructVVT: Instruction-Driven Video Virtual Try-On without Auxiliary Spatial Priors

**arXiv ID:** 2608.14070 | [PDF](https://arxiv.org/pdf/2608.14070v1)

**作者:** Dingbao Shao `[一作]`, Zili Yi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于Diffusion Transformer的指令驱动视频试衣框架InstructVVT，仅使用源视频、参考服装和自然语言指令实现无人工空间先验的视频试衣。

**💡 创新点**

创新点在于通过多模态大型语言模型生成编辑意图token与轻量化服装token相结合的双层参考条件，并采用训练无监督的MLLM奖励与DiffusionNFT后训练优化人类偏好。

**🔧 技术方法**

使用的技术包括Diffusion Transformer (DiT)、多模态LLM (MLLM)、VAE编码器、双向条件注入、流匹配训练、DiffusionNFT强化学习以及基于MLLM的奖励函数。

**📊 数据集**

使用的数据集包括TripVVT-10K、ViViD视频三元组、VITON-HD、DressCode等，评测基准为ViViD-S和TripVVT-Bench。

**📈 对比分析**

与ViViD、CatV2TON、MagicTryOn、TripVVT、UniVideo及Kling等开源/商用编辑器对比，InstructVVT在VFID、CLIP、背景一致性、MLLM-Reward等指标上均取得领先，尤其在TripVVT-Bench用户偏好中获得79.4%的首选率。

**⚠️ 局限性**

局限性包括对极端遮挡/多人物场景的鲁棒性尚未完全解决，LLM推理速度与训练成本较高，以及在极端视角或细节纹理复制方面可能出现失真。

---

## 105. Weird Machines in Transport Layer Security

**arXiv ID:** 2608.13685 | [PDF](https://arxiv.org/pdf/2608.13685v1)

**作者:** Michael Collins `[一作]` (Laboratory for Advanced Cybersecurity Research), Jonathan Takeshita `[通讯]` (Old Dominion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将“奇怪机器”理论扩展到 TLS 协议实现，构造了基于 OpenSSL 的防御式 sentinel 和攻击式认证绕过示例，并对 OpenSSL 与 BoringSSL 的 gadget 可用性做对比。

**💡 创新点**

创新点在于：①将物理驱动的奇怪机器转化为“信任驱动”，即 TLS 计算与认证决策耦合；②提出专门针对 TLS 的八类 gadget 分类法；③用 HRU 证明思路给出 TLS 实现的图灵完备性论证；④展示同一组 gadget 可被同构组合成正反两种用途；⑤指出库设计选择决定可构造的奇怪机器集合。

**🔧 技术方法**

技术方法包括：对 TLS 握手状态机的细粒度拆解、gadget 分类与映射；使用 OpenSSL 1.1.1w API 进行沙箱演示；Docker Compose 搭建真实二进制交互；SHA‑256 等数值运算；重协商、会话缓存等 TLS 机制；理论证明采用 HRU 风格的图灵机构造。

**📊 数据集**

数据集主要是公开的 OpenSSL 1.1.1w 二进制和其源码；BoringSSL 的设计文档作为对比依据；未使用任何外部标注数据集。

**📈 对比分析**

比较方法：在同一 Docker 网络环境下分别运行 sentinel 与攻击示例，验证两者可在同一套 gadget 下实现防御与攻击；对 OpenSSL 与 BoringSSL 的 gadget 可用性做文档层面对比。论文未提供性能计量，只关注功能可行性与安全影响。

**⚠️ 局限性**

局限性：仅在已停止维护的 OpenSSL 1.1.1w 上实验；未在活跃的 OpenSSL 3.x 或 BoringSSL 上实现对照；实验环境仅为 localhost，缺乏真实网络延迟与中间件干扰；仅验证单一组合错误（重协商后的密码强度检查缺失），未探讨其他可能的 gadget 组合；理论证明为证明思路草稿而非正式形式化；未实现自适应攻击者或多路径探测。

---

## 106. When Denoising Hurts: Rethinking the Terminal Step of Diffusion Time Series Forecasters -- Extended Version

**arXiv ID:** 2608.14067 | [PDF](https://arxiv.org/pdf/2608.14067v1)

**作者:** Dat Nguyen-Cong `[一作]` (FPT Corporation), Tung Kieu `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究扩散模型在时间序列预测中的逆扩散过程，并发现后期低噪声阶段会导致统计漂移，从而降低预测质量。

**💡 创新点**

创新点在于提出无标签的全局停止判据来提前终止逆扩散，并引入伯努利时间步采样器，将训练重点放在高噪声阶段。

**🔧 技术方法**

采用条件扩散模型、Patch‑based denoising 网络、线性噪声调度以及自适应伯努利时间步采样技术。

**📊 数据集**

在八个公开数据集（ETTh1/2/m1/2、Weather、Electricity、Exchange、Traffic）上进行评估。

**📈 对比分析**

与多种确定性与概率性基线（如 DeepAR、N‑BEATS、Informer、DLinear、Diffusion‑based 方法）比较，平均 MSE/MAE 提升约 4–7%，CRPS 在多数数据集排名第一，且推理时间下降 20–50%。

**⚠️ 局限性**

主要限制包括对高噪声阶段的假设可能不适用于所有时间序列；在极端非线性或季节性强的场景下仍需进一步验证。

---

## 107. Student-ChatGPT Interaction Visible: Designing a Teacher Dashboard for EFL Writing Education

**arXiv ID:** 2608.13587 | [PDF](https://arxiv.org/pdf/2608.13587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 108. CSG-Mamba: A Convolutional Scoring Gating Vision State Space Network for Endoscopic Polyp Segmentation

**arXiv ID:** 2608.14146 | [PDF](https://arxiv.org/pdf/2608.14146v1)

**作者:** Yuliang Wang `[一作]` (Tiangong University), Shuxia Ren `[通讯]` (Tiangong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Vision State Space模型的终端息肉分割网络CSG-Mamba，并在瓶颈位置加入卷积评分门控模块，以提升边界质量与跨域泛化。

**💡 创新点**

将大核深度卷积与乘法门控结合的卷积评分门控模块嵌入到Mamba瓶颈处，有效弥补SS2D序列化导致的二维连续性丧失，并在不增加显著推理成本的前提下提升性能。

**🔧 技术方法**

Vision State Space模型（Mamba）、SS2D扫描、卷积评分门控（大核深度卷积+点卷积+Sigmoid门控）、EMA、DropPath、深度监督、BCE+Dice损失。

**📊 数据集**

Kvasir-SEG、CVC-ClinicDB、CVC-ColonDB。

**📈 对比分析**

与U-Net、ResUNet、PraNet、TransUNet、VM-UNet、VM-UNetV2等方法在相同数据划分和训练协议下对比，CSG-Mamba在Kvasir-SEG上Dice 0.922、HD95 15.87，在CVC-ClinicDB上mIoU 0.7765、Dice 0.8493，在CVC-ColonDB上Dice 0.7418，均优于基线或保持竞争性，同时参数量仅为U-Net的2倍左右。

**⚠️ 局限性**

实验仅在二维息肉分割任务，未验证对其他医疗任务的通用性；在跨域测试中对某些边界指标仍存在轻微下降；未探索不同训练策略或多中心数据的进一步泛化。

---

## 109. CavityRank: Zero-Extra-Byte Residual Routing for Cuckoo Filters

**arXiv ID:** 2608.13970 | [PDF](https://arxiv.org/pdf/2608.13970v1)

**作者:** Yongjie Guan `[一作]` `[通讯]`, Yongjie Guan

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在四槽 cuckoo filter 中利用查询等价的指纹排列编码四级残余排名，使用单路径最低排名来指导插入，并在插入后以后置图状态更新每个桶，从而在保持 64 位桶大小的前提下显著降低插入失败率。

**💡 创新点**

创新点在于：①把查询不变的指纹顺序作为隐藏的控制状态；②通过两对 lane 的排列实现两比特隐式排名，产生四级残余顺序；③在插入时仅更新路径上的桶，保持无额外元数据和表级工作空间。

**🔧 技术方法**

采用残余路径图、最小排名单路径调度、后置状态备份；实现上使用 Rust 在 Apple M4 Pro 上的 64 MiB 内存环境，利用 XOR16 和 Keyed‑XOR16 两种随机偏移模型。

**📊 数据集**

使用合成随机键流（2,048 条并行流）作为测试数据集，覆盖 4,096 与 65,536 桶规模，配合 5,000 步迁移预算，并与完全容量四阶方向图（oracle）对齐。

**📈 对比分析**

与随机 CF、CavityScan、Local Search Allocation、深度 10 BFS 等传统策略相比：在 4,096 桶时，第一比特消除 86.5% 的搜索缺口，第二比特再消除 90% 以上的剩余缺口；在 65,536 桶时分别为 84% 与 85%；在 64 MiB 方案下达到 97.75% 的负载时，平均每插入仅需 42.53 次逻辑读（比 BFS 低 88%），且无额外桶字节与表级工作空间。

**⚠️ 局限性**

限制包括：删除操作可能导致残余排名失效；实现仅单线程且插入主导，缺乏并发迁移的同步机制；对更大或不同指纹分布的扩展尚未验证；若需要兼容更宽桶或其他指纹压缩策略，则需重新设计排列类。

---

## 110. Regulation, Power, and the Compliance 1 Paradox: A Longitudinal Study of Smart Homes

**arXiv ID:** 2608.13582 | [PDF](https://arxiv.org/pdf/2608.13582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 111. Content Depth Matters in Short-Video Recommendation: Rethinking the Attention Economy

**arXiv ID:** 2608.13990 | [PDF](https://arxiv.org/pdf/2608.13990v1)

**作者:** Liwei Deng `[一作]` (University of Technology Sydney), Guodong Long `[通讯]` (University of Technology Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了短视频内容深度分数（CDS）并构建SCOPE‑Bench基准，用于评估推荐系统的内容深度。

**💡 创新点**

首次量化短视频内容深度的七级评分尺度、列表级内容深度度量（LCDS），并引入基于LLM的可扩展标注流程，构成了内容深度评估的全新框架。

**🔧 技术方法**

利用视频标题、类别标签和ASR转录文本作为输入，通过大型语言模型进行评分；同时对比多模态推荐算法与人类标注的对齐程度。

**📊 数据集**

基准数据来源于公开的ShortVideo数据集，涵盖约1.5万条视频、100万条用户交互记录，并为150k条视频提供了CDS标签。

**📈 对比分析**

在13种代表性推荐算法上进行实验，发现它们在用户参与度指标（如观看时长）上表现优秀，但在内容深度指标（LCDS）上仅略优于随机推荐，表明参与度与内容深度无正相关。

**⚠️ 局限性**

局限在于CDS主要基于文本信息，可能忽略视觉/音频内容；评估过程对类别标签和ASR文本长度存在一定偏倚，未来需要更完整的多模态评估与更细粒度的深度标注。

---

## 112. Vaulted Passkeys: A Device-Bound Proposal for Authenticated Credential Export and Import

**arXiv ID:** 2608.13806 | [PDF](https://arxiv.org/pdf/2608.13806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 113. Fast Implicit Neural Light Field Representation via Geometric Decomposition and Multi-Resolution Low-Rank Features

**arXiv ID:** 2608.13949 | [PDF](https://arxiv.org/pdf/2608.13949v1)

**作者:** Yao Guo `[一作]` (Beijing Information Science and Technology University), Chang Liu `[通讯]` (Beijing Information Science and Technology University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于几何三平面分解和多分辨率低秩特征的快速隐式光场表示方法，能够在保证高质量重建的同时显著降低训练和推理成本。

**💡 创新点**

创新点主要包括：① 将 4D 光场拆分为水平视差、空间纹理和垂直视差三平面，充分利用 EPI 结构；② 采用多分辨率低秩平面表示，将低分辨率 2D 网格与高分辨率 1D 线特征乘积结合，减少 2D 网格冗余；③ 将三平面特征拼接后通过轻量级 MLP 解码，保持连续查询能力。

**🔧 技术方法**

使用了隐式神经表示（坐标到 RGB 的连续映射）、多分辨率哈希/线性插值编码、轻量级 ReLU MLP 解码器，并在训练中采用 Adam 与 Muon 优化、混合精度训练以及余弦学习率调度。

**📊 数据集**

在公开光场数据集 EPFL、INRIA Lytro 和 Stanford Gantry 上进行实验，覆盖多种纹理、深度变化和视角依赖场景。

**📈 对比分析**

与 SIGNET、RSEN、RDF 等主流基线对比，在 PSNR、SSIM、LPIPS 等指标上取得相似或更佳表现；模型参数约 1.55 M，训练时间 21 min，推理 8 s，显著提升了收敛速度与推理效率。

**⚠️ 局限性**

在强视角依赖的非朗伯特反射或透明物体（如 Tarot 场景）中，低秩假设仍受限，可能出现局部模糊或鬼影，尚需进一步改进对高反射/透明效应的建模。

---

## 114. ChartProbe: A Diagnostic Study on Visual Reasoning through Perception, Grounding, and Simple Reasoning

**arXiv ID:** 2608.13766 | [PDF](https://arxiv.org/pdf/2608.13766v1)

**作者:** Mahsa Khoshnoodi `[一作]` (Georgetown University), Sarah Adel Bargal `[通讯]` (Georgetown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ChartProbe诊断框架，能够将图表问答拆分为感知、绑定和简单推理三项技能，并通过程序生成的题目实现无人工标注的精确答案；

**💡 创新点**

创新点在于：① 用可编程代码生成的题目实现精确答案、无噪声标注；② 将单一技能的微调与复杂推理分离，证明提升基础技能即可显著提升复杂推理性能；③ 在不同分布下（未见图表类型、人类写作的Benchmark、非图表视觉域）验证了技能迁移的普适性；

**🔧 技术方法**

技术主要包括：LoRA微调、自然语言问答模板、程序化生成的感知/绑定/推理题库、模糊匹配（RapidFuzz）评估；

**📊 数据集**

使用的数据集包括ChartNet（单系列柱状图/饼图）、CLEVR、ChartQA（人类写作的图表问答）等；

**📈 对比分析**

比较方法：在不同技能微调配置（单技能P/G/SR、组合P+G、P+G+SR）下对Bar、Pie、CLEVR、ChartQA的复杂推理准确率进行对比。结果显示，即使不训练任何复杂推理样本，单一技能微调也能提升30%以上的复杂推理性能，尤其在外部分布上表现突出；

**⚠️ 局限性**

局限性：仅覆盖单系列柱状图、饼图和CLEVR场景，缺乏对自然图像的通用性；技能拆分和模板受限于程序化生成，难以直接迁移到真实图像；组合技能配置使用更多数据，难以区分技能提升与数据量提升的影响；未与直接复杂推理监督做对比；

---

## 115. Recent Advances in Deep Learning-Based Drug-Target Binding Affinity Prediction

**arXiv ID:** 2608.13797 | [PDF](https://arxiv.org/pdf/2608.13797v1)

**作者:** Jafin Khan `[一作]` (Prairie View A&M University), Md Hossain Shuvo `[通讯]` (Prairie View A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并比较最新的深度学习方法用于药物–靶标结合亲和力预测，评估不同表示、网络架构、数据集及评价指标，重点分析在 Davis 与 KIBA 基准上的表现和局限。

**💡 创新点**

首次提供统一的评估框架，揭示数据集偏差、冷启动性能下降、缺乏标准化评估和多模态融合等关键瓶颈，指出预训练蛋白语言模型与分子图表示结合的潜力。

**🔧 技术方法**

采用图神经网络（GCN、GAT、GIN、EGNN 等）、卷积网络、循环网络与 transformer 体系结构，并结合 SMILES、分子图、蛋白序列、结构图、预训练嵌入等多种输入表示。

**📊 数据集**

主要使用公开基准：Davis、KIBA、PDBbind（总集、General、Refined）、BindingDB、Metz、CASF‑2016，涵盖 SMILES、FASTA、pKd/pKi 等多种亲和力测量。

**📈 对比分析**

通过 Concordance Index、MSE、R² 等指标对比各方法，发现 GNN‑基准方法在两大基准上均保持高 CI（≈0.89–0.97）与低 MSE，且在冷启动下性能显著下滑，表明模型泛化有限。

**⚠️ 局限性**

局限包括：基准数据集不够多样、冷启动评估不足、预训练蛋白语言模型使用稀缺、模型可解释性差、缺乏独立外部验证和实验验证。

---

## 116. MedClaw: Heuristic Agent Harness for Long-Horizon Surgical Video Reasoning

**arXiv ID:** 2608.14015 | [PDF](https://arxiv.org/pdf/2608.14015v1)

**作者:** Yingying Fan `[一作]` (Beijing Jiaotong University), Yan Wang `[通讯]` (Beijing Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种训练自由的Agent harness，分离推理与感知，利用文本指挥器和冻结的视觉语言子代理在长时间外科视频中进行长时序推理；

**💡 创新点**

核心创新在于：1）通过“梯度无关、奖励门控的启发式技能蒸馏（HSD）”从代理自身失败轨迹中挖掘可重用的检索技能（如“定向重检”），实现数据高效的上下文演化；2）完全冻结模型权重，所有决策通过工具调用在上下文中可审计；3）构建去泄露、医生基准的MedClawBench，专注长时序推理。

**🔧 技术方法**

使用文本只的指挥器（Opus‑4.8/DeepSeek），冻结的视觉语言子代理（Gemini‑3.1‑flash-lite、Gemini‑3.5‑flash、GPT‑5.5），工具集合包括全视频视图、时间段裁剪、帧检查和知识库检索；通过Heuristic Skill Distillation循环在100条标注示例上挖掘并奖励门控技能；

**📊 数据集**

MedClawBench（1,123道问答，含259道自制长神经外科视频、864道SVU公开讲座视频），SVU‑31K公开数据集、神经外科自制数据；

**📈 对比分析**

与多种基线（单次通用VLM如Qwen2.5‑VL、InternVL3、VideoLLaMA3；视频推理模型如Video‑R1、Video‑Chat、Video‑RFT；长视频Agent如LongVT‑7B‑RFT、ReWatch‑R1‑7B；外科专用SurgVidLM）进行对比，使用四维评分（正确性、细节、上下文、时序）平均分衡量。MedClaw在神经外科分段的四维平均分达2.90，较最佳单次VLM Video‑Chat‑R1‑7B的2.14提升0.76；在SVU也表现优异。

**⚠️ 局限性**

局限性：基准样本数量有限，神经外科部分仅来自少量长视频；对不同外科领域的泛化仍待验证；训练自由但依赖手工制定工具和技能库，仍需手工维护；

---

## 117. Self-Supervised Visual On-Policy Distillation

**arXiv ID:** 2608.14144 | [PDF](https://arxiv.org/pdf/2608.14144v1)

**作者:** Yijiang Li `[一作]` (University Of San Diego), Nuno Vasconcelos `[通讯]` (University Of San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Self‑Supervised Visual On‑Policy Distillation（S²VOPD），通过对学生输入做视觉增强（下采样+高斯噪声等），让教师在原图上预测，学生在降质图上生成轨迹，从而在不需要任何特权标签或奖励的前提下实现教师–学生的不对称知识蒸馏。

**💡 创新点**

创新点在于：①不再通过给教师提供更多信息来产生不对称，而是通过从学生视图中剥夺信息来制造不对称；②利用增强视图产生的预测差距作为自监督训练信号；③系统化地搜索并验证四类增强（信息削减、几何、光度、遮挡）以及最佳组合，证明了中等强度的降质最优。

**🔧 技术方法**

核心技术包括：a) 在对话式视觉语言模型中实现 on‑policy 蒸馏；b) 采用指数移动平均（EMA）教师；c) 使用泛化 Jensen‑Shannon Divergence（JSD）作为分布距离度量；d) 设计多种视觉增强算子和组合策略；e) 在模型内部使用 top‑k 采样和重归一化以稳定训练。

**📊 数据集**

训练使用 FineVision（约 12k 图像+问题）作为无特权数据；评测覆盖六大细粒度感知基准（V*Bench、ZoomBench、HR‑Bench 4K/8K、MME‑RealWorld、MME‑RealWorld‑CN）和三大数学推理基准（MathVista、MathVerse、MathVision）。此外，亦在 Vision‑OPD‑6K 上进行 ablation 对比。

**📈 对比分析**

与多种基线对比：①开源模型（Qwen、MiniCPM、MiMo‑VL‑RL 等），②专有模型（GPT‑5 系列、Gemini‑3 系列），③具特权信息的 OPD（ZwZ、Vision‑OPD）以及自奖励 RL（TTRL、Intuitor、RENT）。在感知任务上，S²VOPD 将 Qwen3.5‑4B 从 70.7% 提升至 77.4%（平均），超过所有开源模型，接近 397B 大模型，且超过 GPT‑5‑4 及 Gemini‑3‑Flash；在数学推理任务上，提升约 5–7%，与自奖励 RL 方法持平或更优。

**⚠️ 局限性**

局限性：①需要精心设计增强策略，过度降质会移除关键信息导致监督失效；②目前仅在视觉语言推理和细粒度感知任务上验证，未必适用于所有视觉任务；③对算力有一定需求（需生成多条轨迹并进行 EMA 更新）；④若学生视图完全不可解答，教师与学生间的差距虽大但信息不具指导性，影响训练效果。

---

## 118. When Lexical Change Misleads: Rethinking Dynamic Topic Model Evaluation with Traditional and LLM-Based Metrics

**arXiv ID:** 2608.13835 | [PDF](https://arxiv.org/pdf/2608.13835v1)

**作者:** Charu Karakkaparambil James `[一作]` `[通讯]` (RPTU University Kaiserslautern-Landau), Charu Karakkaparambil James (RPTU University Kaiserslautern-Landau)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究动态主题模型评估，检验传统连贯度与基于大语言模型的语义相似度在不同词汇变化水平下的有效性。

**💡 创新点**

提出词汇变化感知的多视角评估框架，强调传统连贯度、LLM语义相似度与词汇变化三者互补的重要性。

**🔧 技术方法**

采用传统NPMI连贯度指标与OpenAI GPT-5.5生成的语义相似度评估。

**📊 数据集**

使用NYT、DBLP、arXiv三个时间序列文本语料库，共120条主题轨迹。

**📈 对比分析**

通过与三名人工标注者的评估对齐，发现传统连贯度在不同模型/数据集间相关性不一，LLM语义相似度在CoNTM上表现良好但DLDA不稳定；两者互补，为动态主题提供更全面的评估。

**⚠️ 局限性**

样本量有限，词汇变化分类每组仅含6-7个主题；LLM评估受模型与提示敏感；不同指标对应的人工问题不同，无法直接进行指标间的直接比较。

---

## 119. Contrastive Learning for Interpretable Anomaly Detection at Collider Experiments

**arXiv ID:** 2608.13652 | [PDF](https://arxiv.org/pdf/2608.13652v1)

**作者:** Haoyi Jia `[一作]` (Stanford University), Julia Gonski `[通讯]` (SLAC National Accelerator Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种两阶段的异常检测框架 ORCA：先用监督对比学习构建以物理过程为导向的嵌入空间，再在该空间上训练无监督自编码器生成事件级异常分数。

**💡 创新点**

创新点在于将对比学习与异常检测耦合，生成可解释的高维嵌入；利用该嵌入做多维模板拟合，能够把异常事件分解为已知物理过程的成分，提供比传统单维观测更细粒度的解释。

**🔧 技术方法**

技术包括：Transformer‑based 嵌入器（对象级网络+多头注意力+投影头）、InfoNCE 对比损失+方差正则化、标准自编码器、PCA+ICA 降维、混合Poisson 似然拟合（Composite Likelihood）和 sandwich 协方差估计。

**📊 数据集**

使用 COLLIDE‑2V 公开模拟数据：约 7.5×10⁸ 事件，13.6 TeV 质能，48 个 SM 过程（约 14 类），每事件用 110 维低级特征（10 最高 jet、4 e、4 μ、4 γ、MET）构成。

**📈 对比分析**

与直接在原始特征上训练的基线自编码器相比，ORCA 在 AUC 及固定 FPR=10⁻⁴ 的 TPR 上平均提升 20–40%，在多种信号（Higgs、Di‑Higgs、三向量玻色子等）上表现突出；模板拟合在注入信号测试中恢复率接近 1，优于单维异常分数的拟合。

**⚠️ 局限性**

局限性包括：模板库有限，无法覆盖所有潜在新物理；拟合假设不同维度独立，导致置信区间可能低估；对模拟准确性的依赖；对极少见或未训练过的过程可能失效；两阶段训练需要大规模数据和计算资源。

---

## 120. Voxel-based 3D Facies Segmentation from Seismic Data: A Comparative Study

**arXiv ID:** 2608.14058 | [PDF](https://arxiv.org/pdf/2608.14058v1)

**作者:** Duc-Thanh Pham `[一作]` (FPT Software AI Center), Van Nguyen `[通讯]` (FPT Software AI Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对三种 voxel‑based 3D 深度学习模型（3D‑UNet、UNETR、SegMamba）在 Netherlands F3 与 Parihaka 两个公开地震体数据集上进行 facies 分割实验，提供可复现的基准。

**💡 创新点**

首次构建统一的 3D facies 分割基准，系统评估卷积、Transformer 与 state‑space 模型在低标注量地震数据上的性能，并揭示卷积模型的优势。

**🔧 技术方法**

使用 3D‑UNet、UNETR（Transformer 编码器+卷积解码器）和 SegMamba（基于 Mamba 状态空间 + GSC）网络；采用 Dice 损失、五折交叉验证、128³ 子体积采样等训练与评估策略。

**📊 数据集**

使用 Netherlands F3 Block 和 Parihaka 3D 调查的完整地震体，分别含 6 个 facies 类别。

**📈 对比分析**

通过五折交叉验证计算每类 Dice 分数；3D‑UNet 在 F3（87.69%）和 Parihaka（75.01%）的平均 Dice 分数最高；UNETR 与 SegMamba 分别略低，但 3D‑UNet 在参数、内存和 GFLOPs 上最优。

**⚠️ 局限性**

局限性在于仅针对标注有限的数据集；Transformer/SegMamba 在低样本和高结构复杂度场景下表现不佳；未探索更大子体积、更丰富的数据增强或跨域迁移。

---

## 121. Engineering Signals of Human-AI Collaboration in the Agentic Coding Era: A Longitudinal Analysis of 33,228 Pull Requests from vLLM and SGLang with Implications for Biomedical AI Agents and Bioinformatics Pipeline Developmen

**arXiv ID:** 2608.13884 | [PDF](https://arxiv.org/pdf/2608.13884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 122. How retriever redundancy and diversity impact RAG effectiveness

**arXiv ID:** 2608.13956 | [PDF](https://arxiv.org/pdf/2608.13956v1)

**作者:** Jonathan J Ross `[一作]` (University of Queensland), Guido Zuccon `[通讯]` (University of Queensland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究检索器冗余与多样性对检索增强生成（RAG）模型回答准确性的影响，设计受控实验比较重复、LLM改写和多样化检索结果对生成器的影响。

**💡 创新点**

通过三种受控检索场景（Duplicate、Paraphrased、Diverse）并精确控制答案出现方式，系统验证多样化检索显著提升回答正确率，揭示生成器更偏好多元文档来源。

**🔧 技术方法**

采用检索增强生成框架、LLM生成器、实验式检索设置以及手工构造的多样化检索集合。

**📊 数据集**

使用 FictionalQA 合成问答数据集，保证 LLM 先验知识无法直接回答，答案必须来自检索文档。

**📈 对比分析**

通过比较三种检索策略下的答案正确率，发现多样化检索可提升 17%–47% 的正确率；重复与改写检索对正确率提升不显著。

**⚠️ 局限性**

仅在合成数据集上评估，未验证在真实知识库或多模型场景下的效果；未探讨检索方法多样化对不同 LLM 规模的影响等。

---

## 123. TeachMateGPT: A Multi-Agent Knowledge-Grounded Framework for Pedagogical Assessment Generation from Science Curriculum Materials

**arXiv ID:** 2608.13708 | [PDF](https://arxiv.org/pdf/2608.13708v1)

**作者:** Fatema Tuj Johora Faria `[一作]` (Ahsanullah University of Science and Technology), Jubayer Al Mahmud `[通讯]` (Jashore University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了一套多智能体框架TeachMateGPT，用于生成符合Bangla NCTB第8册科学教材的评估题目，并生成包含检索证据与验证报告的可审计产出；同时利用该框架构建了题目数据集NCTB‑SciGen8。

**💡 创新点**

创新点包括：① COPE——基于教材层次与图结构的多分辨率知识库；② 阶段化、fail‑closed的多智能体流水线，实现检索后才生成；③ SAVER——源归属性验证与证据评分，标记可疑题目；④ 通过流水线自动生成并教师审阅的NCTB‑SciGen8评估数据集。

**🔧 技术方法**

采用的技术有：检索增强生成（RAG）+密集+词典混合检索；图感知多分辨率知识库；分阶段的智能体路由、检索、生成与验证；上下文注入与冲突消解；基于检索证据的源归属性验证；教师‑循环交互。

**📊 数据集**

使用的数据集为Bangla NCTB第8册科学教材文本，经过COPE构建索引；并通过TeachMateGPT自动生成的198道题（143道多项选择、55道创意题）组成NCTB‑SciGen8数据集。

**📈 对比分析**

与vanilla RAG、RAPTOR、GraphRAG、CRAG、Adaptive RAG等基线以及五个组件消融模型进行比较。评价指标包括faithfulness、answer relevancy、context precision/recall以及教师主观评估的Pedagogical Alignment、Stimulus Realism、Linguistic Fluency、Overall Utility。TeachMateGPT在所有指标上均优于基线，faithfulness从0.68提升至0.96，answer relevancy从0.60提升至0.89，context precision/recall从0.54/0.58提升至0.92/0.91；教师评估亦表现最高。

**⚠️ 局限性**

局限性包括：仅针对Bangla NCTB第8册科学教材，难以直接迁移到其他学科、年级或语言；OCR/文本转换误差导致检索召回受限；仅支持文本模式，无法处理图示题目；fail‑closed检索可能过度拒绝，影响召回率；缺乏心理测量难度校准；评估样本与评审者规模有限；仍需教师最终审核，系统不适合全自动考试生成。

---

## 124. Optimal Power Allocation and AI Receiver Design for Superimposed DMRS and Data Transmission

**arXiv ID:** 2608.13809 | [PDF](https://arxiv.org/pdf/2608.13809v1)

**作者:** Sha Hu `[一作]` (Huawei Technologies Sweden AB), Zhongwang Fu `[通讯]` (Huawei Technologies Sweden AB)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了基于Transformer的AI-ICED接收机，对SI-DMRS MIMO-OFDM系统中的信道估计与检测进行联合处理，并提出了功率分配与MSE迭代的理论分析框架。

**💡 创新点**

创新点在于：① 将VWN-TPA-MoE Transformer与物理引导特征构造PGFC结合，实现联合CE+MD的迭代网络；② 推导了迭代CE/MD的MSE闭式平衡，优化SI-DMRS功率分配；③ 在SI-DMRS下，AI-ICED接近最优MLD性能且吞吐量提升。

**🔧 技术方法**

采用的技术包括Transformer Encoder、VWN、Tensor Product Attention (TPA)、Mixture-of-Experts (MoE)、物理引导特征构造 (PGFC)、MIESM映射、LMMSE估计、MLD对照、深度监督训练和多阶段解卷积。

**📊 数据集**

使用合成OFDM-MIMO数据集：2×2 MIMO 800k帧，4×4 MIMO 200k帧，通道采用ETU70多径模型，按81%/9%/10%划分为训练/验证/测试。

**📈 对比分析**

通过与传统LMMSE、MLD、5G-NR基线（非叠加DMRS）以及给定CSI的基线比较。AI-ICED在2-3次迭代内收敛，BLER接近MLD+genie CSI，吞吐量比5G-NR提升约20%，在中高SNR下优于LMMSE。

**⚠️ 局限性**

局限性包括：CE误差比基线高2-4 dB；推理复杂度较高，需多阶段Transformer；对不同SNR/码率需多模型；功率分配可能导致PAPR升高；理论假设高斯独立，实际通道可能产生偏差。

---

## 125. Batch-wise Adaptive Pruning: Periodic Neuron Activation-Aware Weight Pruning for Language Reasoning Model

**arXiv ID:** 2608.14003 | [PDF](https://arxiv.org/pdf/2608.14003v1)

**作者:** Yongmin Kim `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的批量自适应剪枝方法，专门解决大规模推理模型在批量推理时的高计算成本。

**💡 创新点**

通过周期性 top‑k 选择和激活记忆机制，克服了传统阈值剪枝在批量聚合后出现的分布漂移和稀疏率漂移问题。

**🔧 技术方法**

采用元素级最大聚合、定期 top‑k 选取、激活记忆累积、结构化 MLP 剪枝，以及每个更新周期仅一次的选择策略。

**📊 数据集**

在多项推理基准（TinyGSM8K、MATH500、MINERVA Math、AMC23、GPQA‑DIAMOND）上进行评估。

**📈 对比分析**

与 Wanda、Griffin、TEAL 等无训练剪枝基线对比，在批量推理、50% 目标稀疏度下平均准确率提升 39.7 点，速度提升 1.40×。

**⚠️ 局限性**

仅针对结构化稀疏层（MLP）有效，对动态阈值适配性有限，且在极高批量或非推理任务中提升有限。

---

## 126. Knowledge-Data-Dual-Driven Reinforcement Learning for Autonomous Vehicle Control in Mixed Traffic

**arXiv ID:** 2608.13878 | [PDF](https://arxiv.org/pdf/2608.13878v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 127. PROVE: Training-Free Prompt Recovery using Verifiable Evidence

**arXiv ID:** 2608.13671 | [PDF](https://arxiv.org/pdf/2608.13671v1)

**作者:** Rupayan Mallick `[一作]` (Georgetown University), Sarah Adel Bargal `[通讯]` (Georgetown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练和优化均无需求的黑盒 prompt 反演攻击方法 PROVE，用以从目标图像恢复生成该图像的文本提示。

**💡 创新点**

创新点在于：1）基于可验证的场景描述构造 prompt，而非直接优化 token 序列；2）使用冻结的 VLM 与开放词汇检测器生成原子声明，再通过 CLIP 边界验证每条声明的真实性；3）通过几何事实提取实现计数与空间关系的确定性推断；4）所有步骤无生成器参与，避免过拟合与评估循环。

**🔧 技术方法**

核心技术包括冻结的 Vision‑Language 模型 (如 CLIP、BLIP)、开放词汇目标检测、CLIP 边界门控验证、几何事实提取、确定性串行组合。

**📊 数据集**

在 MS‑COCO、Flickr30K 以及 Lexica 三个公开数据集上评估，分别代表真实照片、日常场景与生成艺术作品。

**📈 对比分析**

与七种主流方法（优化型 PH2P/STEPs/VGD、captioning 型 BLIP/CLIP‑IG/强 VLM‑expert、RL 型 PromptMiner）对比，PROVE 在 DINO、LPIPS 与 CLIP 三种图像相似度指标上均取得最佳或最接近最佳的数值，并在防御实验中表现出较高的鲁棒性。

**⚠️ 局限性**

局限性包括：① 对复杂视觉细节或极端图像变形（如强水印、对抗扰动）仍可能失去部分精细信息；② 依赖 VLM 与检测器的性能，若其识别失效则会影响结果；③ 目前仅针对单图像输入，未探讨多图像或视频序列的反演；④ 对生成模型的多样性仍存在一定的泛化挑战。

---

## 128. IterCOMP: Reasoning-aware Adaptive Prompt Compression for Multi-hop Question Answering

**arXiv ID:** 2608.13588 | [PDF](https://arxiv.org/pdf/2608.13588v1)

**作者:** JungMin Yun `[一作]`, YoungBin Kim `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练、统一的多跳问答提示压缩框架 IterCOMP，利用 LLM 的答复可答性判断与缺失信息检索进行迭代压缩，构建紧凑且信息丰富的提示。

**💡 创新点**

创新点在于把多跳推理嵌入压缩循环：①通过 LLM 判断当前证据是否足够，②若不足生成针对性追问以填补信息缺口；同时采用双重语义-词汇相关性评分，且实现完全训练‑free、模型无关。

**🔧 技术方法**

核心技术包括：句子级文档分解；基于预训练 bge‑m3 的语义向量与词汇权重构建双重相关性分数；LLM 作为二分类器进行答复可答性判断与缺失信息识别；迭代压缩循环与早停机制。

**📊 数据集**

在 MuSiQue、2WikiMultiHopQA 与 HotpotQA 三大多跳 QA 基准上进行实验。

**📈 对比分析**

与 LLMLingua、LongLLMLingua、R2C、RECOMP 等现有硬压缩/抽取式方法对比，IterCOMP 在 EM/F1 上均位居榜首（例如 MuSiQue F1 从 19.92 提升至 27.36），压缩比提升 7×，成本降幅 75–79% 并实现 1.13–2.01× 的推理速度提升。

**⚠️ 局限性**

局限性：依赖 LLM 的推理与缺失信息检测，误判可能导致过早终止或冗余迭代；迭代过程相对耗时；性能对阈值 k 与迭代上限敏感；词汇-语义评分机制较为简单，未引入更复杂的匹配策略。

---

## 129. P2Skill: Privacy Preserving Skill Distillation for Cloud-Local LLM Inference Systems

**arXiv ID:** 2608.14094 | [PDF](https://arxiv.org/pdf/2608.14094v1)

**作者:** Myunghoon Ryu `[一作]` (Korea University), Jong-Kook Kim `[通讯]` (Korea University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 P2Skill，利用本地小型语言模型（SLM）执行四个提示技能（分解、PII-aware 路由、改写、重构），并通过云端 LLM 迭代细化这些技能，从而在不更新 SLM 权重或训练辅助检测器的情况下实现云-本地 LLM 推理的隐私保护。

**💡 创新点**

创新点在于：①仅通过提示（Prompt）蒸馏出四个可复用的技能，避免了模型微调或专用检测器的需求；②闭环细化循环由云 LLM 主导，能自动纠正样本特定错误；③在冻结的 SLM 上即可实现对一般 PII 的识别与处理，提升了隐私保障与推理质量的平衡。

**🔧 技术方法**

使用的技术包括：生成式提示工程、Prompt-based Skill Distillation、云 LLM 指导的迭代细化循环、Deterministic Identifier Matcher、PII-aware Routing、Paraphrasing Skill、Reconstruction Skill 等。

**📊 数据集**

使用的数据集：PRISM 160-提示基准（包含 Medical、Banking、Tourism、General 四个领域），以及 70 个用于技能细化的离散提示（PRISM、AI4Privacy PII Masking、NVIDIA Nemotron-Personas、HotpotQA）。

**📈 对比分析**

通过与 Uniform LDP、Selective LDP 两种基线在四个不同 SLM（Gemma4:e2B、Qwen2.5:1.5B、Qwen3.5:2B、Llama3.2:3B）上进行对比，结果显示 P2Skill 在 PP（隐私保留）上实现 1.0 或接近 1.0，平均 PQ（隐私-质量）比 Selective LDP 提升 1.69 倍，比 Uniform LDP 提升 3.66 倍；在 Medical、Banking 等敏感域中，PII 泄露几乎为零。

**⚠️ 局限性**

局限性包括：①PII 泄露主要来自分解阶段未检测到的实体；②整体推理质量仍低于纯云端 LLM，尤其在需要精细信息的领域；③技能需针对不同 SLM 进行手工定制，部署成本相对较高；④评估依赖 LLM 判定，缺乏真实人类主观评估。

---

## 130. XAI-Guided Conservative Decentralized Execution for Offline Multi-Agent Network Slicing

**arXiv ID:** 2608.13982 | [PDF](https://arxiv.org/pdf/2608.13982v1)

**作者:** Eslam Eldeeb `[一作]` (University of Oulu), Merouane Debbah `[通讯]` (Khalifa University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究6G网络切片资源分配，提出XAI‑guided Conservative Decentralized Execution (X-CODE) 的离线多智能体强化学习框架，旨在实现零资源冲突、低延迟且无需执行时通信。

**💡 创新点**

创新点在于：1）将保守 Q‑学习（CQL）与中心化训练-去中心化执行（CTDE）结合，解决离线多智能体的过估计问题；2）利用 SHAP 解释性特征重要性对离线奖励进行重标，指导学习过程；3）在离线环境下通过奖励重标实现隐式协作，部署时完全无需跨智能体通信。

**🔧 技术方法**

技术手段包括：离线 CTDE、保守 Q‑学习（CQL）、价值分解网络（VDN）、SHAP 解释、奖励重标、随机走访数据集收集、基于深度 Q‑网络（DQN）与信息瓶颈的在线基线对比。

**📊 数据集**

数据集：随机走访行为策略收集的 10,000 条离线交互样本，模拟三类切片（eMBB、mMTC、URLLC）在共享边缘服务器上的 CPU 资源分配场景。

**📈 对比分析**

与在线 DQN (Comm+IB)、DQN (No Comm)、X‑CQL、CFCQL、QMIX、VDN 等基线比较，X-CODE 在延迟累积分布函数上表现相当或更优，零资源冲突；通信开销降至 0；在考虑通信延迟模型下推理时间平均降低 88%。

**⚠️ 局限性**

局限性：依赖离线数据覆盖度，若数据质量不足易导致性能退化；奖励重标并不保证保持原环境最优性，需手动调节 μ；目前仅在 3 切片小规模环境验证，扩展到更大规模切片及更复杂网络拓扑仍待验证。

---

## 131. Rewrite Once, Validate Anywhere: Producing OWL-Aware SHACL Constraints (Extended Version)

**arXiv ID:** 2608.14104 | [PDF](https://arxiv.org/pdf/2608.14104v1)

**作者:** Anouk Oudshoorn `[一作]` (Technische Universitaet Wien), Dörthe Arndt `[通讯]` (Technische Universitaet Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于 OWL EL- 片段的 SHACL 约束重写方法，使得验证过程无需单独推理步骤，将本体知识直接内在化到形状约束中。

**💡 创新点**

创新点在于：①提出数据独立的重写算法，将 OWL 术语转化为 SHACL 约束；②在保持可控复杂度的前提下覆盖命名个体、角色链、限定存在约束等表达式；③提供可复用的重写后形状图。

**🔧 技术方法**

使用技术包括 OWL EL- 片段推理、正则路径表达式替换、形状约束重写、角色层次展开、目标重写，并实现为基于 rdflib.js 的 JavaScript 工具。

**📊 数据集**

实验数据为 1500 个随机生成的基准实例，涵盖 15 个难度级别，包含 TBox、SHACL 形状和数据图，托管于 GitHub 与 Figshare。

**📈 对比分析**

与传统的“推理+验证”以及内置推理的 SHACL 验证器比较，重写方法在 1200 秒内完成所有实例验证，平均重写时间仅 2.53 秒，速度快数十倍且覆盖率始终为 100%。

**⚠️ 局限性**

局限性包括：仅适用于 OWL EL- 片段（不支持右侧存在约束和命名个体的完整推理）；重写后形状可能极为复杂，导致部分验证器报告难以阅读；当前不支持递归 SHACL 形状。

---

## 132. A Calibrated Test of Internal Action Maps: State Signals Without Global Affine Closure

**arXiv ID:** 2608.13626 | [PDF](https://arxiv.org/pdf/2608.13626v1)

**作者:** Dekun Yang `[一作]` `[通讯]` (Zhejiang University), Dekun Yang (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对语言模型内部的动作映射进行校准性测试，验证状态信号在未提供源信息时的可重用性。

**💡 创新点**

提出基于证据格子和闭合门的测试框架，区分状态可用性、因果可用性与全局仿射闭合等维度，并在有限世界上校准已知代数。

**🔧 技术方法**

使用Transformer激活的仿射映射拟合、逆推、顺序与可交换性检验、结构化曲率与域共轭、以及因果干预的匹配配对等技术。

**📊 数据集**

在Alchemy语言环境、S5置换群和Z11^2格点这三组数据集上进行实验。

**📈 对比分析**

与已知代数控制、随机分层、跨域交叉拟合等对照，发现Qwen3-4B在h28层的一步映射可被重构但无法满足两步闭合，早期层仅满足几何一致性。

**⚠️ 局限性**

结果仅覆盖单一预训练模型、有限层级、特定检查点，未检验跨模型、跨层、KV缓存或非线性路径等，因而局限性较大。

---

## 133. AI Evaluation Should Work With Humans

**arXiv ID:** 2608.13577 | [PDF](https://arxiv.org/pdf/2608.13577v1)

**作者:** Jan Kulveit `[一作]` (Charles University), Raymond Douglas `[通讯]` (Charles University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文主张将 AI 评估焦点从“是否能替代人类”转向“人类与 AI 组成团队能否更高效完成任务”，并提出一套团队评估框架、指标体系以及分层评估策略。

**💡 创新点**

创新点在于：①提出人机协作评估的新范式；②构建多维度指标（任务性能、用户体验、协作流畅度等）；③将评估分为独立基准、代理评估与真实参与评估三个层级，兼顾效率与真实性；④强调评估应促进社会福利、减少经济不平等。

**🔧 技术方法**

并未使用特定算法或技术，而是综合现有评估方法、HCI 指标与机器学习可解释性工具，构思新的评估流程与平台（如“Human‑AI Interaction Gyms”）。

**📊 数据集**

文中未使用已有数据集；作者建议收集并公开人机协作日志、失败案例等数据，构建新的协作评估数据集。

**📈 对比分析**

对比方式是通过实验设置人类单独、AI 单独、同步人机协作、异步人机协作四种条件，测量任务完成质量、效率、信任校准、学习提升等指标；但论文仅提出框架，并未给出具体实验结果或性能数值。

**⚠️ 局限性**

局限性包括：评估成本高、需 IRB 审批与人力协调；人类参与带来的变异性与可重复性挑战；现有 HCI 指标在大规模排行榜场景下的验证有限；缺乏实证数据支撑新指标与分层评估的有效性。

---

## 134. Proxy-Validated LLM UX Micro-Simulations: An Artifact-First Protocol for Early-Stage Decision Support

**arXiv ID:** 2608.13563 | [PDF](https://arxiv.org/pdf/2608.13563v1)

**作者:** Alexandre Cristovão Maiorano `[一作]` `[通讯]`, Alexandre Cristovão Maiorano

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了可复现的UX微仿真管线，利用LLM生成结构化的用户体验反馈，并通过代理验证协议与公开数据集进行分布对齐；

**💡 创新点**

提出了轻量级的代理验证协议，包括两种重叠度量（Top‑k Jaccard 与加权Jaccard），并引入了score‑then‑select评判器以降低LLM评判的锚定偏差；

**🔧 技术方法**

使用大型语言模型（如GPT‑4.1）进行情境模拟，BGE‑M3多语言嵌入做相似度匹配，以及TF‑IDF、词汇匹配等基线；

**📊 数据集**

验证数据集包括：英文/印尼语App Store评论、支持推文、低评分Amazon评论、标注的Amazon评论、GitHub OSS issue（Grafana、Prometheus等）等；

**📈 对比分析**

通过对比Embedding、Lexical和TF‑IDF三种对齐方法，在加权Jaccard上Embedding通常表现最佳；Top‑k Jaccard在k接近词表大小时会过度提升，导致误导；Bootstrap置信区间表明Lexical和TF‑IDF在大样本下稳定，而Embedding在当前子样本规模下的加权Jaccard不稳定；

**⚠️ 局限性**

局限包括代理数据的噪声与非任务相关性、LLM生成的偏差与不确定性、嵌入对齐的子样本敏感性、语料停用词对真分母估计的影响，以及对特定领域词汇的覆盖不足；

---

## 135. HiCo-GS: Hierarchical Context Aggregation and Geometric Consistency for Octree Gaussian Splatting

**arXiv ID:** 2608.14136 | [PDF](https://arxiv.org/pdf/2608.14136v1)

**作者:** Wei Zhang `[一作]` (Northwestern Polytechnical University), Qi Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HiCo-GS 框架，改进城市规模场景的 3D 高斯光栅化（Octree Gaussian Splatting），实现高保真渲染与更清晰的几何重建。

**💡 创新点**

创新点在于（1）跨层上下文聚合（CLCA）通过八叉树的空间包含关系实现层间双向信息交流，解决先前独立优化导致的颜色漂移和细节模糊；（2）深度-法线几何一致性（DNGC）利用渲染法线与深度派生法线的自洽约束，结合边缘感知平滑，抑制浮动几何并提升平面连续性。

**🔧 技术方法**

技术核心包括多级八叉树 anchor 结构、上下文聚合 MLP（含残差）、深度-法线一致性损失、边缘加权平滑项，以及渐进式热身调度；实现基于 CityGS‑X 代码库，使用 4×RTX‑4090 并行训练。

**📊 数据集**

数据集：Mill19、UrbanScene3D、MatrixCity（合成）以及新建的 China‑Pagoda（8座古塔，每座 1,200+ 张图），用于评估在现代平面建筑和极端非平面建筑上的重建质量。

**📈 对比分析**

与 CityGS‑X、Momentum‑GS、CityGaussianV2 等现有方法对比，HiCo‑GS 在 4 个基准上均取得最高 SSIM、PSNR 与最低 LPIPS；在中国古塔数据集上，平均提升 PSNR 超 2 dB，排名 22/24 场景第一，显著优于基线。

**⚠️ 局限性**

局限性包括：对极端细节仍需要足够高的 anchor 级别，训练仍需大规模 GPU 资源；当前方法主要针对城市尺度平面建筑，复杂曲面和纹理重复区域仍可能出现微小失真；对全景或动态场景的泛化尚未验证。

---

## 136. Limitations of Synthetic Data Generation in Specialized Data-Scarce Domains

**arXiv ID:** 2608.13729 | [PDF](https://arxiv.org/pdf/2608.13729v1)

**作者:** Edward Zhang `[一作]` (University of Pennsylvania), Eric Eaton `[通讯]` (University of Pennsylvania)

**通讯引用:** 2588 | [OpenAlex ID](https://openalex.org/A5020691490)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在医学创伤识别与焊接质量检测这两类极度稀缺、高变异的视觉分类任务中，对多种生成式数据增强方法（如 StyleGAN、Stable Diffusion、DreamBooth、Img2Img、IP-Adapter、GPT 图像生成）与传统 AugMix 的表现进行系统比较。

**💡 创新点**

创新点在于：①首次在多任务、极度稀缺的数据环境下对生成式与非生成式数据增强进行大规模实验；②使用特征空间 MMD 与 t‑SNE 量化生成样本与真实分布的贴合度，并将其与下游分类效果关联；③探究人工挑选的高质量生成样本对模型的补充效应。

**🔧 技术方法**

主要技术包括：扩散模型（Stable Diffusion、DreamBooth）、GAN（StyleGAN2）、图像对图像生成（Img2Img、IP‑Adapter）、文本‑图像生成（GPT‑Image‑1.5）、传统增强（AugMix、图像复制），以及冻结 DINOv2 预训练特征与 MLP 头的分类器。

**📊 数据集**

使用两组数据集：①PRONTO 伤者多视角模拟创伤数据（362 张经人工筛选的高质量图像，划分为5个二/三分类任务）；②Intel 机器人焊接数据集（约 4,000 张焊点图像，二分类：良好 vs 缺陷）。

**📈 对比分析**

对比方法包括：无增强、图像复制、AugMix、StyleGAN、Stable Diffusion、DreamBooth、Img2Img、IP‑Adapter、GPT 图像生成等。评估指标为宏平均召回率（macro‑recall）。实验结果显示，所有生成式方法均未能持续优于 AugMix，且部分生成方法在某些任务甚至表现更差；最佳增益来自 AugMix 或极少量的人工挑选生成样本。

**⚠️ 局限性**

局限性包括：①实验仅覆盖创伤与焊接两类视觉分类，未必能推广至所有稀缺域；②使用的生成模型配置有限，缺乏针对性调参；③评估依赖 DINOv2 特征，可能与真实语义距离不完全一致；④未充分探索不同分类器结构、度量指标或真实部署环境的鲁棒性。

---

## 137. ProFocus: Interpreting Affective Experience in Artistic Images with Progressive Visual Focusing

**arXiv ID:** 2608.13974 | [PDF](https://arxiv.org/pdf/2608.13974v1)

**作者:** Zhiyan Zhang `[一作]` (University of Science and Technology of China), Xun Yang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ProFocus框架，用以通过分层艺术评论和渐进式提示融合来解释艺术图像的情感体验

**💡 创新点**

创新点在于将视觉表征分解为三个层次（氛围、叙事主体、具体细节），并按粗到细的顺序逐步注入视觉特征，模拟人类审美认知流程

**🔧 技术方法**

结合多模态大型语言模型（LLaVA-1.6）生成层次化文本先验，使用多头交叉注意力进行渐进式融合，再由GPT‑2解码情感与解释

**📊 数据集**

使用ArtEmis v1.0和v2.0两大艺术情感数据集进行训练与评估

**📈 对比分析**

与SEVLM、NLX‑GPT2、M2、SAT等SOTA方法对比，ProFocus在情感识别准确率、BLEU、METEOR、ROUGE‑L等指标上均实现显著提升，达成新的最佳表现

**⚠️ 局限性**

主要局限是仍依赖预训练的CLIP视觉编码器和LLaVA生成的文本先验，无法完全避免在极具抽象性或隐喻强的作品中产生误判或解释缺乏细节

---

## 138. ARC: Fair Relative Advantage Comparison in Open-Ended Real-World Interaction

**arXiv ID:** 2608.13622 | [PDF](https://arxiv.org/pdf/2608.13622v1)

**作者:** Yongqi Tong `[一作]` (Ant International), Xin Zhang `[通讯]` (Ant International)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ARC（Advantage Regularization via Conditioning）策略条件化优势估计与通道分离交互框架Inter，解决开放式交互中多策略导致的优势偏差问题，并构建86K策略标注数据集；

**💡 创新点**

核心创新在于通过给训练样本注入策略指令，将优势估计限定在同一策略族内，消除跨策略奖励偏差；同时设计交互通道分离，显式区分用户可见交互与隐藏推理/工具调用，便于观察多样策略；

**🔧 技术方法**

使用策略条件化的组内优势估计、熵正则化的多通道生成、GRPO/PPO/DAPO等基准策略梯度方法、Qwen3-8B/4B大模型、奖励模型评估、训练时策略指令的持续使用与校准；

**📊 数据集**

-86K数据集（57.9K SFT + 28.9K RL），来源于真实部署交互、公共工具使用、推理、问答基准（ToolMind、Musique、KnightsAndKnaves、Opus Distilled）及合成轨迹；

**📈 对比分析**

与无思考、思考、PPO、DAPO、GRPO等基准在τ-bench、τ^2-bench、AIME、IFBench、ArenaHard上对比。ARC在工具使用基准上平均提升5.37点（GRPO），整体在多策略环境中显著优于基线；在小模型4B和策略缩放实验中亦保持优势；

**⚠️ 局限性**

局限性：①策略抽象过于粗糙，缺乏细粒度语义；②主要验证于工具使用场景，泛化能力待进一步测试；③依赖手工/模型注释的策略标签，可能存在偏差；④理论分析仅基于估计器方差，未给出完整收敛保证。

---

## 139. When Does More Correct Data Hurt? Insertion-Stability and the Limits of Dimension-Based Theory

**arXiv ID:** 2608.14020 | [PDF](https://arxiv.org/pdf/2608.14020v1)

**作者:** Joseph Sankoorikal Johny `[一作]` `[通讯]` (Independent Researcher), Joseph Sankoorikal Johny (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对抗性数据注入（Monotone adversary）模型下的学习器进行理论分析，提出插入稳定性定义，并证明闭包算法在交叉闭合集上对任意多插入保持清晰PAC率。

**💡 创新点**

引入插入稳定性概念，利用该性质证明闭包算法在所有交叉闭合集上对多次正确标注插入免疫，并揭示VC和Littlestone维度无法决定此类问题的学习速率。

**🔧 技术方法**

采用插入稳定性引理、样本压缩、Hanneke的压缩上界、以及对Monotone adversary几何与概率特性的分析，构建理论证明框架。

**📊 数据集**

本文为理论工作，未使用具体实验数据集；所有结论均基于数学证明与计算实验验证（如对特定类的穷举检验）。

**📈 对比分析**

通过理论上界与下界对比，证明交叉闭合集类的误差率为Θ(1/n)，而某些非交叉闭合集类（如Mehrotra构造类）误差率为Θ(log n / n)，显示闭包算法在合适类上实现最优率。

**⚠️ 局限性**

仅给出充分条件而非必要条件；缺乏可计算的插入稳定性量化指标；对非交叉闭合集类的完整下界与上界匹配仍是未解的开放问题。

---

## 140. Bootstrapping Niche Multilingual Code Translation via Reinforcement Learning with Execution-Based Verifiable Supervision

**arXiv ID:** 2608.13854 | [PDF](https://arxiv.org/pdf/2608.13854v1)

**作者:** Kouki Yuki `[一作]` (National Institute of Technology), Yutaka Matsuo `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 NicheCodeTranslator，利用从 Python 种子扩展到多语言、执行验证的数据集，训练奖励模型并使用 GRPO 进行多语言代码翻译的强化学习，最终覆盖 25 种语言、600 个翻译方向，并提出 HumanEval‑X++ 作为执行式多语言评测基准。

**💡 创新点**

创新点包括：① 通过规则化的 transpilation 将可执行的 Python 程序扩展为多语言验证数据；② 用奖励模型代替每种语言沙箱，实现 600 方向的可扩展训练；③ 将 GRPO 与奖励模型结合，提升翻译质量；④ 设计 HumanEval‑X++ 执行式评测，使大多数语言可在统一基准上比较。

**🔧 技术方法**

使用的技术包括：大模型（Qwen‑3.5 系列）生成、规则化 transpilation、sandboxed 执行验证、基于 cross‑encoder 的奖励模型（Bradley‑Terry 训练）、GRPO（group‑relative policy optimization）、LoRA 微调、以及多语言代码执行环境（MultiPL‑E、CodeScope）。

**📊 数据集**

使用的数据集有：KodCode（Python 程序种子）、MultiPL‑E（规则化多语言转换）、HumanEval‑X++（基于 HumanEval 的多语言评测集）、CodeScope（多语言翻译评测集）。

**📈 对比分析**

与未训练的 Qwen‑3.5 4B/9B 基线在 HumanEval‑X++ 上对比，平均提升 13%（4B）/13.6%（9B），中等语言提升达 21%；在 CodeScope 上提升至 29.7%；整体表现显著优于基线，尤其在中等和长尾语言上提升显著。

**⚠️ 局限性**

局限性包括：奖励模型依赖 Python 种子，可能对语法结构差异极大的语言（如 Ada、OCaml）提升有限；缺乏多次实验或统计置信区间；训练需要大量算力；对极低资源语言的覆盖仍不充分。

---

## 141. A Bounded Reclaim Actuator for PSI-Guided Compressed Memory: A Controlled Ablation

**arXiv ID:** 2608.13689 | [PDF](https://arxiv.org/pdf/2608.13689v1)

**作者:** Abhiyan Dhakal `[一作]` (Kathmandu University), Sanjog Sigdel `[通讯]` (Kathmandu University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在Linux虚拟机上比较三种内存压缩与回收策略——静态启用zram、延迟启用zram以及在PSI信号触发时向后台cgroup发起一次96MiB回收请求，来评估它们对前台工作负载（计算和SQLite查询）的响应时间影响。

**💡 创新点**

创新点在于将PSI驱动的回收请求与已存在的压缩swap结合，提出并验证了一种“bounded reclaim”机制，并通过对比实验系统性评估其在不同压力场景下的效果，填补了仅关注单一压缩或回收机制的研究空白。

**🔧 技术方法**

使用技术包括Linux PSI（Pressure Stall Information）、zram（压缩块设备）、cgroup v2的memory.reclaim接口、eBPF用于监控、以及自定义压力生成器和前台canary作业。

**📊 数据集**

实验数据集为9台1vCPU、1GiB RAM的Linux虚拟机，配备计算和SQLite前台工作负载；共完成180个匹配的实验案例（每种配置各60个），覆盖四种压力速率和内容类型。

**📈 对比分析**

比较方法为在相同压力生成器和前台负载下执行三种配置，测量压力阶段的p99响应时间，并使用匹配三元组的对数比值和Bootstrap置信区间进行统计。结果显示：bounded reclaim相较于静态zram对计算工作降低约6.2% p99；相较于延迟启用，bounded reclaim对计算和SQLite均提升约30–35% p99；但对SQLite与静态zram的差异不显著，未能证明普适性能提升。

**⚠️ 局限性**

局限性包括：仅在单核1GiB环境中测试，无法代表多核或更大服务器；仅使用一次96MiB的固定回收剂量，未探究自适应或多次回收策略；缺乏页级追踪以验证页淘汰对延迟的影响；前台工作负载为简化的canary，未覆盖真实交互式或数据库服务场景。

---

## 142. Modular Cognitive Architecture Emerges in Large Language Models

**arXiv ID:** 2608.13567 | [PDF](https://arxiv.org/pdf/2608.13567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 143. Strand-based Hairstyle Generation via Large Reconstruction and Multimodal Models

**arXiv ID:** 2608.13679 | [PDF](https://arxiv.org/pdf/2608.13679v1)

**作者:** Conghui Hao `[一作]` (LIGHTSPEED), Kui Wu `[通讯]` (LIGHTSPEED)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于单张图像，利用大型重建模型与多模态模型结合传统几何处理，自动生成高质量、可编辑的基于线条的头发模型。

**💡 创新点**

不需要任何专门训练或数据收集，利用大模型的语义和重建能力，再通过三阶段扩散-纠正-扩散管线，解决方向模糊与内部体积扩散问题，实现复杂发型（如发髻、马尾）的一致性线条重建。

**🔧 技术方法**

大规模重建模型（Tripo 3D）、大规模多模态模型（Nano Banana 2）、结构张量求解、三阶段方向扩散、基于体积样本的曲线积分追踪。

**📊 数据集**

主要使用公开的重建/多模态大模型，无需自建数据集；实验中以多样化的在野图像作为输入，评估与四个单图像发型重建方法（NeuralHDHair、HairStep、DiffLocks、Im2Haircut）对比。

**📈 对比分析**

在20分钟内完成约50k根线条的重建；在IoU（0.75）和方向误差（32.6°）上均超过所有对比方法；与最新方法HairLRM/HairGPT的视觉对比显示形状与细节更好。

**⚠️ 局限性**

对极其复杂的卷发、交错结构以及极细微的线条细节仍表现欠佳，单图像信息不足导致细节缺失，且未提供交互式编辑工具。

---

## 144. Not All Tokens Are Equal: Inflation-Aware Routing for Agentic LLM Systems

**arXiv ID:** 2608.13571 | [PDF](https://arxiv.org/pdf/2608.13571v1)

**作者:** Heming Fu `[一作]` (Stony Brook University), Guojun Xiong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5102620407)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对代理式大型语言模型的通胀感知路由系统，利用四阶段路由器在重试过程中预测并抑制多次调用产生的额外令牌消耗。

**💡 创新点**

创新点在于引入令牌通胀概念、CoT分支熵作为无成本的预执行难度信号、基于语义交换率（SER）的路由目标以及在升阶时使用新鲜调用以避免上下文污染。

**🔧 技术方法**

主要技术包括局部链式推理采样计算CoT分支熵、轻量化多层感知网络预测通胀率、SER计算与模型选择以及新鲜升阶策略。

**📊 数据集**

实验数据集采用多步算数题库GSM8K和需要链式检索的HotpotQA，用于评估通胀率、预测准确度和路由效果。

**📈 对比分析**

与FrugalGPT、All-Small及Confidence Escalation等基线对比，本文方法在固定令牌预算下达到了94.7%的准确率（高于FrugalGPT的91.0%），且使用的令牌减少31%，通胀预测的AUROC达到0.887。

**⚠️ 局限性**

局限性包括仅在两类推理任务上验证，未覆盖开放式生成、多轮对话或工具使用场景，通胀预测模型缺乏跨任务泛化能力，计算比率固定且未考虑动态定价，污染实验样本有限。

---

## 145. GALA: Generation-Aware Cross-Modal Alignment for Text-to-Time-Series Synthesis

**arXiv ID:** 2608.13741 | [PDF](https://arxiv.org/pdf/2608.13741v1)

**作者:** Haochen Zhang `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种两阶段的文本到时序生成框架GALA，通过先对预训练文本编码器和时间序列编码器进行对比学习并加入生成辅助损失实现生成感知的跨模态对齐，再冻结文本嵌入作为条件训练流匹配生成器，从而生成与文本描述高度匹配的时序片段。

**💡 创新点**

核心创新在于把跨模态对齐与生成分离，并在对齐阶段加入生成感知的辅助损失，使得文本嵌入既能与时序保持对齐，又能保持足够的信息以驱动生成；此外采用全局z-score归一化保留幅度信息。

**🔧 技术方法**

使用LoRA微调的预训练文本编码器Gemma-300M和时间序列基础模型Chronos-2，配合对比学习（InfoNCE）和生成辅助损失；生成器为基于DiT的流匹配扩散模型；归一化改为全局z-score；评估使用FID、CTTP、JFTSD等指标。

**📊 数据集**

在TSFragment-600K数据集上进行实验，该数据集包含ETTh1s、ETTm1、电力、交通四个域，分别生成长度为24、48、96的时序片段。

**📈 对比分析**

与VerbalTS、T2S、DiffuSETS、BRIDGE和Text2Motion等五种基线对比，GALA在30/36指标列中取得第一，在平均排名上达到1.08/1.08/1.42，显著提升文本遵从度（CTTP、JFTSD），在FID上也保持领先或相近。

**⚠️ 局限性**

局限性包括对较长时序片段的效果未验证；对齐阶段需要额外训练成本；对齐策略对不同文本编码器的依赖性未充分探索；仅针对单一模态（文本）进行对齐，未考虑多模态交互。

---

## 146. Measuring Fairness in Large Audio Language Models via Semantic-Aware Bias Estimation

**arXiv ID:** 2608.13624 | [PDF](https://arxiv.org/pdf/2608.13624v1)

**作者:** Zhe Liu `[一作]` `[通讯]` (Meta Platforms, Inc.), Zhe Liu (Meta Platforms, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种语义感知混合效应回归框架，用来评估大型音频语言模型（LALM）的公平性，显式控制口语语义差异和说话者特征，减少误判。

**💡 创新点**

创新点在于：①利用同一LALM生成参考文本的语义嵌入作为协变量，①使用混合效应模型同时考虑语义协变量和说话者随机效应；②将语义嵌入的提取方法（Embed‑AGG 与 Embed‑EOWL）与传统外部嵌入对比。

**🔧 技术方法**

技术包括：混合效应Poisson/Binomial回归、主成分分析降维、适应高斯-赫尔米特积分、EOWL提示式语义提取、Embed‑AGG层级聚合、Bootstrap/Delta 方法进行置信区间估计。

**📊 数据集**

使用的数据集包括：LibriSpeech Test‑Clean / Test‑Other（ASR）、AIR‑Bench‑Chat（问答），以及通过Llama‑3 70B 生成的 2000 问题的仿真数据。

**📈 对比分析**

与传统无控制或仅用说话者随机效应的回归比较，语义感知混合效应模型显著降低了伪公平性检验的假阳性率，结果更接近真实无差异；在实测数据上，所有模型的性能差异趋于无统计显著，显示该方法更稳健。

**⚠️ 局限性**

局限性：依赖于同一LALM的嵌入质量；在没有说话者标识的数据集上无法引入随机效应；PCA降维可能忽略细粒度语义信息；实验集中在性别属性和 Qwen2‑Audio，未验证对其他属性和模型的通用性。

---

## 147. High-dimensional nonparametric changepoint detection via low-rank degree-two density projection

**arXiv ID:** 2608.13922 | [PDF](https://arxiv.org/pdf/2608.13922v1)

**作者:** Guoqing Zhang `[一作]` (North Carolina State University), Zhaixin Chen `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于度二密度投影的低秩矩阵CUSUM方法，用来在高维非参数分布变更检测中保持全部二阶信息而不需估计密度

**💡 创新点**

核心创新是将度二投影映射为矩阵均值问题，利用低秩截断去噪并通过多尺度区间删除和交叉拟合细化实现接近下界的定位精度

**🔧 技术方法**

采用Legendre多项式特征映射、矩阵均值估计、低秩截断、矩阵伯恩斯坦不等式、交叉拟合方向与位置细化以及依赖性扩展的几何β混合 Bernstein 边界

**📊 数据集**

实验使用高维高斯-Copula模拟数据（d=20,50,100,200）、多变更d=100序列以及UCI人类活动识别数据（128维）

**📈 对比分析**

与传统均值CUSUM、全度二阶CUSUM以及基准方法对比，低秩方法在高维下保持低定位误差（≈4-6点），而均值CUSUM在高维完全失效；在多变更与UCI数据上实现高准确率（多变更Hausdorff误差≤1，UCI 72%精确恢复）

**⚠️ 局限性**

只能检测度≤2变化，低秩假设限制了可检索的跳跃结构，对高阶或非低秩变更不敏感；依赖性分析仅覆盖预处理阶段，精细化步骤仍假设独立性

---

## 148. The Capturing and Logging Ecological Virtual Experiences and Reality (CLEVER) - Job Simulator Dataset

**arXiv ID:** 2608.13715 | [PDF](https://arxiv.org/pdf/2608.13715v1)

**作者:** Qidi J. Wang `[一作]` (Virginia Tech), Ryan P. McMahan `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了一个名为 CLEVER – Job Simulator 的公开 VR 运动数据集，包含 95 名参与者在 Office Worker 与 Store Clerk 两个域中完成六项任务，并记录了运动轨迹、交互、问卷与视角视频。

**💡 创新点**

创新点在于提供了多设备（Meta Quest 3S 与 VIVE XR Elite）、多域、多任务、多会话的大规模数据集，并使用 CLOVR 自动化收集与问卷同步，显著丰富了可用于机器学习的 VR 运动数据资源。

**🔧 技术方法**

技术手段包括 CLOVR 记录工具、SteamVR 平台、Job Simulator 游戏、Meta Quest 3S 与 VIVE XR Elite 头显、FPS 与 SSQ 问卷，数据以 CSV 与 MP4 形式保存。

**📊 数据集**

使用的数据集是本文自建的 CLEVER – Job Simulator（95 人、5 会话/人），并与 Nair 等、Moore 等等已有公开数据集进行对比分析。

**📈 对比分析**

通过对比表格（用户数、设备数、应用数、域数、任务数、会话数与公开性）展示了该数据集在用户规模与多设备配置方面的优势；论文未给出具体机器学习实验结果，但表明该数据足以支持此类研究。

**⚠️ 局限性**

局限性包括仅覆盖两种域与单一游戏、设备仅限两种型号、实验环境受限、部分参与者存在数据丢失、缺乏跨应用/跨平台多样性，且目前未验证实际模型性能。

---

## 149. Beyond Text Conditioning: A Systematic Study of MLLM-DiT Fusion for Video Generation

**arXiv ID:** 2608.14043 | [PDF](https://arxiv.org/pdf/2608.14043v1)

**作者:** Yanbo Ding `[一作]` (Chinese Academy Of Sciences), Yali Wang `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 BiVidGen，一个将多模态大语言模型生成离散视觉语义 token 并在 Diffusion Transformer 中进行视频渲染的混合框架。

**💡 创新点**

系统化验证并提出了使用 EMA 基础的离散视觉 token、因果自回归生成以及多层交叉注意力条件能够显著提升语义对齐与时序一致性的三大关键设计。

**🔧 技术方法**

采用 Qwen3‑VL 作为 MLLM、Wan‑2.2‑T2V‑5B 作为 DiT、EMA 视觉 tokenizer、因果 attention 的自回归视觉 token 生成、多层交叉注意力融合以及标准 diffusion 训练与 CFG。

**📊 数据集**

训练数据为 10M 视频与 20M 图像，其中精选 2M 高质量视频；评估使用 VBench‑Long 与 DAVIS。

**📈 对比分析**

在 VBench‑Long 上与纯文本条件 DiT 基线比较，基线分数 74.88 提升至 78.18，进一步采用多层交叉注意力维持 78.18；在 30 步采样下仍保持 77.27，latency 31s，展示了优异的质量‑延迟折衷。

**⚠️ 局限性**

推理延迟增加、模型规模与训练数据未与最新大型视频生成系统相当，且对极长视频或多人物交互的泛化能力仍待进一步验证。

---

## 150. Graph-MambaNav: Spatial-Temporal Graph Mamba Leveraging Object-Relation Knowledge for Object-Goal Navigation

**arXiv ID:** 2608.13723 | [PDF](https://arxiv.org/pdf/2608.13723v1)

**作者:** Leyuan Sun `[一作]` (Wuxi University), Yanfei Sun `[通讯]` (Wuxi University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Graph-MambaNav，一种目标感知的空间-时间图编码框架，用于在未知环境中执行对象目标导航。

**💡 创新点**

创新点包括：①利用大语言模型（LLM）生成的常识关系先验同时初始化边权重和节点排序，从而使信息传播顺序与目标相关性对齐；②将 Graph-Mamba 的节点优先级扫描机制引入图神经网络，实现全局长程依赖；③在图层中结合局部 GNN 传播、全局 Mamba 扫描和时间 Mamba 序列建模，形成统一的时空图推理管线。

**🔧 技术方法**

技术细节：State Space Models (SSM) + Mamba、Graph-Mamba、DETR（对象检测）、ResNet-18（全局视觉特征）、GloVe 文字嵌入、跨模态注意力融合、A3C 强化学习策略。

**📊 数据集**

使用的主要数据集包括 AI2-THOR、RoboTHOR 的标准训练/验证/测试拆分，以及在真实机器人平台上的实验环境。

**📈 对比分析**

与多种基线（TSOG、Memory-MambaNav、CGI+GAIL、HOZ、VTNet 等）进行对比。实验显示在 AI2-THOR 上 SR/SPL 达到 83.22%/46.52%，在 RoboTHOR 上分别为 49.82%/28.67%，均显著优于现有方法；同时 ablation 证明局部、全局和时间扫描各自贡献；计算开销低于 Transformer 版本，具备更高 FPS。

**⚠️ 局限性**

局限性：①仅支持固定 22/12 类对象词表，无法处理多实例同类别；②对 LLM 生成的关系先验噪声敏感，关系不够区分时性能下降；③真实环境中物体布局与常识先验偏差时可能导致误导性探索。

---

## 151. Discovery and Spatial Characterisation of Multiple Shortcut Groups for Auditing Vision Model Bias

**arXiv ID:** 2608.14051 | [PDF](https://arxiv.org/pdf/2608.14051v1)

**作者:** Akshit Achara `[一作]` (King's College London), Andrew P. King `[通讯]` (King's College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对已训练的视觉模型进行图像级的短路与任务贡献映射，并将这些映射聚类成可视化的“短路组”，用于识别与子群错误相关的空间模式并指导后续的审计与干预。

**💡 创新点**

创新点在于：①引入条件对齐与残差贡献分解得到图像级短路/任务贡献图；②使用 K‑means 与 NMF 对贡献图进行聚类，挖掘数据集中重复出现的空间短路模式；③提出基于短路组的风险评分与目标性干预方法，显著提升了错误定位与公平性改进。

**🔧 技术方法**

核心技术包括：梯度/注意力归因（LRP、AttnLRP、Grad‑CAM 等）、条件相关（partial correlation）对齐、区域贡献分解、K‑means 与 NMF 聚类、输入/特征空间遮蔽与放大干预。

**📊 数据集**

实验涵盖五个数据集：CelebA、CheXpert、Waterbirds、Camelyon17 与 ISIC2019，并在 ResNet50 与 ViT‑B/16 两种架构上进行验证。

**📈 对比分析**

与传统数据集级归因（如 OSCAR）相比，短路组能捕获更多错误（错误占比显著高于随机选取），并且在输入/特征干预实验中，组合短路抑制与任务放大能将子群性能差距（LPG）降低 6–20% 之多，同时保持整体准确率；实验结果表明该方法在多种模型与归因方式下均表现稳健。

**⚠️ 局限性**

局限性包括：仅处理单一敏感属性，若存在多属性场景需先做切片发现；基于网格的空间分割受图像对齐影响，可能导致不对齐数据的模式不够精确；正向贡献图仅表示支持性区域，无法区分真正的因果或关键特征。

---

## 152. Dynamic Multi-Depot Vehicle Routing with Online Requests: Event-Driven Transformer--DRL and Rolling-Horizon Benchmarking

**arXiv ID:** 2608.13799 | [PDF](https://arxiv.org/pdf/2608.13799v1)

**作者:** Faezeh Ardali `[一作]` (Louisiana State University), Gerald M. Knapp `[通讯]` (Louisiana State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个事件驱动的动态多仓库车辆路线规划框架，结合可行性掩码、固定前缀/灵活后缀路线承诺、MLP与Transformer行为克隆+PPO学习以及滚动视窗基准。

**💡 创新点**

在同一框架下统一处理可行性、路线稳定性与动态插入，首次将固定前缀/灵活后缀承诺与重新分配/重序列化损失分离评估，并比较多种学习与启发式方法。

**🔧 技术方法**

行为克隆、近端策略优化（PPO）、MLP与Transformer注意力网络、确定性可行性掩码、固定前缀/灵活后缀路线承诺、滚动视窗优化（OR-Tools+SCIP）等。

**📊 数据集**

使用合成欧几里得实例，包含小、中、大三种规模（30/50/80请求），共20个未见测试场景及5组独立训练种子。

**📈 对比分析**

在统一的可行性与稳定性协议下对随机、最近可行、最小增量、等待感知、混合、MLP-BC/PPO、Transformer-BC/PPO、滚动视窗等九种方法进行对比。结果显示“最近可行”在目标、距离、等待、稳定性、运行时均最优；学习策略在运行时更快但质量不及强启发式；滚动视窗在等待和完工时间上最好，但计算成本高。

**⚠️ 局限性**

学习策略未能超过最强启发式；Transformer在不同种子下表现波动大；仅在合成数据上验证，缺乏真实网络或时间窗口等更复杂约束；未对多尺寸训练、奖励改进等进行探索。

---

## 153. What to Preserve, Where to Adapt: A Depth-Wise Analysis of Forgetting in Continual Gynecological Image Segmentation

**arXiv ID:** 2608.13660 | [PDF](https://arxiv.org/pdf/2608.13660v1)

**作者:** Amal Saqib `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Mohammad Yaqub `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 2488 | [OpenAlex ID](https://openalex.org/A5088282276)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

研究了持续学习在妇科图像分割中的遗忘问题，分析了不同编码器-解码器深度对性能的影响，并提出了深度受限适应框架。

**💡 创新点**

发现遗忘与网络层深度显著相关，提出仅更新靠近瓶颈的层能有效保持旧任务，展示了深度受限适应导致的记忆保持与分配的非线性转变。

**🔧 技术方法**

使用 block‑wise ablation、depth‑constrained adaptation、nnU‑Net/Lifelong nnU‑Net/MultiTalent nnU‑Net、EWC、LwF、rehearsal 等技术，以及 Dice+Cross‑Entropy 损失。

**📊 数据集**

在 UMD（子宫肌瘤MRI）、ECPC‑IDS（PET+CT 肿瘤）和 UT‑EndoMRI（子宫内膜异位 MRI）三大公开妇科数据集上进行实验。

**📈 对比分析**

与单任务、联合训练及多种持续学习基线（如 Sequential fine‑tuning、Rehearsal、LwF、EWC）对比，使用 Dice 评价；传统方法遗忘率高（>0.7），重放表现最优（≈0.08），但在此高度异质环境下仍表现不佳；深度受限适应在保持旧任务时保留率高，但若更新到浅层会导致急剧遗忘。

**⚠️ 局限性**

仅在三任务序列上验证，缺乏对更长任务序列的评估；只关注 encoder‑decoder 结构，未考虑其他网络；未给出深度相关遗忘机制的理论解释；实现细节与超参数对结果影响显著。

---

## 154. The Query Knows What to Forget: A Second Erase Direction for Linear Attention

**arXiv ID:** 2608.13668 | [PDF](https://arxiv.org/pdf/2608.13668v1)

**作者:** Dhruman Gupta `[一作]` (Truth Audit Labs), Debayan Gupta `[通讯]` (Truth Audit Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在线性注意力模型GDN-2中引入查询衍生擦除方向（QED），通过在擦除向量中加入基于查询、正交于键的第二擦除方向，提升了长上下文检索性能。

**💡 创新点**

创新点在于将擦除操作从仅基于键扩展到同时利用查询信息，既消除了读取时的干扰，又保持了单秩递推更新的稳定性。

**🔧 技术方法**

采用线性注意力、GDN-2、快速权重视角、门控机制、键正交投影以及对QED的多维消融与推理时强度调节。

**📊 数据集**

使用FineWeb‑Edu（Llama‑2分词器）训练模型，评估数据包括S‑NIAH‑1/2/3、FDA、SQuAD completion以及SWDE。

**📈 对比分析**

与基线GDN‑2以及若干消融对照（QED‑rand、QED‑nogate、QED‑nogate‑noproj）在两次随机种子下对比，QED在超过训练窗口的上下文长度（最高32K）中提升检索准确率（最高提升约39%），将可用上下文从约8K扩展至16K，而验证损失和SQuAD completion几乎无变化。

**⚠️ 局限性**

限制在于门控和键正交投影的具体效益尚未在所有实验中稳定，推理时强度调节对不同检查点效果不一致，且实验仅在340M规模模型上验证，需更多规模、更多训练跑以及λ_max的系统性搜索以确定稳定性约束的真正作用。

---

## 155. Ontology-Grounded World Models for Failure Diagnosis and Closed-Loop Repair in Physical AI Systems

**arXiv ID:** 2608.13901 | [PDF](https://arxiv.org/pdf/2608.13901v1)

**作者:** Kailin Wang `[一作]` (Country Garden Services Group), Zhiyou Heng `[通讯]` (Country Garden Services Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于本体的诊断与验证门控纠错接口 Onto‑EV‑WM，能够在 EV‑WM 世界模型之上对预测或仿真状态进行类型化映射，记录缺失谓词、参数及其阈值，并通过确定性诊断规则与路由映射实现闭环重试与纠错；

**💡 创新点**

首次将任务本体（TBox/ABox）与事件-验证世界模型相结合，实现了谓词级别的失败诊断、路线标签分派以及谓词验证门控的纠错机制，且支持直接仿真状态修改与控制器执行的多模态路由；

**🔧 技术方法**

采用视觉特征编码、预测世界模型、事件预测器、事实解码器及本体定义；通过诊断规则 Π_D、路由映射 Π_R 产生纠错提议器（学习/启发式/规划）；采用直接仿真状态修改或控制器执行，最终由原生谓词验证器完成接受判定；

**📊 数据集**

使用 PointMaze 50 次对齐实验；LIBERO‑Goal（10 个操作任务，采样 4 种 seed 的 25 步演示窗口）；LIBERO‑Plus（10,030 个扰动任务的固定注册表，划分为 LIBERO‑10、Goal、Object、Spatial 四组）；

**📈 对比分析**

在对齐的 PointMaze 50 次实验中，EV‑WM 与 Onto‑EV‑WM 均达到 94% 成功率，Onto‑EV‑WM 的平均最终状态距离下降约 32%；在 LIBERO‑Goal 采样窗口协议下，单一校正头修复率约 93.8%（seed 0），四种 seed 的平均 ±%；在 LIBERO‑Plus 注册表上，整体成功率 85%（LIBERO‑Goal/Obj/Spatial >91%，LIBERO‑10 65.98%）；

**⚠️ 局限性**

仅在仿真环境下评估，未验证真实机器人；评估协议不包含完整任务跑、独立训练或组件级因果分解；纠错多为直接状态修改，未测试动态可实现性；未评估跨路由切换与在线学习的效果；

---

## 156. Leading-Silence Augmentation and Multi-Stage Synthetic Supervision for the Second MLC-SLM Challenge

**arXiv ID:** 2608.14150 | [PDF](https://arxiv.org/pdf/2608.14150v1)

**作者:** Kexin Shi `[一作]`, Malu Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在第二届多语种对话语音语言模型挑战赛中，本文分别针对任务一的说话人分配与识别以及任务二的对话语音理解，构建并训练了两个独立的单模型系统；任务一使用VibeVoice-ASR‑7B通过随机前置静音裁剪、时间戳校正和指数移动平均（EMA）训练策略进行微调，任务二则通过 Gemini 2.5 Pro 生成候选问答、Qwen2.5‑Omni‑7B 进行静音音频过滤、分布匹配扩增，并在 Qwen3‑Omni‑30B‑A3B‑Instruct 上实现带标签直接回答。

**💡 创新点**

创新点在于（1）针对无先验说话人标签与语句边界的长音频，提出随机前置静音裁剪与一致时间戳校正的轻量级增强方法；（2）在 ASR 微调中引入 EMA 训练以稳定参数；（3）构建完整的多模态合成问答管线，包括静音过滤、分布匹配扩增以及统一的标签化回答格式，显著提升了音频驱动的问答性能。

**🔧 技术方法**

技术包括 VibeVoice-ASR‑7B + LoRA 微调、随机前置静音裁剪与时间戳校正、EMA 训练策略、Gemini 2.5 Pro 生成合成问答、Qwen2.5‑Omni‑7B 静音过滤、分布匹配数据增强、Qwen3‑Omni‑30B‑A3B‑Instruct 指令微调及标签化回答提取。

**📊 数据集**

使用挑战赛官方训练集（完整、无分段的多语种对话录音），以及通过 Gemini 2.5 Pro 从训练音频生成约 210k 句子问答候选，再筛选并扩增至约 127k 条合成数据；评测则使用官方任务一和任务二的评估集。

**📈 对比分析**

与官方基线比较，任务一的 tcpMER 从 18.30% 降低至 16.73%（约 8.6% 的相对提升）；任务二的单选多选准确率从 78.0% 提升至 86.0%（相对提升 10.2%），分别体现了增强策略和模型微调的显著效果。

**⚠️ 局限性**

局限性包括：①在任务一中仅评估累计配置，缺乏单独的裁剪或 EMA 效果交互分析；②任务二的合成问答仍可能存在与真实评测分布不完全一致的偏差；③两任务均未提供 oracle 语句边界和说话人标签，模型对极端长音频或复杂交互的鲁棒性仍待验证。

---

## 157. The Architect: Interactive Visualization of Deep Learning Mathematics Directly in Microsoft Excel

**arXiv ID:** 2608.13572 | [PDF](https://arxiv.org/pdf/2608.13572v1)

**作者:** Mohammad Imrul Jubair `[一作]` (University of Colorado Boulder), Tom Yeh `[通讯]` (University of Colorado Boulder)

**通讯引用:** 4832 | [OpenAlex ID](https://openalex.org/A5070687718)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过Excel生成可交互的神经网络计算蓝图，让前向、反向传播及参数更新的矩阵运算在电子表格中可视化并可实时编辑。

**💡 创新点**

利用熟悉的电子表格环境实现神经网络数学透明化，自动生成可编辑公式的工作簿，并与PyTorch代码并列显示。

**🔧 技术方法**

使用Office Scripts（TypeScript）生成工作簿，利用Excel动态数组、公式、命名范围等原生功能实现实时计算。

**📊 数据集**

内置tinyDigits数据集（手写数字图像）以及随机/图像输入示例。

**📈 对比分析**

通过课堂实验和案例展示验证可用性，未进行量化性能评估；主要通过学生提交的工作簿与对比实验来展示可视化和交互效果。

**⚠️ 局限性**

仅适用于规模较小、顺序结构的全连接网络，无法处理大规模或卷积/注意力模型；依赖Excel最新功能，计算量随网络大小增长；缺乏正式实验验证学习/调试效果。

---

## 158. Robust Dual-Model Collaborative Random Vector Functional Link Network

**arXiv ID:** 2608.13628 | [PDF](https://arxiv.org/pdf/2608.13628v1)

**作者:** A. Quadir `[一作]` (Indian Institute of Technology Indore), M. Tanveer `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出KRPRVFL模型，结合RVFL与核风险敏感均值p次幂准则，并引入协同学习机制，以提升对噪声、离群点和不平衡数据的鲁棒性。

**💡 创新点**

创新点在于用核风险敏感均值p次幂损失替代最小二乘实现自适应抑制错误标签和异常样本；通过协同学习提升模型在复杂环境中的泛化；利用核映射避免显式隐藏层尺寸选择，保持轻量高效。

**🔧 技术方法**

使用核化随机向量函数网络（RVFL）、核风险敏感均值p次幂（KRP）准则、协同学习机制，以及矩阵闭式求解与迭代固定点映射，配合多种激活函数和正则化。

**📊 数据集**

实验使用37个UCI/KEEL基准分类数据集，包括银行、血液、乳腺癌、棋盘、车辆、心脏等多种数据集。

**📈 对比分析**

通过准确率、平均排名、Friedman检验与Nemenyi后验检验进行比较；KRPRVFL平均准确率85.20%，平均排名2.03，显著优于RVFL、ELM、GB-RVFL等基线模型。

**⚠️ 局限性**

局限在于对核与风险敏感参数高度依赖；对大规模或流式数据的适应性有限；需要手动调参；在极大数据集上可能受限。

---

## 159. Reducing ANN-SNN Conversion Error via Residual Membrane Potential Alignment

**arXiv ID:** 2608.13952 | [PDF](https://arxiv.org/pdf/2608.13952v1)

**作者:** Zirui Chen `[一作]` (Peking University), Zhaofei Yu `[通讯]` (Peking University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过分析残差膜电位分布，提出SRMP正则化调节初始膜电位，并结合SCR‑Conv2d竞争机制降低ANN‑SNN转换误差。

**💡 创新点**

首次将残差膜电位分布正则化与动态初始膜电位调整相结合，同时引入自适应竞争细化层，显著提升极短时步SNN精度，并可兼容多阈值与Transformer结构。

**🔧 技术方法**

采用统计分析、RMPD正则化、动态权重调节、IF神经元、QCFS激活函数、分组卷积竞争层、Transformer ECMT框架等技术。

**📊 数据集**

在CIFAR‑10、CIFAR‑100、ImageNet以及ViT‑B/16等视觉数据集上进行实验。

**📈 对比分析**

与QCFS、SRP、SNM、NQQCFS、RCS‑SRMP等基线在T=2、4、8等低时步下对比，结果显示在VGG‑16、ResNet‑18等网络中精度提升显著（如CIFAR‑10 T=2从95.78%提升至92.84%，ImageNet 16步提升至56.46%）。

**⚠️ 局限性**

方法假设输入近似正态分布，动态权重调节实现复杂，且在高T时步时性能提升有限；在某些网络结构中需额外正则化训练步骤，影响部署便利性。

---

## 160. Buy the Rumor, Sell the News: When Is News Priced In?

**arXiv ID:** 2608.14014 | [PDF](https://arxiv.org/pdf/2608.14014v1)

**作者:** Alireza Kargarzadeh `[一作]` (Tailstate Intelligence Ltd), Arman Khaledian `[通讯]` (Zanista AI Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文利用4.57百万条美国上市公司新闻，构建了17类事件标签和5种属性，对1.68百万事件的异常收益进行事件研究，验证“买传闻、卖新闻”说法，并量化新闻对价格、波动率的影响。

**💡 创新点**

创新点在于：① 将大语言模型教师知识压缩为高精度(≈95%)的稀疏分类器；② 通过嵌入式聚类将文章分为故事/事件；③ 用中性新闻作为placebo基线，剔除基准误差后揭示真实的持续/逆转效应，生成每类事件的漂移表。

**🔧 技术方法**

主要技术包括：Azure-hosted GPT‑5‑mini 教师标注、distilroberta‑base 学生分类器、主动学习、标题+摘要嵌入聚类、beta‑调整的异常收益、日期簇化自举、净化后的基线回归。

**📊 数据集**

数据集为2023‑2026年约3,000只美国上市股的4.57M条公开新闻（经去重、筛选后），包含17类事件标签、5属性，生成1.68M事件窗口；其中364,405条中性情绪事件用作placebo。

**📈 对比分析**

方法上与单因子Beta基准对齐，并用中性新闻placebo校正；结果显示新闻相关价格变动大多发生在发布前后，传闻日捕获几乎全部收益，定量新闻表现出延迟反应，软新闻则超买后逆转；调整后每类事件的漂移表为量化预测的先验，可提升新闻条件预测模型的稳健性。

**⚠️ 局限性**

局限性包括：标注误差集中在“价格评论”“促销”等杂项标签；单因子Beta基准未充分捕获规模等风格效应；日终收盘数据无法区分盘中与盘后反应；仅使用公开新闻，忽略内部信息；经济意义评估采用简化的固定窗口、等权重、平面成本，未充分体现交易成本与策略复杂度。

---

## 161. Simplicial Semantics for Belief Revision

**arXiv ID:** 2608.13763 | [PDF](https://arxiv.org/pdf/2608.13763v1)

**作者:** Philip Sink `[一作]` `[通讯]`, Philip Sink

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在单纯形语义框架中引入信念修订模型，利用“视角”概念定义世界之间的接近度，并给出两种修订规则；同时提出记忆机制以避免反复修订导致的“遗忘”问题。

**💡 创新点**

创新点在于：①将单纯形模型中的节点视作代理的视角，从而用共享视角的数量来度量世界之间的接近度；②提出两种修订规则（全局修订与仅在视角被孤立时修订）；③通过利用原始复形 S 作为记忆来维持已公布信息，解决多次修订后“回退”现象。

**🔧 技术方法**

技术手段包括：单纯形语义（UCF 复形、最大复形、投影函数 π_a）、定义修订函数 ℛ_a、构造修订后模型 [φ]、Grove 变体 [φ]_G 以及记忆版 [φ]_m；并使用非模态公式作为公告语句。

**📊 数据集**

本研究为理论性工作，无使用外部数据集；所有示例均为人工构造的单纯形模型。

**📈 对比分析**

未涉及实验比较或性能评估；本文主要通过对比多组示例（如多子女、泥孩子、私有比特值）来说明模型的直观一致性和潜在缺陷。

**⚠️ 局限性**

局限性包括：缺乏经验验证、未给出对公告可信度和不确定性处理的完整形式化；记忆机制假设公告不可否定，且对更复杂的多代理情境、信任度差异、以及与 AGM 体系的对应关系仍待进一步研究。

---

## 162. The Capacity Region of the Multiple Access Channel with Non-Signaling Assistance

**arXiv ID:** 2608.13860 | [PDF](https://arxiv.org/pdf/2608.13860v1)

**作者:** Yuhang Yao `[一作]` (University of California Irvine), Syed A. Jafar `[通讯]` (University of California Irvine)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文针对 K 发信人无记忆多址信道（DM‑MAC）在提供全局非通信关联（NS）资源时，给出了其容量区域的单字母描述，并通过构造满足 NS 约束的编码方案实现该容量。

**💡 创新点**

创新点在于：①证明了 NS 辅助容量区域与经典容量区域在形式上完全相同，只是允许发信人输入分布产生任意依赖；②给出了可实现乘法增益接近 K 的构造示例（sum‑product‑lock 信道），从而确定了 K 为 NS 辅助容量增益的极限。

**🔧 技术方法**

主要技术包括：NS 约束下的编码构造、对称化（twirling）、典型集分析与大数定律、子分布归一化与凸组合，以及信息论中的互信息/条件互信息计量。

**📊 数据集**

该工作为理论研究，未使用任何实验或公开数据集，所有结果均来自严格的数学推导与构造。

**📈 对比分析**

通过与经典 DM‑MAC 容量区域对比，证明了 NS 辅助可显著扩大可达速率集合；在二元加法信道上实现了对数 3 的容量，且在 sum‑product‑lock 信道上实现了接近 K 的乘法增益。

**⚠️ 局限性**

局限性：仅在 NS 资源同时可用于所有发信人和接收方时给出单字母结果；当 NS 仅提供给发信人时，目前只有多字母表述；对带状态信道（尤其是因果/非因果 CSIT）下的 NS 辅助容量区域仍未完全确定。

---

## 163. Polar Code Based Federated Learning: Convergence Analysis and Resource Allocation

**arXiv ID:** 2608.13961 | [PDF](https://arxiv.org/pdf/2608.13961v1)

**作者:** Han Xiao `[一作]`, Nan Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于极化码的联邦学习方案，通过有限块长度下的UEP对量化比特进行保护，并给出了收敛分析与资源分配优化；

**💡 创新点**

创新点在于将极化码的UEP与量化结合，按比特重要性动态分配保护；同时联合优化量化位数与码长以最小化收敛误差，并提供理论收敛上界与实验证明；

**🔧 技术方法**

采用极化码、无偏量化、二进制擦除通道模型、Bhattacharyya参数分析、混合整数非线性优化（PSO）、MNIST CNN实验等技术；

**📊 数据集**

使用MNIST图像分类数据集进行实验；

**📈 对比分析**

与无编码、固定码长LDPC编码、固定/变码长极化编码对比，采用准确率作为评估指标，极化方案在高擦除概率下收敛更快、准确率更高；

**⚠️ 局限性**

局限在于需要预先拟合Bhattacharyya参数、优化过程复杂、量化位数与码长受极化码结构限制，对非IID或异构设备验证不足，以及通信/计算开销未做深入评估。

---

## 164. StreamHear: Domain-Adapted Pseudo-Labeling for Semi-Supervised Streaming Speech Recognition

**arXiv ID:** 2608.13717 | [PDF](https://arxiv.org/pdf/2608.13717v1)

**作者:** Zefang Liu `[一作]` (Capital One), Sambit Sahu `[通讯]` (Capital One)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 StreamHear，一种一次性半监督流程，用领域自适应的离线教师生成伪标签并微调缓存感知流式 ASR 学生模型，同时通过先验正则化的动态规划重定位步骤纠正块级词语归属。

**💡 创新点**

创新点在于：①一次性无迭代伪标签生成与学生微调；②利用先验正则化的动态规划实现块级词语重新分配，提升训练数据质量；③通过固定离线教师实现高效的无辅助网络迁移，显著缩小流式学生与离线教师的性能差距。

**🔧 技术方法**

采用离线转录器（Parakeet‑TDT 0.6B）作为教师，缓存感知 FastConformer‑RNN‑T 作为学生；使用 CTC‑Segmentation + 先验正则化的 Needleman‑Wunsch DP 进行重定位；伪标签按教师对数似然过滤；训练使用 AdamW、Cosine 退火、SpecAugment 等标准手段。

**📊 数据集**

评估数据集包括公开的 Earnings‑21、Earnings‑22、SPGISpeech（金融通话语料）以及内部银行客服通话 BankCall，涵盖多种域移位场景。

**📈 对比分析**

与仅监督微调的学生模型和离线教师进行对比；在四个数据集上 StreamHear 在标注测试集上平均提升 0.18–0.88 个百分点，在未标注集上提升 0.44–1.85 个百分点，基本逼近离线教师的性能，证明单轮伪标签足以显著提升流式 ASR。

**⚠️ 局限性**

局限性包括：对离线教师质量高度依赖，无法进一步迭代提升伪标签；在极端域迁移或低质量无声音频中效果可能受限；未探索多语言或多方言的跨域适应，且仍存在学生模型与教师之间的性能差距。

---

## 165. SemPlan: Benchmarking Structured Semantic Planning for LLM-Based Queries over Enterprise Data

**arXiv ID:** 2608.13612 | [PDF](https://arxiv.org/pdf/2608.13612v1)

**作者:** Bruno Santos Teixeira `[一作]` `[通讯]` (Universidade Federal de Ouro Preto), Bruno Santos Teixeira (Universidade Federal de Ouro Preto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文在同一 LLM 模型和相同数据集上，比较了四种自然语言到数据库查询的架构（直接 SQL、工具代理、语义请求、含澄清和状态的语义请求），并提出了一个双语合成基准。

**💡 创新点**

创新点在于：①设计了可复现的双语合成企业分析基准；②在同一实验环境下系统性对比四种架构；③提供完整的可复现包和结果哈希。

**🔧 技术方法**

使用了 OpenAI LLM、结构化提示、SQL 语法检查、工具调用接口、语义请求解析及 deterministic 执行。

**📊 数据集**

基准采用合成的 Northstar Commerce 领域数据，共 1,800 个双语案例（900 英语、900 葡萄牙语）。

**📈 对比分析**

通过冻结实验、配对统计（McNemar、bootstrap）和成本/延迟分析，发现语义请求（A3）在答案正确率（≈25%）和稳定性最高；A4 成本最低、误拒率最低；A1 在政策合规性最好。

**⚠️ 局限性**

局限性包括：仅使用单一 LLM 模型和提供商；数据为合成，未覆盖真实企业情境；准确率整体偏低；不涉及多模型、真实数据或 UI 交互评估。

---

## 166. Extracting and Verifying Illicit Bitcoin Addresses from Underground Forum Discussions

**arXiv ID:** 2608.13930 | [PDF](https://arxiv.org/pdf/2608.13930v1)

**作者:** Abdoul Nasser Hassane Amadou `[一作]` (Mohammed VI Polytechnic University), Anas Motii `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并发布了一个可复制、基于证据的比特币地址标签数据集，从HackForums抓取、验证并人工确认了2,438个非法比特币地址，完整公开了提取、验证和标注的整个管道。

**💡 创新点**

将LLM辅助筛选、专家人工审核与链上验证三者结合，形成了可复现的证据支持标签；公开完整的提取流程、时序元数据，并覆盖12种网络犯罪类别，规模大幅提升。

**🔧 技术方法**

使用正则表达式进行地址提取、区块链公共API进行链上验证、Qwen3-30B LLM进行零射击分类、人工专家审核以及统计分析与可视化工具。

**📊 数据集**

主要使用HackForums 42M帖子（CrimeBB学术数据集）以及区块链公共API（Blockchair、Etherscan）提供的比特币交易历史。

**📈 对比分析**

通过人工抽样评估LLM筛选的召回率约74%，精度93.6%；相比传统仅靠社区报告或单一案例的数据集，该方法在可验证性、类别覆盖和规模上显著提升。

**⚠️ 局限性**

仅有单一专家审核缺乏多评标一致性；LLM零射击对非英语或高度隐晦的地下论坛语言识别不佳，导致漏检；正则提取可能产生误匹配；未覆盖隐私币和部分伪地址；对HackForums语境的依赖限制了可推广性。

---

## 167. Reward Machines for Signal Temporal Logic

**arXiv ID:** 2608.13625 | [PDF](https://arxiv.org/pdf/2608.13625v1)

**作者:** Alper Kamil Bozkurt `[一作]` (Virginia Commonwealth University), Yuichi Motai `[通讯]` (Virginia Commonwealth University)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5061499121)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于事件化 STL 的奖励机（STL‑RM）框架，用以将任意 STL 规范编译为一阶时钟交替时钟自动机（OCATA），再通过构造奖励机实现马尔科夫奖励，从而直接在标准强化学习（PPO）中学习满足 STL 规范的控制策略。

**💡 创新点**

创新点包括：① 通过 OCATA 捕捉 STL 的完整语义，支持任意嵌套的时序运算符；② 构造带有非确定性、合取、概率分支的奖励机，兼顾 Büchi 接受条件和鲁棒性；③ 通过内存列表（仅存必要的自动机位置和时钟值）实现高效的状态扩展，避免全历史状态扩展；④ 将观测扰动引入到奖励机中，实现对鲁棒性的概率化处理。

**🔧 技术方法**

使用的技术主要有：事件化 STL 语义、OCATA 的构造与语义，奖励机（RM）设计与更新函数 Δ，SMDP 与 RM 的乘积构造得到的马尔科夫决策过程，PPO 强化学习算法，实时 STL 鲁棒性评估（RTAMT），以及对观测的概率化处理。

**📊 数据集**

使用了 Gymnasium 公开环境的 5 个控制任务：CartPole、Reacher、Pusher、Fetch、Adroit，所有环境的观测信号为一维位置坐标，均统一归一化后应用相同的 STL 规范。

**📈 对比分析**

与基线方法（观察堆叠、LSTM 循环策略）以及离散化版本（STL‑RM (Discrete)）进行对比。实验结果显示：
• 对于部分规范（无时间限制），STL‑RM 与 STL‑RM (Discrete) 在所有环境（除 Fetch）上均实现 100% 的满足率，并在鲁棒性得分上优于基线；
• 对于完整规范（含 30 步稳定性约束），基线方法性能显著下降，而 STL‑RM 与 STL‑RM (Discrete) 能收敛到最优或近似最优策略，满足率几乎达到 1，鲁棒性得分亦高于基线；
• STL‑RM 在 Fetch 环境中表现出更高的鲁棒性得分，显示概率化转移设计的优势。

**⚠️ 局限性**

主要限制在于内存列表的容量有限：当 STL 规范产生大量同时存在的时间约束时，每个约束可能需要一个新的内存条目，导致内存占用急剧增长；此外，构造的 OCATA 可能并非总是可限制确定的，导致仅能保证满足率下界而非精确等价。

---

## 168. TenderKG

**arXiv ID:** 2608.14066 | [PDF](https://arxiv.org/pdf/2608.14066v1)

**作者:** Yacine Mokhtari `[一作]` (IMT Atlantique), Grégory Smits `[通讯]` (IMT Atlantique)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了TenderKG知识图谱数据集，整合了法国2021-2023年公开招标的公司、招标、批次及获奖信息，并附带文本、层级税onomies与地理特征等丰富侧信息，用于公共采购推荐研究。

**💡 创新点**

创新点在于首次公开提供面向投标推荐的知识图谱数据集，利用层级CPV/NAF分类、序数实体（金额区间、员工规模）以及多源实体对齐，实现了结构化、语义化的采购生态建模。

**🔧 技术方法**

使用了工业级文档检索、信息抽取与实体消歧技术构建图谱，并在基线实验中应用了图神经网络（GNN）等知识图谱学习方法。

**📊 数据集**

数据来源为法国官方发布的公共招标公告与中标结果（包括BOAMP、JOUE等平台），并结合欧盟CPV、法国NAF分类及地区行政区划信息。

**📈 对比分析**

通过对GNN推荐模型的基线评估，发现由于交互稀疏且仅包含正向获奖信息，模型表现显著低于传统推荐任务，凸显了数据集的挑战性与作为基准的价值。

**⚠️ 局限性**

主要局限在于仅公开获奖交互，导致合作信号极度稀疏；缺少失标信息、报告不完整性及多源异构性带来的噪声，影响模型训练与评估的全面性。

---

## 169. S2Dialog: Multimodal Dialogue Retrieval with Semantic and Acoustic-Style Modeling

**arXiv ID:** 2608.14029 | [PDF](https://arxiv.org/pdf/2608.14029v1)

**作者:** Xueqi Wang `[一作]` (Inner Mongolia University), Junfeng Zhao `[通讯]` (Inner Mongolia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了S2Dialog框架，实现多模态对话检索，能够同时考虑文本语义与语音风格。

**💡 创新点**

提出专门的对话层文本检索器和语音检索器，并引入对话层文本-语音对比学习以联合优化语义与风格。

**🔧 技术方法**

使用Sentence‑BERT和Wav2Vec2‑IEMOCAP提取单句特征，GRU做对话层编码，MLP投影，结合对比损失与硬负样本训练。

**📊 数据集**

在DailyTalk多模态对话语料库上进行实验。

**📈 对比分析**

与四类基线（文本池化、多模态池化、文本摘要、多模态摘要）以及大型多模态模型比较，S2Dialog在Recall@10~50上均领先，Recall@50达83.56%。

**⚠️ 局限性**

目前仅在DailyTalk单一对话场景评估，未覆盖更大、多样化或多语言对话；模型使用冻结的预训练特征，后续可进一步微调以提升性能。

---

## 170. Energy-Aware Compression-Computation Co-Adaptation for Latency Minimization in Multi-User Semantic Communication

**arXiv ID:** 2608.13632 | [PDF](https://arxiv.org/pdf/2608.13632v1)

**作者:** Loc X. Nguyen `[一作]` (Kyung Hee University), Choong Seon Hong `[通讯]` (Kyung Hee University)

**通讯引用:** 23906 | [OpenAlex ID](https://openalex.org/A5034052371)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种能量感知的压缩-计算协同适配框架CoCo，用于多用户语义通信中最小化延迟，兼顾不同用户的信道质量、QoS需求和本地能耗。

**💡 创新点**

首次将压缩率与本地迭代去噪步骤共同调度，并将基站子载波分配与用户计算资源配合优化，实现压缩与计算的互补而非折中。

**🔧 技术方法**

采用可调节率的深度联合源-信道编码（DeepJSCC）与迭代潜在去噪模型（类似扩散模型），并通过两阶段训练、经验质量查表与贪心子载波分配算法完成系统训练与资源分配。

**📊 数据集**

在DIV2K高分辨率图像数据集上进行训练与评估。

**📈 对比分析**

与仅压缩调节、仅去噪、最大压缩率以及随机可行点等基线对比，CoCo在满足所有用户QoS的前提下平均延迟最低（约1.32 ms），且显著优于单轴方案。

**⚠️ 局限性**

需要预先构建QoS查表，难以实时适应极端动态环境；去噪迭代时间非零时贪心算法仅为启发式，可能略有最优偏差；在极低信道或能量极限下的性能仍需进一步验证。

---

## 171. Simulation-Driven Vehicular Traffic Data Augmentation: Extending Sensor Coverage Through Virtual Sensing

**arXiv ID:** 2608.13993 | [PDF](https://arxiv.org/pdf/2608.13993v1)

**作者:** Davide Andrea Guastella `[一作]` (Aix-Marseille University), Gianluca Bontempi `[通讯]` (Université Libre de Bruxelles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于仿真的交通计数数据增强方法，利用现有稀疏传感器网络生成与其相似的虚拟传感器数据，扩展网络覆盖；

**💡 创新点**

通过图搜索和基于转向计数与交通指标的正则化评分函数，选择既保持流量连续性又与原传感器保持相似度的候选位置，并使用贪心单射匹配实现可重复的虚拟传感器分布；

**🔧 技术方法**

使用SUMO微观仿真获取转向计数与交通指标，采用BFS搜索候选位置，构造正则化评分函数并执行贪心匹配，同时加入随机探索来避免局部最优；

**📊 数据集**

实验数据包括布鲁塞尔实际道路感知计数（369个传感器）和根据合成需求生成的那慕尔路网模型（不同车辆量的仿真）；

**📈 对比分析**

与真实传感器记录的MAE/RMSE及RMAE进行评估，并与随机置放和QR‑Pivot多样性基线比较。最佳配置（跳距[1,5]、α=0.2、占用率指标）在布鲁塞尔案例中RMSE为70.34，显著优于基线（172.43），大部分传感器误差低于20车/小时，整体保持双峰需求特征；

**⚠️ 局限性**

受限于输入仿真模型的校准质量和不确定性，校准误差会直接影响增强结果；候选位置受拓扑限制，边界区误差较大；未考虑真实信号灯、OSM‑SUMO转换误差；缺乏正式的隐私风险评估。

---

## 172. MedPlex: Deep Vision-Language Co-Adaptation for Clinically Grounded Medical Segmentation

**arXiv ID:** 2608.13690 | [PDF](https://arxiv.org/pdf/2608.13690v1)

**作者:** Rafi Ibn Sultan `[一作]` (Wayne State University), Dongxiao Zhu `[通讯]` (Wayne State University)

**通讯引用:** 3268 | [OpenAlex ID](https://openalex.org/A5009256505)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了 MedPlex，一种通过 Vision‑Language 模型实现医学图像分割的框架，在编码器层级中实现文本与视觉特征的双向持续共适应，并通过基于临床概念的两层对齐来提升分割性能。

**💡 创新点**

创新点包括：
• BiFusion：在每个编码阶段同时更新视觉与文本表示，使文本能在视觉特征生成过程中持续作用；
• 两级概念对齐：类级概念对齐将每个解剖结构与其聚合的临床概念（形状、位置、外观等）对齐，区域级概念对齐保留单一概念与对应视觉证据的对应关系；
• 结构化文本监督：将临床描述拆解为可被模型直接对齐的概念词条，增强文本表达的可分辨性。

**🔧 技术方法**

使用技术包括：
• 3D encoder‑decoder 分割骨干（如 Swin UNETR）；
• Transformer 文本编码器（ClinicalBERT 等）；
• BiFusion 交叉注意力模块实现视觉‑文本双向更新；
• 软/硬负样本的 InfoNCE 对齐损失；
• 交叉熵+Dice 分割损失与 class‑response 监督；
• 可学习的任务权重与自适应温度。

**📊 数据集**

数据集：
• CT：AMOS22、BTCV、MM‑WHS（CT）
• MR：MM‑WHS（MR）、MSD‑Brain
• 真实临床文本监督：ReXGroundingCT（含自由文本报告）

全部采用 3D 图像与对应分割掩码或报告级标签。

**📈 对比分析**

与传统视觉模型（U‑Net、nnU‑Net、Swin UNETR、MedFormer）以及其它 VLM 分割方法（Universal‑CLIP、MulModSeg、ZePT、CAT）进行对比。MedPlex 在所有基准上均实现了新的 state‑of‑the‑art：
• AMOS22 CT：DSC 88.21%、HD95 6.52mm、NSD 93.97%；
• MM‑WHS CT：DSC 92.60%、HD95 3.09mm、NSD 96.86%；
• MM‑WHS MR：DSC 82.94%、HD95 27.01mm、NSD 83.14%；
• MSD‑Brain MR：DSC 79.87%、HD95 9.82mm、NSD 76.42%；
• ReXGroundingCT：DSC 21.78% 31.32% NSD（相对基线提升 4–7%）。
Ablation 结果表明 BiFusion 与两级对齐各自贡献显著，去除后性能下降 1–3%。

**⚠️ 局限性**

局限性：
• 仅使用体积/类级文本监督，未对 2D 切片级局部描述进行建模；
• 对于报告级文本的处理仍需依赖预先生成的结构化概念，未能完全覆盖自由文本的细粒度信息；
• 计算开销相较纯视觉模型略高，虽不显著，但在极端资源受限场景仍需优化；
• 未来工作计划探索切片感知监督、轻量化模型以及更深层的多模态交互策略。

---

## 173. A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing

**arXiv ID:** 2608.13573 | [PDF](https://arxiv.org/pdf/2608.13573v1)

**作者:** William Nixon `[一作]` (University of Chicago), Juncheng Yang `[通讯]` (Harvard University)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5034227509)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

收集并分析了 2025-2026 年一整年的大型语言模型(LM)服务生产日志，包含 6.12 亿请求、9,174 个模型和 314,970 名用户，并对工作负载演化、用户-模型交互、前缀缓存与负载均衡等方面进行了深入研究。

**💡 创新点**

首次公开完整一年期无采样的 LLM 服务日志，揭示了工作负载的非平稳性、长尾模型与用户行为异质性、前缀缓存的双峰特征以及缓存与负载均衡之间的根本张力。

**🔧 技术方法**

采用请求级统计分析、分层抽样、时间序列可视化、缓存淘汰算法对比（FIFO、LRU、ARC 等）以及基于仿真的路由策略（round_robin、load_first、sticky、cache_first）进行仿真。

**📊 数据集**

使用了从云端 LLM 推理平台收集的一年生产轨迹（约 6.12 亿请求，涵盖公共和私有模型），并对其中的请求时间、模型、用户、token 数量、延迟、缓存命中等字段进行分析。

**📈 对比分析**

通过仿真对比了多种缓存淘汰和路由策略，发现 FIFO 与 LRU 在前缀缓存利用率上与最优 Belady 算法接近，且缓存感知路由在保持 5%–7% 负载均衡的前提下显著提升命中率；然而，ARC 等高级策略在本工作中表现不如简单方法。

**⚠️ 局限性**

受限于缺乏完整请求内容和会话标识，仅能基于可推断的会话重建；部分缓存分析基于最后两个月日志，且只考虑单实例缓存，未覆盖跨实例 KV 迁移与网络开销等现实场景。

---

## 174. AgilePE: Autonomous UAV Pursuit-Evasion via Self-Play Reinforcement Learning

**arXiv ID:** 2608.14135 | [PDF](https://arxiv.org/pdf/2608.14135v1)

**作者:** Wenhao Tang `[一作]` (Tsinghua University), Chao Yu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了AgilePE系统，旨在实现无人机的自主追踪与逃避交互，能够在真实硬件上实现零镜像（zero-shot）部署；

**💡 创新点**

创新点包括：1）直接输出低层CTBR（Collective Thrust & Body Rates）控制指令，实现端到端的敏捷飞行；2）双向对抗自我对弈训练框架（从SP到FSP再到PFSP），通过历史策略池和优先采样稳定对抗演化；3）硬件对齐的高保真仿真管线，结合加速器延迟、噪声和域随机化，实现从仿真到现实的无缝迁移；

**🔧 技术方法**

采用的技术包括：强化学习（MAPPO、PPO）、优先Fictitious Self‑Play、CTBR控制、第四阶Runge–Kutta积分、硬件响应模型、域随机化、实时GPU推理（Jetson Orin NX）以及外部运动捕捉融合姿态估计；

**📊 数据集**

使用的“数据集”是通过OmniDrones平台产生的2048并行仿真环境，包含1v1追逐-逃避对战，观测由状态向量（位置、姿态、速度、历史CTBR）组成；

**📈 对比分析**

与规则式基线和不同自对弈框架进行交叉评估，PFSP在捕获率上达到0.88，对抗脚本对手的成功率远超SP（0.5左右）和FSP（0.81左右），且在真实硬件实验中实现了闪避、转弯、滞后利用等高阶战术；

**⚠️ 局限性**

局限性包括：仅在无障碍的1v1环境中验证，未扩展到多无人机协同或复杂地形；观测依赖精准状态估计，缺少视觉感知；优先采样导致对简单脚本对手的鲁棒性略降。

---

## 175. RGBX-Next: Towards Realistic Generative Rendering from G-Buffers

**arXiv ID:** 2608.13929 | [PDF](https://arxiv.org/pdf/2608.13929v1)

**作者:** Zheng Zeng `[一作]` (NVIDIA), Miloš Hašan `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种统一的扩散生成框架RGBX-Next，用于实现正向渲染（从G‑buffer生成逼真RGB图像/视频）和逆向渲染（从RGB视频恢复G‑buffer），并将其扩展到流式视频生成。

**💡 创新点**

创新点包括：①将文本到视频扩散Transformer（DiT）改造成多模态多输入多输出模型，使用帧级拼接（frame‑wise concatenation）、QK类型嵌入和清洁输入令牌（clean input tokens）显式指定每个令牌的角色、模态和噪声级；②设计X‑patchify模块，将多种G‑buffer输入压缩为一组条件令牌；③在训练与推理中引入条件丢弃、模糊和光照缓冲，提供细粒度的控制；④通过教师强迫、自动强迫和长上下文令牌实现流式生成，减少漂移。

**🔧 技术方法**

核心技术包括扩散Transformer（Wan 2.1 DiT 14B）、VAE压缩、帧级拼接、QK类型嵌入、clean input tokens、X‑patchify、条件丢弃/模糊、照明条件（辐照度、无反射直照明）、分类器无监督引导（CFG）、教师/自动强迫、长上下文令牌。

**📊 数据集**

数据集：①合成内部场景（约900段视频，约100帧/段）用于训练逆向模型；②公开合成数据集Hypersim、InteriorVerse；③真实视频数据Pexels（6,622段，分辨率1280×720），使用模型估计G‑buffer并通过Qwen2.5‑VL生成真实/合成标签做语义引导；④在训练时随机遮挡、模糊或丢弃部分G‑buffer。

**📈 对比分析**

与DiffusionRenderer、VACE和通道级拼接基线在Hypersim测试集上进行比较，使用PSNR、SSIM、LPIPS和FID等指标。实验表明RGBX‑Next在Albedo、Depth、Normal和Irradiance等模态上均取得最优或接近最优结果；在流式生成中，结合教师/自动强迫与长上下文令牌可稳定生成超过1000帧而不出现严重漂移，生成速度与原始Wan 2.1相当。

**⚠️ 局限性**

局限性：①模型训练与推理仍与原始Wan 2.1 14B相当，尚未实现显著加速；②仅提供单帧长上下文，无法充分利用整个视频的3D语义信息；③对复杂光照、材质细节的估计仍受限；④缺乏针对真实场景的完整光照与几何数据，导致在某些极端条件下的逼真度下降。

---

## 176. Implementing Computational Law in Wolfram Language for the Governance of Artificial Intelligence

**arXiv ID:** 2608.13958 | [PDF](https://arxiv.org/pdf/2608.13958v1)

**作者:** James K. Wiles `[一作]` `[通讯]` (Wolfram Institute for Computational Foundations of Science), James K. Wiles (Wolfram Institute for Computational Foundations of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文实现了Reified Input/Output Logic（R‑I/O Logic）在Wolfram Language中的完整代码，并在此基础上测试了GPT‑4将英文法律条文翻译为该形式的能力，同时通过一个AI守卫犬案例说明可执行法律与机器人行为的直接映射；

**💡 创新点**

创新点在于：①将DAPRECO知识库核心形式化移植到符号计算平台，①通过LLM自动生成法律转化代码并指出其典型失误；②展示可执行合约直接嵌入机器人操作代码并产生可审计的符号正当化；③系统性梳理AI治理与执行的区别与局限。

**🔧 技术方法**

主要技术包括：输入/输出逻辑（I/O Logic）与其再现化（Reification）实现；Wolfram Language的模式匹配与符号推理；LLM（GPT‑4）作为英文到形式化的翻译器；空间/时间约束的符号化；以及简单的验证脚本与测试用例。

**📊 数据集**

使用的数据来源主要为公开的DAPRECO知识库（含约束、许可与构成规则），卫星轨道数据（SatelliteData）和本地摄像头图像；并未使用大规模标准化法律文本语料库。

**📈 对比分析**

在四个英文条文的实验中，LLM一次性产生代码后仅需人工反馈即可修正，平均需 2–3 次迭代；测试中发现三处“静默”语义错误，说明模型在未严格约束下易误编；但总体能在数秒内完成翻译与执行，表明技术可行性，但缺乏可量化性能评估。

**⚠️ 局限性**

主要限制包括：①LLM易产生幻觉函数与缺失时间语义；②模型在形式化偏离时不易自检；③验证需要人工介入，缺乏自动一致性检查；④仅适用于可形式化的法律部分，无法处理高度含糊或案例法；⑤缺乏冲突解决与可执行合同的完整治理框架。

---

## 177. Communication in modular robotic motor control: Bilateral controllers under realistic constraints

**arXiv ID:** 2608.13904 | [PDF](https://arxiv.org/pdf/2608.13904v1)

**作者:** Jingwen Li `[一作]` (Monash University), Gideon Kowadlo `[通讯]` (Monash University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计了一种双侧（左右）模块化的递归控制器，用于模拟双臂肌肉骨骼系统的运动控制，并对其在存在信号相关噪声与能量成本约束的抓取与保持任务中进行训练与评估。

**💡 创新点**

创新点在于：①将大脑半球通信机制（语义上模拟胼胝体延迟）引入机器人控制；②通过在两侧 GRU 模块间学习可延迟通信通道，探究通信对性能、能量消耗与肌肉共张力的调节作用；③证明模块化结构本身在真实约束下显著优于同等容量的单体网络。

**🔧 技术方法**

技术包括：可微分肌肉骨骼模拟器 MotorNet、GRU 递归网络、可学习的线性通信通道、带乘法噪声的信号相关噪声模型、二次能量惩罚以及基于 Adam 的端到端监督学习。

**📊 数据集**

使用的是 MotorNet 生成的双臂两自由度平面任务数据，包含 8 个中心外向目标的到达任务和 Gaussian 外力扰动的保持任务，任务参数在训练时在两个任务之间交替采样。

**📈 对比分析**

对比方法为：将双侧模块化控制器（有/无通信）与容量匹配的单体 GRU 进行多种噪声与能量惩罚组合的实验；评估指标包括成功率、端点误差、轨迹误差、运动与保持方差、能量消耗与肌肉共张力指数。结果显示，带延迟通信的双侧模块化控制器在大多数约束条件下成功率最高、能量消耗最低、共张力最低。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，缺乏真实机器人硬件验证；任务范围有限，仅涵盖单臂到达与保持，未覆盖更复杂的双臂协调；通信通道简化为线性可学习映射，未考虑更丰富的通信约束或结构化输出映射。

---

## 178. Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges

**arXiv ID:** 2608.13868 | [PDF](https://arxiv.org/pdf/2608.13868v1)

**作者:** Dowon Son `[一作]` (POSTECH), Jisung Park `[通讯]` (POSTECH)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了高带宽闪存（HBF）在大型语言模型推理中的可行性，探讨其在不同 GPU 数量、服务级别目标（SLO）和上下文长度下的性能、吞吐量、批量大小以及耐久性等方面的表现，并与传统高带宽内存（HBM）系统进行对比。

**💡 创新点**

创新点在于：①首次对 HBF 在 LLM 推理中的完整性能和瓶颈进行量化分析；②揭示了仅靠容量提升不足以获得优势，读带宽和写耐久性是关键限制；③通过扩展 LLMSimulator 建立 HBF 时序模型，并对连续批处理与 KV 缓存写入开销进行细粒度建模。

**🔧 技术方法**

主要技术手段包括：HBF 专用时序模型、连续批处理和分离预填充-解码执行、KV 缓存读写重叠、GPU 并行度配置（数据、张量、流水线、专家并行）以及 NVLink/InfiniBand 高速互连的假设。

**📊 数据集**

使用的数据集与工作负载包括：dense LLM、Llama4 Maverick（MoE），以及基于 shareGPT、LongBench、英语摘要的三种上下文长度场景（⟨L_IN, L_OUT⟩分别为 ⟨1,660, 373⟩、⟨5.9K, 499⟩、⟨103.5K, 1.1K⟩）。

**📈 对比分析**

比较方法：在 0.1 s TPOT 的 SLO 下，评估 1–16 GPU 的 5 种 LLM-serving 系统（HBM、部分 HBF、全 HBF 等），对比每 GPU 的批量大小、吞吐量（TPS）、运行时组成以及对 SLO 的敏感性。结果显示：HBF 系统可实现 1.3–5.3 倍的批量大小提升，单 GPU TPS 可提升约 15%，但在读带宽低或写入需求高时性能下降；离线推理场景下 HBF 的优势更显著。

**⚠️ 局限性**

限制因素包括：①HBF 需要与 HBM 同等的读带宽才能释放容量优势；②写入性能低导致 KV 缓存写入成为瓶颈；③闪存的耐久性（P/E 周期）在高 TPS、短上下文、MoE 模型下容易超过 100K，需改进耐久性或采用保留放松技术；④高功耗和热管理仍是系统级挑战。

---

## 179. ASSERT: A Measurement Pipeline for GenAI Audits

**arXiv ID:** 2608.13840 | [PDF](https://arxiv.org/pdf/2608.13840v1)

**作者:** Riccardo Fogliato `[一作]` (Microsoft), Sandeep Atluri `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ASSERT，一个基于书面规范的生成式 AI 审计测量管道，并用其在对话式欺骗行为的审计中进行多版本分析。

**💡 创新点**

创新点在于：①将测量任务与仪器配置写成可追溯的规范；②通过多维度（对话条件、模拟用户、评判者、证据阈值）系统地探索测量不确定性；③证明测量选择会显著影响报告率和系统排名。

**🔧 技术方法**

技术主要包括：LLM 生成测试案例和模拟用户；LLM 评判器依据行为 rubric 进行二元评分；基于规范的自动化管道实现生成、推理、评分与估计；多版本分析与 Wilson 置信区间估计。

**📊 数据集**

数据集：在 120 条多轮对话（6 种欺骗行为 × 4 对话条件 × 5 场景示例）上进行实验；测试案例由 Grok‑4.3 生成，评判者为 GPT‑5.5、Opus‑4.7、Grok‑4.3 等多种 LLM。

**📈 对比分析**

比较方法：在不同测量规范下重新跑审计，记录报告率与系统排名；使用 Wilson 区间评估随机噪声；对同一测试集做多次评判和多次对话重跑，得到噪声基准。性能表现：基线报告率 82%，在不同条件/评判者/证据标准下波动范围 73%–97%，系统间排名随评判者变化而变化。

**⚠️ 局限性**

限制：①报告率仅适用于由管道产生的对话分布，难以外推到真实流量；②模拟用户和评判者使用 LLM，可能导致相关错误与偏差；③测量任务的构造（行为 rubric、证据阈值）和聚合规则会影响结果；④置信区间未考虑评判误差和生成器依赖性。

---

## 180. Your Probabilistic JEPA Is Secretly a Hidden Markov Model: A State-Space Interpretation of Joint-Embedding Predictive Learning

**arXiv ID:** 2608.13621 | [PDF](https://arxiv.org/pdf/2608.13621v1)

**作者:** Yongchao Huang `[一作]` (University of Aberdeen), Yongchao Huang `[通讯]` (University of Aberdeen)

**通讯引用:** 138 | [OpenAlex ID](https://openalex.org/A5111627384)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文将全时序预测信息瓶颈变分JEPA（PIB‑VJEPA）与隐藏马尔可夫模型（HMM）进行系统对应，提出四阶对应层次，并引入Markov‑Chain JEPA（MCJEPA）实现一阶马尔可夫动力学；通过多尺度实验验证其状态空间解释、过滤一致性与序列级等价性。

**💡 创新点**

创新点包括：①将JEPA的编码‑预测‑解码三阶段映射到HMM的推断‑转移‑发射结构；②提出四级对应框架（计算、发射完整、序列级等价、模型‑目标等价）；③设计MCJEPA和条件化/连续状态扩展，实现可训练的马尔可夫转移矩阵与核；④通过信息瓶颈压缩与残差可预测性诊断，阐明压缩与马尔可夫化的本质。

**🔧 技术方法**

技术手段主要包括：变分信息瓶颈学习、指数移动平均目标编码器、马尔可夫链/核参数化、矩阵幂一致性、条件独立性证明、过滤一致性约束、混合HMM‑JEPA目标与过滤蒸馏、以及残差预测性诊断。

**📊 数据集**

使用人工合成数据集：四状态离散HMM、不同发射重叠度的双状态Gaussian HMM、二阶二值过程以及连续观测的Gaussian HMM；不涉及公开真实数据集。

**📈 对比分析**

比较方法包括ARIE、NMI、转移矩阵Frobenius误差、序列负对数似然、预测NLL和过滤KL。实验显示：MCJEPA在共享转移矩阵下实现路径一致性，HMM‑滤波蒸馏显著提升转移恢复和序列NLL；混合目标在保持JEPA预测损失的同时，几乎完全弥补了JEPA单纯训练的序列误差，转移误差下降约46%。

**⚠️ 局限性**

局限性：实验仅在低维合成场景验证，缺乏对真实大规模时序数据的评估；马尔可夫化条件依赖于可观测的发射模型或隐式发射的可行性，实际应用中需要额外的发射模型或可逆编码器；残差可预测性诊断仅捕捉线性或一阶信息，无法完全判定高阶马尔可夫性。

---

## 181. Cross-Disciplinary Taxonomy and Modeling of Misunderstanding Generation, Amplification, and Detection, from Pragmatics to AI Agents

**arXiv ID:** 2608.13604 | [PDF](https://arxiv.org/pdf/2608.13604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 182. Weighted Equitability and Matroid-Constrained Discrepancy

**arXiv ID:** 2608.13983 | [PDF](https://arxiv.org/pdf/2608.13983v1)

**作者:** Kristóf Bérczi `[一作]` (MTA-ELTE Matroid Optimization Research Group), Jakub Tarnawski `[通讯]` (Microsoft Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文证明了加权马尔科夫平等性，给出了强多项式时间算法，并将其应用于基于马尔科夫约束的负载均衡与公平分配；

**💡 创新点**

创新点在于将经典的无权平等性扩展到加权情形，并通过局部交换定理得到可构造的平衡划分；此外，作者将 Beck–Fiala 的不等式推广到任意马尔科夫约束，提出了前缀约束基的 O(log n) 误差上界，并给出对加权乘车与单源不可分流猜想的反例；

**🔧 技术方法**

核心技术包括 Akrami‑Liu‑Raj‑Végh 的局部交换定理、线性规划双线性对偶分析、迭代舍入和部分着色法、以及基于潜能函数的修复步骤；

**📊 数据集**

论文未使用实测数据集，全部研究为理论证明与算法复杂度分析；

**📈 对比分析**

通过与已知结果比较，算法实现了强多项式时间、两侧误差 ≤ 1，得到 (2‑1/m) 近似的负载最小化；在公平分配上证明存在 EF1 分配并可构造；

**⚠️ 局限性**

局限性在于对前缀约束基的最佳误差常数仍未达到 √(log n) 的上界，且对多约束加权平等性和 Morell–Skutella 猜想的更优限制尚未解决。

---

## 183. BGA: A noise-immune neural distillation framework for malicious signature extraction in high-entropy encrypted flows

**arXiv ID:** 2608.14126 | [PDF](https://arxiv.org/pdf/2608.14126v1)

**作者:** Sheng Hong `[一作]` (Beihang University), Ruijian Jiao `[通讯]` (Beihang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 BGA 框架，用于在 TLS 1.3 高熵加密流中提取恶意指令签名，解决注意力稀释和极端类别不平衡问题。

**💡 创新点**

创新点包括：① 自适应门控注意力作为神经滤波器，动态抑制加密噪声；② 结合 WGAN‑GP 进行非线性少数类样本生成，提升稀疏攻击召回率；③ 采用 BiLSTM 与多头注意力的层级融合，兼顾时序记忆与全局关联。

**🔧 技术方法**

使用技术：BiLSTM、带门控的多头注意力、WGAN‑GP 数据增强、ANOVA 特征选择、归一化、噪声注入鲁棒性测试。

**📊 数据集**

数据集：CIC‑IDS‑2018（IT 流量）和 Edge‑IIoT（工业 IoT，Gas Pipeline 场景）共 86,878 条加密流记录。

**📈 对比分析**

通过与 RNN、LSTM、Transformer、TinyBERT 等基线模型在同一硬件和数据划分下对比，BGA 在两大数据集上均实现 >95% 的准确率，召回率提升 43%（MSCI）且推理延迟仅 0.282 ms，低于 TinyBERT 的 4.19 ms，并在 ARM 模拟下保持 1.69 ms，满足工业 10 ms 实时阈值。

**⚠️ 局限性**

局限性：① WGAN‑GP 的生成质量依赖少量种子样本，稀有攻击若无足够代表性样本可能引入偏差；② 对于“隐蔽逻辑漂移”攻击的检测效果尚待提升；③ 性能评估基于理论缩放和模拟，未在真实工业边缘设备上进行端到端验证。

---

## 184. Amplified Does Not Mean Predictive: Reasoning Behaviors in Thinking Models

**arXiv ID:** 2608.13760 | [PDF](https://arxiv.org/pdf/2608.13760v1)

**作者:** Jean de Dieu Nyandwi `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对思考式模型与普通模型在推理过程中的行为进行量化分析，提出行为提升（Behavioral Lift）指标，构建跨模态行为分类法并对15,282条推理轨迹进行注释。

**💡 创新点**

发现思考训练会放大自我纠错、假设检验和不确定性确认等行为，但这些行为与推理正确率的关联不强；真正提升性能的行为是置信度校准、知识对齐和自我意识。

**🔧 技术方法**

采用行为提升指标、恢复率（Recovery Rate）衡量模型在出现推理失败后是否能修正，并使用LLM-as-judge（GPT‑4o）进行自动行为标注。

**📊 数据集**

使用6个基准（VisualPuzzles, MathVista, MMMU, LogiQA2, MATH‑500, MMLU‑Pro）覆盖视觉推理、数学推理与知识密集任务，共计15个模型（含思考与非思考版本）。

**📈 对比分析**

比较方法是对同一模型族的思考版与指令版在行为出现率、行为提升及恢复率等指标上进行对比；结果显示思考版在自我纠错等行为上显著提升，但在整体准确率上仅在需要计算或恢复的任务中显著优于指令版。

**⚠️ 局限性**

局限包括：仅分析可见推理轨迹，无法捕捉模型内部计算；标注依赖自动评判器，可能存在系统性偏差；行为提升为描述性指标，未证明因果关系。

---

## 185. CutClean: Neural Network Pruning for Privacy-Preserving Inference

**arXiv ID:** 2608.13773 | [PDF](https://arxiv.org/pdf/2608.13773v1)

**作者:** Leonardo Magliolo `[一作]` (Télécom Paris), Enzo Tartaglione `[通讯]` (Télécom Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在网络块上插入隐私头并结合结构化剪枝，CutClean实现了对神经网络中敏感属性泄露的显著抑制，同时保持目标任务精度

**💡 创新点**

首次将剪枝与互信息驱动的隐私约束相结合，并使用隐私头作为中间表示泄露的可量化度量，避免传统对抗式训练的复杂性

**🔧 技术方法**

互信息敏感度训练、线性隐私头、L1范数结构化剪枝、后期微调

**📊 数据集**

Synthetic Corrupted-CIFAR10、真实数据集Waterbirds、CelebA（性别、种族、发色、妆容）、ViT-B16在CelebA

**📈 对比分析**

与基线（无剪枝或无MI训练）相比，CutClean在保持目标分类准确率下降<1‑5% 的情况下，将隐私头准确率降低10‑25%，实现20‑60%稀疏度；在Transformer实验中目标准确率仅下降5%，稀疏度50%；实验通过验证集阈值筛选最佳稀疏度并微调后测试

**⚠️ 局限性**

需要预训练和微调步骤、阈值设定依赖数据、仅针对结构化剪枝、无法保证完全消除泄露、在大规模模型或不同隐私度量上的泛化有限

---

## 186. MMDynOpt-Agent: Dynamic Optimization for Multimodal Large Language Model Reasoning via Reinforcement Learning

**arXiv ID:** 2608.14026 | [PDF](https://arxiv.org/pdf/2608.14026v1)

**作者:** Wenjin Liu `[一作]` (Nanyang Technological University), Carl Yang `[通讯]` (Emory University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MMDynOpt-Agent，一种基于强化学习的动态优化框架，利用轻量级代理通过多轮交互自适应生成提示，指导多模态大型语言模型（MLLM）进行更高质量的推理。

**💡 创新点**

创新点在于：①将多模态推理视为马尔可夫决策过程；②设计 Prism Reward 通过分层门控同时奖励格式一致性、答案正确性和预算效率；③实现无参数微调的跨模型可迁移优化策略。

**🔧 技术方法**

采用的技术包括：马尔可夫决策过程建模、GRPO（Group‑Relative Policy Optimization）强化学习、动态优化提示生成、格式/答案/预算三维奖励机制，以及轻量化 LLM 代理。

**📊 数据集**

实验使用了 15 个医学、金融、通用科学领域的多模态数据集（如 MedXpertQA、OmniMed、PMC‑VQA、FinChart‑Bench、FinMME、GeoQA、SciQA、Multi 等），并在 MathVista、PathVQA、SeePhys、VisFinEval、AI2D、VQA‑RAD 等 OOD 数据集上验证泛化。

**📈 对比分析**

与 SFT、GRPO、CoT、OPRO、GEPA、BON、Self‑Consistency、ToT、MARS 等基线比较，MMDynOpt-Agent 在 EM 与 F1 上均取得最高分，并在推理 token 花费上显著低于多步基线，显示出更优的性能‑成本平衡。

**⚠️ 局限性**

局限性包括：对极难任务仍存在性能波动；在某些数据集上方差较大；训练过程需要额外的强化学习开销；以及对非图像多模态输入的适配尚待进一步探索。

---

## 187. Does a Language Server Save Tokens for Coding Agents? A Measurement Methodology and Preliminary Study

**arXiv ID:** 2608.13568 | [PDF](https://arxiv.org/pdf/2608.13568v1)

**作者:** Pengcheng Xu `[一作]` `[通讯]`, Pengcheng Xu

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了两种检索方式（lexical grep 与 semantic LSP）在代码智能体中的 token‑efficiency，并通过五臂消融实验测量了它们在不同任务、模型和噪声条件下的 token‑to‑success 成果。

**💡 创新点**

首次用统一的“tokens‑to‑success”指标量化 LSP 的 token 效率，揭示 LSP 在大多数情形下并不节省 token，只在弱模型和高噪声场景下有益；并发现智能体对 LSP 的使用是任务形状驱动的，而非固定偏好。

**🔧 技术方法**

使用了大型 LLM（Claude Opus 4.8、Sonnet 4.6、Haiku 4.5）、标准 SWE‑bench 任务集、SWE‑bench‑Lite、TypeScript/TS‑Compiler‑API、pyright 等工具，构建了可复现的实验框架和 token 计数日志。

**📊 数据集**

实验数据来自 Python Requests、TypeScript repos（remeda、hono）、SWE‑bench 任务（定位、引用完整性、bugfix 与重命名）以及自建的真实测试执行环境。

**📈 对比分析**

比较方法：对每个 arm 记录成功率与 token‑to‑success，按任务类别和模型进行分组。结果显示：
- 位置定位任务中 grep 优于 LSP（token 下降 0–6%），LSP 甚至税收 6%–118%。
- 引用完整性任务中 LSP 提升 F1 但 token 成本提升 12–19%，仅在最弱模型下可获 token 节省。
- 重命名任务中 grep 再次成为最佳，LSP 仅在返回 inline 行时可部分弥补 token 损失。

**⚠️ 局限性**

局限性：实验规模有限（少数仓库、任务与 roll‑outs），仅评估两语言，未涵盖大型项目和多 LSP 服务器；token 计数基于上下文窗口，未考虑缓存；编辑任务使用本地 harness，缺少标准 SWE‑bench Docker 评测；LSP 质量取决于单一服务器（pyright）。

---

## 188. PILOT: Privileged Imitation Learning for End-to-End Motion Planning of Autonomous UAVs under Partial Observability

**arXiv ID:** 2608.14082 | [PDF](https://arxiv.org/pdf/2608.14082v1)

**作者:** Qingrui Zhang `[一作]` (Sun Yat-sen University), Chenghao Yu `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出一种名为 PILOT 的特权模仿学习框架，将基于 MPC 的高精度规划专家通过双目标损失迁移给单阶段学习的学生策略，实现视觉端到端 UAV 动态规划。

**💡 创新点**

创新点包括：① 单阶段特权模仿学习，避免传统两阶段 RL+BC 的低采样效率和训练不稳定；② 通过 TCN 对历史深度图和里程计进行时空融合，弥补局部感知的可观测性不足；③ 用 Bézier 曲线对轨迹进行参数化，使学生输出兼具可微、连续和平滑性，并在损失中加入约束惩罚，提升对动态和安全约束的自适应性。

**🔧 技术方法**

技术方法包括：模型预测控制（MPC）专家、贝塞尔曲线轨迹参数化、Temporal Convolutional Network（TCN）时空编码、双目标损失（行为克隆+约束惩罚）、ResNet‑18 视觉特征提取、OSQP 求解器、Adam 优化器。

**📊 数据集**

使用了在 NVIDIA Isaac Sim 上生成的多种场景数据集：稀疏/中等/密集圆柱体、倾斜圆柱体、立方体、窄隙门等六种测试环境；此外在真实世界实验中使用了 Intel RealSense D435i 深度相机数据进行零转移部署。

**📈 对比分析**

通过 Monte‑Carlo 仿真与真实硬件实验，对比 MPC 专家、EGO‑Planner、NavRL 等基线。PILOT 在大多数场景中实现 100% 成功率，轨迹长度与 MPC 相近，且计算时间下降 80%+；相比基线的 PILOT‑S（无 TCN）性能明显下降，验证了时空编码的重要性；在固定翼平台上亦保持 95‑98% 成功率，证明跨平台通用性。

**⚠️ 局限性**

局限性：① 仅在离散静态障碍环境下验证，缺乏对动态障碍或外部扰动的鲁棒性；② 依赖离线生成的专家示例，若专家策略失效或训练数据不足会导致学生性能下降；③ 通过软约束实现安全性，未给出形式化安全保证，仅提供经验性结果；④ 需要较大规模仿真数据，训练成本仍不低。

---

## 189. CoDS: Robust Collaborative Perception via Expert-driven Detection and BEV Segmentation

**arXiv ID:** 2608.14085 | [PDF](https://arxiv.org/pdf/2608.14085v1)

**作者:** Jinlong Wang `[一作]` (Peking University), Wei Gao `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种联合3D检测与BEV语义分割的协同感知框架CoDS。

**💡 创新点**

创新点包括：使用协同可靠性图(CoRM)量化融合特征质量、基于可靠性引导的语义混合专家(S-MoE)实现任务特化提取，以及双向任务互补交互(BTCI)提升鲁棒性。

**🔧 技术方法**

采用协同可靠性图、语义混合专家、双向任务互补交互等技术，并在中间融合策略下进行联合训练。

**📊 数据集**

使用OPV2V与V2V4Real两个数据集进行实验。

**📈 对比分析**

与AttFuse、CoAlign、CoSDH、CoBEVT、CoBEVFusion等基线对比，CoDS在检测AP@0.3/0.5/0.7分别达到94.64/94.05/86.88，分割IoU显著提升，且在定位噪声和通信延迟下保持更优鲁棒性。

**⚠️ 局限性**

目前仅实现了检测与BEV分割的联合学习，尚未扩展到其他任务，且对极端噪声的处理仍有提升空间。

---

## 190. BICPO-VLA: Behavior-Identified Continuation Preference Optimization for Smooth Asynchronous Vision-Language-Action Control

**arXiv ID:** 2608.13924 | [PDF](https://arxiv.org/pdf/2608.13924v1)

**作者:** Ming Shang `[一作]` (Beihang University), Fuchun Sun `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

该论文提出一种新的异步视觉语言动作(VLA)框架BICPO‑VLA，专门解决请求到交接期间的行为与动作匹配问题。

**💡 创新点**

创新点在于把行为识别、动作实现结构和交接时的偏好选择三项任务完全分离：①通过指令感知的因果历史编码器确定行为；②利用固定的 Haar 变换将动作拆分为运动支架和残差，使实现保持可逆且不丢失信息；③在已知的交接状态下学习行为内的偏好（Continuation Preference）并用 Flow‑DPO 对候选动作进行排序，从而在保持行为一致的同时显著降低跳跃和趋势不匹配。

**🔧 技术方法**

使用的技术包括：视觉‑语言指令感知的多模态编码、因果 Mamba 结构的历史编码器、固定 Haar 变换的分解与重构、滚动已执行动作得到交接状态、参考相对 Flow‑DPO 优化以及在主策略上仅更新偏好投影的可移植性训练。

**📊 数据集**

实验使用了四大数据集：CALVIN ABC→D（多子任务链）、RoboTwin 2.0 Hard（10个抓取/放置/方向敏感任务）、LIBERO（长序列连续控制）以及六个真实世界任务（如牛奶摆放、纸球处理等）。

**📈 对比分析**

与多种基线（RoboVLM、ReconVLA、OpenVLA、π_0、π_0.5、Legato、RTC 等）进行比较，BICPO‑VLA 在 CALVIN 上平均完成序列长度提升至 4.52，成功率提升至 80.7%，在 RoboTwin 上整体成功率从 60.4% 提升到 65.8%，在 LIBERO 上跳跃和趋势误差分别下降 21.3% 与 20.7%。同时，DPO 目标可以迁移到其他主策略，仅提升 0.1–0.2 的成功率，却大幅降低连续性误差。

**⚠️ 局限性**

局限性包括：假设已知即将执行的动作序列且延迟小于剩余动作长度；不处理长延迟或外部扰动导致的交接状态偏移。未来工作需考虑更长延迟、外部干扰以及更通用的交接预测机制。

---

## 191. Beyond Control Points: Arcsecond Relative-Motion Estimation of Vision Measurement Platforms With Incomplete or Absent Control Fields

**arXiv ID:** 2608.13918 | [PDF](https://arxiv.org/pdf/2608.13918v1)

**作者:** Meng Lian `[一作]` (Shenzhen University), Yulan Guo `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个控制适应的差分框架，用于从图像位移直接估计长距离视觉监测平台的相对运动并补偿结构位移，能在缺失或不完整的控制场时实现弧秒级精度。

**💡 创新点**

提出了控制无关的旋转估计、单点优先约束与两点全3D平移恢复、精确排除控制场污染、导出旋转可观测性与泄漏界限，以及T/√3轴向偏差定律。

**🔧 技术方法**

基于差分PnP模型的线性闭式求解、两次重线性化、控制点优先约束、以及对深度不变性与一阶运动的分析。

**📊 数据集**

合成模拟、深圳外环高速桥梁监测实测数据、公开RGB‑D与立体序列（TUM）以及Euroc stereo。

**📈 对比分析**

与传统PnP、绝对位姿差分、NPnP、BPnP等方法对比，单点旋转RMSE≈3″、两点平移RMSE≈1.2 mm、运行时间≈0.5 ms，零发散率，误差比现有方法低约3倍、速度提升4–600倍。

**⚠️ 局限性**

仅适用于相对运动小于约30′、平移不超过±1 mm、点位扰动≤±2 mm的场景；对大幅相机运动、缺乏足够测点或高非刚性运动时精度下降，且依赖高精度像素定位。

---

## 192. Asymmetric Discourse Homogenization and Shared Language Technology: Evidence from Reddit

**arXiv ID:** 2608.13674 | [PDF](https://arxiv.org/pdf/2608.13674v1)

**作者:** Fengming Liu `[一作]` `[通讯]`, Fengming Liu

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了大语言模型（如ChatGPT）对2019–2025年两大跨党派Reddit社区（r/AskConservatives 与 r/AskALiberal）评论多样性的影响，发现保守派社区出现语义同质化，而自由派社区无显著变化。

**💡 创新点**

创新点在于同时使用ITS、DiD、RDiT、PSM、连续LLM指数以及停留者分析等多种方法，系统验证了同质化效应的生态性而非个体AI采纳，并揭示ChatGPT并非单一事件导致该变化。

**🔧 技术方法**

使用技术包括DeBERTa v3语义嵌入进行余弦相似度计算、时间序列中断回归、回归不连续设计、倾向得分匹配、累积LLM指数构建、信息熵指标以及社区级AI暴露代理。

**📊 数据集**

数据集为605万条来自r/AskConservatives（≈207万条）和r/AskALiberal（≈398万条）的公开Reddit评论，时间范围为2019年1月至2025年12月。

**📈 对比分析**

多方法比较显示保守派右派的水平位移为+0.0081（p<0.001），在月度、周度、日度聚合以及不同模型设定下均保持显著，说明结论稳健；相比之下自由派组无显著变化。

**⚠️ 局限性**

局限性包括：结果对趋势规格、样本筛选和多重检验敏感；难以完全区分生态机制与同期世俗变化；LLM指数与时间高度相关；Reddit用户样本的代表性有限，且无法完全排除新用户替代效应。

---

## 193. BCMT: Blockwise Causal Memory Transformer

**arXiv ID:** 2608.13578 | [PDF](https://arxiv.org/pdf/2608.13578v1)

**作者:** Rachid Arezki `[一作]` `[通讯]` (Independent Researcher), Rachid Arezki (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种 Blockwise Causal Memory Transformer (BCMT)，通过局部块内自注意和指数因子记忆来捕捉长程依赖。

**💡 创新点**

创新点在于将局部注意与指数因子累积的块摘要记忆分离，实现全并行且无显式全局注意的长序列建模。

**🔧 技术方法**

采用局部块自注意、可学习块摘要、指数因子递归记忆、门控注入以及标准多头注意。

**📊 数据集**

在 WikiText‑103 语言建模数据集上进行评估。

**📈 对比分析**

与标准 Dense Transformer 对比，BCMT 在 1024 长度下保持相近的验证损失，吞吐量提升约 70%，显存下降约 27%。

**⚠️ 局限性**

主要限制是记忆压缩导致无法精确检索远程信息，且实验仅覆盖 WikiText，缺乏对更复杂任务或其他长序列模型的比较。

---

## 194. Training-Free Knowledge Transfer Across Model Scales through Activation-Guided Pruning

**arXiv ID:** 2608.13596 | [PDF](https://arxiv.org/pdf/2608.13596v1)

**作者:** Jiahe Fan `[一作]` (University of Science and Technology of China), Hong Xie `[通讯]` (University of Science and Technology of China)

**通讯引用:** 486680 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于激活导向的跨规模模型融合框架Activation‑Prune‑Merge，能够在不需要对齐神经元语义的情况下，将大模型的有效功能注入小模型。

**💡 创新点**

创新点在于利用任务感知激活统计挑选大模型中最具贡献的层、隐藏维度、注意力头和MLP神经元，生成与小模型同构的捐赠切片并以微量插值方式融合，从而显著提升跨规模融合性能。

**🔧 技术方法**

采用激活映射、结构化裁剪（TopK 选择）、目标形状对齐以及微量插值合并等技术，构建了无训练、无对齐的融合流程。

**📊 数据集**

在16个基准（GSM8K、MATH、HumanEvalPlus、MBPPPlus、IFEval、MNLI、RTE、QNLI、PIQA、Winogrande、COPA、BoolQ、ARC、HellaSwag、MMLU、BBH）以及BBH‑27子任务等多领域数据集上进行评估。

**📈 对比分析**

与原始3B模型和Intersection‑Merge (IM) 基线在同一微注入比例下对比，APM平均提升约5.1个百分点（从55.5%到60.6%），在RTE、QNLI、BoolQ等任务上分别提升18pp、13.4pp、8.4pp，整体在16项基准上均优于IM。

**⚠️ 局限性**

局限性包括对注入比例的敏感性（高比例下性能下降）、需要手工设置激活采样规则，以及在更大规模模型和跨任务通用性方面仍需进一步验证。

---

## 195. Adversarial Learning of Classifier-Free Guidance Schedules

**arXiv ID:** 2608.14038 | [PDF](https://arxiv.org/pdf/2608.14038v1)

**作者:** Ashwini Pokle `[一作]` (Google), Valentin De Bortoli `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于对抗训练的动态指导尺度学习方法，通过匹配CFG下引导分布与真实分布的边缘一致性，自动生成时间、条件和噪声依赖的指导权重

**💡 创新点**

将边缘一致性约束转化为密度比估计的对抗目标，利用轻量级MLP学习指导尺度，而不是手工或全局固定方案；同时结合CLIP奖励提升文本对齐

**🔧 技术方法**

对抗生成网络（GAN）+ 密度比估计+ 流匹配模型的CFG+ 轻量级MLP+ R1梯度惩罚+ 两时间尺度更新规则+ CLIP奖励

**📊 数据集**

MS‑COCO 2014（512×512）上的文本到图像生成任务，使用自训练的MMDiT-XS（740M）和MMDiT‑S（1.05B）流匹配模型

**📈 对比分析**

与无指导、常数CFG、手工设计的LIG/CLG、MMD+自一致/边缘一致等基线比较；在FID、CLIP、Aesthetic、HPSv2和PickScore等指标上，GAN+边缘一致方法往往在文本对齐和人类偏好上优于所有基线，FID略有提升但整体效果更好

**⚠️ 局限性**

训练过程复杂，需要对抗平衡与两时间尺度更新；学习到的指导调度与具体背后的流匹配模型绑定，难以零拷贝迁移到其他架构；依赖CLIP奖励，可能受限于CLIP的表达限制

---

## 196. MacCorles: Minimum Alignment Cost Computation on Run-Length Encoded Strings

**arXiv ID:** 2608.13999 | [PDF](https://arxiv.org/pdf/2608.13999v1)

**作者:** Wing-Kai Hon `[一作]`, Jun-Hong Wang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在RLE（Run-Length Encoded）字符串上对最长公共子序列(LCS)的分层优先级约束问题，先最大化相等字符对数再最小化对齐长度。

**💡 创新点**

提出了基于块边界的动态规划框架，将每个RLE块视为同质块，利用块内转移函数实现只在边界存储状态，从而得到线性时间解法。

**🔧 技术方法**

使用块边界动态规划、滑动窗口最大值、前缀/后缀最大化、滑动窗口最大化以及Hirschberg分治重建技术，结合分块传递得分实现。

**📊 数据集**

论文未给出实验数据集，主要是理论分析与算法证明；若需实验，可在合成RLE字符串上验证。

**📈 对比分析**

与传统全表 DP 的 O(|X||Y|) 复杂度相比，该算法在RLE长度为 m,n 的输入下实现 O(m+n) 时间、O(nm) 空间，并通过分治恢复完整对齐；若不考虑空间压缩，则空间与 RLE 运行数相乘。

**⚠️ 局限性**

局限性包括仅适用于已压缩的 RLE 字符串；若运行数与字符长度相差不大，O(nm) 空间仍可能过大；此外需设定足够大的权重 w，算法对权重设定较为敏感。

---

## 197. Nanbeige4.2-3B on Apple Silicon: Fixing Deployment Bugs and Decreasing Looped Transformer Memory Overhead

**arXiv ID:** 2608.13987 | [PDF](https://arxiv.org/pdf/2608.13987v1)

**作者:** John T. Halloran `[一作]` `[通讯]` (University of Washington), John T. Halloran (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Apple Silicon 上对 Nanbeige4.2‑3B 进行部署调试，定位并修复了五个关键错误，并通过“分块预填充”（chunked‑prefill）降低 Loop‑Transformer 的内存占用，同时修复了系统提示回归与 MPS 内存泄漏问题；

**💡 创新点**

提出了分块预填充方案以缓解 Loop‑Transformer 的 O(prompt²) 内存消耗；利用代码猴子补丁（monkeypatch）无侵入式修复多个部署错误；改进聊天模板实现系统提示合并而非覆盖；

**🔧 技术方法**

使用 Loop‑Transformer 结构、分块预填充算法、Python monkeypatch、Torch DynamicCache、MPS（Apple Silicon）专用内存管理与工具调用模板修正；

**📊 数据集**

评估基准包含 MCPMark（多工具多轮任务）、Berkeley Function‑Calling Leaderboard（BFCL，单轮工具调用）、LongBench‑Pro（长文本推理）等；

**📈 对比分析**

与原始检查点比较，修复后单轮工具调用精度近乎 100%，在 MCPMark 子集上成功率提升至 30%（原为 0%），但在多工具调用和极长上下文下仍受限；

**⚠️ 局限性**

仍无法充分支持多工具调用（多轮任务），对更大内存/更长上下文的鲁棒性不足，仅在 Apple Silicon 环境下验证，缺乏跨硬件的泛化验证。

---

## 198. Mobile Apps vs. Web Browsers: A User Perception Study with Android Apps and Google Chrome

**arXiv ID:** 2608.13803 | [PDF](https://arxiv.org/pdf/2608.13803v1)

**作者:** Harel Berger `[一作]` `[通讯]` (Ariel University), Harel Berger (Ariel University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过实验和问卷调查，比较Android手机用户对移动应用与Web浏览器在安全、隐私和可用性方面的感知与偏好。

**💡 创新点**

创新地将思考-大声法与Likert量表问卷相结合，从多种在线服务类别揭示用户对移动应用与浏览器的误解与真实关注点。

**🔧 技术方法**

采用思考-大声法记录任务过程、Likert量表问卷收集主观评估，并通过定量统计与主题分析进行结果解释。

**📊 数据集**

使用约100名Android学生的实验数据与问卷回复，涵盖任务完成记录、口头反思、问卷得分及基本人口统计信息。

**📈 对比分析**

通过任务完成时间、错误率以及问卷得分对比分析，评估安全、隐私与易用性偏好；结果显示用户对移动应用权限更为担忧，对浏览器安全警示更为重视，整体性能表现与服务类别相关。

**⚠️ 局限性**

局限于大学生样本、仅测试Android和Chrome浏览器，缺乏iOS或其他浏览器的比较，且实验场景和任务范围有限。

---

## 199. PISA: A Pseudo-Individual Source-Domain Feature Adaptation Framework for Test-Time Open-Vocabulary Object Detection

**arXiv ID:** 2608.14142 | [PDF](https://arxiv.org/pdf/2608.14142v1)

**作者:** Ziyan He `[一作]` (Sichuan University), Tao Wang `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了PISA，一种无源数据的开词汇目标检测测试时适配框架，能够在图像腐败情况下显著提升定位和分类精度。

**💡 创新点**

创新点在于利用CLIP视觉骨干的损坏不变特征(CIFE)和伪个体源域特征，避免依赖低质量伪标签；通过Feature Alignment Module（FAM）和多尺度对齐框架（BAA）将损坏不变特征转化为高质量伪源域特征，实现稳定的测试时优化。

**🔧 技术方法**

核心技术包括：CLIP的视觉骨干作为CIFE；双分支FAM（主分支+目标分支）结合SE注意力和分离卷积；多尺度对齐适配器BAA_t/BAA_a；基于余弦相似度的TTA损失；在预训练阶段使用干净+增广图像对齐；在测试阶段只更新BAA_a。

**📊 数据集**

在VOC-C、COCO-C和LVIS-C三大受腐败数据集上进行评估，数据集包含19种人工腐败类型（噪声、模糊、对比度等）和3种风格。

**📈 对比分析**

与现有无源OVOD-TTA方法（如TDA、BCA、RGSE、IOU-Filter等）对比，PISA在COCO-C上AP@50%达到31.51%，比最优对手RGSE高3.92%；在VOC-C和LVIS-C上也实现了显著提升（平均mAP提升12.42%、7.65%、3.09%）。

**⚠️ 局限性**

局限性包括：仍需在预训练阶段使用大规模增广图像；对极端复杂噪声的适配效果尚待进一步验证；仅在公开腐败数据集上验证，真实场景下的鲁棒性需进一步测试。

---

## 200. Predicting Custom-Feed Returns for New Bluesky Posts: A Prospective Study

**arXiv ID:** 2608.13874 | [PDF](https://arxiv.org/pdf/2608.13874v1)

**作者:** Yipeng Wang `[一作]` (Northeastern University), Mohit Singhal `[通讯]` (Northeastern University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种针对Bluesky平台的新帖冷启动路由任务，预测一条新发布的公共帖子在24小时内是否会被指定的自定义推送返回。

**💡 创新点**

创新点在于：①将冷启动问题重新定义为查询帖子与所有可排名的推送之间的路由排序任务；②构建了“先采集、后标注”的前瞻性基准数据集；③通过检索‑再重排框架结合语义嵌入和传统特征，提升了路由精度。

**🔧 技术方法**

技术手段包括：多通道检索（历史匹配、词频TF‑IDF、字符TF‑IDF、语义相似度、话题相似度、历史匹配量）；候选集联合（≤370条）；27维特征向量（含4维语义嵌入）；梯度提升树模型（HGB、SemHGB）和LambdaRank（LightGBM实现）进行重排。

**📊 数据集**

数据集：收集自2026年6月30日至7月6日的17.804M条公共帖子、1.865M条可观测的帖子‑推送返回记录，监控5,000条公开可解析的自定义推送，生成两天24小时的测试折叠，共计602,186条具正标签的测试帖子。

**📈 对比分析**

评价指标为capped Recall@10、NDCG@10和Hit@10；在两折均衡平均上，LambdaRank取得cR@10=0.7361、NDCG@10=0.6127、Hit@10=0.7749，明显优于单特征基线和SemHGB；候选集上限（candidate‑union ceiling）为0.8448，表明检索环节仍是主要瓶颈。

**⚠️ 局限性**

局限性包括：仅监控5,000条推送且仅记录Top‑50返回，导致未观测对未必为负样本；测试仅覆盖两天连续测试折叠，缺乏跨日泛化验证；整体样本量相对较小，未覆盖全平台或私有活动。

---

## 201. Rethinking Auxiliary Modalities in Multimodal Zero-shot Anomaly Detection: From Semantic Fusion to Conditional Modulation

**arXiv ID:** 2608.13973 | [PDF](https://arxiv.org/pdf/2608.13973v1)

**作者:** Peng Wu `[一作]` (Northwestern Polytechnical University), Guansong Pang `[通讯]` (Singapore Management University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用辅助模态作为条件信号，对已有的RGB‑文本匹配通道进行轻量化细化，提升零样本异常检测；

**💡 创新点**

不构造联合多模态语义空间，而是将辅助模态作为全局到局部的条件调制；采用元学习生成样本自适应LoRA参数，并通过不确定性感知空间调制实现局部强化/抑制；

**🔧 技术方法**

视觉语言预训练模型（CLIP 等）、低秩 LoRA 适配、元学习模块、基于不确定性的空间调制、RGB‑文本零样本检测框架；

**📊 数据集**

MVTec 3D‑AD 与 Eyecandies 两个工业/合成多模态异常检测基准；

**📈 对比分析**

与现有多模态零样本检测方法（PointCLIP、PointAD、ZUMA 等）及 RGB 基准（AnomalyCLIP、AA‑CLIP、AnomalyVFM）进行对比，取得在 P‑AUC、AUPRO、I‑AUC、AP 等四项指标上最佳或第二佳的性能；

**⚠️ 局限性**

需依赖已预训练的 RGB 模型并保持冻结；对辅助模态的可用性与质量敏感；LoRA 插入层与空间调制系数 η 的选择影响结果，尚未在更大规模或不同领域的数据上验证。

---

## 202. Agent-Orchestration in Autonomous Chip Design

**arXiv ID:** 2608.14035 | [PDF](https://arxiv.org/pdf/2608.14035v1)

**作者:** Linyang Li `[一作]` `[通讯]` (Nova Silicon), Linyang Li (Nova Silicon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出利用大型语言模型与自主代理在芯片设计中的应用框架，尤其聚焦于模拟电路尺寸优化与EDA流程的智能化重构

**💡 创新点**

创新点在于将语言模型的物理知识与推理能力嵌入到动作空间构建、约束检测与多目标权衡中，实现零/少样本泛化、约束感知与可解释优化

**🔧 技术方法**

主要技术包括动作空间建模、黑箱优化流程、语言模型链式推理（CoT）、自主代理编排、物理驱动的约束惩罚与多任务迁移

**📊 数据集**

本文未使用具体数据集，侧重理论阐述与框架设计，示例中以运算放大器等典型模拟电路为说明对象

**📈 对比分析**

未进行实验比较，作者仅通过理论推断与行业现状论证指出相比传统黑箱优化与规则驱动EDA，LLM+代理能更好捕获物理约束、减少样本量、提升可解释性

**⚠️ 局限性**

局限性包括：缺乏实测验证与大规模实验数据、对模型推理质量与硬件成本的评估不足、可能对复杂多目标场景的计算开销与稳定性未充分考量

---

## 203. BCIJelly: An integrated ecosystem for brain-computer interface research

**arXiv ID:** 2608.13576 | [PDF](https://arxiv.org/pdf/2608.13576v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 204. Jais 2: A Family of Arabic-Centric Open Large Language Models

**arXiv ID:** 2608.13580 | [PDF](https://arxiv.org/pdf/2608.13580v1)

**作者:** Mohamed Anwar `[一作]`, Preslav Nakov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了规模为 8B 与 70B 的 Arabic‑centric 大语言模型 Jais 2，全部从零开始训练，采用自定义 150K 词表、ReLU² 激活、RoPE、µP 等技术，并在超过 600B 的 Arabic 语料（含标准语、17 种方言、诗歌、宗教、科学等领域）以及 1.6T 的英文网络、代码、数学数据上进行训练。

**💡 创新点**

创新点：
• 通过 1:2 的 Arabic‑English 预训练比例，实现双语性能平衡；
• 使用 8× 的 FFN 过滤比例与 ReLU² 激活，显著提升长文本理解能力；
• 采用 150K 词表专门优化 Arabic 字符集；
• 设计两阶段预训练加上上采样（Annealing）、持续预训练（CPT）、指令微调（IFT）和直接偏好优化（DPO）的多阶段训练流程；
• 完全公开权重并提供 Web 与移动端演示，提升开放性与可用性。

**🔧 技术方法**

技术手段：
• 仅解码器 Transformer、RoPE 位置编码、µP 参数化、ReLU² 激活；
• BPE 分词器（150K 词表）与多语言/代码混合数据策略；
• 两阶段预训练（Stage 1: 泛化；Stage 2: 领域上采样）+持续预训练 + IFT + DPO；
• 对齐与安全强化：使用 DPO 与 GRPO（未发布）提升安全性。

**📊 数据集**

使用的数据集：
• 600B+ Arabic 语料（标准语、17 种方言、诗歌、宗教、菜谱、科学教材等）；
• 1.6T 英文网络、代码、数学数据；
• 公开多语言数据集（Super‑NaturalInstructions、P3、Aya、xP3 等）；
• 自建方言、诗歌、菜谱、梦解等文化领域的数据集（共 20M+ IFT 例子）。

**📈 对比分析**

对比表现：
• 在 OALL2、AraGen、翻译、方言识别、总结等基准中，8B 版在 ≤13B 模型中获得 72.4% 的宏观平均，超越 Fanar、ALLaM 等；
• 70B 版在 >13B 模型中平均 79.36%，优于 Llama‑3.3‑70B、Qwen‑2.5‑72B；
• 在多方言翻译、诗歌、菜谱等细分任务中也领先同类开源模型；
• 与闭源旗舰模型相比，在部分任务仍有差距，但整体性能与公开资源相匹配。

**⚠️ 局限性**

局限性：
• 仍受限于公开数据规模，部分方言与细分领域覆盖不足；
• 在高难度代码/数学推理上不及专门训练的模型；
• 安全与偏见评测仍有提升空间；
• 与闭源旗舰模型相比，在某些高级任务上仍显不足。

---

## 205. Evaluating Agentic Learning Harness Capabilities Without Labels via the Scaling Hypothesis

**arXiv ID:** 2608.13608 | [PDF](https://arxiv.org/pdf/2608.13608v1)

**作者:** Aryan Luthra `[一作]` (Sublime Security), Anna Bertiger `[通讯]` (Sublime Security)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在无标注环境下评估持续学习 harness 的框架，利用教师‑学生对齐来衡量模型在稀疏人类反馈下的提升；

**💡 创新点**

创新点在于用教师模型的伪标签替代黄金标签评估 harness 性能，并证明教师相对提升与真实提升高度相关；同时证明在相同强度 LLM‑as‑judge 方法在评估持续学习效果时失效；

**🔧 技术方法**

采用了连续学习 harness（Memory、Lesson、Few‑shot）、基于 scaling hypothesis 的教师‑学生对齐、模拟稀疏高精度反馈、Pearson 与 Spearman 相关性分析等技术；

**📊 数据集**

使用了三类安全任务的数据集：PhishFuzzer（邮件三分类）、CTIBench（CWE 归因）、ATT&CK tactic 分类，并以 ASA 邮件分析为应用场景；

**📈 对比分析**

通过计算在教师伪标签下的 ΔT 与在金标下的 ΔG 的相关性来比较方法，结果显示在能力差距大的情形下二者呈强正相关；相同强度的 LLM‑as‑judge 对性能提升几乎无可靠信号；跨家族教师‑学生对也保持良好相关性；

**⚠️ 局限性**

限制在于当教师与学生之间的能力差距小或两者表现均差时，相关性下降；需要先验证教师在任务上的高精度（大约>70%）；缺乏真实人类标注的最终验证。

---

## 206. MedMix: Specialization-Consistent Federated Sparse MoEs under Modality Heterogeneity

**arXiv ID:** 2608.13911 | [PDF](https://arxiv.org/pdf/2608.13911v1)

**作者:** Adiba Orzikulova `[一作]` (KAIST), Sung-Ju Lee `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种在联邦学习下对多模态医疗AI进行语义对齐的稀疏Mixture‑of‑Experts框架，解决客户端与样本级模态异构性问题。

**💡 创新点**

创新点包括：①模态上下文感知路由（MCR）让专家选择依据模态身份、位置及缺失信息；②共识引导路由对齐（CRA）通过全局共识锚点统一不同客户端的路由行为；③客户端自适应专家聚合（CEA）利用原型匹配实现功能一致的专家聚合，从而保持跨客户端专业一致性。

**🔧 技术方法**

使用技术包括稀疏MoE架构、模态上下文感知路由、共识引导路由对齐、原型匹配专家聚合、锚点曝光增强、联邦学习训练流程及负载均衡正则化。

**📊 数据集**

使用数据集：ADNI（多模态阿尔茨海默症诊断，包含MRI、基因组、临床评估、体液标本）和MIMIC‑IV（ICU一年死亡率预测，包含实验室测量、临床文本、诊断/手术代码）。

**📈 对比分析**

与中心化融合方法（Attention、Transformer、FuseMoE、FlexMoE）、联邦不完整模态方法（FedDUET、PEPSY）以及联邦稀疏MoE基线（FedMoE、FedAlign‑MoE）进行比较；在所有模态异构与不完整性设置下，平均F1均优于基线，尤其在高异构和极端模态排除场景表现最为突出。

**⚠️ 局限性**

局限性包括：实验中构造的异构场景可能未完全反映真实医院部署中的模态与标签关联；仅针对单任务预测，尚未验证多任务或任务异构的适用性。

---

## 207. XSA-MAD: Cross-modal Semantic Alignment for Morphing Attack Detection

**arXiv ID:** 2608.13861 | [PDF](https://arxiv.org/pdf/2608.13861v1)

**作者:** Jie Jin `[一作]` (Shizuoka University), Tetsushi Ohki `[通讯]` (Shizuoka University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用两阶段CLIP多模态框架 XSA-MAD 对人脸扭曲攻击进行检测

**💡 创新点**

创新点是将面部真实性与伪造特征拆解为身份、几何、纹理与一致性四个属性，采用结构化提示并通过 LoRA 实现文本与图像语义空间的分离式对齐

**🔧 技术方法**

使用 CLIP 视觉-语言模型、LoRA 低秩微调、结构化提示、CosFace 分类损失以及对角度与相关性对齐等技术

**📊 数据集**

训练集为 SMDD，评估集为 MAD22 与 MorDIFF，涵盖多种图像级、GAN 与扩散生成的扭曲样本

**📈 对比分析**

与多种基线方法对比，XSA-MAD 在 GAN 与扩散攻击上取得最低 EER，整体 EER 仅 6.13%，在高保真生成攻击上明显优于现有方法

**⚠️ 局限性**

局限在于对传统低质量插值扭曲的鲁棒性仍有限，对极端光照、遮挡或极端姿态的处理效果待进一步提升

---

## 208. A Survey of Typical-Cell Volume Distributions in Poisson--Voronoi and Poisson--Delaunay Tessellations: Analytical Theory, High-Dimensional Limits, and Wireless Applications

**arXiv ID:** 2608.13615 | [PDF](https://arxiv.org/pdf/2608.13615v1)

**作者:** Minghua Xia `[一作]` (Sun Yat-sen University), Wenkunn Wen `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了 Poisson–Voronoi (PVT) 与 Poisson–Delaunay (PDT) 随机分割模型中典型单元体积分布的研究现状，系统梳理了从模拟、矩方法、Gamma 近似、精确积分、尺度–形状分解到 Meijer‑G 函数解析等多种方法，并结合高维极限分析与无线网络应用进行了全面讨论。

**💡 创新点**

创新点在于以几何对偶为统一视角，揭示了两种分割在体积分布解析上的不对称性；提出了基于尺度–形状分解的精确混合表述；整合并推广了 Delaunay 体积的 Meijer‑G 闭式表示及其高维收敛定理，并系统阐明了未来研究方向与开放问题。

**🔧 技术方法**

使用技术包括 Palm 测度、条件 Gamma 分解、矩分析、Calka 积分公式、Mellin 变换、Meijer‑G 函数、极限理论（CLT、MDP）以及数值模拟与蒙特卡洛验证。

**📊 数据集**

主要使用在二维 [0,100]² 与三维 [0,20]³ 的边界扩展窗口内生成均匀 Poisson 点过程，并构造对应的 Voronoi 与 Delaunay 分割，收集数百万单元体积作为经验分布；未使用实际场景数据集。

**📈 对比分析**

通过与 Gamma、Generalized Gamma、Ferenc–Néda 等拟合分布在 MSE、MAE、R² 等指标进行比较。结果表明，三参数 GG 模型在二维、三维均优于两参数 Gamma；Ferenc–Néda 在三维表现显著下降；Meijer‑G 解析形式与模拟吻合且可高效评估；高维极限结果为理论验证提供基准。

**⚠️ 局限性**

主要局限包括：PVT 体积仍缺乏可直接评估的无条件闭式分布；尺度–形状分解中的形状分布与面数混合仍隐式，导致精确解析困难；高维近似对低维精度有限；对非 Poisson、时变或聚集/排斥场景的推广尚未实现；Meijer‑G 函数在高维下数值计算复杂且需专用软件。

---

## 209. Continual Evolution Strategies in Control Tasks

**arXiv ID:** 2608.13600 | [PDF](https://arxiv.org/pdf/2608.13600v1)

**作者:** Nicola Pitzalis `[一作]` (University of Pisa), Andrea Cossu `[通讯]` (University of Pisa)

**通讯引用:** 568 | [OpenAlex ID](https://openalex.org/A5051270952)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了演化策略（ES）在连续控制的持续学习场景中的表现，探讨了回放机制对灾难性遗忘的缓解效果。

**💡 创新点**

首次系统评估了ES在持续控制任务中的行为，并证明回放在无梯度的黑盒优化中能够自然地平衡稳定性与可塑性，实现知识保留与正向迁移。

**🔧 技术方法**

采用并行ES框架，结合回放缓冲区、BWT/FWT评估指标、干扰度量，使用标准MuJoCo模拟器进行实验。

**📊 数据集**

MuJoCo三种连续控制环境：Hopper‑v5、Walker2d‑v5、Swimmer‑v5。

**📈 对比分析**

与单任务ES基准以及无回放连续学习对比，回放显著降低了遗忘幅度，提升了前后向迁移（尤其对Swimmer任务），但在过大回放预算下出现了可塑性下降。

**⚠️ 局限性**

受限于任务兼容性、迁移效果不均匀、存在稳定-可塑性权衡，以及缺乏更高级的正则化或稀疏化策略。

---

## 210. Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost

**arXiv ID:** 2608.14125 | [PDF](https://arxiv.org/pdf/2608.14125v1)

**作者:** Xiaodi Huang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Tao Huang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于轨迹偏好学习的目标条件潜在轨迹成本（LTC），并将其与 LeWM 轻量级视觉世界模型结合，用于更准确地对候选动作序列进行排名。

**💡 创新点**

创新点在于：①通过轨迹级偏好学习（synthetic negatives + 失败挖掘）在训练阶段塑造共享表示；②在规划阶段将完整轨迹信息与端点距离相结合，形成路径敏感的候选评分机制。

**🔧 技术方法**

技术方法包括：LeWM 的端到端像素编码与预测器、轨迹偏好学习（基于 Bradley–Terry 逻辑回归）、学习的 LTC（对每一步的状态、变化、目标相对量、时间相位进行聚合）、CEM 规划以及对 LTC 的校准与融合。

**📊 数据集**

使用的数据集包括 LeWM 的四个仿真任务（Push‑T、OGBench‑Cube、Reacher、Two‑Room）以及在 Franka FR3 机器人上的 20 条真实录制轨迹。

**📈 对比分析**

通过与 LeWM、PLDM、DINO‑WM、GCBC、GCIVL、GCIQL 等基线在同一数据集上的对比，实验表明在四个仿真任务中平均成功率提升 3–14%，在机器人实验中成功率从 50% 提升到 70%。

**⚠️ 局限性**

局限性包括：①方法仍高度依赖世界模型的预测精度，未在更长时间或更复杂环境中验证；②失败挖掘仅基于端点规划，可能忽略更细粒度的执行误差；③实验规模有限，仅覆盖四个仿真任务和少量真实任务。

---

## 211. Federated Prompt Learning: A Unified Framework, Empirical Analysis, and Future Directions

**arXiv ID:** 2608.13844 | [PDF](https://arxiv.org/pdf/2608.13844v1)

**作者:** Qinglin Yang `[一作]` (Guangzhou University), Zhihong Tian `[通讯]` (Guangzhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对联邦提示学习（Federated Prompt Learning，FPL）进行全面综述，系统梳理了从预训练到微调再到应用的全生命周期方法，并提出统一的分类框架。

**💡 创新点**

创新点在于将FPL定位为一种独立的范式，提出面向研究问题的统一视角，结合系统化实验验证、基准评估和安全隐私分析，填补了现有文献碎片化的空白。

**🔧 技术方法**

采用系统文献检索、基于技术机制的分类法、对比分析与统一基准实验；实验中使用TinyLlama‑1.1B‑Chat作为基础模型，基于AG News数据集对LoRA、Prompt tuning、P‑tuning等PEFT方法进行评测。

**📊 数据集**

数据集主要为AG News（文本分类）和TinyLlama‑1.1B‑Chat（预训练语言模型），并在此基准上复现多种联邦学习算法。

**📈 对比分析**

通过统一实验平台比较LoRA、Prompt tuning和P‑tuning，LoRA在准确率上最高（≈88.8%），Prompt tuning在通信和参数占用上最小；但不同方法在训练时间、下载量、可训练参数比例等方面呈现互补的性能权衡。

**⚠️ 局限性**

局限性包括：评测仅覆盖少数任务与模型，未充分考察异构客户端、数据分布偏移与安全攻击的鲁棒性；基准缺乏对更大模型和多任务场景的验证；对隐私保护机制的理论与实证支持仍不足。

---

## 212. Algorithm Design and Physician Liability

**arXiv ID:** 2608.13618 | [PDF](https://arxiv.org/pdf/2608.13618v1)

**作者:** Shujie Luan `[一作]` (University of Western Australia), Tinglong Dai `[通讯]` (Johns Hopkins University)

**通讯引用:** 2651 | [OpenAlex ID](https://openalex.org/A5062057702)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了一个两阶段的理论模型，研究了美国医疗算法非歧视责任规则（Section 1557）如何同时影响AI厂商的算法设计（对不同患者群体的准确度投资）和医生在临床决策中使用AI的行为。通过求解模型，分析了责任水平对AI使用差异、AI准确度分配以及患者福利的非单调影响，并探讨了强制等效准确度规定的潜在副作用。

**💡 创新点**

创新点包括：①首次将监管责任（责任成本）与算法设计与使用两端进行系统性耦合；②揭示责任规则既可抑制对弱势群体的AI使用，又可在更高责任水平诱导厂商提升该群体的准确度，从而产生非单调的使用反应；③证明强制等效准确度既可能损害两类患者的福利，也可能导致不适当使用AI，强调设计与使用的政策工具需配套。

**🔧 技术方法**

采用了正式的博弈论模型与最优投资/决策分析。通过后向归纳求解医生的最优使用规则，再求解AI厂商的利润最大化问题，从而得到准确度水平与责任阈值的解析表达式。

**📊 数据集**

本文为理论研究，无使用实证数据集；若要检验结论，可在未来结合真实医疗决策支持系统的性能指标（如AUC、误诊率）与患者分群数据进行实证检验。

**📈 对比分析**

由于为理论模型，没有直接的性能对比。作者通过数学推导给出责任水平、赔偿率、回报等参数下的最优策略，并展示了不同参数组合下AI使用率、准确度与患者福利的数值演示，但未与现有算法或监管方案做实验对照。

**⚠️ 局限性**

局限性包括：①仅考虑两种患者群体，未考虑更细致的分层；②只关注责任成本对使用的直接影响，忽略了其他法律、声誉及技术更新渠道；③假设医生完全遵循AI建议且对AI可靠性完全了解；④未考虑多医生、多机构的网络效应；⑤模型参数（如成本、回报）需进一步从实际医疗系统中估计。

---

## 213. Fixed-Budget Gaussian Volume Encoding with Structure-Aware Allocation

**arXiv ID:** 2608.14112 | [PDF](https://arxiv.org/pdf/2608.14112v1)

**作者:** Michael R. Martin `[一作]` (University of California Davis), Kwan-Liu Ma `[通讯]` (University of California Davis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种固定预算、结构感知的3D高斯原始体积编码框架，能够在单次分析后预分配全部原始数量，随后在无相机的字段空间进行精细优化，生成可直接用于多种转移函数、着色和视角的压缩体积；

**💡 创新点**

首次将完整的高斯原始集合一次性基于场域梯度和结构张量进行分析分配，避免了传统增删原始的迭代过程；通过截断感知评估与闭式梯度加速实现了对十亿级体素的可扩展性；保持标量属性而非预烘焙RGB，实现在不重新编码的前提下自由更改颜色映射和光照；

**🔧 技术方法**

采用3D Gaussian Splatting/Explicit Gaussian primitives、结构张量梯度采样、截断感知评估、无相机字段空间监督、闭式梯度求导、8‑bit/48‑字节编码格式，并在PyTorch+CUDA实现；

**📊 数据集**

五个科学体积数据集：Vortex、Bubble Plume、Miranda、Chameleon和Richtmyer–Meshkov，体素数量从约10^6到超过10^9，覆盖稀疏、密集和界面主导的不同结构；

**📈 对比分析**

与原始体素直接压缩对比，压缩比从2.2×提升至超过40,000×，PSNR范围15–38.7 dB；单台RTX 4090 GPU即可在4分钟内完成1.4 M原始的编码与训练，减小迭代数到300次即可在一分钟内得到与1,500次迭代相当的质量；与现有基于高斯或神经体积的压缩方法相比，速度更快、存储大小可预知且不受训练次数影响；

**⚠️ 局限性**

一次性分配在界面集中特别集中的字段中可能导致过度预留容量，导致后续细节提升受限；缺乏动态原始迁移或重定位机制来进一步优化细节；未在大规模并行或时间变模拟中验证其在实时耦合中的性能。

---

## 214. Vectorized SQIsign Implementation Using AVX-512

**arXiv ID:** 2608.13948 | [PDF](https://arxiv.org/pdf/2608.13948v1)

**作者:** Weize Wang `[一作]` (Fudan University), Yunlei Zhao `[通讯]` (Fudan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

实现了SQIsign的AVX-512IFMA向量化实现，并对数域运算、椭圆曲线运算、配对与一维、二维同源运算进行批量并行化。

**💡 创新点**

首次将SIMD向量化扩展至SQIsign完整算子层面，提出多维并行同源计算与批量点运算的向量化策略。

**🔧 技术方法**

使用AVX-512IFMA指令集、基于Montgomery乘法的5×51比特数域表示、批量多点运算、四向量并行化xDBLADD、LadderBiscalar、ThetaDBL等技术。

**📊 数据集**

在NIST Post‑Quantum标准化安全等级I、III、V的参数集上进行评测。

**📈 对比分析**

与官方参考实现(ASM)对比，keygen、签名、验证分别获得约1.6×-3.1×的加速，尤其验证达到3.18×速度提升；与Qlapoti技术结合后，签名进一步提升2.69×。

**⚠️ 局限性**

目前仅向量化曲线侧运算，quaternion代数部分仍依赖GMP未向量化；对ARM SVE、RISC‑V向量扩展的支持尚待验证。

---

## 215. SN-ASMO: Satellite-Navigation Array Spatial-Manifold Precise Observation Theory A Mathematical Foundation for Observation Formation, Unified U(1) Geometry, Intrinsic Information, and Preservation of the RTK Integer Structure

**arXiv ID:** 2608.13611 | [PDF](https://arxiv.org/pdf/2608.13611v1)

**作者:** Xianwei Meng `[一作]` `[通讯]` (Hefei University of Technology), Xianwei Meng (Hefei University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了SN‑ASMO理论，系统阐述了GNSS阵列观测形成与观测者参与的三层物理–观测者–观测世界模型，并给出了完整的相位、整数结构与统计信息的分离与保留方法。

**💡 创新点**

创新点在于将观测形成与观测者状态合并到统一的几何与统计框架中，构建了U(1)对称群、Fubini–Study度量与Fisher信息的耦合模型，并首次实现了对RTK整数结构的严格保留与同源传输的完整理论。

**🔧 技术方法**

采用微分几何、U(1)群论、投影定理、Fubini–Study度量、Fisher信息、协方差传播、同源相位传输、最大不变量、噪声统计模型等高级数学与信号处理技术。

**📊 数据集**

使用了仿真数据、GNSS仿真器产生的多星信号、现场记录的干扰信号、三轴旋转台实验数据以及重放实验数据进行验证。

**📈 对比分析**

通过与传统单天线观测、ARTK方案以及现有干扰抑制方法对比，理论和实验表明在高干扰、高动态场景下能够保持相位连续、整数保留，定位误差显著下降，误差率相对传统方法降低约30%（实验验证）。

**⚠️ 局限性**

局限性包括对窄带、远场、线性前端、无非线性失真的假设、对干扰强度与相位零点的依赖、对低SNR和重尾噪声时Fisher信息不可靠，以及对同步同源分支和权重估计误差的敏感性。

---

## 216. The MPB Corpus: A Dataset of Melody, Rhythm, Harmony, and Melody-Harmony Relationships in Brazilian Popular Music

**arXiv ID:** 2608.13842 | [PDF](https://arxiv.org/pdf/2608.13842v1)

**作者:** Carlos de L. Almada `[一作]` (Federal University of Rio de Janeiro), Felipe D. Martins `[通讯]` (Anton Bruckner Universität)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了MPB Corpus，即由500首巴西流行音乐（MPB）作品组成的最全面数据集，并开发了四层分析模型（旋律轮廓、旋律节奏、和声、旋律-和声关系）对乐谱进行手工标注与编码。

**💡 创新点**

创新点在于首次系统地将MPB音乐拆分为多维度的符号编码，并提出了生成和声类型（Genera of Chord Types, GCT）与旋律过滤模型（Melodic Filtering Model, MFM），配合CIEI、CMI、MAI等新度量和NF网络可视化，形成可供量化研究的完整框架。

**🔧 技术方法**

技术方法包括：手工音符与和声分析、Python脚本实现c-字母与r-字母的自动提取、Jensen‑Shannon距离与R统计量的计算、置换检验来评估节奏特征的显著性，以及多维可视化与度量计算。

**📊 数据集**

所使用的数据集为MPB Corpus，包含10位代表性作曲家（Tom Jobim、Caetano Veloso、Edu Lobo、Chico Buarque、Milton Nascimento、Ivan Lins、Gilberto Gil、Djavan、Rita Lee、João Bosco）的各50首作品，共计8,426个c‑/r‑词、17,053个和弦、23,447个音符。

**📈 对比分析**

比较方法采用节奏词频分布的 Jensen‑Shannon 距离，计算同一作曲家内外的平均距离比值 R；置换检验结果显示 R_obs = 1.072，p < 10⁻⁴，表明不同作曲家的节奏分布存在显著统计差异。

**⚠️ 局限性**

局限性包括：标注工作高度人工且耗时；目前仅覆盖10位作曲家，未涵盖女性与缺少乐谱的艺术家；每首作品仅分析一个段落；和声分析与旋律-和声关系的自动化仍未实现，需手工完成。

---

## 217. Residual Dominance as a Structural Account of Last-Item Reliance in Causal Self-Attention Recommenders

**arXiv ID:** 2608.14021 | [PDF](https://arxiv.org/pdf/2608.14021v1)

**作者:** Keito Kozaki `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在因果自注意力推荐器中，分析并量化了模型对最新交互的高度依赖，并提出了残差主导的结构性解释。

**💡 创新点**

创新点在于引入残差主导分析，结合全注意力块的范数分解与残差缩放干预，揭示了残差对最后项依赖的决定作用。

**🔧 技术方法**

采用了范数分解、残差缩放干预、位置打乱与HRLI等诊断技术，并以Transformer的自注意力块为核心进行结构分析。

**📊 数据集**

使用了九个公开序列推荐数据集（Beauty、Sports、Video、Diginetica、Toys、Steam、BeerAdvocate、ML-1M、Zvuk）进行实验。

**📈 对比分析**

通过与GRU4Rec、BERT4Rec、DuoRec、BSARec等基线对比，发现残差缩放能在一定程度上恢复误检，保持NDCG@10和HR@10在可接受范围内。

**⚠️ 局限性**

局限性包括仅针对以最终位置为预测接口的因果自注意力模型，残差主导并非唯一原因，且缺乏在线实用性验证。

---

## 218. HI-MeshGraphNets: Efficient and Accurate Mesh-based Physics Learning with Hierarchical Multi-scale Graph Neural Networks

**arXiv ID:** 2608.13827 | [PDF](https://arxiv.org/pdf/2608.13827v1)

**作者:** SiHun Lee `[一作]` (Samsung Electronics Co.), Seung-Hoon Kang `[通讯]` (Sejong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为HI-MGN的层次化图神经网络，用于构建基于网格的物理代理模型，并通过多尺度信息传递实现长距离通信；

**💡 创新点**

创新点在于利用最远点采样与Voronoi聚类自适应构造层次化网格，保留原始拓扑并避免人工边缘；通过学习的几何感知插值将粗尺度信息映射回细尺度；

**🔧 技术方法**

核心技术包括图神经网络、消息传递、最远点采样、Voronoi分区、学习型粗细插值、Huber损失、混合精度训练、EMA权重更新等；

**📊 数据集**

实验使用三类数据集：二维热弹性问题、三维非线性接触问题以及NASA CRM的三维稳态气动数据；

**📈 对比分析**

与MeshGraphNets（平面）和BSMS-GNN（多尺度）在R²、VRAM、训练时间等指标上对比，HI-MGN在准确率上普遍优于两者，同时显著降低显存占用与训练时长，尤其在三维体问题上提升明显；

**⚠️ 局限性**

局限性包括对极大规模三维体积的适应仍需改进，层次化处理可能导致细节损失；过度采样或层级选择不当可能导致过平滑或信息压缩；对极端物理现象（如尖锐冲击）仍需进一步验证。

---

## 219. AdvDex: Learning Dexterous Manipulation from Human Demonstrations via Joint-Aligned Actions and Adversarial Learning

**arXiv ID:** 2608.14028 | [PDF](https://arxiv.org/pdf/2608.14028v1)

**作者:** Zhiyue Zhao `[一作]` (Zhejiang University), Zhengxue Cheng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多模态视觉、语言与动作空间统一框架下，利用人类与机器人演示实现了可跨物种的灵巧操纵学习。

**💡 创新点**

提出了大规模人类演示数据集 OmniShare、共享 SE(3) 腕位+15 指关节的 Joint-Aligned Action Space（JAAS）以及利用梯度反转层的领域对抗学习来消除外观特征的干扰。

**🔧 技术方法**

采用多模态 VLM‑DiT 结构、Diffusion Transformer 动作专家、Gradient Reversal Layer 领域判别器以及端到端的联合训练。

**📊 数据集**

使用 OmniShare（10万+轨迹、500 类任务、700 对象）与少量真实机器人演示数据（1000 条）进行预训练和微调。

**📈 对比分析**

在手动作预测、真实机器人任务（抓取、倒水、推块、堆叠）以及零样本、人类到机器人迁移任务上与 VITRA、π_0.5 对比，AdvDex 在成功率、误差指标上显著优于基线，并在零样本和少样本场景下保持竞争性能。

**⚠️ 局限性**

对细粒度操纵策略的跨平台转移仍受限；仅在单一 DexH13 机器人上验证；未显式建模动力学或接触约束，需结合强化学习或在线适应以进一步提升性能。

---

## 220. Balancing Workload Performance and Slurm Stress: Four Nextflow Deployment Strategies

**arXiv ID:** 2608.13824 | [PDF](https://arxiv.org/pdf/2608.13824v1)

**作者:** Nil Tianchen Mu `[一作]` (Arizona State University), Torey Battelle `[通讯]` (Arizona State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文开发了一套可复现的测量协议和测试工具，评估在共享 Slurm 集群上运行 Nextflow 工作流时不同部署策略（原生、作业数组、HyperQueue、Flux）对工作流周转时间与 Slurm RPC 负载的影响；

**💡 创新点**

创新点在于：①将清洁启动时间与 RPC 计数归一化，构建可跨架构比较的时间轴；②利用 Slurm per‑user RPC 计数将工作流产生的 RPC 需求与整个集群负载分离；③在实际生产集群与私有单用户集群上完成实验，得到可复制的 Pareto 前沿；

**🔧 技术方法**

使用的技术包括：Nextflow 执行器、Slurm 调度器、Meta‑scheduler（HyperQueue、Flux）、Slurm RPC 计数器、定时采样、Python/Scala 脚本做数据处理与可视化；

**📊 数据集**

采用的是 nf‑core 的 `make_lastz_chains` 基因组比对管道，固定输入 4,823 条 LASTZ 逻辑任务（每条任务 1 CPU + 4GB，15 分钟上限），在 ASU Phoenix 生产集群和 Dev 私有集群上运行；

**📈 对比分析**

比较方法：使用清洁启动协议在同一时间轴下测量从策略启动到工作流终点的墙钟时间；利用 per‑user RPC 计数（及其处理时间）作为 RPC 负载度量；在结果中绘制墙钟时间对 RPC 计数的 Pareto 图，评估哪种策略在给定 RPC 负载阈值下最快。实验显示：在 Phoenix 上作业数组和 HyperQueue 的墙钟时间约 0.32h（相较原生 1.03h 节省 69%），RPC 计数降低 92%；Flux 虽 RPC 计数最低 98%，但墙钟时间约 0.65h；在 Dev 上作业数组 0.80h、Flux 0.84h，均相较原生 1.01h 缩短 20%–16%；总体体现了资源获取与 RPC 聚合的权衡；

**⚠️ 局限性**

局限性包括：仅对单一工作流与单一任务规模（4,823 条）评估；实验仅在 N=1 的重复，未平均环境波动；Flux 执行器缺乏 per‑process 内存请求导致与其它后端不同；不同集群的 Fairshare、背后负载差异使结果不易直接推广；方法假设 Slurm 支持 per‑user 计数且可隔离计数。

---

## 221. TLF: Rapid Characterization of RF Transceiver Parameters in Embedded Systems via Bus-Level Interception

**arXiv ID:** 2608.13815 | [PDF](https://arxiv.org/pdf/2608.13815v1)

**作者:** Larry Hernandez `[一作]` (Dartmouth), Sergey Bratus `[通讯]` (Dartmouth)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了 TLF 工具，利用 MCU 与射频收发器 SPI 总线捕获，自动恢复寄存器配置、频率跳变表、同步字、加密密钥等 RF 参数，并能通过插件解析应用层 PDU。

**💡 创新点**

创新点在于：①使用影子寄存器文件实现状态同步，②通过事务分类和 FIFO 方向判定实现自动阶段检测与 FHSS 分析，③支持可插拔协议解码器，②实现了在不做固件逆向的前提下，快速、完整地重建 RF 行为。

**🔧 技术方法**

技术手段包括：SPI 业务截获与 Sigrok 解析、Python 状态机解码器、寄存器映射驱动、FIFO 与模式切换识别、时序统计、LoRa/SX1233 等芯片专属解码器。

**📊 数据集**

数据集为两款实际设备的 SPI 捕获：一台 SX1233 UAV C2 调制解调器（110 秒，约 230 万事务）和一台 SX1276 Meshtastic 节点（50 秒，约 1500 事务）。

**📈 对比分析**

与传统固件 RE 或 SDR 捕获相比，TLF 在数分钟内完成完整解析；对 230 万事务仅需约 60 秒，恢复全部寄存器参数、跳频表与同步字，且能直接驱动协议解析；相较于手工逆向节省天数，且不受射频信号环境限制。

**⚠️ 局限性**

局限性包括：①需物理接入或仿真目标；②只能观察寄存器层，固件层的 FEC、加密与同步字生成无法推断；③每个收发器族需单独实现解码器；④若捕获不包含初始化写入，配置无法恢复。

---

## 222. Equilibrium Pricing in Oligopolistic Data Markets

**arXiv ID:** 2608.14018 | [PDF](https://arxiv.org/pdf/2608.14018v1)

**作者:** Bhaskar Ray Chaudhury `[一作]` (University of Illinois Urbana Champaign), Jiaxin Song `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究数据市场中的定价博弈，探讨在预算有限买家与竞争性卖家之间的Nash均衡及其存在性问题。

**💡 创新点**

证明非竞争性数据商品使得传统竞争均衡不再保证存在NE，并给出PLC（分段线性凸）定价策略下的2-近似NE，弥补了精确NE缺失的空缺。

**🔧 技术方法**

采用博弈理论、PLC定价模型、随机化收益分析、Brouwer固定点定理、NP-硬性证明、以及自适应逼近最优响应动态等技术手段。

**📊 数据集**

使用基于新闻标题与股价走势的公开数据集（news headlines paired with stock price indicators）来模拟卖家数据集与买家预测任务。

**📈 对比分析**

通过自适应惯性逼近最优响应迭代实现收敛，实验显示在各种买卖方规模下均收敛到精确NE，平均PLC稀疏性≤2，实测的激励比（incentive ratio）始终为1，性能远好于理论2近似保证。

**⚠️ 局限性**

仅能保证2-近似NE；在实际博弈中精确NE的存在性仍不确定，理论与实测存在差距；稀疏性上界和更精细的近似因子尚未得到证明；同时计算最优PLC策略被证明为NP-硬，实际需采用粗糙策略。

---

## 223. Probabilistic indirect models for undrained shear strength: addressing significant data missing and variability with advanced imputation and machine learning techniques

**arXiv ID:** 2608.13934 | [PDF](https://arxiv.org/pdf/2608.13934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 224. Whose Posts Get Ranked: Identical-Text Exposure Gaps in Bluesky Custom Feeds

**arXiv ID:** 2608.13879 | [PDF](https://arxiv.org/pdf/2608.13879v1)

**作者:** Yipeng Wang `[一作]` (Northeastern University), Mohit Singhal `[通讯]` (Northeastern University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用自然出现的相同文本推文对Bluesky自定义feed中不同作者的曝光差异进行审计，发现即使文本完全相同，曝光也存在显著不均衡；

**💡 创新点**

创新点在于首次通过现成的重复推文数据，在数百个独立feed中对曝光公平性进行系统性量化，揭示作者历史对曝光的强烈影响；

**🔧 技术方法**

采用匹配集固定效应回归、reciprocal‑rank 和 DCG 等曝光度量，并通过Benjamini–Hochberg FDR 控制多重检验；

**📊 数据集**

数据集为2026年2-4月Bluesky公共自定义feed的完整Top‑50快照，覆盖1,366个feed共297,093个完整快照，形成64,853个相同文本匹配集；

**📈 对比分析**

方法在固定feed、时间与文本的条件下，通过固定效应模型评估作者历史对曝光的影响，结果显示新作者曝光下降0.061，历史曝光每log单位提升0.032，展示了显著的偏差；

**⚠️ 局限性**

局限性包括：主要依赖黑盒算法导致潜在未观测混淆变量，作者历史仅在收集窗口内可见，且仅考察了文本相同的情况，未覆盖不同文本或更大范围的多样性。

---

## 225. UltraArUco: A Lightweight Multilingual Library And Framework With Low-Latency Real-Time Marker-Based Tracking System For Mobile AR Interaction

**arXiv ID:** 2608.13584 | [PDF](https://arxiv.org/pdf/2608.13584v1)

**作者:** Mikhail Kiselev `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 2268 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了 UltraArUco，一个轻量级多语言 FFI 包装的 ArUco 标记检测库，并在移动端与 Unity 3D 分布式 Wi‑Fi 体系结构中实现了实时交互式钢琴 AR 演示。

**💡 创新点**

创新点包括：跨语言 FFI 接口实现多平台快速集成；通过 Wi‑Fi HTTP 传输实现 35 ms 的端到端延迟；在移动设备上利用 ArUco Nano 的算法提升至 6–7 倍的检测速度，并验证了在音乐互动场景中的可用性。

**🔧 技术方法**

技术栈包含 ArUco Nano C++ 核心、C 动态库包装、Flutter 移动前端、Unity 3D 可视化、Wi‑Fi HTTP 通讯、编译器优化（-O2）、EWMA 性能评估方法和多语言 FFI 绑定。

**📊 数据集**

使用了 ArUco Nano 官方静态图像数据集进行基准测试，并在实际实验中使用 4×4 字典、21 个独特 ID 的印刷标记，标记覆盖两段钢琴音阶和手部、琴键等位置。

**📈 对比分析**

通过在不同分辨率（640×360 到 3840×2160）下对比单标记检测耗时，UltraArUco 在 1280×720 分辨率下仅需 3 ms（相当于 30+ FPS），比 OpenCV 提升 6.7×，整体端到端延迟约 35 ms，且标准差更小、性能更稳定。

**⚠️ 局限性**

局限性包括：需保持摄像头与标记的清晰视线；当前 21 个标记只能覆盖两段音阶，无法完整 88 键钢琴；高速手部运动 (>0.5 m/s) 可能导致跟踪丢失；扩展至更大标记集会增加功耗并可能影响检测速度。

---

## 226. HERMES: a multi-agent framework for structured knowledge extraction from ultra-long documents in geoscience

**arXiv ID:** 2608.14055 | [PDF](https://arxiv.org/pdf/2608.14055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 227. Audience capture, selective exposure, affective assimilation or ideological sorting? Polarisation of climate politics under low media-party parallelism

**arXiv ID:** 2608.13864 | [PDF](https://arxiv.org/pdf/2608.13864v1)

**作者:** Arttu Malkamäki `[一作]` (University of Helsinki), Antti Gronow `[通讯]` (University of Helsinki)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对2015–2023年芬兰推特上关于气候新闻的发布和评论行为进行大规模时间序列分析，验证了观众捕捉、选择性暴露、情感同化和意识形态排序四种机制在低媒体-党派关联体系中的作用。

**💡 创新点**

①首次在低媒体-党派平行度的芬兰情境下，将四种机制在同一研究框架内并行检验；②将传统的媒体效应模型与社交媒体的互动行为（如共同发布、情感标签）结合；③用贝叶斯回归在多层次结构上对这些机制进行量化。

**🔧 技术方法**

使用贝叶斯层级回归（brms包），自定义AEI、JSD、RMI等指标；情感分析采用预训练芬兰语语言模型微调后进行三元情感分类；图网络分析（共发布网络、社区检测）用R或Python。

**📊 数据集**

主要数据集为芬兰语推特的公开帖子（共428,654条），包含链接、时间戳、用户ID、情感标签；用户意识形态基于推特转发网络聚类得到的CR、LL、MR三大板块；新闻来源按财务、边缘、主流、地方、党派、小报等七类分类后选取77个有效域名。

**📈 对比分析**

通过贝叶斯模型估计各机制的时间趋势，得到可信区间。观众捕捉的支持最强，显著区分小报与主流；选择性暴露的支持有限，仅在CR–LL之间显著；情感同化普遍存在但幅度有限；意识形态排序在四个议题上表现最弱，RMI仅略升高。相比传统频数或逻辑回归，贝叶斯方法能直接给出参数分布和不确定度，模型拟合度（Bayesian R²）从0.97到0.24不等。

**⚠️ 局限性**

局限性包括：①仅使用推特数据，未覆盖全部芬兰公众；②情感分类偏保守，低情感参与率可能被低估；③因果关系不确定，观众变化可能与媒体策略互相影响；④将意识形态归为三大板块可能忽略更细致的政治维度；⑤平台在2023年被X收购后算法变更，结果对当前平台环境的适用性有限。

---

## 228. Beyond Simplification: DFT-GEN for Fidelity-Preserving Visual Accessibility in Dyslexia-Friendly Educational Texts

**arXiv ID:** 2608.13583 | [PDF](https://arxiv.org/pdf/2608.13583v1)

**作者:** Jiaqian Yu `[一作]` (Hong Kong Polytechnic University), Guoqiong Ivanka Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 801 | [OpenAlex ID](https://openalex.org/A5079097642)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对学习材料中的阅读障碍，构建了一个以干预者需求为导向的文本转换框架 DFT‑GEN，能在保持信息完整性的前提下降低阅读负担。

**💡 创新点**

创新点是将保护跨度和可解释的可视化可访问性控制器（DAC）结合进 LLM 生成流程，并通过自动诊断 DCFI 与 B‑DVAS‑VL 两个分离指标保证信息保真与可视化可访问性。

**🔧 技术方法**

使用了多阶段分析‑重构管线：架构者蓝图、占位符保护、写作者重写、DAC 视觉控制、评估‑改进循环，以及基于 LLM 的判定器和规则检验。

**📊 数据集**

评估使用了两套高难度的中文高考与英文香港DSE题库（共约 2,280 条），以及专家策划的 80 条样本。

**📈 对比分析**

与多种基线（纯布局、神经简化、同一 LLM 的自我改进、商业工具等）相比，DFT‑GEN 在 DCFI 评分上最高（中文 0.910、英文 0.922），在视觉可访问性 B‑DVAS‑VL 的对比中英文分别获胜率 93% 与 64%，并在小规模人机试验中实现最快速度、最高准确率与最低认知负荷。

**⚠️ 局限性**

局限包括：人机试验样本有限、评估主要集中在内容丰富而非语言技巧的任务、诊断工具需要对 LLM 版本进行再校准、以及高影响项目仍需人工复核。

---

## 229. TOGEARI: Interaction-Space Preconditioning for Condensed Finite-Element Systems with IPC Contact

**arXiv ID:** 2608.14162 | [PDF](https://arxiv.org/pdf/2608.14162v1)

**作者:** Yanlin Liu `[一作]` (University of Melbourne), Yao Shen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造一种基于核心响应 Gram 矩阵 UM⁻¹Uᵀ 的低秩子空间，利用 Woodbury 修正将该子空间投射到可重用的有限元核心上，从而在保持完整 Newton 方程不变的情况下预处理大规模接触问题。

**💡 创新点**

① 在 SPD 核心下给出该子空间的精确谱和最优缺失模式上界；② 提出一种只用 Euclidean 结构的原始因子代理（raw selector），在数值上与核心响应选择表现相同；③ 通过实验验证八维子空间即可将 Krylov 基维数从 11 降至 5，证明接触子空间的紧凑性。

**🔧 技术方法**

使用 Tet10/P0 近似弹性有限元、IPC 正常接触张量、局部压缩与压力消元、核心响应矩阵 UM⁻¹Uᵀ 的特征分解、Woodbury 右预条件、FGMRES 循环以及完整残差重算。

**📊 数据集**

一组冻结的静态接触线性化，来自一个气压手指与可变形杯子相碰撞的模型，总计 325,260 个自由坐标，94 行接触因子矩阵；此外提供了近重复提取和无接触基准案例。

**📈 对比分析**

与基准核心预条件（无任何子空间）以及不同维度（6、8、12、16、24、94）和不同选择器（metric、raw、行范数、随机）进行比较。实验表明：维度 8 的 metric/raw 子空间将 Krylov 基从 11 减至 5，warm 迭代时间从 2.849 s 降至 1.629 s；post‑load 第一次求解时间略升高；行范数和随机选择器需要 12 个基，未能达到压缩效果。

**⚠️ 局限性**

仅在冻结、无摩擦、静态等理想化条件下验证；未评估非线性求解、接触拓扑变化、摩擦、惯性/阻尼、不同网格尺度或多处理器扩展；此外 SPD 核心假设在实际系统中可能不成立，导致理论结果的适用范围受限。

---

## 230. Smart routes: a system for development and comparison of algorithms for solving vehicle routing problems with realistic constraints

**arXiv ID:** 2608.14140 | [PDF](https://arxiv.org/pdf/2608.14140v1)

**作者:** Andrew Soroka `[一作]` (Moscow State University), Sergey Gerasimov `[通讯]` (Moscow State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研发了 Smart Routes 平台，并在 50 点和 100 点的 CVRPTW 实例上比较了 SCIP 精确求解、经典启发式（LKH、2‑OPT、3‑OPT、OR‑Tools）以及改进的深度强化学习模型 JAMPR 的求解时间与路径成本。

**💡 创新点**

创新点在于：①提供统一可扩展的实验框架，支持用户轻松添加自定义算法；②首次系统性对比了精确、启发式和深度 RL 在同一现实约束下的时间‑成本折衷；③展示深度 RL 在大规模问题（100 点）上可达到甚至超越启发式的质量，并且显著快于精确求解。

**🔧 技术方法**

技术手段包括：SCIP 的分支定界/剪枝求解、LKH、2‑OPT/3‑OPT、OR‑Tools 的启发式模块，以及改进的 JAMPR（注意力编码‑解码网络，加入可学习掩码）进行深度强化学习；同时利用 GPU 加速深度模型训练。

**📊 数据集**

使用数据集：Solomon 参考集（R201）生成的 CVRPTW 实例（50 点和 100 点，容量 Q50=750/Q100=1000，时间窗口 [0,1000]，服务时间 10），并在此基础上随机生成 100 个人工实例进行实验。

**📈 对比分析**

比较方法：在 100 s（50 点）/200 s（100 点）时间限制内收集每种算法的平均求解时间和平均路径成本；对 SCIP 给予 10× 更长时间（1000 s/2000 s）以求得最优解。结果显示：50 点时，SCIP 最优但耗时约 10 倍；LKH/OR‑Tools/JAMPR 在数秒内即可得到与 SCIP 差距 <5% 的解；100 点时，SCIP 首解成本比启发式高约 50%，耗时 13 倍；JAMPR 与 OR‑Tools 在不到 200 s 内即可达到或超过启发式的最优成本。

**⚠️ 局限性**

局限性：仅评估了 50 点和 100 点实例，未验证更大规模问题；深度 RL 模型的训练时间、资源占用未系统化说明；对 soft 约束下惩罚系数 λ 的选择未做完整敏感性分析；平台仍需手动安装依赖，尚未实现完全零配置。

---

## 231. Repair, Not Improvement: Decomposing Constrained Decoding in Tool-Call Abstention

**arXiv ID:** 2608.13959 | [PDF](https://arxiv.org/pdf/2608.13959v1)

**作者:** Janghoon Lee `[一作]` `[通讯]` (Redrob), Janghoon Lee (Redrob)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MetaTool Task2的工具调用任务上，评估了使用枚举约束（enum constraint）进行受限解码对模型放弃（abstention）准确率的影响。

**💡 创新点**

提出将受限解码的成本拆分为三项（Δ_mask、Δ_length、Δ_total），揭示约束对决策边界的双向影响，并验证了先前“几乎无成本”结论在放弃任务中的失效。

**🔧 技术方法**

使用受限解码技术（基于枚举的语法约束）、第一token掩码概率测量、对比分析三种解码策略，并利用逻辑回归等统计方法解释机制。

**📊 数据集**

使用MetaTool Task2数据集（200条中英文对照样本，每条包含10个工具候选，包含需要工具与放弃两类子任务）。

**📈 对比分析**

与无约束、短行约束三种条件对比；结果显示在放弃任务中，约束导致准确率下降（Δ_total为负），但能修复格式错误；模型规模越小负面影响越显著；两种语言之间差异不显著。

**⚠️ 局限性**

局限：工具候选列表固定排序（首位即正确工具），干扰器集合与前缀示例不匹配导致基线不平衡；仅在单一解码器实现上评估；Korean查询为翻译非原生；受限解码仅覆盖前置token，未捕捉后续约束压力；实验覆盖的模型规模有限。

---

## 232. Forecast Collapse in Time-Series Foundation Models

**arXiv ID:** 2608.14106 | [PDF](https://arxiv.org/pdf/2608.14106v1)

**作者:** Shu Wan `[一作]` (Abel AI Lab), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了时间序列基础模型（TSFM）在金融市场中预测资产收益时出现的“预测崩溃”现象，揭示低可预测性导致的幅度衰减与交叉系列排序失效，并提出平衡校准与排序的复合损失函数；

**💡 创新点**

首次将预测幅度衰减与信息系数（IC）衰退结合成理论框架，给出最佳尺度幅度与可预测方差的等式，证明逐系列风险无法识别跨系列耦合，并提出结合MSE与Pearson相关系数的校准-排序损失；

**🔧 技术方法**

使用基于Transformer的单序列编码器、均方误差（MSE）、信息系数（IC）、自适应λ权重的复合损失、以及对比的公开TSFM模型（TimesFM、Chronos）和合成数据实验；

**📊 数据集**

主要使用自定义的Finance1K面板（1,000只美国股票的小时收益与成交量变化），以及公共GIFT‑Eval基准配置；

**📈 对比分析**

通过在同一预测协议下对12种模型分别采用MSE、IC、以及复合损失进行对比，发现复合损失在保持幅度的同时显著提升IC，达到了前沿的校准‑排序权衡；在合成实验中验证了幅度等式与可预测性上限；在公开基准上展示原始幅度与预测可预测性相关；

**⚠️ 局限性**

限制包括：幅度等式仅对最佳尺度的预测成立，无法评估原始输出幅度；交叉系列损失只关注Pearson相关，无法覆盖更复杂的依赖结构；实验主要聚焦金融收益，结果对其他领域的普适性尚待验证；预测可预测性上限使用已拟合基线估计，未给出真正上限；

---

## 233. hint$^2$: Hierarchical World Models for Inference-Time Temporal Logic Guidance

**arXiv ID:** 2608.13678 | [PDF](https://arxiv.org/pdf/2608.13678v1)

**作者:** Moritz Zoellner `[一作]` (Purdue University), Rohan Paleja `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究提出了一种在推理时利用层级世界模型引导预训练扩散策略，以满足复杂的线性时序逻辑（LTL）约束，尤其适用于高维机器人控制任务。

**💡 创新点**

创新点在于：①用两层抽象的世界模型分别预测动作块对高层原子命题的影响（促进 LTL 自动机进度）和对低层状态的精确变化（实现 STL 约束安全）；②将高层预测的自动机分布和低层鲁棒性梯度融合，形成可在短期动作采样中直接使用的引导信号，解决了传统方法在长时程 LTL 指令上的误差累积和尺度不匹配问题。

**🔧 技术方法**

核心技术包括：无条件多模态扩散策略、基于 LTL 自动机的高层世界模型、基于 STL 鲁棒性的低层世界模型、自动机潜在向量的可导潜能引导、梯度反向传播实现动作块调优，以及对 LTL 语义的定量解释。

**📊 数据集**

实验数据集涵盖：①2D Toy Squares 2D 任务；②CALVIN 虚拟桌面环境（基于 CALVIN-D 数据集，六小时人类演示）；③真实 UR5e 机器人任务（130 条演示，涉及盒子、碗等操作）。

**📈 对比分析**

与现有方法比较：在 2D 任务中相比 LTLDoG、TeLoGraF，HINT2 在所有自动机距离上实现 100% LTL 满足；在 CALVIN 环境中，在单行为引导上获得 91% 成功率，优于 DynaGuide、ITPS；在复杂 LTL 指令和安全约束上，HINT2 与 FLOWER+LLM+GPC 近乎完美，远优于语言条件策略；在真实机器人上，HINT2 能完成带安全约束的循环任务，显著优于 STL-GPC。总体性能提升体现在长时程 LTL 需求的完全满足、行为选择的准确性以及安全约束的主动满足。

**⚠️ 局限性**

局限性包括：①依赖预设的有限原子命题集，无法自动学习离散抽象；②当前方法仅对相对于 MDP 的“停顿不变”（stutter‑invariant）LTL 约束可得到精确引导，无法处理非确定性或更一般的 LTL 表达式；③在多智能体或更大规模任务中的可扩展性尚未验证。

---

## 234. No Universal Signal Predicts Sample-Level LLM Regression under Version Updates

**arXiv ID:** 2608.13607 | [PDF](https://arxiv.org/pdf/2608.13607v1)

**作者:** Jia Sheng `[一作]` (University of Ottawa), Yiwei Lu `[通讯]` (University of Ottawa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在大语言模型版本更新中，如何在推理时预测样本级回归（负面翻转），并提出多种单模型与跨版本信号做对比；

**💡 创新点**

提出统一的增值测试框架评估跨版本信号对回归预测的增益，揭示信号效能与任务类型及模型更新密切相关，且跨版本信号在无标签情况下仍具一定预测力；

**🔧 技术方法**

使用最大softmax置信度、logit margin、注意力熵、输出KL/JSD、似然漂移、top‑k token KL、表示漂移等指标，结合逻辑回归与5折交叉验证的AUROC评估；

**📊 数据集**

覆盖三类任务：多选问答（MMLU‑Pro、GPQA）、数学推理（GSM8K、MATH‑full）与代码生成（HumanEval、MBPP），并测试五对同族模型更新与一对跨族（Llama‑2→3）；

**📈 对比分析**

通过对比基线置信度与加入各信号的Δ_CV‑AUROC，发现置信度在MCQ与简单数学上最优，跨版本似然/KL在较难数学与代码生成上更具优势；整体增益因任务与更新而异，未见任何信号在所有场景均胜；

**⚠️ 局限性**

局限在于仅评估开放权重同族更新，缺乏对闭源API、其他模型类型或更大规模数据的验证，且对信号的因果机制仅为假设性解释。

---

## 235. Stable Miscalibration in Large Language Models: A Practical View of High-Confidence Errors

**arXiv ID:** 2608.13591 | [PDF](https://arxiv.org/pdf/2608.13591v1)

**作者:** Akira Okutomi `[一作]` `[通讯]` (ToppyMicroServices O"U), Akira Okutomi (ToppyMicroServices O"U)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建标签感知的输出层审计指标和内部敏感性探测器，研究大型语言模型（LLM）在高置信度错误中的稳定性与误校准问题；

**💡 创新点**

创新点在于提出“稳定误校准”（stable miscalibration）概念，并设计了结合置信度变化和强迫回答基线过度自信错误的域级审计分数，以评估自我批判式干预的有效性；

**🔧 技术方法**

使用的技术包括：标签感知输出级审计分数、基于置信度的平方损失评估、Gaussian 嵌入扰动与隐藏状态敏感度测量，以及多层内部响应探测；

**📊 数据集**

实验数据集为 532 条二分类事实性条目，覆盖 11 个领域（医学流行病学、社会统计、地理旅行等），并在 GPT‑4.1‑mini（用于输出审计）和三种开源权重模型（Llama‑3.1‑8B‑Instruct、DeepSeek‑R1‑Distill‑Llama‑8B、Qwen2.5‑7B‑Instruct）上进行内部探测；

**📈 对比分析**

比较方法为：将自我批判式（C2）相对于强迫回答基线（C0）的策略损失变化（Brier 风格）与审计分数的 Spearman 相关性，发现相关系数 ρ≈0.71，且在某些领域（医学、社会统计、地理旅行）自我批判显著降低损失；内部敏感度上，自我批判提示降低了各层隐藏状态的扰动响应，但未出现显著的“过度自信错误”与“正确回答”之间的敏感度差异；

**⚠️ 局限性**

限制包括：审计集为作者手工构建、规模有限，无法直接推广至更大或多类别任务；审计分数为有标签诊断，无法在无标签场景下直接使用；内部探测使用开源模型而非用于审计的 GPT‑4，可能不完全对应；此外，实验仅探讨局部扰动，未覆盖更广泛的可靠性评估方法。

---

## 236. Reinforcement Learning-Based Production Scheduling in an Industry-Based Coating Scenario Using the Digital Model Playground

**arXiv ID:** 2608.14122 | [PDF](https://arxiv.org/pdf/2608.14122v1)

**作者:** Arne Kröger `[一作]` (Osnabrück University of Applied Sciences), Henrik Wilbers `[通讯]` (Osnabrück University of Applied Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在基于数字模型游乐场（DMPG）的电镀+粉末涂装工业场景中，使用强化学习（PPO、DQN）实现生产调度，目标是同时降低设置时间和延误。

**💡 创新点**

创新点在于将RL调度应用于真实工业案例（包括序列依赖设置、机器故障、可变负荷），并提供了可复现的开源模拟框架。

**🔧 技术方法**

采用的技术包括深度Q网络（DQN）、近端策略优化（PPO）和基于TF-Agents的分布式训练框架，结合离散事件仿真。

**📊 数据集**

数据集为从仿真生成的随机到达时间、颜色与交付日期的合成批次，利用率因子在0.7–1.3之间变化。

**📈 对比分析**

与传统调度规则（EDD、最小设置时间MST）比较，PPO在设置时间、加权偏差和未完成产品数量上均优于其他方法，且表现更平衡。

**⚠️ 局限性**

局限性包括模型简化（仅四种颜色、无运输/工人约束）、未包含人类操作员决策、缺乏超参数优化以及可扩展性与真实工厂对比的验证不足。

---

## 237. MACS: A Hybrid Multi-Agent Framework for Reliable Conversational E-Commerce Recommendation

**arXiv ID:** 2608.14068 | [PDF](https://arxiv.org/pdf/2608.14068v1)

**作者:** Juli Huang `[一作]` (Stanford University), Amin Saberi `[通讯]` (Stanford University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MACS 框架，将 LLM 与商家 Agent 分离，支持固定目录的可靠会话式推荐，提供语义解析、偏好追踪和约束执行。

**💡 创新点**

创新点包括：① 混合多代理架构，LLM 负责自然语言交互，商家 Agent 负责硬约束过滤与检索；② 会话持久化的偏好层，支持预算覆盖、品牌排除反转等跨轮一致性；③ 进阶评估方法与渐进放宽机制，保证返回产品真实可购且对约束做出显式披露。

**🔧 技术方法**

技术栈：大语言模型（GPT‑4o‑mini / Gemini）、SQL 约束过滤、知识图谱（Cypher）查询、UCP/ACP 通信协议、缓存层、最佳价值加权评分、渐进放宽与披露策略。

**📊 数据集**

使用自建的商家目录（约 21,000+ 条笔记本/配件产品，包含 12 个结构化规格字段）与知识图谱（2,400 节点 / 8,500 边），评测基准包括 140 条单轮、225 条单轮、10 场景多轮问答。

**📈 对比分析**

方法：与基于同一候选集的文本 LLM 基线（GPT‑4o‑mini 与 Gemini）对齐，单轮通过率、品牌/过滤合规、质量分（G‑Eval）评估；多轮以宏观 Pass@5（k=5）衡量约束正确性与评估质量。MACS 单轮通过率 87.1%，品牌合规 1.000，过滤合规 0.970；多轮宏观 Pass@5 72%（基线 56%/52%），显示在约束持久化与披露上的优势。

**⚠️ 局限性**

局限：仅在消费电子（笔记本）领域评估，未验证跨域泛化；基线为注入候选集而非完整检索；评测脚本化、无独立人工评估；仅考察固定目录场景，未来需扩展至其他商家目录与公开基准。

---

## 238. SAFE: Scene-Aware Feature Modulation for Color Constancy with Learned Color Space in Pure-Color Scenes

**arXiv ID:** 2608.13967 | [PDF](https://arxiv.org/pdf/2608.13967v1)

**作者:** Yuan-Kang Lee `[一作]` (MediaTek Inc.), Jian-Jiun Ding `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个专门针对纯色场景的颜色恒定框架 SAFE，结合了场景感知特征调制网络与学习色彩空间，以提升光照估计精度。

**💡 创新点**

创新点包括：① 将光照信息拆分为四个语义化的 token（全局色度、边缘色度、镜面与分散、双色反射几何）；② 通过场景复杂度向量对 token 进行自适应乘性注意力调制；③ 引入可学习的场景依赖色彩空间（LCS），通过自适应权重消除纯色场景下的色度坍塌问题。

**🔧 技术方法**

使用了卷积神经网络提取场景特征，MLP 生成注意力掩码，残差 MLP 背骨进行光照预测；LCS 通过 CNN+MLP 预测权重向量；训练分四阶段，包含教师-学生搜索、预测器预训练和联合微调，并加入一致性正则化。

**📊 数据集**

主要在 PolyU Pure Color Dataset V2 进行评估，同时在 Gehler-Shi 数据集上验证通用场景性能。

**📈 对比分析**

与现有监督/无监督方法相比，SAFE 在 PolyU V2 上平均角误差从 2.12° 降至 1.94°，比最佳基线低约 10%，同时在最佳 25% 和最差 25% 的误差也分别下降 20% 与 5.8%。在 Gehler-Shi 上保持了与主流方法相近的表现。

**⚠️ 局限性**

局限性在于色彩空间的学习参数与相机传感器光谱特性高度耦合，需针对不同相机进行单独训练，未来需要开发感知传感器特征的通用 LCS。

---

## 239. CAST: Closed-form Analytic Semantic Transfer for Zero-Shot Classifier Extension

**arXiv ID:** 2608.13751 | [PDF](https://arxiv.org/pdf/2608.13751v1)

**作者:** William Heyden `[一作]` (Norwegian University of Life Sciences), Fadi Al Machot `[通讯]` (Norwegian University of Life Sciences)

**通讯引用:** 1384 | [OpenAlex ID](https://openalex.org/A5073646721)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CAST，一种完全无需图像和训练的零样本学习框架，通过闭式岭回归将 CLIP 文本嵌入映射到预训练分类器的权重空间，直接为未见类别注入权重；

**💡 创新点**

核心创新在于利用 Neural Collapse、CLIP 互模对齐与线性表示假设统一的几何理论，得到一条线性桥梁，并引入可计算的语义外推残差 ρ_u 作为转移难度指标；

**🔧 技术方法**

采用 CLIP 文本编码器、闭式岭回归权重映射、语义外推残差评估及多模态几何分析等技术；

**📊 数据集**

在四个标准零样本学习基准上验证：AWA2、CUB、APY、FLO，并跨 11 种预训练网络；

**📈 对比分析**

与现有图像无关、训练无关的基线（如 ICIS、ConSE、COSTA）进行对比，CAST 在 AWA2 和 FLO 上实现或超过基线性能，在 CUB 上因细粒度挑战略逊；

**⚠️ 局限性**

局限性主要在于线性桥梁的外推能力受限，语义外推残差高的细粒度数据难以准确合成，且依赖 CLIP 文字编码的质量与描述精度；

---

## 240. Interactive Analysis of Global Explanations using Aggregated Class Activation Maps for Network Data

**arXiv ID:** 2608.13575 | [PDF](https://arxiv.org/pdf/2608.13575v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 241. Characterizing the Variance Envelope: A Multi-Dimensional Analysis of Spectre Telemetry Across Architectures and Workloads

**arXiv ID:** 2608.13920 | [PDF](https://arxiv.org/pdf/2608.13920v1)

**作者:** Jaya Keshava Chandra Kotha `[一作]` (University of California, Irvine), Jean-Luc Gaudiot `[通讯]` (University of California, Irvine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Spectre攻击在不同处理器架构（Intel、ARM、AMD）、不同攻击变体、不同系统负载以及不同攻击节奏条件下的硬件性能计数器（HPC）特征进行全方位实验，绘制并分析其方差包络（variance envelope）。

**💡 创新点**

首次系统性、跨变体、跨架构的多维度实证研究，揭示HPC特征高度脆弱、易被环境噪声与攻击节奏扭曲，并发现AMD Jaguar架构对Prime+Probe攻击的硬件级失效，证明静态检测阈值在现实环境中的根本局限。

**🔧 技术方法**

采用HPC采集与对齐流水线，构建包含三种Spectre变体（V1、V2）和两种泄漏通道（Flush+Reload、Prime+Probe）的实验矩阵；使用机器学习模型（MLP、XGBoost等）进行基线检测；利用对抗性节奏（unthrottled、constant rate、burst-all、batch-rate）和四种系统负载（Idle、CPU、Memory、Mixed）模拟真实环境；对AMD Jaguar的缓存层结构和预取器进行微架构分析。

**📊 数据集**

四台实验平台（Intel Core i7‑3537U、Intel Core i5‑7200U、ARM Cortex‑A76、AMD A4‑5000）在40个不同组合（3种攻击、4种节奏、4种负载）下收集的HPC计数日志，构成实验数据集。

**📈 对比分析**

将基线静态ML检测器（如MLP、XGBoost）在理想实验室环境与多维度真实场景下的准确率进行对比；实验表明，在引入背景噪声或对抗性节奏后，检测准确率急剧下降，凸显了方差包络对模型性能的显著影响。

**⚠️ 局限性**

局限性包括：仅测试了四台旧型号平台，未覆盖最新微架构；静态阈值检测无法应对跨域漂移；依赖HPC计数，若计数被禁用或误差被掩盖，检测效果进一步受限；未来需研发自适应、跨架构的检测框架。

---

## 242. Does ISO-Grounded NFR Specification Improve LLM Code Generation? A Comparison of Rich and Structured Interventions against a Natural-Language Baseline

**arXiv ID:** 2608.13742 | [PDF](https://arxiv.org/pdf/2608.13742v1)

**作者:** Joào Pedro Monteiro Pereira `[一作]`, Vinicius Cardoso Garcia `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对LLM代码生成的非功能需求（NFR）在ISO/IEC 25010基准化丰富化与结构化形式进行实验，评估其对功能正确性、代码质量与鲁棒性的影响。

**💡 创新点**

证明ISO标准驱动的NFR丰富化能显著提升静态代码质量与prompt鲁棒性，而格式（自然语言vsJSON）对性能几乎无影响，首次系统对比。

**🔧 技术方法**

采用OpenAI GPT‑4模型快照、Greedy解码，结合HumanEval/HumanEval‑ET benchmark、EvalPlus、Pylint静态分析以及Wilcoxon、Cliff’s δ、Holm‑Bonferroni等统计检验技术。

**📊 数据集**

使用164个HumanEval函数问题，10个prompt变体，按ISO/IEC 25010四个NFR（性能、错误处理、代码异味、可读性）生成。

**📈 对比分析**

通过每个问题的配对Wilcoxon检验和效应量比较NL‑simple与NL‑rich/Structured；结果显示功能正确性无显著提升，静态质量密度下降（可读性等）且prompt方差显著降低。

**⚠️ 局限性**

仅针对单函数、Exact‑output benchmark，模型版本与批次非同步收集，且错误处理NFR与测试或码点冲突，无法体现系统级ISO特性。

---

## 243. GPU Offload in Rust: Portable, Safe, and Fast

**arXiv ID:** 2608.13759 | [PDF](https://arxiv.org/pdf/2608.13759v1)

**作者:** Manuel S. Drehwald `[一作]` (University of Toronto), Johannes Doerfert `[通讯]` (Lawrence Livermore National Laboratory)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文实现了一个跨供应商 GPU 编译框架，直接集成到 rustc 和 LLVM Offload 基础设施中，并提供三种接口（自动、库互操作和手动预加载）来实现安全、可移植的 GPU 核心编程。

**💡 创新点**

创新点包括：①利用 Rust 的类型系统、所有权与 noalias 规则自动推断数据传输方向；②设计两次编译流水线（主机端收集元数据 + 设备端编译），实现多目标（NVIDIA、AMD）支持；③提出基于 PartitioningStrategy 的安全并行切分抽象，使得安全 Rust 代码可直接在 GPU 上并行执行；④在 LLVM OpenMP target 之上实现完整的 offload runtime，支持异步传输与重用。

**🔧 技术方法**

技术手段：rustc 前端改造、LLVM Offload 与 OpenMP target、MIR 分析、两次编译元数据传递、自动/手动数据预加载、预加载类型、异步 Host‑Device 传输、循环不变代码移动、基于 Rust 语义的内存布局推导。

**📊 数据集**

实验基准使用 RAJAPerf（从 RAJA 框架提取的多核 GPU 计算基准），在 AMD MI250X、NVIDIA H100 与 RTX A2000 上评估。

**📈 对比分析**

评估方式：与 RAJA 通过 CUDA/HIP/Seq、BaseCUDA、RAJA_CUDA 及 BaseHIP 对比，测量单核时间、总运行时间、内存传输量与寄存器使用。结果表明：Rust 核心的单核性能与 RAJA 基准几乎相当，自动接口在未优化前可能慢 400×，但经过预加载与异步优化后可逼近手动接口；整体运行时间在 32–46% 范围内相差，性能差异主要由传输同步与编译路径差异导致。

**⚠️ 局限性**

限制与挑战：目前仅支持 NVIDIA 与 AMD，Intel GPU 尚未完成；标准库 GPU 支持缺失；跨目标 ABI 不一致需要进一步验证；自动接口在多核循环或复杂数据共享时仍可能产生冗余同步；两次编译流程增加构建复杂度；缺乏完整的多设备并行扩展。

---

## 244. What preferences can - and cannot - predict in multi-agent online learning

**arXiv ID:** 2608.13810 | [PDF](https://arxiv.org/pdf/2608.13810v1)

**作者:** Omar Abbadi `[一作]`, Panayotis Mertikopoulos `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统研究了有限博弈中偏好图（ordinal incentive）与常规学习动力学（如FTRL、复制子动力学）的长时行为之间的关系。通过组合分析、能量函数和拉普拉斯方法，证明了：动态稳定集的骨架必定是俱乐部（club）或强俱乐部（sclub），并给出子博弈的完全对偶性（子博弈为club ⇔ 其扩展为正态的混合区间是渐近稳定的）。随后指出单靠偏好信息不足，构造了3玩家例子展示偏好图与动态稳定性不等价，并提出“radness”这一基于支付量的判据，证明若一个集合为rad（或srad）则其扩展为正态混合区间是吸引子，且可在多项式时间内检查。

**💡 创新点**

创新点包括：
1) 在无缝接点的前提下，首次给出偏好图与FTRL动力学稳定性的双向联系；
2) 对子博弈给出完全的“俱乐部⇔渐近稳定”等价性；
3) 构造偏好信息不足的反例，说明偏好图不能完全决定动态稳定性；
4) 提出radness这一简单的支付基准，可保证任意集合的稳定性，并给出高效算法；
5) 在复制子动力学下进一步证明rad性是充分条件，并与已知的6环、Shapley六角等案例统一。

**🔧 技术方法**

主要技术手段包括：
- 偏好图的组合结构与连通性分析；
- 拉格朗日正则化与FTRL的连续时间表示；
- 能量函数（Fenchel gap、Fenchel coupling、外部质量函数）用于证明吸引性；
- 多项式时间图算法用于判定radness；
- 级联与连通性理论用于推导子博弈的全局稳定性。

**📊 数据集**

数据集：本研究基于理论分析，使用的是标准的有限博弈表（支付矩阵），并无真实实验数据或外部数据集。

**📈 对比分析**

比较方法：对比已有的潜在游戏、弱可循环游戏等特殊结构下的结果，证明在这些类中radness与传统的强俱乐部等价。与传统求解混合纳什均衡的NP‑hard性对照，radness判定可在多项式时间完成，展示了其计算优势。性能方面：在无缝点游戏中，radness可以精确预测FTRL与复制子动力学的吸引子；但在存在局部源的更一般博弈中仍需进一步研究。

**⚠️ 局限性**

局限性：
- radness虽是充分条件，却并非必要条件，仍有例子不满足radness但稳定；
- 仅针对常规化学习（FTRL、复制子）给出结果，未覆盖所有无耦合学习算法；
- 对于非子博弈、非俱乐部集合的完整稳定性判定仍未完全给出；
- 论文侧重理论证明，缺乏大规模实验验证与性能评估。

---

## 245. Second Thought: Reasoning in Parallel as LLM Agents Act and Observe

**arXiv ID:** 2608.13667 | [PDF](https://arxiv.org/pdf/2608.13667v1)

**作者:** Zhensu Sun `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Second Thought 框架，利用 ReAct 代理在 Action–Observation 间隙的“思考空闲窗口”并行生成辅助思考，随后将这些思考（以原子思想形式）收集并投递到下一轮推理中，从而在不延长主线程顺序解码路径的情况下提升决策质量。

**💡 创新点**

创新点在于：①将思考空闲窗口视为可利用的并行计算资源；②通过四个互补维度（Check、Recall、Rehearse、Alternative）生成并行思考；③使用原子思想结构保证中断友好；④在无需训练的前提下实现辅助思考的无缝融合；⑤通过减少主线程思考量而非仅延长思考长度实现性能提升。

**🔧 技术方法**

技术手段包括：ReAct 推理框架、异步分叉（fork）四条辅助分支、原子思想（Atomic Thought）约束、思考维度设计、观察到达时的即时终止与思考收集、以及将收集到的思考追加至工具观察信息以供后续推理使用。

**📊 数据集**

数据集/基准：SWE‑Bench‑Pro（仓库级软件工程任务）、Terminal‑Bench 2.1（终端操作任务）、τ³‑bench（银行业务多轮对话任务）。

**📈 对比分析**

对比方法：基础 ReAct 代理（base）以及将相同额外思考量放在主线程的 compute‑matched 控制（s1）。实验表明：在所有 9 种模型–基准组合中，Second Thought 均能降低平均回合数，并在 6 种情况下显著减少主线程解码量（最高 43%），Pass@1 在 8 种情况下保持不变或提升（最高 +12.4 %）。在真实时间重放实验中，平均任务延迟降低约 10.9%。

**⚠️ 局限性**

局限性：①由于四个分支共享前缀缓存，整体 API 成本较 base 上升 66.4–181.5 %（主要由前缀读取产生）；②思考空闲窗口长度差异导致收益不均；③在某些任务中四个维度不一定都有用，单维或缺失维度时性能可能下降；④依赖 ReAct 样式的代理，其他形式的多模态或交互式推理尚未验证；⑤需要手动挑选/调节思考维度，未提供自动化选择机制。

---

## 246. EchoRec: Multi-Item Prediction-Empowered Generative Recommendation via Cycle-Consistent Preference Alignment

**arXiv ID:** 2608.14011 | [PDF](https://arxiv.org/pdf/2608.14011v1)

**作者:** Haokai Ma `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出EchoRec框架，利用多时间步未来行为的循环一致性偏好对齐进行密集监督，提升生成式推荐性能。

**💡 创新点**

创新点在于将多代币预测转为顺序依赖的多时隙监督，并通过可验证的循环一致性投影消除伪对齐。

**🔧 技术方法**

使用Transformer、MTP、循环一致性投影、语义ID(tokenization)、基于图的解码等技术。

**📊 数据集**

使用Amazon Reviews 2023的Game、Baby、Arts三大数据集。

**📈 对比分析**

与EAGER、TIGER、LETTER、ETEGRec、SETRec、RPG等基线对比，EchoRec在HR@10/NDCG@10等指标上显著提升，平均相对提升约10-20%。

**⚠️ 局限性**

局限在于对超参数敏感、在极端噪声下效果下降、未探索自适应时隙选择。

---

## 247. Attention Capture Is Not Detection: A Two-Stage Account of How Humans Miss Localized AI Image Edits

**arXiv ID:** 2608.13865 | [PDF](https://arxiv.org/pdf/2608.13865v1)

**作者:** Chiao-Chieh Deng `[一作]` `[通讯]` (National Chengchi University), Chiao-Chieh Deng (National Chengchi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计眼动跟踪实验验证 AI 图像编辑的可检测性分为两阶段，构建生成式扫描路径模型预测注意捕获。

**💡 创新点**

创新点在于把可检测性拆解为“注意捕获”与“判断准确”两独立阶段，并用 Transformer 生成扫描路径量化前阶段。

**🔧 技术方法**

技术包括混合效应模型分析、WebGazer.js 眼动采集、基于 ViT 的 SpatioTemporalGazeGen 生成式扫描路径模型，以及对比线性回归基线。

**📊 数据集**

使用了 60 张基础图像的四种 AI 编辑（大小+可/不可）共 240 张图像，实验招募 59 名参与者进行眼动记录。

**📈 对比分析**

与简单线性基线（编辑面积+可疑度）对比，生成式模型在预测 LBFS（look‑but‑fail‑to‑see）发生率上相关系数提升约 0.07（从 0.48 提升至 0.55），且能提供空间级别的注意分布信息。

**⚠️ 局限性**

局限包括使用 WebGazer.js 的低精度眼动数据、样本量仅 59 人、未进行留一受试者交叉验证、对重访（re‑entry）建模弱化、以及模型在细粒度条件内的相关性相对较低。

---

## 248. SCVIB: Editable State-Conditioned Visual Instance Binding forMulti-Turn Personalized Localization

**arXiv ID:** 2608.14148 | [PDF](https://arxiv.org/pdf/2608.14148v1)

**作者:** Xiongtai Yang `[一作]` (Sichuan University), Tao Wang `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出可编辑的状态条件视觉实例绑定（editable state‑conditioned visual instance binding），构建了多轮个性化定位基准SCVIB，并设计了TT‑VG两阶段推理框架（TSTT进行状态到证据的可执行路由，VEGA进行支持条件下的视觉定位）。

**💡 创新点**

创新点包括：①把多轮交互的目标状态执行与最终证据选取分离，形成可审计的状态‑证据转换；②通过视觉证据包（Visual Evidence Package）将定位证据与查询图像结合，使同实例定位能充分利用已解析的支持证据；③创建了四种目标状态依赖组的可控基准，用以系统评估不同状态更新策略。

**🔧 技术方法**

技术手段：TSTT（Target‑State Transition Tree）编译交互历史为协议事件并执行；VEGA（Visual Evidence Grounding Adaptation）基于LoRA在Qwen3‑VL‑4B上进行参数高效微调；轨迹衍生的同实例对进行训练；低秩适配、可视化证据包构造以及基于自然语言的交互协议。

**📊 数据集**

数据集：SCVIB基于1,050条人工验证的支持‑查询对，覆盖5个视觉域、3个难度级别，扩展为1,500个多轮情节；训练VEGA使用来自LaSOT的5,000条轨迹衍生同实例对；基准评估使用包含2个支持的120条和多支持的1,380条情节。

**📈 对比分析**

与直接Seq‑free MLLM（如Qwen3‑VL‑8B、Qwen2‑VL等）以及结构化状态表提示（Structured State‑Table Prompting）对比，TT‑VG在相同TSTT‑解析证据下实现70.27% Joint@0.5、61.93% mIoU，显著优于GDINO‑MASA‑R50（54.07%）和PerSAM（30.87%）；在目标解析准确率（TR）上，TT‑VG实现100%而传统Seq‑free仅约98%。

**⚠️ 局限性**

局限性：①交互协议为封闭式，缺乏自然语言多样性；②伪名称仅局部使用，缺乏全局身份语义；③VEGA仅训练于单一模型（Qwen3‑VL‑4B）且基准划分依赖该模型的定位性能，可能带来族群偏差；④未考虑跨会话实例持久化或更长交互历史；⑤对复杂视觉变换和长时序的鲁棒性尚待验证。

---

## 249. FLARE MCMC: Fidelity-based Layer-Adaptive REcursive proposals for MCMC

**arXiv ID:** 2608.13774 | [PDF](https://arxiv.org/pdf/2608.13774v1)

**作者:** Harini Venkatesan `[一作]` (University of California Riverside), Mengxuan Wu `[通讯]` (University of California Riverside)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多层次多保真度 MCMC（FLARE MCMC）算法，通过递归嵌套低保真度链作为高保真度链的提议，显著提升采样混合速度；

**💡 创新点**

创新点在于利用递归层级链构造提议，并结合简单的层调优（加权混合均匀分布）以增强低保真度提议的探索性，无需梯度或内部结构假设；

**🔧 技术方法**

使用 Metropolis–Hastings 核心框架、递归链、层调优（对低保真度分布添加常数并重归一化）、Lambert W 求解最优内层步数、自动调节协方差（AM）以及多尺度/多保真度模拟；

**📊 数据集**

在三大领域进行实验：简易摆子模型（ODE）、地下水渗透模型（PDE 网格细化）和宇宙结构形成（N‑体粒子模拟 FastPM），分别采用三种保真度层级；

**📈 对比分析**

与标准 Metropolis–Hastings、MLDA（含 AEM）、MLMCMC、MFMC 等多保真度方法对比，FLARE 在有效样本数/秒（ESS/s）上均显著优于对手，尤其在尾部分布上提升明显，同时保持更少的似然评估次数；

**⚠️ 局限性**

局限性包括：需预先构造多保真度模型且保真度差距需足够大、对层调优参数的经验选择（M、ω）可能不具普适性、在极端昂贵模型（如宇宙学 N‑体）下仍受计算预算限制。

---

## 250. SSP: An Event-Matched Syn2Sim2Phy Cross-Domain Evaluation Framework for Autonomous Driving VLA Models

**arXiv ID:** 2608.14024 | [PDF](https://arxiv.org/pdf/2608.14024v1)

**作者:** Haojie Feng `[一作]` (Tongji University), Lu Xiong `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SSP（Synthetic–Simulation–Physical）框架，对自驾 VLA（视觉‑语言‑动作）模型进行跨域事件匹配评估；

**💡 创新点**

通过事件级别的语义抽取与审核，保证 Synthetic、Simulation、Physical 三个域在相同交互事件下对比；统一行为链评估（文本、轨迹、风险响应）并给出整合 VLA 能力分数；

**🔧 技术方法**

语义抽取与规则校验、CARLA 以及真实测试平台编译、跨域一致性审核、统一行为表示与评价、基于规则+LLM 的文本抽取、轨迹映射到统一坐标与时间窗口；

**📊 数据集**

使用合成长尾事故视频（AVD2 生成）、CARLA 赛道、闭路测试平台；涵盖两类安全关键场景（车道切入、行人/骑行者穿越）与三种数据域；

**📈 对比分析**

采用 2×3×3 的实验矩阵，评估 OpenEMMA、LLaViDA、Alpamayo‑R1 三种 VLA 模型；综合指标 IVCS（包括文本有效率、语义覆盖、关键交互识别、轨迹质量、风险响应）显示：物理域 0.325>仿真 0.291>合成 0.259，Alpamayo‑R1 最高 0.405，OpenEMMA 0.338，LLaViDA 0.131；不同场景下域排名会变动；Qwen3‑VL 30B‑A3B (MoE) 以约 3B 活跃参数获得最佳轨迹质量与 IVCS 0.398；

**⚠️ 局限性**

仅覆盖两类场景与三域，缺乏交叉路口、合并等复杂情形；评估窗口仅 1 s，未覆盖长时行为；跨域事件匹配受合成源质量限制；模型比较受提示、坐标、解析器影响；未进行闭环控制与实时性能评估。

---

## 251. Spectral Efficiency Centrality: An Efficient Spectral Approach for Influential Node Identification in Temporal Networks

**arXiv ID:** 2608.13960 | [PDF](https://arxiv.org/pdf/2608.13960v1)

**作者:** Aksa Urooj `[一作]` (National Institute of Technology), Iqra Altaf Gillani `[通讯]` (National Institute of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Spectral Efficiency Centrality（SEC）并给出基于 Perron–Frobenius 理论的一阶近似 ASEC，用于衡量时变网络中节点对谱半径的影响。

**💡 创新点**

创新点包括：①将谱半径节点去除效应引入时间网络，②使用一阶特征值扰动实现快速近似，③在同一度量框架下兼顾传播效率与结构鲁棒性。

**🔧 技术方法**

技术手段包括谱分析、特征值扰动理论、Perron–Frobenius 定理、Jaccard 相似性检测窗口、Python/NumPy/NetworkX 计算框架。

**📊 数据集**

实验使用三种合成网络（ER、BA、WS）和三大真实时序网络（Enron Email、CollegeMsg、MathOverflow）。

**📈 对比分析**

与 TWC、EffC、Supracentrality、TCC、TempoRank、Temporal Coreness、CENDY、QUBE 等基线在 SI/SIS/IC 模型下的 AUC、Kendall τ、Top‑k 重叠、计算时间和鲁棒性指标进行对比；SEC/ASEC 在传播效果和结构破坏上优于大多数基线，ASEC 的计算时间比 SEC 低一个数量级。

**⚠️ 局限性**

局限性：仅适用于无向无权网络；目前实现为离线批处理，无法处理在线/流式数据；未在有向/加权网络或其他领域进行验证。

---

## 252. Verified Pythagorean Composition for Adaptive Cryptographic Games: Noise Flooding in Homomorphic Encryption

**arXiv ID:** 2608.13846 | [PDF](https://arxiv.org/pdf/2608.13846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 253. Inducing Reward-Free Judging Rubrics that Reduce Over-Crediting in Agent Evaluation

**arXiv ID:** 2608.13564 | [PDF](https://arxiv.org/pdf/2608.13564v1)

**作者:** Darragh Quinn `[一作]` (Trinity College Dublin), Cormac Sheehan `[通讯]` (University College Dublin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在有限的标注轨迹上进行反射进化，学习并冻结一套可读文本的评估 rubrics，随后在不访问环境的情况下用一条 LLM 调用即可判定代理轨迹的成功与否。

**💡 创新点**

创新点在于：①仅优化评估 rubric 文本而不更新模型权重，保持模型的可重复性；②将评估目标“是否成功”通过标注轨迹真实奖励进行 grounding，显著减少过度信用（false‑pass）错误；③生成的 rubric 具有人类可读的评价准则，便于审计和解释。

**🔧 技术方法**

技术方法包括：使用 GEPA（基于反射的 prompt evolution）对 rubric 进行搜索；采用冻结的 Qwen2.5‑7B LLM 作为评估器；反射 meta‑prompt 生成改进后的 rubric 并以验证集选择最佳；最终将最佳 rubric 冻结后作为单次调用的评分器。

**📊 数据集**

实验数据集：
- τ‑bench：220 条 roll‑out，173 条用于训练/验证/测试；
- WebShop：160 条 roll‑out（其中 116 条带有连续评分）。两者均提供可执行的环境奖励。

**📈 对比分析**

与基线对比：Oracle（环境奖励）、generic G‑Eval rubric、FewShot k=4、Heuristic、Majority。评估指标包括二值 agreement、F1、Cohen κ、AUC、Spearman/ Kendall（排序）和绝对校准。结果显示：在二值 agreement 上无显著提升（p=0.248），但 false‑pass 率从 0.173 降至 0.115；在分级任务中排序 Spearman 从 0.370 提升到 0.410，校准稍差。

**⚠️ 局限性**

局限性：
- 仅使用单一 7B 冻结模型，可能已接近 agreement 上限；
- 仅在两项 benchmark 上验证，结果的普适性待进一步测试；
- 正例比例低导致 F1、κ 估计不稳；
- 校准与排序之间存在权衡；
- 需要一定量的标注轨迹进行 rubric 诱导；
- 结果对更大或更强模型的迁移性尚未验证。

---

## 254. CForce: Boosting Parallel Decoding for dLLMs via Consistency Forcing

**arXiv ID:** 2608.13925 | [PDF](https://arxiv.org/pdf/2608.13925v1)

**作者:** Yuji Ren `[一作]` (Shanghai Jiao Tong University), Zhijie Deng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练方法——Consistency Forcing（CForce），通过在模型自身的阈值解码轨迹上构建阶段化对齐约束，提升扩散语言模型在高并行解码时的早期预测可靠性，从而改善速度-质量平衡。

**💡 创新点**

创新点包括：①使用同模型的stop‑gradient目标而非冻结教师，确保训练与推理状态的一致；②设计了自适应的Confidence Adaptive KL（CAD）损失，结合正向KL与基于置信度的逆向KL；③在可编辑扩散模型中利用后期的Token‑to‑Token（T2T）修正为早期Mask‑to‑Token（M2T）预测提供监督；④通过阶段化轨迹和渐进式课程学习稳定训练。

**🔧 技术方法**

技术手段包括：扩散语言模型（dLLM）框架、基于阈值的并行解码、阶段化自回归轨迹构建、CAD损失、交叉熵锚定、渐进式课程策略、以及对可编辑模型的T2T监督。

**📊 数据集**

使用了OpenMath‑Instruct‑2和OpenCodeInstruct训练数据，并在四个基准（GSM8K、MATH500、MBPP、HumanEval）上进行评估。

**📈 对比分析**

与基线LLaDA2.0‑mini、LLaDA2.1‑mini及其CAP版本对比，CForce在编辑模型上将平均TPF从6.94提升至9.08、准确率从85.57提升至86.41；在非编辑模型上将TPF从3.60提升至6.42、保持较高的准确率。相对于其他加速方法（dUltra‑coding、d3LLM、LightningRL等），CForce在高TPF预算下获得更优的Score‑TPF折衷，并在多步生成任务中显著提升性能。

**⚠️ 局限性**

主要局限：训练时使用预先收集的轨迹，随着模型更新轨迹与实际推理状态可能产生差距，限制了对最终分布的精准对齐；此外，目前方法仅在阈值解码设置下验证，需要进一步探讨在线轨迹收集和策略更新机制。

---

## 255. Face Re-morphing: Differential Morphing Attack Detection via Feature-Space Similarity Changes

**arXiv ID:** 2608.13858 | [PDF](https://arxiv.org/pdf/2608.13858v1)

**作者:** Jie Jin `[一作]` (Shizuoka University), Tetsushi Ohki `[通讯]` (Shizuoka University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于再造影像（re‑morphing）的差分攻击检测方法——Face Re‑morphing；该方法通过对文档图像与可信实时图像进行再一次混合，再利用文档–实时与实时–再造影像之间余弦相似度的变化作为判别分数，实现对面部扭曲攻击的检测。

**💡 创新点**

创新点包括：
1) 将现有的混合技术直接嵌入检测流程，利用再造过程本身产生的相似度变化作为判别特征；
2) 引入动态相似度差（Δs = cos(L,Re) – cos(D,L)），而非传统的静态差值或重建误差；
3) 在不需要额外训练的情况下，兼容多种特征提取器和多种再造方法，展示了该指标的普适性。

**🔧 技术方法**

技术细节：
- 特征提取使用 CVLFace 工具包的四种深度网络（IR101、IR50、ViT‑KPRPE、ViT）
- 再造方法包括 MorDIFF（扩散自编码器）、OpenCV（基于Landmark的几何混合）和 StyleGAN2‑ADA（潜空间线性插值）
- 采用余弦相似度并以 Δs 作为最终判别分数；评估指标为 BSCER、MACER 与 EER。

**📊 数据集**

数据集：
- FRLL‑Morphs（多种混合条件下的 102 张真实样本、2175 条 accomplice/ criminal 条件）
- FEI Morph 版本 1 与 2（V1: 400 正面、12000 accomplice/ criminal 条件；V2: 400 正面、28000 accomplice/ criminal 条件）
- 公开的 ACIdA 对照配对信息，避免手工挑选有利样本。

**📈 对比分析**

对比方法与性能：
- 在 AMSL 基准上，使用 OpenCV 再造时几乎无误（EER、B_0.01 均为 0），优于现有的 D‑MAD 方法；
- 在 FEI‑M1 的 Criminal 条件下，EER 0.010、B_0.01 0.015，超过已发布的最佳结果；
- 在 FRLL‑Morphs 与 FEI‑M2 的所有条件下，Δs 在多种特征提取器上均保持低 EER（从 0.00099 到 0.079，取决于再造方法），尤其在 Accomplice 条件下仍优于仅用 s_re 的方案；
- 相比 s_re 或 s_base，Δs 在 MorDIFF 与 OpenCV 再造时显著提升检测效果；
- 综上，Face Re‑morphing 在多数据集、多条件下表现稳健，尤其对 Criminal 情景的检测效果突出。

**⚠️ 局限性**

局限性：
- 性能高度依赖于再造方法，StyleGAN2‑ADA 在某些条件下导致身份漂移、相似度变化弱化；
- 在 Accomplice 条件下检测效果相对较差，尤其对混合因子较低时的混合图像；
- 目前仅在数字图像上验证，缺乏对打印‑扫描、摄像头噪声等实际采集过程的鲁棒性评估；
- 需要准确的面部对齐与 Landmark 估计，对低质量图像或遮挡可能产生误差。

---

## 256. MoE Expert Execution in Disaggregated LLM Serving with a High-Bandwidth ReRAM Near-Memory Architecture

**arXiv ID:** 2608.13962 | [PDF](https://arxiv.org/pdf/2608.13962v1)

**作者:** Kunming Shao `[一作]` (Hong Kong University of Science and Technology), Chi-Ying Tsui `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种基于ReRAM近存储计算的Mixture‑of‑Experts（MoE）FFN池架构ReXpert，用于分离的LLM推理，显著提升FFN池吞吐和延迟，并降低权重移动能耗。

**💡 创新点**

创新点在于：①在高带宽ReRAM near‑memory上实现权重驻留；②引入核心局部多播池化、共激活感知置放和负载感知取数，按需求分配各层通信；③将实际MFU分解为理想MFU与占用率两因子，系统性分析与优化。

**🔧 技术方法**

使用技术包括ReRAM NMC、核心局部多播池化、置放感知与负载感知取数、层级化通信架构（Unit→Core→Die→Package→Node）、BookSim2+DSENT网络建模、roofline模型以及测量+建模的混合评估。

**📊 数据集**

使用的数据集和模型为Qwen3.5‑35B‑A3B、Qwen3.5‑397B‑A17B和GLM‑5.2，并基于Alpaca和HumanEval/Code解码轨迹进行实验。

**📈 对比分析**

通过与同容量、相同峰值计算的H20以及H100/H800 FFN池进行对比，ReXpert在相同峰值计算下FFN池延迟下降9.5×、权重移动能耗降低20×；H20‑attention+ReXpert‑FFN系统相较同尺寸H20池，解码TPOT下降1.25–4.0×（35B）、2.4–10.3×（397B）及2.5–10.4×（GLM‑5.2）。评估采用测量+建模混合方法。

**⚠️ 局限性**

主要局限包括：①依赖可编程、只读ReRAM实现，受设备变异、保留、耐久性及Yield限制；②激活/归约流量与多播争用调度尚未完成；③未实现实际硬件验证，ECC/刷新策略未涵盖。

---

## 257. FIRM: Fine-Grained Intra-Token Representation of Masks for Remote Sensing Reasoning Segmentation

**arXiv ID:** 2608.13980 | [PDF](https://arxiv.org/pdf/2608.13980v1)

**作者:** Weidong Tang `[一作]` (Xi'an Jiaotong University), Xiangyong Cao `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FIRM 方法，利用细粒度的视觉 token 内子格掩码代码和连续渲染实现高精度遥感图像的推理与参照分割。

**💡 创新点**

创新点在于：①将每个压缩的视觉 token 细分为 r×r 二值子格，直接预测掩码代码以保留局部结构；②使用固定查找表和代码边际化生成离散子格掩码与软结构场；③结合 Posterior‑Anchored Continuous Renderer (PACR) 在子格内进一步恢复细节。

**🔧 技术方法**

技术要点包括：基于 Qwen3‑VL‑4B 的多模态 LLM、视觉编码器与 2×2 视觉 token 合并、mask‑token 生成、mask‑query 设计、代码词表、离散/连续渲染、损失函数组合（交叉熵、Hamming、Dice 等）。

**📊 数据集**

使用的遥感数据集有：LaSeRS、EarthReason、DRSeg、RRSIS‑D、RISBench，涵盖推理分割与参照分割两大任务。

**📈 对比分析**

与多种基线（LISA、SegEarth‑R2、SELF1E、STAMP 等）对比，FIRM 在 LaSeRS、EarthReason、DRSeg 等基准上均取得最高 gIoU/cIoU，平均提升 3–20+ 分，且在推理速度与显存消耗上保持与现有方法相近或更优。

**⚠️ 局限性**

局限性：①在极细分辨率下代码数急剧增大，可能导致推理时内存与计算成本上升；②在细化后视觉理解的整体性能略有下降，表明 mask‑code 预测会消耗部分原有视觉知识；③目前仅在已标注遥感数据上验证，未探究对更大规模、更多样化场景的泛化能力。

---

## 258. GRPO Beyond English: A Large-Scale Study of GRPO in Non-English and Multilingual Settings

**arXiv ID:** 2608.13698 | [PDF](https://arxiv.org/pdf/2608.13698v1)

**作者:** Konstantin Dobler `[一作]` (Apple), Simon Lehnerer `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了大语言模型在多语言环境下使用GRPO（Group Relative Policy Optimization）进行强化学习后训练的效果，探讨了英语与本地语言推理奖励的差异；

**💡 创新点**

创新点在于大规模覆盖九种基模型、十一种训练语言及多语言训练，发现本地语言奖励仅略逊于英语奖励，同时揭示了强跨语言迁移与语言/任务特异性回退的并存；

**🔧 技术方法**

主要使用GRPO框架、RLVR（可验证奖励）以及语言奖励机制（英语或本地语言）进行训练；

**📊 数据集**

训练和评估数据集为Multilingual Reasoning Gym（可调节难度的程序化生成任务），并在MGSM、PolyMath等公开评测集上验证；

**📈 对比分析**

与单语训练对比，多语言训练或英语奖励可在多数语言下匹配或接近同语训练收益；在某些低资源语言上可获得与本地语言奖励相近的性能；但在特定模型-语言组合下出现显著回退，表明需要针对性评估；

**⚠️ 局限性**

局限性包括使用模板化、可验证的Gym数据而非真实查询、多语言实验共用相同超参且仅单一随机种子，且未覆盖非可验证任务或工具使用场景。

---

## 259. Consensus-gated Multi-Agent Neural Architecture Search for Seismic Fault Segmentation

**arXiv ID:** 2608.13889 | [PDF](https://arxiv.org/pdf/2608.13889v1)

**作者:** Shehram Baig `[一作]` (Information Technology University), Ahmad Mustafa `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过一个由Claude、GPT‑5.1和Gemini 2.5 Pro三大 LLM 组成的共识辩论门控的 NAS 循环，搜索并发现了一种 425K 参数的编码器‑解码器网络，用于高精度的地震断层分割；

**💡 创新点**

创新点在于：①使用多模型 LLM 辩论达成一致后才生成代码，避免单一模型陷入偏见；②在无预定义搜索空间的前提下，直接对 PyTorch 代码进行编辑、验证与交叉审查；③结合硬参数预算、keep‑or‑revert 机制和停滞规则，显著降低搜索成本；

**🔧 技术方法**

技术手段包括：多 LLM 互评与共识决策、Python AST 代码生成与验证、GroupNorm、SE 门控、strip‑pooling、FPN 融合、Adam 优化器、交叉验证等；

**📊 数据集**

使用 Thebe Fault 数据集（Exmouth Plateau，澳大利亚）中的两块 100 张 600×1500 的 2D 切片，约 5.6% 的像素为断层；

**📈 对比分析**

对所有模型（包括公开的 U‑Net、DeepLabV3‑ResNet50 等）在相同训练协议下重训；发现网络在 43 万参数下达成 F1=0.578、IoU=0.406，明显优于 31M UNet、39M DeepLabV3‑R50 等模型，性能提升约 0.094–0.061 F1；搜索仅消耗约 1 GPU‑日和几美元 API 费用；

**⚠️ 局限性**

局限性包括：仅在单一任务与数据集上验证；搜索仍依赖 LLM 调用；对不同地震场景的泛化尚未评估；模型性能主要基于定量指标，缺乏人工视觉质量评估；

---

## 260. Fine-Tuning Qwen3-27B for C-to-Rust Code Translation: A Three-Stage Curriculum of Pretraining, Debugging-Aware SFT, and Task-Specific SFT

**arXiv ID:** 2608.13681 | [PDF](https://arxiv.org/pdf/2608.13681v1)

**作者:** Pu Zhao `[一作]` (Northeastern University), Yanzhi Wang `[通讯]` (Northeastern University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Qwen3-27B模型进行三阶段细化训练，使其专门用于C到Rust的安全、惯用转换；

**💡 创新点**

创新点在于将Rust专注的持续预训练、调试/自我修复SFT以及LeetCode级别C↔Rust对齐的任务特定SFT结合为一套完整的训练课程，并在SACTOR验证框架下评估；

**🔧 技术方法**

使用自回归语言模型技术、DeepSpeed ZeRO-3并行训练、BF16精度、温度采样与Top-p/Top-k等推理策略，以及agentic修复循环；

**📊 数据集**

训练数据来源包括七大Rust语料库（Strandset-Rust、CodeFIM-Rust-Mellum、rust_instruction_dataset、humaneval-rust、Magicoder-OSS-Instruct-75K-rust、xCodeEval-rust、starcoderdata-rust）、VeruSyn验证/调试数据集、以及C2Rust-Moxin LeetCode级别的C↔Rust对齐数据；

**📈 对比分析**

在200条C程序的SACTOR评估框架下，经过三阶段训练的模型达到87.3%的成功率，显著高于未细化的27B基线（72.3%）以及多种规模更大或同等规模的开源模型，并与Claude 4.6等闭源模型保持竞争水平；

**⚠️ 局限性**

局限在于模型在泛化软件工程任务（如SWE-bench Verified）上略逊于未细化基线，表明在专注训练时存在轻微的灾难性遗忘；此外，训练数据仍以LeetCode级别为主，缺乏真实大型C项目级别的复杂指针和结构体迁移经验。

---

## 261. CLAIR-Fin: An Adversarial Multi-Agent Framework for Claim-Level Verification and Adaptive Debate in Cross-Modal Financial QA

**arXiv ID:** 2608.13706 | [PDF](https://arxiv.org/pdf/2608.13706v1)

**作者:** Fatema Tuj Johora Faria `[一作]` (Ahsanullah University of Science and Technology), Md. Alam Hossain `[通讯]` (Jashore University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种九代理框架 CLAIR‑Fin，用以在多模态金融报告中进行可信问答。

**💡 创新点**

创新点在于将问题拆解为有类型的财务主张，并通过非对称证据权重、链式托管验证、适应性辩论循环以及连续幻觉风险指数来实现跨模态证据权衡与及时检测。

**🔧 技术方法**

使用的技术包括基于大语言模型的多模态检索、类型化证据权重、对抗性辩论、链式责任验证以及终端判定审核。

**📊 数据集**

实验数据集为从孟加拉国中央银行年度报告构建的 500 题 BB‑FinQA‑X。

**📈 对比分析**

在该数据集上与单通道 RAG、HyDE、Graph‑RAG 等基线对比，CLAIR‑Fin 的可信度从 0.780 提升至 0.889，且在精确度、覆盖率等指标均优于所有基线。

**⚠️ 局限性**

局限包括仅针对单一机构单语言的数据、使用固定手工设定的权重与阈值、对跨模态链接的粗糙子串匹配、以及未对多轮交互与不同模型进行评估。

---

## 262. Towards Efficient Multimodal and Multilingual Opinion Extraction for STI: A QLoRA-Based Fine-Tuning Approach

**arXiv ID:** 2608.14152 | [PDF](https://arxiv.org/pdf/2608.14152v1)

**作者:** Sheng Hong `[一作]` (Beihang University), Yuwei Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建多模态、多语言的核心意见提取框架，并通过视觉证据引导文本识别；

**💡 创新点**

利用视觉上下文锚定核心意见、引入模糊累计前景理论的案例价值评估模块，以及基于QLoRA的参数高效微调；

**🔧 技术方法**

使用VideoLLaMA2/2.1作为基底，QLoRA进行低秩量化微调，Fuzzy-CPT进行价值打分；

**📊 数据集**

自建2,194条包含文本、图像和视频的四语言（英、中、俄、西）标注数据集；

**📈 对比分析**

与零样本、提示工程、外部多模态模型对比，Fine-tuned VL2.1在Image-Augmented设置下实现64.98%精确度、51.14% F1，显著优于零样本和提示基线；

**⚠️ 局限性**

仍存在多意见输入的覆盖不足、边缘信息噪声、对视频动态信息利用有限等局限。

---

## 263. AlphaSeek: Trajectory-Level Self-Iterative Factor Mining Framework for Multi-Source Financial Data

**arXiv ID:** 2608.13913 | [PDF](https://arxiv.org/pdf/2608.13913v1)

**作者:** Qilu Zhu `[一作]` (Zhongnan University of Economics and Law), Simon Fong `[通讯]` (University of Macau)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个端到端的LLM驱动Alpha因子挖掘框架AlphaSeek，自动获取挖掘方向、在轨迹级别上进行因子构建与演化，并通过自迭代组合实现因子库与组合的闭环优化

**💡 创新点**

创新点在于将优化单元从单因子提升到完整的研究轨迹，利用演化算子实现多样性与鲁棒性，并引入自迭代组合及冗余感知约束形成完整闭环

**🔧 技术方法**

使用大语言模型进行方向生成、符号因子构造、AST校验与代码编译；采用进化算法（变异、交叉）进行轨迹优化；引入一致性、复杂度、冗余约束；自迭代组合与回测循环

**📊 数据集**

采用中国A股 CSI300 日频数据 2016-2025，划分训练/验证/测试；同时在 CSI500 进行零射转移测试

**📈 对比分析**

与传统机器学习、深度学习模型、因子库以及其他LLM基因子挖掘方法进行比较，AlphaSeek 在 ARR、IR、MDD、IC 等指标上均优于对照组，表现最为突出

**⚠️ 局限性**

限制在于仍受LLM语义漂移和历史过拟合影响，因子库依赖历史市场特征，跨市场泛化尚有限，且计算成本较高

---

## 264. Joint Optimization of Memory and Computing Frequency for Energy-Efficient DNN Inference

**arXiv ID:** 2608.13863 | [PDF](https://arxiv.org/pdf/2608.13863v1)

**作者:** Yunchu Han `[一作]` (Tsinghua University), Zhisheng Niu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了针对移动设备DNN推理的内存频率、计算频率、带宽与传输功率联合优化方案；

**💡 创新点**

首次将内存频率与计算频率一起纳入能耗最小化模型，并给出近似闭式解与启发式算法；

**🔧 技术方法**

采用实际测量得到的推理时延模型、凸优化与KKT分析、线性化传输速率公式及启发式贪心策略；

**📊 数据集**

在Jetson TX1上使用ResNet152与VGG19模型对CIFAR-100数据集进行实验；

**📈 对比分析**

与随机去除、仅优化计算频率、保持最大频率等三种基线对比，实验显示平均能耗可降低至10.4%，近似解误差≤2.5%；

**⚠️ 局限性**

近似解对截止时间要求高时误差略增，且仅针对特定DNN与静态信道模型，未考虑边缘能耗与动态网络条件。

---

## 265. CoANeRV: Coordinate-Aware Token-Space Neural Video Representation

**arXiv ID:** 2608.13938 | [PDF](https://arxiv.org/pdf/2608.13938v1)

**作者:** Jialong Guo `[一作]` (Zhejiang University), Haishuai Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了CoANeRV，一种将视频特定信息存储在固定长度视频令牌中的坐标感知令牌空间框架，实现无视频特定解码器的全片段增量编码与连续时空查询。

**💡 创新点**

创新点在于：① 将NeRV的权重空间表征转为令牌空间表征，实现解码器共享；② 采用轴自适应频率位置编码与温度调制交叉注意力的坐标感知解码器；③ 使用块级时空查询显著降低峰值注意力内存，使高分辨率重建可行。

**🔧 技术方法**

技术包括：卷积令牌化 + 多层Transformer令牌生成器；轴自适应多频率位置编码；温度调制的交叉注意力（通过fused SDPA实现）；块级时空查询；YUV 颜色空间转换；后训练量化和可选哈夫曼编码。

**📊 数据集**

使用 Kinetics-400 作为训练集，评估在 Kinetics-400、Something‑Something V2 与 UCF101 三个保留集；视频长度为 4、8、16 帧，帧分辨率 256×256。

**📈 对比分析**

与 Feed‑Forward INR 基线（TransINR、GINR、FastNeRV）以及视频化 ANR‑V 进行对比；CoANeRV 在 PSNR/SSIM 上领先 6–10 dB，峰值显存仅为 2.6–4 GB（相比 ANR‑V 的 18–74 GB），训练时长大幅缩短（如 70 GPU‑hrs 对比 344 GPU‑hrs）。

**⚠️ 局限性**

主要局限：解码器吞吐量仍低（1080p 仅 7.7 FPS）、仅适用于短片段、缺乏端到端量化与熵建模；未来工作需提升解码速度、扩展长时序令牌、学习令牌压缩与更广泛的迁移。

---

## 266. Adjacency-Based Spectral Proxy Control of Mobile Communication Agents

**arXiv ID:** 2608.13616 | [PDF](https://arxiv.org/pdf/2608.13616v1)

**作者:** Mariana del Castillo `[一作]` (Universidad de la República), Federico Larroca `[通讯]` (Universidad de la República)

**通讯引用:** 365 | [OpenAlex ID](https://openalex.org/A5017017294)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了A‑Fiedler控制器，利用邻接矩阵主特征向量替代传统Fiedler向量，对移动通信代理进行分布式重定位；将Fiedler梯度控制器分解为局部交互规则与潜在几何结构；

**💡 创新点**

创新点在于将Fiedler梯度控制的结构拆解为可分离的局部交互与全局嵌入，允许使用更易分布式估计的谱嵌入（邻接矩阵主特征向量），从而在有限通信预算下显著提升鲁棒性；

**🔧 技术方法**

采用谱图理论、主特征向量估计、功率迭代、最大/平均共识等分布式算法实现控制器；

**📊 数据集**

使用随机生成的任务代理与通信代理网络（规模随总代理数扩展）以及四叶草形轨迹的动态任务代理作为实验数据；

**📈 对比分析**

通过与经典Fiedler梯度控制器在相同及有限通信预算下的对比，利用多商品流量（MNF）指标评估性能；实验表明A‑Fiedler在有限预算下保持与精确版本相近的MNF提升，而L‑Fiedler在同等预算下表现明显下降；

**⚠️ 局限性**

局限性包括分布式共识精度受限导致的估计误差、缺乏对不同嵌入选择的系统性理论分析、以及在更大规模或更极端动态场景下的收敛与稳定性未知。

---

## 267. Active Perception for Embodied Disambiguation

**arXiv ID:** 2608.13605 | [PDF](https://arxiv.org/pdf/2608.13605v1)

**作者:** Yiwei Liu `[一作]` (Chinese University of Hong Kong), Luwei Yang `[通讯]` (Shenzhen Research Institute of Big Data)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5040614720)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出一种以主动感知为核心的嵌入式目标消歧框架，机器人通过视角移动获取新的 RGB‑D 观察，结合视觉‑语言模型判断继续观测、请求澄清或直接选择目标。

**💡 创新点**

创新点在于：①将物理观测与用户澄清视为互补的信息渠道，形成连续的观测–互动–选择流程；②利用主动视角获取缺失的物理证据（遮挡、细纹、未见目标），从而减少对用户澄清的依赖；③在同一框架中同步管理观察证据、澄清答案与最终目标决策。

**🔧 技术方法**

核心技术包括：基于 Qwen3.7‑Plus 的视觉‑语言模型用于决策（观察、澄清、选择）；眼‑手摄像机实现视角变化与新观察获取；实时将新 RGB‑D 数据输入模型更新推理。

**📊 数据集**

实验数据来自自制桌面场景：9 个匹配实验（A1–A9），每个场景包含不同的观察不完整性或语义歧义；使用真实机器人 PiPER 与 DABAI DC1 RGB‑D 摄像机。

**📈 对比分析**

对比方法包括 Passive Direct Selection、Greedy Asking、Static Asking。实验结果显示：在观察不完整场景中，主动感知方法在 6/9 场景中成功解决目标，且无需澄清；在语义歧义场景中，主动感知提升了澄清的准确性和效率；在未见目标场景中，主动感知能扩展搜索范围，完成目标选择。

**⚠️ 局限性**

局限性：①未处理观测与用户反馈之间的冲突；②仅关注需要实时获取的物理证据，未结合预训练知识作为第三信息源；③实验范围局限于桌面物品，未验证在更复杂环境下的可扩展性。

---

## 268. MemoryLake on MemoryArena: A Matched Study of Agent Memory Backends

**arXiv ID:** 2608.13883 | [PDF](https://arxiv.org/pdf/2608.13883v1)

**作者:** Chaoqun Zhan `[一作]` (MemoryLake Team), Qianjin Wang `[通讯]` (MemoryLake Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 MemoryArena benchmark 上对 MemoryLake、Mem0、向量 RAG 和长上下文四种内存后端进行了匹配系统级比较，评估它们在五个任务域的端到端成功率。

**💡 创新点**

创新点在于提出并验证了结构化多轨内存架构 MemoryLake，在保持确认结论、支持证据和可复用经验三条轨道的同时，使用优先级与按需检索策略提升多任务推理性能。

**🔧 技术方法**

使用的技术包括 gpt-5-mini 作为任务代理与内部检索的语言模型，bge-m3 作为稠密检索嵌入器，Mem0 的提取式事实存储，向量 RAG 的文本嵌入检索，以及将全部历史轨迹展平到提示中的长上下文方案。

**📊 数据集**

数据集为 MemoryArena 发布的公开任务集合，包含数学推理（40 篇）、物理推理（20 篇）、旅行规划（30 组）、购物包装（150 包）以及进展检索（20 个子查询）。

**📈 对比分析**

比较方法为在同一代理框架、相同模型别名（gpt-5-mini）以及统一的评分脚本下跑完所有任务，计算每个域的 SR 并求宏平均。MemoryLake 在数学、物理和进展检索域表现最佳，宏平均 SR 为 20.5%，高于最佳对照的 13.6%。

**⚠️ 局限性**

局限性包括样本量小、未进行配对统计检验、部分域使用子集、不同后端同时改动多项机制导致难以归因、嵌入器不匹配、未对资源使用（token、延迟、成本）进行平衡，以及 MemoryLake 内部实现未公开。

---

## 269. Demystifying Agent Skills: Why They Work-Until They Don't

**arXiv ID:** 2608.14036 | [PDF](https://arxiv.org/pdf/2608.14036v1)

**作者:** Zhiyuan Jiang `[一作]` (Princeton University), Yijiang Li `[通讯]` (UC San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型代理中技能（skill）的使用进行了受控实验和轨迹对比分析，探究技能何时有用、为何有效以及何时失效。

**💡 创新点**

提出了三类高层类别与十二种技能使用模式的对比税onomies，揭示技能主要通过程序锚定提升执行稳健性，而非单纯知识注入，并指出检索、适配与上下文不匹配是主要失效根源。

**🔧 技术方法**

采用对比实验、轨迹配对分析、嵌入检索、手工与LLM标注判定等技术，对技能与工作流记忆、无提示等进行多维度对照。

**📊 数据集**

使用 Terminal-Bench 2.0、Terminal-Bench‑Pro 与 SkillsBench 等基准，收集 8,135 条试验记录并抽样 240 条轨迹进行标注。

**📈 对比分析**

与原始执行、工作流记忆、无提示等对照后，技能在原始 55.9% 基线上提升至 61.9%，相对工作流记忆提升 6.06pp；执行层错误率下降至 23.5%；检索实验表明实际使用精度随库大小下降（29.6%→3.3%），但任务成功率基本保持（36%→39%）。

**⚠️ 局限性**

实验仅覆盖命令行/工具使用的基准，模型与框架覆盖有限，标注样本仅占 3%，可能遗漏稀有行为，未涵盖长周期交互或开放式协作场景。

---

## 270. When Personal Memory Has No Single Answer: Evaluating LLM Agents under Irreducible Conflict

**arXiv ID:** 2608.13921 | [PDF](https://arxiv.org/pdf/2608.13921v1)

**作者:** Lu Yang `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TANGLE基准评估LLM代理在个人记忆冲突中的认知行为。

**💡 创新点**

将冲突评估从单一答案恢复转为评估冲突感知、因果推理、置信校准、澄清寻求和记忆忠实度，并引入冲突感知行动策略CAAP。

**🔧 技术方法**

使用大语言模型（Claude、Gemini、DeepSeek、GLM、GPT-4o）结合Oracle与Pipeline两轨、CAAP决策两阶段等技术。

**📊 数据集**

构造包含541实例、40人格、3种冲突类型（CPC、BOC、SCC）的TANGLE数据集，包含精心生成的记忆与同域干扰，以及对应的多轮对话。

**📈 对比分析**

对比Oracle和Pipeline轨道、不同内存抽取器以及固定策略与CAAP，在Oracle下模型表现达15/20以内，Pipeline存在全/部分/无冲突可观测度差异，CAAP相较固定策略提升校准与澄清得分。

**⚠️ 局限性**

局限在于记忆提取缺失冲突结构、干扰噪声导致推理衰退、模型仍难以在BOC等复杂冲突中实现充分的因果推理和行动决策。

---

## 271. Towards Scaling Qualitative Analysis of Video Data

**arXiv ID:** 2608.13594 | [PDF](https://arxiv.org/pdf/2608.13594v1)

**作者:** Shiyi He `[一作]` `[通讯]` (University of Utah), Shiyi He (University of Utah)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了QualiVision，一个集成视频播放、转录、编码、电子表格同步与可视化/AI辅助的交互式工作空间，用于扩展视频定性分析。

**💡 创新点**

创新点在于将工作流碎片化问题通过表格后台同步与多流编码实现整合，并使用可视化与AI作为启发式辅助，支持反思性迭代式意义建构。

**🔧 技术方法**

使用技术包括视频播放器、转录编辑、Google Apps Script同步桥、电子表格存储、可视化仪表盘和AI提示生成。

**📊 数据集**

采用自身的用户研究视频数据集作为实验素材，没有使用公开数据集。

**📈 对比分析**

计划通过对比传统多工具工作流与QualiVision的比较结构观察法（CSO）与案例研究来评估，预期能缩短任务完成时间、降低认知负荷并提升可用性，但尚未得到量化结果。

**⚠️ 局限性**

主要限制包括：技术层面需要稳定的数据模型与双向同步，AI辅助的界限未确定，且目前尚无完整评估结果，系统仍处于原型阶段。

---

## 272. The conditional superiority of fast silicon sampling

**arXiv ID:** 2608.14079 | [PDF](https://arxiv.org/pdf/2608.14079v1)

**作者:** Nickolas Hock Yuen Lam `[一作]` (Nanyang Technological University), Xiangyu Ma `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在新加坡国家代表性调查样本上，使用大语言模型生成硅样本（synthetic surrogates），并对比快速（fast）与慢速（slow）两种采样模式的算法准确度与效率。

**💡 创新点**

提出在前沿模型下，快速硅采样在效率和准确度上不逊于传统慢速模式，甚至更优；同时将多重对应分析与Procrustes比较作为量化评估关系一致性的创新工具。

**🔧 技术方法**

采用 GPT‑5.4 大语言模型进行零/一次提示，使用多重对应分析 (MCA)、Cramer's V、Jaccard 距离、Procrustes（Gower's M^2 与 Tucker's φ）等统计技术评估样本质量。

**📊 数据集**

以 2017–2022 年新加坡 World Values Survey 第 7 波（n=2,012）为真实人类参照数据，构建 30×2,012 份硅样本。

**📈 对比分析**

通过比较均值差异、方差保留比例、Jaccard 距离、Cramer's V 相关系数以及 MCA 形状相似度，发现快速采样在均值、协方差和关系结构上与慢速相当或更好；计算时间提升约 70%，token 消耗降低约 35%。

**⚠️ 局限性**

硅样本整体失真：显著低估意见方差、某些道德条目完全同质化；关联结构与空间与真实样本差距大；对新加坡文化的代表性不足，方法仍处于早期发展阶段，需谨慎使用。

---

## 273. QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction

**arXiv ID:** 2608.13966 | [PDF](https://arxiv.org/pdf/2608.13966v1)

**作者:** Vincent Counathe `[一作]` (Cornell University), Tianyi Zhang `[通讯]` (Together AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种在 QAT 训练循环中持续进行轻量级损失感知重构的算法 QUASAR，降低低位数模型的损失阈值。

**💡 创新点**

创新点在于每步训练时动态搜索裁剪区间并用指数移动平均平方梯度估计的显著性加权最小二乘拟合解码器，直接优化重构映射，消除前向重构与反向更新的误差。

**🔧 技术方法**

使用了指数移动平均梯度显著性估计、裁剪区间搜索、显著性加权最小二乘拟合、损失感知重构误差分析以及理论证明其在 QAT 收敛界中的唯一影响项。

**📊 数据集**

在 Qwen3、Llama-3.1 语言模型上进行实验，并在标准评估集、数学推理基准以及 NVFP4 量化格式下进行验证。

**📈 对比分析**

与多种 PTQ（AdaRound、BRECQ、YAQA 等）和 QAT（LSQ、LLM‑QAT、BitDistiller、QLoRA 等）方法比较，QUASAR 在 2~4 位时实现最低持出 KL 散度，3、4 位至少降低 10%，2 位至少降低 29%；在八个任务平均精度提升 3.5–4.3 个百分点；在数学推理基准比 QAT+PTQ 提升至少 10.9 个百分点。

**⚠️ 局限性**

局限在于训练循环中需要额外搜索与拟合，虽然轻量但仍增加计算开销；目前仅在整数量化和 NVFP4 中验证，对 MXFP4 等其他低精度格式或更低比特宽度的泛化尚未评估，且理论基于 PL 条件，适用范围有限。

---

## 274. From Prediction to Intervention: Personalized Meal-Level Glucose Regulation via an LLM Agent

**arXiv ID:** 2608.13581 | [PDF](https://arxiv.org/pdf/2608.13581v1)

**作者:** Mingyu Huang `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Shuqiang Jiang `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出基于物理模型的个性化葡萄糖预测器PAGP与基于LLM的双阶段迭代餐食优化Agent（PD‑2SMO），实现从预测到干预的闭环个体化血糖调控；

**💡 创新点**

①在高频葡萄糖波动建模中加入可学习的时间生理吸收衰减模块TPADM；②在餐食优化中采用分布优先、必要时才换料的两阶段策略；③使用基于Prompt的OPRO迭代优化，让LLM在预测反馈驱动下持续改进方案；

**🔧 技术方法**

物理建模（TPADM）、变分模态分解+LSTM+Transformer、BERT语义编码、LLM（Qwen3‑Max等）Prompt+OPRO、规则约束过滤；

**📊 数据集**

上海T1DM、上海T2DM、CGMacros、BIG IDEAS四个公共CGM+饮食/运动/药物记录数据集；

**📈 对比分析**

与LSTM、GluNet、GlucoNet、GluFormer等基线在120min预测RMSE上取得显著下降（如SH T2DM RMSE从13.24→10.39），在餐食优化上iAUC和ΔG均显著低于原始餐和ReAct，对不同预测器/LLM基座亦保持优势；

**⚠️ 局限性**

依赖充足的历史CGM与个体化训练，冷启动性能有限；优化方案仅通过模型预测评估，缺乏前瞻性人类试验；LLM可能产生幻觉或偏见，需人工审核与安全过滤；

---

## 275. How Compliant is Sepsis Treatment? An Expert-Guided Neuro-symbolic Pipeline for Generating Clinical Compliance Insights

**arXiv ID:** 2608.13617 | [PDF](https://arxiv.org/pdf/2608.13617v1)

**作者:** Himanshu Tripathi `[一作]` (University of Alabama), Shahram Rahimi `[通讯]` (University of Alabama)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一套专家引导的神经符号管线，对MIMIC-IV中败血症病例进行语义归一化和模糊推理，以评估临床是否遵循SSC准则。

**💡 创新点**

创新点在于将LLM限定为语义归一化器，使用Sugeno模糊推理生成可量化的合规评分，并通过正则表达式与LLM互补验证，突破了纯规则或纯神经网络的局限。

**🔧 技术方法**

技术包括正则表达式+MedGemma 4b-it LLM做语义归一化，Sugeno模糊推理系统编码SSC 8条规则，使用embedding验证互补性，并对评分进行加权求平均。

**📊 数据集**

使用MIMIC-IV v3.1的2376个败血症发作病例（共2,438个事件）作为实验数据。

**📈 对比分析**

方法通过与仅使用正则或仅使用LLM的评分进行对比，验证两者一致性；最终模型平均合规度为36.7%，抗生素时效最低为0.24，显示系统能揭示临床执行缺口并与ICU停留时间呈负相关。

**⚠️ 局限性**

局限包括未评估死亡率、对提前给药的时间窗口未做调整、模糊参数固定且依赖专家设定、以及对资源受限环境中专家支撑的可扩展性不足。

---

## 276. Agentic Transaction: Towards ACID-Compliant Agent Systems

**arXiv ID:** 2608.13900 | [PDF](https://arxiv.org/pdf/2608.13900v1)

**作者:** Zhaoyan Sun `[一作]` (Tsinghua University), Guoliang Li `[通讯]` (Tsinghua University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Agentic Transaction概念和ACID兼容的代理系统框架，并实现了ACID数据代理。

**💡 创新点**

创新点在于将传统ACID属性迁移到代理执行中，定义了语义原子性、语义一致性、语义隔离和语义持久性，并通过事务探索‑执行‑验证循环实现。

**🔧 技术方法**

使用LLM推理、工具调用、事务式技能中心、置信度偏差验证、依赖感知隔离、版本化工作空间、知识图谱记忆等技术。

**📊 数据集**

在KramaBench（104个数据科学任务）上进行评估。

**📈 对比分析**

与Claude Code、Qwen3.5‑397B‑A17B等基线对比，提升整体得分10.6%，并在多跑一致性上显著降低方差。

**⚠️ 局限性**

局限在于对大规模多代理协作、动态模型演化的事务化支持尚未成熟，且需进一步验证跨模型、跨任务的鲁棒性。

---

## 277. A Structural Characterization of Entropy Functionals

**arXiv ID:** 2608.13917 | [PDF](https://arxiv.org/pdf/2608.13917v1)

**作者:** Daniel Lazarev `[一作]` `[通讯]` (Massachusetts Institute of Technology), Daniel Lazarev (Massachusetts Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造一个测度理论框架，阐明了在结构单调性与广义均值组合下可接受的熵函数，并建立了四级层次结构，说明了雷尼熵、香农熵以及新的可接受熵函数的统一归因。

**💡 创新点**

创新点在于提出了仅要求熵对绝对连续性满足上界单调的最小结构假设，并将广义均值生成器的严格凸/凹性与熵函数可接受性等价，从而解决了雷尼提出的广义均值替代算术均值的问题，并提供了构造新熵函数的方法。

**🔧 技术方法**

主要技术包括测度论、Kolmogorov‑Nagumo 广义均值理论、凸分析（对生成器的二阶导数判定）、Csiszár f‑divergence 与数据处理不等式的关联以及对齐加法性（内部与外部）与尺度选择的层次化分析。

**📊 数据集**

无数据集；论文为纯理论性研究，不涉及实验或数据。

**📈 对比分析**

无实验比较或性能评估；论文通过理论证明与层次结构展示不同熵在结构假设下的可接受性与分类，未进行数值比较。

**⚠️ 局限性**

局限性在于仅处理可积、正值的有限测度空间，未讨论无穷测度或零测度的情况；并且对熵函数的构造与适用范围仍需在具体统计模型中进一步验证。

---

## 278. Search or Chat? Comparing How We Learn About Debated Topics

**arXiv ID:** 2608.14113 | [PDF](https://arxiv.org/pdf/2608.14113v1)

**作者:** Ran Yu `[一作]` (GESIS -- Leibniz Institute for the Social Sciences), Jiqun Liu `[通讯]` (University of Wisconsin--Milwaukee)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究在194名参与者上进行了一项对比实验，比较传统搜索引擎与LLM驱动的聊天机器人（基于OpenAI GPT‑5 mini）在学习争议性话题（如“社交媒体是否对社会有益”等）中的效果，采用问卷和文本回答评估学习成果；

**💡 创新点**

创新点在于：①首次用量化指标（论点扩展与批判性推理）客观评估聊天机器人对学习的影响；②结合用户特质（态度强度、知识谦逊）与交互行为进行探索性中介分析；

**🔧 技术方法**

技术实现包括：自定义搜索界面（使用Brave API）和聊天界面（使用OpenAI GPT‑5 mini），同时利用GPT‑5 mini 进行文本编码与评分；

**📊 数据集**

数据集主要来自Prolific的在线受试者（194人），包含前后问卷、文本论点、交互日志；

**📈 对比分析**

比较方法为双因素ANOVA（工具×话题）和回归模型评估交互项；结果显示工具对学习成果（论点扩展、批判性推理）无显著差异，尽管聊天用户在交互时间上更长；

**⚠️ 局限性**

局限性包括：仅研究旧争议话题、样本为在线众包群体、使用自评与文本评分而非客观知识测验、未深入分析聊天对话深度、工具与用户特质交互效果未完全揭示。

---

## 279. SPARGen: Unifying Spatial Perception and Reasoning through Native Multimodal Generation

**arXiv ID:** 2608.14138 | [PDF](https://arxiv.org/pdf/2608.14138v1)

**作者:** Jinsheng Quan `[一作]` (Zhejiang University), Yawei Luo `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SPARGen统一多模态框架，将3D重建、对应与空间推理视为指令条件生成任务，并通过同一模型同时输出离散序列与图像对齐的几何字段。

**💡 创新点**

创新点在于把几何、对应与语义推理全部映射到模型原生生成通道，消除专用回归头和外部几何模块，并利用Mixture-of-Transformer-Experts共享表示实现多任务协同。

**🔧 技术方法**

使用Bagel的MoT架构、预训练视觉语言模型、VAE+rectified flow、自动回归生成、序列序列化与图像对齐字段等技术。

**📊 数据集**

训练使用多源数据集：视觉几何（ASE、BlendedMVS、CO3D、MVS-Synth等）、光流（TartanAir、FlyingThings3D等）以及空间推理（MindCube、OmniSpatial、OST-Bench、SPAR-Bench等）。

**📈 对比分析**

与专用几何模型DUSt3R/FLARE/VGGT以及统一模型G^2VLM对比，在深度、点图、姿态、光流和空间推理基准上均保持竞争力或取得领先，尤其在空间推理上获得最高平均准确率，零样本光流EPE仅4.09。

**⚠️ 局限性**

局限在于冻结VAE的空间压缩会限制几何边缘和高精度物理量的分辨率，且模型无法恢复度量尺度。

---

## 280. AlignFace: Human-Aligned Face Similarity Metric with Interpretable Concept Relations

**arXiv ID:** 2608.14130 | [PDF](https://arxiv.org/pdf/2608.14130v1)

**作者:** Ying Huang `[一作]` (National University of Singapore), Brian Y. Lim `[通讯]` (National University of Singapore)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种面部相似度度量模型AlignFace，结合人类面部相似度的认知原理（特征与构造属性、非线性缩放、群体偏差）进行解释性设计。

**💡 创新点**

创新点在于：①将认知心理学原理融入模型架构，实现面部属性的可解释性；②使用视觉-语言模型与属性门控交叉注意力提取属性差异；③采用概念瓶颈与神经通用加性模型（GAM）实现非线性属性对整体相似度的加性贡献；④构建大规模人类主观相似度标注数据FACETS。

**🔧 技术方法**

技术包括：视觉-语言模型（VLM）编码、属性门控交叉注意力（AGCA）、概念瓶颈模型（CBM）、神经通用加性模型（GAM）以及多任务triplet学习。

**📊 数据集**

使用CMU Multi-PIE与CelebA两大公开人脸数据集，并在此基础上合成多属性编辑图像，构成FACETS数据集。

**📈 对比分析**

通过与传统度量（PSNR、SSIM、SimCLR、CLIP、LPIPS等）和人脸识别模型（FaceNet、ArcFace等）进行对比实验，使用2AFC人类标注作为基准。AlignFace在整体相似度与属性相似度上的人类一致率分别达到约80%及0.79以上，显著优于所有基线。

**⚠️ 局限性**

局限性在于：①数据集中人群多样性不足，尤其是少数族裔样本较少；②实验仅在受控的相似度判断任务，未覆盖真实多场景应用（如隐私遮蔽、化妆转移等）；③对极端难以区分的“硬例”处理有限，可能导致模型在极小差异下的鲁棒性不足。

---

## 281. Mandato: Protocol-Level Enforcement of Digitally Signed Mandates on AI Agent Actions with Cryptographically Chained Audit Trails

**arXiv ID:** 2608.14074 | [PDF](https://arxiv.org/pdf/2608.14074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 282. Bisimulations and Modal Logics for Higher Dimensional Automata

**arXiv ID:** 2608.14117 | [PDF](https://arxiv.org/pdf/2608.14117v1)

**作者:** Safa Zouari `[一作]` (Norwegian University of Science and Technology), Krzysztof Ziemiański `[通讯]` (University of Warsaw)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对高维自动机（HDA）的路径相邻关系进行分解，定义了一系列介于ST-和hhp-平价之间的新等价关系，并为这些等价关系提供了对应的模态逻辑表述，尤其首次给出了hhp-平价在HDA上的模态逻辑刻画。

**💡 创新点**

创新点主要包括：
- 将传统的路径相邻关系拆分为相似（ℓ-相似）和抽象（ℓ-抽象）两种独立的操作，打破了以往对对称相邻的束缚；
- 基于上述拆分构造了一族新的中间平价（如半历史保持、准历史保持等）并证明了它们之间严格的包含关系；
- 提出(I,M)-系统框架，将任意平价关系转化为对路径的强平价，从而得到通用的Hennessy‑Milner定理；
- 通过该框架给出所有平价的模态逻辑特征子语言，首次完成了hhp-平价的模态逻辑刻画。

**🔧 技术方法**

采用的技术方法有：
- 高维自动机的前立方（precubical）定义与路径语义；
- 路径相邻的相似与抽象操作的形式化定义与证明；
- ST-轨迹（ST-trace）的构造与利用；
- (I,M)-系统的定义与Hennessy‑Milner定理的证明；
- 通过构造具体HDA示例进行严格的分离性与包含性证明。

**📊 数据集**

本文没有使用传统意义上的实验数据集，而是通过构造若干离散的高维自动机示例（如Y1、Y2、X1、X2等）来展示新的等价关系及其区别。

**📈 对比分析**

比较方法主要是理论证明：
- 通过构造路径相邻关系的闭包与递归定义，证明各平价之间的包含关系；
- 通过分离示例来证明等价关系的严格性；
- 通过模态逻辑公式来区分不同平价的判定。由于是理论性质，未涉及运行时性能或实验指标。

**⚠️ 局限性**

局限性与开放问题包括：
- 对某些等价关系（如reST与ureST之间的关系）仍存在未能完全判定的情况，留有开放问题；
- 等价关系的判定算法复杂度未被分析，尚未实现可用的工具；
- 目前的模态逻辑框架主要针对路径而非单元格，未来可能需要进一步推广；
- 论文未给出对实际并发系统的验证工具，未来工作需要将理论转化为实践工具。

---

## 283. A Graph-Based Reinforcement Learning Framework for Structured Drift Diagnosis and Recovery in Autonomous LLM Agents

**arXiv ID:** 2608.14109 | [PDF](https://arxiv.org/pdf/2608.14109v1)

**作者:** Ismail El Hamraoui `[一作]` (Assystem EOS), Robert Plana `[通讯]` (Assystem EOS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种图形化框架，将LLM代理的行为漂移后恢复任务拆解为五个诊断节点，并使用单个小型LLM通过GRPO进行节点级微调，实现可插拔的后置恢复模块；

**💡 创新点**

创新点在于将漂移恢复转化为小型状态机的路由遍历，采用单一小模型在不同节点上实现角色特化，并利用结构化XML/JSON输出与LLM‑as‑Judge相结合的复合奖励，在不需重新训练主任务模型的前提下实现自动化漂移诊断与回滚；

**🔧 技术方法**

使用的技术包括GRPO强化学习、规则化结构化奖励与LLM‑as‑Judge语义评分、XML/JSON固定输出格式、LoRA微调、以及AppWorld环境模拟；

**📊 数据集**

使用的数据集为AppWorld基准任务集（涵盖约450个API），并在训练期间通过注入恶意指令构造漂移轨迹；

**📈 对比分析**

通过与无训练基线、GPT‑4o/mini恢复模型以及无漂移上限进行对比，实验表明训练后的Granite 3.3 2B在Type I/II漂移下的任务成功率可达到70–80 %相当于无漂移基准的约70–80 %，虽略低于GPT‑4o恢复，但模型规模和部署成本显著更低；

**⚠️ 局限性**

局限性包括：仅针对读取类漂移（Type I/II）不支持写漂移（Type III）不执行逆操作；依赖外部漂移起始检测；LLM‑as‑Judge缺乏领域知识可能导致误判；奖励设计与奖励偏差风险；缺乏真正的闭环纠错节点。

---

## 284. Human and Artificial Intelligence - Promoting Trustworthy and Understandable Collaboration

**arXiv ID:** 2608.14291 | [PDF](https://arxiv.org/pdf/2608.14291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 285. A Pathway to General-Purpose Scientific AI: Multimodal Comprehension of Scientific Images

**arXiv ID:** 2608.14075 | [PDF](https://arxiv.org/pdf/2608.14075v1)

**作者:** Jennifer D'Souza `[一作]` (TIB Leibniz Information Centre for Science and Technology), Thomas Frederik Jan van Roeden `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并公开了面向原子层沉积与蚀刻（ALD/E）领域的多任务科学图像理解基准（ALD/E-ImageMiner），并在ICDAR 2026竞赛中组织了分类、表格提取、摘要生成与视觉问答（VQA）四个子任务。

**💡 创新点**

创新点在于：①将专家标注的实验与模拟图像与Bloom修订后的认知层级结合，设计多层次、面向科学推理的VQA问题；②提供面向科学图像的全流程评估框架，兼顾语义识别、数值提取、概念推理与证据引用；③提出从“识别”向“创造”扩展的未来研究方向，强调跨面板、跨文献推理与不确定性处理。

**🔧 技术方法**

技术上使用了跨模态视觉语言模型（如GPT‑4V、LLaVA‑Med 等），结合自定义的多模态注释与评估脚本，构建了图像裁剪、面板定位、图表到表格转换、文本摘要与问答答案生成的完整流水线。

**📊 数据集**

使用的数据集为 ALD/E-ImageMiner，共 1,951 张来自 205 篇 ALD/E 文献的实验与模拟图像，涵盖 49 类图形类型，并提供面板级坐标、分类标签、表格结构、摘要与 VQA 注释。

**📈 对比分析**

在竞赛中将多种现有视觉语言模型作为基线进行比较；结果显示，模型在图像分类与数值提取上表现相对稳定，但在 Bloom 级别较高的 VQA 问题（需跨面板推理、因果解释或应用评估）时准确率显著下降，说明当前模型对科学图像的深层推理仍有不足。

**⚠️ 局限性**

局限性包括：①数据集仅覆盖 ALD/E 领域，缺乏跨学科通用性；②多模态模型易产生幻觉，尤其在数值读取与符号识别方面；③缺乏跨文献、跨实验的综合推理与证据验证机制；④评价指标多为单任务指标，未能完整衡量模型的整体科学推理与创造性能力。

---

## 286. Experimental Study on System-Level Performance Impact of Read Disturbance in Modern SSDs

**arXiv ID:** 2608.14073 | [PDF](https://arxiv.org/pdf/2608.14073v1)

**作者:** Yonggon Park `[一作]` (POSTECH), Jisung Park `[通讯]` (POSTECH)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对15款现代 NVMe SSD 进行大规模实验，系统级量化读扰动对 I/O 性能的影响，并设计基于读扰动的 SSD 性能攻击模型。

**💡 创新点**

①首次从系统角度全面衡量读扰动导致的性能波动与设备差异；②提出利用读扰动的 DoS 攻击方案；③给出六条针对主机与 SSD 侧改进方向。

**🔧 技术方法**

使用 FIO 生成顺序/随机读负载、S.M.A.R.T. 日志收集、内置读扰动累积与攻击模式，并统计带宽、IOPS、尾延迟等指标。

**📊 数据集**

采用自建的工作负载：填充 450 GB 数据、顺序/随机读、FileBench 混合读写；未使用公开数据集，全部为实验自制负载。

**📈 对比分析**

通过与厂商规格、平均值对比评估；实际带宽平均低于规格 36% 以上，峰值下降 80% 以上；攻击能使受害者读带宽下降 50%–70%；在读写混合负载下读扰动影响显著降低。

**⚠️ 局限性**

仅覆盖 15 款 NVMe SSD，未评估 SATA 或 2D NAND；实验环境为单机，缺乏多租户场景；攻击参数未完全优化，未给出完整实现细节。

---

## 287. From Fixed Grids to Moving Particles:A Transferable Latent Operator for Fluid Dynamics

**arXiv ID:** 2608.14120 | [PDF](https://arxiv.org/pdf/2608.14120v1)

**作者:** Meng Li `[一作]` (Shanghai Artificial Intelligence Laboratory), Huaxi Huang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可迁移潜在算子（Transferable Latent Operator, TLO），在仅使用固定格点Eulerian场数据训练的情况下，实现对Lagrangian粒子轨迹的零样本（zero‑shot）闭环推演，并能在粒子离开原始Eulerian网格后继续推演。

**💡 创新点**

创新点包括：1）将流动潜在演化与坐标依赖解码器解耦，形成统一的潜在表示；2）利用可变坐标查询实现从Eulerian场预测到粒子轨迹的无监督迁移；3）在同一模型中实现网格外（out‑of‑grid）粒子推演；4）通过稀疏粒子速度微调实现轻量级适配。

**🔧 技术方法**

技术手段包括：基于编码–处理–解码的神经算子架构；层次化 patch 编码器（cross‑attention）提取局部与全局特征；潜在处理器（attention‑slice 机制）实现坐标无关的时间演化；坐标条件解码器（cross‑attention）对任意查询点进行读取；Fourier 特征编码与位置嵌入；单步Eulerian场监督与自回归推演；以及稀疏解码器微调。

**📊 数据集**

使用五个流体动力学基准：DAM2D (SPH)、NS2D (Navier‑Stokes 2D)、TGV3D (Taylor–Green 3D)、Burgers3D (非线性扩散 3D) 和 SEA (海面再分析 2D)。此外，针对三种基准（DAM2D、Burgers3D、SEA）进行粒子轨迹推演评估。

**📈 对比分析**

与多种基准方法（LSM、GINO、GNOT、Transolver、UPT、LNO、DeepLag、GNS）对比。TLO 在 Eulerian 预测（Eul）和零样本 Lagrangian 轨迹（Path）上均达到或超过最优结果；在 Ref（粒子路径上的速度误差）也表现最优。稀疏解码器微调后性能进一步提升，甚至在 DAM2D 上超过专门的粒子基准 GNS，同时显著降低内存与运行时间。

**⚠️ 局限性**

局限性包括：① 仍需高质量、完整的 Eulerian 网格数据；② 对于极大尺度或极稀疏的 Eulerian 观测可能表现不佳；③ 单步训练后需自回归推演，长期推演仍可能累计误差；④ 该方法在不同物理量（如压力、密度）与复杂边界条件下的适应性尚待进一步验证。

---

## 288. Acoustic UAV Detection in Battlefield Scenarios: Handling Noise, Domain Shift, and Weak Labels

**arXiv ID:** 2608.14287 | [PDF](https://arxiv.org/pdf/2608.14287v1)

**作者:** Vadym Vilhurin `[一作]` (Igor Sikorsky Kyiv Polytechnic Institute), Andrii Shevtsov `[通讯]` (Zvook)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种针对战区环境下的噪声、域迁移和弱标签的声学无人机检测框架。

**💡 创新点**

结合PCEN特征、注意力池化、域感知训练、辅助类别与多麦克风数据，实现了对跨域噪声干扰的鲁棒性。

**🔧 技术方法**

使用ConvNeXt‑Tiny骨干、频率投影层、注意力池化、PCEN谱图、混合噪声Mixup、RMS掩蔽、SWA等深度学习技术。

**📊 数据集**

采用来自乌克兰前线的300,000+条9秒音频，包含Mic‑1与Mic‑2两种麦克风，分为训练、验证、测试三组。

**📈 对比分析**

与多种公开和专有基线模型对比，最终ensemble在全数据集上实现82.2% F1，显著高于AST‑Drone 55.4%。

**⚠️ 局限性**

仅处理单个小无人机类别、缺乏时序与距离元数据、对新的麦克风硬件与未知场景的迁移仍有限。

---

## 289. RankT2I: A Submodular Framework for Discovering Interpretable and Diverse Semantics in Text-to-Image Models

**arXiv ID:** 2608.14226 | [PDF](https://arxiv.org/pdf/2608.14226v1)

**作者:** Ritika Allada `[一作]` (Virginia Tech), Pinar Yanardag `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了RankT2I，一种无训练、模型无关的框架，用于自动发现文本‑图像模型中可编辑且多样化的语义；

**💡 创新点**

创新点在于将语义发现建模为子模量集合选择问题，联合优化相关性、可编辑性与多样性，并通过预训练的多模态视觉‑语言模型生成候选语义；

**🔧 技术方法**

核心技术包括多模态视觉‑语言模型（如GPT‑Image‑1.5、CLIP）生成语义候选、基于扩散或FLUX模型生成编辑图像，以及基于子模量覆盖函数的贪婪选择算法；

**📊 数据集**

实验数据涵盖时尚、景观、艺术、面部等多领域图像，使用公开和闭源模型（SDXL、FLUX Schnell、InstructPix2Pix、Ledits++、GPT‑Image‑1.5）生成的图像；

**📈 对比分析**

与NoiseCLR、SliderSpace以及多种属性检索方法相比，RankT2I在多样性（TCE、TIE）、可解释性（CLIP‑T、TIFA、VQA）和编辑质量（I2I）上均取得更高分，同时在发现100个语义所需时间上明显更快；

**⚠️ 局限性**

局限性包括依赖多模态语言模型的召回与幻觉，阈值化可编辑性分数对不同模型和域不一定通用，且对深度伪造的预防仍需进一步完善。

---

## 290. MazeRunner: Nonlinear Task and Clue Orchestration for LLM-driven Black-Box Automated Penetration Testing

**arXiv ID:** 2608.14216 | [PDF](https://arxiv.org/pdf/2608.14216v1)

**作者:** Zhenyuan Li `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 MazeRunner，一套基于多智能体（策划者-执行者-评审者）循环与持久化 Task & Clue Cache 的自动化渗透测试框架，能够在黑盒环境中动态构建攻击图、纠正失败并跨阶段重用线索。

**💡 创新点**

核心创新在于：① 将任务与线索拆分为可持久化的图结构，解决上下文遗忘与任务冗余；② 引入独立的评审者 Agent 对失败进行诊断与反馈，跳出“深度优先陷阱”；③ 通过协作循环实现从任务生成到执行、失败回溯的闭环，显著提升非线性渗透路径的探索与成功率。

**🔧 技术方法**

使用了大语言模型（Claude Sonnet 4.5、GPT‑5.2 及 Codex 版本）作为推理与代码生成核心；实现了 PTY‑based 交互式执行、非交互式扫描；Task & Clue Cache 采用 MySQL 存储；多智能体架构基于 Kimi Code CLI 与自定义协作协议。

**📊 数据集**

评估数据集为 10 个 2026 年 2‑4 月发布的 Hack The Box（HTB）机器（4 易、4 中、2 难），每台机器的目标在 LLM 训练截止日期之后，避免知识泄露。

**📈 对比分析**

对比基准系统 PentestGPT‑V2 与 Claude Code（在同一 LLM 基础上）并在相同 2000 万 token 预算下测评。MazeRunner 在 Claude Sonnet 4.5 下完成 47.7% 子任务，获得 6 只用户级或更高级 shell（含 2 次 root），而基准仅 36.2%/34.2% 子任务完成率，且仅 2 只用户级 shell、无 root。探索宽度平均提升 26%，且执行效率（每 100M token 获得 shell 量）明显高于基线。

**⚠️ 局限性**

局限性包括：无法处理多模态或 GUI 交互任务；对未知漏洞或缺少公开 exploit 的场景仍难以突破；LLM 的统计偏好导致稀有攻击路径被低效探索；评审者与 Cache 的性能受限于外部存储延迟与设计复杂度。

---

## 291. How Much Do Legal RAG Systems Still Hallucinate?

**arXiv ID:** 2608.14210 | [PDF](https://arxiv.org/pdf/2608.14210v1)

**作者:** Souvick Das `[一作]` (University of Luxembourg), Domenico Bianculli `[通讯]` (University of Luxembourg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对八种法律检索增强生成（RAG）系统在GDPR和CIVIL两大法律语料库上进行细粒度的幻觉行为分析，覆盖答案级别和主张级别评估，并按问题类别和用户角色进行分组，最后在独立专家编写的问答集上进行外部验证。

**💡 创新点**

①提供了跨问题类别与用户角色的主张级幻觉细化分析，揭示幻觉率随问题类型和提问者不同而显著变化；②验证了评估趋势在Benchmark之外仍成立；③发现GPT-5在严格幻觉度上优于GPT-4，强调检索质量单一提升不足。

**🔧 技术方法**

使用基于主张的事实核查技术（如Claim Extraction + Entailment / RefChecker），引入CL、CL^s、AL、AL^s等指标以及支持文档的严格/非严格对比；通过统计不支持主张分布（H0、H1、Hk）进一步刻画幻觉严重度。

**📊 数据集**

主要数据集为ClaimRAG-LAW Benchmark（GDPR 149条QA，CIVIL 168条QA），并收集了142条由法律专家手工编写的问答，用于外部验证。

**📈 对比分析**

将8个RAG系统（检索器 BM25/E5 × 生成器 GPT‑4/GPT‑5/Llama‑3.1‑8B‑Instruct/Mixtral‑8x7B）按CL、AL、CL^s、AL^s进行横向对比；结果显示 BM25+GPT‑5 的幻觉率最低（<10%），Llama‑3.1‑8B 系统最高（≈50%）；GPT‑5 在严格幻觉度上显著优于 GPT‑4，外部验证保持了同一排名趋势。

**⚠️ 局限性**

局限性：仅覆盖两大司法语料与有限的法律问题类别（FP、JT样本少）；评估依赖Benchmark标注，未考虑不同提示、分块、检索参数或新模型的影响；跨司法、跨语言的推广性尚未验证。

---

## 292. Revisiting Energy-based Tabular Anomaly Detection: Energy and Reconstruction are Complementary

**arXiv ID:** 2608.14186 | [PDF](https://arxiv.org/pdf/2608.14186v1)

**作者:** Junichiro Niimi `[一作]` `[通讯]` (Meijo University), Junichiro Niimi (Meijo University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 Deep Boltzmann Machine（DBM）的均值场能量作为表格异常检测的显式能量分数，并与八个经典与现代基线（Isolation Forest、OCSVM、LOF、Autoencoder、VAE、COPOD、ECOD、Deep SVDD）进行多种子统计对比；进一步探索 DBM 能量与 Autoencoder 重构误差的融合，验证两者互补并能显著提升检测性能。

**💡 创新点**

首次将经典 DBM 的均值场能量引入表格异常检测，证明其与重构误差形成非冗余互补视角；发现仅 DBM 导出的分数能有效提升 Autoencoder 的集成效果，突出能量模型在该任务中的独特价值；同时展示了通过 rank fusion 或凸组合可实现的显著性能提升。

**🔧 技术方法**

使用两层隐藏的 Deep Boltzmann Machine，训练采用 Persistent Contrastive Divergence 与 Adam；利用 mean-field 推断计算能量分数；对比基线包括传统密度代理（Isolation Forest、OCSVM、LOF）、重构模型（Autoencoder、VAE）以及非参数分数（COPOD、ECOD、Deep SVDD）；融合方法采用参数无关的 rank fusion 与留一种子调优的线性组合；对结果进行配对 t 检验；预处理将连续特征量化为 5 级，类别特征 one‑hot 编码。

**📊 数据集**

UCI Bank Marketing（银行营销）和 NSL‑KDD（网络入侵检测）两大表格数据集，分别代表营销和网络安全领域。

**📈 对比分析**

在 20 种随机种子下与八个基线进行对比，单方法性能显示 DBM 能量与 Autoencoder 相当或略优（在 NSL‑KDD 上统计显著提升）；融合后，rank fusion 或最优线性组合可使 AUROC 分别提升约 0.014（Bank Marketing）和 0.0021（NSL‑KDD），且均为显著改善。

**⚠️ 局限性**

实验仅涵盖两数据集，未覆盖最新深度表格异常检测器；DBM 采用伯努利可见层，需对连续特征离散化，可能导致信息损失；训练成本显著高于传统基线，尽管推理成本低。

---

## 293. Optimal Pricing and Charging Strategy Design for Non-cooperative Battery Swapping Stations

**arXiv ID:** 2608.14167 | [PDF](https://arxiv.org/pdf/2608.14167v1)

**作者:** Huanyu Yan `[一作]` (Chinese University of Hong Kong Shenzhen), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种层级非合作博弈模型，用于在同一聚合器下同时求解电池交换站（BSS）的定价与充电策略，并证明了子博弈完美纳什均衡（SPNE）的存在与唯一性，随后给出基于最佳响应动力学的收敛算法，并通过真实的西安电动车换电站数据验证所提策略能够显著提升站点利润与聚合器峰值削减效果。

**💡 创新点**

创新点包括：①首次将BSS定价与充电问题构建为层级博弈并在聚合器TOU定价框架下分析；②严格证明了两层博弈均存在唯一的纯策略SPNE，提供了理论上最优的协同解；③提出基于最佳响应的闭式解与迭代算法，收敛性得到数学证明；④设计了预测误差处理方法，用以应对实际换电需求波动；⑤通过实验展示了所提策略在利润和峰值削减上的显著优势。

**🔧 技术方法**

技术手段包括：游戏理论（子博弈完美纳什均衡、Rosen 条件、对角严格凹性）、凸优化（二次规划求解充电策略）、最佳响应动力学（迭代求解均衡）、线性TOU定价模型、预测误差处理的二次规划。

**📊 数据集**

使用的数据集：西安地区12个BSS的真实换电需求、充电站电池容量与充电速率、EV距离与数量参数，以及基于新加坡平均负荷的聚合器基准负荷。为验证可扩展性，还随机生成了36个BSS的参数进行实验。

**📈 对比分析**

与基准方案（无协调定价/充电、单站TOU定价、仅站点自我优化等）进行对比。实验显示：所提NE‑NE策略使单站日均利润提升18.1%至超过40%；BSS 7周利润提升约22%（约71,000元对比58,400元）；系统社会福利提升近两倍（约961k元对比467k元）；峰值削减约1.9%（相较无协调充电）。

**⚠️ 局限性**

局限性：假设所有参数公开且完全信息；仅考虑固定换电价，未考虑时变或动态定价；未考虑V2G能量注入；对异构EV偏好时唯一性缺乏严格保证；扩展至更大规模时可能需要更高效的多点搜索或群体优化方法。

---

## 294. A Temporal Barrier Framework for Collision Avoidance in Multi-Agent Autonomous Aerial Vehicles

**arXiv ID:** 2608.14239 | [PDF](https://arxiv.org/pdf/2608.14239v1)

**作者:** Benedikt Barthel Sorensen `[一作]` (Massachusetts Institute of Technology), Themistoklis Sapsis `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了对抗性时间碰撞距离（aTTC）指标，并将其嵌入控制障碍函数（CBF）框架，实现多无人机在动态、不可预知、可能存在敌对环境下的安全协同飞行；

**💡 创新点**

创新点在于：①从时间维度重新定义碰撞风险（aTTC），确保在三维空间中总是有限且可计算；②设计了可微分的神经网络近似器，用于实时估计aTTC，从而避免昂贵的动力学积分；③将aTTC直接作为CBF阈值，得到具有前瞻性、低保守性的碰撞避免控制器；

**🔧 技术方法**

采用控制障碍函数（CBF）和高阶CBF、神经网络（带FiLM调节）、仿射控制、随机动力学与扰动模型、比例导向追踪等技术；

**📊 数据集**

使用基于3D Dubins型无人机动力学的仿真数据：10,000,000 条样本，涵盖不同速度、相对位置、不同目标和敌手速度的组合；

**📈 对比分析**

与传统高阶 CBF（HOCBF）和无 CBF 基线在独立多体与编队飞行的追逐-逃逸场景下进行比较。aTTC-CBF 在无敌手、慢速敌手和高速敌手情形下：碰撞率比 HOCBF 低 2–3 倍，航迹进度比 HOCBF 高 1.5–2 倍；在编队情形下，保持编队一致性更好，核心队列偏差减半；

**⚠️ 局限性**

局限性包括：①在非敌对环境下提升空间有限；②aTTC 对边界情况梯度不光滑，需大量训练样本；③假设已知对手动力学，若未知需自适应更新；④神经网络近似误差在训练分布边缘仍可能导致安全性下降。

---

## 295. Convex losses and their applications to SVM, SVR, and Shallow Neural Networks

**arXiv ID:** 2608.14288 | [PDF](https://arxiv.org/pdf/2608.14288v1)

**作者:** Filippo Portera `[一作]` `[通讯]` (Università Ca' Foscari), Filippo Portera (Università Ca' Foscari)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了多种新的凸损失函数，并在SVM（使用粒子群优化求解原始形式）和浅层神经网络中实现和评估。

**💡 创新点**

创新点在于将模式相关矩阵（F）嵌入损失函数，构造了包含样本间相似度的加权损失，并给出了对应的SVM/NN理论推导。

**🔧 技术方法**

主要技术包括粒子群优化（PSO）、核方法、嵌套交叉验证、自动微分（PyTorch）以及多种基于径向基函数的F矩阵。

**📊 数据集**

使用了七个小型UCI二分类数据集（Sonar、Haberman、Heart、Iono、WDBC、Breast、German）进行实验。

**📈 对比分析**

通过5折外循环、3折内循环的嵌套交叉验证与10次重复，分别与标准SVM/NN（BCE损失）和AdaBoost做对比；结果显示新损失在大多数数据集上与基线相当或略优，但差异不大。

**⚠️ 局限性**

主要限制包括仅在小型数据集上验证、PSO求解效率低、对F矩阵参数敏感、对大规模问题的可扩展性不足以及仍未能求解相应的对偶形式。

---

## 296. PRM-as-a-Judge 1.5: A Toolkit for Robot Process Assessment

**arXiv ID:** 2608.14284 | [PDF](https://arxiv.org/pdf/2608.14284v1)

**作者:** Yuyang Liu `[一作]`, Xiaolong Zheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 PRM‑as‑a‑Judge 1.5 工具包，实现了对机器人执行过程的细粒度评估，并给出了详细的模型评估报告。

**💡 创新点**

创新点包括：①构建了 OPD（Outcome–Process–Diagnosis）完整度量体系，②新增三条条件诊断指标（Failure Near‑Success、Drawdown Recovery Ratio、Success Quality Score），③将进度曲线转化为可解释的评估报告。

**🔧 技术方法**

采用进度判别器（Progress Recognition Model, PRM）估计任务完成度，基于进度曲线计算多维指标，并将其整合为评估报告；实现了端到端的评估流水线。

**📊 数据集**

使用 RoboDojo 真实世界与仿真任务数据集，对数十种视觉‑语言‑动作模型（VLAs）和世界动作模型（WAMs）进行评估；实验涵盖多种任务类别（精度、长时程、通用、开放词汇）。

**📈 对比分析**

对比多种模型后发现：① VLAs 一般优于 WAMs；② 更大的模型参数并不一定带来更好表现；③ π_0.5 在大多数指标上排名靠前；③ 仿真与真实世界性能相关性弱，精细操作任务差距显著。

**⚠️ 局限性**

局限性包括：① PRM 对负向进度（回退）的监督不足；② 需要更多多模态训练数据；③ 现有判别器对新任务、视角和装置的适应性有限；④ 评估结果仍受进度曲线估计误差影响。

---

## 297. Accelerating Large-scale Bundle Adjustment for LiDAR Mapping via Parallel Computing

**arXiv ID:** 2608.14266 | [PDF](https://arxiv.org/pdf/2608.14266v1)

**作者:** Yixi Cai `[一作]` (KTH Royal Institute of Technology), Fu Zhang `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一个完整的GPU并行LiDAR Bundle Adjustment框架，能够在单GPU上高效处理大规模点云。

**💡 创新点**

创新点包括自适应异步数据加载、从最低分辨率开始的bottom-up voxel化以及基于majorization‑minimization的全并行优化。

**🔧 技术方法**

使用了CUDA并行原语（如radix sort、reduce、scan、scatter）、Thrust库、平面特征提取、LM优化和并行增量求解器。

**📊 数据集**

在HeLiPR、MaRS-LVIG和MulRan三大公开城市级数据集上进行实验。

**📈 对比分析**

与HBA和BALM3对比，平均加速约10倍，最大加速近10倍，RMSE与最优方法相当或更好，显示出显著的效率和竞争力。

**⚠️ 局限性**

局限在于仅支持单GPU，受限于显存和并行线程数；需进一步扩展到多GPU以提升更大规模数据的可扩展性。

---

## 298. Multi-Objective Bayesian Optimization for Model Merging

**arXiv ID:** 2608.14264 | [PDF](https://arxiv.org/pdf/2608.14264v1)

**作者:** Utkarsh Agarwal `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Monojit Choudhury `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MOBO-Merge 框架，将模型合并参数的选择建模为多目标黑盒优化问题，并通过多目标贝叶斯优化搜索 Pareto 前沿，实现对不同合并算子（Linear、SLERP、TIES、Block‑Linear 4x）的自动化调参；在 Qwen3‑4B 与 Llama‑3.1‑8B 的两模型（instruction‑math）和三模型（instruction‑math‑code）融合任务上进行系统实验；

**💡 创新点**

1）将多目标贝叶斯优化作为通用搜索层，兼容多种权重融合算子；2）在有限评估预算下逼近 Pareto 前沿，提供可控的能力权衡；3）系统对比不同算子、模型家族和目标集合，揭示 TIES 与 Block‑Linear 在不同场景下的相对优势；

**🔧 技术方法**

使用多目标贝叶斯优化（NEHVI）与 Gaussian Process surrogate；Sobol 初始设计、Batch/Sequential acquisition；MergeKit 实现权重融合；SLERP、Linear、TIES 与 Block‑Linear 4x 合并算子；使用超体积（hypervolume）衡量 Pareto 前沿质量；

**📊 数据集**

I/F Eval（指令遵循），GSM8K（数学推理），HumanEval/CodeGen（代码生成）等公开基准；采用 30%/70% 的 deterministic 划分为优化与 hold‑out 评估集；

**📈 对比分析**

方法对比：在每个算子、模型族与目标集合下，将 MOBO-Merge 与均匀随机搜索进行 100 次评估预算比较；评价指标为 Pareto 前沿的平均超体积；结果显示 MOBO‑Merge 在 11/12 设定上获得更高平均超体积，尤其在高维 TIES、Block‑Linear 4x 与三目标搜索中优势显著；在单维 Linear 上优势微弱，随机搜索略优；

**⚠️ 局限性**

评估成本高，尤其在长推理任务下；仅适用于共享架构的权重空间融合，无法直接扩展到异构模型或非权重融合；在高维/多目标设置中可能需要更强的 surrogate 与收敛策略；对超体积参考点敏感，评估指标有限

---

## 299. Meteorology-driven Causal Nowcasting of Fugitive Landfill Emissions Enables Proactive Public Health Response

**arXiv ID:** 2608.14254 | [PDF](https://arxiv.org/pdf/2608.14254v1)

**作者:** Timothy C. Pearce `[一作]` (University of Leicester), Alessia Freddo `[通讯]` (UKHSA)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

未提供具体论文内容，无法得知研究目标

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

## 300. Polaris : Multi Agentic System for Conversational Enterprise Analytics

**arXiv ID:** 2608.14246 | [PDF](https://arxiv.org/pdf/2608.14246v1)

**作者:** Varuni H K `[一作]` (Couchbase Inc.), Santosh Hegde `[通讯]` (Couchbase Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为Polaris的监督式多代理框架，用于对话式企业数据分析；

**💡 创新点**

创新点在于将动态任务协调（DTC）引入决策理论层面，将代理-任务分配建模为可适应的二分图匹配，实现实时协调、恢复和优化；

**🔧 技术方法**

使用了基于大型语言模型的ReAct式“先思考后行动”代理，结合SQL++、Python REPL、Seaborn/Plotly等可视化工具，并通过DTC进行资源调度；

**📊 数据集**

在Airbnb纽约房源数据集上进行评估，使用合成的40个自然语言查询-答案对进行实验；

**📈 对比分析**

与单一LLM解答比较，Polaris在语义相似度（0.85）、上下文精确度（0.99）和答案相关性（0.90）上均表现优异，100%的样本超越语义相似度与上下文精确度阈值，92.5%超越答案相关性阈值；

**⚠️ 局限性**

局限性包括：部分答案相关性略低，归因于LLM生成的风格差异；DTC的效用模型仍为手工设计，缺乏自学习能力；缺乏全球数据目录支持，可能限制跨组织迁移与泛化。

---

## 301. Could Model Partitioning Make Federated Learning More Sustainable?

**arXiv ID:** 2608.14242 | [PDF](https://arxiv.org/pdf/2608.14242v1)

**作者:** Tobias Frohlich `[一作]` (University of Glasgow), Lauritz Thamsen `[通讯]` (University of Glasgow)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探究在联邦学习中使用模型分区将能源消耗从低碳能源有限的客户端迁移到服务器，以提升可持续性。

**💡 创新点**

首次将模型分区与碳强度、需求响应及现场可再生能源信号结合，实现基于能源信号的动态分区。

**🔧 技术方法**

自定义模型分区框架，采用ResNet和VGG网络，在客户端（NVIDIA L40S）与服务器（NVIDIA H100 NVL）间进行分区训练，使用nvidia‑smi监测功耗。

**📊 数据集**

CIFAR‑10 图像分类数据集。

**📈 对比分析**

与未分区基线（全部在客户端训练）对比，比较能源消耗、训练时间与测试准确率；在ResNet‑152上某些分区点可节省至76%的客户端能源，整体能耗不变甚至下降，准确率保持或略有提升。

**⚠️ 局限性**

仅在单一客户端‑服务器实验，通信开销对早期分区点影响大；模型和硬件范围有限，需进一步验证多客户端、多模型、不同网络与异构硬件下的效果。

---

## 302. AppleScab-LT: A Longitudinal Real-Field Apple Scab Dataset for Temporal Disease Progression Analysis

**arXiv ID:** 2608.14235 | [PDF](https://arxiv.org/pdf/2608.14235v1)

**作者:** Aamir Hilal `[一作]` (National Institute of Technology), Neeraj Goel `[通讯]` (Indian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了AppleScab-LT，一个基于真实果园环境、记录单片苹果叶病害随时间演变的纵向数据集；并对该数据集进行了多阶段的注释、专家验证和质量保障。

**💡 创新点**

①首次提供高分辨率、多时点、单叶病害的纵向数据；②设计了面向时间序列的注释和评估框架；③提出了多维度可靠性验证体系；④提供了像素级、颜色级和归一化相对病害严重度的定量指标。

**🔧 技术方法**

采用了专业相机（Canon EOS R50）进行现场拍摄，使用LabelMe进行多边形注释，并通过Python脚本实现自动化文件、JSON、掩码和时间序列的完整性检查；同时运用专家人工复核与一致性验证。

**📊 数据集**

利用苹果斑叶病（Venturia inaequalis）在两处Kashmir果园收集的多时点图像，共 2,101 张图片，构成 21 条叶片序列，生成 264 条均衡的时间样本。

**📈 对比分析**

对比了现有静态果园数据集与本数据集在时间连续性、病害严重度量和注释质量等维度的差距；实验表明，本数据集能支持时间序列病害预测、严重度估计和阶段分类任务，并在基准实验中显著提升了模型对疾病演变的表征能力。

**⚠️ 局限性**

限制包括：①时间间隔不规律，难以统一建模；②仅涵盖苹果斑叶病，缺乏多病种纵向对比；③样本量相对有限，未来需扩大序列数量和时间跨度；④专业拍摄设备和人工标注成本高，限制了数据规模扩大。

---

## 303. AutoSchema: Live Schema Grounding for Agentic Text-to-Sparql over Heterogeneous Knowledge Graphs

**arXiv ID:** 2608.14228 | [PDF](https://arxiv.org/pdf/2608.14228v1)

**作者:** Yiming Zhang `[一作]` (University of Tokyo), Koji Tsuda `[通讯]` (National Institute for Materials Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“live schema grounding”方法，构建autoschema框架，允许代理在问答时实时从SPARQL端点获取schema证据，避免预先手工编写MIE文件；

**💡 创新点**

创新点在于将schema探索拆分为四个轻量工具（源摘要、实体解析、路径发现、跨库桥接），在问题出现时按需检索并缓存，仅使用当前端点信息；

**🔧 技术方法**

使用的技术包括基于MCP的工具服务、SPARQL查询与缓存索引、LLM代理循环（无训练）、以及开放权重模型（gpt-oss-120b、gemma4-31b）进行文本到SPARQL生成；

**📊 数据集**

评测数据集包括RDF Portal的基因、蛋白、疾病等知识图谱、BioASQ Task B的六年测试集，以及未曾见过的tmQM-RDF化学知识图谱；

**📈 对比分析**

与TogoMCP（预生成MIE）及无schema下界比较，autoschema在资源聚焦和多资源生物KGQA任务中平均事实准确率提升约10-15%，在BioASQ长期评测中保持正向收益，工具调用次数下降，交互成本相近；

**⚠️ 局限性**

局限性：跨库桥接功能使用率低，转移实验仅单次跑，未覆盖所有类型的图谱，尚未实现完整的自动联邦查询，且在某些多资源任务中收益不稳定。

---

## 304. FreeBalance: Pre-Routing Online Moe Load Balancing via Residual Workload Prediction

**arXiv ID:** 2608.14205 | [PDF](https://arxiv.org/pdf/2608.14205v1)

**作者:** Pengfei Chen `[一作]` (Chinese Academy of Sciences), Ling Li `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了FreeBalance框架，通过在目标层路由之前使用残差隐藏层的隐藏表示预测专家负载，从而实现在线负载均衡，并将专家迁移与前置计算阶段重叠；

**💡 创新点**

创新点在于利用残差连接的跨层相似性预先预测专家需求，消除路由与迁移的顺序依赖，使用预算化的pairwise交换规划在可重叠窗口内完成迁移；

**🔧 技术方法**

采用残差工作量预测（调用冻结的路由器两次）、基于成本模型的预算化专家交换规划、对称的专家权重迁移以及保持原始路由决策的无损MoE计算；

**📊 数据集**

在两大MoE模型Qwen3-30B和Moonlight-16B上，使用LongBench的19个长文本子集以及一个混合任务工作负载进行评估；

**📈 对比分析**

与固定专家放置（Vanilla）和历史驱动的EPLB相比，FreeBalance将最大/平均排名负载比降低至1.35（最多32.8%改善），并将预填充延迟降低13.1%（约1.13×速度提升）；

**⚠️ 局限性**

局限在于预测误差仍存在，部分层未能完全改善负载；迁移规划依赖简单的pairwise交换，且在极端任务切换或更复杂网络结构下可能需要更精细的成本模型。

---

## 305. Physics-Bounded mmWave Sensing for Schedulable, Privacy-Preserving Human Pose Estimation

**arXiv ID:** 2608.14176 | [PDF](https://arxiv.org/pdf/2608.14176v1)

**作者:** Shuntian Zheng `[一作]`, Yu Guan `[通讯]` (University Of Warwick)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 PRISM 框架，利用毫米波雷达实现可调度、隐私保护的人体姿态估计；

**💡 创新点**

创新点在于提出物理边界积分处理、物理自适应实例提议与基于截止时间的操作配置，确保执行时间可预测、工作量线性增长并实现实时精度调节；

**🔧 技术方法**

采用毫米波 RAD 张量、常数时间积分查询、Doppler 门控、连通组件、前缀和、MLP 回归以及离线 WCET 估算等技术；

**📊 数据集**

在四个公开数据集（HuPR、XRF55、RT-Pose、mmRadPose）上进行评估，涵盖单人与多人人员场景；

**📈 对比分析**

在单线程 Raspberry Pi 5 上与六个基线进行对比，PRISM 在 45 ms 硬截止时间下实现 0% 延迟超时、99th 分位数延迟降低 24–58% 以及 MPJPE 约 55 mm，且在所有满足截止时间的配置中保持最佳准确率；

**⚠️ 局限性**

局限性包括需针对不同平台离线校准、仅提供 5% 概率的 WCET 保证、对近静止或近距离并置目标的检测率略低，以及假设 Doppler 阈值能跨场景迁移。

---

## 306. LightTeaNet: A Weakly Supervised Lightweight CNN for Multi-Label Tea Leaf Disease Detection and Localization

**arXiv ID:** 2608.14178 | [PDF](https://arxiv.org/pdf/2608.14178v1)

**作者:** Naif Haider Chowdhury `[一作]` (Leading University), Prithwiraj Bhattacharjee `[通讯]` (Leading University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一种轻量级弱监督CNN LightTeaNet，用于多标签茶叶病害检测与定位。

**💡 创新点**

创新点在于将深度可分离卷积、通道注意力与CAM结合，实现无边框注释的自动定位与高效分类。

**🔧 技术方法**

使用了深度可分离卷积、通道注意力、Class Activation Mapping、Grad‑CAM以及混合精度训练等技术。

**📊 数据集**

基于TeaLeafBD数据集，共7类病害（Brown Blight, Gray Blight, Green Mirid Bug, Red Spider, Helopeltis, Tea Algal Leaf Spot, Healthy）。

**📈 对比分析**

与YOLOv8/11/12、RF‑DETR、RetinaNet、Faster R‑CNN等模型对比，LightTeaNet在分类上精度0.9615、召回0.8772、F1 0.9179，定位mAP@0.5为0.181，显示出与全监督模型相近甚至更优的表现。

**⚠️ 局限性**

局限在于CAM定位粗糙，受光照、叶片姿态与背景干扰影响，且对极细粒度病斑分割仍有提升空间。

---

## 307. Seeing Red, Thinking Bad: Color Bias in Vision Language Models

**arXiv ID:** 2608.14286 | [PDF](https://arxiv.org/pdf/2608.14286v1)

**作者:** Kohsuke Ide `[一作]` (National Institute of Advanced Industrial Science and Technology), Yutaka Satoh `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉语言模型对渲染文本中颜色和对比度的敏感性，并提出了Stealth Visual Prompts方法。

**💡 创新点**

创新点在于将颜色和对比度作为隐式控制信号，系统评估其对情感分析和VQA结果的偏移，并用CLIP进行内部表示诊断。

**🔧 技术方法**

使用CLIP作为诊断编码器、四个开源VLM（LLaVA-Mistral、LLaVA-Vicuna、Qwen2-VL、IDEFICS2）以及自定义OCR代理进行评测。

**📊 数据集**

使用自构造的短句/长句情感集、SQuAD问答对和渲染生成的文本图像作为实验数据集。

**📈 对比分析**

通过与全黑基线对比计算情感偏移和VQA错误率，结果显示Qwen2-VL对颜色偏移最敏感，IDEFICS2在低对比度条件下诱发错误率最高。

**⚠️ 局限性**

局限在于仅测试英语文本、单一字体与渲染参数、OCR代理仅覆盖单词级别，未涵盖多语言和更复杂布局。

---

## 308. MAGneT-3D: Monocular and Domain-Generalizable Temporal 3D Detection

**arXiv ID:** 2608.14282 | [PDF](https://arxiv.org/pdf/2608.14282v1)

**作者:** Mohamed Kotb `[一作]` (TU Munich), Daniel Cremers `[通讯]` (TU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MAGneT-3D，一种专为单目视频设计的时空 3D 检测框架，能够在未见域中实现强健的跨域泛化。

**💡 创新点**

创新点包括：
• Domain‑Robust Anchor Generator（DRAG）动态生成与图像特征匹配的 3D 锚点，消除静态查询带来的源域空间偏差；
• Temporal Refinement and Identity Merging（TRIM）通过软分配实现密集监督，并利用对比学习+聚类在推理时去重，提高精度；
• 构建了涵盖 nuScenes、Waymo、Lyft、ONCE 四个真实数据集的跨域基准，系统评估单目时间 3D 检测的零样本泛化能力。

**🔧 技术方法**

使用技术包括：
• 基于 StreamPETR 的 transformer 查询解码器；
• FCOS3D 头预测虚拟深度并实现尺度不变特征分配；
• 软分配（soft assignment）和密集监督；
• 对比学习（Supervised Contrastive Loss）与聚类式去重；
• 虚拟深度、尺度不变分配等域泛化技巧。

**📊 数据集**

所使用的数据集：nuScenes、Waymo、Lyft、ONCE；在训练时仅使用单一源域，验证时跨域评估所有四个数据集。

**📈 对比分析**

与四类主流基线（StreamPETR、Sparse4D v3、BEVFormer v2、Far3D）进行对比。MAGneT‑3D 在跨域 NDS 平均提升至 18.6%（相较于最佳基线 12.1% 的提升 56%），在源域 nuScenes 上 NDS 达到 32.9%（提升 4.4%）。

**⚠️ 局限性**

局限性：
• 仍依赖单摄像头深度推断，受光照、遮挡等因素影响；
• 目前仅处理有限类别（vehicle、pedestrian、bicycle），未探索开放词汇场景；
• 对极端相机配置（极低分辨率或极端焦距）可能仍需进一步鲁棒性改进；
• 软分配与聚类方法在极端多目标场景下的计算开销需要优化。

---

## 309. Solving QBF by Clause Selection

**arXiv ID:** 2608.14274 | [PDF](https://arxiv.org/pdf/2608.14274v1)

**作者:** Mikoláš Janota `[一作]` (INESC-ID), Joao Marques-Silva `[通讯]` (University College Dublin)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于隐式触发集枚举的全局量化布尔公式（QBF）求解算法；

**💡 创新点**

创新点在于将两层量化（∀∃）的隐式触发集枚举方法推广到任意层级的QBF，并将其与CEGAR及冲突/解学习技术相结合；

**🔧 技术方法**

采用SAT求解器作为黑盒子，利用选择变量和学习冲突子句实现多层量化的迭代求解；

**📊 数据集**

使用QBFLIB 2014 Benchmark和2QBF 2010 Benchmark进行实验评估；

**📈 对比分析**

与RAReQS、AReQS、DepQBF、GhostQ等前沿QBF求解器对比，实验显示在2QBF集上获得最多实例，QBFLIB集表现略逊于GhostQ；

**⚠️ 局限性**

局限在于目前仅支持CNF矩阵，对非CNF表达式效果有限，且在QBFLIB集上相对非CNF求解器表现不佳；

---

## 310. Grounding Without Corrective Control: Truth-Tracking Profiles for Large Language Models

**arXiv ID:** 2608.14252 | [PDF](https://arxiv.org/pdf/2608.14252v1)

**作者:** Brett Reynolds `[一作]` `[通讯]` (Humber Polytechnic), Brett Reynolds (Humber Polytechnic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种“路由‑配置”框架，系统性区分大型语言模型（LLM）在部署时的“继承约束”（training‑derived）与“实时纠错路由”（live correction routes）。通过定义五个可衡量的纠错能力特征（目标可达性、可检测性、检查路由独立性、可归因性、有效吸收），该框架能够描述并预测不同LLM架构在各种任务上的真值追踪性能与可纠错性。

**💡 创新点**

创新点在于：① 把“grounding”与“corrective control”分离，证明前者并不必然提供后者；② 通过路由配置与五项纠错特征提供可操作的、可量化的模型架构诊断工具；③ 提出基于任务-路由交互的实验设计，用以检验不同纠错路由（检索、工具、Multimodal、RLHF 等）对性能的选择性提升；④ 强调“继承 vs 实时”二分，在文本‑仅和增强‑模式之间给出可验证的方向性预测。

**🔧 技术方法**

技术上主要采用：
- 结构化的因果/网络分析（参考 Khalidi 2013 的因果网络框架）
- 形式化的五维纠错特征定义与判定方法
- 对比实验设计（如检索 vs 量测的双因素设计，检查器独立性比较）
- 讨论与其他哲学/认知模型的对比（可靠主义、连贯主义、实践主义）

**📊 数据集**

本文并未在现成公开数据集上进行实验，而是给出了若干理论与实验设计的示例：
- 交叉临床设计（检索 vs 量测任务）
- 检查器独立性对比（同一数据集、不同训练）
- 时间序列检验（实时答复的持续可纠错性）。

**📈 对比分析**

比较方法：先预先指定各路由在给定任务中的作用与条件，然后在 held‑out 组合（如检索+工具 vs 单一路由）上进行实验。预期结果为：
- 任务高度依赖检索时，检索增强方案显著提升；
- 需要实时测量的任务时，工具或 Multimodal 方案显著提升；
- 纯文本方案在“密集、稳定文本”场景下仍能保持一定性能。性能评估以“真值追踪成功率”与“纠错率”作为指标，具体数值取决于后续实验实现。

**⚠️ 局限性**

局限性：
- 需要在实验前对每种路由的功能与交互做精确的预设，否则结果可能被后续重写；
- 与传统“信息覆盖”模型相比，难以完全排除后者能同样解释实验结果的可能性；
- 依赖复杂的因果图和架构假设，实际模型部署时可能难以完全测得所有五项特征；
- 目前仅提供理论与实验蓝图，尚未在公开数据集上得到实证验证。

---

## 311. AT-ADD: All-Type Audio Deepfake Detection Challenge Summary

**arXiv ID:** 2608.14249 | [PDF](https://arxiv.org/pdf/2608.14249v1)

**作者:** Yuankun Xie `[一作]` (Communication University of China & Ant Group), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了 ACM Multimedia 2026 年 AT‑ADD 全类型音频深度伪造检测挑战赛的任务设置、数据集设计、评测方法与参赛系统的技术与表现。

**💡 创新点**

创新点在于提出两条轨道（稳健语音检测与全类型检测），并在有限标注条件下通过大规模自监督表示、丰富的数据增强、多裁剪推理以及类型路由与融合等技术，实现了对未知生成器和多模态音频的高精度泛化。

**🔧 技术方法**

主要技术包括预训练自监督音频模型（如 Wav2Vec2‑XLSR、W2V‑BERT、BEATs、EAT‑Large）、多尺度噪声/混响/编解码/重播/速度/音调等信号级增强、滑动窗口多裁剪推理、结构化融合或路由、以及对抗性训练与多任务学习。

**📊 数据集**

使用了 AT‑ADD 设计的公开数据集，包含 Track1（语音）和 Track2（语音、环境声、歌唱、音乐）四类音频的训练/开发/进度/评估分区，数据覆盖多语言、多设备、多生成器以及多类真实与合成样本。

**📈 对比分析**

在闭合设置下对参赛系统进行排行榜比较，Track1 最高宏 F1 达到 90.71%，Track2 最高宏 F1 为 96.10%，榜单前五系统普遍采用自监督表示与融合策略，显示自监督与增强是提升性能的关键。

**⚠️ 局限性**

限制在于对未见生成器的泛化仍不足、在真实声学与通道扰动下的鲁棒性有限、不同音频类型间性能差异显著，并且缺乏对超参数、可复现性细节的详细报告。

---

## 312. Vibration Suppression in Collaborative Flexible Payload Manipulation Using Passive Force Control

**arXiv ID:** 2608.14244 | [PDF](https://arxiv.org/pdf/2608.14244v1)

**作者:** Alaa Abderrahim `[一作]` (VTT Technical Research Centre of Finland Ltd), Shuai Li `[通讯]` (VTT Technical Research Centre of Finland Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文研究了两台异构工业机器人协作操纵柔性重物的控制方法，提出利用主从式领导者–跟随者架构，结合阻抗估计器、输入整形器和阻抗控制来抑制振动，验证了在仿真和实验中的有效性。

**💡 创新点**

创新点在于：1) 采用无振动反馈的主从协作策略；2) 将阻抗估计器与阻抗控制结合，实现跟随者对主导者力的平滑响应；3) 同时使用输入整形器进一步抑制共振；4) 在异构机器人系统（重负载与轻负载）中实现对大尺寸柔性物体的操纵。

**🔧 技术方法**

技术方法包括：阻抗估计器、阻抗控制（仅质量-阻尼模型）、输入整形（基于自然频率的两阶整形器）、能量守恒与被动性分析、Routh-Hurwitz 稳定性判据、Matlab/Simulink + Simscape 仿真、实际使用 KUKA Quantek 与 iiwa 机器人以及 NDI Vega XT 位置传感器。

**📊 数据集**

实验数据集为：10 cm 运动 0.5 s 的柔性丙烯酸板振动记录（使用六个标记测量），采样率最高 60 Hz；仿真使用相同参数的柔性梁模型和机器人动态模型。

**📈 对比分析**

对比方法：无控制、单独阻抗控制、单独输入整形、阻抗+输入整形。实验结果显示：阻抗+输入整形将峰值振幅降至约 79 % 并将稳定时间缩至约 1.2 s，优于单独使用任一方法；单独阻抗控制峰值振幅降低约 57 %，单独输入整形仅 25 %。

**⚠️ 局限性**

局限性包括：仅在二维平面上验证，未覆盖三维运动；未使用真实尺寸的核聚变装置柔性装配板；传感器噪声、控制延迟及机器人动态限制导致实验与仿真结果存在差异；未对障碍物和更复杂约束进行测试。

---

## 313. Training Fair Tabular Foundation Models

**arXiv ID:** 2608.14211 | [PDF](https://arxiv.org/pdf/2608.14211v1)

**作者:** Patrik Kenfack `[一作]` (ETS Montréal), Ulrich Aïvodji `[通讯]` (ETS Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在预训练阶段融入群组公平性的Tabular Foundation Model，利用单前向推理即可实现公平预测。

**💡 创新点**

创新点在于：1）通过自监督方式合成公平任务，无需真实敏感标签；2）在Transformer中引入梯度反转层进行对抗学习，使模型学习对敏感属性不敏感的表示；3）通过可调的公平权重λ，在单模型中实现多点公平-准确性权衡。

**🔧 技术方法**

主要技术包括：stick‑breaking 离散化生成公平任务；梯度反转层 (GRL) 对抗学习；轻量级 nanoTabPFN Transformer 架构；在预训练中联合交叉熵损失，并采用单前向推理的ICL框架。

**📊 数据集**

使用的数据集为：120 个基于 2018 年 ACS PUMS 的公平任务（不同预测目标、敏感属性和州）以及 12 个来自六个公开基准的额外任务。

**📈 对比分析**

与传统 ML 基线（LR、RF、XGB、KNN）、最新的 TFMs（TabPFNv2.5、TabICLv2）以及 FairPFN 进行比较。实验表明，模型在保持或略低的准确率下，DP、EOD、EOP 等公平指标提升 32–75%，且在 AUCROC 上仅轻微下降，Pareto 前沿显示可通过 λ 选择不同的公平‑准确性平衡点。

**⚠️ 局限性**

限制与未来工作：预训练仅基于 TabICL 的先验，可能无法涵盖所有真实敏感属性；在某些严格公平任务（如年龄）上不如专门的公平方法；模型规模较小，尚未在更大 TFMs 上验证；公平任务采样与实际部署的匹配度还有待提升。

---

## 314. Adaptive Protection for Evolutionary Feature Construction in Symbolic Regression with Application to Credit Classification

**arXiv ID:** 2608.14209 | [PDF](https://arxiv.org/pdf/2608.14209v1)

**作者:** Hengzhe Zhang `[一作]` (Victoria University of Wellington), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于特征重要性的自适应保护机制，用于演化特征构造的遗传算法。

**💡 创新点**

创新点在于将特征重要性作为权重动态保护构造特征，实现了对遗传操作的选择性恢复，兼容任意遗传算子。

**🔧 技术方法**

采用遗传编程、多树GP、岭回归/随机树/逻辑回归、SHAP、Marginal Contribution等技术。

**📊 数据集**

使用98个Penn Machine Learning Benchmarks回归数据集以及澳洲信用和德国信用分类数据集进行实验。

**📈 对比分析**

通过与无保护、低交叉/变异等基线比较，实验表明自适应保护显著提升R²/AUC，优于仅降低变异率。

**⚠️ 局限性**

局限在于保护仅在特征层面，缺乏对子结构的动态识别与保护，且对部分任务的函数集可能不够适配。

---

## 315. MMUSV-Sim: A Perception-Oriented Simulation and Data-Generation Platform for Multi-USV Cooperative Perception

**arXiv ID:** 2608.14207 | [PDF](https://arxiv.org/pdf/2608.14207v1)

**作者:** Ziao Li `[一作]` (Sun Yat-sen University), Chenqiang Gao `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了MMUSV‑Sim——一个基于Unreal Engine 5与Project AirSim的多无人水面艇（USV）协同感知仿真与数据生成平台，支持岛屿、海域与港口三类环境、可控波浪诱导的船舶姿态扰动、多模态传感器（RGB、深度、语义、LiDAR、雷达）采集以及统一的世界状态标注与结构化导出；

**💡 创新点**

核心创新点包括：①面向感知的海洋仿真平台，首次将可控波浪扰动与多模态传感集成到USV轨迹模型中；②实现了可批量生成满足几何与感知约束的多USV场景；③引入了场景暂停式一致性采集与全局世界状态同步标注，保证多视角数据与标注一致；④提供了统一结构化导出（兼容OPV2V）和后期注释投影工具；

**🔧 技术方法**

技术实现基于Unreal Engine 5与Project AirSim的自定义插件，采用样条轨迹+合成heave/roll/pitch扰动模型；多模态传感器模拟（RGB、深度、语义、LiDAR、雷达）与传感器外参矩阵；场景暂停采集、世界状态查询与注释投影；Python/Matlab离线后处理生成2D/3D/BEV注释及结构化记录；

**📊 数据集**

使用MMUSV‑Sim自行生成的数据集：41序列、3,129帧，涵盖岛屿、海域与港口环境，包含30+尺寸不同的船舶资产；未使用公开海洋数据集；

**📈 对比分析**

通过比较单机LiDAR、Late Fusion（独立预测后融合）与Early Fusion（点云先合并后检测）三种方法在该数据集上进行BEV船舶检测，评估指标为AP@IoU阈值0.5。结果显示：单机基线AP=45.54，Late Fusion提升至55.47，Early Fusion最高达72.74；在固定延迟（200–600 ms）和位置姿态噪声（0.5/0.5°、1.0/1.0°、2.0/2.0°）条件下，Early Fusion相较于单机保持更高的鲁棒性，延迟600 ms时AP仅降至58.06。

**⚠️ 局限性**

局限性：①仿真中波浪对船舶姿态的物理真实性有限；②未考虑传感器噪声、遮挡误差等真实世界因素；③实验仅针对LiDAR BEV检测，缺少对RGB/雷达等模态的协同评估；④数据集为合成环境，可能难以完全映射至真实海况。

---

## 316. A Near-Optimal Lower Bound for $\ell_p$-Subspace Embeddings, $1\leq p<2$

**arXiv ID:** 2608.14201 | [PDF](https://arxiv.org/pdf/2608.14201v1)

**作者:** Yi Li `[一作]` `[通讯]` (Nanyang Technological University), Yi Li (Nanyang Technological University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对ℓ_p子空间嵌入问题，给出了新的下界：在 1≤p<2 的情况下证明最小嵌入维数 N_p(d,ϵ) 必须满足 N_p(d,ϵ) ≳ d/ϵ² polylog(d/ϵ)。

**💡 创新点**

创新点在于构造了更强的硬实例，利用符号矩阵（sign matrices）以及谱截断与插值技术，将原先的 1/ϵ² 下界提升到包含 d 的量级，并且在 1≤p<2 时已与已知上界在对数因子内匹配，几乎完成了该问题的最优分析。

**🔧 技术方法**

核心技术包括：① 对 2^k×2^k 结构矩阵 M 的谱分解与截断；② 用符号矩阵 S_i 替代标量符号以携带更多信息；③ 通过函数 F_S(v,τ,w) 与其 τ‑导数来重构 S_i；④ 利用非可交换 Khintchine 及 Chebyshev 插值求导逼近；⑤ 通过信息论的位数下界与维数下界的转换。

**📊 数据集**

论文完全基于理论构造，无使用真实数据集；所有硬实例均为合成矩阵。

**📈 对比分析**

与已有的上界（如 N_p(d,ϵ) ≤ C/ϵ² d^{max{1,p/2}} log^c(d/ϵ)）相比，新下界在 1≤p<2 时与之相匹配，仅差对数因子；因此表明在该参数范围内已基本达到最优。

**⚠️ 局限性**

限制：① 结果仅适用于 p∉2ℤ 且 d≳_p log(1/ϵ)；② 对于 p>2 的情况仍未能给出与上界匹配的下界；③ 证明中使用的常数与对数因子可能较大，实际数值性能未给出。

---

## 317. MINT: A Universal Zero-Shot Predictor for Transaction Data

**arXiv ID:** 2608.14198 | [PDF](https://arxiv.org/pdf/2608.14198v1)

**作者:** Parameswaran Kamalaruban `[一作]` (Visa Inc.), Stuart Burrell `[通讯]` (Visa Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 MINT 框架，通过将预训练的交易序列编码器与 decoder‑only LLM 结合，实现了零样本的交易数据预测和问答。

**💡 创新点**

创新点包括：①将交易序列编码器冻结，使用轻量级投射器将交易嵌入注入 LLM；②采用事务‑语言对齐和指令微调；③在不序列化文本的情况下，仅使用压缩嵌入进行多模态推理。

**🔧 技术方法**

技术主要有：Transformer 交易序列编码器、MLP 投射器、LoRA 参数高效适配、事务‑语言对齐训练、指令微调、Chain‑of‑Thought 监督。

**📊 数据集**

使用 Visa 真实支付网络的匿名交易数据集（约 134k 训练/14k 验证/19k 测试客户，包含数十亿笔交易），并通过程序模板和教师 LLM 生成事务描述、问答和 CoT。

**📈 对比分析**

与基于文本序列化的 LLM SFT、基线分类器和对比对齐模型相比，MINT 在预测 QA（ID/OOD）上取得最高准确率，并在输入 token 数量、推理延迟和显存占用上显著优于其他方法。

**⚠️ 局限性**

主要局限：数据为专有且不可公开，隐私与公平性需谨慎；在通用语言基准上因 LoRA 适配导致性能下降；模型仍受限于事务编码器的预训练规模与 LLM 的参数大小。

---

## 318. Positioning with Flexible Reflectors: Solution and Performance Analysis

**arXiv ID:** 2608.14184 | [PDF](https://arxiv.org/pdf/2608.14184v1)

**作者:** Jiajun He `[一作]` (Queen's University Belfast), Hien Quoc Ngo `[通讯]` (Queen's University Belfast)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对 LoS 路径被阻塞的环境，本文提出利用可调节的柔性反射器（FR）进行目标定位，并给出了低复杂度的迭代定位算法以及相应的 CRLB 分析。

**💡 创新点**

创新点在于：① 建立角度相关的 FR 反射模型，将定位问题转化为约束加权最小二乘（CWLS）问题并给出闭式迭代求解；② 推导出闭式 CRLB 并在此基础上得到近似最优的 FR 布置与朝向公式；③ 分析随机 FR 布置下 CRLB 的分布，为网络部署提供统计学指导。

**🔧 技术方法**

采用了角度相关反射模型、泰勒展开线性化、迭代闭式求解、CRLB 理论、遗传算法优化、随机点过程模拟以及 Monte Carlo 仿真等技术。

**📊 数据集**

数据来源为仿真生成：随机布置的 FR、目标与接收机坐标、以及噪声统计参数；未使用公开真实数据集。

**📈 对比分析**

通过与 CRLB、MLE（真值初始化与随机初始化）以及不同 FR 数量的比较，实验表明：① 当 RSS 噪声 ≤0.1 时，所提算法几乎达到 CRLB；② 优化布置后，定位精度比随机布置提升约 15 dB；③ 随着 FR 数量增多，定位精度提升但增益逐渐饱和。

**⚠️ 局限性**

局限性包括：仅在二维单接收机场景下验证；模型假设 FR 角度相关反射系数已知且理想；未考虑硬件误差、动态目标或更复杂的多径干扰；闭式近似在极端条件下可能失效。

---

## 319. Intern-S2-Mobius: Foundation Model with Decoupled Knowledge and Reasoning

**arXiv ID:** 2608.14290 | [PDF](https://arxiv.org/pdf/2608.14290v1)

**作者:** Kai Chen `[一作]` (Shanghai AI Laboratory), Xinyu Zhou `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的语言模型架构——Mobius，旨在通过解耦知识存储与推理过程来提升推理效率与模型表现；

**💡 创新点**

创新点在于（1）实现了向后残差连接，使深层能直接访问信息；（2）采用动态潜在推理（latent reasoning）在较少层次内迭代优化连续向量，减少冗长的链式推理；（3）将知识向量集中为全局共享数据库，打破传统Transformer中知识与推理绑定的限制；

**🔧 技术方法**

技术主要包括：知识向量数据库的水平拼接与MoE式稀疏激活；自回归潜在迭代；与传统自注意力和前馈网络的拆分；对Transformer的自注意力、残差连接与循环深度做改造；

**📊 数据集**

使用了1TB规模的多模态预训练语料进行从零开始训练（7B）和对已有Qwen3.5-35B进行连续预训练，评估数据集包括MMLU、GPQA、IMO Bench、AIME、HMMT、UGD、AMO、SimpleQA、HLE、Biology‑Instructions、Mol‑Instructions、MolecularIQ等；

**📈 对比分析**

与Transformer在相同参数量下对比：Mobius在MMLU和科学任务上均优于Transformer（如MMLU平均分从65.05提升至67.88，科学任务平均分从18.20提升至52.14）；在持续预训练场景下，Mobius实现约4倍的端到端推理速度提升，同时保持或提升推理准确率；此外，Mobius在同等数据量下的MMLU得分比Transformer快1.6倍；

**⚠️ 局限性**

局限性包括：潜在推理与向后残差机制的理论解释尚不完全；大规模共享知识库导致前向激活稀疏化，增加了内存访问压力；模型训练与推理对硬件资源（特别是显存）仍有较高要求；尚需进一步验证在更大规模模型、不同任务及自我进化场景下的可扩展性与鲁棒性。

---

## 320. SimpleOPD: Simple Tokenizer-Agnostic On-Policy Distillation for Long-Context Reasoning

**arXiv ID:** 2608.14277 | [PDF](https://arxiv.org/pdf/2608.14277v1)

**作者:** Haonan He `[一作]` (Shanghai Artificial Intelligence Laboratory), Yu Cheng `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在共享文本空间进行对齐，使用长上下文推理教师模型SU‑01对短上下文学生模型进行 on‑policy distillation（OPD），并引入特殊终止词屏蔽和学生参考 KL 正则化来稳定训练并提升推理能力。

**💡 创新点**

创新点在于：①提出跨词表对齐策略，只对齐在学生与教师 tokenizer 下占据相同文本跨度的 token，从而实现不同模型家族间的 OPD；②通过屏蔽结构终止词的对齐损失和加入学生参考 KL 约束，有效抑制生成长度爆炸和截断；③系统性地展示该方法在多种不同 tokenizer、不同规模模型上的通用性与效果。

**🔧 技术方法**

主要技术包括：on‑policy distillation（OPD）与 PPO 更新；跨 tokenizer 的共享文本对齐；终止词优势屏蔽；学生参考 KL 正则化；使用 SGLang 进行 rollout 与 Slime 框架实现训练。

**📊 数据集**

使用了数学证明相关数据集：Open Proof Corpus（63题）、AoPS（2948题）、Books（900题）、Shuzhimi（617题）等证明任务数据；评估数据包括 ProofBench、AnswerBench、AIME25、HiPhO、HLE、FrontierScience 等。

**📈 对比分析**

与同族模型、跨族模型以及现有 OPD 变体（EOPD、G‑OPD）对比。实验显示 Intern‑S2‑Preview 在 ProofBench 上从 34.0 提升到 55.2，提升 21.2 分，超过 Gemini‑2.5‑Pro；在 Qwen3.5‑35B‑A3B、GLM‑4.7‑Flash、Gemma‑4‑26B‑A4B 等不同家族模型均取得显著提升，表明方法具有良好泛化。

**⚠️ 局限性**

局限性包括：①不同 tokenizer 的对齐仍受限，导致部分 token 无法得到监督；②训练仍需手动调节 KL 正则系数和终止词屏蔽力度；③对更长推理任务仍需更长的 distillation 长度，且计算成本较高；④在加入可验证数学数据后，对 ProofBench 的提升略有下降，表明数据选择对迁移效果敏感。

---

## 321. Designing Mobile and Wearable Sensor-Fused Conversational Agents for Health and Wellbeing

**arXiv ID:** 2608.14273 | [PDF](https://arxiv.org/pdf/2608.14273v1)

**作者:** Hansoo Lee `[一作]` (Imperial College London), Md Haseen Akhtar `[通讯]` (Imperial College London)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在三小时的教程中，作者演示了如何利用 Wearable Sensor‑Dialogue Wellbeing Agent Studio（WSDWAS）将可穿戴设备收集的健康数据与会话式反馈相结合，从监测转向可执行的健康对话。

**💡 创新点**

创新点在于将 Positive Computing 理念与传感器融合的会话式代理结合，并提供可视化仪表盘与自然语言反馈的分工框架。

**🔧 技术方法**

采用的技术包括大语言模型（LLM）驱动的会话代理、可视化仪表盘技术、模拟传感器数据生成与 Prompt 模块化设计。

**📊 数据集**

使用的数据集为在 WSDWAS 平台中生成的模拟可穿戴传感器快照（包含步数、心率、睡眠等指标）。

**📈 对比分析**

方法通过对比仪表盘第一的数值反馈与会话式语义反馈来评估解释负担的减轻，虽然没有定量性能指标，但参与者报告体验更好。

**⚠️ 局限性**

局限性包括缺乏真实设备数据、未进行临床验证、对 LLM 输出的安全性和医疗边界尚未彻底评估。

---

## 322. Zero-Shot Skeleton-Based Action Anticipation

**arXiv ID:** 2608.14243 | [PDF](https://arxiv.org/pdf/2608.14243v1)

**作者:** Hongsong Wang `[一作]` (Southeast University), Qiuxia Lai `[通讯]` (Communication University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出零样本骨架动作预测任务，并给出了结合 GCN、Transformer 与相互信息最大化的基线模型

**💡 创新点**

创新点在于将多模态骨架特征（关节、骨骼、运动）与语义嵌入通过相互信息对齐，从而实现对未见动作类别的泛化

**🔧 技术方法**

使用的技术包括图卷积网络、Transformer 编码器、可学习的空间与时间位置编码、相互信息估计与最大化模块以及关键帧注意力机制

**📊 数据集**

在 NTU RGB+D 60 与 NTU RGB+D 120 两大公开骨架动作数据集上进行实验

**📈 对比分析**

与 SMIE 等基线方法比较，尤其在低观测比例（0.1-0.3）下准确率提升约 4–5%，整体性能优于现有零样本骨架动作预测方法

**⚠️ 局限性**

局限性包括对预训练文本嵌入的依赖，难以处理极端动态或模糊骨架信息，并且在更大未知类别集合上的鲁棒性尚未验证

---

## 323. MathForm: Scaling Mathematical Autoformalization with Knowledge Retrieval and Verification-Guided Refinement

**arXiv ID:** 2608.14221 | [PDF](https://arxiv.org/pdf/2608.14221v1)

**作者:** Lushi Pu `[一作]` (Modelbest Inc), Yudong Wang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合检索、编译与语义反馈的迭代自形式化框架MathForm，并通过该框架构建高质量Lean 4数据集FormalVerse（约 367 k条验证过的自然语言–形式化对），随后在此数据上训练8B模型MathForm-8B。

**💡 创新点**

创新点在于：①使用检索计划器主动提取Mathlib相关定义，减少模型对参数记忆的依赖；②采用编译器诊断与语义一致性判定作为反馈，驱动多轮迭代修正；③在数据构建后重构轨迹，将多轮输出转化为干净、结构化的训练样本；④将构建过程与强化学习奖励直接结合，形成端到端的验证驱动训练。

**🔧 技术方法**

技术包括检索规划（利用gpt‑oss‑120b实现查询Mathlib）、自动化编译验证（Lean 4编译器）、语义一致性判定（gpt‑oss‑120b或QwQ‑32B）、迭代生成与修正循环、轨迹重构、SFT+RL（DAPO）强化学习、以及对抗式数据去污染。

**📊 数据集**

使用的数据集为多来源自然语言数学题集（DeepTheorem、NuminaMath、AceReason-Math等）以及从经典教材抽取的定理和练习；通过MathForm框架生成并验证后得到FormalVerse。

**📈 对比分析**

在六个基准（FormalMATH‑Lite、DeepSeek‑ProverBench、CombiBench、FATE‑M、FATE‑H、FATE‑X）上，以Pass@8（SC与CC）为评测指标，MathForm‑8B在平均SC为88.06%、CC为72.37%上超过了多种32B专用自形式化模型，并在最具挑战性的FATE子集上取得显著优势。

**⚠️ 局限性**

局限性包括：仍对极高抽象程度或极为复杂的定理处理有限；模型生成仍受限于检索范围和编译/语义反馈的精度；数据去污染及评测依赖特定判断器，可能产生偏差；以及在推理速度与资源消耗方面与更大模型相比仍有提升空间。

---

## 324. KV Cache Compression Through the Lens of Transform Coding

**arXiv ID:** 2608.14191 | [PDF](https://arxiv.org/pdf/2608.14191v1)

**作者:** Hannah Laus `[一作]` (Technical University of Darmstadt), Felix Krahmer `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为大语言模型的KV缓存压缩提出了一种基于注意力感知的变换编码方法（AATC），通过对键值对的白化与逆水位分配实现按通道精度分配。

**💡 创新点**

创新点在于推导出注意力感知失真度量，将键值误差分离为可量化的通道/标记因子，并基于此构造了结合变换编码与逆水位的自适应位分配方案。

**🔧 技术方法**

使用了变换编码（白化+SVD）、逆水位分配、注意力权重与输出投影加权的误差度量、以及校准样本统计来决定每通道位数。

**📊 数据集**

校准数据集使用wikitext2、FineWeb、OpenR1-Math-220k的混合样本；评估在Llama和Qwen模型上，覆盖7个英文学科子任务，测试了4k、8k、16k、32k、-m、-cs等长上下文。

**📈 对比分析**

与KIVI、PALU、KVQuant等基线相比，AATC在≈5.8×压缩率下实现了接近FP16的准确率，在所有基准任务上至少不逊于最优基线，且在Qwen长上下文检索和多项选择推理上领先。

**⚠️ 局限性**

局限性包括：假设令牌独立，系统实现缺乏CUDA加速，仅在两类模型上验证，且未结合令牌驱逐或全局缓存场景。

---

## 325. Learning to Forecast Crop Growth from Earth Observation Data

**arXiv ID:** 2608.14281 | [PDF](https://arxiv.org/pdf/2608.14281v1)

**作者:** Dominik Senti `[一作]` (Agroscope), Helge Aasen `[通讯]` (Agroscope)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出利用卫星观测和气象驱动数据，对瑞士冬小麦的叶面积指数（LAI）进行未来 32 天的时间序列预测。

**💡 创新点**

创新点在于引入轻量级的单峰形状正则化器，使模型在稀疏监督下生成符合生态物理学的单峰生长曲线，并通过物理相关性校验正则化强度。

**🔧 技术方法**

使用基于 GRU 和 Transformer 的序列到序列（seq2seq）编码器-解码器架构，并对比 MLP 与 LightGBM 的非时序基线。

**📊 数据集**

数据集为约 20.6 百万像素级的冬小麦 LAI 时序，来源于 Sentinel‑2 10 m 影像与 1 km 瑞士气象数据，覆盖 2021‑2025 5 个生长季。

**📈 对比分析**

实验显示 seq2seq 模型在 32 天预测窗口下平均 RMSE 约 0.51 LAI，R²>0.8，显著优于 MLP（0.59）与 LightGBM（0.81），且在留一年不评估（LOYO）中保持良好泛化。

**⚠️ 局限性**

局限性包括仅针对瑞士冬小麦、使用观测气象而非预测气象、对长时程或多作物、土壤及管理因子未建模、以及 32 天预测范围内的精度递减。

---

## 326. TimeSage-EV: A Live Benchmark for Agentic Time Series Analysis in Evolving Environments

**arXiv ID:** 2608.14270 | [PDF](https://arxiv.org/pdf/2608.14270v1)

**作者:** Qingren Yao `[一作]` (Eindhoven University Of Technology), Joaquin Vanschoren `[通讯]` (Eindhoven University Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个实时的 Agentic 时序分析基准 TimeSage-EV，用于评估大型语言模型在随时间演化的环境中对持续时间序列数据的分析与报告生成。

**💡 创新点**

创新点包括：持续更新、时间截止约束、六大难度层级与1,485个问答对；四轴（关键点准确性、覆盖度、报告真实性、质量）评分体系；以及自演化技能库管理机制。

**🔧 技术方法**

使用的技术包括：大型语言模型与工具调用框架、记忆压缩与代码执行、LLM 判定器、以及自演化技能库管理系统。

**📊 数据集**

使用的数据集由500个公共机构源构成，筛选出60个场景，覆盖农业、气候、能源、金融、医疗、交通等六大领域，累计1,485个场景-周期 QA 对。

**📈 对比分析**

对比方法：在独立与序贯评估下对六大 LLM（Sonnet‑4.6、GPT‑5.4、Kimi‑K2.6、Qwen‑3.5‑397B、Devstral‑2‑123B、Gemma‑4.31B）及自演化代理 TimeSage‑1.0；GPT‑5.4 取得最高分 86，表现随难度下降；Token 成本最高者 Sonnet‑4.6，表现相对较差。

**⚠️ 局限性**

局限性：仅覆盖公开机构的时间序列与文本，缺乏企业数据、图表解析、多语言与多格式多样性；评估可能受 LLm 判断噪声及特定工具框架影响。

---

## 327. On the Robustness of Temporal Vision-Language Models for Surgical Endoscopy Videos

**arXiv ID:** 2608.14262 | [PDF](https://arxiv.org/pdf/2608.14262v1)

**作者:** Darakshan Rashid `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yutong Xie `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在真实临床环境下的端内手术视频中，时序视觉-语言模型的鲁棒性，提出了Endo-C6六类腐败基准和评估协议，并通过少量标注的VeRA微调方法显著提升模型在腐败视频上的性能。

**💡 创新点**

创新点包括：① 设计了 Endo-C6 这套高严重度、易复现的端内真实腐败基准；② 提出了统一的时序提示式评估协议；③ 在冻结视觉编码器的前提下，使用轻量化的 VeRA（向量随机矩阵适配）进行少样本微调，实现鲁棒性的大幅提升。

**🔧 技术方法**

使用了 CLIP 风格的时序视觉编码器（如 SurgVLP、HecVLP、PeskaVLP）与文本编码器，采用 Cosine 相似度和固定模板进行推理；对分类头采用 VeRA 进行少量标注微调，并与传统 LoRA 进行对比。

**📊 数据集**

实验基于公开的 GI 端镜数据集 Kvasir、肝胆切除手术阶段数据集 CholecT50 以及手术动作数据集 TEMSET-24K，并在每个数据集上加入 Endo-C6 的六种高严重度腐败。

**📈 对比分析**

通过对 Clean、六种腐败场景下的 Top‑1 准确率、Mean‑C（平均腐败准确率）和 Worst‑C（最差腐败准确率）进行系统比较。零射击的 TVLM 在 Worst‑C 方面可低至 0.4%，而引入 VeRA 的 RobustEndoCLIP 在 Mean‑C 约提升至 25%~30%，Worst‑C 提升至 10%~20%，显著优于三种基线模型。

**⚠️ 局限性**

局限性在于仅针对提示式分类任务，未扩展到检测、分割等更复杂的端内任务；评估仅覆盖 CLIP 风格的时序 VLM；少样本微调仍需要更广泛的数据验证；未探讨生成式多模态模型在此基准上的表现。

---

## 328. The More Popular, The Harder to Forget: Adaptive Popularity for LLM Unlearning

**arXiv ID:** 2608.14229 | [PDF](https://arxiv.org/pdf/2608.14229v1)

**作者:** Anna Borisiuk `[一作]` (AIRI), Elena Tutubalina `[通讯]` (AIRI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AdaPop方法实现对大型语言模型的目标化遗忘。

**💡 创新点**

创新点在于结合外部流行度信号为每个事实分配指数加权梯度，并用双上升控制器自动平衡遗忘与保留。

**🔧 技术方法**

采用基于外部流行度的指数权重、对数似然的加权梯度、LoRA微调以及双上升控制器。

**📊 数据集**

使用DUET和RWKU两套实体事实问答基准，利用Wikidata sitelink计数或LLM-as-Judge作为流行度代理。

**📈 对比分析**

与GA、GD、NPO、WGA等基线比较，AdaPop在稳定方法中在两大基准上均取得最低的遗忘指标（对抗/同义句），保留性能保持在0.05以内。

**⚠️ 局限性**

局限在于依赖外部流行度代理，且仅在LoRA微调下验证，未覆盖非实体知识和全参数微调。

---

## 329. Attributing Preprocessing Invariance in Spectral Foundation Models

**arXiv ID:** 2608.14227 | [PDF](https://arxiv.org/pdf/2608.14227v1)

**作者:** Dongjun Wei `[一作]` (ESCP Business School), Yinuo Zou `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了光谱基础模型在不同预处理管道下的跨迁移性能，并指出大部分所谓的“预处理不变性”是由模型中的确定性归一化提前实现的，而非学习得到的；

**💡 创新点**

创新点在于提出并验证一种先评估归一化能否去除所评估变换的方法，从而在跨预处理迁移评估中剔除已被归一化消除的变换，澄清模型真正学习到的不变性；

**🔧 技术方法**

技术主要包括：光谱数据归一化（Robust Normal Variate、标准化等）、Transformer encoder预训练、增广对生成对的对抗学习以及基于数值测试的变换消除判定；

**📊 数据集**

使用了六个公开的拉曼光谱分类数据集，涵盖癌细胞系、血液与皮肤、染料、环境化学、血清和唾液等不同实验条件；

**📈 对比分析**

比较方法是构建跨预处理迁移网格，先将模型的编码器去除，只保留归一化步骤作为基线，再与完整模型进行对比；结果显示归一化单独就能获得与完整模型相当的跨管道准确率，完整模型在绝对准确率上仅高出约1.5个百分点；

**⚠️ 局限性**

局限性包括：统计功效有限（仅能检测大于约8个百分点的差异），未评估跨数据集迁移（各数据集标签空间互不重叠），以及对归一化与学习贡献的区分受限于所用的数值测试与理论假设。

---

## 330. A Generalized Parallelogram Rule for Proportional Analogies on Riemannian Manifolds

**arXiv ID:** 2608.14220 | [PDF](https://arxiv.org/pdf/2608.14220v1)

**作者:** Pierre-Alexandre Murena `[一作]` (Hamburg University of Technology), Marcelo Hartmann `[通讯]` (University of Helsinki)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本文中，我们提出了一种基于黎曼几何中测地线中点的本质比例类比关系，并证明其满足比例类比的公理。

**💡 创新点**

创新点在于将比例类比从向量空间推广到任意黎曼流形，利用测地线中点实现完全本质的构造，并给出闭式解在多种重要流形（球面、双曲空间、对称正定矩阵、形状空间、概率分布单形）上的实现。

**🔧 技术方法**

采用了黎曼几何工具（指数映射、对数映射、测地线对称、等距变换）以及对称空间的对称映射，进一步推导出在对称空间中的简单表达式，并实现了数值求解。

**📊 数据集**

在实验方面使用了MovieLens 1M数据集进行偏好转移任务（分类概率分布），以及对SPD矩阵、三角网格形状等的类比求解。

**📈 对比分析**

与留一平均、加性基线以及算术几何类比模型进行对比，实验结果表明在大多数设置下我们的Riemannian类比模型在Jensen‑Shannon散度上均优于基线，Fisher‑Rao几何在多数实验中取得最小误差。

**⚠️ 局限性**

局限性包括：在非测地线凸或非完备流形上可能无解（如Fisher‑Rao单形），对唯一测地线的依赖限制了应用范围，且高维数下指数/对数映射的数值计算仍需改进。

---

## 331. Connected Subspace Clustering: Hardness, a Scalable Heuristic, and an Application to Sea Level Geodesy

**arXiv ID:** 2608.14215 | [PDF](https://arxiv.org/pdf/2608.14215v1)

**作者:** Johanna Hillebrand `[一作]` (Heinrich Heine University Düsseldorf), Bernd Uebbing `[通讯]` (University of Bonn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种同时考虑子空间拟合误差和连通约束的聚类方法——Connected Subspace Clustering，并在全球海平面异常数据上验证其效果。

**💡 创新点**

创新点包括：①在子空间聚类框架中首次引入连通约束并给出NP难度证明；②设计了高效的迭代合并（IterMerge）子程序实现连通性修复；③将该方法与多种初始化策略和连通性维护技术结合，形成可扩展的Lloyd式框架。

**🔧 技术方法**

使用Lloyd式迭代改进、主成分分析（PCA）求子空间、最小距离分配、连通子图合并、邻域图最小堆和并查集等技术；同时比较了Agglo-ST、Conn-Agglo-Euc/ ST、Conn-KMeans++、Conn-Ward 等初始化方法。

**📊 数据集**

主要数据集为 Copernicus Marine Service（CMEMS）全球海平面异常（SLA）数据，经过预处理后得到约8,160个网格点（高分辨率可达532,783点）。

**📈 对比分析**

与基线方法（Conn-Ward、SSC-OMP、EnSC、EGCSC、EKGCSC）对比，使用四种连通性恢复策略（IterMerge、PostMerge、SmoothMerge、IntegratedConn）时，IterMerge在73.75%的配置中获得最低子空间重构误差；相较于基线，Conn-Subspace在保证k个连通区域的同时显著降低误差，性能优于现有连通聚类与子空间聚类方法。

**⚠️ 局限性**

局限性包括对邻域图结构（如孔洞网格）和初始化方式的敏感性；目前方法在同时优化子空间误差和连通性方面未实现全局最优，且在更大规模或不同空间结构的数据上可能需要进一步改进。

---

## 332. APTER: Adaptive Post-Training with Expert-Grounded Rubrics

**arXiv ID:** 2608.14212 | [PDF](https://arxiv.org/pdf/2608.14212v1)

**作者:** Xukai Wang `[一作]` (Ant Group), Xu-Yao Zhang `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 APTER 框架，利用专家基础的 rubrics 对大型语言模型进行后训练，实现细粒度评估与目标优化。

**💡 创新点**

创新点在于将结构化领域知识路由到查询级 rubrics，并通过 Ada‑IFT 将 rubric 判定用于诊断并触发针对性修复，提升模型在专业推理任务中的可解释性与性能。

**🔧 技术方法**

使用了 rubric‑RL、rubric‑based SFT、Ada‑IFT、GRPO 强化学习、LLM‑as‑Judge 以及多角色生成与校准技术。

**📊 数据集**

实验数据集包括数学推理的 DAPO‑Math、AIME‑24/25、MATH500、OlympiadBench，以及医学问答的 LiveMedBench、HealthBench、MedQA、MedMCQA。

**📈 对比分析**

与未改造的 Qwen 系列模型及其他后训练基线（如 SFT、Rubric‑RL、Binary 验证）对比，AP TER 在数学宏平均上提升 14–15 分，在医学宏平均上提升 6–8 分，最高在 AIME‑24 上增幅 25.82 分、HealthBench 上增幅 17.96 分。

**⚠️ 局限性**

局限性包括需先构建专家判定框架、判定结果为二值化、训练成本较高、且对不同领域的迁移能力仍需进一步验证。

---

## 333. Can Language Models Understand mmWave Data? Benchmarking Large Language Models for mmWave Radar-Based Human Understanding

**arXiv ID:** 2608.14179 | [PDF](https://arxiv.org/pdf/2608.14179v1)

**作者:** Jeongwan Shin `[一作]` (DGIST), Jaeho Choi `[通讯]` (DGIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 mmWave-QA 基准，评估大型语言模型（LLM）在毫米波雷达点云上的零/少样本问答推理能力。

**💡 创新点**

创新点在于：① 用最小化的自然语言化接口将毫米波点云转换为可直接喂给 LLM 的文本；② 构建了跨硬件、场景的多模态 QA 数据集；③ 通过多种 prompting（零样本、少样本、链式推理）系统评估 LLM 在毫米波感知中的推理表现。

**🔧 技术方法**

技术主要包括：毫米波点云的坐标量化与文本化；基于 LLM 的问答推理（GPT‑4o、Gemini、GPT‑5 等）；少样本学习、链式推理提示策略；以及对比实验与定量/定性分析。

**📊 数据集**

使用的公开毫米波数据集有：mmBody、MM‑Fi、mRI，并对其进行统一预处理与标签层次化，最终构成 139（精简为 86）动作类别，涵盖 6 种场景、5 类 QA 任务。

**📈 对比分析**

与 RGB 模型（GPT‑4o 处理视频）对比发现：在正常/布置环境下 RGB 更优，但在雨、烟、暗光和遮挡等视觉受损场景下，毫米波 Q&A 的准确率更高或相近；多帧实验显示 16 帧左右最优；少样本+链式推理可将整体准确率提升约 5–10%。

**⚠️ 局限性**

局限性包括：依赖商业 LLM，未验证开源模型效果；仅使用静态、低至中等分辨率雷达点云；对极端多路径/多目标场景的鲁棒性仍待进一步研究；数据集仍偏向实验室环境，缺乏大规模真实世界覆盖。

---

## 334. Structure-Guided Spatiotemporal Attention Graph Neural Network for Traffic Flow Prediction

**arXiv ID:** 2608.14177 | [PDF](https://arxiv.org/pdf/2608.14177v1)

**作者:** Xuanmian He `[一作]` (University of California Berkeley), Wanjing Ma `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结构引导的时空注意力图神经网络（SGSAN），通过先学习静态有向依赖图（DDG）再软耦合动态注意力，以实现可解释且高精度的交通流预测。

**💡 创新点**

创新点包括：① 两阶段训练框架将结构发现与时空注意力分离，避免梯度干扰；② InfoNCE 对比学习作为软耦合机制，使动态注意力在保持灵活性的同时受静态结构引导；③ 通过构造有向无环图（DDG）捕捉交通网络的宏观传播路径，显著降低假相关性。

**🔧 技术方法**

技术手段包括：图注意力网络（GAT）与多头自注意力的时空注意力模块；结构方程模型（SEM）+变分自编码器实现DDG学习；连续 DAG 约束（NOTEARS）保证无环性；InfoNCE 软耦合损失；稀疏正则化；两阶段训练策略。

**📊 数据集**

在加州交通部性能测量系统（PeMS）公开数据集上评估，使用 PeMS03、PeMS07、PeMS08 和 PeMS-Bay 四个不同规模的道路网络。

**📈 对比分析**

与 10 个基线模型（T‑GCN、STGCN、GMAN、ASTGCN、STTN、DST2Former、DCST、CCHMM、PSTCGCN 等）进行对比。SGSAN 在所有数据集和不同预测时长（15/30/60 min）上均实现了 MAE/MAPE/RMSE 的最低值，达到或超过现有最佳模型；训练时间相较于联合训练下降约 30%；推理延迟仅 0.096 ms/样本，优于多数基线且接近最小延迟。

**⚠️ 局限性**

局限性：① 对有向无环结构的假设在某些复杂网络中可能过于约束，导致重要循环关系被忽略；② InfoNCE 软耦合需手动调节温度参数，易受数据分布变化影响；③ 结构发现阶段仍需离线计算，对实时变化的网络拓扑（如道路临时改道）适应性有限。

---

## 335. Concept Guidance: Precise, Training-Free Latent Control for Text-to-Image Generation

**arXiv ID:** 2608.14172 | [PDF](https://arxiv.org/pdf/2608.14172v1)

**作者:** Nikolai Röhrich `[一作]` (LMU Munich), Björn Ommer `[通讯]` (LMU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、无梯度、无外部模型的推理时层跳过指导方法（Concept Guidance, CoG），可对文本生成图像模型进行精确概念级控制。

**💡 创新点**

创新点包括：① 使用每层与概念的互信息衡量层的概念相关性；② 基于概念相关层的加权多层跳过生成负向预测；③ 将该指导与传统分类器无关指导（CFG）无缝集成，提升局部一致性与概念准确性。

**🔧 技术方法**

核心技术：层级互信息分析、层跳过（Skip）机制、性能加权多层预测融合、与CFG的线性外推；实现基于PyTorch的推理时插件。

**📊 数据集**

数据与评测：使用PixArt-α、Stable Diffusion 3/3.5、FLUX.1-dev等公开模型；评估指标包括EasyOCR文本识别分数、手部结构准确度、CLIP美学分数、HPSv3人类偏好、KDD/LPIPS等多维度辅助度量。

**📈 对比分析**

与CFG、APG、PAG等最先进的无训练引导方法对比，CoG_multi平均提升约8.1%（对手部可达42%），在文本、手部与美学三大任务均优于基线；与APG/PAG组合可进一步提升；在多目标场景下仍保持领先。

**⚠️ 局限性**

局限性：需先对每个模型/概念做一次层级性能剖析，耗时且不跨模型迁移；推理时每跳过一层会增加延迟（单层已获得大部分收益）；只优化特定概念，可能导致整体图像多样性与通用质量的轻微折衷。

---

## 336. Integrated Information in the Active Inference Framework

**arXiv ID:** 2608.14165 | [PDF](https://arxiv.org/pdf/2608.14165v1)

**作者:** Carlotta Langer `[一作]` (Max Planck Institute of Molecular Cell Biology and Genetics), Nihat Ay `[通讯]` (Hamburg University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

结合主动推理框架与集成信息理论，构造可计算集成信息的生成模型，训练仿真智能体在racetrack环境中学习避墙，并评估自由能、集成信息、成功率等指标的相关性。

**💡 创新点**

创新点在于：① 将主动推理的内部状态结构化以满足IIT度量的假设；② 通过实证比较IIT 2.0/3.0/4.0以及多信息量与自由能的关系；③ 揭示信息整合与自由能最小化之间的统计关联。

**🔧 技术方法**

使用的技术包括：POMDP基的主动推理与变分自由能；IIT 2.0/3.0/4.0指标计算（含Wasserstein距离、intrinsic difference等）；Python实现仿真、Pearson/Spearman相关性分析；多信息量（Total Correlation）计算。

**📊 数据集**

使用的“数据集”是仿真生成的racetrack轨迹：50,000步步数、二元壁面传感器、双轮运动控制，训练1000个智能体并在每1,000步记录指标。

**📈 对比分析**

比较方法为在每1,000步采样指标并统计Pearson与Spearman相关系数；结果显示Φ_T和多信息量与自由能呈显著负相关，相关性随内部状态数n提升；IIT 3.0/4.0相关性弱但仍存在；成功率与自由能几乎不相关。

**⚠️ 局限性**

局限性包括：① 学习全生成模型容易陷入观测与内部状态独立的局部最优；② IIT 3.0/4.0计算成本高限制模型规模；③ 自由能最小化并不必然导致高成功率，说明两理论关联不完全。

---

## 337. Tripwire: Triggering Aligned Refusal via Statistically Certified Safety Neurons

**arXiv ID:** 2608.14392 | [PDF](https://arxiv.org/pdf/2608.14392v1)

**作者:** Wei Zhao `[一作]` (Singapore Management University), Jun Sun `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型的 jailbreak 攻击，提出了一种无训练、基于神经元级的防御方法

**💡 创新点**

创新点在于：①使用基于假设检验和 Benjamini–Hochberg FDR 控制的统计识别漏斗，精确筛选出安全专用神经元；②采用触发式 clamp 只将这些神经元的激活固定为“有害”条件下的均值，从而触发已学会的拒绝行为；③提供两种等价的部署方式——检测驱动的推理时 clamp 与离线 bias‑patch 权重编辑，兼顾实时性与永久化。

**🔧 技术方法**

技术方法包括：神经元单变量 t 检验与方向/特异性过滤；Benjamini–Hochberg FDR 控制；AUROC 排序与 neuron‑budget 选择；触发式 clamp 以及 bias‑patch 权重编辑；逻辑回归检测器；实验评估采用 GPT‑5.4 安全评判。

**📊 数据集**

使用的数据集：HarmBench（有害提示），Alpaca（正常提示），MMLU（通用任务提示），AdvBench（生成对抗提示）以及 MT‑Bench 用于评估模型实用性。

**📈 对比分析**

与 RepE、TraceRouter、LED、DELMAN 等现有防御进行对比。实验在四个安全对齐开源 LLM（Llama‑2‑7B、Llama‑3.1‑8B、Qwen2.5‑7B、Qwen2.5‑32B）上进行，攻击覆盖 GCG、AmpleGCG、AutoDAN、Jailbreak‑R1。结果显示：在 top‑2500 神经元预算下，ASR 均降至 1%~2%，并且 MT‑Bench 兼容性下降仅为 0.5%~5.3%，显著优于对手（如 TraceRouter 20% 以上 utility 损失）。

**⚠️ 局限性**

局限性：①识别过程依赖离线统计，可能无法捕捉未来演化的攻击模式；②方法仍假设安全行为由稀疏神经元驱动，对大规模或极端攻击的鲁棒性有限；③评估仅覆盖公开的四种 LLM，尚未验证在更大模型或商业部署场景中的可扩展性。

---

## 338. Designing Inclusive Crypto-Asset Dispute Resolution A Hybrid AI and Smart Contract Online Dispute Resolution Framework for Vulnerable Users

**arXiv ID:** 2608.14356 | [PDF](https://arxiv.org/pdf/2608.14356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 339. Can We Defend Against AI-Generated Video Attacks on Real-World Crisis Events? A Systematic Evaluation of Detectors, Generators and Social Dissemination

**arXiv ID:** 2608.14391 | [PDF](https://arxiv.org/pdf/2608.14391v1)

**作者:** Shuo Liang `[一作]`, Wangbo Zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了以真实事件视频为锚点的RA-Bench基准，涵盖10类社会风险场景的1830条真实视频和来自4个开源及5个闭源生成器共16056条AI生成视频；

**💡 创新点**

创新点在于使用真实视频作为对照锚点，系统评估生成器在社会风险情境下的表现，并通过人类判断与社交传播模拟揭示检测器在真实环境中的鲁棒性缺失；

**🔧 技术方法**

使用了传统判别模型、零样本多模态大模型以及专门微调的多模态大型语言模型（MLLM）进行检测；

**📊 数据集**

数据集为RA-Bench，包含1830条真实视频和16056条生成视频，覆盖10大社会风险类别；

**📈 对比分析**

通过AUC、BAcc、FakeR等指标评估三类检测器的泛化性能，结果显示传统检测器从公开基准降至≈45‑57% AUC，零样本模型对提示和源敏感，微调MLLM受时间标签偏差影响，整体检测效果远低于随机；

**⚠️ 局限性**

局限性包括：仅评估图像-视频（I2V）生成，未涵盖音频或完整多模态合成；仅在控制环境下模拟社交传播，未覆盖真实平台的多步骤伪造流程；以及基准随生成器和检测器快速迭代而需持续更新。

---

## 340. Catching the Imposter: Self-Supervised Learning of Physical Coherence with Cross-Entity Feature Permutations

**arXiv ID:** 2608.14372 | [PDF](https://arxiv.org/pdf/2608.14372v1)

**作者:** Aleksei Rozanov `[一作]` (University of Minnesota), Vipin Kumar `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的自监督预训练任务——Imposter（及其多样化版本Multi‑Imposter），通过在不同实体之间交换特征并训练编码器识别哪些特征被替换，从而迫使模型学习跨特征的物理关联；

**💡 创新点**

创新点在于将物理一致性作为自监督信号：利用真实观测值替换特征，所有被替换值本身都合理，但与宿主实体剩余特征不一致，迫使模型捕捉物理耦合；同时提出 Multi‑Imposter 以消除单一 donor 记忆的捷径；

**🔧 技术方法**

技术包括基于 Patch‑Transformer 的时间序列编码器、特征级别的随机替换与二分类损失（BCE）以及与多种基准预训练目标（Contrastive, Next‑Token Prediction, Feat‑MAE, Temp‑MAE, FT‑MAE）进行对比；

**📊 数据集**

使用 ERA5‑Land 再分析数据（21 维环境变量，30 天窗口），并在七个下游任务上评估：Köppen 气候分类（5/30 类）、碳通量（GPP、RECO、NEE）、流量预测（CAMELS）和流域属性重建；

**📈 对比分析**

方法通过单目标、嵌入拼接和联合训练三种方式比较，结果显示没有单一预训练任务在所有任务上均优；Imposter/ Multi‑Imposter 在气候分类与碳通量等需要跨特征物理一致性的任务上表现突出；与基准相比，拼接方式可提升 10–60% 的指标；联合训练提升幅度较小；

**⚠️ 局限性**

限制包括：(1) 物理一致性并不能完全替代实体辨别能力，导致在流域属性等任务上表现不佳；(2) 需要手工挑选合适的 swap 率和多样化策略；(3) 与对比学习相比计算量仍较大；(4) 该方法主要适用于已知物理耦合特征的多变量时间序列，扩展到更复杂或非平衡数据仍需研究。

---

## 341. Designing Sustainable Federated Learning as a Service using Neural Architecture Search

**arXiv ID:** 2608.14359 | [PDF](https://arxiv.org/pdf/2608.14359v1)

**作者:** Keya Patel `[一作]` (Curtin University), Monowar Bhuyan `[通讯]` (Umea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种可持续的 Federated Learning as a Service（SFLaaS）框架，通过神经架构搜索（NAS）在满足消费者碳预算的前提下自动设计模型架构，并实现动态碳感知的消费者调度与迭代搜索；

**💡 创新点**

将消费者级碳可行性评估、需求驱动的搜索空间过滤和多目标演化搜索融入 NAS，首次实现硬碳约束下的架构可行性与性能的联合优化；

**🔧 技术方法**

采用需求驱动的宏观 NAS（绿色超参数）、碳可行性估计模型、动态碳感知调度算法以及基于精英保留的演化搜索策略；

**📊 数据集**

在 CIFAR-10、ImageNet-100、MedMNIST、WISDM 等公开数据集上进行实验；

**📈 对比分析**

与 Accuracy-Driven NAS、Efficiency-Aware NAS、Federated NAS 及 Global Carbon-Aware NAS 对比，SFLaaS-NAS 在保持 85–93% 的精度的同时，参与率提升至 90–96%，碳违规率降至 1–2%；

**⚠️ 局限性**

主要缺点是演化 NAS 过程计算开销大，且仅考虑单一 FLaaS 提供方，未对多供应商协同、资源分配与公平性进行研究。

---

## 342. Beyond Capacity: Scalable MoE LLM Inference via High-Bandwidth Flash with Direct GPU and HBM Paths

**arXiv ID:** 2608.14333 | [PDF](https://arxiv.org/pdf/2608.14333v1)

**作者:** Seeyeon Kim `[一作]` (KAIST), Joo-Young Kim `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将高带宽闪存（HBF）与高带宽内存（HBM）通过双路径（直接GPU–HBF和Relay HBM–HBF–GPU）连接的GPU内存体系结构，并结合早期专家选择和KV写入调度，以实现大规模稀疏专家语言模型的高吞吐量和低延迟推理。

**💡 创新点**

创新点包括：
1) 将HBF提升为主存层而非仅仅是容量扩展；
2) 引入并行的直接路径和Relay路径，实现同一专家数据同时从两条路由流传输；
3) 通过提前完成路由的基准项计算实现专家的“Lookahead”预选，隐藏NAND感应延迟；
4) 针对prefill和decode阶段的KV写入进行分阶段调度，最大限度地利用HBM的细粒度写入和HBF的批量写入，降低写入延迟与磨损。

**🔧 技术方法**

采用UCIe 3.0高速互连、HBM3E与512 GB HBF堆叠、SRAM双缓冲、双路径调度算法、早期路由预计算、KV写入批处理、连续批量推理调度器以及LLM模拟器进行评估。

**📊 数据集**

使用多种大规模稀疏专家模型进行实验，包括Qwen3‑235B‑A22B、DeepSeek‑V2/V3、Llama‑4‑Maverick、Grok‑1等；批量大小、前缀长度和解码长度覆盖了从短预填到长解码的多种工作负载。

**📈 对比分析**

与单路径（RelayOnly/DirectOnly）和压缩版本（Compact-）对比，所提架构在所有测试模型与批量大小下均实现了1.79–1.94×的吞吐量提升，端到端延迟降低约40–48%；在连续批量推理场景下，P90吞吐量提升10–37%，延迟下降35–53%。相较于CPU–GPU拆分的混合方案，性能提升幅度超过10倍。

**⚠️ 局限性**

局限性包括：
1) 依赖尚未公开的HBF规格（读写延迟、并行度、耐久性）；
2) 对HBF闪存耐久性的估计基于理想化的写入放大因子，实际设备可能受坏块、磨损不均影响；
3) 早期专家预选仅适用于无偏置、可缩放的路由器，其他路由器需采用传统后期选取；
4) 在极大KV写入压力下，若写入速率超出HBF可覆盖的窗口，仍会出现写入阻塞。

---

## 343. Program-space Diffusion for Morphology-to-Transcriptomics Prediction

**arXiv ID:** 2608.14330 | [PDF](https://arxiv.org/pdf/2608.14330v1)

**作者:** Ruyter Swann `[一作]` (Sorbonne Université), Racoceanu Daniel `[通讯]` (Sorbonne Université)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

从 H&E 组织切片图像预测空间转录组表达，提出在转录程序空间进行条件扩散模型预测。

**💡 创新点**

创新点在于：①使用 Pearson 残差归一化替代 log 归一化，消除测序深度和技术噪声的影响；②利用共识非负矩阵分解 (cNMF) 获得低维、协同的转录程序作为预测目标；③在程序空间上实施条件扩散 (DDPM)，显著降低输出维度、提升计算效率并保持或提升预测性能。

**🔧 技术方法**

主要技术包括：Pearson 残差归一化、cNMF（多次 NMF + 一致聚类）构建转录程序基、非负最小二乘投影、条件扩散模型（DDPM）在程序空间训练与推断，以及与 STEM gene‑space 模型的对比实验。

**📊 数据集**

使用公开的 HER2+ 乳腺癌空间转录组数据集（36 块切片、8 位患者，第一代 Spatial Transcriptomics），配合对应的 H&E Whole‑Slide Imaging（WSI）图像，采用患者留出交叉验证方案。

**📈 对比分析**

通过与 STEM 的比较（log 归一化和 Pearson 归一化两种协议）进行实验，结果显示在 Pearson 归一化下，程序空间扩散在 Top‑M PCC 上与 STEM gene‑space 相当或略优；在大基因面板（800/2000）下预测质量明显提升；此外，程序空间模型的采样时间与基因数量无关，显著低于 gene‑space 的计算开销。

**⚠️ 局限性**

限制包括：仅在单一 HER2+ 数据集内验证，缺乏跨 Cohort 泛化评估；程序维度 K 固定，未探究最佳 K 的选择；以及未评估预测结果在生物学下游任务中的实际价值，仅使用 PCC 作为性能指标。

---

## 344. AnchorBench: A Multi-Pathway Benchmark for the Anchoring Effect in LLMs

**arXiv ID:** 2608.14320 | [PDF](https://arxiv.org/pdf/2608.14320v1)

**作者:** Yiderigun Borjigin `[一作]` (Saarland University), Roland Aydin `[通讯]` (Saarland University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AnchorBench，一个针对大型语言模型（LLM）锚定效应的统一基准，评估多条锚定路径（External、History、ICL、RAG、Tool）和锚点相关性（无锚、无关、可行），通过结构化的数值判断任务来量化模型的偏移；

**💡 创新点**

①首次将人类锚定理论与LLM上下文传递机制对齐，构建多路径、可控的锚定实验；②引入相关性轴区分无关与可行锚，能区分不合理偏差与合理更新；③系统化评估14个模型，揭示不同路径对锚定敏感度的差异；

**🔧 技术方法**

基于提示工程与工具调用的控制式推理，使用统一的数值聚合任务、三维指标（Acc_10、UAI、Disc_Δ），并结合随机/链式思考等提示调节进行对照实验；

**📊 数据集**

使用自动生成的合成评估集（14,400条提示），每条包含5个数值证据、黄金答案和固定锚点，覆盖6个业务领域；

**📈 对比分析**

与传统单路径评估相比，AnchorBench能在同一任务框架下比较五条锚定路径，结果显示External和RAG路径最易受锚定影响，ICL影响最小；模型准确度与鲁棒性弱相关；前沿API模型虽然准确率高达96%以上，但仍对可行锚敏感；

**⚠️ 局限性**

受限于合成数据的简化假设、部分路径存在格式差异（History单阶段控制、Tool格式不一致）导致的比较偏差；未覆盖完整的自代理检索与工具循环等实际部署场景；

---

## 345. Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Agent RL

**arXiv ID:** 2608.14312 | [PDF](https://arxiv.org/pdf/2608.14312v1)

**作者:** Xiaojun Wu `[一作]` (IDEA Research), Jian Guo `[通讯]` (IDEA Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对终端代理的环境合成，提出了一种前沿感知（frontier‑aware）的提示策略，利用验证器奖励估计每个种子任务的通过率，构造六种投射与演化方向的候选动作，并通过每个种子单独的混合整数线性规划（MILP）选择最合适的动作，随后同步重写指令、测试、Docker 环境并通过金牌验证后才进入 RL 训练。

**💡 创新点**

创新点在于：①将种子难度评估与动作选择解耦，形成基于验证器奖励的前沿感知评分；②在动作空间中加入投射（增、减、扩）与演化方向（深度/宽度）的组合，并通过 MILP 进行最优选择；③将选动作作为条件，统一生成指令、测试、解决方案等多模态组件，保证一致性与可执行性；④通过软技能覆盖扩展实现多目标学习计划。

**🔧 技术方法**

技术手段包括：混合整数线性规划（MILP）进行动作选择；强化学习框架 GRPO；大模型 Qwen‑3.5‑35B 进行环境合成与 RL 训练；验证器奖励回传；Docker 与终端任务的可执行环境构建；以及指令、测试、数据集、Docker 镜像的同步重写与金牌验证。

**📊 数据集**

主要使用的基准数据集为 Terminal‑Bench 样式任务：tb‑core、tb‑2.0 以及 SWE‑bench Verified，实验中每个方法都输出 100 个已验证的环境；种子任务数量为 100。

**📈 对比分析**

与固定提示策略（few‑shot、Self‑Instruct、Evol‑Instruct）和无合成基线（Base）进行比较。实验结果显示，在 Qwen‑3.5‑35B 上，前沿感知策略在 tb‑core 上 Pass@1 提升 9.2pp（从 40.0% 到 49.2%），tb‑2.0 上提升 6.4pp（从 23.0% 到 29.4%），SWE‑bench Verified 上提升 3.7pp（从 73.4% 到 77.1%）。相较于最强固定策略，增益分别为 2.4pp 与 2.1pp。

**⚠️ 局限性**

局限性主要体现在：①仅评估每个种子单独 MILP 方案，未验证组合（portfolio）模式和更大规模种子池的可扩展性；②固定 100 环境导出规模，未探讨不同规模对性能的影响；③对可解释性与公平性等外部因素的考虑有限；④实验仅在 Qwen‑3.5 系列模型上验证，未检验跨模型与跨任务的泛化能力。

---

## 346. Spatial Message Passing in Language Space for Pathology Image Interpretation

**arXiv ID:** 2608.14309 | [PDF](https://arxiv.org/pdf/2608.14309v1)

**作者:** Jing-Cheng Yang `[一作]` (National Taiwan University), Bin Li `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了一种基于语言空间的空间信息传递框架 SLMP，以在不更新模型权重的前提下为多模态大型语言模型（MLLM）提供病理图像的空间上下文。

**💡 创新点**

创新点在于将WSI网格转化为文本图，利用共享的prompt实现自适应的文本卷积式信息传递，并通过文本梯度优化聚合策略，使得空间推理完全以可读自然语言实现，既保持高解释性又可显著提升诊断性能。

**🔧 技术方法**

使用技术包括大型语言模型（GPT‑4.1、PathGEN、Patho‑R1、MedGemma）、文本梯度优化（TextGrad）对prompt进行迭代改进、基于4连通的空间文本图以及LLM驱动的共享聚合策略。

**📊 数据集**

实验数据集为 HER2 乳腺癌 ROI 以及 CAMELYON16 淋巴结转移 ROI。

**📈 对比分析**

通过在每个模型+数据集上测量 tile 级肿瘤分类准确率，并与无空间信息、随机邻居、未优化 prompt 的基线对比，SLMP 在所有八种设置中提升 3.3–19.6个百分点，显著缩小通用模型与专用模型之间的差距。

**⚠️ 局限性**

局限性包括仅在预选 ROI 而非全切片上验证、单轮信息传递未充分利用多层空间上下文、未将图像信息直接融入 LLM、以及对极端病理情况的适应性可能受限。

---

## 347. Detecting Contaminated Code-Generation Prompt Batches via Influence Functions

**arXiv ID:** 2608.14303 | [PDF](https://arxiv.org/pdf/2608.14303v1)

**作者:** Francesco Quinzan `[一作]` (University of Oxford), Stephen Roberts `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于影响函数的无威胁模型假设的批量级恶意代码生成提示检测方法CodeSIFT；

**💡 创新点**

创新点在于用参数空间影响度量与统计检验来检测提示诱发的安全缺陷，而不依赖先验漏洞知识；

**🔧 技术方法**

采用影响函数、K-FAC近似逆Hessian、Welch一侧t检验等技术；

**📊 数据集**

使用新构建的AuthSec和InfraCloud两套含400条恶意与400条安全提示的Python数据集；

**📈 对比分析**

与Bandit、Semgrep、AST taint tracker等静态分析基线对比，CodeSIFT在3B-7B模型上AUROC最高可达0.98，误报率符合显著性水平；

**⚠️ 局限性**

局限在于仅能检测批量提示、仅覆盖Python及两类安全领域、仅评估至7B模型、无法定位单条恶意提示。

---

## 348. Submodular Policy Learning for Distributed Task Allocation in Open Multi-Agent Systems

**arXiv ID:** 2608.14390 | [PDF](https://arxiv.org/pdf/2608.14390v1)

**作者:** Jing Liu `[一作]` (East China University of Science and Technology), Ruggero Carli `[通讯]` (University of Padova)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种适用于开放多智能体系统的分布式任务分配的政策学习方法SubMAPL。

**💡 创新点**

引入分区多线性扩展(PME)匹配因子化类别政策，使用KL镜像更新实现无偏边际梯度反馈，并设计开放政策迁移机制。

**🔧 技术方法**

基于子模子最优性、连续松弛、KL镜像优化、tabular softmax参数化和动态子模子回报分析。

**📊 数据集**

在30×30网格上进行覆盖任务，使用三种信息密度场（均匀、高斯混合、Gaussian process）作为测试环境。

**📈 对比分析**

与MA‑SPL、MA‑OSEA、REINFORCE和OGS进行对比，SubMAPL在所有场景下获得最高覆盖率、最快收敛速度和最佳稳健性。

**⚠️ 局限性**

仅在离散动作与表格策略下验证，缺乏对高维连续策略、复杂通信约束及更一般任务的评估。

---

## 349. CoRun: Padding is Simple and Efficient for Deterministic LLM Inference

**arXiv ID:** 2608.14376 | [PDF](https://arxiv.org/pdf/2608.14376v1)

**作者:** Shiju Zhao `[一作]` (Nanjing University), Xusheng Chen `[通讯]` (Tencent)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套基于调度的 LLM 推理框架，通过在预填充阶段隔离请求、在解码阶段使用固定形状 CUDA Graph、对 Split‑KV 和集合运算进行选择性约束，并将随机数生成绑定到请求，从而在不需要全批量不变核的情况下实现完全确定性推理。

**💡 创新点**

创新点在于：①识别并利用核函数的“位置不变”属性；②通过控制执行形状而非修改核实现来恢复吞吐量；③提出“隔离预填充+固定形状解码+请求绑定 RNG”的组合方案，首次证明可在任意协作请求、批次位置、到达顺序下保证相同的 token 序列。

**🔧 技术方法**

主要技术包括：CUDA Graph 记录与重放、位置不变核的选择与使用、禁用解码阶段的 Split‑KV、固定树 AllReduce 的确定性集合、以及将 RNG 状态与请求绑定的采样实现。

**📊 数据集**

实验使用了三大公开模型：Qwen3‑235B‑A22B、DeepSeek‑V3 和 Hy3；工作负载包括真实 Azure OpenAI 的 BurstGPT、代码编辑类 Blazedit 以及自定义的固定长度合成负载。

**📈 对比分析**

与 vLLM 标准（STD）和批量不变（BI）基线相比，采用该系统在三种模型和六种工作负载下，聚合吞吐量提升 15–324%，相较 BI 能提升 2 倍以上；TTFT 与 TPOT 分别平均下降约 52% 与 49%；在高并发场景下，解码吞吐量几乎与 STD 持平，且整体延迟优于 BI。

**⚠️ 局限性**

主要局限：①隔离预填充导致在低并发或短 Prompt 场景下产生串行瓶颈；②固定形状解码需要预设最大并发，若并发低则会出现大量填充，降低吞吐量；③不适合变负载或需要混合批量/解码的在线服务；④对需要高精度或 MoE 模型的验证性推理方法而言，回滚成本可能较高。

---

## 350. Robust Constraint-Aware Bayesian Tuning of BBRv2 for QUIC under Tactile Internet Constraints

**arXiv ID:** 2608.14318 | [PDF](https://arxiv.org/pdf/2608.14318v1)

**作者:** Muhammad Hanif Lashari `[一作]` (Iowa State University), Ashfaq Khokhar `[通讯]` (Iowa State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个鲁棒约束感知的贝叶斯调优框架，用于在触摸网络条件下调节 QUIC 协议中 BBRv2 的控制参数

**💡 创新点**

1) 在多种网络场景下构造鲁棒的约束优化目标；2) 将尾延迟与丢包率作为硬约束，同时最大化吞吐量；3) 采用 Tree‑Structured Parzen Estimator 进行样本高效的黑盒优化，并引入两阶段评估策略

**🔧 技术方法**

贝叶斯优化（TPE）、黑盒约束优化、Linux 网络仿真、QUIC 协议实现、实验评测

**📊 数据集**

三组仿真网络场景（延迟 5/20/50 ms，抖动 1/5/10 ms，丢包 0.5%/1%/2%），无公开数据集

**📈 对比分析**

与 Reno、CUBIC、BBR、BBRv2 等基线对比，利用 goodput、95th RTT、丢包率、抖动等指标评估；实验结果显示调优后的配置在尾延迟、抖动与丢包上优于基线，同时保持竞争性的吞吐量

**⚠️ 局限性**

仅在 Linux 网络仿真环境下验证，未考虑真实无线、设备处理和完整触觉回路，缺乏对真实网络和硬件的泛化验证

---

## 351. Sensor-Driven Mission Synthesis for UAV/UGV Swarms: A TB-CSPN Coordination Architecture with Hardware-Enforced Safety

**arXiv ID:** 2608.14306 | [PDF](https://arxiv.org/pdf/2608.14306v1)

**作者:** Uwe M. Borghoff `[一作]` (University of the Bundeswehr Munich), Remo Pareschi `[通讯]` (STAKE lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Guarded Swarms架构，结合TB‑CSPN协调与硬件安全回路，实现多模态传感器信息的安全编排与无人机/地面车队执行。

**💡 创新点**

创新点：①将语义化传感器解释与协调分离，使用时间窗口同步与受限非确定性；②在数字层和物理层之间引入硬件强制安全回路；③在TB‑CSPN中加入时间戳与TTL的token模型，实现可审计的决策链。

**🔧 技术方法**

技术：Topic‑Based Communication Space Petri Net (TB‑CSPN)、consultant/supervisor/worker代理、时间窗口同步、硬件安全电路（阈值比较、姿态/速度限制、射频失声触发等）。

**📊 数据集**

未公开具体数据集，案例使用海岸监视的合成/实验传感数据（雷达、RF、声学、EO）。

**📈 对比分析**

未给出定量比较与性能指标，仅在海岸监视案例中通过模拟演示了增量任务生成与安全屏蔽效果，缺乏实测数据。

**⚠️ 局限性**

局限性：缺乏大规模仿真/硬件验证，通信协议示例抽象化，未证明性能与可靠性，依赖于正确实现的硬件安全板。

---

## 352. GBU-Palm: A Multimodal Video Dataset and Benchmark for Palm Presentation Attack Detection

**arXiv ID:** 2608.14389 | [PDF](https://arxiv.org/pdf/2608.14389v1)

**作者:** Yingjie Ma `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并发布了GBU‑Palm大规模多模态视频掌纹PAD数据集与基准，包含RGB+NIR同步视频、六种采集环境、详细攻击谱/材料/裁剪标签以及攻击源信息。

**💡 创新点**

创新点包括：①同一拍摄窗口下的同步RGB‑NIR原始视频，保证运动与光照动态真实性；②六种环境与多维攻击因子，为跨环境、跨模态与跨攻击类型的系统性评估提供平台；③构造泄漏控制协议，拆分身份、攻击线索、环境，系统分析模型在环境移位、光谱融合与时间顺序利用方面的差异。

**🔧 技术方法**

技术手段主要有：四种视频骨干网络（R(2+1)D‑18、ViViT、Video Swin‑T、MViT‑V2‑S）；光谱融合与单模态评估；时间顺序干预（打乱/逆序）与谱掩码；冻结NIR骨干的线性探测；决策分解为TA/TR/FA/FR。

**📊 数据集**

使用GBU‑Palm数据集，包含21,326段视频（105人、210手掌），6,310段同步RGB‑NIR样本，环境E1–E6，分别有11,108真诚、5,584 Print、4,634 Replay。

**📈 对比分析**

通过P1（环境匹配）与P2（跨环境）两套协议，评估AUC与HTER。P1下四个模型AUC约97.4%（R(2+1)D/MViT）至90%（Video Swin‑T）；P2下AUC下降1–11%不等，跨环境表现高度依赖架构。RGB‑NIR融合对R(2+1)D和ViViT提升显著，但对Swin反而不利；时间顺序干预显示不同模型对顺序敏感度差异。

**⚠️ 局限性**

局限性包括：仅提供二维视频与RGB/NIR，缺乏3D、更多传感器与更具挑战性的攻击；基准侧重二分类，未覆盖多攻击融合与细粒度标签的更深层评估；部分光谱与时间分析仍为粗略，模型对环境、光谱与时间顺序的解释仍不充分。

---

## 353. A Survey of Large Models in Sports

**arXiv ID:** 2608.14377 | [PDF](https://arxiv.org/pdf/2608.14377v1)

**作者:** Yichen Xu `[一作]` (Renmin University of China), Qin Jin `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了大型模型在体育中的应用，构建了六大主体群组和19个任务的系统分类。

**💡 创新点**

创新在于提供了完整的分类体系、系统文献筛选（PRISMA+雪球法）以及对数据集分布与瓶颈的多维度分析。

**🔧 技术方法**

综述聚焦LLM/MLLM技术，包括预训练、细调、RAG、代理框架和多模态对齐等。

**📊 数据集**

使用了多种公开体育数据集，如YourSkatingCoach、SoccerNet‑v2、SportQA、FSBench等，对比了不同模型在这些基准上的表现。

**📈 对比分析**

通过表格对比BLEU、t‑AmAP、CIDEr等指标，展示LLM/MLLM在文本生成与动作识别等任务上已达到或超过传统方法，但仍存在精度与实时性不足。

**⚠️ 局限性**

限制包括快速变化的研究进展、语言与地域覆盖偏差、数据与任务的不平衡、缺乏工业部署案例和解释性不足。

---

## 354. Weakly Supervised Polar Low Segmentation in Sentinel-1 SAR Imagery

**arXiv ID:** 2608.14366 | [PDF](https://arxiv.org/pdf/2608.14366v1)

**作者:** Andrea Federici `[一作]` (UiT Arctic University of Norway), Filippo Maria Bianchi `[通讯]` (UiT Arctic University of Norway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了 Crest，一种利用图像级标签在 Sentinel‑1 SAR 图像中进行极低风暴（polar low）弱监督像素级分割的完整框架。

**💡 创新点**

核心创新点包括：① core 模块通过连通性先验限制新挖掘区域的扩张，避免噪声背景侵入；② dynamic‑bootstrapping（DB）损失将挖掘顺序作为可靠性标记，逐步削弱后期噪声伪标签对训练的影响。

**🔧 技术方法**

技术方案基于 adversarial erasing 的迭代挖掘、Grad‑CAM 可解释性、核心连通约束、动态自举损失以及 SegFormer 分割网络。

**📊 数据集**

实验使用 Sentinel‑1 SAR 极低风暴数据集为主；为了可量化评估，还在公交车乳腺超声（Bus）和 Pascal VOC（人类）数据集上进行验证。

**📈 对比分析**

与单次 Grad‑CAM 与标准 adversarial erasing 进行对比，Crest 在 Bus 上提升 Macro IoU 4.6 %（约 8.4 % 前景 IoU），VOC 上提升 4.0 %（约 4.0 % 前景 IoU），并在 SAR 图像中生成更符合极低风暴结构的分割结果。

**⚠️ 局限性**

局限性包括：core 约束在背景极其复杂或边界极其模糊时可能失效；SAR 数据缺乏像素级真值，评估主要依赖定性；方法对阈值、扩张因子等超参数敏感。

---

## 355. Disentangled Shared Representations Improve Morpho-Transcriptomic Integration

**arXiv ID:** 2608.14355 | [PDF](https://arxiv.org/pdf/2608.14355v1)

**作者:** Julian Ostermaier `[一作]` (ESPCI Paris, PSL University), Daniel Racoceanu `[通讯]` (Sorbonne Université)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了在空间转录组学(ST)与H&E染色图像的多模态学习中，显式分离共享潜在表示与模态特定表示对模型性能的影响；

**💡 创新点**

创新点在于：①提出并比较了两大模型范式（VAE与对比学习）的标准共享潜变量模型与其显式分离共享/私有信息的变体；②构建统一的评估框架，包括跨模态重建、下游线性探针和跨模态探针；③系统观察到分离机制在部分指标（如重建精度与部分下游任务AUC）上能提升，但提升不均匀且高度依赖任务、基因面板大小及β参数。

**🔧 技术方法**

主要技术手段包括：多模态变分自编码器（MMVAE、MMVAE+、MMVAE+_sg），对比学习模型（CLIP、disSSL），InfoNCE损失、KL正则化以实现共享/私有潜变量分离；使用32维共享潜变量、留一切片交叉验证、线性探针评估。

**📊 数据集**

使用两组Visium空间转录组数据：一组是7个结肠癌样本（Colorectal Cancer），另一组是10个胶质母细胞瘤样本（Glioblastoma），共计约97,732个空间探针。

**📈 对比分析**

比较方法：在相同实验条件下训练模型，冻结编码器后分别评估跨模态重建（HE→G与G→HE）、下游线性探针（组织类型与组织小区分类）以及跨模态探针/迁移探针。实验结果显示：对比学习模型（CLIP）整体优于VAE模型；在部分指标上，分离变体（MMVAE+、MMVAE+_sg、disSSL）比标准模型提升，但提升并非统一，且受任务、基因面板大小（K）和β值影响显著。

**⚠️ 局限性**

局限性包括：①分离机制对β和其他超参数高度敏感，最佳设置因任务而异；②实验仅限于两种肿瘤类型，缺乏更广泛组织或多模态数据验证其通用性；③评估指标侧重重建与线性探针，未覆盖更复杂下游任务（如聚类、异常检测等）；④未深入分析分离机制为何在某些方向提升而在其他方向下降，需进一步理论与实验探究。

---

## 356. ScienceFlow: A long-horizon agent for ML research, scientific discovery and beyond

**arXiv ID:** 2608.14354 | [PDF](https://arxiv.org/pdf/2608.14354v1)

**作者:** Mingming Zhao `[一作]` (Huawei), Yanhui Geng `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ScienceFlow，一个面向长周期可执行研究的自主智能框架，能够在大规模计算预算下持续进行实验、验证和迭代。

**💡 创新点**

创新点在于将可恢复的可执行工作空间、ESTRA 重新定位机制以及基于证据的资源调度三大模块融合为一个统一体系，使得代理能够在资源受限、决策不确定的环境中实现持续的高质量探索。

**🔧 技术方法**

技术实现包括：可执行状态快照与归档、文本与工具交互的前向研究循环、ESTRA 通过“锚点-方向”决策实现分支和恢复、证据感知的执行控制器（Admission & Review）、以及多工作者间的紧凑记忆压缩与同步。

**📊 数据集**

评测使用了多领域数据集：MLE-bench（75 个 Kaggle 竞赛）、SciModelingBench（12 项科学设计任务）、Circle Packing、Ratio Minimization、Uncertainty Inequality、KTTSP 轨道调度任务以及各类机器学习数据集（表格、图像、文本等）。

**📈 对比分析**

与现有基线相比，ScienceFlow 在 75 任务 MLE-bench 上取得 70.22 ± 1.18% 的 Any‑Medal 成绩，比最强基线高 4.92pp；在数学优化任务上与 ThetaEvolve/MLEvolve 相当，Hermite 误差约 2.5%；在 KTTSP‑hard 轨道调度上排名第三；在 SciModelingBench 上获得 54.41 的组均衡得分，居首。

**⚠️ 局限性**

局限性包括：对大规模算力仍有依赖，工作空间快照存储开销巨大（尤其在无增量压缩时），ESTRA 的决策依赖于 LLM 的判断，若 LLM 产生误判可能导致不必要的回滚；此外，系统的多工作者协同仅通过压缩摘要实现，缺乏直接的工作空间共享，限制了跨工作者的经验迁移。

---

## 357. CORAL: Curriculum-Optimized Reward Adaptation for LiDAR-Based Goal-Directed Urban Driving

**arXiv ID:** 2608.14332 | [PDF](https://arxiv.org/pdf/2608.14332v1)

**作者:** Anisa Saleem `[一作]` (Korea University of Technology and Education), Duksu Kim `[通讯]` (Korea University of Technology and Education)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并训练了一种名为 CORAL 的目标导向城市驾驶策略，利用强化学习在 CARLA 仿真器中实现 LiDAR 基的自动驾驶。

**💡 创新点**

创新点在于：① 将五阶段路程和行为约束课程与阶段感知奖励同步推进；② 使用极简的 99 维状态（包含 64 维极坐标 LiDAR 直方图、路线信息、规则信号和车辆遥测），完全不依赖视觉或点云编码；③ 通过多流 actor‑critic 网络将不同模态特征融合。

**🔧 技术方法**

主要技术包括：Proximal Policy Optimization (PPO)、多流 actor‑critic 神经网络、极坐标 LiDAR 直方图特征、阶段化奖励权重和逐步课程调度。

**📊 数据集**

数据集：在 CARLA 0.9.15 的 Town05 进行训练；在 Town01–Town07 及 Town10 进行零样本迁移评估。

**📈 对比分析**

性能对比：与两种基准 PPO（E2E 和 CuRLA‑Inspired）相比，CORAL 在最难阶段（100–150 m 路程）实现 100% 成功率，而基准仅 5–10%；在 7 个未见城市的迁移测试中，成功率 68–98%（平均 lateral deviation <0.35 m），显著优于基准。

**⚠️ 局限性**

局限性：① 评估环境为静态，LiDAR 直方图对性能贡献不明显；② 红灯合规率仅约 60%；③ 未验证在动态交通、不同光照或天气条件下的鲁棒性；④ 仅使用单一随机种子，未评估算法的稳定性。

---

## 358. Solving QBF with Counterexample Guided Refinement

**arXiv ID:** 2608.14322 | [PDF](https://arxiv.org/pdf/2608.14322v1)

**作者:** Mikoláš Janota `[一作]` (IST/INESC-ID), Edmund Clarke `[通讯]` (Carnegie Mellon University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两种基于CEGAR的QBF求解方法，一种是递归的CEGAR驱动求解器，另一种是将CEGAR作为DPLL求解器的学习机制

**💡 创新点**

创新点在于将CEGAR从传统的两层量化扩展扩展到任意层级的递归求解，并在DPLL中引入新的抽象-细化学习策略，显著减少扩展引起的内存爆炸和剪枝空间

**🔧 技术方法**

核心技术包括：量化抽象与细化（CEGAR）、多游戏抽象、递归抽象求解、增量化SAT求解、单元传播和纯文字规则、DPLL学习与CEGAR学习交叉

**📊 数据集**

实验使用QBF-LIB中来自形式验证与规划的若干家族（如incrementer-encoder、Robots2D、irqlkeapclte等），共计数千个实例，全部预处理后进行评测

**📈 对比分析**

与现有DPLL求解器（如DepQBF）、扩展式求解器（QuBE、QuBE++）以及电路式求解器（QuE）比较，CEGAR驱动求解器在多数家族中最快，整体比第二好约33%，而CEGAR学习在部分家族显著提升（如incrementer-encoder、Robots2D、trafficlight-controller）

**⚠️ 局限性**

主要限制包括：当最终需要完全扩展时，仍会面临与传统扩展求解器相同的内存消耗；细化过程会导致额外的时间开销；在某些家族中CEGAR学习反而略微降低性能；实现中需手动合并相邻量化块，增加实现复杂度

---

## 359. CRAFT: Constrained Reward via Attention Fine-Tuning for Subject Personalization without Composed Targets

**arXiv ID:** 2608.14403 | [PDF](https://arxiv.org/pdf/2608.14403v1)

**作者:** Jihun Park `[一作]` (DGIST), Sunghoon Im `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种无需合成目标图像、仅使用参考图像及其遮罩的奖励式微调框架CRAFT，用于通用主题驱动图像个性化；

**💡 创新点**

创新点在于提出“Where to look”原理，通过注意力级奖励引导噪声和文本对齐到正确的参考区域，并将其生成的注意力掩码作为像素级身份奖励的门控，实现在仅10K参考样本下无需配对训练即可达到SOTA；

**🔧 技术方法**

主要技术包括：基于MMDiT的参考感知扩散模型、LoRA适配器、单步ReFL奖励反馈、噪声-参考、文本-参考和噪声-噪声一致性三种注意力奖励、以及基于DINOv2的像素级身份奖励；

**📊 数据集**

使用10,000张自合成的单主体参考图像及对应Grounded-SAM生成的遮罩，未使用任何合成目标对；

**📈 对比分析**

与现有方法（UNO、MOSAIC、XVerse、UMO等）在XVerseBench、DreamBench和OmniContext上比较，CRAFT在单主体和多主体均获得最高或接近最高的DPG、ID、IP、AES四项指标，整体平均得分达到76.47/77.80，显著优于使用150K–2M配对样本的传统方法；

**⚠️ 局限性**

主要局限：在多主体场景下相较于部分基于合成目标的基线仍有身份保持差距，需较大身份权重才能缩小差距；仅适用于已支持参考令牌的MMDiT模型；奖励式方法受底层模型路由能力限制，无法突破原始模型性能上限。

---

## 360. IRGNN: Efficient Invariant Radar Graph Neural Network for Radar Point Cloud Object Detection

**arXiv ID:** 2608.14394 | [PDF](https://arxiv.org/pdf/2608.14394v1)

**作者:** Xiao Guo `[一作]` (China Agricultural University), Caicong Wu `[通讯]` (China Agricultural University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了IRGNN框架，利用翻译‑旋转不变的图结构对雷达点云进行建模，并通过改进的MPNN实现目标检测；

**💡 创新点**

创新点在于引入了局部参考框架的增强点对特征(EPPF)、残差连接和虚拟节点层，以提升图表示的鲁棒性和全局上下文捕获能力；

**🔧 技术方法**

主要技术包括图神经网络（MPNN）、特征嵌入、残差块、虚拟节点层、旋转不变点对特征EPPF以及后处理的向量化与内存释放优化；

**📊 数据集**

使用RadarScenes雷达点云数据集进行实验；

**📈 对比分析**

在RadarScenes上与RadarGNN、PointNet++、PointPillars、DOG等方法比较，IRGNN在mAP上达到0.591，整体表现最佳；

**⚠️ 局限性**

局限性包括旋转不变特征削弱姿态相关信息导致提升有限、对极稀疏目标如行人仍表现不佳，以及在不同硬件平台上仍需进一步优化部署效率。

---

## 361. Scaling 5G-TSN Bridges: Operating Regimes, Scheduling, and Time Synchronisation Under Heterogeneous Industrial Traffic

**arXiv ID:** 2608.14386 | [PDF](https://arxiv.org/pdf/2608.14386v1)

**作者:** Mohamed Seliem `[一作]` (University College Cork), Dirk Pesch `[通讯]` (University College Cork)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用nascTime+OMNeT++/Simu5G模拟平台，对5G-TSN桥接器进行可扩展性与调度算法评估，研究多种工业流量混合下的端点数、调度器、带宽、衰落等因素对QoS、延迟和时钟同步的影响。

**💡 创新点**

①首次系统性划分三大工作模式（低负载、饱和临界、过载）并量化饱和阈值随带宽的倍数关系；②证明QoS感知比例公平调度（QoS-PF）在饱和与过载阶段能显著降低关键流延迟并保持高交付率；③通过将gPTP放在独立高优先级承载上，显著缓解时钟失稳，但仍无法完全限定在饱和下的gPTP驻留时间。

**🔧 技术方法**

nascTime框架（OMNeT++/INET/Simu5G）、四种MAC调度器（MaxCI、PF、DRR、QoS-PF）、四路SDAP承载、Jakes多径衰落模型、带宽10/20MHz、30SCS numerology。

**📊 数据集**

模拟基准负载：闭环控制（CLC）、机器视觉高低优先级（MV-HP/MV-BE）、批量遥测（BLK）以及IEEE802.1AS/gPTP同步流；端点数从1到40，依据3GPP TR 22.804生成的报文尺寸、周期与时限。

**📈 对比分析**

对比方法：在同一负载、带宽、衰落条件下，测量PDR、平均/99th/99.9th/最大延迟、延迟合规率、空间公平度和gPTP驻留时间；结果显示：在低负载下所有非DRR调度器表现相近；接近饱和时QoS-PF将关键流P99延迟降低1–2个数量级；在过载时仅QoS-PF保持近乎完整的关键流交付；带宽翻倍后饱和阈值约翻倍。

**⚠️ 局限性**

①仅单小区、静态UE，未考虑多小区干扰、切换与协作调度；②仿真模型（Simu5G/INET）与商用基站调度、缓冲区策略可能不同；③只评估了特定的工业流量组合，未考虑事件驱动或更宽的上行/下行比例；④衰落仅采用单一Jakes六路径模型；⑤未模拟TSN侧的时间感知/信用基准形状；⑥随机种子有限，尾部统计不够稳健。

---

## 362. DeaMoE: Efficient MoE Structure for Fast Small-Batch Decoding

**arXiv ID:** 2608.14385 | [PDF](https://arxiv.org/pdf/2608.14385v1)

**作者:** Zewen Jin `[一作]` (University of Science and Technology of China), Cheng Li `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了 DeaMoE 结构与两阶段路由，以提升 MoE 大模型在小批量解码时的速度和吞吐量。

**💡 创新点**

创新点在于将专家按部门聚类共享大部分参数，仅保留少量私有参数；提出两阶段部门‑专家路由机制，显著减少权重加载，降低内存访问瓶颈。

**🔧 技术方法**

技术包括：预训练 7.3B MoE 模型、vLLM+Triton 自定义 kernel、分组 top‑k 路由、非线性激活（SiLU）、专家参数身份初始化以及两阶段路由实现。

**📊 数据集**

使用的主要数据集有：RedPajama‑v1（110B 训练 tokens）、下游评测集（BoolQ、PIQA、SIQA、HSwag、WinoGrande、RaceH、AnliR1‑R3、OpenBookQA）以及语言建模基准（PTB、WikiText‑103/2）。

**📈 对比分析**

与预算匹配的标准 MoE 基线进行对比；在 NVIDIA A40/H100 GPU 上做微基准和 vLLM 端到端服务；DeaMoE 在小批量（4–128）下权重加载下降 49–63%，TPOT 加速 1.02–1.33 倍，DeepSeek‑V3 最高 2.0× 微基准加速。

**⚠️ 局限性**

局限性：在小规模专家或高带宽硬件（如 H100）上加速空间有限；强制部门覆盖会影响模型质量；对非小批量场景收益不明显，需要额外实现与调优。

---

## 363. Wrong but Useful: Trajectory Value Beyond Answer Correctness in Multi-Agent Messages

**arXiv ID:** 2608.14375 | [PDF](https://arxiv.org/pdf/2608.14375v1)

**作者:** Chih-Hsuan Yang `[一作]` (Argonne National Laboratory), Rajeev Thakur `[通讯]` (Argonne National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多代理推理系统中消息的答案正确性与对后续推理价值的区别，并通过多种实验协议对其进行测量。

**💡 创新点**

创新点是提出“提议正确性（proposal correctness）”与“轨迹价值（trajectory value）”的二分法，设计了DHD测量协议、留一回放与重复实验方法，以及成分屏蔽诊断，揭示错误答案消息中仍可帮助推理的现象。

**🔧 技术方法**

使用多代理角色生成、离线缓存、留一法回放（LOO）、重复回放、成分屏蔽等技术，并对不同模型（OSS、Gemma）进行评估。

**📊 数据集**

采用五个科学推理基准：Omni‑MATH‑2、JEEBench、SciBench、LAB‑Bench、MaScQA，涵盖数学、考试、科学与材料等领域。

**📈 对比分析**

通过与独立求解、完整池、以及留一池等对比，发现错误答案消息中约有 40% 以上在改变最终正确性时是有益的；在 OSS 和 Gemma 上平均准确率提升约 1–2 个百分点，验证了轨迹价值与答案正确性不完全重合。

**⚠️ 局限性**

局限性包括：标签高度依赖具体池与整合上下文，难以直接推广到未见问题；实验仅在离线固定池环境下进行，未覆盖交互式辩论等动态场景；结果受模型随机性影响，且仅对 OSS 与 Gemma 两大模型验证；成分屏蔽诊断仅针对部分消息，无法完全解析原因。

---

## 364. Boosting Data Augmentation with Stochastic Weight Averaging

**arXiv ID:** 2608.14373 | [PDF](https://arxiv.org/pdf/2608.14373v1)

**作者:** Longde Huang `[一作]` (Chalmers University of Technology and University of Gothenburg), Jan E. Gerken `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在数据增强下使用随机权重平均（SWA）对深度学习模型的对称性（等变性）提升效果，并给出了理论分析与实验验证。

**💡 创新点**

将SWA建模为Ornstein–Uhlenbeck过程，推导其在等变性方向的性能提升超越普通SWA带来的提升；在无限宽极限下结合神经切线核（NTK）给出等变性提升的下界；提出可观测的等变性损失代理，并与实验结果对齐。

**🔧 技术方法**

采用概率动力学（将SGD视为SDE）、Ornstein–Uhlenbeck过程、NTK理论、群表示论与数值实验相结合。

**📊 数据集**

使用合成任务、MNIST、Fashion‑MNIST、CIFAR‑10/100、ImageNet‑100（图像分类）以及DHFR、BZR、COX2（分子图分类）等数据集，并在这些数据集上采用C4旋转、SO(3)的四面体/八面体/十二面体子群进行数据增强。

**📈 对比分析**

对比SWA前后模型在准确率、Orbit Same Prediction (OSP) 及 KL 等变性损失上的变化；实验表明SWA始终提升等变性，并且 R⊥/R>1，说明等变性提升大于整体性能提升，在多种网络架构（MLP、CNN、ViT、GNN）和多种数据集上均得到验证。

**⚠️ 局限性**

理论分析基于无限宽网络、充分训练、离散群、无结构参数限制等理想假设；在实际有限宽、非均匀参数空间、复杂任务或连续对称性时理论下界可能松散，且对非离散或连续对称性的评估尚未充分探讨。

---

## 365. Mind the Long Tail: Understanding the Difficulty of Delay Detection in Business Processes

**arXiv ID:** 2608.14367 | [PDF](https://arxiv.org/pdf/2608.14367v1)

**作者:** Keyvan Amiri Elyasi `[一作]` (University of Mannheim), Heiner Stuckenschmidt `[通讯]` (University of Mannheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对业务流程剩余时间预测的分布特征和模型误差进行系统分析，评估了不平衡回归技术，并提出利用异方差不确定性提升延迟检测的框架。

**💡 创新点**

创新点在于揭示剩余时间分布的强右偏和随延迟增加的异方差不确定性，证明传统不平衡回归方法改进有限，并首次利用不确定性信息显著提高高延迟案例识别。

**🔧 技术方法**

主要技术包括基于LSTM的剩余时间回归、SMOGN数据重采样、成本敏感加权/BMSE/EAL/SERA算法级不平衡处理、基于离散时间生存分析的预测分布、以及CatBoost的延迟检测分类器。

**📊 数据集**

使用了14个公开事件日志（BPIC20PTC、BPIC15系列、Helpdesk等），覆盖多行业业务流程。

**📈 对比分析**

在各目标区间（many、medium、few）采用nMAE、召回率、F1、PR‑AUC等指标进行比较；不确定性模型相较于传统回归模型，在召回率和F1上提升约40%（从0.21提升至0.61），同时保持或略增精度。

**⚠️ 局限性**

局限性包括：不确定性来源未完全解析，阈值化的二进制延迟检测可能丢失信息，实验仅限公开日志和LSTM架构，缺乏在工业真实环境和其他模型上的验证。

---

## 366. Epistemic Tensions: Reframing A Visualization Co-Design through Entanglement Theory

**arXiv ID:** 2608.14364 | [PDF](https://arxiv.org/pdf/2608.14364v1)

**作者:** Wei Wei `[一作]` (University of Victoria), Sheelagh Carpendale `[通讯]` (Simon Fraser University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2023–2025 年在加拿大温哥华岛的艺术社区与可视化研究者共同开展的社区协同设计项目进行长期反思，运用交织理论重新框定并解读项目中出现的 epistemic tensions。

**💡 创新点**

创新点在于：①将交织理论（来源于女性主义与量子物理的“代理现实主义”）引入可视化研究，提供一种全新的解释视角；②把 epistemic tensions 视为可转化为研究与方法论机会的生产性张力；③提出三种“模糊前端”与“可视化黑暗面”概念，揭示协同设计与可视化管道中潜在的不平等与排斥。

**🔧 技术方法**

主要技术方法包括：参与式工作坊（数据工作坊、数据艺术驻地）、交互原型（Tableau 视觉化、基于 Web 的 Data Painter）、持续反思与迭代式共同设计会议；在反思层面采用叙事性与案例分析。

**📊 数据集**

使用的数据集为：加拿大艺术委员会（Canada Council for the Arts）提供的联邦资助记录，以及加拿大税务局（Canada Revenue Agency）公布的税务数据；项目中未引入外部公开大数据集，且聚焦于社区内部可收集的本地艺术数据。

**📈 对比分析**

本文不进行传统意义上的量化对比或性能评估，而是通过对项目进程的回顾性分析、参与者访谈与多方立场陈述来展示方法的可行性与对知识生产的影响；因此无法给出精确的性能指标。

**⚠️ 局限性**

局限性包括：①缺乏经验性评估与可重复性验证，结果高度依赖特定社区与项目背景；②对交织理论的依赖使得结论在理论取向上可能有限；③项目规模有限，难以推广至更大范围的跨学科可视化实践；④因聚焦反思性研究，缺少对可视化工具的技术细节与性能测评。

---

## 367. Local and Global Regimes of Geometric Complexity in Language Model Representations

**arXiv ID:** 2608.14361 | [PDF](https://arxiv.org/pdf/2608.14361v1)

**作者:** Arwa Osman `[一作]` (Universitat Pompeu Fabra), Iuri Macocco `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了词汇多样性对大型语言模型中上下文化词表示的内在维度 (ID) 的影响，发现 ID 在不同尺度下呈两种不同的模式，并推导出转折点公式 n_transition = N/k。

**💡 创新点**

首次将词汇多样性作为受控变量系统性研究，揭示了邻域尺度与样本量之间的关系导致的 ID 传递，并给出精确无参转折点预测公式。

**🔧 技术方法**

使用邻域基 ID 估计器 GRIDE（以及 MLE 对比），并通过同词近邻比例诊断验证了聚类假设；对 Transformer 隐层隐藏状态进行 ID 计算。

**📊 数据集**

基于 WikiText-103 的 84M 样本，构造 11 种词汇多样性（1–10,000 个唯一名词）且每组总样本固定 10,000 的实验集合。

**📈 对比分析**

对 6 种邻域尺度 k (16–512) 与 2 个 8B 规模模型（Qwen3-8B、Meta-Llama-3-8B）进行 ID 曲线比较；发现转折点公式在所有尺度、模型、估计器上均准确预测，并通过同词比例验证模型的聚类特性，说明 ID 评估在不同模式下需谨慎解释。

**⚠️ 局限性**

仅在英语 Wikipedia 文本上验证，未探究其他语言或领域；仅限 8B 规模模型，无法确定模型大小对转折点的影响；同词近邻假设在后期层弱化，可能限制了理论推导的通用性。

---

## 368. ATLAS: Discovering Agent Strategies through LLM-Guided Abstraction and Automata Learning

**arXiv ID:** 2608.14352 | [PDF](https://arxiv.org/pdf/2608.14352v1)

**作者:** Ignacio D. Lopez-Miguel `[一作]` (TU Wien), Martin Tappler `[通讯]` (TU Wien)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 ATLAS 方法，通过 LLM 驱动的轨迹抽象和自动机学习，自动恢复可解释的概率有限状态机来刻画 LLM‑驱动渗透测试代理的行为策略。

**💡 创新点**

创新点在于将自然语言模型的语义抽象与状态合并的自动机学习相结合，首次将代理轨迹转化为正式可分析的行为模型，并支持知识迁移、解释生成和模型切片等后续工程任务。

**🔧 技术方法**

使用技术包括：LLM 进行动作/观察分类与归一化、Alergia 状态合并算法构建 Markov 链、符号知识迁移、模型切片与简化。

**📊 数据集**

实验数据集为 HackingBuddyGPT 在 12 台 Docker 漏洞机器（含 Docker Scenario 3）上的交互轨迹。

**📈 对比分析**

与传统手工抽象或仅记录执行轨迹的方法相比，ATLAS 能生成更简洁且信息丰富的行为模型，能够揭示成功与失败路径，并实现跨 LLM 的知识迁移；虽然本文未给出数值指标，但实验表明模型在解释性和迁移性上均优于基线。

**⚠️ 局限性**

主要限制包括：高度依赖强大 LLM 进行抽象，长轨迹或多变环境下的可扩展性尚待验证；模型学习过程需要足够多的样本以捕捉所有行为模式；以及当前仅在渗透测试场景验证，需进一步在其他领域证明通用性。

---

## 369. Clearing the Fog: Towards Installing and Refining Proactive Exploration Capabilities in LLM Agents

**arXiv ID:** 2608.14339 | [PDF](https://arxiv.org/pdf/2608.14339v1)

**作者:** Zhizhao Guan `[一作]` (Sichuan University), Anthony G Cohn `[通讯]` (University of Leeds)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型代理的主动探索能力，提出了 SAFARI 方法

**💡 创新点**

创新点：利用树结构上下文建模消除回顾偏差，并通过对比信号引导的RL细化探索边界

**🔧 技术方法**

技术：树结构上下文建模、探索数据构造、对比样本强化学习（DPO）以及蒙特卡洛滚动评估

**📊 数据集**

数据集：WebShop、InterCode‑SQL、ScienceWorld 以及 GPT‑4o 生成的教师轨迹

**📈 对比分析**

对比：在三大基准上相较 SFT‑Only、ETO、IPR、STeCa 等基线提升约10‑15% 任务成功率、8‑18% 探索得分

**⚠️ 局限性**

限制：依赖教师SFT、效率评估指标有限，且无法完全自学探索

---

## 370. From Style Replication to Style Exploration: Enabling Art Style Exploration with Analyze-Experiment-Resituate Framework

**arXiv ID:** 2608.14405 | [PDF](https://arxiv.org/pdf/2608.14405v1)

**作者:** Wen-Fan Wang `[一作]` (National Taiwan University), Bing-Yu Chen `[通讯]` (National Taiwan University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了 Analyze-Experiment-Resituate (AER) 框架，用于支持专业数字艺术家在 Generative AI 环境下进行风格探索，并通过实验与现场研究验证其效果。

**💡 创新点**

创新点在于将风格探索拆解为可解释的三阶段流程（分析、实验、重置），在每个阶段提供可控的提示、可视化的风格元素、以及多角色社会反馈，显著提升艺术家对生成过程的可解释性、创作自主性与反思深度。

**🔧 技术方法**

技术实现主要基于 GPT‑5 进行多模态风格元素提取与解析、Flux‑Context‑Pro/Flux‑2Pro 进行图像生成、AutoGen 框架生成专业艺术家/观众/粉丝三角色反馈，并通过自定义提示工程实现交互。

**📊 数据集**

数据来源包括：10 位专业数字艺术家访谈记录、16 位实验参与者的原始作品与生成结果、4 位现场研究参与者的日常使用日志，以及艺术家自选的参考作品集。

**📈 对比分析**

通过与直接风格迁移基线的对比实验（16 位参与者、双条件设计）和两周现场研究，使用自评问卷（Agency、Creative Self‑Efficacy、TSRI 及其子量表）以及半结构访谈进行评估；结果显示 AER 在代理感、创意自效感与技术支持反思方面均有显著提升（p<0.05，Cohen's d≈0.85–0.92）。

**⚠️ 局限性**

局限性包括：自动化分析可能导致艺术家对风格解析的能力下滑；模拟反馈的真实性与可接受度受限；缺乏文化、历史背景信息的支持；样本主要为专业艺术家，规模有限，且未涵盖新手或跨领域创作者。

---

## 371. Whose doctor does the AI recommend? An algorithm audit of reputation and demographic signals in large language model-assisted physician choice

**arXiv ID:** 2608.14399 | [PDF](https://arxiv.org/pdf/2608.14399v1)

**作者:** Syeda Anshrah Gillani `[一作]` (Heidelberg University), Mirza Samad Ahmed Baig `[通讯]` (Fandaqah)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

未给出

**💡 创新点**

未给出

**🔧 技术方法**

未给出

**📊 数据集**

未给出

**📈 对比分析**

未给出

**⚠️ 局限性**

未给出

---

## 372. LLMs Don't Pay for the Jump

**arXiv ID:** 2608.14397 | [PDF](https://arxiv.org/pdf/2608.14397v1)

**作者:** Paras Balani `[一作]` (Birla Institute of Technology and Science, Pilani), Subhrakanta Panda `[通讯]` (Birla Institute of Technology and Science, Pilani)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过分析Planck量子化假设的溯因过程，提出了“成本耦合溯因”标准，并指出当前大语言模型缺乏将认知误差与物理计算成本耦合的机制，导致其在科学发现中无法产生真正的新假设；

**💡 创新点**

创新点在于将溯因视为需要监管不确定性（即误差产生的物理成本），提出热力耦合的必要性，并将其与LLM软max熵解耦现象关联，首次量化LLM推理的热力学属性；

**🔧 技术方法**

采用热力学耦合定义、软max–吉布斯类比理论推导、以及对现有LLM的熵与准确率实验评测等技术手段；

**📊 数据集**

使用公开大模型（Llama 3B/70B、GPT‑4o‑mini）在Kepler/Newton/Newton‑OOD任务集及InAbHyD基准上的表现数据；

**📈 对比分析**

实验显示LLM的token熵随任务难度基本不变，准确率下降显著，表明当前模型缺乏热力耦合；

**⚠️ 局限性**

局限性在于未实现具备成本耦合的硬件或架构，实验仅验证熵与准确率的关系，未检验耦合对溯因性能的实际提升效果。

---

## 373. AgentRewind: Recoverable Execution for Long-Horizon LLM Agents

**arXiv ID:** 2608.14380 | [PDF](https://arxiv.org/pdf/2608.14380v1)

**作者:** Yu Zhuang `[一作]` (University of Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为AgentRewind的运行时恢复框架，能够在大语言模型代理执行长周期任务时记录并恢复对齐的代理上下文与受控环境状态，并保留回滚记忆；同时构建了MettleBench基准，用来评估多需求工程任务的完成度与部分进展。

**💡 创新点**

创新点在于：①实现了代理与环境状态的同步快照与回滚机制，②提供了回滚记忆以引导后续决策，③设计了多需求的长周期工程任务基准MettleBench，显著补足了现有单一需求评测的不足。

**🔧 技术方法**

技术上采用了LLM上下文记录、工具调用跟踪、文件系统变化捕捉与对齐快照，构建检查点元数据、回滚记忆；通过在代理与受控环境之间插入恢复层实现即时回滚；实验中使用GPT‑5.4、Qwen‑3.7‑Max等模型。

**📊 数据集**

使用的数据集包括：MettleBench（82个工程任务，来源于Terminal‑Bench 2.0、ProgramBench、SWE‑bench、ProjectEval、GitTaskBench），以及对比实验中使用的Terminal‑Bench 2.0完整任务集。

**📈 对比分析**

通过与Continue、Restart with Experiences、Safety Review等执行策略的对比，并在七大模型、三种代理框架和多种任务中进行实验，AgentRewind在任务成功率、检查表进度以及平均轨迹长度等指标上均明显优于基线，尤其在中长轨迹任务中提升幅度最大。

**⚠️ 局限性**

局限性包括：仅能恢复受控环境状态，无法撤销外部服务或网络调用；需要外部验证机制识别停滞状态；以及对系统级恢复（如进程内存、网络状态）的支持尚不足，未来需进一步扩展跨系统恢复与内置进度评估能力。

---

## 374. Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation

**arXiv ID:** 2608.14379 | [PDF](https://arxiv.org/pdf/2608.14379v1)

**作者:** Yuxuan Chen `[一作]` (Shanghai Jiao Tong University), Xiao Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对快速反应与预测能力要求的动态机器人操纵基准 ReflexBench，并在此基准上设计了高效的 Vision‑Language‑Action 模型 ReflexVLA；

**💡 创新点**

创新点包括：① 将推理延迟显式融入评估；② 通过潜在未来预测与多帧时序融合提升预测性；③ 在推理阶段采用批量视觉编码与 CUDA Graph 重放显著降低延迟；

**🔧 技术方法**

使用了 ViT+DINOv2/SigLIP 视觉编码器、Qwen2.5 语言模型、Transformer 的多模态注意力、潜在未来预测损失、多帧时序注意力、CUDA Graph 加速；

**📊 数据集**

数据集为 ReflexBench 的六个动态操纵任务（传送带拣放、捕球、打地鼠、滚球拦截、抛球、旋转插座）共 200 条演示；

**📈 对比分析**

与多种基线（VLA‑Adapter、SmolVLA、DynamicVLA、PUMA、OpenVLA‑OFT、π₀.5）在 ReflexBench 上对比，ReflexVLA 以 1B 参数取得 50.4% 的平均成功率，超过大多数基线且在静态任务 LIBERO 上也保持 97.2% 的性能；

**⚠️ 局限性**

局限性在于：① 未来预测与多帧融合仅在微调阶段使用，缺乏大规模预训练支持；② 仅评估了同步与异步两种推理模式，未探索更高级的推理机制（如 RTC）等。

---

## 375. A Hybrid LLM-Based Framework for Automated Security Annotation Generation in Business Process Models

**arXiv ID:** 2608.14370 | [PDF](https://arxiv.org/pdf/2608.14370v1)

**作者:** Md Kamrul Islam `[一作]` (CentraleSupélec), Sami Souihi `[通讯]` (Université Paris-Saclay)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并实现了一套混合型框架，利用大型语言模型（LLM）与规则引擎自动从自然语言安全需求中提取并映射SecBPMN2安全注解，输出结构合法的BPMN 2.0模型。

**💡 创新点**

核心创新在于将语义抽取与模式约束分离，先用LLM得到语义化安全目标，再通过规则校验与映射确保注解严格符合SecBPMN2的附加规则，兼顾准确性与可验证性。

**🔧 技术方法**

使用GPT‑4.1‑mini和Mistral Small 3.2进行LLM推理，配合检索增强生成（RAG）与低温度提示，规则化验证与JSON‑to‑XML重构，整个管线实现自动化。

**📊 数据集**

使用了27个行业案例（航空、医疗、金融、公共服务、酒店）构建的公开基准数据集，每个实例包含BPMN模型、自然语言安全描述与人工标注的SecBPMN2注解。

**📈 对比分析**

通过与两名安全专家及一名硕士生的手工注解对比，采用精确匹配的精确率/召回率/F1等指标；在所有复杂度层级下，系统平均精确率达到0.58，召回率0.52，F1 0.52，明显优于人类专家（精确率0.29，召回率0.50，F1 0.33），同时注解速度大幅提升。

**⚠️ 局限性**

局限性包括：仅支持BPMN子集；缺乏自动冲突修复与修补能力；数据集规模有限，可能影响结果泛化；检索增强在复杂模型中效果不一；LLM对长文本的上下文利用仍有待改进。

---

## 376. Non-Parametric Spatiotemporal Trajectory Prediction via State-Conditioned Transition Sampling

**arXiv ID:** 2608.14349 | [PDF](https://arxiv.org/pdf/2608.14349v1)

**作者:** Michael Fore `[一作]` (Amazon Web Services), Duncan Botti `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种无训练参数、无GPU的多模态轨迹预测方法，利用历史状态-下一位置对构建转移表并通过多维核加权检索邻居，生成多条预测轨迹。

**💡 创新点**

创新点在于将状态条件转移表与多维（空间、航向、速度、时间）核加权结合，提供可解释且数据高效的多模态预测，并通过空间排斥多样采样和束搜索实现不同预测模式。

**🔧 技术方法**

采用BallTree索引、Nadaraya‑Watson 核回归、von Mises 航向核、Gaussian 速度与时间周期核、空间排斥多样采样及束搜索等技术实现无学习参数的自回归推理。

**📊 数据集**

使用丹麦海事AIS（DMA AIS）数据集，包含 10,605 条训练轨迹和 1,481 条测试轨迹，时间跨度为 2019 年。

**📈 对比分析**

与 TrAISformer、SPNS、NCDM、LSTM 等基线对比，方法在全数据下与 57M 参数 Transformer 接近，在低数据（仅 10% 训练数据）时性能更优，beam 搜索 Top‑1 FDE 为 9.00 km，diverse 采样 best‑of‑16 FDE 为 2.49 km。

**⚠️ 局限性**

局限性包括无法外推未出现的稀有状态、固定速度假设导致对加速/减速行为不敏感、无法一次性满足所有指标（如 CRPS）且在高密度地区邻居数增长需通过表格上限控制。

---

## 377. A Four-Axis Trustworthiness Benchmark for LLM-as-Judge in Principle-Based Regulation

**arXiv ID:** 2608.14329 | [PDF](https://arxiv.org/pdf/2608.14329v1)

**作者:** Dipankar Sarkar `[一作]` `[通讯]` (Independent Researcher), Dipankar Sarkar (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文创建了基于原则监管的评估框架，推出了Principle‑Bench基准数据集，并设计了Ceca校准化的实例集群评估器，用于多轴（准确率、同义改写鲁棒性、对抗鲁棒性、校准性）评估LLM裁判的可信度。

**💡 创新点**

创新点在于①首次构建针对监管原则的专门基准及其扰动套件；②提出Ceca模型，实现闭式可审计的逐实例反事实归因；③以四轴评估标准系统化比较各方法的稳健性与可解释性。

**🔧 技术方法**

采用的大模型评判技术包括大型语言模型裁判、句子Transformer嵌入（MiniLM、BGE、Ollama‑mxbai）、关键词计数、Platt标度校准以及层级式裁决（cascade）。

**📊 数据集**

使用的数据集为168个金融促销场景，覆盖COBS 4.5A.3R与Consumer Duty两项UK FCA原则，按原始、同义改写、关键词对抗、边界四种扰动划分，所有标注均基于预注册的rubric并由作者准备。

**📈 对比分析**

对照方法包括关键词计数、三种Transformer嵌入、开源LLM裁判与级联裁判；实验显示LLM在原始样本准确率最高，但在对抗样本上准确率骤降47个百分点；嵌入器对抗鲁棒性最好；无一方法在四轴上完全占优。

**⚠️ 局限性**

局限性包括：评判模型与语料共用同一模型族导致的模型侧偏差；标注仅由作者完成，缺乏多样化视角；仅覆盖两项UK FCA原则，难以直接泛化至其他监管场景；对抗扰动设计有限，可能未覆盖所有攻击手段。

---

## 378. TRIAGE: Risk-Controlled Pseudo-Label Admission for Annotation-Efficient Semi-Supervised Retinal OCT Classification

**arXiv ID:** 2608.14321 | [PDF](https://arxiv.org/pdf/2608.14321v1)

**作者:** Md Ashraful Hossen Akash `[一作]` (Rajshahi University of Engineering & Technology), Tze Hui Liew `[通讯]` (Multimedia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种半监督学习框架TRIAGE，能够在光学相干断层扫描（OCT）影像分类中利用少量标注、未标注数据以及临床风险约束实现高效诊断；

**💡 创新点**

创新点包括：①基于患者分组的分布式合规风险控制门控，将临床异构成本矩阵嵌入伪标签审批；②层级分类器与部分监督机制，允许在疾病子类型不确定时仅给出异常标签；③3D卷积上下文Transformer教师，用于跨切片一致性验证与风险敏感聚合；

**🔧 技术方法**

技术实现上采用EfficientNet‑B0骨干+MBConv微架构，双EMA教师（单扫描与体积上下文）与层级交叉熵、部分标签损失；Hoeffding‑Bentkus上置信界合规校准；RandAugment、Cutout等数据增强；

**📊 数据集**

实验数据集为：Noor Eye Hospital OCT B‑scan（16,822图像，161患者，554体积）以及无体积信息的OCT‑C8（3类/8类分类任务）；

**📈 对比分析**

与六种主流SSL基线（FixMatch、FreeMatch、CoMatch、FlexMatch、Mean Teacher、Pseudo‑Label）及全监督模型对比。20%标签下TRIAGE实现89.66%准确率、0.0834 UGR，优于所有基线；5%标签下仍保持76.88%准确率、0.1656 UGR；在OCT‑C8 1%/10%标签下分别达到98.00%/95.94%准确率，表现突出；

**⚠️ 局限性**

局限性包括：①依赖专家制定的成本矩阵，未进行系统决策理论校准；②部分标签处理尚未完全独立，影响实验可解释性；③体积上下文验证需要有序B‑scan序列，对无体积数据退化为单扫描模式；④未在多中心临床环境中验证；⑤标签稀缺仍导致性能下降。

---

## 379. Intelligent Detection of Mechanical, Electrical, and Plumbing (MEP) Metrics Based on 2D Floor Plans

**arXiv ID:** 2608.14317 | [PDF](https://arxiv.org/pdf/2608.14317v1)

**作者:** Tarandeep Singh Mandhiratta `[一作]` (Wilfrid Laurier University), Abdul-Rahman Mawlood-Yunis `[通讯]` (Wilfrid Laurier University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 Mask R-CNN 对 2D 楼层平面图中的照明符号进行检测、识别类型并提取相关文本，以实现 MEP 元素的自动识别与能耗预测。

**💡 创新点**

首次将 Mask R-CNN 迁移至 MEP 符号检测领域，结合 PDF 转图、手工标注、COCO 数据格式切片以及后续文本识别，构建了一套完整的自动化工作流。

**🔧 技术方法**

Mask R-CNN、PyMuPDF、VGG Image Annotator、TensorBoard、Pytesseract、MMDetection 等技术。

**📊 数据集**

由赞助方提供的私有 PDF 楼层平面图数据集，转换为图像后手工标注得到 COCO 格式数据。

**📈 对比分析**

在验证集上通过 mAP、mAR 等指标评估，bbox_mAP 达 0.7596、segm_mAP 达 0.7111，且在不同 IoU 阈值下表现优异（bbox_mAP_50=0.9850，segm_mAP_75=0.9219），优于传统手工或规则方法。

**⚠️ 局限性**

仅覆盖照明符号；需要人工标注，符号集对不同项目的适应性有限；对 3D 结构或更复杂的 MEP 符号识别尚未涉及。

---

## 380. Breaking Models to Test the Judge: A Mutation Testing Approach for Semantic Evaluators of Domain Class Diagrams

**arXiv ID:** 2608.14315 | [PDF](https://arxiv.org/pdf/2608.14315v1)

**作者:** Kevin Delcourt `[一作]` (Université de Montréal), Houari Sahraoui `[通讯]` (Université de Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于变异测试的评估框架，用来检验LLM在比较领域类图与文本描述时的语义判断能力。

**💡 创新点**

创新点包括：①设计11种针对类图的语义变异算子，自动生成可检测的缺陷；②采用轻量级语义相似度杀死检测，实现完全自动化评判；③通过与人工评估对比验证了该方法的有效性。

**🔧 技术方法**

使用技术包括：变异测试、LLM（如GPT等）辅助变异、PlantUML模型解析与改写、句子嵌入（sentence transformer）进行相似度匹配，以及阈值判定的杀死策略。

**📊 数据集**

数据集：Golden ModelSet（45对复杂类图+描述）和Easy ModelSet（5对简易类图），共计50个模型–描述对。

**📈 对比分析**

比较方法：对六种LLM+prompt组合分别进行手工精度评估（标注有效/无效）和变异杀死率评估。手工精度最高达63.7%，变异杀死率最高达87%；两种指标相关性为0.624，能够在大多数配置上复现手工评估的排序，证明方法可用于快速预评估。

**⚠️ 局限性**

限制：①评估样本有限，手工标注主观性高；②变异假设基线模型质量足够好，若基线已有缺陷会影响杀死率；③每个变异只引入单一缺陷，未覆盖多缺陷交互；④杀死检测依赖单一句子嵌入阈值，可能对不同语言或提示不够鲁棒；⑤未验证对其他建模语言、DSL或大型模型的可推广性。

---

## 381. One-Sided-Error Parameterized Reductions for the Minimum Distance and Shortest Vector Problems

**arXiv ID:** 2608.14305 | [PDF](https://arxiv.org/pdf/2608.14305v1)

**作者:** Shuichi Hirahara `[一作]` (National Institute of Informatics), Kazuki Ogitsuka `[通讯]` (Graduate University for Advanced Studies, SOKENDAI)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了针对最小距离问题（MDP）和最短向量问题（SVP）的单侧错误随机化归约，并在标准电路下界假设下将其转化为确定性归约，从而得到对所有常数逼近因子的[1]-难度。

**💡 创新点**

创新点在于：①将两侧误差归约升级为单侧误差，保留了对YES实例的高成功概率但完全消除NO实例的误差；②引入 OR 函数与伪随机生成器的组合，利用电路下界假设在单侧误差归约上实现确定性化；③针对参数化版本与非参数化版本均实现了确定性化。

**🔧 技术方法**

主要技术包括：基于 BCH/Reed–Solomon 本地稠密码/晶格构造的稠密 gadget；利用检查块（syndrome 检查）消除“烦人”短向量；随机稀疏化的替代——保留随机性用于寻找合适向量，去除随机化用于消除误差；伪随机生成器与 OR 函数相结合的条件确定性化框架；参数化 OR 函数与非参数化 OR 函数的构造。

**📊 数据集**

无数据集，全部为理论计算复杂度证明。

**📈 对比分析**

通过与先前的两侧误差归约比较，单侧误差归约在保持同等逼近因子下实现更强的确定性化可能性；在确定性下界假设下，得到的确定性[1]-难度与已有的确定性多项式时间归约结果相当，但可扩展到更大逼近因子；性能方面，归约保持了多项式时间与参数化可扩展性。

**⚠️ 局限性**

局限性：仍需依赖电路下界假设；单侧误差归约在某些逼近因子（尤其是SVP 𝓁₂ 中大于 √2 的范围）尚未完全覆盖；对参数化版本的完整逼近范围（如 p>1 时可达的逼近因子）仍有待改进。

---

## 382. Conditional Neural Optimal Transport for Predicting Cellular Phenotypes from Molecular Structure

**arXiv ID:** 2608.14293 | [PDF](https://arxiv.org/pdf/2608.14293v1)

**作者:** Gauthier Avité `[一作]`, Auguste Genovesio `[通讯]` (Ecole Normale Superieure)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一种条件神经最优传输模型，用负控细胞表型和分子结构预测未测细胞表型，提升跨批次表型一致性和对未见分子的预测。

**💡 创新点**

首次将分子条件与神经OT结合并引入Monge‑Gap正则化，形成可在分布层面训练的可推广映射，解决传统OT的转导性限制。

**🔧 技术方法**

核心技术包括DINOv2自监督表型嵌入、多头注意力的分子编码器、条件神经OT框架与Monge‑Gap正则化、以及无偏Sinkhorn重采样。

**📊 数据集**

使用公开的JUMP‑Cell Painting大规模表型数据集（数千实验板、数万条DMSO与活性分子表型）。

**📈 对比分析**

与静态OT、身份映射、DMSO均值、随机等基线进行检索实验，Retrieval@1/10提升显著：在未见分子上取得Plate‑level R@10≈0.095，远超身份映射（≈0.059）和随机（≈0.004），并在跨批次检索中提升mAP至0.65。

**⚠️ 局限性**

主要限制在于分子表征质量（activity cliffs导致低准确率）、未考虑结构极端差异的OOD泛化、模型仅输出嵌入而非可视化图像，且对大规模图谱学习需进一步优化。

---

## 383. Quantum Multi-Armed Bandits and Linear Bandits: Lower Bounds and Algorithms

**arXiv ID:** 2608.14319 | [PDF](https://arxiv.org/pdf/2608.14319v1)

**作者:** Maoli Liu `[一作]` (Chinese University of Hong Kong), John C. S. Lui `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在量子奖励或acles模型下，提出了量子多臂赌博机（QMAB）和量子线性赌博机（QLB）的最优误差下界，并给出一种基于实验设计的消除算法，能够在有限动作集上实现近线性的维度依赖。

**💡 创新点**

创新点包括：①首次证明QMAB的下界为Ω(K log(T/K))，消除了此前对T‑无关上界的疑问；②给出了QLB的下界Ω(d log(T/d))，并设计了低维度、低方差的量子均值估计器与G‑optimal实验设计相结合的算法，维度依赖从先前的d²下降至近线性；③利用多臂测试的Remez型不等式和多项式方法，将单臂测试难度上升到log(1/δ)，实现对全局损失的紧密下界。

**🔧 技术方法**

核心技术包括：多项式方法与Remez不等式对量子测试的度数约束；单臂量子测试转化为多臂上界的归约；低方差量子均值估计器（非破坏性幅度估计）与小支撑G‑optimal设计的配合；设计的整数化与加权预测权重的解析。

**📊 数据集**

本文为理论工作，未使用任何具体实验数据集，所有结果均基于严谨的数学证明与抽象模型。

**📈 对比分析**

相较于先前的O(K log T) QMAB上界与O(d² polylog T) QLB上界，本文的下界与算法实现了近似匹配；在有限动作集且K为多项式d时，算法的期望累计损失为O(d log T polylog d) ，明显优于之前的O(d² polylog T) 上界。

**⚠️ 局限性**

局限性在于：①QMAB下界与上界之间仍保留一个对数因子，尚未完全闭合；②对于通用动作集的QLB，维度上界仍未达到下界级别，最优维度依赖仍是开放问题；③当动作数K为指数级别时，算法的log K 负载无法避免；④实验设计与估计器的高阶常数与对数因子仍可能影响实际性能。

---

## 384. Shift Aware Transfer Learning with Adaptive Dual-Encoder Fusion for PM Forecasting in Data-Limited Environments

**arXiv ID:** 2608.14456 | [PDF](https://arxiv.org/pdf/2608.14456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 385. Participatory Moral AI Is Not Neutral: The Invisible Hand of Developers

**arXiv ID:** 2608.14522 | [PDF](https://arxiv.org/pdf/2608.14522v1)

**作者:** Taenyun Kim `[一作]`, Daniele Quercia `[通讯]` (Nokia Bell Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在肾脏分配、工作代理与已故者生成内容三种AI情境下，特征范围、投票者采样和问题表述三步对道德偏好收集的影响。

**💡 创新点**

将三种开发者决策视为道德规范性选择，提供经验性证据并提出可操作的敏感性审核清单。

**🔧 技术方法**

采用两阶段问卷实验、LLM辅助特征编码、七点量表评估、Moral Foundations Questionnaire、AI Literacy量表，并用Hayes PROCESS 模型进行调节中介分析。

**📊 数据集**

通过Prolific招募的美国样本（N≈809），涵盖肾脏分配、工作代理与已故者生成内容三种情境。

**📈 对比分析**

通过统计比较不同情境下特征相关性、政治意识形态差异与问题表述效应，发现特征重要性差异约30%，意识形态差异约1/3特征，表述可导致1点尺度变化。

**⚠️ 局限性**

仅使用美国样本，缺乏多文化验证；仅采用量表评估而非强制决策；特征阈值为设计选择，可能排除少数观点；未对模型部署进行验证。

---

## 386. Split the Labor: Separating Evidence Interpretation from Decision Aggregation

**arXiv ID:** 2608.14509 | [PDF](https://arxiv.org/pdf/2608.14509v1)

**作者:** Zhelun Wu `[一作]` `[通讯]` (Atlassian), Zhelun Wu (Atlassian)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出将推理与结合拆分为两步：每个信息源单独读取，产生四字段证据元组（假设、可靠性桶、理由、出处），再用明确可审计的规则对元组求和并做决策；同时分析并修正常见的计数-尺度漂移问题，提出使用已校准的对数似然比求和替代权重投票；在此框架下在两种业务情境（已解决与未解决）中分别构造模型并比较效果。

**💡 创新点**

① 证据接口的四字段设计，明确把可靠性和出处纳入元组，解耦推理与聚合；② 发现并数学描述计数-尺度漂移及其对权重投票阈值的影响；③ 给出基于对数似然比的修正算术，适用于多源不确定性集成；④ 在两个不同的任务分解（阅读/组合与序列学习/生存分析）上验证该设计原则。

**🔧 技术方法**

语言模型读取（instruction‑tuned LM），四字段证据元组生成；对数似然比（LLR）求和与块结构依赖折扣；梯度提升树（GBDT）与多输出对数似然损失；GRU编码器与序列化；竞争风险生存模型；阈值、可靠性桶估计（Dawid‑Skene式）。

**📊 数据集**

内部运营数据（≈23,553实例，≈73,638源，12个月时序）；在已解决子集约1,000实例中评估；在未解决子集使用≈69,070快照进行生存预测。

**📈 对比分析**

与内部基线（仅先验、单次拼接、独立无上下文、完整上下文oracle、因果上下文、因果+先验回退）在已解决情境下比较：因果上下文+先验回退在精度与宏F1上表现最佳；在未解决情境下，基线手工特征 0.805 AUPRC，混合模型（GRU+GBDT）达到 0.921 AUPRC。性能提升与基线相比分别约 0.07 及 0.116。

**⚠️ 局限性**

评估样本极小（33实例），无法统计显著性；未检验组合规则的改进；不测量弃权（abstain）对覆盖率的实际影响；特征贡献未被单独隔离；对数似然比校准依赖已估可靠性桶，未证明对稀有标签的稳健性；模型与阈值、桶率等参数为域特定，需重新估算，难以直接迁移。

---

## 387. Twin: Playing an Unknown Game with a Test-Time Digital Twin

**arXiv ID:** 2608.14490 | [PDF](https://arxiv.org/pdf/2608.14490v1)

**作者:** Alexy Skoutnev `[一作]` (Yeshiva University), Iddo Drori `[通讯]` (Yeshiva University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种测试时世界模型推理框架，使用编码代理在交互过程中构建、验证并基于可验证的可执行世界模型进行规划与执行，从而完成ARC-AGI-3游戏。

**💡 创新点**

创新点在于：①将世界模型推理放在测试时完成；②利用可执行Python程序进行精确回放验证；③在无目标信息的环境中通过探索-验证-规划循环主动推断胜利条件；④把模型修复与目标推断分离，形成显式的动态与目标壁垒。

**🔧 技术方法**

核心技术包括：大型语言模型（如GPT‑5.6 Sol）进行程序合成；基于可执行代码的回放验证与修复（CEGIS风格）；基于BFS的模型内规划；启发式候选目标生成与测试；双层验证（动作验证 + 目标假设验证）和回放修复机制。

**📊 数据集**

使用ARC‑AGI‑3公开游戏数据集（25个游戏，共183层），每个游戏是64×64彩色格子迷宫式游戏，规则和胜利条件不公开。

**📈 对比分析**

与OPINE‑World、Prime Agent、EWM等现有系统对比，平均得分93.3/100，清除23/25游戏（179/183层），在人类对标的动作效率上达成61% 以上的平均比例，显著优于基线（如Codex 61.1/100、OPINE‑World 78.4/100）。系统在所有清除的层中，92.9% 的动作是基于已验证模型规划得到的。

**⚠️ 局限性**

局限性：假设世界是确定性且可由Python代码完全描述；对长期依赖历史的动态、隐式状态或连续观测的处理不足；规划与目标搜索受固定搜索预算限制，可能漏掉超远或复杂目标；回放验证只能覆盖已观察的转移，未观测状态的预测可能失效。

---

## 388. Control-Informed Constraint Adaptation in Minimum-Time Trajectory Planning for Autonomous Racing

**arXiv ID:** 2608.14448 | [PDF](https://arxiv.org/pdf/2608.14448v1)

**作者:** Ann-Kathrin Schwehn `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种控制信息反馈的在线轨迹规划框架，利用实时执行误差动态调节可规划空间，以逼近车辆的真实物理极限；

**💡 创新点**

创新点在于将控制层的跟踪误差直接回馈至规划层，形成闭环，能够实时适应轨迹偏差并逐步扩展可行轨迹区域；

**🔧 技术方法**

采用基于模型预测控制（MPC）的时空最优规划、Frenet坐标轨迹表示、点质量车辆动力学模型以及四种基于误差的启发式约束自适应方法；

**📊 数据集**

在高保真仿真环境下验证，使用来自Yas Marina Circuit的真实赛道数据，进行5圈闭环仿真；

**📈 对比分析**

与传统的非自适应基准规划器比较，最优线性启发式（H1）在5圈后将lap时间从116.70s降低至114.91s，减少约1.8s，同时RMSE从0.86m下降至0.41m；运行时间保持在平均25ms以内；

**⚠️ 局限性**

局限性包括依赖手工调参的启发式规则，可能出现过度补偿导致跟踪不稳；仅在仿真中验证，尚未在真实车辆或多车环境中测试；

---

## 389. PACE-Bench: Benchmarking Physics Adaptation via Code Evolution in Dynamic Environments

**arXiv ID:** 2608.14441 | [PDF](https://arxiv.org/pdf/2608.14441v1)

**作者:** Yuhao Zhan `[一作]` (Tsinghua University), Chaojun Xiao `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PACE-Bench，一套包含 144 个源-目标物理适配对的基准，要求智能体在 20 次尝试内利用诊断反馈在物理环境发生突变后对可执行的代码设计进行自我演化。

**💡 创新点**

创新点：①首次针对已成功的可执行设计在环境变化后进行自适应的基准；②系统化隐藏物理参数突变与模拟器支持的反馈；③设计评估协议可同时衡量“知道什么”和“知道如何”两大能力；④对十种自演化方法在六个物理域中的性能与行为进行细粒度分析。

**🔧 技术方法**

技术手段：使用 Box2D 物理模拟、Python 代码生成、诊断沙盒反馈；引入多种自演化范式（Context‑based: Reflexion、Self‑Refine；Memory‑augmented: ACE、ExpeL、ReasoningBank；Inference‑time Search: Tree‑of‑Thoughts、CodeEvolve；Parameter‑based: SEAL、RAGEN、TTT‑Discover）；采用 LLM（Qwen3-4B/8B/14B、GPT‑5.5 等）生成与修正代码；进行代码相似度、错误分类与成本归一化评估。

**📊 数据集**

数据集：PACE-Bench 包含 36 个基准任务，覆盖六个物理域，每个任务在 5 个环境（源 + 4 个突变阶段）中产生 144 个源‑目标对；总计 180 个评估环境、37,860 个提示词、273 条硬约束、292 条原语 API、945 条参数突变。

**📈 对比分析**

比较与性能：对 11 种方法（含 Vanilla）在 Pass@2 和 Score@2 上进行对比。最佳结果为 Reflexion + Qwen3‑14B（35.9% Pass@2），GPT‑5.5 在 Statics 子集上达 66.7%；整体基准尚未饱和；自演化方法相对 Vanilla 增益有限；Reflexion 在所有范式中表现最佳；Tree‑of‑Thoughts 在成本归一化上最有效；模型增大后收益递减；披露物理参数不提升性能上限；视频反馈效果呈现双向偏差。

**⚠️ 局限性**

局限性：仅在 2D Box2D 模拟器中测试，难以推广至 3D 或真实机器人；20 次尝试的预算限制了持续适应的研究；前沿模型、参数披露与视频反馈的实验覆盖有限；高计算成本限制了实验规模。

---

## 390. Style or Signature? Artist-Disjoint Evaluation of Style Classification in Frozen Vision Embeddings

**arXiv ID:** 2608.14435 | [PDF](https://arxiv.org/pdf/2608.14435v1)

**作者:** Rory Ashton `[一作]` `[通讯]` (University of Edinburgh), Rory Ashton (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在冻结的视觉嵌入上评估绘画风格分类，提出了艺术家分离的评估协议，并通过5-近邻探讨不同运动在排除同一艺术家时的分类稳健性。

**💡 创新点**

首次在冻结嵌入的风格分类任务中引入“留一艺术家”分离评估，揭示随机拆分掩盖的艺术家识别对结果的影响，并系统比较不同运动和编码器的表现差异。

**🔧 技术方法**

使用CLIP、DINOv2、ResNet-50等预训练图像编码器提取特征，采用L2归一化后基于余弦相似度的5-近邻分类，以及艺术家识别的对照实验。

**📊 数据集**

构建了包含320幅画作、四个二十世纪艺术运动（印象派、立体派、抽象表现主义、超现实主义），每个运动八位艺术家、每位艺术家十幅作品的平衡数据集。

**📈 对比分析**

在全池和艺术家分离两种划分下比较5-近邻分类准确率，发现整体准确率从0.869下降到0.766，下降幅度对运动差异显著，印象派和立体派几乎不变，而超现实主义下降20点，表明其对艺术家识别的高度依赖；该模式在四种编码器上保持一致。

**⚠️ 局限性**

受限于样本量小、作品主题不匹配、标签来源为众包、未进行内容控制或微调，且仅评估冻结特征与简单最近邻的表现，未探究更复杂模型或文本编码器的贡献。

---

## 391. PriCoRec: A Privacy-Aware Cloud-Device Collaborative Framework for Ad Recommendation under Feature Constraints

**arXiv ID:** 2608.14429 | [PDF](https://arxiv.org/pdf/2608.14429v1)

**作者:** Dairui Liu `[一作]` (University College Dublin), Ruihai Dong `[通讯]` (University College Dublin)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种云端预排序与设备端排序相结合的广告推荐框架 PriCoRec，能够在保持用户敏感特征在设备端的同时，提升整体CTR预测性能。

**💡 创新点**

创新点包括：① 在云端预排序阶段引入 DPP‑启发的多样性正则化，弥补云端特征不足导致的候选质量下降；② 在设备端使用云端预排序输出的 logits 作为辅助特征，采用云引导的轻量化模型训练，兼顾隐私与低延迟。

**🔧 技术方法**

核心技术包括：云端预排序 PNN、DPP 多样性正则化、云引导辅助学习；设备端轻量化 PNN、嵌入维度压缩、FP16 存储与词表映射；统一的云端/设备端特征分离框架。

**📊 数据集**

实验使用三大公开工业数据集：OpenMCC、TaobaoAd、Ali‑CCP，涵盖数百万用户与数千万交互记录。

**📈 对比分析**

与基线（DP‑SGD、DualRec、FedCAR、FedCIA、Popularity）对比，PriCoRec 在 gAUC、Recall@100、Recall@10、nDCG@10 等指标均取得最高值，尤其在设备端排序阶段显著提升了 R@1 与 RR。

**⚠️ 局限性**

局限性：① 仍受限于设备端计算与存储资源，对模型规模与性能存在折衷；② 多样性正则化参数需在验证集上调优，可能对不同业务场景泛化有限；③ 目前未覆盖冷启动、长尾推荐与公平性等实际业务挑战。

---

## 392. Estimating the growth in emissions from AI data centres

**arXiv ID:** 2608.14421 | [PDF](https://arxiv.org/pdf/2608.14421v1)

**作者:** Wim Vanderbauwhede `[一作]` `[通讯]` (University of Glasgow), Wim Vanderbauwhede (University of Glasgow)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用生命周期评估（LCA）方法构建了AI数据中心的运营与内置碳排放模型，并在IEA与McKinsey提供的扩张情景下预测未来CO₂排放。

**💡 创新点**

创新点在于将服务器制造的内置碳排放随技术迭代而演变的趋势动态纳入LCA模型，并通过非线性最小二乘拟合和蒙特卡洛采样获得不确定性区间。

**🔧 技术方法**

使用了生命周期评估框架、Koomey/摩尔定律趋势拟合、Levenberg-Marquardt最小二乘法、蒙特卡洛采样、Python/NumPy/SciPy等计算工具。

**📊 数据集**

数据集包括NVIDIA DGX服务器的规格表（2013–2025）、硬件厂商的制造与碳排放报告、ACT模型参数、行业碳排放数据库以及IEA和McKinsey的电力与容量扩张情景。

**📈 对比分析**

通过对比IEA Lift-Off、IEA Lift-Up、McKinsey Upper-range等情景，评估运营与内置排放比例；结果表明即使在保守情景下年排放也可达1 GtCO₂e，恶劣情景可达3 Gt/yr，说明AI数据中心排放可能逼近甚至超过全球碳预算。

**⚠️ 局限性**

局限性包括：模型仅覆盖数据中心本身，未考虑网络、电力基础设施及地区差异；内置排放趋势假设为指数增长，忽略可能的技术突破；依赖公开硬件数据，可能存在缺失或偏差。

---

## 393. LP-NAS: Linear Programming-based Neural Architecture Search

**arXiv ID:** 2608.14472 | [PDF](https://arxiv.org/pdf/2608.14472v1)

**作者:** Abhishek Shukla `[一作]` (IIT Kanpur), Faiz Hamid `[通讯]` (IIT Kanpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于线性规划的可微神经架构搜索框架 LP‑NAS，并在 DARTS 搜索空间上实现了 S‑LP‑DARTS 与 R‑LP‑DARTS 两个高效变体。

**💡 创新点**

创新点在于通过在一次 LP 约束中同时利用验证损失梯度和训练损失 Hessian，求解得到的下降方向能保证在更新架构时保持权重的下层最优性，从而避免传统 DARTS 中的过拟合与不稳定性。

**🔧 技术方法**

核心技术包括：二阶信息（Hessian）近似、L‑BFGS Hessian 近似、线性规划求解、参数子集选择（S‑LP 与 R‑LP）以及可微架构松弛。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 及 ImageNet32（ImageNet 归一化）上进行架构搜索与评估。

**📈 对比分析**

与标准 DARTS 以及 P‑DARTS、PC‑DARTS、STO‑DARTS 等变体比较，S‑LP‑DARTS 与 R‑LP‑DARTS 在验证准确率、测试准确率、参数量与搜索/评估时间均优于对照组，且在 ImageNet 上表现出良好的迁移能力。

**⚠️ 局限性**

主要局限在于 Hessian 计算与 LP 求解的计算与内存开销，需通过稀疏或子集近似降低成本；此外方法仅适用于可微的连续搜索空间，对离散/硬件约束搜索仍有待扩展。

---

## 394. Handover of In-Context Learning State Across Session Boundaries

**arXiv ID:** 2608.14528 | [PDF](https://arxiv.org/pdf/2608.14528v1)

**作者:** Masahiro Kato `[一作]` (Mizuho-DL Financial Technology, University of Tokyo, RIKEN AIP, and Osaka Metropolitan University), Taka Kato `[通讯]` (NP-hard)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `2704f255-0c84-4173-b83c-0e9a3dbea232` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大语言模型应用中会话交接的理论与方法，提出基于预测充分性的手写记录设计，并给出记忆预算与风险的定量关系。

**💡 创新点**

创新点在于将交接视为任务相关的预测充分性问题，定义最粗预测充分状态，给出固定长度比特需求，并提出三分记录法结合任务约束与统计压缩。

**🔧 技术方法**

采用统计充分性理论、信息论（互信息、Blackwell比较）、高斯线性回归和非参数回归分析，推导记忆-风险下界与上界。

**📊 数据集**

使用合成回归数据（高斯线性模型）和非参数仿真数据（β-Holder光滑函数）来验证理论。

**📈 对比分析**

通过对比自包含与外部记录、精确与近似充分记录，在固定后续过程下计算理想解风险，实验显示有限位数下的预测误差可按理论曲线逼近。

**⚠️ 局限性**

局限在于仅考虑会话边界且假设后续输入外生，缺乏对自适应轨迹和真实多模态应用的验证；实际编码与解析实现细节仍待实证。

---

## 395. Generating Benchmark Health Data Using a Tabular Diffusion Transformer

**arXiv ID:** 2608.14496 | [PDF](https://arxiv.org/pdf/2608.14496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 396. Validating LLM-Modernized Scientific Software Through Differential Fault Injection

**arXiv ID:** 2608.14527 | [PDF](https://arxiv.org/pdf/2608.14527v1)

**作者:** Evan Coleman `[一作]` (University of Mary Washington), Peng Xu `[通讯]` (Iowa State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个差分故障注入平台，用于在共享的SCF驱动代码中注入相同的、可复现的数值扰动，比较原始Fortran实现与LLM现代化后的实现在离常规运行之外的行为差异；并通过实验验证现代化过程是否保留原实现的离常规鲁棒性，同时利用该平台进行针对性硬化改进的评估。

**💡 创新点**

创新点包括：①提出“差分故障注入”这一验证范式，利用共享驱动插桩实现相同注入位置，消除实现差异带来的对比噪声；②构建了面向SCF迭代的多点注入框架（12个注入点、10种扰动模式），可细粒度探测瞬态、持久、精度和并行相关的错误；③将该框架与LLM代码生成流程结合，形成测量‑定位‑修改‑再测量的闭环，首次在AI辅助科学软件现代化中展示了基于实验的鲁棒性改进流程。

**🔧 技术方法**

主要技术包括：差分故障注入机制（基于anchor‑based插桩、可重播的随机数生成器）、自定义故障模型（相对/绝对噪声、位翻转、截断、特殊值、延迟）、SCF驱动的多点注入、并行死锁检测与同步广播修复、以及对比分析脚本（分类、迭代计数、能量误差、注入计数）。

**📊 数据集**

数据集主要是两种小分子体系：CH₂（STO‑2G基组）和甲醇（6‑31G基组），在两种收敛配置（生产模式与裸Roothaan）下进行实验；共计超过2,200次故障注入实验，包括200对原始/现代化实现的配对注入、40对硬化后的对比以及其他持久、并行和精度实验。

**📈 对比分析**

比较方法：对每次注入运行记录注入位置、时序、数值变化，按预设分类（吸收、延迟、SDC、崩溃、挂起）进行统计；对配对运行进行逐对比对，确保状态分类、迭代次数、注入计数和最终能量完全一致。实验表明，原始与现代化实现在200对注入中保持完全一致，硬化后的实现在40对注入中亦无差异；在不同收敛配置下，实验验证了收敛加速器对错误吸收的影响，并通过并行同步修改消除了死锁。整体性能：注入工具占用源文件改动极小（15行），实验可在单节点、单核/多核环境下完成，耗时与无注入跑相当。

**⚠️ 局限性**

限制包括：①实验仅覆盖两种闭壳分子和两种收敛配置，未验证在更大系统、多体积或多核/多节点规模上的可推广性；②故障模型为应用层代理，未直接模拟硬件级错误；③并行实验仅在4个进程下进行，无法覆盖更大并行性导致的同步问题；④注入点固定在SCF驱动，未覆盖其他模块（如UHF、DFT、静态数据）或其他代码路径；⑤对比仅针对单一LLM现代化实例，未检验不同LLM模型或不同翻译策略的差异。

---

## 397. Learning-to-Transition for Large-scale and High-Order MIMO Detection

**arXiv ID:** 2608.14511 | [PDF](https://arxiv.org/pdf/2608.14511v1)

**作者:** Yubo Zhang `[一作]`, Xiaodong Wang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于学习转移（L2T）的完整向量迁移框架，用Transformer在大规模高阶MIMO检测中递归更新候选向量并结合块级自回归采样；

**💡 创新点**

创新点在于将完整向量迁移视作序列决策，使用通道耦合Transformer实现跨向量状态的动态嵌入更新，并通过残差‑到‑BER 训练历程和硬-软迁移策略实现从硬判决到迭代软判决的无缝过渡；

**🔧 技术方法**

技术包括Transformer‑基注意力网络、块级自回归采样、策略梯度强化学习、残差‑到‑BER 训练计划、硬软迁移（tied‑to‑untied）、先验倾斜采样、Rao‑Blackwell化终端混合估计以及多阶段软训练（合成先验→decoder‑in‑loop）；

**📊 数据集**

实验使用Rayleigh衰落下的64-QAM与256-QAM数据，配合5G NR LDPC码和Sionna链路仿真，所有数据均在统一的随机种子下生成；

**📈 对比分析**

与LMMSE、QR‑K‑best、OAMP‑Net2、RE‑MIMO、SGT等基线比较，硬判决时接近K‑best性能，软IDR时BLER在3轮IDR后优于MMSE‑PIC、K‑best list‑MAP、SGT和DUIDD，显著降低误码率并提升GMI；

**⚠️ 局限性**

局限性包括对完美CSI假设、对训练数据分布的依赖、模型参数量大导致部署延迟、在极端信道条件下的泛化不足，以及残差‑到‑BER 学习过程对超参数敏感。

---

## 398. Rollplex: Cross-Phase GPU Spatial Sharing for Vision Language Model Post-Training

**arXiv ID:** 2608.14498 | [PDF](https://arxiv.org/pdf/2608.14498v1)

**作者:** Hanfeng Lu `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种能够在VLM强化学习过程中将前缀计算与rollout解码并行执行的运行时系统。

**💡 创新点**

创新点在于：①引入阶段感知的内存管理，让KV缓存、梯度等仅在其生命周期内驻留HBM；②设计并行感知的权重量化共享机制，使训练和rollout可以在不同的张量并行度下共享同一份模型权重，从而避免内存浪费和复制开销。

**🔧 技术方法**

主要技术包括 CUDA-VMM 以实现虚拟地址稳定性、CUDA IPC 与 MPS 进行进程间共享、分块优化的Adam更新、以及针对张量并行度差异的权重量化共享策略。

**📊 数据集**

使用 Qwen2.5‑VL‑32B 语言视觉模型，并在四个视频推理数据集上评估：CLEVRER、PerceptionTest、LLaVA‑Video‑178K 与 STAR。

**📈 对比分析**

与传统的同机串行（Colocate）和分机异步（Disagg）部署对比，Rollplex 在同等 GPU 预算下实现了 1.23×–1.30× 的步骤时间加速（相较于 Colocate）和 1.57×–2.24×（相较于 Disagg），且训练得到的奖励与基线保持一致。

**⚠️ 局限性**

局限性包括：①依赖 MPS 进行 GPU 共享，缺乏独立的错误恢复和故障隔离；②适用场景需要前缀工作量大且与生成结果无关；③对极小的前缀或对内存预算极其紧张的模型，收益有限。

---

## 399. Lossy Compression via Sparse Regression Codes: Generalized Construction and Finite-length Bounds

**arXiv ID:** 2608.14494 | [PDF](https://arxiv.org/pdf/2608.14494v1)

**作者:** Galen Reeves `[一作]` (Duke University), Ramji Venkataramanan `[通讯]` (University of Cambridge)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对稀疏回归码（SPARCs）在失真率压缩中的性能进行研究，提出一种广义的可加正交回归码框架，涵盖标准SPARCs、符号SPARCs以及K‑稀疏SPARCs，并给出了在两种贪婪编码规则（相关性编码与距离编码）下的非渐近失真上界与中心预测。

**💡 创新点**

创新点在于：1) 将SPARCs推广为可加正交回归码，统一描述并显式刻画了权重分配对失真的影响；2) 提供了关于残差演化的确定性预测与浓缩定理，得到与源能量相关的闭式失真上界；3) 推导出针对每种编码规则的最优/近优功率分配，并证明其满足总功率约束；4) 在理论与实验中揭示符号SPARCs几乎等效于标准SPARCs但可减半设计矩阵大小，距离编码在高码率时优于相关编码。

**🔧 技术方法**

主要技术包括：稀疏回归码设计、可加正交分解、贪婪编码策略（相关性与距离），以及对残差的高斯子高斯浓缩分析；利用高维概率与极大值理论计算Gaussian宽度（ω_l）和距离参数（γ_l），从而得到失真递推式与功率分配公式。

**📊 数据集**

使用的数据集为：i.i.d. 标准高斯源（方差σ²=1）以及相同维度的标准高斯设计矩阵A；实验采用10次 Monte‑Carlo 采样，每个码率点计算平均失真与标准差。

**📈 对比分析**

比较方法：在理论上给出失真预测曲线（Δ_L）并与实验平均失真对比；同时将不同SPARC变体（标准、符号、2‑稀疏）与距离/相关编码的最佳功率分配结果进行对比，结果表明符号SPARCs在相同码率下失真与标准SPARCs相当但矩阵尺寸减半；距离编码在R>≈1.5 bits/symbol时优于相关编码；所有变体的实验失真均紧贴理论预测且接近 Shannon 最优失真曲线。

**⚠️ 局限性**

局限性包括：1) 只在高斯源与高斯设计矩阵下验证，未证明对非高斯或非独立源的适用性；2) 仅分析贪婪编码策略，对更高效但计算量大的编码器（如 AMP）未给出理论保证；3) 对距离编码的非渐近误差界不如相关编码精确；4) 仅关注可加正交结构，无法涵盖更一般的稀疏回归码设计。

---

## 400. The Dynamics of Intelligence Explosions

**arXiv ID:** 2608.14426 | [PDF](https://arxiv.org/pdf/2608.14426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 401. Isomorphism of tournaments with bounded VC dimension

**arXiv ID:** 2608.14486 | [PDF](https://arxiv.org/pdf/2608.14486v1)

**作者:** Simon Raßmann `[一作]` (Technische Universität Darmstadt), Pascal Schweitzer `[通讯]` (Technische Universität Darmstadt)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文解决了有界VC维的锦标赛同构问题，证明了对于VC维为d的锦标赛，其同构问题可以在时间n^O(dlog d)内决定。

**💡 创新点**

创新点在于提出了一种新的方法来进行锦标赛的同构不变分解，并引入了修补锦标赛的概念，分析了修补锦标赛中的有界VC维。

**🔧 技术方法**

使用了递归算法和组合数学中的一些技术，特别是与近双胞胎的结构相关的技术。

**📊 数据集**

使用了锦标赛的理论和VC维的相关性质，具体数据集未明确提及，但涉及到锦标赛的性质和结构。

**📈 对比分析**

与现有方法相比，本文的方法在处理有界VC维的锦标赛同构问题时，能够在多项式时间内计算同构群，性能显著提升。

**⚠️ 局限性**

限制在于对于某些特殊类型的锦标赛，仍然缺乏有效的多项式时间算法，尤其是在处理更复杂的锦标赛结构时。

---

## 402. Information Satisfaction: A Reader-Centered Axis for Summarization Evaluation

**arXiv ID:** 2608.14457 | [PDF](https://arxiv.org/pdf/2608.14457v1)

**作者:** Isabel Cachola `[一作]` (St Edward's University), Mark Dredze `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了“信息满足度（information satisfaction）”这一以用户为中心的摘要评价维度；

**💡 创新点**

创新点在于将读者的角色与查询需求结合为评估目标，并对现有评价指标在信息满足度上的敏感性进行系统测试；

**🔧 技术方法**

采用了传统的 ROUGE、BERTScore、SummaQA 等指标，以及基于大语言模型的评估（FActScore、LLM-as-judge 等）进行量化对比；

**📊 数据集**

使用了 arXiv、PubMed、SciTLDR、eLife 四个科学摘要语料库以及由实验者自建的 Persona-Query 评测集；

**📈 对比分析**

通过扰动实验（加入干扰句、逐句增删、长度变换、受众改写）和人类专家评测，发现绝大多数指标对信息满足度不敏感，且与人工偏好几乎无相关性；

**⚠️ 局限性**

局限在于缺乏能真实捕捉读者信息需求重要性与先验知识的评估模型，现有指标对受众定制化改写的响应不佳，且 Persona 相关评估在鲁棒性和人类一致性上表现不佳。

---

## 403. GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure

**arXiv ID:** 2608.14428 | [PDF](https://arxiv.org/pdf/2608.14428v1)

**作者:** Mohamed Abdelsamad `[一作]` (Bosch Center for AI), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 GhostPoint，一种自监督学习框架，通过在检测实例周围生成并学习“幽灵”体素，弥补 LiDAR 点云中被遮挡和无返回区域的表面偏差。

**💡 创新点**

创新点在于：①利用实例体素膨胀生成局部邻域体素；②引入预测器对非可见体素进行监督，将老师预测的“幻觉”特征与学生预测对齐，从而让表示学习跨越可观测表面，提升 3D 检测的泛化；③在无标注空间中实现目标导向的自监督。

**🔧 技术方法**

技术包括：教师–学生自蒸馏框架、体素化与距离加权 KNN 初始化、预测器（轻量化 Transformer 块）进行邻域特征推理、Softmap 与中心偏移两种监督信号、掩码与无返回体素的非可见监督。

**📊 数据集**

使用公开的 nuScenes 与 Waymo LiDAR 3D 检测数据集进行预训练与下游评估，并在不同标签稀疏程度下测试 label‑efficiency。

**📈 对比分析**

与现有自监督方法（如 DOS、PointINS、Occ‑MAE 等）相比，GhostPoint 在 decoder‑probe 评估中提升 mAP 约 2.8–2.5 分，在全量微调下突破有监督基线 3.7 分；在稀疏扫描、少标签和跨数据集迁移上也保持显著优势。

**⚠️ 局限性**

局限性：①在仅解码器探测（probe）下仍落后于完全监督训练；②未在其他稀疏感知模态（如雷达）中验证；③对大规模实例稀疏时的幻觉质量仍易受背景噪声影响。

---

## 404. Knowing When to Stop: Bayesian Optimal Stopping for LLM Evaluations

**arXiv ID:** 2608.14425 | [PDF](https://arxiv.org/pdf/2608.14425v1)

**作者:** Toby D. Pilditch `[一作]` `[通讯]` (UK AI Security Institute), Toby D. Pilditch (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于贝叶斯可信区间宽度的自适应停止框架，用于大语言模型评估过程中在收集足够数据后自动决定何时停止采样，从而显著降低计算成本。

**💡 创新点**

核心创新在于将顺序测量与层次贝叶斯推断结合，设计了适用于二元、序数和连续评分的停止准则，并引入低表现保守性调整和双路径停止机制，实现无需预先校准题库的灵活评估。

**🔧 技术方法**

使用了层次贝叶斯模型、置信区间宽度阈值判定、滑动窗口稳定性检测、保守性因子扩展以及多路径（模态/熵）判定等技术。

**📊 数据集**

实验以200项、10轮的评估框架为基础，涉及 BIG‑bench、MATH、GPQA、WritingBench 等数据集，并使用 Claude Sonnet 4.5、GPT‑4o、GPT‑3.5 Turbo 等模型。

**📈 对比分析**

通过3×3矩阵实验（不同推断路径与性能水平），与完整采样对比，效率提升57%–97%，平均绝对误差<0.02，贝叶斯等价检验显示截断不产生显著偏差，且保持模型排名不变。

**⚠️ 局限性**

局限性包括对观测可交换性和随机呈现顺序的依赖、极端表现下置信区间覆盖率可能不足，以及仅在单一实验配置下验证，需在更广泛的评估场景进一步测试。

---

## 405. Nodal discontinuous Galerkin methods for non-ideal equations of state: pressure equilibrium preservation and entropy correction

**arXiv ID:** 2608.14506 | [PDF](https://arxiv.org/pdf/2608.14506v1)

**作者:** Jesse CHan `[一作]`, Ayaboe Edoh `[通讯]` (Amentum - U.S. Air Force Research Laboratory)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究高阶离散Galerkin方法在非理想气体（van der Waals、Peng–Robinson）下的压力平衡保持与熵修正技术，以提升数值稳健性与精度。

**💡 创新点**

提出基于密度平均校正的EPEC通量，并将其与APEC、局部拉克斯‑弗里德里希斯惩罚及最小耗散熵修正FCT相结合，实现既保持压力平衡又具低耗散的高阶DG方案。

**🔧 技术方法**

采用节点对差分DG框架，结合密度平均校正的EPEC、APEC通量，局部拉克斯‑弗里德里希斯界面惩罚，以及熵修正FCT（最小耗散校正），并在Julia的Trixi.jl库中实现。

**📊 数据集**

使用van der Waals与Peng–Robinson等价流体方程参数（氮气），在周期域、转临界冲击管、混合层和射流注射等数值实验中进行测试。

**📈 对比分析**

通过与中心、Coppola等人提出的EPEC以及APEC通量的比较，评估CFL、压强误差、稳定性和熵修正系数θ；结果表明校正式EPEC和APEC在高阶时可允许更大CFL、压强误差降至10⁻⁸～10⁻¹³，熵修正显著提升长期模拟的稳健性。

**⚠️ 局限性**

EPEC通量对非理想EOS敏感，需较小CFL；界面惩罚引入压力误差；熵修正虽耗散低但仍产生微量压力误差；缺乏满足非理想EOS的正定性/正压保持限制器，且在更高阶/高密度情形下的鲁棒性仍待进一步验证。

---

## 406. Visualizing Uncertainty in Non-linear Projections with Ensembles

**arXiv ID:** 2608.14513 | [PDF](https://arxiv.org/pdf/2608.14513v1)

**作者:** Kai Nylund `[一作]` (Northeastern University), Lace Padilla `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了非线性降维（NLDR）方法的随机性问题，提出通过多次运行的投影集合（ensemble）并采用中位数、噪声扰动、子采样等策略来构建更稳健的投影，随后评估这些集成投影与单个投影在质量指标上的表现，并展示了五种可视化投影集合的方案。

**💡 创新点**

创新点在于将集成学习与投影不确定性可视化相结合：①使用中位数投影作为共识投影；②在投影前对输入数据添加噪声或子采样以减轻过拟合噪声的稳定性；③提出并比较五种针对投影集合的可视化方法，帮助用户识别投影的随机性与稳定性。

**🔧 技术方法**

主要技术包括：UMLP降维（使用默认超参数）、Procrustes对齐、投影集成（均值、中位数、三分位、中点几何中值）、扰动策略（噪声添加与子采样）、投影质量评估指标（局部、聚类、全局度量）、以及可视化技术（中位数散点图、小倍数图、置信椭圆、运动线、二维核密度+分箱）。

**📊 数据集**

实验基于31个公开数据集，规模从数千到数十万点不等；在每个数据集上生成1000或100次UMLP投影，并构造投影集合。所使用的数据集主要为常见的高维数据集（如MNIST、HAR等）以及人工生成的几何测试集。

**📈 对比分析**

通过ZADU库实现的投影质量指标，对比集成投影与单个投影的排名显示，中位数投影在局部、聚类和全局指标上与单个投影相当，偶尔甚至略优；在噪声扰动下，投影集成可以在全局度量上提升，同时在局部度量上略有下降，表明在抑制噪声过拟合与保持局部结构之间存在权衡。

**⚠️ 局限性**

局限性包括：仅针对UMLP（未涉及t‑SNE、PCA等其他NLDR方法）；使用默认超参数，缺乏针对不同数据集的参数调优；扰动策略的参数选择仍经验化；评估指标集中在传统质量度量，对实际可解释性和业务价值的影响尚未深入研究；以及可视化方法在大规模数据时可能面临渲染与交互的性能瓶颈。

---

## 407. Memory Allocation for Constant-Bounded Programs

**arXiv ID:** 2608.14471 | [PDF](https://arxiv.org/pdf/2608.14471v1)

**作者:** Vinícius Silva `[一作]` (Federal University of Minas Gerais), Márcio Costa e Fernando Magno Quintão Pereira `[通讯]` (Federal University of Minas Gerais)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种针对常数界定程序的静态内存分配器，利用树形扫描与碎片整理实现接近最优的栈空间使用。

**💡 创新点**

创新点在于将控制流图完全展开为树，结合可搬移性（defragmentation）实现多大小变量的最优分配，并证明在所有路径上达到最大活变量和，从而给出多项式时间近似。

**🔧 技术方法**

使用的技术包括循环展开、树形化、SSA、活区最后使用分析、树形扫描分配、动态存储分配与碎片整理等。

**📊 数据集**

实验数据集包括 24 个 eBPF 实际与合成程序、BenchGen 生成的 MLIR 随机程序以及 Honey Potion 编译出的 eBPF 程序。

**📈 对比分析**

与 Trivial、Tree-Scan、Defrag 三种分配器对比，评估栈空间、代码大小、分配时间等指标，结果显示 Defrag 能把栈空间压缩到约 MaxLive+最大变量大小，eBPF 栈空间降低 90% 以上，分配时间保持线性并略高于 Tree-Scan。

**⚠️ 局限性**

局限性包括：代码膨胀可能指数增长、碎片整理虽然稀少但仍需插入指令，且仅适用于常数界定程序，无法直接应用于非常数界定程序。

---

## 408. Expected Free Energy-based Informative Path Planning for Robotic Mars Exploration

**arXiv ID:** 2608.14466 | [PDF](https://arxiv.org/pdf/2608.14466v1)

**作者:** Ajith Anil Meera `[一作]` (Eindhoven University of Technology), Wouter Kouw `[通讯]` (Eindhoven University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于期望自由能（EFE）的预算感知连续路径规划框架，用于机器人在有限移动预算下同时构建信息地图并定位价值最高区域。

**💡 创新点**

①将EFE统一为探索-利用目标，避免手工加权；②引入预算感知温度调度，自适应平衡探索与利用；③使用幻想路径在规划时前瞻性更新GP不确定性；④采用差分进化解决连续轨迹的高维非凸优化。

**🔧 技术方法**

期望自由能理论、Gaussian Process地图建模、幻想路径评估、差分进化（DE）优化、递归滚动预测（MPC）与仿真环境（PyBullet + Mars高度场）。

**📊 数据集**

20 m×20 m 2D 隐藏高斯峰场（合成）以及128×128 的Mars表面高度/反照率贴图模拟，配合起始Sobol点。

**📈 对比分析**

与EI、UCB、MI、VAR、Coverage基准在20个随机种子下比较，aEFE 在地图RMSE和简单回退两项指标上均优于所有方法，EI、UCB 在优化上优但地图差，MI、VAR 在学习上优但未优化；aEFE 通过预算感知温度调度在后期进一步提升性能。

**⚠️ 局限性**

仅验证单机器人二维连续场；未验证多机器人共享信念与真实物理平台的能源/行进约束；对高噪声或地形不平坦的鲁棒性待进一步评估。

---

## 409. THRIVE: Therapeutic Humanoid Robot In Virtual Environment

**arXiv ID:** 2608.14462 | [PDF](https://arxiv.org/pdf/2608.14462v1)

**作者:** Jin Xu `[一作]` (Georgia Institute of Technology), Ayanna Howard `[通讯]` (Ohio State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了面向脑性瘫痪儿童的家庭康复平台THRIVE，集成VR上肢康复游戏、实时摄像头运动跟踪和社会化机器人治疗师，可通过物理机器人、远程存在机器人或虚拟代理交互；

**💡 创新点**

核心创新在于机器人不可知（robot‑agnostic）设计，游戏与机器人实体解耦，提供低成本、可扩展的家用康复方案；利用大型语言模型生成多样化的口头反馈，结合视觉和非语言提示提升交互自然度；

**🔧 技术方法**

技术包括：基于Azure Kinect或同类摄像头的姿态跟踪；Unity/游戏引擎开发VR康复游戏；7-DOF 3D打印机器人与Raspberry Pi 5驱动；文本到语音（TTS）与GPT‑4生成对话脚本；状态机驱动远程存在模式；虚拟代理在游戏内渲染；

**📊 数据集**

未使用公开数据集，系统通过实时摄像头采集儿童运动数据并在游戏中即时分析；

**📈 对比分析**

论文未给出实验对比或性能评估，未来计划在儿童群体中对物理机器人与远程/虚拟机器人配置进行参与度、依从性和上肢功能改进的比较；

**⚠️ 局限性**

局限性包括：缺乏临床实验验证，未评估不同机器人形态对疗效的实际影响；系统依赖高质量摄像头与网络；样本范围有限；可能存在隐私与数据安全风险；

---

## 410. Wyvern: An Agentic Framework for Generating Grounded Multimodal Reports

**arXiv ID:** 2608.14446 | [PDF](https://arxiv.org/pdf/2608.14446v1)

**作者:** Beatrice Alessandra Motetti `[一作]` (Politecnico di Torino), Lukas Cavigelli `[通讯]` (Huawei Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了Wyvern多智能体框架，用于自动生成基于检索证据的多模态技术报告。

**💡 创新点**

创新点包括图像检索与位置插入模块、声明自动修订以提升信息根植性，以及对原子声明的分解与验证。

**🔧 技术方法**

采用了大型语言模型（DeepSeek-R1/V3）、检索增强生成、图像/表格处理、链式推理等技术。

**📊 数据集**

使用公开网络检索结果和学术文献，构建参考文档集合；评测使用27个主题的人工评审报告。

**📈 对比分析**

与STORM、WebThinker、WikiAutoGen比较，Wyvern在图像信息量、报告可用性、引文召回/精确度上分别提升约2.3×、100%/87.5%等，整体表现更好。

**⚠️ 局限性**

局限包括对搜索API的依赖导致检索覆盖不完整、可复现性差、仅支持英文、未解决偏见与错误信息过滤。

---

## 411. More Correct Mass, Worse Answers: Why Power Sampling Can Fail and How to Fix It

**arXiv ID:** 2608.14420 | [PDF](https://arxiv.org/pdf/2608.14420v1)

**作者:** Haohui Yang `[一作]` (Peking University), Xiujun Ma `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在多样本推理中使用Power Sampling时可能出现的覆盖与剂量匹配不一致问题，并提出了相对排名SoftSat方法来修复此类误差

**💡 创新点**

创新点在于首次系统识别并量化了覆盖匹配与剂量匹配导致的性能下降，并通过将绝对对数似然转换为相对排名并对高概率路径进行饱和来实现分布的自适应校正

**🔧 技术方法**

采用了Power Sampling、SoftSat、重要性加权一致性（Weighted Consensus）以及Self‑Consistency等技术，并在同一Base候选池上实现了无额外采样成本的加权推理

**📊 数据集**

实验数据集包括BigCodeBench、LiveAoPSBench以及PHYSICS，覆盖代码、数学与物理推理任务

**📈 对比分析**

与Uniform和传统Power（α=4）基线比较，SoftSat在大多数模型与基准上成功消除了Power造成的显著回退，并在部分设置中保持或略微提升准确率，回退幅度从最大18.5个百分点下降至1.0个百分点以内

**⚠️ 局限性**

局限在于SoftSat需要手动调节参数（如β、m、η、κ），在某些任务和模型上仍可能不如Uniform表现，且对极端概率分布的处理仍不够完善

---

## 412. STINER: Automated Extraction of Strategic Cyber Threat Intelligence from X

**arXiv ID:** 2608.14418 | [PDF](https://arxiv.org/pdf/2608.14418v1)

**作者:** Yasir Ech-Chammakhy `[一作]` (Mohammed VI Polytechnic University), Anas Motii `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个面向社交媒体的战略网络威胁情报(NER)数据集 STINER，并基于该数据集训练了高精度的提取模型。

**💡 创新点**

创新点在于：①首次构建了聚焦战略层面的社交媒体CTI实体类型八类（Target、Actor、Sector、Location、Size、DataType、Price、Date）；②创建了 2,100 条专家标注的真实推文样本；③系统对比了多种模型（域自适应编码器、开源编码器、零射式模型与生成式 LLM），展示了域自适应编码器在速度与精度上的优势。

**🔧 技术方法**

使用的技术包括：BERT 变体（DarkBERT、Twitter-RoBERTa 等）进行细粒度序列标注；CRF 结构化解码；LLM（Llama‑3、Gemma、Qwen 等）在 zero‑shot 与 QLoRA 微调两种配置下的生成式抽取；以及对比实验的评估指标为严格 span‑level Micro‑F1 及毫秒级推理延迟。

**📊 数据集**

数据集为 STINER（2,100 条含 17,281 个实体标注的英文推文），包含八类实体并提供训练/验证/测试的时间序列划分。数据来源于 Apify 的 X 推文爬取，并通过 BERT‑based 过滤后挑选。

**📈 对比分析**

比较方法：将 12 种配置（5 个细粒度编码器、1 个开源编码器、1 个零射式模型、3 个微调 LLM）在同一 2024‑2025 未来样本上评估。结果显示：域自适应编码器 DarkBERT 达到 89.33% strict F1，推理延迟 0.71 ms；生成式 LLM 在零射式下精度高但召回低，微调后仍落后 DarkBERT，且延迟 2–7 秒；GLiNER 在无标注的零射式场景下约 71.3% F1，速度最快。

**⚠️ 局限性**

局限性：①仅覆盖英文内容，忽略俄语、中文等社区的威胁信息；②只采集 X/Twitter，可能低估低调或封闭渠道的攻击；③模型对非常规表达仍易失误，特别是实体边界模糊时。

---

## 413. Designing Compact Neural Architectures via Neuron Gating and Mixed Activation

**arXiv ID:** 2608.14443 | [PDF](https://arxiv.org/pdf/2608.14443v1)

**作者:** Abhishek Shukla `[一作]` (IIT Kanpur), Faiz Hamid `[通讯]` (IIT Kanpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出三种松弛的双层优化框架（NAS-NG、NAS-MA、NAS-NGMA），用于在MLP和CNN中将离散的神经元选择与激活函数混合改为可微的连续变量，从而实现网络结构压缩和性能提升。

**💡 创新点**

通过将离散神经元与激活函数决策连续化，首次实现仅通过梯度优化即可完成全连接与卷积网络的结构搜索，并在MNIST与CIFAR‑10上获得与DARTS相当甚至更优的效果。

**🔧 技术方法**

采用双层优化与超梯度法，结合sigmoid门控与softmax混合激活、Adam/SGD等梯度下降，并使用一阶/二阶近似超梯度实现可微搜索。

**📊 数据集**

在MNIST（手写数字）和CIFAR‑10（小型彩色图像）数据集上进行实验，并与DARTS、密集基线及文献模型进行对比。

**📈 对比分析**

通过验证集/测试集准确率、参数量和效率指标评估；在MNIST上将参数量降低约70%同时保持98.6%准确率，在CIFAR‑10上搜索出的网络参数量更少且准确率高于DARTS。

**⚠️ 局限性**

实验主要聚焦在相对简单的数据集（MNIST/CIFAR‑10），未验证在更大规模数据（如ImageNet）和不同网络类型（RNN、Transformer）上的推广性；此外未针对每种方法单独调优超参数，可能导致结果受限。

---

## 414. RegRole: Regularized Role Detection and Prediction in Temporal Dynamic Networks

**arXiv ID:** 2608.14504 | [PDF](https://arxiv.org/pdf/2608.14504v1)

**作者:** Emily J Evans `[一作]` (Brigham Young University), Carlotta Domenicon `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于时序正则化非负矩阵分解的动态角色检测与预测框架 RegRole，能够在多个时间窗口内保持一致的角色集合并学习统一的角色转移矩阵。

**💡 创新点**

创新点在于引入全局转移矩阵和正则化惩罚，确保角色分配随时间平滑变化，同时保持可解释的角色主题，提升预测性能。

**🔧 技术方法**

使用了时序正则化非负矩阵分解（RegNMF）和乘法更新算法，并结合结构与行为特征的手工工程特征以及自动生成的递归特征。

**📊 数据集**

在五个真实世界数据集（Enron、Facebook、Reality、Slashdot、Scratch）以及一个合成角色模拟数据集上进行实验。

**📈 对比分析**

与 DyNMF、RoleX 等现有方法对比，RegRole 在角色预测误差（Frobenius / KL）、转移矩阵迹值（更稳定）以及整体误差均显著优于对手，表现出更低的预测错误。

**⚠️ 局限性**

局限包括对自动生成特征的适用性不如手工特征、需手动调参（角色数 r 与正则化参数 β）以及在高维稀疏特征下可能收敛较慢。

---

## 415. Approximate Muon with low-rank adapters

**arXiv ID:** 2608.14492 | [PDF](https://arxiv.org/pdf/2608.14492v1)

**作者:** Ben Anson `[一作]` (University of Bristol), Edward Milsom `[通讯]` (University of Bath)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的低秩 LoRA 适配器优化器 sMuon，近似实现 Muon 的正交化梯度更新。

**💡 创新点**

创新点在于对 Muon 目标进行线性化并通过最小二乘求解闭式最优步长，仅使用矩阵乘法即可实现，无需 SVD 等高复杂度分解。

**🔧 技术方法**

技术手段包括线性化、最小二乘、Moore‑Penrose 逆、投影矩阵构造以及矩阵乘法加速的低秩算法。

**📊 数据集**

使用了 CommonsenseQA、Winogrande、MMLU、HumanEval、MBPP 等常见 SFT 评测数据集，并在 ReLoRA 预训练实验中使用 FineWeb 数据集。

**📈 对比分析**

与 AdamW、Per‑factor Muon、LoRA‑Muon、Riemannion 等基线比较，sMuon 在 Muon 预训练模型上获得最多任务最优，在 ReLoRA 预训练中同样表现最佳；在训练速度上与 AdamW 相近，远快于 Riemannion。

**⚠️ 局限性**

局限性包括仅针对低秩 LoRA 适配器；在更大秩或不同模型上性能差异显著；需要额外的矩阵根计算，可能在某些硬件上导致数值不稳定。

---

## 416. Optimal Scheduling of Road Maintenance Jobs Considering Impact on Traffic Flows

**arXiv ID:** 2608.14491 | [PDF](https://arxiv.org/pdf/2608.14491v1)

**作者:** Charitha Nandepu `[一作]`, SangWoo Park `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在维护规划中，用数据驱动的代理模型快速估算道路网络在容量受限时的用户均衡流量。

**💡 创新点**

创新点在于直接从OD需求学习非线性网络算子，用注意力网络捕捉全局交互，从而避免频繁求解凸优化。

**🔧 技术方法**

采用深度学习模型（MLP、CNN、GNN、注意力网络）以及凸优化（Beckmann型用户均衡求解）。

**📊 数据集**

使用StreetLight收集的10月2025年纽瓦克地区49条主要道路的OD流量数据。

**📈 对比分析**

对比四种网络结构，注意力网络RMSE 35.87 veh/h、R² 0.9835，显著优于GNN（RMSE 58.48）、MLP（84.32）和CNN（133.84）。

**⚠️ 局限性**

局限在仅测试单一城市网络、缺乏完整维护排程集成、对极端需求或大规模网络的泛化能力待验证。

---

## 417. Ensuring Safe Physical AI in Urban Mobility via Hazard-Informed Synthesized Envelopes

**arXiv ID:** 2608.14481 | [PDF](https://arxiv.org/pdf/2608.14481v1)

**作者:** Alexei Odinokov `[一作]` (SafePi.ai), Rostislav Yavorskiy `[通讯]` (SafePi.ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套统一的框架，利用基于危险分析的合成安全包围（Safety Envelope）在符号、空间和动态层面上实现城市移动机器人与环境的跨层安全转换，并通过物理 AI Harness（Projection‑Isolation‑Transfer）在运行时对 AI 决策进行持续监督，确保安全约束在部署全过程中始终有效。

**💡 创新点**

创新点：
• 将安全视为跨层抽象的动态属性，在符号、空间、动态三层之间实现安全信息的双向流动；
• 通过危险信息驱动的合成场景生成，系统化地学习安全边界而非仅依赖稀缺的碰撞数据；
• 将 Physical AI Harness 的 Projection‑Isolation‑Transfer 机制与安全包围对齐，实现实时投影、资源隔离和安全切换，从而在 AI 驱动的决策中提供可验证的运行时保障。

**🔧 技术方法**

技术手段：
• 符号安全层：基于规则和约束的高层决策推理；
• 空间安全层：基于地图和预测轨迹的碰撞概率与安全距离计算；
• 动态安全层：车辆动力学模型与控制器执行的物理可行性约束；
• 合成安全包围学习：利用危害信息驱动的仿真场景生成并训练机器学习模型，预测安全裕度；
• Physical AI Harness：Projection、Isolation、Transfer 三个机制实现运行时监督；
• NVIDIA Isaac Sim / Isaac Lab、Cosmos、Omniverse 的集成仿真与数据生成。

**📊 数据集**

数据集：基于危险分析的合成场景数据集，涵盖不同资产、暴露模式和危害情境（如行人、骑行者、恶劣天气、拥堵等）。未使用公开真实道路数据，全部为仿真生成的安全边界训练样本。

**📈 对比分析**

对比与性能：目前论文仅提出概念与仿真验证流程，未给出与现有方法的定量对比或性能指标；计划在 NVIDIA 仿真平台上通过数千个合成场景对安全包围和 PIT 机制进行闭环评估。

**⚠️ 局限性**

局限性：
• 依赖仿真生成的数据，缺乏真实道路实验验证；
• 合成场景可能无法覆盖所有极端情况，导致安全包围泛化不足；
• 需要在实际机器人平台上实现和调试 Physical AI Harness，涉及硬件与软件集成挑战；
• 论文未给出实验结果，实际性能表现尚待评估。

---

## 418. SheetCompass: Hierarchical Relation Graphs for Agentic Spreadsheet Reasoning

**arXiv ID:** 2608.14452 | [PDF](https://arxiv.org/pdf/2608.14452v1)

**作者:** Panjing He `[一作]` (University of Science and Technology of China), Xiaohan Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了SheetCompass框架，实现了从平面文本到层次图谱的结构感知，并通过多智能体协作完成复杂电子表格的自动化推理与执行。

**💡 创新点**

创新点包括：①构建双层（表格-列）层次图谱，显式保存空间布局与跨表语义依赖；②引入双级记忆（专家知识+经验记忆）实现动态知识更新；③设计三角色（探索者、程序员、反思者）多智能体循环，结合图约束和闭环验证提升可靠性。

**🔧 技术方法**

使用技术包括图神经网络/Transformer对列节点特征编码，双层记忆机制，基于LLM的多智能体工作流（Explorer、Programmer、Reflector），以及在安全沙盒中执行Python/Excel脚本。

**📊 数据集**

实验数据集：SCB、SB、SheetRM三大复杂电子表格基准，用于评估执行成功率、功能正确性、软/硬约束通关率。

**📈 对比分析**

与Binder、VBA、SheetCopilot、SheetAgent、OS-Copilot等基线对比，SheetCompass在GPT‑4/5 backbone下分别在pass@1、soft/hard restriction、exec@1等指标上实现了显著提升，尤其在多表交叉依赖任务上表现最突出。

**⚠️ 局限性**

局限性包括：①对图构建和阈值参数敏感，需要手工调优；②仍依赖大规模LLM，推理成本高；③对极端稀疏或非标准表格结构的适应性待进一步验证。

---

## 419. CytoBERT: A Foundation Model for Cytometry Data

**arXiv ID:** 2608.14414 | [PDF](https://arxiv.org/pdf/2608.14414v1)

**作者:** Syed Abdul Haseeb Qadri `[一作]` (University of Rostock), Martin Becker `[通讯]` (University of Rostock)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种可处理可变marker面板的单细胞细胞计数基础模型，利用Transformer编码器实现自监督预训练；

**💡 创新点**

创新点在于构建220个统一marker词表并通过掩码表达预测学习细胞内部的marker共表达关系，实现跨数据集的迁移学习；

**🔧 技术方法**

技术包括Transformer编码器、Marker+表达离散化、掩码表达预测自监督预训练以及基于mean pooling的样本级Fine‑tuning分类；

**📊 数据集**

预训练使用15个公开CyTOF数据集约5千万细胞，后续在7个独立的ImmPort CyTOF数据集进行样本分类评估；

**📈 对比分析**

与逻辑回归、CytoSet及随机初始化模型相比，在分层交叉验证和分组子样本下，该模型在分组拆分下平均排名最佳，尤其在未见过的SDY1708和SDY997上提升显著，但差异不显著；

**⚠️ 局限性**

局限性包括对标注样本提升有限、对少量样本的随机采样敏感，以及缺乏更大规模多中心验证和对噪声标签的鲁棒性评估。

---

## 420. Uncertainty-Aware Jacobi Set Computation

**arXiv ID:** 2608.14409 | [PDF](https://arxiv.org/pdf/2608.14409v1)

**作者:** Daniel Klötzl `[一作]` (University of Stuttgart), Daniel Weiskopf `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于多元正态分布的解析不确定性Jacobi集计算方法；

**💡 创新点**

创新点在于将Jacobi集的边缘判定从离散的梯度对齐转化为闭式的概率判定，并通过期望JS度量实现可视化；

**🔧 技术方法**

采用多元高斯不确定性建模、解析线性梯度传播、二阶矩匹配推导边缘交叉概率以及基于显著性热图的可视化编码；

**📊 数据集**

使用了三类数据集：解析测试数据、流体动力学中的涡街流场以及CERRA天气预报的温度与位势场；

**📈 对比分析**

与10,000次Monte Carlo采样进行对比，误差平均约1.5%并且在计算时间上比MC快约5-6倍；

**⚠️ 局限性**

局限性包括对高斯分布的假设、仅在二维平面上可行、二阶矩匹配带来的近似误差以及对跨域协方差建模的不足。

---

## 421. Individual Rationality in Constrained Hedonic Games: Friends, Enemies, and Neutrals

**arXiv ID:** 2608.14461 | [PDF](https://arxiv.org/pdf/2608.14461v1)

**作者:** Šimon Schierrreich `[一作]` (AGH University of Krakow), Ildikó Schlotter `[通讯]` (ELTE Centre for Economic and Regional Studies)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在朋友、敌人和中立关系约束下，加入大小和数量限制的可个体理性（IR）团体分配问题，并对不同偏好方向（友好导向和敌对导向）及四种团体大小约束变体进行复杂度分类。

**💡 创新点**

主要创新在于：① 将朋友/敌人关系游戏与大小约束结合，给出了完整的经典与参数化复杂度边界；② 发现敌对导向下问题可归约为大小约束图着色，极大简化；③ 对友好导向下的复杂度做细致分析，揭示其受敌对结构、大小严格度、关系对称性等多因素影响，并给出相应的可解与不可解情形；④ 引入多种结构参数（顶点覆盖、树宽、团宽）和对称性区分，提供更细粒度的算法与硬性结果。

**🔧 技术方法**

主要技术包括：参数化复杂度分析、颜色编码（color coding）、有限搜索树（bounded search tree）、在整洁树分解上进行动态规划、图形归约与硬度证明、N-阶整数规划与组合构造等。

**📊 数据集**

本研究为理论计算复杂度研究，无使用实际数据集；所有结果均基于抽象图模型和理论构造。

**📈 对比分析**

通过复杂度分析与参数化归约对比，给出在不同参数（k、大小上限/下限、敌人数、结构参数等）下的可解（多项式/固定参数可解）与不可解（NP‑难/W[1]‑难）边界；并对四种约束变体分别给出完整的表格与证明，展示友好导向与敌对导向在可解性上的显著差异。

**⚠️ 局限性**

局限性：① 仅给出理论复杂度结果，缺乏实验验证或实际实例评估；② 对于某些参数组合的复杂度仍未完全确定（表格中有“?”的情况）；③ 只关注个体理性约束，未讨论更严格的稳定性概念或效率指标。

---

## 422. Large-scale workflow placement in serverless computing using integer nonlinear programming

**arXiv ID:** 2608.14427 | [PDF](https://arxiv.org/pdf/2608.14427v1)

**作者:** Joshua Adamek `[一作]` (Technische Universität Dortmund), Sergio Lucia `[通讯]` (Technische Universität Dortmund)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种面向大规模边缘云服务器无关架构的工作流放置优化框架，采用混合整数非线性规划（MINLP）模型联合考虑工作流部署与运行时选择；

**💡 创新点**

核心创新在于将工作流结构与节点属性统一编码为可分解的MINLP模型，并引入层次化区域化分解策略，实现对数级别的可扩展性，同时通过利用率惩罚实现对等待时间的近似控制；

**🔧 技术方法**

技术手段包括：混合整数非线性规划建模、Gurobi求解器、层次化区域分解、利用率惩罚代替等待时间、加权求和多目标优化、Pareto前沿分析、对参数不确定性的敏感性与稳健性评估；

**📊 数据集**

实验数据基于仿真生成的合成工作流与架构参数，工作流属性（计算时间、数据大小、内存占用）从均匀分布采样，架构参数（云端费用、延迟、速度提升）取自公开云服务估值；

**📈 对比分析**

通过与集中式完整优化以及单云提供商的启发式基线比较，分解策略在保持与集中优化成本差距≤10%的同时，将求解时间从秒级降至毫秒级；在大规模多级架构下，分解算法表现出对数级的求解时间增长，而集中式方案呈指数级；

**⚠️ 局限性**

局限性包括：分解解可能不完全最优（受区域化约束和固定选择策略影响），对实时动态请求与资源争用的适应性有限；对参数估计误差敏感，需在部署前做稳健或再规划；求解仍受MINLP求解器性能限制，难以处理极大节点规模或极端不确定性。

---

## 423. RecipeNet: A Hierarchical Transformer for Recipe Data

**arXiv ID:** 2608.14505 | [PDF](https://arxiv.org/pdf/2608.14505v1)

**作者:** Pin-Yen Huang `[一作]` (Arizona State University), Baoxin Li `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个层次化Transformer架构RecipeNet，专门用于处理具有可变字段集、层次结构和顺序依赖的配方（recipe）数据。

**💡 创新点**

创新点在于：①在步骤级别对异构字段进行Token化并利用Transformer编码，捕获步骤内字段交互；②在配方级别再用Transformer编码步骤序列，建模跨步骤的全局上下文；③整体结构天然支持可变schema，避免传统方法的稀疏填充。

**🔧 技术方法**

技术上使用了：数值字段线性映射、类别字段嵌入、步骤位置与字段身份嵌入相加得到Token；步骤级Transformer（带STEPCLS）聚合步骤内部Token；配方级Transformer（带RECIPECLS）聚合步骤表示；最后通过线性层完成下游任务。

**📊 数据集**

实验数据集包括：固态反应(Solid‑State Reaction)、溶胶‑凝胶前驱体合成(Sol‑Gel Precursor Synthesis)、溶液合成(Solution Synthesis)。

**📈 对比分析**

与传统固定-schema方法（XGBoost、CatBoost、TabNet、NODE）及其他深度表格学习方法（Transformer、Set Transformer、FT‑Transformer、TabTransformer）进行对比。RecipeNet在下游的next‑step预测和masked‑step预测任务中均取得最高或近乎完美的准确率，且训练时间最短，性能优势显著。

**⚠️ 局限性**

局限性包括：模型仍需针对配方结构进行预处理（字段Token化和步骤划分）；在更大规模或更复杂领域的泛化仍待验证；以及与最简模型相比，训练复杂度略高。

---

## 424. Lower Bounds on Black-Box Constructions of Pseudorandom Functions

**arXiv ID:** 2608.14501 | [PDF](https://arxiv.org/pdf/2608.14501v1)

**作者:** Bar Alon `[一作]` (Tel Aviv University), Muthuramakrishnan Venkitasubramaniam `[通讯]` (Ben Gurion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过元攻击与分数函数技术，证明了在完全黑盒构造中，伪随机函数（PRF）从伪随机生成器（PRG）构造时，若仅使用非自适应调用，至少需要Ω(n/log n)次PRG调用；同样地，构造长度为n、输出位长为w(n)=ω(log n)的弱PRF时，单调用即可达到下界；

**💡 创新点**

创新点在于：①提出了“查询上限（query‑bounded）”的完全黑盒降维模型；②利用分数函数与随机阈值相结合的理想与实际对手，构造了对PRG安全性的不可行性证明；③在弱PRF和长输出PRF两种情形下，均给出了新的下界，进一步逼近GGM构造的调用次数；

**🔧 技术方法**

核心技术包括：元攻击范式（meta‑reduction）、对理想对手和真实对手的分数函数定义、独立性与概率集中不等式、以及对PRG查询图的度数与独立集分析；

**📊 数据集**

本文无实验数据集，研究完全基于理论分析与可证明的不可能性结果；

**📈 对比分析**

由于研究性质为理论不可能性，没有直接实验对比；但作者通过与已知的GGM构造（调用ω(log n)次）以及最近对树构造的最优性证明相结合，说明其下界仅相差多项式因子；

**⚠️ 局限性**

局限性：证明仅适用于“查询上限”下的完全黑盒降维；对一般黑盒降维（无查询上限）仍未覆盖；此外，结果仅针对非自适应调用的构造，对于自适应调用的更强上界仍待研究。

---

## 425. The Past and Future of AI Scientists

**arXiv ID:** 2608.14407 | [PDF](https://arxiv.org/pdf/2608.14407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 426. Ex-ante versus Ex-post: Egalitarian Facility Location Mechanism Design

**arXiv ID:** 2608.14499 | [PDF](https://arxiv.org/pdf/2608.14499v1)

**作者:** Zohar Barak `[一作]` (Tel Aviv University), Inbal Talgam-Cohen `[通讯]` (Tel Aviv University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

研究在欧氏空间中，无钱策略性设施定位问题的均等化目标（最大代价）下，随机机制在预期与后验公平度量上的性能，提出了新的随机机制并给出上界与下界。

**💡 创新点**

① 首次把预期公平度量（ex‑ante）与后验公平度量（ex‑post）区分并证明它们在低维下严格区分；② 设计了在二维空间中可突破确定性极限的随机机制（随机旋转角点、Random Rotated Corner）；③ 用距离放大与多维正交单纯形构造，证明高维空间中任何预期策略性机制的逼近比不得优于 2–o(d)；④ 给出二维下 ex‑post 与 ex‑ante 的新下界（分别为 1.605 与 1.09）。

**🔧 技术方法**

采用几何优化、凸性与 Bauer 极值原理、Jensen 不等式、距离放大技术、随机旋转与正交子空间的单纯形放大、离散化圆平均与三角不等式，以及 Gaussian Poincaré 不等式等数学工具。

**📊 数据集**

使用人工合成实例（点集、正多边形、单纯形、圆上离散点等）来构造上界与下界证明；没有使用真实世界数据集。

**📈 对比分析**

与确定性独裁机制（逼近比 2）和坐标中位数机制（逼近比 1+√2≈2.41）进行对比；在 R² 上预期公平度量的最佳随机机制逼近比约为 1.598，后验公平度量的最佳随机机制逼近比约为 1.5，均优于 2；但在高维下随机机制无法突破 2，符合下界 2–o(d)。

**⚠️ 局限性**

1) 仅在二维空间中实现了改进，三维及更高维仍未能突破 2；2) 对于其他公平度量（如分位数成本）或弱策略性（近似策略性）仍未探讨；3) 证明依赖于精细几何构造，实际实现与计算成本未评估。

---

## 427. You Only Pass Once: Answering and Abstaining Together in a Single Forward Pass of a Frozen Language Model

**arXiv ID:** 2608.14465 | [PDF](https://arxiv.org/pdf/2608.14465v1)

**作者:** Ziyang Luo `[一作]`, Yan-Syuan Chen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了YOPO系统，在一个前向传播中同时实现冻结模型的推理写入（steering）和置信度判断（sufficiency），并通过标签无监督的残差重建实现一次通过式答案或拒绝。

**💡 创新点**

将写入器和读取器在同一残差流上联合，提出标签无监督的残差重建来消除写入干扰，并量化容量‑迁移边界，首次在多模型、多族上展示一通过式推理与拒绝的性能提升。

**🔧 技术方法**

使用条件steering probe进行写入、零样本差分均值方向读取、MSE重建映射修正、BCE微调、零样本阈值量化以及链式思维（CoT）等技术。

**📊 数据集**

利用αNLI、HellaSwag、RepLiQA、MuSiQue、SQuAD2.0、BoolQ等公开数据集，以及自构造的四条件红action集合。

**📈 对比分析**

在Qwen2.5-1.5B/3B/7B以及十个不同族模型上与两通道清洁读、单向写、两通道等基准对比；YOPO在三方准确率上比冻结基线提升约120%，单通道下与两通道同等，并在跨域迁移时保持高AUROC。

**⚠️ 局限性**

写入干扰在大模型中已降至零，标签无监督修正对小模型有效，但在极大规模模型上需使用身份映射；构造的红action任务对特定域不敏感，链式思维在未额外训练时受限；多模型实验主要基于单随机种子，未覆盖全部噪声。

---

## 428. Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View

**arXiv ID:** 2608.14430 | [PDF](https://arxiv.org/pdf/2608.14430v1)

**作者:** Yixian Xu `[一作]` (Peking University), Di He `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于路径空间的重要性采样和方差减小的统一强化学习框架，将传统的反向轨迹方法（Flow‑GRPO 等）与前向匹配方法（AWM、DiffusionNFT）统一在同一理论下，并提出了多样本 KDE 值梯度估计与尺度约束权重设计，形成了完整的算法设计空间；

**💡 创新点**

核心创新是：① 通过路径空间重要性采样将原始 RL 目标转化为可行的方差减小估计；② 发现不同方法的差距源于方差而非原则差异；③ 引入 KDE 多样本估计和尺度约束权重，系统化权重与估计器的匹配；

**🔧 技术方法**

使用路径空间重要性采样、Itô 积分方差减小、KDE 值梯度估计、尺度约束权重、流匹配采样和对比式奖励评估；

**📊 数据集**

在 SD‑3.5‑Medium 与 Qwen‑Image 两个 512×512 视觉生成模型上，使用 PickScore、OCR、GenEval 三种奖励函数进行实验；

**📈 对比分析**

与 Flow‑GRPO、DiffusionNFT、AWM 等基线对比，采用相同超参（LoRA、CFG 关闭），实验表明：在 OCR 奖励下速度提升 2–3 倍；在 PickScore 与 GenEval 上与 AWM、DiffusionNFT 对齐或略优；在 Qwen‑Image 上对 HPSv3 奖励的收敛速度提升 4 倍；

**⚠️ 局限性**

局限性：仍需手动调节 KDE 带宽与权重指数，且在极端高噪声或不同任务分布下的鲁棒性未完全验证；方法对采样器选择和奖励函数形状敏感，需要进一步自动化与理论完善。

---

## 429. Effect of Twisted-Yarn Architecture on Pressure and Proximity Sensing Characteristics of Textile Capacitive Sensors for Robotic Skin

**arXiv ID:** 2608.14406 | [PDF](https://arxiv.org/pdf/2608.14406v1)

**作者:** Ishtia Zahir `[一作]` (V-Trion GmbH), Gaffar Hossain `[通讯]` (V-Trion GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将聚二甲基硅氧烷（PDMS）包覆的银线纱丝按1层、2层、4层等多层扭绞方式制备，可控的纱线层级结构实现了纺织集成电容传感器，并系统研究了该结构对有效电极重叠面积、纱线间距以及压力与接近感知特性的影响；

**💡 创新点**

创新点在于提出并验证纱线层级结构作为单一可调设计参数，用以实现电容传感器的压力灵敏度、接近范围、机械强度与热稳定性等性能的可调，并揭示了层数与感知范围之间的权衡关系，形成了结构–测量的定量关系；

**🔧 技术方法**

采用银线纱丝+PDMS封装、丝线多层扭绞、织物网格拼接、LCR电容测量、温度控制、周期加载、接近感知实验，以及将传感阵列集成至PSoC 6、Franka Emika Panda机械臂等硬件平台；

**📊 数据集**

实验数据包括：0.4–3.9 MPa压力下的电容响应、15 000次循环加载、不同材料（手、金属、塑料、纸）在0–60 mm距离下的接近检测、4×4传感阵列的空间映射以及机器人实时控制所需的时延；

**📈 对比分析**

通过对比1层、2层、4层结构在相同频率下的压力灵敏度（4层最高达0.1331 MPa⁻¹）、滞后、响应时间（≈0.9 s）和热漂移（25–90 °C低于1 %），以及机器人系统的端到端延迟（≈403 ms），验证了多层结构在提升灵敏度与保持低滞后、耐久性方面的优异性能；

**⚠️ 局限性**

局限性包括：纱线层级与接近范围存在不可避免的权衡；步进响应时间相对较慢，动态频率受限于5 Hz；实验主要针对特定纱线与PDMS材料，扩展到更大面积或不同环境仍需进一步验证；

---

## 430. Forging Self-Funded Marketplaces among Strategic Agents

**arXiv ID:** 2608.14548 | [PDF](https://arxiv.org/pdf/2608.14548v1)

**作者:** Yuan Deng `[一作]`, Song Zuo `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究自给式（autarkic）市场中，设计既满足预算平衡又最大化价值的机制，并提出基于顺序Best-and-Final-Offer (BAFO) 的拍卖方案。

**💡 创新点**

创新点在于：①首次将Maximin Share (MMS) 作为自给式市场的合理基准；②证明任何DSIC机制对最优价值(OPT)无界逼近；③提出顺序BAFO实现对OPT的O(log V)逼近，对MMS的常数逼近。

**🔧 技术方法**

采用的技术包括：预算可行性约束的价格单调性、BAFO的序列化竞争、微分不等式与积分技巧、随机化猜测策略、分区均衡与期望分析。

**📊 数据集**

该工作为理论研究，无实验数据集；所有结果均为理论证明。

**📈 对比分析**

通过构造对偶实例给出下界；利用顺序BAFO与随机猜测实现上界；在OPT基准上得到O(log V)近似，在MMS基准上得到常数近似。

**⚠️ 局限性**

局限性：对MMS的常数逼近仅在可分数化分配时成立；整数分配下无法保证；DSIC机制对OPT的逼近仍无界。

---

## 431. Decoding the Past: An Uncertainty-Aware Deep Learning Framework for Sex Attribution in Prehistoric Hand Stencils

**arXiv ID:** 2608.14539 | [PDF](https://arxiv.org/pdf/2608.14539v1)

**作者:** Karel Becerra `[一作]` (Azyri), Ramón A. Mollineda `[通讯]` (Universitat Jaume I)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了一种多层不确定性感知的深度学习框架，用于推断史前洞穴手印的生物性别。

**💡 创新点**

创新点在于通过源级、注释级、形态级等多维度生成多样化轮廓，并结合模型多样化、分层聚合与三角化验证（UMAP+kNN、LayerCAM），实现可量化的不确定性推断。

**🔧 技术方法**

采用EfficientNet‑B3与MobileViT‑S双网络集成、SAM分割、结构化轮廓增强、三通道拼接、UMAP嵌入、kNN与LayerCAM可解释性分析等技术。

**📊 数据集**

使用RSNA骨龄挑战集14,036张左右手X光片进行训练/验证/测试，并以9张史前手印图像作为案例验证。

**📈 对比分析**

在现代样本上，EfficientNet‑B3集成以sum聚合实现≈88%准确率，随着年龄提升性能进一步提升；在史前样本上，框架能给出高置信度预测，并与UMAP+kNN和LayerCAM结果一致，部分样本与先前研究结果吻合。

**⚠️ 局限性**

局限在于依赖现代手部样本的性别特征假设，无法直接验证史前人群差异；手印图像分辨率和边界不确定性仍限制模型精度；缺乏真实生物性别标注导致结论仅为概率推断。

---

## 432. Spatiotemporal Tube-Based Safety-Certificate for Autonomous Navigation of Articulated Vehicles

**arXiv ID:** 2608.14531 | [PDF](https://arxiv.org/pdf/2608.14531v1)

**作者:** Mohd. Faizuddin Faruqui `[一作]` (Indian Institute of Science), Pushpak Jagtap `[通讯]` (Indian Institute of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于空间-时间椭圆管（STET）的离线规划框架，用于自适应轨道安全认证（RSC），实现多轴车拖车在复杂路线上的安全导航。

**💡 创新点**

创新点在于：①利用STET对连杆车辆的几何约束做可视化的安全管道；②通过可接受的中心偏移与椭圆收缩，动态修正主车轨迹，保证末尾拖车始终在路段内部；③用最小宽度差值直接给出可判定的路段安全证书（RSC）。

**🔧 技术方法**

采用的技术包括：椭圆管的Minkowski和（凸分析）、LMI求解的可达集上界、切线支撑半径计算、组合映射（Ω、Ξ）的Lipschitz性质、以及两步纠正算法（CORRCEN、SHRINK）。

**📊 数据集**

实验使用基于仿真的多场景数据集：不同数量的拖车（N+1=3~6）、拖杆长度、摆角限幅、拖车尺寸、h_Ω、h_Ξ、以及不同宽度的路段（w_min）。

**📈 对比分析**

与优化式规划、鲁棒控制不变管（RCI）、控制收缩度量（CCM）等方法对比，本文方法不需要复杂的非线性动力学或轨迹优化，计算量线性扩展，运行时间约0.5–1.15 s；在模拟场景中，RSC 正负判定与真实安全状态一致，证明了方法的可行性与效率。

**⚠️ 局限性**

局限性包括：①仅处理几何约束和已知路段宽度，未考虑障碍物与动态环境；②对滑移、外部扰动的鲁棒性仅通过界限摆角实现；③目前仅在仿真验证，缺乏真实道路实验；④在线自适应控制未结合，可能在突发事件下表现不佳。

---

## 433. CPI-Bench: A Comprehensive,Practical and Intelligent Benchmark for Real-World Image Editing

**arXiv ID:** 2608.14546 | [PDF](https://arxiv.org/pdf/2608.14546v1)

**作者:** Qinye Zhou `[一作]` (Alibaba Group), Mengting Chen `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了CPI‑Bench，一套包含通用、实践和智能三大子集的图像编辑评测基准；

**💡 创新点**

创新点在于引入多图像编辑、真实应用场景与高难度推理任务，填补了现有基准的空白；

**🔧 技术方法**

使用了基于Vision‑Language Models的自动评估框架，针对不同任务设计专属打分提示；

**📊 数据集**

数据集来源包括100,000+公开/授权图像，构成30个编辑子任务（含20个单图任务与10个多图任务）、51个实践场景以及1,181个推理实例；

**📈 对比分析**

与GEdit‑Bench、ImgEdit‑Bench、REDEdit‑Bench等传统基准对比，CPI‑Bench能显著放大模型间性能差异，并与Arena Image Edit Leaderboard的排名呈现最高相关性（Spearman最高、MAE最低），说明其更贴近人类偏好；

**⚠️ 局限性**

局限性包括对开放源模型评估仍受推理能力限制，且评估依赖VLM的自动化可能存在主观误差，未来需进一步扩展多模态与更细粒度的推理测试。

---

## 434. Polynomial-Factor Deterministic NP-Hardness for SVP in Every lp Norm with p > 2

**arXiv ID:** 2608.14529 | [PDF](https://arxiv.org/pdf/2608.14529v1)

**作者:** Isaac M Hair `[一作]` (University of California Santa Barbara), Amit Sahai `[通讯]` (University of California Los Angeles)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构造了一个确定性多项式时间的归约，将3CNF公式转化为ℓ_p范数下的最短向量问题（SVP）近似难度的实例；

**💡 创新点**

创新点在于利用同质动量码与Hadamard变换相结合的构造，首次实现对所有有限p>2（以及p=∞）的ε<min{1/2-1/p,1/8}范围内的确定性多项式时间NP难度证明；

**🔧 技术方法**

技术上主要运用了Reed–Solomon码的线性约束、加权霍恩矩阵与多项式重构、构造A提升以及分块Hadamard映射来将稀疏码向量的支持信息转换为ℓ_p范数间隙；

**📊 数据集**

由于研究为理论复杂度结果，未使用任何实际数据集；

**📈 对比分析**

与之前的随机归约相比，本文的确定性归约在相同的逼近因子范围内提供了更强的硬度保证；

**⚠️ 局限性**

局限性包括仅适用于p>2（p=2仍未达到确定性多项式时间硬度），且逼近因子受限于min{1/2-1/p,1/8}，在更大逼近因子下尚未达到。

---

## 435. MagnifiQ: Patch-aware Text Guided Progressive Upscaling for High-Resolution Image Restoration

**arXiv ID:** 2608.14543 | [PDF](https://arxiv.org/pdf/2608.14543v1)

**作者:** Mahesh Reddy `[一作]` (Qualcomm AI Research), Guillaume Berger `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种Patch‑Aware、文本引导的渐进式放大框架MagnifiQ，用于将低质量图像恢复到4096×4096甚至更高分辨率（最多32×缩放）

**💡 创新点**

创新点在于：①用预训练的文本‑图像扩散模型SDXL进行恢复，并用高效的PADRe替代自注意力，降低计算成本；②引入Patch‑Aware Cross‑Attention（PACA），让每个图像块只关注对应的局部文本提示；③采用渐进式放大流程，逐步提升分辨率并在每个阶段生成局部提示，避免直接一次性放大导致的纹理重复与结构漂移

**🔧 技术方法**

技术手段包括：预训练SDXL的输入卷积改造、PADRe高效注意力、Patch‑Aware Cross‑Attention、LLaVA（或DAPE）生成局部文本提示、轻量级降噪步骤和多阶段迭代

**📊 数据集**

使用Aesthetic‑4K、DIV2K、RealSR等公开数据集进行实验，合成与真实退化图像均有测试

**📈 对比分析**

与StableSR、SUPIR、MultiDiffusion、SD Upscaler、SDXL‑IR/NoSA/PD等方法对比，指标包括无参考质量（CLIP‑IQA、LAION‑Aesthetic、PI、NIQE等）以及运行时；MagnifiQ在大多数指标上表现最好，用户研究中约75%用户更偏好其结果；尽管单次推理耗时较高，但相比StableSR等方法仍显著加速

**⚠️ 局限性**

局限性：需要多阶段迭代，导致整体耗时比单次推理方法更长；对局部提示的生成依赖于文本生成模型（LLaVA/DAPE）性能，可能影响细节质量；当前实现以SDXL为基础，扩展到其他架构需要进一步验证

---

## 436. Marionette: Predicting World States, Rendering Geometry, Painting Appearance

**arXiv ID:** 2608.14530 | [PDF](https://arxiv.org/pdf/2608.14530v1)

**作者:** Zian Meng `[一作]` (Alaya Lab), Kaipeng Zhang `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个可控的多实体游戏世界模型，将世界状态显式化，并通过零参数渲染桥实现几何和遮挡的精确度，神经网络仅负责动力学与外观生成。

**💡 创新点**

将世界模型拆分为两阶段因果变压器（动作决策+连续动画）与确定性渲染桥，显式控制令牌可直接被覆盖，实现对姿态和动作的精确可控。

**🔧 技术方法**

使用两阶段因果变压器（ActionGPT与PoseGPT）、零参数几何渲染桥以及基于DiT的控制条件视频扩散观测模型（Wan2.2-Fun-5B）。

**📊 数据集**

利用WildWorld采集的同步数据集，包括两实体（怪物与玩家）每帧 276 维关节状态和对应的 RGB 视频。

**📈 对比分析**

与端到端像素自回归基线对比，观测层 FVD 仅略高（831 对 799），状态层误差以米为单位可量化；控制令牌覆盖能显著改变姿态（误差下降约 31%）。

**⚠️ 局限性**

限制在于未记录实体导致几何缺失、观测模型的外观随时间衰减、生成状态与训练状态分布漂移以及对非完整录制状态的处理不足。

---

## 437. Finding Vulnerabilities via LLM-Augmented Semantics-Aware Type-Checking

**arXiv ID:** 2608.14533 | [PDF](https://arxiv.org/pdf/2608.14533v1)

**作者:** Ruizhe Wang `[一作]` (University of Waterloo), N. Asokan `[通讯]` (University of Waterloo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于LLM的语义感知类型检查框架，能够在Python项目中自动推理变量语义并检测安全漏洞。

**💡 创新点**

将LLM与传统静态分析结合，提出语义型类型系统，利用LLM对代码语义进行推理，避免手工规则，提升检测精度。

**🔧 技术方法**

使用pytype作为静态分析基础，加入LLM推理（Qwen3-32B via Ollama），构建双向类型推断与检查；同时实现端点识别、执行上下文跟踪和漏洞规则匹配。

**📊 数据集**

评估数据集包括SVEN（80个漏洞实例）、近期CVE（9个漏洞）和流行开源Python Web项目，涵盖Flask、Django、FastAPI。

**📈 对比分析**

与CodeQL、Bandit、Vulnhuntr比较，在SVEN上实现87% precision/recall/F1；在近期CVE上分别达到78%整体性能；在大项目中发现15个零日漏洞，9个被确认，且与基线相比检测率提升显著。

**⚠️ 局限性**

存在未覆盖的动态代码、跨语言交互、对Python 3.10依赖、有限的漏洞规则和库支持，LLM可能产生幻觉或误判，导致误报或漏报。

---

## 438. Trust Without Boundaries: An Architectural Analysis of Satellite Flight Software

**arXiv ID:** 2608.14532 | [PDF](https://arxiv.org/pdf/2608.14532v1)

**作者:** Jack Vanlyssel `[一作]` (University of New Mexico), Afsah Anwar `[通讯]` (University of New Mexico)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对NASA的cFS以及多种开源航天软件框架进行架构信任分析，验证恶意组件可利用架构弱点进行攻击，并对不同框架进行系统比较。

**💡 创新点**

提出基于执行、身份、通信、可观测性、持久化五个视角的信任分析框架；通过NOS3仿真实验验证恶意组件在架构层面的攻击能力；揭示多种框架共性弱点，促成对飞行软件安全范式的重新审视。

**🔧 技术方法**

利用文档+源代码静态分析、NASA NOS3仿真环境实验、恶意组件原型实现，以及对cFS、F'、KubOS、NMF、CORDET-C2等框架的对比性分析。

**📊 数据集**

使用NASA NOS3模拟器默认的18个组件配置；cFS开源代码与配置文件；其他框架（F'、KubOS、NMF、CORDET-C2）的源代码与官方文档。

**📈 对比分析**

采用五个“视角”对架构进行评估，并通过实验验证每个弱点；在执行隔离、身份认证、通信授权、可观测性、持久化恢复等维度对比各框架，发现大部分框架缺乏强隔离和可信恢复，安全性不足；未进行量化性能评估。

**⚠️ 局限性**

只对cFS做了实验，其他框架仅做文档+代码分析；NOS3不包含所有硬件、时序或恢复特性；威胁模型仅考虑已植入恶意代码；未评估初始妥协概率与机制，实验环境与真实飞行环境存在差距。

---

