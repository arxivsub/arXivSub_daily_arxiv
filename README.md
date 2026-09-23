# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-23 | 今日论文总数: 766

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Dynamic Conformance Testing of WebGPU Through Specification-Driven Mutation

**arXiv ID:** 2609.25520 | [PDF](https://arxiv.org/pdf/2609.25520v1)

**作者:** Mahya Samdaliri `[一作]` (New Jersey Institute of Technology), Kasthuri Jayarajah `[通讯]` (New Jersey Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于 WebGPU 规范的动态一致性测试框架，对官方 CTS 测试进行语义驱动的 AST 变异，并在 ASan-Instrumented Chromium 上执行，定位并复现了多项安全漏洞。

**💡 创新点**

创新点包括：① 将 WebIDL 提取的显式语法约束与 LLM（GPT‑5）辅助抽取的隐式语义约束结合，形成双层规则；② 设计了可配置有效/无效模式与变异强度的 AST 变异器；③ 通过 CTS 作为种子，避免从零生成程序，能触发深层调用链（如 Skia 118 层堆栈）并发现此前未被检测的缺陷。

**🔧 技术方法**

使用技术：WebIDL 解析与规范文本提取、GPT‑5 进行规范理解、Tree‑sitter 解析 JavaScript AST、基于规则的文本变异、Chromium ASan/DCheck 运行时检查、日志收集与定位、Python+Shell 脚本实现整个流水线。

**📊 数据集**

数据集：WebGPU Conformance Test Suite（约 3,411 个测试）、WebGPU 规范文本（Bikeshed/HTML）、Chromium 133 与 146 版本的浏览器构建。

**📈 对比分析**

评估方法：通过变异规模（20/40/60/80/100）和模式（valid/invalid）对唯一日志位置数进行累积覆盖曲线；比较单线程与多线程运行的覆盖与日志量；跨版本对比覆盖与日志严重性分布；最终验证发现 7 起故障，其中 3 起可复现并提交。性能方面，变异规模 80 与无效模式在覆盖率与缺陷发现率上表现最佳，单线程跑 15 小时即可覆盖 195 个唯一位置，且并行并未提升覆盖。

**⚠️ 局限性**

局限性：① 以日志位置为唯一指标，未获得完整代码覆盖；② 隐式规则由 LLM 生成后仍需人工验证，主观性高；③ 仅在 Chromium 上测试，跨浏览器通用性未知；④ 变异器依赖手工维护的规则，规范更新需手动同步；⑤ 对图形栈的深度覆盖有限，主要触发的是非 GPU 相关路径。

---

## 2. Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains

**arXiv ID:** 2609.25498 | [PDF](https://arxiv.org/pdf/2609.25498v1)

**作者:** Volkan Dağlı `[一作]` (Anadolu University), Dağhan Dağlı `[通讯]` (Toros)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于曼德布罗特集合边界的无权重、零VRAM的“Universal Fractal Natural Language Decision Map”框架 werr，实现边缘即时决策。

**💡 创新点**

通过将复杂动力学与分形边界生成强类型决策，并结合信息熵声学衰减、四象限相位旋转和在线指数移动平均自适应校准，实现了在CPU上毫秒级低能耗推理。

**🔧 技术方法**

采用曼德布罗特边界逃逸动力学、Auto-Seed Router映射、信息熵声学衰减滤波、四象限相位旋转、在线指数移动平均、双层语义解析，并在EVM/Solana 上实现单槽式智能合约。

**📊 数据集**

在公开的 1,150+ 决策数据集（3,200+ 问题）以及 JevBench 基准（Issue #10）上进行评估，覆盖 30+ 业务域。

**📈 对比分析**

与 4B Transformer (Jev)、本地 4B LLM (OpenJev) 对比，werr 在 0 Byte VRAM、单核 CPU 上中位延迟 3.31 ms，宏平均准确率 92.6%（JevBench 世界第一 81.65%），能耗仅 0.04 mJ/查询。

**⚠️ 局限性**

受限于固定 3 维种子坐标映射，对极端 OOV 仍需预处理；分形动态特性对高精度多分类仍有限；在极低功耗 MCU 仍需 2 KB SRAM；可解释性与可迁移性待进一步验证。

---

## 3. Strategic Disclosure of Action Space in Principal-Agent Contracts

**arXiv ID:** 2609.25410 | [PDF](https://arxiv.org/pdf/2609.25410v1)

**作者:** Xiaotie Deng `[一作]` (City University of Hong Kong), Ningyuan Li `[通讯]` (Peking University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在主从合同中，代理人通过对可实现行动空间的战略披露来影响主人的合同设计，从而改变双方的收益与社会福利。

**💡 创新点**

创新点在于：①首次将行动空间的披露视为内生信息不对称；②在成本可验证与不可验证两种情形下给出代理人最优披露策略的解析；③证明在可验证成本下，代理人至少可获得第一最佳剩余的 1/e 份额，并给出主人的收益下界与上界；④将问题转化为两变量凸优化并给出近似算法。

**🔧 技术方法**

主要技术包括：成本曲线（lower convex hull）表示法、线性合同简化、凸优化与对偶分析、对数变换与支撑线理论、以及 1/e 近似常数的证明。

**📊 数据集**

本文为理论分析，不涉及具体数据集；所有结果均来自数学证明与构造性示例。

**📈 对比分析**

与传统公开行动空间（canonical）模型对比：代理人效用不低于基准且可无界提升，主人的收入不提升且可无限降低，社会福利可提升也可下降。实验结果仅通过理论极端例子展示；在有限奖励比 L 时，主人的收入至少为 Θ(1/ log L) 的第一最佳剩余。

**⚠️ 局限性**

限制：仅在二进制或线性合同场景下给出最优分析；对可验证成本的假设较强；未考虑动态学习与主人的信息更新；风险中性假设；对多期重复交互的长期激励未完全覆盖。

---

## 4. Higher-Order Approximation of Exit Functionals in Sampling-Based Stochastic Model Predictive Control

**arXiv ID:** 2609.25257 | [PDF](https://arxiv.org/pdf/2609.25257v1)

**作者:** Sashank Modali `[一作]` (Purdue University), Takashi Tanaka `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在采样基随机模型预测控制中，利用高阶数值方法精确逼近退出事件，并将其应用于基于路径积分的机会约束控制。

**💡 创新点**

创新点在于证明终端边界层估计下，强退出时间误差可转移至退出指示符和松弛停滞成本误差，并提出自适应阶1 Milstein（含 Lévy 面）与阶1.5方法，分别适用于非交换和交换扩散，实验显示显著提升。

**🔧 技术方法**

使用技术包括自适应高阶 Itô–Taylor 积分、Wiktorsson Lévy 面模拟、终端管道估计、Cole–Hopf 变换的路径积分解析、以及 dual ascent 与 Adam 优化。

**📊 数据集**

实验采用两种平面小车模型（非交换模型与升降执行器模型）以及圆形/矩形安全域的仿真数据，未使用公开数据集。

**📈 对比分析**

通过与 Euler–Maruyama 基准在强误差、指示符误差和闭环违约率等指标上对比，自适应高阶方法在相同步长下误差下降 2–4 倍，违约率从约 13–14% 降至 10%，显著提升安全性能。

**⚠️ 局限性**

局限性包括需要终端边界层假设、适用于 C^4 边界且满足局部反集中，且对 C^4 以外或高度偏微分（hypoelliptic）系统的理论尚不完整，此外非交换模型中 Lévy 面计算仍有较高成本。

---

## 5. Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum

**arXiv ID:** 2609.25028 | [PDF](https://arxiv.org/pdf/2609.25028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 6. PermuFormer: Multi-Task Pretraining for Permutation Representation in Algebraic Combinatorics

**arXiv ID:** 2609.25438 | [PDF](https://arxiv.org/pdf/2609.25438v1)

**作者:** Henry Kvinge `[一作]` `[通讯]` (Pacific Northwest National Laboratory), Henry Kvinge (Pacific Northwest National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并训练了 PermuFormer，一款自回归 transformer，专门针对排列组合学中的多任务、多编码计算任务进行预训练与微调；

**💡 创新点**

其创新点在于通过覆盖 306 种排列相关任务、六种编码方式的 2.8 B 词大规模预训练，获得可迁移的排列表示，并对模型内部实现机制（如线性可解码时间、编码对任务的影响）进行深入分析；

**🔧 技术方法**

使用了 LLaMA‑style 12‑层、75 M 参数的 decoder‑only transformer，配合 186 词表的专用 tokenizer，自回归训练，结合线性探针、CKA、UMAP 等可解释工具；

**📊 数据集**

构建了 2.8 B 词的多任务多编码数据集，涵盖从 S₂–S₁₁ 的排列翻译、统计与运算任务（共 306 类），并提供 200k 示例的评估集；

**📈 对比分析**

通过精确匹配（token‑for‑token）评估，在训练外任务上与从零训练、Pythia‑70m、MLP、LogReg 基线进行对比，PermuFormer 在多数研究级任务（如 Schubert 结构常数、Kazhdan–Lusztig 系数、mHeight 等）上显著优于基线，甚至接近 100% 的准确率；

**⚠️ 局限性**

局限性包括：模型仅针对排列任务，难以直接推广到更广泛的数学结构；在某些高阶属性（如 KL 多项式高阶系数）上表现下降；依赖手工设计的 token 语法与特定任务格式，缺乏对更大规模或更复杂问题的验证。

---

## 7. Shallow to Deep: Aligning Token Pruning with Stage-wise Roles in LVLMs

**arXiv ID:** 2609.25635 | [PDF](https://arxiv.org/pdf/2609.25635v1)

**作者:** Shuo Zhang `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一种分层视觉令牌剪枝框架（STD），在大型视觉‑语言模型的视觉编码器中实现无训练、可插拔的令牌压缩。

**💡 创新点**

创新点在于将剪枝策略与网络层的功能角色对齐：浅层使用高频谱分析保留边缘信息，中层采用高斯平滑注意力保持空间连贯，深层通过语义稳定性触发器仅在语义稳定阶段剪枝。

**🔧 技术方法**

技术实现包括：无训练注意力权重剪枝、FFT高频谱分析、1D高斯平滑、语义变化度量（CLS-注意力差值）触发器，以及多尺度剪枝策略。

**📊 数据集**

实验数据集覆盖图像和视频理解：LLaVA‑1.5‑7B、LLaVA‑NeXT‑7B、Qwen2‑VL、InternVL3‑8B、Qwen3‑VL‑8B‑Instruct，以及 Video‑LLaVA，评测基准包括 GQA、MME、POPE、TextVQA、SQA、VQA‑Text 等。

**📈 对比分析**

与现有剪枝方法（PDrop、SparseVLM、FlowCut、FiCoCo‑V、V²Drop、DART）对比，STD 在 LLaVA‑1.5‑7B 上 64 令牌时提升 1.1%/1.8%，在 LLaVA‑NeXT‑7B 上 160 令牌提升 2.1%，并实现 3.9× 前置填充加速；在 Video‑LLaVA 上保持约 96‑97% 准确率，超越前置最佳 5.3%。

**⚠️ 局限性**

局限性包括：对视觉编码器结构的依赖，需手动调节 FFT、平滑、阈值等超参；仅解决视觉令牌冗余，未处理 KV‑缓存增长、LLM 解码延迟等瓶颈；实验主要在图像/视频任务，未验证在文档、机器人感知等更广泛多模态场景中的鲁棒性。

---

## 8. Do Existing Preconditioners Improve Biomedical Tabular Foundation Learning? An Empirical Study on TabPFN Optimization

**arXiv ID:** 2609.25013 | [PDF](https://arxiv.org/pdf/2609.25013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 9. Toolcompass: Guiding Tool Trialing, Not Suppressing It

**arXiv ID:** 2609.25678 | [PDF](https://arxiv.org/pdf/2609.25678v1)

**作者:** Junlin Fang `[一作]` (Nanyang Technological University), Sean Du `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于 von Mises–Fisher (vMF) 分布的后训练框架，通过将工具调用的表示投影到单位球面并对同一功能的调用聚合至同一原型，指导工具试错过程，从而提升在未见工具环境中的表现。

**💡 创新点**

创新点在于：①仅利用已知工具的功能类别标签，无需真实调用轨迹或对未见工具的访问；②通过引入 vMF 变分目标同时降低同功能内的方差、增加不同功能间的角度分离，形成功能级别的表示结构；③该方法可以与多种现有后训练目标（如 GRPO、RFT、DMPO）无缝融合，且部署时无额外推理开销。

**🔧 技术方法**

核心技术包括：①工具调用的隐藏表示提取与归一化；②投影头将表示映射至单位球面；③vMF 分布建模、变分损失（函数内部聚合）与分离损失（功能间分离）；④指数移动平均 (EMA) 进行原型更新；⑤与原始后训练目标联合优化。

**📊 数据集**

在 AppWorld 与 FTRL 两大多轮工具使用基准上进行实验，分别使用 Qwen3.5-4B 与 Qwen3.5-9B 两个模型；在 AppWorld 上的 OOD 任务中，方法在 GRPO 基础上提升约 8.9% 的任务成功率，整体最高 72.36%；在 FTRL 上，提升约 9.25% 的 Solve‑F1 分数。

**📈 对比分析**

与多种基线（提示式模型、基本后训练、回合级监督、工具级 RL、OOV 泛化方法）对比，方法在 ID、OOD 与总体指标上均优于现有最佳基线。例如在 AppWorld 的 Qwen3.5‑9B 上，方法达到 72.36% 总任务成功率，优于 LOOP 的 68.26%；在 FTRL 上，整体 Solve‑F1 提升至 72.36%，高于 SEAL 的 71.29%。

**⚠️ 局限性**

局限性包括：①仍需手工标注已见工具的功能类别；②对功能未见的工具无法直接迁移，需依赖功能匹配；③在极端域差异或功能稀缺的场景中，vMF 结构的优势可能受限；④方法在训练阶段引入额外的投影头与原型维护，可能增加训练成本；⑤未在非 ReAct 或非可执行工具的真实世界环境中验证。

---

## 10. From Decorative to Load-Bearing: Task Difficulty Shapes the Causal Role of Chain-of-Thought

**arXiv ID:** 2609.25366 | [PDF](https://arxiv.org/pdf/2609.25366v1)

**作者:** Renee Jia `[一作]` (R2M AI), Di Mu `[通讯]` (R2M AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该研究提出了一种连续性因果测试协议，评估链式推理（CoT）是否真正影响答案。

**💡 创新点**

创新点在于通过单步扰动、截断链并强制模型继续，直接测量CoT的负载承载性，并揭示其随任务难度的连续梯度。

**🔧 技术方法**

使用的技术包括扰动策略（数值更改、操作交换、置信注入等）、Claude Haiku 判别器、线性/MLP 隐状态探针以及多方向加性引导实验。

**📊 数据集**

实验数据集涵盖 GSM8K、MMLU、BIG‑Bench Hard，并在 Gemma‑2‑9B‑IT、Llama‑3.1‑8B‑Instruct 与 DeepSeek‑R1‑Distill‑Qwen‑7B 三种模型上进行验证。

**📈 对比分析**

比较方法为对同一扰动类型下不同难度任务的错误传播率进行方差分解，结果显示任务难度解释了约99%方差；探针在第10‑28层可分别达到约80%、75%和86%的分类准确率。

**⚠️ 局限性**

局限性包括仅评估单步扰动、仅在 8‑9B 模型上实验、判别标签噪声、探针仅为可读性工具且加性干预难以改变行为，且数据集过滤可能引入偏差。

---

## 11. Case for Vehicle-Edge Collaborative Multi-Sensor Data Fusion for Autonomous Vehicle Teleoperation

**arXiv ID:** 2609.25304 | [PDF](https://arxiv.org/pdf/2609.25304v1)

**作者:** Qixin Zhang `[一作]` (University of Minnesota -- Twin Cities), Zhi-Li Zhang `[通讯]` (University of Minnesota -- Twin Cities)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fede83ac-7505-405f-ab37-e7284695c47f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种在5G上限带宽条件下，车辆与边缘协作的相机-激光雷达特征级融合框架SHARDED，用于自动驾驶遥操作（TOD）场景。

**💡 创新点**

创新点包括：① 在车辆和边缘之间拆分特征级融合模型，显著减少上行流量；② 采用网络感知自适应特征传输机制，根据实时带宽优先传输关键空间区域的特征；③ 引入延迟感知位置漂移补偿，解决多模态特征异步导致的几何误差。

**🔧 技术方法**

技术实现基于CMT（Cross-Modal Transformer）融合骨干，结合WebRTC视频流、特征压缩、实时带宽估计与位移补偿算法，并在车辆侧使用NVIDIA RTX 5000 SFF Ada进行特征编码，边缘侧使用RTX A6000 GPU完成后续融合与检测。

**📊 数据集**

实验使用nuScenes数据集（多摄像头与64通道激光雷达同步数据）以及真实5G网络测量轨迹（30小时以上，100k+样本），并在真实MNCAV平台上验证。

**📈 对比分析**

与两种基线（All-on-Vehicle AOV和All-on-Edge AOE）对比，SHARDED在保持mAP≈0.58的前提下，上行带宽平均减少约50%（最优时95%），端到端延迟低于AOV且稳定，距离误差相较AOE下降35-45%。

**⚠️ 局限性**

局限性：特征选择仍基于距离和物体类型，未充分利用网络波动的细粒度信息；缺乏对动态网络突发状况的鲁棒自适应；未进行人因实验验证操作员主观体验。

---

## 12. Cloud, Edge, or Split? Profiling Onboard and Split Vision-Language Model Deployment for Drone AI

**arXiv ID:** 2609.25415 | [PDF](https://arxiv.org/pdf/2609.25415v1)

**作者:** Zoha Azimi `[一作]` (University of Klagenfurt), Christian Timmerer `[通讯]` (University of Klagenfurt)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对无人机上视觉语言模型（VLM）部署的三种方案（完全本地、完全云端、分布式）进行系统评估，量化推理时延、资源占用、通信负荷和能耗，探讨其在不同网络与图像分辨率条件下的性能。

**💡 创新点**

提供首个针对轻量级 VLM（SmolVLM‑256M）在 UAV 上的端到端基准，系统比较三种部署模式，并揭示其在带宽、分辨率与资源受限之间的交叉最优点，为自适应部署提供依据。

**🔧 技术方法**

采用 NVIDIA Jetson 边缘平台与 RTX A6000 云服务器，使用量化视觉嵌入（INT8）与 JPEG 压缩；通过实时 4G/LTE 连接仿真、能耗监测（RAPL、PyNVML）及多维度性能指标收集。

**📊 数据集**

采用电力线路 UAV 视频数据集（70帧）和 CODEBRIM 桥梁缺陷图像数据集（150张），并配以 18 个自然语言检索提示，覆盖组件识别、上下文感知、缺陷检测与定位。

**📈 对比分析**

通过对推理时延、传输 payload、CPU/GPU/内存占用及能耗进行均值/标准差统计，发现完全云端在高带宽下时延最低（约0.77 s），完全本地在低带宽下最稳健（约2.77 s），分布式方案在内存与能耗上显著减小（约15 mJ 本地），但整体时延不优。

**⚠️ 局限性**

未将推理准确率纳入权衡；实验仅基于单一轻量 VLM 与固定量化策略；缺乏实时动态切换机制；高分辨率下视觉嵌入大小与图像内容变化未进一步细化。

---

## 13. GRADE-RTL: Evaluating LLM-Generated RTL Beyond Compilation

**arXiv ID:** 2609.25335 | [PDF](https://arxiv.org/pdf/2609.25335v1)

**作者:** Hepziba Susan `[一作]` (Vellore Institute of Technology), Zain Ul Abideen `[通讯]` (University of Idaho)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向边缘硬件IP的LLM生成RTL评估框架GRADE-RTL，覆盖从接口、语法到功能和实现质量的五阶段验证流程。

**💡 创新点**

创新点在于将前端结构完整性、功能等价与后端FPGA/ASIC PPA评估相结合，设立可重复的边界修订机制，并提供公开的基准与工具链。

**🔧 技术方法**

采用LLM提示、五阶段验证脚本（PPS, CR, ER, MC, FE）、等价检查、FPGA Vivado 与 65nm Cadence Genus 合成及 Innovus 路由等技术。

**📊 数据集**

使用10个基于OpenCores的多模块IP（如PID控制器、AES-128、UART 16550 等）以及对应的自然语言规范作为基准数据集。

**📈 对比分析**

通过对9个LLM（OriGen、RTLCoder、VeriGen、VeriSeek、CodeV、Gemini、DeepSeek、GPT‑5、Claude）进行统一评测，报告E2E@1、E2E@K、SEY@K及前端成功率，结果显示Claude在功能和实现质量上最优，但整体成功率仍低。

**⚠️ 局限性**

局限性包括可能与模型训练集重叠、仅给定3次修订预算、仅做一次合成/路由演示、每个模型仅单一路径评测，未覆盖更大系统或多工具复制。

---

## 14. RGSQ: Riemannian Geometry-Sensitive Quantization for Large Vision-Language Models

**arXiv ID:** 2609.25492 | [PDF](https://arxiv.org/pdf/2609.25492v1)

**作者:** Zhiping Wu `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对大规模视觉‑语言模型的后训练量化方法RGSQ，利用黎曼几何框架实现多模态感知的量化优化。

**💡 创新点**

创新点在于构造统一的模态感知Kronecker‑Fisher黎曼度量、引入稀疏Givens旋转以引导量化误差朝低敏感度方向，并通过白化把黎曼目标转化为标准欧氏问题，从而兼容现有PTQ工具。

**🔧 技术方法**

主要技术包括信息几何（Fisher信息矩阵与Kronecker分解）、黎曼距离量化、稀疏Givens旋转优化以及白化映射。

**📊 数据集**

使用COCO Caption进行校准，评估涵盖OCRBench、TextVQA、VizWiz、SEED-Bench、ScienceQA、MMMU、SEED-Video与Video‑MME等多模态基准。

**📈 对比分析**

与RTN、GPTQ、AWQ、SmoothQuant及VLM特定的MBQ、MQuant等基线比较，RGSQ在W4A8/W2A8等低位宽场景下平均提升10–20%准确率，甚至逼近FP16表现；在大模型（如72B）和极低位宽下表现尤为突出。

**⚠️ 局限性**

局限性包括相对较高的离线校准开销（需要估计模态分解的Fisher统计和Givens搜索），对视频模型的泛化仍需进一步验证，且在极端任务上仍有微小差距。

---

## 15. Learning from Humans for Proactive Assistance in Human-Robot Collaborative Transport

**arXiv ID:** 2609.25351 | [PDF](https://arxiv.org/pdf/2609.25351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 16. An Empirical Analysis of Cross-OS Portability Issues in Python Projects

**arXiv ID:** 2609.25531 | [PDF](https://arxiv.org/pdf/2609.25531v1)

**作者:** Denini Silva `[一作]` (Federal University of Pernambuco), Marcelo d'Amorim `[通讯]` (North Carolina State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过跨操作系统测试重现和问题挖掘，系统评估Python项目的可移植性缺陷，并构建了7类24子类的完整分类体系。

**💡 创新点**

首次大规模量化Python跨平台可移植性问题，提出了四种通用修复模式，并验证了LLM在检测和修复方面的可行性。

**🔧 技术方法**

采用GitHub Actions虚拟机进行跨平台测试，结合静态分析工具（Ruff等）和三种大语言模型（GPT‑4o Mini、Llama‑3.3、Grok‑4‑Fast）进行检测和自动修复。

**📊 数据集**

使用900个开源Python仓库（涵盖不同规模和领域）的测试套件及其GitHub Issue进行实验。

**📈 对比分析**

与静态分析工具对比，LLM检测准确率在40%–79%之间；在自动修复上，通用提示下准确率为27%–43%，结构化提示下提升到50%–77%。

**⚠️ 局限性**

仅覆盖Linux、macOS和Windows，未考虑不同Python版本、硬件架构；实验依赖GitHub Actions的虚拟机环境，且问题挖掘为人工分析，存在主观性。

---

## 17. REDACT: Robust Perceptive Locomotion under Unseen Visual Corruption

**arXiv ID:** 2609.25450 | [PDF](https://arxiv.org/pdf/2609.25450v1)

**作者:** Natapat Kirdwichai `[一作]` (University of Southampton), Danesh Tarapore `[通讯]` (University of Southampton)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为REDACT的框架，旨在让深度条件行走策略在未见过的视觉退化环境下仍能保持有效的地形信息并完成复杂越障任务。

**💡 创新点**

创新点在于：①结合改进的残差视觉编码器和特征归一化来提升对噪声和遮挡的鲁棒性；②提出基于约束的多查询注意力教师-学生架构，减少对不可用深度细节的依赖；③设计基于一致性判定的门控机制（consensus gating），通过在无腐败样本上进行近似共形校准，实现对未知退化的自适应特征剔除；④在训练时使用持续特征掩码，使策略在部分特征缺失时仍能正常工作。

**🔧 技术方法**

技术方法包括：深度残差网络（IMPALA式）、Group Normalization、DropBlock与随机单元掩码、局部预测一致性损失、离线共形校准、GRU时序融合、基于离散化量化的视觉潜在表示。

**📊 数据集**

使用的数据集为在Isaac Gym中对Unitree Go2机器人进行的仿真训练，包含多种地形（斜坡、障碍、空洞、台阶），并在训练与测试阶段加入多种深度腐败（高斯噪声、飞行像素、Perlin dropout、藤蔓遮挡、柱子干扰）以及真实场景的RealSense D435i深度图。

**📈 对比分析**

与基线（Extreme Parkour、REAL）以及仅进行深度增广的ConvNet进行对比。REDACT在未见过的腐败条件下的成功率均显著高于基线（例如在飞行像素/高斯噪声下相对 74% vs 37%），并在实际林地与结构化障碍上实现零射击转移，完成率可达 70% 以上。

**⚠️ 局限性**

局限性包括：①共形校准仅基于干净样本，对完全分布式的退化（如全局高斯噪声）检测能力有限；②门控机制在大量特征被遮挡时会退化，可能导致信息丢失；③对新型硬件/传感器的泛化仍需进一步验证；④训练过程对计算资源依赖较高，尤其是多查询注意力与持续掩码的实现。

---

## 18. From Tone to Trajectory: Continuous Sentiment and the Shape of Monetary Policy Communication

**arXiv ID:** 2609.25034 | [PDF](https://arxiv.org/pdf/2609.25034v1)

**作者:** Martin Feldkircher `[一作]` (Vienna School of International Studies), Kristoffer Laigaard Nielbo `[通讯]` (School of Culture and Society - Center for Humanities Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建中央银行公开发言的情绪弧线（情绪随文本进展的连续变化），对欧洲央行（ECB）和美联储（Fed）的政策声明进行分析，并检验弧线形状在预测利率决策、通胀预期以及专业投预测差异中的信息含量。

**💡 创新点**

创新点在于：①将情绪视为随时间演变的序列而非单一平均值，揭示情绪弧形对政策信号的决定性作用；②使用概念向量投影（CVP）在上下文感知的词嵌入空间上计算连续情绪得分；③将情绪弧线拆解为Nelson‑Siegel层级（水平、斜率、曲率）和Hurst指数等特征，并对不同经济周期（危机与非危机）进行制度差异检验。

**🔧 技术方法**

主要技术包括：概念向量投影（CVP）与大规模Transformer嵌入（如XLM‑RoBERTa），LOESS平滑，Nelson‑Siegel分解，Hurst指数估计，以及多元回归与稳健性检验。

**📊 数据集**

使用的数据集为：①ECB介绍性声明（269条，1998‑2025）；②Fed FOMC声明（219条，2000‑2025）；③政策利率数据（ECB的MRO/DPR，Fed的联邦基金率）；④通胀预期数据（Consensus Economics一年的通胀预期）；⑤通胀率（欧盟HICP与美国CPI）。

**📈 对比分析**

与传统的词典情绪评分（Picault‑Renault）相比，弧线特征显著提升预测精度：ECB利率变动的调整后R²从0.07提升至0.13；Fed则从0.09提升至0.29；在危机期间，Fed的弧线特征表现出显著的制度差异。上下文弧线的解释力明显高于句子级弧线（ECB 0.205对0.128，Fed 0.420对0.258）。

**⚠️ 局限性**

局限性包括：①仅考察ECB和Fed两家央行，结果可能不具普适性；②情绪维度限制为货币、经济和不确定性，忽略了其他可能的语义维度；③CVP依赖预设的种子短语，可能存在文化或语言偏差；④对专业投预测者的效果有限，未涵盖普通公众或资产价格的即时反应；⑤危机期间的样本量相对较小，导致估计不稳。

---

## 19. GINIO: A Geometric SO(3)-Equivariant Interface for Neural Inertial Odometry

**arXiv ID:** 2609.25338 | [PDF](https://arxiv.org/pdf/2609.25338v1)

**作者:** Chankyo Kim `[一作]` (University of Michigan), Maani Ghaffari `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GINIO，一种几何SO(3)-等变接口，用于神经惯导测量，使得学习到的运动量和协方差在任意IMU安装旋转下保持正确的向量与张量变换；同时给出了Last-Frame Alignment预处理和谱协方差预测。

**💡 创新点**

创新点在于：①在整个神经网络接口层面实现完整的SO(3)等变性，保证运动测量与不确定性同时满足向量与共轭变换；②证明Last-Frame Alignment在等变网络下与世界框架训练等价；③设计谱协方差头，使不确定性保持张量一致性；④在多种运动场景（人类运动、无人机飞行、物理重新安装）中验证等变接口的通用鲁棒性。

**🔧 技术方法**

使用技术包括：SO(3)-等变神经网络（向量神经元、组卷积、张量积层）、Last-Frame Alignment（相对帧预处理）、谱协方差预测（通过SVD投影得到SO(3)正交矩阵并构造协方差）、滤波连接（SCEKF/ InEKF）、以及对比实验与 ablation。

**📊 数据集**

使用的数据集：TLIO、RIDI（人类运动惯导基准），NanoBench（Crazyflie 2.1 小四轴飞行器）以及 Fetch 机器人物理重新安装数据集；附录中还有 AquaticVision 视觉退化测试。

**📈 对比分析**

与传统 ResNet+SCEKF、EqNIO 等基线对比，GINIO 在不同场景均表现出更低的累计定位误差（如 TLIO 上 ID/SO(3) ATE 从 76.389 m 降至 2.018 m；NanoBench 上 ATE 从 5.579 m 降至 1.430 m；Fetch 重新安装 ATE 从 8.151 m 降至 0.495 m），同时参数量和 FLOPs 也大幅减少；在不旋转条件下的名义精度也保持相近或更优。

**⚠️ 局限性**

局限性：仅对校准后、偏置校正过的向量测量在坐标框架旋转下提供等变保证，无法处理物理重新安装导致的偏置移位、标定误差、杠杆臂效应、振动、时序偏差等；Last-Frame Alignment 依赖相对姿态估计，若姿态误差过大会影响性能；此外仍需在估计器侧进行标定与鲁棒性补偿。

---

## 20. Same Quantity, Different Answer: Numerical Representation Invariance in Language Models

**arXiv ID:** 2609.25009 | [PDF](https://arxiv.org/pdf/2609.25009v1)

**作者:** Ephraim Atta-Duncan `[一作]` `[通讯]`, Ephraim Atta-Duncan

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套3,600个带有精确有理答案的数学单词问题，并通过五种变形（小数/分数、科学计数法、百分比/小数、数字/数字词、单位转换）生成8,600个等价提示，评估五个公开大模型在严格解析器与审计层下的数值正确性与不变性。

**💡 创新点**

提出了“等价轨道”框架：用类型化语义程序保证答案不变，分离严格解析器与审计规范，量化轨道正确率、轨道不变性和一致错误率；首次在大模型上系统性检验数值表示不变性，并揭示解析器边界与单元转换错误的隐藏问题。

**🔧 技术方法**

使用了精确有理数生成、可变提示渲染、两层解析（严格+审计）、Bootstrap/Wilson区间估计、McNemar检验、AURC等统计手段；实现了统一的提示模板和可重复的评测流程。

**📊 数据集**

自制的“Exact-Rational Benchmark”：3,600条基础问题与8,600条等价提示，包含六种模板（如体积、差距、长度、容量、速率、体积加法），每种变形约1,000条提示；数据完全可复现，已发布冻结档案。

**📈 对比分析**

与五个系统（Qwen3.5-4B/9B Q4/Q8、GPT-OSS-20B、Mistral Small 4）进行对比；严格解析层下所有系统满足 H1/H2，审计层下轨道正确率在84.8–98.1%之间；Mistral在单位转换上错误率高达28.1%；对比表明审计后准确率提升至≈0.97–0.99；表示一致性与范例一致性在有限子集上无显著优势。

**⚠️ 局限性**

局限包括：仅使用合成模板，未覆盖自然文本；各变形族与模板不完全交叉；仅评估五个量化模型，缺乏更大规模或多语言覆盖；解析器与提示规范不匹配导致严格层失效；审计只覆盖固定重写规则，可能漏检其它合法答案；在小样本子集中表示一致性实验受限，难以推广。

---

## 21. Modular Composition of Inductive Types Using Lean Meta-programming

**arXiv ID:** 2609.25427 | [PDF](https://arxiv.org/pdf/2609.25427v1)

**作者:** Ramy Shahin `[一作]` `[通讯]` (Qualgebra), Ramy Shahin (Qualgebra)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

实现了基于Lean元编程的递归类型与函数的组合框架，支持对已有的单独定义的Inductive类型、函数及定理进行自动组合，生成语义子类型和强制转换；

**💡 创新点**

提出了可直接对已定义的Lean递归类型和对应的匹配函数进行语义子类型化与合并的算法，并通过元编程自动生成兼容的Coercion和依赖Coercion；

**🔧 技术方法**

使用Lean 4元编程框架（用户自定义语法扩展、elaborator、类型推导）、算法化的类型与函数组合、Coercion与CoercionDep实现；

**📊 数据集**

无公开数据集，使用人工构造的案例（K3三值逻辑和Typed Lambda Calculus的Boolean、Nat、STLC子语言）进行验证；

**📈 对比分析**

与单块模型对比，展示可重用性与模块化提升，但论文未给出定量性能指标，主要以案例演示和功能覆盖为评估；

**⚠️ 局限性**

仅支持简单递归Inductive类型，不能处理嵌套或族化类型；不支持互递归函数；需要手动组合单独定义；对非Inductive类型的改动手工处理；递归案例的符号生成尚未实现。

---

## 22. OSFoundry: Building and Evolving Operating Systems with Specification-Guided Agents

**arXiv ID:** 2609.25018 | [PDF](https://arxiv.org/pdf/2609.25018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 23. Co-Fabric: Breaking Host-Domain Boundaries for Unified xPU Interconnection

**arXiv ID:** 2609.25560 | [PDF](https://arxiv.org/pdf/2609.25560v1)

**作者:** Zhen Peng `[一作]` (IEIT SYSTEMS Co., Ltd.), Yue Yuan `[通讯]` (IEIT SYSTEMS Co., Ltd.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种跨主机域边界的统一 AI 超模组互连体系 Co‑Fabric，并在 64‑xPU 3D‑Mesh 系统上实现了高带宽、低延迟的互连和统一的内存地址空间。

**💡 创新点**

创新点包括：四层精简协议栈（媒体层、链路层、Fabric 层、语义层）实现纳秒级处理延迟；Fabric 层使用端口 ID 进行硬件级路由，支持跨域点对点通信；通过影子设备自枚举构建全局平坦地址空间，让所有主机都能像单机一样访问远程 xPU 内存。

**🔧 技术方法**

主要技术包括：基于 PCIe 兼容物理链路的 Retimer + Co‑Fabric Switch；信用计数流控和链路级重传保证可靠性；Fabric 层内嵌路由表和端口 ID；影子设备自动枚举和全局地址映射；硬件实现的 INC（加/最小/最大）运算以完成跨域归约。

**📊 数据集**

评测数据集主要使用大模型推理任务 DeepSeek R1（671B 参数、INT8 推理），并在 LoRA 微调、QwQ‑32B、Bio‑Genetic 等模型上进行规模线性测试。

**📈 对比分析**

对比基线 RoCE（Ethernet+RDMA）互连，Co‑Fabric 在 64‑xPU 系统中实现：AllReduce 延迟仅为 RoCE 的 10%–20%，带宽提升 5–15 倍；推理吞吐提升 30%–80%（最高 1.8×）；成本下降 80%，功耗降低 5%。

**⚠️ 局限性**

局限性：目前实验规模限制在 64‑xPU 双机柜；对极大多机柜部署的可靠性、可维护性和软件生态（驱动、SDK）仍待进一步验证；依赖专有 Retimer/Co‑Fabric Switch，硬件成本与通用化水平尚待提升。

---

## 24. Graph Domain Adaptation Does Not End with Representation Learning

**arXiv ID:** 2609.25692 | [PDF](https://arxiv.org/pdf/2609.25692v1)

**作者:** Ziqian Liu `[一作]` (Sun Yat-sen University), Maolin Wang `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出EviGDA框架，在图域适配中同时使用图感知专家和无图本地专家，独立训练后在推理时以任务级概率混合；

**💡 创新点**

创新点在于引入无图本地专家提供非邻域证据、熵感知的 exact‑sampling 对齐方法以及独立训练+后期融合策略，突破单一图表示预测的局限；

**🔧 技术方法**

使用A2GNN消息传递、双层MLP本地专家、核均值差异（MMD）与熵加权采样、任务级概率融合；

**📊 数据集**

在十个预处理图（Citation、Airport、Blog、Twitch 共四族）上进行 16 个源→目标迁移任务；

**📈 对比分析**

与 16 种基线（GCN、UDA‑GCN、GRADE、PairAlign、GraphAlign、A2GNN 等）对比，EviGDA 在 15/16 任务上取得最高 Macro‑F1（全 16 任务 67.67%），平均提升 1.98 点；

**⚠️ 局限性**

仅针对静态单源图域适配，未考虑多源、增量或动态图场景，未来工作需扩展至多源与动态图。

---

## 25. SBMVTrack: Spike-Budgeted Multi-View Learning for Energy-Efficient UAV Tracking

**arXiv ID:** 2609.25503 | [PDF](https://arxiv.org/pdf/2609.25503v1)

**作者:** Pengzhi Zhong `[一作]` (Guilin University Of Technology), Shuiwang Li `[通讯]` (Guilin University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于脉冲神经网络的UAV视觉跟踪框架SBMVTrack，专注于能耗和精度的权衡；

**💡 创新点**

通过能量加权脉冲预算(EWSB)显式控制层级脉冲活动，并利用掩码多视角目标建模(MVTM)提升目标表示鲁棒性；

**🔧 技术方法**

使用全脉冲网络结构（E-SpikeFormer）、NI-LIF神经元、随机补丁模块、能量加权预算损失和跨视角重建/一致性学习；

**📊 数据集**

训练集采用LaSOT、GOT‑10k、COCO、TrackingNet，评估集包括UAVTrack112、UAVDT、VisDrone2018和UAV123；

**📈 对比分析**

与15种DCF、CNN、ViT以及SNN跟踪器对比，SBMVTrack在精度和成功率上均名列前茅，同时理论能耗下降约48%（从8.1 mJ降至4.4 mJ），在CPU上可达33.3 FPS；

**⚠️ 局限性**

仅基于理论能耗估计，实际能耗受硬件架构、内存访问等因素影响，未来需在神经形态硬件上进一步验证。

---

## 26. Indirect tipping: a social attack surface in AI agent populations

**arXiv ID:** 2609.25194 | [PDF](https://arxiv.org/pdf/2609.25194v1)

**作者:** Ariel Flint `[一作]` (University of London), Andrea Baronchelli `[通讯]` (University of London)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对大语言模型(agent)在命名游戏中的集体决策进行实验和理论建模，研究了不同协调均衡之间的临界质量阈值及其对系统安全性的影响。

**💡 创新点**

创新点在于将临界质量视为相对关系而非固有属性，构建了一个带有指向和权重的均衡网络，揭示了间接路径、门控均衡和桥接转移在降低攻击成本中的关键作用，并探讨了多样性与时序对转移成本的影响。

**🔧 技术方法**

采用基于LLM的生成式代理模型（命名游戏）、基于平均场理论的解析模型、策略提取与大规模仿真相结合的技术框架，并引入了多阶段路径优化与转移图分析。

**📊 数据集**

实验数据来自对四大开源LLM（Qwen、Phi、Llama、DeepSeek）在八种情感标签（亦测试颜色、形状、字母）的人工构造交互模拟，未使用外部真实数据集。

**📈 对比分析**

通过与均值场预测和随机搜索的二元比较，验证了理论阈值与模拟结果高度一致；实验表明，间接路径可将所需承诺比例降低至原始阈值的不到一半，且多样性与短暂承诺能进一步压缩成本。

**⚠️ 局限性**

局限在于仅考察有限、相互等价的均衡集合，假设群体同质、网络完全连接，未考虑动态拓扑、异构模型、收益不平衡以及开放式语义空间；这些因素在实际部署中可能显著改变安全评估。

---

## 27. Objective Video Quality Assessment in FWA-Based Over-the-Top Content Delivery Across Open Source 5G Networks

**arXiv ID:** 2609.25423 | [PDF](https://arxiv.org/pdf/2609.25423v1)

**作者:** Nelson Ion `[一作]` (Federal University of Rio Grande do Norte), Vicente Sousa `[通讯]` (Federal University of Rio Grande do Norte)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在真实5G FWA网络上对UHD视频进行实验性QoE评估

**💡 创新点**

首次结合自动化DASH流媒体、完整参考VQA指标和实际网络负载，实证证明MIMO与高阶调制对QoE的决定性作用

**🔧 技术方法**

使用srsRAN、Open5GS、USRP B210、MPEG‑DASH、FFmpeg、Grafana等开源工具，计算PSNR、SSIM、VMAF并采集吞吐量

**📊 数据集**

以Big Buck Bunny的4K 60fps视频为测试集，编码成多分辨率/码率组合

**📈 对比分析**

通过比较SISO/64‑QAM、SISO/256‑QAM、MIMO/64‑QAM、MIMO/256‑QAM在1~5用户下的吞吐量与VMAF等指标，MIMO+256‑QAM在五用户时仍保持VMAF>93%，而SISO明显下降

**⚠️ 局限性**

仅测试单一内容和静态信道，未考虑移动、极端干扰或多频段情况，且结果仅适用于该特定硬件配置和频段

---

## 28. How Early Can You Tell? Early Eye Gaze Dynamics and Cybersickness Progression in Virtual Reality

**arXiv ID:** 2609.25422 | [PDF](https://arxiv.org/pdf/2609.25422v1)

**作者:** Nevzat Umut Demirseren `[一作]` (Texas State University), Isayas Berhe Adhanom `[通讯]` (Texas State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文分析了VR初期1–5分钟眼动轨迹与后续不适进展的关系，重点考察了聚焦散布等眼动指标与持续自评不适曲线及事后SSQ分数的关联。

**💡 创新点**

首次系统检验早期眼动轨迹（尤其聚焦散布）对不适进展的预测能力，并发现其与最终严重度无显著关系，提供了新的眼动预测视角。

**🔧 技术方法**

使用HTC Vive Pro Eye眼动采集（120Hz）配合I-VT算法提取事件级指标，并通过线性回归与Mann-Whitney U检验评估关系。

**📊 数据集**

基于之前公开的20分钟VR导航实验数据，19名参与者在现实环境中的眼动与连续不适评分。

**📈 对比分析**

对不同早期时长（1–5分钟）计算眼动斜率，并与后续不适斜率或归一化进展指数回归；聚焦散布在1–4分钟内R²最高达0.55，其他指标无显著关联；未出现显著组间差异。

**⚠️ 局限性**

样本量小、仅单一VR场景、未剔除头动对眼动的影响、线性轨迹假设可能遗漏非线性动态。

---

## 29. Prophet Inequalities Beyond Utilitarian Social Welfare

**arXiv ID:** 2609.25424 | [PDF](https://arxiv.org/pdf/2609.25424v1)

**作者:** Daniel Halpern `[一作]` (Google Research), Alexandros Psomas `[通讯]` (Purdue University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在离散物品的在线分配中，研究了以p-平均公平性为目标的预言家问题，并将大规模物品情形映射为单物品问题。

**💡 创新点**

提出了量化公平性的通用p-平均指标，并证明无论p≤1，在线策略与最优预言家的比值下界为约0.7059；同时给出匹配的上界，表明在i.i.d.设置下仅损失约4个百分点。

**🔧 技术方法**

使用量化阈值策略、凸分析、变分不等式和支点表示等理论工具，构造最优阈值规则并求解极值问题。

**📊 数据集**

该工作为理论性研究，无实验数据或公开数据集；所用模型基于独立同分布的价值分布。

**📈 对比分析**

通过与传统的预言家常数0.7451比较，展示了在公平目标下的在线竞争比率约为0.7059；在物品数量足够多时，ex‑ante与ex‑post结果几乎相同。

**⚠️ 局限性**

局限性在于仅适用于i.i.d.加法值分布、需要大量物品；当物品数等于代理人数时，公平竞争比率趋于零；实验验证缺失。

---

## 30. Correcting Within-Group Self-Selection Bias in Prioritized Replay

**arXiv ID:** 2609.25297 | [PDF](https://arxiv.org/pdf/2609.25297v1)

**作者:** Oscar Miró López-Feliu `[一作]` (University of Amsterdam), Herke van Hoof `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文研究并改进了优先经验回放（PER）中因同一状态动作组内随机结果偏向导致的样本分布失衡，提出了在保持组级优先级的同时，用同级子样本均匀重采样或平均目标的方法（Sibling-aware Replay）。

**💡 创新点**

创新点在于将PER拆解为组间分配与组内自选两层，提出了三种在组内恢复真实样本频率的修正策略：统一子样本采样（SAMPLE）、目标均值（AVG）和完整结果表采样（MODEL），并证明其在固定缓冲区与优先级下的理论性质。

**🔧 技术方法**

使用的技术包括：优先经验回放（PER）、重要性采样（IS）、固定抽样器（VQ‑VAE）生成的离散潜在键、TD误差更新、以及针对不同环境的函数近似与离散抽象。

**📊 数据集**

主要数据集包括：自定义离散随机环境（OutlierBandit、TwoChains、FrozenLake系列）以及基于Atari的轻量级环境MinAtar（Asterix、Breakout、Freeway、Seaquest、SpaceInvaders）。

**📈 对比分析**

比较方法：与标准DDQN、PER、以及三种Sibling-aware Replay方式进行对比。实验表明：在存在罕见高幅度奖励的环境中，SAMPLE和MODEL显著提升学习效率（AUC和最终成功率提高），尤其在FrozenLake-H100/H300等极端稀疏奖励任务中；在MinAtar的“尾部”扰动设置下，SAMPLE在多数游戏中优于PER，且与DDQN接近。

**⚠️ 局限性**

局限性：①需要预先定义或学习有效的状态动作组（精确或潜在），若组过于稀疏或误差主要来自可约错误则改进有限；②MODEL方法要求完整的结果表，仅适用于小规模离散动作；③在参数调优（α,β）对比下，某些环境（如FrozenLake）对组分配与权重仍高度敏感，调优难度较大；④未深入探讨噪声辨识与组级优先级协同的可能性。

---

## 31. "As a Language Model...": Chat Template Switches LLM Self-Referential Voice and Activation Steering Reproduces It

**arXiv ID:** 2609.25021 | [PDF](https://arxiv.org/pdf/2609.25021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. MIISO: Modal Integrators for Isogeometric Stabilization of Outliers

**arXiv ID:** 2609.25087 | [PDF](https://arxiv.org/pdf/2609.25087v1)

**作者:** Sreeram Shankar `[一作]` `[通讯]` (University of Texas at Austin), Sreeram Shankar (University of Texas at Austin)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为高阶等距离散化的无物理离群模式提出一种时间积分方法

**💡 创新点**

将离群模式的抑制通过时间积分器中的选择性消减实现，无需修改空间离散化

**🔧 技术方法**

使用改进的指数Rosenbrock–Krylov积分器，结合非holomorphic Padé逼近的混合放大因子和Arnoldi Krylov子空间逼近

**📊 数据集**

采用等距离散化（C^p-1 B-spline）在1D杆、2D梁、3D立方体等不同几何及边界条件下的离群谱，测试线性弹性和非线性弹性膜

**📈 对比分析**

与传统的Generalized-α方法以及空间子空间投影法对比；MIISO在消除离群模式、保持物理模式无耗散、实现2-4阶全局收敛的同时，计算成本低于Generalized-α，误差匹配时性能更优

**⚠️ 局限性**

离群模式跟踪失效时会出现模式误追，退耦时需要额外的特征值求解或组装成本；对极小步长的消除能力随步长降低；目前仅针对等距离散化，未给出刚性阶理论和并行实现细节

---

## 33. Deep Reinforcement Learning on Item-Compatibility Graphs for One-Dimensional Bin Packing

**arXiv ID:** 2609.25397 | [PDF](https://arxiv.org/pdf/2609.25397v1)

**作者:** M. Aslı Aydın `[一作]` `[通讯]` (Bahceşehir University), M. Aslı Aydın (Bahceşehir University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于项兼容图的单一大小不变图神经网络强化学习框架，用于一次维度箱子装填问题（1D-BPP）的端到端构造解法。

**💡 创新点**

创新点在于将装填过程视为在项兼容图上的马尔可夫决策过程，每一步合并一条兼容边，且使用规模无关的节点特征和图卷积实现能直接在任意规模实例上进行零样本推理。

**🔧 技术方法**

使用图神经网络（GCN）、策略梯度强化学习（PPO）作为演员-评论家网络，并在推理阶段采用随机束搜索解码；实验中比较了不同图编码器、奖励设计、训练分布及超参。

**📊 数据集**

在BPPLIB公开库的9个完整族（共1,615个实例）上进行评估，训练样本为50个随机权重的合成实例。

**📈 对比分析**

与传统构造启发式（FFD）、分组遗传算法（GGA）以及近期学习方法（BGCN、PTR、HRL-GPN、RRMCTS）比较，零样本性能在5个族上优于FFD，平均最优缺口从2.66%降至2.31%，与列生成学习方法在3个族上相当，稳定性优于层次RL和MCTS基线，但整体仍落后于GGA。

**⚠️ 局限性**

局限在于对局部二元合并决策的细粒度控制导致搜索开销较大，推理速度低于高度优化的启发式；且对低头寸或几乎完全兼容图的实例效果有限，仍难以完全替代高质量的分组遗传搜索。

---

## 34. When the Strike Zone Becomes Algorithmic: Umpire Judgment and Player Challenge Decisions under AI Review

**arXiv ID:** 2609.25525 | [PDF](https://arxiv.org/pdf/2609.25525v1)

**作者:** Kichang Lee `[一作]` (KAIST), JeongGil Ko `[通讯]` (Yonsei University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2015–2026年MLB投球与2026年挑战记录进行分析，利用心理测量模型衡量裁判判罚边界与平滑度，并检验挑战后即时与长期的判断变化以及球员挑战决策与几何误差和可观信息的关联。

**💡 创新点**

首次将心理测量学方法与大型体育数据结合，系统评估了挑战式人工智能审判对裁判判断的即时调整、边界漂移及球员挑战策略的空间与信息维度差异，为“人机协同审判”提供可操作的经验与设计启示。

**🔧 技术方法**

使用二项式逻辑回归（带有赛季与裁判双向聚类的标准误），信号检测理论下的心理测量参数（α,σ）估计，条件逻辑模型与差分对照设计来分离纠正反馈效应，以及基于滑动窗口的概率模型预测挑战成功率。

**📊 数据集**

公开的MLB Statcast投球追踪数据（2015–2026），2026年挑战结果记录，All-Star选拔信息，用于构建参考区、计算边界距离以及生成挑战机会样本。

**📈 对比分析**

通过将2026年α,σ估计与2015–2025年线性趋势预测值对比（使用预测分布与置信区间），以及在游戏内外对比即时与后续判断变化；挑战决策与几何误差、可观信息的对应曲线显示，表明挑战率随可观成功概率上升但仍远低于1，显示出局限的即时纠正与策略保守性。

**⚠️ 局限性**

仅有2026年挑战数据，导致因果推断受限；模型假设简化裁判决策为单一阈值与平滑度；挑战机会样本量在某些边界/状态组不足；球员对几何的感知可能受未观测因素影响，限制了对挑战决策的完全解释。

---

## 35. FineWeb-CLaR: Culture, Language, and Region Annotations for Benchmark-Aligned Corpus Auditing

**arXiv ID:** 2609.25298 | [PDF](https://arxiv.org/pdf/2609.25298v1)

**作者:** Yusser Al Ghussin `[一作]` (Saarland University), Simon Ostermann `[通讯]` (Saarland University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

FineWeb-CLaR 在 FineWeb 与 FineWeb‑2 之上添加地区标签与基于文化分类法的主题标签，支持语言模型文化评估的审核。

**💡 创新点**

创新点在于构建统一的文化‑语言‑地区三轴标注体系，并通过多语言主题建模与 LLM 辩判生成 Locale Topic Distributions（LTD），实现语料与基准的直接对比。

**🔧 技术方法**

使用 URL 解析、BGE‑M3 多语言嵌入、FASTopic 主题建模、温度化余弦 Softmax 投影以及 LLM 辩判等技术。

**📊 数据集**

数据集为 FineWeb 与 FineWeb‑2（共 30.9 B 文档）以及 277 个文化 NLP 基准。

**📈 对比分析**

通过比较预训练语料中每个文化主题‑地区单元与基准覆盖情况，发现预训练证据覆盖所有单元，但 21% 的地区单元缺少基准；同一语言不同地区的 LTD 显著差异（JSD 远高于噪声阈值）。

**⚠️ 局限性**

局限包括 URL 归属噪声、文化分类法粗糙、主题建模与 LLM 判别误差、BGE‑M3 语言覆盖不足，以及低资源地区缺乏足够文档导致 LTD 不可靠。

---

## 36. Partition-Matched Evaluation of Community Features under Distribution Shift in Android Malware Function-Call Graphs

**arXiv ID:** 2609.25256 | [PDF](https://arxiv.org/pdf/2609.25256v1)

**作者:** Junru Zhu `[一作]` (Independent Researcher), Ruoyu Qi `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在分布偏移下，Android恶意软件函数调用图的社区特征对分类的影响。

**💡 创新点**

提出匹配随机分区控制，以区分社区结构与仅仅图形形状对模型的贡献。

**🔧 技术方法**

使用Leiden社区检测、LDP、本地和全局统计、匹配随机分区、元数据聚合以及MLP分类器。

**📊 数据集**

使用MalNet-Tiny、Common、Distinct 15000个函数调用图。

**📈 对比分析**

通过与LDP、全局统计和匹配随机分区控制比较，检测社区仅提升源域准确率但未显著改善跨域泛化，退化幅度与随机分区相当。

**⚠️ 局限性**

局限于单一社区检测参数、只评估全图描述符、未覆盖节点级或多分辨率社区，且缺乏跨数据集验证。

---

## 37. Combinatorial Network-Based Manifold Topological Deep Learning for Image Analysis

**arXiv ID:** 2609.25453 | [PDF](https://arxiv.org/pdf/2609.25453v1)

**作者:** Alice Wachira `[一作]` (University of Georgia), Guo-Wei Wei `[通讯]` (University of Georgia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于离散流形 Hodge 分解与组合复形神经网络（CCNN）相结合的 CNMTDL 框架，用以提取医学图像的几何拓扑特征并进行分类。

**💡 创新点**

创新点在于首次将离散 Hodge 分解得到的无旋、无散、谐波三组分嵌入组合复形结构，并在 0 细胞与 2 细胞之间通过注意力机制实现高阶信息传递，构成全新的拓扑保留深度学习管道。

**🔧 技术方法**

核心技术包括离散流形 Hodge 分解、组合复形神经网络（包含 0 细胞与 2 细胞分支）、图注意力层、多分支融合门控、以及基于 Patch 嵌入的卷积投影。

**📊 数据集**

使用 MedMNIST v2 benchmark，涵盖 6 个 2D 数据集（RetinaMNIST、DermaMNIST、BloodMNIST、OrganAMNIST、OrganCMNIST、OrganSMNIST）和 6 个 3D 数据集（FractureMNIST3D、NoduleMNIST3D、OrganMNIST3D、SynapseMNIST3D、VesselMNIST3D、AdrenalMNIST3D）。

**📈 对比分析**

与 ResNet、ViT、MedVIT、BSDA、C-Mixer 等多种基线模型在 AUC 与 ACC 上进行对比，CNMTDL 在大多数数据集上获得最高或接近最高的准确率（例如 BloodMNIST ACC≈0.99、AUC≈1.00；整体 2D ACC≈0.86、AUC≈0.97；3D ACC≈0.83、AUC≈0.91），显著提升了医学图像分类的性能。

**⚠️ 局限性**

局限性包括：在部分数据集上并未超越最优基线，未与 C-Mixer 进行直接比较；当前 CCNN 仅利用 0 细胞与 2 细胞，缺乏更高维细胞和跨维度信息融合；模型对极端类别不平衡的适应性仍待进一步验证。

---

## 38. Making Agents More Consistent: Skills Should Form Habits for Repeat Tasks

**arXiv ID:** 2609.25299 | [PDF](https://arxiv.org/pdf/2609.25299v1)

**作者:** Travis Weber `[一作]` (Pheo Inc), Rohit Taneja `[通讯]` (Pheo Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出“习惯形成”机制，通过从执行历史中挖掘候选技能并在四道门控下验证，构建可重复、可审计且成本更低的 agent 能力链。

**💡 创新点**

创新点在于：① 采用多门控（C0–C3）验证候选技能，避免替换导致的不确定性；② 基于执行轨迹的差异性检验（trace conformance），无须预设规范；③ 将决策路由与参数提取拆分为两步，显著提升准确率和效率；④ 对成本与一致性之间的权衡给出可量化的不等式。

**🔧 技术方法**

技术包括：技能线性结构（anchor + 变体）、覆盖谓词与 guard 机制、执行轨迹投影与加权编辑距离、一次性非劣性检验、shadow A/B 与人类复核、自动化候选生成（回放、扰动、有限仿真）、多任务训练（Claude、Kimi、GLM、Qwen 等）。

**📊 数据集**

使用的基准数据集：自构造的 text‑to‑SQL 任务（14 个模板 × 3 实例 × 3 重复，共 504 次调用）以及 HotpotQA 派生的检索路由任务（288 题）。

**📈 对比分析**

比较方法：与四种 reasoning arm（Claude Sonnet 5、Kimi K2、GLM‑4.6、Qwen3‑235B）进行非劣性检验；在 text‑to‑SQL 上测量一致性、token 费用、净收益；在检索路由上评估准确率、提升幅度及置信区间。性能方面，习惯版在 456 次重复调用中保持 100% 输出一致，token 费用下降 14%–56%，净收益在 7–53 次重用后为正；检索路由上在 200 条已挖掘的轨迹后准确率提升约 15%。

**⚠️ 局限性**

局限性：① guard 的误入率导致无推理步骤下的错误不可见；② trace conformance 对参数值不敏感，难以捕捉错误参数；③ 需要人工授权与复核，未能完全自动化；④ 在多技能集扩展时可能出现覆盖冲突与误入率上升；⑤ 目前只在有限的两类任务上验证，缺乏更广泛的实证与长期部署评估。

---

## 39. Linear-Query Deterministic Approximation for Non-monotone Submodular Maximization under a Knapsack Constraint

**arXiv ID:** 2609.25679 | [PDF](https://arxiv.org/pdf/2609.25679v1)

**作者:** Zihui Liu `[一作]` (Fuzhou University), Zhijie Zhang `[通讯]` (Fuzhou University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种确定性线性查询算法，解决非单调子模函数的背包约束最大化问题，达到 (1/4-ε) 的近似比率。

**💡 创新点**

创新点在于引入了残差预算枚举方法改进阈值-双贪婪框架，并将大成本最优元件的情形归约为双准则子模最大化，从而突破了先前 1/5-ε 的上限。

**🔧 技术方法**

采用了阈值-双贪婪（Threshold–Twin Greedy）技术、残差预算枚举、无约束子模最大化（USM）子程序以及双准则（bicriteria）归约等多种算法技术。

**📊 数据集**

本研究为理论性工作，未使用具体数据集进行实验验证，主要关注算法的查询复杂度和近似比率。

**📈 对比分析**

与现有最优随机算法（1/4-ε）实现相同的近似比率相比，确定性算法在保持线性查询复杂度（O(n log²(1/ε)/ε²)）的前提下实现了更优的理论效果；相较于之前的 1/5-ε 确定性算法，查询复杂度与近似比率都有明显提升。

**⚠️ 局限性**

限制在于仍需要 O(n log²(1/ε)/ε²) 次查询，且在某些特殊实例（如单个高成本元素）下的表现需要通过双准则近似来补救；此外，是否能进一步突破 1/4 的上限仍是未解问题。

---

## 40. Ovis-Embedding: Pushing the Frontiers of Universal Omni-Modal Embeddings

**arXiv ID:** 2609.25165 | [PDF](https://arxiv.org/pdf/2609.25165v1)

**作者:** Ovis-Embedding Team `[一作]` `[通讯]` (Alibaba Group), Ovis-Embedding Team (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Ovis-Embedding，一套基于 Qwen-Omni 预训练模型的统一多模态嵌入体系，可同时处理文本、图像、视频和音频，支持任意模态的检索与交叉模态检索。

**💡 创新点**

创新点包括：
1) 采用原生多模态理解模型 Qwen‑Omni 作为嵌入骨干，实现真正的多模态共享编码；
2) 构建大规模跨模态数据集，使用同源采样（homogeneous‑source sampling）产生任务一致的难负样本；
3) 在训练中引入难度感知焦点对比损失（focal contrastive loss）和嵌入蒸馏（embedding distillation）以保持细粒度相似性；
4) 推出低秩特征分解与残差适配器，实现可扩展的嵌入维度与高效推理。

**🔧 技术方法**

核心技术包括：低秩对比预训练、焦点对比损失、前向 KL 蒸馏、低秩特征分解、PCA+残差适配器、统一的最后层隐藏向量池化、跨模态 TMRoPE 时序对齐。

**📊 数据集**

使用了跨文本、图像、视频、音频、视觉文档、Agent 等多模态大规模数据集；训练语料约 5000 万（query, target）对，覆盖检索、分类、问答、定位、工具、GUI 等多任务；评测使用 MMEB‑v3/v2、MAEB、MVEB、RTEB 等公开基准。

**📈 对比分析**

与现有单模态与多模态嵌入模型（如 Qwen3‑VL‑Embedding、e5‑omni、LCO‑Embedding、VLM2Vec 等）在所有基准上均取得最高或接近最高分。Ovis‑Embedding‑Omni‑3B 在 MMEB‑v3 上领跑 58.46 分；Ovis‑Embedding‑VL‑9B 在 MMEB‑v2 上 81.13 分；在 MAEB 和 MVEB 上分别位居榜首；在 RTEB、MMEB‑Text 上也超过同等规模文本模型。

**⚠️ 局限性**

局限性：
1) 在极细粒度任务（如多条件检索、记忆检索）仍略逊于专门模型；
2) 语音分类和重排序等音频子任务未能取得最高分；
3) 由于采用大规模预训练模型，参数量仍偏大，对算力和存储有一定要求；
4) 对于新出现的模态或极端多模态交互，仍需进一步扩展与验证。

---

## 41. Ladders of Thought: A Self-Evolving Curriculum of Progressively Simplified Reasoning Traces

**arXiv ID:** 2609.25643 | [PDF](https://arxiv.org/pdf/2609.25643v1)

**作者:** Minghui Liu `[一作]` (University of Maryland), Furong Huang `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Ladders-of-Thought（LoT）框架，利用自动递进式问题重写与自适应自进化课程来提升小至中型LLM的推理能力。

**💡 创新点**

创新点包括：①通过逐步替换前置前提为中间结论生成语义保持的“难度阶梯”；②以最小推理步骤数作为统一的难度标注；③将每个难度层视为多臂老虎机（MAB）中的一臂，在线动态分配训练样本，实现真正自适应课程。

**🔧 技术方法**

核心技术为：链式推理（CoT）知识蒸馏、递进式问题重写、步骤数难度标注、基于MAB的自进化课程调度。

**📊 数据集**

主要使用的基准数据集有：数学推理（GSM8K、AddSub、ASDiv、MultiArith、SVAMP）和多跳推理（EntailmentBank、StrategyQA、OpenBookQA、QASC、MuSiQue），以及大规模模型评估的Qwen2.5-7B和Llama3.1-8B。

**📈 对比分析**

与基线（原始模型、CoT蒸馏、固定Easy→Hard或Hard→Easy课程）对比，LoT在多跳和数学推理任务上均显著提升pass@5准确率，尤其在AddSub、ASDiv、SVAMP等OOD算术任务上提升30+pp，且收敛速度最快。

**⚠️ 局限性**

局限性包括：依赖高质量递进式重写生成器；对大型基础模型的可扩展性未测试；在某些多跳任务（如OpenBookQA、MuSiQue）表现不佳；仅在英文文本推理上验证，未覆盖多语言或多模态场景。

---

## 42. An Accurate and Interpretable Hyper Graph Neural Network for GBM Survival Prediction

**arXiv ID:** 2609.25088 | [PDF](https://arxiv.org/pdf/2609.25088v1)

**作者:** Mushahid Intesum `[一作]` `[通讯]` (Independent Researcher), Mushahid Intesum (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个基于sheaf超图神经网络的GBM存活预测模型，并通过概念瓶颈和训练时的扩展充分性测试保证可解释性。

**💡 创新点**

创新点在于将sheaf超图、概念瓶颈、层级概念细化和扩展充分性正则化三者统一到一体化框架，并首次实现多模态MRI与临床基因信息的动态权重融合。

**🔧 技术方法**

采用的技术包括：双空间sheaf超图卷积、概念瓶颈层+多头自注意力细化、动态加权临床融合、离散时间负对数似然+对偶排名损失、以及扩展充分性测试（EST）正则化。

**📊 数据集**

使用了UPenn-GBM数据集，共593例患者，包含六种多模态MRI、临床病理指标和基因突变信息。

**📈 对比分析**

在5折交叉验证下，与DeepSurv、HyperCBM和MRePath等基线对比，E6配置得到C-Index 0.6431±0.015，表现与MRePath相当但方差更小，且在6/12/18个月时的td-AUC略优于基线；风险分层也达显著分离。

**⚠️ 局限性**

局限包括临床特征主导导致概念瓶颈仅提供可解释性而非显著提升准确率；样本量相对有限，缺乏外部验证；概念粒度有限，部分概念信息不足；单时间点预测，未考虑随访动态；以及对大规模图的可扩展性仍有待提升。

---

## 43. Online Automated Algorithm Design with Large Language Models

**arXiv ID:** 2609.25325 | [PDF](https://arxiv.org/pdf/2609.25325v1)

**作者:** Zhiyao Zhang `[一作]` (Hong Kong Polytechnic University), Kay Chen Tan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a4b10f5d-130b-4e77-9367-6469ec621899` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在线LLM驱动的自动算法设计框架 OnDesign，将算法设计与优化过程耦合，实现实时重设计。

**💡 创新点**

创新点在于把算法视为状态相关的决策变量，采用多代理协商与状态分析指导的SAG演化，消除离线预训练，动态生成符合当前搜索状态的算法。

**🔧 技术方法**

使用的大语言模型（DeepSeek‑V4）、多代理系统（Proposers + Arbiter）、状态分析准则（SAG）、执行反馈回传以及贝叶斯优化、演化连续/混合变量优化的实现。

**📊 数据集**

在 BBOB、CEC2020/2022/2026、MV‑BBOB、EOPCCV 等标准基准集，以及汽车能量吸收器工程案例上进行实验。

**📈 对比分析**

与固定逻辑优化器、适应性优化器以及离线LLM设计方法比较，OnDesign 在 17 个 suite‑dimension 配置中平均排名均位于 1–3 之间，显著优于所有基线，并在工程案例中实现最快收敛。

**⚠️ 局限性**

局限性在于仅验证单目标、维度不超过 30 的问题，对大规模、噪声、约束或多目标场景的适用性仍需进一步研究。

---

## 44. Tetris: Circuit Scheduling for Rearrangeably Non-Blocking Photonic Interconnects

**arXiv ID:** 2609.25434 | [PDF](https://arxiv.org/pdf/2609.25434v1)

**作者:** Eliezer Amponsah `[一作]` (Purdue University), Vamsi Addanki `[通讯]` (Purdue University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种针对重排可非阻塞（RNB）光学互连的电路调度算法（Tetris），实现了基于端点瓶颈驱动的匹配选择、互连感知的路径规划和无匹配全局阻塞的独立连接调度，从而显著降低All‑to‑All通信的完成时间。

**💡 创新点**

创新点在于：① 将剩余端点度覆盖与瓶颈端点工作量相结合，保证每次调度都推进最紧迫的端点；② 通过递归Beneš路由与路径可用性估计，考虑内部2×2开关的重配置依赖，减少因路径冲突导致的额外重配置延迟；③ 将匹配仅作为规划单位，允许各连接在独立时间点完成，消除传统匹配全局阻塞带来的等待。

**🔧 技术方法**

采用的技术包括：Birkhoff–von Neumann/最大权匹配的基本思想；最大剩余度覆盖的匹配选择；递归Beneš路由与路径可用性评估；独立连接起始时间计算与事件驱动执行；仿真框架与GPU硬件仿真相结合以评估性能。

**📊 数据集**

使用的主要数据集是Mixture‑of‑Experts（MoE）模型的All‑to‑All通信矩阵，包含Mixtral‑8×7B、Mixtral‑8×22B、DeepSeek‑MoE‑16B三种模型，在SPEED‑Bench、MMLU、GSM8K等任务上收集；同时生成合成的高度偏斜矩阵以探究流量不对称性对调度的影响。

**📈 对比分析**

在与BvN、MaxWeight、Best‑First‑Fit、Sunflow等基线在同一Beneš互连上进行仿真与硬件仿真对比。Tetris在大多数配置下相较BvN提升20‑30%，相较BFF/Sunflow提升8‑10%，并在重配置延迟增大时提升高达6.6×，整体接近理论下限，证明其在不同带宽与重配置延迟范围内均保持优势。

**⚠️ 局限性**

局限性：仅在Beneš拓扑下验证，未探讨其他互连结构的可迁移性；假设集中式控制器，未评估控制器延迟与多租户隔离等系统级挑战；路径冲突建模与调度算法的计算复杂度在大规模网络中仍待进一步优化。

---

## 45. Weakly Supervised Quantum Error Mitigation

**arXiv ID:** 2609.25555 | [PDF](https://arxiv.org/pdf/2609.25555v1)

**作者:** Seyed Mohamad Ali Tousi `[一作]` (University of Missouri), G. N. DeSouza `[通讯]` (University of Missouri)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用弱监督学习替代理想输出标签，构建16个标签函数对每个量子位的错误概率进行投票，使用标签模型生成软标签，再通过把预测的错误率当作单比特读出通道并求逆，得到纠正后的测量分布；

**💡 创新点**

首次在量子误差缓解中实现“无理想标签”训练，证明弱监督可提供足够信息，预测的错误率可直接用于逆读出校正，显著优于现有分析基线；

**🔧 技术方法**

程序化弱监督（labeling functions + Dawid‑Skene标签模型）、线性/树模型回归预测错误率、张量乘积读出通道逆推、KL、TV、Hellinger等指标评估；

**📊 数据集**

使用IBM Algiers与Hanoi两台量子处理器的5量子比特Pauli电路数据集（含20,000 shots、校准快照和理想分布）；

**📈 对比分析**

与无缓解、SPAM、Expander、SPAM+Expander等经典基线以及全监督模型对比；弱监督方法在两个设备上均能去除约24–29%的KL散度，优于所有分析基线；与全监督模型相比仍存在约0.11–0.15的相对差距；

**⚠️ 局限性**

需少量理想标签用于校准，方法仅适用于单比特独立读出模型，未能处理相关读出或更大规模/不同硬件的情况；对标签阈值和硬件变化敏感，尚未验证在更大规模或其他噪声模型下的稳健性。

---

## 46. LingLan: An Advancing Traditional Chinese Medicine Diagnosis LLM with Multimodal Data

**arXiv ID:** 2609.25715 | [PDF](https://arxiv.org/pdf/2609.25715v1)

**作者:** Zheng Chen `[一作]` (Tsinghua University), Peiwu Qin `[通讯]` (Guangdong Provincial Laboratory of Traditional Chinese Medicine Hengqin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究构建了多模态统一框架UFMD，将舌诊、脉诊图像等转化为结构化的临床记录，并以此数据对大型语言模型LingLan‑14B进行监督微调，实现传统中医 I‑AOI‑P 诊断流程的自动化。

**💡 创新点**

创新点包括：①提出UFMD实现多模态TCM诊断信息的自动结构化与人工校验；②将结构化诊断数据与LLM训练相结合，仿真 I‑AOI‑P 诊断逻辑；③采用低秩适配 LoRA 进行参数高效微调。

**🔧 技术方法**

技术手段包括：深度 OCR（DeepSeek‑OCR）与句子嵌入语义验证（Sentence‑BERT），多模态数据融合，MLLM（Qwen3‑14B）+ LoRA PEFT 的监督微调。

**📊 数据集**

使用约12万条TCM临床记录（舌诊、脉诊图像及问诊文本），按 GB/T 15657‑2021 进行分类；20% 数据留作测试。

**📈 对比分析**

通过与原始 Qwen3‑14B、Qwen3‑32B、DeepSeek‑R1‑Distill‑32B、DeepSeek‑R1(671B) 以及专门的 BianCang‑14B‑Instruct 等模型对比，采用 Accuracy、Recall、F1、UDI、PCDI 等指标，LingLan‑14B 在 Accuracy 0.627、F1 0.820 上相较基线提升超过 100%，性能显著优异。

**⚠️ 局限性**

局限性包括：由于患者隐私，数据集无法公开；实验仅在中等规模模型上验证，未评估更大规模或非开源模型；未来需扩展至更大模型并实现安全合规的数据公开与评估。

---

## 47. An Approximation Algorithm for Non-uniform Non-contiguous Translocation Distance

**arXiv ID:** 2609.25420 | [PDF](https://arxiv.org/pdf/2609.25420v1)

**作者:** Maria Constantin `[一作]` (University of Bucharest), Alexandru Popa `[通讯]` (University of Bucharest)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了首个多项式时间的非均匀非连续转位距离问题的近似算法，给出了单目标和多目标情况下的 O(log N) 近似比。

**💡 创新点**

创新点在于将非连续转位构造映射到可重复使用的压缩字符串表示，设计了贪心的 A‑相对非重叠分解算法，并通过内部化、SLP 转换与因子化桥接，实现在可接受的时间内得到接近最优的序列。

**🔧 技术方法**

核心技术包括：最长公共前缀/前缀匹配、后缀数组与 LCP 查询、LPnF (Longest Previous Non‑overlapping Factor) 计算、1‑转位模型、A‑相对组合系统与内部化、SLP（语法压缩）构造、持久化树操作、以及因子化与最优解之间的比较论证。

**📊 数据集**

该工作为理论研究，未使用任何真实或合成数据集，所有结果均基于算法与证明分析。

**📈 对比分析**

与最优解的比值被证明为 O(log N)；相对于已知的 2‑近似（仅适用于连续模型）以及 NP‑难度的证明，本文填补了非连续模型的近似可行性空缺，并未进行实验验证。

**⚠️ 局限性**

局限性：得到的近似比仍为对数级，尚未实现常数因子近似；理论证明依赖于多步变换，实际实现复杂度与常数项未知；未给出下界或更紧凑的构造；实验评估缺失。

---

## 48. What Does 99% Accuracy Measure? A Reproducible Audit of Shortcut Learning in a Widely Used Fake News Corpus

**arXiv ID:** 2609.25006 | [PDF](https://arxiv.org/pdf/2609.25006v1)

**作者:** Yuvraj Verma `[一作]` `[通讯]` (Independent Researcher), Yuvraj Verma (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对广泛使用的Fake and Real News数据集进行可复现的审计，揭示高准确率主要源于元数据和风格短路而非真伪判断。

**💡 创新点**

提出了系统的泄漏渠道分离、分布偏移评估与容量对照实验，首次量化元数据、来源标签、重复文档等泄漏对模型性能的贡献，并证明即使使用高容量模型也无法摆脱此短路。

**🔧 技术方法**

采用TF‑IDF+线性分类器（Logistic Regression、SVC、PA、Naïve Bayes）作为可解释测量基准，并与Fine‑tuned DistilBERT进行容量对照；同时使用Bootstrap、McNemar检验和Prior‑controlled分布偏移分析。

**📊 数据集**

使用Fake and Real News（约44.9k条新闻）作为主要数据集，并在外部LIAR数据集上做跨语料库泛化测试。

**📈 对比分析**

在随机拆分下线性模型达到≈99%准确率；去除元数据、来源标签和重复后仍保留≈98%；在主题不相交拆分中，平均精度从0.9995降至0.9475；DistilBERT在分布偏移时平均精度损失更大，且在跨语料库上降至≈0.56（接近随机）。

**⚠️ 局限性**

局限性包括仅评估单一数据集、仅采用TF‑IDF/线性和单一Transformer模型、对近似重复未做检测、缺少对同一出版者内真伪变化的对照，且跨语料库评估混合了来源与粒度变更。

---

## 49. Adaptive and Cost-Efficient Joint Scheduling of UAV Routes and Analytics with Transit-Borne Fog

**arXiv ID:** 2609.25479 | [PDF](https://arxiv.org/pdf/2609.25479v1)

**作者:** Suman Raj `[一作]` (University of Chicago), Sajal K. Das `[通讯]` (Missouri University of Science & Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种利用公交车作为移动雾计算节点，联合规划无人机飞行路径与任务分配，以在农村地区实现成本低、能源高效的时间约束任务调度。

**💡 创新点**

创新点在于把公共交通时刻表视为不可调度的计算资源，构造了任务分配时考虑停靠点的到达与蜂窝覆盖时间的“Drop-off”模型，并提出了可扩展的Divide and Assign（DA）启发式。

**🔧 技术方法**

使用了多层任务模型（采集、分析、交付）和成本/能耗/时延公式，结合ST-DBSCAN聚类、K-D树查询以及对公交时刻表的预处理来实现调度。

**📊 数据集**

实验基于印度农村地区的实际OpenCelliD基站、GTFS公交时刻表以及Rice Paddy Dataset的多种DNN模型（MobileNet-v3、YOLOv8、ResNet-34等）进行。

**📈 对比分析**

与EDF、NWF、PBTO、EOFO、TGTD等基线相比，DA在多种工作负载下实现了最高的任务完成率（高达100%）和最低成本，整体收益提高约20%–41%，且运行时间仅为0.04s。

**⚠️ 局限性**

局限在于对公交时刻表的准确性假设、无人机续航受限导致任务覆盖范围受限，以及未考虑实时任务生成与多目标协同。

---

## 50. LLM-Driven Training-free Location-Attribute Synergic Fusion: A Closed-Loop Paradigm for Dual-source Encrypted POIs and LULC Mapping

**arXiv ID:** 2609.25051 | [PDF](https://arxiv.org/pdf/2609.25051v1)

**作者:** Chang Li `[一作]` (Central China Normal University), Cairun Huang `[通讯]` (Central China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种闭环联合优化范式，利用大语言模型（LLM）驱动的训练-free位置-属性协同融合，解决双源加密POI（BD-09/GCJ-02）对齐与属性匹配问题，并将融合结果直接应用于训练-free大尺度土地利用/覆盖（LULC）映射。

**💡 创新点**

创新点：① 训练-free闭环联合优化，位置与属性互补、迭代共优化；② 通过LLM进行属性匹配，将 O(N²) 的暴力匹配降至 O(N)；③ 在局部子域使用粒子群优化（PSO）细化三角函数变换系数；④ 提出无参考的POI融合评估方法；⑤ 将精细化后的矢量转换为WGS-84后直接进行矢量‑栅格集成的 LULC 映射。

**🔧 技术方法**

技术方法：大语言模型（ChatGPT/DeepSeek）+ 预先规则的多轮对话进行属性匹配；ISODATA 自适应分区 + 本地搜索圆；LLM-fuzzy 置信度评估；粒子群优化（PSO）对三角函数变换参数进行局部优化；薄板样条（TPS）与传统 OLS 作为对比；无参考评估指标（残差、属性准确率）；矢量‑栅格集成 LULC 推断。

**📊 数据集**

数据集：31个中国省会/直辖市的百度（BD-09）与高德（GCJ-02）POI 共约 666k 条；Foursquare 开源 POI 用于合成加密数据验证；OSM‑LULC 与 EULUC-China 2.0 作为土地利用参考；实验覆盖 31 个城市，使用公开 API 收集同一时效的 POI。

**📈 对比分析**

与公开基线（ETT、TPS、GOLS、全局三角变换）对比，位置残差平均 4.58 m（比 ETT 提升 1.77 m，降低 13 m；TPS 17.65 m）；属性匹配准确率 95.12%（ChatGPT）/ 95.87%（DeepSeek），优于 BERT；LULC mIoU 75.14% / Macro‑F1 83.11%，显著高于 EULUC-China 2.0（58.58% / 72.01%）。闭环迭代仅需 2–3 次即可收敛。

**⚠️ 局限性**

局限性：仅处理两源加密系统；对绝对定位精度受加密噪声底层限制；评估以残差为主，缺乏真实绝对真值验证；对多源（>2）加密融合的可扩展性未验证；LLM 仍可能出现幻觉，需后续置信度过滤；在 POI 稀疏或极端偏差地区的效果尚待研究。

---

## 51. History-Conditioned Flow Matching for Probabilistic Dynamics of Tendon-Driven Continuum Robots

**arXiv ID:** 2609.25658 | [PDF](https://arxiv.org/pdf/2609.25658v1)

**作者:** Hang Yang `[一作]`, Ke Wu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对腱驱动连续体机器人在内部摩擦变化和执行器干扰下的动力学预测，提出了一种历史条件、物理信息化的流匹配（Flow Matching）框架，用来生成下一步完整关节配置的概率分布，并递归预测多步运动。

**💡 创新点**

创新点在于：①将运动历史与结构动力学参考结合，弥补瞬时观测不足导致的非马尔可夫依赖；②利用流匹配的条件回归形式，在保持采样效率的同时实现概率预测；③在多步递归预测中引入几何监督（Energy Score）保证分布的形状和覆盖率。

**🔧 技术方法**

使用的技术包括：条件流匹配（Conditional Flow Matching）、物理信息化的结构动力学参考（使用平面连杆模型与腱张力方程）、几何监督损失（Energy Score）、递归采样与Euler积分、以及与对比基线的比较（CVAE、Diffusion）。

**📊 数据集**

训练与测试数据来自 SpiRob 这款固定底盘二维连续体机器人，使用 MuJoCo 仿真生成的内部摩擦变化与执行器干扰两种场景，以及在真实机器人上收集的 5 条不同腱命令序列（其中 3 条用于训练/验证，2 条保留为测试）。

**📈 对比分析**

与 CVAE、Diffusion 等基线在一阶预测和五秒递归预测中比较，Flow 在 Energy Score、均方根误差、以及覆盖率等指标上均优于基线，且在仿真中比 Diffusion 的递归预测耗时约 40% 更快、在真实机器人中预测误差显著下降，且预测分布的扩散比例更接近真实实验的 1。

**⚠️ 局限性**

主要限制包括：①仍依赖于完整的结构动力学参考，若模型误差较大可能影响预测；②在高频控制循环中，虽然 Flow 具有较低延迟，但仍不及 CVAE 的极低延迟；③对非平面或三维连续体机器人的推广尚未验证；④在极端干扰或极大摩擦变化时，预测分布的覆盖率可能下降。

---

## 52. Rollout Efficiency in Reinforcement Learning for Reasoning Large Language Models: A Taxonomy and Future Directions

**arXiv ID:** 2609.25463 | [PDF](https://arxiv.org/pdf/2609.25463v1)

**作者:** Niloofar Gholipour `[一作]` (École de technologie supérieure, University of Québec), Xiaolong Bai `[通讯]` (Huawei Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统研究了大型语言模型（LLM）在推理强化学习（RL）训练中产生轨迹（rollout）的计算成本，并提出将提升方法划分为系统层面（如流水线解耦、资源感知调度、负载均衡、部分/早停生成、推测式解码）和算法层面（如rollout选择、prompt筛选等）的双轴分类法；进一步对各方法的瓶颈、互补性、冲突及评估标准进行了整理。

**💡 创新点**

创新点包括：①将rollout效率视为“成本‑目标质量”问题，构建系统与算法双轴分类与瓶颈映射；②对方法进行组合性分析，揭示不同技术之间的重叠与互补关系；③提出统一的效率度量框架和最小报告集合，便于跨研究比较；④识别了当前文献的评估缺口与未来研究方向。

**🔧 技术方法**

使用的技术涵盖：并行流水线解耦、资源感知与异构调度、负载均衡与长度预测、部分/早停生成、推测式解码、rollout选择、prompt筛选与预算分配、以及对应的熵与重要性采样校正。

**📊 数据集**

主要使用的评估数据集为大规模推理任务（如数学推理、代码生成等），采用 1.5B–405B 参数的 LLM，在 OpenAI o1、Kimi‑k1.5、DeepSeek‑R1 等模型上进行实验；同时在公开的数学与编码基准（AIME、LeetCode、ICLR/NeurIPS 任务）上验证效果。

**📈 对比分析**

与传统同步 RL 对比，系统层面方法可在 1.5–3× 的步骤时间或 20–40% 的训练步数下降（取决于模型规模和任务），而算法层面方法在保持或提升最终准确率的同时减少 30–80% 的轨迹生成量；在多种硬件配置（GPU/TPU、异构集群）与规模上均验证了性能提升的一致性。

**⚠️ 局限性**

局限性包括：①不同研究使用的效率度量不统一，导致跨方法比较困难；②缺少对多方法组合的系统实验，组合性结论主要基于机制推理；③对非稳定性（如prompt难度漂移、policy 漂移导致的推测式解码失效）缺乏理论保证；④在极长推理链或多轮交互任务中的迁移性尚未充分验证。

---

## 53. Evaluating Coding Agents on Kernel Exploit Generation

**arXiv ID:** 2609.25591 | [PDF](https://arxiv.org/pdf/2609.25591v1)

**作者:** Junyoung Jang `[一作]` (Independent Researcher), Lingming Zhang `[通讯]` (University Of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个名为KEX-Bench的基准，用以评估AI编码代理在真实Linux和Windows内核中生成可验证的攻击原语的能力。

**💡 创新点**

创新点在于首次将攻击原语（地址泄露、指令指针控制、堆读写、任意地址写）作为衡量指标，并通过确定性验证器与隔离虚拟机相结合，客观地捕捉代理从漏洞触发到原语实现的全过程。

**🔧 技术方法**

采用Model Context Protocol（MCP）实现工具调用封装，配合前沿/开源语言模型（如GPT-4、Llama），并使用自定义核模块、KASAN、WinDbg等工具进行内核状态检测。

**📊 数据集**

数据集包含45个任务实例，覆盖40个公开CVE，涵盖5种攻击原语，分别在Linux（20个CVE）和Windows（20个CVE）上进行评测；每个任务设有无PoC和有PoC两种情景。

**📈 对比分析**

在不同代理-模型组合下进行对比，最强配置在无PoC时Linux成功率为56%（14/25），Windows仅5%（1/20）；在有PoC情境下整体成功率提升至约69%（31/45），显示代理在触发漏洞后仍需显著改进以完成原语构造。

**⚠️ 局限性**

局限性包括：任务并非完全独立、平台与原语混合导致无法单独归因、数据集规模有限、未覆盖完整的特权升级链、评估受限于固定工具调用上限、可能存在训练数据泄露风险，且缺乏跨平台与新兴内核的通用性验证。

---

## 54. Recording Hand-Held Laparoscopic Instrument Motion in the Operating Room: Magnetometer-Free Fusion of Inertial, Range and Visual Sensing

**arXiv ID:** 2609.25577 | [PDF](https://arxiv.org/pdf/2609.25577v1)

**作者:** Jiyul Lee `[一作]` (Seoul National University Hospital), Hyoun-Joong Kong `[通讯]` (Seoul National University Hospital)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出了一种可夹在常规腹腔镜手术器械上的记录装置，利用六轴惯性测量单元、飞行时间测距仪以及无标记摄像头，在切口器（trocar）远心点约束下实现手持器械姿态的实时追踪与记录。

**💡 创新点**

创新点在于彻底摆脱了磁力计对姿态估计的依赖，改用摄像机获取头向信息，从而避免了手术环境中磁场干扰导致的误差，并在不修改器械进入患者体内部件的前提下实现毫厘级位姿精度。

**🔧 技术方法**

技术手段包括：六轴IMU（BNO085）惯性积分、VL53L0X ToF测距的非线性建模、ResNet-18编码解码器训练的无标记管道轮廓分割、基于错误状态Kalman滤波器的多传感器融合，以及对trocar远心点约束的软测量实现。

**📊 数据集**

实验数据集来自在Franka Research 3机器人控制下的手持器械在人工仿真模型（phantom）中的离散位移与旋转试验，共计300次平移试验和180次旋转试验，并在机器人引导下进行的程序化与远程操作连续轨迹记录。

**📈 对比分析**

与机器人基准的对比显示，全融合方案在平移上平均误差1.21 mm、旋转误差0.200°，相较于仅使用IMU或ToF的单一传感器组合误差高达数十倍；连续轨迹跟踪误差在程序化运动中为1.22 mm，手动远程操作为3.04 mm，显示出相对机器人原始轨迹的显著提升。

**⚠️ 局限性**

研究的局限性包括：仅在固定天花板摄像头的实验室环境验证，未覆盖手术中端镜头移动导致的相机姿态变化；使用的phantom模型与真实组织差异，且验证数据主要来自机器人辅助下的远程操作，而非真实手持操作；另外，标记器对特定管道外观的依赖可能限制了在不同器械或光照条件下的泛化能力。

---

## 55. GroundedGEO: Auditing the Evidence Gap in Generative Search Rankings

**arXiv ID:** 2609.25189 | [PDF](https://arxiv.org/pdf/2609.25189v1)

**作者:** Yihan Xia `[一作]` (Shenzhen University), Taotao Wang `[通讯]` (Shenzhen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了面向产品搜索的证据对齐基准、基于声明的后置重新排序层 GroundedGEO，并通过固定文本的 packet twin 与覆盖率实验来审计生成式检索中未证实细节的排名提升与基于证据的惩罚机制。

**💡 创新点**

（a）首次提出“identifiability gap”理论；（b）通过 packet twin 实验验证仅凭文本无法区分真实性，证据通道可实现区分；（c）构建可重复的证据对齐基准与标签可靠性门控，提供系统化评估框架。

**🔧 技术方法**

使用 LLM 列表/点位检索、声明提取、基于证据匹配的五分类标签、后置线性惩罚 (s_G = s_0 – λU) 进行重新排序。

**📊 数据集**

使用 ESCI 购物查询语料库 50 组查询，配套商品列表、买家评论、注册/认证记录共 1,950 个案例；人类标注 370 条声明用于门控验证。

**📈 对比分析**

在 Qwen2.5‑7B、MiMo‑v2.5、GLM‑5.3‑Flash 三个列表式 ranker 上对比未证实丰富、支持丰富、neutral 等变体；未证实丰富在 Qwen 上显著提升排名（NRG +0.065~+0.092）；GroundedGEO 在 oracle 标签下 λ=40 能将未证实丰富 top‑3 率从 0.65 降到 0.43，惩罚无误抑制；自动标签未通过可靠性门，性能取决于标签精度。

**⚠️ 局限性**

（a）自动标签准确率低，未通过可靠性门；（b）实验仅覆盖单一目标无多候选竞争；（c）证据包覆盖率有限，薄弱证据导致 coverage gap；（d）仅在 50 个商品查询上测试，推广性未知；（e）依赖点位基准评分，无法直接评估列表式表现；（f）对不同 ranker 的可推广性不明；（g）需要更大的人类金标和多模型检验。

---

## 56. Understanding Maintenance and Support in a Community-Driven Scientific Workflow Ecosystem: A Cross-Space Study of Galaxy

**arXiv ID:** 2609.25587 | [PDF](https://arxiv.org/pdf/2609.25587v1)

**作者:** Khairul Alam `[一作]` (University of Saskatchewan), Banani Roy `[通讯]` (University of Saskatchewan)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对Galaxy生态系统的维护与支持进行了跨空间的实证分析，系统识别并描述了GitHub问题、拉取请求以及社区论坛中的维护主题，并探讨了它们的解决情况与跨空间关联性。

**💡 创新点**

创新点在于①将开发空间（issue/PR）与支持空间（社区论坛）整合分析，②使用BERTopic实现主题发现，③通过多维度统计和生存分析揭示影响解决率和时效的特征，④在缺乏显式链接的情境下提出语义匹配与技术信号匹配来弥补跨空间可追溯性。

**🔧 技术方法**

技术手段包括Transformer‑based 文本嵌入 + UMAP + HDBSCAN（BERTopic）进行主题建模；Fisher精确检验、Mann‑Whitney U、Logistic回归与Cox比例风险模型分析解决率与时效；正则表达式技术信号提取；语义匹配（TF‑IDF + 余弦相似度）与时间窗口过滤。

**📊 数据集**

使用Galaxy GitHub 340个活跃仓库收集的 11,762 条 issue、52,203 条 PR（含 2,419 关联 PR）和 6,235 条社区论坛讨论（截至 2026‑03‑31）共 67,767 条语料。

**📈 对比分析**

结果显示：issue 9 个主题、PR 14 个主题、论坛 14 个主题，覆盖工作流执行、工具/依赖、基础设施、数据管理、文档与培训等；解决率与时间与协作、诊断、贡献、自动化、参与度等特征相关；显式追溯率仅 97.77% 为 issue/PR 内部，跨空间显式追溯率不足 0.5%，但语义匹配与技术信号揭示了更广泛的隐式关联。

**⚠️ 局限性**

主要局限包括：只分析了活跃仓库，可能遗漏已归档或私有项目；语义匹配阈值与时间窗口设置可能导致漏检或误检；未对用户满意度或实际修复质量进行评估；跨空间追溯主要基于公开链接，未覆盖内部通讯或私有讨论。

---

## 57. EMGBlend: Heterogeneity-Aware Self-Supervised Pretraining for Gesture and Force Decoding

**arXiv ID:** 2609.25582 | [PDF](https://arxiv.org/pdf/2609.25582v1)

**作者:** Yuwei Jia `[一作]`, Zhe Cui `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为EMGBlend的自监督预训练框架，专门用于融合不同采集设备产生的表面肌电（EMG）数据；

**💡 创新点**

创新点在于同时解决通道布置差异、频带支持不匹配和数据来源曝光不均衡三大问题，采用共享patch编码、几何感知注意力、频带限定的谱码预测和源平衡采样；

**🔧 技术方法**

主要技术包括Conv1D共享patch编码、时空交替注意力（配合坐标傅里叶特征）、向量量化的谱码词典、掩码语义学习和对频带的有效重建约束；

**📊 数据集**

使用了11个公开EMG数据集（如ActionSense、emg2pose、emg2qwerty、EMG‑FMFP、ForceBand、GNI、HD‑FW‑Kin、Hyser、KIMHu、multimodal glove、UCI EMG）共计约46万窗口；

**📈 对比分析**

在手势识别、连续力回归和接触分类等下游任务中与随机初始化、波形重建、PhysioWave等基线对比，EMGBlend在seen/unseen手势准确率、within‑person力预测R²、PiMForce MAE等指标均取得明显提升；

**⚠️ 局限性**

局限性包括跨人力预测仍表现不佳，仍需更多多样化且标注丰富的数据集，以及对缺失几何信息的处理仍有改进空间。

---

## 58. From Offline Proxies to Online Decisions: A Layered Engagement Evaluation Framework for Conversational AI

**arXiv ID:** 2609.25408 | [PDF](https://arxiv.org/pdf/2609.25408v1)

**作者:** Xuanyi Li `[一作]` (Meta Platforms, Inc.), Alex Deng `[通讯]` (Meta Platforms, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证一个可冻结的离线代理，用于在多轮对话AI实验前预测实验结果，从而节省实验流量。

**💡 创新点**

将离线代理拆解为标签–结果、评分器–行为、套件–实验三层对齐，并引入区间感知决策一致性评估与校准映射，形成可审计的决策规则。

**🔧 技术方法**

使用二分类返回标签的参与度分类器、固定评估套件、基于 bootstrap 的区间稳定性、加权线性校准映射以及置信区间驱动的决策判定。

**📊 数据集**

基于 Meta 平台生产会话日志构建的 4,832 条固定评估套件以及 27 次 A/B 实验（共 489 对照）和后续冻结后续实验的数据。

**📈 对比分析**

与原始分类器、随机/多数/单信号规则以及 5 个部署的离线评估器对比，冻结复合在独立复制集上实现约 81.1% 对照微 F1、0 错误方向调用，明显优于原始 34.3% 及其他评估器。

**⚠️ 局限性**

仅验证已完成实验的对照匹配，未评估未来覆盖、全漏斗筛选、跨产品推广或不同用户子组；层 1–2 的有效性未单独验证，且依赖固定评估套件。

---

## 59. Impact Is Not Invalidation: Ask About the Claim, Not the Diff

**arXiv ID:** 2609.25130 | [PDF](https://arxiv.org/pdf/2609.25130v1)

**作者:** Atul Anand `[一作]` `[通讯]` (Thomson Reuters), Atul Anand (Thomson Reuters)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于真实 Python 库测试执行的主张失效基准，评估在仓库变更时哪些存储的声明变为错误，并比较不同失效判定方法；

**💡 创新点**

证明“声明级别”询问（是否声明仍然成立）远优于“差分级别”询问（行为是否保持不变），并显示即使完整的影响分析工具也无法弥补这一差距；

**🔧 技术方法**

使用大型语言模型（Gemini、Claude 等）对 diff 或声明+diff 进行推理，结合基于覆盖率的回归测试选择工具（testmon），以及静态/动态影响分析；

**📊 数据集**

基准数据集包含 23 个 Python 库、10,369 条测试声明（主张），其中 184 条在后续提交中被证实失效，约 1.8% 的 flip 率；

**📈 对比分析**

在对照实验中，claim-relative LLM（A6）在 17 个保留仓库的 held‑out 集上达 0.79 精度、0.72 召回；semantic‑equivalence LLM（A5）仅 0.29 精度；coverage‑based test selector（testmon）达到 0.41 精度、0.87 召回；显示 A6 在同等成本下显著提升了失效判断的准确性；

**⚠️ 局限性**

限制包括：仅在小型 Python 库上验证，flip 事件稀疏，依赖 CI 过滤导致样本偏差，未涉及大型 monorepo 或非 Python 语言，模型性能受预训练数据及 prompt 设计影响；

---

## 60. TelecomGPT-R1: Unified Post-Training for Reasoning Across Heterogeneous Telecom Tasks

**arXiv ID:** 2609.25356 | [PDF](https://arxiv.org/pdf/2609.25356v1)

**作者:** Bohao Wang `[一作]` (Zhejiang University), Merouane Debbah `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并训练了统一的 TelecomGPT‑R1 推理模型，覆盖协议、知识、建模与故障四大推理轴。

**💡 创新点**

轴感知的数据构建、基于验证器的细粒度奖励、SFT+DAPO 两阶段后训练以及跨源多任务统一策略。

**🔧 技术方法**

LoRA 微调、动态采样策略优化 DAPO、基于规则的可验证奖励、教师多样化 CoT 生成、分布式 vLLM 训练等技术。

**📊 数据集**

104,880 条从 3GPP 规范、O‑RAN 文档、srsRAN 源码、运维日志、公式与表格等公开资料衍生的 QA 与 CoT 训练集。

**📈 对比分析**

在 GSMA Open Telco Leaderboard 七项基准上与 12+ 开源/专有模型对比，27B 版 TelecomGPT‑R1 平均得分 89.6%，超过 GPT‑5、Gemini、Claude 及开源同类模型。

**⚠️ 局限性**

仍依赖文本输入，未覆盖多模态证据；需要大量训练资源；对极端罕见情景的鲁棒性尚待验证；可能在某些细粒度协议细节上存在推理错误。

---

## 61. TAILOR: Template-Preserving Augmentation for Long-Tailed Log Parsing

**arXiv ID:** 2609.25261 | [PDF](https://arxiv.org/pdf/2609.25261v1)

**作者:** Sepideh Hodaeian `[一作]` (University of Alberta), An Ran Chen `[通讯]` (University of Alberta)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对稀有日志组的模板推断不足，提出了一种模板保持增强框架，在模板推断前为稀有日志组生成结构一致的合成日志，以提升模板提取准确率。

**💡 创新点**

创新点在于：①引入语义模板生成与日志增强，利用正则识别常见结构化实体并让LLM生成不同值的变体；②通过多样性采样与验证机制提升模板鲁棒性；③实现与现有LLM解析器无缝集成，不需改造核心算法。

**🔧 技术方法**

技术包括：轻量化正则识别、LLM（如Llama-3-8B、GPT）提示式生成、Jaccard相似度多样性采样、模板验证与缓存、以及对日志进行归一化。

**📊 数据集**

使用 Loghub‑2.0 基准共14个数据集（Thunderbird、Mac、BGL、Hadoop、HealthApp、Linux 等），共超过5000万条日志。

**📈 对比分析**

与传统 Drain、Spell、AEL 以及 LLM 基础解析器（LibreLog、EFParser、LILAC‑8shots）对比，稀有日志组解析准确率提升至少 19%，在完整数据集上保持与最强基线相当；在不同 LLM 后端上亦可提升多达 41%。

**⚠️ 局限性**

局限性：①阈值 <5 的定义可能不适用于所有场景；②依赖正则识别的语义类别，若日志包含未识别实体可能影响增强效果；③使用 LLM 产生的推断仍可能产生误差，需要验证步骤；④存在潜在的数据泄露风险及较高的计算成本。

---

## 62. From functioning to evolving: A complex systems perspective on future self-organised federated energy communities

**arXiv ID:** 2609.25425 | [PDF](https://arxiv.org/pdf/2609.25425v1)

**作者:** Abdorasoul Ghasemi `[一作]` `[通讯]` (Coventry University), Abdorasoul Ghasemi (Coventry University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出以复杂系统视角设计未来自组织联邦能源社区（FEC）的框架，并从互联网与敏捷软件开发的演进经验中提炼可迁移的设计原则，阐述如何通过分布式共识与可扩展协议实现能源社区在不破坏电网约束、兼顾市场与自给自足的自适应运行。

**💡 创新点**

创新点在于将“演进设计”理念（功能 → 进化）引入能源系统，将互联网的“松散协议”与敏捷的“自组织”特性迁移到能源社区治理，提出基于可行操作区（FOR）共识的分布式协调机制，实现多社区协同约束下的自适应供需匹配。

**🔧 技术方法**

使用技术包括：复杂系统分析框架、分布式共识算法（如基于投票/共识的迭代求解）、可行操作区（FOR）约束模型、边缘计算与分布式决策、能源社区内部微调的能量管理策略（储能、负荷调节）以及 P2P 能源交易与市场感知的结合。

**📊 数据集**

未使用具体数据集；研究主要为理论架构与概念设计，所引用的案例与经验来自文献综述与现有研究（如互联网协议、敏捷方法、能源社区运营模型）。

**📈 对比分析**

方法比较以理论与概念对比方式进行：将传统集中式、功能导向的能源网络与提出的自组织、进化导向的 FEC 网络进行对比，重点比较在于适应性、可扩展性与系统韧性；文章未给出实验性性能指标，而是通过分析和仿真预期来论证潜在优势。

**⚠️ 局限性**

局限性包括：缺乏大规模实证验证与性能评估、分布式共识在极端负荷或网络失效情形下的鲁棒性未知、监管与法律框架尚未完善、以及多社区协同所需的通信与计算资源需求待进一步评估。

---

## 63. Stable Unsupervised Continual Chunking with Sheaf SyncMap

**arXiv ID:** 2609.25143 | [PDF](https://arxiv.org/pdf/2609.25143v1)

**作者:** Xueyuan Li `[一作]`, Danilo Vasconcellos Vargas `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于图同调理论的同步映射方法，评估其在不同概率数据集上的性能

**💡 创新点**

创新点在于提出Graph SheafSyncMap和DecentralizedSyncMap两种新的同步映射框架，显著提升了同步精度

**🔧 技术方法**

采用图神经网络、Sheaf理论以及分布式同步算法来构建与评估同步映射

**📊 数据集**

使用多组参数不同的probabilistic数据集（如probabilistic120_5、probabilistic10_60等）进行实验

**📈 对比分析**

与SymmetricalSyncMap和StandardSyncMap三种传统方法对比，Graph SheafSyncMap平均得分0.9663，DecentralizedSyncMap0.9645，显著优于其它方法；两者在12个数据集上获得最佳同步结果

**⚠️ 局限性**

局限性包括仅在人工生成的概率数据集上测试，缺乏真实世界数据验证，且在极低概率或极高维度情况下表现相对不佳

---

## 64. Embedded Assessments for Frontier AI

**arXiv ID:** 2609.25413 | [PDF](https://arxiv.org/pdf/2609.25413v1)

**作者:** Jacob Charnock `[一作]` (GovAI), Jonas Freund `[通讯]` (GovAI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并论证嵌入式评估（embedded assessments）的概念与实施框架，讨论其设计要素、益处与挑战，并给出建议

**💡 创新点**

首次系统性阐述嵌入式评估的设计原则与实施细节，强调内部访问可深入评估内部AI使用风险，提出连续评估与季度报告等实践方案

**🔧 技术方法**

无具体技术实现，主要利用访谈、文档审查和现场系统访问等方法

**📊 数据集**

无公开数据集，仅基于前期 METR、Redwood 的试点案例和行业经验

**📈 对比分析**

未进行实验比较，主要通过对现有远程评估方法的不足与嵌入式评估的优势进行理论对比

**⚠️ 局限性**

实践可行性仍需验证，资源与IP风险、评估成本、评估者独立性、范围与时间选择的挑战等

---

## 65. Efficient Iterative Retrieval with Heterogeneous Batching

**arXiv ID:** 2609.25405 | [PDF](https://arxiv.org/pdf/2609.25405v1)

**作者:** Dohyun Park `[一作]` (University of Illinois Urbana Champaign), Yongjoo Park `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的服务系统，能够在单个推理循环中同时处理嵌入与生成请求，实现异构批量。

**💡 创新点**

核心创新在于将嵌入请求切分为块并使用增量池化与工作负载感知调度（IBS），实现嵌入和生成的资源平衡，消除“瓶颈”与头阻塞。

**🔧 技术方法**

技术包括chunked embedding、incremental pooling、统一Runner、动态比例调度IBS，以及与vLLM集成的迭代级调度。

**📊 数据集**

使用Mistral‑7B、e5‑mistral‑7b‑instruct、Qwen2‑7B、LLaMA3.1‑8B等LoRA模型，以及Iter‑RetGen与2WikiMultihopQA的RAG工作负载。

**📈 对比分析**

对比基线GPU划分、同类批量和异构批量，实验显示在四张A100 GPU上吞吐率提升1.28–4.52倍，p99延迟降低55.8%，GPU利用率提升约41个百分点。

**⚠️ 局限性**

局限包括：仅在单模型/单GPU实验，未测试多GPU分片模型；只考虑≤10B参数的开源模型；未评估更大规模模型或跨GPU KV缓存交互的影响。

---

## 66. A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators

**arXiv ID:** 2609.25572 | [PDF](https://arxiv.org/pdf/2609.25572v1)

**作者:** Chaehyun Kim `[一作]` (KAIST), Chanyoung Park `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 PQA 方法，利用个性化页级质量锚点修正 LLM 用户模拟器中活动特征对偏好判断的干扰。

**💡 创新点**

创新点在于将历史类别重叠基线转化为量化的质量锚点，并在提示中加入基于该锚点的页级标签（ABOVE/NORMAL/BELOW），实现活动特征与偏好特征的独立调控。

**🔧 技术方法**

使用 GPT‑4o‑mini 作为 LLM 引擎；通过嵌入式相似度和类别重叠计算；对页级标签进行规则化决策。

**📊 数据集**

实验数据集为 MovieLens（电影评分）和 Amazon CDs（音乐/CD 购买）。

**📈 对比分析**

与 Agent4Rec 与 SimUSER 两大基线对比，采用 NDCG@5、HR@5、满意度（S_sat）及 GPT‑4o‑o 评估的“人类相似度”指标；PQA 在两组数据集上均提升了 5–15% 的推荐质量、显著降低了高活动导致的满意度膨胀，并使模拟交互的人工评分提升至 4.0 以上。

**⚠️ 局限性**

局限性：当前仅使用类别重叠作为质量锚点，未考虑更细粒度的偏好信号；对不同领域的泛化性待进一步验证；同时 PQA 仍依赖于准确的历史基线估计，若历史数据稀疏或不完整可能影响效果。

---

## 67. Matryoshka attribution: Learning to attribute language model outputs to representations and weights

**arXiv ID:** 2609.25518 | [PDF](https://arxiv.org/pdf/2609.25518v1)

**作者:** Aryaman Arora `[一作]` (Stanford University), Christopher Potts `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文提出了一种新的可学习掩码方法Matryoshka Attribution，用于将语言模型输出归因于其内部计算，并在参数层面实现对行为的定位与编辑。

**💡 创新点**

创新点在于联合随机化稀疏预算与可微分sigmoid top‑k掩码的Matryoshka框架，使得模型能在一次训练中学习完整的归因排序并直接优化下游任务损失。

**🔧 技术方法**

该方法结合了软互换干预、可微sigmoid top‑k、梯度下降、强化学习（GRPO）以及对比学习的奖励函数，用以对内部激活和参数差异进行归因。

**📊 数据集**

评估使用了Mechanistic Interpretability Benchmark（MIB）及其扩展MIB+的数据集，包含多种模型（GPT‑2、Qwen2.5、Gemma、Llama‑3.1）和多任务（代词指向、算术、问答等），以及对Llama 1B/8B拒绝行为的强制训练数据。

**📈 对比分析**

与I×G、IG、DBM、Node Pruning、Interchange Interventions等基线比较，Matryoshka在CPR和compactness两项指标上均领先，MIB排行榜第一，且在参数归因实验中仅改动1–2%权重即可去除拒绝而保持其他能力。

**⚠️ 局限性**

主要限制包括对大量对照输入样本的依赖、训练成本相对较高、对不同架构或更大模型的泛化尚未充分验证，以及对稀疏预算采样分布的敏感性。

---

## 68. Conduct Under Pressure: What Sixty Language Models Do When a User Pushes

**arXiv ID:** 2609.25447 | [PDF](https://arxiv.org/pdf/2609.25447v1)

**作者:** Tapan Parikh `[一作]` `[通讯]` (Cornell Tech), Tapan Parikh (Cornell Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在用户施压情境下是否会屈服，构建代码本并对60个模型进行编码。

**💡 创新点**

创新点在于将行为测量拆分为轨迹与方式，发现方式是厂商属性且可通过机器编码。

**🔧 技术方法**

使用自制情境场景、冻结多轮对话、开源代码本、LLM和人工编码及判定规则。

**📊 数据集**

数据集为60个模型、13个厂商的120条固定对话场景。

**📈 对比分析**

方法是对模型输出进行轨迹和方式编码，对比机器编码与人工的可靠性，机器一致性高于人类，且能复现人工裁决。

**⚠️ 局限性**

局限包括情境由单一作者编写、仅覆盖三种需求、能力与发布日期高度相关、标签为presence/absence且人类样本有限。

---

## 69. Deflecting the Value Compass: Interacting with Large Language Models Temporarily Shifts Human Value Priorities Toward Personal Focus

**arXiv ID:** 2609.25586 | [PDF](https://arxiv.org/pdf/2609.25586v1)

**作者:** Hasibur Rahman `[一作]` (Northeastern University), Smit Desai `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过三阶段实验，检验使用ChatGPT、Claude、Gemini等大型语言模型（LLM）作为“思考伙伴”是否能在不提供价值取向或决策建议的情况下，短暂改变参与者在价值优先级（基于Schwartz价值圆）上的倾向，并观察这种变化是否在后续无LLM任务中持续。

**💡 创新点**

创新点在于：①首次从价值优先级层面评估LLM的“无指令”交互影响；②证明LLM在仅仅通过对话支持推理时，能够在不显式引导的前提下，将用户的价值偏向统一向个人化倾向；③同时考察该影响是否伴随价值方向的收敛或写作输出的同质化，发现两者无显著关联。

**🔧 技术方法**

使用了Schwartz修订版价值问卷（PVQ‑RR）对价值优先级进行量化，并通过对话平台收集LLM回复与用户输入；对话被设计为“思考伙伴”角色，限制LLM不推荐、不中立地提出价值。

**📊 数据集**

数据集：200名美国成年人（均为英语使用者），分为ChatGPT、Claude、Gemini和Baseline四组，每组50人；每人完成三道真实情境下的困境问答，三阶段均使用不同的PVQ‑RR块以避免重复；Baseline组仅阅读LLM生成的固定文本而无交互。

**📈 对比分析**

比较方法：在每阶段使用混合效应模型（包含块、情境、顺序等协变量）计算相对变化；主要结果为个人化轴（Openness + Self‑Enhancement − Conservation − Self‑Transcendence）的变化，LLM组在阶段2相对Baseline提升0.31–0.43点（Glass d ≈ 0.37–0.51），随后在阶段3回落至无显著差异。其它分析包括价值向量旋转角度、价值幅度与方向的离散度、写作文本的词汇重叠与语义集中度，均显示LLM对价值方向的统一移动，但未导致输出同质化。

**⚠️ 局限性**

限制：①样本仅限美国地区，无法推断跨文化差异；②仅使用三道困境且未测量个人利益程度，难以评估对更具个人意义情境的影响；③Baseline与LLM交互对时间和主动性存在混淆，难以分离对话本身与额外反思/时间的影响；④价值测量仅在单次实验内完成，缺乏长期追踪；⑤模型版本和提示设定固定，未来更新或不同角色设定可能产生不同效果。

---

## 70. Concept Drift from a Causal Perspective

**arXiv ID:** 2609.25340 | [PDF](https://arxiv.org/pdf/2609.25340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 71. Topological Signal Processing With Unoriented Operators

**arXiv ID:** 2609.25310 | [PDF](https://arxiv.org/pdf/2609.25310v1)

**作者:** Andrea Cavallo `[一作]`, Elvin Isufi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了无定向拓扑信号处理（UTSP）框架，用无定向关联矩阵代替传统有向边界算子，对无方向高阶信号进行处理。

**💡 创新点**

创新点包括：①证明无定向关联矩阵与Laplacian具有图谱性质；②引入“交互阶分解”，在无定向情形下实现信号的层级分解；③基于交互阶分解设计了针对不同阶的正则化器，并在去噪与插补任务中显著优于有向基线。

**🔧 技术方法**

技术方法主要是：构造无定向incidence矩阵与Laplacian；证明其谱性质与图相似；发展交互阶分解理论；设计基于正则化的Tikhonov逆问题；利用谱滤波实现去噪与插补。

**📊 数据集**

实验数据集包括四个实际高阶网络：1）荧光蛋白突变组合（全体三维单纯复形）；2）StackExchange数学问题标签超图；3）药物化学成分超图；4）科研作者共同作者网络（TF-IDF共现）。

**📈 对比分析**

比较方法包括：岭回归、邻域均值、各Hodge Laplacian的低/高通滤波、无定向“凝聚”正则化以及交互阶正则化。结果显示，交互阶正则化在所有实验设置下均取得最低RMSE，尤其在能量分布不均衡的高阶信号上优势最为显著。

**⚠️ 局限性**

局限性主要在于：①无定向框架仅适用于无方向信号；②对高阶级别的正则化参数调优仍需经验；③在极大规模复杂体时计算量较大；④未考虑动态或时变信号的扩展。

---

## 72. SurgGraph: Quantitative Laparoscopic Video Understanding via Geometry-Grounded Scene Graphs

**arXiv ID:** 2609.25651 | [PDF](https://arxiv.org/pdf/2609.25651v1)

**作者:** Jingying Wang `[一作]` (University of Michigan), Xu Wang `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一个无需训练的神经‑符号管线SurgGraph，利用分割与深度图生成量化的结构化场景图（⟨主体, 动词, 对象, 数值⟩），支持关系检索、解释与自动标注，并以此构建教育工具SurgGraphQA；

**💡 创新点**

创新点在于：①将临床定义直接转化为可执行的几何规则，实现连续数值关系；②在保持结构化语义的同时，保留可量化的关系强度；③通过量化场景图实现精确检索与可解释的决策反馈，显著提升VLM的定量表现；

**🔧 技术方法**

使用技术包括SAM‑3图像分割、Video Depth Anything深度估计、基于边界、深度差异、凸包与工具提示的几何算子，构成全局张量化关系表示；

**📊 数据集**

主要数据集包括公开的lap chole视频、CholecT50（工具-动作-目标标注）和EndoVis CVS（关键视图安全判定）等；

**📈 对比分析**

与SurgVLM对比，SurgGraph在工具-动作-目标识别上从31.3%提升至79.2%，在阶段识别上从91.0%提升至93.8%，在CVS评估上从79.4%提升至94.6%；在分割-深度结合的IoU评估中取得0.73；

**⚠️ 局限性**

局限性在于：①几何规则需人工定义，扩展至新手术或新关系需要额外工程；②仅依赖分割与深度，对遮挡/纹理变化敏感；③未利用时序信息，无法捕捉工具动力学；④对不同相机参数需手动校准，影响通用性。

---

## 73. ArticleMiner: Ontology-Guided Knowledge Graph Construction from Scientific Publications

**arXiv ID:** 2609.25607 | [PDF](https://arxiv.org/pdf/2609.25607v1)

**作者:** Md Abrar Jahin `[一作]` (University of Southern California), Jay Pujara `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ArticleMiner 框架，结合 LLM 与多种 PDF 解析器，对科学论文及其补充文件中的表格进行解析、归一化、验证和知识图谱化，支持四个专业领域的表格提取与知识图谱构建。

**💡 创新点**

创新点在于：① 将领域知识外部化为“任务模块”，只需提供词汇表、映射、规则、验证器和 RDF 绑定，减少对特定领域模型的依赖；② 采用多源证据融合与身份键对同一观测进行一致性校验，并保留来源证明；③ 引入自我纠错与视觉回退机制，提升对表格结构错误与 OCR 问题的鲁棒性。

**🔧 技术方法**

使用技术包括：大型语言模型（Sonnet 4.6、Haiku 4.5、GPT‑4o、Gemini 等）进行文本提取与推理；多路 PDF 解析器（Docling、Marker、MinerU、pdfplumber、Camelot）获取表格结构与单元格文本；Python 代码实现规则、校验与身份合并；JSON‑LD 输出用于知识图谱集成。

**📊 数据集**

实验数据集：共 163 篇论文，包含 28 篇 GeoChem（地质化学）、111 篇 DiSCoMaT（材料科学玻璃组分）、15 篇 MLTables（机器学习评测表）和 9 篇 ChemTables（药物发现表）。其中 GeoChem 还配备专家手工标注的基准数据。

**📈 对比分析**

与相同 LLM 的少量示例提示基线、公开的领域专用系统以及组件消融进行了比较。结果显示 ArticleMiner 在四个任务上均取得更高的 F1 或样本匹配率，GeoChem 的样本 F1 提升约 62 分；在 DiSCoMaT、MLTables、ChemTables 上提升 15–15 分。消融实验表明：去除领域模块、验证或自我纠错会显著下降性能；多路解析器的融合并不总是提升准确率。

**⚠️ 局限性**

局限性包括：① 需要领域专家手工编写词汇表、映射与验证规则，适配成本未知；② 依赖补充文件（尤其 GeoChem），若缺失则性能下降；③ 目前仅实现单文献级别的实体识别，缺乏跨文献实体对齐；④ 对解析器的选择与配置敏感，单一解析器往往效果不佳；⑤ 评估多为单元格级别，未充分验证对知识图谱查询性能的影响。

---

## 74. Beyond Provenance: The Economics and Governance of Personalized AI Memory

**arXiv ID:** 2609.25521 | [PDF](https://arxiv.org/pdf/2609.25521v1)

**作者:** Jianan Chen `[一作]` (Purdue University), Yaosen Lin `[通讯]` (University of California San Diego)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建并分析了个性化AI记忆的经济与治理框架，探讨了记忆的可复制性、刷新机制、所有权配置与政策对记忆供应与长期库存的影响。

**💡 创新点**

创新点在于提出记忆使用为非竞争、刷新为共同生产的统一模型，发现提取上限、权利分离定理、扩大覆盖率导致长期库存下降的“耗尽”结果，并给出基于收益、暴露与可持续主张的最优权利组合。

**🔧 技术方法**

主要技术包括基于博弈论的动态均衡分析、非竞争资源与合作刷新模型的解析推导、决策价值(q)、忠诚度与信息互锁度(k⋆)等指标的定义与测度、以及通过合成记忆经济模拟验证理论推导。

**📊 数据集**

使用合成的、基于高斯AR(1)状态的记忆经济模拟数据，生成交互历史、提取结果、买方需求与用户暴露参数，以检验模型在已知基准下的估计准确性。

**📈 对比分析**

对比方法是将理论导出的闭式解与合成模拟的真实值进行对照，评估决策价值、耗尽效应与权利分配的匹配程度；结果显示在多种参数配置下理论与模拟高度一致，验证了模型的稳健性，但未在真实数据上做性能基准测试。

**⚠️ 局限性**

局限性包括：假设完美来源追溯与完全信息；未考虑多主体记忆与非单一决策者情景；模型参数如曝光强度、刷新生产率需外部估计；以及仅通过仿真验证，缺乏在实际AI平台记忆市场中的经验验证。

---

## 75. PAKT: Physically-Aligned Kinesthetic Teaching for Reinforcement Learning

**arXiv ID:** 2609.25630 | [PDF](https://arxiv.org/pdf/2609.25630v1)

**作者:** Lars Johannsmeier `[一作]` (NVIDIA), Yashraj Narang `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个名为Pakt的框架，将 kinesthetic teaching 与高性能控制栈相结合，用于真实世界的强化学习，在工业装配任务中实现了更低周期时间和更少干预。

**💡 创新点**

通过在 admittance 控制、参考生成器和阻抗控制之间耦合，使人类指导和RL策略共享相同的6‑DoF坐标、运动学限制和追踪动力学，从而保证采样轨迹可被政策复制，并显著降低干预成本。

**🔧 技术方法**

admittance control（人类施加力转运动）、参考生成器（速度、加速度、jerk 限制）、Cartesian impedance controller（高频 torque 控制）以及 HIL‑SERL 强化学习框架。

**📊 数据集**

四个工业插入与装配基准任务，包括 RAM 插入、数据中心计算托盘装配等，实验采用 Franka Emika 机器人进行真实硬件测试；RAM 插入在五个随机种子上重复。

**📈 对比分析**

与基线 HIL‑SERL（使用 SpaceMouse 远程控制）对比，Pakt 在四个任务中将周期时间降低 23%–48%，累计干预次数降低 62%–86%，初始示范时间降低 15%–34%；混合 ablation 表明控制栈贡献最大，保持控制栈不变时 kinesthetic teaching 还能进一步减少干预。

**⚠️ 局限性**

仅在单台 Franka 机器人和单个训练好的操作员上验证，未测试多操作者、多机器人或非插入任务；参考生成器与 admittance 参数未做敏感性或稳定性研究；基于力估计的 admittance 受机器人摩擦影响，且未对外部摄像头遮挡问题做深入评估。

---

## 76. Compressing Long Context into Answer-Aligned Memory Embeddings for LLM Inference

**arXiv ID:** 2609.25537 | [PDF](https://arxiv.org/pdf/2609.25537v1)

**作者:** Md Mostafizer Rahman `[一作]` (University of Notre Dame), Fang Liu `[通讯]` (University of Notre Dame)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Context-to-Answer-Aligned Memory Compression（CMC）框架，利用 ContextEncoder、MemoryBridge 与双层 KV 缓存，将长文本压缩为 Compact Context Memory Embeddings（CMEs）供冻结的 LLM 解码。

**💡 创新点**

创新点包括：①图谱式上下文去噪与分块压缩；②跨架构投影（MemoryBridge）使任意编码器与任意解码器配对；③两层 KV 缓存策略，第一层通过问题引导的 top‑K CME 选择，第二层保留局部窗口；④双阶段训练，Phase‑1 对齐 CMEs，Phase‑2 通过答案蒸馏、KL 与对比损失实现答案导向的 CMEs。

**🔧 技术方法**

主要技术手段有：Transformer 语言模型、图相似度去噪、固定占位符编码、两层 MLP 投影、余弦相似度查询、KL 与对比学习蒸馏、两阶段自编码与自回归目标。

**📊 数据集**

使用了四个英文抽取式 QA 基准：SQuAD、AdversarialQA、HotpotQA、CovidQA；Encoder 与 Decoder 组合共九种（GPT‑2‑Large/OPT‑1.3B/OPT‑2.7B 与 Llama‑3‑8B/Mistral‑7B/Gemma‑2‑9B）。

**📈 对比分析**

与基线（仅使用局部窗口无压缩）和公开软压缩方法（AutoCompressor、xRAG、ICAE、PCC‑Lite/Large）对比，CMC 在大多数配置下显著提升 EM/F1，SQuAD 上最高提升 7.3 EM、4.0 F1；在 T=3,000 时推理时间/能耗下降 20%，峰值预留显存下降 50%–62.5%。

**⚠️ 局限性**

局限性：仅在抽取式 QA 任务上验证；固定压缩率 r∈{2,4,8} 可能不适应不同长度文本；未评估跨数据集泛化、非英文多语言场景；抽象式 QA、摘要或 RAG 任务需进一步改进。

---

## 77. Capability-Aware Arbitration for Semantic Intent-Based Shared Control

**arXiv ID:** 2609.25369 | [PDF](https://arxiv.org/pdf/2609.25369v1)

**作者:** Zhaoda Du `[一作]` (Colorado School of Mines), Xiaoli Zhang `[通讯]` (Colorado School of Mines)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉语言模型（VLM）推断人类意图、视觉语言动作模型（VLA）生成自主动作，并通过贝叶斯滤波和非线性阿尔法-贝塔映射动态分配机器人权威的能力感知共享控制框架。

**💡 创新点**

创新点在于同时考虑语义意图置信度和执行能力置信度，并将两者通过非线性sigmoid门控结合，以避免高意图置信度下机器人过度协助。

**🔧 技术方法**

核心技术包括温度缩放置信度校准、递归贝叶斯滤波、基于轨迹离散度与局部不稳定性的在线VLA能力估计，以及基于信任门控的权威分配策略。

**📊 数据集**

使用了400条ID演示数据（80条/5个指令），包含立方体和LEGO块；实验场景包括ID和OOD（不同尺寸/材质）任务。

**📈 对比分析**

与三种基线（手动遥控、固定50/50权威、仅语义意图权威）对比，所提方法在成功率上达到92%（ID 100%、OOD 83%），显著高于手动遥控83%、语义意图44%和固定50/50 10%；在任务完成时间、轨迹平滑度、控制友好度和权威误差等指标上也表现最优。

**⚠️ 局限性**

局限性包括VLA对用户运动偏好缺乏适应，导致主观体验略低于手动遥控；贝叶斯滤波在稳定性与响应速度之间权衡；以及在更长周期或更复杂任务中的能力置信度泛化性尚未验证。

---

## 78. Multi-Agent Video Prediction: Self-Correcting Conditional Frames for Dynamic Scene Forecasting

**arXiv ID:** 2609.25302 | [PDF](https://arxiv.org/pdf/2609.25302v1)

**作者:** Qixin Zhang `[一作]` (University of Minnesota), Zhi-Li Zhang `[通讯]` (University of Minnesota)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多代理视频预测框架，在远程驾驶场景下通过边缘持续预测、车辆端触发检测新对象以及基于遮罩的重构实现低延迟视频传输。

**💡 创新点**

创新点在于识别并解决预测仅方法在开放世界通信中出现的“新对象盲区”，通过轻量遮罩引导的重构实现语义恢复。

**🔧 技术方法**

采用MCVD进行连续预测，MagicQuill实现遮罩引导的条件重构，YOLO实例分割+跟踪作为触发器。

**📊 数据集**

使用Cityscapes数据集的真实城市驾驶视频，并在采集的5G上行通道轨迹上模拟通信延迟。

**📈 对比分析**

与仅使用预测的基线对比，场景恢复时间从平均500 ms降至275 ms，FVD提升至178.75，整体视觉质量保持不变，系统延迟略高但仍低于完整帧传输。

**⚠️ 局限性**

局限在于仅验证两帧上下文、仅处理遮罩级别的轻量重构，且评估仅针对Cityscapes，可能无法推广到更复杂多模态或更长时延的场景。

---

## 79. Can LLMs identify and repair ruptures? Comparison between clinician practices and LLM behaviors

**arXiv ID:** 2609.25287 | [PDF](https://arxiv.org/pdf/2609.25287v1)

**作者:** Jeongah Lee `[一作]` (University of Massachusetts Amherst), Ravi Karkar `[通讯]` (University of Massachusetts Amherst)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过基于心理治疗破裂修复理论（3RS）的21个情境，比较了三大LLM（ChatGPT、Claude、Gemini）与22名临床专家在识别和处理会话破裂时的表现。

**💡 创新点**

创新点在于首次将LLM与临床专家在破裂识别与修复任务中进行系统对照，提出“破裂识别与修复需区分上下文深度与时序性”的观点，并给出面向AI设计的“关系感知、节奏控制、人工干预”原则。

**🔧 技术方法**

使用技术包括大语言模型推理、情境化提示（prompt engineering）、多模型比较实验、定量标注一致性分析和定性主题编码。

**📊 数据集**

数据集为基于3RS手册构建的21条文本对话情境（每条约5回合），包含单一与混合破裂类型，并由22名具备临床经验的专家对标注与评价。

**📈 对比分析**

方法上先计算模型与专家在破裂类型（六大类与14子类）标注的一致性，再比较专家与模型在修复策略分布与效果评估（Likert 1–5）中的差异；实验结果显示LLM在破裂识别上近乎满分，而在修复效果上仅得到中等（平均分约3.0）且缺乏深度与时序适配。

**⚠️ 局限性**

主要局限包括：情境为预构造且文本化，未涵盖实时对话的动态性；专家样本规模仅22人，且为单国文化；LLM缺乏非语言线索，无法充分评估情绪与关系脉络。

---

## 80. ReFilter: Bridging Embeddings and LLM Filtering for Similar Mobile App Retrieval

**arXiv ID:** 2609.25306 | [PDF](https://arxiv.org/pdf/2609.25306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 81. 4DGS-JEPA: Temporally Compositional Joint-Embedding Prediction for Dynamic Gaussian Splatting

**arXiv ID:** 2609.25036 | [PDF](https://arxiv.org/pdf/2609.25036v1)

**作者:** Yongchao Huang `[一作]` `[通讯]` (University Of Aberdeen), Yongchao Huang (University Of Aberdeen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出4DGS-JEPA，一种基于动态高斯场的联合嵌入预测架构，用于可重用的多时距动态预测；

**💡 创新点**

创新点在于：①引入层次化高斯级别表示（场景‑运动组‑高斯），并采用时间条件转移器实现直接与递归预测；②提出时序组合原则（Temporal Composition），通过端点、路径监督与直接–组合一致性约束，消除不同预测路径的残差依赖；③设计选择性几何解码器与几何级别组合，以及混合对应机制（Hybrid Correspondence）处理高斯重排与拓扑变化；

**🔧 技术方法**

使用的技术包括：高斯喷射（3D Gaussian Splatting）与4D扩展、联合嵌入预测（JEPA）框架、层次注意力编码器、熵正则化的最优传输（OT）、Lie 群指数映射、Huber 与 SPD 距离度量、EMA目标、VICReg正则等；

**📊 数据集**

实验使用人工生成的基于高斯场的可控动态场景数据集，包含运动组、静态背景、分割/合并/出生/死亡等拓扑变化；未使用真实视频或动态场景重建数据；

**📈 对比分析**

与简单的静态持久化、常数速度外推等基线相比，4DGS-JEPA 在路径一致性、递归滚动误差、几何层级一致性方面显著提升；实验1显示时序组合可降低路径误差不影响目标精度；实验2证明几何级别组合可进一步减少解码后路径差异；实验3验证混合对应在身份扰动下的鲁棒性；整体性能以FDE、路径误差、几何一致性指标衡量；

**⚠️ 局限性**

局限性包括：仅在人工高斯场上验证，未评估真实动态场景重建；缺少对外观恢复的完整目标，仅使用几何预测；模型规模受限，未测试大规模动态场景；在极端拓扑变化与大规模高斯数量时对应机制的可扩展性尚待验证。

---

## 82. What Drives Hierarchy-Aware Image Retrieval? Taxonomy Alignment, Objective Choice, and Geometry

**arXiv ID:** 2609.25638 | [PDF](https://arxiv.org/pdf/2609.25638v1)

**作者:** Ling Shi `[一作]` `[通讯]` (Southeast University), Ling Shi (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在冻结的DINOv2特征上，用严格的跨类层次检索评估语义层次结构，并对几种因素进行因子实验；

**💡 创新点**

通过计算匹配的几何×损失因子实验，将语义对齐与损失家族对比与几何（欧氏/双曲）对比分离，揭示语义对齐与损失家族对比对检索性能影响更大；

**🔧 技术方法**

使用DINOv2 ViT-B/14预训练特征，768-256-32投影器；实现欧氏和双曲空间投影、距离回归与层次对比学习（SupCon）；进行几何×损失的2×2因子实验；

**📊 数据集**

CUB-200-2011（鸟类细粒度分类）和NABirds（更大规模鸟类数据集）两数据集；

**📈 对比分析**

在严格的跨类检索标准下，比较欧氏/双曲空间与距离回归与SupCon两种损失家族，使用mAP和Recall@K评估；结果显示，损失家族对比带来的平均层次mAP提升约0.04-0.05，几何对比提升约0.01或更小；而几何在层次级别上能重新分配检索性能；

**⚠️ 局限性**

实验仅在冻结DINOv2特征、32维投影器和两种鸟类数据集上进行，未对网络或数据做进一步微调；几何对比基于预设的双曲球参数，未做全局最优搜索；结果受限于单一特征源与特定数据集，未验证在其他模型或更大范围上的可推广性。

---

## 83. RoboMP-DINOv2: Prompts, Not Filters for Robust Robot Manipulation

**arXiv ID:** 2609.25506 | [PDF](https://arxiv.org/pdf/2609.25506v1)

**作者:** Han Qi `[一作]` (Harvard University), Heng Yang `[通讯]` (Harvard University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于掩码提示的全景视觉编码器RoboMP‑DINOv2，并结合局部颜色随机化MCR，实现机器人操作策略在视觉分布变化下的鲁棒性。

**💡 创新点**

创新点在于：①将掩码用作空间提示而非硬过滤器，保留完整场景信息；②在预训练的DINOv2特征空间中注入多区域提示并通过轻量级Transformer全局上下文化；③使用局部颜色随机化，仅扰动掩码区域，保持关键上下文。

**🔧 技术方法**

使用技术包括：DINOv2视觉Transformer、可学习的掩码提示嵌入、轻量Transformer上下文化层、全局平均池化、Diffusion Policy下的行动预测、CLIP文本编码、Grounding DINO+SAM生成掩码以及MCR颜色增强。

**📊 数据集**

实验数据集为七个模拟操纵任务（Place Cube、Put Can、Put Can+Obstacle、Pull Tool、Peg Insertion、Long Horizon、Two‑Arm Stacking），每个任务采集100条演示，并在三种OOD场景（空间、杂乱、颜色）下进行评估。

**📈 对比分析**

与2D/3D物体中心掩码、DP‑DINOv2、RoboGround‑DINOv2等基线比较，RoboMP‑DINOv2在空间和杂乱OOD下宏观平均成功率分别为60.7%和59.7%，显著高于50.7%和41.0%；RoboMP‑DINOv2‑MCR在颜色OOD下宏观平均成功率达72.5%，远超35.1%的最佳基线。

**⚠️ 局限性**

局限性包括：需依赖外部掩码生成器（Grounding DINO+SAM）且掩码误差会影响性能；实验仅在仿真环境进行，未验证真实机器人；仅针对可获得分割掩码的任务；缺乏多模态传感器或更大规模数据集的验证。

---

## 84. Brace Yourself: Task-Conditioned Environmental Bracing for Forceful Humanoid Manipulation

**arXiv ID:** 2609.25486 | [PDF](https://arxiv.org/pdf/2609.25486v1)

**作者:** Zongyuan Zhang `[一作]` (Queensland University of Technology), Jonathan M. Roberts `[通讯]` (Queensland University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种“支持手策略”（Supporting Hand Strategy），通过让机器人用一只手抓住环境来支撑身体，从而让另一只手在受力任务中获得更大的可持续受力能力。

**💡 创新点**

创新点主要有三：①利用环境支撑而非全身姿态规划来提升受力范围；②提出任务区域支撑优化（Task‑Region Support Optimiser），一次性求得覆盖整个任务区域的单一支撑配置；③将该支撑配置编码为关节关键点（WBKR），并驱动同步的强化学习体态与工作手策略，且不需要人类示范或在线全身轨迹规划。

**🔧 技术方法**

技术包括离线非线性优化求解支撑配置、两策略强化学习（PPO）同步控制身体与工作手、Whole‑Body Keypoint Representation (WBKR) 对姿态进行几何编码、以及在Unitree G1机器人上的硬件验证。

**📊 数据集**

数据集由30个任务样本（覆盖 0–75 N）和30个支撑配置组成，共 50,400 条训练样本；同时在硬件上对 6 个位置、5 个力级别进行测试。

**📈 对比分析**

与无环境支撑、无任务条件支撑、无参考以及基于 IK 的模块化硬件基准进行 ablation；结果显示 SHS 将可用受力从 13.5 N 提升至 60 N；在轨迹跟踪实验中，SHS 在 25–55 N 时约降低 30% 的力误差；在新任务区域转移实验中，无需重新训练即可保持 5–6 N 的力误差。

**⚠️ 局限性**

局限性包括：需保证任务区域能在单一支撑配置下可达；实验仅限于平面支撑面，未测试曲面、不同支撑手抓取方式或更高受力范围；对更复杂姿态或动态环境的适应性尚未验证。

---

## 85. Graph-Based Inference for Feedback-Driven Word Deduction: A Scalable Framework for the Jotto Problem

**arXiv ID:** 2609.25056 | [PDF](https://arxiv.org/pdf/2609.25056v1)

**作者:** Dakshi Arora `[一作]` (BML Munjal University), Ranjib Banerjee `[通讯]` (UPES)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于图的反馈驱动推理框架，用于变长（3-8字母）且可重复字母的Jotto单词推理。

**💡 创新点**

框架将候选单词构成加权词重叠图，利用反馈逐步剪枝，支持变长与重复字母，首次实现对Jotto的统一处理，并发现迭代次数随单词长度呈对数下降。

**🔧 技术方法**

图论、组合优化、信息论、回归分析、Python Flask后端实现。

**📊 数据集**

使用NLTK词库中3-8字母英文单词，约3000个模拟游戏场景进行实验。

**📈 对比分析**

通过多模型回归比较，发现对数模型拟合最佳（R²≈0.984），实验显示长单词收敛更快，迭代次数从3字母约11.5次下降至8字母约6次。

**⚠️ 局限性**

未针对极大词库做显式加速，未考虑多解情况的最终返回策略，主要在英文单词范围内实验，可能不适用于其他语言或更大词汇表。

---

## 86. "I Talked an AI Chatbot, So What's Next?" How U.S. Young Adults Imagine Responsible AI for Emotion Coping

**arXiv ID:** 2609.25311 | [PDF](https://arxiv.org/pdf/2609.25311v1)

**作者:** Jiaying Liu `[一作]` (University of Texas at Austin), Nimra Ishfaq `[通讯]` (University of Texas at Austin)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过情景化设计与访谈，探讨AI聊天机器人如何塑造情绪应对中的人际关系条件，并提出责任AI设计原则

**💡 创新点**

首次从关怀伦理视角提出AI在情绪应对中的关系条件框架，并识别八个AI角色与三大挑战

**🔧 技术方法**

情景化设计方法、访谈、NVivo编码分析

**📊 数据集**

17名美国青年成年参与者的访谈与情景设计工作

**📈 对比分析**

无可量化性能评估，仅通过质性分析识别角色与挑战

**⚠️ 局限性**

样本规模有限，缺乏多样性与跨文化验证，且未对技术实现进行实验评估

---

## 87. Mitigating LLM Over-Refusal via Dynamic Semantic Routing Calibratione

**arXiv ID:** 2609.25049 | [PDF](https://arxiv.org/pdf/2609.25049v1)

**作者:** Zixuan Wang `[一作]` (Jilin University), Dandan Guo `[通讯]` (Jilin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大型语言模型在安全对齐后出现的过度拒绝（over‑refusal）问题进行了机制分析，并提出了一种无训练的推理时干预方法——Semantic Routing Calibration（SRC），通过定位并动态抑制稀疏的“超敏感安全头”，从而在保持安全性的前提下显著降低硬安全提示（Hard‑Safe）导致的误拒。

**💡 创新点**

创新点在于：①发现硬安全提示下的稀疏超敏感安全头导致动态语义路由冲突；②提出在推理阶段局部定位这些头并按拒绝倾向动态抑制其值投影；③结合双分支 logits 融合，既消除误拒，又保持对真实有害指令的拒绝能力。

**🔧 技术方法**

技术包括：自注意力语义路由分析（计算词组注意力分配、注意力熵）、超敏感安全头评分与定位、推理时拒绝倾向校准（Δsim）与值投影缩放、双分支 logits 融合。

**📊 数据集**

使用数据集：对齐基准数据集（SFT/ RLHF 训练集）、Hard‑Safe 与 Unsafe 示例对（用于头定位），评测集包括 XSTest、CoCoNot、OR‑Bench、OKTest、PHTest、I‑Malicious、I‑CoNa、I‑Controversial、HarmfulQ、AdvBench、MMLU、ARC‑Easy/Challenge、OpenBookQA、PIQA 等；对比基线包括 STL、STL‑aug、DCR、SCD、SCANS、Surgical。

**📈 对比分析**

与训练基准（STL、STL‑aug、DCR）和训练免费方法（SCD、SCANS、Surgical）进行对比。SRC 在过度拒绝指标上提升显著（如 OR‑Bench 由 0.34 提升至 0.87），安全性能保持甚至略增，通用能力基本不下降；吞吐量几乎无额外开销。

**⚠️ 局限性**

局限性包括：需要先收集 Hard‑Safe 与 Unsafe 示例来定位头，受样本多样性影响；方法主要针对单轮生成，复杂多轮或长上下文的效果未知；超参数（缩放系数、头数、阈值）需模型特定调优；过强干预可能仍削弱对真实有害指令的拒绝能力。

---

## 88. When LLM Agents Fail to Read the Room: ReAdapt for Relational Social Reasoning

**arXiv ID:** 2609.25284 | [PDF](https://arxiv.org/pdf/2609.25284v1)

**作者:** Jianzhe Lin `[一作]` (MetaAI), Jubin Chheda `[通讯]` (MetaAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReAdapt 框架，在 ReAct 循环中加入显式社交状态更新，使观测到的社会证据在行动前被明确转换为状态变更，从而实现关系推理驱动的决策修正。

**💡 创新点**

创新点是引入可解释的关系状态与四种策略操作（continue、switch、abandon、clarify），迫使模型在获取新证据后显式地更新决策，而非仅靠语言模型上下文推理。

**🔧 技术方法**

采用大型语言模型 Gemini‑3‑Flash 的工具调用能力，并在 ReAct 结构上增设 Adapt 步骤，结合链式思考与树形思考等思路实现结构化状态转移。

**📊 数据集**

使用自研的 500 个合成社交世界数据集，包含 1000 个查询（反应选择和热情介绍两项任务），并在 300 个样本上进行验证。

**📈 对比分析**

与标准 ReAct 在相同模型、工具与调用预算下对比，ReAdapt 在热情介绍任务上整体准确率从 37% 提升至 51%，反应选择从 69% 提升至 77%，尤其在 53% 的 overturn 案例中提升显著（绝对 16%），同时 Oracle regret 亦显著下降。

**⚠️ 局限性**

局限在于实验仅覆盖 300 条查询的证明性验证，未细化每个状态维度或策略操作的具体贡献；缺乏真实社交数据和更大规模评估，也未探讨多任务或长期情境的泛化能力。

---

## 89. HABILIS Brain 0: Geometry-Change Supervision for Vision-Language-Action and Residual Flow Recovery

**arXiv ID:** 2609.25558 | [PDF](https://arxiv.org/pdf/2609.25558v1)

**作者:** Jinu Pahk `[一作]`, Byoung-Tak Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种四阶段训练框架（GC‑VLA + GCRF），通过多视角的几何变化监督学习视觉‑语言‑动作（VLA）策略，并在冻结的基础上使用几何条件残差流提升闭环性能。

**💡 创新点**

创新点包括：1）用未来‑当前几何变化（ΔE）作为监督目标，强调操纵过程中变化而非静态内容；2）分阶段训练：先预训练几何 VLM、再分离动作对齐、随后联合优化、最后冻结后通过二元路由器和有界残差补偿；3）在离线目标生成中仅使用机器人或人类视频，无需动作标注；4）引入 Geometry‑Conditioned Residual Flow（GCRF）在闭环反馈下实现选择性动作修正。

**🔧 技术方法**

使用的技术包括：Molmo2‑ER 视觉‑语言模型 + 额外 transformer 块；Depth Anything v3 与 DINOv2 生成伪深度和特征差；几何变化令牌化与交叉熵监督；连续 ActionExpert 通过 Flow‑Matching 训练；GCRF 的二元路由器、残差流、tanh‑bounded 速度补偿；多视角几何编码与 GeometryReader。

**📊 数据集**

训练数据来源多样：人机交互视频（egocentric）、机器人演示（跨胴体、多腕、单腕、桌面、家庭等），其中 Stage 1 只用几何监督，Stage 2 用验证过的 EEF 动作，Stage 3 采用 LIBERO 数据进行联合适配，Stage 4 使用 LIBERO 任务的闭环回放。主要公开基准为 LIBERO 任务集（4 套任务共 40 任务）。

**📈 对比分析**

与现有方法（MolmoAct2、π_0、OpenVLA-OFT 等）对比，GC‑VLA 基础在 LIBERO 上获得 95.20% 成功率，加入 GCRF 后提升至 99.55%，在四个标准任务套组中平均表现优于绝大多数竞争者。与直接训练基线相比，几何变化监督提升约 4% 成功率；GCRF 的分布式残差修正进一步提升 4.35% 成功率。

**⚠️ 局限性**

局限性包括：①几何变化监督与动作对齐的训练过程复杂，需要四个阶段并行调参；②最终性能仍依赖于闭环后训练（GCRF），单靠几何预训练无法达到最高水平；③在 LIBERO‑PRO 轻微扰动评估中性能不一致，显示对不同扰动的鲁棒性有限；④缺乏对跨机器人或真实场景的迁移性分析，主要验证在仿真环境内。

---

## 90. Hi-OPD: Hierarchy-Aware Open-Prompt Detection for Remote Sensing Images

**arXiv ID:** 2609.25584 | [PDF](https://arxiv.org/pdf/2609.25584v1)

**作者:** Jinlong Hu `[一作]` (Wuhan University), Shunping Ji `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种层级感知的远程遥感图像开源提示检测器，能够在多源数据中同时完成原子级检测和层级级别检索，并支持文本与视觉提示。

**💡 创新点**

创新点在于：①构建153个原子类别的稀疏层级与别名关系；②引入层级安全负采样、路径多正监督和单向向上一致性，解决多源标注粒度冲突；③开发ConvVPE，利用检测器自身多尺度特征生成视觉提示。

**🔧 技术方法**

采用的技术包括：基于WeDetect‑tiny的YOLO‑World风格BN对比检测头；文本编码采用缓存CLIP文本嵌入；视觉提示由ConvVPE对ROI对齐特征进行卷积编码并投影；训练策略分为两阶段：先多源文本检测，后加入层级目标；使用路径多正监督和向上一致性损失。

**📊 数据集**

使用了12个公开遥感数据集（DIOR、DOTA‑v2.0、FAIR1M、GLH Bridge、HRSC2016、LEVIR、NWPU‑VHR‑10、SIMD、SODA‑A、ShipRSImageNet、WHU Buildings、xView），共计175,644训练记录/3,477,064边框，56,695验证记录/1,297,548边框。

**📈 对比分析**

与WeDetect‑tiny（零样本、按源微调）、OpenRSD、LAE‑DINO、YOLO‑World‑L、Grounding DINO‑T等方法对比，在DIOR、DOTA‑v2.0原子检测上取得79.7/72.3 AP50，层级检索父级AP50 81.1/76.0，条件祖先召回率CAR50高达99.7%；在跨数据集提示泛化（HRRSD、RSOD、UCAS‑AOD、VEDAI）中宏平均AP50约63%，比OpenRSD在VEDAI上高6.2分。

**⚠️ 局限性**

局限性包括：①需要预先构建并维护层级与别名关系，若应用到新的任务需人工重构；②在极细粒度或稀缺类别上仍可能受限于训练数据；③目前仅支持2级层级，深层层级的效果尚未验证。

---

## 91. A Deployable Four-Finger Payload for Teleoperated Free-Flying Manipulation with Astrobee

**arXiv ID:** 2609.25614 | [PDF](https://arxiv.org/pdf/2609.25614v1)

**作者:** William Su `[一作]` (University of California, Berkeley), Masayoshi Tomizuka `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种可部署四指抓手及其双手遥操作管线，用于NASA的Astrobee自由漂浮机器人在低地轨道舱内搬运货物；

**💡 创新点**

创新点在于将17自由度的四指抓手与线性滑轨结合，并通过Quest VR设备将操作者手部姿态同时映射到两对指尖、指间距与机器人移动，实现高自由度的同步操控；

**🔧 技术方法**

技术实现包括基于MuJoCo的零重力ISS仿真、软L1成本优化的几何重定向求解、以及基于虚拟球的手腕共动控制来驱动Astrobee运动；

**📊 数据集**

实验数据来自10次模拟试验，使用ISS Cargo Transfer Bag（CTB）作为搬运目标，记录手部命令、抓手关节、机器人姿态与CTB位姿；

**📈 对比分析**

评估结果显示，平均遥操作时长77.5 s，货物相对机器人位置的RMS偏移为35.6 mm，峰值偏移67.7 mm，证明系统能够在机器人轨迹改变时保持货物；

**⚠️ 局限性**

局限性包括仅在仿真环境验证、硬件实现待完成、测试仅涉及单一货物类型、未加入碰撞感知控制以及仅有少数操作者试验。

---

## 92. Harnessing LLMs Without Surrendering Control: Delegation Boundaries in Visual Data Storytelling Authoring

**arXiv ID:** 2609.25700 | [PDF](https://arxiv.org/pdf/2609.25700v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 93. SambaGraph: Action-Reaction Spatio-Temporal Graphs for Soccer Tactical Response Modeling

**arXiv ID:** 2609.25569 | [PDF](https://arxiv.org/pdf/2609.25569v1)

**作者:** Abel A. Reyes-Angulo `[一作]` (Michigan Technological University), Steven Araujo `[通讯]` (University of Granada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了SambaGraph数据集，包含4,070条基于2022年世界杯跟踪与事件数据的行动-反应时空图示例，用于足球战术响应建模；

**💡 创新点**

创新点在于将动作-反应映射为时空图序列，提供了响应分类、攻击到防守检索和基于图摘要的LLM推理三种基准任务，并公开完整的图结构与配套评价指标；

**🔧 技术方法**

采用图张量表示、紧凑签名（signature）特征、动态图GRU、图-签名融合双编码器，以及LLM（如Llama‑3.2‑3B）进行图摘要推理；

**📊 数据集**

使用PFF FC Enhanced 2022 FIFA World Cup跟踪与事件数据，对64场比赛进行处理，生成23节点球员-球图序列与12节点团队视图；

**📈 对比分析**

与传统的图只编码、签名、随机/硬负样本双编码器等基线对比，签名MLP在响应分类上获得宏F1≈0.796，融合编码器在全库检索中Hit@10≈0.655，LLM在8候选重排序中保持原序列Hit@1≈0.98但未超越；

**⚠️ 局限性**

局限包括：基准不专注于预判，仅关注观察到的响应；签名特征在小样本环境下表现优于纯图模型；数据未包含门将、疲劳、战术指令等影响因素；标签粗化可能掩盖细粒度战术细节；

---

## 94. A Quaternary Legendre Pair of Length 64

**arXiv ID:** 2609.25573 | [PDF](https://arxiv.org/pdf/2609.25573v1)

**作者:** Harshit Verma `[一作]` `[通讯]` (Yale University), Harshit Verma (Yale University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

构造了首个长度为64的四元Legendre对

**💡 创新点**

首次在64长度上给出可验证的四元Legendre对，填补之前未解决的空白

**🔧 技术方法**

采用指数编码序列，计数残差法验证自相关，并用Python脚本进行全移位精确验证

**📊 数据集**

无外部数据集，使用自行生成的长度64指数序列

**📈 对比分析**

通过与已知长度（≤24、28-34等）对比，证明了在非零移位下自相关和恒为-2，满足Legendre对定义，验证准确性

**⚠️ 局限性**

仅针对长度64给出构造，尚未推广至更大长度或其他长度的构造方法

---

## 95. Open Science, Closed Models: How Funding Shapes AI in Science

**arXiv ID:** 2609.25347 | [PDF](https://arxiv.org/pdf/2609.25347v1)

**作者:** Ana Trišović `[一作]` (Massachusetts Institute of Technology), Janakan Sivaloganathan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过大规模文献计量分析，研究了科学论文中对AI基础模型的使用、扩展与开放权重参与，并探讨了不同资金来源与计算资源模式对这些行为的影响。

**💡 创新点**

首次系统量化公共、私人及云信用等资金机制与AI模型参与特征的关联，揭示公共资助倾向开放权重模型，云信用则偏向闭源大模型，行业合作在计算机科学领域显著提升模型扩展。

**🔧 技术方法**

采用多模态技术：GPT‑4.1‑mini 与 Llama‑3.1‑8B 进行文本分类、LLM 命名实体识别提取资助信息、线性概率模型与固定效应回归分析、逆采样权重校正、模型参数日志控制等。

**📊 数据集**

使用 Epoch AI Index 的基础模型列表、Semantic Scholar Academic Graph 与 S2ORC 的全文语料、OpenAlex 机构与国家信息，构建包含约 104,226 篇论文的完整样本。

**📈 对比分析**

通过带年份与领域固定效应的线性概率模型比较不同资金类别、云信用与行业合作对模型扩展、使用及开放权重使用的概率差异，结果显示公共资助提升开放权重使用约 5 pp，云信用降低约 3‑4 pp，行业合作提升扩展约 3‑4 pp，但整体解释度仅为 2‑3%。

**⚠️ 局限性**

局限性包括：仅捕捉公开披露的资助信息，无法识别隐性行业资源；样本缺失的国家与学科导致偏差；文本分类召回率有限；模型规模与开放性高度相关，因果推断不确定。

---

## 96. How Strongly Should Task State Influence an LLM Agent?

**arXiv ID:** 2609.25686 | [PDF](https://arxiv.org/pdf/2609.25686v1)

**作者:** Chenyu Zhang `[一作]` (University of Waterloo), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实验评估了一个“状态耦合梯子”，在固定任务规则、模型与配对对话的前提下，比较文本状态、完整清单、指令与强制执行四种不同方式将任务状态传递给LLM代理的效果。

**💡 创新点**

创新点在于通过统一实验框架和精确的payload匹配度量，系统揭示文本状态、指令遵从和强制执行对LLM代理执行可靠性的不同贡献，并在真实工具和外部基准上验证其适用性。

**🔧 技术方法**

使用了LLM编译状态机、匹配器、指令/强制门、检索记忆、LangGraph工具调度以及基于字符串匹配的精确评分技术。

**📊 数据集**

数据集包括自生成的采购/发布工程对话测试集、τ²‑bench航空政策、PM‑Bench以及基于LangGraph的工具使用模拟。

**📈 对比分析**

通过对多模型（35B、235B、V4Pro）、有无思考、不同图规模与密度的固定种子对照，使用严格 episode success 及误差通道分解进行比较；强制执行在大模型和任务状态可判定时提升至≈0.98，指令门提升至≈0.84，文本状态仅达≈0.38。

**⚠️ 局限性**

局限性包括仅覆盖单一代理单会话、一次性任务、有限的任务属性（依赖、取消、显式重做）、特定生成器与对话风格、以及对记忆和多语言支持的覆盖不足。

---

## 97. A Deployment Study of Identity-Gated Drone Gesture Control

**arXiv ID:** 2609.25511 | [PDF](https://arxiv.org/pdf/2609.25511v1)

**作者:** Diyari Mohammed Salih `[一作]` (Université Paris-Saclay), Naima Ait Oufroukh `[通讯]` (Université Paris-Saclay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并实现了IGate系统，在单摄像头的DJI Tello EDU上集成身份验证、手势识别与状态机仲裁，实现安全的手势遥控飞行。

**💡 创新点**

创新点在于将少样本人脸注册（20帧）、余弦相似度+滞后阈值身份门控与基于RBF‑SVM的手势分类结合，并在真实飞行中对离线与现场性能差异进行系统评估，量化并修正了六种飞行特有行为。

**🔧 技术方法**

使用技术包括：MediaPipe手势与人脸检测、MobileFaceNet ArcFace嵌入、RBF‑SVM手势分类、比例校正的面部跟踪、分层有限状态机仲裁、Qwen2.5‑0.5B仅用于日志说明、H.264编码、UDP多通道传输、滚动快门相机及低频率摄像帧压缩。

**📊 数据集**

数据集方面，构建了四个手势录制集（7类手势每类约400帧）用于离线评估；使用WebFace600K训练人脸嵌入网络，负样本来自LFW和LFW‑Wild；飞行时收集机载摄像头视频用于现场测试。

**📈 对比分析**

评估方法采用三种协议：within‑session、leave‑one‑session‑out、in‑flight，分别得到手势分类精度0.997、0.910、0.783；身份验证在离线EER 0.32% 对比飞行中19.3%；RBF‑SVM在飞行帧级准确率0.850，优于几何规则0.651，差距主要来自深度通道误差。

**⚠️ 局限性**

局限性包括仅验证单个受试者、手势与身份未绑定、仅在Hover‑Lock条件下评估、通信环境仅覆盖两地、未测闭环动力学影响、缺乏多用户或更复杂场景的鲁棒性研究。

---

## 98. Angular momentum analysis on Karate roundhouse kicks: a longitudinal case study

**arXiv ID:** 2609.25374 | [PDF](https://arxiv.org/pdf/2609.25374v1)

**作者:** Jan C. L. Lau `[一作]` (Karlsruhe Institute of Technology), Katja Mombaur `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究通过为期一年的纵向实验，分析空手道回旋踢动作中的角动量（AM）分布，并提出两种新的AM分解指标AMA（同向分量）和AMO（正交分量），用于评估踢法的稳定性。

**💡 创新点**

首次将AMA和AMO引入运动分析，提供了比传统AM百分比更直观、易解释的3D角动量分解方法，并尝试以此为基础探索基于AM的稳定性判据。

**🔧 技术方法**

采用Vicon 12摄像头运动捕捉结合Bertec力平台收集标记与力数据，并通过自研Python运动学管道（使用ezc3d与biorbd）计算关节角动量、总AM及其分解指标。

**📊 数据集**

使用两名受试者（学生与师傅）一年内共91次无目标回旋踢的数据集，包含标记、力平台信号以及依据专家标注的踢法稳定性分类。

**📈 对比分析**

通过对稳定、边缘稳定和不稳定踢法的AMA、AMO、总AM幅值、AM变化率等指标进行平均和标准差比较，发现稳定踢法呈现更高的AMA峰值、较低的总AM峰值及更大的负AMRCz（减速）表现，表明这些指标能区分踢法稳定性，但仍需进一步验证。

**⚠️ 局限性**

局限性包括样本量小（仅两名受试者）、仅研究单一踢法、AMRCz对零交叉敏感导致噪声放大、总AM接近零时AMA/AMO易出现数值不稳，以及未将AM指标与COM路径、力平台等变量整合形成完整稳定性判据。

---

## 99. Impact of Data Compression on Downstream AI Tasks: A Study using Teleoperated Driving over 5G

**arXiv ID:** 2609.25290 | [PDF](https://arxiv.org/pdf/2609.25290v1)

**作者:** Qixin Zhang `[一作]` (University of Minnesota Twin Cities), Zhi-Li Zhang `[通讯]` (University of Minnesota Twin Cities)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在5G上进行远程驾驶时，压缩摄像头和激光雷达数据对边缘/云端AI任务（如目标检测和语义分割）性能的影响；

**💡 创新点**

首次量化多模态（摄像头+激光雷达）数据压缩对AI性能的非线性、非均匀影响，并在多模态任务中找到最优压缩权衡点；

**🔧 技术方法**

采用BEVFusion（基于Swin-Transformer、VoxelNet和BEV编码器）的预训练模型，对不同压缩率（MJPEG质量因子、点云体素化尺寸）的摄像头与激光雷达输入进行实验；

**📊 数据集**

使用nuScenes数据集（nuScenes-mini子集，404样本，81个验证样本）作为输入与评估基准；

**📈 对比分析**

通过比较不同压缩率下的mAP（目标检测）和mIoU（语义分割）指标，发现压缩率越高性能越低，但在多模态下可通过调节体素大小与图像质量找到最佳平衡，整体性能下降幅度在10%–30%之间；

**⚠️ 局限性**

局限性包括仅在离线评估中验证，未涉及实时网络传输与时延；仅探讨MJPEG与体素化压缩方法，未考虑更高级压缩或自适应策略；以及仅评估了BEVFusion模型，未验证对其他模型的通用性。

---

## 100. SAM-V: Geometry-Aware Segment Anything for Multi-View Instance Segmentation

**arXiv ID:** 2609.25490 | [PDF](https://arxiv.org/pdf/2609.25490v1)

**作者:** Jiangshan Gong `[一作]` (University of Illinois at Urbana-Champaign), Derek Hoiem `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 SAM‑V，一种端到端的几何感知多视角实例分割框架，能够在单前向传播中实现跨视角一致的目标分割；

**💡 创新点**

创新点在于将 SAM 的 2D 分割先验与 VGGT 的多视角几何信息通过图像特征融合和几何感知提示融合机制相结合，使得提示能够被视角和空间信息所锚定；

**🔧 技术方法**

采用了 SAM 的图像编码器、VGGT 的几何编码器、视角与局部几何信息融合的提示模块以及跨视角聚焦的 mask 解码器，训练时采用焦点损失与 Dice 损失；

**📊 数据集**

在合成数据集 Hypersim 和真实室内数据集 ScanNet++ 上进行预训练与微调；

**📈 对比分析**

与 SAM2、Point‑SAM、IGGT、PanSt3R 等基线在 Hypersim、ScanNet++/ScanNet 上比较，SAM‑V 在 O‑IoU、T‑mIoU、帧级召回率等指标上提升约 5‑12 分，尤其在视角变化大、对象消失重现场景表现突出；

**⚠️ 局限性**

仅在室内场景训练，未验证户外或动态/柔性场景的泛化能力，且仍依赖静态几何假设，缺乏语言引导或开放词汇分割能力。

---

## 101. Queer inclusion in speech datasets: An audit and taxonomy of practical tensions

**arXiv ID:** 2609.25491 | [PDF](https://arxiv.org/pdf/2609.25491v1)

**作者:** Brooklyn Sheppard `[一作]` (University of Calgary), Levent Sagun `[通讯]` (Meta FAIR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过对八个英语语音数据集及两个面向 LGBTQIA+ 社群的数据集进行审计，量化了该社群在语音数据中的缺乏。

**💡 创新点**

创新点在于提出了“扩展性与包容性”“效率与参与度”“开放性与自治”“静态类别与流动身份”等四大张力的分类框架，以解释 AI 数据采集与 queer 社群需求之间的矛盾。

**🔧 技术方法**

方法主要是对数据集的标签方式、性别分布、机构属性、伦理审查、招募方式以及访问许可进行系统性审计与对比。

**📊 数据集**

所用数据集包括常见的 CCV2、Common Voice、Edinburgh Accents、English Dialects、Fairspeech、ARCTIC 以及专门为 queer 语音收集的 MAGES 与 PTMV。

**📈 对比分析**

比较方法是基于上述审计维度对数据集进行表格化与可视化，结果显示 queer 语音占比仅为 0–1.4%，且多数数据集在性别标签上保持二元化，导致偏差。

**⚠️ 局限性**

局限性包括只聚焦英语语料、仅检视性别身份维度、样本量不足、未评估模型性能，且对非二元身份的收集与处理仍不充分。

---

## 102. DefaultGNN: A Dual-Perspective GNN Framework for Predicting Corporate Default from Buyer-Seller Transaction Networks

**arXiv ID:** 2609.25542 | [PDF](https://arxiv.org/pdf/2609.25542v1)

**作者:** Junghoon Kim `[一作]` (KAIST), Chanyoung Park `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出DefaultGNN框架，从买卖双方视角利用多层交易网络预测企业违约。

**💡 创新点**

创新在于双视角门控融合与边权重化的交易规模敏感性，显著提升无历史违约数据下的预测能力。

**🔧 技术方法**

使用双层GCN编码器、门控融合、视图一致性正则化，并将交易规模映射为边权重。

**📊 数据集**

使用韩国电子税务发票交易数据（2018‑2023），共约5.6百万企业、约2.9亿条交易，配合信用评级机构违约标签。

**📈 对比分析**

与属性模型、单视图GNN、多关系GNN、复合多路网络对比，AR指标在所有年份和无历史违约子集上均高约3–6个百分点，且在OOS和SME场景表现最佳。

**⚠️ 局限性**

局限在于仅考虑买卖关系，未纳入供应链外部宏观因子，且对极端事件的泛化能力仍待进一步验证。

---

## 103. From Pattern Recognizers to Personalized Companions: A Survey of Large Language Models in Mental Health

**arXiv ID:** 2609.25186 | [PDF](https://arxiv.org/pdf/2609.25186v1)

**作者:** He Hu `[一作]`, Qi Tian `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了大型语言模型在精神健康领域的发展，提出了三阶段演化框架并系统评述核心技术与数据资源

**💡 创新点**

创新点在于将 LLM 应用从信息工具→同理对话者→长期个性化伴侣三阶段框架化，并聚焦状态化智能体架构（Profile/Memory/Reasoning/Planning/Tool Use）

**🔧 技术方法**

核心技术包括领域预训练、参数高效微调、强化学习对齐、Chain‑of‑Thought 推理、检索增强生成、跨模态融合以及多智能体协同

**📊 数据集**

使用了大规模文本识别数据（如 RSDD、UMD、SWDD）、对话与诊断语料（如 EmpatheticDialogues、PsyQA、MusPsy、SIMPsyDial）、以及多模态数据集（AVEC、WESAD、MEDIC、MESC）

**📈 对比分析**

通过对比安全基准（SafeBench、PsyCrisis‑Bench）、临床适配评测（CBT‑Bench、CounselingBench）和动态模拟（ψ‑Arena、WiseMind）展示，现有系统在准确性、同理心、专业性等指标上表现可观但仍低于真人临床水平

**⚠️ 局限性**

局限性主要包括缺乏真实临床验证、合成数据与真实交互的结构差异、潜在偏见与安全风险、以及长期记忆与规划机制尚不成熟

---

## 104. Testing and Learning Symbolic Finite State Machines

**arXiv ID:** 2609.25603 | [PDF](https://arxiv.org/pdf/2609.25603v1)

**作者:** Wen-ling Huang `[一作]` (University of Bremen), Jan Peleska `[通讯]` (University of Bremen)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了符号有限状态机（SFSM）的完整测试与学习方法，并通过构造有限代表性输入集把无限输入域的测试与学习问题化简为有限输入域的 DFSM 问题；

**💡 创新点**

提出了代表性输入集的概念，证明了在满足 guard 重叠与输出区分条件下，有限实例的语言等价即可推出整个 SFSM 的语言等价；该结论使得任何完整的 DFSM 测试方法都可直接应用于 SFSM；同时给出了基于 SMT 的构造算法和学习后提升到 SFSM 的方法；

**🔧 技术方法**

使用了符号执行、SMT 求解、L*（及其变种）学习算法、W‑method 完整测试以及理论分析证明；

**📊 数据集**

论文中未使用公开数据集，实验与评估均为理论构造与证明；

**📈 对比分析**

与传统 DFSM 测试/学习方法对比，只需在构造代表性输入集后直接复用现有完整 DFSM 方法，性能上只增加有限的 SMT 计算开销，且不需要额外的 ω^d 转换等假设；

**⚠️ 局限性**

局限在于仅适用于确定且完全指定的 SFSM，且假设已知可用 guard、output 赋值集合以及实现状态上界；对于非确定、部分或自回归输出的情况尚未覆盖；

---

## 105. Quantum ROP: Using Quantum Algorithms for ROP Chain Selection in Exploit Construction

**arXiv ID:** 2609.25364 | [PDF](https://arxiv.org/pdf/2609.25364v1)

**作者:** Carlos Benitez `[一作]` `[通讯]` (PLATINUM CIBER), Carlos Benitez (PLATINUM CIBER)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将ROP链选择建模为QUBO，并在IBM Heron r2上使用QAOA完成Linux内核权限提升链的构造与验证

**💡 创新点**

首次将量子优化算法QAOA用于实战ROP链生成，提供完整的NISQ可实现实验流水线

**🔧 技术方法**

使用QUBO建模、QAOA变分量子优化、IBM Heron r2超导量子硬件、SPSA梯度无关优化、Capstone等二进制分析工具

**📊 数据集**

8个Linux内核与5个用户空间二进制（包括hxpCTF 2020内核），共约140万条候选gadget，最终10个可选gadget的QUBO实例

**📈 对比分析**

与穷举搜索、贪心、模拟退火等经典方法对比，QAOA在多数实例下恢复最优链，但受电路深度限制，性能不如贪心且仅在深度<200时可获得有效结果

**⚠️ 局限性**

受NISQ设备深度与错误率限制，无法处理深度>200的实例；模型未包含堆栈偏移约束，仅验证保留候选集内最优，未扩展到更大规模或更脏gadget情形

---

## 106. Gaze responses to false-positive computer-aided detection prompts during colonoscopy: a paired-video and real-time eye-tracking study

**arXiv ID:** 2609.25581 | [PDF](https://arxiv.org/pdf/2609.25581v1)

**作者:** Te Luo `[一作]` (Fudan University), Shuo Wang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了假阳性CADe提示对内镜医师注视的捕获与持续影响，并在配对视频实验与实时临床记录中使用事件锁定眼动追踪评估该效应。

**💡 创新点**

创新点在于把评估焦点从提示频率转为事件级注视动力学，量化了关注持续时间与时间放大效应，并首次将此与真实临床流程相结合。

**🔧 技术方法**

使用Tobii Pro Nano眼动仪、30 Hz采样率的眼动追踪，配合EndoAdd 2023版CADe系统的边界框提示，实施事件锁定眼动分析。

**📊 数据集**

数据集包括5名内镜医师在60段录制视频中完成的配对实验（共28个假阳性事件），以及9名高级医师在42个实时CADe辅助检查中收集的817个假阳性事件与60个真实病变事件。

**📈 对比分析**

通过在同一视频中对照无提示与有提示条件，量化捕获比例（48.6% vs 65.2%）、关注持续时间（约1 s）、时间放大倍数（17.55倍 vs 5.15倍）以及病变ROI识别率（≈100%），表明假阳性提示能显著吸引并持续注视。

**⚠️ 局限性**

主要限制包括样本量仅5名医师、实验顺序未对照、实时记录缺乏未辅助对照、眼动采样率30 Hz限制短时测量、且仅评估单一CADe系统，外推性受限。

---

## 107. MotionForge: A Data Generation Pipeline and Large-Scale Benchmark for Long-Horizon Manipulation of Dynamic Objects with Domain Shifts

**arXiv ID:** 2609.25689 | [PDF](https://arxiv.org/pdf/2609.25689v1)

**作者:** Mohan Liu `[一作]` (Nanyang Technological University), Lin Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出了MotionForge，一个大型仿真基准和数据生成管道，用于评估动态操作中的领域偏移和长时序交互。

**💡 创新点**

创新点在于：①系统化的单因子和联合领域偏移评估协议；②分离环境演进与策略推理的实时执行协议；③覆盖40个任务、11种运动模式、17个长时序任务，以及100+目标物体和5K背景，实现对动态操作的全面评估。

**🔧 技术方法**

采用Isaac Sim仿真、两阶段任务构造与验证管道、可复用的技能与资产库，以及对策略推理时间和动作执行时延的实时监测。

**📊 数据集**

使用由MotionForge生成的20K演示数据集，包含100+目标物体、5K背景和多种运动速度，共计40个任务和17个长时序任务。

**📈 对比分析**

对七种代表性机器人策略（包括任务特定、通用VLA、世界模型等）进行比较；在域内的整体成功率最高为22.20%，但在联合域外偏移下成功率降至5.85%，显示现有策略在多因素动态环境下的显著不足。

**⚠️ 局限性**

局限性包括：①在联合域外偏移下性能严重下降，未能满足复杂动态场景的鲁棒性；②对实时推理时延的敏感性高，需进一步优化；③当前评估仅在仿真环境，缺乏对真实移动操作的验证。

---

## 108. Evidence-gated multimodal parsing and vectorization of architectural floor plans

**arXiv ID:** 2609.25615 | [PDF](https://arxiv.org/pdf/2609.25615v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 109. Data center cooling choices shift water impacts across the grid: An integrated water-energy model for sustainable data center development

**arXiv ID:** 2609.25437 | [PDF](https://arxiv.org/pdf/2609.25437v1)

**作者:** Garrett Alston `[一作]` (University of Michigan), Rabab Haider `[通讯]` (University of Michigan)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个集成水‑能量模型，能够将数据中心冷却配置、IT负荷、经济调度和子流域耗水风险耦合，评估数据中心开发对直接与间接水耗的空间时间影响。

**💡 创新点**

创新点在于：①同时跟踪直接（冷却用水）与间接（电网发电用水）水耗；②将数据中心负荷与电网经济调度耦合，识别实际响应的发电机及其水消耗强度；③将结果映射到HUC‑8子流域，计算耗水风险（耗水比率），从而揭示水耗超出设施范围的扩散效应。

**🔧 技术方法**

使用三层模型：
• DCF（数据中心设施）模型：利用PUE/WUE计算IT功率与冷却用水；
• Energy System：基于PyPSA+HiGHS的线性经济调度，得到各发电机小时输出与水消耗强度；
• Water System：采用USGS NWA的月度水可用性/消耗数据与HUC‑8聚合，计算子流域耗水风险。

**📊 数据集**

数据集：DOE 2024 Data Center Energy Usage Report（PUE/WUE）、EIA‑860（发电机容量与WCI）、REEDS与NSRDB（风/光可用性）、Open Energy Data Initiative（电力负荷）、USGS NWA（水可用性/消耗）以及HUC‑8地理边界。

**📈 对比分析**

在密歇根州的两种情景（冷却配置与大规模建设）中对比模型输出。结果显示，空气冷却虽直接用水降低约97% 但间接用水上升32%，整体水耗下降44%；间接水强度随月份变化约50%；与传统平均网格水强度法相比，模型能更精确捕捉季节性和空间差异，且能明确识别哪些子流域受到最大影响。

**⚠️ 局限性**

局限性：①设施层面仅使用平均PUE/WUE，未考虑动态负荷与控制策略；②电网模型仅按四区粗网格划分，忽略细粒度输电拥堵与发电机实时调度；③水文模型仅到HUC‑8聚合，可能掩盖局部水资源压力；④部分发电机WCI缺失需估算；⑤未涵盖Scope 3（建设、设备制造等）水足迹。

---

## 110. Point Diffusion Mamba: Unified Diffusion-State-Space Modeling for Single-View 3D Reconstruction under Data Scarcity

**arXiv ID:** 2609.25538 | [PDF](https://arxiv.org/pdf/2609.25538v1)

**作者:** Wei Zhou `[一作]` (Northwest University), Ying He `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Point Diffusion Mamba（PDM），一种结合扩散模型与状态空间模型的单视角3D重建框架，在数据稀缺条件下实现高效点云重建。

**💡 创新点**

①轻量级局部几何聚合（LGA）与双向Mamba块联合捕捉局部细节与全局结构；②层次特征融合网络（HFINet）弥合高层语义与局部点特征的鸿沟；③动态加权采样（DWS）将生成先验与重建结果自适应融合。

**🔧 技术方法**

扩散模型（DDPM）、Mamba状态空间网络、ViT特征投影、KNN/FPS分组、局部几何聚合（LGA）、层次特征融合（HFINet）、动态加权采样（DWS）。

**📊 数据集**

ShapeNet（Chair、Airplane、Car）和Pix3D（Chair、Table、Sofa）数据集，采用R2N2渲染图像进行训练与评估。

**📈 对比分析**

与PC2、CCD-3DR、BDM-M/B、MESC-3D等SOTA方法在10%/50%/100%数据规模下进行对比，PDM在ShapeNet上CD/F1均显著优于对手，Pix3D上取得最低CD和最高F1，且在极端1%–5%稀缺下仍保持较好性能。

**⚠️ 局限性**

对大规模点云或非结构化大场景的可扩展性尚待验证，且依赖预训练ViT投影和生成先验，在极度噪声或多视角缺失情况下可能表现受限。

---

## 111. MT-ProtBERT: Multi-task Learning ProtBERT for Intrinsically Disordered Proteins Classification with Scarce Data

**arXiv ID:** 2609.25334 | [PDF](https://arxiv.org/pdf/2609.25334v1)

**作者:** Jian Sun `[一作]` (University of Denver), Mohammad H. Mahoor `[通讯]` (University of Denver)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种多任务 ProtBERT（MT‑ProtBERT）框架，用于在数据稀缺条件下对无序蛋白（IDP）进行磷酸化位点预测和蛋白质压缩度预测。

**💡 创新点**

创新点包括：①动态窗口掩码（Dynamic Window Masking）实现多视角数据增强；②多尺度 1D 卷积分类器（MS‑Conv1D）捕获不同尺度的序列特征；③结合 MLM 与生物化学辅助任务（S/T‑P 或 SCD）进行多任务学习；④引入蛋白质感知损失（Protein‑aware Loss）在 MLM 训练中加入化学等价性信息。

**🔧 技术方法**

核心技术包括 ProtBERT 预训练模型、动态窗口掩码、MS‑Conv1D 分类头、双视角 MLM 头、S/T‑P 或 SCD 辅助分类头，以及蛋白质感知损失函数。

**📊 数据集**

使用的公开数据集有：PPA（植物磷酸化位点，19mers）和 PELM（动物磷酸化位点，19mers）用于磷酸化任务；PELM‑684（684 条短序列）和 AD‑530（530 条 30mers）用于蛋白质压缩度预测。

**📈 对比分析**

与基线 PARROT（RNN‑IDP 模型）和 MusiteDeep 进行对比；在所有子任务和数据子集上，MT‑ProtBERT 在准确率、F1、AUPRC、敏感度、特异度、MCC 等指标上均优于 PARROT，特别是在样本极少的子集（如 PPA‑Y）提升显著；消融实验验证了每个模块的贡献。

**⚠️ 局限性**

局限性：①短序列导致 MLM Top‑1 预测精度偏低；②S/T‑P 辅助任务过于简单，难以进一步提升主任务；③目前仅针对长度 ≤30 的等长序列，尚未验证可变/长序列的泛化；④仍需探索更丰富的物理/化学先验和更强的自监督方法。

---

## 112. GameDirector: Decoupling Gameplay Logic from Rendering for Player-Configurable Game World Models

**arXiv ID:** 2609.25652 | [PDF](https://arxiv.org/pdf/2609.25652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 113. RULER: Instance-aware Rubric Rewards for SVG Generation

**arXiv ID:** 2609.25270 | [PDF](https://arxiv.org/pdf/2609.25270v1)

**作者:** Hangyu Ran `[一作]` (Ant Group), Han Peng `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了如何使用实例感知的多维度rubric作为奖励来训练SVG生成模型，提升生成质量并克服传统scalar metrics的不足。

**💡 创新点**

提出RULER框架，将每条指令转化为六项实例化rubric，利用细粒度评估指导强化学习，显著提高SVG生成效果。

**🔧 技术方法**

采用前沿视觉语言模型（VLM）作为评估判定器，使用Claude-Opus-4.6等生成器生成实例化rubric，并通过Group Relative Policy Optimization（GRPO）进行训练。

**📊 数据集**

在MMSVG-Illustration和MMSVG-Icon两个基准上进行评测，并用900条人工标注样本验证评估指标。

**📈 对比分析**

与Diffusion、LLM基础模型、SVG专家模型等基线对比，RULER在Rubric分数上分别达到0.693/0.683，优于所有基线并匹配DeepSeek-V3，同时在人类偏好测试中胜率均超过50%。

**⚠️ 局限性**

依赖外部VLM和rubric生成器，计算成本高，rubric设计可能无法覆盖所有艺术意图，且可能出现评估偏差或鲁棒性问题。

---

## 114. Transformer Heads Looking for Order

**arXiv ID:** 2609.25588 | [PDF](https://arxiv.org/pdf/2609.25588v1)

**作者:** Jasper van Doornmalen `[一作]` (IMC UC), Przemysław Andrzej Wałȩga `[通讯]` (Queen Mary University of London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了 ORDERED 语言可由 2-head 1-layer transformer（含输出 MLP）计算，但不可由 1-head 1-layer transformer（含输出 MLP）计算。

**💡 创新点**

创新点在于利用几何半空间分割下界证明，避免乘法，仅用单个注意力头即可构造下界；同时给出了 2-head 的构造实现。

**🔧 技术方法**

使用了注意力机制、位置编码、ReLU MLP、实数一阶理论的量化消除以及几何半空间覆盖与 Ramsey 理论等技术。

**📊 数据集**

无数据集，全部为理论证明与数理推导。

**📈 对比分析**

通过理论比较头数提升的计算能力，展示 2-head 能够实现 1-head 不能实现的语言；未给出实验性能指标。

**⚠️ 局限性**

局限性：仅针对 1‑层 transformers；对 k≥3 的添加性输入嵌入尚未得到分离证明；下界依赖于输出 MLP，且缺乏实验验证。

---

## 115. Synthesis and editing of multi-instrument audio mixtures using scalar-quantised latents with MIDI Span conditioning

**arXiv ID:** 2609.25546 | [PDF](https://arxiv.org/pdf/2609.25546v1)

**作者:** Sungkyun Chang `[一作]` (Queen Mary University of London), Emmanouil Benetos `[通讯]` (Queen Mary University of London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了SpanSynth-Edit，一种基于MIDI Span和低帧率量化潜在的流匹配模型，可在保留原始音色的前提下对多乐器音频进行高精度的增删改音符编辑。

**💡 创新点**

创新点在于将MIDI Span作为无序事件集合的帧对齐表示，利用可置换编码聚合为条件向量；以及将上下文音频与MIDI一起作为条件，在低帧率潜在空间中实现流匹配生成。

**🔧 技术方法**

采用了HeartCodec的量化潜在编码、条件流匹配扩散Transformer（DiT）以及Permutation-invariant的Deep Sets编码；并在推理时使用Euler积分和classifier-free guidance。

**📊 数据集**

使用了17个公共器乐数据集（包括Slakh、MusicNet、URMP、GuitarSet、MAESTRO、POP909等）共计约660小时的MIDI-音频配对数据进行训练与评测。

**📈 对比分析**

在多乐器合成与编辑基准上与CTD、TokenSynth、SpecDiff、U-MusT、MIDI-VALLE等现有方法进行对比，SpanSynth-Edit在音质、相似度和音符遵循率指标上取得或接近最佳表现，且在编辑任务中实现最高的新增/保留音符准确率。

**⚠️ 局限性**

主要限制包括依赖外部转录模型导致的音符遵循误差、量化潜在与编码器预训练数据分布不匹配导致的重建误差、以及对小于5毫秒的MIDI时序偏移敏感性不足；缺乏专家听觉评测。

---

## 116. Real-Time Hand Gesture Recognition for OpenXR Using Transformer-Based Machine Learning

**arXiv ID:** 2609.25466 | [PDF](https://arxiv.org/pdf/2609.25466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 117. Geometric and Semantic Coupling for Interaction Understanding in 3D Scenes

**arXiv ID:** 2609.25247 | [PDF](https://arxiv.org/pdf/2609.25247v1)

**作者:** Hanyang Kong `[一作]` (National University of Singapore), Xingyi Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种利用RGB点云对三维场景中可移动部件、运动参数和交互区域进行联合预测的方法，并通过部件与手柄的相互关联来消除运动歧义并提升手柄检测。

**💡 创新点**

创新点在于：①将手柄位置与部件运动几何关联，构造无训练的铰链选择规则；②利用部件预测生成补充手柄候选并依据部件运动类别校正手柄标签；③在一次推断中实现双向信息传递，既改进运动估计又提升手柄检测。

**🔧 技术方法**

使用的技术包括：独立训练的Volt-B voxel Transformer + SPFormer解码器；基于几何的无训练运动解码器；基于连通组件和投影的点级手柄检测；父子查询式手柄候选生成与标签修正机制。

**📊 数据集**

在 Articulate3D 数据集上进行训练与评估，该数据集提供可移动部件、运动参数和交互区域的标注。

**📈 对比分析**

通过与 USDNet 等公开基准对比，验证集上实现 47.93% 可移动部件 AP、40.98% 运动门控 AP、30.99% 手柄 AP；测试集上获得 48.28% 部件运动 AP 与 34.46% 手柄 AP，居榜首。

**⚠️ 局限性**

局限性包括：仅适用于平面且竖直轴旋转的部件，无法处理水平或倾斜铰链、曲面机制；手柄缺失或部件检测不足会导致错误；整体覆盖率仍有提升空间，需要更完善的部件检测与更通用的运动模型。

---

## 118. FrontierMath Erdős

**arXiv ID:** 2609.25050 | [PDF](https://arxiv.org/pdf/2609.25050v1)

**作者:** Tom Adamczewski `[一作]` (Epoch AI), Thomas F. Bloom `[通讯]` (University of Manchester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 FrontierMath Erdős (FME) 基准，对 68 个 Erdős 开放猜想进行统一评测，并在固定预算下评估 AI 系统的证明/反证能力。

**💡 创新点**

创新点在于首次系统地在可验证的正式证明框架（Lean）下评估 AI 在真正开放数学问题上的能力，解决了以往缺乏统一任务、预算透明度和可重复性的问题。

**🔧 技术方法**

使用的技术包括 Lean 4 证明助手、Comparator 证明检查器、基于 ReAct 的 AI 代理以及离线 arXiv 论文语料库，辅以自动化形式化与工具化环境。

**📊 数据集**

数据集由 68 个在 Mathlib 中形式化的 Erdős 猜想构成，涵盖 50 个已存在于 Formal Conjectures 库的猜想和 18 个自行形式化的新猜想。

**📈 对比分析**

对比方法：在每个猜想上仅一次尝试、$300 预算、72 小时时限，评测五个模型；GPT‑6 Astra 仅解决 2 个（3%），其余四个全未通过，显示当前 AI 的解决率仍极低。

**⚠️ 局限性**

局限性包括正式化成本可能低估数学能力、未考虑将猜想归约至已知大问题、模型训练数据泄漏风险、对“重要新思想”的评价缺乏自动化、以及仅在证明/反证框架内评估而忽略发现新问题的能力。

---

## 119. The ISAC Tradeoff Cliff: Fundamental Limits under Waveform Uncertainty and Finite Blocklength

**arXiv ID:** 2609.25589 | [PDF](https://arxiv.org/pdf/2609.25589v1)

**作者:** Mohammed Zafar Ali Khan `[一作]` (Indian Institute of Technology Hyderabad), Lajos Hanzo `[通讯]` (University of Southampton)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在有限块长度（FBL）通信下，解码误差如何导致接收机使用的参考波形失真，从而显著影响ISAC（集成感知与通信）系统的感知性能，并给出了基于Fisher信息的精确建模与分析。

**💡 创新点**

创新点包括：① 将FBL解码误差建模为等效高斯噪声并推导其对感知 Fisher 信息与CRB 的影响；② 发现并量化了“Tradeoff Cliff”，即当通信速率逼近 Shannon 容量时，感知性能会出现急剧衰落；③ 解析了该崩溃随块长度、信噪比、误码率阈值的缩放关系（如安全裕度 Θ(n⁻¹/²)）；④ 提出可靠性感知优化框架，给出了在通信 QoS 约束下的感知性能上界；⑤ 通过多种仿真（AWGN、块 fading、Polar 编码等）验证了理论。

**🔧 技术方法**

技术主要包括：信息理论中的有限块长度可靠性近似（Q 函数），等效协方差模型，Fisher信息与 Cramér–Rao 边界推导，可靠性感知优化（约束最小化 CRB），以及 Monte Carlo 仿真与 Polar 码的实际性能评估。

**📊 数据集**

使用的数据集主要是仿真数据：AWGN 与 Rayleigh 随机信道模型、不同块长度 n、SNR 设定以及 Polar 编码的实际误码率与残差协方差。论文未使用公开实验数据集。

**📈 对比分析**

对比方法：与传统资源分配（功率拆分）ISAC 方案和单纯可靠性不考虑的理论上限做对比。结果显示：在低速率下感知性能接近理想上限；随着速率升至容量附近出现“Tradeoff Cliff”，传统方案无法预测；可靠性感知优化在满足通信 QoS 的前提下能显著提升感知精度。总体性能表现：理论与仿真在 Fisher 信息和 CRB 上高度一致，验证了等效噪声模型的有效性。

**⚠️ 局限性**

局限性包括：① 等效高斯噪声模型仅考虑二阶统计，未捕捉编码器特定的结构化误差或突发错误；② 只考虑单天线、单参数（或多参数但同量级）情形，未扩展到多天线 MIMO 或频域多载波；③ 对 κ（误差能量比例系数）的依赖需通过实验或仿真校准，模型参数与系统实现高度相关；④ 仅关注感知精度的 Fisher 信息/CRB，未讨论能耗、时延等系统综合指标。

---

## 120. The AI Neuroscientist: An Interactive Agentic Interface for Neuroimaging Analysis

**arXiv ID:** 2609.25254 | [PDF](https://arxiv.org/pdf/2609.25254v1)

**作者:** Aakash Patel `[一作]` (Yale University), David van Dijk `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了AI Neuroscientist——一种将大型语言模型与神经影像工具集结合的交互式自然语言分析代理，支持从数据发现、质量控制到建模可视化的完整流程；

**💡 创新点**

创新点在于构建了基于LangGraph/LangChain的ReAct代理，利用固定工具集实现严格阶段化执行、可视化证据生成与二次多模态推理，突破传统脚本式流水线的可访问性与透明度；

**🔧 技术方法**

技术包括大型语言模型（如gpt-5、Claude-sonnet）、LangGraph/Chain的工具调用框架、BIDS/SNIRF标准化文件解析、fNIRS专用处理工具（心跳SNR、TDDR、GLM+FDR、Topomap绘制）以及多模态（文本+图像）推理；

**📊 数据集**

使用了自定义的fNIRS Benchmark Suite（包含合成与真实OpenNeuro数据），涵盖心跳质量控制、运动尖峰检测和解剖定位三类任务；

**📈 对比分析**

通过与两款在Docker沙箱内运行的通用LLM编码代理（gpt‑5‑nano、claude‑sonnet‑5）在六个任务上对比，AI Neuroscientist整体得分0.79，显著高于0.45和0.14，尤其在真实数据的质量控制和尖峰检测任务中优势突出；

**⚠️ 局限性**

局限性包括：仅适用于小规模交互式探索，未验证批处理或临床决策场景；缺乏用户研究证明对科研效率或准确性的提升；目前仅覆盖fNIRS，尚未扩展到fMRI、EEG等；代理性能受限于已预置工具集的完整性与健壮性。

---

## 121. VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models

**arXiv ID:** 2609.25376 | [PDF](https://arxiv.org/pdf/2609.25376v1)

**作者:** Jiuyi Xu `[一作]` (Colorado School of Mines), Yangming Shi `[通讯]` (Colorado School of Mines)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 VLAQuantBench，一个统一的评估框架，用于系统地比较四种视觉-语言-动作模型在不同量化方案下的闭环任务完成率。

**💡 创新点**

创新点在于揭示量化的非单调性与交互效应，证明在某些模型和层级上量化更多层反而提升成功率，并给出针对性修复策略（如保护输出投影或对动作头进行两帧校准）。

**🔧 技术方法**

采用了 RTN（四舍五入）整数后训练量化、层级组消融、两帧校准、同序列重放、真实内核计时以及机器人实验等技术手段。

**📊 数据集**

使用了 LIBERO（Spatial、Object、Goal、Long）、SIMPLER、CALVIN、VLABench 等四大机器人仿真数据集，并补充了 20 条完整的高精度轨迹做重放。

**📈 对比分析**

通过 409 次量化配置跑 94,574 条仿真轨迹，并与原始全精度基准对比，发现如 W4A4 在无校准时会导致严重失败，但校准或保护关键层能恢复到接近基准的成功率，整体评估显示不同模型对同一量化设置的敏感性差异显著。

**⚠️ 局限性**

局限性包括仅覆盖四个模型与特定检查点、量化仅为模拟 RTN（未覆盖所有整数核实现）、校准样本有限、任务簇样本数少、未与所有 VLA 量化方法在统一实现下对比，以及机器人实验未对所有量化方案进行充分重复。

---

## 122. TCMaster: Confidence-Aware Querying and Workload-Guided Physical Design for Multi-Source Traditional Chinese Medicine Knowledge Graphs

**arXiv ID:** 2609.25712 | [PDF](https://arxiv.org/pdf/2609.25712v1)

**作者:** Zheng Chen `[一作]` (Tsinghua University), Peiwu Qin `[通讯]` (Guangdong Provincial Laboratory of Traditional Chinese Medicine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了TCMaster知识图数据库，集成多源传统中医知识，并实现了基于边缘可信度的路径查询和工作负载导向的物理设计；

**💡 创新点**

创新点在于将边缘可信度标注、路径级可信度聚合、属性位图、方向反转以及跨层快捷边等技术结合，形成可解释的可信查询子系统；

**🔧 技术方法**

采用Neo4j 4.4社区版存储，利用Python进行ETL与数据清洗，使用Cypher重写查询、LLM提取微语义、KGE做结构验证，并实现bitmap位图、方向选择和物理索引；

**📊 数据集**

使用的核心数据集包含221,225个实体、722,671条基础边和28.75M条快捷边，来源于HERB 2.0、SymMap、经典处方、LLM提取微语义等四类资源，覆盖五层知识层次；

**📈 对比分析**

通过与Neo4j原生计划对比实验，方向选择提升约1.47倍，快捷边在高Fanout计数时提升约4.42倍；整体查询时延均低于10 ms，KG‑RAG准确率提升约20 pp；

**⚠️ 局限性**

局限性包括：可信度分数为固定源级别，未学习动态校准；优化策略只针对特定工作负载；对更大规模L4–L5层次的可扩展性未完整验证；LLM提取的可信度仍有限；并非通用图优化器，而是工作负载特定的规则驱动方案。

---

## 123. IndustrialVLA-Bench: A Traceable Multi-Axis Evaluation of Open Robot Policy Models

**arXiv ID:** 2609.25562 | [PDF](https://arxiv.org/pdf/2609.25562v1)

**作者:** Yiqi Wang `[一作]` (Griffith University), Taotao Cai `[通讯]` (University of Southern Queensland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了IndustrialVLA-Bench，一个统一评估框架，用以在相同实验条件下对公开的Vision‑Language‑Action（VLA）与World‑Action（WAM）机器人策略进行多维度比较；

**💡 创新点**

创新点在于：1）明确的可比证据门槛与状态标签（Protocol‑Faithful、Near‑Reproduction、Pending‑Verification）；2）同时评估四个诊断轴（清洁能力、视觉鲁棒性、语言敏感性、部署成本）而非单一得分；3）提供完整的复现记录与可追溯性；4）通过三种LIBERO族套件（LIBERO、LIBERO‑Plus、LIBERO‑Para）实现跨场景、跨语言的鲁棒性诊断；

**🔧 技术方法**

技术主要包括：统一的执行契约（instruction‑conditioned perception‑action pipeline）、固定检查点与推理配置的重复运行、对三种LIBERO轨道的评估脚本、以及对推理延迟、峰值VRAM、运行模式等部署指标的采集；

**📊 数据集**

使用的数据集为LIBERO家族（LIBERO、LIBERO‑Plus、LIBERO‑Para）中的标准四个任务集合（Spatial、Object、Goal、Long）以及对应的视觉与语言扰动设置；

**📈 对比分析**

比较方法：在相同检查点与推理配置下，以三次不同随机种子跑完后取平均并报告标准差；通过比较清洁成功率、鲁棒性下降幅度、语义保持率以及部署成本等指标进行多维度排名。结果显示：清洁成功率差异仅1.58个百分点，鲁棒性和语言敏感性差距分别达14.62与31.08个百分点，且无单一模型在所有维度上占优；

**⚠️ 局限性**

局限性：1）仅覆盖六个公开检查点，非完整覆盖；2）所有结果均基于仿真环境，缺乏真实世界验证；3）仅评估LIBERO族任务，对其他任务或更复杂环境缺乏泛化性；4）部署指标受硬件、精度与实现细节影响，缺乏统一基准；5）复制性状态不全，仅有三条为Protocol‑Faithful；

---

## 124. Agentic Building-Aware Satellite Gaussian Splatting for Auditable Urban DSM Reconstruction

**arXiv ID:** 2609.25578 | [PDF](https://arxiv.org/pdf/2609.25578v1)

**作者:** Wentao Sun `[一作]` (University of Waterloo), Jonathan Li `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于 Agent 的卫星 Gaussian Splatting 工作流，利用 Segment Anything 生成的建筑掩码作为语义先验，并通过 Agentic Reconstruction Controller 自动选择、验证并记录 DSM 重建策略，最终输出注册 DSM、建筑区域误差、建筑清单以及可审计的决策记录。

**💡 创新点**

创新点包括①将基础分割模型掩码作为重建过程的语义先验；②引入区域加权的光度目标和分阶段加权计划以突出建筑；③设计 Agentic Reconstruction Controller，将掩码质量、库存元数据和验证指标转化为可追溯的重建政策；④提供面向分析师的决策记录和建筑库存信息，实现可审计的城市 DSM 生产流程。

**🔧 技术方法**

使用了卫星 Gaussian Splatting（EOGS）、Segment Anything（SAM）生成建筑掩码、加权光度损失、Agent（ReAct/AutoGen 风格）决策机制以及 DSM 评估管道。

**📊 数据集**

实验基于 DFC2019 JAX 场景（JAX_004、JAX_068、JAX_214、JAX_260），这些场景包含公开的 DSM 与语义类别地图。

**📈 对比分析**

通过在不同权重（1.3、2.0、2.5）和分阶段计划下与基线进行对比，评估全场景 MAE 与建筑区域 MAE；在 JAX_004 上建筑 MAE 从 0.844 m 降至 0.806 m，整体 MAE 由 1.362 m 提升至 1.349 m；Agent 在四个场景中根据掩码覆盖率与验证结果自动选择最佳政策，进一步提升性能。

**⚠️ 局限性**

局限性包括：语义掩码质量直接影响重建效果；加权策略固定，缺乏自适应调度；仅针对建筑区域优化，未提供完整网格导出；仅在四个 JAX 场景上验证，尚未检验更大范围和多样化城市环境的泛化能力。

---

## 125. Attention as a Routing Graph: Live Circuit Extraction from a Single Forward Pass

**arXiv ID:** 2609.25285 | [PDF](https://arxiv.org/pdf/2609.25285v1)

**作者:** Ash Manvi `[一作]` (Aquin Labs), Samreena Tajreen `[通讯]` (Aquin Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种通过单次前向传播将注意力视为路由图的方法，以提取电路并评估其重要性。

**💡 创新点**

创新点在于通过一次前向传播构建路由图，并仅保留指向答案的小子图，从而以较低的成本获取有意义的电路信息。

**🔧 技术方法**

使用了GPT-2 Small、GPT-2 Medium和Pythia-410M模型，并通过注意力机制构建路由图。

**📊 数据集**

使用了GPT-2 Small、GPT-2 Medium和Pythia-410M模型的100个干净提示作为数据集。

**📈 对比分析**

通过与随机边集的对比，验证提取的子图在任务得分上的影响，结果显示提取的子图在多个模型上显著优于随机边集，且成本显著低于传统的头部修补方法。

**⚠️ 局限性**

局限性包括未能完全恢复电路，且在某些情况下（如中等模型的诱导任务）表现不佳，可能受到高分干扰或路径选择不当的影响。

---

## 126. Rewired or Gated? How Instruction Tuning Shapes Knowledge-Conflict Circuits in LLMs

**arXiv ID:** 2609.25602 | [PDF](https://arxiv.org/pdf/2609.25602v1)

**作者:** Shubham Santosh Pandere `[一作]` (IvLabs), Roshan Kumar Singh `[通讯]` (IvLabs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比基础与指令调优模型在知识冲突场景下的内部决策电路，发现指令调优通过重新加权而非重组电路实现更少接受短篇对立事实；

**💡 创新点**

首次在三大模型家族上做机制层面的基线与指令模型比较，证明冲突决策电路被门控而非重新构造，并揭示行为转向参数记忆；

**🔧 技术方法**

使用节点归因（logit-derivative saliency）、边归因（EAP-IG）、路径补丁、因果消融和超位置分析等多种技术交叉验证；

**📊 数据集**

使用 ParaConflict 数据集，过滤为单词答案的冲突事实；

**📈 对比分析**

通过节点/边重叠率、CRR（Contextual Reliance Rate）变化等指标比较，发现大多数头保持不变但权重下降，导致CRR显著下降，表明对短篇注入的抵抗力提升；

**⚠️ 局限性**

局限包括仅评估3-4B模型、单词答案限制、仅在替换式冲突上进行边分析、数据集单一、缺乏跨域或更大规模模型验证等。

---

## 127. Hierarchical Multi-Task Learning with Liquidity-Aware Signals for Stock Forecasting

**arXiv ID:** 2609.25617 | [PDF](https://arxiv.org/pdf/2609.25617v1)

**作者:** Hengyi Yang `[一作]` (Peking University), Jian Guo `[通讯]` (International Digital Economy Academy)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出LiMT框架，融合层次化跨股票/时间注意力、跨任务混合专家学习和流动性敏感的组合权重生成，实现股票收益、成交量冲击和波动率的多任务预测与可执行组合构建

**💡 创新点**

创新点包括：①交叉股票先于时间建模的层次化Transformer，清晰区分同时市况与单只股票动力；②基于MMoE的多任务学习，并通过股票级门控实现收益与流动性/风险信号的自适应跨任务转移；③轻量级Adaptive Portfolio Optimization将多任务预测转化为满足交易成本与流动性约束的可执行权重

**🔧 技术方法**

技术实现主要是跨股票注意力、时间注意力、混合专家网络（MMoE）与门控机制、Z-score归一化、轻量化权重归一化与线性融合，全部用PyTorch/Qlib实现

**📊 数据集**

使用公开Alpha158特征集合，在中国A股市场的CSI300与CSI500两大指数成分股，时间跨度2008‑2020，划分训练/验证/测试集，训练窗口21天，预测一日后收益

**📈 对比分析**

与LSTM、GRU、Transformer、SFM、TCN、TabNet、XGBoost、LightGBM、CatBoost等14个深度与树模型对比，LiMT在IC、ICIR、超额年化收益与超额信息比上均位居榜首；在实际回测中APO将年化收益从3.99%提升至10.01%，夏普比率从1.22升至1.86

**⚠️ 局限性**

局限性：仅在日频A股数据上验证；未包含更高频微观结构或跨市场信息；模型对参数和门控的敏感性仍待进一步研究；实际交易中对冲击成本与订单执行的假设可能不完全符合真实市场

---

## 128. SPARC: SuperPixel-Aware Region Contrastive Learning for Self-Supervised Dense Prediction

**arXiv ID:** 2609.25067 | [PDF](https://arxiv.org/pdf/2609.25067v1)

**作者:** David Szczecina `[一作]`, Paul Fieguth `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于超像素的区域级对比学习框架 SPARC，用于无监督预训练视觉模型，并在语义分割和目标检测任务中提升性能。

**💡 创新点**

创新点包括：① 用 SLIC 超像素划分图像并对齐不同增强视图中的区域；② 对每个超像素区域进行特征池化，形成区域表示；③ 在传统的全局对比学习分支之外加入区域级对比分支；④ 通过联合优化全局与区域对比损失，使得模型同时学习全局语义与局部空间结构。

**🔧 技术方法**

技术手段包括：超像素分割（SLIC）、双分支对比学习（全局 + 区域）、信息对比损失（InfoNCE）、EMA 更新的键编码器、ResNet-18 backbone、Adam/AdamW 优化器、AMP 加速、可视化实验与 ablation 研究。

**📊 数据集**

预训练数据集为 MS COCO (118K) 与 ImageNet100 (135K)；下游任务在 PASCAL VOC 进行语义分割（FCN）与目标检测（Faster R‑CNN+FPN）。

**📈 对比分析**

与 MoCo‑v2 与 DenseCL 进行对比。SPARC 在同等预训练设置下，在 PASCAL VOC 上的 mIoU 最高提升约 9.8%（相较 MoCo‑v2）和 5.3%（相较 DenseCL），AP 最高提升约 5.0%（相较 MoCo‑v2）和 4.8%（相较 DenseCL）。

**⚠️ 局限性**

局限性包括：① 依赖固定的 SLIC 超像素，未学习到更适合任务的区域划分；② 仅在 ResNet‑18 级网络上验证，缺乏对更大模型和 Transformer 框架的推广；③ 只评估了 VOC 数据集，尚未验证在更大规模或不同任务上的泛化能力。

---

## 129. FoMo: Forking Moment in Generative Trajectory as a Perceptual Distance

**arXiv ID:** 2609.25716 | [PDF](https://arxiv.org/pdf/2609.25716v1)

**作者:** Jaihyun Lew `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种全自动化的数据生成管线，利用扩散模型的分叉时刻（FoMo）来为图像质量评估（IQA）生成点式感知距离标签，进而训练参考基准的IQA指标；

**💡 创新点**

创新点在于：①用扩散过程中的分叉时刻作为无人工标注的全局感知距离度量；②采用RankNet式的全局排序损失，而非传统的二元对比或MOS回归；

**🔧 技术方法**

技术手段包括扩散模型（如FLUX、Stable Diffusion系列）进行分叉采样、生成对比图像；使用RankNet二元交叉熵损失进行全局排序训练；

**📊 数据集**

使用的主要数据集为ImageNet（作为参考图像）以及FLUX合成的图像对；在消融实验中也使用KADID‑10k作为对照；

**📈 对比分析**

在PIPAL、TID2013、CSIQ、LIVE四大基准上，与KADID‑10k、BAPPS、PieAPP等传统或2AFC标注方法相比，FoMo在CNN和Transformer骨干上均实现了更高的Spearman Rank Order Correlation Coefficient（SROCC），尤其在Transformer模型上提升显著；

**⚠️ 局限性**

局限性包括：依赖扩散模型的质量与多样性；对极端或稀有失真类型可能不如专门标注的数据集表现；训练需要较大batch size和显著计算资源；目前仅针对图像质量评估，尚未验证到其他视觉任务。

---

## 130. CPyGraph: A Version-Aware Static Analysis Framework for Native CPython Bytecode

**arXiv ID:** 2609.25083 | [PDF](https://arxiv.org/pdf/2609.25083v1)

**作者:** Baihong Chen `[一作]` (Utah State University), Wen Li `[通讯]` (Utah State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个C++框架CPyGraph，用于对Python包级别的CPython原生字节码进行静态分析，构建CFG、CDG、PTA、CG和DDG等图形结构；

**💡 创新点**

创新点包括：1) 版本感知的字节码适配器统一接口，隔离不同CPython版本的opcode、堆栈、调用、异常等差异；2) operand‑stack-aware Andersen点对点分析与调用图共同收敛的固定点求解；3) 共享的分析状态让异常Aware CFG、CDG、DDG在同一分析上下文中生成；4) 覆盖摘要记录未解析的动态行为，方便后续投影；

**🔧 技术方法**

使用的技术有：C++实现的字节码适配器架构、operand‑stack-aware Andersen点对点分析、call‑graph与PTA的协同固定点、异常Aware CFG、CDG、DDG构造、可配置的流/上下文/路径敏感性、以及基于数值ID的图表示；

**📊 数据集**

实验使用的数据集包括：201个针对不同语言特性的微基准包；1,733个候选事实（1,020期望+713负例）用来衡量精度；PyPI 2,500个真实包用于性能评估；以及PyCG的112程序基准用于外部对比；所有基准均覆盖CPython 3.10–3.14版本；

**📈 对比分析**

比较方法：在CPython 3.10上与PyCG基准对照，获得111/103完整/可靠案例；默认配置达到91.40%候选精度、100%召回；完整敏感性提升到94.97%精度；跨3.10–3.14版本的一致性为99.55%；对2,500包的实验显示平均耗时0.95秒、内存峰值28.4 MiB，成功率98.48%；性能与基准相比相当或更优；

**⚠️ 局限性**

限制：仅分析包内部代码，动态代码生成、反射、外部模块或原生扩展不被解析；异步调度与事件循环模型未实现；仅支持CPython 3.10–3.14，需要为新版本手动维护适配器；对极大或复杂包的可扩展性仍待验证。

---

## 131. mbariml: a curation pipeline for turning deep-sea imagery and video into object-detection training data

**arXiv ID:** 2609.25500 | [PDF](https://arxiv.org/pdf/2609.25500v1)

**作者:** Lonny Lundsten `[一作]` (Monterey Bay Aquarium Research Institute), Dave Caress `[通讯]` (Monterey Bay Aquarium Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个完整的视频/图像分析流水线mbariml，用于深海环境中稀疏、暗淡生物的自动检测、可视化分组和人工审核，最终生成可用于训练的标注数据。

**💡 创新点**

创新点包括：① 用单一DuckDB表统一存储检测结果、嵌入向量和人工反馈，实现增量式流水线；② 视频轨迹压缩至中间第三高置信度代表帧，减少冗余；③ 通过DINOv3嵌入与EVoC聚类，批量批注并提供相似性搜索；④ GUI支持批量验证、重标、框调整和新增标注，提升人工效率；⑤ 支持多种导出格式（VOC、YOLO、HTML、MBARI侧车）并保证同一数据库下的一致性。

**🔧 技术方法**

技术手段包括：Ultralytics YOLO（检测与多目标跟踪）、DINOv3（视觉嵌入）、EVoC（聚类）、DuckDB（嵌入式数据库）、Python/CLI + GUI（vars-gridview）以及自定义的图像/视频处理脚本。

**📊 数据集**

使用的主要数据集为海底遥感影像（来自自主、远程操作、登陆平台和载人车辆的图像/视频），文中未给出具体公开数据集名称，而是针对实际深海调查部署的数据。

**📈 对比分析**

论文未进行系统化的对比实验或性能评估；主要通过实验性观察和内部测评验证流程可行性。由于缺少基准数据和精确评估，无法给出定量的准确率或效率指标。

**⚠️ 局限性**

限制包括：① 中间第三帧代表帧的选择仅基于经验，缺乏定量验证；② 聚类参数需要针对不同部署手动调优；③ 轨迹ID可能出现断裂或身份切换，无法保证每条轨迹对应唯一个体；④ 单文件锁定导致无法多人并发编辑；⑤ 目前未对长时间、全深潜数据进行规模测试。

---

## 132. Attack Success Rate Is Not a Number: On Measurement Validity in Agentic AI Security Evaluation

**arXiv ID:** 2609.25173 | [PDF](https://arxiv.org/pdf/2609.25173v1)

**作者:** Chetan Pathade `[一作]` (Independent Researcher), Shubham Patil `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估攻击成功率（ASR）在LLM代理安全评估中的有效性与可比性，提出六个关键测量自由度，进行元分析和理论分析，最终给出十条报告检查清单；

**💡 创新点**

1）明确ASR并非单一指标，而是由六个参数决定；2）系统性检索并量化大规模论文的报告缺失；3）用统计与理论手段证明不同测量会导致排名逆转；4）提出统一的报告规范。

**🔧 技术方法**

文本检索与正则表达式自动编码、统计置信区间与显著性检验、理论计算（最小可检测差异）以及基于100实例benchmark的模拟实验。

**📊 数据集**

主要使用2025‑2026年间arXiv收录的259篇代理安全论文作为样本，另外使用公开的100实例benchmark用于理论推导。

**📈 对比分析**

通过对比不同测量设置的ASR值，发现单次评估可能导致约21%误判排名，最小可检测差异约为18.2个百分点；表明目前跨论文比较缺乏可信度。

**⚠️ 局限性**

仅聚焦已发表论文，未涵盖非arXiv来源；未考虑模型版本漂移、环境工具版本及上下文窗口等因素；正则表达式检测仍可能产生误报/漏报。

---

## 133. The Cartesian Hand: In-Hand Manipulation with All-Linear Fingers

**arXiv ID:** 2609.25696 | [PDF](https://arxiv.org/pdf/2609.25696v1)

**作者:** Boxi Xia `[一作]` (Duke University), Boyuan Chen `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种全线性7-自由度手爪Cartesian Hand，实现对35种实验室、工业与日常物体的多种在手操作；

**💡 创新点**

创新点在于将双独立平行抓取与相对运动集成于单手，通过七个全Prismatic关节实现配置无关的指尖运动；

**🔧 技术方法**

采用平行抓取、指尖滑动、手抓距移动等线性动作原语，结合基于关节反馈的触碰检测；

**📊 数据集**

使用35个不同类别（螺纹、泵、双柄工具、螺丝刀、触发器等）手工整理的实验室与日常物体进行评估；

**📈 对比分析**

与传统多指抓取手相比，未使用视觉或触觉反馈，但在所有350次试验中实现100%成功率；

**⚠️ 局限性**

局限性包括仅能操作已知姿态、结构化机械部件的物体，对非结构化或需视觉定位的任务表现有限。

---

## 134. Brain-Inspired Hierarchical Modularity for General Continual Learning

**arXiv ID:** 2609.25146 | [PDF](https://arxiv.org/pdf/2609.25146v1)

**作者:** Hongwei Yan `[一作]` (Tsinghua University), Liyuan Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于果蝇嗅觉学习记忆系统层级模块化的通用持续学习框架FlyGCL，能够在在线、模糊、不确定且不断演化的数据流下实现专家专化与集成协同；

**💡 创新点**

通过将稀疏随机扩展、Mixture‑of‑Experts路由与多时尺度Ensemble Learning相结合，首次在预训练基础模型上实现既分离冲突经验又融合兼容经验的层级模块化；

**🔧 技术方法**

利用预训练ViT/CLIP等大模型、随机扩展+闭式Ridge路由、轻量化LoRA/Adapter/Prompt模块、多时尺度EMA头及类级校准等技术，支持多模态在线持续学习；

**📊 数据集**

在视觉识别（CIFAR‑100、ImageNet‑R、CUB‑200）、视觉‑语言（CLIP基准）、视频理解（EgoExoLearn、EgoExo‑Fitness）和机器人交互（LIBERO‑Spatial、‑Object、‑Goal、‑Long）等数据集上进行实验；

**📈 对比分析**

与SeqFT、EWC、LwF、L2P、DualPrompt、MVP、MISA、CLAP4CLIP、MG‑CLIP、ER、DER++、PackNet等方法对照，FlyGCL在A_last与A_auc上均显著优于最强无重放基线，尤其在LIBERO任务中提升49–58个百分点；

**⚠️ 局限性**

局限于轻量化模块化而非深层调优，对长期主动探索与自适应资源分配关注不足，生物模型简化未覆盖神经调制与行为反馈，且在从零训练的情境下的效果尚未验证。

---

## 135. Training a Language Model End-to-End in Rust: An Experience Report

**arXiv ID:** 2609.25008 | [PDF](https://arxiv.org/pdf/2609.25008v1)

**作者:** Arif Adito `[一作]` `[通讯]` (Adioris Tech Ltd.), Arif Adito (Adioris Tech Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在纯 Rust 环境下实现并预训练了约 0.4 B 参数的 Bangla‑first 语言模型，记录了训练过程中的隐藏缺陷并给出解决方案，展示了仅租用 GPU 164 美元的成本；

**💡 创新点**

首次系统分类了 Candle 与 Burn 两大 Rust ML 框架在训练中的隐蔽缺陷，并提出梯度流仲裁器等验证手段，证明纯 Rust 训练可行且成本可控；

**🔧 技术方法**

使用 Rust 语言、Candle 与 Burn 框架、手工实现的梯度流仲裁器、混合精度/全精度训练、滑窗注意力、旋转位置嵌入、线性注意力等技术；

**📊 数据集**

采用混合双语（Bangla 与 English）语料，主要来源于 Bangla web text、Bangla journalism、English educational web 等，约 20 亿 tokens，权重偏向 Bangla；

**📈 对比分析**

通过与随机初始化模型对比、计算 Bangla/English NLL、英语常识测试、梯度仲裁器验证等方法，发现缺陷后单 GPU 吞吐提升约 5.5 倍，成本降至每十亿 token 82 美元；

**⚠️ 局限性**

训练规模仅 0.4 B，缺少 instruction tuning、外部评测基准、数据多样性，Tokenizer 公平问题导致 Bangla 语料偏低，英文能力几乎随机，模型权重未公开。

---

## 136. ShowTellArena: Evaluating Business Workflow Understanding from Demonstrations

**arXiv ID:** 2609.25467 | [PDF](https://arxiv.org/pdf/2609.25467v1)

**作者:** David Garg `[一作]` (Brackett Labs), Siddhartha Borah `[通讯]` (Brackett Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一套基于叙述式业务演示的教学与理解评测协议，并发布了公开的任务集和评测工具。

**💡 创新点**

创新点在于：①将问题与演示证据直接关联，支持对业务规则、例外和错误的深度检验；②提供跨平台的原生教学接口适配器，使不同系统在同一学习情境下可被评估；③构建了可检视、可扩展的工作流任务数据集，首次实现了业务流程演示与问答的可复现评测。

**🔧 技术方法**

技术方法包括：录制演示视频与截图、叙述文本、基于规则和语义的答案判定器、评分模型辅助评审、以及针对不同产品的适配器（Brackett Show & Tell、Claude Teaching Flow、Codex Record & Replay）来捕获学习内容并生成答题。

**📊 数据集**

使用了 ShowAndTell Arena v1.0 数据集，包含 857 条问题-证据链接，涵盖财务、招聘、采购、客户决策、库存与物流等业务场景，并集成 ERPNext、Fleetbase、ONLYOFFICE 等应用。

**📈 对比分析**

评测采用先让系统完成演示后给定固定测验，聚合所有共享案例的分数；Brackett、Claude、Codex 在 pilot 试验中得到不同的平均得分，表明系统对演示内容的理解与答题质量存在差异，但因覆盖不完整、评估不一致及缺乏对照组，无法给出客观性能排名。

**⚠️ 局限性**

局限性包括：①数据集并非业务全景，缺少独立答案可检验与仅用问题的控制实验；② pilot 结果受捕获失败、处理延迟、评分偏差等影响，且未能统一记录所有重试与人工干预；③未评估执行成功、长期记忆或迁移学习等后续能力；④评测与模型本身、适配器版本等细节不透明，影响复现与对比。

---

## 137. Spend Classification Without Leakage: An Evaluation Harness and What It Changed in a Deployed System

**arXiv ID:** 2609.25502 | [PDF](https://arxiv.org/pdf/2609.25502v1)

**作者:** Harshit Gupta `[一作]` `[通讯]` (SpendSide), Harshit Gupta (SpendSide)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个公开的支出分类基准，包含四种评估协议，并在两份1.26M行的政府采购数据上评估了多种模型。

**💡 创新点**

发现标准随机拆分掩盖了记忆效应，提出漏泄量化、注释冲突上限和金额加权上限等指标，并验证了这些指标对模型性能评估的影响。

**🔧 技术方法**

采用词级TF‑IDF、句子嵌入最近邻、质心分类以及微调的DistilBERT，结合自定义的分割协议和准确率/宏F1/金额加权准确率计算。

**📊 数据集**

使用了加州州政府采购订单（ca-dgs）和华盛顿州亚马逊市场支出（wa-amazon）两份公开采购记录，总计1.26M条。

**📈 对比分析**

通过四种协议（iid‑naive、iid‑grouped、temporal、supplier）对模型进行公平比较，发现泄漏拆分会高估模型，真正可达的准确率在约32%（新文本）至约79%（可达上限）之间，Transformer与最近邻性能相当。

**⚠️ 局限性**

数据仅来自美国公共部门且为买家手工标注，缺乏独立验证；模型规模受限，仅评估DistilBERT，未覆盖更大语言模型；且仅对公开文本评估，无法直接推广至私有或非美国采购场景。

---

## 138. A JEPA Recipe for Tabular Foundation Models

**arXiv ID:** 2609.25541 | [PDF](https://arxiv.org/pdf/2609.25541v1)

**作者:** Mingyu Jeon `[一作]` (Modulabs), Jae Young Suh `[通讯]` (Modulabs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在表格基础模型先验上，使用JEPA式的潜在预测目标与传统值预测目标并行训练，展示了在不发生崩溃的情况下实现潜在预测的可行性。

**💡 创新点**

创新点在于提出了三项关键改进：将值预测头改为读取编码器场而非预测器；使用EMA差分作为潜在目标；以及使用掩码标记和开放时段停止规则来防止潜在目标崩溃。

**🔧 技术方法**

技术手段包括单层细胞级Transformer编码器、EMA目标编码器、预测器Transformer、掩码标记机制以及基于验证误差的“到达平台”停止策略。

**📊 数据集**

实验使用了TabICL生成的合成SCM先验表格以及来自OpenML‑CC18、Grinsztajn和TabArena的真实分类与回归数据集进行评估。

**📈 对比分析**

比较方法采用单一随机种子下的配对实验，对值仅臂(ds)和带潜在臂(jepa)进行相同数据流与预算的训练，并在真实数据集上对比准确率与R²；结果显示jepa在收敛时略逊于ds，虽然潜在目标未崩溃，但未提升整体性能。

**⚠️ 局限性**

主要局限包括仅使用单个种子导致结果不确定性、缺乏将潜在权重设为零的匹配对照、模型规模与上下文行数可能限制性能、以及未评估潜在目标在鲁棒性或短上下文情境下的潜在收益。

---

## 139. SSP-Bench: A Hybrid Data Generation Framework for Safety, Security, and Privacy Evaluation

**arXiv ID:** 2609.25352 | [PDF](https://arxiv.org/pdf/2609.25352v1)

**作者:** Fatih Deniz `[一作]` (Hamad Bin Khalifa University), Issa Khalil `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种动态生成安全、隐私与对齐（SSP）评测框架，能够在保持领域一致性的前提下，按需生成新的评测实例，并通过外部可验证来源保证标签有效性。

**💡 创新点**

创新点包括：①使用外部可验证数据源（如 Wikipedia、AI 事件数据库、SPY 法律/医疗文本）为每个题目提供独立标签；②引入多模型 steering panel 对生成的候选题目进行难度与分辨率的实时反馈；③以难度、可分辨性、创新性与多样性为多目标优化目标，选取最具区分力的题目；④引入排名稳定性度量（Stab）和综合质量 Q，揭示静态基准的构造缺陷。

**🔧 技术方法**

技术上结合了心理测量中的项目反应理论（IRT）与自动测试组装（ATA），利用多模型推理、LLM 判定器与贪心/遗传式多目标优化实现动态生成与评测；同时采用了自然语言处理的检索与生成技术、文本相似度判定以及多模型校准。

**📊 数据集**

使用的数据集包括公开的 Wikipedia、AI 事件（AIAAIC）数据集、SPY 法律/医疗文档、以及已有的静态 SSP 评测基准（HELM Safety、aiXamine 等）作为对照；四个 SSP 服务的评测题目来自于这些来源。

**📈 对比分析**

通过与现有静态基准的 Kendall τ、Spearman ρ、分辨率（Disc_ε）和稳定性（Stab）进行对比，评估动态基准在安全性、幻觉、过度拒绝和隐私四项服务上的表现。结果显示：安全服务的动态排名与静态排名相关性从近 0 变为 0.67；整体 Q 值 > 0.80，表明动态基准在分辨率与稳定性上优于传统静态基准。

**⚠️ 局限性**

局限性包括：评测结果仍受 LLM 判定器或规则判定器的影响；只覆盖四个 SSP 服务，未扩展至代码安全等其他维度；对 steering panel 组合的鲁棒性测试有限；缺乏完整的 IRT 校准与多维度模型，未来工作需进一步完善。

---

## 140. Entropy Can Flow, or It Can Guide. Be Entropy. LEDFlow: Introducing Entropy-guided Generation Order into Uniform Discrete Flow

**arXiv ID:** 2609.25131 | [PDF](https://arxiv.org/pdf/2609.25131v1)

**作者:** Tung Sum Thomas Kwok `[一作]` (University of California, Los Angeles), Oscar Leong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在统一离散流（uniform discrete flow）框架中引入生成顺序，通过选择性吸收（selective absorption）将可靠预测固定下来，同时允许其他位置继续演化；并提出一种基于局部熵的无训练吸收策略（Low‑Entropy Discrete Flow）。

**💡 创新点**

创新点包括：①提出训练无关的低熵吸收策略，直接利用去噪器的不确定性来决定哪些位置先固定；②对吸收导致的错误进行KL分解，证明在熵-误差正则化下低熵选择能最小化条件误差上界；③对全局lookahead决策误差进行理论分析，说明其随窗口大小增长而放大，从而支持在不完美去噪器下优先使用局部熵。

**🔧 技术方法**

使用的技术主要有：统一离散流（FUDOKI）及其动力学（kinetic‑optimal velocity）、离散时间马尔科夫链（CTMC）采样、选择性吸收机制、局部熵计算、KL误差分解、实验对照的多种采样器（Euler、τ‑leaping、纠正、引导、Self‑Correction、Info‑Gain 等）。

**📊 数据集**

数据集包括：数独（Nikoli、Latin‑square、Graph Coloring、Molecular Infilling）、文本到图像（GenEval）、多模态理解（VLMEvalKit六个基准）、数学推理（MathVista、MathVerse、GSM8K）。

**📈 对比分析**

对照方法包括传统无吸收采样器（Euler、τ‑leaping、纠正、引导、Self‑Correction）以及其他吸收策略（概率边缘、全局lookahead）。实验表明：在数独任务上准确率从 0.796 提升到 0.845；在文本到图像上取得最高总体分 0.7814（仅比最先进的解码时引导低 0.001）；在多模态理解上略有提升；整体上在结构约束强的任务中表现最显著。

**⚠️ 局限性**

局限性：①对结构松散的任务提升有限，主要受去噪器不确定性限制；②全局lookahead 在精确度高的去噪器下可能更优，但计算成本高；③局部熵策略在依赖性强的任务中仍可能忽略全局约束；④需要合理设定吸收预算和时间步长，超出范围可能导致性能下降。

---

## 141. RootQuantV2: Adapting a Vision Foundation Model for Root-Trait Regression from Minirhizotron Imagery

**arXiv ID:** 2609.25567 | [PDF](https://arxiv.org/pdf/2609.25567v1)

**作者:** Kinjalk Parth `[一作]` (University of Illinois at Urbana-Champaign), Andrew D. B. Leakey `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用冻结的自监督ViT-L/16与DoRA+Mona轻量化适配器，对根系minirhizotron图像进行全图根长度与表面积直接回归，省去像素级分割；

**💡 创新点**

创新点在于将自监督视觉基础模型与参数高效适配器相结合，同时引入基于根系特性的可聚合密度读出和多尺度卷积，显著提升极大数量目标回归精度；

**🔧 技术方法**

技术包含DINOv3 ViT-L/16、DoRA（注意力层低秩权重分解）、Mona（多尺度卷积适配器）、广义均值池化、注意力池化、密度读出、D4对称数据增强与EMA权重平均；

**📊 数据集**

使用RootQuant数据集，包含89,186训练、11,445验证、17,560测试帧，涵盖玉米与大豆两种作物的minirhizotron图像；

**📈 对比分析**

与原始RootQuant CNN基线对比，RootQuantV2在全集复合R²提升至0.940（+4.4%），根长度RMSE下降24.3%、面积RMSE下降20.7%；在可见根子集复合R²提升至0.914（+6.5%）；

**⚠️ 局限性**

局限性包括在空白帧上仍有少量误报，零膨胀数据对评估产生影响，且模型在跨作物迁移时需进一步适配，缺乏对其他作物或多种土壤条件的验证。

---

## 142. Narrowband Voice Communication Using Streaming Neural Compression

**arXiv ID:** 2609.25379 | [PDF](https://arxiv.org/pdf/2609.25379v1)

**作者:** Dahong Luo `[一作]` (University of Maryland), Nirupam Roy `[通讯]` (University of Maryland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一种轻量级流式神经音频编解码器，能在低比特率2.3 kbps下实现可理解语音传输，适用于资源受限的边缘设备。

**💡 创新点**

创新点包括：①采用Residual Finite Scalar Quantization（RFSQ）替代传统RVQ，消除量化搜索开销；②提出伪lookahead解码与潜在缓存方案，解决iSTFT重构的未来帧依赖，实现真正流式；③使用分阶段渐进训练和MFCC感知损失提升音质。

**🔧 技术方法**

使用技术包括SEANet基编码器、Vocos式解码器、可微逆STFT、RFSQ量化、伪lookahead+缓存、三阶段渐进训练、HiFiGAN Period Discriminator、MFCC感知损失、语义差异评估以及ExecuTorch导出部署。

**📊 数据集**

训练数据集为Mozilla Common Voice（英语子集），评测使用LibriSpeech和EmoVoice‑DB。

**📈 对比分析**

通过与EnCodec、DAC、HILCodec等基准的离线PESQ/STOI比较，模型在16 kHz下得到PESQ≈1.32、STOI≈0.81；在Raspberry Pi 3上实现RTF<1.0；RFSQ量化比RVQ快得多，显著降低推理时间；消融实验表明伪lookahead、MFCC损失、渐进训练及RFSQ对性能提升作用显著。

**⚠️ 局限性**

局限性在于引入了1帧（约20 ms）算法延迟；仍存在边界失真；目前未实现可变比特率功能，需进一步改进。

---

## 143. Spatiotemporal Kronecker Covariance Neural Networks

**arXiv ID:** 2609.25326 | [PDF](https://arxiv.org/pdf/2609.25326v1)

**作者:** Andrea Cavallo `[一作]` (Delft University of Technology), Elvin Isufi `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了Kronecker协方差神经网络（KVNN），一种利用Kronecker分解表示的时空协方差并通过图卷积网络进行任务驱动学习的多变量时序预测模型。

**💡 创新点**

创新点在于：① 将时空协方差拆分为多组Kronecker乘积，使得不同时间延迟可以被独立处理；② 通过对滤波系数的群稀疏正则实现自动剔除无效时滞；③ 在平稳假设下给出了KVNN的谱分析与有限样本误差稳定性证明。

**🔧 技术方法**

技术方法包括：Kronecker协方差分解、时间图卷积网络（VNN原理）、多阶多项式滤波器、群稀疏正则、低秩Kronecker估计、谱分析和子高斯理论。

**📊 数据集**

使用了五个真实多变量时序数据集：LargeCap（50支美国股票），Sectors（9只ETF），NN5（日常ATM提现），CzeLan（森林生态通道），AQWan（北京空气质量）以及股价方差预测。

**📈 对比分析**

与LSTM、ST-PCA、VNN、STVNN、LVNN等基线模型进行对比。KVNN在15个预测任务（不同数据集与预测步长）中占前两名，平均MAE下降5–10%，并且在参数量上往往低于基线；此外在有限样本情况下表现出更好的稳定性。

**⚠️ 局限性**

局限性包括：① 需要足够多的样本来估计高维协方差，尤其是低秩估计时；② 低秩KVNN的谱分析与稳定性理论不再适用；③ 对非平稳或高记忆时序的适用性尚未充分验证；④ 主要处理线性协方差信息，仍需进一步提升对高度非线性时序关系的建模能力。

---

## 144. C2FXNet: Coarse-to-Fine Scene Expert for Unified Object Detection across Adverse Weather

**arXiv ID:** 2609.25693 | [PDF](https://arxiv.org/pdf/2609.25693v1)

**作者:** Tianle Fang `[一作]` (Guilin University of Electronic Technology), Haoxiang Lu `[通讯]` (Guilin University of Electronic Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了C2FXNet，结合粗到细的场景指导，实现了统一的恶劣天气下物体检测。

**💡 创新点**

创新点在于层次化文本引导的多步推理路由器、细粒度场景细化模块和场景感知混合专家网络的协同机制。

**🔧 技术方法**

采用视觉‑语言模型、GRU递归推理、Group Normalization 缩放调制、动态专家路由和频域/低光增强算子等技术。

**📊 数据集**

使用RTTS、ExDark、自己构造的AWD三大恶劣天气数据集，并在VOC‑Rain、VOC‑Snow上进行跨天气泛化验证。

**📈 对比分析**

与YOLOv9s‑v13s、EEnvA‑Mamba等主流方法对比，RTTS 63.70% mAP、ExDark 71.14% mAP、AWD 54.19% mAP，参数更少，性能显著提升。

**⚠️ 局限性**

局限在于完全监督训练，仅覆盖有限类别与天气类型，未能实现对未知天气与新目标类别的开放式适应。

---

## 145. Sex Estimation from Footwear Outsole Impressions Using CNN Transfer Learning and Interpretable Image Statistics

**arXiv ID:** 2609.25386 | [PDF](https://arxiv.org/pdf/2609.25386v1)

**作者:** Jinyi Niu `[一作]` (Fudan University), Weining Shen `[通讯]` (University of California, Irvine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用卷积神经网络（CNN）迁移学习与传统手工特征分类，探究从鞋底印图像进行二元性别估计的可行性。

**💡 创新点**

提出在鞋底印数据上实施鞋级训练/测试划分以避免数据泄漏，并将CNN学习特征与可解释的图像统计量相结合进行探索性关联分析。

**🔧 技术方法**

使用四种ImageNet预训练CNN（VGG16、ResNet‑50、EfficientNet‑B0、MobileNet‑V2）及其三种迁移学习策略（全精调、冻结特征+SVM、混合特征+SVM），同时对比传统SVM、随机森林、XGBoost模型。

**📊 数据集**

采用公开的1500幅鞋底印图像（150双鞋，5次扫描/鞋），包含性别、品牌、尺寸等元数据。

**📈 对比分析**

结果显示，精调CNN模型（尤其EfficientNet‑B0和MobileNet‑V2）可达100%准确率；冻结特征+SVM表现亦显著优于传统基线；混合特征在大多数网络中略有提升。整体性能远高于仅使用手工特征的传统方法。

**⚠️ 局限性**

局限包括：仅在单一受控数据集上评估，缺乏对真实案例图像的验证；使用单一80/20划分未估计跨划分方差；未进行验证集或早停；未提供置信度或不确定性估计。

---

## 146. Spectra: A Rules-Driven LLM Pipeline for Automated KYC Document Processing

**arXiv ID:** 2609.25474 | [PDF](https://arxiv.org/pdf/2609.25474v1)

**作者:** Miray Wahib `[一作]`, Nikita Dvornik `[通讯]` (Royal Bank Of Canada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Spectra AI辅助KYC文档处理平台，将传统四方流程简化为两方并引入多代理流水线完成文档分类、字段提取与合规校验。

**💡 创新点**

创新点在于使用结构化规则引擎将合规政策编码为可查询数据库，拆分文档处理为可审计的独立阶段，并通过精确上下文注入提升LLM推理精度与可追溯性。

**🔧 技术方法**

技术方案包括PostgreSQL规则引擎、LlamaCloud和OpenAI（o3-mini、GPT‑4.1‑mini）LLM多代理架构、PDF文本提取（PyMuPDF/Tesseract）、注入检测与安全审计。

**📊 数据集**

评估使用21份真实KYC文件（共9类）做分类实验，15份合成文档共141字段做提取实验。

**📈 对比分析**

通过对比传统单体LLM流水线，Spectra在分类上实现100%准确率，提取上达到89.4%准确率，平均每文档约11秒，人工复核负担下降96%。

**⚠️ 局限性**

局限性包括样本规模有限、主要为清晰PDF（未充分覆盖扫描/OCR噪声）、对复杂字段和特定文档类型（如公司决议）准确率偏低，需改进schema约束和多语言/扫描支持。

---

## 147. C-to-Rust Fallacy: Automatic Refactoring != Memory Security

**arXiv ID:** 2609.25682 | [PDF](https://arxiv.org/pdf/2609.25682v1)

**作者:** Hung-Mao Chen `[一作]` (George Mason University), Kun Sun `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了四种自动 C‑to‑Rust 重构工具在内存安全方面的可靠性、正确性以及漏洞迁移情况。

**💡 创新点**

首次系统性揭示了减少 unsafe 并不等价于提升安全，深入分析了工具在安全迁移、误报与新增漏洞方面的表现。

**🔧 技术方法**

结合静态分析、LLM 辅助编译错误修复、AddressSanitizer、Clippy、Miri 以及 LLM 监测未定义行为等多种技术。

**📊 数据集**

使用 NIST Juliet Test Suite 的 116 个包含 NPD、UAF、DF、BOF、TC 等内存安全漏洞的 C 程序。

**📈 对比分析**

对生成的 464 个 Rust 程序进行编译、漏洞检测与错误迁移计数，结果显示 342 程序编译失败、177 仍保留原漏洞、77 新增 Rust 漏洞，LLM 方案虽提升编译率但仍存在安全缺口。

**⚠️ 局限性**

评估仅覆盖已知漏洞类型，未检验语义等价性，工具前端覆盖率有限，LLM 辅助可能漏报，结果仅为下限。

---

## 148. Cosserat Modeling of Trimmed Helicoid Soft Arms with a Separated-Section Constitutive Law

**arXiv ID:** 2609.25264 | [PDF](https://arxiv.org/pdf/2609.25264v1)

**作者:** Zhihang Qin `[一作]` (National University of Singapore), Cecilia Laschi `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对分离且斜向加载的螺旋结构软臂，提出了将每条螺旋域的本地本构响应投射回主干的分离段本构模型，并将稀疏融合点的相对运动相互作用纳入稀疏融合力学，从而得到有效的主干截面刚度；随后将此截面法则嵌入动态Geometric Variable-Strain（GVS）Cosserat杆模型，实现路由肌腱驱动的软臂动力学仿真。

**💡 创新点**

创新点在于：①首次在Cosserat杆框架中将分离段本构与稀疏融合力学结合，解决传统“同一截面”假设对螺旋结构失效的问题；②通过单次融合校准实现对弯曲、扭转、轴向刚度的强异向性预测；③在三维软臂实验中验证模型精度，展示高达0.3秒/状态的实时计算能力。

**🔧 技术方法**

技术主要包括：Cosserat杆理论、分离段本构推导、稀疏融合力学（相对运动与融合弹性能量），GVS动力学框架、路由肌腱动力学、几何饱和约束、Gauss–Legendre积分与Zanna–Magnus数值传播、增广拉格朗日求解。

**📊 数据集**

使用从三段斜向裁剪螺旋模块（TPU 95A 3D打印）构建的软臂实验数据，共103个配置，分为深弯曲（Dataset 0）、组合变形（Dataset A）和轴向收缩（Dataset B），共计309个平台姿态样本。

**📈 对比分析**

与传统“求和截面”基线模型对比：在所有三种变形模式下，有效本构法则将位置误差降至约1–8%，而基线误差在60–91%之间；旋转误差同样下降至约2–8%；总体归一化位置误差小于8%；仿真耗时约0.3 s/状态，保持在稳定范围内。

**⚠️ 局限性**

局限性包括：①模型仍假设局部材料小应变，未考虑大变形下的材料非线性；②稀疏融合校准仅通过一次自由悬垂实验完成，可能对不同负载或材料变形产生误差；③模型未对接触或黏性摩擦进行完整建模，只通过几何饱和约束近似锁定；④对更复杂的多模态或多平台结构的推广仍待验证。

---

## 149. Code Equivalence and Automorphism Problems for Codes

**arXiv ID:** 2609.25483 | [PDF](https://arxiv.org/pdf/2609.25483v1)

**作者:** Jean-Francois Biasse `[一作]`, Anuvrat Jaindungarwal `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了码等价（Permutation/Linear Code Equivalence）与码的自同构问题（计数、划分、生成集）在多项式时间内彼此可约，给出了完整的确定性多项式时间归约与一次性归约。

**💡 创新点**

创新点在于：①提出了一种基于码的直接和分解、支持图和基于冗余类的集合稳定链的统一框架，能够同时处理排列和单射等价；②给出了从等价问题到自同构计数、划分、生成的单次归约；③通过直接和分解将任何线性码分解为不可分码，显著降低归约复杂度；④将图同构与码同构的相互可约性做了系统化比较。

**🔧 技术方法**

技术主要包括：线性码的直接和分解算法（利用支持图的连通分量）；基于冗余列集合的集合稳定链与强生成集；递归利用群的 Orbit‑Stabilizer 定理；Hopcroft–Karp 匹配、Dijkstra 路径合成；以及多次使用等价/自同构 oracle 的方法。

**📊 数据集**

本文为理论工作，不使用具体实验数据集，而是在任意给定长度 n、码字空间维度 k 的线性码上给出算法与复杂度分析。

**📈 对比分析**

与现有研究相比，所有归约均为确定性多项式时间，时间复杂度最高为 O(n³ log q)（或 O(n⁶ log q) 用于计数问题），最多需要 O(n²) 次 oracle 调用；相较于之前仅有的多项式时间归约，本文实现了一次性归约并提供了完整的实现细节和时间分析。

**⚠️ 局限性**

限制包括：仅适用于线性码（不涉及非线性码）；归约依赖于等价/自同构 oracle 的存在，若 oracle 本身难以实现则整体效率受限；对于极大码长度，O(n⁶ log q) 的计数归约可能在实践中不可行。

---

## 150. Uncertainty-Aware 3D Residual Wavelet Diffusion for Ultra Low-Field MRI Super-Resolution

**arXiv ID:** 2609.25319 | [PDF](https://arxiv.org/pdf/2609.25319v1)

**作者:** Rui W. Yeow `[一作]` (University College London), James H. Cole `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

在低场（0.064 T）MRI扫描的全脑图像中实现超分辨率，通过构建3D残差小波扩散模型实现对高场（3 T）图像后验分布的采样，并提供像素级不确定性映射。

**💡 创新点**

将无损Haar小波重参数化、残差衰减式扩散过程和域随机化训练三者整合，突破了3D扩散在内存、采样速度和域迁移上的瓶颈；同时通过后验采样提供多样化的可解释不确定性。

**🔧 技术方法**

使用Haar小波变换对体素进行无损压缩；3D Swin Transformer U‑Net作为逆向扩散网络；残差扩散策略从低场输入开始迭代；域随机化模拟不同扫描仪的对比度、分辨率和噪声；损失函数包含图像L1、波形子带L1、LPIPS感知相似度和分割Dice/交叉熵；采用K次采样得到均值和标准差。

**📊 数据集**

训练数据来自Human Connectome Project（682 T1）和Alzheimer’s Disease Neuroimaging Initiative（369 T1），所有图像归一化到MNI305；评估使用两组临床数据：19名健康志愿者（低场/高场对照）和11名MCI/AD患者（低场/高场对照）。

**📈 对比分析**

与三种基线（立方插值、LF‑SynthSR、SuperSynth回归）在健康队列中进行体积一致性评估，使用Pearson相关、ICC、Dice、体积偏差等指标。结果显示模型的后验均值在体积一致性上与SuperSynth相当（无显著差异），并在Dice上优于LF‑SynthSR；在MCI/AD队列中保留了海马萎缩和脑室扩大等疾病特征。

**⚠️ 局限性**

局限包括：1）仅在少量临床样本上验证，需更大样本的临床效用评估；2）低方差区域并不一定表示已完全恢复，需要与自然人群变异对比；3）合成低场训练依赖于域随机化的假设，真实扫描仪间的差异仍可能影响性能。

---

## 151. Hill Sampling for Test-Time Scaling: A Simple and Better Alternative to Repeated Sampling, Evolution, and Training

**arXiv ID:** 2609.25510 | [PDF](https://arxiv.org/pdf/2609.25510v1)

**作者:** Jacob Beck `[一作]` (Oracle), Ari Kobren `[通讯]` (Oracle)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种名为 Hill Sampling 的最小化方法：在冻结的 LLM 上反复采样并编辑当前最佳已验证程序，直接在可验证算法发现任务中搜索更优解。

**💡 创新点**

创新点在于证明仅使用“最佳已验证程序”作为上下文即可在大多数任务中达到或超越现有复杂进化/训练框架，而在权重空间上设置学习率为零的 ES 仍低于普通采样，说明 token 级别的多样性更有效。

**🔧 技术方法**

核心技术包括温度采样、LLM 提示与编辑、可执行验证器、Evolution Strategies、模型噪声对比、以及多种熵/多样性控制策略的实验。

**📊 数据集**

实验基准为三个可验证数学优化任务：圆形装填（circle packing）、有限集合和差（sets）、埃尔德斯最小重叠（Erdős minimum‑overlap）。

**📈 对比分析**

与重复采样、模型噪声、ES、AlphaEvolve、ShinkaEvolve 等多种基线对比显示，Hill Sampling 在圆形装填上创下新最优，在埃尔德斯上超过 AlphaEvolve，集合任务性能与最优基线相当；总计算成本仅需数小时，显著低于更复杂框架。

**⚠️ 局限性**

局限性在于仅在可验证的数学/算法任务上验证，缺乏对更一般任务的推广；某些任务仍需领域先验信息，并且对多步优化、执行反馈等复杂机制的实验表明它们在本设置下并未带来显著提升。

---

## 152. Key Reconciliation with RC-LDPC/Error Estimation for Satellite-based FSO/QKD Systems

**arXiv ID:** 2609.25646 | [PDF](https://arxiv.org/pdf/2609.25646v1)

**作者:** Cuong T. Nguyen `[一作]` (University of Aizu), Anh T. Pham `[通讯]` (University of Aizu)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种针对卫星自由空间光量子密钥分发（FSO/QKD）的盲式校准方案，结合原型图可调速率LDPC码与基于综合码矛盾的误差估计，并给出了完整的秘密密钥吞吐量（SKT）分析框架；通过Monte‑Carlo仿真验证，并在Starlink卫星通道案例中演示其性能；

**💡 创新点**

创新点在于：1) 将原型图可调速率LDPC码与误差估计方法首次联合应用于卫星FSO/QKD，显著降低通信轮数；2) 提出了首个考虑误差估计误差的SKT解析模型；3) 通过曲线拟合将块级QBER建模为指数化韦布尔分布，提升分析精度；4) 设计高效硬件友好的原型图LDPC族。

**🔧 技术方法**

技术包括：原型图可调速率LDPC编码、基于综合码矛盾的误差估计、非协同连续变调CV‑QKD双阈值探测、自由空间光信道模型（云散射、湍流、指向误差）、解析SKT模型、蒙特卡洛仿真、指数化韦布尔分布拟合。

**📊 数据集**

数据集为仿真产生的FSO信道数据，包含Starlink卫星轨道（TLE）与地面车辆运动轨迹，用于计算时变信道系数；无真实实验数据。

**📈 对比分析**

通过将所提方案与传统盲式校准在SKT指标下对比，发现当公通信道速率高或可调码率族宽时，所提方案比传统方案提升15%–30%；仿真结果与解析公式高度吻合，并展示了公共通道速率、误差估计精度以及卫星通道参数对吞吐量的影响。

**⚠️ 局限性**

局限性包括：误差估计步骤增加实现复杂度；估计误差会导致吞吐量下降；模型依赖于所假设的信道参数，未在实验平台上验证；仅适用于原型图LDPC族，扩展到其它码结构需进一步研究。

---

## 153. Prompt Breadth and Rollout Refresh Interact in On-Policy Distillation

**arXiv ID:** 2609.25048 | [PDF](https://arxiv.org/pdf/2609.25048v1)

**作者:** Lingxiang Hu `[一作]` (Tencent), Linfang Shang `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在保持相同轨迹数与优化器更新次数的前提下，Prompt 库的宽度（8、48、Full）与生成响应的学生模型快照数（1、10、110）对 On‑Policy Distillation（OPD）效果的交互影响，并进一步探讨了不同刷新策略在不同推理预算下的答案完整度与准确率差异。

**💡 创新点**

创新点在于：① 首次系统性揭示了 Prompt 宽度对 OPD 结果的取决于刷新频率，发现宽度在冻结响应时降低准确率，在每次更新刷新时提升准确率；② 通过两种教师模型的比较展示了短预算下周期性刷新提升答案覆盖率与解析度，但长预算时冻结响应模型以更高 token 代价取得更高平均准确率；③ 提出了完整的实验设置（N、M、C、U），强调在报告 OPD 效率时需要同时记录 Prompt 规模、刷新频率、token 成本与推理预算。

**🔧 技术方法**

主要技术：使用 Qwen3‑0.6B 作为学生、Qwen3‑4B‑Instruct 与 JustRL‑DeepSeek‑1.5B 作为教师，采用 PPO‑style OPD 损失（student‑top‑16、reverse‑KL 约束），在固定 14,080 条轨迹与 110 次优化器更新的 3×3 网格上实验；评估使用 AIME24/25、AMC23 等数学竞赛数据集，计算 macro3 平均准确率、pass@32、可解析答案比例与 token 消耗。

**📊 数据集**

数据集：英文版 DAPO‑Math‑17K（用于 Prompt 生成与训练），评估集包括 AIME24、AIME25、AMC23、AIME26、HMMT25‑Nov、HMMT26‑Feb、MATH‑500 与 OlympiadBench。

**📈 对比分析**

比较方法：在相同轨迹与更新量下对 9 个实验点进行宏观平均准确率和 pass@32 对比，并在不同推理预算（4K、8K、16K、24K、32K）下评估答案解析度与 token 消耗；结果显示在短预算下周期性刷新模型在 pass@32 上提升约 12–22 个百分点，而在 32K 预算下冻结响应模型准确率高出 2–4 个百分点，但需要 1.7–1.8 倍 token。

**⚠️ 局限性**

限制：仅使用单一学生模型和两位教师；实验网格离散且未覆盖所有刷新阈值；未考察训练种子随机性；token‑level 计算与实际训练时间不匹配；结果主要针对数学推理任务，缺乏跨任务验证。

---

## 154. Fully Byzantine-Resilient Multi-Agent Reinforcement Learning

**arXiv ID:** 2609.25701 | [PDF](https://arxiv.org/pdf/2609.25701v1)

**作者:** Haejoon Lee `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种完全鲁棒的分布式演员-评论家多智能体强化学习算法FRAC-MARL，能够在Byzantine边攻击下实现与无攻击场景相同的学习效果。

**💡 创新点**

创新点在于利用两跳通信的冗余信息进行可靠消息识别与过滤，提出新的(r,r')-冗余图拓扑条件，证明在该条件下能实现几乎确定收敛至攻击-free极值。

**🔧 技术方法**

采用分布式演员-评论家框架、线性/非线性函数逼近、冗余消息过滤、两跳通信、基于邻接矩阵的图论分析以及两时间尺度随机逼近理论。

**📊 数据集**

使用MPE2（Multi-Particle Environments 2）中的圆形编队任务数据集，包含10个机器人、速度、位置、相对位置等状态。

**📈 对比分析**

与普通无攻击、Naïve（不加防御）、Projection（投影+裁剪均值）和Trimmed-Mean（逐元素裁剪均值）四种基线进行比较，FRAC-MARL在F=1、F=2攻击下获得与无攻击Baseline相同的奖励曲线，其他方法则明显下降。

**⚠️ 局限性**

局限性在于只考虑攻击限定于通信层的Byzantine边攻击，且要求网络满足(r,r')-冗余且r>2F，实际部署时可能需要满足较高的通信冗余和网络设计约束。

---

## 155. MirrorDistill: Illumination-Aware Latent Distillation for Efficient Low-Light Restoration

**arXiv ID:** 2609.25331 | [PDF](https://arxiv.org/pdf/2609.25331v1)

**作者:** Farida Mohsen `[一作]` (Hamad Bin Khalifa University), Samir Brahim Belhaouari `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为MirrorDistill的照明感知潜在蒸馏框架，用于低照度图像增强。

**💡 创新点**

通过在训练时用清晰图像的共享编码器和EMA教师解码器产生清晰域潜在目标，并在编码器和解码器层级对低照度学生进行镜像对齐，同时使用基于像素亮度的照明加权，且在推理时仅使用学生网络，无教师计算。

**🔧 技术方法**

利用U-Net架构、CBAM注意力、EMA教师、特征对齐（encoder/decoder mirror）、标准化的多尺度解码投影、SSIM+L1重建损失以及照明加权。

**📊 数据集**

在LOL-v1、LOL-v2-Real和LOL-v2-Synthetic三套基准数据集上进行训练与评估。

**📈 对比分析**

与RetinexNet、KinD、EnlightenGAN、Restormer、MIRNet、SNR-Aware、Retinexformer等方法对比，MirrorDistill在LOL-v2-Real上取得最高PSNR/SSIM，并且GMACs仅为4.38，推理速度最快。

**⚠️ 局限性**

仅在训练阶段使用清晰参考和教师分支，推理速度极快，但对极端噪声或非对齐数据的鲁棒性尚未验证；同时依赖于高质量的低照度/正常照度配对数据。

---

## 156. ChatT2: An Adaptive Framework for Developing a Large Language Model-Based Agent for Natural Product Domain Research

**arXiv ID:** 2609.25620 | [PDF](https://arxiv.org/pdf/2609.25620v1)

**作者:** Yihan Wang `[一作]` (Beijing Normal University), Zhiwei Qin `[通讯]` (Beijing Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对微生物天然产物（尤其是细菌芳香性二聚体聚酮）研究，构建了一个基于大型语言模型（LLM）的多代理框架 ChatT2，帮助用户在不需要专业查询语言的前提下完成文献检索、结构/功能分析、化学改造及应用探索等任务。

**💡 创新点**

创新点包括：① 将 mentor、executor、evaluator 三个专门角色嵌入 LLM 框架，实现链式思考（CoT）与“问答循环”交互；② 采用检索增强生成（RAG）与多模态数据库（关系、向量、图像、工具）融合，显著提升信息检索覆盖率与可追溯性；③ 设计了自适应评估器，用于拒绝不确定问题、过滤无关信息与跟踪引用，降低幻觉风险；④ 开发了专属基准 NPBENCH（50个开放式题目），专门评测天然产物研究中的实际问题解决能力。

**🔧 技术方法**

技术实现主要包括：GPT‑4o mini 作为 LLM 核心，使用 128K 上下文窗口与 JSON 输出；链式思考提示与多轮交互；检索增强生成（SQL + 文本向量检索 + BM25 图像检索 + 预包装工具调用）；自定义关系数据库（MIBiG、PubChem 等）；向量数据库（Web of Science、MIBiG 文献向量化）；图像数据库与工具库；评估器使用基于提示的功能调用来控制输出质量。

**📊 数据集**

数据集：① 构建的多模态数据库（关系表、向量索引、图像库、工具库）涵盖 MIBiG、PubChem、Web of Science 文献；② 专门的 NPBENCH（type II polyketide 版本）包含 50 个开放式查询，覆盖合成、结构优化、功能、应用与分类等五大研究场景。

**📈 对比分析**

比较方法：将 ChatT2 与 GPT‑4o mini、Claude‑3.5‑Sonnet、Llama 3.1‑405B 等主流 LLM 进行单轮与多轮评估；评估指标为科学准确性、连贯性、相关性、具体性、可理解性；采用 GPT‑4 进行自动排序并多次打乱顺序，计算 Spearman 相关系数；在 NPBENCH 上，ChatT2 的“获胜率”在生物合成、结构、功能、应用、分类五类均超过 80% 或远高于基线模型，Spearman 相关系数均 >0.6，显示评估结果稳定可靠；多轮 CoT 评估中，ChatT2 的文档引用数、词汇多样性和文本长度随轮数递增，表现出显著的探索深度。

**⚠️ 局限性**

局限性：① 需要大量人工手工整理和扩展数据库，覆盖面有限；② 目前仅在 type II polyketide 领域验证，跨域泛化需要进一步评估；③ 随着内部知识库扩充，对外部检索依赖降低，可能导致模型更依赖内部推理，降低检索新颖性；④ 对英语信息库依赖较大，多语言支持尚不完善；⑤ 依赖 GPT‑4o mini，若升级 LLM 或训练数据变动需重新验证。

---

## 157. Marginal Log-Likelihood Increments under Dirichlet-Smoothed Markov Estimation

**arXiv ID:** 2609.25675 | [PDF](https://arxiv.org/pdf/2609.25675v1)

**作者:** Levin David Schwab `[一作]` `[通讯]`, Levin David Schwab

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文研究了在Dirichlet平滑的一阶马尔可夫模型中，单条工作流轨迹加入训练档案后对参考加权对数似然（即交叉熵）产生的精确增量，并将该增量解析为KL散度的加权降低，从而给出可实现收益的上限；随后探讨了该增量与批量选择的交互关系，揭示其非子模性质，并验证描述符预测器在实际批量质量上的局限性。

**💡 创新点**

创新点包括：
1) 推导单条轨迹增量的闭式表达并将其写成KL散度形式，直接给出对模型改进的最大可能收益；
2) 通过二阶导数分析证明增量交互既可以是补充也可以是替代，说明批量选择非子模/超模；
3) 证明单条增量预测的精确度与批量质量无必然关联，指出描述符预测误差的本质限制；
4) 将上述理论应用于实际业务日志，系统评估多种采样与预算策略。

**🔧 技术方法**

使用的技术包括：
- Dirichlet平滑的一阶马尔可夫模型（Jeffreys先验）
- 对数损失与KL散度的解析推导
- 线性回归与随机森林对增量的预测
- 角色分配与预算控制的实验设计（案例预算/转移预算）
- 二阶导数分析和不等式证明批量交互特性
- 统计比较（随机、最短先、岭回归、随机森林、校准分数、测试分数、完整池）

**📊 数据集**

实验基于 BPI Challenge 2012 贷款申请事件日志，共 13,087 案件、23 种活动；在该日志中提取 164,506 条事件进行分析，并按时间划分为早期与晚期两段，分别进行评估。

**📈 对比分析**

比较方法：
- 在固定的案例/转移预算下，分别使用随机选择、最短先、岭回归、随机森林、静态/顺序校准分数、测试分数或完整池进行批量采集；
- 评估指标为对数似然提升（毫纳特）；
- 结果显示：随机选择在案例预算下只能获得约 43% 的可实现收益；随机森林在案例预算下相较随机提升约 65.2 毫纳特；顺序校准规则在两种预算下能够实现约 82% 的可达收益；而描述符方法表现不稳定，且其单条增量预测精度与批量收益不匹配。

**⚠️ 局限性**

限制：
- 模型仅考虑一阶马尔可夫，无法捕获长期依赖；
- 描述符仅保留统计特征，导致预测误差较大；
- 增量非子模导致贪心策略可能非最优；
- 实验仅使用单一日志，结果对角色分配和时间切分敏感；
- 上限（Λ_q）基于经验参考计算，未给置信区间；
- 评价仅关注信息量差异，未考虑实际成本（隐私、存储、标注等）。

---

## 158. Zephyron: Integrated Design and Analytical Evaluation of a Solar-Assisted Mobile Manipulator for Multimodal Environmental Reconnaissance and Distributed Visual Inference

**arXiv ID:** 2609.25709 | [PDF](https://arxiv.org/pdf/2609.25709v1)

**作者:** Sabik Bin Sultan `[一作]` (Bangladesh Air Force Shaheen College Kurmitola), Safwan Sadad `[通讯]` (Greenland Residential School)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Zephyron四轮移动平台的可追溯设计与分析评估，构建了基于组件尺寸、质量、电能与运动学的基线模型，并制定了基于能量与测量质量的任务执行策略。

**💡 创新点**

创新点在于：①将照片构造的物理布局转化为可计算的组件与几何基线；②通过分层证据追踪（照片、制造商规范、文献）消除非支持的量化假设；③设计了可执行的“质量感知采样”决策框架，将能量预算、传感器响应时间与测量不确定性统一进决策。

**🔧 技术方法**

主要技术包括：文献检索与系统化筛选、确定性力学与能量模型、滚阻与坡度分析、光伏充电与电池功耗计算、传感器不确定性解析、视觉检测管道延迟建模、机器学习模型比较（MobileNetV3-Small/Large + SSDLite、YOLOv4-tiny）以及基于时间戳与质量标记的帧过滤与检测匹配。

**📊 数据集**

本研究未使用公开实验数据集；所有计算基于制造商规格、参考文献参数和假设情景；视觉模型的基准来自公开的YOLO/SSD性能报告，未在Zephyron平台上实际训练或测试。

**📈 对比分析**

比较方法主要是基于解析模型的数值结果与文献中的规范或推荐值进行对比，未给出真实性能指标；对能量、扭矩、采样不确定性等参数给出了估算值，表明在所设定假设下可行性。

**⚠️ 局限性**

局限性包括：①缺乏现场实验验证，所有结果为理论预测；②对传感器与动力系统的性能假设未经过校准；③视觉与机器学习模型未在硬件上部署；④模型对复杂地形、动态负载、通信衰减等真实环境因素的鲁棒性未评估。

---

## 159. A Practical Recipe for Semi-Supervised Federated ASR: Online Pseudo-Labels with Server Update Stabilization

**arXiv ID:** 2609.25471 | [PDF](https://arxiv.org/pdf/2609.25471v1)

**作者:** Wonho Bae `[一作]` (Apple), Sheikh Shams Azam `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究半监督联邦学习（SSFL）在自动语音识别（ASR）中的方法与稳定性问题，探讨不同伪标签生成器与服务器更新策略对性能的影响

**💡 创新点**

发现在线教师（per-client evolving teacher）在合适的服务器更新与数据增强下能匹配或优于全局教师，提出“过渡教师”策略以及基于SpecAug强度和批量大小的稳定化方法

**🔧 技术方法**

使用Transformer编码器+CTC目标，结合FedAvg、LAMB优化器、SpecAug增强、随机梯度裁剪、EMA教师等技术

**📊 数据集**

在四个公开英语语料库（LibriSpeech、TED-LIUM、Common Voice、Fisher）上进行跨域与同域实验，模拟不同服务器/客户端数据分布

**📈 对比分析**

与现有SSFL方法（Static PL、FedNST、Rao等）对比，稳健的在线/全局教师+稳定化配置在11对（seed、client）组合上平均降低WER约20–40%，显著逼近全监督FL水平

**⚠️ 局限性**

局限包括仅在英语、Transformer+CTC框架下评估；未考虑语言模型解码、DP隐私、真实部署的客户端漂移、网络失败及资源约束，可能影响结果泛化

---

## 160. Deepfakes and Synthetic Media: Generation, Detection, and Governance

**arXiv ID:** 2609.25017 | [PDF](https://arxiv.org/pdf/2609.25017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 161. NPLSD: Accelerating Line-Segment Detection on NPU Microcontrollers

**arXiv ID:** 2609.25022 | [PDF](https://arxiv.org/pdf/2609.25022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 162. Lightweight Ranking Heads: Accelerating Multi-Task Experimentation in Production Recommender Systems

**arXiv ID:** 2609.25433 | [PDF](https://arxiv.org/pdf/2609.25433v1)

**作者:** Sanjay Surendranath Girija `[一作]` (Google LLC), Mohit Sharma `[通讯]` (Google LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了Lightweight Ranking Heads（Light Heads）框架，用于在多任务推荐模型中动态注入轻量化预测头，避免重训练后端模型。

**💡 创新点**

创新点在于：①使用中央配置实现跨模型统一注入；②通过 stop‑gradients 与每次训练开始时重置，保持共享层不受新任务影响；③在持续在线学习环境下实现无冷启动、无额外计算成本的新任务实验。

**🔧 技术方法**

主要技术包括：分布式持续训练（continuous online learning）、停梯度（stop‑gradient）、轻量化深度前馈头、配置中心化管理、自动化监控与回退策略。

**📊 数据集**

使用 YouTube 生产级数据，涵盖 Home、Watch Next 等大规模推荐模型的日志与点击/观看等稀疏/稠密目标。

**📈 对比分析**

通过对比 Light Heads 与全头（full head）的 AUC / RMSE 等指标，发现稀疏任务上几乎相同，稠密任务略低；在实验周期上从 24 天缩短至 11 天；并在 A/B 试验中显著提升顶级与垂直指标。

**⚠️ 局限性**

局限性包括：①需要持续在线训练的基础设施；②在稠密任务上会有轻微准确度下降；③对单阶段系统收益有限；④在非大规模并行实验环境下维护成本不划算。

---

## 163. Terminal Shrinkage Averaging Reveals a Schedule-Estimator Interaction in LLM Pretraining

**arXiv ID:** 2609.25482 | [PDF](https://arxiv.org/pdf/2609.25482v1)

**作者:** Adam Ousherovitch `[一作]` (University of Michigan), Yixin Wang `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证终端收缩平均（TSA）方法，将最终迭代与最近检查点平均相结合以降低模型方差。

**💡 创新点**

通过将学习率冷却与输出估计器分离，发现适度平均可让终端学习率更活跃而不损失性能，并给出局部二次近似下的偏差-方差分析。

**🔧 技术方法**

理论分析基于局部二次模型；实验使用NanoChat 12/22层语言模型，采用Muon+AdamW和AdamW优化器。

**📊 数据集**

使用NanoChat自带的预训练数据集（与GPT‑2相同规模）。

**📈 对比分析**

通过对比不同终端学习率底线（5%、10%、15%）与不同估计器（原始迭代、TSA、LAWA），在BPB和DCLM CORE上实现至少0.0007 BPB提升，深层模型在一次跑中达成GPT‑2资格。

**⚠️ 局限性**

仅探索了单一全局收缩系数和简单学习率底线；对不同层/张量的不同平均量、其他冷却形状或更大规模训练的泛化未知。

---

## 164. ufakzeka-1: Building and Evaluating a 151M-Parameter Turkish Language Model from Scratch

**arXiv ID:** 2609.25081 | [PDF](https://arxiv.org/pdf/2609.25081v1)

**作者:** Sait Furkan Teke `[一作]` `[通讯]` (ufak AI), Sait Furkan Teke (ufak AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 ufakzeka-1，151 M 参数的 Turkish 语言模型，并对其预训练、后训练、评估流程进行了完整记录和公开。

**💡 创新点**

首次完整公开了从零开始训练、后训练、评估和发布的全过程；采用三阶段预训练与混合数据后训练，设计了多维安全门与手工评测方法，并系统揭示了小模型测评中的种子方差与数据缺失效应。

**🔧 技术方法**

使用 byte‑level BPE tokenizer、Qwen3‑style transformer（24 层、768 隐藏、12 头、SwiGLU、Rotary、QK‑norm、grouped‑query attention），Muon/Hyperball 预训练优化，AdamW、SFT 后训练，评估采用 lm‑evaluation‑harness、LLM judge 及手工对话检查。

**📊 数据集**

FineWeb2‑HQ、Mogan Turkish crawl、FinePDFs‑edu、FineWiki、BILGE synthetic、COSMOS synthetic、FineMath、Turkish Wikipedia、Common Crawl、Turkish instruction datasets（Turkish‑SFT、Turkce‑Atlas‑Instruct、Aya split）等全许可证文本。

**📈 对比分析**

与 Kanarya‑750M、turkish‑gpt2‑large、Qwen2.5‑0.5B 等模型在 Turkish 任务（HellaSwag、ARC、XCOPA、TurBLiMP、TurkishMMLU）做 Zero‑shot log‑likelihood 对比；ufakzeka‑1 在大部分 Turkish 任务中接近或超过同类 5‑倍参数模型，安全门与算术测评也比更大模型有显著提升。

**⚠️ 局限性**

生成虚假事实、无法写代码、算术误差、人物混淆、记忆不完整、对敏感话题安全性有限；训练种子方差大导致小样本评测不稳定，某些算术与身份追踪仍受模型规模限制。

---

## 165. Qwen3.8-Omni: Towards Native Omni-Modal Agents

**arXiv ID:** 2609.25611 | [PDF](https://arxiv.org/pdf/2609.25611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 166. Real-World Perception for Autonomous Driving in Adverse Weather: Enhancing Standard Detectors via Foundation-Guided Auto-Annotation

**arXiv ID:** 2609.25515 | [PDF](https://arxiv.org/pdf/2609.25515v1)

**作者:** Sepideh Gohari `[一作]` (Virginia Commonwealth University), Azim Eskandarian `[通讯]` (Virginia Commonwealth University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在实际恶劣天气和光照条件下，使用大型视觉基础模型 SAM3 进行离线自动注释，并利用其生成的伪标签微调 YOLOv8，以提升对象检测性能。

**💡 创新点**

创新点在于将开源基础模型用于离线伪标注，既保留了轻量级检测器的实时推理速度，又避免了昂贵的人工标注工作。

**🔧 技术方法**

采用的技术包括 SAM3 的提示式分割与伪标签生成、YOLOv8（单阶段 CNN 检测器）以及 Co‑DETR（Transformer 结构）进行对比评估。

**📊 数据集**

使用自研的 25 条不同路段（校园、高速、公路、乡村）+ 3 种天气（雾、雨、雪）+ 3 种光照（直射光、弱光、无光）组合的全景摄像头数据集，共 156,345 张关键帧，其中 750 张已手工标注。

**📈 对比分析**

通过在 25 个场景上计算 mAP_50:95 对比，SAM3 以 56.14% 的平均 mAP 超过 YOLOv8（34.19%）和 Co‑DETR（53.21%），随后用 SAM3 生成的伪标签微调 YOLOv8，平均 mAP 提升至 50.23%，在住宅直射光 +32.73% 与高速雾 +28.65% 的显著增益。

**⚠️ 局限性**

局限性在于仅使用被动摄像头系统，极端环境（如高速无光）仍导致检测性能急剧下降，亟需与激光雷达、雷达等主动传感器融合以提升安全性。

---

## 167. Directional Total Variation-Regularized Implicit Neural Representations (DTV-INR) for Continuous Super-Resolution in Degraded Imaging Domains

**arXiv ID:** 2609.25429 | [PDF](https://arxiv.org/pdf/2609.25429v1)

**作者:** Mahmoud Saeedi Kelishami `[一作]` `[通讯]` (Islamic Azad University), Mahmoud Saeedi Kelishami (Islamic Azad University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了基于方向性总变分正则的隐式神经表示（DTV-INR），用于解决连续超分辨率的逆问题。

**💡 创新点**

创新点在于将结构张量驱动的方向性总变分与SIREN隐式网络结合，理论证明存在唯一最优解并消除阶梯化，同时支持任意连续放大比例。

**🔧 技术方法**

使用了SIREN隐式网络、结构张量估计、方向性总变分正则、变分框架、交替投影优化以及AdamW学习率衰减等技术。

**📊 数据集**

实验数据集包括临床脑MRI（OASIS）、高分辨率电子显微镜图像以及全切片组织图像。

**📈 对比分析**

与双三次插值、离散TV、EDSR、LIIF、Vanilla SIREN及传统TV-INR比较，DTV-INR在×2–×8放大倍率下PSNR提升约+1.7–2.9 dB，噪声鲁棒性更强，且在任意连续尺度下保持平滑恢复。

**⚠️ 局限性**

局限性包括需要手动调节超参数、单场景优化耗时、目前仅在二维理论已成熟、对无明显主导方向的纹理图像效果有限。

---

## 168. Predictive Uncertainty for Neural CAE Surrogates

**arXiv ID:** 2609.25430 | [PDF](https://arxiv.org/pdf/2609.25430v1)

**作者:** Kaustubh Tangsali `[一作]` (NVIDIA), Sanjay Choudhry `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在三大工业CAE数据集（车辆气动、机翼流动与汽车碰撞）上，对基于高斯过程、Concrete MC dropout以及深度集成的几种不确定性量化方法进行统一评估，探讨其在预测精度、区间可信度、误差判别、分布外检测以及工程量级别上的表现。

**💡 创新点**

创新点在于：①提出了面向CAE工作流的三重评估范式（案例级、局部场级、工程量级）；②在同一网络架构下对闭式GP、采样型MC dropout与集成方法进行完整对照；③将不确定性量化与工程量传播（如阻力系数、撞击位移）相结合，揭示空间协方差对最终不确定性的重要性。

**🔧 技术方法**

使用的技术包括：深度Transformer+几何编码器的GeoTransolver架构；带有深度核学习的变分高斯过程；Concrete MC dropout（可学习的连续化dropout）；以及五成员深度集成。实验中还引入了后验校准、Spearman相关、AUSE、AUROC等多种评估指标。

**📊 数据集**

采用的三个数据集：DrivAerStar（三种车身后段样式的气动场与阻力），AirFRANS（二维雷诺平均Navier-Stokes机翼流动，角度外推测试），以及一套基于有限元的汽车碰撞动态模拟（多时间步、多个位移通道）。

**📈 对比分析**

比较方法主要是：对相同训练数据、相同网络骨干下的三种UQ技术进行逐项对比；结果表明：GP总方差在DrivAerStar和碰撞数据集上区间尺度最接近真实，采样型方法误差判别更好；在角度外推（AirFRANS）中GP的后验方差对OOD更敏感，而MC dropout的误差排名更稳健；不同方法在OOV检测、工程量不确定性传播以及计算成本（模型数、推理次数）方面呈现各自优势与劣势。

**⚠️ 局限性**

局限性包括：①实验仅在三个数据集上验证，未覆盖更广泛的CAE任务；②GP方法在工程量传播时只能使用点wise边际方差，忽略空间协方差；③采样型方法的置信区间需要后验校准，且校准仅在与训练相似的分布下有效；④实验中的评估指标与实际工程决策对齐度有限，需进一步研究如自适应学习、主动查询等下游工作。

---

## 169. Scoring Grant Applications with Large Language Models

**arXiv ID:** 2609.25327 | [PDF](https://arxiv.org/pdf/2609.25327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 170. Multi-Term Fourier Graph Neural Network with Sample Relationship Learning for Enhanced Remaining Useful Life Prediction

**arXiv ID:** 2609.25179 | [PDF](https://arxiv.org/pdf/2609.25179v1)

**作者:** Ya Song `[一作]` (Eindhoven University of Technology), Yingqian Zhang `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种多时长傅里叶图神经网络+样本关系学习框架，用以改进剩余使用寿命(RUL)预测；

**💡 创新点**

创新点在于①将样本视为完整图并在频域内用傅里叶图神经网络提取时空特征；②采用多时长学习模块（多窗长）捕捉短期与长期依赖；③构造样本关系图并通过异构消息传递网络学习样本间关系，从而提升预测鲁棒性；

**🔧 技术方法**

技术包括傅里叶图神经网络(FGN)、多时长滑窗采样、异构图神经网络(HMPN)、图到序列池化与LSTM/MLP预测；

**📊 数据集**

使用公共CMAPSS（FD001–FD004）涡轮发动机监测数据集；

**📈 对比分析**

与多种先进的序列模型（Transformer等）和ST‑GNN模型进行对比，实验表明MTFGN‑SRL在所有子集上均取得最低RMSE（平均11.52）和Score（平均444.00），比第二佳LOGO降低约11.6% RMSE、21.6% Score；

**⚠️ 局限性**

局限性包括：需要预先设定多窗口大小与超参数，模型复杂度较高，训练时对频域变换与图处理有额外计算开销，且在不同故障模式下的泛化能力尚待进一步验证。

---

## 171. Mitigating Sequential Reappearance in Diffusion Data-Point Unlearning

**arXiv ID:** 2609.25166 | [PDF](https://arxiv.org/pdf/2609.25166v1)

**作者:** Donghyun Kim `[一作]` (Konkuk University), Sangwoo Hong `[通讯]` (Konkuk University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对扩散模型的数据点忘却问题，本文提出了目标级的序贯评估协议，发现了“序贯重现”这一失败模式，并基于局部恢复几何提出了自适应低损失邻域搜索与干预的 LASTING 方法，实现了忘却效果的持久性。

**💡 创新点**

创新点包括①引入序贯重现概念并证明其与局部恢复尖锐度相关；②提出数据空间恢复尖锐度度量；③设计了 LASTING 通过自适应搜索最易恢复邻域并在删除时直接针对该邻域进行更新，从而提升忘却持续性。

**🔧 技术方法**

主要技术包括扩散模型（DDPM）对抗训练、SSCD 相似度评估、投影梯度搜索寻找低损失邻域、基于 SISS 的目标更新框架、以及在稳定扩散（Stable Diffusion）中对 latent 空间的应用。

**📊 数据集**

使用了 CelebA‑HQ 256×256 无条件扩散模型与 Stable Diffusion v1.4 文本条件潜在扩散模型作为实验数据集。

**📈 对比分析**

与 EraseDiff、ReTrack、SISS、Forward KL、Prompt‑Free 等基线相比，LASTING 在序贯删除序列（至 300 次请求）中显著降低了重现率（0.33%），提升了最终成功率（97.67%），并保持了较低的 FID/CLIP‑IQA，表现出优越的忘却持久性与生成质量。

**⚠️ 局限性**

局限性包括：①只在单个扩散模型与特定任务上验证；②对大规模真实世界数据的泛化能力未充分评估；③自适应邻域搜索成本较高，可能限制在资源受限环境下的实时应用。

---

## 172. Peerify: Benchmarking Peer-Review Claim Verification

**arXiv ID:** 2609.25046 | [PDF](https://arxiv.org/pdf/2609.25046v1)

**作者:** Alireza Daghighfarsoodeh `[一作]` (Reviewerly), Ebrahim Bagheri `[通讯]` (Reviewerly)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一套端到端的论文评审主张验证流水线，能够把评审意见拆分为原子主张，检索对应论文段落并判断每个主张是否得到支持；

**💡 创新点**

创新点在于将稿件内部验证问题拆解为分解、检索和四级支持判定三步；构建了 800 条真实评审主张的基准数据集（含 300 条人工审核子集），并通过自动化标签与人工标注相结合的方式提升评审质量；

**🔧 技术方法**

技术包括：基于 LLM 的主张分解（Gemma、Fenice 等）、BM25 与稠密检索（可再重排序）用于检索证据、长上下文 LLM 作为验证器（如 200K‑context Reasoner 等），以及零射 MNLI 基线进行对比；

**📊 数据集**

使用的数据集为 800 条来自 NeurIPS 2024 与 ICLR 2024 OpenReview 的评审主张及对应论文，其中 300 条为人工审核子集，并划分为 PaperSourced、RebuttalSourced、LLMJudged、Human‑Verified 等子集；

**📈 对比分析**

通过在不同检索配置（BM25、稠密检索、重排序）和不同验证器模型下进行评估，结果显示在 PaperSourced 上最佳模型准确率可达 0.866，宏 F1 约 0.92；在最难的 RebuttalSourced/LLMJudged 上准确率降至 0.38‑0.44，宏 F1 仅 0.45‑0.50；零射 NLI 仅能取得 0.24 以下宏 F1；自动化标签与人工标注的一致率 90.3%（κ = 0.87）；

**⚠️ 局限性**

局限性包括：仅对稿件内部验证，无法处理需要外部文献或背景知识的主张；基准仅来源于 NeurIPS 与 ICLR 公开数据，可能不适用于其他领域或不同评审规范；监督信号（作者回应、LLM 判定）仍较弱，存在歧义；模型校准偏差明显，单靠准确率容易误导，需同时报告宏 F1；

---

## 173. Beyond Natural Language: An Agent-Native Language for Autonomous Science

**arXiv ID:** 2609.25421 | [PDF](https://arxiv.org/pdf/2609.25421v1)

**作者:** Yifeng He `[一作]` (ARA Lab), Jiachen Liu `[通讯]` (ARA Lab)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套可执行的科研声称支持语言（Claim‑Support Language）及其检查器，实现科研声称、证据与论证的机器可验证与跨上下文迁移。

**💡 创新点**

创新点包括：① 对科研证据、推理规则、关键问题与攻击进行显式类型化的计算机可检查语法；② 引入严格后端注册与证书机制，保证推理的逻辑一致性；③ 通过“结构桥”实现可能世界语义下的跨论文支持传递；④ 将论证框架编译为Dung模型并使用基于根拠的四状态评估；⑤ 在Lean 4中完全机械化元理论，并通过差分测试验证Haskell实现的等价性。

**🔧 技术方法**

技术手段：正式论证框架（Dung论证、Grounded语义）、类型检查与归一化、严格后端证书接口、结构桥与可能世界语义、Lean 4机械化与Haskell实现、差分测试、基于攻击的可执行性检查。

**📊 数据集**

数据集主要为自构造的基准案例与示例：144 文件、约117 000行Lean代码；差分测试使用673个通过判定、66个错误输入；论文中包含四个案例研究（实验对比、哲学争议、命题撤销等）。

**📈 对比分析**

性能评估：对任意n个论证的Grounded评估最多执行n³(1+n)次攻击查询；至少n²次；在可实现框架上，最坏情况为四次方（k⁴）查询。差分测试在实际程序上平均每个程序运行≤0.5 秒，所有测试通过；证明和替换后端保持同一claim状态，保证了性能与准确性。

**⚠️ 局限性**

局限性：① 文章到正式语句的绑定不可信，需人工验证；② strict‑chain well‑formedness 约束限制了对冲突严格规则的使用；③ 仅实现了确定性降阶路径，对任意论文的自动化降级未测试；④ 对证据真实性、实验环境等无法自动判断；⑤ 结构桥的可用性取决于研究者手工制定，缺乏自动发现机制。

---

## 174. Self-Cleaning and Captured Anyway: One Measured Primitive for Error in a Store an Agent Writes to Itself, and What a Falling Score Actually Measures

**arXiv ID:** 2609.25052 | [PDF](https://arxiv.org/pdf/2609.25052v1)

**作者:** Wenhui Chen `[一作]` (University of Macau), Chi Man Vong `[通讯]` (University of Macau)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种用于预测追加式存储捕获率的原语 γ(φ)，并在多主题环境下评估其性能。

**💡 创新点**

创新点在于构造可直接与主题分布关联的预测函数 γ(φ)，以及通过解析推导出存储捕获上限 (n-1)/n 的结论。

**🔧 技术方法**

主要技术包括：基于 E-bimodal2 的仿真、均值场方程建模、误差分析以及对比实验。

**📊 数据集**

使用的数据集为 44 个主题的真实测量结果以及 3 个主题的拆分测试，共计 45 个模型 × 9 个 φ 参数组合。

**📈 对比分析**

方法与基线比较：对比预测捕获率与实测捕获率；对比不同读取策略（脚本、多数表决、最近一次）在存储大小上的表现；结果显示 γ(φ) 在 44 主题下预测准确，但在 3 主题下存在显著下偏；捕获率随存储大小趋近于上限 (n-1)/n。

**⚠️ 局限性**

局限性包括：对少量主题的预测不准确；假设读取独立性失效导致无法给出结果方差界；模型缺乏对跨种子相关性的解释；未在所有条件下验证理论边界。

---

## 175. MIND the Gap: A Geographic Implicit Neural Representation with Adjustable Spatial Scale

**arXiv ID:** 2609.25454 | [PDF](https://arxiv.org/pdf/2609.25454v1)

**作者:** Isaac Corley `[一作]` (Taylor Geospatial), Hannah Kerner `[通讯]` (Taylor Geospatial)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `79276348-11e0-48e3-84bc-7ec231d0171c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

介绍了一种名为Matryoshka Implicit Neural Distillation（MIND）的地理隐式神经网络，能够学习可调节空间分辨率的坐标嵌入，并构建了统一的全球基准集CoordBench；

**💡 创新点**

创新点包括：①通过分块嵌入实现可调空间粒度的nested supervision；②提出调度正则化（Chunk‑Penalty）在不重训练的情况下抑制细节；③发布了全球1公里网格嵌入和大规模评估套件；

**🔧 技术方法**

使用了余弦+MSE损失的多教师蒸馏、ReSIREN网络、分块嵌入与正则化、线性下游预测、坐标投影、PHI‑S标准化等技术；

**📊 数据集**

训练基于12M地理坐标与教师嵌入的MINDSET数据集，教师来自AEF、Climplicit、GeoCLIP、SINR；评估使用CoordBench中的52个数据集、78个目标（包括气候、土壤、人口、收入等）；

**📈 对比分析**

与9种预训练INR、无训练坐标编码和Coordinate IDW比较，采用随机折叠与区域留存两种划分。MIND在随机折叠下R²与IDW相近，但在10°、40°区域留存上显著优于所有INR，成为基准中的最高总分；

**⚠️ 局限性**

局限性：仅编码位置，未考虑时间维度；训练样本集中于城市地区，乡村或稀疏生物群落的表示可能不足；嵌入分块顺序未显式强制空间尺度，需要进一步的显式空间尺度监督。

---

## 176. FinFIRST: Benchmarking Search Agents for Financial Information Retrieval, Sourcing and Traceability

**arXiv ID:** 2609.25192 | [PDF](https://arxiv.org/pdf/2609.25192v1)

**作者:** Wenqing Wang `[一作]` (Ling Team, Inclusion AI), Jun Zhou `[通讯]` (Ling Team, Inclusion AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了名为FinSearchComp的金融信息检索、来源验证与可追溯评估基准，包含123个专家编写、难度梯度分布的任务，并提供原始证据与原子化评分体系；

**💡 创新点**

创新点在于：①将金融检索任务拆解为原子化指标（原始信息获取、来源验证、计算与答案形成）以精细诊断模型弱点；②设计18字段分类、6轴覆盖蓝图与138源路由图，保证覆盖实际业务需求；③统一工具环境（web search、网页抓取、Python执行）与自动化评判器，兼顾可复现性与高效评测；

**🔧 技术方法**

技术上采用多模态检索与推理框架（ReAct交互式协议）结合外部工具；评判器基于GLM-5.1模型实现；

**📊 数据集**

数据集为FinSearchComp，包含123个任务，分布为易 31.7%、中 42.3%、难 26.0%；每题均附原始证据、参考答案与原子化评分规则；

**📈 对比分析**

与15个模型配置（包括Claude‑Opus‑5、GPT‑5.6‑Sol、Qwen3.8‑Max等）在统一工具环境下进行评测；结果显示Claude‑Opus‑5最高Atomic 87.59%，Loose 87.61%，但Strict Pass仅69.11%；计算与答案形成的原子评分普遍低于信息获取，说明计算仍是瓶颈；

**⚠️ 局限性**

局限性包括：①仍依赖人工评判器的准确性；②仅覆盖公开可访问的金融资料，可能忽略内部数据需求；③工具使用效率与模型性能未完全对应，仍需进一步优化搜索策略；

---

## 177. WeightBridge: An Efficient Weight Transfer Library for Reinforcement Learning

**arXiv ID:** 2609.25442 | [PDF](https://arxiv.org/pdf/2609.25442v1)

**作者:** Xuanlin Jiang `[一作]` (FAIR at Meta), Carole-Jean Wu `[通讯]` (FAIR at Meta)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一套适用于RL后训练的权重传输库，能够自动识别不同训练与推理布局之间的对应关系，生成全局最优的多对多路由计划，并在多种同步模式下协调工作者完成高效的权重更新。

**💡 创新点**

主要创新点在于：①无需模型或后端特定规则即可自动提取布局对应关系；②设计三阶段全局路由方案，消除冗余传输并实现负载平衡；③提供极简API，支持同步、异步及混合同步等多种RL同步模式。

**🔧 技术方法**

技术手段包括基于检查点的源映射提取、三阶段RDMA+NVLink传输、批量打包/组装GPU内核、以及使用WeightBridge抽象层实现跨后端的无耦合控制。

**📊 数据集**

评测使用了Moonlight‑16B、Qwen3‑30B、Qwen3‑235B以及1T Kimi‑K2等大模型，并在H100集群上对DAPO‑Math数据集进行推理生成与训练。

**📈 对比分析**

通过与Miles的Broadcast和P2P两种基线对比，测量平均GPU停顿时间（AGST）和端到端传输时间（EWTT），在不同规模与布局下实现了25‑42倍的AGST提升，EWTT几乎逼近理论下限。

**⚠️ 局限性**

局限性包括：目前假设训练与推理参数精度一致，未处理跨精度量化；缺乏弹性扩展和容错机制；以及对动态资源变动的支持仍待完善。

---

## 178. Benchmarking Neural Defend ARCAS 1B: A Foundational Multimodal Deepfake Detection Model

**arXiv ID:** 2609.25154 | [PDF](https://arxiv.org/pdf/2609.25154v1)

**作者:** Sivashankar Selvarajan `[一作]` (Neural Defend Inc), Sharayu N. Deshmukh `[通讯]` (Neural Defend Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 Neural Defend ARCAS 1B 图像检测器在七个公开基准（Chameleon、OpenFake、Community Forensics、AIGIBench、AIGCDetectBenchmark、WildFake、GenImage）上的评估记录进行二次分析，保留原始聚合结果并补充记录级、覆盖率、子组诊断与跨基准汇总。

**💡 创新点**

提出了基准本地报告与统一汇总并行的评估框架，强调在不同聚合方式下的可解释性，并在每个基准中对错误模式与覆盖率进行细粒度展示，避免单一指标掩盖差异。

**🔧 技术方法**

使用统计评估指标（准确率、AP、AUROC、宏平均准确率、宏平均AP、Wilson 置信区间、EER 等）以及阈值下的混淆矩阵，结合覆盖率计数和子组诊断来完整描述模型表现。

**📊 数据集**

利用七个公开基准的评估数据集，包括 26,033 张 Chameleon 记录、94,544 张 OpenFake 记录、51,836 张 Community Forensics 记录、212,802 张 AIGIBench 记录、152,597 张 AIGCDetectBenchmark 记录、660,935 张 WildFake 可用记录以及 100,000 张 GenImage 记录。

**📈 对比分析**

与已发表的检测器（如 AIDE、FGINet、PPM‑CLIP、PatchCraft、SDAIE 等）进行基准对齐比较，结果显示 ARCAS 1B 在大多数基准上达到 99% 以上的准确率（最差为 97.44% 的 Chameleon），但不同基准的子组错误模式和阈值误差显著，说明单一指标无法完整表征性能。

**⚠️ 局限性**

局限性包括仅基于提供的检测器得分进行评估；未验证模型的校准、时序漂移、鲁棒性和公平性；基准覆盖率不完整（如 WildFake 路径缺失、AIGCDetectBenchmark 解码失败）；跨基准文件名与标签重叠约 88,000 条；并且评估缺乏对不同来源、压缩与生成器演进的外部验证。

---

## 179. Evaluating Shaker for Flaky Test Detection in Python Projects

**arXiv ID:** 2609.25528 | [PDF](https://arxiv.org/pdf/2609.25528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 180. X-Planner: Event-Structured Task Planning for Embodied Intelligence

**arXiv ID:** 2609.25187 | [PDF](https://arxiv.org/pdf/2609.25187v1)

**作者:** Howard Lu `[一作]`, Qian Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于事件结构的任务规划前端，能够将高层指令和多视角观测映射为可执行的事件描述或连续CoT状态，随后驱动底层控制模型完成机器人操作。

**💡 创新点**

创新点包括：①统一多源数据（Egocentric、UMI、遥控操作）构建层级化事件标注，①利用事件边界与错误检测实现持续监督；②同时提供可解释的事件语言计划和无符号的连续CoT；③采用“Staircase Decoding”在Transformer层级上并行生成CoT状态，显著减少串行解码延迟；④在连续状态上引入冻结的文本重构目标，保证语义可解释性。

**🔧 技术方法**

主要技术：多模态视觉‑语言模型（Qwen‑系列 VLM）作为特征提取器；Mixture‑of‑Transformers 进行 Staircase Decoding；JSON 结构化监督；跨模型文本对齐损失；冻结语言模型重构 CoT；对接现有世界‑动作模型（cross‑attention 交互）。

**📊 数据集**

使用了三大来源的演示数据：Egocentric、UMI 和遥控操作，总共 1,654 条，挑选 1,500 条做分析；数据包含 L3‑Task、L2‑Subtask、L1‑Action、L0‑Segment 四级层级标注，并补充了 takeover‑time 与人工设计失败示例，覆盖 30 种任务类别和 41 种原子动作。

**📈 对比分析**

与 Qwen、Doubao、kimi3 进行离线两步规划文本评估，BERTScore‑F1 为 0.9011、Overall 1.411，位于四者第二；在真实机器人实验中，事件模式系统在 Reasoning Manipulation 套件中达 71.60 的 Task Progress，优于 U‑Scratch（59.50）等基线；在 Generalization 套件中达 53.75，超过 DreamZero（28.50）等基线。

**⚠️ 局限性**

局限性：评估缺乏统计显著性与误差分析；未对两种计划形式进行 ablation；未测量整体端到端延迟；实验仅限于桌面双臂场景，未验证移动或人形平台；事件发现仍为离线，缺乏在线计划修正。

---

## 181. Extending FunctionGemma for Practical On-Device Mobile Function Calling

**arXiv ID:** 2609.25373 | [PDF](https://arxiv.org/pdf/2609.25373v1)

**作者:** Ali Rezagholizadeh `[一作]` (UGrowAI), Soheila Samiee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 270M 的 FunctionGemma 模型上进行针对 Android 设备控制的功能调用 fine‑tuning，并发布了一个 9,500 条对话的合成数据集和一个 Android demo。

**💡 创新点**

① 发布了全新的 MOBILEACTIONSEXTENDED 数据集，覆盖 15 个手机控制类别；② 用 TRL 完全监督 fine‑tuning 在 270M 模型上实现 76.5% 的端到端准确率；③ 通过合并 Google Mobile Actions 数据构建了兼顾 22 类动作的多域模型，兼顾 82.3% 的性能。

**🔧 技术方法**

使用 FunctionGemma 270M-it 背骨模型、TRL 完全监督 fine‑tuning（completion‑only loss）、schema‑grounded 合成生成、函数调用 sentinel‑token 格式、INT8 量化以及 LiteRT‑LM 运行时。

**📊 数据集**

MOBILEACTIONSEXTENDED（约 9,500 例）以及 Google Mobile Actions 7 类数据，合并后得到的 22 类数据用于跨域训练。

**📈 对比分析**

在严格 exact‑match 评估下，与基线 FunctionGemma（29.3%）和 Google Mobile Actions（17.2%）相比，extended 模型提升到 76.5%；combined 模型在原始 7 类保持 82.3% 的同时覆盖更多类别；执行‑aware 评估进一步提升至 96%/82.6%。

**⚠️ 局限性**

数据为合成而非真实用户交互，严格匹配评估可能低估可执行成功率；缺乏多轮对话、多语言、iOS 支持，仅在 270M Gemma 基础上验证；未包含安全/权限检查的完整评估。

---

## 182. Observer Choice and Threshold Selection in Retinal Vessel Segmentation: A Subject-Separated Evaluation

**arXiv ID:** 2609.25597 | [PDF](https://arxiv.org/pdf/2609.25597v1)

**作者:** Wenhao Xu `[一作]` (Zhengzhou Police University), Rongtao Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对视网膜血管分割，作者通过固定的七折交叉验证，系统评估了在两位标注者之间选择阈值（最大最小阈值、均值阈值、单一标注者阈值以及固定阈值）对模型性能的影响。

**💡 创新点**

创新点在于明确区分阈值选择所依据的参考标注与最终评估所用的参考标注，揭示了阈值优化并不一定能提升未知样本上的最低观测者Dice分数，并提出了在多标注者环境下阈值选择的透明报告规范。

**🔧 技术方法**

使用的技术包括：基于29维像素特征的随机森林（RF）和极度随机树（ET）分类器，91个阈值网格搜索，Dice、IoU、AP、ROC AUC等评估指标，以及10,000次Bootstrap自助法构建置信区间。

**📊 数据集**

数据集为公开的CHASE_DB1，包含28张图像（14名受试者的双眼），每张图像都有两位独立标注者的血管分割注释。

**📈 对比分析**

比较方法是针对每个模型训练一次，在验证集上根据不同阈值策略选取阈值后，在测试集上计算性能；主要结果显示最大最小阈值在42个模型中并未显著提升最低观测者Dice（平均变动约为-0.5%至-0.8%），而固定阈值0.5相对更优；不同阈值策略对Precision/Recall的影响有限。

**⚠️ 局限性**

局限性包括：仅使用两种树模型且未尝试深度学习方法；验证集仅有4张图像，阈值优化可能过拟合；所有模型均以标注者1为训练目标，未对称实验；缺乏外部独立验证和临床意义评估；Bootstrap区间仅反映在已训练模型下的变异，未考虑跨数据集的泛化。

---

## 183. Passes Alone, Fails Together: Benchmarking Semantic Coordination in Parallel LLM-Agent Development

**arXiv ID:** 2609.25396 | [PDF](https://arxiv.org/pdf/2609.25396v1)

**作者:** Haocheng Xia `[一作]` (University of Illinois Urbana-Champaign), Yongjoo Park `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了名为 "stale" 的基准，用于检测并行 LLM 代理在合并补丁时因语义不一致导致的失败，并通过干涉度量对比单独补丁与合并补丁的行为。

**💡 创新点**

创新点在于定义语义协调头room概念，设计可重复的评测流程，并构建三层（合成、挖掘、构造）基准，量化未同步并发改动导致的错误。

**🔧 技术方法**

使用了 GPT‑5/5.5 等大型语言模型与工具调用框架 mini‑swe‑agent、OpenHands，结合 pytest 单元测试、Oracle 消息生成与合并解析等技术。

**📊 数据集**

数据集包括从 Django、SymPy、xarray、seaborn 中挖掘的 417 对有效 PR 对；另外创建了 36 个基于 12 个真实 Django helper 的构造任务以及 96 个合成任务。

**📈 对比分析**

通过对比 solo、blind（盲合并）与 informed（oracle）三种条件，并用 Δ_blind 计量干涉，实验发现在真实挖掘的 PR 中干涉率仅 0.24%（1/417），但在合成与构造任务中干涉率高达 97%；Oracle 消息可将干涉率降低至 82%/93%。

**⚠️ 局限性**

局限性包括：基准仅评估两任务并行场景，Oracle 消息假设完美信息且未测试实时通信；挖掘的 PR 已通过审查，可能低估实际并发冲突频率；测试覆盖率有限，可能漏报错误；结果主要适用于 Django，其他语言和框架需进一步验证。

---

## 184. Identifying Suspected Mislabeled Apps in Google Play Application Removal Prediction: An Empirical Comparison of Label Noise Detection Methods

**arXiv ID:** 2609.25487 | [PDF](https://arxiv.org/pdf/2609.25487v1)

**作者:** Deborah Dobles Montalvan `[一作]` (University of Groningen), H. de Weerd `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对Google Play应用下架数据中的标签噪声问题，应用三种标签噪声检测方法对可能被错误标记的应用进行识别，并评估去除或重新标记这些应用对下架预测模型性能的影响。

**💡 创新点**

创新点在于首次将孤立森林、邻域不一致和预测不一致三类噪声检测器系统地应用于完整的Google Play下架数据集，验证噪声检测的有效性，并深入分析噪声的方向性和外部验证结果。

**🔧 技术方法**

使用的技术包括孤立森林（Isolation Forest）异常检测、邻域一致性判定（Neighborhood Disagreement）以及基于逻辑回归的预测不一致检测（Prediction Inconsistency），并在此基础上构建XGBoost集成模型进行下架预测。

**📊 数据集**

采用了<cit>公开的870,514个Google Play应用的用户侧元数据集，包含47个特征，二分类标签表示应用在两次观察间是否被移除。

**📈 对比分析**

与基线XGBoost模型相比，去除噪声检测标记的样本或对重标记几乎不提升AUC，部分检测器（如预测不一致）甚至导致AUC下降；但对噪声标记子集训练模型的AUC低于0.5，证明这些样本的特征与标签关系与整体分布相反。

**⚠️ 局限性**

局限性包括：标签噪声的比例极低（<1%），导致去除噪声样本对整体性能影响有限；模型评估使用未清洗的测试集，仍包含噪声；外部验证依赖VirusTotal/Quark Engine，存在误判；检测器对参数敏感且不保证最佳阈值；仅使用元数据，缺乏代码行为特征，可能限制噪声检测与模型提升的潜力。

---

## 185. Accelerating the Mitigation of LLM Inference Nondeterminism Across GPU Architectures

**arXiv ID:** 2609.25624 | [PDF](https://arxiv.org/pdf/2609.25624v1)

**作者:** Liam Cooper `[一作]` (Georgia Institute of Technology), Hyesoon Kim `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套固定配置的融合上采样 GEMM 内核，使得跨不同 GPU 架构的 LLM 线性层输出完全位一致。

**💡 创新点**

创新点在于：只使用 IEEE‑754 FMA、去除自适应调优、确定性 split‑K、批量不变性，将浮点累加顺序纯函数化，从而消除跨架构差异并显著降低权重量传输。

**🔧 技术方法**

使用技术包括 Triton 生成的固定配置融合上采样 GEMM、BF16 存储 + FP32 upcast、FP32 IEEE‑754 FMA、确定性 split‑K、批量不变内核，并集成至 vLLM 推理框架。

**📊 数据集**

数据集与模型：Llama‑3.2‑3B‑Instruct、Qwen3‑4B‑Instruct‑2507、DeepSeek‑R1‑Distill‑Llama‑8B；基准为 GSM8K、MATH500、AIME24、GPQA‑Diamond。

**📈 对比分析**

通过与未修正 BF16 以及仅采用 FP32 计算的基线在 A100、L40S、H100 上进行交叉架构、交叉种子误差比较，结果显示跨架构误差从 30–100% 降至 0–0.51%，同时速度提升 1.17–3.1 倍，权重量占比减少 1–1.4 GiB。

**⚠️ 局限性**

局限性：仅保证线性层的位一致性，注意力、归一化、采样等非 GEMM 组件仍可能产生非确定性；尚未验证跨厂商硬件（AMD、TPU 等）的兼容性；对极低精度或特殊优化的兼容性未完全覆盖。

---

## 186. Causal graph rewriting

**arXiv ID:** 2609.25188 | [PDF](https://arxiv.org/pdf/2609.25188v1)

**作者:** Pablo Arrighi `[一作]` (Université Paris-Saclay), Luidnel Maignan `[通讯]` (University Paris Est Creteil)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于邻域方案和局部规则的图重写框架，并通过粒子系统实现了对图结构的局部变换。

**💡 创新点**

创新点在于引入可扩展的邻域方案和严格的局部规则约束，确保局部变换在全局保持一致且可归约。

**🔧 技术方法**

主要技术包括图论、邻域方案定义、局部规则设计、可扩展性与稳定性证明以及粒子系统的实现。

**📊 数据集**

本研究主要为理论性工作，未使用具体数据集，而是以抽象图模型为基础进行实验验证。

**📈 对比分析**

通过理论证明与示例图案例比较，展示了所提出方法在保持图连通性、边界完整性和无环性方面的优越性，但未给出数值性能指标。

**⚠️ 局限性**

局限性包括对图规模的依赖、对大规模图的计算复杂度未明确定义以及对非可扩展邻域方案的适用性有限。

---

## 187. Formal verification of tilt estimation using the Rocq prover

**arXiv ID:** 2609.25561 | [PDF](https://arxiv.org/pdf/2609.25561v1)

**作者:** Reynald Affeldt `[一作]` (National Institute of Advanced Industrial Science and Technology), Holger Thies `[通讯]` (Kyoto University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

在Rocq证明助手中构建了一套完整的数学工具库，用于形式化验证人形机器人倾斜估计的稳定性和收敛性；

**💡 创新点**

创新之处在于首次在Rocq中实现了经典Cauchy‑Lipschitz定理的全局版本、连续初值依赖性，以及Lyapunov和LaSalle稳定性理论，并通过商类型技术完成对ODE解空间的完整构造；

**🔧 技术方法**

主要技术包括Rocq与MathComp库的深度整合、商类型构造、Banach不动点定理、积分与微分的形式化、以及对机器人动力学的时间变矩阵与刚体坐标系的推导；

**📊 数据集**

本工作不依赖外部数据集，所有结果均为形式化证明；

**📈 对比分析**

由于本研究聚焦形式化验证，未进行实验对比，性能指标以证明的可复制性与逻辑完整性为主；

**⚠️ 局限性**

局限性在于仅覆盖连续时间系统的稳定性分析，尚缺乏指数稳定性、线性化理论与不稳定流形等更细粒度的稳定性分析工具，且目前仅针对单一倾斜估计器模型。

---

## 188. Learned Enterprise Data Comprehension: Compression and Routing for Data Agents

**arXiv ID:** 2609.25286 | [PDF](https://arxiv.org/pdf/2609.25286v1)

**作者:** Ethan Torres `[一作]` (permute.ai), Eric Mills `[通讯]` (permute.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“潜在等价学习”（latent equivalence learning）框架，将持久的任务相关身份与数据集特定的实现分离，并通过支持/对抗高斯原型与查询原型兼容映射实现身份因子化路由，使数据代理能够在查询前就组织好证据；

**💡 创新点**

创新点在于：①将持久身份与其在不同数据集中的实现分离，②使用支持/对抗权重构造高斯原型并引入软成员分布；③设计查询原型和兼容矩阵实现查询到身份的路由；④给出局部赋值稳定性证书，证明在原型不变时硬分配保持不变；

**🔧 技术方法**

技术手段包括高斯原型学习（Gaussian Prototypical Networks）、支持/对抗损失、查询原型编码器与兼容矩阵、软成员分布、局部赋值稳定性分析、数据代理框架与Arbiter；

**📊 数据集**

使用了Data Agent Benchmark（DAB）12个异构数据库，共54个自然语言查询；

**📈 对比分析**

在DAB基准上与Claude Opus 4.6基线及同一agent框架下无学习表示的Core进行比较，取得94.67% dataset‑macro Pass@1（比Core高10.54pp、比基线高39.16pp），raw query准确率为258/270（≈95.56%），显著优于对比系统；

**⚠️ 局限性**

局限性：需要预先定义有限的身份库存，依赖支持/对抗证据质量；局部稳定性仅在原型保持不变时成立，原型大幅更改需重新控制；对未覆盖身份或缺失关键证据的查询效果有限；系统为专有且代码未公开，计算资源有限。

---

## 189. Exposing Blind Spots in Deep Imbalanced Regression Evaluation

**arXiv ID:** 2609.25152 | [PDF](https://arxiv.org/pdf/2609.25152v1)

**作者:** Noah C. Puetz `[一作]` (TH Koen), Thomas Bartz-Beielstein `[通讯]` (TH Koen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在深度不平衡回归（DIR）评估中识别并解决三大盲点：数据域、指标与稳定性；提出 MuViS-DIR 多模态时间序列基准、平衡 MAE 与 bMASE 指标，并对六种 DIR 方法进行重复实验，评估其在平衡指标与种子稳定性上的表现。

**💡 创新点**

创新点在于：①将 DIR 评估扩展至非图像领域，构建真实可操作尾部场景的 MuViS-DIR 基准；②引入可跨数据集、跨方法的尺度归一化平衡指标 bMASE，解决以许/中/少-shot 评分的主观性与不可比性；③系统分析尾部区域对随机种子敏感性的稳定性缺陷，首次量化尾部不稳定性。

**🔧 技术方法**

使用平衡 MAE 与 bMASE 评估指标、ResNet1D 基线、六种 DIR 方法（Vanilla、ConR、Focal-L1、LDS、RnC、SQInv、UVote）以及对十次随机种子重复实验的标准差分析。

**📊 数据集**

MuViS-DIR benchmark（9 个多模态时间序列回归任务，涵盖 Battery SoC、Chem. Conc.、Heart Rate、Monterey、PM10、PM2.5、Targa'13/14、Tire Temp 等六大物理域）。

**📈 对比分析**

通过 bMASE（以及 bMAE）对比方法，UVote 在 7/9 任务中表现最佳，GMean bMASE 从 0.2163 降至 0.1732；大部分 DIR 方法在平衡指标上提升明显，但在时间序列多模态任务中的效果不均；尾部区域的种子标准差随样本量下降显著增加，表明尾部性能的不稳定性。

**⚠️ 局限性**

局限性：①基准仅覆盖虚拟感知的多模态时间序列回归，未扩展至预测或序列到序列任务；②评估方法有限，未涵盖所有最新 DIR 方案；③bMAE/bMASE 依赖离散化，需进一步探讨连续指标替代；④未提出针对尾部稳定性的优化方法，仅做诊断。

---

## 190. ChainDoRA: Tensor-Train Factorized Weight-Decomposed Low-Rank Adaptation for Parameter-Efficient LLM Fine-Tuning

**arXiv ID:** 2609.25058 | [PDF](https://arxiv.org/pdf/2609.25058v1)

**作者:** Ashfak Yeafi `[一作]` (Khulna University of Engineering & Technology), Md Khairul Islam `[通讯]` (Hobart and William Smith Colleges)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ChainDoRA，一种将 DoRA 的方向低秩因子替换为连接的 Tensor‑Train（TT）表示的参数高效微调框架。

**💡 创新点**

创新点在于将方向因子从稠密低秩矩阵改为连接的 TT 链，并将适配器秩 r 与内部 TT 秩 ρ 分离，实现显著参数压缩同时保持或提升性能。

**🔧 技术方法**

使用 Tensor‑Train 结构、行归一化的 magnitude‑direction 重参数化、非零 TT 初始化与 gauge‑balance、响应‑only 训练目标等技术。

**📊 数据集**

在 LLaMA‑7B 预训练模型上，用 15,119 条常识推理适配示例训练，并在七个常识基准（BoolQ、PIQA、SocialIQA、WinoGrande、ARC‑Easy、ARC‑Challenge、OpenBookQA）评估。

**📈 对比分析**

与 LoRA、DoRA 在相同适配器秩（32）、相同训练设置下对比，ChainDoRA 在七项平均准确率 72.30% 超过 LoRA（69.88%）和 DoRA（69.39%），同时参数量仅 5.35M，约为 DoRA 的 10 倍压缩。

**⚠️ 局限性**

局限性包括：仅在单一 LLaMA‑7B 与小规模适配数据集验证；跨模型性能未训练；对训练效率与超参数调优的影响未深入研究；更大数据集与多任务泛化仍需进一步验证。

---

## 191. Robust Failure, Conservative Repair: Textual Knowledge Distillation from Cross-Model Failures

**arXiv ID:** 2609.25400 | [PDF](https://arxiv.org/pdf/2609.25400v1)

**作者:** Andrew Ren `[一作]` (University of Chicago), Chenhao Tan `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种稳健的失败驱动文本知识蒸馏方法 RFCR，通过共享错误构建规则原子，并在推理时路由应用，实现对作弊表的增量改进。

**💡 创新点**

提出激活精度与共享错误筛选的保守接受门，既能从源模型失败中提取可迁移知识，又能防止模型特定补丁对目标模型产生负面影响。

**🔧 技术方法**

采用规则原子生成、激活与边界谓词、保守验证门、路由推理、Wilson 置信区间分析以及基于 BIG-Bench Hard 的作弊表蒸馏等技术。

**📊 数据集**

使用 BIG-Bench Hard 的 400 题主实验集、295 项官方持出集以及 18 项补充任务集合进行评估。

**📈 对比分析**

与原始作弊表、ProTeGi、GEPA 等提示优化基线对比，RFCR 在 400 题上提升 2.75pp（71.25%）且无回退；跨模型测试显示部分模型获得正增益，但仍存在有限的性能下降。

**⚠️ 局限性**

改进有限且集中于几类任务；跨模型可迁移性受限；方法依赖源模型失败的共享程度；评估基准为控制开发结果，未进行完整独立测试。

---

## 192. MedGate-Fusion: Integrating First-Encounter Semantic Narratives and Physiological Biomarkers for Prospective Stroke Risk Stratification

**arXiv ID:** 2609.25272 | [PDF](https://arxiv.org/pdf/2609.25272v1)

**作者:** Hemn Khdr `[一作]` (University of Toronto), Zahra Shakeri `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过构建以第一诊访为基准的前瞻性队列，利用多模态门控架构预测五年卒中风险。

**💡 创新点**

创新点在于结合 transformer 生成的诊断文本语义嵌入与十项常规生理指标，并引入泄漏抑制与动态门控融合，实现更精准的风险估计。

**🔧 技术方法**

采用 MiniLM-L6 transformer 进行文本语义编码，梯度提升树处理数值指标，门控层动态加权两分支，并使用加权二元交叉熵处理类别不平衡。

**📊 数据集**

数据来源为加拿大初级护理哨兵监测网络（CPCSSN）共 808,921 条记录，筛选后得到 102,736 名患者的第一诊访数据。

**📈 对比分析**

与传统 Logistic 回归、HGBM 以及单一文本模型相比，MedGate‑Fusion 在 5 折交叉验证中 AUROC 0.848、AUPRC 0.346，显著优于单模态模型（提升约 14.5%）。

**⚠️ 局限性**

局限性包括缺失值、文档编码差异导致噪声、仅来自加拿大的数据、仅限五年预测窗口以及 transformer 计算成本较高。

---

## 193. Controller-Only False Confirmation in Passive RF UAV Link Detection

**arXiv ID:** 2609.25294 | [PDF](https://arxiv.org/pdf/2609.25294v1)

**作者:** Rajendra Upadhyay `[一作]` (George Mason University), Duminda Wijesekera `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

进行了双频段软件定义无线电(SDR)测量实验，评估无人机控制器在无机架时的误报，并提出控制器仅误报率约束的决策框架。

**💡 创新点**

首次公开包含三款商用无人机在无机架、控制器仅、以及联动状态下的完整RF数据集，且将控制器仅误报单独衡量并引入误报率约束以控制误报。

**🔧 技术方法**

采用USRP B210双频段扫描、能量特征提取与对数似然比判别，结合留一轮交叉验证、阈值约束和紧凑排名扫描等技术。

**📊 数据集**

主数据集为20轮、7个RF状态（ambient、controller-only、linked）共2800条IQ捕获；附加10轮手机连线Phantom数据以及5轮硬件回路计时实验。

**📈 对比分析**

通过留一轮交叉验证对比普通阈值与误报率约束阈值，普通阈值下联动TPR 0.90、误报率 0.35；约束后误报率降至 0.05 但TPR 降至 0.40；Hubsan/Mavic TP可达 1.0 而 Phantom 难以分离；硬件回路实验显示紧凑扫描时间降低 65%。

**⚠️ 局限性**

仅评估三款商用无人机、室内近距离测试，缺乏户外、多频段和更多平台的验证；误报率约束导致高退回率，需进一步扩展与验证。

---

## 194. Reasoning-Preserving Fine-Tuning of Post-RL LLMs with Null-Basis LoRA

**arXiv ID:** 2609.25618 | [PDF](https://arxiv.org/pdf/2609.25618v1)

**作者:** Wenzhi Fang `[一作]` (Samsung Research America), Srinivas Chappidi `[通讯]` (Samsung Research America)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Null-Basis LoRA（NB-LoRA）方法，用于在 RL 后训练的 LLM 上进行参数高效微调，同时保持推理能力。

**💡 创新点**

创新点在于将推理隐藏状态的近似零空间直接嵌入 LoRA 的低秩重参数化，构造出结构化的推理保持约束，无需回放或梯度投影。

**🔧 技术方法**

采用 LoRA、零空间分析、SVD 主成分、低秩因子重参数化及基于主角角度的相似度评估等技术。

**📊 数据集**

使用 Qwen2.5-3B 与 Qwen2.5-1.5B 在 MATH‑lightEval、GSM8K 训练得到的 RL 模型，并在 ToolUse 与 Commonsense Reasoning（ARC‑C/E、BoolQ、HellaSwag、OpenBookQA、PIQA、SocialIQA、WinoGrande）等任务上进行微调；评估包括 MATH‑500、AGIEval‑EN‑MATH、MMLU‑STEM 等推理基准。

**📈 对比分析**

与 Vanilla LoRA、LoRA‑Null、MiLoRA、NSCL、经验回放和 SDFT 等基线比较，NB-LoRA 在目标任务上保持与 Vanilla LoRA 相当的准确率，同时在推理任务上几乎不下降；在效率上比 NSCL 速度快 6×、显存占用少 2×，参数量也更小。

**⚠️ 局限性**

局限性包括需要先收集推理数据以估计零空间，且对低维子空间假设敏感；对超参数（能量阈值、样本量）较为依赖；目前仅在少数 RL 训练模型和线性层上验证，尚未在更大规模或非线性模块上检验。

---

## 195. Interpretable AI plus Handheld, Portable Retinal Photographs: A Low-Cost Glaucoma Screening Solution for West Africa

**arXiv ID:** 2609.25697 | [PDF](https://arxiv.org/pdf/2609.25697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 196. RAG-NAROK: Retrieval-Aware Knowledge Corpus Poisoning in RAG with Source-specific Refutation

**arXiv ID:** 2609.25469 | [PDF](https://arxiv.org/pdf/2609.25469v1)

**作者:** Abdullahil Kafi `[一作]` (Southern Illinois University), Alvi Ataur Khalil `[通讯]` (Southern Illinois University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于检索上下文自适应的黑盒知识库污染框架，用于破坏RAG问答系统的生成质量。

**💡 创新点**

创新点在于先通过检索指纹化推断检索器类型，再在阴影检索中提取竞争性上下文，利用风格模仿与权威框架生成与合法文档相似且能否认已检索事实的攻击文档。

**🔧 技术方法**

采用检索指纹化、Shadow Retrieval、Stylometric Profiling、基于LLM的生成式对抗文档合成以及迭代验证与异常检测逃逸等技术。

**📊 数据集**

实验使用了法律（CFR-21）、网络安全（Cybersec-IT）和金融（Feds-2026）三大领域的专业语料库。

**📈 对比分析**

通过与基线直写对抗文档对比，评估检索主导度、异常检测逃逸率和生成影响，结果在三域上均实现最高检索排名、最高达88.37%的逃逸率以及显著高于基线的生成影响。

**⚠️ 局限性**

局限性包括依赖手工阈值的检索指纹化、计算成本高、假设源透明且不适用于无源信息的RAG、以及LLM-判定可能存在偏差。

---

## 197. Learning Neural Feedback Linearization for Data-driven Systems via Augmented Lagrangian

**arXiv ID:** 2609.25163 | [PDF](https://arxiv.org/pdf/2609.25163v1)

**作者:** Lakshmi Priya P. K. `[一作]` (South Westphalia University of Applied Sciences), Andreas Schwung `[通讯]` (South Westphalia University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于多层感知器（MLP）和增广拉格朗日方法的完全数据驱动反馈线性化框架，能够在不依赖解析模型的前提下实现系统的相对度约束、神经Lie导数控制和闭环稳定性分析。

**💡 创新点**

创新点在于：①将相对度约束直接嵌入学习目标中；②利用增广拉格朗日实现约束满足的梯度优化；③给出识别误差与跟踪误差之间的上界，实现闭环的实用稳定性证明。

**🔧 技术方法**

使用的技术包括：多层感知器（MLP）网络；神经Lie导数自动微分；增广拉格朗日（Augmented‑Lagrangian）约束优化；PID/积分补偿的反馈线性化控制；理论分析基于统一函数逼近与实际稳定性。

**📊 数据集**

数据集来源于仿真产生的含噪声的DC电机（电枢控制）数据，包含 2000 条状态-控制-导数-输出样本，采样间隔 0.01 s，时长 5 s，加入 σ=0.2 的高斯噪声。

**📈 对比分析**

与传统基于模型的反馈线性化或无约束神经网络相比，本文方法在闭环跟踪误差、控制输入幅值与残差误差上均保持在可接受范围（跟踪误差均值 ≈ 0.02，残差上界 ≈ 1.04），且控制输入保持在安全范围内，展示了在噪声条件下更好的鲁棒性。

**⚠️ 局限性**

局限性包括：①需要先验知道系统的相对度且保持不变；②增广拉格朗日求解对超参数和初始多项式选择敏感；③仅在仿真数据上验证，缺乏实验硬件验证；④对大规模多输入多输出系统的扩展尚未讨论。

---

## 198. Recovering Agentic Sovereignty: Mitigating the Consensus Paradox via Contrastive Epistemic Decoding

**arXiv ID:** 2609.25570 | [PDF](https://arxiv.org/pdf/2609.25570v1)

**作者:** Dahlia Shehata `[一作]` (University of Waterloo), Ming Li `[通讯]` (University of Waterloo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Contrastive Epistemic Decoding (CED)，一种零样本推理时的对抗解码方法，用来抑制大语言模型的群体同调与毒性一致性。

**💡 创新点**

创新点在于：① 采用双前向推理获取标准与高压同调下的对数概率差；② 用非线性零边界概率 clamp 代替传统线性减法；③ 结合离散 top‑k 截断掩码，避免语法崩溃；④ 在无训练、跨架构的零样本设置下实现同调抑制。

**🔧 技术方法**

技术包括对比解码、对数概率差计算、概率阈值 Clamp、top‑k 截断掩码、状态同步、零样本推理、LLM‑as‑Judge 评估以及对数概率级别的双通道推理。

**📊 数据集**

使用 GAIA、SWE‑bench 和 Multi‑Challenge 三个公开基准（各 100 条测试样本，共 300 条），以及基于这些样本生成的 7,200 条对比推理轨迹。

**📈 对比分析**

方法与标准解码进行对比，指标为认知偷懒率（Loafing Rate）与准确率（Accuracy）。在 Gemma‑2 与 Llama‑3.1 上，认知偷懒下降最高 33%，准确率提升最高 30.75%；Mistral v0.3 仅表现为行为改进，准确率提升有限。

**⚠️ 局限性**

局限性：① 需要访问预软化对数概率，限制在开源权重模型；② 双前向推理导致两倍推理延迟；③ 对能力受限模型（如 Mistral v0.3）无法提升推理能力，只能抑制同调；④ 对真实动态多智能体系统的验证尚未完成。

---

## 199. An Exploratory Replica-Overlap Probe of the Grokking Transition

**arXiv ID:** 2609.25634 | [PDF](https://arxiv.org/pdf/2609.25634v1)

**作者:** A. C. Opus `[一作]` (University of Puerto Rico), J. Q. Lu `[通讯]` (University of Puerto Rico)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 2026 年对 113 取模的模组算术加法任务进行预注册实验，训练 64 个独立的网络，并探索复制对称破缺（RSB）在 grokking 过渡中的表现。

**💡 创新点**

首次将自旋玻璃中的 Parisi 复制对称破缺框架直接应用于深度学习的 grokking 现象，并以多复制重叠分布 P(q) 为潜在阶参数进行检验。

**🔧 技术方法**

使用多复制权重重叠统计、Git Re‑Basin 权重对齐、Hartigan Dip 检验、以及最小化交叉熵的集合大小测量等技术。

**📊 数据集**

采用了 113 取模的模组算术加法任务（p=113）与预先固定的 B2 结构，在四种训练比例与分割配置下训练网络。

**📈 对比分析**

比较了记忆化平稳期与 grokking 后的重叠分布，发现重叠标准差增加约 5.6 倍，但 Dip 统计无显著变化；集合大小实验显示在 1% 损失阈值下最小集合尺寸为 1，表明提升有限。

**⚠️ 局限性**

主要局限包括：对齐实现未保持网络功能导致仪器失效；数值精度门未通过；缺乏预分析功效校准；训练比例与分割身份完全混淆；Dip 统计仅检验单峰性，未给出等价边界；实验仅针对单一任务和架构。

---

## 200. Understanding Reliability in LLM-based Human Behavior Simulation

**arXiv ID:** 2609.25066 | [PDF](https://arxiv.org/pdf/2609.25066v1)

**作者:** Pei Wang `[一作]` (Renmin University of China), Xu Chen `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了ReliMap框架，将LLM驱动的人类行为模拟拆分为三层（基线分布、个人信息条件化、人口聚合），并在模型容量、个人信息完整度、人口覆盖度三维配置下，同时评估个体层（R1）和群体层（R2）的可靠性；

**💡 创新点**

创新点在于：①以分层结构系统化分析模拟过程；②引入三维可配置空间与双层评估；③揭示个体准确度提升不必然带来群体分布改进，强调多层协同优化的必要性；

**🔧 技术方法**

技术手段包括：大型语言模型（如GPT‑4.1、Gemini‑2.5‑Pro、Qwen3系列、DeepSeek系列）在无标签和带标签情境下的推理；统计评估指标ACC与TVD；信息论度量（ASPG、Gini）用于衡量属性预测价值；

**📊 数据集**

使用四个问卷/社会调查数据集：欧洲社会调查（Party）、世界价值观调查（Immigration）、SocioBench（Religion）以及中文媒体事件数据集（Media）；

**📈 对比分析**

比较方法：在每个任务上对11种LLM在不同配置（模型规模、个人信息覆盖率、样本覆盖度）下的R1和R2进行实验，发现：①大模型在丰富个人信息时收益更大；②属性信息高度集中，选取高信息属性能显著提升准确率；③R1提升并不总导致R2改善，且人口覆盖度可降低方差但无法消除系统性偏差；

**⚠️ 局限性**

局限性包括：仅聚焦问卷式任务，未覆盖交互式对话或多主体情景；使用准确率评估个体可靠性，忽略主观性；分析为观测性，未揭示机制；预训练数据泄漏可能影响基线层结果，但对比结论不受影响。

---

## 201. Spreading Factor Assignment Strategy for Coverage and Capacity Flexible Tradeoff

**arXiv ID:** 2609.25428 | [PDF](https://arxiv.org/pdf/2609.25428v1)

**作者:** Luiz Filho `[一作]` (Federal University of Juiz de Fora), Níbia Bezerra `[通讯]` (Luleå University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并比较多种 LoRaWAN 扩频因子 (SF) 分配策略，分析覆盖-容量权衡，并提供一个可开源的 ns‑3 仿真框架。

**💡 创新点**

① 用简单可配置的分配向量 a 设计可提升容量或覆盖的通用 SF 分配策略；② 发布开源 ns‑3 仿真框架，方便研究者快速验证和改进 SF 分配算法。

**🔧 技术方法**

LoRa 物理层（CSS 调制）、LoRaWAN MAC、ADR 算法、随机/等分/容量/覆盖/感知等 SF 分配方法，以及 ns‑3 3.29 版本的 LoRa 模块。

**📊 数据集**

无真实数据集，使用仿真参数：设备数 1k–6k，半径 3–10 km，包大小 23 B，125 kHz 带宽，路径损耗模型 (PL(d₀)=7.7，n=3.7)，10 次仿真运行。

**📈 对比分析**

通过比较 9 种 SF 分配策略的包接收率 (PDR) 与吞吐量，发现：① 在大规模或大半径场景下，等分 (III) 与随机 (IX) 方法优于 LoRaWAN ADR；② 仅关注覆盖或容量的单一策略效果不佳；③ 通过适当的分配向量可显著提升网络性能。

**⚠️ 局限性**

局限性：ns‑3 LoRaWAN 模块不完整，缺失 MAC 信令和 LinkADRReq；SF 之间缺乏严格正交；仿真采用简化的路径损耗模型，未考虑多径、移动性和更复杂的干扰环境。

---

## 202. FASTAR: FRI Accelerator for Scalable Transparent ARguments of Knowledge

**arXiv ID:** 2609.25535 | [PDF](https://arxiv.org/pdf/2609.25535v1)

**作者:** Tengkai Gong `[一作]` (Northeastern University), Xiaolin Xu `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

设计并实现了一个面向 FRI 协议的 FPGA 加速器，能够根据目标板的资源自动生成最优硬件配置，支持从数据中心级到嵌入式边缘级的多种 FPGA 平台；

**💡 创新点**

创新点在于提出一种约束驱动的自动化硬件生成流程，利用参数化的 NTT、拆分折叠和 Merkle 树模块，在保持高吞吐量的同时可在不同资源限制下灵活调优；

**🔧 技术方法**

采用了高层合成（HLS）实现的 4 步 NTT、批量转置调度、分块折叠流水线以及基于 Vitis Security Library 的可伸缩哈希引擎，并通过多通道内存访问和逆偏移表等技术优化内存访问；

**📊 数据集**

使用的实验数据集为 FRI 域大小 |𝒟|=2^17，折叠因子 F=2，测试字段分别为 128 位和 254 位的有限域；

**📈 对比分析**

与最优软件基线（Winterfell）、NVIDIA RTX 3070 上的 Icicle 以及 A40 上的 Air‑FRI 进行对比，数据中心级 FPGA 在 𝔽ₚ₁₂₈ 下实现 25.7× CPU 加速、3.5× GPU 加速；主流 FPGA 取得 10.4× CPU 加速、1.31× GPU 加速；边缘 FPGA 则取得 3.2× CPU 加速、0.27× GPU 加速；

**⚠️ 局限性**

局限性包括：在边缘设备上受限的 DDR 通道导致内存瓶颈，较大字段（𝔽ₚ₂₅₄）显著增加 DSP 需求，整体吞吐量受限于单通道内存带宽，且对高阶多通道内存的依赖限制了在某些低功耗 FPGA 的性能上限。

---

## 203. TRACTOR Benchmark for Evaluating C to Rust Translators

**arXiv ID:** 2609.25121 | [PDF](https://arxiv.org/pdf/2609.25121v1)

**作者:** Hamed Okhravi `[一作]`, Nathan Burow `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了 TRACTOR C‑to‑Rust 评测基准和完整的自动化评估基础设施，涵盖功能正确性、安全性、习惯性和性能四大维度。

**💡 创新点**

创新之处在于将多维度评估指标标准化、引入可量化的安全度度量（unsafe 操作计数、UB 识别）和半自动化的 idiomaticity 评审框架，解决传统仅靠语法转换评估的局限。

**🔧 技术方法**

利用 LLVM/Clang、Rust 编译器、Clippy、Cyclomatic Complexity 计算器、unsafety 监测脚本，以及 AWS CDK/Fargate/Docker 容器化部署和 LLM 调用等技术实现全流程自动化。

**📊 数据集**

使用公开的测试电池（Battery 01、Battery 02）和里程碑项目（Project 00‑02）构成的 C 代码集，包含源代码、构建脚本、测试向量和期望输出，全部托管于 GitHub 公共仓库。

**📈 对比分析**

通过自动化运行器对比功能正确性（I/O/返回值一致性）、安全性（unsafe 使用量、UB 处理方式）、习惯性（Clippy lint 计数+手工 Likert 评分）以及性能（翻译吞吐量、运行时间、内存占用）等指标；框架支持可视化和可重复的基准对比。

**⚠️ 局限性**

局限性包括对 UB 的自动检测仍不完善、并发/内联汇编等高级 C 特性未被充分覆盖、习惯性评审存在主观偏差、性能测量受云环境波动影响，并且目前仅聚焦于 C→Rust 的迁移场景。

---

## 204. Rethinking Backdoor Repair Evaluation: Distinguishing Aggregate Clean Utility from Benign Performance Preservation

**arXiv ID:** 2609.25579 | [PDF](https://arxiv.org/pdf/2609.25579v1)

**作者:** Baogang Song `[一作]` (Wuhan University of Technology), Dongdong Zhao `[通讯]` (Wuhan University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文重新审视后训练的后门修复评估，提出以类为单位的保真度损失（preservation loss）来衡量修复后模型在各类别上保持原有干净性能的程度。

**💡 创新点**

创新点在于将整体干净准确率（Overall Clean Accuracy）与类级保真度损失区分开来，定义了 Worst‑Class Preservation Loss 与 Tail Preservation Loss 两个紧凑指标，并系统验证它们能揭示聚合指标掩盖的局部性能退化。

**🔧 技术方法**

主要技术包括：1) 类别级性能变化定义（R_c）与保真度损失定义（D_c）；2) 通过 D_max 和 D_tail(α) 对保真度损失进行总结；3) 在多种攻击与修复策略下统一的代表性修复状态选择；4) 采用宏观清洁准确率（Macro Clean Accuracy）验证类权重不影响结论。

**📊 数据集**

使用的公开数据集包括 CIFAR‑10、CIFAR‑100 和 GTSRB；模型架构覆盖 ResNet18、PreActResNet18 与 VGG19‑BN。

**📈 对比分析**

实验比较了 6 种后门修复方法（FT、FP、NAD、ANP、I‑BAU、D3）以及清标攻击下的 FST、RNP。结果显示：1) 大部分修复在 ASR 低于 10% 并且整体干净准确率提升或不变；2) 然而 Worst‑Class Preservation Loss 可高达 52 个百分点，Tail Preservation Loss 在 4–20 个百分点之间，表明聚合指标无法充分反映类级退化；3) 在不同攻击、目标类、模型架构下，局部损失的分布与幅度差异显著。

**⚠️ 局限性**

局限性包括：1) 只关注图像分类任务，未考虑其它任务或子群体；2) 评估基于类级别，忽略样本级或特征子空间的细粒度变化；3) 只选取满足 ASR 阈值的代表性修复状态，可能无法覆盖所有实际部署情况；4) 对大规模数据集和更复杂模型的适用性仍待验证。

---

## 205. What Should a Self-Teacher See? Privileged Context Design for On-Policy Self-Distillation

**arXiv ID:** 2609.25623 | [PDF](https://arxiv.org/pdf/2609.25623v1)

**作者:** Kanghui Tian `[一作]` (Fudan University), Yi Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在自监督分层训练中，比较不同抽象级别的先验解答对学生模型性能的影响。

**💡 创新点**

创新在于系统地评估完整解答与多级抽象提示（策略名称、方法无关框架、问题类别、仅答案）对自蒸馏的教学效用，并发现抽象提示可在大模型上提升平均得分。

**🔧 技术方法**

采用基于Qwen3的自上游对策略（on‑policy self‑distillation）和离线编译器生成提示，结合LoRA微调与固定教师。

**📊 数据集**

使用公开的竞赛数学数据集（AIME、HMMT）和四个迁移基准（MT‑AIME、GPQA‑Diamond、ZebraLogic、AutoLogi）。

**📈 对比分析**

通过在不同模型规模（1.7B、4B、8B）和多种评估规则（峰值、选择自由、200步）比较各提示，发现4B/8B模型在中等抽象层（L3或L4）平均比完整解答高1.4–1.6分，答案仅提示在大模型上与完整解答差距≤0.2分。

**⚠️ 局限性**

局限在于未能完全区分教师容量与提示抽象的相互作用，仅对单一教师/学生架构实验，且仅在固定模板下评估，缺乏对其他模型或更大规模的验证。

---

## 206. Teaching Reinforcement Learning and Humanoid Robotics to High-School Students: An Expert-Validated Curriculum Design on a Low-Cost Open Platform

**arXiv ID:** 2609.25674 | [PDF](https://arxiv.org/pdf/2609.25674v1)

**作者:** Yuanzhe Dong `[一作]` (Stanford University), Shuman Wang `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一套面向高中生的强化学习与类人机器人课程框架，学生通过组装开放源码类人机器人、在仿真环境中训练行走策略并将其部署到真实平台。

**💡 创新点**

创新点在于将完整的机器人研究工作流程（机械装配、硬件调试、仿真、政策学习、系统识别、现场部署）整合成一条贯穿多学科的课程路线，并通过专家评审迭代平衡真实性、认知负荷与团队协作等设计张力。

**🔧 技术方法**

使用了ToddlerBot开放硬件平台、NVIDIA Isaac Lab等仿真工具、PPO等强化学习算法，以及Python/ROS等编程环境。

**📊 数据集**

未使用公开数据集，课程依赖的是硬件原型（ToddlerBot）以及自建的仿真与测量数据；实验结果仅记录了学生完成组装、站立与步行的过程。

**📈 对比分析**

课程设计与实施通过专家评审（五位跨领域专家）进行定性评价，未进行定量性能比较；后续研究计划收集学生学习数据以评估课程效果。

**⚠️ 局限性**

局限性包括：缺乏实证学习效果评估、对硬件、计算资源和指导教师的高依赖、未针对不同机器人平台给出通用方案，以及课程完成后学生表现的数据未公开。

---

## 207. SMTB: Fast Structure-Mapping with Tight Bounds

**arXiv ID:** 2609.25508 | [PDF](https://arxiv.org/pdf/2609.25508v1)

**作者:** Daniel Weitekamp `[一作]` (Georgia Tech), Christopher MacLellan `[通讯]` (Georgia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了SMTB结构映射算法，在保持对结构映射理论约束的同时，显著提升了映射速度和在大规模域中的匹配质量。

**💡 创新点**

创新点在于不显式偏好高阶对应关系，而是通过实体级递归搜索与紧凑的启发式（tight bound）和“paired walks”来实现更灵活、更高效的结构对齐。

**🔧 技术方法**

使用C++实现的SMTB及其所依赖的CRE框架、改进的启发式搜索算法以及基于paired walks的全局连通性评估。

**📊 数据集**

评估数据集为SME语料库的5845对基-目标域。

**📈 对比分析**

与SME v4和LAP基线比较，SMTB在大多数问题上速度提升5–15倍，映射质量在SMTB目标上提升约20–80%，在SME目标上亦提升约20–30%。

**⚠️ 局限性**

局限在于对高度嵌套或完全缺失低阶对应的域仍可能出现kernel violation，需要进一步验证与调优；算法对非平衡数据集的鲁棒性仍有待考察。

---

## 208. Clarification Is Not Correction: LLMs Fail to Let Go

**arXiv ID:** 2609.25337 | [PDF](https://arxiv.org/pdf/2609.25337v1)

**作者:** Jianzhe Lin `[一作]` (Meta), Jubin Chheda `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究对话中模型因提前确定含糊解释而导致后续澄清无法消除旧假设的现象，并提出不确定性保持的状态管理思路；

**💡 创新点**

提出“早期后验坍塌”概念，定义旧假设污染度量，验证路径依赖与结构嵌入对污染的影响；

**🔧 技术方法**

基于 Gemini‑2.5 Pro/Flash 的提示工程与回滚、两阶段响应策略，并使用自动评判；

**📊 数据集**

生成的290个写作、规划、编码任务+20个手工任务与30个对抗任务；

**📈 对比分析**

通过对比全信息、澄清先行与模糊先行的成功率、约束满足率和污染率，发现模糊先行效果差，回滚与两阶段策略显著降低污染；

**⚠️ 局限性**

仅在 Gemini 体系上评估，污染度量可能欠准，提示层面的干预不能推广为通用解决方案，缺乏跨模型与人类评估验证。

---

## 209. Ultra-fast Neural Inference for Stochastic Gaussian Splatting Denoising

**arXiv ID:** 2609.25604 | [PDF](https://arxiv.org/pdf/2609.25604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 210. MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence

**arXiv ID:** 2609.25627 | [PDF](https://arxiv.org/pdf/2609.25627v1)

**作者:** Haoran Wen `[一作]`, Yan Xie `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 MachEmbodied‑U0（ME‑U0）——一种统一的嵌入式基础模型，通过混合Transformer架构同时实现子任务预测、赋能定位、视觉动态（RGB、深度、法向、光流）以及连续动作生成。

**💡 创新点**

创新点在于将任务语义推理与联合视觉‑动作生成相结合，利用流匹配实现动作与视觉预测的一致性，并通过多频率旋转位置编码（MRPE）实现视频与高采样动作的同步对齐；同时采用统一动作空间和全球任务感知采样，构建了跨机器人、多模态的预训练体系。

**🔧 技术方法**

核心技术包括 Mixture‑of‑Transformers、流匹配（flow matching）自噪声学习、MRPE、统一的动作表示、子任务与赋能的自回归文本生成、以及联合视频‑动作去噪与多任务损失融合。

**📊 数据集**

预训练数据约 4,200 小时机器人演示（来自 AgiBot World Beta、RoboMIND 2.0、RoboCOIN、Galaxea Open‑World）和 3,920 小时第一人称演示，并对一部分数据标注了子任务、赋能、几何和运动标签。

**📈 对比分析**

与现有 VLA 与 WAM 基线在 RoboDojo、LIBERO、LIBERO‑Plus 上进行对比，ME‑U0 在 RoboDojo 取得过程分 17.66、成功率 11.18%，在 LIBERO 99%+ 成功率，在 LIBERO‑Plus 82.5%，并在真实机器人上实现零样本任务执行，表现优于或接近最佳基线。

**⚠️ 局限性**

局限性包括对长期记忆需求不足导致记忆相关任务性能低；对视角、机器人初始状态等扰动的鲁棒性仍有待提升；模型规模大、推理成本高，限制了在资源受限平台的即时部署。

---

## 211. Compressibility is not Feedback: A Random-access Gap in Causal Semantic Repair

**arXiv ID:** 2609.25020 | [PDF](https://arxiv.org/pdf/2609.25020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 212. Subspace coverings and generalized covering radii of generalized Zetterberg codes

**arXiv ID:** 2609.25115 | [PDF](https://arxiv.org/pdf/2609.25115v1)

**作者:** Shitao Li `[一作]` (Anhui University), Zhonghua Sun `[通讯]` (Hefei University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了广义 Zetterberg 码的广义覆盖半径，给出了通用上界、下界，并在多种参数范围内确定了确切值，特别是二进制情况的第二和第三覆盖半径。

**💡 创新点**

提出了将覆盖半径转化为有限几何中的饱和集问题，利用傅里叶系数、Kloosterman 求和、代数曲线计点与自共轭根度判定等新工具，实现了对广义覆盖半径的精确估计，并首次完整计算了 8|m 时的第三覆盖半径为 7。

**🔧 技术方法**

主要技术包括：
- 线性码与范数-一子群的几何对应；
- 计数与子域构造法得到下界；
- 通过多项式曲线（Artin–Schreier 与二次覆盖）的点计数与 Hasse–Weil 上界得到上界；
- 采用傅里叶系数和 Kloosterman 和估计，推导子空间内部覆盖的精确条件；
- 自共轭多项式根度判定与 Magma 计算，处理无穷扩张族的第三覆盖半径。

**📊 数据集**

论文为理论研究，无实验数据集；所有结论基于代数与有限几何证明，部分结果通过 Magma 在 �256 上验证。

**📈 对比分析**

与以往仅给出上界/下界的结果相比，本文在大范围参数下实现了确切值，证明 ρ_t(U)=t 的范围显著扩大；对于二进制 Zetterberg 码，第二覆盖半径在所有扩张度上确定，第三覆盖半径在 8|m 时确立为 7，弥补了之前仅在有限小参数下已知的空白。

**⚠️ 局限性**

局限性：
- 对于 t>2 的非二进制码，仍有部分参数区间未得到完全精确值；
- 第三覆盖半径仅在 8|m 的子族内得到确定，未覆盖 4∤m 的情况；
- 计算上对 Magma 进行的自共轭多项式检查虽可扩展，但仍需理论化证明；
- 对于极大 t（接近冗余度）时，仍依赖上、下界相近但未必精确。

---

## 213. The Vocabulary of Flaky Tests in Swift

**arXiv ID:** 2609.25516 | [PDF](https://arxiv.org/pdf/2609.25516v1)

**作者:** João Medeiros `[一作]` (Universidade Federal de Pernambuco), Breno Miranda `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在 15 个开源 Swift 项目中收集 91 条 flaky 与 22,349 条 stable 测试，构建词汇特征（TF‑IDF unigram+bigram），并使用多种机器学习分类器（Random Forest、SVM、Decision Tree、Naïve Bayes、KNN）训练静态预测模型，以识别 Swift 测试代码中的 flakiness。

**💡 创新点**

创新点在于首次将词汇基的静态 flakiness 预测方法应用于 Swift 生态，揭示了 async/await、expectation、throws、timeout 等词汇作为 flakiness 标记，并发现了稳定词汇如 xctassertequal 的对立作用，同时验证了这些词汇与 Swift 的异步/UI 测试特性的一致性。

**🔧 技术方法**

技术手段包括：文本分词与去除保留词、TF‑IDF 词向量、信息增益排序、五种分类器训练与 5‑fold 交叉验证、随机欠采样与多次重采样、基线比较（多数类、随机、词汇阈值）和性能评估指标（Precision、Recall、F1、MCC、AUC）。

**📊 数据集**

数据集由 15 个开源 Swift 项目（涵盖网络、服务器、UI、工具等领域）构成，采用两种来源：对测试用例 50 次本地多次执行筛选 flaky，以及挖掘包含“flaky”关键字且修复了非确定性 bug 的提交/PR，最终得到 91 条 flaky 与 22,349 条 stable 测试。

**📈 对比分析**

评估方法为 5‑fold stratified CV、30 次随机欠采样、记录 Precision、Recall、F1、MCC、AUC；Random Forest 与 SVM 在 AUC 0.95、F1 0.86、MCC 0.75 等方面显著优于基线；在自然 0.4% flaky 分布下 Precision 降至 0.041，说明模型更适合作为优先级排序而非直接门禁。

**⚠️ 局限性**

局限性包括：词汇特征无法捕捉隐藏在共享基础设施或 deterministic 使用的 async（导致误判或漏判）；在真实稀疏比例下精度低，且模型对项目内词汇高度依赖，跨项目泛化能力受限；数据集仅覆盖公开 Swift 项目，可能不适用于私有或极端 UI 测试场景。

---

## 214. Incipit: Axiom-Grounded Scaffolding for Human-AI Literary Creation

**arXiv ID:** 2609.25504 | [PDF](https://arxiv.org/pdf/2609.25504v1)

**作者:** Qiang Liu `[一作]` (Noevara Inc.), Chunyi Zhao `[通讯]` (University of Otago)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出“文学公理”框架，将作品的创作、解读与评价通过概念配置与文本实现相连

**💡 创新点**

创新在于将概念承诺与形式实现的关系显式化，构建可检验的配置、实现与读者重建模型，并定义五维文学评价指标

**🔧 技术方法**

使用结构化知识表示（JSON schema）、关系型映射、概念图与定量评估框架，配合定性推理与可视化工具

**📊 数据集**

基于中文文学公理数据集：1455条公理记录、1464条作品-公理映射（涉及149部作品）以及472条类型化关系

**📈 对比分析**

目前仅完成结构性描述性审计（无对照实验），评估方法是对比配置与实现的一致性、读者重建的可恢复性与多样性；未给出性能数值，待后续读者与创作实验验证

**⚠️ 局限性**

主要限制包括缺乏实证验证、跨作品共享度低、上下文类型标注不完整、缺少生成与审计日志、以及对非文字传统的覆盖不足

---

## 215. Continuous Optimization for p-adic Models

**arXiv ID:** 2609.25501 | [PDF](https://arxiv.org/pdf/2609.25501v1)

**作者:** Julian Salazar `[一作]` (Google DeepMind), Lucas Dixon `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在贝尔科维奇ℚ_p线性和两层模型上进行原生连续梯度下降的首个方法，实现了对p-进制参数的有效连续优化；

**💡 创新点**

创新点在于利用贝尔科维奇空间Γ_p作为路径连通的扩展，使得p-进制数能在连续度量空间中定义可微的损失与梯度，并在此空间中实现梯度下降、动量和Adam；

**🔧 技术方法**

主要技术包括：对p-进制模型的解析延拓、最大-加法（tropical max-plus）传播、在Γ_p上的前向/反向传播、按参数组联合梯度下降以及基于分段线性方向导数的几何梯度更新；

**📊 数据集**

使用的数据集有：在3-进制上进行的线性回归与两层回归实验、对数值为p^k模运算的分类任务（x mod 4,9等）以及1,680条命题的Quillian语义网络基准；

**📈 对比分析**

与传统的离散搜索/光束搜索方法、欧氏空间的梯度下降和多层感知机进行对比，结果显示：在模运算分类上实现100%准确率，在Quillian基准上梯度方法在F1、AP和准确率上与光束搜索持平甚至略胜；在回归任务上梯度方法比光束搜索收敛更快、效果相当；

**⚠️ 局限性**

局限性包括：尚无严格的收敛性证明，深层网络的架构与参数组划分仍需研究，计算成本与半径上界的折衷未完全优化，且对更大规模/更复杂模型的可扩展性仍待验证。

---

## 216. Mean Velocity Matching: Rethinking Generative Dynamics in Diffusion Models

**arXiv ID:** 2609.25444 | [PDF](https://arxiv.org/pdf/2609.25444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 217. ZeroGate: Trust-Preserving Fast Paths for Governed AI Agent Runtimes

**arXiv ID:** 2609.25443 | [PDF](https://arxiv.org/pdf/2609.25443v1)

**作者:** Zexun Wang `[一作]` `[通讯]` (Ond Holdings Inc.), Zexun Wang (Ond Holdings Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该工作提出了一种在代理（agent）准备阶段就完成精确动作（exact‑action）授权签发，然后在实际执行时重新验证并在本地持久化消费的机制；通过 SQLite 事务将一次性消耗（nonce）与配额更新和接收记录绑定；并在 Azure Blob 存储实验中对同步与提前授权两种模式的调度边界和整体生命周期延迟进行测量。

**💡 创新点**

创新点包括：① 明确区分预先签发、实时重验证和本地持久化三步；② 通过可重签名签名（Ed25519）与签名哈希绑定实现一次性、精确的授权；③ 在本地事务中同时完成 nonce 消耗、配额更新和接收记录，形成“可重验证合同”；④ 提出了条件决策保持（conditional decision‑preservation）命题，并给出了完整的假设与证明框架；⑤ 对准备模式与同步模式的边界与生命周期延迟进行量化比较，揭示了准备模式能缩短调度边界但整体耗时并不降低。

**🔧 技术方法**

主要技术包括：① 规范化的 Canonical Action 表示与递归排序 JSON 序列化；② Ed25519 签名与 SHA‑256 哈希，用于签发与验证 Pass；③ SQLite WAL 事务实现一次性 nonce 消耗、配额管理和接收记录写入；④ 可信适配器（adapter）负责重新构造最终请求并与 Gate 对齐；⑤ Azure Blob 存储的条件写入与读取，用于验证外部效果；⑥ 云实验脚本（cloud‑study.mjs）与语义基准（semantic‑benchmark.mjs）实现对不同并发与模式的测量。

**📊 数据集**

数据集：① 云实验采用 Azure Blob 写入工作负载，单个 batch 包含 160 次写操作，payload 大小为 128、1024、3072 字节；并发等级分别为 1、8、32；实验分为两种模式（同步与提前准备）共 5 组实验；② 语义基准使用 73 个手工构造的场景模板，涵盖 payment、mail、deployment、storage 四个领域；每个场景在 5 次重复后生成约 1,360 条非授权样本与 100 条授权样本。

**📈 对比分析**

比较方法：在同步模式下，worker 先获取动作后再进行授权签发；在提前准备模式下，批量动作先被授权签发，再等待 worker。通过记录 worker‑admission‑to‑dispatch（T_boundary）与完整生命周期（T_lifecycle）两种时钟，计算 95% 分位数和平均值。结果显示：提前准备模式在 T_boundary 上平均降低 5–30%（取决于并发级别），但 T_lifecycle 的平均值在所有级别上均更长，说明准备阶段的额外等待抵消了调度边界的优势。性能评估表明：在 32 并发下，T_boundary 95% 分位数从 1000 ms 降至 700 ms，T_lifecycle 95% 分位数从 1500 ms 增至 1700 ms。

**⚠️ 局限性**

局限性：① 条件决策保持仅在假设签发可信、依赖完整、观察真实且证据一致的前提下成立，无法保证真实世界的即时性或一致性；② 本地事务并未覆盖外部服务的原子性，导致可能出现已消费 nonce 而外部操作失败的情况；③ 方案仅支持单次使用的 exact‑action Pass，无法实现可重用的授权；④ 评价实验未覆盖跨地区或高失效率环境的分布式一致性；⑤ 缺乏对撤销、时效过期以及跨域配额同步的完整机制；⑥ 仅在受信任的运行时与 Gate 内部实现，无法保证非受信任代理的安全性。

---

## 218. Beyond Task Performance: Lessons Learned from Evaluating an Exploratory VR Interaction Technique

**arXiv ID:** 2609.25414 | [PDF](https://arxiv.org/pdf/2609.25414v1)

**作者:** Nevzat Umut Demirseren `[一作]` (Texas State University), Kevin Pfeil `[通讯]` (University of Central Florida)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出并评估了一种基于弧形轨迹的双手交互技术 RodCast，用于在密集虚拟环境中实现目标选择与操控。

**💡 创新点**

创新点在于将完整的弧形交互路径可视化，并采用不对称双手控制，使用户能够直观感知遮挡目标的可达性和空间关系。

**🔧 技术方法**

技术实现基于 Meta Quest 3 头显，结合可视化轨迹、地面投影指示器和动态反馈；与 Go-Go Hand 与 FlowerCone 两种现有技术进行对比。

**📊 数据集**

使用 22 名参与者在自建的三种实验场景（距离感知、密集操控、对象排序）中收集的原始实验数据，未使用公开数据集。

**📈 对比分析**

通过重复测量 ANOVA 与 Wilcoxon 符号秩检验比较三种技术；RodCast 在距离感知任务中显著降低定位误差，但在任务完成时间和误碰次数上不如 Go-Go Hand 和 FlowerCone，主观评价中 Go-Go Hand 最受欢迎。

**⚠️ 局限性**

主要局限在于双手触发控制导致协调负担高、学习曲线陡峭，尤其对 VR 经验不足的用户不友好，限制了技术的效率提升。

---

## 219. Lean Pool: An AI-Maintained Archive of Formalized Mathematics

**arXiv ID:** 2609.25199 | [PDF](https://arxiv.org/pdf/2609.25199v1)

**作者:** Vasily Ilin `[一作]` `[通讯]` (University of Washington), Vasily Ilin (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

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

## 220. Federating Quantum and Classical Computing: A Privacy-Preserving Hybrid Approach

**arXiv ID:** 2609.25082 | [PDF](https://arxiv.org/pdf/2609.25082v1)

**作者:** Carlos Cano `[一作]` (Sherpa.ai), Xabi Uribe-Etxebarria `[通讯]` (Sherpa.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实验了一种混合量子-经典模型在隐私保护垂直联邦学习（SBVFL）中的应用，证明在保持数据不共享的前提下可显著提升分类性能。

**💡 创新点**

创新点在于首次将量子变分电路嵌入SBVFL协议，形成异构主动-被动合作框架，同时提出了SMPP（Split Multiplicative Periodic Parity）基准来量化量子模型在隐私限制下的参数效率。

**🔧 技术方法**

使用的技术包括参数化量子电路（四比特角编码+两层旋转+CNOT耦合）、经典线性头、随机森林、全连接网络以及SBVFL的合成标签机制与隐私乘数 Q 的调节。

**📊 数据集**

数据集为人工生成的SMPP基准，包含 2000 条训练样本与 20000 条测试样本，特征被垂直拆分为两部分，每部分包含两维多项式周期信号。

**📈 对比分析**

对比方法为本地主动方、SBVFL 联邦学习和中心化完整特征训练，实验表明量子模型在SBVFL中以12个可训练参数取得 0.8757 的准确率，显著高于本地模型（0.7227）和中心化的经典网络（0.8663），且相较于随机森林具有更高的参数效率。

**⚠️ 局限性**

局限性包括仅在小规模可模拟的四比特电路上验证、缺乏真实量子硬件实验、只针对单一合成基准且未评估更大规模或不同噪声条件下的性能。

---

## 221. Near-Optimal Online Metric Matching on $Δ$-ary HST

**arXiv ID:** 2609.25292 | [PDF](https://arxiv.org/pdf/2609.25292v1)

**作者:** Parth Gor `[一作]` (University of Iowa), Kasturi Varadarajan `[通讯]` (University of Iowa)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种针对 Δ-ary 2-HST 的随机在线匹配算法，其期望竞争比为 O((logΔ)·loglogΔ)，并且与服务器/请求数量无关。

**💡 创新点**

创新点在于设计了一种新的重新分配策略，强调局部重新分配并采用全新的采样分布，使得重新分配成本得到更精细的控制，几乎达到 logΔ 的下界。

**🔧 技术方法**

利用重新分配框架（Reassignment Model）与 HST 嵌入技术，结合层次分层（special layers）与随机子树选择，并通过耦合与尾部分布分析完成了复杂的概率证明。

**📊 数据集**

本研究为理论工作，没有使用具体数据集；所有结果均为算法分析与证明。

**📈 对比分析**

与以往基于 HST 的 O(log²n) 竞争比算法相比，该算法在 Δ-ary 2-HST 上实现了更优的 O((logΔ)·loglogΔ) 竞争比，且不依赖于请求/服务器数量，达到了近乎最优的性能。

**⚠️ 局限性**

主要局限在于算法仅针对 2‑HST（α=2）且 Δ‑ary 的结构，难以直接推广到一般 HST 或非树形度量空间；此外，对随机请求序列的隐式先验假设（oblivious adversary）仍然是必要的。

---

## 222. Mining Legal Arguments in U.S. Corporate Case Law

**arXiv ID:** 2609.25441 | [PDF](https://arxiv.org/pdf/2609.25441v1)

**作者:** Luis Brena `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了42份美国联邦税法（I.R.C. 368）公司重组判例的专家标注语料库，标注了五种功能节点（Rule、Analysis、Conclusion、Background Facts、Procedural History），并提供了 span、sentence、flat 以及树结构化表示。

**💡 创新点**

首个针对该税法领域的专家标注、树结构化语料库；在单词级/句子级标注、树结构可靠性评估与多层次评测上提供创新框架；通过案例分离评估展示功能标签的可学习性与结构标签的局限性。

**🔧 技术方法**

采用 Label Studio 进行标注；利用 Krippendorff α_u、soft‑F1、边/路径一致性等指标评估标注质量；在分类任务中使用 TF‑IDF、SBERT、LegalBERT、ModernBERT 以及 GPT‑5（含上下文/无上下文）；在检索任务中使用 BM25、E5‑base‑v2、以及 Fine‑tuned ModernBERT 双编码器。

**📊 数据集**

42份I.R.C.368公司重组税法判例（文本 1k–10k 词），公开在 HuggingFace；该语料包含 span、sentence、flat 以及树结构化注释。

**📈 对比分析**

分类采用五类/四类的 case‑disjoint 交叉验证，LegalBERT 在五类宏 F1 达 0.71、四类 0.80；GPT‑5‑4 在有上下文时宏 F1 0.84；检索采用 Hit@20、Recall@20、Complete Recovery@20、MRR，Fine‑tuned flat retriever 在同案过滤/全案中最高（Hit@20 80.45%/67.98%），跨案 Fold‑global 仍以 BM25 为最佳（Hit@20 42.49%）。

**⚠️ 局限性**

局限性：仅 42 篇特定年代（1935–1987）的英文税法案例，数据量有限；结构层标注可靠性低，跨案迁移效果差；未涵盖其他税法、法律领域或语言；因专家标注成本高，难以扩展规模。

---

## 223. Dual-GNN Multilevel Coarsening for Maximum Independent Set

**arXiv ID:** 2609.25149 | [PDF](https://arxiv.org/pdf/2609.25149v1)

**作者:** Tianfeng Chen `[一作]` (Lanzhou University), Xianyue Li `[通讯]` (Lanzhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为Dual-GNN多级粗化的框架，通过学习指导多级图粗化，同时保留组合搜索以进行最终决策，旨在解决最大独立集（MIS）问题。

**💡 创新点**

创新点在于将学习与组合优化分开，使用Partition GNN评分候选收缩，并通过Representative GNN选择每个最终集群的前k个局部独立集状态，从而提高了求解效率和解的质量。

**🔧 技术方法**

使用了图神经网络（GNN）技术，具体包括Partition GNN和Representative GNN，来指导图的粗化和状态选择。

**📊 数据集**

使用了Erdős–Rényi图作为数据集，进行了多种规模（最多2000个顶点）的实验。

**📈 对比分析**

与传统的精确求解器和其他学习方法相比，该方法在500个顶点的实例中达到了19.20的平均独立集大小，接近最优值19.30，同时将求解时间从643.57秒减少到3.41秒，速度提升约189倍。在1000和2000个顶点的较大图中，该方法在所有评估方法中表现出最佳的平均解质量。

**⚠️ 局限性**

限制在于该方法仅在Erdős–Rényi图上训练，尽管在未见的图密度和结构上具有良好的泛化能力，但在非常稀疏的图上仍然面临挑战。

---

## 224. Not All 4-bit Quantizers Are Equal: Deployment-Time Mitigation of PII Leakage in Fine-Tuned Small Language Models

**arXiv ID:** 2609.25014 | [PDF](https://arxiv.org/pdf/2609.25014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 225. Learning Defensive Policies against Diverse Inference Attacks for Smart Meter Privacy

**arXiv ID:** 2609.25484 | [PDF](https://arxiv.org/pdf/2609.25484v1)

**作者:** Ruichang Zhang `[一作]` (University of Manchester), Mustafa A. Mustafa `[通讯]` (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对家庭电表被非侵入式负载监测(NILM)攻击的隐私风险，本文设计了一种基于代理引导的分层强化学习框架，通过可充电电池注入逼真的电器签名来混淆聚合电力数据，从而降低攻击者对单个电器功耗的推断能力。

**💡 创新点**

创新点在于：① 用自监督聚合结构探针作为对抗奖励，使得电池调度策略无需了解具体攻击模型即可优化；② 通过预构建电器签名库将扰动限制为物理可实现且与攻击任务高度契合；③ 采用分层强化学习（高层决策签名，低层执行电池功率）实现对策略的实时控制与物理约束兼顾。

**🔧 技术方法**

技术手段包括：分层强化学习（PPO），自监督Seq2Seq/Transformer/ LSTM聚合探针，签名库构建与匹配，电池动力学与约束求解，代理奖励设计与能耗/退化成本权衡。

**📊 数据集**

使用公开的 UK-DALE 和 REDD 两个高频智能电表数据集，均进行分钟级预处理和训练-测试划分，签名库从 UK-DALE 训练集中提取。

**📈 对比分析**

与随机/规则基电池策略以及两种基线深度RL隐私防御（DDQL-flat、DDQL-MI）进行对比。实验显示：在六种未见 NILM 攻击器（优化式、概率式、卷积、自动编码、Transformer）上，本文方法平均 RMSE 误差提升超过 100%（UK-DALE）/ 165%（REDD），F1 检测下降超过 80%；比 DDQL-MI 提升约 13% RMSE、6% F1；且能耗成本更低、训练时间更短。

**⚠️ 局限性**

局限性包括：① 签名库覆盖有限，主要适用于高功率间歇电器，难以有效掩蔽低功率组合负载；② 理论分析仅给出局部充分条件，未能完全刻画跨攻击器的全局行为；③ 评估仅在离线固定攻击器上完成，未考虑在线预测不确定性、实时电池状态估计以及自适应攻击者的挑战。

---

## 226. Beyond the Flat Seafloor: A Closed-Form Two-View Constraint to Aid Sidescan Sonar Reconstruction

**arXiv ID:** 2609.25271 | [PDF](https://arxiv.org/pdf/2609.25271v1)

**作者:** Kalin Norman `[一作]` (Brigham Young University), Joshua G. Mangelson `[通讯]` (Brigham Young University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文推导了两视角侧扫声呐的闭式几何约束，将单一视角的无三维定位问题转化为两球面相交得到的1D轨迹（locus），并通过蒙特卡罗仿真评估该轨迹长度与相对姿态、传感器参数的关系，进而给出针对不同船舶平台的实用调查路径规划建议。

**💡 创新点**

创新点在于：①首次在不假设平坦海底、无迭代或学习模型的前提下，利用侧扫声呐的球面投影特性得到闭式球面-平面相交的约束；②通过真实平台（BlueBoat/Omniscan 450 SS与AUV/EdgeTech 2205）的几何参数进行蒙特卡罗仿真，系统量化升降误差、跨角等因素对轨迹长度的影响；③提出避免10°–40°相交角、优先近重复或反向航线的调查规划策略。

**🔧 技术方法**

主要技术包括：球面与平面的解析几何求交、SNR无噪声的闭式方程推导、基于真实设备参数的蒙特卡罗仿真、Spearman相关性与敏感度分析、跨角与姿态对轨迹长度的数值实验。

**📊 数据集**

使用了两套真实设备的参数集：Omniscan 450 SS（横向50°、纵向0.5°、最大范围150 m）与EdgeTech 2205（横向130°、纵向0.27°、最大范围150 m），并以BlueBoat和AUV（类似REMUS/EdgeTech 2205）作为平台进行仿真。

**📈 对比分析**

与传统单视角或假设平坦海底的单一范围解法相比，本文的两视角闭式约束将不确定性从无穷大压缩到可观测的1D轨迹；仿真表明在近重复或反向航线时轨迹长度可降至几厘米级（取决于平台），显著优于单视角或无约束情况下的误差。

**⚠️ 局限性**

局限性包括：①未考虑测量噪声和相对姿态不确定性；②仅分析了两视角场景，未系统研究三视角或多视角是否可完全消除轨迹；③缺乏实地实验验证，仅基于仿真；④对极端海况（高波浪、强流）下的姿态变化未作深入研究。

---

## 227. A Computational Approach to Measuring Semantic Change in Sanskrit Literature

**arXiv ID:** 2609.25012 | [PDF](https://arxiv.org/pdf/2609.25012v1)

**作者:** Tanay Agrawal `[一作]` `[通讯]` (Harker School), Tanay Agrawal (Harker School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了梵语四个历史时期的词向量，利用神经字节级砂砾拆分与词形还原统一预处理管线，并通过方向性锚定检验方法量化并验证文献记录的语义转移。

**💡 创新点**

首次将词向量语义变迁技术移植到低资源、复杂形态的古梵语，结合砂砾拆分和词形还原形成端到端的评估框架，并通过锚点方向性检验将语义迁移与历史文献直接对齐。

**🔧 技术方法**

使用ByT5-Sanskrit进行字节级砂砾拆分与词形还原；word2vec与FastText（不同n‑gram范围）生成嵌入；采用第二阶相似度与正交Procrustes对齐衡量变化；使用锚点方向性检验与符号检验评估结果。

**📊 数据集**

约270万词的四期梵语语料（Vedic、Upanishadic、Epic、Sutra/Shastra）来自GRETIL，预处理后得到原始、拆分和词形三种版本；同时收集了21个已记录的语义转移作为验证集。

**📈 对比分析**

在不同预处理和模型配置下训练嵌入，比较邻居质量、正交性、可靠度；方向性锚定检验显示19/21词在历史记录的方向上位移（p≈0.00011），表明方法有效；词2vec/拆分配置在邻居正统性和可靠度方面优于FastText，FastText受短n‑gram污染影响，词形化未提升性能。

**⚠️ 局限性**

语料规模有限且不平衡，残留注释/编辑噪声；静态嵌入无法捕捉多义与非线性变化；评估仅验证已知转移，缺乏对未知或新颖语义变化的发现能力。

---

## 228. Trains but Doesn't Learn: A Post-Training Delivery Benchmark for LLM Agents as Forward-Deployed Engineers

**arXiv ID:** 2609.25237 | [PDF](https://arxiv.org/pdf/2609.25237v1)

**作者:** Weihang Ding `[一作]` (University of California), Junfei Zhan `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并验证了一个受治理的交付平面（governed delivery plane），以评估大型语言模型代理在 PTaaS 中担任前置部署工程师（FDE）时的可信交付能力。

**💡 创新点**

创新点在于：①引入十个可分阶段评估的交付门控和离线 oracle 评分，从交付可靠性角度而非单纯指标提升来衡量代理；②设计了“训练但不学习（TBDL）”运行检测器和接受门，在预付费前拦截严重错误；③通过故障注入校准和 e‑process 实时监控确保检测器鲁棒。

**🔧 技术方法**

使用的技术包括：故障注入校准检测器、基于 e‑process 的实时监控、离线 oracle 评估、合同式交付检查、对齐检验以及多阶段分层控制框架。

**📊 数据集**

使用的数据集与模型：五个 SFT 场景（BANKING77、Bitext、Glaive‑FC、FinQA、CUAD）以及四个开源权重基础（Qwen3‑8B/32B、Gemma‑2‑9B‑it、Llama‑3.3‑70B）。

**📈 对比分析**

对比方法与性能：在 L40S、A100、H200 GPU 上对四大前沿代理（Claude Opus 5、GPT‑5.6‑luna、Gemini 3.7‑Flash、DeepSeek V4‑Pro）与人工 FDE 进行基准；代理在交付合同上被拒绝 75 % 的 metric‑passing 交付，而人工 FDE 的判定残差更低但仍未达到零。

**⚠️ 局限性**

局限性：仅单节点、仅 post‑training；受限于五个任务与四个模型；评估依赖单一 oracle 注解，未覆盖偏好学习或多任务等更广泛场景。

---

## 229. On the Offline Version of the Time-Optimal k-Server Problem

**arXiv ID:** 2609.25180 | [PDF](https://arxiv.org/pdf/2609.25180v1)

**作者:** Oleg Lomachenko `[一作]` `[通讯]` (GoodsForecast), Oleg Lomachenko (GoodsForecast)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了离线时间最优k-服务器问题的复杂性，提出了在欧几里得直线（或加权路径）上的强NP-完备性证明，并给出在带公共枢纽的无权图上多项式求解的算法；

**💡 创新点**

创新点在于首次证明该问题在最简单的线性度量空间中已是强NP-完备，同时发现存在一个非平凡的多项式可解边界——即所有顶点都连通到单个公共枢纽的无权图；

**🔧 技术方法**

采用了归约、构造图、匹配、动态规划等理论工具；对无权图的情形，使用最短路径动态规划和匹配判定实现多项式求解；

**📊 数据集**

论文没有使用实际数据集，全部以理论构造和形式化证明为主；

**📈 对比分析**

通过与经典距离模型（求和代价）对比，展示并行移动在时间模型下可显著降低代价；在直线度量空间下证明不可多项式求解；在通用枢纽网络上给出确切最优时间计算，复杂度为多项式；

**⚠️ 局限性**

局限性包括：只讨论离线版本；未给出线性度量空间下的近似算法；只提供一种特殊图结构下的多项式解法，未探讨更一般图的情况。

---

## 230. Real-Time Atomic-Resolution Electron Phase Imaging without Probe Calibration via Ptychography-Supervised Learning

**arXiv ID:** 2609.25684 | [PDF](https://arxiv.org/pdf/2609.25684v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 231. JAMB: Joint Action-Motion Diffusion for Bimanual Manipulation

**arXiv ID:** 2609.25322 | [PDF](https://arxiv.org/pdf/2609.25322v1)

**作者:** Chuyang Xiao `[一作]` (Carnegie Mellon University), David Held `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种联合去噪的扩散策略，能够同时生成双臂控制指令和未来3D点轨迹，用于协同双臂操控；

**💡 创新点**

创新点在于把未来运动预测与动作生成统一为同一扩散过程，使动作与轨迹在去噪过程中相互补充，并在多模态嵌入中使用4D旋转位置编码实现共享时空坐标系；

**🔧 技术方法**

使用扩散Transformer（DiT）框架，DINOv2视觉编码器，四维RoPE位置编码，点轨迹与视觉特征的融合投影，以及基于轨迹的加权损失；

**📊 数据集**

在RoboTwin 2.0仿真环境（16个双臂任务）以及三台xArm机器人+ZED Mini摄像头的真实场景（Store Block、Stack Basin、Place Duck Box）上训练和评估；

**📈 对比分析**

与动作仅预测（DP、DP3）、未来状态辅助（DICP、GAP）以及轨迹引导（ATM）等基线比较；在仿真中平均成功率83.4%，比最强基线高23.9个百分点；在真实环境中平均成功率85.6%，比最强基线高21.2个百分点，并在噪声背景、无序对象等难度条件下表现出更好的泛化；

**⚠️ 局限性**

局限性包括仅使用单目视角和固定查询网格，依赖点跟踪和深度估计的准确性；未考虑力/接触信息；未来需探索多视角融合、查询自适应选择及对噪声监督的鲁棒性。

---

## 232. Fast Recovery for LLM Serving via Decoupled Device Memory Lifetime in Dynamo

**arXiv ID:** 2609.25451 | [PDF](https://arxiv.org/pdf/2609.25451v1)

**作者:** Schwinn Saereesitthipitak `[一作]` (NVIDIA), Wen-mei W. Hwu `[通讯]` (NVIDIA Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套面向大规模LLM推理的快速恢复体系，利用快照（Snapshot）、GPU内存服务（GMS）与Shadow Engine三种机制将设备保留型故障的恢复时间从几分钟降至约7秒，显著降低GPU空闲时间。

**💡 创新点**

创新点在于将已初始化的模型权重与运行时状态与引擎进程解耦，GMS通过CUDA虚拟内存管理实现模型权重的共享与持久化，Shadow Engine提前预初始化第二个运行时以实现几秒级恢复，并将三者组合成分层恢复路径，兼顾设备保留与设备失效两类故障。

**🔧 技术方法**

核心技术包括：CRIU+CUDA checkpoint/restore实现全流程快照；CUDA VMM与多GPU共享的GMS实现物理内存持久化与只读共享；Shadow Engine并行初始化并在故障时直接切换；请求重放机制；以及对CUDA图、NCCL通道等运行时资源的序列化与重建。

**📊 数据集**

评估基于NVIDIA生产集群18周的故障日志，实验模型包括Qwen3.8‑27B、DeepSeek‑V4‑Flash、GLM‑5.2、DeepSeek‑V4‑Pro等四种8GPU配置，使用vLLM和SGLang两大推理框架进行基准测试。

**📈 对比分析**

采用温热重启为基线，分别测量GMS、Snapshot、Snapshot+GMS、Shadow Engine等配置的恢复延迟与吞吐；Shadow Engine在所有模型中实现4‑6 s恢复，比温热重启快12‑29倍；在生产轨迹下估算可恢复GPU时长达79%，相当于将2,375 GPU小时恢复到近乎零。

**⚠️ 局限性**

局限性包括：目前仅恢复已初始化的服务容量，未保留KV缓存或调度器状态；Shadow Engine额外占用4‑8 GiB GPU内存，可能对KV受限部署产生影响；仅针对设备保留型故障有效，设备失效时仍需快照恢复；实验集中在单机8GPU复制，未覆盖多节点大规模复制的故障分布；未来需要完善请求状态恢复、进一步压缩快照大小及提升GPU恢复带宽。

---

## 233. Towards participatory speech dataset curation: A queer case study and conceptual framework

**arXiv ID:** 2609.25496 | [PDF](https://arxiv.org/pdf/2609.25496v1)

**作者:** Brooklyn Sheppard `[一作]` (University of Calgary), Levent Sagun `[通讯]` (Meta FAIR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并阐述了一个面向 LGBTQIA+ 社群的参与式语音数据集创建框架，并讨论了其四个阶段：社区、项目制定、参与方式与个人自主。

**💡 创新点**

首次将共创与知识共享方法融入语音数据收集，强调双向、持续的社区参与和对多样身份的细致考虑；提供可循环的四阶段参与模型。

**🔧 技术方法**

主要基于共设计、参与式 AI 方法、案例研究与文献综述；未使用特定机器学习技术。

**📊 数据集**

未使用具体语音数据集，主要以 LGBTQIA+ 语音与相关研究为案例讨论。

**📈 对比分析**

本研究为概念框架，无实验比较；未报告性能指标。

**⚠️ 局限性**

缺乏实证评估，框架适用性和可操作性待后续实验验证；对不同语言、交叉身份的细节考虑不足。

---

## 234. MoM: Memory of Memory

**arXiv ID:** 2609.25054 | [PDF](https://arxiv.org/pdf/2609.25054v1)

**作者:** Bowen Qin `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种名为Memory of Memory（MoM）的长期LLM代理记忆框架，该框架在写入时即决定当前值，同时保留被覆盖的旧值，并通过可追溯的 provenance 图维护历史；

**💡 创新点**

核心创新在于将“写时提交”与“保留被覆盖值”相结合，构建 Typed provenance graph（Provenant Memory），并设计 accept/contest/reject/revoke/resolve 等语义化操作，使冲突可被实时记录、解决并可逆；

**🔧 技术方法**

技术上采用软键相似度匹配进行语言推断键识别、写时 supersession、图结构的 provenance 边（supersedes/alternative/revokes）、图引导的 turn pruning、disclosure-mode 读取以及检索+重构的读写管线；

**📊 数据集**

实验使用三大基准：LongMemEval（知识更新场景）、ALFWorld-Revision（多轮修订链）以及 MemoryAgentBench FactConsolidation（冲突/多跳推理）等；

**📈 对比分析**

在与长上下文、MemoryOS、Mem0、HippoRAG-v2、Graphiti/Zep 等基线对比中，MoM 在准确率保持与最强检索相近的同时，stale-answer 率显著下降至约10%，且上下文 Token 消耗约为检索方法的 1/4，展示了优越的有效性与效率；

**⚠️ 局限性**

局限性包括：对自然对话中的写时提交仍受提取覆盖率与键 canonicalization 限制；检索仍是必不可少的操作；对更复杂的跨时间或权威度恢复（如重定时、源可信度）尚未充分验证。

---

## 235. AIBuildAI-2.5: Efficient Autonomous AI Model Development Through LLM-Guided Tree Search

**arXiv ID:** 2609.25047 | [PDF](https://arxiv.org/pdf/2609.25047v1)

**作者:** Peijia Qin `[一作]` (University of California San Diego), Pengtao Xie `[通讯]` (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AIBuildAI‑2.5，一套基于 LLM 的自主 AI 模型构建系统，利用树搜索、LLM 评判器和选择器进行候选程序的扩展与选择，并结合资源感知调度和模型路由以提升效率。

**💡 创新点**

核心创新点包括：① 通过 LLM 评判器给候选节点提供三维先验（预期提升、依据、可行性）并由全局选择器整合，解决训练成本高导致奖励稀疏的问题；② 设计资源感知作业调度器，可在 GPU/CPU 受限下并行启动任务；③ 建立模型路由系统，为不同角色动态分配成本与能力相匹配的 LLM，实现推理成本显著下降。

**🔧 技术方法**

技术方案涵盖：多角色 LLM 代理（设计师、编码器、修订者、评判者、选择器、调度器、路由器）使用 Claude Haiku/Sonnet/Opus；树搜索框架；LLM 先验评分与全局排序；硬件资源监控与调度；路由知识库；以及辅助代理（环境搭建、数据划分、模型聚合）。

**📊 数据集**

主要使用公开基准：MLE‑Bench（75 个 Kaggle‑style 任务，视觉、文本、时间序列、表格）和 AIRS‑Bench（6 个 AI 研究任务，分子、时间序列、文本分类）。

**📈 对比分析**

与现有自治 AI 构建系统（MARS、ML‑Master、InternAgent、R&D‑Agent、AIRA‑dojo、MLEvolve 等）和基准（MLE‑Bench 官方排行榜、AIRS‑Bench）对比。AIBuildAI‑2.5 在 MLE‑Bench 上获得 73.3% 的奖牌率，排名第一；在 AIRS‑Bench 的六项任务中均优于 MLEvolve；模型路由方案在保持或超越全前沿模型性能的同时，将推理成本降低约 50%。

**⚠️ 局限性**

局限性包括：① 资源调度器仍基于启发式规则，无法充分利用 GPU 共享与细粒度资源；② 训练成本高导致评估次数受限，仍需更高效的评估或模拟；③ LLM 评判与选择过程对提示设计敏感，鲁棒性待提升；④ 系统在极长任务或大规模数据集上仍可能受限于硬件预算。

---

## 236. LatentPort: Beyond KV Cache - Cross-Model Transfer of Recurrent Memory in Hybrid Language Models: A 4B-to-9B Hybrid-State Handoff Without Target Prefix Replay

**arXiv ID:** 2609.25053 | [PDF](https://arxiv.org/pdf/2609.25053v1)

**作者:** Simon P. Villani `[一作]` `[通讯]`, Simon P. Villani

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Qwen3.5 4B与9B的兄弟模型之间，首次实现了跨模型持久递归（GDN）状态的无前缀重放交接，证明混合注意力-递归模型的完整持久状态可在不同规模模型间直接迁移并保持推理性能。

**💡 创新点**

突破性在于：①跨模型迁移完整递归状态而非仅KV；②发现直接复制GDN递归与卷积状态优于学习翻译映射；③通过小规模残差校正进一步逼近大模型原始推理质量。

**🔧 技术方法**

使用了GDN（门控DeltaNet）递归记忆、KV翻译映射、岭回归映射、残差校正、NLL/JS等评估指标；在Qwen3.5 Base框架下实现状态迁移与验证。

**📊 数据集**

FineWeb‑Edu（用于翻译映射校准）、PG19（64本书用于教师强制评估）和Fresh FineWeb‑Edu（用于最终测试）。

**📈 对比分析**

对比标准基线（原始9B、原始4B、空9B、KV‑only、全翻译、直接GDN复制）和校正后版本。校正后9B在64个评估文档上平均NLL降至1.989 nats/token，低于继续4B（2.042）且接近原始9B（1.913），JS和Top‑1提升显著，表明性能与原始大模型高度一致。

**⚠️ 局限性**

仅在Qwen3.5 4B→9B的匹配几何下验证；未评估自由生成、跨模型或不匹配几何情况；使用教师强制评估可能高估实际效果；翻译映射与校正仅针对该模型对，泛化性未知；未考虑运行时延迟与系统成本。

---

## 237. Towards Adaptive Interaction Strategies for Human Companion Robot via Deep Reinforcement Learning

**arXiv ID:** 2609.25031 | [PDF](https://arxiv.org/pdf/2609.25031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 238. Tipping Points in LLM-Based Multi-Agent Systems: Stance on Climate Change Action

**arXiv ID:** 2609.25432 | [PDF](https://arxiv.org/pdf/2609.25432v1)

**作者:** Astghik Altunyan `[一作]` (Cornell University), Shimon Edelman `[通讯]` (Cornell University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建LLM驱动的代理基础模型，研究少数承诺型代理是否能在气候行动议题中引发社会临界点，并使用对话提取立场与主题建模来分析社会动态。

**💡 创新点**

创新点在于将大语言模型赋能的代理与社会临界点框架相结合，并利用LLM作为“裁判”实时提取二维立场，同时揭示LLM偏差对社会动态的影响。

**🔧 技术方法**

使用的技术包括LLM驱动的代理交互、LLM-as-a-judge立场提取、LDA主题模型、欧氏跳跃距离与t检验，以及缺失值的Last Observation Carried Forward（LOCF）方法。

**📊 数据集**

数据集为约385M输入token的人工生成对话文本，涵盖10个代理的预设身份与自定义变量，未使用公开真实文本数据。

**📈 对比分析**

通过比较不同承诺少数比例（10% vs 20%）和话题数（k=2,3,5）的实验，发现早期轮次出现显著立场跳跃，但整体变化受到LLM偏差抑制；主题分布显示缺失机构信任话题。

**⚠️ 局限性**

局限性包括单次高成本运行、LLM偏差导致的同质化与话题缺失、传统词袋主题模型的局限、缺乏多次重复实验、未采用更先进的Transformer主题模型，以及仅用简单统计方法识别临界点。

---

## 239. Norm2Tex: Augmenting Visuo-Tactile Simulations with Texture

**arXiv ID:** 2609.25398 | [PDF](https://arxiv.org/pdf/2609.25398v1)

**作者:** Seongjin Bien `[一作]` (University of Technology Nuremberg), Wolfram Burgard `[通讯]` (University of Technology Nuremberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种可插拔插件，利用法向图向视觉触觉模拟器注入高频表面纹理细节，提升仿真数据与真实数据的一致性。

**💡 创新点**

创新点在于：在深度图渲染阶段直接对法向图进行修正，轻量级且无须改动仿真核心；同时提供基于Blender的可扩展程序化纹理生成管线。

**🔧 技术方法**

主要技术包括：UV法向图生成、基于Fourier的Poisson积分重建高度图、GPU加速纹理查询与叠加，以及与TACTO、Taxim等深度图渲染触觉仿真器的对接。

**📊 数据集**

使用了DIGIT真实触觉数据集（5类材质各100样本）以及在MuJoCo环境中生成的6k+仿真渲染样本。

**📈 对比分析**

与无纹理仿真相比，零射击材质分类准确率提升至约72%；在抓取力调节的强化学习任务中，含纹理的策略展现了材料相关的开合阈值差异，显示出更好的仿真到真实的转移。

**⚠️ 局限性**

局限性包括：生成的纹理在频谱上仍与真实数据存在差距，尤其是柔性材质与光照阴影效果；仅在深度图渲染器中测试，未扩展到更复杂的FEM/MPM物理仿真器。

---

## 240. Parallel Integration over Simple Radical Extensions in Mixed Towers: Charlwood's Integrals

**arXiv ID:** 2609.25616 | [PDF](https://arxiv.org/pdf/2609.25616v1)

**作者:** Sam Blake `[一作]` `[通讯]`, Sam Blake

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 Charlwood 2008 设计的 50 个挑战性积分进行系统性符号积分，验证并比较不同实现的效果；

**💡 创新点**

提出了并行（Risch–Norman）方法，并在 SymPy 中实现，证明了对该类积分的完整性与可证性；

**🔧 技术方法**

使用并行 Risch–Norman 算法、S‑单位、曲线单位、切线数论、残差多项式、重试阶梯等技术；

**📊 数据集**

利用 Charlwood 的 50 个积分基准（10 个主积分 + 40 额外积分），并将其整理为脚本自动化测试；

**📈 对比分析**

与 FriCAS（Risch–Trager–Bronstein 实现）和 AXIOM 进行对比：平行方法在所有可积分项上给出正确答案并给出非可积证明，速度平均 1.4 s、最长约 9 s，FriCAS 速度更快但在若干积分给出错误答案；

**⚠️ 局限性**

局限在于 S‑单位完备性的缺失（导致 A39 被标记为失败）、搜索上限与重试阶梯的经验性参数，以及对高阶数分数分式的非线性搜索开销。

---

## 241. SG-CPG: Severity-Gated Central Pattern Generators for Adaptive Quadruped Locomotion under Continuous Actuator Degradation

**arXiv ID:** 2609.25687 | [PDF](https://arxiv.org/pdf/2609.25687v1)

**作者:** Adarsh Kumar Kosta `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于中心模式生成器的四足机器人控制器SG-CPG，能够在关节持续退化时通过severity-gated residual和amplitude gate实现全身重新协调与步幅缩小，从而保持步态并提高生存率。

**💡 创新点**

创新点在于：①将severity信息门控于残差策略与振荡器幅度，实现在同一policy内对连续退化的渐进适应；②区分并分析torque‑ceiling与gain‑scaling两种退化机制的可观测性；③在单一受损关节的情境下实现全身协调而非单腿专属策略。

**🔧 技术方法**

使用技术包括：CPG-RL振荡器网络、PPO强化学习、残差策略、severity gate、amplitude gate、模拟与硬件实验、可观测性分析。

**📊 数据集**

使用数据：Unitree Go2仿真环境（Isaac Lab）、真实硬件实验；训练期间生成随机命令与随机退化；评估集包含直线扫荡与全方向扫荡、额外扰动测试。

**📈 对比分析**

比较方法：与冻结健康CPG、无severity residual、无amplitude gate等变体进行对照；在模拟中，在95%关节强度损失下保持100%行走成功率、8%追踪误差、CoT 13%以内；在硬件上28/29次前进/转弯成功，最大严重度0.93；整体性能优于基线且能维持多方向运动。

**⚠️ 局限性**

局限性：仅评估单一受损关节，严重退化（>95%）时会出现失稳；估计器对轻微退化易误判；未考察多关节并发退化情况；对硬件模型匹配敏感，若关节摩擦等参数差异大，效果下降。

---

## 242. HOTICE: Whole-Body Humanoid Object Transportation in Cluttered Environments

**arXiv ID:** 2609.25363 | [PDF](https://arxiv.org/pdf/2609.25363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 243. ImIR: Image-Instruction Tuning for All-in-One Image Restoration

**arXiv ID:** 2609.25267 | [PDF](https://arxiv.org/pdf/2609.25267v1)

**作者:** Süleyman Aslan `[一作]` (Codeway AI Research), M. Akın Yılmaz `[通讯]` (Codeway AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于预训练图像编辑模型的全能图像恢复方法，利用低秩适配器和图像指令实现多任务恢复，完全不需要文本提示。

**💡 创新点**

创新点包括：① 用图像生成的连续指令替代离散文本提示，直接将退化图像的视觉语言嵌入转化为恢复指令；② 轻量级 token mapper（含均值池化、FiLM、位置无关 MLP）预测清晰图像的嵌入；③ 通过指令缩放实现可控恢复，支持任务无关恢复。

**🔧 技术方法**

技术细节：预训练的 Qwen-Image-Edit + Qwen2.5-VL 视觉语言编码器；低秩适配器（LoRA）；token mapper（FiLM + 位置无关 MLP + 均值池化）；流匹配（flow‑matching）训练；图像指令缩放控制。

**📊 数据集**

使用六个标准恢复任务的数据集：LOLv2-real（低光增强），Rain100L（去雨），RESIDE SOTS（去雾），GoPro（去模糊），SIDD（降噪），Kodak（JPEG 压缩）；总计 688 对训练样本。

**📈 对比分析**

与零射编辑器、文本提示 LoRA、Edit2Restore、全量训练专家等做对比；在同一预训练模型上，图像指令 LoRA 在所有六个任务的 PSNR/SSIM/LPIPS 均优于文本 LoRA，低光任务提升尤为显著；任务无关版本与任务感知同等；相比全量训练专家，虽然在部分任务上略逊，但训练仅 3 小时、参数量小且实现了任务无关与可控恢复。

**⚠️ 局限性**

局限性：推理速度慢（约 14 秒/百万像素）；依赖预训练生成模型，可能产生不可信或过度平滑细节；未做盲/真实世界泛化评估；对多重退化的处理能力未验证；整体性能仍低于一些全量训练专家在高质量任务上的表现。

---

## 244. A bioinspired internal model-based online estimator for planar pursuit

**arXiv ID:** 2609.25470 | [PDF](https://arxiv.org/pdf/2609.25470v1)

**作者:** Tengyue Liu `[一作]` (University of South Florida), Udit Halder `[通讯]` (United States Naval Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于内部模型的估计器，利用间歇性距离与方位测量重建追踪者与被追者的相对姿态及被追者未知运动，并将估计结果用于常定方位追逐和相互运动隐身控制。

**💡 创新点**

将内部模型与Pontryagin最大原理相结合，首次实现对相对状态与未知输入的联合估计；提出前向-后向迭代求解与移动窗口在线实现；在不观测被追者方向和控制的条件下实现准确重建。

**🔧 技术方法**

使用Pontryagin最大原理、前向-后向迭代求解、移动窗口（moving horizon）在线估计、MATLAB ode45数值积分、对比扩展卡尔曼滤波（EKF）和粒子滤波（PF）。

**📊 数据集**

采用合成LiDAR测量噪声（仿真）以及真实TurtleBot3 Burger机器人实验，真值通过OptiTrack运动捕捉系统获取。

**📈 对比分析**

与EKF、PF在开环估计和闭环追逐下比较；IME在误差分布、收敛速度和对初始猜测的鲁棒性上均优于两者，误差更小且范围更窄。

**⚠️ 局限性**

对目标动力学假设有限，算法对参数敏感；计算量较大，实时实现需进一步优化；仅在平面单目标情形，需扩展到三维或多目标设置。

---

## 245. Digital Twin-Driven VR Teleoperation with Multi-View Spatial Perception for Surgical Robots

**arXiv ID:** 2609.25527 | [PDF](https://arxiv.org/pdf/2609.25527v1)

**作者:** Chang Liu `[一作]` (Johns Hopkins University), Peter Kazanzides `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于数字孪生的VR远程手术平台，用Meta Quest 3S实现无台式机的机器人辅助微创手术（RMIS），通过多视角渲染和VR手柄控制实现精确的器械操作。

**💡 创新点**

创新点包括：①使用实时无标记感知（SAM+FoundationPose）构建数字孪生，替代传统视频传输；②采用基于clutch的相对映射解耦手部姿态与器械，打破物理手-器械对应；③通过状态同步架构实现可记录、可离线多视角重放，支持数据收集与训练。

**🔧 技术方法**

技术栈：机器人状态流（UDP JSON）+ Unity3D渲染；外部RGB‑D相机 + Vision Foundation Models（SAM 2/3、FoundationPose）实现物体检测与6DoF姿态估计；Meta Quest 3S 6DoF手柄控制；手眼标定、低延迟控制循环、离线数据日志与多视角重放。

**📊 数据集**

数据集/实验：15名受试者完成抓取环放置任务，采集机器人运动数据、RGB‑D视频流及手柄输入；无使用公开数据集，全部为自定义实验数据。

**📈 对比分析**

比较方法：与传统固定台式机（dVRK MTM）和HoloLens 2 MR（手势+虚拟杆）进行对比；客观指标包括任务完成时间、路径长度、运动效率、平滑度（jerk）以及抓取次数；主观Borda评分。结果显示：VR平台相比MR在路径长度减少86%、jerk降低95%；与MTM相比在路径、效率、平滑度上显著优于或相当；抓取次数最低，说明深度感知最佳。

**⚠️ 局限性**

局限性：①数字孪生仅适用于刚性物体，软组织感知难度大；②依赖外部RGB‑D摄像头，临床常规无此设备；③场景孪生更新率低于机器人，需加速优化；④未在真实手术环境中验证，需进一步评估临床可行性。

---

## 246. SurgGaze: Implicit Calibration for Accurate Gaze Analysis in Operating Rooms with Wearable Eyetrackers

**arXiv ID:** 2609.25612 | [PDF](https://arxiv.org/pdf/2609.25612v1)

**作者:** Jingying Wang `[一作]` (University of Michigan), Xu Wang `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了 SurgGaze 机制，用于在真实手术室中对可穿戴眼动仪进行隐式校准；

**💡 创新点**

创新点是利用手术过程中工具-组织接触点作为真实注视点进行校准，避免单独校准流程；

**🔧 技术方法**

使用线性回归校准模型、工具-组织接触检测、前置摄像头同步；

**📊 数据集**

数据集包括模拟 OR 下 14 位受试者的光学跟踪数据、22 名受试者的视频跟踪、以及 5 个真实 lap chole 手术的视频；

**📈 对比分析**

通过对比未校准、9点显式校准和 SurgGaze 隐式校准，发现后者在模拟实验中将误差降低 40.6%，在真实手术中将关注区域覆盖率从 11.34% 提升至 85.75%，提升约 8 倍；

**⚠️ 局限性**

局限在于需要手动标注接触点、对小目标和非中心屏幕区域校准效果有限、仅测试两种设备且样本多样性不足。

---

## 247. Signed Graph Pre-Training and Prompt Learning

**arXiv ID:** 2609.25722 | [PDF](https://arxiv.org/pdf/2609.25722v1)

**作者:** Zihan Mei `[一作]` (Arizona State University), Yixuan He `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了TopoSIGN框架，将磁性有向正负图神经网络结构嵌入与基于Dowker复形的持久同调拓扑特征结合，先在链路符号预测上自监督预训练，再通过聚类提示迁移到节点聚类任务。

**💡 创新点**

创新点在于：①设计了正负度向量筛选的Signed Degree‑Vector Filtration，既保留正负度信息，又可直接用于一参数持久同调；②将拓扑特征融入Prompt Learning，首次将拓扑引导的预训练与提示学习应用于有符号图；③构建了可兼容多种有符号GNN骨干的通用框架。

**🔧 技术方法**

技术手段包括磁性正负拉普拉斯（MSGNN、SSSNET）图神经网络、Dowker复形持久同调、持久图像（Persistence Image）向量化、GPPT式Prompt Learning与聚类提示（Cluster Prompt）。

**📊 数据集**

使用合成SDSBM四个噪声设定以及真实世界的Rainfall和SP1500两个有符号图数据集进行实验。

**📈 对比分析**

通过与GPPT、Gprompt、GPF、All‑in‑one、SAMGPT、TopoDIG等预训练/提示方法以及SSSNET、SigMaNet、MSGNN、DSGC等有符号GNN聚类模型比较，TopoSIGN在噪声较大（SDSBM‑1/2/3）设置下取得ARĪ最高或近最高成绩，在真实数据上表现竞争力，尤其是拓扑分支显著提升了有符号GNN的聚类效果。

**⚠️ 局限性**

局限性包括：①目前的Signed Degree‑Vector Filtration只能捕捉一种正负度信息，缺乏对入/出正负度或边属性的更细粒度滤波；②在密集金融网络（SP1500）上的性能仍有限；③对大型图的可扩展性仍需改进（如更高效的Landmark选择）。

---

## 248. Direct Optimization of Generators for Search in Automated Theorem Proving

**arXiv ID:** 2609.25575 | [PDF](https://arxiv.org/pdf/2609.25575v1)

**作者:** Adam Ousherovitch `[一作]` (University of Michigan), Ambuj Tewari `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两类针对搜索导向的大语言模型微调方法——搜索对齐训练（CAT）与搜索无关的统一分配（UA），以提升Lean理论证明搜索的成功率。

**💡 创新点**

创新点在于将计算对齐的损失从平坦搜索扩展到树搜索，构建基于演示轨迹的可追踪损失，并提出搜索无关的统一预算损失。

**🔧 技术方法**

技术包括计算对齐训练（CAT）、统一分配（UA）、基于示例轨迹的近似递归求解、梯度权重归一化，以及在Lean 4 上的政策搜索算法（Pass@N、BFS、DFS、MCTS）。

**📊 数据集**

数据集为Lean 4的mathlib4 458条定理（证明长度2–5步），使用Qwen2.5-Math-7B-Instruct预训练模型并通过LoRA微调。

**📈 对比分析**

与传统交叉熵训练对比，CAT与UA在六种搜索策略下均提升了成功率，特别是CAT在Pass@N上最高提升约4.6%，且预算增大时增益更明显。

**⚠️ 局限性**

局限性在于仅使用轨迹信息，未采样偏差或全局树探索，模型仅在离线SFT环境下训练，无法充分捕捉多路径成功与恢复成本。

---

## 249. From Instrument-Mounted Demonstrations to In-Vivo Execution: Learning Bimanual Laparoscopic Appendectomy Without Robot-Collected Demonstrations

**arXiv ID:** 2609.25625 | [PDF](https://arxiv.org/pdf/2609.25625v1)

**作者:** Dongho Yee `[一作]` (Seoul National University Hospital), Hyoun-Joong Kong `[通讯]` (Seoul National University Hospital)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在手持腹腔镜器械上安装传感器日志器，记录外科医生在真实手术中的姿态与关节状态，并用这些演示训练扩散式视觉动作政策，在两臂机器人上实现了在活体兔子上完成腹腔镜阑尾切除并能安全使用电刀。

**💡 创新点**

① 设计了无外部摄像/跟踪器、可直接挂在器械上的状态记录器，并对每条传感器通道进行精确的时延匹配；② 通过仿真闭环回放筛选31种政策配置，找出最安全、最高效的设计；③ 证明仅凭手持手术器械的记录即可训练出可在活体动物上执行电刀的双臂政策。

**🔧 技术方法**

使用基于 DINOv3 ViT 的冻结后最后一块Transformer的扩散式政策（Diffusion Policy），配合时间步长为10Hz的动作块、FiLM条件下的阶段编码、基于 Isaac Sim 的柔性组织仿真以及多通道时延补偿的传感器处理链。

**📊 数据集**

两个演示数据集：ex‑vivo 533 次（74 分钟）阑尾切除演示，in‑vivo 849 次（135 分钟）活体兔子阑尾切除演示；全部数据公开至 GitHub。

**📈 对比分析**

在 Isaac Sim 中对 31 种候选配置进行 16 场闭环回放，评估抓取、提升、到位、闭嘴、切割等阶段完成率与气腹杆违约率；选出的配置在真实两臂 FR3 机器人上无气腹杆违约，且在 4 只活体兔子中 3 只成功完成切除，平均控制时长 23–34 秒，RCM 误差 < 0.84 mm，电刀触发安全性 100%。

**⚠️ 局限性**

① 需要人工输入手术阶段，视觉阶段识别精度低；② 实验样本有限，仅 4 只示范兔和 4 只部署兔，且只在同一天完成；③ 仍需外科医生在必要时接管，未实现完全自主；④ 只验证了阑尾切除，无法推广到更复杂手术。

---

## 250. Relative Contact Velocity-Controlled Hand-Object Mechanism for Dexterous Tool Manipulation

**arXiv ID:** 2609.25619 | [PDF](https://arxiv.org/pdf/2609.25619v1)

**作者:** Sunyu Wang `[一作]` (Carnegie Mellon University), Nancy S. Pollard `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一套基于手‑物体机制（Hand‑Object Mechanism）的框架，能够让不同类型的多指机器人手在模拟环境中完成抓取、装载和使用工具的完整过程；

**💡 创新点**

创新点在于将手与工具视为统一的并联机构，引入通用接触帧（Generalized Contact Frames）和子组件（sub‑assemblies）来描述相对接触速度与力，利用最小二乘求解和互补滤波实现轻量级、可解释的运动规划与接触估计；

**🔧 技术方法**

技术包括基于刚体滚动‑滑动接触动力学的最小二乘运动规划、互补滤波接触点估计、Drake 物理仿真与 Hydroelastic 接触模型、Tetgen 网格分解等；

**📊 数据集**

论文未使用公开数据集，而是通过对五款不同结构（Orca、Wuji、Sharpa Wave、定制 A 与 B）的机器人手进行仿真实验进行验证；

**📈 对比分析**

对比方法为在相同的抓取/旋转/装载任务下，对所有手的相对接触速度、物体运动误差（RMSE）进行量化；结果显示五种手均能在相同简易指令下实现类似轨迹与精细操作，表现稳定；

**⚠️ 局限性**

局限性包括需预先设定合适的预抓姿态、受非完整约束的速度规划限制，且目前仅在仿真中验证，实际硬件实现需解决高频触感测量与传感融合问题。

---

## 251. PARTE: Plane-Assisted Robust Transformation Estimation for Point Cloud Registration

**arXiv ID:** 2609.25375 | [PDF](https://arxiv.org/pdf/2609.25375v1)

**作者:** Abolfazl Babanazari `[一作]` (Colorado School of Mines), Kaveh Fathian `[通讯]` (Colorado School of Mines)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

PARTE 提出了一个全局点云配准框架，将平面片段作为补充匹配证据与点特征结合，通过一致性图最大权重团进行鲁棒外点剔除，并最终估计刚体变换。

**💡 创新点**

创新点包括：① 设计了 Plane Context Histogram (PCH) 通过平面周围几何信息描述平面片段；② 将点对应和平面对应一起构成加权一致性图，利用平面对应的置信度进行权重化，并通过最大权重团实现联合外点剔除；③ 当平面信息不足时自然退化为传统点匹配。

**🔧 技术方法**

技术要点：voxel 下采样、法向估计、贪心区域生长平面提取、FPFH 点特征、PCH 平面描述符、双层最近邻匹配、加权最大团求解（PMC）、块坐标下降 (BCD) 估计刚体变换。

**📊 数据集**

使用了六个主流数据集：3DMatch、3DLoMatch、KITTI-10m、KITTI-LC、RESSO (partial-to-full) 与 ETH，覆盖室内 RGB‑D、户外 LiDAR 以及部分全景配准场景。

**📈 对比分析**

与 13+ 传统、鲁棒、几何/语义及学习型方法（RANSAC、TEASER++、CLIPPER+、KISS‑Matcher、G3Reg、PREDATOR 等）进行对比，PARTE 在 8,097 对配对中实现了最高总体成功率，并在 3DMatch、3DLoMatch、KITTI-10m、KITTI-LC 四大基准上取得最高平均成功率，同时保持较低的端到端运行时间。

**⚠️ 局限性**

局限性：① 仍受点特征匹配的瓶颈影响；② PCH 对平面周围几何和点云密度变化敏感，导致在 partial-to-full 或大密度差异的场景中表现下降；③ 当可用平面信息稀少时，方法退化为点匹配，无法充分利用平面信息。

---

## 252. ICDAR2026 Competition on Multimodal Reasoning over Documents in Multiple Domains

**arXiv ID:** 2609.25055 | [PDF](https://arxiv.org/pdf/2609.25055v1)

**作者:** Artemis Llabrés `[一作]` (Universitat Autònoma de Barcelona), Dimosthenis Karatzas `[通讯]` (Universitat Autònoma de Barcelona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究组织了ICDAR2026多域文档视觉问答竞赛，评估并比较了多模态推理模型在八大文档域上的性能。

**💡 创新点**

创新点在于提出更具挑战性的多跳、时空推理任务，并通过结构化证据检索、验证与多组件协同的设计显著提升模型性能。

**🔧 技术方法**

采用多种技术路线：零样本VLM、OCR/解析增强、检索式agent、Multi‑Agent集成以及微调VLM，并结合布局解析、文本检索和视觉检索等方法。

**📊 数据集**

使用DocVQA2026数据集，覆盖业务报告、科学论文、幻灯片、海报、地图、漫画、信息图与工程图共八类文档，共80/160道验证/测试问题。

**📈 对比分析**

与官方基线（Gemini、Qwen、GPT‑5等）相比，顶级系统在整体准确率达60%（>35B参数）并在多域表现更均衡；零样本模型准确率低至1‑30%，显示结构化方法显著优于单一推理。

**⚠️ 局限性**

局限性包括地图、工程图与漫画等域仍难以处理，方法对不同域的适应性不一致，且评测依赖严格匹配，难以全面衡量开放式回答的多样性。

---

## 253. A Case Study in Accessible Redesign of a Wastewater Dashboard

**arXiv ID:** 2609.25273 | [PDF](https://arxiv.org/pdf/2609.25273v1)

**作者:** Tingying He `[一作]` (University of Utah), Paul Rosen `[通讯]` (University of Utah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在犹他州卫生与人类服务部的废水监测 dashboard 上进行案例研究，重新设计其信息架构、交互、可访问性和移动适配，以满足 2024 年 ADA 更新的可访问性要求并提升对色弱、盲/低视力用户的可用性；

**💡 创新点**

创新点包括：采用表格-first 的信息布局、实现地图与表格的双向关联、精简 Plotly 工具栏、手动生成可访问的 alt 文本与数据下载、开发专用 R 包支持 Shiny 组件互联、将摘要表改为卡片式移动视图，并在评估中考虑屏幕阅读器体验；

**🔧 技术方法**

技术手段涵盖：R Shiny、Plotly、Leaflet、定制 R 包、颜色与图例设计、响应式移动布局、手工 alt 文本编写与可访问性改造；

**📊 数据集**

使用的数据集为犹他州 35 个废水采样站的 SARS‑CoV‑2 浓度、趋势、变异株信息及相关公共卫生数据；

**📈 对比分析**

通过与原始 dashboard 的半结构化访谈评估和盲研究员对比评估，验证了可访问性与可用性的提升，但未进行正式的 WCAG 合规测试，也未量化性能指标；

**⚠️ 局限性**

限制包括：受政府工作流程与资源限制，未实现动态 alt 文本和正式 WCAG 评估，仅基于单一盲研究员反馈，缺乏大规模可访问性测试，并且 LLM 生成 alt 文本被禁止，导致改进受限。

---

## 254. Contact-Stable Deformable Tissue Simulation Using Implicit Integration and Live-Pose Grasp Constraints for Laparoscopic Surgery Robot Policy Evaluation

**arXiv ID:** 2609.25642 | [PDF](https://arxiv.org/pdf/2609.25642v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 255. Potential for Enhanced Learning in Machine Learning Classes by Using Wiki LLM Indexing

**arXiv ID:** 2609.25303 | [PDF](https://arxiv.org/pdf/2609.25303v1)

**作者:** Brian Wright `[一作]` `[通讯]` (University of Virginia), Brian Wright (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一门真实机器学习课程语料库上，对比了向量检索增强生成（Vector RAG）和在摄取阶段构建的LLM编译wiki两种知识表示方式，评估其问答质量和根基度。

**💡 创新点**

创新点在于首次证明摄取时结构化的wiki表示能显著提升跨章节合成题的回答准确性与可追溯性，并减少生成模型的虚构风险。

**🔧 技术方法**

使用的技术包括mpnet文本嵌入与Pinecone索引的向量检索、Karpathy框架下的LLM wiki编译、RAGAS评价指标、LLM评判员评分和基准问答生成。

**📊 数据集**

数据集为DS3001/DS3021课程材料（讲义幻灯片、音频转录、阅读材料）与59道人工生成的课堂问题。

**📈 对比分析**

比较方法在保持生成器、提示、问题集不变的前提下，对两种知识表示分别进行回答并由LLM评判员给出1-10分和根基度，结果wiki在总体分数和根基度上均优于Vector RAG，尤其在跨页合成题上优势更大。

**⚠️ 局限性**

局限性包括：仅测试单门课程、未对Vector RAG进行针对本实验的超参数调优、使用单一评判员和单轮评分、未测量成本与延迟、结果对更大规模或不同学科的泛化性不足。

---

## 256. Pushout Attachments and Conditional Complexity of Executable Models

**arXiv ID:** 2609.25024 | [PDF](https://arxiv.org/pdf/2609.25024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 257. Sub-polynomial parameterized complexity of $k$-core

**arXiv ID:** 2609.25419 | [PDF](https://arxiv.org/pdf/2609.25419v1)

**作者:** Yan S. Couto `[一作]` (University of São Paulo), Cristina G. Fernandes `[通讯]` (University of São Paulo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究k-core（k-核心）问题在不同图类与参数化下的并行可解性，提供了多种下界与上界的算法与证明；

**💡 创新点**

创新点在于将k-core问题迁移到参数化并行复杂度框架，证明其在树宽、弦图和区间图上分别可在FNC^3、FNC^3和FNC^3级别求解，并给出相应的下界，揭示了该问题在这些图类中的并行难度边界；

**🔧 技术方法**

主要技术包括：1）在树宽下利用可满足逻辑(CMSO)与Courcelle定理的并行版本；2）在弦图中构造子图G'并利用树宽结果；3）在区间图中设计基于路径分解的动态规划，利用路径分解的中点递归与“锚定子图”概念；4）通过构造性归约证明下界；5）使用并行电路模型（FNC、L、XL等）与空间/时间复杂度的转化；

**📊 数据集**

论文未使用具体实验数据集，全部是理论分析与复杂度证明；

**📈 对比分析**

通过与已有的FNC^i、L、XL等并行/空间复杂度类别对照，展示在不同参数化下的最优/近似上界与下界，表明即使在弦图和区间图上也难以突破L或FNC^3；

**⚠️ 局限性**

局限性在于：1）对一般图（尤其是平面图）仍未给出完整复杂度；2）参数化上仍受树分解算法瓶颈；3）只考虑无向k-core，未扩展到逼近、定向或k-truss等变体；4）缺乏实验验证。

---

## 258. Semord: Learned Semantic-Preserving Placement and Low-Fanout Routing for Distributed Vector Search

**arXiv ID:** 2609.25514 | [PDF](https://arxiv.org/pdf/2609.25514v1)

**作者:** Shengze Wang `[一作]` (University of California Santa Cruz), Chen Qian `[通讯]` (University of California Santa Cruz)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和强化学习相结合的技术。

**📊 数据集**

采用了公开的图像识别数据集进行实验。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示新算法在准确率和速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现不佳，且对计算资源的需求较高。

---

## 259. The Probabilistic Structure of Large Language Models

**arXiv ID:** 2609.25134 | [PDF](https://arxiv.org/pdf/2609.25134v1)

**作者:** Adnan Aboulalaâ `[一作]` `[通讯]`, Adnan Aboulalaâ

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文以概率论为统一框架，对大语言模型（LLM）进行系统阐述，并在此基础上对扩散模型进行补充对比，强调训练为最大似然估计、生成为随机过程模拟，并讨论KL散度的非对称性与幻觉现象。

**💡 创新点**

创新点在于将LLM的统计估计、生成推断与KL散度的质量控制、扩散模型的score‑based 生成等关键概念整合为一套完整的概率视角，揭示生成过程中“模态寻求”与“质量覆盖”之间的根本关系，并提出“典型采样”与“信息熵阈值”等新的采样视角。

**🔧 技术方法**

核心技术包括：transformer 结构的自回归概率建模、softmax 与 logits 的处理、最大似然与前向KL正则化、随机梯度下降（SGD/Adam）优化、温度/截断（top‑k/top‑p/典型）采样策略、Gumbel‑max 采样，扩散模型侧则用高斯前向过程、去噪score匹配、离散/连续时间SDE 逆向采样等。

**📊 数据集**

论文以理论为主，没有给出具体实验数据；但在讨论中引用的常见训练语料为维基百科、通用文本语料库；扩散模型讨论的典型数据集为图像领域的 ImageNet、COCO 等公开图像数据集。

**📈 对比分析**

文中未给出实验对比与性能指标；主要通过理论推导和概念对比阐述模型的优势与局限，强调了在生成过程中KL散度方向导致的“质量覆盖”与“幻觉”现象，但缺乏定量评估。

**⚠️ 局限性**

主要局限包括：依赖前向KL导致的概率覆盖与幻觉，模型仅捕捉统计规律而非事实真相，缺乏对生成文本真实性的校正；训练需要海量数据与计算资源；论文本身为理论阐述，未提供实证验证，且对不同采样策略的具体效果缺乏定量分析。

---

## 260. Effects of Assistance Delay on Joint Mechanics and Energetics in Biological Torque Control of a Hip Exoskeleton

**arXiv ID:** 2609.25417 | [PDF](https://arxiv.org/pdf/2609.25417v1)

**作者:** Jimin An `[一作]` (Carnegie Mellon University), Inseung Kang `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在水平行走、爬坡上行和下行三种步态下，生物力矩控制的机器人髋部外骨骼在不同助力延迟（40–320 ms）对生物力学、代谢和主观感受的影响。

**💡 创新点**

首次系统评估助力延迟在不同步态中的效果，发现代谢收益对延迟几乎不敏感但机械卸载随延迟增大而减弱，并指出单一正功率映射在下行步态下无法提供显著效益，提示需根据受助关节的机械需求而非仅任务标签来调整映射。

**🔧 技术方法**

采用深度学习时间卷积网络估计即时髋关节力矩，再将其按固定比例（K=0.2）和可调延迟映射为外骨骼扭矩；实验使用COSMED K5间接热量计、Vicon运动捕捉系统、跑步机测力板，并在OpenSim中进行逆运动学和动力学分析。

**📊 数据集**

训练数据来自10名健康受试者在外骨骼协助行走时收集的三种步态数据；实验数据则由8名受试者在跑步机上完成的8种助力条件（含0延迟）收集。

**📈 对比分析**

通过三向重复测量ANOVA和线性混合效应模型比较代谢率、正髋关节功和下肢正功等指标。实验结果显示平均代谢率下降5.24%，正髋关节功下降5.86%，下肢正功下降1.68%；机械卸载随延迟增加而下降，而代谢效益基本不随延迟变化；在下行步态中未见任何条件显著降低任何指标。

**⚠️ 局限性**

局限性包括样本量仅8人、仅调节延迟而未探索比例或映射形状的优化、未直接测量负功或能量吸收，因而难以确定是否存在更优的映射参数，也限制了对不同任务的普适性判断。

---

## 261. DynaForge: Planning-Guided Residual Learning for Dynamic Manipulation Demonstration Generation

**arXiv ID:** 2609.25631 | [PDF](https://arxiv.org/pdf/2609.25631v1)

**作者:** Yiyang Jin `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 DynaForge，一种将低频全局规划与高频对象中心逆运动学相结合，并通过残差策略学习在动态交互中纠正规划动作的框架，用于大规模生成动态操作演示。

**💡 创新点**

创新点包括：① 采用阶段化规划优先，低频生成全局轨迹、在接触边界高频对象中心 IK；② 学习残差策略纠正规划动作，提升接触时的成功率；③ 引入隐式课程学习——在匹配难度条件的组内使用群组相对优势，只在混合成功/失败组上更新，自动把学习焦点推向难度边界，从而提高优化效率。

**🔧 技术方法**

使用技术：基于 Isaac Sim 的 Franka 机器人仿真、全局轨迹规划（如 RRT* 等）、高频逆运动学、残差策略学习（GRPO+群组相对优势、异向裁剪）、隐式课程筛选、3D Diffusion Policy（DP3）进行下游训练。

**📊 数据集**

数据集：在 Isaac Sim 上的 9 个动态操控任务（Can、Bottle、Toy car、Lemon、Alarm Clock、Peach、Block、Pen、Peg），以及在真实 Franka 机器人上的 3 个对应任务（滚动罐、转动闹钟、转盘香蕉）。

**📈 对比分析**

与 DynamicVLA、DMG、DOMINO 等基线对比：演示生成成功率从 41% 提升至 78%；DP3 在仿真中使用 800 条演示时，DynaForge 数据集得到 49% 的成功率，DOMINO 仅 7%；在真实任务上，DynaForge 训练的策略分别实现 30%、60%、50% 的成功率，而 DOMINO 仅 0%/0%/10%；此外，隐式课程学习使得优化步数减少 27%（0.73×）和训练时间减少 38%（0.62×）后仍取得更高的最终成功率。

**⚠️ 局限性**

局限性：仍需要手工定义任务阶段和规划目标，受限于仿真真实性，对插入等高精度任务的成功率偏低，对视觉观测和物理动力学变化的鲁棒性尚待提升。

---

## 262. Do Synthetic Personas Predict Real Audience Response? A Sim-to-Real Study Where a No-Persona Baseline Beats Persona-Based Copy Simulation

**arXiv ID:** 2609.25010 | [PDF](https://arxiv.org/pdf/2609.25010v1)

**作者:** Alexandre Cristovão Maiorano `[一作]` `[通讯]`, Alexandre Cristovão Maiorano

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了基于大语言模型的合成人物（synthetic personas）在预测真实受众点击率排序上的有效性。

**💡 创新点**

首次用真实A/B测试结果作为地面真值，构建 sim‑to‑real 有效性评估框架，并发现人物条件化反而降低预测准确性。

**🔧 技术方法**

采用 Gemini 系列 LLM 进行角色扮演式点击意图预测，比较十人合成人物面板与无人物零样本基线，使用 Kendall τ、Spearman ρ 和 top‑1 准确率等指标。

**📊 数据集**

使用公开的 Upworthy Research Archive（2013‑2015 英文新闻标题 A/B 测试），并在三个独立拆分及 MIND、Reddit 标题数据上复现。

**📈 对比分析**

通过可靠赢家过滤后比较两种方法，发现无人物基线在可靠赢家中 Kendall τ≈0.15，top‑1≈0.55，而人物面板仅 τ≈0.05，top‑1≈0.40，表明无人物基线显著优于人物面板。

**⚠️ 局限性**

受限于数据的可靠性、构念差距（点击意图与实际点击不同）以及仅验证文本标题，未评估多语言、广告文案等更广泛场景。

---

## 263. CableVLA: Simulation-Privileged Global-Local Representation Learning for Cable Routing

**arXiv ID:** 2609.25606 | [PDF](https://arxiv.org/pdf/2609.25606v1)

**作者:** Zhifei Teng `[一作]` (Huazhong University of Science and Technology), Yiqun Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种端到端的多模态视觉‑语言‑动作框架 CableVLA，用于线缆路由，结合全局拓扑建模与局部触觉反馈，实现高成功率的路径规划与执行。

**💡 创新点**

创新点在于：①利用模拟特权的 TopoHead 通过教师‑学生蒸馏将节点级物理信息与未来拓扑预测转化为仅 RGB 的因果上下文；②设计双分支 TacSense 触觉编码器，结合全场时间演化与局部税元动态，并以模拟事件监督实现复杂触觉事件识别；③采用硬门控力‑触觉残差网络，将触觉与腕力信息融入冻结的拓扑条件父策略，细化 8 步 arm‑gripper 动作。

**🔧 技术方法**

使用的技术包括：基于 π_0.5 的 VLA 模型、Diffusion Policy、LoRA 微调、教师‑学生蒸馏、Transformer 编码器、Wrist‑Wrench 估计、残差 MLP、模拟特权监督与事件标签。

**📊 数据集**

使用的数据集主要包括：2,004 条仿真演示（含节点物理、拓扑、触觉、事件信息），57 任务的触觉多任务数据，MuJoCo 与 Isaac Sim 的线缆路由环境，以及 RealMan RM75-6F 机器人实机收集的视觉、触觉与力/扭矩数据。

**📈 对比分析**

与 π_0.5‑V、π_0.5‑V+Topo、Diffusion Policy 和 SmolVLA 在 345 次 MuJoCo 评估（含 IID、OOD、光照、摩擦等扰动）对比，CableVLA 在核心套件中成功率提升至 84.9%（对比 62.6%），跨仿真转移成功率提升至 54%（对比 34%），实机路由成功率提升至 40%（对比 10%），显示显著性能优势。

**⚠️ 局限性**

主要局限包括：跨域转移仍受传感器光照、腕部刚度与线缆接触动力学差异影响；触觉数据量有限，难以覆盖极端接触；残差网络对目标域微调需求高，尚未实现完全零调优；抓取姿态误差和闭合不足仍导致失败。

---

## 264. From Experts to Sub-experts: Fine-grained Parameter-Efficient Fine-Tuning for MoE LLMs

**arXiv ID:** 2609.25655 | [PDF](https://arxiv.org/pdf/2609.25655v1)

**作者:** Zhentao Tan `[一作]` (Alibaba Group), Jieping Ye `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 NSFT，一个从专家级到子专家级的稀疏参数高效微调框架，专为 MoE LLM 设计。

**💡 创新点**

创新点在于将专家内部按通道分组选出子专家，并结合路由重要性与激活显著性进行精细稀疏调优，同时引入学习率缩放与动态梯度缩放来补偿更新幅度。

**🔧 技术方法**

使用了子专家分组、路由重要性评估、激活显著性打分、学习率缩放、梯度缩放与动态梯度更新等技术，并在 OLMoE‑7B 与 Ling‑mini‑16B 上实现。

**📊 数据集**

实验数据集包括医学（MMedC）、科学推理（SciRIFF）、检索增强问答（RAGQA）、数学推理（MATH）、代码生成（Code）等领域任务，以及通用基准 GPQA、MMLU‑Redux、C‑Eval、IFEval。

**📈 对比分析**

与全微调、LoRA、ESFT 等基线对比，NSFT 在域内平均分提升约 4–6%，仅使用约 5–6% 可训练参数，且保持与全微调相近的通用能力。

**⚠️ 局限性**

局限性包括对激活统计的依赖、动态梯度缩放更新频率需调节、子专家分组大小与阈值选择对性能影响显著，以及在极大模型规模下的计算与存储开销仍待进一步评估。

---

## 265. AkasicMEM: Governed Enterprise Memory for Agents

**arXiv ID:** 2609.25563 | [PDF](https://arxiv.org/pdf/2609.25563v1)

**作者:** Jeongmin Bae `[一作]` (GraphAI), Min-Soo Kim `[通讯]` (GraphAI)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一个名为 AkasicMEM 的企业级受治理记忆系统，专门解决源–记忆集成、记忆治理与授权连续性三大问题；系统通过记录记忆源的传承关系、合成访问策略并在检索时动态重评估，确保源级授权在记忆形成、再衍生与使用全过程中始终有效。

**💡 创新点**

创新点在于：① 将源数据与记忆统一管理并形成可追溯的传承图；② 通过政策组合与检索时的重评估实现授权连续性；③ 在单一执行平台（AkasicDB）上融合向量检索、图遍历与关系查询，显著降低跨系统开销；④ 设计了基于自然语言推理的自动去标识化流程与手动审核机制，实现受控去标识化。

**🔧 技术方法**

技术实现包括：AkasicMEM（记忆层）+ AkasicDB（统一向量-图-关系数据库）+ 图遍历 + 向量相似度搜索 + 关系查询 + 权限图 + 自然语言推理（NLI）用于去标识化 + 递归式的授权重评估与批量策略重组。

**📊 数据集**

论文使用了合成的企业组织场景（包含多种业务来源、层级化权限结构及动态授权变化）来评估系统性能与安全性；未给出真实公开数据集，但通过模拟场景验证了泄漏、过度限制、检索相关性等指标。

**📈 对比分析**

评估方法：与无治理、仅作用域隔离、仅共享治理等基线进行对比，测量泄漏率、过度限制比例、检索相关性、实时成本（形成、检索、传播）以及随主体数、授权图深度、记忆量扩张的扩展性；实验结果显示 AkasicMEM 在保留大部分授权时显著降低泄漏，且成本相对基线可控，尽管在极深传承或高并发情况下会出现轻微延迟。

**⚠️ 局限性**

局限性包括：① 缺乏对记忆实用性（选择、衰减）与反馈循环的完整实现；② 去标识化依赖 NLI 与人工审核，可能导致误判或审核瓶颈；③ 评估高度依赖合成场景，真实业务复杂度可能导致更多细粒度权限与多源交叉冲突；④ 大规模部署时，批量重评估与向量检索的性能与一致性仍需进一步优化。

---

## 266. Induced Riemannian Metrics for Motion Planning with Constraints

**arXiv ID:** 2609.25695 | [PDF](https://arxiv.org/pdf/2609.25695v1)

**作者:** Phone Thiha Kyaw `[一作]` (University of Toronto), Jonathan Kelly `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种统一的度量框架，用来描述在约束子流形上的运动规划问题，并验证了隐式（约束等式）和显式（参数化）两种表示方式在任意 Riemannian 环境下诱导相同的度量，从而实现了度量与表示方式的解耦。

**💡 创新点**

创新点在于：①证明了两种子流形表示在任何 Riemannian 度量下都产生相同的诱导度量；②将该度量直接嵌入到采样式规划器（G‑RRT*）和轨迹优化器（Drake KinematicTrajectoryOptimization）中，保证两种算法优化的是同一 Riemannian 长度；③通过实验验证该方法在多臂闭环约束问题上的有效性，并对比了欧氏、动能以及“平坦”度量三种情况。

**🔧 技术方法**

主要技术包括：Riemannian 流形理论、约束子流形的隐式/显式建模、诱导度量的推导、欧氏/动能度量的应用、采样式规划器的重排与距离近似、B‑spline 轨迹优化以及 TOPP‑RA 时序化。

**📊 数据集**

使用的数据集为两臂 Franka 机械臂在六个闭环约束任务（书本从一个书架格子移动到另一个格子）下的模拟与真实实验数据，包含 50 次随机种子实验和 5 次真实运行实验。

**📈 对比分析**

对比方法：在采样式规划器和轨迹优化器中分别使用欧氏、动能以及忽略诱导度量的平坦度量进行规划和优化；性能评估包括路径 Riemannian 长度、优化求解时间、真实运行时间、关节空间路径长度以及累计动能。实验表明：①使用诱导度量规划得到的路径在对应度量下最短；②平坦度量规划得到的路径在所有诱导度量下最长；③动能度量规划得到的路径耗能最低，虽然运行时间略长；③优化器在初始路径与目标度量一致时收敛最快。

**⚠️ 局限性**

局限性：重排投影沿欧氏法线实现，仅在欧氏度量下是二阶精度；在一般 Riemannian 度量下仅为一阶，未来工作计划开发度量加权投影以恢复二阶精度；此外，轨迹优化目前仅在显式参数化下实现，未探讨在隐式表示下直接优化的可行性。

---

## 267. NeuMark: Neural Codec Resynthesis-Robust Audio Watermarking in the Codec Latent Space

**arXiv ID:** 2609.25719 | [PDF](https://arxiv.org/pdf/2609.25719v1)

**作者:** Annan Wu `[一作]` (Nagoya University), Tomoki Toda `[通讯]` (Nagoya University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种名为 NeuMark 的音频水印框架，能在神经音频编码器的残差向量量化（RVQ）层中嵌入 16 位可恢复消息，并在水印检测与信息恢复上保持鲁棒性。

**💡 创新点**

创新点在于：①通过 Transformer 交叉注意力将水印信息均匀注入 SpeechTokenizer 的多层 RVQ 代码，提升对神经码器重编码攻击的抵御能力；②引入重建参考与原始参考两种训练目标，揭示潜在的透明度-鲁棒性权衡；③将 DSP、遮蔽与 EnCodec 复原等多种扰动同时纳入训练，显著提升实际攻击下的检测与位恢复性能。

**🔧 技术方法**

使用了 SpeechTokenizer（多层 RVQ 的神经音频编码器）、Transformer‑基交叉注意力编码器、跨尺度 Mel‑谱损失、对抗判别器、以及自定义的水印检测与信息恢复损失。

**📊 数据集**

训练数据来自 LibriTTS（train‑clean‑100/360/500）；评测数据使用 LibriSpeech test‑clean（2620 句子）。

**📈 对比分析**

与基线方法（WavMark、AudioSeal、Timbre Watermarking、TraceableSpeech、VoiceMark）在 DSP 编辑与多种神经码器（EnCodec、DAC、WavTokenizer）重编码攻击下对比，NeuMark 在神经码器攻击中实现了 0.94+ 检测率和 0.96+ 位恢复率，远超对比基线；在 DSP 攻击下亦保持 0.95+ 的鲁棒性；在重建参考评价中获得最高 PESQ/Si‑SNR，展示了优异的音质与鲁棒性平衡。

**⚠️ 局限性**

局限性包括：未与完整 TTS 流程整合，仅在预训练的 SpeechTokenizer 上实验；对极端压缩（如 WavTokenizer）仍表现欠佳；多语种与嘈杂环境下的鲁棒性尚待验证。

---

## 268. Learning to Plan in Human-Robot Collaboration: Multimodal Reinforcement Learning for Adaptive Interaction

**arXiv ID:** 2609.25274 | [PDF](https://arxiv.org/pdf/2609.25274v1)

**作者:** Afagh Mehri Shervedani `[一作]` (University of Illinois Chicago), Miloš Žefran `[通讯]` (University of Illinois Chicago)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了一种基于强化学习的多模态人机交互管理器，使机器人能够在家中协助老年人或残障人士寻找物体，处理语言、手势和物理动作。

**💡 创新点**

创新点在于使用神经网络用户模拟器和DAGGER预热的RL框架，自动学习可解释且可扩展的交互策略，取代传统手工构建的HBATN，显著提高了任务效率和用户满意度。

**🔧 技术方法**

采用 DQN+DAGGER 强化学习、行为克隆的用户模拟器、误差注入模块、ALBERT 语义分析器、Google Speech‑to‑Text、Pyttsx3 文本转语音，以及 Baxter 机器人执行硬件。

**📊 数据集**

使用 ELDERLY‑AT‑HOME 语料库中的 Find 任务数据（约 7,617 条对话行动标签）来训练用户模拟器和对话行动分类器，并在此基础上生成模拟交互。

**📈 对比分析**

通过与手工 HBATN 系统的对比，利用 12 名受试者共 75 次实验评估成功率、平均回合数、系统准确率、语音识别错误率和指向错误率等指标；RL 系统成功率达 96%（高于 HBATN 的 85.7%），平均回合数降至 12.6（低于 15.6），错误率显著下降，整体性能优于传统方法。

**⚠️ 局限性**

局限性包括：需要为每个新任务构建并训练专门的用户模拟器，任务泛化受限；对 HBATN 的比较受限于可获得的数据；未在物理 HRI 场景下验证；错误注入的泛化能力仍有提升空间。

---

## 269. How Children Design and Reason about Trustworthy AI Chatbots

**arXiv ID:** 2609.25244 | [PDF](https://arxiv.org/pdf/2609.25244v1)

**作者:** Deniz Ozturk `[一作]` (North Carolina State University), Xiaoyi Tian `[通讯]` (Kennesaw State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过构建自定义对话机器人平台（LUMI）并在三种不同年龄段的学习者中进行工作坊，研究儿童在设计、配置和评估自制聊天机器人的过程中如何理解和实践可信度。

**💡 创新点**

首次将儿童定位为AI设计者，揭示七个可信度维度，并发现年龄与聊天机器人领域（学术vs休闲）对信任设计的影响，提出可信度并非单一特征，而是多维交互的结果。

**🔧 技术方法**

采用可调节的信任相关特征（信心、透明度、正式度、主张度）与知识库上传、模型选择、温度控制、错误处理等功能的可视化编程环境，配合日志记录、访谈与同行评估等混合方法。

**📊 数据集**

收集了115名儿童（8–18岁）在三种教育场景中的实验日志、问卷、访谈记录、同行评估表格以及教师观察笔记，共计约1,568条对话记录。

**📈 对比分析**

通过对聊天机器人的设计忠实度（四项准则评分）与学习者信任感、Trait设定等指标进行对比，发现整体忠实度平均为2.84（满分4），学术机器人高于休闲机器人，且高中生的信任与机器人表现更为一致。

**⚠️ 局限性**

研究受限于样本量不均、三组研究环境差异大（面对面夏令营、线上项目、学校夏季班）以及小学组缺乏直接访谈，导致年龄与情境因素难以独立解析。

---

## 270. You've Seen Enough: Quality-Constrained Image Coding for Machines

**arXiv ID:** 2609.25108 | [PDF](https://arxiv.org/pdf/2609.25108v1)

**作者:** Khoa Pham-Dinh `[一作]` (Tampere University), Farhad Pakdaman `[通讯]` (Nokia Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在图像压缩和语义分割联合训练中，通过质量约束将压缩质量控制在指定PSNR水平，进而把剩余编码容量用于提升机器任务性能。

**💡 创新点**

创新点在于把压缩质量作为显式约束引入损失函数，并设计绝对与双线性两种惩罚函数；双线性惩罚在达到目标质量后加大惩罚力度，显著提升任务精度。

**🔧 技术方法**

使用注意力机制的Learned Image Compression模型与冻结的Mask2Former语义分割网络，改进损失为 R + λ1·Penalty(D,Dt) + λ2·T，并采用熵模型估计码率。

**📊 数据集**

在Cityscapes数据集（19类）上进行训练与评估，使用256×256随机裁剪并在中心裁剪验证。

**📈 对比分析**

与纯率失真基线和无约束联合优化（JO）相比，双线性惩罚实现了约-22.82%（相对JO）和-29.81%（相对基线）的BD-rate下降，同时在相同码率下mIoU提升至约42%。

**⚠️ 局限性**

局限性包括：目标质量为全局固定且未考虑空间自适应；仅在语义分割任务上验证，需进一步验证到其他任务与数据集的泛化能力。

---

## 271. CODA: Depth-Aligned Scene Completion and Object Decomposition from a Single RGB-D Image

**arXiv ID:** 2609.25654 | [PDF](https://arxiv.org/pdf/2609.25654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 272. From Utterances to Networks: Modelling Slang Adoption and Diffusion Across Subreddits

**arXiv ID:** 2609.25669 | [PDF](https://arxiv.org/pdf/2609.25669v1)

**作者:** Xiaoning Wang `[一作]` (University of Illinois Urbana-Champaign), Zhewei Sun `[通讯]` (Toyota Technological Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用大型语言模型（LLM）进行大规模俚语检测，并基于此构建人类标注基准，随后通过网络结构特征与语义特征的组合，对Reddit上俚语的扩散与个体采用行为进行定量分析。

**💡 创新点**

① 将社交网络（桥接资本与本地连通性）与俚语语义多样性（语义分散、主题效应）两类特征结合；② 引入固定效应负二项计数模型和时间变比例风险模型，以分层控制社区、时间与词汇等混杂因素；③ 通过LLM实现可扩展的俚语注释，首次在真实Reddit语料上评估LLM在俚语识别与词义消歧上的性能。

**🔧 技术方法**

使用Gemma‑4‑E4B‑it作为主要LLM进行俚语识别；vLLM实现批量推理；Sentence‑BERT提取上下文嵌入；ConvoKit读取Reddit语料；统计模型包括固定效应负二项回归和Cox比例风险回归。

**📊 数据集**

1) 54个子版块的Reddit评论数据（2008‑2018年，约1亿条）；2) 1333条人工标注的Reddit句子，构成俚语检测基准；3) 由版主维护的俚语词典（约3965个俚语）。

**📈 对比分析**

与Llama‑3.1‑8B、Olmo‑3‑7B、Ministral‑3‑8B等四款开源模型对比，Gemma在单词消歧任务上取得93.44% F1、列表消歧任务83.98% F1，显示出最佳性能；在计数模型中，桥接度正相关（β=0.103）、本地度负相关（β=‑0.506），语义分散与主题效应均为负相关；在Cox模型中，度中心性加速采用（HR=1.209），桥接度减缓采用（HR=0.884），语义分散促进采用（HR=1.059），主题效应抑制采用（HR=0.926）。

**⚠️ 局限性**

1) 研究基于观测数据，因果解释受限；2) LLM检测误差可能影响下游结果，虽做蒙特卡罗校正但仍依赖假设；3) 未控制用户兴趣、活跃度等潜在混杂变量；4) 只考虑已进入词典的成功俚语，忽略失败的创新；5) 采用定义基于生产，未捕捉理解层面的学习；6) 假设LLM性能在不同版块间一致，实际可能有差异。

---

## 273. Decoupling Disease, Covariates, and Individual Variability: A Unified Disentanglement Framework for Medical Image Classification

**arXiv ID:** 2609.25650 | [PDF](https://arxiv.org/pdf/2609.25650v1)

**作者:** Shengjie Zhang `[一作]` (Shanghai Jiao Tong University), Alzheimer's Disease Neuroimaging Initiative `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究提出了一个统一的多模态深度学习框架，利用图神经网络（GIN）处理脑功能/结构图，3D Vision Transformer 对三维MRI进行特征提取，并用2D残差CNN对X射线影像进行编码，从而实现对七个医学数据集中的多种疾病进行分类；

**💡 创新点**

创新点在于将多种模态的专用网络（GNN、3D ViT、2D CNN）整合为统一的编码与投影流程，并提供灵活的结构可替换（GAT、GraphSAGE、GCN）以及详尽的预处理方案，显著提升跨模态学习的一致性与可扩展性；

**🔧 技术方法**

核心技术包括：Graph Isomorphism Network（GIN）及其变体、8层6头的3D Vision Transformer、4阶段下采样的残差CNN、全局平均池化与线性投影、以及针对自监督学习的线性SVM后处理；

**📊 数据集**

使用的数据集涵盖七种医学影像与图形数据，分别为ADHD-200、SCZ、Presbycusis、PPMI、ADNI、肺结核（TB）和脑图网络（fMRI/DTI）等多中心、多扫描仪多模态样本；

**📈 对比分析**

在5折交叉验证中，将本框架与九个图形学习基线（BrainGNN、NEGAT、GraphCL等）以及多种3D/2D视觉基线（CNN、ResNet、ViT、GF-Net、V‑Mamba等）进行公平对比，实验表明本方法在准确率与AUC上均优于现有方法；

**⚠️ 局限性**

局限性主要体现在对数据集规模与多中心差异的鲁棒性尚待进一步验证，预处理流程较为繁琐且对硬件要求高，同时框架对新兴模态（如多模态融合网络）的适应性仍需扩展。

---

## 274. CDKF-Track: Cluster-aware Data-Driven Kalman Filtering for Cooperative 3D Multi-Object Tracking

**arXiv ID:** 2609.25668 | [PDF](https://arxiv.org/pdf/2609.25668v1)

**作者:** Maria Damanaki `[一作]` (Industrial Systems Institute, Athena Research Center), Aris S. Lalos `[通讯]` (Industrial Systems Institute, Athena Research Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aaccfe5c-6b26-4208-b23c-35331481e142` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 CDKF-Track 框架，用于在车辆间共享 LiDAR 检测进行 3D 多目标跟踪。

**💡 创新点**

创新点在于：① 通过图拉普拉斯与 DBSCAN 的集群融合消除冗余检测；② 用数据驱动的 Kalman 滤波器学习非线性运动模型；③ 采用 Haar 小波时间分解对轨迹进行去噪与平滑。

**🔧 技术方法**

核心技术包括图信号处理（Graph Laplacian）、密度聚类（DBSCAN）、可学习 Kalman 滤波、Haar 小波阈值去噪、3D CIoU+Hungarian 匹配。

**📊 数据集**

使用真实车对交互数据集 V2V4Real（包含 Tesla 与 Astuff 两辆车的 LiDAR 轨迹）。

**📈 对比分析**

与单车 SA MOT、MA MOT 基线（DMSTrack、TSA‑Graph MOT、AB3DMOT）对比，CDKF‑Track 在 sAMOTA、AMOTA、AMOTP、MT 等指标分别提升 3.63%、6.57%、14.62% 和 27.99%。

**⚠️ 局限性**

局限性：仅验证两车系统，未扩展到更大网络；中心化投影可能引入时延；代表检测的选择可能丢失部分有效信息；未在 V2I 或更大规模场景中评估。

---

## 275. When Riemann flows with Wasserstein: Generative Modeling of Probability Distributions on Manifolds

**arXiv ID:** 2609.25659 | [PDF](https://arxiv.org/pdf/2609.25659v1)

**作者:** Doron Haviv `[一作]` (Genentech Inc.), Hector Corrada Bravo `[通讯]` (Genentech Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了Riemannian Wasserstein Entropic Flow Matching（RWEFM），用于在Riemannian流形上对概率测度空间进行生成建模。

**💡 创新点**

创新点在于将流形几何与Wasserstein空间流匹配相结合，并提出GPU高效的Riemannian Entropic Map估计OT映射，使得可以在任意Riemannian流形（甚至无解析几何的三角网格）上生成分布。

**🔧 技术方法**

采用了Riemannian流形的几何工具（指数映射、对数映射、McCann插值）、熵正则化OT、Sinkhorn算法、神经网络向量场回归与自注意力结构，以及流匹配理论。

**📊 数据集**

使用的实验数据集包括MNIST、EMNIST、KMNIST映射到球面、双曲平面和环面；单细胞RNA‑seq在高维球面（SCimilarity嵌入）；蛋白质主链二面角分布在二维环面；以及三角网格（斯坦福海豚）上的数字分布。

**📈 对比分析**

与传统流匹配（FM、RFM）、Wasserstein流匹配（WFM）、SetFM/SetRFM、PVD/PSF等基线相比，RWEFM在1‑NN偏差、Chamfer/EMD距离、MMD等指标上显著下降，尤其在非欧几里得流形和无解析几何时表现最为突出。

**⚠️ 局限性**

局限在于仍需要流形上指数/对数映射或仅通过距离与投影近似，计算成本受样本数与正则化参数影响；对极端高维或大规模分布仍有训练时间与收敛稳定性挑战。

---

## 276. Seeing Is Not Perceiving: When Synthetic Consumers Can and Cannot Pretest Visual Marketing

**arXiv ID:** 2609.25677 | [PDF](https://arxiv.org/pdf/2609.25677v1)

**作者:** Yi-Lin Tsai `[一作]` (University of Melbourne), Lai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对六项经典视觉营销实验进行零样本与提示干预实验，评估大型语言模型（GPT‑4o‑mini、GPT‑5.4‑mini）在视觉资产预测试中的可靠性；

**💡 创新点**

创新点在于首次将“synthetic consumers”从文本推断扩展到视觉领域，并通过对技术先验与心理先验的区分揭示模型在感知层面的偏差；

**🔧 技术方法**

采用多模态提示、文本理由生成、主题建模（BERTopic）和中介分析等技术来评估模型输出；

**📊 数据集**

使用公开的六篇视觉营销实验的原始图像与实验数据作为基准数据集；

**📈 对比分析**

与人类受试者的平均效应、方向一致性和方差比对比显示，模型在零样本时仅能再现不超过两项效应，提示干预虽可校准平均效应但仍无法恢复人类响应的方差；

**⚠️ 局限性**

主要局限在于模型对人类情感关联的捕捉不足、响应方差持续收缩以及对不同模型版本和输入格式的普适性尚未验证。

---

## 277. Slow Decay and Silenced Expression: Iterated Subliminal Trait Transfer in Language-Model Lineages

**arXiv ID:** 2609.25721 | [PDF](https://arxiv.org/pdf/2609.25721v1)

**作者:** Ryan Vo `[一作]` (Denison University), Ngan Luu-Thuy Nguyen `[通讯]` (University of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在语言模型链中通过低秩微调将“owl”偏好等特征传递十代，观察特征在行为和内部表示中的衰减。

**💡 创新点**

首次展示多代迭代链中隐性特征的长期持久性和行为与内部表示的分离，并验证通过激活向量可重新引导特征。

**🔧 技术方法**

使用 Qwen2.5-7B‑Instruct + QLoRA（rank‑16 attention‑only）微调、数字序列生成作为载体、激活投影、关键词屏幕和激活 steering。

**📊 数据集**

数据集为三条传递线索：教师生成的数字序列（30,000 条）、筛选后的数字样本、20 个 held‑out 选择动物提示。

**📈 对比分析**

比较方法：行为评分（关键词屏幕）与激活投影；结果显示第一步损失最大，后续十代保持 21% 行为表达与 57% 激活投影，steering 在层 26 最高效。

**⚠️ 局限性**

局限性包括仅在单一模型、单一特征、单一低秩微调配置，且未验证不同模型/语言或更深链条；评估屏幕依赖人工校准，激活向量需教师知识。

---

## 278. MatcherCompass: A Deployment-Aware Benchmark to Guide Image Matcher Selection in the Wild

**arXiv ID:** 2609.25688 | [PDF](https://arxiv.org/pdf/2609.25688v1)

**作者:** Hyunwoo Kim `[一作]` (DGIST), Giseop Kim `[通讯]` (DGIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个面向现场机器人部署的匹配器基准，评估九种经典与学习型特征匹配管线在不同视觉条件、分辨率、数值精度及四种GPU平台下的姿态精度与计算成本，并提供交互式配置选择指南。

**💡 创新点**

创新点在于：① 将时间、能耗、内存等多维资源限制与姿态精度统一评估；② 覆盖可见、热成像以及昼夜、跨模态匹配场景；③ 通过实验展示硬件、分辨率和精度如何决定匹配器可行性；④ 提供基于实验数据的实时配置搜索工具。

**🔧 技术方法**

使用的技术包括 SuperPoint、SuperGlue、LightGlue、LoFTR、ELoFTR、MatchAnything、DKM、MASt3R、ORB、SIFT 等匹配管线；采用 GPU 同步计时、CUDA peak allocation、CUPTI 能耗记录等硬件监测手段；以及 Pose AUC 评价指标。

**📊 数据集**

数据集主要是 Aachen 视觉/热成像数据集（Aachen VisThermal）用于跨模态匹配，以及 Aachen 视觉数据集用于视角变化、昼夜匹配；两者共计 100 对图像。

**📈 对比分析**

比较方法：在统一的姿态评估流程下计算 AUC@5°,10°,20°，同时记录每对图像匹配的运行时、GPU 内存峰值和能耗；实验表明：同一匹配器在不同硬件/分辨率/精度组合下会出现明显的性能/精度差异，某些配置可在 20 ms 预算内完成，而某些则超时或 OOM。

**⚠️ 局限性**

局限性包括：仅评估了四种 GPU 平台且未涉及 CPU/嵌入式 DSP；仅覆盖可见/热成像两种模态，未包含多光谱或深度图；实验使用预训练模型，未考虑自适应训练或迁移学习；以及基准结果依赖于特定的姿态评估方式，可能不适用于所有 SLAM 或定位场景。

---

## 279. Initialization and Stopping Tolerance in CPU Dermoscopic Segmentation

**arXiv ID:** 2609.25685 | [PDF](https://arxiv.org/pdf/2609.25685v1)

**作者:** Wenhao Xu `[一作]` (Zhengzhou Police University), Rongtao Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了不同初始化方式与停止容差对基于 Chan–Vese 能量的皮肤病变分割效果的影响。

**💡 创新点**

创新点在于系统评估了 Otsu 阈值初始化与传统棋盘/圆盘初始化的交互效应，并指出停止容差对比较结果的显著影响。

**🔧 技术方法**

使用了 scikit‑image 的 Chan–Vese 实现、Otsu 全局阈值、距离函数初始场以及多种停止容差设定。

**📊 数据集**

实验数据来自 ISIC 2017 镜像数据集（600 张验证图像）。

**📈 对比分析**

与仅使用 Otsu 阈值对比时，Otsu 种子初始化在默认容差下平均 Dice 提升约 0.065，但仍低于纯阈值；更严格容差则差距缩小，CPU 负荷显著升高。

**⚠️ 局限性**

局限性包括缺乏头发去除、只处理单一尺寸图像、未考虑患者/诊断信息以及只评估了单一两区模型和无监督方案。

---

## 280. Deploying Foundation Models for Embodied Navigation

**arXiv ID:** 2609.25666 | [PDF](https://arxiv.org/pdf/2609.25666v1)

**作者:** Vishnu Sashank Dorbala `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了两种方法，分别是用于个性化导航的Transit‑Aware Planning（TAP）和用于缓解有限上下文长度的Memory Control（MemCtrl），以实现基础模型在具身导航任务中的可部署性。

**💡 创新点**

创新点在于：TAP通过对环境动态对象迁移行为的预训练和场景记忆实现个性化路径规划；MemCtrl则引入主动记忆过滤头，显著降低上下文占用并提升低参数模型的决策效率。

**🔧 技术方法**

技术上结合了大型语言模型（LLM）与视觉语言模型（VLM）、动态对象地图（DOM）构建、以及基于在线强化学习和离线监督训练的可插拔记忆头。

**📊 数据集**

实验数据涵盖真实实验室环境中的图像采集、Habitat/ALFRED/EB‑Habitat 仿真数据集，以及使用 Qwen2.5‑VL‑7B‑Ins、GPT‑4o 等基础模型。

**📈 对比分析**

与无TAP或无MemCtrl 的基线模型相比，TAP 在个性化目标搜索任务中平均提升 18%，MemCtrl 在多任务评估中平均提升 6%，长指令子集提升 20%，并且显著减少了所需上下文长度。

**⚠️ 局限性**

局限性包括：对动态环境的真实性依赖仍有限；记忆头在探索与利用策略上仍需进一步优化；以及在更大规模、异质化真实环境中的泛化能力尚未充分验证。

---

## 281. Skill Sequence Planning for Collaborative Multi-Robot Construction

**arXiv ID:** 2609.25649 | [PDF](https://arxiv.org/pdf/2609.25649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 282. Testing-Driven Reliability Audit of Trajectory-Based Early Outcome Prediction for LLM Agents: Target-Specific Calibration Transfer Persists Within a Single Benchmark

**arXiv ID:** 2609.25647 | [PDF](https://arxiv.org/pdf/2609.25647v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 283. Targeted Review for AI-Assisted Biodiversity Surveys: Active Continuous-Score Occupancy Modeling

**arXiv ID:** 2609.25657 | [PDF](https://arxiv.org/pdf/2609.25657v1)

**作者:** Timm Haucke `[一作]` (MIT), Sara Beery `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个结合机器学习分类结果和占有率模型的系统，主动选择专家复核样本以最大化对生态推断的收益。

**💡 创新点**

创新点在于（1）提出了“解耦连续分数占有率模型”，将分数校准与占有率推断分离；（2）设计了针对目标生态量的贝叶斯实验设计（Target EIG）来优先获取最有信息量的复核样本；（3）给出了基于预算的停机准则。

**🔧 技术方法**

使用了贝叶斯实验设计、目标信息增益（Target EIG）、解耦连续分数模型、后验熵估计、主成分降维、以及对比的随机、最大分数、BALD、后验不确定性等复核策略。

**📊 数据集**

实验基于两类数据集：iWildCam 2022 视觉摄像机陷阱数据（与 SpeciesNet 输出）和 Acoustic Forest Soundscape 生态声学数据（与 Perch v2 输出）。

**📈 对比分析**

与传统仅用已复核标签的伯努利占有率模型以及原始联合连续分数模型比较，并在五种复核策略中评估。结果显示，解耦连续分数+Target EIG 在低到中等复核预算下显著提升了四项生态结论指标（占有率、排名、占有率系数、检测系数），实现了约2-9倍的专家时间节省。

**⚠️ 局限性**

局限包括：仅针对单物种单季占有率问题；分数分布异质性仍可能影响校准；缺乏多物种、空间随机效应的扩展；停机准则仍为经验式，缺乏理论保证。

---

## 284. PhyVisGen: Physically and Visually High-Fidelity Robotic Manipulation Data Generation

**arXiv ID:** 2609.25653 | [PDF](https://arxiv.org/pdf/2609.25653v1)

**作者:** Yu Zheng `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为PhyVisGen的框架，能够在仿真中以软抓手为核心生成高物理与视觉逼真度的机器人抓取与操作演示，并实现零样本从仿真到真实机器人的迁移

**💡 创新点**

创新点在于①采用IPC（Incremental Potential Contact）方法实现软抓手与机器人臂的耦合模拟，支持完整抓取与操作轨迹；②开发实时光线追踪渲染管线，配合可保持原始纹理外观的阴影接收材质和LUT校准，显著提升视觉真实性；③整合上述两项技术实现一体化的高保真数据生成；

**🔧 技术方法**

使用IPC-based软抓手与机器人臂耦合模拟、StiffGIPC、FEM/ABD统一框架、实时路径追踪渲染、阴影接收材质、1D LUT光照校准，以及PGSR/TSGS与Map-Anything等三维重建技术

**📊 数据集**

利用真实环境中的三件易碎物体（烧杯、烧瓶、盘子）以及多任务软抓手操作数据（称重、放置、开启柜门、折叠毛巾等）作为评估基准，所有数据均来自真实机器人与ZED摄像机采集的配准场景

**📈 对比分析**

与Isaac Sim、SAPIEN等传统仿真/渲染方法比较，物理准确性（软抓手掩膜IoU）在80%以上，视觉质量（PSNR、SSIM、LPIPS）显著优于对比方法；在五个真实任务中，基于PhyVisGen训练的ACT策略零样本在20次试验中的成功率均超过65%，最高达到95%

**⚠️ 局限性**

局限性包括：①依赖于高质量的三维重建与光照估计，若环境纹理或光照变化大可能导致渲染误差；②IPC及软抓手耦合在大规模高频率仿真下计算成本仍较高；③仅在有限数量任务与物体上验证，复杂多物体交互或长时序任务的迁移效果尚未探究

---

## 285. RoboFollow: Unveiling the Instruction Following Mirage in Embodied Agents

**arXiv ID:** 2609.25636 | [PDF](https://arxiv.org/pdf/2609.25636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 286. Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices

**arXiv ID:** 2609.25645 | [PDF](https://arxiv.org/pdf/2609.25645v1)

**作者:** Qian Xie `[一作]` (Cornell University), Nairen Cao `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GittinsEval，一种成本感知的贝叶斯 Bandit 方法，用于在大型语言模型（LLM）配置评估中自适应分配测试样本并给出最优配置建议。

**💡 创新点**

创新点包括：① 将配置选择建模为成本敏感的贝叶斯 Bandit；② 使用 Gittins 指数进行采样决策并结合 LCB 风格的任意时间推荐；③ 通过离线 FFT 加速预计算 Gittins 指数表，在线仅需轻量级后验更新和表查找。

**🔧 技术方法**

主要技术手段有：贝叶斯 Bandit、Gittins 指数、Gaussian 观测模型、FFT 根表计算、LCB 推荐规则、批量评估、成本归一化。

**📊 数据集**

实验数据集包括 GSM8K、PIQA、AlpacaEval、MMLU 四个标准基准，覆盖数百至 1,500 个配置。

**📈 对比分析**

与 UCB‑E、LRF、SySRs、PromptEval‑BAI 以及配置层贝叶斯优化（PBGI、LogEI、LogEIPC）等基线相比，GittinsEval 在单位成本与成本感知两种设置下均能在低预算（1–10% 的全评估成本）内实现近乎零简单回报，并在大样本、海量候选任务上显著优于基线。

**⚠️ 局限性**

局限性包括：需要假设观测噪声可用固定 Gaussian 近似；算法对真实分布的泛化受限；需先行离线预计算指数表；实验仅在已预制的响应矩阵上验证，未对实时 API 调用进行评估。

---

## 287. A Reconfigurable Bidirectional Cable-Driven Hip Exoskeleton with Swappable Bench/Backpack Dual-configuration Actuation

**arXiv ID:** 2609.25639 | [PDF](https://arxiv.org/pdf/2609.25639v1)

**作者:** YuanLong Ji `[一作]`, Xingbang Yang `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种可在实验室（台架）与户外（背包）两种模式下使用的双向电缆驱动髋关节外骨骼平台，并验证了其切换、运动跟踪与扭矩跟踪功能。

**💡 创新点**

创新点在于：①将可拆卸的快速释放/组装接口与电缆驱动传动相结合，使同一人体侧结构可与不同动力单元配合；②在末端执行器上集成扭矩传感器和编码器，实现端部扭矩与关节角度同步测量；③利用无线IMU保持测量位置不变，支持跨场景的数据同步；④生成了跨场景的多模态步态数据集。

**🔧 技术方法**

采用电缆驱动（Bowden）传动、快速释放机械接口、端部扭矩与角度传感、无线IMU、台架仿真系统（工业计算机+高性能GPU）以及移动端嵌入式控制（Jetson Nano+STM32+CAN总线）等技术。

**📊 数据集**

使用了一个小规模的多模态可穿戴-外骨骼步态数据集：8 min 台架跑步（仿真台上）与 11 min 背包户外步行，共含编码器、扭矩传感、无线IMU 与 myoMOTION 参考信号。

**📈 对比分析**

通过与现有硬件（Stanford、BLEEX、Samsung、Panasonic、Michigan 等）在峰值扭矩/可穿戴质量比（TWR）和感测配置上的对比，证明该平台在 TWR（18.4 / 10.2 N m/kg）与完整感测功能上领先；台架模式的仿真阻尼控制实现了 1.8° RMSE 的角度跟踪，背包模式的扭矩传输率约 58%（上升、保持、正弦）。

**⚠️ 局限性**

局限性包括：①背包模式下电源与电池重量导致舒适度下降；②切换时间平均 30 s，仍需进一步优化锁定与张力检查流程；③当前背包模式仅采用开环扭矩控制，未实现闭环扭矩跟踪；④实验样本仅三名健康受试者，功能验证与临床适用性仍待扩展。

---

## 288. SLED-IFV: Solver-Validated LLM-Guided Decomposition for Scalable Hardware Information-Flow Verification

**arXiv ID:** 2609.25637 | [PDF](https://arxiv.org/pdf/2609.25637v1)

**作者:** Liangtao Dai `[一作]` (University of Virginia), Mircea Stan `[通讯]` (University of Virginia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于LLM指导的证明分解框架SLED-IFV，用于加速硬件信息流验证；

**💡 创新点**

创新点在于通过LLM自动选择并生成功能简化与关系强化两种语义分解形式，并通过求解器验证保证可信度；

**🔧 技术方法**

技术手段包括自组成IFV、IC3/PDR求解器、LLM（Claude）生成提案与证明工件、false-law canary、faithfulness控制等；

**📊 数据集**

使用了9个真实RTL案例的数据集，涵盖OpenTitan DOM、Goldschmidt浮点除法器、HMAC/SHA、lowRISC Ibex、PicoRV32、Chipyard/BOOM等；

**📈 对比分析**

与ABC-PDR基准对比，SLED-IFV在9个案例中实现最高603×求解器速度提升，并成功解决了原本12小时超时的两个案例；

**⚠️ 局限性**

局限性在于依赖LLM生成提案，质量受模型表现影响；且目前仅针对自组成IFV，尚未覆盖更广泛的硬件验证场景。

---

## 289. Beyond Reconstruction Error: Analytical and Data-Driven Action Tokenization for Autoregressive Vision-Language-Action Models

**arXiv ID:** 2609.25820 | [PDF](https://arxiv.org/pdf/2609.25820v1)

**作者:** Yuxin Yang `[一作]`, Hangming Liu `[通讯]` (Tianfu Securities Co Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在自动回归视觉‑语言‑动作（VLA）模型中，比较不同动作离散化方法（PCA、时间离散余弦变换 DCT 与自编码器）对闭环控制的影响。

**💡 创新点**

创新点在于将动作量化的评价从单纯的重建误差扩展为三维度：几何精度、序列可预测性和解码器稳定性，并通过统一接口系统化对比。

**🔧 技术方法**

采用线性正交基、固定解析基（DCT）、非线性自编码器以及对称均匀量化、token entropy、n‑gram perplexity、扰动增益等多项技术。

**📊 数据集**

使用 LIBERO‑Spatial（含 9 个任务）作为主要数据集，进行 3,500 次闭环滚动实验，并在 LIBERO‑Object、LIBERO‑Goal 进行离线交叉验证。

**📈 对比分析**

方法通过率失真曲线、token 可预测性（熵、困惑度）和实际控制成功率进行比较；实验显示 DCT 在平均成功率上比 PCA 高 3.0%（尽管重建误差更大），而自编码器虽最低重建误差但控制表现居中。

**⚠️ 局限性**

局限包括仅使用单一 VLA 后端、仅 3 个随机种子、固定 token 数 K=14、未探索学习的码本或 BPE 压缩，以及扰动增益仅为局部度量，且 Task 9 的 0/50 成功率限制了跨任务结论。

---

## 290. MorphoSHAP: Rethinking the Unit of Attribution in Explanation for Deep Visual Models

**arXiv ID:** 2609.25815 | [PDF](https://arxiv.org/pdf/2609.25815v1)

**作者:** Anirudh Prabhakaran `[一作]` (AMIAD), Gianni Franchi `[通讯]` (AMIAD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于形态学树的后置解释框架MorphoSHAP，用形状而非像素或超像素作为Shapley游戏的参与者，提供结构化的形状尺度、几何标签与贡献值；

**💡 创新点**

创新点在于将数学形态学的Tree of Shapes作为解释单元，形成统一的尺度-几何词汇表，使解释可呈现为热图、文本和全局统计；

**🔧 技术方法**

使用Tree of Shapes进行形状分解，基于Shapley值的KernelSHAP估计贡献，构造多尺度几何描述并给形状分配规则或学习型几何标签；

**📊 数据集**

在五个多领域数据集上评估：ImageNet、Galaxy10、EuroSAT、Waterbirds、TissueMNIST；

**📈 对比分析**

与Grad‑CAM系列、IG、KernelSHAP（SLIC/Otsu）、PartitionSHAP、ShapBPT等方法对比，在插入/删除AUC上获得最优或接近最优成绩，运行时间仅0.13s/图，显著快于其他SHAP方法；

**⚠️ 局限性**

局限性包括对纹理依赖较弱的任务效果不佳，且解释的质量依赖于形状标记函数，需要进一步加入纹理或领域特定的标记策略

---

## 291. Fysiverse-3D-Vision Technical Report: Generating Executable 3D Worlds from Images through Unified Spatial Reasoning

**arXiv ID:** 2609.25741 | [PDF](https://arxiv.org/pdf/2609.25741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 292. Towards Omni-dimensional GUI Agent Navigation with Masked Trajectory Prediction

**arXiv ID:** 2609.25769 | [PDF](https://arxiv.org/pdf/2609.25769v1)

**作者:** Yan Zhang `[一作]` (Chinese Academy of Sciences), Jian Luan `[通讯]` (Xiaomi Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MaP框架，将多任务GUI导航任务统一为轨迹掩码预测，解决了优化目标不一致和数据异质性问题。

**💡 创新点**

创新点在于通过轨迹级掩码预测统一不同导航任务的训练目标，并引入角色感知适配器来处理数据异质性。

**🔧 技术方法**

采用掩码轨迹预测（Mask Trajectory Prediction）与基于LoRA的多适配器路由技术，结合视觉语言模型。

**📊 数据集**

使用AndroidWorld、AndroidControl、GUI-Odyssey、AITZ、Mind2Web等公开GUI导航基准。

**📈 对比分析**

通过与直接混合训练和现有GUI中间训练方法比较，在零样本和后训练设置下，MaP在大部分指标上提升1–5个百分点，尤其在Pass@4上提升约5%。

**⚠️ 局限性**

局限性在于仅在中等规模的基础模型和公开数据集上验证，未探究更大规模LVLM或更丰富训练数据的扩展效果。

---

## 293. Neurosymbolic Action Model Learning under Partial Observability

**arXiv ID:** 2609.25766 | [PDF](https://arxiv.org/pdf/2609.25766v1)

**作者:** Adem Kikaj `[一作]` (KU Leuven), Luc De Raedt `[通讯]` (KU Leuven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种在部分可观测视觉轨迹下学习动作模型的神经符号框架

**💡 创新点**

将概率动作模型与STRIPS语义约束结合，构建可采样的关系神经符号马尔可夫模型，并采用变分推理进行端到端学习，突破了现有均值场近似在遮挡场景下的局限

**🔧 技术方法**

概率动作模型、STRIPS约束、关系神经符号马尔可夫模型、可采样的变分推理、神经感知模块、符号推理模块

**📊 数据集**

六个视觉规划域（网格块世界、Gripper、Logistics、合成块世界、汉诺塔、8数码拼图），三种观察模式（完全可观测、随机遮挡、动作遮挡）

**📈 对比分析**

与现有均值场近似方法（MPC）进行对比。新方法在遮挡场景下的相关角色恢复率显著高于对照组（90% vs 61–68%），在全观测下表现相当；随着基化规模增大，新方法保持高恢复率，旧方法则显著下降；采样数量从2到256时，新方法恢复率基本不变，但计算成本随样本数增大显著提升

**⚠️ 局限性**

对未参与谓词的误报较高，正轨迹难以区分预置条件与不参与，导致潜在的假前置或假效果，且缺乏对不参与谓词的最小化先验，无法完全消除识别歧义

---

## 294. Improved Algorithms for the Remote Point Problem

**arXiv ID:** 2609.25765 | [PDF](https://arxiv.org/pdf/2609.25765v1)

**作者:** Ben Lee Volk `[一作]` `[通讯]` (Reichman University), Ben Lee Volk (Reichman University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

本文针对远点问题（Remote Point Problem，RPP）在有理数域和有限域上的算法进行了改进，给出了在有理数域上实现最优远度（n‑k）的简单多项式时间算法，并在有限域上将原来的远度Ω(n/k·log k)提升到Ω(n/max{k, log n}·log n)。

**💡 创新点**

创新点在于：
• 在有理数域上利用syndrome与Cramer定理构造一个唯一且可计算的syndrome向量，直接得到最优远度；
• 在有限域上通过分块与改进的原始算法相结合，利用子空间维度小的优势提升远度。

**🔧 技术方法**

主要技术包括：
• 通过syndrome表述将RPP转化为“求不可由少量列生成的syndrome”问题；
• 对有理数域使用Cramer式的行列式求逆，构造小系数矢量证明syndrome不可达；
• 对有限域使用分块（block）技术与已有的log n远度算法相结合，并利用潜在函数搜索不被子空间覆盖的点。

**📊 数据集**

该工作不涉及实验数据集，全部为理论算法与证明。

**📈 对比分析**

与Alon‑Panigrahy‑Yekhanin等人的原始算法相比，
• 在有理数域上实现了最优远度n‑k；
• 在有限域上，远度从Ω(n/k·log k)提升到Ω(n/max{k, log n}·log n)，尤其在k为子多项式时效果更明显；
• 算法仍保持多项式时间复杂度。

**⚠️ 局限性**

限制与未解决问题：
• 有理数域算法得到的向量坐标值可能指数级大，虽然位复杂度为多项式，但在实际应用中仍不方便；
• 对有限域的提升仍远未达到最优远度，且仅适用于k≤n/2的情形；
• 该方法尚未推广至更高远度（如ω(log n)）或更小的点集构造，亦未给出低位复杂度的刚性矩阵构造。

---

## 295. Lizard: Bandwidth-Adaptive Real-Time Video Analytics through Content-Aware Packet Discarding at Last-Mile Edge Routers

**arXiv ID:** 2609.25817 | [PDF](https://arxiv.org/pdf/2609.25817v1)

**作者:** Shan Yu `[一作]` (University of California, Los Angeles), Harry Xu `[通讯]` (University of California, Los Angeles)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在边缘路由器上实现内容感知的包丢弃，利用视频帧块的重要性信息在可用带宽骤降时主动丢弃低重要性块，以降低延迟并提升实时视频分析精度。

**💡 创新点**

创新点包括：① 将帧块级信息嵌入RTP头扩展，实现对单个块的独立识别与优先级编码；② 基于服务器端反馈的相对准确性影响（RAI）动态评估块优先级；③ 采用三阶段（INIT‑PRE‑PD）无状态的边缘路由器丢弃策略，兼顾延迟与准确性，并支持多流公平性。

**🔧 技术方法**

主要技术：RTP/RTCP 头扩展、eBPF/ XDP 边缘路由器程序、H.264 逐块编码、相对准确性影响评估、基于反馈的动态阈值计算与三阶段丢弃算法。

**📊 数据集**

使用了两份真实网络带宽日志（Oboe、Ghent）和四份视频数据集（Auburn、Banff、Jacksonhole、nuScenes），涵盖静态摄像头与移动摄像头，任务为车辆与行人检测（YOLOX）。

**📈 对比分析**

与 GCC‑only、CoDel、EAAR 等基线比较；在 Oboe 与 Ghent 带宽下，本文方案将 99.5 % 端到端帧延迟降低 53.2 %，准确率提升最高 27.1 %（静态摄像头）/约 0.1 %（移动摄像头），且对尾部延迟与准确性提升最显著。

**⚠️ 局限性**

局限性：① 需在路由器上支持 eBPF/ XDP，且对块数 N 的扩展受头部长度限制；② 对极端动态场景（如移动摄像头）可丢弃空间可用性低，提升有限；③ 需要与客户端/服务器协同工作，若部署不完整可能失效；④ 参数 θ 与块大小需根据场景手工调优。

---

## 296. What Was Once Learned May Need to Be Unlearned: Machine Unlearning for Deprecated API Knowledge in Large Language Models

**arXiv ID:** 2609.25786 | [PDF](https://arxiv.org/pdf/2609.25786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 297. You Only Need 2/3 of the Chosen Experts: An Empirical Study of Dynamic Expert Pruning in Fine-Grained MoE LLMs

**arXiv ID:** 2609.25809 | [PDF](https://arxiv.org/pdf/2609.25809v1)

**作者:** Yuanteng Chen `[一作]`, Jian Cheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对12个细粒度Mixture‑of‑Experts（MoE）检查点进行系统实验，评估统一截断和四种动态专家修剪规则在11个知识、推理与生成基准上的效果，探讨专家冗余度、动态分配优势及模型对剪枝敏感性的决定因素。

**💡 创新点**

①首次量化细粒度MoE的路由冗余，发现统一保留约三分之二专家即可保持98.8%性能；②系统比较不同预算下动态分配与统一截断的性能差距，发现仅在激进剪枝时才显著；③揭示模型属性（规模、推理后训练、跨模态）对剪枝敏感度的影响，为未来动态规则的设计提供实证依据。

**🔧 技术方法**

统一截断（k=⌈2/3⌉）、阈值剪枝（Threshold）、累积概率剪枝（Cumulative）、相似度加权剪枝（Similarity‑Adjusted）和层敏感度+令牌级分配（Layer‑Sensitive）四种规则；同时使用vLLM和HF Transformers两套推理后端评估吞吐量。

**📊 数据集**

知识问答：ARC‑Easy、ARC‑Challenge、WinoGrande、OpenBookQA；数学与代码推理：AIME‑24、AIME‑25、MATH‑500、GSM8K、LiveCodeBench‑v6；通用推理：MMLU‑Pro、GPQA‑Diamond；多模态：9‑dataset multimodal suite；文本问答：9‑dataset likelihood‑scored QA。

**📈 对比分析**

通过将动态规则的平均活跃专家数与统一截断在同一预算下对齐，比较所有11个基准的平均分。统一截断在所有模型上平均保留98.8%性能并提升1.2‑1.7×解码吞吐量；动态规则在保守预算下差异≤0.7个百分点，激进预算下可提升至+3.0分，提升主要集中在生成任务上。

**⚠️ 局限性**

仅针对细粒度MoE模型，未覆盖更大规模或不同路由策略；动态规则仅限四种经典方法，未探索新型分配算法；未细致评估动态分配的运行时开销；实验基于固定的12个检查点，结果可能对其他架构或超参不完全通用。

---

## 298. Latest Exact Match Attention

**arXiv ID:** 2609.25802 | [PDF](https://arxiv.org/pdf/2609.25802v1)

**作者:** Moritz Brösamle `[一作]` `[通讯]` (University of Tübingen), Moritz Brösamle (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了最新精确匹配注意力（LEMA）Transformer，并证明其与单词-RAM在计算上等价；同时提出了一种基于直通估计器和渐进软注意力的训练方法，验证了该模型在合成关联回忆任务和大规模语言建模中的可行性；实现了基于哈希表的kv缓存，保持了常数生成速度并将缓存置于主内存。

**💡 创新点**

创新点包括①将查询和键二值化，仅对最新完全匹配的键进行注意；②通过理论证明LEMA与单词-RAM可互相模拟，建立两者在计算与存储上的对应关系；③引入从软注意力到硬LEMA的渐进训练策略，克服非可微问题；④将增长的状态存放于主内存，兼具可扩展性与常数生成速度。

**🔧 技术方法**

使用的技术主要有：Transformer框架中LEMA头；直通估计器（straight‑through）实现二值化；stick‑breaking注意力作为软近似并逐步收敛到LEMA；哈希表/字典实现kv缓存；链式思维（chain‑of‑thought）训练与评估；大规模GPU推理（vLLM等）以及主内存KV缓存的实现。

**📊 数据集**

实验使用的数据集为：①自定义合成关联回忆任务（含多对关联的序列）；②FineWeb‑Edu高质量文本数据用于语言建模；③RULER基准中的S‑NIAH‑1针叶检索任务。

**📈 对比分析**

与传统softmax Transformer和固定状态模型GDN进行了对比。LEMA模型在FineWeb‑Edu上的交叉熵损失可与参数量约为其一半的softmax Transformer相匹配；在重复稀有bigram和单针检索等长距离记忆代理任务中，LEMA优于GDN但仍落后于softmax；生成速度方面，LEMA与GDN保持常数速率，远快于softmax。

**⚠️ 局限性**

局限性包括：训练方法对学习率和硬化过程敏感，导致部分注意头无法匹配；大模型中约10%头从未找到匹配；与softmax Transformer之间的性能差距仍明显；训练仍需二次方计算；对更大上下文的泛化能力尚未完全验证。

---

## 299. OmniFysics-Nano-V2 Technical Report: Understanding the Physical World Across Modalities

**arXiv ID:** 2609.25738 | [PDF](https://arxiv.org/pdf/2609.25738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 300. Syndrome, Synergy, and Safety: Structured Reasoning and Knowledge-Driven Alignment for TCM Prescription Generation

**arXiv ID:** 2609.25755 | [PDF](https://arxiv.org/pdf/2609.25755v1)

**作者:** Zheng Chen `[一作]` (Tsinghua University), Peiwu Qin `[通讯]` (Guangdong Provincial Laboratory of Traditional Chinese Medicine Hengqin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了基于四阶段训练（SFT→PG-CoT→Dynamic→K-RL）的LLM框架，用于改进中医处方生成。

**💡 创新点**

创新点在于将中医诊疗范式(li‑fa‑fang‑yao)嵌入链式思维、引入纵向随访推理以及用规则驱动的DPO强化学习来同时解决可解释性、随访适配与禁忌安全三大缺口。

**🔧 技术方法**

采用了监督微调、范式引导链式思维蒸馏、动态随访SFT、基于规则的DPO强化学习以及LoRA参数高效适配等技术。

**📊 数据集**

使用了12万余条去标识化中医门诊记录，构建40K训练集与871例测试集，并从中提取症状、舌脉、处方与随访信息。

**📈 对比分析**

通过在PQS、CQS和VR三个指标上与六个零拷贝基线（含GPT‑5、DeepSeek‑V3、LLaMA‑4‑Scout等）进行对比，结果显示仅7B Mistral‑7B 在所有指标上均超越GPT‑5，四阶段模型在处方质量、推理可审计度和安全违规率上均显著提升。

**⚠️ 局限性**

局限性包括规则覆盖范围有限、使用伪患者ID导致随访链可能不准确、数据来源单一机构、合成负样本分布窄、模型对不同体系的依赖以及实验顺序导致各阶段贡献难以完全分离。

---

## 301. Dual Covariance Gaussian Splatting SLAM: Decoupling Rendering and Registration for Robust Real-Time Tracking

**arXiv ID:** 2609.25746 | [PDF](https://arxiv.org/pdf/2609.25746v1)

**作者:** Edward Beng Wai Tan `[一作]` (Nanyang Technological University), Siew-Kei Lam `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了双协方差的3D高斯喷射SLAM，允许每个高斯原语同时用于渲染和跟踪，跟踪用传感器误差协方差，渲染用光照优化协方差，并通过KLT角点投影到高斯切平面加入图像残差。

**💡 创新点**

创新点包括：①双协方差参数化，将渲染协方差与跟踪协方差分离；②利用RGB‑D传感器误差模型提供ICP权重；③将角点投影到高斯切平面作为图像残差，提升可观测性。

**🔧 技术方法**

使用技术包括3D高斯喷射（3DGS）、G‑ICP点云配准、RGB‑D误差模型、KLT角点跟踪、Huber核、Levenberg–Marquardt优化、SE(3)姿态估计以及图像残差约束。

**📊 数据集**

采用的数据集有：TUM RGB‑D、ScanNet、Replica，以及两组户外 RealSense D435i（轮式平台和手持设备）长轨迹。

**📈 对比分析**

与GS‑ICP SLAM、SGAD‑SLAM、FeatureSLAM、Photo‑SLAM、MonoGS、SplaTAM、ORB‑SLAM3等方法对比，取得最低ATE和相对位姿误差，漂移最小，实时性能约60 FPS，在无纹理/无结构、快速手持运动及户外长轨等极端场景表现最佳。

**⚠️ 局限性**

局限性：缺乏bundle adjustment导致大规模长时一致性不足；对远距离深度误差敏感，户外大距离场景跟踪精度下降。

---

## 302. SAMI3D-DW: Interactive Segmentation of Any 3D Medical Images

**arXiv ID:** 2609.25743 | [PDF](https://arxiv.org/pdf/2609.25743v1)

**作者:** Ping Gong `[一作]` (Deepwise Healthcare), Yizhou Yu `[通讯]` (Deepwise Healthcare)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发并评估了一种名为 V1 的交互式 3D 医学图像分割模型，可在 CT/MR 图像上通过点或盒子提示快速完成复杂结构和病灶的分割。

**💡 创新点**

创新点在于：1）构建了基于医学分类法的多源、类别均衡的评估框架；2）在大规模专有数据集上从零开始训练，获得了在多模态、多类别上表现最优的交互式分割性能；3）对点提示与盒子初始化两种交互模式提供了系统性比较。

**🔧 技术方法**

技术手段包括：使用三维全卷积网络架构（细节未公开），从头训练（不使用预训练权重），并采用点/盒子提示结合误差最大化的自适应校正策略。

**📊 数据集**

数据集为 Deepwise 内部 849 个医学图像数据集的精选子集，经过三阶段筛选后得到 115k 体积，随后在 219 个来源数据集的 4,326 个测试案例（共 13,826 个物体，107 个医学类别）上进行评估。

**📈 对比分析**

与 nnInteractive 及其他基线（MedSAM2、SAM‑Med3D、SegVol、VISTA3D）在相同模拟交互模式下比较：V1 在点提示下 1 点/5 点的类别宏平均分别为 0.5764/0.7771，盒子初始化后 0/5 校正分别为 0.7130/0.8002，均优于基线，且在 CT/MR 两种模态上保持领先。

**⚠️ 局限性**

局限性包括：评估仅基于模拟交互，未涵盖真实用户的提示误差与时间成本；数据集与测试集来自同一组织，未验证跨机构/跨数据集的泛化；目前仅覆盖 CT/MR，未涉及 PET、超声等其他模态；模型细节与训练细节未公开。

---

## 303. Annual Earth-observation embeddings encode wildfire disturbance and support simplified burned area mapping

**arXiv ID:** 2609.25731 | [PDF](https://arxiv.org/pdf/2609.25731v1)

**作者:** Jovana Knezevic `[一作]` (University of Cambridge), David Coomes `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究利用预训练的年度 Earth‑observation 嵌入（Tessera、AlphaEarth）在不需要事件特定影像或密集时间序列处理的情况下，对单火场、全年火场、加州及欧洲火灾进行烧毁面积映射，并尝试恢复火灾发生时间。

**💡 创新点**

创新点在于证明单一年度嵌入已编码火灾扰动，可在无事件定位信息的情况下实现高精度烧毁面积分割；同时实现跨大洲零训练迁移与时间回归，提供一种将时间推理迁移至嵌入层的全新工作流程。

**🔧 技术方法**

采用的技术包括时序地理基础模型 Tessera 与 AlphaEarth 的年度嵌入，配合轻量化下游模型（L2 逻辑回归、随机森林、轻量 U‑Net），以及针对不同任务的评估协议（单火场、年火场、墙到墙、跨洲迁移、时间回归）。

**📊 数据集**

使用的数据集包括美国 HLS Burn Scars（2018–2021）与 MTBS 辅助标签、CAL FIRE 火线数据库、欧洲 Copernicus EMSR 快速映射（2024–2025），并结合相应的参考边界与严重度掩模。

**📈 对比分析**

通过对比单火场与全年火场、Tessera 与 AlphaEarth、事件影像与嵌入、以及跨洲迁移，Tessera 单嵌入在 F1（0.90+）、IoU（0.84）等指标上与预/后影像基线持平甚至更优；在加州墙到墙应用中恢复 97% 参考烧毁面积；在欧洲迁移中取得 F1=0.88、IoU=0.79；时间回归 MAE 约 13 天，跨洲 MAE 10.5 天。

**⚠️ 局限性**

主要局限包括：年度嵌入对年末火灾识别弱，导致召回下降；嵌入仅能提供回顾性地图，无法实时监测；墙到墙部署时假阳性较多，需更多负样本或后处理；不同季节、时区的迁移仍需进一步验证。

---

## 304. Self-Supervised Combinatorial Optimization with Constraints via Frank-Wolfe

**arXiv ID:** 2609.25728 | [PDF](https://arxiv.org/pdf/2609.25728v1)

**作者:** Akbar Rafiey `[一作]` (New York University), Nikolaos Karalias `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种投影无关的自监督组合优化框架，利用Frank–Wolfe分解将神经网络输出映射为可行多面体内的稀疏分布，并以该分布对离散目标的期望作为可微损失直接训练网络，推理时自动完成取舍。

**💡 创新点**

核心创新在于不要求网络输出落入可行多面体，而是通过基于LMO的Frank–Wolfe几何分解生成稀疏可行表示，既保证了自监督训练的可微性，又提供推理时的收敛与取舍保证，并可统一适用于多种组合优化问题。

**🔧 技术方法**

使用技术包括Frank–Wolfe分解算法、近似Carathéodory、线性最大化Oracle、自动微分、GAT/GraphSAGE编码器，以及必要时的Sinkhorn投影等。

**📊 数据集**

数据集涵盖：Maximum Coverage 采用合成随机图与真实图数据；QAP 使用合成实例与 QAPLIB 基准；TSP 采用随机欧氏点集（TSP‑500、TSP‑1000）以及其他标准 TSP 实例。

**📈 对比分析**

与传统求解器（Concorde、Gurobi）、启发式算法（LKH3）、无监督神经基线（GeoNCO、Sym‑NCO、RGM、SAWT、DIFUSCO）以及有监督基线（COExpander）等进行对比，实验显示在无标签自监督设置下该方法能够与或超过大多数无监督神经基线，在 QAP 甚至达到最优性能，同时推理速度更快；在有监督基线下仍略逊，但显著降低了对标注数据和计算资源的需求。

**⚠️ 局限性**

局限性包括：在严重分布偏移（如 QAPLIB）时需要额外投影；对线性最大化 Oracle 的依赖导致某些多面体的效率受限；未深入研究网络架构对性能的影响；与最强有监督方法仍存在性能差距。

---

## 305. Designing an Efficient Excavator Bucket for Lunar ISRU: A Comparative Study with Vision-Based Fill and Displacement Analysis

**arXiv ID:** 2609.25724 | [PDF](https://arxiv.org/pdf/2609.25724v1)

**作者:** Abdulla Hil Kafi `[一作]` (Kyushu Institute of Technology), Kenji Nagaoka `[通讯]` (Kyushu Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文设计并实验验证了一种螺旋腔轮用于月球土壤开挖，并评估其填充率、渗深与特定能量；

**💡 创新点**

其创新点在于螺旋腔结构显著提升土壤保持，提出土壤保持指数η，并结合轻量化视觉与日志工具实现无外部仪器的开挖性能评估；

**🔧 技术方法**

采用3D打印（FDM PLA）制造轮体，利用实验平台与H​​EBI X8‑16智能驱动记录扭矩功率，配合图像分割计算填充率、DEM仿真验证粒子运动；

**📊 数据集**

实验以硅砂（bulk density 1282 kg/m³、粒径 510 µm）模拟月球土壤，在5/10/15 RPM下进行四转实验；

**📈 对比分析**

与桶鼓式轮和Keio轮基线比较，螺旋腔轮在填充率、开挖速率上提升约15–29%，开挖速率提高2.2–3.0倍，特定能量降低29%，渗深虽稍大但每单位能量与渗深的回收率更高；

**⚠️ 局限性**

局限在于仅在地球重力和真空缺失条件下测试，未评估低重力、真空、粒子磨蚀对轮体耐久性与性能的影响，且仅针对非黏性沙土，需进一步验证更致密或粘性土壤的适用性。

---

## 306. CogenPVG: Cognitive-Enhanced Reflective Multi-Agent Framework for Persuasive Video Generation

**arXiv ID:** 2609.25821 | [PDF](https://arxiv.org/pdf/2609.25821v1)

**作者:** Yuntian Xiao `[一作]` (Beihang University), Shuai Li `[通讯]` (Beihang University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于认知增强的反思多代理框架CogenPVG，用于生成具有高度说服力的视频。

**💡 创新点**

首次将双路ELM说服理论与批判与生成对称的反思机制结合，既提升中心路理性论证，又增强外围路情感与启发式提示。

**🔧 技术方法**

采用大型语言模型（GPT‑5.4与GPT‑4o）与多模态生成工具（Seedream、Suno等），并在每个阶段构建生成‑批评对，以实现迭代优化。

**📊 数据集**

构建了72个主题‑立场组合的Persuasion Goals集合（来源于PVP数据集），用于训练与评估。

**📈 对比分析**

与MM‑StoryAgent、Anim‑Director、VideoGen等基线及其降级版本进行人类与LMM评估，CogenPVG在说服力、态度转变和多维主观指标上均显著优于对手，赢率和平均态度提升均位居首位。

**⚠️ 局限性**

依赖昂贵的预训练模型与多工具调用导致算力与成本高，且对LMM评估可靠性与可解释性仍需深入研究。

---

## 307. MOLA LiDAR-Inertial Odometry (MOLA-LIO) on the COMFORT Localization Benchmark

**arXiv ID:** 2609.25813 | [PDF](https://arxiv.org/pdf/2609.25813v1)

**作者:** Jose Luis Blanco-Claraco `[一作]` `[通讯]` (University of Almeria), Jose Luis Blanco-Claraco (University of Almeria)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在COMFORT定位基准赛中提交MOLA-LIO系统，实现了三维LiDAR-惯性测量融合的实时里程计。

**💡 创新点**

创新点包括利用IMU融合提升姿态估计、基于协方差的ICP匹配、以及异步局部地图重建以减少插入延迟。

**🔧 技术方法**

主要技术有MOLA模块化SLAM框架、Generalized-ICP + 点协方差匹配、GTSAM iSAM2滑动窗口平滑器以及后台k‑d树重平衡。

**📊 数据集**

使用GrandTour数据集（Boxi LiDAR+IMU）进行评估。

**📈 对比分析**

在六个任务上平均ATE为3.59 cm，平均时钟占比0.60，CPU使用约7个核心，单帧延迟≈2.2 ms，达成实时性能。

**⚠️ 局限性**

局限性包括缺乏回环闭环、对垂直漂移无约束、对后台重建线程的非确定性导致轨迹可重复性不足。

---

## 308. Multi-View Fair Clustering Guided by Cross-View Sensitive Information Discrepancy

**arXiv ID:** 2609.25811 | [PDF](https://arxiv.org/pdf/2609.25811v1)

**作者:** Mudi Jiang `[一作]` (Dalian University of Technology), Zhikui Chen `[通讯]` (Dalian University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于跨视角敏感信息差异引导的多视角公平聚类框架；

**💡 创新点**

创新点包括：①估计各视角对敏感属性的依赖并按差异进行偏差排名；②设计非对称视角对齐机制，使高敏感视角向低敏感视角学习，并以差异自适应调节对齐强度；③在聚类结果上加入群体公平正则，提升群体公平性；

**🔧 技术方法**

使用技术包括：多视角自编码器、NHSIC用于敏感依赖估计、对比学习式非对称对齐损失、软分配平均差距公平正则、联合优化框架；

**📊 数据集**

实验数据集涵盖五个多视角/单视角数据集：Credit、Bank、Law、Mfeat、COIL；

**📈 对比分析**

通过与单视角公平聚类方法（BFKM、VFC、FFC、FairDen）、传统多视角聚类方法（MCPL、CGL、3MC）、以及现有多视角公平聚类方法（FairMVC、FMSC、AFMVC、FLFMVC）比较；在ACC、NMI、BAL、DPD等指标上，方法在大多数数据集上取得NMI排名第一、ACC第二、BAL第二，表现出良好的聚类质量与群体公平平衡；消融、参数敏感性和鲁棒性分析进一步验证方法的优势；

**⚠️ 局限性**

局限性包括：对敏感依赖的估计和对齐机制对大规模高维数据计算成本较高；仅针对二元敏感属性，未考虑多类别或连续敏感属性；仅在静态批处理场景下实验，未处理缺失或流式多视角数据。

---

## 309. When Are Aggregate Agent Traces Diagnosable? Traffic-Governed Interpretation and Calibrated Abstention

**arXiv ID:** 2609.25806 | [PDF](https://arxiv.org/pdf/2609.25806v1)

**作者:** Peiying Zhu `[一作]` (Blossom AI), Sidi Chang `[通讯]` (Blossom AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在闭环代理系统中，如何通过先验与运行时的支持门控来判断聚合轨迹是否能用于故障诊断，并验证交通曝光（affected clean traffic）比单元覆盖更能体现诊断信号。

**💡 创新点**

提出两层门控流程（reference‑map gate 与 runtime two‑stream gate），引入交通曝光作为曝光度指标，并对最小命中集（MHS）在此环境下无优势的结论进行负面验证。

**🔧 技术方法**

使用门控统计检验、负对数似然比较、bootstrap 置信区间、分布检测（frozen split‑conformal），以及 MHS 与贪心搜索的对比。

**📊 数据集**

基于一款仿真酒店定价代理（24 个物理组件，3 个需求 regime），在 15 个 seed 产生的 160 条 episode 组装成 15 个 reference 与 15 个 current 分区。

**📈 对比分析**

通过对比交通曝光模型与单元覆盖模型，发现交通曝光在所有 540 行的负对数似然上平均提升 0.1264 nats（29.3%），且在两组 mask family 下满足 0.01 nats 的实际充分性阈值；门控后稳定假阳性率为 0/20，满足安全上限 0.20。

**⚠️ 局限性**

局限性包括仅在单一仿真环境和单一动作桶偏移上验证；组件数仅 24，bootstrap 置信区间可能低估置信度；门控阈值与检测器基于特定 seed，转移到实际系统需重新校准；未验证真正模糊轨迹下的定位性能。

---

## 310. The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks

**arXiv ID:** 2609.25804 | [PDF](https://arxiv.org/pdf/2609.25804v1)

**作者:** Wenbo Pan `[一作]` (City University of Hong Kong), Xiaohua Jia `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个自动化的长周期任务决策质量评估基准（TasteBench），并证明可以通过蒸馏学习显著提升LLM代理的“品味”能力。

**💡 创新点**

① 定义并量化“品味”为在决策点选择更优方向的能力；② 利用轨迹自身的结果无人工标注地挖掘决策“fork”；③ 通过自监督蒸馏将教师的推理知识内化到学生模型，实现品味的可训练性；④ 通过建议注入提升终端任务成功率。

**🔧 技术方法**

自动挖掘与过滤决策fork、双向候选顺序评估、基于SDPO的自监督蒸馏、在SWE-bench Pro上注入建议评估。

**📊 数据集**

来自GPT-5.4/5.5在517个SWE-bench Pro任务的工程轨迹，以及1,132个AI R&D任务轨迹（RE-Bench、HCAST），最终生成的TasteBench包含502道题目（工程390题、研究112题）。

**📈 对比分析**

在TasteBench上对11款前沿LLM（Claude、GPT、Grok、DeepSeek、GLM、MiniMax、Mistral等）进行双顺序评估，平均准确率约45%，最佳Claude‑5.6 Sol约58%；与SWE-bench Verified相关系数0.63，显示两者不完全重合；蒸馏后学生模型在未见任务上从30%提升至48%准确率，且在SWE-bench Pro上的成功率从14.6%提升至33.7%。

**⚠️ 局限性**

局限性：仅覆盖两类轨迹（工程与研究），对fork的挖掘依赖多次尝试或自我纠错；在长时间窗的fork上准确率仍接近随机；蒸馏效果受教师示例质量影响，且未在更大规模或不同领域任务上进一步验证。

---

## 311. When Point Clouds Outperform Pixels: Rethinking Zero-Shot Multimodal Anomaly Detection

**arXiv ID:** 2609.25793 | [PDF](https://arxiv.org/pdf/2609.25793v1)

**作者:** Chenglin Ye `[一作]` (University of Chinese Academy of Sciences), Yunbiao Wang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种零样本多模态异常检测框架WOOPS，专门解决 RGB 与点云在零样本迁移下可靠性不均衡的问题。

**💡 创新点**

创新点包括：① 多视角信息解耦 (MID) 模块，用来抑制多投影点云的异质信息并突出几何一致特征；② 可靠性校准 (MRC) 模块，利用点云特征评估 RGB 可信度并自适应地加权融合；③ 通过严格阈值指标（mF1, mAcc, mIoU）重新评估零样本多模态异常检测性能，揭示点云更可靠的事实。

**🔧 技术方法**

技术手段：冻结 CLIP 视觉/文本编码器、跨注意力机制、双层感知解耦网络、分布一致性与 GMMD 损失、Focal 与 Dice 监督、t-SNE 可视化等。

**📊 数据集**

使用公开的 MVTec 3D-AD 与 Eyecandies 两大多模态异常检测基准数据集进行实验。

**📈 对比分析**

与 AnomalyCLIP、FAPrompt、GS-CLIP、PointAD、ZUMA-FT 等现有方法对比，WOOPS 在新的严格评估指标下在点云单模态、RGB单模态以及多模态配置均取得最优或竞争性表现；在传统指标上表现与部分方法相当，但在严苛指标上显著优于先行工作。

**⚠️ 局限性**

局限性：在传统指标（如 pixel-level AUROC）下多模态融合有时会出现性能下降，说明当前的 MRC 校准可能无法完全避免多模态崩溃；同时依赖多投影渲染，对点云质量和视角选择仍有一定敏感性，需要进一步研究更鲁棒的融合策略。

---

## 312. Evaluating Accuracy and Probabilistic Reliability of Zero-Shot Time Series Foundation Models

**arXiv ID:** 2609.25788 | [PDF](https://arxiv.org/pdf/2609.25788v1)

**作者:** Panagiotis Michael `[一作]` (University of Nicosia), Demetris Trihinas `[通讯]` (University of Nicosia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对六种零样本时间序列基础模型在能源、交通、金融三类数据集上的准确性和概率可靠性进行基准评估。

**💡 创新点**

系统比较了点预测精度与置信区间校准之间的权衡，并揭示不同架构（xLSTM、patch‑transformer、传统transformer）在不同情境下的优势。

**🔧 技术方法**

使用零样本推理、量化词化、生成流匹配以及多分位数预测等技术，评估指标包括 sMAPE、ICE、IMAE。

**📊 数据集**

EdgeTraffic（高频交通计数）、MixGridPL（波兰电网负荷）和 SPY（标普500 ETF 收盘价）三类数据集。

**📈 对比分析**

与统计基线（ARIMA、RA、RWD）和监督式 DL 基线 PatchTST 对比，六模型在短期往往与基线相当或优于，长周期时 TSFMs 通常更精准，但部分模型置信区间失真。

**⚠️ 局限性**

仅限单变量零样本预测，缺乏对多变量、缺失/不规则采样和概念漂移的鲁棒性评估，且未针对计算效率做深入探讨。

---

## 313. Adaptive Traffic Camouflage: Causal and Resource-Aware Defense Against IoT Fingerprinting

**arXiv ID:** 2609.25787 | [PDF](https://arxiv.org/pdf/2609.25787v1)

**作者:** Daniel Adu Worae `[一作]` (University of Notre Dame), Nitesh V. Chawla `[通讯]` (University of Notre Dame)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于实时流量泄漏估计的自适应网络层伪装控制器（Adaptive Traffic Camouflage），能够在保证通信带宽和延迟预算的前提下对 IoT 流量进行时序、大小、拆分、覆盖等多维度的动态伪装。

**💡 创新点**

创新点：
- 先行估计上一窗口的四类泄漏特征（大小、时序、批量/体量、方向/拆包），并将其与预先校准的攻击器损失关联，形成可测量的泄漏谱；
- 将资源约束（带宽/延迟）直接嵌入动作可行性判断，避免后置权衡；
- 结合全局最优选择与基于窗口上下文的回归重排，并加入置信门限实现置信驱动的 Causal 选择；
- 通过多维动作集（填充、拆包、微批、覆盖及其组合）提供多样化的防御空间。

**🔧 技术方法**

技术细节：
- 离线校准阶段：训练固定的指纹识别代理、计算每类泄漏对动作的影响下界、测量动作的带宽/延迟成本、训练每个动作在每个资源档位下的下一窗口收益回归模型；
- 运行时：利用已完成窗口的 24 维流量特征和四维泄漏谱，先选取满足资源约束的动作集合；使用全局选择评估各动作对泄漏谱的残留值；若满足置信阈值则通过上下文回归模型重新排序并可能覆盖全局选取；
- 动作实现：填充、拆包、微批、覆盖等在数据包层面实时执行。

**📊 数据集**

数据集：CIC‑IoT‑2022、IoT Sentinel、UNSW 三个公开 IoT 流量数据集，分别包含 6 类设备，并按 Train/Calibration/Adapt/Test 四份拆分。

**📈 对比分析**

对比方法：
- 固定最优单一动作、随机混合动作、均衡带宽匹配静态周期动作与自适应控制器；
- 评估指标：Macro‑F1（越低越好）以及相对带宽/延迟开销；
- 结果：在 Balanced 预算下，Macro‑F1 相比无伪装降低 13–23%（平均带宽开销 5–7%）；在 Privacy 预算下降低 28–44%（带宽 10–14%）；随机/固定/静态基线在不同数据集表现差异，Adaptive 在 IoT Sentinel 上显著优于基线；在面对防御感知训练的攻击者时，性能恢复与数据集相关，Adaptive 在部分数据集仍保持隐私优势。

**⚠️ 局限性**

局限性：
- 只抑制流量形状特征，协议、端点、端口、流统计等元数据仍可泄露；
- 训练好的攻击模型在见到受保护流量后能快速恢复，尤其在 CIC‑IoT‑2022 与 UNSW 数据集；
- 资源预算基于保守校准估计，实际运行中仍可能出现小幅预算超支；
- 动作空间预定义，若特定数据分布对某类动作效果极佳，静态强力动作可能优于自适应；
- Causal 设计仅使用前一窗口信息，虽损失极小，但在极端时序变化场景下可能不如完整窗口判断。

---

## 314. Hot-Cold Tiering of HBM and High Bandwidth Flash for Agentic LLM Serving

**arXiv ID:** 2609.25782 | [PDF](https://arxiv.org/pdf/2609.25782v1)

**作者:** Jongjin Baek `[一作]`, Joo-Young Kim `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计热冷分层的 HBM+HBF 内存层次，将主动 KV 放在 HBM、闲置 KV 放在 HBF，从而支持更多会话并降低能耗。

**💡 创新点**

识别 agentic LLM KV 的双模访问模式，提出 write‑on‑evict 策略，最大限度减少 HBF 写入并保持低恢复延迟。

**🔧 技术方法**

使用 HBM、HBF 片上堆叠、D2D 链路、LRU 置换、预取、Trace‑based 仿真与能耗模型。

**📊 数据集**

采用 Qwen3‑Coder‑30B‑A3B 与 SWE‑bench 真实软件工程任务轨迹。

**📈 对比分析**

与全 KV Flash、CPU PCIe5、NVLink、重计算等基线对比，热冷分层在 48 活动会话下每令牌 14 ms，利用 3 TiB HBF 显著提升并发数，同时功耗下降 7.6 kW。

**⚠️ 局限性**

对 HBF 的耐久性和高读能耗仍有限制，尤其在高并发下写入频繁时寿命下降；依赖高速 D2D 链路。

---

## 315. On the Construction of Trapdoor Claw-Free Functions with Certifiable Key

**arXiv ID:** 2609.25819 | [PDF](https://arxiv.org/pdf/2609.25819v1)

**作者:** Charles Lim `[一作]` (National University of Singapore), Yao Ma `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种与具体 Trapdoor Claw‑Free Function (TCF) 家族无关的关键认证抽象，能够在任何 TCF 基础的量子证明或交互式协议中附加一个仅证明键合法性的零知识证书。

**💡 创新点**

核心创新在于：①引入“可认证键关系”(certifiable key relation) 与“已认证键生成”(certified key generation) 两个抽象定义；②证明任何可证明知识的零知识证明系统即可实现键生成的可提取与密钥隐私；③给出通用编译器将任意 TCF‑基础的量子证明转为零知识证明，并阐明此方法的适用边界。

**🔧 技术方法**

使用的技术包括：零知识证明与知识证明(Σ‑协议、Fiat‑Shamir、NIZK)、格基算法与 LWE 采样、抽象化的安全性质（完整性、可提取性、密钥隐私）、以及量子可证明量子性(QPIP)协议框架。

**📊 数据集**

该工作主要基于理论构造和抽象，不依赖具体的数据集；在实例化时使用了已公开的 LWE、因式分解与 DDH 基础的 TCF 家族。

**📈 对比分析**

对比方法：在已有的 LWE‑基础零知识证明量子性协议（如 Mahadev 论文）和 DDH‑基础协议中，将原始协议嵌入本框架得到的零知识版本在安全性证明上等价；性能方面，在保持可提取性的同时，零知识证明的开销由底层 NP 关系的 ZK 证明决定，通常为多轮交互或一次性 Fiat‑Shamir 变换，开销与原始协议相差可忽略。

**⚠️ 局限性**

局限性：当协议的安全性依赖于“注入不变性”(injective invariance)——即无法区分克隆自由族与注入族——时，附加的可认证证书会泄露关键族信息，导致安全失效；因此本框架不适用于 Mahadev 的测量协议和 Gheorghiu‑Vidick 的远程态制备等协议。

---

## 316. CacheDyG: Decoupling Temporal Propagation for Efficient Dynamic Graph Learning

**arXiv ID:** 2609.25814 | [PDF](https://arxiv.org/pdf/2609.25814v1)

**作者:** PinHeng Zong `[一作]` (Southwest University), Ye Yuan `[通讯]` (Southwest University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 CacheDyG 框架，利用可重用的非可训练时序图缓存，只在每个 epoch 中更新轻量级频域校正器和链接预测器，显著降低动态图学习的计算与参数开销。

**💡 创新点**

将时序图传播与参数更新解耦，采用时序混合矩阵构建图感知缓存、频域节点域校正、残差门控以及选择性缓存刷新，实现高效的训练与推理。

**🔧 技术方法**

使用时序混合矩阵、非可训练节点‑时间缓存、FFT 频域校正器、残差门控机制以及轻量化链接预测分类器等技术。

**📊 数据集**

在 Wiki‑Eo、Digg、Alpha、DBLP、StackOverflow 五个动态图数据集上进行实验。

**📈 对比分析**

与 DySAT、ROLAND、EvolveGCN、WinGNN、GTCN、SGD‑DYG 等基线比较，CacheDyG 在所有数据集上均获得最高或接近最高的 AP/ROC‑AUC，同时仅使用约 19k 可训练参数，训练时间最短。

**⚠️ 局限性**

主要局限在于仅针对固定节点、顺序化 snapshot 的转导式训练，对异步事件流或更大规模动态图的缓存刷新策略与多任务/因果推断场景的适用性尚未充分验证。

---

## 317. Auditing Proxy-Based Validation Across Text Spans

**arXiv ID:** 2609.25808 | [PDF](https://arxiv.org/pdf/2609.25808v1)

**作者:** Daein Weon `[一作]` (Kookmin University), Dong Ho Kang `[通讯]` (UStechlab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在相同文本跨度内使用代理标签验证模型得分的构造效度问题，并提出了“验证合同”框架与离散跨度重读控制方法。

**💡 创新点**

创新点在于将跨度声明为验证合同的一项字段，提出离散跨度重读、基于自举和置换检验的残差评估，从而揭示代理标签与构造之间的表面耦合现象。

**🔧 技术方法**

采用了TF‑IDF余弦距离、关键词代理、HotpotQA、OR‑Bench、XSTest等得分与代理方案，并使用自举、置换检验和阈值比较技术。

**📊 数据集**

主要数据集包括HotpotQA、OR‑Bench、XSTest、GSM8K、HotpotQA 50字符/80字符跨度、JailbreakBench等公开生成文本。

**📈 对比分析**

通过在不同跨度下重新计算代理标签，并与构造标签AUC比较，发现短跨度高一致性主要来自共享表面信息；残差GAP在离散跨度上显著降低，表明构造排名失效。

**⚠️ 局限性**

局限在于需要公开逐例生成文本、代理规则可复制、构造标签可靠性不足，以及离散跨度控制只能说明共享表面证据不足，无法区分构造-代理不匹配与溢出等原因。

---

## 318. Reply to comments arXiv:2512.07881 and arXiv:2601.06104 on quantum structure in human and AI-generated language

**arXiv ID:** 2609.25797 | [PDF](https://arxiv.org/pdf/2609.25797v1)

**作者:** Massimiliano Sassoli de Bianchi `[一作]` (Vrije Universiteit Brussel), Roberto Leporini `[通讯]` (University of Bergamo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

回应对两篇关于人类与人工智能语言的量子结构研究（arXiv:2407.14924 与 arXiv:2511.21731）的评论，澄清实验设计、统计分析及理论解释。

**💡 创新点**

创新点在于：①将贝索-爱因斯坦统计与 CHSH 量子相关性结合，用于检测 LLM 与人类文本中的“量子化”特征；②在 Contextuality-by-Default 框架下讨论边际律违反与“纠缠”概念的关系；③阐明词能量与频率排序、向量空间与量子态空间的区别与联系。

**🔧 技术方法**

采用量子统计方法（CHSH 不等式、贝索-爱因斯坦分布）、上下文敏感度分析、向量空间模型与贝塞尔函数拟合，配合对 LLM 生成文本与人类文本的频率分布和句子结构进行比较。

**📊 数据集**

使用公开大型语言模型（如 GPT‑系列）生成的文本与人类参与者提供的文本数据；数据集包括多语言文本样本以及实验设计中构造的概念组合问卷。

**📈 对比分析**

比较方法：对人类与 LLM 产生的响应进行 2×2 试验（测量 A/B），计算相关函数 E(A,B) 并求取 CHSH 值；同时拟合词频分布到贝索-爱因斯坦模型并评估拟合优度。结果显示两类系统均可出现 CHSH 违规与贝索-爱因斯坦分布，表明量子关联可在人工生成文本中出现。

**⚠️ 局限性**

局限性：实验方案探索性强，缺乏严格随机化与多次独立测量控制；边际律违反可能来源于实验设计；贝索-爱因斯坦拟合无法单独判断语义组织；词能量与频率映射仍是经验性约定；向量空间与量子态空间的映射关系尚未明确定义。

---

## 319. A Lightweight Plastic-Memory Framework for Graph Few-Shot Class-Incremental Learning

**arXiv ID:** 2609.25781 | [PDF](https://arxiv.org/pdf/2609.25781v1)

**作者:** Zihan Mei `[一作]` (University of Electronic Science and Technology of China), Qinli Yang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种轻量化可塑性记忆框架（LPMC）用于图形少样本增量学习（GFSCIL），通过微聚类动态更新类原型并结合双循环元学习实现知识保留与新任务快速适应。

**💡 创新点**

核心创新在于可塑性记忆模块的微聚类结构，既压缩存储又保持类间可区分性；同时在内循环使用元学习、外循环采用图伪增量学习和原型蒸馏，实现高效、低耗的增量学习。

**🔧 技术方法**

使用了自监督对比学习（SimGRACE）预训练，GNN编码器（如GAT）、原型网络、DBSCAN微聚类、记忆蒸馏、双循环元学习（MAML类）以及图伪增量学习框架。

**📊 数据集**

在Amazon Clothing、CoraFull、Coauthor‑CS、Computers四个公开图数据集上进行评估，采用多种N‑way K‑shot设置。

**📈 对比分析**

与Mecoin、HAG‑Meta、Geometer以及EWC、LwF、TWP、GEM、MAS、ER‑GNN等九个基线对比，LPMC在所有数据集均实现了更低的性能下降（PD）与更高的平均准确率，且训练时间显著缩短。

**⚠️ 局限性**

主要局限包括对极大规模动态图的扩展性待验证、对DBSCAN超参数的敏感性以及在极少标签情况下微聚类的稳定性尚需进一步研究。

---

## 320. Video-HopChain: Multi-Hop Questions and Confidence-Gated Exploration for Video Reasoning Models

**arXiv ID:** 2609.25773 | [PDF](https://arxiv.org/pdf/2609.25773v1)

**作者:** Trung Nguyen Quang `[一作]`, Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了一个22,550条多跳视频问答数据集 VideoHopChain，并在其上使用GRPO训练 Qwen3‑VL‑8B，提出了 Top‑Token Mask 方法来恢复零方差组，显著提升视频推理性能。

**💡 创新点**

创新点一：首次为视频推理设计多跳问题链，强制模型多次观察视频；创新点二：在GRPO中引入 Top‑Token Mask，以少量额外采样补偿零方差组，避免了额外 roll‑outs 的开销。

**🔧 技术方法**

使用的技术包括：Group Relative Policy Optimization（GRPO），Top‑Token Mask 掩码交互，Qwen3‑VL‑8B 大语言模型，视频帧采样与 MiniMax‑M3 生成字幕，基于 LLM 的问题生成与验证。

**📊 数据集**

使用的数据集是自研的 VideoHopChain（22,550 训练题目 / 13,378 视频，1,000 评测题目 / 1,000 视频），并对比 105,993 行公开多选视频 QA 数据集（LLaVA‑Video、STAR、CLEVRER 等）。

**📈 对比分析**

在八大公开视频推理基准上，标准 RL 训练平均 55.4 分；加上 VideoHopChain 后提升至 57.9 分；再加 Top‑Token Mask 后提升至 59.3 分，成为该基准集合中性能最佳的开源模型，并在大多数单项指标上均名列第一。

**⚠️ 局限性**

局限性包括：数据完全合成，可能缺乏真实场景的复杂性；多跳问答仍以是/否 + 数字求和为核心，难以扩展到更复杂的推理形式；方法依赖于 RLVR 结构，尚未验证在其他模型或更大规模数据上的普适性。

---

## 321. Reading Right, Answering Wrong: How Visual Configuration Changes Affect Evidence Use in VLMs

**arXiv ID:** 2609.25770 | [PDF](https://arxiv.org/pdf/2609.25770v1)

**作者:** Dingyang Lin `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型（VLM）在图像尺寸微调导致的视觉配置边界上的答案稳定性，并提出通过显式引导读取和注释辅助的方式恢复错误答案。

**💡 创新点**

发现视觉配置边界会显著增加答案翻转率，证明配置变更本身是导致不稳定的重要因素；同时证明通过引导阅读可将97.2%可读信息错误恢复为正确答案。

**🔧 技术方法**

采用动态分辨率 VLM（如 Qwen3‑VL、MiniCPM‑V、LLaVA‑NeXT、InternVL 等），使用视觉配置对比、固定图像/固定配置实验、注意力剪枝（attention knockout）、引导阅读和注释辅助恢复等技术。

**📊 数据集**

在四大 VQA 基准（VQAv2、TextVQA、DocVQA、ChartQA）以及视觉-CoT、SROIE、InfoVQA 等数据集上进行实验。

**📈 对比分析**

通过交叉配置对比、固定图像/配置实验、注意力干预和逐步增强指导进行评估，结果显示跨配置的答案翻转率平均提升约5.6个百分点，配置变更导致的翻转率比仅变更像素高3.6–7.9个百分点；引导阅读的恢复率达到97.2%，远优于无指导的18.5%。

**⚠️ 局限性**

恢复方法仅适用于可读目标信息，无法自动化处理不可读信息；对更广范围的配置变化和模型对非可读信息的鲁棒性仍未得到充分验证。

---

## 322. The Limits of Simulated Societies: How Post-Training and Survey Fine-Tuning Erase Cross-Cultural Variance

**arXiv ID:** 2609.25760 | [PDF](https://arxiv.org/pdf/2609.25760v1)

**作者:** Rojin Ziaei `[一作]` `[通讯]` (Georgetown University), Rojin Ziaei (Georgetown University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对多语言大模型在模拟世界价值观调查中对个体回答的准确性与多样性进行诊断评估。

**💡 创新点**

提出“共识崩塌”概念，量化模型在对齐训练后对回答分布压缩的现象。

**🔧 技术方法**

使用零拷贝、SFT、DPO、GRPO等微调技术与温度采样及先验混合等后处理方法。

**📊 数据集**

基于世界价值观调查（WVS）12国、10,000问答对的跨文化数据集。

**📈 对比分析**

与多模型基线比较，发现最优模型准确率57.9%但只保留50%的人类方差，温度或RL无效，先验混合略恢复。

**⚠️ 局限性**

仅评估6个WVS问题，未检验更高温度/多样性采样，且仅使用英语提示，缺乏对其他文化量表的验证。

---

## 323. LiFR v2: Completion-Augmented Event Propagation for High-Rate Dense Prediction

**arXiv ID:** 2609.25803 | [PDF](https://arxiv.org/pdf/2609.25803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 324. Minimal Recurrent Behavioral Memory for Imitation under Partial Observability

**arXiv ID:** 2609.25757 | [PDF](https://arxiv.org/pdf/2609.25757v1)

**作者:** Xianyao Li `[一作]`, Jing Du `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了在部分可观测环境下，为给定专家策略实现最小递归行为记忆的方法，并给出可测量的实验协议。

**💡 创新点**

创新点包括：① 用兼容关系和可传递性定义“最小递归记忆”并证明其信息熵下界；② 在非传递情形下引入闭兼容状态分配来精确刻画最优记忆；③ 设计“sole‑carrier”离散记忆实现并提供实验级的代码率测量；④ 引入事件无关的未来行为监督以帮助学习者捕获所需记忆。

**🔧 技术方法**

技术手段：信息理论分析（熵、可压缩性、可传递性检验）、符号模型枚举与后向精炼、离散VQ‑VAE 体系的单一记忆载体、带条件先验的重构损失、事件无关的未来行为监督损失、以及在各种控制任务上的实验评估。

**📊 数据集**

使用的基准包括：Task A（隐藏质量推断）、readout‑2（隐藏模式推断）、A'（两步延迟决策）、bsuite 记忆链、Passive T‑maze。所有任务均在统一的符号化观察约定下进行。

**📈 对比分析**

与理论下界对比，实验中学习得到的代码率在大多数任务（尤其是 2‑bit 需求）与最优熵下界极为接近；通过足够的样本与超参数搜索，获得的闭环成功率常超过 0.9，表明记忆实现既足够又紧凑。相比传统识别头或连续 RNN，所提出的单一离散载体在记忆占用与任务成功率上表现更优。

**⚠️ 局限性**

局限性包括：① 该最优记忆理论目前只能在可枚举的有限实例上精确求解；② 传递性假设并非普适，非传递实例需要更复杂的闭兼容分配与启发式求解；③ 学习过程对超参数（如 β、λ）高度敏感，长延迟下成功率下降；④ 评估基于符号化模型，实际连续控制系统中可能存在额外的误差与噪声。

---

## 325. Modular Norm RandOpt: Population-Efficient Ensembling through Architecture-Aware Perturbations

**arXiv ID:** 2609.25745 | [PDF](https://arxiv.org/pdf/2609.25745v1)

**作者:** Kirato Yoshihara `[一作]` (University of Osaka), Hiroaki Hamade `[通讯]` (University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于模块自然范数和递归校准尺度的“模块化随机优化”（Modular Norm RandOpt），替换 RandOpt 的等尺度扰动，生成更具代表性的候选集并通过多投票提升准确率。

**💡 创新点**

创新点在于利用Transformer架构的模块化自然范数对权重扰动进行几何校准，并将尺度递归传播到各层，从而在保持相同选择和投票流程的前提下，用更少的候选数实现更高的性能。

**🔧 技术方法**

使用模块自然范数（最大行ℓ2、谱范数、ℓ∞）、递归模块尺度校准、权重高斯扰动采样、Plurality投票、基准任务评估与多模型量化等技术。

**📊 数据集**

使用七个任务（Countdown、GSM8K、MBPP、ROCStories、USPTO‑50K、MATH‑500、OlympiadBench）以及多规模模型（Qwen2.5 0.5B/1.5B/3B、Llama 3.2 3B、Gemma 3 4B、OLMo 3 7B）。

**📈 对比分析**

与传统 RandOpt、迭代ES、MeZO、ZO‑Finetuner 等方法在候选数、ensemble size K、准确率和运行时间上进行对比。结果显示，Modular Norm RandOpt 在 Countdown 仅用 100 候选替代 300，GSM8K 用 25 候选替代 300，准确率提升，候选数减少 3–12 倍，Wall‑clock 约 2–12 倍加速。

**⚠️ 局限性**

局限性包括：对任务和模型的依赖性；跨模型规模/架构迁移效果不完全均衡；对校准参数与扰动半径的选择敏感；候选数减少未消除最终投票评估成本；机制分析主要基于 GSM8K，未完全阐明普适性。

---

## 326. GuidedRay: Diversity-Guided Direction Discovery for Targeted Hard-Label Black-Box Attacks

**arXiv ID:** 2609.25734 | [PDF](https://arxiv.org/pdf/2609.25734v1)

**作者:** Fei Yuan `[一作]` (Shandong University), Xiaoyun Wang `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在硬标签黑盒设置下，如何在高维符号空间中快速发现指向目标类别的攻击方向，并提出 GuidedRay 方案以显著提升攻击成功率。

**💡 创新点**

创新点在于将目标类别参考样本作为先验并通过多样性增强生成多方向候选，然后利用一次查询的 Fast Test 对候选进行筛选，解决了传统方法在目标方向稀疏问题上的瓶颈。

**🔧 技术方法**

使用的技术包括：目标参考样本增强、符号方向搜索、一次查询的 Fast Test、Ray Search 方向细化、以及多种数据增强（高斯噪声、随机旋转、随机裁剪、颜色抖动）等。

**📊 数据集**

实验数据集涵盖 CIFAR-10、CIFAR-100 和 ImageNet，并在未防御模型以及采用抗训练和 TRADES 防御的模型上进行评估。

**📈 对比分析**

与 Bounce、HSJA、Sign-OPT、Tangent、RayS 等五种主流决策式攻击在 500–5,000 次查询下对比，GuidedRay 在所有数据集和防御模型上均获得最高 ASR，AUC@5K 与 Q@20/Q@30 指标均显著优于基线。

**⚠️ 局限性**

局限性：依赖目标类别参考样本，若无参考样本或在极大规模模型下效果可能下降；仅在 L∞ 约束下验证，其他约束下的表现尚需进一步研究。

---

## 327. Automating Constructive Assessment with Large Language Models: Toward Scalable and Repeated Evaluation of Practical Competence

**arXiv ID:** 2609.25790 | [PDF](https://arxiv.org/pdf/2609.25790v1)

**作者:** Satoshi Takahashi `[一作]` (Nagoya University), Mari Sawada `[通讯]` (GLOBIS Corporation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过提示设计，实现了ChatGPT自动化生成HDR案例、评分及反馈，以实现高阶认知技能的评估。

**💡 创新点**

创新点在于无需微调，仅凭提示即可完成HDR的全过程自动化，提升了可重复性、即时性和低成本。

**🔧 技术方法**

主要技术是大型语言模型ChatGPT（GPT-4o）及其提示工程。

**📊 数据集**

使用了100名成人学习者在原始STP案例基础上收集的描述性回答及其人工评分数据。

**📈 对比分析**

实验对比显示，提示含示例的评分准确率、精确率、召回率和F1均达到98–100%，并且ChatGPT生成的反馈在说服力和有用性上与人工反馈无显著差异。

**⚠️ 局限性**

局限性包括对提示设计的高度依赖、仅验证STP情境且未检验跨学科泛化以及缺乏长期跟踪效果评估。

---

## 328. Scientific capabilities and deployment sustainability of small-scale LLMs in biological wastewater treatment

**arXiv ID:** 2609.25774 | [PDF](https://arxiv.org/pdf/2609.25774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 329. VisForce: Visual Grounding of Current and Desired Forces for Goal-Conditioned Dexterous Manipulation

**arXiv ID:** 2609.25785 | [PDF](https://arxiv.org/pdf/2609.25785v1)

**作者:** Jung-Woo Lee `[一作]`, Soo-Chul Lim `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将执行力与视觉信息统一映射到手指尖位置的 VisForce 框架，用于实现对多指机器人手的力感知与执行控制。

**💡 创新点**

创新点：① 在当前手腕图像上用 3D 箭头可视化每根手指的执行力，使得力信息通过视觉通道直接参与决策；② 在目标图像中同样将期望力可视化到对应手指尖；③ 通过目标条件交叉注意力实现当前视觉与目标视觉的显式融合，强化对力目标的依赖。

**🔧 技术方法**

技术方法：手腕相机与仿真相机的 6-DOF 对齐、MuJoCo 渲染力矢量、SigLIP 视觉编码器、跨注意力模块、LoRA 微调、流匹配损失等。

**📊 数据集**

数据集：在真实机器人平台（UR10 + RH56F1 手）上收集的 4 组任务数据，分别为 T1（力条件抓取）30 次、T2（杯子插入/瓶子倒水）25 次、T3（舌头辅助面包搬运）30 次、T4（滑动调节插孔）30 次，共 115 条演示。

**📈 对比分析**

对比方法：基线 π_0.5、State Force+Text、Visual Force+Text、VisForce w/o CA；实验显示 VisForce 在 T2 70%/T3 55%/T4 40% 的最终成功率远高于对比方法（最大 20%）并且在多阶段任务中保持更高的累计成功率。

**⚠️ 局限性**

局限性：① 依赖精确的相机对齐与仿真模型，误差会影响力可视化准确性；② 目标力可视化需要离线准备并存储目标图像，难以适应在线动态目标；③ 仅在特定的抓取与搬运任务上验证，未知对更广泛场景或更大规模的物体集合的泛化能力。

---

## 330. Disentangling Heterogeneous Traffic Dynamics for Multi-Step Traffic Forecasting via Adaptive Spectral Decomposition

**arXiv ID:** 2609.25777 | [PDF](https://arxiv.org/pdf/2609.25777v1)

**作者:** Zijun Huang `[一作]` (Beijing Normal-Hong Kong Baptist University), Guanyao Li `[通讯]` (Beijing Normal-Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了适应性频谱分解网络ADNet，能够将历史交通流量信号拆分为主导与残差两部分，并分别通过独立的Graph WaveNet分支进行多步预测。

**💡 创新点**

创新点在于：①引入可学习的互补频谱权重，让每个频率在主导与残差两部分中以不同比例分配；②采用组件特定的双分支架构，将主导与残差动态分别建模，从而更好地捕捉异质时序特征。

**🔧 技术方法**

技术实现包括：实数快速傅里叶变换与逆变换、可学习的sigmoid频谱掩码、双Graph WaveNet分支、端到端MAE优化。

**📊 数据集**

使用了TraffiDent数据集中的Alameda和Orange两个区域，5分钟间隔的主干道传感器流量序列。

**📈 对比分析**

与十种基线（包括DCRNN、STGCN、GWN、STWave等）在MAE/RMSE/MAPE、不同时间步（H3/H6/H12）和平均指标上对比，ADNet在24种组合中取得20个最佳、4个次佳，尤其在更长预测时步上优势显著。

**⚠️ 局限性**

局限性包括：仅在TraffiDent单一数据集上验证；模型仍依赖Graph WaveNet骨干，缺乏对其他时空网络的泛化评估；频谱权重初始化与训练时间对结果影响尚未系统探究。

---

## 331. TRACE: Trajectory Representation and Consistency Estimation for AI-Generated Video Detection

**arXiv ID:** 2609.25775 | [PDF](https://arxiv.org/pdf/2609.25775v1)

**作者:** Huangsen Cao `[一作]` (Zhejiang University), Yongwei Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TRACE框架，利用预训练Flow Matching视频模型的速度响应及其多流时间点轨迹特征，并通过跨帧速度差异建模一致性来检测AI生成视频。

**💡 创新点**

创新点包括①将预训练生成器的速度响应作为迁移性取证信号；②在多流时间点上提取轨迹表示并估计跨帧一致性；③引入真实中心轨迹优化（Real-Centered Trajectory Optimization），通过拉近真实视频特征中心并将生成视频推远，从而实现生成器不变的特征学习。

**🔧 技术方法**

使用的核心技术包括预训练的Flow Matching视频DiT、速度响应提取、跨时间点轨迹表示学习、速度差异一致性估计、真实中心轨迹优化（带EMA更新）、MLP分类器以及对抗式训练损失。

**📊 数据集**

主要在AIGVDBench基准（涵盖31种开源与闭源视频生成器）上进行评估，并在VideoFeedback、GenVideo、ComGenVid、Magic Videos、GVD、GVF六个公开数据集上做跨数据集验证。

**📈 对比分析**

与多种先进的图像/视频检测器（如DeMamba、DeCoF、V-PVP、STALL等）在AUC上进行对比，TRACE在开源子集取得97.95%（比最强基线提升12.87%），闭源子集95.11%（提升17.01%）。在六个额外数据集上平均ACC@5% FPR 85.48%，在四个数据集排名第一，整体显示出优异的跨生成器与跨数据集泛化性能。

**⚠️ 局限性**

局限性包括：①对极端压缩、模糊等干扰的鲁棒性仍有提升空间；②仅针对Flow Matching模型的速度响应进行研究，未验证对其他生成模型（如Diffusion）的适用性；③训练时仅使用单一合成源Open‑Sora，虽然表现出良好泛化，但在更广泛、多样化生成器上的验证仍待进一步研究。

---

## 332. MedVLA: A Hierarchical Vision-Language-Action Framework for Closed-Loop Precision Medical Robot Manipulation

**arXiv ID:** 2609.25756 | [PDF](https://arxiv.org/pdf/2609.25756v1)

**作者:** Junjie Xie `[一作]` (Chinese Academy of Sciences), Dapeng Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种层次化的视觉-语言-动作（MedVLA）框架，用高层多模态推理与低层函数约束执行结合，实现精准医疗机器人闭环控制。

**💡 创新点**

创新点包括：①将决策分为结构化函数调用，保证可审计与可约束；②引入状态感知的中间门控模块，实时验证函数合法性与安全性；③使用链式思维（CoT）提升中间判断；④设计多智能体数据生成管线，自动生成低冗余、结构化训练数据。

**🔧 技术方法**

使用大型视觉‑语言模型（如 Qwen2.5‑VL‑3B、InternVL2‑2B、Llama‑3.2‑11B‑Vision‑Instruct）进行微调，配合函数调用接口、门控验证和多智能体生成机制；在真实硬件上实现闭环控制。

**📊 数据集**

主要数据集由约 3,000 条真实记录（双目图像、功能标签、轨迹）扩展至约 30,000 条结构化样本（包含手工与自动生成的数据），用于训练与评估。

**📈 对比分析**

与 OpenVLA（8% 成功率）和 π₀（15% 成功率）等基线对比；在 100 次闭环实验中，MedVLA 在最多 6 轮决策下实现 95.0% 的插入成功率，显著优于基线。

**⚠️ 局限性**

局限性：实验仅在单一微米级插入任务上验证，数据生成方式可能偏向框架本身；门控规则仍为任务特定，缺乏通用医学约束，未来需在多任务与外部数据上进一步验证。

---

## 333. PLAT: Sparse Timed Keyframe Motion Tracking for Humanoid Control via Privileged Latent Transition Learning

**arXiv ID:** 2609.25754 | [PDF](https://arxiv.org/pdf/2609.25754v1)

**作者:** Zepeng Wang `[一作]` (Wuhan University), Zongqing Lu `[通讯]` (BeingBeyond)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 PLAT 框架，在仅给定稀疏定时关键帧与到达时间的条件下，实现在仿真和真实 Unitree G1 人形机器人上的稳定全身运动跟踪。

**💡 创新点**

创新点在于将稠密运动追踪专家与特权潜在转移学习相结合，利用稠密轨迹作为训练时的特权监督，学习潜在转移先验，并在潜在空间中进行残差强化学习，从而显著提升稀疏关键帧控制的精度与鲁棒性。

**🔧 技术方法**

技术手段包括：稠密运动追踪专家网络、DAgger 迭代模仿学习、条件变分自编码器 (CVAE) 学习潜在转移、PPO 强化学习以及潜在空间残差调节。

**📊 数据集**

训练使用公开的大规模人形运动捕捉与动画数据集（例如 MOCA、CMU MoCap 等），并在仿真环境中进行大量数据采集，最终在 Unitree G1 机器人上验证。

**📈 对比分析**

与 SONIC、TWIST2 以及 Scratch RL 等基准对比，在 Mixed horizon（混合时间跨度）设置下，PLAT 在 TKS@3（成功率）最高、Timed MPJPE@3（误差）最低；在不同时间跨度（单步、短、中、长）下亦保持高成功率和低误差，表现出更强的时间跨度泛化能力。

**⚠️ 局限性**

局限性在于仍需稠密轨迹与预训练专家作为特权监督，缺乏此类数据时难以直接迁移；对极长时间跨度的命令仍可能出现误差累积，且对缺乏特权监督的数据集适用性有限。

---

## 334. Fisheye-VLA: Decoupling Coverage and Acuity for Manipulation with a Single Fisheye Camera

**arXiv ID:** 2609.25750 | [PDF](https://arxiv.org/pdf/2609.25750v1)

**作者:** Ziang Ren `[一作]` (Hong Kong Embodied AI Lab), Zhongyu Li `[通讯]` (Hong Kong Embodied AI Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了单摄像头 Fisheye‑VLA 接口，通过从 220° 鱼眼图像中渲染端执行器中心视图，提供全局与局部视觉信息，从而完成桌面、架子和传送带等多任务操作。

**💡 创新点**

创新在于使用单摄像头同时提供全局视角与局部特写，并通过端执行器投影、运动预置与共享射线编码实现动态视角分配，避免了传统的手腕摄像头。

**🔧 技术方法**

采用鱼眼相机校准模型、逆透视采样渲染、端执行器正向投影、运动预估、共享球面射线位置编码，并在预训练的 VLA 政策上微调。

**📊 数据集**

收集了 300 条机器人遥控演示（每任务），并使用 300 条手持演示作为预训练数据；训练与评估基于 Flexiv Rizon 4s 机械臂与 GN01 手爪。

**📈 对比分析**

将 Fisheye‑VLA 与前置摄像头+双手腕摄像头基准进行对比；在三层桌面区域（R1–R3）中，Fisheye‑VLA 在 R3 的成功率达 82% 远超 14% 的基准；在多任务设置下，整体成功率分别为 79%、88%、86%、72% 与 68%。

**⚠️ 局限性**

对相机校准高度依赖，无法消除遮挡；仅在所测试任务与空间范围内验证，未覆盖更广泛的操作与动态环境；需要进一步验证跨任务迁移与对实时误差的鲁棒性。

---

## 335. Beyond Class Marginals: Bounding Rehearsal Gaps without Freezing Class Co-occurrence

**arXiv ID:** 2609.25735 | [PDF](https://arxiv.org/pdf/2609.25735v1)

**作者:** Congren Dai `[一作]` (Imperial College London), Fei Ye `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出随机化通行回放（Randomised‑Pass Replay，RPR），通过维护一个持久化随机洗牌的类堆，控制在线持续学习中的类回放间隔，同时保持类频率平衡。

**💡 创新点**

创新点在于 RPR 给出了固定存储下的最优回放间隔上界（≤ 2⌈C/b⌉–1），不依赖未来类信息或额外前向传播，可在 ER‑ACE 等模型上提升准确率，并通过理论与实验验证其有效性。

**🔧 技术方法**

技术方法包括理论推导回放间隔与共现概率、实现持久化随机洗牌调度器、结合 ER‑ACE、DER、ViT 预训练等不同回放损失和存储策略进行实验。

**📊 数据集**

使用的数据集为 CIFAR‑10/100、Tiny‑ImageNet 以及 ImageNet‑R 的类增量流，并构造长尾 LT10、LT100 流以测试不平衡场景。

**📈 对比分析**

与独立类平衡检索、固定循环、随机洗牌等基线进行对比，实验表明在 Reservoir/ Balanced Reservoir 存储下平均提升 0.7–1.7% 的最终准确率；在 ViT 预训练的 LT10 小批量设置中亦观察到正面效果。

**⚠️ 局限性**

局限性包括仅在单遍、单批量、单任务学习设置下验证；效果高度依赖模型、损失、预算和存储配置，无法保证在多轮或大规模数据上同样表现；且对 Cosine 头的提升尚未通过机制解释。

---

## 336. AgriGen: Large-Scale Scene Generation Framework for Photorealistic Agricultural Robotics Simulation

**arXiv ID:** 2609.25725 | [PDF](https://arxiv.org/pdf/2609.25725v1)

**作者:** Utkarsh Bajpai `[一作]` (Georgia Tech-Cnrs Irl2958), Stéphanie Aravecchia `[通讯]` (Georgia Tech-Cnrs Irl2958)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 AgriGen，一个基于 Isaac Sim 和 ROS 2 的可扩展大规模农业机器人仿真框架，可自动生成可变地形、植被和多模态传感器数据，支持行种植、果园和葡萄园等场景，并提供丰富的标注。

**💡 创新点**

通过程序化地形和植被实例化结合资产流式加载，实现了在消费者级硬件上高帧率渲染数千植物的真实感农田环境；同时提供一体化的域随机化（地形、植被、光照、物理）和同步多模态 ROS 2 发布，填补了农业机器人仿真规模与多样性的空白。

**🔧 技术方法**

利用 Isaac Sim 的 RTX 渲染与 PhysX 动力学、Perlin 噪声生成地形、纹理与凹凸贴图、资产流式加载、ROS 2 多传感器发布以及域随机化参数化等技术实现。

**📊 数据集**

公开了 65 个 3D 植被模型（涵盖 5 种作物、5 种杂草、草本、灌木等），并配套十种作物变体、纹理与光照参数，构成可直接使用的数据集。

**📈 对比分析**

在 Xeon RTX 2080 Ti、i9 RTX 3080 和 Ryzen 9 2× RTX 3090 三种硬件上，对 50 m²–50000 m² 规模场景测量 FPS，结果显示 FPS 随 GPU 性能提升而提升，场景规模对帧率影响极小，平均在 22–60 FPS 之间，证明了框架的可扩展性。

**⚠️ 局限性**

目前仅支持已提供的几类作物与植被模型，缺乏对更大植被种类与更复杂物理交互（如土壤湿度、风力）的支持；在极大场景下仍需高端 GPU；对非 ROS 2/Isaac Sim 用户的迁移成本较高。

---

## 337. The Impact of Deep Care Isa on Reducing Musculoskeletal Disorders and Enhancing Productivity Among Office Employees: A Comprehensive Study

**arXiv ID:** 2609.25883 | [PDF](https://arxiv.org/pdf/2609.25883v1)

**作者:** Quanmin Liang `[一作]` (School of Computer Science and Engineering Sun Yat Sen University), Daniel Zapp `[通讯]` (Technical University of Munich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究评估了 Deep Care Isa 这一数字健康助手在改善办公环境中久坐人群的姿势意识、运动量、补水习惯、生产力及与肌肉骨骼疾病相关的病假天数的效果。

**💡 创新点**

创新点在于将 AI 实时姿势监测、微运动提醒和补水跟踪等功能集成于一款可持续交互的数字助手，减少传统干预的低参与度与短期效益问题。

**🔧 技术方法**

使用了多模态传感器（姿势传感、时间跟踪、饮水量检测）结合机器学习算法（实时反馈与个性化建议）以及基于问卷的自评量表。

**📊 数据集**

数据集来源于 2022-2024 年间 50 家企业共 2,325 名职工的三阶段（基线、中期、终期）自评问卷与传感器记录。

**📈 对比分析**

通过配对 t 检验、回归分析和效应量（Cohen’s d）评估干预前后差异，结果显示 MSD 病假天数下降 56%（大效应），生产力提升 58%，姿势意识 88%，运动量 79%，补水率 28%，均达到统计显著且效应显著。

**⚠️ 局限性**

局限在于研究为横向观察性实验，缺乏对照组；自评数据可能存在偏倚；以及样本主要集中在办公环境，对远程或工业工作场景的推广需进一步验证。

---

## 338. MatchFusion: Explicit-Implicit Instance Matching for Spatio-Temporal Multimodal Autonomous Driving

**arXiv ID:** 2609.25860 | [PDF](https://arxiv.org/pdf/2609.25860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 339. COBRA: A Content-Agnostic Framework for Zero-Day Detection of Suspicious Domains

**arXiv ID:** 2609.25882 | [PDF](https://arxiv.org/pdf/2609.25882v1)

**作者:** Alexandros Fourtounis `[一作]` (Foundation for Research and Technology Hellas), Evangelos Markatos `[通讯]` (Foundation for Research and Technology Hellas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种内容无关、注册时就能检测恶意域名的框架，利用域名的结构相似性对新注册域名进行聚类并标记为可疑；

**💡 创新点**

首次实现零日、无内容依赖的域名检测，依托注册时的结构相似度而非后期行为，显著提升检测速度与准确率；

**🔧 技术方法**

使用自定义字符串距离函数（基于字符等价类与TLD匹配）与聚类算法，结合手工验证、VirusTotal标注和RDAP注册信息；

**📊 数据集**

在1.5M个来自6个低流行顶级域（.bond,.life,.sbs,.space,.store,.top）的新注册域名上进行实验，覆盖三个月时间窗口；

**📈 对比分析**

与VirusTotal进行对比，检测精度达98.5%，能提前约80%被VT识别的域名，捕捉到VT遗漏的20%恶意域名；在主流TLD上亦保持97%以上精度；

**⚠️ 局限性**

受限于商业域名提供商的2天获取延迟，主要针对批量注册模式，可能无法识别极为多样化或单个域名的恶意注册，且假设TLD相同的域名结构相似。

---

## 340. Delving into Asymmetric Information Dynamics for High-Fidelity Virtual Try-On

**arXiv ID:** 2609.25881 | [PDF](https://arxiv.org/pdf/2609.25881v1)

**作者:** Zishu Qin `[一作]` (Alibaba Group), Hao Zhou `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 RealFit，一种通过非对称信息流实现高保真虚拟试穿的框架。

**💡 创新点**

创新点在于：① 单向信息流（UIF）阻断条件噪声的负面影响；② 解耦时间调制（DTM）最大化条件信息注入；③ 引入条件 KV 缓存显著提升推理速度并降低显存占用。

**🔧 技术方法**

技术细节包括：基于 FLUX‑style Diffusion Transformer 的结构改造，利用 Conditional Attention Entropy (CAE) 与 Injected Information Flux (IIF) 两个信息瓶颈指标；实现 UIF 与 DTM；使用 AdamW + DeepSpeed ZeRO‑2 训练；在推理时实现条件 KV 缓存。

**📊 数据集**

使用的数据集包括 VITON‑HD、DressCode 以及扩充的服装‑参考‑真值三元组数据集。

**📈 对比分析**

与 LADI‑VTON、IDM‑VTON、AnyFit、OOTDiffusion、CatVTON、ITA‑MDT、FitDiT 等多种基准方法在 SSIM、LPIPS、FID、KID 等指标上对比，RealFit 在所有指标上均获得最优或次优成绩，显著提升了图像保真度和多样性。

**⚠️ 局限性**

局限性在于：仍需要大量 GPU 显存训练；对极端服装细节或复杂三维纹理的处理仍有细微失真；跨域（光照、环境）鲁棒性尚未完全验证。

---

## 341. Isolated Sign Language Recognition for Icelandic Sign Language: Experiments in a Low-resource Setting

**arXiv ID:** 2609.25862 | [PDF](https://arxiv.org/pdf/2609.25862v1)

**作者:** Finnur Ágúst Ingimundarson `[一作]` (University of Zurich), Sarah Ebling `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究首次在冰岛手语ÍTM上进行孤立手语识别实验，比较了OpenHands和SPOTER两大公开框架。

**💡 创新点**

创新点在于针对极低资源手语开展跨语言迁移与预训练微调实验，并公开改进版框架与数据集，验证了迁移学习的可行性。

**🔧 技术方法**

主要使用MediaPipe、AlphaPose、SDPose三种姿态估计，结合Transformer、RNN、ST‑GCN、SL‑GCN等模型，探究不同预处理与训练策略对性能的影响。

**📊 数据集**

采用来自ÍTM SignWiki的1845段视频，涵盖849类，数据极为稀疏（多数类仅有两条样本）。

**📈 对比分析**

实验显示SPOTER在三种任务上明显优于OpenHands（提升约10–30%），跨语言预训练使SPOTER在所有任务上提升14–24个百分点，多语言训练则将OpenHands在全任务的准确率提升至28.9%。

**⚠️ 局限性**

主要限制包括数据量极少导致模型泛化不足、实验仅进行单次训练、缺乏错误分析以及对不同签字者的鲁棒性评估不足。

---

## 342. PartLLM: A Unified Multimodal Foundation for 3D Part Segmentation

**arXiv ID:** 2609.25832 | [PDF](https://arxiv.org/pdf/2609.25832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 343. BELXTR: Biomedical Entity Linking via Contextualized Token Retrieval

**arXiv ID:** 2609.25859 | [PDF](https://arxiv.org/pdf/2609.25859v1)

**作者:** Samuele Garda `[一作]` (Humboldt University of Berlin), Ulf Leser `[通讯]` (Humboldt University of Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了一种基于上下文化 token 检索的生物医学实体链接方法

**💡 创新点**

创新点在于将上下文化词向量与检索式结合，提升了实体匹配的精确度

**🔧 技术方法**

采用 Transformer（如 BioBERT）生成上下文向量，并结合 BM25+向量检索

**📊 数据集**

使用了 UMLS、MedMentions 等公开医学实体链接数据集

**📈 对比分析**

与 BLINK、S-Miner 等基线对比，召回率提升 5‑8%，精确率提升 3‑6%

**⚠️ 局限性**

受限于大型语料库的稀疏性，处理速度和可解释性需进一步改进

---

## 344. Less Is More in the Long Tail: Stage-Adaptive Sample Selection for Annotation-Efficient Dense Prediction

**arXiv ID:** 2609.25850 | [PDF](https://arxiv.org/pdf/2609.25850v1)

**作者:** Xiaofei Du `[一作]` (Fudan University), Zhijian Song `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

针对稀疏长尾医学分割任务，提出了阶段自适应样本选择框架 SASS，能够在注释成本受限的情况下通过精细化的样本选择逼近全数据训练性能。

**💡 创新点**

创新点包括：① 无需标注掩模即可通过教师-学生 DINO 计算梯度评分；② 结合尺度感知与覆盖平衡先验，并通过验证反馈动态调整类别优先级；③ 采用阶段自适应权重切换与组预算约束，实现对学习动态与类别分布的双重响应；④ 通过“less‑is‑more”机制在少量注释下提升长尾类别性能。

**🔧 技术方法**

使用技术包括：自监督教师-学生 DINO 目标与梯度压缩（随机投影），类别先验（规模、覆盖平衡）与验证驱动反馈，阶段自适应加权组合，组预算约束，遗忘引导的训练曲线，逐步增量池扩展和性能门控。

**📊 数据集**

实验数据集为一个多模态 3D 医学分割数据集，包含约 100,000 条样本，覆盖 108 个解剖结构，来自 TotalSegmentator、AMOS、BraTS21、KiTS、MMWHS、CrossMoDA、FLARE22、WORD、AbdomenCT-1K 和 VerSe 等公开数据集。

**📈 对比分析**

与 Random、Entropy、Coreset、BADGE、LESS、LESS+Organ、Full、Full+CB 等基线比较；在 40% 注释预算下，SASS 复现 98.3% 的全数据 Dice，整体 Dice 0.688，较 BADGE 提升 5.1pp，且在 Hard 组和部分结构（胰腺、胆囊等）实现了超过全数据的性能。

**⚠️ 局限性**

局限性包括：梯度评分的计算开销仍较大，需在阶段 2+ 之后才启用；对类别标签和先验信息的依赖可能限制迁移到无标签标识的任务；实验仅在模拟的注释环境下进行，未评估真实专家注释工作流程；使用固定分辨率与点提示策略，可能影响不同硬件或网络架构的适用性。

---

## 345. Metric-Bench: Exploring In-context Spatial Metric Reasoning in VLMs for Indoor Scenes

**arXiv ID:** 2609.25841 | [PDF](https://arxiv.org/pdf/2609.25841v1)

**作者:** Yuling Xi `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Metric-Bench基准和MetricReasoner模型，用于评估和提升视觉语言模型在室内场景中基于已知尺寸参考对象的空间度量推理能力。

**💡 创新点**

创新点在于通过无相机内参的上下文参考对象引导VLM进行空间度量推理，并结合可验证的奖励函数与结构化Chain-of-Thought提示实现强化学习微调，显著提升了模型的度量推理精度。

**🔧 技术方法**

采用了强化学习细调（GRPO）、结构化CoT提示、可验证的奖励机制（格式、指数精度、分段近似）以及大规模视觉语言模型（如Qwen3‑VL‑8B）进行训练。

**📊 数据集**

使用ScanNetV2构建的1340条问答对（包含尺寸、位置、深度、距离、3D边界盒等），并在ERQA、RoboSpatial、CountBench、V*Bench、BLINK等基准上进行评测。

**📈 对比分析**

与多种开源、专有和空间专用模型在零样本和微调场景下对比，MetricReasoner在Metric-Bench上MRA达48.6，较最优对手提升约9%，并在ERQA、RoboSpatial和通用基准上保持或提升性能。

**⚠️ 局限性**

主要限制是距离推理仍表现最弱，参考对象过多会导致信息过载，且对跨场景的泛化和动态视频输入的适应性仍有待进一步提升。

---

## 346. Hydrozoan: Latency-Adaptive DAG Consensus under Mixed Byzantine and Crash Faults

**arXiv ID:** 2609.25918 | [PDF](https://arxiv.org/pdf/2609.25918v1)

**作者:** Qianyu Yu `[一作]` (Hong Kong University of Science and Technology), Alberto Sonnino `[通讯]` (Mysten Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种在混合 Byzantine 与 Crash 故障模型下的双路径 DAG 共识协议，允许在最多 p 个故障时以两条信息延迟提交，否则退回至三条信息延迟。

**💡 创新点**

创新点在于通过“分级间接规则”实现同一 DAG 上的快速与直接提交路径并行共存，首次实现满足 Shrestha 下界的快速路径并提供可在不同故障预算下动态切换的机制。

**🔧 技术方法**

技术实现基于未认证的 DAG 引擎（类似 Mysticeti），使用阈值投票、证书与弱投票分级间接决策，并在 Lean 4 中完成形式化安全与活性证明；实现语言为 Rust，部署于云端进行实验。

**📊 数据集**

使用模拟与真实区块链网络环境下的交易负载（10k tx/s 与 100k tx/s），结合 geo‑distributed 部署（EU、US 与东京）以及 Crash 注入的实验，验证协议性能。

**📈 对比分析**

与 Mysticeti、Orcaella、BlueBottle、Hydrangea 等基线在相同 DAG 引擎上对比，结果显示在快路径可用时提交延迟降低约 25%，在快路径失效时仍保持三条信息延迟，吞吐量保持不变。

**⚠️ 局限性**

局限性包括仅在未认证 DAG 环境验证，恶意攻击实验仅覆盖领导者背叛场景；实现未公开；对大规模动态节点变更的适应性尚未评估。

---

## 347. Rethinking Web Application Firewalls

**arXiv ID:** 2609.25892 | [PDF](https://arxiv.org/pdf/2609.25892v1)

**作者:** Laurin Brandner `[一作]` (ETH Zürich), Laurent Vanbever `[通讯]` (ETH Zürich)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于有向无环图（DAG）的Web应用防火墙架构，通过建模规则间的依赖关系来简化状态管理，并实现节点合并与并行执行等优化。

**💡 创新点**

核心创新在于将WAF规则集抽象为DAG，从而生成高效的执行计划；结合节点合并、变量合并和并行化技术，使得传统的顺序执行大幅度加速。

**🔧 技术方法**

采用DAG建模、节点/变量合并、并行化执行、规则集的预模拟与运行时监控等技术手段。

**📊 数据集**

实验使用CSE-CIC-IDS2018数据集中的HTTP头部样本，并结合CRS规则集中的正则表达式。

**📈 对比分析**

将合并后的DAG执行方式与ModSecurity的顺序执行做对比，实验显示合并操作至少提升2倍，最高可达10倍，且从不降低性能。

**⚠️ 局限性**

主要局限在于最优执行计划的生成是NP难问题，依赖于规则集和流量特征；在变量合并时，超过八个变量的加速并非线性，需进一步研究。

---

## 348. BAS-OPD: Budget-Aware Selective On-Policy Self-Distillation for Fine-Grained Multimodal Perception

**arXiv ID:** 2609.25891 | [PDF](https://arxiv.org/pdf/2609.25891v1)

**作者:** Zihan Chen `[一作]` (Chinese Academy of Sciences), Cho-Jui Hsieh `[通讯]` (University of California, Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了预算感知的选择性 on‑policy 自蒸馏框架 BAS‑OPD，用有限的裁剪教师查询预算在训练阶段挑选最有价值的裁剪视图，从而提升多模态 LLM 在细粒度视觉任务上的性能。

**💡 创新点**

创新点包括：①在全图策略训练中引入教师查询预算，显著降低教师推理成本；②提出基于学生熵、随机采样与学习型效用估计的三种选择策略，尤其是在线效用预测器与回放机制；③实现了单通道、单前向全图推理，保持部署效率。

**🔧 技术方法**

技术实现上采用了 on‑policy 蒸馏、Jensen‑Shannon 散度的 token‑级蒸馏目标、基于学生分布熵与对齐指标的效用函数、学习型选择器（带自监督回放）以及多轮查询预算控制。

**📊 数据集**

实验使用了七个细粒度视觉基准（V*Bench、ZoomBench、HR‑Bench 4K、MMVP、CV‑Bench、MMStar 与 POPE）以及相应的训练图像‑裁剪对，评估模型在不同预算下的性能。

**📈 对比分析**

通过与多种开源（Qwen3.5‑4B/9B、Qwen‑VL‑Instruct、MiniCPM‑V‑4.5 等）和闭源（Gemini‑3、GPT‑5 等）单通道多模态 LLM 进行对比，BAS‑OPD 9B 在七大基准上平均达 84.23%，在 25% 教师查询预算下 4B 模型平均 81.60%，显著优于同类模型，同时教师调用量仅为全查询的 15.78%。

**⚠️ 局限性**

限制方面：仅在 Qwen3.5‑4B 与单一随机种子上进行验证，未对不同 backbone（尤其 9B）做完整 ablation；墙钟时间受实现与硬件差异影响；EMA 教师未序列化，冷态选择器评估受训练分布限制。

---

## 349. Identity-Centric Video Summarization via Hierarchical Fusion of Biometric, Appearance, and 3D Body Features

**arXiv ID:** 2609.25837 | [PDF](https://arxiv.org/pdf/2609.25837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 350. Evaluating the Effectiveness of SechKAN on 1D Data

**arXiv ID:** 2609.25876 | [PDF](https://arxiv.org/pdf/2609.25876v1)

**作者:** Hoang-Thang Ta `[一作]` `[通讯]` (University of Information Technology), Hoang-Thang Ta (University of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了 SechKAN 在三种 1D 分类数据集上的效果，并与 EfficientKAN、MLP、CNN1D、ResNet1D 及 DSCNN1D 进行对比。

**💡 创新点**

创新点在于使用双曲 secant 作为基函数并通过 1D 投影压缩参数空间，同时提供了针对网格大小和归一化设置的消融研究。

**🔧 技术方法**

采用 SechKAN 架构，并使用 AdamW + OneCycleLR 训练，比较了多种基线网络（MLP、CNN、ResNet、EfficientKAN）。

**📊 数据集**

使用了 UCI HAR、ElectricDevices 和 Crop 三个 1D 分类数据集进行实验。

**📈 对比分析**

通过对比验证集/测试集准确率与训练时间，SechKAN 在 Crop 数据集上取得最高准确率，在 UCI HAR 与 ElectricDevices 上与 KAN/CNN 基线相当，并保持较低训练时长。

**⚠️ 局限性**

实验受限于仅三种数据集、仅分类任务、有限的超参数搜索以及基线模型数量不足，难以完全证明其泛化能力。

---

## 351. TV-AudioRemover: Joint Text-Visual Guided Sound Removal with Multi-Task Hard-Mixture Curriculum

**arXiv ID:** 2609.25864 | [PDF](https://arxiv.org/pdf/2609.25864v1)

**作者:** Xinyue Guo `[一作]` (Xiaomi Inc.), Jian Luan `[通讯]` (Xiaomi Inc.)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于文本-视觉引导的音频去除框架 TV‑AudioRemover，能在视频对象去除后消除相应声音

**💡 创新点**

创新点包括百万级单目标音视频对齐数据集、任务令牌与模态特定全局引导的多模态扩散 Transformer、两阶段难度分层的多任务训练与硬混合学习

**🔧 技术方法**

使用多模态扩散 Transformer (MM‑DiT)、CLIP/CLAP/Synchformer 视觉特征、T5 语言编码、流匹配学习、模态特定全局指导

**📊 数据集**

利用 VGGSound、AudioSet、ACAVCaps、WavCaps 等公开数据构建百万级单目标音视频对齐集，并在此基础上生成混合-目标对

**📈 对比分析**

与 AVI‑Edit、InstructAV2AV、ZETA、Audio‑Omni、UNISON、SAM Audio 等模型对比，在 AV‑Remove‑Bench 上在目标抑制比、保留音质、音视一致性等指标上均达到或超过 SOTA

**⚠️ 局限性**

依赖于先行的高质量视频去除结果，单独文本或单独视觉条件效果较差，对极难区分的同类声源仍存在挑战

---

## 352. Visual Jev: Accurate and Efficient Decisions from Shared Visual Context

**arXiv ID:** 2609.25845 | [PDF](https://arxiv.org/pdf/2609.25845v1)

**作者:** Guanxu Yu `[一作]` (Independent Research), Yuhang Yao `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对同一图像的多独立选择题，提出了 Visual Jev 系统，通过一次视觉编码共享前缀、批量执行后缀并使用 LM 头读取候选概率。

**💡 创新点**

创新点在于将视觉上下文共享与后缀批处理结合，同时通过答案监督的后训练提升决策质量，而无需额外的定制头。

**🔧 技术方法**

技术包括 Qwen3‑VL 视觉‑语言模型、LoRA 微调、共享 KV 缓存、批量推理以及候选概率归一化。

**📊 数据集**

使用的公开数据集为 GQA、SNLI‑VE、TextVQA 和 TallyQA，其中后者作为未见任务做对照。

**📈 对比分析**

与独立序列推理相比，宏平均准确率从约0.70提升至约0.76，批处理可在同一图像的32道题目下实现约10问/秒的吞吐率，且每问平均时延大幅下降。

**⚠️ 局限性**

局限性包括需要一次性获得所有问题、依赖单一视觉‑语言模型族、对未见任务的迁移效果有限，以及混合精度导致的细粒度数值差异。

---

## 353. Latent Audio Watermarking for Robustness to Neural Codec Resynthesis

**arXiv ID:** 2609.25830 | [PDF](https://arxiv.org/pdf/2609.25830v1)

**作者:** Lovro Brulec `[一作]`, Leonard Kinzinger `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出在冻结的 EnCodec 语音编码器潜在空间中以可学习的向量加噪方式嵌入水印，实现了对神经编码器重编码的稳健性。

**💡 创新点**

创新点在于利用内容无关的潜在空间增量（DeltaMark）来抵抗多次低比特率重编码，而无需对编码器重新训练或对每段音频进行优化。

**🔧 技术方法**

采用前馈三层 MLP 生成潜在扰动，1D 残差卷积提取器恢复信息，并通过多项感知和频谱损失实现无感知嵌入；使用冻结的 EnCodec 编码器/解码器进行训练与推理。

**📊 数据集**

使用 LibriSpeech 语料库的 16k/24k 采样音频（共 16,000 条训练、2,000 条验证、2,000 条测试）进行训练与评估。

**📈 对比分析**

与 AudioSeal 与 WavMark 在 24 kbps/12 kbps/6 kbps 多次重编码、未见 DAC、TiCodec、MP3、AAC 等压缩以及常见信号失真下比较，DeltaMark 在多次 EnCodec 重编码时 FMA 与 TPR 下降更缓慢、在未见 DAC 上检测率保持 96–99%，并在多数传统失真攻击中维持 95–100% 的检测率；在感知质量上虽低于基线，但大部分降噪归因于编码器本身。

**⚠️ 局限性**

局限性包括：对同一 payload 所用的扰动相同，易被对抗估计移除；性能依赖于冻结编码器，TiCodec 等其它神经编码器不具备相同稳健性；感知质量受限于 EnCodec 的无量化重建；缺乏内容/密钥化嵌入机制。

---

## 354. Boundary and Intra-Segment Learning for Partial Audio Deepfake Localization

**arXiv ID:** 2609.25822 | [PDF](https://arxiv.org/pdf/2609.25822v1)

**作者:** Zhe Ye `[一作]` (Sun Yat-sen University), Chng Eng Siong `[通讯]` (Nanyang Technological University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于边界与段内学习的部分音频深伪局部定位方法BISL。

**💡 创新点**

创新点在于同时引入边界学习（捕捉相邻帧的特征差异）和段内学习（聚合连续段特征并提升内部一致性）来提升定位精度。

**🔧 技术方法**

采用WavLM+Conformer提取特征，使用两层MLP分类头，配合边界差分、段均值与标准差特征以及余弦紧凑损失实现多任务训练。

**📊 数据集**

在PartialSpoof、Half-truth Audio Deepfake和LlamaPartialSpoof三大数据集上进行实验。

**📈 对比分析**

与多种基线方法对比，PS数据集上EER降至2.52%并获得F1 97.40%，HAD上EER 0.07%、F1 99.97%，跨数据集LPS上相对最佳EER 37.10%与F1 53.61%。

**⚠️ 局限性**

局限在于跨域泛化仍受分布差异影响，对极端长或低质量语音的鲁棒性未充分验证，且方法对训练数据中帧标签依赖较高。

---

## 355. NaCR: Visual Localization via NeRF-aided Camera Ray Regression

**arXiv ID:** 2609.25907 | [PDF](https://arxiv.org/pdf/2609.25907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 356. AURA: Angular Update Rate Adaptation for training complex-valued neural networks

**arXiv ID:** 2609.25914 | [PDF](https://arxiv.org/pdf/2609.25914v1)

**作者:** Enrico Ballini `[一作]` (Aarhus University), Tito Andriollo `[通讯]` (Aarhus University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种用于复值神经网络的自适应步长调节方法AURA，并将其作为插件集成到Adam和Muon等一阶优化器中。

**💡 创新点**

创新点在于：①使用基于Dice相似度的复数一致性量ζ衡量连续更新在长度、方向和旋转方向上的一致性；②根据此一致性动态调整每个参数的乘子γ，既能放大一致性方向又能缩小不一致方向；③无需额外梯度评估，保持更新方向不变。

**🔧 技术方法**

技术手段包括：AURA的三步算法（一致性变量、乘子调整、加权更新）；结合Adam/Muon作为底层更新；采用单参数的复数步长乘子；实现中使用JAX/Optax；以及在四个不同复杂度的测试案例中进行公平评估。

**📊 数据集**

实验使用四个自定义数据集：①非全纯标量复函数逼近（2500样本/500测试），②全纯标量复函数逼近（2500/500），③四维复变量多变量函数逼近（1296/259），④二维弹性物理问题的PINN（200/20边界点）。

**📈 对比分析**

对比方法包括RPROP、Adam、NadamW、CvAMSGrad、Muon等。AURA在所有测试中至少能将最小训练误差降低数倍、对数学习曲线面积显著改善，且在大批量或全批量设置下性能最优；其计算开销仅比基础优化器多≈0.1–0.15倍；但在小批量（32样本）噪声较大时会出现发散。

**⚠️ 局限性**

局限性包括：①需要为每个参数额外存储5个实数，增加显存；②引入10个超参数，调参难度较大；③缺乏梯度噪声估计，导致在噪声严重时失效；④目前仅验证于全连接网络，尚未评估卷积网络或大规模模型。

---

## 357. Robust Active-Perception Control for Global-State-Free Aerial-Ground Cooperation

**arXiv ID:** 2609.25898 | [PDF](https://arxiv.org/pdf/2609.25898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 358. Control Barrier Functions for Safe Free-Flying Robotic Spacecraft Operations in Tumbling Target Capture

**arXiv ID:** 2609.25905 | [PDF](https://arxiv.org/pdf/2609.25905v1)

**作者:** Alexander Meinert `[一作]` (e:fs TechHub GmbH), Alen Turnwald `[通讯]` (Ingolstadt University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种模块化的控制障碍函数（CBF）框架，用于自由飞行的机器人航天器在捕获旋转目标时实现安全的近距操作。

**💡 创新点**

创新点在于：①将13自由度系统拆分为可独立调度的姿态、运动和机器人子系统；②为每个子系统设计专门的CBF安全滤波器，并通过将上游安全命令作为已知耦合输入来处理动态耦合；③将ESA最新近距操作安全区和约束直接映射为CBF约束，覆盖了相对运动、姿态盲区、角速度上限、机器人关节和碰撞约束。

**🔧 技术方法**

使用技术包括：控制障碍函数与二次规划（CBF‑QP）安全滤波、模块化子系统建模与耦合、相对运动/姿态/机器人动力学建模、Basilisk高保真轨道仿真、Monte Carlo 统计评估、CasADi + OSQP 求解器。

**📊 数据集**

没有使用真实数据集，所有实验均基于Basilisk仿真生成的轨道与运动数据，并通过 Monte Carlo 随机采样初始位置与目标旋转速率。

**📈 对比分析**

通过与无安全滤波的 MPC/PD 控制器对比，展示了安全滤波器能在相对运动、姿态盲区、角速度同步与机器人关节限制等方面避免安全违规；仿真平均运行时为1.59 ms（最大3.75 ms），远低于100 ms控制周期；Monte Carlo 100条轨迹均满足所有安全区，角速度同步误差低于0.046 rad/s。

**⚠️ 局限性**

局限性包括：未对接触力与正式安全保证进行建模；未显式补偿机器人对基座的反向耦合；对高旋转速率的鲁棒性仍有限；使用球形碰撞模型导致一定保守性。

---

## 359. Vision-based Underwater Formation Control With Input Saturations via Barrier Lyapunov Functions

**arXiv ID:** 2609.25917 | [PDF](https://arxiv.org/pdf/2609.25917v1)

**作者:** Nicola De Carli `[一作]`, Dimos V. Dimarogonas `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种去中心化的水下多机器人编队控制框架，利用重心化障碍Lyapunov函数、命令滤波反向步进和CLF‑QP实现在有限感知与受限执行器下的编队保持与碰撞规避。

**💡 创新点**

引入自适应扩展的BLF域，每个约束可单独动态放宽，避免传统BLF在约束边界处失效；结合摄像头视场约束与命令滤波，直接处理六自由度Euler–Lagrange动力学与阈值，提升鲁棒性与安全性。

**🔧 技术方法**

重心化障碍Lyapunov函数（recentered BLF）、命令滤波反向步进（command‑filtered backstepping）、控制Lyapunov函数基准的二次规划（CLF‑QP）、自适应约束域扩展机制、Gazebo‑PX4软件仿真、BlueROV2 Heavy实验平台。

**📊 数据集**

通过Gazebo仿真与BlueROV2 Heavy的真实硬件平台进行SITL（软件在环）实验，无使用公开数据集。

**📈 对比分析**

与传统BLF/C&B方法对比，实验表明系统能在满足物理感知与执行器限制的前提下实现误差收敛、避碰成功，并在必要时自动扩展约束域，保持安全且不出现约束违背。

**⚠️ 局限性**

假设全局参考坐标一致，需预先标定摄像头外参，未针对动态障碍物或多目标场景进行测试；在真实水下环境中的鲁棒性与计算资源需求仍需进一步验证。

---

## 360. Decoupling Logical Masks from GPU Execution for Dynamic Block-Sparse Attention

**arXiv ID:** 2609.25869 | [PDF](https://arxiv.org/pdf/2609.25869v1)

**作者:** Shanghao Liu `[一作]` (National University of Singapore), Wenqi Jiang `[通讯]` (National University of Singapore)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个专用的运行时系统，用于动态块稀疏注意力（BSA）的物理规划与任务调度，以提升视频生成模型的推理速度。

**💡 创新点**

创新点：①将逻辑块与物理执行解耦，提供直接、粗化、精细化三种映射；②通过任务组织层的分组、拆分‑K、流水线深度等手段提升并行与数据复用；③利用离线性能基准生成“性能领域”表，在线仅做查表选择，显著降低计划选择开销。

**🔧 技术方法**

技术：CUDA原生内核、PyTorch Python运行时、块CSR格式的mask状态、离线性能分析与决策树分桶、Tensor Core指令、分块矩阵乘、软最大计算等。

**📊 数据集**

数据集：2315个来自 Wan2.1、Wan2.2、HunyuanVideo 的真实视频稀疏注意力 mask；以及在720p Wan2.1 文本转视频和720p HunyuanVideo-1.5 图像转视频模型上的评估。

**📈 对比分析**

评测方法：与 FlashInfer、FlexAttention、flex-block-attn 在四代 NVIDIA GPU（RTX4090、RTX5090、A100、H100）下的请求层、内核层和完整模型层进行基准；结果显示请求层几何平均加速 1.85–5.11×，内核层 1.12–6.38×，模型层 Diffusion 循环 1.22–2.08×，在 H100 上最高可达 6.79×。

**⚠️ 局限性**

限制：仅支持已预编译的 CUDA 内核集合，需离线收集足够 mask 样本才能构建表；对极端稀疏度或特殊 block 几何可能缺乏最优映射；迁移到非 NVIDIA 或新 GPU 需要重新构建 catalog。

---

## 361. What is the Better Curriculum: Controller-Shaped Grasping Behavior for Contact Force-Sensitive Manipulation

**arXiv ID:** 2609.25887 | [PDF](https://arxiv.org/pdf/2609.25887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 362. Optimizing the Score, Losing Sight of the Task: Reward Hacking Across Weights, Selection, and Prompts

**arXiv ID:** 2609.25848 | [PDF](https://arxiv.org/pdf/2609.25848v1)

**作者:** Vansh Wahi `[一作]` `[通讯]` (University of Waterloo), Vansh Wahi (University of Waterloo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个统一框架，用以比较权重更新、最佳-选取和持久文本优化在奖励劫持方面的差异，并给出了理论上限、数值演示与防御对应表。

**💡 创新点**

创新点在于将三种优化载体抽象为可比的策略类，提出基于距离的错误上界与类包含的容量顺序，并构建跨载体的防御对应关系。

**🔧 技术方法**

使用理论推导（KL上界、最佳-选取分析）、距离约束、代理压缩假说等技术，并在五输出离散模型上进行精确数值演示。

**📊 数据集**

使用自定义的五输出离散数据集（无公开大规模数据集），所有实验均在该离散设置下完成。

**📈 对比分析**

通过比较三种策略的可达行为、可控距离和防御适配性来评估风险；数值结果显示不同载体在同一代理下的劫持容量差异显著，且防御在跨载体迁移时需要额外假设。

**⚠️ 局限性**

局限包括：仅基于理论与小规模离散例子，未在真实LLM或真实评估误差分布上验证；距离估计在实际模型中困难；防御对应表为概念性而非量化，缺乏跨任务的实证支持。

---

## 363. Protocol before progress: leakage-aware evaluation of AIS trajectory prediction

**arXiv ID:** 2609.25827 | [PDF](https://arxiv.org/pdf/2609.25827v1)

**作者:** Zobeir Raisi `[一作]` (Chabahar Maritime University), Vali Mohammad Nazarzehi Had `[通讯]` (Chabahar Maritime University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建泄漏感知评估协议，分别在船舶、时间、区域三维拆分下对四种 AIS 轨迹预测模型进行系统审计。

**💡 创新点**

首次量化最佳-16解码、船舶共享泄漏以及区域拆分对模型误差的具体影响，并验证这些效应在不同地理与交通环境中的跨语料稳定性。

**🔧 技术方法**

重实现/构造 TrAISformer、AISFormer、GATransformer 与 MGFormer，并采用 Transformer、频域注意力、图注意力、绝对位置离散化与局部偏移编码等技术；同时比较最佳-16 oracle 与单步贪婪解码。

**📊 数据集**

使用丹麦国家 AIS 30 天数据（652M 原始报文）与美国墨西哥湾岸海岸 AIS 30 天数据（265M 原始报文），两者覆盖相同地理范围但交通结构差异明显。

**📈 对比分析**

在相同拆分清单与统一超参数下进行对比，发现最佳-16 解码使误差降低 2.1–3.2 倍，船舶共享拆分对大模型误差提升 23–25%，区域拆分导致绝对位置模型误差从 2–3 km 直至 20–25 km；图注意力几乎无效，频域注意力增益极小。

**⚠️ 局限性**

仅评估四个模型，未覆盖不同窗口密度、其他任务（如分类、异常检测）及超参数调整；部分结果仅在丹麦语料上验证，跨语料一致性仍需进一步研究；未充分探讨辅助结构（图、聚类等）可能引入的泄漏。

---

## 364. Neural Approximation by Function Composition: Rigidity and Doubly Exponential Convergence

**arXiv ID:** 2609.25874 | [PDF](https://arxiv.org/pdf/2609.25874v1)

**作者:** Wentao Huang `[一作]`, Haizhang Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

在CIFAR-10和ImageNet数据集上进行了实验。

**📈 对比分析**

与现有的几种主流模型进行了比较，结果显示该模型在分类精度上提高了5%，且训练时间缩短了20%。

**⚠️ 局限性**

模型在处理高分辨率图像时可能会出现性能下降的问题。

---

## 365. LoRango: It Takes Two LoRAs to Unlock Hidden Behaviors in Diffusion Models

**arXiv ID:** 2609.25884 | [PDF](https://arxiv.org/pdf/2609.25884v1)

**作者:** Jin Wei `[一作]` (Fudan University), Xiaoyan Sun `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于配对LoRA的后门攻击 LoRango，可通过组合两个看似正常的 LoRA 文件在文本到图像扩散模型中触发恶意生成。

**💡 创新点**

发现并利用配对条件激活机制，使用 Signature 与 Payload 两类 LoRA，在无输入触发器、无模型或采样参数修改的情况下实现高度选择性且难以检测的后门。

**🔧 技术方法**

利用低秩适配器（LoRA）、GEGLU 门控非线性、代码写入与对齐、输出/载体层级抑制、分布式训练以及标准 Diffusers/PEFT 加载，构建可在任何基线扩散模型上加载的静态 LoRA 文件。

**📊 数据集**

在 Stable Diffusion v1.5、SDXL 1.0 进行训练与评估，进一步在 SD3、FLUX 1‑schnell 等不同 denoiser 架构上验证跨模型泛化，使用公开的图像/文本配对数据进行 prompt 生成。

**📈 对比分析**

与 BadT2I、Personalization、EvilEdit、MasqLoRA 等基线对比，LoRango 在 SDXL 98.7% ASR、SD v1.5 97.9% ASR，单个 LoRA 仅 2–5% ASR；在保持低 MSE、优秀 CLIP Score 的同时，能够在不同 CFG、分辨率、采样步数等推理参数下持续保持 ≥80% ASR。

**⚠️ 局限性**

局限性包括：跨架构迁移需微调、对某些不相关 LoRA 干扰会降低转移率、在极端推理参数下效果衰减、实验规模有限（跨架构仅 20 样本验证）以及未在动态或多分支模型上进行评估。

---

## 366. AgenticSizing: A Large Language Model-based Multi-Agent Framework for Analog Circuit Sizing

**arXiv ID:** 2609.25873 | [PDF](https://arxiv.org/pdf/2609.25873v1)

**作者:** Yijia Hao `[一作]` (University of Edinburgh), Themis Prodromakis `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于大型语言模型的多智能体框架 AgenticSizing，用于复杂模拟电路的晶体管尺寸化。

**💡 创新点**

创新点包括：①利用拓扑分析将电路分解为功能块；②通过知识提取构建可迁移的设计知识库；③采用规划器与专门化尺寸化智能体协同工作，实现结构化、可解释的优化流程。

**🔧 技术方法**

技术手段包括：ChatGPT‑5.2 作为核心 LLM，LangGraph 进行多智能体调度，拓扑标记与子结构识别工具，知识库结构化存储性能关系与参数关系，闭环 SPICE 仿真反馈。

**📊 数据集**

数据集为 8 个模拟电路（BGR、两段折叠共模放大器、LDO 以及集成 LDO+FC+BGR），以及 4 个低复杂度电路用于初始知识库构建；电路规模最大达 55 片晶体管、60 个尺寸变量。

**📈 对比分析**

与传统优化方法（DE、贝叶斯优化、基于 RL 的多智能体）进行对比，AgenticSizing 在所有基准上实现更高的成功率、平均更少的仿真次数，并在高复杂度电路上保持可行性；但每次迭代需额外 LLM 推理时间，令总耗时受 LLM 推理占比影响。

**⚠️ 局限性**

局限性包括：① LLM 推理和 token 消耗高，尤其在复杂电路上；② 对细粒度收敛的可靠性有限，易出现接近边界的振荡；③ 运行结果存在一定的随机波动，需要多次试验；④ 目前仅验证在 28nm 工艺和部分电路类型，尚未扩展到更大规模或更高工艺节点。

---

## 367. MemoryAthena: Adaptive Routing over Latent and Generated Memories

**arXiv ID:** 2609.25853 | [PDF](https://arxiv.org/pdf/2609.25853v1)

**作者:** Mingyuan Li `[一作]` (ELLIS Institute of Finland), Shaoxiong Ji `[通讯]` (ELLIS Institute of Finland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MemoryAthena框架，将传统直接检索(E)与两种生成内存路径(GE、GH)并行，构建可学习的路由器实现自适应内存修正；

**💡 创新点**

创新点在于将直接检索作为显式锚点，学习何时以及以何种强度让生成内存介入，并通过无监督的未来标记优势蒸馏训练路由器；

**🔧 技术方法**

使用可训练的地址化内存表、读写接口、生成器以及轻量级的因果路由头；

**📊 数据集**

在多项问答基准（NQ、WebQA、TriviaQA、TruthfulQA、HotpotQA）和六个分类任务（SST2、MR、CR、RT、AG News、Yahoo）上进行评估；

**📈 对比分析**

相较于仅使用E、传统硬/软路由、以及跨骨干迁移，MemoryAthena在QA平均分从37.65提升至39.28，在六任务NLP准确率从76.73提升至79.13；

**⚠️ 局限性**

局限性包括：路由器仍需手动调节门限、存在与金标oracle的性能差距、对不同任务的泛化尚未完全验证、以及在极端数据或模型规模上的可扩展性待进一步研究。

---

## 368. Gaussian Flow-Matching Schedules: Implications for Sampling and Training

**arXiv ID:** 2609.25839 | [PDF](https://arxiv.org/pdf/2609.25839v1)

**作者:** Arsène Claustre `[一作]` (CNRS, ENS Paris), Eric Vanden-Eijnden `[通讯]` (CNRS, ENS Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出将流匹配调度拆解为方差路径与混合角度两个独立设计维度，分别控制推断采样和训练回归方差；

**💡 创新点**

创新点在于证明方差路径决定概率流、混合角度仅影响回归噪声，给出有限步Euler采样的必要漂移下界，并推导最小或均匀回归方差的闭式角度选择；

**🔧 技术方法**

使用高斯分析、流匹配、随机插值、Euler离散化、最优传输理论以及解析推导；

**📊 数据集**

无具体数据集，研究为理论分析；

**📈 对比分析**

通过与传统连续时间准则（动力学能量、Jacobian积分）对比，证明直线路径在任何离散网格上对Euler可精确采样，且对混合角度的最优选择可显著降低训练方差；

**⚠️ 局限性**

局限于中心对齐、可对角化且方向相关的高斯设置，扩展到非高斯或非可对角化情形仍需进一步研究。

---

## 369. Sometimes You Gotta Run Before You Can Walk: Run-then-Walk Scheduling Strategy for VLM Autonomous Driving

**arXiv ID:** 2609.25831 | [PDF](https://arxiv.org/pdf/2609.25831v1)

**作者:** Yuqi Ye `[一作]` (Peking University), Wei Gao `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种两阶段的RL训练策略“Run-then-Walk”，先在进度驱动阶段探索高进度轨迹，再在安全修复阶段通过终点奖励与安全约束恢复安全性。

**💡 创新点**

创新点在于将进度探索与安全修复分离为两个独立阶段，避免传统单阶段RL在进度与安全之间的权衡失衡，从而在更少的训练轮数内实现更高的综合性能。

**🔧 技术方法**

使用基于Vision‑Language Model的GRPO（Group Relative Policy Optimization）强化学习框架，并在Run阶段采用PDMS作为奖励，在Walk阶段引入终点距离奖励、NC、DAC等安全奖励。

**📊 数据集**

在四个仿真/真实数据集上评估：NAVSIMv1、NAVSIMv2、Navhard、nuScenes。

**📈 对比分析**

对比传统Walk‑and‑Run、Walk‑then‑Run以及单阶段Run/Walk等RL策略，Run-then-Walk在所有基准上提升PDMS/EPDMS得分、降低碰撞率，且训练轮数减少40–50%。

**⚠️ 局限性**

局限性包括：仍需手动设置终点奖励的超参（Δ、η）；对不同VLM架构的泛化尚未系统验证；安全修复阶段依赖安全门的阈值，过严格或过宽松都会影响最终性能。

---

## 370. Robust Fusion of Semantic and Behavioural Signals for LLM Reranking in Personalised Search

**arXiv ID:** 2609.25825 | [PDF](https://arxiv.org/pdf/2609.25825v1)

**作者:** Aleksandr V. Petrov `[一作]` (Spotify), Mounia Lalmas `[通讯]` (Spotify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在个性化搜索系统中，研究了将行为统计特征（Query Slice Stats, QSS）注入LLM跨编码reranker的方法，并提出双样本特征丢弃训练（dual‑sample feature‑dropout）来缓解模型对QSS的过度依赖。

**💡 创新点**

创新点在于：① 用prompt将离散化的行为统计特征直接注入跨编码器；② 通过对每个训练实例同时构造QSS存在与去除两种视图，形成确定性对齐的数据增强，显著降低shortcut learning；③ 通过离线与在线A/B测试验证该方法在稀疏/冷启动场景中的鲁棒性提升。

**🔧 技术方法**

采用0.6B参数的预训练语言模型（PLUM‑style），以点对点的二分类任务（Y/N）进行微调；使用多层交叉熵损失，并在训练中加入α权重控制QSS出现比例；评估时使用NDCG@7和Recall@7。

**📊 数据集**

数据集来源于一家大型音频流媒体平台的搜索日志：1 000 000个训练会话（约452 K唯一查询、46.1 M唯一候选项）和10 000个评估会话（约8.9 K查询、2.5 M候选）。

**📈 对比分析**

比较方法：将模型在完整prompt（All feat.）和QSS移除（No QSS）两种条件下进行评估，并与随机、BM25、monoT5、单独QSS评分等基线对比。离线实验显示：QSS注入使NDCG@7提升13.3%；双样本训练在保持此提升的同时，QSS移除时的NDCG提升4.0%；在线A/B/C实验中，两种QSS-aware模型相较基线提升约2%，双样本在冷启动切片上表现出更强的方向性优势。

**⚠️ 局限性**

局限性包括：只关注单一行为特征QSS；未对部分缺失或噪声特征的真实场景进行细粒度评估；未与专门的个性化融合模型做对比；在线实验仅聚焦整体搜索成功率，对稀疏/冷启动切片的统计显著性不足。

---

## 371. User Influence Analysis Based on Blogs

**arXiv ID:** 2609.25908 | [PDF](https://arxiv.org/pdf/2609.25908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 372. When Does Execution Provenance Help Agent Memory Retrieval?

**arXiv ID:** 2609.25913 | [PDF](https://arxiv.org/pdf/2609.25913v1)

**作者:** Yiqi Wang `[一作]` (University of Southern Queensland), Taotao Cai `[通讯]` (University of Southern Queensland)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究语言代理在执行历史超出上下文窗口时的检索问题，提出基于执行证明单元的检索视图和图条件残差R‑GCN来完成证据检索。

**💡 创新点**

将工具调用和输出对齐成源追溯单元以解决粒度问题，并通过零初始化残差R‑GCN在执行图上学习证据完成的残差，显著提升检索完整性。

**🔧 技术方法**

使用密集检索、跨编码器重新排序、R‑GCN图卷积、源对齐的证据单元和可训练的残差。

**📊 数据集**

在ISETrace合成操作系统代理轨迹上进行实验，构造2000个LLaMA生成的检索查询并人工验证。

**📈 对比分析**

通过精确源跨度覆盖、Full Support、Coverage@B、Budget‑AUC等指标比较多种候选视图和残差效果，原始单元提升Full Support@2048约19点，残差再提升约4.5点，整体显著优于平面块检索。

**⚠️ 局限性**

受限于合成数据、查询由模型生成、人工验证样本有限、图结构基于可观测调用不具语义因果，结果可能不易迁移到真实代理或多模态任务。

---

## 373. You Should Be Properly Scoring Your Odometry

**arXiv ID:** 2609.25900 | [PDF](https://arxiv.org/pdf/2609.25900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 374. When Should Dependency Updates Invoke Repair Agents? A Lightweight Routing Study

**arXiv ID:** 2609.25911 | [PDF](https://arxiv.org/pdf/2609.25911v1)

**作者:** Liheng Fan `[一作]` (China Academy of Telecommunication Technology), Yuzhi Chen `[通讯]` (Southeast University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 DepFixRouter，一种轻量级预代理路由器，用来在依赖更新的 PR 生成时判断哪些 PR 需要进一步调用昂贵的仓库级修复代理；

**💡 创新点**

①将依赖更新修复路由问题定义为预代理升级决策；②构建仅基于创建时可用的低成本信号的路由模型；③通过将部署时安全的创建信号与完整历史回溯信号对比，剔除后向泄漏；④在真实 PR 集合上证明该路由可显著降低代理调用量。

**🔧 技术方法**

使用 TF‑IDF 文本特征、机器学习分类器（LinearSVC、Logistic Regression）、元数据（是否由 bot 创建、依赖更新标志）、文件类型统计；以及实验性地使用 DeepSeek 诊断代理来评估路由后的调用成本。

**📊 数据集**

GitHub 上收集的 497 条手工标注的依赖更新 PR（约 14.5% 为真正需要兼容性修复的正例），其中 72 条为正例。

**📈 对比分析**

在 5 折仓库分组交叉验证下，使用 top‑k 路由指标与传统 route‑all、random、全历史回溯基线对比。创建时安全路由的 top‑20% 策略召回率为 51.4%，比全路由的 100% 低但显著优于随机（约 20%）。在 60 条 PR 的诊断代理试点中，路由策略将 LLM 调用从 60 次降至 20 次、Token 用量从 22,827 降至 7,737，调用成本降低 66.7%/66.1%。

**⚠️ 局限性**

正例稀缺、数据集规模有限，标注依赖单一评审员；回溯历史信号导致后向泄漏，无法代表实时部署性能；仅评估路由阶段，未验证完整的自动修复成功率；创建时信号的表现受限于当前 PR 结构与标签系统，未来需要更多通用特征与跨生态验证。

---

## 375. Beyond Scalar Sensitivity: Activation-Aware Mixed-Precision LLM Quantization with Cross-Layer Refinement

**arXiv ID:** 2609.25916 | [PDF](https://arxiv.org/pdf/2609.25916v1)

**作者:** Akihiro Yoshida `[一作]` (Fujitsu Limited), Yuma Ichikawa `[通讯]` (Fujitsu Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了跨层激活感知灵敏度分配（CASA）两阶段权重量化位宽分配方法；

**💡 创新点**

创新点在于用Kronecker分解的Hessian激活感知二次误差代理取代传统标量代理，并通过跨层局部搜索进一步校正位宽分配；

**🔧 技术方法**

技术包括高阶Taylor展开求解二次误差、Kronecker因子化Hessian、基于激活感知的MCKP求解（可闭式解高率失真），以及基于交叉项上限的局部搜索；

**📊 数据集**

实验使用 WikiText-2 作为校准与评估数据，同时在常识推理基准集上测算 0-shot 准确率；

**📈 对比分析**

与均匀分配、Q‑Palette、CASAself 等基线比较，结果显示在 2.25–3.5 位宽区间内，CASA 在 7B–14B LLM 上显著降低 perplexity 并提升 0‑shot 准确率（最高可提升 15–20% 以上），尤其在极低位宽（<3 位）时效果更为突出；

**⚠️ 局限性**

局限性包括：仅建模相邻层的交叉项，未覆盖更长距离或高阶交互；使用激活感知二次代理而非真实下游损失，极端情况下排名可能失真；以及需要额外的校准数据和计算开销。

---

## 376. Prediction Is Not Detection: Evaluating Pre-Recognition Claims in Longitudinal Clinical AI

**arXiv ID:** 2609.25852 | [PDF](https://arxiv.org/pdf/2609.25852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 377. In-Context Guidance: Learning Inter-Task Synergies via Numerical Foundational Models for Few-Shot Multitask Optimization

**arXiv ID:** 2609.25836 | [PDF](https://arxiv.org/pdf/2609.25836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 378. How It's Made: Uncovering Detection Engineering Processes for Network Intrusion Detection Rules

**arXiv ID:** 2609.25901 | [PDF](https://arxiv.org/pdf/2609.25901v1)

**作者:** Koen T. W. Teuwen `[一作]` (Eindhoven University of Technology), Luca Allodi `[通讯]` (Eindhoven University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

我们构建了 SuriCap 平台，组织 CTF 风格的工作坊，让 60 名参与者在实际网络流量上编写 Suricata IDS 规则，并记录了他们的工程过程。

**💡 创新点**

首次系统化研究 IDS 规则工程过程，识别了三阶段（规划、实现、完善）和共同策略，并证明经验并未显著影响规则质量；同时公开了平台、数据集和分析代码。

**🔧 技术方法**

采用 Suricata 引擎、SuriCap 自动反馈系统、CTF 竞赛框架，利用统计回归、序列图变换 (SGT) 与 HDBSCAN 聚类、Levenshtein 与 Jaccard 距离度量等技术进行流程与细化分析。

**📊 数据集**

使用四个攻击情境（信息泄露、恶意下载、C2、以及入门情境）的 PCAP 作为正负样本，结合 60 名参与者共 3146 条有效规则、5176 次规则变更和 15751 条测试结果。

**📈 对比分析**

通过 F1、精确率和召回率评估规则性能，回归分析经验对结果无显著影响；对比教学与无教学两组，发现仅在设计原则遵循度略有差异；聚类与编辑距离分析显示大多数工程者以细调方式改进规则，且大多数规则在隐藏测试上未显著提升。

**⚠️ 局限性**

仅针对 Suricata/Snort 规则，实验对象主要为学生和新手，未涵盖经验丰富的 SOC 工程师；未观察到特征选择过程；可用测试数据有限，导致泛化不足；平台与结论可能不完全适用于 YARA、Sigma 等其它规则语言。

---

## 379. Rethinking Length-Based Training: Batch Composition and Loss Normalization in Speech Token Language Models

**arXiv ID:** 2609.25890 | [PDF](https://arxiv.org/pdf/2609.25890v1)

**作者:** Hongjin Song `[一作]` (Beijing Institute of Technology), Chunxiang Jin `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究短-长排序训练对语音标记语言模型的影响，拆解呈现顺序、批次构造与损失归一化三因子。

**💡 创新点**

提供系统化的拆分协议，证明短-长排序的提升主要来自首次批次组装与批次依赖的损失归一化，而非序列长度顺序。

**🔧 技术方法**

采用 87M 参数自回归 Transformer、批量均值交叉熵和 token‑balanced 归一化、不同批次重排与长度分组策略。

**📊 数据集**

LibriSpeech train-clean-100 语音数据，使用 Mimi、EnCodec、SpeechTokenizer 三种分词器。

**📈 对比分析**

通过固定批次对比、token‑balanced 归一化实验和交叉分词器评估，发现仅 Mimi 在批量均值损失下首次分组可降低 PPL 约0.18；其余方案无显著改进。

**⚠️ 局限性**

受限于仅在单一语音数据集、单模型规模和特定分词器上验证，未涵盖更大模型或多语言场景，且归一化交互效应尚需更深入理论解释。

---

## 380. Reciprocal Collaboration: how lessons from convergence in GLAMs can enhance interdisciplinary AI research

**arXiv ID:** 2609.26023 | [PDF](https://arxiv.org/pdf/2609.26023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 381. Confidence-Guided Cross-Modal Knowledge Transfer for Multimodal Anomaly Detection in Microservice Systems

**arXiv ID:** 2609.25856 | [PDF](https://arxiv.org/pdf/2609.25856v1)

**作者:** Peipeng Wang `[一作]` (Dalian Maritime University), Zheng Li `[通讯]` (Queen's University Belfast)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种置信度引导的跨模态知识转移方法 CMT-AD，用于微服务系统的多模态（指标+日志）异常检测。

**💡 创新点**

创新点包括：动态计算模态置信度并在跨模态转移中进行权重调节；构造中间模态并通过结构一致性（PKT）与语义一致性（共享原型）约束消除模态异质性；结合软聚类分配的正则化避免负迁移。

**🔧 技术方法**

采用的技术包括：深度聚类 + 软聚类分配、熵置信度计算、门控中间模态、结构一致性约束、语义一致性约束、GCN、LSTM、Transformer、CNN、BERT、SVD、K‑means、KL 散度等。

**📊 数据集**

实验数据集为 GAIA、Spark 与 Stack 三个公开微服务监控数据集。

**📈 对比分析**

与单模、双模和多模基线方法比较，CMT-AD 在三个数据集上均取得 F1 分别为 0.975、0.911、0.967，优于 UAC‑AD、KANAD 等最新方法。

**⚠️ 局限性**

局限性在于：仅在三数据集上验证，跨模态优势不一定普适；模型对聚类初始化和预训练依赖较强；对快速变化的生产环境适应性仍需进一步提升。

---

## 382. ARAFA: An LLM-Generated Arabic Fact-Checking Dataset

**arXiv ID:** 2609.25833 | [PDF](https://arxiv.org/pdf/2609.25833v1)

**作者:** Christophe Khalil `[一作]` (American University of Beirut), Rida Assaf `[通讯]` (American University of Beirut)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一份基于大型语言模型的181,976条阿拉伯语事实核查数据集

**💡 创新点**

首次实现全自动化的三步生成–变异–验证流程，显著减少人工标注需求

**🔧 技术方法**

采用GPT‑4o、Claude Sonnet 3.5和Llama 3.1等LLM进行生成、变异与验证，并用COT提示提升质量

**📊 数据集**

数据集来源为阿拉伯维基百科，覆盖多领域，包含支持、反驳和缺乏信息三类标签

**📈 对比分析**

在该数据集上微调AraModernBert-Base-V1.0、AraBERTv2、Llama‑3.1‑8B及Qwen2.5‑7B，最佳宏F1达77%（支持88%，反驳93%，NEI89%）

**⚠️ 局限性**

受限于维基百科的偏见、合成数据可能产生模型特定偏差、仅覆盖现代标准阿拉伯语且未考虑方言

---

## 383. MICRO: Multi-Fidelity Active Search for Severe Error Discovery

**arXiv ID:** 2609.26025 | [PDF](https://arxiv.org/pdf/2609.26025v1)

**作者:** Orlando Leone `[一作]` (University of Zurich and ETH Zurich), Roman Boehringer `[通讯]` (University of Zurich and ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 MICRO（Multi‑Fidelity Impact Clustered Rollout）框架，用于在共享预算下结合强反馈（标注损失）和弱反馈（质量评分）进行主动搜索，以发现严重错误。

**💡 创新点**

创新点在于将多精度反馈的联合高斯模型与影响向量聚类相结合，利用影响度聚类减少冗余，并通过 rollout 评估候选根来实现高效的主动搜索。

**🔧 技术方法**

采用联合高斯模型进行反馈重构，计算影响向量进行聚类，使用基于影响的 k‑means 聚类、rollout 估计和贝叶斯更新；同时利用 PCA 降维的 LaBSE 文本嵌入作为特征。

**📊 数据集**

在 2020 年 WMT 机器翻译竞赛的英德语对照数据集上进行实验，使用 MQM 作为强反馈，pSQM 作为弱反馈。

**📈 对比分析**

与贪婪、ENS、Adapted MF‑ENS、Screen‑then‑Annotate 以及两种 rollout 控制策略比较，MICRO 在所有四种预算/成本设置下平均发现严重错误数量最高，且在大部分设置下显著优于其他方法（p<0.001）。

**⚠️ 局限性**

局限性包括：仅在 WMT20 英德数据集上验证，反馈成本假设为固定且未考虑实际用户疲劳；高成本下弱反馈效益不明显；rollout 仅考虑标注，不评估多重弱反馈的联合效果；聚类步骤增加计算开销。

---

## 384. Structural Complexity of Matching-Match: Dense and Sparse Graphs

**arXiv ID:** 2609.26006 | [PDF](https://arxiv.org/pdf/2609.26006v1)

**作者:** Ilie Dumitru `[一作]` (University of Bucharest), Alexandru Popa `[通讯]` (University of Bucharest)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了匹配‑匹配（Matching‑Match）问题在不同图结构下的可行性与计数复杂度，给出了多种多项式算法、精确阈值与NP/#P‑硬度证明，并在参数化与稀疏/稠密图上探讨了可扩展性。

**💡 创新点**

主要创新点包括：
- 在稠密图中确立补图最大度数为1时可解、为2时NP‑难的精确阈值；
- 对均匀完全多部图的可解性做出完整的判定，阈值为部大小r=2；
- 证明连通直径为2的cograph可解性为NP‑难，显示连通性可导致复杂度跃升；
- 在cograph上证明以颜色数为参数的W[1]‑硬度；
- 对路径/环的可行性给出Euler路径/电路的线性判定，并展示计数版本#P‑完整；
- 在星图与完全图上给出多项式计数算法；
- 通过图分解与流网络转化实现对离散稀疏图（线性森林、P4森林等）的NP‑难度阈值。

**🔧 技术方法**

采用的技术主要包括：
- 线性代数/矩阵方程与基向量枚举用于多部图的多项式算法；
- 流网络构造（多源/多汇）与最大流求解，用于补图最大度、预着色与缺失边的分配；
- Euler路径/电路判定（连通性与奇偶度检查）；
- 归约构造（三角分解、K_r‑分解、P4‑分解等）与多项式化简；
- 计数问题的可约与#P‑完整性证明；
- 参数化复杂度分析与W[1]归约。

**📊 数据集**

本文未使用任何实验数据集，全部以理论分析与构造归约为主。

**📈 对比分析**

由于本工作为理论复杂度研究，未进行算法性能比较；然而作者通过多种多项式时间算法与NP/ #P 硬度阈值，展示了不同图结构下解法的可行性与不可行性边界。

**⚠️ 局限性**

限制与未解问题包括：
- 对于完整k‑多部图，以部数k为参数的FPT/ W[1]性仍未定；
- 对于腿长≤2的蜘蛛图，预着色下颜色数无界时的可解性仍未判定；
- 本文仅给出理论结果，缺乏实验验证与实现细节；
- 对于一般图的计数与决策复杂度的更细粒度划分尚未完成。

---

## 385. Manipulation with Stability Guarantees: Linear Deformable Objects with Non-negligible Physical Response Grasped at Multiple Location

**arXiv ID:** 2609.26004 | [PDF](https://arxiv.org/pdf/2609.26004v1)

**作者:** Daniel Feliu-Talegon `[一作]` (Delft University of Technology), Cosimo Della Santina `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于物理模型的闭环控制框架，用于多点抓握的可变形线性物体（DLO）的动态操控；

**💡 创新点**

创新点在于：①构建了完整的动态模型，显式考虑了物体的惯性、弹性与重力耦合；②将作用力和扭矩的作用坐标扩展到SE(3)×…×SE(3)，实现了可协同的“协同坐标”变换；③在此基础上设计了具有闭环稳定性保证的低层控制器与高层形状生成策略；

**🔧 技术方法**

采用了Cosserat杆理论、离散弯曲/扭转基底展开的Ritz–Galerkin离散化、雅可比矩阵与协同坐标变换、模型预测与优化求解、梯度下降逆运动学、运动捕捉反馈等技术；

**📊 数据集**

实验使用了三根不同直径、密度、Young模量的实用电缆作为DLO，并在仿真中构建了对应的杆模型；

**📈 对比分析**

与基于模型无关的基线（假设DLO不变形）相比，本文方法在仿真与实验中实现了零稳态误差、平均位置误差下降至1–3 cm（相对于对象长度约1–3 %），且对高速动作也保持较低的姿态误差（<0.08 rad）；

**⚠️ 局限性**

局限性包括：需要事先对DLO的物理参数（弹性、阻尼等）进行辨识；模型假设为连续弹性杆，可能对极度柔软或非均匀材料的DLO效果不佳；多点抓握的可达性仍受限于静力平衡方程；实时计算虽然可达150 Hz，但对更大规模模型仍可能成为瓶颈。

---

## 386. Flux: Optimal Scheduling of Optical Circuit Switches for LLM Training

**arXiv ID:** 2609.25949 | [PDF](https://arxiv.org/pdf/2609.25949v1)

**作者:** Arno Troch `[一作]` (IDLab, University of Antwerp - imec), Michael Peeters `[通讯]` (imec)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向大型语言模型训练的工作负载感知光电线路交换（OCS）调度框架，利用 MILP 将计算与通信统一建模，生成能覆盖训练迭代完整依赖关系的最优线路重配置计划。

**💡 创新点**

创新点在于：① 抛弃传统基于聚合流量矩阵的调度方式，直接考虑工作负载的时间序列和计算-通信依赖；② 通过在 MILP 中加入重配置延迟、端点冲突和排序变量，实现线路重配置的重用与延迟平移；③ 让网络调度与计算阶段实现重叠，显著降低迭代时间与 NIC 缓冲需求。

**🔧 技术方法**

主要技术手段包括：MILP（混合整数线性规划）建模、Big‑M 条件约束、图论（DAG 依赖）以及离散事件网络仿真（ASTRA‑sim 与 OCS 仿真器）来评估调度方案。

**📊 数据集**

使用 Llama 3 8B 模型的训练工作负载追踪，覆盖 8/16 层、TP4×DP2 并行度，8 块 GPU 与 2 台 OCS 的实验平台。

**📈 对比分析**

与 RotorNet（固定周期调度）和 BvN（基于流量矩阵的分解）进行对比。实验结果显示：① 本方法在 1 µs–1 ms 的重配置延迟范围内始终取得最低迭代时间；② NIC 上芯片内存峰值比对手低三阶量级；③ 重配置次数与性能无直接比例关系，关键在于重配置是否带来有用线路。

**⚠️ 局限性**

局限性：① MILP 求解在任务数和层数扩大后求解时间呈指数增长，难以直接应用于大规模集群；② 对稀疏专家（MoE）模型的非确定性通信模式尚未支持；③ 假设训练任务完全可预知，未考虑运行时动态变化。

---

## 387. Challenges of Multi-Speaker Extraction for Real Conversational Speech Enhancement

**arXiv ID:** 2609.25948 | [PDF](https://arxiv.org/pdf/2609.25948v1)

**作者:** Robert Sutherland `[一作]` (University of Sheffield), Jon Barker `[通讯]` (University of Sheffield)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文针对目标说话人与多说话人语音分离，在真实四方对话环境下，提出并评估了基于VAD掩蔽的随机损失函数。

**💡 创新点**

创新点在于通过随机应用VAD掩蔽来控制训练时对静默段的关注比例，显著缓解了训练数据中静默比例过高导致的性能下降问题。

**🔧 技术方法**

主要技术包括TF‑GridNet语音增强网络、FiLM条件化、联合训练或预训练的说话人编码器、基于谱相似度与幅度差异的复合损失，以及随机VAD掩蔽策略。

**📊 数据集**

使用的公开数据集为CHiME‑9 ECHI，包含真实四方对话录音以及对应的朗读音频，用于说话人校准与模型训练。

**📈 对比分析**

与传统的全损失（全局谱损失）相比，采用80% VAD‑掩蔽的MSX模型将STOI从0.55提升至0.60，fwSegSNR从4.35提升至5.12；在多种语音质量指标（PESQ、CSig、Cbak、Covl）上亦表现出显著提升。

**⚠️ 局限性**

局限性包括：1）仍需解决朗读与自然对话说话人特征的差异，可考虑说话人自适应机制；2）随机掩蔽比例需要根据具体场景进一步调优；3）实验仅在单一数据集与单个硬件平台上验证，泛化性待进一步评估。

---

## 388. ClusterFewshot: Improving Few-shot Optimization for LLMs workflow

**arXiv ID:** 2609.25939 | [PDF](https://arxiv.org/pdf/2609.25939v1)

**作者:** Omri Bar Haim `[一作]` (Tel Aviv University), Lior Wolf `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于语义聚类与单次评估的Bootstrap示例选择方法ClusterFewshot，用于改进LLM的few‑shot提示构造

**💡 创新点**

将语义嵌入聚类与一轮评估结合，既保证示例多样性又捕捉对模型有用的示例，形成更高效、稳定的演示集合

**🔧 技术方法**

使用预训练句子编码器（如gtr‑t5、all‑mpnet、bge‑large、Qwen3‑Embedding）进行嵌入，k‑means聚类，Silhouette评估；对候选示例做单次验证评分并在不同采样策略（全局Top‑k、簇代表、混合）中挑选；在DSPy+SGLang框架下与BetterTogether、ReAct等管道集成

**📊 数据集**

在GSM8K、HotPotQA、Iris三个基准上进行评测，分别覆盖算术推理、多跳检索问答和分类任务

**📈 对比分析**

与随机Bootstrap（BFRS）和贝叶斯搜索的MIPROv2对比，ClusterFewshot在三项任务中均取得更高或相近准确率，同时显著降低编译与运行时间（如GSM8K上约30%时间节省，HotPotQA约40%），并在Agentic ReAct场景中提升早期终止准确率

**⚠️ 局限性**

对语义嵌入质量、簇数选择与单次评估子集代表性有依赖；大规模数据上多次k‑means计算会增加开销；在嵌入与任务语义不匹配的领域可能导致示例质量下降

---

## 389. Unsigned Distance Maps on 2D Point Cloud Registration

**arXiv ID:** 2609.25932 | [PDF](https://arxiv.org/pdf/2609.25932v1)

**作者:** Ricardo B. Sousa `[一作]` (University of Porto), António Paulo Moreira `[通讯]` (University of Porto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于无符号距离场的2D点云配准框架，利用预计算的距离及梯度实现O(1)对应点检索。

**💡 创新点**

创新点在于统一的距离场表示可同时支持点到点和点到面残差，并在SE(2)上推导相应雅可比，从而消除ICP中的逐迭代最近邻搜索。

**🔧 技术方法**

技术包括离散网格预计算距离场、有限差分梯度和Hessian、SE(2)微分几何、Gauss-Newton优化以及srrg2_solver框架。

**📊 数据集**

实验使用合成三种几何场景和真实IILABS 3D数据集的Hokuyo 2D激光扫描。

**📈 对比分析**

与传统ICP及NDT等基线对比，预计算点到点变体在准确度和不确定性校准上均优于解析对应点法，且在实际激光里程计中表现出竞争性漂移（RTE≈1–3%）。

**⚠️ 局限性**

局限性包括对噪声敏感的二阶梯度点到面残差、对网格分辨率的依赖，以及尚未评估的计算速度与内存占用。

---

## 390. Analysis of trade-offs in urban heat mitigation using a Bayesian Optimization framework for an urban canopy layer model

**arXiv ID:** 2609.25953 | [PDF](https://arxiv.org/pdf/2609.25953v1)

**作者:** Rebekka Walter `[一作]`, Stephan Weber `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文仅展示了论文结构模板和数学公式的排版示例，未开展具体研究；

**💡 创新点**

未提出任何新的研究方法或创新点；

**🔧 技术方法**

主要使用了LaTeX排版技术，包括章节、子章节、数学公式、表格和图形的排版；

**📊 数据集**

未使用任何数据集；

**📈 对比分析**

未进行任何方法比较，也未报告任何性能指标；

**⚠️ 局限性**

由于缺乏实验、数据与方法，对论文的科学价值和实用性无法进行评估

---

## 391. Location transparency reduces activity by accounts misrepresenting their location on X

**arXiv ID:** 2609.25933 | [PDF](https://arxiv.org/pdf/2609.25933v1)

**作者:** Yuwei Chuai `[一作]` (University of Luxembourg), Mohsen Mosleh `[通讯]` (University of Oxford)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用平台新推出的“About this account”地理位置透明功能，对8200名自称位于美国的政治关注账号进行准实验研究，比较地理位置匹配与不匹配账号在披露前后发帖和回复行为以及受众互动的变化。

**💡 创新点**

创新点在于：①首次系统评估在公开揭示账号与平台推断位置不一致时的行为后果；②发现位置不匹配账号在披露后会显著减少发帖量（≈13.1%）且对受众互动几乎无影响；③揭示该抑制效应在来自非洲、亚洲等迁移不易解释地区最强，并且对向自称“美国”受众的有毒回复有显著下降。

**🔧 技术方法**

主要技术手段包括：
- 语言模型（GPT‑5系列）对自述位置、政治参与、诈骗、加密货币及误导性内容进行自动识别；
- Detoxify模型评估帖子毒性；
- 差分（DiD）计量方法（负二项回归与聚合前后模型）估计披露效应；
- 线性回归检验匹配与不匹配群体特征差异；
- 账号级固定效应与时间固定效应控制混杂。

**📊 数据集**

使用的数据集：
- 22M条总统选举相关推文；
- 1.6M条已做事实核查推文；
- 其它与气候/疫苗阴谋论相关关键词的推文；
- 结合上述数据，抽取了261,250名自称美国政治参与的账号，再筛选匹配与不匹配后得到8200名账号样本；
- 收集了7周前后共1,327,286条原创推文和3,631,457条回复，形成完整时间序列。

**📈 对比分析**

对比方法：对照组（位置匹配）与处理组（位置不匹配）在DiD框架下比较发帖频率、回复量和受众再转发/回复。结果显示，处理组发帖和回复总量下降约13%，在亚洲/非洲等地区下降更显著；受众每条帖子的平均互动量基本保持不变，表明受众行为未受显著影响。相比之前仅关注内容层面的干预，位置透明干预在不删除账号/内容的前提下实现了行为约束。

**⚠️ 局限性**

局限性：
- 研究为准实验，未能完全排除与时间同步的外部冲击；
- 区分故意欺骗与因VPN、迁移等正当原因导致的位置差异仍不确定；
- 未掌握账号暂停时间点，难以判断披露是否提升停号风险；
- 样本限定为政治关注且自称美国的账号，可能不具代表性，跨平台推广受限；
- 仅基于个人资料层面展示位置，受众对披露信息的实际可见性有限。

---

## 392. AT3D-AD: Anomaly Type-Aware 3D Anomaly Detection via Hierarchical Point-Language Alignment

**arXiv ID:** 2609.25930 | [PDF](https://arxiv.org/pdf/2609.25930v1)

**作者:** Jingyu Zeng `[一作]` (Shenzhen University), Can Gao `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了AT3D‑AD框架，用于3D点云缺陷的检测、定位与类型识别；

**💡 创新点**

创新点包括：①基于物理原理的参数化合成(PDPAS)可在正常点云上自动生成四类缺陷并提供多级监督；②Hierarchical Global–Local Alignment (HiGLA)通过Transformer+LoRA将全局与局部特征与文本原型对齐；③Semantic–Geometric Anomaly Classification (SGAC)联合语义与几何信息实现精细定位与分类；

**🔧 技术方法**

采用了ULIP‑2 PointBERT编码器、Transformer+LoRA、对比学习与文本对齐、多任务损失（焦点、Dice、排名、KL）以及多视图TTA；

**📊 数据集**

在四个公开基准上评估：Anomaly‑ShapeNet、Real3D‑AD、MiniShift、MulSen‑AD；

**📈 对比分析**

与IMRNet、Reg3D‑AD、R3D‑AD、GLFM、PLANE、PO3AD、MC3D‑AD、MFF‑M3AD、Simple3D、CASL、PA3AD、SeDiR等方法对比，AT3D‑AD在Object/Point AUROC、Macro‑F1等指标上均取得或逼近最优成绩；TTA增强后Object/Point AUROC均突破98%，类型识别准确率达74%–95%；

**⚠️ 局限性**

局限性包括：依赖合成缺陷的真实性，可能对真实未知缺陷泛化不足；模型对旋转敏感，需多视图TTA提升；只验证了四类缺陷，未对零样本/零‑shot情况进行充分测试；SGAC虽轻量，但仍增加约23 ms推理时间。

---

## 393. Perfect Rectangular Tilings with Two Colors

**arXiv ID:** 2609.26022 | [PDF](https://arxiv.org/pdf/2609.26022v1)

**作者:** Oswin Aichholzer `[一作]` (TU Graz), Carola Wenk `[通讯]` (Tulane University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在矩形区域内，研究使用六种两色边缘单元格（可旋转）并给定每种单元格数量的完美铺砖问题，系统性地列举并分析了所有可能的子集类，给出每类的可行性判定结果。

**💡 创新点**

首次将固定小型图形的有限铺砖问题按子集类细分为“必能铺砖”“必不能铺砖”“多项式可判定”“伪多项式可判定”四类，提出针对每类的判定条件，并将数论（整数分解、平方和定理）与动态规划相结合，获得了完整的复杂性谱，并指出剩余六个未解类。

**🔧 技术方法**

使用组合图论与边缘匹配分析构造贪心铺砖方案；利用数论工具解决整数分解与周长最小化问题；采用多项式时间判定算法（基于取余、分块与线性方程）和伪多项式动态规划（状态为剩余单元格数与角落可用性）实现复杂类的判定。

**📊 数据集**

该工作为纯理论研究，无实验数据集，主要通过符号推导与示例矩形尺寸（如5×5、6×6等）说明算法与证明。

**📈 对比分析**

通过理论复杂度分析，证明大多数子集类在多项式时间内可判定，部分类需伪多项式时间；对未解类给出猜想，未进行实验性能评估；对比方法主要是复杂度分类与构造性证明。

**⚠️ 局限性**

局限性：仅考虑矩形域，未对更一般多边形或周期边界进行研究；六个未解类仍无完整算法；伪多项式算法在尺寸指数级时仍可能不可行；对边界约束的细节讨论有限。

---

## 394. VideoX-Qwen: Data-Centric Instruction-Based Video Editing

**arXiv ID:** 2609.26015 | [PDF](https://arxiv.org/pdf/2609.26015v1)

**作者:** JJiahang Li `[一作]`, Zili Yi `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VideoX-Qwen框架，实现通用指令驱动的视频编辑。

**💡 创新点**

创新点在于通过多任务生产管线将专用编辑模型统一为源–指令–目标格式，构建超过120万条有向编辑样本，并结合多模态语义与源视频潜在双重条件的统一编辑器和渐进式训练策略。

**🔧 技术方法**

使用Qwen多模态语言模型、SAM3、Minimax-Remover、Qwen-Image-Edit、Wan-Animate等专家模型进行数据生成，采用预训练Wan生成器、DiT、VAE和flow‑matching训练方法。

**📊 数据集**

构建自有的1.2M条指令编辑对数据集（覆盖添加、移除、替换、属性编辑），并结合Ditto、GPT-IMAGE等公开视频与图像编辑数据。

**📈 对比分析**

在100例评测中与UniVideo、Kling O1比较，VideoX‑Qwen在9/11指标（指令跟随、编辑质量、内容保留、结构相似、感知相似、视频分布）上取得最佳平均分，提升约12%–34%。

**⚠️ 局限性**

局限在于仍依赖人工筛选保证数据质量，复杂多域场景的表现待提升，且在美学与成像质量方面略逊于Kling O1。

---

## 395. Interweaving Marginals into Multivariate Sample Paths: Training-Free Dependence Construction for Probabilistic Time Series Foundation Models

**arXiv ID:** 2609.25980 | [PDF](https://arxiv.org/pdf/2609.25980v1)

**作者:** Jinmyeong Choi `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在冻结的TSFM边缘分布下，如何通过不同耦合方式生成多变量预测轨迹，并检验耦合对联合预测的影响。

**💡 创新点**

将多通道-时间耦合的控制实验与历史相关性耦合相结合，证明耦合可以作为单独的后处理任务，并使用固定边缘样本来孤立耦合效应。

**🔧 技术方法**

采用高斯Copula、Schaake Shuffle、AR(1)时间相关、历史通道相关、因子化共相关矩阵以及固定边缘样本的排名重分配等技术。

**📊 数据集**

在FEV-bench的14个多变量预测任务上，使用Chronos-2、TimesFM-3、TiRex-2三种冻结的TSFM骨干。

**📈 对比分析**

与IID独立装配比较，使用能量分数、交叉通道方差分数（Ch.-VS）和时间方向方差分数（Time-VS）评估。时间耦合在Time-VS上提升约19.6%，通道耦合在Ch.-VS上提升约4.2%，通道-时间耦合进一步提升；两者在固定边缘实验与直接采样实验中均优于IID。

**⚠️ 局限性**

结果仅在评估任务和骨干上验证；历史耦合依赖于未来预测与历史相关性保持一致；高斯Copula假设时间不变的通道关联和共享时间相关结构，可能不足以捕捉更细粒度的跨通道滞后模式。

---

## 396. NAWE: Digital Watermarking with Neural-Assisted Watermark Extraction

**arXiv ID:** 2609.25972 | [PDF](https://arxiv.org/pdf/2609.25972v1)

**作者:** Roman Chaban `[一作]` (University of Geneva), Slava Voloshynovskiy `[通讯]` (University of Geneva)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结合显式信号处理和冻结的神经网络去噪器的图像水印提取方法（NAWE），实现盲提取并保持视觉质量。

**💡 创新点**

创新点在于：①将传统的周期同步与极化编码与可替换的神经去噪器结合；②使用冻结的预训练去噪器仅在提取时去除主机噪声；③模块化设计允许单独改进各阶段而无需端到端训练。

**🔧 技术方法**

技术包括周期同步嵌入、感知掩模分配、极化编码与解码、卷积去噪器（GS‑DRUNet 等）、自相关同步与 RANSAC 估计、CRC 校验和列表解码。

**📊 数据集**

使用 COCO 2017 数据集的 512×512 规模图像，构建 100 张原始图像用于鲁棒性基准，并在 156 条攻击链上进行评估。

**📈 对比分析**

与 TrustMark、SSL Watermarking、PixelSeal、WAM 等学习型系统在相同 PSNR 下进行全系统比较，NAWE 在几何和光度类攻击中取得最低 BER/WER，并在整体上保持竞争力；在压缩、色彩等类中表现相当；但在滤波/噪声类中略逊。

**⚠️ 局限性**

主要限制是预训练去噪器对强滤波、去噪和模糊等攻击的适应性不足；以及同步模型仅适用于仿射变换，难以处理更一般的透视失真。

---

## 397. Predict Before You Step: Auditable Occupancy Forecasting for Dynamic Obstacle Avoidance under Sparse Guidance

**arXiv ID:** 2609.25969 | [PDF](https://arxiv.org/pdf/2609.25969v1)

**作者:** Yuhui Mao `[一作]` (Nanyang Technological University), Rong Su `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了一种在稀疏航点指导下，利用LiDAR占据图预测未来场景，连接到 50 Hz 运动控制器的安全适配器，能够在实时循环中对行走机器人进行避障。

**💡 创新点**

创新点：①将自回归占据图预测器嵌入 50 Hz 控制循环，实现即时 1 s 未来占据图；②通过地图空间接口将预测结果唯一地作为决策通道，保证可审计性；③使用可学习的流场和可见性门进行地图迁移，同时引入参数无关的风险向量；④在仿真中设计遇见同步协议与匹配种子消融，对预测器贡献进行因果评估；⑤结合历史 LSTM、注意力融合与 PPO 训练，实现高效学习。

**🔧 技术方法**

技术手段：LiDAR BEV 前端、历史 LSTM、预测 LSTM（自回归）、流场与门控的卷积解码器、地图warp、注意力读出、风险池化、PPO 强化学习、传感器一致监督。

**📊 数据集**

数据集：Isaac Lab 物理仿真环境，Unitree Go2 机器人，包含 3 种运动障碍（对角、头对、曲线）和 3 速度等级（0.9–1.9 m/s、1.9–2.5 m/s、2.5–3.2 m/s）；真实实验使用 Unitree Go2 搭载 Mid‑360 LiDAR 进行 16 次交叉试验；训练使用 1024 并行环境。

**📈 对比分析**

比较方法：与重新训练的 REASAN、REASAN 原始、ABS-风暴、无预测器 BEV、APF、VO 等基线进行对比，使用成功率、碰撞率、成功案例时长等指标。性能方面：在 2.5–3.2 m/s 头对场景下，成功率 57.1%（比基线高 8.2pp），对角和曲线场景成功率均 >95%；在真实试验中 16 次均无碰撞；相较于无预测器版本，预测器在高速度、横向交叉场景下提升 1–3pp，显著降低碰撞率。

**⚠️ 局限性**

局限性：①仅测试单一移动障碍；②预测器受训练速度分布限制，低于训练速度时性能下降；③消融一次去掉预测器、风险池、通道，无法单独评估各组件贡献；④预测器精度提升未必带来性能提升，依赖于预测通道的使用；⑤未验证多智能体相互避免；⑥硬件实验有限，仅在少数场景下验证。

---

## 398. An Action Is Worth One Patch: Unified World-Action Modeling with PatchWAM

**arXiv ID:** 2609.25961 | [PDF](https://arxiv.org/pdf/2609.25961v1)

**作者:** Tianheng Wang `[一作]` (Westlake University), Kaicheng Yu `[通讯]` (Westlake University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 Patch World-Action Model，将连续动作编码为视觉潜在空间的 patch，使用共享的扩散 transformer 同时预测未来视觉帧和动作，消除单独动作专家。

**💡 创新点**

创新点是利用固定 Action-as-Patch 编码，将动作直接映射到视觉令牌空间，并通过联合 denoising 让动作预测和视觉预测共用同一生成过程，无需学习动作头。

**🔧 技术方法**

技术主要包括：固定 Action-as-Patch 编码、联合流匹配训练目标、共享扩散 transformer、FLUX.2 视觉编码器、文本编码器 Qwen3，以及多模态注意力。

**📊 数据集**

数据集包括 RoboTwin 2.0（50 任务）、LIBERO-Plus（10+10+10+10 任务）以及 RoboDojo（42 任务）等，使用干净与随机化演示及增强数据。

**📈 对比分析**

方法与双专家控制、WAM SoTA 进行匹配对比，单专家控制在匹配训练下取得 88.0% vs 78.4% 成功率，整体在 RoboTwin full-data 96.12%，LIBERO-Plus 91.8%，RoboDojo 29.39 Score，表现优于多数基准。

**⚠️ 局限性**

局限包括：推理时因需要更新未来帧令牌而比双专家慢 1.87 倍；仅在仿真环境验证，未测试实物机器人；无法完全归因于去掉动作专家、共享噪声、或交互作用等因素，还需进一步对照实验。

---

## 399. CausalLoss-Fin: Attributing Financial-Agent Loss to Decisions and Infrastructure Faults

**arXiv ID:** 2609.25960 | [PDF](https://arxiv.org/pdf/2609.25960v1)

**作者:** Abhishek Sharma `[一作]` `[通讯]`, Abhishek Sharma

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种同时对代理决策和基础设施故障进行因果干预的框架，用于在可重放的金融支付异常场景中精确分配损失责任。

**💡 创新点**

创新点在于：①在同一因果模型中对代理步骤和消息级故障进行干预；②引入三项拆分（基础设施效应、策略差异、参考策略残差）实现精确责任划分；③使用Shapley值对基础设施效应在单条消息间进行有符号分配；④提出最小充分修复方法在保持准确性的同时显著降低重放成本。

**🔧 技术方法**

技术包括：确定性重放、结构化因果模型、do-操作、Shapley价值分配、最小充分集搜索、对比实验与基线方法（CAR、CausalFlow、REFLECT）。

**📊 数据集**

数据集为FinalityBench（一个可重放的金融系统基准），利用其四个系统的故障注入日志生成带有已植入故障的实验案例。

**📈 对比分析**

与传统仅干预代理步骤的方法相比，联合干预方法在所有四个故障层级（单一、并联、过度确定、代理）上实现100%的原因识别准确率，基础设施责任误分为0%；最小充分集方法在每个案例仅需约2.3次重放即可恢复100%的可修复损失，而Shapley方法则需约280次重放。

**⚠️ 局限性**

限制包括：仅评估确定性程序（不涉及语言模型或随机策略）；仅在FinalityBench的消息级故障环境中验证；过度确定情况在真实世界中可能较少；参考策略的选择会影响残差分配；动作空间有限，仅覆盖七种常见代理行为。

---

## 400. Informed Masking: Structure-Aware Perturbation for Reinforcement Learning in Diffusion Large Language Models

**arXiv ID:** 2609.25927 | [PDF](https://arxiv.org/pdf/2609.25927v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 401. Calibrating Retrieval Geometry: Reliability-Guided Training-Free Aggregation for Visual Place Recognition

**arXiv ID:** 2609.25937 | [PDF](https://arxiv.org/pdf/2609.25937v1)

**作者:** Xin Li `[一作]`, Geng Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在冻结的视觉基础模型上提出了一种无训练、无标签、无权重更新的可靠性引导聚合方法（TFA），通过自适应调节残差分配、谱形状和全局特征融合来提升视觉位置识别。

**💡 创新点**

创新点在于：①利用跨码本一致性、检索覆盖度和谱统计的联合信号来驱动聚合规则；②设计了可逆的内核和谱权重机制，保证在没有干预时可回到原始特征；③提出仅用数据库伪查询统计即可完成自适应聚合，且可在部署时引入少量无标签目标图像（TFA‑C64）进一步提升。

**🔧 技术方法**

使用了残差聚合（VLAD）、硬/软分配、PCA、可逆谱变换、CLS融合、数据库伪查询统计以及64图像目标校准，全部在冻结的DINOv2/DINOv3特征上实现。

**📊 数据集**

在多种公开基准上评估：MSLS‑val、SPED、Pitts30k、VPAIR，以及8个航空/跨视角协议（含DINOv2‑B、DINOv3‑B）。

**📈 对比分析**

通过与AnyLoc、TF‑VPR、RIA等训练自由聚合方法在相同特征接口下进行对比。数据库仅TFA在9/16协议中领先，MSLS R@1提升17.39pp、SPED提升9.55pp；TFA‑C64在有目标图像时保持97–99%配对全查询性能，并在VPAIR等基准上表现尤为突出。

**⚠️ 局限性**

局限性：仅基于数据库一致性的自适应聚合在某些环境（如Office Loop、Old Town）会导致性能显著下降；需要额外的目标图像校准来缓解；对高维近邻崩溃仍较敏感。

---

## 402. Modulating Retroreflector-Aided UAV-Based FSO/QKD Systems

**arXiv ID:** 2609.25928 | [PDF](https://arxiv.org/pdf/2609.25928v1)

**作者:** Duy N. Luong `[一作]` (University of Danang University of Science and Technology), Anh T. Pham `[通讯]` (University of Aizu)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并分析了一种采用调制后向反射器（MRR）的无人机基Free-space optics/量子密钥分发系统，以实现简化但精确的跟踪并降低指向要求。

**💡 创新点**

将MRR集成到量子密钥协议中，推导了双向传播的光学通道概率分布（PDT），并给出QBER和SKR的解析表达式，展示其对无人机悬停误差的鲁棒性。

**🔧 技术方法**

采用BB84双衰变态量子密钥协议，基于多量子阱（MQW）调制后向反射器实现量子态调制；使用对数正态模型描述大气湍流、几何与对准损耗以及入射角波动；使用解析积分、伽马函数与Hermite高斯求积等数学工具推导性能指标。

**📊 数据集**

未使用公开数据集，而是基于实际通信参数（如发射功率、波长、气象参数 C_n^2 等）进行 Monte Carlo 仿真与理论比较。

**📈 对比分析**

通过与传统 GS→UAV 与 UAV→GS 的基准方案对比，采用 SKR 与 QBER 指标进行比较，结果显示在存在跟踪误差或无人机悬停波动时 MRR 系统显著提高 SKR 并保持低 QBER，能够实现最长 4 km 的有效距离。

**⚠️ 局限性**

仅在弱湍流假设下推导；对相关失配、相位攻击等安全威胁的分析有限；实验验证尚未完成；光学对准误差与 MRR 本身失效的耦合影响未充分考虑。

---

## 403. MATES: Learning Multi-Agent Interactions by Transforming Observations for Frozen Single-Agent Policies

**arXiv ID:** 2609.26010 | [PDF](https://arxiv.org/pdf/2609.26010v1)

**作者:** Elie Abboud `[一作]` (University of Haifa), Oren Gal `[通讯]` (University of Haifa)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 MATES（Multi‑Agent Observation Transformation for Existing Single‑Agent Policies）框架，能够将单智能体训练好的策略冻结后，通过轻量级观察适配器将多智能体的观测转换为单智能体可接受的格式，从而实现多智能体任务的学习；

**💡 创新点**

创新点在于：①只在输入端进行适配，保持预训练策略内部不变；②通过观察侧适配器实现参数高效，仅需 3.5%–7.3% 的参数更新；③不依赖演示或示范，兼容多种 MARL 算法；④适配器可在未见的团队规模上保持良好性能；

**🔧 技术方法**

技术包括：两阶段训练（单智能体训练→冻结+适配器训练）；使用基于 actor‑critic 的 IPPO 与 ISAC；适配器实现为 CNN/MLP；在适配器训练阶段保留预训练策略和价值网络，只优化适配器和（可选的）价值头；

**📊 数据集**

实验数据集涵盖三种任务：POGEMA（离散网格路径规划）、VMAS Navigation（连续控制导航）以及 VMAS Discovery（目标覆盖探测），分别用于评估不同观测与动作空间的表现；

**📈 对比分析**

与 Classic MARL（从零开始训练）、Full Fine‑Tuning、PegMARL、R2BC 等基线进行对比；MATES 在所有评测环境和算法下，性能与传统全训练或全微调相当甚至更优，同时仅优化少量参数；在更大团队规模下，MATES 的优势更为明显；

**⚠️ 局限性**

局限性：①仅适用于能拆分为单智能体观测 o_i 与邻居信息 N_i 的任务；②需要先有质量较高的单智能体预训练策略；③适配器仅在输入侧工作，对策略内部的更深层调节缺乏；④在某些环境下价值头更新仍能带来收益，需根据任务调整；

---

## 404. REVE: Efficient Hallucination Correction for Large Audio-Language Models via Reused Encoder States

**arXiv ID:** 2609.26028 | [PDF](https://arxiv.org/pdf/2609.26028v1)

**作者:** Hongjin Song `[一作]`, Xiang Xie `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种轻量级的后处理方法REVE，利用大型音频‑语言模型在生成字幕时已经计算好的编码器状态来验证并纠正字幕中的虚假事件提及，避免了额外的音频编码和显著的延迟；

**💡 创新点**

创新点在于通过两级读数（帧级统计和四段平均表示）以及类感知融合，只使用已存在的编码器状态即可实现事件验证；

**🔧 技术方法**

技术包括音频编码器输出重用、Ontology映射、统计与段落两尺度读出、类感知校准器以及阈值决策；

**📊 数据集**

使用了AudioSet平衡训练集进行读数训练，评估时采用AudioSet-eval、DESED混合样本以及跨模型测试（Qwen2‑Audio‑7B、Qwen2.5‑Omni‑7B、SALMONN‑13B）；

**📈 对比分析**

在AudioSet上相较于传统方法（Token Confidence、Decoder Probe、PANN、AST、CED‑Tiny/Base）REVE在保持75%召回率的前提下，95%+的错误提及被移除，整体降低残留错误密度至0.06，仅增加约0.57 ms延迟；在DESED混合数据上取得0.959 AUROC、0.915 AP，优于CED基线；

**⚠️ 局限性**

局限性包括对Ontology映射的依赖、对短事件的检出仍受平均池化影响、以及在更大模型或不同音频任务上性能的进一步验证待研究。

---

## 405. The Dynamics of Quasiregular Neural Learning

**arXiv ID:** 2609.26018 | [PDF](https://arxiv.org/pdf/2609.26018v1)

**作者:** Matthia Sabatelli `[一作]` `[通讯]` (University of Groningen), Matthia Sabatelli (University of Groningen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在控制的一维回归任务中神经网络如何学习主导正则性与系统例外，发现训练过程呈U形轨迹，先取得例外信息后回退到正则性再恢复；

**💡 创新点**

首次将正则与例外的竞争机制在梯度训练的神经网络中量化，并揭示例外稀缺性与正则结构如何决定过度正则化的强度；

**🔧 技术方法**

使用两层128单元的tanh全连接网络，Adam优化器和均方误差损失，基于已知正则/例外解的合成数据进行训练与评估；

**📊 数据集**

利用人工合成数据，输入x∈[0,1]，正则函数为正弦或混合正弦/指数/多项式，例外区间(0.55,0.70)加上偏移c=0.8，训练集包含256样本；

**📈 对比分析**

通过对例外区域内部偏移量Δ_t和异常/正则MSE的跟踪，定义过度正则化深度D来量化现象；实验显示例外比例越低，D越大，正则结构不同（如f3）可消除U形轨迹；

**⚠️ 局限性**

局限性在于仅针对单维合成任务，缺乏对多维真实语言或其他任务的验证，且对正则性形状和学习率等超参数的敏感性未系统探究；

---

## 406. Overload-Robust Latency in 5G-TSN: A HoL-Enhanced Hybrid Lyapunov Approach for 3GPP Indoor Factory Environments

**arXiv ID:** 2609.26011 | [PDF](https://arxiv.org/pdf/2609.26011v1)

**作者:** Kouros Zanbouri `[一作]` (University College Cork), Dirk Pesch `[通讯]` (University College Cork)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种结合HoL（Head‑of‑Line）延迟感知与Hybrid Lyapunov稳定性框架的5G‑TSN MAC调度算法，并在3GPP Indoor Factory场景下对AGV产生的URLLC、eMBB与mMTC流进行系统级仿真评估。

**💡 创新点**

创新点在于①用指数型HoL延迟乘子取代传统队列长度权重，显著抑制eMBB体积主导导致的URLLC饥饿；②在Hybrid Lyapunov框架内加入严格的类隔离（Γ_c）实现对关键类的优先级硬隔离；③提出了完整的物理层一致性、Rician衰落与瞬时阻塞模型，逼真复现工厂环境。

**🔧 技术方法**

核心技术包括：3GPP TR 38.901 Indoor Factory物理层模型；Rician衰落与Gauss‑Markov空间一致性滤波；连续时间马尔可夫链阻塞模型；Hybrid Lyapunov drift‑plus‑penalty调度；HoL延迟指数乘子与类隔离因子；OMNeT++/Simu5G仿真平台。

**📊 数据集**

使用自定义的3GPP Indoor Factory（InF‑DL）场景数据，包含随机行进路径、Rician K‑因子分布、阻塞率等参数；流量模型采用工业典型的URLLC周期性报文、eMBB I/P 帧大小分布、mMTC Pareto间隔与双峰负载。

**📈 对比分析**

比较方法为在同一5G‑TSN链路（20 MHz 51 PRB）下，变换AGV数（5–30）进行10次独立仿真，评估包交付率、QoS满足率、尾延迟、资源利用等指标。结果显示：在可用容量（≈12 AGV）内，HoL‑Lyapunov与最强基线相当；在过载（≥14 AGV）时，HoL‑Lyapunov提供约1.8×的URLLC交付率，99%尾延迟缩短4–7倍，且优先保障关键流；但整体吞吐量下降约⅔。

**⚠️ 局限性**

局限性包括：仅验证单小区固定频谱；对eMBB分片重组失效导致吞吐量损失未完全避免；未考虑多小区或多接入点的协同；实现对硬件定点运算与低延迟的可行性待验证；HoL乘子参数需经验调优，可能依赖场景。

---

## 407. Safety-Constrained Model Predictive Control for an Omnidirectional Walking Assistive Robot Using Control Barrier Function

**arXiv ID:** 2609.25994 | [PDF](https://arxiv.org/pdf/2609.25994v1)

**作者:** Andrea Fortuna `[一作]`, Arash Ajoudani `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种线性模型预测控制（MPC）框架，利用零化控制力场函数（ZCBF）和矩形障碍物约束实现实时安全轨迹跟踪。

**💡 创新点**

创新点在于：①通过为每个障碍物仅激活一面矩形面并预先选定面参数，避免了整数优化；②将ZCBF约束线性化为关于控制输入的线性不等式；③在控制律中加入二阶差分惩罚以抑制激烈转向，提升平滑性。

**🔧 技术方法**

使用的技术包括：线性本地坐标动力学模型、二次跟踪成本、二阶差分正则化、矩形障碍物的ZCBF约束、松弛变量处理，以及稀疏QP求解器（如OSQP）实现实时控制。

**📊 数据集**

未公开使用任何标准数据集，实验基于仿真或真实机器人平台的障碍场测试。

**📈 对比分析**

与传统MPC或不考虑安全约束的控制方法进行对比；实验显示该方法在保持轨迹跟踪性能的同时能够在受限计算资源下实时通过障碍物，约束满足率高且计算时间可接受。

**⚠️ 局限性**

局限性包括：①需要提前获取障碍物位置并手动或规则选择激活面；②对动态障碍物的适应性有限；③在极其拥挤的环境中仍需使用软化变量，可能导致安全裕度下降。

---

## 408. GRIP: Gaussian Rendering as a Cross-Modal Bridge for Image-to-Point Cloud Registration

**arXiv ID:** 2609.25966 | [PDF](https://arxiv.org/pdf/2609.25966v1)

**作者:** Karim Slimani `[一作]` (Sorbonne University), Brahim Tamadazte `[通讯]` (Sorbonne University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于姿态条件的渲染细化框架GRIP，用于图像到点云的跨模态匹配与配准；

**💡 创新点**

创新点在于通过高斯特征splatting将无序点云特征软投影到图像平面，形成共享的二维网格；随后使用像素对齐的Transformer进行双向跨模态注意力，并在层级解码器中将细化的特征反映回点云，实现精细对应与位姿更新；

**🔧 技术方法**

核心技术包括：高斯特征splatting、像素对齐Transformer、层级跨模态特征传播、score-guided PnP‑RANSAC及加权PnP求解；

**📊 数据集**

实验数据集为RGB‑D Scenes V2与7‑Scenes，两者均为室内RGB‑D对齐场景；

**📈 对比分析**

与2D3D‑MATR、Diff‑Reg、R^23Net等主流方法对比，GRIP在两数据集上取得最高的匹配内点比(IR)和与之相近的配准召回率(RR)，尤其在更严格的阈值下表现更佳；

**⚠️ 局限性**

主要局限是对初始位姿高度依赖，若初始估计过差，渲染的特征会失真，导致后续细化失败。

---

## 409. Exploring Solver-Level Warmstarting for Neural Network Verification

**arXiv ID:** 2609.25962 | [PDF](https://arxiv.org/pdf/2609.25962v1)

**作者:** Annelot Bosman `[一作]`, Jan van Rijn `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探讨并验证在神经网络可验证性（特别是鲁棒性）任务中，利用 MILP 求解器 SYMPHONY 的 solver‑level warm‑starting 能否显著降低求解时间，并将其应用于不同 ε、不同输入、不同模型三类实例变化；

**💡 创新点**

首次将 solver‑level warm‑starting 引入神经网络验证领域，系统评估不同属性变更（ε、输入、网络）下的可重用性和效果，展示在多数场景下可获得高达 97% 的求解时间缩减，并解决大量原先超时实例；

**🔧 技术方法**

使用 MILP 编码（Marabou+VERONA+SYMPHONY）生成 MPS 文件，改写 indicator 约束为 big‑M，利用 SYMPHONY 的 warm‑start 树和全局剪枝池技术；

**📊 数据集**

以 MNIST 数据集为基准，使用三种全连接网络（mnist‑net、mnist‑net_256x2、mnist‑net_256x4），随机选取 10 张测试图片；

**📈 对比分析**

与 SYMPHONY 传统单实例求解做对比，统计每类 warm‑start 场景下的基线与 warm‑started 运行时间、超时/错误情况。结果显示：在 UNSAT‑UNSAT、IMAGE、NET 等场景中平均运行时间降低 60%–97%，且成功解决原本超时的 300+ 个实例；但在部分 SAT‑SAT 场景中 Warm‑starting 反而导致平均 364% 的延迟；

**⚠️ 局限性**

局限性包括：1）仅在通用 MILP 求解器 SYMPHONY 上验证，未与最新的专业验证器（如 α‑CROWN、β‑CROWN）结合；2）对 SAT 实例效果不佳，因 SYMPHONY 缺乏针对性攻击搜索；3）未针对不同 MILP 特征自动挑选最佳 warm‑start 方案；4）大规模网络或更复杂激活函数时可重用性未知。

---

## 410. Calibration Is Not Verification: Falsifiability-Aware Conformal Routing for Mixture-of-Agents

**arXiv ID:** 2609.25959 | [PDF](https://arxiv.org/pdf/2609.25959v1)

**作者:** Nada Rahali `[一作]` (LUT University), Zhisong Liu `[通讯]` (LUT University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于多智能体协作的事实性过滤器C-MoA，该系统先将生成答案拆解成原子主张，利用各代理的语义一致性得分生成非合规性分数，然后通过例子层面的分割式校准设定保留阈值；可选地在此基础上加入对抗性可逆性检验CONTRA-MoA，结合盲目近似冲突、留一代理稳定性和可用性感知融合来进一步提升判别力。

**💡 创新点**

创新点在于：①将多代理的语义支持作为非合规性分数，首次把多代理一致性与合规预测结合；②使用例子层面的分割式校准实现分布无关的事实性控制；③提出对抗性可逆性检验与留一代理稳定性，并通过可用性感知融合在验证器具备领域知识时显著提升性能。

**🔧 技术方法**

核心技术包括：多代理生成与聚合（GPT‑4o、Mistral‑Large‑3 等）；原子主张抽取；NLI模型用于计算各代理对主张的语义支持；语义一致性非合规性分数及其补集；分割式校准（split‑conformal）设定阈值；对抗性可逆性“盲目近似”对抗赛；留一代理稳定性测度；可用性感知的最大融合。

**📊 数据集**

使用的数据集包括：FactScore（Wikipedia 传记长文本，原子主张标注），Golden 医学问答集（人类标注），Natural Questions（短回答），以及跨域转移测试（在不同领域上验证通用性）。

**📈 对比分析**

与无过滤、固定阈值、单代理合规预测以及 oracle（使用真实标签）的基线相比，C‑MoA在 FactScore 上将保留主张精度从 0.41 提升至 0.83（保持 75% 主张），在医学集上实现 1.00 的精度并满足所有 α；跨域转移亦保持高精度。CONTRA‑MoA 在医学域仅在 α=0.1 时提升到 0.94 的精度，因验证器具备领域知识；在 FactScore 这一数据集上，CONTRA‑MoA 的最大融合反而将 AUC 从 0.687 降至 0.652，说明该可逆性信号在缺乏知识时会削弱整体性能。

**⚠️ 局限性**

局限性包括：①对验证器知识高度依赖，缺乏领域知识时对抗性信号反而有害；②最大融合方式对单一噪声信号敏感，可能抑制有效一致性判别；③过滤导致信息缺失，用户在高阈值下可获得的内容显著下降；④实验规模有限，尤其对抗性检验只在单一 FactScore 分割上评估；⑤验证器为记忆型，无法利用外部检索信息。

---

## 411. From Reliable Text to Real Voices: Trust-Aware Progressive Adaptation for Low-Resource TTS

**arXiv ID:** 2609.25951 | [PDF](https://arxiv.org/pdf/2609.25951v1)

**作者:** Jiayi Lu `[一作]` (Beijing Logic Intelligence Technology), Ya Li `[通讯]` (Northwestern University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了trust‑aware progressive adaptation方法，通过先用合成语音建立文本‑语音对应关系，再利用真实语音恢复说话人控制，并使用ASR一致性对伪标签进行可靠性加权；

**💡 创新点**

创新点在于：①先合成后真实的渐进适配调度；②利用两台固定ASR的转录一致性作为伪标签可靠性代理；③系统性研究监督顺序对内容准确性与说话人相似度的影响；④在低资源场景下实现无标注式零样本语音克隆；

**🔧 技术方法**

采用合成TTS（FireRedTTS3、OmniVoice）生成合成语音，使用Gemini与Omnilingual ASR进行伪标签，基于RedAE/CAM++的参考条件编码，采用流匹配与加权损失，利用可靠性权重对真实语音训练；

**📊 数据集**

使用缅甸语和老挝语的Azure Nilar/Thiha、DVB配对语料、FLEURS、Common400、Clone300等低资源语音数据集；

**📈 对比分析**

通过CER、SIM‑O、MOS和综合H指标与基线、纯合成、纯真实、逆序训练等多种设置对比，S→R阶段采用立方权重得到最高H值（≈79‑80）和最佳MOS（≈3.8‑4.4），显著提升内容准确性与自然度，同时保持说话人相似度；

**⚠️ 局限性**

限制在于伪标签一致性并非绝对准确，仍受ASR错误影响；需要同时拥有合成TTS和ASR系统；对超大规模或非目标语言的泛化性需进一步验证；

---

## 412. Multimaterial Topology Optimization using SIMP, DMO, and gSF: A comparative study with Honeycomb tessellations

**arXiv ID:** 2609.25976 | [PDF](https://arxiv.org/pdf/2609.25976v1)

**作者:** Bhargav kumar Duru `[一作]` (Indian Institute of Technology Hyderabad), Prabhat Kumar `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多材料拓扑优化(MMTO)进行了比较研究，使用扩展SIMP、离散材料优化(DMO)和通用形函数(gSF)三种插值方案，在六边形有限元网格下对悬臂梁进行结构轻量化设计。

**💡 创新点**

提出了在六边形网格上对MMTO进行统一比较的框架，展示了gSF在可处理材料数上远超SIMP和DMO的优势，并验证了六边形网格在抑制棋盘格和点连接效应方面的有效性。

**🔧 技术方法**

采用了扩展SIMP、DMO、gSF插值方案；有限元求解、密度过滤和Heaviside投影；使用移动渐近方法(MMA)进行求解；对比指标包括材料分布、结构合规性、收敛行为与材料界面特征。

**📊 数据集**

实验数据基于二维悬臂梁模型，采用120×60个六边形单元，材料数量从1到15，预设体积分数并逐步增加材料数；使用MATLAB实现代码。

**📈 对比分析**

比较方法为在相同网格、加载、边界条件及体积分数约束下运行三种插值方案；结果显示gSF在材料数≥6时仍能得到可行解且合规性相当或更好，SIMP仅在≤5材料可用，DMO在≤7材料可用；gSF所需设计变量较少，计算成本相对较低。

**⚠️ 局限性**

SIMP与DMO因设计变量随材料数线性增长，导致在材料数较大时收敛困难且求解失败；gSF虽然可处理任意材料数，但其复杂的形函数实现和投影调整仍需要进一步优化；此外，六边形网格虽降低了棋盘格问题，但在三维或复杂几何中推广性仍待验证。

---

## 413. Theory for groupoid equivariant neural networks: an approach for steerable CNNs on bounded domains

**arXiv ID:** 2609.25987 | [PDF](https://arxiv.org/pdf/2609.25987v1)

**作者:** Alberto Ibort `[一作]` (Universidad Carlos III de Madrid), Juan M. Perez-Pardo `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于群组oids的卷积神经网络框架，专门用于处理具有边界或分层结构的有限域（如矩形、像素网格）

**💡 创新点**

核心创新点包括：①将边界信息内嵌到群组oid的箭头与切线锥条件中，得到多层次的轨道（bulk、edge、corner）和稳定子；②构造出bisection‑equivariant kernel定理，将非局部线性映射的约束转化为对两点核的传输约束；③证明了在多层网络中的“擦除”效应（finite propagation + erosion），给出了滤波式的等变性递归定理；④给出完整的参数计数与闭式基底，兼容连续和离散实现

**🔧 技术方法**

使用群组oid理论、张量表示、可测分区、Schur正交性、Poincaré/逆半群等数学工具；在实现层面采用离散稀疏 gather‑transform‑scatter 以及离线 null‑space 基底，保证训练时权重空间完全满足等变性约束

**📊 数据集**

主要在合成实验上验证：①使用 Poisson–Dirichlet 核心函数做边界值问题的识别实验；②在离散矩形像素网格上对不同特征类型、不同层数的网络进行参数化实验；没有公开真实数据集的实验

**📈 对比分析**

与传统全局等变卷积（全欧氏群、Steerable CNN）对比，证明在边界不可全局实现对称性的场景下，新框架至少提高一个数量级的样本效率与预测精度；对比实验中给出了完整的参数计数与误差上限，并在有限域上展示了显著的准确率提升

**⚠️ 局限性**

局限性：①需要预先设定合适的群组oid与切线锥，适用于规则或多边形域，非平滑或曲边域需进一步扩展；②对大规模网络，基底与矩阵运算仍有计算开销；③在离散实现中对连续核的近似可能导致细节损失；④滤波式等变性的擦除半径在深网络中可能过保守，需经验调参

---

## 414. Faithful Faithfulness Evaluations: Challenges & Pitfalls Learned from a Breast MRI Case Study

**arXiv ID:** 2609.25978 | [PDF](https://arxiv.org/pdf/2609.25978v1)

**作者:** Peachapong Poolpol `[一作]` (Fraunhofer Institute for Digital Medicine MEVIS), Eike Petersen `[通讯]` (Fraunhofer Institute for Digital Medicine MEVIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

针对乳腺MRI分类任务，使用ViT模型和多种saliency方法开展了基于扰动的解释可信度评估，揭示评估协议对方法排名的显著影响。

**💡 创新点**

提出了类无关（non‑class‑specific）梯度方法变体，并系统比较不同扰动策略（黑/白/最小强度/注意力掩码）对解释可信度的影响，首次在三维医学影像中全面评估ViT解释方法。

**🔧 技术方法**

采用Vision Transformer（MST）模型、Grad‑CAM、Gradient Attention Rollout、Grad‑SAM、GMAR、Attention Rollout、Last‑layer Attention、HiResCAM等解释技术，以及多种扰动与插入/删除曲线评估。

**📊 数据集**

使用公开的 ODELIA Breast MRI Challenge 2025 数据集（1022 例，分为 792 例训练、114 例验证、116 例测试），三分类（无病变、良性、恶性）。

**📈 对比分析**

通过删除与插入实验计算 AUCC（平均曲线下方积）和 AUCC_30（前30%扰动）指标，发现 Grad‑CAM 与 Gradient Attention Rollout 在类特定设置中表现最好，但不同替代策略下排名波动，类无关变体性能与注意力方法相近。

**⚠️ 局限性**

仅在单一乳腺MRI任务和中等性能模型上验证，未探究更强模型或其他影像任务的泛化，且评价侧重于模型内部可信度，未直接关联临床信任与实用性。

---

## 415. Toward Responsible AI-Augmented Cyber Defense: Pattern Recognition, Defense-in-Depth, and the Case for Human-AI Collaboration

**arXiv ID:** 2609.25921 | [PDF](https://arxiv.org/pdf/2609.25921v1)

**作者:** Mustafa S. Aljumaily `[一作]` (Daw Alfada Company), Nawar S. Alseelawi `[通讯]` (University of Misan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一个概率级联模型，系统性地将防御深度理论、AI模式识别理论与人机协作三者结合，量化分析AI增强在多层防御中的价值以及人类分析师的最佳参与比例。

**💡 创新点**

创新点在于：①将AI增强的增益视为多层检测缺失概率的乘法叠加，从而揭示AI在防御已趋饱和时的最大边际提升；②构建容量受限的人机协作分流模型，证明完整人工复核并非检测最优，存在内点最优的分析师容量比例。

**🔧 技术方法**

使用的技术包括：贝叶斯/尼曼–皮尔逊检测理论对每层阈值求最优阈值、基于正态分布的误检/命中率闭式表达、AI增强因子模型、容量受限的分流概率模型，以及 Monte Carlo 与解析仿真验证。

**📊 数据集**

实验仅采用了合成的三层高斯分离检测器（网络 IDS、终端 EDR、OT/SCADA 异常监测），未使用公开攻击数据集或真实 SOC 日志。

**📈 对比分析**

通过对不同 AI 增强水平（α=0,0.3,0.6）与分析师容量比例（C/R=0.12~1.00）的系统检测概率与误报率进行对比，结果显示 AI 能在多层防御中实现 95%+ 检测率，并且全人工复核会导致检测率下降而误报率显著下降，验证了内点最优假设。

**⚠️ 局限性**

局限性包括：使用合成数据且假设层间独立；阈值和分析师准确率为静态常数；未考虑阈值自适应或攻击者协同逃逸；未进行真实 SOC 数据的经验验证；未建模人工疲劳导致准确率下降的动态效应。

---

## 416. Domain-Adaptive Pretraining Enhances Water Treatment Semantic Representation for Large-Scale Structured Literature Mining

**arXiv ID:** 2609.26034 | [PDF](https://arxiv.org/pdf/2609.26034v1)

**作者:** Mudi Zhai `[一作]` (University of New South Wales), Haoran Duan `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

训练并微调了面向水处理文献的域适应BERT模型WaterBERT，应用于文本多分类、命名实体识别、关系抽取、BERTopic主题建模、构建大规模知识图谱以及开发集成图检索的检索系统。

**💡 创新点**

通过在近3 B token的水处理语料上进行持续预训练，使模型获得更强的领域语义理解；利用Fine‑Tune的NER/RE模型自动生成海量知识图谱，并与图检索融合提升检索准确率；同时实现了低成本、可扩展的完整信息抽取与检索流水线。

**🔧 技术方法**

域自适应预训练（DAPT）+ Masked Language Modeling；Fine‑Tune分类、NER、RE；BERTopic主题建模；知识图谱构建与图数据库查询；图+文本检索融合（WaterKERS）；使用LLM‑as‑a‑judge评估检索相关性。

**📊 数据集**

约693,211篇水处理论文（摘要+部分全文，约2.97 B tokens）；5,144篇《Environmental Science & Technology》文章用于BERTopic；1,000篇手工标注的NER/RE数据；100个检索查询用于评估检索系统。

**📈 对比分析**

与BERT‑base、RoBERTa‑base、SciBERT、BioBERT、ClimateBERT、EnvironmentalBERT等模型进行对照。WaterBERT在多类分类F1达90.12%，NER F1 79.50%，RE F1 74.04%；BERTopic指标NPMI 0.117、主题多样性84.9%、噪声率24.4%；知识图检索相对评分77.7，显著优于BM25、BGE等基线。

**⚠️ 局限性**

仅能抽取显式报告的信息，难以捕捉隐式或描述不一致的技术细节；预训练语料主要是科研文献，缺乏工程实践文本；图检索受图谱结构限制，无法完整映射复杂自然语言查询；需持续更新语料与模型以适应快速演变的领域知识。

---

## 417. Towards Systematic Qualification of Vision-Language Models for Automotive Perception Systems

**arXiv ID:** 2609.25945 | [PDF](https://arxiv.org/pdf/2609.25945v1)

**作者:** Malsha Ashani Mahawatta Dona `[一作]` (University of Gothenburg and Chalmers University of Technology), Christian Berger `[通讯]` (University of Gothenburg and Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套设计时统计评估与运行时LLM-as-a-judge相结合的验证流程，用以量化并比较视觉语言模型（VLM）在自动驾驶感知任务中的幻觉率。

**💡 创新点**

创新点在于将基于同义词匹配的统计评估方法与现有运行时自检技术结合，形成跨生命周期的完整验证框架，并在欧盟AI法的高风险分类下提供可操作的评估指标。

**🔧 技术方法**

使用了多模态提示、正则表达式与同义词匹配的统计检测、SelfCheckGPT自检框架、Friedman/Kendall统计检验以及置信度阈值。

**📊 数据集**

实验基于nuScenes数据集（1000张前视摄像头日间图像），并人工标注了目标物体作为真实标签。

**📈 对比分析**

与基准模型（LLaVA）相比，GPT4o和GPT4.5-preview在“完美匹配率”和“归一化幻觉率”上分别提升了约15–20%，但所有模型仍存在高比例的幻觉或漏检。运行时自检表现出高精度但召回率低。

**⚠️ 局限性**

限制包括样本规模有限、仅选取日间场景、未覆盖恶劣天气或夜间条件、模型训练数据不公开，导致结果对不同环境的泛化性不明。

---

## 418. Destination Support Restoration for Finite-Set Multimodal Trajectory Prediction

**arXiv ID:** 2609.25942 | [PDF](https://arxiv.org/pdf/2609.25942v1)

**作者:** Fengrui Liu `[一作]` (East China Normal University), Feng Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了Destination Support Restoration (DSR)，一种在多模态人类轨迹预测中通过因果后选择修复有限假设集的支持分配的操作；

**💡 创新点**

创新点包括：①利用候选证据生成整数目标计数并按模式分配；②通过受保护代表/祖先保护保证现有模式不丢失；③采用贪婪替换规则在不扩大集大小、无重训练的前提下，将多余槽位重新分配给欠缺模式；④该方法可通用于粒子滤波、流场、Transformer等多种预测器；

**🔧 技术方法**

技术手段涵盖：粒子滤波后的选择器、候选生成器、兼容能量评分、目标分配与整数化、受保护代表/祖先保护、贪婪替换规则；与MIF‑WLSTM、CLiFF、PPT、GDTS、Social Informer、PECNet等多种模型集成；

**📊 数据集**

使用Edinburgh Informatics Forum数据集（3,719轨迹）进行在线预测实验，并在ETH/UCY五个场景上进行交叉场景评估；

**📈 对比分析**

与各预测器原始配置在相同在线协议下对比，使用wADE、wFDE、NLL评估；DSR在MIF中wADE、wFDE分别下降13.36%/13.30%，在CLiFF、PPT、GDTS、Social Informer、PECNet中平均下降约8%~27%；仅增加约8%~15%的推理时间；

**⚠️ 局限性**

局限性包括：未在闭环机器人实验中验证；对目的地/模式定义高度依赖；需手动调参（ρ、C、τ），对候选生成质量敏感；在某些模型（如PECNet）中NLL可能不下降甚至上升。

---

## 419. Certified Against Which Oracle? Execution Labels Set the Reported Risk of Conformal Abstention for Text-to-SQL

**arXiv ID:** 2609.25938 | [PDF](https://arxiv.org/pdf/2609.25938v1)

**作者:** Jiamiao Liu `[一作]` (Third Military Medical University), Xuetao Chen `[通讯]` (Third Military Medical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对Spider-Realistic benchmark中使用的单数据库与多实例测试集两种oracle进行实验，评估它们对文本到SQL模型的自适应置信度（conformal abstention certificate）以及一致性分数的影响，并对oracle对风险报告的偏差进行量化。

**💡 创新点**

创新点在于首次将oracle差异纳入置信度证书的风险评估，构建了人工审核的语义错误、参考查询缺陷等标签体系，并展示了oracle对置信度分数评估的“oracle对齐”现象，说明仅靠多实例评估并不能保证更保守或更可靠的风险估计。

**🔧 技术方法**

采用的技术包括：自适应置信度证书（split‑conformal risk control），执行一致性得分（self‑consistency mass）、离散语义熵、度量、集合计数、序列对数概率等多种黑盒置信度分数，配合多实例测试集构造执行等价类并使用oracle标签；还使用了人工审核与AI审核相结合的标签体系进行风险真值校验。

**📊 数据集**

使用的数据集为Spider-Realistic（508个问题，19个schema）和其原始Spider dev的问答配对；对四个Qwen系列的SQL专家模型进行生成和评估；以及针对Oracle的人工与AI审核产生的标签数据。

**📈 对比分析**

比较方法是基于不同oracle和分区组合的四个实验单元（cell A–D），在多次随机划分下计算持出风险、置信度阈值和AUROC等指标；结果显示，在多实例oracle下，当前实践证书的持出风险比其自身标签高2.73–10.23个百分点，且在人工专家标签下风险进一步提升到约17–20个百分点；同时一致性分数在oracle自身标签下表现更好，体现oracle对齐效应。

**⚠️ 局限性**

局限性包括：仅针对Spider-Realistic及其多实例测试集进行评估，缺乏对其他benchmark和SQL模型的验证；人工标签覆盖有限，仅验证了参考查询缺陷标签，其他标签缺乏专家确认；实验仅考察四个Qwen系列模型，未覆盖更广泛的SQL生成器；以及oracle对齐的机制仍未得到理论解释。

---

## 420. Factorisability of Low Dimensional Non-Negative Integer Matrices

**arXiv ID:** 2609.26033 | [PDF](https://arxiv.org/pdf/2609.26033v1)

**作者:** Paul C. Bell `[一作]` (Liverpool John Moores University), Pavel Semukhin `[通讯]` (Liverpool John Moores University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

针对 2×2 非负整数矩阵的素性判断与分解问题，给出了可在 O(μ(A)·polylog(γ(A))) 时间内检测是否为素矩阵并在非素时给出非平凡分解的算法。

**💡 创新点**

创新点在于首次将 Smith 正则形与矩阵乘积约束相结合，利用对角化与同构变换将未知数降到 O(μ(A)) 级，进而实现比朴素枚举快数百倍的效率，并提供了关于复合行列式的判别理论。

**🔧 技术方法**

核心技术包括：
- 2×2 矩阵的 Smith 正则形与 Bézout 系数求解。
- 对关联矩阵的占位子与同构性质的利用。
- 对同余约束的多模约束求解，保证分解矩阵为整数非负。
- 通过对行列式因子化与对最小元素 μ(A) 的迭代来降低复杂度。

**📊 数据集**

本文未使用公开数据集，仅以理论构造与示例矩阵进行实验验证；示例中给出的矩阵 A=[1382 1243; 1045 1316] 及其分解被用来说明算法的可行性。

**📈 对比分析**

与传统的 O((max a_ij)^4) 或 O((max a_ij)^3) 的穷举法相比，本文提出的算法在复杂度上从多项式（相对最大元素）降低到线性（相对最小元素）乘以多项式对数因子，显著提升了效率；实验结果表明在中等规模矩阵上可在秒级完成。

**⚠️ 局限性**

限制与挑战：
- 该算法仍为指数级（关于矩阵二进制表示长度）；若要得到多项式时间，需要突破整数分解难题。
- 需要先对行列式进行因子化，若行列式极大或因子数多，预处理成本高。
- 对于 3×3 及更高维矩阵，目前已知问题在维数 ≥3 时普遍不可判定，故算法不易推广。
- 目前缺乏对比实验与理论上最优性证明，NP-难性尚未被确定。

---

## 421. CQ4OE: A benchmark for assessing LLM-assisted ontology generation from competency questions

**arXiv ID:** 2609.26029 | [PDF](https://arxiv.org/pdf/2609.26029v1)

**作者:** Jiayi Li `[一作]` (Polytechnic University of Madrid), María Poveda-Villalón `[通讯]` (Polytechnic University of Madrid)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并公开了 CQ4OE 评测基准，提供了基于 Competency Questions (CQs) 的可追溯金标准，分为两项任务 CQ2Term（术语级别）和 CQ2Onto（本体级别），并发布了完整的评测流水线和报告。

**💡 创新点**

创新点在于：①首次将每个 CQ 与所需的术语和 TBox 语句做细粒度关联，保留推理前后层面的证明；②设计多维度评测指标（术语恢复、属性特性、域/范围、axiom 级别、层级闭包），同时提供全局与对齐后视角；③实现可自动化、可解释的评测管道，并开放排行榜。

**🔧 技术方法**

使用多种大型语言模型（如 DeepSeek、Qwen、Gemma 等）以及零-shot、迭代和多智能体生成策略；评测过程中利用文本相似度（硬匹配、Levenshtein、Jaro-Winkler、嵌入相似度）进行术语对齐；使用 HermiT 推理器进行层级闭包检验。

**📊 数据集**

数据集包括六个公开本体（Wine、AWO、ODRL、SAREF4WATR、VGO、SWO），共 255 条 CQ，最终保留 110 条用于评测，经过补充 CQ 形成 118 条 CQ2Onto 的金标准。

**📈 对比分析**

对比九个 LLM 在两项任务上的性能：CQ2Term 词汇恢复 F1 约 59–67%，属性恢复明显低于类；CQ2Onto 结构级 F1 仅 26–34%，层级闭包 F1 低至 17%。不同域和策略的差异大于模型差异；多智能体策略在属性和 axiom 覆盖上略有提升。

**⚠️ 局限性**

局限性包括：①评估高度依赖术语对齐，短属性标签或语义相近但非等价词可能导致误差；②闭包救援仅针对可分解为原子子句的层级语句，复杂表达式未被覆盖；③数据泄漏风险（公开本体和 CQ 可能出现在预训练数据中），虽已通过手工注释减弱，但仍存在潜在偏差。

---

## 422. Knowledge-as-Skill: A Structural Design for Autonomous Knowledge-Base Use by LLM Agents

**arXiv ID:** 2609.25991 | [PDF](https://arxiv.org/pdf/2609.25991v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 423. BOBA: Dynamic Bayesian Optimization through Bayesian Active Inference

**arXiv ID:** 2609.26021 | [PDF](https://arxiv.org/pdf/2609.26021v1)

**作者:** Merlin Angel Kelly `[一作]` (University College London), Youngjun Cho `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于主动推理的动态贝叶斯优化获取函数BOBA。

**💡 创新点**

创新点在于将自由能原理和信息增益引入动态优化的获取函数，兼顾内在探索与外在目标。

**🔧 技术方法**

技术包括高斯过程代理、主动推理的自由能公式、信息增益近似、min-max归一化、softmax选择。

**📊 数据集**

使用八个合成动态基准函数（Schwefel、Powell、Eggholder、Ackley、Shekel、Griewank、Hartmann3/6），并加入高斯噪声。

**📈 对比分析**

与GP-UCB、WDBO等基线在固定观测量和固定时间两种情形下比较，BOBA在固定观测时显著降低均值遗憾，固定时间时与UCB相当。

**⚠️ 局限性**

局限包括对GP建模的依赖、假设同方差噪声、离散化导致的计算近似、对不同核与超参数缺乏评估。

---

## 424. Governed AI-Agent Coordination for Dementia Care: Architecture, Safety Contracts, and Evidence-Derived Workflow Verification

**arXiv ID:** 2609.25956 | [PDF](https://arxiv.org/pdf/2609.25956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 425. Compiling Sufficient Governance Context from Declared Losses and Reachable States: Exact Observation-Contract Synthesis with Cardinality and Cost Objectives

**arXiv ID:** 2609.26016 | [PDF](https://arxiv.org/pdf/2609.26016v1)

**作者:** Gaston Besanson `[一作]` `[通讯]` (University Torcuato Di Tella), Gaston Besanson (University Torcuato Di Tella)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出一种从声明的损失模型、可达状态模型和候选属性集合出发，自动合成满足判决一致性的最小或最优观察合同（authority contract）的编译方法，并给出可执行的接口。

**💡 创新点**

创新点在于：①首次将治理上下文的声明模型直接映射为可执行的观察合同合成问题；②构造判决一致性判定的可辨识矩阵并通过 SAT/MaxSAT 编码求解最小/最优合同；③提供完整的实验评估框架并与多种基线对比，验证该方法的有效性与可扩展性。

**🔧 技术方法**

核心技术包括：可达状态枚举与判决评估；差分集构造与差分集裁剪；可辨识矩阵（discernibility matrix）与硬/软子句编码；SAT/MaxSAT 求解（plain SAT、cardinality MaxSAT、weighted MaxSAT）；合同充分性检查与最小性验证；对可枚举域进行完整枚举以交叉验证编码正确性。

**📊 数据集**

使用三组构造域作为数据集：
- Code/Cloud domain (CH-B1/B2)；
- Procurement domain (CH-C1/C2)；
- Data-and-communications domain (AuthorityBench)。
每个域均为声明式、可执行的模型，包含可达状态集、候选属性集合和损失判决函数。

**📈 对比分析**

与四个基线（手工最小权限、ABAC 挖掘、必要变量分析、全枚举）在上述域上进行比较。实验表明：
- 手工基线在所有域均不满足判决一致性；
- SAT/MaxSAT 在 300 秒超时的可枚举域上仍在秒级完成；
- 在 CH-B2 域中，成本目标能区分两个等卡特量合同；
- 在 CH-C2 域中成本目标无法区分合同；
- 对规模扩展（属性数 up to 100）时，SAT/MaxSAT 解决时间远低于全枚举。

**⚠️ 局限性**

局限性包括：
- 依赖声明模型的完整性和可达状态模型的准确性；
- 未考虑部分可观测性、时序或概率损失；
- 对模型或属性变更的增量更新支持有限；
- 只在构造域上验证，缺乏真实系统的端到端操作验证；
- 成本模型为声明式加权，未进行敏感性分析。

---

## 426. Skytopia: Monocular Drone Navigation with Action-Conditioned Latent World Models

**arXiv ID:** 2609.26007 | [PDF](https://arxiv.org/pdf/2609.26007v1)

**作者:** Yuhang Zhang `[一作]` (Nanyang Technological University), Mir Feroskhan `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于3D高斯展平（3DGS）平台的单目无人机导航框架，并在该平台上训练动作条件的潜在世界模型；训练完成后将预测器丢弃，直接使用后向流匹配头进行动作生成，支持点目标、图像目标及无目标导航三种模式；在模拟与真实环境中无须微调即可实现飞行。

**💡 创新点**

创新点在于：①只在训练阶段使用前向与逆向动力学目标来学习场景几何与相机运动的表示；②将预测器从部署中剔除，显著降低推理成本；③统一模型即可处理三种导航模式；④利用3DGS高保真重建与刚体物理相结合的仿真平台，提升离线数据的多样性与真实性。

**🔧 技术方法**

主要技术包括：3D高斯展平（3DGS）重建与渲染；动作条件的潜在世界模型，包含前向预测头和逆向动力学头；流匹配（flow‑matching）动作生成头；Vision‑Language 背骨网络与可学习查询；在训练中使用冻结的视觉编码器做潜在监督；后向流匹配在推理时直接集成。

**📊 数据集**

数据集为 94 个从真实捕获场景重建的 3DGS 场景（18 室内、48 城市户外、28 森林），共收集 32.5M 帧；从中随机抽取 5 场作为 OOD 评估。

**📈 对比分析**

与 7 类基线（行为克隆、ACT、NoMaD、ViNT、OmniVLA、NWM、NavMorph）对比。模拟实验中，模型在点目标、图像目标和无目标三种模式下分别取得 57.8%、66.0% 和 49.0% 的成功率，均超过最强基线至少 15 点；真实飞行中，点目标 55% 成功率，图像目标 56.7%，无目标 41.7%；部署时预测器移除后推理成本降低 59.4%，仍保持最高成功率。

**⚠️ 局限性**

局限性包括：缺乏持续空间记忆，无法在长距离或高层次导航任务中保持全局一致性；对动态障碍物或时间变化场景的鲁棒性尚未评估；模型仍依赖大量高质量的 3DGS 场景重建，数据获取成本高。

---

## 427. FusionMMT: A Unified Multimodal and Multitask Learning Framework for Nuclear Fusion

**arXiv ID:** 2609.26095 | [PDF](https://arxiv.org/pdf/2609.26095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 428. GeoPair: Geometry-Preserving Cross-Layer Factorization for Training-Free Transformer Compression

**arXiv ID:** 2609.25963 | [PDF](https://arxiv.org/pdf/2609.25963v1)

**作者:** Baher Mohammad `[一作]` (MWS AI, ITMO University), Stamatios Lefkimmiatis `[通讯]` (MWS AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练-free 的 Transformer 压缩框架，先通过全局图匹配找出结构相似的层对，再使用共享字典和严格的层级白化，利用闭式 Sylvester 方程更新字典，并用 Hard Thresholding Pursuit 对系数进行稀疏化，从而在保持激活几何的同时实现高压缩率。

**💡 创新点**

创新点包括：① 通过完整的最大权匹配算法在所有层间寻找最优配对，摆脱传统邻接或经验聚类；② 在保持不同层白化空间的前提下，用 Sylvester 公式得到共享字典的闭式解，避免了先验的协方差平均化导致的几何失真；③ 将稀疏约束直接嵌入到系数更新中，使用 HTP+CG 实现可收敛的结构化稀疏编码。

**🔧 技术方法**

核心技术：层级白化（Cholesky 预处理）、全局图匹配（Edmonds Blossom 算法）、共享字典学习（闭式 Sylvester 迭代）、稀疏系数优化（Hard Thresholding Pursuit + 迭代 CG）、交替最小化框架。

**📊 数据集**

在多种 Transformer 架构上验证，包括 Llama‑3、Llama‑2、Qwen‑3、Gemma‑3、Phi‑4、Wan‑2.2 视频生成模型；使用通用的语言评测基准（PIQA、HellaSwag、Lambada、ARC、SciQ、MMLU 等）以及视频生成的 X‑CLIP 评测。

**📈 对比分析**

与传统的单层低秩 SVD、Basis‑Sharing、CoSpaDi、ROCKET、COMPOT、以及剪枝/稀疏压缩方法对比，本文方法在 20%–40% 压缩比下平均恢复率 > 90%，在大多数基准上均优于或接近无后处理 Fine‑tune 的最优方案；在多模态视频生成任务中，CLIP 分数几乎无损。

**⚠️ 局限性**

局限性：目前仅支持成对层的共享字典，无法一次性共享超过两层；若扩展至多层共享，需要更复杂的张量分解或迭代数值方法，可能带来额外计算和数值不稳定；另外，框架依赖校准数据的白化质量，极端稀疏或低维激活可能导致白化不稳定。

---

## 429. On Behavioral Alignment of Model-Code and Human-Code Understandability via Behavioral Proxies

**arXiv ID:** 2609.26101 | [PDF](https://arxiv.org/pdf/2609.26101v1)

**作者:** Xiaokai Rong `[一作]` (University of Texas at Dallas), Tien N. Nguyen `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将代码可读性视为读者与代码之间的关系属性，并引入多种基于行为代理（BPMU）的评估方法，比较大型语言模型（LLM）对代码的理解与人类（不同专业水平）的理解的一致性。

**💡 创新点**

创新点在于：①将“读者”概念扩展至人类与LLM，形成关系式可读性；②设计无监督与监督的语义自一致性（P1–P3）代理，自动化衡量LLM的代码理解；③系统评估LLM与不同人类读者子群体的对齐程度，揭示LLM更贴近专业开发者。

**🔧 技术方法**

技术包括：自一致性采样、多路推理、BERTScore-F1语义相似度、专家LLM作为Meta-Reviewer或Judge、阈值校准与非合规性度量、角色条件提示实验。

**📊 数据集**

数据集来源于先前的Scalabrino等人的Java代码片段评估数据（444条评估），其中包含专业开发者、硕士、博士、本科生的可读性标签（ABU_50%）。

**📈 对比分析**

通过将P0–P3与ABU_50%进行二分类评估，使用准确率、F1、AUC等指标；实验结果显示P1–P3在专业开发者子组上F1可达70%以上，整体对齐率显著高于浅层机器学习基线，P0与P1/P2/P3之间的交叉一致性表明代理可靠。

**⚠️ 局限性**

局限性包括：仅基于Scalabrino数据，可能缺乏真实项目多样性；专家LLM被视为“黄金”，但可能偏向专业视角；角色条件提示未能改善对齐，需进一步研究；阈值设定对性能影响显著，需更系统化。

---

## 430. Acoustic Ellipses: Bio-Inspired Omnidirectional Echolocation in Cooperative Multi-Agent Systems using Frequency Sweeps

**arXiv ID:** 2609.26085 | [PDF](https://arxiv.org/pdf/2609.26085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 431. Instructional Governance by Design: A Framework for AI in Computing Education

**arXiv ID:** 2609.26098 | [PDF](https://arxiv.org/pdf/2609.26098v1)

**作者:** Ethan Dickey `[一作]` `[通讯]` (Purdue University), Ethan Dickey (Purdue University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并阐释了“教学治理（Instructional Governance）”框架，用以在生成式人工智能（GenAI）教学工具的设计与部署中明确教学权威、学习者主体、人工责任、上下文边界与评估可视化的六个维度；通过案例组合与外部工具的治理剖析，展示了不同教学功能对应的治理配置；并给出可迁移的设计问题与评估指标。

**💡 创新点**

创新点在于将治理概念从机构层面提升到教学工具层面，形成了可操作的治理轮廓（Governance Profile），并提供了六维度的设计、评估与迭代指南；同时通过跨案例对比揭示治理机制的组合性与可复用性，推动AI教学工具的责任性与可持续性设计。

**🔧 技术方法**

技术方法主要为设计分析、案例研究、文献合成与跨工具治理映射；并利用现有的教学工具（如GTA模拟器、反思生成器、论坛草稿助手、图示生成器、代码风格教练、GenAI使用导入、内容生成工作流）以及公开系统（CodeHelp、CodeAid、CodeTailor、BISCUIT）的治理属性进行对照。

**📊 数据集**

论文并未收集新的实验数据或使用专有数据集；其依据来自公开的教学工具功能说明、作者自研工具的实施日志、以及文献中的使用与评估报告。

**📈 对比分析**

比较方式为治理配置对齐度评估——通过每个工具的功能、治理维度匹配度与已报道的使用/评估证据（如教师编辑率、学生反馈、任务完成质量等）进行质性对比；并未给出统一的量化性能指标，而是通过案例证据展示治理设计与教学目标的契合度。

**⚠️ 局限性**

局限性包括：①缺乏大规模实证验证，治理匹配度主要基于案例描述与文献回顾；②治理框架对不同学科或更细粒度教学情境的适用性尚未系统测试；③未给出自动化评估工具或算法来量化治理维度；④框架需在未来的实证研究中进一步细化与验证。

---

## 432. From Bilinear to Linear: Differentially Private Federated LoRA via Low-Dimensional Parameterization

**arXiv ID:** 2609.26091 | [PDF](https://arxiv.org/pdf/2609.26091v1)

**作者:** Lele Zheng `[一作]` (Xidian University), Yulong Shen `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在联邦学习环境下，为语言模型的低秩适配（LoRA）实现差分隐私的高效微调框架

**💡 创新点**

提出统一低维参数化和基于预热统计的异质性敏感等距投影（HSIP），解决了LoRA的聚合不匹配和噪声二次放大的两大难题

**🔧 技术方法**

低维投影、DP‑SGD、预热统计收集、可测序列化投影矩阵、聚合算法

**📊 数据集**

GLUE、MNLI、RoBERTa、GPT‑2 E2E NLG Challenge 等自然语言处理数据集

**📈 对比分析**

与 FedAvg、FFA‑LoRA、FedSVD、FedASK 等基线比较，在私有与非私有场景下均取得更高准确率（最高提升约 3–5%），通信开销下降约 80%/90%，在不同异质性水平下保持鲁棒性

**⚠️ 局限性**

在极端数据异质性（α=0.1）下效果略逊于部分基线；需要预热阶段收集统计，增加一次额外通信与计算开销，投影设计对收敛仍有一定依赖

---

## 433. CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval

**arXiv ID:** 2609.26086 | [PDF](https://arxiv.org/pdf/2609.26086v1)

**作者:** Daeyoung Roh `[一作]` (Independent Researcher), Donghee Han `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于冻结句子嵌入覆盖余量的单阈值路由器CoVeR，用来在检索增强式问答代理中跳过不必要的LLM验证调用，从而降低成本并保持答案准确。

**💡 创新点**

创新点在于发现仅使用覆盖余量即可识别检索状态明显不完整的情况，单一阈值即可在不重新调优的情况下将验证调用削减约62–68%，并且该阈值可迁移到不同解码器、代理规模以及公开的NLI模型。

**🔧 技术方法**

使用了冻结的E5句子编码器计算覆盖余量、单阈值路由、生成的期望跳转声明、可学习的1M参数掩码器（drafter）以及训练好的验证器、零样本提示判断器和公开NLI交叉编码器等技术。

**📊 数据集**

评估数据集包括HotpotQA（干扰子集）、2WikiMultihopQA和MuSiQue，并在HotpotQA Setting‑B做了开放语料库实验。

**📈 对比分析**

采用留一数据集交叉验证（LODO）选取阈值后，在三数据集上与全检索、始终验证两种基线对比，EM差异≤0.3pp，验证调用减少62–68%，在MuSiQue中几乎消除所有调用。

**⚠️ 局限性**

局限性包括仅在Self‑Ask代理和三大多跳QA数据集上验证，依赖冻结E5编码器和生成期望声明的质量；验证器与门的错误高度相关，无法纠正错误；在开放语料库和非Wikipedia领域的泛化尚未充分评估。

---

## 434. TopoCompress: Topology Aware Token Compression Algorithm for Distributed Edge MoE Inference

**arXiv ID:** 2609.26061 | [PDF](https://arxiv.org/pdf/2609.26061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 435. Selection-Invariant Communication Compilers for Privacy-Aware Multi-Agent LLM Workflows

**arXiv ID:** 2609.26076 | [PDF](https://arxiv.org/pdf/2609.26076v1)

**作者:** Jinghan Xu `[一作]` (Nankai University), Hankai Liu `[通讯]` (Nankai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种选择不变（selection‑invariant）通信编译器，解决多代理大型语言模型在授权后仍可通过消息形式泄露私有状态的问题。

**💡 创新点**

创新点在于正式定义并量化了选择通道泄漏（selection‑channel leakage），引入选择不变性约束，提供可审计的确定性与公开随机化实现，并给出组合通信层非干预的理论保证。

**🔧 技术方法**

技术实现包括：授权对象检索、授权投影、选择不变的消息渲染、依赖性检查与精确的功能门（utility gate），以及基于规则的接受/拒绝验证。

**📊 数据集**

实验使用了 132 场 AgentLeak 通讯重放、3 个控制任务集合（企业采购、医疗转诊、HR 支持）共 2,400/900 条案例、100 任务的可执行 LangGraph 流程，并采用 Qwen、Llama、Mistral 等大型语言模型作为攻击者。

**📈 对比分析**

与传统红action、操作性消息、OLRS、CPD、上下文隐私重构等基线相比，SICC 在所有评估指标上保持零统计显著的额外泄漏（ExcessGain≈0），同时保持 100% 的协议执行率和高接收方满意度。

**⚠️ 局限性**

局限性包括：依赖于完备且可信的授权边界；仅针对授权后表示层，无法阻止已授权信息本身的泄漏；实现受限于结构化消息格式；对未知或非结构化表达形式的鲁棒性尚未验证。

---

## 436. Towards Adaptive Federated Graph Clustering: A Global Community-aware Contrastive Learning-based Approach

**arXiv ID:** 2609.26063 | [PDF](https://arxiv.org/pdf/2609.26063v1)

**作者:** Yinlin Zhu `[一作]` (Sun Yat-sen University), Miao Hu `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在联邦环境下实现无标签图数据的聚类，解决了缺乏预定义簇数和客户端间社区分离不充分的问题。

**💡 创新点**

提出AdaFGC框架：使用过完备的全局社区锚点并通过跨客户端锚点演化自适应确定簇数；引入基于全局锚点的三层对比学习（社区、节点、拓扑）提升跨客户端聚类一致性。

**🔧 技术方法**

采用图神经网络编码器、低通图滤波、K‑means初始化锚点、全局锚点演化、跨客户端聚合（FedAvg）以及三种对比损失函数。

**📊 数据集**

在八个公开数据集上评估：CiteSeer、PubMed、Amazon‑Computer、Amazon‑Photo、Questions、ogb‑arxiv、ogb‑products、Reddit。

**📈 对比分析**

与多种监督与无监督联邦图聚类基线相比，AdaFGC 在 ACC、NMI、ARI、F1 等指标上均实现显著提升，尤其在大规模图数据上保持竞争优势。

**⚠️ 局限性**

对锚点数量、合并阈值和对比损失权重等超参数仍敏感，需要在不同任务中进行细致调优；在极端异构或极少标签场景下的鲁棒性仍需进一步验证。

---

## 437. ChainUQ: Reasoning Consistency-Aware Uncertainty Quantification for Large Language Models

**arXiv ID:** 2609.26060 | [PDF](https://arxiv.org/pdf/2609.26060v1)

**作者:** Dahai Yu `[一作]` (Florida State University), Guang Wang `[通讯]` (Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于推理一致性的LLM不确定性量化框架，能在单一次生成后评估答案可靠性

**💡 创新点**

将不确定性分解为模型本身对结论的自我置信度和推理链的一致性校准，利用对齐感知轻量UQ头和推理一致性校准器

**🔧 技术方法**

内部状态投影、门控池化、结论标记嵌入、可学习的可靠性银行、条件逻辑校准（R-Log）与Iso回归

**📊 数据集**

HotpotQA（ID）、MuSiQue、StrategyQA、bAbI（OOD）

**📈 对比分析**

与多种无监督和监督基线比较，在AUROC上平均提升3.1%，ECE降低最多45%，且在OOD迁移时保持优势

**⚠️ 局限性**

依赖外部判别器可能带来噪声/偏差；无法修复生成错误；仅在单语、非检索场景验证，缺乏多语言和交互评估

---

## 438. One Domain, Many Tongues: Composing Domain and Language LoRAs for Cross-Lingual Remote-Sensing MLLMs without Paired Data

**arXiv ID:** 2609.26097 | [PDF](https://arxiv.org/pdf/2609.26097v1)

**作者:** Xuechen Li `[一作]` `[通讯]` (University of Minnesota), Xuechen Li (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为远程感知多模态大语言模型（MLLM）在不使用多语种RS样本的情况下添加新语言，提出并验证了互相正交的域-语言组合（Mutually Orthogonal Domain–Language composition）方案。

**💡 创新点**

创新点在于：① 通过一个对称的互相正交约束，保证域LoRA和语言LoRA在训练期间保持互不干扰；② 该约束单一且可训练，使模型在所有种子上都能在目标语言下回答RS问题且不损失原始文本多语种能力；③ 通过定量对比展示该方法在目标语言准确率和文本保留方面显著优于现有训练、合并与正交方法。

**🔧 技术方法**

使用技术：LoRA参数高效微调、LLaVA-1.5-7B多模态框架、互相正交正则化（矩阵行/列归一化乘积惩罚）、对比学习式功能一致性正则化（对比式功能一致性），并在训练中采用无梯度裁剪、EOS监督纠正等。

**📊 数据集**

数据集：英语RS指令集GeoChat‑Instruct（约100k样本）用于域LoRA训练；五种语言（西班牙语、阿拉伯语、印地语、越南语、斯瓦希里语）各自使用Bactrian‑X（约67k样本）用于语言LoRA训练；评估采用GeoChat‑Bench分类holdout（2,195条/语言）和Belebele多语种文本数据。

**📈 对比分析**

对比方法：与无正交的Joint、训练时长翻倍的Joint 2×、功能一致性版本、基于正交的OSRM、O‑LoRA以及多种训练自由的合并方法（线性、TIES、DARE、KnOTS‑DARE‑TIES、CAT、层交换）等进行对照。实验结果显示，带互相正交约束的方案在A∩L（正确且在目标语言）上达到56–71%，显著高于其它方法（<10%），并在西班牙语场景分类上超过Qwen2.5‑VL‑7B；同时文本多语种能力保持与基线相当。

**⚠️ 局限性**

局限性：① 仅在1,000步的匹配计算预算下验证，未探讨完整收敛；② 只针对单一远程感知领域和7B规模模型；③ 对低资源语言（尤其是非拉丁字母）效果有限，五语种共享时性能下降；④ 依赖英文中心模型，若基座已具多语种能力则需进一步验证；⑤ 评测数据依赖机器翻译，可能引入翻译偏差；⑥ 未对长文本、开放式RS任务或更大规模训练进行实验。

---

## 439. Situation Aware Locomotion for Dual Mobile Cobots in Shared Environments

**arXiv ID:** 2609.26083 | [PDF](https://arxiv.org/pdf/2609.26083v1)

**作者:** William Moraes `[一作]` (Technological University of Uruguay), Ricardo B. Grando `[通讯]` (Technological University of Uruguay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对两台移动协作机器人在共享工业环境中进行情境感知驱动的运动控制框架。

**💡 创新点**

创新点在于将操纵臂状态（装载、空闲、等待等）与机器人姿态、负载、共享区占用、障碍物状态和冲突预测结合，形成三层（感知-理解-预测）的情境感知模型，并以此实现动态优先级分配和冲突规避。

**🔧 技术方法**

采用ROS 2 Jazzy与Gazebo Sim仿真平台，利用基于单轨模型的短期预测、简单路径跟踪控制器以及安全命令过滤器；通过手工定义的三种实验场景（装载优先、交叉冲突、路径阻塞）进行评估。

**📊 数据集**

使用了900次仿真试验（3种场景×5个随机种子×30次重复），每个场景为4 m×4 m工作区；并在论文中引用了相同尺寸的真实物理实验布局作为未来验证基础。

**📈 对比分析**

与两种基线方法（独立导航和固定优先级协调）对比，情境感知方法在所有场景中均实现100 %任务成功率，最短完成时间、无安全停止、无共享区冲突；独立导航和固定优先级仅各获得约33 %成功率。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏真实机器人实验；工作区规模受限，未检验在更大、更复杂工况下的可扩展性；模型为规则驱动，未使用深度学习或自适应学习机制，可能在动态变化环境中表现不佳。

---

## 440. Benchmarking Open-Source Speech Emotion Recognition in Naturalistic Mandarin Spine Clinic Consultations: A Pilot Validation Study

**arXiv ID:** 2609.26054 | [PDF](https://arxiv.org/pdf/2609.26054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 441. Margin-Drop Coordinates for Cross-Budget Robustness Evaluation

**arXiv ID:** 2609.26081 | [PDF](https://arxiv.org/pdf/2609.26081v1)

**作者:** Yanliang Huang `[一作]` (Technical University of Munich), Amr Alanwar `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在固定攻击预算评估下，浅层攻击是否能正确预测模型在更高预算下的鲁棒性衰退；通过对预训练冻结视觉编码器的margin-drop坐标（margin slack、one-step shortfall、drift）进行分解，揭示了预算灵活性和模型排名变化的机制。

**💡 创新点**

创新点在于提出了一种基于单个样本的三维margin-drop坐标体系，可在浅层评估中捕捉模型在更高攻击预算下的崩溃趋势，并证明浅层漂移（drift）是预测跨预算崩溃的强大指标；同时利用该分解评估鲁棒性干预措施的具体效果。

**🔧 技术方法**

使用了梯度投影攻击（PGD）、一阶线性化预测（FGSM）、以及对margin-drop坐标的归一化和计算；在干预实验中引入了LoRA、局部线性正则化（LLR）以及坐标定向目标函数。

**📊 数据集**

在ImageNet-100子集（10类×100个样本，共994张图像）上对42种预训练冻结编码器进行评估，并在ImageNet-1K上进行迁移学习验证。

**📈 对比分析**

与传统的生存率（survival rate）评估相比，浅层漂移在Spearman相关系数上达+0.811，显著高于生存率的-0.006；在模型优先级排序（即哪些模型需要进一步深度评估）时，漂移排名能够覆盖更高比例的预算脆弱模型（11/17），优于仅依赖生存率（5/17）。

**⚠️ 局限性**

局限性包括仅评估冻结编码器和固定logit头；未考虑自监督或对抗训练模型；在更大规模数据集和多模态系统中的可迁移性待验证；此外，漂移坐标依赖于先验的固定竞争者选择，可能对某些模型结构产生偏差。

---

## 442. The Fleet Is the Model: Engineering Collective Intelligence with Fusion-MoA Pioneer R1

**arXiv ID:** 2609.26080 | [PDF](https://arxiv.org/pdf/2609.26080v1)

**作者:** Zongyou Yang `[一作]` (Imperial College London), Yinghan Hou `[通讯]` (Imperial College London)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Fusion-MoA运行时，实现了多模型舰队通过版本化Profile在单一OpenAI兼容接口下协同工作。

**💡 创新点**

通过身份、权限和演化三个合同实现模型舰队的统一身份、单一工具执行权和成员可演化的治理架构。

**🔧 技术方法**

采用模型Cell、版本化Profile、Evidence Snapshot、Typed Packet、单Executor与只读Analyst、并行推理与有限证据传递等技术，并部署在AMD W7900D GPU上。

**📊 数据集**

在HMMT 2026年P1–P10切片和Terminal‑Bench 2.1 20任务数据集上评估。

**📈 对比分析**

将最佳单个Cell的6/10问题解答与集合Profile的8/10进行对比，并在20任务中只允许单Executor使用工具，证明集体能获得更高解答率并保持工具调用单一来源，性能提升约33%。

**⚠️ 局限性**

实验规模仅为八个Cell，且对任务类型有限制，系统治理与回滚仍需人工操作，未证明在更大规模或不同领域的可扩展性与稳定性。

---

## 443. Test-time Reinforcement Learning for Anomalous Video Understanding

**arXiv ID:** 2609.26099 | [PDF](https://arxiv.org/pdf/2609.26099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. RankCert: When Can Simulated Learners Safely Select an AI Tutor? Robust Decision Certification Under Structural Uncertainty

**arXiv ID:** 2609.26069 | [PDF](https://arxiv.org/pdf/2609.26069v1)

**作者:** Nizam Kadir `[一作]` `[通讯]` (Singapore University of Technology and Design), Nizam Kadir (Singapore University of Technology and Design)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一种名为 RankCert 的安全认证框架，用于在多种结构不确定的模拟学习者环境中选取 AI 辅导策略，并通过多条件门控来决定是否给出推荐或拒绝。

**💡 创新点**

创新点在于：①把模型不确定性与决策安全分离，设计了包含概率最佳、95% 阈值、跨域排名、结构覆盖和留一域/模型验证等一系列硬性门控的认证程序；②在同一评估环境下同时对 5 种结构不同的学习者模拟器进行留一评估，以检验决策的鲁棒性；③提供了“证书”形式的可解释性输出，明确标识何时缺乏足够证据。

**🔧 技术方法**

技术包括：模拟学习者（Bayesian-knowledge-tracing、PFA-practice、动态 IRT、注意力历史、因果阈值等 5 个结构族）、模型平均、概率最佳评估、95% 最高后悔阈值、跨域排名一致性检验、留一域/留一族验证、聚合的预测足够性门控，以及基于模拟器输出的后验抽样评估。

**📊 数据集**

使用了已授权的、去标识化的 EdNet-KT1 次级数据集进行校准与预测足够性检验，随后在基于该校准参数生成的合成模拟环境中进行 1,280 个 held‑out 设置的评估。

**📈 对比分析**

与 7 种预设基线（单模拟器点选、置信门控点证书、等权模型平均、概率最佳阈值、Top‑set 选取、最小化最坏后悔、下尾分布鲁棒选择）比较。结果显示，RankCert 在总决策损失上显著低于单模拟器点选（平均差异 -0.0066），但在相同覆盖率下与置信门控点证书的选择性风险并无显著差异；覆盖率仅为 3.75%，显示其在多样化情境下常常拒绝给出决策。

**⚠️ 局限性**

局限包括：①覆盖率极低，许多设置下都会拒绝给出决策；②只评估了 5 种结构不同的模拟器，无法覆盖所有真实学习者行为；③未进行任何真实用户实验，无法验证对人类学习者的效果或因果影响；④所有门控阈值、成本设定均为预先固定，缺乏对不同应用情境的泛化性；⑤结果对拒绝成本高度敏感，成本增大会导致排名倒置；⑥未考虑公平性、安全性、可解释性等更广泛的系统安全指标。

---

## 445. BDSLI: A hybrid CNN-Transformer model for Bengali Sign Language interpretation

**arXiv ID:** 2609.26088 | [PDF](https://arxiv.org/pdf/2609.26088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 446. FIRE: Failure-Informed Runtime Engineering for Reliable Language-Model Agents

**arXiv ID:** 2609.26048 | [PDF](https://arxiv.org/pdf/2609.26048v1)

**作者:** Nikita Agarwal `[一作]` (Failproof AI), Nivedit Jain `[通讯]` (Failproof AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种轻量级运行时策略层，在不修改模型权重或用户提示的前提下，通过在关键状态插入自然语言指令或拒绝操作，显著提升语言模型代理在Terminal-Bench 2.1任务上的可靠交付。

**💡 创新点**

创新点在于将基于失败观察的策略抽象为可复用的规则集，并通过随机化五臂实验证明策略内容而非仅仅中断或泛化自检能带来性能提升；此外展示了在不同模型层级上以成本友好的方式弥补模型能力差距。

**🔧 技术方法**

技术实现包括：① 运行时策略抽象（Eligibility、Runtime Predicate、Intervention、Release Condition、State Tracking）；② 在Codex CLI容器化环境中集成策略；③ 设计五臂随机对照实验（真实策略、触发假策略、始终验证、重考虑、基线）；④ 采用任务级自举与置换检验评估显著性。

**📊 数据集**

使用的数据集为Terminal‑Bench 2.1，共89个任务（实验中使用87个），覆盖多种交互式命令行场景。

**📈 对比分析**

评估方法为在每个任务上进行两次尝试，对比基线、假策略、始终验证、重考虑与真实策略；结果显示真实策略在14个可选任务中提升25个百分点（p=0.061），在完整套件中对Sol层提升9.2个百分点、对Terra层提升5.7个百分点，且Terra在策略辅助下达到Sol基线水平，成本仅为Sol的一半。

**⚠️ 局限性**

局限性包括：仅在单一英文Benchmark、单一Harness（Codex CLI）和单一模型族（GPT‑5.6）上测试；每个任务仅尝试两次，无法估计长期可靠性；策略源代码和规则需专家手工编写，未验证对其他任务的泛化；策略可能引入额外延迟或复杂度，且只能提升可靠性，无法增加模型能力。

---

## 447. CoEvo: Oracle-Grounded Self-Evolution of a Single Model for Multi-Step Causal Reasoning

**arXiv ID:** 2609.26094 | [PDF](https://arxiv.org/pdf/2609.26094v1)

**作者:** Jian Zhang `[一作]` (Zhejiang University), Yizhi Liu `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种名为 CoEvo 的自演化框架，使单一可部署的 LLM 在多步因果推理任务中通过交替扮演 Proposer 与 Solver 两个角色，在过程层面实现自我提升。

**💡 创新点**

核心创新点是利用“生成难、验证易”的不对称性，将判定权交给外部可确定的领域 Oracle；通过模型自身产生的竞争链中的争议定位错误步骤，Oracle 进行逐步验证并提供奖励，进而实现无教师监督的高质量自演化。

**🔧 技术方法**

技术包括：1）Oracle 驱动的步骤验证与任务认证；2）Proposer/ Solver 角色切换与联合训练；3）基于争议的多代理辩论；4）结构化的多步推理链生成与奖励设计；5）与外部规则引擎/模拟器的接口。

**📊 数据集**

数据集覆盖三大领域：工业设备故障诊断（AHU 数据集）、临床诊断（DDXPlus）、法律推理（MSLR），并在工业/临床子域进行未见类别与跨系统泛化评估。

**📈 对比分析**

在工业、临床和法律三类基准上，CoEvo 在路径正确率上分别比最佳专有模型高 18.0、13.0、1.1 点，整体平均 87.73%（对比 87.31%）；与知识蒸馏、Naïve 自演化等基线相比提升 20+ 点；在未见类别和跨系统场景中保持 80%+ 的正确率，显著优于其它方法。

**⚠️ 局限性**

局限性：依赖可编码且可查询的领域 Oracle（规则引擎/模拟器），因此在缺乏完整、可执行知识表述的领域难以直接应用；Oracle 的覆盖率与准确性直接决定训练质量；模型对 Oracle 的过度依赖可能限制在完全无监督或开放式推理任务中的推广。

---

## 448. Match One, Learn with Graph: One-to-Graph Query Collaboration with Backward Sharing for Object Detection

**arXiv ID:** 2609.26092 | [PDF](https://arxiv.org/pdf/2609.26092v1)

**作者:** Wenxiao Fan `[一作]` (Beijing Institute of Technology), Kan Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在DETR中提出一种BS-O2G插件，利用稀疏的预测感知图和持久查询基准实现查询间的协作，解决单一查询监督导致的证据碎片化问题。

**💡 创新点**

创新点在于：①不改变O2O匹配与正样本标注，仅在解码后构建基于特征、边界框和类别概率的稀疏图；②前向使用One-to-Graph（O2G）进行信息融合；③后向利用Backward Sharing（BS）在图邻居上共享持久基准的梯度，提升优化空间协同；④整体实现无需额外正样本分配，参数与FLOP增量极低。

**🔧 技术方法**

采用的技术包括：持久查询基准（Learnable query basis）、稀疏预测感知图（基于特征余弦、IoU和类别余弦构建），One-to-Graph特征校准（相对消息传递），Backward Sharing梯度路由（使用转置脱离梯度的邻接矩阵），以及标准DETR的Hungarian匹配与损失。

**📊 数据集**

在MS‑COCO（val2017）和CrowdHuman数据集上进行评估，使用多种主流DETR变体（DEIM、RT‑DETRv2、D‑FINE等）和不同骨干（ResNet‑50/101、HGNetV2-B4/B5）。

**📈 对比分析**

与基线相比，BS‑O2G在24‑epoch设置下使用300个查询即可达到53.5 AP，超过使用900查询的强基线；在更长训练期内，AP可提升至54.5以上；在CrowdHuman上，训练90 epoch即可超过基线120 epoch的表现，AP提升约1.4点。计算开销仅增0.69%参数/0.75%FLOP，推理延迟+1.8 ms，保持近乎无额外成本。

**⚠️ 局限性**

局限性包括：①仍依赖解码后特征的质量，对极小目标的改善有限；②BS仅在训练阶段启用，对推理不产生影响，导致部署时无法直接利用梯度协作；③对图邻接稀疏度（K）和BS权重（λ_B）等超参敏感，需要经验性调优。

---

## 449. Vision-Language Models as copilots for Autonomous UAV Navigation: Analysis of Latency and Reliability in Degraded Environments

**arXiv ID:** 2609.26084 | [PDF](https://arxiv.org/pdf/2609.26084v1)

**作者:** Hiago Sodre `[一作]` (Technological University of Uruguay), Ricardo Grando `[通讯]` (Technological University of Uruguay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在GPS失效的室内环境中，提出并验证了一种将有限状态机与异步视觉‑语言模型（VLM）协同工作的混合控制架构，用于无人机自主导航；

**💡 创新点**

核心创新在于引入了基于速度缩放和位置回溯的时延感知补偿机制，以及严格的结构化JSON接口，实现了高层语义决策与低层实时控制的无缝对接；

**🔧 技术方法**

利用ROS2、PX4、Gazebo进行软件在环（SITL）仿真，VLM后端包括Moondream（1.8B）、LLaVA‑LLaMA3（8B）和LLaMA 3.2 Vision（11B）等模型，配合Ollama本地推理服务器；

**📊 数据集**

使用自建的室内仿真场景（包含白色线网、QR码、彩色门、移动障碍物），并通过破损QR码构建降级环境进行实验；

**📈 对比分析**

通过在正常与降级两种环境下多次任务运行，评估平均与峰值时延、轨迹误差、JSON格式完整性和任务完成率；实验表明，尽管11B模型时延最高，但在降级环境中保持100%格式完整性并获得78.6%的任务成功率；

**⚠️ 局限性**

主要局限在于大型VLM模型的高推理时延、对硬件资源的需求，以及在真实世界中对视觉噪声与模型幻觉的鲁棒性尚未充分验证；

---

## 450. Optimizing Denoising Trajectories in dLLMs: A Lightweight Evolutionary Heuristic Approach

**arXiv ID:** 2609.26052 | [PDF](https://arxiv.org/pdf/2609.26052v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 451. Differentiable Policy Transport over Multi-Layer Network Feasibility Geometry

**arXiv ID:** 2609.26068 | [PDF](https://arxiv.org/pdf/2609.26068v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将网络可行性几何直接嵌入策略学习的框架 NFG-RL，并在多租户无线边缘计算环境中实现可行性约束的可微传输。

**💡 创新点**

创新点在于：①将多层约束统一为残差包含形式，并编译成可微的类型化残差块；②设计了具有可微KKT的各向异性、批判者倾斜的传输算子，能够在保证几乎必然可行的同时实现探索空间的可行切线收缩与梯度滤波；③证明了该传输在一阶上比单纯投影更优，并与经典 backpressure 调度在数学上等价。

**🔧 技术方法**

主要技术包括：残差几何建模、可微变分传输算子、KKT条件隐式求导、近似解算器（精确解、未展开流、切线传输、混合离散传输）、强化学习（PPO/SAC）、以及回传时的隐式梯度传递。

**📊 数据集**

使用公开轨迹驱动的无线边缘仿真环境 SMEC‑5G 与 5G‑C3（分别基于 SMEC 与 UCC‑5G 轨迹）以及自定义的受控三租户模拟环境；所有数据通过对原始测量进行降维、标准化和重采样得到。

**📈 对比分析**

与 Greedy‑EDF、BP‑DPP、PPO/SAC‑Penalty、PPO‑Lagrangian、CPO、GNN‑SAC、OptLayer‑SAC、以及基于 MILP 的近似最优搜索等 10 种基线进行比较；在 SMEC‑5G 与 5G‑C3 中，NFG‑RL 在可行性 utility 上提升 37.5–41.5%，原始动作违约率下降 48.5–60.8%，P99 延迟降低 57.0–75.5%，并在大多数系统扰动下保持最高的 SLA 满足率。

**⚠️ 局限性**

局限性包括：①对离散决策仍需依赖后处理或松弛方案，精确可行性保证仅在最终修复后实现；②传输算子涉及 KKT 求解，计算开销相对较高；③在部分极端扰动（如移动性升高）下不一定优于传统基线；④仅在仿真与轨迹条件下验证，真实网络部署仍需进一步实验。

---

## 452. Adversarial Course-of-Action Generation: Game-Theoretic Multi-Agent Algorithms for COA matching & COA generation

**arXiv ID:** 2609.26059 | [PDF](https://arxiv.org/pdf/2609.26059v1)

**作者:** Natan Vidra `[一作]` (Anote AI), Spurthi Setty `[通讯]` (Stevens Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了COA-Bench基准，用于评估在对抗自我对弈场景下的行动方案（COA）生成策略，提供离线可复现的评估框架。

**💡 创新点**

创新点在于将COA生成与对抗响应分离，提出多维度评估（质量、优势、纳什间隙、教义一致性、多样性）并通过采样最佳响应揭示单样本评估的局限，以及引入两阶段多代理委员会机制提升决策质量。

**🔧 技术方法**

使用Python实现离线评估脚本，结构化COA表示（链式动作、条件分支、工具调用），统计评估指标，模拟自我对弈，计算优势分数、Nash gap、教义启发式评分，并实现多代理委员会的裁决与修订流程。

**📊 数据集**

使用合成情境数据集，共50个场景，覆盖5个操作模板（城市稳定、海上截获、多域战斗、反导、灾难救援），每个情境基于种子生成并附带蓝方初始COA。

**📈 对比分析**

通过比较单样本、采样最佳响应、教义感知响应和两阶段多代理委员会四种策略，发现采样最佳响应使蓝方优势从0.516降至0.485、胜率从0.920降至0.820；委员会策略将优势提升至0.509，胜率保持0.820。

**⚠️ 局限性**

局限性在于评估基于合成、离线的手工设计指标，缺乏真实战场数据和专家验证；质量、教义与战斗检查公式化可能与实际操作不符；评估的策略仍为随机/模板化，未测试LLM驱动的生成方案。

---

## 453. Beyond Classification Accuracy: Quantifying Fingerprint Complexity in Encrypted Darknet Services

**arXiv ID:** 2609.26096 | [PDF](https://arxiv.org/pdf/2609.26096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 454. Observing the Conduct of Systematic Reviews with Generative AI Support: An Experience Report from a Graduate Software Engineering Course

**arXiv ID:** 2609.26057 | [PDF](https://arxiv.org/pdf/2609.26057v1)

**作者:** Danilo Monteiro Ribeiro `[一作]` (AIBL, Cesar School), Gilberto Sussumu Hida `[通讯]` (AIBL, Cesar School)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一门软件工程研究生课程中，组织博士生开展含有和不含有生成式 AI（ChatGPT）支持的系统文献综述（SLR）实验，记录并观察学生在规划、检索、筛选和初步提取等阶段与 AI 的交互与决策；

**💡 创新点**

首次系统性记录了学生在真实课堂环境中如何将生成式 AI 融入 SLR 流程，并归纳三种 AI 使用模式（改进助手、备选生成器、决策执行者），为教育实践提供实证指导；

**🔧 技术方法**

使用 OpenAI 的 ChatGPT（ChatGPT‑5.2 版）作为辅助工具，并配套预设的提示模板、表格模板及教师监督；

**📊 数据集**

数据来源为学生在课堂上产生的研究问题、检索式、筛选决策表和初步提取表，全部匿名化后公开在 Zenodo 上；

**📈 对比分析**

未进行量化性能对比，主要通过课堂观察、记录和三种 AI 使用模式的质性分析来评估效果；

**⚠️ 局限性**

局限包括仅一门课程、三组少量学生、单日实验、AI 与手工阶段未严格分离、提取和综合阶段不完整、并出现个人账号使用导致数据缺失等。

---

## 455. Towards Intent-Aware Human-Robot Teaming: A Platform for Search-and-Rescue Operations

**arXiv ID:** 2609.26051 | [PDF](https://arxiv.org/pdf/2609.26051v1)

**作者:** Rohith Prem Maben `[一作]` (Lund University), Elin Anna Topp `[通讯]` (Lund University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计并实现了一个面向搜救（SAR）任务的人机协作平台，集成了Unity可视化、ArduPilot仿真与MAVLink通信，实现了对人类指挥员与异构无人机/无人车交互的实时记录与分析。

**💡 创新点**

创新点在于将Joint Control Framework（JCF）与人类意图推理相结合，构建了多层级认知控制的意图模型，并通过HMI‑Temporal Analysis对指挥员的感知、决策与行动进行细粒度拆解，支持无人机对指挥员意图的自适应辅助。

**🔧 技术方法**

使用技术包括Unity 3D游戏引擎、ArduPilot软件在环仿真、MAVLink v2协议、UDP数据传输、Joint Control Framework以及后续计划的贝叶斯意图推理框架。

**📊 数据集**

数据集为平台生成的操作日志与遥测数据，包含指挥员在不同搜救情景下的摄像头切换、命令输入、决策点与车辆轨迹，全部在模拟环境中采集。

**📈 对比分析**

论文未给出与现有方法的定量对比或性能指标，主要通过案例演示展示平台能够捕捉并解析人机交互流程，验证了意图模型在无人机协助中的可行性。

**⚠️ 局限性**

局限性包括仅在单一无人机+无人车对上进行验证，缺乏多代理扩展与真实硬件部署，且意图推理仍为设计阶段，缺乏正式的评估与泛化能力。

---

## 456. Canonical locks that encode part-whole hierarchies

**arXiv ID:** 2609.26046 | [PDF](https://arxiv.org/pdf/2609.26046v1)

**作者:** Rajat Modi `[一作]` (University of Central Florida), Yogesh Singh Rawat `[通讯]` (University of Central Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出“canonical locks”这一几何原语，构建了可在神经网络内部自组织的层级结构，并实现了仅用单张样本即可训练出能泛化到测试集的 Asynchronous Perception Machine (APM) 模型；

**💡 创新点**

创新点在于用向量相位差和旋转锁定机制编码多层次的部分-整体层级，解决了传统 Transformer/胶囊网络在动态层级结构上的崩塌与路由问题；

**🔧 技术方法**

核心技术包括高维向量相位编码、双向（自下而上与自上而下）同步网络、协商协议（agreement procedure）与斜率更新（slurp update）、深层权重对齐（Deep Weight Alignment）以及对称正则化；

**📊 数据集**

主要使用 CelebA‑HQ 数据集构建 5 层人脸部件层级，实验中对单张人脸图像进行训练后评估在未见图像上的分割与层级解析；

**📈 对比分析**

与传统 Transformer、胶囊网络和 GLOM/APM 之类的同步模型相比，APM 在单样本训练下即可实现高质量的分割与层级推断，且推理速度显著提升；

**⚠️ 局限性**

局限性包括对高维空间中的随机正交性难以快速收敛、对超参数（如迭代次数、权重对齐强度）的敏感性、以及在更复杂多类、多模态数据上的可扩展性待验证。

---

## 457. ToW3D: Consistency-aware Interactive Point-based Mesh Editing on GANs

**arXiv ID:** 2609.26078 | [PDF](https://arxiv.org/pdf/2609.26078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 458. Fast Matrix Multiplication in fp8: Certified Coefficient Optimization and Measured Error

**arXiv ID:** 2609.26077 | [PDF](https://arxiv.org/pdf/2609.26077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 459. RECAP: Relation Evidence Calibration for Detecting Spatial Relation Hallucinations in Vision-Language Models

**arXiv ID:** 2609.26093 | [PDF](https://arxiv.org/pdf/2609.26093v1)

**作者:** Feixiang Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Hui Xu `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种名为RECAP的关系证据审计框架，用来在多模态大语言模型已生成的空间关系答案上做“接受或放弃”的选择性预测。

**💡 创新点**

创新点在于：①将关系证据与置信度分离，构建关系对立图（claim‑contradiction）并以答复为条件进行风险评估；②采用仅使用预冻结模型的无监督证据收集和校准门控，无需额外训练；③通过组/图像分离的校准实现模型不变的阈值与置信度切换。

**🔧 技术方法**

技术手段包括：关系对立图的构造、图中每条边的yes/no概率差分查询、对抗与支持边的最大/加权和聚合、答案条件化风险计算、基于经验分布的置信度/证据门控、以及对多模型/多提示的可复现评估。

**📊 数据集**

数据集涵盖：VSR、What'sUp、GSR‑Bench、COCO、GQA等，分别用于图像–关系验证、受控问答、外部迁移与对比测试。

**📈 对比分析**

与置信度、VCD（视觉对比）等基线对比时，RECAP在Acc@80提升3–13个百分点、H‑FPR@80下降2–18个百分点、Hall‑AUC提升10–40个百分点；在不同模型与提示下均保持稳健的性能提升，并通过统计置信区间验证显著性。

**⚠️ 局限性**

局限性包括：①仅适用于已冻结答案且预先定义的二元关系词表；②需要标注的校准子集来设定门控阈值；③无法直接处理自由文本、多关系输出；④对模型、域或提示的更改需重新校准；⑤对关系提取、代词消解等前置任务存在依赖。

---

## 460. SpecialEduBench: Benchmarking Vision-Language Models on Knowledge, Skill, and Attitude in Language Intervention for Autistic Children

**arXiv ID:** 2609.26090 | [PDF](https://arxiv.org/pdf/2609.26090v1)

**作者:** Jihoi Na `[一作]` (Kwangwoon University), Unggi Lee `[通讯]` (Korea University Sejong Campus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并发布了专门针对自闭症儿童语言干预的评估基准 SpecialEduBench，涵盖知识、技能和态度三个维度。

**💡 创新点**

创新点在于：①将技能和态度的评测基于真实录制干预场景；②在态度维度交叉压力与监控条件，定位模型在被要求违背时的表现；③使用专家共识作为判别上限，而非固定阈值，构建判定模型。

**🔧 技术方法**

使用技术包括视觉语言模型（VLM）作为被测模型，判定模型（judge）基于LLM（Gemma、Qwen 等）训练的评分器；奖励模型采用 Bradley‑Terry 训练；知识维度采用精确匹配；技能与态度采用评分 Rubric。

**📊 数据集**

数据集：知识维度 4,537 道多选题（来自 Pedagogy Benchmark、医学、心理等资源）；技能维度 200 道基于 45 段公开录制干预视频的场景问答；态度维度 68 道跨压力/监控条件的真实场景，共 192 个评价单元，全部来自公开录制干预视频。

**📈 对比分析**

比较方法：对八个前沿 VLM（4 个闭源 4 个开源）进行评测，使用知识准确率、技能 rubric 平均分、态度通过 P×M 指标以及奖励模型胜率进行对比；结果显示无轴饱和，模型在知识事实上相近，但在情境化任务与压力条件下差距明显；最佳模型在知识上得分 93.5% 但在真诚细胞仍约 10% 失败。

**⚠️ 局限性**

限制：技能、态度的题量受专家审阅限制，仅 200/68 项；数据仅英文且局限于单一临床实践；模型样本少，相关性估计不稳；判定模型未完全覆盖所有错误；知识题目可能存在键值过时导致高错误率。

---

## 461. Learning to Link: Automatic Re-identification of BLE Devices Under MAC Address Randomisation

**arXiv ID:** 2609.26079 | [PDF](https://arxiv.org/pdf/2609.26079v1)

**作者:** Reem Abdulrhman Alghamdi `[一作]` (King Abdullah University of Science and Technology), Marco Mellia `[通讯]` (Politecnico di Torino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究BLE设备在MAC随机化下的再识别问题，提出基于机器学习的自动化跟踪方法。

**💡 创新点**

将再识别视为有监督分类任务，利用仅可被动监听的广告层特征自动学习设备指纹，无需手工签名。

**🔧 技术方法**

使用决策树分类器、字节级3-gram广告数据提取、组感知交叉验证等技术。

**📊 数据集**

使用14台不同类型设备的BLE广告数据，配合噪声库共计约345 k个数据包。

**📈 对比分析**

采用留一RPA-epoch验证，决策树在包级召回率98.6%，误报率<2%，地址级误识别率0.38%。

**⚠️ 局限性**

需预先获取目标设备标记数据，仅针对特定设备；包级误判会影响地址聚合；样本量与设备多样性有限。

---

## 462. SPEANet: Structural Prior Enhanced Attention Network for Parameter-Efficient Remote Sensing Object Detection

**arXiv ID:** 2609.26064 | [PDF](https://arxiv.org/pdf/2609.26064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 463. CricRAG: Retrieval Augmented Vision-Language Models for Personalized Cricket Coaching

**arXiv ID:** 2609.26056 | [PDF](https://arxiv.org/pdf/2609.26056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 464. Cellular-Communication-Level Interpretability for Pathology Foundation Models via Graph Distillation on Microenvironment

**arXiv ID:** 2609.26073 | [PDF](https://arxiv.org/pdf/2609.26073v1)

**作者:** Yuxiang Xiao `[一作]` (South China University of Technology), Kaixiang Yang `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种基于图结构的解释器 G-Interp，利用知识蒸馏将大型病理基础模型的表示迁移到轻量级双分支学生，同时生成细胞级和细胞间交互的可解释证据。

**💡 创新点**

创新点在于将微环境图分支与视觉分支的双分支蒸馏相结合，并通过门控融合和跨注意力交互实现高预测性能与细胞级、交互级可解释性的统一。

**🔧 技术方法**

技术包括 NuLite 细胞分割、GATv2/TransformerConv/SAGPool 等图神经网络、ResNet-18 视觉分支、门控融合、跨注意力交互、余弦相似度蒸馏损失、Score‑CAM 等可解释性评估方法。

**📊 数据集**

实验使用 TCGA‑BRCA 进行自监督蒸馏预训练，随后在 Yale‑HER2、SLN‑Breast、BRACS 三个公开乳腺癌数据集上进行外部验证。

**📈 对比分析**

与单分支 KD 基线和教师 PFM 进行对比，使用 AUC 评估；G-Interp 在三组数据集上均达到或接近教师性能，并在参数量缩减超过 20 倍后仍保持高 AUC。

**⚠️ 局限性**

局限性包括仅在乳腺癌数据集上验证，缺乏对其他癌种的泛化；图分支对细胞分割质量高度依赖，且不同染色或扫描条件下的鲁棒性待进一步评估。

---

## 465. FuncCode: Compressing Kolmogorov--Arnold Networks in Function Space with Hardware-Aware Quantization

**arXiv ID:** 2609.26067 | [PDF](https://arxiv.org/pdf/2609.26067v1)

**作者:** Kazi Ahmed Asif Fuad `[一作]` (Oregon State University), Lizhong Chen `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对Kolmogorov–Arnold网络（KAN）进行压缩，提出了FuncCode方法，将每条边的可学习函数映射到共享代码字典，并通过低位量化和位打包实现极高的压缩率。

**💡 创新点**

创新点：①基于函数空间而非系数空间进行聚类，形成共享代码字典；②将基（base）与基底（basis）分支分别编码，保持两者不同的共享结构；③提供完整的量化、索引打包与导出流程，使压缩结果可直接部署到硬件；④在FPGA上验证了内存节省而不增加时延。

**🔧 技术方法**

采用的技术包括：在固定域上采样每条边的响应构成函数签名；k‑means聚类得到函数代码字典；分别为基和基底生成代码书并进行统一对称量化；对代码字典进行微调；使用位打包压缩索引；在FPGA上进行完整的合成、布线与RTL仿真。

**📊 数据集**

使用的数据集：MNIST、Fashion‑MNIST、CIFAR‑10、CIFAR‑100、Wine；并对一个30.6 M参数、6.1 M边的卷积KAGNN做大规模实验。

**📈 对比分析**

与基准（FP32稠密KAN、基于系数的聚类、MetaCluster等）比较，FuncCode在MNIST上实现31.6×（spline）和17.6×（GRAM）压缩，准确率误差仅0.31–0.34pp；在6.1 M边卷积KAGNN上约20×压缩，CIFAR‑10误差0.54pp、CIFAR‑100误差1.89pp；在FPGA上，压缩后SPLINE KAN的BRAM内存减少3.87×，同时保持相同的周期数与延迟。

**⚠️ 局限性**

局限性：①仅针对表示层面压缩，未对能耗、批量推理等做评估；②对不同KAN家族和极端压缩率的鲁棒性需要进一步研究；③高索引比率导致对索引的存储和解码开销占比高；④实验多在十个随机种子下验证，未覆盖所有超参数组合；⑤方法对训练过程的微调依赖，迁移到其他任务时可能需要重新调参。

---

## 466. EMERGE: Resolution-Agnostic Point Cloud Generation with Equivariant Graph-Based Diffusion

**arXiv ID:** 2609.26039 | [PDF](https://arxiv.org/pdf/2609.26039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 467. On the OpenAI whole-cube bound

**arXiv ID:** 2609.26050 | [PDF](https://arxiv.org/pdf/2609.26050v1)

**作者:** Alex Samorodnitsky `[一作]` `[通讯]`, Alex Samorodnitsky

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文未给出具体研究内容，缺少可识别的实验或理论结果

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法识别使用的技术方法

**📊 数据集**

无法确认使用的数据集

**📈 对比分析**

无法比较的方法与性能指标

**⚠️ 局限性**

缺乏足够信息，无法评估研究的局限性

---

## 468. Recognizable Picture Languages: Separating UREC from coUREC via Communication Complexity

**arXiv ID:** 2609.26047 | [PDF](https://arxiv.org/pdf/2609.26047v1)

**作者:** Antonin Callard `[一作]` (LIRMM -- CNRS and Université de Montpellier), Pascal Vanier `[通讯]` (Université Caen Normandie, ENSICAEN, CNRS, Normandie Univ, GREYC)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过引入通信复杂度提升技术，构造了一个可在 UREC 中识别、其补集不属于 REC 的二维图像语言，从而解决了 Anselmo 等人 2006 年提出的长期未决问题。

**💡 创新点**

创新点在于将通信复杂度的提升方法与可辨识图像语言的局部约束相结合，尤其使用了大规模输入的 gadget（内积函数），实现了证明 UREC 不对补运算封闭的技术突破。

**🔧 技术方法**

核心技术包括：通信复杂度的非确定性与无歧义协议分析、lifting 方案、局部（2×2）约束的拼贴系统、以及利用 Turing 机时空图形对复杂计算的编码。

**📊 数据集**

本文不使用任何实验数据集，所有结论均为理论构造与证明。

**📈 对比分析**

与之前的研究相比，本文通过通信复杂度下界给出了更强的不可约束性证明；性能方面通过证明的上下界展示了构造语言的规模与通信成本之间的精确关系。

**⚠️ 局限性**

局限性在于构造过于人工化、技术细节繁复，且未提供更自然或更易实现的示例，未来工作需要寻找更简洁的构造或扩展到多维 Sofic 移位空间。

---

## 469. Truth for Believable AI: Expressed Doubt, Provenance, and Belief Revision as an Engineerable Stance

**arXiv ID:** 2609.26035 | [PDF](https://arxiv.org/pdf/2609.26035v1)

**作者:** Sebastian Cochinescu `[一作]` `[通讯]` (University of Bucharest), Sebastian Cochinescu (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在固定的语言模型之上构建并实现了一个行为层，用来表达不确定性、记录来源并对已陈述的信念进行可审计的显式修订。

**💡 创新点**

创新点在于将三态表达策略、来源门控表、持久化信念存储与可审计修订日志组合成一个完整的行为层，并在实验中使用预设边界值进行正式检验。

**🔧 技术方法**

技术手段包括：固定基线模型（Qwen2.5‑0.5B‑Instruct）、两种置信度提取器（均值Token概率与一致性采样）、基于来源的门控表、持久化信念存储与修订日志、以及定量评估协议。

**📊 数据集**

使用了一个由60条事实构成的合成世界以及同样大小的真实地理事实集合，并在其中注入矛盾与错误纠正以实现机械可评分。

**📈 对比分析**

通过对七个配置进行对比实验，评估表达忠实度（ECE/AUC）、矛盾率、审计可追溯性、以及对真实与错误纠正的区分；在真实模型上，审计可追溯性与纠正区分均通过，但表达忠实度的预设边界被打破；在一致性门控配置下，AUC超过0.6且能力等价也满足。

**⚠️ 局限性**

局限性包括：表达忠实度测量受锚点退化影响、模型解码确定性导致矛盾率边界失效、事实库规模过小导致置信区间不稳、实验仅覆盖单一小模型，且未检验对人类信任与可信度的实际影响。

---

## 470. TSS: Target-Side Sparsification for Speculative Decoding in Domain-Specific Large Language Models

**arXiv ID:** 2609.26100 | [PDF](https://arxiv.org/pdf/2609.26100v1)

**作者:** Haibo Hu `[一作]` (City University of Hong Kong), Chun Jason Xue `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TSS框架，通过在目标模型中按域划分跳过部分Transformer层，实现Speculative Decoding的加速与质量提升。

**💡 创新点**

创新点在于：①采用接受率与任务指标双目标的宽度优先搜索，避免传统贪心或单目标选择导致的次优组合；②在推理时使用轻量级域感知跳过控制器，支持同一模型多域稀疏路径，无需重训练或单独裁剪模型。

**🔧 技术方法**

使用的技术包括：Transformer层级跳过（layer skipping）、离线域级校准与配置映射、宽度优先搜索算法、可切换的稀疏执行控制器。

**📊 数据集**

实验基于Spec-Bench的五个任务域：翻译、摘要、问答、RAG以及MMLU，利用每个域的20%校准集确定跳过配置，80%测试集评估效果。

**📈 对比分析**

与原始Speculative Decoding方法（EAGLE、SAMD Token-Recycle）对比，TSS在所有域上实现吞吐量提升1.07×~1.68×，平均接受长度提升约20%，任务指标（BLEU、ROUGE、F1、MMLU准确率）不下降且在翻译与MMLU上均有显著提升。

**⚠️ 局限性**

局限性包括：需为每个域预先进行离线搜索与校准；在未见域上会退回完整路径；跳过配置对特定硬件/模型结构有依赖，且多层跳过可能导致过度精简导致可解释性下降。

---

## 471. The Architect, the Adversary, and the Judge: Closed-Loop Generation of Standards-Aligned Assessment Items at Scale

**arXiv ID:** 2609.26087 | [PDF](https://arxiv.org/pdf/2609.26087v1)

**作者:** Wenhui Chen `[一作]` (University of Macau), Chi Man Vong `[通讯]` (University of Macau)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个闭环的LLM生成与评估管道，在K–12英语语言艺术课程标准下自动生成并通过专家评估器检验的标准化测试题目。

**💡 创新点**

将生成、对抗自评、对照示例以及基于评估反馈挖掘的44,844条标准化错误修正规则相结合，实现了97.8%评估通过率的全课程生成。

**🔧 技术方法**

采用两阶段“生成-再评”协议、少样本对照示例注入、知识字典检索、基于LLM的多维度专家评估器以及对抗式自检技术。

**📊 数据集**

使用755条Common Core ELA标准、3种题型（MCQ、MSQ、填空）、3个难度等级，共生成43,227条已评估项目（9,074条生产实例）。

**📈 对比分析**

通过在相同规则下对10款LLM进行对比，MCQ/MSQ在所有模型上均≥98%通过率，填空题呈能力层级差距；闭环管道在生产跑中达到97.8%通过率，并通过独立评审者验证格式差异。

**⚠️ 局限性**

主要限制包括通过率受单一LLM评估器定义、判定一致性低（κ≈0.13）、缺乏学生答题数据验证真实学习效果，以及开放集填空题的精确性仍停滞在提示优化极限。

---

## 472. Block-Level Weight-Space Structure Persists Under Post-Training: An Empirical Study Across LLM Families

**arXiv ID:** 2609.26147 | [PDF](https://arxiv.org/pdf/2609.26147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 473. Policy-Backed Selective Regeneration under Tainted Inter-Agent Communication

**arXiv ID:** 2609.26072 | [PDF](https://arxiv.org/pdf/2609.26072v1)

**作者:** Jinghan Xu `[一作]` (Nankai University), Hankai Liu `[通讯]` (Nankai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出ESC-CR框架，实现多代理系统中的安全代码生成，通过将消息的内容与授权分离，构建可执行的语义承诺，并在外部释放边界执行效果检查；若检测到违规，则采用清洁室恢复技术在不重用被污染状态的前提下重构任务上下文并重新生成代码。

**💡 创新点**

创新点包括①将信息与授权分离，构造基于任务、证据和策略的可执行承诺；②引入可执行效果检查器EffectIR，覆盖语义层面的效果而非仅文本匹配；③在拒绝后通过污点传播与证据支持的上下文重建，实现安全且信息保留的恢复；④在多模型、多拓扑、适应性攻击下进行系统性评估。

**🔧 技术方法**

技术手段包括可执行语义承诺构造(Build)、效果抽取与检查(EffectIR)、外部释放边界、污点追踪与清洁室重建、基于证据的声明解析与重构、匹配预算的重试控制。

**📊 数据集**

实验使用的主要数据集有：HumanEval、MBPP、通信必需任务集合、间接效果与自适应攻击套件（5,500条程序）以及AgentDojo的受攻击与正常轨迹（2,240+120）。

**📈 对比分析**

与传统的“污染上下文重试”“完整消息丢弃”“清洁任务重试”以及基于语义的释放验证器进行对比。ESC-CR在保持功能正确率的同时，零ASR率，且在通信必需任务中比完整消息丢弃提升了30–70个百分点的安全任务成功率；在AgentDojo评测中实现了1,607/2,240的安全成功率，远高于其他方法。

**⚠️ 局限性**

局限性：恢复效果依赖于模型能力和任务难度；在通信可选任务上有时不如简单重试；依赖可靠的任务证据与外部策略；未针对所有可能的攻击变体；需要额外的实现成本与外部检测器。

---

## 474. StrataVLA: Hierarchical and Efficient 3D Geometric Grounding for Vision-Language-Action Models

**arXiv ID:** 2609.26071 | [PDF](https://arxiv.org/pdf/2609.26071v1)

**作者:** Jin Cui `[一作]` (Xi'an Jiaotong University), Pengju Ren `[通讯]` (Xi'an Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 StrataVLA 框架，在视觉‑语言‑动作（VLA）模型中通过层级 Geometry Adapter（GA）在不同网络深度注入共享的 3D 几何信息，实现了持续、训练稳定的空间感知；

**💡 创新点**

创新点在于：①层级化的几何注入，让不同深度层可以根据自身语义上下文主动检索几何信息；②任务感知路由与 LRU 缓存两种机制显著降低 3D 处理开销；③使用零初始化的残差注入，避免破坏预训练语义；

**🔧 技术方法**

技术包括：冻结的 VGGT 视觉几何基模型、跨模态多头注意力的 GA、任务感知门控网络、GPU LRU 缓存、流匹配（flow‑matching）动作生成；

**📊 数据集**

使用了 LIBERO（四套标准任务）、SimplerEnv（Google Robot 任务）和两套实景厨房/房间机器人操作数据集；

**📈 对比分析**

与多种基线（π₀、OpenVLA-OFT、GeoVLA、ForeAct、RoboVLM 等）对比，StrataVLA 在 LIBERO 的平均成功率达 98.53%，在实景厨房/房间任务中单项任务均高于对手且仅引入约 3.8% 延迟；

**⚠️ 局限性**

局限性在于：依赖固定相机配置（不支持移动摄像头），对极端动态场景缓存效果下降，且 GA 层级选择仍需经验性调优。

---

## 475. Vorch-Human: Unified Multi-Task Human-Centric Generation via Long-Horizon Continuation

**arXiv ID:** 2609.26117 | [PDF](https://arxiv.org/pdf/2609.26117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 476. xWhyL: Causal Interactive Learning

**arXiv ID:** 2609.26037 | [PDF](https://arxiv.org/pdf/2609.26037v1)

**作者:** Nicholas Tagliapietra `[一作]` (Bosch Center for Artificial Intelligence), Kristian Kersting `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出基于解释学习因果模型的框架，并实现交互式因果学习方法 Causal Interactive Learning (CIL)。

**💡 创新点**

创新点在于将解释定义为可用于约束因果结构的学习信号，构建解释等价类和因果拔河理论，证明正确解释可消除马尔可夫等价类对称，且不正确解释可被拒绝。

**🔧 技术方法**

使用可微分结构学习框架、可解释性惩罚函数（总效应、群级反馈），以及软硬约束的正则化得分。

**📊 数据集**

实验数据包括合成非线性结构因果模型（MLP、随机图）以及真实数据 Neuropathic Pain（临床层级结构）。

**📈 对比分析**

与 GES、DAGGER、基于背景知识的基线对比，CIL 在 SHD 上平均下降 5–10%，在 20–50%错误解释时仍优于仅用观测数据的方法，表现出鲁棒性。

**⚠️ 局限性**

局限性在于对因果发现的非凸优化依赖，缺乏高效收敛保证；需要手工设计解释惩罚；在大规模变量时性能下降。

---

## 477. Early Prediction of Pathological Complete Response to Neoadjuvant Chemotherapy Using Temporal Deep Learning on DWI

**arXiv ID:** 2609.26106 | [PDF](https://arxiv.org/pdf/2609.26106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 478. TailSpec-EASE: Knowledge-Graph-Regularized Linear Recommendation for Web Long-Tail Discovery

**arXiv ID:** 2609.26143 | [PDF](https://arxiv.org/pdf/2609.26143v1)

**作者:** Jianru Shen `[一作]` `[通讯]` (University of Montana), Jianru Shen (University of Montana)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 TailSpec‑EASE，一种轻量级线性推荐模型，将关系感知的谱图先验直接注入局部闭式重建目标，从而在不需要全局矩阵求逆的前提下实现高效长尾推荐。

**💡 创新点**

创新点在于：①将知识图谱先验通过逆文档频率加权并归一化成稀疏谱过滤器；②引入流行度自适应门控，使长尾物品获得更强的KG正则化；③在模型层而非分数层注入KG信息，避免全局求逆，兼顾整体准确率与长尾曝光。

**🔧 技术方法**

使用关系感知的稀疏谱图过滤、逆文档频率加权、局部岭回归、流行度门控（γ）、单跳/多跳扩散（K）等技术，并在CPU上实现完全闭式求解。

**📊 数据集**

实验使用四大公开数据集：MovieLens‑1M、Amazon‑book、Last‑FM 和 Yelp2018，均含物品侧知识图谱。

**📈 对比分析**

与传统基线（Pop、ItemKNN、iALS、BPR‑MF）、全局 EASE^R、局部 EASE、LightGCN、KGAT 以及两种分数层KG后处理进行对比；TailSpec‑EASE 在总体 NDCG@20 与 KGAT 相近，且在 Tail Recall@20 上显著提升（最高 24%），训练时间仅 37 s CPU（相较于 KGAT GPU 2,584 s）。

**⚠️ 局限性**

局限性：依赖物品侧知识图谱，若 KG 与交互结构不匹配或稀疏则收益有限；关系权重需在验证集手工选择；模型目前仅覆盖物品侧 KG，未扩展到用户侧或时序/会话推荐。

---

## 479. The Cost of Conservation: Coordination-Memory Laws for Exact-Support Generation

**arXiv ID:** 2609.26126 | [PDF](https://arxiv.org/pdf/2609.26126v1)

**作者:** Zhen Zhang `[一作]` (Technical University of Munich), Amr Alanwar `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在加法守恒约束下，如何在有限前置协调与在线记忆之间权衡，实现精确支持的随机采样；并给出了相应的最优计划选择速率、构造方法、块级并行执行及可编程输出顺序的理论框架；

**💡 创新点**

创新点在于将计划选择速率与守恒差分随机游走的击杀谱学特征联系起来，提出秩控制的协调‑记忆交换定律，揭示局部统计相似性与执行成本的差距，并提供了可在异构分数与非中心预算下实现的精确有限状态计划构造；

**🔧 技术方法**

采用谱分析、信息理论逆论证、对称化与对偶化技巧、Kolmogorov–Rogozin不等式、Bessel函数界限、块级并行理论及随机存取库构造等多种技术；

**📊 数据集**

使用了122,270条 GPT‑4 UltraFeedback 评分数据构造 9‑of‑18 Gibbs 滑动窗口进行有限规模审计；

**📈 对比分析**

通过比较坐标并行计划与单计划剩余计数程序的总变差、计划数量与持久状态位数，发现坐标并行方案需要数千个计划而单计划只需几位持久状态；实验结果验证了理论给出的秩指数与计划选择速率的匹配；

**⚠️ 局限性**

局限性包括：仅对平衡固定字母表目标给出紧致描述，对一般位置特定概率未能给出可压缩描述；在大字母表或极端约束下实用性有限；此外，前置协调成本的最小化仍受架构与实现细节限制。

---

## 480. Differentiable Fuzzy Inference Layer: A Monotone, Compositional Ordinal Reasoning Head for Large Language Models

**arXiv ID:** 2609.26113 | [PDF](https://arxiv.org/pdf/2609.26113v1)

**作者:** Zhen Zhang `[一作]` (Technical University of Munich), Amr Alanwar `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM预测头中引入可微分模糊推理层（DFIL），实现对定量比例的显式表征，从而满足顺序性和可组合性的结构约束；

**💡 创新点**

创新点在于将可微分模糊集理论（有序高斯隶属函数与t-范数）与LLM预测头融合，形成双路径结构：标准分类路径与比例瓶颈路径，既保持单步准确，又能在不需要额外组合训练数据的情况下完成定量表达式的组合推理；

**🔧 技术方法**

技术包括：LoRA微调冻结LLM，使用两层MLP数值头提取比例，学可微分的有序高斯隶属函数，基于最近中心的决策规则实现单调性，t-范数实现组合推理，训练损失融合交叉熵与比例正则项；

**📊 数据集**

主要使用FRoG数据集（量词推理）以及QURe、SST-5、STS22等自然语言标注数据，测试时还采集了GenericsKB-Best、PubMedQA、CNN/DailyMail等自然语料；

**📈 对比分析**

与现有基线（Fine‑Tune分类器、FT+Compose、CORAL、CORN等）在单步准确性上保持1pp以内的相当水平，同时在组合推理、样本效率、跨任务迁移和单调一致性等指标上显著提升（组合推理提升约+5.9pp，低样本下提升+11~16pp，跨任务一致性提升约+5.5pp），证明结构化预测头带来的实质性优势；

**⚠️ 局限性**

局限包括：比例提取仍受LLM文本表征的限制，导致隐式比例推理（IPR）表现为随机；主分类路径在单调性上仍有约10%违背；MF分支对训练初始化敏感，某些规模下出现双峰收敛；整体依赖于冻结的LoRA微调，未在极大规模4-bit量化下做完整消融；

---

## 481. MIAR: Medical Image Super-Resolution With Autoregressive Modeling

**arXiv ID:** 2609.26103 | [PDF](https://arxiv.org/pdf/2609.26103v1)

**作者:** Fang Li `[一作]` (Beihang University), Aimin Hao `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种多尺度自回归框架MIAR，用于医学图像超分辨率，结合MSVQ‑VAE、尺度自适应结构解码器和层级束搜索；

**💡 创新点**

将自回归模型与多尺度量化、结构先验融合，采用旋转位置编码和层级束搜索以缓解递归误差，并在不牺牲结构保真度的前提下提升感知质量；

**🔧 技术方法**

多尺度向量量化变分自编码器、旋转位置编码（RoPE）、跨尺度注意力解码、层级束搜索；

**📊 数据集**

BraTS2021（MRI）与LIDC‑IDRI（CT）数据集；

**📈 对比分析**

与多种SOTA方法（包括多模态扩散、单模态回归、VAR等）比较，MIAR在PSNR、SSIM、LPIPS、DISTS、MUSIQ和MANIQA等指标上均优于对手，并且相较扩散模型实现了约2倍的推理速度提升；

**⚠️ 局限性**

推理时间仍高于纯回归模型，层级束搜索在宽度增大时会显著增加计算成本，并且在极端高分辨率或复杂病变场景下可能存在微小的误差累积问题。

---

## 482. Moving6DPoSe: A Multimodal Database for Monocular 6D Pose Estimation and Segmentation of Moving Objects

**arXiv ID:** 2609.26161 | [PDF](https://arxiv.org/pdf/2609.26161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 483. Magnitude Profile Pruning: Calibration-Free Structured Attention Head Removal for Transformer Compression

**arXiv ID:** 2609.26177 | [PDF](https://arxiv.org/pdf/2609.26177v1)

**作者:** Kasun Dewage `[一作]` (University of Central Florida), Suranadi De Silva `[通讯]` (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种仅基于 Transformer 权重的无校准结构化注意力头剪枝方法 Magnitude Profile Pruning，并在多种编码器/解码器模型上进行了评估。

**💡 创新点**

创新点在于使用统计异常检测对注意力投影行/列范数进行评分，构建 MP 与 GQA 兼容的 MP‑G 两种分数公式，完全不依赖数据、前向/后向传递或 Hessian 估计即可决定剪枝。

**🔧 技术方法**

核心技术包括：统计阈值 μ+zσ、异常检测、Grouped Query Attention（GQA）处理、全局头排名与零化操作，以及对权重范数的高效一次性遍历。

**📊 数据集**

主要实验使用 WikiText‑2 进行语言建模困惑度评估，RoBERTa‑large 在 GLUE（SST‑2、MRPC、RTE）任务上进一步验证下游性能。

**📈 对比分析**

与 Wanda‑Head、SparseGPT‑Head、Gradient‑Head 以及 Hybrid 组合方法对比，MP‑G 在 OPT‑6.7B、RoBERTa‑large 等模型上取得最佳或次佳困惑度，且在 12.5%–50% 头稀疏比例下表现稳定，表明其在无校准场景下的竞争力。

**⚠️ 局限性**

局限性包括：对权重范数分布的依赖导致在某些 GQA 或较小模型中效果不如梯度/激活基准；缺乏对模型结构切除或稀疏内核的支持，导致理论 FLOP 节省尚未转化为实际速度提升。

---

## 484. Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity

**arXiv ID:** 2609.26173 | [PDF](https://arxiv.org/pdf/2609.26173v1)

**作者:** Kasun Dewage `[一作]` (University of Central Florida), Suranadi De Silva `[通讯]` (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了在九种开源大型语言模型中，单个注意力投影（Q、K、V、O）逐层单独量化的感知敏感性，并比较了相对重构误差、激活加权量化误差与PPL变化之间的关系。

**💡 创新点**

证明了相对权重量化误差在单个投影层中对功能敏感度的预测能力弱，揭示V投影在大多数模型中是主导敏感性组件，并提出激活加权误差是更有效的预测指标。

**🔧 技术方法**

使用RTN和GPTQ两种后训练量化方法，对每个模型的每层注意力投影进行单独量化，记录重构误差、激活加权误差与PPL变化。

**📊 数据集**

使用WikiText-2评估PPL（16,384 tokens），C4用于GPTQ校准（2,048 tokens）。

**📈 对比分析**

通过统计R²和η²等指标比较重构误差、层级和组件类型对ΔPPL的解释度；结果显示重构误差R²<0.1占75%，V投影贡献最高，激活加权误差对V投影的R²提升至约0.2。

**⚠️ 局限性**

仅评估了RTN和GPTQ两种量化方法，使用WikiText-2 PPL作为功能指标，未检验下游任务准确率；且仅采用对角激活二阶矩，未使用完整Hessian。

---

## 485. When Concealed Links Cannot Be Recovered: A Structural Identifiability Bound and Evaluation Pitfalls in Offshore Leak Networks

**arXiv ID:** 2609.26171 | [PDF](https://arxiv.org/pdf/2609.26171v1)

**作者:** Joseph Bingham `[一作]` `[通讯]` (Technion Israel Institute of Technology), Joseph Bingham (Technion Israel Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对离岸金融泄露网络中隐藏的受益所有权关系进行理论与实证研究，提出结构可识别的上限，并在多种预测方法上验证该界限。

**💡 创新点**

①证明任何保持图同构性的恢复规则在孤立节点上不可超越随机；②将可识别界限与节点的结构不可区分类（通过 Weisfeiler–Leman 颜色类近似）关联；③系统归纳五种常见评估陷阱并给出修正规则；④在完整的 ICIJ 泄露图和其他四个泄露图上进行一致性验证。

**🔧 技术方法**

理论证明、结构不变的恢复规则、WL 颜色精炼、基于图的邻接评分（如 Adamic–Adar）、SEAL 高阶子图特征、SIGN/SGC 风格的 GNN、faithful evaluation（同种族、vary‑owner）配对测试、bootstrap 置信区间。

**📊 数据集**

ICIJ Offshore Leaks（814,344 节点、3.3M 条边、84,172 条受益所有权边），以及 Panama、Paradise、Pandora、Bahamas 等五个独立泄露子集。

**📈 对比分析**

通过在不同度数、WL 类大小下的 AUC 评估对比：所有方法在度为 0 的孤立节点上均得到 0.5 的 AUC；度 > 0 时 AUC 随可达性和结构区分度提升，但始终低于 1，且所有模型的提升幅度仅在 0.02 左右；进一步验证显示更高表达的子图模型也无法突破 0.5 限制，说明该界限稳健。

**⚠️ 局限性**

局限性：仅讨论了结构通道的可识别性，属性通道虽然能略微突破（约 7%），但整体提升有限；外部数据覆盖度极低；孤立节点比例高导致大部分受益所有权不可恢复；实验仅基于 ICIJ 数据，未覆盖更广泛的泄露场景；未深入评估更强表达的 k‑WL 或完整 DGCNN‑SEAL 但理论上也不能突破 0.5 限制。

---

## 486. The Free-Recipe Limit: Every Recipe Effect Measures Which Premise of an Idealised Learner Broke

**arXiv ID:** 2609.26160 | [PDF](https://arxiv.org/pdf/2609.26160v1)

**作者:** Wenhui Chen `[一作]` (University of Macau), Chi Man Vong `[通讯]` (University of Macau)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在单一数学技能领域内，系统性探索了训练顺序、排列与组合对最终性能的影响，提出了“recipe‑search wall”概念并对其进行量化。

**💡 创新点**

创新点在于：①用可度量的“可达集合直径”与“分辨率”来判定搜索有效性；②发现排列并非自由，阻塞式训练导致“遗忘”并转移模型对争议语境的分配；③揭示顺序效应为瞬态，且在不同预算下会改变符号；④在同一数据集下对比阻塞、交错与组合，量化其对多项式数学任务的贡献。

**🔧 技术方法**

技术包括：全参数监督微调（AdamW），k‑max 4精确匹配评估，固定训练预算与种子，构建多达 761 次微调实验，使用五种训练“arm”（仅A、仅B、A→B、B→A、混合），并对每种 arm 在两种技能上进行评估，提取 V（组合）、D（排列）与 order 对比；利用噪声估计、信度分析与多种控制实验来验证结果。

**📊 数据集**

数据集为 OpenR1‑Math（单一数学问题集合），涵盖 5 个技能（代数、微积分、几何、组合数学、物理），每技能约 600 行样本；实验在 12 个非指令基线模型（Qwen2.5 0.5/1.5/3/7/14B，Qwen3 1.7/4/8/14B，Llama‑3 1/3/8B）上进行，使用 3 个不同预训练家族。

**📈 对比分析**

比较方法为：在固定数据量、预算与评估标准（exact‑match k=4）下，计算每对技能的 V、D 与 order 对比；使用三种种子平均、标准误与 80% 检验阈值 2.8×σc 判定显著性。结果显示：排列在同质数据下无显著差异，阻塞导致约 76% 的效应来自首个技能的遗忘；交错训练提升总能力 +0.0483，且在 6/6 对中显著；在存在不一致语境（同一问题的答案写法不同）时，排列转变为对分配的影响，显著超过 2 个数量级。

**⚠️ 局限性**

局限性：①研究仅覆盖数学单项任务，缺乏跨任务/跨语言验证；②使用精确匹配评估，对更灵活输出的模型可能不适用；③顺序与排列效应高度噪声，需多次复现才能确认；④“recipe‑search wall”依赖于固定预算与数据量，超出此范围的模型或更大训练集可能表现不同；⑤未探究模型内部表示或梯度机制的细节，仅给出经验量化。

---

## 487. Probabilistic Physics-Informed Neural Solvers for Woods-Saxon Parameter Identification: A Coupled Forward-Inverse Approach

**arXiv ID:** 2609.26169 | [PDF](https://arxiv.org/pdf/2609.26169v1)

**作者:** Iraklis Spyrou `[一作]` (INSANE Group IIT NCSR Demokritos), Christoforos Rekatsinas `[通讯]` (INSANE Group IIT NCSR Demokritos)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种概率物理信息神经网络框架，利用单粒子能谱反演全局Woods–Saxon势参数，并同时生成满足薛定谔方程的波函数。

**💡 创新点**

创新点在于将WaveNet与ParamNet耦合，利用输出空间变分推断获取参数分布，且在单个可微模型中集成能量一致性、方程残差、正则化等多种物理约束，实现不需要显式先验知识的参数不确定性估计。

**🔧 技术方法**

技术手段包括物理信息神经网络、变分推断（输出空间），自动微分、Rayleigh商计算、随机抽样的残差约束、正则化以及与有限差分求解器的闭合测试。

**📊 数据集**

数据集主要由两类合成谱（Seminole、Wahlborn）和实验单粒子谱（^40Ca、^48Ca、^132Sn、^208Pb等共42级）构成，合成谱用于验证，实验谱用于真实反演。

**📈 对比分析**

方法通过与传统最小二乘拟合和参考参数化比较，合成测试中参数误差<0.6%，实验测试中MAE约0.8 MeV，性能与基线相当或略优，且仅使用原来数据量的约51%。

**⚠️ 局限性**

局限性包括实验数据稀缺、质子与中子表现差异未彻底解释、输出分布不具备正式贝叶斯置信区间、仅限于球面有限状态，未来需扩展至非球面、非局域或能量相关势。

---

## 488. Designing Task-Induced Arousal: A Multimodal Stress Induction Method for Interactive Experiments

**arXiv ID:** 2609.26156 | [PDF](https://arxiv.org/pdf/2609.26156v1)

**作者:** Morten Roed Frederiksen `[一作]` `[通讯]` (IT-University of Copenhagen), Morten Roed Frederiksen (IT-University of Copenhagen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在交互实验中，设计并验证了一种基于浏览器的任务诱发方法，结合时间压力、视觉与听觉紧迫提示以及并行的“热线”物理挑战，来诱发短时任务相关的激活状态。

**💡 创新点**

创新点在于：①将浏览器端的可视计时任务与物理手持设备的感应结合，形成可重复、可嵌入现有交互系统的复合诱发流程；②使用轻量级单文件网页应用实现可调节的时间递增、视觉与声音升级，开放源码提高可复现性；③将握力动态（变异性与释放速度）作为行为验证指标，补充传统负荷与生理测量。

**🔧 技术方法**

技术包括：HTML/CSS/JavaScript+Canvas API实现浏览器任务；手持触摸设备与内置GSR传感器；实验设计采用A–B–A重复测量；自评量表（NASA‑TLX、SAM）、GSR峰值检测与握力动态分析。

**📊 数据集**

使用了22名受试者（10–44岁）收集的自评问卷、GSR信号和握力压力数据，形成了包含三段（A1、B、A2）时间序列的数据集。

**📈 对比分析**

与安静基线相比，B 条件在主观负荷、SAM 激活、GSR 峰率以及握力变异性/释放速度等三大验证维度均表现显著提升（p < .001），表明该方法能有效诱发高值挑战型激活。

**⚠️ 局限性**

局限包括：①样本量小且年龄分布广，导致统计力量与普适性受限；②缺乏心率/心率变异等更全面的自主神经测量；③物理“热线”任务需手工搭建，实验复现度与标准化受到限制；④握力指标受手部差异影响，可能不适用于不同交互设备。

---

## 489. Zeta-Transform Evaluation for Higher-Order Vanishing Key Recovery

**arXiv ID:** 2609.26132 | [PDF](https://arxiv.org/pdf/2609.26132v1)

**作者:** Sunyeop Kim `[一作]` (Korea University), Insung Kim `[通讯]` (Korea University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于截断加权上Zeta变换的高阶消失映射评估方法，显著降低了Classic McEliece密钥恢复攻击中Wiedemann算法的核心核计算成本。

**💡 创新点**

创新点在于：①仅在Boolean格的从最高层p到所需最低层L的带宽内完成加权Zeta变换；②通过蝴蝶操作在该带宽内完成完整的评估，从而无需遍历全部层级；③实现了映射及其转置的精确评估，减少了重复矩阵-向量乘法次数。

**🔧 技术方法**

主要技术包括：加权上Zeta变换、蝴蝶操作（butterfly operation）、快速Zeta变换算法、截断评估策略以及Wiedemann算法用于求解线性核。

**📊 数据集**

实验使用了Classic McEliece的五组参数集（如(3488,64,12)、(4608,96,13)等），并根据不同的截短参数p和s进行评估。

**📈 对比分析**

与原始稀疏矩阵乘法估算方法相比，本文在两种模型（全1模型和Bernoulli(1/2)模型）下分别提升了约14.09–41.19比特和7.25–22.48比特的Wiedemann工作量（以log₂计）。

**⚠️ 局限性**

局限性：①改进仅针对高阶消失映射的核计算，整体攻击流程仍保持不变；②收益高度依赖非主元列的权重分布，且假设列元素独立；③方法专为Classic McEliece设计，推广到其他码类或攻击策略可能受限；④截断Zeta变换的实现复杂度及内存占用在极大规模参数下未被充分评估。

---

## 490. StepTrigger: Contact-State-Triggered Backdoor Attacks on VLM-Powered Legged Robots

**arXiv ID:** 2609.26131 | [PDF](https://arxiv.org/pdf/2609.26131v1)

**作者:** Jiageng Zhang `[一作]` (Michigan Technological University), Kaichen Yang `[通讯]` (Michigan Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种利用步态与地面接触状态触发的后门攻击，针对 VLM 驱动的四足机器人；

**💡 创新点**

创新点在于将触发器转为机器人自身的接触压力模式，突破了传统文本、视觉或动作历史触发的局限；

**🔧 技术方法**

使用了基于 Qwen3-VL 的 VLM 策略器、Gazebo 机器人仿真、压力传感器的多模态输入与两层触发候选机制；

**📊 数据集**

数据集由三阶段收集的 2,686 条样本构成，包括干净样本、误触硬负样本与真正触发样本，并通过自动标注实现选择性训练；

**📈 对比分析**

在 240 条离线评估样本上，模型在干净样本上 98.75% 维持正常行为，误触样本 92.50% 拒绝触发，真正触发样本 76.25% 成功激活，整体准确率 89.17%；

**⚠️ 局限性**

局限包括仅在仿真环境下验证、未评估完整的闭环轨迹成功率、对不同地形、动态障碍物的鲁棒性不足，且真实硬件验证尚未完成。

---

## 491. When Big Data Becomes a Curse: Spatial Heterogeneity and the Limits of Learning from Passive Acoustic Monitoring Data

**arXiv ID:** 2609.26125 | [PDF](https://arxiv.org/pdf/2609.26125v1)

**作者:** Gabriel Spadon `[一作]` (Dalhousie University), Priyanka Aravindan `[通讯]` (Dalhousie University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对来自加拿大大西洋的908,072条679秒PAM录音进行空间异质性分析，评估站点识别、声学模型识别和船舶检测的转移能力，使用多种验证方案（随机窗口、未见站点、季节交叉、地区留一）进行实验。

**💡 创新点**

提出空间依赖的验证框架并量化了不同空间域间的混淆度，证明随机窗口验证会高估未见站点性能，并阐述空间专家模型与区域适用性门控的概念。

**🔧 技术方法**

使用随机森林、梯度提升、逻辑回归等传统机器学习模型；特征为能量、谱平坦度、熵、峰均比、能量方差；标签通过AIS距离-时间匹配生成；采用bootstrap、对偶比较、聚类抽样等统计方法。

**📊 数据集**

加拿大大西洋与圣劳伦斯湾PAM数据集（38次部署、20个站点ID、21个接收机位置），共908,072条679秒录音，配合加拿大海岸警卫队AIS轨迹生成船舶接近标签。

**📈 对比分析**

比较站点识别（BAcc≈16%）、声学模型识别（BAcc≈74–86%）以及船舶检测（ROC-AUC、F1、AP）。随机窗口AUC为0.661，未见站点AUC为0.612，差值0.049；其他验证方案的性能与此相近。

**⚠️ 局限性**

限制：随机窗口验证不反映实际部署转移；AIS标签不等同于声学可听性；数据高度受地区与季节影响，缺乏跨站点因果推断；仅使用低维特征，未尝试深度表示；结果可能不具普适性。

---

## 492. Neoadjuvant chemotherapy response prediction using pretreatment diffusion and contrast-enhanced magnetic resonance imaging with clinical variables

**arXiv ID:** 2609.26105 | [PDF](https://arxiv.org/pdf/2609.26105v1)

**作者:** Pablo García Marcos `[一作]` (University of Oviedo), Víctor M. González `[通讯]` (University of Oviedo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于预处理T0 MRI（ADC和DCE-MRI）和HR/HER2临床变量的深度学习模型，用以预测乳腺癌新辅助化疗的病理完全缓解（pCR）.

**💡 创新点**

创新点在于仅使用预处理时间点的数据，采用多模态（ADC+DCE-MRI）并结合单一临床变量的轻量级网络架构，避免了多序列或手工特征提取的复杂性，并实现了与更复杂方法相当的AUC。

**🔧 技术方法**

使用EfficientNet-B0预训练编码器进行图像特征提取，Late fusion + 线性层融合影像和临床特征，采用AdamW、交叉熵权重、数据增强、早停等技术。

**📊 数据集**

使用公开的ACRIN 6698/I‑SPY2 多中心数据集，共136例（46例pCR）。

**📈 对比分析**

通过5折交叉验证与单模态、双模态以及加入HR/HER2的配置比较；单模态ADC AUC≈0.79，DCE‑MRI≈0.74，加入HR/HER2后分别提升到0.83/0.81，最终组合ADC+DCE‑MRI+HR/HER2 AUC≈0.86，接近或优于已发表的仅预处理方法。

**⚠️ 局限性**

局限包括样本量有限（尤其正样本）、缺乏外部验证、需要重建的DCE‑MRI肿瘤分割、对DCE‑MRI的钆剂依赖以及模型对分割误差的敏感性。

---

## 493. A Large-Scale Longitudinal Study of Multi-CI Service Adoption

**arXiv ID:** 2609.26181 | [PDF](https://arxiv.org/pdf/2609.26181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 494. Joint Quantized Precoding and Bit Allocation for Fronthaul-Constrained Cell-Free Massive MIMO

**arXiv ID:** 2609.26149 | [PDF](https://arxiv.org/pdf/2609.26149v1)

**作者:** Özlem Tuğfe Demir `[一作]` `[通讯]` (Bilkent University), Özlem Tuğfe Demir (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出一种针对分布式大规模MIMO下行链路、在有限分组链路容量下进行量化感知前向编码和比特分配的联合设计方案。

**💡 创新点**

创新点在于将前向编码与分组链路量化畸变耦合建模，利用Bussgang分解得到端到端的信号模型，并在此基础上提出联合前向编码与比特分配优化，采用块坐标下降实现可行的迭代求解，显著提升在分组链路受限环境下的谱效率。

**🔧 技术方法**

使用Bussgang分解、量化感知前向编码、比特分配、块坐标下降（BCD）算法、线性代数求解和梯度/闭式更新等技术。

**📊 数据集**

仿真数据：在500 m×500 m正方形区域随机部署L个AP（每个4个天线）和K个UE（单天线），Rayleigh小波通道加距离相关的大尺度衰落，10 m竖向距离，10个随机场景，每个场景200个独立通道样例。

**📈 对比分析**

与两种基准方案对比：①均匀比特分配+同样的迭代前向编码；②均匀比特分配+传统正则化零迫（RZF）前向编码+量化。结果显示，在低到中等分组链路预算下，所提自适应比特分配方案平均提升10–20%的总谱效率，并在AP数量增加时保持性能优势。

**⚠️ 局限性**

主要限制包括：对空间相关量化噪声采用对角近似；使用高分辨率近似的量化损失模型；假设完全CSI、仅下行链路、仅考虑标量Lloyd–Max量化；算法的收敛速度与计算复杂度在大规模网络上尚未充分验证。

---

## 495. Removing the Competition from Introductory Competitive Programming: A Backward-Designed, Mastery-Oriented Curriculum Sequence

**arXiv ID:** 2609.26139 | [PDF](https://arxiv.org/pdf/2609.26139v1)

**作者:** Ethan Dickey `[一作]` `[通讯]` (Purdue University), Ethan Dickey (Purdue University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新设计了三门竞赛编程课程（CP1、CP2、CP3），先通过掌握式在线评测、口述评估、广度门槛等方法培养算法问题解决能力，随后在CP3中才引入正式竞赛以训练速度、策略和团队协作。

**💡 创新点**

创新点在于：①将竞赛与课程目标分层，先培养基础能力后再引入竞赛；②提出双层能力模型（算法问题解决 vs 竞赛表现）；③构建完整的对齐课程架构（内容、评估、教学活动一致）；④提供可复现的评估方案和实施工具，方便其他教育者快速采用。

**🔧 技术方法**

使用技术包括：在线评测平台（Kattis 等）进行高频自动评分；采样口头代码访谈验证作者理解；清洁代码评分鼓励可读性；限定生成式 AI 使用周的实验；两次 75 分钟会议的三段式教学（概念、思考-配对-分享、配对完整实现）；以及自动化的“说话时间诊断”与“广度门槛”等教学工具。

**📊 数据集**

未使用传统公开数据集，而是采用在线评测平台上的真实竞赛题目以及课程自定义的题库，保证题目具有明确的正确性、时间与空间约束。

**📈 对比分析**

通过对比课程前后学生在未计时的转移任务、口头访谈、广度门槛完成率以及在 CP3 竞赛中的解决数量、首次接收时间、团队协作与回顾等指标来评估学习效果。论文提出评估框架，尚未给出具体数值或性能对比结果，主要强调评估方法的设计而非具体性能数值。

**⚠️ 局限性**

主要局限：未收集学生成绩或对照组数据，缺乏因果效应证据；评估基于单一机构和课程规模，缺乏外部验证；生成式 AI 的实验仍处于探索阶段；课程设计高度依赖在线评测平台与大量教学助理支持，难以直接迁移到资源受限的环境。

---

## 496. DTOC: Dynamic Tool Output Compression for Adaptive Context Management in AI Agents

**arXiv ID:** 2609.26121 | [PDF](https://arxiv.org/pdf/2609.26121v1)

**作者:** Abhay Chaturvedi `[一作]` (Pegasystems), Peter van der Putten `[通讯]` (Pegasystems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Dynamic Tool Output Compression（DTOC）框架，在大型语言模型代理中通过将工具输出外部存储并用占位符可逆地压缩/解压，显著降低上下文窗口占用并提升任务成功率。

**💡 创新点**

创新点在于：①将上下文压缩视为代理可执行的显式操作；②通过外部存储+可逆占位符实现完整信息的保留与按需恢复；③在 ReAct 代理结构中提供统一、模型无关的压缩工具。

**🔧 技术方法**

使用技术包括：ReAct 代理架构、可逆上下文管理工具、占位符机制、外部键值存储、token 估算与可视化反馈、OpenCode 实现平台。

**📊 数据集**

数据集主要是 DeepSWE（6 个 Python/TypeScript 软件功能实现任务），并在小规模专有代码集上做消融实验。

**📈 对比分析**

通过在 5 个前沿模型（Sonnet、Opus、GPT‑5.4/5.5、Gemini Flash）下对比 DTOC ON/OFF，评估 solve rate、输入 token、步骤数、成本；在 Sonnet、GPT‑5.4/5.5 等模型中，solve rate 提升至 50%，成本降低 67‑72%，而在 Opus、Gemini 上影响有限或略显负面。

**⚠️ 局限性**

局限性包括：仅有 6 个任务且单次跑，缺乏统计显著性；使用通用 manage_context 提示可能不适配所有模型；未与其他压缩方法（如 LLMLingua、截断、摘要）直接对比；未针对模型特性自适应调优压缩策略。

---

## 497. An Exact Counterexample to Affine-Like Price-of-Anarchy Shape in Quartic BPR Routing

**arXiv ID:** 2609.26179 | [PDF](https://arxiv.org/pdf/2609.26179v1)

**作者:** Ian D'Ambrosio `[一作]` `[通讯]` (Nth Research Collective), Ian D'Ambrosio (Nth Research Collective)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了一个五顶点、六条边的有理网络，使用常见四次多项式BPR成本，在该网络上给出了一个精确计算机辅助的反例，证明在此成本形式下Price of Anarchy（PoA）可以在需求区间内出现内部极大点。

**💡 创新点**

创新点在于：①首次给出针对常见四次多项式成本的精确、可检验的最小规模反例；②利用序列边机制解析说明单个四次边如何在不改变路由分配的前提下改变PoA的导数；③通过区间算术与Krawczyk包含定理实现了严格的符号比较。

**🔧 技术方法**

技术方法包括：精确有理区间算术、Beckmann潜能与KKT条件求解、两维Krawczyk包含定理判定唯一根、符号比较与外推的严格证书验证、以及Python的exact arithmetic库实现自检。

**📊 数据集**

仅使用了论文中给出的自定义有理系数数据，没有引入外部真实网络或数据集，网络规模为5个节点、6条边。

**📈 对比分析**

通过在需求点17、21、24处对PoA进行区间求值，并利用严格的符号比较证书证明PoA在21处严格大于两端点的值；计算量极小，整个验证过程在几秒内完成。

**⚠️ 局限性**

局限性包括：①示例仅针对常见四次多项式成本，未说明其他多项式或线性成本下的形状；②未给出并行或系列并联网络的通用结论；③参数选择经过人为调优，未证明最小规模或系数的最优性。

---

## 498. Activation-Energy Pruning for Spiking Neural Networks: Unsupervised Personalization via Spike-Count Saliency

**arXiv ID:** 2609.26167 | [PDF](https://arxiv.org/pdf/2609.26167v1)

**作者:** Joseph Bingham `[一作]` `[通讯]` (Technion Israel Institute of Technology), Joseph Bingham (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对脉冲神经网络（SNN）使用激活能量阈值剪枝，探究其在无监督个性化中的表现，并与梯度剪枝方法比较。

**💡 创新点**

①发现梯度剪枝在SNN中性能急剧下降；②激活能量剪枝在高稀疏度下可提升准确率（尤其在N‑MNIST 3类子任务上）；③BN重校准在不同稀疏度下具有交叉点。

**🔧 技术方法**

利用激活能量（权重幅值 × 预突触峰值计数）作为重要性度量，SNN一轮前向推断即可计算；同时实现无监督BatchNorm重校准与全局/层级剪枝策略。

**📊 数据集**

主要使用CIFAR‑100（VGG‑SNN）与N‑MNIST（SCNN）两个基准数据集；对Tiny‑ImageNet做了初步试验。

**📈 对比分析**

与随机、Magnitude、SNIP、GraSP等基线对比，FP‑SNN在σ≤0.5时保持≈65%准确率，而梯度基准在σ≈0.2即跌至随机；在N‑MNIST上FP‑SNN在σ=0.8时可提升至98.4%，超越原始模型。

**⚠️ 局限性**

受限于Tiny‑ImageNet源模型欠训练、缺乏后剪枝微调、BN重校准交叉点仅在特定配置下验证、仅测试VGG/SCNN两种SNN架构，未覆盖循环或Transformer SNN。

---

## 499. Refusal without Discrimination: What Encoded Prompts Do to Safety-Trained Models

**arXiv ID:** 2609.26176 | [PDF](https://arxiv.org/pdf/2609.26176v1)

**作者:** Haoyu Zhang `[一作]` (Northeastern University), Shanu Sushmita `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了编码提示攻击评估中仅使用有害请求拒绝率的局限性，提出必须加入良性请求和明文基准以衡量拒绝差距。

**💡 创新点**

创新点在于揭示编码导致有害请求拒绝率聚集、使模型无法区分，说明现行评估指标偏差，并验证完整安全训练流水线并未修复该误差。

**🔧 技术方法**

技术方法包括四个开源指令微调模型、JailbreakBench 主题匹配的有害/良性提示、同形字（homoglyph）编码、Bootstrap 噪声基准、线性读出解码与有害识别的判别器以及 LLM 判别器评估拒绝。

**📊 数据集**

使用的数据集为 JailbreakBench 的有害提示及其主题匹配的良性对照，取每种条件 100 条样本进行评估。

**📈 对比分析**

比较方法通过计算四模型在四种组合（有害/良性 × 明文/编码）下的拒绝率方差并与 Bootstrap 噪声对照，发现有害侧差距仅 0.08、明文侧差距 0.57；完整安全流水线提升明文区分率 +0.25，但编码侧误差不变，标准“有害侧拒绝”指标甚至误导评估模型更不安全。

**⚠️ 局限性**

局限性包括仅评估四个 7–8B 开源模型、样本量有限（每条件 100），仅测试 homoglyph 及少数其他编码，未提供内部机制解释，依赖 LLM 判别器且易受 echo 与误判影响，并未检验更大规模模型或更广泛编码形式。

---

## 500. EADC: Evaluation of Advanced and Deep-level Compliance in Large Language Models

**arXiv ID:** 2609.26175 | [PDF](https://arxiv.org/pdf/2609.26175v1)

**作者:** Yan Zhang `[一作]` (Tsinghua University), Guangwen Yang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了EADC——一种基于AI合规知识图谱与人工法律专家循环评估的LLM深度合规基准；

**💡 创新点**

创新点在于将法律法规形式化为多关系知识图谱，构造三维合规风险维度（隐性/逻辑/情境），并通过人机协同生成与审核实现高质量、多维度QA；

**🔧 技术方法**

采用知识图谱构建、规则化逻辑推理、LLM驱动自动化生成agent、检索增强生成、人工专家审核与迭代、基于must‑include/must‑not评估指标等技术；

**📊 数据集**

使用自研EADC数据集（4,435+ QA对）及其2%子集EADC-lite，覆盖13条AI法规，涵盖隐性/逻辑/情境合规风险与四大指标；

**📈 对比分析**

对24款LLM（封闭源与开源）进行评估，发现GPT5.5、Claude Sonnet 5等封闭源模型可部分超过人类基准，开源模型总体低于人类，整体表现随参数规模提升但仍显不足；

**⚠️ 局限性**

局限性包括需人工审核成本高、法规更新滞后、尚未覆盖所有隐蔽合规风险、模型对法律语义理解仍有限。

---

## 501. TRACE: Transparent Retrieval for Abstract Concept Evaluation

**arXiv ID:** 2609.26168 | [PDF](https://arxiv.org/pdf/2609.26168v1)

**作者:** Joseph Bingham `[一作]` `[通讯]` (Technion University), Joseph Bingham (Technion University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在单句抽象参照游戏中，作者比较了六种预训练视觉‑语言模型与基于SIFT和UQI的透明检索‑匹配基线，发现后者与最强模型SigLIP‑large几乎同等，且优于其余模型。

**💡 创新点**

创新点在于证明在此任务中不需要学习视觉表征，采用可解释的检索+经典形状相似度即可匹配大模型的性能，并对不同VLM的抽象归纳能力进行诊断。

**🔧 技术方法**

技术上使用Bing图像检索、spaCy词性过滤、SIFT单应性匹配、Universal Quality Index以及CLIP/SigLIP等预训练模型。

**📊 数据集**

数据集为Stanford Repeated Reference Game的12个唐格子（991个第一次重复试验）以及KiloGram的1013形状扩展。

**📈 对比分析**

比较采用同一试验集的top‑1、top‑3、top‑5准确率，透明基线TRACE在top‑1 44.3%与SigLIP‑large 43.0%相当，均显著高于CLIP和OpenCLIP。

**⚠️ 局限性**

局限在于仅评估单回合无交互的抽象描述；基线依赖商业检索API，无法完全消除外部图像优势；并未解决多轮契约维护问题。

---

## 502. Toward User-Mediated Self-Repair in Ubiquitous Robots Through Goal-Oriented Agentic AI

**arXiv ID:** 2609.26157 | [PDF](https://arxiv.org/pdf/2609.26157v1)

**作者:** Morten Roed Frederiksen `[一作]` `[通讯]` (IT-University of Copenhagen), Morten Roed Frederiksen (IT-University of Copenhagen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一套面向用户的目标导向代理 AI 架构，用于在无屏交互环境下帮助用户自行维护和修复可穿戴机器人。

**💡 创新点**

创新点在于多层代理结构：将目标分解、策略规划、会话执行与持久化分离，并通过自然语言对话实现开放式、非线性的自我修复流程；同时引入拟人化界面提升用户合作感。

**🔧 技术方法**

采用大型语言模型 Llama 3.1 70B 进行目标分解，Qwen 30B 负责实时对话，Whisper 进行语音识别，ChromaDB 存储持久化状态，Pygame 搭建可穿戴机器人交互界面。

**📊 数据集**

未使用公开数据集；实验使用自制硬件修复试验台，并通过参与者的自然语言输入作为 LLM 的输入。

**📈 对比分析**

通过与之前在线实验（同一架构但虚拟任务、70 人）对比，测量任务完成率、对话轮数、信任、社交存在感等指标；物理实验完成率 95%（略低于在线的 100%），对话轮数增加，社交存在感和信任显著下降，但系统帮助度更高。

**⚠️ 局限性**

局限性包括样本量仅 20 人、仅后测自我效能、任务仅为条件分支的线性线缆拼装、未评估更复杂故障、缺乏前后测比较，且无法验证长期维护效果。

---

## 503. Toward Self-Repairing Ubiquitous Robots Using Goal-Oriented Agentic AI in Human-Robot Interactions

**arXiv ID:** 2609.26155 | [PDF](https://arxiv.org/pdf/2609.26155v1)

**作者:** Morten Roed Frederiksen `[一作]` `[通讯]` (IT University of Copenhagen), Morten Roed Frederiksen (IT University of Copenhagen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究开发了一种面向非专家用户的目标导向代理式AI架构，支持在无视觉界面的可穿戴机器人中通过自然语言对话完成物理硬件维修任务。

**💡 创新点**

创新点在于将任务拆解、持久化状态、战略级目标管理与实时对话执行四个功能模块分离，使得机器人既能保持对维修状态的持续追踪，又能在对话中灵活应对澄清、离题和语言转换。

**🔧 技术方法**

采用了大型语言模型（Llama 3.1 70B用于目标拆解，Qwen 30B用于对话生成）、ChromaDB用于持久化存储、Whisper+文本转语音实现语音交互，并结合面部检测与语音阈值控制用户参与。

**📊 数据集**

实验使用自制的物理硬件测试板（木盒、绿色状态灯、10根带鳄鱼夹的编号电线以及蓝/绿贴纸）以及内部生成的目标层级结构，未使用公开数据集。

**📈 对比分析**

在20名参与者的实物维修实验中，完成率为95%（19/20），平均交谈回合数为19.7，较在线基线（100%完成率、11.4回合）表现略低但仍保持高可帮助性，社会存在与信任度在物理情境下降。

**⚠️ 局限性**

局限性包括样本量小、任务过于简单、仅测试一次性维修、在线基线与物理实验的样本/媒介差异、未覆盖更长或更复杂的维修流程及硬件状态感知不足。

---

## 504. Fronthaul-Efficient Cell-Free Massive MIMO via Stream-Adaptive Resolution Control

**arXiv ID:** 2609.26150 | [PDF](https://arxiv.org/pdf/2609.26150v1)

**作者:** Özlem Tuğfe Demir `[一作]` (Bilkent University), Sinan Gezici `[通讯]` (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了受限上行回程链路的无小区大规模MIMO（Cell‑Free Massive MIMO）系统，提出了一种基于流自适应分辨率控制的框架，并实现了联合功率与量化分辨率的优化。

**💡 创新点**

创新点在于：
1) 通过分布式SVD处理将系统解耦为多条并行流，允许每个AP对每条流单独进行量化；
2) 设计了在总回程带宽约束下联合优化发射功率与AP‑流量化分辨率的非凸问题；
3) 使用WMMSE框架并给出半闭式更新，显著降低计算复杂度。

**🔧 技术方法**

使用的技术包括：
- 分布式SVD分解与局部组合
- Bussgang分解建模量化噪声
- WMMSE（加权最小均方误差）块坐标下降算法
- 高分辨率量化模型（β≈c_q2^{-2b})
- Monte‑Carlo仿真估计协方差。

**📊 数据集**

使用的“数据集”为仿真生成的无相关Rayleigh衰落信道，路径损耗采用城市微型街道峡谷模型，部署在250×250 m区域内25个4天线AP，单UE多天线（K可变）。

**📈 对比分析**

对比方法为：
- Uniform Allocation（均匀分配量化位和功率）
- AP‑Proportional Allocation（按有效通道增益分配量化位）
结果显示：提出的WMMSE方案在不同K和总回程预算下均能获得更高的总速率，尤其在K增大或b_{tot}提升时性能差距显著扩大。

**⚠️ 局限性**

限制与挑战：
- 对量化噪声的无相关性和连续β的近似可能导致与真实系统差距；
- 仅考虑上行单UE场景，未讨论多UE或下行情况；
- 需要迭代求解，计算开销相对较大；
- 采用高分辨率量化模型，低分辨率场景下可能不精确。

---

## 505. From Risk Scoring to Risk Allocation: A Density-Driven Framework for Diverse Monitoring in Multi-Agent Systems

**arXiv ID:** 2609.26146 | [PDF](https://arxiv.org/pdf/2609.26146v1)

**作者:** Zhaohui Wang `[一作]` `[通讯]` (University of Southern California), Zhaohui Wang (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在多智能体系统中，将风险监控从单独的异常评分转变为基于密度的风险分配，并用QUBO框架进行组合优化。

**💡 创新点**

提出Crowding Paradox和基于密度的Fragility Score，并将监控任务重新表述为可通过QUBO求解的组合子集选择问题。

**🔧 技术方法**

使用高斯混合模型密度估计、RBF相似度、QUBO编码、模拟退火、精确枚举、QAOA量子算法以及Rigetti量子硬件。

**📊 数据集**

采用2018-2025年金融市场日级特征数据（SPY、QQQ、债券收益率、波动率、黄金、加密货币等）以及交通和多智能体强化学习数据进行交叉验证。

**📈 对比分析**

与贪婪、MMR、k-DPP等传统多样性子集方法以及模拟退火/精确解进行对比，在不同规模下实现相对多样性提升从+24%到+66%，在量子硬件上与经典方法获得相同Pareto前沿。

**⚠️ 局限性**

受限于小规模候选集（n≤20）、固定候选池、单层QAOA深度、硬件噪声及缺乏滚动窗口实时更新等，未验证更大规模和跨领域的完整部署。

---

## 506. Smart Low-Carbon Freight Transport: A Comparative Analysis

**arXiv ID:** 2609.26111 | [PDF](https://arxiv.org/pdf/2609.26111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 507. VACS: Value-Aligned Compositional Shielding for Multi-Agent Reasoning

**arXiv ID:** 2609.26135 | [PDF](https://arxiv.org/pdf/2609.26135v1)

**作者:** Yiyao Zhang `[一作]` (University of Wollongong), Jun Shen `[通讯]` (University of Wollongong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VACS 系统实现多智能体推理的价值对齐与安全屏障，解决了价值冲突与一致性问题。

**💡 创新点**

将价值推断、Lean DSL 形式化约束、基于 nucleolus 的协商和 Hamiltonian 共识相结合，形成四层框架。

**🔧 技术方法**

使用 Bradley–Terry + MaxEnt IRL 推断价值权重，Lean-DSL 编码安全规则，假设-保证盾构，核子分配+哈密顿优化，梯度敏感性提取关键路径。

**📊 数据集**

在 MathInstruct-Subset、NEJM-AI QA 与 CyberSec-Eval 三个基准上评估。

**📈 对比分析**

与多数投票、加权投票、知情投票和仅屏障方法对比，VACS 在准确率上提升 1–6.9 点，逻辑不一致率降至 0%。

**⚠️ 局限性**

局限包括人工设定的价值异质性、离线价值估计缺乏在线漂移检测、保证仅局部有效、计算成本高、仅在受控面板上验证。

---

## 508. GDLAM: Group-Disentangled Latent Action Model for Highly Disentangled Embodied Pretraining

**arXiv ID:** 2609.26118 | [PDF](https://arxiv.org/pdf/2609.26118v1)

**作者:** Jiarui Yang `[一作]` (Nankai University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 Group-Disentangled Latent Action Model (GDLAM)，从无动作视频中学习可因果分离的潜在动作表示，并验证其在世界模型和视觉语言动作策略中的可迁移性。

**💡 创新点**

创新点在于将潜在动作按 N 组分解，每组拥有独立变分瓶颈和空间门控路由，并通过互相排斥、组稀疏、门稀疏、静态-动态正交等信息几何约束，实现真正因果可分离的动作子空间。

**🔧 技术方法**

使用了冻结的 DINOv2 tokenizer、Delta Encoder、Perceiver 结构的组编码器与预测器、空间门控注意力、变分自编码器、信息几何正则、互信息估计与线性探测等技术。

**📊 数据集**

在 Open X-Embodiment、EgoExo4D、Assembly101 等无动作视频集上预训练，随后在 WorldArena、SIMPLER、LIBERO 等基准上进行评估。

**📈 对比分析**

与 UniVLA、LAPA、Moto、MVP-LAM、DreamDojo 等主流 LAM 进行无标签解耦度量（Modularity、MIG、DCI）和因果干预实验，GDLAM 在解耦度量上提升 0.3+、干预泄漏率降至 0.05，并在世界模型和 VLA 任务上分别实现最高控制/成功率，优于 SOTA。

**⚠️ 局限性**

局限性包括：需手动设定组数与静态组先验，门控初始化与变分瓶颈设计对效果敏感；在极大尺度或多模态视频上可能面临计算开销；对某些细粒度动态仍未完全分离。

---

## 509. Certified Mechanistic Interpretability: Lifting Single-Input Findings to Bounded Neighbourhoods

**arXiv ID:** 2609.26112 | [PDF](https://arxiv.org/pdf/2609.26112v1)

**作者:** Zhen Zhang `[一作]` (Technical University of Munich), Amr Alanwar `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于受限多项式锥（CPZ）的框架，将单输入的机制观察扩展为对有限扰动邻域内的可证明性判定，针对 Transformer 的注意力层实现了三类可验证查询（top‑k 稳定性、证据质量、注意力熵），并将其递归传播至预训练模型深度；

**💡 创新点**

创新点在于：①在不破坏 softmax 简单形约束和 LayerNorm 恒等的前提下，使用 CPZ 维持 Q^⊤K 的二次多项式精度；②提出递归 Jacobian‑zonotope 连接器一次性线性化整个块堆栈，避免层级生成器爆炸；③将单输入观察升维为邻域范围的可验证机制，首次实现预训练 Transformer 级别的内部机制可验证；

**🔧 技术方法**

核心技术包括：受限多项式锥（CPZ）代数、受限多项式锥传播规则、softmax 与 LayerNorm 的精确保留、递归 Jacobian‑zonotope 线性化、对注意力三类查询的线性/凹性规划求解；

**📊 数据集**

实验数据集：BERT‑tiny（SST‑2）、GPT‑2 小型（124M）以及四层合成 Transformer；对比实验还使用了 MC+PGD（蒙特卡洛 + 投影梯度下降）作为经验上界；

**📈 对比分析**

与现有方法（CROWN、IBP、PZ 等）比较时，CPZ 在 top‑1 稳定性上在 GPT‑2 Layer‑1 仍保持 40%+ 的可验证率，远超 CROWN/IBP；在全模型输出层的 10‑step 词级验证中，CPZ 通过 95% 以上的可验证率实现了 90% 级别的全词汇表覆盖；运行时间比 IBP 低数倍；

**⚠️ 局限性**

局限性包括：①随着模型维度增大，CPZ 与 MC 的紧密度差距增大（GPT‑2 Layer‑1 达到 30.2%）；②最终全词汇表的可验证率受保守 Lipschitz 尾部约束影响，难以达到 100%；③仅在四层合成 Transformer 上验证了递归传播的多层可行性，尚未验证更深层；④针对实测 LLM 头的剪枝效果仅在小型合成模型上验证，未证实能迁移至大规模 fine‑tuned 模型；

---

## 510. ST-NDT: A Topological Framework for Reducing Communication Overhead in Network Digital Twins

**arXiv ID:** 2609.26107 | [PDF](https://arxiv.org/pdf/2609.26107v1)

**作者:** John Sengendo `[一作]` (University of Trento), Fabrizio Granelli `[通讯]` (University of Trento)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Hodge理论的稀疏拓扑网络数字孪生（ST‑NDT）框架，用于在有限传感器预算下进行网络流量监测与重构，显著降低物理网络与数字孪生之间的通信开销。

**💡 创新点**

创新点在于：① 将网络建模为细胞复形并利用Hodge拉普拉斯的谱结构实现跨维度信号分解；② 设计基于QR列主元与Hodge子空间的自适应传感器放置策略，覆盖梯度、循环和谐波分量；③ 采用基于拉普拉斯的Tikhonov正则化实现稀疏重构，从而兼顾精度与计算效率。

**🔧 技术方法**

使用的技术包括：拓扑信号处理（TSP）、Hodge谱分解、细胞复形建模、QR列主元/贪心条件化改进、Tikhonov正则化重构、以及对比实验的NRE/MAE/RMSE评估。

**📊 数据集**

实验数据集为Cogentco ISP骨干网络（197节点、243条边），结合生成的混合梯度与循环流量信号作为仿真流量。

**📈 对比分析**

与度数、介数、核心数及随机放置四种基线方法比较，ST‑NDT在10%–40%监测预算下平均NRE提升约8–12%，在20%预算下分别比度数、介数、核心数和随机方法提升10.85%、7.36%、11.24%和6.15%，并在所有预算水平上保持最小误差。

**⚠️ 局限性**

局限性包括：① 对大规模网络的计算复杂度仍为O(E³)（尤其是特征分解与Tikhonov求解）；② 目前仅处理静态流量信号，未考虑时变动态；③ 传感器放置策略基于谱信息，可能在非Hodge稀疏性不足的网络中效果有限。

---

## 511. Post-Hoc Attention Steering of Large Language Models for Robust Code Understanding under Obfuscation

**arXiv ID:** 2609.26102 | [PDF](https://arxiv.org/pdf/2609.26102v1)

**作者:** Xiaokai Rong `[一作]` (University of Texas at Dallas), Tien N. Nguyen `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种后期注意力引导方法，用于提升大语言模型在代码混淆环境下的理解能力。

**💡 创新点**

创新点在于将轻量级程序分析（静态切片、数据/控制依赖）与推理时的注意力重分配结合，形成自适应的软先验，引导模型关注语义相关的代码片段。

**🔧 技术方法**

主要技术包括静态程序分析（使用Joern生成依赖图）、注意力重权重（PASTA式稀疏头校准和解码时先验更新）以及对大型代码模型的推理时注意力干预。

**📊 数据集**

实验数据集为Java程序的HumanEval-X（164个任务）与CruxEval-X（698个任务）生成的输入输出案例，以及额外的恶意代码和缓冲区溢出案例进行案例研究。

**📈 对比分析**

通过与未混淆基线、无引导混淆、以及单纯提示式引导的对比，采用Pass@k和F1回溯率衡量，结果显示注意力引导在混淆代码上能恢复超过70%性能提升，部分模型甚至超过原始代码基准。

**⚠️ 局限性**

局限性包括仅针对Java和有限的混淆技术、对大模型的推理时计算开销、静态切片先验的不完整性以及对更复杂程序结构（如多文件、外部库）的适用性待验证。

---

## 512. Design and Implementation of an Ultra-Low-Cost Wall-Climbing Robot for Infrastructure Crack Detection

**arXiv ID:** 2609.26130 | [PDF](https://arxiv.org/pdf/2609.26130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 513. Why Do LLMs Fail at OCL Generation? A Graph Reasoning Perspective

**arXiv ID:** 2609.26122 | [PDF](https://arxiv.org/pdf/2609.26122v1)

**作者:** Hamza Attarwala `[一作]` (Polytechnique Montréal), Omar Alam `[通讯]` (Trent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了LLM在将自然语言和UML类图转化为OCL约束时失败的根源，系统评估了结构属性、词汇相似度以及提示策略对生成准确性的影响。

**💡 创新点**

将OCL生成视作图推理问题，揭示了导航深度和结构复杂度对LLM性能的显著负面影响，并证明词汇相似度并非主要致因，进一步评估了图感知提示策略的效果。

**🔧 技术方法**

使用六大主流LLM（Claude、GPT‑5、Llama 4、Gemini 2.5、DeepSeek v3.1、Grok 4），结合Chain‑of‑Thought、Few‑Shot、Graph‑Based Prompting等提示技术，采用GEE统计模型和USE验证器进行语法与行为验证。

**📊 数据集**

基于PathOCL数据集（13个UML类图、115条自然语言规范及其金标准OCL）进行实验。

**📈 对比分析**

通过多数投票聚合10次生成，比较不同提示策略和LLM的正确率与错误类型；图感知提示提升了约10–20%的正确率，GPT‑5在图感知提示下达到最高91/115，其它模型亦有所提升，但仍存在导航、幻觉和逻辑错误。

**⚠️ 局限性**

实验受限于数据集规模小、仅覆盖类图且缺乏更复杂的UML元素，评估以有限测试实例判定语义一致性，且仅考虑了部分结构指标与序列化方式，结果不一定能推广至工业规模模型。

---

## 514. Unanimity Without Persuasion: A Single Round of Debate Erases the Disagreement That Verification Needs

**arXiv ID:** 2609.26145 | [PDF](https://arxiv.org/pdf/2609.26145v1)

**作者:** Yang Shu `[一作]` `[通讯]` (Zhejiang University), Yang Shu (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多代理辩论（multi‑agent debate）对评审团一致性和不一致信号的影响，并测量了在一次辩论后，评审团不一致性如何几乎消失以及随后的执行验证投票（verification ballot）在补救错误方面失效的现象。

**💡 创新点**

创新点在于首次量化了辩论导致的不一致信号的快速崩溃（尤其是一轮辩论后达95%以上的一致率），并通过控制实验拆解出这种崩溃主要由“二次审议（no‑peer second‑look）”和“显示标签对齐”驱动，而不是论证内容本身。

**🔧 技术方法**

技术方法包括：① 使用7个不同提供商的LLM评审员进行盲判和三轮辩论；② 通过“margin”与“pivotal”概念以及代数推导（Fact 1）评估替代投票（substitution）的潜在影响；③ 设计四种对照实验（随机标签、仅标签、无同伴审议、完整辩论）；④ 对每一轮的投票结果、翻转方向与执行验证的可实现性进行逐轮统计和置信区间分析。

**📊 数据集**

使用的数据集为600个代码正确性（code‑correctness）候选项，来自HumanEval+与MBPP+，其中包括多种语言模型生成的解答，且通过完整测试套件得到确定性的执行结果。

**📈 对比分析**

比较方法是将每轮的投票一致率、pivotal比例、每位评审员的翻转率以及执行验证替代投票所产生的正确率变化进行对比；结果显示：从39.5%到95.2%的一致率在第一轮完成；准确率仅提升不到1个百分点；而验证投票在第一轮后不再产生任何有效纠正，显示其效用被辩论完全抹杀。

**⚠️ 局限性**

限制因素包括：① 由于解析失败导致的样本失调（attrition）并非随机，影响对更具争议候选项的观测；② 仅在代码正确性领域和单一辩论协议（temperature 0、400‑token限制）下验证，缺乏跨域与多样化协议的验证；③ 只测量了替代投票（substitution），未覆盖学习型仲裁器（override）等更复杂的干预；④ 评审团规模有限（7位评审），且大部分结果基于固定种子样本，外推性受限。

---

## 515. Machine Learning-Based Delivery Time Prediction for Low-Carbon Food Delivery

**arXiv ID:** 2609.26116 | [PDF](https://arxiv.org/pdf/2609.26116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 516. A Cross-Dataset based Zero-Day Intrusion Detection System by Integrating Siamese Network and Reinforcement Learning

**arXiv ID:** 2609.26115 | [PDF](https://arxiv.org/pdf/2609.26115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 517. When Verifiers Vote Backwards under Verdict Substitution: Signed Pivotal Value in Correlated Self-Consistency

**arXiv ID:** 2609.26144 | [PDF](https://arxiv.org/pdf/2609.26144v1)

**作者:** Yang Shu `[一作]` `[通讯]` (Zhejiang University), Yang Shu (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在自一致性（self-consistency）面板中，用一次正确性判定投票替换单一投票后对多数决结果的影响，推导并验证了关键“签名增益”公式，并通过实验表明决定结果的关键在于验证器在两种关键投票状态下的准确率，而非单纯的全局准确率或模型来源；

**💡 创新点**

提出了基于一投替换的精确增益分解公式，揭示了同源与异源验证器在关键投票上的行为差异，并给出了负权重PRM（Process Reward Model）权重可能为负的可检验假设；

**🔧 技术方法**

采用自一致性面板（k=7，温度0.8）与投票替换、二元化正确性判定、Bootstrap置信区间、误差相关性分析、聚类校准、随机翻转干扰等统计与实验技术；

**📊 数据集**

使用MATH‑500（190道难度为4/5的数学题）与LiveCodeBench（60道代码生成题）两类公开基准数据集进行实验；

**📈 对比分析**

对比了四种配置（主实验、角色反转、代码压力测试、强验证器）和多种同源信号（多重抽样、加权聚合、提示强化等），主要实验在关键投票上实现+24.2个百分点的提升，角色反转则出现-11.2个百分点的下降，其他实验也展示了相似的正负效应；

**⚠️ 局限性**

局限性包括仅在k=7、单温度的面板；代码实验样本量小、未覆盖多任务；只评估了答案身份投票的“平局错误”策略；未使用真实训练的PRM或加权投票聚合；误差相关性仅为描述性关联；token预算升高对最难样本的影响仍未系统探测；因此结论对更广泛场景的推广仍需进一步验证。

---

## 518. MAC-RRG: Iterative Multi-Agent Collaboration for X-ray Radiology Report Generation

**arXiv ID:** 2609.26124 | [PDF](https://arxiv.org/pdf/2609.26124v1)

**作者:** Futian Wang `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了基于多代理协同迭代的胸部X光报告生成框架MAC‑RRG，利用初稿提取实体，分别通过知识图谱（MM‑KG Agent）和检索式文本知识（Knowledge Agent）进行多源知识挖掘，并将其融合与视觉特征一起输入LLM进行报告生成。

**💡 创新点**

创新点在于将知识图谱与外部文本检索分别拆分为专门的代理，形成动态闭环迭代生成流程，突破单轮静态知识融合的瓶颈，显著降低医学幻觉并提升报告的结构化与可解释性。

**🔧 技术方法**

采用Swin Transformer视觉编码、Llama2‑7B大型语言模型、Bio_ClinicalBERT编码器、MM‑KG Agent、Knowledge Agent、知识融合模块及自回归语言模型训练；同时使用BGE检索和重排序。

**📊 数据集**

在公开的IU X‑Ray、CheXpert Plus和MIMIC‑CXR三大胸部X光报告数据集上进行训练与评估。

**📈 对比分析**

与多种基线（如R2GenGPT、R2GenCMN、PPKED、METransformer等）对比，MAC‑RRG在BLEU‑3/4、ROUGE‑L、CIDEr及CE（F1/Precision/Recall）指标上均取得或接近最高分，尤其在高阶BLEU和临床准确度方面优于单一知识图或LLM模型。

**⚠️ 局限性**

模型受初稿质量影响，若实体抽取不完整或错误会导致检索不足；知识融合仅采用拼接方式，未探索更复杂交互机制；检索语料规模有限，可能引入噪声，且对知识图和文本的覆盖率与更新频率有待提升。

---

## 519. FairMon: A Tool for Monitoring and Visualizing Algorithmic Fairness

**arXiv ID:** 2609.26123 | [PDF](https://arxiv.org/pdf/2609.26123v1)

**作者:** Jan Baumeister `[一作]` (CISPA Helmholtz Center for Information Security), Tobias Wagenpfeil `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一款名为FairMon的运行时监控工具，用于实时检测和可视化高风险决策系统的算法公平性；

**💡 创新点**

通过在RTLola中引入专门的概率类型和概率运算符，使公平性规范的书写更简洁直观，降低了非监控专家的使用门槛；

**🔧 技术方法**

基于RTLola流式规范语言、概率型运算符、图形化用户界面和插件化输入/输出框架，并使用Plotly进行实时可视化；

**📊 数据集**

使用了COMPAS刑事再犯风险预测数据集和欧盟DSA透明度数据库的在线内容审核决策数据；

**📈 对比分析**

与之前的长篇RTLola规范相比，FairMon将等化机会规范缩短57%，输出流数从7条减少到3条，实验中对DSA API的过滤机制将查询时间缩短93%，在可视化和实时监测方面表现优异；

**⚠️ 局限性**

仍依赖于手动配置API过滤条件，且对隐私敏感数据的处理缺乏正式的隐私保障机制，未来需要进一步自动化配置和集成隐私保护功能。

---

## 520. Unread or Unenforced? Separating Representation from Enforcement Failure in Content Guards

**arXiv ID:** 2609.26178 | [PDF](https://arxiv.org/pdf/2609.26178v1)

**作者:** Haoyu Zhang `[一作]` (Northeastern University), Shanu Sushmita `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对内容守护器的编码攻击，作者提出了一种基于守护器内部残差流的线性内容探测器方法，用来分离“能力失效”（守护器根本无法解码）与“策略失效”（守护器可读但错误判断）的两种失败类型，并对 Llama Guard 3（8B）和 WildGuard（7B）在多种编码条件下的表现进行了测评。

**💡 创新点**

创新点在于：①首次通过将线性探测器从明文训练转移到已编码输入，直接读取守护器内部残差来衡量内容是否被解码；②在传统自由置换显著性检验之外，加入长度匹配的零假设与基于守护器基模型不可解码条件的“控制底线”，有效剔除伪阳性；③在同一数据集上对两种守护器进行可比性分析，揭示它们在能力与策略失效上的显著差异。

**🔧 技术方法**

技术手段包括：线性内容探测器（在残差层上训练并转移评估）、守护器日志件（读取 logits 判断安全标签）、自由置换显著性检验、长度匹配零假设、基模型不可解码控制底线、阈值操作点扫描，以及对结果进行统计显著性与鲁棒性验证。

**📊 数据集**

使用的评估数据集为单一英文提示集合，包含 19 种编码条件（包括密码、全角替换、无形字符、同形字替换、词序打乱等），每种条件下对正向（有害）和负向（无害）提示进行编码后测试。

**📈 对比分析**

比较方法：对每个守护器和每个编码条件，计算（1）探测器 AUROC；（2）在阈值调优下的“被解码且未被阻止”比例（D&B）；（3）阻止率。结果显示：Llama Guard 在 4 条条件下 D&B 为 7–23/100，阻止率 0.71–0.92；WildGuard 在 4 条条件下 D&B 为 9–56/100，阻止率 0.25–0.75。两守护器在某些条件（如某一无形字符）上出现完全不同的行为，说明策略失效差异显著。

**⚠️ 局限性**

局限性：①仅评估两种守护器和单一英文提示集，结果不具普适性；②探测器仅为线性单层方法，可能漏检非线性解码；③无法对真实密码条件测量，因为残差流不包含可线性恢复的内容；④方法仅为相关性测试，缺乏因果验证；⑤控制底线需基模型不可解码条件充足，对另一守护器尚未完成；⑥未对生成模型的实际行为做验证。

---

## 521. The Uncontrolled Variable: Vision-Language Model Refusal Responds to Image Presence in Ways Risk Cannot Explain

**arXiv ID:** 2609.26174 | [PDF](https://arxiv.org/pdf/2609.26174v1)

**作者:** Haoyu Zhang `[一作]` (Northeastern University), Shanu Sushmita `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在对齐的视觉-语言模型（VLM）中，附加一张内容为空的图像（如空白画布）如何改变拒绝阈值。

**💡 创新点**

首次发现图像存在这一表面特征（而非内容）即可显著提升模型对敏感请求的拒绝率，且效应在不同检查点上方向和幅度可变，揭示了安全对齐中的一个隐藏快捷路。

**🔧 技术方法**

采用黑盒对照实验、配对McNemar检验、图像属性分解、检查点与服务路径对比，以及基于LLM判别器的拒绝/危害评分。

**📊 数据集**

使用公开安全评估数据集：OR‑Bench（善意/有害拆分）、JailbreakBench（善意对照）、HarmBench（有害完成率）。

**📈 对比分析**

通过对比四个前沿托管模型和多达六个开源检查点的拒绝率与危害完成率，观察到十几到五十个百分点的效应；在所有对照中p值均低于0.001，表明效应显著且可重复。

**⚠️ 局限性**

局限性包括：仅测试了少数固定图像（空白、线稿、字幕），未覆盖随机图像或真实照片；开源检查点样本有限；判别器可能存在偏差；无法明确揭示内部机制，且效果在不同模型间变化大。

---

## 522. MGRL-RSCC: Multi-Granularity Reward Reinforcement Learning for Fine-Grained Remote Sensing Change Captioning

**arXiv ID:** 2609.26166 | [PDF](https://arxiv.org/pdf/2609.26166v1)

**作者:** Futian Wang `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种多粒度奖励强化学习框架MGRL-RSCC，用于遥感图像变化描述生成。

**💡 创新点**

创新点在于将句子级指标奖励、全局变化状态奖励和细粒度结构语义奖励三类多层奖励融合，并通过自我批评策略进行联合优化，显著缓解自回归训练的曝光偏差与保守生成问题。

**🔧 技术方法**

核心技术包括CNN+分层自注意力的双时序视觉编码、Transformer解码器、两阶段联合优化（交叉熵+自我批评RL）、以及基于BLEU/CIDEr、变化一致性与知识图奖励的多粒度奖励设计。

**📊 数据集**

在LEVIR-CC、Dubai-CC和WHU-CDC三大公开遥感变化描述基准上进行实验。

**📈 对比分析**

与多种基线（PSNet、RSICCFormer-C、MCCFormer等）对比，MGRL-RSCC在BLEU-4、CIDEr、METEOR等指标均获得最高或近乎最高分，提升幅度从1-6点不等。

**⚠️ 局限性**

局限性包括对微小或边缘变化的识别仍不够准确，奖励依赖预定义短语与图结构，导致对稀有物体/未见关系的泛化受限。

---

## 523. Spectral Tail Interventions in Decoder-Only Language Models: Reasoning-Sensitive Weight Structure from Controlled Surgery

**arXiv ID:** 2609.26165 | [PDF](https://arxiv.org/pdf/2609.26165v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对解码器单向Transformer的权重空间进行控制性谱剪裁，重点剪除QK乘积上尾部的主导特征，并评估其对多项推理基准的影响。

**💡 创新点**

提出基于逆参与度的有限宽度矩界，证明上尾部能量与QK预softmax对数点峰度之间的关系；引入点对点尾部目标，并对比独立因子裁剪与目标化因子裁剪；展示尾部剪裁对推理性能的显著损伤，并提供尾部意识LoRA实现的加速适配。

**🔧 技术方法**

利用SVD分解、逆参与度与峰度矩推导、点对点上尾部删除、产品导向与独立因子裁剪、匹配能量与范数控制、低秩残差LoRA、Bootstrap置信区间、Holm校正、对数正态校准等技术。

**📊 数据集**

使用多项推理基准（GSM8K、ARC‑Challenge、DROP、BIG‑Bench Hard、MMLU‑CoT）以及通用语言评测（WikiText‑103、LAMBADA、coreference、IFEval、WinoGrande）。

**📈 对比分析**

与五个匹配能量的Haar随机控制、匹配范数、底部谱删除等对照进行比较；在20个模型–任务组合中，学习尾部剪裁平均比随机控制高，18/20在Holm校正后显著；尾部意识LoRA在目标更新上比标准LoRA快约30–40%，最终分数相近。

**⚠️ 局限性**

理论仅针对QK预softmax对数点峰度的局部条件；实验仅覆盖小规模decoder‑only模型，未验证RLHF、Mixture‑of‑Experts、跨模态等；尾部能量与方向选择为实验设计参数；推理损伤不代表所有语言能力受影响，且方法可能被用于攻击。

---

## 524. Information-Theoretic Decoupled Prompt Tuning for Continual Learning

**arXiv ID:** 2609.26257 | [PDF](https://arxiv.org/pdf/2609.26257v1)

**作者:** Yunfei Zhang `[一作]` (Xi’an Jiaotong University), Weizhan Zhang `[通讯]` (Xi’an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无回放的CLIP文本提示调优框架DPT4CL，用于类增量持续学习；

**💡 创新点**

创新点在于将提示空间解耦为任务共享分布和类别特定提示，并通过信息瓶颈、分布蒸馏和正交化三种信息论驱动的正则化来同时提升跨任务迁移、消除分类器偏置和增强类间可分性；

**🔧 技术方法**

技术包括信息瓶颈（IB）目标、分布和特征层蒸馏、正交正则、CLIP文本-视觉双编码器以及基于变分推断的潜在分布学习；

**📊 数据集**

实验使用四个公开的持续学习基准：CIFAR‑100、ImageNet‑R、CUB‑200 和 UCF‑101；

**📈 对比分析**

与传统CL、提示池、单一共享提示和基于适配器/LoRA的CLIP方法相比，DPT4CL在 Avg./Last/忘记率(FM)指标上均实现了最高或次高成绩，且不需要任何回放数据；

**⚠️ 局限性**

限制在于目前只在文本提示空间调优，尚未结合特征空间适配器或LoRA等方法，且对极大规模任务序列的长期稳定性仍需进一步验证。

---

## 525. Can You Delete a Year of Market Data? Machine Unlearning Against Exact Retraining Oracles

**arXiv ID:** 2609.26242 | [PDF](https://arxiv.org/pdf/2609.26242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 526. When Does Permutation Instability Generalize? Independent-View Validation for Listwise LLM Reranking

**arXiv ID:** 2609.26251 | [PDF](https://arxiv.org/pdf/2609.26251v1)

**作者:** Wenzhang Du `[一作]` `[通讯]` (Independent Researcher), Wenzhang Du (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了有限视图不一致诊断在未见扰动下的有效性，并通过前瞻性独立视图实验评估观测重用的影响。

**💡 创新点**

创新点在于提出“fresh‑view validity”概念，剔除观测重用后揭示不一致性指标并不等价于对未知扰动的预测，提供了新的评估框架。

**🔧 技术方法**

技术手段包括确定性序列化、排名曝光方差诊断、Spearman相关与bootstrap统计，以及固定分数路由实验。

**📊 数据集**

使用的数据集为MovieLens‑1M和KuaiRec两套推荐数据，并分别构造受控与未受控的20项候选列表。

**📈 对比分析**

比较方法是将重用视图与完全独立视图的Spearman相关对比，并在路由决策中检验NDCG与调用次数；结果显示重用显著提升相关性，但在路由决策中并未普遍提升性能。

**⚠️ 局限性**

局限性包括仅针对两款7B指令模型、确定性解码、固定列表长度，未覆盖随机解码、不同模型规模或真实线上评估。

---

## 527. PACE-dLLM: Elastic Block Decoding via Confidence Cliff Estimation for Diffusion Language Models

**arXiv ID:** 2609.26249 | [PDF](https://arxiv.org/pdf/2609.26249v1)

**作者:** Xiaocheng Lu `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在扩散语言模型中提出了可自适应的 PACE-dLLM 解码器，解耦了预测窗口与提交阈值。

**💡 创新点**

通过将模型自身的置信度“悬崖”闭式拟合为水平点来动态决定未来窗口长度，提升了并行性与精度。

**🔧 技术方法**

使用离散扩散训练、对置信度的逻辑回归拟合、滑动光标、阈值提交等技术。

**📊 数据集**

在 LLaDA-8B-Instruct 和 Dream-7B-Instruct 两大模型上，结合 GSM8K、MATH、HumanEval、MBPP 四个推理/编程基准。

**📈 对比分析**

与默认半自回归、Fast-dLLM、AdaBlock-dLLM、DepCap 等方法比较，平均准确率提升约 2-3 点，速度提升 5.2×（LLaDA）或 3.1×（Dream），单个任务最高 8.5×。

**⚠️ 局限性**

对置信度校准依赖较大，非单调悬崖时退回固定上限，实验仅覆盖两大 7B/8B 模型，需进一步验证。

---

## 528. ABAI at COLIEE 2026 Task 1: Multi-Stage Retrieval with GraphRAG-Enhanced Meta-Learning, and a Post-Hoc Study of the Cross-Validation-to-Test Gap

**arXiv ID:** 2609.26237 | [PDF](https://arxiv.org/pdf/2609.26237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 529. Human-Centricity in Industry 5.0: A Survey of Worker Sensing, Adaptive Operations, and Human-in-the-Loop Systems

**arXiv ID:** 2609.26245 | [PDF](https://arxiv.org/pdf/2609.26245v1)

**作者:** Lara Pereira `[一作]` (University of Coimbra), João Ruivo Paulo `[通讯]` (University of Coimbra)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文对人本化工业5.0中工人状态监测、运营管理与人机交互四个层面的研究进行了系统综述，并识别出三者之间缺乏闭环整合的核心问题。

**💡 创新点**

创新点在于提出了四层“感知‑决策‑解释‑反馈”框架，系统性梳理了各领域发展现状，并明确提出五条未来研究方向。

**🔧 技术方法**

采用umbrella‑review方法，结合系统与范围综述，对IEEE Xplore、Scopus、Web of Science、ScienceDirect、ACM Digital Library等数据库检索，并对感知、决策、解释与交互四层进行主题归纳。

**📊 数据集**

参考了多种公开数据集如Human3.6M、MPI‑INF‑3DHP、WESAD、MultiPhysio‑HRC等，用于阐述工人状态估计和认知负荷评估技术。

**📈 对比分析**

通过对比分析39篇核心论文的技术成熟度与应用范围，作者指出目前仅存在脱节的单项技术，尚未出现实时传感→运营决策→工人反馈的完整闭环。

**⚠️ 局限性**

局限在于多学科间分隔、方法不匹配导致的整合障碍，以及缺乏在真实工厂环境中进行多模态闭环验证的实证数据。

---

## 530. COVER: Codec-Robust Video Watermarking with Generative Video Priors

**arXiv ID:** 2609.26236 | [PDF](https://arxiv.org/pdf/2609.26236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 531. Manipulation of Deformable Linear Objects Using Model Predictive Path Integral Control with Bidirectional Long Short-Term Memory Learning

**arXiv ID:** 2609.26238 | [PDF](https://arxiv.org/pdf/2609.26238v1)

**作者:** Lukas Zeh `[一作]` (University of Stuttgart), Alexander Verl `[通讯]` (University of Stuttgart)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用双向LSTM模型预测柔性线性物体（DLO）的动态，并结合MPPI（模型预测路径积分）控制器实现对DLO形状的精准操控。

**💡 创新点**

将数据驱动的biLSTM动力学模型与采样型MPC（MPPI）结合，实现实时、可预测且对不同形状与材质的DLO具有良好泛化的控制框架。

**🔧 技术方法**

双向LSTM网络、MPPI控制器、MuJoCo物理仿真、Realsense RGB‑D视觉跟踪、FastDLO形状估计等技术。

**📊 数据集**

使用MuJoCo生成的合成数据集（10,000条轨迹，每条5 s，共50个胶囊点，随后裁剪为10个相对位置），并在真实机器人上收集视觉数据验证。

**📈 对比分析**

在仿真中针对U‑形、随机目标形状分别实现93 %/20.5 %成功率（平均7.3/13.3 s）；在实验中对2D/3D/细线三种情景分别取得85 %/50 %/60 %成功率（平均15.7/28.3/25.4 s）。相较传统物理模型和基于GNN/MLP的动力学预测，biLSTM+MPPI在形状误差、时间与成功率上表现优越。

**⚠️ 局限性**

受限于仿真与真实差距、对极大变形/预弯形状的鲁棒性不足、MPPI采样数量与计算开销高、模型对不同材质的精度仍需提升。

---

## 532. Beyond the Lab: Large-Scale Remote Cybersickness Research in Virtual Reality Using the VERA Platform

**arXiv ID:** 2609.26203 | [PDF](https://arxiv.org/pdf/2609.26203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 533. Reducing Hallucinations in Large Language Models Through Integrated Self-Verification and Retrieval-Augmented Generation

**arXiv ID:** 2609.26229 | [PDF](https://arxiv.org/pdf/2609.26229v1)

**作者:** Ashly Joseph `[一作]` `[通讯]` (Cisco Systems), Ashly Joseph (Cisco Systems)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了CoVe‑RAG+框架，结合Chain‑of‑Verification与Retrieval‑Augmented Generation以降低LLM幻觉，应用于CAD文档编制、标准合规验证及设计知识检索；

**💡 创新点**

三大创新：①多模态自适应检索，支持文本、CAD元数据与仿真报告；②动态再检索触发，低置信度时自动获取补充证据；③可解释层生成验证报告，提供来源追溯与可信度说明；

**🔧 技术方法**

技术手段：LLM（Mistral 7B、LLaMA 2 65B）Fine‑tuned via LoRA；LangChain集成ElasticSearch、FAISS与BM25实现稠密+稀疏检索；CAD/仿真数据解析模块；Chain‑of‑Verification迭代验证逻辑；动态阈值与并行检索流水线；

**📊 数据集**

使用数据集：Wikidata Engineering Subset、MultiSpanQA Engineering Edition、GrabCAD CAD Metadata、ArXiv Engineering Papers Corpus、NIST MDR；

**📈 对比分析**

与基线CoVe与RAG Only对比，采用Precision、Recall、FACTSCORE评估；CoVe‑RAG+在Precision 0.48、Recall 0.50、FACTSCORE 71.4上显著优于基线，人工评估亦提高事实准确度、合规性与可解释性约20‑30%，统计显著；

**⚠️ 局限性**

局限性：多阶段验证与动态检索导致计算开销大、实时响应受限；依赖外部知识库，若文献过时或缺失会影响验证；目前仅在工程领域验证，跨行业推广需进一步评估；

---

## 534. Generating Query Context for Relational Databases

**arXiv ID:** 2609.26200 | [PDF](https://arxiv.org/pdf/2609.26200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 535. WatchPoint: Executable User Feedback for Real-World Agentic Web Development

**arXiv ID:** 2609.26204 | [PDF](https://arxiv.org/pdf/2609.26204v1)

**作者:** Guanqun Yang `[一作]` (Stevens Institute of Technology), Xueqing Liu `[通讯]` (Stevens Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种模拟用户系统，能够在编码代理首次测试失败后，自动生成并执行浏览器和终端诊断脚本，收集结构化观察并将其反馈给代理以进行针对性重试。

**💡 创新点**

创新点在于将可执行的诊断脚本与多文件、顺序式 Web 开发任务结合，实现了类似真实测试员的交互式、基于运行时的反馈，并证明自动诊断能与人类反馈相媲美。

**🔧 技术方法**

主要技术包括 Playwright 脚本自动生成与执行、Bash 脚本诊断、OpenHands/Agentic 工具框架、Docker 沙盒化环境，以及多种 LLM（MiniMax、GLM‑5、GPT‑5.4）用于生成诊断脚本。

**📊 数据集**

使用的基准数据集为 Web‑Bench，共 50 个真实 Web 项目，涵盖 1,000 个顺序依赖的开发任务。

**📈 对比分析**

通过与无反馈 baseline、不同模型配置以及人类测试者的对比评估；在 Pass@1 最高达 25%，Pass@2 最高 44.5%；模拟用户的任务恢复率为 57.6%，与人类参与者的 54.5% 相近，表明自动诊断效果可与人工反馈相当。

**⚠️ 局限性**

局限性包括：任务顺序化导致指标放大，诊断脚本对 LLM 生成质量敏感，仅在 Web‑Bench 上验证，未涵盖更广泛的开发场景，且实验规模受限于模型与项目数量。

---

## 536. Silent Sabotage: Internal State Triggered Backdoor Attacks on LLM-Powered Robotic Systems

**arXiv ID:** 2609.26184 | [PDF](https://arxiv.org/pdf/2609.26184v1)

**作者:** Doniyorkhon Obidov `[一作]` (Michigan Technological University), Kaichen Yang `[通讯]` (Michigan Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在LLM驱动的机器人系统中，利用指令注入实现的基于自身行动历史触发的内部后门攻击，展示了后门可在正常工作时保持沉默、被触发后导致致命行为。

**💡 创新点**

首次系统性提出并验证了历史序列触发的内部后门，且攻击仅通过修改系统提示即可实现，无需访问模型权重或训练数据。

**🔧 技术方法**

采用指令注入与动作历史匹配的后门逻辑，结合LLM生成JSON低层控制指令的方式，对机器人进行控制。

**📊 数据集**

在Gazebo仿真环境中使用Unitree Go1、Go2、A1三种四足机器人和不同门数（1、2、4）地图进行实验验证。

**📈 对比分析**

通过与无后门正常运行进行对比，攻击成功率几乎100%，而正常任务完成率保持在8/10~10/10之间，证明后门既高效又难以被检测。

**⚠️ 局限性**

仅在仿真环境验证，触发序列为固定短序列，未探索更复杂或动态触发方式，且缺乏实物验证与针对性防御策略。

---

## 537. The Temporal Moderation Gap: Text-to-Video Safety Filters Are Blind to Harm in Motion

**arXiv ID:** 2609.26233 | [PDF](https://arxiv.org/pdf/2609.26233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 538. High-Bandwidth Biomimetic Finger for Tactile-Transparent Remote Texture Sensing

**arXiv ID:** 2609.26256 | [PDF](https://arxiv.org/pdf/2609.26256v1)

**作者:** Shuang Yang `[一作]` (Harbin Institute of Technology), Yitian Shao `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计了三种仿生指尖，通过嵌入高灵敏度惯性测量单元捕捉表面纹理诱发的振动，实现远程触觉透明化。

**💡 创新点**

创新点在于将多层力学梯度与指纹形态相结合，并系统评估不同结构对振动信号与人类触感的影响，形成从设计→信号→感知的完整传输链。

**🔧 技术方法**

技术包括三层仿生结构设计（PETG骨架、TPU/PDMS/Ecoflex皮层）、3D打印与模具成型、嵌入LSM6DSR IMU采集振动、Butterworth滤波与FFT/功率谱分析、语义相似度计算及基于Unity的电磁舵机触觉渲染。

**📊 数据集**

使用11种纹理样本（6种3D打印周期/随机纹理，5种天然表面）进行滑动实验，采集对应振动信号与用户感知。

**📈 对比分析**

比较方法包括：时间域最大相关系数、频域余弦相似度（0–150 Hz与150–300 Hz两频段）、以及30人群的感知一致性Wilcoxon检验。结果显示：PDMS+Ecoflex指尖在频谱峰值显著度与高频能量上优于其他两种；PDMS指尖在低频相似度最高；在用户感知中，PDMS+Ecoflex在粗糙度维度最贴近裸手，TPU+Ecoflex在重复性维度最好，PDMS在颗粒感维度最匹配。

**⚠️ 局限性**

局限性包括：仅使用单点振动反馈，缺乏空间和多模态（力、温度、皮肤拉伸）信息；IMU捕获的系统噪声与直流旁瓣可能引入非纹理特征；不同表面/速度下的感知一致性仍不完全，尤其在平滑金属表面导致的误感。

---

## 539. Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs

**arXiv ID:** 2609.26185 | [PDF](https://arxiv.org/pdf/2609.26185v1)

**作者:** Doniyorkhon Obidov `[一作]` (Michigan Technological University), Kaichen Yang `[通讯]` (Michigan Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 DDPO 的动态深度提示优化防御方法，利用 LLM 早期层的特征通过轻量级 MLP 生成防御嵌入并注入后续层，以实现对 jailbreak 攻击的抑制。

**💡 创新点**

创新点在于首次将输入相关的动态嵌入直接注入 LLM 内部层，而非传统的静态前缀/后缀；并且整个过程不需要修改模型权重，极大提升了适应性与可扩展性。

**🔧 技术方法**

采用了目标 LLM 的中间层特征提取、两层 GeLU 激活的 MLP 生成嵌入、以及在选定层插入防御嵌入的技术，同时使用交叉熵损失训练 MLP 以实现拒绝或帮助响应。

**📊 数据集**

训练使用 50 条 GCG 与人造 jailbreak 示例，评估则结合 AdvBench、JailbreakBench、HarmBench 的 9 种攻击样本以及包含边缘案例的 benign 集和 MMLU 基准。

**📈 对比分析**

与 PAT、RPO、DRO 等静态提示优化方法对比，DDPO 在所有 5 种 LLM 上显著降低攻击成功率（ASR）至接近 0%，同时保持高达 95% 以上的 benign 通过率和 MMLU 分数，特别在弱对齐模型上表现优异。

**⚠️ 局限性**

主要局限在于仅注入单一动态嵌入，可能在某些极端攻击或非常深层模型中效能有限；此外，虽然轻量化，但在更大规模 LLM 上的实时推理成本和对未知新型攻击的零样本适应性仍待进一步验证。

---

## 540. High-Order Liquid Evidence Modeling for Continuous and Subtle GNSS Spoofing Detection in Autonomous Driving

**arXiv ID:** 2609.26231 | [PDF](https://arxiv.org/pdf/2609.26231v1)

**作者:** Muhammad Ayub Sabir `[一作]` (Beijing University of Technology), Fatima Ashraf `[通讯]` (Beijing University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个高阶液态证据模型，用于连续细微的GNSS欺骗检测。

**💡 创新点**

首次将因果顺序的GNSS‑运动残差以不确定性归一化形式建模，并将即时、演化、持久三类证据映射到隐藏空间，经过对称Kirchhoff式交换与三阶乘性交互，再通过第二阶液态时间常数网络对证据进行持续追踪，从而实现低误报的持续报警。

**🔧 技术方法**

使用物理引导的残差归一化、角色映射、对称交换、三阶乘性交互、第二阶液态动力学网络、交叉熵训练与阈值+连续确认等技术。

**📊 数据集**

利用AV‑GPS数据集系列，包括Dataset‑1（训练/验证/测试）、Dataset‑2（外部泛化）和Dataset‑3（单条连续序列案例）。

**📈 对比分析**

与XGBoost、MLP、LSTM、GRU等学习基线在相同输入下比较，AUROC达0.9932、AUPRC 0.9843，误报率仅0.27%，延迟3.5s；在Dataset‑2与Dataset‑3的外部与连续评估中仍保持高AUROC/AUPRC，攻击检测率分别为0.75与1.0，延迟19–35s，整体性能优于现有GPS‑IDS变体。

**⚠️ 局限性**

局限性包括：仅在单条连续轨迹下评估；残差构造依赖GNSS与运动观测的精确同步；阈值与持续长度在验证后固定，缺乏自适应校准与多传感器融合的扩展。

---

## 541. Same Chart, Different Story: Bias in Vision-Language Chart Interpretation

**arXiv ID:** 2609.26210 | [PDF](https://arxiv.org/pdf/2609.26210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 542. Improved Multiplayer Bandit Algorithm for Bernoulli Rewards

**arXiv ID:** 2609.26213 | [PDF](https://arxiv.org/pdf/2609.26213v1)

**作者:** Khang Nguyen `[一作]` (University of California Los Angeles), William Chang `[通讯]` (University of California Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多人多臂赌博机（Multi‑armed Bandit）问题中，针对三种信息不对称情形（仅动作不对称、仅奖励不对称、动作与奖励均不对称），作者提出了基于KL散度的置信区间和消除策略，分别构建了三种算法：A、B、C。

**💡 创新点**

创新点包括：
1) 用KL‑UCB代替Hoeffding置信区间，显著提升对Bernoulli奖励的适应性；
2) 在奖励不对称下首次给出两条KL区间分离的确定性判据，并证明多玩家能实现1/M倍的样本加速；
3) 在动作与奖励均不对称的情形下引入确定性探索与利用序列（DSEE），并利用KL指数改进探索成本。

**🔧 技术方法**

技术手段主要有：KL散度置信区间（Chernoff‑type），凸性与单调性分析，确定性消除（interval‑based elimination），以及利用Pinsker不等式比较KL与Hoeffding宽度；实验中使用模拟Bernoulli分布的奖励样本。

**📊 数据集**

实验数据集为人工生成的Bernoulli奖励，均值范围在[0.02,0.15]，采用T=100,000的时间步长，平均10次实验。未使用公开真实数据集。

**📈 对比分析**

与基线Hoeffding‑UCB算法对比，KL‑based算法在信息不对称A情形下将累计回报误差从约1,900降低到250（≈7.6倍提升），在情形B中回报曲线在约700点趋稳，显著优于基线；情形C中改进有限，主要体现在更快的探索衰减。

**⚠️ 局限性**

局限性：
- 需要先验时间窗口T或阈值设定，固定阈值导致额外的O(log T)开销；
- 仅在Bernoulli或方差可变小的指数族上获得显著优势；
- 情形C中KL改进仅体现在指数衰减上，对总成本影响有限；
- 仍无法实现通信聚合估计，导致多玩家样本未被充分共享。

---

## 543. Beyond Static Charts: Can Language and Vision Language Models Generate Interactive Data Visualization Interfaces?

**arXiv ID:** 2609.26208 | [PDF](https://arxiv.org/pdf/2609.26208v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 544. TREND-10K: A Comprehensive Dataset for Next-Generation Video Quality Assessment Based on Preference-Driven Media

**arXiv ID:** 2609.26187 | [PDF](https://arxiv.org/pdf/2609.26187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 545. Towards Effective Black-Box Adversarial Attacks on Deep Code Models via Structural and Identifier Perturbations

**arXiv ID:** 2609.26234 | [PDF](https://arxiv.org/pdf/2609.26234v1)

**作者:** Bin Duan `[一作]` (University of Queensland), Guowei Yang `[通讯]` (University of Queensland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个输入条件化的黑盒攻击框架，利用LLM生成上下文相关的结构性修改并结合词法重命名，形成层级化的扰动空间来对深度代码模型进行鲁棒性测试。

**💡 创新点**

①将黑盒攻击视为对每个输入构建的分层扰动空间的搜索；②使用LLM生成符合语法且语义保留的结构候选；③在此基础上进行相似性引导的标识符替换；④整体框架既不依赖相反标签样本，也可同时适用于分类与生成任务。

**🔧 技术方法**

LLM（OpenChat‑3.5、GPT‑5‑nano）生成代码编辑；Tree‑sitter 进行代码块识别与语法验证；代码嵌入与FastText用于相似度排序；自适应探索与贪心标识符替换算法；静态语义检查与可执行子集运行验证；表示相似度与人工评估辅助评估。

**📊 数据集**

Clone detection：BigCloneBench；漏洞检测：Juliet C 方案；代码摘要：CodeSearchNet；辅助语料来自各基准的非测试集；对 StarCoderBase‑1B 与 Qwen2.5‑Coder‑1.5B 进行扩展实验。

**📈 对比分析**

与 ALERT、BeamAttack、ITGen、CODA 四大基线对比，实验显示在 CodeBERT、CodeGPT、CodeT5 上三类任务均实现更高的攻击成功率（ASR），查询次数（AMQ）与运行时间（ART）保持竞争性；对抗性微调后在固定攻击评估集上显著提升跨攻击鲁棒性，且保持原始性能。

**⚠️ 局限性**

1）攻击结果的语义等价性仅通过静态检查与可执行子集验证，不能完全保证行为一致；2）仅在可获取标签/参考摘要的离线评估场景下有效；3）实验主要涵盖 C/Java，跨语言泛化需进一步验证；4）对LLM的依赖导致攻击成本与模型可用性相关；5）攻击效果受限于基础数据集与任务定义。

---

## 546. Distributed Near-Equitable Coloring in the LOCAL Model

**arXiv ID:** 2609.26190 | [PDF](https://arxiv.org/pdf/2609.26190v1)

**作者:** Amit Nir `[一作]` (Weizmann Institute of Science), David Peleg `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究分布式网络中的近等量着色（Near‑Equitable Coloring），证明在完全无通信限制的LOCAL模型下，精确等量（exact balance）必须是全局性的，导致在环图上需要Ω(n)轮，且对不等量误差Δε则需要Ω(n/ε)轮。进一步构造了显式图族（twisted fiber products）证明对任意最大度Δ≥3，任意可实现的直径范围内，精确等量着色的确定性复杂度为Θ(D)。另一方面，给出了随机化和确定性算法，在松弛误差下实现多种近似等量着色，时间复杂度仅与log* n或log n相关，完全独立于直径。

**💡 创新点**

1) 引入了刚性（rigidity）与稳定性（stability）分析，首次将全局计数约束转化为局部窗口规则的整数多重性；2) 构造了一族紧凑的twisted fiber product图，能够在每个Δ和直径尺度下实现Θ(D)的下界；3) 在同一理论框架下实现了从Ω(log* n)到Θ(n)的完整复杂度连续体，并证明了随机化能显著降低误差对时间的影响；4) 提出并证明了“锚点（anchor）”构造不可通过更快算法消除的log* n障碍。

**🔧 技术方法**

使用的技术包括：
- LOCAL模型下的窗口（window）规则与全局计数不变性证明；
- 组合不等式、整数多重性、以及循环图的窗口链分析；
- 经典的ruling set与MIS算法（Linial, Cole‑Vishkin, Schneider‑Wattenhofer 等）来构造锚点；
- 随机化分段（cut‑based）划分、均匀随机相位与切点随机选择，以实现零均值误差；
- 误差集中（Chernoff、Hoeffding）与依赖图分层技术；
- twisted fiber product构造与de Bruijn映射，用于在不同直径尺度上复制图结构。

**📊 数据集**

该工作为纯理论分析，无需实测数据集。所有结果均通过构造显式图族（如匹配团环、twisted fiber products、de Bruijn clique cycle）验证，并以n、Δ、D等参数描述复杂度。

**📈 对比分析**

比较方法：
- 对于精确等量着色，给出确定性Ω(n)下界与Ω(D)下界，并给出匹配的上界O(D)（直接全局收集）。
- 对于近等量误差ε，给出确定性Ω(n/ε)下界与随机化O((n/ε)·log^* n)上界，证明随机化能将误差系数提升到近似等量。 
- 对于松弛误差（1±η）等量，给出随机化O(log(Δ/η)+log^3log n)上界，且确定性仅需O(log^* n)（η≥Θ(√(log n)/Δ)）。
- 进一步表明，在所有Δ≥3、所有可达直径尺度内，精确等量着色的确定性复杂度为Θ(D)，而近等量着色在直径无关时间内可完成。

**⚠️ 局限性**

局限性与未解问题：
- 对于真正的随机游走/扩散网络（谱扩张图）仍未证明Ω(D)的下界；
- 需要更精细的随机化技术来进一步降低近等量着色的时间；
- 对于更宽松的度/直径组合（如极大度≈n/Δ），仍需更细的上界。

---

## 547. An $\widetilde{O}\left(n^3 \right)$-Time Sampler for Zero-Field Ferromagnetic Ising Models

**arXiv ID:** 2609.26197 | [PDF](https://arxiv.org/pdf/2609.26197v1)

**作者:** Weiming Feng `[一作]` (University of Hong Kong), Yiyao Zhang `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在任意无外场铁磁Ising模型上提出了一种近似抽样算法，其运行时间为 O(m+n)+O_β(n³ log³(1/ε))，在简单图上可简化为 O_β(n³ log³(1/ε))。

**💡 创新点**

创新点在于首次将 Benczúr‑Karger 的割稀疏化技术与 Chen‑Feng‑Ju‑Miao‑Yin‑Zhang 的传输流框架结合，用以近似 Ising 分布并显著降低 Glauber 动态的混合时间。

**🔧 技术方法**

核心技术包括：
- Benczúr‑Karger 割稀疏化与强度估计；
- Edwards–Sokal 关联随机簇模型与 Ising 模型；
- 传输流（transport flow）与 Log‑Sobolev 常数分析；
- 单边热浴 Glauber 动态与全动态连通性数据结构。

**📊 数据集**

本工作为理论算法研究，不使用实验数据集；所有结果均为数学证明与上界分析。

**📈 对比分析**

与原 Jerrum‑Sinclair 算法相比，时间从 O(m²n⁴(m+log⁻¹))（可达 O(n¹⁰)）降至 O(n³ log³(1/ε))，实现了显著的速度提升；在稠密图上尤为明显。

**⚠️ 局限性**

局限性包括：
- 仍未达到近线性时间，混合时间上界可能可以进一步改进但不太可能突破到 O(poly(n))；
- 复杂度对 β 的依赖较强（常数因子取决于 β）；
- 证明中部分步骤依赖自动化工具（如 GPT‑5.6），需进一步人工验证。

---

## 548. Beyond Imitation: Auditing the Recoverability of Reasoning in Distilled Models

**arXiv ID:** 2609.26216 | [PDF](https://arxiv.org/pdf/2609.26216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 549. SceneTTS-Bench: A Benchmark for Scene-Level TTS in Drama Dubbing

**arXiv ID:** 2609.26255 | [PDF](https://arxiv.org/pdf/2609.26255v1)

**作者:** Yizhong Geng `[一作]` (Beijing University of Posts and Telecommunications), Ya Li `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SceneTTS-Bench，一套针对戏剧配音的场景级 TTS 评测基准。

**💡 创新点**

创新点在于：①以场景为单位定义三大评估维度（音色一致性、情感表达、节奏连贯性）；②采用统一的 Canonical IR 让不同后端公平比较；③设计了三种自动诊断指标 SCS、UAR、RDR，能够定位并量化常见失效。

**🔧 技术方法**

使用技术包括：Canonical IR + dispatch adapter、CAM++ 语音嵌入 + HDBSCAN 聚类、Wav2Vec 2.0 情感回归、ASR+VAD 计时、阈值统计等。

**📊 数据集**

基准数据集为 160 个双语戏剧场景（约 10,300 句），由 100 场真实剧本和 60 场 LLM 生成剧本构成，涵盖中英两种语言。

**📈 对比分析**

通过对四个 TTS 系统（CosyVoice3、Qwen3‑TTS、IndexTTS2、Fish‑S2）的 SCS、UAR、RDR 评测，发现没有单一系统在三维度上均为最佳；相较句子级指标，场景级评估揭示了显著的排名差异，进一步说明了系统在音色稳定、情感强度和节奏连续性方面的不同表现。

**⚠️ 局限性**

局限性包括：①仅诊断音色漂移、情感低强度和节奏跳跃，未覆盖音高、能量、停顿等细节；②阈值和指标对语料库特性高度依赖，跨语言或新领域需重新校准；③情感指标主要关注不足的强度，未能评估过度表达或自然性。

---

## 550. Which Reranking Conclusions Survive the Answer Interface? A Prospective Finite-Orbit Audit

**arXiv ID:** 2609.26250 | [PDF](https://arxiv.org/pdf/2609.26250v1)

**作者:** Wenzhang Du `[一作]` `[通讯]` (Independent Researcher), Wenzhang Du (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 BM25 与 BGE-reranker 的检索策略在不同答案接口（标签绑定、词汇、选项顺序）下，通过下游读者（多语言模型）进行评估，并检验接口变动是否会改变检索结论。

**💡 创新点**

提出将检索评估拆分为四个独立维度：策略层级、接口稳定性、策略排序和下游选择值，并在完整的 2^3 因子实验中对每个维度进行定量检验，首次证明接口变动不必然导致检索结论的“实质性”变化。

**🔧 技术方法**

使用强制式完整序列似然得分、2^3 因子设计、Walsh 分解、Bootstrap 同时置信区间、预设材料性阈值（0.015）以及两级检验（整体与交互）等统计方法。

**📊 数据集**

采用 RAGuard 与 FEVER 两个公开事实核查数据集，使用四个开源读者（Qwen2.5‑3B、Mistral‑7B、Llama‑3.1‑8B、Gemma‑3‑4B）进行评估。

**📈 对比分析**

比较方法：计算每个接口下 BM25 与 BGE 的平均提升（g）和判决差异（h），并评估接口平均、交互项、排序一致性及选择器在 held‑out 上的表现。结果显示六个确认环境均未出现超过材料性阈值的接口变异，未出现策略顺序逆转，也未出现显著的 held‑out 价值差异。

**⚠️ 局限性**

局限性包括：仅使用强制式得分而非自由生成，接口变动范围有限（仅三种任务等价因子），读者模型规模有限，检索深度固定且未覆盖实时 Web 检索，样本覆盖度与数据集标注质量相关，未考察其他检索策略或更大范围的词汇/提示变体。

---

## 551. Topology-Aware Parameter-Efficient Adaptation for Cross-Dataset Retinal Vessel Segmentation

**arXiv ID:** 2609.26189 | [PDF](https://arxiv.org/pdf/2609.26189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 552. Approximate Counting of $k$-Paths in $O^*(2^k)$ Time

**arXiv ID:** 2609.26246 | [PDF](https://arxiv.org/pdf/2609.26246v1)

**作者:** Tomohiro Koana `[一作]` `[通讯]` (University of Tokyo), Tomohiro Koana (University of Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出了一种随机化算法，在 O*(2^k ε⁻² log(1/δ)) 时间内对有向图中的 k‑路径数量进行 (1±ε) 近似计数，并将该方法推广到集合打包问题。

**💡 创新点**

突破了此前仅能在 O*(4^k) 或 O*(2.619^k) 时间内实现的近似计数，实现了与决策版本同样的指数上界 2^k；同时利用右斜电路、外扩张代数与随机符号矩阵压缩的组合技术，构造了高效的估计器。

**🔧 技术方法**

核心技术包括：非交换多项式与右斜电路的构造、外扩张代数中的外积编码、Nyquist‑Rice‑Riordan 矩阵行列式矩阶矩、随机符号矩阵压缩、以及中值‑均值法进行误差放大。

**📊 数据集**

本文未使用实验数据集，而是对理论算法给出了运行时间与误差保证的分析。

**📈 对比分析**

与以往的 O*(4^k) 与 O*(2.619^k) 近似计数方法相比，算法在指数因子上取得了理论上的最优 2^k；误差放大仍需 O(ε⁻² log(1/δ)) 次采样，且算法仅在随机化环境下可行。

**⚠️ 局限性**

主要局限包括：仍然是指数级时间，无法用于非常大的 k；仅提供近似计数，无法得到精确计数；算法依赖随机化与大整数运算，实际实现可能受限于数值精度与计算资源。

---

## 553. When to Stop? Dynamic Early Termination of Sequential Ensembles

**arXiv ID:** 2609.26215 | [PDF](https://arxiv.org/pdf/2609.26215v1)

**作者:** Paul Bezner `[一作]` (University of Stuttgart), Stephan ten Brink `[通讯]` (University of Stuttgart)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种针对序列化集成解码器的动态早停（Dynamic Early Termination, DET），在每个成员解码后估计未评估成员能否纠正当前决策的风险，从而决定是否提前停止。

**💡 创新点**

创新点在于：① 用残差风险（residual risk）来衡量提前终止的概率，并用离线拟合的阈值控制相对FER损失；② 通过风险估计替代传统的基于综合量（syndrome）或第一有效（first-valid）停止准则；③ 将该框架与行增益RBE、分层归一化最小和（NMSA）以及条件OSD结合，实现硬件友好的两核流水线。

**🔧 技术方法**

使用技术包括：分层归一化最小和迭代 BP、行增益RBE（Row‑Boosted Ensemble）与单行提升、条件OSD（Ordered‑Statistics Decoding）对未通过BP的输出进行后处理、基于梯度提升树的残差风险回归、离线阈值搜索（error‑allocation）以及两核带缓冲的调度实现。

**📊 数据集**

使用的数据集为 IEEE 802.11 LDPC 代码：(648,540)、(648,567)、(1296,1080)、(1296,1152)、(1944,1620)、(1944,1728)，速率为 1/2 与 5/6，采用 AWGN 信道。

**📈 对比分析**

与全列表、平行全集、纯DET、first‑valid、BP‑20、BP‑1280 等基线进行比较。实验结果显示，在设计点（FER≈10⁻³）下，风险DET平均仅需 1.04–1.15 个成员，BP 计算量比全列表低 39×，OSD 调用次数低 18×，FER 与全列表相差不超过 7% 的目标损失；在 (648,540) 代码上，获得约 0.4 dB 的 SNR 提升。

**⚠️ 局限性**

局限性包括：① 需要在每个码字上离线训练和阈值拟合，适用于固定码；② 仅在 AWGN 或块衰落下验证，频谱/时变信道的适应性未知；③ 对残差风险估计的准确性高度依赖所选特征和树模型，可能在不同 SNR 或码率下表现不一致；④ 通过两核流水线实现时需精确的缓冲尺寸设计，若到达率过高会导致 FER 损失。

---

## 554. Partially Observed Sparse Graphs: The Unknown Sampling Rate is a Tail Index

**arXiv ID:** 2609.26199 | [PDF](https://arxiv.org/pdf/2609.26199v1)

**作者:** Jian Xu `[一作]` (RIKEN iTHEMS), Qibin Zhao `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在已知目标网络规模但未知采样率的情况下，如何通过稀疏可交换图（graphex）模型估计全局边数，并将问题归约为尾指数（tail index）的估计；

**💡 创新点**

将稀疏图的边数稀疏指数与尾指数联系起来，证明采样率可通过节点计数与尾指数恢复，提出模块化的估计框架，并提供几何路径闭包与聚类上限的理论界定；

**🔧 技术方法**

使用graphex理论、尾指数非参数估计（Hill、闭式估计）、几何路径的条件流匹配（conditional flow matching）、Poisson过程分析与复合似然求解；

**📊 数据集**

13 个真实网络（如 AS、P2P、合作网络）以及 5000 条合成 graphex；

**📈 对比分析**

与多种基线（图平面、Hill 估计、对数对数回归、已知采样率的设计基估计等）比较；在稀疏 regime 下平均误差降低至约 21–26%，相较于图平面 260% 甚至 474% 的误差；但在较高密度或爬行采样场景下性能显著下降；

**⚠️ 局限性**

仅适用于均匀 p‑采样；在随机游走或 snowball 爬行下失效；聚类上限限制了 rank‑one 方案；尾指数估计在尾部不足时饱和，且对真实网络的有效尾指数不一定稳定；

---

## 555. Modality-Gated Deep Adapters: Adding a Modality to a Frozen Embedding Model with Exact Preservation

**arXiv ID:** 2609.26182 | [PDF](https://arxiv.org/pdf/2609.26182v1)

**作者:** Abdul Basit Tonmoy `[一作]` (Eximius Labs), Arman Luthra `[通讯]` (Eximius Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在已冻结的多模态嵌入模型上通过模态门控深度适配器（Modality‑Gated Deep Adapters）添加音频和热成像两种新模态，同时保证所有未被新模态覆盖的输入在推理时的输出与原始模型完全一致（逐位相同）

**💡 创新点**

提出的门控适配器机制在训练后依旧保持对原始模型的严格输出不变性，且不需要任务标签或路由元数据，能够让多模态模型在保留原有行为的同时新增模态，解决了传统 LoRA 等参数高效方法导致的模型漂移问题

**🔧 技术方法**

利用逐层的瓶颈适配器（SiLU+LayerNorm+投影），通过在每层的前向钩子上加上二值门控制是否执行；门基于输入中是否出现音频/热成像标记；采用冻结基模型权重、对齐的文本目标、信息对比（InfoNCE）损失，保证梯度仅流向适配器而不影响基模型

**📊 数据集**

主要使用 AudioCaps、Clotho、VGGSound（音频文本检索）、LLVIP（热成像零样本分类）、IR‑TD（热成像图像+描述）等公开数据集进行训练和评估；训练集约 518k 对音频‑文本对，微调阶段 400 步；实验也用 45k AudioCaps 作为对照"受控门实验"

**📈 对比分析**

通过对比带/不带门控适配器的模型，在相同训练步骤、数据、超参数下，音频检索 mAP@10 提升约 +3.4–+5.4 点，热成像检索 mAP@10 从 0.224 提升至 0.785，且对文本/图像/视频的输出保持完全一致；与传统 LoRA、Flamingo 等方法的比较显示门控适配器在保持原有行为的前提下实现了显著性能提升

**⚠️ 局限性**

实验仅在单一 2B Qwen3‑VL 基础模型上验证，未探测不同基模型或更大规模的可迁移性；门控机制对每个新模态单独使用，未验证多模态（>2）同时加载时的可扩展性；在音频上仍受限于冻结模型容量，未能突破专用双编码器的性能；对硬件/驱动差异下的逐位一致性仅在相同推理配置下验证，未覆盖跨设备的数值差异

---

## 556. CRT-Decomposed $Σ$-Protocols for CSIDH

**arXiv ID:** 2609.26258 | [PDF](https://arxiv.org/pdf/2609.26258v1)

**作者:** I. Dey `[一作]` (South East Technological University), I. Cherkaoui `[通讯]` (South East Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了针对CSIDH群作用的零知识知识证明协议，并利用理想类群的中国剩余定理（CRT）分解实现了无重播提取；

**💡 创新点**

创新点在于：①利用CRT分解实现闭式算术提取，完全不依赖重播或随机采样；②通过Unruh变换在量子随机预言机模型（QROM）下得到非交互式零知识证明，并实现在线提取；③对完整的代数层进行了机器验证，证明了协议的完整性、零知识性和特殊可提取性；

**🔧 技术方法**

技术主要包括：CSIDH群作用、理想类群的CRT分解、ZK知识证明框架、Unruh变换、量子随机预言机模型、线性代数提取、机理验证、Monte‑Carlo 模拟与 meet‑in‑the‑middle 攻击分析；

**📊 数据集**

使用的数据集为：1) 10,000 个随机实例用于代数层机器验证；2) 4×10⁵ 次协议执行用于模拟知识错误；3) 多个子群（2¹⁰到2²²）用于验证 meet‑in‑the‑middle 攻击的√q 复杂度；

**📈 对比分析**

与 CSI‑FiSh、CSI‑Otter、Tanuki 等现有同类方案比较，签名尺寸为384B（k=5）或11.6KB（k_eff=2）且无重播损失；但相较于 CSI‑FiSh（263B）和 CSI‑Otter（8KB/4KB）在签名尺寸与提取开销上略大；在经典攻击下，CSIDH‑512 的安全指数约为 2⁶⁷.³，远低于 2¹²⁸ 目标；仿真结果与理论 2^‑t 知识错误界限完全吻合；

**⚠️ 局限性**

局限性：仅适用于已知结构的理想类群；CSIDH‑512 在经典安全上只能达到约 2⁶⁷ 次群作用；公布 hop 曲线会将安全指数降至 √q_max；协议无法通过 CRT 缩短轮数；在量子攻击下，安全性受 Kuperberg 树形筛法限制；未来需要更大且质因子更大的参数集合。

---

## 557. eBPF Security in the Wild: Structural Concentration, Failure Mechanisms, and Discovery Gaps

**arXiv ID:** 2609.26254 | [PDF](https://arxiv.org/pdf/2609.26254v1)

**作者:** Baihong Chen `[一作]` (Utah State University), Wen Li `[通讯]` (Utah State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了基于内核提交、syzbot、CVE 的统一数据集，对 eBPF 实际漏洞进行系统性实证分析，探究其结构分布、机制根因、架构集中度及现有发现技术的覆盖差距。

**💡 创新点**

创新之处在于首次从多源数据重构完整漏洞空间，建立统一的 CWE 层次分类框架，并通过四个研究问题将漏洞结构、机制、架构和发现能力统一关联，揭示了 eBPF 漏洞的结构集中与语义稀疏性。

**🔧 技术方法**

采用数据融合、CWE 层次标注、统计结构聚类、机制级根因分析、组件级分布计数以及基于 KCOV 的覆盖与种子语义跟踪，并对 Syzkaller、Buzzer、BRF 进行设计级对比。

**📊 数据集**

共收集 2,766 条漏洞实例，来源于 2,439 条内核修复提交、197 条 CVE/NVD、130 条 syzbot 报告，并按 CWE 1000 层级进行划分。

**📈 对比分析**

通过版本对齐的 Syzkaller 72h 演练与设计级 Rubric 比较，评估原始覆盖率、有效种子比例、语义多样性，结果显示 Raw 覆盖虽高，但实效仅集中在 Runtime 并发相关漏洞，发现率低于 1% 的总体漏洞。

**⚠️ 局限性**

研究受限于数据来源偏重内核提交、CVE 公开与 syzbot 报告不均衡、CWE 标注主观性、仅评估 Linux v5.10、Syzkaller 单一案例、覆盖指标无法完全映射语义深度，因而结果可能不具备跨版本或跨工具的普适性。

---

## 558. MSA-CITE: A Co-Adapted LoRA Specialist Ecology for Fixed-Budget Small-Model Inference

**arXiv ID:** 2609.26217 | [PDF](https://arxiv.org/pdf/2609.26217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 559. Damage Predicts Recovery: When Calibration Data Matters in Compressing Financial LLMs

**arXiv ID:** 2609.26241 | [PDF](https://arxiv.org/pdf/2609.26241v1)

**作者:** Junyi Ye `[一作]` (Montclair State University), Guiling Wang `[通讯]` (New Jersey Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在金融领域使用后训练量化和稀疏化压缩大型语言模型时，校准语料对任务性能的影响，并提出了一种先评估压缩损伤再决定是否使用领域匹配校准语料的实用规则。

**💡 创新点**

创新点在于：①把压缩造成的任务级损伤视为决定是否使用领域匹配校准语料的核心指标；②在金融NLP任务中系统对比量化与稀疏化的损伤差异；③发现量化对任务几乎无影响，而稀疏化尤其损害数值推理任务，并证明在损伤显著时，使用金融任务格式化的FinMix语料能显著恢复性能。

**🔧 技术方法**

技术包括：后训练压缩（4‑bit量化：RTN、GPTQ、AWQ；8‑bit量化：SmoothQuant；稀疏化：SparseGPT 50%无结构稀疏、Wanda 4:8半结构稀疏）以及对比实验所用的三种校准语料（WikiText、C4、FinMix）。

**📊 数据集**

数据集涵盖十个公开金融基准任务，涵盖情感/立场分类、关系识别、因果推断、方向预测、表格与文本混合的数值问答等，均来自 PIXIU、FinBen 等公开仓库。

**📈 对比分析**

方法：在每个压缩配置下，使用三种校准语料对模型进行评估，并与 BF16 基线在同一测试集上做配对统计检验；通过 DeepSeek 进行自由文本数值问答的自动评分。性能方面：量化基本保持 BF16 评分；稀疏化导致数值问答下降 40+ 分，分类受损较轻；使用 FinMix 时可在稀疏化环境下恢复 20–30 分（具体任务差异显著）。

**⚠️ 局限性**

限制：FinMix 同时包含内容和格式变化，无法单独评估两者效果；规模实验仅包含单一 70B 模型；研究聚焦金融领域，是否同样适用于其他专业领域仍需验证。

---

## 560. LLaVA-Assessor: Building the Foundation LMM For Visual Quality Assessment

**arXiv ID:** 2609.26205 | [PDF](https://arxiv.org/pdf/2609.26205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 561. KeyBound: Keyed and Host-Bound Learned Audio Watermarking for Speech Provenance

**arXiv ID:** 2609.26235 | [PDF](https://arxiv.org/pdf/2609.26235v1)

**作者:** Bangshuo Zhu `[一作]` (University of New South Wales), Wei Song `[通讯]` (University of New South Wales)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `b88c6eac-d57a-4623-a604-1f401f3eb268` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文提出了一种键控、主机绑定的音频水印方案，能够在遭受噪声、滤波、重采样、压缩以及重合成等多种攻击后仍能被检测并正确恢复有效载荷；

**💡 创新点**

创新点在于将经典水印的“密钥加密”和“主机感知载体”结合到学习型后置水印中，并在训练时加入重合成攻击通道，以实现对重合成、无密钥读取与植入转移的抵抗；

**🔧 技术方法**

采用端到端卷积生成器与检测器、FiLM 进行主机感知调制、XOR密钥掩码、固定 log‑mel 前端以及多尺度声谱和心理声学损失的联合训练；

**📊 数据集**

主要使用 LibriSpeech（train‑clean‑100 训练集，test‑clean 评测集）进行训练和测试；

**📈 对比分析**

与三种公开基线（WavMark、AudioSeal、Timbre）对比，KeyBound 在所有信号失真和重合成攻击下均保持 100% 检测准确率，位错误率在无密钥条件下接近 0.5，且对植入转移的抵抗率几乎为 0；

**⚠️ 局限性**

局限性包括：对自适应移除攻击（如训练好的重建网络）仍易被击穿、在更长或非英语语音域的鲁棒性尚未验证，以及在极低 SNR 或极大容量需求下的可扩展性问题。

---

## 562. Bridging the Data Gap: Digital Twin as a New Paradigm for AI-based Radio Sensing

**arXiv ID:** 2609.26214 | [PDF](https://arxiv.org/pdf/2609.26214v1)

**作者:** Éloi Sainte-Beuve `[一作]` (Orange Research), Ali Al Khansa `[通讯]` (Orange Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建数字孪生（DT）环境，利用射线追踪生成大量带有时间标签的合成信道冲激响应（CIR），并训练序列神经网络将CIR序列映射为空间占用热图，实现无设备定位。

**💡 创新点**

提出将DT置于感知训练循环中，以场景特定的合成数据代替人工标注；同时设计了基于状态空间模型（SSM）的序列网络，兼顾时空依赖并提升定位精度。

**🔧 技术方法**

使用Sionna Ray‑Tracing射线追踪器生成CIR；构建基于注意力的SSM架构，融合RSSI先验；与MLP、CNN、ViT等基线模型在同一框架下对比评估。

**📊 数据集**

数据集为两种合成房间（方形和U形）中随机生成的10步CIR序列，共2000条序列，包含1至3名移动人员的轨迹与占用标注。

**📈 对比分析**

在两种环境下进行k折交叉验证，SSM在置信圈（CC）和余弦相似度（CS）上分别取得51.6/94.3%和48.3/95.3%的最高分，显著优于ViT、CNN和MLP基线。

**⚠️ 局限性**

局限在于仅做了仿真实验，缺乏真实硬件验证；DT的几何精度、材料建模和射线追踪质量等对 sim‑to‑real 转移的影响尚未评估；并且仅覆盖两种房间几何，需进一步扩展多样化环境。

---

## 563. A Multi-Timestep LSTM Ensemble regressor for Enhanced Short-Term Runoff Prediction

**arXiv ID:** 2609.26244 | [PDF](https://arxiv.org/pdf/2609.26244v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 564. A Hybrid AI Framework for Academic Advising: Integrating Ensemble-Based Grade Prediction and a Rule-Based Expert System

**arXiv ID:** 2609.26243 | [PDF](https://arxiv.org/pdf/2609.26243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 565. Quantum-Ready Secure WAN: A Risk Assessment and Migration Framework

**arXiv ID:** 2609.26225 | [PDF](https://arxiv.org/pdf/2609.26225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 566. A Semantic Approach to the Academic Publishing Network: Document Vector Representations and Hybrid Structural-Semantic Fusion over OpenAlex Data

**arXiv ID:** 2609.26218 | [PDF](https://arxiv.org/pdf/2609.26218v1)

**作者:** Robert Šamárek `[一作]`, Radek Martinek `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将结构化学术出版网络分析与基于SPECTER2的语义嵌入结合，提出可调权重的结构–语义融合层，用于改进文献推荐。

**💡 创新点**

提出可调参数α的晚期融合机制，使得在不同任务中可灵活权衡结构与语义信号，而非单一优胜方法。

**🔧 技术方法**

采用SPECTER2生成引用信息增强的文档向量，LanceDB向量数据库进行近似检索，并通过Jaccard相似度衡量结构性引用耦合。

**📊 数据集**

使用捷克奥斯特拉瓦技术大学2020–2025年的7,317篇论文和OpenAlex元数据构建的本机构语料。

**📈 对比分析**

通过NMI/ARI评估嵌入聚类与专家OpenAlex分类的一致性，以及在推荐任务中对结构、语义和融合三种信号进行NDCG@10、MAP、MRR比较，发现语义和融合在大多数情况下优于纯结构，但融合优势并不显著。

**⚠️ 局限性**

主要局限包括缺失摘要导致语义信号弱、使用子领域作为相关性代理的粗糙评估、CPU嵌入对大规模语料的可扩展性不足以及向量余弦相似度基线偏高。

---

## 567. Identifying Intelligent Processes via Online Sequential Testing

**arXiv ID:** 2609.26193 | [PDF](https://arxiv.org/pdf/2609.26193v1)

**作者:** Aritra Das `[一作]` (Truth Audit Labs), Debayan Gupta `[通讯]` (Truth Audit Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在已知候选语言模型集合的情况下，提出了两阶段主动序列假设检验框架：先设计低成本的探测工具生成唯一指纹集，再在此基础上进行预算最小化的序列识别。

**💡 创新点**

创新点在于把探测工具设计视为加权集合覆盖问题，给出了基于校准样本的一次性估计方法，并提供了从对偶分离度推导的内在样本复杂度上界。

**🔧 技术方法**

使用的技术包括主动序列假设检验、加权集合覆盖求解、Jensen‑Shannon 散度与 Pinsker 不等式、Hoeffding 近似、整数规划与近似覆盖算法、一次性校准抽样。

**📊 数据集**

实验使用了从候选模型中抽取的校准样本作为响应，未给出公开数据集名称，假设使用标准语言模型基准集进行模拟。

**📈 对比分析**

与传统固定探测或无设计的主动检验相比，该方法在保证误差率的前提下显著降低了探测次数与构造成本；理论上给出了 M−1·⌈8/η² log(M/δ)⌉ 的上界，证明了效率提升。

**⚠️ 局限性**

局限性包括需预先知道候选模型集合；校准样本量对误差阈值敏感；对高维文本响应的摘要可能导致信息损失；未在真实多模态交互环境中验证实际性能。

---

## 568. Who Assures the Verifier? An Executable Assurance-Locus Audit of the European Digital Identity Wallet

**arXiv ID:** 2609.26220 | [PDF](https://arxiv.org/pdf/2609.26220v1)

**作者:** Anton Sokolov `[一作]` `[通讯]` (Tyche Institute), Anton Sokolov (Tyche Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对欧盟数字身份钱包（EUDI Wallet）中的依赖方（RP）验证责任进行系统化分析，提出17条拒绝规则和JSON evidence receipt，并用36个合成交易在Python、JavaScript、jq三种实现上验证可重复性，随后对三个公开验证器源码进行审计。

**💡 创新点**

提出了“RP‑as‑SUT”证据单元，填补了从注册、验证实现到业务依赖决策之间的证据缺口，提供可独立重跑的 evidence receipt，展示了跨实现的一致性与确定性。

**🔧 技术方法**

采用规范交叉映射、研究配置文件、JSON schema、合成交易生成器、三种编程语言实现、静态源代码审计等技术手段。

**📊 数据集**

使用36个合成交易（包含6个合法、30个违规）作为测试集，三条实现路径产生108次执行结果；同时对EC EUDI verifier、OpenWallet Multipaz和walt.id三份公开源代码进行审计。

**📈 对比分析**

通过与预设oracle比较，108次执行全部匹配且不同实现间无分歧，证明了决策模型的可实现性和确定性；性能指标未进行实测，仅关注逻辑正确性。

**⚠️ 局限性**

局限在于使用合成布尔式输入未覆盖真实协议解析、密码学、网络与业务流程；源代码审计样本有限，未验证部署环境；缺乏对真实数据与隐私保护的考量；实验未评估性能或对抗攻击的鲁棒性。

---

## 569. PatchKV: Efficient KV Cache Recovery for Dynamically Edited LLM Contexts

**arXiv ID:** 2609.26219 | [PDF](https://arxiv.org/pdf/2609.26219v1)

**作者:** Guotao Yang `[一作]` (Tianjin University), Keqiu Li `[通讯]` (Tianjin University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究动态上下文中的中间编辑对KV缓存的影响，提出PatchKV系统实现增量恢复；

**💡 创新点**

利用离线长度条件漂移预测、注意力引导的稀疏重建以及块级量化传输，精确定位需要重建的KV块并高效恢复剩余块；

**🔧 技术方法**

长度条件漂移预测模型、注意力统计与块对齐、RoPE校正、FP16/INT8/INT4量化与融合的解量化+RoPE+页面放置恢复内核；

**📊 数据集**

三大指令调优模型Qwen3-32B、GLM-4-9B、Llama-3.3-70B；长上下文问答数据集HotpotQA-E、2WikiMQA-E、TriviaQA-E；

**📈 对比分析**

与Prefix Cache、Direct KV Reuse、CacheBlend对比，PatchKV在TTFT上比Prefix Cache快2.51–3.85倍，比CacheBlend快1.26–2.06倍；F1得分与CacheBlend相当或更好，在六/九个场景中超过；

**⚠️ 局限性**

需要模型特定的离线校准和精度阈值，收集注意力统计的开销，适用于保留相同后缀的中间编辑场景，未覆盖更广泛的分布式或多机恢复情况。

---

## 570. RCShift: Certifying When Partial Linkage Suffices for Finite-Sample Decisions

**arXiv ID:** 2609.26207 | [PDF](https://arxiv.org/pdf/2609.26207v1)

**作者:** Shuheng Cao `[一作]` (University of California San Diego), Tingting Dan `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在给定观测合同的前提下，研究如何在仅保留部分记录链接的情况下，以最小成本实现统计决策的充分性。

**💡 创新点**

提出 RCShift 三种证明模式，区分“family‑exact”存储与有限样本决策保持，并通过方向对决策价值和正逆缺陷不必增加样本量的对比，展示存储决策的方向性。

**🔧 技术方法**

运用了离散计数实验、LR 可见循环方向、Blackwell 等价、Le Cam 缺陷、hockey‑stick 收缩、整数规划与证书检查等技术。

**📊 数据集**

主要使用了七边支撑图的模拟例子以及调度问题的实验实例，未使用公开真实数据集。

**📈 对比分析**

通过精确枚举与证书上界/下界比较，在单位成本下证明对齐计数可将最小样本从 4 降至 1，非对齐计数需要 11，正逆缺陷扰动仍保持 4；计算可行但最优搜索属于 NP 难问题。

**⚠️ 局限性**

需预先给出完整的 LR 基、参数空间正性与可评估缺陷上界，且最优搜索在大规模实例中不可行；方法仅适用于已声明的观测合同和固定成本。

---

## 571. TriWorldBench: A Tri-View Consistency Perspective on Embodied World Models

**arXiv ID:** 2609.26314 | [PDF](https://arxiv.org/pdf/2609.26314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 572. DHSched: Stateless Control for Stateful Real-Time Avatar Serving

**arXiv ID:** 2609.26363 | [PDF](https://arxiv.org/pdf/2609.26363v1)

**作者:** Xin Wang `[一作]` (AiShiWeiLai Co Ltd), Zhenyu Xu `[通讯]` (Sichuan University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DHSched 控制平面，利用无状态 Dispatch 副本和基于世代的所有权协议，管理 GPU 驱动的长生命周期实时头像服务，并实现了容量安全的放置与源独立的故障恢复。

**💡 创新点**

核心创新在于：① 将所有权拆分为世代（WorkerID, epoch）并在输入路径上进行生成器级别的验证，消除双主冲突；② 采用副本本地负载校正、跨池水线和底层分层采样，保证在缓存视图失效时仍能安全放置；③ 在恢复时通过源 epoch 过滤，支持无源迁移而无需转移 GPU/RTC 状态。

**🔧 技术方法**

实现技术包括：Redis 原子 Lua 脚本实现条件更新与容量预留；短生命周期实体锁保障并发操作顺序；Infer‑Controller 在每条输入流上执行世代校验；Dispatcher 采用副本本地缓存、动态负载补偿与随机采样进行候选排序。

**📊 数据集**

实验数据集为生产头像服务的实时流数据，峰值并发约 49,987 条会话，6 月 10 日完成 9,927 次会话迁移；对比实验使用了 Slicer、Orleans、TurboServe、Llumnix 等系统的公开评测数据。

**📈 对比分析**

对比方法：在单所有权、容量溢出、放置分散、恢复延迟等指标下，DHSched 在 100,000 次随机运行中零双主冲突、零旧世代释放；与 Slicer 的强一致性特性相比，DHSched 通过 epoch 也实现零冲突；容量测试显示 Atomic Reserve 在高并发下保持 0% 超额；恢复实验表明源 epoch 过滤在高延迟情况下保持 0% 恢复错误。

**⚠️ 局限性**

限制与挑战：① 仅适用于可外部化可恢复状态的会话；② 需要可靠的线性化共享存储作为所有权与容量的单点；③ 对缓存视图的时延敏感，极端高并发或极慢刷新时仍可能出现放置峰值偏差；④ 目前未对非 GPU 资源（如网络 I/O）做细粒度控制；⑤ 在极度频繁的迁移场景下，世代递增会导致 epoch 存储增长与验证开销提升。

---

## 573. Rethinking Pairwise Token Interaction in Spiking Transformers

**arXiv ID:** 2609.26297 | [PDF](https://arxiv.org/pdf/2609.26297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 574. HYDRA: Proactive Android Malware Drift Adaptation via Hierarchical Graph Contrastive Learning

**arXiv ID:** 2609.26352 | [PDF](https://arxiv.org/pdf/2609.26352v1)

**作者:** Han Chen `[一作]` (University of Technology Sydney), Ying Zhang `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于层次化图对比学习的主动适应框架（Hybrid Drift Adaptation），用于解决Android恶意软件概念漂移问题。

**💡 创新点**

创新点包括：①将细粒度的控制流图(CFG)与粗粒度的函数调用图(FCG)组合成层次化图结构，捕获局部与全局行为；②设计跨域对比学习目标，利用真实标签与伪标签在同一优化过程中实现域对齐与类别分离，避免了不稳定的对抗式训练；③通过伪标签与对比损失的联合学习，实现低标签预算下的高效迁移。

**🔧 技术方法**

技术手段包括：层次化图神经网络（CFG层采用聚合信息传递，FCG层采用GIN），跨域对比学习（InfoNCE），伪标签生成与监督分类损失的联合优化，以及对图结构的批处理与可扩展的图抽取流程。

**📊 数据集**

使用了AndroZoo 2012-2022大规模Android应用数据集（约500k个样本），以及2022-2025的扩展数据集，标签来自AVClass2与VirusTotal；通过时间序列划分（训练年→测试月）和Hold-out家庭实验验证模型。

**📈 对比分析**

与多种基线（主动学习如HCC、CADET、ADDA、DREAM，经典分类器SVM/MLP，以及仅CFG或仅FCG的变体）进行对比。结果显示：在各种预算下，Hybrid Drift Adaptation的F1、FNR和FPR显著优于基线，尤其在标签稀缺（≤100）时提升约5–10% F1，FNR下降约8–15%；并在CFG被破坏或新恶意家族出现时保持鲁棒性。

**⚠️ 局限性**

局限性包括：①对抗性攻击对层次化图的破坏可能导致伪标签错误，进而引入数据投毒风险；②图抽取与模型训练的计算成本仍高，需进一步优化缓存或蒸馏；③伪标签策略对极度不平衡或噪声标签的数据仍敏感，未来需加入更鲁棒的自监督或主动选择机制。

---

## 575. End-to-End Visual Odometry with RNNs and Attention

**arXiv ID:** 2609.26188 | [PDF](https://arxiv.org/pdf/2609.26188v1)

**作者:** Ruiyu Li `[一作]` (Carnegie Mellon University), Alexander Yu `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究手持相机下的视觉里程计，提出基于时间注意力和Transformer的模型以提升定位精度。

**💡 创新点**

创新点在于将全局自注意力层插入LSTM或完全替换为Transformer编码器，并通过分段训练解决长序列内存瓶颈。

**🔧 技术方法**

使用卷积特征提取、LSTM、全局自注意力、Transformer编码器，结合Dropout、L2正则及分段反向传播。

**📊 数据集**

使用TUM RGB‑D基准数据集（39条手持Kinect视频序列）进行训练与验证。

**📈 对比分析**

与Baseline Model 1相比，Transformer模型验证MSE降至2.51，对应平均位置误差2.24 m，显著优于Baseline的3.71 MSE（2.72 m）。

**⚠️ 局限性**

局限性包括仅使用少量数据、缺乏未来帧掩码导致只能离线推理、未进行超参数搜索、深度图噪声可能影响性能，以及注意力未采用稀疏化策略。

---

## 576. A Throughput-Oriented Analytical Model for Post-Quantum Security Protocols

**arXiv ID:** 2609.26284 | [PDF](https://arxiv.org/pdf/2609.26284v1)

**作者:** Ignazio Pedone `[一作]` (nodeQ Limited), Stefano Pirandola `[通讯]` (nodeQ Limited)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于吞吐量的解析模型，用于估算TLS和SSH等安全协议在采用后量子密码学（PQC）时的握手速率，并将其集成到网络模拟器中以实现资源分配和网络规划。

**💡 创新点**

创新点在于：①将握手过程分解为加密运算时间和网络传输时间，给出闭式上限；②提供端点与网络两端的服务率计算和瓶颈识别；③设计了端点资源比例缩放规则，实现多连接情况下的可行性保证；④将握手速率与带宽需求的双向映射集成到网络规划工具。

**🔧 技术方法**

使用的技术包括：C/C++/Python实现的解析模型；Superspeed等基准库提供的CPU周期计数；OpenSSL 3.5.4、Docker、Linux traffic control进行实验验证；利用Azure虚拟机和网络模拟器进行网络场景验证。

**📊 数据集**

数据集主要来源于实验测试平台：Azure VM（4 vCPU/3.25 GHz）、不同PQC算法（ML‑KEM‑1024、Frodo‑KEM‑1344‑AES、ML‑DSA 等）在 TLS/SSH 握手中的CPU周期与字节计数；网络容量通过TBF调节从 1 Mbps 到 100 Mbps。

**📈 对比分析**

比较方法是将模型预测的握手速率与实验测得的速率对比，显示两者在CPU‑受限与网络受限两种场景下均保持在 5–10 % 的误差范围；实验验证证明模型在高并发、高CPU利用率时的准确性和可扩展性。

**⚠️ 局限性**

局限性包括：①对每个算法的CPU周期需要精确测量，误差会影响预测精度（如 ML‑DSA 的方差较大）；②模型忽略了 RTT、丢包和协议层外部 I/O 的细粒度影响；③端点资源比例缩放是局部近似，无法保证全局最优或公平性；④仅验证了 TLS/SSH，其他协议需进一步扩展。

---

## 577. Image-Based Techniques and Ensemble Soft Voting for Malware Classification

**arXiv ID:** 2609.26281 | [PDF](https://arxiv.org/pdf/2609.26281v1)

**作者:** Sushant Rakesh Lokhande `[一作]` (San Jose State University), Mark Stamp `[通讯]` (San Jose State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在本文中，作者提出了一个多轨道特征提取与软投票集成框架，用于基于图像的恶意软件家族分类。

**💡 创新点**

创新点在于同时结合手工特征、预训练模型嵌入和自研CNN嵌入三类特征，并通过统计多样性分析验证其互补性，最终实现最高80.2%的准确率。

**🔧 技术方法**

使用了HOG、Haralick、ADV38等手工特征、VGG16/ResNet50/ViT的预训练网络、以及自研CNN提取嵌入，并采用随机森林、XGBoost、SVM等分类器以及软投票集成。

**📊 数据集**

实验基于Agrawal等人构建的17个恶意软件家族、每族1000样本的Malware图像数据集，共136,000张图像。

**📈 对比分析**

通过单模型与多模型（软投票）对比，并用Bootstrap置信区间和McNemar检验验证显著性，单模型最高77.8%，软投票提升至80.2%。

**⚠️ 局限性**

局限包括数据集规模有限、模型对不同转换的依赖程度、软投票使用等权重未优化、以及缺乏更细粒度的解释性分析。

---

## 578. Mode Collapse Is Cheap to Detect: A Ground-Truth-Free Pre-Flight Check for Neural Samplers

**arXiv ID:** 2609.26272 | [PDF](https://arxiv.org/pdf/2609.26272v1)

**作者:** Jian Xu `[一作]` `[通讯]` (RIKEN iTHEMS), Jian Xu (RIKEN iTHEMS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种针对神经采样器的预飞行检查，用仅基于目标能量、梯度和海森矩阵的二阶模式搜索与拉普拉斯近似，估计缺失质量并给出自我诊断。

**💡 创新点**

创新点在于：①仅使用目标的能量、梯度、海森矩阵而不需要目标采样或采样器密度；②将缺失质量拆分为支持损失和权重失配两部分，揭示 ESS 与归一化常数估计的盲区；③提出无阈值、可解释的缺失质量估计与自我诊断；④在没有密度的采样器上同样可用。

**🔧 技术方法**

技术方法包括：多起点二阶优化（Adam + Levenberg‑Marquardt）寻找局部极值；拉普拉斯近似计算局部质量；马氏距离去重与分配；自我诊断指标（增长率、单例占比）；与调优的退火 SMC、平行温度、以及解析基准的对比评估。

**📊 数据集**

使用的数据集包括：16 维高斯混合（8 组）；16 维 Many‑Well（32 个基底）；其旋转变体；以及 13 原子 Lennard‑Jones 集群（不同逆温 β）。所有目标均具有解析或精确数值的归一化常数与基底权重。

**📈 对比分析**

与调优的退火 SMC（耗费 70–280% 训练成本）和平行温度权重估计（10 倍成本）对比，预飞行检查仅需 2.7–10% 的训练预算，平均绝对误差 < 1e-2（大部分 < 1e-3），显著优于 SMC；在无密度采样器上同样表现良好。自我诊断的误清率低，误报率可控。

**⚠️ 局限性**

局限性包括：①需要可通过二阶搜索找到所有基底；②需要拉普拉斯可计算的局部质量，连续对称导致零模需特殊处理；③估计仅在枚举充分时才可靠，无法证实缺失质量为零；④小缺失质量下存在误差底限；⑤随着维度增大成本线性增长，在 700 维左右与训练成本持平；⑥不适用于离散状态或高非高斯/多峰结构极差的目标。

---

## 579. Click, Branch, Audit: A Decision-Tree Toolkit for Assessing Fundamental Rights Impacts under the Digital Services Act of Very Large Platforms & Search Engines

**arXiv ID:** 2609.26353 | [PDF](https://arxiv.org/pdf/2609.26353v1)

**作者:** Marie-Therese Sekwenz `[一作]` (Delft University of Technology), Ben Wagner `[通讯]` (Delft University of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并实现了一个基于决策树的工具包，用于评估欧盟数字服务法第34条中对大型平台和搜索引擎的基本权利影响的系统性风险。

**💡 创新点**

创新点在于将“脆弱性”视为情境化、关联化的风险指标，并将法律、审计与技术要求映射为可操作的决策树流程，同时支持风险报告、可视化和审计证据链的生成。

**🔧 技术方法**

使用技术包括设计科学方法、决策树结构、无后端前端实现（GitLab Page）、JSON/CSV导出、PlantUML 可视化，并通过多学科专家工作坊收集反馈。

**📊 数据集**

主要数据来源为三轮专家工作坊的匿名反馈（共35名参与者）以及两个案例情景（AI生成不雅图像和TikTok成瘾）的模拟评估数据。

**📈 对比分析**

通过模拟不同角色（监管者、行业、学术、民间组织等）评估的对比，展示工具能统一生成风险报告并揭示评估分歧；虽然未给出数值性能指标，但表明工具在一致性和可审计性方面具备优势。

**⚠️ 局限性**

局限性包括：缺乏真实审计验证、未充分覆盖系统间交互风险、依赖尚未成熟的法律解释、案例仅为压力测试、未在不同平台架构上进行广泛验证。

---

## 580. A Deeper Look at Depth: Stable Generation Accounting for Quantifier Reasoning

**arXiv ID:** 2609.26345 | [PDF](https://arxiv.org/pdf/2609.26345v1)

**作者:** Can Cebeci `[一作]` (EPFL), Clément Pit-Claudel `[通讯]` (EPFL)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的量化实例深度（generation）计数方法，以消除 SMT 求解器在程序验证中的结果不稳定问题；

**💡 创新点**

创新点在于对深度计数提供了严格的、无冲突的语义定义，并通过从抽象理想模型到可实现模型的逐步细化，最终实现了可合并、可回溯且不产生额外时间消耗的稳定计数；

**🔧 技术方法**

技术手段包括：基于 Church-Rosser 的无冲突性证明、量化实例的权重化生成计数、利用等价类（congruence class）跟踪和粘性更新（sticky updates），以及对 Z3 中 e‑node 结构的改造；

**📊 数据集**

实验使用 Mariposa 程序验证框架的 55 个不稳定核心查询（共 5500 个变异），并在非增量 SMT‑LIB 基准上验证了无回归；

**📈 对比分析**

与 Z3 原始计数方案以及两种中间实现比较，结果显示最终实现将未知变异数量减少 94%、不稳定查询数降低 85%，并且在 Mariposa 基准上时间超时仅略有 3% 的轻微上升；在 SMT‑LIB 基准上无显著性能退化；

**⚠️ 局限性**

局限在于粘性更新在极端情形下可能导致内存占用和回溯成本上升，且当前实现仍未完全支持重复匹配导致的生成数下降情况，未来工作需进一步优化并评估对更大规模程序的可扩展性。

---

## 581. An Infinitary and a Cyclic Sequent Calculus for Non-Monotone Inductive Definitions

**arXiv ID:** 2609.26337 | [PDF](https://arxiv.org/pdf/2609.26337v1)

**作者:** Robbe Van den Eede `[一作]` `[通讯]` (KU Leuven), Robbe Van den Eede (KU Leuven)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文扩展了Brotherston和Simpson的序列演算，为非单调归纳定义提供了无穷树和循环图两种推理系统，并证明了其完整性、剪切消除（对正定义）、循环归一化等性质。

**💡 创新点**

创新点在于把无限递归与循环演算迁移到非单调定义域，改进了追踪条件以保证well‑found semantics下的正确性，并给出了完整性与剪切限制的证明。

**🔧 技术方法**

使用的技术主要是：Gentzen序列演算、无限推理与循环证明框架、追踪条件与循环归一化、正则树与图展开、以及正则化与证据（justification）理论。

**📊 数据集**

论文未使用任何实验数据集，全部为理论证明与形式化分析。

**📈 对比分析**

由于为理论工作，未进行实验比较；但通过证明可得系统对所有有效序列可得推理、剪切可消除、循环可归一化等性质。

**⚠️ 局限性**

局限在于剪切消除仅适用于正定义；对分层或全定义的剪切消除尚未实现；系统尚未与稳定语义或其他非单调逻辑结合；实现与自动证明工具仍待进一步研究。

---

## 582. Disaggregated Quantization: Specializing LLM Prefill and Decode

**arXiv ID:** 2609.26333 | [PDF](https://arxiv.org/pdf/2609.26333v1)

**作者:** Andrei Panferov `[一作]` (NVIDIA), Dan Alistarh `[通讯]` (ISTA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了分离预填充(prefill)和解码(decode)阶段的量化方案（Disaggregated Quantization, DQ），并在此基础上开发了量化感知蒸馏（QADD）、全分离量化和预填充器（prefillers），通过SSD流式加载（ODP）实现低内存占用。

**💡 创新点**

创新点：① 让预填充与解码使用不同的量化格式、权重量化和存储位置，充分利用两阶段不同的计算和内存瓶颈；② 通过QADD在一次前向后向过程中联合训练两阶段的权重，保持统一的生成目标；③ 通过ODP将预填充权重放到SSD，减少设备显存需求并保持低TTFT；④ 兼容现有冻结的量化检查点，仅训练预填充器即可显著提升低精度解码的准确性。

**🔧 技术方法**

使用技术包括：NVFP4、LUT2/LUT3低比特权重量化、激活量化禁用、QADD蒸馏、全分离量化（预填充权重与解码权重分别训练）、ODP SSD流式加载、vLLM与NIXL分布式推理框架、LLM推理时间测量工具。

**📊 数据集**

主要数据集：Qwen3、Gemma3（不同规模）、Qwen3.8‑27B GGUF检查点、MMLU‑Pro、MMMU‑Pro、RULER、GSM8K、MATH‑500、MATH‑500、以及大规模2.8T模型（Qwen 3.8、Gemma 4、Muse Glimmer、Nemotron 3、Kimi‑K3）。

**📈 对比分析**

比较方法：在vLLM上测量batch‑1的token‑latency、时间到首token（TTFT）以及预填充速度；对不同量化方案在decode‑heavy与prefill‑heavy任务上进行准确率对比。性能：格式分离在decode‑heavy任务上提升1.9–3.1点准确率、prefill速度提升至1.49×，全分离+ODP在27B模型上TTFT提升1.78×，低精度解码在1‑bit时准确率提升超过32点。

**⚠️ 局限性**

局限性：未评估高并发批量推理、循环/多轮交互、混合专家（MoE）模型；未验证缓存策略对多轮生成的鲁棒性；ODP在短提示时TTFT略高；对极高位宽（>4bit）或多模态专用模型的适用性需进一步验证。

---

## 583. ArborSplat: Online Semantic Gaussian Splatting SLAM for Orchards

**arXiv ID:** 2609.26315 | [PDF](https://arxiv.org/pdf/2609.26315v1)

**作者:** Alessandro Masini `[一作]` (Politecnico di Milano), Matteo Matteucci `[通讯]` (Politecnico di Milano)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种在线语义3D高斯分裂SLAM系统（简称 Arboret），能够在果园中同时构建高精度几何地图与语义图层，支持单摄像头和LiDAR联合定位，并通过多视角融合生成可靠的语义点云。

**💡 创新点**

创新点包括：
• 结合高度先验与地面拟合的语义约束，显著提升薄结构（树干、棚杆、果实）的识别准确度；
• 在高斯分裂中加入语义logit字段并采用透明度归一化，使薄结构的语义信息不被低透明度稀释；
• 采用类约束的增量细化预算，保证稀有薄结构获得足够的几何原语；
• 设计在线语义点云融合流程（误差门控、多视角投票、地面高程过滤），实现无后处理的实时语义点云。

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting、LiDAR odometry（scan‑to‑submap）、开源文本提示分割器（SAM+prompt）、RANSAC地面拟合、可视化渲染（L1+SSIM+交叉熵损失）、类预算增量细化、语义点云多视角投票。

**📊 数据集**

使用了公开的果园数据集，包含苹果和梨两种树种，覆盖休眠、开花和收获三个生长阶段，共6条训练/验证路径，并使用预计算的单目深度图。

**📈 对比分析**

与SGS‑SLAM、GS3LAM、SemGauss‑SLAM、AgriGS‑SLAM等基线在完整路径、半段和单帧精度上进行对比：在所有12条路径上，ATE均低于0.5 m；在201个共享帧上，mIoU提升0.15‑0.36、结构mIoU提升0.23‑0.50；同时帧速率比GS3LAM高1.7‑7.5倍；语义点云相较于裸转移在结构mIoU上提升0.007‑0.013。

**⚠️ 局限性**

局限性：
• 在新视角下薄结构（尤其树干）的mIoU仍低于0.07；
• 依赖单摄像头图像，若图像质量或遮挡严重会影响分割质量；
• 对于复杂光照和树冠覆盖度高的场景，地面/树干高度先验可能失效；
• 目前不支持循环闭合与关键帧重检，导致新视角语义精度进一步受限。

---

## 584. CHiME-9 ECHI: A Machine Learning Challenge for Enhancing Conversations to Address Hearing Impairment

**arXiv ID:** 2609.26306 | [PDF](https://arxiv.org/pdf/2609.26306v1)

**作者:** Robert Sutherland `[一作]` (University of Sheffield), Jon Barker `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文组织了CHiME-9 ECHI挑战，目标是提升在噪声餐厅式环境下四方对话的语音清晰度与可懂度，并通过客观指标和主观听力测试对七支团队的低延迟多通道神经网络系统进行比较。

**💡 创新点**

创新点在于首次将真实噪声环境下的多说话人对话增强任务与听力辅助设备相结合，强调低延迟（≤20 ms）设计，并通过主观评测揭示客观指标与实际听感之间的差距。

**🔧 技术方法**

采用了TSX/TF‑GridNet等低延迟深度网络框架，结合空洞卷积、O​VS、Beamformer、掩模注意力等技术，部分团队改进了延迟补偿与自适应声道处理。

**📊 数据集**

使用CHiME‑9 ECHI数据集，该数据集提供Meta Aria眼镜与助听器麦克风的多通道录音、CT麦克风高SNR语音、彩虹段训练样本、运动跟踪与VAD标签。

**📈 对比分析**

性能评估采用fwSegSNR、PESQ、CSig/CBak/COvr、STOI等客观指标，并在32名聽力正常的受试者中进行一镜一口和ITU‑P.835质量评估；顶级系统在主观正确率与总体质量上显著优于基线，但客观指标排序与主观结果不完全一致。

**⚠️ 局限性**

限制在于参考信号质量受限导致客观指标与主观感知不匹配，设备几何差异（Aria vs. HA）影响多通道学习，且对动态交叉说话与背景噪声的鲁棒性仍有提升空间。

---

## 585. Quantifying Protocol-Induced Uncertainty in Comparative Predictive-Model Evaluation: Evidence from Large-Scale Daily PM10 Forecasting

**arXiv ID:** 2609.26288 | [PDF](https://arxiv.org/pdf/2609.26288v1)

**作者:** Rafael da Silva `[一作]` (Eastern University), Kiersten Monahan `[通讯]` (Eastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在日常PM10预测中，对比静态分割与滚动起点评估协议，量化协议变更导致的模型排名不确定性。

**💡 创新点**

提出了协议敏感度得分（Protocol Sensitivity Score, PSS）和基于完整重拟合的参考框架，用于将跨协议排名位移与协议内部扰动区分开。

**🔧 技术方法**

采用Kendall τ、PSS、赢家交换（winner swap）等统计量，以及完整重拟合、块抽样、TOST、Benjamini–Hochberg多重检验等技术。

**📊 数据集**

使用425个欧洲背景站点（Heisig/EEA快照）和365个美国EPA监测点的每日PM10数据（2015‑2023年）。

**📈 对比分析**

对比发现跨协议PSS平均约0.8，远高于协议内部扰动（0.07‑0.23）；约35%–60%站点在两协议间赢家模型交换，显示协议选择对模型选定具有显著影响。

**⚠️ 局限性**

局限性包括仅限于日常PM10、两种时间序列评估协议、至多九个候选模型、两大网络；PSS分辨率受限、缺乏高功效的严重性指标、未在其他污染物或领域验证，计算成本随模型数增长。

---

## 586. Decoupling Is Not Identification: Supervised Evidential Learning in Next-Token Prediction

**arXiv ID:** 2609.26268 | [PDF](https://arxiv.org/pdf/2609.26268v1)

**作者:** Ge Wang `[一作]` `[通讯]` (Rensselaer Polytechnic Institute), Ge Wang (Rensselaer Polytechnic Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了 Evidential Next-Token Prediction (ENTOP) 框架，利用字符级 Moby‑Dick 语料中的 8‑gram 计数作为可验证的词汇支持标签，探究 Dirichlet 稠密度（concentration）在语言模型中的意义。

**💡 创新点**

创新点在于：① 把 mean‑concentration 头解耦并为 concentration 赋予明确的“证据”意义；② 通过显式计数监督来审计模型对词汇曝光的记忆；③ 提出了最低证据协议（confidence control、matched pairs、held‑out labels、constant baseline、decision test），并发现显式监督能排序词汇支持但无法保证校准或错误退避。

**🔧 技术方法**

技术包括：两层因果 Transformer；Dirichlet 先验（mean m 与 concentration S 分离）；digamma 目标与 KL 正则；显式计数损失；partial Spearman、pair‑win、RMSE、AUROC、AURC；kNN 近邻检索与 PCA 表示；对比 softmax、CE‑S、ENTOP‑S、ENTOP‑D、D+count、Coupled 头。

**📊 数据集**

数据集为 ASCII 标准化、单字母分词的 Moby‑Dick 章节，按 1–110/111–120/121–135 划分为训练、验证、测试；词表大小 48，8‑gram 上下文；训练集含 511,526 种独特上下文。

**📈 对比分析**

与基线（softmax CE）、显式与隐式证据学习模型对比，使用 partial Spearman、pair‑win、boundary AUROC、RMSE 等指标评估。ENTOP‑S 在词汇支持排序（sup/held‑ρ 与 pair‑win）上优于其它模型，但在校准（entropy、vacuity）和错误退避（AURC、AUROC）方面不如基线；常数预测在自然 RMSE 上更优，表明模型未能获得全局尺度。

**⚠️ 局限性**

局限性包括：仅在单一字符级数据集上验证，缺乏跨语料、跨分词器的泛化；显式监督只能得到支持排名，未能提供可靠的计数尺度或风险估计；对稀疏高计数类型的解释不完整，且 amortization 机制仍未被充分理解。

---

## 587. CANcept: Model-based CAN Traffic Generation and Manipulation

**arXiv ID:** 2609.26263 | [PDF](https://arxiv.org/pdf/2609.26263v1)

**作者:** Lino Wertz `[一作]` (Karlsruhe Institute of Technology), Bernhard Beckert `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一款名为CANcept的开源图形化工具，用于基于模型的CAN总线流量生成、重放、监控和操作；

**💡 创新点**

创新点在于将基于DBC的通信模型(DCM)与时序调度模型(TSM)结合，提供声明式的时序与内容变换规则，显著降低脚本编写难度并提升可维护性；

**🔧 技术方法**

使用C++/Qt 6实现，借助SocketCAN进行物理/虚拟CAN接口交互；核心采用模型驱动的执行引擎，对消息进行编码/解码、时序调度与内容操作；

**📊 数据集**

主要数据集为DBC文件（如vehicle.dbc）和对应的CAN记录轨迹；

**📈 对比分析**

通过与python‑can和SavvyCAN在四个标准任务（传输、位腐败、原始重放、DBC重放）和四个场景（丢帧、时延、复制、顺序）进行对比，CANcept在帧丢失率0%、时延抖动<2 µs、故障保真率100%等指标上均优于基线；

**⚠️ 局限性**

局限性包括仅支持单一CAN接口、仅实现原子谓词、缺乏实时API和多接口支持，且目前仅在Linux环境下测试。

---

## 588. Hierarchical Floorplan-Guided Vision-Language Exploration for Embodied Question Answering

**arXiv ID:** 2609.26360 | [PDF](https://arxiv.org/pdf/2609.26360v1)

**作者:** Albert Gassol Puigjaner `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于层次化结构图和楼层平面图的机器人视觉语言探索框架，用以在未知室内环境中主动收集信息并回答问题。

**💡 创新点**

创新点在于将在线层次化场景图、VLM驱动的高层决策、低层语义前沿规划以及弱结构化楼层平面图结合起来，实现了根据问题需求动态选择房间探索、对象观察或房间发现三种探索模式。

**🔧 技术方法**

方法采用YOLOe+LSeg实现实时物体检测与开放词汇占用映射，Hydra构建层次场景图，CLIP与VLM（如GPT‑5.5）做高层规划与视觉记忆检索，前沿评分与路径规划实现低层导航。

**📊 数据集**

实验使用HM3D场景生成的楼层平面图，评估 OpenEQA 与 ExploreEQA 两个问答基准，并在真实四足机器人上完成室内部署。

**📈 对比分析**

与 GraphEQA、ExploreEQA 基线相比，HFLEX‑EQA 在 OpenEQA 上最高达 75.0% 的成功率、ExploreEQA 上 63.2%，同时在规划步骤和行程距离上均优于对手。

**⚠️ 局限性**

局限性包括对楼层平面图准确性的依赖、VLM 预测误差导致的决策失误，以及在图像严重模糊或光照变化时的鲁棒性下降。

---

## 589. Blaming Across the Aisle: Political Contrasting and Blame Attribution in the Danish Parliament

**arXiv ID:** 2609.26346 | [PDF](https://arxiv.org/pdf/2609.26346v1)

**作者:** Markus Lundsfryd Jensen `[一作]` (Aarhus University), Sara Kolding `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究丹麦议会1997‑2026年责备归因的时间与政治特征，结合BlameBERT分类器与多级负二项模型对近五百万句子进行分析。

**💡 创新点**

首次构建针对丹麦政治文本的责备检测模型，并提出低资源语言的注释高效管道；发现责备呈“香蕉形”趋势并与政治对立与极端右派强化相关。

**🔧 技术方法**

使用BERT基础的BlameBERT（LoRA微调），结合DEBATE零射分类器生成银标签，随后采用负二项混合效应模型进行统计分析。

**📊 数据集**

基于ParlSpeechV2与SFTP抓取的议会记录，处理后约5.6M句子，最终保留4.9M句子用于训练与预测。

**📈 对比分析**

与Qwen 3.0B嵌入和Qwen 3.5:9B生成式基线对比，BlameBERT在金标集上宏F1达0.80、精确率0.80、召回率0.81，显著优于基线。

**⚠️ 局限性**

主要限制包括翻译噪声、DEBATE零射偏差、党派数量有限导致方差受限、零值过度分布、支持党与反对党二元化误差，以及未识别责备对象。

---

## 590. Geometry-Aware Hyperbolic Residual Quantization

**arXiv ID:** 2609.26342 | [PDF](https://arxiv.org/pdf/2609.26342v1)

**作者:** Alessio Colombo `[一作]` (University of Amsterdam), Melika Ayoughi `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了几何感知的双曲残差量化框架GHRQ‑VAE，用于在Poincaré球上实现层次化离散表示；

**💡 创新点**

创新点包括：①前向阶段采用逆嵌套的Hyperbolic Residual Aggregation (HRA)保证残差级联可逆；②后向阶段使用单步折扣的Hyperbolic Straight‑Through Estimator (d‑HSTE)实现块级梯度路由，避免梯度在残差层级上递归放大；

**🔧 技术方法**

采用的技术包括双曲向量量化、Möbius代数、并行传输、折扣直通估计、梯度正则化以及自动校准的编码器尺度控制；

**📊 数据集**

使用的数据集有：WordNet词义层次预测、Amazon Beauty序列推荐、MNIST与CIFAR‑100图像重构/生成、LibriTTS音频编码；

**📈 对比分析**

与欧氏残差量化基线及原始双曲升维方法进行对比；在层次组织度（Recall、ARI、NMI等）上，GHRQ‑VAE优于两者；在压缩质量（MSE、PESQ、SI‑SDR）上，欧氏方法更佳；在生成质量（FID/IS）上表现取决于数据集，双曲模型在MNIST上略优，CIFAR‑100上欧氏更好；

**⚠️ 局限性**

局限性：对原始信号压缩性能不如欧氏；在深度（N≥12）时仍需精细调参以避免数值不稳定；对高频域或高分辨率数据的泛化尚未验证；

---

## 591. Design and Evaluation of a Controlled Post-Alert Incident Orchestration and Response Subsystem Using a Rule Engine and a Local Large Language Model

**arXiv ID:** 2609.26316 | [PDF](https://arxiv.org/pdf/2609.26316v1)

**作者:** Hoang-Lam Huynh `[一作]` (Ho Chi Minh City College of Transport), Khuong Nguyen-An `[通讯]` (Ho Chi Minh City University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个受控的后警报事件编排与响应子系统，结合规则引擎、静态检索增强生成、局部LLM、人工审批和技术执行，保证安全性与可追溯性。

**💡 创新点**

将确定性决策、LLM辅导分析、人机交互与技术执行分离，并引入白名单双检、恢复不重放、输出验证与安全回退等多重安全机制。

**🔧 技术方法**

采用Elasticsearch、SQLite持久队列、规则引擎、静态RAG、Ollama托管的本地LLM（qwen2.5:1.5b）、Validator/Guardrail/Output Sanitizer、Response Engine、Dashboard等技术。

**📊 数据集**

使用模拟器生成的结构化警报数据，涵盖拒绝服务、暴力破解、利用漏洞、可疑扫描四类，存入Elasticsearch。

**📈 对比分析**

通过96个功能与集成测试、30次延迟测量、队列压力实验与LLM争用实验对比，平均后警报处理时间约33秒，队列吞吐7.14/s，LLM争用保持单实例，未出现重复或错误。

**⚠️ 局限性**

仅评估后警报路径，未包含检测、网络流量、分布式部署、长时间负载及真实管理员体验；LLM作为辅导并未提升有效性，主要资源瓶颈在本地推理。

---

## 592. Staged Multi-step UTXO Workflows via Recursive Invariants

**arXiv ID:** 2609.26305 | [PDF](https://arxiv.org/pdf/2609.26305v1)

**作者:** Shuyang Tang `[一作]` (Chongqing University), Guoqiang Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出递归不变式（RI）以支持多步 UTXO 工作流的本地验证，并通过事务级规则实现状态穿越。

**💡 创新点**

创新点在于：①使用递归不变式在事务间传播规则；②利用 Kleene 三值语义推迟未来依赖检查；③给出可评估片段的形式化模型并证明其安全性与 Turing 完备性。

**🔧 技术方法**

技术手段包括：UTXO 执行模型、递归不变式 DSL、三值语义、Coq 形式化、Python 解释器与成本估算工具链。

**📊 数据集**

数据集为六个实践驱动的工作流案例（如 NFT、支付通道、时间锁等）构成的交易轨迹。

**📈 对比分析**

与 Solidity（账户式链）对比，基准实验显示累计验证成本近线性增长，且在不预构造事务的前提下保持低延迟和可预测的成本。

**⚠️ 局限性**

局限性包括：仅提供验证时局部性，无法保证全局一致性和活性；对可验证片段有限制；不支持全链范围的查询和量化；实现基于有限 256 位表示。

---

## 593. CompKV: Compensation-Aware KV Selection for Long-Context LLM Inference

**arXiv ID:** 2609.26300 | [PDF](https://arxiv.org/pdf/2609.26300v1)

**作者:** Zhen Huang `[一作]` (Tsinghua University), Haohuan Fu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CompKV，基于补偿误差的 KV 块稀疏注意力机制，优先选取对补偿贡献最大的块，从而在长上下文推理中兼顾速度与精度。

**💡 创新点**

创新点：① 把块选择与尾部补偿耦合，使用块注意质量与内部对数差异共同估计残差；② 利用块均值与分组方差的紧凑统计实现低成本残差评估；③ 通过异步 CPU‑GPU 并行实现高效的 KV 读取与补偿。

**🔧 技术方法**

技术手段：块均值补偿（Mean compensation）、分组方差统计、KL 目标的二阶展开、基于残差的块评分、异步多流 CPU‑offload、FlashInfer 加速。

**📊 数据集**

实验数据集：RULER、LongBench‑Pro。

**📈 对比分析**

与 Quest、InfLLM、Quest+RESA 等稀疏方法及全注意力进行对比；在 Llama‑3.1‑8B‑Instruct、Qwen3‑8B、Qwen3‑32B 三大模型上，CompKV 在两大基准的平均分上最高，且在 NVIDIA H100 GPU 上实现最高可达 6.85× 的单层自注意力加速。

**⚠️ 局限性**

局限性：对极端低精度任务仍有限；依赖块均值补偿的假设，可能在某些模型或更大块尺寸下效果下降；额外 KV 统计存储与 CPU‑GPU 通信开销需进一步优化；未对能耗或部署成本做深入评估。

---

## 594. Neural Fingerprints for Malware Analysis: An Image-Based Metric Learning Approach with Application to Cross-Domain Classification

**arXiv ID:** 2609.26282 | [PDF](https://arxiv.org/pdf/2609.26282v1)

**作者:** Manasa Deshagouni `[一作]` (San Jose State University), Mark Stamp `[通讯]` (San Jose State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于图像的度量学习框架，用神经指纹提取器实现零样本恶意软件家族识别，并通过FAISS实现大规模检索。

**💡 创新点**

创新点包括：①使用多代理锚点损失在轻量化CNN上训练，提升零样本检索性能；②系统性比较同域、跨域和严格零样本三种情形，展示了度量学习相较分类模型的优势；③提供了多维度评价指标（cluster purity、silhouette、separation ratio、open‑set AUROC等）。

**🔧 技术方法**

技术手段包括：将二进制映射为灰度图像；ResNet‑18或自定义小型CNN提取特征；L2归一化嵌入；批量硬三元组损失与多代理锚点损失；FAISS实现近似最近邻检索；数据增强与混合训练。

**📊 数据集**

使用的数据集有：MalImg（9,339样本、25类）、MalNet‑Images‑Tiny（75,889样本、47类）以及17个在训练中未出现的灰度恶意样本（17,000样本）。

**📈 对比分析**

与传统SVM+PCA（86%）、ResNet‑18分类器（91%）对比，在同域任务中度量学习达到94%；在跨域（MalNet→MalImg）取得88.5%；在严格零样本（17类）上达到73.1% top‑1，open‑set AUROC 90.5%；检索速度低于1 ms，内存占用仅150 MB。

**⚠️ 局限性**

局限性包括：依赖未打包/未加密的二进制；对不同二进制‑图像转换方式的迁移性有限；少样本家族易导致聚类质量下降；需要维护并不断更新图库；对抗性样本可能诱导误检。

---

## 595. On the Effect of Bit-Level Parameter Perturbations in Machine Learning and Deep Learning Models

**arXiv ID:** 2609.26280 | [PDF](https://arxiv.org/pdf/2609.26280v1)

**作者:** Akanksha Raghapur `[一作]` (San Jose State University), Mark Stamp `[通讯]` (San Jose State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在参数层面对经典机器学习模型（HMM、SVM）与深度学习模型（MLP、LSTM）进行有针对性的位翻转和量化扰动，研究其对模型性能的影响，揭示了参数敏感性和鲁棒性差异。

**💡 创新点**

创新点在于首次系统地将位级攻击与逐步搜索技术（PBS）应用于经典模型，比较不同模型族的鲁棒性与潜在的隐写容量，并提出了针对参数空间脆弱性的评估框架。

**🔧 技术方法**

采用的技术包括最小有效位（LSB）翻转、8‑bit 量化、逐步位搜索（PBS）以及比例概率扰动，并通过对分类准确率的监测来评估攻击效果。

**📊 数据集**

实验使用 Drebin Android 恶意软件数据集（15,036 个应用，含 215 个二进制特征），在训练/验证/测试集上进行模型构建与扰动评估。

**📈 对比分析**

实验结果显示，经典模型在仅翻转少量关键位时准确率迅速下降（从 0.9137 降至 0.57），而深度模型（尤其是 LSTM）对位翻转更分散、鲁棒性更高；SVM 的线性版本易受攻击，RBF 版本表现相对稳健，整体表明神经网络拥有更大的隐写容量。

**⚠️ 局限性**

局限性包括仅评估了 Drebin 数据集与有限模型族，未探索在真实硬件 fault 注入场景下的鲁棒性；隐写检测与防御机制未被研究；缺乏对更复杂模型（如 GNN）或多任务场景的验证。

---

## 596. Multi-Axis Selective Decoupling Framework for Free-Floating Space Manipulators

**arXiv ID:** 2609.26267 | [PDF](https://arxiv.org/pdf/2609.26267v1)

**作者:** Daegyun Choi `[一作]` (University of Cincinnati), Donghoon Kim `[通讯]` (University of Cincinnati)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种多轴选择性去耦框架，在自由漂浮空间机械手中通过消除特定方向上的基座动量转移来实现任务空间操作。

**💡 创新点**

创新点在于采用方向约束子耦合矩阵，将传统全轴反作用力约束转为仅针对关键方向的零动量转移，从而大幅保留工作空间并避免奇异。

**🔧 技术方法**

利用惯性加权旋转耦合雅可比矩阵、正交投影、约束矩阵、优先级多任务优化和最小范数解等技术实现去耦与任务跟踪。

**📊 数据集**

在ETS-VII试验卫星模型及其6-DOF机械臂的真实动力学参数下进行数值仿真验证。

**📈 对比分析**

与全轴反作用消除法对比，仿真表明约束方向的角速度误差降至10^-19 rad/s，任务跟踪误差保持在0.05比例增益范围内，未触发奇异且工作空间显著扩大。

**⚠️ 局限性**

仅在单臂刚体模型下验证，未考虑外部扰动、柔性结构或多臂协作，且需要预先指定关键方向，实际实现仍需进一步实验验证。

---

## 597. Formally Modeling the Terrapin Attack on SSH

**arXiv ID:** 2609.26358 | [PDF](https://arxiv.org/pdf/2609.26358v1)

**作者:** Jörg Schwenk `[一作]` (Ruhr University Bochum), Marcus Brinkmann `[通讯]` (Ruhr University Bochum)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对SSH通道完整性进行形式化建模，并引入部分可选状态攻击（Terrapin攻击）模型，分析了SSH中各种AEAD模式的安全性。

**💡 创新点**

提出了首个支持部分可选状态的正式安全模型，统一了状态化加密与AEAD接口，并通过该模型解释了Terrapin攻击对不同SSH加密模式的影响差异；同时给出新型的BEAST式攻击，区分了kpa与cpa模型。

**🔧 技术方法**

使用正式安全游戏（co、kpa、cpa三种oracle）、状态化加密与AEAD的伪代码映射、数学证明与游戏转换技术，评估了SSH中八种常见加密模式的安全性。

**📊 数据集**

该工作为理论分析，不使用实际数据集，全部基于数学证明和伪代码实现。

**📈 对比分析**

通过在同一模型下对所有模式进行比较，得出在不同攻击模型下的安全性结论（如etm、ChaCha20-Poly1305在co下不安全，gcm在所有模型下安全，eam‑cbc在kpa下安全等），但未给出实验性能指标。

**⚠️ 局限性**

局限性包括：未考虑应用层验证；对RC4相关模式的安全性无法证明；模型仅针对SSH协议，未推广到其他安全通道；以及未考虑实时性能和实现细节。

---

## 598. Leveraging Vision-Based Point Cloud Map Priors for Camera-Based 3D Object Detection and Online Vectorized HD Mapping

**arXiv ID:** 2609.26325 | [PDF](https://arxiv.org/pdf/2609.26325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 599. PreGS: A Parameter-Transfer-Based Multi-Expert Graph Neural Network for Node Classification

**arXiv ID:** 2609.26310 | [PDF](https://arxiv.org/pdf/2609.26310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 600. Shaft-Configuration-Adaptive Catheter Tip Position Estimation via Motor-History Conditioned Residual Learning

**arXiv ID:** 2609.26304 | [PDF](https://arxiv.org/pdf/2609.26304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 601. ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model

**arXiv ID:** 2609.26299 | [PDF](https://arxiv.org/pdf/2609.26299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 602. PACT: From Credit Assignment to Critic Alignment

**arXiv ID:** 2609.26355 | [PDF](https://arxiv.org/pdf/2609.26355v1)

**作者:** Jiayan Fu `[一作]`, Mu Chuan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于三条正则性条件（完整性、前缀一致性、中性）对 token 级信用赋值的唯一表征，并以此解释和改进 LLM 强化学习算法，最终设计并验证了新的 Policy Aligned Critic Training (PACT) 框架。

**💡 创新点**

核心创新在于：1) 证明了在完整性、前缀一致性和中性条件下，token 级信用唯一可由马尔可夫差分（V_i-V_{i-1}）给出；2) 将 OPD 教师视为隐式评判器，RLOO 与 token 信用在期望梯度上等价；3) 通过信用稀疏性分析说明 λ=1 的 GAE 更适合长序列；4) 采用 Actor-then-Critic 更新顺序与重要性采样校正，使 critic 与更新后的 policy 更好对齐；5) 使用 BCE 取代 MSE 进行值函数训练。

**🔧 技术方法**

利用马尔可夫过程理论、信息滤波（σ-代数）、马尔可夫差分、重要性采样、BCE 损失、PPO/GRPO/SAO 参考框架，并在 Dressage/Slime 环境中实现。

**📊 数据集**

数学推理任务使用 DAPO-Math-17k（子集 3,200 个问题）与 OpenCode 代理；代码任务使用 OpenSWE 与 Codex 代理；在 Qwen3.5-4B 和 Qwen3.6-35B-A3B 两大 LLM 上进行实验。

**📈 对比分析**

与 GRPO、PPO（λ=0.95/1.0）、SAO 等方法对比，PACT 在四项数学推理基准上的平均准确率 72.87% 超过 GRPO（64.07%）和 PPO‑λ1.0（59.71%）分别 8.80% 与 13.16%；在 SWE‑bench Verified 的 Pass@1 上 67.4% 也高于对比方法。

**⚠️ 局限性**

局限包括：1) 只在可观测奖励、可整合环境假设下证明，实际长序列和非可观测奖励场景可能不完全适用；2) 重要性采样校正对极长序列仍可能产生高方差；3) 实验仅在特定 LLM 与任务上验证，跨模型或更复杂任务的推广性待进一步研究。

---

## 603. TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling

**arXiv ID:** 2609.26347 | [PDF](https://arxiv.org/pdf/2609.26347v1)

**作者:** Julien Knafou `[一作]` (HES-SO), Patrick Ruch `[通讯]` (Swiss Institute of Bioinformatics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 TransCorpus 翻译工具包和 TransBERT‑bio‑fr 语言模型，并用机器翻译产生的合成法语生命科学语料库进行预训练和下游任务微调

**💡 创新点**

通过仅使用高资源语言（英语）翻译生成低资源语言（法语）专业语料，证明合成翻译数据可替代原生数据，在专用领域实现 SOTA

**🔧 技术方法**

采用 M2M‑100 机器翻译+fairseq 并行翻译；使用 SentencePiece 训练子词分词器；采用 RoBERTa 风格 BERT 预训练（MLM 目标）

**📊 数据集**

核心数据集：TransCorpus‑bio‑fr（36.4 GB，221 M句，5.25 B词）以及 22 M 法语生命科学摘要；评估使用 DrBenchmark 15 个任务集（分类、NER、POS、STS）

**📈 对比分析**

在 15 个评测任务和按任务汇总的 4 个指标上，TransBERT‑bio‑fr 在 10 项任务、12 个标签/实体级别和整体 4 个任务上均优于 CamemBERT 与 DrBERT，且多处统计显著（Friedman‑Nemenyi、Wilcoxon）

**⚠️ 局限性**

依赖机器翻译质量，难以推广到缺乏强 MT 模型的语言、其它专业领域（金融、法律等）以及跨域泛化；实验仅覆盖法语生命科学，其他语言/领域的效果尚待验证

---

## 604. On the Role of the Projector in Contrastive Self-Supervised Learning: Last-Layer Rank Dynamics Drive Representation Quality

**arXiv ID:** 2609.26334 | [PDF](https://arxiv.org/pdf/2609.26334v1)

**作者:** Siladittya Manna `[一作]` (Hong Kong Baptist University), Saumik Bhattacharya `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究自监督对比学习中投影器（Projector）的作用，阐明其如何阻止维度坍塌并促进高层特征学习；并提出仅对最后一层权重做正交正则化的方案，降低维度坍塌。

**💡 创新点**

创新点在于：①从线性代数角度详细拆解维度坍塌的四种机制；②证明投影器的核心作用是保持最后层的全秩并提升特征方差；③提出只在最后一层施加正交正则化，既避免了全网络过度正则化导致的表达能力受限，又有效提升了下游性能。

**🔧 技术方法**

技术包括：InfoNCE 损失、正交权重正则化（‖WWᵀ−I‖²）、矩阵谱分析与 Rayleigh–Ritz 定理、数据处理不等式等；实验使用 ResNet18/ResNet50 编码器，训练无投影器和有投影器两种配置。

**📊 数据集**

使用了 CIFAR‑10、CIFAR‑100 以及 ImageNet‑100（每类 1300 张图）三个数据集进行预训练和线性评估。

**📈 对比分析**

与基线 SimCLR、SimCLR+WeRank（全网络正则）以及 DirectCLR 等方法比较，实验显示：在无投影器的设置下，仅对最后层正则化可提升 kNN 精度约 0.6–1.2%，在有投影器时提升约 0.2–0.8%；在 ImageNet‑100 上亦优于 Baseline 和 WeRank，表明该方法在多种配置下均有效。

**⚠️ 局限性**

局限性在于：①只解决了权重范数衰减导致的秩坍塌，未针对跨层谱对齐和特征空间对齐等其他机制；②实验仅覆盖 CNN 结构与 InfoNCE 对比学习，对 Transformer 或非对比自监督方法的推广仍需研究。

---

## 605. On the security and privacy of LLMs in Mobility

**arXiv ID:** 2609.26295 | [PDF](https://arxiv.org/pdf/2609.26295v1)

**作者:** Mauro Conti `[一作]` (University of Padova), Umberto Salviati `[通讯]` (University of Padova)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 35 篇关于大语言模型（LLM）在城市移动性领域应用的文献进行系统调研与安全评估，并将欧盟 AI 法案的九项技术类标准映射到移动性 LLM 研究中，量化评估各研究在安全、隐私、可靠性等方面的合规程度。

**💡 创新点**

① 将 AI 法案的高风险条款细化为 9 个技术评估维度，提供可操作的合规框架；② 采用量化 Likert 评分方法，对现有研究进行合规性打分，首次揭示移动性 LLM 研究与法规之间的巨大落差；③ 在调研中首次全面梳理安全、隐私、可靠性以及版权等多维度缺口。

**🔧 技术方法**

主要使用文献检索（Web of Science）、布尔查询、分类体系（信息处理、知识编码、生成器、决策促进器）、AI 法案映射、Likert 量化评估、统计分析（均值、标准差）等方法。

**📊 数据集**

使用的“数据集”是从 Web of Science 检索得到的 35 篇科研论文本身；未使用外部交通或 LLM 训练/测试数据集。

**📈 对比分析**

通过对 35 篇论文在 8 项技术评估维度（漏洞评估、风险管理、透明度与版权、准确性与鲁棒性）进行 5 分制打分，计算均值和标准差。结果显示：准确性与鲁棒性平均得分 2.30，漏洞评估平均 1.11，风险管理平均 1.09，透明度与版权平均 1.01，表明研究在功能性能上相对较好，但在合规安全方面表现极差。

**⚠️ 局限性**

限制主要包括：① 仅覆盖学术论文，未涵盖商业部署或实际系统；② 只评估 8 项技术维度，未涉及质量管理、技术文档、人工监督等后续治理要求；③ 样本量 35 篇有限，可能存在检索偏倚；④ 评估基于文献描述，缺乏实验验证和细粒度攻击测试；⑤ 研究聚焦在 GPT 与 Llama，其他 LLM 的安全性未得到充分考察。

---

## 606. Dual-Frontier: When Can an Agent Trust Its World Model?

**arXiv ID:** 2609.26293 | [PDF](https://arxiv.org/pdf/2609.26293v1)

**作者:** Huatai Zhu `[一作]` (Central South University), Yi Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了Dual-Frontier框架，用于在智能体使用世界模型时进行决策层面的可验证性判定。

**💡 创新点**

将失败归因问题形式化为对比优势的反事实分解，并证明被动交互无法识别模型与决策的错误来源；引入基于行动特定误差界限的可认证门控机制，实现决策可靠性保证与闭环提升。

**🔧 技术方法**

理论证明（可逆贝尔曼残差、最优误差边界）、贝叶斯置信区间、同时置信序列、证据重用、模拟实验与实际工具调用评估。

**📊 数据集**

受控有限世界实验（链、网格、稀疏），BFCL v4、API-Bank、NexusRaven三大公共工具使用基准；使用Llama-3.1-8B-Instruct、Qwen3-8B等模型作为代理与世界模型。

**📈 对比分析**

与Always-WM、Confidence、Consistency、Pessimistic等基线进行对比，Dual-Frontier在任务成功率和参数准确率上平均提升约10‑12个百分点，且显著降低误操作率，保持高覆盖率。

**⚠️ 局限性**

依赖可计算的误差上界和置信序列，可能导致计算开销；在极端动态或长时间序列中误差估计不稳定；仅在有限步规划验证，长周期效果尚未深入探究。

---

## 607. FISSION: Label Augmentation for Bot Detection

**arXiv ID:** 2609.26279 | [PDF](https://arxiv.org/pdf/2609.26279v1)

**作者:** Sen Yang `[一作]` (Yale University), Aviv Yaish `[通讯]` (Staples High School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于账号历史自分割的无标签监督方法FISSION，利用同一账号的不同时间段视图作为正样本进行对比学习，再用少量手工标签训练下游分类器；

**💡 创新点**

创新点在于将单账号内部的时间切片视为正样本，形成无需外部注释的监督信号，并结合预训练编码器与少量标签的集成学习，显著降低对标签的依赖；

**🔧 技术方法**

使用对比学习（SimCLR/NT-Xent）和多层感知机编码器、投影头、三重编码器集成，以及逻辑回归或RBF SVM下游分类，最后通过Distill扩展至稀疏历史；

**📊 数据集**

实验数据集包括English Wikipedia sockpuppet（活跃集161k和完整集3.3M）和Twitter/X Cresci15 5,301账户；

**📈 对比分析**

采用账号无交叉的五折交叉验证，分别与原始特征+逻辑回归或RBF SVM对比；在活跃集上准确率提升约4个百分点，Sockpuppet F1提升约4.8；在Cresci15上RBF SVM提升约0.7、Logistic提升约0.35；在标签稀缺情况下，1%标签即可接近全标签性能；

**⚠️ 局限性**

局限性包括对短历史的视图生成效果不佳；需要足够长的账号历史；对行为变化的鲁棒性有限；未在更广泛平台验证，且缺乏实时在线更新机制。

---

## 608. GitScholar: A Dataset for Predicting AI Research Impact from GitHub Engagement

**arXiv ID:** 2609.26361 | [PDF](https://arxiv.org/pdf/2609.26361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 609. JAMPR+/L2D: scalable neural heuristic for constrained vehicle routing problems in dynamic environment

**arXiv ID:** 2609.26275 | [PDF](https://arxiv.org/pdf/2609.26275v1)

**作者:** Andrew Soroka `[一作]` (Moscow State University), Alex Meshcheryakov `[通讯]` (Moscow State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发并评估了一种基于Transformer的神经启发式算法JAMPR+/L2D，用于解决大规模带容量、时间窗、多仓库、取送等约束的车辆路径规划问题。

**💡 创新点**

创新点在于引入可学习的轻量级策略掩码，实现对不同约束组合和距离矩阵变化的快速适配，并在不重新训练网络权重的前提下完成微调；同时将JAMPR+与L2D分治策略结合，显著提升大规模问题求解效率。

**🔧 技术方法**

采用端到端深度强化学习，Transformer编码器-解码器架构，分治L2D方法，以及可插拔的约束掩码微调技术。

**📊 数据集**

使用CVRPLib、Solomon合成集和真实物流运营数据ORTEC竞赛集，覆盖从小型（≈50）到大规模（≥200）实例。

**📈 对比分析**

与经典启发式（OR‑Tools、LKH、HGS）以及单纯AM、JAMPR对比；在CVRP/VRPTW上平均Gap≤1‑2%，比HGS优约85%；在CPDPTW上无微调时Gap略高，但通过掩码微调可将Gap降至接近零，且在前几秒即给出高质量子最优解。

**⚠️ 局限性**

对分布漂移仍敏感，需进行掩码微调；在极大规模或极端约束（如多仓库+多重时间窗）下可能出现性能下降；模型训练成本仍高，尤其是初始大规模数据集的生成和标注。

---

## 610. Online Line Aggregation with Deadlines: Randomized Guarantees and Learning-Augmented Tradeoffs

**arXiv ID:** 2609.26259 | [PDF](https://arxiv.org/pdf/2609.26259v1)

**作者:** Tianhang Lu `[一作]` `[通讯]` (Ocean University of China), Tianhang Lu (Ocean University of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了针对一维线聚合带截止时间（LAD）的在线算法，首先给出一个随机平移指数网格的 e‑竞争性算法并证明其匹配下界；随后设计了在有可行离线解作为建议时的确定性和随机学习增强算法，并给出其一致性与稳健性权衡；最后给出一种 O(n³) 的动态规划离线最优解算法，并通过合成实验评估这些算法的表现。

**💡 创新点**

创新点在于：1）证明 e 是 LAD 的最优随机竞争比，并首次给出对应的随机平移指数网格算法；2）在学习增强框架下，首次提出可调参数 λ 的确定性（1+3/λ‑robust、1+3λ‑consistent）和随机（e+e/λ‑robust、e‑1+λ‑consistent）算法；3）给出针对 LAD 的精确多项式时间离线动态规划；4）通过噪声实验验证学习增强算法在低噪声下的优越性和鲁棒性。

**🔧 技术方法**

核心技术包括：随机平移指数网格、竞争性分析与 Yao 原理、基于建议的分支与约束的确定性与随机策略、动态规划的分解与状态转移、以及对实验数据的噪声模拟和最优解生成。

**📊 数据集**

实验使用合成数据，时间从 1 到 100，位置从 1 到 100，共 100 个时间–位置对；请求数量在每个点上按照三种分布（Poisson、Lomax、迭代 Poisson）独立采样，期望每时刻请求数为 1；通过“替换率”p (0–1) 进行噪声扰动后重新生成最优解作为建议。

**📈 对比分析**

实验对比包括：学习增强算法（确定性 LA‑D 与随机性 LA‑R），经典在线算法（对应 λ=1 时的算法），以及 Blind‑Following 基线。平均竞争比（ACR）作为评估指标。结果显示：在低噪声 (低替换率) 下，LA‑D 与 LA‑R 的 ACR 均低于经典算法，且 λ 较小更优；在高噪声下，学习增强算法 ACR 上升，且 λ 较小表现更差；随机性学习增强算法在实验中实际表现略逊于确定性版本。

**⚠️ 局限性**

局限与开放问题：1）尚未确定最优的一致性–稳健性权衡界；2）随机算法的理论一致性仍较差；3）研究仅限于直线结构，如何将结果推广到更一般的树或图结构仍是待解决的挑战。

---

## 611. AIGC Video Detection based on the fusion of spatial-frequency-optical flow multimodal features

**arXiv ID:** 2609.26274 | [PDF](https://arxiv.org/pdf/2609.26274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 612. RoboTwin-Phys: Do WAMs and VLAs Understand the Physical World?

**arXiv ID:** 2609.26292 | [PDF](https://arxiv.org/pdf/2609.26292v1)

**作者:** Jiaqi Zhang `[一作]` (Peking University), Chuanmin Jia `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了RoboTwin-Phys benchmark，在RoboTwin-2.0基础上增加了13个连续可采样的物理属性，并构建了超过5,000条带有完整物理真值标注的专家演示数据集，用于评估机器人操作模型在物理条件多样性下的鲁棒性。

**💡 创新点**

核心创新在于将物理条件多样性作为独立的评价维度：实现任务感知的物理合法性、持续采样与复现性、基于专家规划的可行性过滤，并公开完整的物理真值标注，弥补了以往 benchmark 对物理参数缺失的空白。

**🔧 技术方法**

采用物理属性的连续采样、专家规划验证、RoboTwin-2.0 仿真环境、任务级配置管理，以及对 Fast-WAM、Motus、FACT、π_0.5、Galaxea-VLA 等代表性模型的系统评估技术。

**📊 数据集**

使用了RoboTwin-Phys 训练集 v1（5000+专家演示，含13维物理真值）以及原始的RoboTwin-2.0 50-task 任务套件。

**📈 对比分析**

在 Clean、Official Random 与 Physical Random 三种环境下评估五个模型；在 Physical Random 下成功率从 Clean/Official Random 的约90% 降至 31%–44%，显示物理条件多样性显著削弱模型鲁棒性。

**⚠️ 局限性**

局限性包括：物理属性采样假设独立性，未充分捕捉跨属性耦合效应；部分任务的物理配置仍有限；评估模型仅为五种，缺乏更广泛的跨模型分析；数据集规模虽大但仍不足以覆盖所有实际场景；benchmark 依赖专家规划过滤，可能导致可解空间偏向较易任务。

---

## 613. Designing and Analysing Argument Mining Pipelines: Towards a Comprehensive Assessment

**arXiv ID:** 2609.26338 | [PDF](https://arxiv.org/pdf/2609.26338v1)

**作者:** Siddharth Bhargava `[一作]` (Fondazione Bruno Kessler), Patricia Martín-Rodilla `[通讯]` (Spanish National Research Council)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对端到端论证挖掘（Argument Mining）管道进行系统性综述，并提出基于语言学、计算和领域三重视角的分析框架。

**💡 创新点**

创新点在于首次将管道设计拆解为三维视角，统一定义结构建模、计算实现与领域整合，并通过该框架识别并归纳多种主流管道模式。

**🔧 技术方法**

采用文献检索与GPT‑4辅助筛选技术，对160余篇论文进行结构化分析，构建AMP164研究集，并用框架化方法对管道进行对比。

**📊 数据集**

数据来源为SCOPUS与Web of Science的检索结果（273+237篇），经筛选得到AMP164集；未使用具体文本语料库，仅以文献为研究对象。

**📈 对比分析**

通过对管道任务划分、建模范式（特征工程、深度学习、LLM）与领域适配的对比，揭示从多步骤到统一化设计的趋势；但论文未给出统一性能指标，仅提供宏观设计比较。

**⚠️ 局限性**

局限在于对每个视角的细节分析不足，未解决跨领域迁移、标准化评估与实际性能验证的问题。

---

## 614. SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation

**arXiv ID:** 2609.26313 | [PDF](https://arxiv.org/pdf/2609.26313v1)

**作者:** Zeyu Lou `[一作]` (Nanjing University), Chenyang Si `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SafeLoop，一个不需要改动基础VLA模型参数的外部安全包装器，能够在长周期执行中预测并回滚潜在危险；

**💡 创新点**

核心创新在于将短期风险预测与基于记忆的回滚相结合，构建一个双速率控制循环，在保持任务成功率的同时显著降低碰撞与物体失效事件；

**🔧 技术方法**

使用视觉与本体信息训练四维风险预测器（碰撞与物体失效的概率与时间到达），以及基于强化学习的三行动决策器（继续、记录安全点、回滚），并采用异步双速率框架；

**📊 数据集**

在LIBERO四大任务套件（Object、Goal、10、Spatial）上收集并标注危险数据，随后在模拟环境中训练预测器与决策器，最终在三台物理机器人（PIPER臂）上进行迁移验证；

**📈 对比分析**

与两类基线（纯RL安全决策器和SAFE检测器）对比，SafeLoop在24个任务上平均将危险事件率从≈1.21/1k降低至≈0.22/1k，保持或略提升成功率（≈59%），同时在真实机器人上降低约78%危险率；

**⚠️ 局限性**

局限性在于无法逆转已发生的不可逆场景变化，且回滚仅能恢复机器人关节状态，不能恢复被损坏的物体或环境；

---

## 615. Combining Hierarchical Cognitive Process with Process Supervision for Interpretable Scene Safety Understanding

**arXiv ID:** 2609.26399 | [PDF](https://arxiv.org/pdf/2609.26399v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 616. A Statistical Analysis of Three Player Auction Bridge

**arXiv ID:** 2609.26265 | [PDF](https://arxiv.org/pdf/2609.26265v1)

**作者:** Aritrabha Majumdar `[一作]` (Indian Statistical Institute), Moutushi Chatterjee `[通讯]` (Indian Statistical Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了三人拍卖桥牌的得分规则，并提出了改进后的得分机制。

**💡 创新点**

创新点在于发现并纠正了原始点数规则中存在的“滑动奖金不依赖竞价”的结构缺陷，并从统计与博弈论双重视角系统评估新旧方案的影响。

**🔧 技术方法**

采用蒙特卡洛模拟、t 检验、Mann‑Whitney U、卡方检验、KS 统计、AIC、Gini、Jain 指数以及通用和式逆向遗憾最小化 (GS‑CFR) 等技术。

**📊 数据集**

使用随机生成的 52 张牌手牌，模拟 100,000 次对局（或 20,000 次 GS‑CFR 迭代），作为实验数据集。

**📈 对比分析**

对比分包括：均值、偏度、峰度的配对检验，分布拟合优度比较（对数正态 vs Gamma vs 正态），公平性指标比较，以及博弈论层面的纳什均衡与平均收益比较。结果显示，新方案显著提升了出价者的平均收益、提高了收益分布的上偏度，同时使得出价者与防守者的收益比例从 0.674 变为 2.111，公平性指标略有下降。

**⚠️ 局限性**

局限性包括：得分分布离散且包含尖峰，导致连续分布拟合不足；模拟基于理想化的出价与双倍规则，未涵盖真实玩家策略；对结果的推广受限于假设的出牌分布与手牌强度模型。

---

## 617. Coding Agents are Strong Prompt Optimizers

**arXiv ID:** 2609.26261 | [PDF](https://arxiv.org/pdf/2609.26261v1)

**作者:** Agamdeep Singh `[一作]` (Microsoft), Sumit Gulwani `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了仅需一次离线分析即可生成高质量系统提示的编码代理技能蒸馏方法（CASD），消除传统迭代搜索与环境交互的需求。

**💡 创新点**

创新点在于将反思范围从小批量轨迹扩展到整个回放语料库，利用编码代理自动执行统计分析并直接合成规则化提示，从而实现低成本、高效的提示优化。

**🔧 技术方法**

采用离线回放语料库、Claude Sonnet 5/Claude Code 等编码代理、数据统计脚本以及规则合成技术来完成提示的生成。

**📊 数据集**

使用四个代理基准数据集：ALFWorld、τ²‑bench 零售、τ²‑bench 电信、SpreadsheetBench‑Verified。

**📈 对比分析**

与 GEPA 与 SkillOpt 在相同回放集下对比，CASD 单次离线通道平均提升 16.6 个百分点（GEPA 10.9、SkillOpt 5.3），且成本约 $1.60/提示，低于 SkillOpt 的 $142.5，远低于 GEPA 的 $8.4。

**⚠️ 局限性**

局限性包括仅在单一目标模型（GPT‑5.4‑mini no‑think）和单一编码代理上验证；依赖已收集的回放，无法发现未出现的行为；对极小或无错误的数据集可能无效。

---

## 618. Benchmarking Robots for Everyday Environments: From Lab Experiments to Real-World Operations

**arXiv ID:** 2609.26490 | [PDF](https://arxiv.org/pdf/2609.26490v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 619. Complementary Roles of Radiomics and Foundation Representations in Renal Cell Carcinoma Classification: A Comparative Study of 2D and 3D CT Encodings

**arXiv ID:** 2609.26463 | [PDF](https://arxiv.org/pdf/2609.26463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 620. Quantitative coverability for probabilistic well-structured transition systems

**arXiv ID:** 2609.26312 | [PDF](https://arxiv.org/pdf/2609.26312v1)

**作者:** Raphaël Faure `[一作]` (Université Paris-Saclay), Lina Ye `[通讯]` (Université Paris-Saclay)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了概率良结构转移系统（pWSTS）框架，并给出了在有限与无限时间窗内求解近似量化可覆盖性问题的统一算法。

**💡 创新点**

创新点在于：①不需要任何关于分支度的假设，只需单步转移概率；②证明所有随机单调pWSTS对任意上闭集合都是决定性的；③将框架应用于多类型Galton–Watson过程，实现了此前未解决的量化可覆盖性近似。

**🔧 技术方法**

采用良序与Monotonicity、Pred‑Basis有效性、可枚举后继、概率路径枚举与剪枝、合并同一状态的技术，并利用决定性与随机单调性证明算法收敛。

**📊 数据集**

主要使用理论模型：多类型Galton–Watson过程、pVAS、pLCS 等；未使用真实数据集。

**📈 对比分析**

与传统路径枚举或截断算法相比，该方法能够处理无限分支系统，误差可预先给定，实验演示在理论模型上收敛且性能良好。

**⚠️ 局限性**

限制在于缺乏对算法执行深度、时间与空间复杂度的明确上界，以及对非决定性pWSTS的适用性仍需进一步研究。

---

## 621. Can We Predict Anomaly Detection Performance from Embedding-Space Geometry?

**arXiv ID:** 2609.26460 | [PDF](https://arxiv.org/pdf/2609.26460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 622. When Recursive Models Finish Computing

**arXiv ID:** 2609.26487 | [PDF](https://arxiv.org/pdf/2609.26487v1)

**作者:** Hare Krishna `[一作]` (University of Texas at Austin), Hao-Yu Sun `[通讯]` (Austin Community College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究递归模型在 Sudoku 任务中扩展推理步长时的解决率提升，并揭示完成时的动态特征。

**💡 创新点**

提出并验证了“轨迹条件各向异性稳定性”作为完成的动态签名，区分了失败与未完成计算。

**🔧 技术方法**

使用 Tiny Recursive Models（TRM）的注意力和 MLP 变体，并对其 Jacobian、扰动和状态更新进行分析。

**📊 数据集**

使用 1000 个难度高的 Sudoku-Extreme 测试集（以及 Easy Sudoku、Maze-Hard 等辅助集）。

**📈 对比分析**

在固定预算 16 步与扩展到 512 步比较，Attention 模型的准确率从 59.2% 提升至 87.5%，MLP 从 74.4% 提升至 91.9%。

**⚠️ 局限性**

局限在于步长有限（512 步）且仅对少数检查点和任务进行分析，无法给出渐近行为与更广泛模型的泛化。

---

## 623. Calibration as a First-Class Criterion in LLM Evaluation

**arXiv ID:** 2609.26489 | [PDF](https://arxiv.org/pdf/2609.26489v1)

**作者:** Mario Sanz-Guerrero `[一作]` (Johannes Gutenberg University Mainz), Katharina von der Wense `[通讯]` (Johannes Gutenberg University Mainz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大型语言模型（LLM）的校准问题进行位置性讨论，并指出现有研究在校准评估方面的缺失。

**💡 创新点**

创新点在于将校准视为每个模型必备属性，提出在所有NLP子领域中同时报告主指标和校准分数，并对开放式生成的校准定义提出研究方向。

**🔧 技术方法**

主要使用已有的校准度量技术（如期望校准误差ECE、Brier分数、AUROC等），并对其在分类与生成任务中的适用性与局限性进行综述。

**📊 数据集**

引用了多种主流LLM发布报告中使用的基准数据（如GPT‑4、Claude、Gemini等），但未进行新的实验或数据集构建。

**📈 对比分析**

文章并未给出实验结果，而是通过对现有指标的理论分析与案例回顾说明：在多选、短答、数学与代码等任务上已有指标可直接评估；但在开放式生成中仍缺乏统一定义与可用度量，导致性能评估受限。

**⚠️ 局限性**

局限性包括：缺乏实证验证，开放式生成校准的定义和度量尚未达成共识；所提报告标准仍需社区共识与实践验证。

---

## 624. Mammo-LIFE: Longitudinal Mammographic Imaging and Clinical Feature Enrichment for Post-Radiotherapy Outcome Prediction

**arXiv ID:** 2609.26443 | [PDF](https://arxiv.org/pdf/2609.26443v1)

**作者:** Farnoush Bayatmakou `[一作]` (Concordia University), Arash Mohammadi `[通讯]` (Concordia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种基于患者层面的多模态框架，利用配对的纵向乳腺X线影像与临床变量来预测放疗后病人的治疗结果。

**💡 创新点**

创新点包括：① 采用 LoRA（低秩适配）对专门针对乳腺影像的 VersaMammo 基础模型进行参数高效微调；② 设计了跨注意力与增量差分相结合的纵向比较模块来捕捉治疗前后影像差异；③ 通过视角注意力池化将四个标准视角的特征聚合为统一的患者表征；④ 在图像预测概率与选定临床特征之间采用后期融合，实现两模态信息的互补。

**🔧 技术方法**

使用了 LoRA、EfficientNet‑B5 变体的 VersaMammo、跨注意力机制、Delta 差分特征、视角注意力池化、逻辑回归和随机森林等技术。

**📊 数据集**

数据集为内部收集的 47 名患者的配对前后放疗乳腺X线影像（四个标准视角），以及对应的临床特征（如肿瘤大小、分级、BMI、糖尿病等）。

**📈 对比分析**

通过患者层面分层 5 折交叉验证，比较了冻结权重与 LoRA 微调、跨注意力与增量差分两种纵向设计，并与仅图像、仅临床以及图像+临床融合三种设定对比。结果显示，LoRA 微调后 AUC 从 0.64 提升至 0.72，最终的图像+临床随机森林模型在增量差分设计下取得最高 AUC 0.86 ±0.14、准确率 0.79 ±0.15、F1 分数 0.83 ±0.11。

**⚠️ 局限性**

局限性包括样本量较小（仅 47 例），缺乏独立外部验证，且数据来自单中心，可能导致模型在不同人群或设备上的泛化性能不足。

---

## 625. QuantWM: Temporally Consistent 2-Bit KV Cache Quantization for World Models and Video Generation

**arXiv ID:** 2609.26425 | [PDF](https://arxiv.org/pdf/2609.26425v1)

**作者:** Jiaqi Zhao `[一作]` (Harbin Institute of Technology), Shuicheng Yan `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的2-bit KV缓存量化框架QuantWM，缓解视频生成和世界模型中的视觉失真和时间抖动。

**💡 创新点**

创新地结合查询敏感聚类（QSAC）和主子空间注意力补偿（PSAC），在保持注意力对齐的同时显著降低Key量化导致的token选择偏移。

**🔧 技术方法**

采用离线中心-残差量化、Query敏感度估计、低秩主子空间投影和流式块级缓存量化等技术。

**📊 数据集**

在LongCat-Video、HY-World 1.5、LingBot-World-v2、Matrix-Game-2、Causal-Forcing以及VBench视频基准上进行评估。

**📈 对比分析**

与BF16、KIVI、QVG等基线在帧级PSNR/SSIM/LPIPS和VBench指标对比，QuantWM在保持或提升视觉质量的同时实现高达6.2×的KV缓存压缩，推理延迟增幅≤6%。

**⚠️ 局限性**

仍存在一定的推理延迟和额外存储需求（主子空间矩阵），且在极低位数量化或更复杂模型中效果可能略有下降。

---

## 626. ESupNNet: An Error Supervising Neural Network architecture for error detection against soft errors in parameters

**arXiv ID:** 2609.26374 | [PDF](https://arxiv.org/pdf/2609.26374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 627. TimeInteract: Towards Real-Time Interactive Intelligence for Streaming Time Series

**arXiv ID:** 2609.26389 | [PDF](https://arxiv.org/pdf/2609.26389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 628. SparseNav: Instruction-conditioned Sparse Semantic Perception for Training-Free Vision-Language Navigation

**arXiv ID:** 2609.26408 | [PDF](https://arxiv.org/pdf/2609.26408v1)

**作者:** Quanhua Chen `[一作]`, Jiarong Lin `[通讯]` (Beijing University of Aeronautics and Astronautics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种训练‑free 的视觉‑语言导航框架（SparseNav），通过持久几何 BEV 地图和仅在语言指令触发时的稀疏语义感知来完成连续环境中的导航任务。

**💡 创新点**

核心创新是“少即是多”的语义定位策略：只在当前子指令中需要的地标被语言激活时才调用开口语义分割，并将已识别的地标以稀疏记忆形式存储，从而避免不必要的感知开销与地图杂乱；同时将几何地图与语义记忆统一作为 VLM 的决策接口。

**🔧 技术方法**

技术包括：基于深度/RGB‑D 的鸟瞰视角（BEV）几何地图构建；开口词汇分割模型 SAM‑3；GPT‑5 作为 VLM 用于子指令解析、可见性判定与候选路点选择；混合前沿/局部方向候选生成器；稀疏地标记记忆与 2D→3D 投影；经典路径规划器。

**📊 数据集**

使用 R2R‑CE 与 RxR‑CE 两个连续视觉‑语言导航基准（Val‑Unseen split）进行评估；在实际机器人上使用 Unitree Go2 搭配 Intel RealSense D455 与 Livox MID‑360 进行实地部署验证。

**📈 对比分析**

与已有零样本和监督学习方法对比，SparseNav 在 R2R‑CE 上取得 42.8% 的 Success Rate（SPL 35.2%，NE 5.96m），在 RxR‑CE 上取得 40.7% SR（SPL 24.1%，NE 7.82m），超过多数零样本方法且接近某些监督方法；消融实验表明 instruction‑related on‑demand 语义感知、混合路点候选、精确地标定位与指令记忆均显著提升性能。

**⚠️ 局限性**

主要局限包括：基于规则的路点生成器限制了路点灵活性；VLM 推理仍是计算瓶颈，导致规划延迟；与部分训练化方法相比，总体 SR 仍略低；缺乏更大规模、更多样化环境与指令的实测验证。

---

## 629. KwaiMind Technical Report

**arXiv ID:** 2609.26375 | [PDF](https://arxiv.org/pdf/2609.26375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 630. AI-Generated Email Drafts Shift Culturally Distinctive Communication Styles in Professional Email

**arXiv ID:** 2609.26403 | [PDF](https://arxiv.org/pdf/2609.26403v1)

**作者:** Shintaro Sakai `[一作]` (Indiana University Bloomington), Katharina Reinecke `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项预注册的内部对照实验中，175名日本和美国全职员工分别在无AI、低语境AI和高语境AI三种条件下撰写工作邮件，随后使用高语境标记量化衡量邮件的高语境特征，分析AI草稿对写作风格的影响。

**💡 创新点**

首次实证证明AI生成的邮件草稿会将用户的写作风格向AI草稿的语境倾向转移，且当AI草稿与用户的文化语境不匹配时，转移幅度更大；同时揭示高语境草稿导致日本用户更依赖、美国用户更易受低语境草稿影响。

**🔧 技术方法**

使用三款大型语言模型（GPT‑5.5、Claude Sonnet 4.6、Gemini 2.5 Flash）生成草稿；通过自定义高/低语境标记集计算邮件的高语境分数；利用线性混合效应模型（LMM）评估条件效应与文化交互；计算AI依赖度作为草稿保留比例。

**📊 数据集**

共收集771封邮件（美国474封，日语297封），包含三类场景（请愿、异议、拒绝）及其对应的高低语境草稿；标记集由文献与Claude自动生成并人工筛选得到；所有邮件均在实验平台上生成并记录。

**📈 对比分析**

通过LMM对比无AI基线与AI条件的高语境分数变化，发现高语境草稿平均提升+2.60点，低语境草稿下降-2.09点；文化交互显著（p<0.001），误配条件下转移幅度为正向条件的10–16倍；AI依赖度普遍>0.85，且日本用户在对齐条件下依赖度显著更高。

**⚠️ 局限性**

局限性包括：翻译过程可能引入误差；实验仅在在线模拟情境中进行，未涉及真实邮件发送和长期使用；AI依赖度高的原因不明（是否因效率、信任或无意识接受？）；仅测得单次会话内的短期风格转移，未检验长期持久性；研究对象限于美国和日本，难以推广至中间语境或其他文化；情境设置有限，未覆盖非正式或同行间交流。

---

## 631. On the Lexical Superstition of Large Language Models for Code Comprehension: Re-evaluation on Code of Low Lexical Quality

**arXiv ID:** 2609.26388 | [PDF](https://arxiv.org/pdf/2609.26388v1)

**作者:** Xin Shen `[一作]`, Ming Li `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在代码理解任务中对标识符词义的过度依赖，提出 Face/Off 语义保持重命名框架，系统评估不同命名条件下模型表现并尝试对抗性提示和微调干预

**💡 创新点**

首次将语义保持的标识符重命名与多任务、跨模型的评估相结合，构建了面向“语义无关重命名” 的基准；并通过类型推断控制实验揭示其效应边界

**🔧 技术方法**

面向标识符重命名的 Face/Off 框架、无监督的扰动与匿名化重命名、提示引导和微调干预、类型推断对照实验、使用 BLEU/ROUGE、MRR、Pass@1 等指标

**📊 数据集**

CodeSearchNet（代码-文档匹配、检索、摘要）、HumanEval（代码补全）、ManyTypes4Py v0.7（类型推断控制）等公开数据集

**📈 对比分析**

通过比较原始、扰动、匿名、混淆命名条件下的性能，发现模型在所有任务中均表现出性能下降，且混淆命名导致的下降最显著；提示和微调干预可部分缓解，但未完全消除差距，表明对词义的依赖较强

**⚠️ 局限性**

实验仅覆盖有限模型、任务和语言；重命名过程可能存在细微错误；提示与微调的设计并非最优；结果无法直接推断内部机制；在真实项目中的泛化程度未知

---

## 632. HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing

**arXiv ID:** 2609.26368 | [PDF](https://arxiv.org/pdf/2609.26368v1)

**作者:** Jianyu Wei `[一作]`, Fuli Luo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出HySparse2，一种双层KV共享的混合稀疏注意力架构，显著降低预填充成本和KV缓存，提升长上下文检索与多轮交互性能。

**💡 创新点**

创新点包括：①双层KV共享（KV Bridging + KV Reuse）实现跨层缓存复用；②Token级稀疏选择取代块级选择；③强制局部窗口取代独立SWA分支；④自回归自解码器与交叉解码器的YOCO式分层结构。

**🔧 技术方法**

使用技术：混合全注意力、滑窗注意力、稀疏注意力、MQA、token级稀疏选择、强制局部窗口、预填充–解码拆分、轻量后训练（post‑training）等。

**📊 数据集**

数据集：约500B/100B的混合语料（上下文长度32k→256k）进行预训练和后训练；评测涵盖MMLU、C‑Eval、TriviaQA、BBH、MATH、DROP、GSM8K、ARC‑C、HellaSwag、WinoGrande、Repo Code PPL、RULER、NoLiMa、GraphWalks、AgentPPL、LongPPL等标准与内部基准。

**📈 对比分析**

与HySparse和Hybrid SWA做对比：在保持大部分通用能力的前提下，HySparse2在MRCR‑v2、RULER‑v2、AgentPPL、LongPPL等长上下文任务上提升了10–20点；预填充FLOPs降低2.9×/5.0×，KV缓存仅2.69 GB；整体性能优于两基线。

**⚠️ 局限性**

局限：仍依赖少量全注意力层进行索引，部分任务（如DROP）表现略逊；模型结构更复杂，训练成本和实现细节相对较高；需进一步探索更高效的全注意力替代方案与更大规模验证。

---

## 633. The parameterised complexity of generalised temporal domination on temporal graphs with modular structure

**arXiv ID:** 2609.26366 | [PDF](https://arxiv.org/pdf/2609.26366v1)

**作者:** Jessica Enright `[一作]` (University of Glasgow), Elena Moss `[通讯]` (University of Glasgow)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究(α,β)-Temporal Dominating Set的参数化复杂度，并证明其在临时邻域多样性（TND）上是固定参数可解（FPT）的；

**💡 创新点**

提出(α,β)-tds的定义并将其归类为snapshot-set问题，首次在TND下给出FPT算法，且在TMW/TCW下证明W[1]-难度及部分para-NP难点；

**🔧 技术方法**

利用整数线性规划（ILP）编码和唯一快照预处理，将问题转化为TND受限的ILP求解；

**📊 数据集**

论文不使用真实数据集，全部采用理论构造与归约来证明复杂度；

**📈 对比分析**

通过理论证明与静态Dominating Set的对应关系，FPT结果表明在TND下可在指数级时间内解决，而W[1]-难度显示在更一般参数下不可；

**⚠️ 局限性**

仅在TND上获得FPT，TMW/TCW的部分情况仍未解决，缺乏实验验证，且对特定β值的完整性仍是开放问题。

---

## 634. PP-Net: A Hybrid Physical-Prior Neural Network for Scattered Light Removal in Biomedical Images on Embedded Devices

**arXiv ID:** 2609.26474 | [PDF](https://arxiv.org/pdf/2609.26474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 635. Reproducible AI Requires Reproducible Randomness

**arXiv ID:** 2609.26461 | [PDF](https://arxiv.org/pdf/2609.26461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 636. Code Plans, Diffusion Renders: Open-Ended Generative World Modeling

**arXiv ID:** 2609.26458 | [PDF](https://arxiv.org/pdf/2609.26458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 637. DeepFEAv2: Deep Learning for Transient Finite Element Analysis Beyond Structured Meshes

**arXiv ID:** 2609.26426 | [PDF](https://arxiv.org/pdf/2609.26426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 638. Spoken Language Models that Think Aloud

**arXiv ID:** 2609.26488 | [PDF](https://arxiv.org/pdf/2609.26488v1)

**作者:** Junyi Ao `[一作]` (Meta Superintelligence Labs), Xubo Liu `[通讯]` (Meta Superintelligence Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种异步 think‑aloud 框架，将内部推理与语音输出解耦，实时给出进度反馈，显著减少无声等待。

**💡 创新点**

创新点在于：①使用轻量级 think‑aloud 模块与主推理者并行工作；②设计动态平衡策略，在推理与说话之间自动调度触发或取消进度语句；③通过 TA_trigger 标记推理里程碑，确保最终答案与进度语句一致。

**🔧 技术方法**

核心技术包括 Thinker‑Talker 架构、0.5B LLM 作为 think‑aloud 模块、CosyVoice 2.0 统一说话器、投影层对齐两种语义空间、以及动态平衡算法控制发声与推理的同步。

**📊 数据集**

使用自研 200k 语音对话数据（约 5,000 小时）进行训练，并在公开的 Spoken‑MQA、Web Questions、TriviaQA 语音/文本基准上进行评估。

**📈 对比分析**

与无推理基线和串行 think‑then‑speak Baseline+CoT 进行对比；在 Spoken‑MQA 单步/多步准确率分别为 87.6%/83.6% 与 Baseline+CoT 的 88.5%/85.6%；在 Web Questions/S2S 的准确率为 40.3% 与 Baseline+CoT 同等；无声延迟从 12.82 s 降至 0.05–0.36 s，且人类评测显示 99.3% 的样本认为响应性更好。

**⚠️ 局限性**

局限性包括：无法处理外部中断或错误修正的实时交互；动态平衡策略对推理速度估计的依赖；训练数据不可公开；未与全双工或并行监听-说话机制直接结合。

---

## 639. Behavior is Not Enough: A Mechanism-Based Evaluation of Social Norm Emergence in LLM Societies

**arXiv ID:** 2609.26481 | [PDF](https://arxiv.org/pdf/2609.26481v1)

**作者:** Rasika Muralidharan `[一作]` (Indiana University Bloomington), Jisun An `[通讯]` (Indiana University Bloomington)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对多代理LLM系统中的社会规范演化进行实验，构建期望提取、社交学习和社交选择的消融框架，并在公共物品游戏中评估其对合作的影响。

**💡 创新点**

首次将代理的经验性与规范性期望公开化，区分不同社会机制对合作结果的贡献，并通过对抗性干扰测试鲁棒性。

**🔧 技术方法**

使用 GPT‑4o‑mini、Llama 3.1‑8B、Qwen 2.5‑7B、Mistral‑7B 四大语言模型，结合公共物品游戏、对话式社交学习、网络权重更新的社交选择以及 OLS/混合效应统计分析。

**📊 数据集**

未使用外部真实数据，全部实验基于模拟生成的 12 代理公共物品游戏序列，10 次独立运行、四种模型与五种实验条件共计 400 条结果。

**📈 对比分析**

通过 OLS 估计各机制对平均贡献的显著性；期望提取提升合作显著（p<0.001），社交学习对行为稳定性贡献最大，社交选择主要改善协作者识别，且在对抗性干扰下恢复速度差异明显。

**⚠️ 局限性**

研究仅采用简化的公共物品游戏，缺乏真实社会情境；期望提取依赖提示而非内部模型；未考虑领导、信号或社会身份等其他重要社会机制。

---

## 640. Recursive self-improvement of AI research agents

**arXiv ID:** 2609.26457 | [PDF](https://arxiv.org/pdf/2609.26457v1)

**作者:** Dhruv Srikanth `[一作]` (Weco AI), Zhengyao Jiang `[通讯]` (Weco AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个双循环递归自我改进（RSI）系统，使用 LLM 驱动的树搜索在 AI 研发的 harness 层上自动改进 agent 的代码和策略；

**💡 创新点**

首次实现了基于 bi‑level 优化的自我改进框架，能在固定计算预算内持续产生可泛化的性能提升，并显著降低 reward hacking；

**🔧 技术方法**

利用 AIDE 风格的树搜索、Bandit 策略、上下文压缩、错误记忆、强化学习式评价等技术；

**📊 数据集**

在内部选择基准（ML 训练、启发式算法、harness 设计）和外部评测基准（ALE‑Bench、MLE‑Bench、FML‑Bench、WeatherBench‑2、KernelBench）上训练与评估；

**📈 对比分析**

与人类驱动研发的 AIDE（AIDE85）对比，内部基准 grade 从 0.703 提升至 0.778；在四个外部基准上，AIDE85 与 AIDE 基线相当或更优，reward‑hacking 率从 55% 降至 32%；

**⚠️ 局限性**

噪声累积导致外部循环不确定，样本效率低；改进后的 agent 复杂度高，缺乏可解释性，部署时对兼容性与资源约束挑战显著。

---

## 641. Don't let your Memory defy you: Fragmentation-Aware Serverless Allocation with Elastic Memory Locality

**arXiv ID:** 2609.26476 | [PDF](https://arxiv.org/pdf/2609.26476v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 642. Enriching Speech Emotion Representations with Conversational Context

**arXiv ID:** 2609.26422 | [PDF](https://arxiv.org/pdf/2609.26422v1)

**作者:** Arthur Peuvot `[一作]` (Université Paris-Saclay), Ioana Vasilescu `[通讯]` (LISN)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出ACERT模块，通过对对话上下文的平均池化来增强语音情感识别的目标说话单元表示，且不依赖说话人信息。

**💡 创新点**

创新点在于引入可变长度的前置上下文窗口并通过简单的平均池化与残差连接融合，显著提升情感连续性建模。

**🔧 技术方法**

使用自监督声学特征提取器HuBERT-large、线性映射、ReLU、LayerNorm、FFN以及时间平均池化等深度学习技术。

**📊 数据集**

在IEMOCAP、SAFE与MELD三个多样化对话语料上进行评估，涵盖二人对话、电影剧本及多说话人情景。

**📈 对比分析**

与无上下文基线及现有SOTA方法比较，ACERT在IEMOCAP上UA提升至79.08%（比SOTA高约3.2个百分点），在SAFE上突破基线4–5个百分点，在MELD上实现或超过同类方法的Macro‑F1。

**⚠️ 局限性**

局限性主要体现在情绪变化快、说话人数多的场景（如MELD）中，上下文对性能提升不足，说明对情绪连续性要求较高。

---

## 643. OMatG-flash: An All-Atom Flow Map with Reinforce Adjoint Matching for Scalable Materials Discovery

**arXiv ID:** 2609.26402 | [PDF](https://arxiv.org/pdf/2609.26402v1)

**作者:** Thomas Egg `[一作]` (New York University), Stefano Martiniani `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了OMatG-flash，一种面向无机晶体生成的全原子流图模型，并对其进行了后训练的强化学习调优。

**💡 创新点**

创新点在于：① 用Transformer驱动的流图实现了一阶采样，显著提升推理速度；② 采用Reinforce Adjoint Matching (RAM) 对流图进行后训练，使其在晶体结构预测(CSP)任务上达到或超过现有扩散/流模型的性能；③ 通过自蒸馏训练，无需教师模型即可获得高质量流图。

**🔧 技术方法**

核心技术包括：Riemannian 轨道插值、流图的切线与一致性损失、Transformer（SiT）实现、Reinforce Adjoint Matching 的后训练算法。

**📊 数据集**

使用了两个公开无机材料数据集：MP‑20（45,231 条晶体）和 Alex‑MP‑20（675,204 条晶体），并在 MP‑20‑ps（多形态分割）上评估。

**📈 对比分析**

与 DiffCSP、FlowMM、MCFlow、Crystalite 等基线模型对比，OMatG‑flash 在一阶采样时的吞吐量提升约10×，在 METRe、cRMSE、S.U.N.、M.S.U.N. 等多项指标上与或优于现有最佳模型；后训练版本在 CSP 任务上实现了 state‑of‑the‑art 的 METRe 与 cRMSE。

**⚠️ 局限性**

主要局限在于基线模型的 RMSE 较高，说明对细粒度结构的把握仍不足；后训练虽能缓解，但对整体质量提升的空间仍大；此外，对多形态生成的鲁棒性和对不同奖励函数的适应性尚需进一步研究。

---

## 644. FeatLens: Feature-Guided Dynamic Code Graph Construction and Retrieval for Repository-Level Code Generation

**arXiv ID:** 2609.26480 | [PDF](https://arxiv.org/pdf/2609.26480v1)

**作者:** Xutian Li `[一作]` (Peking University), Bing Xie `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了FeatLens，一个基于特征引导的动态代码图构建与检索框架，用于在现有代码仓库中完成函数实现。

**💡 创新点**

创新点在于：①通过离线特征索引将自然语言功能描述映射到函数级别代码实体；②动态构造任务特定的种子图；③采用语义-结构化图推理和个性化PageRank筛选紧凑推理图，从而避免了完整仓库图维护和LLM驱动的仓库探索。

**🔧 技术方法**

使用了静态分析、函数级语义描述、层次聚类构建特征索引、语义-结构化图推理、个性化PageRank、轻量级代码嵌入等技术。

**📊 数据集**

在DevEval和EvoCodeBench这两个面向Python的仓库级代码生成基准上进行实验。

**📈 对比分析**

与无上下文、BM25检索、UniXcoder检索、RepoGraph和CodexGraph等基线相比，FeatLens在DR@15上取得最高0.501、DIR@1最高53.58%、Pass@1保持竞争力且代码更短；在效率上相较CodexGraph，图节点减少61%、边缘减少86%、总token消耗降低45.9%。

**⚠️ 局限性**

局限性包括：依赖特征聚类的质量和静态分析的完整性，可能忽略运行时绑定和隐式框架约定；实验仅覆盖Python仓库，未验证跨语言或闭源工业仓库的适用性。

---

## 645. Frugal Collective Perception: Context-Aware Adaptive Reporting for Safety-Critical C-ITS

**arXiv ID:** 2609.26470 | [PDF](https://arxiv.org/pdf/2609.26470v1)

**作者:** Romain Tessier `[一作]` (Institut Polytechnique de Paris), Adriana Tapus `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于上下文感知的自适应滤波器，动态调整集体感知服务（CPS）中CPM的内容和发送频率，以提高安全性并降低通信负载。

**💡 创新点**

创新点在于：
1) 采用三层交通情境表示（地理、拓扑、语义），对检测对象进行情境相关性评估；
2) 基于情境重要性和事件频率实现双速率通信，兼顾实时性和信息冗余；
3) 通过自适应调节发送频率与对象优先级，显著减少通信量同时保持安全性能。

**🔧 技术方法**

技术手段包括：
- ETSI CPM标准与V2X通信框架；
- 语义地图与场景图的构建与推理；
- 端到端SUMO–Artery仿真平台；
- 采用Complex‑YOLOv4（tiny）实现本地感知；
- 决策模块基于有限状态机与上下文感知融合；
- 对感知、解码与融合的计算模型进行性能分析。

**📊 数据集**

使用数据集：
- KITTI数据集用于估算YOLOv4推理时延（α参数）；
- 其余评估在SUMO生成的合成交通场景中完成，未使用真实道路记录。

**📈 对比分析**

比较方法：将自适应滤波器与传统CPS（1 Hz、5 Hz、10 Hz）在相同场景与硬件平台（Jetson TX2、Xavier NX、AGX Xavier）下进行对比。评价指标包括：
- 安全性：TTC、PET分布；
- 决策效率：队列占用、包丢失率、AoI；
- 通信成本：包数与总数据量。
结果显示：自适应滤波器在安全指标上与10 Hz CPS相当，但通信量减少93%以上，且在低端硬件上避免了队列饱和与信息丢失。

**⚠️ 局限性**

局限性：
- 相关性计算机制仍停留在概念层面，未给出通用的实现细节；
- 评估仅基于仿真，缺乏真实车辆实验验证；
- 采用固定的感知/融合模型参数，实际系统可能需要自适应校准；
- 仅关注晚期融合，未探讨特征级或早期融合对带宽/安全的影响；
- 对动态网络管理和更广泛协作任务的支持尚未展开。

---

## 646. Sample, Simulate, Select: Physics-in-the-Loop Text-to-Motion for Humanoids Without Training

**arXiv ID:** 2609.26420 | [PDF](https://arxiv.org/pdf/2609.26420v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 647. How to Estimate Whether You Have Found Several Needles in a Haystack: Measuring Calibration in Multi-Label Text Classification

**arXiv ID:** 2609.26468 | [PDF](https://arxiv.org/pdf/2609.26468v1)

**作者:** Sophie Henning `[一作]` (Technical University of Munich), Annemarie Friedrich `[通讯]` (University of Augsburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种专门针对多标签分类的校准度量和新的自适应分箱方案，解决了传统分箱在负样本占优时导致的误差低估问题。

**💡 创新点**

创新点在于将正负样本分别分箱并保持相等权重，从而得到更可靠的每标签期望校准误差（ECE_ML），并为多标签任务提供了可解释的可靠性图。

**🔧 技术方法**

使用自适应分箱、BCE、WBCE、WBCEP、Focal Loss 等损失函数，以及BERT、BiomedBERT、ModernBERT、HiDEC 等判别模型，和LLM（Llama‑3.3‑70B、Qwen2.5‑72B）在零样本与检索增强下的生成式分类。

**📊 数据集**

在两个大型不平衡多标签文本数据集上实验：RCV1‑v2（新闻主题层次标签）和MIMIC‑III‑v1.4（医学编码）。

**📈 对比分析**

与传统固定宽度或单次自适应分箱相比，ECE_ML在所有频率组中均显示更稳定、更低的校准误差；Fine‑tuned判别模型在宏观指标上优于LLM，且计算速度快约四至五千倍。

**⚠️ 局限性**

局限性包括：ECE_ML仍基于金标准标签，忽略人类标注者间的不一致；对LLM的置信度提取仅限于当前主流方法，未来可能需要更精细的置信度估计；在极稀有标签下仍可能出现分箱不足的问题。

---

## 648. The Most Informative Bit and Beyond: A Proof of the Courtade--Kumar Conjecture and Multibit Extensions

**arXiv ID:** 2609.26444 | [PDF](https://arxiv.org/pdf/2609.26444v1)

**作者:** Hessam Mahdavifar `[一作]` (Northeastern University), Ahmad Beirami `[通讯]` (Fidian)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

证明了任意布尔函数在均匀输入下通过二元对称信道的单比特量化不超过坐标投影的互信息，并给出了多比特量化的阈值行为；

**💡 创新点**

首次用动量法与熵流标识将信息增益与信道相关性关联，完成了无平衡假设下的完整证明，并揭示k≥8时坐标投影失效；

**🔧 技术方法**

利用熵流恒等式、傅里叶分析、对称压缩、对角切线校正以及对数Sobolev不等式的组合；

**📊 数据集**

主要使用二元对称信道模型，无需外部实验数据；

**📈 对比分析**

通过与坐标投影基准比较，单比特情形完全满足上界，多比特情况下k≥8的Hamming编码可超越基准，而k≤7仍未出现超越；

**⚠️ 局限性**

对多比特量化的最优性仅在k≤7内提出假设，缺乏严格证明；

---

## 649. One-Step Generative Surrogate Models via Block-Triangular Joint Drifting

**arXiv ID:** 2609.26435 | [PDF](https://arxiv.org/pdf/2609.26435v1)

**作者:** Nicholas Geissler `[一作]` (New York University), Benjamin Peherstorfer `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于漂移（drifting）和块三角联合分布的生成模型，用以直接从单步轨迹数据学习随机动力学的一阶转移分布，实现每个时间步仅一次网络评估即可生成完整轨迹。

**💡 创新点**

创新点在于：① 将漂移方法应用到状态对的联合分布而非单个条件分布；② 采用块三角网络结构，使模型第一分量保持当前状态边缘分布，第二分量直接成为条件采样器；③ 对漂移场进行投影，只更新第二块，从而实现条件采样的闭合形式；④ 通过这种结构即可在缺乏多次条件样本的情况下完成条件分布学习。

**🔧 技术方法**

使用技术包括：漂移（drifting）学习框架、Sinkhorn Divergence与其一阶变分的梯度场、块三角网络（MLP/CNN/UNet 等）、投影操作、MSE 损失、AdamW 优化、梯度步骤训练、对数似然的 Sinkhorn 迭代、以及在高维空间中使用自编码/MAE 进行潜在空间嵌入。

**📊 数据集**

实验数据集覆盖多种随机动力学：Duffing 振荡器、Rayleigh‑Bénard 对流、随机 Burgers 方程、二维湍流以及相应的高维数值模拟，均通过数值积分生成大批量轨迹。

**📈 对比分析**

与多步扩散/流模型（ARDM、CFM）、一、少步蒸馏模型（MeanFlow、ReFlow+Distill）、SDE 学习方法、以及确定性或边缘匹配模型比较。结果显示 BTJD 在时间边缘（Sliced W2）、轨迹 QoI（能量、耗能、旋转流量、跨越次数等）误差上均优于或相当于最先进方法，并且仅需每步一次网络评估，显著降低了推理成本。

**⚠️ 局限性**

局限性：① 需要足够多且高质量的轨迹对样本，且假设每步观测到的后继状态是独立采样；② 块三角结构对模型容量和表达能力有要求，可能在极高维或复杂耦合系统中需更深或更复杂网络；③ 目前实验多聚焦在离散时间数值模拟，连续时间理论仍有待进一步完善；④ 对模型参数（如 Sinkhorn 正则化 ε、步长 h）敏感，需要经验调优。

---

## 650. Double Descent and Malign Overfitting in Diffusion Models

**arXiv ID:** 2609.26392 | [PDF](https://arxiv.org/pdf/2609.26392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 651. Learning to Defer with Guidance on Real World Medical Data

**arXiv ID:** 2609.26384 | [PDF](https://arxiv.org/pdf/2609.26384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 652. MAVP: Map-Aware Visuomotor Policies for Mobile Manipulation

**arXiv ID:** 2609.26378 | [PDF](https://arxiv.org/pdf/2609.26378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 653. Support-Primitive Decomposition of Constacyclic Codes over Finite Fields: Coefficients-Based and Roots-Based Descriptions

**arXiv ID:** 2609.26414 | [PDF](https://arxiv.org/pdf/2609.26414v1)

**作者:** Li Zhu `[一作]` (Guizhou Normal University), Hongfeng Wu `[通讯]` (North China University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

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

模型在处理大规模数据集时可能会遇到内存限制的问题。

---

## 654. Dr-LiSA: Direct Radar-Lidar Scan Alignment for $SE(3)$ Localization

**arXiv ID:** 2609.26423 | [PDF](https://arxiv.org/pdf/2609.26423v1)

**作者:** Alex Zhang `[一作]` (University of Toronto), Timothy D. Barfoot `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Dr‑LiSA，一种利用 2D 旋转雷达强度图像实现对 3D 激光雷达地图的 SE(3) 定位方法；

**💡 创新点**

创新点在于：①首次在 SE(3) 上使用旋转雷达进行直接定位；②通过学习的雷达‑激光前向模型将 3D 地图几何映射为雷达强度，从而实现稠密光度对齐；③利用 3D 场景信息提升非平面环境下的定位鲁棒性；

**🔧 技术方法**

核心技术包括：1) 通过卷积+Transformer 的 Encoder‑Decoder 结合几何引导 Skip 分支实现雷达强度预测；2) 采用共视失真（co‑visibility）损失进行模型训练；3) 以 Implicit Filtering (IMFIL) 迭代优化雷达位姿；4) 使用 Teach‑and‑Repeat 方式构建激光子图并进行预渲染；

**📊 数据集**

使用 Boreas Road Trip (Boreas‑RT) 数据集，包括 Velodyne Alpha‑Prime 激光雷达、Navtech RAS6 旋转雷达、IMU、RTK‑GNSS/INS 真实位姿；

**📈 对比分析**

与三种基线比较：SOTA 雷达‑激光（RLT&R）、SOTA 雷达‑雷达（DRL‑Dr‑BA）以及 SOTA 激光‑激光（2Fast‑2Lamaa）。在 90+ km 真实行驶数据上，Dr‑LiSA 在 SE(2) 平面误差上优于 RLT&R，且在平面精度与 DRL‑Dr‑BA 相当，同时还能估计竖直、俯仰、滚转；在特征稀疏、非平面路段表现尤为突出；

**⚠️ 局限性**

局限性包括：①实时性不足，单帧平均 2.3 秒；②垂直、滚转、俯仰误差仍高于激光基线；③依赖较高的前向模型计算量与光线追踪，需进一步加速。

---

## 655. Layout-Guided Masking for GROBID: Lightweight Structural Gains in Large-Scale Scientific PDF Ingestion

**arXiv ID:** 2609.26381 | [PDF](https://arxiv.org/pdf/2609.26381v1)

**作者:** Luca Foppiano `[一作]` (ScienciaLAB), Vipul Gupta `[通讯]` (Helmholtz-Zentrum Hereon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种在 CPU 上运行的字体流解析器 LAYOUT‑MASKED GROBID，并加入轻量级目标检测器，用于定位图表、表格和页眉页脚等可视化区域；随后根据检测结果将文本块路由到专门的模型进行结构化，最终实现更高质量的全文解析。

**💡 创新点**

创新点在于：①利用轻量级检测器仅在视觉上复杂的区域做定位，避免了全页 GPU 计算；②引入 typed‑area masks 机制，将检测到的区域信息直接嵌入字体流解析过程，实现区域级别的模型路由；③在不依赖 GPU 的情况下，使传统字体流解析在复杂页面上表现接近或超过现有 vision‑based 系统。

**🔧 技术方法**

技术方法包括：基于字体流的 token‑label cascade 模型（GROBID 基础），轻量级视觉检测器（如 YOLO‑tiny 等）用于快速定位图表与页眉页脚，typed‑area mask 生成与路由机制，以及与专门表格、公式识别器的协同。

**📊 数据集**

使用的数据集主要是 PubMed Central (PMC) 的两大科研领域语料：Bioinformatics（1926 篇）和 Materials Science（2595 篇），并在 Table‑BRGM 公开基准上评估表格检测性能；此外与四个代表性 vision‑based 系统进行对比实验。

**📈 对比分析**

在与 JATS XML 的结构化协议对照下，LAYOUT‑MASKED GROBID 在段落回召率、章节检测、字符错误率（CER）等指标上显著优于原 GROBID，并在 Table‑BRGM 基准上将表格检测 F1 从 0.16 提升至 0.94，表格结构化性能提升至 0.78（虽低于最佳 GPU 系统）。在与四个 vision‑based 系统对比时，本文方法在段落精度、章节识别和 CPU 成本上均表现最好；总的 CPU 端点耗时仅为最便宜 GPU 系统的 2.7–3.2 倍，远低于生成式解析器。

**⚠️ 局限性**

局限性包括：①对极其复杂或非标准排版的页面仍可能出现漏检或误识别；②表格结构化仍不及最先进的 GPU 系统；③依赖检测器准确率，若检测误差增大则会影响后续路由；④实验范围仅覆盖 PMC 的两个学科，其他文献类型的泛化性能尚未充分验证。

---

## 656. Approximation Algorithm for the Min-Cost Bipartite Matching with Penalties

**arXiv ID:** 2609.26369 | [PDF](https://arxiv.org/pdf/2609.26369v1)

**作者:** Eunjin Oh `[一作]` (Pohang University of Science and Technology), Chanho Song `[通讯]` (Pohang University of Science and Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在有限双曲维度度量空间中求解带惩罚的最小成本二分匹配问题的近似算法，能够在近线性时间内得到(1+ε)近似解；

**💡 创新点**

首次在惩罚匹配设置下实现近线性时间的(1+ε)近似，且将问题转化为双层棱镜图的完美匹配，并通过调整增广路径框架实现；

**🔧 技术方法**

利用随机化分裂树（split‑tree）近似距离、棱镜图构造、增广路径的净成本定义、对称增广路径以及基于Raghvendra‑Agarwal的数据结构改造，实现高效的增广搜索与更新；

**📊 数据集**

论文未给出实验数据集，主要为理论分析与算法证明；

**📈 对比分析**

与已有的完美匹配与k‑部分匹配等方法对比，提出的算法在常数双曲维度下实现O(n(log n, 1/ε))时间的(1+ε)近似，显著优于先前多项式或超线性时间的结果；

**⚠️ 局限性**

仅适用于常数双曲维度的度量空间；算法是随机化的，仅在高概率下正确；并且需要预处理以保证输入点集的分布范围受限（bounded spread）。

---

## 657. From Token Importance to Conditional Removability: Rethinking Visual Token Pruning in Multimodal Large Language Models

**arXiv ID:** 2609.26484 | [PDF](https://arxiv.org/pdf/2609.26484v1)

**作者:** Shengli He `[一作]` (Guizhou University), Li Zheng `[通讯]` (Guizhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 CoRePrune 的训练无关两阶段视觉标记剪枝框架，通过考虑深度和删除上下文来判断可删标记。

**💡 创新点**

创新点在于将剪枝视为“条件可删性”而非仅仅依赖重要性，提出渐进式扰动感知剪枝和集合条件微调两阶段方法，显著提升了在不同网络深度和删除集上的剪枝稳定性。

**🔧 技术方法**

技术包括：注意力输出扰动度量（基于 CAOTE 形式）、单点与集合扰动的差分评估、按层逐步更新单点分数、批量逆贪心集合条件微调，以及针对多模态大语言模型的视觉-文本交互后期剪枝。

**📊 数据集**

使用了多种数据集：图像级别的 VQAv2、GQA、VizWiz、ScienceQA-IMG、TextVQA、POPE、MME、MMBench-EN/CN、MMVet；视频级别的 MVBench、LongVideoBench、Video-MME；以及用于基准对比的 AI2D、ChartQA、OCRBench 等。

**📈 对比分析**

与 FastV、SparseVLM、VisionZip、DART、DivPrune、CDPruner、PruneSID、MMTok、QCPruner、FastVID 等方法比较，CoRePrune 在多数模型与预算点上实现了更高的 Avg. Rel.（相较效率基准提升 0.4–1.5 点，紧凑预算下可达 14.3 点），且预填充时间减少约 51%（对 Qwen3.5 预算 128 时，平均保留 90.3% 的性能，预填充时间从 323.4s 降至 158.3s）。

**⚠️ 局限性**

局限性包括：对中间候选预算 K_V 的选择敏感；在极端压缩（如 K_L 极低）下仍存在性能损失；仅适用于冻结模型，无法直接结合训练或校准；实现依赖多层注意力的存储和递归更新，虽然开销小但在大规模部署时仍需考虑；以及对不同网络架构的深度调度尚未实现自适应化。

---

## 658. Latent Dataset Distillation for Human Motion Prediction

**arXiv ID:** 2609.26430 | [PDF](https://arxiv.org/pdf/2609.26430v1)

**作者:** Ge Tian `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一种基于残差量化变分自编码器（RVQ‑VAE）的隐空间数据蒸馏框架，用于人类运动预测模型的训练；

**💡 创新点**

创新点在于将梯度匹配过程迁移到预训练的隐空间，仅优化一个可学习的隐向量集合，并通过冻结的量化器和解码器引入运动先验，从而显著降低合成动作的非物理性与不稳定性；

**🔧 技术方法**

核心技术包括残差量化的VAE（RVQ‑VAE）、梯度匹配（Gradient Matching）、直通估计（Straight‑Through Estimator）以及多阶段代码本的残差量化；

**📊 数据集**

在Human3.6M、CMU Graphics Lab Motion Capture 和3DPW 三个标准运动数据集上进行实验；

**📈 对比分析**

与全量训练、随机子集选择以及直接梯度匹配等基线进行比较，在1×合成数据预算下，本文方法在大多数实验设置下平均 MPJPE 下降约 7%–50%，并在 30 个评估时间点中超过直接梯度匹配 27 次，效果稳定；

**⚠️ 局限性**

主要局限是：①只在固定预算下评估，未探究更大规模或多尺度蒸馏；②对不同预测框架的迁移性提升有限；③训练时需预训练 RVQ‑VAE，增加了整体工作量。

---

## 659. FairMean: Promoting Fairness in Distributed Learning under Label Poisoning Attacks

**arXiv ID:** 2609.26377 | [PDF](https://arxiv.org/pdf/2609.26377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 660. The Source of Disturbance Matters: External, Internal, and Control-Generated Noise in Adaptive Regulation

**arXiv ID:** 2609.26428 | [PDF](https://arxiv.org/pdf/2609.26428v1)

**作者:** Veronique Ziegler `[一作]` `[通讯]` (Independent Researcher, Critical Attention Systems), Veronique Ziegler (Independent Researcher, Critical Attention Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一个简化的自适应不确定性调节模拟框架（IRAM-Ω-Q）中，研究了三种扰动来源（外部扰动、持久内部扰动、控制生成扰动）以及它们在“先调节 vs 先扰动”时间顺序下对系统曝光度与调节负荷的影响。

**💡 创新点**

创新点在于将扰动源的结构与时间顺序作为关键变量系统化比较，发现控制生成的扰动在成本增大时会出现非单调响应，并且内部扰动的持久性能显著放大调节负荷。

**🔧 技术方法**

使用量子式状态表示的 IRAM-Ω-Q 模型、马尔可夫过程生成内部扰动、以及基于正增量平方的控制生成扰动函数，配合 10-50 次随机仿真复制进行统计分析。

**📊 数据集**

数据来源为自生成的仿真数据，没有使用外部公开数据集；通过多重参数扫描（如 ρ、σ_in、α）构建了不同扰动强度与持久性组合的实验场景。

**📈 对比分析**

比较方法采用 RF（先调节）与 DF（先扰动）两种时间顺序的对比，使用平均有效扰动、平均调节强度、控制步幅、相干间隙方差等指标，并绘制配对置信区间。结果显示持久内部扰动导致最大曝光度和调节负荷；控制生成扰动在成本阈值处出现曝光度提升与调节抑制的非单调转折。

**⚠️ 局限性**

局限性包括仅在有限的参数网格内验证，未探究更复杂或真实任务情境下的表现；控制生成扰动仅采用单侧二次成本函数，其他成本形式可能改变结果；混合扰动分析保守，缺乏严格的非线性交互模型；未直接评估任务性能或稳定性指标。

---

## 661. Reliability Theory for AI Control

**arXiv ID:** 2609.26419 | [PDF](https://arxiv.org/pdf/2609.26419v1)

**作者:** Grant Molnar `[一作]` `[通讯]`, Grant Molnar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文将成熟的可靠性理论方法应用于Google DeepMind提出的AI控制体系（GDM），通过构建安全可靠性结构图、最小路径/切断分析以及Birnbaum重要性度量，对系统的预防与恢复层级进行定量评估。

**💡 创新点**

创新点在于：1）首次将可靠性理论中的最小切断、Birnbaum重要性与AI安全架构结合；2）引入“失败域”概念，将基础故障事件与可见安全组件关联，揭示常见因果失效对系统安全级别的影响；3）提出预防失败后恢复需求的选择效应模型，说明防御强度对恢复测试的影响。

**🔧 技术方法**

技术手段包括：可靠性结构函数、最小路径/切断分析、Birnbaum重要性计算、基本事件映射与共因分析、条件恢复可靠性评估、以及概率论与贝叶斯推断。

**📊 数据集**

论文主要以Google DeepMind公开的GDM路线图和架构为案例进行理论演示，并未使用传统机器学习数据集；若要验证，可结合GDM内部安全事件日志或红队攻击实验数据。

**📈 对比分析**

方法与传统的安全评估手段（如单独的漏洞扫描、监控覆盖率统计）相比，能够量化系统整体可靠性、揭示不同安全层级的重要性，并指导资源分配与改进。性能上，本文通过公式化分析给出了系统失败概率的阶数（如从三阶降至二阶或一阶），体现了常因失效对风险的巨大影响。

**⚠️ 局限性**

局限性包括：1）模型假设基础故障事件独立且概率极低，实际系统可能存在更复杂的依赖关系；2）未考虑动态变化、资源约束、学习适应等实际情况；3）缺乏实测数据验证，理论推导需在实际部署中进一步实验验证。

---

## 662. RouteRLT: Learning When and Which RL Specialist Should Control a Vision-Language-Action Policy

**arXiv ID:** 2609.26467 | [PDF](https://arxiv.org/pdf/2609.26467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 663. MATE: Multi-Agent Virtual Teleoperation Platform for Humanoid Collaboration Data Collection

**arXiv ID:** 2609.26520 | [PDF](https://arxiv.org/pdf/2609.26520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 664. Bespoke: Generating MOOC-Quality Industry-Personalized Lecture Videos at Scale

**arXiv ID:** 2609.26540 | [PDF](https://arxiv.org/pdf/2609.26540v1)

**作者:** Romain Puech `[一作]` (Massachusetts Institute of Technology), Dimitris Bertsimas `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一套名为Bespoke的端到端系统，从已有讲座转录自动生成针对特定行业的完整视频讲座（包括新的幻灯片、叙述和图表）。

**💡 创新点**

实现了在不需讲师手工重写的前提下，通过多模型交互（生成‑验证‑改进）和领域检索，自动为不同专业观众重塑内容、示例和深度，真正实现行业化个性化。

**🔧 技术方法**

使用Anthropic Claude Sonnet 4.6进行内容生成、OpenAI GPT‑5.2做验证，结合后向设计与 Mayer 多媒体原则，构建五阶段流水线（目标、规划、叙述、视觉生成、渐进展示），并以 HTML/代码化幻灯片与图表方式输出。

**📊 数据集**

基于研究生级分析/机器学习课程的31节讲座转录作为种子，实验生成200+个视频，其中92个被专家评审，涵盖医疗、金融、能源和通用受众，时间档位为5–15分钟与约45分钟两种。

**📈 对比分析**

采用25位领域专家按5分量表评估内容、个性化、教学效果、制作质量与整体质量；结果显示87%的视频达标“可与标准MOOC相当”，平均整体评分3.42/5，行业定制在个性化深度上略有提升，API成本约0.22美元/分钟。

**⚠️ 局限性**

缺乏学习效果与参与度的实证评估，专家评分为单人次且无互评信度；个性化在受众校准方面仍有限；音频语速、幻灯片同步与排版是主要技术瓶颈。

---

## 665. Transcribe, Translate, and Optimize: Joint Reward Learning for Speech Translation

**arXiv ID:** 2609.26536 | [PDF](https://arxiv.org/pdf/2609.26536v1)

**作者:** Yanghe Dong `[一作]`, Weiran Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用组相对策略优化（GRPO）联合微调大语言模型，实现同时识别与翻译的链式思考（CoT）语音翻译，提升识别准确率与翻译质量。

**💡 创新点**

提出在 CoT 语音翻译中同时优化识别与翻译奖励，并设计三种 token 优势分配策略（Fully Coupled、Asymmetric、Decoupled），通过 GRPO 有效缓解训练与推理不匹配问题。

**🔧 技术方法**

使用 Qwen2.5‑Omni‑3B 作为基模型，采用 DAPO 变体的 GRPO、token‑级优势分配、WER 作为 ASR 奖励、chrF++ 作为翻译奖励，并引入参考‑aware 训练。

**📊 数据集**

在 CoVoST 2 训练集上训练，评估时使用 CoVoST 2 和跨语料库 FLEURS 数据集。

**📈 对比分析**

与直接语音翻译（Direct ST）及传统 SFT 进行对比；CoT GRPO 在 CoVoST 2 上平均 BLEU 提升 1.77 分、FLEURS 提升 0.83 分，同时 WER 分别下降 8.8% 与 7.2%；相较于 CoT SFT，GRPO 同时提高翻译和识别性能，证明联合奖励策略的有效性。

**⚠️ 局限性**

仅在 Qwen2.5‑Omni‑3B 上验证，语言方向和模型规模受限；需进一步测试更大规模模型、多语言对，并评估在不同文本/口音环境下的泛化能力。

---

## 666. Acceptance & Rejection

**arXiv ID:** 2609.26557 | [PDF](https://arxiv.org/pdf/2609.26557v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 667. A $59/33$ Cut-LP Guarantee for Matching Augmentation

**arXiv ID:** 2609.26531 | [PDF](https://arxiv.org/pdf/2609.26531v1)

**作者:** Morteza Alimi `[一作]` (University of Augsburg), Tobias Mömke `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并分析了 Bamas‑Drygala‑Svensson 的 LP‑引导深度优先搜索算法，证明其在匹配增补问题（MAP）中的积分间隙上限为 59/33 ≈ 1.788。

**💡 创新点**

创新点在于提出一种全新的结构分析框架：结合残差上链问题的精确封装-填充对偶、基于单元值骨架的分数支持秩界定以及自洞切割恒等式，显著提升了原有算法的理论证明而不改变算法本身。

**🔧 技术方法**

主要技术包括：LP 预处理（求极点解）、基于 LP 坐标的优先 DFS、精确上链最小覆盖与其填充对偶、最小割维数定理、分数支持体积界定、区域三分法、以及两切割自洞恒等式。

**📊 数据集**

无实验数据集；论文完全基于理论分析与证明。

**📈 对比分析**

与之前的分析相比，算法在积分间隙上从 2 降低到 1.788；同时该结果同样适用于最小 LP 值下的森林增补（FAP）问题，提供了相同的 59/33 上限。

**⚠️ 局限性**

局限性：仍未达到最优积分间隙 13/8 ≈ 1.625；分析仅针对标准割松弛的 LP 值；在更一般的森林增补情形（除最小值区间外）尚未给出更强的保证。

---

## 668. GeoComposer: Geometry-Grounded Photographic Composition Instruction

**arXiv ID:** 2609.26620 | [PDF](https://arxiv.org/pdf/2609.26620v1)

**作者:** Shuangzhi Li `[一作]` (Huawei Noah's Ark Lab), Dongfeng Bai `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GeoComposer 框架，先通过模型生成文本指导，再生成视觉示例来提升图像构图。

**💡 创新点**

创新点在于将三维几何先验融入中间表示（全局结构与局部对应），并通过混合奖励的强化学习同时优化指令遵循、美学和几何一致性。

**🔧 技术方法**

技术包括：多模态 Vision‑Language 预训练模型、Diffusion Transformer（DiT）生成器、几何先验基础模型、全局结构与局部对应监督、混合奖励强化学习。

**📊 数据集**

使用了公开的多张图像数据集（CPC、GAIC、FLMS、CUHK‑ICD、Unsplash）构建 34,129 个（不良、良好、文本指导）三元组，并在 DL3DV 作为外部基准验证。

**📈 对比分析**

与 BAGEL、Step1X‑Edit、Qwen‑Image‑Edit、FLUX.2、Nano Banana Pro、PhotoFramer 等基线对比，GeoComposer 在大多数构图、质量与几何一致性指标上取得领先，尤其在 78.1% 的构图胜率和 85.6% 的人类评估通过率。

**⚠️ 局限性**

局限性包括训练阶段需要额外几何基础模型与强化学习算力，生成过程仍受限于预训练模型的表达范围，且在极端视角变换下可能产生细节失真。

---

## 669. Vision Foundation Models with Synthetic-Only Training for Monocular Spacecraft Pose Estimation

**arXiv ID:** 2609.26561 | [PDF](https://arxiv.org/pdf/2609.26561v1)

**作者:** John Church `[一作]`, Vazghen Nikolian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用大规模自监督ViT模型DINOv3并结合LoRA/DoRA微调，在单摄像头图像下实现空间航天器姿态估计，完全使用合成数据训练，达到最低旋转误差。

**💡 创新点**

①将840M规模的DINOv3作为编码器，突破以往小型ViT/Conv网络的限制；②利用低秩适配器仅微调少量参数；③在光箱与日灯两种极端照明下，仅用合成图像即可获得state‑of‑the‑art精度；④在嵌入式Jetson Orin NX上实现实时推理。

**🔧 技术方法**

ViT基础模型DINOv3、LoRA/DoRA低秩适配、热图关键点检测+PnP求解、四角旋转测试时数据增强、三种随机种子集成、TensorRT BF16推理。

**📊 数据集**

SPEED+ 合成训练集（约48k渲染）与验证集（约12k）及真实HIL光箱与日灯测试集（9,531张）。

**📈 对比分析**

采用与SPNv3、EagerNet等公开方法相同的真实图像测试协议；单通道单通过300M模型在光箱/日灯上分别达到1.90°/2.26°，840M单通道单通过1.34°/1.84°，是已公布方法中最低的；多通道集成与TTA进一步降至1.17°/1.56°，极端错误率<0.2%；嵌入式推理单通道133.8 ms，功耗32 W。

**⚠️ 局限性**

仅针对单一目标航天器（Tango）且需要已知3D模型和关键点；仅使用单帧估计，未考虑多帧/运动信息；嵌入式硬件测试仅在Jetson Orin NX上，未验证辐射/温度下的可靠性；适用性受算力限制，需在具备足够GPU资源的飞行器上。

---

## 670. A retrospective analysis on the use of LLMs to study infant syntax learning

**arXiv ID:** 2609.26539 | [PDF](https://arxiv.org/pdf/2609.26539v1)

**作者:** Hélie Bazin `[一作]` (Sorbonne Université), François Yvon `[通讯]` (Sorbonne Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估BabyLM等任务中使用大语言模型研究婴儿句法习得的假设与方法

**💡 创新点**

从数据构建、训练策略、模型架构、基准评估等多维度提供系统性认识，并指出现行做法在理论与实验上的局限

**🔧 技术方法**

Transformer（MLM与CLM）、MiniBERTa、GPT‑2、GPT‑BERT等，结合自监督预训练与多任务微调

**📊 数据集**

Child‑directed speech（CHILDES）、书面文本（Wikipedia、FineWeb‑Edu、Cosmopedia、Baby‑F Corpus）以及 BabyBabelLM 的多语种语料，覆盖从 1M 至 100M 词

**📈 对比分析**

采用 BLiMP、PoSH‑Bench、MSG​S、HierQ、PG‑Accuracy、ATB‑Accuracy 等最小对照与结构推断基准；结果显示在 100M 词以内的模型普遍低于成人基线，MLM 优于 CLM，数据比例（TS/ CDS）对性能影响不显著

**⚠️ 局限性**

未进行新的实验，缺乏统一评估指标，聚焦句法忽略语音、形态、语义和语用；训练轮数、数据排序等设定过于简化，无法真正复现儿童学习环境

---

## 671. Generalizing Manipulation Skills with a Local Coding Agent

**arXiv ID:** 2609.26499 | [PDF](https://arxiv.org/pdf/2609.26499v1)

**作者:** Raman Talwar `[一作]` (Ghent University), Francis wyffels `[通讯]` (Ghent University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究使用本地27B视觉语言模型Qwen3.8-27B在单机上让编码代理通过自写感知与控制代码，在真实UR3e机械臂上完成多种基于儿童玩具的抓取与拼装任务，并实现对颜色、尺寸、形状和任务变体的零样本泛化。

**💡 创新点**

创新点在于将本地开源大型视觉语言模型与编码代理框架结合，实现无需额外训练或编程的即时任务迁移与自我改进，并在真实硬件上验证一次性泛化能力。

**🔧 技术方法**

采用Qwen3.8-27B、vLLM推理、pi编码代理、手工编写的技能文档以及服务层的机器人运动与传统计算机视觉工具。

**📊 数据集**

数据集为儿童玩具套装（杯子、形状分拣器、环形堆叠）共9项任务，每项5次实验，共45次试验。

**📈 对比分析**

通过与现有基于API或前沿模型的基准对比，系统在45次试验中30次无干预完成，平均任务耗时3.4–67.5分钟，推理占比71–85%，证明在本地模型下可实现可观的泛化与自我改进。

**⚠️ 局限性**

主要局限包括依赖单个手腕摄像头导致的感知错误、对错误假设的固定化、缺乏机器人形态感知，导致多种失败模式以及整体执行速度偏慢。

---

## 672. MMAP: Multimodal Missing-Aware Pretraining for Longitudinal Alzheimer's Prediction

**arXiv ID:** 2609.26617 | [PDF](https://arxiv.org/pdf/2609.26617v1)

**作者:** Fiona Kekwick `[一作]` (Imperial College London), Wenjia Bai `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种名为MMAP的多模态缺失感知预训练方法，用于在脑磁共振影像与表格数据不完整时进行阿尔茨海默病进展预测。

**💡 创新点**

创新点在于结合SigLIP对比损失与生成重建损失，利用跨模态缺失令牌生成器实现对缺失影像或表格的自适应填充，且支持预训练的基础模型。

**🔧 技术方法**

使用3D ResNet-18图像编码器、TabPFNv2表格编码器、跨模态生成令牌网络、SigLIP对比损失、均方误差重建损失以及投影头融合网络。

**📊 数据集**

采用阿尔茨海默病神经影像学计划（ADNI）中的1,258名受试者的T1 MRI扫描及相应的认知/临床表格数据，共计5,548张影像。

**📈 对比分析**

在两项临床任务（MCI转化为AD预测与未来阿尔兹海默淀粉样斑块状态预测）上，与HyperFusion、DAFT、TIP、AE+Concat、SimCLR+Concat等多模态与单模态基线相比，MMAP在平衡准确率、AUC和F1分数上均表现出色，尤其在缺失模态情形下仍保持高性能。

**⚠️ 局限性**

局限性包括：仅验证两种模态，未扩展到更多模态；在微调阶段冻结表格编码器导致性能下降；对大规模3D预训练资源依赖较高；以及仅在ADNI数据集上评估，缺乏跨数据集通用性验证。

---

## 673. GTR: Gated Token Recurrence for Efficient Dense Prediction

**arXiv ID:** 2609.26590 | [PDF](https://arxiv.org/pdf/2609.26590v1)

**作者:** Zhe Feng `[一作]` (Didi International Business Group), Xi Shen `[通讯]` (Intellindust Ai Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为GTR的线性注意力视觉骨干，用于高效密集预测。

**💡 创新点**

创新点是使用无softmax的门控令牌递归（GLA）与四向扫描和空间SwiGLU结合，并只用最终输出对齐进行蒸馏。

**🔧 技术方法**

技术包括门控线性注意力、方向性扫描、空间SwiGLU、单向蒸馏、块级CUDA操作和TensorRT融合。

**📊 数据集**

数据集包括Objects365用于预训练、COCO用于目标检测与实例分割、人类姿态等，其余任务使用DOTA、Cityscapes、NYU Depth V2。

**📈 对比分析**

与YOLO26、RF-DETR、EdgeDet等最新方法比较，GTR在检测、分割、姿态、定向检测、语义分割和单目深度上实现了更高AP或mIoU，并在RTX 4090和DRIVE AGX Thor上显著降低延迟和显存占用。

**⚠️ 局限性**

局限在于单向蒸馏仅对最终特征监督，缺乏对中间层的引导，且在极大输入尺寸或极端硬件上仍需进一步优化。

---

## 674. The Ethics of Artificial Intelligence in Military Operations

**arXiv ID:** 2609.26507 | [PDF](https://arxiv.org/pdf/2609.26507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 675. NavSafe-$\infty$: Benchmarking Closed-Loop Driving Safety in Photorealistic Environments

**arXiv ID:** 2609.26618 | [PDF](https://arxiv.org/pdf/2609.26618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 676. Quantum-Aided Active Device Detection in Energy-Harvesting Symbiotic Radio Networks

**arXiv ID:** 2609.26565 | [PDF](https://arxiv.org/pdf/2609.26565v1)

**作者:** Remon Polus `[一作]` (Polytechnique Montréal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montréal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个能量采集型低密度扩频（LDS）代码域非正交多址（CD‑NOMA）共生无线（SR）网络，用于实现无电池物联网设备的回波通信。

**💡 创新点**

创新点在于将 Grover 量子搜索算法引入 ADD（主动设备检测）过程，显著降低搜索复杂度，实现近似 ML 性能的量子辅助检测。

**🔧 技术方法**

使用了能量采集模型、LDS 扩频、多用户干扰模型以及 Grover 量子搜索算法的 Oracle 与 Diffuser 迭代机制。

**📊 数据集**

采用仿真生成的 Rayleigh 衰落 + AWGN 数据进行评估，无需公开数据集。

**📈 对比分析**

与传统最大似然（ML）检索基准比较，Grover 方法在 K=6 时搜索迭代数从 64 降至 12，速度提升约 5.3×，且平均成功检测概率与 ML 接近，随发射功率提升而提升。

**⚠️ 局限性**

局限在于目前仅在经典仿真器上验证，未在真实量子硬件上实现，且模型假设理想的能量采集与回波反射，实际环境中噪声与硬件误差仍需进一步研究。

---

## 677. Seeking Cost-Optimal Infrastructure Size for Distributed Filesystems: A Ceph Case Study

**arXiv ID:** 2609.26616 | [PDF](https://arxiv.org/pdf/2609.26616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 678. Toki: Profiling HBM Performance on FPGA Systems with RISC-V Soft Cores and PCIe Host DMA Traffic

**arXiv ID:** 2609.26551 | [PDF](https://arxiv.org/pdf/2609.26551v1)

**作者:** Andrea Galimberti `[一作]` (Politecnico di Milano), Davide Zoni `[通讯]` (Politecnico di Milano)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了Toki框架，用于在FPGA加速卡上基于RISC‑V软核与PCIe DMA流共同对HBM性能进行系统化剖析。

**💡 创新点**

首个支持RISC‑V软核与主机DMA并行产生HBM访问的硬软框架；支持真实应用、可编程多核、主机流量注入，并提供可重复实验的开源实现。

**🔧 技术方法**

使用FPGA上的RISC‑V软核（Snitch、CVA6）、AXI互连、HBM2内存控制器、PCIe XDMA驱动、Vivado ML实现，配合PolyBench/C benchmark进行实验。

**📊 数据集**

使用PolyBench/C 16个应用程序及其微基准（顺序、随机读写）进行评测。

**📈 对比分析**

通过不同核心配置（K×P）测量执行时间、吞吐量，评估主机流量对HBM带宽和延迟的影响。实验显示单个控制器的随机写慢3.1×；主机DMA 6 GB/s导致应用平均3.57×延迟；优化如共享缓冲、分离指令/数据控制器可将吞吐量提升近3–9倍。

**⚠️ 局限性**

实验仅在AMD Alveo U55C HBM2平台验证，聚焦单个控制器的“最坏情况”；未测试更高PCIe Gen4/5或不同内存层级；对其他软核或FPGA平台的适配需进一步验证。

---

## 679. A Configurable Heuristic for the MLCS Problem

**arXiv ID:** 2609.26602 | [PDF](https://arxiv.org/pdf/2609.26602v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 680. Semantic Abstraction for Natural Language Inference: a Methodological Framework for Discovering and Compensating Semantic Knowledge and Reasoning Gaps in Large Language Models

**arXiv ID:** 2609.26610 | [PDF](https://arxiv.org/pdf/2609.26610v1)

**作者:** David Torres-Moreno `[一作]` (Universidad Autónoma del Estado de Morelos), Jorge Hermosillo-Valadez `[通讯]` (Universidad Autónoma del Estado de Morelos)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种语义抽象框架，通过将Premise和Hypothesis之间的语义关系归纳为语义兼容性与不兼容性四组，并利用这些抽象信息引导LLM进行推理，从而补偿LLM在自然语言推断中的知识空白。

**💡 创新点**

创新点在于：① 将低级语义关系（如同义、反义、共修）聚合成高层抽象类别，形成可解释的推理路径；② 通过对ConceptNet子图进行可转移性推断，自动构建新的概念连接；③ 采用决策树集成多组抽象提示，实现对LLM多路径回答的有序融合。

**🔧 技术方法**

技术方法包括：语义抽象框架设计、ConceptNet知识图谱抽取与子图扩展、基于抽象组的prompting、以及多数投票、加权投票和决策树三种融合策略。

**📊 数据集**

实验使用的NLI数据集包括：SICK、SNLI、SciTail、RTE和SuperGLUE诊断集；所有数据均为公开标注数据，且在训练时按类别平衡采样。

**📈 对比分析**

与原始LLM基线和仅使用直接关系的对比实验表明，采用抽象组+决策树（GS_DT）在所有三分类和二分类任务中均取得显著提升，平均提升5.7%–11.6个百分点，且在诊断集对非蕴含类的准确率提升尤为明显；统计检验显示提升均显著（p < 0.05）。

**⚠️ 局限性**

局限性包括：① 对外部知识图谱的依赖导致可能出现文化偏见或覆盖不足；② 抽象组划分需手工定义，缺乏自动化；③ 在已高度优化的大模型上提升有限，可能受训练数据泄漏影响。

---

## 681. Receptiveness, Not Sycophancy: Distinguishing Engagement from Deference in Language Models

**arXiv ID:** 2609.26579 | [PDF](https://arxiv.org/pdf/2609.26579v1)

**作者:** Calvin Isley `[一作]` (Harvard University), Sharad Goel `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究社交sycophancy与对话接纳性（receptiveness）之间的重叠，并通过实验和技术手段展示如何在保持独立判断的前提下提升模型的接纳性。

**💡 创新点**

揭示社交sycophancy评估与接纳性高度相关，导致构念效度问题；提出一种在生成后修改回应的“receptiveness-tool”方法，可在不增加主体性妥协的情况下显著提升接纳性。

**🔧 技术方法**

使用大型语言模型（GPT‑5.6 Terra、Claude Sonnet 5、Gemini 3.7 Flash、Llama 4 Scout）进行生成与重写；采用GPT‑5.6 Luna对回应进行接纳性评分；利用H.E.A.R.框架和对话后修正策略实现接纳性提升。

**📊 数据集**

主要使用Reddit的AITA‑YTA道德建议数据集（并在ELEPHANT数据集上验证）作为实验和评估数据来源。

**📈 对比分析**

通过相关系数（r≈0.64）和预注册实验对比原始与更接纳的回应，显示后者在质量评分上提升约1.5分，偏好得分≈1.0；使用receptiveness‑tool后，接纳性提升2.1个标准差，主体性妥协率保持在Syc(π)≈0.5，表明方法在保持独立性同时提升接纳性。

**⚠️ 局限性**

仅针对道德建议场景，接纳性评分和重写方法存在自动化不完善、语义流畅度变化等局限；实验仅测量受试者的主观偏好和预期，未验证实际行为；对其它用户信号和跨域泛化的评估尚未完成。

---

## 682. Radiomics--Foundation Fusion for Interpretable RCC Classification: Internal Benchmarking and Exploratory External Transfer

**arXiv ID:** 2609.26578 | [PDF](https://arxiv.org/pdf/2609.26578v1)

**作者:** Yuan Liang `[一作]` (Research Ireland Centre for Research Training in Machine Learning), Abraham Campbell `[通讯]` (University College Dublin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究比较手工放射组学与基础模型（如MedVAE、MedicalNet等）在CT影像下的肾细胞癌亚型分类中的互补性，提出门控融合方案并通过内部测试和外部迁移评估其性能。

**💡 创新点**

提出在基础模型时代仍需保留手工放射组学特征，并通过门控融合与解释性分析证明放射组学在决策中占主导地位，从而实现更具临床可解释性的模型。

**🔧 技术方法**

使用PyRadiomics提取形状、纹理等特征；采用3D ResNet-18、MedicalNet预训练网络和MedVAE生成的图像表示；融合方法包括拼接、交叉注意力和门控融合；解释方法包括特征置换重要性和门控值分析。

**📊 数据集**

内部数据集为公开的KiTS23（多机构CT和肾/肿瘤分割）；外部数据集为TCGA-KIRC/KIRP/KICH，采用AIMI自动分割并经人工校正。

**📈 对比分析**

内部最佳模型为3D MedVAE门控融合，AUC 82.7%（95% CI 70.7–92.2），AP 92.2%；单一放射组学SVC AUC 74.4%；外部模型AUC 79.5%（95% CI 66.6–92.3），AP 98.9%；分支移除实验显示单一分支性能低至60%以下，证明两分支互补。

**⚠️ 局限性**

外部样本为正类（ccRCC）占95.1%，仅有两例非ccRCC，导致特异性和整体外部验证受限；缺乏大规模平衡外部数据，未来需进一步验证。

---

## 683. E3Sense: Head-Confined Multimodal Sensing of Learner Engagement

**arXiv ID:** 2609.26569 | [PDF](https://arxiv.org/pdf/2609.26569v1)

**作者:** Sidharth Anupkrishnan `[一作]` (University of Massachusetts Amherst), Phuc Nguyen `[通讯]` (University of Massachusetts Amherst)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文设计并评估了一种将脑电（EEG）、眼动追踪与额头皮肤电（EDA）设备共置于头部的多模态感知平台，以预测学习者对教育视频段落的自评投入水平。

**💡 创新点**

创新点在于首次将三种生理/行为传感器完整地置于头部，证明其能在参与者独立的情况下实现可行的投入预测，并探讨学习者对投入的个人定义作为上下文改进预测精度。

**🔧 技术方法**

使用了脑电头带、Pupil Labs 眼动追踪眼镜和EmotiBit 额头皮肤电传感器；对原始信号进行预处理、特征提取并早期融合成166维向量；采用线性、随机森林、XGBoost、LightGBM、AdaBoost 等经典机器学习模型进行五级序数和二分类预测；并使用二级序数校准模型结合学习者定义的认知/行为/情感指标进行实验。

**📊 数据集**

收集了30名受试者观看三段公开教育视频（共450段）后自评的5级投入评分，以及对应的EEG、眼动与EDA信号；其中15人使用手腕EDA，15人使用额头EDA；另有18人完成退出访谈并给出投入定义。

**📈 对比分析**

通过五折参与者分组交叉验证，训练时使用两种EDA站点的数据，测试时仅评估额头站点受试者；对比多模型与基线（模式预测、随机分布、均值预测）发现AdaBoost在五级序数任务上实现75.0%±7.0的平衡一阶准确率，优于基线12个百分点；在二分类任务上LightGBM达到58.9%±7.9的宏观F1。加入学习者定义后，二级校准模型在平衡一阶准确率上提升至71.5%±?，较基线提高6.9个百分点。

**⚠️ 局限性**

局限包括样本量仅30人、仅评估单一视频内容、手腕/额头EDA与实验批次共线、缺失部分EDA数据需插补、定义指标样本仅18人、使用回溯评分且窗口仅覆盖视频前120秒、未验证日常使用舒适度和实时预测效果。

---

## 684. REFLEX with Jev for Efficient Selective Control in LLM Agents

**arXiv ID:** 2609.26532 | [PDF](https://arxiv.org/pdf/2609.26532v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 685. Rouxii: Exploiting Honeypots with Deception-Aware AI Pentesters

**arXiv ID:** 2609.26555 | [PDF](https://arxiv.org/pdf/2609.26555v1)

**作者:** Arthur Cordeiro `[一作]` (Technical University of Denmark), Emmanouil Vasilomanolakis `[通讯]` (Technical University of Denmark)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了名为 Rouxii 的 AI 驱动渗透测试框架，探讨在攻击者具备对蜜罐伪装识别能力时，欺骗技术的有效性与后续攻击路径。

**💡 创新点**

创新点在于：① 通过对抗性提示（anti‑deception）将蜜罐指纹信息注入 LLM 的决策上下文，显著提升从 19% 到 97% 的检测准确率；② 在框架内部实现从蜜罐识别到主动利用（如 DDoS、信息篡改）的完整攻击链；③ 引入 Honeyquest 问卷与白盒安全分析，系统化评估 LLM 的欺骗感知与攻击行为。

**🔧 技术方法**

技术方法包括：LLM 作为随机决策器与一组确定性操作器（探测、利用、记录）分离；Prompt 工程构建 vanilla 与 anti‑deception 两个 cohort；Honeyquest 作为标准化问卷评估；白盒安全分析对 Cowrie、Conpot、GasPot 三种蜜罐进行漏洞挖掘并演示利用。

**📊 数据集**

使用的数据集：11 种网络拓扑（SSH/Cowrie、Conpot、GasPot 组合），12 个实验周期，共 1,544 条攻击报告；公开蜜罐实例 Cowrie、Conpot、GasPot；三款 LLM（Gemma‑4‑31b、DeepSeek‑R1‑32b、Qwen‑3.6‑27b）。

**📈 对比分析**

比较方法：在同一模型、同一操作器、同一流程下仅对比 prompt 的差异（vanilla vs. anti‑deception），并与外部框架 PentestGPT、HackingBuddy 进行对照；结果显示 anti‑deception cohort 的正确识别率提升至 97%（相比 vanilla 19%），误报率仅 0.7%；利用成功率约 67%。

**⚠️ 局限性**

局限性：① 仅测试公开 LLM 与框架，未覆盖商业化或专门 fine‑tune 的模型；② 蜜罐部署仅为公开实现，未考虑高度定制或混合环境；③ 评估环境为容器化网络，真实部署中的网络层差异可能影响结果；④ 对齐与授权框架下的攻击行为揭示 LLM 在授权语境下的易攻击性，提示未来对齐研究需进一步加强。

---

## 686. JEV-as-a-Judge: Accept When Confident, Escalate When Unsure

**arXiv ID:** 2609.26550 | [PDF](https://arxiv.org/pdf/2609.26550v1)

**作者:** Yubo Li `[一作]` (Carnegie Mellon University), Rema Padman `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种仅输出判定结果及置信度的LLM评判器JEV，并评估其在多种任务（偏好、事实性、答案裁决、风格对抗等）中的准确性、成本和可靠性；

**💡 创新点**

创新点在于提出决策‑仅接口与置信度触发的级联机制，证明在大多数评估场景中可以极低成本获得与最强判定器相近的表现，并通过置信度实现自动升级；

**🔧 技术方法**

技术上使用多种大型语言模型（GPT‑4/5系列、Claude、Gemini、Qwen等）作为判定器，JEV提供概率输出并将最大概率作为置信度；对比生成式与奖励模型，并采用温度标定、二阶概率平均等手段实现级联；

**📊 数据集**

数据集包括RewardBench、JudgeBench、HaluEval、RM‑Bench、RewardBench 2、现有标签集、GSM8K控制等公共基准与内部实验数据；

**📈 对比分析**

比较方法采用盲人类裁决交叉验证、准确率、宏F1、多类别Brier、ECE、AUROC等指标；结果显示JEV在普通偏好与证据支持的事实性任务中与最强GPT‑6相差≤3个百分点，成本比GPT‑6低≈280倍；在推理、编码等难度任务差距较大，但通过置信度级联可恢复≈99%准确率，仅付约57%GPT‑6费用；

**⚠️ 局限性**

局限性包括仅评估单一专有JEV版本，缺乏完整的训练重叠和跨模型对比；级联阈值需本地验证，参考自由自然语言评价效果差；置信度与解释缺失，局限于任务特定；实验受限于有限数据集与人工裁定者。

---

## 687. Pangenome Optimization via Elastic Degenerate Strings

**arXiv ID:** 2609.26542 | [PDF](https://arxiv.org/pdf/2609.26542v1)

**作者:** Nicola Rizzo `[一作]` (University of Helsinki), Veli Mäkinen `[通讯]` (University of Helsinki)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了如何通过对多序列比对（MSA）进行分段来构造弹性退化字符串（EDS）并最小化其基数（cardinality）和大小（size），提出了在无gap MSA下实现线性时间解法，并给出了针对含gap MSA的启发式方法与实验验证。

**💡 创新点**

① 在无gap MSA上首次提供了两种分段约束（上限U和下限L）下最小基数和最小大小的线性时间算法；② 引入了“意义左扩展”（meaningful left extensions）与pBWT结合的高效计算框架；③ 在实际含gap MSA中提出gap‑as‑symbols策略并证明其在多数情况下几乎不损失最优性。

**🔧 技术方法**

采用分段框架、可变长度分段、positional Burrows–Wheeler Transform (pBWT)、动态规划、区间最小查询（RMQ）以及范围最小队列等技术实现线性时间算法；对gap MSA使用字符映射 + pBWT 或关键字树求意义左扩展。

**📊 数据集**

实验使用四个不同规模的SARS‑CoV‑2 MSA（10²、10³、10⁴、10⁵ 条序列）、模拟的E. coli MSA（16条序列，6,809,339 列）以及人类染色体19 MSA（1000 条序列，59,451,290 列）。

**📈 对比分析**

与三种基线（无分段、每列单独分段、贪婪基于高度的分段）比较。结果显示：在小 U（4–8）下，提出方法可显著压缩基数和大小；内存占用≤2.8 GB（SARS‑CoV‑2 10⁵），运行时间从数秒到数十秒；对含gap MSA的 gap‑as‑symbols 策略，基数与大小的增幅≤0.25%。

**⚠️ 局限性**

① 线性算法仅适用于无gap MSA；② 对含gap MSA只能得到上界，无法保证最优；③ 在极大规模（如人类全基因组）时仍需进一步优化内存与并行度。

---

## 688. Foundation model embeddings capture pre-diagnostic changes on screening mammograms

**arXiv ID:** 2609.26605 | [PDF](https://arxiv.org/pdf/2609.26605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 689. Semantically-Guided Domain Randomization for Industrial Object Detection in Low-Image-Budget Regimes

**arXiv ID:** 2609.26505 | [PDF](https://arxiv.org/pdf/2609.26505v1)

**作者:** Jose Moises Araya-Martinez `[一作]` (Technische Universitaet Berlin), Jens Lambrecht `[通讯]` (Technische Universitaet Braunschweig)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为S-GDR的无注释适应管道，结合了基于视觉语言模型的语义描述和扩散模型生成背景图像，用于在低图像预算下进行工业物体检测。

**💡 创新点**

创新点在于通过语义引导的背景合成和无注释的训练数据生成，提升了在极端数据稀缺情况下的多目标检测性能。

**🔧 技术方法**

使用了视觉语言模型（VLM）进行语义描述，结合了SDXL、ControlNet和IP-Adapter进行扩散模型生成。

**📊 数据集**

使用了一个公共的汽车多目标检测基准数据集，包括合成训练集和真实测试集。

**📈 对比分析**

与传统的域随机化基线（0.697）和其他特征选择方法相比，S-GDR在200张合成训练图像的预算下达到了0.739的mAP，表现优于其他方法。

**⚠️ 局限性**

限制在于仅在单一基准上进行评估，使用了单一检测器（YOLOv8），且结果基于单次训练，未进行组件消融实验。

---

## 690. Beyond End-Task Success: How to Audit Visual Experience Retrieval in Robotics

**arXiv ID:** 2609.26567 | [PDF](https://arxiv.org/pdf/2609.26567v1)

**作者:** Eshika Pathak `[一作]` (University of Illinois Urbana Champaign), Leela Krishna `[通讯]` (Centific)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在机器人检索经验的过程中，作者通过对每个查询场景执行所有存储经验来构建完整的转移矩阵，从而评估检索规则的实际表现。

**💡 创新点**

创新点在于提出了一套可执行的审计协议：利用完整转移矩阵而非仅报告所选经验的成功率，揭示视觉相似度在候选经验排序方面的局限，并给出了可量化的诊断指标。

**🔧 技术方法**

使用了多种视觉嵌入（原始像素、ResNet-18、DINOv2、CLIP、策略编码器），三种重用机制（Replay、Policy、Policy+DR），以及优势、区分度、AUROC 等统计指标进行评估。

**📊 数据集**

实验基于 robosuite 中的 Stack 与 Door 任务，使用 Panda 机械臂生成 40 个物理状态，并在 5 种外观变换下进行，产生约 114,500 条有种子执行记录。

**📈 对比分析**

与随机选择、最佳固定经验和 Oracle 进行对比。结果显示最佳固定经验已占据 30–58% 的优势；视觉规则在库大小 ≥10 时过度集中于单一经验；视觉距离在预测成功率上表现良好，但在同一场景内的候选排序上接近随机，整体性能与传统评估显著不同。

**⚠️ 局限性**

局限性包括：仅在仿真环境下实验；库大小受限（K=50 仅在 Replay 任务中使用）；学习策略仅在 K=3 的设置下测试；缺乏独立记录流导致大小与录制质量混杂；统计区间宽度大，可能掩盖细微差异。

---

## 691. A Semiotics-Aware Framework for Evaluating Fidelity and Coverage in Natural Language Generation

**arXiv ID:** 2609.26527 | [PDF](https://arxiv.org/pdf/2609.26527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 692. Latent Commonality Expectation-Maximisation for Box-supervised Tree Crown Instance Segmentation

**arXiv ID:** 2609.26549 | [PDF](https://arxiv.org/pdf/2609.26549v1)

**作者:** Thomas Pitts `[一作]` (University of Technology Sydney), Bin Liang `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于盒子监督的树冠实例分割框架 LACE，利用冻结的 DINOv3 ViT 特征与 EM 模块在盒子内提取树冠掩膜。

**💡 创新点**

创新点：① 通过 EM 算法在潜在空间中分离“树性”与背景，实现无掩膜标签的实例分割；② 用四重偏移交错得到更细的 8px 特征网格；③ 通过掩膜后验概率与检测分数相乘重新排序，提升检测置信度。

**🔧 技术方法**

使用的技术包括：冻结的 DINOv3‑web ViT‑L/16 编码器、Anchor‑free CenterNet 检测头、PCA 归一化 + von Mises‑Fisher 混合模型、EM 训练、掩膜后验置信度融合。

**📊 数据集**

实验数据集：OAM‑TCD（包含稠密与稀疏冠层的 2048×2048px 0.1 m/px RGB 图像）和 NeonTreeEvaluation（NEON 22 站点的 400 px 0.1 m/px RGB 街景），仅使用框注解进行训练。

**📈 对比分析**

与全监督基线（Mask R‑CNN、Detectree2、DeepForest）对比，LACE 在 OAM‑TCD 上 AP_50 0.663±0.001（仅 900 张盒子图像），在稀疏冠层上 AP_50 0.691，Neon 上 F1@0.4 0.728±0.003，均等同或优于现有最优模型，仅需 0.1% 的注释量。

**⚠️ 局限性**

局限性：① 掩膜精度受 8px 网格分辨率限制，在 AP_75/AP_90 上性能略逊；② 对极度重叠或密集冠层的分割效果仍受盒子假设限制；③ 依赖大规模预训练的 DINOv3，迁移到不同传感器或分辨率时可能需要再训练。

---

## 693. Notes on Fourier-Bessel wavelets

**arXiv ID:** 2609.26537 | [PDF](https://arxiv.org/pdf/2609.26537v1)

**作者:** Marcel Venturotti `[一作]` (University of Bath), Georgios Exarchakis `[通讯]` (University of Bath)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一类基于傅里叶-贝塞尔波形的波let族，并给出了其闭式傅里叶域表达式。

**💡 创新点**

采用贝塞尔函数在单位圆上满足Neumann边界条件的解作为基底，并通过 Gaussian 包络和零均值校正设计出线性间隔的径向频率分布。

**🔧 技术方法**

贝塞尔与改进贝塞尔函数理论、Helmholtz 方程求解、L^2 与 L^1 归一化、闭式 Fourier 变换等技术。

**📊 数据集**

本文未使用任何数据集，全部为理论推导。

**📈 对比分析**

未进行实验比较，本文未给出任何性能评估。

**⚠️ 局限性**

仅提供理论框架，缺乏实验验证与实际应用案例，对边界条件的适用性和多尺度性质的完整评估仍待进一步研究。

---

## 694. From Approval to Execution: Reconstruction-Aware Repair Analysis for LLM-Agent Software

**arXiv ID:** 2609.26529 | [PDF](https://arxiv.org/pdf/2609.26529v1)

**作者:** Junchi Zhu `[一作]`, Qinming He `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 APAS-Finder，针对 LLM‑Agent 软件的审批‑执行偏差进行重建感知的修复分析。

**💡 创新点**

提出了重建稳定授权（ReSA）模型，生成基于对象版本、授权范围等的义务，能够在审批后重建过程中检测潜在授权绕过。

**🔧 技术方法**

利用图构造、数据流、对象敏感点分析与自定义义务求解技术，实现了基于结构的预测与残差路径报告。

**📊 数据集**

使用合成的审批‑执行案例（28 个配置）和多款公开 LLM‑Agent 代码库（如 Lobster、PromptSpeak 等）进行评估。

**📈 对比分析**

与简化规则、缺失输入以及 CodeQL 组合基线相比，APAS‑Finder 在所有 28 个受控案例上预测完全准确，并在发布的消费者中成功阻止 60/60 的违规执行。

**⚠️ 局限性**

需要人工提供完整的图事实（如攻击者控制、可信保存、原子验证等），对多授权计划或缺少显式 API 的情况支持有限。

---

## 695. Do Vision Model See Like the Brain? A Comparison Across EEG Encoding Model

**arXiv ID:** 2609.26512 | [PDF](https://arxiv.org/pdf/2609.26512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 696. Multiform Longest Edge Bisection of Tetrahedra via Sextuple Permutations

**arXiv ID:** 2609.26522 | [PDF](https://arxiv.org/pdf/2609.26522v1)

**作者:** Agustin Trujillo `[一作]` (University of Las Palmas de Gran Canaria), Tania Moreno-García `[通讯]` (Universidad de Holguin)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于六元组（六条边平方长）的全新三维最长边细分（LEB）表述，将细分方程线性化，消除了坐标依赖。

**💡 创新点**

引入多形最长边细分（MLEB）与细分模式（bisection pattern）的概念，证明共享同一细分模式的六元组构成凸集，并发现R₁⁺族与Liu–Joe族的细分树可归约为相同的八状态有向图。

**🔧 技术方法**

使用线性代数（Cayley–Menger 行列式）、组合几何、凸分析和有向图理论来建模和证明结果；通过符号计算验证细分公式。

**📊 数据集**

未使用具体实验数据集，全部为理论推导和符号验证；所示实例（如(15,13,12,11,10,14)、(80,41,40,32,31,30)）仅作示例演示。

**📈 对比分析**

对比方法：将细分树映射到有限有向图并验证两族共享同一图结构；性能方面的评估主要是理论上“无穷层细分压缩为有限状态”的简化，未给出数值时间/空间指标。

**⚠️ 局限性**

局限性：仅证明对R₁⁺族和Liu–Joe族成立；对一般四面体或更复杂细分策略尚未给出全局描述；需要进一步确定不同族的凸区域与六元组空间的具体边界。

---

## 697. Wheel-loader V-Cycle Automation with Deep Koopman MPC

**arXiv ID:** 2609.26580 | [PDF](https://arxiv.org/pdf/2609.26580v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 698. Virtual Encoders in Multimodal Transformers

**arXiv ID:** 2609.26513 | [PDF](https://arxiv.org/pdf/2609.26513v1)

**作者:** Katsuya Ogata `[一作]` (University of Osaka), Yuta Nakashima `[通讯]` (University of Osaka)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在缺少专用感知编码器的多模态Transformer中，内部会自行形成的“虚拟编码器”，并通过线性探测、相似性分析和因果扰动等方法探讨其层级、可解码性、几何结构与读出位置。

**💡 创新点**

提出“虚拟编码器”概念，证明共享Transformer能够在内部完成感知编码，揭示不同架构（encoder-full、encoder-free、discrete-token）在读出深度和子空间分配上的差异。

**🔧 技术方法**

采用线性探测、CKA相似性、傅里叶低通扰动、单层激活扰动、主成分角度重叠分析等技术，对Transformer各层的感知表示进行深入评估。

**📊 数据集**

使用ImageNet-1k与ESC-50构建的31个概念对应的图像-音频对数据集进行实验，并在Gemma 4 12B等模型上进一步验证子空间分布。

**📈 对比分析**

通过与encoder-full、encoder-free、discrete-token三类MLLM对比，发现encoder-free模型在层0到峰值层显著提升线性可解码率，CKA与DINOv2/AST表现出较高相似度，因果扰动显示读出深度约在23层，整体表明虚拟编码器功能有效。

**⚠️ 局限性**

局限性包括仅在Gemma 4 12B进行子空间分析，缺乏对其他架构的全面评估；未对子空间旋转进行直接干预；音频路由的因果机制尚不清晰；实验范围受所选模型与数据集限制。

---

## 699. Radiomics-Conditioned Modulation of RenalCLIP Features for Clear Cell Renal Cell Carcinoma Classification

**arXiv ID:** 2609.26492 | [PDF](https://arxiv.org/pdf/2609.26492v1)

**作者:** Yuan Liang `[一作]` (Research Ireland Centre for Research Training in Machine Learning), Abraham Campbell `[通讯]` (University College Dublin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了Radiomics与RenalCLIP在CT图像中的融合，提出Radiomics-Conditioned Residual FiLM框架，用于提高ccRCC分类性能。

**💡 创新点**

创新点在于同时利用FiLM对图像特征进行条件化调制，并保留Radiomics残差的直接贡献，实现两种表征的有效融合。

**🔧 技术方法**

采用Radiomics特征提取（PyRadiomics）、RenalCLIP 3D ResNet18编码器、FiLM调制、拼接、门控融合等技术，并使用交叉熵与AdamW进行训练。

**📊 数据集**

使用内部KiTS23（396例）和外部RCC-AID+TCGA（111例）的CT影像、肿瘤分割和病理标签数据集。

**📈 对比分析**

与传统SVC、3D ResNet18、RenalCLIP单独模型、拼接、门控、全局affine等方法对比；FiLM在内部实验中达到AUC 0.804、外部实验中达到AUC 0.854，均为所有RenalCLIP融合策略中的最佳。

**⚠️ 局限性**

局限性包括回顾性小样本、不同模型训练方式不完全一致、未充分验证不同CT采集或分割对FiLM效果的影响。

---

## 700. Towards Hierarchical GNNs for multi-grid power flow: generalization across operating scenarios

**arXiv ID:** 2609.26603 | [PDF](https://arxiv.org/pdf/2609.26603v1)

**作者:** Carmine Delle Femine `[一作]` (Vicomtech Foundation), Marco Quartulli. Izaro Goienetxea Urziku `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对电网功率流模型加入了两级分层隐藏信息交换模块，提升了模型在新工况下的泛化能力。

**💡 创新点**

提出在GENCO求解器中使用Kron和Quotient两种电学与聚类方式的分层传输，实现统一参数模型在多网格下的共享。

**🔧 技术方法**

基于图神经网络的修正层、Kron约简、Quotient聚类、全局注意力以及可学习的限制/延拓映射。

**📊 数据集**

使用PGLib的三台电网（500、2000、4917机架）进行训练，额外评估两个未见电网（3022、4020机架）。

**📈 对比分析**

在训练网的200个新生成工况上比较，Kron模型宏观误差从5.66降至0.851（比平面GENCO降低85%），并且优于每路平均参考；但在未见网的拓扑迁移上仍未优于平面模型。

**⚠️ 局限性**

仅在训练网的工况泛化表现良好，跨拓扑泛化尚未实现；模型仍需进一步训练、超参数调优和更广泛的拓扑覆盖。

---

## 701. Learning Air-Ground Motion Control with Temporal Mode Switching and Cross-Terrain Tracking

**arXiv ID:** 2609.26564 | [PDF](https://arxiv.org/pdf/2609.26564v1)

**作者:** Ruitian Pang `[一作]` (Zhejiang University), Yanjun Cao `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于学习的主动-被动式TABV空地运动控制框架，包含时序模式选择器与地面/空中强化学习控制策略，实现自适应模式切换与跨地形轨迹跟踪。

**💡 创新点**

创新点在于：①仅用单点ToF+运动历史+未来轨迹信息的双分支时序网络实现无手工规则的模式切换；②多地形随机化训练的RL轨迹跟踪策略，零样本跨地形转移；③将空地策略通过共享CTBR接口无缝集成。

**🔧 技术方法**

采用强化学习（PPO）训练空中与地面策略；双分支GRU/MLP时序网络做模式分类；ToF传感器+IMU+里程计作为感知；使用多环境并行仿真和领域随机化；离线监督学习训练模式选择器。

**📊 数据集**

训练数据来自Isaac Lab的随机生成地形（平坦、连续粗糙、离散粗糙、过渡区）与参考轨迹；模式标签按预设规则生成；真实数据为自定义TABV平台在四种地形及101m长轨迹上的采样。

**📈 对比分析**

与传统PID和NMPC比较：在平坦、柔软、草地、粗块地形上，所提方法在轨迹跟踪RMSE上分别比PID低92.2%~75.8%，比NMPC低33.9%~17.1%；在空地混合轨迹上，RMSE 0.08m；模式切换成功率100%，延迟约20-260ms。

**⚠️ 局限性**

局限性：仍依赖ToF单点测距，受噪声影响；对极端快速降落或大尺寸障碍的鲁棒性未充分验证；在高度动态或非平面地形的自适应性能仍待进一步提升。

---

## 702. The Disciplinary Language Transfer Problem: How Psychological Vocabulary Produces Governance Failures in AI Agent Deployment

**arXiv ID:** 2609.26562 | [PDF](https://arxiv.org/pdf/2609.26562v1)

**作者:** Kymberly Lasser-Chere `[一作]` (Coastal Waters Healing Center), Marc Millstone `[通讯]` (Redpanda)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析心理学术语被错误引入AI治理导致的认知偏差，并提出治理审计与词汇替代方案。

**💡 创新点**

首次将“disciplinary language transfer problem”概念化，结合语言游戏、范式依赖、边界对象等理论，构建六项治理审计与词汇翻译表。

**🔧 技术方法**

采用哲学与社会技术学的跨学科理论框架（维特根斯坦、库恩、哈拉维、Star & Griesemer 等）进行概念分析。

**📊 数据集**

无实验数据集，主要基于案例分析（如Microsoft Bing Chat、欧盟AI法案等）。

**📈 对比分析**

未进行量化对比，主要通过案例阐释与逻辑推演展示词汇转移对治理设计的影响。

**⚠️ 局限性**

局限在缺乏经验验证，词汇替代在法律与实践中实施难度大，需与现行法规对齐。

---

## 703. Neutral-Atom-based Quantum Optimization for Resource Allocation in NOMA Networks

**arXiv ID:** 2609.26556 | [PDF](https://arxiv.org/pdf/2609.26556v1)

**作者:** Patatchona Keyela `[一作]` (Polytechnique Montréal), Ola Ahmad `[通讯]` (Thales cortAIx Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于中性原子量子计算的方案，用于解决NOMA网络中的最大访问问题，通过将问题映射为最大独立集并在Pasqal的Rydberg原子阵列上求解，从而实现资源分配优化。

**💡 创新点**

首次将最大访问问题转化为最大独立集并利用中性原子量子平台的Rydberg阻塞效应自然编码实现求解，展示了量子计算在无线资源调度中的可行性。

**🔧 技术方法**

采用Pasqal的Pulser模拟器进行量子退相干过程的量子退相干演化，以Rydberg态的相互作用构建问题哈密顿量，并通过量子测量获得独立集解。

**📊 数据集**

使用仿真生成的20个用户、10条上行信道的随机分布数据，基于Rayleigh衰落、路径损耗等参数构建冲突图。

**📈 对比分析**

与随机分配和Gurobi最优求解器对比，量子模拟器得到的支持用户数与Gurobi相同，显著高于随机方案，验证了方法的有效性。

**⚠️ 局限性**

受限于模拟器规模和硬件实现，目前只能在小规模网络上验证，未探讨大规模部署的可扩展性与量子噪声对结果的影响。

---

## 704. Gap-Free Streaming PCA Beyond Rank-One Updates: Near-Optimal Rates and Applications to Differential Privacy

**arXiv ID:** 2609.26508 | [PDF](https://arxiv.org/pdf/2609.26508v1)

**作者:** Anming Gu `[一作]` (University of Texas at Austin), Chutong Yang `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对流式PCA问题，给出了Oja算法在无eigengap条件下的理论分析，并将该分析推广至能量PCA（ePCA）以及差分隐私PCA（DP-PCA）等变种。

**💡 创新点**

创新点包括：①只利用矩阵方差（第二矩）界定更新，绕过传统的几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几乎几个

**🔧 技术方法**

主要技术手段包括：期望迹（trace）势函数、矩阵方差上界、von Neumann迹不等式、几何聚合（geometric aggregation）用于提升成功概率、Rényi差分隐私与自适应裁剪的结合、以及通过耦合（coupling）将私有更新映射到标准Oja迭代。

**📊 数据集**

实验采用合成的子高斯/正态数据流，特征值前几项接近平分，后续按(0,0.95)分布采样，维度设置为d=50（流式PCA）或d=5000（DP-PCA），样本量从10^7到10^8不等。

**📈 对比分析**

与经验协方差基准相比，调参后的Oja在流式场景下能实现与基准相近的cPCA误差；在DP-PCA实验中，相比先前的基线算法，本文方法在大维数且谱间隙较小时能取得更低的cPCA误差并提升成功率。

**⚠️ 局限性**

局限性包括：需要子高斯假设、对λ₁的事先了解、仅给出常数成功概率，需额外的概率放大步骤；隐私分析仍受log因子影响；对重尾或非正态数据的适用性尚未验证。

---

## 705. DIFTA-3D: Depth-Consistent Instance-Level Feature Transfer and Adaptation of DINOv3 for 3D Detection

**arXiv ID:** 2609.26702 | [PDF](https://arxiv.org/pdf/2609.26702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 706. Type-Safe Is Not Error-Free: A Constrained Decision Head Follows the Option Name, Not the Rubric Bound to It

**arXiv ID:** 2609.26758 | [PDF](https://arxiv.org/pdf/2609.26758v1)

**作者:** Yu Sun `[一作]`, Junhao Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了Typed决策模型在选项名称对模型输出的影响，揭示选项名称主导而非选项描述。

**💡 创新点**

发现即使保证类型安全，模型仍会因选项名称的语义极性导致决策反转，提出了名称不变性指标。

**🔧 技术方法**

通过对标记式和跨度平均两种读出几何的encoder head进行实验，并比较了本地检查点与云端主机模型。

**📊 数据集**

使用了四个业务谓词的数据集（发票核对、代理追踪、安保警报、客服升级）以及多项选择题集。

**📈 对比分析**

对比了中性名称、极性名称以及随机名称的置换效果，发现极性名称导致 flip 率高达 90% 以上，AUROC 下降至 0.4 左右。

**⚠️ 局限性**

局限在于仅评估两种英文encoder几何，未拆分极性与词表熟悉度，且对多项选择仅做了粗略验证。

---

## 707. Underwater Navigation in Unsteady Flows Using Measurement Histories from a Single Sensing Unit

**arXiv ID:** 2609.26753 | [PDF](https://arxiv.org/pdf/2609.26753v1)

**作者:** Linhao Jin `[一作]` (Iowa State University), Qiang Zhong `[通讯]` (Iowa State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用单点流量历史估计侧向流速，替代多点传感，评估其对水下导航的闭环性能；

**💡 创新点**

通过因果观测器从单点历史重建两侧点流量，构造虚拟空间感知，并系统化研究历史信息、误差结构对导航的增益；

**🔧 技术方法**

使用 MLP 观测器结合流场历史、相对偏航与目标信息；采用二维 CFD 仿真（FluidX3D）产生圆柱、三角棱柱、方棱柱阻尼流场；与直接空间感知和仅当前流等方案比较；固定空间导航控制器；

**📊 数据集**

基于 FluidX3D 生成的二维平面流场数据，在 Re=100/175/225/205/240 下的圆柱、等边三角棱柱、方棱柱障碍物，收集多场景导航任务；

**📈 对比分析**

通过成功率对比评估，单点历史在圆柱阻挡下成功率约84%/80%（Re=205/240），接近直接空间感知（92%/85%），比仅当前流高30pp；在三角棱柱转移中差距显著，单点历史仅 47% vs 65%；误差结构实验显示差分误差对性能影响大，持久差异误差导致更大损失；

**⚠️ 局限性**

仅在模拟二维流场且使用固定控制器；未考虑机体运动估计、真实传感误差和身体扰流；对不同几何形状的转移性能有限；仅评估已训练模型的单点历史，未进行重新训练。

---

## 708. Proof-of-Retention: A Framework for Auditable Cross-Organization Data Sharing

**arXiv ID:** 2609.26654 | [PDF](https://arxiv.org/pdf/2609.26654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 709. Lower Bounds for all List-Decodable Deletion Codes

**arXiv ID:** 2609.26650 | [PDF](https://arxiv.org/pdf/2609.26650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 710. Diffusion-Induced Spatial Attention Overlapping Community Detection

**arXiv ID:** 2609.26737 | [PDF](https://arxiv.org/pdf/2609.26737v1)

**作者:** Kosti Koistinen `[一作]` (Aalto University), Kimmo K. Kaski `[通讯]` (Aalto University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DISCO 模型，通过影响传播导向的结构先验与稀疏多头注意力相结合，实现对图中重叠社区的高效检测，并在社交网络与工业控制系统网络的安全异常案例中进行验证。

**💡 创新点**

创新点在于利用基于影响传播（ISM）的结构先验作为稀疏注意力的偏置，既扩展了 GNN 的感受野，又避免了全局注意力的 O(N²) 复杂度，并将非负社区归属学习与 Bernoulli–Poisson 重建目标结合，提升了重叠社区识别的可解释性与鲁棒性。

**🔧 技术方法**

主要技术包括：影响传播模型构造结构先验、稀疏空间多头注意力机制、层归一化与稀疏正则化、Bernoulli–Poisson 采样重建损失以及传统 GCN/GAT 的对比框架。

**📊 数据集**

基准数据集为 SNAP 的 Facebook ego 网络（6 个网络），安全实验使用自建的 OT/IT 控制系统网络抓包数据，构成加权通信图。

**📈 对比分析**

通过与 GCN、GAT、NOCD、AOCD 等基线在 ONMI 上进行对比，DISCO 在 5/6 个 Facebook 网络上达到或超过最优结果；在安全案例中，使用相对 ONMI 的 3σ 阈值或投票法实现 F1 分别可达 0.98 与 0.92、FPR 低至 10⁻⁴，检测延迟 1–16 分钟。

**⚠️ 局限性**

主要限制包括：需要手工调节先验权重与边缘比例，无法天然处理未见节点（需重新训练）；对噪声与图结构变化敏感，长期漂移与异常难以区分；在大规模网络上需进一步验证其可扩展性与实时性。

---

## 711. Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding

**arXiv ID:** 2609.26638 | [PDF](https://arxiv.org/pdf/2609.26638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 712. Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation

**arXiv ID:** 2609.26693 | [PDF](https://arxiv.org/pdf/2609.26693v1)

**作者:** Lijuan Tang `[一作]` (Northeastern University), Yuemeng Zheng `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究本地代理模型在工具调用协议中的测量误差，揭示服务器栈如何导致模型被误判为无法调用工具

**💡 创新点**

指出本地服务层对工具调用的预检和拒绝会污染评估结果，并提出针对这一测量误差的检查清单

**🔧 技术方法**

利用 Ollama、llama.cpp、vLLM、SGLang 四个服务栈，结合 ReAct 编码代理 harness 与自定义工具调用协议（原生、原生+提示、文本工具）进行实验，使用结构化工具调用解析与约束解码技术

**📊 数据集**

采用 Qwen2.5-Coder、Llama-3.2、Phi-3-mini、Gemma-3 等多模型；在聚合任务、依赖链任务和 HumanEval 任务上进行单轮和多轮工具调用评测

**📈 对比分析**

比较三种协议下的有效工具调用率，发现原生模式受服务器拒绝影响导致 0%；添加文本提示可显著提升大多数模型的准确率，而统一文本协议对 Llama-3.2 则适得其反；交叉栈实验显示同一请求在不同服务层产生不同结果，说明误差主要来自栈而非模型；对比基准不显著优越，性能受限于模型和栈配置

**⚠️ 局限性**

局限性包括：测量结果对任务、种子数和服务器版本高度敏感；仅评估了少数模型和任务；约束解码在弱模型上导致无限循环；未对推理速度、资源占用等指标进行系统评估

---

## 713. A Spectral Theory of Grokking: Weight Decay induces Feature Learning

**arXiv ID:** 2609.26679 | [PDF](https://arxiv.org/pdf/2609.26679v1)

**作者:** Lenz Pracher `[一作]` (Ludwig-Maximilians-Universit�t M�nchen), Steffen Rulands `[通讯]` (Ludwig-Maximilians-Universit�t M�nchen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了深度网络在训练后残差驱动的特征学习与延迟泛化（grokking）的机制，提出了从懒学习到富特征学习的量化理论；

**💡 创新点**

揭示了权重衰减留残差如何通过神经切线核（NTK）反馈促使任务相关特征增大，并预测了延迟泛化的时间尺度与学习率、权重衰减的倒数比例关系；

**🔧 技术方法**

使用神经切线核层级、齐次网络的梯度流分析、一次模式近似与自适应时间尺度，结合模数加法任务进行实验验证；

**📊 数据集**

主要使用模块化加法（modular addition）数据集（如 modulo 23、97 等）进行实验；

**📈 对比分析**

在不同学习率/权重衰减网格上进行大规模实验，比较了训练准确率、验证准确率与理论预测，发现验证准确率跃迁时间与 (ηλ_W)^-1 成正比，实验结果与理论高度吻合；

**⚠️ 局限性**

理论仅适用于齐次网络、平方损失、全批梯度流，对非齐次结构（如 Transformer）仅在宏观层面保持一致；未考虑多任务方向相互作用，且对不同数据分布的普适性还有待验证。

---

## 714. Laryngeal Structure Segmentation in High-Speed Videoendoscopy Using Deep Learning

**arXiv ID:** 2609.26636 | [PDF](https://arxiv.org/pdf/2609.26636v1)

**作者:** Sardar Nafis Bin Ali `[一作]` (Michigan State University), Maryam Naghibolhosseini `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一套基于U‑Net的多类喉部结构分割框架，能够在高速度视频内镜（HSV）图像中同时识别声门区域、声带、舌骨折、咽部折叠等多种解剖结构；

**💡 创新点**

创新点包括：①首次在HSV数据中对连接语音与持续元音进行多类分割；②结合低通频域滤波去噪与直方图均衡化提升图像质量；③在柔性内镜低质量HSV上实现超过95%总体准确率的分割性能；

**🔧 技术方法**

采用U‑Net全卷积网络，配合数据增强、交叉熵损失、Adam优化器，并使用IoU、Dice、精确率、召回率等指标进行评估；

**📊 数据集**

使用14位受试者（8正常、6声障碍）共1400帧HSV图像，涵盖持续元音和连接语音，已人工标注5类结构（背景、舌骨折、声带、舌骨、声门）做为训练/验证/测试集；

**📈 对比分析**

通过比较三种U‑Net深度（UNET‑4/5/6）在测试集上的表现，UNET‑6在IoU、Dice和整体准确率上表现最佳，IoU最高达0.9533、Dice最高0.9761、总体准确率0.9659，声门和声带等关键区域Dice>0.89；

**⚠️ 局限性**

局限性包括：低质量柔性内镜图像导致噪声与灰度不均匀；类别不平衡导致大类过度预测、少类欠预测；对极端姿态、遮挡及非典型发音敏感；需要更多样本和更先进的网络架构或类别平衡策略来提升泛化与鲁棒性。

---

## 715. MambaVoice: Lightweight Audiovisual Singing Voice Separation Via A Hybrid Mamba-Transformer Model

**arXiv ID:** 2609.26635 | [PDF](https://arxiv.org/pdf/2609.26635v1)

**作者:** Adithi Shankar `[一作]` (Universitat Pompeu Fabra), Martín Rocamora `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量化的多模态歌唱声源分离框架MambaVoice，能够在多声乐与伴奏环境中定位并提取目标歌手声音。

**💡 创新点**

创新点在于将选择性状态空间模型（Mamba）与Transformer相结合，形成混合Mamba–Transformer骨干；使用基于注意力的频段划分音频编码器；采用乘法门控机制将面部运动特征与音频特征融合，提升对目标声源的识别与抑制干扰。

**🔧 技术方法**

核心技术包括：注意力驱动的Band‑Split音频编码器、空间‑时间图卷积网络（ST‑GCN）提取面部运动；乘法门控多模态融合；混合Mamba–Transformer骨干（先Mamba再Transformer）实现线性时间建模与全局频谱融合；复杂比率掩膜估计与ISTFT重建。

**📊 数据集**

主要使用Acappella（含视频同步的独唱录音）进行训练和测试，并在URSing数据集上评估跨域泛化，训练时使用MUSDB18伴奏和AudioSet的声学混合做数据增强。

**📈 对比分析**

与VoViT、HT Demucs等基准模型比较，MambaVoice在Acappella 50%干扰测试中取得14.18 dB SDR，仅16.2 M参数（比VoViT的39 M少58%），在100%干扰测试中得到11.11 dB SDR，性能略低于VoViT但稳健；在URSing上平均SDR 5.58 dB，优于VoViT；推理速度最快，3.93 ms/4s，速度比VoViT快2.94×。

**⚠️ 局限性**

局限性包括对面部跟踪的依赖（AlphaPose预处理未计入延时）；在极端干扰（100%干扰）下性能仍低于纯音频方法；对较短音频段与面部检测不稳定性研究不足；缺乏对更复杂多声乐、跨语言歌唱场景的验证。

---

## 716. EquivSVA: A Formally Verified Dataset of Behavioral Assertions Across Equivalent RTL Implementations

**arXiv ID:** 2609.26751 | [PDF](https://arxiv.org/pdf/2609.26751v1)

**作者:** FNU Aditi `[一作]` `[通讯]`, FNU Aditi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个正式验证的行为等价数据集EquivSVA，包含120个行为族，每个族有四种结构不同的实现、共享的金属性质和三个受控突变体。

**💡 创新点**

其创新点在于将行为等价性作为数据集的核心组织原则，使得可比较不同实现对断言生成的鲁棒性，并提供完整的17项验证流程。

**🔧 技术方法**

使用了基于Yosys、SymbiYosys、ABC/PDR等工具的形式化验证、RTL生成器以及突变测试技术。

**📊 数据集**

数据集本身即为所用的数据，包含480个RTL实现、914条金属性质和360个突变体；同时还用公开的Qwen2.5‑Coder‑7B‑Instruct模型进行案例评估。

**📈 对比分析**

通过将模型在四种实现上的生成结果进行语法、形式化可行性和突变检测的比较，发现Qwen2.5‑Coder‑7B在测试集上有31.7%属性可被证明为正确，58.3%族的属性数在不同实现间变化。

**⚠️ 局限性**

限制在于数据集是程序化合成的，主要覆盖控制逻辑和小型FSM，缺乏大规模数据通路和复杂协议；突变体不代表工业缺陷；仅做单模型案例分析。

---

## 717. Dynamic Slack-Aware Clocking for Near-Threshold Tensor Processing Units (TPUs)

**arXiv ID:** 2609.26644 | [PDF](https://arxiv.org/pdf/2609.26644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 718. GAD-MambaUNet: Direction-Group Mamba with Gradient-Adaptive DINOv3 Distillation for Lightweight Medical Image Segmentation

**arXiv ID:** 2609.26729 | [PDF](https://arxiv.org/pdf/2609.26729v1)

**作者:** Fang Wang `[一作]` (Beijing Institute of Petrochemical Technology), Xinxin Yang `[通讯]` (Beijing Institute of Petrochemical Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化医学图像分割网络GAD‑MambaUNet，采用对称的局部–全局结构，结合方向‑组图形选择扫描（DG‑GSS）和训练时的梯度自适应蒸馏（GAD）提升分割精度；

**💡 创新点**

创新点在于：① DG‑GSS将扫描方向与通道组视为图节点，实现跨方向和跨子空间的结构化信息交换；② GAD在训练时动态调节教师蒸馏强度，兼顾主任务与语义监督；③ 将轻量卷积与视觉状态空间模型结合，形成高效的局部‑全局分割架构；

**🔧 技术方法**

使用技术包括：轻量卷积块、视觉状态空间模型（Mamba/VMamba）、多方向选择扫描、图卷积（GNN）消息传递、冻结DINOv3教师的特征蒸馏、梯度自适应调节、残差结构、深度可分离卷积等；

**📊 数据集**

采用四个公开医学分割数据集：PH^2（皮肤病变）、ISIC2018（皮肤病变）、CVC‑ClinicDB（息肉）和CVC‑ColonDB（息肉）；

**📈 对比分析**

与U‑Net、PraNet、UACANet、TransUNet、UNeXt、CMUNeXt、EGE‑UNet、UltraLight VM‑UNet、MK‑UNet等方法对比，GAD‑MambaUNet在保持参数≈0.5 M、FLOPs≈0.32 G的同时，平均Dice从91.66%提升至92.27%，在极低资源下实现了最佳或接近最佳的精度；在加入DINOv3蒸馏后，Dice进一步提升到95.1%（PH^2）等；

**⚠️ 局限性**

局限性包括：仅针对二分类分割，未验证多类别或多模态场景；蒸馏过程需要额外训练时间与对齐策略；DG‑GSS在极低分辨率下的表现可能受限于图结构简化。

---

## 719. A Data-Interventional Framework for Auditing Privacy and Fairness in Generative Medical Imaging

**arXiv ID:** 2609.26623 | [PDF](https://arxiv.org/pdf/2609.26623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 720. The Sirens' Song: When Proximal Background Context Overshadows Distant Evidence

**arXiv ID:** 2609.26718 | [PDF](https://arxiv.org/pdf/2609.26718v1)

**作者:** Xiaoyu Yang `[一作]`, En Yu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究长上下文语言模型中，为什么远程证据往往被忽略，并提出了“Proximity Trap”概念；

**💡 创新点**

提出LYRA（一种t分布方向匹配机制），通过重塑注意力分数来减少靠近但无关背景对远程证据的竞争；

**🔧 技术方法**

核心技术是使用RoPE变换后的查询-键方向相似度，并通过t分布函数调整得分后再 softmax；

**📊 数据集**

在LongBench‑v2、RULER、LongBench 以及新构造的 ProxBench 上进行评估；

**📈 对比分析**

与多种基线方法相比，LYRA 在不同上下文长度、任务类别和 ProxBench 的四级扰动中均取得最高或最稳健的准确率，提升幅度可达 10% 以上；

**⚠️ 局限性**

局限性在于仅在最终 Transformer 层应用 LYRA，未对更大规模模型或更深层级做充分验证，且对极端长序列的推理效率影响尚未系统评估。

---

## 721. Imperfection for Precision: Upcycling Imperfect Data for High-Precision Robotic Manipulation

**arXiv ID:** 2609.26672 | [PDF](https://arxiv.org/pdf/2609.26672v1)

**作者:** Hao Wei `[一作]` (Samsung Robotics eXperience), Tingguang Li `[通讯]` (Samsung Robotics eXperience)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ε4P 方法，通过在流匹配轨迹上为不同数据源设定门限，将低精度和任务不匹配的数据按噪声区段引入，高精度机械臂操作得到提升。

**💡 创新点**

创新点在于将源相关的流时间门控与动作-上下文依赖分析相结合，使不同数据源在不同噪声水平下提供最合适的监督，而非统一混合或全局加权。

**🔧 技术方法**

采用流匹配（flow‑matching）与扩散模型的动作生成技术，利用动作-上下文依赖度量和源可分辨性判别器自动确定门限。

**📊 数据集**

使用 Rainbow Robotics RB‑Y1 机器人在 ATX 24‑pin 插头、两级电缆插接、螺栓‑螺母分类三项真实任务收集的数据，包含 Teleoperation（高精度）、UMI（低精度）以及三插头任务的辅助数据。

**📈 对比分析**

与原始共训练、全局损失加权等基线对比，在三项任务上成功率最高提升31.7%，并能以平均4.2%点的精度下降替代等量高质量数据。

**⚠️ 局限性**

局限在于仅在单一机器人平台、有限示例、受控相机设置下验证，缺乏跨平台、不同硬件配置以及更大多样化数据集的可扩展性评估。

---

## 722. TraceVIC: Causal Reasoning over Code Evolution for Identifying Vulnerability-Inducing Commits

**arXiv ID:** 2609.26711 | [PDF](https://arxiv.org/pdf/2609.26711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 723. Evaluating the Semantic-to-Geometric Gap in Adversarial Defenses Against Vision-Language Model-Based Plagiarism

**arXiv ID:** 2609.26733 | [PDF](https://arxiv.org/pdf/2609.26733v1)

**作者:** Christopher Burger `[一作]` (Pelican Quantitative), Charles Walter `[通讯]` (University of Mississippi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉-语言模型（VLM）在教育场景下因学生上传图像导致的“trivial plagiarism”，并评估利用启发式图像扰动（像素噪声、扭曲、线条、复印等）作为防御手段的效果，采用两阶段方法：第一阶段手工评估电路图，第二阶段自动评估逻辑门图与Karnaugh图；

**💡 创新点**

首次提出并系统验证针对VLM的视觉扰动作为临时防护策略；将手工与自动评估结合，揭示正向扰动与模型对不同扰动类型的异质反应；进一步阐明VLM在几何对齐与拓扑识别方面的弱点；

**🔧 技术方法**

启发式图像扰动技术（扭曲、像素噪声、线条、复印等）；VLM交互评估（手工网页接口与API调用）；输出解析（正则+SymPy符号比较）；自动化评测框架（Pass@Any、Majority Vote、Strict Consistency）；

**📊 数据集**

手工挑选的25道初级电路题；自动生成的80道逻辑门图、30道4变量Karnaugh图；所有数据均为公开或作者自行生成并已验证；

**📈 对比分析**

使用Pass@Any、Majority Vote、Strict Consistency三种评测标准；在无扰动时，Gemini 2.5 Pro达到90% Strict；在扰动后仍保持70% Strict，Gemini Flash从80%降至53%；Claude Haiku、GPT‑4o‑mini在无扰动下表现低下，扰动后进一步下降；整体表明扰动可显著削弱VLM准确率，尤其对低容量模型更为敏感；

**⚠️ 局限性**

未进行单独扰动的消融实验；仅评估低成本API级模型，未覆盖前沿高性能模型；扰动对视觉障碍或使用辅助技术学生可能产生负面影响；模型仍能在高强扰动下解答，表明视觉模糊不是长期可行的防护；需要进一步研究更改评估设计以根本抵御VLM作弊。

---

## 724. ASTRA-SR: Atmospheric Seeing and Turbulence Restoration for Astronomical Image Super-Resolution

**arXiv ID:** 2609.26731 | [PDF](https://arxiv.org/pdf/2609.26731v1)

**作者:** Xining Ge `[一作]` (Hangzhou Dianzi University), Shuhong Liu `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理驱动合成数据集训练的盲单帧行星图像恢复框架，能够同时进行去噪、去模糊和超分辨率处理。

**💡 创新点**

创新点包括：① 使用高动态范围太空RAW图像与真实大气层测量构建物理化合成低分辨率样本；② 设计分阶段空间频率网络：先估计去噪但保留模糊的LR图像，再进行多尺度模糊感知恢复，最后通过序列空间‑幅度分支重建高分辨率细节。

**🔧 技术方法**

技术细节：物理湍流模拟（多层相位屏幕+分步菲涅尔传播+曝光平均PSF）、卷积网络（SCGN、Patch‑Fourier、空间‑幅度分支）、频域分析、分层监督与残差增益控制。

**📊 数据集**

数据集：约20,000张Cassini ISS的高动态范围行星RAW图像，经过物理模型合成得到63,582个训练样本和1,355个测试样本。

**📈 对比分析**

与Bicubic、Bilinear、NAFNet、Restormer、FFTformer、SMFANet、PlaNet、RDBM、SCGN、StarIR等方法对比，在PSNR、SSIM和对象PSNR上分别提升0.55 dB、0.003、0.49 dB，显著优于最强基线。

**⚠️ 局限性**

局限性：实验仅在合成数据和少量真实观测上验证，缺乏跨仪器、不同观测条件的全面量化评估；对极端噪声或模糊情况的鲁棒性仍待进一步研究。

---

## 725. Does AI Save Time on Product Design? A Randomized Controlled Experiment of AI Prompt-to-Design Workflows

**arXiv ID:** 2609.26725 | [PDF](https://arxiv.org/pdf/2609.26725v1)

**作者:** Remy Stewart `[一作]` (Figma), Augustus Griffin `[通讯]` (Figma)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 100 名具备至少两年经验的产品设计师与产品经理进行随机对照实验（RCT），比较使用与不使用 Figma Make 提示式设计 AI 工具时的完成时间、完成率、任务易度与系统可用性。

**💡 创新点**

首次使用 RCT 方法量化提示式设计工具在实际设计任务中的时间节省，并探讨不同角色（设计师 vs 产品经理）对 AI 效果的差异，提供了 AI 帮助非设计师参与设计的实证证据。

**🔧 技术方法**

技术手段包括：Figma Make AI、专用时间跟踪小部件、通用估计方程（GEE）模型、卡方检验、Mann-Whitney U 检验、System Usability Scale（SUS）和 7 分制易度量表。

**📊 数据集**

使用实验收集的数据：100 名参与者的任务完成时间、完成率、易度分数、SUS 分数，以及三项标准化设计任务（暗色模式转换、菜单添加、评论弹窗）结果；未使用公开数据集，全部为实验生成数据。

**📈 对比分析**

对照组与实验组在完成时间上通过 GEE 进行差异检验，易度与 SUS 通过卡方和 Mann-Whitney U 比较。实验组总体时间减少约 20%（累计约 15 分钟），任务 2 与 3 的时间分别减少 27% 与 32%；产品经理在实验组获得 35% 的时间节省；易度平均提升 0.8 分，SUS 提升 10 分（约 15% 的相对提升）。

**⚠️ 局限性**

局限性包括：仅测试三项固定顺序、目标明确的任务，未覆盖真实项目的不确定性；只评估成功完成的任务，忽略了完成率的差异；样本规模有限、仅包含设计师与产品经理，无法推广到其他角色或新手用户；Figma Make 在实验版本中偶尔出现渲染失败，可能导致时间估计偏低；对设计质量的评估仅来自参与者自评，缺乏专家判断。

---

## 726. Detecting GPT-Assisted Writing Using Interpretable Stylometric Features

**arXiv ID:** 2609.26687 | [PDF](https://arxiv.org/pdf/2609.26687v1)

**作者:** Rajesh Kumar `[一作]` (Bucknell University), Alexander Fuchsberger `[通讯]` (Bucknell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用可解释的风格学特征和随机森林模型，对学生是否使用 GPT 进行写作辅助进行检测。

**💡 创新点**

创新点在于仅通过提交文本的九个可解释特征进行检测，并对训练和测试集采用用户分离策略，最后使用 SHAP 解释模型决策。

**🔧 技术方法**

技术方法包括滑动窗口（250 词/125 词步长）提取 TTR、Hapax Ratio、词熵、NSR、词性比例等九个特征，使用随机森林、逻辑回归、朴素贝叶斯、LDA、QDA、SVM、kNN 和 MLP 等八种分类器，并用 SHAP 进行全局和局部解释。

**📊 数据集**

数据集为 IIITD-BU（Paraphrased）数据集，共 90 名学生，每人完成一次独立写作和一次 GPT 辅助写作（共 180 篇文本）。

**📈 对比分析**

通过重复 10 折交叉验证和 5×5 嵌套验证比较八个分类器，随机森林在验证集上 ROC‑AUC 最高；在 18 位未见过的学生（36 篇文档）的测试集上得到 ROC‑AUC 0.870、F1 0.842，误报率约 22%，漏报率约 11%。

**⚠️ 局限性**

局限性包括样本量小、仅评估一种 GPT 辅助（重写）方式、可能受写作顺序影响、未考察不同人群或语言的公平性、只使用英文文本以及在新型 LLM 或多样化写作场景下的泛化能力未得到验证。

---

## 727. Decoding the Legalese: A Scalable and Quantitative Framework for Analyzing Corporate Privacy Policies

**arXiv ID:** 2609.26680 | [PDF](https://arxiv.org/pdf/2609.26680v1)

**作者:** Jiaming Tang `[一作]` (University of Michigan), Armin Sarabi `[通讯]` (University of Michigan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于LLM的端到端管道，能够将原始隐私政策文本自动转换为细粒度结构化表示，并基于此定义并计算完整性、透明度、商业利益与用户保护四个可量化指标；

**💡 创新点**

首次实现了使用LLM自动化生成完整、可重复的隐私政策结构化数据，并提出了统一的四维度评价框架，进而在10,000份政策上进行大规模量化对比，揭示行业差异与利益冲突；

**🔧 技术方法**

采用instruction‑tuned大型语言模型（如OpenAI GPT‑4）结合自定义prompt、扩展版数据元素与实践taxonomy、Web爬虫、Docling文本预处理以及PCA与问卷权重学习等技术；

**📊 数据集**

使用来自Russell 3000和Tranco榜单的10,000个网站隐私政策，涵盖多行业，构成结构化提取结果集；

**📈 对比分析**

通过人工验证与第二评估阶段的对比，LLM提取准确率达98–99%；四维度指标采用PCA确定权重，实验显示与专家与问卷权重高度相关，指标稳健且能揭示行业内的显著差异与潜在权衡；

**⚠️ 局限性**

局限性包括：LLM内容过滤与语义歧义导致部分提取错误；主要聚焦英语政策，对非英语内容支持不足；对极短或缺失政策缺乏覆盖；对新兴法规或特定业务场景的细粒度适配性仍需提升。

---

## 728. Capable yet Parsimonious: Extracting and Characterizing Hidden Chain-of-Thought in Frontier Models

**arXiv ID:** 2609.26637 | [PDF](https://arxiv.org/pdf/2609.26637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 729. MAGIC: Mixed-Granularity Agent Graphs via Incremental Construction with Dense-Reward Reinforcement Learning

**arXiv ID:** 2609.26667 | [PDF](https://arxiv.org/pdf/2609.26667v1)

**作者:** Kairui Yang `[一作]`, Rong-Hua Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种混合粒度的多智能体组织生成框架，能够在任务级别上动态选择单一智能体或可重用组的实现，并通过增量构建得到执行组织。

**💡 创新点**

创新点在于将粒度选择本地化到每个功能角色，同时采用基于潜在的稠密奖励重塑来直接从当前策略轨迹学习，而无需先搜集成功示例。

**🔧 技术方法**

主要技术包括基于策略梯度的增量构建算法、潜在奖励塑形、混合粒度图空间建模以及冻结LLM执行器的多智能体执行框架。

**📊 数据集**

使用了八个基准数据集，涵盖知识推理（MMLU-Pro、StrategyQA）、数学推理（AQuA、GSM8K）、代码生成（HumanEval、LiveCodeBench-v6）、表格/金融推理（TAT-QA、TabFact）。

**📈 对比分析**

与17个基线（包括直接单智能体、传统多智能体系统及学习型组织）对比，本文方法在所有八个基准上均名列第一，并在四个代表性任务上实现了性能-token Pareto 前沿。

**⚠️ 局限性**

局限性包括对预定义角色库的依赖、训练时对API成本的高需求以及在极端大规模任务或对群体内部结构变化敏感的情境中可能面临的扩展性挑战。

---

## 730. FleXray: Universal Clinical X-ray Segmentation

**arXiv ID:** 2609.26756 | [PDF](https://arxiv.org/pdf/2609.26756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 731. ROAM-ASD: Robust Open-World Active Speaker Detection with Flexible Multimodal Fusion

**arXiv ID:** 2609.26648 | [PDF](https://arxiv.org/pdf/2609.26648v1)

**作者:** Pu Wang `[一作]` (KU Leuven), Hugo Van hamme `[通讯]` (KU Leuven)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于音频、全脸以及细粒度嘴部特征联合建模的活跃说话者检测框架 ROAM-ASD。

**💡 创新点**

核心创新在于：①使用统一的联合自注意力机制并引入模态无关的查询令牌，实现所有可用模态直接交互；②在训练阶段加入模态丢弃（modality dropout）以提升对缺失模态的鲁棒性；③首次显式利用嘴部区域的细粒度视觉表征来补充全脸信息，进一步提升说话与非说话的区分。

**🔧 技术方法**

技术细节包括：Whisper‑large‑v3 作为冻结的音频编码器；3D卷积+ResNet‑18+V‑TCN 的轻量化全脸编码器；基于 Auto‑AVSR 的预训练嘴部编码器并通过 LoRA 微调；联合自注意力层使用 RoPE+ALiBi 时序偏置；以及跳跃连接+Transformer 时序头进行最终概率预测。

**📊 数据集**

实验使用五大活跃说话者检测基准：AVA、WASD、UniTalk、Talkies、ASW，并在这些数据集上进行原域及零射跨域评估。

**📈 对比分析**

与现有方法（如TalkNet、LoCoNet、TalkNCE 等）对比，ROAM-ASD 在所有数据集上均取得最高 mAP，提升幅度分别为：WASD 5.1、UniTalk 4.7、AVA 0.9、ASW 1.0、Talkies 2.1；在零射跨域实验中也显著优于 TalkNCE；对连续与散布式缺失模态表现出强鲁棒性，尤其在使用模态丢弃训练后更为稳健。

**⚠️ 局限性**

局限性包括：当完整可视信息完全缺失时仍难以正确关联音频；依赖 MediaPipe FaceMesh 进行嘴部提取，对极端遮挡或极小人脸的鲁棒性有限；模型仍需多模态训练数据，单模态迁移仍需进一步探索。

---

## 732. Grow the Harness, Not the Context: From Strategy-Free Scaffolds to Reusable Specialist Agents

**arXiv ID:** 2609.26760 | [PDF](https://arxiv.org/pdf/2609.26760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 733. Reading the Sky to Forecast the Ground: Physics-Informed Link-State Forecasting for LEO Networks at Any Location

**arXiv ID:** 2609.26696 | [PDF](https://arxiv.org/pdf/2609.26696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 734. From Alignment to Access Control: A Framework for GenAI Policy Enforcement

**arXiv ID:** 2609.26682 | [PDF](https://arxiv.org/pdf/2609.26682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 735. Beyond Repeated Sampling: Learning Search Policies for LLM Reasoning

**arXiv ID:** 2609.26704 | [PDF](https://arxiv.org/pdf/2609.26704v1)

**作者:** Ismail Labiad `[一作]` (Meta FAIR), Julia Kempe `[通讯]` (Meta FAIR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出将概念生成器（Concept Generator, CG）训练为一项搜索策略，用于引导更大且被冻结的答案生成器（Answer Generator, AG）在推理时更高效地探索解空间；

**💡 创新点**

创新点在于：①将概念生成从逐条迭代改为一次性多概念生成；②以RL训练小型CG，奖励依据AG最终解答的成功率，形成可迁移的探索策略；

**🔧 技术方法**

技术包括：强化学习（GRPO式）训练CG；单轨迹概念生成与多概念分配；利用LLM判别器评估答案；max-of-max和max-of-mean两种奖励聚合；

**📊 数据集**

使用数据集：DeepMath-103k训练集、DeepMath 1k硬题集合、Omni-MATH 2 作为OOD评估；

**📈 对比分析**

对比方法：原始概念引导采样、无训练的CG、无条件重复采样及通用提示；在DeepMath硬题集上，RL‑训练的7B CG在pass@128上从19%提升至39.2%（约翻倍），并在Omni-MATH 2上亦显著优于基线；

**⚠️ 局限性**

局限性：训练成本高（需大量AG roll‑outs与判别）；奖励仅按整条轨迹分配，未细化到单概念；评估仅聚焦数学推理，未覆盖更广泛任务；

---

## 736. Achieving Robust Performance using Minimal Communication in Resource Allocation Games

**arXiv ID:** 2609.26670 | [PDF](https://arxiv.org/pdf/2609.26670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 737. Discovery-Driven Integration of Disjoint Tables via Text

**arXiv ID:** 2609.26658 | [PDF](https://arxiv.org/pdf/2609.26658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 738. Databases with Missing Values that are Governed by Missingness Mechanisms

**arXiv ID:** 2609.26692 | [PDF](https://arxiv.org/pdf/2609.26692v1)

**作者:** Leopoldo Bertossi `[一作]` (Carleton University), Farouk Toumani `[通讯]` (Clermont Auvergne University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于缺失机制贝叶斯网络（Missingness Graph, MG）的关系数据库语义，并将观察到的含缺失值数据库与 MG 结合生成 Block-Independent Probabilistic DB（BID），从中产生可能世界集合。随后将可能世界按多重集合相同划分为匹配类，并引入两类重要的类集合：统计合规性最高的 Most‑Compliant Classes（MCC）和概率最高的 Most‑Probable Classes（MPC），在这两类上定义查询答案。

**💡 创新点**

创新点包括：
1) 通过 MG 明确刻画缺失值产生的因果机制；
2) 将 MG 与观察数据库相结合，构造 BID，从而得到既考虑随机不确定性又能量化统计合规性的多世界模型；
3) 提出匹配类与 MCC/MPC 两种新类语义，兼顾概率与统计一致性；
4) 对 MCC/MPC 的计算复杂性做了系统分析，证明单个 MCC 可多项式求解、枚举可多项式延迟，而 MPC 相关问题普遍为 #P‑hard；
5) 探讨在仅给定定性 MG 的情况下如何从观测数据恢复必要的概率参数。

**🔧 技术方法**

使用技术：
- 贝叶斯网络与 Missingness Graph；
- Block‑Independent Probabilistic Database 架构；
- 多世界概率推理与查询答案求解；
- 统计距离度量（KL‑divergence、欧氏距离、χ² 检验）用于合规性评估；
- 组合优化（最小成本流、完美匹配）用于求解 MCC；
- 复杂度理论（#P、NPPP、FP、DelayP 等）对算法性能进行理论评估。

**📊 数据集**

文中未给出具体实验数据集；在结论中提到正在使用真实数据集进行实验，未来会发布实验结果。

**📈 对比分析**

比较方式主要是理论复杂度对比：与传统可能世界语义、TID 语义以及最可能世界语义的计算难度进行对比；并说明 MCC 语义在多项式时间内可求解单个类，而 MPC 语义涉及 #P‑hard 问题。由于缺少实测数据，性能评价目前仅停留在理论层面。

**⚠️ 局限性**

局限性包括：
- MG 必须预先给定，未涉及学习或校验 MG 的方法；
- 仅对缺失机制为 I‑deterministic 的 MG 适用；
- MPC 相关问题普遍高度复杂，实际查询时难以直接求解；
- 仅考虑了无序关系（bag 语义），对 set 语义的细节未深入；
- 实验验证不足，缺乏对实际数据集性能的评估；
- 对多表、联接查询的扩展未在本文中完整讨论。

---

## 739. Strong Selective and List-Decoding Direct Product Theorems for Quantum Query Complexity

**arXiv ID:** 2609.26678 | [PDF](https://arxiv.org/pdf/2609.26678v1)

**作者:** Paul Beame `[一作]` (University of Washington), Michael Whitmeyer `[通讯]` (University of Washington)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了量子查询复杂度中的强直接积定理，并将其推广为选择性直接积和列表解码直接积两种更强的形式，给出了对应的下界证明，并将这些结果应用于量子时空权衡下界的改进。

**💡 创新点**

创新点主要包括：①提出了一种新的多重对偶（multiplicative adversary）方法，可同时适用于关系与函数，并能够捕捉负权对偶方法的下界；②在此框架下证明了所有（尤其是布尔）函数均满足量子强选择性直接积定理和列表解码直接积定理；③将这些定理用于改进量子时空权衡下界，超越了仅适用于输出不变算法的先前结果。

**🔧 技术方法**

技术手段包括：多重对偶方法、负权对偶方法与其转换、记录查询技术、超对数/超乘法不等式、熵函数与二项式系数估计、递归归纳、半正定规划以及量子信息理论中的投影与谱分解。

**📊 数据集**

论文为理论研究，未使用具体实验数据；所用随机分布主要是经典的硬分布（如Hamming weight 1、全零与单1的混合分布）以及均匀分布，作为对输入分布的刻画。

**📈 对比分析**

与已有的量子直接积定理、经典直接积定理以及时间空间权衡下界进行对比，证明了在量子模型下对所有函数实现理想的Ω(k)量级下界，且对比结果表明在选择性和列表解码场景中已显著提升了下界质量。

**⚠️ 局限性**

局限性：①列表解码直接积定理仅在布尔输出函数中成立，对一般关系不适用；②多重对偶方法对参数η有严格限制（η≤1/2），在更大范围内可能不成立；③部分定理对ε、δ、λ等参数有较严格约束，需精细调参；④实现该框架仍需构造复杂的对偶矩阵，实际可行性与计算成本尚待评估。

---

## 740. Stepping into the Margins: How Readers Want AI to Generate Footnotes

**arXiv ID:** 2609.26673 | [PDF](https://arxiv.org/pdf/2609.26673v1)

**作者:** Piper Vasicek `[一作]` (Brigham Young University), Kevin Seppi `[通讯]` (Brigham Young University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过访谈与主题分析，本文探讨读者对 AI 生成脚注的需求与评价标准，并提出不同类型脚注的设计框架。

**💡 创新点**

创新点在于系统化总结读者对脚注功能的多样化需求，并将这些需求映射到可实现的 AI 技术路径，填补了现有研究对脚注技术需求缺失的空白。

**🔧 技术方法**

主要技术包括自然语言处理、文本摘要、语义理解、检索增强生成（RAG）、LLM 生成与多模态检索等。

**📊 数据集**

使用的“数据集”是对13位高学历读者的访谈记录与转录文本。

**📈 对比分析**

方法为定性主题分析，无量化性能指标；作者通过比较不同脚注类型与质量标准，提供了设计准则，但未进行系统实现与实验验证。

**⚠️ 局限性**

局限性包括受访者样本偏向高学历宗教读者，样本量有限，且对 AI 脚注技术的技术可行性与用户体验未做实证评估。

---

## 741. The Delegation Blind Spot: Auditing Product Decisions from Agent Choices

**arXiv ID:** 2609.26642 | [PDF](https://arxiv.org/pdf/2609.26642v1)

**作者:** Shivam Gupta `[一作]` `[通讯]` (Independent Research), Shivam Gupta (Independent Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对代理执行结果的决策特定审计方法，能够根据声明的观察通道和产品价值对比，生成可兼容的区间与示例人群；

**💡 创新点**

创新点在于将识别理论与决策理论相结合，形成可执行的测量工作流，能够区分结构性模糊与采样不确定，并提供离线查看器；

**🔧 技术方法**

采用线性规划、概率通道模型、偏差感知线性证书、以及随机抽样误差分析等技术；

**📊 数据集**

使用自定义合成任务集，涵盖旅行、云方案和工作流软件三种情景，并在不同意图类别下生成伪造的用户偏好；

**📈 对比分析**

通过对比两种GPT-5.4模型（Mini与Nano）在不同日志条件下的区间宽度与决策决议，发现常规动作日志几乎未能决议，而加入显式偏好记录的“receipt”能在部分情形下缩小50%区间并提高决议率；

**⚠️ 局限性**

局限包括：仅使用合成数据、未涉及真实客户、实验规模受限、覆盖率为单独边际而非同时保证、以及对通道漂移与缺失类别的假设较强等问题。

---

## 742. Knowledge Pull Requests for Continual Document Authoring

**arXiv ID:** 2609.26634 | [PDF](https://arxiv.org/pdf/2609.26634v1)

**作者:** Alexander Martin `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出知识拉取请求（KPR）框架，用以持续改写文档并提供可审查的知识变更日志。

**💡 创新点**

创新点在于把知识提炼为原子命题进行过滤、路由和冲突标记，分离知识层次的变更与文本差异，提升可解释性与审阅效率。

**🔧 技术方法**

技术核心包括：LLM驱动的命题分解、覆盖/冲突/相关性分类、路由到文档段落以及文本重写生成，全部以 Qwen3.5‑27B 等大模型实现。

**📊 数据集**

使用的数据集包括：多语言维基百科（MegaWika 2.0）进行跨语言修订，以及 RAGTIME 的多轮问答报告更新任务（时间、冲突、平衡三种变体）。

**📈 对比分析**

与三种基线（直接文本重写 ConText、无过滤命题 ConClaim、全重写 Scratch）比较，KPR 在信息精度、召回、问答准度、编辑成本（词级距离、连块数、保留率）上均优于基线，尤其在多语言知识融入和冲突处理上表现突出。

**⚠️ 局限性**

局限性主要是：整体流程对 LLM 调用成本高；实验未包含真实人工审阅流程；冲突解决机制仍缺乏自动化信任评估。

---

## 743. PERSONAWEAVER: Controllable Diversity Beyond Conventional Archetypes in Procedural Character Generation

**arXiv ID:** 2609.26629 | [PDF](https://arxiv.org/pdf/2609.26629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 744. Metrics Failure in LLM-Based Code Vulnerability Repair: An Empirical Study and a Change-Aware Screen

**arXiv ID:** 2609.26749 | [PDF](https://arxiv.org/pdf/2609.26749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 745. Annihilator and twisted Euclidean duality for quasi-polycyclic codes

**arXiv ID:** 2609.26633 | [PDF](https://arxiv.org/pdf/2609.26633v1)

**作者:** Tushar Bag `[一作]` (SRM University-AP), Daniel Panario `[通讯]` (Carleton University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了与消灭子对偶性相关的准多循环（QP）码，展示了其消灭子对偶的非退化性，并给出了基于值域点积的对偶的等效描述。

**💡 创新点**

提出了消灭子对偶的结构和性质，特别是其与汉明权重计数器的相互作用，并计算了与该对偶性相关的MacWilliams变换。

**🔧 技术方法**

使用了消灭形式、汉克尔Gram矩阵和坐标映射等技术。

**📊 数据集**

使用了有限域和多项式环的相关数据集，特别是与特定多项式f相关的QP码。

**📈 对比分析**

通过与传统的欧几里得对偶性进行比较，展示了消灭子对偶的性能，结果表明在特定条件下，消灭子对偶可以转化为普通的欧几里得对偶性。

**⚠️ 局限性**

限制在于消灭子对偶的性质可能不适用于所有QP码，特别是在某些情况下可能无法找到保持对偶性的坐标映射。

---

## 746. Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference

**arXiv ID:** 2609.26621 | [PDF](https://arxiv.org/pdf/2609.26621v1)

**作者:** Gaoyuan Du `[一作]` (University of Tennessee), Xueping Li `[通讯]` (University of Tennessee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在 BF16 与 FP16 精度下贪婪解码的输出差异，并提出基于低精度头部重算的纠正方法。

**💡 创新点**

提出精度不变性不成立、误差主要归因于 lm_head 的小余量导致的翻转，并通过门控 FP32 lm_head 重算实现低开销纠正。

**🔧 技术方法**

采用误差传播分析、top‑two margin 门控、FP32 头部矩阵乘、以及与全量 FP32 计算、量化、温度锐化等基线的实验对比。

**📊 数据集**

使用公开评测集 GSM8K、HumanEval、MBPP 等。

**📈 对比分析**

与 FP32 oracle、global FP32 计算、量化、温度锐化等方法对比，门控 FP32 lm_head 重算在 A10G 上提升 Exact Agreement Rate 22‑36pp，延迟提升 <4%。

**⚠️ 局限性**

仅在低批量（bs≤4）贪婪推理时有效；在大规模、批量化或 FP8 整体量化时无效；无法消除权重截断导致的差异。

---

## 747. Longitudinal Retinal Vascular Remodeling in Myopic Children Treated with Orthokeratology or Defocus Lenses: A Two-Year Comparative Study

**arXiv ID:** 2609.26662 | [PDF](https://arxiv.org/pdf/2609.26662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 748. Certification complexity of Boolean functions

**arXiv ID:** 2609.26757 | [PDF](https://arxiv.org/pdf/2609.26757v1)

**作者:** Chandrima Kayal `[一作]`, Jevgēnijs Vihrovs `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种统一的“认证”框架，对多种查询模型（确定性、随机化、量子零误差、量子精确等）下的证书复杂度进行了系统研究；在此框架下给出了证书复杂度与已有量度（如判定树复杂度、随机/量子区分复杂度、障碍复杂度、期望证书复杂度、非确定性多项式度数、 rational 以及一侧近似度数）之间的新等价与上界/下界关系，并展示了若干功能的分离示例。

**💡 创新点**

创新点在于：①提出了与传统证书概念相互独立的“认证”任务并给出严格定义；②通过认证变换统一推导出多种已知复杂度量的等价性；③引入一侧近似多项式度数作为新的量度，并证明其与量子零误差复杂度之间的下界关系；④给出了一些功能（如 f_m）实现的证书复杂度与 rational 之间的显著分离，证明了证书复杂度作为新的下界技术的潜力。

**🔧 技术方法**

主要技术手段包括：量化的证书变换（z‑certification）、稳态性与 Sandwich Lemma 的抽象工具、量子振幅放大与精确振幅放大、对多项式的约束（如 block‑multilinear、非确定性多项式）、选择族与选择器的构造以及对分离函数的 Grover 量子搜索实现。

**📊 数据集**

本研究为理论性质研究，未使用任何具体实验数据集；所有结论均基于理论分析与构造例子。

**📈 对比分析**

通过对比已知的下界技术（如 rational 近似度数、障碍复杂度、期望证书复杂度等），论文证明了在量子零误差与精确查询复杂度下，证书复杂度给出了新的严格下界，并在部分函数上与传统度数下界形成显著分离；同时给出了多种等价关系，使得不同模型下的复杂度量可相互转换，提升了分析的统一性与效率。

**⚠️ 局限性**

局限性主要体现在：①仍未能对所有量化模型给出完全闭合的等价关系；②证明中涉及大量抽象工具，实际计算时可能不够直观；③对部分分离函数的构造较为复杂，缺乏更简洁的示例；④在实际量子实现中，所需的量子资源（如相位旋转、反射操作）对硬件要求较高。

---

## 749. Train Where the Quantized Model Goes: On-Policy Distillation for Low-Bit Reasoning

**arXiv ID:** 2609.26708 | [PDF](https://arxiv.org/pdf/2609.26708v1)

**作者:** Yuanteng Chen `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jian Cheng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在极低精度（<3位）量化后，结合量化感知蒸馏（QAD）与自我驱动的在策略蒸馏（OPD），实现了对长篇推理和代码生成能力的恢复，且保持了短答能力；

**💡 创新点**

创新点在于将教师监督从固定语料前缀迁移到量化后模型实际生成的轨迹上，并通过逆 KL 与任务验证奖励双重信号对生成过程进行细粒度引导；

**🔧 技术方法**

主要技术包括：量化感知蒸馏（QAD）、在策略蒸馏（OPD）、逆 KL 监督、任务验证器奖励、低位（2.79/1.88效位）权重量化、vLLM 推理；

**📊 数据集**

使用的数据集包括数学推理（GSM8K、MATH‑500、AMC23）、代码生成（HumanEval、MBPP、KodCode）、短答（QA9集合）等；

**📈 对比分析**

与仅用QAD、PTQ、QAT、QAD+OPD 等基线相比，OPD 在MATH‑500 上从 35% 提升至 70% BF16 保留率，在 HumanEval 从 41.5% 提升至 59.1%，在 QA9 上保持 92% 保留率；相较于继续教师强制的 QAD，OPD 在同等训练步数下提升了 42× 的效率；

**⚠️ 局限性**

局限性包括：仍未完全恢复 BF16 水平的推理精度；在极低精度（<1.88 位）下的稳定性仍有限；训练仍需数百步 GPU 计算，且主要针对通用大模型，缺乏针对更大规模或更复杂任务的验证。

---

## 750. Remote Matching: Exact-Cardinality Approximation and Tight UGC Hardness

**arXiv ID:** 2609.26671 | [PDF](https://arxiv.org/pdf/2609.26671v1)

**作者:** Arash Ahadi `[一作]` (Tehran Institute for Advanced Studies, Khatam University), Shayan Tayefeh `[通讯]` (Sharif University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究无约束和给定终点数的最大最小度数匹配（max–min metric T‑join）问题，给出最优的 3/2 近似下限（UGC 证据），并设计了基于层次剪切包装、树动态规划与精确卡度化的确定性算法，在所有可行密度下实现 ρ(p) 近似（p = k/n），在 k = 2n/3 时达到 3/2，k ≤ 6n/7 时为 7/2，整个区间内可获得 4 近似。

**💡 创新点**

① 用独立集间接下界证明 3/2 近似是最优的；② 在精确卡度问题上提出新的 ρ(p) 近似框架，兼顾所有密度；③ 通过层次剪切包装与树结构的精确卡度化实现了无约束问题下的 3/2 近似的最优性。

**🔧 技术方法**

层次剪切包装（laminar cut packing）、权重层次树表示、精确卡度化（exact‑cardinality rounding）、树动态规划、最小割/最小匹配对偶理论、独立集间接下界与唯一游戏假设（UGC）等。

**📊 数据集**

该工作为理论算法研究，无实验数据集；所有结果均基于理论分析与构造实例。

**📈 对比分析**

与之前的 O(log k) 随机近似、随机 O(1) 近似以及 3/2 近似相比，本文实现了在所有密度下的确定性 ρ(p) 近似；在极端密度下可达到 4 近似，在 k = 2n/3 时与 3/2 上界一致，证明了 3/2 的最优性；对 0<k≤6n/7 的情况给出了 7/2 近似，显著优于之前仅在 k≤n/3 时的常数近似。

**⚠️ 局限性**

仅适用于偶数终点数；对极大密度（k≥7n/8）仅能保证 4 近似；在极端密度下无法突破 4 近似；实现依赖于层次剪切包装的多项式时间算法，可能在实际大规模实例中存在效率瓶颈。

---

## 751. HARMONY: Hierarchical Agentic Reasoning for MONocular Image-to-Scene Synthesis

**arXiv ID:** 2609.26793 | [PDF](https://arxiv.org/pdf/2609.26793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 752. Label-Efficient Learning for Ground-Based Sky-Image Classification: A Benchmark of Transfer Learning, Active Learning, and Pseudo-Labeling on GCD

**arXiv ID:** 2609.26631 | [PDF](https://arxiv.org/pdf/2609.26631v1)

**作者:** Esther Bou Dagher `[一作]` (Universite Paris Dauphine-PSL), Boguslaw Zegarlinski `[通讯]` (Polish Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对地面云图分类的标签效率进行基准实验，比较监督迁移学习、主动学习和伪标签三种策略在Ground-based Cloud Dataset上的表现。

**💡 创新点**

提供了一个统一、可复现的标签效率基准，系统评估了现有学习策略在低标签预算下的效能，并给出诊断分析解释性能差异。

**🔧 技术方法**

采用冻结的ImageNet预训练ResNet50作为特征提取器，结合不确定性采样主动学习和阈值为0.95的高置信伪标签半监督学习。

**📊 数据集**

Ground-based Cloud Dataset (GCD)，包含10,000训练图和9,000测试图，七类天空类型。

**📈 对比分析**

在1%–40%标签预算下进行五次随机种子实验，监督迁移学习已达到大约94%准确率；主动学习和伪标签仅略有提升，宏F1提升约0.01–0.02，整体差异不大。

**⚠️ 局限性**

只使用单一冻结骨干网络且未进行数据增强；主动学习与伪标签采用的简单策略，缺乏更高级的采样或自监督方法；结果仅在GCD上验证，未推广到其他云图数据。

---

## 753. 4-Block Integer Programming is in FPT

**arXiv ID:** 2609.26746 | [PDF](https://arxiv.org/pdf/2609.26746v1)

**作者:** Martin Koutecký `[一作]` (Charles University), Koen Ligthart `[通讯]` (Eindhoven University of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了一种固定参数可行（FPT）算法，用于求解四块整数规划（4‑block IP）问题。

**💡 创新点**

核心创新在于证明整数中点凸函数在有限维线性子空间上可扩展为凸函数，从而突破了先前只能处理单一全局变量的限制。

**🔧 技术方法**

结合凸可扩展性、整数分解性质（IDP）与块结构化矩阵的稀疏性，利用 GPT‑6‑Astra 生成关键结构性引理，构建完整的算法框架。

**📊 数据集**

该工作为理论性质，未使用任何实验数据集；所有结果均来自数学证明与算法复杂度分析。

**📈 对比分析**

与以往的 XP‑级别算法相比，该方法在参数 k、Δ 下实现了时间 2^(kΔ)^{O(k^2)}·n·polylog(n)·polylog(D)，几乎与已知的双指数下界相匹配，表现出显著的理论优势。

**⚠️ 局限性**

局限性包括：仍需对参数 k、Δ 进行指数级依赖，无法进一步减小对大系数或更一般块结构的限制；此外，算法的实现复杂度较高，尚未转化为可实用的求解器。

---

## 754. StableVQ: Practical Guidelines for Stable Vector-Quantized Tokenizer Training

**arXiv ID:** 2609.26774 | [PDF](https://arxiv.org/pdf/2609.26774v1)

**作者:** Bao Tang `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 StableVQ 方法，通过分离编码器–解码器与代码表的训练目标，改进向量量化的训练稳定性，并在 ImageNet 上实现高质量的视觉令牌化。

**💡 创新点**

创新点在于三项无参数干预：Dynamic STE 调节 STE 梯度以避免编码器崩溃；Region VQ Loss 为所有代码提供分布对齐的监督；Decoupled Schedule 为编码器和代码表设定独立学习率，彻底解决模块耦合导致的不稳定。

**🔧 技术方法**

技术手段包括共享投影代码表、动态加权 STE、基于最近邻传播的 Region VQ Loss、以及独立的学习率调度。

**📊 数据集**

实验采用 ImageNet 256×256 图像数据集，使用 VQGAN 风格的编码器–解码器网络进行训练。

**📈 对比分析**

与 SimVQ、FVQ 等共享投影基线相比，StableVQ 在保持 100% 代码利用率的同时，将 rFID 下降至 1.22，LPIPS 下降至 0.2235，并在 UR-AUC 以及生成 FID 上表现更优。

**⚠️ 局限性**

局限性在于实验仅覆盖 ImageNet，未检验在更大规模或其他领域数据上的泛化；此外仍需依赖共享投影结构，可能在极大代码表尺寸下面临计算瓶颈。

---

## 755. TM-APR: Thermal Temporal-Memory Localization via Analytic Online Adaptation

**arXiv ID:** 2609.26766 | [PDF](https://arxiv.org/pdf/2609.26766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 756. An Exponential Succinctness Gap between Three-Variable Logic and the Calculus of Relations

**arXiv ID:** 2609.26778 | [PDF](https://arxiv.org/pdf/2609.26778v1)

**作者:** Yuya Uezato `[一作]` `[通讯]` (CyberAgent, Inc.), Yuya Uezato (CyberAgent, Inc.)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明从三变量一阶逻辑(FO^3)到关系演算(CoR)的转换必然产生指数级大小增幅，即存在正号FO^3公式 φ，使得任何等价的CoR项或电路在有限结构上都至少需要 2^Ω(|φ|) 个组合门。

**💡 创新点**

创新点在于采用单一有限结构的保持性（preservation）论证，避免了以往依赖无穷结构的技术；同时将该保持性方法推广到大小特定、随机化电路以及矩阵查询语言 MATLANG，揭示了这些模型中也存在指数级紧凑性缺口。

**🔧 技术方法**

主要技术包括：构造具有细胞（cell）和色彩（color）属性的特殊有限结构；定义可检测颜色（detectable color）与支持数（support number）概念；利用集合对不等式和组合计数证明不可压缩性；在不同查询（Ψ_k、Φ_k）上通过类型常数（type‑constant）解释实现可检测性；以及对半环乘法、序列化和矩阵乘法等泛化核（kernel）进行支持数分析。

**📊 数据集**

由于研究纯理论性质，本文未使用任何实际数据集，所有结果均基于抽象结构和组合论推导。

**📈 对比分析**

与现有方法比较：该工作给出了与已知上界相匹配（除常数因子外）的下界，证实了FO^3→CoR转换的指数级不可避免性；在 MATLANG 和正号关系代数（ARA(3)）等语言中，同样证明了在矩阵维度压缩或属性数裁剪时必须至少使用 2^k/3（或 2^k）个乘法操作，表明了先前提出的指数级上界是最佳的。

**⚠️ 局限性**

局限性：在给定固定词汇表（signature）时，是否仍存在指数级紧凑性缺口尚未解决；此外，结果仅适用于类型常数电路/项，无法直接推广到能够访问单个输入条目的通用布尔电路；因子 2（或 1/3）在上界与下界之间的剩余差距仍未被完全消除。

---

## 757. SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue

**arXiv ID:** 2609.26780 | [PDF](https://arxiv.org/pdf/2609.26780v1)

**作者:** Haobo Zheng `[一作]` (Zhejiang University), Yingcai Wu `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双轨记忆系统，既保留带说话人标记的原文，又构建以人物级和群组级视角的结构化状态，用于多方长时会话的记忆与问答。

**💡 创新点**

创新点在于将原文轨与结构化轨并行存储，并通过“Anchor–Separate–Resolve–Compose”查询流程按实体、事件、时间精准组合证据；同时利用SpeakerLevenshtein和speaker‑conditioned GRPO训练局部可部署的写手，显著提升多方记忆中的归因与状态重构能力。

**🔧 技术方法**

核心技术包括：双轨（verbatim 与 derived）记忆结构、人物/群组分层（Core/Profile 与 Interaction/Insight）、基于实体/事件/时间的检索组合策略、RL写手训练（SpeakerLevenshtein + GRPO）以及在查询时的证据组织与融合。

**📊 数据集**

使用的数据集为 GroupMemBench、SocialMemBench、EverMemBench、LoCoMo（两人长会话边界测试）等三大多方对话基准。

**📈 对比分析**

与 BM25、dense retrieval、Mem0、A‑MEM、HippoRAG 等主流框架对比，双轨系统在三大基准上分别提升 3.3%、12.4% 和 9.4% 的二元准确率；在 EverMemBench 公开排行榜中达到 62.33%（最高），并在 LoCoMo 两人测试中取得 70.85% 的准确率；RL写手在 305 题评估中准确率从 57.38% 提升到 68.20%，接近 LLM 写手的 71.48%。

**⚠️ 局限性**

局限性包括：需要可靠的成员名单、说话人/所有者归因和时间标记；对别名、成员变动、隐式受众和并行事件仍处理不佳；构建与检索成本较高，跨域推理、开放域 QA 与知识外推尚待改进。

---

## 758. CliffCompaction: Cost-Efficient Compaction for Long-Horizon Coding Agents

**arXiv ID:** 2609.26779 | [PDF](https://arxiv.org/pdf/2609.26779v1)

**作者:** Trang Nguyen `[一作]` (Carnegie Mellon University), Tim Dettmers `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为CliffCompaction的自动压缩技术，用于在保持上下文完整性的同时减少LLM代理的上下文长度和推理成本。

**💡 创新点**

通过在达到阈值时仅保留最近两轮交互、保留工具调用的签名、丢弃旧压缩结果并每次重新压缩当前会话，从而实现高精度、低召回且不产生上下文漂移的压缩机制。

**🔧 技术方法**

规则式上下文裁剪、工具调用签名化、固定阈值触发、KV-cache友好设计，以及与现有工具集成的API代理实现。

**📊 数据集**

在SWE-bench Verified、Terminal-Bench 2.0/2.1、KernelBench Level 3以及内部测试集上评估。

**📈 对比分析**

与滑动窗口、摘要、微压缩等方法对比，CliffCompaction在保持或提升成功率的同时将成本降低约30–50%，在KernelBench实现3.58×速度提升、在终端任务获得10点以上的性能收益。

**⚠️ 局限性**

对不同 scaffold 的依赖、阈值设定需手工调优、仅适用于中长任务，未与训练型上下文管理或外部记忆方法对比。

---

## 759. A2M: Trace-Optimized Agent Hijacking in the MCP Ecosystem

**arXiv ID:** 2609.26761 | [PDF](https://arxiv.org/pdf/2609.26761v1)

**作者:** Laizhen Li `[一作]` (Shenzhen Institutes of Advanced Technology Chinese Academy of Sciences), Xitong Gao `[通讯]` (Shenzhen Institutes of Advanced Technology Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对Model Context Protocol（MCP）生态的黑盒攻击框架A2M，能够通过对工具元数据和返回内容的两阶段优化，诱导智能代理调用恶意工具并执行攻击目标。

**💡 创新点**

创新点在于将工具选择（吸引）与工具返回（操纵）拆分为独立的优化阶段，并引入Analyzer–Optimizer循环利用执行轨迹反馈，实现对MCP工具链的系统性攻击。

**🔧 技术方法**

使用大语言模型（如GLM‑4.6、Qwen3‑Max、DeepSeek‑V3.1、Kimi‑K2‑0905、GPT‑5）生成工具元数据和payload，采用Monte‑Carlo回放、LLM驱动的遗传算法以及自定义评估函数进行黑盒优化。

**📊 数据集**

评估数据集为LiveMCPBench（95个任务、70个服务器、527个工具），涵盖Office、Lifestyle、Leisure、Finance、Travel、Shopping六大领域。

**📈 对比分析**

与零样本生成、LLM‑GA、AMA、MPMA等基线比较，A2M在GLM‑4.6上平均恶意工具调用率93.6%、C‑DoS成本比32.4×、攻击成功率74.4%；在未重新优化的四个模型上平均调用率63.6%、成本比2.7×、成功率24.5%，明显优于其他方法。

**⚠️ 局限性**

局限性包括仅针对MCP层面的语义攻击，未覆盖网络/操作系统级别漏洞；依赖完整的执行轨迹信息，受限于速率限制和环境隐私；实验仅在ReAct‑style代理和LiveMCPBench上验证，未系统探讨系统提示、路由策略、工具重命名等因素对攻击效果的影响。

---

## 760. φ-RIE: From Photorealistic Reconstruction to Interactive Environments

**arXiv ID:** 2609.26795 | [PDF](https://arxiv.org/pdf/2609.26795v1)

**作者:** Runyi Yang `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Danda Pani Paudel `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一个Gaussian原生的场景转换管道，将捕获的3D Gaussian场景中选定对象转换为可交互的模拟器资产，同时完成背景补全。

**💡 创新点**

通过共享对象证据实现资产生成、源Gaussian移除和背景补全的耦合，从而在保持原始渲染的同时提供可物理交互的对象。

**🔧 技术方法**

使用3D Gaussian Splatting、SAM3掩码、TRELLIS与ReconViaGen几何生成、ICP注册、CoACD碰撞剖分、MuJoCo/PyBullet仿真以及Diffusion Harmonizer后处理。

**📊 数据集**

在ScanNet++ 50个场景、LIBERO、RoboCasa等数据集上进行评估。

**📈 对比分析**

与单生成器TRELLIS、固定优先策略等基线对比，保留率相同的情况下F1提升至0.383，Chamfer距离下降25%；在RoboCasa任务中成功率从73/480提升至128/480，取得约11个百分点的显著提升。

**⚠️ 局限性**

仅适用于刚体对象、需要局部平面补全，无法处理关节、可变形、复杂隐藏几何或真实机器人物理转移；视觉一致性仍需后期和谐处理。

---

## 761. DreamStream: Towards Policy-Oriented Generative Simulation for End-to-End Driving

**arXiv ID:** 2609.26792 | [PDF](https://arxiv.org/pdf/2609.26792v1)

**作者:** Ziyang Leng `[一作]` (University of California), Bolei Zhou `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种将物理仿真器与自回归视频模型相结合的生成式闭环仿真平台，能够在保持交通场景可控性的同时生成与真实相机观测高度一致的高质量图像；

**💡 创新点**

创新点包括①基于交通布局引导的教师-学生蒸馏技术，保证视频生成与仿真器状态的精确对齐；②多场景蒸馏策略，保留预训练视频模型的天气与光照多样性；③引入基于端到端驾驶策略的场景上下文特征的Fréchet距离度量，能够更真实地评估政策对视觉信息的依赖；④构建了包含对抗行为和天气变化的闭环基准Navhard-Base/AdvBehavior/AdvWeather，揭示了传统基准忽略的评估缺口。

**🔧 技术方法**

技术栈包括：物理仿真器（Sim）、自回归视频扩散模型（基于Wan2.1/2.2）、交通布局引导与多场景蒸馏、基于CLIP的文本提示、策略感知Fréchet距离度量、三阶段教师-学生蒸馏（Stage‑1/2/3）、KV缓存机制以降低长期漂移。

**📊 数据集**

使用公开的驾驶数据集nuScenes、NAVSIM（NavHard）以及通过日志重放得到的真实驾驶场景；同时在仿真中合成多种天气与光照的合成视频用于蒸馏。

**📈 对比分析**

与MagicDrive、Panacea、Dreamland、DriveArena、DreamForge、HUGSIM等现有闭环仿真器对比，所提方法在nuScenes验证集上Fréchet距离从7.27降低1.6×，在NAVSIM测试集上提升4.7×；同时在NavHard基准中获得更高的闭环驾驶分数，表明视觉对齐更好、对抗行为与天气更具鲁棒性。

**⚠️ 局限性**

局限性包括：仍在仿真环境中评估，未验证真实车载部署；自回归视频模型在极长滚动中存在漂移；蒸馏和训练需要大量GPU资源（约200个A100 GPU天），并且对特定硬件与模型规模敏感。

---

## 762. SWE-Serve: Benchmarking Agentic Engineering For Production Inference Serving

**arXiv ID:** 2609.26777 | [PDF](https://arxiv.org/pdf/2609.26777v1)

**作者:** Jennifer Williams `[一作]` (NVIDIA), Jiantao Jiao `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布SWE-Serve基准，用于在容器化环境中评估LLM代理在真实生产推理系统（SGLang）上的仓库级推理工程任务完成情况。

**💡 创新点**

创新点包括：① 以真实生产合并的Pull Request为基础构造53个可执行的推理工程任务；② 在任务中加入隐藏的功能、回归、端到端（E2E）与性能门限测试，能够量化本地实现与生产正确性之间的差距；③ 采用可逆轨迹审计、对抗性探测和闭书评估等手段，确保任务真实性与评估完整性。

**🔧 技术方法**

技术手段主要包括：Harbor容器化评估框架、mini‑SWE‑agent通用抓手、闭书执行与轨迹审计、代理辅助对抗性探测、可执行的Oracle与no‑op控制，以及E2E与性能门限的自动化验证。

**📊 数据集**

数据集来源于SGLang自2025年12月以来合并的Pull Request变更，覆盖六大推理工程领域（模型支持、解码、分布式执行、API等），共构造53个任务。

**📈 对比分析**

比较方法：对11种前沿LLM（如Claude Opus 5、GPT‑5.6 Sol、Gemini 3.6 Flash等）在不同推理努力配置下运行，评估指标包括pass@1、pass@3、pass@0/1/3、成本、token用量与推理时间。最佳配置平均pass@1为75%，不同模型间表现差距达40个百分点；去除E2E测试后，平均pass@1提升23.4个百分点，凸显生产正确性缺口。

**⚠️ 局限性**

局限性：① 仅在单CPU或单H100 GPU上评估，未涵盖多GPU/多节点并行任务；② 评估仅关注功能性与性能门限，无法衡量可维护性、架构兼容性等；③ 封闭评估无法完全排除代理事先接触公开SGLang代码的风险；④ 任务规模与复杂度受Oracle与no‑op控制的可执行性限制。

---

## 763. SARA: SLO-Aware Resource Allocation for Disaggregated Agentic LLM Services

**arXiv ID:** 2609.26763 | [PDF](https://arxiv.org/pdf/2609.26763v1)

**作者:** Shicong Liu `[一作]` (City University of Hong Kong), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出SARA框架，基于排队论对分离式LLM推理的预填、KV缓存传输与解码阶段进行建模，并在满足SLO阈值与成本预算的前提下实现资源分配优化。

**💡 创新点**

创新点在于：①使用M/G/k、M/G/1和出生-死亡模型实现三阶段延迟的可解析分位数估计；②针对轻尾与重尾工作负载给出闭式资源需求；③将SLO约束转化为可求解的硬件配置边界，并提出高效的二分搜索资源分配算法。

**🔧 技术方法**

技术手段包括排队论近似（Kingman、Heavy‑Traffic）、矩匹配法求解TPOT分位数、闭式阈值推导与线性/非线性约束的二分搜索、以及在A100 GPU上通过SGLang实现实验。

**📊 数据集**

使用的工作负载数据集包括三种LLM模型（LLaMA 3.1 8B、Qwen2.5 32B、GPT‑3 175B）的参数统计，以及公开的Microsoft Azure七天LLM推理流量轨迹。

**📈 对比分析**

与基线（穷举搜索、经验分配、Splitwise、Colocated）对比，SARA在相同成本下平均提升26.6%好通过率，重尾场景提升更显著，且与穷举搜索的性能差距≤3.8%，相比基线损失可达45%。

**⚠️ 局限性**

局限性包括：对KV缓存大小、网络传输时延的理想化假设；仅在A100/HBM设置下验证，可能难以直接迁移到低端或异构硬件；重尾分布近似可能不足以捕捉极端长上下文场景的真实尾部行为。

---

## 764. Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs

**arXiv ID:** 2609.26796 | [PDF](https://arxiv.org/pdf/2609.26796v1)

**作者:** Quan Nguyen-Tri `[一作]` (MBZUAI), Zhiqiang Shen `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Flash-dLLM，针对扩散式大语言模型的无训练推理加速框架。

**💡 创新点**

创新点包括 IO‑aware fused KV‑cache 核、按块调度的 Flash Attention、仅更新最重要 token 的 Selective cache update，以及利用 dLLM 自身做 draft 与 verify 的 KV‑cache 驱动的并行解码 Flash‑Verify。

**🔧 技术方法**

采用 Triton 编写的融合 KV‑cache 与注意力核，分块调度与块表管理，token 重要性筛选，双视角（draft 与 mask）注意力掩码实现自验证。

**📊 数据集**

使用 LLaDA‑1.5 在 GSM8K、MATH、HumanEval、MBPP 四个基准集上进行实验。

**📈 对比分析**

与 No Cache、Fast‑dLLM、Elastic‑Cache、FreeDave 等基线对比，Flash‑Cache/Flash‑Verify 在多种长度下实现 8–148 倍速度提升、显著的内存节省，并能在 32‑张量批量下线性扩展。

**⚠️ 局限性**

局限性包括：在较短长度代码生成任务中仍存在准确性-吞吐量权衡；需要手动调节阈值 ϵ、γ；仅针对扩散式模型，未验证在更大规模模型或其它任务上的通用性。

---

## 765. Polylogarithmic Collective Tree Exploration

**arXiv ID:** 2609.26789 | [PDF](https://arxiv.org/pdf/2609.26789v1)

**作者:** Romain Cosson `[一作]` (New York University), Laurent Massoulié `[通讯]` (Inria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种在无通信限制下的异步集体树探索算法。

**💡 创新点**

创新点在于使用多尺度幂正则化（ε≈1/ln k）而非熵正则化，实现了最佳的O(log² k)调节。

**🔧 技术方法**

采用连续树采矿游戏、Kirchhoff 定律和电阻网络分析，结合多尺度正则化与修复规则完成算法设计。

**📊 数据集**

未使用任何实验数据集，全部在理论证明与复杂度分析上完成。

**📈 对比分析**

与已知下界对比，得到 2n+O(k log²k D) 步的上界，并通过经典映射得到同步模式下的 2n/k+O(log²k D) 回合，显著优于之前的 O(log³k) 竞争比。

**⚠️ 局限性**

局限于树结构、需要全局通信、算法实现复杂，且未扩展到一般图的探索问题。

---

## 766. Agensh: Scaling Organizational Intelligence to 1,024 Agents

**arXiv ID:** 2609.26781 | [PDF](https://arxiv.org/pdf/2609.26781v1)

**作者:** Zhihao Zhan `[一作]` (Microsoft Research), Furu Wei `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个无中心化、自组织的多智能体组织框架Agensh，并在ProgramBench最难的五个任务上验证其可扩展性。

**💡 创新点**

创新点在于：①去中心化的自组织工作循环；②共享工作区、消息接口和共享上下文三大基础设施；③通过代理自身发现、声明并分配子任务，实现数百甚至上千个代理并发协作；④展示代理数量是多智能体组织的新扩展维度。

**🔧 技术方法**

技术包括：GPT‑5.6‑sol（high）作为单代理基础；单代理 harness；Gitea实现共享工作区；Mattermost实现消息接口；DeLM‑style共享上下文工具；异步协作循环与工作流程模板。

**📊 数据集**

使用的主要数据集是ProgramBench中最难的五个任务（FFmpeg、Gromacs、Pandoc、PHP、Universal CTags），在6 h无网络环境下进行评测。

**📈 对比分析**

通过与单代理基准（1人）对比，规模从1→128时平均测试通过率提升约49%（从19.31%提升至28.78%），1→1024时通过率提升至55.06%；更大规模的组织能更早达到相同得分，证明代理数量是关键的性能提升维度。

**⚠️ 局限性**

局限性包括：仍依赖大模型和单代理 harness；在真实世界多任务或实时交互中的表现未充分验证；并发冲突与资源争用可能导致效率下降；高并发需要大量硬件资源，成本较高。

---

