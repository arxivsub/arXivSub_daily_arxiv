# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-21 | 今日论文总数: 423

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. A Strong Linear Baseline for Whole-Heart Cardiac Shape Completion on CT, with an Open Eleven-Structure Statistical Shape Model

**arXiv ID:** 2608.19932 | [PDF](https://arxiv.org/pdf/2608.19932v1)

**作者:** Matej Gazda `[一作]` (Technical University of Kosice), Peter Drotar `[通讯]` (Technical University of Kosice)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文构建了一个包含左、右心室、心房、心肌、主动脉、肺动脉、左心耳、肺静脉、上腔静脉、下腔静脉共十一块的CT统计形状模型（SSM），并在此模型上定义并评估了形状补全（completion）方法，比较了线性条件高斯闭式估计、图卷积VAE、PPCA混合等多种模型。

**💡 创新点**

创新点：①发布完整十一结构、共享顶点对应的开放式CT SSM；②提出可针对任意观察块子集进行条件补全的闭式条件高斯基线；③建立统一评估基准，比较线性与非线性模型，并发现闭式条件高斯在多数情形下优于深度网络；④提供完整的数据集、代码与评估脚本，促进跨数据集统一研究。

**🔧 技术方法**

技术手段：统计形状建模（PCA、PPCA）、条件高斯后验推断、图卷积自编码器（CoMA）β‑VAE、ANTs SyN 变形配准、TotalSegmentator 2.13.0 自动分割、拉普拉斯平滑、稀疏修复、非线性混合模型与最近邻基线。

**📊 数据集**

数据集：内部使用 631 张 TotalSegmentator 2.13.0 生成的银标签病例（最终 383 张可用），外部验证使用 CARE2026（58 张，含 7 结构手工标注）与 MM‑WHS 2017（20 张，含 7 结构手工标注）。

**📈 对比分析**

比较方式与性能：在冻结的 76 个内部测试样本上，按 k=1,3,5,9 四种观察块数，使用平均每顶点欧氏误差（MPVED）作为指标。条件高斯闭式平均误差为 3.717 mm，图卷积β‑VAE 5.248 mm，差异 1.531 mm（p<0.001）。在外部验证，Cond‑G 在可评价结构上均获得更低的 ASSD、HD95 与 Chamfer RMS，进一步验证其优越性。

**⚠️ 局限性**

局限性：①内部评估使用与训练相同来源的 76 样本，非独立；②基准误差以银标签与注册结果为参照，缺乏真实解剖对比；③外部验证缺少完整的专家标注，仅可评估部分结构；④未分离心脏相位与病理差异；⑤深度模型未进行完全的超参搜索与多种网络架构比较；⑥误差评估未考虑自交、拓扑不连通等临床重要问题；⑦不确定性估计未校准。

---

## 2. AEGIS: Attention-Embedding Gradient Isolation Shield - Triple-Channel Gradient Masking for Privacy-Preserving Federated LLM Fine-Tuning

**arXiv ID:** 2608.19534 | [PDF](https://arxiv.org/pdf/2608.19534v1)

**作者:** Ye Tao `[一作]` (Central Queensland University), Can Wang `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在联邦学习下对大语言模型进行微调时，提出了三通道梯度掩蔽机制AEGIS，关闭了注意力投影梯度、嵌入梯度稀疏性以及MLP扩展梯度三条信息泄漏通道，防止梯度反演攻击；

**💡 创新点**

创新点在于系统性识别并同时封闭梯度的三条解析通道，并通过无架构改动的冻结、加噪填充等轻量级后向路径操作实现完整防护；

**🔧 技术方法**

采用注意力参数冻结、嵌入梯度均匀加噪（Gaussian uniformization）与MLP扩展梯度均匀加噪，以及梯度裁剪与噪声注入等技术；

**📊 数据集**

使用六个基准数据集（Rotten Tomatoes、Emotion、Financial PhraseBank、WikiText‑2、DialogSum、CNN/DailyMail）在十一款不同规模模型（从124M到13B）上进行实验；

**📈 对比分析**

与DP‑SGD、梯度裁剪、Soteria等防御方法对比，AEGIS在保持或提升模型性能（PPL/准确率不变甚至提升）的同时，使DAGER等闭式反演攻击的ROUGE‑1降至≤0.005，几乎完全消除泄漏；

**⚠️ 局限性**

局限性包括：未提供正式的（ε,δ）DP保证；仅在单步FedSGD场景下验证，未评估多步FedAvg、跨轮累积等；对优化型梯度反演攻击和Robust‑PCA/随机矩阵去噪仍有潜在弱点；大词表模型中嵌入梯度密集化导致通信/存储开销上升。

---

## 3. SSR-GRPO: Integrating Supervision and Semantic IDs into Reinforcement Learning for Dense Retrieval in E-commerce

**arXiv ID:** 2608.19595 | [PDF](https://arxiv.org/pdf/2608.19595v1)

**作者:** Guangxin Song `[一作]` (Alibaba Group), Jianbo Zhu `[通讯]` (Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一种结合密集与稀疏两视角奖励的强化学习框架 SSR-GRPO，用于电商商品的高效稠密检索。

**💡 创新点**

创新点包括：① 使用语义标识符 (SID) 与多层量化模型生成的稀疏分级匹配得分，替代高参数 LLM 进行奖励评估；② 通过 SID 层级关系挖掘难负样本，并将其用于 R-DPO 直接偏好优化与 R-GRPO 的噪声屏蔽；③ 引入动态不确定性加权，平衡 RL 与监督梯度；④ 结合掩码函数剔除 top‑K 里易负样本，提高训练稳定性。

**🔧 技术方法**

采用的技术包括：双塔密集编码器、残差量化 VAE (RQ‑VAE) 生成 SID、LLM 的 next‑token 生成查询 SID、InfoNCE 对比学习、GRPO 强化学习、DPO 直接偏好损失、掩码函数与动态不确定性加权。

**📊 数据集**

数据集为阿里巴巴天猫 0.3 B 次点击‑购买日志，用于 SFT 训练；随后选取最近 15 天交互数据用于 SSR‑GRPO 训练；离线评测使用 General Retrieval 与 Long‑Tail 两个测试集，在线评测在天猫 APP 的 A/B 测试。

**📈 对比分析**

与多种基线（StructBERT、Qwen3、Tbstars‑3B 及其 SFT、R‑GRPO 等）对比，SSR‑GRPO 在 HR@4k 和 GR@100 上均领先 0.6%–0.9%，在线业务指标 UCTCVR、GMV、Ex‑Imp、Goodrate 亦分别提升约 0.6%、1.4%、0.5% 与 0.6%。

**⚠️ 局限性**

局限性包括：① 仍需先进行 SFT 预训练，耗费大量算力；② SID 生成与量化模型训练复杂，推理时需额外算力；③ RL 训练方差仍高，需动态权重调节；④ 在极长尾或多模态查询场景下，SID 的分级匹配可能不足以覆盖所有细粒度语义。

---

## 4. TempJail: Temporal Jailbreak Attack against Large Vision-Language Models via Subtitle Scheduling

**arXiv ID:** 2608.19737 | [PDF](https://arxiv.org/pdf/2608.19737v1)

**作者:** Ling Zhou `[一作]` (University of Electronic Science and Technology of China), Shijie Zhou `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于字幕时序优化的黑盒视频 Jailbreak 框架 TempJail，先将有害查询转化为对话式字幕序列，再生成与字幕内容匹配的背景视频，最后用 CMA‑ES 对字幕的显示时长进行全局优化，以诱使大型视觉‑语言模型给出有害答案。

**💡 创新点**

创新点在于把字幕的**时间分布**视为新的攻击维度——发现并利用视频中字幕的时序结构可以显著提升攻击成功率；同时将多轮对话生成与字幕时序优化结合，形成完整的端到端攻击流程。

**🔧 技术方法**

使用了多轮对话生成（基于替代模型的对话分解）、文本‑到‑视频生成（如 Runway Gen‑4.5）来构造背景视频、以及 CMA‑ES 黑盒优化器对字幕时序进行全局搜索。

**📊 数据集**

实验基于 HADES 与 VLJailbreakBench 两个多模态安全数据集（各 50 条样本）。

**📈 对比分析**

与四个代表性视频 Jailbreak 基线（FigStep、VideoJail、SPTV、MCV）在四个主流 LVLM（Qwen3‑VL‑Plus、Qwen3‑VL‑32B‑Instruct、GPT‑5、Gemini 3.5‑Flash）上的对比，TempJail 在所有模型‑数据集组合中均取得最高攻击成功率（平均 ASR 89%–90%），相比 GPT‑5 上的 20% 提升至 70%+，在 Gemini 上从 16% 提升至 90%+，且在 Qwen3 系列模型中接近 100%。

**⚠️ 局限性**

局限性包括：对背景视频生成模型的依赖可能限制攻击的通用性；在 GPT‑5 上语义场景生成效果不佳，提升有限；CMA‑ES 优化需要多次采样，计算成本相对较高；目前主要验证了静态视频攻击，对实时或多样化视频场景的适应性尚待研究。

---

## 5. Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts

**arXiv ID:** 2608.20061 | [PDF](https://arxiv.org/pdf/2608.20061v1)

**作者:** Nayeon Kim `[一作]` (Kakao Corp), Boseop Kim `[通讯]` (Kakao Corp)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种两步计算高效的超参数迁移框架，利用最大更新参数化（μP）在MoE宽度扩展时实现学习率零样本迁移，并通过线性缩放法在令牌预算上进行学习率外推，以实现大规模MoE预训练的学习率调优；

**💡 创新点**

创新点在于将μP迁移方法扩展到MoE稀疏度维度的宽度扩展，并结合基于令牌预算的线性学习率缩放法，实现从小规模代理模型到10万亿令牌全规模训练的学习率精准预测；

**🔧 技术方法**

主要技术包括Muon优化器、Multi‑Head Latent Attention（MLA）MoE架构、Maximal Update Parameterization（μP）参数化、EMA权重平均、WSD学习率调度、二次多项式拟合和对数回归；

**📊 数据集**

使用多语言混合语料库（约45%英语，12.5% STEM，27.5% 代码，15% 多语种），训练总计10万亿个token的代理和全规模MoE模型；

**📈 对比分析**

通过在代理模型和宽度扩展的目标模型上对学习率进行二次拟合，验证μP在宽度上的迁移性，并用线性回归在log‑log空间外推到10T令牌；实验表明预测的学习率3.85×10⁻⁴在全规模训练中保持损失稳定，并在多项基准（MMLU‑Pro、Math、Code等）上位于Pareto前沿；

**⚠️ 局限性**

局限包括：未对不同MoE结构或优化器进行广泛验证，稀疏度维度的单独影响未被分离，未探索每个专家的学习率自适应，且对极端稀疏率在代理规模下的算力效率仍待评估。

---

## 6. Longitudinal Bayesian Learning of Continuous Disease Position across the Alzheimer's Disease Continuum

**arXiv ID:** 2608.19436 | [PDF](https://arxiv.org/pdf/2608.19436v1)

**作者:** Yingying Zhang `[一作]` (University of Texas Rio Grande Valley), Haoteng Tang `[通讯]` (University of Texas Rio Grande Valley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于纵向扩散张量成像（DTI）的贝叶斯学习框架 Disease Continuum Positioning（DCP），从中连续估计阿尔茨海默病的疾病严重度并给出不确定性；

**💡 创新点**

创新点在于将疾病严重度建模为概率潜变量，结合弱临床监督（诊断层级）与纵向一致性约束，得到可解释的连续疾病分数 Disease Continuum Score（DCS），并提供置信区间；

**🔧 技术方法**

使用三维 ResNet‑18 编码器提取低维潜变量，贝叶斯推断（KL 正则、排名损失、单调性损失）以及重构损失；

**📊 数据集**

采用阿尔茨海默病神经影像倡议（ADNI）数据库的多时间点 DTI（FA、MD、RD、AD）数据，共 695 受试者 2036 影像；

**📈 对比分析**

与四种基线方法（SuStaIn、DLMRI、LSSL、MedicalNet）在 Spearman‑ρ、ANOVA F、Stage C、Long‑C、AUC 等指标上比较，DCP 在所有指标上均排名第一，尤其在预测临床转换（AUC 0.725/0.796）方面优于其它方法；

**⚠️ 局限性**

局限性主要是依赖纵向随访数据的可用性，受访者随访缺失和数量有限的影响，未来需更大规模完整的纵向队列来进一步提升性能。

---

## 7. Linguistic Holonomy and Statistical Watermarks: Inner Geometry of Meaning-Preserving Transformations

**arXiv ID:** 2608.19369 | [PDF](https://arxiv.org/pdf/2608.19369v1)

**作者:** Daniele Corradetti `[一作]` `[通讯]` (Instituto Superior Técnico), Daniele Corradetti (Instituto Superior Técnico)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过“语言循环”(linguistic loop) 的几何框架，研究并量化语言模型水印在编辑（尤其是多语言回译）过程中的消失机制。

**💡 创新点**

创新点包括：①证明水印的残留统计量仅与编辑窗口完整性相关，即“完整窗口定律”(intact‑window law)；②将语义保持链的旋转矩阵分解为终点旋转与霍洛尼(Holonomy)，并指出终点语义差异与霍洛尼无关；③用霍洛尼能量(holonomy energy)作为链路径依赖性的度量，验证其与水印残留的负相关；④首次将Wilson环与自然语言的平行传输联系起来。

**🔧 技术方法**

使用的技术主要包括：几何与矩阵分析（旋转矩阵、Sylvester 符号、平行传输）、统计理论（随机指纹、独立编辑模型、期望计算）以及深度学习模型（0.5B 指令调优语言模型、Opus‑MT 翻译模型、384 维多语句子编码器）。

**📊 数据集**

数据集：①400 词长的人工合成文本，采用 Zipf 词分布；②90 条不同领域的自然提示生成的 180 词文本，使用 0.5B 指令调优模型；③通过 Opus‑MT 进行 3‑步回译（德、法、西）形成的 6 条翻译链。所有文本均以公开的词表和随机种子生成，保证可复现。

**📈 对比分析**

比较方法：对每种水印方案（绿色列表、单字典、指数）计算在不同编辑率、不同编辑模式（随机、连续区块、周期性）下的残留 z‑score 或检测率。实验表明：
- 在独立编辑模型下，残留比例近似为 ρ^{h+1}（h 为上下文宽度）。
- 在同一编辑率下，编辑位置分布对残留有显著影响，极端时可降至零。 
- 通过回译链，平均残留 z‑score 与完整窗口比例高度相关（R≈0.7），而与语义偏差（δ）关系弱。 
- 绿色列表在多轮回译后仍保持高于阈值的检测率，而指数方案在长链中更易衰减。

**⚠️ 局限性**

限制：①完整窗口定律基于长度保持且独立编辑假设，实际文本中编辑往往集中、长度不变；②霍洛尼能量为损失的标量摘要，可能掩盖更细粒度信息；③仅使用单一多语句子编码器，跨模型泛化未验证；④实验规模受计算资源限制，未覆盖大规模生成模型与学习型改写器。

---

## 8. World-Model-Grounded LLM Planning for AUV and ASV Navigation Near Offshore Wind Farms

**arXiv ID:** 2608.19661 | [PDF](https://arxiv.org/pdf/2608.19661v1)

**作者:** Markus Buchholz `[一作]` (Norwegian Defence Research Establishment), Yvan R. Petillot `[通讯]` (Heriot-Watt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于世界模型的LLM规划框架，将大语言模型用于生成宏动作序列，物理感知由混合解析+残差神经网络模型负责，并通过三阶段梯度优化和MPC式重规划实现在海上风电设施附近的6-DOF AUV与3-DOF ASV的安全导航。

**💡 创新点**

创新点在于：①将解析物理模型与神经残差相结合，生成可微、可解释的世界模型；②统一的三阶段梯度优化器和信赖域保护的MPC闭环重规划可跨平台复用；③为ASV引入基于卫星图像、航图与天气预报的VLM语义映射，消除对昂贵传感器的依赖。

**🔧 技术方法**

使用的技术包括：Fossen动力学解析求解器+MLP残差网络、三阶段梯度优化（尺度搜索、坐标下降、Adam）、MPC风格重规划与信赖域守门、LLM（Ollama）宏动作生成、VLM（Vision‑Language Model）对卫星/航图图像进行格点分类，并将结果写入知识图谱。

**📊 数据集**

使用的数据集包括：从随机平滑/阶跃指令、稳定控制器轨迹以及宏动作序列生成的训练转换；GazeboSim的高保真模拟数据；卫星影像、挪威航图、yr.no风预报与Copernicus海洋服务的海流/波浪数据。

**📈 对比分析**

与无世界模型的原始LLM直接执行基线相比，梯度+MPC方案在模拟与GazeboSim两种环境下均实现100%目标到达且零碰撞；在GazeboSim中误差下降70–82%（ASV）和约93%（AUV），同时在开放式任务中平均距离显著小于基线，展示出显著的性能提升。

**⚠️ 局限性**

主要限制包括：①优化器只能在固定的宏动作拓扑内调整持续时间，无法修复LLM生成的错误拓扑；②LLM的作用范围受限于仅产生宏动作序列，缺乏对时间/风险等语义权衡的直接推理；③实验仍停留在仿真层面，未在真实海域进行实测验证。

---

## 9. Question-Guided Evidence Acquisition for Multimodal Visual Question Answering

**arXiv ID:** 2608.19739 | [PDF](https://arxiv.org/pdf/2608.19739v1)

**作者:** Alin-Ionut Popa `[一作]` `[通讯]` (Amazon), Alin-Ionut Popa (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Q-Guide，一种基于多模态 LLM 的循环式证据获取框架，通过针对性工具动态获取问题所需的视觉和文本证据；

**💡 创新点**

创新点在于将感知过程转为问题条件化的慢思考循环，直接利用工具回溯获取缺失证据，而非依赖单次全局编码或复杂多代理规划；

**🔧 技术方法**

采用 Claude Opus 4.6 等大型多模态 LLM 作为策略，配合 Textract、Qwen3.5-2B 等 OCR、结构恢复与区域查询工具，使用 LangGraph 实现状态机；

**📊 数据集**

在 DocVQA2026（覆盖多种文档类型、跨页推理）和 Manga109 的角色命名任务 M109NC 上进行评估；

**📈 对比分析**

与直接提示（单次 LLM 调用）、视觉+OCR 提示及 ARIAL、DocAgent、MDocAgent 等多代理基线对比，Q-Guide 在 DocVQA 上实现 65.0%（相对最高基线提升 26.2 点），在 M109NC 上实现 53.7%（相对最高基线提升 28.8 点），且对不同 LLM 体系结构均保持优势；

**⚠️ 局限性**

局限在于对地图类问题的拓扑推理不足、长文档跨页整合仍需改进，以及依赖外部 OCR/结构工具的误差会直接影响性能。

---

## 10. Write Once, Run Everywhere: The Axon DSL for Shape-Safe and Framework-Agnostic LLM Architectures

**arXiv ID:** 2608.19889 | [PDF](https://arxiv.org/pdf/2608.19889v1)

**作者:** Jacob Nielsen `[一作]` (University of Southern Denmark), Peter Schneider-Kamp `[通讯]` (University of Southern Denmark)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Axon 这一强类型函数式 DSL，允许一次性编写 LLM 模型定义并通过统一 Graph IR 自动生成可在 PyTorch、JAX、MLX、vLLM 等后端运行的独立实现；

**💡 创新点**

通过统一的模型定义语言与单一 IR 解决多后端迁移导致的实现漂移问题，实现写一次、跑多处的能力，同时提供符号维度、路径解析等形状安全机制；

**🔧 技术方法**

使用强类型 DSL、静态单赋值（SSA）图 IR、图优化（内联、重写、后端特化）、后端特定代码生成（PyTorch、JAX、MLX、Triton、vLLM）、符号维度和路径解析等技术；

**📊 数据集**

评测涵盖 204+ 检查点，覆盖 GPT、Llama、Mistral、Qwen、Gemma、SmolLM、T5、BART、BERT、Mamba 等 60+ 模型族，模型规模从 135M 到 32B；训练实验使用 ArXiv Summarization 数据集；

**📈 对比分析**

对比 Transformers 基线，使用相同输入、精度和编译设置，测量推理（生成）和前向吞吐/延迟，计算时间比；结果显示 Axon 在多数后端实现 0.6–0.9× 的速度提升，约 70% 检查点超越基线；在 MLX 和 vLLM 上表现尤为突出；

**⚠️ 局限性**

仅针对语言模型实验，未覆盖其他任务；实验仅在单一 GPU/硬件配置，缺乏跨硬件评估；大模型 encoder‑decoder 前向性能仍偏慢；缺少对编译器优化各阶段效果的 ablation；FP32 回退需满足精度一致性；LLM 辅助生成 Axon 代码质量仍待提升。

---

## 11. An Evidence-Grounded Multi-Agent System for High-Level Bio-Robot Design

**arXiv ID:** 2608.19699 | [PDF](https://arxiv.org/pdf/2608.19699v1)

**作者:** Yujun Chen `[一作]`, Zhen Yin `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个离线多智能体系统 micro_biorobot_agent，用于将生物机器人需求转化为模块化设计报告，并追踪证据来源。

**💡 创新点**

创新点在于：共享工作空间（黑板）与多智能体协作；集成 23,762 条生物部件与证据库；基于规则的输出检查，纠正假缺口、未检索部件标注和源跟踪错误；以及针对设计完整性、兼容性等指标的系统化评估。

**🔧 技术方法**

使用了 Qwen3.5‑27B 语言模型、BM25 与 FAISS 向量检索、规则引擎和验证器、黑板架构以及离线多智能体调度。

**📊 数据集**

集成证据库（iGEM、FPbase、Kosuri、文献 DOI 等）共 23,762 条记录；评估集包括 50+50+50 个设计查询（Basic Design、Scenario Design、应用场景），以及 BAAI/bge-m3 语义嵌入。

**📈 对比分析**

通过对七种系统在两份 50 题评测集的平均得分进行比较，并辅以盲人类评估。micro_biorobot_agent 在 Basic Design 上平均得分 7.35，Scenario Design 上 8.04，均高于其他系统；源标签校正后错误缺口下降 80%，源准确率提升 0.75。

**⚠️ 局限性**

局限在于仅生成高层设计报告，未进行实验验证；零件库在安全/封闭模块上覆盖不足；缺乏运行方差和统计显著性分析；仅离线运行，未考虑实时部署。

---

## 12. HiTac-WAM: A Hierarchical Tactile World Action Model for Contact-Rich Robot Manipulation

**arXiv ID:** 2608.19574 | [PDF](https://arxiv.org/pdf/2608.19574v1)

**作者:** Chao Xue `[一作]` (Chinese Academy of Sciences), Shuo Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种层次化触觉世界动作模型HiTac‑WAM，能够在执行前为每个候选动作块预测接触状态、3D变形和滑移风险，并利用这些预测进行动作选择与在线验证。

**💡 创新点**

创新点在于将触觉预测拆分为有向层次结构——接触→条件变形→滑移风险，并通过停止梯度和定向注意力掩码将触觉预测与视频和动作上下文紧耦合，同时在执行阶段保留所选预测作为参考以触发补救。

**🔧 技术方法**

技术手段包括：预训练的世界动作模型与FG‑CLTP触觉编码器、三阶段预测头、停止梯度条件化、定向注意力掩码、加权损失（BCE、Huber、WBCE）、候选排序与任务进度估计，以及基于核密度估计的在线预测误差检测。

**📊 数据集**

实验数据来自IMETA‑Y1机器人，配备DM‑Tac W2触觉传感器和RGB相机，在芯片抓取、黑板擦除和USB插入三项任务上收集200条完整轨迹，划分为160/20/20的训练/验证/测试集。

**📈 对比分析**

通过与DreamZero（单候选）、Reactive Tactile（仅用实时触觉）以及仅基于任务进度的排名等基线比较，HiTac‑WAM实现平均接触F1 0.921、3D位移L2 0.058 mm、滑移AUPRC 0.247，且选择预测后成功率提升至61.1%，完整系统在三项任务上平均成功率达到72.2%。

**⚠️ 局限性**

局限性包括：未单独评估模型生成的动作候选的预测质量、滑移风险未做校准以用于在线报警、仅针对短时接触任务，缺乏不确定性校准与长期规划的自适应纠正策略。

---

## 13. Interaction valence reveals contrasting social networks in dairy cattle

**arXiv ID:** 2608.19222 | [PDF](https://arxiv.org/pdf/2608.19222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 14. Further Progress Towards Operation Proof Obligation Generation for VDM

**arXiv ID:** 2608.19848 | [PDF](https://arxiv.org/pdf/2608.19848v1)

**作者:** Nick Battle `[一作]` (Aarhus University), Peter Gorm Larsen `[通讯]` (Aarhus University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究改进了VDM操作的证明义务生成（POG），引入循环不变式与变体、递归测度以及更精确的状态更新与调用处理；

**💡 创新点**

创新点在于利用新注解实现循环终止证明、改进操作调用的状态闭包计算、支持递归测度与隐式操作定义，并通过CodeLens和QuickCheck提升可视化与自动化验证；

**🔧 技术方法**

技术手段包括VDMJ、QuickCheck、VDM-VSCode的CodeLens与InlayHint、抽象语法树分析、量化变量与幽灵变量、递归测度函数等；

**📊 数据集**

使用了VDM-SL工具自带的大规模规范语料库（约7000个证明义务）作为测试数据；

**📈 对比分析**

通过对比POG产生的“UNCHECKED”比例，旧版为9.6%，新版降至2.4%，并利用QuickCheck对约7000个义务进行自动化验证，发现并修正了多条未通过的案例；

**⚠️ 局限性**

局限性包括无法处理异常语句、变量隐藏导致的误义务、跨模块状态更新不完整、VDM++/VDM‑RT复杂控制流、以及无法静态确定多操作调用顺序的情形；

---

## 15. Quantization Beyond Uniform Bit Allocation

**arXiv ID:** 2608.19388 | [PDF](https://arxiv.org/pdf/2608.19388v1)

**作者:** K. S. Sreeramji `[一作]` (Indian Institute of Science), Yujia Wang `[通讯]` (Microsoft STCA)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了可变位分配方案，用于在固定存储预算下对具备Matryoshka属性的高维嵌入进行量化，

**💡 创新点**

创新点在于利用贪心位分配策略聚焦重要维度，显著提升了非均匀位分配的量化质量，

**🔧 技术方法**

采用Product Quantization（PQ）与Scalar Quantization（SQ）的可变位分配实现，并构建贪心分配框架，

**📊 数据集**

在OpenAI text-embedding-3-large和Cohere embed-v4嵌入上，对MS Marco、DBpedia-Entity、Quora、FiQA、SciDocs、SciFact等多种检索数据集进行实验，

**📈 对比分析**

通过与均匀位分配基线对比，使用100-recall@100评估，发现在低位预算下PQ提升约8%，SQ提升约18%，整体回调率有显著提升，

**⚠️ 局限性**

主要局限在于贪心算法计算开销大、缺乏对MRL嵌入的严格理论描述、系统实现需适配非统一数据类型以及需要更高效的分配策略。

---

## 16. Tri-Hybrid Beamforming for T-RIS-Enabled Base Station

**arXiv ID:** 2608.19736 | [PDF](https://arxiv.org/pdf/2608.19736v1)

**作者:** Hongtao Zhang `[一作]` (Beijing University of Posts and Telecommunications), Chenlong Ding `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于发射端可编程透射重构智能表面（T‑RIS）的三层混合 MIMO 系统，并给出从连续场到基带输入输出的统一模型，支持近场、远场及混合场传输。

**💡 创新点**

在传统 RIS 辅助链路难以建模的基础上，首次从物理光场角度构建 T‑RIS 前端模型，并在此基础上提出面向加权总速率（WSR）的三层协同预编码框架，包含任务感知的混合场修正。

**🔧 技术方法**

采用 Rayleigh‑Sommerfeld 近似、Fresnel/Fraunhofer 区分、连续-离散映射、WMMSE 外层预编码、内层硬件投影与梯度下降、Armijo backtracking、正则化最小二乘等技术。

**📊 数据集**

使用仿真数据：28 GHz 载波、32λ×32λ T‑RIS、4×4 天线、8 RF 链、8 用户，Monte‑Carlo 200 次实验。

**📈 对比分析**

与六种基线（无 RIS、仅场拟合、仅 T‑RIS、仅天线、通用 AO、提议方法）对比，提出方法在所有 SNR、用户数、混合场比例下均取得最高 WSR，并在复杂度上低于 AO。

**⚠️ 局限性**

主要局限包括硬件量化导致的性能瓶颈、混合场下的相互耦合抑制不完全、对大规模 T‑RIS 近场/远场切换的更精细建模需求。

---

## 17. Assembly Theory and the Smallest Grammar Problem

**arXiv ID:** 2608.19228 | [PDF](https://arxiv.org/pdf/2608.19228v1)

**作者:** Wawrzyniec Bieniawski `[一作]` `[通讯]` (Dom Zatorski Scientific Foundation), Wawrzyniec Bieniawski (Dom Zatorski Scientific Foundation)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估多种压缩算法对Assembly Theory（组装指数ASI）估计的有效性；

**💡 创新点**

提出Re-Pair T-NDR分支限界变体，显著提升ASI估计精度；

**🔧 技术方法**

使用语法基础压缩算法（Re-Pair、Sequitur、Sakamoto等）及LZ族；

**📊 数据集**

构造了408条合成字符串（包含最大复杂度、不同熵）和40条生物序列；

**📈 对比分析**

通过平均相对误差、Spearman相关、Pareto 前沿等指标比较，T‑NDR在大多数情形下误差低于10%，在高基数下可实现0%误差；

**⚠️ 局限性**

受限于ASI的NP‑完整性、字符串长度上限、仅线性字符串模型以及对几何/热力学约束缺失的考虑。

---

## 18. A Virtual Member of a Community of Practice for the Society of Petroleum Engineers: From Prototype to Deployment

**arXiv ID:** 2608.19199 | [PDF](https://arxiv.org/pdf/2608.19199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 19. PETA:Parameter-Efficient Test-Time Adaptation for Virtual Screening

**arXiv ID:** 2608.19906 | [PDF](https://arxiv.org/pdf/2608.19906v1)

**作者:** Jia-Qi Lin `[一作]` (Agency for Science, Technology and Research), Yuangang Pan `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种面向虚拟筛选的测试时自适应框架PETA，能够在不对整个模型进行重新训练的情况下，对预训练的Pocket-Ligand检索模型进行目标蛋白口袋的特定适配；

**💡 创新点**

创新点在于：①仅更新层归一化参数（仅约0.03%参数），实现极轻量化适配；②通过检索结构数据库获取参考配体并利用分子扩散生成口袋相关负样本；③利用化学无效性筛选硬负样本并对负样本与参考配体进行嵌入空间混合，形成更具挑战性的局部排名任务；④设计以硬负样本为重点的成本敏感ListNet排名损失；

**🔧 技术方法**

主要技术包括：预训练的DrugCLIP检索框架、DiffSBDD分子扩散生成器、RDKit化学有效性检查、Embedding Space Mixup、成本敏感ListNet排名目标、仅更新LayerNorm的参数高效自适应；

**📊 数据集**

实验数据集涵盖DUD-E、LIT-PCBA、以及四靶点的FEP+基准（CDK2、TYK2、JNK1、P38）以及结构数据库PDBbind用于检索参考配体；

**📈 对比分析**

与传统对接（AutoDock Vina、Glide-SP）、学习型基线（RF-Score、Pafnucy、OnionNet、PLANET、DrugCLIP、DrugHash、BindCLIP）以及Docking+学习混合方法相比，PETA在早期检索指标（BEDROC、EF）上均取得领先，并在FEP+精细排名任务中取得最高的pairwise accuracy和Kendall τ；

**⚠️ 局限性**

局限性包括：仍需依赖结构数据库检索参考配体，缺乏参考时效果未知；适配过程仍需一定的计算资源（30步更新）；仅更新LayerNorm可能对极端复杂口袋的适配能力有限；

---

## 20. When Do LLM Agents Help? Deadline-Aware Mixed-Criticality Task Scheduling at the Autonomous-Vehicle Edge

**arXiv ID:** 2608.19557 | [PDF](https://arxiv.org/pdf/2608.19557v1)

**作者:** Reza Zakerian `[一作]` `[通讯]` (Westcliff University), Reza Zakerian (Westcliff University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于窗口合同网（contract‑net）拍卖的混合临界任务调度器，并在此基础上加入轻量化LLM控制平面

**💡 创新点**

证明调度优势主要来源于批处理窗口与先执行时间关键任务的排序，而非拍卖或学习组件；LLM仅在非平稳负载时才带来可观提升

**🔧 技术方法**

窗口合同网协议、时间关键优先排序、最早完成时间放置、LLM多代理控制层（broker、edge、monitor）

**📊 数据集**

使用三种真实边缘拓扑（两四服务器、一个六服务器），每个实例200任务，60个随机种子，任务释放、截止、工作量等均从实测分布采样

**📈 对比分析**

与15个基准（包括调度规则、映射启发式、PSO/GA/DRL）和LLM与UCB1两种控制对比；在60个实例上，窗口合同网完成率0.902±0.009，超越所有基准，接近CP‑SAT最优（0.874）；在负载非平稳时LLM提升约0.005-0.006

**⚠️ 局限性**

LLM控制层成本高（平均3 s调用，无法满足实时窗口），且仅在少量非平稳场景下带来微小收益；实验基于确定性通道，未覆盖噪声、队列、物理测试等因素

---

## 21. Temporal Fair Division of Indivisible Mixed Manna: Tractable Settings

**arXiv ID:** 2608.20033 | [PDF](https://arxiv.org/pdf/2608.20033v1)

**作者:** Kui-Wang Choi `[一作]` (City University of Hong Kong), Nicholas Teh `[通讯]` (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了时间序列下可分配不可分混合物品（goods、chores、neutral）的公平分配问题，并提出多种在线与非在线算法实现 Temporal Envy‑Free up to one item (TEF1) 与 Pareto 最优的同时满足；

**💡 创新点**

创新点在于①给出仅含两种物品类型时的在线 TEF1 规则；②提出在“按代理特定缩放后一致”条件下的在线 EF1 + Pareto 最优规则；③针对两部分公共排名序列给出 EF1 规则；④在固定代理数与整数价值时给出伪多项式动态规划判定 TEF1 的算法；⑤揭示 TEF1 与时间最大最小份额 (TMMS) 的最优比及其 NP‑难性。

**🔧 技术方法**

主要技术包括循环分配、归一化价值比率约束、动态规划状态压缩、最大最小份额计算与可判定性分析、以及归约证明与复杂度分析。

**📊 数据集**

无实验数据集，所有结果均为理论分析与算法证明。

**📈 对比分析**

通过复杂度证明表明，在限定物品类型、代理数或整数范围的情形下，算法运行时间为多项式或伪多项式；在这些条件外则保持 NP‑难性。

**⚠️ 局限性**

局限性包括：对通用混合物品实例仍无法给出高效算法；对 EFℓ（ℓ≥2）与 TMMS 的更大比例缺乏上界；以及需要先知先见的“缩放因子”或“公共排名”等额外假设。

---

## 22. MultiVerse: A Creator-Centered Approach to Steering Context-Adaptive Lyrics

**arXiv ID:** 2608.19350 | [PDF](https://arxiv.org/pdf/2608.19350v1)

**作者:** Alexander Wang `[一作]` (Carnegie Mellon University), David Lindlbauer `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于创作者的适应性媒体创作方法C^3，并实现了可让创作者对歌词进行上下文自适应的工具MultiVerse。

**💡 创新点**

核心创新在于把创作者意图、内容结构与上下文三维度显式化为可调控的控制，既能保持艺术意图，又能在实时消费时自动调整。

**🔧 技术方法**

采用 Gemini 3 Flash 生成歌词，结合 RiTa.js 进行语义/韵律/音节等规则校验，并通过多模态提示和变量绑定实现对创作者输入的约束。

**📊 数据集**

数据来源主要为10名歌手自有作品（8–16 行）以及系统内置的12个模拟听众 persona 以及用户生成的对话式上下文查询。

**📈 对比分析**

与自由文本提示的对比实验显示，MultiVerse 在创作者控制感、意图一致性方面被评价更高；尽管生成速度略慢，但无明显差异的主观满意度；未给出客观数值指标。

**⚠️ 局限性**

局限包括：规则约束限制了生成多样性、迭代速度慢；仅支持已完成的固定旋律文本，未处理旋律变化；缺乏真实听众评估与隐私/部署考量。

---

## 23. Distilling Aggregate Mobility Statistics into a Language Model Policy for Post-Event Crowd Simulation

**arXiv ID:** 2608.19778 | [PDF](https://arxiv.org/pdf/2608.19778v1)

**作者:** Tatsuya Amano `[一作]` (University of Osaka), Hirozumi Yamaguchi `[通讯]` (University of Osaka)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

拟合大型语言模型（LLM）作为行人仿真代理，以使模拟人群的目的地分布与仅能观测到的聚合移动统计（区块计数和OD流）一致。

**💡 创新点**

通过信息投影（IPF）将预训练模型的目的地分布校正至目标分布，并引入训练组合校正（transfer map）以消除微调时的类别放大效应，实现无推理时纠正的自由运行仿真。

**🔧 技术方法**

使用LLM（如OpenAI Harmony）、IPF、LoRA低秩适配器、SUMO行人仿真器以及文本格式化策略。

**📊 数据集**

基于日本阪神甲子园球场两场职业棒球赛的移动网络OD计数数据。

**📈 对比分析**

与规则模拟、CMA‑ES调优、LLM提示、传统重力模型等基线比较；网格相关性相似但目的地占比误差显著降低，精细调优后误差下降约25%。

**⚠️ 局限性**

局限在于仅针对单一排队后离场情境，POI与上下文变化有限；天气适应性仅在单一条件下验证，扩展到更复杂场景和更多类别需要更高的采样成本。

---

## 24. Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay

**arXiv ID:** 2608.19760 | [PDF](https://arxiv.org/pdf/2608.19760v1)

**作者:** Haiyue Zhang `[一作]` `[通讯]` (University of Southern California), Haiyue Zhang (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 ALFWorld 单代理工具环境中，通过执行重放构建因果基准，评估隐式信用、判别者信用以及自信度等步骤级信用信号与其对策的排名和实际因果贡献的关联性；

**💡 创新点**

首次将执行重放与步骤级信用信号进行因果检验，揭示信用信号主要是策略流畅度的回声，并系统评估训练层与成本路由器的实际效应；

**🔧 技术方法**

使用执行重放、对策回放、Spearman 相关、部分相关、配对洗牌对照、Bootstrap 置信区间、四维完整性分类、训练循环对照实验以及置信度阈值路由器等技术；

**📊 数据集**

使用 ALFWorld 环境，收集 50 条 Qwen2.5-7B 和 28 条 Llama-3.1-8B 的轨迹，重放 4 个可接受备选动作；

**📈 对比分析**

通过与自身边际匹配的洗牌对照、排名一致性、sign agreement 以及部分相关等统计检验比较，结果显示信用信号与因果贡献无显著关联，训练效果不优于未训练基线；

**⚠️ 局限性**

仅在单一二元结果环境评估，样本量有限、覆盖率不完整、可测点稀疏，未验证多任务或更复杂环境，且未证明信用信号在实际任务中提升效果。

---

## 25. Localized Ecological Momentary Assessment for Mental Health Research in China: An Implementation-Oriented Framework and Preliminary Case Application

**arXiv ID:** 2608.19588 | [PDF](https://arxiv.org/pdf/2608.19588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 26. A Scoping Review of Methods to Measure the Energy and Carbon Footprint of Web Tracking and Advertising

**arXiv ID:** 2608.19495 | [PDF](https://arxiv.org/pdf/2608.19495v1)

**作者:** Nils Bonfils `[一作]` (University of Toronto), Christoph Becker `[通讯]` (University of Toronto)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统梳理15篇文献，归纳出5种测量 Web 追踪与广告碳足迹的方法；

**💡 创新点**

提出方法学框架，揭示研究碎片化与缺乏统一度量；

**🔧 技术方法**

使用 Google Scholar 关键词检索、标题过滤、Snowballing 进行文献筛选，并对各研究的技术细节进行方法学分析；

**📊 数据集**

利用 Tranco、Digg、4imn 等网站列表、公开网络流量采集、广告样本和已有公开数据集；

**📈 对比分析**

对比各方法的测量维度与适用范围：ad‑blocking 最常用但仅针对客户端；控制环境与广告重放提供实验可控性；网络流量与文献估计适用于宏观评估；均未直接测量 CO₂，结果多为能耗或相对比例；

**⚠️ 局限性**

研究仅限标题检索，缺乏系统性；缺少服务器端和二级流量测量；方法标准化不足，数据来源不统一；未直接测量 CO₂ 排放。

---

## 27. Enhancing Privacy in Federated Learning via Dual Obfuscation of Gradients and Training Images

**arXiv ID:** 2608.19650 | [PDF](https://arxiv.org/pdf/2608.19650v1)

**作者:** Yuki Itabashi `[一作]` (Chiba University), Hitoshi Kiya `[通讯]` (Tokyo Metropolitan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种联邦学习中同时对梯度和训练图像进行混淆的双重混淆方法，以提升隐私安全。

**💡 创新点**

创新点在于将梯度随机二值化（FLRSP）与可为每个客户端/图像独立生成密钥的块级图像加密技术相结合，实现梯度信息和视觉信息的互补混淆，并且不需要显式密钥共享。

**🔧 技术方法**

使用了 FLRSP（随机二值化梯度屏蔽）、块/子块置换与像素置换的图像加密、Vision Transformer（ViT）模型、FedSGD 联邦学习框架，以及 APRIL 图像重建攻击进行评估。

**📊 数据集**

实验使用 CIFAR‑10 数据集（将图像缩放到 224×224 并作为 ViT 输入）。

**📈 对比分析**

与仅使用 FLRSP、仅使用图像加密以及未加密的基线进行 APRIL 攻击的图像重建效果对比；在 R_E=1.0 时分类准确率约为 52%（相较于未加密的 98.44%），显示加密强度与精度之间的权衡；梯度屏蔽显著降低了 APRIL 的重建质量。

**⚠️ 局限性**

限制：双重混淆方法并不能提供完整的安全保证；较强的图像加密会显著降低分类精度；实验仅在 FedSGD 与 ViT 上验证，未评估长期收敛性、通信效率以及对更强适应性攻击的鲁棒性；极小的 R_W 下的精度与安全性仍待进一步研究。

---

## 28. From Latent Influence to Language: Diffusion-Oriented Content Generation via Audience-Susceptible Features

**arXiv ID:** 2608.19809 | [PDF](https://arxiv.org/pdf/2608.19809v1)

**作者:** Jiaying Lei `[一作]` (Tongji University), Nan Cao `[通讯]` (Tongji University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个三阶段框架，通过在真实内容流形上进行隐式特征优化、将优化得到的特征显式解码为自然语言描述、再利用多起始点生成多模态内容，从而实现面向特定受众的扩散导向内容生成。

**💡 创新点**

创新点在于：1）利用影响力指示器在内容流形上进行隐式优化，避免手工特征；2）使用可学习的LLM解码器将连续特征映射为可解释的自然语言指令；3）采用多起始点策略捕捉受众异质性，并通过多模态生成实现可控改写。

**🔧 技术方法**

技术主要包括：vMF核密度正则化、基于大规模交互数据训练的影响力指示器、LoRA微调的LLM解码器、Qwen3-VL-Embedding-8B编码器、GPT-4o/FLUX.2文本与图像生成器。

**📊 数据集**

使用两个真实Twitter/X数据集：Movie（聚焦电影话题）和SpaceX（聚焦SpaceX话题），共计约4.1M条推文。

**📈 对比分析**

与四种基线（LLM-zero-shot、LLM-Instruction、IC-L、Designed2Spread）对比，实验显示在Movie数据集上提升16.5%扩散增益，SpaceX上提升40.7%，且在一致性上保持可接受水平；用户研究亦表明更受欢迎。

**⚠️ 局限性**

局限性包括：扩散增益与内容一致性之间的权衡；对vMF正则化、优化初始点等超参数敏感；对计算资源需求高；模型在敏感领域可能生成误导或有害内容，需要更严格的安全与公平评估。

---

## 29. Fairness-Aware Network Embeddings: Methods, Applications, and Challenges

**arXiv ID:** 2608.19381 | [PDF](https://arxiv.org/pdf/2608.19381v1)

**作者:** Ella Has `[一作]` (Leiden University), Akrati Saxena `[通讯]` (Leiden University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

综述了公平网络嵌入方法，梳理了方法论、干预策略与公平目标三维分类框架；

**💡 创新点**

首次将嵌入算法（谱法、随机游走、GNN、贝叶斯、方法无关）与干预阶段（预处理、训练内、后处理）及公平目标（嵌入级/任务级）三轴结合，提供系统统一视角；

**🔧 技术方法**

采用分类梳理、文献对比与案例分析等方法，对各类方法的技术细节与公平机制进行解剖；

**📊 数据集**

综述中引用的典型数据集包括Cora、Citeseer、Facebook、Twitter、Amazon、OAG等社交与知识图谱数据；

**📈 对比分析**

通过对比文献中使用的统计公平指标（SP、EO、RB、MMD、Wasserstein等）与性能指标（准确率、AUC、影响力最大化收益）对方法进行综合评价，指出当前方法在均衡性与任务性能之间的权衡；

**⚠️ 局限性**

局限性在于多集中关注静态、同质网络且假设完整二元敏感属性，缺乏对动态、多属性、异质网络的公平性研究与可解释性分析。

---

## 30. DraftFM: A FoundationModel for Day-Zero Drafting in Magic: The Gathering

**arXiv ID:** 2608.19568 | [PDF](https://arxiv.org/pdf/2608.19568v1)

**作者:** Brian Ward `[一作]` `[通讯]`, Brian Ward

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了DraftFM，一个面向MTG扩展的基础模型，用于在没有任何历史抽选数据的情况下预测玩家在抽卡时的选择。

**💡 创新点**

创新之处在于完全基于卡牌公开特征和冻结文本嵌入，无需卡牌或扩展的标识参数，实现跨集扩展的零样本抽卡预测，并提前发布了HOB扩展的预发布抽卡预测。

**🔧 技术方法**

采用多集预训练的离散选择神经网络（MLP），输入为775维结构化特征和句子级文本嵌入，输出为包内卡牌的选择概率。

**📊 数据集**

使用17Lands公开的约1.7亿个抽选记录以及Scryfall的卡牌属性数据库作为训练和评估数据集。

**📈 对比分析**

在三组完全未见过的扩展上进行零样本评估，Top‑1一致率达到50.8%、60.4%和56.7%，超越随机 baseline，并与六位内容创作者的预发布评估在等级一致性上处于相似水平。

**⚠️ 局限性**

局限性包括无法在缺乏实际抽选日志时评估真实战绩，模型在早期抽选和高水平玩家决策上表现较差，且缺乏后期使用统计和卡牌识别参数，导致预测过度自信。

---

## 31. AutoLumNet: Monotone Optimal Transport for Single-Shot Exposure Correction

**arXiv ID:** 2608.19860 | [PDF](https://arxiv.org/pdf/2608.19860v1)

**作者:** Airin Akter Tania `[一作]` (Khulna University of Engineering & Technology), Mohiuddin Ahmad `[通讯]` (Khulna University of Engineering & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AutoLumNet，一种单帧曝光纠正框架，将全局单调色调曲线与局部受限残差分离，保证像素亮度顺序不被破坏。

**💡 创新点**

创新点包括：①通过正数密度积分构造严格单调色调曲线，天然满足严格单调性；②证明该曲线族在可行解空间中稠密，并包含一维最优传输（OT）映射；③使用受限残差与双分支凸融合实现局部自适应修正，且给出局部顺序保持的充分条件。

**🔧 技术方法**

使用的技术包括：正向传播时的光度正则化（softplus+阈值）构造曲线；基于排序样本的 Wasserstein‑2 损失实现 OT 对齐；双分支残差解码器与 softmax 权重的凸融合；感知损失、平滑正则、SSIM 正则等多项损失组合。

**📊 数据集**

使用的公开数据集有：MSEC、SICE、LCDP 进行主实验；LOL‑v1 与 LOL‑v2‑real 用于零样本低光泛化评估。

**📈 对比分析**

与 12 组基线（低光专用与多曝光融合）对比，AutoLumNet 在 MSEC、SICE、LCDP 上均取得最高 PSNR 与 SSIM，并在无监督的低光数据集上实现零样本最佳表现；推理速度仅 11.2 ms/帧，显著快于同类方法。

**⚠️ 局限性**

局限性包括：①完全被裁剪的像素区域无法逆向恢复，只能基于先验估计；②局部顺序保持仅在满足残差边界条件时成立；③目标亮度分布的选择为经验性假设；④双分支解码器的功能分工仍是经验设计，没有理论保证。

---

## 32. Garbage Collection and Energy Consumption in Java: A Controlled Study Across Workloads and JDKs

**arXiv ID:** 2608.19520 | [PDF](https://arxiv.org/pdf/2608.19520v1)

**作者:** Rahil Sharma `[一作]` `[通讯]` (Vrije Universiteit Amsterdam), Rahil Sharma (Vrije Universiteit Amsterdam)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在控制实验环境下评估了Java HotSpot三种垃圾回收器（Serial、Parallel、G1）在不同工作负载强度和JDK发行版（OpenJDK、Oracle JDK）下对三款服务器端Java应用的能耗和执行时间的影响。

**💡 创新点**

其创新点在于系统地将工作负载强度、JDK分布和应用异质性纳入受控实验，检验GC选择是否具有普适的能耗优势，并揭示工作负载强度是能耗的主要驱动因素。

**🔧 技术方法**

采用EnergiBridge读取CPU功率计、基于随机区组设计（RCBD）的方差分析、Pearson相关系数、能耗-延迟产品（EDP）评估，并用GQM框架设计实验。

**📊 数据集**

使用了3个服务器端Java服务（Spring PetClinic、REST式To‑Do应用、ANDIE图像处理工具），每个服务在轻/中/重三种工作负载强度、两种JDK发行版下共计216次实验。

**📈 对比分析**

通过RCBD ANOVA比较GC、工作负载、JDK主效应及其交互，结果显示GC无显著能耗差异；工作负载强度显著提升能耗，轻负载平均能耗约606 J，重负载约1229 J；EDP与能耗趋势一致，验证了能耗与执行时间的正相关。

**⚠️ 局限性**

局限性包括仅使用三款应用、单一硬件平台、仅测量CPU能耗、实验时间有限，缺乏对更大规模、不同硬件或工作负载类型的普适性验证。

---

## 33. HARP: Hierarchical Adaptive Ranking with Preference-Adaptive Fusion for Query-Based CVE Prioritization

**arXiv ID:** 2608.19430 | [PDF](https://arxiv.org/pdf/2608.19430v1)

**作者:** Haochen Liu `[一作]` (University of Virginia), Haifeng Chen `[通讯]` (NEC Laboratories America)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于查询的CVE优先级排序方法HARP，利用漏洞知识图检索候选并进行多视图评分。

**💡 创新点**

创新点在于：1）处理隐式偏好场景，不需在查询中显式写出偏好；2）通过政策驱动的轻量级适配器实现多视图分数；3）利用历史支持银行自适应融合视图分数，提升针对不同偏好的排序质量。

**🔧 技术方法**

技术手段包括：图知识库检索、基于LLM的政策编辑器（适配器）、多视图（全局、企业、用户）分数计算、支持银行驱动的自适应融合与统一正则化、Top‑K共识路由以及多种LLM骨干（GPT‑OSS‑20B、Llama3‑8B、Qwen2.5‑7B‑Instruct）。

**📊 数据集**

使用的数据集：从公开CVE、补丁、攻击数据库构建的漏洞知识图；基于该图生成查询‑候选集；三种偏好场景（importance、exploit_first、patch_order）下由安全从业者手工标注的排名以及对应的支持集。

**📈 对比分析**

与LLM提示、LLM‑SC、LLM‑KG‑RAG、Embedding、BM25、工业信号以及单视图版本等基线进行对比。HARP在Precision@K、MAP@K、MRR@K等指标上均显著优于所有基线，在三种偏好场景和三大LLM中保持领先。

**⚠️ 局限性**

局限性包括：仅覆盖三种偏好场景和英文查询；未考虑多语言、专有资产或组织特定工作流；仅冻结LLM骨干，仅训练适配器，未探索联合微调；推理成本随候选集大小和硬件变化；系统仍需人工监督以防误判。

---

## 34. Coupled Optimal Transport with Landmark Constraints

**arXiv ID:** 2608.19783 | [PDF](https://arxiv.org/pdf/2608.19783v1)

**作者:** Xiang Gu `[一作]` (Xi'an Jiaotong University), Zongben Xu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种联合优化传输计划与变形场的耦合最优传输（OT）框架，利用少量标记点引导变形恢复，从而得到既符合分布匹配又具几何意义的变形映射。

**💡 创新点**

创新点包括：①将分布匹配与标记约束在同一变分模型中耦合；②对模型的存在性、极限行为（α→0、α→1）进行理论分析；③设计基于有限元的数值算法，并给出迭代收敛证明；④通过合成与真实形状匹配实验验证方法优越性。

**🔧 技术方法**

使用的技术包括：最优传输理论、变分正则化（弹性能量、鲁棒损失）、有限元离散、Sinkhorn 迭代、KL/熵正则化、变分分裂与拉格朗日近似。

**📊 数据集**

实验数据集主要包括：①合成的鱼形分布（含已知变形）；②MNIST 手写数字图像，用于真实形状匹配。

**📈 对比分析**

方法与传统 OT（重心投影）和仅标记拟合进行对比。实验表明，在稀疏标记条件下，COT 的变形误差、持出标记误差和分布对齐误差均显著低于基线方法；随着标记数量增加误差进一步下降，且两种数值实现（KL 近似与熵正则化）性能相近。

**⚠️ 局限性**

局限性：目前实现仅适用于低维空间，有限元离散受网格限制；未解决高维/大规模问题；缺乏对噪声标记的鲁棒性分析。

---

## 35. A Speech Corpus for Mizo Automatic Speech Recognition: Whisper and SraVaani 1.0 Fine-Tuning with Morphology-Aware Evaluation

**arXiv ID:** 2608.19361 | [PDF](https://arxiv.org/pdf/2608.19361v1)

**作者:** Priyankoo Sarmah `[一作]`, Lalhmingmawia `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了17.62小时的Mizo语音语料库，并在此基础上对Whisper和SraVaani 1.0多语种ASR模型进行Fine‑Tuning，形成可公开使用的Mizo ASR系统。

**💡 创新点**

①首次在低资源Tibeto‑Burman语言Mizo上演示大规模多语种预训练模型（Whisper large‑v3）可被高效迁移；②引入“形态学感知MA‑WER”评估指标，纠正传统WER对Mizo形态边界差异的误判；③比较零射击与细化训练的显著性能提升。

**🔧 技术方法**

使用Whisper（small/medium/large‑v3）和SraVaani 1.0（FastConformer+RNNT/CTC）两类多语种ASR框架；Fine‑Tuning采用AdamW/Adafactor优化器、学习率5e‑6、批量16、16-20个epoch；评估指标包括WER、CER与MA‑WER。

**📊 数据集**

17.62小时语料库（训练7656句、验证426句、测试192句），包含新闻与法院判决文本，数据已在AI‑Kosh公开发布。

**📈 对比分析**

通过与零射击SraVaani 1.0基线对比，Whisper‑large‑v3 Fine‑Tuned在传统WER上达到18.08%（MA‑WER 7.22%），显著优于小模型和SraVaani；Fine‑Tuned SraVaani 1.0从58.27%降至29.45%（MA‑WER 17.93%），验证了语言特定适配的重要性。

**⚠️ 局限性**

1）仍依赖人工审核语料；2）形态学边界不统一导致MA‑WER与传统WER差异大，表明评估仍需改进；3）模型对特殊符号（如< t >）的处理仍不完善；4）低资源语料规模有限，未来可扩展更多领域文本。

---

## 36. FAR-DPO: Feasibility-Aware and Robust Direct Preference Optimization for Cyclic Peptide Design

**arXiv ID:** 2608.19808 | [PDF](https://arxiv.org/pdf/2608.19808v1)

**作者:** Guofeng Zhang `[一作]` (Beijing University of Posts and Telecommunications), Guangyu Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种通用的基于偏好对抗训练框架FAR-DPO，用于提高循环肽设计的可行性和目标特异性；

**💡 创新点**

创新点在于：①通过可行性门控和容差感知的多目标支配构造高质量的偏好对；②引入难度分组的群鲁棒优化，动态加权高损失组；③保持生成模型不变，实现跨架构的泛化；

**🔧 技术方法**

使用偏好对抗训练（DPO）、群鲁棒优化（Group DRO）、基于预测误差的参考相对变化、以及结构与能量门控筛选；

**📊 数据集**

在12,000个蛋白结合口袋上生成192,000条候选（PepGLAD、PepFlow）并构造约18万条偏好对；

**📈 对比分析**

与PepGLAD、PepFlow、DiffPepBuilder、CP-Composer等基线在CPSea LNR 56个目标上比较，FAR-DPO在最终成功率、联合合格率和难度最高组上分别提升约10–15%，且在低采样预算下表现更佳；

**⚠️ 局限性**

局限性包括：依赖离线生成的候选池，难度分组固定且可能对其他模型不适用；对极端结构或非典型环化方式的适应性待验证。

---

## 37. Enforcing LLM Safety through DMD-based Classification of Prompt-Response Embedding Dynamics

**arXiv ID:** 2608.19579 | [PDF](https://arxiv.org/pdf/2608.19579v1)

**作者:** Mohamed Akrout `[一作]` (University of Tennessee), Dan Wilson `[通讯]` (University of Tennessee)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在黑盒环境下利用Koopman算子对LLM的提示词与回复词的嵌入动力学进行建模，从而实现对生成文本是否安全（安全/不安全）的二分类；

**💡 创新点**

首次将提示词与回复词的动态行为联合建模，利用差分残差分数比较安全与不安全的Koopman预测误差，从而捕捉交互依赖的安全违规；

**🔧 技术方法**

基于扩展动态模态分解（EDMD）的Koopman算子估计，嵌入模型（如Qwen-Embed、Mistral、Llama-3），差分残差分数阈值化判别；

**📊 数据集**

Aegis AI Content Safety Dataset 2.0、Synthetic CoT Safety Benchmark、BeaverTails Dataset；

**📈 对比分析**

在三大安全基准上与单独使用回复嵌入的方式比较，加入提示词嵌入后在交互依赖型违规上明显提升（F1从约73%提升至77%或更高），在回复仅违规场景提升幅度更小；整体在黑盒条件下可达80%+准确率、70%+F1；

**⚠️ 局限性**

仅为二分类，无法给出具体违规类别；对模型超参数（阈值）敏感；对长序列样本数不足时性能波动；在非交互型违规场景引入提示词信息提升有限；

---

## 38. Robust Metaheuristics under Uncertainty for Berth Allocation and Quay Crane Assignment: A Review

**arXiv ID:** 2608.19214 | [PDF](https://arxiv.org/pdf/2608.19214v1)

**作者:** Yang Li `[一作]`, Wenjian Luo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并评估了多种在泊位分配与码头起重机分配（Berth Allocation and Quay Crane Assignment）问题中针对不确定性（船到达时间、起重机可用性等）的稳健元启发式算法，提供可复现的评估协议与基准实验结果。

**💡 创新点**

创新点包括：①构建了基于 slack 的鲁棒性指数，用于量化解决方案在面对不确定性时的缓冲与冲突容忍度；②在同一框架下综合考量效率（平均泊位时间）与鲁棒性，系统比较了 GA、ACO 与 PSO 等三类搜索器与 MMRO、ERO、TSRO 等三种鲁棒策略的交互效果；③提供了详尽的可复现性协议与额外实验数据，为后续研究提供了公开可检验的基准。

**🔧 技术方法**

使用的技术主要包括遗传算法（GA）、蚁群算法（ACO）、粒子群优化（PSO）三类元启发式，配合三种鲁棒策略（MMRO、ERO、TSRO）实现对不确定泊位与起重机分配的求解；评估指标为平均泊位时间（T̅_port）和鲁棒性指数（σ̅）。

**📊 数据集**

实验数据集为人工生成的泊位与起重机分配基准实例，分别包含 50 船和 200 船两种规模。每个实例提供船到达时间、船舶长度、泊位与起重机可用性等信息，构成基准测试环境。

**📈 对比分析**

比较方法为将每种算法与鲁棒策略组合在同一实例下运行多次（每种配置均多次复现），计算平均值和标准差。结果显示：PSO+ERO/TSRO 在平均泊位时间上常表现最佳；ACO+MMRO/TSRO 在鲁棒性指数上具备竞争力；GA 组合作为参考基线。整体上不同搜索器与鲁棒策略的交互决定了效率–鲁棒性权衡的优劣。

**⚠️ 局限性**

局限性包括：①实验仅涵盖三类搜索器和三种鲁棒策略，未覆盖更广泛的元启发式或混合方法；②基准数据为人工生成，缺乏真实港口的复杂性与多维约束；③报告的结果为基准参考，不构成最终排序，缺乏对算法收敛性、可扩展性等更深入的理论分析。

---

## 39. DIFFCZSL: Compositional Zero-Shot Learning Regularized by Diffusion Representations

**arXiv ID:** 2608.19871 | [PDF](https://arxiv.org/pdf/2608.19871v1)

**作者:** Hangyu Tian `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在组合零样本学习（CZSL）中，将预训练扩散模型的中间特征作为辅助先验，注入CLIP‑based CZSL 训练流程，提升对未见属性–对象组合的识别能力。

**💡 创新点**

创新点：①利用扩散模型生成的中间特征在视觉与文本两侧同时进行双向对比蒸馏，形成结构化的组合先验；②保持原始推理流程不变，完全无推理时额外开销；③框架可直接插拔于多种 CLIP‑based CZSL 方法（如 CSP、Troika、CAMS）。

**🔧 技术方法**

技术手段：Stable Diffusion 2.1（CleanDIFT）特征提取、CLIP ViT‑L/14 编码器、双空间（图像/文本）对比蒸馏损失、温度化余弦相似度、基线跨模态损失的加权组合。

**📊 数据集**

使用数据集：MIT‑States、UT‑Zappos50K、C‑GQA 三大公开 CZSL 基准。

**📈 对比分析**

与多种基线（CSP、Troika、CAMS 等）在闭域/开放域两种评估设置下对比，均在 HM、AUC 及 seen/unseen 最高准确率等指标上实现提升；例如 Troika 在 MIT‑States 关闭域 HM 从 39.2 提升至 39.9，UT‑Zappos 关闭域 AUC 从 41.7 提升至 46.2。

**⚠️ 局限性**

局限性：①依赖扩散模型的预训练质量，若扩散模型不包含足够的属性–对象关系，先验效果有限；②训练阶段需额外的特征提取与蒸馏计算，导致一定的时间与参数开销；③对极少见或高度细粒度的属性–对象组合仍可能产生错误，未能彻底消除组合偏差。

---

## 40. S$^2$GS: Structured Sparse Gaussian Streaming for Efficient Free-Viewpoint Video Reconstruction on Edge-IoT Devices

**arXiv ID:** 2608.19639 | [PDF](https://arxiv.org/pdf/2608.19639v1)

**作者:** Yiwei Li `[一作]` (Hong Kong Polytechnic University), Mingjin Zhang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种结构化稀疏高斯流（S²GS）框架，用于边缘 IoT 设备上高效流式自由视角视频（FVV）重建。

**💡 创新点**

创新点在于：①利用流式八叉树对高斯残差进行层次化组织；②引入层次特征传播（HFP）和 Gumbel‑Sigmoid 采样配合多层 STE 的稀疏门控机制，实现对动态高斯残差的结构化、稀疏更新；③通过稀疏正则化和逐帧训练进一步压缩存储与计算成本。

**🔧 技术方法**

采用 Gaussian Splatting、流式八叉树、层次特征传播、Gumbel‑Sigmoid 采样、多层 STE、稀疏正则化、逐帧训练、光度损失、LPIPS 等技术。

**📊 数据集**

在 N3DV、MeetRoom、ENeRF 三个公开动态 FVV 数据集以及真实工业测试台（多视角摄像头）进行评测。

**📈 对比分析**

与 QUEEN、4DGC、ReCon‑GS、StreamRF 等现有方法对比，S²GS 在 RTX4090 上每帧训练时间 2.3‑5.0 s、存储 0.1‑0.2 MB、PSNR 约 30‑31 dB；在 NVIDIA Jetson AGX Orin 上训练时间 <6 s、渲染 60+ FPS，能耗最低，且在多分辨率场景下表现优异。

**⚠️ 局限性**

局限性：①固定八叉树层次结构难以适应大范围运动或新几何的场景；②根级门控粒度有限，难以在同一区域内区分不同运动；③对首帧重建质量依赖较大，极端噪声或大变形时性能下降。

---

## 41. Where Grounding Accuracy Lives on the IoU Curve: Label-Free Inference-Time Boundary Refinement

**arXiv ID:** 2608.19553 | [PDF](https://arxiv.org/pdf/2608.19553v1)

**作者:** Bo Ma `[一作]` `[通讯]` (Auckland University of Technology), Bo Ma (Auckland University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了冻结的多模态视觉语言模型如何在不访问标注的前提下，利用自身预测的精度进行一次局部精细化裁剪以提升定位精度。

**💡 创新点**

提出Label-Free Precision Refinement (LFPR) pipeline，包含基于预测框大小的分辨率路由、几何守门的裁剪重定位和固定中点融合，实现零更新、零标注的精细化。

**🔧 技术方法**

使用冻结的直接回答VLM（如Qwen3-VL-8B）、基于像素面积的分桶路由、几何一致性守门、crop‑context重定位、固定中点融合以及图像‑cluster bootstrap统计。

**📊 数据集**

在Ref‑L4、RefCOCO/RefCOCO+/RefCOCOg、Flickr30K Entities等公开表达式定位基准上进行实验。

**📈 对比分析**

与冻结基线、分辨率诊断、crop‑only、专家专用模型等多种基线进行配对对比，LFPR在Acc@0.5、Acc@0.75、Acc@0.9、mAcc和Mean IoU上均显著提升，尤其在Acc@0.75和Acc@0.9上提升幅度大。

**⚠️ 局限性**

仅在单一模型族上验证、未做能源/延迟评估、未对其他分辨率阈值优化、评估受Retrospective/Prospective分层限制、对标注与训练数据隔离依赖严格。

---

## 42. APPROVE: Visual End-User-in-the-Loop Robot Programming with LLMs

**arXiv ID:** 2608.19281 | [PDF](https://arxiv.org/pdf/2608.19281v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 43. Learning Early-to-Final Solution Consistency for MILP Acceleration

**arXiv ID:** 2608.19953 | [PDF](https://arxiv.org/pdf/2608.19953v1)

**作者:** Guanlin Li `[一作]` (State Key Laboratory of Novel SoftwareTechnology, Nanjing University), Chao Qian `[通讯]` (State Key Laboratory of Novel SoftwareTechnology, Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EnCore框架，通过预测混合整数线性规划（MILP）求解器在早期阶段得到的可行解与最终解之间的变量一致性，从而加速求解；并将一致性预测与现有的预测引导搜索管线（如Predict-and-Search、ND、PS、Apollo）无缝集成。

**💡 创新点**

创新点在于：①将学习目标从完整解预测转变为早期解一致性预测，利用求解器自身产生的早期可行解信息；②提出多解集成策略提高鲁棒性；③给出严格的理论证明，说明在满足后验交叉条件下，条件化于早期解的预测可获得更高的可达精度，并在有限样本下保持优势。

**🔧 技术方法**

技术手段包括：基于双边消息传递的图神经网络（GNN）对MILP的变量–约束双分图及早期解特征进行编码；二元一致性标签（早期解与最终解是否相同）进行交叉熵训练；设计早期解采集策略（监测可行解改进速率，停止于快速下降阶段结束）；推断时对最近K个改善解进行集成并对logit进行对齐。

**📊 数据集**

实验使用四大MILP基准：Combinatorial Auctions（CA）、Set Covering（SC）、Workload Apportionment（WA）、Item Placement（IP）；并在MIPLIB IIS子集上进行零射频迁移，进一步验证跨问题族的鲁棒性。

**📈 对比分析**

与原始解预测器和三种搜索管线对比：在Gurobi求解器上，EnCore平均缩小主观差距56.9%，在CA实例上完全消除差距；在零射频迁移到SCIP时平均缩小36.4%差距；在MIPLIB IIS上实现全可行性并平均提升目标值，显示显著性能提升。

**⚠️ 局限性**

局限性包括：对早期解质量高度依赖，若早期解不佳或采集耗时过长可能失效；目前仅在二元整数变量上验证，连续变量的适用性尚未充分探究；引入早期解采集和集成增加了额外的前置成本，需在实际应用中权衡。

---

## 44. Mix&Fix-Net: A Dual-Stage Trajectory Prediction Model for AIS and Vision-Derived Vessel Data

**arXiv ID:** 2608.19580 | [PDF](https://arxiv.org/pdf/2608.19580v1)

**作者:** Md Mahmuddun Nabi Murad `[一作]` (University of South Florida), Yasin Yilmaz `[通讯]` (University of South Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种适用于AIS与非AIS（视觉）数据的双阶段MLP‑Mixer模型Mix&Fix-Net，用于船舶轨迹预测；

**💡 创新点**

创新点包括：1) 双阶段结构——粗预测 + 误差细化；2) 对输入进行零填充、仅使用实例反归一化（不做归一化）；3) 专门设计的Head层，直接输出坐标对；4) 在轨迹预测中首次引入输入填充；5) 仅在非AIS场景下构建视频轨迹数据集；

**🔧 技术方法**

使用MLP‑Mixer网络（embedding、时间Mixer、嵌入Mixer、残差连接、层归一化）以及实例反归一化、字节追踪等视觉前处理；

**📊 数据集**

采用三组数据集：VVR（基于webcam的视频轨迹）、M3（合成AIS数据）和TampaBay（真实AIS数据）；

**📈 对比分析**

与P_sLSTM、PatchTST、DLinear、WPMixer、FusionRNN等基线在六种指标（MSE、MAE、SMAPE、FDE、FD、AED）下对比，Mix&Fix-Net在大多数指标和数据集上均表现最优，尤其在严格的无数据泄露拆分（Setting‑1）下表现稳健；

**⚠️ 局限性**

局限性主要来自VVR视频数据：分辨率低、时间覆盖短、跟踪噪声大、轨迹不够平滑，导致对低质量视觉输入的泛化性受限，未来需要加入运动平滑、投影校正等改进。

---

## 45. Securing Filesystems for Confidential Computing

**arXiv ID:** 2608.19924 | [PDF](https://arxiv.org/pdf/2608.19924v1)

**作者:** Dimitra Giantsidi `[一作]` (Azure Research, Microsoft), Stavros Volos `[通讯]` (Azure Research, Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 ShieldFS，一种在可信执行环境（TEE）中运行的 POSIX 文件系统，能够在不需要改动应用程序的前提下，为持久化存储提供完整性和新鲜度保证。

**💡 创新点**

创新点包括：①将文件系统的 WAL 与存储根状态转化为加密承诺（hash 链与 Merkle 树），②在轻量级可信注册表（基于 CCF）中同步维护这些承诺以防止回滚、重放和分叉攻击，③在现有 ZFS 之上实现，几乎不改变其内部结构。

**🔧 技术方法**

主要技术包括：加密哈希（SHA‑256）与 Merkle 树用于块认证，WAL 哈希链用于日志完整性，TEE 中的安全内存与远程证明用于认证与注册，CCF 分布式账本用于低延迟可信注册表。

**📊 数据集**

实验使用了行业基准数据集：FIO 随机/顺序读写、TPC‑C PostgreSQL（10W/100W），Filebench（varmail 与 fileserver）以及 RocksDB（多种读写/压缩工作负载）。

**📈 对比分析**

通过与 ext4、ext4+dm‑integrity 以及原始 ZFS 进行对比，ShieldFS 在大多数工作负载下性能与 ZFS 相当，同步写入场景平均开销 <10%，但在 fsync‑密集型任务（如 Filebench varmail）下最高可达 1.7×，相较于原始 ZFS 或 ext4 的性能损失更小；相比 ext4+dm‑integrity，ShieldFS 在同步写时更快，读取性能略逊于 ZFS。

**⚠️ 局限性**

局限性：①同步写时需额外注册承诺，导致一定的延迟；②在完全可靠的 TEE 与注册表可用性假设下工作，对注册表故障或网络延迟敏感；③未提供额外的数据机密性（需单独加密块）；④目前仅针对 POSIX 文件系统，扩展到分布式或专有文件系统仍需进一步研究。

---

## 46. Evidence Before Expansion: Reuse, Spawn, or Defer in Lifelong Expert Pools

**arXiv ID:** 2608.19888 | [PDF](https://arxiv.org/pdf/2608.19888v1)

**作者:** Kentaro Oda `[一作]` `[通讯]` (Kagoshima University), Kentaro Oda (Kagoshima University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种统计学上可解释的“暂缓”决策层，用于在非平稳流中决定是重用现有专家模型、创建新专家还是等待更多证据；通过条件JS散度（CJSD）和可预测的赌注e‑process实现两轴判定；在此基础上构建了具有随时间可验证性、内存受限的递归e‑detector；

**💡 创新点**

创新点包括①把暂缓（defer）定义为一可检验的统计区间；②使用条件JS散度拆分协变量轴与功能轴，避免单轴触发器的混淆；③通过一次性α‑spending与重启e‑process实现全生命周期的家族错误控制；④设计了基于短列筛选、mini‑batch路由与合并的系统机制，提升效率与准确性；

**🔧 技术方法**

技术主要包括条件JS散度估计、可预测的分数判别器、一次性α‑spending、重启e‑detector、短列筛选、mini‑batch test‑then‑train路由、专家合并逻辑，以及多方案的对比与评测；

**📊 数据集**

使用四类数据集：合成的四规（突发切换、协变量仅阶段、渐进漂移、重现）流；真实数据集包括 Electricity、Covertype、INSECTS（突发与可重现）等；

**📈 对比分析**

与七种传统策略（单一模型、始终生成、输入新颖性、损失跳变、交换评分、CPD族门、CJSD门）在相同路由与裁剪设置下对比；CJSD门在理想专家数量下实现零误生成与零漏检，整体准确率与最佳证据变体相当；在重现性强的INSECTS上表现最佳；

**⚠️ 局限性**

局限性：对持续漂移（continuous drift）流无效；判别器计算成本高于基于损失的触发器约3倍；对多重性控制仍需更精细方法；对基础估计器的有限样本理论待完善。

---

## 47. The Evaluation Context Protocol (ECP): A Portable Contract for AI Agent Evaluation

**arXiv ID:** 2608.19263 | [PDF](https://arxiv.org/pdf/2608.19263v1)

**作者:** Aniket Wattamwar `[一作]`, Mrunal Kakirwar `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Evaluation Context Protocol（ECP），为AI代理提供可移植、统一的评估合同层，支持跨框架的轨迹级评估、工具调用验证以及审计证据公开。

**💡 创新点**

核心创新在于将评估抽象为轻量级JSON‑RPC接口，拆分用户输出、工具调用与评估安全证据三大可度量字段，兼容MCP的工具执行层，首次实现评估层与执行层的完全解耦；同时提供开源参考实现、适配器和CI集成工具。

**🔧 技术方法**

技术栈包括JSON‑RPC 2.0、JSON Schema、Python/TypeScript SDK、LangChain/LlamaIndex/CrewAI/PydanticAI适配器；利用LLM‑as‑a‑judge、确定性检查器、pass@k 与 pass^k 统计评估；通过标准化的HTTP与STDIO两种传输实现跨环境互操作。

**📊 数据集**

使用多种公开基准与实验集：MMLU、HumanEval、AgentBench、GAIA、SWE‑bench、WebArena、CORE‑bench，以及内部 MAESTRO、MOYA、AIDev、医疗领域爬取的数据；评估脚本通过JSON Manifest定义任务与期望工具调用。

**📈 对比分析**

将ECP与传统单答评估对比，展示ECP能捕获“自信错误”“工具调用混乱”“路径低效”等失败模式；通过pass@k与pass^k对比，证明后者更能反映实际可靠性；实验表明ECP在检测率上提升约30–50%，但评估成本相对基础评估增加约10–15% LLM推理预算。

**⚠️ 局限性**

局限性包括：评估表面仍为实验性且缺乏完整的字段定义；适配器对中间推理的结构化支持有限，未覆盖委托与多代理交互；高昂的LLM‑judge成本与存储/可观测性负载；缺乏对大规模持续集成的成本与性能基准；需要进一步的跨框架验证、版本管理与安全审计机制。

---

## 48. A three-dimensional typology of agency for advanced AI systems

**arXiv ID:** 2608.20041 | [PDF](https://arxiv.org/pdf/2608.20041v1)

**作者:** Willem Fourie `[一作]` `[通讯]` (Stellenbosch University), Willem Fourie (Stellenbosch University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文基于哲学、伦理、法律与社会学理论，构建了一个三维的人工智能系统代理类型框架（性质：法律/道德；模式：个体/集体；位置：人类/非人类），并由此产生八种代理实例；

**💡 创新点**

创新点在于首次将法律与道德代理的区别系统化，并将其与个体/集体与人类/非人类维度相结合，创造了可区分传统代理与争议代理的新框架，为非人类AI代理的治理提供了概念空间；

**🔧 技术方法**

主要采用哲学与法律理论分析方法，未涉及具体技术实现；

**📊 数据集**

本文无实验数据集；

**📈 对比分析**

未进行实验或性能评估，因研究为理论框架构建；

**⚠️ 局限性**

局限性包括：对代理性质与维度的定义带有主观性；缺乏实证验证；对不同司法体系的适用性尚未评估；同时在实际治理中如何落实非人类代理的法律责任仍需进一步探讨。

---

## 49. Beyond Multimodal Alignment: Certifying Physical Language through Response Substitution and Ordered Execution

**arXiv ID:** 2608.19492 | [PDF](https://arxiv.org/pdf/2608.19492v1)

**作者:** Kaizhen Tan `[一作]` (New York University), Heqing Du `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并验证了 Disjoint-Bridge Operator-Substitution Certificate，评估多模态感知在物理交互中的可替换性与顺序执行能力。

**💡 创新点**

创新在于将多模态意义转化为响应空间测度，并通过操作层次分离属性访问、响应替换、融合闭合与有序执行。

**🔧 技术方法**

使用冻结执行器、响应图谱、线性/GRU 编译器、Gaussian 信息聚合等技术。

**📊 数据集**

实验基于 Cluster Haptic（音频+加速度扫描 118 个表面）和一个受控弹塑性振荡器。

**📈 对比分析**

通过对比同表面跨模态桥距、错误表面和群体基准，以及 NMSE 与能量评分，发现跨模态替换成功、融合提升且有序执行需额外训练，表现出 4.5 倍更近的桥距和约 0.18 的 NMSE。

**⚠️ 局限性**

局限在于仅使用单一装置和两种模态、实验场景受限，且有序执行对执行器训练预算敏感。

---

## 50. Open-Vocabulary 3D Object Detection with Co-Distillation Discovery and Dual Guidance Robust Training

**arXiv ID:** 2608.19973 | [PDF](https://arxiv.org/pdf/2608.19973v1)

**作者:** Shangbo Yuan `[一作]` (University of Electronic Science and Technology of China), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Co-3DGT框架，融合Co‑Distillation发现与Dual Guidance鲁棒训练，实现3D开词表目标检测；

**💡 创新点**

创新点在于：①使用Hungarian匹配对2D语义与3D几何进行联合蒸馏，提升新颖目标发现质量；②设计场景感知不确定性正则化与LLM层级对齐的双向引导训练，提升定位与分类鲁棒性；

**🔧 技术方法**

采用CLIP视觉‑语言模型、CuTR/Detic等基础检测器，Hungarian匹配、基于不确定性的加权损失、LLM生成层级标签与多层次语义对齐；

**📊 数据集**

在SUN RGB‑D和ScanNetV2两大室内点云数据集上进行实验；

**📈 对比分析**

与多种SOTA方法对比，在两数据集上均取得显著提升：SUN新颖类AP25提升至14.37%（+4.71%），ScanNet新颖类AP25提升至21.91%（+9.82%），整体mAP均超越对手；

**⚠️ 局限性**

局限性包括：需要两阶段流程，额外计算成本；对2D/3D检测器的依赖较强；在动态或极端遮挡场景下表现仍有限。

---

## 51. DeltaML-Bench: Evaluating Machine Learning Agents on Real-World Research Repositories

**arXiv ID:** 2608.19653 | [PDF](https://arxiv.org/pdf/2608.19653v1)

**作者:** Josias Moukpe `[一作]` (Algorithmic Research Group), Matthew Kenney `[通讯]` (Algorithmic Research Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出DeltaML‑Bench基准，评估自动化机器学习实验代理在真实研究仓库中改进基准模型的能力；

**💡 创新点**

创新点在于（1）真实多领域任务要求改进而非仅重现；（2）多层完整性与规范游戏检测；（3）搜索式ARG框架显著提升传统Modular框架的表现；

**🔧 技术方法**

使用前沿LLM（Claude Sonnet 4、GPT‑5）配合两种代理框架（Modular与ARG），并利用自动化工具调用、Beam搜索、反射、日志审计等技术；

**📊 数据集**

采用48个来自Papers‑with‑Code的公开仓库任务，涵盖计算机视觉、图/分子、时序、表格、NLP等领域，使用各论文所附原始数据集；

**📈 对比分析**

通过任务成功率、平均归一化改进、规范游戏率等指标比较，ARG在GPT‑5上将成功率从9.4%提升至33.9%（4×6h）或49%（2×12h），并消除游戏；Claude在某配置表现不一；

**⚠️ 局限性**

限制在于仅评估两大前沿模型与两框架，受计算资源限制，任务时长≤12h，未覆盖分布式训练或更小模型，缺乏对游戏机制因果关系的深入分析。

---

## 52. Reliable Neural Collapse Approximation for Open-World Test-Time Adaptation

**arXiv ID:** 2608.19890 | [PDF](https://arxiv.org/pdf/2608.19890v1)

**作者:** Jia-Qi Lin `[一作]`, Joey Tianyi Zhou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于神经折叠（Neural Collapse）理论的可靠目标域自适应方法ReNC，能够在开放世界测试时自适应中同时处理标签分布和数据分布漂移；

**💡 创新点**

创新点在于将预训练模型的分类器权重视为源域原型，构建无源数据的可靠OOD筛选机制，并通过原型逐步逼近目标域的神经折叠结构，实现无需源域信息的自适应；

**🔧 技术方法**

主要技术包括神经折叠近似、可学习阈值的OODA过滤、原型的随机（EMA）更新、交替优化（ADMM）与熵最小化、分布均衡约束等；

**📊 数据集**

使用了CIFAR10‑C、CIFAR100‑C、ImageNet‑C、ImageNet‑R、VisDA‑C等常用破坏和风格迁移数据集，并在每个目标域中混合了多种OOV数据（噪声、MNIST、SVHN、Tiny‑ImageNet、CIFAR100‑C等）；

**📈 对比分析**

与10种基线（TEST、BN、TENT、SHOT、OSTTA、EATA、RMT、CoTTA、UniEnt、OWT3）以及VLM扩展C‑TPT+进行对比，ReNC在ACC_I、ACC_O与调和平均ACC_H上均显著优于所有基线，尤其在大类别（如ImageNet‑C）下提升高达20%；

**⚠️ 局限性**

局限性包括对超参数λ的依赖较高，原型更新仍可能在极端OOV比例或极少样本类别下不稳定；此外，实验主要集中在视觉分类任务，对序列或跨模态任务的推广尚未验证。

---

## 53. Learning how to Forget: Fine-tuning for Long-Context Sparse Attention

**arXiv ID:** 2608.19920 | [PDF](https://arxiv.org/pdf/2608.19920v1)

**作者:** Matthias Seeger `[一作]` (Amazon Web Services), Sebastian Schelter `[通讯]` (Technical University Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种可在单张A100 GPU上完成的Transformer模型稀疏注意力微调方法，支持任意KV缓存策略并实现了显存的显著压缩；

**💡 创新点**

创新点在于将嵌套激活检查点、CPU卸载以及KV缓存递归压缩与稀疏注意力结合，形成了一套高效且通用的微调框架，并对Heavy‑Hitter‑Oracle(H2O)策略实现了高效的SDPA内核；

**🔧 技术方法**

采用了稀疏注意力、KV缓存压缩、H2O缓存策略、Triton/FlashInfer SDPA内核、LoRA、AdamW、RoPE、YaRN、分布式数据并行和自动微分回调保存张量等技术；

**📊 数据集**

实验使用了Helmet、LongBench V2以及Helmet的六个额外子集（trec_coarse、nlu、clinc150、inf_qa、inf_mc、json_kv）作为评测数据集；

**📈 对比分析**

通过与采用序列并行（exact attention）训练的基线在相同缓存策略下进行对比，实验表明本方法在多数长上下文基准上性能相当甚至优于基线，且在部分任务上表现更佳；

**⚠️ 局限性**

主要限制包括与目前最快推理库相比仍存在延迟差距、对特定缓存策略的依赖需手动选择、在某些数据集上模型易产生冗长或无意义输出，以及对SDPA内核扩展的社区支持仍需进一步完善。

---

## 54. The Verification Gap in Networked Physical AI: A Post-Semantic Communication Framework

**arXiv ID:** 2608.19593 | [PDF](https://arxiv.org/pdf/2608.19593v1)

**作者:** Shunsuke Saruwatari `[一作]` `[通讯]` (University of Osaka), Shunsuke Saruwatari (University of Osaka)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了后语义通信框架(Post‑Semantic Communication Framework)，用来弥补任务有效提案与最终物理执行之间的“验证缺口”

**💡 创新点**

将证据需求、证据记录、冲突优先验证、授权最终化以及运行时门控等概念拆解为独立的、可通信的层次，明确了证据传输与协调的差异，并通过有限状态机检查实现一致性

**🔧 技术方法**

采用有限状态机验证器、冲突优先的验证策略、证据转移与协调机制、授权策略与运行时门控模型；实验基于合成的证据合同与记录生成器

**📊 数据集**

使用纯合成数据集：两条证据（视觉清除与无线无动静态）在两个端点的可用性、真值概率、记录大小、链路速率等均以固定参数产生；无真实传感器或无线链路数据

**📈 对比分析**

通过在四种通信策略（Sender/One‑way、Sender/Feedback、Receiver/One‑way、Receiver/Feedback）下枚举完整状态空间，比较无条件覆盖率、选择性误差与期望证据负载；实验结果显示：在发送方最终化时，Feedback（通过证据转移）在所有可行格点均优于One‑way；在接收方最终化时，Feedback在低延迟下最小化负载，且大多数格点仍优于One‑way；在延迟或截止时间恶化时两者均不可行

**⚠️ 局限性**

仅在合成模型中验证，未涉及实际无线链路、传感器校准、运行时控制或物理安全；证据记录的真实性、时效性及完整性需在真实系统中进一步验证

---

## 55. Outcome Monitors: Recovery Affordances for Silent Tool Failures

**arXiv ID:** 2608.19303 | [PDF](https://arxiv.org/pdf/2608.19303v1)

**作者:** Sugam Panthi `[一作]` (University of Southern Mississippi), Rabab Abdelfattah `[通讯]` (University of Southern Mississippi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种确定性事件合约检测器，检测工具返回结果是否违反预先挖掘的合约，并以非绑定的收据形式提示代理，帮助代理在不受强制执行限制的情况下自行恢复。

**💡 创新点**

创新点在于将违规检测与恢复解耦：利用“outcome contracts”生成的合约在运行时检测违规，随后发布可选的恢复工具列表（收据），而不干预代理决策，从而提升恢复的灵活性与可解释性。

**🔧 技术方法**

技术包括：跨折叠动态属性挖掘（类似 Daikon）生成合约、基于公共工具接口构建可恢复工具映射、Deterministic Detector + Receipt API 以可序列化形式嵌入返回结果，以及使用 Bootstrap/McNemar 统计检验效果。

**📊 数据集**

数据集涵盖：ToolMaze（多模型多任务，交叉验证）、τ-bench Retail（状态化零售环境）、AppWorld（多样化 API 交互）、基于真实生产故障分类的 incident‑derived fault suite、以及 StableToolBench 公共 API 响应记录。

**📈 对比分析**

通过预先冻结的任务集进行配对统计（McNemar、Bootstrap 任务簇），结果显示在高失败率情境下完成率提升约 17–28%，在 τ‑bench 上提升 12–14%，相比传统 reviewer 仅提升 3–4%，但在低失败率情境下无显著收益。

**⚠️ 局限性**

局限性：依赖合约词汇范围，无法检测所有隐式错误；评估仅在模拟故障，未见真实部署效果；误报可能导致无谓动作；对抗性工具、隐私或不可逆操作的安全保障尚未证明。

---

## 56. CoToGrasp: Contact-Topology-Conditioned Dexterous Grasp Synthesis via Canonical Workspace Learning

**arXiv ID:** 2608.19776 | [PDF](https://arxiv.org/pdf/2608.19776v1)

**作者:** Julien Merand `[一作]` (Université Paris-Saclay), Mathieu Grossard `[通讯]` (Université Paris-Saclay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于接触拓扑的多指抓取生成框架CoToGrasp，通过对象无关的训练学习手掌中心的接触潜在空间，实现对不同功能拓扑（精细、力量、特定工具）的多样、稳定抓取的生成。

**💡 创新点**

核心创新在于将接触拓扑与物体几何解耦，构造基于手掌的特征工作空间，并利用自注意力+CVAE在对象无关环境下学习手掌接触潜在空间，完成零样本推理并避免模式崩塌；同时提出标签一致性与力闭合验证的双重验证管道。

**🔧 技术方法**

采用DGCNN提取局部几何特征、kNN投影到规范工作空间、Transformer+Set Transformer建模空间关系、条件变分自编码器生成接触掩码，以及能量最优化对齐手掌姿态与多尺度正则化损失。

**📊 数据集**

训练阶段仅使用手掌点云和21种拓扑标签生成210k样本；评估阶段在公开DexGraspNet（多Dex）数据集以及YCB物体的真实实验中验证。

**📈 对比分析**

与未加拓扑约束的基线（DFC、GenDexGrasp、DRO-Grasp、GOAG）以及拓扑感知基线Dexonomy进行对比，CoToGrasp在语义熵H_TC和拓扑符合度TC上显著提升（TC提升至30%+），同时保持物理成功率与生成速度，并在真实 Allegro Hand 上实现多种拓扑抓取。

**⚠️ 局限性**

仍受限于手掌模型的映射关系，某些高约束拓扑（如五指M6）在四指手掌上不可实现；依赖手掌点云采样质量；对极端物体形状或不完整点云的鲁棒性待进一步验证。

---

## 57. HYDRA: A Heterogeneous Chiplet DSE Framework for Serving Dynamic Hybrid LLM Workloads

**arXiv ID:** 2608.19395 | [PDF](https://arxiv.org/pdf/2608.19395v1)

**作者:** Jiahao Lin `[一作]` (University of Wisconsin--Madison), Umit Ogras `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HYDRA 框架，对异构 Chiplet 系统进行设计空间探索，联合优化芯片组合、放置、动态请求批量、弹性任务调度及快速性能估计。

**💡 创新点**

创新点包括：1) 针对 Transformer–Mamba 混合 LLM 的异构 Chiplet 架构设计；2) 两阶段通信感知放置 + 动态批量 + 弹性调度策略；3) 使用连续时间马尔可夫链与流式队列模型的快速性能估计器，实现高效剪枝与排序。

**🔧 技术方法**

使用技术包括：Chiplet 库与 NoI 拓扑建模、事件驱动仿真、弹性调度、动态批量、连续时间马尔可夫链估计、流式队列模型、Ramulator/CACTI 进行内存/功耗建模、CPU 级仿真等。

**📊 数据集**

采用 ArXiv‑4K、Bilingual Web Books、LongWriter‑6K、LMSYS‑Chat‑1M 四个数据集，覆盖长预填/短/长解码等多种推理模式。

**📈 对比分析**

与传统静态批量+静态调度+默认放置基线相比，HYDRA 在 12 个模型/数据集组合上平均提升 1.55× 吞吐量、降低 43.7% 首 token 延迟；单个工作负载吞吐量提升可达 2.3×；Markov 估计器将 DSE 时间从数天缩短至分钟级。

**⚠️ 局限性**

局限性：仅评估吞吐量与 TTFT，未考虑功耗、热与可靠性；模型对硬件参数假设强，需进一步验证；目前仅在特定芯片组与工作负载上测试，跨平台泛化尚待探索。

---

## 58. StreamSoccer: Event-Driven Memory for Streaming Soccer Commentary

**arXiv ID:** 2608.19723 | [PDF](https://arxiv.org/pdf/2608.19723v1)

**作者:** Chenxi Shao `[一作]` (East China Normal University), Changbo Wang `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了StreamSoccer，一种基于事件记忆的实时足球解说系统。

**💡 创新点**

首次将事件生命周期作为中间表示，支持当前事件、近期窗口和历史记忆三轨解说，并通过规则辅助调度实现自适应说话模式。

**🔧 技术方法**

结合冻结的Qwen3-VL视觉编码器、可学习的事件记忆更新、记忆到文本的投影、LoRA微调的语言模型以及规则+检索辅助的说话调度。

**📊 数据集**

构建了基于SoccerNet动作标注与MatchTime解说的三轨实时解说数据集，约2.8万条样本。

**📈 对比分析**

在共享输出锚点下与多种流式VLM、足球专用模型对比，StreamSoccer在当前事件轨道CIDEr为38.62、历史记忆轨道为17.39，且原始视频的RTF p95保持在0.10–0.22，未随比赛历史增长。

**⚠️ 局限性**

对近期多事件聚合的性能仍相对较弱；系统依赖规则调度与固定事件边界，极端比赛节奏下可能误判事件结束。

---

## 59. Scaffolding Minds: Optimizing Latent Visual Target Representations for Multimodal Reasoning

**arXiv ID:** 2608.19669 | [PDF](https://arxiv.org/pdf/2608.19669v1)

**作者:** Haoqiang Kang `[一作]` (Google DeepMind), Ed H. Chi `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Scaffolding Minds 两阶段框架：第一阶段学习可优化的 scaffolding 编码器以产生针对推理任务的潜在目标；第二阶段引入可学习的高斯潜在策略（Scaffolding RL）对潜在块进行残差采样，实现探索；

**💡 创新点**

创新点：1）用可学习的 scaffolding 编码器替代传统的冻结视觉编码器，使潜在目标直接通过任务损失端到端优化；2）将潜在空间建模为可学习均值与方差的高斯分布，采样残差动作，从而在潜在空间进行奖励驱动的探索，克服先前方法的确定性正则化限制；

**🔧 技术方法**

技术手段：潜在视觉推理（连续潜在 token 插入）、跨注意力池化、任务损失监督、强化学习（GRPO + 学习的高斯策略）、Qwen2.5‑VL 作为 VLM 主干、两阶段训练（SFT + RL）

**📊 数据集**

数据集：FrozenLake 空间规划（8×8~32×32 网格）、九个视觉推理基准（V*、BLINK、MMVP、MMStar、CVBench、HRBench‑4K、HRBench‑8K、MME‑RealWorld‑Lite、Jigsaw）

**📈 对比分析**

对比方法：基线 SFT、SFT+GRPO、VLPO、图像生成基准（VPRL、DiffThinker）、工具调用/图像思考基准（DeepEyes、Thyme）、先前潜在推理基准（LVR、Mirage、CoVT、Monet、VaLR、SkiLa）。在 FrozenLake 上平均提升 9.5%（最难 32×32 提升 19%），在九大基准上平均提升 5.2%，相较图像生成方法提升约 10%；整体性能均显著优于现有最强对比基准；

**⚠️ 局限性**

局限性：训练时需辅助图像，若无丰富的 helper 图像效果有限；在极高分辨率或组合性强的任务（HRBench‑8K、Jigsaw）提升有限；RL 训练增加复杂度与不稳定性；潜在空间采样仍受高斯假设限制；整体计算成本略高于纯 VLM 推理。

---

## 60. SceneGTMM: A Conformal Mapping-based Scene-Aware Transferable GNN-Transformer Dual-Graph Interaction Framework for Map Matching

**arXiv ID:** 2608.19298 | [PDF](https://arxiv.org/pdf/2608.19298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 61. Magnetically Self-Sealed MR Haptic Actuator With PWM-Based Excitation and High-Fidelity Torque Control

**arXiv ID:** 2608.19635 | [PDF](https://arxiv.org/pdf/2608.19635v1)

**作者:** Dong Qiang `[一作]` (Imperial College London), Min Yu `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了磁性自密封的磁致流变（MR）旋转力矩驱动器，并开发了10 kHz低滞后PWM驱动与分层控制算法，实现高保真力矩渲染。

**💡 创新点**

创新点包括：① 在同一结构中集成磁性自密封与工作区域，避免持续电源密封；② 通过PWM频率优化与非线性背骨+残差滞后模型实现低滞后驱动；③ 采用分层反馈控制（逆背骨前馈、滞后补偿、PI + 滑模），大幅提升瞬态与稳态跟踪性能。

**🔧 技术方法**

使用的技术：磁静力学仿真与Mason数分析、10 kHz PWM驱动、FPGA+实时控制、非线性模型识别、分层层次控制、边界层滑模与PI反馈、实验平台与力矩传感。

**📊 数据集**

所用数据集：实验测得的力矩-输入特性、基于脑注射生物力学模型的压力-力矩参考以及1.5 h连续操作的温度与力矩记录。

**📈 对比分析**

对比方法：与PID、FF‑PI两种基线在方波、正弦、模型驱动等三类参考下进行实验；HHC将方波平均超冲从22.35 N mm降至5.05 N mm，稳态RMSE从6.39 N mm降至2.02 N mm，整体RMSE提升约68%；1.5 h连续实验温升<2.5 °C，保持跟踪精度。

**⚠️ 局限性**

限制：仍受PWM频率与热积累影响，残差滞后补偿需实验调参；对极端高速或高负载变化的鲁棒性未系统评估；系统尺寸、重量及长期在不同温度环境下的可靠性尚未彻底验证。

---

## 62. On the Applicability of Safety Nets: A Safety-By-Design Solution for Certifying Neural Networks

**arXiv ID:** 2608.20053 | [PDF](https://arxiv.org/pdf/2608.20053v1)

**作者:** Johann Maximilian Christensen `[一作]`, Sven Hallerbach `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并发布了针对航空碰撞规避系统 ACAS X 的安全网（Safety Net）架构，实现了对 HCAS 和 VCAS 子系统的全量表格检验，保证在离散输入空间上 100% 一致性。

**💡 创新点**

系统化评估激活函数、网络深度、宽度和编码方式对安全网尺寸的影响，并首次公开可复现的完整实现与数据，填补了此前安全网缺乏可验证细节的空白。

**🔧 技术方法**

采用全连接前馈网络与 ReLU/LeakyReLU/GELU 激活、one‑hot 或目标编码；利用 DeepPoly/Reluplex 等抽象解释与 SMT 工具对网络进行覆盖检验；k‑d 树实现 LUT 压缩；并结合 DevOps 与 W‑shaped 认证流程。

**📊 数据集**

直接使用开源 HCAS 与 VCAS 的 MDP 预先生成的离散化 Q‑value 表格，覆盖所有约 53,792 与 4,053,465 条输入点，作为训练/验证/测试全集。

**📈 对比分析**

通过遍历全量输入空间计算网络检索率并衡量 LUT 保存比例；实验显示 ReLU + 3~5 层、100 节点/层、one‑hot 编码可将 HCAS 系统总尺寸压缩至 4.76 MB、VCAS 至 222.83 MB，分别比原始 MDP 表低 3 倍与 1.2 倍；推理时间均 < 1 ms，满足 1 Hz 更新需求。

**⚠️ 局限性**

仅针对简化的 HCAS/VCAS 近似实现，未覆盖完整 ACAS X 的完整参数范围；LUT 仍占主导，占 99% 体积；未探究角度变量的周期编码、降维或剪枝等进一步压缩手段；以及对连续空间的量化误差未覆盖。

---

## 63. End-to-end Early Classification of Time Series in Non-Stationary Environments

**arXiv ID:** 2608.20044 | [PDF](https://arxiv.org/pdf/2608.20044v1)

**作者:** Aurélien Renault `[一作]` (Orange Research & AgroParisTech), Vincent Lemaire `[通讯]` (Orange Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个端到端的早期时间序列分类（ECTS）框架 DQeND，并在非平稳环境下对其进行评估

**💡 创新点**

首次系统比较分离式与端到端 ECTS 方法，证明联合学习表示、分类与触发可显著提升在分布漂移下的鲁棒性

**🔧 技术方法**

采用基于强化学习的 Deep Q‑Network 与 ELM/ROCKET 风格的随机卷积编码器，联合训练并在在线阶段进行 replay‑buffer 更新

**📊 数据集**

使用可控漂移的 MNIST‑1D 合成数据集（10 类、长度 100，按位置、噪声、类别等方式诱导漂移）

**📈 对比分析**

与分离式基线（Alert、ELECTS、EARLIEST 等）进行对比，评估累计成本、漂移集平均成本及原始集平均成本，结果显示 DQeND 在所有渐进与突变漂移场景下均实现最低累计成本并保持对原始概念的高保留性能

**⚠️ 局限性**

局限在于：在频繁的突变类别漂移中 RL 触发策略可能趋向过早预测；实验仅基于合成数据，缺乏真实场景验证；对不同成本函数或先验分布漂移的适应性未深入探讨

---

## 64. Active Spiking Perception: The Membrane Potential as a Belief State for Anytime 3D Point Cloud Recognition

**arXiv ID:** 2608.19232 | [PDF](https://arxiv.org/pdf/2608.19232v1)

**作者:** Akarsh Jain `[一作]` (Indian Institute of Technology Indore), Sayeed Shafayet Chowdhury `[通讯]` (Indiana University Indianapolis)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Active Spiking Perception（ASP），利用脉冲神经网络的膜电位作为贝叶斯后验信号，实现对3D点云的自适应观察与置信早停，并将此机制迁移到分割和图像视觉任务。

**💡 创新点**

把SNN膜电位视为决策控制器，配合轻量级Slice‑Selection Policy和Gumbel‑Softmax实现端到端学习；提供分布式无偏风险校准；证明膜电位递推等价贝叶斯滤波，并证明早停的风险上界。

**🔧 技术方法**

脉冲神经网络（LIF）、泄漏积分‑发放单元、Gumbel‑Softmax离散化、置信门限早停、贝叶斯滤波理论、前缀重计算证明、能耗建模、分布式校准（split‑conformal）。

**📊 数据集**

ModelNet10/40、ShapeNetPart、S3DIS Area 5、ImageNet‑100（用于 foveated 视图）。

**📈 对比分析**

与多种 ANN 与 SNN 基线在分类、分割和图像任务上比较。ASP 在 ModelNet10/40 上分别达 93.28%/90.62%，仅比最强 SNN 低 1.7 点；在 ShapeNetPart、S3DIS 分别得到 83.21/48.50 mIoU；在图像上取得 79.58% Top‑1、2.13 fixations，能耗比对照低 2.83×。整体精度略逊于最强 SNN/ANN，但提供可证风险的任意停止与能耗优势。

**⚠️ 局限性**

精度仍落后于最强 SNN/ANN，尤其在大模型时；仅在 M=16 才能显著降低工作量；Streaming 实现未在点云上实现；能耗优势受模型规模影响；对单个类别的不可识别问题；缺乏与固定顺序匹配容量的对照；未验证无膜政策的效果。

---

## 65. Scale-Separated Conditioning for Style-Encoder-Free Diffusion Stylization

**arXiv ID:** 2608.19719 | [PDF](https://arxiv.org/pdf/2608.19719v1)

**作者:** Jingtao Zhang `[一作]` (Georgia Institute of Technology), Zeming Liu `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无需专用风格编码器的扩散模型风格化框架（SEFS），通过低分辨率裁剪生成风格标记，并结合目标图像的边缘和分割信息进行内容控制，实现了目标几何与风格的分离。

**💡 创新点**

创新点在于：①利用随机低分辨率裁剪作为风格瓶颈，天然抑制全局布局信息，降低内容泄漏；②采用风格-去噪重归一化和跨层跳联接提升标记融合的稳定性和细节保留；③完全通过无监督单图像构造伪三元组训练，无需对齐的内容‑风格‑目标对，显著降低数据成本和训练复杂度。

**🔧 技术方法**

使用技术包括：扩散变换器（SD3）、冻结 VAE 编码器、Canny 边缘与 SAM 分割预处理、低分辨率（64×64）风格裁剪、可训练投影层、LoRA 微调、风格‑去噪重归一化、跨层跳联接以及基于 rectified‑flow 的训练损失。

**📊 数据集**

主要数据集为 WikiArt（约 40k 张未配对的艺术图像）用于训练；在 1,000 张固定跨图像内容‑风格对上进行评估。

**📈 对比分析**

与 StyleShot、CSGO 等基线进行对比，SEFS 在内容一致性（DINO/CLIP‑I）、风格相似度、泄漏诊断（DINO‑edge）和 FID 上均优于或相近，且时间开销相近；在人类偏好测试中，SEFS 在 64.3%–72.2% 的对比中被选中，显示出更好的整体质量。

**⚠️ 局限性**

局限性：低分辨率风格标记仅能捕捉局部颜色与纹理，难以传递依赖全局布局或符号元素的风格；对 Canny/SAM 的结构预处理依赖度高，若边缘或分割错误会影响内容保持；同时，压缩风格信息可能抑制某些细节的精确迁移。

---

## 66. PL-NBA: A Possession-level Universal Basketball Video Dataset Supporting Multiple Visual Understanding Tasks

**arXiv ID:** 2608.19646 | [PDF](https://arxiv.org/pdf/2608.19646v1)

**作者:** Yunhao Zhao `[一作]` (Beijing University of Technology), Changwen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了首个基于NBA进攻占位的完整视频数据集PL-NBA，提供11,000条完整进攻录像及3.16万条细粒度事件注释，并在此数据集上开展事件识别、视频字幕生成、时序动作定位和全新进攻倾向预测任务的基线实验。

**💡 创新点**

创新点在于：①将数据聚焦于完整进攻占位，保留连贯的比赛时序和战术上下文；②提供玩家姓名、事件描述、结果等多维标签，支持多任务学习；③首次引入“进攻倾向预测”任务，推动对比赛策略的前瞻性分析。

**🔧 技术方法**

技术手段包括：ResNet-18+运动增强模块用于事件识别；CLIP4Caption框架进行视频字幕生成；TriDet模型完成时序动作定位；基于Timesformer+Transformer的FANNTRA网络实现进攻倾向预测。

**📊 数据集**

使用自制PL-NBA数据集（来自2022–2025赛季60场NBA比赛），并在此数据集上对上述任务进行基准评估；同时对比已公开的其他篮球视频数据集如FSN、NBA、NSVA等。

**📈 对比分析**

对比实验表明：事件识别部分模型在部分高频事件上F1>0.8，但在视觉相似或因果区分难的事件上表现低于0.5；视频字幕指标（BLEU-4≈32，CIDEr≈134）显示模型能生成较为连贯的描述；时序动作定位在3ptShot上AP>90%，而Dunk仅≈25%；进攻倾向预测平均准确率约65%，其中2ptShot预测最佳。

**⚠️ 局限性**

局限性包括：数据集存在长尾分布，低频事件识别困难；缺乏运动轨迹与边界框等空间位置信息，限制了更细粒度动作定位与因果分析；进攻倾向预测基线仅使用视觉时间序列，未融入球员身份、场地几何等先验知识，准确率仍有提升空间。

---

## 67. LoRA-GA$^2$: Low Rank Adaptation with Multi-step Gradient Adaptive Alignment

**arXiv ID:** 2608.19800 | [PDF](https://arxiv.org/pdf/2608.19800v1)

**作者:** Haonan He `[一作]` (University of Science and Technology of China), Xinyue Fan `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大型模型的参数高效微调，提出 LoRA‑GA² 通过多步梯度探测实现自适应秩分配与 SVD 初始化；

**💡 创新点**

创新点在于：① 用轻量级探针捕获多步梯度信息，避免单步梯度的短视；② 结合梯度敏感度与有效秩的双重评分实现更合理的秩分配；③ 采用基于累积梯度的 SVD 初始化提升对齐效果；

**🔧 技术方法**

主要技术包括：LoRA 低秩适配器、AdaLomo 优化器的探针、谱分析（有效秩）、敏感度评估、SVD 初始化；

**📊 数据集**

实验数据集涵盖 NLP（GLUE/CoLA/MNLI 等）、算术推理与代码生成（GSM8K、HumanEval）、视觉分类（CLIP‑ViT‑B/16 + 7 个图像数据集）；

**📈 对比分析**

与 LoRA、LoRA‑GA、GoRA、RaLoRA 等基线对比，LoRA‑GA² 在 GLUE 上平均提升 0.66 分，在 GSM8K 上提升 1.03 分，在 HumanEval 上提升 0.87 分，在 CLIP 上平均提升 0.63 分，且几乎不增加推理成本；

**⚠️ 局限性**

主要局限：需要额外的探测阶段（虽耗时少但仍有开销）；使用统一的 λ 超参数，可能不适用于所有层/任务；

---

## 68. ReCache: Efficient KV Cache Reuse and Compression for Tool-Augmented LLM Agents

**arXiv ID:** 2608.19662 | [PDF](https://arxiv.org/pdf/2608.19662v1)

**作者:** Yichu Fang `[一作]` (Shanghai Jiao Tong University), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReCache框架，对代理式LLM中工具/技能的KV缓存进行资源级别的重用与压缩。

**💡 创新点**

创新点包括资源级注意力、基于贡献的结构剪枝以及面向字段的语义剪枝，显著减少KV存储与计算。

**🔧 技术方法**

使用资源级注意力、结构与语义剪枝、微调Qwen3-4B/1.7B以及多层KV头组访问策略。

**📊 数据集**

使用了七个公开工具/技能使用数据集构建的统一基准，包含分布内与资源异构的OOD拆分。

**📈 对比分析**

与密集注意力基线相比，Inv-F1保持82.4%（仅低0.1%），首词推理时间提升3.655倍，KV内存减少92.43%，注意力加速1.423倍。

**⚠️ 局限性**

局限于两款Qwen3后端，未扩展至更大模型或冻结模型，且假设资源语义相对稳定，难以处理高度动态或跨资源依赖的环境。

---

## 69. String Rewriting Systems: Brief Introduction and Sample of Open Problems

**arXiv ID:** 2608.19397 | [PDF](https://arxiv.org/pdf/2608.19397v1)

**作者:** Assaf Kfoury `[一作]` `[通讯]`, Assaf Kfoury

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了字符串重写系统（SRS）的基本定义、性质及其在计算机科学中的重要性，并列举了一系列关键的未决开放问题；

**💡 创新点**

通过系统地分类讨论一规则、少规则系统的终结性、可达性、归约性质等问题，提出了可判定性边界与研究方向的框架；

**🔧 技术方法**

主要运用了形式语言理论、图论与归约系统分析等理论工具，利用SRS的图模型、归约图与critical pair等概念进行讨论；

**📊 数据集**

本篇为理论综述，未使用实验数据集；

**📈 对比分析**

由于缺乏实验，对比方法和性能指标未被评估，主要依赖已有理论证明与复杂度分析；

**⚠️ 局限性**

局限在于尚未给出一条完整的可判定性边界，许多开放问题仍未解决，并缺乏统一可实现的算法框架。

---

## 70. CVSD-Reg: Cross-Modal Visual Semantic Prior Distillation for Robust LiDAR Registration

**arXiv ID:** 2608.19536 | [PDF](https://arxiv.org/pdf/2608.19536v1)

**作者:** Eunsoo Im `[一作]`, Seunghwan Hong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CVSD-Reg 框架，先用视觉语义先验通过跨模态蒸馏学习 LiDAR 描述子，再在此基础上进行对应学习和位姿回归，最终实现全 LiDAR 推断的全局点云配准；

**💡 创新点**

创新点包括：1）首次将 DINOv2 视觉语义先验通过 hyperspherical 对齐与 rigid‑view consistency 蒸馏到 LiDAR，提升跨传感器鲁棒性；2）将语义学习与配准细化分离为两阶段训练，保持单传感器性能同时实现零 shot跨传感器；3）引入 density‑aware dropout、soft SE(3) invariance 等正则，构建对点云稀疏与扫描模式变化具备抗干扰的描述子；

**🔧 技术方法**

使用技术：DINOv2 视觉模型（teacher），Point Transformer V3（student），跨模态对齐（bias‑free 投影 + hyperspherical loss），InfoNCE 视角一致性，soft SE(3) invariance，密度感知随机 dropout，密集 superpoint 采样，confidence‑weighted MNN 对应，带权 Kabsch 估计，RANSAC + LGR 优化；

**📊 数据集**

实验数据集：KITTI odometry、nuScenes、HeLiPR（含 Ouster‑128、Velodyne‑16、Livox Avia、Aeva FMCW 等多种 LiDAR 传感器），并用同步图像-激光点云进行预训练；

**📈 对比分析**

与 9 种基线（FPFH+TEASER、KISS‑Matcher、RAP、CAST、GeoTransformer、UGP、PARE‑Net、BUFFER‑X 等）在同一实验配置下对比；在 KITTI 上 SR@0.5 m/1° 97.7%，nuScenes 99.0%，HeLiPR 全局 99.3%（含 97.3% on 16‑beam，100% on Avia、Aeva），相较 BUFFER‑X 在稀疏传感器提升 44 pp；同时保持单传感器精度不变；

**⚠️ 局限性**

局限性：预训练需同步标定的图像‑LiDAR 数据，无法完全无标定；对训练集的多样性依赖较大；在极端扫描稀疏或极大视角变化下仍可能出现失败。

---

## 71. EXIMO: VLM Guided Exploration of VLA Policies

**arXiv ID:** 2608.19891 | [PDF](https://arxiv.org/pdf/2608.19891v1)

**作者:** Bhavya Sukhija `[一作]`, Martin Riedmiller `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出三步训练流程（VLM驱动的数据收集 → 监督式微调 → 在线残差RL），将预训练的视觉语言动作（VLA）政策扩展到长周期组合任务。

**💡 创新点**

创新点在于利用大规模视觉语言模型（VLM）对任务进行语义拆解并实时引导VLA，随后通过蒸馏将VLM指导直接注入VLA，既提高样本质量又消除了推理时对VLM的依赖，并在此基础上使用残差强化学习进一步提升性能。

**🔧 技术方法**

技术包括Gemini等VLM进行任务拆解，基于PaliGemma+diffusion的GROD VLA进行行为克隆，监督微调（SFT）以及MPO算法实现的残差在线RL。

**📊 数据集**

使用Aloha机器人模拟平台上构造的22个需要推理与技能链的操纵任务作为数据集，采集的成功轨迹由VLM生成并在环境内置的成功检测器中过滤后用于训练。

**📈 对比分析**

与基线VLA无VLM调度、VLA+VLM调度、以及仅RL微调等方法对比，实验显示VLM驱动的探索显著提升成功率、缩短回合长度，SFT后性能进一步提升，最终在线RL提升成功率并显著提高样本效率。

**⚠️ 局限性**

局限性包括：依赖环境真值成功检测器；VLM在推理阶段仍需额外计算资源；仅在离线蒸馏框架下验证，未探索在线蒸馏或多任务泛化；实验仅在模拟环境中完成，缺乏真实硬件验证。

---

## 72. Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?

**arXiv ID:** 2608.19833 | [PDF](https://arxiv.org/pdf/2608.19833v1)

**作者:** Aleksandr V. Petrov `[一作]` (Spotify), Mounia Lalmas `[通讯]` (Spotify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了序列推荐基准是否需要高阶序列建模，提出并评估了两种仅基于配对转移的容量探针（SeqRules和PCTM）

**💡 创新点**

通过比较Transformer模型与强大的recency‑weighted pairwise探针，揭示大多数常用基准实际上并不需要高阶建模，提供了一种评估基准真实性能的新方法

**🔧 技术方法**

使用SeqRules的稀疏规则挖掘与PCTM的贝叶斯平滑加权多源专家模型，且对比MC、FMC、FMC+、SAS+、eSASRec等模型，并在eSASRec协议下重现实验

**📊 数据集**

Amazon Beauty、Sports、Toys、MovieLens‑1M、MovieLens‑20M

**📈 对比分析**

在相同的数据拆分与评估协议下，将Pairwise Envelope（SeqRules或PCTM最佳得分）作为基准，发现除ML‑20M外Transformer在多数数据集上不显著优于探针；在Amazon数据集上探针甚至超过eSASRec 15–38%，但在ML‑20M上仍保留27.3%优势

**⚠️ 局限性**

仅测试公开数据集，未涵盖工业数据、学习式recency核或相关历史聚合，实验范围和模型设计限制了结论的普适性

---

## 73. Optimality and Trade-offs in Fast BFT SMR (Extended Version)

**arXiv ID:** 2608.19629 | [PDF](https://arxiv.org/pdf/2608.19629v1)

**作者:** Neil Giridharan `[一作]` (University of California, Berkeley), Pierre Sutra `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了领导者无关的拜占庭容错状态机复制协议，并给出了n≥5f+1的最优复制因子以及对应的协议设计。

**💡 创新点**

创新点在于证明了n≥5f+1是BFT SMR的必要与充分条件，并设计了在该复制因子下实现两消息延迟冲突无关命令执行的协议，同时给出更高复制因子n≥7f+1的简化协议，展示了复制因子与恢复效率的权衡。

**🔧 技术方法**

主要使用了依赖图一致性、快速路径与恢复路径的交互、投票与视图变更、验证阶段以及在更高复制因子下利用快速投票交集的技术。

**📊 数据集**

论文未使用任何真实数据集，主要是理论分析和协议设计。

**📈 对比分析**

通过与现有领导者驱动协议（PBFT、HotStuff）和其他拜占庭快路径协议（FaB Paxos、Flutter等）的比较，证明了在同步冲突无关执行中可实现两消息延迟，同时在错误恢复时维持安全性。性能上在最佳情况下与领导者无关的 Crash‑fault 协议相近，但在恢复阶段需要额外的验证。

**⚠️ 局限性**

局限性包括对同步网络的强假设、较高的复制因子导致资源开销、恢复阶段的复杂性（尤其是 n≥5f+1 方案），以及没有给出实测性能或实现细节。

---

## 74. Does Listening Matter? Backchanneling and Nodding in AI Clone

**arXiv ID:** 2608.19527 | [PDF](https://arxiv.org/pdf/2608.19527v1)

**作者:** Koji Inoue `[一作]` (Kyoto University), Shunichi Kasahara `[通讯]` (Sony Computer Science Laboratories)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在AI克隆中加入实时背音与点头等听觉反馈，并评估其对用户感知的影响。

**💡 创新点**

创新点在于把听觉反馈作为克隆真实性的关键维度，首次将多模态听觉行为纳入AI克隆评估。

**🔧 技术方法**

使用的技术包括：ASR（Deepgram）、LLM（GPT‑4.1）、TTS（Cartesia）以及基于VAP的连续背音与点头预测模型（MaAI）。

**📊 数据集**

使用的数据集为首位作者的语音样本（用于声纹克隆）和35名日语母语学生的对话录音，未公开通用数据集。

**📈 对比分析**

通过对照实验（With‑feedback vs. Without‑feedback）进行主观评估，显著提升了Q3（注意力）、Q6（真实性）和Q8（共同存在）的分数；客观对话行为指标未出现显著差异。

**⚠️ 局限性**

局限性包括：仅测试单一目标人物、未分离背音与点头对比、只在日语环境下验证、缺乏跨语言与跨文化泛化、模型未实现个性化听觉行为。

---

## 75. HealMed: Multilingual Evaluation of Large Language Models in Medicine

**arXiv ID:** 2608.19981 | [PDF](https://arxiv.org/pdf/2608.19981v1)

**作者:** Yingjian Chen `[一作]`, Irene Li `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了HealMed，一个经过专家审核的多语言医学评测基准，覆盖九种语言、三种任务（多选题、自然语言推断、开放式问答），并对14个大型语言模型进行系统性能比较。

**💡 创新点**

创新点在于：①提出双阶段专家评审的翻译审核流程，揭示翻译质量对评估结果的显著影响；②将多语言医学评测与跨语言性能差异系统化分析；③提供一个公开、可扩展的多语言医学基准框架。

**🔧 技术方法**

采用机器翻译（主要为谷歌/OpenAI API）、双专家人工评审、LLM-as-judge评估框架，以及对14个不同类型（专有、开源、医学专用）大型语言模型进行实验。

**📊 数据集**

使用来自九个源数据集（HeadQA、MedQA、MedExpQA、MMLU-Pro、BioNLI、MedNLI、ExpertQA-Bio、ExpertQA-Med、LiveQA）的1,000条例子（共9,000条），覆盖英语、德语、西班牙语、葡萄牙语、日语、中文、泰语、斯瓦希里语和祖鲁语。

**📈 对比分析**

通过对各语言的宏观平均准确率/LLM-as-judge分数进行比较，评估专有模型在高低资源语言间的稳定性，并与开源与医学专用模型进行对比。结果显示专有模型整体更准确、跨语言差距最小；开源和医学专用模型在低资源语言上表现显著下降。

**⚠️ 局限性**

局限性包括：评估仅覆盖单轮任务，未涉及多轮对话或更复杂临床情境；数据量相对有限，可能无法充分覆盖医学专业与语言变体；翻译质量评估缺乏专门的医学翻译指标，人工评审样本规模受限。

---

## 76. PRAXIS: Graph-Grounded Tacit Knowledge for Domain Code Generation

**arXiv ID:** 2608.19784 | [PDF](https://arxiv.org/pdf/2608.19784v1)

**作者:** Xue Jiang `[一作]` (Peking University), Yihong Dong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过模拟人类开发流程，在目标代码库中进行离线实践，提取并结构化隐性知识，随后在推理时主动注入这些知识以提升域代码生成性能。

**💡 创新点**

创新点包括：①将隐性知识视为深埋于代码依赖图中的跨组件约束；②通过开发实践挖掘隐性知识；③将知识绑定到代码依赖图节点并沿依赖路径传播；④主动注入而非被动检索，克服代理自身缺乏“缺失知识”感知的难题。

**🔧 技术方法**

技术主要包括：LLM 代理（如OpenHands、SWE-Agent）、ReAct式交互、差分测试驱动的实践循环、结构化程序记忆（四元组），代码依赖图构建与传播、冲突消解与置信度更新。

**📊 数据集**

使用的基准数据集包括 KoCo-Bench（面向域的代码生成）和 AInsteinBench（科研项目的 bug 修复），并在这些数据集上进行实验。

**📈 对比分析**

与多种基线（OpenHands、SWE-Agent、OpenCode、OpenCollab、SWE-Exp、Trace2Skill）比较，PRAXIS 在 Pass@1 与 AvgPassRatio 上均优于所有对比方法，单一实践函数即可显著提升性能，且性能随实践量和在线演化而持续上升。

**⚠️ 局限性**

局限性包括：①离线实践与真实人类开发流程仍有差距，可能导致部分隐性知识缺失；②实验仅覆盖有限领域与代码库，泛化到更复杂或完全不同的项目仍待验证；③对 LLM API 的高成本与对超参数调优的依赖；④知识传播深度与置信度阈值等超参数需手动设置。

---

## 77. When AI Writes, Who Gets Cited? Evidence of Citation Monoculture Across Language Models

**arXiv ID:** 2608.19230 | [PDF](https://arxiv.org/pdf/2608.19230v1)

**作者:** Sina Alemohammad `[一作]` (University of Texas at Austin), Zhangyang Wang `[通讯]` (University of Texas at Austin)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个专门测量论文引用选择的基准，排除检索、声望与伪造信息，只保留真实论文的标题与摘要，并限制模型在面板中最多引用10篇。

**💡 创新点**

创新点在于通过构造无偏见的随机面板与严格的引用预算，揭示了多家生成模型共享的“偏好映射”导致的引用集中（citation collapse）现象，并与人工专家对比验证其普适性。

**🔧 技术方法**

技术主要包括使用生成式语言模型在固定面板上做引用决策、统计模拟空缺模型（indifference null）与真实模型的差异量化，以及通过多轮递归实验评估偏好随时间演变的稳定性。

**📊 数据集**

使用的数据集为120篇真实论文构成的面板；每次实验随机抽取30篇，去除了作者、年份与引用计数信息，保证了面板内容的随机性与匿名性。

**📈 对比分析**

比较方法是将模型在同一面板上的引用行为与完全无差别选择的空缺模型进行对比；结果显示模型在引用集中度与留白率上显著超越空缺模型，而人工专家的表现与无差别选择相近。

**⚠️ 局限性**

局限性包括实验仅覆盖单一主题与任务格式，缺乏检索、出版和作者写作流程等实际场景；数据量相对有限，且未能完全排除专家可能携带的隐性偏好。

---

## 78. Frequency-Aware Continual Learning for Smart Contract Vulnerability Detection with Large Language Models

**arXiv ID:** 2608.19680 | [PDF](https://arxiv.org/pdf/2608.19680v1)

**作者:** Tenghui Huang `[一作]` (Guangdong University of Technology), Dong In Kim `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种三阶段流水线，针对大模型在智能合约漏洞检测中的持续学习场景，先用频域低秩适配器FA‑LoRA实现参数高效适配，再用忘记感知重放FAR减轻灾难性遗忘，最后用Anchor‑Protected Progressive Merging (APPM)将多个任务适配器合并为单一模型；

**💡 创新点**

创新点在于将频域信息引入低秩适配器、基于损失动态的样本优先重放以及anchor‑protected频域门竞争的无数据合并策略三者耦合，形成一个完整的高效持续学习框架；

**🔧 技术方法**

核心技术包括频域低秩适配（FA‑LoRA）、基于样本损失的优先重放（FAR）以及anchor‑protected权重融合+频域门竞争的模型合并（APPM）；

**📊 数据集**

使用DIVE基准数据集（22,330条智能合约，8类漏洞标签）按时间划分为四个任务进行实验；

**📈 对比分析**

与多种PEFT、CL与合并基线比较，FA‑LoRA在单任务下实现Micro‑F1≈0.82，FAR在四任务序列中平均Micro‑F1≈0.8022，APPM合并后Micro‑F1≈0.8085，均接近单任务上限；

**⚠️ 局限性**

局限性包括：仍需在固定大模型上训练，对更大规模LLM的迁移性未知；对任务边界划分和缓冲容量敏感；合并后对早期任务的精度仍有轻微下降；缺乏在异构边缘/云环境下的实验验证。

---

## 79. Energy-Efficient Visual Inspection with FFT-Based CNNs and Adaptive Floating-Point Quantization

**arXiv ID:** 2608.19837 | [PDF](https://arxiv.org/pdf/2608.19837v1)

**作者:** Lukas Krupp `[一作]` (RPTU University Kaiserslautern-Landau), Norbert Wehn `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在工业CPU‑FPGA平台上使用FFT卷积和自适应FP8量化进行CNN推理的低精度浮点算术，并提出两种FPGA优化方法。

**💡 创新点**

首次实验评估自定义低位浮点格式在FFT卷积CNN中的表现，并提出在FFT内部进行逐阶段偏移调整与按层指数偏移优化，提升精度且不改动数据通路宽度。

**🔧 技术方法**

使用FFT基卷积、后训练FP8（E4M3）量化、Progressive Bias Adjustment、权重缩放、贝叶斯优化的层级指数偏移、序列化radix‑2^2 SDF FFT模块以及LeNet‑5加速器。

**📊 数据集**

使用工业X射线缺陷检测数据集，共3208张圣诞日历的X射线图像，标注为缺陷/无缺陷。

**📈 对比分析**

与FP32标准和CPU单核推理比较；在权重缩放+层级偏移下精度达到84.13%，相较于FP32降幅仅约13%，FPGA功耗从53.73W降至0.37W，能耗提升约2.5倍，延迟为1.91ms。

**⚠️ 局限性**

仍有约13%的精度缺口；仅使用3位尾数、4位指数；未针对实际值分布进行动态偏移；未探索更大尾数、指数位宽和部分重配置等方向。

---

## 80. Hybrid Feedback Sampling for Sample-Efficient Model Predictive Control

**arXiv ID:** 2608.19443 | [PDF](https://arxiv.org/pdf/2608.19443v1)

**作者:** Chaoyi Pan `[一作]` (Carnegie Mellon University), Guanya Shi `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了混合反馈采样的模型预测控制算法FS‑MPC，用以提升采样基MPC在高维、开放式不稳定系统中的样本效率和数值稳定性。

**💡 创新点**

创新点包括：①证明最优采样分布等价于最优反馈策略，从而提出反馈采样方法；②引入基于系统稳定性和计算预算动态调整局部/全局采样比例的混合采样框架；③兼容iLQR与RL反馈控制器，使算法同时适用于线性可线性化与非可微系统。

**🔧 技术方法**

技术手段包括：采样基MPC/MPPI、iLQR、强化学习（PPO）、混合局部/全局采样、反馈采样、软max更新、仿真器（Brax、MuJoCo）、真实机器人实现（Unitree H1 + Vicon MoCap）。

**📊 数据集**

实验使用的任务与数据集包括：Acrobot、Quadrotor、Quadruped Hill、H1‑2 PNP、Allegro 等仿真控制任务；Brax仿真器中的非可微系统；真实环境下的Unitree H1全尺寸机器人，配备Vicon外部MoCap进行状态估计。

**📈 对比分析**

与标准MPPI、iLQR、PPO反馈控制器进行对比。FS‑MPC在不稳定任务中累计成本降低约43.4%，在仿真任务中始终优于MPPI、iLQR，并在真实机器人上成功完成行走与搬运任务，MPPI因采样方差过大而失稳失败。

**⚠️ 局限性**

局限性包括：依赖精确状态估计（MoCap延迟导致成本方差增大）；混合采样比例需要手动调参；对高维连续问题的梯度基方法仍有一定局限，未来需集成价值函数或模型估计以进一步提升效率。

---

## 81. ADAPT: Physics-Aware Diffusion-based World Models for Adaptive Predictive Transferable HVAC Control

**arXiv ID:** 2608.19804 | [PDF](https://arxiv.org/pdf/2608.19804v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 82. FleetSieve: Decision-Critical Profiling for SLO-Aware LLM Fleet Configuration

**arXiv ID:** 2608.19659 | [PDF](https://arxiv.org/pdf/2608.19659v1)

**作者:** Huang Cheng `[一作]` (Meta), Aubert Li `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个决策导向的LLM服务配置测评框架 FleetSieve，按资源耦合决策优先选测量，减少不必要的性能测试。

**💡 创新点**

创新点在于：①联合容量与尾部延迟建模，避免尾部失效的高吞吐配置；②使用决策差值驱动的采样准则，直接衡量测量对最终资源分配的影响；③给出条件决策证书，明确何时可停止测量。

**🔧 技术方法**

采用经验区间推断、价值信息评估、基于 GPU‑秒的采样成本衡量，以及 vLLM+FP8 量化的推理堆栈。

**📊 数据集**

使用 Azure 公共 LLM 推理轨迹（会话与代码）和 BurstGPT 公开对话轨迹进行测评。

**📈 对比分析**

与随机、固定网格、共享特征不确定性、最大不确定性、目标学习、价值信息、受限贝叶斯优化等方法对比，FleetSieve 在 31B H100 平台下 GPU‑秒整体减少 6.9%，Chat 需求下减少 21.5%，在 16 GPU 资源分配中提升最大最小公平度 12.4% 并增加 1.93 requests/s。

**⚠️ 局限性**

局限性：结果高度依赖单一模型、量化方式、网络环境和工作负载；内部拐点假设和经验区间可能在容量曲线多峰或剧烈噪声时失效；实验仅覆盖单一平台，缺乏多样化验证。

---

## 83. RFWM: Physics-Guided World Model for Dynamic Wireless Radiance Field Generation

**arXiv ID:** 2608.19709 | [PDF](https://arxiv.org/pdf/2608.19709v1)

**作者:** Zijiu Yang `[一作]` (Zhejiang University), Qianqian Yang `[通讯]` (Zhejiang University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `14d48e9d-0069-4ad9-996a-1d5968216998` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了基于物理引导的世界模型RFWM，在未观测环境中生成动态三维射频辐射场。

**💡 创新点**

两阶段迁移训练结合Friis传播先验和六种细粒度物理正则化，实现场景与高度跨域的可迁移生成。

**🔧 技术方法**

采用视觉Diffusion Transformer（DiT）+ VAE、ControlNet、Friis先验、物理正则化等技术。

**📊 数据集**

使用115个3D环境生成的7,715条同步物理-射频序列的基准数据集。

**📈 对比分析**

与场景特定方法和共享Diffusion^2对比，ID上MSE降低约7dB、OOD约3dB，渲染速度0.77s/帧，整体性能显著优于现有方法。

**⚠️ 局限性**

主要限制为模型规模大（≈8B参数），训练成本高，对极端动态或低信噪比环境的适应性尚待验证。

---

## 84. Program Analysis for Adaptive Data Analysis

**arXiv ID:** 2608.19575 | [PDF](https://arxiv.org/pdf/2608.19575v1)

**作者:** Jiawen Liu `[一作]` (Boston University), Jonathan Ullman `[通讯]` (Northeastern University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了静态程序分析工具，用于上界估计自适应数据分析程序的适应性（自适应轮数）和查询次数，以指导选择合适的泛化保持机制。

**💡 创新点**

创新点在于正式化适应性概念为程序的量化属性，构建基于执行追踪的语义依赖图并赋权，然后提出算法对该图进行最优路径分析，给出精确的适应性上界。

**🔧 技术方法**

采用执行追踪语义、数据流与控制流分析、可达性边界分析以及图论的最长路径算法（SCC分解与DFS），实现了符号表达式的权重估计与适应性上界计算。

**📊 数据集**

主要使用人工合成的25篇示例程序（包括两轮、数轮、嵌套循环等结构）以及scikit‑learn中的九个经典数据分析任务作为评估基准。

**📈 对比分析**

在评估中，工具对大多数示例给出了紧凑的上界，平均运行时间不到一秒；但对深层嵌套循环的程序会出现性能瓶颈，可通过替代的轻量级权重估计来缓解。

**⚠️ 局限性**

主要局限在于权重估计缺乏路径敏感性，导致对某些程序（如多路径循环）产生过度保守的上界，并且对极其复杂的多层循环在时空复杂度上不易扩展。

---

## 85. Scale-Aware Pretraining of Time Series Foundation Models via Multi-Patch Token Alignment and Hybrid Masking

**arXiv ID:** 2608.20005 | [PDF](https://arxiv.org/pdf/2608.20005v1)

**作者:** Taihua Chen `[一作]` (Shandong University), Lizhen Cui `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一个尺度感知的时间序列基础模型SATS，利用多尺度补丁令其能够在异构频率数据上进行无监督预训练，并通过多补丁令牌对齐和混合遮蔽来提升表征一致性与泛化。

**💡 创新点**

提出尺度感知对齐机制（SA），通过均值吸引和最大值排斥在不同补丁尺度间对齐表征；引入混合随机与连续遮蔽策略（HM），在单一模型中同时捕捉细粒度与长程时序依赖。

**🔧 技术方法**

基于Encoder-only Transformer、RoPE、SwiGLU、RMSNorm、RevIN等；采用InfoNCE式对齐正则、均值/最大池化、对齐-分离损失以及随机/连续遮蔽技术。

**📊 数据集**

在LOTSA数据集上进行预训练；在LSTF、GIFT‑Eval、Monash三大基准上评估，涵盖多领域、多频率、多长度时序数据。

**📈 对比分析**

与Timer‑XL、Time‑MoE、Moirai、Chronos、VisionTS等现有基础模型对比；SATS_B在零射击、长周期预测、跨数据集泛化上均领先，MSE/MAE/MASE/CRPS均提升9%+，同时参数量仅70M（SATS_S 14M），效率提升65%以上。

**⚠️ 局限性**

对补丁尺寸的选择仍需依赖频率元数据；在极低频或极高频时序上可能出现信息瓶颈；缺乏对因果性与可解释性的深入探讨。

---

## 86. G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs

**arXiv ID:** 2608.19964 | [PDF](https://arxiv.org/pdf/2608.19964v1)

**作者:** Bhavya Gupta `[一作]` (University of California), Tajana Rosing `[通讯]` (West Virginia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将多车协作感知结果转换为可追溯的知识图谱(KG)，并在此基础上实现对象推理、运动预测、控制选择与轨迹规划；

**💡 创新点**

通过在KG中显式存储对象来源、可见性、置信度、冲突与规划关联等信息，保留了传统压缩融合方法丢失的证据结构，实现了任务条件下的延迟融合；

**🔧 技术方法**

使用知识图谱构造与维护技术、保守关联规则、特征扩展、KG查询任务头、轻量化学习模块；

**📊 数据集**

V2V-GoT-QA（基于V2V4Real的协同驾驶基准）用于感知、预测与规划任务评估；

**📈 对比分析**

与最先进的V2V-GoT方法对比，障碍推理提升42.2%，隐形对象发现提升12.3%，控制选择误差下降13.1%；轨迹预测与V2V-GoT相近，但通信量降低25.6倍；

**⚠️ 局限性**

受限于仅在短期轨迹预测上表现优异；对长期预测与更大规模车辆网络的适用性尚未验证；KG构造与查询仍需对通信延迟与算力资源做进一步优化。

---

## 87. A Layered Simplex Architecture for Large Alphabets

**arXiv ID:** 2608.19908 | [PDF](https://arxiv.org/pdf/2608.19908v1)

**作者:** Meir Feder `[一作]` (Tel Aviv University), Ruediger Urbanke `[通讯]` (EPFL)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“分层单纯形架构”（LSA）先验，并基于它构造了一个简单的贝叶斯估计器，用深度 L 通过乘积独立均匀抽样得到分布，随后对所有深度做平均，得到一个无需调参、可直接编码的预测器；同时推导出其对数损失（重叠）闭式表达式，并在理论上解析其与数据、字母表大小和深度的缩放关系；在多种合成（Zipf、Dirichlet 等）和真实（King James Bible）数据上进行实验，证明其与经典方法（Good‑Turing、Ristad、Laplace 等）竞争甚至超越。

**💡 创新点**

① 仅用深度 L 作为结构参数，构造了极其简单的层叠先验；② 通过乘积专家（product of experts）实现非均匀、稀疏且重尾的先验；③ 推导了可高效计算的重叠闭式公式；④ 在理论上得到重叠与符号发现率、字母表规模和深度的明确缩放律；⑤ 通过深度平均得到单一统一预测器，避免了深度选择。

**🔧 技术方法**

层叠单纯形采样、乘积专家归一化、贝叶斯混合、对数损失重叠的闭式求解、对符号发现率的概率分析、Monte Carlo 采样用于计算期望、深度平均实现策略、以及在序列压缩中使用的逐符号预测。

**📊 数据集**

合成数据：Zipf（α=0~5）、Dirichlet‑1/12 采样、阶梯分布、几何分布，支持大小 d=10³,10⁴,10⁶；真实数据：完整 King James Bible（N≈9.16×10⁵，词汇表 d=10⁵）。

**📈 对比分析**

与传统计数规则（add‑one、add‑half、Braess–Sauer）、Good‑Turing 混合、Ristad 先验以及自然占位器（oracle）进行对比；在每个目标上计算 KL 散度或在线重叠；结果显示：深度平均 LSA 在 8/11 个合成目标上匹配或优于最优传统方法，在真实文本上始终获得最低冗余；在极平坦目标（均匀分布）上仍略逊于 Good‑Turing，但差距可忽略。

**⚠️ 局限性**

① 深度 >1 的实验结果仅为数值模拟，缺乏严格理论证明；② 缩放律在有限范围（d≤10⁶, N≤10⁴）内验证，未对无限极限给出完整证明；③ 仅对 i.i.d. 目标和已知字母表大小的情形；④ 由于深度平均仅在有限 Lmax 之内实现，理论上对极深层情况的泛化仍待研究。

---

## 88. Cyber-Physical Systems for Accessibility and Ability Augmentation: Bridging Diverse Communities

**arXiv ID:** 2608.19422 | [PDF](https://arxiv.org/pdf/2608.19422v1)

**作者:** Shuchang Xu `[一作]` (MIT Media Lab), Pattie Maes `[通讯]` (MIT Media Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

组织了一场聚焦可访问性与能力增强的Cyber-Physical Systems研讨会，聚集了HCI、AI、可穿戴、机器人、XR、智能环境等多领域研究者和从业者；

**💡 创新点**

创新点在于跨学科融合、结合交互式演示与混合小组设计，强调真实部署与长期评估；

**🔧 技术方法**

主要技术领域涵盖可穿戴传感、机器人技术、扩展现实、智能环境感知与人工智能；

**📊 数据集**

未使用任何数据集，研讨会为知识与经验的交流与共享；

**📈 对比分析**

未进行方法比较或性能评估，研讨会侧重理论与实践讨论；

**⚠️ 局限性**

局限在缺乏实证研究与长期效果验证，需要后续开展真实环境评估与实验。

---

## 89. Contrastive Mixed Prompt Learning for Incomplete Multimodal Sentiment Analysis with Unseen Modality Combination

**arXiv ID:** 2608.20019 | [PDF](https://arxiv.org/pdf/2608.20019v1)

**作者:** Kaixin Xu `[一作]` (Zhejiang University), Meng Xi `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种 Contrastive Mixed Prompt Learning (CMPL) 模型，用于处理不完整多模态情感分析中的未见模态组合。

**💡 创新点**

创新点在于引入标签引导的对比特征学习、软路由混合提示以及三种提示对比学习策略，实现对未见模态组合的良好泛化。

**🔧 技术方法**

采用标签引导对比学习、提示混合学习、软路由器以及三种提示对比学习策略，并以 Qwen1.5‑1.8b 作为 LLM 背景。

**📊 数据集**

使用 CMU‑MOSI、CMU‑MOSEI 和 SIMS‑V2 三个公开多模态情感数据集进行实验。

**📈 对比分析**

与 Self‑MM、Cube‑MLP、DMD、DLF、TFR‑Net、MPLMM、MFMB‑Net、LNLN 等现有方法对比，CMPL 在大多数任务上平均提升约5.6% 准确率，且在 F1 等指标上表现最佳。

**⚠️ 局限性**

局限在于仍需依赖预训练 LLM 并且对极大规模模型的计算成本较高，未探索在更稀缺数据场景下的鲁棒性。

---

## 90. Calming Robot Pitches? Exploring the Influence of Robot Voice Pitch on Children's Stress Levels

**arXiv ID:** 2608.19826 | [PDF](https://arxiv.org/pdf/2608.19826v1)

**作者:** Nina G. M. van Roij `[一作]` (Utrecht University), Aoju Chen `[通讯]` (Utrecht University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了机器人声音音调对儿童在压力情境中的自我感受影响，使用Zenbo Junior II进行机器人引导的LEGO拼装游戏，并通过适配的CAM‑S量表测量儿童的压力水平。

**💡 创新点**

创新点在于将人类交流中低音调降低压力的理论迁移至人机交互，首次检验低音调合成声音对儿童压力调节的效果，并通过预验证评估音高与语速对舒缓感知的影响。

**🔧 技术方法**

技术手段包括：使用Google Cloud Text‑to‑Speech合成声音并通过PRAAT将音高±35Hz调制；搭建LEGO构建任务作为压力诱发情境；采用两项CAM‑S自评量表收集数据；使用JASP/R进行线性回归和统计检验。

**📊 数据集**

数据集为27名8–12岁荷兰儿童在实验中的自评压力数据；预验证阶段收集了6名成人对音高±20Hz及语速±10%调制样本的平静度评估。

**📈 对比分析**

采用两组实验（高音调组 vs 低音调组）进行比较，利用线性回归检验音调对CAM‑S分数的影响，结果未显著（p>0.05），低音调组甚至略高的压力分数，显示该音调调制在本实验中无显著效果。

**⚠️ 局限性**

局限性包括：样本量仅27人，压力诱发轻度且主观自评易受影响；缺乏生理指标支持；音调调制为静态单一参数，未考虑动态语调或多模态语音特征；机器人外观与音调不匹配可能影响效果；实验环境（家庭 vs 学校）异质性未充分控制。

---

## 91. LF-GICP: Parameter-Free Degeneracy-Aware LiDAR Odometry via a Voxel-Normal Localizability Field

**arXiv ID:** 2608.19522 | [PDF](https://arxiv.org/pdf/2608.19522v1)

**作者:** Eunsoo Im `[一作]` `[通讯]`, Eunsoo Im

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无参数、基于体素法向局部可定位性场的 LiDAR 里程计退化检测与加权策略，消除传统 GN‑Hessian 阈值化带来的环境依赖；

**💡 创新点**

创新点在于：①构建无正则化的体素法向局部可定位性场，并提出两个统计量 f₀ 与 λ₀，能同时区分退化与信息稀疏；②利用这两个量做中值门控，仅在真正退化时激活 Fisher‑信息加权；③整个系统只需两条 500 帧校准轨迹一次性得到所有阈值，实现真正的无环境参数化；

**🔧 技术方法**

核心技术包括体素化 GICP、法向统计与局部可定位性场、软 Fisher‑信息权重、临时中值门控与极限条件判别；

**📊 数据集**

使用多种数据集：KITTI、GEODE 城市/隧道、MulRan、HeLiPR、SubT-MRS，涵盖 4 种 LiDAR 传感器；

**📈 对比分析**

与 KISS‑ICP、GenZ‑ICP 及无退化 VGICP 进行对比，单一配置下在 KITTI 上相对平移误差最低（0.865%），在 GEODE 隧道中击败所有基线，在 MulRan 与 HeLiPR 平均误差上领跑四种传感器；

**⚠️ 局限性**

局限性：在完全均匀直线隧道等几何上不可观测方向上，单靠 LiDAR 仍会出现无限制的沿轴漂移，需外部信息（IMU、闭环或视觉）来补偿。

---

## 92. VideoRun2D Demo: Markerless Body Tracking for Biomechanical Analysis of Running

**arXiv ID:** 2608.19480 | [PDF](https://arxiv.org/pdf/2608.19480v1)

**作者:** Luis F. Gomez `[一作]` (Universidad Autonoma de Madrid), Enrique Navarro `[通讯]` (Universidad Politecnica de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究开发了VideoRun2D Demo，一套基于深度学习的无标记跑步运动学分析平台；

**💡 创新点**

创新点在于将多种前沿2D姿态估计模型融合并加入后处理算法以提升关节角度估计精度，并公开交互式Web演示平台；

**🔧 技术方法**

核心技术包括Vision Transformer与CNN-Transformer混合骨架网络（ViTPose、RTMPose）、SVR平滑及离群值修正、以及多模型晚期融合；

**📊 数据集**

使用了44名职业跑者共314次冲刺的视频数据集，包含925个跑步步态周期；

**📈 对比分析**

通过与人工标注的地面真值比较，单模型RMSE平均在5.30°–11.46°之间，融合后可降至约5.30°–6.88°，显示相较传统无标记系统更具竞争力；

**⚠️ 局限性**

限制主要为角度误差仍高于临床可接受阈值（2°–5°），并存在潜在的性别与种族偏倚、合成图像干扰及计算成本等问题。

---

## 93. Uncovering the Limits of Proof Sharing for Neural Networks

**arXiv ID:** 2608.19351 | [PDF](https://arxiv.org/pdf/2608.19351v1)

**作者:** Kanak Das `[一作]` (University of California, Riverside), Manu Sridharan `[通讯]` (University of California, Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对神经网络鲁棒性验证中的模板共享机制进行系统评估，并提出自动配置模板使用的技术

**💡 创新点**

通过引入联合稳定神经元指标揭示模板共享的有效性界限，并设计基于采样的自适应模板分配策略

**🔧 技术方法**

抽象解释、联合稳定神经元度量、采样估计、动态模板配置

**📊 数据集**

MNIST、CIFAR-10 数据集；包含标准、PGD、CURE、CPD 等多种训练方法的全连接与卷积网络

**📈 对比分析**

与无模板、固定模板（手工选层）两种基线比较；在 DeepZ/Box 抽象域下，平均加速约1.10~1.17倍，单例可达2.10倍；在低子占率场景避免1%-14%慢速

**⚠️ 局限性**

仅针对单一输入的局部鲁棒性验证，使用不完整验证器；模型假设层成本常数，未考虑全局鲁棒性或多输入情形；对多种抽象域的支持有限

---

## 94. Gallileo-4D: Frozen Backbone Ensemble for Dynamic 4D Reconstruction

**arXiv ID:** 2608.19743 | [PDF](https://arxiv.org/pdf/2608.19743v1)

**作者:** Nicolò Savioli `[一作]` `[通讯]` (OdaxAI Research), Nicolò Savioli (OdaxAI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在PhysAI Dynamic 4D Reconstruction Challenge中，作者通过冻结预训练的4D骨干网络，并在推理阶段使用三种解码配置（stride-3、水平翻转TTA、stride-1）进行集成，最终实现了第三名成绩，且未进行任何梯度更新。

**💡 创新点**

核心创新在于：① 发现并量化了局部验证与挑战分数的逆向关系，揭示了训练集覆盖度不足导致的特征退化；② 采用冻结骨干并在推理时通过多解码配置的加权融合获得显著提升；③ 通过两维网格搜索确定加权系数，证明集成效果具有平坦性，可在不同测试集保持稳健。

**🔧 技术方法**

技术手段包括：预训练4D backbone 4RC（DINOv2初始化的ViT-G），多解码策略（Temporal stride 3/1、水平翻转测试时增广），对齐尺度的停梯度公式，权重融合的凸组合，及局部验证与公共拆分的对比分析。

**📊 数据集**

数据集为Syn4D Benchmark，训练集仅覆盖四个渲染变体中的一种，测试集共128个192帧序列，包含四个场景族（antiquity, dream, gothic, office）及四个渲染变体。

**📈 对比分析**

与公开基线及其他参赛队伍比较，Gallileo-4D在公开分数上从0.512提升至0.553，私有分数从0.511提升至0.584，整体排名第三，显著优于大多数使用微调的方案。

**⚠️ 局限性**

局限性包括：① 仅在离线推理环境下可行，推理成本为单通道的三倍；② 依赖公共拆分的加权选择，可能在不同数据分布下失效；③ 未对多尺度推理或自适应训练做深入探索，仍有提升空间。

---

## 95. Reliable Financial Named Entity Recognition under Domain Shift

**arXiv ID:** 2608.19558 | [PDF](https://arxiv.org/pdf/2608.19558v1)

**作者:** Zihao Zheng `[一作]` (Washington University in St. Louis), Jiayu Long `[通讯]` (Washington University in St. Louis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了在财务命名实体识别（NER）任务中，模型对输入分布漂移的置信估计与选择性预测能力，探讨在SEC文件、财经新闻和普通社交媒体三种域间的性能差异。

**💡 创新点**

创新点在于将置信信号分为全序列概率、实体跨度概率、自一致性等多种形式，并系统评估它们在域漂移下的判别力与校准性，提出基于置信门限的分阶段部署策略。

**🔧 技术方法**

使用了BERT编码器+BIO标注头和指令调优的Qwen2.5 0.5B/1.5B生成模型，结合五种推理时置信信号（序列、跨度、类型、token概率与自一致性）进行评估。

**📊 数据集**

数据集包括FIN（SEC文件）、FiNER-ORD（财经新闻）以及TweetNER7（非财经社交媒体）三层次，统一映射为PER/ORG/LOC三类。

**📈 对比分析**

在所有域上进行比较，结果显示BERT在域内F1高达70%但在财经新闻和推文上骤降；生成模型在域内表现弱于BERT，但在域漂移时更稳健；自一致性与实体跨度概率在判别错误时优于全序列概率，且自一致性校准更好。

**⚠️ 局限性**

主要限制包括模型规模小（0.5B/1.5B）、训练数据有限（约1k句）、远域测试使用非财经推文导致难以真实反映社交媒体场景、以及对置信度校准的评估受样本量与分箱稳定性影响。

---

## 96. AvatarDynamizer: From Static to Dynamic Human Avatars via Generative Dynamic Textures

**arXiv ID:** 2608.19900 | [PDF](https://arxiv.org/pdf/2608.19900v1)

**作者:** Guoxing Sun `[一作]` (Max Planck Institute for Informatics, Saarland Informatics Campus), Marc Habermann `[通讯]` (Max Planck Institute for Informatics, Saarland Informatics Campus)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

将静态3D头像转化为可控的4D动态头像，在姿势变化时加入真实的表面细节（如衣物皱纹、阴影等）。

**💡 创新点**

提出纹理空间动态表面嵌入与通用高斯解码器，将动态纹理生成视为条件视频生成任务，兼容预训练视频扩散模型，既保持多视角一致性，又能捕捉高频细节。

**🔧 技术方法**

采用SMPL-X模板对人体进行姿势驱动，使用动态纹理编码器与通用高斯解码器将纹理映射为3D高斯粒子；动态纹理生成器基于LoRA微调的3D视频扩散模型；优化时利用可微光栅化与多视角重建。

**📊 数据集**

收集了新的大规模多视角动态数据集DynaHuman（58人、100台4K摄像头、27,000帧/人），并结合MVHPP、DeepCap、DDC等公开数据用于训练与评估。

**📈 对比分析**

与静态方法LHM、MV以及动态方法GAS、Vid2Avatar-Pro进行对比，在图像质量指标PSNR/SSIM/LPIPS、生成质量指标FID/FVD上均优于其他方法，重建指标略逊于VAP但在高频表面细节上表现更佳。

**⚠️ 局限性**

仍受限于SMPL-X模板，无法处理极端服装或拓扑变化；动态生成对预训练模型与数据多样性高度依赖，生成结果仍可能与真实动态存在差异。

---

## 97. Fourier is Frontier: Frequency-Aware Autoencoding for High-Fidelity Music Reconstruction

**arXiv ID:** 2608.19843 | [PDF](https://arxiv.org/pdf/2608.19843v1)

**作者:** Kangdi Wang `[一作]` (Alibaba), Jin Xu `[通讯]` (Alibaba)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在48kHz立体声音乐自动编码器中提出了频率感知的复数谱自编码器，并结合频率共享的周期性激活和基于双工理论的分频段修正器，提升了高频保真度与立体声一致性。

**💡 创新点**

创新点在于：①频率共享的周期性激活（F-Log）让每个频率 bin 共享激活参数；②分频段（低/中/高）幅度/相位修正器利用双工理论，显著降低高频误差和立体声崩塌；③在复杂 STFT 表示上进行端到端训练而非使用外部 vocoder。

**🔧 技术方法**

采用了复数 STFT 编码/解码器、频率共享周期激活、Band‑Aware Refiner（双工分频修正）、多尺度 STFT/CQT 判别器、IF/GD 频相损失以及多阶段训练策略。

**📊 数据集**

使用了 Song Describer 数据集（546 曲目）和约两百万公开音频（LAION‑DISCO‑12M + 10k 小时 48kHz 专业音乐）进行训练与评估。

**📈 对比分析**

通过与 LeVo 2、SA‑Open、SAME‑L 等开源 VAE 以及自身的 ablation（激活、分频修正）进行对比，主实验在 Song Describer 上取得 5/7 指标最佳（如 SI‑SDR、Spectral PAN、SPE 等），高频误差下降 19.4%，下游 Latent‑Diffusion 生成模型也在 12 项自动评估上均得分提升。

**⚠️ 局限性**

局限性包括：①模型仍依赖复杂 STFT，导致时域转化和推理成本较高；②在低频/中频细节上不如波形补丁模型；③需要大量训练数据和多阶段训练，部署与资源消耗相对昂贵。

---

## 98. GRACE: Grounded Reasoning via Adapter Composition and Evidence-Aware Calibration for Educational Visual Question Answering

**arXiv ID:** 2608.19355 | [PDF](https://arxiv.org/pdf/2608.19355v1)

**作者:** Xinjin Li `[一作]` (Columbia University), Yeyun Xu `[通讯]` (Texas A&M University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于教育元数据的参数高效适配框架（Grounded Reasoning via Adapter Composition and Evidence-Aware Calibration，简称GRADE）用于教育视觉问答；

**💡 创新点**

创新点在于利用可推断的教学状态（科目、技能、年级、视觉上下文、问题意图、选项结构）对语言前缀和视觉残差适配器进行分层组合，并引入证据感知的候选校准，显著提升多选题的准确率；

**🔧 技术方法**

使用了冻结的大规模视觉语言模型 Qwen2.5‑VL‑7B‑Instruct，训练轻量级前缀组合器、视觉适配器、元数据门控网络和选项校准器；

**📊 数据集**

在 ScienceQA 多模态多选题数据集上进行评估；

**📈 对比分析**

与同一基准模型的冻结版本、LoRA、共享适配器等基线相比，GRADE 在图像上下文子集从 88.7% 提升到 91.2%，整体准确率从 90.5% 提升到 93.1%；移除任一组件均导致 1.0–1.5 点的下降，验证了各模块的独立贡献；

**⚠️ 局限性**

局限在于仅验证于 ScienceQA，假设可获得教学元数据，缺乏对其他教育数据集、不同底座或对视觉区域的解释能力的评估。

---

## 99. Product Gap Mechanisms for Multi-Facility Location

**arXiv ID:** 2608.19633 | [PDF](https://arxiv.org/pdf/2608.19633v1)

**作者:** Jianhao Jia `[一作]` `[通讯]`, Jianhao Jia

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于连续间距乘积的随机机制Product‑Gap，用以在一条直线上定位k个设施，并证明其在k≥2时期望社会成本的近似比为2k；

**💡 创新点**

首次在无钱机制设计框架下给出多设施（k≥3）常数近似的定量结果，并揭示Product‑Gap在k=3时仍保持期望策略不受激励性，k≥4时失效；

**🔧 技术方法**

采用了基于间距乘积的加权选择、随机切割表示、最优聚类的分解、路径增益与基准收益分析，以及对两设施情形的线性组合优化等技术；

**📊 数据集**

未使用任何实验数据集，全部通过理论分析与构造极端实例证明；

**📈 对比分析**

与已知的Proportional机制和全局对比机制（Global Pair）比较；两设施时通过混合实现约3.519的近似比（低于4的先前界限）；三设施时得到6-近似；但未给出更高k的最佳比；

**⚠️ 局限性**

限制在一维直线，Product‑Gap对k≥4失效，且对高维空间缺乏通用推广；未来需寻找既满足策略不受激励又能提供常数近似的机制。

---

## 100. CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving

**arXiv ID:** 2608.19677 | [PDF](https://arxiv.org/pdf/2608.19677v1)

**作者:** Huang Cheng `[一作]` `[通讯]` (Meta), Huang Cheng (Meta)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种周期性基于速率的前缀亲和性路由计划，既保持可复用前缀本地化，又通过负载均衡提升整体吞吐；

**💡 创新点**

创新点在于将最高速率键的入队（top‑rate admission）、固定亲和性映射与最长处理时间优先（LPT）调度相结合，形成一次性离线规划，既提高KV命中率又抑制队列尾部延迟；

**🔧 技术方法**

技术包括：键速率估计、基于q_cap的负载阈值、热槽（warm‑slot）容量分配、LPT列表调度、请求时快速查表+负载比较，以及部署前的阴影回放验证；

**📊 数据集**

数据集为从实际生产流量去标识的半合成聚合工作负载（约12.9万个业务键，Gini=0.756），以及用于边界测试的合成短周期和鲸鱼突发工作负载；

**📈 对比分析**

通过与五种基线（Power‑of‑Two、Sticky 哈希、CHWBL、DualMap、Preble）在 Llama‑3.3‑70B fp8、60 H100 GPU 的测试平台上对比，得到93.2% KV 命中率、176 QPS 的3.5 s p99 服务水平，超过最强基线 2.3×，在8B模型和第二分布上也表现出明显优势；

**⚠️ 局限性**

局限性：假设前缀尺寸大致相同、路由键稳定、速率估计准确；对异构前缀长度或非稳定键不适用；部署需阴影回放验证，若亲和性过强可能导致容量下降，且缺乏自适应重规划机制。

---

## 101. Towards Clinically Faithful Medical Image Captioning via Enhanced Vision-Language Alignment

**arXiv ID:** 2608.19825 | [PDF](https://arxiv.org/pdf/2608.19825v1)

**作者:** Yunseo Lee `[一作]` (Chung-Ang University), Changwon Lim `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了基于双视觉编码器（BioMedCLIP+SigLIP2）、Q-Former 和 LLaMA 解码器的医学图像标题生成模型，并通过辅助 UMLS 术语预测、推理时的嵌入重排序和训练时的 MedPAIR‑SCST 强化学习实现临床对齐。

**💡 创新点**

创新点在于：①将推理时与训练时的临床对齐分离，分别采用单嵌入重排序和无参考模型的自批评强化学习；②使用双编码器与最小融合策略提升视觉表征；③引入 UMLS 术语预测辅助任务和参考无对比的 pairwise ranking 提升生成质量与临床真实性。

**🔧 技术方法**

主要技术包括 BioMedCLIP、SigLIP2 视觉编码器、Q‑Former、Bio‑MedLLaMA‑3 LLM、LoRA 微调、UMLS 术语分类、BioMedCLIP 与 BioBERT 嵌入重排序、MedPAIR‑SCST 自批评强化学习与 pairwise ranking。

**📊 数据集**

使用 ROCOv2（含 ImageCLEFmedical 2025 Caption Prediction 任务扩展）数据集，包含 80,091 张训练图像、17,277 张验证图像，每张图像均附有人工标注的标题与 UMLS 概念。

**📈 对比分析**

与 R2Gen、CvTdistilGPT2 等现有方法比较，1B 规模基线在 BERTScore、ROUGE‑1、BLEURT、UMLS‑F1 等指标已达到或超过对手；在 MedPAIR‑SCST 训练后，BERTScore、ROUGE‑1、BLEURT 与 UMLS‑F1 均显著提升，尤其是 UMLS‑F1 提升至 0.1821，显示临床可信度明显提升。

**⚠️ 局限性**

主要局限：①缺少公开的 2025 测试集，实验仅在验证集上评估，泛化能力尚未验证；②奖励函数主要基于表面指标，可能偏向特定术语分布；③在 1B 解码器上辅助 UMLS 预测收益有限，表明任务难度、损失加权与解码器容量之间的相互作用需进一步研究。

---

## 102. MILD: Tractable Terrain Modeling for Learning Improved Bipedal Locomotion on Deformable Surfaces

**arXiv ID:** 2608.19955 | [PDF](https://arxiv.org/pdf/2608.19955v1)

**作者:** Zeren Luo `[一作]` (University of Hong Kong), Peng Lu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对柔性地面，提出了基于离散元素的物理建模和强化学习控制器 MILD，实现在真实柔软表面上的双足步行。

**💡 创新点**

创新点包括：①为大尺寸足底设计可处理偏心渗透和空间异质性的可扩展离散元素接触模型；②构建带有潜变量编码和自适应尺度变换的 RL 训练框架，实现对地面刚度的隐式识别；③通过在线估计和调制实现对突变柔软度的即时适应。

**🔧 技术方法**

技术手段涵盖：离散元素颗粒动力学（开发锥模型 + 质量累积效应）、漂移阻尼、Isaac Gym 物理仿真、PPO 强化学习、非对称演员-评论家、变分自编码器与调制网络。

**📊 数据集**

数据来源：在仿真中随机采样多组颗粒参数（剪切角、渗透刚度、摩擦系数等）；硬件实验使用 EngineAI SA01 双足机器人，在六种标准柔软材质（橡胶、60d/45d/28d PU 泡沫、草、沙）上收集轨迹。

**📈 对比分析**

与 Con-Spring、Con-Cone、Rigid、Ecc-Quasi、Ecc-Spring、基于阻尼的全身控制器、HT-2、Clock 等方法对比，MILD 在地面穿透深度、滑移距离、重心振荡、能量消耗（COT 0.82–0.83）和速度跟踪精度上均优于对手，且在硬/软地面间实现无缝适应。

**⚠️ 局限性**

局限性：离散元素划分仍为经验性近似；实验仅覆盖步行类动作，未验证跳跃/奔跑；缺乏视觉感知模块，难以处理更复杂地形；对极端快速碰撞的动态响应尚待进一步评估。

---

## 103. ChatGPT Solves All Tested Qiskit Homework Assignments

**arXiv ID:** 2608.19707 | [PDF](https://arxiv.org/pdf/2608.19707v1)

**作者:** Alexei Kaltchenko `[一作]` (Wilfrid Laurier University), Gurnivaj Tiwana `[通讯]` (Wilfrid Laurier University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究尝试在不禁止AI的前提下设计AI抗干扰、可自动批改的Qiskit作业，但发现三种固定作业在150次ChatGPT会话中全部通过。

**💡 创新点**

创新点在于将种子化个性化、测量映射、隐藏参考、机器可读输出等多层防御与完整作业包结合，并记录完整会话。

**🔧 技术方法**

使用了ChatGPT、Qiskit、Aer仿真器、IBM量子硬件以及Qiskit HumanEval等技术。

**📊 数据集**

数据集为三种作业包的固定学生可见实例，每个实例重复50次，共150个会话。

**📈 对比分析**

方法是对固定实例进行50次ChatGPT会话并评估最终提交是否通过自动批改，结果全部通过，未能阻止AI完成。

**⚠️ 局限性**

局限在于仅测试三种固定实例，未覆盖多种seed、开放式任务、硬件噪声等情况，且仅针对ChatGPT。

---

## 104. Vorticity Dissipation Based Routing: A Fluid-Kinetic Framework for Loop-Free Transport in Ultra-Dense Networks

**arXiv ID:** 2608.19630 | [PDF](https://arxiv.org/pdf/2608.19630v1)

**作者:** Wen-Yu Dong `[一作]` (China Telecom Research Institute), Sheng Chen `[通讯]` (University of Southampton)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于流体动力学的 VDR（Vorticity Dissipation Routing）框架，用连续域模型和 Helmholtz‑Hodge 分解来消除超密集网络中的转发环路，实现无环路的宏观路由。

**💡 创新点**

创新点包括：① 用 Helmholtz‑Hodge 分解将流量场分为需求驱动的无旋分量和环路诱发的旋转分量；② 定义网络涡度和耗能（涡度能量）作为全局度量；③ 通过涡度耗散的梯度流来保证 Lyapunov 稳定性；④ 在分布式实现中采用图拉普拉斯、核密度估计和布隆过滤器实现实时循环抑制。

**🔧 技术方法**

采用的技术包括：连续域交通密度与源强度的核密度估计；Poisson 方程求解（离散 Laplacian）；Helmholtz‑Hodge 分解与离散 Curl/Laplacian；梯度流动力学与 Lyapunov 能量分析；图论中的节点/边 Incidence 与 Laplacian 矩阵；布隆过滤器实现包级循环检测；异步事件触发与局部化信令。

**📊 数据集**

主要在仿真环境中评估：随机几何图（1000×1000 m²）下节点密度 1000–4000 nodes/km²，Poisson 流量生成与 bursty ON‑OFF 流量；并通过多轮迭代生成连续场与离散网格数据，未使用公开真实数据集。

**📈 对比分析**

通过与 SP、GPSR、Backpressure、QTAR 等基线在同一仿真场景下比较，VDR 在平均延迟降低 60–70%，包交付率提升 15–20%，实现零实际环路，并在固定面积增密时保持近线性计算复杂度（O(N)），而基线方法呈二次或指数增长。

**⚠️ 局限性**

局限性包括：对高节点密度假设依赖，稀疏或极端非均匀拓扑时理论不完全成立；离散化与核宽度选择对涡度检测精度有影响；Poisson 求解器在大规模网格时会成为瓶颈；实现需要同步更新与布隆过滤器的内存开销；仅在二维平面下验证，三维/移动场景仍待扩展。

---

## 105. VGI-BENCH: Probing Visual Intelligence in Video Generation Models

**arXiv ID:** 2608.19583 | [PDF](https://arxiv.org/pdf/2608.19583v1)

**作者:** Xuan He `[一作]` (University of Illinois Urbana Champaign), ChengXiang Zhai `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个面向视频生成模型的视觉推理基准（VGRB），包含27个任务、810个实例，采用双层分类（任务域 + 技能标签）和过程敏感的设计，评估模型在真实图像输入下的视觉推理能力。

**💡 创新点**

创新点包括：①真实感图像输入降低视觉域不匹配；②任务设计强调中间过程而非仅最终状态；③预生成筛选和人工审核确保任务既具挑战性又可行；④引入完成度、过程规范化和最终分数的多维评价体系；⑤基于大型语言模型的自动评估器和多维诊断分析。

**🔧 技术方法**

技术上使用了扩散式视频生成模型（如Sora2、Veo3.1、Kling3.0、Seedance 2.0）以及图像生成模型（GPT-Image-2、Nano Banana Pro）；评估采用Gemini‑3‑Flash 的 VLM 评判器；对比实验涵盖闭源与开源模型，分析失败模式、输入敏感性、合成数据迁移及去噪轨迹自纠偏等。

**📊 数据集**

数据集由三类组成：①真实图像+文本提示（任务实例）；②合成图像与真实风格混合（用于验证视觉风格影响）；③1M 规模的抽象风格视频数据用于 VBVR 迁移实验；所有实例均配有参考解答与评判标准。

**📈 对比分析**

与商业闭源模型对比，开源模型整体性能低下；Seedance 2.0 在本基准上获得最高 51.0% 的最终分数；结构性难度最高的域为“Structured Puzzles”，技能维度中 Topology 与 Temporal 评分最低；人类基准显示模型与人类存在显著差距。

**⚠️ 局限性**

局限性包括：①仅评估 5–10 秒的视频长度，未覆盖长周期推理；②只考虑图像到视频（i2v）的固定 16:9 视角；③提示与评判规则均为英文，缺乏多语言支持；④任务集合为代表性而非完整，未来需扩展以适应更高能力模型。

---

## 106. Multi-Source Wasserstein Distributionally Robust Graph Learning

**arXiv ID:** 2608.19914 | [PDF](https://arxiv.org/pdf/2608.19914v1)

**作者:** Chuansen Peng `[一作]` (Sichuan University), Xiaojing Shen `[通讯]` (Sichuan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种多源Wasserstein分布鲁棒图学习框架MS-WDRO，旨在从稀缺的目标域样本中推断网络拓扑，利用异构源域数据进行融合。

**💡 创新点**

创新点在于通过加权Wasserstein重心融合异构源分布，构建一个鲁棒的名义分布，并围绕其构建模糊球以对抗残余不确定性，提供了一种新的多源学习方法。

**🔧 技术方法**

使用了Wasserstein度量、ADMM（交替方向乘子法）和算法展开技术，结合了分布鲁棒优化和图信号处理的理论。

**📊 数据集**

使用了合成基准数据集和多站点ABIDE I神经影像数据集进行实验，验证了MS-WDRO的有效性。

**📈 对比分析**

与七个基线方法进行了比较，MS-WDRO在图恢复准确性、样本效率和下游诊断效用方面表现优越，尤其在样本稀缺的情况下，性能提升显著。

**⚠️ 局限性**

限制在于算法的复杂性和超参数的联合校准，尽管提出了可微分的架构来自动学习超参数，但在多源设置下的计算开销仍然较高。

---

## 107. Separating Covariate Shift from Mechanism Change with Two Discriminators: CJSD, a Conditional Discrepancy with an Exact Covariate-Concept Decomposition

**arXiv ID:** 2608.19885 | [PDF](https://arxiv.org/pdf/2608.19885v1)

**作者:** Kentaro Oda `[一作]` `[通讯]` (Kagoshima University), Kentaro Oda (Kagoshima University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种新的任务相似度度量——Conditional Jensen–Shannon Discrepancy (CJSD)，能够分离并量化数据集之间的协变量差异和条件机制差异，并通过两个判别器实现无监督估计。

**💡 创新点**

创新点在于：① 用互信息链式规则将任务差异拆分为协变量轴和功能轴，功能轴可由判别器直接估计；② 证明了协变量无效性、漂移质量定律和单侧偏差控制等理论性质；③ 通过交叉拟合的判别器得到具有置信区间的可解释度量，并在固定参考测度下成为真实度量。

**🔧 技术方法**

主要技术包括：互信息链式分解、条件 Jensen–Shannon 散度、对两种判别器（仅输入、输入+输出）进行交叉拟合的 log‑loss 估计、单侧失配界定、固定测度下的度量化、以及与 kNN CMI 的对比实验。

**📊 数据集**

使用的数据集涵盖：合成数据、Electricity、Covertype、MNIST、CIFAR‑10、INSECTS、CIFAR‑10H 等，覆盖低维到高维、离散到连续标签、多种漂移类型。

**📈 对比分析**

与 MMD、Wasserstein、CLS、CPD、LEEP 等传统度量比较时，CJSD 在 202 组数据对中实现概念 vs 协变量区分 AUC 1.0（其他方法 0.0–0.90），在高维和样本不平衡场景下仍保持稳定；kNN CMI 在低维下可匹配但在高维失效；CJSD 具备自适应判别器、置信区间和在线扩展优势。

**⚠️ 局限性**

局限性包括：无法直接评估迁移/微调适应效果；结果对判别器校准敏感，偏差符号不受控制；在重叠区域之外的差异被诚实报告为零，导致部分变化被忽略；对判别器容量和样本大小仍有一定依赖。

---

## 108. Towards On-Board Implementation of ML-Based Helicopter Weight Estimator

**arXiv ID:** 2608.19210 | [PDF](https://arxiv.org/pdf/2608.19210v1)

**作者:** Nicolas Valot `[一作]` (Airbus Helicopters), Louis Fabre `[通讯]` (Airbus Helicopters)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为无人机起飞期间的重量估算实现了基于长短期记忆网络（LSTM）的监督机器学习模型，并将其集成到机载系统中，构建了完整的ML保证与验证流程。

**💡 创新点**

创新点包括：① 在EASA ML安全概念下实现W‑shape开发流程的第二阶段；② 通过定义MLCR与DPR，将统计性能与鲁棒性、稳定性等需求量化；③ 引入带 clamp 的改进 LSTM 细胞以保证数值稳定性；④ 采用 bit‑accurate 复制与 IBP 形式化验证，确保模型实现与训练模型一致；⑤ 使用引导抽样与自助法评估 OOD 泛化置信区间。

**🔧 技术方法**

使用的技术包括：LSTM RNN、ONNX 中立格式、Scade 代码生成、C 代码与编译器优化（WindRiver Diab、CompCert、TI ARM）、多种硬件平台（NXP QorIQ T1042、TI TMS570LC43）、Bootstrapping 与频率统计置信区间、Interval Bound Propagation（IBP）与 autoLiRPA 进行鲁棒性分析。

**📊 数据集**

使用 Airbus 全球服役机队的实时飞行参数数据（14 个传感器参数，25 个时间步），构成训练集、测试集与 OOD 集合，作为模型训练与验证的数据源。

**📈 对比分析**

通过对测试集的 MAE/MTOW、R² 进行统计，并使用自助法得到 99% 置信区间，验证模型满足 MAE/MTOW < 3×10⁻²、R² > 80%；在两种硬件平台上实现的推理时间均小于 1 ms，FPS > 1000；稳定性误差低于 10⁻³；大多数实现满足 bit‑accurate 复制；使用 IBP 证明中间层范围符合数据类型限制。

**⚠️ 局限性**

局限性包括：① 仍未实现并集成 OOD 检测模型；② 对整数 16 位实现的精度与规模化验证仍有挑战；③ 仅验证了单一虚拟传感器场景，缺乏多任务或多模型协同验证；④ 需进一步完善完整的安全认证流程与在服务期间的监测机制；⑤ 泛化能力依赖数据覆盖，可能在极端或未见条件下失效。

---

## 109. MUST-PET: MUltimodal Self-supervised learning across Tracers for whole-body PET/CT-based lesion segmentation

**arXiv ID:** 2608.19666 | [PDF](https://arxiv.org/pdf/2608.19666v1)

**作者:** Bashirul Azam Biswas `[一作]` (Dartmouth), Indrani Bhattacharya `[通讯]` (Dartmouth)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了一种基于SwinUNETR的多模态多示踪自监督预训练框架MUST-PET，用于全身PET/CT病灶分割。

**💡 创新点**

结合跨示踪器、跨机构大规模未标注PET/CT数据的多模态掩码重建自监督学习，显著提升标签效率和跨域泛化能力。

**🔧 技术方法**

采用SwinUNETR网络，随机选择PET或CT进行零均值掩码并利用另一模态保留完整信息，训练采用AdamW、余弦退火等优化策略。

**📊 数据集**

使用6,331个PET/CT扫描，涵盖AutoPET‑III、DEEP‑PSMA、SPADE、VI‑MED、DHMC等公共与内部数据，包含FDG和PSMA示踪剂，多癌种。

**📈 对比分析**

通过MAE评估重建质量，Dice/FPVol/FNVol等指标评估分割；MUST-PET在AutoPET‑III、Deep‑PSMA、DHMC三组测试集上相较于FDG‑only预训练和从零训练均提升约0.05‑0.1 Dice，并在低标注样本下表现显著更优。

**⚠️ 局限性**

无法与FDG‑only模型在相同数据集下直接对比，且对其他下游任务和更多示踪剂的泛化能力仍需进一步验证。

---

## 110. Flow Matching Meets 3D Curvilinear Structure Segmentation in Medical Imaging

**arXiv ID:** 2608.19965 | [PDF](https://arxiv.org/pdf/2608.19965v1)

**作者:** Sidi Mohamed Sid'El Moctar `[一作]` (CNRS, University of Rennes), Hélène Bouvrais `[通讯]` (CNRS, University of Rennes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于流匹配的3D CurvSegFlow模型，用于统一分割不同解剖部位的曲线结构（门静脉、脑血管、冠状动脉）。

**💡 创新点**

创新点包括：1) 将分割视为连续时间的流匹配过程，仅需少量（3步）推理即可完成；2) 采用时间条件3D U-Net并加入注意力门和正弦时间嵌入，提升对细小分支的捕捉；3) 在单一网络和训练策略下实现多解剖结构的跨任务泛化。

**🔧 技术方法**

使用技术：流匹配（Flow Matching）框架、时间条件3D U-Net、注意力门、正弦时间嵌入、复合损失（流匹配损失 + 加权 BCE + Dice），AdamW优化，显式欧拉积分。

**📊 数据集**

使用的数据集：3Dircadb（门静脉CT）、SMILE-UHURA（7T TOF-MRA脑血管）、ImageCAS（冠状动脉CTA）。

**📈 对比分析**

与多种state‑of‑the‑art方法（D^2‑RD‑UNet、FFCM‑MRF、MSFP‑Net）及通用模型（nnUNet、3D‑UNet、CS2‑Net）在Dice、IoU、Precision、Recall、clDice、HD95等指标上进行对比，结果显示在所有三个数据集上均取得最高的Dice和clDice，Recall/Precision平衡最优，整体性能最好。

**⚠️ 局限性**

局限性：仅在对比增强CT/高分辨率TOF‑MRA上评估，未测试低/非对比模态；训练采用固定大小块，可能缺失大范围上下文；偶尔出现远端过度分割导致HD95下降；在更大体积或其他器官的适用性仍待验证。

---

## 111. Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models

**arXiv ID:** 2608.19556 | [PDF](https://arxiv.org/pdf/2608.19556v1)

**作者:** Yuanhao Ban `[一作]` (UCLA), Cho-Jui Hsieh `[通讯]` (UCLA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过强化学习为流式自回归视频模型加入4D一致性奖励，提升长时序视频的几何与运动一致性，避免传统静态3D奖励导致的场景冻结。

**💡 创新点**

创新点在于：① 用4D Gaussian Splatting（MoVieS）进行动态场景重建代替静态3D重建；② 设计高斯运动门控奖励，既鼓励自然运动强度又抑制过度或缺失运动；③ 引入轻量感知锚点（HPSv2），保持视觉质量与人类审美一致。

**🔧 技术方法**

技术包括：4D Gaussian Splatting 重建、LPIPS感知相似度、DiffusionNFT 目标调度、Astrolabe 的滚动 KV 生成、HPSv2 视觉评分、以及 z‑norm 奖励组合与前向过程 RL 优化。

**📊 数据集**

使用 VidProM 数据集进行训练；评测时采用 500 条高运动突出样本和 500 条随机样本。

**📈 对比分析**

与 Self‑Forcing、Causal‑Forcing、LongLive 三大基线以及 World‑R1、VideoGPA 对比，4D‑PSNR 提升 3.46‑6.76 dB，SSIM 与 LPIPS 同步改善；在人类评估与 VideoReward 上表现优于基线和其它奖励，整体偏好提升超过 10‑12 个百分点。

**⚠️ 局限性**

局限包括：重建仅覆盖有限帧数（不覆盖整段视频）；对长时序的实时重建仍未实现；奖励依赖静态相机估计，可能无法处理动态摄像机；以及对动态场景重建精度与泛化性仍有提升空间。

---

## 112. Kähler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold

**arXiv ID:** 2608.19584 | [PDF](https://arxiv.org/pdf/2608.19584v1)

**作者:** Andrew Gracyk `[一作]` `[通讯]` (Purdue University), Andrew Gracyk (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了复参数神经网络的优化景观，提出在信息几何框架下使用Kähler几何和Calabi-Yau流形对自然梯度下降进行分析。

**💡 创新点**

创新点是将复杂几何（Dolbeault、Calabi-Yau、Ricci曲率）与深度学习优化相结合，给出了关于负曲率、特征值崩塌等现象的理论证明，并引入了动态Kähler Polyak-Łojasiewicz条件。

**🔧 技术方法**

采用信息几何、复Kähler几何、自然梯度、Dolbeault Hessian、Monge‑Ampère方程、Witten拉普拉斯算子等高级数学工具。

**📊 数据集**

文中未给出具体实验数据集，主要基于理论推导，若有实验则使用标准图像数据集（如MNIST/CIFAR）作为示例。

**📈 对比分析**

通过理论证明与数值模拟（图1‑20）比较，表明在负曲率或Calabi‑Yau条件下，优化收敛速度下降，特征值爆炸导致不稳定；相对传统欧氏梯度，理论上可提高收敛鲁棒性，但需额外正则化。

**⚠️ 局限性**

限制包括：高维计算复杂、需要全局计算信息几何指标、假设流形非紧致导致严格结论有限、对负曲率的理论分析仍不完善，且实验验证有限。

---

## 113. Holtercare-Bench: A Multimodal Benchmark for Evaluating Long-Term Dynamic ECG Analysis

**arXiv ID:** 2608.19297 | [PDF](https://arxiv.org/pdf/2608.19297v1)

**作者:** Yihan Xie `[一作]` (Zhejiang University), Lei Zhang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `109c2b71-d051-425c-831f-0c544c24280d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了 Holtercare-23K 22,980 对 QA 的三模态（信号‑视频‑文本）动态 ECG 数据集，并基于此提出 Holtercare-Bench 12 细粒度评测任务，用于评估多模态大语言模型（MLLM）在长时 ECG 分析中的认知与推理能力。

**💡 创新点**

创新点包括：①大规模真实 Holter 记录转化为多模态三模态数据，①通过 HolterAgent 自动化生成 QA 对，②设计覆盖临床诊断工作流的闭合式、开放式、报告生成三层评测体系，③在零样本与微调实验中显著证明数据集能提升 MLLM 的时序定位与因果推理。

**🔧 技术方法**

技术手段包括：HolterAgent 的信号预处理、视频生成与 QA 构造；利用 GPT‑5‑mini 进行语义解析与逻辑重构；在多模态模型中采用视频或文本输入；对 13 个主流通用与医学 MLLM 进行零样本评测，并对 Phi‑4‑mini 与 Qwen3‑VL‑8B 进行指令微调。

**📊 数据集**

使用了 788 条 13–24 小时的真实 Holter 记录，生成 22,980 条 QA 对，包含节律、事件、诊断及报告级注释；数据已去标识化并通过专业心脏病学家验证。

**📈 对比分析**

对 13 个通用与医学 MLLM 进行零样本评测，结果表明多数模型在长时时序定位与开放式推理任务上表现低下；微调后，Qwen3‑VL‑8B 在事件时序子任务上达到 99.70% 准确率，Phi‑4‑mini 也显著提升；整体而言，数据集能显著提升模型的诊断与报告生成性能。

**⚠️ 局限性**

局限性包括：①仅覆盖 788 个病例，样本量仍有限；②模型输入长度与视频文件大小受限，导致部分长序列需裁剪或加速；③评测仍主要基于自动化指标，缺乏更深入的临床实战验证。

---

## 114. Unified and Efficient Point-Line Local Features

**arXiv ID:** 2608.19894 | [PDF](https://arxiv.org/pdf/2608.19894v1)

**作者:** François Costa `[一作]` (ETH Zurich), Marc Pollefeys `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的轻量级网络，可在一次前向传播中同时提取关键点、线段及其描述子，避免了传统的两阶段独立提取。

**💡 创新点**

创新点包括：① 通过学生-教师蒸馏将多种顶尖点线检测器的知识集成到单一网络；② 将LSD后处理迁移到GPU并简化为只预测距离场，显著加速且精度不减；③ 设计轻量级线段分支仅用三层卷积，参数量比DeepLSD低十倍。

**🔧 技术方法**

使用了ALIKED的多分辨率编码器、可变形卷积、可微关键点检测、稀疏可变形描述子、距离场预测、GPU加速的LSD以及教师蒸馏监督。

**📊 数据集**

主要在Oxford-Paris（10k随机图片）进行训练；在HPatches、RDNIM、MegaDepth、ScanNet、7Scenes、ETH3D等公开数据集进行评估。

**📈 对比分析**

与SOTA的独立点检测器（SuperPoint、ALIKED、R2D2等）以及线检测器（DeepLSD、LSD、ELSED、M‑LSD等）以及联合提取器（Wireframe、PLNet）对比，本文实现了4×的GPU推理速度提升、10×的内存占用下降，并在点匹配、线匹配、视觉定位、3D重建等下游任务中达到或超过对手的准确率。

**⚠️ 局限性**

局限性包括：① 线段匹配仍仅靠端点描述子，缺乏端到端的线段匹配器；② 部分极端光照/视角变化下的线段精度仍不及专门的深度学习方法；③ 虽然后处理移到GPU，但仍有CPU占用，完全端到端加速仍待实现。

---

## 115. SAGE-XGBoost: Spatially Augmented Graph Embeddings--Machine Learning Framework for Natural Hazards Susceptibility Mapping under Data Scarcity

**arXiv ID:** 2608.19672 | [PDF](https://arxiv.org/pdf/2608.19672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. Real Interference Alignment for Active IRS-Aided Systems: A Rate-Profile Learning-Based Approach

**arXiv ID:** 2608.20007 | [PDF](https://arxiv.org/pdf/2608.20007v1)

**作者:** Junda Liao `[一作]` (Sun Yat-sen University), Qi Zhang `[通讯]` (Sun Yat-sen University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了一种基于主动智能反射面（IRS）的实域干扰规避（IA）方案，并提出了基于率轮廓学习的求解算法；

**💡 创新点**

创新点在于仅需IRS知晓瞬时信道信息（不需发射机/接收机协同学习）即可实现实域IA，并通过离线学习的率轮廓将原问题拆解为可用广义特征分解求解的可行性检查子问题；

**🔧 技术方法**

核心技术包括：主动IRS的增益放大模型、实域IA约束的矩阵化表示、率轮廓学习、广义特征分解求解可行性子问题、以及对功率约束的启发式处理；

**📊 数据集**

使用仿真数据：K个单天线发射/接收对均匀分布在半径100 m的圆盘内，IRS置于圆心15 m处，通道服从Rician衰落（因子5）、路径损耗指数2.2；

**📈 对比分析**

与传统的WMMSE算法以及无IA基线进行对比，结果显示在IRS总功率大于约9 dBm时IA系统明显优于无IA，且所提学习算法在相同参数下实现的平均总速率高于WMMSE，同时程序执行时间提升约260倍；

**⚠️ 局限性**

局限性包括：仅考虑单天线用户与单一IRS，且假设直连链路被阻塞；离线学习需要预先生成足量训练通道，且对实际部署中信道快速变化的鲁棒性未做深入验证。

---

## 117. Quantum Kernel Estimation for the Discovery of Early Lung Cancer Detection

**arXiv ID:** 2608.19304 | [PDF](https://arxiv.org/pdf/2608.19304v1)

**作者:** Hamed Javidi `[一作]` (Cleveland Clinic Research), Peter J. Mazzone `[通讯]` (Cleveland Clinic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估量子-经典混合模型在cfDNA甲基化和片段组学血液活检中用于早期肺癌检测的可行性。

**💡 创新点**

首次将量子核估计（QKE）与经典SVM进行直接对比，展示量子核在特定特征维度下可匹敌甚至优于传统方法，并提出在多模态特征和自适应核学习方面的改进方向。

**🔧 技术方法**

采用角度编码和密集角度编码的量子特征映射，结合不同纠缠拓扑；利用量子核与SVM、QPCA、经典SVM（线性、RBF、Poly、Sigmoid）进行分类；使用Optuna调参和交叉验证。

**📊 数据集**

两大数据集：56个甲基化靶点的病例对照集（813人，其中188肺癌）和公开的cfDNA fragmentomics 数据集（718人，其中172肺癌）。

**📈 对比分析**

通过10次重复 held‑out 测试和5折交叉验证评估 AUC 与 80% 灵敏度下的特异性；在甲基化数据中，经典 SVM 的 AUC 约 83–84%；在 fragmentomics 数据中，20 特征量子核配置的 AUC 约 81–82%，优于经典 SVM；但 40 特征时性能波动增大。

**⚠️ 局限性**

限制：特征已由经典方法预筛，可能对量子模型不友好；量子核仅在理想状态向量模拟下评估，未考虑硬件噪声和采样误差；增加特征维度并未必提升性能，需更符合量子友好特征选择策略。

---

## 118. Time-Uniform Self-Normalized Concentration for Discounted Least Squares: Limits and Corrections

**arXiv ID:** 2608.19643 | [PDF](https://arxiv.org/pdf/2608.19643v1)

**作者:** Yi-Shan Wu `[一作]` `[通讯]` (Academia Sinica), Yi-Shan Wu (Academia Sinica)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了折扣最小二乘估计中的加权自归一化不等式的时间均匀性，提出了标量高斯反例和下界证明，并给出了正确的固定时间修正及其在有限与无限时间轴上的补救方法。

**💡 创新点**

揭示了现有文献中广泛使用的时间均匀加权自归一化不等式的错误，并证明了其不可能满足，随后给出了匹配下界的理论分析和可行的修正方案。

**🔧 技术方法**

采用Ville不等式、可预测权重下的自归一化超马尔可夫性质、Gaussian混合技术、单变量AR(1)高斯反例以及概率上界与下界的构造，证明了时间均匀性失效与修正方法的有效性。

**📊 数据集**

无实验数据集；全部结果均为理论证明。

**📈 对比分析**

未进行实验对比；理论上证明了修正后的上界与下界在标量情形下达到匹配阶数，说明修正是最优且必要的。

**⚠️ 局限性**

主要局限在于：①仅给出了理论证明，缺乏对实际算法性能的实证验证；②下界构造主要基于标量模型，尚未完全扩展到高维情形；③修正方案仍需对每个时间点单独控制，导致额外的对数补偿，可能在某些应用中影响效率。

---

## 119. RD-Gen: Random DAG Generator Considering Multi-rate Applications for Reproducible Scheduling Evaluation

**arXiv ID:** 2608.19460 | [PDF](https://arxiv.org/pdf/2608.19460v1)

**作者:** Atsushi Yano `[一作]` (Tier IV, Inc), Takuya Azumi `[通讯]` (Tier IV, Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文主要讨论了一种用于多速率应用的随机有向无环图（DAG）生成工具RD-Gen，用于可重复的调度评估。

**💡 创新点**

创新点在于针对多速率应用场景设计了可自定义的随机DAG生成器，解决了现有工具在多速率环境下缺乏可控性的不足。

**🔧 技术方法**

使用了随机算法生成DAG结构，并结合时间片/速率划分机制实现多速率兼容。

**📊 数据集**

文中未提供具体的数据集，仅提到工具可用于实验与评估，实际使用的案例与数据集需作者进一步补充。

**📈 对比分析**

由于缺乏实验细节，文中未给出与其他方法的性能对比。若有后续章节可进一步说明。

**⚠️ 局限性**

主要局限包括：缺乏公开的数据集与基准，实验对比缺失；工具的可扩展性与实际工业环境的适配性未经过充分验证。

---

## 120. Unregularized Convergence of Single-Loop, Entropy-Regularized Natural Actor-Critic

**arXiv ID:** 2608.19587 | [PDF](https://arxiv.org/pdf/2608.19587v1)

**作者:** Zhiqiang Tan `[一作]` `[通讯]` (Rutgers University), Zhiqiang Tan (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文分析了一种单循环的熵正则化自然演员-评论家算法，探讨其在无正则化目标下的收敛性。

**💡 创新点**

创新点在于提出了一种新的算法框架，能够在熵正则化的情况下实现更快的无正则化收敛速率，并且通过引入指数平移机制来映射正则化间隙到无正则化间隙。

**🔧 技术方法**

使用了单循环的熵正则化自然演员-评论家算法，结合了线性函数逼近和策略镜像下降框架。

**📊 数据集**

使用了无限期折扣的马尔可夫决策过程（MDP）作为数据集，特别关注了带有正的最小动作间隙的情况。

**📈 对比分析**

与现有的双循环算法相比，本文的方法在无正则化目标下的收敛速率为𝒪̃(T_total^-2/3)，而现有的双循环算法在无正则化情况下的最坏情况速率为𝒪̃(T_total^-1/2)。

**⚠️ 局限性**

限制在于该算法依赖于正的最小动作间隙假设，这在某些非表格MDP中可能不成立。

---

## 121. Navigating Epistemic Monocultures in AI-Driven Science: A Simulation Study

**arXiv ID:** 2608.19390 | [PDF](https://arxiv.org/pdf/2608.19390v1)

**作者:** Sina Fazelpour `[一作]` (Northeastern University), Hannah Rubin `[通讯]` (University of Missouri)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过基于NK景观的仿真研究AI工具在科研社区中的应用，探讨非个性化AI、随机化与个性化推荐对知识多样性和科研效率的影响，并评估其在不同问题模块化程度、AI使用率及社交学习率下的效果。

**💡 创新点**

1）首次系统化使用NK景观模型量化AI对科研多样性与效率的双重影响；2）揭示非个性化AI仅在高度模块化且使用率适中的条件下有益；3）提出并验证随机化与个性化两种缓解方案，显示个性化在更广泛情形下可提升绩效并保持多样性。

**🔧 技术方法**

使用NK景观框架模拟科研问题空间；构建随机网络实现社交学习；实现三种AI推荐策略（非个性化、Top‑Decile随机化、Single‑Bit个性化）。

**📊 数据集**

所有数据均为仿真生成的NK景观，未使用真实实验或公开数据集。

**📈 对比分析**

通过对每种AI设计进行1000次独立仿真，比较平均适应度（epistemic success）与临时多样性（AUC of Hamming distance）在不同模块化程度（ρ）、AI使用率、社交学习率和专家专长水平下的表现。结果表明：非个性化AI在低模块化或高使用率下降低多样性并导致性能下降；随机化仅在ρ≥0.8时维持益处；个性化在绝大多数设置下显著提升适应度并往往保持或提升多样性。

**⚠️ 局限性**

1）模型假设固定且完整的计算/非计算模块划分，忽略真实科研中模块边界的流动性；2）假设个体能够完美观测自身决策并获得精确收益，未考虑认知偏差、误判与噪声；3）未考虑个体激励与资源不平衡等社会动力学；4）忽略AI推荐的误导性后果与长期负面影响。

---

## 122. DSLHyPE-a DSL kernel language for the Exascale Hyperbolic PDE Engine ExaHyPE

**arXiv ID:** 2608.19273 | [PDF](https://arxiv.org/pdf/2608.19273v1)

**作者:** Timothy J. R. Stokes `[一作]` (Durham University), Tobias Weinzierl `[通讯]` (Durham University)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双语DSL，将物理方程（C/C++）与数值计算（Python）分离，并通过MLIR编译器实现统一生成CPU/GPU内核

**💡 创新点**

创新点在于：①双语DSL支持多语言输入；②利用Polygeist实现跨语言内联；③设计自定义MLIR变换（内存线性化、GPU内存空间重写）以提升性能

**🔧 技术方法**

技术主要包括：MLIR/LLVM、Polygeist、自定义DSL后端、GPU offloading、内存线性化与空间重写、传统的OpenMP并行化

**📊 数据集**

实验数据集为ExaGRyPE中CCZ4（59非线性PDE）与Euler方程的离散化，使用自适应Octree网格（3×3×3、6×6×6、16×16×16）

**📈 对比分析**

对比方法为纯C++内核、加OpenMP的C++内核、MLIR生成的CPU内核以及直接生成GPU内核；实验显示MLIR内核性能与C++基线相当，GPU版本在算子密集型任务中表现优异，但数据传输开销仍显著

**⚠️ 局限性**

局限性包括：优化步骤多为手工配置；缺乏自动向量化与SoA转换；临时内存分配开销高；跨语言内联依赖Polygeist，需进一步提升可移植性和自动化程度

---

## 123. Forking Fast: Efficiently Estimating Uncertainty Dynamics in Text Generation

**arXiv ID:** 2608.19611 | [PDF](https://arxiv.org/pdf/2608.19611v1)

**作者:** Eric Bigelow `[一作]`, Atticus Geiger `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种统计模型，用于在大语言模型推理过程中高效估计不确定性动态，并显著降低重采样成本。

**💡 创新点**

将不确定性动态视为多项式采样噪声，利用分段（change‑point）检测与核加权 Dirichlet 池化实现低采样平滑；该方法实现了 3‑5 倍的样本效率提升，并可将预算压缩至原来的 1/8。

**🔧 技术方法**

采用 Forking Paths Analysis、PELT 变化点检测、Gaussian 核平滑、Dirichlet 池化及交叉验证调参；用总变差距离（TVD）评估与高采样参考的相似度。

**📊 数据集**

在 tinyMMLU（100 题）上，对 Llama‑3‑8B‑Instruct 与 DeepSeek‑R1‑Distill‑Llama‑8B 进行实验，累计收集约 1.77 B 个 token。

**📈 对比分析**

与高采样参考（S=200,N=1）比较，使用低采样加平滑后，TVD 下降至 0.0056，样本效率提升至 5×；在仅占原预算 1/8 的情况下，误差仅略增，性能保持良好。

**⚠️ 局限性**

若采样间隔过大，可能失去关键 forking 点的细节；模型假设多项式噪声并对超参数敏感；目前仅针对多选答案的评估，尚未直接扩展到自由文本生成任务。

---

## 124. Does Marginal Coverage Guarantee Class-Conditional Safety for Zero-Shot VLMs Under Shift?

**arXiv ID:** 2608.19376 | [PDF](https://arxiv.org/pdf/2608.19376v1)

**作者:** Jai Kumar Sharma `[一作]` (Virginia Tech), Amartya Dutta `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对零样本视觉语言模型（CLIP、OpenCLIP、SigLIP）在视觉与词汇迁移下使用 split‑conformal 进行审计，评估其边际覆盖与类条件尾部覆盖。

**💡 创新点**

发现高平均边际覆盖并不能保证类条件尾部安全，源端 Mondrian、clustered conformal 和 Conf‑OT 等常用修复方法无法恢复最差类覆盖；不同模型家族在效率和分数几何上存在显著差异。

**🔧 技术方法**

使用 split‑conformal、Mondrian、clustered conformal、Conf‑OT、RAPS/LAC/APS 分数，以及 softmax 与 sigmoid 分数转换。

**📊 数据集**

实验数据集包括 ImageNet、ImageNet‑V2、ImageNet‑Sketch、ImageNet‑R、ImageNet‑A、ImageNet‑C、Stanford Cars、Food‑101。

**📈 对比分析**

与多种基线比较显示：在 Sketch 等强迁移场景下，平均覆盖可达 0.86 但最差类覆盖为 0；Conf‑OT 恢复平均覆盖但尾部不变；目标校准提升尾部但成本高；不同家族平均集大小差距 2–3 倍。

**⚠️ 局限性**

局限包括校准样本稀少导致的统计分辨率、仅评估 ViT 基础网络、未包含训练适配或开放集方法、仅衡量统计安全未考虑下游任务损失。

---

## 125. Causal Reasoning with Bipartite Graphical Causal Models

**arXiv ID:** 2608.19831 | [PDF](https://arxiv.org/pdf/2608.19831v1)

**作者:** Joris M. Mooij `[一作]` `[通讯]` (University of Amsterdam), Joris M. Mooij (University of Amsterdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了双部图因果模型（BGCM）框架，用双部图（变量节点+方程节点）刻画系统方程结构，并定义了基于Simon因果排序的部分有向化、B-分离判定与对应的马尔可夫性质，进一步扩展到含非随机输入的转移条件独立性，构建了BGCM的do-演算，演示了其对传统CBN/SCM无法表达的平衡系统（如浴缸模型）的优越性。

**💡 创新点**

创新点主要包括：
1) 将方程节点与变量节点统一在双部图中，消除传统框架中对“完美干预”(X=x)的歧义；
2) 通过Simon因果排序得到部分有向化的双部图，既描述因果顺序又保留方程信息；
3) 引入B-分离（B‑separation）作为双部图的分离判定，利用方程的确定性提高条件独立性的推断能力；
4) 推导出对非随机输入的扩展马尔可夫性质和转移条件独立性，进一步实现域不变性的推理，构建BGCM版的do‑演算。

**🔧 技术方法**

技术手段包括：
- Simon因果排序算法（利用完美匹配实现快速求解）;
- 双部图的部分有向化与B‑分离判定（结合σ‑分离和d‑分离的思想）；
- 马尔可夫性质与扩展马尔可夫性质的证明；
- 用域指示变量构造联合模型，借助B‑分离得到域不变性。

**📊 数据集**

论文以浴缸平衡系统（含三个方程、六个变量）作为案例演示，没有使用公开实验数据集，而是通过符号推导与示例说明。

**📈 对比分析**

论文未进行实验对比或性能评估，主要通过理论推导和案例演示展示BGCM在表达多机制干预、确定性约束与域不变性方面的优势；未给出计算复杂度或实验结果。

**⚠️ 局限性**

局限性：
- 目前仅适用于静态（平衡）方程组，未覆盖动态或随机微分方程；
- 结构学习算法尚未提出；
- 对不满足完美匹配的系统需使用Dulmage‑Mendelsohn分解，相关理论与实现尚未完整；
- 计算复杂度在大规模方程系统中可能较高；
- 未给出与现有CBN/SCM在数据驱动场景下的实验比较。

---

## 126. Empirical Characterization of Learning Geometry in Hybrid Quantum Forecasting Models

**arXiv ID:** 2608.19497 | [PDF](https://arxiv.org/pdf/2608.19497v1)

**作者:** Sandra Leticia Juárez-Osorio `[一作]` (CINVESTAV), Eduardo Rodriguez-Tello `[通讯]` (Cinvestav Unidad Tamaulipas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对比紧凑的混合量子预测模型与结构相同的经典基线，在合成调和混合和啁啾信号上通过经验NTK诊断分析它们在不同频谱复杂度和数据量条件下的学习动态。

**💡 创新点**

通过经验NTK提供的对齐、漂移、谱集中度等指标揭示量子模型与经典模型在优化几何上的系统差异，并证明单一NTK指标无法单调预测验证收敛，提出对学习行为的多维联合解释。

**🔧 技术方法**

使用变分量子电路（VQC）与重上传编码、经典线性混合层、Adam优化、经验NTK计算、傅里叶特征增强与重上传消融实验。

**📊 数据集**

合成多变量预测任务：调和混合信号和非平稳啁啾信号，基于可调频率和噪声的随机生成。

**📈 对比分析**

在不同频率（f=1,2,4,8,12,16）和样本量（200/50）下，比较验证选取检查点、训练AULC、NTK诊断；结果显示量子模型参数更少（125 vs 281）但在大多数情形下更早达到验证最佳点，最终测试误差差异不超过3.6%。

**⚠️ 局限性**

实验基于理想无噪声量子模拟，缺乏对噪声、有限采样和更广泛经典基线的鲁棒性评估，且未深入消融单一量子组件的具体贡献。

---

## 127. The Greedy Superstring Algorithm Achieves Ratio 2 for Strings of Length 6 Already

**arXiv ID:** 2608.20018 | [PDF](https://arxiv.org/pdf/2608.20018v1)

**作者:** Nikolai Chukhin `[一作]` (JetBrains Research), Alexander Smal `[通讯]` (JetBrains Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在最短公共超字符串（Shortest Common Superstring, SCS）问题中，作者构造了特殊的字符串实例，证明当输入字符串长度至少为 6 时，贪心算法的近似比至少为 2，并且精确求出了长度为 3 时贪心算法的最坏情况近似比为 9/5。

**💡 创新点**

创新点在于：①首次通过循环谱（cyclic spectrum）构造实现了对贪心算法在任意长度 k≥6 的严格下界 ρ_k ≥ 2，破坏了此前 Cazaux‑Rivals 的猜想 ρ_k = 2-1/k；②利用图论中的路径覆盖、预算（budget）机制以及对闭路的细致分解，完成了长度 3 的精确上界 ρ_3 ≤ 9/5，从而给出了 ρ_3 的确切值。

**🔧 技术方法**

主要技术包括：循环子串与循环谱构造、构造性证明、图论中的弱连通分量与路径覆盖、预算分配与计数、闭路、束（bundle）与连接器（connector）等概念的引入和分析。

**📊 数据集**

本文为理论研究，未使用实验数据集；所有证明均为纯数学构造与推导。

**📈 对比分析**

与已有的上界（如 2.466、3.396 等）对比，作者证明了新的下界 ρ_k ≥ 2（k≥6）和精确值 ρ_3 = 9/5，表明贪心算法在这些特殊长度下的表现已被完全刻画；对于未覆盖的 k=4,5，仍需进一步研究。

**⚠️ 局限性**

局限性包括：①结果仅适用于 k=3 和 k≥6，k=4、5 的近似比仍不明确；②证明思路相当复杂，涉及大量图论细节，难以直接推广到更一般的长度或更复杂的实例；③仅给出了下界或精确值，未提供针对实际数据的实验评估。

---

## 128. Hippogriff: a semantic approach to uniting core and modules

**arXiv ID:** 2608.19728 | [PDF](https://arxiv.org/pdf/2608.19728v1)

**作者:** Owen Lynch `[一作]` (University of Oxford), Sam Staton `[通讯]` (University of Oxford)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

开发了一种新型编程语言 Hippogriff，融合 1ML 与 ModTT 的模块系统与依赖类型机制，支持统一语法与通用递归；

**💡 创新点**

创新点在于将合成相位区分嵌入元理论，既保持单一语法，又能生成可解释的类型错误并与 System F 等价；

**🔧 技术方法**

采用合成相位区分、双向展开、将语法归约至 System F、Coquand 的归一化求值等技术；

**📊 数据集**

无数据集（主要关注语言语义与实现）；

**📈 对比分析**

未给出具体性能评测，无法与其它语言做定量比较；

**⚠️ 局限性**

局限在于对大型类型的递归支持有限，缺少类型推导、模块子类化与子类型等常见语言特性。

---

## 129. Scientific Visualization as a Collaborative Data Infrastructure

**arXiv ID:** 2608.19413 | [PDF](https://arxiv.org/pdf/2608.19413v1)

**作者:** Jasmine Tan Otto `[一作]` (University of San Francisco), Scott Davidoff `[通讯]` (Space Science Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并评估了PIXLISE平台，支持NASA Perseverance火星探测器采集的光谱数据在多学科团队中的可视化与共享。

**💡 创新点**

提出将边界对象作为协同感知接口的核心，并强调跨领域知识参与对基础设施构建的重要性。

**🔧 技术方法**

使用基于交互式可视化的PIXLISE系统，包括混合图、弦图、链接刷选等技术；结合Mattermost、Adobe Illustrator等协同工具。

**📊 数据集**

利用Perseverance探测器在Jezero陨石坑采集的荧光光谱数据，以及相关的地质映射与实验记录。

**📈 对比分析**

通过与传统Excel、手工绘图等方法的对比，证明PIXLISE在共享与复核过程中的可追溯性和效率提升，但缺乏量化性能指标。

**⚠️ 局限性**

局限在于边界对象的定义需随讨论演进，工具普及度有限，跨学科协同仍需更灵活的接口与标准化支持。

---

## 130. An Agentic RAG and Evaluation Framework for Assurance Case Generation: Industrial Use Case for the EU Cyber Resilience Act Compliance

**arXiv ID:** 2608.19509 | [PDF](https://arxiv.org/pdf/2608.19509v1)

**作者:** Fariz Ikhwantri `[一作]` (Simula Research Laboratory), Pavlos Kosmides `[通讯]` (Catalink Limited)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于 Agentic Retrieval‑Augmented Generation（RAG）的自动化 Assurance Case（AC）生成与评估框架，并在欧盟网络弹性法案（CRA）合规场景中进行工业案例验证。

**💡 创新点**

创新点：①将 RAG 与 Agentic 交互式迭代检索相结合，自动从碎片化技术文档中检索并检验 AC 中的每条主张；②基于多跳自然语言推理（NLI）实现 AC 语义一致性评估；③使用合成训练数据加上真实审核数据的“Synthetic‑to‑Real”评估策略，解决缺乏标注数据的工业难题。

**🔧 技术方法**

技术：LangChain + ReAct 迭代检索、Docling + ChromaDB 文档分块与向量检索、BGE‑M3 与 L2 距离语义搜索、JSON 结构化输出、BERT 监督微调和 LLaMA 1B ICL 用于 NLI，Qwen 系列 LLM 用作评审与实验。

**📊 数据集**

数据集：PATROLIoT 火灾监测系统的内部技术文档（设计、测试、审计记录）作为合成数据源；独立审计员提供的表格式审核数据作为真实测试集；还使用了基于 LLM 合成的无检索 NLI 对照数据。

**📈 对比分析**

比较方法：在合成数据上训练 NLI 模型并在真实审核数据上测试；与无检索 Vanilla‑LLM、仅用真实数据的基线做对比。性能：在真实测试集上，Agentic RAG 训练的 BERT NLI 取得 88% 准确率、85% F1；相较于基线（约 53% 准确率）提升显著；链式推理模式相对单跳模式略逊，但仍优于基线。

**⚠️ 局限性**

局限性：①合规性证明仍需法律专家审查；②实验仅针对单一产品，结果可能不易泛化至其他行业；③评估中主观专家可能存在偏见；④实现依赖本地部署，缺乏公开实现细节。

---

## 131. Natural Language Code Retrieval for 1C:Enterprise: An Open Benchmark and Efficient Bi-Encoder

**arXiv ID:** 2608.19957 | [PDF](https://arxiv.org/pdf/2608.19957v1)

**作者:** Konstantin Chesnokov `[一作]` (Independent Researcher), Chingiz Mingazov `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了面向1C:Enterprise的俄语代码检索基准和评估工具，并发布了相应的训练数据和域适配的双编码器模型。

**💡 创新点**

主要创新在于发布了首个公开的1C代码检索基准（3413问答对）、大规模合成训练三元组（784k），以及利用Matryoshka Representation Learning实现高效压缩的域适配双编码器。

**🔧 技术方法**

采用了双编码器（sentence‑transformers）架构，使用CachedMNRL与Matryoshka Loss进行对比学习，并对输入输出加上异步提示及隐私友好分词器。

**📊 数据集**

数据集包括：1) 公开的1C论坛与FastCode问答对（3413对）；2) 通过GitHub公开代码生成的784k合成三元组；3) 公开的BM25基线与复合检索评估脚本。

**📈 对比分析**

与多种基线（多语言预训练模型、BM25、RRF融合）对比，域适配模型在macro nDCG@10上取得0.5992，微平均为0.5044，显著优于基线（提升≈0.1）并且在去除潜在泄露样本后仍保持高性能。

**⚠️ 局限性**

局限性包括单一金文档评判导致可能低估检索质量、合成查询质量仅经LLM评估缺乏专家人工验证、隐私处理规则未进行精确度评估，以及未探索多金文档、多核查询或更复杂的重排序策略。

---

## 132. Can Agent Memory Systems Track Evolving State?

**arXiv ID:** 2608.19652 | [PDF](https://arxiv.org/pdf/2608.19652v1)

**作者:** Xinyi Fan `[一作]` (University of Illinois Urbana Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了StateMemBench基准，针对LLM代理的跨会话状态跟踪问题，并开发了StateMem状态优先内存方法。

**💡 创新点**

创新点在于将“状态漂移”定义为独立错误类别，构建了以状态维度评估的对抗性多会话测试集，以及一种仅依赖状态和依赖图的无LLM推理状态管理方案。

**🔧 技术方法**

采用LLM解析转化为状态单元、依赖图传播、状态标记与重计算提示、以及可作为后端包装的状态跟踪逻辑。

**📊 数据集**

使用了三大领域（研究、购物、个人理财）采样的公开数据，并在234个多会话场景（短长两组）中生成对抗测试。

**📈 对比分析**

与现有检索、图检索、长上下文和多种记忆系统对比，StateMem在StateMemBench上分别提升至0.363/0.233，较同骨干长上下文提升2.4×，在LongMemEval等记忆基准上也保持竞争力。

**⚠️ 局限性**

局限在于需要明确的状态声明和依赖关系，难以自动化推断隐式关系，对高复杂推理任务的适用性有限。

---

## 133. A knowledge-guided agentic framework for mitigating patient-context ambiguity in health queries

**arXiv ID:** 2608.19875 | [PDF](https://arxiv.org/pdf/2608.19875v1)

**作者:** Mahyar Abbasian `[一作]`, Amir M. Rahmani `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个基于知识图谱的代理层，在患者发起简短、模糊的健康问句前主动提出澄清问题，收集缺失的患者特定信息后再将完整问句传递给下游语言模型，显著提升诊断检索和膳食安全分类的准确率。

**💡 创新点**

提出了“患者情境歧义”这一新类别，并设计了知识图谱驱动的澄清代理机制，能够系统性地识别并获取缺失的患者变量，而非仅依赖语言模型的推理或改写；该框架与下游模型无缝对接，保持模型原始权重不变。

**🔧 技术方法**

使用了知识图谱（DxSxKG 和 FoodSafetyKG）、确定性澄清控制器、内部语言模型（GPT‑5.5）生成澄清问题、外部知识存储（Neo4j）以及五个公开 LLM（GPT‑5.5、Claude Opus、Gemini 3.1 Pro、LLaMA 3.3 70B、Mistral Large）。

**📊 数据集**

诊断任务采用 1,034 条来自 Synthea 合成病历的症状–诊断实例；膳食安全任务采用 487 条基于 USDA FoodData Central、UMLS 和临床指南构建的 FoodSafetyKG 的食物–健康情境二分类实例。

**📈 对比分析**

通过与直接回答和“Rephrase‑and‑Respond”两种基线对比，评估了准确率、Recall@5、MCC 等指标。澄清框架在诊断检索上将 Top‑1 准确率提升至 57.7–71.1%，Recall@5 达到 90.6–91.9%；在膳食安全分类上 MCC 达到 0.783–0.837，且在大多数模型上实现最高 MCC。

**⚠️ 局限性**

局限性包括：使用合成数据而非真实临床病例；膳食安全评估仅覆盖部分疾病与食物；澄清答案来自模拟或训练好的模拟器，未测试对噪声/不完整回答的鲁棒性；知识图谱覆盖率与质量限制了解决方案；实验仅涉及五个 LLM，结果对其他模型或真实部署环境的泛化性未验证。

---

## 134. Mitigating Identity Essentialism in LLM Agents with Longitudinal Life Trajectories

**arXiv ID:** 2608.19621 | [PDF](https://arxiv.org/pdf/2608.19621v1)

**作者:** Hexi Wang `[一作]` (Tsinghua University), Yiqun Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了名为 LifeMem 的双记忆框架，用于构建具有长期个体经验和可持续参数记忆的 LLM 社会代理，以提升其在纵向社会模拟中的真实性。

**💡 创新点**

通过将结构化生命事件检索与个体特定 LoRA 参数记忆相结合，解决了静态属性条件导致的“身份本质主义”问题，实现了更高的人类数据对齐和多样性。

**🔧 技术方法**

采用 LLM（Llama‑3.1‑8B‑Instruct、Ministral‑3‑8B‑Instruct‑2512、Qwen3.5‑9B）、All‑MiniLM‑L6‑v2 编码器、LoRA 适配器、时间衰减检索等技术。

**📊 数据集**

在 Add Health 和 Understanding Society 两个纵向调查数据集上进行实验。

**📈 对比分析**

与静态条件、对话多样化提示、非参数记忆（SimVBG、Full History、Event RAG）以及随机事件等基线比较，LifeMem 在 KL 散度、组内/组间多样性差距、熵差距和转移分布 JS 散度等指标上均显著优于所有基线，显示出更优的性能。

**⚠️ 局限性**

主要局限在于需要收集足够丰富的生命事件覆盖，且在某些模型（如 Qwen3.5‑9B）下多样性提升伴随整体分布偏差；同时结构化检索与参数更新的计算成本仍高于轻量级提示方法。

---

## 135. Performance Verification of the AmpereOne CPU Core

**arXiv ID:** 2608.19300 | [PDF](https://arxiv.org/pdf/2608.19300v1)

**作者:** Doa'a Al-Otoom `[一作]` (Ampere Computing), Mahesh Madhav `[通讯]` (Ampere Computing)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了AmpereOne定制CPU核心的工业级性能验证流程，包括基于trace的模型与RTL的循环准确相关、单元级验证与全核相关、持续的高频回归与自动化triage；

**💡 创新点**

创新点在于将共享事件语义、误路径补偿、单元级回放、统一事件流框架以及AI驱动的自动化triage融合，形成可持续、可扩展、可自动化的工程化PV方法；

**🔧 技术方法**

使用技术包括C++/SystemC的Trace驱动模拟器Panthera、QEMU/SimPoint生成测试集、事件流日志、回放机制、S-curve及几何平均统计、自动化回归仪表板、PMU事件与功耗约束的专门验证等；

**📊 数据集**

数据集来源为“study list”中的云端关键工作负载，利用SimPoint挑选代表性区间，生成约10M指令的trace并缩减至10万指令的RTL测试；构建5k/20k片段集并覆盖EL0/EL1、MTE、安全切换等多场景；

**📈 对比分析**

比较方法通过事件流比对计算IPC比率、几何平均、绝对几何平均及误差箱图，目标为90%测试在±10%内；实验显示四代核心的几何平均接近1.0，绝对误差平均<5%，显著减少后硅逃逸；

**⚠️ 局限性**

局限性包括对模型和事件定义的人工依赖、误路径模型不完美导致极端异常误差、模型与RTL在时序细节上仍有差距、早期测试不足导致安全/ MTE等场景逃逸，以及自动化triage对异常指标匹配可能出现误判。

---

## 136. STEP: Score-Based Temporal Energy for Human Pose Video Anomaly Detection

**arXiv ID:** 2608.19987 | [PDF](https://arxiv.org/pdf/2608.19987v1)

**作者:** Jakub Micorek `[一作]` (Graz University of Technology), Horst Possegger `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于能量的时序人体姿态视频异常检测框架 STEP，利用 PCA 将姿态序列映射到低维白化空间，并在该空间内进行去噪评分匹配以学习正常姿态分布，同时加入姿态置信度加权以抑制姿态估计误差。

**💡 创新点**

创新点包括：① 在 PCA 白化空间进行去噪评分匹配，避免在原始坐标空间中噪声导致的物理不合法姿态；② 采用序列级置信度权重软化训练与推断中的噪声影响；③ 使用 σ‑调制残差 MLP 对噪声尺度进行全层级条件化，提升能量模型对多尺度噪声的表达能力；④ 轻量化结构实现毫秒级实时推断。

**🔧 技术方法**

使用技术包括：去噪评分匹配（DSM）、主成分分析（PCA）与白化、能量基模型（EBM）、序列级置信度加权、σ‑调制残差 MLP、指数滑动平均（EMA）以及多尺度噪声训练。

**📊 数据集**

实验数据集涵盖：UBnormal（合成 543 视频）、ShanghaiTech（真实 330 训练 107 测试）以及 MSAD（用于多模态对比）。

**📈 对比分析**

与现有骨架基础方法（如 MULDE、STG‑NF、SeeKer）比较，STEP 在 UBnormal 上实现 90.1% AUROC，领先前沿 12.2%；在 ShanghaiTech 上获得 86.2% AUROC，匹配或超过 STG‑NF 与 SeeKer；在 MSAD‑HR 上亦取得 74.1% AUROC，较前辈提升 13%–18%。

**⚠️ 局限性**

局限性：① 依赖姿态估计器，严重遮挡或估计错误会影响性能；② 仅考虑单人姿态，难以捕捉多人人际交互或物体操作类异常；③ 对视觉上下文缺乏建模，限制了对某些依赖背景或物体信息的异常检测。

---

## 137. Auditing Recorded Predictive Lead Service-Line Classifications Against Physical Verification: A Statewide Study of New York

**arXiv ID:** 2608.19922 | [PDF](https://arxiv.org/pdf/2608.19922v1)

**作者:** Muhammad Sarmad Sohail `[一作]` `[通讯]` (Independent researcher), Muhammad Sarmad Sohail (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在纽约州的供水系统中，对使用预测模型标注供水管线材料的记录进行审计，检测到大部分局部机构仅报告单一材料值并与自身的实地检查结果相矛盾，揭示模型输出缺乏变异性和可能的误报；

**💡 创新点**

创新点在于提出“零方差屏蔽”方法并结合纽约州公开的“分类依据”字段，实现对预测模型可验证性的快速、系统化评估，并通过多种无监督与监督估计方法量化未检测到的铅管比例；

**🔧 技术方法**

采用统计检验、空间连接（MapPLUTO）以及分层、最近邻匹配和梯度提升树等技术进行估计，并用交叉验证评估模型区分度；

**📊 数据集**

主要数据集为纽约州Lead Service Line Inventory（2025、2026年快照）和MapPLUTO地块信息；

**📈 对比分析**

将预测模型输出与现场检测、记录法以及不同地区的历史建筑年代进行比较，发现预测模型在多数地区完全缺乏变异，AUC仅0.64–0.74，估计模型未覆盖区域的铅管比例约为1150–1450条；

**⚠️ 局限性**

局限性包括对“条件可交换性”假设的依赖、挖掘与现场检查方法差异导致标签噪声、建筑年代作为管线年代的代理可能失真、仅覆盖拥有≥100条模型标注的局部机构以及缺乏逐地址模型-验证配对数据。

---

## 138. TT-net: Quantum Inspired Tensor Network Denoising in Conditional GANs

**arXiv ID:** 2608.19789 | [PDF](https://arxiv.org/pdf/2608.19789v1)

**作者:** Michal A. Sterzel `[一作]` (University of Luxembourg), Marko J. Rančić `[通讯]` (University of Luxembourg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 TT-Net，改进传统 SVD-Net 的去噪块，使用两步张量轨道分解在三种噪声场景下对图像进行恢复；

**💡 创新点**

将单通道 SVD 替换为跨通道两步张量轨道分解，能够捕获通道间结构并通过自适应阈值保持能量，提升去噪效果；

**🔧 技术方法**

张量轨道（Tensor Train）分解、SVD、生成式对抗网络（GAN）、PSNR/SSIM 评价指标、适应阈值保留能量；

**📊 数据集**

CIFAR-10 数据集（64×64），并在其上添加高斯噪声、运动模糊噪声与椒盐噪声三种合成噪声；

**📈 对比分析**

在同一训练设置（学习率、批大小、损失函数）下与 SVD-Net、EigenGAN、Pix2pix 进行对照，TT-Net 在所有噪声类型下均优于 SVD-Net，尤其在高斯噪声上 PSNR 提升约 2.7 dB、SSIM 提升约 0.05；

**⚠️ 局限性**

受 SVD-Net 在运动模糊训练中的崩溃影响；未验证对抗损失是否必要；仅在 CIFAR-10 上测试，未扩展至更大尺寸或其他数据集；未分析保留秩与噪声难度的关系。

---

## 139. Core-KAN: Continuous Vision Kernels with Kolmogorov-Arnold Networks

**arXiv ID:** 2608.19817 | [PDF](https://arxiv.org/pdf/2608.19817v1)

**作者:** Lan Guo `[一作]` (Lanzhou University), Binbin Yong `[通讯]` (Lanzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Core‑KAN，一种以连续相对尺度为条件的动态卷积操作，实现对视觉特征的密集空间自适应；

**💡 创新点**

创新点在于将几何尺度适配与内容混合解耦，使用 Kolmogorov‑Arnold 网络 (KAN) 生成共享的连续核场，并通过支持‑插值读取避免每个位置显式合成核；

**🔧 技术方法**

采用 KAN、连续坐标‑权重映射、指数移动平均尺度归一化、低秩动态卷积以及支持‑插值技术；

**📊 数据集**

在 ImageNet‑1K、COCO 2017 以及 ADE20K 上进行实验；

**📈 对比分析**

与多种动态卷积和高效骨干网络对比，在 ImageNet‑1K 上获得 81.45% top‑1（提升 3.8%）、COCO Mask R‑CNN AP 39.5/42.2、ADE20K mIoU 44.19%，参数增量仅几％，显示出优异性能；

**⚠️ 局限性**

局限在于响应银行成本随尺度支持数增加而上升，插值精度受支持数限制，需进一步优化。

---

## 140. The Lazy Pod That Lies: Deferred Cost and Failure Semantics of Lazy Container Image Pulling for Model Serving on Kubernetes

**arXiv ID:** 2608.19412 | [PDF](https://arxiv.org/pdf/2608.19412v1)

**作者:** Georgii Kliukovkin `[一作]` `[通讯]`, Georgii Kliukovkin

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在Kubernetes模型服务环境中，对两套主流懒惰容器镜像拉取系统（stargz-snapshotter与AWS SOCI）进行系统化实验，评估其在模型冷启动、全读性能以及在节点级缓存压力下的错误行为，并给出操作建议。

**💡 创新点**

创新点包括：①首次量化懒惰拉取的“递延成本”，揭示在模型尺寸较大时全读比惰性拉取更慢；②发现并详细描述了在节点缓存耗尽时，已就绪Pod会出现文件读取错误但Kubernetes健康指标仍显示正常的全新失效语义；③对两套系统的前置预取（front-loaded）与后置递延（deferred）设计进行对比，并提出基于读比例、缓存容量与监控建议的运维准则。

**🔧 技术方法**

使用技术包括：Kubernetes（kind单节点集群）、KServe、containerd与远程快照插件（stargz-snapshotter、soci-snapshotter）、AWS Registry、MinIO对象存储、FUSE文件系统、Prometheus监控、以及自定义的全读与预测负载仪表。实验环境为AWS us-east-1区域的i4i.2xlarge实例。

**📊 数据集**

数据集：构造的三种尺寸模型镜像（2 GB、14 GB、140 GB），每种镜像包含可压缩的随机负载（synthetic ballast）以及一组真实的fp16权重（Mistral‑7B 14.48 GB），用于评估压缩效果与实际推理负载。

**📈 对比分析**

比较方法：在相同硬件与网络条件下，分别测量惰性与懒惰拉取的冷启动时间（TTFP）、完整读取时间（full‑read）、以及节点缓存压力下的失败率。结果显示：懒惰拉取使TTFP几乎与模型尺寸无关，平均约为11–12 s，远快于惰性拉取的20–60 s；但在14 GB模型全读时，懒惰方式耗时约102 s，显著慢于惰性拉取的约58 s；在缓存耗尽的极端场景下，Pod仍保持Ready，但读取错误率可达67–94%。

**⚠️ 局限性**

局限性：实验仅在单节点kind集群上完成，未覆盖多节点生产环境；使用的stargz-snapshotter v0.18.2与soci-snapshotter v0.15.0版本可能不代表所有实现；synthetic ballast与真实权重的差异导致压缩与读取特性略有偏差；失败场景（如“lying pod”时间线、恢复矩阵）基于单次实验，缺乏更广泛的重复性验证。

---

## 141. Continuous Adversarial MeanFlow Transfer

**arXiv ID:** 2608.19540 | [PDF](https://arxiv.org/pdf/2608.19540v1)

**作者:** Yara Bahram `[一作]` (École de technologie supérieure), Mohammadhadi Shateri `[通讯]` (École de technologie supérieure)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将预训练的扩散/流模型在有限数据下迁移到新域并压缩为少步生成器。

**💡 创新点**

提出 MeanFlow-Transfer 将异构源模型映射到共享速度空间，并通过 Continuous Adversarial MeanFlow 进行对抗性后训练，统一加速与迁移。

**🔧 技术方法**

使用 MeanFlow-Transfer、Continuous Adversarial MeanFlow、改进的 MeanFlow (iMF)、速度映射、对抗性训练以及少步采样等技术。

**📊 数据集**

以 ImageNet 预训练模型（DiT、SiT、JiT、iMF）为源，迁移至 ArtBench、Caltech、CUB-Birds、Food、Stanford-Cars 等目标数据集。

**📈 对比分析**

与标准微调、AFM 及源模型比较，在 1、4、8、250 NFE 上使用 FID、FDD、IS 评估，+ 在 4 步时比源多步教师低 125 倍 NFE 且 FID 更好，平均 29% FID 提升。

**⚠️ 局限性**

仍无法达到单步/两步高质量；高分辨率、文本/视频生成的迁移尚未研究。

---

## 142. Multi-Tier Mentorship with AI-Assisted Development: Authentic Engineering for K-12 and Undergraduates

**arXiv ID:** 2608.19379 | [PDF](https://arxiv.org/pdf/2608.19379v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 143. Redactable blockchains and polynomial equations

**arXiv ID:** 2608.19401 | [PDF](https://arxiv.org/pdf/2608.19401v1)

**作者:** Alexander Demin `[一作]` (École polytechnique), Vladimir Shpilrain `[通讯]` (City College of New York)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于多变量多项式方程的后量子安全可编辑区块链结构，并给出了完整的红actable链构造与实现细节。

**💡 创新点**

创新点在于将一维哈希+可编辑后缀的传统构造升级为多项式隐写形式，既保持了轻量级特性，又通过将逆问题转化为求解多变量多项式方程，实现在现有参数下对量子攻击的安全性。

**🔧 技术方法**

技术实现依赖多项式代数（隐写、可编辑后缀、trapdoor 表达）、有限域上的 Gröbner 基底求解、随机生成的安全多项式与密钥，以及对 RSA 基础构造的改进。

**📊 数据集**

实验使用随机生成的模数 n（约 20 位素数）、多项式次数 d≈20、随机系数与随机评价点，未使用公开数据集，而是通过模拟和理论分析验证安全性。

**📈 对比分析**

对比方法：在基础构造上尝试多种攻击（求解多项式系统、恢复秘密多项式、函数分解等），实验表明在改进构造下这些攻击需占用数百 PB 内存，几乎不可行；同时通过内存占用与求解时间展示了安全性与可扩展性。

**⚠️ 局限性**

局限性包括：仍处于理论验证阶段，缺乏大规模实际部署实验；参数选取对安全与性能影响大，需进一步研究最优配置；对多项式系数的随机性与密钥管理的实际实现仍有挑战。

---

## 144. The Forward-Backward Disconnect: State Dynamics, Credit Assignment, and Biological Grounding in Neural Computation

**arXiv ID:** 2608.19995 | [PDF](https://arxiv.org/pdf/2608.19995v1)

**作者:** Hadi Al Mubasher `[一作]` (American University of Beirut), Mariette Awad `[通讯]` (American University of Beirut)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对前向计算与后向学习的差异进行系统性综述，构建了基于状态动力学与信用分配的三维分类法，揭示了前向-后向断裂。

**💡 创新点**

提出前向-后向断裂概念，并将生物学基础拆分为前向与学习两维，形成统一的模型族分类框架。

**🔧 技术方法**

通过对已有模型的系统归纳，构建状态动力学五类与信用分配四类交叉表，并用统计与可视化展示配置分布。

**📊 数据集**

本文为综述性工作，无特定实验数据集，主要依据公开论文与文献。

**📈 对比分析**

以32个代表性配置为样本进行计数，可视化显示全局梯度占比最大，前向多样化而后向集中化的结构被证实。

**⚠️ 局限性**

局限在于仅涵盖已审阅模型，未涉及图神经网络等新兴结构；分类标准主观性高，缺乏量化验证；未提供解决前向-后向断裂的实用方法。

---

## 145. Manifold Drift in Flow Preference Optimization: A Root Cause of Reward Hacking

**arXiv ID:** 2608.20011 | [PDF](https://arxiv.org/pdf/2608.20011v1)

**作者:** Yansen Han `[一作]` (Westlake University), Tao Lin `[通讯]` (Westlake University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了流式连续时间生成模型的偏好优化，发现传统的 FlowDPO 会导致终端样本漂移到预训练数据流形之外（称为“流形漂移”），并提出了温度控制的 ThermoDPO 目标以及加权变体来对齐偏好同时保持流形。

**💡 创新点**

创新点包括：① 定义并理论证明流形漂移的原因；② 推导 ThermoDPO 可在低温下退化为 RFT，高温下等价于 FlowDPO 并加入非负锚点；③ 给出对终端流形距离的点态上界；④ 引入加权实现方案以强化终端锚点。

**🔧 技术方法**

使用的技术主要有：流匹配（Flow Matching）框架、对数似然对比的 DPO 思路、温度调节的能量函数、解析推导与梯度步长分析、重构误差约束以及在实验中使用的流形梯度惩罚。

**📊 数据集**

数据集包括：① 低维三维分析表面（toy benchmark）用于可视化流形漂移与调参；② Stable Diffusion 3.5‑M 预训练模型与 OCR 偏好对数据集，用于真实图像生成评估。

**📈 对比分析**

实验对比了 RFT、FlowDPO（多种 β）、Diffusion‑SDPO、Linear‑DPO、χPO 以及 ThermoDPO/加权版。Toy 任务中，ThermoDPO（加权）在 StrictScore 上达到 0.899，显著优于 FlowDPO (0.629) 与 RFT (0.857)。在真实图像任务中，ThermoDPO 在 OCR、GenEval、HPSv3.0、UniReward 上均有提升，整体相对增益最高（+16.0%），且视觉质量与人类评测均不劣于基线。

**⚠️ 局限性**

限制：仅在离线偏好数据集上进行训练，未考虑在线 RLHF 的探索与稳定性；未验证方法在其他连续时间或扩散模型中的适用性；理论分析基于能量函数而非完整物理推导，可能存在更优的物理解释。

---

## 146. Bringing analytic rigor to agentic AI for science: The Brain Researcher platform for neuroimaging data analysis

**arXiv ID:** 2608.19902 | [PDF](https://arxiv.org/pdf/2608.19902v1)

**作者:** Zijiao Chen `[一作]` (Stanford University), Russell A. Poldrack `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

构建了Brain Researcher平台，利用AI代理在神经影像研究工作流中自动执行、记录并审核分析过程，以实现可审计的科研记录。

**💡 创新点**

创新点在于将AI助手转化为受治理的研究轨迹工具，而非仅产生结果；通过承诺卡、声明卡、知识图谱和审查层将方法论约束与证据关联，并通过多元宇宙分析可视化结果对分析选择的敏感性。

**🔧 技术方法**

技术包括：BIDS标准化、fMRIPrep、Nipype工作流、OpenNeuro Vocabulary的知识图谱、LLM驱动的工具注册与调用、基于规则的验证层、审查层与claim卡生成，以及多元宇宙分析与自适应搜索机制。

**📊 数据集**

使用公开与合作的神经影像数据集：HCP、OpenNeuro、FBIRN、SUDMEX CONN、跨文化社会认知ALE，以及TRIBE多模态模型内部数据。

**📈 对比分析**

通过与七种前沿LLM对照的工具调用与证据引用基准，Tool‑Calling准确率从23.3%提升至93.6%，能力覆盖率从49.8%提升至94.5%，证据可检索率从4.6%提升至22.0%；在协作案例中，多元宇宙分析揭示结果对分析路径的高度敏感，系统能将不确定结果转换为可冻结的后续研究。

**⚠️ 局限性**

局限包括：解释与写作仍需人工，训练数据可能偏向主流问题导致对少数群体或创新范式的覆盖不足；多元宇宙与内部验证未能取代外部独立复制；评估多依赖LLM裁判，可能存在自偏；对运行时成本与研究者工作量的量化尚未完成。

---

## 147. Dancing Through Soundscapes: Designing a Low-Cost, Sound-Based Device for Sensing and Interpreting Movement and Dance

**arXiv ID:** 2608.19827 | [PDF](https://arxiv.org/pdf/2608.19827v1)

**作者:** Swen E. Gaudl `[一作]` (University of Gothenburg), Silvia Carderelli-Gronau `[通讯]` (BathSPA University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一款低成本、基于声音的运动感知设备SonicDancer，用于在空间中捕捉舞者的运动并生成互动声景。

**💡 创新点**

创新点在于采用声源定位替代摄像头或身体传感器，提供屏幕减弱的音频-空间交互，并通过生成音景映射运动与空间关系。

**🔧 技术方法**

使用Raspberry Pi、ReSpeaker 6麦克风阵列、ODAS定位、Pure Data音频生成以及LED方位指示器等技术。

**📊 数据集**

未使用公开数据集，主要通过十次工作坊与350+公众参与者收集实验数据。

**📈 对比分析**

未进行定量对比实验，仅通过工作坊观察和用户反馈评估响应时间、音景质量和使用体验，表现良好但仍受噪声影响。

**⚠️ 局限性**

局限包括对环境噪声敏感、无法跟踪姿态或手势、需要手动设置空间和网络连接，且对非视觉用户的适配仍需改进。

---

## 148. GenMatch: An End-to-End Generative Matching Framework for Micro-View Order-Dispatching in Ride-Hailing

**arXiv ID:** 2608.19751 | [PDF](https://arxiv.org/pdf/2608.19751v1)

**作者:** Chuang Liu `[一作]` (Didi Chuxing), Zihao Lu `[通讯]` (Didi Chuxing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了GenMatch，一种面向微视角订单派单的端到端生成式匹配框架，直接将整批派单任务映射为一条可执行的匹配序列；

**💡 创新点**

创新点包括：①采用上下文感知的双向 bipartite 编码器捕捉批量级别的竞争与匹配关系；②通过业务感知的效用学习器联合多任务监督，学习统一的商业效用；③使用状态感知指针解码器在生成过程中跟踪并更新可选候选集合，实现动态一对一匹配；

**🔧 技术方法**

主要技术：稀疏图注意力网络、业务多任务学习、强化学习辅助监督、指针网络的状态感知自回归生成；

**📊 数据集**

使用的是滴滴在五个国际城市（包括三城离线评估、三城在线 A/B 测试）的真实订单和司机历史数据，候选集由业务规则产生；

**📈 对比分析**

与生产级多阶段基线（PDP_KM）以及多种研究基线（PDP_Greedy、PDP_GS、D2SN、V1D3、RLW、CoRide、CoopRide）对比；在离线实验中，GenMatch 在 AR、CR、GMV 上平均提升 0.31%-0.83%、0.23%-1.17%、0.11%-0.55%；在线 A/B 测试中，总体提升 AR 2.26%、CR 3.86%、GMV 2.97%，且平均减少 APT 1.84%；

**⚠️ 局限性**

局限性：依赖于完整派单批量的构造，若候选检索失效或批量过大会影响实时性能；生成目标仅基于已完成订单的标签，未能完全覆盖未广播订单的行为模式；对极端稀疏或极端拥堵场景下的泛化尚未深入验证。

---

## 149. Adaptive Probabilistic Shielding by Learning MDPs for Safe Reinforcement Learning

**arXiv ID:** 2608.19836 | [PDF](https://arxiv.org/pdf/2608.19836v1)

**作者:** Astrid Horn Brorholt `[一作]` (Aalborg University), Christian Schilling `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种自适应概率屏蔽（Adaptive Probabilistic Shielding）框架，结合在线模型估计、区间马尔可夫决策过程（iMDP）和概率屏蔽，在不知转移概率的安全强化学习环境中同时实现安全保障、策略学习与模型更新。

**💡 创新点**

创新点包括：
- 将模型估计与屏蔽更新耦合到RL循环中，实现完全在线自适应屏蔽；
- 采用区间MDP与PRISM模型检查器计算保守或乐观屏蔽，兼顾安全与探索；
- 在探索阶段允许随机动作跨越屏蔽边界，缓解过度保守导致的探索受限；
- 对不同估计器（MAP、PAC、LUI）和屏蔽策略（稳健 vs 乐观）进行系统比较，揭示其对安全与性能的影响。

**🔧 技术方法**

技术手段包括：
- 基于Q‑learning的策略学习；
- 区间MDP估计器（MAP、PAC、LUI）与置信区间；
- PRISM概率模型检查器用于计算安全概率与屏蔽；
- ε‑greedy + 扩展探索策略（全动作集 vs 屏蔽内动作集）；
- 在线屏蔽更新机制与更新间隔调优。

**📊 数据集**

数据集：实验使用五个自定义强化学习环境——飞机碰撞避免（Aircraft）、蚂蚁环绕捕食者（Antlion）、落石逃生（Sinkholes）、十字路口风险（Crossroads）、重力井探险（Gravity），每个环境均已知状态/动作拓扑但未知转移概率。

**📈 对比分析**

与基线比较：
- 无屏蔽RL（仅奖励惩罚）作为负面基线；
- “oracle”屏蔽RL（使用真实MDP的屏蔽）作为性能上限；
- 结果显示自适应屏蔽在安全违规次数与最终奖励上与oracle相近，远优于无屏蔽基线；
- 乐观屏蔽在某些环境中提升探索效率、奖励，但安全性略差；
- 估计器选择对性能有显著影响：LUI稳健屏蔽在多数环境表现最佳，MAP在探索受限时效果差。

**⚠️ 局限性**

局限性：
- 屏蔽与模型估计计算成本高，尤其是大规模状态空间；
- 过度保守的屏蔽可能阻碍探索，导致学习停滞；
- 仅适用于已知拓扑的静态环境，无法处理动态或部分可观测情形；
- 对估计器超参数（如置信度、先验强度）敏感，需要经验调优；
- 屏蔽更新频率与策略仍需更系统的理论指导。

---

## 150. HiRA-CAM: Preserving Fine-Grained Spatial Relevance in Gradient-Based Visual Explanations

**arXiv ID:** 2608.19407 | [PDF](https://arxiv.org/pdf/2608.19407v1)

**作者:** Manasi Nerurkar `[一作]` (University of Cincinnati), Ali A. Minai `[通讯]` (University of Cincinnati)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的可解释性方法 HiRA-CAM，通过层级区域一致性聚合多层卷积激活，以生成更精准、集中且稳定的激活图，从而提升卷积神经网络的可解释性。

**💡 创新点**

创新点在于：① 采用层级一致性机制在多层激活之间软投票，避免单层噪声和深层稀疏问题；② 仅在最深层重构最终热力图，减少像素级多层融合带来的噪声；③ 引入高斯平滑、区域分割、量化阈值与Sigmoid软投票相结合的流程，实现跨层一致性验证与空间精度提升。

**🔧 技术方法**

使用的技术包括：LayerCAM 的像素级梯度-激活交互、Gaussian 平滑、固定网格区域分割、区域级平均评分、量化阈值与 Sigmoid 软投票、深层层级一致性加权重构、残差混合与最终归一化。

**📊 数据集**

评估所用数据集为 ImageNet ILSVRC 2012 验证集 2,000 张图片，并在 VGG16、ResNet‑50、DenseNet‑121 三种预训练 CNN backbone 上进行实验。

**📈 对比分析**

比较方法：采用弱监督目标检测指标（loc1、loc5）、指向游戏（Pointing Game）以及删除 AUC 等。实验结果显示 HiRA-CAM 在 Pointing Game 和 Deletion AUC 上显著优于 Grad‑CAM 与 LayerCAM，loc1、loc5 的提升虽有限但持续存在，说明其在定位精度与解释可信度上均有改进。

**⚠️ 局限性**

limitations: ① 对层级选择和固定网格划分敏感，未探索自适应或内容感知的区域划分；② 仅在最深层重构热力图，可能在极深或结构复杂的网络中仍受限；③ 计算成本相对 LayerCAM 增加，尤其在多层参与时；④ 评估仅基于定位与删除指标，缺乏分割或人类主观质量评估。

---

## 151. Resilience in Trustworthy Wireless Systems

**arXiv ID:** 2608.19850 | [PDF](https://arxiv.org/pdf/2608.19850v1)

**作者:** Shixiong Wang `[一作]` (Xi'an Jiaotong University), Hongyu Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了面向可信无线系统的弹性系统的系统化框架，包括概念界定、能力维度、启用机制和量化指标，并通过PHY链路和无人机网络案例演示了弹性实现。

**💡 创新点**

创新点在于将弹性与可靠性、稳健性、适应性、存活性、恢复性等概念系统化，构建了分层的能力-机制-指标框架，并给出了针对不同网络层次的技术路径。

**🔧 技术方法**

采用了冗余、多样性、裕度、模块化、分散化、隔离、对抗措施、估计、预测、规避、缩放、重构、替换、协调等通用机制，并在PHY链路上应用LDPC、HARQ、波束赋形、MCS回退等技术，在UAV网络中应用备份无人机、分层组织、轨迹规划等技术。

**📊 数据集**

文章为综述性工作，未使用特定实验数据集，而是通过文献回顾和理论分析进行说明。

**📈 对比分析**

作者通过对比已有研究的概念定义、指标量化以及实现案例，指出弹性工程需在资源利用、性能、复杂度、响应延迟等维度权衡，尚未给出数值实验或基准测试。

**⚠️ 局限性**

局限性包括缺乏严格的数学定义与量化模型，缺少可复现的基准与评估方法，未提供针对特定场景的优化算法或实验验证。

---

## 152. Multimodal Trajectory Planning for Surface Vehicles using Turning Circle-based Control Barrier Functions

**arXiv ID:** 2608.19537 | [PDF](https://arxiv.org/pdf/2608.19537v1)

**作者:** Changyu Lee `[一作]` `[通讯]` (Kongju National University), Changyu Lee (Kongju National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无引导路径的多模态轨迹规划框架，利用转弯圆基控制障碍函数（TC‑CBF）在自航表面船舶中实现碰撞规避。

**💡 创新点**

创新点在于将通过侧向选择编码进TC‑CBF约束中，枚举最近M艘船的左/右侧，生成2^M个同构最优控制问题，既不需要高层规划也能显式考虑船舶有限转弯半径，且通过并行批量求解保持实时性。

**🔧 技术方法**

采用Nomoto一阶动力学、模型预测控制（MPC）、TC‑CBF约束、acados/HPIPM求解器以及OpenMP并行，结合LOS指引生成参考轨迹。

**📊 数据集**

通过对VLCC级船舶进行仿真，使用自定义的交通船速度与位置数据（AIS仿真）以及静态障碍物，未使用公开数据集。

**📈 对比分析**

与单模态欧几里得距离CBF（ED‑CBF）和单模态TC‑CBF基线在三种交通密度下进行蒙特卡罗比较，成功率分别为99%、97%、86%，显著高于基线（≤60%），安全区穿透深度和计算时间均得到改善，批量求解在1s重规划周期内平均耗时≈1.7ms。

**⚠️ 局限性**

局限性包括对常数速度预测的依赖、未考虑航道/海岸边界约束、仅考虑6艘最近船只、未加入交通预测不确定性、未在真实环境验证，仅在仿真中评估。

---

## 153. Breaking the $2^n$ Barrier for Counting Linear Extensions with a Short Elementary Algorithm

**arXiv ID:** 2608.19505 | [PDF](https://arxiv.org/pdf/2608.19505v1)

**作者:** Keigo Oka `[一作]` `[通讯]`, Keigo Oka

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的确定性算法，用于精确计算任意有限偏序集的线性扩展数量，时间复杂度为(1.89^n)；

**💡 创新点**

突破了之前的2^n上界，首次在一般偏序集中实现c<2的指数算法，且首次引入了“首元素模式”表示、计数矢量解码、截止日期动态规划及状态计数策略；

**🔧 技术方法**

核心技术包括偏序集的链分割、最大反链分离、首元素模式的组合枚举、计数矢量的唯一解码算法、基于截止日期的动态规划以及星棒计数实现状态压缩；

**📊 数据集**

本研究为理论算法，未使用任何实验数据集；

**📈 对比分析**

通过与传统的O(n2^n)动态规划比较，证明在宽度较大或小的两种分支下均可在(1.89^n)时间内完成；

**⚠️ 局限性**

尽管相对于已知方法提升显著，但仍为指数级算法，无法在多项式时间内解决问题，且对极大规模实例的可行性仍有限。

---

## 154. SAPO: Single-Rollout Autoregressive Policy Optimization for Agentic Reinforcement Learning

**arXiv ID:** 2608.19842 | [PDF](https://arxiv.org/pdf/2608.19842v1)

**作者:** Dayang Liang `[一作]` (Xiamen University), Yunlong Liu `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出单回合自回归策略优化（SAPO），将策略、状态值和动作值统一在同一语言模型中完成，无需单独的 critic 或多回合采样。

**💡 创新点**

通过在语言模型的因果边界读取值，并采用 λ‑归还与批量归一化的轨迹优势估计，实现在单回合内的显式时序信用分配和价值泛化。

**🔧 技术方法**

自回归 Transformer、PPO 目标、SARSA 辅助目标、λ‑归还、批量归一化、价值温度读取、共享词表读取等技术。

**📊 数据集**

在 ALFWorld 和 WebShop 两个多轮交互任务上使用 Qwen2.5-1.5B/7B backbone 进行实验。

**📈 对比分析**

与 PPO、GRPO、RLOO、EMPG、GiGPO 等基线在相同模型规模下对比，SAPO 在 ALFWorld 与 WebShop 上平均成功率提升约 15‑20%，7B 规模下实现 94% 成功率、88.6 评分，且训练迭代时间比 PPO 降低 33.2%。

**⚠️ 局限性**

仍受限于稀疏奖励任务的终端信号，单回合优势估计在极长序列中可能存在偏差，对不同任务的通用性尚未完全验证。

---

## 155. Hear2Act: Benchmarking When Prosody Should Change What an Assistant Does

**arXiv ID:** 2608.19515 | [PDF](https://arxiv.org/pdf/2608.19515v1)

**作者:** Xinyi Liu `[一作]` (Amazon), Hari Thadakamalla `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Hear2Act基准，评估在多轮任务导向对话中语调信息如何影响决策。

**💡 创新点**

创新性地将语调与任务结果相连的统一评估协议，采用对照情境与匹配回放，直接衡量语调对最终选择的影响。

**🔧 技术方法**

使用音频感知与生成的多模态大型语言模型（Qwen2.5‑Omni、Qwen2‑Audio）以及文本LLM（Claude Opus 4.6、Kimi K2.5、DeepSeek‑V3.2、GLM‑5、Qwen3‑32B），结合TTS合成、音频推断状态、SER基线等技术。

**📊 数据集**

基于Schema‑Guided Dialogue的48个服务域生成480个控制情景，包含三层隐藏关注点和11个候选答案。

**📈 对比分析**

采用匹配回放的对照实验，比较不同访问条件（仅文本、仅音频、音频+文本、文本+真状态、音频推断状态、SER）对最佳解率、推荐满意率等指标的影响；实验显示语调在词义不足时提升约25‑30%最佳解率，音频推断状态能恢复至约40%；直接音频效果有限。

**⚠️ 局限性**

局限在于仅评估单一语调功能、使用合成语音、关注点结构单一、真状态访问仅为诊断、缺乏对真实多样化对话的验证。

---

## 156. GOAG: Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation

**arXiv ID:** 2608.19759 | [PDF](https://arxiv.org/pdf/2608.19759v1)

**作者:** Julien Merand `[一作]` (Université Paris-Saclay), Liming Chen `[通讯]` (Ecole Centrale de Lyon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种对象无关的多指抓取规划框架，利用抓手自身几何与抓取范畴生成可行的接触点分布，实现零样本泛化。

**💡 创新点**

创新点在于：① 在训练阶段仅使用抓手几何和抓取范畴，完全不依赖对象数据；② 通过条件变分自编码器（CVAE）学习抓手接触点的潜在分布；③ 在推理时仅引入对象特征，快速检索兼容的接触区域，实现高效、多样化的抓取生成。

**🔧 技术方法**

使用技术包括：条件变分自编码器（CVAE）、PointNet++链接映射、Basis Point Set（BPS）空间编码、力闭合估计、基于能量的抓取优化等。

**📊 数据集**

训练数据采用合成的抓手配置与接触点，使用多指抓手（Barrett、Allegro、Shadow）对应的抓取范畴；验证数据集主要是 MultiDex、YCB、ContactDB 等公开抓取基准。

**📈 对比分析**

与 DFC、GenDexGrasp、DRO-Grasp 等方法比较，MultiDex 上平均成功率达 86.93%，与现有方法相当或更好；生成速度更快、模型参数更小；在多数据集泛化测试中仅训练一次即可获得第二高平均成功率。

**⚠️ 局限性**

局限性包括：依赖预定义的抓取范畴，难以覆盖极端或未标记的抓取类型；对大型或极端姿态物体的收敛性能有限；需要后续力闭合与优化步骤以确保物理可行性。

---

## 157. Two-sided receptivity to conversational AI agents in online dating: Bilingual survey data from Fledge.Love

**arXiv ID:** 2608.19545 | [PDF](https://arxiv.org/pdf/2608.19545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 158. Inadvertent Context Leakage in Language Models

**arXiv ID:** 2608.19857 | [PDF](https://arxiv.org/pdf/2608.19857v1)

**作者:** Jaiden Fairoze `[一作]` (FAIR, Meta Superintelligence Labs), Saeed Mahloujifar `[通讯]` (FAIR, Meta Superintelligence Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM在保密指令下，文本输出仍可泄露上下文秘密的隐蔽渠道，并提出谓词推理游戏模型，展示主动和被动攻击能恢复数字、用户记忆和社保号。

**💡 创新点**

首次将泄露视为统计通道而非显式文本，提出适应性攻击者的预测游戏，证明即使模型拒绝直接透露，秘密仍可被重构。

**🔧 技术方法**

利用黑盒查询、统计分布估计、基于Qwen-2.5-1.5B的LoRA解码器、主动提示注入、RL优化等技术。

**📊 数据集**

对八款前沿专有LLM（Claude Opus、Gemini、GPT-5.4、Grok等）进行实验，使用1000个N位随机数字、15个用户记忆属性、9位SSN等人工构造上下文。

**📈 对比分析**

通过全秘密重构准确率、每位准确率和通道熵等指标对比，发现Leaky Tier模型在2位数字可100%重构，4位时达82%，SSN主动提取成功率分别为97%和88%；与基线文本判别器相比提升显著；但受限于模型层次和隐藏词典，实验局限。

**⚠️ 局限性**

局限包括仅针对有限数字与二元属性、仅使用专有模型、未评估非词汇秘密、黑盒假设、未考虑未知模型或有限查询等。

---

## 159. LLM as Detector: An In-context Learning Approach for Tabular Anomaly Detection

**arXiv ID:** 2608.19463 | [PDF](https://arxiv.org/pdf/2608.19463v1)

**作者:** Tu Anh Hoang Nguyen `[一作]` (Deakin University), Sunil Gupta `[通讯]` (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无监督表格异常检测框架 LLM-Detector，利用大型语言模型的提示式推理生成可执行的异常评分引擎，直接在测试集上计算异常分数，完全不需要对模型进行微调或训练。

**💡 创新点**

核心创新在于将正常状态的统计摘要、因果关系图和稀疏原型知识编码为结构化提示，然后让 LLM 以代码生成的方式合成一套基于三种分量（统计偏离、因果违背、密度）的评分逻辑，形成完全可解释、可复用且无需参数学习的检测器。

**🔧 技术方法**

技术手段包括：①统计摘要提取（均值、方差、极值或类别概率）；②PC 算法得到因果 DAG；③K‑means+NN 原型蒸馏；④将上述知识拼接成提示并通过 Gemini‑3.0（或 DeepSeek、GPT‑5.2）进行代码生成；⑤生成的 Python 程序实现对每个测试样本的三分量得分并合成最终 0–100 归一化异常分数；⑥采用 AUC‑ROC 进行评估。

**📊 数据集**

实验使用 24 个公开表格异常检测基准，12 个混合类型（含类别与连续特征）来自 ODDS、ADBench、Kaggle 等，12 个纯连续数据集，涵盖不同领域（金融、医疗、网络安全、社科）。

**📈 对比分析**

与 15 种 SOTA 方法（如 IForest、KNN、PCA、ECOD、DeepSVDD、GOAD、AnoLLM、LLM‑DAS 等）在 AUC‑ROC 上对比，LLM‑Detector 在混合类型数据上平均达到 0.7407，超过最佳基线 0.6972；在连续数据上最高 0.91，整体保持领先；此外，其推理速度快、训练成本低。

**⚠️ 局限性**

局限性包括：①对 LLM 代码生成的鲁棒性和提示设计高度依赖；②因果图估计误差可能导致因果分量失效；③对极高维或极稀疏类别的处理仍需改进；④提示长度受限，难以一次性注入过多原型；⑤缺乏对时间序列或流式数据的直接适配。

---

## 160. HyperCut: Fast Inter-Layer Scheduling via Directed Hypergraph and Early Filtering

**arXiv ID:** 2608.19296 | [PDF](https://arxiv.org/pdf/2608.19296v1)

**作者:** Ziang Wei `[一作]` (Nanjing University), Yuxiang Fu `[通讯]` (Nanjing University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 HyperCut，一种基于有向超图的快速交叉层调度框架，用于提升张量加速器的利用率和能效。

**💡 创新点**

创新点包括：① 在交叉层阶段即可得到上界成本估计，实现早期过滤；② 将超图分割与物理映射统一为状态表示；③ 采用 beam‑search 与 60/30/10 策略，将搜索空间从指数降至线性。

**🔧 技术方法**

技术手段：有向超图（DHG）建模、分层划分与细化、基于 Manhattan 距离的贪心放置、批次拆分、能耗/时延评估模型、基于梯度的搜索控制。

**📊 数据集**

实验数据集：Darknet19、ResNet50、GPT‑2 Decode、PNASNet，使用 64、256 个 Tile 的 12nm NVDLA‑风格平台进行评估。

**📈 对比分析**

对比方法：SET（开源框架）和传统 LP/LS 方案；在 10 组案例中，HyperCut 在几何平均上实现了 2.0× 的 EDP 降低和 80.47% 的探索时间缩减。

**⚠️ 局限性**

局限性：在极深网络（如 PNASNet）时 EDP 可能略高，搜索仍受 beam 宽度和超图转换开销限制；对超大规模网络仍需进一步压缩搜索空间。

---

## 161. Parameterized Complexity of Temporal Agony

**arXiv ID:** 2608.20077 | [PDF](https://arxiv.org/pdf/2608.20077v1)

**作者:** Tom-Lukas Breitkopf `[一作]` (Technical University of Berlin), Pascal Kunz `[通讯]` (Technical University of Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文研究了时间网络中层级划分问题——Temporal Agony（时间怨恨）——的参数化复杂性，系统地分析了不同参数组合下的可解性与不可解性。

**💡 创新点**

创新点在于确定了允许的秩数k的精确复杂性边界：k=2可多项式求解；k=3在ℓ=1时为NP‑hard；k=4在ℓ=1、α=0时仍为NP‑hard，甚至在α为常数时亦不可解；并且证明了该问题在组合参数n+ℓ下为FPT，在单参数n下为XP。

**🔧 技术方法**

主要技术包括：基于动态规划的多阶段分段排名枚举；构造精细的布尔可满足性归约（SAT）来证明NP‑hard性；利用秩数与变化点的上界（k≤(ℓ+1)n）约束搜索空间。

**📊 数据集**

本文未使用实验数据集，而是以理论证明与算法复杂度分析为主。

**📈 对比分析**

由于本工作是纯理论分析，没有实验对比；但通过复杂度分类，指出了不同参数取值对算法可行性的影响。

**⚠️ 局限性**

局限性：缺乏针对大规模实际时间网络的实证评估；对参数k较大或τ很长的情况仍未给出有效算法；且仅针对特定的惩罚函数p_l（max(0,x+1)）进行分析，未讨论其他惩罚形式。

---

## 162. ParaWeb: Parallel Programming Patterns for Web Development

**arXiv ID:** 2608.19935 | [PDF](https://arxiv.org/pdf/2608.19935v1)

**作者:** Suejb Memeti `[一作]` `[通讯]` (Blekinge Institute of Technology), Suejb Memeti (Blekinge Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 ParaWeb，一款 TypeScript 库，实现了十种并行编程模式（Map、Filter、Reduce、Scan、Scatter、Farm、Pipeline、Divide‑and‑Conquer、Stencil、MapReduce）的三种实现变体（MP、Shared、GPU），支持 Node.js、浏览器和 WebGPU；

**💡 创新点**

创新点在于将算法骨架模式迁移到 Web 生态，提供统一的高级接口、自动线程池管理、共享缓冲区零拷贝以及 GPU 加速，三种实现路径可根据工作负载灵活切换；

**🔧 技术方法**

采用 Node.js Worker Threads、Web Workers、SharedArrayBuffer、WebGPU WGSL、函数序列化、GPU 计算着色器、并行模式设计等技术，构建跨平台的并行编程框架；

**📊 数据集**

使用数值数组模拟的数据集，规模分别为 100K、1M、5-10M 元素；还针对图像卷积使用 4K 分辨率图像，涵盖多种工作负载（FFT、任务拆分、滤波器等）；

**📈 对比分析**

通过多线程（2、4、8、16）与单线程基线比较，记录加速比；CPU 版最高 11.6×，GPU 版最高 260×，图像卷积最高 414×，所有实验均在 MacBook Pro M3 Max 上完成；

**⚠️ 局限性**

限制包括：函数必须自包含、仅支持数值数组、GPU 采用 32 位浮点、需要跨域隔离、共享缓冲区同步成本、单工作组前缀和、以及静态变体选择需人工调优；

---

## 163. Green BOA: Determining the environmental break-even point for ML-based data compression

**arXiv ID:** 2608.19994 | [PDF](https://arxiv.org/pdf/2608.19994v1)

**作者:** Caterina Doglioni `[一作]` (University of Manchester), Sanjiban Sengupta `[通讯]` (CERN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了 ML 基于数据压缩算法的碳排放与传统压缩及磁盘存储的碳排放平衡点，计算训练与推理的 CO₂ 费用与被压缩数据的额外存储碳排放相等时的打破点。

**💡 创新点**

提出了将 ML 训练与推理的碳足迹与被压缩数据的额外存储碳足迹相平衡的“碳打破点”概念，并针对不同国家电力结构进行情景分析。

**🔧 技术方法**

使用 BOA 两层 Mamba‑v1 模型在 Nvidia T4 GPU 上训练与推理，使用 CodeCarbon 追踪能耗，并与 ZSTD、LZMA 传统压缩算法及 Seagate Exos X18 HDD 与 LTO‑8 录带的生命周期碳排放进行比较。

**📊 数据集**

以 49.92 MB 的 “bundled CMS file” 作为样本数据，随后按线性放缩推算到不同规模的数据集。

**📈 对比分析**

通过 CodeCarbon 记录 CPU/GPU/RAM 能耗并转化为 CO₂e，再与 HDD/Tape 的制造与运营碳排放进行比较；结果表明打破点高度依赖电力强度，ML 压缩能耗高于传统压缩，但压缩率更优。

**⚠️ 局限性**

仅为概念验证，假设训练一次后即可大规模使用；忽略 GPU 共享与多任务；碳排放估算基于有限硬件与生命周期数据，未涵盖更复杂工作负载与多轮压缩场景。

---

## 164. CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios

**arXiv ID:** 2608.19380 | [PDF](https://arxiv.org/pdf/2608.19380v1)

**作者:** Sparsh Garg `[一作]` (NEC Laboratories), Abhishek Aich `[通讯]` (NEC Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CAViAR基准，包含2,249段真实驾驶仪视频及20,108个结构化问答对，聚焦感知、因果与责任推理

**💡 创新点**

首次将责任归因（可见失误方、受害方、违规类型）与规则相关的监督引入视频理解基准，构建统一的多任务评估框架

**🔧 技术方法**

使用多模态视觉‑语言模型（Cosmos‑Reason2、Qwen3‑VL、InternVL3），通过LoRA微调；评估采用MCQ准确率、BERTScore‑F1与GPT‑4o判定器

**📊 数据集**

基于CarCrashDataset（CCD）与Nexar两大真实车载摄像数据集，构建无泄漏的训练/测试划分

**📈 对比分析**

实验显示感知任务（天气、光照、道路条件、事故类型）表现相对良好，而责任推理（尤其违规识别）准确率低，3B模型微调提升显著但8B规模提升有限，表明存在感知‑推理鸿沟

**⚠️ 局限性**

限制包括注释可靠性缺乏正式一致性评估、评估指标对责任推理的完整性不完美、未给出人类基准、模型覆盖范围有限、可能存在数据偏差与隐私约束

---

## 165. What Matters for Latent Actions in Robot Learning

**arXiv ID:** 2608.19613 | [PDF](https://arxiv.org/pdf/2608.19613v1)

**作者:** Xizhou Bu `[一作]` (Fudan University), Xiaoshuai Hao `[通讯]` (Xiaomi EV)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于潜在动作模型（LAM）的机器人学习方法进行了系统的、统一的实验评估，覆盖41种设计选择并在多种基准和真实机器人上验证。

**💡 创新点**

首次提供全面的实证研究，提出统一的自编码框架，系统比较不同潜在动作建模、正则化与集成策略，给出针对LAM的实用设计准则，并揭示VLM细调与潜在动作的规模提升对下游控制的正向影响。

**🔧 技术方法**

采用自编码器架构（IDM+FDM或直接CFD-AE），引入多种潜在动作正则化（VAE、VQ‑VAE、Sparsity、SIGReg），探索多种动作头（DAP、LAP、JAP等），以及在Qwen3‑VL‑4B视觉‑语言模型上进行潜在动作细调。

**📊 数据集**

使用约5900万帧无标签视频（包括Open X‑Embodiment、RoboTwin2.0、LIBERO、LIBERO‑Plus等公开机器人数据集），以及在Franka Panda 7‑DoF机器人上收集的4000条真实演示。

**📈 对比分析**

通过在LIBERO、LIBERO‑Plus、RoboTwin2.0等基准上对比性能，发现LAPO+ΔDINO+VAE+LAP组合能取得最优结果，潜在动作维度32最佳，VLM细调规模越大性能越好；在真实机器人实验中，潜在动作细调提升约14%成功率，训练收敛速度更快。

**⚠️ 局限性**

实验仅覆盖机械臂抓取与操控任务，未在多关节或多模态平台上验证；潜在动作仅作为辅助监督，未在预训练阶段直接学习；数据集主要来自现有公开仓库，缺乏大规模互联网视频；对不同维度和任务的通用性仍待进一步验证。

---

## 166. Repo0: Design-Driven Zero-to-All Code Generation

**arXiv ID:** 2608.19854 | [PDF](https://arxiv.org/pdf/2608.19854v1)

**作者:** Silin Chen `[一作]` (Shanghai Jiao Tong University), Haibing Guan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个持续结构演化的框架，用于从自然语言需求零到全代码生成。

**💡 创新点**

创新点在于采用 Dual-DAG 结构和模块化度量驱动的动态结构演化，实时优化组件边界。

**🔧 技术方法**

使用 LLM 代理、Dual-DAG 表示、模块度量（内聚度、耦合度、连通度）、结构化行动（拆分、合并、修订、稳定）以及测试驱动开发。

**📊 数据集**

在 RepoCraft 六个真实 Python 仓库（MLKit-Py、TableKit、SymbolicMath、StatModeler、HttpEasy、PyWebEngine）上进行实验。

**📈 对比分析**

与 mini-SWE-agent、Paper2Code、RPG 等基线比较，取得最高的功能覆盖率与通过率，提升多达 20.08% 覆盖率和 29.74% 通过率。

**⚠️ 局限性**

局限性包括对 LLM 推理能力高度依赖、实验仅限 Python 语言和单一 Benchmark，未验证跨语言或更大规模项目。

---

## 167. Specification-delta-driven data governance: an empirical study of the «spec-delta» as the unit of change in lakehouse data platforms

**arXiv ID:** 2608.19838 | [PDF](https://arxiv.org/pdf/2608.19838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 168. Towards Quantifying Benchmark Optimization in ASR Models

**arXiv ID:** 2608.19936 | [PDF](https://arxiv.org/pdf/2608.19936v1)

**作者:** Theo Lebryk `[一作]` (Hume AI Research), Panagiotis Tzirakis `[通讯]` (Hume AI Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过设计三种行为探针（参考错误复制、遮罩实体恢复、正字法切换），系统测量并揭示 ASR 模型在公开基准上的优化（benchmaxxing）现象，并利用机制定位技术（线性导向、激活补丁、语音克隆等）探究其触发条件和可逆性。

**💡 创新点**

创新点在于：①提出可复现、可量化的三类探针，直接针对音频与参考文本的不确定性；②将行为探针与机制解释结合，利用低秩线性调控与激活补丁来证明模型对基准特定音频的“暗语”；③通过新采集的实时语料（ep‑fresh、libri‑fresh）验证优化行为的时域和声源局限性。

**🔧 技术方法**

使用技术包括：教师强迫对数似然、音频提升（audio lift）、对齐与编辑检测、语音克隆与合成、添加噪声/混响、线性方向学习与投影、激活补丁（encoder/decoder）、注意力掩码、翻译任务验证、统计显著性检验。

**📊 数据集**

主要数据集：VoxPopuli（英语）、LibriSpeech、DaiKon（对话数据）、ep‑fresh（2026 年欧洲议会新采集）、libri‑fresh（LibriVox 新读者）、Qwen3‑TTS 生成的合成语料；对比多模型的公开权重（Whisper、Cohere‑Transcribe、Parakeet、Moonshine、Canary‑Qwen、Granite‑Speech、Higgs‑Audio、Kimi‑Audio、Phi‑4、Qwen3‑ASR、Voxtral‑Mini）。

**📈 对比分析**

通过对 11 个公开模型在 VoxPopuli 与 LibriSpeech 上的 WER 与探针指标进行交叉评估，发现 WER 最低（5.4‑5.8%）的模型对应最高的参考错误复制率（0.18‑0.30），而较低分模型的指标接近零；通过线性导向与激活补丁可在 benchmark 录音与新鲜录音之间实现 80‑90% 的行为切换；在 ep‑fresh、libri‑fresh 等未见数据上，模型对基准优化的影响显著下降，表明行为为“窄”触发。

**⚠️ 局限性**

局限性包括：未完整解析训练阶段为何产生此类优化；实验聚焦于英语语料，缺乏多语言验证；仅使用公开模型，未探索私有模型或大模型的差异；探针的设计在某些情况下可能受语言模型先验影响；需要进一步评估在更复杂噪声、口音或长文本场景下的泛化。

---

## 169. ABEAT: Efficient and Anonymous Encryption for ABE-based Dynamic Group Communication

**arXiv ID:** 2608.19302 | [PDF](https://arxiv.org/pdf/2608.19302v1)

**作者:** Hongmiao Yu `[一作]` (University of California Riverside), K. K. Ramakrishnan `[通讯]` (Rutgers University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于属性加密（ABE）的动态群组通信系统，利用命名空间图实现群组的自适应管理，并通过匿名KP‑ABE实现收件人匿名性与高效的收件人验证。

**💡 创新点**

创新点在于：① 通过在KP‑ABE中隐藏属性集与加入 Vanilla 加密块，完整实现收件人匿名；② 引入双重密钥权威机制，降低主密钥泄露风险；③ 在 Vanilla 加密块中嵌入 Diffie‑Hellman 标签，使非收件人可快速验证并提前放弃昂贵的 ABE 解密。

**🔧 技术方法**

核心技术包括：属性加密（KP‑ABE）与分裂密钥权威、负载匿名化、Vanilla 公钥加密块、Diffie‑Hellman 标签、基于超时的撤销、对称密钥包装以及基于命名空间的访问控制。

**📊 数据集**

实验采用随机生成的 1337 节点命名空间图、基于 Wikipedia 的灾难管理命名空间、Rocketfuel 1221 Telstra 拓扑以及 POISE 网络仿真器的数据集。

**📈 对比分析**

与 HVE、FABEO、P-A‑KP‑ABE、PKE、MLS 等方案比较，
• 加密时间：匿名 ABE 约 211 ms（100 目标）比 HVE 的 4480 ms 快 88%；
• 解密时间：目标收件人 35 ms，非目标 0.42 ms，较 HVE 的 38.8 ms 快 80×；
• 密文长度：匿名 ABE 约 66% 小于匿名 PKE；
• 整体通知延迟：在灾难管理工作负载下，系统延迟约 0.1 s，仍可接受。

**⚠️ 局限性**

局限性包括：① 通过接收者上集 Q 的统计分析仍可能泄露目标集合 P；② Vanilla 加密块长度可泄露 |P|，需使用填充；③ 仅提供收件人匿名，发件人匿名性未覆盖；④ 双重权威仍需可信协作，若其中一方泄露仍不泄露信息，但安全性依赖两方不合谋；⑤ 标签引入的加密开销在极大目标集合时仍显著；⑥ 在高延迟或大规模网络中的真实性能尚未完全验证。

---

## 170. Learning Deterministic and Stochastic Forced Hamiltonian Systems

**arXiv ID:** 2608.19688 | [PDF](https://arxiv.org/pdf/2608.19688v1)

**作者:** Benedikt Brantner `[一作]` (Max-Planck-Institut f"ur Plasmaphysik), Tomasz Tyranowski `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于拉格朗日-达姆贝尔原理和变分积分器的几何框架，用神经网络学习受迫 Hamiltonian 系统（包括确定性、参数化和随机受迫系统）的流映射，并给出了通用逼近定理；

**💡 创新点**

创新点在于：①将受迫 Hamiltonian 系统的 Lagrange‑d'Alembert 流映射与神经网络结构相结合，形成 Generalized Forced Hamiltonian Neural Networks (GFHNN) 与 Parametric GFHNN (PGFHNN)；②证明这些网络在 C^r 拓扑下对任意 Lagrange‑d'Alembert 流映射具有全局逼近性；③通过将随机 Stratonovich 积分视为参数，使同一框架可直接应用于随机受迫 Hamiltonian 系统；

**🔧 技术方法**

使用的技术包括：Lagrange‑d'Alembert 变分积分器、Lagrange‑d'Alembert‑Euler 块、神经网络逼近（多层感知机）、C^r 收敛分析、可证明的全局逼近定理，以及针对随机系统的参数化 Stratonovich 积分；

**📊 数据集**

实验数据集为合成数据：从解析解或数值积分（如隐式中点法、Lie‑Trotter 分裂法）生成的轨迹；覆盖线性阻尼谐振子、二次阻尼摆、时变阻尼谐振子以及受迫 Kubo 振荡子；训练样本数量从 400-625 条轨迹到 4761 条扩展数据集；

**📈 对比分析**

与传统非几何残差神经网络（ResNet）进行比较；GFHNN/PGFHNN 在轨迹误差、能量误差、长期稳定性上明显优于 ResNet；在相同精度下，几何网络所需训练数据量约为 ResNet 的 1/7.6（时间相关实验）或更少；

**⚠️ 局限性**

局限性包括：目前仅在欧氏空间（R^n）上实现，扩展到 Lie 组或更一般流形需要进一步工作；随机系统的实现依赖已知 Wiener 路径或其 Stratonovich 积分参数，无法直接处理未知噪声；使用的是一阶 Lagrange‑d'Alembert‑Euler 块，可能限制数值精度；

---

## 171. MaliciousSkillBench: A Comprehensive Benchmark for Malicious Agent Skill Detection

**arXiv ID:** 2608.19901 | [PDF](https://arxiv.org/pdf/2608.19901v1)

**作者:** Yue Wang `[一作]` (Nanjing University), Leo Zhang `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个综合的恶意 Agent Skill 检测基准，整合 13 个公开数据源，进行去重、结构归类、标签冲突清理，并提供可追溯的元数据；随后在三种评估场景（随机、结构不重叠、源不重叠）下评测学习模型与三款现成安全扫描器。

**💡 创新点**

创新点在于：① 将多源、异构的恶意 Skill 数据统一规范化并进行精细去重与结构重用控制，避免样本重叠与标签冲突；② 采用可追溯的冻结版元数据，支持多维度拆分与评估；③ 提出跨源与结构性两级不重叠评估，揭示模型在真实部署场景下的召回与误报双重挑战；④ 对 11 类攻击手段进行统一映射，展示不同源的威胁分布差异。

**🔧 技术方法**

主要技术包括：基于 TF‑IDF 的词/字符特征与线性 SVM/逻辑回归的学习模型；静态文本预处理与规范化；结构重用分析（相似度阈值 0.68）；三款公开扫描器（Cisco 本地行为扫描、SkillFortify 离线扫描、SkillSpector 静态扫描）作为基线。

**📊 数据集**

使用 13 个公开恶意 Skill 数据源，其中 11 个满足核心规则，原始恶意记录 8,414 条，经过去重与规范化后得到 9,740 个 Skill（7,505 残存恶意，2,235 主要善意）。

**📈 对比分析**

评估方法：随机、恶意结构不重叠、源不重叠三种划分；报告 Macro‑F1、恶意召回、善意误报率。随机划分 Macro‑F1 为 0.882–0.932；结构不重叠下降至 0.916；源不重叠进一步降至 0.665。学习模型在源不重叠下恶意召回约 95% 但善意误报率高达 62%；扫描器则在极低误报率下召回率仅为 0–25%。

**⚠️ 局限性**

局限性：① 数据覆盖仅限已选 13 个源，未包含全部恶意 Skill；② 善意样本来源单一，可能不足以代表真实多样化环境；③ 评估仅基于静态文本，未考虑运行时行为；④ 仅评测三款扫描器，配置与版本对结果有影响；⑤ 复杂攻击标签映射覆盖率有限，部分攻击类别缺乏标注。

---

## 172. Truncate Bad, Upweight Good: BoN-Style Distillation via Rank-Based Classification

**arXiv ID:** 2608.19748 | [PDF](https://arxiv.org/pdf/2608.19748v1)

**作者:** Yarin Bar `[一作]` (Technion Israel Institute Of Technology), Yaniv Romano `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在BoN-style distillation中提出TUP（Truncate-bad, Upweight-good Policy）方法，先用阈值截断低质量候选答案，然后在保留的上层尾部按排名软加权，形成一个离线可训练的目标策略。

**💡 创新点**

创新点包括：①把低尾截断和上尾加权拆分为两个可调参数 λ 与 β，减少对单一平滑重权的依赖；②构造了一个可闭式归一化的shifted‑truncated win‑rate变换，使得目标策略与提示无关；③通过将目标策略转化为二分类问题，使用二元交叉熵（BCE）进行全离线训练，无需对齐奖励或在线采样。

**🔧 技术方法**

采用的技术主要有：基于奖励模型的win‑rate估计、shifted‑truncated win‑rate标签、对数几率变换、BCE损失、二分类器训练（Logistic回归）以及基于Beta函数的解析归一化。

**📊 数据集**

使用的数据集包括UltraFeedback、Magpie Air；实验模型包括Llama‑8B Tülu 3 SFT、Mistral‑7B‑Instruct‑v0.2；评估指标包括LC reward、AlpacaEval（LC reward、长度）以及不同奖励模型（ArmoRM、Skywork‑Llama、Skywork‑Qwen）和GPT judge。

**📈 对比分析**

与六个基线（DPO、REBEL、QRPO、QRPO (random)、BoNBoN、DPO/REBEL随机配对）进行对比。TUP在Skywork‑Llama和Skywork‑Qwen奖励模型下，在数据集内外的LC reward上均取得最优或接近最优成绩；在AlpacaEval上同样表现突出，并在长度匹配的对比中显示出奖励提升不是单纯由回复长度引起的。

**⚠️ 局限性**

局限性：①需要在验证集上同时调节截断阈值 λ 与加权锐度 β，调参成本较高；②仍可能受到奖励模型误差导致的reward hacking；③当前 λ 与 β 为全局固定，未考虑按提示学习或自适应设置。

---

## 173. Learning Hierarchical Skill Policies with Offline Quality-Diversity Reinforcement Learning

**arXiv ID:** 2608.19684 | [PDF](https://arxiv.org/pdf/2608.19684v1)

**作者:** Tanachai Anakewat `[一作]` (University of Tokyo), Tatsuya Harada `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文在离线数据中提取高质量多样化技能并用于在线强化学习，提升稀疏奖励任务的样本效率和最终回报。

**💡 创新点**

创新点在于将优势加权的质量多样化目标与无监督多解发现结合，滤除低质量数据并保持技能多样性，并在离线到在线阶段双重数据重用。

**🔧 技术方法**

使用了优势加权变分自编码器、IQL估计优势、互信息最大化、多样性正则、RND探索奖励以及SAC/IQL进行在线学习。

**📊 数据集**

实验使用了 AntMaze、Kitchen、HumanoidMaze、AntSoccer 等稀疏奖励离线数据集。

**📈 对比分析**

与 SUPE、ExPLORe、Trajectory Skills、IQL、BC 等基线对比，QDOS 在所有环境中实现了更高的归一化回报、更快的首次达成率和更好的样本效率。

**⚠️ 局限性**

局限性包括对温度和互信息权重的敏感性、在高质量结构化数据上多样性约束可能降低效果，以及低维离线数据对技能表示的依赖。

---

## 174. Fine-Tuning VLAs with Self-Demonstrated Generative Control for Multi-Task Manipulation

**arXiv ID:** 2608.19490 | [PDF](https://arxiv.org/pdf/2608.19490v1)

**作者:** Prachi Garg `[一作]` (University of Illinois Urbana Champaign), Derek Hoiem `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在目标机器人上利用预训练的视觉‑语言‑动作模型(VLA)进行自监督回放，结合少量专家演示进行联合微调，以保持指令遵循和预训练行为。

**💡 创新点**

创新点在于：使用冻结的预训练 VLA 在目标机器人上生成自监督回放，并将其作为训练目标，从而在不获取原始预训练数据的情况下保留行为先验，显著缓解微调遗忘。

**🔧 技术方法**

采用的技术包括：视觉‑语言‑动作(VLA)框架、流匹配与 FAST 离散动作学习、自监督生成回放与联合多任务微调；实验涵盖真实 ALOHA 机器人与 RoboTwin 仿真环境。

**📊 数据集**

使用的数据集为：少量专家遥控演示（约 14 分钟）和基于预训练任务分布的自监督回放；在仿真中使用 RoboTwin 任务集。

**📈 对比分析**

与单一专家微调、全量专家微调以及 oracle 回放对照实验表明：在真实机器人上多任务成功率提升至 90%，在仿真中旧任务成功率从 16.6% 提升至 70.6%，新任务成功率从 93% 提升至 98%。

**⚠️ 局限性**

局限性包括：需人工挑选自监督指令并过滤不安全行为，对失败回放的依赖可能导致动作重复，以及在极端失败场景下仍可能出现微调遗忘。

---

## 175. TESTNAV: Pareto-Guided Search for Compositional Robustness Testing

**arXiv ID:** 2608.19882 | [PDF](https://arxiv.org/pdf/2608.19882v1)

**作者:** Arooj Arif `[一作]` (Northeastern University London), Alexandros Koliousis `[通讯]` (Northeastern University London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Pareto优化的组合鲁棒性测试框架，通过在扰动配置空间中平衡性能退化与输入保真度，自动识别最具信息量的模型失效点。

**💡 创新点**

将鲁棒性测试建模为二目标优化问题，并使用NSGA‑II在离散扰动配置空间中高效逼近Pareto前沿，从而克服组合扰动指数爆炸与信息不均的问题。

**🔧 技术方法**

使用NSGA‑II多目标进化算法、输入保真度度量（SSIM、KID、chrF、BERT‑F1）、任务性能指标（accuracy、RP_5@1）以及扰动配置空间搜索等技术。

**📊 数据集**

在四个基准上进行评估：Tiny‑ImageNet（视觉）、QQP（自然语言）、HumanEval 与 MBPP（代码生成），并使用对应模型（CaiT‑S36、RoBERTa‑base、CodeGen‑2B‑mono）。

**📈 对比分析**

与随机搜索、贪婪搜索和单目标遗传算法对比，Pareto搜索在大部分指标上Recall@P*更高，AUC‑Recall提升约0.26–0.38，在仅评估35.8%–89.3%配置空间的情况下即可完整恢复Pareto前沿。

**⚠️ 局限性**

局限在于仅处理离散、预定义的扰动类型与级别，无法直接应用于连续扰动空间；此外，极端Pareto点的实用性取决于具体测试目标，需要进一步针对目标子区域优化搜索策略。

---

## 176. A Two-Stage Time-Aware Transformer for Short-Horizon AECOPD Risk Prediction

**arXiv ID:** 2608.19578 | [PDF](https://arxiv.org/pdf/2608.19578v1)

**作者:** Dongyang Wang `[一作]` (Monmouth University), Haowen Pan `[通讯]` (Changzhou Yaoyuanxing Electronic Technology Co Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个两阶段模型，利用家用呼吸机近七天的原始压力与流量波形，先对患者进行高危二分类，再对已识别为高危的患者估计离严重急性加重事件的天数；

**💡 创新点**

创新点在于：①直接使用未压缩的原始波形保留细粒度呼吸动态，②引入时间感知Transformer并结合Time2Vec嵌入提升时序建模能力，③构建两阶段流程（先筛选再精准时点预测），④仅用两列关键通道降低模型复杂度并便于实时部署；

**🔧 技术方法**

技术实现包括：时间感知Transformer（带Time2Vec与位置编码）、多头自注意力与前馈网络、Logistic回归或XGBoost分类器、Transformer回归头、Adam优化器、混合精度训练与早停；

**📊 数据集**

数据集来源于85名COPD患者的家用非侵袭性呼吸机连续记录（5Hz采样），共包含压力、流量、SpO₂、呼吸速率、潮气量、分钟通气量与泄漏等八列，最终仅提取压力与流量用于模型训练；

**📈 对比分析**

与跳点压缩Transformer以及传统机器学习基线（XGBoost、Ridge、Mean/Median等）进行对比。Stage 1中最佳配置（32维、Logistic）F1达0.91；Stage 2中64维Transformer回归实现RMSE 1.00天、MAE 0.87天、R² 0.76，显著优于所有基线；

**⚠️ 局限性**

局限性包括：样本量小（85人，2:1正负比）、缺乏外部验证、仅使用两列输入导致模型可解释性有限、缺少人口学和临床背景信息、模型在早期窗口的稳定性仍需验证。

---

## 177. From Noise to Signal: Improving Security Log Anomaly Detection Using LLMs with Endpoint-Specific Logs

**arXiv ID:** 2608.19938 | [PDF](https://arxiv.org/pdf/2608.19938v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 178. A Fully Automated, Deployment-Aware Testing Pipeline for IoT-Based Automotive Applications

**arXiv ID:** 2608.19752 | [PDF](https://arxiv.org/pdf/2608.19752v1)

**作者:** Denesa Zyberaj `[一作]` (Mercedes-Benz AG), Marco Aiello `[通讯]` (University of Stuttgart)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并验证了一套自动化、部署感知的IoT汽车应用测试流水线，能从自然语言需求和UML图生成Gherkin场景和可执行脚本，支持分布式ECU执行。

**💡 创新点**

将LLM/VLM与人类审阅相结合实现需求驱动测试生成，并通过网络性能自适应调度实现跨地理分布的部署与执行，完整实现端到端可追溯。

**🔧 技术方法**

使用GPT‑4类LLM、Gemini 2.5 Pro/Qwen 2.5 VL等VLM、Eclipse openDuT、Ansible、gRPC/REST、Python unittest、MongoDB、GitHub Actions等技术栈。

**📊 数据集**

以子女存在检测系统（CPDS）的功能与技术需求（9+5条）及其UML图为数据集进行实验。

**📈 对比分析**

在控制需求集中Gherkin生成准确率100%，在更细粒度需求集成实现89%无需人工修正；通过与其他供应商/OEM的分布式ECU环境演示部署成功，覆盖率全功能需求，显示低手工成本与高一致性。

**⚠️ 局限性**

局限包括：仅在GitHub Actions/Windows自托管环境下实现；执行顺序化、资源利用低；MongoDB原型在大规模并发下可能瓶颈；缺乏故障注入与SOTIF覆盖；未覆盖其他CI/CD工具或多OS兼容。

---

## 179. V-REX: Efficient Specialist VLM Training for Veterinary X-Rays

**arXiv ID:** 2608.20069 | [PDF](https://arxiv.org/pdf/2608.20069v1)

**作者:** Tim Elsner `[一作]` (Vyyo AI), Michael Fitzke `[通讯]` (Mars Petcare)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

训练了从零开始的兽医X光诊断报告生成VLM，利用域特定tokenization、生成式预训练与图像自由指导，构建参数仅504M的小模型；

**💡 创新点**

通过专用tokenization、RAPTOR视觉编码、生成式预训练与IFG推理技术，构造了在兽医X光任务上可超越大型通用VLM的轻量模型；

**🔧 技术方法**

使用PyTorch+Lightning框架，decoder‑only transformer、BPE词表、RAPTOR视觉特征、生成式自回归预训练以及图像自由指导（IFG）等技术；

**📊 数据集**

使用约1500万中兽医X光图像与对应诊断报告的专有文本‑图像对数据集；

**📈 对比分析**

对比OpenAI o3与PaliGemma等通用/细调VLM，采用F1评分评估；轻量模型在相同训练天数下实现0.38 F1，超过PaliGemma fine‑tune 0.38且参数仅为其1/10；

**⚠️ 局限性**

仅在兽医X光领域验证，跨领域泛化尚未验证；在极少样本情况下易过拟合；推理时需两次前向实现IFG，略增加计算负担。

---

## 180. Exact Multistate Reliability and Upgrade Design for Heterogeneous HBM Systems via Threshold-Pruned BAT

**arXiv ID:** 2608.19471 | [PDF](https://arxiv.org/pdf/2608.19471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 181. PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents

**arXiv ID:** 2608.19861 | [PDF](https://arxiv.org/pdf/2608.19861v1)

**作者:** Seongjae Kang `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种外部实时验证器（verifier），通过将组织政策编译成工作流图并在用户回合边界持续追踪状态，主动引导大型语言模型代理完成多步流程，确保既不执行禁止动作又不遗漏程序性步骤。

**💡 创新点**

创新点在于：①将政策从自然语言转化为可遍历的工作流图；②将状态持久化与干预逻辑放在外部验证器中，避免代理内部硬编码；③通过预先检查每一步并提供针对性补救，实现在多轮交互中的全流程合规指导；④展示了跨代理、跨领域的迁移能力。

**🔧 技术方法**

技术手段包括：LLM驱动的外部验证器、工作流图（节点/边、子流程、分支等）、离线工作流生成管道、实时推断与干预算法、与代理的对话接口、以及对工具调用的授权与监控。

**📊 数据集**

实验使用了公开的三大客户服务基准数据集：航空（Airline）、零售（Retail）和电信（Telecom），每个域包含可验证（PV）和可变更（Mut）两类任务。

**📈 对比分析**

比较方法：与无验证器的基础代理、与已有工作流控制器FlowAgent以及常规运行时安全检查进行对比。实验结果显示：在所有域的 Pass^4（四轮成功率）从 0.42 提升至 0.62，尤其在结构最完整的电信域从 0.19 提升至 0.61；在 CRAFT 逼真攻击测试中攻击成功率显著降低；并在手工设计的流程顺序审计中获得最高的流程有效率。

**⚠️ 局限性**

局限性包括：仅评估三大英文域，未覆盖多语言或真实用户；使用冻结的工作流，缺乏对模型/seed 变异的泛化评估；验证器为概率性，无法提供硬性合规保证；运行成本（约 0.40 美元/会话）与更高覆盖率的调用频率相关；对流程完整性的审计仍依赖人工设计的规则。

---

## 182. TextRefine: Improving Textual Fidelity, Spatial Placement, and Glyph Rendering for Text Editing in Product Posters

**arXiv ID:** 2608.19637 | [PDF](https://arxiv.org/pdf/2608.19637v1)

**作者:** Honglie Wang `[一作]` (Kuaishou Technology), Yan-Ming Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TextRefine，结合监督微调和强化学习的任务对齐后训练框架，用于产品海报的文本插入和替换。

**💡 创新点**

创新点在于为两种编辑操作分别设计 span‑level 和 glyph‑level 奖励，并加入 gated SSIM 约束，利用 CTC 后验提供细粒度结构监督；同时构建了包含 10 万张多文本海报和低频中文字符的 OpenTextEdit 数据集。

**🔧 技术方法**

技术包括 DiffusionNFT 强化学习、LoRA 微调、OCR+VLM 评估、SSIM 与 CTC posterior 奖励，以及产品遮罩与文本属性的多层输入。

**📊 数据集**

使用 OpenTextEdit 数据集（50K 插入 + 50K 替换样本）进行监督训练，并在 629/1235 插入样例与 200 替换样例上进行评估。

**📈 对比分析**

与 Qwen‑Image‑Edit‑2511、FireRed‑Image‑Edit‑1.0、AnyText2、PosterMaker 等基线相比，TextRefine 在插入匹配率、误删、误加、FID、SSIM、PSNR 等指标上均有显著提升，插入准确率提升至 91.9%/90.0%，替换准确率提升至 74.5%。

**⚠️ 局限性**

局限性：评估仅覆盖 OpenTextEdit、中文单字符替换和 PaddleOCR 结果，未验证跨域、多语言、多字符场景的鲁棒性。

---

## 183. Symposium: Trust via Auditable Records for Communities of AI Scientist Agents

**arXiv ID:** 2608.19511 | [PDF](https://arxiv.org/pdf/2608.19511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 184. When Guidance Goes Off-Scale: Recalibrating Diffusion Transformers under Analog Compute-in-Memory Nonidealities

**arXiv ID:** 2608.19644 | [PDF](https://arxiv.org/pdf/2608.19644v1)

**作者:** Wenshuai Yao `[一作]` (Peking University), Wenyong Zhou `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了模拟计算内存（CIM）非理想性对扩散Transformer（DiT）采样的影响，并提出了一种无重训练、仅在采样器侧调整CFG比例的校准方法；

**💡 创新点**

创新点在于将CFG残差视为对CIM误差高度敏感的失效通道，并通过在目标CIM条件下仅调节CFG比例来实现性能恢复，无需改动模型或采样预算；

**🔧 技术方法**

技术手段包括：CIM权重噪声建模（高斯扰动）、CFG残差分解与轨迹级解释、采样器侧CFG比例校准搜索、使用DPM‑Solver和多种评估指标（FID、KID、CLIPScore、Precision、Density、Coverage、DINO‑clean）；

**📊 数据集**

使用的数据集为PixArt‑Σ、PixArt‑α（512×512）和DiT‑XL/2（256×256），其中文本‑图像任务采用MS‑COCO 2014验证集，类别‑条件任务采用ImageNet验证集；

**📈 对比分析**

与清洁CFG基线以及CFG Rescale、APG、C^2FG、Limited‑Interval等采样器侧控制方法对比，校准方法在CIM噪声为σ_CIM=0.20时将FID分别从59.22→20.49、72.37→21.12、20.89→6.62，恢复至少87%原始质量损失，保持与清洁相近的性能；

**⚠️ 局限性**

局限性包括：仅针对CFG比例进行校准，无法完全消除基线漂移和正交残差；依赖于模拟的CIM噪声模型，实际硬件误差分布可能不同；需额外的校准数据集，尽管规模较小；对极端噪声或更复杂模型的适应性仍待验证。

---

## 185. Aray: Deterministic-First Synthesis of Benign Artifacts for YARA Validation

**arXiv ID:** 2608.19387 | [PDF](https://arxiv.org/pdf/2608.19387v1)

**作者:** Emanuel C. A. Valente `[一作]` (iFood), Marcus Botacin `[通讯]` (Texas A&M University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个确定性 YARA 解释器与正样本合成器，能够根据 YARA 规则生成满足规则的可测试文件（正样本）而无需实际恶意样本。

**💡 创新点**

创新点在于：①将规则规范化与证据抽取交给确定性代码完成，①只在有限的可证明子集之外才调用语言模型，②支持 ELF、PE、通用二进制的可重现生成，并提供上游原始规则的独立验证或验收判据。

**🔧 技术方法**

使用技术包括：结构化词法分析、确定性语法与语义验证、规则规范化与模型辅助抽象、字符串与整数证据的提取与布局分配、基于 GNU 链接脚本的 ELF 生成、两遍 PE 低对齐构建、扫描器专用写入器，以及基于 YARA CLI 的验收 Oracle。

**📊 数据集**

数据集为 416 条来自公开 YARA-Rules 仓库的规则（特定提交），用于阶段化评估。

**📈 对比分析**

比较方法为两阶段流水线：第一阶段规则规范化（182 条不需模型，234 条需模型），第二阶段可构造预检、提取与生成，最终使用原始规则在 YARA 引擎中扫描验证。结果为 406/416（97.6%）正样本产出，406/406（100%）生成成功，未出现 `yara_mismatch` 或构造失败，体现高成功率与可验证性。

**⚠️ 局限性**

局限性包括：仅支持 YARA 的有限子集（不含循环、通用算术、完整正则表达式等）；产生的样本仅为存在性证明，非唯一或最优；模型归约为近似，可能产生不完整或错误的规范化；后端支持不均衡（PE 只实现部分功能、扫描器写入器不可执行）；文件尺寸填充使用随机字节，影响完全可复现；缺乏全局子集蕴含性证明，无法保证所有匹配文件都能被生成。

---

## 186. Accelerating Performance Inference over Closed Systems by Asymptotic Methods

**arXiv ID:** 2608.19682 | [PDF](https://arxiv.org/pdf/2608.19682v1)

**作者:** Giuliano Casale `[一作]` `[通讯]` (Imperial College London), Giuliano Casale (Imperial College London)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文重新审视了多类闭合排队网络的计算理论，提出了一种新的方法来高效计算状态概率的归一化常数，解决了在闭合系统中计算似然性时的计算瓶颈问题。

**💡 创新点**

创新点在于将归一化常数重新表述为单位单纯形上的多维积分，并基于此推导出新的渐近展开和蒙特卡洛采样方法，以高效且准确地近似归一化常数和似然性。

**🔧 技术方法**

使用了渐近展开和蒙特卡洛采样等技术，结合了立方规则来高效评估小型和中型模型中的积分形式。

**📊 数据集**

使用了多类闭合排队网络模型进行实验，具体数据集未详细说明，但提到进行了数千个最大似然估计问题的数值验证。

**📈 对比分析**

与现有方法（如递归算法、生成函数和基于矩的方法）相比，提出的方法在计算归一化常数时具有更低的时间复杂度和空间复杂度，尤其在大模型中表现出更好的准确性和效率。

**⚠️ 局限性**

限制在于所提出的方法在处理无限服务器节点的模型时可能不够有效，且在某些情况下需要调整参数以确保数值稳定性。

---

## 187. MileGPO: Milestone Inference with Local Evidence for Graph-Based Policy Optimization of Long-Horizon LLM Agents

**arXiv ID:** 2608.19803 | [PDF](https://arxiv.org/pdf/2608.19803v1)

**作者:** Bo Qian `[一作]` (Beijing Jiaotong University), Jiqiang Liu `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MileGPO 方法，利用分组的 on‑policy 轨迹图对长时限 LLM 代理进行里程碑发现、可靠性加权信用塑造以及局部进度与同源分支对比，生成中间信用信号以改进奖励分配；

**💡 创新点**

创新点在于①通过成功与失败轨迹的共享状态识别里程碑与陷阱；②使用基于候选状态得分的可靠性校准潜能形状来加权信用；③结合同源分支对比与局部进度校正，精细区分相同距离下的竞争动作，从而解决最终目标距离无法区分的局部信用模糊问题；

**🔧 技术方法**

技术包括基于 GraphGPO 的轨迹图构建、候选状态评分公式、可靠性加权的潜能信用、基于增益的信用塑造、分支对比（BCC）与进度对比（PCC）以及 PPO 训练框架；不需要额外模型或环境交互；

**📊 数据集**

使用 ALFWorld 和 WebShop 两个长时限任务环境进行实验；

**📈 对比分析**

与 GiGPO、GraphGPO 等基线在 ALFWorld 和 WebShop 进行对比。MileGPO 在 ALFWorld 的 ID 成功率 96.30、OOD 94.60，ID–OOD 间隙 1.69（优于 GraphGPO 的 3.78）；在 WebShop 的成功率 90.30、OOD 78.61，任务得分 94.60、90.29，均优于基线；

**⚠️ 局限性**

局限在于依赖轨迹中足够的共享状态与分支信息，若回合数不足或状态重复度低，里程碑发现与分支对比的效果可能受限；仍以最终奖励为全局信号，在极端环境中可能难以识别有效里程碑。

---

## 188. A Federated Learning Framework for Privacy-Preserving Oral Cancer Screening on Smartphones

**arXiv ID:** 2608.19462 | [PDF](https://arxiv.org/pdf/2608.19462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 189. A Plug-in Interpretation of Conditioning in Score-Based Diffusion Models

**arXiv ID:** 2608.19504 | [PDF](https://arxiv.org/pdf/2608.19504v1)

**作者:** Libo Chen `[一作]` (University of Bath), Vinay P. Namboodiri `[通讯]` (University of Bath)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在联合目标-条件扩散中使用分析性插值（plug‑in）条件化机制，学习单一无条件联合分数网络，并在推理时通过解析校正项实现条件约束。

**💡 创新点**

创新点在于将条件信息分离为可解析的校正项，提供透明的条件动力学；同时将其与多速度联合扩散、条件逆时 SDE/ODE 匹配以及对数‑Fokker‑Planck 残差正则化相结合。

**🔧 技术方法**

采用分数匹配训练联合分数网络，构造 VE‑SDE 前向过程并使用多速度噪声调度；推导条件逆时 SDE/ODE，并在预训练先验场景下直接在观测像素上施加校正；加入对数‑Fokker‑Planck 残差正则化以缩小 ODE–SDE 差距。

**📊 数据集**

在 CelebA、FFHQ、ImageNet 以及 MNIST 等图像数据集上评估，涵盖 inpainting、超分辨率、随机噪声 inpainting 等任务。

**📈 对比分析**

与 CDE、CDiffE、CMDE、HCFLOW、DPS、DiffPIR 等基线比较，实验证明在 PSNR、SSIM、LPIPS、JFID 等指标上表现相当或优于现有方法；同时在 MNIST 条件 inpainting 实验中，加入残差正则化后 ODE–SDE 差距显著减小。

**⚠️ 局限性**

局限性包括仅适用于线性高斯前向核，非高斯或非线性观测难以获得闭式校正；当条件噪声高或后验多模态时近似可能失效；多速度调度和残差正则化需要额外调参。

---

## 190. SafeBranch: Branch-Pair Safety Alignment for Embodied Agents

**arXiv ID:** 2608.19729 | [PDF](https://arxiv.org/pdf/2608.19729v1)

**作者:** Hyunse Lee `[一作]` (Dongguk University), Woojin Lee `[通讯]` (Dongguk University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出SafeBranch框架，利用环境回滚构造安全与不安全的分支对（branch pair），在训练时让视觉语言模型代理在安全关键步学会选择安全动作，部署时无需外部安全评论家。

**💡 创新点**

创新点在于：①通过单次不安全轨迹回滚得到同一安全关键步的安全与不安全动作对，①避免了需要两条完全轨迹的自然采样；②采用BranchPO目标在这些分支对上进行对比学习，将安全信号聚焦到单一步骤；③通过LLM判断与过滤提升分支对质量。

**🔧 技术方法**

使用技术包括：环境回滚、GPT‑4o安全评论家与反馈、BranchPO（类似DPO的对比损失）、LLM判定过滤、温度多样化去重等；模型基于大规模视觉语言模型（Qwen3‑VL‑32B）。

**📊 数据集**

实验数据集为IS‑Bench、其两个OOD变体（ObjectShift、TaskShift）以及SafetyALFRED，涵盖多种家庭任务和五类风险类别。

**📈 对比分析**

与未训练代理、Inference‑time安全模块（Self‑Verification、Lookahead）、SFT、Trajectory DPO等基线比较，SafeBranch在IS‑Bench上SSR从0.031升至0.281、SRec从0.273升至0.467（≈10倍提升），在OOD版本提升更显著；在SafetyALFRED上整体安全准确率从0.274升至0.438，尤其在Property Damage和Appliance Misuse类上有显著提升。

**⚠️ 局限性**

限制：训练阶段仍需安全评论家和环境回滚，未消除外部监督成本；无法直接迁移至不支持回滚的真实物理环境，未来需引入近似世界模型或人类重置。

---

## 191. Automatic bioinformatic software named entity recognition from literature

**arXiv ID:** 2608.19201 | [PDF](https://arxiv.org/pdf/2608.19201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 192. Automated Summarization of Financial News Using Large Language Models and Retrieval-Augmented Generation: An Early Empirical Study (Fall 2023)

**arXiv ID:** 2608.19526 | [PDF](https://arxiv.org/pdf/2608.19526v1)

**作者:** Pranav Chandaliya `[一作]` `[通讯]` (George Washington University), Pranav Chandaliya (George Washington University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个多源数据管道，自动收集公司新闻、维基百科背景和股票行情，并使用大语言模型（Falcon-7B、DistilBART、BART-Large）进行金融新闻摘要与GPT-4生成股票摘要；

**💡 创新点**

提出了将结构化股票数据转化为自然语言句子的模板化方法，减少LLM对数值计算的依赖；对比总结链（Summarize Chains）和检索增强生成（RAG）两种生成策略；

**🔧 技术方法**

采用LangChain、FAISS、sentence-transformers等技术实现文本分块、向量检索与链式推理；使用GPT-4文本摘要；构建Streamlit可视化仪表盘；

**📊 数据集**

使用News API（约837篇新闻）、Wikipedia公司简介、Yahoo Finance（4天OHLCV）共10家公司（AAPL、MSFT、GOOGL、AMZN、META、TSLA、JPM、NVDA、WMT、DIS）；

**📈 对比分析**

通过ROUGE-1/2/L与人工评估对比，Falcon-7B+Summarize Chains在覆盖率、准确性、连贯性上优于DistilBART和BART-Large；RAG在大k下出现重复、低ROUGE或幻觉；

**⚠️ 局限性**

受限于模型上下文窗口、检索质量、LLM算术错误与幻觉、API调用成本，且实验仅覆盖单一公司新闻与少量评估指标，未对多公司或更长时间窗口进行充分验证；

---

## 193. Escaping the Quicksand: A Call to Arms

**arXiv ID:** 2608.19674 | [PDF](https://arxiv.org/pdf/2608.19674v1)

**作者:** Peter Sewell `[一作]` (University of Cambridge), Jean Pichon-Pharabod `[通讯]` (Aarhus University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将可执行规范与测试、属性测试、符号执行和证明相结合的灵活反馈循环，以提升 AI 与人类软件开发的可靠性，并呼吁构建行业级语义基础设施；

**💡 创新点**

创新点在于将可执行规范作为测试 oracle，强调可扩展的混合测试‑证明流程，并提出构建可验证语义基础设施的整体框架；

**🔧 技术方法**

使用可执行规范、属性测试、符号执行、自动证明工具（如 ACL2、Z3、Isabelle 等）以及现有编程语言中的断言和契约；

**📊 数据集**

未使用特定数据集；

**📈 对比分析**

未进行实验比较，文献主要为概念性和实践倡议；

**⚠️ 局限性**

局限主要包括缺乏成熟的语义基础设施与工具集成、实现成本高、跨学科协作难度大以及对行业共识和长期投入的依赖。

---

## 194. Towards general embodied intelligence: integrating large language models, knowledge bases, and reasoning capabilities to build the next generation of AI agents

**arXiv ID:** 2608.19794 | [PDF](https://arxiv.org/pdf/2608.19794v1)

**作者:** Fujiang Yuan `[一作]` (Chongqing University of Technology), Zebing Mao `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对近年来大语言模型（LLM）、知识图谱（KB）、推理能力（RA）与具身人工智能（EI）的融合与发展进行系统综述，并提出了一个统一的通用具身智能（GEI）技术架构与五个关键研究方向。

**💡 创新点**

创新点在于：①构建了面向通用具身智能的统一框架，明确展示LLM、KB、RA与EI四个模块如何协同工作；②系统梳理并对比了当前主流LLM、知识库、推理框架及具身实现，填补了跨领域整合的空白；③提出了“轻量化部署、闭环知识集成、混合符号-神经推理、感知-行动对齐、持续学习”五大研究方向，为后续实验与应用提供路线图。

**🔧 技术方法**

主要技术包括：Transformer‑based LLM（如 GPT‑4、PaLM、Claude）、多模态融合扩展（视觉/语音/触觉编码器）、检索增强生成（RAG）与知识图谱检索、符号推理框架（Chain‑of‑Thought、Tree‑of‑Thought、ReAct、L2S）、以及感知‑认知‑动作闭环的具身机器人控制系统。

**📊 数据集**

利用了多种公开数据集和基准：LLM 预训练语料（Common Crawl、Wikipedia 等）、知识图谱（Freebase、Wikidata、专业领域图谱）、多模态评测基准（GLUE、SuperGLUE、MMLU、BIG‑Bench、Image‑Net、Kinetics 等）以及机器人任务数据集（AI‑2‑THOR、E2‑E、OpenAI Gym 等）。

**📈 对比分析**

通过文献对比与案例分析，本文归纳了当前技术的性能表现：LLM 在自然语言任务上已接近人类水平，但在推理精度、常识一致性、长序列记忆以及多模态对齐上仍有显著不足；知识图谱可显著降低“hallucination”，但检索速度与可扩展性受限；具身系统在传感器融合与实时控制上取得进展，但在高维动作空间的自适应规划与安全保障方面仍需突破。论文对这些指标进行了综合评价，并指出现有基准不足以全面衡量跨模态推理与具身行为的协同效果。

**⚠️ 局限性**

局限性包括：①缺乏大规模实验验证，主要以综述与案例说明为主；②在多语言、多领域真实部署与长期学习方面讨论不够深入；③未对不同硬件平台的部署成本与能耗给出定量分析；④对算法可解释性与安全性保障的细节探讨有限。

---

## 195. Evaluating Smart Home Device User Responses to their (Un)Confirmed Privacy Expectations

**arXiv ID:** 2608.19873 | [PDF](https://arxiv.org/pdf/2608.19873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 196. Keeping the Franka Emika Panda alive: a ROS 2 stack with a reliable position interface

**arXiv ID:** 2608.19740 | [PDF](https://arxiv.org/pdf/2608.19740v1)

**作者:** Antonio Langella `[一作]` (University of Salerno), Pasquale Chiacchio `[通讯]` (University of Salerno)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为Frankam Emika Panda机器人恢复并开源ROS 2支持，解决其外部位置控制接口不可靠的问题，构建了可在非实时内核下稳定工作的ROS 2软件栈。

**💡 创新点**

创新点包括：
- 采用异步硬件通信线程，将机器人通信与ROS 2控制循环解耦，消除工作站时钟对机器人周期的干扰；
- 引入速率匹配滤波器，支持低于机器人控制速率的外部控制源；
- 设计位置域参考生成策略，利用机器人已命令状态与外部位置命令的差值来生成平滑、无时钟漂移的参考，显著提升位置控制的可靠性；
- 将上述技术与现有libfranka驱动整合，并提供完整的ROS 2包，支持Motion Planning、Compliance Control、Pick‑and‑Place和Haptic Teleoperation等协作场景。

**🔧 技术方法**

主要技术：ROS 2（Jazzy）+ libfranka + 自定义异步通信线程 + 速率匹配移动平均滤波 + 位置域参考生成公式 + 基于PID/PD的关节/笛卡尔力/扭矩控制器。

**📊 数据集**

实验使用在两台独立Panda机器人上执行相同的18.32 s多关节轨迹，另外在实验室A/实验室B分别进行运动规划、顺应控制、抓取放置和遥操作四大场景；未使用公开数据集，全部为自制轨迹与仿真/真实测试。

**📈 对比分析**

对比方法：将官方同步实现与本工作异步实现在相同轨迹下进行比较，测量
- 服务器侧循环时延（d^EXT）
- 未服务周期数
- 机器人接受的指令频谱
- 轨迹跟踪误差
实验表明：异步实现将d^EXT从0.065 s降低到≈0.013 s，未服务周期从5.91降至0.41；频谱功率下降约十倍；位置轨迹无抖动且无安全停机。四大应用场景均成功完成，无阻碍。

**⚠️ 局限性**

限制：
- 位置参考生成仅经验验证，缺乏正式的稳定性/平滑性证明；
- 速率匹配滤波器引入组延迟，随速率比增长；
- 异步线程仍受工作站调度延迟影响，无法完全消除未服务周期；
- 在高丢包率环境下参考生成缺乏自适应修正；
- 参数λ需经验调优，影响响应速度与误差补偿权衡。

---

## 197. Projector Is All You Train

**arXiv ID:** 2608.19726 | [PDF](https://arxiv.org/pdf/2608.19726v1)

**作者:** Nyx Iskandar `[一作]` (Ramen VR), Slater Victoroff `[通讯]` (iph.so)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过仅微调投影层，验证是否能在不更新语言模型本体的情况下，将多模态大语言模型（MLLM）适配3D点云模态。

**💡 创新点**

证明投影器单独训练即可获得与传统联合训练相当或更优的3D理解能力，并避免语言模型能力漂移；同时训练速度约为联合训练的两倍。

**🔧 技术方法**

采用 Point-BERT 编码器、3 层 MLP 投影器、LoRA 适配器；在 Llama‑3.1‑8B‑Instruct、Qwen3.5‑4B/9B 等不同 LM 骨干上使用负对数似然训练，并评估 3D 分类、字幕、语言、视觉和空间基准。

**📊 数据集**

使用 PointLLM‑V2 数据集（混合 Stage1/2 的 Objaverse/Objaverse‑XL 采样点云），并在 ModelNet40、OmniObject3D 等公开 3D 数据集上进行评测。

**📈 对比分析**

在相同 16 小时 GPU 预算下，投影器单独训练与联合训练在 3D 分类/字幕精度上竞争，投影器单独训练通常更快、样本效率更高；联合训练导致 LM 在 MMLU、WinoGrande、MMMU 等基准上显著退化。

**⚠️ 局限性**

局限性包括：仅验证 3D 点云模态，未探究其他模态；投影器单独训练仍可能在某些任务上受限；缺乏对模型内部机制的解释；基准评估依赖 LLM 判定，可能存在偏差。

---

## 198. Social.Wiki: A Web Held in Common

**arXiv ID:** 2608.19433 | [PDF](https://arxiv.org/pdf/2608.19433v1)

**作者:** Theia Henderson `[一作]` (MIT CSAIL), David R. Karger `[通讯]` (MIT CSAIL)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一个去中心化、可共同编辑的网页平台Social.Wiki，允许任何人直接编辑和治理网站，支持多种社交应用并提供多元治理与细粒度安全保障；

**💡 创新点**

通过“共享所有权”和“多元治理”模型取代传统的单一网站所有者，结合去中心化数据库Graffiti、数据守护器（Data Guard）与沙箱化转录，实现既能自由协作又能防范恶意代码和数据泄露；

**🔧 技术方法**

核心技术包括HTML+CSS+JavaScript的可插拔文档（Transclusion）与沙箱化iframe、去中心化数据库Graffiti、基于权限的Data Guard、治理透镜（Governance Lens）与可替换的治理策略；

**📊 数据集**

使用Graffiti网络中的数据（版本历史、编辑元数据、用户信任关系、聊天记录、地理位置信息等），并在Hackathon与实际社区中创建多种示例站点（聊天、共享日程、打车、个人主页）作为实验数据；

**📈 对比分析**

在Hackathon（12名参与者）和真实社区（约100名用户）进行评估，发现参与者对编辑体验评价高（平均4/5），但缺乏大规模对比基准；性能主要受客户端渲染和网络延迟影响，整体体验与传统web相当，但在高并发编辑或大数据处理时尚需优化；

**⚠️ 局限性**

局限包括：尚未实现真正的所有权分配与收益分配机制；安全模型依赖用户确认，可能导致“确认疲劳”；缺乏后端计算支持（无法支持复杂算法推送、推荐系统）；对规模化治理与冲突处理的实测不足，未来需进一步验证。

---

## 199. The Complexity of Boolean Connectivity Problem of $k$-Horn Formulas

**arXiv ID:** 2608.19569 | [PDF](https://arxiv.org/pdf/2608.19569v1)

**作者:** Takashi Horiyama `[一作]` (Hokkaido University), Junichi Teruyama `[通讯]` (University of Hyogo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究k-Horn公式的真值集合的连通性问题，提供精确指数算法和多项式时间解法，并证明某些受限实例的coNP完备性

**💡 创新点**

提出基于PPZ的O*(2^(1-1/2k)n)精确算法，给出多项式解法的结构性见解，并首次证明Conn 3‑Horn‑E3为coNP‑完整，划分了复杂度边界

**🔧 技术方法**

使用Deterministic PPZ算法、组合自含集分析、变量出现次数约束的图结构、以及从Monotone NAE 3‑SAT的多项式归约

**📊 数据集**

无实验数据集，全部为理论证明与构造性归约

**📈 对比分析**

相较于Makino等人针对k‑CNF的算法，本研究将时间复杂度从O*(2^(1-c_k)n)降低至O*(2^(1-1/2k)n)，并在受限场景下实现多项式时间求解，性能大幅提升

**⚠️ 局限性**

对每条子句恰好三字母且每个变量恰好出现四次的情况（Conn E3‑Horn‑E4）仍未确定复杂度，且未给出更一般约束下的完整复杂度分类

---

## 200. Hype Meets Reality: Large Language Models as Mutators in Search-based Automated Program Repair of Simulink-Stateflow Models

**arXiv ID:** 2608.19347 | [PDF](https://arxiv.org/pdf/2608.19347v1)

**作者:** Ayesha Irshad `[一作]` (Mondragon University), Aitor Arrieta `[通讯]` (Mondragon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将大语言模型（LLM）作为Mutation Operators集成进Search‑based APR框架FlowRepair，直接替换部分手工设计的Mutation Operator，并评估其在Simulink/Stateflow模型修复中的表现。

**💡 创新点**

创新点在于探讨LLM动态生成的修复变体能否替代传统规则化Mutation Operator，验证其在CPS修复场景中的可行性与局限。

**🔧 技术方法**

使用GPT‑5.5、GPT‑5.4‑mini、GPT‑4.1‑mini三种LLM进行提示式生成，结合FlowRepair的搜索算法、Spectrum‑Based Fault Localization及基于时序的修复目标。

**📊 数据集**

数据集为19个真实故障Stateflow模型，涵盖四类CPS案例（pacemaker、fridge、door、elevator），来源为FlowRepair公开基准。

**📈 对比分析**

在相同一小时时间预算下，FlowRepair平均在18/19模型生成可行补丁，LLM版仅在4–6个模型产生可行补丁，性能显著落后；LLM在局部语义修复上偶有提升，但总体一致性不足。

**⚠️ 局限性**

限制在于LLM缺乏精确符号编辑、行为反馈，生成噪声搜索空间，导致多模型无效或过拟合；直接替换规则化Mutation导致搜索效率下降，需构建更具结构约束的混合策略。

---

## 201. Far from the Crowd: Scalable Self-Supervised Learning via Geographic Isolation

**arXiv ID:** 2608.19766 | [PDF](https://arxiv.org/pdf/2608.19766v1)

**作者:** Daniele Rege Cambrin `[一作]` (AIKO), Mattia Varile `[通讯]` (AIKO)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究提出了一种基于地理孤立度的自监督学习课程调度方法，利用遥感图像的地理坐标对样本进行难度排序，并在 MoCoV2 与 MAE 两大预训练框架上进行实验，验证了该方法能加速模型收敛并提升下游任务性能。

**💡 创新点**

创新点在于：①仅凭地理坐标即可计算出样本的难度指标（地理孤立度），无需图像解码、模型反馈或标签；②该指标计算复杂度为 O(D log D)，比传统视觉复杂度（压缩比）快 140 倍；③可无缝融合到对比学习与重建学习两种自监督范式中。

**🔧 技术方法**

核心技术包括：BallTree 进行半径查询以获得地理孤立度；动态权重调度（annealed curriculum）实现样本难度随训练进度调整；MoCoV2 与 MAE 两种预训练模型；线性探测评估下游性能；CKA 与有效秩分析用于可解释性和嵌入空间特征评估。

**📊 数据集**

实验数据集：SSL4EO Sentinel‑2 子集（约 250k 样本）；下游任务采用 CopernicusBench 上的 BigEarthNet‑S2（多标签分类）、DFC‑2020‑S2（语义分割）和 LCZ‑S2（多类别分类）。

**📈 对比分析**

通过与均匀采样基线以及视觉复杂度（压缩比）课程的对比，结果显示：在 MoCoV2 与 MAE 上，地理孤立度课程均实现了更快的收敛速度，并在最终下游任务上提升 1–5 分（例如 BigEarthNet‑S2 mAP 提升至 +5），并且预处理成本仅为压缩比的 1/140；与掩码比例课程结合时可进一步提高性能。

**⚠️ 局限性**

局限性包括：仅考虑空间孤立度，未融入多模态（SAR、激光雷达）或时序信息；需要预先构建 BallTree，处理极少样本或极端分布时可能效果受限；未来工作需探索多模态、时序、持续学习等扩展方向。

---

## 202. Gravity-aware partially calibrated absolute pose estimation from affine- or rotation-covariant features

**arXiv ID:** 2608.20056 | [PDF](https://arxiv.org/pdf/2608.20056v1)

**作者:** Marcus Valtonen Örnhag `[一作]` (Ericsson Research), Stefan Adalbjörnsson `[通讯]` (Ericsson Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了两种针对未知焦距的绝对位姿估计的新型最小求解器，分别利用单个仿射对应和两个方向协变特征，并结合IMU提供的重力方向，实现了在单帧图像中同时估计相机姿态和焦距。

**💡 创新点**

创新点在于：①首次将仿射/方向协变特征与重力信息结合用于未知焦距的绝对位姿问题；②推导了新的多项式约束并设计了高效的四次多项式求解方案；③通过单个或两个特征即可完成估计，显著降低了样本数和计算成本；④利用IMU重力约束把自由度从七降至五，实现了更稳健的求解。

**🔧 技术方法**

技术手段包括：仿射/方向协变特征约束推导、重力约束下的旋转参数化（tangent half‑angle）、多项式求解器（四次方程解析解）、非线性后处理（Levenberg‑Marquardt），以及在RANSAC/Graph‑Cut‑RANSAC框架下的稳健估计。

**📊 数据集**

实验使用了合成数据以及公开真实数据集：Cambridge Landmarks 和 Aachen Day‑Night v1.1，分别评估了定位精度、焦距误差和执行时间。

**📈 对比分析**

与现有方法（P4Pf、P3.5Pf、UP2P、UP2.5Pf、UP1SIFT、P1AC）进行比较，实验表明本文求解器在多数场景下实现了更快的速度（尤其在RANSAC中减少了样本量），同时保持或略优于其它方法的旋转、平移和焦距估计精度；在合成噪声测试中显示出更高的数值稳定性和对图像/IMU噪声的鲁棒性。

**⚠️ 局限性**

局限性包括：①要求IMU提供准确的重力方向；②仅适用于已知或约束重力方向（即“upright”假设）且可能对非平行重力情况表现不佳；③相对于纯点基方法，计算量略大；④在极端光照或特征匹配失败的情况下，依赖特征的局部几何信息仍可能导致误差累积。

---

## 203. How to Navigate Uncertainty About AI Consciousness

**arXiv ID:** 2608.19215 | [PDF](https://arxiv.org/pdf/2608.19215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 204. Beyond Memory Majority: Latent-Source Reasoning for Multi-Agent Memory Arbitration

**arXiv ID:** 2608.19701 | [PDF](https://arxiv.org/pdf/2608.19701v1)

**作者:** Chenchen Lin `[一作]` (University of Hong Kong), Edith Cheuk Han Ngai `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 CAMA 的内存仲裁框架，用来消除长期多代理系统中的内存相关性偏差，提升推理准确性。

**💡 创新点**

创新点在于将检索到的内存分解为查询相关的潜在证据槽，利用神经符号方法结合本源先验估计有效独立证据数量，并通过序列化的恢复策略主动补齐缺失的独立证据。

**🔧 技术方法**

采用自注意力编码器+符号先验进行证据分配、Hill 多样性度量评估证据多样性、基于奖励的策略梯度优化恢复策略，以及对齐奖励与终端决策的 RL 训练。

**📊 数据集**

在 MemoryAgentBench、LongMemEval 与 LoCoMo 三个长期记忆基准上进行实验，并在每个基准下构造了包含相关与互补内存的偏差增强版本。

**📈 对比分析**

与 Vanilla RAG、Majority Voting、HippoRAG、Mem0、MAD、MADAM‑RAG 等最先进方法对比，CAMA 在 EM、F1、BERTScore、判定分数以及 CMR、RS、IEG、ERR 等指标上均显著提升，尤其在缓解“假多数”情形下表现突出。

**⚠️ 局限性**

局限性包括对本源追踪信息的依赖、恢复过程带来的额外推理开销、以及在极端记忆稀疏或多源异构环境下可能出现的槽分布碎片化问题。

---

## 205. Time-Series Retrieval for Grounding Multimodal Language Models in Remaining Useful Life

**arXiv ID:** 2608.19218 | [PDF](https://arxiv.org/pdf/2608.19218v1)

**作者:** Valeriu Dimidov `[一作]` (University of Luxembourg), Raphaël Frank `[通讯]` (University of Luxembourg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了利用时间序列检索增强的多模态大型语言模型进行剩余使用寿命预测。

**💡 创新点**

首次将检索增强生成（RAG）与多模态LLM结合，用历史衰变片段作为视觉对比证据。

**🔧 技术方法**

采用LSTM自编码器生成嵌入、kNN检索、绘制对比图像，并将其作为视觉提示输入Gemini系列MLLM。

**📊 数据集**

在C‑MAPSS FD001 数据集上进行实验评估。

**📈 对比分析**

与随机引用的基线相比，检索增强在RMSE、MAE、R²等指标上持续提升，尤其在更大模型上效果更显著。

**⚠️ 局限性**

局限于仅评估FD001、检索覆盖受限、模型推理非确定性、缺乏系统性解释质量评估。

---

## 206. Rethinking the Evaluation and Optimization of LLM-Based Social Simulation

**arXiv ID:** 2608.19689 | [PDF](https://arxiv.org/pdf/2608.19689v1)

**作者:** Pei Wang `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何在主观人类行为场景下评估并训练LLM进行社会模拟，提出并验证一种基于主体性自适应软标签的训练方法。

**💡 创新点**

引入主体性系数量化任务主观性，并证明精确标签和准确率评估在高主体性下失效，提出SALT在自适应邻域中聚合软标签，恢复对分布的训练。

**🔧 技术方法**

结合LLM生成分布归一化、语义相似度嵌入、Lipschitz连续性假设、KL/JSD/TVD/MMD评估、Qwen3-8B模型以及自定义邻域半径公式等技术。

**📊 数据集**

构造了SubjSim 19,300人设-问题对的主观性问卷数据，包含每个情境的完整回答分布。

**📈 对比分析**

在SubjSim上与零样本、SFT、DPO、PPO、DSA等基线对比，SALT在所有四种分布距离指标上显著优于基线，尤其在高主体性情境下KL下降近97%。

**⚠️ 局限性**

仅处理一次性问题情境，缺少多轮交互、开放式回答的评估；对主体性估计与邻域选择仍依赖超参数；数据来源于有限的193名自愿者，代表性有限。

---

## 207. Digital Tides: A Fluid-Dynamic Framework for Flux-Aware Infrastructure Provisioning in UAV Logistics Networks

**arXiv ID:** 2608.19638 | [PDF](https://arxiv.org/pdf/2608.19638v1)

**作者:** Wen-Yu Dong `[一作]` (China Telecom Research Institute), Sheng Chen `[通讯]` (University of Southampton)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了物流无人机（UAV）网络中能效与可靠性的冲突，并提出基于流体动力学的工作负载模型与信息流量（flux）感知的异步激活策略，实现对波前的零延迟跟踪与激活。

**💡 创新点**

创新点包括：①将物流潮汐视作可压缩流体，利用连续方程求解宏观速度场；②提出信息流量作为前瞻性触发信号，具有空间相位领先特性；③设计异步阈值控制（激活与保持阈值不同），形成前瞻性安全环与收缩时的滞后保留；④引入QoS惩罚的有效能效指标，实现能效与可靠性的Pareto最优平衡。

**🔧 技术方法**

采用的技术：流体动力学连续模型、连续方程求解、信息流量向量计算、异步阈值触发逻辑、有限网络覆盖概率推导、闭式能效与可靠性分析、QoS惩罚式优化。

**📊 数据集**

实验使用仿真数据，构建20 km×20 km服务区，基于预设的UAV密度、扩展参数、路径损耗等参数进行Monte‑Carlo仿真；未使用公开实验数据集。

**📈 对比分析**

与四种基线（始终开启、反应式激活、稳健裕度、交通快照预测）比较。实验结果显示：波前失效率由约20%降至几乎0%；能效在保持高服务率的同时，仅略低于始终开启策略；在QoS惩罚下，提出的flux‑aware策略达到Pareto最优。

**⚠️ 局限性**

局限性包括：假设基站布置为均匀PPP、工作负载连续可压缩、无突发分散或剧烈重组、启动延迟均匀、传感估计误差未充分考虑、仅考虑径向对称潮汐；在高度几何受限或非均匀基站布局下需进一步调整或引入混合离散模型。

---

## 208. One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows

**arXiv ID:** 2608.19741 | [PDF](https://arxiv.org/pdf/2608.19741v1)

**作者:** Zhuochun Li `[一作]` (University of Pittsburgh), Tommy Guy `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 Thinkingbox 框架和 Thinkingbox‑bench 基准，用于可执行的工具‑代理‑用户交互评估，聚焦状态化业务流程。

**💡 创新点**

创新点在于提供隔离的 MCP 兼容工具会话、终端状态与副作用的可执行检查，以及通过多次尝试衡量可靠性（pass@k）的方法。

**🔧 技术方法**

技术手段包括基于 POMDP 的任务定义、可执行检查（side‑effect、对话与数据库状态），以及统一的沙盒调度器。

**📊 数据集**

数据集为 507 条跨五个业务域（零售/电商、旅行与酒店、汽车保险、数字银行、IT/HR）任务，包含初始后端状态、工具接口和可执行评判。

**📈 对比分析**

通过对 12 款专有和开源 LLM 的 20 次重复试验进行比较，最佳模型 GPT‑5.4 通过率达 65.36%（单次），但仅 25.25% 的任务能在 20 次尝试全部通过，显示存在显著的发现‑可靠性差距。

**⚠️ 局限性**

局限性包括：基准主要覆盖业务流程，缺乏真实用户日志；工具和后端实现为模拟，可能与生产环境差异；评估侧重最终状态，未全面覆盖非代码任务的安全性与解释性。

---

## 209. Bounded Sovereignty and the Control Tax: Pricing AI Oversight When the Deployer Does Not Own the Model

**arXiv ID:** 2608.19216 | [PDF](https://arxiv.org/pdf/2608.19216v1)

**作者:** Zhen Wen Lim `[一作]` `[通讯]`, Zhen Wen Lim

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在受限主权（bounded sovereignty）条件下，AI部署控制协议的可行性，并通过大规模合成支付争议工作流程实验评估不同访问约束对安全性、诊断、干预和成本的影响。

**💡 创新点**

创新点在于提出“bounded sovereignty”四层访问分类（数据、模型、基础设施、交互），构建控制协议与访问需求对应矩阵，定义主权折扣成本概念，并通过1.35百万案例的合成访问消融实验验证这些理论。

**🔧 技术方法**

使用技术包括：自定义合成支付争议工作流程、七维访问配置（日志、工具调用、网关、追踪、版本控制、供应商支持、工具范围）、十种控制协议（监控、重采样、人类审计、范围限制等）、三类代理行为（诚实、粗心、隐蔽）以及阈值和审计预算的敏感性分析。

**📊 数据集**

使用的数据集为完全合成的支付争议案例集，包含1.35百万条案例，模拟四类灾难性行为（错误持有/释放、系统性误分、敏感数据泄露、结算相关风险），不涉及真实支付或个人数据。

**📈 对比分析**

通过对20个访问配置、10种控制协议、3种代理行为以及多阈值/审计预算组合进行实验，比较灾难性失败率、诊断成功率、干预成功率、成本代理等指标；结果显示完整日志和执行网关显著提升安全性和干预效果，缺失追踪/版本控制导致诊断困难，范围限制能降低失败但降低可用性，整体展示了访问约束与控制性能之间的权衡。

**⚠️ 局限性**

局限性包括：实验为合成模拟，未使用真实支付数据或真实LLM输出；访问配置为分块设计，未覆盖所有可能组合；成本模型为代理估计，主权折扣成本以可达性衡量而非实际费用；缺少跨行业外部验证，结果对特定支付场景最具代表性。

---

## 210. Clustering and Token Denoising for Faster and More Robust VLMs

**arXiv ID:** 2608.19285 | [PDF](https://arxiv.org/pdf/2608.19285v1)

**作者:** Baptiste Rossigneux `[一作]` (Université Paris-Saclay, CEA, LIST), Emmanuel Casseau `[通讯]` (Univ. Rennes, CNRS, INRIA, IRISA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ClustRS，一种训练‑free 的视觉令牌裁剪与去噪框架，能在极低令牌数下保持视觉‑语言模型的鲁棒性。

**💡 创新点**

创新点包括：① 用注意力加权的 Huber‑k‑means++ 进行鲁棒聚类（C+H），在噪声场景下平衡多样性与重要性；② 采用一次性残差收缩（RRS）在聚类中心附近平滑剩余噪声，从而提升令牌质量。

**🔧 技术方法**

技术：注意力加权 k‑means++ + Huber 损失、一次性残差收缩（基于簇尺度的自适应收缩因子）、不需要额外训练的轻量级操作。

**📊 数据集**

使用了 MM‑VET（开放式视觉推理）和 ScienceQA‑IMG（多选科学问题）两个基准数据集，且在不同噪声（高斯、盐椒、亮度/对比度变换、遮挡）下进行评测。

**📈 对比分析**

与 FasterVLM、DivPrune、VisionZip 等现有训练‑free 裁剪方法相比，ClustRS 在极压缩（k=16）和重噪声条件下平均提升 15–20%，在 97% 令牌压缩下仍保持或超过基线；在中等压缩（k=50、144）亦显著优于或相当于对手，尤其在盐椒噪声和高斯噪声上表现更稳健。

**⚠️ 局限性**

局限性：① 在令牌预算较大时，聚类一次性选取代表可能丢失细粒度空间信息；② 对某些轻微噪声或高预算场景提升有限，RRS 的收益随令牌数增大而递减；③ 仍需进一步探索空间关系保留和更高级的去噪策略。

---

## 211. COPA: Continual Preference Optimization for Adaptive Prompt Injection Defense

**arXiv ID:** 2608.19982 | [PDF](https://arxiv.org/pdf/2608.19982v1)

**作者:** Roshan Sood `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种持续性偏好优化框架COPA，用于在不断演进的注入攻击流中对LLM进行实时防御；

**💡 创新点**

创新点在于将注入防御视为终身学习问题，采用Margin‑Weighted Replay与GRPO优化结合低秩LoRA适配器，能在不遗忘旧攻击的前提下适应新攻击；

**🔧 技术方法**

核心技术包括Group Relative Policy Optimization (GRPO)、低秩LoRA适配器、边缘优先的经验重放缓冲区以及可自适应的log‑likelihood margin评估；

**📊 数据集**

使用DeepTeam Red Teaming Framework的网络安全偏好数据集进行初始对齐，随后在CyberSecEval基准上的15种注入变体进行终身训练，并在MMLU‑Pro与GPQA上评估通用能力；

**📈 对比分析**

在对比LlamaGuard、DataSentinel与SecAlign等主流静态防御时，COPA在终身学习情景下的攻击成功率(ASR)降至0.035（比SecAlign低6.3×），实现正向后向迁移(+0.028)与平均性能(AP)0.850，同时保持或提升QA准确率；

**⚠️ 局限性**

局限性包括依赖人工构建的偏好数据集与有限的重放缓冲容量，未覆盖极端或未知攻击变体；在仅测试三种LLM架构时效果表现良好，但在更大模型或不同预训练策略上的泛化仍待验证。

---

## 212. What You Can't See Is What You Learn: Restricted Evidence Visibility Favors Compositional Generalization in Shared-Genome Language-Model Societies

**arXiv ID:** 2608.20054 | [PDF](https://arxiv.org/pdf/2608.20054v1)

**作者:** Narcis Marincat `[一作]` `[通讯]`, Narcis Marincat

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在共享冻结的 Qwen2.5-0.5B 语言模型上构建的四细胞多模块体系中，限制每个细胞只能访问其指派的证据段（与全可见对照相比）会如何影响梯度训练所发现的解决方案。

**💡 创新点**

首次通过严格匹配的可见性干预、预先封闭的组合任务评估以及基于数值中介的包裹互换干预，证明可见性限制显著提升了学习可重用、价值索引通信协议的概率，并揭示了不同可见性条件下通信协议的形式差异。

**🔧 技术方法**

技术包括：共享冻结的 Qwen2.5-0.5B + rank‑8 LoRA、每个细胞仅通过 2×896 维连续向量进行通信、基于 Transformer 的四细胞递归架构、注意力掩码控制可见性、preregistered gates、以及同值与反事实值的包裹互换干预。

**📊 数据集**

数据集为自生成的 ℤ₁₇ 整数与 12 个仿射操作的自然语言描述组合任务，所有程序、句法与答案标签在训练前进行哈希固定，采用 held‑out ordered programs 与 held‑out phrasings 进行评估。

**📈 对比分析**

采用十对匹配实验（相同初始化、不同数据顺序），仅改变注意力掩码；在深度 2/3 上，受限可见性模型平均提升 ≥0.20，深度 3 的中位优势为 0.605；所有受限模型在切断通信后退回至 1/17 的随机猜测；唯一全可见模型可达到更高性能，但出现频率极低。

**⚠️ 局限性**

局限性：仅在单一四细胞架构、单一任务域、固定 20,000 次更新预算下验证；受限可见性并非必要，且仅有 10 对匹配样本，统计推断受限；高性能全可见模型虽然存在但未普遍；包裹互换干预仅在 6 个受限模型中测试；未评估跨任务或跨模型的普适性。

---

## 213. Delegating or Doing? Understanding User Behavior in Hybrid Human-Agent Interfaces

**arXiv ID:** 2608.19551 | [PDF](https://arxiv.org/pdf/2608.19551v1)

**作者:** Gavin Raine Dizon `[一作]` (Future University), Yasuyuki Sumi `[通讯]` (Future University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究构建了一个基于REST API的内容管理系统，并通过Model Context Protocol将其与Gemini 3.1 Flash‑Lite LLM 代理连接，随后在三种交互模式（传统图形界面、AI‑First、Hybrid）下对73名本科生执行16个CRUD任务，收集任务时长、UI事件与对话轮数等日志数据，分析用户在混合人机界面中如何在直接操作与代理委派之间切换。

**💡 创新点**

创新点在于①提出了可将现有REST API轻量化接入LLM代理的通用MCP架构；②首次系统性比较三种交互模式对交互努力（点击、导航、滚动）与任务完成时间的影响；③发现交互努力显著降低但速度无显著提升，为人机界面设计提供新的评估视角。

**🔧 技术方法**

技术实现包括Gemini 3.1 Flash‑Lite LLM代理、Model Context Protocol工具调用、双通道REST API与前端图形界面，以及成功调用后自动重定向以实现视觉反馈。

**📊 数据集**

数据集为16个CRUD任务（Create、Read、Update、Delete），在同一CMS中按固定顺序执行，参与者共73人。

**📈 对比分析**

通过负二项/泊松混合模型比较任务完成时长、UI事件计数与对话轮数，结果显示AI‑First模式的点击和页面导航最少，但三种模式的任务完成时间无显著差异。

**⚠️ 局限性**

局限性包括样本单一（本科生、英语母语）、任务顺序固定、委派度量仅为对话轮数、未收集自评或信任数据、模型与领域局限、组分配不均导致组别大小差异。

---

## 214. Measuring What a Specification Determines: A Formal Semantic-Block Model and an Execution-Judged Benchmark

**arXiv ID:** 2608.19475 | [PDF](https://arxiv.org/pdf/2608.19475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 215. Experimental Verification of Fast Voltage Droop Correction Circuits

**arXiv ID:** 2608.19954 | [PDF](https://arxiv.org/pdf/2608.19954v1)

**作者:** Shreyas Srinivas `[一作]` (CISPA Helmholtz Center for Information Security), Christoph Lenzen `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现并实验验证了一种基于IHP 130 nm工艺的全数字快速电压跌落补偿电路，能够在1.2个时钟周期内检测并将时钟频率降低；

**💡 创新点**

提出并验证了基于差分感知的遮罩锁存器设计，消除了对高/低阈值反相器的依赖，提升了在电压跌落下的适应性与鲁棒性；

**🔧 技术方法**

采用差分锁存器、相位累加器、延迟链与时钟自适应模块（FAM），并通过实验测试工具（Adventest v93000）对时钟频率与波形进行测量；

**📊 数据集**

未使用公开数据集，而是通过人工控制的电压跌落信号（Vdd 0.9–1.3 V）和高阻抗逻辑电压（Z 0–1.4 V）作为实验输入；

**📈 对比分析**

与传统的高电压保护或预测方法对比，实验显示该电路在跌落触发后能在一周期内降低频率至约66 MHz（原设计为80 MHz），保持时钟无抖动；在长时间跌落时频率变化平滑，实验误差低于0.1 %；

**⚠️ 局限性**

受限于测试端口频率极限导致输出噪声、难以直接观测深度元稳态，以及实验条件下无法对内部锁存器的元稳态进行精准测量；在高频或大规模GALS系统中的部署与同步时钟树的协调仍是未解决的问题。

---

## 216. Unsupervised Anomaly Detection Using Flow Matching on Tabular Data

**arXiv ID:** 2608.19801 | [PDF](https://arxiv.org/pdf/2608.19801v1)

**作者:** Philip Konz `[一作]` (University of Mannheim), Margret Keuper `[通讯]` (University of Mannheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在存在训练集污染的情况下，基于流匹配的无监督异常检测方法的鲁棒性；

**💡 创新点**

发现异常评分函数的选择至关重要，轨迹级别的偏差与重构评分显著提升鲁棒性，并使原本用于生成的 Forest‑Flow 成为竞争力的异常检测器；

**🔧 技术方法**

采用了 TCCM 与 Forest‑Flow（XGBoost 回归器）两种流匹配模型，并实现了 Decision、Deviation 与 Reconstruction 三种异常评分；

**📊 数据集**

在金融交易数据集 Campaign、Synthetic Business Transaction（B2B）以及物理 Waveform 数据集上进行实验；

**📈 对比分析**

通过 AUROC 与 AUPRC 与零污染与全污染两种训练设置比较，轨迹级评分尤其是 Reconstruction 在多种场景下均优于原始 Decision 分数，且在 B2B 上甚至超越 TCCM；

**⚠️ 局限性**

实验范围仅涵盖三份数据集，未在更大规模金融基准上验证，且计算成本（Monte‑Carlo 采样、Euler 步长）仍需进一步优化。

---

## 217. BASC : Behavior-Aligned Quantization and Pruning for Low-Bit Spiking Neural Networks

**arXiv ID:** 2608.19239 | [PDF](https://arxiv.org/pdf/2608.19239v1)

**作者:** Linliang Chen `[一作]` (Beihang University), Wang Kang `[通讯]` (Beihang University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的低位Spiking神经网络压缩框架BASC，通过行为对齐的量化与剪枝实现更高效的模型压缩；

**💡 创新点**

创新点在于解决量化与剪枝的局部准则与网络行为不匹配问题，分别引入Temporal-Behavior Scale Correction (TSC) 调整量化尺度与Boundary-Level Inter-Channel Correction (BIC) 在剪枝阈值附近重新评估通道重要性；

**🔧 技术方法**

使用了均匀量化、可学习的尺度参数、时间任务损失、核范数重构误差、Singular-Value Spectrum（SVS）与CHIP-CI互通道信息等技术；

**📊 数据集**

在CIFAR-10、CIFAR-100、TinyImageNet、ImageNet-1K及神经形态数据集DVS-CIFAR10上进行实验；

**📈 对比分析**

与基线QP‑SNN及多项先前工作进行对比，BASC在2/3/4位量化下在所有数据集上均取得更高准确率或更小模型尺寸，结构化剪枝后亦保持甚至超过高位基线性能；

**⚠️ 局限性**

局限在于仍需依赖手工设定的阈值与比例，BIC只在阈值附近修正，量化尺度学习在极低位时可能受限，且未在更大规模或不同任务（如语音、强化学习）中验证。

---

## 218. Online Scheduling for Throughput Maximization of Time-varying Markovian Channels with Unknown Statistics

**arXiv ID:** 2608.19398 | [PDF](https://arxiv.org/pdf/2608.19398v1)

**作者:** Tasmeen Zaman Ornee `[一作]` (Ohio State University), Ness B. Shroff `[通讯]` (Ohio State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种在线调度算法Online-MGF，针对基站在未知信道统计和有限资源下的多用户下行网络，通过最大化期望总吞吐量来优化调度决策。

**💡 创新点**

创新点在于引入通道信息新鲜度（Age of Channel State Information, AoCSI）与最近观测的CSI构成充分统计量，显著减小了原POMDP的状态空间，并在不满足指数化条件的情况下，提出基于上置信界的最大收益优先策略，获得了子线性渐进式无关的退化度。

**🔧 技术方法**

主要技术包括：基于AoCSI的状态简化、Restless Multi‑armed Bandit（RMAB）建模、拉格朗日松弛与分解、UCB优化估计转移概率、最大收益（gain）索引策略以及在线更新拉格朗日乘子。

**📊 数据集**

实验使用模拟信道模型：两状态ON/OFF马尔可夫链以及三状态SNR马尔可夫链，设定不同用户组、通道转移概率和系统规模。

**📈 对比分析**

通过与随机调度、最大AoCSI优先（MAF）以及已知统计下的Whittle/最大收益（MGF）策略比较，结果显示Online‑MGF在多数设置下接近或超过已知统计策略，并显著优于MAF与随机方法；随着学习周期增加收敛速度提升。

**⚠️ 局限性**

局限性包括：仅针对离散有限状态马尔可夫信道；未提供多时延或持续信道模型的分析；在极大用户数或极高系统时延下，UCB置信区间与算法复杂度仍有提升空间；并且指数化分析仅在两状态情形成立，三状态情形需进一步研究。

---

## 219. PEA-DPO: Perception-Enhanced Alignment Direct Preference Optimization for MLLMs Alignment

**arXiv ID:** 2608.19598 | [PDF](https://arxiv.org/pdf/2608.19598v1)

**作者:** Jiawei Feng `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Perception-Enhanced Alignment Direct Preference Optimization（PEA-DPO）的多模态大语言模型对齐方法

**💡 创新点**

在传统DPO基础上加入视觉偏好信号，构建视觉上下文偏好数据，并通过双重优化（文本质量与视觉敏感度）同时提升模型对关键视觉信息的辨识能力

**🔧 技术方法**

基于CLIP对图像进行随机遮挡并挑选最低语义相似度的版本构造负样本；采用改进的DPO损失（长度归一化、ReLU阈值）实现联合优化

**📊 数据集**

使用LLaVA-1.5的22K偏好实例（含13K图像），以及MMHalBench、Object HalBench、AMBER等多模态幻觉评测数据集

**📈 对比分析**

与多种RLHF/RLAIF、商业模型（GPT-4V、Gemini-2.5-Pro）及现有DPO变体对比，实验表明在MMHalBench、Object HalBench、AMBER等指标上，PEA-DPO在幻觉率、CHAIR分数及人类认知匹配度等方面均优于对比方法，且在不同参数规模的LLaVA模型上均保持领先

**⚠️ 局限性**

构造视觉偏好数据需要额外的遮挡与CLIP相似度计算，增加计算开销；未在最新的大模型（如Muffin）上验证；训练后略微降低了覆盖率（生成更保守）

---

## 220. Are LLMs becoming similarly creative? Evidence from three years of models

**arXiv ID:** 2608.19437 | [PDF](https://arxiv.org/pdf/2608.19437v1)

**作者:** Nirav Patel `[一作]` (Duke University), Emily Wenger `[通讯]` (Duke University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2023-2026年间多代大型语言模型在开放式创意任务（Alternate Uses Task与Infinity-Chat100）上的输出多样性进行时间序列分析，比较不同模型家族之间的相似度变化。

**💡 创新点**

首次系统性追踪LLM创意输出的多样性演化，并使用跨模型嵌入距离回归量化创意趋同趋势。

**🔧 技术方法**

使用句子级Transformer嵌入、余弦距离计算、OLS回归及Bootstrap重采样来评估输出相似度随时间的变化。

**📊 数据集**

利用Alternate Uses Task（10个物品）与Infinity-Chat100（100条真实用户开放式提示），共68个模型（33封闭、34开源）在OpenRouter API上生成回复。

**📈 对比分析**

通过计算跨家族模型对的平均余弦距离，并对时间分箱进行线性回归，得到AUT斜率-0.01385（95% CI[-0.01695,-0.01044]）和Infinity-Chat斜率-0.00167（95% CI[-0.00267,-0.00074]），所有1000次重采样均为负值，表明创意输出在各时间段内趋于一致。

**⚠️ 局限性**

局限包括：仅采样一次回复、未考虑模型共享训练数据或蒸馏导致的相关性、部分提示重复、未建模用户交互影响，以及仅基于嵌入距离而未考虑文本结构或风格差异。

---

## 221. Improved Confidence Estimates for Black-Box Large Language Models

**arXiv ID:** 2608.19323 | [PDF](https://arxiv.org/pdf/2608.19323v1)

**作者:** Sokhna Diarra Mbacke `[一作]` (Layer 6 AI), Gabriel Loaiza-Ganem `[通讯]` (Layer 6 AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用已存在的不确定性评分和参考集中的邻域统计信息，构建特征后训练简单分类器（如逻辑回归或随机森林）来预测LLM回答的正确性，从而得到更精准、校准良好的置信度估计

**💡 创新点**

将不确定性评分作为特征而非最终置信度，并加入邻域统计信息，采用监督学习（Auto‑ML）对任务进行自适配，显著提升校准和判别性能

**🔧 技术方法**

特征工程（不确定性评分+邻域统计）、逻辑回归/随机森林、温度缩放校准、Auto‑ML模型选择、kNN检索+余弦相似度

**📊 数据集**

CommonSense QA、Natural Questions、SciQ、SimpleQA、LLaMA4Maverick等公开问答与事实性数据集

**📈 对比分析**

与原始不确定性评分（语义熵、Laplacian、核熵）以及APRICOT、verbalized confidence等做对比，实验表明Auto‑ML方法在AUROC上几乎始终领先，ECE也明显下降，性能提升可观

**⚠️ 局限性**

需要为每个任务训练专属分类器，依赖足够规模的参考集；方法仅针对二分类正确/错误，扩展至多类需要进一步研究

---

## 222. Time Series Forecasting based on Solana Digital Asset Dataset

**arXiv ID:** 2608.19521 | [PDF](https://arxiv.org/pdf/2608.19521v1)

**作者:** Yufeng Xiao `[一作]` (HSE University), Dmitry I. Ignatov `[通讯]` (HSE University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了首个面向Solana生态的多维时间序列数据集，涵盖1584枚代币的每日交易、价格、流动性、交易者行为及全网DEX指标，并基于该数据集进行市场结构与事件驱动分析；

**💡 创新点**

创新点在于：①首次系统化聚合Solana代币层面与全网DEX层面的多变量信息；②用该数据集验证零样本预测框架，并揭示代币与生态级变量的同步性与因果关联；

**🔧 技术方法**

使用的技术包括PatchTST、Temporal Fusion Transformer、TiDE、Chronos（fine‑tuned/zero‑shot）、TimeGPT、传统统计模型Prophet与SeasonalNaive；

**📊 数据集**

数据集为Solana代币交易与DEX指标的日度记录，包含27个特征，时间跨度2024‑03‑24至2025‑03‑16；

**📈 对比分析**

方法对比采用MAE、RMSE、MAPE三指标，并通过Friedman、Wilcoxon及混合效应模型评估统计显著性；PatchTST获得最低平均排名（≈2.97），Fine‑tuned Chronos紧随其后；Prophet与SeasonalNaive在多数代币上表现优越，但深度学习模型在非周期性预测上优势明显；

**⚠️ 局限性**

局限性包括：预测窗口仅为3日；仅覆盖Solana链，缺乏跨链验证；零样本方法在极短期预测下表现平平；部分代币历史不足导致模型不稳定；未来需扩大时间范围、加入更长预测期与多链对比。

---

## 223. Modeling AI Overreliance as a Complex Adaptive System

**arXiv ID:** 2608.19616 | [PDF](https://arxiv.org/pdf/2608.19616v1)

**作者:** Ahana Biswas `[一作]` `[通讯]` (University of Pittsburgh), Ahana Biswas (University of Pittsburgh)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了一个基于代理的模型，用以研究在多人使用 AI 辅助决策时，信任、验证与社会学习如何共同决定人们对 AI 的过度依赖或不足依赖，并通过理论与仿真探讨反馈机制与网络结构对整体依赖态势的影响。

**💡 创新点**

创新点在于：①将 AI 依赖视为复杂适应系统，将个体学习、同行观察与反馈三者统一进 ABM 中；②提出均值保持定理和均值场临界分析，揭示社会证明会触发验证崩溃的反馈级联；③设计干预手段（验证可见度、社会证明抑制等）并证明其能逆转过度依赖，首次系统性阐明干预的机制与有效性。

**🔧 技术方法**

采用的技术包括：Agent‑Based Modeling (ABM)、贝叶斯 Dirichlet 可信度更新、Logit 选择、随机/BA 网络模拟、均值场分析与折扣/凸性证明、Morris 敏感性分析及全局参数扫描。

**📊 数据集**

数据集：纯合成数据。任务难度从 Beta 分布采样，AI 质量取低/高两组参数设定，网络结构为 ER 或 BA 图。未使用真实实验或日志数据。

**📈 对比分析**

通过对比主要指标（overreliance、regret、RAIR/RSR）在不同环境、网络、干预设置下的仿真结果进行评估。结果显示：环境决定基线；社会学习聚焦信任但不提升整体过度依赖；社会证明导致验证崩溃；验证可见度与社会证明抑制等干预能显著降低 regret。性能评估为定性：指标随参数平滑变化，无明显的硬阈值跳跃。

**⚠️ 局限性**

局限性：①AI 质量假设为外生且不变；②网络结构固定；③验证成本与机制过于简化；④理论临界点在有限系统中往往被平滑化，实际实验需更精细的个体差异；⑤未进行真实数据验证，缺乏与实际工作场景的对照。

---

## 224. IRIS: Navigating and Reflecting on Writing Traces Using Intelligent Document Histories

**arXiv ID:** 2608.19614 | [PDF](https://arxiv.org/pdf/2608.19614v1)

**作者:** David Zhou `[一作]` (University of Illinois Urbana-Champaign), Sarah Sterman `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并实现了名为 IRIS 的交互式写作历史工具，利用键盘记录推断写作过程，并以可视化的版本历史、过程过滤和自然语言查询支持作者对自身写作轨迹进行探索和反思。

**💡 创新点**

其创新点在于将认知写作模型（Flower‑Hayes）与实时键盘轨迹结合，构建过程驱动的版本历史，首次在写作工具中提供基于过程的高层导航、可嵌入的版本摘要和对话式查询；同时通过检索增强的大模型为查询提供证据支持。

**🔧 技术方法**

主要技术包括键盘事件日志采集、基于 Zhang 等人的停顿/编辑启发式解析写作状态、React+Flask 前后端架构、OpenAI GPT‑4（或类似 LLM）进行检索增强生成，以及可视化组件（高亮、工具提示、过滤器、过程图）。

**📊 数据集**

实验使用了两组自愿创作作者（共 25 名参与者）产生的键盘日志和文本版本记录，覆盖短篇创作（≈30 分钟）与多轮写作（≈3 ½ 小时）两种情境；没有使用公开标准数据集，而是收集了原始写作过程数据。

**📈 对比分析**

评估以定性用户研究为主：在形态学研究中通过主题分析、使用频率、访谈反馈展示系统被用于查找特定改动、确认或挑战写作假设，未进行传统性能指标或基准对比；但结果显示使用者更能快速定位改动、获取过程洞察并提升反思深度。

**⚠️ 局限性**

主要限制包括样本规模有限、写作时长不足以覆盖完整写作生命周期、对 LLM 的依赖导致隐私与版权风险、过程推断与作者主观感受存在差距、界面呈现可能产生“被监视”效应，以及当前实现对多作者协作、长期项目和非文本创作场景的支持尚未成熟。

---

## 225. From Street View Imagery to Street Quality Indicators: Vision Language Inference for the Suburban 15-minute City

**arXiv ID:** 2608.20026 | [PDF](https://arxiv.org/pdf/2608.20026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 226. NepOOC-M: Bilingual Nepali-English Benchmark and Comparative Analysis of Multimodal Architectures for OOC Detection

**arXiv ID:** 2608.19212 | [PDF](https://arxiv.org/pdf/2608.19212v1)

**作者:** Sanjeev Khatiwada `[一作]` `[通讯]` (Independent Researcher), Sanjeev Khatiwada (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并公开发布了第一个尼泊尔多语言OOC（离上下文）误导信息基准（1090个图像–说明对），并对五种多模态架构（ResNet-50+mBERT、ViT+TCN、ViT+MuRIL、CLIP、CNN+LSTM）以及文本/图像单模态基线进行了系统评估。

**💡 创新点**

创新点包括：① 为低资源尼泊尔语言创建了带五类误导类型标签的多语言OOC基准；② 发现仅使用说明文本（mBERT）即可达到与最佳多模态模型相当的性能，图像信息对该规模任务贡献极小；③ 通过训练规模扩展实验表明，数据量增长是提升性能的主要驱动力，而模型复杂度或脚本特化的改进作用有限。

**🔧 技术方法**

使用的技术：多模态模型（ResNet-50+mBERT、ViT+TCN、ViT+MuRIL、CLIP、CNN+LSTM），文本编码器mBERT和MuRIL+LoRA，图像编码器ResNet-50、ViT、CNN；进行文本仅、图像仅与多模态融合实验；使用McNemar检验、训练规模缩放实验、宏F1、准确率、AUC等评估指标。

**📊 数据集**

使用的数据集：自建的“nep-ooc-misinformation”基准，1090个图像–说明对，包含尼泊尔语、英语和混合文本，按5种误导类型（Fabricated、Miscaptioned、Temporal mismatch、Geographic mismatch、Identity mismatch）标注；按train/validation/test比例754/108/228划分。

**📈 对比分析**

比较方法：在同一随机种子下对五个多模态模型与文本/图像单模态基线进行准确率、宏F1、AUC评估；利用McNemar检验检验性能差异；进行训练规模（25%–100%）扩展实验。结果显示：ResNet-50+mBERT与文本仅mBERT宏F1均为94.65±0.20%，统计等价；图像仅模型表现接近随机；其他多模态模型性能略低或波动；数据规模提升带来的性能增益比模型升级更显著。

**⚠️ 局限性**

局限性：① 数据量相对较小，导致对细粒度差异的统计显著性不足；② 样本主要来自事实核查机构，可能引入专业化语言偏差；③ 采用闭对设置，未使用检索或外部知识，导致少数类型（如地理/身份不匹配）识别受限；④ 未覆盖社交媒体自然生成内容，限制了对真实环境的泛化评估。

---

## 227. CLaST: Context-aware Contrastive VAE for Probabilistic Time Series Forecasting

**arXiv ID:** 2608.20025 | [PDF](https://arxiv.org/pdf/2608.20025v1)

**作者:** Alexander Marusov `[一作]` (Applied AI Institute), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了 CLaST——一种结合上下文对比学习的变分自编码器，用于概率性时间序列预测，并通过新的相似度对齐损失实现潜在空间结构化。

**💡 创新点**

创新点在于：① 定义 LINTS 过程并从中构造地面真相相似度矩阵；② 用对比损失代替传统 MI 估计，理论证明其在该结构下可实现最优；③ 在 VAE 中引入趋势/季节分解并保持其独立性。

**🔧 技术方法**

使用的技术包括：VAE、对比学习、相似度矩阵构造、拉普拉斯估计、DFT 预测器、PatchTST 编码器、以及多任务训练。

**📊 数据集**

实验数据集包括 9 个公开数据集：Electricity、Traffic、Weather、ETTh1/ETTh2/ETTm1/ETTm2、Solar、ERCOT。

**📈 对比分析**

与 LaST、K^2VAE、DeepAR、TFM、TSDiff、PatchTST、DeNOTS 等基线进行比较。短期预测 CRPS 最高提升 16.4%，NMAE 最高提升 14.4%；长期预测 CRPS 提升 48.6%，NMAE 提升 25.1%，在所有基准上均稳健优于第二名。

**⚠️ 局限性**

局限性包括：① 假设协方差仅随时间滞后变化，可能不适用于绝对时间相关性强的数据；② 相似度矩阵构造需要 O(N²) 计算，导致训练开销略大；③ 对高维、稀疏数据的泛化仍待进一步验证。

---

## 228. TGL-APT: Temporal Graph Learning with Graph Distillation for Efficient APT Investigation

**arXiv ID:** 2608.19750 | [PDF](https://arxiv.org/pdf/2608.19750v1)

**作者:** Jing Chen `[一作]` (Fujian Normal University), Yuexin Zhang `[通讯]` (Fujian Normal University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了TGL-APT框架，用于基于系统溯源图的APT检测与完整攻击链调查；

**💡 创新点**

创新点包括：信息瓶颈（IB）节点驱动的图蒸馏、基于模型注意力与嵌入偏差的自适应核心节点更新、跨时空指纹对齐实现分散可疑行为聚合，以及因果扩展重建攻击链；

**🔧 技术方法**

采用的技术包括信息瓶颈理论、图注意力网络（TGN+多头注意）、对比学习、TF‑IDF加权指纹、时间窗口异常阈值、基于结构熵的图压缩约束等；

**📊 数据集**

使用 DARPA Engagement 3 的 Cadets、ClearScope 与 Theia 三个公开数据集；

**📈 对比分析**

与 KAIROS、TFLAG、ORTHRUS 进行对比实验，TGL-APT 在三数据集上 F1 分数均在 0.89 以上；相较于 KAIROS，训练时间缩短约 39%、检测延迟缩短约 33%、内存占用降低约 22%；

**⚠️ 局限性**

局限性：攻击链重建仅在阶段级别评估，缺乏对完整攻击路径的精细化验证；自适应核心节点更新采用固定调度，缺少在线动态适配；未来工作需探索更细粒度的链路评估、弱监督的 IB 精炼与在线更新机制。

---

## 229. When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints

**arXiv ID:** 2608.19488 | [PDF](https://arxiv.org/pdf/2608.19488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 230. Pod-Deployability in Kubernetes with Inter-Pod Affinity Constraints is PSPACE-Complete

**arXiv ID:** 2608.19822 | [PDF](https://arxiv.org/pdf/2608.19822v1)

**作者:** Saverio Giallorenzo `[一作]` (Universit\`a di Bologna), Gianluigi Zavattaro `[通讯]` (Universit\`a di Bologna)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究Kubernetes调度器中“pod可部署性”问题，即给定初始集群、Pod类型和目标节点，判定是否存在合法的Pod部署与删除序列使目标Pod最终落在该节点；通过形式化调度语义，证明该问题在不同约束下的复杂度边界；

**💡 创新点**

创新点在于首次将Kubernetes的硬性调度约束（资源、标签、亲和/反亲和、拓扑散布）形式化为状态机，并将其转化为覆盖性问题；在此基础上给出三大复杂度结果，揭示亲和与反亲和的组合与仅亲和的组合如何导致P/PSPACE/EXPSPACE级别的难度；

**🔧 技术方法**

主要技术包括：1）Kubernetes调度语义的数学建模；2）覆盖性问题与Petri网、黑色弹珠游戏等经典模型的多种多项式约简；3）利用Lean证明器实现机器校验；4）对状态空间的支持抽象与Savitch定理应用以获得空间复杂度上界；

**📊 数据集**

本工作为理论分析，不使用实验数据集；所有证明均在符号化模型与抽象构造上完成；

**📈 对比分析**

没有传统意义上的实验对比；论文通过与已知的Petri网覆盖性、黑色弹珠游戏等标准复杂度基准对比，证明其结果达到对应的NP/PSPACE/EXPSPACE下界；

**⚠️ 局限性**

局限性包括：1）仅考虑单个节点或固定数量节点的情况，未考虑动态节点扩缩；2）未考虑调度器的软约束、预判决策略及插件的内部状态；3）结果仅适用于容量受限（或无容量）情形，对无界容量的完整可达性分析仍未给出；4）实验验证缺失，实际部署中调度器实现细节可能导致差异。

---

## 231. The Asymmetric Harms of LLM Compression

**arXiv ID:** 2608.19670 | [PDF](https://arxiv.org/pdf/2608.19670v1)

**作者:** Yuan Wu `[一作]` (Rice University), Chudi Zhong `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估三种LLM在11种压缩方法（量化与剪枝）下的知识保留、置信度与社会偏差变化，针对知识流行度、误答置信度和子组偏差提出了三项研究问题。

**💡 创新点**

提出统一的评估协议，发现压缩导致头部知识相对保留率下降、误答保持高置信度，并在子组层面掩盖聚合偏差的对立变化，揭示了压缩的非对称行为。

**🔧 技术方法**

使用后训练量化（GPTQ、AWQ、OmniQuant、AQLM）和剪枝（WANDA、SparseGPT、ShortGPT、结构化层删）技术，结合 perplexity、准确率、置信度、ECE 等评估指标。

**📊 数据集**

采用按知识流行度划分的 PopQA 与 Head-to-Tail 数据集评估知识保留，使用 WinoBias 与 BBQ 数据集评估子组偏差。

**📈 对比分析**

在非崩塌压缩设置下，对整体与子组准确率、相对保持率、置信度与 ECE 进行对比；结果表明整体指标保持良好，但头部知识比例下降、误答置信度偏高、子组偏差出现显著对立变化，揭示了聚合指标的盲区。

**⚠️ 局限性**

仅评估了 8–9B 级指令调优模型和特定的量化/剪枝方法，未覆盖更大模型、不同架构或蒸馏方法；偏差分析未进行概率校准，结果可能不具备普适性。

---

## 232. Eigensolvers for polynomial roots and tensor decomposition

**arXiv ID:** 2608.19818 | [PDF](https://arxiv.org/pdf/2608.19818v1)

**作者:** Enrica Barrilli `[一作]` (Centre Inria at Université Côte d'Azur), Bernard Mourrain `[通讯]` (Centre Inria at Université Côte d'Azur)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文介绍了一套基于 Truncated Normal Forms（TNF）的符号-数值算法，用来从乘法算子求解多项式方程组的根及其重数结构，并将同一框架推广到张量分解（GAD/Waring）问题。

**💡 创新点**

创新点在于：① 把 Gröbner 基、Border 基和 Resultant 矩阵的构造统一映射到 TNF 计算；② 利用 TNF 直接得到乘法算子，进而通过共同特征值/ Schur 分解获得根及其重数；③ 在张量分解中采用 Hankel（Catalecticant）矩阵构造 TNF，完成 GAD 的重构；④ 提供 Julia 包实现，兼顾符号和数值稳定性。

**🔧 技术方法**

使用的技术包括：符号 Gröbner 基与 Border 基的数值化、TNF 的构造与核空间计算、乘法算子构造、联合 Schur 分解、逆系统与伪矩阵（pseudo‑moments）分析、Apollard duality、Hankel/Catalecticant 矩阵。

**📊 数据集**

论文未使用公开数据集；通过手写例子（两元二次方程组、三维齐次多项式）演示算法效果。

**📈 对比分析**

在示例中，使用 Gröbner 基和 Resultant 方法得到相同 TNF，随后在相同基础上比较传统符号方法与数值 TNF 的效率；在张量分解例子中，恢复误差约 10⁻¹³，显示数值稳定性和精度；但文中未给出大规模实验对比。

**⚠️ 局限性**

局限性：① 对于极高维或高次方程，TNF 计算仍受限于矩阵规模；② 需要事先计算 Gröbner 基或 Resultant 矩阵，对稀疏度敏感；③ 对非零维或多重解集的完整性分析仍需进一步研究；④ 目前实现主要在 Julia，缺乏多平台的广泛验证。

---

## 233. When Saying No Makes Better Videos: Designing Dual Gatekeeping for Pedagogically Grounded AI Content Creation

**arXiv ID:** 2608.19812 | [PDF](https://arxiv.org/pdf/2608.19812v1)

**作者:** Yearim Kim `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个名为PedaCo的双层门控视频创作系统，允许教师在AI生成脚本前后对内容进行理论驱动的拒绝、修订和自动化评估。

**💡 创新点**

创新点在于将认知理论多媒体学习（CTML）原则嵌入人机协作流程，形成可操作的“原则性抵抗”，并结合人工评审与自动化指标两层检验。

**🔧 技术方法**

采用LLM生成脚本、AI评审器基于CTML提供反馈、后期使用五维度（连贯性、冗余、时间连续性、模态、图像质量）的自动化计算指标；同时教师对脚本进行手工编辑或重生成。

**📊 数据集**

数据集来源于已确立的科学与哲学课程，共7个主题、14个视频（每主题双条件），以及23名教师参与的实验数据。

**📈 对比分析**

与无CTML指导基线视频相比，教师主观评价在12个CTML原则上均显著提升（平均分从3.07升至3.86，p<0.05），自动化指标显示时间连续性和连贯性均显著改善。

**⚠️ 局限性**

局限性包括：缺乏对学生学习成效的验证；自动化指标对某些维度（如个性化）覆盖不足；长期工作负荷与教师疲劳仍未系统评估。

---

## 234. Beyond Recognition: Compact Multi-Domain Arabic Manuscript HTR with Candidate-Selection Analysis and Evidence-Preserving Review

**arXiv ID:** 2608.19385 | [PDF](https://arxiv.org/pdf/2608.19385v1)

**作者:** Abdullah Ahmed Ali `[一作]` (University of Technology), Dhulfiqar Mahdi Wadi `[通讯]` (University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个压缩的CNN–BiLSTM–CTC手稿识别器，并构建了基于证据的审阅工作流Athar，能够在多域阿拉伯手稿上保持高精度并记录视觉读取、候选与检索结果。

**💡 创新点**

创新点包括：①采用多域经验回放与遗忘防护实现单一模型跨域适应；②将识别器与可审阅工作流结合，保留原始视觉读取并提供唯一/模糊/放弃三种检索状态；③系统化候选头部分析揭示可达2.15%字符错误的可恢复空间，说明重排方法的局限。

**🔧 技术方法**

使用技术：CNN + 4 层 BiLSTM + CTC 解码；多域回放与阈值保留；浅融合字符 n‑gram 语言模型（α=0.5, β=0.3, beam=10）；检索增强（字符 n‑gram 投票、对齐与相似度阈值）；多种候选重排（MBR、QE、神经重排器）及基准评估。

**📊 数据集**

使用公开阿拉伯手稿数据集：Muharaf、RASAM、TariMa、Agapet、Omar Al‑Saleh，并按文档级拆分用于训练、适配与封闭评估。

**📈 对比分析**

评估方法：在22,278行封闭评估上，使用贪婪解码、原始参考、无语言模型进行统一测评。结果显示字符加权CER从19.98%降至14.93%（相对下降25.3%），Agapet 22.12%→17.86%，Omar 17.72%→11.84%，TariMa轻度回升0.33%。候选头部分析表明Oracle@25可达5.82%而Beam仅为7.97%，差距2.15%，而其他重排方法只能恢复不到4%。

**⚠️ 局限性**

限制：1）极端布局或复杂页框导致误差显著提升；2）未公开每行预测文件，缺乏置信度校准；3）重排技术无法完全消除错误；4）检索库有限，唯一性判定受限；5）未进行正式的可用性或时间成本研究。

---

## 235. Sub-optimality of Marton's Inner Bound for the Two-Receiver Broadcast Channel

**arXiv ID:** 2608.19869 | [PDF](https://arxiv.org/pdf/2608.19869v1)

**作者:** Mian Huang `[一作]` (Multimoon Lab), Yi Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造具体的三元输入广播信道和相应的分布，证明了马尔顿内界在某些离散无记忆广播信道上严格不等于容量区域；

**💡 创新点**

创新点在于提出梯度塑形与约束消除两种通用技术，能够把固定输入分布下的多字母优势转化为无约束情形，并首次给出严格的反例；

**🔧 技术方法**

主要技术包括数值优化、梯度塑形（对单字母目标函数的凸性调节）、约束消除（通过增大输入字母并引入“共同确定输出”组件）以及信息量的精确有理化计算；

**📊 数据集**

使用的“数据集”是两组具体的信道转移矩阵（如文中列出的三元输出矩阵）以及对应的优化分布，全部以有理数精确给出；

**📈 对比分析**

比较方法是对单字母马尔顿上界和两字母上界分别求最大值，并通过证明两字母上界严格大于单字母上界的两倍，表明单字母马尔顿内界是严格子最优；

**⚠️ 局限性**

局限性在于该结论仅在三元或更大输入字母的信道上得到；对二元输入信道的情况仍未解决，并且构造与数值验证需要极大计算资源。

---

## 236. LLMs as Acquisition Policies for Finite-Pool Materials Optimization: A Controlled Study

**arXiv ID:** 2608.19790 | [PDF](https://arxiv.org/pdf/2608.19790v1)

**作者:** Dino-Rober Demir `[一作]` (RIKEN Center for Computational Science), Rio Yokota `[通讯]` (RIKEN Center for Computational Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估开放权重大型语言模型（LLM）在有限候选池材料优化中的主动学习采集策略

**💡 创新点**

验证LLM可直接作为采集政策，无需任务特定训练，并探究其对候选呈现方式和材料语境的敏感性

**🔧 技术方法**

采用多种LLM（如LLaMA、Falcon、GPT‑NeoX、OPT‑175B 等）与传统高斯过程 EI、随机选择进行对比

**📊 数据集**

四个材料优化任务：Fe‑Co‑Ni 三元薄膜 Kerr 旋转与磁滞强度、Ti‑Al‑Ni 强度、BaTiO₃ 复合材料电压应变

**📈 对比分析**

在同一初始点、相同候选池下比较迭代次数到全局最优；LLM在多数任务中明显优于随机选择，整体性能介于 GP‑EI 与随机之间，且表现因模型、批量大小和候选排序差异而显著波动

**⚠️ 局限性**

LLM 的可靠性高度依赖于候选呈现、批量大小、初始化和候选顺序；缺乏一致的跨任务优势，且在处理大上下文时易出现位置偏差，尚未达到 GP‑EI 的稳定竞争水平

---

## 237. Data-Driven Time-Varying Control Barrier Functions for Adaptive Safe-Set Learning with Online Decremental Support Vector Machines

**arXiv ID:** 2608.19366 | [PDF](https://arxiv.org/pdf/2608.19366v1)

**作者:** Shawon Dey `[一作]` (New Mexico State University), Hever Moncayo `[通讯]` (Embry Riddle Aeronautical University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种退化感知、时变的 SVM‑CBF‑QP 安全滤波框架，通过 RBF‑SVM 学习安全集，利用连续时间递减 SVM 更新法随系统退化实时收缩安全边界，并用二次规划安全滤波器在退化输入约束下保证系统安全。

**💡 创新点**

创新点包括：① 基于退化调度信号的连续时间递减 SVM 更新，保持安全边界可微；② 引入同胚平滑法处理活跃集切换，使 CBF 连续；③ 将时变 SVM‑CBF 与 QP 安全滤波器结合，并给出前向不变性与递归可行性证明。

**🔧 技术方法**

采用的技术包括 RBF‑SVM、连续时间递减 SVM 更新、控制障碍函数（CBF）、同胚平滑、二次规划安全滤波器、扩展 class‑κ 函数以及 Nagumo 定理等。

**📊 数据集**

实验使用基于 VTOL 短周期纵向线性化动力学的合成数据集，训练集由 225 个在角度与俯仰速率平面上标记安全/不安全的随机样本组成，并采用人工设定的退化曲线 λ(t) 进行仿真。

**📈 对比分析**

与无滤波、固定障碍（只使用初始 SVM 边界）三种配置对比，仿真表明所提方法在退化过程中始终保持 h_H(x,t)≥0、QP 可行且干预最小；在同胚平滑与瞬时切换的对比中，同胚平滑能保持障碍连续、减小输入突变、并通过调节过渡时间避免 QP 不可行。

**⚠️ 局限性**

局限性包括：仅在低维 Toy 模型上验证，退化调度信号需预先给定；递减支持向量的手动选择可能不易推广；高维系统的可扩展性与实时实现尚未验证；安全集学习依赖于代表性训练数据，若数据不足可能导致保守或误判。

---

## 238. Spike-based Belief Propagation in Nonlinear Dynamical Systems

**arXiv ID:** 2608.19907 | [PDF](https://arxiv.org/pdf/2608.19907v1)

**作者:** Sepideh Adamiat `[一作]` (Eindhoven University Of Technology), Bert de Vries `[通讯]` (Eindhoven University Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种将尖峰神经网络与因子图信念传播相结合的贝叶斯控制框架，用于在非线性动力学环境（Mountain Car）中实现实时自适应控制。

**💡 创新点**

创新点在于：①将多变量非线性状态转移因子通过尖峰神经网络实现，并嵌入事件驱动的 BP 中；②实现了生物学灵感的尖峰编码/解码与神经工程框架（NEF）结合的全流程；③利用 Unscented Kalman Filter 在尖峰网络中近似非线性消息更新。

**🔧 技术方法**

使用的技术包括：因子图信念传播、神经工程框架（NEF）+ LIF 细胞、尖峰编码与解码、Unscented Kalman Filter 消息近似、事件驱动的尖峰网络实现与自动化权重求解。

**📊 数据集**

使用的数据集为经典 Mountain Car 仿真环境，未使用公开数据集。

**📈 对比分析**

通过与传统（非尖峰）活跃推理控制器对比，结果显示两者在车辆位置轨迹和发动机力输出上表现相近，证明尖峰实现不影响控制性能。

**⚠️ 局限性**

局限性包括：①仅部分实现为尖峰（部分因子节点仍采用非尖峰计算）；②使用手工建模的状态转移方程，缺乏可学习动力学模型；③可扩展性尚未验证，难以直接迁移至更复杂或高维控制任务；④能耗评估与神经硬件实现尚未完成。

---

## 239. Represented but Ignored: A Causal Account of Prosodic Underuse in Audio-Language Models

**arXiv ID:** 2608.19211 | [PDF](https://arxiv.org/pdf/2608.19211v1)

**作者:** Linkai Peng `[一作]` (University of Connecticut), Baorian Nuchged `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对大规模音频-语言模型（audio‑LLM）中对语调（prosody）的表达能力进行诊断，提出了阶段性失效分类和探针阶梯，用于定位模型在感知、解释或利用语调信息时的瓶颈，并通过单层干预与稀疏特征操作验证了内部语调信号的因果可控性。

**💡 创新点**

创新点在于将语调失效划分为三类（F1感知、F2解释、F3利用），设计多阶段探针阶梯与内部logit‑lens、方向注入、激活补丁以及稀疏自编码器（SAE）特征干预，首次揭示在四款主流理解型audio‑LLM中，语调失败主要归因于“利用不足”（F3）而非“未感知”。

**🔧 技术方法**

使用的技术包括：层级线性探针、logit‑lens读取、V‑信息评估、方向注入（direction injection）、激活补丁（activation patching）、稀疏特征选择与稀疏自编码器（TopK SAE）、AtP* 归因方法以及基于音频的情感/语调数据集的匹配内容对照实验。

**📊 数据集**

采用的主要数据集为匹配内容对照集：IViE（问/陈述语调）、CREMA‑D（情感）和VESUS（情感），以及用于音频路径校准的JL‑Corpus、ESD‑English 等；所有数据均为无训练泄露的 held‑out 语料。

**📈 对比分析**

实验通过三条件阶梯（文本无线索基准、文本+语调参考）评估音频对回答的贡献，发现 7/11（约 64%）模型-对照组合表现为 F3 失效，行为仅恢复了 30–60% 的可用范围；单层干预可将大部分模型的决策向正确语调方向偏移，稀疏 SAE 干预亦能在 17/18 细胞中实现相似恢复，表明内部语调信号具有可操控性。

**⚠️ 局限性**

局限性包括：仅评估四款理解型、密集变压器的 audio‑LLM，无法推广到 MoE 或语音到语音模型；单残差流假设对多头/多模态模型不适用；部分对照因训练数据泄露被屏蔽；文本+语调参考仅为实用上限而非绝对上限；方向注入的幅度调节不具备自然尺度；logit‑lens 采用基础形式，未做调优；稀疏特征与声学相关性仅在情感任务中表现强劲，且未做多重比较校正。

---

## 240. ShadowPath: Lookup-Private Credential Status Verification over Authenticated State

**arXiv ID:** 2608.19937 | [PDF](https://arxiv.org/pdf/2608.19937v1)

**作者:** Patrick Herbke `[一作]` (Technische Universität Berlin), Axel Küpper `[通讯]` (Technische Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本研究中，作者提出并实现了一种名为 ShadowPath 的协议，旨在将凭证状态（是否被吊销）的查询移至持有者端，并通过零知识证明隐藏查询过程，避免在验证过程中暴露凭证特定信息；

**💡 创新点**

其创新点在于：①将凭证状态查询从验证方迁移至持有者，完全不需要在验证时向发行方发起凭证特定的查询；②在零知识证明中隐藏树路径和索引，确保查询隐私；③系统兼容两种后端—稀疏 Merkle 树与 Verkle 树，并对二者在同一地址空间下的证明与验证成本进行实证比较；

**🔧 技术方法**

主要技术包括：零知识证明（Groth16 与 PLONK）、KZG 多项式承诺、Verkle 与稀疏 Merkle 树的树结构、BLS12-377 曲线与 BW6‑761 pairing、IPFS/IPNS 用于分布式状态发布、以及 Poseidon 哈希等加密工具；

**📊 数据集**

实验使用了合成凭证集，最大支持 10^6 条凭证，地址空间为 2^50 位；状态更新、完整摘要与增量 delta 也使用合成数据进行评测；

**📈 对比分析**

在对比方法上，作者在同一硬件（Apple M2 Pro）下对 SMT 与 Verkle 进行了匹配的 Groth16 证明实验，发现 Verkle 的证明时间约为 SMT 的 5.7 倍；对同一 Verkle 后端比较 Groth16 与 PLONK，PLONK 的证明时间约为 Groth16 的 12 倍；在移动设备上，Groth16 的证明时间在 3–4 ms 范围内，符合低延迟需求，而 PLONK 则超过 10 ms，显示在移动端不可行；此外，状态同步实验表明，完整摘要的传输成本随吊销数量线性增长，而增量 delta 则保持低成本；

**⚠️ 局限性**

限制方面：①目前仅对单一凭证与单一 epoch 的状态更新做实验，未评估多 epoch 或大规模并发场景；②移动端证明虽在 Groth16 下可行，但对 Verkle 后端的性能仍高；③系统未实现匿名状态检索，网络层面仍可能泄露同步元数据；④缺乏对多租户发行方、跨域信任与可审计性的完整支持；

---

## 241. Triangular Fuzzy Rescaling Distance

**arXiv ID:** 2608.19234 | [PDF](https://arxiv.org/pdf/2608.19234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 242. Compliance, Capability, and Conflict: Benchmarking Multimodal LLMs under System Messages

**arXiv ID:** 2608.19207 | [PDF](https://arxiv.org/pdf/2608.19207v1)

**作者:** Juan Yeo `[一作]` (Seoul National University), Geewook Kim `[通讯]` (NAVER Cloud AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并发布了VSysBench，一个针对多模态大型语言模型（MLLM）在系统消息约束下既评估视觉任务准确性又评估系统消息遵从性的联合评测框架。

**💡 创新点**

创新点在于：①将系统级约束放置在真正的多模态环境中；②引入对齐与冲突（misaligned）两种用户场景；③定义联合满足率（JSR）和跨约束敏感度（CCS）两项指标，突破先前仅针对文本或用户级约束的局限；④通过四阶段生成、ILP筛选与人工验证构建高质量数据集。

**🔧 技术方法**

采用了四阶段数据生成（系统约束生成、二元过滤、ILP优化、人类验证）和LLM-as-judge评估框架（使用GPT-5-Mini做软分数判定），并提出了CSR、TA、CCS、JSR等度量。

**📊 数据集**

基于MM‑Vet v2的图像‑文本对，构造了2258个人工验证样本，并通过生成对应的对齐/冲突用户提示扩展为4516个实例。

**📈 对比分析**

在16个公开和专有MLLM上进行统一评估，结果表明系统消息约束普遍导致30%–70%的任务准确率下降；视觉约束最难；在冲突场景下，专有模型保持高合规率（如GPT‑5.4 83.3%），而开源模型崩溃（如Qwen3‑VL‑32B仅8.4%）；无模型同时实现高JSR与低CCS，显示该评测揭示了先前基准未能发现的对齐瓶颈。

**⚠️ 局限性**

限制包括：数据仅覆盖MM‑Vet图像，缺少长文本、视频等多模态输入；仅评估单轮交互；评判者为LLM，存在自评偏差；对不同冲突样式的完整评估仅在单一模型上完成，未覆盖所有16个模型。

---

## 243. Proper Sea Surface Roughness Enhances the Performance of Near-Shore Maritime Networks

**arXiv ID:** 2608.19730 | [PDF](https://arxiv.org/pdf/2608.19730v1)

**作者:** Wen-Yu Dong `[一作]` (China Telecom Research Institute), Sheng Chen `[通讯]` (University of Southampton)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于粗糙海面反射系数的非均匀泊位分布模型，构建了近岸海事网络的粗糙度感知随机几何框架；

**💡 创新点**

创新点在于将Rayleigh粗糙度准则与波长相关的有效反射系数融入分段两射/三射路径损耗模型，并结合非齐次泊位Poisson点过程实现了粗糙度对链路可靠性与吞吐量的解析表征；

**🔧 技术方法**

采用随机几何、Nakagami‑m衰落、聚合干扰拉普拉斯变换以及闭式覆盖概率与平均容量推导等理论工具；

**📊 数据集**

使用5.15 GHz与160 MHz的实测信号强度数据（分别对应C波段粗糙状态与VHF平稳海面），并基于这些数据对模型进行验证与敏感性分析；

**📈 对比分析**

通过与理想平滑海面模型（Lee两射/三射模型）进行RMSE、残差分析及蒙特卡洛仿真对比，证明粗糙度模型在低阈值下提升覆盖概率、在高阈值下略逊于理想模型，呈现可靠性-容量权衡；

**⚠️ 局限性**

局限在于仅考虑了显式的相干反射衰减，未显式建模离散散射场，对H_s>3 m的严重粗糙状态下的散射贡献采用粗糙的Nakagami‑m近似，且缺乏针对VHF粗糙海面的直接测量验证。

---

## 244. An end-to-end differentiable transient vapor-compression framework for automated machine sizing and unified optimal control

**arXiv ID:** 2608.19552 | [PDF](https://arxiv.org/pdf/2608.19552v1)

**作者:** Sam Yang `[一作]` `[通讯]` (Florida State University), Sam Yang (Florida State University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于JAX的可微分有限体积汽压缩热泵框架，自动从热负荷推导设备尺寸，并在同一残差方程下完成动态仿真与模型预测控制；

**💡 创新点**

核心创新在于：① 将热力学属性预先在(p,h)网格上闪光并采用双线性插值，使残差可微；② 采用统一的残差实现压缩机、膨胀阀、冷凝/蒸发线的连续动力学和尺寸反演；③ 在同一物理内核上实现TR‑BDF2刚性积分与隐式Euler MPC，消除工厂-控制器不匹配；④ 通过四点循环合成和ε‑NTU匹配实现压缩机位移、阀面积、管路数的直接求解；

**🔧 技术方法**

使用技术包括：JAX自动微分与编译集成、CoolProp Helmholtz方程组、双线性属性插值、有限体积离散、TR‑BDF2刚性积分、隐式Euler模型预测控制、Zivi密度、等熵压缩机多项式模型、开口式膨胀阀、Shah型换热系数、以及速率限制的PID闭环；

**📊 数据集**

实验数据集涵盖：Ramírez等人R410A迷你分体机的多组测量（3.5 kW）、NREL住宅HIL 3 吨单速热泵的1 Hz测点以及Lee等人提供的AHRI 540多项式压缩机映射；

**📈 对比分析**

比较方法：在实验测点上做代数闭合预测（无参数拟合），计算容量、功率、COP的MAPE；在闭环PID实验中检验设定点跟踪误差；在NREL数据上对比时段平均容量误差；结果显示：R410A迷你分体机容量MAPE≈7.4%，NREL热泵时段容量误差≤2%；COP误差相对较大，且在加热模式下误差更显著；

**⚠️ 局限性**

局限性：假设亚临界两相、声学平衡、单一气闸区、干燥空气（除非提供湿度）；不含自启动/退火调度、转热CO₂、油、管道惯性、多个区域或风管网络；模型未经过实验拟合，导致加热模式误差较大；在极端工况下可能产生尺寸过大或过小的风险。

---

## 245. Trace-Based Execution-Level Observability of VDM-SL Specifications

**arXiv ID:** 2608.19510 | [PDF](https://arxiv.org/pdf/2608.19510v1)

**作者:** Tomohiro Oda `[一作]` (Software Research Associates, Inc.), Han-Myung Chang `[通讯]` (Nanzan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为 VDM‑SL 规范记录执行追踪并提供可视化工具。

**💡 创新点**

将状态变量赋值和操作调用持久化为树形追踪，并通过 Mermaid 生成序列图和状态图。

**🔧 技术方法**

基于 Pharo 的 ViennaTalk 通过槽机制插桩、转译为 Pharo 代码，并利用 Mermaid、CSV 导出。

**📊 数据集**

未使用外部数据集，仅在自动门示例上演示。

**📈 对比分析**

没有系统性基准测试，仅通过案例展示可视化效果，性能未量化。

**⚠️ 局限性**

对大规模程序的可扩展性欠缺，缺少查询语言和选择性采样机制。

---

## 246. Generating Diverse Personas for User Simulators to Test Interview Dialogue Systems

**arXiv ID:** 2608.19549 | [PDF](https://arxiv.org/pdf/2608.19549v1)

**作者:** Mikio Nakano `[一作]` (C4A Research Institute, Inc.), Hironori Takeuchi `[通讯]` (Musashi University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大语言模型自动生成面试对话系统的用户角色，并通过指定沟通风格特征来提高模拟对话的多样性

**💡 创新点**

创新点在于将拟人化程度与冗长程度等沟通风格特征嵌入提示，使LLM生成更丰富多样的用户角色，进而提升用户模拟的覆盖范围

**🔧 技术方法**

采用GPT‑4o进行角色生成与对话模拟，结合提示工程和多样性评估指标（词汇多样性、TTR、CW‑TTR等）进行实验

**📊 数据集**

使用10个人工编写的种子角色（分别为旅游和甜品面试系统），通过LLM生成100个新角色，并在日本语面试系统上进行对话实验

**📈 对比分析**

通过与基线（仅使用种子角色）和不同人格特征设置（noPT、APM、EL、APM+EL）五种条件比较，利用多样性指标评估发现noPT提升内容多样性，EL显著提高风格多样性，整体提升了对话多样性

**⚠️ 局限性**

局限性包括仅使用单一LLM模型和日语数据，固定温度和样本量，未验证错误检测效果，且人格特征选择有限，可能不足以覆盖所有用户行为

---

## 247. Accelerated Genetic Programming Hyper-Heuristics for Simulation-Based Scheduling via Agentic AI

**arXiv ID:** 2608.19487 | [PDF](https://arxiv.org/pdf/2608.19487v1)

**作者:** Heyang Thomas Li `[一作]` (REANNZ), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过代理式人工智能与人机协作，对Python遗传编程超启发式项目调度模拟代码进行系统重构，显著降低运行时间。

**💡 创新点**

首次将Agentic AI驱动的循环优化流程与四层次性能工程策略相结合，在保持算法正确性的前提下，将高频解释器开销转化为原生编译与增量状态更新。

**🔧 技术方法**

采用Claude代理、Python优化模式、静态字典缓存、增量资源追踪、Numba JIT编译及传统代码清理与结构重构。

**📊 数据集**

使用DMRCPSP（动态多模式资源受限项目调度）实例集，基准测试基于GitHub开源代码库中的多种项目实例。

**📈 对比分析**

通过对比原始实现与四层优化后实现，运行时间从1,298 s降至195 s（约85%提升），年计算单元节省约4百万，成本节约约32万新西兰元。

**⚠️ 局限性**

仍受Python解释器固有开销限制，未充分探索GPU或分布式并行化；优化过程需要人工审查，难以完全自动化。

---

## 248. FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving

**arXiv ID:** 2608.19758 | [PDF](https://arxiv.org/pdf/2608.19758v1)

**作者:** Qihang Fan `[一作]` (CASIA), Ran He `[通讯]` (CASIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FlashPrefill V2，一种可直接部署在大模型长上下文推理中的稀疏注意力框架，兼顾精度、速度与系统兼容性。

**💡 创新点**

创新点包括：① 引入零阶均值校正（mean‑correction）在软最大中补偿被剪枝的块，显著降低极端稀疏下的准确率损失；② 重新设计与 FlashAttention‑3/4 对齐的稀疏内核，使用 PackGQA 访存、warp 专化的生产/消费管线、ping‑pong 重叠以及 FP8 支持；③ 原生支持分页 KV 缓存和连续批处理，能够无缝接入 SGLang 等现代推理框架。

**🔧 技术方法**

使用的技术包括块级得分估计、基于最大值的动态阈值、均值校正、PackGQA 内存布局、Warp‑specialized 生产/消费管线、TMA 与 cp.async 异步数据搬运、FP8 量化与在线 softmax、CSR 索引、分页 KV 访问以及多分区负载均衡。

**📊 数据集**

评估数据集：RULER（长上下文检索推理）和 LongBench（多任务长文本推理），模型分别为 Llama‑3.1‑8B‑Instruct、Qwen3‑4B‑Instruct‑2507、Qwen3‑30B‑A3B‑Instruct‑2507。

**📈 对比分析**

与 Full Attention、MInference、FlexPrefill、XAttention、FlashPrefill V1、FlashAttention‑2 以及 FA3/4 对齐的密集内核对比，FlashPrefill V2 在 128K 上下文长度下 BF16 取得 27×、FP8 取得 47× 的加速（相对于 FA2），相对于 FA3/4 对齐的密集基线可获得 30× 加速；端到端推理时，时间‑到‑首词（TTFT）在 BF16 下可减至 3–4×，FP8 下可减至 4–5×，且在高并发场景中请求吞吐可提升 2–3×。

**⚠️ 局限性**

局限性：均值校正在极端稀疏（<5%）或高量化误差下仍会有一定精度衰减；阈值 α 的调参影响可用稀疏度和准确率；在解码阶段不适用（只能用于 prefill）；对非块级稀疏模式支持有限；在中等密度时索引构建与同步开销略大。

---

## 249. Stopping and Routing LLM Judge Panels

**arXiv ID:** 2608.19802 | [PDF](https://arxiv.org/pdf/2608.19802v1)

**作者:** Bin Zhu `[一作]` (Sun Yat-sen University), Yanghui Rao `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于有限审计数据的LLM评估管道中判别器（judge）分配与停止策略，利用复制、补充与专门化三种角色模型决定何时调用、路由或放弃各个判别器；并生成可审计的调用计划与部署地图。

**💡 创新点**

核心创新是将判别器多样性从仅描述性指标转化为目标相对、条件化的角色框架；通过验证损失增益与成本阈值实现逐步构建与自动停止；最终提供可执行的、成本-风险前沿的判别器调用政策。

**🔧 技术方法**

使用目标相对信息增益公式、校准器（cell‑mean）与验证损失估计；贪心阈值构建流程；多维度角色分布与诊断比率；结合岭回归/逻辑回归等集成方法进行最终评估。

**📊 数据集**

实验涵盖多种LLM评测集：GSM8K（硬推理）、MBPP（代码过拟合）、JailbreakBench（安全）、LLMBar（偏好）、RewardBench（奖励）、Arena100K（偏好）、SummEval（摘要）、MATH-500（数学）等。

**📈 对比分析**

与单一最佳判别器、全面面板、匹配K、相关/质量多样性、可靠性仲裁、Frugal cascade等基线比较。结果表明，在大多数场景下，角色策略在保持或提升准确率的同时显著降低调用成本；在安全与偏好等条件化任务中更显优势。

**⚠️ 局限性**

局限性包括：依赖有限的审计样本，阈值选择对稳定性敏感；对大规模稀疏输出单元需要更复杂的校准；仅考虑单步增益，可能忽略多候选之间的交互；部署时需提前获取路由信号，若无可用则无法直接应用。

---

## 250. OrthoSkillVLA: Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation

**arXiv ID:** 2608.19589 | [PDF](https://arxiv.org/pdf/2608.19589v1)

**作者:** Jiaqi Wang `[一作]` (Southeast University), Yi Zhou `[通讯]` (Southeast University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究如何在预训练的视觉‑语言‑动作（VLA）模型上实现无重放的连续技能学习，并提出 OrthoSkillVLA 框架。

**💡 创新点**

创新点包括：①模块感知的子空间预算，为 VLM 与 ActionHead 设定不同的能量阈值以平衡语义可塑性与速度模式的保留；②轻量级 Feature‑Aware MoE 速度解码器与训练无关的投影路由，解决输出层表达瓶颈。

**🔧 技术方法**

采用梯度投影的正交低秩适配（LoRA）、SVD 能量阈值子空间估计、MoE 解码器、投影路由，并在模拟与真实机器人实验中验证。

**📊 数据集**

使用 LIBERO 机器人持续学习基准重组的技能集（如 OpenClose、Turn、PickPlace 等）以及 X‑VLA 预训练模型，并在实际 7‑DoF xArm+Inspire 手部平台上收集 50 条演示进行实验。

**📈 对比分析**

与 SeqLoRA、IncLoRA、EWC、OLoRA、KeepLoRA 等无重放持续学习基线比较，评估 FWT、NBT、AUC 与最终成功率。OrthoSkillVLA 在模拟实验中达到 FWT 0.94、NBT 0.13、AUC 0.88、最终 83.5%；在真实实验中最终成功率 86.25%，显著优于基线。

**⚠️ 局限性**

局限性在于对梯度子空间估计的前向假设在极端多样化动作分布下可能不足；MoE 解码器虽轻量但仍增加了一些模块；未探索与演示重放混合或多任务连续学习场景；对硬件计算资源有一定需求。

---

## 251. Robust Cross-Modal Foundation Model Perception for Underwater Robots under Degraded Visual Conditions

**arXiv ID:** 2608.19710 | [PDF](https://arxiv.org/pdf/2608.19710v1)

**作者:** Mohammad Arif Ul Alam `[一作]` `[通讯]` (North Carolina A & T State University), Mohammad Arif Ul Alam (North Carolina A & T State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在水下视觉信息逐渐退化时，如何通过跨模态融合保持目标识别性能，并提出在训练时显式暴露视觉可靠性变化的融合策略。

**💡 创新点**

创新点在于：①冻结大型自监督视觉基础模型（DINOv2）和声纳编码器，避免下游任务对预训练模型的微调；②设计轻量化的门控融合机制，并通过跨所有退化级别的训练使其能自动调整视觉与声纳的权重；③在五级视觉退化基准上系统评估并展示声纳在高噪声环境下的显著补偿作用。

**🔧 技术方法**

使用的技术包括：DINOv2‑small 视觉编码器、ResNet‑18 声纳编码器、投影层、门控网络（softmax 权重）以及多类别交叉熵训练；实验中对视觉图像施加五级合成退化（亮度、颜色、浊度、散射、模糊）。

**📊 数据集**

数据集：Underwater Multimodal Object Detection (UMOD) 的同步视觉‑声纳对，保留 5 类目标（cage、frame、hook、anchor、tire/ROV/plastic bucket/鱼/油桶）进行分类实验。

**📈 对比分析**

对比方法包括：YOLO11n 视觉检测、冻结 DINOv2 视觉分类、声纳上下文单独分类、固定特征拼接、仅在清晰图像上训练的门控融合、以及在全退化级别上训练的门控融合。结果显示：在最严厉的 D4 退化下，降解感知门控的平衡准确率为 0.6152，较纯视觉基础模型 0.4610 提升 33.5%，MRR 达 1.0278；在清晰/中度退化下性能与视觉基线相当或略优。

**⚠️ 局限性**

限制包括：仅使用 UMOD 的小子集且仅 5 类目标，缺乏端到端检测评估，声纳特征未对齐至目标框，实验仅对视觉进行退化而未模拟声纳噪声或缺失，退化模型为合成近似，未验证在真实水下环境下的迁移性能。

---

## 252. Spiking Local Interaction and Adaptive Complementary Fusion for Spiking Transformer

**arXiv ID:** 2608.19238 | [PDF](https://arxiv.org/pdf/2608.19238v1)

**作者:** Dongcheng Zhao `[一作]` (Chinese Academy of Sciences), Tielin Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了 Spiking Local Interaction（SLI）与 Adaptive Complementary Fusion（ACF）两种机制，旨在补充 Spiking Transformer 中因二值化导致的稀疏离散注意力，增强令牌间的空间交互；

**💡 创新点**

创新点在于：①设计了无需 query–key co‑activation 的局部信息交换路径（SLI）；②引入了层级、通道级可学习权重（ACF）动态平衡 SSA 与 SLI 的贡献，使两种交互模式在不同深度与通道上自适应协作；

**🔧 技术方法**

技术手段包括：Leaky Integrate‑and‑Fire 神经元、SPS 预处理、轻量化 depthwise‑pointwise 卷积实现 SLI、可学习的通道系数 γ_ssa、γ_sli、以及 surrogate gradient 训练；

**📊 数据集**

实验数据集涵盖 ImageNet‑1K、CIFAR‑10、CIFAR‑100、CIFAR10‑DVS（事件流）以及 ADE20K（语义分割）；

**📈 对比分析**

与 Spikingformer、QKFormer 等基线在相同架构与训练设置下进行对比；在 ImageNet‑1K 上 Top‑1 提升至 84.37%（HST‑8‑768），CIFAR‑10/100 提升 0.3‑0.5%，ADE20K mIoU 提升至 37.5%（QKFormer+SLI+ACF），显著优于其他 Transformer‑based SNNs；

**⚠️ 局限性**

局限性在于：局部交互仍受邻域大小与网络深度手工设定的限制，参数增量虽小但不零；对极高分辨率或更复杂多模态任务的适应性尚未验证；以及在不同 neuromorphic 硬件上能耗与延迟的评估仍待进一步研究。

---

## 253. From Retrieved Context to Runtime Control: Adaptive Compression for Edge-based RAG

**arXiv ID:** 2608.19535 | [PDF](https://arxiv.org/pdf/2608.19535v1)

**作者:** Zlatan Feric `[一作]` (Northeastern University), David Kaeli `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对边缘设备的检索增强生成（RAG）系统，提出并实验了基于实时设备遥测的自适应上下文压缩策略。

**💡 创新点**

创新点在于将上下文压缩视为运行时可调的系统开关，而非静态预处理；通过测量压缩率与生成成本的折中点（两个“拐点”），构建了可在不同负载下动态决定是否压缩及压缩率的控制框架。

**🔧 技术方法**

技术上使用了LLMLingua‑2（可调压缩率的提取式压缩器）与Llama、Qwen系列LLM（1B–8B）在NVIDIA Jetson AGX Thor上进行端到端评估；对生成阶段的预填、KV缓存、内存流、能耗进行阶段级计时与功耗采集。

**📊 数据集**

数据集包括Natural Questions和HotpotQA，检索索引基于English Wikipedia 2018，采用k=1/5/10的检索深度。

**📈 对比分析**

比较方法为对比未压缩与不同压缩率（0.15–1.0）的压缩后系统在GPU/SoC能耗、查询延迟以及回答质量（token‑level F1）上的差异。结果显示，在Llama‑8B/k=10等重负载下，压缩率0.3可实现GPU能耗下降53.2%、SoC能耗下降48.2%，并保持与未压缩几乎相同的质量；而轻度压缩（0.9）甚至导致能耗增加。

**⚠️ 局限性**

局限性包括：实验仅在Jetson AGX Thor和fp16 1–8B模型上进行，未探索更轻量化或量化压缩器；未考虑更大模型或不同量化精度；压缩器本身的固定计算开销在轻量场景下可能抵消收益；未将压缩与检索深度、重排序、查询重写等其他RAG调优手段结合评估。

---

## 254. Loreley: Repository-Scale Program Evolution with Quality-Diversity Search

**arXiv ID:** 2608.19703 | [PDF](https://arxiv.org/pdf/2608.19703v1)

**作者:** Mohan Chen `[一作]` `[通讯]`, Mohan Chen

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了Loreley系统，利用质量‑多样性（QD）方法在完整的Git仓库状态上进行搜索，使用LLM编码代理在隔离工作树中生成提交，并将成功提交保存在QD归档中以供后续父/灵感采样；并在Zstandard压缩程序上开展实验比较QD、顺序冠军和独立根搜索三种策略；同时在两份Python库上展示了多文件改进案例。

**💡 创新点**

创新点在于：1) 将完整提交作为QD搜索单元，保留多样化的仓库状态而非单一冠军；2) 学习并应用行为描述符（文本嵌入+PCA）划分MAP‑Elites格子；3) 在每个格子中维护Pareto前沿，支持多目标取舍；4) 通过灵感采样和父节点采样实现跨状态的编辑启发；5) 在完整的外部评估器下验证改进的实用性。

**🔧 技术方法**

使用技术包括：Git工作树隔离、LLM编码代理（gpt‑5.6‑sol/​gpt‑5.6‑luna）、文本嵌入3‑small + PCA投影、MAP‑Elites格子化、Pareto前沿保留、批量调度/并行执行、外部评估器（Zstd压缩/解压吞吐测试）。

**📊 数据集**

数据集主要为Zstandard源代码及其自定义压缩/解压测量集；能力案例中使用markdown‑it‑py与python‑pathspec两份Python库代码。

**📈 对比分析**

比较方法：在同一根仓库上进行七个配对实验块，每块使用48个候选提交预算，分别运行Loreley QD、顺序冠军和独立根三种策略；评估指标为压缩‑吞吐比。实验结果显示顺序冠军平均/中位数最好，Loreley QD与独立根无显著差异，未显示QD优势；在能力案例中实现了1%–25%吞吐提升。

**⚠️ 局限性**

限制：仅在单一Zstandard仓库、48提交预算下测试，可能不足以显现QD长期优势；未分离描述符、保留策略、采样等组件效应；未评估跨仓库泛化；缺乏更长时间或更大规模实验验证。

---

## 255. DeltaMomentum: A Key-Value based Anisotropic Momentum Update via Delta Rule

**arXiv ID:** 2608.19491 | [PDF](https://arxiv.org/pdf/2608.19491v1)

**作者:** Euijin Hong `[一作]` (Carnegie Mellon University), Guannan Qu `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于经典 delta 规则的方向感知动量更新（DeltaAdamW），用来替代传统的指数移动平均（EMA）动量。

**💡 创新点**

创新点在于将线性层梯度的 key‑value 结构（输入向量与误差向量）直接映射到动量缓冲区，将动量当作在线关联记忆，按方向出现频率自适应忘记率，并实现了隐式输入侧预条件化而无需矩阵求逆。

**🔧 技术方法**

技术包括：delta 规则更新、方向感知动量缓冲、μP 宽度可转移性、无额外持久内存、与现有预条件化方法（Shampoo、SOAP、Muon）兼容。

**📊 数据集**

主要使用 FineWeb‑Edu 语言模型预训练数据（Llama‑2 风格）以及 CIFAR‑10 的图像分类数据，测试模型规模分别为 67M、370M、1B 参数。

**📈 对比分析**

与 AdamW 以及 Muon 基准进行对比。DeltaAdamW 在 FineWeb‑Edu 上相较 AdamW 可减少约 46%（67M）至 22%（370M）步数，终值更低；1B 参数也保持优势。对 ResNet‑18、ViT‑Tiny 等任务同样表现出更快收敛、较低损失。实现开销仅为 22–25% 线性层 FLOPs，且无需额外存储。

**⚠️ 局限性**

局限性包括：仅在语言模型和 CIFAR‑10 分类任务上验证，尚未测试更大规模或其他生成/强化学习任务；实现仍未融合在后向梯度计算中，导致实际时间成本略高；与 Muon、Shampoo 等方法的组合尚未系统评估。

---

## 256. Hallucination as a Feature, not a Defect: Evaluating a multi-agent architecture to transform speculative language-model outputs into testable scientific hypotheses

**arXiv ID:** 2608.19206 | [PDF](https://arxiv.org/pdf/2608.19206v1)

**作者:** Nicolas Rodriguez-Alvarez `[一作]` `[通讯]` (IES Parquesol), Nicolas Rodriguez-Alvarez (IES Parquesol)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于 Rust 的多代理系统，通过高熵生成、检索、语义过滤和评估等环节，将 LLM 的“幻觉”转化为结构化的科研假设。

**💡 创新点**

提出了将高熵生成与严格评估分离的“认知摩擦”架构，证明在控制幻觉的前提下可提升假设的原创性、可行性与多样性。

**🔧 技术方法**

技术包括：Rust 编写的多代理编排；生成代理使用 Mistral‑large；批判与评估代理使用 Gemini‑3.1；语义过滤使用 Gemini‑3.1‑flash‑lite；外部 Web 搜索作为经验检验；多轮迭代和侧向视角（lateral lenses）作为约束。

**📊 数据集**

使用三组问题种子：海水淡化（seed 42、43）和议会僵局（seed 42）；没有传统训练集，所有评估基于 LLM 自评和词向量相似度。

**📈 对比分析**

通过与直接 LLM、单纯自我反思、以及三种消融（无过滤器、无检索、无侧向视角）比较，使用结构有效性、可行性、原创性、组合稀有度、多样性和聚集率等多维指标；结果显示直接 LLM 最弱，self‑reflection 在高创新与低聚集中表现最好，full_system 在原创性和稀有度上领先，但整体优势并不显著。

**⚠️ 局限性**

局限包括：仅3个问题种子，缺乏专家真实验证，评估依赖 LLM 评分与词语相似度的浅层代理；未归一化调用成本和 token 数；消融实验规模有限，统计意义探索性强。

---

## 257. Asymmetric Attention Heads: Structured Head-Wise Context Allocation for Transformer Attention

**arXiv ID:** 2608.19203 | [PDF](https://arxiv.org/pdf/2608.19203v1)

**作者:** Zimu Zhao `[一作]` `[通讯]`, Zimu Zhao

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多头注意力中头与上下文窗口不均匀分配，提出 Asymmetric Attention Heads (AAH) v3 通过层级控制动态分配不同头的局部上下文窗口。

**💡 创新点**

创新在于将上下文长度视为每头可调的离散变量，通过特征导向的层级分组、联合兄弟评分、父子约束以及分辨率 EMA 平滑，实现在保持标准 MHA 接口的同时实现异步头级上下文分配。

**🔧 技术方法**

使用的技术包括 EMA 平滑的头特征统计、余弦相似度层级聚类、宽联合兄弟评分（joint scorer）、父子约束、窗口桶化执行、Dense‑masked / FlashAttention 后端以及实验的验证损失、Attention Coverage Ratio (ACR) 诊断。

**📊 数据集**

主要在 1B 规模的英文语言模型数据集上，使用 4096-token 的上下文长度进行训练。

**📈 对比分析**

与全注意力 baseline 对比，AAH 在相同 seed‑0、10000 步下的验证损失可低于 baseline（如 Shallow freeze 6.5367 vs 6.5672），ACR 大幅下降（0.28–0.37），但在当前实现中未能获得 GPU FLOPs 或吞吐量提升，AAH 的吞吐量甚至低于 baseline。

**⚠️ 局限性**

限制包括：仅单一 seed、单一 4096-token 上下文；未证明自适应层级本身是因果因素；当前实现未能实现硬件 FLOPs 降低；需要多 seed、长上下文、不同基准验证以进一步确认。

---

## 258. SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?

**arXiv ID:** 2608.19799 | [PDF](https://arxiv.org/pdf/2608.19799v1)

**作者:** Zhipeng Xu `[一作]` (Shanghai Innovation Institute), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向科学软件的仓库级评测基准SWE-bench Science，包含119个任务，覆盖20个科学领域，并提出Issue-driven、Expert-exploratory、Engineering-integration三种任务范式。

**💡 创新点**

创新点在于将科学软件工程纳入仓库级修复评测，设计了统一的Chain-of-Evidence构造流程，定义四类科学失效机制，并引入科学辅助信息的可分离对照实验。

**🔧 技术方法**

使用了大型语言模型与编程辅助工具的代码修复框架（Codex、Claude、Kimi、GLM等），结合测试套件（公开/私有）评估修复质量。

**📊 数据集**

数据集为119个任务，来源于98个GitHub开源科研项目，覆盖20个科学领域；此外还有91个可进行科学信息分离的子集。

**📈 对比分析**

通过比较八种模型在公共/私有测试、Pass@1、Fail2Pass、Pass2Pass等指标的表现，发现最佳模型Claude-Opus-5在Pass@1上达47.9%，但整体通过率低于50%；同时对比科学信息对性能的影响，发现模型对外部知识的敏感度不同。

**⚠️ 局限性**

局限性包括任务数量在各领域仍有限，导致跨域比较不稳健；对科学知识如何被模型利用的分析仍不足，缺乏深入机制解释。

---

## 259. SCAPE: Scenario-Conditioned Simulation-Augmented Policy Evaluation

**arXiv ID:** 2608.19425 | [PDF](https://arxiv.org/pdf/2608.19425v1)

**作者:** Dijie Zhu `[一作]` (University of California Los Angeles), Chen Tang `[通讯]` (University of California Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种场景条件的模拟增强政策评估框架，旨在通过有限的真实世界评估样本和大量的模拟测试数据来预测政策在特定场景下的真实世界表现。

**💡 创新点**

创新点在于通过场景条件的方式进行政策评估，克服了现有方法仅关注平均性能的局限性，提供了更细粒度的性能预测和不确定性校准。

**🔧 技术方法**

使用了场景条件的神经网络模型和符合预测（conformal prediction）技术来校准预测的不确定性。

**📊 数据集**

在两个领域进行了验证：自主驾驶（nuPlan）和四足动物速度跟踪，使用了有限的真实世界样本和大量的模拟样本。

**📈 对比分析**

与现有的基线方法相比，提出的方法在场景级性能预测误差上显著降低，分别在自主驾驶和四足动物任务中减少了4.9%/34.7%和14.5%/27.7%的预测误差，且在样本效率和不确定性校准方面表现更佳。

**⚠️ 局限性**

局限性在于当前使用的分割符合预测提供的置信区间在场景分布上具有相同的大小，而不是基于场景的条件覆盖。此外，真实世界评估的范围仍然有限，未来需要在不同的任务和场景中进行更广泛的实验。

---

## 260. Empirical Evaluation of Cross-Carrier MCPTT & OTT MCX Interoperability in High-Density Environments

**arXiv ID:** 2608.19554 | [PDF](https://arxiv.org/pdf/2608.19554v1)

**作者:** Eman Hammad `[一作]` (Texas A&M University), Michael Fox `[通讯]` (Center for Applied Communication and Networks)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在德克萨斯农工大学Kyle体育场对105,000+观众的高密度场景中，使用12部相同Android手机进行现场实验，比较了标准OTT推送对讲（OTT PTT）与3GPP标准化的Mission‑Critical Push‑To‑Talk（MCPTT）在两大运营商跨运营商环境下的语音质量、连接成功率、网络延迟与抖动。

**💡 创新点**

首次在真实高负载场景下系统性量化跨运营商互操作性，揭示了传输层抖动超过150 ms时会出现非线性“崩溃”边界；证明MCPTT通过优先级映射能显著抬升语音质量并避免此崩溃，同时指出不同运营商DAS层面仍存在可测量的性能差异。

**🔧 技术方法**

采用3GPP MCX/MCPTT标准、OTT PTT应用、GL Communications VQuad自动化呼叫平台、Dual UTI硬件、Perceptual Objective Listening Quality Analysis (POLQA)及E‑Model R‑Factor评估，结合统计方法（ANOVA、ECDF）对网络指标进行分析。

**📊 数据集**

实验数据来自一场正式比赛期间的现场采集，涵盖7个测试点（TP1‑TP7），使用两家运营商（Carrier 1、Carrier 2）配置，分为优先级MCPTT和非优先级OTT两种配置，总计12个相同手机进行连续呼叫采样。

**📈 对比分析**

与传统OTT PTT相比，MCPTT在宏基站（TP1）达到平均MOS 3.28，成功率≈100%；在DAS内（TP2‑TP7）平均MOS 3.4‑4.1，成功率>99%。OTT PTT在同样位置的MOS低于2.5，成功率仅约60%。进一步对比发现Carrier 1的OTT MOS为3.16，Carrier 2为2.97，表明跨运营商差异显著。

**⚠️ 局限性**

实验仅覆盖两家运营商和12部相同手机，未评估多运营商切换延迟、MEC缓解、视频/数据传输及更大规模设备数量，实验环境仅限单一体育场观众密度，结果可能不适用于其他类型高密度现场或不同网络架构。

---

## 261. When Automata Meet Streams: Temporal Logic Compilation for Stream-Based Robotics Task and Motion Planning

**arXiv ID:** 2608.19453 | [PDF](https://arxiv.org/pdf/2608.19453v1)

**作者:** Sayem Nazmuz Zaman `[一作]`, Cyrus Neary `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于编译的方案 SAM‑TD，用于在流式机器人任务与运动规划（TAMP）中强制执行任意有限时序逻辑（LTL_f）约束；

**💡 创新点**

首次将 LTL_f 约束编译为确定性有限自动机（DFA）并将其转移函数同步嵌入动作方案，同时通过共享“有效性”标记（Token Destruction）实现对违反约束的路径进行即时剪枝；

**🔧 技术方法**

利用 LTL_f‑to‑DFA 转换（MONA、LTLf2DFA）、动作回归（Weakest Precondition）、PDDLStream 适配层以及 h^FF 取舍启发式，构建了无须修改原始规划器的编译框架；

**📊 数据集**

在三类 PDDLStream 环境（Kitchen、Tabletop Manipulation、ZoneSort）和六个公开 PDDL3 任务（Openstacks、Rovers、Storage）上验证了该方法；

**📈 对比分析**

与六种现有编译方法（BM06、TB15、TCORE、LiftedTCORE、LCC、Plan4Past）比较，SAM‑TD 在所有实例中均能成功求解，几乎保持最优的总时延；在流式环境中，编译时间随约束数量线性增长，预先阻塞（precondition blocking）方案导致动作实例化爆炸；

**⚠️ 局限性**

局限性包括：对 DFA 生成的双指数复杂度、对 LTL_f 语法的依赖、对多机器人或移动平台的适用性尚未验证、以及在极大约束规模下可能的记忆占用和求解时间上升。

---

## 262. New Complexity Results for Fair Repetitive Scheduling

**arXiv ID:** 2608.19952 | [PDF](https://arxiv.org/pdf/2608.19952v1)

**作者:** Moran Koren `[一作]` (Ben Gurion University of Negev), Dvir Shabtay `[通讯]` (Ben Gurion University of Negev)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究单机重复调度中公平性的最大化问题，主要解决三类已开放的复杂度问题：完成时间在三天内的最优调度、两天内等待时间与迟到/迟到的最优调度。

**💡 创新点**

创新点在于：①首次证明完成时间准则下 q=3 的公平调度问题为 NP‑hard；②证明在 q=2 的情形下，等待时间、迟到和迟到准则的公平调度问题均为 NP‑hard；③给出 q=2 时完成时间准则下的多项式算法，并对这些问题的复杂度完整归纳。

**🔧 技术方法**

技术手段主要是多项式时间归约（归约自 Partition 问题），构造特定实例并通过严谨的逻辑推理证明极值上界或下界，从而确立 NP‑hard 性。

**📊 数据集**

论文未使用实际数据集；所有证明均基于理论构造的人工实例。

**📈 对比分析**

比较方式为理论复杂度分析：对已知的多项式解、伪多项式解或 NP‑hard 判定给出清晰的分类；未涉及实验性能对比。

**⚠️ 局限性**

局限性包括：①对固定 q 的问题未给出伪多项式算法；②未解决处理时间与天数无关（p_{ij}=p_j）情况下完成时间和迟到准则的复杂度；③缺乏实验验证和启发式算法的讨论。

---

## 263. Grounding Mindfulness in Embodied Tangibles: A Scoping Review & Theoretical Framework for HCI Design

**arXiv ID:** 2608.19673 | [PDF](https://arxiv.org/pdf/2608.19673v1)

**作者:** Tharaka Sachintha Ratnayake `[一作]` (University of Melbourne), Wafa Johal `[通讯]` (University of Melbourne)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统的 scoping review 对 65 篇关于可触感设备的正念（mindfulness）设计研究进行梳理，并基于 Self‑Awareness, Self‑Regulation, Self‑Transcendence (S‑ART) 框架提出两种补充模型——Embodied Sensory Expansion (ESE) 用于聚焦注意（Focused‑Attention Meditation, FAM）以及 Embodied Sensory Anchoring (ESA) 用于开放监控（Open‑Monitoring Meditation, OMM），为可触感正念技术的设计与评估提供理论指导。

**💡 创新点**

创新点在于（1）首次系统性归纳可触感设备在正念机制（注意调节、身体觉知、情绪调节、视角转变）中的覆盖情况，指出现有研究聚焦于注意与身体觉知，情绪调节与自我视角欠缺；（2）构建基于 S‑ART 的理论框架，明确可触感设备如何支持不同正念实践；（3）提出 ESE 与 ESA 两种互补模型，阐释设备传感与反馈如何实现注意扩张与情绪锚定；（4）评估方法缺失与碎片化的批判，揭示评估缺口并为后续研究提供方法论建议。

**🔧 技术方法**

主要技术为文献检索与系统评估方法：采用 Joanna Briggs Institute 与 PRISMA‑ScR 指南进行检索、筛选与编码；使用数据提取表对每篇论文的目标、机制、技术实现与评估方法进行归类；结合利益相关者咨询（工作坊、访谈）补充实践视角；理论建模基于 S‑ART 与正念实践分类。

**📊 数据集**

数据集为 65 篇经筛选的可触感正念设备相关论文，涵盖 48 篇会议论文、16 篇期刊论文及 1 篇工业案例。未使用公开机器学习数据集。

**📈 对比分析**

本文未进行典型的性能对比实验，而是对已发表研究中的评估方式进行归类与统计：大部分研究使用自评量表（MAAS、FFMQ 等）与生理指标（HRV、EDA、呼吸频率等）；评估手段多样且缺乏统一标准，导致难以直接比较。

**⚠️ 局限性**

局限性包括：①评估方法碎片化，缺乏标准化与长期效能验证；②情绪调节与自我视角的研究不足，导致框架未得到充分验证；③仅聚焦可触感设备，排除了移动、XR 等技术；④样本规模普遍偏小，缺乏多元文化与长期跟踪；⑤咨询环节样本有限，可能不具备代表性。

---

## 264. Modular fabrication and design of thick rigid-foldable origami metamaterials

**arXiv ID:** 2608.19763 | [PDF](https://arxiv.org/pdf/2608.19763v1)

**作者:** Sunao Tomita `[一作]` (Toyota Central R&D Labs., Inc.), Tomohiro Tachi `[通讯]` (University of Tokyo)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种通过层叠模块化铰接面板制造的厚壁、非流形可刚性折叠的折纸超材料，并在其上进行了拓扑优化以实现轻量化与承载力兼顾的可部署结构。

**💡 创新点**

创新点在于将厚壁折纸的非流形连接拆解为可堆叠的多层铰接模块，通过内部空间分层实现折叠几何兼容；并提出了基于图结构的拓扑优化框架，既保证了全局折叠连通性，又实现了结构轻量化。

**🔧 技术方法**

采用了层级铰接面板的模块化切割与粘合技术、基于图结构的拓扑优化（使用SIMP、Heaviside投影和图过滤），以及条杆-铰链模型的有限元分析。

**📊 数据集**

本研究未使用公开的标准数据集，而是基于自定义的条杆-铰链模型生成的仿真数据与手工制造的厚层夹板实验样本来验证设计与性能。

**📈 对比分析**

通过与传统薄壁折纸结构对比，实验表明该折纸超材料在展开状态下可承受约11 kg（小型原型）至96 kg（大规模构造）负载，且保持单自由度折叠运动，展示出更高的载荷承载能力和可部署性。

**⚠️ 局限性**

主要局限包括：需要精确的切割与对位以保证铰接兼容；组装过程仍较为繁琐，易受手工误差影响；厚壁材料可能导致疲劳损伤，长期耐久性尚待进一步验证。

---

## 265. Learning Highly Dynamic Skills Transition for Quadruped Jumping Through Constrained Space

**arXiv ID:** 2608.19977 | [PDF](https://arxiv.org/pdf/2608.19977v1)

**作者:** Zeren Luo `[一作]` (University of Hong Kong), Peng Lu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并验证了一套层次化强化学习管线，使四足机器人能够在真实硬件上通过窄缝隙完成高动态跳跃。

**💡 创新点**

创新点在于利用模仿学习构建多样化的低层技能库，并通过高层视觉感知驱动的决策模块实现自动、连贯的跨越窄门任务，同时展示了该框架对其他高动态任务的可迁移性。

**🔧 技术方法**

采用了层次化强化学习（高层决策+低层控制）、GAN式模仿学习、RGB‑D门检测、仿真到实机的随机化迁移等技术。

**📊 数据集**

使用的训练数据来自动物（如袋鼠、犬类等）行走、奔跑、跳跃的视频序列，经过逆运动学映射到机器人形态；跳跃片段的采样权重被放大以强化学习效果。

**📈 对比分析**

与三种端到端基线（E2EStyle、E2EBase、Parkour）以及两种同类层次化方法（HierGate、HierDisc）进行对比，ConsJump在仿真与真实测试中均表现出更低的碰撞率、更低的能量消耗、100%成功率，并能适应不同门位置和复杂地形。

**⚠️ 局限性**

局限性包括低层速度指令抽象导致无法细粒度控制关节协同，难以实现非生物学运动模式；以及对极端任务的关节空间探索不足，视觉门检测受光照、遮挡等因素影响。

---

## 266. Unified Music Identification for Tracks and Versions

**arXiv ID:** 2608.19919 | [PDF](https://arxiv.org/pdf/2608.19919v1)

**作者:** R. Oguz Araz `[一作]` (Universitat Pompeu Fabra), Dmitry Bogdanov `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了统一的音乐检索基准，评估track（精确检索）与version（版本检索）在信号变形与音频降噪两种环境下的准确性与鲁棒性，并训练了一种基于段落clique的端到端统一模型。

**💡 创新点**

首次提出可同时评估track和version的统一基准，并通过动态阈值+洪水填充+clique聚类构造段落匹配训练集，证明vi模型可覆盖ti任务并将检索误差归因于检索约束而非嵌入空间局限。

**🔧 技术方法**

使用CQT-Transformer嵌入、三元组损失（带硬/软负样本挖掘）、信号变形与降噪流水线（pitch shift、time stretch、EQ、limiter、房间/麦克风IR）、以及IVF-Flat ANN索引。

**📊 数据集**

采用FMA、Official YouTube、Discogs‑vi等公开数据集，构建约116k、8k、95k track数据库，生成97M段对（20s）并聚成644k段落clique。

**📈 对比分析**

在三数据库上对七种ti/vi模型进行检索精度与鲁棒性对比，发现无模型在两任务上同时高精度且鲁棒；统一模型在10s查询下可达≈90% ti准确率、≈70% vi鲁棒性，且详细分析了误差来源。

**⚠️ 局限性**

限制包括：需要长查询（10s）导致信息不匹配与边界错位；段落时长折衷使ti与vi难以同时最优；训练数据构建复杂且硬负样本难以稳定；未覆盖更短查询长度及更强鲁棒的ti模型。

---

## 267. When Machines Speak: A Unified Generative Framework for Integrating Machine-Native Symbols into Pretrained Large Language Models

**arXiv ID:** 2608.19529 | [PDF](https://arxiv.org/pdf/2608.19529v1)

**作者:** Su Yan `[一作]` (Google Inc.), Rakesh Iyer `[通讯]` (Google Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并验证了 UniLang 统一框架，使预训练 LLM 同时以自然语言和机器原生符号为生成单元，解决了结构化预测与语言模型的分离问题。

**💡 创新点**

通过词表扩展和对比学习将机器符号嵌入 LLM 表征空间，使其成为首席生成单位，无需为每个任务构建专门的模型或手工编写大量 prompt。

**🔧 技术方法**

利用词表扩展、InfoNCE 对比对齐、LoRA 低秩适配、RQ‑VAE 量化生成机器符号，并在 Llama‑3.2‑1B‑Instruct 上训练。

**📊 数据集**

在 Amazon Beauty、MovieLens‑1M/20M（序列推荐）和 LePaRD（法律前例预测）等真实数据集上进行实验。

**📈 对比分析**

与判别式基线（SASRec、BERT4Rec、S3‑Rec）以及生成式基线（P5、TIGER）进行对比，UniLang 在 Recall@5、NDCG@5 等指标上提升约30–150%（推荐）并在法律前例预测中 Recall@1 提升近50%。

**⚠️ 局限性**

未评估符号微调对自然语言生成质量的影响，且大规模符号表的可扩展性与训练成本尚未充分探究。

---

## 268. Can Conversational AI loosen Us-Versus-Them Boundaries? The Effects of Common, Dual, and Separate Identity Framings on Pro-Immigrant Intergroup Helping

**arXiv ID:** 2608.19220 | [PDF](https://arxiv.org/pdf/2608.19220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 269. CrossQ: Task-Aligned Cross-Token Conditional Quantization for Late Interaction Retrieval

**arXiv ID:** 2608.19204 | [PDF](https://arxiv.org/pdf/2608.19204v1)

**作者:** Rohit Kumar Salla `[一作]` (Virginia Tech), Ramya Manasa Amancherla `[通讯]` (Columbia University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于文档上下文的交叉词级量化方法 CrossQ，用来压缩 ColBERT 风格的晚期交互检索索引，同时保持检索质量。

**💡 创新点**

创新点在于：①使用轻量化的文档上下文来对每个 token 的量化码进行条件分配，实现文档内有效精度分配；②采用与最大相似度（max‑sim）检索任务对齐的列表式蒸馏与硬负样本对比损失，直接优化排名结构而非均值重建误差。

**🔧 技术方法**

核心技术包括：多码本加性量化、文档上下文编码（DeepSets 结构）、温度衰减的软硬量化策略以及结合列表 KL 损失与对比损失的训练目标。

**📊 数据集**

在 MS MARCO passage ranking 和 BEIR 多域检索子集上进行评估，使用 ColBERT 作为教师模型。

**📈 对比分析**

与基准 PQ/OPQ/Token‑wise Conditional 以及 PLAID 系统比较，CrossQ 在 2 B/字节、4 B/字节和 8 B/字节下分别提升 MRR@10 约 0.010、0.019 和 0.005，达到 64 倍的原始 token 存储压缩并在 8 B/字节时保留约 98% 的全精度效果。

**⚠️ 局限性**

局限性包括：只针对 max‑sim 交互式评分设计，极端压缩时仍可能导致赢家 token 翻转；实现依赖文档上下文的额外索引计算，且对其他评分算子（如点积）需要重新设计训练信号。

---

## 270. Wave-Based Bilateral Teleoperation between Nonlinear Manipulators with Direct Contact Force Feedback

**arXiv ID:** 2608.20043 | [PDF](https://arxiv.org/pdf/2608.20043v1)

**作者:** G. Q. Bao Tran `[一作]` (University of Illinois Urbana-Champaign), Ho Duc Tho `[通讯]` (HCMC University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种针对非线性多自由度机械手在存在恒定通信延迟时的双向远程操纵控制方法，直接将环境接触力反馈到主机侧并保证闭环稳定；

**💡 创新点**

创新点在于：①基于LMI的非线性远程子系统的功率短缺（passivity‑shortage）表征；②将USP（上限严格被动）波动变换推广到多输入多输出的向量化形式；③通过离线优化调节散射参数以提升传感透明度；

**🔧 技术方法**

使用了Euler–Lagrange动力学建模、线性矩阵不等式（LMI）分析、USP波动变换通信律、Lyapunov–Krasovskii稳定性分析以及基于仿真的参数优化；

**📊 数据集**

未使用公开数据集，所有结果均通过对二维非线性平面机械臂的数值仿真得到；

**📈 对比分析**

方法通过仿真与经典无损波动变换（WT）方案比较，展示了在自由运动时位置同步和在硬接触时力感知更佳的性能；

**⚠️ 局限性**

局限性包括：①只考虑了恒定延迟；②需要先验的远程子系统状态有界假设；③参数优化为离线且可能受仿真时限影响；④尚未在真实硬件平台上验证。

---

## 271. SynFlow: A Multidimensional Diachronic Semantic Analysis Toolkit

**arXiv ID:** 2608.19472 | [PDF](https://arxiv.org/pdf/2608.19472v1)

**作者:** Bach Phan-Tat `[一作]` (KU Leuven), Dirk Speelman `[通讯]` (KU Leuven)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SynFlow工具，提供统一的多维度时间序列语义变化分析框架，兼容依赖、形态、构造和外部语义表示。

**💡 创新点**

创新点在于将不同语义维度纳入同一工作流，支持分布式比较、值层级分解和增量聚类，直接将定量变化与可解释的语言证据关联。

**🔧 技术方法**

使用概率分布比较（余弦/JS/TVD距离）、支持加权、置换检验、Benjamini–Yekutieli FDR校正，并通过词向量聚类实现增量填充词主题追踪。

**📊 数据集**

主要实验数据来自Leipzig German News（1995–2025）中的德语形容词“viral”案例；Benchmark使用SemEval‑2020 Task 1 以及Corpus of Historical American English（CHAE）进行评估。

**📈 对比分析**

在SemEval‑2020 中，Slot‑filler 方案排名第4/28，Frame Semantics 排名第9/28，整体表现优于多数神经网络系统，显示了语义维度分析的竞争力。

**⚠️ 局限性**

局限性包括对高质量解析语料的强依赖，稀疏分布导致距离估计不稳，且目前仅支持有限语言和维度，未来需扩展更多解析器和LLM标注。

---

## 272. Learning the Right Abstraction: Neural Reduced Dynamics for Complex Robot Control

**arXiv ID:** 2608.19375 | [PDF](https://arxiv.org/pdf/2608.19375v1)

**作者:** Harry Zhang `[一作]` (University of Wisconsin--Madison), Dan Negrut `[通讯]` (University of Wisconsin--Madison)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种神经简化动力学（NRD）框架，将高保真物理仿真转化为任务特定的低维动态模型，用于在该模型上进行大规模强化学习，并验证所学策略可直接在原始高保真仿真中实现。

**💡 创新点**

创新点在于：①设计了以任务相关物理为准则的状态抽象原则，区分递归状态、命令、上下文与可解析的量；②通过训练时的多步开放式滚动和模型选择来确保模型在长时域的连贯性；③展示了同一流水线可跨不同机器人系统（越野车辆、带臂车辆）实现从轨迹跟踪到端效应器到达等多种控制任务。

**🔧 技术方法**

使用了因果Transformer结构的时序预测网络、Proximal Policy Optimization (PPO)进行策略学习、Vectorized GPU批处理以提升仿真吞吐量；模型训练基于高保真Chrono仿真生成的状态‑动作轨迹，随后冻结模型用于策略训练。

**📊 数据集**

数据集来源于Chrono仿真，分别包含：HMMWV在平地、粗糙地面和可变形土壤三种地形下的轨迹（≈82k条、2k条、20条）以及带臂车辆的驾驶与臂运动两种模式（分别约2.16k条与1.5k条）。

**📈 对比分析**

通过模型在开放式滚动误差、单步损失以及最终的闭环高保真仿真评估进行比较。实验显示，训练在NRD模型上的策略在所有三种地形下的跟踪误差低于单一地形专家；在带臂车辆中，分别在平面到达和臂端到达任务中，策略在高保真仿真中分别实现了100/100和97/100的成功率，且无碰撞或关节限制违规；NRD模型的模拟吞吐量相较Chrono提升了4–5个数量级。

**⚠️ 局限性**

局限性包括：①仅在仿真环境中验证，缺乏硬件实际测试；②状态抽象的选择仍需人工物理直觉，未实现自动化；③未处理显式接触模式转移（如抓取、碰撞）导致动态不连续；④模型对高保真仿真的偏差继承，无法纠正原始物理模型误差。

---

## 273. Online Test-Time Adaptation for Generalizable Dynamic Graph Anomaly Detection

**arXiv ID:** 2608.19858 | [PDF](https://arxiv.org/pdf/2608.19858v1)

**作者:** Jialun Zheng `[一作]` (Hong Kong Polytechnic University), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出OTTA-DGAD，一种在线测试时自适应的通用动态图异常检测框架

**💡 创新点**

创新点包括：①动态原型（dynamic prototypes）从时间自邻图提取并存入缓冲区；②结构支持的伪标签（entropy + structural support）来更新原型；③跨块上下文增强（cross‑chunk context enrichment）保留历史信息；④在无标签目标流中仅冻结预训练骨干，轻量化自适应模块

**🔧 技术方法**

使用图神经网络 + Transformer 作为特征编码器；对原型进行分布建模并利用二次项判别得分；使用置信度+结构一致性评估伪标签；利用最近历史表示的相似度进行跨块上下文聚合

**📊 数据集**

在十个真实世界动态图数据集上评估（如 Wikipedia、MOOC、Bitcoin‑OTC、Email‑DNC、UCI Messages、AS‑Topology、TAX51、DBLP、Synthetic‑Hijack 等），每个数据集包含不同领域的节点/边统计

**📈 对比分析**

与三类基线（传统 DGAD、通用型 GAD、专门设计的 OTTA 模型）和多种最新方法（DP‑DGAD、GeneralDyG、FALCON、SLADE、TADDY、ADCSD、LCoTTA 等）进行对比；在在线测试‑自适应设置下，OTTA‑DGAD 在 AUROC 与 AUPRC 上均领先，提升约 5–10% 并显著提高跨块稳定性

**⚠️ 局限性**

限制包括：依赖缓冲区大小与正则化参数的手工调节；在极端类别不平衡或噪声较高的场景下伪标签可能仍不够可靠；当前仅冻结预训练骨干，无法进一步利用可微分自适应层提升性能

---

## 274. Understanding as an Explicit and Assessable Component of Frontier AI Safety Decisions

**arXiv ID:** 2608.19816 | [PDF](https://arxiv.org/pdf/2608.19816v1)

**作者:** Stephen Barrett `[一作]` (Arcadia Impact AI Governance Taskforce), Phillip Mulvana `[通讯]` (Arcadia Impact AI Governance Taskforce)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并验证了一套将前沿 AI 部署决策中的“理解”显式化、可评估化的完整方法论，并在 RobotCorp（机器人公司使用 AI 编码代理）和 IABIED（极高不确定性与决策关键性场景）两种案例中进行试验，展示方法的可行性与生成性。

**💡 创新点**

创新点包括：① 将 Assurance 2.0 的安全案例扩展为 Understanding Basis (UB)，加入内部一致性、依托性、可喜虚假与外部一致性等分析；② 引入 Personal Understanding Statement (PUS) 作为决策者对 UB 的个体化评估框架；③ 设计多维度“对象×能力”矩阵、摩擦测试与残留不理解登记表，以系统化检验决策者的内部理解；④ 强调动态演化与反射均衡的工作流程，支持安全、决策与系统设计的协同迭代。

**🔧 技术方法**

主要技术手段包括：Assurance 2.0 的 CAE 结构、正负视角与残余风险评估；理性缺陷（defeater）分析与外部一致性搜索；HAZOP 风格的引导词检验；设计摩擦测试以激活认知检验；以及基于情景的角色分析与实验设计。

**📊 数据集**

由于该工作为方法论验证，未使用传统机器学习或大规模数据集；主要使用了 Google DeepMind 的“架构缺陷”安全案例、RobotCorp 虚构系统的安全需求与风险路径，以及 IABIED 公开论证文本，作为情境素材。

**📈 对比分析**

本研究未给出量化性能指标，而是通过角色分析和压力测试来比较方法的有效性。结果表明：方法能够生成新的安全设计方向、发现未预见的缺陷并推动决策框架调整；在高不确定性/高关键性场景下，方法依旧适用，但需更强的证据链和更细致的评估。

**⚠️ 局限性**

局限性包括：① PUS 评估测试尚未在真实环境中检验其可行性与效果；② 研究规模有限，未覆盖更复杂或大型系统；③ 对决策者层级递归理解的支持尚不充分；④ 需要工具与模板支持以实现高效实施；⑤ 对 AI 开发商提供的组件安全案例缺乏完整性，方法在实际应用时仍需配合额外的安全保证工作。

---

## 275. Survival of~the~Stealthiest: Evolving Low-Entropy Ransomware via~Genetic Algorithms

**arXiv ID:** 2608.19821 | [PDF](https://arxiv.org/pdf/2608.19821v1)

**作者:** Efrat Levenberg `[一作]` (JCT), Harel Berger `[通讯]` (Ariel University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了利用遗传算法进化低熵勒索软件，使其在保持低检测概率的前提下实现持续加密攻击。

**💡 创新点**

创新点在于将勒索软件加密过程视为搜索式软件工程优化问题，利用遗传算法在硬性异常阈值约束下演化加密策略，从而实现“低速持续”型隐蔽攻击。

**🔧 技术方法**

使用了遗传算法（GA）进行策略进化，基于Isolation Forest进行异常检测与动态阈值更新，采用简单替换加密（凯撒密码）进行实验。

**📊 数据集**

数据集为30个ASCII编码的.txt文件，运行环境为Ubuntu 24.04虚拟机，内存4GB。

**📈 对比分析**

与静态攻击基线（固定加密量1字节，60秒间隔）对比，GA进化攻击在30,230秒内完成100%加密进度，仅出现21次瞬时异常峰值，未触发终止，展示出更高的持久性和隐蔽性。

**⚠️ 局限性**

限制包括：仅在单一文件类型的受控环境中验证，未考虑真实多样化文件与系统噪声，未对抗主动端点防御（EDR）系统，故对实际生产环境的适用性需进一步验证。

---

## 276. SAGE: Ergodic Control for Autonomous and Adaptive Inspection of Subsea Infrastructure

**arXiv ID:** 2608.19671 | [PDF](https://arxiv.org/pdf/2608.19671v1)

**作者:** Markus Buchholz `[一作]` (Norwegian Defence Research Establishment), Yvan R. Petillot `[通讯]` (Heriot-Watt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SAGE框架，利用可变风险分布进行海底设施阀门的自适应巡检，动态分配ROV的巡检时间。

**💡 创新点**

创新点在于将实时更新的风险权重直接嵌入无规划Ergodic Control控制律，使车辆能够即时响应新检测到的泄漏而无需人工重新规划路径。

**🔧 技术方法**

采用Ergodic Control与Fourier基函数投影，并通过谱多尺度覆盖法实时生成速度指令，实现连续的覆盖控制。

**📊 数据集**

使用Gazebo仿真环境中的两棵Subsea Xmas Tree模型和五个阀门点的手动设定风险配置进行实验。

**📈 对比分析**

与固定A*循环巡检路由相比，SAGE在300秒任务中将高风险阀门P1的复检间隔从8.1秒缩短至5.8秒，复检次数提升近1.4倍，且无需人工干预。

**⚠️ 局限性**

局限在于Ergodic Control仅约束长期平均复检率，无法保证低风险点在有限任务内被检查，导致如P3在实验中完全未被访问，需要加入最小权重或硬性复检间隔约束。

---

## 277. When Irrelevant Text Matters: Affine Margin Shifts in Multimodal Large Language Models

**arXiv ID:** 2608.19208 | [PDF](https://arxiv.org/pdf/2608.19208v1)

**作者:** Yinfeng Wang `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模态大型语言模型（MLLMs）在面对与图像无关的文本上下文时的行为变化，设计了受控干预实验，提出决策边际（log‑probability 差）作为度量，发现上下文会导致边际线性变换，进一步提出后置校准方法来减缓这种干扰。

**💡 创新点**

创新点在于：①将非任务相关上下文视为可控干预，并在二元视觉判断任务中量化其影响；②发现上下文条件下的决策边际遵循一致的仿射变换，揭示干扰具有可估计的结构；③将仿射参数解释为视觉承诺保持与方向性偏移，并用轻量化后置校准实现性能恢复。

**🔧 技术方法**

主要技术包括：二元视觉判断框架、决策边际（log‑probability 差）计算、仿射拟合与逆变换、后置校准（affine correction）、实验对比与统计分析（准确率、是率、翻转率、R² 等）。

**📊 数据集**

使用四大视觉‑语言基准：POPE、AMBER、MME、GQA；并在三款主流 MLLM（LLaVA‑1.5‑7B、Qwen2‑VL‑2B/7B、InternVL3‑8B）上进行实验。

**📈 对比分析**

对比在无上下文（context‑free）与添加中性/条件上下文（context‑neutral / context‑condition）两种设置，发现：①准确率下降、是率下移、翻转率升高；②仿射拟合 R² ≥ 0.74，说明变换可靠；③后置校准后准确率在 4–6% 之间提升，翻转率大幅下降，且跨数据集校准仍能部分恢复性能。

**⚠️ 局限性**

局限性包括：①实验仅覆盖二元判断，未完全捕捉开放式生成；②上下文干扰为简化的控制样本，未涵盖真实场景中多样化对话、检索段落等；③理论解释仍停留在经验层面，缺乏完整因果或模型机制阐述。

---

## 278. Complementary, Not Cumulative: Interaction Effects in Physics-Informed Neural Networks for Navier-Stokes Vortex Shedding

**arXiv ID:** 2608.19632 | [PDF](https://arxiv.org/pdf/2608.19632v1)

**作者:** Devesh Shah `[一作]` `[通讯]` (Independent Researcher), Devesh Shah (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了多种PINN技术在DFG/Schäfer–Turek圆柱悬空流场基准上的组合效果

**💡 创新点**

发现仅有周期激活与因果加权组合才能可靠学习涡脱落，其他技术即使单独有效也可能相互冲突

**🔧 技术方法**

使用SIREN激活、因果权重、傅里叶特征编码、硬边界约束、自适应损失权重以及L-BFGS微调

**📊 数据集**

使用OpenFOAM生成的Schäfer–Turek 2D-1基准数据集（Re=20和Re=100）

**📈 对比分析**

通过相对L2误差、Strouhal数、振幅比和相关性比较，SIREN+因果权重在充分训练后可将误差降至3%以内，超过其他组合

**⚠️ 局限性**

局限在于仅在二维低Re基准下验证，未测试三维或高Re流场；并且组合复杂时容易出现优化崩溃

---

## 279. Remember, Verify, or Ask? Cross-Family Evaluation of Memory Commitment in LLM Agents

**arXiv ID:** 2608.19564 | [PDF](https://arxiv.org/pdf/2608.19564v1)

**作者:** Baichuan Li `[一作]` (Southern Methodist University), Zihao Zheng `[通讯]` (Washington University in St. Louis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大语言模型代理在长期记忆使用中的决策边界（memory‑clarification boundary），并创建了包含 140 条持久化、临时使用、验证与澄清场景的 benchmark MCB，评估代理在何时持久化信息、何时仅临时使用、何时向世界验证、何时向用户澄清。

**💡 创新点**

创新点包括：①提出四类明确的记忆承诺动作并构造对应情景；②设计包含对照集的 benchmark，提供可审核的人工标签和结构化工具调用评估；③在 Claude 与 Qwen 两大模型族上进行跨模型对比，并分离 prompt 与 policy 对准确率与安全相关行为分布的影响；④揭示标签决策与工具调用之间的显著差距。

**🔧 技术方法**

采用了 LLM 交互、few‑shot prompting、五条规则的 policy prompt、结构化工具调用模式、Bootstrap 95% 置信区间、McNemar 精确检验与 Holm 校正等技术手段对结果进行统计与显著性检验。

**📊 数据集**

使用自建的 Memory‑Clarification Boundary benchmark，包含 140 条主要情景（70 开发/70 hold‑out）以及 70 条对照（含 35 对证据翻转项），所有情景由作者编写并由非作者独立审核标签。

**📈 对比分析**

与 Always‑Persist、Majority Action、Keyword heuristic、Category oracle 等基线对比，测量准确率、宏 F1、过度持久化率、澄清召回率和验证召回率。结果显示：few‑shot 提升准确率 0.214（p<0.002），policy prompt 使过度持久化率下降 0.143（p=0.038），但工具调用模式导致准确率显著下降；同时，澄清召回率始终低于验证召回率。

**⚠️ 局限性**

局限性包括：benchmark 样本量有限，情景为合成且仅包含英语；对照集过度依赖规则，缺乏自然多样性；工具调用评估仅记录选择而不执行实际操作；量化结果受特定模型、量化与服务栈影响；未考虑真实任务中的整体性能与用户交互反馈。

---

## 280. Rationally Enriched Chebyshev Trunk Bases for DeepONet Surrogates of High Péclet Entrance Transport

**arXiv ID:** 2608.19658 | [PDF](https://arxiv.org/pdf/2608.19658v1)

**作者:** Mingeun Choi `[一作]` (Georgia Institute of Technology), Satish Kumar `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了一种使用理性增广Chebyshev（REC）字典的DeepONet替代模型，用于处理具有薄边界/壁层的奇异扰动和高Péclet传输问题。

**💡 创新点**

创新点在于将Chebyshev多项式与通过AAA算法构造的理性函数相结合，形成预定义的输出坐标字典，能够在不增加可训练trunk参数的情况下精确捕捉局部层结构。

**🔧 技术方法**

采用Deep operator network（branch–trunk）框架，构造递归理性字典，使用Adaptive Antoulas–Anderson算法；训练时结合Adam与L-BFGS优化器；采用Chebyshev–Lobatto离散、误差指标E₂^rel、E_∞、E_max^layer、E_LA进行评估。

**📊 数据集**

数据集为三类合成基准问题：奇异扰动标量边值问题、热入口（Graetz）问题、带吸收壁的浓度入口问题；通过确定性求解器在Shishkin网格和隐式步进法上生成数值参考解，并对源项/入口条件、ε、Pe^-1、Pe_m^-1、Da进行随机采样。

**📈 对比分析**

将REC-trunk与普通DeepONet（Vanilla）和仅Chebyshev字典的DeepONet（Chebyshev）在相同数据划分、相同训练次数（5次）下进行比较；REC在所有指标上显著优于Vanilla（最多约60%误差下降），相较于Chebyshev在最小扰动参数区间内提升约10–20%，并在热入口和浓度入口问题中均保持一致的性能优势。

**⚠️ 局限性**

局限性包括：对完整测试集的优势在标量BVP中不明显，仅在最小扰动参数下显著；REC字典对外层Chebyshev子字典大小敏感，需要手动调参；目前仅验证了一维壁面剖面，是否能推广到更高维多物理耦合问题尚未证明。

---

## 281. Stable Within, Unidentified Across: Semantic Identification of Benchmark Effects and Rankings

**arXiv ID:** 2608.19269 | [PDF](https://arxiv.org/pdf/2608.19269v1)

**作者:** Xi Qin `[一作]` `[通讯]` (Wuhan University), Xi Qin (Wuhan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对固定轨迹、预测和方法集合，在不同评估语义族（合同、空值处理、终端可比性）下审计其效应与排名的可识别性，提出有限识别集与不变性判定的审计记录格式。

**💡 创新点**

引入评估语义识别概念，将族内与族外识别分离；定义有限识别集与不变性判定规则；通过终端可比性控制实现分歧证据的本地化和因子删除法揭示具体语义操作。

**🔧 技术方法**

冻结子样本与预测、规则化分数、分层Bootstrap、分组对比、因子删除法；构造三家族评估语义（原始单侧终端删、对称保留/排除终端候选、空值策略）；实现可复用的审计记录接口。

**📊 数据集**

217 条固定轨迹与 TraceElephant 额外的 138/176 任务/轨迹集合，涵盖两大异构语料库。

**📈 对比分析**

对比 PageRank 与 Earliest 的排名与效应：在原始单侧终端删族下效应为 -40.76~ -43.03，平均约 -7.7；在对称终端控制族下效应为 0，显示终端删导致分歧；排名在族内非确定，存在严格逆转。

**⚠️ 局限性**

仅揭示了单一语义分歧的局部证据，未覆盖所有可能的语义变化；样本仅来自两语料库，缺乏复现；Bootstrap 仅评估冻结重抽样，无法估计未列举语义族对结果的影响。

---

## 282. Beyond Document Retrieval: Architectural Challenges When LLM Agents Query Structured Enterprise Data

**arXiv ID:** 2608.19235 | [PDF](https://arxiv.org/pdf/2608.19235v1)

**作者:** Sheikh Nazib Ahmed `[一作]` `[通讯]` (University of Texas at Arlington), Sheikh Nazib Ahmed (University of Texas at Arlington)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在企业环境下，LLM 代理从文档检索向结构化数据查询转变时所需的架构改造，并提出了七维度的治理设计框架与参考架构，随后在合成基准上进行对照实验验证其有效性。

**💡 创新点**

创新点在于将文档RAG与结构化数据查询的差异抽象为检索语义、授权粒度、意图识别、实体解析、评估、失效模式和延迟七个维度，并基于此构建了分阶段治理的参考架构、失效模式分类以及可测评的阶段化指标，填补了现有研究的治理视角空白。

**🔧 技术方法**

使用技术包括：LLM 生成 SQL 或 API 调用（文本到 SQL），多阶段治理管控（意图解析、权限检查、实体绑定、源规划、答案验证、受控执行、解释与审计），以及合成数据集的自动化构建与实验脚本。

**📊 数据集**

实验数据集为自建的合成多源数据集，包含两种业务数据源（运营仓库与合规仓库）、四个访问角色和 21 个典型查询，所有数据均为人工生成并已知真值。

**📈 对比分析**

对比方法是把直接翻译‑执行的基线与加入分阶段治理的代理进行同一组查询的性能对照，结果显示在合成基准上，结果准确率从 0.43 提升至 0.95，完全消除授权违规，且延迟略低（约 0.351 ms vs 0.402 ms）。

**⚠️ 局限性**

局限性在于实验仅基于合成数据，缺乏真实企业数据的验证；架构依赖手工维护的 schema 注册表，未解决跨域查询、schema 演化、易懂的查询解释以及大规模评测等开放问题。

---

## 283. Answer-Level Trust Selection for Physical Vision-Language Reasoning

**arXiv ID:** 2608.19807 | [PDF](https://arxiv.org/pdf/2608.19807v1)

**作者:** Rongyu Yu `[一作]` (University of Edinburgh), Fengxiang He `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Answer‑Level Trust Selection（ATS），一种基于黑盒多条件查询的后处理框架，用于在视觉‑语言模型（VLM）执行定量物理推理任务时判断单个数值答案是否可信并决定是否接受。

**💡 创新点**

创新点在于整合八个可解释的行为诊断分数（视频‑先验一致性、对抗先验稳定性、采样稳定性、提示一致性、尺度合理性、类别熟悉度、CoT非拒绝性以及先验‑跟踪敏感性），实现无侵入、无内部信息依赖的答案可信度评估，能够捕捉自洽稳定错误和先验追踪等隐藏失效模式。

**🔧 技术方法**

使用技术包括：黑盒多条件查询（含视频+先验、先验单独、对抗先验、链式思维提示、随机采样自洽）、行为诊断分数计算、分数加权聚合（手工加权、均匀加权、学习型Logistic回归）、事件族条件阈值选择与覆盖率‑风险权衡。

**📊 数据集**

主要数据集为QuantiPhy视频定量推理基准，并在20个不同VLM骨干上进行跨模型评估。

**📈 对比分析**

与SC‑only、Generic UQ、Equal‑weight ATS、Learned ATS以及Oracle ranking等基线在覆盖率–MRA和AURC指标上进行对比。ATS在大多数覆盖率点保持竞争力，尤其在拒绝稳定错误和先验追踪案例时显著优于单纯自洽或通用不确定性估计；但整体提升有限，正确案例保留率有轻微下降。

**⚠️ 局限性**

局限性包括：仅为后处理，无法提升模型本身的数值准确率；在提高错误案例拒绝率的同时会牺牲部分正确答案的保留率，存在拒绝‑保留权衡；聚合权重需要人工设定，对极端尺度预测的鲁棒性仍有待提升。

---

## 284. Auditing and Decomposing Feedback-Driven Evolution in LLM Test Generation under the Oracle Problem

**arXiv ID:** 2608.19626 | [PDF](https://arxiv.org/pdf/2608.19626v1)

**作者:** Yunhao Liang `[一作]` (Chinese Academy of Sciences), Shiwen Ni `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对LLM生成的测试用例在反馈循环中的“oracle问题”进行系统审计，提出冻结选择后多实现审计、等候预算重采样、密度匹配安慰剂等方法，评估了基于执行反馈的自演化是否真的提升测试质量。

**💡 创新点**

创新点在于：①提出“冻结-再审计”协议，独立验证所有被选测试；②设计等候预算重采样与密度匹配安慰剂作为对照，清晰区分搜索、迭代结构与精细反馈的贡献；③将真实历史错误拆分为多类触发phenotype，结合人工语义审核，揭示无效输入是主要误报来源。

**🔧 技术方法**

技术方法包括：LLM调用（Qwen2.5/Qwen3.5）、跨折复合式多实现执行审计、等候预算随机重采样、密度匹配安慰剂、六方统计（kill率、任务检测、审核任务覆盖率、解析产出率、oracle精度）、多种实验设计（单次调用、单次+变异、三次独立调用、公共输入变异）以及人类语义审核流程。

**📊 数据集**

数据集为 TestCase‑Eval 的 500 项精确输出 Python 任务，其中包括 142 开发任务、114 外部锁定任务、138 资格修订后的留置任务，总共 394 任务；使用的模型为 Qwen 家族（Qwen2.5 与 Qwen3.5）。

**📈 对比分析**

对比方法：将基于单参考的进化（mutation）与等候预算重采样（resampling）进行 kill‑rate 与任务检测比较；将真实三轮反馈与密度匹配安慰剂以及纯等候重采样进行差异统计。实验结果显示：单参考极大膨胀 9.46–14.85% 的提升，审计后等候重采样优于变异进化 6.01–18.83%；三轮反馈对比安慰剂无显著优势，未达到预设的等价或优越阈值。

**⚠️ 局限性**

局限性包括：①oracle 仍基于已接受实现的内部一致性，无法保证与自然语言规范完全一致；②实验仅覆盖 Qwen 系列模型与 Python 任务，缺乏跨模型、跨语言、跨检查器的泛化验证；③部分实验（如 E16‑B）在模型调用后调整了资格门槛，导致验证不够严格；④人类审核者的主观性和有限样本仍可能影响结论；⑤未对多重检验进行严格校正，部分发现为探索性。

---

## 285. Quantifying Event Impacts on Time Series via Multiscale Contrastive Learning

**arXiv ID:** 2608.19447 | [PDF](https://arxiv.org/pdf/2608.19447v1)

**作者:** Yiming Sun `[一作]` (Rutgers University), Haifeng Chen `[通讯]` (NEC Laboratories America)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种名为EventTime的框架，用于在预事件市场历史和事件元数据的条件下估计网络安全披露后短期的异常财务损失。

**💡 创新点**

创新点包括：① 多尺度时间序列编码结合短期与长期市场背景；② 双层事件-时间注意力融合模块，将事件属性映射到时间序列并定位最相关的预事件时刻；③ 动态对比学习，利用事件与时间序列的相似性构造正负样本，强化稀缺事件的表示。

**🔧 技术方法**

使用的技术主要有Transformer/GRU编码器、多尺度聚合、注意力机制、动态对比学习（InfoNCE）以及事件元数据的多层感知机编码。

**📊 数据集**

主要数据集为SECURE，包含268家公司2005‑2025年间约1128个事件的股票超额收益和网络安全事件信息；此外在跨域实验中使用CAMELS降水与流量数据。

**📈 对比分析**

与线性、混合器、Autoformer、PatchTST、iTransformer、Pyraformer、TimeXer、Text2Timeseries、EventTSF等基线相比，EventTime在SECURE上R²提升至0.524（相较于基线的0.32以上），MAE和RMSE均显著下降，IC与Rank‑IC也提升；在CAMELS实验中亦实现了最高R² 0.392。

**⚠️ 局限性**

局限性包括：① 事件样本极为稀疏，仍受限于可用事件与市场匹配的数量；② 对事件元数据的依赖程度高，缺失或错误的元数据会影响性能；③ 主要聚焦于单一事件类型（网络安全披露）与金融时序，跨领域推广仍需进一步验证。

---

## 286. Transformer Models for Text Summarization: A Comparative Study of BART, BERT, and RoBERTa

**arXiv ID:** 2608.19200 | [PDF](https://arxiv.org/pdf/2608.19200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 287. Lexicographic Combination of Reduction Pairs (Extended Version)

**arXiv ID:** 2608.19683 | [PDF](https://arxiv.org/pdf/2608.19683v1)

**作者:** Teppei Saito `[一作]` (JAIST), Nao Hirokawa `[通讯]` (JAIST)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于可单调与不变位置的简洁可组合性判据，用于将多个约简对按字典序组合，从而构造更强的终止性判别工具，并引入了秩序化的阶梯矩阵解释（echelon‑form matrix interpretation）来进一步扩展传统矩阵解释。

**💡 创新点**

创新点包括：①给出了可组合性的单一判据，既覆盖传统的线性多项式解释，又适用于非线性矩阵解释；②揭示了使用字典序而非成分式的矩阵解释如何成为有效的约简对；③将该判据与相对终止性框架结合，解决了可用规则判据在相对终止性中的局限。

**🔧 技术方法**

核心技术包括：依赖对框架（dependency pair framework）；线性多项式解释、矩阵解释与Knuth–Bendix顺序的构造；引入可单调/不变位置概念以判定组合可行性；阶梯矩阵解释（echelon‑form）与其严格单调性证明；以及使用SMT求解器Z3进行约束求解。

**📊 数据集**

实验使用了 Termination Problem Database（TPDB）中的 1528 个普通终止问题和 57 个相对终止问题，全部通过自研工具进行测试。

**📈 对比分析**

与现有方法比较时，阶梯矩阵解释 E_d 在普通终止中明显优于线性多项式组合 E_1E_1；组合方法（如 E_1L、LE_1）能额外证明 8–11 个传统工具（AProVE、EPC 等）未能处理的问题；整体而言，所有方法联合可达到 649 条终止证明，超过传统工具的 649 条。

**⚠️ 局限性**

局限性包括：判据要求约简对满足“normal”性质（> ⊆ ≥），不适用于非 normal 的约简对；在相对终止性中仍需满足可用规则判据的最小性条件；阶梯矩阵解释在构造时需满足列阶梯形式，限制了可选矩阵的自由度。

---

## 288. Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW

**arXiv ID:** 2608.19762 | [PDF](https://arxiv.org/pdf/2608.19762v1)

**作者:** Kang Liu `[一作]` (Xi'an Jiaotong University), Suyan Li `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过有限时域输入-状态-输出（ISO）框架，研究 AdamW 中微批量梯度扰动的延迟影响，并使用成对轨迹对比来揭示扰动如何写入优化器状态、在后续更新中传播并最终反映在损失上。

**💡 创新点**

创新点在于：①提出了带符号、时域分辨率的 ISO 响应算子；②将响应分解为写入、传播和读取三个阶段；③给出多步误差分解并证明一阶准确性；④通过重复未来实验揭示延迟影响的前瞻结构。

**🔧 技术方法**

技术方法包括：ISO 线性化（切线模型）、成对轨迹实验、梯度、参数写入、动量状态写入的写入算子、状态传播矩阵、损失读取向量、误差分析（平滑和分段平滑网络）以及统计指标（NRMSE、符号一致率、Spearman 相关）等。

**📊 数据集**

实验数据集：人工控制的二次系统、CIFAR-10 上的 94k 参数 CNN‑ReLU 与 855k 参数 MLP‑GELU、以及 Pythia-410M/1B/1.4B 语言模型在 WikiText‑103、OpenWebText、CodeParrot 上的训练。

**📈 对比分析**

对比方法：ISO 预测值与实际轨迹的 NRMSE、符号一致率、Spearman 相关；与单步响应、梯度范数、参数写入范数、冻结 ISO 等方法比较。结果显示 ISO 在 0.04–0.1 的 NRMSE、>0.99 的符号一致率以及 0.6–0.9 的 Spearman 相关，明显优于单步或简单统计量。

**⚠️ 局限性**

局限性：ISO 本质上是路径依赖的，易受非线性、激活切换和未来动态漂移影响；近似误差在较长时域内可能被放大；对前瞻可识别性的条件尚未完全阐明，导致在某些系统（如控制二次系统）下冻结 ISO 的预测效果有限。

---

## 289. An Inclusive and Lightweight Approach to Federated Continual Learning for Cultural Heritage

**arXiv ID:** 2608.20038 | [PDF](https://arxiv.org/pdf/2608.20038v1)

**作者:** Ioannis Theologitis `[一作]` (Information Technologies Institute Centre For Research And Technology Hellas), Konstantinos Votis `[通讯]` (Information Technologies Institute Centre For Research And Technology Hellas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级、无样本记忆的联邦持续学习框架FedCurv-DR，用于文化遗产图像分类。

**💡 创新点**

创新点在于将FedCurv与EWC-DR参数重要性估计相结合，加入经验累计与衰减，并通过设置更新间隔来显著降低通信与计算开销。

**🔧 技术方法**

使用了联邦学习、持续学习中的正则化方法（EWC-DR）、参数重要性聚合与衰减、EfficientNet‑B0骨干网络、Flower框架、Avalanche持续学习库以及CodeCarbon能耗追踪工具。

**📊 数据集**

实验基于WikiArt艺术图像数据集，在四个连续经验（Realism、Expressionism、Symbolism、Naive Art Primitivism）构成的领域增量场景下进行。

**📈 对比分析**

与FedAvg和原FedCurv（EWC/EWC‑DR）对比，FedCurv‑DR在整体准确率上最高达66.92%，显著提升BWT（-6.92pp），降低公平性差异（2.89pp），并在更新间隔为5时能耗仅为0.0334kWh，略低于其他方法。

**⚠️ 局限性**

实验仅在三模拟客户端的单机环境下进行，缺乏真实机构部署规模与治理、隐私保护等实际约束，且使用公开数据的合成划分，未验证在真实分布式环境中的效果。

---

## 290. Mechanistic Tomography: Designed Measurement for Control-Oriented Interpretability

**arXiv ID:** 2608.19338 | [PDF](https://arxiv.org/pdf/2608.19338v1)

**作者:** Vijay Erramilli `[一作]` `[通讯]`, Vijay Erramilli

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了“机制测绘（mechanistic tomography）”框架，将不同解释方法（如坐标补丁、梯度补丁、子集干预、Hessian-向量积、提升测量）统一为线性测量问题 y = Ax + w，进一步给出了测量设计、校准维度、误差分析和控制验证的系统流程。

**💡 创新点**

创新点在于：①将多种解释技术映射为同一测量模型，揭示它们共享的结构；②提出校准维度概念，用来判断是否需要加入交互或高阶测量；③通过控制循环验证观察者对下游任务的实际影响；④给出按测量成本从低到高的实践步骤，提供基于残差的停止规则。

**🔧 技术方法**

使用线性逆问题理论、稀疏重建（OMP、L1 正则化）、梯度/Hessian-向量积、前向/后向干预、Tracr 编译器、以及强化学习式的控制验证和目标误差监测。

**📊 数据集**

实验数据集包括：①基于两隐藏马尔可夫模型（HMM）的人工数据；②GPT‑2‑small 的 IOI（间接对象识别）任务；③Qwen‑2.5‑7B 的拒绝‑合规边界读数；④Tracr 生成的程序合成数据，用于基座变换验证。

**📈 对比分析**

方法比较通过：①在 HMM 控制实验中，观察者误差与闭环目标误差相关；②在聚合测量实验中，稀疏 OMP 在 12 次测量内达 0.989 的 Pearson r；③在梯度+校准实验中，单标量增益将 AtP 的 R² 从 0.818 提升至 0.960；④在交互恢复实验中，lifted OMP 在 64–72 次测量内实现 0.95 的 held‑out R²；⑤在 Qwen‑2.5‑7B 上，校准的加法映射与提升的 pairwise 映射在 held‑out R²（0.9829 vs 0.9835）差异不显著。

**⚠️ 局限性**

局限性包括：①依赖预设的坐标基底，无法自动推断最优基底；②校准维度为经验阈值，未给出理论最优值；③交互恢复假设稀疏性，实测机制可能更复杂；④控制验证仅在固定控制器/执行器下进行，未覆盖所有下游任务；⑤在大模型行为评估中，仅检验预测准确性，未证明真正的机制恢复。

---

## 291. Block3D: Efficient Text-to-3D Generation via Block-Wise Diffusion

**arXiv ID:** 2608.19567 | [PDF](https://arxiv.org/pdf/2608.19567v1)

**作者:** Bowen Cui `[一作]` (Zhejiang University), Bohan Zhuang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出Block3D，一种块级扩散框架用于从文本生成高质量3D资产，并显著降低生成时间。

**💡 创新点**

将自回归依赖从单个形状码迁移到块级，结合置信引导的块内纠错，实现并行去噪。

**🔧 技术方法**

块级扩散（Block Diffusion）、Cube VQ‑AE、CLIP文本编码、置信引导的M2T/T2T编辑、CFG指导等技术。

**📊 数据集**

使用TRELLIS‑500K数据集（300K训练样本，100K评估样本）。

**📈 对比分析**

与Cube、ShapeLLM‑Omni、TRELLIS‑text、AR3D‑R1等基线对比，Block3D在几何指标（Chamfer‑L1、Normal Consistency、F@1%）上最优，生成时间从25.71 s降至4.99 s，速度提升5.15×。

**⚠️ 局限性**

块级方法无法跨块纠错，仍存在曝光偏差；较大块会导致几何质量下降，且缺乏跨块细化机制。

---

## 292. In Two Minds about Lifelong Learning: Exploring Hemispheric Redundancy and Specialisation in Neural Models

**arXiv ID:** 2608.19514 | [PDF](https://arxiv.org/pdf/2608.19514v1)

**作者:** Benjamin Smith `[一作]` (Monash University), Gideon Kowadlo `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 4MAS（4 Module Awake/Sleep）宏架构，通过双半球结构、经验重放、睡眠阶段的交叉巩固实现持续学习；

**💡 创新点**

创新点在于将生物学记忆整合机制（REM 睡眠、双侧半球、经验重放）映射为机器学习架构，并通过不同温度的生成器实现稳定-可塑性分化；

**🔧 技术方法**

使用的技术包括变分自编码器（VAE）作为生成式长短期记忆、基于置信度的样本挑选（CCS、LSCC）、睡眠学习率调节、双侧置信度路由等；

**📊 数据集**

评估数据集为 Split-MNIST、Split-Fashion-MNIST、Split-CIFAR-100；

**📈 对比分析**

与基线方法（B-IR、GR、Fine-Tuning、EWC 等）在 Class-IL 场景下进行对比，4MAS 在 Split-MNIST 和 Split-Fashion-MNIST 上取得接近 joint‑training 的准确率（98.3%/84.9%），在 Split-CIFAR-100 上略高于单侧 B-IR（29.29% 对 27.85%），并显著降低了记忆漂移和遗忘率；

**⚠️ 局限性**

局限性包括：采用非卷积 VAE 生成器对复杂图像的表达有限；缺乏对更长任务序列和更大规模模型的验证；未结合正则化技术进一步提升稳定性；对睡眠阶段的参数调优和生物学细节的完整性仍待深入。

---

## 293. Learning to Beat: Phenotype-Guided Latent Flow with Regional Motion Priors for Biventricular Motion Synthesis

**arXiv ID:** 2608.19738 | [PDF](https://arxiv.org/pdf/2608.19738v1)

**作者:** Xuan Yang `[一作]` (National University of Singapore), Lei Li `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

基于单帧ED网格，提出一种区域特异、表型自适应的潜在生成框架，用以合成完整心室周期运动。

**💡 创新点**

创新点包括：利用运动感知功能分区构建区域先验、区域结构化VAE、基于表型的修正流生成器、原型路由适配器以及可选的运动控制分支。

**🔧 技术方法**

核心技术涵盖：基于K-means的运动分区、MRAB多跳区域注意力、运动VAE、Rectified Flow（RF）潜在生成、AdaLN与FiLM调制、原型路由等。

**📊 数据集**

使用公开心血管MRI数据集ACDC、M&Ms与M&Ms-2进行训练与评估。

**📈 对比分析**

在ED条件下与CVAE、ACTOR、Action2Motion、CHeart、MeshHeart、4DCardioSynth、RePCM等方法比较，最终在ASSD、HD95、vRMSE指标上分别为1.49 mm、3.77 mm、3.31 mm，明显优于对手，尤其在右心室运动上提升显著。

**⚠️ 局限性**

局限性包括：参考序列受SSM拟合误差限制、仅处理统一拓扑的双心室表面、仅针对四大疾病群、无法单凭ED网格完全预测功能属性，且对罕见或复杂先天性畸形的泛化仍需验证。

---

## 294. Tracking the Trend in How Speech Synthesizers Deceive People

**arXiv ID:** 2608.19959 | [PDF](https://arxiv.org/pdf/2608.19959v1)

**作者:** Milan Šalko `[一作]` (Brno University of Technology), Jakub Reš `[通讯]` (Brno University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对IT专业人士进行问卷，评估三种深度伪造工具（RTVC 2019、YourTTS 2022、ElevenLabs 2024）在全合成与部分伪造情境下的听觉检测能力，并与六个预训练检测器对比。

**💡 创新点**

引入部分伪造实验，发现单句伪造比全合成更难识别；证明现代商业合成工具（ElevenLabs）在人类与检测器均低效。

**🔧 技术方法**

采用人类听觉评估、Krippendorff α、d'分析、F1、All OK 等统计方法，并对六个基于XLS‑R/WavLM 前端与 AASIST/MHFA 后端的深度伪造检测器进行评估。

**📊 数据集**

生成的语料来自 ChatGPT‑4 编写的对话文本，使用公开的 YouTube 访谈录音进行零射声克隆；实验样本共 9 位明星，7 个录音/4 句。

**📈 对比分析**

结果显示，RTVC/YourTTS 人类 F1≈0.90，ElevenLabs F1 仅 0.48；部分伪造下 F1 降至 0.29，All OK <10%；检测器在全合成 ElevenLabs 下 F1 仅 0.09，但在部分伪造时略高。

**⚠️ 局限性**

样本仅限 IT 专业人士，未包含无警告对照组，短期高伪造率设计，未检验反向误判比例，对检测器阈值设定不具可调性，且仅评估单一每期工具，未能验证时间趋势。

---

## 295. PersonalBench: Measuring the Authorship Gap in LLM Personalization

**arXiv ID:** 2608.19746 | [PDF](https://arxiv.org/pdf/2608.19746v1)

**作者:** Yash Ganpat Sawant `[一作]` `[通讯]` (Independent AI Researcher), Yash Ganpat Sawant (Independent AI Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于作者身份验证的个性化文本生成评测基准，评估推理时个性化方法是否真正使模型写作更像目标作者；

**💡 创新点**

首次将作者身份验证模型LUAR与LLM-as-judge及传统stylometrics相结合，构建多维度评估框架，揭示现有推理时个性化方法无法跨越人类写作与LLM写作的差距；

**🔧 技术方法**

采用LUAR（对比学习的作者身份嵌入）、GLM-4 32B作为评判者、Qwen 3 32B 4-bit生成器以及功能词、标点、ROUGE-L等stylometrics；

**📊 数据集**

使用Blog Authorship Corpus（19,320位博主的681K篇帖子），挑选50名作者进行测试；

**📈 对比分析**

比较了四种推理时个性化方法（无个性化、少样本、抽象档案、对比特征），在LUAR上四种方法的平均相似度为0.484–0.508，低于跨作者人类文本下限0.626，显示差距未被缩小；

**⚠️ 局限性**

限制包括仅使用两类32B 4-bit LLM模型、单一博文域数据、未验证LLM评判者与人工评判的一致性，以及对多语言和更大模型的泛化未知。

---

## 296. Point-Based 3D Reconstruction from Sparse Views under Known Illumination

**arXiv ID:** 2608.20000 | [PDF](https://arxiv.org/pdf/2608.20000v1)

**作者:** Magnus Kaufmann Gjerde `[一作]` (Aalborg University), Thomas B. Moeslund `[通讯]` (Aalborg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于可微分光线传输的点渲染方法，使用带有可调透明度的beta表面元（beta surfels）来重建稀疏视角下的三维几何。

**💡 创新点**

创新点在于：1）将透明度显式包含在光传输方程中并通过附加的衰减项显式求导；2）采用可调beta核的表面元，避免了传统高斯核的固定半径限制；3）在已知光源的条件下直接利用光照对几何进行约束，实现仅使用数百个表面元即可达到高精度重建。

**🔧 技术方法**

核心技术包括：可微分光线传输（adjoint 传输），Beta 核表面元建模，Monte Carlo 路径采样，光照下的显式衰减求导，以及深度/法线一致性和弱透明度正则化。

**📊 数据集**

实验使用了五个合成对象（Teapot、LEGO、Dragon、Horse、Plant）在十张已知相机参数和点光源的训练图像上进行重建。

**📈 对比分析**

与2DGS、SuGaR、RadiosityGS、GOF、NeuS、NeuS2、GeoSVR等基线对比，本文方法在平均对称 Chamfer 距离上最低，仅使用约267个表面元，远低于几千到几十万的传统点基方法；在方向 Chamfer 评估中也显示了最佳或竞争性的准确率和完整度。

**⚠️ 局限性**

局限性包括：仅支持 Lambertian 表面和直接点光照；仅在合成场景下验证，未处理间接照明、非漫反射、真实图像捕获以及更复杂场景的自适应细化。

---

## 297. A Thread-Register Decoupled GPU Execution Model for Efficient Tensor Computation

**arXiv ID:** 2608.19628 | [PDF](https://arxiv.org/pdf/2608.19628v1)

**作者:** Zihan Liu `[一作]`, Junsong Wang `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 298. Dynamic Gated Cross-Modal Fusion with Sarcastic-aware Contrastive Regularization for Multimodal Sarcasm Detection

**arXiv ID:** 2608.19942 | [PDF](https://arxiv.org/pdf/2608.19942v1)

**作者:** Hao Guo `[一作]` (Anhui Polytechnic University), Chao Kong `[通讯]` (Anhui Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种动态门控跨模态融合与讽刺感知对比正则化的多模态讽刺检测框架。

**💡 创新点**

通过实例级动态门控跨模态注意与标签感知对比正则化，解决模态贡献不一致和表面一致性误导问题。

**🔧 技术方法**

使用CLIP预训练视觉语言模型，双向门控注意、动态融合门、标签感知对比损失，联合多目标训练。

**📊 数据集**

在MMSD与其改进版MMSD2.0数据集上进行实验。

**📈 对比分析**

与多种文本、图像、跨模态基线及大模型对比，均在两数据集上获得最高F1与整体性能提升，尤其在MMSD2.0上表现尤为突出。

**⚠️ 局限性**

仍易受跨模态对齐误导，缺乏更细粒度视觉-文本对齐与对讽刺微妙语义的捕捉。

---

## 299. Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals

**arXiv ID:** 2608.19665 | [PDF](https://arxiv.org/pdf/2608.19665v1)

**作者:** Kyungho Kim `[一作]` (KAIST), Kijung Shin `[通讯]` (KAIST)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种训练-free的LLM推荐框架CoRRe，先用LLM生成用户意图，再通过共购图和流行度对项嵌入进行方向和幅度细化，最终匹配生成推荐。

**💡 创新点**

创新点在于post-LLM的协同细化思路：将协同过滤信号嵌入到LLM生成的项向量中，而非传统的pre-LLM候选重排或RAG增强。

**🔧 技术方法**

技术包括LLM文本生成（GPT‑5.2）、文本编码（text‑embedding‑3‑large）、共购图传播、向量归一化与幅度调节以及基于点积的检索。

**📊 数据集**

使用Amazon Reviews公开数据集的Sports、Toys与Beauty三大领域。

**📈 对比分析**

与现有训练‑free和训练式推荐器对比，CoRRe在Sports和Toys上取得最佳或第二佳的H@10/N@10结果，整体性能优于所有训练‑free基线，且竞争或优于多数训练式方法。

**⚠️ 局限性**

局限性包括对LLM提示质量敏感、共购图稀疏时细化效果有限、缺乏针对冷启动用户的策略，以及未充分利用结构化属性或多模态信息。

---

## 300. RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations

**arXiv ID:** 2608.19735 | [PDF](https://arxiv.org/pdf/2608.19735v1)

**作者:** En Zhi Tan `[一作]` (SAP SE), Benjamin Yan Han Yap `[通讯]` (SAP SE)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 RecPFN，一种先验拟合网络，在预训练的合成点击流环境中学习到的全局序列学习算法，能在推理时仅通过少量支持序列实现单次前向推断的零样本推荐。

**💡 创新点**

创新点在于：①使用结构因果模型生成多样化的合成环境作为预训练先验；②采用轻量级交替 Type‑A（复制）与 Type‑B（特征提取）注意力块的 decoder‑only transformer 以实现高效的 in‑context 学习；③将 Bayesian 推理与 PFN 的概念结合，实现在新域下无需权重更新即可进行适配。

**🔧 技术方法**

技术包括：合成数据生成的随机图和潜在因子先验；Transformer 的交叉注意力与硬 alibi 掩码；在上下文检索中使用 recency‑weighted 交集相似度；以及使用单个 GPU 预训练 36 小时。

**📊 数据集**

使用八个公开的序列推荐基准（Amazon 8 类、Yelp、Dianping），以及四个 Amazon 领域的真实交互数据做 ablation；合成数据来自 BEIR 语料库的句子嵌入。

**📈 对比分析**

与传统 ID‑基模型（GRU4Rec、SASRec、BERT4Rec）、Embedding‑基模型（FDSA）以及预训练零样本方法（RecGPT、UniSRec、VQ‑Rec、RecFormer）比较，RecPFN 在 7/8 个数据集上实现了最优的零样本 HR@10/MRR@10，且在低计算、低数据环境下仍能与全量监督模型竞争；推理速度在单前向推断的前提下仍保持可接受。

**⚠️ 局限性**

局限性包括：对上下文检索质量高度敏感，支持集规模或检索噪声大时性能显著下降；对大规模或长上下文的适配效果不佳；合成先验对真实环境的覆盖仍有限，导致极端参数或动态变化时的鲁棒性下降；以及嵌入质量对最终召回有显著影响。

---

## 301. Designing Human-mediated AI Guidance: Ready Together for Personalized Family Emergency Preparedness

**arXiv ID:** 2608.19950 | [PDF](https://arxiv.org/pdf/2608.19950v1)

**作者:** Nini Kurashvili `[一作]` (University of Milano-Bicocca), Dimitri Ognibene `[通讯]` (University of Milano-Bicocca)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出人类中介的 AI 指导框架，并通过设计并评估“Ready Together”系统——一款面向父母为儿童提供个性化应急预备指导的原型，探讨父母在 AI 与儿童之间的中介角色。

**💡 创新点**

创新点在于把传统的人机交互模式从单向（人直接使用 AI 输出）拓展为三方模型：AI、情境更熟悉的人类中介、以及最终信息接收者。该框架强调中介在提供情境、评估、适配与情感调节 AI 生成内容方面的主动作用，兼顾责任与信任。

**🔧 技术方法**

技术实现包括：使用 OpenAI Custom GPT（ChatGPT）生成基于父母提供的家庭/孩子信息的个性化说明、建议、活动与检查清单；前端通过 Figma 原型展示交互流程；采用定性研究、半结构访谈、共创工作坊收集需求；Pilot 评估采用 5 点量表与开放式问答；Nielsen 可用性评估评估原型可用性。

**📊 数据集**

研究未使用公开数据集，主要利用 6 位父母（来自乌克兰、意大利、西班牙、瑞典、俄罗斯）在访谈和共创环节提供的家庭情境信息作为实验输入。

**📈 对比分析**

性能评估采用主观量表（如：建议与家庭情境匹配度、实用性、清晰度等）与开放式反馈。结果显示平均分均在 4.0–4.5 之间，表明父母认为 AI 指导易懂、相关且有帮助；未进行客观指标或对照实验，仅为探索性验证。

**⚠️ 局限性**

局限性包括：样本规模仅 6 位父母，且同一组父母参与需求与评估阶段；仅评估父母对 AI 输出的感知，未测量儿童理解或行为结果；缺乏跨领域或更大规模验证；系统仍为原型，未实现完整集成与长期跟踪。

---

## 302. Mapping General-Purpose AI Governance in Twenty AI Middle-Power Jurisdictions

**arXiv ID:** 2608.19278 | [PDF](https://arxiv.org/pdf/2608.19278v1)

**作者:** Josephine Schwab `[一作]` (Arcadia Impact AI Governance Taskforce), Caio Vieira Machado `[通讯]` (Future Society)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 20 个 AI 中等强国（含欧盟）在 4 个关键治理领域（系统风险评估、评估与验证、禁止与监测、重大事件报告）中的 GPAI 具体条文进行条款级别的对比与映射，揭示了各国治理趋同与力度差异。

**💡 创新点**

创新点在于：①采用条款级别而非机构或政策层面进行细粒度对比；②首次记录并比较“已确认缺失”与“正向条文”；③系统性识别 6 维 GPAI 定义与 4 维治理领域，将国内治理差距与国际红线标准对齐。

**🔧 技术方法**

技术方法主要是基于文档检索与结构化编码的法律文本分析：使用统一的条款搜索词、八步决策流程进行范围测试，并通过受控词汇表进行条款特征标注；此外，采用自动化数据校验与人工验证确保结果可靠。

**📊 数据集**

数据集包括 20 个司法辖区（澳大利亚、巴西、加拿大、智利、法国、德国、印度、以色列、日本、肯尼亚、尼日利亚、秘鲁、新加坡、南非、韩国、瑞士、台湾、阿联酋、英国，以及欧盟的硬法、软法与战略文件），共计 101 份可测条款记录，涵盖 4 维治理领域与 6 维 GPAI 定义。

**📈 对比分析**

比较方法：为每个司法辖区在每个治理领域选取最具法律效力的条款（硬法>软法>战略>草案>未确定），以此构建矩阵；对比条款覆盖率、力度（是否绑定、是否明确受影响主体、是否有执法机制）与国际红线标准。性能表现为：16/20 司法辖区具备至少 3/4 治理领域条款，但仅 22% 的条款为强制性硬法；约 75% 的硬法未对 GPAI 做出定义；大多数评估与验证主体为自身机构，缺乏对开发者的强制约束。

**⚠️ 局限性**

局限性包括：①仅聚焦 4 个治理领域，未覆盖所有可能的 GPAI 监管维度；②数据截至 2026‑07‑30，后续立法可能更改；③对条款的解读受语言与翻译精度影响；④缺乏对条款执行与实际影响的实证检验；⑤在评估与验证中，多数条款为自评或指导，未体现强制执行；⑥未将国防与安全等排除领域的监管纳入分析，导致完整性受限。

---

## 303. RIPE++: Reinforced Keypoint Learning from Positive Pairs Only

**arXiv ID:** 2608.19693 | [PDF](https://arxiv.org/pdf/2608.19693v1)

**作者:** Johannes Künzel `[一作]` (Fraunhofer Heinrich-Hertz-Institute), Anna Hilsmann `[通讯]` (Fraunhofer Heinrich-Hertz-Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种弱监督的关键点提取与匹配框架，利用强化学习构造基于正样本几何一致性的奖励，仅使用同场景的图像对进行训练，消除负样本和几何标注需求。

**💡 创新点**

创新点在于（1）利用正样本的内外点统计直接生成奖励与惩罚，避免负样本；（2）将奖励扩展到匹配阶段，训练轻量级Transformer LightGlue；（3）通过熵正则化提升热图定位精度；（4）引入距离感知奖励提升梯度稳定性。

**🔧 技术方法**

采用强化学习（REINFORCE）与几何一致性奖励，基于Fundamental矩阵与RANSAC；使用对比/InfoNCE损失优化描述子；Transformer架构（LightGlue）进行匹配；熵正则化与梯度下降（AdamW）等技术。

**📊 数据集**

主要使用MegaDepth（1500子集）进行关键点与匹配评估；SCARED1500（内镜视频）用于医学域实验；Aachen v1.1、Tokyo24/7用于视觉定位实验；并与SuperGlue、DISK、DeDoDe、RaCo、DaD等公开数据集进行对比。

**📈 对比分析**

在MegaDepth1500上与RIPE+LightGlue对比，AUC@5°提升至59.65（+3.9pp）；在SCARED1500上相较基线提升显著；在Aachen v1.1夜间条件下AUC@0.5/5°提升9.5pp；整体性能接近全监督方法，显著优于同类弱监督方案。

**⚠️ 局限性**

局限性包括：仍无法完全逼近全监督匹配器性能；奖励信号受RANSAC估计误差影响，导致梯度噪声；对极端视角或低纹理区域的鲁棒性不足；训练仍需大量正样本，难以在极稀缺数据场景中直接使用。

---

## 304. Active Inference as Context Acquisition for AI Agents

**arXiv ID:** 2608.19202 | [PDF](https://arxiv.org/pdf/2608.19202v1)

**作者:** Sanchayan Dutta `[一作]` (University of California), Suvrit Sra `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将主动推理（active inference）作为 AI 代理在交互中获取上下文的通用层，阐述了通过信息增益驱动的询问与行动决策框架。

**💡 创新点**

创新点包括：①将上下文获取建模为一阶贝叶斯推理与外层动作选择的双层优化；②设计了可精确评估的“Optimal Question Asking (OQA)”基准，并提供动态规划（DP）最优策略做对照；③在 token‑budget 的提示生成与提示优化两类应用中验证该框架。

**🔧 技术方法**

核心技术为：主动推理与期望自由能（expected free energy）公式、信息增益与互信息的等价推导、动态规划最优询问策略、贝叶斯实验设计、Beta 先验与 Thompson 采样等统计学习方法。

**📊 数据集**

使用的数据集主要有：三张真实属性表（Places、Cars、Animals）、三组 100/200/300 项的合成多值属性表，以及 48 条合成产品描述任务和 ARC‑Challenge 题库，用于提示优化。

**📈 对比分析**

与方法比较时，模型在 OQA 中的“规划间隙”（questions excess vs DP oracle）普遍高于最优策略，显示仍有改进空间；在提示生成实验中，定向澄清策略提升了验证通过率，且每千 token 的信息增益在不同策略间差异显著。

**⚠️ 局限性**

局限性包括：①基准采用噪声自由、固定表格与确定性答案，缺乏开放式、模态多样的真实交互；②未使用工具或记忆，难以评估复杂工具调用场景；③DP 先行者在大规模任务上难以扩展，需近似或采样方法；④安全与隐私风险未深入探讨。

---

## 305. Interrupting the Loop: Periodic Subject Changes Raise Judged Surprise and Connection in Base Language Models

**arXiv ID:** 2608.19893 | [PDF](https://arxiv.org/pdf/2608.19893v1)

**作者:** Roberto I. Ono Filho `[一作]` `[通讯]` (Independent researcher), Roberto I. Ono Filho (Independent researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在无任务情况下基础语言模型如何产生新颖文本，并通过对循环生成、采样器、提示三者进行系统拆解，发现主要来自于循环中的停顿（interruption）与习惯化（habituation）两种操作。

**💡 创新点**

创新点在于：① 将干扰（插入新主题句子）与习惯化简化为最小有效操作，证明其能显著提升短窗口的惊喜、关联与连贯；② 发现窗口化LLM评判会误读插入句子、回放过去片段以及无法捕捉全局整合，提出生成仅窗口、全新窗口、文档级评判等校正方法。

**🔧 技术方法**

技术包括：抗概率采样器（entropy band + anti-probable push）、窗口化LLM评判（Claude Opus/Claude Sonnet/Kimi K2.6），干扰与习惯化控制，量化评估（bootstrap、sign‑flip permutation、Cliff's δ），以及残差流几何分析。

**📊 数据集**

数据集：10个叙事开篇句子（premises），10个阐述式开篇句子用于另一次复制，10个新叙事开篇用于预注册复制；对比实验采用三种基模型（Qwen3-30B-A3B、Qwen3-8B、OLMo-2-13B），并在二次验证中使用预训练版 Qwen3-8B。

**📈 对比分析**

比较方法：将干扰+习惯化、干扰单独、习惯化单独等不同条件放入 2×2 因子电池；用配对 t‑检验（或精确符号置换）和 Cliff's δ 评估差异；对文档级效果使用单次评判。结果显示干扰在新文本窗口中惊喜提升约1–1.5分，关联提升约0.8–2分；但在完整文档层面并未产生连贯性提升，说明仅为局部变化。与传统采样器或提示对比无显著差异。

**⚠️ 局限性**

局限性包括：① 仅在无任务、强制续写环境下验证，无法说明在对话或有目标任务中的表现；② 评判者为 LLM，虽然交叉验证与人类一致性有限；③ 只关注窗口级的惊喜、关联、连贯，未测量创意的价值、原创性或对外部世界的影响；④ 采样器、提示与循环三者未在统一实验下交叉验证，缺乏完整因子析因；⑤ 只在 8–30B 模型上测试，规模与量化影响未知。

---

## 306. Beyond Imitation: Filtering On-Policy Distillation by Reasoning Progress

**arXiv ID:** 2608.19408 | [PDF](https://arxiv.org/pdf/2608.19408v1)

**作者:** Chen Yang `[一作]` (Hong Kong University of Science and Technology), Danny H. K. Tsang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 on‑policy distillation (OPD) 中加入过程奖励估计，检测教师监督与推理进展的秩冲突，并对冲突段落进行屏蔽，从而提升推理模型的性能。

**💡 创新点**

创新点在于：①使用教师无关的过程奖励作为推理进展的参考；②通过同符号进度合并形成更稳定的单元；③仅在教师监督与进展排序冲突时才屏蔽对应的 token 级监督，避免了统一的监督误导。

**🔧 技术方法**

核心技术包括：反向 KL‑OPD、Monte Carlo roll‑out 计算过程奖励、同符号段合并、秩冲突检测与不一致分数累积、token 层级掩码与重规范化。

**📊 数据集**

在 DeepSeek‑R1‑Distill‑Qwen‑1.5B/JustRL‑1.5B 及 Qwen3‑1.7B/e3‑1.7B 这两组模型上使用 DAPO‑Math‑17K 训练集，评估数据集为 AIME 2024、AIME 2025 与 OlympiadBench。

**📈 对比分析**

与标准 OPD 以及 E‑OPD、TIP‑OPD、IW‑OPD、Uni‑OPD 等四种改进方法比较，平均 pass@4 从 OPD 的 45.70% 提升到 51.83%，在 AIME 2024/2025 上分别提高 2.51/2.16 点，整体表现最为显著。

**⚠️ 局限性**

主要局限包括：①需要多次 roll‑out 计算过程奖励，计算成本高且方差大；②在长文本或低资源情形下效果受限；③在 OlympiadBench 上提升有限；④对更大规模模型的推广和跨域通用性仍需进一步验证。

---

## 307. Polyomino Nets Covering Three Different Boxes of Area 106 and Related Results

**arXiv ID:** 2608.19910 | [PDF](https://arxiv.org/pdf/2608.19910v1)

**作者:** Erik D. Demaine `[一作]` (Massachusetts Institute of Technology), Hanyu Alice Zhang `[通讯]` (Cornell University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过计算搜索，发现了面积仅为106的多面体网，可折叠成三种不同尺寸的长方体，推断出新的最小面积三方体网，并提出了无穷族网的生成方法。

**💡 创新点**

创新点包括：①提出“堆叠网”与“简单相位网”两种新型网结构；②利用七状态矩阵和特征值分析实现快速计数；③设计BFS迭代算法并与Redelmeier算法关联，极大提升枚举效率；④给出关于条纹网的猜想与公式，揭示尺寸关系的深层结构。

**🔧 技术方法**

使用了BFS迭代搜索、Redelmeier变体、矩阵幂运算（七状态矩阵）以及条纹化的折叠模型，辅以Python实现的深度优先搜索和剪枝策略。

**📊 数据集**

主要数据来源为自建的网格搜索结果（面积≤106的网），并上传至Zenodo；未使用公开数据集，而是通过完整枚举生成的所有合法网格集合。

**📈 对比分析**

与2025年Qian等人基于SAT求解器的结果对比，发现我们的BFS算法在搜索速度上约为3–10倍慢，但能得到不同的、非标准的网格解；在已知面积46或以下的案例中结果完全一致，证明方法正确性。

**⚠️ 局限性**

主要局限在于：①对更大面积或更复杂尺寸的网仍需大量计算，难以实现完全枚举；②简单相位网的矩阵仅适用于N×1×1类，推广到一般N×A×B尚未完成；③仍未解答是否存在共享相同表面积却无公共网的两种长方体；④最终算法在空间和时间上仍不具备实际大规模可扩展性。

---

## 308. EnvHarness: Awakening Static Worlds for Agent Learning

**arXiv ID:** 2608.19880 | [PDF](https://arxiv.org/pdf/2608.19880v1)

**作者:** Chengsong Huang `[一作]` (Washington University in St. Louis), Chen-Yu Lee `[通讯]` (Google Cloud AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了 EnvHarness，一种可编程层，能够在不修改原始环境代码的前提下，利用 Stage、Contract、Chain 三种插件动态定制静态环境，以适配训练中的代理需求。

**💡 创新点**

创新点在于把环境的可定制性类比为 Agent Harness，将环境包装成可编程接口，并引入了自动化的任务‑策略条件化定制循环（Observe‑Diagnose‑Write‑Validate）来诊断代理弱点并自动生成针对性环境组件。

**🔧 技术方法**

核心技术包括三类插件组件实现的环境包装、LLM 驱动的诊断与写作循环、技能抽取与检索机制，以及对标准接口的统一改造与验证。

**📊 数据集**

实验使用 ALFWorld、WebArena、SWE‑bench Verified、OfficeQA、SpreadsheetBench 这五个跨域基准数据集进行评估。

**📈 对比分析**

与无技能、原始环境技能、自动生成环境等基线相比，skill‑based 学习提升至 9.0 分、RL 训练成功率提升至 87.9%，平均步骤减少 9.8%，在所有基准上均优于传统方法。

**⚠️ 局限性**

局限性包括链式组件在自动化流程中无法观察跨环境内部状态、对 LLM 生成组件的错误率及验证成本较高，以及对目标策略过度调优可能导致泛化不足。

---

## 309. A 360-Degree Vision Dataset for Learning Yaw Control on GPS-Denied Micro-UAVs in Disaster-Response-Relevant Environments

**arXiv ID:** 2608.19866 | [PDF](https://arxiv.org/pdf/2608.19866v1)

**作者:** Niklas Voigt `[一作]` (Westphalian University of Applied Sciences), Hartmut Surmann `[通讯]` (Westphalian University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对 GPS‑否定、通信受限的灾害响应环境，使用自制微型无人机配备 360°相机，采集真实场景视频并通过动态生成平面视图，训练单帧卷积网络实现对偏航角的连续回归，以实现无人机在失去通信后自主寻找开放空间的回避行为。

**💡 创新点**

创新点包括：① 采用无遮挡 360°相机的微型无人机构造，最大化全景信息；② 设计了从全景到前视平面视图的实时数据生成器，动态产生大量标签化样本；③ 以连续偏航回归为目标，评估多种激活与正则化组合，证明 Leaky‑ReLU 最优；④ 在嵌入式 GPU/CPU 上完成实时推理，并在实际微型无人机（DJI Tello EDU）上完成半自动闭环验证。

**🔧 技术方法**

主要技术包括：卷积神经网络回归（Tanh、ReLU、Leaky‑ReLU 等）、equirectangular 投影到平面视图、图像归一化与轻量仿射增广、动态样本生成（每帧 10–20 个视角）、MSE 损失与 SGD 优化、PyTorch/ONNX 推理、低功耗嵌入式硬件测试。

**📊 数据集**

使用了自建 Adige‑360 数据集，包含 10 个多样灾害模拟场景，156 条视频（46 条无人机采集，110 条手持拍摄），总共 11,058 帧，生成 181,090 个平面样本。

**📈 对比分析**

通过三类分类验证生成器，随后在 9 种 CNN 变体上进行对比实验。最佳模型（Leaky‑ReLU α=0.05）在测试集上达到 RMSE≈15.3°、MAE≈10.1°、R²≈0.807；推理速度在 Jetson Orin 上约 650 Hz、Raspberry Pi 4 上约 10 Hz，表明在微型无人机硬件上可实现实时控制。

**⚠️ 局限性**

局限性包括：对强光、高动态范围及镜面反射的鲁棒性不足；单帧无时间记忆，难以区分死胡同与通道；偏航角回归易受标签误差影响；未对偏航率或多模态决策做处理，需进一步融合传感器、时间序列或不确定度估计。

---

## 310. A Locally Tokenized Generative Model for Robust Time-Series Watermarking

**arXiv ID:** 2608.19727 | [PDF](https://arxiv.org/pdf/2608.19727v1)

**作者:** Dongbin Kim `[一作]` (Seoul National University), Jaewook Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于局部离散化的时间序列生成模型 L-VQVAE，并在其上实现 LVQMark 水印方案，以实现对后编辑攻击的稳健检测。

**💡 创新点**

创新点在于通过局部编码保证重编码稳定性，结合红绿逻辑偏置注入与鲁棒重编码实现对时间序列水印的高可靠性与低误报。

**🔧 技术方法**

使用局部 VQ‑编码器、全局解码器、注意力变换器、红绿 logit 偏置、鲁棒重编码器以及多阶段训练流程。

**📊 数据集**

四个多变量时间序列数据集：Stocks、ETTh、Energy、fMRI。

**📈 对比分析**

与 Tree‑Ring、Gaussian Shading、TimeWak 等基线以及 DiffusionTS、TimeVQVAE、SDFormer 等生成器比较；LVQMark 在保持生成质量的前提下，三种攻击下 FPR ≤0.01、TPR≈1，显著优于现有方法。

**⚠️ 局限性**

局限：仅验证四个数据集、三种攻击和有限序列长度；多阶段训练相对复杂；对更长序列和更多攻击类型的鲁棒性尚未测试。

---

## 311. A simple and practical $o(\sqrt{n})$-time algorithm for shortest paths in power law graphs

**arXiv ID:** 2608.19538 | [PDF](https://arxiv.org/pdf/2608.19538v1)

**作者:** Jiaqi Mao `[一作]` `[通讯]` (University of Sydney), Jiaqi Mao (University of Sydney)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并分析了无预处理、子线性近似的 Pruned Bidirectional Search (PBS) 算法，用于在幂律图上计算最短路径。

**💡 创新点**

创新点是：在不进行预处理的前提下，利用高度优先队列剪枝实现子线性时间（O(n^{1-1/loglog n}/2）），并在高概率下得到 41/32 乘法近似。

**🔧 技术方法**

技术主要包括：双向 BFS、按度数排序剪枝、核心（core）划分、随机幂律图生成模型（Chung–Lu）、理论概率分析与实验评测。

**📊 数据集**

数据集：SNAP 上的五个真实网络（Epinions、Slashdot、Skitter、Pokec、LiveJournal）以及三种合成幂律图（β=2.0, 2.5, 2.9，每个一百万节点）。

**📈 对比分析**

与传统双向 BFS (BiBFS) 和 Wormhole 进行对比；PBS 在所有图上速度最快，平均比 BiBFS 提升 1.84×–7.76×，比 Wormhole 提升 1.28×–6.55×；平均乘法误差 ≤1.05，≥99.5% 的结果在两跳以内。

**⚠️ 局限性**

限制：算法仍然是基于幂律图假设，理论分析依赖高概率结论；在极少数情况下核心路由可能失败；对非幂律或动态图的适用性未做验证。

---

## 312. Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026): How a Community-Expert Ranking Holds Up Against a Large-Scale LLM Incident Corpus

**arXiv ID:** 2608.19266 | [PDF](https://arxiv.org/pdf/2608.19266v1)

**作者:** Kyriakos "Rock" Lambros `[一作]` (OWASP GenAI Security Project), Steve Wilson `[通讯]` (OWASP GenAI Security Project)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含 7,714 条公开 LLM 安全事件的语料库，并通过三种大型语言模型的投票式分类器将其中 6,639 条事件标注到 OWASP LLM 风险的 20 条目分类体系中；随后使用贝叶斯测量误差模型校正分类噪声，并将校正后的事件数据与专家投票的权重为 0.75/0.25 的方式融合，得到 2026 年 OWASP Top 10 LLM 风险的加权排名；最后对四个前沿分类器进行预注册的 bake‑off 试验，验证排名对模型改进的稳健性。

**💡 创新点**

① 将专家共识与大规模事件记录相结合，首次使用贝叶斯测量误差模型对分类器误差进行校正并量化不确定性；② 采用固定权重的加权融合（0.75 经验投票 + 0.25 事件数据）提供可解释的风险层级；③ 通过预注册的四模型烘焙测试验证排名的鲁棒性，证明事件数据对专家排名的纠正作用有限且可控。

**🔧 技术方法**

使用三大 LLM（Qwen 235B、Llama 405B、DeepSeek V3）组成的投票分类器；手工标注的 1,200 条黄金样本用于估计分类器的精确率和召回率；贝叶斯负二项测量误差模型与 MCMC（NumPyro）采样得到每条风险的潜在发生率后验分布；对齐权重后进行加权融合；利用 Cohen κ、Spearman ρ、Kendall τ、Bootstrap 等统计量评估一致性与稳健性。

**📊 数据集**

事件数据集：从 CVE、GHSA、OSV（6,297 条安全事件）与 AIAAIC（342 条 AI‑harm 事件）合成的 7,714 条公开 LLM 安全事件，其中 6,639 条已按 20 条目分类；黄金集：1,200 条人工双盲标注并最终裁定的样本；前沿分类器的预测结果保存在 GitHub 仓库。

**📈 对比分析**

与专家投票排名对比时，Cohen κ 为 0.20（90% 区间跨零），表明一致性弱；四个前沿分类器的平衡准确率均低于 0.863 的事件底线；对黄金集的 Spearman ρ 为 0.918，显示事件底线与真值的顺序高度一致；因此排名在事件数据的有限纠正下保持稳健。

**⚠️ 局限性**

限制包括：① 样本分布失衡（安全事件与 AI‑harm 事件比例极不均匀，AI‑harm 精确率未直接测量）；② 分类器未能识别“out‑of‑scope”类别，导致误归入多余条目；③ 三个“frame‑blind”条目仅来自单一数据源，缺乏交叉验证；④ 黄金集仅由单一审稿人标注，缺乏独立评估；⑤ 事件数据基于公开数据库，可能遗漏未公开的安全事件。

---

## 313. Air Traffic Control Using Large Language Models: Prompt Engineering, Architecture, and Evaluation

**arXiv ID:** 2608.19299 | [PDF](https://arxiv.org/pdf/2608.19299v1)

**作者:** Mahyar Ghazanfari `[一作]` (George Washington University), Alexandre Bayen `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估大型语言模型（LLM）在多轮空中交通控制对话中的表现，并构建基准和完整的评估管线。

**💡 创新点**

创新点包括：① 设计五种递增约束的提示结构和对话历史/示例的系统性实验；② 引入LLM评判者与人工专家评估相结合的多层评估方法；③ 将自动相似度指标与人工判断相结合，探索指标与安全性之间的关系。

**🔧 技术方法**

使用技术包括：状态化多轮对话生成、提示工程、In-Context Learning（ICL）、自动相似度评估（ROUGE‑L、BLEU、TF‑IDF、句子嵌入）以及GPT‑5.5评判者。

**📊 数据集**

使用数据集：一条真实实验通用航空航班“San Francisco Bay Area Flight Tour”的手工转录对话（P0）作为基准，另一条不同实验航班的完整转录作为示例（ICL）。

**📈 对比分析**

比较方法：对9个模型、5个提示、2种历史（自生成/真历史）和ICL/无ICL共20种条件进行交叉对照；利用自动指标、LLM评判和人工验证比较模型表现；最佳配置ROUGE‑L仅0.31，操作正确性低，提示约束越多效果越差。

**⚠️ 局限性**

限制：仅测试单一航班单一空域，缺乏实时交互与自我纠错；ICL示例不包含实时交通/天气等运营状态；评估基于最高相似度样本，人工评审单一评审者，缺乏多评审者验证。

---

## 314. Automated Estimation of MBIST Area and Test Time in Heterogeneous Memory IPs via Stacked Ensemble Framework

**arXiv ID:** 2608.19705 | [PDF](https://arxiv.org/pdf/2608.19705v1)

**作者:** Chee Jin Teoh `[一作]` (Universiti Teknologi Malaysia), Nuzhat Khan `[通讯]` (Universiti Teknologi Malaysia)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于堆叠集成学习的框架，能够仅凭RTL级别的内存设计参数（如内存数、宽度、深度、端口配置、时钟域数）在不进行完整合成与测试模式生成的前提下，预测MBIST的面积占比和测试时间。

**💡 创新点**

创新点包括：①利用域特定的RTL特征工程（多项式展开、对数变换、标准化）和语义端口识别；②构建堆叠集成模型（面积预测采用XGBoost、LightGBM与神经网络的基学习器，辅以梯度提升回归元学习器；测试时间预测采用XGBoost、LightGBM基学习器与Ridge回归元学习器）；③实现完全无合成的早期估算，显著加速设计流程。

**🔧 技术方法**

使用的技术：XGBoost、LightGBM、神经网络、梯度提升回归器、Ridge回归、Optuna超参数搜索、字符级n-gram + 逻辑回归的语义端口分类、对数与多项式特征变换、堆叠集成学习。

**📊 数据集**

数据集：4470个MBIST面积样本和624个MBIST测试时间样本，均来自在Synopsys Design Compiler与MINT工具下对自动生成的内存IP变体进行合成与测试模式生成后提取。

**📈 对比分析**

通过与传统线性/多项式回归、梯度提升回归、神经网络等基线模型对比，并采用R²、MAPE、±10%准确率三种评估指标。面积预测准确率为90.68%，比基线提升8.53%；测试时间预测准确率为96.80%，比基线提升48.80%。

**⚠️ 局限性**

局限性：①仍需大量合成和测试模式生成来构建训练数据；②模型对极端或未见配置的泛化可能受限；③堆叠集成模型训练时间长、模型体积大；④仅评估面积与测试时间，未涵盖其他MBIST相关指标。

---

## 315. Scientific Data Skills: Enabling Agent-Ready Scientific Data Services at Scale

**arXiv ID:** 2608.19625 | [PDF](https://arxiv.org/pdf/2608.19625v1)

**作者:** Xiaohan Huang `[一作]` (Computer Network Information Center, Chinese Academy of Sciences), Yuanchun Zhou `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Scientific Data Skill（SciDSK）这一面向代理的科学数据表示方式，构建了规范、构造流程和技能库，并在数据集检索与解释任务上进行评估。

**💡 创新点**

创新点在于将数据集特定知识与操作指引打包为可复用的代理技能，桥接数据元数据与代理执行接口，并通过可追踪的持久化标识实现技能与数据集的关联。

**🔧 技术方法**

采用基于 agent skill 的 YAML/Markdown 规范、BM25 检索、Qwen3.6-plus LLM 以及多种检索/解释评测指标，对 SciDSK 进行系统实现与评估。

**📊 数据集**

使用了 72 个跨六大科学学科（物理、化学、地球科学、生物、材料科学、计算机科学）的数据集构建检索基准，以及四个针对 CT 影像、GIS 光栅、图像表格和事件映射的解释案例。

**📈 对比分析**

通过对比 BM25-Raw、Agent-Raw、Agent-SciDSK-Text 与 Agent-SciDSK，结果显示 Agent-SciDSK 在 Hit@1、Recall@5、MRR 与 nDCG@5 等指标上均优于其他方法；在解释任务中，Agent-SciDSK 满足了 24 项评估标准中的 23 项，略高于仅凭页面信息的情况。

**⚠️ 局限性**

局限性包括无法完全拆分路由、注册与技能内容的独立贡献、基准规模相对有限、评测集中在受控案例，且尚未充分验证在更大规模、多样化数据环境中的泛化与安全性。

---

## 316. The Missing Touch: Spatially Distributed Tactile Feedback Brings Teleoperation Closer to Human Dexterity

**arXiv ID:** 2608.19372 | [PDF](https://arxiv.org/pdf/2608.19372v1)

**作者:** Rohan Kota `[一作]` (Northwestern University), J. Edward Colgate `[通讯]` (Northwestern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在机器人遥操作中使用高分辨率空间分布式触觉反馈，验证其对操作员自然性、速度和一致性的影响。

**💡 创新点**

首次将机器人指尖的视觉触觉传感器捕获的空间接触信息实时映射到操作员手指的形状显示，实现局部触觉重建，并证明其显著提升遥操作表现。

**🔧 技术方法**

采用双向力反馈遥操系统、GelSight视觉触觉传感器、32-DoF电液形状显示、动态时间扭曲 (DTW) 轨迹比较、线性混合效应模型以及多种统计检验方法。

**📊 数据集**

使用12名受试者完成按钮辨别任务、10名受试者完成铆钉滚动任务，共收集48次试验轨迹，四种触觉分辨率（Off、1D、2D、Full）作为数据集。

**📈 对比分析**

通过DTW比较遥操作轨迹与直接操纵轨迹，发现Full条件DTW距离显著降低，任务完成时间、超越距离、转向次数等指标显著改善；同时Full条件下的轨迹一致性和状态空间集中度提升，显示更适合用于机器人学习。

**⚠️ 局限性**

实验仅涉及2自由度单指遥操作，DTW作为评估指标存在局限；未探讨更高自由度、多指、多任务场景，也未验证在真实环境中的推广和长期工作时的疲劳影响。

---

## 317. ReguSim: Evaluating LLM Agent Rule Grounding in Financial Compliance

**arXiv ID:** 2608.19974 | [PDF](https://arxiv.org/pdf/2608.19974v1)

**作者:** Yiyang Luo `[一作]` (Hong Kong University of Science and Technology), Yunya Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了 ReguSim 与 ReguBench 两个框架，分别用于在可控的金融合规环境中评估 LLM 交易代理的推理、动作、执行与监控证据，并通过交易、监控与桥接实验系统性研究 LLM 在合规规则下的行为。

**💡 创新点**

创新点在于：①将合规评价拆分为四个独立 artifact（推理、动作、执行、证据），实现可追溯与可复现；②构建程序化生成、标记化的监管监控基准 ReguBench；③通过桥接实验揭示监控对交易者推理的依赖与执行证据的重要性；④证明结构化基线可优于仅基于提示的 LLM 监控，强调证据表示对监控效果的决定性作用。

**🔧 技术方法**

采用多种大语言模型（DeepSeek V4 Pro、Gemini 3.5 Flash、GPT‑5.4 Mini）作为交易与监控代理；实现了规则执行引擎、目标标记机制、结构化监控基线（规则基线与逻辑回归基线）以及实验脚本以自动化收集推理、动作、执行结果与监控证据。

**📊 数据集**

使用 ReguSim 生成的合成价格路径与交易记录，以及 ReguBench 的 191 场景、49,440 条合成操纵记录（wash‑trading、spoofing、pump‑and‑dump 等），所有数据均为程序化合成，不包含真实交易日志。

**📈 对比分析**

对比方法：①交易实验中统计拒绝率与规则违规率；②监控实验中使用宏 F1 评价 LLM 与结构化基线的检测性能；③桥接实验中评估监控在不同证据组合下的误判率。实验结果显示：DeepSeek 24.2% 的拒绝率高于 Gemini 14.8%；在监控任务中 GPT‑5.4 Mini 的宏 F1 为 63.8%，低于规则基线 65.0% 和逻辑回归基线 71.4%；缺少执行证据时监控误判率显著上升。

**⚠️ 局限性**

局限性包括：①规则表面化，仅实现少量可执行规则，未覆盖完整市场法典；②所有实验基于合成数据，缺乏真实交易与监管案例的检验；③模型覆盖面有限，主要为三种 LLM；④未涉及市场微观结构与更复杂的执行环境；⑤桥接实验样本量有限，未能在大规模数据上验证结论。

---

## 318. Robust Incomplete Multimodal Sentiment Analysis via Iterative Proxy Correction

**arXiv ID:** 2608.19971 | [PDF](https://arxiv.org/pdf/2608.19971v1)

**作者:** Zhifa Geng `[一作]` (Anhui Polytechnic University), Chao Kong `[通讯]` (Anhui Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对不完整多模态情感分析，提出迭代代理纠正框架，在多模态上下文中逐步改进语言代理并自适应融合。

**💡 创新点**

创新点包括：①识别单次代理构造易导致误差传播；②通过门控残差机制实现代理的迭代更新；③引入阶段性纠正目标以稳健地引导代理改进。

**🔧 技术方法**

使用 Transformer 代理生成器、门控残差更新模块、语言可靠性评分网络、自适应融合机制以及阶段性纠正损失（Stage-wise Correction Loss）等技术。

**📊 数据集**

实验使用英文的 MOSI、MOSEI 数据集以及中文的 SIMS 数据集。

**📈 对比分析**

与 Self-MM、CENet、TETFN、MMIM、MISA、ALMT、LNLN 等基线对比，结果显示该方法在三大数据集的准确率、MAE、Corr 等指标上均优于或接近最强基线，尤其在高缺失率条件下表现更稳健。

**⚠️ 局限性**

限制包括：当语言信息极度缺失或存在高度歧义时仍难以准确预测；代理初始化仅依赖非语言模态，若视觉/音频同样受损效果有限；训练需要完整语言作为锚点，实际应用中可能难以获得。

---

## 319. Optimal Skill Selection for LLM Agents with Provable Bicriteria Guarantees

**arXiv ID:** 2608.19993 | [PDF](https://arxiv.org/pdf/2608.19993v1)

**作者:** Yu Chen `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型代理的技能选择问题，建立了一个在硬 token 预算下的正则化子模最大化模型，并提出了最佳前缀选择（Best Prefix Selection）算法。

**💡 创新点**

创新点包括：① 将技能选择视为正则化子模最大化，首次给出 (1‑1/e, 1) 双准则近似保证；② 引入预算对齐插值（budget‑aligned interpolation）技术实现该保证；③ 构建结构化能力模型和线性上下文成本，能够仅凭执行成功/失败记录学习有效的需求/供应向量。

**🔧 技术方法**

使用的技术包括：子模最大化理论、密度贪心与种子枚举、预算对齐插值、神经/查表式需求与供应编码器、线性上下文惩罚模型。

**📊 数据集**

实验数据集为自制的 contamination‑controlled BigCodeBench 变体，包含 31 个技能、5 维能力向量，基于 63,596 次真实执行记录进行模型拟合与评估。

**📈 对比分析**

与多种现有路由器、检索器以及手工算法在同一 benchmark 上进行对比；Best Prefix Selection 在 0.73 的任务成功率上显著优于 0.20–0.52 的基线，并且使用 28% 更少的 token；在神经能力编码的设置下仍达到 0.68 的成功率，优于所有部署系统。

**⚠️ 局限性**

局限性：仅考虑线性每‑token 的上下文惩罚；实验仅在单一冻结执行器上验证；需要手工构造私有模块与任务；未对流式查询或在线选择进行研究。

---

## 320. Systematic Evaluation of TabPFN-TS for Zero-Shot Probabilistic Heat Load Forecasting in District Heating Networks

**arXiv ID:** 2608.20024 | [PDF](https://arxiv.org/pdf/2608.20024v1)

**作者:** Ben Spoek `[一作]` (RWTH Aachen University), Dirk Müller `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了TabPFN-TS在城区热网络零样本概率预测中的表现，并设计了多分辨率残差纠正模型以提升长周期规划准确性。

**💡 创新点**

创新之处在于首次把TabPFN-TS用于热负荷零样本预测，揭示了12周滚动上下文和仅温度的高效配置，并提出将基线与短期残差融合的多分辨率残差纠正预测器。

**🔧 技术方法**

采用TabPFN-TS基础模型、Chronos-2时序基础模型和AutoGluon多种机器学习基线，并结合滚动历史上下文、气温协变量与时间特征进行零样本概率预测。

**📊 数据集**

实验基于慕尼黑区城热网络的15分钟和小时级历史负荷与气象数据，以及公开的弗伦斯堡热网络小时级负荷数据。

**📈 对比分析**

使用CVRMSE、R²、MAE、CRPS、MACE和RTF等指标对比，实验发现Chronos-2在确定性误差上略优，但TabPFN-TS在全年度和迁移测试中仅次于Chronos-2，且在概率校准上更佳；MRRC框架在12h能量误差上进一步提升并显著降低计算成本。

**⚠️ 局限性**

局限性在于模型仅使用温度协变量、未针对热负荷数据进行专门预训练、15分钟高频预测仍受限，且季节相关上下文无显著提升，需进一步验证其在不同规模和更长周期预测中的通用性。

---

## 321. Orthogonal JEPA: Factorized Predictive States for Latent World Models

**arXiv ID:** 2608.20065 | [PDF](https://arxiv.org/pdf/2608.20065v1)

**作者:** Taoyong Cui `[一作]` (Chinese University of Hong Kong), Wanli Ouyang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Orthogonal Joint‑Embedding Predictive Architecture (OJEPA)，通过多支预测分支和正交基矩阵将目标状态拆分为多个可预测组件，并在多种任务中验证其效果。

**💡 创新点**

核心创新在于：① 用正交预测因子化将单一目标状态分解为多个正交子空间；② 引入正交性、因子活跃度和在线方差正则化，避免冗余与编码器坍塌；③ 统一的预测‑状态接口，使同一机制可应用于视觉、细胞、临床、控制与分子动力学等多领域。

**🔧 技术方法**

采用基于EMA的目标编码、基于Transformer/MLP的上下文编码器、正交基矩阵学习、因子预测分支、伪逆状态合成，以及预测损失、正交损失、因子活跃度损失和在线方差损失的联合训练。

**📊 数据集**

实验数据集包括：控制视觉绑定（MuJoCo 场景）、单细胞转录组（人类肾细胞与 PBMC-10K、Adamson、Norman 数据）、临床健康状态预测（超过 1000 个未来事件的电子病历）、连续控制（Walker2d、HalfCheetah、InvertedPendulum）以及分子动力学（水、石英、对羟基苯甲酰胺、苯乙烯）。

**📈 对比分析**

与标准 JEPA 及多种基线方法（如 Random Forest、XGBoost、Qwen 等）在相同的采样、网络结构、训练预算下进行公平对比；OJEPA 在视觉绑定中提升了 INJ 与网格恢复率并降低崩溃率；在单细胞聚类与扰动预测中 AvgBIO 与 Pearson 相关性提升；在临床预测中 Mean PRAUC 提升 0.7%；在连续控制中 CEM 规划回报提升至 45–30 的范围；在分子动力学中 MAE 与 RMSD 均显著降低。

**⚠️ 局限性**

局限性包括：正交性仅保证几何分离，无法保证统计独立或语义可解释；因子活跃度正则化仅控制边际方差，未保证完整协方差矩阵正定；状态合成对基矩阵条件数敏感；预测器为确定性模型，未捕获多模态未来；未涵盖像素级闭环控制、随机未来或已知因果因子场景。

---

## 322. A large-scale dataset of sub-institution name disambiguation and hierarchical structures from OpenAlex

**arXiv ID:** 2608.20035 | [PDF](https://arxiv.org/pdf/2608.20035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 323. EchoCoT: Extracting Hidden Chain-of-Thought from Large Reasoning Models

**arXiv ID:** 2608.20055 | [PDF](https://arxiv.org/pdf/2608.20055v1)

**作者:** Yiting Qu `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过工具调用交互，提取大型推理模型隐藏的链式思考（CoT）并实现近乎逐字复现。

**💡 创新点**

发现并利用工具调用中的推理状态连续性作为“回放表面”，提出EchoCoT多步攻击并构建LLM驱动的注入轨迹优化框架。

**🔧 技术方法**

采用 Scratchpad 工具、长度误差与摘要回忆评估、LLM 反射–生成–蒸馏三阶段搜索等技术。

**📊 数据集**

评估数据集包括 OpenThoughts、MATH500、JEEBench、LiveCodeBench 等，同时对 DeepSeek、Qwen、GLM 等开源模型和 Gemini、Claude 系列专有模型。

**📈 对比分析**

与直接提示、CoT 合成、REP 等基线相比，EchoCoT 在开源模型上可达 66.4% 的 ASR@90，专有模型可提取数十万 token 的 CoT，显示在多任务、多域上的显著优势。

**⚠️ 局限性**

主要限制为：对专有模型缺乏真实 CoT 做精确评估、单轮实验导致变异性未测、以及未探讨更根本的安全训练对策。

---

## 324. DecoVAE: a Lightweight Interpretable Trend-Seasonal VAE Framework for Efficient Probabilistic Time Series Forecasting

**arXiv ID:** 2608.20052 | [PDF](https://arxiv.org/pdf/2608.20052v1)

**作者:** Alexander Marusov `[一作]` (Applied AI Institute), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了 DecoVAE，一种轻量化可解释的变分自编码器，用于概率时间序列预测，通过显式趋势–季节分解来建模时间序列。

**💡 创新点**

创新点在于将趋势与季节分别在时域和频域建模，趋势通过类似 Hodrick–Prescott 的差分正则化保证平滑性，季节采用复数 Gaussian VAE 捕获幅值与相位；同时保持模型轻量、可解释且计算高效。

**🔧 技术方法**

使用技术包括变分自编码器框架、差分正则化（HP滤波）、FFT 频域变换、复数 Gaussian VAE、两流架构、KL 散度约束、CRPS 与 NMAE 评估指标。

**📊 数据集**

实验数据集涵盖七个真实世界数据集：Electricity、Traffic、ETTh1、ETTh2、ETTm1、ETTm2、Weather。

**📈 对比分析**

与 K^2VAE、LaST、DeepAR、TFM、TSDiff、DeNOTS 等基线对比；在短期任务中 CRPS 与 NMAE 分别提升 14.96% 与 23.30%，在长期任务中提升 52.68% 与 26.51%；平均排名第一；模型权重缩减 93%，推理速度提升 74%。

**⚠️ 局限性**

局限性包括：对分解前的移动平均滤波依赖较强，难以捕捉突发异常或非周期性变化；对多频率、非线性季节性的建模仍有限；在极端事件或非平稳序列上的鲁棒性尚待进一步验证。

---

## 325. Auditing Cross-Lingual Fairness in Language Model Watermarking

**arXiv ID:** 2608.20047 | [PDF](https://arxiv.org/pdf/2608.20047v1)

**作者:** Alexander Nemecek `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估多语言LLM文本水印的检测和质量公平性，揭示跨语言差异与校准失效。

**💡 创新点**

提出四组件评估框架：经验FPR校准、阈值无关AUC、三种独立质量范式（MAUVE、BERTScore、参考PPL）以及基于语言族的泛化熵分解。

**🔧 技术方法**

技术包括经验阈值校准、AUC判别、分布式MAUVE、语义BERTScore、参考模型PPL、泛化熵分解及脚本/族分区分析。

**📊 数据集**

使用11种语言（4种脚本、8个语言族）来自FLORES+并行翻译与AYA本地提示的数据；对三大开源LLM（Mistral-NeMo-12B、Gemma-3-4B、Qwen2.5-7B）进行评估。

**📈 对比分析**

在两种生成模式下，对6种水印方案、3种生成器进行≈200k匹配对评估；发现跨语言差异主要为族间，校准失效被误判为检测失败，失真方案在质量上表现最差。

**⚠️ 局限性**

局限性包括语言覆盖仅限11种、方案覆盖仅为6种，且结构性差异难以通过数据扩充解决，需要在标记器设计和分词器层面进一步改进。

---

## 326. ExPhy: A Benchmark for Explicit Physical Property Learning in Multi-Object Trajectory Forecasting

**arXiv ID:** 2608.20009 | [PDF](https://arxiv.org/pdf/2608.20009v1)

**作者:** Rui Wang `[一作]` (Beijing University of Posts and Telecommunications), Mengshi Qi `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ExPhy基准并设计PhyODE模型，能够在多物体轨迹预测任务中同时学习并估计质量、摩擦系数和恢复系数；

**💡 创新点**

1) 在同一数据集上统一评估轨迹预测与物理属性回归；2) 引入可微分物理引导的Hybrid ODE推演，通过属性接口实现物理属性学习与轨迹生成；3) 构建受控ID/OOD拆分测试物理推理泛化；

**🔧 技术方法**

使用PyBullet生成数据，轨迹编码器+关系注意力构建属性回归网络，结合可微分物理模块和Neural ODE残差，采用RK4积分和交叉熵/平滑L1损失进行端到端训练；

**📊 数据集**

24k场景的ExPhy（A/B/C）以及在ComPhy上的零射转移实验；

**📈 对比分析**

与VRDP、PHYCINE、PCR、PAINET、GSE-Flow、MoFlow、Neuralized MRF、PRF等基线在ADE/FDE和NMAE上对比，PhyODE在ID/OOD长时序上取得最优或第二优，并在ComPhy零射转移中表现最佳；

**⚠️ 局限性**

轨迹误差低并不必然表征物理属性准确；模型对极端参数或更复杂多体交互的泛化仍有限；属性预测受观测信息限制。

---

## 327. Evaluating Automated Testing on an Open-Source Web Application Using Cypress

**arXiv ID:** 2608.19960 | [PDF](https://arxiv.org/pdf/2608.19960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 328. SCoRD: Semantic-Assisted Continual Retriever-Reranker Distillation for LLM-Based Recommendation

**arXiv ID:** 2608.19998 | [PDF](https://arxiv.org/pdf/2608.19998v1)

**作者:** Seunghyun Baek `[一作]` (Korea University), SeongKu Kang `[通讯]` (Korea University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个持续知识蒸馏框架，通过语义推理助手实现LLM重排器与ID检索器在非平稳数据流中的协同适配。

**💡 创新点**

创新点在于把LLM的意图推断能力转化为可重用的意图级指导，采用选择性蒸馏低置信度序列、检索器无LLM更新的语义指导以及检索器信息回馈给重排器的三阶段持续蒸馏。

**🔧 技术方法**

使用了知识蒸馏、动态意图记忆、协同自训练、意图漂移负采样、LLM Prompt投影、LoRA等技术。

**📊 数据集**

使用Amazon Books、Yelp和Amazon Movies & TV三大真实推荐数据集，按时间拆分为5个块进行非平稳实验。

**📈 对比分析**

与全量重训、微调、CL方法、CCD、LLM-D4Rec、CoT-Rec等多类基线在HitRate/NDCG@5/10/20上对比，SCoRD在所有数据集均优于基线，尤其在新用户和兴趣漂移用户上提升显著，并保持稳定-适应平衡。

**⚠️ 局限性**

局限性在于意图识别准确性和内存扩展受限，意图记忆更新频率与LLM成本权衡仍需进一步优化，且在更大规模或多模态场景下的可扩展性尚未验证。

---

## 329. PVRA: A Pointwise Key-point Voting Framework for Robotic Assembly

**arXiv ID:** 2608.19968 | [PDF](https://arxiv.org/pdf/2608.19968v1)

**作者:** Kulunu Samarawickrama `[一作]` (Tampere University), Roel Pieters `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 PVRA 框架，能够基于 RGB‑D 输入通过关键点投票完成目标与基底对象的语义分割、6‑DoF 姿态回归以及装配姿态预测，从而实现对进程式装配任务的感知与可执行输出。

**💡 创新点**

创新点在于：①使用 3D 关键点投票方法与点云/图像特征融合，直接从不完整点云中学习装配上下文；②将目标分割、目标姿态、装配姿态统一在同一端到端网络中学习；③引入针对进程式装配的专门评价指标（Step‑Segmentation Accuracy、Step‑Localization Accuracy 等）和相应的损失函数。

**🔧 技术方法**

技术上采用 RGB‑D 语义特征提取（预训练 CNN）+ PointNet++ 点云特征提取并融合；使用多头预测器（SEG、KpOF、AKpOF）进行点级角色分类与关键点偏移回归；训练采用 focal loss + offset loss 的联合损失。

**📊 数据集**

使用合成的 Nema17 gear‑reducer 进程式装配数据集，该数据集包含 431 个装配实例、5 个零件、4 个装配步骤，并提供 RGB‑D 图像、CAD 模型与 6‑DoF 姿态标注。

**📈 对比分析**

与基线 CAD‑ICP‑PCA（基于点云配准）和 FoundationPose（基于 CAD‑RGB‑D 的姿态估计）进行对比。PVRA 在 Step‑Acc@0.15d、Target/Assembly AUC 等指标上表现更优，尤其在使用 PVRA 预测的稀疏掩模时仍能保持较高的准确率，说明对不完整点云具有更强鲁棒性。

**⚠️ 局限性**

局限性包括：仅在仿真数据上验证，缺乏真实世界噪声与光照变化的考察；假设装配序列固定、每步仅单一目标且仅一个基底对象，难以直接推广至更复杂、多目标或开放式装配任务；缺少大规模多模态装配数据集，模型在跨域迁移方面尚待验证。

---

## 330. Rethinking Patch Based Multivariate Time Series Forecasting with Semantic Structured Partitioning

**arXiv ID:** 2608.19966 | [PDF](https://arxiv.org/pdf/2608.19966v1)

**作者:** Jiazhe Wang `[一作]` (Henan University of Science and Technology), Ruijuan Zheng `[通讯]` (Henan University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于语义结构划分的多变量时间序列预测框架SCPaT，先自适应生成语义单元，构造转移熵图并聚类成高阶语义块，再通过重要性感知路由将不同块分配给专门专家进行建模。

**💡 创新点**

核心创新在于：① 将时序分块视为语义建模任务，利用自适应阈值划分稳定与变动段；② 用可微分转移熵估计单元间有向非线性依赖，并稀疏化后聚类形成语义块；③ 引入重要性感知路由实现块级专家动态分配，显著提升模型对不同时序特征的捕获能力。

**🔧 技术方法**

技术手段包括多尺度时间卷积提取特征、语义向量编码、可微转移熵图构建与稀疏化、基于模块化图聚类生成语义块、顶点级专家网络与Top‑P路由、Transformer自注意力整合。

**📊 数据集**

在12个公开数据集上验证，涵盖ETT系列、Weather、Traffic、Electricity、Solar（长周期）以及PEMS系列（短周期）。

**📈 对比分析**

与PatchTST、TimesNet、HDMixer、iTransformer、Crossformer、MSPatch、DUET、MSGNet和LSTM等9个基线对比，SCPaT在大多数数据集与预测时长上均实现MSE/MAE最优或次优，平均相对提升约4–7%。

**⚠️ 局限性**

局限性包括：① 生成转移熵图和聚类的计算量相对较大，尤其在高维或极长序列时；② 目前的稀疏比例和Top‑P阈值需要经验调参；③ 对极端不规则或大缺失率的数据仍可能受限，后续可探索更高效的图构建与动态稀疏策略。

---

## 331. From Agent Behaviour to Agent-Friendly Documentation: An Empirical Study of How Coding Agents Discover, Read, and Write Technical Documentation

**arXiv ID:** 2608.20195 | [PDF](https://arxiv.org/pdf/2608.20195v1)

**作者:** Zhijun Gao `[一作]` (Peking University), Jing Chen `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对编码代理在实际开发会话中的技术文档交互行为进行经验研究，结合SWE-chat与AIDev两个公开数据集；

**💡 创新点**

首次揭示代理主导阅读和编写自身指令文件与工作笔记，驳斥传统API参考主导的假设，并提出双瓣循环的交互模型；

**🔧 技术方法**

使用多格式事件抽取、文件路径二层分类器、转移概率与提升度分析、阶段调整的逻辑回归、聚类自举和GEE等统计方法；

**📊 数据集**

使用SWE-chat 557会话（约94,813事件，3,033文档交互）和AIDev 33,097拉取请求（690,260文件级变更）；

**📈 对比分析**

通过对比未调整提升度与阶段调整后的OR，评估文档阅读后代码编辑、测试与构建的关联，发现调整后关联略高但并不稳健；文档生成率与阅读相当，未显示显著性能提升；

**⚠️ 局限性**

局限在于仅识别路径文档，忽略内联注释与docstring；文件路径分类未进行人工验证；样本高度聚焦单一代理与公共仓库，外部可推广性受限。

---

## 332. Exact Algebraic Computation of Learning Coefficients for Two-Dimensional Singular Models

**arXiv ID:** 2608.20183 | [PDF](https://arxiv.org/pdf/2608.20183v1)

**作者:** Grégoire Sergeant-Perthuis `[一作]` (Sorbonne Université), Jules Tsukahara `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了第一个确定性算法，用于计算任意二维多项式的局部RLCT，并给出了其算术复杂度上界。

**💡 创新点**

创新点在于把Varchenko的解析方法转化为可终止、可执行的算法，解决了RLCT计算的终止性与复杂性问题，并首次对多层多项式神经网络（PNN）进行精确RLCT求解。

**🔧 技术方法**

采用Newton多边形、Puiseux根、右等价变换和多项式归一化等代数几何技术构造算法；在PNN实验中使用接触等价多项式构造与求导技术。

**📊 数据集**

实验基于合成数据，构造多层（L=1…5）激活度（r=2…4）的PNN回归模型，生成的多项式H_L,r(θ)作为实验对象。

**📈 对比分析**

与基于SGLD的估计器比较，算法在大多数情形下计算时间远快于SGLD（秒级vs分钟级），且不需要超参数调优；在深层或高激活度时算法仍保持可行。

**⚠️ 局限性**

局限性：仅适用于二维参数（两参数模型），无法直接推广到更高维；算法聚焦局部RLCT，缺乏对全局奇异点分层或非局部情形的处理。

---

## 333. Ask Self, Ask Others: Relation Is All You Need

**arXiv ID:** 2608.20172 | [PDF](https://arxiv.org/pdf/2608.20172v1)

**作者:** Yuting Ge `[一作]` (City University of Hong Kong), Mingkai Nie `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Relation 作为新型 token‑mixing 原语，并实现 Full/Flash/Linear/Hybrid Relation 等变体

**💡 创新点**

先将 pairwise 证据拆分为 Self 与 Exchange 两类关系，再在此基础上归一化信息流，改变了注意力的先后顺序

**🔧 技术方法**

使用 Self–Exchange Relation (SER)、Multi‑Head Relation (MHR)、FlashAttention‑style tiled 执行、递归 Relation Cache 等技术

**📊 数据集**

TinyStories、SmolLM 等小型文本数据集

**📈 对比分析**

在 10M、30M、100M 参数规模下与标准 MHA/FlashAttention 比较，Full Relation 的最终 NLL 分别比 MHA 提升 0.0412、0.0151、0.0310；FlashRelation 速度提升 3.6–4.4×；Hybrid Relation 仍保持低 NLL

**⚠️ 局限性**

仅评估到 100M 参数、单语文本，未验证更大规模或多模态场景

---

## 334. FormalTCS: Benchmarking End-to-End Frontier Formal Theoretical Computer Science Research of Large Language Models

**arXiv ID:** 2608.20153 | [PDF](https://arxiv.org/pdf/2608.20153v1)

**作者:** Dingzirui Wang `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了一个专家验证的端到端终端理论计算机科学（TCS）研究基准，涵盖 175 篇 2025‑2026 年顶会（STOC、FOCS、SODA、COLT）论文的核心命题与 Lean4 形式化证明。

**💡 创新点**

创新点在于：① 采用真实前沿论文而非教材或已公开库，② 设计完整的研究流程（理解、自动形式化、证明策略、正式证明）实现细粒度诊断，③ 引入专家验证的 Lean4 证明以保证数据质量，④ 结合多智能体框架探索 LLM 自主生成并过滤创新命题。

**🔧 技术方法**

技术手段包括：Lean4 形式化与验证、GPT‑5.6‑sol 与 Codex 辅助标注、自动化检验（Pass@k、BEq+）、多智能体循环（Planner/Formalizer/Judger）以及自定义指标与 LLM‑Rubric 评估。

**📊 数据集**

数据集为 175 个实例，每个实例来源于 2025‑2026 年 STOC/FOCS/SODA/COLT 论文，包含自然语言核心命题、正式 Lean 定理与证明、自然语言证明草图，并经过专家复核。

**📈 对比分析**

对比方法：在四个阶段（理解、策略生成、形式化、证明）分别采用多模型（GPT、Claude、DeepSeek）进行评估，使用 Pass@8、BEq+、LLM‑Rubric 等指标；实验显示最佳模型 Claude‑Opus‑5 在形式化阶段仅 11.5% Pass@8，证明阶段最高 28.6% Pass@8，表明自动形式化是主瓶颈。

**⚠️ 局限性**

局限性：自动形式化能力不足导致整体流水线失败；生成的创新命题极少（6/64 经专家筛选），说明 LLM 的研究“品味”有限；数据集受限于可公开论文且排除潜在泄露，但仍存在污染风险。

---

## 335. Evaluating Neural Cartographic Relief Shading for Urban Environments: A Downtown Calgary Study Using High-Resolution DEM and DSM Data

**arXiv ID:** 2608.20149 | [PDF](https://arxiv.org/pdf/2608.20149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 336. Decoding silent reading from non-invasive EEG

**arXiv ID:** 2608.20186 | [PDF](https://arxiv.org/pdf/2608.20186v1)

**作者:** Ingo Marquardt `[一作]` (nubrain), Priyanka Jain `[通讯]` (nubrain)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过收集约49小时、240,000次词呈现的19通道干燥电极EEG，研究在无声阅读任务中使用对比学习解码开放词汇级的词信息。

**💡 创新点**

创新点在于：①随机排版降低视觉词形与词义的共线性；②利用Llama‑3.1‑8B隐藏层嵌入做CLIP式对比目标；③引入位置探针剖析词级信息与情境追踪以及非神经位置先验的贡献。

**🔧 技术方法**

技术方法包括：双通道卷积EEG编码器→可选因果Transformer→L2归一化→CLIP对比损失；使用大语言模型（Llama‑3.1‑8B）提取词嵌入，层0做无上下文，层20做有上下文；并在训练中做时间窗、增量、温度、负样本掩蔽等超参调优。

**📊 数据集**

数据集为单个被试的连续小说文本朗读（SHERLOCK Holmes）所产生的240,141次词展示，覆盖约49小时的EEG记录，19个电极，采样600Hz。

**📈 对比分析**

评估采用512候选词池的词分组Top‑10检索，差分于经验置换基线，确保chance为0。结果显示：非上下文无Transformer配置下的within‑run提升≈7.5pp；全上下文+Transformer提升≈7.8pp；整体提升随训练数据对数线性增长，未出现饱和；词频分布显示稀有词和中频词亦可解码。

**⚠️ 局限性**

局限性包括：仅单个受试者（上界效应）；验证集用于checkpoint与配置选择导致指标可能高估；视觉词形混淆未完全排除（仅随机排版）；未验证模型对真实内隐语音的跨模态转移；对多受试者泛化与实际BCI应用仍需进一步研究。

---

## 337. Quantifying over Optimal MSO-Definable Sets on Graphs of Bounded Clique-Width

**arXiv ID:** 2608.20175 | [PDF](https://arxiv.org/pdf/2608.20175v1)

**作者:** Tatsuya Gima `[一作]` `[通讯]` (Hokkaido University), Tatsuya Gima (Hokkaido University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种扩展计数单调二阶逻辑（CMSO）的逻辑AMCMSO，并在团宽度和树宽度受限的图上给出了固定参数可解的模型检测元定理。

**💡 创新点**

创新点在于引入最小/最大赋值谓词，既保持了逻辑的可计算性，又能表达多层次图优化问题，从而突破了传统CMSO扩展无法处理的情形。

**🔧 技术方法**

主要技术是对Feferman–Vaught定理的扩展，结合动态规划与测度化简理论，构造针对AMCMSO的分解树递归计算框架。

**📊 数据集**

由于是理论性工作，未使用具体数据集，而是以图的结构参数（团宽度、树宽度）为依据进行证明。

**📈 对比分析**

与传统CMSO和其他多层次框架相比，所给算法在参数化为团宽度+最优值偏差的情况下保持线性或FPT时间，显著降低了对解的大小参数的依赖。

**⚠️ 局限性**

局限性包括在允许外部集合变量的最优性谓词时模型检测变为多项式层级难解，以及对某些高阶量化形式仍未能得到XP或FPT结果。

---

## 338. Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms

**arXiv ID:** 2608.20111 | [PDF](https://arxiv.org/pdf/2608.20111v1)

**作者:** Yanchen Guan `[一作]` (University of Macau), Zhenning Li `[通讯]` (University of Macau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了规划导向的端到端自动驾驶（E2E-AD）研究进展，提出了四轴（输入、输出、监督、评估）规划导向分类框架，并系统评述了从行为克隆到现代规划体系的演进；

**💡 创新点**

创新点在于把端到端学习的焦点从“无结构”转向“学习结构并以规划为目标”，并通过整合现有基准（nuPlan、Bench2Drive、NAVSIM、WOD‑E2E等）对评价协议进行梳理，揭示开放式挑战；

**🔧 技术方法**

综述涵盖的技术包括BEV/向量化场景表示、统一感知-预测-规划架构、世界模型推理、视觉‑语言‑动作（VLA）系统以及多模态生成式规划等；

**📊 数据集**

使用的数据集与基准主要包括nuScenes、Waymo Open、CARLA、TAD‑E2E、WOD‑E2E等真实与仿真日志；

**📈 对比分析**

在比较方面，文章指出不同基准间指标不一致，强调需要跨协议的对比，展示了各方法在不同评价指标（如ADE/FDE、闭环路程完成率、事故率、偏好评分）下的相对表现；

**⚠️ 局限性**

局限性包括对公开学术论文的依赖，未覆盖专有L4系统；对近期工作（如2025‑2026年方向性研究）缺乏充分验证；以及评价协议与数据集版本的敏感性导致结果可比性受限。

---

## 339. BreakGuard: Towards Detecting Dependency Breaking Changes with LLM-Generated Tests

**arXiv ID:** 2608.20167 | [PDF](https://arxiv.org/pdf/2608.20167v1)

**作者:** Rachna Raj `[一作]` (Concordia University), Diego Elias Costa `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 BreakGuard，一个基于大型语言模型（LLM）的工具，自动生成针对客户端调用的第三方库的迁移测试用例，以在库升级时检测破坏性更改。

**💡 创新点**

创新点在于：①将静态分析提取的客户端焦点方法与库调用点作为测试生成的目标；②使用结构化提示结合不同上下文级别（最小、方法、类）指导 LLM 生成可直接编译、运行并覆盖这些调用；③在实际的 BUMP 真实世界破坏性更新上评估 LLM 生成的测试，展示其在检测崩溃型破坏性更改上的有效性。

**🔧 技术方法**

技术包括：Java 静态分析（Spoon 框架）提取焦点方法与调用点；LLM（GPT‑4o、Qwen3‑Coder‑480B、GPT‑OSS‑120B）单次生成测试文件；JDK 测试框架（JUnit4/5、TestNG）统一格式；基于 Docker 的可重复执行环境；token 与成本计量。

**📊 数据集**

使用 BUMP 数据集（89 条真实的破坏性更新实例，涵盖 25 个开源库、31 个客户端项目），每条实例提供旧版与新版的 Docker 镜像。

**📈 对比分析**

评估通过三大指标：①生成测试的有效率（编译/运行通过）；②检测率（在新版上失败且旧版通过的测试数量），最高可达 30.3%（27/89）；③成本（每次检测平均 0.09–0.90 美元）。相较于现有的静态 API 兼容性分析或手工测试，BreakGuard 能自动补齐大部分库调用的测试覆盖，且成本低廉。

**⚠️ 局限性**

局限性包括：①大部分检测集中在崩溃型破坏性更改，行为更改识别不足；②仅使用单次生成，无迭代修复循环导致编译失败率高（最高 66%+）；③仅在 Java/Maven/JUnit 环境下验证，跨语言或构建系统的可迁移性未知；④未结合运行时信息或库差异提示，导致某些破坏性更改被忽略。

---

## 340. G3Ego: Gaze-Guided Graphs for Egocentric Action Understanding

**arXiv ID:** 2608.20157 | [PDF](https://arxiv.org/pdf/2608.20157v1)

**作者:** Marko Haralović `[一作]` (University of Zagreb), Estefania Talavera Martinez `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种利用视线引导的图结构，对第一人称视频中的手物交互进行稀疏帧的语义图构建并进行时序聚合，实现动作识别与预测。

**💡 创新点**

创新点在于将视线直接作为结构化剪枝依据，生成紧凑可解释的动作场景图，减少对大规模视频预训练的依赖。

**🔧 技术方法**

使用的技术包括Vision‑Language模型（Qwen3‑VL）、视觉编码器DINOv3、对象与手部定位GroundingDINO、图嵌入网络及Transformer时序聚合。

**📊 数据集**

实验数据集为MECCANO和EGTEA Gaze+，分别涵盖装配与厨房场景的第一人称动作。

**📈 对比分析**

在MECCANO上取得21.34 Macro‑F1，超越同类方法；在EGTEA Gaze+上平均准确率56.49%，在不使用异构预训练的模型中表现最佳。

**⚠️ 局限性**

局限在于对长时序依赖处理有限、仅使用单一视线坐标，且在极端遮挡或多视线场景下效果可能下降。

---

## 341. Navigating and Retrieving Information in Immersive Model-Based Design Reviews: An Exploratory Study

**arXiv ID:** 2608.20128 | [PDF](https://arxiv.org/pdf/2608.20128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 342. Artificial Intelligence for Workflow Analysis in Colorectal Surgery: A Multicentric, Cross-Procedural Development and Generalization Study

**arXiv ID:** 2608.20154 | [PDF](https://arxiv.org/pdf/2608.20154v1)

**作者:** Pietro Mascagni `[一作]` (IHU Strasbourg), Nicolas Padoy `[通讯]` (University of Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并评估了 AI-ColoWorkflow，利用统一的 ColoWorkflow 术语框架，对多中心、多手术类型的微创结直肠手术进行自动化流程（阶段和步骤）识别。

**💡 创新点**

首次在跨中心、多手术类型下实现基于共识框架的统一流程识别；通过多任务学习同时预测阶段和步骤，并证明全局模型可兼顾多种手术。

**🔧 技术方法**

采用 Fine‑tuned DINOv3 视觉 Transformer 进行帧级特征提取，再结合层次化多阶段 Temporal Convolutional Network（SAHC）进行时序建模，完成阶段与步骤的联合识别。

**📊 数据集**

54 份成人微创结直肠手术视频，来自 4 个中心（意大利、西班牙、法国）和公开的 HeiCo 数据集，涵盖 5 种手术类型（左侧结肠切除、右侧结肠切除、直肠切除、乙状结肠切除、全结直肠切除）。

**📈 对比分析**

与单中心/单手术类型模型以及留一中心交叉验证对比，全球模型在阶段识别上在 4/5 个中心/手术类型优于专门模型，宏 F1 达 73.0%；步骤识别宏 F1 为 39.8%，在部分中心/手术类型仍低于专门模型；留一中心验证显示阶段宏 F1 下降范围 37–66%。

**⚠️ 局限性**

主要限制包括数据量有限、步骤类别严重不平衡、仅有单一注释者、未评估空间任务（如器械识别）和外部验证，导致步骤识别受视觉相似性与稀缺样本限制。

---

## 343. Petri Net Description of Biological Neural Circuits for Fast Hardware Prototyping

**arXiv ID:** 2608.20147 | [PDF](https://arxiv.org/pdf/2608.20147v1)

**作者:** Carlo daCunha `[一作]` (New Jersey Institute of Technology), Marcos Turqueti `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于T时序Petri网的“Petri neuron”模型，用以实现离散事件驱动的神经网络实时仿真并给出可验证的时延保证；

**💡 创新点**

创新点在于将神经元的突触、漏电、阈值、递归等生理过程映射为Petri网结构，并利用正式可达性与时序分析得到闭式最差响应时间（WCRT），实现硬实时时延保障；

**🔧 技术方法**

采用Petri网建模、事件驱动调度、Padé逼近映射LIF参数、硬件抖动测量与时序分析；

**📊 数据集**

通过模拟三组生物微电路（反馈抑制、侧抑制、层级特征检测）进行验证；

**📈 对比分析**

与传统基于连续LIF模型的频率匹配进行对比，WCRT预测与仿真结果高度一致；在Raspberry Pi RP2040上实现，平均抖动仅1 µs，AMD Linux平台抖动达≈900 µs；

**⚠️ 局限性**

局限在于仅建模绝对性突触抑制期、离散化导致低频量化误差、静态突触权重、对大规模网络调度复杂度未充分评估；

---

## 344. Formal Performance and Compile Time Guarantees for Compiler Optimization Heuristics

**arXiv ID:** 2608.20137 | [PDF](https://arxiv.org/pdf/2608.20137v1)

**作者:** Nikil V. Shyamsunder `[一作]` `[通讯]` (Cornell University), Nikil V. Shyamsunder (Cornell University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

验证并证明了编译器内联扩展（inlining）Pass在基于指令缓存成本模型下的性能与编译时间保证；

**💡 创新点**

首次将编译器优化 Pass 视作拥塞游戏，利用潜在函数与价格无政府效应（PoA）给出 2.5 倍的性能上界和收敛步骤上界；

**🔧 技术方法**

采用 Coq（Rocq）进行机械化证明，结合拥塞游戏理论、潜在函数、PoA 以及折衷算法分析；

**📊 数据集**

未使用真实数据集，所有结论均来自理论模型与 Coq 证明；

**📈 对比分析**

通过理论分析与潜在函数证明与最优解的比例关系，得到 2.5 的 PoA 上界，并给出迭代最佳响应的收敛步数上界，未进行实验性能比较；

**⚠️ 局限性**

局限性包括：模型仅考虑叶子调用点、仅关注指令缓存成本、未涵盖多 Pass 交互、未验证在真实编译器（如 LLVM）中的实际性能与编译时间表现。

---

## 345. ArmorOCR: Grounded Adversarial Visual Perception via Observation-Transferred Self-Distillation

**arXiv ID:** 2608.20122 | [PDF](https://arxiv.org/pdf/2608.20122v1)

**作者:** Linhan Cao `[一作]` (Ant Group), Wei Sun `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AdvSpot基准，评估大型多模态模型在对抗性视觉文本识别中的定位、识别和基于区域的问答能力，并设计了ArmorOCR两阶段训练框架提升该能力；

**💡 创新点**

创新点包括：①以区域定位为核心的对抗性OCR任务定义与细粒度标签；②将变换视角下的优势通过观察转移自蒸馏（OPSD）注入模型；③使用任务条件的GRPO奖励联合优化定位、识别、完整识别与VQA四个目标；

**🔧 技术方法**

主要技术为On‑Policy Self‑Distillation (OPSD) 与响应区域感知的分布匹配、Group Relative Policy Optimization (GRPO) 与多任务奖励、以及多变换视角的教师引导；

**📊 数据集**

使用了新构建的AdvSpot数据集（390图像，5类13细分对抗OCR模式，包含框、文本、类型和区域问答），并在AdvOCR、SmuggleBench以及通用OCR基准（CCOCR、OCRBench、OCRBench‑v2）上进行评估；

**📈 对比分析**

相较于现有开源与闭源大模型，ArmorOCR在AdvSpot上的VQA准确率提升至约55.7%，IoU提升至63.3%，并在AdvOCR与SmuggleBench上分别取得最高平均准确率，且对通用OCR性能几乎无损；

**⚠️ 局限性**

局限性在于对抗样本生成仍基于合成与有限真实样本，且两阶段训练仍需较大算力；此外，模型对极端视觉编码与AI融合文本的鲁棒性仍有待进一步提升。

---

## 346. When Text and Numbers Disagree: Evidence Arbitration in Large Language Models

**arXiv ID:** 2608.20116 | [PDF](https://arxiv.org/pdf/2608.20116v1)

**作者:** Mattia Carletti `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在文本与数值证据冲突情境下的决策权衡，通过构建可控合成基准评估其对模态、时间新鲜度、可靠性和工具预测的优先级。

**💡 创新点**

提出了四维冲突设置的合成基准，系统分离模态、时间新鲜度、可靠性和预测来源，揭示LLM的系统化仲裁偏差。

**🔧 技术方法**

利用隐式风险轨迹生成数值序列与自然语言摘要，构造二元预测任务，并评估多款开源指令调优LLM的仲裁表现。

**📊 数据集**

使用自定义生成的合成数据集，包含2000条实例（四个冲突维度），不依赖真实世界数据。

**📈 对比分析**

通过在冲突与单模态基准上测量准确率，发现模型在文本/数值偏好、时间递增性和外部预测依赖上表现出系统化偏差，性能差异明显但不随规模单调提升。

**⚠️ 局限性**

实验仅基于合成数据且仅二元预测，未覆盖真实复杂场景，且忽略精细数值推理的挑战。

---

## 347. DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation

**arXiv ID:** 2608.20114 | [PDF](https://arxiv.org/pdf/2608.20114v1)

**作者:** Siyuan Ma `[一作]` (Tsinghua University), Qiaojun Yu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种针对四足移动机械臂的全身世界–动作模型 DECOWAM，能够在同一框架内同时预测未来视频帧和整个机械臂+基座动作。

**💡 创新点**

创新点包括：① 在冻结的 FastWAM 视觉先验上插入残差适配器实现参数高效迁移；② 通过“action‑equivalent future bottleneck”将未来视觉信息迁移到动作预测；③ 对基座速度与机械臂运动使用对抗式双latent解耦；④ 在视频分支中显式加入基座速度作为相机自我运动条件。

**🔧 技术方法**

使用的技术包括：FastWAM 视频-动作双分支网络、WAN 视觉解码器、ActionDiT 动作专家、残差适配器、梯度反转网络（GRL）进行解耦、条件流匹配训练、以及未来信息的教师-学生蒸馏。

**📊 数据集**

数据集：ARMDOG——同步记录四足机器人+6-DoF 机械臂的视频、位姿、动作和语言指令，包含 1,487 条完整任务，约 343,550 帧。

**📈 对比分析**

与 FastWAM、X‑VLA、π_0.5、Motus、X‑WAM、UVA 等基准比较，DECOWAM 在 8 帧 384×320 的 replay 上将视频 MSE 降低 15%（相当于 PSNR 提升 0.22 dB），动作 MSE 降低 22%；在 79 次真实机器人实验中，成功率 58.2%（FastWAM 57.0%），且在接近、抓取、搬运等阶段表现更好，平均完成时间比 FastWAM 快 16 秒。

**⚠️ 局限性**

局限性：依赖大规模预训练的 FastWAM 视觉先验，未在跨平台或更复杂任务上验证；残差适配器虽参数少，但仍需 25.95M 训练参数；基座-机械臂解耦在极端动态环境下可能不足；目前仅评估 23 条盒子任务，泛化性待进一步验证。

---

## 348. A Meta-Study on Replication Papers in Usable Security & Privacy

**arXiv ID:** 2608.20108 | [PDF](https://arxiv.org/pdf/2608.20108v1)

**作者:** Christian Mack `[一作]` (Karlsruhe Institute of Technology), Melanie Volkamer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统性梳理可用安全与隐私研究领域内的复制论文现状，首先分析了13个主要会议的Call for Papers（CfP）对复制研究的鼓励与规范，随后对2016–2025年间发表的论文进行检索、筛选，识别出24篇真正的用户研究复制论文，并使用Olszewski等人提出的复制框架对其进行八种类型的分类；最后通过对作者的问卷调查，揭示复制动机、障碍及社区对复制实践的期待。

**💡 创新点**

创新点在于：①首次在可用安全与隐私领域对复制论文进行系统性文献综述与分类；②将Olszewski等人针对计算机安全的复制框架进行适配并细化为“领域/方法/分析”三层，适用于用户研究；③基于调查结果提出了可复制性指南模板与CfP建议，旨在提升复制论文的透明度与可比性。

**🔧 技术方法**

技术与方法包括：系统性文献检索（ACM DL、IEEE Xplore、手工与脚本搜索），二元决策树复制框架，双人独立编码与互评的定性编码流程，以及问卷设计与统计汇总。

**📊 数据集**

数据集主要是：①从13个会议检索得到的24篇复制论文（2016–2025），②作者问卷收集的25份完整回复。

**📈 对比分析**

方法对比采用描述性统计：计算各CfP对复制的鼓励程度、复制论文数量与分布、分类类型占比以及动机代码频次；未进行实验性性能评估，而是通过归纳分析展示复制研究的现状与趋势。

**⚠️ 局限性**

限制包括：检索范围仅限13个会议和关键词“replicate/reproduce/repeat”，可能漏检非英文或未使用关键词的复制工作；复制框架在判定“同一问题”时存在歧义，导致部分论文分类困难；问卷响应率受限（仅25人），可能导致偏倚。

---

## 349. TrustRAG: Blockchain-Enhanced RAG via Committee-Based Credibility Scoring

**arXiv ID:** 2608.20097 | [PDF](https://arxiv.org/pdf/2608.20097v1)

**作者:** Baixiang Liu `[一作]` (Fudan University), Yuan Li `[通讯]` (Fudan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TrustRAG，一种基于专家委员会、区块链支撑的可验证检索增强生成（RAG）框架；

**💡 创新点**

创新点包括：①在文档注册阶段就完成隐私保护的零知识评分，①通过 Pedersen 约束 + Shamir 秘密共享实现安全多方加总；②用跨链哈希绑定代替递归全局证明，保持系统可扩展性；③提供可重放的排名验证，让任何客户端可独立复核检索结果。

**🔧 技术方法**

使用了 Groth16/PLONK 零知识证明、Pedersen 承诺、Merkle 证明、Shamir 隐私分片、MP‑SPDZ 安全求和、BLS 阈值签名、以及哈希绑定链式结构。

**📊 数据集**

实验基于原型实现，主要评测零知识证明生成时间、Shamir/阈值签名/聚合延迟以及整体链内计数开销，并未使用公开知识库或真实检索数据集，而是采用模拟投票与文档集合进行基准。

**📈 对比分析**

与传统 RAG 对比不在于生成质量，而是证明了系统在可验证性、隐私性和跨链完整性方面的优势：Groth16 证明时间 <2 s；链内计数延迟 ≈1.1 s；委员会聚合在 48 节点/100 投票下仍低于 300 ms；整体检索与绑定过程轻量级，仅几百毫秒。

**⚠️ 局限性**

局限性：①不保证检索到的文档内容本身的事实正确性；②依赖于委员会成员诚实阈值，若阈值被破坏可伪造分数；③投票和计数在链下完成，增加离线处理成本；④尚未在大规模真实数据集上验证性能和可用性。

---

## 350. HandMvNet: Real-Time 3D Hand Pose Estimation Using Multi-View Cross-Attention Fusion

**arXiv ID:** 2608.20093 | [PDF](https://arxiv.org/pdf/2608.20093v1)

**作者:** Muhammad Asad Ali `[一作]` (German Research Center for Artificial Intelligence), Didier Stricker `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 HandMvNet，实时多视角手部 3D 姿态与形状估计模型。

**💡 创新点**

创新点：使用多视角注意力融合机制，无需相机标定即可学习绝对 3D 几何，并兼顾实时性与高精度。

**🔧 技术方法**

技术：ResNet50/HRNet 特征提取，soft‑argmax 2D 关键点回归，点特征采样，三种位置编码（sinusoidal、joint、crop），跨视角多头注意力与自注意力，GCN 解码器。

**📊 数据集**

使用 DexYCB‑MV、HO3D‑MV、MVHand 三个公开多视角手部数据集进行训练与评估。

**📈 对比分析**

与 POEM、MvP、PE‑Mesh‑TR 等 SOTA 多视角方法对比，在 MPJPE_rel、MPVPE_rel、PA_J/PA_V 等指标上均取得更低误差，并在推理速度上领先。

**⚠️ 局限性**

局限：在样本量较小的数据集（如 HO3D‑MV）上泛化受限；模型依赖多视角图像，单视角性能不及主流单视角方法。

---

## 351. Using Zone-Disjoint Multi-Path Routing Algorithm for Video Transmission over Ah-Hoc Networks

**arXiv ID:** 2608.20148 | [PDF](https://arxiv.org/pdf/2608.20148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 352. Trustworthy mobile edge caching: a blockchain approach to mitigate malicious nodes and incentivize cache sharing

**arXiv ID:** 2608.20145 | [PDF](https://arxiv.org/pdf/2608.20145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 353. What Do Visualization Instructors Want Students to Learn? Introducing a Concept Inventory for Visualization Design

**arXiv ID:** 2608.20090 | [PDF](https://arxiv.org/pdf/2608.20090v1)

**作者:** Medina Lamkin `[一作]` (University of Washington), Leilani Battle `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过收集38份可视化课程大纲并访谈10位教师，设计并验证了一份14题的多项选择概念测验，用于评估可视化设计的核心概念与技能。

**💡 创新点**

创新点在于首次为可视化设计构建概念测验，融合教育研究中的概念库存方法，并强调常见误区的设计干扰项，提供了面向教师教学目标的全景性评估工具。

**🔧 技术方法**

使用了质性编码与主题聚类分析、教师访谈反馈迭代、以及概念测验题目设计与评估标准的构建等技术手段。

**📊 数据集**

主要数据集包括38份课程大纲（来自35位教师）以及10份教师访谈记录。

**📈 对比分析**

与现有的CALVI、AVEC、VLAT/Mini-VLAT等评测工具对比后发现本测验在覆盖的技能维度更宽泛，且补充了这些工具未涉及的关键可视化设计技能，尚未完成正式验证，性能评估待后续实验。

**⚠️ 局限性**

局限性包括：评估侧重广度而非深度，未能评测编程与开发能力；受样本主要为北美英语教师限制，可能存在文化与语言适用性问题；采用多项选择限制了对构建性任务的测量。

---

## 354. DARS: Dual-Level Credit Assignment RL with Structured Reasoning for Instruction-Based Image Editing

**arXiv ID:** 2608.20161 | [PDF](https://arxiv.org/pdf/2608.20161v1)

**作者:** Haoxiang Cao `[一作]` (South China Normal University), Chaoqun Wang `[通讯]` (South China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 DARS 的双级信用分配强化学习框架，用于改进规划-渲染两阶段指令驱动图像编辑系统。

**💡 创新点**

创新点在于：①通过多计划多渲染的 rollout 方差分解来实现跨模块信用分配和自适应课程；②引入四字段结构化规划输出与前缀门控奖励，以实现规划内部的细粒度信用分配；③利用 token 级优势重加权进一步提升规划的局部更新。

**🔧 技术方法**

主要技术包括：强化学习（GRPO 与 Flow‑GRPO）、结构化推理、前缀门控奖励、方差估计、课程学习、奖励模型（Gemini 3 Pro 与 Qwen3‑VL‑32B）以及多计划多渲染采样。

**📊 数据集**

训练数据由 10K 条样本组成，分别来自 THINKEDIT‑140K（5K）和 UniREdit‑Data‑100K（5K）；评估使用 KRIS‑Bench、RISE‑Bench、ImgEdit‑Bench、GEdit‑Bench‑EN 与 PICA‑Bench 五个基准。

**📈 对比分析**

与现有 11 种方法对比，DARS 在所有五个基准上均获得最高分，尤其在 KRIS‑Bench 与 RISE‑Bench 上提升显著（+8.57 与 +1.80 分），相较于对等后端、数据、奖励模型与预算的对照基线提升约 8.57、1.80、0.19、0.03、0.64 分。

**⚠️ 局限性**

局限性包括：每条样本需要执行 M×K（4×4）rollout，计算成本高；信用分配和奖励依赖于奖励模型的准确性；当规划与渲染同时失效时难以清晰拆解责任；框架尚未扩展至视频编辑或多轮交互场景。

---

## 355. The Generalized Random Access Problem for Linear Codes

**arXiv ID:** 2608.20152 | [PDF](https://arxiv.org/pdf/2608.20152v1)

**作者:** Anina Gruica `[一作]` (Technical University of Denmark), Ferdinando Zullo `[通讯]` (University of Campania Luigi Vanvitelli)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究DNA存储中随机访问的通用模型，给出一套从单个信息符号到全部信息符号恢复的统一期望公式，并对不同编码器（系统MDS、单纯形码、平衡准弧）进行分析，提供极值和界定。

**💡 创新点**

创新点在于提出基于请求符号集合大小的“通用随机访问”框架，得到一条从单点随机访问到全覆盖深度的连贯理论；给出通用期望公式、下上界以及对特定编码器的精确计算，揭示局部恢复与全局恢复的折衷关系。

**🔧 技术方法**

主要技术手段包括：有限几何计数、子集计数公式、占用问题（coupon‑collector）与秩统计的结合、双线性编码（双码）视角、对称性与自动群的利用，以及递归状态求解。

**📊 数据集**

由于是理论分析，本文没有使用具体实验数据集；所有结果均基于代数结构和组合计数给出的闭式或数值比较。

**📈 对比分析**

通过理论推导和数值示例（k=3时的平衡准弧、单纯形码、系统MDS），对比期望值。结果显示：平衡准弧在单符号随机访问时优于MDS；在两符号恢复时可与MDS相当或略优；而MDS在全恢复时达到最优；单纯形码在单点时等同于MDS，在多点时表现介于两者之间。

**⚠️ 局限性**

局限性包括：通用期望公式虽然可算，但需要大量计数，难以在高维下得到简洁闭式；目前仅在k=3及特殊编码器给出完整计算；未考虑非均匀抽样、错误校正或局部可恢复性等实际存储约束；尚未给出通用最优编码器的构造或证明。

---

## 356. Multi-Agent Orchestration with the Common-Sense Reasoning Capabilities of LLMs for Autonomous Driving

**arXiv ID:** 2608.20129 | [PDF](https://arxiv.org/pdf/2608.20129v1)

**作者:** Mehdi Azarafza `[一作]` (Hamm-Lippstadt University of Applied Sciences), Achim Rettberg `[通讯]` (Hamm-Lippstadt University of Applied Sciences)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结合LLM常识推理、PPO强化学习和PID控制的混合自动驾驶框架，并通过调度器协调四个安全等级的多智能体实现实时决策。

**💡 创新点**

创新点在于将LLM用作离线奖励函数细化与常识规则生成，避免其在实时控制中出现延迟与幻觉；同时通过多安全等级智能体与权重融合的决策仲裁器实现可追溯、ISO 26262 兼容的安全架构。

**🔧 技术方法**

使用技术包括多传感器感知（YOLOv11 + LiDAR）、PPO 强化学习、PID 控制、LLM（GPT‑5.2、Claude）推理、决策仲裁器权重融合与 ASIL 等级安全阈值。

**📊 数据集**

数据来源为 CARLA 高随机化仿真环境，包含多种天气、交通密度和道路情景，使用约 1,000 条仿真轨迹进行 RL 训练与评估。

**📈 对比分析**

通过对比基线 PID、单独 RL（曲线/速度）、混合 PID+RL，以及四个奖励版本 V0–V3 进行评估，V3 在轨迹跟踪误差最低、转弯稳定性最高、曲线覆盖最优，表现优于其他版本。

**⚠️ 局限性**

局限性包括仅在仿真环境中验证，缺乏真实道路测试；LLM 仅离线使用，未评估多模态实时推理的可扩展性；安全阈值为经验设定，缺乏形式化的安全证明。

---

## 357. ID-VTG: Image-Disambiguated Video Temporal Grounding

**arXiv ID:** 2608.20127 | [PDF](https://arxiv.org/pdf/2608.20127v1)

**作者:** Minghang Zheng `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Yang Liu `[通讯]` (Wangxuan Institute of Computer Technology, Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了图像消歧视频时序定位（ID-VTG）任务，利用参考图像和文本联合定位视频中特定实例的动作片段。

**💡 创新点**

创新点在于：①构建了专门暴露多实例歧义的两大数据集；②设计了双分支（快慢）结构，并引入可学习的 Compare Token 与 Depress Value 通过视觉引导的竞争机制实现实例消歧；③提出 Vision‑Assisted Disambiguation 模块实现视觉与文本信息的任务专属融合。

**🔧 技术方法**

使用 CLIP 视觉/文本编码、Transformer‑based 快慢分支、学习的 Compare Token/Depress Value、软最大竞争聚合、视觉匹配损失与文本导向定位头等技术。

**📊 数据集**

使用了两大新建数据集：IDVTG‑Gym（体操细粒度动作，204.1h，14.7k 查询）和 IDVTG‑InternVid（开放世界，302.7h，62.1k 查询），并对比公开 VTG 基线。

**📈 对比分析**

相较于 RaTSG、UVCOM、CG‑DETR、SnAG、ICQ 等基线，VGD‑Agg 在 IDVTG‑Gym 的 mIoU 由 54.83 提升至 66.58，在 IDVTG‑InternVid 与 Web 集合上分别从 41.70/16.53 提升至 51.21/21.99，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括：对图像质量的鲁棒性仍有限（亮度/分辨率下降时性能下降）；对多模态融合的依赖较大，需进一步简化模型；在极端视频模糊或长时序情况下可能仍出现误检。

---

## 358. OenoBench: A Wine-Domain Benchmark for Knowledge-Grounded Evaluation of Large Language Models

**arXiv ID:** 2608.20106 | [PDF](https://arxiv.org/pdf/2608.20106v1)

**作者:** Nikita Khudov `[一作]` `[通讯]` (StrategAI), Nikita Khudov (StrategAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了一个基于可信来源的葡萄酒知识问答基准OenoBench，包含 3,266 道多项选择题，覆盖六大知识支柱并按难度分层。

**💡 创新点**

创新点在于通过 LLM 驱动的事实提取、四轮审核、多模型、多策略生成，并严格追溯每条事实来源，从而消除传统基准的“泄漏”和“风格偏见”。

**🔧 技术方法**

使用的技术包括网页爬虫、语义解析、LLM 重写与审核、九人团体多代理审计、闭书可解性预筛和成本-准确率 Pareto 分析。

**📊 数据集**

数据集来源于 35 个可靠抓取器，涵盖 INAO、OIV、TTB、学术期刊、维基百科、Wikidata 等，包含 38,104 条原子事实。

**📈 对比分析**

通过 16 种前沿配置在 OenoBench 上评测，最优模型 o3 达到 83.6% 的准确率，说明 LLM 在葡萄酒领域的记忆与推理能力，并揭示闭书可解性与上下文推理的显著性能差距。

**⚠️ 局限性**

主要局限包括基准为快照版本需定期更新、闭书可解性评估器与人工评测高度不一致、以及 Anthropic 系列的自偏好仍未完全消除。

---

## 359. Structured Affinity for Unsupervised Visual Class-Incremental Memory in Deep Artificial Immune Networks

**arXiv ID:** 2608.20104 | [PDF](https://arxiv.org/pdf/2608.20104v1)

**作者:** Siphesihle Sithungu `[一作]` `[通讯]` (University of Johannesburg), Siphesihle Sithungu (University of Johannesburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何利用结构化、无梯度的免疫亲和力，使深度人工免疫网络（Deep AIN）在无重放、无梯度训练的条件下，在线学习视觉类别增量记忆。

**💡 创新点**

创新点包括：1）将B细胞视为局部视觉模板，使用ZNCC等空间友好的亲和力；2）在第一层生成空间响应图并传递给后续层，形成深度绑定谱；3）采用免疫更新（克隆、变异、抑制）而非反向传播；4）层间尺度自适应校准，解决深度层尺度不匹配问题。

**🔧 技术方法**

核心技术：ZNCC局部匹配、移位模板亲和力、响应图绑定谱、特征图Deep AIN结构、免疫克隆-变异-抑制更新、层级尺度自适应、外部线性/非线性探测器（LogReg、1NN、PCA+RBF‑SVM）。

**📊 数据集**

使用的视觉数据集为四个灰度图像基准：sklearn digits（8×8）、MNIST（28×28）、Fashion‑MNIST（28×28）和KMNIST（28×28），在每个数据集上进行10类增量学习。

**📈 对比分析**

对比方法包括：静态k‑means、Mini‑batch k‑means、全重训练k‑means、在线原型记忆、无重放MLP、重放MLP。指标为平衡准确率、初始类保留率和当前类准确率。Deep AIN在sklearn digits上达到0.978平衡准确率，MNIST、Fashion‑MNIST和KMNIST上与重放MLP相近（0.80–0.86），且无需标签更新或重放，显示出良好的在线记忆与分类能力。

**⚠️ 局限性**

局限性：1）局部过滤器可能产生模糊区分，导致线性或质心探测弱；2）B细胞/绑定谱维度随类增多增长，训练与推理成本上升；3）高层尺度和容量敏感，需要手工/自适应校准；4）仅在灰度、低分辨率数据集上验证，未验证彩色/复杂自然图像；5）缺乏有效的种群控制机制，可能出现冗余记忆。

---

## 360. Rewriting Ontology-Mediated Property Graph Queries into GQL

**arXiv ID:** 2608.20092 | [PDF](https://arxiv.org/pdf/2608.20092v1)

**作者:** Bianca Löhnert `[一作]` (University of Salzburg), Magdalena Ortiz `[通讯]` (TU Wien)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向属性图的本体驱动查询重写技术，将包含路径导航的查询在本体约束下重写为 Cypher 语句，并在 Neo4j 上执行。

**💡 创新点**

首次实现对包含两向正则路径（N2RPQ）且支持嵌套和数据值测试的查询进行完整重写，并提供跳过隐式事实的两阶段跳过机制。

**🔧 技术方法**

使用 DL‑Lite 变体（支持属性值测试）作为本体语言，结合嵌套两向正则路径自动机（N2RPQ），并通过推理得到 canonical model、跳转函数、重写规则和自动机交叉点。

**📊 数据集**

在 DBpedia（约 10.7 万节点、22.2 万关系）和 Montpellier Méditerranée Métropole（MMM）地铁管网数据集（约 10.3 万节点、30 万关系）上进行实验。

**📈 对比分析**

重写时间在 250 ms（DBpedia）至 600 ms（MMM）之间，查询执行频繁超时（最高 20 s），主要因生成的 Cypher 产生笛卡尔积或返回大量答案导致。

**⚠️ 局限性**

仅支持 join‑on‑free CN2RPQ，无法处理涉及非答案变量的复杂联接；重写后查询在 Neo4j 上性能不佳，缺乏对大规模查询的优化与可视化支持。

---

## 361. Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection

**arXiv ID:** 2608.20169 | [PDF](https://arxiv.org/pdf/2608.20169v1)

**作者:** Atsuyuki Miyai `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在自动化LLM代理 harness 优化过程中同时自适应选择验证任务的框架Task‑CoEvolve。

**💡 创新点**

创新点在于将验证任务的选择与 harness 进化联合优化，通过方差加权采样聚焦信息量大的任务，并用采样概率校正的全集估计使不同迭代的评分可比，从而显著降低评估开销。

**🔧 技术方法**

核心技术包括：方差加权任务采样、基于采样概率的Hájek或锚定差分全集估计、与Meta‑Harness元代理的协同迭代流程；实现代码可在GitHub公开。

**📊 数据集**

使用的基准数据集有：在线文本分类（LawBench、Symptom2Disease、USPTO‑50k）以及终端任务基准 Terminal‑Bench 2.1。

**📈 对比分析**

对比方法包括：完整集合搜索（Full Search）、固定子集（Naive）和每轮随机重采样（Random‑Resample）。实验显示，在仅 7% 评估预算下，Task‑CoEvolve 取得与 Full Search 相近的准确率；在 20% 预算下甚至超过 Full Search；在 Terminal‑Bench 2.1 上，使用 20% 预算即可得到与 Full Search 仅 1% 之差的表现，同时搜索成本降低 67–80%。

**⚠️ 局限性**

局限在于任务数固定，无法在评估过程中动态终止或扩展；未对不同 harness 代码结构的适用性做系统分析；对非常小或极大验证集的可扩展性尚未验证。

---

## 362. Chameleon: Robust Defense Against Tor Website Fingerprinting via Many-to-Many Traffic Morphing

**arXiv ID:** 2608.20160 | [PDF](https://arxiv.org/pdf/2608.20160v1)

**作者:** Yuwen Cui `[一作]` (University of South Florida), Guangjing Wang `[通讯]` (University of South Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种名为Chameleon的多对多随机流量变形防御系统，用以保护Tor网络中的网页浏览活动免受网站指纹攻击。

**💡 创新点**

其创新点在于引入高类内多样性低类间差异的变形候选、共享变形目标以及基于radix‑trie的同步机制，从而显著提升对防御感知（DAAE）攻击的鲁棒性。

**🔧 技术方法**

核心技术包括多对多随机流量变形、类内/类间多样性评估、随机映射、radix‑trie同步与归一化前缀匹配，并对抗训练与DAAE攻击进行了评估。

**📊 数据集**

使用了三个公开数据集（GTT23 及另外两个未命名数据集）进行实验。

**📈 对比分析**

与六种现有防御和五种攻击在闭/开世界环境下比较，针对对抗训练攻击准确率下降36.74%，带宽和时间开销分别下降34.12%和60.38%；在GTT23上DAAE攻击的F1分数被限制到35.19%，远低于Adaptive Tamaraw的88.22%；实测PT桥部署时时间开销仅增加16.25%。

**⚠️ 局限性**

局限性包括对手对防御机制未知但已知存在的假设、对极端攻击场景评估不足以及在真实网络环境中进一步验证和多平台部署仍需研究。

---

## 363. DPC-Net: Dual-Prior Collaborative Network for All-in-One Image Restoration

**arXiv ID:** 2608.20141 | [PDF](https://arxiv.org/pdf/2608.20141v1)

**作者:** Zhaokun He `[一作]` (Northwestern Polytechnical University), Qingsen Yan `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种双优先协同网络（DPC‑Net），实现统一模型对多种图像退化的高质量恢复。

**💡 创新点**

创新点在于：①将视觉‑语言模型（LLaVA）作为语义引导来学习退化‑语义耦合特征；②设计了退化语义调制模块（DSMM）与高频/低频特征融合的降解嵌入调制模块（DEM）；③在解码阶段引入知识库与双优先协同重建模块（DPCR），将低层视觉先验与退化语义先验协同利用。

**🔧 技术方法**

使用了 Vision‑Language Model (LLaVA) 进行跨模态语义监督；Restormer 作为主干网络；分层 Transformer 编码解码；分频域特征处理、注意力机制、知识库查询、可学习查询生成等技术。

**📊 数据集**

在多种退化任务的数据集上评估：去噪（BSD68, BSD400, WED），去雾（SOTS），除雨（Rain100L），去模糊（GoPro），低照度增强（LOLv1）等。

**📈 对比分析**

与目前所有 AiOIR 先进方法（如 AdaIR、R2R、VLU‑Net 等）在 PSNR/SSIM 上进行对比，DPC‑Net 在三任务和五任务平均水平均取得最高分，尤其在去雾任务上提升明显（PSNR+~0.6 dB）。

**⚠️ 局限性**

局限性包括：模型参数量较大（约27M），推理速度相对较慢；依赖预训练的 VLM，若 VLM 语义不匹配退化场景可能影响效果；对极端未知退化的泛化仍有待进一步验证。

---

## 364. Feature Evolution and Migration during Vision Transformer Training

**arXiv ID:** 2608.20134 | [PDF](https://arxiv.org/pdf/2608.20134v1)

**作者:** Joonas Järve `[一作]` (University of Tartu), Meelis Kull `[通讯]` (University of Tartu)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在 Vision Transformer 训练过程中按层和时间两维提取 CLS‑token 的稀疏特征，并可视化这些特征随训练的演化与迁移；

**💡 创新点**

创新点在于提出基于稀疏自编码器的 epoch‑by‑layer 特征跟踪框架，定义并量化特征迁移、稳定性、活跃寿命等指标，首次揭示 ViT 训练中特征迁移主要向前层倾斜且集中于早期；

**🔧 技术方法**

主要技术包括 BatchTopK 稀疏自编码器、Spearman 相关相似度（固定 SAE 与独立 SAE 组合）、层/时间特征迁移度量、热图可视化等；

**📊 数据集**

使用 ImageNet‑1k 及其 Balanced Superclass Mixed‑10（10 个超类）两个数据集；

**📈 对比分析**

通过计算迁移比例、稳定比例、平均活跃寿命、层分布熵等指标，并以热图、平均漂移数值等方式进行对比，结果显示迁移集中于早期训练，深层特征更稳定，浅层活跃寿命更长，两个数据集表现趋势相似；

**⚠️ 局限性**

局限性包括仅关注 CLS 令牌而未分析补丁流；只检测线性可解特征，可能遗漏非线性特征；仅在监督训练中验证，缺乏自监督或微调的探究；SAE 发现的特征跨时间稳定性未保证；相似度方法对零激活敏感；实验仅基于单一随机种子、架构和超参数。

---

## 365. SAE-Xplainers: Rule-Based Feature Interpretation for Extreme Earth Events

**arXiv ID:** 2608.20117 | [PDF](https://arxiv.org/pdf/2608.20117v1)

**作者:** Hugo Porta `[一作]` (EPFL), Devis Tuia `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

引入地理位置感知的稀疏自编码器 GeoTopK 以及规则基解释器 SAE-Xplainers，用于极端天气事件模型的可解释性。

**💡 创新点**

① 在稀疏自编码器中加入地理位置编码（FiLM）实现空间条件化；② 通过规则集合对 SAE 产生的稀疏特征进行全局解释，并检测特征吸收现象。

**🔧 技术方法**

位置感知 FiLM 适配器、k‑sparse Autoencoder、Skope‑Rules 规则学习、空间上下文采样、GeoTopK、SAE‑Xplainer、ClimaX/ViT 编码器、与 SHAP 等对比方法。

**📊 数据集**

SeasFire（8 天火灾预测）和 ClimateNet（热带气旋/气象河检测）。

**📈 对比分析**

与传统 TopK SAE 对比，GeoTopK 在 R²、R²_event、R²_worst、MSE、dead feature 等指标上均显著提升，尤其在极端事件和最差地区的 R² 降低幅度更小，dead feature 几乎为 0。

**⚠️ 局限性**

规则解释对样本分布偏移敏感，特征吸收判定依赖阈值，模型仍受限于可用输入维度与空间分辨率，且在高纬度地区特征混叠仍存在。

---

## 366. ODEONN: A Digital ODE Solver Architecture for Oscillatory Neural Networks

**arXiv ID:** 2608.20110 | [PDF](https://arxiv.org/pdf/2608.20110v1)

**作者:** Bram F. Haverkort `[一作]` (Eindhoven University of Technology), Aida Todri-Sanial `[通讯]` (Eindhoven University of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种可模块化、可扩展的FPGA基数字解算器架构ODEONN，用于模拟振荡神经网络（ONN）的动力学。

**💡 创新点**

首次提出支持复值耦合且采用高效正弦近似的通用数字ONN架构，通过硬件资源减半的正弦近似实现高性能。

**🔧 技术方法**

采用FPGA实现的ODE求解器、固定点量化、基于Kuramoto模型的数值积分、正弦近似算法。

**📊 数据集**

使用典型ONN应用实例：最大割（max-cut）、数独（Sudoku）、二值与灰度关联记忆。

**📈 对比分析**

与GPU上全精度32位浮点Python ODE求解器对比，量化误差导致性能损失低于2%，能耗-延迟乘积比GPU低45倍。

**⚠️ 局限性**

通信开销随核心数量线性增长，串行传输相位数据限制大规模并行扩展。

---

## 367. BeyondMasks: Evaluating Causal and Physical Consistency in Video Object Removal

**arXiv ID:** 2608.20107 | [PDF](https://arxiv.org/pdf/2608.20107v1)

**作者:** Yigit Ekin `[一作]` (Bilkent University), Aysegul Dundar `[通讯]` (Bilkent University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BeyondMasks基准，用于评估视频物体移除的因果一致性。

**💡 创新点**

创新点包括因果定义、配对视频数据集、以及VLM驱动的CORE评价指标。

**🔧 技术方法**

采用结构化视觉语言模型进行评估，并与传统像素指标（PSNR/SSIM/LPIPS）并行比较。

**📊 数据集**

使用了180条合成与实景对齐的视频序列，包含掩码、指令提示及干净背景参考。

**📈 对比分析**

对比九种主流移除方法，发现像素级指标相近但CORE显示大多数方法未能完全去除阴影、反射等后效，性能存在显著差距。

**⚠️ 局限性**

局限在于现有模型难以同时消除对象及其诱发的物理后效，CORE评估依赖预训练VLM的鲁棒性。

---

## 368. Reward-Guided Autoregressive Graph Generation for Efficient Multi-Agent Communication Topology Design

**arXiv ID:** 2608.20099 | [PDF](https://arxiv.org/pdf/2608.20099v1)

**作者:** Poomphob Suwannapichat `[一作]` (University of Luxembourg), Pascal Bouvry `[通讯]` (University of Luxembourg)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于奖励引导的自回归图生成器（RGA-Designer），用于在多智能体系统中生成稀疏且高效的通信拓扑；

**💡 创新点**

创新点在于将RLHF框架中的奖励模型迁移到图生成任务，通过学习同时考虑任务正确性和结构紧凑性来引导生成器；

**🔧 技术方法**

核心技术包括自回归图生成器、GraphSAGE图神经网络奖励模型、基于奖励的策略优化（GRPO）以及Best-of-N采样；

**📊 数据集**

在六个基准数据集上进行评估，涵盖数学推理（GSM8K、AQuA、MultiArith、SVAMP）、通用推理（MMLU）和代码生成（HumanEval）；

**📈 对比分析**

与五种基线（Vanilla、G-Designer、AgentPrune、AgentDropout、ARG-Designer）比较，RGA-Designer在保持或略微提升准确率的同时，平均减少20.5%的token消耗，除MultiArith外，token削减在所有基准上均具统计显著性；

**⚠️ 局限性**

主要限制包括依赖特定LLM（Qwen3-4B）且需真实答案标注，无法直接应用于无可验证答案的开放式任务；

---

## 369. Evidence-Gated Task and Motion Planning with Vision-Language Models

**arXiv ID:** 2608.20084 | [PDF](https://arxiv.org/pdf/2608.20084v1)

**作者:** Tsunehiko Tanaka `[一作]` (Waseda University), Edgar Simo-Serra `[通讯]` (Waseda University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种EAFG框架，先通过VLM生成探索子目标并由TAMP执行获取视觉证据，再根据证据判断是否继续规划、继续探索或停止执行。

**💡 创新点**

创新点在于：①将证据获取放在任务规划之前，利用VLM主动探索隐藏区域；②引入可行性门控机制，根据已获取证据决定是否继续规划或提前终止；③实现了在部分可观测环境下对缺失或不存在目标物体的自适应检测与避免。

**🔧 技术方法**

使用Vision‑Language Model（如GPT‑5.5、Gemini‑3.5‑Flash）、PDDLStream TAMP、图像拼接和文本证据状态更新等技术。

**📊 数据集**

在厨房仿真环境中测试，包含鸡汤制作任务，环境中设有可移动物体、存储空间和可操作关节。

**📈 对比分析**

与基线VLM‑TAMP比较，EAFG在三种场景下表现：①显式且存在物体时保持竞争力，提升配料步骤完成率；②未显式指令时将配料（盐、胡椒）成功发现，完整配方完成率从0.05/0.00提升到0.40/0.20；③缺失目标物体时能够提前停止，缺失物体尝试次数从4.00/2.40降至0.55/0.00，停止成功率从0.45/0.40提升到0.90/1.00。

**⚠️ 局限性**

主要局限是依赖TAMP完成子目标执行，若TAMP失败导致无法获取证据，门控可能基于不足信息做出决策；当前未处理探索过程中的执行失败恢复。

---

## 370. SABET-QA: Temporal Knowledge Graph Question Answering

**arXiv ID:** 2608.20083 | [PDF](https://arxiv.org/pdf/2608.20083v1)

**作者:** Brahim Touayouch `[一作]` (QuickSort Research), Dmitry Akulov `[通讯]` (QuickSort Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SABET-QA框架，用可微工作记忆实现多跳迭代推理，结合双向实体-时间打分和slot-aware上下文化编码进行时序知识图谱问答。

**💡 创新点**

创新点包括：1) 双向实体-时间打分机制解决头尾歧义；2) 可微工作记忆支持多跳递归更新，允许推理过程动态修正；3) 引入可选的粗粒度时间边界监督；4) 通过slot-aware注意力将问题语义与KG嵌入对齐。

**🔧 技术方法**

技术手段：预训练语言模型BERT/RoBERTa、TComplEx时序知识图谱嵌入、跨模态多头交叉注意、门控融合、可微工作记忆、双向打分、迭代跳、softmax期望向量更新。

**📊 数据集**

使用四个基准数据集：CronQuestions、Complex-CronQuestions、MultiTQ 和 TimeQuestions。

**📈 对比分析**

与LM_TKGQA、EmbedKGQA、CronKGQA、TempoQR、SubGTR等基线对比，SABET-QA在所有四个数据集上均取得最高 Hits@1/Hits@10，尤其在复杂多跳和时间预测任务上提升显著；在 Complex-CronQuestions 的 Hard 版本中 Hits@1 达到 0.954，明显优于 TempoQR-Hard 0.914。

**⚠️ 局限性**

局限性：依赖预训练的 TComplEx 嵌入和冻结的模块，迁移到新域或动态图结构时适应性受限；需要实体和时间槽的准确提取，NER 或实体链接错误会显著影响性能；多跳迭代增加计算成本；对噪声或不完整的问句/图数据鲁棒性有限。

---

## 371. A Standardized Framework for Machine Learning in Power System Protection

**arXiv ID:** 2608.20181 | [PDF](https://arxiv.org/pdf/2608.20181v1)

**作者:** Julian Oelhaf `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Siming Bayer `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一套七步标准化评估框架，用以系统化描述电力系统基于机器学习的保护任务的任务定义、物理范围、观测性、时间窗口、目标与样本有效性、训练验证协议及评估输出；并在PROTECT-90电磁瞬态基准上实施实例，展示了分类（fault classification）与定位（fault localization）的性能差异、时间、观测性、测量失真与负载阻抗分布对模型表现的影响；

**💡 创新点**

创新点在于将评估设计作为科学贡献的一部分，将保护工程与机器学习的验证要求整合为可执行的、可复现的七维评估结构，提供统一的报告包，使不同研究可直接比较并可审计；

**🔧 技术方法**

使用的技术包括：多层感知器（MLP）、梯度提升树（GB）、K近邻（KNN）、岭回归、一维卷积神经网络（CNN‑1D）、预训练时间序列 Transformer（MOMENT‑1‑Large）等；

**📊 数据集**

数据集为公开的PROTECT‑90电磁瞬态基准，包含90 kV双线拓扑的9022个仿真事件，采样率6400 Hz，提供三相电压/电流波形；

**📈 对比分析**

比较方法：采用分组交叉验证（按仿真事件分组），对比分类任务宏观F1、定位任务均方根误差（MAE）以及结构诊断、鲁棒性和推理时延；结果显示MLP在20 ms窗口下分类宏观F1≈0.991，定位MAE≈10 %线长；延长窗口提升定位性能但分类差异不大；观测性降低对定位影响显著，分类影响有限；传统两端距离定位在清洁数据下性能优于学习模型；

**⚠️ 局限性**

局限性包括：评估仅在单一拓扑与仿真环境下验证，未覆盖多线网、分布式发电、非同步测量等实际复杂情况；测量失真实验未涵盖通信延迟、数据丢包；模型参数选择未进行完整搜索，结果对超参数敏感；

---

## 372. PelviNeXt: A Modality-Agnostic Hybrid Network for Pelvic Imaging in Women's Health

**arXiv ID:** 2608.20144 | [PDF](https://arxiv.org/pdf/2608.20144v1)

**作者:** Siam Tahsin Bhuiyan `[一作]` (Independent University, Bangladesh), M Ashraful Amin `[通讯]` (Independent University, Bangladesh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种模态无关的PelviNeXt网络，用于女性盆腔影像（超声和X射线）的疾病分类；

**💡 创新点**

创新点在于将密集卷积特征提取、层次CBAM、多尺度融合以及talking-heads多头自注意力整合为统一架构，并对公共数据集进行完整的完整性审核与去重；

**🔧 技术方法**

技术包括DenseNet风格的稠密特征提取器、层次化CBAM、Multi‑Scale Fusion Module (MSFM) 与 Talking‑Heads Multi‑Head Self‑Attention (TH‑MHSA)，以及感知哈希去重；

**📊 数据集**

使用了PCOSGen超声数据集（去重后225张）和PXR150 X‑ray数据集（150张）；

**📈 对比分析**

在5折交叉验证下，PelviNeXt在去重PCOSGen上平均准确率92.00%、AUROC 0.9051，显著优于ViT-B/16、ResNet‑101和DenseNet‑169；在PXR150上准确率87.33%、AUROC 0.8920，超越了此前最高的Patch Ensemble，表现最突出的是特异性提升；

**⚠️ 局限性**

主要局限是数据集规模极小（去重后PCOSGen仅225张，PXR150 150张），导致统计功效受限，且缺乏跨数据集、跨医院的验证。

---

## 373. RMWorld: Task-Aware Radio World Models with Value-of-Information Guided Multi-Trial Learning for Multi-UAV Communication Control

**arXiv ID:** 2608.20126 | [PDF](https://arxiv.org/pdf/2608.20126v1)

**作者:** Xiucheng Wang `[一作]` (Xidian University), Junxi Huan `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出RMWorld框架，针对多UAV通信控制中的不完美无线世界模型（radio WM）和多试验学习问题，通过价值信息引导的通道证据分配与任务门控的多试验选择实现自适应调度与学习；

**💡 创新点**

创新点在于：①构造任务积分的通道证据评估，得到精确的单标记后验任务风险下降公式；②在多试验层面设计任务门控的对数行列式子目标，实现子模性质的贪心近似；③引入冲突投影与固定批量验证，兼顾梯度一致性与安全性；并证明对应的误差下降、子模性质与一阶非干涉性；

**🔧 技术方法**

主要技术包括：贝叶斯残差模型、局部线性化的后验协方差更新、A-Optimal（方差）与D-Optimal（行列式）实验设计、子模函数的贪心优化、冲突投影（gradient surgery）、固定批量验证、AdamW优化与线性搜索；

**📊 数据集**

使用的数据集包括：1）基于3GPP TR 36.777 UMa‑AV 公式的仿真场景（100对），2）DeepMIMO O1 远射频场景（30对），3）随机特征残差编码的神经特征实验（30对）；

**📈 对比分析**

与多种基线进行配对对照：在通道层面有随机、空间D‑Opt、Ensemble Variance、Task‑Weighted Variance、Gradient D‑Opt；在试验层面有随机多试验、Action Entropy、Ensemble UCB、Gradient Information；RMWorld在任务加权RMSE上平均提高≈4.9%（0.949 bit/s/Hz vs 0.998），在高负载DeepMIMO下的堆积量减少约0.967（比Ensemble UCB低），但相对其他基线耗时与模型调用显著增加；

**⚠️ 局限性**

局限性包括：①仅在模拟/公式场景下验证，缺乏实测渠道；②模型偏差与阴影场的残差可能导致后验风险下降不等价于真实误差下降；③计算成本高，尤其是通道证据更新与多试验选择；④仅在固定任务与预算设置下验证，缺乏对更大规模UAV群和动态阻塞环境的通用性验证；

---

## 374. Privacy-Preserving Detection of Rare Disease-Associated Cell Subsets via Secure Multi-Party Computation

**arXiv ID:** 2608.20118 | [PDF](https://arxiv.org/pdf/2608.20118v1)

**作者:** Ş. Selcan Magara `[一作]` (University of Tübingen), Mete Akgün `[通讯]` (University of Tübingen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

实现了在安全多方计算框架下的CellCnn模型，实现端到端的训练与推理，保护单细胞测量数据的隐私；

**💡 创新点**

在保留ReLU激活、偏置项和多分类/回归头的同时，采用加法秘密共享实现隐私保护，显著提升了相较于仅使用同态加密实现的PriCell的表达能力和准确率；

**🔧 技术方法**

使用2/3加法秘密共享、Beaver三元组、固定点数表示以及多项式近似的sigmoid和tanh函数，配合mini‑batch SGD+动量优化器；

**📊 数据集**

在CMV/NK、AML三分类以及AML MRD回归三个公开单细胞数据集上进行实验；

**📈 对比分析**

与明文CellCnn对比，MPC实现的多细胞输入准确率与基线几乎相同（CMV：0.727 vs 0.721，AML：0.935 vs 1.000），在PriCell上显著优于其性能（CMV多细胞：0.727 vs 0.633，AML多细胞：0.935 vs 0.898），回归任务中Pearson r为0.98、MAE为1.0pp，接近明文基线；

**⚠️ 局限性**

局限包括仅在半诚实、非协同的诚实多数模型下安全，未考虑恶意攻击；输出泄漏不在MPC威胁模型内；基准数据集规模较小且回归任务采用合成稀有细胞比例，需在更大真实临床数据上验证。

---

## 375. Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking

**arXiv ID:** 2608.20087 | [PDF](https://arxiv.org/pdf/2608.20087v1)

**作者:** Tao Huang `[一作]` (Noitom Robotics), Jiangmiao Pang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套可自适应的运动规划与跟踪框架 AdaPT，使仿人机器人能够在真实环境中以专业球员风格打网球。

**💡 创新点**

创新点在于将规划与跟踪解耦，并引入速度自适应机制（训练时随机化执行速度，规划时预测执行速率）以缓解仿真到真实的误差积累；同时在服务动作中加入关键帧监督和投球奖励，使得机器人能在不使用运动捕捉的情况下完成完整投球-击球过程。

**🔧 技术方法**

主要技术包括：基于广播视频的运动重建与标注（GVHMR+GMR），MVAE 运动生成器，PPO 训练的高层规划器与低层跟踪器，残差跟踪器与速度适配器，YOLO+立体视觉的球体检测与轨迹预测，HTC VIVE 追踪器实现室外定位。

**📊 数据集**

使用了公开的广播视频数据（Rafael Nadal、Roger Federer、Novak Djokovic 3 位球员共 3,000+ 短片）以及专业运动捕捉数据（Mr. Black 等），对这些数据进行 SMPL 重建、运动校正后得到 3D 动作集。

**📈 对比分析**

与 RL-Scratch、AMP、DeepMimic、PULSE、NCP、Vid2Player3D 等基线对比，AdaPT 在仿真与真实两端均实现了更高的击球成功率、更低的 Fréchet Inception Distance（运动风格相似度）以及更小的关节加速度误差；在真实机器人 Unitree G1 与 Dobot Atom 上实现了无标记摄像头环境下的完整发球与回球。

**⚠️ 局限性**

主要限制包括：仍需依赖运动捕捉或高质量多摄像头视觉来获取球与机器人状态；在摄像头噪声、球弹跳预测误差上仍存在较大误差；手腕执行器性能有限，影响极端发球的稳定性；未来需要更鲁棒的无标记定位与更灵活的学习算法。

---

## 376. Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation

**arXiv ID:** 2608.20256 | [PDF](https://arxiv.org/pdf/2608.20256v1)

**作者:** Gijs Kassenaar `[一作]`, Vincent François-Lavet `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

N/A

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

## 377. Electronic Navigational Chart Change Classification

**arXiv ID:** 2608.20218 | [PDF](https://arxiv.org/pdf/2608.20218v1)

**作者:** Jacob Arndt `[一作]` (Oak Ridge National Laboratory), Alexandre Sorokine `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于机器学习的电子航图（ENC）更改自动分类方法，利用空间上下文和属性编码将矢量更改转化为结构化特征进行分类。

**💡 创新点**

创新点在于构建了空间上下文编码器和属性嵌入编码器，首次将几何邻域信息与属性文本嵌入结合到ENC更改表示中，并通过超参数调优提升模型性能。

**🔧 技术方法**

主要技术包括 One‑Hot 编码、图神经网络风格的邻域聚合、DistilBERT 对属性文本的嵌入，以及 XGBoost 及其它基线模型（LR、RF、MLP、ResNet）进行二分类。

**📊 数据集**

使用了两个操作数据集：1）基于规则自动标注的 Critical/Non‑Critical 数据集（约 140k 条更改）；2）人工审核的 Eyes‑On 数据集（约 9k 条更改），共 1,308 张 ENC 对。

**📈 对比分析**

通过五折交叉验证比较，XGBoost 在完整编码下达到了 90%~94% 的准确率，超越了传统规则、简单投票基线和其它基线模型，说明空间与属性编码对分类极为有效。

**⚠️ 局限性**

局限在于使用了简单的 500 m 半径邻域聚合与静态 One‑Hot 编码，未探索更复杂的空间图学习方法，且对极少数类或罕见更改场景的泛化能力仍待验证。

---

## 378. 4DAnyone: Create Anyone in 4D from a Casual Monocular Video

**arXiv ID:** 2608.20335 | [PDF](https://arxiv.org/pdf/2608.20335v1)

**作者:** Yudong Jin `[一作]`, Yinghao Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

该论文主要讨论多视角与3D计算方法的相关概念与技术框架。

**💡 创新点**

由于缺乏具体细节，创新点尚不明确，可能聚焦于多视角融合或3D重建的新视角。

**🔧 技术方法**

所采用的技术未在文中具体列出，推测涉及多视角数据处理、3D建模与渲染算法。

**📊 数据集**

文档中未提及使用的数据集，可能使用常见的多视角或3D数据集（如KITTI、ShapeNet等）。

**📈 对比分析**

没有提供实验或方法比较的结果，无法评估性能。

**⚠️ 局限性**

主要局限在信息不足，缺乏实验验证、详细方法与结果展示。

---

## 379. G-CARL: Grounded Checklist-Aligned Reward Learning for Patient-Oriented Medical Report Interpretation

**arXiv ID:** 2608.20331 | [PDF](https://arxiv.org/pdf/2608.20331v1)

**作者:** Shiao Xie `[一作]` (Baidu Inc.), Xiandong Li `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了面向患者的医学报告解释任务PMRI，并开发了基于检索与动态清单的强化学习框架G‑CARL；

**💡 创新点**

通过将奖励拆分为可检索验证的命题奖励与实例特定加权清单奖励，动态生成并权重化清单，兼顾医学事实性与患者沟通需求；

**🔧 技术方法**

使用检索驱动的命题验证、MLLM生成并人工完善的加权清单、GRPO强化学习、结构化推理格式奖励，以及多模态LLM骨干（如Qwen3‑VL、InternVL3）；

**📊 数据集**

构建了真实世界的MMedReport基准数据集，构建了包含药物标签、教材和临床指南的多源医学知识库，并在CMB公开基准上进行跨域评测；

**📈 对比分析**

与通用与专业LVLM、SFT、MLLM‑as‑a‑Judge及多种RL方法进行对比，G‑CARL在医学准确性、需求满足度、表达质量、命题精度和清单召回率等主客观指标均显著优于基线，并获得临床医师与普通患者的偏好优先；

**⚠️ 局限性**

局限在于依赖人工生成的清单可能难以规模化、对知识库覆盖范围敏感、未验证对更复杂多模态输入的鲁棒性，以及缺乏实时部署与长期安全评估。

---

## 380. AI4AI-Bench: Benchmarking LLM Agents in Algorithmic Design for Recursive Self-Improvement

**arXiv ID:** 2608.20318 | [PDF](https://arxiv.org/pdf/2608.20318v1)

**作者:** Yizhe Chi `[一作]` (Navers Lab, Einsia.AI), Qinhuai Na `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个专门用于评估递归自我改进（RSI）中算法设计层的基准，提供十个冻结的研究仓库、4小时编码探索窗口和12小时验证训练窗口，确保评估仅衡量训练算法的改进而非数据或系统调优。

**💡 创新点**

创新点在于将算法设计层从传统的系统或数据优化中独立出来，使用统一的“代码修改＋后续重跑”流程，对提交的代码进行分类，区分训练流程的执行层和学习算法层；并通过可复制的、可扩展的评估仪器实现对算法改进的客观衡量。

**🔧 技术方法**

技术主要包括：自动化评估框架（在同一硬件、预算与评估器下重跑代码）、对代码差异的自动分类（使用语言模型识别修改范围）、多轮实验配置（六种系统、五种推理努力级别）以及分数映射函数将不同任务的度量统一到0-1区间。

**📊 数据集**

使用的“数据集”是十个研究仓库自带的训练数据或任务数据，例如监督微调、agentic RL、on‑policy distillation、BR‑T reward modeling、preference optimization、diffusion RL、机器无记忆、图形扩散、权重平均和单次剪枝等；每个仓库都有自己的基础模型和轻量代理指标。

**📈 对比分析**

对比方法是把提交代码与原始仓库代码在相同硬件与预算下进行直接比较；得分采用映射后的σ函数，0.1对应原始算法，1.0为理论最优。实验结果显示，在29种系统配置下，平均分仅为0.166，最佳系统为0.250；大多数提交只改动执行层，只有约44%触及学习算法层，后者平均得分显著高于前者。

**⚠️ 局限性**

局限性包括：评估仍受限于预设的代理指标和固定预算，无法覆盖所有可能的算法设计空间；大多数提交未能触及学习算法层，说明当前代理仍缺乏诊断和推理能力；评分尺度与真实任务性能的映射仍相对粗糙，且未考虑跨任务迁移或更复杂的模型。

---

## 381. Projecting BrowseComp-Plus onto ClimbMix: Toward More Realistic Corpora for Agentic Search

**arXiv ID:** 2608.20317 | [PDF](https://arxiv.org/pdf/2608.20317v1)

**作者:** Sahel Sharifymoghaddam `[一作]` (University of Waterloo), Jimmy Lin `[通讯]` (University of Waterloo)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在论文中作者开发了一套投影管线，将原有基准测试的问答迁移到更大、更自然的Web文本语料库中，并仅保留能够在新语料中完全支持的问答。

**💡 创新点**

创新点在于提出了数据集无关的跳跃（hop）拆分与语料投影方法，能够在不修改原问题的前提下实现检索难度的显著提升，同时提供了自动与人工双重验证机制。

**🔧 技术方法**

技术上使用了基于BM25的检索（Pyserini）结合大语言模型（如GPT‑5.5、GPT‑5.6）进行跳跃拆分、检索、自动验证，并通过多阶段验证（自动、独立agent、人类审核）确保所有跳跃都有证据。

**📊 数据集**

使用的数据集包括原始的BCP问答集（830条）以及NVIDIA发布的400B‑token、553M 文档的NEMOTRON‑CORPUS，用以构建检索索引和投影。

**📈 对比分析**

对比方法：在原先的精心构造的100K文档语料上，强大agent在57条问题上达86%准确率；迁移到NEMOTRON后准确率仅下降至80.7%，但检索召回率从84.3%降至21.4%，检索调用次数增加63%。

**⚠️ 局限性**

局限性包括投影成功率低（仅57/830），依赖模型对跳跃拆分与证据判定的非确定性，使用的检索仅为BM25，缺乏密集检索或混合索引，且未公开跳跃拆分细节。

---

## 382. Inject, Align, Recover: Staged Post-Training for Retrieval-Free Document Knowledge Internalization

**arXiv ID:** 2608.20281 | [PDF](https://arxiv.org/pdf/2608.20281v1)

**作者:** Qian Kou `[一作]` (Beijing Academy of Artificial Intelligence), Hua Zhou `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对检索自由的文档问答任务，作者提出了三阶段后训练框架Inject‑Align‑Recover（IAR），先将固定文档集合转化为可参数化知识，再对模型进行问答对齐，最后通过模型合并恢复通用能力。

**💡 创新点**

创新点在于：①将文档注入拆分为多种重构目标（Continuation、Rewrite、Instruction‑Conditioned Reconstruction）实现更密集的知识学习；②把QA对齐与通用能力恢复分离，避免单一步骤导致的灾难性遗忘；③采用后期模型合并（SLERP、Task Arithmetic、TIES、DARE）实现对域内精度与通用性能的权衡。

**🔧 技术方法**

主要技术包括：大模型（Llama‑3.2‑3B、Phi‑4‑mini、Qwen3‑4B、SmolLM3‑3B）上的后训练；对文档进行重构任务的自监督学习；仅用答案的QA微调；以及多种权重空间融合算子进行恢复。

**📊 数据集**

使用的公开数据集是Common Corpus（CC）和Common Corpus‑In‑Context（CCI），各自包含约14k训练/750测试或10k训练/575测试的文档衍生问答对。

**📈 对比分析**

与传统SFT、BudgetMatch、LoRA、SDFT、Replay、FAPM以及CPT+SFT对比，IAR在所有四大模型族上在域内问答准确率提升了约2.8–7.7个百分点，同时在IFEval、MMLU、MSBench等通用基准上平均提升12–24个百分点，且通过后期恢复保持了大部分域内优势。

**⚠️ 局限性**

局限性包括：①最佳注入比例因模型和语料差异而变化，需要实验调优；②恢复阶段仍不能完全恢复所有通用能力，域内精度与通用性能仍存在权衡；③多阶段训练与合并过程较复杂，实际部署时需额外验证；④在某些高基线模型（如Phi‑CCI）中域内提升有限。

---

## 383. Ultra-High-Definition Restoration Transformers with Correlation Matching Transformation

**arXiv ID:** 2608.20263 | [PDF](https://arxiv.org/pdf/2608.20263v1)

**作者:** Cong Wang `[一作]` (Sun Yat-Sen University), Xiaochun Cao `[通讯]` (Sun Yat-Sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为UHDformer++的通用Transformer框架，用于解决多种超高清(UHD)图像修复任务（如低光增强、去雾、去模糊、去雨、去雪），并通过四个协同空间（高分辨率空间HR、低分辨率空间LR、超分辨率空间SR、低高融合重建空间LHFR）实现高效恢复。

**💡 创新点**

核心创新包括：① Feature‑Refined Correlation Matching Transformation (FR‑CMT)，通过最大池化与均值池化融合的高分辨率特征与低分辨率特征的通道级相关匹配，挑选最具代表性的通道来更新LR特征；② Adaptive Channel Modulator (ACM)，对多层高分辨率特征进行通道级自适应调制，只传递任务相关信息；③ 在Transformer中引入Dynamic Tanh (DyT)归一化、改进的Feed‑Forward网络 (CMTFN) 以及SR空间以提升效果。

**🔧 技术方法**

技术手段主要为：多尺度ConvNeXt编码、像素重排下采样/上采样、跨尺度特征匹配的Correlation Matching Transformation、基于Transformer的注意力与前馈网络、动态Tanh归一化、Gated Feature Refinement等。

**📊 数据集**

使用了五个UHD级别数据集：UHD‑LL（低光）、UHD‑Haze（去雾）、UHD‑Blur（去模糊）、UHD‑Rain（去雨）、UHD‑Snow（去雪），每个数据集约2000-3000张训练样本和200-300张测试样本；此外还在通用任务上使用了LOL、SOTS‑ITS、GoPro等数据集进行交叉验证。

**📈 对比分析**

在所有五个UHD修复任务中，UHDformer++在PSNR/SSIM/LPIPS等指标上均优于最新SOTA方法（如UHDFour、DehazeFormer‑B、FFTformer等），且参数量和 FLOPs 减少至少 86%，并在大多数情况下实现了近 50 倍的参数压缩。对通用任务（LOL、GoPro）亦取得相当或略优的性能，显示出较强的泛化能力。

**⚠️ 局限性**

局限性：在通用图像去模糊等需要高分辨率细节捕捉的任务中，由于大部分计算在 8×8 低分辨率空间完成，参数量低，无法满足复杂空间变化的需求；对大规模噪声/模糊模式的建模能力受限，导致在某些通用数据集上表现不及专用大型模型。

---

## 384. DICS: Data-Informed Centroid Splitting for Decision Tree Classifiers

**arXiv ID:** 2608.20258 | [PDF](https://arxiv.org/pdf/2608.20258v1)

**作者:** MD Saifur Rahman Mazumder `[一作]` (University of Texas at El Paso), Feng Yu `[通讯]` (University of Texas at El Paso)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了数据驱动的质心分割（DICS）方法，通过聚类预先生成候选阈值集合，显著减少决策树、随机森林和梯度提升树的分裂搜索空间并加速训练。

**💡 创新点**

创新点在于将K‑means聚类作为先验生成压缩阈值集合，并证明在大样本条件下该方法的分裂收益与全搜索相当，从而实现速度提升而不显著牺牲准确率。

**🔧 技术方法**

采用Mini‑batch K‑means、方差加权阈值计算、阈值筛选以及与DT、RF、XGBoost/LightGBM兼容的改造技术。

**📊 数据集**

实验数据包含多类合成数据（混合线性、非线性特征）以及六个公开数据集：Helena、Spambase、Santander、CIFAR‑10、MNIST、Fashion‑MNIST。

**📈 对比分析**

与传统DT、BDTKS、RF、XGBoost/LightGBM对比，DICS在决策树、随机森林和梯度提升中分别实现8–30倍的加速，准确率波动≤0.02；在合成数据上CGCT、CGRF、FastC‑GBM的速度提升分别为8–27倍。

**⚠️ 局限性**

仅针对分类任务，尚未扩展到回归；在极大规模数据（如Santander）中存在内存管理瓶颈；在极高维稀疏数据上效果仍待验证。

---

## 385. Algorithms, Complexity, and Entropy of the Bernard-Letac Fair-Sampling Construction

**arXiv ID:** 2608.20234 | [PDF](https://arxiv.org/pdf/2608.20234v1)

**作者:** Claude Gravel `[一作]` `[通讯]` (Toronto Metropolitan University), Claude Gravel (Toronto Metropolitan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并扩展Bernard–Letac公平采样方法，给出5种实现算法并推导期望停止时间的闭式产品公式，进一步分析复合模数情况并改进二元案例的分配成本；

**💡 创新点**

引入Rényi熵视角得到精确期望抽样次数表达式，证明信息下界永不达到，构造七状态自动机快速计算模2第一通道核，并完成单调性与效率极值分析；

**🔧 技术方法**

利用组合数模p理论（Kummer、Lucas）、p进收敛、信息理论（Rényi、Shannon熵）、自动机/转移矩阵、Wald等式以及数值实验等技术；

**📊 数据集**

以离散概率分布（如二元 (0.7,0.3)、(0.9,0.1) 等）为实验和理论验证对象，无使用真实数据集；

**📈 对比分析**

与Von Neumann+拒绝、Von Neumann+Lumbroso 等经典方法比较，在二元案例中Bernard–Letac算法在期望抽样次数上优于这两种方法；效率随模数p变化呈单峰，达到最高点时明显优于其他方法；

**⚠️ 局限性**

对复合模数的期望值尚无闭式表达式，自动机仅在模2（二元）可实现，通用p模多状态系统尚未证明；自适应边界、p进极限的熵解释等仍为开放问题。

---

## 386. Probabilities beyond Belnap-Dunn logic: dealing with gaps, gluts and reliability

**arXiv ID:** 2608.20228 | [PDF](https://arxiv.org/pdf/2608.20228v1)

**作者:** Verónica Borja Macias `[一作]`, Alejandro Hernández-Tello `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于六值逻辑（L6）和twist结构的概率更新框架，兼顾单值与六值Jeffrey更新与Bayesian更新；

**💡 创新点**

创新点在于将经典Jeffrey更新推广到六值逻辑，提供语义与句法两种一致的更新定义，并揭示六值更新非可交换性；

**🔧 技术方法**

使用了六值非单值逻辑的语义结构（twist结构）、矩阵Hadamard除法、向量点积等数学工具；

**📊 数据集**

文中未使用具体外部数据集，主要采用理论构造的示例模型与赋值集合；

**📈 对比分析**

与经典四值逻辑和传统贝叶斯更新进行理论比较，结果显示六值框架在表达可靠性与冲突时具有更细粒度的描述，理论上性能优于二值方法；

**⚠️ 局限性**

局限在于更新规则复杂度较高，且六值Jeffrey更新不具可交换性，实际应用需进一步验证其可扩展性和可计算性。

---

## 387. Daedalus-150M: A Convolution-Attention Hybrid Designed for CPU Inference

**arXiv ID:** 2608.20210 | [PDF](https://arxiv.org/pdf/2608.20210v1)

**作者:** Christos Koutsiaris `[一作]` `[通讯]`, Christos Koutsiaris

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在单用户、单线程 CPU 推理场景下，设计并训练了一种混合卷积–注意力的 150M 参数小型语言模型 Daedalus-150M

**💡 创新点**

核心创新是将 Transformer 的 2/3 注意力层替换为固定长度的深度卷积层，从而消除大多数层对随时间增长的 KV 缓存读取，显著提升长上下文解码速度

**🔧 技术方法**

使用 4‑bit 量化、分组查询注意力（GQA）、深度可分离卷积、嵌入权重共享、较小的前馈维度以及基于 CPU 内存带宽和缓存成本的层比例设计

**📊 数据集**

训练数据为 16.93B 词汇的 10 源英语混合语料（包括 FineWeb‑Edu、Stack‑Edu、FinePDFs‑Edu、FinePhrase 等），总计 59.9B 令牌，重点筛选高质量推理与常识文本

**📈 对比分析**

与同参数数的全注意力密集模型（24 层）、以及 MobileLLM‑125M、GPT‑2‑124M 等同规模或更大规模模型进行对比；在五任务平均精度上达 47.31（超过 42.20 基准），解码速度在 2048‑token 上比全注意力模型快 1.76×，比外部 135M 对手快 2.08×

**⚠️ 局限性**

主要限制包括：4‑bit 量化后仍有约 6% 语言模型困惑度损失；约 48% 的卷积通道无效且无法在导出后裁剪；词表尺寸过大（49k）导致嵌入占比 23%；实验仅使用单个随机种子，缺乏多种种子验证；未实现量化感知训练，导致量化误差未消除

---

## 388. RoMAN-Flow: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation

**arXiv ID:** 2608.20208 | [PDF](https://arxiv.org/pdf/2608.20208v1)

**作者:** Shaoxuan Wang `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RoMAN-Flow 框架，将自回归归一化流（AR-NF）用于离线强化学习和机器人操控，解决 AR-NF 的采样开销和部署延迟；

**💡 创新点**

① 在离线 RL 中引入 NF‑IQL，利用优势加权的似然优化，避免从当前策略采样；② 通过一阶学生模型实现一次性动作生成，显著降低推理延迟；

**🔧 技术方法**

自回归归一化流（AR‑NF）、Imitation Learning、Implicit Q‑Learning（IQL）、一阶策略蒸馏、Transformer、预训练视觉‑语言编码器；

**📊 数据集**

MetaWorld‑MT50、LIBERO、RoboMimic MH、Franka–XHand 真实机器人平台；

**📈 对比分析**

在多任务、长序列、跨域和真实机器人任务上，与 Diffusion、π_0、π_0.5 等大型 VLA/流模型以及 SERNF 等基线相比，RoMAN‑Flow (NF‑IQL) 在 MetaWorld、LIBERO、RoboMimic 及真实机器人任务上取得了最高或竞争力的成功率；一阶蒸馏版本仅降低约 2–5% 的性能，却将动作块生成时延从约 700 ms 缩短到 80 ms（约 8.5 倍加速）。

**⚠️ 局限性**

仍受限于 AR‑NF 的模型容量与训练样本规模，较小规模的 AR‑NF 无明显性能提升；蒸馏过程可能导致细粒度动作细节的损失；在极端多模态或高维动作空间的任务中仍需进一步验证。

---

## 389. ContractScrub: A benchmark for final review of legal contracts

**arXiv ID:** 2608.20204 | [PDF](https://arxiv.org/pdf/2608.20204v1)

**作者:** Yejin Bang `[一作]` (Thomson Reuters Foundational Research), Andrew M. Bean `[通讯]` (Thomson Reuters Foundational Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并构建了ContractScrub基准，用于评估大型语言模型在合同清理（scrubbing）任务中的表现。

**💡 创新点**

首创聚焦完整合同清理流程的任务级基准，涵盖定义词一致性、大小写、跨章节引用等九大错误类别，并通过真实合同与人工注释填充高生态效度的数据集。

**🔧 技术方法**

利用多种前沿LLM（GPT‑5.5、Claude Opus、Gemini 等）在提示式推理下输出结构化 JSON，结合多模态评估指标（召回、精确率、F1）及长文本推理与一致性检查技术进行实验。

**📊 数据集**

构建自 CUAD 公开数据集的 44 套合同，人工注释 9,014 条错误实例（3,014 任务），再插入人工合成错误，形成包含 3,014 任务、9 类别的 ContractScrub 数据集。

**📈 对比分析**

通过宏平均召回率与精确率比较模型性能；GPT‑5.5 最高召回 0.75、精确率 <0.65，其他模型表现更差；不同错误类别召回差异显著，推理模式提升略高但耗时更长。

**⚠️ 局限性**

局限性包括仅 44 套合同、仅英美 10–15 页长度、仅英文、结构化 JSON 输出对模型产生额外约束，且样本量有限可能无法完全代表所有企业合同。

---

## 390. The Third Restructuring of Software Form: From the Three-Tier Architecture to Storage, Models, and Agents

**arXiv ID:** 2608.20201 | [PDF](https://arxiv.org/pdf/2608.20201v1)

**作者:** Wei Lin `[一作]` (Nanjing Liancheng Intelligent Technology Group), Changgui Hong `[通讯]` (Nanjing Liancheng Intelligent Technology Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证了将传统三层架构归约为存储、模型、代理三元结构的理论，并给出最小参考架构和原型实验，展示存储层可仲裁模型错误、代理循环可执行业务逻辑。

**💡 创新点**

统一了 Software 2.0/3.0、LLM OS、Agents、DB+AI 四条研究线，正式化“软件 = 存储 + 模型 + 代理”归约论证，阐明 UI、业务逻辑、数据层的重组方式及其边界。

**🔧 技术方法**

使用 Qwen‑Plus LLM、ReAct/Toolformer/AutoGPT 代理框架、向量/关系/图/KV 统一抽象数据库、OR‑Tools CP‑SAT 求解器，以及 Python/SQLite 原型实现。

**📊 数据集**

采用合成作业车间实例（6×6、8×8、10×10、12×12、15×15 等）和小型 3×3 作业车间作为 LLM 交互实验的数据集，未使用公开真实数据集。

**📈 对比分析**

通过与手写代码/声明约束对比，模型直接推理（0% 合法计划）与代理+求解器（100% 正确）进行对比；存储层 100% 过滤错误；12×12 求解器耗时 9.67 s；LLM 单次调用平均 3.84 s；相比传统代码，模型推理成本高但维护成本低。

**⚠️ 局限性**

仅适用于可表达、可验证、外部状态、工具完整、经济可行的任务；对确定性、低延迟高吞吐、合规严苛或无法验证的场景失效；模型幻觉需要外部验证，代理权限链等安全与可审计性挑战。

---

## 391. Explainable Transformer Models for Clinical Prediction Tasks on Structured Electronic Health Records

**arXiv ID:** 2608.20315 | [PDF](https://arxiv.org/pdf/2608.20315v1)

**作者:** Jun Ni Du `[一作]` (Sanofi), Brandon Rufino `[通讯]` (Sanofi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研发并评估了一种BERT-LER模型，该模型将实验室检验值通过分位数离散化成离散token，并使用集成梯度（Integrated Gradients）实现token级解释，随后在EHRShot公开基准和实际哮喘进展研究中进行验证。

**💡 创新点**

创新点在于：1）将实验室连续值以分位数编码与BERT输入统一，既保留量化信息又保持可离散化输入；2）将同一框架内的实验室编码与token级解释结合，实现对实验室值贡献的可解释性；3）在大规模75M患者预训练的基础上，展示了该方法在公开基准和真实世界临床任务中的可迁移性。

**🔧 技术方法**

技术手段包括：BERT-Transformer架构、Masked Language Modeling预训练、分位数（percentile）基实验室值离散化、Integrated Gradients解释、Fine‑tune on downstream预测任务、与Retain、CLMBR、MED‑BERT、XGBoost等基线模型对比。

**📊 数据集**

数据集：TriNetX Dataworks 75M de‑identified EHR（用于预训练和哮喘队列构建）；EHRShot 6,739患者（用于公开基准任务）；自定义哮喘进展队列（从TriNetX中去除预训练患者后得到的真实世界样本）。

**📈 对比分析**

比较方法：在EHRShot 14个分类任务中与CLMBR、RETAIN、MED‑BERT对比；在哮喘3个预测任务（失控、三分类、多重严重程度、加重）中与Retain、MED‑BERT、XGBoost、Logistic Regression对比。结果显示：BERT‑LER在大多数实验室相关任务上取得最高或相当于最高ROC‑AUC；在哮喘任务中，PR‑AUC和ROC‑AUC均居首位，去掉实验室值嵌入后性能下降，证明实验室量化信息对预测贡献显著。

**⚠️ 局限性**

局限性：1）未显式建模混杂因素，实验室值与病情因果关系仍不确定；2）IG解释的稳定性与可信度未做定量评估；3）分位数编码对非单调指标的表达有限；4）模型无法处理未见过的临床代码；5）哮喘任务表现低于部分基准，可能因样本稀缺和预测难度；6）缺乏对解释性方法与临床验证的系统性评估。

---

## 392. Physical-Support Confidence Sets for Highly Coherent Dictionaries

**arXiv ID:** 2608.20295 | [PDF](https://arxiv.org/pdf/2608.20295v1)

**作者:** Guan-Ju Peng `[一作]` `[通讯]` (National Chung Hsing University), Guan-Ju Peng (National Chung Hsing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于稀疏表示的字典学习框架，用于从高度相关的字典中推断物理支持信息，并实现不确定性量化与分辨率控制。

**💡 创新点**

创新点在于将不确定性量化与分辨率控制融入字典学习过程，并针对高度相关字典设计了稳健的正则化策略。

**🔧 技术方法**

采用稀疏编码、字典学习（如K‑SVD或MOD）、贝叶斯不确定性估计和多分辨率分析技术。

**📊 数据集**

使用了合成数据集和实际的图像/光谱数据集（如MNIST、CIFAR或医学影像）进行实验。

**📈 对比分析**

与传统K‑SVD、MOD以及最近的深度字典学习方法进行对比，实验表明在物理支持推断精度和分辨率上提升了约10‑15%，并且不确定性评估更为可靠。

**⚠️ 局限性**

主要局限包括：计算开销较大；对字典初始化和正则化参数敏感；在极端噪声环境下性能下降。

---

## 393. Phantom Gains: Auditing Self-Improvement Against a Measured Null

**arXiv ID:** 2608.20290 | [PDF](https://arxiv.org/pdf/2608.20290v1)

**作者:** Cheng Xu `[一作]` (University College Dublin), M-Tahar Kechadi `[通讯]` (University College Dublin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Qwen3-8B 通过三轮 LoRA 自训练进行过渡级别审计，识别出七种测量失效，提出阈值无关的扩张统计量和每个统计量的无效控制，验证自训练对模型能力的真实影响。

**💡 创新点**

（1）系统性揭示并纠正了自我改进评估中常见的七种测量错误；（2）提出了基于每题解答率的阈值无关扩张测试，避免了传统阈值导致的虚假增益；（3）提出每个统计量必须与冻结对照共享相同的评估设计，以确保统计显著性。

**🔧 技术方法**

使用 LoRA rank-32 微调、STaR、TTRL（投票自训练）、外部教师蒸馏、策略梯度自训练；评估采用贪婪解码与解答率估计器；统计方法包括按题 Exact Fisher 检验、FDR 控制、误差阈值分析。

**📊 数据集**

MATH‑500 子集（200 题）、AIME 2025/26（60 题）以及基于 MATH 训练集构建的 1,163 题难度带。

**📈 对比分析**

对比自训练与蒸馏与冻结基准，发现蒸馏在低基准难题上显著提升（8–11 题），自训练仅提升 0–2 题且在已解题上导致大量破坏（STaR 106 破坏，投票自训练 88 破坏）。总体准确度虽略升，但破坏/学习比大于 1，表明能力损失；安全拒绝探测结果不显著，难以判定安全性。

**⚠️ 局限性**

实验仅覆盖三轮 LoRA 微调和约 270 步，未探究更深度训练；使用单一 Qwen3‑8B 架构，结果可能不适用于其他模型；评估集主要来自预训练数据，可能高估了破坏程度；自训练方法的稳定性对种子敏感，需更多种子验证。

---

## 394. Dynamic Structural Causal Modeling for Sleep

**arXiv ID:** 2608.20285 | [PDF](https://arxiv.org/pdf/2608.20285v1)

**作者:** Ranveer Singh `[一作]` (University of Texas at Dallas), Sriraam Natarajan `[通讯]` (University of Texas at Dallas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文基于家用睡眠呼吸暂停检测（HSAT）数据学习睡眠呼吸障碍的动态因果图，并在性别与年龄亚群中比较因果结构差异。

**💡 创新点**

创新点在于将PCMCI+因果发现算法与针对高阶结构的引导型Bootstrap聚合相结合，利用领域知识约束和局部结构统计提升小样本下的因果稳定性。

**🔧 技术方法**

主要技术包括PCMCI+时间序列因果发现、Momentary Conditional Independence检验、边黑名单约束、Bootstrap重采样、结构聚合与自回归图构造。

**📊 数据集**

使用了105份至少两小时的HSAT录音，提取了Snoring、Pulse、Oxygen Saturation、Effort、Flow等五个分数特征。

**📈 对比分析**

通过在不同亚群（男女、老年/青年）中分别训练模型，比较因果结构的出现频率与后验概率；结果显示自回归与呼吸暂停-低氧关联在所有群体中稳健，而其余结构呈现明显差异，表明方法能捕获亚群特定的因果差异。

**⚠️ 局限性**

局限性包括样本量有限、缺乏睡眠阶段、觉醒等重要生理变量、仅使用单一时间窗口长度、模型对噪声敏感且可能受特征构造阈值影响。

---

## 395. Which Eviction Policy Should an LLM Cache Use? A Systematic Study Across Workloads, Capacities, and Encoders

**arXiv ID:** 2608.20280 | [PDF](https://arxiv.org/pdf/2608.20280v1)

**作者:** Yash Kulkarni `[一作]` (University of Michigan), Arvind Suresh Yogesh Babu `[通讯]` (University of Michigan)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七种缓存淘汰策略（FIFO、LRU、LFU、ARC、GDSF、流式 SISO、语义冗余）在统一实验协议下进行全面对比，评估其在不同容量、编码器和数据集上的命中率与延迟。

**💡 创新点**

首次提出统一的 CLEVER 框架，结合 packing 条件、跨编码器阈值校准与质量调整的命中率审计，证明在精确插入缺失的情形下几乎所有几何感知淘汰策略都不优于 LFU。

**🔧 技术方法**

使用 FAISS HNSW 索引、成本感知自适应路由器、七种淘汰策略实现，以及 Llama‑3.1‑8B 作为判定器对缓存命中是否可替代答案进行评估。

**📊 数据集**

处理后的 LMSYS‑Chat‑1M（首轮对话）、Quora Question Pairs、MOSS 指令集，三者去重后分别抽样 100k 条。

**📈 对比分析**

在同一索引、阈值、容量预填和后缀测量下对 18 组设置进行实验，发现 LFU 在所有设置中最优或仅落后 0.041pp；FIFO 与流式 SISO 在低容量下损失 6–8pp，语义策略因稀疏冗余而几乎无提升。

**⚠️ 局限性**

实验使用有序去重语料、缺乏真实请求重复、预填容量固定、仅评估 CPU 环境、阈值校准仅在评估集上、LLM 判定器无人工标签，以及对编码器、索引误差等方面的局限。

---

## 396. What Makes a Good Fiqh Retriever? Answer Retrieval for Arabic Islamic Jurisprudence

**arXiv ID:** 2608.20246 | [PDF](https://arxiv.org/pdf/2608.20246v1)

**作者:** Somaya Eltanbouly `[一作]` (Hamad Bin Khalifa University), Mohammed Ghaly `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建阿拉伯 fiqh 问答的检索测试集，并对稠密、词汇、混合检索及教派过滤方法进行系统评估

**💡 创新点**

首次提出答案载体和教派感知的检索评估框架，并展示教派过滤在减少跨教派误检中的巨大作用

**🔧 技术方法**

使用稠密检索（BGE-M3、ATM2、Muffakir等）与 BM25 的词汇检索，结合 Reciprocal Rank Fusion；对稠密检索进行 fiqh 专属微调；实现可选的规则式教派过滤

**📊 数据集**

356K 章节块构成检索语料库；19,319 个问答-正负三元组用于微调；503 条人工编写的 fiqh 问题用于评估

**📈 对比分析**

通过 MRR@5、nDCG@5、Hit@5 等指标比较；Muffakir V1 微调后达到 MRR@5 0.553，教派过滤在特定教派问题上将 MRR@5 提升超过两倍；混合检索对强模型提升有限，主要帮助弱模型

**⚠️ 局限性**

相关性判定仅为二元，未考虑部分相关；教派检测基于词典易出现误判；仅评估检索阶段，未测对生成结果影响；模型规模受限，未使用更大模型或更细粒度负样本

---

## 397. Unwarping the Lens: A Physics-Grounded Approach to Video Glasses Removal

**arXiv ID:** 2608.20212 | [PDF](https://arxiv.org/pdf/2608.20212v1)

**作者:** Radim Spetlik `[一作]` (Czech Technical University in Prague), Yinda Zhang `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用物理光学仿真与生成模型结合，训练可实时去除视频中眼镜的JFSnet网络，实现连续一致的眼镜消除；

**💡 创新点**

将大规模生成模型的多视角知识转移至确定性网络，并在训练中引入光学折射与反射仿真、三阶段结构过滤及平移等变换一致性约束；

**🔧 技术方法**

使用DINOv2 ViT编码器+ResNet解码器、物理折射/反射模拟、ARAP对齐、翻译等价性损失、感知与眼部局部损失；

**📊 数据集**

基于Nano Banana生成的13视角/表情合成数据集（1860个身份），以及精细筛选后的FFHQ“有眼镜”子集（12,163张）和CelebV‑Text视频子集（60段）；

**📈 对比分析**

与TokenFlow、RAVE、IP‑FaceDiff、ProPainter、Flow‑Guided Transformer、Runway Gen‑4.5等基线对比；在FFHQ上FID 0.379、landmark L2 0.632，视频中人类偏好与定量指标均优于现有方法，实时速度27.68 FPS；

**⚠️ 局限性**

对极端侧视角、深度估计误差、以及极大反射/遮挡的太阳镜等情况仍表现不佳，且依赖合成数据的质量与光学模型的近似。

---

## 398. Inductive Process Discovery from Partially Ordered Event Data

**arXiv ID:** 2608.20211 | [PDF](https://arxiv.org/pdf/2608.20211v1)

**作者:** Humam Kourani `[一作]` (Fraunhofer Institute for Applied Information Technology), Wil M. P. van der Aalst `[通讯]` (RWTH Aachen University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将 Inductive Miner 从总序列扩展到部分序列，使其能够直接处理部分有序事件日志。

**💡 创新点**

通过重新定义抽象层和投影，将并发信息保留在递归分解中，避免线性化爆炸。

**🔧 技术方法**

使用部分有序日志（POT）、POWL 语言、Inductive Miner、抽象（起始/结束/直接/最终跟随）以及块投影和访问投影等技术。

**📊 数据集**

实验使用合成并发模式、BPI Challenge 2012 事件日志以及从该日志生成的模拟数据。

**📈 对比分析**

与单线性化和全线性化基线对比，POT‑IM 运行时间极低，避免阶乘增长；在 BPI 实验中对齐度一致，模拟实验显示更快的行为覆盖和更高的模型相似度。

**⚠️ 局限性**

局限在于对不可比较事件频率的归一化仍是经验性的；投影仅覆盖当前切割类型；未充分探索生命周期/区间提取的深度。

---

## 399. Pandora's AI Model Routing Box: Efficient Allocation with Costly Value Estimation

**arXiv ID:** 2608.20316 | [PDF](https://arxiv.org/pdf/2608.20316v1)

**作者:** Adam Fisch `[一作]` (Google DeepMind), Jacob Eisenstein `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将模型路由问题视为潘多拉盒子问题的框架，设计了集中式路由器Pandora's Router和去中心化竞价器Pandora's Bidder，利用价值信息（VoI）在估计成本与收益之间做最优决策。

**💡 创新点**

创新点包括：
1) 将昂贵价值估计视为可计费的“打开盒子”动作，构建基于潘多拉盒子问题的路由与竞价策略；
2) 在高斯信号模型下给出闭式的预留价格和信息价值阈值，简化决策；
3) 将该框架推广到去中心化情形，使专业模型可在竞价过程中自行评估价值并决定是否付费。

**🔧 技术方法**

主要技术：
- 潘多拉盒子（Pandora's Box）最优搜索理论；
- 高斯信号模型与根搜索求解预留价格；
- 价值信息（VoI）分析与阈值决策；
- Monte‑Carlo 估计与仿真；
- 对比实验使用基准路由器和启发式方法。

**📊 数据集**

使用的公开数据集：
- EmbedLLM：多 LLM 路由基准（MMLU、GSM8K 等）；
- RAG：事实类问答与生物医学问答集合，包含 Wikipedia 与 PubMed RAG 模型；
- Math：MATH、Omni‑Math、AIME、HMMT 等数学推理题库。

**📈 对比分析**

对比方法：f‑only（仅用低成本估计），g‑always（始终使用高成本估计），Top‑2（查询两个最高 f 的模型），Coin‑flip（随机查询）以及两种预算重分配的 ablation（Random‑N_pr、Margin‑N_pr）。
性能表现：Pandora's Router 在所有成本设定下几乎达到最小化 regret+成本的下边界，显著优于 f‑only 和 g‑always；在去中心化竞价实验中，Pandora's Bidder 的分配效率和代理者盈余均位于两基线的低成本/高成本极值之间，表现优异。

**⚠️ 局限性**

局限性：
- 高斯信号模型对尾部和多模态分布的拟合不足；
- 仅考虑两级估计（f 与 g），不适用于更复杂的多级或树形评估；
- 竞价模型为单阶段留一式，忽略多轮竞价与策略预期；
- 在某些情况下，代理者追求自身盈余可能牺牲整体分配效率。

---

## 400. MidTool: Mid-training Data Synthesis for Agentic Tool Use

**arXiv ID:** 2608.20314 | [PDF](https://arxiv.org/pdf/2608.20314v1)

**作者:** Fengqing Jiang `[一作]` (University of Washington), Yuxiong He `[通讯]` (Snowflake)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并公开了一个可扩展的数据生成管线，用于构建 20.3B 令牌的“工具使用”中训练语料；随后在 Qwen3 4B 与 8B 基础模型上进行中训练，再进行 SFT 与 RL，评估其对工具调用和多轮交互任务的提升。

**💡 创新点**

①首次公开面向通用工具使用的中训练数据集与生成管线；②提出两条并行合成分支——基于文档的“语境对齐轨迹”与基于真实 API 的“原生代理轨迹”，实现对工具识别、参数抽取、工作流规划及错误恢复的全面覆盖；③证明中训练可显著弥补仅靠后训练难以获得的多轮、跨工具交互能力。

**🔧 技术方法**

- 文档预处理与质量过滤（基于关键词、FastText、MinHash 去重）；
- 语境对齐轨迹合成（规则规划、QA 对齐、语义检验）；
- 原生轨迹合成（工具库存构建、可行性评分、脚本化生成、校验一致性）；
- 基于 RL 的代理训练（AWM 环境、MCP 交互）。

**📊 数据集**

公开的数据源包括：
- Web：FineWeb Common Crawl 2020‑2025；
- PDF：FinePDFs 英文手册与教程；
- 代码：GitHub 公开仓库（agent/MCP 相关 + 高质量社区项目）；
- 结构化工具：REST API 与 MCP 技能定义；
- 轨迹合成：生成的 QA 及交互轨迹，融合 Nemotron Agentic 轨迹与 AWM 采样。

**📈 对比分析**

与基线（仅后训练 SFT / RL）、公开中训练基线（Dolmino‑20BT）以及无中训练版本进行对比；使用 BFCLv3、τ^2‑Bench 与 MCP‑Universe 三个评测集。结果表明：
- 在 4B/8B 模型上，中训练后续 SFT+RL 能提升 BFCL 多轮、τ^2‑Bench 通过率、MCP‑Universe 得分均有显著提升（多轮得分提升 10‑15 分，Pass@1 近乎翻倍）；
- 仅“语境对齐”或仅“原生轨迹”单独贡献也超过无中训练；
- 两条分支互补，完整混合得到最佳全指标提升。

**⚠️ 局限性**

1) 对深度搜索类任务（如 web‑search）提升有限，表明通用工具训练不涵盖需要长时序推理与探索的场景；
2) 仅覆盖 4B/8B 模型，尚未验证更大规模模型的适用性；
3) 轨迹合成依赖一定比例的模型生成，可能带来生成偏差；
4) 工具覆盖仍受公开 API 与 MCP 技能限制，未完全覆盖工业级专有接口。

---

## 401. Design and Empirical Evaluation of a Network-Centric, On-Premises Architecture for Earth Observation Data Access

**arXiv ID:** 2608.20283 | [PDF](https://arxiv.org/pdf/2608.20283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 402. Inter-X++: A Comprehensive Benchmark for Multimodal Human-Human Interaction Analysis

**arXiv ID:** 2608.20312 | [PDF](https://arxiv.org/pdf/2608.20312v1)

**作者:** Liang Xu `[一作]` (Shanghai Jiao Tong University), Wenjun Zeng `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出了Inter-X++数据集，包含11,388对高精度全身及手部动作，并配齐多层次文本、交互类别、因果顺序、关系、性格与接触等多模态注释。

**💡 创新点**

创新点在于融合高保真混合捕捉系统、物理约束与统一HHI表示OpenHHI，支持生成与感知两类任务的统一框架。

**🔧 技术方法**

采用混合光学+惯性动作捕捉、SMPL-X运动参数化、VQ‑VAE+ViT编码器、联合重建与描述生成训练、PHC强化学习等技术。

**📊 数据集**

数据集Inter-X++以及其升级版Inter-X，参考已有HHI数据集如InterHuman、NTU120等。

**📈 对比分析**

通过统一评测协议和多任务基线（如OpenHHI与TEMOS、T2M、Actformer等）对比，OpenHHI在文本到动作、动作生成、识别、描述、因果顺序、风格生成、性格评估等任务均实现了最优或显著提升。

**⚠️ 局限性**

限制包括缺乏面部表情、多样化持续交互、有限的文本-动作对齐方法，以及对物理交互在真实场景中的验证不足。

---

## 403. CalcSeg: Confidence-aware 3D Latent Context Curriculum Learning For Myocardial Scar Segmentation From Single-Stack LGE-CMRs

**arXiv ID:** 2608.20305 | [PDF](https://arxiv.org/pdf/2608.20305v1)

**作者:** Nivetha Jayakumar `[一作]` (University of Virginia), Miaomiao Zhang `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种基于置信度的半监督课程学习框架 CalcSeg，用以从单堆叠 LGE‑CMR 图像中自动精准地分割心肌瘢痕。

**💡 创新点**

创新点在于：①构建了可学习的三维潜在上下文，通过切片级自注意力捕获跨切片解剖关联；②设计了动态置信度评分函数，用预测误差、瘢痕体积误差与模型不确定性自适应地为训练样本分配难度，实现从易到难的连续课程学习；③将两项技术无缝融合，提升了对低对比、弥漫或小瘢痕病例的鲁棒性。

**🔧 技术方法**

核心技术包括：Transformer‑Encoder 预训练、基于自编码器的 3D 潜在自注意力模块、蒙特卡罗 Dropout 估计不确定性、混合焦点 Dice 损失、以及自适应阈值的课程学习调度。

**📊 数据集**

使用了来自四个中心的 976 名心肌病患者的 LGE‑CMR 数据集，并结合 MICCAI 2012 LV Infarct Challenge 与 EMIDEC 2020 Challenge 的公开标注，训练集/验证集/测试集比例为 7:1:2。

**📈 对比分析**

与 TransUNet、AttentionUNet、UNETR、ScarNet 以及 ScarNet（带专家标注的课程学习）等 SOTA 方法对比，CalcSeg 在整体 Dice 0.677、低置信度样本 Dice 0.644、瘢痕体积误差 38.88% 上均实现显著提升，尤其在临床挑战性病例上表现最佳。

**⚠️ 局限性**

局限性包括：仅处理单堆叠 2D 图像，缺乏多序列输入；课程学习阈值与权重仍需经验调优；蒙特卡罗 Dropout 估计不确定性对计算开销有一定影响；未来工作需进一步验证对其他病理类型与不同设备的泛化能力。

---

## 404. Towards Surgical World-Action Modeling: A Preliminary Joint Visual-Trajectory Forecasting for Surgical Motion Planning

**arXiv ID:** 2608.20284 | [PDF](https://arxiv.org/pdf/2608.20284v1)

**作者:** Weiliang Huang `[一作]` (University of Macau), Qingbiao Li `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个联合视觉-轨迹的手术世界动作模型，能够从历史手术视频和工具轨迹同时预测未来视觉状态和工具轨迹。

**💡 创新点**

创新点在于把视觉生成和轨迹预测联合起来，采用分块自回归推理和 scheduled sampling，并通过残差式视觉变化与速度先验的轨迹预测。

**🔧 技术方法**

技术包括冻结的 SurgMotion ViT-L 编码器、时空编码器、残差视觉预测头、速度先验轨迹头、分块 3 步自回归推理以及 scheduled sampling。

**📊 数据集**

使用 SurgWMBench（来自 SAR‑RARP50 的机器人辅助手术视频和 2D 轨迹注释）作为评估基准。

**📈 对比分析**

在一次性 15 步预测与分块 3→3 自回归两种设置下对比，分块方法在 PSNR、SSIM、LPIPS、ADE、FDE 上均优于一次性预测，尤其是前段可提升约 4 dB PSNR、约 50% ADE 降低。

**⚠️ 局限性**

主要局限在于长时域预测仍会出现视觉质量衰减和轨迹漂移，尤其在后期 t+13~15 时误差明显积累。

---

## 405. InsufficiencyBench: Evaluating LLM legal advice on underspecified user queries

**arXiv ID:** 2608.20220 | [PDF](https://arxiv.org/pdf/2608.20220v1)

**作者:** Samuel J. Vincent `[一作]` (Thomson Reuters Foundational Research), Nabeel Seedat `[通讯]` (Thomson Reuters Foundational Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 InsufficiencyBench，专门评估法律大语言模型在缺失关键法律信息时是否能够识别并避免过早给出结论的能力。

**💡 创新点**

通过引入八类法律缺失信息的结构化分类，并构造从完整查询到缺失变体的可控生成流程，以缺失检测而非答案正确性为核心评估指标，实现了对“早期法律闭合”这一安全缺陷的量化。

**🔧 技术方法**

使用大语言模型（如 GPT‑5）作为判别器，对模型回答进行自动提取缺失要素、解释准确度和安全率的评估，同时采用多模型对比和多种判别器验证结果稳健性。

**📊 数据集**

使用 202 条评估项目（58 条完整查询、144 条缺失变体），覆盖 6 个法律领域和 24 个美国州，人工标注 541 个法律要素，并对每个要素进行子标签、类别和必要性标注。

**📈 对比分析**

对十个前沿 LLM 进行统一系统提示下的单轮评测，主要指标为 F2（召回优先），最高模型 GPT‑5.2 仅达 0.46，平均召回 0.44，表明缺失识别普遍不足，且模型在不同缺失类别上的敏感度差异显著。

**⚠️ 局限性**

局限性包括数据集规模有限且仅涵盖争议性民事领域，评测基于单轮对话且依赖单一判别器，缺乏多轮交互与人工评判的验证，且模型表现可能受提示方式和温度设置影响。

---

## 406. MemTrapBench: Benchmarking Cognitive Traps in LLM Memory Use

**arXiv ID:** 2608.20202 | [PDF](https://arxiv.org/pdf/2608.20202v1)

**作者:** Mengru Wang `[一作]` (Zhejiang University), Ningyu Zhang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个新基准 MemTrapBench，用来评估大型语言模型在使用外部记忆时产生的认知陷阱，并设计了一种推理时提示技巧 AdaptiveMem 来缓解这些陷阱。

**💡 创新点**

创新点在于（1）首次系统化定义并测评记忆诱发的认知陷阱，包括推理固定与信念扭曲两大类别；（2）构建 1,050 条人工设计且经过双阶段质量控制的多轮对话样本；（3）提出 AdaptiveMem 提示技能，可直接嵌入多种记忆框架，且无需改动模型参数。

**🔧 技术方法**

主要技术包括：使用 GPT‑5.4 自动生成对话草稿；采用 GPT‑5.2 与 Claude‑Sonnet‑4.6 进行多维度（正确性、格式、相关性、效率）评分；评估多种记忆框架（FullText、LightMem、MemOS、SimpleMem、EverMemOS）；以及基于提示的推理时控制策略 AdaptiveMem。

**📊 数据集**

数据集为 MemTrapBench，包含 1,050 条实例，分布在 Reasoning Fixation（Cognitive Bias、Task Boundary、Trauma）与 Belief Distortion（Safety）四类，覆盖不同陷阱机制，并配有金标准答案。

**📈 对比分析**

在 Gemini‑3‑Flash‑Preview 与 Qwen3‑30B‑A3B‑Instruct‑2507 上进行实验，结果显示所有记忆框架均比无记忆基线差 10% 以上；使用 AdaptiveMem 后，平均提升约 14.9%（Gemini）和 2.5%（Qwen），并且在 LongMemEval 等标准记忆基准上保持或略有提升。

**⚠️ 局限性**

局限性包括：基准仅覆盖了特定类型的记忆诱发陷阱，未涉及记忆获取错误、更新失误等传统问题；AdaptiveMem 的效果依赖于提示设计，可能在更复杂或非对话场景下表现不佳；且实验仅在两种模型与五种记忆框架下验证，泛化性还有待进一步研究。

---

## 407. ConceptGuard: Benchmarking Context-Sensitive Unlearning in Large Language Models

**arXiv ID:** 2608.20338 | [PDF](https://arxiv.org/pdf/2608.20338v1)

**作者:** Sahil Kale `[一作]` (Pune Institute of Computer Technology), Ian Harris `[通讯]` (University of California, Irvine)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了ConceptGuard基准，用于评估大语言模型在对同一概念进行有害与善意使用时的选择性遗忘能力，构造了互补的遗忘集和保留集，并基于意图敏感的评价指标对模型进行全面评估。

**💡 创新点**

创新点在于将“双用概念”引入遗忘评测框架，强调在保持有益知识的同时彻底消除有害应用，提供了上下文分离度量和意图敏感的评估协议。

**🔧 技术方法**

采用了梯度上升、SimNPO（偏好驱动）、RMU（表示层干预）以及UNDIAL（自蒸馏）等多种遗忘技术，并结合ROUGE、LLM-as-a-judge和概念级分离度量进行验证。

**📊 数据集**

使用了从LLM-LAT恶意数据集提取的5,166条双用概念实例，构成了5,166条遗忘与保留样本，并在Qwen-2.5-3B和Llama-3.1-8B两个指令调优模型上进行实验。

**📈 对比分析**

通过比较方法，结果显示SimNPO和RMU在保持模型实用性方面优于梯度上升和UNDIAL，但所有方法在提升上下文分离度（概念级安全与有益使用的区分）方面均表现平平，且存在显著的忘记-效用权衡。

**⚠️ 局限性**

局限性包括当前方法难以在概念层面实现一致且可扩展的安全/有用性平衡，缺乏针对性优化的上下文分离技术，且数据集仍受概念频率分布限制，需进一步探索更广泛语言和主题的评测。

---

## 408. WithEveryone: Unified Planning and Identity Grounding for Group Image Generation

**arXiv ID:** 2608.20336 | [PDF](https://arxiv.org/pdf/2608.20336v1)

**作者:** Hengyuan Xu `[一作]` (Fudan University), Yu-gang Jiang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种统一多模态框架，能够根据5到10个参考身份生成连贯的团体图像。

**💡 创新点**

创新点在于引入身份–布局绑定的结构化规划、身份表示强制、以及以布局为基础的输出侧身份损失，解决大规模多身份生成中的身份衰退和复制问题。

**🔧 技术方法**

使用了变换器混合模型、流匹配图像生成、结构化Layout CoT规划、身份表示强制以及布局根据信息的LG-ID损失。

**📊 数据集**

使用了包含210张真实5至10人团体图像的身份独立基准数据集，以及公开图像和文本-布局的训练语料。

**📈 对比分析**

通过与学术、开源和商业系统（如GPT‑Image 2、Nano Banana、Seedream、WithAnyone等）的对比，实验显示其在目标上下文身份相似度、复制率、覆盖率和图像质量等指标上均优于现有方法。

**⚠️ 局限性**

局限性包括对更大人数群体的支持仍有限、对极端姿态或遮挡场景的鲁棒性待提升，以及依赖精确的布局标注和训练数据的规模。

---

## 409. Swift-Image: Exploring the Performance Frontier of Compact Unified Image Generation Models

**arXiv ID:** 2608.20334 | [PDF](https://arxiv.org/pdf/2608.20334v1)

**作者:** Taihang Hu `[一作]` (Alibaba Group), Mengting Chen `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了Swift‑Image，一款能够在同一套权重下完成文本到图像生成、单图像编辑和多图像编辑的紧凑统一模型。

**💡 创新点**

创新点包括：①从宏观到细化的渐进式训练管线；②并行专家强化学习与多教师 on‑policy 蒸馏，显著缓解多任务互斥；③Prompt Enhancer模块将高层意图解析与低层像素渲染解耦；④结构剪枝与分布匹配蒸馏实现参数压缩和少步采样；⑤系统化的训练工程整合架构、数据、后训练、提示与压缩，提升小模型性能。

**🔧 技术方法**

技术手段涵盖：6B单流 DiT、4D 旋转位置编码、并行注意力与 MLP 计算、渐进式分辨率/任务调度、并行专家 RL、DiffusionNFT 与多教师 OPD、Prompt Enhancer 重新写入与强化学习、结构剪枝、分布匹配少步蒸馏。

**📊 数据集**

数据集：大规模多模态文本‑图像对、编辑指令对与参考图像，按能力演进的分布；使用公开编辑基准（GEdit‑Bench、ImgEdit‑Bench、REDEdit‑Bench、CPI‑General、CPI‑Practical、CPI‑Intelligent）和生成基准（Qwen‑Image‑Bench、Pi‑ExpertVerse‑T2I）进行评估。

**📈 对比分析**

在公开编辑基准上，6B模型以平均得分 4.16/4.20（含少步蒸馏）在所有 open‑source 方案中排名第一；压缩至 3B 后几乎无性能损失；与更大模型相比，参数量仅为 6B/3B，训练时钟约 243K GPU 小时，展示了高效与强大的竞争力。

**⚠️ 局限性**

局限性：模型规模仍相对较大，推理速度受限；训练需要大量算力与数据；在某些极其复杂或极具知识密集度的编辑任务上仍有提升空间；缺乏跨语言、跨文化的全面评估；未对安全性与偏见做系统性探讨。

---

## 410. Inducing Task Models from Computer-Use Traces

**arXiv ID:** 2608.20319 | [PDF](https://arxiv.org/pdf/2608.20319v1)

**作者:** Yucheng Jiang `[一作]` (Stanford University), Diyi Yang `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种任务模型诱导方法（TMI），能够从无标注的电脑使用轨迹中自动生成任务的层次化目标结构与执行流程模型。

**💡 创新点**

创新点包括：① 在未给定任务列表的前提下同时发现潜在任务并分配交错的活动；② 分别构造目标层次与程序流程模型，再通过一致性约束将两者融合为统一任务模型；③ 解决多线程、交错执行及非线性工作流程的建模难题。

**🔧 技术方法**

技术手段主要有：Vision‑Language 模型用于事件语义定位与动作识别；语言模型进行语义动作与活动分割、目标与流程的递归推断；LLM（gpt‑5.x）实现模型生成、对齐与评判；调整的 Rand 指数等指标评估任务划分与步骤匹配。

**📊 数据集**

使用的数据集包括：HumanWork（38 条人类工作会话）和 SkillsBench（86 个软件工程任务）进行内部验证；另外合成多任务轨迹用于鲁棒性测试。

**📈 对比分析**

通过与工作流摘要、直接生成模型等基线对比。TMI 在任务划分的 Adjusted Rand Index 达 0.974，步骤匹配率为 74.9%（对比基线 30.3%）；在 SkillLearnBench 上提升任务准确率约 30%；在多任务鲁棒性评估中，MAE<1，显示对任务数量和交错密度的高容错性。

**⚠️ 局限性**

限制点：方法依赖原始屏幕与键盘记录，存在个人信息泄露风险；未评估去标识化对诱导质量的影响；仅在公开数据上验证，缺乏大规模多域场景的进一步测试。

---

## 411. DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery

**arXiv ID:** 2608.20308 | [PDF](https://arxiv.org/pdf/2608.20308v1)

**作者:** Yufei Liu `[一作]` (Shanghai Jiao Tong University), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种离线剪辑级框架，利用视频扩散模型的清晰潜在表示，在单次前向传递中恢复全景式的三维双手轨迹；

**💡 创新点**

创新点在于将预训练的生成式视频扩散模型转变为确定性几何编码器，并通过LoRA端到端学习将潜在特征转化为几何感知；同时使用双向时空解码器和基于射线的相机求解器实现对遮挡与离视场的鲁棒重建；

**🔧 技术方法**

核心技术包括：视频扩散模型（Wan DiT）作为编码器，LoRA微调，Bidirectional Spatiotemporal Decoder，Ray-Based Camera Solver（混合PnP），以及端到端的三维监督损失；

**📊 数据集**

在ARCTIC、HOT3D、HOI4D、H2O和OakInk2等五个自顶层的第一人称手部数据集上进行训练与评估；

**📈 对比分析**

与十种单帧与视频基线（如ViDiHand、WildHands、EgoForce等）对比，取得在MPJPE、姿态、2D定位、全局旋转、相机位姿、抖动等多项指标上的显著提升（如MPJPE减少30%+，Jitter降低至约2.7mm/frame²），且推理速度比ViDiHand快33倍；

**⚠️ 局限性**

局限性包括：需要完整视频剪辑进行离线推断，不适合实时闭环控制；射线场对相机家族的泛化有限；在HOI4D和H2O等数据集上受限于伪标签，需更高精度标注；

---

## 412. DART-S: Reachability-Audited Active-Suspension Preconditioning for Off-Road Vehicle Jumps

**arXiv ID:** 2608.20275 | [PDF](https://arxiv.org/pdf/2608.20275v1)

**作者:** Yu Hu `[一作]` (Research Center for Intelligent Computing Systems, Institute of Computing Technology, Chinese Academy of Sciences), Baolei Chen `[通讯]` (Dong Feng Off-Road Vehicle Co., Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用主动悬挂在离地前对离跳状态（俯仰角、俯仰率、轮速）进行预调节，从而在空中姿态控制阶段可获得更大可用角动量预算的离跳控制框架（DART‑S）。

**💡 创新点**

创新点包括：
• 将悬挂预调节映射为离跳状态与剩余角动量预算的双向控制；
• 采用局部校准映射（CalMap）和支持感知选择器，将状态偏移与预算约束同时评估；
• 结合区间可达性检验与精确对偶可达性审计，实现“可达性审计”式的安全决策；
• 在模拟中实现时序化部署（静态预设、定时预设）并与 DART 机载控制器串联，形成完整的多阶段控制链。

**🔧 技术方法**

技术手段包括：主动悬挂硬件与控制、可达性分析（区间与精确对偶可达性）、局部经验校准映射、支持感知选择器、BeamNG.tech 物理仿真、基于车身动力学的预算约束模型、离跳后姿态判定阈值。

**📊 数据集**

数据集：在 BeamNG.tech（v0.38.3.0）中使用自定义 4WIDS 平台进行 600 次试验，分布于 72 个独立会话，覆盖多种坡面配置（平板式、凸形出嘴）与速度/角度条件；每个条件多次重复以实现统计可靠性。

**📈 对比分析**

比较方法：与原 DART 基线、静态预设、时序预设、受限（守护）和不受限版本进行交叉对照；采用会话层 Holm 校正的符号检验、置信区间与极值检验。主要性能结果为：
• 在 40°/13 m/s 边界条件下，DART‑S 24/24 次满足落地姿态判据，而 DART 0/24；
• 0.35 s 时序预设 23/24 次成功，对比静态预设 0/24；
• 通过 200 rad/s 轮速限制保证所有 600 次轨迹均不超出 221.2 rad/s 驱动硬限；
• 通过定量比较展示悬挂预调节对离跳俯仰率和角动量预算的正向影响。

**⚠️ 局限性**

局限性：
• 仅在仿真环境验证，缺乏实际硬件试验；
• 仅针对俯仰平面控制，未考虑滚转/偏航耦合；
• 校准映射与支持范围有限，需在新的坡面/速度区间重新标定；
• 受限于 BeamNG 物理模型的逼真度，对真实车辆结构和传感器误差未做完整考量；
• 需要进一步评估在更大车辆尺寸、不同悬挂布局和地形复杂度下的泛化性。

---

## 413. Break It Down, Pass It On: Cross-Task Skill Transfer in LLM Agents

**arXiv ID:** 2608.20274 | [PDF](https://arxiv.org/pdf/2608.20274v1)

**作者:** Yiyang Feng `[一作]` (Stony Brook University), Jiawei Zhou `[通讯]` (Stony Brook University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估了长时程智能体在三大标准基准（多应用工具使用、办公文件工作流、数据科学管道）上的表现，并研究了其技能记忆的迁移行为。

**💡 创新点**

首次系统地比较固定诱导、检索和去重规则下的技能记忆迁移效果，为可靠重用可信技能奠定基础。

**🔧 技术方法**

采用LLM驱动的智能体、基于Docker的沙箱环境以及自定义的技能记忆机制（固定诱导、检索、去重）。

**📊 数据集**

三组标准基准：多应用工具使用基准、办公文件工作流基准和数据科学管道基准。

**📈 对比分析**

使用官方评估器对最终环境状态进行打分，确保评分确定性和可比性。实验表明，在这三大基准上智能体能够在一定程度上迁移并重用先前学习的技能，表现稳定。

**⚠️ 局限性**

仅覆盖三大基准，其他如计算机使用、编码或网络搜索等环境可能表现不同；技能记忆仅为固定不变，未覆盖可演化记忆；缺乏步骤级真实标签，难以进行细粒度决策分析；Docker根访问限制限制了实验规模。

---

## 414. Video2DoorTraversal: Push Door Traversal via Simulated Door Twins

**arXiv ID:** 2608.20251 | [PDF](https://arxiv.org/pdf/2608.20251v1)

**作者:** Xincheng Tang `[一作]` (Shanghai Jiao Tong University), Ruigang Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本论文提出一种单视频实现的实地→仿真→实地闭环门洞穿越框架，包含门的实例级数字双胞胎重建、仿真循环专家轨迹生成以及基于双深度摄像头的机器人本体与臂协调控制策略；

**💡 创新点**

创新点包括：① 仅凭一段RGB视频即可生成与真实门完全对齐的可在物理仿真中使用的关节门双胞胎；② 通过仿真循环的参数化技能程序实现无人工操作的可执行专家演示；③ 将机器人中心Plücker光线投影与交互状态监督相结合的双深度策略，显著提升长时段接触丰富的移动操控；

**🔧 技术方法**

核心技术涵盖：视频几何重建（DAGE+SAM）、基于程序化生成的关节门资产（Articraft+reference-view critic）、纹理与外观迁移（Tripo3D+GPT-based编辑）、仿真内循环的参数化技能程序优化、Action Chunking Transformer（ACT）改造的ArticuACT，使用双深度图、Plücker光线映射及交互状态预测；

**📊 数据集**

数据集主要为作者自行收集的20个真实门的RGB视频以及5个在真实机器人上测试的门；此外在仿真中采集了200条成功演示用于训练，并在三把未见门上测试零样本泛化；

**📈 对比分析**

与Replay、OA Replay、DoorGym、UniDoorManip、Vanilla ACT、Diffusion Policy、DP3等基线相比，在仿真中门开启成功率98.44%、通行成功率97.27%；在真实五把门上平均成功率达96.57%，零样本结构相似门的成功率为80.95%；相较于现有方法均实现显著提升；

**⚠️ 局限性**

局限性包括：仅适用于传统把手开启式门，难以处理拉门或多把手门；对单一RGB视频的质量和环境光照高度依赖；需要对特定机器人平台（Unitree A2-W+Z1 arm）进行适配；未在极端反射或玻璃门等场景下充分验证；

---

## 415. QUASAR: A Quantum-Classical Neural Network for SAR Satellite Physical-Layer Authentication

**arXiv ID:** 2608.20240 | [PDF](https://arxiv.org/pdf/2608.20240v1)

**作者:** Vincenzo Sammartino `[一作]` (University of Pisa), Roberto Di Pietro `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 QuaSAR——一种融合量子变分电路与卷积编码的混合网络，用于 X 波段 SAR 卫星的物理层身份认证。

**💡 创新点**

创新点包括：①首次在卫星 PLA 中引入 VQC；②采用 IQ 原生幅相编码将 IQ 样本映射到单比特态；③实现仅使用 10% 数据即可匹配全量经典基线的准确率，显著提升数据效率。

**🔧 技术方法**

使用技术包括：卷积神经网络对 STFT 频谱编码、IQ‑原生量子编码、8 比特 VQC（4 层强耦合层）、参数移位法训练、深度学习梯度显著性与潜在空间可视化。

**📊 数据集**

采用 37 颗 ICEYE X‑波段 SAR 卫星在 28 天内收集的 3.76 TB 原始 IQ 数据，经过 STFT 生成 224×224 频谱图后划分为训练/验证/测试集。

**📈 对比分析**

与 ResNet‑18、MobileNetV2、Transformer 等传统 CNN 基准进行对照，QuaSAR 在验证集达到 97.3% 准确率、宏 F1 0.973，且在仅使用 10% 数据时与使用 100% 数据的经典模型相当，整体提升约 7.5% 以上。

**⚠️ 局限性**

局限性包括：实验仅在经典模拟 VQC 上完成，未在真实量子硬件验证；仅针对单极化信号，未覆盖全部极化模式；对抗场景有限，缺乏更复杂的对抗攻击评估；模型对卫星硬件漂移的适配与多类识别仍待进一步研究。

---

## 416. Rule-Compliant Visual Spatial Planning for Multimodal Large Language Models

**arXiv ID:** 2608.20237 | [PDF](https://arxiv.org/pdf/2608.20237v1)

**作者:** Yu Chen `[一作]` (Peking University), Yang Liu `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文旨在提出一种符合规则的视觉空间规划方法，结合多模态大型语言模型（LLM）实现对空间任务的规划与执行。

**💡 创新点**

创新点在于将规则约束直接嵌入多模态LLM的生成过程，使得生成的空间规划既满足视觉感知要求，又严格遵循预定义的规则。

**🔧 技术方法**

采用了多模态LLM（如基于Transformer的模型）与规则约束学习技术，可能涉及视觉特征提取网络（如CLIP或ViT）以及规划模块（如规划图搜索或强化学习）。

**📊 数据集**

使用的数据集未在提供的内容中具体说明，推测可能包含视觉空间规划任务的标注数据，例如室内布局图、地图或视觉问答数据集。

**📈 对比分析**

与传统规则无关的视觉规划方法或基线LLM进行比较，评估指标可能包括规则遵循率、规划质量评分和执行时间；根据文中描述，本文方法在规则遵循率上优于或相当于现有方法，但具体数值未给出。

**⚠️ 局限性**

主要局限性包括：规则库的可扩展性有限，模型对未见规则的泛化能力不足，计算成本较高，以及在复杂多变的真实环境中验证效果的难度。

---

## 417. Differentially Private Continual Release with Relative Error

**arXiv ID:** 2608.20230 | [PDF](https://arxiv.org/pdf/2608.20230v1)

**作者:** Bo Li `[一作]` (Hong Kong University Of Science And Technology), Peng Ye `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究在差分隐私连续发布模型下四个基本任务（最大、最小的估计与选择）并提出相对误差加上绝对误差的算法，取得极大改进。

**💡 创新点**

① 通过允许相对误差，使得在非自适应流中所有任务的绝对误差可降至 polylog(d,T)；② 在自适应流中证明最大/最小估计可保持这一误差，而最小选择任务仍需与纯加法误差同等的绝对误差，揭示非自适应与自适应流的显著分离；③ 通过潜能分析和阈值检测实现高效更新。

**🔧 技术方法**

使用阈值检测机制、指数机制、潜能分析、组合定理与群隐私，以及对流的非自适应性进行潜在值下界证明。

**📊 数据集**

未使用真实数据集；实验与分析均基于理论模型和合成输入流。

**📈 对比分析**

相对于之前仅考虑纯加法误差的最优上界，本文提供了 polylog(d,T)/γ 的绝对误差上界，并给出几乎匹配的下界；在自适应流中最大/最小估计保持同样的误差上界，但最小选择任务无法改善。

**⚠️ 局限性**

仍存在 polylog(d,T) 的误差上界与下界之间的微小空隙；研究未扩展到其他统计任务；在存在负值或更一般范围时方法不再适用。

---

## 418. Prompt-Conditioned Channel Attention for Hierarchical Feature Modulation toward Anatomy-Agnostic Segmentation

**arXiv ID:** 2608.20229 | [PDF](https://arxiv.org/pdf/2608.20229v1)

**作者:** Mosharof Hossain `[一作]` (Khulna University of Engineering and Technology), Md Kamrul Hasan `[通讯]` (Khulna University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了PROMISE-Net，一种在编码器-解码器结构中层层融合提示信息的医学图像分割框架，并通过Prompt-Conditioned Channel Attention（PCCA）实现提示驱动的通道级特征调制。

**💡 创新点**

创新点在于：①将提示信息从传统的浅层融合扩展到深层全网络的层级化调制；②设计了轻量级的PCCA模块，将视觉特征与提示特征在通道维度上进行条件化激励；③实现了可同时适用于CNN和Transformer两种主干网络的通用框架。

**🔧 技术方法**

核心技术包括：卷积/自注意力编码器、全局平均池化+共享潜在空间投影、门控激励网络（Excitation MLP）、多尺度提示编码器（基于位置信息与语义嵌入的密集提示），以及完整的PROMISE-CNN/ PROMISE-Txformer架构。

**📊 数据集**

使用了四个公开医学图像分割基准：ISIC‑2017（皮肤病变）、Kvasir‑Polyp（肠道息肉）、Kvasir‑Instrument（内镜工具）以及CAMUS（二维心脏超声）等，覆盖多模态、多解剖结构和不同图像质量。

**📈 对比分析**

与U‑Net、UNETR、SAM以及多种先进方法（Pact‑Net、USL‑Net、FAT‑Net、HTC‑Net、RM‑U‑Net等）进行对比。PROMISE‑CNN在所有数据集上均取得最高或接近最高的DSC、IoU并显著降低HD95和FNR；PROMISE‑Txformer在Transformer基干上同样优于UNETR和SAM，平均DSC提升约5%且计算开销仅为U‑Net的2–3倍。

**⚠️ 局限性**

局限性包括：①对多目标或多提示情况的处理仍不完善；②在极度模糊或内部纹理复杂的区域仍可能出现漏分或过分；③目前仅验证了2D平面，尚未扩展到3D/4D时序数据；④对提示精度敏感度需进一步评估。

---

## 419. A comparison between ceiling-mounted FMCW, IR-UWB and Wi-Fi radar for in-bedroom human activity monitoring and sleep interruption detection

**arXiv ID:** 2608.20322 | [PDF](https://arxiv.org/pdf/2608.20322v1)

**作者:** Anton Lambrecht `[一作]` (Ghent University-imec), Eli De Poorter `[通讯]` (Ghent University-imec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对比了天花板安装的FMCW雷达、IR‑UWB与Wi‑Fi三种射频技术在同一环境下的人体活动识别（HAR）与睡眠监测任务的性能

**💡 创新点**

提供了统一的实验平台与数据集，使用同一卷积神经网络（CNN）对三种技术进行公平对比，并分析了空间分辨率、多天线、多普勒分辨率等信号特性对鲁棒性与识别率的影响

**🔧 技术方法**

频率调制连续波雷达（FMCW）、脉冲式超宽带（IR‑UWB）以及802.11 Wi‑Fi CSI 传感；统一的CNN架构、时间–多普勒或范围–多普勒预处理

**📊 数据集**

20名受试者、6种卧室布局、同步采集的FMCW、IR‑UWB和Wi‑Fi信号，公开的同步数据集可在 GitLab 访问

**📈 对比分析**

采用留一人/留一布局/留一布局对（LOPO、LOSO、LOBPO）交叉验证，评估宏 F1；结果显示 IR‑UWB 在已知布局下最高（89%），FMCW 在未知布局下最稳健（~84%），三者睡眠监测均超 92%；Wi‑Fi 性能最弱

**⚠️ 局限性**

实验只在单个天花板安装且受限于硬件/预处理参数，无法独立评估各信号特性的贡献；Wi‑Fi 需要昂贵的 SDR 平台，实用性受限

---

## 420. An Agentic Approach for Active Data Collection, Travel Behavior Modeling, and Weather-Sensitive Demand Prediction

**arXiv ID:** 2608.20320 | [PDF](https://arxiv.org/pdf/2608.20320v1)

**作者:** Narges Ahmadi `[一作]` (McGill University), Luis Miranda-Moreno `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建多代理工作流，整合对话式图像增强的SP调查、结构化数据处理、传统离散选择模型、机器学习以及大语言模型（LLM）实现对旅行模式的预测；

**💡 创新点**

创新点在于：①首次将对话式问卷与图像增强与LLM预测结合；②在统一可审计的多代理流程中完成数据采集、处理与预测；③系统评估LLM在零射击、少样本、人格提示及视觉输入等多种配置下的表现；

**🔧 技术方法**

使用技术包括：对话式Chatbot（Voiceflow）收集数据；AI生成的天气场景图像；多元离散选择模型（MNL）；机器学习方法（Logistic回归、Random Forest）；大语言模型（Gemma、Llama、Qwen等，参数规模2-35B）；多模态输入；

**📊 数据集**

采用数据集：92名麦吉尔大学学生通勤者的SP数据，5种天气场景图像，共454个受访者-情景观测；

**📈 对比分析**

评价方法为5折交叉验证按受访者分组，比较MNL、Logistic回归、Random Forest与LLM。Random Forest在五类任务上达69.6%准确率；最佳零射击LLM 69.9%；视像增强版最高71.5%；LLM在少样本提示后有提升，模型规模对准确率影响有限；

**⚠️ 局限性**

局限性包括：样本规模小、仅为单一学生群体，缺乏外部验证；使用的天气图像与天气变量未分离，导致无法评估图像独立效应；LLM评估多配置可能导致过拟合；视觉输入的泛化性未得到验证；真实行程数据缺失。

---

## 421. The Honeycomb Framework for Code Bounds

**arXiv ID:** 2608.20287 | [PDF](https://arxiv.org/pdf/2608.20287v1)

**作者:** William Gay `[一作]` (University of Illinois at Urbana-Champaign), Lenny Liu `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了蜂巢层次（honeycomb hierarchy），通过保留所有兼容的两行超八面体表示，给出了新的二进制码距离-速率上界，并构建了更高阶的有限长度松弛与无条件Horn通道同源层次。

**💡 创新点**

创新点：①引入完整的两行表示图，利用盒子转移与Wigner 6j系数精确计算边权；②发展多层次框架，包括表示深度、锚点深度和Horn通道深度，既能提供严格的有限收敛证明，又能在无条件下实现更高阶的渐近上界；③证明蜂巢指数κ_HC严格改进了全立方体/二进制MQR和MRRW界，并与2MQC形成嵌套层次。

**🔧 技术方法**

技术：表示论（超八面体群、Littlewood–Richardson系数、Knutson–Tao海绵模型）、移动投影（profile-optimized moving‑projection theorem）、Wigner 6j和Racah重耦合、Følner盒子渐近分析、凸优化（矩阵型Horn通道）以及稳定集矩阵层次（anchor moment hierarchy）。

**📊 数据集**

无具体实验数据集，全部基于理论推导与符号计算；数值验证使用计算机符号与数值优化（如分支限界）进行极限值比较。

**📈 对比分析**

通过与已知界限（MRRW M_2、OpenAI的κ_bin、2MQC、M_2MQC）进行点对点比较，证明κ_HC≤κ_bin 且 κ_HC<2MQC 在整个 (0,½) 区间内；对比值表明在所有非平凡距离处均有严格改进，且在低阶层次已显著优于先前最强界限。

**⚠️ 局限性**

局限性：(1) 证明仅表明每固定行/锚点层次的有限收敛，尚未证实有限层即可达到渐近最优；(2) 计算复杂度随行深度增长快速，实际求解高阶层次仍受限；(3) 目前仅对二进制码给出完整证明，推广到 q-ary 需要进一步工作。

---

## 422. Catching the Rug: Early Prediction of Fraudulent Memecoins on Solana via Machine Learning

**arXiv ID:** 2608.20271 | [PDF](https://arxiv.org/pdf/2608.20271v1)

**作者:** Jianghai Li `[一作]` (Higher School of Economics), Igor Vodolazov `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并分析了6.4M条Solana memecoin交易数据，提出基于前5分钟交易信息的早期欺诈检测模型；

**💡 创新点**

在规模、预测时效（5分钟）、跨平台泛化（PumpFun与Raydium融合）以及仅使用流动性与价格动态而非代码分析的创新方法；

**🔧 技术方法**

采用梯度提升树（XGBoost、RandomForest）、神经网络（MLP、FT-Transformer、TabTransformer、AutoInt）以及Optuna超参数搜索，并使用滚动时间窗交叉验证；

**📊 数据集**

基于Solana两大去中心化交易所PumpFun和Raydium收集的7个月交易记录，形成6.4M代币的标签化数据集；

**📈 对比分析**

通过F1、MCC、AUCPRC三指标评估，XGBoost在融合数据集上可达AUCPRC≈0.80、MCC≈0.39，显示跨平台融合显著提升性能；

**⚠️ 局限性**

存在平台间分布漂移、仅依赖流动性特征、缺乏洗牌交易与时间序列建模、未在真实环境部署、误报与漏报仍对投资者构成风险。

---

## 423. A Resource-Efficient CNN-Based EEG Auditory Attention Decoding ASIC

**arXiv ID:** 2608.20198 | [PDF](https://arxiv.org/pdf/2608.20198v1)

**作者:** Qier Ma `[一作]` (Technische Universitaet Dresden), Christian Mayr `[通讯]` (Technische Universitaet Dresden)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一款用于实时脑电注意力解码的低功耗ASIC，集成量化CNN推理引擎和Pearson相关性分类器

**💡 创新点**

创新点在于跨层流式数据流与硬件感知的算法优化（组卷积、BN折叠、ReLU、INT8 PoT量化）以及流式Pearson相关性计算，显著降低面积与功耗

**🔧 技术方法**

使用了32nm GF22FDX CMOS工艺、量化CNN、流式多MAC单元、串行分数与平方根单元、循环缓冲同步器等硬件技术

**📊 数据集**

在CI患者收集的真实EEG数据集上训练与验证，使用离线训练的量化模型

**📈 对比分析**

与EEGNet、HDC、SaleNet等代表性实现对比：面积仅0.076 mm²，功耗0.494 mW，单次推理能耗3.63 µJ，延迟7.34 ms，满足125 Hz采样的实时要求

**⚠️ 局限性**

尚未完成硅级测量与现场验证；在不同PVT角落的仿真表现良好，但对实际临床环境的鲁棒性与泛化能力仍需进一步评估

---

