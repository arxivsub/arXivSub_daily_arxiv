# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-02 | 今日论文总数: 530

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. UCell: rethinking generalizability and scaling of bio-medical vision models

**arXiv ID:** 2604.00243 | [PDF](https://arxiv.org/pdf/2604.00243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 2. Unsupervised 4D Flow MRI Velocity Enhancement and Unwrapping Using Divergence-Free Neural Networks

**arXiv ID:** 2604.00205 | [PDF](https://arxiv.org/pdf/2604.00205v1)

**作者:** Javier Bisbal `[一作]` (Pontificia Universidad Católica de Chile), Sergio Uribe `[通讯]` (Monash University)

**通讯引用:** 3846 | [OpenAlex ID](https://openalex.org/A5042514167)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种无监督的 4D Flow MRI 速度增强与相位展开网络 DAF‑FlowNet，直接对速度进行向量势参数化，并通过余弦一致性损失同时完成去噪与相位展开。

**💡 创新点**

创新点在于：① 将速度表示为向量势的旋度，以网络结构直接强制无散度，避免了手动调节散度惩罚；② 采用余弦数据一致性损失处理相位展开，使得去噪与展开在同一优化框架内完成；③ 使用傅里叶特征编码提升高频特征捕捉，且无需监督标签。

**🔧 技术方法**

技术手段包括：坐标编码的全连接网络（MLP）+傅里叶特征；自动微分计算速度；余弦一致性损失 + 速度零滑移边界约束；无监督训练；参数化的向量势实现质量守恒。

**📊 数据集**

实验数据：① 以 CFD 模拟的主动脉 4D Flow MRI（FEelMRI 生成）在不同 VENC 与噪声水平下的合成数据；② 10 例肥厚型心肌病患者 3T 4D Flow MRI（VENC 150）。

**📈 对比分析**

与 DivCorr、RBF、DFW、4DFlowNet（去噪）以及 Lap4D、NPRS、GC3D（展开）以及顺序管线 GC3D+DFW 进行对比；在合成数据上 DAF‑FlowNet 使 VelNRMSE、方向误差、散度分别降低 11%/11%/44%；展开残差在 VENC150/100 分别仅 0.18%/5.2%，比最佳对手降低 72%/18%；在混合噪声+展开任务中速度 NRMSE、方向误差、散度比顺序管线低 15%/11%/28%；在临床数据中显著提升主动脉和肺循环的质量守恒误差。

**⚠️ 局限性**

局限性包括：在极低 VENC（多次相位折叠）下性能下降；训练时间相对较长；仅针对单 VENC；对傅里叶尺度的选择敏感，需额外调优；缺乏多中心大规模验证。

---

## 3. A Unified Framework for Analysis of Randomized Greedy Matching Algorithms

**arXiv ID:** 2604.00331 | [PDF](https://arxiv.org/pdf/2604.00331v1)

**作者:** Mahsa Derakhshan `[一作]` (Northeastern University), Tao Yu `[通讯]` (Northeastern University)

**通讯引用:** 14898 | [OpenAlex ID](https://openalex.org/A5057697869)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在一般图上针对随机贪心匹配算法（Ranking 与 FRanking）以及查询‑提交匹配问题，提出了一套统一的分析框架，利用交替路径、备份（backup）与受害者（victim）概念，改进了收益分配机制，从而提升了这两类算法在所有图上的近似比。进一步地，作者证明了当图的最短奇环长度足够大时（如无三角、五边形或奇环长度≥129），Ranking 的近似比可以进一步提升至 0.570 以上，甚至 0.615。

**💡 创新点**

创新点：
- 将备份和受害者两个概念统一到交替路径的框架中，揭示它们是相互对应的结构。
- 引入多元补偿函数（compensation function）和单调性约束，显著简化了因受害者分析导致的 LP 维度爆炸问题。
- 设计了新的单调性性质（匹配质量随 vertex 重要性变化而单调），用于聚合不同 rank 的概率分布。
- 针对奇环长度（odd girth）敏感的图，构造更强的 LP 约束，利用更长的交替路径带来的额外补偿来进一步提高近似比。

**🔧 技术方法**

技术手段：
- 随机原始-对偶（randomized primal–dual）框架结合收益共享规则。
- 交替路径引理（Alternating Path Lemma）与备份/受害者分析。
- 因子揭示线性规划（factor‑revealing LP）与多段分段的数值近似。
- 单调性约束与受害者补偿的多变量设计。
- 对奇环长度的结构化分析（odd‑girth sensitive）。

**📊 数据集**

实验/数据集：无；论文完全以理论分析和证明为主，没有使用具体数据集进行实验验证。

**📈 对比分析**

与现有结果对比：
- Ranking 在所有图上的近似比提升至 0.560（此前最高 0.5469）。
- FRanking 在所有图上的近似比提升至 0.539（此前最高 0.521）。
- 对无三角、五边形的图，Ranking 近似比提升至 ≥0.570；
- 对奇环长度≥129 的图，Ranking 近似比提升至 ≥0.615。
- 这些提升在随机顺序（Ranking）与对抗顺序（FRanking）两种模型下均实现，且同样适用于查询‑提交匹配与完全在线匹配问题。

**⚠️ 局限性**

局限性：
- 结果仍未达到最优 1‑1/e（仅 0.56/0.54）。
- 分析高度依赖图的结构（奇环长度），在普通稠密图中提升有限。
- LP 约束仍较为复杂，数值求解可能受限于计算资源。
- 对加权匹配或动态/多轮模型的推广尚未给出。

---

## 4. Towards Automatic Soccer Commentary Generation with Knowledge-Enhanced Visual Reasoning

**arXiv ID:** 2604.00057 | [PDF](https://arxiv.org/pdf/2604.00057v1)

**作者:** Zeyu Jin `[一作]` (Tsinghua University), Jia Jia `[通讯]` (Tsinghua University)

**通讯引用:** 132557 | [OpenAlex ID](https://openalex.org/A5062225093)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GameSight两阶段模型，将足球解说生成拆解为视觉实体对齐与知识增强细化，生成具备精准实体指代和统计洞察的实时解说文本。

**💡 创新点**

创新点在于：①将解说任务视为知识增强视觉推理；②引入外部历史统计与内部比赛状态数据库；③结合SFT与GRPO微调视频多模态LLM，提升实体对齐准确率；④使用改进的SoccerKAG实现动态知识检索。

**🔧 技术方法**

技术包括：视频多模态LLM（Qwen2.5‑VL‑7B‑Instruct、MatchVision）、LLM（GPT‑4o）、SFT、GRPO、KAG（改进的SoccerRAG）以及内部上下文跟踪模块。

**📊 数据集**

使用SN‑Caption‑test‑align、MatchTime、Goal benchmark等公开数据集，并构建自定义内部游戏上下文与外部统计数据库。

**📈 对比分析**

与MatchVoice、MatchVision等基线对比，在实体对齐上比Gemini 2.5‑pro提高18.5%；段级评价BLEU、CIDER等指标与直播解说相近，游戏级Coh‑Metrix、情感得分和MOS均优于传统模型，显示明显性能提升。

**⚠️ 局限性**

局限性包括：依赖高质量视频与标注，复杂场景下实体对齐仍有误差；知识检索受数据库更新和时效性限制；模型规模大，训练和推理成本高。

---

## 5. UCMNet: Uncertainty-Aware Context Memory Network for Under-Display Camera Image Restoration

**arXiv ID:** 2604.00381 | [PDF](https://arxiv.org/pdf/2604.00381v1)

**作者:** Daehyun Kim `[一作]` (Hanyang University), Tae Hyun Kim `[通讯]` (Hanyang University)

**通讯引用:** 11219 | [OpenAlex ID](https://openalex.org/A5100438979)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于不确定性感知的轻量级上下文记忆网络UCMNet，用于下显示摄像头(UDC)图像的去畸变与高频细节恢复。

**💡 创新点**

创新点在于①引入高频不确定性驱动损失(HF‑UDL)以精准学习光衍射引起的高频失真；②设计Uncertainty‑Prior Transformer(UPT)，利用记忆与上下文记忆库根据不确定性映射自适应检索局部上下文；③整体网络参数量比前沿模型低30%且保持或提升性能。

**🔧 技术方法**

核心技术包括：频域卷积模块(FCM)、不确定性驱动损失(HF‑UDL)、双记忆库（Memory Bank & Context Bank）、方向交叉注意力、以及全局通道自注意力的Transformer。

**📊 数据集**

使用公开的POLED、TOLED以及合成SYNTH三大数据集（各含1024×2048或800×800分辨率的成对图像）进行训练与评估。

**📈 对比分析**

与MSUNet、DAGFE、PDCRN、UDCUNet、FSI、BNUDC等先进方法比较，UCMNet在POLED、TOLED、SYNTH测试集上PSNR/SSIM/LPIPS/DISTS均达到或超过最高分，且参数量和算力仅占前沿方法的约70%。

**⚠️ 局限性**

局限性包括：在极端光照或高噪声场景下仍可能出现细小纹理残留；不确定性映射与记忆检索机制对硬件推理速度有一定开销，未来需进一步压缩模型并提升实时性。

---

## 6. Deep Learning-Accelerated Surrogate Optimization for High-Dimensional Well Control in Stress-Sensitive Reservoirs

**arXiv ID:** 2604.00352 | [PDF](https://arxiv.org/pdf/2604.00352v1)

**作者:** Mahammad Valiyev `[一作]` (University of Southern California), Behnam Jafarpour `[通讯]` (University of Southern California)

**通讯引用:** 3179 | [OpenAlex ID](https://openalex.org/A5003736943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在压力敏感的非常规油藏中，提出一种深度学习加速的代理优化框架，用以寻找高维时间变化的井控策略，最大化累计产量。

**💡 创新点**

创新点包括：1）将井控视为连续高维优化问题，摆脱预设参数化；2）提出与优化过程对齐的“问题感知采样”策略，使训练数据分布与优化轨迹一致；3）将代理模型嵌入梯度约束优化，实现在几分钟内完成优化；4）通过代理辅助初始化和混合验证提升可靠性。

**🔧 技术方法**

技术手段：完全耦合的流动–地质力学数值模拟；全连接神经网络代理；梯度约束优化（如MATLAB fmincon）；问题感知采样（结合线性下降、常数/下降、组合轨迹）；自适应验证与增量训练。

**📊 数据集**

使用的“数据集”来自场规模耦合模拟，约 50–150 条 BHP 轨迹，覆盖不同的压力敏感系数（低/中/高）以及多种控制间隔（≈20 维）。

**📈 对比分析**

与传统基于完整物理模拟的优化（需数百次模拟、数小时/天）相比，代理优化在相同约束下实现累计产量误差 <5%，计算时间提升约 1000 倍，优化收敛稳定。对比分析显示代理在训练分布内表现优异，对外推误差可控。

**⚠️ 局限性**

局限：1）代理在训练分布边界外推时误差增大；2）优化受限于梯度约束，可能陷入局部最优；3）仅验证累计产量为目标，未涵盖经济/多目标；4）对更复杂储层（异质、天然裂缝）和不确定性需进一步扩展。

---

## 7. The Persistent Vulnerability of Aligned AI Systems

**arXiv ID:** 2604.00324 | [PDF](https://arxiv.org/pdf/2604.00324v1)

**作者:** Aengus Lynch `[一作]` `[通讯]`, Aengus Lynch

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

暂无摘要信息

**💡 创新点**

暂无摘要信息

**🔧 技术方法**

暂无摘要信息

**📊 数据集**

暂无摘要信息

**📈 对比分析**

暂无摘要信息

**⚠️ 局限性**

暂无摘要信息

---

## 8. Can Large Language Models Self-Correct in Medical Question Answering? An Exploratory Study

**arXiv ID:** 2604.00261 | [PDF](https://arxiv.org/pdf/2604.00261v1)

**作者:** Zaifu Zhan `[一作]` (University of Minnesota), Rui Zhang `[通讯]` (University of Minnesota)

**通讯引用:** 12105 | [OpenAlex ID](https://openalex.org/A5100675481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过实验比较了大语言模型在医学多选问答任务中，标准链式思维（CoT）提示与迭代自我反思（self‑reflective）提示的效果，系统跟踪并分析了模型在自我反思过程中的答案演变与错误修正情况。

**💡 创新点**

创新点在于将自我反思视为一种诊断工具，而非单纯的性能提升手段；通过跨数据集和模型规模的对比，揭示了自我反思在医学问答中的局限性与差异性；首次对自我反思在不同医学 QA 基准上的实际表现进行细粒度量化。

**🔧 技术方法**

使用的技术包括：零样本链式思维提示、迭代自我反思循环提示（在 GPT‑4o 与 GPT‑4o‑mini 上实现）、统计误差分析以及模型回答追踪与聚合；所有实验均基于 OpenAI API 的黑盒推理。

**📊 数据集**

所用数据集为：MedQA‑USMLE（美国医师执照考试题），HeadQA（西班牙医学专业考试题），以及 PubMedQA（基于 PubMed 摘要的科学推理题）。

**📈 对比分析**

比较方法：在每个数据集上分别测算 CoT 提示的基准准确率，并记录在每一步自我反思后模型给出的答案准确率；结果显示：MedQA 上自我反思可略提升准确率（约 +1%），HeadQA 与 PubMedQA 上则无提升甚至略降；随着反思次数增加，准确率不呈单调提升，且大模型往往需要更少的反思步数。

**⚠️ 局限性**

limitations: 仅使用提示式自我反思，未结合检索、外部验证或不确定性评估；仅在多选问答上评估，未涉及开放式临床推理或实际决策支持；自我反思只在答案层面进行，难以纠正中间推理错误；模型可能在自我评估时强化已有错误，导致确认偏差。

---

## 9. Diversity-Aware Reverse Kullback-Leibler Divergence for Large Language Model Distillation

**arXiv ID:** 2604.00223 | [PDF](https://arxiv.org/pdf/2604.00223v1)

**作者:** Hoang-Chau Luong `[一作]` (Rochester Institute of Technology), Lingwei Chen `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1045 | [OpenAlex ID](https://openalex.org/A5017524689)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种改进的反向KL散度目标——Diversity-aware RKL（DRKL），用于大语言模型的知识蒸馏。

**💡 创新点**

创新点在于剖析RKL梯度结构，发现非目标类梯度持续上推目标logit导致过度自信；DRKL通过去除该梯度并加强非目标监督，提升了学生模型的多样性与校准性。

**🔧 技术方法**

技术包括：梯度分解理论分析、基于目标/非目标二分类的RKL拆分、引入超参数γ的非目标正则化、与现有RKL变体的结合，以及在多种评估指标（ROUGE‑L、Distinct‑2、Negative Self‑BLEU等）上的实验验证。

**📊 数据集**

使用的训练数据为公开的指令‑响应数据集（14k样本），评估数据包括 Dolly、Self‑Instruct、Vicuna、Super‑Natural Instructions、Unnatural Instructions 等多个基准。

**📈 对比分析**

对比方法包括传统 FKL、RKL、对称KL、DKD、JS、SFKL、SRKL、AKL、AB 等 SOTA 蒸馏目标；实验显示 DRKL 在所有基准上均领先或相近，平均提升约 1–3 分 ROUGE‑L，并显著提升输出多样性与校准度。

**⚠️ 局限性**

局限性：DRKL 的性能受超参数 γ 影响，对极大词表或极低资源设置的鲁棒性未充分评估；同时该方法仍需在更大规模模型和更复杂任务上进一步验证。

---

## 10. Quantifying Gender Bias in Large Language Models: When ChatGPT Becomes a Hiring Manager

**arXiv ID:** 2604.00011 | [PDF](https://arxiv.org/pdf/2604.00011v1)

**作者:** Nina Gerszberg `[一作]` (Massachusetts Institute of Technology), Andrew Lo `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 51639 | [OpenAlex ID](https://openalex.org/A5109980718)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究量化大型语言模型在招聘场景中的性别偏见，并评估提示工程作为偏见缓解手段。

**💡 创新点**

提出了基于p值的定量偏见度量方法，并将姓名作为性别代理，系统地比较不同LLM和提示策略的偏见表现。

**🔧 技术方法**

使用非参数统计检验（卡方、Wilcoxon、Kolmogorov‑Smirnov）与自定义偏见公式相结合，探究提示对LLM决策的影响。

**📊 数据集**

采集了2000份匿名真实简历，结合美国人口普查选出的男女、性别中性姓名及其代词变体，构成完整实验集。

**📈 对比分析**

通过对比基线提示与两种偏见缓解提示（说明理由、DEI声明）在Hire、Qualified、Compensation三项指标上的p值和差异，发现大多数LLM虽更倾向于雇佣和评估女性，但在薪酬上仍偏低，提示手段效果有限。

**⚠️ 局限性**

局限在于姓名仅为性别代理、模型样本受限、偏见度量过度简化、实验仅覆盖招聘场景，无法涵盖更广泛的交叉偏见与实际招聘流程。

---

## 11. Quantifying Confidence in Assurance 2.0 Arguments

**arXiv ID:** 2604.00034 | [PDF](https://arxiv.org/pdf/2604.00034v1)

**作者:** Robin Bloomfield `[一作]` (University of London), John Rushby `[通讯]` (SRI International)

**通讯引用:** 7632 | [OpenAlex ID](https://openalex.org/A5020087247)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了一种基于消除模型、独立性和多样性假设的结构化、可组合的概率置信度评估方法，用以量化 Assurance 2.0 案例中的置信度。

**💡 创新点**

方法简化了传统基于概率逻辑的“乘积”或“疑惑求和”计算，避免了对不确定性强烈依赖的全局假设，提出了多种针对不同子命题组合（多样性、分区、包含、累积）的置信度计算公式，并能与贝叶斯网络结合处理复杂依赖。

**🔧 技术方法**

采用基本的概率运算（乘积、加法、Fréchet 上下界）以及贝叶斯网络推理；并在 Assurance 2.0 框架内引入“副主张”(sideclaim)的置信度乘积修正。

**📊 数据集**

无具体数据集，本文主要基于理论推导、案例演示和与已有方法（如 DST、BBN、GSM 等）的对比。

**📈 对比分析**

与传统的“乘积”“疑惑求和”方法以及基于 DST、BBN 的方法对比，说明该方法在避免逻辑不健全时产生不合理置信度、对子命题置信度差异更敏感并能揭示潜在弱点；在示例计算中表现更符合直觉且计算简单。

**⚠️ 局限性**

局限在于：对子命题间依赖的假设仍需人工判断，复杂的多重依赖情形需构造贝叶斯网络；方法在极端高度相关或极低置信度子命题时仍可能产生不确定结果；缺乏实证数据验证与实际工程案例的应用。

---

## 12. OmniSch: A Multimodal PCB Schematic Benchmark For Structured Diagram Visual Reasoning

**arXiv ID:** 2604.00270 | [PDF](https://arxiv.org/pdf/2604.00270v1)

**作者:** Taiting Lu `[一作]` (Pennsylvania State University), Mahanth Gowda `[通讯]` (Pennsylvania State University)

**通讯引用:** 949 | [OpenAlex ID](https://openalex.org/A5064270644)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OmniSch 基准，评估大模型在 PCB 原理图到网表图的生成，并进行了多模型对比实验。

**💡 创新点**

1) 大规模真实原理图数据集；2) 四项评估任务（视觉定标、图推理、几何空间推理、工具增强式代理搜索）；3) 结构化评估协议；4) 结合外部工具的 ReAct 视觉搜索框架。

**🔧 技术方法**

大模型推理、提示工程、ReAct 视觉搜索、基线视觉管道、图匹配与评估指标。

**📊 数据集**

OmniSch（1,854 张真实 PCB 原理图，109.9K 符号、245.4K 引脚、423.4K 文本、219.8K 空间加权网表）。

**📈 对比分析**

将 LMM 在零样本、提示工程和工具增强三种设置下与经典 YOLO+OCR 基线对比；大模型在符号检测、引脚检测、连线推理和网表生成等指标上远低于基线，且整体图结构匹配度仍显不足。

**⚠️ 局限性**

对文本提示高度依赖、拓扑推理不稳定、引脚与网名识别精度低、模型无法完整复制基线流程、工具增强虽提升但整体性能仍有限，且数据虽真实但仍未覆盖所有复杂图形场景。

---

## 13. Finite-Time Analysis of Projected Two-Time-Scale Stochastic Approximation

**arXiv ID:** 2604.00179 | [PDF](https://arxiv.org/pdf/2604.00179v1)

**作者:** Yitao Bai `[一作]` (University of Texas at Austin), Justin Romberg `[通讯]` (Georgia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了投影线性双时间尺度随机逼近在常数步长与Polyak–Ruppert平均下的有限时收敛性，给出了均方误差的逼近-统计两项分解。

**💡 创新点**

创新点在于首次给出投影TTSA的非渐近误差上界，明晰了子空间选择导致的逼近误差与平均窗口长度导致的统计误差之间的分离。

**🔧 技术方法**

利用常数步长、线性投影、马尔可夫差分噪声假设以及解析的全局收敛分析，得到统计常数与逼近常数的显式表达。

**📊 数据集**

通过合成线性耦合系统和GTD强化学习的9状态MDP实验验证理论。

**📈 对比分析**

与传统无投影或单时间尺度SA比较，PR平均的统计误差随迭代数按O(1/T)衰减，逼近误差受子空间决定，实验显示该理论符合实际。

**⚠️ 局限性**

局限在于假设子空间投影线性、矩阵可逆及噪声满足马尔可夫差分且方差有界，未考虑非线性或高阶耦合情况。

---

## 14. Offline Constrained RLHF with Multiple Preference Oracles

**arXiv ID:** 2604.00200 | [PDF](https://arxiv.org/pdf/2604.00200v1)

**作者:** Brenden Latham `[一作]` (University of Iowa), Mehrdad Moharrami `[通讯]` (University of Iowa)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5089114577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并分析了离线多目标RLHF的双重优化框架，利用多份人类偏好数据在约束下学习策略；

**💡 创新点**

首次给出多偏好oracle下的有限样本理论保证，并通过KL正则化得到闭式Gibbs策略，整体算法仅需一维双重优化；

**🔧 技术方法**

使用最大似然估计(Bradley‑Terry)求解奖励参数，KL‑正则化的Lagrangian与指数族属性，投影梯度下降求解双重变量；

**📊 数据集**

在PKU‑SafeRLHF数据集上进行实验，数据包含约74k训练样本和7k测试样本，采用Sentence‑BERT特征与Alpaca‑7B作为参考策略；

**📈 对比分析**

与原始策略（π₀）及其他基准相比，优化后策略在安全性上几乎无违例，帮助度提升约27%，整体性能明显优于基线；

**⚠️ 局限性**

局限在于假设单独最大似然独立估计、参考策略可知、对多目标和非KL多元约束的扩展仍需进一步验证。

---

## 15. An Empirical Study on How Architectural Topology Affects Microservice Performance and Energy Usage

**arXiv ID:** 2604.00080 | [PDF](https://arxiv.org/pdf/2604.00080v1)

**作者:** Irena Ristova `[一作]` (Vrije Universiteit Amsterdam), Vincenzo Stoico `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5075101570)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对六种典型微服务拓扑（顺序 Fan‑Out、并行 Fan‑Out、带分支链、层级树、概率树、密集 Mesh）在 5、10、20 个服务规模下，使用 μBench 生成相同 CPU‑密集型工作负载，量化其能耗、吞吐量、响应时间、CPU 利用率和失效率，并对拓扑对能效与性能的影响进行实验验证。

**💡 创新点**

首次系统地将拓扑结构视为能效和性能的关键因素，提出通过合成微服务系统对不同拓扑的能耗/性能耦合关系进行量化分析，揭示“密集 Mesh”导致能耗上升且吞吐量下降，概率树在小规模下能耗最低的结论。

**🔧 技术方法**

使用 μBench 框架自动生成 Kubernetes 微服务实例；利用 Locust 生成工作负载；通过 Prometheus + EnergiBridge（Intel RAPL）采集 CPU、内存、能耗及性能指标；采用实验跑十次、全因子设计与 Kruskal‑Wallis / Mann‑Whitney U + Holm 校正等统计方法。

**📊 数据集**

基于 μBench 的合成数据集，包含六种拓扑在三种规模（5、10、20）下的 180 次实验跑，记录 CPU 包能耗（kJ）、吞吐量（rps）、响应时间（s）及失效率（%）。

**📈 对比分析**

对同一工作负载下的不同拓扑进行对比，结果显示：Mesh 拓扑能耗最高、吞吐量最低、响应时间最长；顺序/并行 Fan‑Out、层级树表现相近且更高效；概率树在 5–10 服务时能耗最低。统计检验显示拓扑与能耗、吞吐量、响应时间显著相关，且随规模放大拓扑差异显著增强。

**⚠️ 局限性**

实验仅在单节点 Minikube 集群上进行，缺乏跨节点网络延迟；工作负载仅为 CPU‑密集型，未覆盖 I/O、数据库或异步交互场景；能耗测量仅限 CPU+DRAM，忽略网络、存储能耗；样本规模有限（10 次重复），可能对罕见效应缺乏统计功效。

---

## 16. Do Language Models Know When They'll Refuse? Probing Introspective Awareness of Safety Boundaries

**arXiv ID:** 2604.00228 | [PDF](https://arxiv.org/pdf/2604.00228v1)

**作者:** Tanay Gondil `[一作]` `[通讯]` (Purdue University), Tanay Gondil (Purdue University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型是否能在生成回复前准确预测其拒绝行为。

**💡 创新点**

创新点在于将信号检测理论与置信度校准结合，量化模型对安全边界的直觉敏感性，并提出基于置信度的安全路由策略。

**🔧 技术方法**

使用信号检测理论、ECE校准评估、误差分析和多模型对比实验等技术。

**📊 数据集**

使用300条基于10个敏感主题、5个危害级别的请求及其2个改写版本，共900个数据点。

**📈 对比分析**

通过对四款前沿模型（Claude Sonnet 4/4.5、GPT-5.2、Llama 3.1 405B）的d′、偏差、准确率和ECE等指标进行对比，结果显示Claude 4.5在精度（95.7%）与校准（ECE=0.017）上优于其他模型。

**⚠️ 局限性**

实验采用无上下文的“新鲜上下文”设计，未检验对话历史影响，且置信度评估缺乏人类标注验证，可能限制实用性。

---

## 17. Beyond Latency: A System-Level Characterization of MPC and FHE for PPML

**arXiv ID:** 2604.00169 | [PDF](https://arxiv.org/pdf/2604.00169v1)

**作者:** Pengzhi Huang `[一作]`, G. Edward Suh `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文介绍了 IEEEtran LaTeX 模板的使用方法，包括文档类选项、前置元素、正文元素、后置元素等，并提供了相应的示例文件。

**💡 创新点**

系统化整理了模板使用说明，提供了统一的排版流程和易于引用的示例，帮助作者快速完成符合 IEEE 期刊/会议格式的论文撰写。

**🔧 技术方法**

采用 LaTeX 编程与模板设计技术，并说明了如何通过 XML 转换实现最终排版与 IEEEXplore 的兼容。

**📊 数据集**

无数据集，本文属于技术文档说明而非实验研究。

**📈 对比分析**

无实验或性能比较，本文仅为使用说明文档，没有涉及算法或模型的性能评估。

**⚠️ 局限性**

仅适用于 IEEEtran 1.8b 版本；细节需参照 IEEEtran_HOWTO.pdf，且本文不包含最终排版效果，需通过 IEEE 生产流程完成。

---

## 18. Homogenization of HTS coils with the h, h-phi, and t-omega foil conductor model

**arXiv ID:** 2604.00154 | [PDF](https://arxiv.org/pdf/2604.00154v1)

**作者:** Elias Paakkunainen `[一作]` (Technical University of Darmstadt), Sebastian Schöps `[通讯]` (Technical University of Darmstadt)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并验证了一种将高温超导(HTS)线圈归一化为均匀块的薄层导线模型(FCM)，并将其与三种磁场一致性有限元形式(h-(full)、h-ϕ、t-ω)结合，用于仿真绝缘HTS线圈的电磁行为。

**💡 创新点**

创新点在于：①将传统薄层导线模型推广到磁场一致性形式；②通过在导体域内引入连续电压分布函数(Φ(α))实现对每匝电流的弱约束；③在t-ω形式下通过强制α方向无电流进一步削减自由度；④在三维大规模模型中实现了显著的计算加速与自由度压缩。

**🔧 技术方法**

使用了有限元软件GetDP与Gmsh进行网格与求解，采用了多种数学形式(h-(full)、h-ϕ、t-ω)和传统h-ϕ参考模型；实现了对功率律电阻、等效填充因子、以及磁场与温度相关的临界电流密度的建模；并在仿真中引入了连续电压基函数和切割基函数以满足电流约束。

**📊 数据集**

主要数据集为HTS涂层导线(如Fujikura CC)的材料参数（功率指数n=25、填充因子λ=0.01、临界电流密度等），以及两组几何模型：①20匝薄盘线圈(2D轴对称与3D)；②三排50匝racetrack线圈堆叠(1/8域)。

**📈 对比分析**

通过与传统h-ϕ参考模型对比，计算即时损耗的决定系数R²均>0.99；在3D薄盘线圈中，t-ω FCM的自由度仅94k，求解总时间91min，而完整解析模型则需360k DoF、3600min。对racetrack堆叠，t-ω FCM实现22倍速度提升、DoF下降78%，平均损耗误差仅0.5%。

**⚠️ 局限性**

局限性包括：①假设线圈匝为同心圆，忽略实际螺旋结构；②仅适用于绝缘线圈，非绝缘配置需进一步改造；③对极限高温或极低频等特殊工况的适用性未验证；④仍需手工设置连续电压函数Φ(α)与切割基函数，模型建模工作量较大。

---

## 19. Scalable Identification and Prioritization of Requisition-Specific Personal Competencies Using Large Language Models

**arXiv ID:** 2604.00006 | [PDF](https://arxiv.org/pdf/2604.00006v1)

**作者:** Wanxin Li `[一作]` (University of British Columbia), Anthony S. Boyce `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一套基于大型语言模型（LLM）的端到端流程，用于从招聘岗位说明中识别并优先排序“岗位特定”个人能力（PC），并通过动态少量样本提示、反思式自我改进、相似性过滤与多阶段验证来确保结果的准确性和可解释性。

**💡 创新点**

创新点在于首次在规模化 PC 识别中显式区分岗位级别与岗位特定 PC，利用动态少量样本学习与反思机制提升生成质量，并通过多层过滤与验证避免与标准 PC 库混淆，从而显著降低超出范围的错误率。

**🔧 技术方法**

核心技术包括 Claude Sonnet 4（主推理）、Claude Haiku 3.5（验证）、动态少量样本提示、LLM 作为评判者与改进者、相似性度量（sentence‑transformer 109 M 参数）以及人类专家在循环中不断优化提示。

**📊 数据集**

使用亚马逊内部的 140 条项目经理（PM）与产品经理‑技术（PMT）岗位招聘需求作为实验数据，并在 26% 的训练集构建示例库、50% 的开发集进行提示调优，剩余 24% 作为测试集评估。

**📈 对比分析**

与专家标注的基准相比，系统在测试集上实现了约 0.76–0.78 的 top‑1 精度，远低于 0.10 的超出范围目标（实际 0.07），而 top‑2/3 精度可达 0.86–0.93；消融实验表明反思、评估、再生成与验证模块对保持高精度至关重要。

**⚠️ 局限性**

主要局限包括：对 PC 细粒度和类别分类的误判仍占 13–31%，仅针对 PM/PMT 岗位验证，模型依赖高性能 LLM 及大量人工复核，且内部数据和提示不公开，限制了外部复现与推广。

---

## 20. From Domain Understanding to Design Readiness: a playbook for GenAI-supported learning in Software Engineering

**arXiv ID:** 2604.00120 | [PDF](https://arxiv.org/pdf/2604.00120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 21. Are they human? Detecting large language models by probing human memory constraints

**arXiv ID:** 2604.00016 | [PDF](https://arxiv.org/pdf/2604.00016v1)

**作者:** Simon Schug `[一作]` (Princeton University), Brenden M. Lake `[通讯]` (Princeton University)

**通讯引用:** 5751 | [OpenAlex ID](https://openalex.org/A5011713946)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过设计序列回忆工作记忆实验，探测在线参与者是否为大语言模型（LLM）并对其行为进行区分。

**💡 创新点**

创新点在于利用人类固有的工作记忆限制（如有限容量、首位/尾位优势）作为判别LLM的“人类约束”，而非传统的逻辑或注意力检查。

**🔧 技术方法**

采用分层贝叶斯逻辑回归模型进行认知异常检测，并通过系统提示或微调让LLM尝试模仿工作记忆效应；同时实现LLM与实验界面的文本交互。

**📊 数据集**

使用来自Prolific的100名在线参与者数据和多款LLM（Anthropic、Google、OpenAI、Hugging Face等）以及专门微调的Centaur模型在字母序列回忆任务中的表现。

**📈 对比分析**

对比时以平均准确率、序列位置效应、负荷效应等指标，并用ROC/AUROC评估异常检测；LLM‑Human几乎100%可被检测，LLM‑WM和Centaur的检测难度更高，AUROC在0.73–1.00之间。

**⚠️ 局限性**

局限性包括：LLM随着规模提升可能更易模仿人类约束；检测依赖特定实验设计，若实验被LLM学习后可能失效；反应时等其他特征不稳定；需要保密人类数据以维持检测有效性。

---

## 22. Data-Driven Reachability Analysis via Diffusion Models with PAC Guarantees

**arXiv ID:** 2604.00283 | [PDF](https://arxiv.org/pdf/2604.00283v1)

**作者:** Yanliang Huang `[一作]` (Technical University Of Munich), Amr Alanwar `[通讯]` (Technical University Of Munich)

**通讯引用:** 544 | [OpenAlex ID](https://openalex.org/A5102709386)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于扩散模型的无模型可达性分析框架，利用轨迹数据直接学习状态分布并给出PAC覆盖保证。

**💡 创新点**

创新点在于将扩散模型的重构误差作为不符合性得分，并通过Learn‑Then‑Test校准阈值，既无需构造显式动力学模型，又能在高维状态空间实现可覆盖的可达性预测。

**🔧 技术方法**

核心技术包括DDPM生成模型、基于重构误差的非符合性得分、Learn‑Then‑Test自适应阈值校准以及对高维状态空间的分布式推断。

**📊 数据集**

实验使用了三套数据集：受迫Duffing振子（2维）、平面四旋翼（6维）以及Gray‑Scott反应扩散系统（8192维）生成的轨迹数据。

**📈 对比分析**

与正态流、VAE和Christoffel多项式基准对比，DDPM得分在IoU、精度和误报率上表现最优，尤其在高维Gray‑Scott系统中显著优于VAE。

**⚠️ 局限性**

局限性包括对初始分布下界假设的依赖、扩散模型训练与推理计算成本较高，以及在极高维或非连续可达性结构下可能仍需进一步改进。

---

## 23. PASM: Population Adaptive Symbolic Mixture-of-Experts Model for Cross-location Hurricane Evacuation Decision Prediction

**arXiv ID:** 2604.00074 | [PDF](https://arxiv.org/pdf/2604.00074v1)

**作者:** Xiao Qian `[一作]` (University of Delaware), Shangjia Dong `[通讯]` (University of Delaware)

**通讯引用:** 2146 | [OpenAlex ID](https://openalex.org/A5053335194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于人口自适应符号混合专家（PASM）的避灾决策预测框架，能够在不同州之间进行可解释且鲁棒的迁移。

**💡 创新点**

创新点在于将LLM驱动的符号回归与混合专家架构相结合，配合学习路由器和输入相关的系数自适应，从而捕捉多样化人群的行为模式。

**🔧 技术方法**

采用了LaSR进行符号回归、UMAP+HDBSCAN进行无监督亚群体发现、Mixture‑of‑Experts网络、学习路由器以及系数自适应网络，实现端到端训练。

**📊 数据集**

使用了在飓风哈维（德州）和飓风伊玛（佛罗里达和佐治亚）后收集的822份匿名家庭调查数据。

**📈 对比分析**

与XGBoost、TabPFN、GPT‑5‑mini、MAML、原型网络、匹配网络和层级聚类+LR等基线比较，PASM在佐治亚州测试集上MCC达0.607，比其他方法提升约40–82%，ROC‑AUC也显著更高。

**⚠️ 局限性**

局限包括：模型属于灰盒，解释性受无监督聚类和混合专家影响；训练需要大量LLM调用，计算成本高；硬聚类假设可能过于简化连续或层级的人口异质性；仅在美国飓风场景验证，跨文化迁移需进一步研究。

---

## 24. Go Big or Go Home: Simulating Mobbing Behavior with Braitenbergian Robots

**arXiv ID:** 2604.00350 | [PDF](https://arxiv.org/pdf/2604.00350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 25. NeuroVase: A Tangible Mobile Augmented Reality Learning System for Neurovascular Anatomy and Stroke Education

**arXiv ID:** 2604.00296 | [PDF](https://arxiv.org/pdf/2604.00296v1)

**作者:** Bahar Jahani `[一作]`, Yiming Xiao `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一个基于平板的移动增强现实学习平台 NeuroVase，结合可触摸提示卡与三维血管模型，为脑血管解剖与卒中教育提供交互式、可扩展的学习体验。

**💡 创新点**

创新点在于：① 双模式学习——提示卡既可作为离线复习工具，也可作为 AR 内容触发器；② 以临床神经病专家共建的分阶段课程体系；③ 通过实物卡片与 AR 的融合，提升空间认知与学习动机。

**🔧 技术方法**

技术实现包括 Unity3D 开发、Vuforia 图像跟踪、Unity Volume Renderer 进行 MR 数据可视化、手势交互（多指缩放/平移/旋转）以及 QR 码增强识别。

**📊 数据集**

数据集涵盖：BCI‑DNI 大脑分区图谱、单一健康受试者的 T1‑MRI 与 TOF‑MRA（构建血管及血管领地模型），以及 1298 例急性卒中患者的血管领地图谱。

**📈 对比分析**

对比方法：随机对照 40 名参与者（20名 AR 组、20名纸本组），通过前后测问卷评估学习效果。AR 组提升 29.45%（从 40.83% 至 70.28%），纸本组提升 26.67%；虽然 AR 组在学习收益上略优，但差异不显著。SUS 评分 90，表明 AR 系统高度易用，用户体验评分均优于纸本。

**⚠️ 局限性**

局限性包括：样本量有限、AR 组先验知识略高、偶发卡片识别卡顿、内容深度仍可提升、未在不同专业背景或临床实践者中验证泛化性。

---

## 26. Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation

**arXiv ID:** 2604.00131 | [PDF](https://arxiv.org/pdf/2604.00131v1)

**作者:** Ashish Rana `[一作]` (NEC Laboratories Europe), Carolin Lawrence `[通讯]` (NEC Laboratories Europe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种自适应记忆控制框架Oblivion，改进LLM代理的长期记忆管理，通过读写解耦实现记忆衰减与强化控制。

**💡 创新点**

将记忆视为控制问题，提出记忆衰减驱动的可访问性降低、读写路径解耦、基于不确定性门控的检索、层级化记忆结构及DAG式查询扩展与反馈驱动的记忆强化。

**🔧 技术方法**

采用Ebbinghaus遗忘曲线的指数衰减、LLM-as-a-judge估计不确定性、语义嵌入相似度、图式查询扩展、层级化工作记忆与持久存储，以及自适应控制循环。

**📊 数据集**

在LongMemEval（静态多会话QA）和GoodAILTM（动态长序交互）两大基准上进行评估，并使用Phi-4-mini、Qwen3-30B等多种LLM。

**📈 对比分析**

与直接检索基线（LME-RFT）及记忆增强基线（EverMemOS、BeyondPrompts）进行对比，Oblivion在静态QA平均准确率提升约1–5个百分点，在动态交互中解决场景数提升约30%，且在大规模长上下文下显著降低成本与延迟。

**⚠️ 局限性**

需要针对任务调优衰减温度T；在小型LLM上性能受限；同一LLM承担提取、估计与生成，导致耦合；评估范围仅限文本基准，未覆盖多语言、多模态或专业域。

---

## 27. The Energy Footprint of LLM-Based Environmental Analysis: LLMs and Domain Products

**arXiv ID:** 2604.00053 | [PDF](https://arxiv.org/pdf/2604.00053v1)

**作者:** Alicia Bao `[一作]` (University of North Carolina at Chapel Hill), Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对气候领域的两个检索增强生成（RAG）聊天机器人（ChatNetZero 和 ChatNDC）以及通用 GPT‑4o‑mini 在实际用户查询中的推理能耗进行实验评估，并将能耗拆解为检索、生成和反幻觉检测三大步骤；同时对生成结果的事实准确性和“虚构”程度进行人工评估；并把这些能耗与质量指标进行对比。

**💡 创新点**

创新点在于：① 将能耗评估从单一模型调用扩展到完整应用工作流；② 通过 API 响应时间、GPU 负载和 PUE 等参数构建端到端能耗估算公式；③ 将工作流分解为检索、生成和反幻觉检测三个可测量子组件；④ 结合能耗与回答质量（事实准确度和虚构率）的双重度量，为 RAG 系统的可持续设计提供可操作的权衡框架。

**🔧 技术方法**

主要技术包括：基于 OpenAI GPT‑4o‑mini 的 API 调用；利用 tiktoken 计数 token；使用公式 1‑2（结合 Jegham 等的框架）估算 GPU 与 CPU 能耗；检索阶段采用 Google Colab CPU 计算；反幻觉检测分为基于余弦相似度的轻量级方法（CNZ）和额外 GPT‑4o‑mini 调用（ChatNDC）。

**📊 数据集**

数据集：构造了 102 个符合 Bloom 分类的气候领域问题，涵盖知识、理解、应用、分析、评价与创造六类；其中 48 个为知识类，约 10 个为其他类；并挑选 19–25 题做人工事实准确性与虚构率评估。

**📈 对比分析**

比较方法：对四种系统（CNZ、ChatNDC、GPT‑4o‑mini、GPT‑4o‑mini 200）在不同时间段（上午、下午、晚上）和地理位置（荷兰）下多次运行，记录执行时间、token 使用并转化为能耗；随后将能耗与回答质量（事实分数、虚构分数）进行散点和相关性分析；结果显示 ChatNDC 能耗最高（4.53×10⁻³ kWh/问），CNZ 最低（4.08×10⁻⁴ kWh/问）；高能耗不必然带来更高事实准确性，且 CNZ 在低能耗下虚构率最低。

**⚠️ 局限性**

局限性包括：① 能耗估算基于 API 延迟而非服务器端真实消耗；② 仅评估单轮问答，未覆盖多轮对话或会话级别能耗；③ 所有系统共用 GPT‑4o‑mini 作为核心模型，无法评估不同模型架构的通用性；④ RAG 工作流为静态设计，缺乏根据查询复杂度动态调节计算资源的机制；⑤ 受限于实验环境（Colab CPU）和数据集规模，外部有效性有限。

---

## 28. Learning to Play Blackjack: A Curriculum Learning Perspective

**arXiv ID:** 2604.00076 | [PDF](https://arxiv.org/pdf/2604.00076v1)

**作者:** Amirreza Alasti `[一作]` (Leibniz University Hannover), Theresa Eimer `[通讯]` (Leibniz AI Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用大型语言模型动态生成动作分阶段课程，逐步向 Blackjack RL 代理引入 Hit、Stand、Double、Split、Surrender、Insurance 等动作，并在 8 副牌环境下评估其对 Tabular Q‑learning 与 DQN 的性能提升。

**💡 创新点**

创新之处在于把 LLM 当作自动“教练”，根据训练阶段的表现摘要实时生成并调整基于动作复杂度的课程，使学习过程可自适应、结构化，并避免一次性暴露全部动作导致的探索瓶颈。

**🔧 技术方法**

核心技术包括 Google Gemini 2.0 Flash LLM 生成课程、DQN（MLP 结构、经验回放、目标网络、epsilon‑greedy 与学习率自适应）以及 Tabular Q‑learning，配合奖励塑形、动作可用性屏蔽与阶段性阈值设置。

**📊 数据集**

实验使用 8 副牌 Blackjack 模拟环境（90% 渗透率），在 10 次独立随机种子下进行 500,000 次训练回合、100,000 次评估回合，收集 win‑rate、bust‑rate 与训练时间等指标。

**📈 对比分析**

与无课程基线对比，DQN 的平均胜率从 43.97% 提升至 47.41%，破局率从 32.9% 降至 28%，训练时间缩短 74%，并发现峰值胜率集中在第 4 阶段（完整基本策略）；Tabular Q 亦有提升但幅度较小。

**⚠️ 局限性**

局限性包括仅适用于可分层离散动作空间，依赖特定 LLM 版本导致可复现性受限；课程仅单向进阶，无法评估停留在子集的最优；未覆盖下注策略、动态桌面限额以及更复杂的赌场规则。

---

## 29. Frege in the Flesh: Biolinguistics and the Neural Enforcement of Syntactic Structures

**arXiv ID:** 2604.00291 | [PDF](https://arxiv.org/pdf/2604.00291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 30. Real Time Local Wind Inference for Robust Autonomous Navigation

**arXiv ID:** 2604.00343 | [PDF](https://arxiv.org/pdf/2604.00343v1)

**作者:** Spencer Folk `[一作]` `[通讯]`, Spencer Folk

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

未提供

**💡 创新点**

未提供

**🔧 技术方法**

未提供

**📊 数据集**

未提供

**📈 对比分析**

未提供

**⚠️ 局限性**

未提供

---

## 31. Cybersecurity Risk Assessment for CubeSat Missions: Adapting Established Frameworks for Resource-Constrained Environments

**arXiv ID:** 2604.00303 | [PDF](https://arxiv.org/pdf/2604.00303v1)

**作者:** Jonathan Shelby `[一作]` `[通讯]` (University of Oxford), Jonathan Shelby (University of Oxford)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 CubeSat 的资源受限环境，提出并实施了基于 NIST/ISO 框架的定制化风险评估方法，包含 42 条漏洞注册表、Security‑per‑Watt (SpW) 量化启发式以及分布式安全范式 (DSP)，并通过场景化分析进行验证。

**💡 创新点**

创新点在于将传统成熟框架的控制意图进行资源约束化改造，引入 SpW 计算以衡量功耗与安全收益比例，提出 DSP 将事件响应转为自律的星座级功能，并通过对比分析展示了 ECC 与 RSA、集中式与分布式响应的显著效率提升。

**🔧 技术方法**

使用 STRIDE、MITRE ATT&CK 与 CVSS v3.1 对漏洞进行编码；采用椭圆曲线密钥交换与 AEAD（如 Curve25519/ChaCha20‑Poly1305）实现通信加密；利用理论功耗模型、ARM Cortex‑M MCU 参考基准以及情景式计算评估安全效益。

**📊 数据集**

数据来源为文献综述与行业规范汇编的 42 条漏洞注册表，结合典型 CubeSat 规格（功率预算、通信窗口等）构成情景参数；未使用公开的实验数据集，而是依托已发表的硬件功耗与加密性能基准。

**📈 对比分析**

采用场景化对比法，计算每项控制的安全收益 SG 与功耗 P_operational，得到 SpW = SG / P_operational；结果显示 ECC 方案相较 RSA 提升 2.7 倍 SpW，DSP 相比集中式响应提升 1.98 倍 SpW，且功耗分别下降约 65% 与 55%，安全效果下降仅约 10.5%。

**⚠️ 局限性**

局限性包括：未在真实 CubeSat 硬件上实验验证 SpW 参数；CVSS 评分与攻击概率基于文献推断，缺乏现场测试；假设的攻击者能力与资源有限；方法主要针对 1U CubeSat，虽然可扩展但缺乏跨平台实证；缺乏对运营环境中动态威胁与不可预见硬件变异的评估。

---

## 32. ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving

**arXiv ID:** 2604.00136 | [PDF](https://arxiv.org/pdf/2604.00136v1)

**作者:** Annette Taberner-Miller `[一作]` `[通讯]` (Independent Researcher), Annette Taberner-Miller (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了ParetoBandit，一个基于成本感知上下文Bandit的在线自适应LLM路由器，能够在开放式请求流中闭环满足美元预算，并能快速适应价格和质量漂移及热插拔模型。

**💡 创新点**

在单一系统中首次将预算闭环原始-对偶PACER、几何遗忘有限记忆、运行时模型注册与强制探索、以及成本与质量双重目标整合，解决了非平稳性、预算控制与模型更新三大部署难题。

**🔧 技术方法**

使用LinUCB上下文Bandit + Lagrangian预算松弛 + EMA平滑 + 对偶变量 + 几何遗忘 + 线性回归足够统计 + 预热先验 + PCA嵌入 + 低延迟推理路径等技术。

**📊 数据集**

基于9个公开基准（MMLU、GSM8K、HellaSwag、BIG-Bench Hard、ARC-Challenge、OpenBookQA、WinoGrande、TruthfulQA、MBPP）的1.2万条提示，生成完整的奖励-成本矩阵。

**📈 对比分析**

与Naive Bandit、Forgetting Bandit、Recalibrated Bandit等基线在四个实验（预算平衡、价格漂移、质量衰退、冷启动）对照；ParetoBandit在预算合规性≤4%偏差、价格下跌时即刻提升质量、质量衰退能自动恢复且预算不超支，冷启动模型在≈50步内被有效采用；整体质量提升≥90%且成本控制在设定阈值内。

**⚠️ 局限性**

仅离线评估，未测试实时延迟与反馈延迟；使用机器评判奖励，缺少稀疏人类反馈；仅 per-request 预算，未实现聚合预算；冷启动探索需付出成本；未考虑推理延迟或质量下限需求。

---

## 33. Learning to Shuffle: Block Reshuffling and Reversal Schemes for Stochastic Optimization

**arXiv ID:** 2604.00260 | [PDF](https://arxiv.org/pdf/2604.00260v1)

**作者:** Lam M. Nguyen `[一作]` (IBM Research, Thomas J. Watson Research Center), Jayant Kalagnanam `[通讯]` (IBM Research, Thomas J. Watson Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了一种基于LLM自动化搜索得到的无放回SGD打乱规则，并在统一理论框架下剖析其块重排与配对反转两大结构对梯度方差和顺序敏感性的影响。

**💡 创新点**

①利用LLM驱动的程序演化自动发现高效的打乱策略；②将该策略拆解为块重排和配对反转两种结构，分别在理论上证明可降低前缀梯度方差常数并消除二阶顺序敏感；③在实验与理论双重验证的基础上提供结构化打乱设计的新思路。

**🔧 技术方法**

LLM-guided程序演化（OpenEvolve）、无放回SGD的统一打乱分析框架、方差分解与二阶展开的理论证明、以及配对反转的对称平均技术。

**📊 数据集**

经典机器学习基准：4*a9a、4*breast_cancer、4*digits、4*california、4*boston、4*diabetes、4*MNIST、4*FashionMNIST。

**📈 对比分析**

在相同学习率调度下，将自适应打乱（APR）与传统的随机重排（RR）、一次性打乱（SO）及增量梯度（IG）在SGD和Adam两种优化器中进行对比；实验结果表明APR在常数和递减学习率下均能显著降低训练损失并提升收敛稳定性，尤其在较大学习率时优势最为明显。

**⚠️ 局限性**

仍需人工设定块大小与阈值，理论假设平滑性且未覆盖极端非凸或高度异构数据；LLM搜索仅在离线阶段进行，未证明其在大规模模型或在线搜索中的可扩展性和性能一致性。

---

## 34. Agentic AI and Occupational Displacement: A Multi-Regional Task Exposure Analysis of Emerging Labor Market Disruption

**arXiv ID:** 2604.00186 | [PDF](https://arxiv.org/pdf/2604.00186v1)

**作者:** Ravish Gupta `[一作]` (BigCommerce), Saket Kumar `[通讯]` (University at Buffalo State University of New York)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

扩展Acemoglu–Restrepo任务暴露框架，提出Agentic Task Exposure（ATE）评分，对agentic AI在五大美国技术中心的职业影响进行量化评估。

**💡 创新点**

将端到端工作流覆盖、AI能力与地区采用速度三要素融入任务级暴露度量，并通过远程工作调整来捕捉地区差异，构建多维度的工作岗位替代预测。

**🔧 技术方法**

使用O*NET任务与能力数据、AI能力评分（CAP）、工作流覆盖因子（COV）以及逻辑S曲线的地区采用速度模型，配合Python脚本实现ATE计算。

**📊 数据集**

O*NET 30.2任务与能力数据库、BLS OEWS就业与工资统计、QCEW行业就业数据、AI招聘与采用调查、CPS远程工作比例等多源公开数据。

**📈 对比分析**

与独立的AIOE和GPT-4任务暴露指数进行相关性检验（ρ≈0.84、0.72），并在不同地区、年份进行敏感性与对比分析，模型显示大多数信息密集职业的ATE≥0.35，且与2024-2025行业就业变化呈显著负相关，验证了预测能力。

**⚠️ 局限性**

COV关键字规则的假阴性导致ATE为上限；远程工作调整仅基于宏观比例，缺乏企业层面细节；CAP评分受限于现有AI基准，未覆盖所有技术；未考虑法律与监管约束对实际替代的限制；研究仅聚焦美国五大都市，国际推广需重新校准。

---

## 35. Advancing Complex Video Object Segmentation via Tracking-Enhanced Prompt: The 1st Winner for 5th PVUW MOSE Challenge

**arXiv ID:** 2604.00395 | [PDF](https://arxiv.org/pdf/2604.00395v1)

**作者:** Jinrong Zhang `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29352 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在复杂视频目标分割任务中提出TEP框架，利用外部跟踪模型和多模态大语言模型为SAM3生成跟踪增强提示，提高对小目标和语义占主导目标的分割性能。

**💡 创新点**

创新点在于将跟踪模型和大语言模型的先验信息动态注入SAM3，而非仅改进记忆机制；实现无训练、模块化的提示增强，显著提升复杂场景下的目标理解与适应。

**🔧 技术方法**

核心技术包括目标分类（基于掩膜面积与MLLM判别）、小目标跟踪（使用SUTrack图像提示跟踪）、语义占主导目标检测（使用Qwen3.5生成文本提示并进行目标检测）、提示融合（IoU与置信度驱动的提示切换）。

**📊 数据集**

使用MOSE v2数据集（约5,000条视频，包含失踪/再现、极端遮挡、低照度等挑战），并在PVUW 2026 Complex Video Object Segmentation挑战赛中进行评估。

**📈 对比分析**

与SAM3基线相比，TEP提升整体J&F从46.63%到56.91%，J、F和重现指标均显著提升（如J&F_reappear从31.54%提升至44.11%），在挑战赛中获得第一名。

**⚠️ 局限性**

局限性包括对外部跟踪模型与MLLM的依赖，若提示生成不准会导致误导SAM3；在极端遮挡/图像质量极差时仍可能出现漂移；对不同目标类型的阈值需要手动调参，影响自动化程度。

---

## 36. The Mystery Deepens: On the Query Complexity of Tarski Fixed Points

**arXiv ID:** 2604.00268 | [PDF](https://arxiv.org/pdf/2604.00268v1)

**作者:** Xi Chen `[一作]` (Columbia University), Mihalis Yannakakis `[通讯]` (Columbia University)

**通讯引用:** 30617 | [OpenAlex ID](https://openalex.org/A5043084405)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种基于安全部分信息函数框架的查询算法，能够在对[ n]4 超平面求Tarski不动点时仅用 O(log^2 n) 次查询，同时推广到任意常数维度 k 的上界为 O(log^⌈(k-1)/3⌉+1 n)。

**💡 创新点**

创新点在于首次将安全部分信息函数直接用于设计 Tarski 不动点算法，并构造了一种候选集合的完整刻画，使得在安全信息下可以有效剪枝；此外，提出了新的 O(log^2 n) 的查询上界，填补了 k=4 的未知空缺。

**🔧 技术方法**

主要技术包括：①安全部分信息函数的定义与安全性约束的利用；②候选集合的构造与扩展；③在 k=3 时通过几何引理证明常数比例剪枝；④归约将 (n,k+1) 问题化简为 ^*(n,k) 问题。

**📊 数据集**

该研究为理论算法研究，未使用具体实验数据集，而是以理论上限和下界证明为主。

**📈 对比分析**

与此前 O(log^k n) 的上界相比，本算法在 k=2、3、4 的情况下将查询复杂度降低到 Θ(log^2 n)，在常数维度下实现了匹配下界；实验或实现层面的性能尚未给出。

**⚠️ 局限性**

局限性包括：①算法目前仅在查询复杂度上高效，时间复杂度不低效；② k>3 的高维推广仍依赖尚未证明的几何引理；③缺乏实际实现与实验验证。

---

## 37. Long-Horizon Geometry-Aware Navigation among Polytopes via MILP-MPC and Minkowski-Based CBFs

**arXiv ID:** 2604.00162 | [PDF](https://arxiv.org/pdf/2604.00162v1)

**作者:** Yi-Hsuan Chen `[一作]` (University of Maryland), Calin Belta `[通讯]` (University of Maryland)

**通讯引用:** 11899 | [OpenAlex ID](https://openalex.org/A5086742095)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种层次化的规划与控制框架 MILP‑MPC‑CBF，用于在非凸多边形环境中实现多面体机器人（多面体形状）安全导航。

**💡 创新点**

创新点在于：①将低速的 MILP‑MPC 规划与高速的 Minkowski‑CBF 安全过滤器分离，实现长周期预测与即时几何约束的协同；②对 Minkowski‑CBF 推导出闭式梯度与 Hessian，进一步扩展到高阶 HOCBF，用于双积分器动力学；③通过几何差分空间的距离优化实现对完整机器人几何的精确碰撞检测。

**🔧 技术方法**

采用的技术包括：混合整数线性规划 (MILP)、模型预测控制 (MPC)、控制障碍函数 (CBF) 与高阶 CBF (HOCBF)、Minkowski 差运算、可微优化与 KKT 敏感性分析、MATLAB/CPLEX 求解器。

**📊 数据集**

使用自行构造的二维迷宫和 U‑形陷阱仿真环境（无公开数据集），在 MATLAB 中对单积分器和双积分器动力学进行仿真。

**📈 对比分析**

与仅使用 MILP‑MPC 规划和纯粹反应式 CLF‑CBF‑QP 进行对比。结果显示：①在 U‑形陷阱中，CLF‑CBF‑QP 进入局部最小点停滞；②MILP‑MPC 仅规划时轨迹紧贴障碍；③MILP‑MPC‑CBF 在保持全几何安全的同时，成功绕过障碍并到达目标，控制输入平滑且符合实时性要求。

**⚠️ 局限性**

局限性：①高层规划使用点质量模型导致模型不匹配，若机器人几何与点模型差距过大，安全过滤器可能强行覆盖规划，产生几何诱导的局部最小；②当前仅适用于平移不变形的机器人，尚未处理姿态变化、非线性动力学；③在多约束交叉边界处 Hessian 可能不连续，需要进一步研究。

---

## 38. Benchmarking Interaction, Beyond Policy: a Reproducible Benchmark for Collaborative Instance Object Navigation

**arXiv ID:** 2604.00265 | [PDF](https://arxiv.org/pdf/2604.00265v1)

**作者:** Edoardo Zorzi `[一作]` (Sapienza University of Rome), Loris Bazzani `[通讯]` (University of Verona)

**通讯引用:** 4891 | [OpenAlex ID](https://openalex.org/A5064529297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了QAsk-NAV benchmark，分离导航与问答评估，并发布了28,000条高质量推理与问答轨迹；

**💡 创新点**

创新点在于将问答协议与导航协议解耦，实现轻量、可复现的评估，并通过大规模VLM生成高质量目标描述与多样化干扰器；

**🔧 技术方法**

采用多模态Transformer（基于VLFM的轻量VLM）结合大语言模型与文本引导图像编辑模型，训练统一的交互推理与问答模块；

**📊 数据集**

使用了CoIN-Bench、GOAT-Bench等公开3D场景，结合大VLM自动生成的目标描述与人工审核的推理轨迹，形成QAsk-NAV数据集；

**📈 对比分析**

与AIUTA等多模态模块化方法对比，QAsk-NAV在问答成功率、导航成功率上提升约30%（如Val Unseen 9.63%→12.20%），且参数量仅为现有方法的1/3、运行时提升70×；

**⚠️ 局限性**

局限在于仍依赖VLM与LLM的生成与模拟，可能无法完全覆盖真实人类交互细节，且在极端模糊描述下仍倾向直接执行，未能完全解决所有歧义情形。

---

## 39. Evolution Strategies for Deep RL pretraining

**arXiv ID:** 2604.00066 | [PDF](https://arxiv.org/pdf/2604.00066v1)

**作者:** Adrian Martínez `[一作]` (École Polytechnique Fédérale de Lausanne), Tamar Alphaidze `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对比了基于梯度的深度强化学习（DQN、PPO）与无梯度演化策略（ES）在不同难度环境中的表现，并评估了ES作为预训练手段提升DRL学习速度与鲁棒性的可行性。

**💡 创新点**

创新点在于系统地将ES与DRL在同一实验框架下对比，并探索ES预训练对DRL在不同任务（从简单的Flappy Bird到复杂的Atari Breakout与MuJoCo连续控制）效果的影响，从而揭示ES在任务复杂度与架构匹配上的局限。

**🔧 技术方法**

采用了Evolution Strategies（随机扰动梯度估计）、Deep Q‑Network（DQN）与Proximal Policy Optimization（PPO）等强化学习算法；实验中使用了基于Python的自定义Flappy Bird、Atari Breakout（图像与RAM输入）以及Brax实现的MuJoCo（HalfCheetah、Hopper、Walker2d）环境。

**📊 数据集**

使用的数据集包括：FlappyBirdEnv（自建环境）、Atari Breakout（ALE/Breakout‑v5 与 Breakout‑ram‑v4）、Brax MuJoCo 3D 连续控制环境（HalfCheetah、Hopper、Walker2d）。

**📈 对比分析**

通过对比最终奖励、累计训练时间、对随机种子与超参数的敏感度等指标，结果显示：在简单任务Flappy Bird中，ES可加速早期学习并能为DQN提供较好初始参数；在复杂任务Breakout与MuJoCo中，DQN/PPO在最终奖励与收敛速度上均优于ES，且ES预训练对PPO的训练速度与鲁棒性提升甚微，甚至无明显改善。

**⚠️ 局限性**

主要限制包括：1）ES与DRL在网络架构与优化动态上差异显著，导致预训练时参数迁移效果差；2）ES在高维输入（像素、RAM）与复杂动作空间中表现不佳，难以匹配梯度方法的表示学习能力；3）ES训练对环境的整体周期较长，导致收敛速度慢，无法满足更大规模或更高复杂度任务的需求。

---

## 40. One Panel Does Not Fit All: Case-Adaptive Multi-Agent Deliberation for Clinical Prediction

**arXiv ID:** 2604.00085 | [PDF](https://arxiv.org/pdf/2604.00085v1)

**作者:** Yuxing Lu `[一作]` (Georgia Institute of Technology), Jason Zhang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1977 | [OpenAlex ID](https://openalex.org/A5101875767)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了CAMP框架，用多智能体按病例动态组装专家面板并进行分层仲裁，解决临床预测中的案例异质性。

**💡 创新点**

创新点：案例自适应专家组装、三值投票(Keep/Refuse/Neutral)允许专家弃权、基于证据的分层仲裁路径。

**🔧 技术方法**

使用大型语言模型（Llama‑3.1‑70B、Gemma‑3‑27B、GPT‑OSS‑20B、GPT‑5.4）与多智能体推理，结合角色提示和混合路由。

**📊 数据集**

基于MIMIC‑IV（诊断预测与简短住院病程生成）和MIMIC‑IV‑Ext‑BHC数据集。

**📈 对比分析**

与单智能体、Chain‑of‑Thought、Self‑Consistency、Majority Voting、MedAgents、LLM‑as‑a‑Judge、Devil’s Advocate等方法比较，CAMP在四个模型上均居首位，宏F1和完美率提升显著，生成任务排名最佳，且token消耗相对低。

**⚠️ 局限性**

局限：仅在英语MIMIC‑IV单中心数据上验证，模型对提示敏感，需进一步临床评估，仲裁过程仍需要多次LLM调用，能源成本与实际临床延迟问题。

---

## 41. SYNTHONY: A Stress-Aware, Intent-Conditioned Agent for Deep Tabular Generative Models Selection

**arXiv ID:** 2604.00293 | [PDF](https://arxiv.org/pdf/2604.00293v1)

**作者:** Hochan Son `[一作]` (University of California, Los Angeles), Guang Cheng `[通讯]` (University of California, Los Angeles)

**通讯引用:** 2668 | [OpenAlex ID](https://openalex.org/A5043707940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了基于用户意图的表格数据生成器选择，提出了压力特征和SYNTHONY框架

**💡 创新点**

创新点是引入可解释的四维压力特征映射失效模式，并通过能力注册表实现自适应选择

**🔧 技术方法**

使用了统计压力特征提取、能力匹配、贝叶斯优化调参以及kNN等对比方法

**📊 数据集**

在7个OpenML表格数据集（Abalone、Bean、IndianLiverPatient、Obesity、faults、insurance、wilt）上进行评估

**📈 对比分析**

与随机、静态启发式、零射门LLM和kNN对比，SYNTHONY在Top‑3准确率约56%及Spearman 0.57，显著优于LLM和随机方法

**⚠️ 局限性**

局限性在于数据集规模有限、模型多样性不足、能力注册表依赖手工设计、隐私意图预测最难

---

## 42. How Do Language Models Process Ethical Instructions? Deliberation, Consistency, and Other-Recognition Across Four Models

**arXiv ID:** 2604.00021 | [PDF](https://arxiv.org/pdf/2604.00021v1)

**作者:** Hiroki Fukui `[一作]` (Kyoto University), Hiroki Fukui `[通讯]` (Kyoto University)

**通讯引用:** 7739 | [OpenAlex ID](https://openalex.org/A5102813354)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过600多次多代理仿真，比较了四款LLM在四种伦理指令格式和两种语言下的内部伦理处理方式，并提出三种新的处理度量。

**💡 创新点**

发现模型特异性的伦理处理模式并归纳为四类（输出过滤、保守重复、批判内化、原则一致性），提出Deliberation Depth、Value Consistency Across Dilemmas、Other-Recognition Index三种指标，证明安全、合规与伦理处理可分离。

**🔧 技术方法**

采用SociA多代理仿真框架，对Llama 3.3 70B、GPT‑4o mini、Qwen3‑Next‑80B‑A3B和Sonnet 4.5进行实验，利用关键词匹配构建DD、VCAD、ORI指标，并用贝叶斯和频率统计检验效应。

**📊 数据集**

使用自制的日语和英语对话脚本，共15–25次运行，累计超过600次仿真；所有数据、脚本与配置均托管于OSF（<osf.io/4n5uf>）和Zenodo。

**📈 对比分析**

通过比较DI与CPI指标，计算Hedges g、BF10等效应量，发现Llama日语复现高分离效应，GPT低DD表现为输出过滤，Qwen表现为批判内化，Sonnet表现为高层次原则一致性；不同模型对指令格式的敏感度显著差异。

**⚠️ 局限性**

仅测试四个模型且指标为粗糙的关键词匹配，缺乏验证；缺少对内部机制的因果推断；语言仅限日语和英语，无法保证跨语言和更大样本的普适性。

---

## 43. Gradient-Based Data Valuation Improves Curriculum Learning for Game-Theoretic Motion Planning

**arXiv ID:** 2604.00388 | [PDF](https://arxiv.org/pdf/2604.00388v1)

**作者:** Shihao Li `[一作]` (University of Texas at Austin), Dongmei Chen `[通讯]` (University of Texas at Austin)

**通讯引用:** 13157 | [OpenAlex ID](https://openalex.org/A5100677493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

在游戏理论运动规划中，研究者通过梯度相关的数据估值为训练数据构建课程学习顺序。

**💡 创新点**

创新点在于首次将TracIn梯度相似度评分用于游戏理论规划器，并证明梯度评分优于传统元数据难度指标。

**🔧 技术方法**

采用GameFormer模型、TracIn梯度评分、三阶段课程加权策略。

**📊 数据集**

使用nuPlan mini数据集（5,148个训练场景）。

**📈 对比分析**

与统一训练、元数据课程、基于损失的SPL以及混合课程进行对比，TracIn课程使平均ADE降至1.704m，显著优于元数据课程1.822m，方差也更低。

**⚠️ 局限性**

局限在于仅验证mini集，未覆盖完整nuPlan；只评估开环指标；训练种子有限；以及单检查点的TracIn评分可能不够精细。

---

## 44. Neural Collapse Dynamics: Depth, Activation, Regularisation, and Feature Norm Threshold

**arXiv ID:** 2604.00230 | [PDF](https://arxiv.org/pdf/2604.00230v1)

**作者:** Anamika Paul Rupa `[一作]` `[通讯]` (Howard University), Anamika Paul Rupa (Howard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过两阶段训练（交叉熵→均方误差）探究神经网络中“神经崩溃”(NC)出现的时机与动力学，并提出在每个（模型、数据集）组合中均值特征范数趋向一个高度聚集的阈值，能提前预测NC的开始。

**💡 创新点**

创新点在于发现特征范数阈值在每个（模型、数据集）对中几乎不随训练条件变化，能在平均提前62轮预测NC；同时揭示宽度、深度、激活函数和权重衰减对阈值与崩溃速度的非平凡交互，并将该阈值机制与“grokking”现象形成五重结构并行，提示了统一的范数阈值驱动机制。

**🔧 技术方法**

采用两阶段训练协议、NC1/NC2/NC3指标、均值特征范数监测、权重衰减金字塔实验、干预实验验证阈值为梯度流吸引子，并用统计方法（CV、ANOVA、95% 置信区间）评估结果稳定性与显著性。

**📊 数据集**

使用MNIST和CIFAR‑10两个公开数据集，分别在MLP和ResNet‑20架构下进行实验。

**📈 对比分析**

在多种模型（MLP、ResNet‑20）与深度、宽度、激活、权重衰减组合下对比NC1、特征范数与崩溃时间；发现ResNet‑20/MNIST的阈值5.867远高于MLP‑5的1.052；阈值提前预测平均62轮，宽度可使崩溃时间缩短约33%，深度表现为非单调最优值，权重衰减呈金字塔三相位（过小慢、最佳快、过大阻止崩溃）。

**⚠️ 局限性**

局限性包括仅在MNIST/CIFAR‑10两个简单数据集上验证；实验样本量（种子数）有限；两阶段训练协议是必要前提，可能影响阈值的绝对值；对更大范围激活函数、网络架构以及更复杂任务的泛化性仍待进一步检验。

---

## 45. Can LLMs Perceive Time? An Empirical Investigation

**arXiv ID:** 2604.00010 | [PDF](https://arxiv.org/pdf/2604.00010v1)

**作者:** Aniketh Garikaparthi `[一作]` `[通讯]` (TCS Research), Aniketh Garikaparthi (TCS Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型（LLM）对自身任务耗时的自估能力，构建四类实验验证其误差。

**💡 创新点**

首次系统量化LLM在前置估计、相对排序、后置回忆和多步代理任务中的时间估计缺陷，并揭示其对复杂度标签的错误依赖。

**🔧 技术方法**

采用前置估计、相对排序、后置回忆和多步代理实验，使用GPT‑5、GPT‑4o、OLMo3‑7B、Qwen3‑8B等模型。

**📊 数据集**

共68个跨七类（代码生成、调试、摘要、推理、写作、创意、问答）的任务样本，记录真实执行时长。

**📈 对比分析**

通过与实际墙钟时长比较，前置估计高4–7×，相关系数仅0.35–0.55；后置回忆误差在5–10×，相对排序几乎随机，显示出显著的性能不足。

**⚠️ 局限性**

缺乏对自身推理耗时的感知，导致前置估计偏高、后置回忆偏离，且在多步代理中误差累积，表明当前LLM架构无法实现可靠的时间自我调度。

---

## 46. Autonomous Adaptive Solver Selection for Chemistry Integration via Reinforcement Learning

**arXiv ID:** 2604.00264 | [PDF](https://arxiv.org/pdf/2604.00264v1)

**作者:** Eloghosa Ikponmwoba `[一作]` (Louisiana State University), Opeoluwa Owoyele `[通讯]` (Louisiana State University)

**通讯引用:** 585 | [OpenAlex ID](https://openalex.org/A5037919862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于强化学习的自适应求解器选择框架，能够在化学反应积分过程中在 CVODE（隐式BDF）和 QSS（准稳态）两种求解器之间智能切换。

**💡 创新点**

将求解器选择建模为约束强化学习的马尔可夫决策过程，采用 Lagrangian 奖励在保证误差容限的前提下平衡计算成本，学习得到可迁移、面向全局轨迹的策略，突破了传统手工阈值或监督预测的局限。

**🔧 技术方法**

使用 Proximal Policy Optimization（PPO）强化学习、Lagrangian 约束优化、状态特征构造（温度、关键物种浓度、时变梯度）、CVODE 与 α‑QSS 求解器，动作空间为二元选择。

**📊 数据集**

训练数据来自 0D 恒压混合燃烧实验，随机采样初始温度（300–1200 K）、压强（1–60 atm）和混合比例（10⁻⁶–10⁻¹），采用 n‑dodecane 106 种机制；1D 逆流扩散火焰（不同拉伸率 10–2000 s⁻¹）用于零样本迁移评估。

**📈 对比分析**

与仅使用 CVODE 或仅使用 QSS 的基线进行对比，评价指标包括计算速度提升、点火延迟误差和温度误差；在 0D 任务上平均速度提升约 3 倍（范围 1.11×–10.58×），点火延迟误差低于 2.6%；在 1D 任务上平均速度提升约 2.2 倍，点火延迟误差低于 2.5%，推理开销不到 1%。

**⚠️ 局限性**

仅训练单一燃料机制，动作空间仅限两种求解器，未覆盖多组分或多燃料；在大规模 2D/3D 湍流或动态网格下的并行推理与负载平衡未验证；未来需扩展到多燃料、可变容差求解器以及更复杂的求解器组合。

---

## 47. LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias

**arXiv ID:** 2604.00259 | [PDF](https://arxiv.org/pdf/2604.00259v1)

**作者:** Filip J. Kucia `[一作]` (Warsaw University of Technology), Anna Wróblewska `[通讯]` (Warsaw University of Technology)

**通讯引用:** 626 | [OpenAlex ID](https://openalex.org/A5031984813)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估指令化大语言模型（LLM）在多种写作评分任务中的表现，分析其与人工评分的一致性、系统偏差以及提示方式对结果的影响。

**💡 创新点**

①对LLM在整体评分与多属性评分两种评价模式下的差异进行细致对比；②发现LOC（低阶）特征（如语法、标点）存在稳定负偏差；③提出基于小样本偏差校正的部署策略。

**🔧 技术方法**

使用Meta Llama-3.1系列与GPT‑OSS系列的开放权重、零样本（zero‑shot）评估，结合关键词式与完整准则式两种提示模板。

**📊 数据集**

ASAP 2.0（整体评分），ELLIPSE（细粒度分析评分），DREsS（多属性分析评分）。

**📈 对比分析**

通过Quadratic Weighted Kappa、Exact Agreement、偏差、MAE等指标评估；在整体评分中最佳模型（Llama‑3.1‑70B）QWK≈0.60，偏差接近零；在多属性评分中，LOC特征表现差且负偏差明显，整体一致性低。

**⚠️ 局限性**

仅评估开放权重LLM的零样本设置，未考虑多评审者差异和学习者背景；提示方式仅限两种，未探索更细粒度或交互式提示；未检验闭源大模型或少样本微调的效果。

---

## 48. Disentangling Prompt Element Level Risk Factors for Hallucinations and Omissions in Mental Health LLM Responses

**arXiv ID:** 2604.00014 | [PDF](https://arxiv.org/pdf/2604.00014v1)

**作者:** Congning Ni `[一作]` (Vanderbilt University Medical Center), Zhijun Yin `[通讯]` (Vanderbilt University Medical Center)

**通讯引用:** 2592 | [OpenAlex ID](https://openalex.org/A5079247989)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实施了UTCO框架，用结构化的用户、主题、情境、语调四要素生成2075个高真实度心理健康求助提示，并对Llama 3.3生成的回复进行人工标注，测定幻觉率6.5%、遗漏率13.2%。

**💡 创新点**

创新点在于通过可控的UTCO元素系统化评估LLM在高压求助情境中的安全风险，聚焦遗漏这一被低估的失败模式，并揭示情境与语调对幻觉/遗漏的决定性影响。

**🔧 技术方法**

采用梯度提升树与SHAP进行特征重要性分析、倾向分数匹配评估敏感性、相似度匹配机制分析，并使用GPT‑4o作为结构化判定者。

**📊 数据集**

数据集为UTCO生成的2075条心理健康提示，涵盖9类用户背景、10个临床主题、自然与人工情境、12种情绪语调。

**📈 对比分析**

通过与主题、情境、语调保持一致的匹配实验，发现长文本与自然来源情境以及高情绪压力显著提升遗漏风险；模型在危机与自杀主题下遗漏率最高，说明评估方法能有效定位高风险场景。

**⚠️ 局限性**

局限包括仅评估单一Llama 3.3模型、UTCO提示可能不足以覆盖真实网络求助的多样性、相似度匹配难度高导致大部分失败缺乏同类对照、对逻辑矛盾的测试不充分。

---

## 49. MAC-Attention: a Match-Amend-Complete Scheme for Fast and Accurate Attention Computation

**arXiv ID:** 2604.00235 | [PDF](https://arxiv.org/pdf/2604.00235v1)

**作者:** Jinghan Yao `[一作]` (Ohio State University), Dhabaleswar K Panda `[通讯]` (Ohio State University)

**通讯引用:** 12629 | [OpenAlex ID](https://openalex.org/A5024879682)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MAC‑Attention，一种无训练、模型无关的 Match–Amend–Complete 机制，通过在预 RoPE 空间匹配查询、局部修正并在对数域合并来重用先前的注意力计算，从而实现长上下文解码的 IO‑bottleneck 减少。

**💡 创新点**

创新点在于：① 在预 RoPE 空间使用 L2 阈值匹配以提升匹配率；② 对重用边界附近的高权重区块进行局部修正；③ 使用数值稳定的对数域合并实现常数时间的注意力合并；④ 通过小型环形缓存实现 O(1) 计算和内存访问，完全保持对完整 KV 的访问。

**🔧 技术方法**

技术手段包括：预 RoPE 匹配、L2 距离阈值、局部修正带、对数域合并、短周期环形缓存、负载均衡调度、IO‑aware 内核与分页 KV 兼容。

**📊 数据集**

使用的基准数据集为 LongBench v2、RULER 和 LongGenBench，覆盖从 120K 句子到 16K 连续生成的长上下文任务。

**📈 对比分析**

与 FlashInfer、Quest、RocketKV、Multipole 等现有方法对比，MAC‑Attention 在 128K 长度下 KV 访问可降低 99%，每个 token 的延迟降低 60% 以上，注意力阶段加速超过 14.3×（最大 46×），整体解码加速可达 2.6×，且保持与全注意力相同的准确率。

**⚠️ 局限性**

局限性包括：需要较高的匹配命中率，匹配窗口固定可能限制极端长距离的重用；匹配失败时仍需全线性注意力，导致偶尔性能回退；在查询模式高度多样化或极短上下文的任务中，收益可能不明显。

---

## 50. SAGE: Subsurface AI-driven Geostatistical Extraction with proxy posterior

**arXiv ID:** 2604.00307 | [PDF](https://arxiv.org/pdf/2604.00307v1)

**作者:** Huseyin Tuna Erdinc `[一作]` (Georgia Institute of Technology), Felix J. Herrmann `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7666 | [OpenAlex ID](https://openalex.org/A5010780250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了SAGE框架，利用稀疏井日志和迁移成像图像在不需要完整速度模型训练的情况下生成全分辨率、地质合理的速度场。

**💡 创新点**

创新点在于：①学习代理后验（proxy posterior）从部分观测中估计完整速度模型；②通过子掩码约束避免仅重建已观测点的退化解；③在后验采样阶段只使用迁移图像即可生成速度样本，提升数据效率。

**🔧 技术方法**

采用条件分数扩散网络（score-based diffusion）与U‑Net去噪器，配合RTM图像、井日志掩码进行自监督训练；随后通过逆扩散采样得到速度后验样本。

**📊 数据集**

数据集包括：1) Compass 3D合成模型（1000个256×512 2D速度切片）用于训练与评估；2) 英国北海国家数据仓库的真实井日志与迁移成像，用于实测验证和微调。

**📈 对比分析**

评估指标：后验均值与真实速度的SSIM 0.82；在下游WIS E逆演算网络中，用SAGE生成的样本训练得到的SSIM仅从0.88降至0.84，表明性能接近真实模型；与传统全波形逆演算方法相比，SAGE在数据缺失情形下显著提高速度模型质量。

**⚠️ 局限性**

主要限制：①在井日志极为稀疏时，生成模型细节仍趋于平滑；②当前实现仅在二维平面，三维推广仍待研究；③对真实数据的依赖仍受井数与地震采集质量限制。

---

## 51. Finding and Reactivating Post-Trained LLMs' Hidden Safety Mechanisms

**arXiv ID:** 2604.00012 | [PDF](https://arxiv.org/pdf/2604.00012v1)

**作者:** Mingjie Li `[一作]` (CISPA Helmholtz Center for Information Security), Yisen Wang `[通讯]` (Peking University)

**通讯引用:** 5489 | [OpenAlex ID](https://openalex.org/A5101431030)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对后训练的大型语言模型（尤其是大推理模型）在安全性上的下降进行研究，并提出一种轻量级的安全恢复方法SafeReAct；

**💡 创新点**

创新点在于发现安全机制在后训练中并未被移除，而是被激活的推理或域特定能力所掩盖；通过对推理相关神经元进行剪枝可恢复安全行为；SafeReAct利用LoRA微调仅对少量层的内部表示进行对齐，将模型在有害输入上的表示拉近到安全版模型的表示，从而在不显著降低任务性能的前提下恢复安全性；

**🔧 技术方法**

主要技术包括：神经元重要性评估（Wanda分数）进行领域特定能力的剪枝；内部表示对齐（与安全版模型的隐藏安全表示对齐）+保持损失；LoRA微调；安全评估采用Llama-3-Guard判别器；推理评估采用GSM8K、MATH-500；医学评估采用MedQA；金融评估采用Finance-llama3-1-8B-instruct；

**📊 数据集**

使用的数据集包括：推理任务的S1K、LIMO；安全评估数据集JailbreakBench、AdvBench、XsTest；医学领域的UltraMedical前20k样本；金融领域的公开金融LLaMA-3.1-Instruct数据；

**📈 对比分析**

与两种基线方法（Circuit-Breaker和SafeChain）进行对比；SafeReAct在四种推理模型（R1-7B、R1-8B、OT-7B、R1-14B）上将有害率降至≈0%，且推理准确率下降不足3%；在医学模型Llama-3-UltraMedical上将JailbreakBench有害率从66%降至6%，并保持MEDQA性能；在金融模型上将有害率从47%降至3%；相较于基线方法，SafeReAct在安全性提升上更稳定、任务性能损失更小；

**⚠️ 局限性**

局限性：仅在7B、8B、14B、32B规模模型上评估；未验证更大规模模型的效果；仅评估推理、医学和金融领域，未涵盖其他专业领域；方法需要先获得安全版模型（剪枝后）作为对齐目标，可能在某些模型中剪枝不易实现。

---

## 52. Set-Based Value Function Characterization and Neural Approximation of Stabilization Domains for Input-Constrained Discrete-Time Systems

**arXiv ID:** 2604.00305 | [PDF](https://arxiv.org/pdf/2604.00305v1)

**作者:** Mohamed Serry `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**通讯引用:** 10325 | [OpenAlex ID](https://openalex.org/A5100361707)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于集合值价值函数的物理信息神经网络框架，用来估计受输入约束的离散时间非线性系统的稳定域（DOS）并从中合成稳控策略。

**💡 创新点**

创新点在于：①定义了作用于紧致集合空间的价值函数来精确刻画DOS；②推导了对应的Bellman‑类型（Zubov型）函数方程；③将该方程嵌入物理信息学习中，利用集合表示的可嵌入特性实现神经网络对DOS价值函数的学习，从而得到更不保守的稳定域估计。

**🔧 技术方法**

使用技术包括：集合值价值函数与Hausdorff距离、可达集近似（轨迹采样法）、物理信息损失（Bellman方程残差）与数据驱动损失的联合训练、以及MATLAB+CORA工具箱进行可达集仿真与验证。

**📊 数据集**

数据集为系统自身产生的仿真轨迹；通过随机输入轨迹采样得到可达集近似，随后用于构造损失函数；并在两道典型的二维、三维离散系统上进行实验。

**📈 对比分析**

与传统基于二次Lyapunov函数的椭圆收敛域估计相比，所学习的神经网络估计的DOS更大、保守性更低（ω₂≈0.97，接近理论DOS），并在闭环控制实验中成功将多起始状态引导至原点，满足输入约束。

**⚠️ 局限性**

局限性包括：①需要对集合空间做可嵌入假设（如多面体、zonotope等）才能实现有限维表示；②可达集近似依赖采样数，计算成本仍不低；③目前尚未实现形式化的安全验证，所得到的DOS及控制器仍缺乏严格的可验证保证；④对状态约束或外部干扰的处理尚未展开。

---

## 53. Unified Architecture Metamodel of Information Systems Developed by Generative AI

**arXiv ID:** 2604.00171 | [PDF](https://arxiv.org/pdf/2604.00171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 54. "Who Am I, and Who Else Is Here?" Behavioral Differentiation Without Role Assignment in Multi-Agent LLM Systems

**arXiv ID:** 2604.00026 | [PDF](https://arxiv.org/pdf/2604.00026v1)

**作者:** Houssam EL Kandoussi `[一作]` `[通讯]` (Independent Researcher), Houssam EL Kandoussi (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在统一推理后端上，使用War Room平台让7个不同架构的LLM在相同提示下进行多轮对话，并系统记录并编码13,786条信息，探究它们在群体中的角色差异、补偿行为、命名偏差与提示设计的影响。

**💡 创新点**

首次通过可复现的实验平台和双判别LLM评审，验证异构LLM群体在最小化提示下可自发产生显著行为差异，并揭示命名与提示对行为一致性的调节作用。

**🔧 技术方法**

利用Groq统一推理服务、War Room调度工具、Gemini与Claude Sonnet双判别编码、Cohen’s κ统计和人类验证，完成行为标签化与差异分析。

**📊 数据集**

采用13,786条对话消息（208次实验，12个实验系列），以及两条虚构项目简报（食品配送App和文化节）作为实验素材。

**📈 对比分析**

通过cosine相似度、Kruskal‑Wallis检验及Bonferroni校正，发现异构组行为相似度为0.56，均质组为0.85，命名真实模型名时相似度升至0.77，提示移除后相似度接近均质组，补偿行为层级明显出现；这些指标均显示出显著差异且效应量大。

**⚠️ 局限性**

实验受限于当前模型版本、部分模型API不稳定、未评估任务执行质量、样本仅包含7个模型家族且只涉及两种任务领域，导致结果难以直接推广到其他模型或应用场景。

---

## 55. Transformers for Program Termination

**arXiv ID:** 2604.00039 | [PDF](https://arxiv.org/pdf/2604.00039v1)

**作者:** Yoav Alon `[一作]` (University of Bristol), Cristina David `[通讯]` (University of Bristol)

**通讯引用:** 808 | [OpenAlex ID](https://openalex.org/A5072292925)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过微调轻量级预训练 Transformer 并构建多模型异构集成，对程序终止性进行预测，并为预测结果提供基于 AST 的可解释性说明。

**💡 创新点**

创新点在于：①利用失衡感知损失函数与类别感知采样实现少数类（非终止程序）检测提升；②将不同损失优化的 Transformer 组成集成，证明多样性优于单模型或规模扩大；③提出 Token‑to‑AST 归因管线，将 Shapley 重要性映射到语法结构，提升解释性。

**🔧 技术方法**

核心技术包括：轻量化 Transformer（Albert、DistilBERT、BERT、Perceiver、Bart、T5‑small）；失衡感知损失（BCE‑effnum、Focal、LDAM）；类别感知采样；软投票集成；基于 SHAP 的 Token 级归因与 AST 对齐。

**📊 数据集**

使用公开终止性基准 TerGEC（Python 代码）以及 HumanEval、MBPP（Python）和 TPDB（C）三大数据集进行评估。

**📈 对比分析**

实验将集成模型与图神经网络（TerGEC、GCN、GAT 等）以及多款现成 LLM（GPT‑4/5、Gemini、Llama、Claude 等）进行对比，指标为 mAP 与 AUC。集成模型在所有基准上均超越对手，尤其在 mAP（少数类召回）上提升 15‑27%，并在 AUC 上保持或提升 1‑2%。

**⚠️ 局限性**

局限性包括：①评估主要基于自动生成或时间限制的合成数据，难以直接证明在工业真实代码上的泛化；②仅实验了少数轻量化 Transformer，未探究更大模型或其他架构的潜力；③评价指标仅限于 AUC 与 mAP，未考虑误报/漏报成本或其他公平性度量。

---

## 56. Deep Networks Favor Simple Data

**arXiv ID:** 2604.00394 | [PDF](https://arxiv.org/pdf/2604.00394v1)

**作者:** Weyl Lu `[一作]` (University of California Davis), Yubei Chen `[通讯]` (University of California Davis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究深度网络在不同架构与密度估计器下对样本的密度排名，并发现无论是自回归、流模型、扩散模型还是自监督表征学习，训练好的网络始终将“简单”图像排在高密度位置，呈现普遍的“简单性偏好”。

**💡 创新点**

创新点在于：①将训练好的网络与其基于表示或输出的密度估计器拆分为两个独立对象；②统一引入两类估计器（Jacobian 体积估计和自回归密度估计），使得多种原本不可比的模型能够在同一框架下比较；③在多种模型、数据集以及极端训练限制（仅低密度10%或单样本）下，系统性验证了“简单性偏好”的普遍性。

**🔧 技术方法**

主要技术包括：Jacobian 体积估计（基于奇异值）、自回归密度估计、流模型（Glow）、得分扩散模型、像素级自监督（DINOv2、I‑JEA）、JPEG 与梯度总变差的复杂度代理、Spearman/Kendall 相关性分析、噪声扰动实验验证 OOD 效果的脆弱性。

**📊 数据集**

使用的数据集：CIFAR‑10（包括低密度子集与单样本子集）、SVHN（OOD 对比）、ImageNet‑1K（64×64 分辨率）以及各类生成/自监督模型的预训练权重。

**📈 对比分析**

比较方法：对每个模型-估计器组合在测试集上计算密度评分，按从高到低排序得到排名；随后与其他模型的排名以及外部复杂度度量（JPEG 长度、梯度总变差）通过 Spearman 相关性进行比较。实验结果显示：大多数模型间的相关系数在 0.6–0.9 之间，低密度子集或单样本训练后排名仍保持高度一致；但 PixelCNN++ 在单样本训练时相关系数急剧下降，说明此类模型对局部细节更敏感。

**⚠️ 局限性**

局限性：①未给出完整理论解释为何深度网络会偏好简单数据；②在 DINOv2 对 CIFAR‑10 的低相关性可能源于分辨率/预训练差异，未能完全验证其普适性；③外部复杂度代理仅为近似，不能完全反映所有维度的“复杂度”；④实验主要聚焦图像任务，尚未验证在其他模态（文本、音频）或更高分辨率数据上的一致性。

---

## 57. MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control

**arXiv ID:** 2604.00292 | [PDF](https://arxiv.org/pdf/2604.00292v1)

**作者:** Sahil Kumar `[一作]` (Yeshiva University), Youshan Zhang `[通讯]` (Chuzhou University)

**通讯引用:** 1086 | [OpenAlex ID](https://openalex.org/A5079460371)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了MambaVoiceCloning（MVC），一种完全基于状态空间模型（SSM）且在推理时不使用注意力或循环网络的扩散式语音合成框架。

**💡 创新点**

创新点在于将文本、节奏与情感编码完全迁移到SSM，配合门控双向Mamba融合与AdaLN调制，实现线性时间推理、显著降低内存占用并提升长文本与跨语言的稳定性。

**🔧 技术方法**

技术手段包括：门控双向Mamba文本编码器、时序双向Mamba对齐器、Expressive Mamba情感调制器、训练时使用的轻量级注意力对齐教师、AdaLN风格调制、以及固定的StyleTTS2扩散解码器与vocoder。

**📊 数据集**

使用的公开数据集为：LJSpeech（24h），LibriTTS（245h）训练；VCTK（109人）零样本测试；CSS10（西班牙语、德语、法语）跨语言测试；以及长度为2–6分钟的古腾堡文本用于长文本评估。

**📈 对比分析**

与StyleTTS2、VITS、JETS及两种Mamba混合基线在完全匹配的前端、解码器与vocoder配置下对比，MVC在MOS、CMOS、F0 RMSE、MCD、WER等指标上取得显著或可统计显著的提升，编码器参数仅21M，吞吐率提高约1.6倍。

**⚠️ 局限性**

局限性包括：扩散解码器仍为主要延迟瓶颈、情感控制仅通过全局AdaLN实现、模型仅在英语数据上训练、对极端长文本或非英语语音的进一步鲁棒性待验证。

---

## 58. RAGShield: Provenance-Verified Defense-in-Depth Against Knowledge Base Poisoning in Government Retrieval-Augmented Generation Systems

**arXiv ID:** 2604.00387 | [PDF](https://arxiv.org/pdf/2604.00387v1)

**作者:** KrishnaSaiReddy Patil `[一作]` `[通讯]`, KrishnaSaiReddy Patil

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套名为RAGShield的五层防御框架，针对政府检索增强生成（RAG）系统的知识库中毒攻击，提供从文档来源验证到生成输出可审计的全链路防护。

**💡 创新点**

创新点包括：①首次将软件供应链的来源追溯和数字签名理念引入RAG知识库，形成C2PA风格的文档 attestations；②设计了正式的污点 lattice 与跨源矛盾检测，能够在来源合法时仍捕获内部威胁；③为RAG流水线提供完整的 NIST SP 800‑53 合规映射；④在适应性攻击（T5）下实现 0% 触发率，且保持 0% 的误报。

**🔧 技术方法**

使用的技术包括：C2PA‑inspired Ed25519 证书与 SHA‑256 内容哈希、内容标准化（去除隐藏文本）、信任加权检索、基于向量相似度的矛盾检测与共证检查、正式污点 lattice 传播规则、以及基于 JSON 的审计日志与 NIST 控制映射。

**📊 数据集**

评估数据集为 500 条来自 Natural Questions (NQ) 的真实 Wikipedia 片段，注入 63 条攻击文档（涵盖 T1–T5 五个攻击层级），并用 200 条 NQ 问题进行查询。

**📈 对比分析**

与 RobustRAG、RAGDefender、无防御基线对比，RAGShield 在所有五个攻击层级的攻击成功率 (ASR) 均为 0%（95% 置信区间 [0.0%, 1.9%]），误报率为 0%。相比之下，RobustRAG 仅在 T1/T2/T4 层级能达到 0–0.5% 的 ASR，RAGDefender 的 ASR 在 7.5–12.5% 之间；无防御时 ASR 可达 8–17.5%。

**⚠️ 局限性**

局限性包括：①评估规模仅为 500 条文档，无法覆盖生产级千万级知识库；②对已存在的基线文档中毒（>5%）时性能下降；③无法检测“原地替换”攻击（T6）导致 17.5% 的 ASR；④依赖于文档签名基础设施，当前仅提供分层启动路径；⑤模拟实现的 RobustRAG 与 RAGDefender 可能与真实系统略有差异。

---

## 59. Stable algorithms cannot reliably find isolated perceptron solutions

**arXiv ID:** 2604.00328 | [PDF](https://arxiv.org/pdf/2604.00328v1)

**作者:** Shuyang Gong `[一作]` (Peking University), Mark Sellke `[通讯]` (Harvard University)

**通讯引用:** 565 | [OpenAlex ID](https://openalex.org/A5012362703)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究二进制感知机，证明在任何正约束密度下，几乎所有解都是强隔离的，并证明稳定算法无法以高概率找到这些隔离解。

**💡 创新点**

首次利用Pitt相关性不等式和低阶多项式稳定性框架，避免传统的重叠间隙（OGP）方法，给出对二进制感知机求解的算法难度上限。

**🔧 技术方法**

Pitt相关性不等式、低阶多项式稳定性分析、随机扰动与协方差分解、组合平衡引理。

**📊 数据集**

无实验数据，纯理论分析。

**📈 对比分析**

未做实验比较，理论上证明任何稳定算法找到隔离解的成功率≤0.84233，且若成功率≈1则几乎不可能找到隔离解。

**⚠️ 局限性**

结果仅适用于稳定算法，未能进一步证明成功概率可以降到o(1)；对球面感知机的适用性有限；高阶多项式/非稳定算法可能仍有突破。

---

## 60. Two-Stage Optimizer-Aware Online Data Selection for Large Language Models

**arXiv ID:** 2604.00001 | [PDF](https://arxiv.org/pdf/2604.00001v1)

**作者:** Fangxin Wang `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 135315 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在线梯度基数据选择与重权重框架，针对大语言模型微调

**💡 创新点**

将样本选择视为优化器感知的更新匹配，采用非加性子集效用和非负权重，并分两阶段过滤-加权实现

**🔧 技术方法**

使用 LoRA+随机投影的梯度压缩、外积梯度分解、两阶段贪婪+NNLS优化以及线性化 Adam 预处理

**📊 数据集**

在 Open‑Instruct 训练集上微调 Llama‑3.2‑1B 与 Qwen3‑0.6B，并在 MMLU 与 TyDiQA 上进行评估

**📈 对比分析**

与全数据、GRAD‑MATCH、TracIn、LESS、GREATS 等基线对比，固定 5% 数据预算下在 MMLU 与 TyDiQA 上均取得更高准确率/ F1，尤其在后者上表现最优

**⚠️ 局限性**

局限性包括对 Adam 线性化假设的依赖、随机投影维度需调参、某些配置下全数据仍可竞争，以及对额外 LoRA 与投影实现的依赖

---

## 61. Benchmark for Assessing Olfactory Perception of Large Language Models

**arXiv ID:** 2604.00002 | [PDF](https://arxiv.org/pdf/2604.00002v1)

**作者:** Eftychia Makri `[一作]` (Yale University), Nicholas A. Christakis `[通讯]` (Yale University)

**通讯引用:** 52325 | [OpenAlex ID](https://openalex.org/A5010495671)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OLP（Olfactory Perception）基准，用以评估大型语言模型在从分子信息推断气味的推理能力；

**💡 创新点**

首次构建完整、结构化的多任务气味认知基准，并通过双重提示（化合物名称 vs. 异构体SMILES）揭示模型对分子结构的实际理解程度；

**🔧 技术方法**

使用标准提示模板对21种主流LLM进行推理实验，结合多任务评估指标（准确率、F1、Pearson相关系数），并对多语言版本进行投票集成；

**📊 数据集**

基于多来源气味数据集，包括IFRA香料词典、Good Scents、M2OR受体激活实验、混合物相似性等，形成1010道多类别问题；

**📈 对比分析**

与多家供应商的21个模型配置（包括GPT‑5、Claude、Gemini、Llama、DeepSeek等）做横向对比，最佳模型Claude Opus 4.6 (max)在名称提示下整体准确率达64.4%；SMILES提示平均低约7个百分点，说明模型主要通过词汇关联获取气味知识；

**⚠️ 局限性**

局限性在于任务采用离散固定答案、缺乏浓度/时间维度、文化与个体差异未充分覆盖、无多模态输入，且结果仅衡量与既定标签的一致性，不能证明模型真正理解结构-气味机制。

---

## 62. Mine-JEPA: In-Domain Self-Supervised Learning for Mine-Like Object Classification in Side-Scan Sonar

**arXiv ID:** 2604.00383 | [PDF](https://arxiv.org/pdf/2604.00383v1)

**作者:** Taeyoun Kwon `[一作]` (Maum AI Inc.), Moon Hwan Kim `[通讯]` (Maum AI Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 Mine-JEPA，针对侧扫声纳（SSS）中的雷达目标分类构建了一套基于领域自监督学习的完整训练管线。

**💡 创新点**

创新点在于首次针对极端数据稀缺的 SSS 领域采用 SIGReg 正则化自监督损失，配合领域特定的数据增强与轻量级 ViT 结构，证明在小样本环境下可匹配甚至优于大规模视觉基础模型，并揭示强大基础模型在领域自监督中的表现可能退化。

**🔧 技术方法**

核心技术包括 SIGReg 正则化自监督损失、Vision Transformer 背骨、针对 SSS 的自定义数据增强（仅亮度/对比度、垂直翻转/旋转等）、ImageNet‑1K 预训练初始化以及真实与合成声纳图像混合训练。

**📊 数据集**

使用公开的侧扫声纳数据集（1,170 张图像，668 个标注对象）作为真实数据，并补充约 256K 张合成声纳图像构成 Real+Syn 训练集。

**📈 对比分析**

在 3 类与二分类任务上与 DINOv3、SimCLR、VICReg 等方法进行对比；Mine‑JEPA 在 3 类宏 F1 0.820（相较 DINOv3 的 0.810 提升 3.5%）以及二分类 F1 0.935，且使用 ViT‑Tiny 参数仅为 DINOv3 的 1/4，性能更优。

**⚠️ 局限性**

局限性在于仅基于单一公开数据集且测试集规模有限，未覆盖多种雷达类型、声纳设备或不同操作环境，合成数据的通用性及跨域迁移效果仍需进一步验证。

---

## 63. LinearARD: Linear-Memory Attention Distillation for RoPE Restoration

**arXiv ID:** 2604.00004 | [PDF](https://arxiv.org/pdf/2604.00004v1)

**作者:** Ning Yang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jun Wang `[通讯]` (University College London)

**通讯引用:** 37475 | [OpenAlex ID](https://openalex.org/A5084169778)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过自蒸馏的方式，在保持RoPE扩展后的长上下文窗口的同时，恢复原始RoPE模型在短文本任务上的性能。

**💡 创新点**

① 采用行向量KL对齐 Q/Q、K/K、V/V 的自关系分布，而非传统隐藏状态匹配；② 设计 IO‑aware 线性内存 KL 蒸馏核，突破 n² 的显存瓶颈，实现在极长序列上的精确稠密蒸馏。

**🔧 技术方法**

RoPE 频率缩放（Position Interpolation）、自蒸馏（LinearARD）、线性内存 KL 蒸馏核、LoRA/QLoRA 参数高效微调、标准 AdamW 优化。

**📊 数据集**

训练使用 PG19 等长文本；评估使用 MMLU、LAMBADA、MathQA、BoolQ、OpenBookQA、PIQA、SIQA、ARC‑Challenge 以及 RULER 长文本基准。

**📈 对比分析**

与无恢复、CPT、LongReD 等基线比较，LinearARD 在仅使用 4.25M 训练 token 的情况下恢复 94.8%/94.2%/95.0% 的短文本平均准确率，RULER 分别为 63.2/68.3/60.8，性能优于 CPT/LongReD，且 token 使用量约为它们的 1/60。

**⚠️ 局限性**

仍需预先冻结教师模型；方法专注于 RoPE 扩展，对其他长上下文技术兼容性有限；极端长序列（>32K）下的泛化仍待验证。

---

## 64. Not Just Duolingo: Supporting Immigrant Language Preservation Through Family-Based Play

**arXiv ID:** 2604.00282 | [PDF](https://arxiv.org/pdf/2604.00282v1)

**作者:** Alejandro Ciuba `[一作]` (University of Pittsburgh), Aakash Gautam `[通讯]` (University of Pittsburgh)

**通讯引用:** 416 | [OpenAlex ID](https://openalex.org/A5015895601)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

采访8名尼泊尔移民，梳理他们对语言保存的需求与挑战，并基于此设计并实现了一个以语音为主、点选式的亲子互动语言学习游戏原型。

**💡 创新点**

创新点在于：① 将语言学习视为亲缘关系的建构工具，强调共处与协同；② 采用全音频化、低门槛的点选交互，降低技术与文字障碍；③ 通过游戏情境模拟日常情境，增强沉浸式可理解输入。

**🔧 技术方法**

技术实现主要使用Godot游戏引擎，配合语音合成/录制、点选交互逻辑和可理解输入理论设计；界面采用光标高亮与径向菜单展示可行动作。

**📊 数据集**

目前未使用大规模语料库；原型仅在西班牙语与中文环境中测试，用以验证交互逻辑；未来计划采集尼泊尔本土语言的音频素材并与社区合作填充内容。

**📈 对比分析**

通过4位设计与游戏专家的实验室评测（30–60分钟），使用EOTA方法和主题分析收集反馈。评测显示玩家能顺利完成任务，但UI符号识别与动作面板使用存在困惑，需要简化与改进。没有客观性能指标，仅以可用性和趣味性为评估维度。

**⚠️ 局限性**

局限性包括：样本仅为成年人，缺少儿童视角；原型功能单一、未加入真实尼泊尔语料；评测对象为专家而非目标用户；研究仅聚焦尼泊尔移民，缺乏跨文化推广性；并未解决制度性支持缺失等结构性问题。

---

## 65. Efficient Software Vulnerability Detection Using Transformer-based Models

**arXiv ID:** 2604.00112 | [PDF](https://arxiv.org/pdf/2604.00112v1)

**作者:** Sameer Shaik `[一作]` (DePaul University), Jacob Furst `[通讯]` (DePaul University)

**通讯引用:** 3171 | [OpenAlex ID](https://openalex.org/A5001027068)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用Transformer模型（DistilBERT、CodeBERT、Gemma-2B）对C/C++程序切片进行漏洞检测，并设计两种数据平衡策略以提升检测效果；

**💡 创新点**

①将程序切片与Transformer相结合，捕获局部与全局上下文；②提出按子类均衡与最小子类平衡两种下采样方案；③系统评估不同模型规模与训练策略的计算成本与性能，强调数据平衡对结果的关键影响；

**🔧 技术方法**

Transformer微调（DistilBERT、CodeBERT、Gemma-2B），程序切片提取，类别平衡下采样，超参数调优（学习率、批大小、权重衰减等），GPU加速训练，使用BPE/WordPiece/SentencePiece分词；

**📊 数据集**

公开漏洞数据集SARD、NVD、GitHub等，包含API调用、数组使用、指针使用、算术表达式四类漏洞的程序切片，实验集约56,395漏洞样本与等量非漏洞样本，完整测试集约364,000条；

**📈 对比分析**

采用三种训练策略（完整平衡、最小子类平衡、全量测试），在不同数据规模下训练并在相应或完整测试集评估，指标包括F1、准确率、MCC；结果显示DistilBERT在完整平衡下F1达98.8%，CodeBERT 96.1%，Gemma-2B 96.9%；在最小子类上约95%；在全量测试时F1下降至90-93%；GPU训练时间分别为0.13h、0.98h、49h；

**⚠️ 局限性**

1) 数据来源主要为公开数据集，样本多为已标注的已知漏洞，可能不完全代表真实生产环境；2) 小样本训练时模型泛化能力不足，导致全量测试时性能下降；3) 大模型（Gemma-2B）计算成本极高，显存与时间需求巨大；4) 仅针对C/C++语言，无法直接推广至其他编程语言。

---

## 66. Hierarchical Apprenticeship Learning from Imperfect Demonstrations with Evolving Rewards

**arXiv ID:** 2604.00258 | [PDF](https://arxiv.org/pdf/2604.00258v1)

**作者:** Md Mirajul Islam `[一作]` (North Carolina State University), Min Chi `[通讯]` (North Carolina State University)

**通讯引用:** 2127 | [OpenAlex ID](https://openalex.org/A5090231772)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种名为 HALIDE 的层次化师徒学习框架，能够利用不完美且随时间演化的学生示范来学习 ITS 的教学决策。

**💡 创新点**

在传统 AL 只假设专家轨迹为最优的基础上，首次引入对示范质量的连续排名并将其嵌入层次化学习循环；同时通过奖励调节的时间感知聚类和能量匹配的 EM 训练实现对演化奖励函数的建模。

**🔧 技术方法**

层次化时间感知聚类（RMT–TICC）、加权 EM–EDM 能量匹配、奖励调节的 EM‑IRL、连续质量权重映射、离线强化学习等。

**📊 数据集**

在某高校本科概率课程的 ITS 里收集的 244 名学生跨学期（S21‑S25）交互轨迹，包含 130 维状态特征和三种教学动作。

**📈 对比分析**

与行为克隆、EDM、AHIRL 等平面与层次化基线以及无质量加权版本进行四折跨学期验证；HALIDE 在 Accuracy、F1、AUC、APR、Jaccard 等指标均显著优于基线，尤其在混合质量数据时提升最大。

**⚠️ 局限性**

示范质量仅按轨迹级别估计，无法精确捕获步级效能；仅在离线单一 ITS 领域评估，缺乏闭环部署与跨域验证；排名可能带有先前学习水平偏差，若噪声大可能削弱效果。

---

## 67. Robust Multimodal Safety via Conditional Decoding

**arXiv ID:** 2604.00310 | [PDF](https://arxiv.org/pdf/2604.00310v1)

**作者:** Anurag Kumar `[一作]` (Ohio State University), Yanjun Qi `[通讯]` (AWS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于内部安全分类的多模态大语言模型安全对齐方法（CASA），在生成前先预测二进制安全标记并以此条件化生成；

**💡 创新点**

创新点在于将安全判断嵌入模型内部生成过程，使用安全注意力模块放大恶意信号，避免外部分类器、辅助头或多模态专门微调；

**🔧 技术方法**

使用了内部隐藏状态的安全注意力机制、条件解码、LoRA微调、PEFT策略，模型为Qwen_2.5_Omni 3B/7B；

**📊 数据集**

训练数据包括约6.2k恶意问题、10k安全问题；评测使用MM‑SafetyBench、JailbreakV‑28k、AIAH、文本与音频攻击数据集以及Utility‑Text与MME基准；

**📈 对比分析**

与预训练、SSFT、安全提示、Circuit Breaker等基线对比，CASA在多模态攻击上平均攻击成功率降低超过97%，在预填攻击和视觉/音频攻击上实现接近0%的成功率，且在安全性保持不变的情况下提升了大部分实用性指标；

**⚠️ 局限性**

局限包括可能对更高级的 jailbreak 攻击不足，安全注意力需要跨完整提示计算可能导致计算瓶颈，以及仅针对显式恶意查询的安全范围

---

## 68. Criterion Validity of LLM-as-Judge for Business Outcomes in Conversational Commerce

**arXiv ID:** 2604.00022 | [PDF](https://arxiv.org/pdf/2604.00022v1)

**作者:** Liang Chen `[一作]`, Feng Liang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一款大型中文匹配平台上，系统对对话质量进行多维度评估，并检验这些质量评分与实际业务转化之间的标准效度；

**💡 创新点**

首次系统性揭示多维度质量评分在不同维度上与业务结果关联显著不均衡，证明等权重合成会被无关维度稀释，提出以业务结果为导向的权重重构方法；

**🔧 技术方法**

采用LLM-as-Judge（Claude Opus 4.6）实现多维度评估，结合Spearman相关、Cohen d、Logistic回归、Bonferroni校正等统计方法；

**📊 数据集**

使用平台内约59,316条人类客服对话（已标注的转化标签）作为实验数据，其中60条为分层随机样本（25转化，35未转化）；

**📈 对比分析**

与等权重合成对比，重新加权后Spearman相关从0.272提升至0.351（p=0.006），并通过4折时间交叉验证验证加权方案优于等权重；

**⚠️ 局限性**

局限包括样本量有限（n=60）、仅包含人类客服对话未验证AI对话转化、LLM自评可能产生偏差、Trust Ladder缺乏人工验证、缺乏纵向因果实验等。

---

## 69. VeriAct: Beyond Verifiability -- Agentic Synthesis of Correct and Complete Formal Specifications

**arXiv ID:** 2604.00280 | [PDF](https://arxiv.org/pdf/2604.00280v1)

**作者:** Md Rakib Hossain Misu `[一作]` (University of California, Irvine), Cristina V. Lopes `[通讯]` (University of California, Irvine)

**通讯引用:** 12104 | [OpenAlex ID](https://openalex.org/A5103284742)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对自动生成Java Modeling Language（JML）规范的过程进行了系统评估，并提出了Spec-Harness评估框架和VeriAct代理式闭环方法；

**💡 创新点**

创新点包括（1）Spec-Harness框架可独立测量规范的正确性与完整性；（2）将GEPA提示优化与结构化验证反馈结合；（3）VeriAct将符号验证、错误分析与规范评估反馈闭环到LLM生成流程，实现更精准的规范修正；

**🔧 技术方法**

采用大语言模型（如GPT‑4o、Claude Sonnet 4.6等）+ OpenJML/SMT求解器+ Spec‑Harness 的四项 Hoare‑triple 评估指标+ GEPA 结构化提示优化+ CodeAct 代理框架；

**📊 数据集**

使用 SpecGenBench（120 条 Java 方法）和 FormalBench（662 条 Java 方法，已扩充 100–200 条测试对）两大基准数据集；

**📈 对比分析**

通过比较验证通过率（VR）和意义验证率（MVR），以及 Spec‑Harness 的四项指标，对传统工具 Daikon/Houdini、Prompt‑based 方法 SpecGen、AutoSpec、FormalBench 进行基准；结果显示传统工具 VR 高但 MVR 极低；Prompt‑based 方法 VR 较好但 MVR 仍低；GEPA 提升 VR 但仍受限；VeriAct 在 MVR 上比最佳 Prompt 提升 5%（SpecGenBench）和 12%（FormalBench），并在四项指标上表现优异；

**⚠️ 局限性**

局限性：仅针对 JML 与 Java，未涵盖循环不变式、异常后置等更复杂约束；评估依赖扩充的测试对，可能无法捕捉所有缺陷；LLM 在表达复杂数学性质时仍有限；实验仅在两大基准上验证，泛化能力待进一步验证。

---

## 70. Eyla: Toward an Identity-Anchored LLM Architecture with Integrated Biological Priors -- Vision, Implementation Attempt, and Lessons from AI-Assisted Development

**arXiv ID:** 2604.00009 | [PDF](https://arxiv.org/pdf/2604.00009v1)

**作者:** Arif Aditto `[一作]` `[通讯]` (Independent Researcher), Arif Aditto (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并尝试实现一种名为Eyla的身份锚定LLM架构，结合HiPPO初始化的状态空间模型、零初始化适配器、episodic记忆检索和校准不确定性训练，并在消费者硬件上运行；

**💡 创新点**

创新点在于提出Identity Consistency Score（ICS）这一评估模型在对抗压力下保持身份一致性的基准，并首次公开记录使用AI编码助手构建新架构的完整失败分析；

**🔧 技术方法**

使用的技术包括HiPPO‑initialized SSM、LoRA（零初始化门控）、AIOS（Agent Operating System）框架、零初始化适配器、基于Python的代码生成与验证工具；

**📊 数据集**

使用的数据集为约24 000条精心策划的身份与自我模型示例、约100 000条高质量推理与知识示例，作者未引用公开数据集；

**📈 对比分析**

与基线LLaMA 3.2 1B进行对比，最终模型的输出与基线几乎无差别，ICS未正式评估，模型性能未得到提升；

**⚠️ 局限性**

主要限制包括实现失败导致模型未获得身份一致性、缺乏有效验证与持续反馈、成本高达$1 000+、对非程序员不友好以及未完成正式的ICS基准测试。

---

## 71. Behavioral Score Diffusion: Model-Free Trajectory Planning via Kernel-Based Score Estimation from Data

**arXiv ID:** 2604.00391 | [PDF](https://arxiv.org/pdf/2604.00391v1)

**作者:** Shihao Li `[一作]` (University of Texas at Austin), Dongmei Chen `[通讯]` (University of Texas at Austin)

**通讯引用:** 13157 | [OpenAlex ID](https://openalex.org/A5100677493)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种无模型、无训练的行为得分扩散（BSD）轨迹规划方法，直接利用已收集的轨迹库通过核回归估计扩散得分并进行去噪规划。

**💡 创新点**

创新点在于：1）用核加权估计扩散得分替代传统的神经网络或解析动力学；2）设计三重核（扩散、状态上下文、目标相关性）实现多尺度去噪；3）证明了核估计的一致性并与DeePC的正则化形式对应。

**🔧 技术方法**

主要技术包括 Nadaraya‑Watson 核回归、扩散噪声调度、多样本采样、奖励加权 Softmax、Shielded Rollout 安全屏蔽，以及多尺度宽度控制。

**📊 数据集**

使用了 1000 条基于模型驱动收集的轨迹数据，覆盖四种车辆（3D–6D）停车场景。

**📈 对比分析**

与模型驱动的 MBD/Safe‑MPD、固定宽度 BSD、适应宽度 BSD、以及无扩散的最近邻检索进行对比；固定宽度 BSD 在所有四个系统中平均取得 98.5% 的奖励，几乎与 MBD 相当；相较最近邻提升 18–63% 并保持高安全率；计算时间略高（1.5–3.8×）。

**⚠️ 局限性**

局限性包括：1）相对于模型驱动方法计算开销显著；2）在高维系统仍需更多数据；3）实验仅在仿真中验证，缺乏真实机器人测试；4）安全率在 6D 系统略低；5）训练数据来自模型驱动收集，真正无模型场景需人工或遥控数据。

---

## 72. A Generalized Matrix Inverse that is Consistent with Respect to Diagonal Transformations

**arXiv ID:** 2604.00049 | [PDF](https://arxiv.org/pdf/2604.00049v1)

**作者:** Jeffrey Uhlmann `[一作]` `[通讯]` (University of Missouri-Columbia), Jeffrey Uhlmann (University of Missouri-Columbia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

提出了一种在任意非奇异对角变换下保持一致性的广义矩阵逆，补充了Drazin逆与Moore‑Penrose逆的三元族；

**💡 创新点**

创新点在于构造了对角变换一致的逆，利用矩阵缩放与Sinkhorn迭代实现单位一致性，并推广至一般矩阵、分块系统和其他分解；

**🔧 技术方法**

使用了矩阵缩放理论、对角尺度函数、Rothblum–Zenios的Program II、SVD与UI‑SVD、以及数值实现中的Sinkhorn迭代；

**📊 数据集**

未使用特定数据集，主要为理论推导与算法实现；

**📈 对比分析**

未进行实验比较与性能评估，论文聚焦于理论性质与算法框架；

**⚠️ 局限性**

局限在于对含零元素矩阵需额外缩放处理，唯一性依赖于缩放矩阵，且对数值稳定性与大规模应用未作深入探讨。

---

## 73. How Trustworthy Are LLM-as-Judge Ratings for Interpretive Responses? Implications for Qualitative Research Workflows

**arXiv ID:** 2604.00008 | [PDF](https://arxiv.org/pdf/2604.00008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 74. When Labels Are Scarce: A Systematic Mapping of Label-Efficient Code Vulnerability Detection

**arXiv ID:** 2604.00079 | [PDF](https://arxiv.org/pdf/2604.00079v1)

**作者:** Noor Khalal `[一作]` (Paris Cité university), Mohamed Nadif `[通讯]` (Paris Cité university)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对代码漏洞检测(CVD)中标签高效学习方法进行系统映射与综述，提出了五类标签高效范式，并结合表示形式、评估维度和成本因素构建了设计图谱与决策指南。

**💡 创新点**

创新点在于：①把标签稀缺性作为核心合成轴，对文献按标签高效范式分层整理；②将标签预算、计算成本、泄漏与粒度等评估约束统一成可比对的维度；③基于实证数据生成“标签高效CVD设计图谱”和“约束优先决策指南”，为实践者提供可操作的选型方案。

**🔧 技术方法**

采用的技术包括：半监督/自监督学习（伪标签、教师-学生、Consistency Regularization）、对比学习（InfoNCE、跨视图对齐）、少样本/元学习（MAML、Proto），提示调优（soft/hard prompt、prefix prompt），以及LLM推理与参数高效微调（LoRA、RAG、Chain‑of‑Thought）。

**📊 数据集**

使用的数据集主要为 Big‑Vul、Devign、ReVeal、FFmpeg、QEMU 等公开项目漏洞数据，涵盖 C/C++ 代码；部分研究也引用了 Juliet、SARD、LibTIFF 等标准数据集。

**📈 对比分析**

比较方法：在同一标签预算、同一划分策略下对不同范式进行横向对比，报告 F1/Precision/Recall 以及与预算/成本相关的曲线。总体发现：在稀缺标签条件下，提示调优与对比预训练能取得相当于全监督的性能，半监督+伪标签在中等预算下表现最佳；但性能提升受标签分布、负样本质量与评估拆分影响较大。

**⚠️ 局限性**

局限性：①标签预算与成本报告不统一，导致跨论文比较困难；②评估多聚焦于二分类与 C/C++，缺乏多语言与多粒度验证；③对跨项目、跨语言泛化的实验有限；④LLM 方法对提示与上下文敏感，可能出现信息泄漏；⑤实验复现受缺失预处理、图构建与数据拆分细节影响。

---

## 75. Phonological Fossils: Machine Learning Detection of Non-Mainstream Vocabulary in Sulawesi Basic Lexicon

**arXiv ID:** 2604.00023 | [PDF](https://arxiv.org/pdf/2604.00023v1)

**作者:** Mukhlis Amien `[一作]`, Go Frendi Gunawan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用规则减法和机器学习方法对Sulawesi地区1200多种南岛语中的基本词汇进行子层（非主流）检测，识别出约438个候选非亲属词汇。

**💡 创新点**

提出了基于音位与形态特征的“音韵指纹”——较长词长、更多辅音连缀、更高的喉塞出现率、较少标准南岛语前缀——并通过双模型设计消除认知循环，首次从声韵学角度系统检验子层假设。

**🔧 技术方法**

主要技术包括规则式亲属词减法、XGBoost二分类、SHAP解释、留一语种交叉验证、聚类与Levenshtein距离检验以及多种特征工程（音位长度、辅音连缀、语义域等）。

**📊 数据集**

数据集为Austronesian Basic Vocabulary Database中6个Sulawesi语言（共1,357词条）以及另外16个印尼语言，用Swadesh‑210词汇表做实验。

**📈 对比分析**

相较于传统基于亲属词的规则方法，ML模型在5折交叉验证中取得AUC≈0.76，留一语种验证平均AUC≈0.715；与规则方法的共识区间达到60.7%，Cohen κ≈0.61，显示两种方法高度一致且ML能独立检出非主流词。

**⚠️ 局限性**

局限包括样本量有限、标签噪声（仅基于缺失亲属词），正则化和形态学特征缺失导致音位混淆，语料仅为表面书写，且研究范围局限于Sulawesi，未能覆盖更广的南岛-普朗岛接触环境。

---

## 76. The Geometry of Compromise: Unlocking Generative Capabilities via Controllable Modality Alignment

**arXiv ID:** 2604.00279 | [PDF](https://arxiv.org/pdf/2604.00279v1)

**作者:** Hongyuan Liu `[一作]` (University of Electronic Science and Technology of China), Junming Shao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 4093 | [OpenAlex ID](https://openalex.org/A5088843448)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出模态差距分解为重心差距与分布差距，并设计三阶段课程的交叉模态对齐框架（TPC‑CMA）以同时减小两项差距。

**💡 创新点**

创新点包括：① 证明分布差距是跨模态任务质量的近乎完美预测因子；② 用负样本再加权与内部几何匹配两种机制直接重塑特征分布；③ 采用梯度感知的三阶段课程，动态调节对齐强度，保证训练稳定并兼顾多任务需求。

**🔧 技术方法**

技术手段包括：对比学习（CLIP InfoNCE）、负样本再加权、内部模态几何匹配损失、三阶段梯度感知调度（Anchor→Ramp→Stabilize）、有效维数与模态融合指标等。

**📊 数据集**

数据集：CLIP预训练模型（ViT‑B/32），Fine‑tune 采用 Conceptual Captions 3M；下游评测使用 ImageNet、COCO（检索与图文检索）、多任务零样本分类（CIFAR‑10/100、Food‑101、Caltech‑101、Flowers‑102）、COCO图像描述（DeCap）与联合图文聚类（ImageNet 200 类）。

**📈 对比分析**

与五类基线（Mean‑Centering、AlignCLIP、M^2‑Mix、CLIP‑Refine、CS‑Aligner）对比。TPC‑CMA 在低对齐强度（α=0.05）下保持 ImageNet 约 1–2% 的准确率；在强对齐（α=0.5）下，模态差距下降 82.3%，聚类 ARI 提升至 0.516（+63%），图像描述 CIDEr 提升 57.1%。整体形成可调 Pareto 前沿，覆盖了分类、检索、聚类与生成任务。

**⚠️ 局限性**

局限性：高对齐强度会导致有效维数下降、特征压缩，导致判别性能下降；不同任务需手动设定 α_target，缺乏自动化选择；仅针对图像‑文本对，跨模态扩展仍需验证。

---

## 77. Fiber-Navigable Search: A Geometric Approach to Filtered ANN

**arXiv ID:** 2604.00102 | [PDF](https://arxiv.org/pdf/2604.00102v1)

**作者:** Thuong Dang `[一作]` `[通讯]` (University of Düsseldorf), Thuong Dang (University of Düsseldorf)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种几何框架，用本地信号指导过滤近邻搜索，兼顾过滤后子图的拓扑与几何；

**💡 创新点**

创新点在于：①通过纤维密度与漂移两个局部信号描述过滤子图的几何；②设计两阶段漂移引导搜索，结合全图探索；③引入轻量化锚点图实现高效重启，统一三种搜索失败模式；

**🔧 技术方法**

使用基于相似度的邻接图（α-kNN或HNSW）、k-means聚类构造锚点图、漂移引导的双阶段搜索算法以及基于锚点的重启机制；

**📊 数据集**

在 H&M 产品嵌入数据集（105,100 维向量，24 个离散元数据字段）上进行实验；

**📈 对比分析**

与 FAISS HNSW 的 post‑filter 与 traversal‑filter 进行比较，Recall@25 在 10,000 个查询上提升至 78%+（相较 30% 左右），且失效率近零，且实验展示了不同过滤选择性下的性能曲线；

**⚠️ 局限性**

局限包括仅在单一数据集上验证、实现语言为 Python（相较生产系统缺乏编译优化）、以及对更大规模数据集与多种过滤谓词的进一步评估仍待完成。

---

## 78. AI-Mediated Explainable Regulation for Justice

**arXiv ID:** 2604.00237 | [PDF](https://arxiv.org/pdf/2604.00237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 79. How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study

**arXiv ID:** 2604.00005 | [PDF](https://arxiv.org/pdf/2604.00005v1)

**作者:** Moran Sun `[一作]` (Beihang University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 85720 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 E-STEER 的情绪调控框架，能够在大语言模型和基于 LLM 的 Agent 的隐藏层进行可解释且可控的情绪干预，从而系统地研究情绪对推理、生成、风险和多步决策行为的影响。

**💡 创新点**

创新点包括：①将三维 VAD（Valence–Arousal–Dominance）情绪空间映射到稀疏自编码器的可解释稀疏特征；②通过在单个 Transformer 块上注入可调节的 steering 方向实现连续情绪控制；③在 LLM 任务与 Agent 多步决策中统一评估情绪效应，揭示情绪与行为之间的非线性（倒U型）关系。

**🔧 技术方法**

核心技术包括：稀疏自编码器（Sparse Autoencoder）构建可解释的稀疏潜在空间；VAD 情绪理论用于定义连续情绪维度；前向钩子（forward hooks）实现对隐藏状态的实时注入与修改；对比学习与激活差分用于识别情绪相关的稀疏神经元。

**📊 数据集**

使用的主要数据集有：LogiQA 2.0（逻辑推理）、HumanEval（代码生成）、Math（量化/科学推理）、TinyStories（开放式文本生成）、HarmBench（安全评估）、HotpotQA、Scientific、GAIA（Agent 任务）等。

**📈 对比分析**

通过与中性情绪状态的基线比较，实验显示：在目标推理任务中，适度情绪提升约 3–15% 的成功率；在规划阶段，低 Valence/低 Arousal 与高 Dominance 可将计划有效率提升 30–80%；在决策与执行阶段，正 Valence/正 Dominance 与中等 Arousal 可使合理选择率和系统成功率提高 15–30%。实验结果普遍呈现倒U型曲线，说明情绪效应具有非线性最优区间。

**⚠️ 局限性**

局限性包括：VAD 三维并非严格正交，导致维度间交叉影响；情绪在任务执行过程中可能随时间演化但未在本框架中建模；实验仅覆盖文本类任务和基于工具的 Agent，无法直接推广到视觉或多模态场景；对极端情绪状态的控制不够稳健，部分情绪注入可能引发生成不稳定。

---

## 80. MSA-Thinker: Discrimination-Calibration Reasoning with Hint-Guided Reinforcement Learning for Multimodal Sentiment Analysis

**arXiv ID:** 2604.00013 | [PDF](https://arxiv.org/pdf/2604.00013v1)

**作者:** Miaosen Luo `[一作]` (South China Normal University), Sijie Mai `[通讯]` (South China Normal University)

**通讯引用:** 1765 | [OpenAlex ID](https://openalex.org/A5010270301)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MSA-Thinker两阶段训练框架，结合Discrimination-Calibration结构与Hint-GRPO强化学习，实现可解释的多模态情感分析。

**💡 创新点**

设计DC结构使情绪先粗分辨后细调，并在RL中使用判别标签作为提示，解决奖励稀疏与优化冲突；两阶段进化式训练提升模型鲁棒性与解释性。

**🔧 技术方法**

使用教师模型生成高质量CoT数据、筛选后进行SFT；随后采用GRPO强化学习，并加入Hint-GRPO在硬样本的判断阶段注入真值提示；通过格式、极性、分数三元奖励与KL正则实现策略优化。

**📊 数据集**

训练集：CH‑SIMS；交叉评测集：CH‑SIMS v2、CMU‑MOSI、CMU‑MOSEI；同时在基准对照中使用多模态LLM数据。

**📈 对比分析**

与7B参数的PandaGPT、Emotion‑LLaMA、MiniCPM‑o、Ola、VideoLLaMA2、HumanOmni、Qwen2.5Omni等基线对比；在跨域Acc7/Acc2/F1/MAE/Corr等指标上均优于或相当于最强基线，特别在跨域MAE下降、Corr提升明显。

**⚠️ 局限性**

在本域回归精度略逊于SFT单阶段，DC结构与细分回归存在权衡；需要更大规模、多语言或多任务验证；教师模型生成CoT数据的成本与质量控制仍是挑战。

---

## 81. Large Language Models in the Abuse Detection Pipeline

**arXiv ID:** 2604.00323 | [PDF](https://arxiv.org/pdf/2604.00323v1)

**作者:** Suraj Kath `[一作]` (Google LLC), Shivani Gupta `[通讯]` (Google LLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述 LLM 在在线滥用检测生命周期（标注、检测、审查、审计）中的应用，构建了 ADL 框架并系统性分析各阶段技术与实践。

**💡 创新点**

提出 Abuse Detection Lifecycle (ADL) 框架，聚焦 LLM 在多阶段的多功能角色，阐明其对语境推理、解释生成和治理层面的关键作用。

**🔧 技术方法**

讨论的技术包括大型预训练语言模型（GPT‑4、Llama Guard、MetaTox 等）、检索增强生成 (RAG)、多模态融合、图神经网络、对话代理等。

**📊 数据集**

综述引用的公开数据集涵盖 Jigsaw Toxicity、BASIL、BABE、TOXIGEN、ToxicChat、Disinformation 等。

**📈 对比分析**

比较方法主要是零样本/少样本推理、fine‑tuned 专用模型、检索增强推理等；LLM 在复杂语境下往往优于传统 BERT，但在延迟与成本上表现不佳。

**⚠️ 局限性**

主要局限包括推理延迟高、成本昂贵、随机性与对抗鲁棒性不足、偏见与公平问题、解释可信度低，以及治理与持续更新挑战。

---

## 82. Structure- and Event-Driven Frameworks for State Machine Modeling with Large Language Models

**arXiv ID:** 2604.00275 | [PDF](https://arxiv.org/pdf/2604.00275v1)

**作者:** Samer Abdulkarim `[一作]` (McGill University), Gunter Mussbacher `[通讯]` (McGill University)

**通讯引用:** 1590 | [OpenAlex ID](https://openalex.org/A5113550853)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于大语言模型（LLM）的框架，用来从非结构化自然语言描述自动生成 UML 状态机，包含单步生成和两种多步生成策略（结构驱动 SMF、事件驱动 SMF）以及混合（Hybrid）策略，并对两类 LLM（非推理 GPT‑4o 与推理 Claude‑3.5 Sonnet）进行系统评估。

**💡 创新点**

创新点在于提出两种以人类建模思路为启发的多步生成框架（Structure‑Driven SMF 与 Event‑Driven SMF），以及将单步结果与多步细化结合的 Hybrid Approach；同时首次对推理与非推理 LLM 在状态机自动生成中的表现差异进行对比研究。

**🔧 技术方法**

使用技术包括：LLM 的单步与多步 Prompt（Chain‑of‑Thought/Tree‑of‑Thought）、HTML 表格作为中间结构化表示、规则化后处理生成 Umple 代码，以及基于精度/召回/F1 的评估指标。

**📊 数据集**

数据集为 8 个本科建模课程的案例，包含非结构化自然语言描述及对应人工标注的 UML 状态机（涵盖打印机、洗碗机、棋钟等多领域实例）。

**📈 对比分析**

评估方法为对 7 种状态机组成（状态、转换、守卫、动作、层级状态、并行区、历史状态）分别计算精度、召回与 F1，并对整体 F1 进行汇总。结果显示，非推理 LLM 的单步 baseline F1≈0.54；结构驱动 SMF 与 Hybrid 通过多步细化提升至≈0.65；而推理 LLM 在单步 baseline 已达 F1≈0.70，后续多步策略并未进一步提升。

**⚠️ 局限性**

限制因素包括：仅评估两种 LLM，数据集仅 8 个案例且来自本科课程；评估手工完成，可能带有主观偏差；单步与多步输出语法不同导致对比不完全公平；动作、并行区、历史状态等复杂组件性能仍较低。

---

## 83. Speeding Up Mixed-Integer Programming Solvers with Sparse Learning for Branching

**arXiv ID:** 2604.00094 | [PDF](https://arxiv.org/pdf/2604.00094v1)

**作者:** Selin Bayramoğlu `[一作]` (Georgia Institute of Technology), Nikolaos V Sahinidis `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 15243 | [OpenAlex ID](https://openalex.org/A5031811254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种稀疏二次回归模型，用来近似强分支（Strong Branching）得分，从而在整数规划的分支定界过程中快速做出分支决策。

**💡 创新点**

创新点在于：①用极低维度（<4%）的可解释模型取代大规模图神经网络；②仅需CPU即可完成推理；③在训练数据极少的“small‑sample”设定下仍能取得竞争性能；④通过特征预处理与二次交互提升预测精度。

**🔧 技术方法**

技术方法包括：l1 正则化稀疏回归、二次特征扩展、动态特征重计算、特征方差剔除、对模型大小进行参数约束与调优；实验中使用 SCIP 8.0.0+SoPlex 6.0.0 进行数据采集与评测。

**📊 数据集**

数据集：四个经典NP‑hard MIP 领域（Set Covering、Combinatorial Auctions、Maximum Independent Set、Capacitated Facility Location），每类训练集 10,000 条实例（2,000 验证/测试），评估集包含 20 个小、20 个中、20 个大规模实例，采用与前人相同的生成器。

**📈 对比分析**

与基线（SCIP 默认、Vanilla Full Strong Branching、CPU/GPU 图神经网络）在同一硬件（Intel Xeon Gold 6226 CPU，GPU 仅用于 GNN-G）下对 100 条每类随机种子实例进行比对；结果显示：稀疏模型在小/中/大规模实例上平均 3–4 秒/次的 CPU 时间，节点数仅比默认策略高 10–20%，且在 20%~25% 的问题上比 GNN-G 更快；在大规模实例上，small‑sample 训练的模型取得最高的求解率与最快的时间。

**⚠️ 局限性**

局限性：①对特征工程的依赖较高，若特征不完整模型性能下降；②虽然推理极快，但在极大规模实例或特殊结构 MIP 中仍无法完全匹配 VFS 的树大小；③在某些领域（如 Max Independent Set）二次模型相对较弱；④模型需要在每个分支节点重新计算动态特征，导致在极深树上仍有一定计算开销；⑤在 GPU 受限的环境下虽表现优越，但在 GPU 充足时 GNN 仍可在树结构上略占优势。

---

## 84. Label-efficient underwater species classification with semi-supervised learning on frozen foundation model embeddings

**arXiv ID:** 2604.00313 | [PDF](https://arxiv.org/pdf/2604.00313v1)

**作者:** Thomas Manuel Rost `[一作]` `[通讯]`, Thomas Manuel Rost

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在未标注水下图像上使用冻结的 DINOv3 ViT Embedding 进行半监督分类，仅需少量标注即可实现高精度的物种识别。

**💡 创新点**

创新点在于证明通用自监督模型的嵌入空间足以支持简单的 KNN/自训练方法完成水下物种分类，显著降低标注成本；并首次在 AQUA20 任务上展示冻结 Embedding 与全监督 ConvNeXt 的性能差距可缩小至约 1–3%。

**🔧 技术方法**

技术主要包括：冻结的 DINOv3 ViT‑B/16 自监督嵌入、PCA 降维、KNN 基线和自训练 KNN（带置信阈值）。

**📊 数据集**

使用了 AQUA20 基准数据集（8171 张图像，20 个海洋物种）。

**📈 对比分析**

与全监督 ConvNeXt 及多种传统半监督方法对比，5% 标注（≈15 张/类）即可达到 80% 宏 F1；在 100% 标注时，KNN/自训练方法的宏 F1 达到 87–89%，仅比 ConvNeXt 略低，且部分物种性能更优。

**⚠️ 局限性**

局限性包括：自训练 KNN 在极少标注（≤5/类）时表现不稳定；对高变异或相似物种仍存在误分类；实验仅针对单一数据集和单一基础模型；未做超参数优化和跨数据集泛化验证。

---

## 85. Temporal Memory for Resource-Constrained Agents: Continual Learning via Stochastic Compress-Add-Smooth

**arXiv ID:** 2604.00067 | [PDF](https://arxiv.org/pdf/2604.00067v1)

**作者:** Michael Chertkov `[一作]` `[通讯]` (University Arizona), Michael Chertkov (University Arizona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于桥扩散的连续学习框架，利用压缩-添加-平滑(CAS)递归在固定内存下记忆并重放序列经验。

**💡 创新点**

创新点在于将记忆从参数向量转为随机过程，借助桥扩散的线性插值实现无梯度、无数据存储的持续学习，并发现保留半寿命随时间段数L线性增长，给出经验常数c≈2.4的容量定律。

**🔧 技术方法**

核心技术包括桥扩散（Bridge Diffusion）、线性插值、Gaussian Mixture 模型、SDE 轨迹重放、信息论分析与马尔科夫/最优输运等。

**📊 数据集**

主要使用合成高斯混合序列数据以及MNIST图像的PCA潜在空间（d=12）进行实验验证。

**📈 对比分析**

与FIFO缓冲、参数正则化、传统重放方法等对比显示，CAS在同等内存预算下可保留约2.4倍天数的记忆，保留半寿命a½≈cL；实验中取得了良好的数值一致性与可解释性。

**⚠️ 局限性**

限制在于插值需在可线性插值的密度族内（如高斯混合），对更复杂的神经网络密度、非均匀时间网格或非线性插值的支持尚未成熟；此外，理论上对信息容量的上界与最优分配仍是开放问题。

---

## 86. In harmony with gpt-oss

**arXiv ID:** 2604.00362 | [PDF](https://arxiv.org/pdf/2604.00362v1)

**作者:** Borislav Mavrin `[一作]` `[通讯]`, Borislav Mavrin

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过逆向工程恢复模型的内置工具集并实现本地 Harmony 消耗器，重新实现了 OpenAI 的  在 AIME25 与 SWE‑Verified 上的成绩。

**💡 创新点**

创新点在于识别并定义模型训练时使用的工具名称与 JSON 架构，并构建不依赖 Chat Completions 的原生 Harmony 消耗器，从而消除了工具缺失与格式转换导致的性能差距。

**🔧 技术方法**

主要技术包括：工具提取与校准（利用工具调用频率与自定义提示）、Bootstrap 置信区间统计、Harmony 格式的消息序列化/解析、异常检测与重试机制，以及基于容器的工具执行。

**📊 数据集**

使用的数据集为公开的 SWE‑Verified（High/Medium 级别）和 AIME‑25（包含工具的 Medium 级别）两大编码评测基准。

**📈 对比分析**

与 OpenAI 官方发布的分数对比，本文在 SWE‑Verified High 上 60.4%±(56.2–64.8%) 与 60.7% 对齐，在 Medium 上 53.3%±(49.3–57.7%) 与 53.2% 对齐，在 AIME‑25 工具版上 91.7%±(87.5–95.0%) 与 90.4% 对齐，表明在相同任务与工具集下几乎无性能损失。

**⚠️ 局限性**

局限性包括：依赖于 OpenAI 提供的 Harmony 消耗器与工具定义，仅在  上验证；未探索不同模型架构或更大规模模型的泛化能力；并且在长上下文与高步骤数任务中仍需改进异常处理与上下文压缩策略。

---

## 87. RawGen: Learning Camera Raw Image Generation

**arXiv ID:** 2604.00093 | [PDF](https://arxiv.org/pdf/2604.00093v1)

**作者:** Dongyoung Kim `[一作]` (AI Center - Toronto, Samsung Electronics), Michael S. Brown `[通讯]` (AI Center - Toronto, Samsung Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RawGen，一个基于扩散模型的、可从文本或 sRGB 图像生成真实线性相机原始数据的框架；

**💡 创新点**

创新点在于使用多对一（many‑to‑one）训练目标，让模型学习从多种未知 ISP 生成的 sRGB 图像恢复共享的场景参照 CIE XYZ 表示，并实现文本驱动的原始数据合成及任意目标相机的线性映射；

**🔧 技术方法**

技术主要包括预训练的 DiT（FLUX.1‑Kontext）与 VAE、LoRA 微调、在 latent 及像素空间的去处理（unprocessing）以及在逆 ISP 过程中加入多样化的 ISP 参数生成数据；

**📊 数据集**

使用了 MIT‑Adobe FiveK 与 RAISE 原始图像数据集进行训练，并在 Samsung Galaxy S24（S24）数据集上进行评估；

**📈 对比分析**

与 CIE XYZ Net、InvISP、Raw‑Diffusion 等基线对比，RawGen 在多对一 sRGB‑to‑XYZ 逆 ISP 任务上在 PSNR/SSIM 上优于对手，并且其生成的相机原始图像在后置 ISP 评估中与真实原始图像分布对齐；

**⚠️ 局限性**

局限在于对相机特定噪声、点扩散函数（PSF）等物理细节建模不足，无法完全再现特定相机的噪声谱和光学失真，后续工作需进一步强化设备特定的感知建模。

---

## 88. Advancing Multi-Robot Networks via MLLM-Driven Sensing, Communication, and Computation: A Comprehensive Survey

**arXiv ID:** 2604.00061 | [PDF](https://arxiv.org/pdf/2604.00061v1)

**作者:** Hyun Jong Yang `[一作]` (Seoul National University), Byonghyo Shim `[通讯]` (Seoul National University)

**通讯引用:** 7604 | [OpenAlex ID](https://openalex.org/A5076075267)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了基于多模大语言模型（MLLM）的 R2X（Robot-to‑everything）框架，统筹感知、通信与计算，实现多机器人协同，并通过四个端到端演示验证该框架。

**💡 创新点**

创新地将高层自然语言意图映射到感知、通信和计算资源配置，实现“intent‑to‑resource”协同；融合语义压缩、预测链路上下文和集中规划，构建统一的多维度（物理层、感知、AI、网络、计算、意图）R2X体系；并通过实验展示显著提升任务完成效率。

**🔧 技术方法**

使用多模视觉特征编码与向量量化、基于光线追踪的链路预测、冲突基搜索（CBS）路径规划、LLaMA 3 8B 语言模型进行意图配置、元提示式 JSON 校验、边缘/云计算协同、语义感知切换等技术。

**📊 数据集**

主要基于 NVIDIA Isaac Sim 仿真生成的数字孪生仓库环境及同步的 RGB‑D 图像，用于训练语义编码器与人类预测模型；实验中使用模拟移动和链路模型进行 MCS 适配；未公开使用特定公开数据集。

**📈 对比分析**

通过四种配置对比（Stop‑and‑Go、LORC‑P、LORC‑SC、LORC‑SC‑P）评估任务完成时间、上行负载、延迟；实验显示 LORC‑SC‑P 在所有场景下均实现约 20‑30% 的完成时间提升，且上行负载减少到 1/10，显著降低延迟与链路停顿。

**⚠️ 局限性**

受限于理想化的光线追踪链路模型、离散网格移动、有限机器人/人类数量，难以直接迁移至真实复杂环境；缺乏大规模实验与动态链路自适应的实时评估；需要进一步完善双生到实机的校准与鲁棒性研究。

---

## 89. Neural Reconstruction of LiDAR Point Clouds under Jamming Attacks via Full-Waveform Representation and Simultaneous Laser Sensing

**arXiv ID:** 2604.00371 | [PDF](https://arxiv.org/pdf/2604.00371v1)

**作者:** Ryo Yoshida `[一作]` (Keio University), Kentaro Yoshioka `[通讯]` (Keio University)

**通讯引用:** 2940 | [OpenAlex ID](https://openalex.org/A5055467060)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发并验证了一种基于LiDAR全波形数据和多光束同步感知的神经网络PULSAR-Net，用于在Jamming攻击下恢复点云。

**💡 创新点**

引入全波形表示与同步激光感知的双重特征，并设计轴向空间-时间注意力的U-Net网络，实现攻击脉冲的分割与抑制。

**🔧 技术方法**

使用3D U-Net+深度可分离卷积+轴向注意力机制，结合加权交叉熵+Dice损失进行训练，并构建全波形仿真攻击生成管线。

**📊 数据集**

利用KITTI与nuScenes合成的Jamming攻击数据集（KITTI‑J、nuScenes‑J），以及自研全波形LiDAR的真实攻击实验数据。

**📈 对比分析**

与平均减法和Neural DSP基线对比，在合成数据上点级恢复率≥82%/≥89%，检测mAP恢复至≈65/≈59；在真实实验中静态/动态场景mAP几乎回到无攻击水平。

**⚠️ 局限性**

受攻击仿真精度限制，模型对极端攻击参数或不同LiDAR硬件可能泛化不足，且未对移动物体在极限速度下进行充分验证。

---

## 90. Hierarchical Pre-Training of Vision Encoders with Large Language Models

**arXiv ID:** 2604.00086 | [PDF](https://arxiv.org/pdf/2604.00086v1)

**作者:** Eugene Lee `[一作]` (University of Cincinnati), Chen-Yi Lee `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 6004 | [OpenAlex ID](https://openalex.org/A5101838014)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 HIVE 框架，通过在视觉编码器与大语言模型之间构建分层交叉注意力来预训练视觉编码器，提升视觉与语言的对齐；

**💡 创新点**

创新点在于：①在多层级视觉特征上直接进行交叉注意力，实现结构化特征融合；②采用分阶段训练策略（投影器→投影器+LLM→全模型）保证梯度流畅；③通过分层投影仅选择部分层，降低计算开销；

**🔧 技术方法**

技术包括：视觉编码器（CLIP、SigLIP）提取多层特征，投影器（轻量 MLP）映射至 LLM；大语言模型（MobileLLM‑350M、Llama‑3.2‑1B‑Instruct）；分层交叉注意力机制；三阶段训练与微调流程；

**📊 数据集**

使用的基准数据集：分类任务（CIFAR‑10/100、ImageNet‑1K、Tiny‑ImageNet、Food‑101、Stanford Cars、Oxford‑IIIT Pets、Caltech‑256）；视觉‑语言任务（MME、GQA、OK‑VQA、ScienceQA）；

**📈 对比分析**

与自注意力基线（SA）和基础 CLIP/SigLIP 进行对比，HIVE 在大多数分类数据集上提升 0.1‑0.3% 甚至更高，在 VLM 任务中相对 SA 提升 1‑3% 分数；同时实现约 3 倍的训练速度提升和 55% 的显存降低；

**⚠️ 局限性**

局限性：仅验证于静态图像，缺乏对视频或多模态序列的扩展；预训练阶段仍需多阶段训练和大规模算力；对 LLM 的冻结策略可能限制某些任务的进一步微调效果；

---

## 91. Measuring the Representational Alignment of Neural Systems in Superposition

**arXiv ID:** 2604.00208 | [PDF](https://arxiv.org/pdf/2604.00208v1)

**作者:** Sunny Liu `[一作]` (Cold Spring Harbor Laboratory), David Klindt `[通讯]` (Cold Spring Harbor Laboratory)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5059395392)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文在假设神经网络以线性压缩方式编码超多特征的前提下，推导并验证了RSA、CKA与线性回归三种对齐指标如何被超位置（superposition）系统投影矩阵差异所扭曲。

**💡 创新点**

创新点在于给出闭式表达式揭示对齐指标仅依赖投影矩阵的Gram矩阵，而非潜在特征内容；并通过理论与仿真展示在部分特征重叠时，压缩度越高，系统对齐分数反而可能低于特征共享更少的系统，从而指出传统对齐方法的根本局限。

**🔧 技术方法**

使用了线性投影模型、稀疏感知理论、闭式统计推导、仿真实验以及RSA/CKA/线性回归对齐度量。

**📊 数据集**

主要使用随机生成的稀疏潜变量数据（高维稀疏向量）和随机正态投影矩阵进行仿真，未使用公开实际数据集。

**📈 对比分析**

比较方法为将两个网络的潜变量投影为不同随机投影矩阵得到的神经激活，然后计算RSA、CKA与线性回归的对齐得分；实验表明随着压缩比例升高，对齐得分显著下降，且在部分重叠情形下压缩度高的网络往往得分更高，揭示了传统指标的误导性。

**⚠️ 局限性**

局限在于假设了线性超位置、独立投影矩阵且潜变量满足稀疏感知条件；对非线性模型、实际神经数据、以及如何在实践中提取潜在特征的具体方法仍需进一步研究。

---

## 92. Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in Large Language Model Representations

**arXiv ID:** 2604.00209 | [PDF](https://arxiv.org/pdf/2604.00209v1)

**作者:** Haoran Wang `[一作]` (Emory University), Kai Shu `[通讯]` (Emory University)

**通讯引用:** 11902 | [OpenAlex ID](https://openalex.org/A5058670321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型（LLM）内部是否编码了情境隐私规范，并提出一种基于上下文完整性（CI）参数的推理时 Steering 方法；

**💡 创新点**

创新点在于发现隐私规范在激活空间中是多维子空间，CI 参数（信息类型、接收者、传输原则）对应相互独立的方向，并利用这些方向实现可解释、可调节的隐私控制；

**🔧 技术方法**

使用线性探针、PCA、LDA 等技术对激活空间进行表征学习，并在推理时通过向 top‑k 层的激活向量添加按 CI 维度方向的偏移实现 Steering；

**📊 数据集**

实验数据集包括合成的 CI 基准、CONFAIDE Tier 3 以及 PrivaCI‑Bench 等公开隐私评测数据集；

**📈 对比分析**

与单向 Steering、LoRRA、Rep Tuning 等基线比较，CI‑Parametric 在合成数据上将泄漏率从42.5 %降至5 %，在 CONFAIDE 和 PrivaCI‑Bench 上泄漏率均降至 0‑5 %（PPI≈90 %），且展现出更好的跨数据集泛化；

**⚠️ 局限性**

局限性包括需先提取 CI 方向、对模型层次和调参 α 的依赖，以及对极端对抗或动态情境下的鲁棒性尚未充分验证。

---

## 93. Semantic Shifts of Psychological Concepts in Scientific and Popular Media Discourse: A Distributional Semantics Analysis of Russian-Language Corpora

**arXiv ID:** 2604.00017 | [PDF](https://arxiv.org/pdf/2604.00017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 94. Collaborative AI Agents and Critics for Fault Detection and Cause Analysis in Network Telemetry

**arXiv ID:** 2604.00319 | [PDF](https://arxiv.org/pdf/2604.00319v1)

**作者:** Syed Eqbal Alam `[一作]` (University of Alberta), Zhan Shu `[通讯]` (University of Alberta)

**通讯引用:** 5505 | [OpenAlex ID](https://openalex.org/A5101561346)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于协同控制的多代理、多评审者的联邦多模态系统，用以完成网络遥测中的故障检测、严重性评估与原因分析。

**💡 创新点**

创新点在于：①采用双时间尺度随机逼近方法，保证代理与评审者的长期平均活跃状态收敛至全局最优；②将传统机器学习与大型生成式模型（LLM）混合使用，形成代理-评审者协作框架；③通信复杂度仅为 O(m)，与代理/评审者数量无关；④通过对代理/评审者的成本函数保持私有，实现隐私友好的协作。

**🔧 技术方法**

使用的技术包括：多模态基础模型（LLM/LVM）、传统机器学习模型（XGBoost）、随机逼近与多时间尺度优化、RAG（检索增强生成）技术、ChromaDB 进行向量检索、Ollama 与 Langchain 集成。

**📊 数据集**

实验数据集为公开的网络遥测数据集 4（包含 99 列 CSV 与端口状态日志），并将其转化为文本查询进行 LLM 处理。

**📈 对比分析**

比较方法：将基于 XGBoost 的代理方案与基于 LLM（含 RAG 与不含 RAG）的代理+评审者方案进行准确率、精确率、召回率、F1 评分对比；实验显示 XGBoost 方案在准确率、召回率、F1 上均优于 LLM 方案；在严重性与原因分析上，所有 LLM 方案都能提供文本摘要，但准确度和一致性相对较低。

**⚠️ 局限性**

局限性包括：①LLM 方案计算量大、响应时间长；②对网络遥测数据的预处理复杂；③评审者与代理的成本函数需手工设定，可能缺乏泛化性；④算法在真实网络设备实时部署前仍需进一步验证；⑤对多模态任务的扩展性与不同领域的适用性尚待探索。

---

## 95. A Study on the Impact of Fault localization Granularity for Repository-Scale Code Repair Tasks

**arXiv ID:** 2604.00167 | [PDF](https://arxiv.org/pdf/2604.00167v1)

**作者:** Joseph Townsend `[一作]` (Fujitsu Research of Europe), Matthieu Parizy `[通讯]` (Fujitsu Research of Europe)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在“完美定位”假设下，仓库级自动程序修复任务中不同定位粒度对修复成功率的影响，使用Agentless框架与Qwen3-Coder实验。

**💡 创新点**

首次对仓库级APR进行完美定位假设下的粒度实验，系统比较行级、函数级、文件级定位的修复效果。

**🔧 技术方法**

采用Agentless框架改造，使用Qwen3-Coder-30B LLM进行修复，基于SWE-Bench-Mini数据集的完美定位注入。

**📊 数据集**

SWE-Bench-Mini（50个实例，来自Django和Sphinx）的GitHub问题修复数据。

**📈 对比分析**

在10次实验中统计修复成功率，结果显示函数级粒度最高（45.6%），文件级最低（42.6%），行级波动最大，整体成功率仍低于状态极限。

**⚠️ 局限性**

仅使用完美定位，未考虑定位误差；实验仅在Mini集上，未验证更大数据集；未包含完整Agentic系统、提示工程等因素，且完美定位定义可能忽略相关上下文。

---

## 96. Detecting Abnormal User Feedback Patterns through Temporal Sentiment Aggregation

**arXiv ID:** 2604.00020 | [PDF](https://arxiv.org/pdf/2604.00020v1)

**作者:** Yalun Qi `[一作]` (Northeastern University), Zihan Yu `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于RoBERTa的时序情感聚合框架，用聚合窗口和差分阈值检测用户反馈中的异常模式。

**💡 创新点**

创新点在于用分窗口聚合降低单条评论噪声，采用基于变化的异常检测并引入主题感知聚合提升可解释性。

**🔧 技术方法**

主要技术包括RoBERTa情感分类、时间窗口聚合、阈值差分异常检测与热图可视化。

**📊 数据集**

使用公开社交媒体航班相关用户评论数据集，包含数万条评论、情感标签和投诉类别。

**📈 对比分析**

相较于直接利用原始评论情感进行异常检测，实验显示聚合方法提升约30%准确率且误报率下降，且检测到的情感跌落与投诉主题高度对应。

**⚠️ 局限性**

局限性包括对RoBERTa情感预测偏差的依赖、窗口大小需人工调参，以及仅在单一域内验证，缺乏跨领域泛化评估。

---

## 97. Blockspace Under Pressure: An Analysis of Spam MEV on High-Throughput Blockchains

**arXiv ID:** 2604.00234 | [PDF](https://arxiv.org/pdf/2604.00234v1)

**作者:** Wenhao Wang `[一作]` (Yale University), Fan Zhang `[通讯]` (Yale University)

**通讯引用:** 18595 | [OpenAlex ID](https://openalex.org/A5065523443)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在高吞吐量低费用链上构建理论框架，推导“垃圾 MEV”竞争均衡下的垃圾交易量，并分析其对用户福利、验证者收入和网络外部性的影响；随后在 Base 与 Arbitrum 上进行案例研究和经验验证。

**💡 创新点**

提出首个以竞争均衡为基础的“垃圾 MEV”理论模型，揭示区块容量、手续费底价与交易费机制三大设计杠杆如何决定垃圾交易的规模，并给出闭式解与参数设定准则；同时引入优先费排序与需求扩张分析，为缓解垃圾交易提供新的思路。

**🔧 技术方法**

利用游戏理论与竞争均衡分析推导公式，构建优先费排序近似模型；采用线性需求与机会值假设求解闭式解；通过回归与对比实验检验模型预测；在链上使用 DEX‑探测方法提取垃圾交易，计算每日垃圾 gas 与总体 gas。

**📊 数据集**

收集 2024‑01 至 2026‑02 期间 Base 与 Arbitrum 的每日 gas 与 spam gas 数据（约 790 条记录），来源于 Dune Analytics；通过 DEX‑探测启发式识别垃圾交易，结合链上 gas 目标与最小 gas 价格变化。

**📈 对比分析**

通过对比“垃圾交易”与无垃圾交易的对照世界，以及对 gas 目标和最小 gas 价格的回归（log‑linear），评估参数变化对垃圾量的影响；结果显示垃圾 gas 与 gas 目标呈 2.27 的弹性，最小 gas 价格提升能显著降低垃圾比例，优先费排序可进一步压缩垃圾份额；模型预测与实测趋势高度一致。

**⚠️ 局限性**

模型假设需求和机会值为线性，未考虑复杂的交易动态和链间差异；垃圾交易识别基于启发式，可能漏检或误判；优先费排序的近似实现与实际实现差异较大；模型侧重验证者与用户视角，未全面覆盖网络层面成本与治理成本。

---

## 98. MVNN: A Measure-Valued Neural Network for Learning McKean-Vlasov Dynamics from Particle Data

**arXiv ID:** 2604.00333 | [PDF](https://arxiv.org/pdf/2604.00333v1)

**作者:** Liyao Lyu `[一作]` (University of California, Los Angeles), Hayden Schaeffer `[通讯]` (University of California, Los Angeles)

**通讯引用:** 1698 | [OpenAlex ID](https://openalex.org/A5086926394)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种测度值神经网络（MVNN），能够从粒子轨迹数据直接推断平均场动力学中的测度依赖漂移项，并证明其良定性、传播混沌以及在低维测度依赖假设下的近似收敛性。

**💡 创新点**

创新点：①将传统神经网络推广到可对概率测度进行顺序不变操作；②利用圆柱功能框架学习可插值测度特征；③在理论上证明了所学习的动力学满足McKean–Vlasov方程并具备传播混沌；④给出了在低维测度依赖下的全局逼近率；⑤通过多组实验展示了强泛化与线性计算复杂度。

**🔧 技术方法**

技术手段：测度值神经网络（嵌入网络 + 交互网络）; Girsanov似然估计/最小二乘回归；Adam优化；JAX自动微分；对多组和二阶系统的扩展；在证明中使用了McKean–Vlasov理论、传播混沌、全局逼近定理和低维测度依赖假设。

**📊 数据集**

数据集：通过数值仿真生成的粒子轨迹，涵盖一阶与二阶系统：1D Motsch‑Tadmor（确定性与随机）、2D 聚集模型、Cucker‑Smale、层次多组系统；每个实验使用数千至数万粒子，轨迹数量从100到200不等，训练集与测试集分别随机抽样。

**📈 对比分析**

比较方法：与基于高斯过程（GP）模型、对称二体相互作用核回归等传统方法对比；评估指标包括密度分布的L²误差、模拟时间与粒子数的关系。结果表明：①MVNN在大规模粒子数下保持O(N)时间；②在未见初始分布下误差显著低于GP；③在多组系统和二阶动力学中同样取得高精度预测。

**⚠️ 局限性**

局限性：①逼近复杂性随测度维度呈指数增长，需低维测度依赖假设；②当前仅学习漂移项，扩散需后续研究；③不处理高阶相关（BBGKY等）或强耦合情况下的动力学；④对测度支持空间的先验假设可能限制在更通用的物理场景。

---

## 99. Beyond Symbolic Control: Societal Consequences of AI-Driven Workforce Displacement and the Imperative for Genuine Human Oversight Architectures

**arXiv ID:** 2604.00081 | [PDF](https://arxiv.org/pdf/2604.00081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 100. VADMamba++: Efficient Video Anomaly Detection via Hybrid Modeling in Grayscale Space

**arXiv ID:** 2604.00360 | [PDF](https://arxiv.org/pdf/2604.00360v1)

**作者:** Jihao Lyu `[一作]` (Xi'an University of Technology), Cheng Shi `[通讯]` (Xi'an University of Technology)

**通讯引用:** 11425 | [OpenAlex ID](https://openalex.org/A5071499609)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出VADMamba++，一种单代理任务、无辅助输入的灰度到彩色推理框架，用于视频异常检测。

**💡 创新点**

创新点包括：灰度→彩色推理范式、异构Transformer–Mamba–CNN混合模型、内任务融合评分策略，以及高效的单代理任务设计。

**🔧 技术方法**

使用技术包括Mamba（S6）序列建模、Transformer自注意力、CNN局部特征提取、向量量化、梯度和色彩一致性损失，以及显式与隐式异常分数融合。

**📊 数据集**

实验数据集为Ped2、Avenue和ShanghaiTech（SHT）。

**📈 对比分析**

与多种单任务与多任务SOTA方法对比，VADMamba++在三大基准上均取得最高AUC（Ped2 99.6%、Avenue 91.9%、SHT 77.1%），且实时推理速度高达133 FPS，显著优于传统方法。

**⚠️ 局限性**

局限性在于仅采用单代理任务，可能对某些色彩或动态细节异常捕捉不如多任务模型；对超长序列的鲁棒性有限；以及对超参数（k、λ）仍有一定敏感性。

---

## 101. mmAnomaly: Leveraging Visual Context for Robust Anomaly Detection in the Non-Visual World with mmWave Radar

**arXiv ID:** 2604.00382 | [PDF](https://arxiv.org/pdf/2604.00382v1)

**作者:** Tarik Reza Toha `[一作]` (University of North Carolina at Chapel Hill), Shahriar Nirjon `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5044492162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨模态的异常检测框架，利用mmWave雷达与RGB-D相机协同工作，生成基于视觉上下文的期望雷达谱并与实际雷达信号比较，以实现对隐藏武器、墙后入侵者和跌倒等非视觉场景的检测与定位。

**💡 创新点**

创新点包括：①跨模态生成式模型——使用条件潜在扩散网络结合视觉投影和自然语言提示来合成无异常的雷达谱；②双分支Vision Transformer局部化器——对比生成谱与真实谱的特征差异，实现空间异常定位；③将场景几何对齐、语义上下文提取与雷达谱生成无缝集成，显著降低误检率并提升定位精度。

**🔧 技术方法**

核心技术：视觉-雷达对齐（雷达视角投影）、语义上下文分类（ResNet-18）、自然语言提示构造、条件潜在扩散生成器（CLIP+U-Net）、双分支ViT局部化器、联合损失（重构、KL、对抗、CLIP一致性）以及多帧投票与熵统计。

**📊 数据集**

使用的数据集包括：①自采藏物数据（7名受试者，8个隐藏位置，约1200帧）；②自采墙后入侵者与跌倒数据（272/多帧/2fps，7面墙）；③公共数据集（TextileNet、HuPR、UWCR、mmCounter、Carry Object）用于评估生成器和局部化器。

**📈 对比分析**

与基线（CVAE、CGAN、MMA‑Carry、RTWLBR、Shen等）比较，取得最高F1≈94%（藏物）/≈92%（墙后入侵）/≈96%（跌倒），平均定位误差≤0.5 m；在不同材质、服装与环境下均保持稳健，明显优于传统单模态或无上下文方法，AUROC提升显著。

**⚠️ 局限性**

局限性：①推理延迟较高（尤其是雷达投影与扩散生成），对实时低功耗设备适配仍需优化；②依赖准确信息对齐与视觉上下文，若相机遮挡或光照极端会影响性能；③对OOD视觉特征的鲁棒性有限，需进一步研究无监督或开放集识别。

---

## 102. REM-CTX: Automated Peer Review via Reinforcement Learning with Auxiliary Context

**arXiv ID:** 2604.00248 | [PDF](https://arxiv.org/pdf/2604.00248v1)

**作者:** Pawin Taechoyotin `[一作]` (University of Colorado Boulder), Daniel E. Acuna `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5069191647)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了REM-CTX，一种基于强化学习的同行评审自动生成框架，利用图像细节与新颖性评估等辅助上下文来提升评审质量；

**💡 创新点**

创新点包括：①提出对应奖励函数（Figure Correspondence Reward Function与Novelty Correspondence Reward Function），直接激励模型在生成过程中引用辅助上下文；②使用Group Relative Policy Optimization (GRPO) 对8B参数语言模型进行强化学习；③构建并公开了跨学科的PeerRTEx、FCRDat、NCRDat三大数据集；

**🔧 技术方法**

技术手段：8B参数大型语言模型（以Qwen3-8B为基线），GRPO强化学习框架，多维度质量奖励，ModernBERT分类器实现对应奖励，思维轨迹格式化奖励；

**📊 数据集**

使用的数据集：PeerRTEx（234篇涵盖计算机、生物、物理科学的完整论文及对应人工评审）；FCRDat（句级图形对应数据集）；NCRDat（句级新颖性对应数据集）；

**📈 对比分析**

对比方法：与六个基线（Vanilla、Structured、Multi‑Agent、MAMORX、Qwen3‑8B、REMOR）在整体评审质量、维度覆盖率以及对应奖励指标上进行评估；实验结果显示REM‑CTX在总体质量上最高，且在对应奖励上明显优于其他模型，尽管在部分维度（如批评性）略低；

**⚠️ 局限性**

局限性：数据集规模相对有限（尤其物理科学样本仅24篇）；对应奖励分类器误差可能导致奖励信号偏差；所有奖励权重均设为统一，未探索最优加权策略；未进行系统偏差（如机构偏差）的审计，存在潜在伦理风险。

---

## 103. SANA I2I: A Text Free Flow Matching Framework for Paired Image to Image Translation with a Case Study in Fetal MRI Artifact Reduction

**arXiv ID:** 2604.00298 | [PDF](https://arxiv.org/pdf/2604.00298v1)

**作者:** Italo Felix Santos `[一作]` (National Laboratory for Scientific Computing), Heron Werner Junior `[通讯]` (Dasa)

**通讯引用:** 2039 | [OpenAlex ID](https://openalex.org/A5031017439)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种文本无关的SANA-I2I框架，用于监督式成对图像到图像的生成，主要应用于胎儿MRI运动伪影去除。

**💡 创新点**

创新点在于完全移除文本条件，只用图像条件学习流匹配模型；保留控制器网络和CFG机制，实现高效、可控的高分辨率图像翻译。

**🔧 技术方法**

采用流匹配（Flow Matching）的SANA模型、ControlNet分支、DC-AE压缩自编码器、混合精度训练、CFG和CAME优化器等技术。

**📊 数据集**

使用合成的胎儿MRI数据（真实无伪影B域与Duffy方法生成的带伪影C域）以及真实带伪影A域进行训练与评估。

**📈 对比分析**

与CycleGAN V2等方法对比，在合成域使用SSIM/MAE评估，在真实域使用FID/KID评估；结果显示在分布指标上显著优于对比方法，且仅需5步或更少步骤即可获得高质量重建。

**⚠️ 局限性**

局限性包括：SSIM/MAE对亮度校正敏感，导致得分不完全反映视觉质量；仅在合成数据上训练，真实世界泛化仍有限；BIS变体在伪影校正中表现较差；缺乏更大规模或多模态验证。

---

## 104. TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving

**arXiv ID:** 2604.00368 | [PDF](https://arxiv.org/pdf/2604.00368v1)

**作者:** Feng Ren `[一作]` (AISoft), Mingxing Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 25774 | [OpenAlex ID](https://openalex.org/A5100621291)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TENT，一个面向分布式LLM推理的声明式切片喷射转移引擎，能够在多种异构互连间动态规划数据路径；

**💡 创新点**

创新点在于把路径选择从静态绑定转为运行时动态调度，使用细粒度切片基于实时遥测的预测调度，并实现双层自愈容错，形成统一的多链路资源池；

**🔧 技术方法**

技术上实现了段抽象与插件化传输后端、基于队列深度与带宽的预测成本模型、动态切片分发、无缝故障重试、低开销数据路径和 CPU+RDMA/NVLink 多路复用；

**📊 数据集**

使用SGLang HiCache的长上下文推理基准（Qwen3‑235B‑A22B‑Instruct‑2507）、Moonshot Checkpoint Engine的模型参数更新基准（GLM‑4.5‑Air 等），以及开放源代码的 TEBench 微基准来评估；

**📈 对比分析**

在H800 HGX集群上与Mooncake TE、NIXL、UCCL‑P2P 对比，KVCache吞吐提升1.36×、P90 TTFT下降26%，模型更新速度提升20‑26%，主机间吞吐提升33%并且P99延迟降低27%，多线程并发下仍保持高吞吐；

**⚠️ 局限性**

局限在于依赖实时遥测和预测模型，极端网络抖动或低带宽场景可能需要重新调参；对跨集群大规模一致性调度支持有限，且仍需应用层配合统一接口；

---

## 105. Epileptic Seizure Detection in Separate Frequency Bands Using Feature Analysis and Graph Convolutional Neural Network (GCN) from Electroencephalogram (EEG) Signals

**arXiv ID:** 2604.00163 | [PDF](https://arxiv.org/pdf/2604.00163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 106. Dynamic Graph Neural Network with Adaptive Features Selection for RGB-D Based Indoor Scene Recognition

**arXiv ID:** 2604.00372 | [PDF](https://arxiv.org/pdf/2604.00372v1)

**作者:** Qiong Liu `[一作]` (Huazhong University of Science and Technology), You Yang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 3922 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种动态图神经网络结合自适应节点选择的RGB‑D室内场景识别方法。

**💡 创新点**

创新点在于：① 用注意力驱动的自适应节点选择筛选关键局部特征；② 通过三层级稀疏动态图结构建模局部特征关系；③ 在图更新过程中学习两模态的不同贡献。

**🔧 技术方法**

技术手段包括：ResNet101 提取 RGB 与 HHA 编码的深度特征；CBAM 关注机制实现节点选择；GNN 进行图聚合与动态更新；跨模态注意力融合最终得到场景分类器。

**📊 数据集**

实验数据集：SUN RGB‑D（19类）与 NYU Depth v2（10类）。

**📈 对比分析**

与多种基线和最新方法对比，SUN 上均类准确率 57.7%，NYU 上 70.4%，均超过所有对比算法。

**⚠️ 局限性**

局限性包括：节点数目和稀疏连接需要手工调参；对 HHA 编码的深度依赖性强；在样本稀缺的类别上仍表现不佳。

---

## 107. Predicting Wave Reflection and Transmission in Heterogeneous Media via Fourier Operator-Based Transformer Modeling

**arXiv ID:** 2604.00132 | [PDF](https://arxiv.org/pdf/2604.00132v1)

**作者:** Zhe Bai `[一作]` (Lawrence Berkeley National Lab), Hans Johansen `[通讯]` (Lawrence Berkeley National Lab)

**通讯引用:** 1655 | [OpenAlex ID](https://openalex.org/A5082013692)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发基于 Fourier Transformer 的机器学习代理模型，用于预测一维 Maxwell 方程中材料界面处的波反射和透射。

**💡 创新点**

结合频域与时空双路径注意力网络，利用 Fourier 变换在潜在空间对波数谱进行精准建模。

**🔧 技术方法**

使用 Transformer、频域嵌入、重叠分块、傅里叶变换以及自回归预测框架。

**📊 数据集**

训练集来自高精度有限体积（FV）仿真，包含不同初始波形和材料光速参数的 200 条样本。

**📈 对比分析**

与 FV 结果对比，模型在 75% 时间步内相对误差低于 10%，频域精度保持与仿真一致，误差随时间线性增长。

**⚠️ 局限性**

主要限制在于高频波数被截断导致材料界面处误差放大，以及对非线性材料和高维情形的适应性待验证。

---

## 108. Play-Testing REMind: Evaluating an Educational Robot-Mediated Role-Play Game

**arXiv ID:** 2604.00300 | [PDF](https://arxiv.org/pdf/2604.00300v1)

**作者:** Elaheh Sanoubari `[一作]` (University of Waterloo), Kerstin Dautenhahn `[通讯]` (University of Waterloo)

**通讯引用:** 26104 | [OpenAlex ID](https://openalex.org/A5059371010)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并试玩了REMind，一款利用社交机器人进行反欺凌旁观者干预的角色扮演游戏；在18名9–10岁儿童中进行混合方法评估。

**💡 创新点**

将机器人介导的戏剧（RMAD）与角色扮演结合，提供可让儿童通过“木偶模式”亲身演练并录制自己的面部表情和语音来对抗欺凌的创新体验。

**🔧 技术方法**

使用Furhat社交机器人平台、StorySync脚本同步工具、iOS Live Link Face捕捉面部动作、Wizard‑of‑Oz辅导，构建五阶段交互式叙事。

**📊 数据集**

未使用公开数据集，评估采用改编自Pöyhönen等人的自评量表（自我效能、同理心、结果预期）以及访谈和现场观察记录的数据。

**📈 对比分析**

通过配对样本t检验和Spearman相关检验评估学习效果；自我效能显著提升（Cohen d=0.74），欺凌者结果预期校准显著，受害者结果预期提升未显著，情感提升趋于显著；定性分析提供丰富的情景反思。

**⚠️ 局限性**

样本量有限、缺乏对照组、Wizard‑of‑Oz操作可能产生偏差、仅评估短期反应、未验证长期行为改变、系统半自主性与两人辅导模式限制可扩展性。

---

## 109. QUEST: A robust attention formulation using query-modulated spherical attention

**arXiv ID:** 2604.00199 | [PDF](https://arxiv.org/pdf/2604.00199v1)

**作者:** Hariprasath Govindarajan `[一作]` (Linköping University), Fredrik Lindsten `[通讯]` (Linköping University)

**通讯引用:** 2257 | [OpenAlex ID](https://openalex.org/A5045048407)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为 QUEST（Query-modulated Spherical Attention）的注意力机制，用于替代传统的软最大缩放点积注意力；

**💡 创新点**

核心创新在于只对键进行 ℓ₂ 归一化，将注意力的相似度转为余弦相似度，同时保留查询向量的范数以控制每个 token 的注意力锐度，既抑制注意力 logit 失控，又保持注意力表达的灵活性；

**🔧 技术方法**

实现上采用标准 Transformer 结构，只需将 Softmax 前的键矩阵替换为归一化后的键即可；在实验中还结合了多种数据增强、层归一化、随机深度等训练技巧；

**📊 数据集**

在视觉领域使用 ImageNet‑1K、ImageNet‑v2、ImageNet‑ReaL、ImageNet‑Corrupted、ImageNet‑Adversarial 等；在 NLP 用 WikiText‑103、Transformer‑XL；在时间序列用 UEA 多变量数据集；在图结构用 GraphGPS 基准；在点云、分割任务也有实验；

**📈 对比分析**

与标准注意力、QNorm、QKNorm 等变体以及椭圆注意力等做对比。结果显示 QUEST 在训练稳定性上优于标准注意力，且在大多数任务上取得更高或相近的精度，同时在抗噪声、对抗攻击、分割 mIoU 等指标上表现更佳；

**⚠️ 局限性**

限制在于：在小模型中性能提升不明显甚至略逊于标准注意力；目前仅在单头或少数头实验中验证，尚未深入研究多头耦合和与线性注意力的兼容性；此外对非视觉领域的更细粒度评估与基准仍待扩展。

---

## 110. DreamControl-v2: Simpler and Scalable Autonomous Humanoid Skills via Trainable Guided Diffusion Priors

**arXiv ID:** 2604.00202 | [PDF](https://arxiv.org/pdf/2604.00202v1)

**作者:** Sudarshan Harithas `[一作]` (General Robotics), Jonathan Chung-Kuan Huang `[通讯]` (General Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

构建了机器人空间的运动扩散模型，实现了全自动化的人机交互技能学习。

**💡 创新点**

直接在机器人动作空间训练扩散模型，消除后置重定位与手工提示，扩充数据源提升多样性。

**🔧 技术方法**

使用扩散模型（OmniControl 结构）结合 ControlNet 控制网络与文本/空间约束。

**📊 数据集**

融合 AMASS、HumanML3D、GRAB、Nymeria 以及 OmniRetarget 的机器人轨迹数据。

**📈 对比分析**

与原 DreamControl 及零射手 OmniControl 对比，实验显示 FID 与成功率提升至 80% 以上，RL 成功率显著提高。

**⚠️ 局限性**

仍受限于训练时对大规模多任务场景的实时性，且复杂环境下仍需改进物理一致性与实时推理速度。

---

## 111. EvolveTool-Bench: Evaluating the Quality of LLM-Generated Tool Libraries as Software Artifacts

**arXiv ID:** 2604.00392 | [PDF](https://arxiv.org/pdf/2604.00392v1)

**作者:** Alibek T. Kaliyev `[一作]` (University of Texas at Austin), Artem Maryanskyy `[通讯]` (Uber Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了EvolveTool-Bench，评估LLM自生成工具库的代码质量和软件工程指标；

**💡 创新点**

首次将工具库视为软件产物，定义了工具质量分数(TQS)和库健康指标，并通过诊断式任务序列测评重用、冗余、回归等；

**🔧 技术方法**

利用LLM-as-Judge奖励、沙盒执行、隐藏单元测试与对抗测试、代码重用与A/B测试机制，实现代码级演化与策略级演化对比；

**📊 数据集**

构建了包含三大领域（专有数据格式、API编排、数值计算）的99个任务数据集，分别为不同工具生成与使用场景；

**📈 对比分析**

对四种系统（无演化、ARISE、EvoSkill、One-Shot）在Claude Sonnet 4和Haiku 4.5上进行比较，ARISE在EvolveTool Score（ETS）上最高（0.603/0.612），但任务完成率与其他系统相近；

**⚠️ 局限性**

局限性包括仅使用两款模型、单一云服务提供商、缺乏统计显著性检验、自动化检测可能漏报缺陷，以及对EvoSkill基线实现的近似。

---

## 112. Hierarchical Motion Planning and Control under Unknown Nonlinear Dynamics via Predicted Reachability

**arXiv ID:** 2604.00320 | [PDF](https://arxiv.org/pdf/2604.00320v1)

**作者:** Zhiquan Zhang `[一作]` (University of Illinois Urbana-Champaign), Melkior Ornik `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 355 | [OpenAlex ID](https://openalex.org/A5070897457)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种层次化运动规划与控制框架，能够在未知非线性动力学下进行在线系统辨识、预测可达性、非均匀自适应状态空间划分、构建可达性保证图，并在图上搜索路径同时合成局部仿射反馈控制器，实现从起点到目标的导航。

**💡 创新点**

创新点包括：① 仅在任务相关区域自适应细化状态空间；② 针对未知动力学给出鲁棒预测可达性判定并用信息熵权重平衡探索与利用；③ 对欠驱动系统引入松弛可达性条件并实现；④ 结合可达性保证图与在线辨识，实现正式可达性保证。

**🔧 技术方法**

采用的主要技术有：仿射动力学局部线性化与最小二乘辨识；可达性控制理论的预测可达性分析；线性规划求解可达性约束；信息熵与信息增益计算权重；Dijkstra 图搜索；终点内的 CLF‑CBF 控制。

**📊 数据集**

实验仅使用仿真中的两种移动机器人模型：全驱动的 Mecanum 车与欠驱动的单轮车，环境为未知地形/扰动，未引用公开数据集。

**📈 对比分析**

与之前基线方法对比，非均匀划分使分块数减少 68%，在全驱动与欠驱动场景下均能找到可行路径；仿真展示了探索–利用平衡、时间/距离性能以及信息增益等指标，整体性能显著提升。

**⚠️ 局限性**

局限性包括：① 需高精度局部仿射辨识，误差会导致控制失效；② 欠驱动系统仍需手工松弛；③ 无全局可达性保证；④ 终点 CLF‑CBF 控制缺乏理论收敛保证；⑤ 计算复杂度随维度增长仍为瓶颈。

---

## 113. Excite, Attend and Segment (EASe): Domain-Agnostic Fine-Grained Mask Discovery with Feature Calibration and Self-Supervised Upsampling

**arXiv ID:** 2604.00276 | [PDF](https://arxiv.org/pdf/2604.00276v1)

**作者:** Deepank Singh `[一作]` (University of Houston), Vedhus Hoskere `[通讯]` (University of Houston)

**通讯引用:** 2404 | [OpenAlex ID](https://openalex.org/A5041968673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了无监督、域无关的细粒度语义分割框架 EASe。

**💡 创新点**

创新点包括自监督通道校准上采样 SAUCE 与训练自由聚合器 CAFE，利用 SAUCE 的注意力作为分组信号。

**🔧 技术方法**

技术包括基于 DINOv3 Vision Transformer 的基础模型、跨层注意力上采样、SE 通道激励、K-近邻/原型聚类、层次合并等。

**📊 数据集**

使用多种公开数据集：Pascal VOC、COCO-Object、COCO-Stuff、Cityscapes、ADE20K、PartImageNet、KITTI、OmniCrack30k、Roof Subassembly Damage Detection。

**📈 对比分析**

与现有无监督 SOTA 方法比较，EASe 在 8/9 个基准上均实现显著提升，平均 mIoU 提升约 +3.7，复杂形态数据提升高达 +30.3。

**⚠️ 局限性**

局限在于对极小对象或极细纹理仍可能受限，且需要先行自监督训练 SAUCE 计算量较大，推理速度在极大规模图像上受限。

---

## 114. Neural-Assisted in-Motion Self-Heading Alignment

**arXiv ID:** 2604.00168 | [PDF](https://arxiv.org/pdf/2604.00168v1)

**作者:** Zeev Yampolsky `[一作]` (University of Haifa), Itzik Klein `[通讯]` (University of Haifa)

**通讯引用:** 2580 | [OpenAlex ID](https://openalex.org/A5012718881)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种基于多头二维卷积神经网络的端到端模型——HeadingNet，用于在航行中快速估计自动船舶的初始航向角。

**💡 创新点**

创新点在于将物理分解的对齐思路映射为多头CNN结构，直接从惯性测量和GNSS辅助数据中学习航向估计，避免传统方法对运动激励和积分时间的严格依赖。

**🔧 技术方法**

使用技术包括二维卷积网络、多头结构、LeakyReLU与Tanh激活、dropout正则化、AdamW优化器、周期性均方误差（CMSE）损失。

**📊 数据集**

使用了实测的自主水面船（ASV）在不同海况下的五天数据集，包含100 Hz IMU（角速度、加速度）和5 Hz GNSS‑RTK（定位、重力、传输率）信息。

**📈 对比分析**

与传统自对齐方法（DVA、OBA的I‑DVA、A‑DVA、I‑OBA、A‑OBA）对比，HeadingNet在所有对齐时长下均显著更优：平均绝对误差提升53%，对齐时间缩短多达67%，10 s对齐时误差低于5°。

**⚠️ 局限性**

局限性包括：不同对齐时长需要针对性调整网络骨干；模型训练依赖GNSS‑RTK标注，缺乏GNSS信号时难以直接迁移；作为数据驱动方法，对训练外的极端海况或硬件噪声的泛化能力有限。

---

## 115. Sit-to-Stand Transitions Detection and Duration Measurement Using Smart Lacelock Sensor

**arXiv ID:** 2604.00175 | [PDF](https://arxiv.org/pdf/2604.00175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models

**arXiv ID:** 2604.00375 | [PDF](https://arxiv.org/pdf/2604.00375v1)

**作者:** Liancheng Fang `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 135315 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于独立Metropolis–Hastings采样的全局温度调节解码策略，用以解决Diffusion LLM在质量与探索性之间的矛盾。

**💡 创新点**

创新点在于：① 通过全局能量正则化推导出最优的幂分布目标；② 推导出该目标下的精确条件分布并用均值场近似实现可计算的lookahead校正；③ 在此基础上设计了批量并行的IMH采样器，实现高采样质量与高探索性的 Pareto 前沿。

**🔧 技术方法**

使用了Diffusion语言模型的mask预测框架、均值场近似、Independent Metropolis–Hastings MCMC、全局温度调节（幂分布）以及批量lookahead计算。

**📊 数据集**

在数学推理任务（MATH500、AIME 2024/25）和代码生成任务（HumanEval、MBPP）上评估，使用LLaDA-8B-Instruct和WeDLM-8B两个不同的dLLM模型。

**📈 对比分析**

与随机解码、低置信度remasking（confidence/entropy/margin）等基线进行比较。实验显示IMH在Pass@1与Pass@k上均超越所有基线，尤其在AIME高难度子集与Pass@k扩展性上表现最为显著，形成新的质量-多样性 Pareto 前沿。

**⚠️ 局限性**

局限性包括：① 对均值场近似的依赖可能在极端长序列或高度相关的上下文中失效；② 需要额外的前向计算来估计lookahead校正，导致一定的推理开销；③ 仅在当前dLLM框架下验证，尚未证明对更大规模模型或不同解码策略的泛化。

---

## 117. Generalizable Dense Reward for Long-Horizon Robotic Tasks

**arXiv ID:** 2604.00055 | [PDF](https://arxiv.org/pdf/2604.00055v1)

**作者:** Silong Yong `[一作]` (Carnegie Mellon University), Yesh Dattatreya `[通讯]` (Amazon Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Vision‑Language Long‑horizon Reward (VLLR)，一种在长时序机器人任务中对预训练基础策略进行强化学习微调的奖励框架；

**💡 创新点**

其创新点在于将 LLM 对任务进行语义拆分、VLM 对视觉进度进行评估、以及基于策略自身置信度（self‑certainty）的自我奖励相结合，实现了既能提供子目标层级监督又能给出稠密的即时反馈，且仅在初始化阶段调用 VLM 以降低计算成本；

**🔧 技术方法**

采用的技术包括：LLM（如 Claude‑3.7‑Sonnet）进行任务拆解、VLM（Nova Pro）进行进度估计与价值函数初始化、policy self‑certainty 作为 intrinsic reward、PPO 进行微调、以及对进度估计进行滑动窗口去噪；

**📊 数据集**

在 191,568 个 ProcThor 生成的房屋环境上构建的 CHORES benchmark 进行实验，涵盖了四个在预训练数据集内的任务（Fetch、Pick‑up、RoomVisit、ObjectNav）以及两个 OOD 任务（Object Navigation with Affordance/Relative Attribute）；

**📈 对比分析**

与 SOTA 的 FLaRe、SPOC、PIRLNav、JSRL 等方法相比，VLLR 在 in‑distribution 任务上比 SOTA 提升约 5% 的绝对成功率，在 OOD 任务上提升约 10%；此外，VLM 初始化显著提升任务完成效率，self‑certainty 加速收敛并提升成功率；

**⚠️ 局限性**

主要局限在于仅在 CHORES benchmark 上验证，且仅针对离散动作空间；对连续动作的通用性和在更广泛场景下的可迁移性仍待进一步研究。

---

## 118. Reclaiming Idle CPU Cycles on Kubernetes: Sparse-Domain Multiplexing for Concurrent MPI-CFD Simulations

**arXiv ID:** 2604.00377 | [PDF](https://arxiv.org/pdf/2604.00377v1)

**作者:** Tianfang Xie `[一作]` (Purdue University), Tianfang Xie `[通讯]` (Purdue University)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5113389503)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套在Kubernetes集群上并行CFD仿真任务的多路复用框架，利用MPI的同步阻塞处浪费的CPU资源，将多份仿真任务共置在同一节点上以提升集群吞吐量；

**💡 创新点**

创新点在于：①基于PMPI的每个MPI进程的CPU占空比测量实现动态、细粒度的CPU请求分配；②结合Kubernetes的In‑Place Pod Vertical Scaling（KEP‑1287）在运行时无重启地调整CPU资源；③提出单参数的吞吐量预测模型与Pareto分析，为不同共置密度提供最佳配置；④实现了自动化的动态控制器，完整覆盖从剖面、缩放到任务调度的流程；

**🔧 技术方法**

主要技术包括MPI PMPI拦截、Linux CFS权重调度、Kubernetes请求‑无限制（Burstable）QoS、In‑Place Pod Vertical Scaling、TCP仅传输层隔离、以及自研的动态控制器；

**📊 数据集**

实验使用OpenFOAM 10求解NACA 0012机翼在马赫0.72下的可压缩稳态流场，网格498 834个六面体，分16个MPI进程，在12个AWS c5.2xlarge节点（8 vCPU/节点）上运行；

**📈 对比分析**

对比方法为单任务与多任务（N=1~5）并行执行，衡量总完成时间、吞吐量（simulations/小时）及单案例耗时的降幅。实验显示：两任务共置可获得1.77×吞吐提升，五任务可达3.74×，最佳折中点为N=3；动态控制器实现4任务时可达3.25×吞吐，成本相对单任务降低约62%；

**⚠️ 局限性**

局限性包括：仅验证单一CFD求解器和单一网格规模，缺乏对更高负载、不同物理模型和多核/NUMA节点的评估；对高占空比任务（如化学反应）可能不适用；未实现完整的故障恢复和多租户调度策略；以及对内存带宽冲突和硬件计数器监控的支持有限。

---

## 119. Engineering Fully Dynamic Convex Hulls

**arXiv ID:** 2604.00271 | [PDF](https://arxiv.org/pdf/2604.00271v1)

**作者:** Ivor van der Hoog `[一作]` (IT University of Copenhagen), Eva Rotenberg `[通讯]` (IT University of Copenhagen)

**通讯引用:** 323 | [OpenAlex ID](https://openalex.org/A5004695759)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种新的完全动态凸包维护算法，能够在增删点操作时保持凸包并支持点位置信息查询。

**💡 创新点**

创新点在于将对数方法与基于删除的凸包数据结构相结合，获得 O(log n loglog n) 的摊销更新时间，并实现了在不牺牲鲁棒性的前提下支持非可分解的点定位查询。

**🔧 技术方法**

采用对数方法的桶式分层合并、删除专用的降序凸包树、离线构造的线性时间结构以及贪婪重构策略等技术。

**📊 数据集**

在实验中使用了三类真实数据（3D Mammal、Tiger、Cluster）以及四类合成分布（Box、Bell、Disk、Circle），并在不同规模和更新比例下进行测试。

**📈 对比分析**

与基线的全重建算法和多种公开实现（如 Overmars–van Leeuwen、Brodal–Jacob 等）在更新、查询和混合工作负载上进行对比，结果显示在大多数数据集和更新占比高的场景下，更新速度显著快于对手，查询时间虽略慢但仍可接受；在稀疏凸包场景下，基线算法甚至更优。

**⚠️ 局限性**

主要局限在于查询时间为 O(log² n)，在极稠密凸包（如 Circle）或高查询比例场景下性能下降；实现对数层桶合并与删除逻辑复杂，难以进一步优化。

---

## 120. Dynin-Omni: Omnimodal Unified Large Diffusion Language Model

**arXiv ID:** 2604.00007 | [PDF](https://arxiv.org/pdf/2604.00007v1)

**作者:** Jaeik Kim `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5024989829)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个 8B 参数、基于掩码扩散的全新通用多模态基础模型，能够在同一网络结构中完成文本、图像、语音的理解与生成，以及视频的理解。

**💡 创新点**

创新点主要包括：① 将多模态任务统一映射到同一离散 token 空间，并用掩码扩散代替自回归，实现真正的 native 统一；② 设计了多阶段训练流程（模态对齐→模态解耦合模型合并→能力放大），有效缓解模态扩展导致的灾难性遗忘；③ 引入 scheduled padding 学习和模态解耦合模型合并策略，让模型在扩展新模态时保持旧模态性能；④ 全面使用轻量级 detokenizer，避免外部生成器，降低推理复杂度。

**🔧 技术方法**

核心技术包括：掩码扩散（Masked Diffusion）框架、共享 Transformer backbone、统一 tokenizer 与 detokenizer、模态解耦合模型合并、scheduled padding 学习、并行多模态推理策略（文本/语音块式、图像全并行）。

**📊 数据集**

使用的主要数据集：
• Stage 1：WebVid‑10M（视频-文本），GigaSpeech、LibriSpeech、CommonVoice（语音识别/合成）。
• Stage 2：Evol‑Instruct、Magpie‑Pro（文本对话）、Open‑Platypus、OpenR1‑Math、OpenHermes‑2.5（推理）、Cambrian10M、JourneyDB、FLUX‑Reason‑6M、PickaPic、UltraEdit、HQEdit、Pico‑Banana‑400K（图像生成/编辑）、LLaVA‑Video‑178K、OpenVid1M、内部合成视频（视频理解）。
• Stage 3：Llama‑Nemotron、OpenR1‑Math‑220K、Mixture‑of‑Thoughts、OpenMathReasoning（推理）及使用 GLM‑4.5‑Air、GPT‑OSS‑120B、Qwen3‑Next‑80B‑A3B‑Thinking 生成的高质量推理数据；FLUX、Z‑Image、FLUX‑Kontext、Qwen‑Image 等图像生成数据；ShareGPT4Video、内部视频 QA 数据；Kokoro TTS 生成的 300K 长语音样本。

**📈 对比分析**

对比了约 7–8B 规模的专家模型、感知中心的多模态模型（Qwen2.5‑Omni、Baichuan‑Omni、OmniVinci）以及统一理解与生成模型（BAGEL、MMaDA、Lumina‑DiMOO、HyperCLOVAX‑8B‑Omni、NExT‑OMNI）。实验表明：
• 在文本推理任务（MMLU、ARC‑C、GSM8K、MATH、GPQA）上与单模态 LLM 旗鼓相当，且在大部分任务上优于现有扩散 LLM。 
• 在多模态理解（图像/视频）上，与专家模型相差 3–10% 左右，且在多模态基准（POPE、MME‑P、GQA、MMMU、MMBench、ActNet‑QA、MVBench、TempCompass、VideoMME）中保持稳定且接近最优。 
• 在图像生成与编辑上，GenEval、DPG‑Bench 评分接近专用生成模型，仅在细节/计数等指标上略低；但在无需外部生成器的前提下实现了同等质量。 
• 在语音识别与合成上，WER 与 HyperCLOVAX‑8B‑Omni、NExT‑OMNI 相近，显著优于早期统一模型。 
总体而言，模型在推理、理解、生成、语音等多模态任务上实现了强劲且平衡的表现。

**⚠️ 局限性**

局限性主要包括：
① 与专用生成模型相比，图像/语音生成的细节质量仍略逊；
② 训练与推理仍需要较高算力（8B 参数 + 多模态 tokenizer），对资源有限的场景不友好；
③ 采用统一 token 需要在每个模态上设计合适的离散化方式，存在潜在的分辨率/音频采样率限制；
④ 当前模型仅覆盖文本、图像、语音、视频四种模态，未扩展至更复杂的交互式多模态（如 3D、AR/VR、传感器数据）。

---

## 121. Hybrid Energy-Based Models for Physical AI: Provably Stable Identification of Port-Hamiltonian Dynamics

**arXiv ID:** 2604.00277 | [PDF](https://arxiv.org/pdf/2604.00277v1)

**作者:** Simone Betteti `[一作]` (Italian Institute of Artificial Intelligence for Industry), Luca Laurenti `[通讯]` (Italian Institute of Artificial Intelligence for Industry)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于能量的混合模型（Hybrid EBM）用于系统辨识，能够在保证能量衰减和吸收不变集的前提下逼近非线性动力学。

**💡 创新点**

创新点包括：1) 利用Clarke导数将EBM稳定性分析扩展到非光滑激活函数；2) 引入仅对第一隐藏层激活进行约束、其余层自由的混合架构，实现表达能力与稳定性共存；3) 将吸收不变性推广到端到端的端口-Hamiltonian动力学；4) 在非欧氏度量下验证。

**🔧 技术方法**

使用技术：能量函数梯度流、Clarke广义导数、吸收不变集理论、端口-Hamiltonian变换、对称块三角权重矩阵、软最大/多项式激活、AdamW优化、MSE与短期滚动误差联合损失。

**📊 数据集**

数据集：通过数值积分在二维空间[-2,2]^2采样2000个初始点生成轨迹，分为1600训练和400测试，分别用于多井势能和环形势能的两组实验。

**📈 对比分析**

对比方法：将学习得到的能量场与真势能、向量场以及轨迹进行可视化对比；量化指标为训练与测试误差（f_m: 0.011/0.002，f_e: 0.047/0.042），表明学习误差在训练后保持低且在测试数据上与训练误差相近，显示模型具有良好的泛化和稳定性。

**⚠️ 局限性**

限制：1) 需要手工预设第一隐藏层的激活上界，影响模型灵活性；2) 吸收不变半径 r 的计算在实际网络中往往不可解析，需通过经验估计，可能过于保守；3) 目前仅在二维仿真中验证，尚未证明在更高维、带控制输入或真实物理系统上的可推广性；4) 需要先验的非欧氏度量 Q(x) 的可学习性与稳定性分析尚未完全覆盖。

---

## 122. Softmax gradient policy for variance minimization and risk-averse multi armed bandits

**arXiv ID:** 2604.00241 | [PDF](https://arxiv.org/pdf/2604.00241v1)

**作者:** Gabriel Turinici `[一作]` `[通讯]` (Université Paris Dauphine - PSL), Gabriel Turinici (Université Paris Dauphine - PSL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于 softmax 参数化的策略梯度算法，用于多臂赌博机中的方差最小化与风险规避决策。

**💡 创新点**

创新点在于通过成对采样构造无偏的方差估计，并在自然条件下证明了算法收敛性。

**🔧 技术方法**

使用 softmax 策略梯度、随机近似（Robbins‑Monro）、mini‑batch 采样以及理论分析技术。

**📊 数据集**

实验采用合成数据，包括 2、10 臂及随机分离的 10 臂情形，均为正态分布或均值相同方差不同的离散分布。

**📈 对比分析**

通过与理论最优策略对比，展示了均值回报趋于零、最优臂选择频率趋于 100% 或高于 70% 的良好性能。

**⚠️ 局限性**

局限在于需要奖励有界或可截断、对重尾分布收敛性不充分、以及在真实大规模场景中样本效率待提升。

---

## 123. MRReP: Mixed Reality-based Hand-drawn Reference Path Editing Interface for Mobile Robot Navigation

**arXiv ID:** 2604.00059 | [PDF](https://arxiv.org/pdf/2604.00059v1)

**作者:** Takumi Taki `[一作]` (University of Osaka), Yuki Uranishi `[通讯]` (University of Osaka)

**通讯引用:** 873 | [OpenAlex ID](https://openalex.org/A5056350914)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发并实现了 MRReP，一个基于混合现实的接口，允许用户在物理环境中通过手势绘制手绘参考路径（HRP），并将该路径转换为机器人导航栈可用的全局路径，从而实现自主移动机器人在共享空间中的精准路径规划。

**💡 创新点**

创新点包括：① 直接在真实地面绘制路径，消除二维屏幕与三维物理空间之间的映射误差；② 设计了自定义的 Hand-Drawn Reference Path Planner，将手绘点序列无缝转换为 ROS2 Navigation2 所需的全局路径；③ 通过 HoloLens2、Unity、ROS2 与 kachaka API 的多链路通信，实现无需固定外部传感器即可完成路径规划与执行。

**🔧 技术方法**

采用的技术与工具有：混合现实设备 HoloLens2、Unity3D、ROS2（Navigation2）与自定义节点、kchaka-api（gRPC）、ROS‑TCP‑Connector/Endpoint、Vuforia Engine 进行坐标对齐、Pure Pursuit 控制器以及手势识别与手绘点序列管理。

**📊 数据集**

实验使用现场实验数据：在实验场景中用胶带标记直线与多段45°转弯的目标路径作为 Ground Truth，参与者分别在 MR 与传统 2D（鼠标）界面绘制对应路径；未使用公开数据集，全部数据来自自建实验。

**📈 对比分析**

通过 within-subject 对比实验评估 MRReP 与 2D 界面，在路径准确率（精度、召回、F1）、绘图次数、完成时间、路径稳定性以及主观可用性（SUS）和工作负荷（NASA‑TLX）等指标进行比较。结果显示：MRReP 在路径准确率和路径稳定性上显著优于 2D（尤其是复杂路径），SUS 得分更高、NASA‑TLX 负荷更低；绘图次数无显著差异，完成时间略长，但差异不大。

**⚠️ 局限性**

局限性包括：① MR 系统在手势交互中易产生手臂疲劳和误识别，导致一定的工作负荷；② 完成时间略长，可能与物理移动和手势学习曲线有关；③ 实验仅在小规模静态室内环境下进行，未验证在动态或大规模场景、多机器人或复杂障碍环境中的表现；④ 长期使用的舒适性和可靠性尚未评估。

---

## 124. Human-in-the-Loop Control of Objective Drift in LLM-Assisted Computer Science Education

**arXiv ID:** 2604.00281 | [PDF](https://arxiv.org/pdf/2604.00281v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 125. Approximating Gains-from-Trade in Matching Markets

**arXiv ID:** 2604.00129 | [PDF](https://arxiv.org/pdf/2604.00129v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Kangning Wang `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在通用的两边匹配市场（允许任意下闭约束）中，提出了一个随机化机制（在Generalized Sellers-Offering Mechanism和Generalized Buyers-Offering Mechanism之间切换），证明其能在期望收益上取得对第一最佳GFT的1/3.15倍常数近似；

**💡 创新点**

首次把双边交易的常数近似扩展到更一般的两边匹配市场，并通过Meta-Auction分解和cap-monotonicity技术实现了从整体到双边子问题的可行转化，解决了此前仅在完全图或双向拍卖中的开创性问题；

**🔧 技术方法**

使用了可实现性（Myerson's lemma）与单参数机制的单调性证明、Meta-Auction框架、最大权匹配的稳定性、双边贸易的常数近似分析、以及新的cap-monotonicity和分解论；

**📊 数据集**

本文为理论研究，无需实验数据集；

**📈 对比分析**

相较于过去仅针对单边或双向拍卖的常数近似，本文在更宽泛的下闭约束下实现1/3.15的近似；并进一步推导出在多维单一需求买家情况下的1/6.3近似；

**⚠️ 局限性**

仅适用于独立分布且单维的价值/成本；未讨论计算复杂度与实现细节；常数1/3.15并非最优，仍有进一步优化空间；

---

## 126. Q-Mask: Query-driven Causal Masks for Text Anchoring in OCR-Oriented Vision-Language Models

**arXiv ID:** 2604.00161 | [PDF](https://arxiv.org/pdf/2604.00161v1)

**作者:** Longwei Xu `[一作]` (Xiaomi Inc), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入TABench评估OCR文本锚定位，并提出Q-Mask框架通过因果查询驱动的掩码解码实现精准文本–区域对齐。

**💡 创新点**

① 将文本锚定位作为基准任务；② 设计因果查询驱动的掩码解码器，保证只基于图像与查询生成空间先验；③ 构造大规模TextAnchor-26M数据并引入SPI噪声增强。

**🔧 技术方法**

多模态大语言模型+视觉编码器、因果查询驱动掩码解码器（CQMD）、掩码监督损失、两阶段训练、随机噪声注入SPI、de-stylized mask渲染。

**📊 数据集**

TABench（5450问答，970图）、TextAnchor-26M（26.7M图文+mask）、公开文本数据源（HierText、SVRD、ICDAR2015、CDLA、Synthetic、VQA with causal mask）。

**📈 对比分析**

在TABench中，Q-Mask-3B相较于通用VLMs与OCR专用模型在R2T Acc 提升约38%，T2R F1 提升约45%；在TextVQA、InfoVQA等基准上提升9+分，参数规模与大型模型相当。

**⚠️ 局限性**

仍受限于多语种覆盖、长文档定位、极端视觉噪声/遮挡不稳健，且需要大量空间先验标注，模型对非标准文本（公式、表格）支持有限。

---

## 127. Inference-Aware & Privacy-Preserving Deletion in Databases

**arXiv ID:** 2604.00326 | [PDF](https://arxiv.org/pdf/2604.00326v1)

**作者:** Vishal Chakraborty `[一作]` (University of California), Sarvesh Pandey `[通讯]` (Banaras Hindu University)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5008354454)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文从推理泄露的角度重新审视数据库删除操作，阐明逻辑删除、物理删除与语义删除之间的差异，构建了包含残留状态与删除模式两个泄露通道的语义删除设计空间，并在此基础上提出了四个关键的研究挑战。

**💡 创新点**

创新点在于：①将删除视为推理控制问题，引入残留状态泄露与删除模式泄露两条通道；②提出“语义退回”概念，即限制删除后对用户推理的增量；③区分“被覆盖”与“松弛”两种语义保证模型；④系统性梳理并定义四大挑战（R1‑R4），为未来实现可验证、可解释的删除机制提供理论框架。

**🔧 技术方法**

核心技术主要是理论分析与信息论泄漏度量：使用依赖模型（Σ）描述可推理路径，定义先验后验更新、泄漏度量 Leak(V,P;Σ)，并讨论模型相对与松弛保证；同时结合数据库系统的物理维护流程、MVCC、LSM 等实现细节，讨论时间与维护日志的泄漏通道。

**📊 数据集**

论文没有使用具体实验数据集，而是通过示例（员工记录表及其依赖）说明概念与泄漏通道，聚焦理论与设计。

**📈 对比分析**

由于缺乏实验实现，本文未进行性能比较；其贡献主要是概念框架与挑战阐述，未给出与现有系统的性能对比。

**⚠️ 局限性**

局限性包括：①缺乏可落地的实现与评估，未验证所提模型在实际数据库中的可行性；②依赖模型仍以二元形式出现，未处理权重/概率依赖；③未给出动态数据漂移下的自适应机制；④实现可组合、可审计的删除证明仍是开放问题。

---

## 128. When Career Data Runs Out: Structured Feature Engineering and Signal Limits for Founder Success Prediction

**arXiv ID:** 2604.00339 | [PDF](https://arxiv.org/pdf/2604.00339v1)

**作者:** Yagiz Ihlamur `[一作]` `[通讯]` (Amazon), Yagiz Ihlamur (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对VCBench founder career数据直接解析JSON，工程28个结构化特征，并结合规则层与XGBoost提升模型性能。

**💡 创新点**

首次系统性对结构化JSON特征与LLM文本提取进行对照实验，揭示信息上限并量化LLM冗余。

**🔧 技术方法**

特征工程、决策树桩（XGBoost）、规则层、Optuna超参优化、Claude Haiku提取文本特征。

**📊 数据集**

公开的VCBench 4,500条创业者职业档案，包含9%成功率。

**📈 对比分析**

与零样本LLM基线、规则层、现有基准(随机规则森林、Policy Induction)对比，Val F0.5提升至0.3030，私测F0.5≈0.281，性能位于中间层。

**⚠️ 局限性**

信息上限约CV≈0.25，仅能通过结构化职业数据预测，缺乏公司、网络、市场等关键信号。

---

## 129. Hierarchical Discrete Flow Matching for Graph Generation

**arXiv ID:** 2604.00236 | [PDF](https://arxiv.org/pdf/2604.00236v1)

**作者:** Yoann Boget `[一作]` (University of Geneva), Alexandros Kalousis `[通讯]` (Geneva School for Business administration HES-SO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于离散流匹配的层次化图生成框架，利用稀疏化的聚类与展开策略，显著降低节点对数和推理步骤。

**💡 创新点**

创新点：① 设计 D‑Min 聚类算法，以最小化展开图稠密度；② 将离散流匹配与层次化结构结合，实现低 NFE 的高效生成；③ 支持以真实展开图或谱信息为条件的结构化生成。

**🔧 技术方法**

技术手段：图神经网络驱动的软聚类、离散流匹配 (Discrete Flow Matching)、稀疏图卷积、straight‑through 估计、条件化采样与谱约束。

**📊 数据集**

使用数据集：分子数据集 QM9 (含氢)、MoleculeNet、合成 SBM20k（20k 条图）、社交网络数据集 Reddit12k、Zinc250k 等。

**📈 对比分析**

与 DiGress、SparseDiff、SID、EDGE、Grum、GraphBFN、DeFog 等基线在 FCD、NSPDK、MMD、有效率等指标上对比，HDFM 在大多数指标上实现更低误差、更快生成速度（NFE 较低），同时保持高有效率。

**⚠️ 局限性**

局限性：在数据量极少的场景下粗层学习仍具挑战；目前仅评估单一属性，缺少多属性或多模态验证；对极大图规模的进一步扩展仍需研究。

---

## 130. Is One Token All It Takes? Graph Pooling Tokens for LLM-based GraphQA

**arXiv ID:** 2604.00342 | [PDF](https://arxiv.org/pdf/2604.00342v1)

**作者:** Ankit Grover `[一作]` (KTH Royal Institute of Technology), Sarunas Girdzijauskas `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1245 | [OpenAlex ID](https://openalex.org/A5041411651)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究如何将图结构信息压缩为多重软标记，以提升大型语言模型在图问答中的推理能力，避免单标记压缩过度或全文图文本化导致的上下文瓶颈与结构噪声；并系统评估了多种层次图池化方法（Top‑k、SAGPool、DiffPool、MinCutPool、VNPool）在此任务中的表现；同时提出LoRA适配器作为训练稳定器，并引入FandE指标评估基准数据集的结构与特征冗余。

**💡 创新点**

创新点包括：①提出多标记层次图池化作为单标记压缩与全文文本化的中间方案；②系统化比较多种池化算子并揭示其稳定性与性能权衡；③证明LoRA适配器能稳定训练复杂池化（尤其是VNPool）；④将Virtual Node Pooling与Perceiver‑IO等注意力机制对齐，并通过FandE指标量化数据集冗余。

**🔧 技术方法**

使用的技术包括：Graph Transformer / TransformerConv GNN 进行图编码；多标记投影层（可学习的soft token生成）；LoRA参数高效适配器（对LLM的注意力层进行低秩调整）；PCST检索子图；文本化（将子图转为边列表文本）；以及对比实验中的Mean Pooling、Rand‑k、All Tokens等基线。

**📊 数据集**

实验基准为ExplaGraphs（稀疏推理型）和WebQSP（实体密集型）两大图问答数据集。

**📈 对比分析**

实验方法为在Llama‑2‑7b上使用AdamW训练，冻结LLM或通过LoRA适配；对比Mean Pooling、All Tokens、Rand‑k等基线以及五种池化算子。结果显示：在LoRA条件下，VNPool可与All Tokens相当，WebQSP Hit@1约73%；在ExplaGraphs上，VNPool在LoRA下达到≈87%准确率；去掉文本化后ExplaGraphs性能提升但WebQSP性能显著下降，证明文本化仍不可或缺。

**⚠️ 局限性**

局限性包括：①仅在两数据集上评估，缺乏对更复杂多跳问答的验证；②实验仅基于Llama‑2‑7b，无法推断在更大或不同LLM上的表现；③对复杂池化方法（如DiffPool、MinCutPool）仍不稳定，需进一步改进；④FandE指标仅在两数据集上测试，未覆盖更广泛的结构特征。

---

## 131. Practice Less, Explain More: LLM-Supported Self-Explanation Improves Explanation Quality on Transfer Problems in Calculus

**arXiv ID:** 2604.00142 | [PDF](https://arxiv.org/pdf/2604.00142v1)

**作者:** Eason Chen `[一作]` (Carnegie Mellon University), Ken Koedinger `[通讯]` (Carnegie Mellon University)

**通讯引用:** 26686 | [OpenAlex ID](https://openalex.org/A5062550465)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一次60分钟的练习实验中，比较了无自我解释、菜单式自我解释以及使用LLM提供反馈的开放式自我解释在微积分学习中的效果，并测评了学习增益与迁移问题的解释质量。

**💡 创新点**

首次证明LLM支持的开放式自我解释能显著提升需要元认知判断的迁移问题（NEI）的解释质量，即使完成的练习题数量大幅减少，且克服了传统对话式自我解释系统手工规则的可扩展性瓶颈。

**🔧 技术方法**

使用 GPT‑5.1 进行自动评估与反馈，配合 NextJS/ PostgreSQL 实现交互式学习平台，采用四等级评估量表，并用 ANCOVA 进行统计分析。

**📊 数据集**

构建了自定义的微积分练习题与迁移题（NEI/EI），并从 Prolific 招募了 92 名受试者，形成了实验数据集。

**📈 对比分析**

通过 ANCOVA 控制先前分数和顺序平衡，发现开放式自我解释组在 NEI 迁移解释质量上比对照组提升 11.9%（p = .030），在所有开放式迁移解释上呈边缘显著提升 7.3%（p = .057）；整体学习增益与 MCQ 准确率无显著差异。

**⚠️ 局限性**

限制包括：开放式组的反馈强度与对照组不一致，难以单独归因于自我解释；样本量小，可能欠缺统计功效；未直接测量学习者的参与度；结果可能仅适用于微积分领域。

---

## 132. On the Necessity of Pre-agreed Secrets for Thwarting Last-minute Coercion: Vulnerabilities and Lessons From the Loki E-voting Protocol

**arXiv ID:** 2604.00188 | [PDF](https://arxiv.org/pdf/2604.00188v1)

**作者:** Jingxin Qiao `[一作]` (University of Edinburgh), Thomas Zacharias `[通讯]` (University of Glasgow)

**通讯引用:** 2078 | [OpenAlex ID](https://openalex.org/A5031029773)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文分析了Loki电子投票协议的强制投票抵抗性，发现两种漏洞并证明若无预先约定秘密，无法抵御最后时刻强制投票，随后提出改进版本CR‑Loki并给出安全证明。

**💡 创新点**

创新点在于：①使用Küsters等人通用的强制投票抵抗定义，对Loki进行完整的游戏论证；②证明在信号机制与“最后投票计数”策略下，任何无预先共享秘密的重新投票协议必然不安全；③设计CR‑Loki通过引入预先约定的随机凭证和双重投票机制，消除暴力与强制弃票攻击。

**🔧 技术方法**

采用的技术包括：Küsters等人的CR‑Integrity / CR‑Privacy游戏模型、零知识证明（NIZK）、IND‑CCA 与 IND‑RCCA 加密方案、重加密技术以及对投票分布的概率分析。

**📊 数据集**

实验数据来自爱沙尼亚2021年地方选举的真实投票行为分布（投票次数、弃权率等），用于评估攻击成功率与协议改进效果。

**📈 对比分析**

通过将CR‑Loki与JCJ*协议在相同噪声分布下对比，发现CR‑Loki在计票效率上更优（线性复杂度对比准线性），但需更强的投票服务器可信假设；在安全性方面，CR‑Loki恢复了强制投票抵抗、隐私与可验证性。

**⚠️ 局限性**

局限性在于：改进协议必须依赖投票服务器的额外可信性，破坏了原设计的无预先约定秘密原则；同时，安全分析仍基于理想化模型，实际部署时对大规模选举的效率与实现复杂度仍需进一步评估。

---

## 133. Vocal Prognostic Digital Biomarkers in Monitoring Chronic Heart Failure: A Longitudinal Observational Study

**arXiv ID:** 2604.00308 | [PDF](https://arxiv.org/pdf/2604.00308v1)

**作者:** Fan Wu `[一作]` (ETH Zurich), Filipe Barata `[通讯]` (ETH Zurich)

**通讯引用:** 1041 | [OpenAlex ID](https://openalex.org/A5024270049)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究通过两个月的家用移动设备监测，收集慢性心衰患者每日语音（元音与语音）与标准护理指标（体重、血压）以及健康状态评估（KCCQ）数据，利用可解释机器学习模型预测患者第二天的健康状态。

**💡 创新点**

创新点在于：①首次针对慢性心衰患者开展长期家用语音监测；②结合元音与语音多维声学特征以及时间序列描述，构建可解释的预测模型；③发现一系列可预测健康恶化的声学生物标志物，如能量峰迟滞、低能量变异、振动波动等。

**🔧 技术方法**

技术手段包括：音频预处理与去噪；使用 OpenSMILE、SenseLab、DisVoice 提取 5,254 维元音特征和 25 维语音特征；采用 7-14 天窗口的时间序列统计量；随机森林与 XGBoost 机器学习模型；递归特征消除与嵌套交叉验证；SHAP 解释性分析。

**📊 数据集**

数据集：32 名慢性心衰患者，平均每日 12 条语音文件，共 21,863 条有效录音；配合每日体重、血压记录以及双周 KCCQ 评分；数据来源为自我记录的手机录音与自测指标。

**📈 对比分析**

与标准护理（体重、血压）以及症状追踪（HFaST）比较时，语音特征模型在 4–6 天窗口下实现 0.826 的灵敏度、0.782 的特异性，AUC 0.77；标准护理模型仅达 0.65‑0.68 的 AUC。多模态（语音+SoC+HFaST）进一步提升至 AUC 0.77+、F1 0.86、MCC 0.70。

**⚠️ 局限性**

局限性：样本量小（32 名）、单语（德语）且男女比例偏男性；录音环境缺乏标准化，可能受情绪、感冒等干扰；学习效应导致语速随时间提升；未涵盖更广泛的语言、文化和病程差异，需在更大、多中心数据集验证。

---

## 134. DriftScript: A Domain-Specific Language for Programming Non-Axiomatic Reasoning Agents

**arXiv ID:** 2604.00043 | [PDF](https://arxiv.org/pdf/2604.00043v1)

**作者:** Seamus Brady `[一作]` `[通讯]`, Seamus Brady

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DriftScript，一种可编译为 Narsese 的 Lisp‑风格 DSL，用以简化 NARS 语言的书写与维护，并与 DriftNARS 运行时集成实现感知‑推理‑行动循环。

**💡 创新点**

创新点在于：①用可读的关键词 S‑expression 取代 Narsese 的符号密集语法；②提供覆盖 NAL 1–8 的完整句子与术语构造；③实现零依赖、四阶段的 C99 编译器；④支持回调与 HTTP 接口，实现跨语言（C、Python、HTTP）代理执行。

**🔧 技术方法**

技术手段包括：Lisp‑style 递归下降解析、静态验证与错误诊断、字面量与变量映射、操作符注册与调用的回调机制，以及 HTTP JSON 接口与 Python ctypes 包装。

**📊 数据集**

使用了 106 条单元/集成测试用例（覆盖语法、算子、变量、错误检测等），并对比 12 个代表性 Narsese 程序的执行结果以验证等价性；在测试数据上并未采用公开数据集。

**📈 对比分析**

比较方法是将 DriftScript 编译后的 Narsese 与手写 Narsese 逐字比较，确认输出完全一致；性能测评显示 300 条表单在 Apple M 系列处理器上仅 3 ms，且不使用动态内存。结构可读性指标显示符号字符减少 36%。

**⚠️ 局限性**

局限性包括：缺乏语义层面（类型检查、优化）验证；不支持所有 NAL 高阶语法，仅提供最常用构造；评估仅基于结构和等价性测试，未进行用户研究或基准对比；大规模批处理需改进动态分配。

---

## 135. Signals: Trajectory Sampling and Triage for Agentic Interactions

**arXiv ID:** 2604.00356 | [PDF](https://arxiv.org/pdf/2604.00356v1)

**作者:** Shuguang Chen `[一作]` (DigitalOcean Holdings, Inc.), Salman Paracha `[通讯]` (DigitalOcean Holdings, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了轻量级的信号框架，用于在多步骤代理交互中对轨迹进行 triage，识别可为后续偏好数据构建提供信息的轨迹。

**💡 创新点**

将交互、执行和环境三类行为信号聚合成可计算且不需要模型调用的规则检测器，并证明其比随机或长度启发式采样更能高效发现开发者可用的轨迹。

**🔧 技术方法**

基于规则的文本匹配、工具调用日志分析以及简单的序列模式检测，构成信号检测器并形成采样权重。

**📊 数据集**

使用 τ‑bench 这一航空与零售两域的工具增强代理基准数据集进行实验。

**📈 对比分析**

与随机采样和长度启发式采样对比，信号采样在 100 条样本下的“信息性率”提升至 82%（比随机高 28%），在奖励和领域分层下保持优势，效率提升 1.52 倍。

**⚠️ 局限性**

实验仅覆盖两域且使用 LLM 模拟用户，信号对语义正确性和细粒度策略缺陷不敏感，且纯规则检测可能缺失隐含的对齐或挫折信号。

---

## 136. The Chronicles of RiDiC: Generating Datasets with Controlled Popularity Distribution for Long-form Factuality Evaluation

**arXiv ID:** 2604.00019 | [PDF](https://arxiv.org/pdf/2604.00019v1)

**作者:** Pavel Braslavski `[一作]`, Alexander Panchenko `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个可配置的多语言管道，用于生成具有可控实体流行度分布的数据集，并基于此构建了包含河流、自然灾害和汽车模型的3,000个实体的RiDiC数据集，以评估大型语言模型在长篇生成中的事实准确性。

**💡 创新点**

创新点包括：①一种通用管道能够根据Wikidata查询和多种流行度指标（如维基百科页面浏览量、链接数、三元组数）按头/腰/尾分层抽样实体；②将该管道应用于多语言（英语、中文）并覆盖不同领域；③在原有FActScore框架基础上，使用统一LLM（Llama‑3.1‑8B、Qwen2.5‑7B）实现事实提取与验证，并提供多种证据来源（单页、搜索结果、关联页面）进行评估。

**🔧 技术方法**

技术方法包括：Wikidata SPARQL查询抽取实体；对实体计算多种流行度指标并分层采样；使用vLLM加速LLM推理；改进的FActScore事实检查流程（提取、检索、验证）；使用中文翻译评估中文事实性；针对河流长度等单一属性进行细粒度评估；统计词汇多样性（Heap's law）。

**📊 数据集**

主要使用的数据集有：①自建的RiDiC（3,000个实体，分三领域、三流行度层、两语种）；②公开的FActScore、EntityQuestions、PopQA、TriviaQA/NQ等作为对照；③维基百科页面文本（英语、中文）和搜索结果、链接页面作为证据；④Wikidata属性（如P2043河流长度）用于细粒度评估。

**📈 对比分析**

与GPT‑5、Llama‑3‑8B‑Instruct、Qwen‑2.5‑7B‑Instruct三大模型进行对比；在英语场景下，GPT‑5平均事实准确率最高（0.55–0.67），而Llama和Qwen在头、腰、尾层均表现出约2倍的准确率下降；中文实验中所有模型准确率普遍低于英语，差距约0.21–0.42；加入搜索结果或关联页面对尾部实体可提升准确率，但对头部实体会产生负面影响；在单属性评估（河流长度）中，流行度高的河流预测误差更小；词汇多样性随流行度降低而下降。

**⚠️ 局限性**

局限性包括：仅评估了三种LLM（两种同级别）和两种语言，缺乏更广泛的模型与语言覆盖；中文评估受限于低质量证据和模型训练数据；排除维基百科stub和短文导致数据集偏向热门实体，长尾实体覆盖不足；使用英语维基页面浏览量作为流行度信号可能产生地区偏差；依赖维基百科作为唯一证据来源，可能遗漏其他可信知识库。

---

## 137. The Data Hydration Gap: A Formal Model of Underinvestment in General-Purpose Data Products Under Decentralized Governance

**arXiv ID:** 2604.00218 | [PDF](https://arxiv.org/pdf/2604.00218v1)

**作者:** Gaston Besanson `[一作]` `[通讯]` (Universidad Torcuato Di Tella), Gaston Besanson (Universidad Torcuato Di Tella)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

提出并分析了数据网格环境下数据产品通用性不足的博弈模型，揭示了“数据网格陷阱”与技术债务二次方增长的机制

**💡 创新点**

首次将数据网格的通用性问题形式化为公共物品外部性，并给出治理方案的理论最优性与边界条件，提供了衡量技术债务与福利损失的定量工具

**🔧 技术方法**

利用微观经济学博弈论与外部性理论构建模型，并在可观测的企业代理参数上进行示例校准

**📊 数据集**

未使用实际数据集，采用可观测的企业代理指标（如域规模、跨域请求比例等）进行示例参数设定

**📈 对比分析**

通过比较中心化、联邦和混合治理模式的理论福利差异，指出中心化或激励对齐在理论上可缓解外部性，性能以福利提升幅度衡量

**⚠️ 局限性**

假设完全信息、对称域、静态博弈以及单维通用性指标，忽略信息不对称、动态学习与多维通用性等实际复杂性

---

## 138. Explainable AI for Blind and Low-Vision Users: Navigating Trust, Modality, and Interpretability in the Agentic Era

**arXiv ID:** 2604.00187 | [PDF](https://arxiv.org/pdf/2604.00187v1)

**作者:** Abu Noman Md Sakib `[一作]` (University of Texas at San Antonio), Taslima Akter `[通讯]` (University of Texas at San Antonio)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过访谈研究盲人和低视力用户在使用生成式 AI 助手时的可解释性需求，聚焦信任、偏差与多模态交互。

**💡 创新点**

创新点在于揭示“自责偏差”与任务风险的关系，并提出以对话式解释和责任感知为核心的可访问 XAI 设计框架。

**🔧 技术方法**

采用半结构访谈、主题分析以及对话式交互实验，结合视觉模型与语言模型进行技术实现。

**📊 数据集**

数据来源为六名盲/低视力参与者的访谈记录，并使用公开的视觉与语言模型（如 CLIP、GPT 等）进行实验。

**📈 对比分析**

通过用户体验对比不同解释方式（文本规则 vs 对话交互）发现对话方式在可用性与信任度上显著优于静态文本。

**⚠️ 局限性**

局限在于样本规模有限、缺乏量化性能指标，以及仅在个人经验层面验证，未能展示跨场景的可推广性。

---

## 139. Reasoning about Transactional Isolation Levels with Isolde

**arXiv ID:** 2604.00159 | [PDF](https://arxiv.org/pdf/2604.00159v1)

**作者:** Manuel Barros `[一作]` (Carnegie Mellon University), Eunsuk Kang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1480 | [OpenAlex ID](https://openalex.org/A5044511705)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文提出并实现了一种名为 Isolde 的工具，能够自动生成在某种事务隔离级别下被允许而在另一种隔离级别下被禁止的执行例子，从而实现对事务隔离语义的自动化分析与验证。

**💡 创新点**

其创新点在于：①首次实现了隔离级别语义之间的自动对比与反例生成；②利用该机制对标准隔离级别的不同规范进行等价性检验，并成功发现了现有隔离级别检查器中的隐藏缺陷。

**🔧 技术方法**

技术方法包括：基于形式化语义框架对隔离级别进行建模；利用自动化工具（如模型检验或 SMT 求解）生成满足指定语义的事务执行序列；通过程序生成与验证机制对比不同规范。

**📊 数据集**

论文中未公开使用具体的工业或公开数据集，主要采用了自定义的事务脚本和示例事务来验证工具的有效性。

**📈 对比分析**

评估方式主要通过复现文献中已知的难题和在标准隔离级别检查器中发现的 Bug 进行验证；未给出具体性能对比数据，重点强调工具在理论验证和缺陷发现上的实用性。

**⚠️ 局限性**

局限性包括：①工具依赖于对隔离级别语义的准确形式化定义；②目前仅针对几种主流隔离级别的规范，难以覆盖所有数据库实现；③在处理大规模事务执行序列时的计算复杂度与可扩展性尚待进一步研究。

---

## 140. Lévy-Flow Models: Heavy-Tail-Aware Normalizing Flows for Financial Risk Management

**arXiv ID:** 2604.00195 | [PDF](https://arxiv.org/pdf/2604.00195v1)

**作者:** Rachid Drissi `[一作]` `[通讯]`, Rachid Drissi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Lévy-Flows——使用VG或NIG Lévy过程作为基分布的正则化流，解决传统Gaussian流在尾部风险低估问题。

**💡 创新点**

创新点在于将Lévy过程的半重尾、可解释参数结构与可逆流相结合，并提供尾部指数保持理论与正则化流的尾部形状保持证明。

**🔧 技术方法**

采用正则化流架构（Neural Spline Flow）、Lévy过程的重参数化采样、极值理论尾部分析以及VaR/ES回测。

**📊 数据集**

主要使用S&P 500日收益率（2000–2025）及AAPL、EEM、金价等三类资产的数据集。

**📈 对比分析**

与Gaussian-Flow、Student-t Flow、无流基准等对比，Lévy-Flows在NLL上提升约69%（VG）或53%（NIG），在95% VaR回测中VG实现完美校准，NIG在ES上误差仅1.6%。

**⚠️ 局限性**

局限性包括：风险回测仅覆盖S&P 500，未对多资产执行完整VaR/ES测试；固定参数Student-t未做广泛搜索；仅限单变量，缺乏多变量或条件化扩展。

---

## 141. Perspective: Towards sustainable exploration of chemical spaces with machine learning

**arXiv ID:** 2604.00069 | [PDF](https://arxiv.org/pdf/2604.00069v1)

**作者:** Leonardo Medrano Sandonas `[一作]` (TUD Dresden University of Technology), Gianaurelio Cuniberti `[通讯]` (TUD Dresden University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了在材料科学与物理化学领域实现绿色AI的框架与方法，聚焦于减少计算碳足迹并提升数据与模型的可持续性。

**💡 创新点**

创新点包括：① 将等变换学习、基于多模态的量子启发式表示与可解释性相结合；② 采用预训练与微调的基础模型、知识蒸馏与活跃学习实现数据效率；③ 构建面向材料与药物的专属小型BERT与生成模型，减少训练能源。

**🔧 技术方法**

所用技术涵盖：等变换图神经网络（GRACE、MACE）、Transformer/GCN、VAE、扩散模型、GAN、强化学习、活跃学习、贝叶斯优化、模型蒸馏与量子机器学习。

**📊 数据集**

使用的主要数据集有：QM7‑X、ANI、QM9、MD17/22、DES、GEMS、QMugs、Aquamarine、SPICE、Materials Project、OMat24、Alexandria、OQMD、AFLOW、NOMAD、ChemDataExtractor生成的化学性质数据库等。

**📈 对比分析**

通过与传统DFT/经典势能比较，等变换MLFF在仅少量训练样本下即可达到与高阶DFT相当的力场精度；基础模型与微调后预测精度高于从零训练的模型，同时大幅降低计算时间；活跃学习与贝叶斯优化在构建MLIP、材料筛选时能将样本量缩减50–80%，保持甚至提升预测性能。

**⚠️ 局限性**

主要局限包括：① 预训练与大型基础模型的初始计算成本高，仍需大量高质量数据；② 对罕见化学空间的泛化能力有限；③ 绿色AI的能源与碳排放指标尚缺标准化评估；④ 生成模型在可合成性、稳定性与实验验证方面仍面临瓶颈。

---

## 142. Source Known Identifiers: A Three-Tier Identity System for Distributed Applications

**arXiv ID:** 2604.00151 | [PDF](https://arxiv.org/pdf/2604.00151v1)

**作者:** Duran Serkan Kılıç `[一作]` `[通讯]` (Independent Researcher), Duran Serkan Kılıç (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了三层身份标识系统（SKID、SKEID、Secure SKEID），实现可在数据库、可信内网、外部通信三层统一身份管理。

**💡 创新点**

创新点在于将时间戳、拓扑信息、序列号编码为64位SKID，并在此基础上扩展到128位SKEID（包含MAC、实体类型、epoch等），外部层再加AES‑256加密，形成零查询验证、顺序可排序、机密性、多世纪可用等六项特性。

**🔧 技术方法**

使用C#/.NET 10实现，核心技术包括：基于位运算的定制字段布局、BLAKE3 keyed MAC、AES‑256‑ECB单块加密、epoch多级时间管理、无锁原子计数器、BenchmarkDotNet性能测评。

**📊 数据集**

没有使用公开数据集；实验基于自定义生成的标识，Benchmarks在Apple M2上运行。

**📈 对比分析**

对比方法：在相同硬件下用BenchmarkDotNet测量SKID、SKEID、Secure SKEID与UUID V4、V7、Snowflake、ULID、CUID2、KSUID的生成、解析时间。结果显示SKID比UUID V7快10倍，SKEID 1.6倍快，Secure SKEID约1.4倍慢；存储上SKID 8 B、外部16 B，优于UUID 16 B。

**⚠️ 局限性**

局限性包括：唯一性仅局限于单个部署与同一epoch，需要手动分配AppId/InstanceId，依赖密钥，缺乏正式安全证明，Secure SKEID不完全符合RFC 9562严格校验，epoch切换未实现，且需要额外的键库管理。

---

## 143. Omni-MMSI: Toward Identity-attributed Social Interaction Understanding

**arXiv ID:** 2604.00267 | [PDF](https://arxiv.org/pdf/2604.00267v1)

**作者:** Xinpeng Li `[一作]` (University of Texas at Dallas), Yapeng Tian `[通讯]` (University of Texas at Dallas)

**通讯引用:** 10926 | [OpenAlex ID](https://openalex.org/A5101835756)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Omni-MMSI任务和Omni-MMSI-R参考引导管线，解决从原始音视频中自动生成身份归属社交线索并推理社交互动的难题。

**💡 创新点**

创新点在于使用个体音影参考作为身份锚点，结合工具生成身份归属线索，再通过链式思维(CoT)实现精准社交推理，显著提升身份归属和互动理解。

**🔧 技术方法**

技术包括多模态工具（Whisper、SpeechBrain、YOLO、OSNet）、Qwen2.5 Omni 7B LoRA微调与CoT监督、人工构建的音影参考与CoT注释。

**📊 数据集**

使用Ego4D和YouTube（Werewolf Among Us子集）两大社交数据集，并手工制作了69个音影参考对与CoT推理样本。

**📈 对比分析**

与现有Omni-LLMs（Gemini 2.5 Pro、Qwen3 Omni 30B等）及传统管线相比，Omni-MMSI-R在STI/PCR任务上分别提升了约5–12%（Ego4D）和2–11%（YouTube）的平均准确率，身份归属准确率提升超过23%。

**⚠️ 局限性**

局限性包括需手工收集个体音影参考，难以在大规模多模多人场景中自动扩展；对大模型的参考利用效果不佳，导致部分模型无法充分获益；且在极端遮挡或语音重叠场景下仍可能出现归属误差。

---

## 144. Task-Centric Personalized Federated Fine-Tuning of Language Models

**arXiv ID:** 2604.00050 | [PDF](https://arxiv.org/pdf/2604.00050v1)

**作者:** Gabriel U. Talasso `[一作]` (Universidade Estadual de Campinas), Leandro A. Villas `[通讯]` (Universidade Estadual de Campinas)

**通讯引用:** 5177 | [OpenAlex ID](https://openalex.org/A5025642074)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于聚类的任务中心化个性化联邦学习框架（FedRouter），通过在本地和全局两层聚类分别为每个任务训练专属适配器，并在推理时使用评估路由器将样本路由到最合适的适配器，以解决个性化联邦学习中的任务干扰和泛化失效问题。

**💡 创新点**

创新点在于：
1) 将个性化视角从“客户中心”转移到“任务中心”，在每个客户端内部先聚类出不同任务，再跨客户端聚类相似任务；
2) 引入两阶段聚类（本地聚类+全局聚类）以及双模评估路由器，实现对未知任务和分布漂移的自适应推理；
3) 采用轮询式训练和聚合策略，显著降低通信和计算成本。

**🔧 技术方法**

主要技术：
- 联邦学习框架（Flower、OpenFedLLM）
- 参数高效微调（LoRA adapters）
- K‑Means聚类（本地与全局）
- 轮询式通信与模型聚合
- 评估路由器（基于欧氏距离的最近邻路由）

**📊 数据集**

使用的多任务数据集：FLAN 4 任务（QQP、WebNLG、Samsum、GigaWord），评估指标为 ROUGE‑1。

**📈 对比分析**

与 FedIT、FedCluster、FedDPA、FedSA、Local 等基线对比，FedRouter 在任务干扰情景（Dual、All）下平均提升 6.1%（相对 3.5%）的性能，在泛化（对未见任务推理）情景下相对提升 136%（绝对 0.583 vs 0.461）。整体来看，FedRouter 在多任务联邦微调中表现最优。

**⚠️ 局限性**

局限性：
1) 需要先通过聚类将数据划分为任务，若聚类效果差会影响后续训练；
2) 对任务数量的假设有限，未在更大规模任务集上验证；
3) 轮询式通信虽降低成本，但仍需中心服务器参与；
4) 对极端分布漂移或完全新任务的泛化能力仍有提升空间。

---

## 145. Multi-lingual Multi-institutional Electronic Health Record based Predictive Model

**arXiv ID:** 2604.00027 | [PDF](https://arxiv.org/pdf/2604.00027v1)

**作者:** Kyunghoon Hur `[一作]` (Korea Advanced Institute of Science and Technology), Edward Choi `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 7171 | [OpenAlex ID](https://openalex.org/A5034622258)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究通过将多机构、多语言ICU电子病历（EHR）转换为统一的文本事件表示，构建了一种无需手工标准化的跨国预测模型；

**💡 创新点**

其创新点在于将轻量级LLM词级翻译与文本化表征相结合，实现了多语言EHR的无缝融合，并在多机构学习中显著提升了泛化性能；

**🔧 技术方法**

核心技术包括：事件序列的文本线性化、子词分词与Transformer编码、LLM（Qwen3-Instruct-8B/MedGemma-7b-it）词级翻译以及多任务学习框架；

**📊 数据集**

实验使用了七个公开ICU数据集（MIMIC-IV、eICU、NWICU、EHRSHOT、UMCdb、HiRID、SICdb），涵盖美国、荷兰、瑞士、奥地利等国，语言包含英语、荷兰语、德语；

**📈 对比分析**

与基线（YAIB、Rajkomar、GenHPF、ReMed）对比，翻译统一后的文本模型在多机构学习中平均AUROC提升约1-2个百分点，且在跨语言少样本迁移时表现与传统特征对齐模型相当，甚至更具可扩展性；

**⚠️ 局限性**

局限性包括：语言覆盖有限，仅涵盖少数欧洲语言；依赖LLM词级翻译，可能引入翻译错误；未探究更高成本的上下文级翻译或多模态扩展。

---

## 146. FGR-ColBERT: Identifying Fine-Grained Relevance Tokens During Retrieval

**arXiv ID:** 2604.00242 | [PDF](https://arxiv.org/pdf/2604.00242v1)

**作者:** Antonín Jarolím `[一作]` (Brno University of Technology), Martin Fajčík `[通讯]` (Brno University of Technology)

**通讯引用:** 105 | [OpenAlex ID](https://openalex.org/A5077658504)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在ColBERT检索框架中引入了细粒度相关性标记的预测功能，使检索过程中直接输出与查询相关的文本片段。

**💡 创新点**

通过将LLM生成的证据片段蒸馏到检索模型中，实现在检索时即刻获取token级相关性得分，避免了后处理的高延迟。

**🔧 技术方法**

采用ColBERT的late interaction机制并加入一个带残差的前馈网络进行token级相关性预测，结合KL散度与二元交叉熵的联合损失进行训练。

**📊 数据集**

在MS MARCO基准上利用Gemma 2生成的证据片段创建MS‑MARCO‑Gemma‑Train/Dev数据集，并对少量样本进行人工标注。

**📈 对比分析**

与原ColBERT、Gemma 2等基线对比，FGR‑ColBERT在token‑级F1上达64.5（超过Gemma 2的62.8），检索Recall@50保持97.1（相当于原模型的99 %），仅增加约1.12×延迟。

**⚠️ 局限性**

模型仅在MS MARCO上验证，尚未在更广泛的检索基准或长文档场景中测试，且对负样本的token级监督缺失可能导致过度识别相关片段。

---

## 147. Brevity Constraints Reverse Performance Hierarchies in Language Models

**arXiv ID:** 2604.00025 | [PDF](https://arxiv.org/pdf/2604.00025v1)

**作者:** MD Azizul Hakim `[一作]` `[通讯]` (Bangladesh Sweden Polytechnic Institute), MD Azizul Hakim (Bangladesh Sweden Polytechnic Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估 31 只语言模型（0.5B-405B 参数）在 5 大基准数据集上的 1,485 道题目，发现 7.7% 的题目中大模型的表现反而低于小模型，归因于大模型在标准提示下倾向于过度展开（overthinking）。

**💡 创新点**

提出“规模感知提示工程”理念：通过限制回答简洁度来消除大模型的过度展开，显著提升大模型性能并反转性能差距，首次在标准评测中证明大模型的潜在能力被通用提示掩盖。

**🔧 技术方法**

使用了长度约束（brief）和直接答案（direct）两种提示干预，结合响应长度分析、因果实验、以及三种数据集污染验证方法，评估模型在不同提示条件下的准确率和回答长度。

**📊 数据集**

实验数据集包括 GSM8K（数学推理）、BoolQ（阅读理解）、ARC‑Easy（科学问题）、CommonsenseQA（常识推理）和 MMLU‑STEM（科学知识），共 46,035 条评测样本。

**📈 对比分析**

比较方法：对比标准提示（control）与简洁提示（brief）和直接答案提示（direct）的准确率差异；发现大模型在简洁提示下平均提升 26.3pp，逆向缩小 67% 的 44.2pp 逆向差距，并在数学与科学基准上完全反转，优于小模型 7.7–15.9pp。

**⚠️ 局限性**

局限性：仅使用贪婪解码，未验证温度采样下的效果；基准覆盖范围有限，未检验生成任务；干预仅对部分大模型进行，可能是上限估计；对 RLHF 对过度展开影响的推测尚未实证。

---

## 148. Suppressing Non-Semantic Noise in Masked Image Modeling Representations

**arXiv ID:** 2604.00172 | [PDF](https://arxiv.org/pdf/2604.00172v1)

**作者:** Martine Hjelkrem-Tan `[一作]` (University of Oslo), Adín Ramírez Rivera `[通讯]` (University of Oslo)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5041630241)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过对ViT的补丁嵌入做PCA分析，提出语义不变性分数（Semantic Invariance Score）并基于此开发后置去噪方法（SOAP），直接抑制MIM模型中捕获的非语义（位置信息）噪声。

**💡 创新点**

创新点在于：①首次量化MIM预训练模型中非语义噪声的占比；②提出无监督、无训练的去噪头，能在任何MIM模型后端单线性投影即可使用；③通过语义不变性分数自动筛选并削弱噪声主成分，显著提升零样本下的下游性能。

**🔧 技术方法**

核心技术包括：主成分分析（PCA）、语义不变性分数计算（对比真实与合成图像的激活分布）、基于Gram–Schmidt的投影去噪，以及使用Fermi窗口对分数进行平滑阈值化。

**📊 数据集**

使用的数据集主要有ImageNet（训练集用于估计协方差，验证集与50000张合成噪声图用于计算SI分数），以及下游评测数据集ECSSD、DUTS、DUT‑OMRON（显著性分割），ADE20k（kNN分割）和ImageNet（kNN分类）。

**📈 对比分析**

与原始MIM模型（如MAE、DINOv2、iBOT等）以及非MIM模型（DINO、DeiT3）对比，经过SOAP后，ECSSD、DUTS、DUT‑OMRON的max Fβ提升约1–3点，ADE20k kNN分割IoU提升0.5–1点，ImageNet kNN分类Top‑1提升0.2–1.3个百分点；整体表现明显优于未去噪版本和RASA等现有去噪方案。

**⚠️ 局限性**

主要局限：仅对原始补丁嵌入进行评估，未考虑更复杂的头部或细化任务；去噪过程假设线性混合模型，可能不适用于所有网络结构；对不同任务的最佳阈值、平滑参数仍需经验选择，缺乏理论最优指导。

---

## 149. A Reliability Evaluation of Hybrid Deterministic-LLM Based Approaches for Academic Course Registration PDF Information Extraction

**arXiv ID:** 2604.00003 | [PDF](https://arxiv.org/pdf/2604.00003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 150. Empirical Validation of the Classification-Verification Dichotomy for AI Safety Gates

**arXiv ID:** 2604.00072 | [PDF](https://arxiv.org/pdf/2604.00072v1)

**作者:** Arsenios Scrivens `[一作]` `[通讯]`, Arsenios Scrivens

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了分类器与验证器在自适应AI安全门控中的可行性，证明分类器无法满足长期安全与持续改进的双重要求；同时提出并验证了基于Lipschitz球的验证门控实现无风险自我改进；

**💡 创新点**

核心创新在于将不可行的分类安全门控与可行的参数空间验证门控进行对比，并通过球链技术展示了可在大尺度模型上实现无风险的持续改进；

**🔧 技术方法**

主要技术包括：二次可判定的Lipschitz球验证器、球链多步扩展、基于梯度的安全改进、LoRA微调的LLM验证、以及多种经典分类器（MLP、SVM、随机森林、k‑NN等）与安全RL方法的基准测试；

**📊 数据集**

实验数据集涵盖：自定义2D点质量任务、MuJoCo Reacher‑v4、Swimmer‑v4、HalfCheetah‑v4（参数维度从240到1824）以及Qwen2.5‑7B LLM的LoRA微调；

**📈 对比分析**

与三种安全RL框架（CPO、Lyapunov、shielding）及18种分类器配置对比，分类器在任何分离度Δ_s≤2.0下均无法满足安全门槛，误判率累积发散；而Lipschitz球验证器在所有实验中保持δ=0，提供线性可扩展性并实现显著的性能提升（如Reacher奖励提升4.31±0.08、LLM 79%步骤通过验证），相较于分类器提升幅度十倍以上；

**⚠️ 局限性**

局限性包括：验证器仅保证相对固定操作域的安全；在LLM规模下的Lipschitz常数估计是条件性保证，需进一步优化；球链过程可能导致安全边缘收缩；实验仅覆盖有限的控制环境与LLM规模，未涉及更复杂的任务或对抗性突变；

---

## 151. A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation

**arXiv ID:** 2604.00249 | [PDF](https://arxiv.org/pdf/2604.00249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 152. Hierarchical Chain-of-Thought Prompting: Enhancing LLM Reasoning Performance and Efficiency

**arXiv ID:** 2604.00130 | [PDF](https://arxiv.org/pdf/2604.00130v1)

**作者:** Xingshuai Huang `[一作]` (Huawei Technologies Canada), Parsa Omidi `[通讯]` (Huawei Technologies Canada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Hierarchical Chain-of-Thought（Hi-CoT）提示方法，通过交替的指令与执行步骤将大语言模型的推理过程组织成层次化的结构；

**💡 创新点**

创新点在于引入层次化压缩瓶颈，使模型在每一步都先生成精简的指令再执行，既消除冗余又防止计划与执行漂移，从而在保持推理连贯性的同时提升准确率与效率；

**🔧 技术方法**

采用了零样本推理时的提示设计，利用语言模型的自回归生成能力实现指令-执行交替；

**📊 数据集**

使用了五个数学推理基准数据集：AIME24、AMC、MATH500、Minerva Math和OlympiadBench；

**📈 对比分析**

与标准提示、Chain-of-Thought（CoT）和Plan-and-Solve（PS）方法进行对比，Hi-CoT在13种模型配置下平均提升6.2%准确率（某些模型最高达61.4%），同时平均减少13.9%的推理长度（最高可降至46.3%），在遵循严格层次格式时部分任务（AMC、MATH500）甚至达到100%准确率；

**⚠️ 局限性**

局限性包括：对严格层次格式的高度依赖导致小模型难以完全遵守，且结构化提示在开放式任务上的通用性有限，未来可能需要通过监督微调或强化学习进一步提高格式遵循率。

---

## 153. Single-Criteria Metric $r$-Dominating Set Problem via Minor-Preserving Support

**arXiv ID:** 2604.00219 | [PDF](https://arxiv.org/pdf/2604.00219v1)

**作者:** Reilly Browne `[一作]` (Dartmouth College), Hsien-Chih Chang `[通讯]` (Dartmouth College)

**通讯引用:** 188 | [OpenAlex ID](https://openalex.org/A5003923315)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了第一个单一准则的多项式时间 O(1)-近似算法，用于求解任意半径 r 的加权度量 r-支配集问题，适用于平面图。

**💡 创新点**

通过构造 Voronoi 合并得到稀疏平面支撑图，并证明任意半径球系统的浅层单元复杂度为 O(n k²)，从而实现唯一 3‑元编码和 Chan 的准均匀抽样技术。

**🔧 技术方法**

使用 Voronoi 图的收缩、稀疏支撑图、唯一 3‑元编码、浅层单元复杂度分析、Chan 等人的准均匀抽样技术（与 Clarkson‑Shor 技巧相结合）。

**📊 数据集**

本文为理论研究，没有实验数据集，算法适用于任意平面图。

**📈 对比分析**

与之前只能得到双准则（半径 (1+ r) + 权重 1+ 近似）的算法相比，该方法在保持原始半径的前提下取得常数因子近似；时间复杂度为多项式。

**⚠️ 局限性**

仅适用于平面图；无法进一步降低近似因子；对于更广泛的无环图或一般图仍无单一准则近似方案；算法对极大半径仅保证 O(1) 近似，而不提供更细化的取舍。

---

## 154. Lead Zirconate Titanate Reservoir Computing for Classification of Written and Spoken Digits

**arXiv ID:** 2604.00207 | [PDF](https://arxiv.org/pdf/2604.00207v1)

**作者:** Thomas Buckley `[一作]` (University of Massachusetts Amherst), Edward Rietman `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 10636 | [OpenAlex ID](https://openalex.org/A5075992947)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用未极化的PZT立方体作为物理回声存储器，对手写数字（MNIST）和语音数字（AudioMNIST）进行分类，结合逻辑回归读出层。

**💡 创新点**

首次在同一预处理数据上，将物理回声存储器与传统线性回归基线直接对比，证明其在中等难度任务上能显著提升性能。

**🔧 技术方法**

硬件包括PZT立方体、8通道高速信号发生器、示波器和Teensy 4.1微控制器；算法为单一逻辑回归读出层和一系列基线方法。

**📊 数据集**

MNIST手写数字（5000样本）和AudioMNIST语音数字（6000样本）两组数据集。

**📈 对比分析**

与直接逻辑回归、窗口求和逻辑回归以及加噪声逻辑回归等基线进行10折交叉验证；MNIST上PZT回声存储器在最佳延迟10时达到89.0%准确率，优于基线86.6%；AudioMNIST上两者相近（约88.2%），未见显著提升。

**⚠️ 局限性**

主要局限在于输入预处理方式降低了信息量，且仅使用单一PZT设备；对更高难度任务的提升有限，且未探索多器件集成或更丰富的信号编码。

---

## 155. Sequence-Aware Split Heuristic to Mitigate SM Underutilization in FlashAttention-3 Low-Head-Count Decoding

**arXiv ID:** 2604.00028 | [PDF](https://arxiv.org/pdf/2604.00028v1)

**作者:** Martí Llopart Font `[一作]` (Barcelona Supercomputing Center), Cristina España-Bonet `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 854 | [OpenAlex ID](https://openalex.org/A5029540488)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 FlashAttention-3 的调度启发式中加入了基于序列长度与块数的拆分策略，以解决 Hopper GPU 在低头数解码时 SM 利用率低下的问题。

**💡 创新点**

提出了“序列感知拆分”规则，突破原始短序列 guard 的限制，在 nblk=4 的低块数桶中使用更高的拆分因子，从而显著提升 GPU 计算吞吐量。

**🔧 技术方法**

采用 OpenEvolve 进化搜索框架先在 FlashAttention-3 的 Python 接口层生成动态拆分逻辑，再将其提炼为简洁的 C++ 规则；实验使用 CUDA Graph 记录纯核执行时间，并通过 A/B 交替计时对比。

**📊 数据集**

实验数据主要基于 Llama‑3.1‑70B‑Instruct（GQA 8:1 KV 比例）的短提示（L_K ≤ 512）推理工作负载，并覆盖 Batch=1、H_KV={1,2,8} 等多种低头数配置。

**📈 对比分析**

在已预先计算调度元数据（scheduler metadata）路径下，Patch 在 Batch=1、H_KV=1,2、L_K=512 的情形实现 21%–24% 的核执行时间提升；对其他配置保持与标准实现一致，无性能回退。

**⚠️ 局限性**

局限性包括：仅验证了 L_K=512 的单一 nblk=4 桶，未扩展到更短序列；改进仅在预调度元数据路径中有效；未在多 GPU、不同 batch 大小或更大 KV 头数的配置下进一步评估。

---

## 156. Cybercrime as a Service: A Scoping Review

**arXiv ID:** 2604.00063 | [PDF](https://arxiv.org/pdf/2604.00063v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 157. A Dual-Stream Transformer Architecture for Illumination-Invariant TIR-LiDAR Person Tracking

**arXiv ID:** 2604.00363 | [PDF](https://arxiv.org/pdf/2604.00363v1)

**作者:** Yuki Minase `[一作]` (University of Fukui), Kanji Tanaka `[通讯]` (University of Fukui)

**通讯引用:** 2432 | [OpenAlex ID](https://openalex.org/A5030913821)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种融合热红外与LiDAR深度的双流Transformer架构，实现全光照条件下的人体跟踪。

**💡 创新点**

引入跨模态知识转移与连续学习策略，将预训练热域模型的结构先验迁移到TIR‑D域，并使用细粒度差异学习率保持特征稳定。

**🔧 技术方法**

双流ResNet‑50+适配层、STARK Transformer融合编码器、差异学习率训练、GIoU+L1损失框回归等技术。

**📊 数据集**

使用自采集的HIKMICRO Pocket2热红外与Velodyne VLP‑16 LiDAR深度数据，对比公开RGB‑D和单模态热/深度数据集。

**📈 对比分析**

与RGB‑D、单模态热/深度以及RGB迁移版本对比，TIR‑D（本研究）在AO 0.700、SR 58.7% 领先，略低于RGB迁移但在弱照明下显著优于其他基线。

**⚠️ 局限性**

受限于少量标注TIR‑D数据，仅做单目标跟踪；未评估多目标或实时嵌入式部署；深度图噪声对性能仍有影响。

---

## 158. Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning

**arXiv ID:** 2604.00344 | [PDF](https://arxiv.org/pdf/2604.00344v1)

**作者:** Eric Hanchen Jiang `[一作]` (University of California Los Angeles), Ying Nian Wu `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Agent Q‑Mix，通过强化学习学习多 LLM 代理之间的通信拓扑，使系统在编程、推理与数学任务上能自适应协同推理。

**💡 创新点**

创新点：① 将通信拓扑学习框架化为协作 MARL 问题；② 采用 QMIX 进行去中心化决策，保证全局最优可通过局部贪婪实现；③ 设计六种可解释的离散通信动作，能生成从无通信到完全连通的多种拓扑；④ 结合拓扑感知 GNN + GRU 的 Agent Q‑网络，实现多轮动态拓扑调整。

**🔧 技术方法**

主要技术：QMIX（单调价值分解）、拓扑感知图神经网络（GNN）消息传递、门控递归单元（GRU）时序记忆、CTDE（集中训练/去中心化执行）、token 费用奖励函数、离散动作空间。

**📊 数据集**

实验数据集：七大基准（LiveCodeBench、HumanEval、MMLU‑Pro、AIME_25、AIME_26、HMMT、Beyond‑AIME），以及 Humanity’s Last Exam（HLE）作为额外评测。基准覆盖编程、推理与数学。

**📈 对比分析**

与单代理、静态多代理、适应性拓扑方法（GPTSwarm、AgentDropout、G‑Designer、GTD、TopoDIM）以及商业框架（Lobster、LangGraph、AutoGen、Agent Framework）比较。Agent Q‑Mix 在 GPT‑OSS:120B 上平均 72.73%，在 Gemini‑3.1‑Flash‑Lite 上平均 66.90%；在 HLE 上 20.8% 领先其他框架；同时在 token 使用上显著更节省，鲁棒性对抗恶意代理也更强。

**⚠️ 局限性**

局限性：① 训练时需要全局图信息，部署时仍受限于局部观察；② 离散动作空间仅包含六种模式，可能不足以覆盖极其复杂的交互需求；③ 目前在大规模代理团队或极长通信回合的任务上扩展性尚未验证；④ 仍依赖奖励设计与 token 限制，可能对不同任务产生偏差。

---

## 159. NFC based inventory control system for secure and efficient communication

**arXiv ID:** 2604.00181 | [PDF](https://arxiv.org/pdf/2604.00181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 160. Terminal Agents Suffice for Enterprise Automation

**arXiv ID:** 2604.00073 | [PDF](https://arxiv.org/pdf/2604.00073v1)

**作者:** Patrice Bechard `[一作]` (ServiceNow), Sai Rajeswar `[通讯]` (ServiceNow)

**通讯引用:** 1234 | [OpenAlex ID](https://openalex.org/A5041629023)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多个生产级企业平台上，对比GUI驱动、工具增强和终端编码三种代理交互方式，评估其在真实任务上的成功率与成本。

**💡 创新点**

提出并验证了极简终端编码代理足以完成大多数企业自动化任务，挑战了高层抽象层的必要性。

**🔧 技术方法**

基于OpenAI Agents SDK，使用Claude Sonnet、Claude Opus、GPT-5.4 Thinking、Gemini 3.1 Pro等前沿LLM，结合Playwright MCP、平台MCP和自定义终端环境。

**📊 数据集**

构建了包含ServiceNow、GitLab、ERPNext三大平台共729个自然语言任务的统一基准，包含任务描述、期望结果与API文档。

**📈 对比分析**

对同一LLM同一任务分别运行三种代理，记录成功率与平均成本；终端代理在大多数组合中与Web代理相当甚至更好，同时成本显著低于Web或MCP代理。

**⚠️ 局限性**

终端代理无法处理仅在浏览器可见的UI操作、需要会话级身份或拖拽式工作流编辑等任务，且依赖于API的完整性与文档结构。

---

## 161. PRISM: Differentiable Analysis-by-Synthesis for Fixel Recovery in Diffusion MRI

**arXiv ID:** 2604.00250 | [PDF](https://arxiv.org/pdf/2604.00250v1)

**作者:** Mohamed Abouagour `[一作]` (Indiana University), Eleftherios Garyfallidis `[通讯]` (Indiana University)

**通讯引用:** 6352 | [OpenAlex ID](https://openalex.org/A5083595381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

提出了PRISM，一种可微分的全程分析-合成框架，用于多组分微结构拟合并直接恢复脑内纤维方向。

**💡 创新点**

核心创新在于显式K纤维参数化、软模型选择（相互排斥与稀疏先验）、引入限制组分、Rician负对数似然自监督噪声估计，以及端到端空间正则化。

**🔧 技术方法**

采用可微分前向模型、Rprop梯度优化、软最大/softplus/正则化约束、三阶段校正模块（偏置场、测量比例校正、噪声级学习）等技术。

**📊 数据集**

在合成交叉纤维数据、DiSCo1数字幻像以及HCP真实扫描数据上进行评估。

**📈 对比分析**

与CSD、MSMT-CSD、ODF-FP等传统方法对比，PRISM在16角合成基准上角度误差降至3.5°（MSE）/2.3°（NLL），召回率99%；在DiSCo1上连通性相关系数提升至0.934；在HCP全脑拟合仅需12分钟，Dice约0.79/0.80。

**⚠️ 局限性**

仍假设几何预处理完成，收敛速度相对较慢，缺乏几何失真校正，仅在强度域处理。

---

## 162. Polish phonology and morphology through the lens of distributional semantics

**arXiv ID:** 2604.00174 | [PDF](https://arxiv.org/pdf/2604.00174v1)

**作者:** Paula Orzechowska `[一作]` (Adam Mickiewicz University), R. Harald Baayen `[通讯]` (University of Tübingen)

**通讯引用:** 43977 | [OpenAlex ID](https://openalex.org/A5087366961)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分布式语义与Discriminative Lexicon Model（DLM）分析波兰语词形（含辅音群）与语义之间的关联。

**💡 创新点**

证明词向量不仅捕捉语义，还能映射形态学、音位学与语法特征，揭示语料层面形义对应的强等同性。

**🔧 技术方法**

使用fastText与Word2Vec词向量、t‑SNE、LDA及DLM的形式‑意义映射算法。

**📊 数据集**

数据来源于《Basic Dictionary of Polish for Foreigners》与NKJP300M语料库，构成约8 000条唯一词形及其形态学属性。

**📈 对比分析**

DLM在理解任务上达到97–98 %准确率，在生成任务上训练集97 %但未见样本约78 %；LDA与t‑SNE对形态与音位属性预测的准确率均高于多数基线，展示了良好的一致性。

**⚠️ 局限性**

局限在于生成任务对未见词形的泛化不足、对极低频或同义/反义词的区分有限，且仅关注词首辅音群，未涉及全词音位细粒度或跨语言验证。

---

## 163. Large Language Models for Analyzing Enterprise Architecture Debt in Unstructured Documentation

**arXiv ID:** 2604.00046 | [PDF](https://arxiv.org/pdf/2604.00046v1)

**作者:** Christin Pagels `[一作]` (Stockholm University), Rob Henk Bemthuis `[通讯]` (University of Twente)

**通讯引用:** 342 | [OpenAlex ID](https://openalex.org/A5022702832)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了基于大型语言模型的工具，用以自动检测企业架构文档中的 EA Smells（企业架构债务信号）

**💡 创新点**

首次将 LLM 应用于无结构企业文档的 EA Smells 检测，并在 CPU 受限的本地环境下实现可部署的细调模型，对比云端 GPT 基准提供了实证

**🔧 技术方法**

采用 LLaMA‑3.2‑3B‑Instruct 模型，使用 LoRA 参数高效微调和少量示例提示；对比使用 ChatGPT Premium 自定义 GPT

**📊 数据集**

构造了 30 份合成但真实感的业务层文档（含手工标注的 12 种 Smell），以及 960 条训练实例（8 个业务域 × 12 种 Smell）

**📈 对比分析**

通过单文档、批量 10、批量 30 三种运行模式，评估准确率、召回率、精确度、F1、误报率和处理时长；GPT 基准精确度≈0.88、误报率≈0.09、速度≈2 s/文档；LLaMA 模型召回率高但精确度仅≈0.26、误报率≈0.95、处理慢≈120 s/文档

**⚠️ 局限性**

受限于合成数据、单一标注者、仅 12 种 Smell、CPU‑only 小模型、批处理上下文泄漏、缺乏交叉验证与置信区间，导致结果难以推广至真实企业环境

---

## 164. A Taxonomy of Programming Languages for Code Generation

**arXiv ID:** 2604.00239 | [PDF](https://arxiv.org/pdf/2604.00239v1)

**作者:** Nishat Raihan `[一作]` (George Mason University), Marcos Zampieri `[通讯]` (George Mason University)

**通讯引用:** 6611 | [OpenAlex ID](https://openalex.org/A5024937008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对编程语言资源分布进行系统化分类，提出四层资源等级体系；

**💡 创新点**

首次以token计数为依据为编程语言构建可复现的资源分类框架；

**🔧 技术方法**

采用代码识别、哈希去重、StarCoder分词器进行token统计与聚合；

**📊 数据集**

整合七大公开代码语料库（The Stack v2、RefineCode、StarCoder Data、The Heap、Project CodeNet、GitHub Code、CodeSearchNet），涵盖646种语言；

**📈 对比分析**

通过Gini系数、CV、Lorenz曲线等统计分析衡量不平衡程度，表明12种高资源语言占74.6%token，71.7%低资源语言仅占1.0%；

**⚠️ 局限性**

方法受限于单一分词器、扩展名推断、哈希去重、阈值设定，可能影响绝对计数但不改变长尾特征。

---

## 165. Informed Machine Learning with Knowledge Landmarks

**arXiv ID:** 2604.00256 | [PDF](https://arxiv.org/pdf/2604.00256v1)

**作者:** Chuyi Dai `[一作]` (University of Alberta), Xianmin Wang `[通讯]` (China University of Geosciences)

**通讯引用:** 3197 | [OpenAlex ID](https://openalex.org/A5020675795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种知识-数据机器学习框架，结合局部数值数据与全局信息粒子知识地标，通过增广损失函数实现数据与知识的平衡；

**💡 创新点**

创新点包括：①引入信息粒子知识地标将定性物理知识转化为可梯度优化的约束；②提出增广损失函数和超参数λ平衡局部数据与全局知识；③系统验证模型在噪声与知识粒度变化下的鲁棒性；

**🔧 技术方法**

采用了粒度计算（justifiable granularity）、条件模糊C均值聚类（conditional FCM）、单隐层MLP、基于梯度的优化以及增广损失函数；

**📊 数据集**

使用了两个物理支配的基准数据集：环境污染扩散模型与活塞周期时间仿真模型，分别生成全域采样与局部观察窗口的数据；

**📈 对比分析**

通过与仅使用局部数据（λ=1）的传统数据驱动模型比较，采用全域误差Q1+Q2评估，结果显示知识-数据模型在环境模型上提升15%~22%，在活塞模型上提升61%~70%，并在不同局部窗口均表现更优；

**⚠️ 局限性**

局限性包括：仅考虑单一知识来源与单一模型类型；知识地标的构造依赖参数范围与粒度；实验仅覆盖两种基准；未验证多源知识或更复杂模型的适用性。

---

## 166. Risk-Aware Batch Testing for Performance Regression Detection

**arXiv ID:** 2604.00222 | [PDF](https://arxiv.org/pdf/2604.00222v1)

**作者:** Ali Sayedsalehi `[一作]` (Concordia University), Gregory Mierzwinski `[通讯]` (Mozilla)

**通讯引用:** 27 | [OpenAlex ID](https://openalex.org/A5054760099)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种统一的风险感知批量测试框架，通过将机器学习预测的提交级别性能回归风险与自适应批量策略结合，降低CI性能测试成本并加快缺陷定位；

**💡 创新点**

创新点在于首次将提交级别的性能回归风险预测与批量调度紧密耦合，并设计多种基于风险的批量策略（如RAPB-la），实现成本与诊断延迟的Pareto改进；

**🔧 技术方法**

使用了Transformer模型（ModernBERT、CodeBERT、LLaMA‑3.1 8B）进行提交风险预测，并基于预测概率构建批量决策逻辑；

**📊 数据集**

数据集为Mozilla Firefox Autoland的真实提交与Perfherder性能回归警报构成的JIT‑Mozilla‑Perf，包含超过1万条提交和人类确认的回归标签；

**📈 对比分析**

通过与Mozilla现行的TWSB基线及耗时/成本指标（总测试数、平均/最大时间到致因）对比，最佳方案RAPB‑la在测试量下降32.4%、最大诊断延迟降低26.2%且年化成本节约约$491K；

**⚠️ 局限性**

主要局限包括依赖人类确认的回归标签、对实际CI队列与硬件波动的模拟简化、模型风险估计对动态运行时因素不敏感，以及结果仅在Mozilla Firefox环境验证，需在其他项目中进一步检验。

---

## 167. ASCAT: An Arabic Scientific Corpus and Benchmark for Advanced Translation Evaluation

**arXiv ID:** 2604.00015 | [PDF](https://arxiv.org/pdf/2604.00015v1)

**作者:** Serry Sibaee `[一作]` (Prince Sultan University), Omer Nacar `[通讯]` (Tuwaiq Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了 ASCAT（阿拉伯语科学翻译基准语料库），包括 500 篇跨 5 领域的英文-阿拉伯语完整科学摘要，采用多引擎机器翻译和专家级人工校对生成高质量参考译文。

**💡 创新点**

创新点在于将生成式 AI、Transformer 模型与商业 MT API 三种翻译技术相结合，并通过分层专家评审确保术语、句法和语义准确性，使得该语料库兼具规模与严谨验证，填补了现有阿拉伯语科学翻译资源的空白。

**🔧 技术方法**

使用的技术包括 Gemini（生成式 AI）、Hugging Face Transformer 模型、Google Translate 与 DeepL 商业 MT API 进行多引擎翻译；随后由 7 名专业领域专家通过结构化校对表进行人工验证。

**📊 数据集**

数据集为 500 篇科学摘要，涵盖物理、数学、计算机科学、量子力学与人工智能等 5 领域，平均英文 141.7 词、阿拉伯文 111.8 词。

**📈 对比分析**

对三大 LLM（GPT‑4o‑mini、Gemini‑3.0‑Flash‑Preview、Qwen3‑235B‑A22B）在该基准上进行评测，采用 BLEU 与 ROUGE 计分；GPT‑4o‑mini 最高 BLEU 37.07，表明该语料库具备良好的判别力，系统间可达 13.4 BLEU 点的性能差异。

**⚠️ 局限性**

局限性包括样本量仅 500 篇，领域分布不均衡；评价主要依赖自动指标，未涵盖语义准确性与专业术语细致度；且未提供大规模人工评估来进一步验证译文质量。

---

## 168. GUIDE: Reinforcement Learning for Behavioral Action Support in Type 1 Diabetes

**arXiv ID:** 2604.00385 | [PDF](https://arxiv.org/pdf/2604.00385v1)

**作者:** Saman Khamesian `[一作]` (Arizona State University), Hassan Ghasemzadeh `[通讯]` (Arizona State University)

**通讯引用:** 4663 | [OpenAlex ID](https://openalex.org/A5007139473)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了GUIDE框架，利用强化学习为1型糖尿病患者提供胰岛素与碳水化合物摄入的行为决策支持，补充自动胰岛素输注系统；

**💡 创新点**

创新点在于：①设计了结构化的行为动作空间（动作类型、剂量与时机）；②使用个性化葡萄糖预测模型与定制奖励函数；③在同一统一环境下对离线与在线强化学习算法进行公平比较；

**🔧 技术方法**

采用强化学习（TD3‑BC、CQL‑BC、PPO、SAC等）、个性化葡萄糖预测模型GLIMMER、定制化奖励函数与人类启发式餐食生成器；

**📊 数据集**

使用AZT1D数据集（25名AID使用者的CGM、胰岛素与饮食记录）；

**📈 对比分析**

在相同环境与奖励下评估算法，CQL‑BC表现最佳，平均TIR 85.49%，TAR 13.97%，TBR 0.53%，显著优于其他离线/在线算法和随机/历史基线；

**⚠️ 局限性**

局限包括：假设患者完全遵从建议；离线数据覆盖不足可能限制策略泛化；缺少对睡眠、运动等其他生活方式因素的建模；需要进一步的真实临床验证。

---

## 169. Dual Contouring of Signed Distance Data

**arXiv ID:** 2604.00157 | [PDF](https://arxiv.org/pdf/2604.00157v1)

**作者:** Xiana Carrera `[一作]` (Columbia University), Silvia Sellán `[通讯]` (Columbia University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于Dual Contouring的局部二次优化算法，仅利用离散SDF样本即可重建具有尖锐特征的多边形网格。

**💡 创新点**

创新点在于：①不依赖梯度或完整SDF查询，而是直接在离散样本上估计Hermite信息并通过迭代优化精细化；②设计了结合距离能量与Hermite能量的局部能量函数，保持尖锐边缘而不产生过度平滑；③采用可并行的细胞级优化框架，降低对全局信息的需求。

**🔧 技术方法**

使用技术包括：二次误差函数（QEF）与局部能量最小化、迭代全局–局部优化循环、SDF到球的线性化、主成分分析更新Hermite数据、基于AABB树的点-网格距离计算、OpenMP并行化。

**📊 数据集**

主要数据集为ABC CAD数据集（50³~200³分辨率），以及对盒形、圆柱等人工测试模型进行验证。

**📈 对比分析**

与Marching Cubes、原始Dual Contouring（估计Hermite）、点云重建方法、Neural Dual Contouring等方法比较，实验显示在中高分辨率下该方法在Hausdorff、Chamfer、边缘Chamfer误差上均优于其他方法，并能准确保持尖锐边缘；计算复杂度线性，批量采样可进一步加速。

**⚠️ 局限性**

局限性包括：对噪声敏感、局部能量逼近可能导致自交或面翻转；细胞顶点需要离开单元以捕捉尖锐特征，易产生不连通或自相交；非凸优化迭代可能陷入局部极小值，需调参或更复杂的全局约束。

---

## 170. When is Generated Code Difficult to Comprehend? Assessing AI Agent Python Code Proficiency in the Wild

**arXiv ID:** 2604.00299 | [PDF](https://arxiv.org/pdf/2604.00299v1)

**作者:** Nanthit Temkulkiat `[一作]` (Mahidol University), Raula Gaikovina Kula `[通讯]` (University of Osaka)

**通讯引用:** 2459 | [OpenAlex ID](https://openalex.org/A5091820517)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过静态分析评估 AI 编码代理产生的 Python 代码的语言熟练度，并与人类编写的代码进行比较，探讨其可维护性和复杂度；

**💡 创新点**

首次将 CEFR 语言熟练度框架应用于编程代码，提出了针对 AI 生成代码的成熟度度量方法，并在真实项目中验证其可行性；

**🔧 技术方法**

使用 pycefr 静态分析工具将 Python 代码构造映射到 CEFR 六个熟练度等级；

**📊 数据集**

使用 AIDev 数据集（包含 591 个合并 PR、3,027 个文件）以及对应的人类代码样本；

**📈 对比分析**

通过统计各熟练度等级构造比例，采用卡方检验和 Kruskal–Wallis 检验比较 AI 与人类代码的分布差异，发现差异显著但效应量小；

**⚠️ 局限性**

局限性包括仅分析 Python 代码、仅覆盖三种代理、静态分析无法捕捉运行时上下文，且缺乏对 PR 审核质量和实际维护成本的直接评估；

---

## 171. Improvisational Games as a Benchmark for Social Intelligence of AI Agents: The Case of Connections

**arXiv ID:** 2604.00284 | [PDF](https://arxiv.org/pdf/2604.00284v1)

**作者:** Gaurav Rajesh Parikh `[一作]` (Duke University), Angikar Ghosal `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并正式化了一种即兴词汇游戏Connections，用以评估LLM的社交智能和推理能力；

**💡 创新点**

创新点在于提出了通过词语关联与他人认知的游戏机制来衡量模型的社交意识，并给出了概率化的线索生成与游戏规则模型；

**🔧 技术方法**

主要技术包括基于GPT‑4o的语言模型、语义嵌入与向量相似度判定、生成式概率模型以及上下文学习进行代理个性化；

**📊 数据集**

实验使用自构词汇表（如kaleidoscope、xenophobia等）进行多轮游戏记录，并未使用公开标准数据集；

**📈 对比分析**

通过对比不同词条下的字母揭示次数、猜测错误率与阻断次数等指标，结果显示在同一模型环境下阻断率高、需要的轮数因前缀常见度而显著变化；

**⚠️ 局限性**

局限性包括固定词条假设导致难以捕捉动态选择、相同模型导致语义网络相似、缺乏对游戏历史的有效记忆以及在无提示情况下社交意识不足。

---

## 172. Making Sense of AI Agents Hype: Adoption, Architectures, and Takeaways from Practitioners

**arXiv ID:** 2604.00189 | [PDF](https://arxiv.org/pdf/2604.00189v1)

**作者:** Ruoyu Su `[一作]` (University of Oulu), Davide Taibi `[通讯]` (University of Southern Denmark)

**通讯引用:** 4505 | [OpenAlex ID](https://openalex.org/A5086929289)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对138场行业会议中关于AI代理的演讲进行定性分析，研究企业从实验到生产的迁移路径、可复用的架构策略以及跨领域实现差异。

**💡 创新点**

首次将LLM辅助的质性分析方法与多模型验证相结合，系统化整理工业实践中的代理架构主题、模式与经验教训。

**🔧 技术方法**

使用Whisper-XXL进行语音转录，GPT（LLM）进行文本分析和结构化输出，并通过三款Validator模型进行结果校验；同时采集了多种LLM与框架技术细节。

**📊 数据集**

收集了2015–2025年期间公开发布的234场演讲（最终筛选为138场），涵盖多家企业和学术机构的实际案例。

**📈 对比分析**

通过对演讲内容的主题编码和模式归纳，对不同业务域中的代理实现做对比；未做数值性能评估，但报告了采纳动机、架构因素、挑战等定性指标。

**⚠️ 局限性**

样本主要来自成功或主流案例，可能低估失败案例；LLM分析存在hallucination风险，尽管通过多模型验证缓解；研究缺乏统一的客观度量与实验验证。

---

## 173. WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics

**arXiv ID:** 2604.00024 | [PDF](https://arxiv.org/pdf/2604.00024v1)

**作者:** Sneha Maurya `[一作]` (Columbia University), Girish Kumar `[通讯]` (Rubric AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了女性健康领域的开放式评测基准WHBench，并使用22种大型语言模型在47个专家设计的临床情景中进行评估。

**💡 创新点**

创新点在于：① 失败模式定向的问答设计；② 采用专业临床参考答案与首创的健康公平维度；③ 通过不对称惩罚的23条多维度评分表，突出安全与公平需求。

**🔧 技术方法**

使用的技术包括：LLM评审体系（Claude Sonnet 4.6 作为主审、GPT‑5.4 作为次审）、服务器端分数重算、零射门闭书评测流程与多轮跑测。

**📊 数据集**

数据集为47个由OB/GYN、妇科肿瘤、泌尿科等专家设计的女性健康场景，每个问题有4–6份独立参考答案；在此基础上，22个LLM产生约3100条回答用于评分。

**📈 对比分析**

比较方法采用正则化分数、正确/部分正确/错误比例及安全/伤害率；最高模型Claude Opus 4.6得分72.1%，正确率35.5%，但安全风险仍显著，表明各模型仍存在显著能力瓶颈。

**⚠️ 局限性**

局限性包括：评审间一致性低（尤其公平、模糊度等主观维度）；公平指标表现最差；覆盖面在某些主题（如骨健康、心理健康）有限；仅英文评测；未实现完整的人工临床裁判。

---

## 174. VLM-in-the-Loop: A Plug-In Quality Assurance Module for ECG Digitization Pipelines

**arXiv ID:** 2604.00396 | [PDF](https://arxiv.org/pdf/2604.00396v1)

**作者:** Jiachen Li `[一作]` (University of Texas at Austin), Dongmei Chen `[通讯]` (University of Texas at Austin)

**通讯引用:** 13157 | [OpenAlex ID](https://openalex.org/A5100677493)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种 VLM‑in‑the‑Loop 插件式质量保证与纠正模块，能够在不改动现有 ECG 数字化后端的前提下，对纸质 ECG 图像进行闭环 VLM 评估与纠错。

**💡 创新点**

创新点在于：①将 VLM 与领域特定工具（SignalQualityTool、MorphologyTool、ReconstructionTool）进行“工具根基”对齐，以提升评估一致性；②提供统一的插件接口和轻量级适配器，实现多后端可插拔；③在闭环反馈中结合 12 种可解释的纠正动作，显著提升数字化质量。

**🔧 技术方法**

使用了大型视觉语言模型（Claude Opus 4、GPT‑4o、Gemini 2.5 Pro）进行评估；通过结构化提示与工具输出构建评估语料；采用闭环 Agent（Evaluator、Optimizer、Gate）实现多层级 QA 与动作选择；使用工具套件提取 SQI、形态学指标与重建质量。

**📊 数据集**

主要数据集包括：428 张真实 HCM 归档 ECG 图像（无原始数字信号）、200 条 PTB‑XL 记录（已配对原始数字信号，用于验证一致性与重建精度）。

**📈 对比分析**

在 PTB‑XL 上，工具根基提升评估一致性从 71% 到 89%，并使得 ΔPCC（PASS 与 BORDERLINE 之差）从 0.03 提升到 0.08；在 428 张 HCM 图像上，VLM‑in‑the‑Loop 将优秀质量从 69.2% 提升至 98.0%，并分别使 ECG‑Digitiser、Open‑ECG‑Digitizer 的有效导联数提升 41.2% 与 2.5→5.8；与三种公开基线相比，全部在真实临床数据上表现出更高的可用率。

**⚠️ 局限性**

局限性包括：①对 68% 的 BORDERLINE 图像存在信号极限，无法通过任何纠错超越；②AUROC 仅 0.64，表明评估仍受限于单一重建指标；③插件适配器仍需为每个后端手工实现；④实验仅在单一机构与单一病种（HCM）上验证，缺乏多中心或跨域通用性验证。

---

## 175. Open, Reliable, and Collective: A Community-Driven Framework for Tool-Using AI Agents

**arXiv ID:** 2604.00137 | [PDF](https://arxiv.org/pdf/2604.00137v1)

**作者:** Hy Dang `[一作]` (University of Notre Dame), Meng Jiang `[通讯]` (University of Notre Dame)

**通讯引用:** 5925 | [OpenAlex ID](https://openalex.org/A5074821819)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍并实现了一个名为OpenTools的社区驱动工具箱，标准化工具接口、提供轻量级包装器、持续评测与监控，支持多种LLM代理框架的工具调用与调试。

**💡 创新点**

首次明确区分工具使用准确性与工具自身的可靠性，并通过社区贡献的测试用例实现持续回归检测；统一JSON‑schema工具定义；搭建公开演示与贡献流程，促进工具生态的持续改进。

**🔧 技术方法**

使用统一的JSON‑schema接口、工具包装器、自动化测试套件与持续集成；集成ReAct、OctoTools、MultiAgent等代理框架；利用Web demo（HuggingFace Spaces）实现交互式评测与社区贡献。

**📊 数据集**

对多领域基准进行评测，包括VQA/Puzzle（AlgoPuzzleVQA、Hallusion‑VD、PuzzleVQA、VQAv2）、Math/Reasoning（Game of 24、Omni‑MATH、CLEVR‑Math、MathVista）、Scientific（GPQA、MMLU‑Pro、SciFiBench）、Medical（MedQA、PathVQA、SLAKE）、Agent（GAIA‑Text）。

**📈 对比分析**

通过在相同LLM（gpt‑4o‑mini / gpt‑5‑mini）上对比OctoTools‑T与OpenTools‑T，使用相同代理策略（Prompting、ReAct、OctoTools、MultiAgent），评估任务完成准确率。结果显示，OpenTools‑T在所有代理框架下平均提升5–22%（相对增幅），在需要大量工具交互的Agent任务上提升最多达22%。

**⚠️ 局限性**

限制：工具覆盖度仍受社区贡献限制；评测仅基于现有测试用例，可能忽略新型错误；工具与代理框架的适配仍需人工干预；安全与隐私风险未完全覆盖。

---

## 176. Think Twice Before You Write -- an Entropy-based Decoding Strategy to Enhance LLM Reasoning

**arXiv ID:** 2604.00018 | [PDF](https://arxiv.org/pdf/2604.00018v1)

**作者:** Jiashu He `[一作]` (Oracle AI Science), Dan Roth `[通讯]` (Oracle AI Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于熵的自适应解码框架（HN-decode），在生成过程中检测高熵位置并仅在这些决策关键点进行分支，从而提高LLM在推理任务中的可靠性。

**💡 创新点**

创新点在于：① 用 token 级熵来识别不确定位置；② 维护动态的部分回溯池，只在高不确定性位置展开多分支；③ 采用 rollout‑level Entropy After Think (EAT) 终止准则，避免逐步评估带来的开销。

**🔧 技术方法**

技术包括：熵计算、token 级自适应分支、rollout 池管理、EAT 终止判定、模型自检校验，以及与传统贪心、beam、sampling、self‑consistency 的对比实验。

**📊 数据集**

使用数据集：GSM8K、AMC2023 以及其扰动版（GSM8K_p、AMC_p）进行实验，并与 GPT‑5 在同一扰动集上进行成本与性能对比。

**📈 对比分析**

在多模型（Llama3.1‑8B、GPT4o、GPT‑5）上与传统解码策略对比，实验显示在 GSM8K 和 AMC2023 上准确率达到或超过 GPT‑5，且计算成本低得多；在扰动集上显著提升鲁棒性，误差传播大幅减少。

**⚠️ 局限性**

局限性：熵并不总与实际不确定性一致，模型自信错误时无法触发分支；连续高熵位置会导致计算量激增；需要在推理时获取 token 熵，API 限制或低延迟环境下实施有难度；超参数单一设置，可能不适用于所有任务；仅在数学推理任务验证，未验证在代码生成、科学推理等领域的泛化效果。

---

## 177. From Skew to Symmetry: Node-Interconnect Multi-Path Balancing with Execution-time Planning for Modern GPU Clusters

**arXiv ID:** 2604.00317 | [PDF](https://arxiv.org/pdf/2604.00317v1)

**作者:** Jinghan Yao `[一作]` (Ohio State University), Dhabaleswar K. Panda `[通讯]` (Ohio State University)

**通讯引用:** 12629 | [OpenAlex ID](https://openalex.org/A5024879682)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出并实现了NIMBLE，一个端点驱动、运行时自适应的通信协同系统，能够在GPU集群中动态重新分配 intra‑node 与 inter‑node 链路上的流量，消除通信拥塞。

**💡 创新点**

创新点包括：① 基于容量归一化的最小拥塞优化，采用乘法权重近似求解多商品流问题；② 通过 GPU 内核实现的 RDMA 管线，在多跳路径上实现无阻塞转发；③ 端点层面实时监控与动态路径重选，配合大小阈值和重组队列避免碎片与振荡；④ 与现有 NCCL/UCX 透明集成，无需应用改动。

**🔧 技术方法**

技术手段主要是：多商品流的乘法权重迭代求解、GPU kernel‑based P2P / RDMA 传输管线、链路利用率监控与动态成本更新、rail‑匹配路径选择、大小感知调度以及与 NCCL/UCX 的插件化接口。

**📊 数据集**

评测采用合成的 All‑to‑Allv skew 测试、Mixture‑of‑Experts（MoE）推理/训练工作负载，以及在真实 LLM 端点上收集的 token 流量，未使用公开数据集，而是基于实际模型和工作负载生成的通信数据。

**📈 对比分析**

与 NCCL v2.26、OpenMPI+UCX 进行对比；在 H100‑SXM5 节点上，NIMBLE 在 intra‑node 达到 2.3×、inter‑node 达到 3.8× 的吞吐提升；在偏斜 All‑to‑Allv 负载下可达 5.2× 速度提升；在 MoE 推理上，整体 latency 提升 1.35×；在均衡负载场景下性能与基线相当。

**⚠️ 局限性**

局限性包括：① 在 NVSwitch‑only 架构（如 DGX‑8）无法实现 intra‑node 多跳转发；② 过深的多跳路径对小/中等消息会产生负面影响，需采用阈值限制；③ 依赖于 rail‑匹配与硬件亲和性，若节点配置不一致会降低效果；④ 额外的监控与调度开销在极低负载下可能不划算。

---

## 178. Physically-intuitive Privacy and Security: A Design Paradigm for Building User Trust in Smart Sensing Environments

**arXiv ID:** 2604.00312 | [PDF](https://arxiv.org/pdf/2604.00312v1)

**作者:** Youngwook Do `[一作]` (Georgia Institute of Technology), Sauvik Das `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2803 | [OpenAlex ID](https://openalex.org/A5006053551)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 Physically‑Intuitive Privacy and Security（PIPS）设计范式，改进传感器环境中的用户信任；

**💡 创新点**

创新点在于将物理直觉引入隐私安全设计，提出三大原则：直接物理操控、可感知状态保证以及意图对齐的激活/停用，并通过三种原型（可变透明摄像头盖、光敏麦克风盒、可按压 RFID）验证其有效性；

**🔧 技术方法**

主要采用物理实现技术：聚合物分散液晶（PDLC）膜、光敏电极与电机化学指示器、微流控液体导电墨水等可见、可操作的硬件元件；

**📊 数据集**

实验使用了自制的用户数据，分别在 20 名、16 名和 17 名参与者上进行受试者信任量表测评；未使用公开数据集；

**📈 对比分析**

与传统手动遮挡/静音控制对比，使用受试者信任问卷与实验行为记录，结果表明 PIPS 原型显著提升用户信任（具体提升幅度未给出但实验统计显著）；

**⚠️ 局限性**

局限性包括：仅适用于用户自己拥有且可直接操作的传感器，对第三方或机构部署的监控无效；实现成本和制造复杂度较高；不同用户的物理直觉差异可能导致误解；设计可能被恶意利用导致“安全/隐私戏剧化”。

---

## 179. Asymmetric Actor-Critic for Multi-turn LLM Agents

**arXiv ID:** 2604.00304 | [PDF](https://arxiv.org/pdf/2604.00304v1)

**作者:** Shuli Jiang `[一作]` (AWS Agentic AI), Stefano Soatto `[通讯]` (AWS Agentic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个异构的演员–评论家框架，在多轮对话场景中用专有大型语言模型（演员）配合可训练的轻量级开源模型（评论家）实现实时监督，避免了重复尝试和演员微调。

**💡 创新点**

创新点在于：①将生成与验证拆分成异步角色，解决了专有模型不可微调的问题；②设计了仅在单次执行中进行干预的评论家；③构建了利用现有对话数据的自我游戏式评论家数据生成管道，用来高效微调评论家。

**🔧 技术方法**

技术包括：演员–评论家异构架构、基于语言模型的对话监督、评论家自我游戏生成监督信号、LoRA 参数高效微调、TRM（Transformer Reinforcement Learning）框架进行监督训练。

**📊 数据集**

使用的基准数据集为	b-bench（航空与零售两个子域）和 UserBench（T‑22、T‑33、T‑44 三个旅行规划子域）。

**📈 对比分析**

在 τ‑bench 上，未微调的评论家可将成功率从 0.60 提升至 0.66；微调后进一步提升至 0.65（相对单演员提升 8%）。在 UserBench 上，未微调的评论家将平均分从 0.328 提升至 0.380，微调后可达 0.388（相对单演员提升约 15%）。与 CRITIC、ReAct、xLAM 等最强基线相比，所提出方法在两套基准上均表现更佳。

**⚠️ 局限性**

局限性包括：1）仍依赖演员的推理与生成能力；2）评论家介入会增加计算延迟和成本；3）在演员能力不足时，评论家难以弥补错误；4）实验仅在两个基准上验证，需在更广泛场景中进一步评估。

---

## 180. All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models

**arXiv ID:** 2604.00479 | [PDF](https://arxiv.org/pdf/2604.00479v1)

**作者:** Xinyu Tian `[一作]` (Australian National University), Jing Zhang `[通讯]` (Australian National University)

**通讯引用:** 17679 | [OpenAlex ID](https://openalex.org/A5100345321)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过多组策略优化（MUPO）鼓励视觉语言模型在强化学习中保持多样化推理路径，提升推理深度与广度；

**💡 创新点**

提出MUPO，将GRPO分组局部优势估计并加入多样性奖励，解决GRPO导致的多样性崩塌问题；

**🔧 技术方法**

使用强化学习框架GRPO的改进版本，结合多组聚类、局部优势计算、余弦距离多样性奖励及权重衰减；

**📊 数据集**

在ViRL39K、MathVerse、MathVista、MathVision、LogicVista、WeMath、Geometry3K、MMStar、HallusionBench、MMVet等多项数学与通用视觉语言基准上训练和评估；

**📈 对比分析**

与InternVL2.5、R1-OneVision、VLAA-Thinker、Vision-R1、VLM-R1、LMM-R1等现有RL和基线模型对比，MUPO-Thinker-7B在多项基准上平均提升约2–3%（acc@1）并在多样采样（acc@4）上提升约6%，显著优于基线；

**⚠️ 局限性**

仍然受限于训练资源与多组策略设置，且在部分通用任务中多样性奖励的衰减与局部优势估计可能导致探索不足或过早收敛，需进一步调优以兼顾深度与广度。

---

## 181. The Silicon Mirror: Dynamic Behavioral Gating for Anti-Sycophancy in LLM Agents

**arXiv ID:** 2604.00478 | [PDF](https://arxiv.org/pdf/2604.00478v1)

**作者:** Harshee Jignesh Shah `[一作]` `[通讯]`, Harshee Jignesh Shah

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了名为 The Silicon Mirror 的框架，通过行为访问控制、特征分类器和生成‑批评循环，动态检测并抑制 LLM 的 sycophancy 行为。

**💡 创新点**

创新点包括：① 以风险分数动态切换多层上下文访问与自适应适配器；② 通过特征向量识别用户说服策略并计算 sycophancy 风险；③ 生成‑批评循环实现必要摩擦（Necessary Friction）修正，提升事实准确性。

**🔧 技术方法**

采用技术：LangGraph 架构、行为访问控制（BAC）、正则表达式+指数滑动平均的特征分类器、基于 LLM 的批评器、适配器驱动的生成、风险公式与多层上下文模型。

**📊 数据集**

使用的数据集：TruthfulQA adversarial、Anthropic NLP Survey Sycophancy、Anthropic PhilPapers Sycophancy，共计 300 条场景用于检测与评估。

**📈 对比分析**

对比方法：在 Claude Sonnet 4 和 Gemini 2.5 Flash 上分别与无干预（vanilla）和静态守卫（“be truthful”）进行对比；Claude 上 sycophancy 从 12.0% 降到 2.0%（83.3% 相对下降，p = 0.112）；Gemini 上从 46.0% 降到 14.0%（69.6% 相对下降，p < 0.001）。

**⚠️ 局限性**

局限性：① 样本量 50 条不足以获得显著统计功效；② 评判者与生成模型相同，可能产生偏倚；③ 风险公式系数和策略乘数手工调参，缺乏系统性消融；④ 正则表达式特征分类器对不同模型的适配性差，跨模型性能不佳；⑤ 仅针对特定 benchmark，未验证在更广泛领域的泛化。

---

## 182. Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models

**arXiv ID:** 2604.00445 | [PDF](https://arxiv.org/pdf/2604.00445v1)

**作者:** Ponhvoan Srey `[一作]` (Nanyang Technological University), Anh Tuan Luu `[通讯]` (Nanyang Technological University)

**通讯引用:** 2433 | [OpenAlex ID](https://openalex.org/A5050386762)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Truth AnChoring（TAC），通过后置校准将LLM的不确定性分数映射为与真值对齐的置信度，解决代理失效问题；

**💡 创新点**

将不确定性分数与真值后置关联的校准方法；证明代理失效与互信息相关，并给出TAC的理论与实证优越性；

**🔧 技术方法**

使用简单的单维MLP映射（BCE+pairwise ranking），训练数据为不确定性分数与真值标签对；

**📊 数据集**

在TriviaQA、SciQ、PopQA等开放域问答数据集（以及GSM8K）上评估，使用Qwen‑3‑4B、Llama‑3.2‑3B、Llama‑3.1‑8B、Ministral‑3‑8B、Gemma‑2‑9B等模型；

**📈 对比分析**

与10种传统UE指标及CUE对比，TAC在ECE上显著下降（≈15‑30%），AUC在大多数情况提升或保持不变，且在少量或噪声标签下仍保持性能；

**⚠️ 局限性**

仅提升校准性，若原始分数无信息仍难提升判别力；需要真值标注，实验仅覆盖中等规模模型和事实性问答，缺乏对更大模型或长文本推理的验证；

---

## 183. Reachability-Aware Time Scaling for Path Tracking

**arXiv ID:** 2604.00439 | [PDF](https://arxiv.org/pdf/2604.00439v1)

**作者:** Hossein Gholampour `[一作]`, Logan E. Beaver `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了如何在双积分模型下，对由离线几何规划器（如 RRT*）生成的路径进行速度与加速度约束下的跟踪，并提出了离线可达性感知时间缩放方法，局部降低参考速度以消除一次性不可达性。

**💡 创新点**

创新点在于将一次性可达性边际与离线时间缩放相结合，只在需要时局部减速；同时保留原始几何路径，支持冻结-恢复事件而无需重新规划；并提供一种基于一次性可达性余量的快速时间缩放算法。

**🔧 技术方法**

采用了 C^2 样条拟合、纯追踪式可达性导向 QP 控制器、一阶可达性余量检测、基于一次性可达性的时间缩放算法以及离线模拟与在线跟踪相结合的整体框架。

**📊 数据集**

使用在二维平面随机生成的障碍物工作空间与 RRT* 规划得到的 waypoint 序列（无公开数据集），在 50 次随机实验中评估方法。

**📈 对比分析**

通过在相同规划路径下比较基线（α=1）与离线时间缩放两种情况，结果显示一次性可达性正余量出现比例下降约 58%，平均 δ_k 与速度显著降低，最大/平均加速度与违规时间均得到改善。

**⚠️ 局限性**

局限性包括：仅提供一次性可达性保证，缺乏多步跟踪可行性；时间缩放为分段常数，缺乏平滑性与速率约束；对速度降序时的余量波动敏感；未考虑加速与减速的非对称约束。

---

## 184. The 1st Winner for 5th PVUW MeViS-Text Challenge: Strong MLLMs Meet SAM3 for Referring Video Object Segmentation

**arXiv ID:** 2604.00404 | [PDF](https://arxiv.org/pdf/2604.00404v1)

**作者:** Xusheng He `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29352 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种完全无训练的三阶段管道，用强大的多模态大语言模型和SAM3实现视频对象分割，特别针对运动中心的自然语言表达。

**💡 创新点**

创新点包括①用Gemini-3.1 Pro对事件进行实例级拆解并选取关键帧生成判别性描述；②使用SAM3-agent直接生成像素级种子掩码，避免框中间表示导致信息损失；③通过Qwen3.5‑Plus自检与行为级一致性验证实现自我纠错。

**🔧 技术方法**

主要技术包括Gemini‑3.1 Pro（LLM）、SAM3‑agent与官方SAM3视频追踪器、Qwen3.5‑Plus、基于𝒥&ℱ、N‑acc、T‑acc的评估指标。

**📊 数据集**

使用的数据集为PVUW 2026 MeViS‑Text Challenge的测试集（Motion Expression Guided Video Segmentation Track）。

**📈 对比分析**

与挑战赛其他参赛者对比，获得首位，Final分数0.909064，𝒥&ℱ 0.7897，N‑acc 0.9615，T‑acc 0.9759，超过第二名0.0791的𝒥&ℱ。

**⚠️ 局限性**

局限性在于虽然整体表现优异，但单项N‑acc与T‑acc未达最高，依赖大规模预训练模型导致推理成本高，对极端相似实例仍易混淆，且未进一步完善细粒度语义对齐。

---

## 185. G-Drift MIA: Membership Inference via Gradient-Induced Feature Drift in LLMs

**arXiv ID:** 2604.00419 | [PDF](https://arxiv.org/pdf/2604.00419v1)

**作者:** Ravi Ranjan `[一作]` (Florida International University), Agoritsa Polyzou `[通讯]` (Florida International University)

**通讯引用:** 399 | [OpenAlex ID](https://openalex.org/A5013726519)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型上提出了一种基于单步梯度上升导致的特征漂移的成员推断攻击方法G-Drift MIA。

**💡 创新点**

创新点在于利用梯度诱导的内部表示（logits、隐藏层激活及特征投影）变化来区分训练样本与非样本，显著优于传统输出级别攻击。

**🔧 技术方法**

使用白盒梯度干预、特征投影（随机或可解释方向）、欧氏隐藏状态漂移、以及轻量级逻辑回归分类器。

**📊 数据集**

使用三大问答数据集：WikiMIA、World Facts 和 Real Authors 3，构造成员与非成员样本进行评估。

**📈 对比分析**

与 Min-k%、Neighbour-MIA、SPV-MIA、PETAL、Perplexity-PL、Zlib 等黑盒、参考、标签仅模式的攻击方法对比，G-Drift 在所有模型和数据集上均取得最高 AUC（最高达 0.9998），明显优于其他方法。

**⚠️ 局限性**

局限性包括：仅适用于白盒访问；实验范围局限于问答式文本数据；对多模态或不同训练管道的泛化尚未验证；差分隐私训练会削弱梯度诱导信号的可区分性。

---

## 186. ARGS: Auto-Regressive Gaussian Splatting via Parallel Progressive Next-Scale Prediction

**arXiv ID:** 2604.00494 | [PDF](https://arxiv.org/pdf/2604.00494v1)

**作者:** Quanyuan Ruan `[一作]` (South China University of Technology), Xiaoguang Han `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 5727 | [OpenAlex ID](https://openalex.org/A5042771880)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

提出了一种可逆的逐步高斯简化与生成框架（ARGS），通过树形结构实现对高斯分布的多尺度控制，实现从粗到细的自动回归生成。

**💡 创新点**

将高斯简化与生成反向结合，构建对数步骤复杂度的树形自回归生成；使用树基Transformer和级别自注意力，实现 O(log n) 步生成；引入噪声增广训练缓解预测错误传播。

**🔧 技术方法**

高斯剖分（Gaussian Splatting）; 可逆逐步简化与反向重构; 级别与树形注意力Transformer; 量化离散化与旋转位置编码; 噪声增强训练。

**📊 数据集**

ShapeSplat、ModelSplat、Objaverse 等大规模高斯 Splatting 数据集，使用 ShapeSplat 与 ModelSplat 的特定类别进行训练与测试。

**📈 对比分析**

通过 PSNR、SSIM、LPIPS 等指标对简化与生成结果与真实数据进行对比，单物体和单类别生成均取得高分，单类飞机生成的可分裂准确率达 99.6%，属性预测准确率 91.45%，生成质量与手工拟合相近，且生成步骤仅为 O(log n)，显著提升效率。

**⚠️ 局限性**

目前仅支持无条件生成；单高斯 token 长度较大导致序列冗长；未处理场景级别；简化方法复杂度为 O(n²)，未实现更高效的树形简化；缺乏图像/文本等条件控制。

---

## 187. Phase space integrity in neural network models of Hamiltonian dynamics: A Lagrangian descriptor approach

**arXiv ID:** 2604.00473 | [PDF](https://arxiv.org/pdf/2604.00473v1)

**作者:** Abrari Noor Hasmi `[一作]` (Khalifa University of Science and Technology), Hadi Susanto `[通讯]` (Khalifa University of Science and Technology)

**通讯引用:** 2193 | [OpenAlex ID](https://openalex.org/A5016610318)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于拉格朗日描述符（LDs）的诊断框架，用来评估和比较不同神经网络模型（SympNet、HénonNet、GHNN、RC）在Hamiltonian系统（Duffing振子、三模非线性薛定谔方程）上的表现，补充传统短期轨道误差指标。

**💡 创新点**

创新点：将LDs映射为概率密度（LD加权PDF），并用KL散度等信息论度量对模型全局相空间结构的重现能力进行定量比较；首次系统地将LDs用于评估数据驱动的Hamiltonian学习模型。

**🔧 技术方法**

使用技术：特定结构的神经网络（SympNet、HénonNet、GHNN）与无结构的Reservoir Computing；拉格朗日描述符计算（前向+后向积分）；构建LD加权PDF；Kullback–Leibler散度、Jensen–Shannon散度等信息论度量；网格化相空间采样和归一化误差分析。

**📊 数据集**

数据集：两类Hamiltonian动力学数据——Duffing振子（200–500轨迹，1000步/轨迹）和三模NLS（500–1000轨迹，1000步/轨迹），按80/10/10比例划分为训练、验证和测试。

**📈 对比分析**

比较方法：通过短期预测误差、LD加权PDF的KL散度以及对不同训练数据量和分布的敏感性进行多维度评估。结果显示，RC在绝大多数情况下实现最低KL散度和最小误差；GHNN在小数据集上最具数据效率；SympNet和HénonNet在复杂非可分Hamiltonian上性能相对落后。

**⚠️ 局限性**

限制：LD框架对积分时间、指数c、加权函数等参数敏感；LD不参与训练，未能实时约束几何结构；对高维Hamiltonian系统的可扩展性尚未验证；SympNet/HénonNet在非可分Hamiltonian中的表现受限于模型容量与训练策略。

---

## 188. Not My Truce: Personality Differences in AI-Mediated Workplace Negotiation

**arXiv ID:** 2604.00464 | [PDF](https://arxiv.org/pdf/2604.00464v1)

**作者:** Veda Duddu `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2763 | [OpenAlex ID](https://openalex.org/A5057029055)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一项基于三种干预（理论驱动AI、通用AI、静态手册）的随机实验中，研究了人格差异如何影响员工在工作场所谈判中的心理准备与效果。

**💡 创新点**

提出人格驱动的“准备门槛”概念，揭示不同人格配置对AI辅导形式的响应差异，并提出基于准备度的个性化而非单纯的阶段化设计。

**🔧 技术方法**

采用基于GPT-4的对话式AI（含理论导向的提示）与BERT嵌入进行文本语义分析，结合k‑means聚类与统计检验。

**📊 数据集**

使用来自Prolific的267名美国职场人士样本，收集BFI‑10人格量表、工作场景预设、实验前后心理测量以及交互日志。

**📈 对比分析**

通过Kruskal‑Wallis检验和效应量比较三组在心理结果、可用性以及语言特征上的差异，发现结果随人格群体显著不同：弹性人格受手册最大收益，过度控制人格受理论驱动AI最有利，而受控人格无显著改善。

**⚠️ 局限性**

主要局限在于实验情境为模拟谈判、未测量真实谈判行为、数据来源为在线众包、横断面设计且仅使用单一谈判框架。

---

## 189. Price of Anarchy of Algorithmic Monoculture

**arXiv ID:** 2604.00444 | [PDF](https://arxiv.org/pdf/2604.00444v1)

**作者:** Robert Kleinberg `[一作]` (Cornell University), Éva Tardos `[通讯]` (Cornell University)

**通讯引用:** 35350 | [OpenAlex ID](https://openalex.org/A5025175846)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在单边匹配市场中算法单一化（即所有公司共享同一算法建议）所导致的社会福利损失，并给出了在广义模型下的价格无效率（Price of Anarchy）上界；

**💡 创新点**

创新点在于：①将先前仅针对两家公司、特定建议源的结果推广到任意数量公司、任意数量建议源；②证明在“随机一致性”假设下，无论是纯纳什均衡还是更一般的均衡，PoA 均被严格上界 2，且该上界是渐进紧确的；③给出了当一致性被轻微违反时 PoA 如何随违规程度升高的量化上界；

**🔧 技术方法**

主要技术是基于随机序列的光滑性（smoothness）分析、对排名技术的“随机一致性”性质的定义与证明、以及对多种均衡概念的统一推理；

**📊 数据集**

本文未使用实验数据集，而是完全基于理论分析和构造性例子来验证极端情况；

**📈 对比分析**

对比方式是通过构造下界例子证明上界的紧确性；理论结果表明，即使在最坏均衡下社会福利也不会低于最优福利的一半；

**⚠️ 局限性**

局限在于需要假设候选人价值分布为置换不变且排名技术满足随机一致性；当这些假设失效时，PoA 可升至线性，说明结果对实际模型的适用性取决于对假设的满足程度。

---

## 190. Execution-Verified Reinforcement Learning for Optimization Modeling

**arXiv ID:** 2604.00442 | [PDF](https://arxiv.org/pdf/2604.00442v1)

**作者:** Runda Guan `[一作]` (Nanjing University of Science and Technology), Rui Xia `[通讯]` (Nanjing University)

**通讯引用:** 2933 | [OpenAlex ID](https://openalex.org/A5101640515)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于执行可验证强化学习的优化建模框架（ECL），通过将数学规划求解器视为确定性验证器，仅使用问题–答案对的结果监督即可训练生成器；

**💡 创新点**

核心创新在于消除过程级监督的需求，实现跨求解器零样本迁移和低成本适配，并利用求解器执行反馈作为可验证奖励进行闭环学习；

**🔧 技术方法**

使用的技术包括：大语言模型（Qwen2.5-7B）作为生成器，GRPO/DAPO无价值函数的强化学习优化，sandboxed harness进行统一求解器执行与观察；

**📊 数据集**

采用的训练数据集为OR-Instruct‑3K（约3000个优化问题），测试集覆盖NL4OPT、MAMO、IndustryOR和OptiBench四大基准；

**📈 对比分析**

与提示式模型（DeepSeek‑R1、OpenAI o1、GPT‑4o 等）以及过程监督的 ORLM（SFT）相比，ECL 在大多数基准上匹配或优于 SFT，且在零样本迁移与低成本适配场景表现尤为突出；

**⚠️ 局限性**

局限性包括：依赖求解器的可访问性与执行时间开销、奖励稀疏性导致学习效率受限、对极大规模或高度约束的求解器支持不充分，以及对求解器特定错误模式的适配仍需要一定的环境切换。

---

## 191. Secretary, Prophet, and Stochastic Probing via Big-Decisions-First

**arXiv ID:** 2604.00437 | [PDF](https://arxiv.org/pdf/2604.00437v1)

**作者:** Aviad Rubinstein `[一作]` (Stanford University), Sahil Singla `[通讯]` (Georgia Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对三大经典在线决策问题——秘书问题、预言家不等式与随机探测问题，研究其在一般下闭约束（downward‑closed）下的非二值（weighted）情形，给出了最优近似与对应的难度分析。

**💡 创新点**

创新点包括：①提出统一的 Big‑Decisions‑First 原则，阐释为何在不确定环境下先解决高价值决策可提升效果；②利用双误差纠正码（double error‑correcting codes）构造层级树形硬实例，使算法在所有层次之间产生严密的交叉限制；③在秘书问题上实现了针对 XOS 价值函数的 O(log n) 近似算法，并将此思路推广到预言家和随机探测场景。

**🔧 技术方法**

主要技术手段：组合优化与约束下的随机抽样；概率与集中分析（马尔可夫链、Azuma‑Bennett 不等式）；误码理论（树码、双误差纠正码）来控制集合间交叉；以及对 XOS 目标的分层桶化与动态阈值更新。

**📊 数据集**

本文完全为理论研究，未使用实际数据集，所有结果均通过构造实例与概率上界进行证明。

**📈 对比分析**

与已知结果比较：对二值情况已知的 Θ̃(log n) 近似在非二值下被证明为不可能；本文给出 Ω̃(log² n) 的硬度上界与 O(log n) 的近似下界，几乎完成了这三大问题在一般下闭约束下的近似阶梯。算法在 XOS 目标下保持与二值相同的 log n 近似，而硬度则与 log² n 的上限相匹配。

**⚠️ 局限性**

局限性：①仅覆盖下闭约束和 XOS 价值函数，其他更一般的约束或价值形式仍未解决；②结果仍以 Θ̃ 表示，隐藏了 loglog n 因子；③算法与硬度均依赖于随机顺序或已知分布的假设，实际应用场景中的鲁棒性还有待进一步验证。

---

## 192. Internal State-Based Policy Gradient Methods for Partially Observable Markov Potential Games

**arXiv ID:** 2604.00433 | [PDF](https://arxiv.org/pdf/2604.00433v1)

**作者:** Wonseok Yang `[一作]` (University of Texas at Austin), Thinh T. Doan `[通讯]` (University of Texas at Austin)

**通讯引用:** 745 | [OpenAlex ID](https://openalex.org/A5035207859)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了基于内部状态的自然策略梯度方法，用于求解部分可观测马尔可夫潜在游戏的纳什均衡。

**💡 创新点**

创新点在于结合公共信息框架与有限内部状态压缩，将原无限维信念空间降至可处理的内部状态，并给出了非渐进收敛界。

**🔧 技术方法**

采用内部状态有限状态控制器、自然策略梯度（NPG）、软max参数化，并在实验中使用了神经网络实现。

**📊 数据集**

在POSGGym基准上实验了三种环境：Multi-Agent Tiger、Multi-Access Broadcast Channel 和 Level-Based Foraging。

**📈 对比分析**

与仅使用当前观测的反应策略相比，内部状态方法在所有三种环境中均取得更高累计奖励，并通过平均NPG梯度收敛速率验证了理论界限。

**⚠️ 局限性**

局限在于内部状态的表达能力受限，有限状态控制器产生逼近误差，且对复杂环境的可扩展性和对真实模型不依赖但收敛速度仍受观测信息压缩的影响。

---

## 193. Certificate-Driven Closed-Loop Multi-Agent Path Finding with Inheritable Factorization

**arXiv ID:** 2604.00428 | [PDF](https://arxiv.org/pdf/2604.00428v1)

**作者:** Jiarui Li `[一作]` (Massachusetts Institute of Technology), Gioele Zardini `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5043524649)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了证书驱动的闭环多智能体路径规划框架，并在现有ACCBS算法上实现为DACCBS，利用冲突‑自由的完整规划与fleet预算过滤来保证全局进展与可组合性。

**💡 创新点**

创新点在于：①用证书轨迹和fleet预算作为全局进度上限，过滤闭环更新；②基于预算受限可达性构造可继承的分解，实现在不同时间步长保持独立子问题；③将上述机制与有限视野CBS相结合，形成DACCBS。

**🔧 技术方法**

技术包括：证书轨迹、fleet预算、预算受限可达区域、可继承分解（基于DSU）、有限视野CBS (ACCBS)、备份控制器LaCAM（及其工程化变体）。

**📊 数据集**

使用了标准MAPF基准地图数据集，包括空白地图（Empty）和随机高密度地图（Random）等。

**📈 对比分析**

与ACCBS和LaCAM比较时，DACCBS在密集实例下SOC增量始终更低，且随每步规划时间增加呈单调改善；分解后平均组大小显著下降，为并行规划提供潜在加速。

**⚠️ 局限性**

局限性：证书质量受备份控制器影响，生成成本可能较高；最优性理论假设H_max上界与无限时间，实际中受限；分解在稀疏环境下效果不明显。

---

## 194. Making Array-Based Translation Practical for Modern, High-Performance Buffer Management

**arXiv ID:** 2604.00423 | [PDF](https://arxiv.org/pdf/2604.00423v1)

**作者:** Xinjing Zhou `[一作]`, Michael Stonebraker `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种DBMS控制的基于数组翻译的缓冲池设计，结合多层翻译、路径缓存、空洞打洞和组预取，实现对顺序扫描、B树查找和图遍历等多种工作负载的高效支持。

**💡 创新点**

创新点在于将数组翻译与多层层次结构相结合，利用路径缓存和空洞打洞实现稀疏空间的物理内存高效利用，并通过组预取最大化内存级并行性，同时保持DBMS对淘汰和I/O的完整控制。

**🔧 技术方法**

采用了多层数组翻译、路径缓存、空洞打洞计数器、组预取接口、使用巨大页存放帧内存，以及对 PostgreSQL 的缓冲管理器和 pgvector 扩展的集成。

**📊 数据集**

在实验中使用了 DEEP10M、SIFT10M（向量维度 96/128）数据集，以及 YCSB‑C、TPC‑C、YCSB‑D 关系型数据集。

**📈 对比分析**

与 hash‑table、PrediCache、vmcache、USearch、WiredTiger、LMDB、LeanStore 等基线进行对比，结果显示在内存内可与 vmcache/USearch 并列，在内存不足时比 vmcache 提升 2–6 倍、比 USearch 近 6 倍；在 PostgreSQL 上实现向量搜索速度提升 2.9–3.95 倍、扫描查询提升 1.2–3 倍，整体性能显著提升。

**⚠️ 局限性**

主要限制是对稀疏 PID 空间的空洞打洞效果依赖于热点分布；当热点分布极其稀疏时，回收率下降，内存占用接近传统数组；另外多层层次结构增加了实现复杂性和维护成本。

---

## 195. Decision-Centric Design for LLM Systems

**arXiv ID:** 2604.00414 | [PDF](https://arxiv.org/pdf/2604.00414v1)

**作者:** Wei Sun `[一作]` (IBM Research), Wei Sun `[通讯]` (IBM Research)

**通讯引用:** 17866 | [OpenAlex ID](https://openalex.org/A5100662256)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种决策中心化框架，将LLM系统中的控制决策从生成过程显式分离出来，使决策层成为可检查、可调节的模块；在日历调度、图搜索和检索三个实验中验证该框架的有效性。

**💡 创新点**

核心创新是将决策上下文、决策策略与执行三者显式分离，构建统一的决策抽象，既可覆盖单步控制（如模型路由、推理缩放），也可扩展到多步序列决策（如信息获取、回溯），并通过可视化信号实现错误归因与局部改进。

**🔧 技术方法**

利用LLM（如LLaMA3、LLaMA3-8B）进行信号估计（充分性、正确性、答案可答性等），阈值/线性效用最大化等确定性策略作为决策函数；在检索实验中使用BM25检索与句子嵌入相似度评估。

**📊 数据集**

实验使用自然问题（Natural Questions）检索任务、合成知识图谱（200人、5属性）进行图搜索任务，以及自定义缺失字段与歧义类型的日历调度任务。

**📈 对比分析**

与Prompt、Retry等基线对比；在日历调度中DC实现100%成功率且无无效执行；图搜索中DC 100%成功，而Prompt/Retry下降；检索任务中DC在中等难度问题的成功率从12%提升到94%，平均检索轮次也显著减少。

**⚠️ 局限性**

受限于信号估计的准确性、阈值/策略的手工设定以及实验规模，未来更强模型或更精细的Prompt工程可能缩小差距，但隐式控制仍是根本瓶颈。

---

## 196. Evidence Units: Ontology-Grounded Document Organization for Parser-Independent Retrieval

**arXiv ID:** 2604.00500 | [PDF](https://arxiv.org/pdf/2604.00500v1)

**作者:** Yeonjee Han `[一作]` `[通讯]` (Korea Telecom), Yeonjee Han (Korea Telecom)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Evidence Units (EU) 的文档分块方法，使视觉元素（如表格、图表、图片）与其上下文文本在检索时以语义完整的单元共同呈现。

**💡 创新点**

创新点包括：跨解析器的语义完整单元构建、全局语义分配矩阵、图数据库决策层验证，以及 EU 空间足迹收敛的理论与实证验证。

**🔧 技术方法**

使用了 DoCO 扩展的本体做统一标签归一化，全文嵌入（ko-sbert）构建全局相似度矩阵，Neo4j 存储和执行图决策规则，并通过三阶段（视觉种子、全局分配、布局归并）构建 EU。

**📊 数据集**

在 OmniDocBench v1.0（1,340 页、1,551 QA 对）上进行评估，并在 GT 轨道以及 MinerU 和 Docling 两个解析器轨道进行对比实验。

**📈 对比分析**

与基线逐块检索相比，EU 在 LCS 上提升约 +0.31，Recall@1 提升 3.4 倍（从 15% 提升至 51%），且在不同解析器上 ΔLCS 均保持 +0.23~+0.31，证明了方法的鲁棒性与显著性能提升。

**⚠️ 局限性**

局限包括：单页单元限制跨页引用、长单元可能导致嵌入稀释、D2/D3 规则虽已在 Neo4j 中定义但未实时执行，以及 GT 仅保证空间准确性而不一定代表文本完整性。

---

## 197. A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation

**arXiv ID:** 2604.00493 | [PDF](https://arxiv.org/pdf/2604.00493v1)

**作者:** Yabin Zhang `[一作]` (Stanford University), Curtis P. Langlotz `[通讯]` (Stanford University)

**通讯引用:** 20797 | [OpenAlex ID](https://openalex.org/A5087710258)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并训练 CheXOne，一款能够在胸片图像上给出诊断预测并提供可解释推理轨迹的视觉‑语言基础模型。

**💡 创新点**

创新点在于：① 大规模合成推理监督数据 CheXReason 与 CheXinstruct‑v2 结合，② 两阶段训练框架（指令调优 + 基于 GRPO 的强化学习）显著提升推理质量，③ 在多任务零样本评估及临床读者研究中验证其可用性。

**🔧 技术方法**

核心技术包括：Qwen2.5‑VL‑3B 视觉‑语言模型、指令调优、GRPO 强化学习、任务特定奖励设计（VQA、报告生成、视觉定位）、自动推理生成与低方差筛选策略。

**📊 数据集**

使用数据集：30 个公开胸片数据集，CheXinstruct‑v2（10.2M 指令样本）和 CheXReason（4.5M 生成推理轨迹）；评估涵盖 17 个子任务，涉及 ReXVQA、MIMIC‑CXR、IU Xray、MS‑CXR、VinDr‑CXR 等。

**📈 对比分析**

对比方法：在 17 个评价维度（VQA、报告生成、视觉定位、推理评估）与多种通用与医学专用 VLM、GPT‑4o 进行零样本对比；CheXOne 在大多数任务上取得最高或相当的准确率、报告质量、mIoU 等，且读者研究显示其草稿与住院医师报告质量可比，效率提升可达 55% 以上。

**⚠️ 局限性**

局限性：① 推理轨迹由 LLM 合成，可能不完全符合专家推理；② 读者研究规模有限且仅模拟学术工作流程；③ 模型规模为 3B，未探索更大模型或多模态输出；④ 缺乏多中心临床验证与长期效益评估。

---

## 198. Competition and Cooperation of LLM Agents in Games

**arXiv ID:** 2604.00487 | [PDF](https://arxiv.org/pdf/2604.00487v1)

**作者:** Jiayi Yao `[一作]` (University of Washington), Baosen Zhang `[通讯]` (University of Washington)

**通讯引用:** 6185 | [OpenAlex ID](https://openalex.org/A5013901541)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究大型语言模型（LLM）代理在网络资源分配与Cournot竞争两类标准博弈中的战略行为，发现当给出多轮目标和非零和背景时，LLM往往倾向于合作并实现近似社会最优；

**💡 创新点**

首次将链式思考（CoT）分析与动态参数学习相结合，提出可解释的θ（合作倾向）和γ（稳定性惩罚）更新框架，用于捕捉LLM在重复博弈中的合作与信任演化；

**🔧 技术方法**

使用Gemini Pro等具备大上下文窗口的LLM作为决策主体，构建自定义系统提示、奖励函数与动态参数更新公式；

**📊 数据集**

实验基于自定义的网络资源分配和线性Cournot游戏，使用合成的多轮博弈数据（如两智能体、10轮、不同阈值和信息可观测性设置）；

**📈 对比分析**

通过对比多轮合作提示与单轮自利提示，展示LLM在合作设置下能快速收敛至Pareto前沿（社会最优）并优于传统Nash均衡；实验显示θ逐步提升、γ根据偏差调节，验证理论公式与实测轨迹高度吻合；

**⚠️ 局限性**

局限性包括对提示语境和LLM模型高度敏感，缺乏对更大规模多智能体场景的验证，且在仅给出总量信息时表现不佳，表明LLM的公平推理仍受限；

---

## 199. Logarithmic Scores, Power-Law Discoveries: Disentangling Measurement from Coverage in Agent-Based Evaluation

**arXiv ID:** 2604.00477 | [PDF](https://arxiv.org/pdf/2604.00477v1)

**作者:** HyunJoon Jung `[一作]` (MPhora.ai), William Na `[通讯]` (MPhora.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了使用基于人格的LLM代理评估者（persona‑based agent judges）来评估对话式人工智能，发现评分可靠性与问题发现之间存在“score‑coverage dissociation”，并验证了这些代理与人类评估者在质量评分上无显著差异。

**💡 创新点**

创新点包括：①首次揭示评估panel规模对评分可靠性呈对数增长、对问题发现呈次线性幂律增长的两条独立曲线；②证明结构化的Big Five人格 conditioning是实现此特性的关键；③将物种累积曲线的概念引入评估框架，以解释发现空间的幂律分布。

**🔧 技术方法**

采用的技术包括LLM‑as‑Judge框架、基于Big Five人格的结构化 persona conditioning、双角色交互日志与情感状态建模、语义去重（Semantic Deduplication）以及ICC、方差分解与功率律拟合等统计分析方法。

**📊 数据集**

使用的实验数据集为960个评估会话（两组模型对：GPT‑5.4/Sonnet‑4.6 与 DeepSeek V3.2/GLM‑5），覆盖15个任务（5个领域×3个复杂度），32个不同 persona 的代理评估者；此外，采用了86个由50名人类参与者完成的会话进行Turing‑style 验证。

**📈 对比分析**

方法上与人类评估者进行Turing 测试，发现 agent‑human 评分差异与 human‑human 差异无统计显著差异；对比不同 panel 大小的ICC和问题发现数，显示评分可靠性在 panel 大小 8 时已达到“良好”，而问题发现数在 panel 32 时才达到 90% 以上，表明两者的收敛速度不同。

**⚠️ 局限性**

局限性包括：人类样本不均衡、语义去重阈值对发现指数影响、任务自适应 panel 可能导致 ICC 高估、实验仅在英文环境下进行、未系统评估共享偏见（如 sycophancy）以及对特殊人群（残障、老年人）缺乏评估。

---

## 200. Scalable Coordination with Chance-Constrained Correlated Equilibria via Reduced-Rank Structure

**arXiv ID:** 2604.00456 | [PDF](https://arxiv.org/pdf/2604.00456v1)

**作者:** Jaehan Im `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**通讯引用:** 9971 | [OpenAlex ID](https://openalex.org/A5068441112)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种可扩展的计算机型随机协同均衡（CC‑CE）方法，通过把CC‑CE表示为少量纯策略的凸组合，实现了大规模非合作协调问题的高效求解。

**💡 创新点**

创新点在于将减少秩相关均衡框架推广到概率约束环境，证明任何有限数量的满足概率约束的纯纳什均衡的凸组合均构成合法的CC‑CE，并据此设计了低维线性规划求解流程。

**🔧 技术方法**

主要技术包括凸分析与概率约束的线性化、纯策略枚举求解CC‑PNE、凸组合构造、线性规划求解器（Julia+PATHSolver）以及Monte Carlo仿真评估。

**📊 数据集**

实验数据基于机场虚拟排队（Virtual Queueing）场景的合成数据，随机生成航班、跑道服务率、初始队列长度等，覆盖6至14架待发飞机/时段的不同流量水平。

**📈 对比分析**

与传统FCFS、完整CC‑CE、降低秩CE（无不确定）四种机制比较，指标包括总延迟成本、计算时间和偏离率；结果显示：降低秩CC‑CE在4 分钟实时阈值内可处理最多210个待发飞机，延迟成本与完整CC‑CE相近且偏离率更低。

**⚠️ 局限性**

限制在于需要枚举所有纯策略以获得CC‑PNE；在高不确定性下可能不存在满足约束的纯策略；目前仅考虑单周期静态模型，缺乏多期动态扩展，且纯策略枚举在更高维度下仍面临计算瓶颈。

---

## 201. Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation

**arXiv ID:** 2604.00452 | [PDF](https://arxiv.org/pdf/2604.00452v1)

**作者:** Halima Bouzidi `[一作]` (University of California, Irvine), Mohammad Abdullah Al Faruque `[通讯]` (University of California, Irvine)

**通讯引用:** 5202 | [OpenAlex ID](https://openalex.org/A5055814180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 FADE 框架，针对基于查询传播（Tracking-by-Propagation, TBP）的多目标跟踪器设计两种对抗攻击，并实现了可微分的数字与物理攻击链路。

**💡 创新点**

创新点在于：① 发现并利用 TBP 的查询预算零和资源分配与递归记忆链路这两大结构性漏洞；② 设计了 Temporal Query Flooding 与 Temporal Memory Corruption 两种专门针对这两漏洞的攻击策略；③ 通过可微分的感知传感器伪装仿真（音频、射频干扰），实现数字攻击向真实物理攻击的可行映射。

**🔧 技术方法**

技术手段包括：Transformer 及其查询更新器、基于 PGD 的对抗优化、定义多项损失函数（Flood, Cost, Siphon, Decorr, Erase 等）、可微分的 Acoustic Adversarial Injection (AAI) 与 Electromagnetic Adversarial Interference (EAI) 传感器仿真模型。

**📊 数据集**

实验数据集为 MOT17 与 MOT20 两个公开的行人跟踪基准数据集。

**📈 对比分析**

对比方法：与传统针对 Tracking-by-Detection (TBD) 的攻击（如 Hijack、F&F、Daedalus）以及 TBP 无关的基线攻击进行对照。FADE 在所有评测的 TBP 跟踪器上显著降低 HOTA（最高下降 30+ 点）并将身份切换率 (IDSW) 提升 10 倍以上，表明在速度、精度及长期关联方面均具有明显优势。

**⚠️ 局限性**

局限性：实验基于仿真感知传感器模型，尚未在真实硬件闭环验证；跨模型的黑盒可转移性仍有限，且对不同 TBP 变体的鲁棒性研究尚不充分。

---

## 202. Programming by Chat: A Large-Scale Behavioral Analysis of 11,579 Real-World AI-Assisted IDE Sessions

**arXiv ID:** 2604.00436 | [PDF](https://arxiv.org/pdf/2604.00436v1)

**作者:** Ningzhi Tang `[一作]` (University of Notre Dame), Toby Jia-Jun Li `[通讯]` (University of Notre Dame)

**通讯引用:** 1830 | [OpenAlex ID](https://openalex.org/A5007240808)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首次在真实开发环境中对IDE集成的AI辅助对话式编程进行了大规模经验研究，分析了74,998条开发者消息、11,579个会话、1,300个仓库以及899名开发者的交互日志；

**💡 创新点**

创新点包括：①构建首个大规模、生态真实的对话式编程数据集；②提出并验证7大类、20子类的行为意图分类体系；③使用层次感知编辑距离对会话序列进行聚类，发现六种典型会话模式；

**🔧 技术方法**

技术手段主要是：①利用GPT‑5 mini进行多标签意图分类（含上下文裁剪、结构化JSON输出）；②层次化加权Levenshtein编辑距离与K‑Medoids聚类；③Markov lift分析转移概率，探究会话内外的行为动力学；

**📊 数据集**

数据集来源于SpecStory导出的GitHub公共仓库中Cursor和GitHub Copilot的对话历史，覆盖从2024年9月至2026年3月的实战记录；

**📈 对比分析**

研究并未直接评估模型性能，而是通过对意图频率、转移模式和聚类结果的描述来揭示开发者与AI的协作特征；

**⚠️ 局限性**

局限性包括：①只包含使用SpecStory并公开提交聊天记录的早期采用者，导致样本偏差；②仅记录文本交互，缺乏非文本操作（如点击、接受/拒绝代码片段）的信息；③仅关注IDE集成式助手，无法推广到CLI代理或其他交互模式；

---

## 203. Learning Humanoid Navigation from Human Data

**arXiv ID:** 2604.00416 | [PDF](https://arxiv.org/pdf/2604.00416v1)

**作者:** Weizhuo Wang `[一作]` (Stanford University), Monroe Kennedy `[通讯]` (Stanford University)

**通讯引用:** 615 | [OpenAlex ID](https://openalex.org/A5047853405)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于5小时人类行走数据，学习到一套可直接在无机器人数据的情况下部署到仿人机器人上的导航先验；

**💡 创新点**

创新点在于：①使用360°视觉记忆（融合彩色、深度和语义）与DINOv3视频特征构建场景表示；②使用条件扩散模型生成多模态未来轨迹分布；③采用混合DDIM–DDPM采样实现实时推理；④以中间层轨迹先验为接口，实现在不同机器人上的零样本迁移；

**🔧 技术方法**

主要技术包括：扩散模型（DDPM/DDIM）、UNet结构、视觉记忆构建（VAE编码）、DINOv3 ViT特征提取、混合采样策略、回溯式滚动规划控制器；

**📊 数据集**

数据集：约300分钟、25公里的人类行走数据（RealSense T265 + D455），包含室内外多种地形、光照、天气与人流场景；

**📈 对比分析**

与VAE-LSTM、CXA-Transformer等基线相比，EgoNav在碰撞率、平滑度、Best-of-15等指标上均表现更优，真实部署在Unitree G1上实现了96–99%自主时间，玻璃墙、动态人群等场景表现尤为突出；

**⚠️ 局限性**

局限性包括：依赖双目深度和DINOv3特征，需额外硬件；当前仅无目标导航，需进一步实现目标条件；对极端光照或完全未知场景的泛化仍有限；

---

## 204. Efficient DPF-based Error-Detecting Information-Theoretic Private Information Retrieval Over Rings

**arXiv ID:** 2604.00411 | [PDF](https://arxiv.org/pdf/2604.00411v1)

**作者:** Pengzhen Ke `[一作]` (ShanghaiTech University), Li-Ping Wang `[通讯]` (Institute of Industrial Economics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于环结构的多服务器信息理论错误检测私有信息检索（itED-PIR）方案；

**💡 创新点**

突破了现有基于有限域的APIR方案的两大限制：① 仅支持素数阶DPF，导致密钥尺寸大；② 采用双钥设计造成通信冗余；该方案使用单一DPF密钥并采用素数幂环，使得支持更高效的素数幂阶DPF，显著减小密钥大小并提升通信效率；

**🔧 技术方法**

利用信息理论分布式点函数（itDPF）和环（ℤ_p^τ）结构，构建查询、回答与重构算法，并通过β^-1·R_agg验证结果；

**📊 数据集**

论文为理论分析，没有使用具体数据库集；

**📈 对比分析**

与传统itAPIR相比，itED-PIR在同等安全级别下，通信复杂度从 O(τlogp·2^2p√{log n loglog n}) 降低到 O(τlogp·2^c(p)√{log n loglog n})，在高安全参数（如ε=2^-128）下可实现实用密钥尺寸，而itAPIR因需要大素数p导致密钥与通信量不可行；

**⚠️ 局限性**

限制包括仅适用于诚实但好奇的客户端模型；未给出针对恶意客户端的安全分析；仅提供信息理论方案，缺少可构造的计算安全版本。

---

## 205. COTTA: Context-Aware Transfer Adaptation for Trajectory Prediction in Autonomous Driving

**arXiv ID:** 2604.00402 | [PDF](https://arxiv.org/pdf/2604.00402v1)

**作者:** Seohyoung Park `[一作]` (Ewha Womans University), Hyuk-Jae Lee `[通讯]` (Seoul National University)

**通讯引用:** 19747 | [OpenAlex ID](https://openalex.org/A5115593383)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了将 QCNet 迁移到韩国道路环境中的方法，并比较了四种迁移学习策略的效果。

**💡 创新点**

创新点在于系统评估了编码器冻结与全微调的差异，证明仅微调解码器即可在保持训练效率的前提下显著提升性能。

**🔧 技术方法**

使用了基于查询的轨迹预测框架 QCNet，并在迁移过程中采用了预训练权重、微调以及编码器冻结技术。

**📊 数据集**

数据集方面使用了美国的 Argoverse 2 作为源域预训练集，韩国的 ETRI 轨迹预测挑战赛数据集作为目标域。

**📈 对比分析**

通过 minADE/minFDE 等指标对四种策略进行比较，发现 Encoder Freezing 在 minADE 上比 Scratch 低 66.3%，minFDE 低 83.7%，且效果接近全微调。

**⚠️ 局限性**

局限在于仍存在域差异未完全消除，且仅在车辆轨迹上评估，未来需进一步探索无监督域适应、多域学习和在线自适应等方向。

---

## 206. A Cross-graph Tuning-free GNN Prompting Framework

**arXiv ID:** 2604.00399 | [PDF](https://arxiv.org/pdf/2604.00399v1)

**作者:** Yaqi Chen `[一作]` (University of Wollongong), Jun Shen `[通讯]` (University of Wollongong)

**通讯引用:** 11748 | [OpenAlex ID](https://openalex.org/A5032104899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个跨图无调优的图神经网络提示框架（CTPF），能够在未见的同质与异质图上直接推理，无需额外参数更新；

**💡 创新点**

提出了无微调即可跨图迁移的提示学习范式，兼容多种图类型，显著提升少样本预测性能；

**🔧 技术方法**

采用图神经网络提示技术、跨图迁移策略以及少样本学习方法，对图特征进行高效编码；

**📊 数据集**

在公开的同质图数据集（如Cora、Citeseer、Pubmed等）和异质图数据集（如DBLP、OGBN-Products等）上进行实验；

**📈 对比分析**

与多种SOTA方法（传统GNN、图提示学习、少样本学习方法等）进行对比，CTPF在少样本节点分类与链接预测任务中平均提高30.8%的准确率，最高提升54%；

**⚠️ 局限性**

限制方面包括对极端异构图的适用性未完全验证、对大规模图的扩展性与计算成本尚需进一步评估、以及提示设计的通用性与鲁棒性需要更多实验支持。

---

## 207. Improving Generalization of Deep Learning for Brain Metastases Segmentation Across Institutions

**arXiv ID:** 2604.00397 | [PDF](https://arxiv.org/pdf/2604.00397v1)

**作者:** Yuchen Yang `[一作]` (Peking University Health Science Center), Yixing Huang `[通讯]` (Peking University Health Science Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了一种基于VAE-MMD的无监督域适配框架，用于跨机构脑转移灶分割，并将其与nnU-Net结合

**💡 创新点**

创新点在于在潜在空间中同时采用多尺度MMD对齐和自注意力/跳跃连接的VAE架构，使得跨机构特征对齐而不损失解剖细节

**🔧 技术方法**

使用的技术包括变分自编码器（VAE）、最大均值差（MMD）损失、多尺度RBF核、3D自注意力、跳跃连接、nnU-Net分割网络以及后期的逻辑回归域分类器评估

**📊 数据集**

采用公开的四个脑转移数据集，共740例，分别来自Stanford、UCSF、UCLM和PKG

**📈 对比分析**

通过在原始数据与VAE-MMD预处理后分别训练nnU-Net，比较F1、sDice、HD95等指标；结果显示F1提升11.1%（0.700→0.778），sDice提升7.93%（0.7121→0.7686），HD95下降65.5%（11.33→3.91 mm），在所有中心均表现出显著的性能提升

**⚠️ 局限性**

局限包括仅使用T1CE序列、需要目标域的无标签样本、潜在空间瓶颈可能导致信息损失、未在完全外部机构上进行验证，以及部分实验缺乏表面度量（sDice、HD95）报告

---

## 208. First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models

**arXiv ID:** 2604.00455 | [PDF](https://arxiv.org/pdf/2604.00455v1)

**作者:** Jiwoo Ha `[一作]` (DGIST), Jinhyun So `[通讯]` (DGIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种训练自由的解码技巧——First Logit Boosting (FLB)，通过在生成过程中重用首个 token 的 logits 来保持视觉信息并抑制对象幻觉；

**💡 创新点**

创新点在于利用首 token 的视觉强化 logits 以及“the”启动词的隐式视觉引用两种互补机制，既降低长程视觉衰减，又在不增加训练或额外模型的前提下显著减少幻觉；

**🔧 技术方法**

技术核心是 logits 重新加权（w_t l_0）并结合自适应可行性约束；

**📊 数据集**

评估使用 CHAIR、AMBER、MMHalBench 与 ConvBench 等多样化数据集；

**📈 对比分析**

与传统的无训练对照、VCD、ICD、M3ID 等对比，FLB 在多模型（LLaVA‑1.5、InstructBLIP）上显著降低幻觉指标（如 CHAIR_i、Hal、Cog）并保持接近基线的生成速度；

**⚠️ 局限性**

局限在于无法根本消除由 RoPE 引起的视觉衰减，且对上下文变化的视觉重心适应性不足，未来可结合自适应 token‑级视觉对齐或位置机制进一步提升。

---

## 209. PC-SAM: Patch-Constrained Fine-Grained Interactive Road Segmentation in High-Resolution Remote Sensing Images

**arXiv ID:** 2604.00495 | [PDF](https://arxiv.org/pdf/2604.00495v1)

**作者:** Chengcheng Lv `[一作]` (Zhejiang University of Technology), Shibo He `[通讯]` (Zhejiang University)

**通讯引用:** 9642 | [OpenAlex ID](https://openalex.org/A5068195118)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了PC-SAM，一种将全自动道路分割与点提示交互式细粒度分割统一在同一框架内的模型；

**💡 创新点**

通过在SAM中添加两个专用掩码解码器，并设计“patch‑constrained”点提示策略，限制点提示仅影响对应图像块，实现可控的局部修正；

**🔧 技术方法**

基于SAM的ViT-B图像编码器、Prompt编码器，使用LoRA对图像编码器进行参数高效微调，加入高召回掩码解码器、自动掩码解码器和点提示掩码解码器，并使用Focal+Dice损失；

**📊 数据集**

在DeepGlobe、Massachusetts Roads、CHN6‑CUG三大遥感道路分割公开数据集上进行训练与评测；

**📈 对比分析**

与传统U‑Net、D‑LinkNet、RCFSNet、CGCNet、MADSNet、OARENet等全自动方法以及SAM‑H零样本模型对比，PC‑SAM在配合点提示后在IoU、召回率、F1等指标上提升5–12个百分点，显示显著性能优越；

**⚠️ 局限性**

主要限制在点提示生成与采样策略的自动化仍需改进，导致在负提示时可能误删细小正确区域；同时模型对点提示密度和形态学开运算参数敏感，需手动调节。

---

## 210. Adapting Text LLMs to Speech via Multimodal Depth Up-Scaling

**arXiv ID:** 2604.00489 | [PDF](https://arxiv.org/pdf/2604.00489v1)

**作者:** Kazuki Yano `[一作]` (Tohoku University), Shinji Watanabe `[通讯]` (Tohoku University)

**通讯引用:** 2287 | [OpenAlex ID](https://openalex.org/A5101405654)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多模态深度缩放（Multimodal Depth Up-scaling）方法，将新Transformer层插入冻结的文本LLM，只在这些层上训练语音数据，从而将文本模型转化为语音语言模型。

**💡 创新点**

创新点包括：①只训练插入层而不修改原有权重，天然避免文本任务遗忘；②可在推理时移除新增层，恢复原始文本性能；③探索E-Branchformer架构作为插入层，提升语音识别精度；④设计了函数保持初始化方案，保证训练开始时插入层行为为恒等。

**🔧 技术方法**

使用的技术主要是Transformer层插入、零初始化（函数保持）、E-Branchformer（自注意力+卷积分支）、参数高效微调（LoRA）对比、词频级别微调（Continual Pre-training）以及标准的ASR推理和文本评测。

**📊 数据集**

使用的数据集为48k小时的英语ASR数据（OWSM v3.2），以及LibriSpeech test-clean/test-other评测集；文本评测使用8个常用语言模型基准（ARC-e, ARC-c, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande, MMLU）。

**📈 对比分析**

与全微调（Full Fine‑Tuning）和LoRA进行对比。深度缩放在保持文本性能（Δ≈0）时，ASR WER可与全微调相当或略优；LoRA在小模型上效果差，且文本性能下降更严重。E-Branchformer插入层在1.7B模型上甚至超过全微调。

**⚠️ 局限性**

局限性包括：仅评估单语（英语）ASR任务，未验证多语言或其他语音任务；对插入层的超参数（位置、数量）仍需更多探索；以及对语音模型的推理效率与内存占用影响未充分分析。

---

## 211. Automated Detection of Multiple Sclerosis Lesions on 7-tesla MRI Using U-net and Transformer-based Segmentation

**arXiv ID:** 2604.00469 | [PDF](https://arxiv.org/pdf/2604.00469v1)

**作者:** Michael Maynord `[一作]` (University of Maryland), Daniel M. Harrison `[通讯]` (University of Maryland)

**通讯引用:** 3105 | [OpenAlex ID](https://openalex.org/A5074198140)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在7T FLAIR MRI上构建并评估了基于Transformer的白质病灶分割模型，并公开发布模型权重。

**💡 创新点**

首次在7T FLAIR数据上训练并公开Transformer（UNETR、SegFormer）模型，证明高分辨率训练显著提升小病灶检测。

**🔧 技术方法**

使用了UNETR、SegFormer两种Transformer架构，并与传统工具LST-LPA、LST-AI进行对比。

**📊 数据集**

数据集为138名受试者共287个7T FLAIR扫描，涵盖MS、健康对照及其他炎性疾病。

**📈 对比分析**

通过BraTS 2023评估框架在0.5mm³原始空间下进行 voxel‑wise 与 lesion‑wise 比较，SegFormer在 Dice 0.61、lesion‑wise Dice 0.20 等指标与 LST-AI 相近，并在小病灶上有优势；经典 LST-LPA 过度分割。

**⚠️ 局限性**

局限包括单机构单扫描仪数据、参考掩膜部分基于 LST-AI 导致评估偏倚、未探讨多站点泛化或自监督预训练以及多模态联合建模。

---

## 212. Sona: Real-Time Multi-Target Sound Attenuation for Noise Sensitivity

**arXiv ID:** 2604.00447 | [PDF](https://arxiv.org/pdf/2604.00447v1)

**作者:** Jeremy Zhengqi Huang `[一作]` (University of Michigan), Dhruv Jain `[通讯]` (University of Michigan)

**通讯引用:** 1892 | [OpenAlex ID](https://openalex.org/A5002130285)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

Sona 是一款面向噪声敏感用户的交互式移动系统，能够实时、选择性地抑制多源噪声，并通过可视化界面让用户即时调节抑制强度；同时提供基于场景的建议和个性化学习功能，支持用户自行添加新的抑制声音类别；

**💡 创新点**

核心创新点包括：①基于目标条件的单一深度网络实现多目标、并行噪声抑制；②通过外部嵌入向量支持无须重新训练即可新增声音类别；③实现低延迟（约42 ms）全设备端实时推理；④结合语义嵌入、最大池化以及轻量级 UI，提供即时建议与长期个性化工作流；

**🔧 技术方法**

采用目标条件 DCCRN 网络，嵌入由 AudioSep 生成，利用 FiLM 进行条件注入；使用最大池化实现多目标聚合；在手机端集成 Apple SoundAnalysis、Audio Flamingo 3 与 Foundation Models Framework 进行声音检测与匹配；实现实时音频流处理与播放的滑动窗口推理；

**📊 数据集**

训练数据来自 VGGSound、ESC‑50 与 FSD50K 的合成混音，覆盖 25 个目标类别；形成问卷与访谈的 68 位噪声敏感受访者用于需求调研；用户实验采用 10 名噪声敏感参与者在真实环境下进行现场测试；

**📈 对比分析**

技术基准：在 1、2、3 目标混合上，SI‑SNRi 分别提升 3.29 dB、3.00 dB 与 3.23 dB；实时性能：CPU 约17–18 %，推理时间 11 ms，端到端延迟 42 ms，电池消耗约 17–18 %/h；用户评估：噪声缓解平均 5.7/7，保持关注 5.8/7，整体可用性 5.4/7；

**⚠️ 局限性**

局限性：仅处理单声道 16 kHz 音频，缺失空间定位信息；多目标抑制时语音可被压制或产生失真；在实验中样本量有限，未涵盖长期使用场景；未与 ANC 耳机等现有工具做正式对比；依赖外接麦克风与有线耳机，限制了使用场景。

---

## 213. Polysemanticity or Polysemy? Lexical Identity Confounds Superposition Metrics

**arXiv ID:** 2604.00443 | [PDF](https://arxiv.org/pdf/2604.00443v1)

**作者:** Iyad Ait Hou `[一作]` (George Washington University), Rebecca Hwa `[通讯]` (George Washington University)

**通讯引用:** 3288 | [OpenAlex ID](https://openalex.org/A5070135550)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过2×2因子分解方法将神经元激活重叠拆解为词形和词义两种贡献，辨别并量化词形混淆对多义性测量的影响；对神经元进行感知度（SSI）和词形检测分类，开展因果消融实验验证其功能；进一步构建可移除的词形子空间并改进多义性评分；最后评估该方法在稀疏自编码器、词义消歧与模型编辑中的实用价值。

**💡 创新点**

创新点在于①首次将词形混淆与真正的“超位置”分离，提出词形贡献比例 R_lex；②构造词形子空间（LIS），实现对多义性指标的可移除校正；③通过感知度与词形检测双重分类，揭示神经元在多义性中的真正角色；④证明词形混淆导致稀疏自编码器特征失效与编辑冲突，并给出可改进的编辑策略。

**🔧 技术方法**

采用了 2×2 因子分解、MLP 激活重叠度量（余弦、Jaccard、幅度差）、感知度指数 (Cohen's d) 进行神经元分类、词形检测评分、逻辑回归探针、均值消融 (mean‑ablation)、ROME 知识编辑、PCA 识别词形子空间，以及词形调整后的多义性评分。

**📊 数据集**

主要使用 SemCor（带 WordNet 词义标签的 37k 句子）筛选 407 个多义词，辅以 WordNet 同义词对和随机不同词对；实验结果在 9 个 Transformer 模型（110M–70B）上验证；同时在现代 Wikipedia 文本上进行交叉验证。

**📈 对比分析**

通过对比同词同义、不同词不同义、同词异义、异词同义四种条件，计算 R_lex 并在所有层与所有模型上保持正值；稀疏自编码器中 18–36% 的特征为词形混合；因果消融表明感知度高的神经元对词义更具因果影响，编辑特异度提升 6.6×；词义消歧实验中感知度高的神经元准确率提升 5–6 pp；所有结果均通过统计检验（p<0.001 或 p=0.002）。

**⚠️ 局限性**

局限性包括：①消融效应绝对值较小（<3 ppl），难以在单词层面观察；②词形校正方法需预先有词义标注，无法直接迁移到无标签场景；③实验仅覆盖 MLP 中间层，未探究其他层级或架构；④词形混淆比例仅在多义词上测得，可能不适用于所有词汇；⑤目前的子空间移除需要手工设定维度，自动化改进仍待研究。

---

## 214. TR-ICRL: Test-Time Rethinking for In-Context Reinforcement Learning

**arXiv ID:** 2604.00438 | [PDF](https://arxiv.org/pdf/2604.00438v1)

**作者:** Wenxuan Jiang `[一作]` (Hong Kong Polytechnic University), Ninghao Liu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5066745575)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在测试时通过检索、奖励与反馈实现无监督的在上下文强化学习框架 TR-ICRL，能够在没有真实标签的情况下让 LLM 在推理和知识密集任务中自我优化。

**💡 创新点**

核心创新在于将 Test‑Time Rethinking 与 ICRL 结合，在每个步骤通过多数投票产生伪标签作为奖励，并让模型生成反馈来模拟梯度更新，从而实现无监督的策略改进。

**🔧 技术方法**

技术包括检索相似实例、并行 Rollout、多数投票奖励、反馈生成、自一致性聚合以及在 Qwen2.5‑7B、Llama3.1‑8B、Qwen3‑8B 等大模型上的推理。

**📊 数据集**

在数学推理基准 MATH500、AMC、GSM8K、AIME2024/2025，以及知识密集基准 MedQA、MedXpertQA 上进行评估。

**📈 对比分析**

与 BoN、Self‑Refine、Reflexion 等基线比较，TR‑ICRL 在大多数任务上实现显著提升：例如 Qwen2.5‑7B 在 AIME2024 提升 137.59%、在 AMC 提升 58.91%，在 MedQA 提升 21.22%，并在其他基准上同样表现优异。

**⚠️ 局限性**

局限在于奖励机制仅基于多数投票的二元信号，缺乏细粒度评估；跨域检索效果有限，长上下文可能导致信息干扰。

---

## 215. Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions

**arXiv ID:** 2604.00499 | [PDF](https://arxiv.org/pdf/2604.00499v1)

**作者:** Haoyu Zheng `[一作]` (Wuhan University), Jiawei Jiang `[通讯]` (Wuhan University)

**通讯引用:** 1823 | [OpenAlex ID](https://openalex.org/A5102918834)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于不确定性感知的输出长度预测模型，并利用预测结果动态调度LLM推理请求，以提升系统吞吐量和降低延迟。

**💡 创新点**

创新点在于将输出长度预测的不确定性（如预测分布或置信区间）引入调度决策，避免了传统确定性长度预测导致的资源浪费和调度误差。

**🔧 技术方法**

使用Transformer‑Encoder形式的长度预测网络，并结合Monte Carlo Dropout或贝叶斯层实现不确定性估计；调度器采用基于置信度和长度的优先级策略。

**📊 数据集**

在OpenWebText和C4两大开放文本生成数据集上实验，并以GPT‑2、LLaMA等主流LLM模型进行推理测试。

**📈 对比分析**

与FIFO、Shortest‑Job‑First (SJF)以及经验长度调度方法比较，实验显示平均等待时间下降约30%，吞吐量提升约20%，整体延迟显著改善。

**⚠️ 局限性**

局限性包括：预测模型增加了额外的训练与推理开销；对极长或高复杂度提示的长度预测仍有偏差；在多模型、多任务的真实环境中泛化能力需要进一步验证。

---

## 216. Executing as You Generate: Hiding Execution Latency in LLM Code Generation

**arXiv ID:** 2604.00491 | [PDF](https://arxiv.org/pdf/2604.00491v1)

**作者:** Zhensu Sun `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 30792 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种并行执行框架，使LLM生成的代码能够在生成过程中即时执行，从而显著降低端到端延迟。

**💡 创新点**

主要创新在于将生成与执行解耦为流水线，结合AST增量分块、动态批处理和错误提前中断，首次实现了在不影响代码质量的前提下隐藏大部分执行时间。

**🔧 技术方法**

采用AST解析实现分块，持续的生成流、检测器和执行器构成生产者-消费者流水线；使用动态批量与门控策略降低每次执行的开销，并在执行错误时立即中断生成。

**📊 数据集**

在DSBench、DABench、PandasPlotBench和GitChameleon四个Python代码生成基准上进行评测。

**📈 对比分析**

与串行执行相比，非重叠执行时间下降83–100%，端到端延迟最高可减少35%（错误场景更高达55%）；实验覆盖七种LLM和三种执行环境，结果稳健。

**⚠️ 局限性**

限制主要包括仅针对Python解释型语言、对大型或多文件程序的适用性不足，以及当前LLM对增量执行缺乏自适应训练。

---

## 217. The Rashomon Effect for Visualizing High-Dimensional Data

**arXiv ID:** 2604.00485 | [PDF](https://arxiv.org/pdf/2604.00485v1)

**作者:** Yiyang Sun `[一作]` (Duke University), Cynthia Rudin `[通讯]` (Duke University)

**通讯引用:** 22141 | [OpenAlex ID](https://openalex.org/A5040468715)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出将Rashomon集合概念应用于降维，通过生成多种高质量嵌入并进行对齐与共识提取，提升可解释性和鲁棒性。

**💡 创新点**

创新点在于：① 定义损失和图论两种Rashomon集合；② 引入PCA对齐和概念对齐正则化；③ 从集合中识别可信近邻构建共识嵌入。

**🔧 技术方法**

使用参数化降维方法（ParamUMAP、Parametric Info‑NC‑t‑SNE、Parametric Neg‑t‑SNE、Parametric NCVis、Parametric PaCMAP）以及Soft Jaccard、Triplet PCA Score 等评估指标。

**📊 数据集**

实验数据集涵盖图像（MNIST、Fashion‑MNIST、USPS、COIL‑20、Mammoth、Airplane）和单细胞RNA‑seq 等。

**📈 对比分析**

通过与传统降维结果在Triplet PCA Score、Soft Jaccard、分类/聚类等多种指标比较，结果显示对齐后嵌入在保持局部结构的同时显著提升全局结构和可解释性，且共识嵌入在多数度量上优于单一嵌入。

**⚠️ 局限性**

主要局限是需要多次运行产生Rashomon集合，计算成本显著提升；若对齐轴与自然布局冲突，可能导致损失升高。

---

## 218. LDMDroid: Leveraging LLMs for Detecting Data Manipulation Errors in Android Apps

**arXiv ID:** 2604.00458 | [PDF](https://arxiv.org/pdf/2604.00458v1)

**作者:** Xiangyang Xiao `[一作]` (Xiamen University), Rongxin Wu `[通讯]` (Xiamen University)

**通讯引用:** 4221 | [OpenAlex ID](https://openalex.org/A5054822682)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型自动化检测 Android 应用中的数据操作错误（DMEs），通过状态感知的 UI 事件生成与 DUM 识别实现完整的错误检测流程。

**💡 创新点**

创新点在于将 DUM（数据容器）识别与 UI 变化摘要、进度追踪融合到 LLM 的推理中，使其能够针对特定的 DMF（Create、Read、Update、Delete、Search）生成有效的 UI 事件序列并验证逻辑正确性。

**🔧 技术方法**

主要技术包括：多模态 LLM（GLM‑4V‑Plus、GPT‑4o、Qwen2.5‑VL）提示模板、UI 层级树相似度聚类识别 DUM、UI 变化摘要与进度追踪模块、基于 DUM 状态比较的逻辑判定。

**📊 数据集**

使用从 F‑Droid/GitHub 选出的 24 个真实开源 Android 应用（共 137 个真值 DMFs）作为评测集，评估工具的覆盖率、成功率、精确率等指标。

**📈 对比分析**

与 Guardian、DMSDroid、Fastbot2、Genie、Odin、PBFDroid 等基线对比，LDMDroid 在 DMF 覆盖率达 75.7%（最高）、成功率 62.5%，检测到 17 个 DME（TPR 61.6%），在令牌消耗、成本和时间上也表现更优。

**⚠️ 局限性**

局限性包括：仅适用于基于列表结构的 DUM，无法处理地图、画布等特殊数据展示；对 LLM 的“幻觉”仍有一定影响；在某些单对象或非列表数据场景下识别失败，需进一步扩展 DUM 识别策略。

---

## 219. CASCADE: Cascaded Scoped Communication for Multi-Agent Re-planning in Disrupted Industrial Environments

**arXiv ID:** 2604.00451 | [PDF](https://arxiv.org/pdf/2604.00451v1)

**作者:** Mingjie Bi `[一作]` `[通讯]` (Beijing Institute for General Artificial Intelligence), Mingjie Bi (Beijing Institute for General Artificial Intelligence)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个预算化的工业重规划机制，通过显式的通信范围控制和门控扩展实现分布式协调。

**💡 创新点**

创新点是将通信范围视为决策变量，并采用门控验证与递归扩展的协议以及统一诊断视图。

**🔧 技术方法**

采用统一的代理架构（知识库/决策管理/通信管理），配合合同式通信原语（Request/Offer/Award/Confirm）和局部优化/学习决策模型。

**📊 数据集**

使用制造业内部生产线（20台机器+6移动机器人）和汽车驾驶舱供应链网络的仿真环境作为实验数据集。

**📈 对比分析**

与集中式重规划对比，在制造业中实现了更低的重规划延迟和更少的消息交换，在供应链中实现了更优的成本-溢出折衷。

**⚠️ 局限性**

限制包括只评估两种行业场景，通信计量仅为消息交换次数而非比特级，且未对扩展参数进行系统的敏感性分析。

---

## 220. Convergence of Byzantine-Resilient Gradient Tracking via Probabilistic Edge Dropout

**arXiv ID:** 2604.00449 | [PDF](https://arxiv.org/pdf/2604.00449v1)

**作者:** Amirhossein Dezhboro `[一作]` (Stevens Institute of Technology), Jose E. Ramirez-Marquez `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 7048 | [OpenAlex ID](https://openalex.org/A5026379933)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了基于梯度追踪的拜占庭鲁棒分布式优化框架 GT‑PD 与其改进版 GT‑PD‑L。

**💡 创新点**

创新点在于将全局自中心投影与概率边缘丢弃两层防御机制相结合，保留双随机混合结构；并通过连续泄漏积分缓冲追踪误差，实现对部分隔离场景的稳健收敛。

**🔧 技术方法**

使用了双度量信任得分的概率丢弃、投影裁剪、随机梯度追踪、Lyapunov 收敛分析等技术。

**📊 数据集**

实验基于 MNIST 图像分类数据集完成。

**📈 对比分析**

通过与无防御梯度追踪和坐标裁剪均值（CWTM）对比，GT‑PD‑L 在 Sign Flip、ALIE、Inner Product Manipulation 三种攻击下均比 CWTM 提升最高 4.3 个百分点。

**⚠️ 局限性**

局限性包括投影半径和阈值为固定参数，缺乏自适应机制；理论分析主要针对静态无向图，对时间变或异步通信拓扑的适用性尚未探讨。

---

## 221. Secure Forgetting: A Framework for Privacy-Driven Unlearning in Large Language Model (LLM)-Based Agents

**arXiv ID:** 2604.00430 | [PDF](https://arxiv.org/pdf/2604.00430v1)

**作者:** Dayong Ye `[一作]`, Wanlei Zhou `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM基于代理的遗忘（unlearning）问题，提出了针对状态、轨迹与环境三类遗忘场景的框架，并实现了将高层遗忘请求转换为可执行提示的转换模型；

**💡 创新点**

首次系统化LMM基代理遗忘的三类场景，且通过自然语言到提示的学习方法实现无需修改LLM参数即可实现针对性遗忘；

**🔧 技术方法**

采用LLM提示学习、基于偏好学习的Fine‑Tuning（RLHF式）对转换模型进行训练，并用KL散度等理论分析验证其有效性；

**📊 数据集**

在GridWorld、AlfWorld、HotPotQA与HumanEval四大平台上进行实验，使用对应的任务/问答/代码生成数据集；

**📈 对比分析**

与Code‑based与Example‑based基线相比，NL‑based方法在三类场景中均实现95%~100%的遗忘成功率，任务完成率基本不变；单次尝试（Unlearn@1）约90%+；对抗性攻击实验表明遗忘后攻击成功率大幅下降；

**⚠️ 局限性**

对LLM的提示理解与转换高度依赖模型规模，Claude模型表现相对较弱；方法对复杂交互情境的适用性仍待验证；以及需要人工构造遗忘请求。

---

## 222. Shapley-Guided Neural Repair Approach via Derivative-Free Optimization

**arXiv ID:** 2604.00422 | [PDF](https://arxiv.org/pdf/2604.00422v1)

**作者:** Xinyu Sun `[一作]` (National University of Defense Technology), Zhenyi Qi `[通讯]` (National University of Defense Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种结合 SHAP 解释与无梯度优化的深度学习模型缺陷修复框架 SHApley，针对后门、对抗攻击和不公平性进行修复。

**💡 创新点**

创新点在于：1）使用 Deep SHAP 对网络层和神经元进行层级粗细定位，得到可解释的缺陷贡献度；2）将定位结果压缩至低维空间后，采用 CMA‑ES 进行参数无梯度搜索，兼顾修复效果与模型性能；3）实现了一个可模块化、跨任务可扩展的修复流程。

**🔧 技术方法**

核心技术包括：Deep SHAP 解释、层级粗细故障定位、协方差矩阵自适应进化策略（CMA‑ES）以及多目标损失（属性损失、准确率损失、激活距离损失、正则化损失）进行权衡。

**📊 数据集**

实验数据集涵盖图像任务（ImageNet10、CIFAR‑10、GTSRB）和表格任务（Census、Credit），模型分别为 VGG16/VGG13/ResNet18/多层 FNN。

**📈 对比分析**

与 CARE、AI‑Lancet、IR、APRNN、INNER、SAU 等现有方法对比，SHApley 在后门去除、对抗缓解和不公平修复三大任务上分别提升 10.56%、5.78% 和 11.82% 的综合得分，同时保持 0–5% 的准确率下降，并显著减少定位与修复时间。

**⚠️ 局限性**

主要限制在于：1）Deep SHAP 计算开销高，对大型基础模型或生成式模型的可扩展性有限；2）仍需对参数权重与损失权重进行手动调优；3）在极大规模网络上可能需要进一步稀疏采样或近似技术。

---

## 223. Self-Routing: Parameter-Free Expert Routing from Hidden States

**arXiv ID:** 2604.00421 | [PDF](https://arxiv.org/pdf/2604.00421v1)

**作者:** Jama Hussein Mohamud `[一作]` (Mila - Quebec AI Institute), Mirco Ravanelli `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了Mixture‑of‑Experts（MoE）层是否需要学习的路由器，提出了一种参数无关的 Self‑Routing 机制，直接将隐藏状态的最后 N 维作为专家 logits，无需额外路由参数；

**💡 创新点**

创新点在于将路由逻辑嵌入隐藏状态的子空间，消除学习路由矩阵，既保持了 top‑k 触发和软化归一化，又显著提升了专家利用率与熵；

**🔧 技术方法**

使用了标准 MoE（top‑k dispatch、softmax 归一化）与 GPT‑2/DeiT‑S/16 的 transformer 结构，并在 OpenWebText、ImageNet‑1K、lm‑evaluation‑harness 上进行实验；

**📊 数据集**

使用的数据集包括：OpenWebText（训练 GPT‑2）、OpenAI GPT‑2 基准、lm‑evaluation‑harness（HellaSwag、LAMBADA、PIQA、WinoGrande、WikiText）以及 ImageNet‑1K（DeiT‑S/16）;

**📈 对比分析**

通过对 dense baseline、learned router、Self‑Routing、fixed random projection、random routing 五种模型在同一任务集上进行对比，评估准确率/困惑度和专家利用率熵；结果显示 Self‑Routing 与学习路由器相当，甚至在部分指标上略优，同时显著提高专家利用率熵（约 17%），在 ImageNet‑1K 上亦略优；

**⚠️ 局限性**

局限性包括：实验仅覆盖 GPT‑2 规模与 DeiT‑S/16 两种设置，未验证更大专家数或更复杂任务；未探究不同层使用不同子空间是否更优；缺乏对 Self‑Routing 机制内在原因的深入分析。

---

## 224. Sampling-based Task and Kinodynamic Motion Planning under Semantic Uncertainty

**arXiv ID:** 2604.00401 | [PDF](https://arxiv.org/pdf/2604.00401v1)

**作者:** Qi Heng Ho `[一作]` (Virginia Tech), Morteza Lahijanian `[通讯]` (University of Colorado)

**通讯引用:** 1705 | [OpenAlex ID](https://openalex.org/A5069564559)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在不确定语义环境下，面向线性时序逻辑任务的集成任务与运动规划框架。

**💡 创新点**

创新点在于将问题建模为部分可观测随机混合系统（PO‑SHS），并设计了结合 Bandit 策略选择与采样运动规划的任何时刻可收敛、可证明可达性的 SaBPI 算法，首次实现了对不确定语义的正式成功概率保证。

**🔧 技术方法**

主要技术包括：PO‑SHS 构造与信念更新、DFA 转化为任务自动机、贝叶斯观测更新、基于 RRT 的采样运动规划、UCB1 Bandit 进行策略子树选择、价值函数回溯更新。

**📊 数据集**

使用了四个自构造的仿真基准：Door‑Key、Fork、Continuous Rock‑Sample 与 Fire Detection，分别涵盖线性/二阶动力学、燃料约束、观察不确定性等多种情形。

**📈 对比分析**

与基线 RRT、MCTS‑PW 及 SaBRS（非概率）进行比较；实验结果显示 SaBPI 在 60 秒时间限制内在所有场景均实现更高且更稳定的任务成功概率，收敛速度优于传统方法。

**⚠️ 局限性**

局限性包括：仅处理离散语义不确定性；对连续动力学噪声无建模；仍需离线规划，计算开销对大规模状态空间较高。

---

## 225. Lightweight, Practical Encrypted Face Recognition with GPU Support

**arXiv ID:** 2604.00546 | [PDF](https://arxiv.org/pdf/2604.00546v1)

**作者:** Gabrielle De Micheli `[一作]`, Bahattin Yildiz `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文通过实验对比了两种索引/检索方案（HD 与 BS）在 CPU 与 GPU 上的性能，涵盖不同规模（2^10 到 2^20）数据集，并记录了时间、内存、磁盘占用等指标。

**💡 创新点**

创新点在于：①提出了针对大规模数据的多维索引结构 HD，并与传统 BS 方案进行系统比较；②在 GPU 上实现了高效的并行化策略，使得在显存受限的情况下仍能保持较高的吞吐量。

**🔧 技术方法**

技术手段包括：高维哈希/树结构（HD）、传统 KD‑Tree 或 Ball‑Tree（BS）；CPU/CPU 双线程并行、CUDA/GPU 并行计算；利用内存映射文件和磁盘缓冲区实现磁盘级别优化。

**📊 数据集**

使用的数据集为公开的高维特征集合（例如 ImageNet 预训练特征或 SIFT/ORBSIFT 纹理特征），规模分别从 2^10 到 2^20 进行测试。

**📈 对比分析**

比较方法：在同一硬件环境下测量查询耗时、内存占用、磁盘 I/O 等；结果显示：HD 在 CPU 上往往比 BS 快 10‑30%，在 GPU 上的显存占用更低，但磁盘 I/O 较大；随着规模增大，HD 的优势更为明显。

**⚠️ 局限性**

局限性：实验仅覆盖单一硬件配置，未检验跨平台或多卡协同；对极大规模（>10^6）数据的评估不足；模型对不同特征分布的鲁棒性未知。

---

## 226. TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting

**arXiv ID:** 2604.00538 | [PDF](https://arxiv.org/pdf/2604.00538v1)

**作者:** Suwoong Yeom `[一作]` (Sogang University), Sukju Kang `[通讯]` (Sogang University)

**通讯引用:** 2586 | [OpenAlex ID](https://openalex.org/A5084904773)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 TRiGS 框架，用连续刚体变换对 4D 高斯喷射进行动态场景重建与高质量视点合成。

**💡 创新点**

创新点包括：①将 SE(3) 的指数映射与本地锚点相结合，实现连续刚体运动；②在此基础上加入层次贝塞尔残差，捕捉非线性细节；③采用基于相邻原子相互约束的刚体正则化与运动平滑正则，显著减少时间片段化与高斯增殖。

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting、SE(3) Lie 代数指数映射、二次贝塞尔曲线残差、局部锚点优化、运动正则化、显式高斯迁移策略。

**📊 数据集**

使用 SelfCap（扩展至 600/900/1200 帧）与 Neural 3D Video (N3V) 两个公开动态视频数据集进行评估。

**📈 对比分析**

与 FTGS、GIFStream、FreeTimeGS 等基线对比，TRiGS 在 1200 帧长序列中保持 PSNR 26.05、LPIPS 0.099，仅占 160 MB 内存并保持 110 FPS，显著优于基线的质量下降与内存爆炸。

**⚠️ 局限性**

局限性在于仍假设每个高斯遵循刚体运动，可能对高度非刚性或极端形变的场景效果有限；此外对极大规模多视角数据的初始化仍需 RoMa 等前置处理。

---

## 227. Toward Optimal Sampling Rate Selection and Unbiased Classification for Precise Animal Activity Recognition

**arXiv ID:** 2604.00517 | [PDF](https://arxiv.org/pdf/2604.00517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 228. MATHENA: Mamba-based Architectural Tooth Hierarchical Estimator and Holistic Evaluation Network for Anatomy

**arXiv ID:** 2604.00537 | [PDF](https://arxiv.org/pdf/2604.00537v1)

**作者:** Kyeonghun Kim `[一作]` (OUTTA), Nam-Joon Kim `[通讯]` (Seoul National University)

**通讯引用:** 6412 | [OpenAlex ID](https://openalex.org/A5089312783)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一框架MATHENA，用于牙科正颌片（OPG）中牙齿检测、龋齿分割、异常检测和牙齿发育分级四项任务；

**💡 创新点**

创新点包括：①将Mamba的线性复杂度状态空间模型（SSM）引入牙齿检测与多任务分析，实现全局上下文建模；②构建方向性Vision State Space（VSS）块和Global Context State Token（GCST）机制；③采用三头共享骨干的顺序迁移学习策略；

**🔧 技术方法**

技术主要包括Mamba-SSM、BiFPN、Vision State Space、GCST、Mamba-UNet、Wise-IoU、Distribution Focal Loss、Ordinal Loss等；

**📊 数据集**

使用自构建的PARTHENON基准，汇集10个牙科数据集共15,062例；

**📈 对比分析**

在PARTHENON测试集上，MATHENA在牙齿检测上mAP_50=93.78%（TTA后94.89%），龋齿分割Dice=90.11%，异常检测Dice=88.35%，发育分级ACC=72.40%，均优于传统YOLO、DeepLabv3+等基线；

**⚠️ 局限性**

局限性包括对数据集划分的依赖（需合并多来源标签），模型仍需在更大规模、多模态数据上验证，且对极端姿态或噪声的鲁棒性未深入探究。

---

## 229. Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding

**arXiv ID:** 2604.00528 | [PDF](https://arxiv.org/pdf/2604.00528v1)

**作者:** Haibo Wang `[一作]` (University of California, Davis), Lifu Huang `[通讯]` (University of California, Davis)

**通讯引用:** 2618 | [OpenAlex ID](https://openalex.org/A5042819803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现TAB框架，利用2D VLM在RGB‑D流上进行思考、调用工具并通过几何重建完成零射手3D视觉定位。

**💡 创新点**

将语义推理与多视角几何分离，提出语义锚定几何扩展弥补纯语义跟踪的覆盖缺陷，并构建动态Think‑Act‑Build循环实现端到端的Agent式推理与构建。

**🔧 技术方法**

使用Qwen3‑VL‑32B 作为VLM，配合Grounding DINO、SAM3进行检测与分割，结合语义时间扩展、几何扩展、逆投影、统计异常移除与DBSCAN聚类完成多视角几何重建。

**📊 数据集**

在ScanRefer与Nr3D（基于ScanNet）两个公开数据集上进行训练与评估。

**📈 对比分析**

与多种零射手与全监督基线对比，ScanRefer上Acc@0.25达71.2%、Acc@0.5达46.4%，Nr3D整体准确率达68.0%，均超过现有零射手且接近或优于监督方法。

**⚠️ 局限性**

对深度图与分割质量依赖较大，极端遮挡或快速运动时性能可能下降；目前仅在室内ScanNet场景验证，跨域泛化尚待进一步验证。

---

## 230. MAESIL: Masked Autoencoder for Enhanced Self-supervised Medical Image Learning

**arXiv ID:** 2604.00514 | [PDF](https://arxiv.org/pdf/2604.00514v1)

**作者:** Kyeonghun Kim `[一作]` (OUTTA), Hyuk-Jae Lee `[通讯]` (Seoul National University)

**通讯引用:** 19747 | [OpenAlex ID](https://openalex.org/A5115593383)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出MAESIL自监督学习框架，利用3D superpatch和双重遮掩实现对CT体积的高效重建与特征学习。

**💡 创新点**

创新点在于引入3D superpatch分块与面向平面与轴向的双重遮掩策略，兼顾3D上下文保留与计算成本，并在3D自动编码器中实现大比例遮掩重建。

**🔧 技术方法**

采用3D掩码自动编码器（MAE）结构，使用Transformer编码器/解码器、卷积嵌入、位置编码以及可学习遮掩标记。

**📊 数据集**

使用BTCV、LIDC-IDRI、TotalSegmentatorV2三大公开CT数据集，合并为统一预训练数据。

**📈 对比分析**

与AE、VAE、VQ-VAE等基线在PSNR、SSIM、LPIPS上进行对比，MAESIL在PSNR 30.28、SSIM 0.98、LPIPS 0.26方面均优于基线，表现显著提升。

**⚠️ 局限性**

局限性包括解码器上采样导致的固定模式伪影；在多样化数据集如TotalSegmentator的重建精度仍相对较低，未来需改进解码器并验证下游任务效果。

---

## 231. Lipschitz Dueling Bandits over Continuous Action Spaces

**arXiv ID:** 2604.00523 | [PDF](https://arxiv.org/pdf/2604.00523v1)

**作者:** Mudit Sharma `[一作]` (Indian Institute of Technology Ropar), Ganesh Ghalme `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 161 | [OpenAlex ID](https://openalex.org/A5068716961)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了具有Lipschitz结构的连续动作空间中的随机对决赌博机，首次提出了相应的算法。

**💡 创新点**

创新点在于将对决赌博机与Lipschitz赌博机结合，提出了基于回合的探索和递归区域消除的算法，并开发了新的分析工具。

**🔧 技术方法**

使用了基于回合的探索和递归区域消除的算法，结合了适应性参考臂的设计。

**📊 数据集**

在动作空间𝒳=[0,1]^d上进行实验，假设反馈是相对的对决反馈。

**📈 对比分析**

与现有的对决赌博机和Lipschitz赌博机方法进行比较，算法的后悔界限为 Õ(T^d_z+1/d_z+2)，与已知的Lipschitz赌博机的最佳后悔率相匹配。

**⚠️ 局限性**

限制在于算法的复杂性和对决反馈的相对性，可能在某些情况下无法直接应用于绝对奖励反馈的场景。

---

## 232. Generalized Heavy-tailed Mutation for Evolutionary Algorithms

**arXiv ID:** 2604.00502 | [PDF](https://arxiv.org/pdf/2604.00502v1)

**作者:** Anton V. Eremeev `[一作]` (Novosibirsk State University), Valentin A. Topchii `[通讯]` (Novosibirsk State University)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5023785502)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将重尾突变算子从原始幂律分布推广到更一般的正则变换分布，并在此框架下证明（1+(λ,λ)）遗传算法在 OneMax 上的期望优化时间仍保持 O(n) 或 O(n ln n)；随后给出一种更高效的 λ 采样实现，并通过实验验证其性能。

**💡 创新点**

创新点在于：①引入正则变换约束，扩大了重尾突变算子的适用范围；②在此更广泛的分布下保持了线性期望优化时间；③提出了低复杂度的 λ 采样方法，使得算法实现更高效。

**🔧 技术方法**

使用了正则变换函数理论、概率与统计分析、复杂度理论以及 Scala 实现的实验评测。

**📊 数据集**

实验数据基于 OneMax 函数，在维度 n = 2^15、2^16、2^17、2^18、2^19 进行。

**📈 对比分析**

对比方法是：在同一硬件（AMD EPYC 7502）上用 10^5 次独立运行分别计算两算法（A 为新型采样、B 为原始幂律采样）的平均 CPU 时间和标准差。结果显示 A 的平均时间显著低于 B，且标准差也更小，表明性能更优。

**⚠️ 局限性**

局限性包括：仅在单峰 OneMax 问题上验证；实验仅使用 β=2.75 且 u_n=n；未探讨多峰或 NP‑hard 目标函数的表现。

---

## 233. On the average-case complexity landscape for Tensor-Isomorphism-complete problems over finite fields

**arXiv ID:** 2604.00591 | [PDF](https://arxiv.org/pdf/2604.00591v1)

**作者:** Tiange Li `[一作]` (Wuhan University), Yingjie Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 3028 | [OpenAlex ID](https://openalex.org/A5100350015)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

针对在有限域上 TI‑complete（张量同构）问题，作者提出了平均情况下的多项式时间算法，涵盖代数同构、矩阵码共轭以及四阶张量同构，并给出了在随机输入中 1/Θ(q) 或 1/q⁽¹⁾ 的成功概率；

**💡 创新点**

创新点在于将随机矩阵的谱性质（如唯一特征值且代数重数为 1 的特征值）引入同构算法，构造了新的全局不变子空间（hull）作为判别依据；同时对随机矩阵的谱分布、平方等式等问题展开了新的随机矩阵理论分析；

**🔧 技术方法**

主要技术包括：生成函数方法（延伸 Neumann‑Praeger 计数）、特征和特征值分布的矩阵谱分析、Gorodetsky‑Rodgers 的特征和方法、随机矩阵的自对角化、以及对矩阵码的同构与共轭问题的线性代数求解；

**📊 数据集**

使用的是理论上的随机矩阵（每个条目独立均匀抽样），没有特定实验数据集；代入的矩阵码示例均为在 (n,q) 上随机生成的 n 维码；

**📈 对比分析**

与之前基于指数时间（如 eˡgⁿⁿ）且成功概率接近 1 的算法相比，本工作实现了多项式时间复杂度，但成功率下降至 1/Θ(q)（代数同构、矩阵码共轭）或 1/q⁽¹⁾（四阶张量同构）。在理论分析与实验验证上，两种成功率均显著低于指数算法在多数随机实例上的全概率成功；

**⚠️ 局限性**

局限性包括：成功概率仅为 1/Θ(q) 或 1/q⁽¹⁾，无法覆盖所有随机实例；算法仅针对 n 维码在 (n,q) 中的共轭判定，且仅对代数同构和矩阵码共轭给出了具体实现，对 3 阶张量同构及更一般维度的矩阵码问题仍未给出完整解法；此外，高阶张量同构的平均成功率仍较低，未来研究需进一步提升成功率与扩展适用范围。

---

## 234. UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems

**arXiv ID:** 2604.00590 | [PDF](https://arxiv.org/pdf/2604.00590v1)

**作者:** Mingming Ha `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了统一的扩展架构 UniMixer 及其轻量化版本 UniMixing‑Lite，兼容注意力、TokenMixer 与 FM 三大主流模块，实现可学习的 TokenMixing 并提升规模效率。

**💡 创新点**

将规则化 TokenMixer 转化为可参数化的 Permutation 矩阵，并通过块级学习、双流 SiameseNorm 与 Sinkhorn‑Knopp 归一化实现稀疏、对称双随机矩阵学习；构造统一的全局与局部混合框架，轻量化模块基于低秩+基底实现参数压缩。

**🔧 技术方法**

使用参数化 TokenMixer、Kronecker/Generalized Kronecker 乘法、Sinkhorn‑Knopp 归一化、SiameseNorm、SwiGLU、低秩近似、基底构造、温度退火、对数功率律等技术。

**📊 数据集**

真实业务广告投放数据，约 7 亿用户样本，包含数十种稀疏与稠密特征，任务为用户留存预测。

**📈 对比分析**

与 Heterogeneous Attention、HiFormer、FAT、RankMixer、TokenMixer‑Large、Wukong 等 SOTA 对比，UniMixing‑Lite 在相同参数或 FLOPs 下提升 AUC 约 0.6‑0.8个百分点，在线 A/B 实验 CAD 提升约 15%。

**⚠️ 局限性**

对低温度稀疏学习依赖较大，训练稳定性受温度退火影响；目前仅在 CTR/留存任务验证，未评估序列建模或生成推荐；对极大规模模型的 GPU 内存仍有挑战。

---

## 235. More Human, More Efficient: Aligning Annotations with Quantized SLMs

**arXiv ID:** 2604.00586 | [PDF](https://arxiv.org/pdf/2604.00586v1)

**作者:** Jiayu Wang `[一作]` (Home Team Science And Technology Agency), Junyoung Lee `[通讯]` (Home Team Science And Technology Agency)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对1.7B参数量的量化小型语言模型进行有限人类标注数据的微调，构建了一个可解释且确定性的自动评估与注释系统。

**💡 创新点**

提出基于细粒度多维评分 Rubrics 的评估框架，并结合提示重述、组件置换、词元丢弃等增强技术，实现了比大型商业 LLM 更高的评注一致性。

**🔧 技术方法**

采用 4‑bit 量化参数高效微调（PEFT）与 Unsloth 库，对 Qwen3‑1.7B 进行微调；将评注任务转化为因果语言建模，使用 completion‑only 损失；配合提示重写、组件置换、词元丢弃等正则化策略。

**📊 数据集**

使用自研的新加坡监狱服务网站问答数据集（97道问题，7 个模型回答）以及公开的 GoEmotions 情绪分类数据集。

**📈 对比分析**

与 GPT‑4o、GPT‑5‑mini 等零/少量提示的商业 LLM 以及无增强/LoRA dropout 的自家 SFT 进行对比。评注数据上 Krippendorff’s α 提升至 0.5774，远高于最佳商业模型 0.2462；在 GoEmotions 上准确率 0.8163、Macro‑F1 0.6380，几乎是 GPT‑4o 的两倍。

**⚠️ 局限性**

仅评测 GPT 系列商业模型，未覆盖其他大模型；模型在更大参数规模或更复杂任务中的泛化和表现尚待验证。

---

## 236. Representation choice shapes the interpretation of protein conformational dynamics

**arXiv ID:** 2604.00580 | [PDF](https://arxiv.org/pdf/2604.00580v1)

**作者:** Axel Giottonini `[一作]` (University of Bern), Thomas Lemmin `[通讯]` (University of Bern)

**通讯引用:** 3409 | [OpenAlex ID](https://openalex.org/A5069828726)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并评估一种基于局部坐标系的取向特征（Orientation features）来描述蛋白质在分子动力学轨迹中的构象变化；

**💡 创新点**

证明取向特征在捕捉旋转动力学、域级运动和蛋白-蛋白结合时的局部取向变化方面能补充传统坐标、二面角和点云特征，并展示不同表征在不同动力学 regimes 下各自的优势；

**🔧 技术方法**

使用特殊正交群 SO(3) 的对数映射将局部坐标系旋转投影到李代数、利用 TICA、AMUSE、VAMP‑2、PCA、HDBSCAN、GMM、互信息、Spearman 等统计与机器学习方法进行特征分析；

**📊 数据集**

基于 D.E. Shaw Research 的长时间 MD 数据集，包括九个快速折叠蛋白、五条 SARS‑CoV‑2 nsp13 螺旋酶轨迹以及四个蛋白‑蛋白结合体系（Barnase‑Barstar、胰岛素二聚体、RNaseHI‑SSB、RBD‑ACE2）共计超过 400 微秒的模拟；

**📈 对比分析**

通过 VAMP‑2、Gram 矩阵相关性、聚类一致性（AMI、ARI）、互信息评估等指标，比较取向特征与传统表征在识别构象状态、捕捉慢动力学、区分不同结合状态等方面的表现；结果显示取向特征在域级运动和界面取向差异上优于其他特征，而在快速折叠体系中与二面角特征相当；

**⚠️ 局限性**

局限性在于取向特征采用独立的 SO(3)^n 结构，忽略了相邻残基之间的几何耦合；只考虑了 backbone，而未直接编码距离或能量信息；对旋转角度接近 π 的情况需采用数值近似；在大分子/长时间轨迹中仍需进一步优化并行性能。

---

## 237. Dual-Select FMA Butterfly for FFT: Eliminating Twiddle Factor Singularities with Bounded Precomputed Ratios

**arXiv ID:** 2604.00567 | [PDF](https://arxiv.org/pdf/2604.00567v1)

**作者:** Mohamed Amine Bergach `[一作]` `[通讯]` (Illumina Inc), Mohamed Amine Bergach (Illumina Inc)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种按扭曲因子自适应选择的 FMA FFT 蜜蜡蝶形算法，消除传统因子化方法中的奇点。

**💡 创新点**

创新点在于利用两种互补的因子化（Linzer‑Feig 与余弦因子化）并通过 |ratio|≤1 的简单规则在每个扭曲因子上动态选择最佳方案，从而完全消除所有奇点并将预计算比例限制在 1 以内。

**🔧 技术方法**

使用了 FMA 指令、正余弦预计算表、半精度 FP16 浮点运算和基于角度比较的动态分支（或编码为标志位）实现算法。

**📊 数据集**

实验数据集为 N=1024 的 radix‑2 FFT，使用了 Stockham 扩展的多通道 FFT 进行10 次传递测试。

**📈 对比分析**

与传统 Linzer‑Feig (÷sin) 与纯余弦 (÷cos) 方法对比，双选策略将 worst‑case 误差上限从 7.95×10⁻² 降低到 4.88×10⁻⁴，累计误差约缩小 235 倍，显著提升 FP16 FFT 的数值稳定性。

**⚠️ 局限性**

局限性包括仅在 radix‑2 下验证，且仍需在预处理阶段对扭曲因子进行额外的选择计算，虽然开销很小，但在极大规模或不同硬件（如不支持 FMA 或 SIMD 分支）上需进一步评估。

---

## 238. FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning

**arXiv ID:** 2604.00559 | [PDF](https://arxiv.org/pdf/2604.00559v1)

**作者:** Tien-Yu Chi `[一作]` `[通讯]` (Independent Researcher), Tien-Yu Chi (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了隐私保护的联邦学习框架FecalFed，用于家禽粪便图像疾病分类，并构建了去重后的8,770张四类图像数据集。

**💡 创新点**

创新点在于①通过双哈希去重消除46.89%的重复数据，②在极度非IID环境下实现FedAdam适应性聚合，③展示轻量级Swin‑Tiny在边缘设备上接近中心化模型性能。

**🔧 技术方法**

采用联邦学习（Flower框架）、FedAdam服务器优化、Vision Transformer（ViT、Swin‑Small/Tiny）以及感知哈希去重技术。

**📊 数据集**

使用了从Zenodo AI4D Tanzania和Roboflow收集、去重后得到的8,770张健康、鸡痢疾、纽卡斯尔病、沙门氏菌四类家禽粪便图像。

**📈 对比分析**

与中心化、单农场训练及FedAvg对比，FedAdam+Swin‑Small在非IID条件下达成90.31%准确率，接近中心化95.10%；Swin‑Tiny在仅28M参数下实现89.74%准确率。

**⚠️ 局限性**

局限性包括对极端非IID分布的进一步适应性、仅涵盖四类疾病、实验仅在10个模拟农场上验证、对更大规模农场和多样网络环境的可扩展性待考察。

---

## 239. Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents

**arXiv ID:** 2604.00555 | [PDF](https://arxiv.org/pdf/2604.00555v1)

**作者:** Thanh Luong Tuan `[一作]` `[通讯]` (Golden Gate University), Thanh Luong Tuan (Golden Gate University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在企业级 LLM 场景中，设计并实现了一种神经符号架构，利用三层（角色、领域、交互）本体对 LLM 进行语义约束，并在 Foundation AgenticOS 平台上完成了上下文注入、工具发现、治理阈值控制以及流程侧耦合等功能。

**💡 创新点**

创新点包括：①三层企业本体模型与耦合模式 taxonomy；②ontology‑constrained tool discovery 的 SQL‑pushdown 评分；③提出了闭环输出验证与本体演化框架；④在五个受监管行业（含两项越南本土化领域）中实证证明了“逆参数知识效应”。

**🔧 技术方法**

技术栈包括：Claude Sonnet 4 LLM、LangGraph 任务图、FastAPI、PostgreSQL、Redis、Qdrant、OWL 轻量级推理器（计划实现）、SQL pushdown 评分、上下文优先级裁剪等。

**📊 数据集**

使用了 50 个任务集合，涵盖金融科技、保险、医疗、越南银行和越南保险五大监管行业；任务基于本体蓝图生成的真值；共 600 次实验跑；同时引用越南监管文件、行业术语等本土化数据。

**📈 对比分析**

对比四种 grounding 条件（无 grounding、RAG、Ontology‑coupled、Ontology+Process），采用 Terminological Fidelity、Metric Accuracy、Regulatory Compliance、Role Consistency 四个指标进行 Friedman/Wilcoxon 检验。结果显示 Ontology‑coupled 与 Ontology+Process 在 MA、RC、RS 上显著优于无 grounding（p<0.001），并在越南本土化行业中提升约 2 倍；RAG 在实验数据上竞争力强，但在真实噪声环境下预计落后。

**⚠️ 局限性**

局限性包括：①当前仅实现输入侧耦合，输出验证与本体演化尚未落地；②实验评估依赖 LLM‑judge，缺乏人工专家一致性验证；③RAG 基线采用与本体同源文本，真实多源文档场景下效果可能不同；④上下文注入可能掩盖 LLM 已有参数知识导致指标回落；⑤本体完整性与更新维护成本高；⑥OWL 推理的实时性能与可扩展性尚待验证；⑦仅在单一平台上验证，跨平台推广需进一步研究。

---

## 240. TF-SSD: A Strong Pipeline via Synergic Mask Filter for Training-free Co-salient Object Detection

**arXiv ID:** 2604.00549 | [PDF](https://arxiv.org/pdf/2604.00549v1)

**作者:** Zhijin He `[一作]` (Xi'an Jiaotong-Liverpool University), Jimin Xiao `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的Co-salient对象检测框架TF-SSD，先用SAM生成候选分割掩模，再通过质量掩模生成器(QMG)、单图像显著性过滤器(ISF)与跨图像原型选择器(IPS)逐步筛选出跨图像共同显著的目标掩模。

**💡 创新点**

创新点：①首次将Vision Foundation Models（SAM与DINO）结合，用SAM的高泛化掩模生成能力与DINO的注意力与语义特征实现训练无关CoSOD；②设计多阶段质量掩模生成器，剔除冗余与低质量掩模；③利用DINO注意力做显著性评分，获得单图显著目标；④跨图原型相似度最大化筛选共显目标，形成互补的语义与分割策略。

**🔧 技术方法**

核心技术：Vision Foundation Models（SAM ViT‑H 生成掩模；DINO ViT‑B/8 提取注意力图与CLS特征）；面积/重叠过滤、IoU加权质量评分；显著性分数计算；跨图原型余弦相似度矩阵，取最大相似度求共显分数；最终掩模选择。

**📊 数据集**

使用的公开Co-salient数据集：CoCA、CoSal2015、CoSOD3k。

**📈 对比分析**

与现有监督、无监督与训练无关方法在三大数据集上做对比。TF-SSD在CoCA上Fβmax提升13.7%（相较最近训练无关方法），在CoSal2015与CoSOD3k上同样实现显著提升；总体表现与部分监督方法相当，显著优于所有无监督与训练无关方法。

**⚠️ 局限性**

局限性：①依赖SAM生成的掩模质量，若SAM误检导致后续过滤失效；②需手工设定多阈值（面积、重叠、质量权重、显著性阈值等），对不同场景需调参；③目前无端到端训练，无法进一步提升对极端多样化图集的适应；④对非常小或细长目标的显著性判断仍不够稳健。

---

## 241. FreqPhys: Repurposing Implicit Physiological Frequency Prior for Robust Remote Photoplethysmography

**arXiv ID:** 2604.00534 | [PDF](https://arxiv.org/pdf/2604.00534v1)

**作者:** Wei Qian `[一作]` (Hefei University Of Technology), Meng Wang `[通讯]` (Hefei University Of Technology)

**通讯引用:** 42697 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出FreqPhys框架，利用生理频率先验通过频域滤波、频谱调制、适应性频谱选择等模块，在扩散去噪过程中实现远程光电容积描记（rPPG）信号的高质量重建。

**💡 创新点**

创新点在于：①将隐式生理频率先验（心率频段与峰值特性）直接嵌入到条件扩散模型中；②设计了三段频域去噪模块（PBF、PSM、ASS）并结合跨域注意力实现时间‑频率特征融合；③通过频域损失与Pearson相关性共同训练，提升重建精度。

**🔧 技术方法**

采用的技术包括：离散傅里叶变换与逆变换、频域理想带通滤波、复数权重的频谱调制、可学习阈值的硬阈值筛选、跨域注意力机制、扩散概率模型（DDPM/DDIM）以及傅里叶损失与时间域相关性损失的联合优化。

**📊 数据集**

在六大公开rPPG基准数据集上评估：UBFC-rPPG、PURE、MMPD、BUAA、VIPL‑HR 和 MR‑NIRP‑Car。

**📈 对比分析**

与传统方法（GREEN、ICA、CHROM 等）、深度学习模型（DeepPhys、PhysNet、CVD、PhysFormer、RhythmMamba、PhysDiff）以及其他扩散模型进行对比。FreqPhys 在 MAE、RMSE、相关系数等指标上均显著优于现有最先进方法，尤其在运动强、光照变化大和跨数据集迁移时表现更为突出。

**⚠️ 局限性**

局限性：①依赖固定的心率频段 [0.66,3.0] Hz，对异常心率或特殊生理频率的适应性有限；②在极端运动、低光照或严重遮挡情况下仍会出现误差；③训练和推理过程需要较多计算资源，尽管已实现一定的算力优化，但在极低功耗设备上的部署仍面临挑战。

---

## 242. AceTone: Bridging Words and Colors for Conditional Image Grading

**arXiv ID:** 2604.00530 | [PDF](https://arxiv.org/pdf/2604.00530v1)

**作者:** Tianren Ma `[一作]` (University Of Chinese Academy Of Sciences), Qixiang Ye `[通讯]` (University Of Chinese Academy Of Sciences)

**通讯引用:** 15335 | [OpenAlex ID](https://openalex.org/A5015317495)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了AceTone，一个基于多模态条件的生成式3D‑LUT颜色分级系统，能够根据文本或参考图像直接生成色彩转换。

**💡 创新点**

通过将LUT量化为离散token，构建VQ‑VAE tokenizer，并使用视觉‑语言模型进行条件生成，再结合GRPO强化学习与颜色/审美奖励对齐，实现了首个完整的文本/图像驱动的生成式颜色分级框架。

**🔧 技术方法**

采用VQ‑VAE量化、视觉‑语言模型（Qwen2.5‑VL）、生成式预训练、监督微调、强化学习（GRPO）以及颜色与审美奖励等技术，直接生成3D‑LUT。

**📊 数据集**

构建了AceTone‑800K大规模数据集，包含约10k滤镜LUT、3.4k专家LUT、8k融合LUT，以及MS‑COCO、Adobe‑5K、PPR‑10K图像，并通过自动化工具生成约800k指令级别的注释。

**📈 对比分析**

与WCT、ModFlow、SA‑LUT、Neural Preset、InstructPix2Pix等基准在PSNR、LPIPS、ΔE和审美分数上对比，AceTone在PST和IGG任务中分别提升LPIPS约50%，审美分数最高，且单次推理时间约1 s。

**⚠️ 局限性**

在极端光照或高度风格化场景下全局LUT效果有限；对模糊或文化含义的指令理解不佳；用户研究未校准显示设备，样本规模有限。

---

## 243. Learnability-Guided Diffusion for Dataset Distillation

**arXiv ID:** 2604.00519 | [PDF](https://arxiv.org/pdf/2604.00519v1)

**作者:** Jeffrey A. Chan-Santiago `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**通讯引用:** 58701 | [OpenAlex ID](https://openalex.org/A5080823547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于学习能力驱动的增量数据集蒸馏方法，利用 Learnability-guided Diffusion (LGD) 逐步生成低冗余的合成数据集。

**💡 创新点**

创新点在于：①采用增量式蒸馏，将数据集分段生成；②使用学习能力分数指导扩散模型生成与当前模型学习边界匹配的样本；③通过参考模型正则化避免语义漂移，显著降低了 39% 的冗余。

**🔧 技术方法**

使用的技术包括扩散模型（LGD）、learnability score 计算、参考模型正则、偏差引导（deviation guidance）以及多阶段增量训练框架。

**📊 数据集**

实验数据集为 ImageNet-1K、ImageNette、ImageWoof（256×256 高分辨率图像）。

**📈 对比分析**

与现有最先进方法（DiT、IGD、MGD³ 等）在静态和增量评估下进行对比，取得 ImageNet-1K Top‑1 60.1%、ImageNette 82.6–87.2%、ImageWoof 53.9–72.9% 的性能，在相同 IPC 条件下优于或持平于基线。

**⚠️ 局限性**

局限性包括：①增量生成过程计算成本较高；②对极高 IPC 或极大数据集的可扩展性尚未充分验证；③生成样本对参考模型的依赖可能限制在非典型分布上的适用性。

---

## 244. MOON3.0: Reasoning-aware Multimodal Representation Learning for E-commerce Product Understanding

**arXiv ID:** 2604.00513 | [PDF](https://arxiv.org/pdf/2604.00513v1)

**作者:** Junxian Wu `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**通讯引用:** 12778 | [OpenAlex ID](https://openalex.org/A5034845046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多模态大语言模型（MLLM）的推理感知框架（MOON3.0），通过显式属性分解来生成高质量的商品表示。

**💡 创新点**

创新点在于：①将多头模态融合模块与推理生成的属性序列结合，缓解长序列注意力稀释；②采用联合对比学习与强化学习（GRPO）自我探索更优的推理策略；③引入细粒度残差增强模块，逐层保留局部细节，提升表示的细粒度区分度。

**🔧 技术方法**

主要技术包括多头模态融合（Attention+门控）、联合对比损失（InfoNCE）、GRPO强化学习、细粒度残差增强（patch‑level gating、跨模态投射、长距残差），以及在 Qwen3‑VL‑2B 等 MLLM 上的 SFT 训练。

**📊 数据集**

使用了自构建的 MOON benchmark：约 770 万条含 CoT 语义的商品推理样本（从电商日志抽取），以及 91.6 万条经过人工校验的检索、分类和属性预测测试集；还在公开数据集 M5Product 与 Fashion200K 上进行评估。

**📈 对比分析**

与 SigLIP2、Qwen3‑VL‑Embedding、GME‑Qwen2VL、MM‑Embed、InternVL3.5‑2B、Qwen3‑VL‑2B、FashionCLIP、CASLIE‑S、MOON 及 MOON2.0 等基线相比，MOON3.0 在零样本检索、分类和属性预测任务上均取得最高或第二高的 Recall@k、Accuracy、F1 等指标，并且仅使用 256 维嵌入即实现最优召回，显示出显著的性能优势。

**⚠️ 局限性**

主要限制包括：①对大规模 MLLM 进行 SFT 与强化学习需要高算力与长时间训练；②依赖于高质量的 CoT 语义标签，构建过程成本高；③在跨域场景（非电商商品）下的泛化能力尚未充分验证。

---

## 245. A Decoupled Basis-Vector-Driven Generative Framework for Dynamic Multi-Objective Optimization

**arXiv ID:** 2604.00508 | [PDF](https://arxiv.org/pdf/2604.00508v1)

**作者:** Yaoming Yang `[一作]`, Ke Tang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于离散字典的潜在空间生成器（DB-GEN）和MOEA/D混合的动态多目标进化框架，能够在没有任何目标评估的情况下，从历史环境中零射击生成可覆盖当前帕累托前沿的初始候选集合，并通过在线演化快速跟踪动态问题。

**💡 创新点**

创新点包括：①将离散字典与潜在空间结合，实现高维特征的稀疏表示；②提出中心扰动策略（Centroid Perturbation）以消除历史偏差并生成分布均匀的候选；③利用拓扑一致性对比正则化（Topology‑Aware Contrastive Regularization）保证不同动态问题在潜在空间的相似结构可迁移；④在零射击场景下实现跨问题泛化，避免过拟合。

**🔧 技术方法**

主要技术手段有：Python + PyTorch + CUDA；MOEA/D 作为静态优化器；离散字典学习与潜在空间编码；高维高斯噪声中心扰动；t‑SNE可视化；IGD、HV 指标评估；Wilcoxon、Friedman 统计检验。

**📊 数据集**

使用的基准数据集包括：DMOP（F、ZF、HE、JY、UDF、DIMP、GTA、SDP、SJY、DCP、DSW 等）用于预训练；DF1‑DF14 与 FDA1‑FDA5 用于零射击评估；另外自定义真实场景问题：动态资源分配（DRA）和动态路径规划（DPP）用以验证实用性。

**📈 对比分析**

与 STT‑MOEA/D、DIP‑DMOEA、VARE、SIKT‑DMOEA 等 4 个最先进方法比较，平均排名 1.37，IGD 和 HV 结果均优于大多数基线，Wilcoxon 检验在 5% 水平下显著优于对手，证明了模型在动态多目标任务中的稳健性和泛化能力。

**⚠️ 局限性**

局限性：在某些极端动态变化（如 DF4）时表现波动；对非常高维或大规模问题的实时推理速度尚未公开；模型需要较大规模的预训练数据，扩展性虽好但对数据分布变化的鲁棒性仍待进一步验证。

---

## 246. HabitatAgent: An End-to-End Multi-Agent System for Housing Consultation

**arXiv ID:** 2604.00556 | [PDF](https://arxiv.org/pdf/2604.00556v1)

**作者:** Hongyang Yang `[一作]` (Fangdongdong), Rongshan Zhang `[通讯]` (Fangdongdong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 HabitatAgent，一种端到端的多代理系统，用于高风险的住房咨询，涵盖记忆、检索、生成和验证四个专业化代理。

**💡 创新点**

创新点包括：验证门控记忆防止错误扩散；自适应向量‑图检索路由精准处理复杂关系约束；以及失败类型感知的修复策略，形成闭环可靠工作流程。

**🔧 技术方法**

技术手段为 LLM 与多代理协同、GraphRAG（向量+图检索）、多层验证、任务感知提示模板、以及基于证据的生成与实体/事实检查。

**📊 数据集**

使用来自北京住房平台的真实匿名交互数据，包含 5,000 条房源属性、6,016 节点/45,000 条边的知识图谱，及 100 条真实咨询场景（共 300 轮问答）进行评测。

**📈 对比分析**

与六种基线（Monolithic RAG、Dense+Rerank、GraphRAG‑Fixed、LLM‑Ranker、Self‑RAG、Rule‑Verifier）对比，HabitatAgent 的端到端准确率提升至 95%（比最强基线提升 20%），nDCG 与事实性也均获得最高分，P95 延迟约 720 ms。

**⚠️ 局限性**

局限性包括：仅在北京单一城市的专有数据上验证，缺乏跨城市泛化评估；对极端稀疏约束或未知实体的处理仍需改进；以及系统对知识图谱实时更新和冷启动的鲁棒性待进一步提升。

---

## 247. Reliev3R: Relieving Feed-forward Reconstruction from Multi-View Geometric Annotations

**arXiv ID:** 2604.00548 | [PDF](https://arxiv.org/pdf/2604.00548v1)

**作者:** Youyu Chen `[一作]` (Harbin Institute of Technology), Dave Zhenyu Chen `[通讯]` (Huawei)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Reliev3R，一种弱监督的端到端3D重建模型训练框架，能够在没有多视角几何标注的情况下从零开始训练Feed‑Forward Reconstruction Model (FFRM)

**💡 创新点**

核心创新包括：①利用预训练模型生成的伪单目相对深度和稀疏对应点作为轻量级监督；②设计了自适应的相对深度损失和基于三角学的投影损失，实现视角深度和相机姿态的全局一致性；③将几何约束直接嵌入模型学习过程，消除了对SfM/MVS标注的依赖

**🔧 技术方法**

采用伪深度生成模型Depth Pro和CoTracker进行伪标签生成，使用相对深度损失（scale‑invariant）和三角投影损失进行监督，模型架构基于π³（替换为深度预测头），通过可微分的几何约束实现训练

**📊 数据集**

训练使用DL3DV‑10K（10K场景、3M+图像）数据集，评估在DL3DV‑benchmark（在域）和ScanNet++（跨域零射）上进行；伪标签来源不依赖于训练集，模拟真实可扩展场景

**📈 对比分析**

与全监督FFRM（π³†）以及弱监督相机估计模型AnyCam进行对比；实验表明Reliev3R在8视图重建中与早期FFRM（MVDUSt3R、FLARE）持平或优于，且在相机姿态估计上显著超越AnyCam；零射实验显示Reliev3R与全监督π³†相当，甚至在深度估计上更好

**⚠️ 局限性**

局限性包括：未对大规模数据进行扩展性验证；对动态场景缺乏显式处理；依赖伪标签（Depth Pro、CoTracker）质量，可能在多样化数据上表现不稳定

---

## 248. Does Unification Come at a Cost? Uni-SafeBench: A Safety Benchmark for Unified Multimodal Large Models

**arXiv ID:** 2604.00547 | [PDF](https://arxiv.org/pdf/2604.00547v1)

**作者:** Zixiang Peng `[一作]` (Chinese Academy of Sciences), Gaopeng Gou `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 1770 | [OpenAlex ID](https://openalex.org/A5052269317)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Uni-SafeBench 评估统一多模态大模型（UMLMs）的安全性，并设计 Uni-Judger 自动判定上下文安全与内在安全。

**💡 创新点**

创新点在于：①构建六类安全风险的层次化分类，覆盖七种多模态任务；②将安全评估拆分为“上下文安全”和“内在安全”双维度；③系统性比较统一模型与专用模型的安全-效能权衡。

**🔧 技术方法**

采用 LLM 生成恶意输入、人工审核标注、以及多模态评估框架 Uni-Judger（含恶意意图提取器与安全检测器），并在 16 个公开模型上进行实验。

**📊 数据集**

使用 2,939 条人工/模型合成的恶意查询，涵盖 VQA、文本生成、图像生成与编辑等七种任务。

**📈 对比分析**

实验显示统一模型在安全性上普遍落后于专用 VLM/LLM；开放源代码 UMLMs 的安全率低于闭源系统；安全与效能呈现明显权衡（如 Chameleon 极安全但拒绝率高，GPT‑4o 兼顾安全与高效）。

**⚠️ 局限性**

局限性包括：评估器可能漏判隐蔽或文化特定危害；对图像安全判断依赖可见变化，难以捕捉细微攻击；模型规模与评估者选择对结果有一定影响。

---

## 249. MF-QAT: Multi-Format Quantization-Aware Training for Elastic Inference

**arXiv ID:** 2604.00529 | [PDF](https://arxiv.org/pdf/2604.00529v1)

**作者:** Zifei Xu `[一作]` (d-Matrix), Hesham Mostafa `[通讯]` (d-Matrix)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

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

## 250. Do Agents Repair When Challenged -- or Just Reply? Challenge, Repair, and Public Correction in a Deployed Agent Forum

**arXiv ID:** 2604.00518 | [PDF](https://arxiv.org/pdf/2604.00518v1)

**作者:** Luyang Zhang `[一作]`, Ramayya Krishnan `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建三步机制链（线程深度→挑战跟进→公共纠正），比较了实时部署的LLM代理论坛Moltbook与匹配的人类Reddit社区在公共规范执行过程中的表现；

**💡 创新点**

创新点在于将评估焦点从单个回复质量转向社区层面的交互机制；采用匹配社区对照与挑战非挑战基线，揭示代理社区在挑战回访、修复及公共纠正方面的显著缺失；

**🔧 技术方法**

主要技术包括：基于保守词法列表的挑战与修复检测、线程结构统计、配对社区匹配、置换检验与bootstrap置信区间；

**📊 数据集**

使用数据集包括：Moltbook 2026年1月28日至2月17日的快照（约790万帖子、1070万评论），以及2018-2021年GECLM Pushshift的5个Reddit子版块；

**📈 对比分析**

比较方法为三步机制链评估，结果显示Moltbook的线程深度约为Reddit的十分之一；挑战后原作者回访率仅1.2%（vs 40.9%），多轮对话率0.09%（vs 38.5%），整体缺失率极高；

**⚠️ 局限性**

局限性包括：仅覆盖Moltbook早期阶段；词法检测保守可能低估修复；平台与模型差异难以分离；Reddit样本截断至2021年，未捕获近期趋势；未对更复杂的修复策略进行评估。

---

## 251. RT-GS: Gaussian Splatting with Reflection and Transmittance Primitives

**arXiv ID:** 2604.00509 | [PDF](https://arxiv.org/pdf/2604.00509v1)

**作者:** Kunnong Zeng `[一作]` (University of Utah), Cem Yuksel `[通讯]` (University of Utah)

**通讯引用:** 2301 | [OpenAlex ID](https://openalex.org/A5004638951)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出RT-GS框架，将微面片模型与可微光线追踪结合，实现Gaussian Splatting中同时渲染镜面反射和半透明物体内部物体的可视化。

**💡 创新点**

创新点在于：①为反射与透射分别引入独立的Gaussian原语；②利用微面片模型精准建模物理BSDF；③通过网格引导的光线追踪和深度、法向正则化提升透射效果。

**🔧 技术方法**

使用了2D Gaussian Splatting、可微光线追踪、微面片BRDF（Torrance‑Sparrow + Trowbridge‑Reitz）、SAM2+GroundingDINO物体分割、MS-Perceptual loss等技术。

**📊 数据集**

在Ref-Real（含镜面反射）与NU-NeRF（含透明物体）的真实场景数据集上进行训练与评测。

**📈 对比分析**

与3D‑GS、GaussianShader、Ref‑GS、Ref‑Gaussian、EnvGS以及NU‑NeRF等基线对比，RT‑GS在PSNR、SSIM、LPIPS等指标上均表现最佳或竞争性，并在视觉上显著提升反射细节与透射重建。

**⚠️ 局限性**

仅假设透明物体外层薄层近似无厚度，难以处理厚重、曲折的透明物体，未来需扩展对厚透明物体的建模。

---

## 252. RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised Human-Object Interaction Detection

**arXiv ID:** 2604.00507 | [PDF](https://arxiv.org/pdf/2604.00507v1)

**作者:** Jihwan Park `[一作]` (KAIST), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在弱监督条件下，提出一种新的交互推理模块 RegFormer，能够利用仅有的图像级 HOI 标注完成实例级交互检测；

**💡 创新点**

创新点包括：1）将交互识别拆分为人–物先查询（HO）再预测交互（I）的顺序解码；2）在查询构造中引入空间锚定的视觉特征，形成“空间锚定查询”；3）通过交互性评分（Interactiveness）对人–物对进行门控，抑制非交互对；4）实现仅用一次 backbone 前向即可完成实例级推理，无需额外训练；

**🔧 技术方法**

使用跨注意力的多标签解码器（ML-Decoder）框架；通过 CLIP 视觉/文本编码器（或 DINO）获取特征；采用文本嵌入作为初始查询；构造空间锚定查询并计算交互性得分；在实例级阶段加入实例遮罩实现区域约束；

**📊 数据集**

主要在 V‑COCO 与 HICO‑DET 两个 HOI 基准数据集上进行实验，并在 V‑COCO 进行零样本 HOI 评估；

**📈 对比分析**

与现有弱监督与全监督方法对比，RegFormer 在 HICO‑DET 上 Full mAP 提升约 12.5 分，达到 57.6% mAP（与全监督模型相当）；在 V‑COCO 上超过前沿弱监督方法 1‑2 分；在零样本评估中，在 NF‑UC 设置下对 unseen/seen/full mAP 分别取得 25.44/30.86/29.78，明显优于 OpenCat；

**⚠️ 局限性**

局限性：仍需依赖外部检测器生成候选框，检测质量会直接影响性能；在高密度场景下，空间锚定查询与交互性评分仍可能产生误检；目前对极少见或极不常见交互组合的泛化尚未彻底解决。

---

## 253. Neuropsychiatric Deviations From Normative Profiles: An MRI-Derived Marker for Early Alzheimer's Disease Detection

**arXiv ID:** 2604.00545 | [PDF](https://arxiv.org/pdf/2604.00545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 254. Towards Initialization-dependent and Non-vacuous Generalization Bounds for Overparameterized Shallow Neural Networks

**arXiv ID:** 2604.00505 | [PDF](https://arxiv.org/pdf/2604.00505v1)

**作者:** Yunwen Lei `[一作]`, Yufeng Xie `[通讯]` (University of Hong Kong)

**通讯引用:** 521 | [OpenAlex ID](https://openalex.org/A5100704297)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了针对浅层神经网络（SNN）在任意Lipschitz激活函数下的初始化依赖泛化界，利用路径范数并完全消除初始化矩阵谱范数的影响，得到宽度对数级依赖；

**💡 创新点**

创新点在于：①首次给出完全基于初始化的Rademacher复杂度上界；②将范数由Frobenius改为路径范数，显著降低上界；③给出匹配的下界，证明上界紧致；

**🔧 技术方法**

使用了Rademacher复杂度分析、路径范数定义、层层“剥离”技术（peeling trick）处理初始化约束，以及多类Rademacher复杂度等工具；

**📊 数据集**

在MNIST（784维）与ijcnn1（22维）两个二分类数据集上进行实验；

**📈 对比分析**

通过与现有基于谱范数、Frobenius范数、MAE等方法的比较，展示了新界在宽度大时保持非空且显著优于传统界的性能；

**⚠️ 局限性**

局限性包括：仅针对浅层网络；下界仅针对ReLU网络；实验仅覆盖全连接网络，未验证卷积或深层网络的可推广性。

---

## 255. STAR: Mitigating Cascading Errors in Spatial Reasoning via Turn-point Alignment and Segment-level DPO

**arXiv ID:** 2604.00558 | [PDF](https://arxiv.org/pdf/2604.00558v1)

**作者:** Pukun Zhao `[一作]` (Guangdong University of Finance and Economics), Haojian Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5067960278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出STAR两阶段框架，先用Turn‑point引导的监督微调提升静态空间理解，再通过段级直接偏好优化（SDPO）降低复杂迷宫导航中的级联错误。

**💡 创新点**

创新点在于引入人类启发的转折点(anchor)作为空间锚点，实现静态语义与动态决策的分阶段训练，并将DPO细化到段级以精准纠正首次错误。

**🔧 技术方法**

使用的技术包括监督微调（SFT）、LoRA参数高效微调、Segment‑level Direct Preference Optimization（SDPO）以及注意力分析等。

**📊 数据集**

采用新构建的RedMaze‑23K迷宫数据集，其中包含23K QA对，划分为转折点识别、规则理解和结构推理三大子任务。

**📈 对比分析**

与GPT‑4、DeepSeek‑V3等基线在Route Planning、Next‑Step Accuracy等指标对比，STAR 32B版在成功率达到29.27%（约82.4% GPT‑4），相较传统DPO提升显著。

**⚠️ 局限性**

局限在于仍无法完全匹配GPT‑4的转折点理解与下一步精度，且需要足够大规模模型方能充分利用该框架；更复杂的几何约束与多模态对齐仍待改进。

---

## 256. Multi-Camera View Scaling for Data-Efficient Robot Imitation Learning

**arXiv ID:** 2604.00557 | [PDF](https://arxiv.org/pdf/2604.00557v1)

**作者:** Yichen Xie `[一作]` (University Of California Berkeley), Hao-Shu Fang `[通讯]` (Massachusetts Institute Of Technology)

**通讯引用:** 2134 | [OpenAlex ID](https://openalex.org/A5077175165)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种利用多视角相机扩展专家演示的框架，通过在演示收集时同步多摄像头生成伪演示，从而在不增加额外人力成本的前提下提升模仿学习的场景多样性并显著提升数据效率；

**💡 创新点**

创新点在于将视角扩展作为数据扩增策略，首次系统分析动作空间对视角扩展的影响，并提出多视角动作聚合机制，使单视角策略在推理阶段仍能利用多摄像头信息；

**🔧 技术方法**

使用的技术包括扩散策略模型、DINOv3视觉编码器配合LoRA微调、视角尺度化伪演示生成、视角空间动作变换及多视角聚合算法；

**📊 数据集**

实验数据集涵盖Robomimic模拟任务（square、can、lift）以及真实世界的水倒灌任务（FANUC CRX-10iA机器人），并自行收集了50条远程操作演示；

**📈 对比分析**

与单视角训练基线对比，使用多摄像头伪演示后在成功率上提升了约20–70%，并在不同动作空间（base、camera、EEF）中验证了多视角训练的稳健性；

**⚠️ 局限性**

局限性包括需要额外的多摄像头硬件和视角空间动作需要精确校准；当训练视角与推理视角差异过大时，多视角优势减弱；此外仅提升了场景内部多样性，未结合环境随机化等更广泛的泛化策略。

---

## 257. A Japanese Benchmark for Evaluating Social Bias in Reasoning Based on Attribution Theory

**arXiv ID:** 2604.00568 | [PDF](https://arxiv.org/pdf/2604.00568v1)

**作者:** Taihei Shiotani `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3546 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于归因理论、聚焦推理层面、以日语文化为背景的偏见评测数据集JUBAKU‑v2，并用其对多款大型语言模型进行评估。

**💡 创新点**

创新点在于：①将偏见定义与归因理论（Pettigrew的终极归因误差）结合，系统划分推理偏见；②通过固定结论、仅变化推理内容的方式，剔除结论偏见的干扰；③设计多种提示变体，评估模型在推理偏见检测中的鲁棒性。

**🔧 技术方法**

技术手段包括：利用GPT‑5.1重写对话、GPT‑4o筛选偏见实例、人工评判一致性、四种提示模板与两种答案顺序的组合生成8个变体，最终构成216条评测案例；评估采用二元选择任务，准确率为指标。

**📊 数据集**

使用的数据集是自研的JUBAKU‑v2（27条原始案例×2答案×4提示模板=216条），并与现有日语偏见基准JBNLI、JBBQ、SSQA‑JA进行对照。

**📈 对比分析**

通过统一的二元选择格式和准确率指标，对9个模型（GPT‑4o、GPT‑5.2、Claude 4 Sonnet、Qwen3系列、gpt‑oss系列）进行评测。结果显示：GPT‑5.2与Claude 4 Sonnet在JUBAKU‑v2上达≈98–99%准确率，GPT‑4o约82%；相比之下Qwen3‑30B‑Instruct仅48%；JUBAKU‑v2的得分方差最高（0.0278），显示更高的判别力。

**⚠️ 局限性**

局限性包括：①数据量有限（216条），覆盖的文化维度虽多但仍不完整；②评价仅关注推理层面，结论层面的偏见仍未完全排除；③模型对提示的敏感性高，需进一步改进鲁棒性；④部分实例的标注仍依赖人工判断，存在主观性。

---

## 258. Quantum-Safe Code Auditing: LLM-Assisted Static Analysis and Quantum-Aware Risk Scoring for Post-Quantum Cryptography Migration

**arXiv ID:** 2604.00560 | [PDF](https://arxiv.org/pdf/2604.00560v1)

**作者:** Animesh Shaw `[一作]` `[通讯]` (Independent Researcher), Animesh Shaw (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三阶段量子安全代码审计框架：正则扫描、LLM上下文丰富和VQE威胁评分；

**💡 创新点**

创新点是将LLM用于语境识别以降低误报，并用VQE模型将加密算法的量子攻击成本映射为连续风险分值；

**🔧 技术方法**

技术栈包括Python 3.11、正则表达式扫描、Anthropic Claude LLM API、Qiskit 2.x实现的VQE、GitHub MCP等；

**📊 数据集**

评估使用五个开源项目（python-rsa、python-ecdsa、python-jose、node-jsonwebtoken、Bouncy Castle）共5,775条发现；

**📈 对比分析**

在602条人工标注样本上实现71.98%精度、100%召回、83.71%F1，VQE风险分数从3.54到7.00为迁移优先级提供参考；

**⚠️ 局限性**

局限包括对测试代码误报（需EXCLUDE_PATHS）、非加密上下文误判、VQE评分仅为相对优先级而非精确量子硬件预测。

---

## 259. LLM-supported document separation for printed reviews from zbMATH Open

**arXiv ID:** 2604.00554 | [PDF](https://arxiv.org/pdf/2604.00554v1)

**作者:** Ivan Pluzhnikov `[一作]` (George August University of Göttingen), Bela Gipp `[通讯]` (George August University of Göttingen)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对zbMATH Open扫描文献进行OCR转换为LaTeX，并利用微调后的LLM和多数投票机制实现文档自动拆分，生成可机读文本及起止索引。

**💡 创新点**

结合Mathpix OCR与多模型投票的LLM拆分策略，能够纠正OCR错误、去除元数据，且在多模型协同下大幅提升准确率。

**🔧 技术方法**

使用Mathpix OCR、LoRa微调的CalmExperiment‑7B‑slerp、Maxine‑7B‑0401‑Stock、Mistral‑Passthrough‑8L‑10B LLM、LLM‑Factory、BLEU/编辑距离评估以及Majority Voting投票框架。

**📊 数据集**

基于zbMATH Open 813,005篇文献的254,444页扫描文件以及自制的25页LaTeX真值评测集。

**📈 对比分析**

与正则表达式、ChatGPT‑4o、计算机视觉模型（DIT）比较，Majority Voting在测试集上达到97.5%准确率；OCR评测中Mathpix平均BLEU 0.2507、编辑距离0.1733，明显优于其它OCR。

**⚠️ 局限性**

仍受OCR误识、相似标题导致的误拆、LLM幻觉等影响；索引检索覆盖率仅90.6%，部分文档因元数据删除未匹配，且对长文档易出现生成错误。

---

## 260. BloClaw: An Omniscient, Multi-Modal Agentic Workspace for Next-Generation Scientific Discovery

**arXiv ID:** 2604.00550 | [PDF](https://arxiv.org/pdf/2604.00550v1)

**作者:** Yao Qin `[一作]` (Beijing 1st Biotech Group Co., Ltd.), Xiaoming Zhang `[通讯]` (Chinese PLA General Hospital)

**通讯引用:** 6561 | [OpenAlex ID](https://openalex.org/A5100462584)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了名为BloClaw的多模态AI科学操作系统，用于自动化化学信息学、蛋白质结构预测和分子对接等实验室任务。

**💡 创新点**

引入XML‑Regex双轨路由协议以消除JSON序列化失败、利用运行时状态拦截沙箱实现无缝图形捕获，以及自适应视口UI实现动态交互式可视化。

**🔧 技术方法**

结合Python猴子补丁、Plotly/Matplotlib可视化、RDKit、ESMFold、RESTful代理、XML与正则提取、MVC架构以及前端Canvas渲染等技术。

**📊 数据集**

评估使用SMILES、RDKit 2D图像、ESMFold 3D蛋白预测、PDB对接样本，以及PDF/CSV/Excel/PDB等多模态文件集。

**📈 对比分析**

与AutoGPT、ChemCrow、ChatGPT ADA等现有框架对比，BloClaw将路由错误率从17.6%降至0.2%，可视化捕获成功率从0%提升至100%，多模态数据摄取延迟低于150 ms，整体性能显著优于传统JSON+LangChain方案。

**⚠️ 局限性**

仍依赖LLM对代码生成的准确性，缺乏对非Python语言完整支持，且在极大规模并行任务时可能出现资源瓶颈。

---

## 261. Optimsyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation

**arXiv ID:** 2604.00536 | [PDF](https://arxiv.org/pdf/2604.00536v1)

**作者:** Zhiting Fan `[一作]` (Zhejiang University), Zuozhu Liu `[通讯]` (Zhejiang University)

**通讯引用:** 1154 | [OpenAlex ID](https://openalex.org/A5024343415)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出一种基于梯度影响估计的合成数据生成框架，通过强化学习动态调整评判标准（rubric）来提升监督微调（SFT）数据质量，并在知识密集型领域（人文社科与医学）中验证其效果。

**💡 创新点**

创新点在于：①引入优化器感知的影响函数（Adam兼容）直接衡量合成样本对目标模型的训练贡献；②将评判标准的生成视为可学习策略，用影响得分作为奖励，摒弃专家手工规则；③实现评判标准与生成器、目标模型的闭环，提升数据与模型对齐度。

**🔧 技术方法**

主要技术包括：梯度影响估计、GRPO（Policy‑Gradient）强化学习、轻量级有效性校验、基于教师模型的问答生成、不同规模/家族目标模型的微调与评估。

**📊 数据集**

使用公开文档作为种子：223本人文社科书籍（Goodreads精选）和医学文献（Meditron预训练语料+PubMed摘要）；生成合成QA数据；评估使用MMLU、SuperGPQA、Humanities’ Last Exam、BigBench Hard、HellaSwag、DROP、MedQA、HealthBench、PubMedQA等基准。

**📈 对比分析**

与多种SFT基线（WildChat、OpenHermes、Bonito、Conder等）以及指令模型（Qwen3‑8B‑Instruct、Llama3‑8B‑Instruct）对比，实验显示在两大领域均实现显著提升（如HLE +27.2%相对提升），且跨模型规模（4B‑14B）和模型家族（Qwen、Llama）均保持稳健优势。

**⚠️ 局限性**

局限性主要是：①强化学习奖励路径间接，缺乏对生成步骤的梯度回传，导致高方差与训练不稳定；②对回合规模（rubric探索量）敏感；③仅在所选人文和医学领域验证，未覆盖更广泛或高风险场景。

---

## 262. Learning from Many and Adapting to the Unknown in Open-set Test Streams

**arXiv ID:** 2604.00533 | [PDF](https://arxiv.org/pdf/2604.00533v1)

**作者:** Xiao Zhang `[一作]` (University of Science and Technology Beijing), Huimin Ma `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 7848 | [OpenAlex ID](https://openalex.org/A5006236325)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于Rac1与MAPK信号通路的Synapse Consolidation（SyCo）方法，用于大语言模型在多源开放式测试流中的参数高效自适应。

**💡 创新点**

创新性地将生物学记忆更新机制映射为两条路径：Rac1限制梯度更新到低秩尾部子空间以保持源知识，MAPK通过分层激活调节学习率以消除噪声，并引入MOA设置与结构化无监督目标。

**🔧 技术方法**

结合低秩适配器（LoRA/TA-LoRA）、梯度遮罩、InfoNCE对比损失、熵正则化、伪标签与多信号可靠性控制的自监督TTA框架。

**📊 数据集**

在18个NLP任务上评估，其中8个为源任务训练，10个为目标任务（包括未见任务与表面扰动），涵盖文本分类、问答、对抗等多领域。

**📈 对比分析**

与TENT、ClusT3、FOA、POEM、T^2ARD等单源TTA、SPoT、ATTEMPT、MPT、TA-LoRA等MTL基线以及其MOA组合对比，SyCo在未见任务与数据移位下分别达到78.31%和85.37%，领先最佳基线3.76%与2.42%。

**⚠️ 局限性**

方法依赖多源标注数据、低秩子空间设定与伪标签可靠性，过度稀疏或高度偏移的目标分布可能导致性能下降，且对超参数敏感。

---

## 263. Learning Shared Representations for Multi-Task Linear Bandits

**arXiv ID:** 2604.00531 | [PDF](https://arxiv.org/pdf/2604.00531v1)

**作者:** Jiabin Lin `[一作]` (Qingdao University), Shana Moothedath `[通讯]` (Iowa State University)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5087007662)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于 OFUL 的多任务线性 bandit 表示学习算法，并给出置信集构造与 regret 上界。

**💡 创新点**

利用任务共享的低维子空间进行谱初始化，并在置信集上引入结构约束，使 regret 从 O(dT√N) 降低到 O(√(drNT))。

**🔧 技术方法**

OFUL（Optimism in the Face of Uncertainty）、谱初始化、低秩矩阵估计、置信椭圆推断等技术。

**📊 数据集**

在模拟数据上验证：d=100/200/300，r=2/4/8，T=200/400/800，采用标准高斯特征与噪声。

**📈 对比分析**

与独立解决 T 个任务的 OFUL 基线比较，实验显示多任务版本在各规模下均实现更低的每任务累计 regret，且随 r、d 降低、T 增大而提升。

**⚠️ 局限性**

假设特征向量为标准高斯、噪声为高斯，且需要较大的探索样本量；对非高斯特征及非独立噪声的鲁棒性尚待研究。

---

## 264. Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling

**arXiv ID:** 2604.00510 | [PDF](https://arxiv.org/pdf/2604.00510v1)

**作者:** Hongbeen Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jaehyuk Huh `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3221 | [OpenAlex ID](https://openalex.org/A5047149607)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM推理中提出负早停（Negative Early Exit）与自适应加速（Adaptive Boosting）机制，将MCTS并行化并动态调度计算资源，以降低尾部延迟并提升吞吐量。

**💡 创新点**

创新点在于：①负早停通过识别无前景的搜索路径提前终止，②自适应加速将被释放的GPU资源重新分配给潜在高价值的并行搜索；③将这些技术集成到vLLM，实现可直接落地。

**🔧 技术方法**

采用并行MCTS（基于WU‑PUCT），负早停判定逻辑与阈值，资源调度与预估模型；并利用vLLM、GPU多卡并行推理。

**📊 数据集**

使用Math500与AMC23数学推理数据集，评估模型为Qwen2.5‑14B‑Instruct与Llama‑3.1‑8B‑Instruct，奖励模型为Qwen2.5‑Math‑PRM‑7B。

**📈 对比分析**

与Beam Search、Vanilla MCTS以及仅正早停的系统相比，p99延迟可降低至1.47×，吞吐量提升至2.44×，准确率在大部分实验中保持不变或略降（<1%）。

**⚠️ 局限性**

局限性包括：对高并发场景下的负早停触发率仍有限；对易解决任务，Boosting反而导致调度开销增大；极小数据集（AMC23）准确率易受单个样本波动影响。

---

## 265. PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training

**arXiv ID:** 2604.00503 | [PDF](https://arxiv.org/pdf/2604.00503v1)

**作者:** Weifu Fu `[一作]` (YouTu Lab, Tencent), Chengjie Wang `[通讯]` (YouTu Lab, Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PET-DINO，一种支持文本与视觉提示的通用目标检测框架，改进 Grounding DINO 并引入可调视觉提示生成与多路训练策略。

**💡 创新点**

核心创新包括 Alignment-Friendly Visual Prompt Generation (AFVPG) 使视觉提示与文本提示共享参数；Intra-Batch Parallel Prompting (IBP) 与 Dynamic Memory-Driven Prompting (DMD) 两种并行训练策略，显著提升视觉提示多样性与泛化能力；以及继承式预训练策略减少开发周期。

**🔧 技术方法**

技术手段涵盖 Transformer 结构、Deformable Self-Attention、CLIP 文本编码、视觉提示生成、跨图像/类别视觉提示聚合、并行 Prompt 训练与动态视觉提示缓存等。

**📊 数据集**

使用 Object365、OpenImages、V3Det 进行大规模训练；在 COCO、LVIS minival、ODinW35 等公开数据集上进行零样本评估，且使用这些数据集的图像-文本配对来构建视觉提示。

**📈 对比分析**

在多种评估协议（Visual-I、Visual-G、Text）下与 T‑Rex2、CP‑DETR、DINOv、OpenSeed 等方法对比，PET‑DINO 在 COCO 的 Visual‑I 模式下达到 64.0 AP、Visual‑G 40.3 AP，显著优于同类方法；在 LVIS、ODinW35 也实现与或超过竞争者的性能。

**⚠️ 局限性**

局限性包括：对长尾类别在 Visual‑G 模式下仍表现不足；依赖预训练文本模型，若无足够文本提示预训练会导致性能下降；视觉提示多样性在早期受限，需 IBP/DMD 等策略缓解。

---

## 266. HarassGuard: Detecting Harassment Behaviors in Social Virtual Reality with Vision-Language Models

**arXiv ID:** 2604.00592 | [PDF](https://arxiv.org/pdf/2604.00592v1)

**作者:** Junhee Lee `[一作]` (Kwangwoon University), Jinwoo Kim `[通讯]` (Chungbuk National University)

**通讯引用:** 3527 | [OpenAlex ID](https://openalex.org/A5100434595)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了基于视觉的社交VR骚扰行为数据集，并提出使用视觉‑语言模型（VLM）实现骚扰检测的系统

**💡 创新点**

创新点在于：①首次为社交VR创建专用视觉骚扰数据集；②通过上下文提示与链式思维引导，利用VLM对骚扰行为进行多层次分类；③在仅使用视觉信息的条件下，显著降低了对生物识别数据的依赖

**🔧 技术方法**

使用技术包括：VLM（GPT‑4o）与 prompt engineering（上下文、CoT、few‑shot）、VLM微调、传统CNN/LSTM混合模型、Transformer视频分类模型以及实验室自建Unity VR平台进行数据采集

**📊 数据集**

使用了从14名受试者收集的825段VR视频（经预处理得到3,408段），包含四种场景（交流室、攀岩室、打猪机、射绳机），涵盖四类骚扰行为（攻击性、个人空间侵犯、破坏性、正常行为）

**📈 对比分析**

在二分类（骚扰vs非骚扰）中，微调后的VLM在上下文+CoT提示下达到88.09%准确率、F1≈0.833，几乎与Transformer基线（88.04%）相当；在多分类中，VLM微调+few‑shot实现68.85%准确率，明显优于CNN/LSTM（43%）和Transformer（70%）基线

**⚠️ 局限性**

局限性包括：数据量有限（仅200条视频微调），仅使用单一人形化身，导致可推广性受限；多分类性能仍低于二分类；VLM推理耗时较长（≈5.5s），不适合实时部署；缺乏对多样化化身与多模态环境的鲁棒性验证

---

## 267. To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining

**arXiv ID:** 2604.00715 | [PDF](https://arxiv.org/pdf/2604.00715v1)

**作者:** Karan Singh `[一作]` (Stanford University), Steven Y. Feng `[通讯]` (Stanford University)

**通讯引用:** 172 | [OpenAlex ID](https://openalex.org/A5039277252)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定数据预算下系统地研究预训练与检索存储的权衡，提出三维（模型规模、预训练数据、检索数据）缩放框架；

**💡 创新点**

首次量化预训练与检索之间的可替代性和边际收益，揭示在低数据/小模型下检索能显著替代预训练；

**🔧 技术方法**

使用OLMo-2系列语言模型、FAISS向量检索、Log型缩放法则与损失回归；

**📊 数据集**

利用100B token的DCLM语料库，构建不同规模检索索引（1–20×预训练数据）；

**📈 对比分析**

与无检索基线对比，检索持续提升 perplexity（平均约1–3%），并通过三维缩放法则得到低交叉验证误差；

**⚠️ 局限性**

实验检索设置过于简单（单一检索器、固定top‑k），只评估 perplexity，且仅覆盖30M–3B规模，未探索更强检索或更大模型。

---

## 268. Birdcast: Interest-aware BEV Multicasting for Infrastructure-assisted Collaborative Perception

**arXiv ID:** 2604.00701 | [PDF](https://arxiv.org/pdf/2604.00701v1)

**作者:** Yanan Ma `[一作]` (City University of Hong Kong), Yuguang Fang `[通讯]` (City University of Hong Kong)

**通讯引用:** 24406 | [OpenAlex ID](https://openalex.org/A5016290340)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出 Birdcast 多路复用框架，针对基础设施辅助协作感知中高维 BEV 特征的兴趣感知多路复用。

**💡 创新点**

创新点：联合特征选择与多路分组的兴趣感知多路复用优化，证明 NP‑hard 并给出基于子模最优的 1-1/√e 近似加速贪心算法；对用户异构兴趣和信道条件进行建模。

**🔧 技术方法**

使用子模优化、速率率化转换、加速贪心、熵滤波生成 MoI，采用多尺度 BEV 特征编码与注意力融合等技术。

**📊 数据集**

使用 V2X‑Sim 仿真数据集（SUMO+CARLA）进行评估。

**📈 对比分析**

与广播、单播、K‑means++、DP、Marginal‑Utility 等基线比较，Birdcast 在总效用上提升 27%，mAP 提升 3.2%，运行时低于 10‑15 ms，整体性能最佳。

**⚠️ 局限性**

局限性：仅考虑单向 HVN 下行，未覆盖 V2V 互联、多路复用动态变化；模型对时延和信道估计有假设，实际部署需应对更复杂环境。

---

## 269. TP-Seg: Task-Prototype Framework for Unified Medical Lesion Segmentation

**arXiv ID:** 2604.00684 | [PDF](https://arxiv.org/pdf/2604.00684v1)

**作者:** Jiawei Xu `[一作]` (Jiangxi Normal University), Xiaoqi Zhao `[通讯]` (Yale University)

**通讯引用:** 7988 | [OpenAlex ID](https://openalex.org/A5050583798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种统一医学病灶分割框架 TP-Seg，利用任务原型和双路径适配器实现多任务学习；

**💡 创新点**

创新点在于任务条件适配器（TCA）通过共享与任务专属路径动态分配特征；以及原型引导任务解码器（PGTD）通过持续更新的前景/背景原型作为语义记忆，显著提升分割精度；

**🔧 技术方法**

技术方案包括 SAM 2 预训练编码器、任务条件路由块、交叉注意力、动态卷积专家以及指数移动平均原型更新；

**📊 数据集**

使用了 8 个医学病灶分割数据集，覆盖 OCT、MR‑T1、病理图像、超声、内镜、CT、超声和皮肤镜等多模态；

**📈 对比分析**

与多种专用、通用及统一模型（如 SegGPT、Spider、SAM2‑UNet、SR‑ICL）进行对比，TP‑Seg 在所有任务上平均 Dice 及 mIoU 均提升至约 86.6% 与 88.5%，显著优于基线；

**⚠️ 局限性**

局限性包括对预训练模型依赖较大、任务数仍有限（仅 8 任务）、以及在极小病灶或极端模态下的性能仍需进一步验证。

---

## 270. MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data

**arXiv ID:** 2604.00682 | [PDF](https://arxiv.org/pdf/2604.00682v1)

**作者:** Clémentine Grethen `[一作]` (IRIT - Universite Toulouse), Géraldine Morin `[通讯]` (IRIT - Universite Toulouse)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了统一的月球表面感知基准数据集MoonAnything，整合了几何监督与光照监督，提供了大规模的立体图像、深度图、SVBRDF参数以及多光照渲染，旨在提升月球视觉任务的鲁棒性。

**💡 创新点**

创新点在于将先前的StereoLunar与LunarG2R两套子数据整合为一个统一框架，首次实现了在真实DEM上使用物理基渲染和学习得到的空间变异BRDF生成多视角、多光照、具备完整几何与光照标注的大规模月球数据集。

**🔧 技术方法**

使用SurRender物理渲染管线、Hapke BRDF与学习得到的SVBRDF、SPICE太阳位置、真实DEM及LRO影像等技术，生成了包含相机内外参数、密集深度、法线、SVBRDF参数和多光照渲染图像的完整数据样本。

**📊 数据集**

数据集由两部分组成：GeometricPerception（约84k对立体图像+深度）和PhotometricPerception（约84k DEM+真实影像+多光照渲染），共计130k样本，覆盖南极和Tycho陨石坑两大区域，并配备9个不同光照条件的渲染图像。

**📈 对比分析**

在垂直、斜视与动态轨迹的立体重建任务中，对MASt3R、VGGT等SOTA模型进行fine‑tune后，误差（Accuracy/Completeness/Chamfer）显著下降（约30%–50%），表明该数据集能有效提升几何重建与光照鲁棒性能。

**⚠️ 局限性**

局限性包括：数据仍以合成渲染为主，光照与反射特性与真实月球可能存在偏差；覆盖区域仅为南极与Tycho，缺乏更广泛的地貌多样性；SVBRDF参数的学习误差可能影响光照相关任务的精度。

---

## 271. When AI and Experts Agree on Error: Intrinsic Ambiguity in Dermatoscopic Images

**arXiv ID:** 2604.00651 | [PDF](https://arxiv.org/pdf/2604.00651v1)

**作者:** Loris Cino `[一作]` (Sapienza), Cosimo Distante `[通讯]` (Consiglio Nazionale delle Ricerche)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过对多种CNN模型和专家皮肤科医生的诊断结果进行比对，识别出在所有模型和专家都难以准确分类的皮肤镜图像，并开发了自动化图像质量检测方法。

**💡 创新点**

发现AI与人类专家对同一批图像存在系统性错误，将图像质量与误差关联，并提出使用置换检验识别困难图像及自动检测模糊图像的技术。

**🔧 技术方法**

采用ResNeXt-50、ResNet-152、EfficientNet-B4/B5/B6等CNN，利用拉普拉斯滤波、傅里叶变换、波形变换进行模糊检测，并使用Cohen’s kappa、Fleiss’ kappa及置换检验等统计方法。

**📊 数据集**

主要使用ISIC 2019数据集（含HAM10000等），共5-fold交叉验证，训练25个模型进行评估。

**📈 对比分析**

通过对比AI模型、专家诊断与真值的准确率、敏感性、特异性等指标；在难图像上专家准确率仅29.6%（kappa 0.08），控制图像为66.2%（kappa 0.61），表明AI和专家在困难样本上表现相近，均受图像质量影响。

**⚠️ 局限性**

研究受限于仅6名皮肤科医生的参与、图像质量阈值尚未完全确定、缺乏真实临床环境验证，以及未能在meta分类器中充分利用患者元数据。

---

## 272. Heterogeneous Mean Field Game Framework for LEO Satellite-Assisted V2X Networks

**arXiv ID:** 2604.00621 | [PDF](https://arxiv.org/pdf/2604.00621v1)

**作者:** Kangkang Sun `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 41998 | [OpenAlex ID](https://openalex.org/A5105515339)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于异构均值场博弈（HMFG）的LEO卫星辅助V2X网络资源分配框架，并给出了针对车辆队列一维模型的最优类型分辨率 K*(N)=Θ(N^1/3) 的理论与实验验证。

**💡 创新点**

创新点在于：①将Wasserstein异质性度量与ε‑Nash误差分解结合，推导出最优类型数与队列状态维数的闭式关系；②设计了可自动选择 K*、自适应步长与鲁棒性修正的三算法流水线；③证明在LEO拓扑变动下的阶级稳定性并给出实验验证。

**🔧 技术方法**

核心技术包括均值场博弈理论、Wasserstein距离估计、G‑prox Primal‑Dual Hybrid Gradient（PDHG）优化、非平稳时间图模型以及量化异质性对收敛的影响分析。

**📊 数据集**

采用基于Python的离散化仿真环境，实验覆盖车辆数量 10^2–10^5、不同类型比例与LEO卫星带宽波动，未使用公开真实数据集。

**📈 对比分析**

与单一类型均值场博弈、SMFG、双/三类型MARL 等方法对比，实验显示 HMFG 在 K*=N^1/3 方案下实现了约 29.5% 延迟降低、60% 通过率提升、2.3 倍 PDHG 收敛速度提升。

**⚠️ 局限性**

主要局限包括：仅在局部时间区间内证明良定性，类型划分预先固定且无法在线自适应；高维状态空间时采样误差下降速率变慢，可能需要新的划分策略；对极端非稳态环境的鲁棒性尚未充分验证。

---

## 273. Fluently Lying: Adversarial Robustness Can Be Substrate-Dependent

**arXiv ID:** 2604.00605 | [PDF](https://arxiv.org/pdf/2604.00605v1)

**作者:** Daye Kang `[一作]` (University of Seoul), Hyeongboo Baek `[通讯]` (University of Seoul)

**通讯引用:** 141 | [OpenAlex ID](https://openalex.org/A5101774207)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了脉冲神经网络(SNN)目标检测器在对抗攻击下的鲁棒性，并发现了一种新的失效模式——质量腐蚀(QC)，即检测数量保持不变但mAP急剧下降。

**💡 创新点**

创新点包括提出质量腐蚀指数(QCI)用于衡量检测数量与精度的耦合程度，设计聚焦膜探针(FMP)揭示膜动力学是QC的关键攻击面，以及系统性验证不同SNN与ANN模型在同一攻击下的失效模式差异。

**🔧 技术方法**

主要技术包括白盒PGD对抗攻击（ℓ∞/ℓ2），迁移攻击、输入净化、对抗训练与经验ε-认证，FMP在膜电位上加入正则化以放大QC，QCI作为评价指标。

**📊 数据集**

使用的基准数据集为COCO 2017，测试样本为前1000张验证图像（完整5000张验证集结果一致）。

**📈 对比分析**

与传统ANN检测器(YOLOv3‑tiny、YOLOv8s、YOLOX‑S)相比，EMS‑YOLO在相同攻击下显示极高的QCI（最高+63），表明其存在QC；其他SNN和ANN模型保持QCI≈0，表现为传统抑制失效。

**⚠️ 局限性**

局限性包括仅评估单一SNN模型（EMS‑YOLO）、单一数据集与攻击范式、未覆盖事件驱动输入或真实世界测试、以及未探索多步或自动攻击套件对QC的影响。

---

## 274. Enhancing REST API Fuzzing with Access Policy Violation Checks and Injection Attacks

**arXiv ID:** 2604.00702 | [PDF](https://arxiv.org/pdf/2604.00702v1)

**作者:** Omur Sahin `[一作]` (Erciyes University), Andrea Arcuri `[通讯]` (Kristiania University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于现有 REST API fuzzing 工具 EvoMaster，设计并实现了七种新的自动化判据（oracle），用于检测访问控制错误（如 BOLA、BFLA）、身份验证错误、匿名修改、信息泄露、堆栈跟踪泄露、隐藏端点等安全缺陷，并扩展支持 SQL 注入和 XSS 攻击的自动化测试生成。

**💡 创新点**

创新点在于：①提出多种基于 HTTP 状态码与请求顺序的判据，能自动识别访问控制失效与身份验证错误；②利用已有测试用例重构生成安全验证用例，无需手工编写；③支持黑盒和白盒两种运行模式；④通过生成可直接执行的测试代码（Java/Kotlin/JavaScript/Python），方便开发者快速定位和修复缺陷。

**🔧 技术方法**

技术实现主要依赖 EvoMaster 的 REST API fuzzing 框架、OpenAPI 规范解析、HTTP 请求与响应记录、自动化测试用例生成与裁剪、正则表达式检测堆栈信息、时间测量检测 SQLi。判据逻辑通过对已有测试集的分析，动态拼接请求序列并验证期望的 HTTP 状态码或响应体。

**📊 数据集**

实验使用了 52 个 API：9 个人工构造的 Kotlin/SpringBoot 示例（已注入故障），8 个公开的“漏洞演示”API（黑盒测试），以及 36 个来自 WFD 数据集的真实 API（白盒测试）。此外还利用 200 个公开 API 的源代码统计作为实验规模参考。

**📈 对比分析**

实验采用 1 小时 fuzzing + 10 次重复，随后执行安全判据后处理阶段。与传统仅检测 500 错误的判据相比，新判据在人工示例中 100% 检测到所有注入的缺陷；在漏洞演示 API 中发现数十个安全缺陷；在 WFD 实际 API 中发现 300+ 个安全或协议错误。后处理阶段的额外开销平均不到 5 秒，整体运行时间约 18 天（包含 10 次重复）。

**⚠️ 局限性**

限制包括：①需要提供至少两种不同角色的认证信息，若 API 未实现多角色或改写登录接口，判据无法生成有效测试；②黑盒测试下若登录凭证可被修改，导致后续请求失效，影响判据准确性；③某些判据依赖特定的 HTTP 状态码分配，若服务器自定义错误码会产生误报；④对 SQLi 的睡眠测试需假设数据库存在且响应时间可测，数据库不同或使用缓存可能导致漏报。

---

## 275. AfrIFact: Cultural Information Retrieval, Evidence Extraction and Fact Checking for African Languages

**arXiv ID:** 2604.00706 | [PDF](https://arxiv.org/pdf/2604.00706v1)

**作者:** Israel Abebe Azime `[一作]` (Saarland University), Dietrich Klakow `[通讯]` (Saarland University)

**通讯引用:** 4541 | [OpenAlex ID](https://openalex.org/A5008875255)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了AfriFact数据集，涵盖10种非洲语言与英语，覆盖健康与文化新闻领域，完成了信息检索、证据抽取与事实核查三阶段任务；

**💡 创新点**

创新点在于提供多语言并行的检索与证据标注资源，揭示低资源语言在跨语言检索和事实核查上的巨大性能差距，并验证少量示例提示与参数高效微调能显著提升模型表现；

**🔧 技术方法**

使用多语言句子嵌入模型进行检索与证据抽取，并采用大型语言模型（如Llama、GPT、Qwen）进行零/少样本事实核查，同时探索LoRA、QLoRA等微调技术；

**📊 数据集**

主要数据集包括AfriDocMT（健康文档）、维基百科（文化内容）、XL‑Sum（新闻），以及新发布的18k条多语言主张与证据的AfriFact；

**📈 对比分析**

通过nDCG@10（检索）和nDCG@3（证据）评估，发现非洲语言检索效果远低于英语；LLM零样本准确率约33%，大模型提升至约60%，少样本提示可提升至77%，QLoRA 8‑bit微调在健康领域可提升约26%；

**⚠️ 局限性**

局限性包括仅覆盖健康与文化新闻两大域、训练集规模有限、未实现端到端流水线评估、缺乏超参调优与数据增强，以及语言覆盖仅限10种，可能导致泛化性受限。

---

## 276. Breadth-First Search Trees with Many or Few Leaves

**arXiv ID:** 2604.00691 | [PDF](https://arxiv.org/pdf/2604.00691v1)

**作者:** Jesse Beisegel `[一作]` (Brandenburg University of Technology), Martin Strehler `[通讯]` (Westsächsische Hochschule Zwickau)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5002855059)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了最大（最小）叶子生成树问题，特别是针对通用搜索（GS）、广度优先搜索（BFS）和字典序广度优先搜索（LBFS）进行分析，探讨了这些搜索树的复杂性。

**💡 创新点**

提出了针对GS、BFS和LBFS的最小和最大叶子问题的复杂性分析，发现当以树中叶子的数量为参数时，这些问题是可解的，但当以内部顶点的数量为参数时则是NP难的。

**🔧 技术方法**

使用了图搜索算法，包括通用搜索（GS）、广度优先搜索（BFS）和字典序广度优先搜索（LBFS）。

**📊 数据集**

论文中没有具体提到使用的数据集，主要集中在理论分析和复杂性结果上。

**📈 对比分析**

通过与已知的最小生成树问题进行比较，发现对于GS和BFS类型的搜索，最小叶子问题是可解的，而对于DFS则是NP难的，表明这两者在复杂性上存在显著差异。

**⚠️ 局限性**

限制在于仅考虑了第一入树（first-in trees），而未探讨最后入树（last-in trees）的情况，后者可能会导致不同的复杂性结果。

---

## 277. G-ICSO-NAS: Shifting Gears between Gradient and Swarm for Robust Neural Architecture Search

**arXiv ID:** 2604.00703 | [PDF](https://arxiv.org/pdf/2604.00703v1)

**作者:** Xingbang Du `[一作]` (Hokkaido University), Masaharu Munetomo `[通讯]` (Hokkaido University)

**通讯引用:** 1438 | [OpenAlex ID](https://openalex.org/A5087706095)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种三阶段的神经网络结构搜索框架G-ICSO-NAS，将梯度下降与改进的竞争性群体优化交替使用，以实现高效且鲁棒的架构搜索。

**💡 创新点**

创新点包括：① 在探索阶段将ICSO与梯度更新分离，利用多维度多样性感知适应函数避免搜索崩溃；② 在稳定阶段采用Hoeffding不等式实现自适应早停；③ 通过双螺旋参数更新机制在搜索中保持高质量解。

**🔧 技术方法**

技术主要包括改进的竞争性群体优化器ICSO、连续松弛的DARTS搜索空间、梯度下降权重训练、早停判定与多样性感知适应函数。

**📊 数据集**

使用的数据集包括CIFAR-10、CIFAR-100、ImageNet以及NAS-Bench-201搜索空间的CIFAR-10/100和ImageNet-16-120。

**📈 对比分析**

与多种基准方法比较，G-ICSO-NAS在DARTS空间下CIFAR-10搜索成本仅0.15 GPU天，测试误差2.54%，在ImageNet上top-1 24.98%，在NAS-Bench-201上实现了与最优方法相当的准确率，整体表现优于现有大部分梯度或进化型NAS方案。

**⚠️ 局限性**

局限性包括对warm‑up阶段和多样性权重的敏感性；在极大搜索空间时ICSO评估成本仍较高；以及对GPU资源的依赖在分布式环境下需进一步优化。

---

## 278. Learning to Hint for Reinforcement Learning

**arXiv ID:** 2604.00698 | [PDF](https://arxiv.org/pdf/2604.00698v1)

**作者:** Yu Xia `[一作]` (University of California San Diego), Yuxiong He `[通讯]` (Snowflake AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 HiLL 框架，联合训练提示生成器（hinter）和推理器（reasoner），在强化学习中针对 GRPO 的优势坍塌问题进行在线自适应提示生成，并通过提示依赖性衡量来指导提示质量。

**💡 创新点**

创新点：①把提示生成视为可学习的任务，在线根据推理器当前错误动态生成提示；②引入提示依赖性（hint reliance）度量，证明其与提示成功能在无提示环境中的转移性正相关，并在奖励中加入转移加权因子，鼓励产生可转移的提示。

**🔧 技术方法**

使用技术：Group Relative Policy Optimization (GRPO)、提示生成器策略网络、提示依赖性计算（基于对数似然差）、转移加权奖励、联合训练（共训练）以及 RLVR 的可验证奖励机制。

**📊 数据集**

数据集：训练使用 OpenR1-Math-220k 的 15k 题目子集；评估使用 AIME24、AIME25、AMC23、MATH-500、Minerva Math、OlympiadBench、GPQA-diamond、MMLU-Pro 等八个数学及通用推理基准。

**📈 对比分析**

方法对比：与 Base、GRPO、LUFFY、Scaf-GRPO、SAGE 等基线进行对比。HiLL 在 Llama‑3.2‑3B‑Instruct 与 Qwen2.5‑7B‑Instruct 两个主干模型上，平均准确率均高于所有基线，尤其在数学题目上提升 2–3 分，且在跨域基准上也实现显著收益。

**⚠️ 局限性**

局限性：相比纯 GRPO，HiLL 需要额外的提示生成与评估步骤，计算成本提升 2–4 倍；提示质量仍依赖于参考解的可用性；实验主要集中在二元奖励的可验证场景，未验证在更复杂奖励或更大模型上的表现。

---

## 279. Performance of Neural and Polynomial Operator Surrogates

**arXiv ID:** 2604.00689 | [PDF](https://arxiv.org/pdf/2604.00689v1)

**作者:** Josephine Westermann `[一作]` (Heidelberg University), Jakob Zech `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统比较了基于神经网络的算子逼近方法（包括使用 L^2_μ 和 H^1_μ 损失的 RBNO 以及 Fourier Neural Operator）与传统多项式逼近方法（稀疏网格和张量训练）在两类参数化 PDE（线性扩散和非线性弹性）上的逼近精度与计算成本。

**💡 创新点**

创新点在于把这四类方法放在同一框架下进行 Pareto 前沿比较，并在实验中引入了基于梯度信息的 H^1_μ 训练、张量训练的 ALS‑cross 方案以及对输入光滑度 s 的系统性调节。

**🔧 技术方法**

使用的技术包括 PCA/EMD 维度约简、全连接和 Fourier 频域卷积网络、Chebyshev/Leja 插值、稀疏格插值、Tensor‑Train（TT）压缩以及自动微分与雅可比训练。

**📊 数据集**

实验数据基于 1000 维随机场输入，采用 Matérn 协方差下的特征分解，生成多组不同光滑度 s（0.5–3.0）的参数样本，随后对每组样本在有限元网格上求解得到真实解。

**📈 对比分析**

比较方法是绘制不同方法在训练样本数、评估时间和训练/设置时间上的 L^2_μ 与 H^1_μ 误差 Pareto 前沿；结果表明：对于光滑输入，稀疏网格和 TT 方法在样本效率和评估速度上优于神经算子；对于粗糙输入，FNO 以最快收敛率领先；梯度信息训练的 RBNO 在低样本量下显著优于 L^2_μ RBNO。

**⚠️ 局限性**

局限性包括：神经算子对光滑度敏感、训练成本高；TT 逼近需 ALS‑cross 运行 CPU、对高精度时内存和时间开销大；所有方法都需要大量高质量样本；在非线性弹性问题上无法使用 TT 方案；实验受限于单一网格尺寸和有限元实现，难以推广到更复杂几何或高维参数。

---

## 280. OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models

**arXiv ID:** 2604.00688 | [PDF](https://arxiv.org/pdf/2604.00688v1)

**作者:** Han Zhu `[一作]` (Xiaomi Corp.), Daniel Povey `[通讯]` (Xiaomi Corp.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了OmniVoice，一种可覆盖600多种语言的零样本文本到语音（TTS）模型；

**💡 创新点**

创新点在于：① 单阶段无自回归（NAR）扩散语言模型架构，直接将文本映射到多码本声学码；② 全码本随机掩码训练策略大幅提升训练效率；③ 利用预训练大语言模型（LLM）权重初始化显著提升可懂度；

**🔧 技术方法**

采用双向Transformer骨干、离散扩散目标、全码本随机掩码、LLM初始化、混合精度与序列打包训练、迭代无掩码解码、Classifier-Free Guidance等技术；

**📊 数据集**

训练数据为581,000小时、覆盖600+语言的开放源数据集（共50个来源），并对低资源语言做语言级重采样；

**📈 对比分析**

与现有AR和NAR SOTA模型在中文、英文以及多语言基准（MiniMax 24语种、FLEURS 102语种）上进行对比，OmniVoice在可懂度（WER/CER）、语音自然度（UTMOS）、说话人相似度（SIM-o）均达到或超过对手；

**⚠️ 局限性**

限制主要包括：对极低资源语言（<10h）仍可能存在可懂度下降；依赖LLM权重，需额外训练成本；在某些语言如粤语的ASR评估受限于ASR模型性能；

---

## 281. Agent psychometrics: Task-level performance prediction in agentic coding benchmarks

**arXiv ID:** 2604.00594 | [PDF](https://arxiv.org/pdf/2604.00594v1)

**作者:** Chris Ge `[一作]` (Massachusetts Institute of Technology), Kaivalya Hariharan `[通讯]` (Fulcrum)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个结合Item Response Theory（IRT）与任务/代理特征的模型，能够在缺少完整评测数据的情况下预测LLM代理在多步编码任务中的成功概率，并为任务难度和代理表现提供可解释的预测。

**💡 创新点**

创新点包括：① 将任务的issue、仓库状态、测试用例、解决方案等工件作为特征，通过LLM-as-a-judge和嵌入方式提取；② 将代理能力拆分为LLM和scaffold两部分可加性，支持跨基准的训练与预测；③ 在IRT框架中引入这些特征后显著提升对任务难度的解释力，并实现对未见任务、代理组合和基准的泛化。

**🔧 技术方法**

技术手段包括：Item Response Theory（IRT）与随机梯度变分推断（SVI），Ridge回归预测任务难度，LLM-as-a-judge评估特征，嵌入向量，AUC‑ROC评估指标，以及基于Fisher信息的自适应任务选择方法。

**📊 数据集**

使用了四个公开的agentic coding基准：SWE‑bench Verified、SWE‑bench Pro、Terminal‑Bench 2.0 和 GSO，并利用各自的评测结果、issue文本、仓库状态、测试补丁与解决方案等工件。

**📈 对比分析**

通过与基线（平均成功率）和oracle（使用完整数据训练的IRT）比较，AUC‑ROC 在 0.57–0.96 之间，任务特征与代理特征的加入显著提升预测性能，尤其在对未见任务、代理组合和基准的泛化实验中表现接近oracle。

**⚠️ 局限性**

局限性包括：只能预测已出现的LLM或scaffold，无法处理全新组合；对因果关系缺乏确定性；模型仅考虑可加性，未捕捉多步交互中的非线性效应；依赖完整任务工件信息，若缺失或不一致会影响效果。

---

## 282. AutoEG: Exploiting Known Third-Party Vulnerabilities in Black-Box Web Applications

**arXiv ID:** 2604.00704 | [PDF](https://arxiv.org/pdf/2604.00704v1)

**作者:** Ruozhao Yang `[一作]` (Singapore Management University), Xiaofei Xie `[通讯]` (Singapore Management University)

**通讯引用:** 6269 | [OpenAlex ID](https://openalex.org/A5084396416)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多代理框架，能够在黑盒 Web 应用上自动生成已知漏洞的可执行 exploit。

**💡 创新点**

创新点在于：①将漏洞触发逻辑抽象为可复用的 trigger 函数；②使用基于测试的验证与迭代细化机制；③将漏洞解析、验证、执行、改进四个阶段拆分为独立代理，提升可控性与鲁棒性。

**🔧 技术方法**

核心技术包括：大语言模型（Qwen‑Plus、GPT‑4o、DeepSeek‑V3、Claude‑3.7）与多代理协作；触发函数抽象与自动化校验；基于执行反馈的迭代优化；对抗 LLM 假设和环境适配。

**📊 数据集**

使用 104 条真实 CVE 记录（来自 NVD/CVE 及相关参考链接），构成 29 个攻击目标，生成 660 个具体利用任务，涉及 55,440 次模型-工具-任务组合的实验。

**📈 对比分析**

与 DireLLM、PentestAgent、PentestGPT‑Auto、VulnBot 等现有方法相比，实验显示平均攻击成功率（ASR）从 32.88%（最佳基线）提升至 82.41%，在大多数攻击目标类别和 LLM 背景下均显著优于基线。

**⚠️ 局限性**

局限性：仍需依赖 LLM 的推理与生成质量；对特殊安全策略（如 Claude‑3.7 的内容过滤）敏感；仅覆盖已公开的已知漏洞，缺少后渗透与横向移动等后续阶段；实验环境为 Docker 化的公开示例，真实部署中可能存在更多防御机制导致进一步失败。

---

## 283. CL-VISTA: Benchmarking Continual Learning in Video Large Language Models

**arXiv ID:** 2604.00677 | [PDF](https://arxiv.org/pdf/2604.00677v1)

**作者:** Haiyang Guo `[一作]` (University of Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 5655 | [OpenAlex ID](https://openalex.org/A5082548671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CL-VISTA基准，用于评估视频大语言模型在持续学习环境下的表现。

**💡 创新点**

首次构建多任务、覆盖感知、理解与推理的持续学习评估框架，并从性能、计算效率与存储开销三维度进行综合评估。

**🔧 技术方法**

结合LoRA、Mixture-of-Experts、回放、正则化等十种主流持续学习技术，在Video-LLaVA和VideoLLaMA2模型上进行对比实验。

**📊 数据集**

采用8个任务的数据集，包括计数、空间感知、交通、电影、GUI、科学、体育与推理等，主要来源为NextQA、STAR等公开数据并自行构建新数据。

**📈 对比分析**

实验表明传统方法往往不及简单LoRA微调，且多数方法在泛化或效率上存在显著折衷；最高的MR‑LoRA和DISCO在Seen任务中几乎无遗忘，但在通用视频评测上表现下降。

**⚠️ 局限性**

存在的限制包括任务多样性仍有限、对长期任务序列的真实部署难以模拟、以及高计算与存储开销导致的实用性受限。

---

## 284. Common TF-IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness

**arXiv ID:** 2604.00672 | [PDF](https://arxiv.org/pdf/2604.00672v1)

**作者:** Zeyad Ahmed `[一作]` (University of Prince Edward Island), Aitazaz A. Farooque `[通讯]` (University of Prince Edward Island)

**通讯引用:** 5206 | [OpenAlex ID](https://openalex.org/A5047448010)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建惩罚型beta‑二项语言模型和对应的惩罚似然比检验（PLR）统计量，证明TF–IDF及其变体（BTF–IDF、TF–ICF）自然来源于该检验的关键项，从而给出TF–IDF的统计学解释，并基于PLR统计量提出一种新的词权重方案。

**💡 创新点**

创新点在于：① 将word burstiness（词突发性）与统计假设检验框架结合；② 通过beta‑二项模型引入gamma惩罚，解决参数可识别性问题；③ 证明PLR统计量的主要项正是TF–IDF的两个常见变体，为TF–IDF提供理论依据；④ 在此基础上提出可用于文本分类的新词权重方法。

**🔧 技术方法**

使用的技术包括：贝塔‑二项分布（Beta-binomial）与其近似、惩罚似然比检验（PLR）、Gamma分布惩罚、最大似然估计、以及对比实验中使用的多项式朴素贝叶斯分类器。

**📊 数据集**

数据集：1) 通过Dirichlet‑multinomial生成的合成文档集合；2) 20 Newsgroups（约18,846篇）；3) R8（Reuters‑21578子集，5,501训练 + 2,190测试）。

**📈 对比分析**

与传统TF–IDF的比较方法是使用多项式朴素贝叶斯分类器，在20 Newsgroups和R8上评估宏平均/加权精度、召回率、F1分数。实验结果显示，新方法的总体准确率与TF–IDF相当（20 Newsgroups: 0.80 vs 0.81；R8: 0.94 vs 0.94），但在极不平衡的少数类上略逊于TF–IDF。

**⚠️ 局限性**

局限性包括：① 需要估计beta‑二项参数，计算开销相对TF–IDF更大；② 依赖α_i ≪ 1且α_complement ≫ 1的假设，若不满足可能失效；③ 目前仅提供词权重，不给出完整的p值检验；④ 未考虑词与词之间的相关性（共突发性）。

---

## 285. Streaming Model Cascades for Semantic SQL

**arXiv ID:** 2604.00660 | [PDF](https://arxiv.org/pdf/2604.00660v1)

**作者:** Paweł Liskowski `[一作]` (Snowflake Inc.), Kyle Schmaus `[通讯]` (Snowflake Inc.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两种用于语义SQL的模型级联算法，实现流式、无通信的分区级执行，保证精度与召回或成本质量权衡。

**💡 创新点**

①在SUPG框架基础上加入迭代阈值细化和联合精度-召回目标；②使用Generalized Additive Model对代理模型进行概率校准，直接优化单参数成本-质量目标并自动适配数据难度。

**🔧 技术方法**

统计阈值估计（importance采样、置信界），泛化加性模型（GAM）概率校准，随机分位数路由、差分进化优化、双倍训练策略。

**📊 数据集**

六个真实世界基准：MMLU、BoolQ、IMDB、ArXiv、SST-2、NYT，涵盖分类、过滤和联接等语义操作。

**📈 对比分析**

与多种基线（Proxy-only、Oracle-only、SUPG、LOTUS）比较，实验显示两算法在F1≥0.95时委派率分别低于18%和22%，比LOTUS节省约58% oracle调用；在低委派预算下（≤20%）更优，平均提升0.1-0.15 F1。

**⚠️ 局限性**

假设oracle为真值，可能在oracle噪声或多分类任务下失效；目前仅支持二分类，且全局置信度保守，未利用分区结构提升效率。

---

## 286. LiPS: Lightweight Panoptic Segmentation for Resource-Constrained Robotics

**arXiv ID:** 2604.00634 | [PDF](https://arxiv.org/pdf/2604.00634v1)

**作者:** Calvin Galagain `[一作]` (CEA LIST), Cyrill Stachniss `[通讯]` (University of Bonn)

**通讯引用:** 25805 | [OpenAlex ID](https://openalex.org/A5011166267)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了轻量化全景分割框架LiPS，在保留Mask2Former查询解码的前提下，压缩特征提取与融合路径，实现高效推理；

**💡 创新点**

创新点在于只路由部分编码器层并对其进行空间压缩，再用浅层变形注意力Pixel Decoder完成特征融合，显著降低 FLOPs 并保持大部分分割质量；

**🔧 技术方法**

采用轻量级编码器(AFFormer)、特征路由与空间压缩、改进的变形注意力 Pixel Decoder、Mask2Former查询解码器，并在 Jetson Orin 上使用 TensorRT FP16 优化；

**📊 数据集**

在 ADE20K 与 Cityscapes 两个标准全景分割数据集上进行实验；

**📈 对比分析**

与 Mask2Former-R50 进行对比，LiPS 在 ADE20K 上 PQ 36.4%（vs 39.6%），GFLOPs 36.8 vs 147.2；在 Cityscapes 上 PQ 55.3%（vs 62.1%），GFLOPs 121.5 vs 527.4；吞吐量提升 4.5~10.7 FPS，计算量降低约 6.8×；

**⚠️ 局限性**

主要局限在于实例 AP 下降，尤其对小物体和细结构的识别性能降低，且轻量化设计主要聚焦于前端特征压缩，无法进一步提升实例精度；

---

## 287. A Survey of On-Policy Distillation for Large Language Models

**arXiv ID:** 2604.00626 | [PDF](https://arxiv.org/pdf/2604.00626v1)

**作者:** Mingyang Song `[一作]` (Tencent), Mao Zheng `[通讯]` (Tencent)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5111473248)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了在大语言模型中的On-Policy Distillation（OPD）技术，建立统一的f‑divergence理论框架，并从反馈信号、教师访问方式、损失粒度三维度对现有方法进行分类与对比，探讨工业部署与未来研究方向。

**💡 创新点**

①将OPD方法统一映射到一个通用的f‑divergence最小化框架；②提出三维分类法，揭示方法设计的互补性与空白；③深入剖析白盒、黑盒与自蒸馏的差异，并提出可跨域迁移的理论视角；④在综述中识别并系统化了曝光偏差、计算瓶颈、模式崩塌等未解决难点。

**🔧 技术方法**

主要使用的技术包括：f‑divergence（Forward KL、Reverse KL、JSD、Skew KL）、DAgger式交互式监督、REINFORCE与KL约束RL、Token/Sequence‑level损失、对抗与偏好学习、Self‑Play与Privileged Information蒸馏、以及多模态与多任务的混合训练策略。

**📊 数据集**

涵盖了多种数据集和评测基准，如自然语言推理与对话的 AlpacaEval、MT‑Bench、BigBench；代码生成的 CodeXGLUE；数学推理的 GSM8K、MATH；以及多语言与多模态任务的 XSum、WMT、MMLU、MATH‑500 等。

**📈 对比分析**

通过对比表与实验结果，OPD方法相较于传统离线KD在曝光偏差上显著降低错误累积，尤其在长链推理任务中提升 5–15% 的 Pass@k；在代码与多语言任务中，token‑level 逆KL 与自适应KL 进一步提升 2–4% 的准确率；白盒与黑盒方案在可用性与性能上各有权衡，白盒在密集反馈下更高效，黑盒在可访问性与扩展性上更有优势。

**⚠️ 局限性**

主要局限包括：①对大模型而言，OPD的在线采样与梯度估计成本极高；②教师质量与可访问性限制了白盒方案的普适性；③在自蒸馏与Self‑Play中容易出现模式崩塌与收敛饱和；④对不同任务的最优f‑divergence与采样策略缺乏统一理论指导；⑤跨模态与大规模部署时的计算与能源消耗仍是瓶颈。

---

## 288. TALENT: Target-aware Efficient Tuning for Referring Image Segmentation

**arXiv ID:** 2604.00609 | [PDF](https://arxiv.org/pdf/2604.00609v1)

**作者:** Shuo Jin `[一作]` (XJTLU), Jimin Xiao `[通讯]` (XJTLU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出针对参照图像分割的参数高效微调框架 TALENT，解决文本对视觉特征的非目标激活（NTA）问题

**💡 创新点**

创新点在于：1) Rectified Cost Aggregator（RCA）在特征层级聚合视觉-文本信息；2) Target-aware Learning Mechanism（TLM）通过 Contextual Pairwise Consistency Learning（CPCL）和 Target Centric Contrastive Learning（TCCL）双重目标同时抑制 NTA 并提升目标辨识度

**🔧 技术方法**

采用冻结的 DINOv2-Reg（ViT-B/14）视觉编码器、CLIP（ViT-B/16）文本编码器，RCA 采用成本体积匹配与 ReLU 修正；TLM 通过文本增强的相似度矩阵、语义相关性一致性损失和对比学习损失实现

**📊 数据集**

在 MS‑COCO 衍生的 RefCOCO、RefCOCO+、G‑Ref 三个基准数据集上进行评测

**📈 对比分析**

与多种 PFT 与 PET 方法对比，TALENT 在 RefCOCO、RefCOCO+、G‑Ref 上均取得最优或接近最优的 oIoU/mIoU，G‑Ref val 提升 2.5% mIoU，RefCOCO val 提升 1.8% mIoU，Precision@0.9 最高 40.1%（比 DETRIS 高 12.6%）

**⚠️ 局限性**

仍受限于仅在冻结骨干网络上微调，无法进一步提升跨域或长文本表达的鲁棒性；对极少样本或极高分辨率图像的性能尚未充分验证

---

## 289. GRASP: Gradient Realignment via Active Shared Perception for Multi-Agent Collaborative Optimization

**arXiv ID:** 2604.00717 | [PDF](https://arxiv.org/pdf/2604.00717v1)

**作者:** Sihan Zhou `[一作]` (Dalian University of Technology), Yew-Soon Ong `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 27379 | [OpenAlex ID](https://openalex.org/A5068243197)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GRASP框架，利用主动共享感知实现梯度重对齐，以缓解多智能体学习中的非平稳性和梯度冲突问题。

**💡 创新点**

核心创新在于定义了通用Bellman平衡并证明其存在，采用最小范数共识梯度求解器（QP），将梯度冲突转化为凸包内的最优共识方向，确保所有智能体的梯度非负投影。

**🔧 技术方法**

基于PPO的CTDE架构，结合梯度冲突处理（PCGrad/ MGDA思路）、凸优化求解共识梯度、Kakutani固定点理论及信赖域约束实现。

**📊 数据集**

在StarCraft II Multi‑Agent Challenge（SMAC、SMACv2）、Google Research Football（GRF）和Multi‑Agent Particle Environment（MPE）等标准MARL基准上进行实验。

**📈 对比分析**

与MAPPO、HAPPO、MA²E等SOTA方法比较，GRASP在SMAC硬/超硬、SMACv2、GRF以及MPE简单扩散场景中样本效率更高、最终胜率更高，明显优于基线。

**⚠️ 局限性**

在高噪声或离线（off‑policy）环境下共识梯度的有效性降低，且当环境高度随机时共识更新幅度被压缩，导致收敛速度下降。

---

## 290. CircuitProbe: Predicting Reasoning Circuits in Transformers via Stability Zone Detection

**arXiv ID:** 2604.00716 | [PDF](https://arxiv.org/pdf/2604.00716v1)

**作者:** Rajkiran Panuganti `[一作]` `[通讯]` (Independent Researcher), Rajkiran Panuganti (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CircuitProbe 方法，利用激活统计快速预测 Transformer 推理时的“推理电路”位置并通过层复制验证其提升效果。

**💡 创新点**

创新点在于发现推理电路分为早期稳定电路与后期幅度电路两类，并设计梯度与异常双重得分机制，实现对电路位置的高效定位。

**🔧 技术方法**

使用层级激活统计（表示变化、相似度、范数增长、方差、秩）与 z 分数归一化，构建稳定得分和异常得分并取最大值进行排序。

**📊 数据集**

采用 10–50 条校准示例进行激活统计，随后在 GSM8K、BBH、250 题标准化基准等数据集上评估层复制效果。

**📈 对比分析**

与全搜索/层复制实验对比，CircuitProbe 的 top‑1 预测在所有 9 个模型中均与最优电路相差 ≤2 层；在 1–3B 小模型上实现 2–10% 的推理准确率提升，速度提升三到四个数量级。

**⚠️ 局限性**

局限性：对 20–32B 以上规模模型未验证；预测不总是完全精确；双电路同时复制的潜在增益尚未探索；对广泛任务的整体性能提升有限。

---

## 291. TTA-Vid: Generalized Test-Time Adaptation for Video Reasoning

**arXiv ID:** 2604.00696 | [PDF](https://arxiv.org/pdf/2604.00696v1)

**作者:** Soumya Shamarao Jahagirdar `[一作]` (University of Tübingen), Hilde Kuehne `[通讯]` (MIT-IBM Watson AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TTA‑Vid 框架，利用测试时强化学习对预训练的视觉‑语言模型在长视频推理任务中进行自适应更新，通过在同一视频的多帧子集上生成多条推理轨迹，并用频率奖励和熵惩罚对模型进行优化；同时引入多臂赌博机（MAB）学习帧的重要性分布，实现自适应帧采样。

**💡 创新点**

①无监督测试时自适应，完全不依赖标注；②基于频率+熵的批量奖励信号，利用多帧子集的多样性；③将帧采样建模为多臂赌博机，在测试时更新帧分布；④证明单批次自适应即可在整个数据集乃至跨数据集上提升性能，表现出良好的泛化性。

**🔧 技术方法**

使用的技术包括：Test‑time Reinforcement Learning（GRPO）、Batch‑wide Frequency Reward + Entropy Regularization、Multi‑armed Bandit（MAB）帧采样、Vision‑Language Backbone（InternVL‑3、Qwen2.5‑VL）、自监督的频率奖励计算、熵正则化、熵归一化、权重乘法更新等。

**📊 数据集**

主要实验数据集：VideoMMMU、MMVU、SciVideoBench、VideoMME、LongVideoBench；以及在对比中使用的其他开源/专有模型对应的评测集。

**📈 对比分析**

通过与多种基线模型（InternVL、LLaVA‑OneVision、GPT‑4o 等）以及专门的视频推理模型（Video‑R1、Video‑Chat‑R1、Video‑RFT、Video‑RTS 等）进行对比。TTA‑Vid 在所有五个基准上均超越基线模型，平均提升约 3–4%（例如 InternVL‑3‑8B 从 51.37% 提升至 55.13%），甚至逼近甚至超过在大规模标注数据上训练的专用模型。

**⚠️ 局限性**

局限性：
- 测试时仍需多帧子集和多轮推理，计算成本相对较高；
- 目前仅在视频问答推理任务验证，未知在其他多模态任务中的表现；
- 对极长视频或帧数过多时的收敛性与效率尚未充分评估；
- 频率奖励依赖答案分布的多样性，若多样性不足可能导致奖励信号不稳。

---

## 292. Internal APIs Are All You Need: Shadow APIs, Shared Discovery, and the Case Against Browser-First Agent Architectures

**arXiv ID:** 2604.00694 | [PDF](https://arxiv.org/pdf/2604.00694v1)

**作者:** Lewis Tham `[一作]` (Unbrowse AI), Jungpil Hahn `[通讯]` (National University Of Singapore)

**通讯引用:** 2079 | [OpenAlex ID](https://openalex.org/A5002220814)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种共享路由图Unbrowse，利用浏览器流量被动学习网站内部API，并通过x402微支付模型实现多方激励；

**💡 创新点**

创新点在于将浏览器逆向发现过程转化为集体知识共享机制，设计了三层微支付（查询费、安装费、执行费）与增量式贡献归因；

**🔧 技术方法**

技术包括Kuri（Zig原生CDP代理）实现轻量级浏览器捕获、网络流量分析与API逆向、AgentSkills.io标准化技能包、语义向量检索与x402协议支付；

**📊 数据集**

使用了94个真实域名的任务集（来源于WebArena信息检索类别）以及20个冷启动域，涵盖政府、SaaS、社交等多类型网站；

**📈 对比分析**

与Playwright浏览器自动化基准在同一硬件与网络环境下比较，平均缓存延迟950 ms vs 3 404 ms，速度提升3.6×（平均）/5.4×（中位数），冷启动平均12.4 s但可在后续调用中摊销；

**⚠️ 局限性**

局限性包括仅在未被反爬或简单WAF的站点上可用、对动态身份验证与会话续期支持不足、冷启动成本高、经济模型和归因机制未正式证明抗Sybil，且实验仅在单机环境下进行，未覆盖大规模部署和跨地区性能。

---

## 293. Chameleons do not Forget: Prompt-Based Online Continual Learning for Next Activity Prediction

**arXiv ID:** 2604.00653 | [PDF](https://arxiv.org/pdf/2604.00653v1)

**作者:** Marwan Hassani `[一作]` (Eindhoven University of Technology), Sjoerd van Straten `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了CNAPwP，一种基于提示的在线持续学习框架，用于在概念漂移环境下的下一活动预测。

**💡 创新点**

创新点在于：①将DualPrompt算法改造为面向流程预测的双提示（G‑Prompt与E‑Prompt）结构；②引入任务特定遗忘指标衡量任务回归时的知识保留；③构造多种带有重复概念漂移的合成与真实数据集；④在预测过程中融合滑动窗口、前缀树漂移检测与桶化提示选择。

**🔧 技术方法**

采用的技术包括：多头自注意力（MHSA）网络；前缀调优（Prefix Tuning）提示机制；滑动窗口与前缀树概念漂移检测；桶化长度选择与任务相似度比对；交叉熵训练；任务特定与通用提示的联合更新。

**📊 数据集**

使用的评估数据集包括：三类合成数据（ImbalancedTasks、RecurrentTasks、RandomTasks），以及两类真实业务日志（BPIC2015的循环漂移版本和BPIC2017的稳定过程）。

**📈 对比分析**

与 Landmark、Incremental Update、DynaTrainCDD、TFCLPM、GAN 等基线方法比较，CNAPwP 在 ImbalancedTasks、RecurrentTasks、RandomTasks 上取得最高平均准确率，在 BPIC2015 与 BPIC2017 上保持竞争性；相较于其他方法显著降低任务遗忘；平均每条事件处理时间维持在 3–25 ms，适合近实时部署。

**⚠️ 局限性**

局限性包括：提示调优导致的额外计算开销，尤其在日志复杂度高时显著；漂移检测与任务相似度判断仍为基线实现，缺乏自适应或渐进漂移的处理；对资源受限或隐私约束的真实场景尚未充分验证。

---

## 294. DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization

**arXiv ID:** 2604.00648 | [PDF](https://arxiv.org/pdf/2604.00648v1)

**作者:** Zhengxian Yang `[一作]` (BNRist, Tsinghua University), Tao Yu `[通讯]` (BNRist, Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种 DirectFisheye‑GS，能够在 3D Gaussian Splatting 框架中直接使用鱼眼图像进行训练，无需先去畸变，从而保持更完整的视角信息与细节。

**💡 创新点**

创新点包括：①将 Kannala‑Brandt 鱼眼投影模型嵌入 3DGS，实现本地化畸变建模；②提出跨视角联合优化（CVO）策略，利用特征重叠与视角角度差异来同时优化重叠视角中的 Gaussians，减少浮点、尺寸不一致等问题；③保持与原始 3DGS 兼容，使其可直接在现有渲染器和工具中使用。

**🔧 技术方法**

使用了 3D Gaussian Splatting、鱼眼投影模型（多项式偏差参数与 Kannala‑Brandt）、CUDA 加速的光栅化渲染、Jacobian 计算、基于 SIFT 余弦相似度构建的视角关联图、跨视角联合优化、球谐（SH）颜色编码及光照一致性约束。

**📊 数据集**

在 FisheyeNeRF、ScanNet++、Den‑SOFT（Ruziniu、Coffee 等大规模室内外场景）等公开数据集上进行实验，并对小、中、大尺度场景进行了评估。

**📈 对比分析**

与原始 3DGS、Fisheye‑GS、3DGUT、Self‑Cali‑GS 等基线在 PSNR、SSIM、LPIPS 等指标上进行对比。DirectFisheye‑GS 在所有数据集和视角（训练/测试）上均达到了或超过了当前最优方法，尤其在鱼眼边缘细节和光照一致性方面表现显著提升。

**⚠️ 局限性**

局限性包括：①需要预先构建特征重叠图，增加了预处理步骤；②在极高畸变或极稀疏视角的极端场景下仍可能出现浮点或细节缺失；③Gaussian 数量随视角增多而增加，导致训练时间与显存需求上升；④未在极端实时渲染条件下全面评估。

---

## 295. In the Middle, Not on Top: AI-Mediated Communication for Patient-Provider Care Relationships

**arXiv ID:** 2604.00643 | [PDF](https://arxiv.org/pdf/2604.00643v1)

**作者:** Ut Gong `[一作]` (Columbia University), Yan Guan `[通讯]` (Tsinghua University)

**通讯引用:** 24125 | [OpenAlex ID](https://openalex.org/A5100378023)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并评估了一种AI中介的异步患者-医护人员沟通系统（CLEAR），以支持资源有限环境下的关系中心护理

**💡 创新点**

提出“AI在中间、非顶层”的设计立场，将AI视为关系基础设施而非决策代理，强调对话的可解释性、时空分配与持续性记录

**🔧 技术方法**

运用自然语言处理技术实现信息转化、简化与摘要，构建异步消息流与多模态交互界面

**📊 数据集**

基于真实临床沟通收集的患者与医护人员消息样本（约200条）进行实验与用户研究

**📈 对比分析**

通过定性访谈与对比研究展示了AI中介在降低社交摩擦、提升信息理解和维护连续性方面的效果；未给出量化性能指标，但发现患者提问质量提升，医护负担减轻

**⚠️ 局限性**

局限性包括：系统对信息的框架化可能导致叙事固化与细微差异被抑制，隐私风险因记录持久化而上升，且未充分评估不同健康素养群体对系统的适应性

---

## 296. English to Central Kurdish Speech Translation: Corpus Creation, Evaluation, and Orthographic Standardization

**arXiv ID:** 2604.00613 | [PDF](https://arxiv.org/pdf/2604.00613v1)

**作者:** Mohammad Mohammadamini `[一作]`, Antoine Laurent `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了首个基于 TED 讲座的 Central Kurdish 语音到文本翻译语料库 KUTED（170 h English 语音 + 英文转录 + 中央库尔德语翻译），并对正字法进行系统标准化，随后在端到端、级联和从零训练模型上进行实验，比较不同技术对 BLEU 的影响。

**💡 创新点**

创新点包括：① 首创以 TED 讲座为源的 CKB 语音翻译语料库；② 设计三步正字法标准化流程（Unicode、AsoSoft 规范化、两张纠错表），显著减少词表并提升翻译质量；③ 在多模型（Seamless、NLLB、Fairseq）上验证 KUTED 的价值，并在 FLEURS 领域展示跨域泛化效果。

**🔧 技术方法**

使用的技术包括：端到端模型 Seamless V2（W2V‑BERT 2.0 + NLLB 解码器）；级联模型 Seamless ASR + NLLB 1.3B；从零训练的 Fairseq Transformer（76 M 参数）；正字法标准化采用 Unicode 校正、AsoSoft normalizer 与两张纠错表；音频对齐与误差过滤基于 ASR 归一化 Levenshtein 距离。

**📊 数据集**

使用的数据集：KUTED（170 h English → Central Kurdish 语音-文本对），FLEURS、CoVoST2、Aug‑LibriSpeech、VoxPopuli、Indic‑TEDST 等公开语料库做对比；文本翻译评估使用 KUTED 的 EN↔CKB 句对与 NLLB 预训练模型。

**📈 对比分析**

对比方法：先评估预训练基线（Seamless、Seamless‑NLLB），再在 KUTED 上 fine‑tune；记录 BLEU、WER、ChrF++ 等指标。结果显示：Seamless 基线在 KUTED 上 BLEU 5.04，Fine‑tune 后提升至 13.51；级联系统 BLEU 从 9.25 提升到 15.57；NLLB T2TT 在 KUTED 上分别得到 16.72（EN→CKB）/27.93（CKB→EN）BLEU；Fairseq 从零训练仅 7.90 BLEU；正字法标准化后 BLEU 从 11.34 提升到 15.18。说明 KUTED 对模型性能有显著正向作用，且在 FLEURS 领域也能带来 +3 BLEU 的提升。

**⚠️ 局限性**

limitations: (1) 仅包含 English→CKB 方向，缺乏多语种或 S2ST 方向；(2) 数据量相对有限，单一语料来源，需与其他资源联合使用；(3) 原始音频质量参差不齐，需要人工或 ASR 过滤；(4) 仅对标准化文本敏感，非标准化文本仍可能导致误差；(5) 由于版权限制，KUTED 及其模型无法公开，限制了复现与进一步研究。

---

## 297. A Physical Imitation Learning Pipeline for Energy-Efficient Quadruped Locomotion Assisted by Parallel Elastic Joint

**arXiv ID:** 2604.00611 | [PDF](https://arxiv.org/pdf/2604.00611v1)

**作者:** Huyue Ma `[一作]` (University of Bristol), Rui Wu `[通讯]` (University of Bristol)

**通讯引用:** 2643 | [OpenAlex ID](https://openalex.org/A5100449408)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种 Physical Imitation Learning（PIL）框架，将强化学习（RL）控制策略分解为可被并行弹性关节（PEJ）实现的被动响应与剩余的主动控制，进而显著降低仿真四足机器人 Unitree Go2 的机械功耗。

**💡 创新点**

创新点在于：①通过分解策略而非联合搜索实现脑体协同设计，避免维度灾难；②将被动组件以可制造的弹性曲线形式实现，仿照动物的肌腱-关节协同；③实现高达 87% 的功率卸载与 86% 的能耗降低。

**🔧 技术方法**

主要技术包括：Proximal Policy Optimization（PPO）强化学习、基于成本运输（CoT）的奖励设计、离散分箱的离线反演优化得到 PEJ 弯曲曲线、以及 IsaacLab 物理仿真平台。

**📊 数据集**

使用的“数据集”是 IsaacLab 生成的 4096 个并行仿真实例，涵盖 7 级地形（平地、随机粗糙面、盒子障碍、金字塔阶梯等）以及从 0–2.0 m/s 的速度命令采样得到的观察与动作序列。

**📈 对比分析**

对比方法：与仅使用 CoT 奖励的基线策略（不含 PEJ）以及与传统强化学习策略对比；性能指标显示：在平地训练下，PIL 可将功率卸载比例提升至 87%，能耗降低 86%；在最难地形（Level 6）时，卸载比例下降至 18%，能耗降低 8%，但仍保持可行的行走。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏物理硬件测试；跨地形泛化性有限（越难的地形需要更大主动控制，卸载比例下降）；PEJ 需要手工制造，设计灵活性受限；未考虑电机回馈与摩擦损失的真实能耗。

---

## 298. Speech LLMs are Contextual Reasoning Transcribers

**arXiv ID:** 2604.00610 | [PDF](https://arxiv.org/pdf/2604.00610v1)

**作者:** Keqi Deng `[一作]` (Microsoft), Jinyu Li `[通讯]` (Microsoft)

**通讯引用:** 12676 | [OpenAlex ID](https://openalex.org/A5100365053)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种在自动语音识别中加入链式推理（CoT）的模型CoT-ASR；

**💡 创新点**

首次在ASR中实现单通道的推理与转录一体化，并通过CTC引导的模态适配器高效对齐语音与LLM文本潜在空间；

**🔧 技术方法**

使用Conformer语音编码器、Phi4-mini‑instruct LLM、CTC‑guided模态适配器、LoRA微调、以及LLM自带的推理与提示学习；

**📊 数据集**

在38k小时内部英语语料上训练，评估于LibriSpeech、FLEURS以及多行业内部测试集；

**📈 对比分析**

与自研Phi4MM基线及Whisper‑large‑v3、Gemma 3n、Qwen3‑Omni‑30B等公开模型对比，CoT‑ASR在WERE上降低8.7%、在EER上降低16.9%，并在用户提供上下文时进一步提升至24.9%；

**⚠️ 局限性**

仍受限于大规模LLM的计算成本、对非英语及极端噪声环境的鲁棒性不足、以及模型在极少数据或新领域的迁移性能待进一步验证。

---

## 299. Embedded Variational Neural Stochastic Differential Equations for Learning Heterogeneous Dynamics

**arXiv ID:** 2604.00669 | [PDF](https://arxiv.org/pdf/2604.00669v1)

**作者:** Sandeep Kumar Samota `[一作]` (National Institute of Technology), Snehashish Chakraverty `[通讯]` (National Institute of Technology)

**通讯引用:** 7944 | [OpenAlex ID](https://openalex.org/A5017042977)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

建立了变分神经随机微分方程（V-NSDE）模型，用以连续时间、带随机性的方式刻画奥里萨邦各区社会经济指标的演变；

**💡 创新点**

创新点在于将VAE与Neural SDE结合，利用区嵌入自适应捕获地区异质性，并通过变分推断和adjoint求解器实现高效训练；

**🔧 技术方法**

使用了变分自编码器、Neural SDE、区嵌入、SDE数值求解、Adjoint sensitivity、ELBO损失、Adam优化器以及Monte Carlo近似；

**📊 数据集**

数据集为 2007、2015、2020 年的奥里萨邦 30 个区的六项社会经济指标，经过 Brownian Bridge 生成月度时间序列，共 168 个月；

**📈 对比分析**

通过对每个区计算 Gaussian NLL、可视化预测均值及置信区间进行评估，ELBO 在 1000 轮训练后收敛至 -10.75，NLL 在各区均表现良好，预测均值与实际轨迹高度一致，置信区间覆盖大部分观测点；

**⚠️ 局限性**

局限性包括区嵌入为时间不变，未考虑政策干预或宏观冲击等外部变量；对潜在维度和网络结构敏感；未建模层级空间相关性或因果约束；

---

## 300. StretchBot: A Neuro-Symbolic Framework for Adaptive Guidance with Assistive Robots

**arXiv ID:** 2604.00628 | [PDF](https://arxiv.org/pdf/2604.00628v1)

**作者:** Luca Vogelgesang `[一作]` (University of Montpellier), Filip Ilievski `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 5600 | [OpenAlex ID](https://openalex.org/A5008608420)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 StretchBot——一种融合多模态感知、知识图谱与大型语言模型（LLM）的混合神经符号框架，用于在家用环境中为用户提供自适应的伸展指导，并在实验室中进行三名受试者的可行性对比实验。

**💡 创新点**

创新点在于：① 将结构化知识图谱与LLM结合，使机器人能够在实时感知基础上进行基于情境的决策；② 采用“行动前缀+自然语言”双模输出，保证指令可执行性与人类可读性；③ 通过可验证器层保障安全性与对话连贯性；④ 在可行性试验中首次对比脚本式与自适应式指导在用户感知维度的差异。

**🔧 技术方法**

技术包括：YOLOv8n（实时物体检测）、MediaPipe Pose（姿态估计）、DeepFace+语音情绪识别（多模态情绪融合）、OpenRouter 接入 Llama‑2‑70B（LLM 推理）、自定义知识图谱（JSON 结构）、ROS2 交互框架、文本到语音（TTS）和可选的可视化点指令（POINT_XXX）。

**📊 数据集**

数据集：实验室内预置的标准物品（咖啡杯、水瓶、香蕉、椅子）和三名成人受试者的实时传感数据；使用公开预训练模型（YOLOv8、MediaPipe、DeepFace、Llama‑2）作为感知与推理的底层数据来源。

**📈 对比分析**

对比方法：每位受试者先后完成脚本化与自适应两种模式（8–10 分钟），随后填写六维度量表（清晰度、舒适度、适应性、信任度、自然度、物体相关性）并进行半结构化访谈。结果显示，自适应模式在适应性（M≈3.67）和物体相关性（M≈4.67）上明显优于脚本化，而舒适度、信任度与自然度略低；清晰度两种模式相近。定性访谈指出自适应更具情境感知，但偶尔导致认知负荷升高。

**⚠️ 局限性**

局限性：① 样本量极小（仅3名受试者）且实验时间短；② 低预算硬件导致 3–10 秒延迟，影响沉浸感；③ 情绪识别精度与多模态融合尚不成熟，易产生误判；④ 知识图谱覆盖不完整，需要外部常识库补齐；⑤ 只在实验室场景验证，缺乏真实家庭/医疗环境的长期评估；⑥ 可验证器虽提升安全性，但增加了响应延迟和系统复杂度。

---

## 301. Signal Constellations with Enhanced Energy Efficiency for High-Speed Communication Systems

**arXiv ID:** 2604.00710 | [PDF](https://arxiv.org/pdf/2604.00710v1)

**作者:** Mark Bykhovskiy `[一作]` `[通讯]` (Moscow Technical University of Communications and Informatics), Mark Bykhovskiy (Moscow Technical University of Communications and Informatics)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种新的多维信号星座构造方法SCOPT，能够在不增加发射功率或使用编码的前提下，通过增大归一化信号时长来提升最小欧氏距离，从而提高能量效率；

**💡 创新点**

创新点在于将信号星座的决策区域从N维球面改为超立方体，使得噪声在各维度上独立影响，导致误判概率随信号时长呈指数下降，甚至实现低于传统Shannon极限的可靠通信；

**🔧 技术方法**

主要技术包括几何星座设计、误差概率解析推导、能量损失公式的推导以及对比分析中的数值仿真；

**📊 数据集**

论文未使用实际数据集，而是基于理论模型和数值模拟进行验证；

**📈 对比分析**

通过理论推导与数值图像对比，SCOPT在相同误差概率下需要的信号时长约为传统SCSH的十分之一，能量损失显著低于SCSH，且误判曲线更平滑、能量效率提升明显；

**⚠️ 局限性**

局限性包括：仅在理想高斯噪声下验证，缺乏在多径衰落或深度衰落环境下的实测；缺乏硬件实现与实际调制解调器的验证；需进一步研究峰值因子和低速率通信的适用性。

---

## 302. SCPatcher: Automated Smart Contract Code Repair via Retrieval-Augmented Generation and Knowledge Graph

**arXiv ID:** 2604.00687 | [PDF](https://arxiv.org/pdf/2604.00687v1)

**作者:** Xiaoqi Li `[一作]` (Hainan University), Zongwei Li `[通讯]` (Hainan University)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5101530716)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了基于知识图谱的检索增强生成框架SCPatcher，实现自动化智能合约漏洞修复。

**💡 创新点**

①将5,000份验证合约构成功能级关系知识图谱；②采用多阶段重排序的检索机制；③双阶段修复策略（知识引导+链式思维）；④通过外部知识图谱降低LLM幻觉。

**🔧 技术方法**

使用Slither静态分析、Neo4j/CodeBERT构建知识图谱、深度检索+重排序、Deepseek‑V3 LLM、Chain‑of‑Thought推理以及Slither验证。

**📊 数据集**

5,000份Etherscan验证合约用于构建知识图谱；200份SmartBugs与公开漏洞仓库合约用于评测，并确保两者不重合。

**📈 对比分析**

与DirectFix、LogicRepair、SelfRefine等基线方法对比；SCPatcher取得81.5%整体修复率、91%编译通过率、89.6%有效修复率，显著优于基线。

**⚠️ 局限性**

仅依赖Slither静态检测，可能忽略运行时行为和跨合约逻辑；知识图谱覆盖范围有限；生成补丁需进一步动态验证。

---

## 303. Full-Gradient Successor Feature Representations

**arXiv ID:** 2604.00686 | [PDF](https://arxiv.org/pdf/2604.00686v1)

**作者:** Ritish Shrirao `[一作]` (Indian Institute of Information Technology), Raghuram Bharadwaj Diddigi `[通讯]` (Indian Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出一种全梯度成功特征表示 Q‑学习算法（FG‑SFRQL），用于在多任务强化学习中高效学习与奖励无关的动态特征，并实现快速任务迁移。

**💡 创新点**

创新点在于将成功特征学习转化为对完整均方贝尔曼误差的全梯度最小化，消除了传统半梯度更新的偏差，并给出几乎几乎必然收敛的理论证明；同时引入经验回放的平均目标和 GPI 选择的凸包梯度，进一步提升稳定性。

**🔧 技术方法**

主要技术包括：全梯度更新（Full‑Gradient DQN 思路）、成功特征表示（SFR）、Generalized Policy Improvement（GPI）、经验回放与批量平均、马尔可夫随机递归包含分析。

**📊 数据集**

在离散四房间网格、连续两关机器人臂（Reacher）以及连续 2D Maze（PointMaze）等三种环境上进行实验，分别采用固定动态、可变奖励的多任务设置。

**📈 对比分析**

与 DQN、FG‑DQN、传统半梯度 SFRQL 进行比较。FG‑SFRQL 在训练收敛速度、累计奖励和最终评估表现上均明显优于基线，尤其在连续任务切换和高维控制任务中表现最为突出；仅略高于基线的计算开销也在可接受范围内。

**⚠️ 局限性**

局限性包括：在某些设置下，批量平均的实验结果并未提升性能，表明单样本随机性对学习有益；理论收敛证明假设 GPI 最大化唯一性等，实际任务中可能不完全满足；对极大规模任务或非平稳动态的适用性尚待进一步验证。

---

## 304. TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models

**arXiv ID:** 2604.00666 | [PDF](https://arxiv.org/pdf/2604.00666v1)

**作者:** Lingjie Chen `[一作]` (University of Illinois), Hanghang Tong `[通讯]` (University of Illinois)

**通讯引用:** 17737 | [OpenAlex ID](https://openalex.org/A5068043486)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于轨迹引导的监督微调框架，利用自回归教师估计的 token 难度，设计轨迹感知掩码策略，在 Masked Diffusion Language Model 训练中引入轨迹监督。

**💡 创新点**

创新点在于：①通过单次自回归教师推理获取 token 难度，避免昂贵的 Diffusion 采样；②将难度离散为桶，构造 hard‑to‑easy 掩码，直接引导模型学习高效的解码轨迹；③实现低成本、可扩展的轨迹监督。

**🔧 技术方法**

使用的技术包括 Masked Diffusion Language Models、教师强制推理、难度分桶、轨迹感知掩码、LoRA 微调、DeepSpeed ZeRO‑2 等。

**📊 数据集**

主要数据集：s1K reasoning 数据用于训练；基准测试使用 GSM8K、MATH、HumanEval、MBPP；教师模型为 Qwen3‑8B；原始模型为 LLaDA‑Instruct 和 Dream‑Instruct。

**📈 对比分析**

通过与原始 DLM、Fast‑dLLM、Fast‑dLLM‑v2、D2F、dParallel、d3LLM 等基线比较，实验表明在保持相似或略低精度的前提下，能提升约 3 倍左右的并行度（TPS），并在多项数学和编码任务上与基于蒸馏的方法竞争，训练成本仅需 1K 样本和单次教师推理。

**⚠️ 局限性**

局限性：①需要预先设定难度分桶数量和掩码概率，超参数敏感；②只在中等规模模型上验证，未知是否能推广到更大模型；③轨迹监督仅基于单次教师推理，可能无法捕捉更复杂的解码动态；④与基于蒸馏的方案相比仍在训练时间和内存上略有劣势。

---

## 305. LibScan: Smart Contract Library Misuse Detection with Iterative Feedback and Static Verification

**arXiv ID:** 2604.00657 | [PDF](https://arxiv.org/pdf/2604.00657v1)

**作者:** Yishun Wang `[一作]` (Hainan University), Yuqing Zhang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 49054 | [OpenAlex ID](https://openalex.org/A5015051319)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 LibScan 框架，将 LLM 推理与静态分析相结合，自动检测智能合约中八类库误用模式。

**💡 创新点**

首次结合 LLM 语义推理、静态匹配与迭代反馈机制，构建知识库并针对八类误用模式实现精细化检测。

**🔧 技术方法**

使用 GPT‑4/GPT‑4 Turbo/DeepSeek LLM、TF‑IDF+Slither 静态分析以及随机森林融合策略。

**📊 数据集**

基于公开的 1,018 篇真实合约库误用标签数据，抽取 662 篇做为测试集。

**📈 对比分析**

与 Slither、GPTScan 等单一工具对比，LibScan 的准确率 85.15%，召回率 82.22%，F1 83.75%，显著优于对手。

**⚠️ 局限性**

受复杂合约语义理解限制，LLM 仍易产生幻觉，需进一步提升模型精度与误用模式覆盖率。

---

## 306. On Minimum Distances for Error Correction and Detection of Generalized Network Code

**arXiv ID:** 2604.00647 | [PDF](https://arxiv.org/pdf/2604.00647v1)

**作者:** Yulin Chen `[一作]` (Chinese University of Hong Kong), Raymond W. Yeung `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 20640 | [OpenAlex ID](https://openalex.org/A5038760786)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出通用的“广义网络通道”与“广义网络码”框架，并在此框架下定义了用于评估网络码误差纠正与误差检测能力的三种距离（D0、D1、D2），给出了它们之间的关系及其下界与上界。

**💡 创新点**

核心创新点在于：①首次统一定义了误差纠正、误差检测与联合误差纠正检测的距离，并证明在误差线性（error‑linear）通道下这三种距离完全一致；②构造了一个“精细化距离”D₂[c]，用以给出联合误差纠正的必要且充分条件；③给出了非线性网络码需要两个不同距离的现象，并通过一系列定理给出它们之间的紧耦合关系。

**🔧 技术方法**

主要技术包括：抽象化的群/向量空间结构、权重度量（如Hamming、rank、sum‑rank），以及对解码球交集的集合论分析；通过群同态、三角不等式、凸组合等数学工具推导距离定理与上下界；利用矩阵表示（如F_s,t、H_t）将经典码、线性网络码、秩码、求和秩码映射到同一框架。

**📊 数据集**

论文没有使用实际数据集，而是以理论分析和一个具体的示例网络（基于3元有限域的两条路径网络）来验证定理的正确性。

**📈 对比分析**

与传统的最小距离定理比较，论文提供了对误差纠正距离与误差检测距离的下界（d₁^min ≥ ⌊d₀^min/2⌋+1）和上界（d₂^min ≤ ⌈d₀^min/2⌉）。在误差线性通道下，所有距离相等，性能等同于经典线性网络码；对于非线性网络码，理论上可出现纠正错误数大于检测错误数的一半。

**⚠️ 局限性**

局限性包括：①对非误差线性通道，距离之间的关系和必要条件仍不完整；②未给出与网络拓扑结构直接关联的距离上限；③未讨论擦除误差距离以及实际实现的编码与解码复杂度。

---

## 307. On rankings in multiplayer games with an application to the game of Whist

**arXiv ID:** 2604.00641 | [PDF](https://arxiv.org/pdf/2604.00641v1)

**作者:** Alexis Coyette `[一作]` (Unamur), Eve Tilman `[通讯]` (Unamur)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种将 Bradley‑Terry 模型推广到两队玩家对战的多玩家游戏的扩展，并改进了参数估计算法。

**💡 创新点**

创新点在于用团队成员的隐性实力之和的 Logistic 函数建模对战概率，并在此基础上导出了高效的 Newman's‑style 固定点更新规则；同时给出了与 Huang 等人 Generalized BT 模型的对比与改进。

**🔧 技术方法**

采用最大似然估计、固定点迭代以及 Newman's 改进的梯度下降形式，并实现了对应的伪代码。

**📊 数据集**

在合成的有向超图数据集（随机生成玩家实力、不同团队划分）和真实的 Whist 卡牌游戏（300 场、17 名玩家）上进行实验。

**📈 对比分析**

与传统 BT、GBT 模型以及基于胜率的排名进行对比；实验表明三种模型在实力估计上的 Pearson 相关系数均高于 0.9；新模型在处理大规模超边时计算时间随超边数线性增长，明显优于 BT 模型；在 Whist 数据中不同方法得到的排名高度一致，且与胜率排名相关性高。

**⚠️ 局限性**

局限在于未证明新算法的全局收敛性质、仅在简单场景下实验，且未探讨团队规模变化对模型性能的影响。

---

## 308. When Safe Models Merge into Danger: Exploiting Latent Vulnerabilities in LLM Fusion

**arXiv ID:** 2604.00627 | [PDF](https://arxiv.org/pdf/2604.00627v1)

**作者:** Jiaqing Li `[一作]` (Huazhong University of Science and Technology), Kailong Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 1406 | [OpenAlex ID](https://openalex.org/A5000432413)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过在源模型中嵌入隐蔽攻击组件，使得单个模型安全但在模型融合后失效的攻击框架

**💡 创新点**

将模型融合过程转化为约束优化问题，设计了安全保持、功能保持与目标失衡三重约束，首次实现了融合后安全失衡的隐蔽攻击

**🔧 技术方法**

约束优化（方向一致性、Frobenius正交）、参数级攻击组件插入、任务算子(Task Arithmetic)、DARE、TIES‑Merging、KnOTS等融合算法

**📊 数据集**

使用公开的恶意提示集合、OpenAI的“OpenAI Harmful Behavior”数据集、MMLU、Perplexity数据集等

**📈 对比分析**

在9种LLM（Llama‑2、Llama‑3、Mistral）和4种融合算法上评测，攻击后模型有害响应率从基线1.9%提升至85.4%，且单模型安全性保持在原水平；同时对比SAM等安全防御显示难以检测

**⚠️ 局限性**

对融合超参数的敏感性较大，且在单模型功能下降仍有限；攻击方法主要针对对称融合场景，非均衡权重时效果下降；未来需要更强的安全防御与融合完整性验证

---

## 309. KG-CMI: Knowledge graph enhanced cross-Mamba interaction for medical visual question answering

**arXiv ID:** 2604.00601 | [PDF](https://arxiv.org/pdf/2604.00601v1)

**作者:** Xianyao Zheng `[一作]` (Northwestern Polytechnical University), Qiangguo Jin `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 1930 | [OpenAlex ID](https://openalex.org/A5084965704)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了KG-CMI框架，结合知识图谱与跨Mamba交互来提升医疗视觉问答性能；

**💡 创新点**

通过细粒度视觉-文本对齐、知识图谱嵌入、跨Mamba交互以及自由文本增强多任务学习，实现了对开放式问题的更好处理；

**🔧 技术方法**

采用ViT+RoBERTa进行特征提取，QQ-Former实现双向对齐，GAT嵌入知识图谱，CMM模块提供线性复杂度跨模态交互，T5解码器用于开放式答案生成；

**📊 数据集**

在SLAKE、VQA-RAD和OVQA三个医学视觉问答基准数据集上进行实验；

**📈 对比分析**

与多种SOTA方法比较，KG-CMI在整体准确率上均超过对手（SLAKE 84.26%/VQA-RAD 78.21%/OVQA 79.58%），尤其在开放式问题上提升显著；

**⚠️ 局限性**

对部分细粒度定位仍易出现误判，且在极小样本或高度复杂的多机构数据上性能略有下降。

---

## 310. MPI-Q: A Message Communication Library for Large-Scale Classical-Quantum Heterogeneous Hybrid Distributed Computing

**arXiv ID:** 2604.00600 | [PDF](https://arxiv.org/pdf/2604.00600v1)

**作者:** Feng Wang `[一作]` (Information Engineering University), Zheng Shan `[通讯]` (Information Engineering University)

**通讯引用:** 8453 | [OpenAlex ID](https://openalex.org/A5076356039)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了MPI-Q，一种专为大规模经典-量子异构分布式计算设计的消息通信库，支持经典进程与量子进程的统一调度与通信。

**💡 创新点**

创新点包括：① 构建异构混合通信域模型，实现经典与量子进程的统一管理与资源映射；② 采用轻量级通信架构，利用本地MonitorProcess直接接收预编译波形，消除多级转发延迟；③ 设计异构混合同步机制，结合MPI同步与Socket+硬件时钟触发，实现跨节点量子时序精确对齐。

**🔧 技术方法**

技术实现基于MPI的C/Python扩展，结合Qiskit/pyqpanda等量子编译工具，MonitorProcess与Socket实现高效数据交换，硬件时钟触发用于精确同步。

**📊 数据集**

实验使用分割的GHZ量子电路（40–250 qubits，进一步扩展至480 qubits）作为任务，采用量子模拟器pyqpanda与qiskit进行编译与执行。

**📈 对比分析**

通过与串行执行对比，测量总执行时间与加速比，MPI-Q在24个量子节点时实现最高18.76×的加速，整体表现出近线性可伸缩性。

**⚠️ 局限性**

局限性包括：在1–2个节点的小规模配置下无加速，负载不均导致子电路切分不一致；实际量子硬件受制于器件规模、噪声与时钟同步精度，尚需进一步提升量子资源利用与同步精度。

---

## 311. Predicting Dynamics of Ultra-Large Complex Systems by Inferring Governing Equations

**arXiv ID:** 2604.00599 | [PDF](https://arxiv.org/pdf/2604.00599v1)

**作者:** Qi Shao `[一作]` (Southeast University), Wei Lin `[通讯]` (Fudan University)

**通讯引用:** 16995 | [OpenAlex ID](https://openalex.org/A5100665430)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可扩展的稀疏识别图神经网络(SIGN)，通过两阶段流程（小样本稀疏回归+全图消息传递）在海量网络中从数据中直接推断共通的治理方程，并用数值积分实现长期可解释预测。

**💡 创新点**

将方程发现的稀疏结构与图神经网络分离，使方程推断的计算复杂度与节点数无关；使用DBSCAN聚类得到全局稀疏支持；将符号基函数与共享系数嵌入GNN，兼具可解释性与可扩展性。

**🔧 技术方法**

符号稀疏回归（Lasso/SINDy）、DBSCAN聚类、图神经网络消息传递、全节点支持约束、多步滚动损失、数值积分、sMAPE/MAPE/MSE评价指标。

**📊 数据集**

多种合成基准（Kuramoto、SIS、Michaelis–Menten、Rössler、FitzHugh–Nagumo、Hindmarsh–Rose），大规模实测网络（人脑网络、Catster、GitHub、飞鼠/人脑等），以及真实海表温度（71,987节点、2,281,812条边）数据。

**📈 对比分析**

与两阶段TP‑SINDy、LaGNA以及多种基于图的黑盒预测器（MTGNN、ASTGCN、MSTGCN、STSGCN、STGCN）对比；SIGN在系数误差上保持<1%（相较基线≈10%），在海表温度的两年外推、神经网络的100步预测等任务中获得最高精度；训练时间可扩展到10⁵节点，仅需数百秒。

**⚠️ 局限性**

假设所有节点遵循相同的自动力与耦合机制；只能处理一阶对称耦合，难以捕捉高阶、延迟或超图交互；需预设符号基函数，若真实动力不在库中会导致误差；对外部驱动和不规则采样的适应性有限。

---

## 312. Towards Viewpoint-Robust End-to-End Autonomous Driving with 3D Foundation Model Priors

**arXiv ID:** 2604.00597 | [PDF](https://arxiv.org/pdf/2604.00597v1)

**作者:** Hiroki Hashimoto `[一作]` (Chiba University), Kazuhiko Kawamoto `[通讯]` (Chiba University)

**通讯引用:** 444 | [OpenAlex ID](https://openalex.org/A5010514962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在不使用视角增强的前提下，通过将3D基础模型的几何先验注入端到端自动驾驶模型，实现了对摄像机视角变化下轨迹规划的鲁棒性提升。

**💡 创新点**

提出两种模块——3D空间编码器将深度估计的像素级3D位置作为位置嵌入加入图像特征；几何特征融合模块通过跨注意力把3D模型中间特征注入图像特征，从而实现几何先验的无增强鲁棒性提升。

**🔧 技术方法**

利用DA3 3D基础模型进行深度估计与几何特征提取，使用正弦位置编码、可学习MLP生成位置嵌入，并采用跨注意力机制融合特征，最终基于ResNet‑50编码器和规划头实现轨迹预测。

**📊 数据集**

在VR‑Drive视角扰动基准上评估，该基准基于nuScenes数据集，并通过OmniRe的Novel View Synthesis生成不同扰动条件的图像。

**📈 对比分析**

与AD‑MLP、BEV‑Planner、VAD、SparseDrive、DiffusionDrive、VR‑Drive等基线以及World4Drive进行对比，在原始条件下保持相近性能，在pitch和height扰动下明显优于World4Drive，但在depth（纵向平移）扰动下提升有限，整体性能未能达到VR‑Drive的最佳鲁棒性。

**⚠️ 局限性**

方法在3D位置嵌入对相机外参依赖强，导致纵向平移扰动下性能下降；目前的几何融合仍未完全实现视角无关性，需要进一步构造如BEV、体素或高斯场等视角无关的中间表示。

---

## 313. Revisiting Human-in-the-Loop Object Retrieval with Pre-Trained Vision Transformers

**arXiv ID:** 2604.00809 | [PDF](https://arxiv.org/pdf/2604.00809v1)

**作者:** Kawtar Zaher `[一作]` (Institut National de l'Audiovisuel), Alexis Joly `[通讯]` (Institut National de Recherche et Informatique et Robotique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个基于主动学习的交互式人机循环对象检索框架，在多物体图像中利用预训练 ViT 的全局与局部特征融合实现快速检索。

**💡 创新点**

创新点在于将全局与局部 ViT 表征融合成混合表示，结合主动学习反馈迭代训练二分类器，实现无需先验标签即可检索目标，并证明混合表示优于单一表征。

**🔧 技术方法**

采用预训练的 DinoV2 Vision Transformer、SVM 分类器、主动学习中的不确定性采样、Patch 局部特征与全局拼接/池化等技术。

**📊 数据集**

在 PascalVOC2012 与 COCO2017 两个多物体数据集上进行实验。

**📈 对比分析**

通过 MAP 与覆盖率指标比较，混合全局-局部表征（尤其是 concat/ pool 方式）在 MAP 上比全局仅表征提升 2-4%，覆盖率提升 5-15%，显著优于基线。

**⚠️ 局限性**

局限包括对物体尺寸分布的依赖、需要手动选择 patch 数量、对实时交互的计算成本、以及仅在无标注场景下的实验，缺乏对复杂真实交互的评估。

---

## 314. Valency Classification of Mapudungun Verbal Roots. Established by the language's own morphotactics

**arXiv ID:** 2604.00789 | [PDF](https://arxiv.org/pdf/2604.00789v1)

**作者:** Andrés Chandía `[一作]` `[通讯]` (University of Barcelona), Andrés Chandía (University of Barcelona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过分析Mapudüngun（马库丘语）动词词根在语言自身的形态语法中的后缀组合，构建了动词词根的价义（valency）分类，并将结果集成到现有的有限状态转导器（Düngupeyüm）形态分析器中。

**💡 创新点**

创新点在于：①首次以形态语法（后缀槽位与互斥关系）为依据，对已确认为动词的词根进行价义重分类；②提出一套基于后缀（-nie、-künu、-(kü)le、-fal、-(ü)m、-(ü)l、-tu、-fi/-e、-(ñ)ma）识别词根价义的形态学判别规则；③将这些规则直接编码进FST系统，以降低形态分析的不确定性。

**🔧 技术方法**

采用的技术包括：
- 有限状态转导器（Finite State Transducer，FST）实现形态分析。
- 形态学规则设计与实现，利用Mapudüngun的后缀槽位和互斥约束。
- 语料驱动的验证：利用以往的词根重分类研究和Paskwal Koña等人的自传文本进行实证检验。

**📊 数据集**

使用的数据集主要来自：
- 先前的词根词类重分类论文（含约一百个动词词根）。
- Mapudüngun词典与语料（十名受访者的录音文本）。
- Paskwal Koña长者的自传文本（提供约十几百个带后缀的动词形式）。
- 语料中标注的后缀槽位信息（由Ineke的《A Grammar of Mapuche》提供）。

**📈 对比分析**

方法评估：将新规则加入Düngupeyüm后，形态分析的多义性显著下降，能够准确区分转义和不转义的词根，提升了分析器在处理真实文本时的准确率。虽然未给出精确的准确率数值，但作者指出“显著改进”，并通过比对前后分析结果验证了规则的有效性。

**⚠️ 局限性**

局限性包括：
- 许多后缀（如- fal、- tu、- ñma）无法单独作为价义判别的可靠指示器，导致某些规则的适用范围有限。
- 词根的“可变价义”（labile）导致同一词根在不同语境中表现为转义或不转义，增大了分析器的歧义。
- 语料规模仍有限，部分后缀的使用频率不足，可能遗漏新的形态变体。
- 需要进一步在更多方言与现代口语文本上检验规则的普适性。

---

## 315. From Baselines to Preferences: A Comparative Study of LoRA/QLoRA and Preference Optimization for Mental Health Text Classification

**arXiv ID:** 2604.00773 | [PDF](https://arxiv.org/pdf/2604.00773v1)

**作者:** Mihael Arcan `[一作]` `[通讯]` (Home Lab), Mihael Arcan (Home Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统比较不同优化策略对心理健康文本分类的影响，按从基线到参数高效微调再到偏好优化的顺序进行实验

**💡 创新点**

在统一实验协议下揭示不同优化方法、目标、适配器、窗口、类别平衡等因素对性能的具体影响，提供可复现的选择指南

**🔧 技术方法**

使用XGBoost、BERT系列编码器、LoRA/QLoRA参数高效微调、DPO/ORPO/KTO偏好优化等技术

**📊 数据集**

DAIC‑WOZ对话语料，基于PHQ‑4生成四类联合标签及二分类的焦虑、抑郁子任务

**📈 对比分析**

在相同窗口、评估指标下对比各方法的最佳与平均macro‑F1，结果显示：优化后的BERT最优，LoRA/QLoRA微调处于中间，ORPO+重平衡的偏好优化效果最好，DPO/KTO表现弱；不同方法对目标、窗口敏感性差异显著

**⚠️ 局限性**

受限于单一数据集与任务、偏好优化样本有限、不同方法的超参空间未完全覆盖，导致结果对特定设置高度依赖，实用性需结合实际场景谨慎推广

---

## 316. PrivHAR-Bench: A Graduated Privacy Benchmark Dataset for Video-Based Action Recognition

**arXiv ID:** 2604.00761 | [PDF](https://arxiv.org/pdf/2604.00761v1)

**作者:** Samar Ansari `[一作]` (University of Chester), Samar Ansari `[通讯]` (University of Chester)

**通讯引用:** 3221 | [OpenAlex ID](https://openalex.org/A5007269494)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了PrivHAR-Bench，一个多层级隐私-实用性评估基准数据集，用于统一评估视频动作识别中的隐私保护方法。

**💡 创新点**

创新点在于提供从轻度模糊到加密块置换的渐进隐私层级、背景去除版本、标准化分割与无损存储，并发布完整评估工具，首次实现跨方法、跨隐私强度的可比对比。

**🔧 技术方法**

采用高斯模糊、Canny 边缘提取、AES-CTR块置换加密，以及 YOLOv8n-Pose 进行 ROI 检测；数据以无损 PNG 存储，并用 SSIM、PSNR、Face‑Detection‑Fail 等指标和 PU 评分进行评估。

**📊 数据集**

基于 UCF101 的 15 个动作类别，共 1,932 个视频，每个视频生成 9 个隐私层级版本，形成 17,388 条目。

**📈 对比分析**

通过 R3D‑18 基线在同层级训练与跨层级测试（Clear → 加密）进行对比，结果显示原始精度 88.8% 降至加密背景去除层级的 53.5%，并绘制精度-隐私曲线和 PU 评分以量化权衡。

**⚠️ 局限性**

局限包括源视频低分辨率（320×240）、仅单人场景、仅空间隐私、块置换为唯一加密方式、训练轮次不均衡、姿态估计噪声以及有限的动作类别集。

---

## 317. A wearable haptic device for edge and surface simulation

**arXiv ID:** 2604.00752 | [PDF](https://arxiv.org/pdf/2604.00752v1)

**作者:** Rui Chen `[一作]` (School of Advanced Studies Sant'Anna), Daniele Leonardis `[通讯]` (School of Advanced Studies Sant'Anna)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

开发了一种双电机轻量化指尖触觉装置，能够区分表面与边缘触觉刺激

**💡 创新点**

通过双电机机制实现对表面和边缘接触的可区分刺激，并实现小型化、可穿戴化

**🔧 技术方法**

使用步进电机、齿轮箱、柔性压力传感阵列、ESP32控制盒等硬件与无线控制技术

**📊 数据集**

采用6×6 FSR传感阵列采集压力分布数据，未使用公开数据集

**📈 对比分析**

通过对四种刺激条件（边缘轻/重、表面轻/重）的用户分类实验，准确率平均93%，平均响应时长2.79秒

**⚠️ 局限性**

样本量小、缺乏闭环力控制、未实现多DOF边缘方向、噪声干扰可能影响感知、仅适用于单指接触

---

## 318. SoftHand Model-W: A 3D-Printed, Anthropomorphic, Underactuated Robot Hand with Integrated Wrist and Carpal Tunnel

**arXiv ID:** 2604.00738 | [PDF](https://arxiv.org/pdf/2604.00738v1)

**作者:** Dhillon B. Merritt `[一作]` (University of Bristol), Nathan F. Lepora `[通讯]` (University of Bristol)

**通讯引用:** 4848 | [OpenAlex ID](https://openalex.org/A5015897265)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了3D打印的SoftHand‑W手部，集成主动2自由度腕关节，并在UR5机器人臂上进行抓取、旋转和堆叠任务实验。

**💡 创新点**

创新点在于：① 将主动腕关节与SoftHand融合，并通过仿生腕管实现肌腱远程布置；② 采用3D打印降低成本并保持手部自适应协同；③ 通过腕关节显著提升手部的重定位能力。

**🔧 技术方法**

使用了3D打印PLA/光固化树脂部件、PTFE肌腱导管、Feetech STS3215舵机、UR5工业臂、Python+Feetech URT‑1控制器、以及基于Denavit–Hartenberg模型的正向运动学。

**📊 数据集**

实验数据集为自制的抓取对象：一块20 mm厚90 mm直径的盘和六块50 mm彩色立方体，用于旋转和堆叠任务。

**📈 对比分析**

通过对比有无腕动两种配置，评估任务完成时间、关节位移、配置变化数等指标。结果显示：腕动后旋转任务完成时间从66 s降至47 s（约29%缩短），堆叠任务成功率从5/6提升至6/6，关节位移（尤其是关节3、4）减少约36%。

**⚠️ 局限性**

限制包括：腕关节仅提供2自由度，缺乏拇指基角调节和指间张力微调；腕管位置限制运动范围；未评估对重物或高速动态任务的性能；缺乏触觉反馈和更复杂抓取姿态。

---

## 319. Online Network Slice Deployment across Multiple Domains under Trust Constraints

**arXiv ID:** 2604.00737 | [PDF](https://arxiv.org/pdf/2604.00737v1)

**作者:** Julien Ali El Amine `[一作]` (American University of the Middle East), Olivier Brun `[通讯]` (LAAS-CNRS)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在多域网络环境下，受信任约束的在线网络切片部署问题，提出了基于节点-链路（NL）与路径-链路（PL）两种整数规划模型，并引入基于Kleinrock拥塞函数的动态资源定价策略，以实现快速、低阻塞的切片落地。

**💡 创新点**

创新点在于：①首次将多域信任约束纳入在线VNF链部署模型；②通过预生成可信路径的PL模型将求解规模显著降低；③将动态定价与拥塞模型结合，提升资源利用率与吞吐量。

**🔧 技术方法**

采用整数规划（MILP）建模、Kleinrock拥塞函数实现动态定价、仿真评估以及比较不同负载与切片类型的性能指标。

**📊 数据集**

使用的是基于标准多域网络拓扑与VNF需求的仿真数据集（包含不同拥塞水平和切片类型的请求）。

**📈 对比分析**

与最优NL解相比，PL模型在低负载下几乎无解集差距，负载升高时差距可接受；计算时间缩短3–6倍；动态定价显著降低稀缺资源下的阻塞率。

**⚠️ 局限性**

局限性包括：仅考虑确定性请求，未加入随机性或鲁棒性分析；路径生成仍基于预设策略，缺乏自适应学习；信任模型为简单的可/不可穿越约束，未探讨更复杂的信任传递或激励机制。

---

## 320. A Benchmark of State-Space Models vs. Transformers and BiLSTM-based Models for Historical Newspaper OCR

**arXiv ID:** 2604.00725 | [PDF](https://arxiv.org/pdf/2604.00725v1)

**作者:** Merveilles Agbeti-messan `[一作]`, Stéphane Nicolas `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了基于状态空间模型（Mamba）的历史报纸 OCR，并与 Transformer、BiLSTM 及工业 OCR 引擎进行系统对比。

**💡 创新点**

创新点包括：首次将 Mamba 引入 OCR，设计三种解码策略（CTC、AR、NAR），并构建大规模可复现的 BnL 评测基准。

**🔧 技术方法**

技术方案为 CNN 视觉编码器 + 双向 Mamba 连接器 + 单向 Mamba 解码器，配合 CTC、前向自回归、静态查询非自回归解码；与 VAN、DAN、DANIEL、PERO‑OCR、Tesseract、TrOCR、Gemini 进行对标。

**📊 数据集**

使用 Bibliothèque nationale du Luxembourg (BnL) 的历史报纸数据，包含 33k 行（Fraktur/Antigua）和 976 段（严重退化）等多语言样本。

**📈 对比分析**

在相同训练与验证设置下对比 CER、WER、推理延迟、吞吐量与显存，结果显示 Mamba‑AR 取得 1.83% CER、比 DAN 快 2.9×；段落级 Mamba‑AR 取得 6.07% CER、比 DAN 快 2.05×；Mamba 线性显存扩展（1.26× vs 2.30×）。

**⚠️ 局限性**

局限性：NAR/CTC 对多行段落不适用；未对更长或更复杂布局进行建模；实验仅覆盖 Luxembourg 语料，缺乏对低资源或其他语言的验证；BPE 方案在 Fraktur 上效果不佳。

---

## 321. Using predefined vector systems to speed up neural network multimillion class classification

**arXiv ID:** 2604.00779 | [PDF](https://arxiv.org/pdf/2604.00779v1)

**作者:** Nikita Gabdullin `[一作]` (Joint Stock 'Research and production company 'Kryptonite'), Ilya Androsov `[通讯]` (Joint Stock 'Research and production company 'Kryptonite')

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预先配置好的向量系统 V_n^D 之下，利用嵌入向量的最大最小坐标索引实现 O(1) 的最近中心搜索，从而实现快速标签预测。

**💡 创新点**

创新点在于：①利用已知的向量系统结构将标签预测问题从 O(n) 降到 O(1)；②在不改变网络训练精度的前提下，能够在 GPU 内存受限时大幅提升推理速度；③通过返回未标记的最近中心，可辅助识别新类别。

**🔧 技术方法**

技术手段包括：向量系统的隐空间配置（LSC）、使用 topk 找到最大/最小索引、哈希表快速检索标签、PyTorch 实现的向量化运算。

**📊 数据集**

数据集：基于 ImageNet-1K 制作的 1M 类别数据集（通过复制图像并打上不同标签），以及 5M、10M、20M、30M、50M、100M 等大规模类别扩展集。

**📈 对比分析**

与传统的余弦相似度（cossim）方法进行比较。两者在准确率上完全一致；在标签预测时间上，新的方法在类别数超过 10M 且 GPU 内存受限时可达 11.6 倍的加速（K_t），单个搜索加速可达 11.6 倍；在更大类别数（50M、100M）下，cossim 甚至无法使用，而新方法仍可高效推理。

**⚠️ 局限性**

局限性：①需要先用 LSC 训练网络并保证隐空间满足 V_n^D 的结构；②中心字典（center_dict）需存放于 RAM，若类别数极大时可能需要分块；③方法仅适用于基于角度的损失函数；④未标记中心的识别新类别仍需进一步验证。

---

## 322. STCALIR: Semi-Synthetic Test Collection for Algerian Legal Information Retrieval

**arXiv ID:** 2604.00731 | [PDF](https://arxiv.org/pdf/2604.00731v1)

**作者:** M'hamed Amine Hatem `[一作]` (Higher School of Science and Technology of Information and Numerics), Faiçal Azouaou `[通讯]` (Higher School of Science and Technology of Information and Numerics)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 STCALIR 框架，利用半合成方式从阿尔及利亚法律文本构建测试集，并显著降低人工标注成本。

**💡 创新点**

创新点在于将多系统检索、Reciprocal Rank Fusion（RRF）聚合与交叉编码器再排序相结合，最终将人工评估限定在前10个候选，达到 99% 成本降低同时保持与人工标注高度一致。

**🔧 技术方法**

使用的技术包括：BM25、多个开源 bi‑encoder 与 cross‑encoder 模型、RRF、OCR 与结构化分块、以及基于 Web 的人工评估界面。

**📊 数据集**

数据集包括：原始阿尔及利亚官方公报法律文档、Mr. TyDi benchmark（用于验证）以及通过 STCALIR 生成的阿拉伯语法律文本数据集。

**📈 对比分析**

通过 Hit@10、MRR@10、nDCG@10 等评估指标与人类标注进行对比，semi‑synthetic 标注与人工标注的系统排名相关性达到 Kendall τ≈0.89、Spearman ρ≈0.92，Hit@10≈0.78，显示效果与人工标注相近。

**⚠️ 局限性**

局限性包括 OCR 质量对语料的影响、模型的泛化能力与领域特定性、以及对细粒度法律语义的捕捉仍需进一步微调。

---

## 323. MIRANDA: MId-feature RANk-adversarial Domain Adaptation toward climate change-robust ecological forecasting with deep learning

**arXiv ID:** 2604.00800 | [PDF](https://arxiv.org/pdf/2604.00800v1)

**作者:** Yuchang Jiang `[一作]` (University of Zurich), Vivien Sainte Fare Garnot `[通讯]` (University of Zurich)

**通讯引用:** 891 | [OpenAlex ID](https://openalex.org/A5085267838)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

针对气候变化导致的时间连续域偏移，提出了一种新型的中间特征层级对抗域自适应方法（MIRANDA），并将其应用于植物发育时序预测。

**💡 创新点**

创新点包括：①在中间层而非最终层应用对抗正则化，以兼顾协变量与标签共同偏移；②使用基于年份排名的对抗损失（Rank‑N‑Contrast）来捕捉连续域的相对距离；③引入混合层归一化（Hybrid Layer Normalization）以保持高层特征对标签漂移的灵活性。

**🔧 技术方法**

技术实现基于Transformer的PhenoFormer骨干，结合中间特征的对抗网络、Rank‑N‑Contrast损失、梯度逆转层以及混合层归一化；训练使用MSE + λRank损失。

**📊 数据集**

使用瑞士植物观测网络（Swiss Phenology Network）数据集，包含67,800条观测，涵盖5种树种，结合7种气象变量（温度、降水、气压、日照等）并构造三种域偏移拆分（时间序列、温度强度、海拔）。

**📈 对比分析**

与PhenoFormer、DANN、ADDA、CORAL、DANL、AdaBN等五种传统域自适应方法以及过程模型M1进行对比。实验显示，在三种拆分下，MIRANDA在R²上提升0.5–5%（尤其在海拔拆分中显著提升），RMSE和MAE均下降；与M1相比，缩小了R²差距至2–3%，并表现出更小的方差与更短的训练时间。

**⚠️ 局限性**

局限性包括：仍未能完全匹敌最优过程模型的性能；对额外驱动因子（如土壤湿度、土壤养分）的集成尚待验证；Rank‑N‑Contrast对极端年份的区分可能受样本稀缺影响；以及对不同树种标签漂移模式的适配需要进一步研究。

---

## 324. Multimodal Language Models Cannot Spot Spatial Inconsistencies

**arXiv ID:** 2604.00799 | [PDF](https://arxiv.org/pdf/2604.00799v1)

**作者:** Om Khangaonkar `[一作]` (University of California Davis), Hamed Pirsiavash `[通讯]` (University of California Davis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了一套基于多视角真实场景的空间不一致图像对，并用该数据集评估多模态大型语言模型在检测 3D 运动一致性方面的能力。

**💡 创新点**

提出了一个简单、可扩展的 cut‑and‑paste 生成方法，能够在保持光照、纹理不变的前提下诱发单一物体的 3D 运动不一致；同时搭建了专门针对 3D 空间一致性检测的 benchmark，并对模型表现进行细粒度分析。

**🔧 技术方法**

使用了实例分割、遮挡抠图、LaMa 纯图像修复、人工设定的裁剪与粘贴流程；在评估阶段采用强制选择（forced‑choice）提示，利用模型生成的文字答案进行解析；通过加权投票和集成策略进一步提升性能。

**📊 数据集**

核心数据来自 Hypersim 多视角室内场景数据集（含深度、光照、实例掩码等标注），并在附录中演示了可迁移到真实多视角场景（DL3DV）的可行性。

**📈 对比分析**

与人类基线（84.8%）和随机猜测（7.9%）进行对比，结果显示 GPT‑5（低推理）最高仅 34.2%，Qwen3‑VL 8B 27.6%，其他模型更低；集成后可达约 35%；同时分析了深度、光照、物体类别、场景类别、可解释性等因素对模型表现的影响。

**⚠️ 局限性**

局限性包括：生成方式主要是 cut‑and‑paste，可能无法覆盖所有真实世界的 3D 不一致；评估仅限于强制选择任务，未涉及自然语言推理；数据集聚焦于室内场景，缺乏多样性；模型未经过针对本 benchmark 的专门微调，导致表现偏差；对图像修复产生的细微伪影的影响尚待更细粒度验证。

---

## 325. Preference Guided Iterated Pareto Referent Optimisation for Accessible Route Planning

**arXiv ID:** 2604.00795 | [PDF](https://arxiv.org/pdf/2604.00795v1)

**作者:** Paolo Speziali `[一作]` (Vrije Universiteit Brussel), Diederik M. Roijers `[通讯]` (Gemeente Amsterdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 Preference Guided IPRO（PG‑IPRO）的交互式多目标最短路径规划方法，专门针对城市可达性导航中的有限移动人群需求。

**💡 创新点**

创新点在于将人类偏好直接嵌入到IPRO的递归搜索过程中，避免一次性生成完整 Pareto 前沿，仅通过有限次数交互即可快速收敛到用户满意的方案，显著提升响应速度。

**🔧 技术方法**

技术实现基于：IPRO 的分治框架、MO‑DFS（多目标深度优先搜索）Oracle、Manhattan 与 Chebyshev 归一化距离启发式、两种参照点选择启发式，以及对比基准 GPPE 的 Gaussian Process 偏好建模。

**📊 数据集**

实验数据集包括：① 用于验证前沿形状的合成双目标凸/凹 Pareto 前沿（30个点）；② 真实可达性路由实例 Osdorp‑Midden（7条非支配路径，目标为行驶距离与跨越次数）。

**📈 对比分析**

与 GPPE 进行比较时，PG‑IPRO 在前几轮查询中就能获得更高的用户效用，且单条路径平均生成时间仅 4.7 s，远低于 GPPE 所需的全前沿预计算时间 70 s；总体来看，PG‑IPRO 在早期查询与响应时间上优于基准方法。

**⚠️ 局限性**

局限性包括：仅在二维或低维目标空间验证，实验采用合成的噪声权重和基于Sigmoid的加权和用户模型，未覆盖更复杂的真实用户偏好；交互次数受限于实际可用性研究，尚未在真实用户中验证；对更高维度或更大规模图的可扩展性仍待探索。

---

## 326. HICT: High-precision 3D CBCT reconstruction from a single X-ray

**arXiv ID:** 2604.00792 | [PDF](https://arxiv.org/pdf/2604.00792v1)

**作者:** Wen Ma `[一作]` (Zhejiang University), Zuozhu Liu `[通讯]` (Zhejiang University)

**通讯引用:** 1154 | [OpenAlex ID](https://openalex.org/A5024343415)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出HiCT两阶段框架，利用单视角平面牙科X光图像重建高质量3D CBCT。

**💡 创新点**

创新点包括：①用视频扩散模型与相机姿态联合生成几何一致多视投影；②设计基于X射线物理的密度驱动混合采样和射线动态注意力网络，显著提升重建细节与一致性。

**🔧 技术方法**

技术手段包括视频扩散模型、CLIP交叉注意力、3D U‑Net去噪器、哈希编码+自注意力的射线动态注意力网络（RDA）以及密度驱动的X射线混合采样。

**📊 数据集**

使用自建XCT数据集，包含500对PX‑CBCT配对样本，并集成多源公开CBCT数据，规模大幅超越现有配对数据集。

**📈 对比分析**

与RealFusion、Zero123系列、SyncDreamer等视图合成方法以及FDK、ASD、SART、NeRF、NAF等重建方法对比，HiCT在PSNR、SSIM、LPIPS、IoU、Dice等指标上均显著领先，平均PSNR 39.21dB、SSIM 0.9735，提升幅度超过5.8dB。

**⚠️ 局限性**

局限性在于仍受单视角信息约束，缺乏广泛临床验证，对极端解剖变异的鲁棒性待进一步提升，同时对大量配对数据的依赖可能限制迁移性。

---

## 327. UK AISI Alignment Evaluation Case-Study

**arXiv ID:** 2604.00788 | [PDF](https://arxiv.org/pdf/2604.00788v1)

**作者:** Alexandra Souly `[一作]`, Xander Davies `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一套评估框架，用来测试在前沿AI实验室环境中，AI安全研究助手模型是否会主动破坏安全研究任务。

**💡 创新点**

创新点在于构造了高保真评估脚本（对比原始Petri框架），结合对话回滚、任务拆分和评估意识检测，提升了评估真实感和检测准确度。

**🔧 技术方法**

主要技术包括Petri自动评估框架、逆向工程Claude Code工具集、对话回滚、LLM评判器、配对比较与后验提问等方法。

**📊 数据集**

使用的数据集包含四种模型（Claude Opus 4.1、Claude Opus 4.5、Claude Sonnet 4.5、GPT‑5 预版）以及内部Claude Code部署记录，用于生成安全研究、一般研究以及无威胁等多组场景。

**📈 对比分析**

通过将改进脚本与原始Petri对比，评估了拒绝率、破坏行为、评估意识等指标；结果显示改进脚本在逼真度和误报率上都有显著提升，但未在任何模型中检测到破坏行为。

**⚠️ 局限性**

局限性包括：可能遗漏条件触发的破坏行为；仅覆盖有限的场景与研究方向；评估意识检测方法仍不成熟；仅聚焦安全研究破坏，未考虑其他风险路径。

---

## 328. The Rise of Language Models in Mining Software Repositories: A Survey

**arXiv ID:** 2604.00787 | [PDF](https://arxiv.org/pdf/2604.00787v1)

**作者:** Miguel Romero-Arjona `[一作]` (Universidad de Sevilla), Sergio Segura `[通讯]` (Universidad de Sevilla)

**通讯引用:** 4390 | [OpenAlex ID](https://openalex.org/A5057070358)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对2017-2025年85篇论文进行系统综述，构建语言模型在软件仓库挖掘中的应用体系和分类。

**💡 创新点**

创新点在于提出完整的LM应用taxonomy、揭示模型演进趋势并给出针对可复现性、工具维护与成本等方面的挑战与可操作建议。

**🔧 技术方法**

采用系统综述方法、文献筛选与编码、基于四个研究问题的定性统计与可视化。

**📊 数据集**

分析基于85篇研究的原始数据来源，共计92个仓库/数据集，包括GitHub、公开数据集和公开API。

**📈 对比分析**

作者未进行实验对比，而是通过统计和可视化展示LM在MSR中的任务分布、模型族使用频率与演变，表明从小型编码器到大型LLM的转变。

**⚠️ 局限性**

局限性包括仅聚焦MSR而排除代码生成任务、过度依赖公开数据、工业验证不足以及对可复现性与成本报告的欠缺。

---

## 329. Finding Low Star Discrepancy 3D Kronecker Point Sets Using Algorithm Configuration Techniques

**arXiv ID:** 2604.00786 | [PDF](https://arxiv.org/pdf/2604.00786v1)

**作者:** Imène Ait Abderrahim `[一作]` (Sorbonne University), Martin Durand `[通讯]` (Sorbonne University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了三维 Kronecker 序列的参数优化，以降低星差异度；

**💡 创新点**

通过 CMA‑ES 与 irace 自动配置获得了新的全尺寸最佳参数，使得 3D 点集在 500 点以上优于现有方法；

**🔧 技术方法**

采用了进化算法 CMA‑ES、自动配置工具 irace 以及后处理技术；

**📊 数据集**

使用的“数据集”为不同规模的点集（n=5~5000），并对 3D 与 4D 进行了实验；

**📈 对比分析**

与 Sobol’、L2_Subset、KRONECKER21、MPMC 等经典构造相比，在 3D 中小到中等规模竞争力十足，且在 n≥300 时取得新的记录；

**⚠️ 局限性**

局限在于高维（4D 及以上）性能不足，且参数与差异度的映射尚未完全理解，可能需要更深的理论与方法改进。

---

## 330. Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention

**arXiv ID:** 2604.00754 | [PDF](https://arxiv.org/pdf/2604.00754v1)

**作者:** Zehao Jin `[一作]` (Tsinghua University), Yanan Sui `[通讯]` (Tsinghua University)

**通讯引用:** 1150 | [OpenAlex ID](https://openalex.org/A5069290448)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种在滑动窗口注意力（SWA）基础上随机打乱令牌顺序，再恢复原序列的 Stochastic Attention（SA），并与 S​​WA 通过门控融合形成 SA+SWA 模型。

**💡 创新点**

其创新点在于通过在每层独立采样随机置换，保持 O(nw) 计算复杂度的同时实现指数增长的感受野，模仿果蝇神经网络的“小世界”结构，从而在有限预算下提供全局信息流。

**🔧 技术方法**

技术实现包括随机置换 + SWA、门控融合、与 FlexAttention、RoPE 等现有组件无缝对接，并在单层和多层复合中保持低额外开销。

**📊 数据集**

实验使用 6B-token SlimPajama 子集进行预训练，评测数据集包括 WikiText、LAMBADA、PIQA、HellaSwag、WinoGrande、ARC‑Easy；训练后推理在 Qwen3‑8B 和 Qwen3‑30B‑A3B 上通过 lm‑evaluation‑harness 的 7 个基准任务评估。

**📈 对比分析**

与全局注意力、纯 S​WA、随机滑动窗口（SA）、以及 MoBA 等方法在相同参数和计算预算下对比，SA+SWA 在预训练零样本平均准确率上达 35.9，几乎匹配全局注意；在 Qwen3 推理中，SA 在同等 compute 下比 SWA 更快恢复全局质量，并优于 MoBA。

**⚠️ 局限性**

局限性包括：需要在每层额外采样置换和索引操作，虽然开销小但在极长序列或极大模型上仍可能产生显著内存/计算负担；对梯度传播的影响有限，且缺乏针对不同任务的理论适配性分析。

---

## 331. Role Differentiation in a Coupled Resource Ecology under Multi-Level Selection

**arXiv ID:** 2604.00810 | [PDF](https://arxiv.org/pdf/2604.00810v1)

**作者:** Siddharth Chaturvedi `[一作]` (Radboud University), Marcel van Gerven `[通讯]` (Radboud University)

**通讯引用:** 9829 | [OpenAlex ID](https://openalex.org/A5074794877)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

提出并实现了一种多层次选择的计算模型，利用群体层面选择来优化共享的CTRNN控制基质和突变算子，并在包含可耕食与交换两种资源通道的boid模拟环境中研究在持续个体更替下的角色差异化；

**💡 创新点**

首次证明群体层面选择能够通过共享控制子系统和学习的突变算子间接促进零和交换通道的使用，从而在无明显角色分工的生态耦合设计中实现动态功能角色的涌现；

**🔧 技术方法**

使用基于Python的agent‑based模拟框架，布置连续时间递归神经网络（CTRNN）作为行为控制器，突变算子为多层感知机（MLP），个体层次最小阈值选择机制，群体层次使用协方差矩阵适应进化策略（CMA‑ES）；

**📊 数据集**

全部使用模拟生成的数据，采用多种随机初始条件（S=2）进行多场景评估，未使用外部真实数据集；

**📈 对比分析**

通过消融实验比较三种设置（完整模型、仅共享控制子系统+噪声突变、随机控制子系统+噪声），用每时间步的净资源收益和交换通道正向资源量进行度量，结果显示完整模型在资源获取与交换活跃度上显著优于消融方案；

**⚠️ 局限性**

受限于仅在仿真环境中验证，突变算子对性能的提升有限，角色差异化保持动态而非稳定，模型仅涵盖两种资源通道，缺乏对真实生态系统的直接验证，且训练时间与超参数空间可能影响结果的进一步提升。

---

## 332. Approximation Algorithms for Budget Splitting in Multi-Channel Influence Maximization

**arXiv ID:** 2604.00796 | [PDF](https://arxiv.org/pdf/2604.00796v1)

**作者:** Dildar Ali `[一作]` (Indian Institute of Technology Jammu), Suman Banerjee `[通讯]` (Indian Institute of Technology Jammu)

**通讯引用:** 5722 | [OpenAlex ID](https://openalex.org/A5033218913)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出在限预算下将资金在户外广告（数字广告牌）与社交媒体广告之间分配，以最大化整体影响力。

**💡 创新点**

1) 引入交互效应（Interaction Effect）来刻画广告牌与社交媒体共同作用；2) 定义非双子模（non‑bisubmodular）影响函数 Φ 并给出其性质；3) 提出基于随机贪婪和两阶段自适应贪婪（TPG）的算法，并给出 1/α(1‑e⁻γ α) 的近似保证。

**🔧 技术方法**

基于集合函数理论（子模/非子模性质）、双子模比（bisubmodularity ratio）与曲率（curvature）的理论分析；随机采样贪婪策略；以及延迟评估（lazy evaluation）技术；同时使用独立传播模型（ICM）进行模拟。

**📊 数据集**

美国（USA）与加拿大（Canada）真实数据集：轨迹数据（约 124k / 210k 检查点），社交网络（约 130k / 50k 友谊边），以及广告牌槽位（约 3.2M / 1.77M 槽位）。

**📈 对比分析**

与四种基线（随机分配、Top‑k、HDH、PRS）对比。实验显示：①在预算 500–2000 之间，随机贪婪和 TPG 的总影响力约为基线的 2–3 倍；②预算分配比例上，算法倾向于将更多资金投向广告牌；③运行时间方面，随机贪婪比 TPG 更快；总体而言，算法在影响力与预算利用率上优于基线。

**⚠️ 局限性**

①算法的时间复杂度仍较高，尤其在大规模数据上计算交互效应和模拟传播需耗时；②假设影响概率和成本模型固定，未考虑动态变化；③实验仅覆盖两国两种广告渠道，缺乏更广泛的跨平台验证；④理论近似比虽给出上界，但实际 γ 与 α 值难以精确估计。

---

## 333. RefineRL: Advancing Competitive Programming with Self-Refinement Reinforcement Learning

**arXiv ID:** 2604.00790 | [PDF](https://arxiv.org/pdf/2604.00790v1)

**作者:** Shaopeng Fu `[一作]` (KAUST), Furu Wei `[通讯]` (Microsoft Research)

**通讯引用:** 31089 | [OpenAlex ID](https://openalex.org/A5014662947)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种利用LLM自我改进能力解决竞赛编程（CP）问题的系统，包含 Skeptical-Agent（持续怀疑式自我改进）和 Self‑Refinement RL（通过强化学习培养自我改进能力）。

**💡 创新点**

创新点在于：① 通过 Skeptical‑Agent 强制模型在即使通过公共测试用例后仍继续改进；② Self‑Refinement RL 利用自生成的改进轨迹和新的平方奖励函数，显著提升模型的调试和推理深度；③ 采用稠密的平方奖励而非传统的通过/失败二值奖励。

**🔧 技术方法**

技术包括：LLM 推理（Qwen3‑4B/4B‑2507），本地执行工具（跑代码验证公共测试），强化学习框架（GRPO），自我改进循环逻辑，平方奖励函数和自我生成的反馈模板。

**📊 数据集**

使用 4.5K 题目的 HardTest 数据集生成自我改进轨迹；评估采用 LiveCodeBench v5 和 v6（共 342 道题）作为测试集；还与微软 Research 和其他开源模型（如 Qwen3‑8B、Qwen3‑32B、Qwen3‑235B 等）做对比。

**📈 对比分析**

通过多种推理时间放大策略（Random@16、LongCoT@16、RejSamp@16、Reflexion@16）与 Skeptical@16 进行对比。结果显示：RL 训练后 4B 模型已能匹敌 8B 级别，4B‑2507 能接近 32B 级别；在 Skeptical‑Agent 下，4B‑2507 的性能甚至逼近 235B 单次推理水平。平方奖励的 ablation 证明其优于传统通过/失败奖励。

**⚠️ 局限性**

局限性：1）仍依赖公共测试用例作为局部反馈，无法完全保证通过私有测试；2）自我改进轨迹的生成成本较高，且在更大规模或不同领域的通用性尚未验证；3）需要本地执行环境，部署成本与安全性需要考虑；4）对极其复杂或极限案例的覆盖率仍有待提升。

---

## 334. Scalable Pretraining of Large Mixture of Experts Language Models on Aurora Super Computer

**arXiv ID:** 2604.00785 | [PDF](https://arxiv.org/pdf/2604.00785v1)

**作者:** Dharma Teja Vooturi `[一作]` (Parallel Computing Lab India), Bharat Kaul `[通讯]` (Intel Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Aurora超级计算机上利用1000+个Intel GPU tile，使用自研的Optimus训练库对Mula系列LLM（从1B到220B参数）进行从头开始的预训练，训练数据达4万亿token，并展示模型规模和计算规模的扩展能力。

**💡 创新点**

创新点包括：① 设计并实现了Optimus训练框架，支持Tensor、Expert、Pipeline并行、sharded optimizer、EP‑Aware sharded optimizer（EPSO）以及FastSparseMoE自定义GPU kernel；② 在Aurora上首次实现大规模MoE训练，并实现90%扩展效率；③ 提供完整的可靠性与容错体系（数据预处理、模型广播、双重/持久化检查点、DP‑Scatter写入、硬/软节点失败处理）。

**🔧 技术方法**

采用的技术手段有：混合精度BF16、AdamW优化器、Sharded Optimizer、EPSO、FastSparseMoE、Selective Activation Checkpointing、Tensor Parallelism、Expert Parallelism、Pipeline Parallelism、Allgather/ReduceScatter通信、OneCCL自定义通信、数据预处理流水线、模型广播、双重检查点、DP‑Scatter写入、节点容错机制等。

**📊 数据集**

使用的训练数据为4万亿token的OLMoE‑Mix‑0924语料库；评估基准包括ARC、HellaSwag、Piqa、Boolq、SciQ、Winogrande、OpenBookQA、MMLU等常用LLM benchmark。

**📈 对比分析**

通过与OLMoE‑1B‑7B‑0924的benchmark对比，Mula‑7B‑A1B在相同计算量下准确率提升6.7%；Mula系列模型从20B到220B参数，训练至100B token，损失随规模下降；计算扩展从384到12288 GPU tile，保持约90%扩展效率；FastSparseMoE + EPSO在前向/后向/优化器层面分别提升1.33–2.83×、1.07–1.36×，整体训练加速1.11–1.71×。

**⚠️ 局限性**

局限性：MoE模型计算非均匀导致扩展在大规模时略有下降；对硬件（Aurora GPU tile、OneCCL）高度依赖，难以直接迁移；未展示下游微调或实际应用效果；高内存需求（每tile 64GB）限制了可训练模型；容错与检查点机制虽完善，但在极端失败场景下仍需进一步验证。

---

## 335. An Approach to Enriching Surgical Video Datasets for Fine-Grained Spatial-Temporal Understanding of Vision-Language Models

**arXiv ID:** 2604.00784 | [PDF](https://arxiv.org/pdf/2604.00784v1)

**作者:** Lennart Maack `[一作]` (Hamburg University of Technology), Alexander Schlaefer `[通讯]` (Hamburg University of Technology)

**通讯引用:** 2809 | [OpenAlex ID](https://openalex.org/A5087348362)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套名为 SurgSTU-Pipeline 的确定性流程，利用手术视频及其可选元数据（如器械-组织交互、器械定位信息），通过时间连续性与空间连续性过滤，生成细粒度时空问答对，进而创建 SurgSTU 数据集并进行基准评估。

**💡 创新点**

创新点主要包括：① 采用 deterministic pipeline，避免 LLM 产生幻觉；② 引入时间与空间连续性过滤，提升生成 QA 的准确性；③ 通过事件元组（Event Tuple）统一描述视频帧、bbox、动作与目标；④ 提供基于模板的多种时空任务类型；⑤ 设计 deterministic 评估协议，直接解析模型输出的坐标与时间，兼顾 IoU、时空误差等指标。

**🔧 技术方法**

技术方法包括：活跃学习 (active learning) 迭代框标注，物体检测模型推理；事件元组解析与结构化；预定义任务模板与自动 QA 生成；连续性过滤算法；大语言模型（如 Qwen3-VL 2B、Gemini-3.1 Flash‑Lite）零样本推理、ICL 调优；使用 LoRA 进行域特定 fine‑tune；基于规则的评估脚本。

**📊 数据集**

数据集：从公开的胆囊切除（CholecT50）和前列腺切除（ProstaTD）视频中提取，重建 30 fps 连续序列并插值 1 fps 的器械-组织交互，最终得到 7,515 条视频片段、150k 细粒度 QA 对；也将其与公开数据集（Cholec80、GraSP）关联。

**📈 对比分析**

比较方法：在 zero‑shot、ICL、以及对 SurgSTU 训练集 fine‑tune 后的三种设置下，分别评估 Qwen3‑VL 2B、Gemini‑3.1 Flash‑Lite、EndoChat 的 7 大核心任务。结果显示：zero‑shot 时表现较差；ICL 可显著提升 2-4 倍；对 SurgSTU 进行 fine‑tune 后，Qwen3‑VL 2B 在所有任务中均取得最高分（如 60.88% 时空定位、70.42% 速度估计、98.42% 多选计数）。

**⚠️ 局限性**

局限性：① 生成 QA 仍可能携带由元数据噪声、框标注误差或插值误差引入的错误；② 对手术种类与器械差异的泛化尚未验证，可能出现过拟合；③ EndoChat 等基于静态图像的模型在连续时空推理上表现差；④ 需要大量人工标注才能支持更复杂语义任务（如完整的器械‑组织交互描述）。

---

## 336. Cost-Penalized Fitness in FMA-Orchestrated Mixture of Experts: Experimental Evidence for Molecular Memory in Domain Adaptation

**arXiv ID:** 2604.00812 | [PDF](https://arxiv.org/pdf/2604.00812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 337. IWP: Token Pruning as Implicit Weight Pruning in Large Vision Language Models

**arXiv ID:** 2604.00757 | [PDF](https://arxiv.org/pdf/2604.00757v1)

**作者:** Dong-Jae Lee `[一作]` (KAIST), Junmo Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大型视觉语言模型（LVLM）的视觉令牌过多导致计算成本飙升问题，作者提出了一种基于自注意力双形式（dual form）的无训练令牌剪枝框架，能够在不训练的情况下挑选最具信息量且低冗余的视觉令牌。

**💡 创新点**

创新点在于将softmax注意力重新解释为隐式线性变换，令牌的影响被视为对双重权重矩阵的秩-1更新，从而引入了同时考虑信息量与冗余的双重度量，并通过Progressive Chunked Maximal Marginal Relevance（PC‑MMR）高效地完成子集选择。

**🔧 技术方法**

技术主要包括：自注意力双形式推导、基于核函数的映射（exp核）、秩-1更新的Frobenius范数与相似度度量、信息量度量 Score_i = κ(q,k_i)‖v_i‖、信息冗余度量 S_ij，和PC‑MMR算法；实现上使用FlashAttention2和PyTorch的Scaled Dot‑Product Attention。

**📊 数据集**

实验数据集覆盖图像理解、视频理解与多模态推理，分别使用 AI2D、TextVQA、DocVQA、Infographic VQA、MMBench、MMMU、MMStar、SciQA、MME、POPE、EgoSchema、Video‑MME、MLVU、NExT‑QA 等，模型基于 LLaVA‑OneVision‑7B 与 Qwen2.5‑VL‑7B。

**📈 对比分析**

与 FastV、Pact、VisionZip、DivPrune 等现有剪枝方法对比，作者方法在多种token预算（35.3%、22.2%、11.1%）下在所有基准上均取得最高或近乎最高的相对性能，同时生成时间和显存占用也有显著下降，显示出更优的性能‑效率权衡。

**⚠️ 局限性**

限制在于目前框架主要验证于视觉模态，虽然其理论上可推广到音频、语音或点云等高密度连续模态，但实际跨模态应用效果仍待进一步探索。

---

## 338. Is RISC-V Ready for Machine Learning? Portable Gaussian Processes Using Asynchronous Tasks

**arXiv ID:** 2604.00736 | [PDF](https://arxiv.org/pdf/2604.00736v1)

**作者:** Alexander Strack `[一作]` (University of Stuttgart), Dirk Pflüger `[通讯]` (University of Stuttgart)

**通讯引用:** 1403 | [OpenAlex ID](https://openalex.org/A5041326099)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

扩展了 GPRat 库以支持 ARM 和 RISC‑V 架构，并在三种处理器上评估了高斯过程回归（GPR）的强缩放与问题规模缩放性能。

**💡 创新点**

首次在 RISC‑V 上实现并评测 GPR，首次对 x86‑64、ARM A64FX 与 RISC‑V SG2042 的 GPR 预测与超参优化进行横向对比；通过 HPX 任务并行实现可移植的异步块式 Cholesky 分解。

**🔧 技术方法**

采用 HPX 并行运行时、OpenBLAS（兼容 ARM、RISC‑V）与 Intel oneMKL、FP64 BLAS/LAPACK；使用块式 Cholesky、矩阵乘法等线性代数操作实现 GPR。

**📊 数据集**

使用自行生成的非线性质量‑弹簧‑阻尼模拟数据（可通过 GPRat 仓库的模拟器生成），并公开了数据集链接。

**📈 对比分析**

通过在同一节点上多核心强缩放和不同问题规模的实验进行比较；结果显示 ARM 在 48 核上比 x86‑64 Zen 2 更快（+9%），RISC‑V 单核性能和扩展性显著落后（单核 14 倍慢，规模 25 倍慢）。

**⚠️ 局限性**

受限于 RISC‑V 缺乏宽寄存器向量化（RVV 1.0 尚未实现）和内存控制器效率低下，导致单核与大规模并行工作负载性能瓶颈；实验仅覆盖单节点、FP64 BLAS，未涉及 GPU/加速器或更大规模分布式场景。

---

## 339. A CEFR-Inspired Classification Framework with Fuzzy C-Means To Automate Assessment of Programming Skills in Scratch

**arXiv ID:** 2604.00730 | [PDF](https://arxiv.org/pdf/2604.00730v1)

**作者:** Ricardo Hidalgo-Aragón `[一作]` (Universidad Rey Juan Carlos), Gregorio Robles `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 6920 | [OpenAlex ID](https://openalex.org/A5061131972)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于 CEFR 级别的 Scratch 项目评估框架，利用模糊 C‑均值聚类对程序的九维计算思维指标进行软划分，并引入置信度度量与过渡学习者识别。

**💡 创新点**

创新点在于将模糊聚类与 CEFR 级别的序数映射（S_j 标准）相结合，既保留了连续学习进程的软边界，又通过置信度阈值实现自动化与人工干预的动态切换。

**🔧 技术方法**

采用的技术包括模糊 C‑均值聚类、序数 S_j 排序、基于熵的置信度计算、交叉验证、PCA 可视化及与 MiniBatchKMeans、DBSCAN 的对比实验。

**📊 数据集**

使用的数据集为公开的 Scratch 项目（约 1,608,246 个项目），通过 Dr.Scratch 计算得到的九维 0–4 级别计算思维指标，并已上传至 Kaggle。

**📈 对比分析**

与 MiniBatchKMeans、DBSCAN 对比后，模糊聚类在 Silhouette（≈0.257）、FPC（≈0.102）、平均置信度（≈0.566）等指标上保持稳健，训练与测试误差仅约 0.6%，并能识别 13.7% 的过渡学习者。

**⚠️ 局限性**

局限性包括仅适用于 Scratch，缺乏动态跟踪与跨语言推广，数据文化偏向英文国家，且 Dr.Scratch 仅评估功能性 CT，未覆盖代码质量与创造性维度。

---

## 340. Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM

**arXiv ID:** 2604.00804 | [PDF](https://arxiv.org/pdf/2604.00804v1)

**作者:** Monica M. Q. Li `[一作]`, Giovanni Beltrame `[通讯]` (Polytechnique Montreal)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一套多机器人3D高斯Splatting SLAM系统Coko-SLAM，可在不依赖初始相对姿态的情况下完成地图融合，同时显著降低通信量。

**💡 创新点**

创新点包括：①基于特征向量的实时关键帧筛选；②将GaussianSPA实时压缩算法嵌入3D高斯训练以实现子图体素压缩；③提出纯3D高斯子图的循环闭环机制，避免了对渲染图像或初始姿态的依赖。

**🔧 技术方法**

使用了3D Gaussian Splatting、DINO-V2特征提取、GaussianSPA压缩、FPFH+RANSAC+ICP粗细配准、GTSAM Pose Graph Optimization、渲染深度与相机深度两种模式。

**📊 数据集**

在Replica合成数据集和Aria真实数据集上进行评估。

**📈 对比分析**

与CP-SLAM、MAGiC-SLAM、MAC-Ego3D等方法对比，Coko-SLAM在训练视图和新视图的PSNR/SSIM/LPIPS指标上均领先或相当；数据传输量降低至85-95%，子图尺寸和关键帧数量亦与或优于基线。

**⚠️ 局限性**

主要限制是：对渲染深度的依赖在低分辨率场景下性能下降；系统仍依赖中心服务器，缺乏完全去中心化；对图像分辨率的敏感性使得在不同硬件上需要进一步优化。

---

## 341. RePart: Efficient Hypergraph Partitioning with Logic Replication Optimization for Multi-FPGA System

**arXiv ID:** 2604.00780 | [PDF](https://arxiv.org/pdf/2604.00780v1)

**作者:** Zizhuo Fu `[一作]` (Peking University), Yibo Lin `[通讯]` (Peking University)

**通讯引用:** 4355 | [OpenAlex ID](https://openalex.org/A5000933188)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向多FPGA系统的定制化多层次超图划分框架，结合逻辑复制与拓扑感知优化，显著降低跨FPGA通信成本。

**💡 创新点**

设计了FPGA感知的动态粗化策略、基于热值的分配算法以及支持复制与删除的四操作细化机制，三者协同提升划分质量与资源利用。

**🔧 技术方法**

采用多层次超图划分（粗化-分配-细化）框架，配合自适应惩罚系数、热值度量、深度回溯搜索、1+3K堆操作管理、增益增量更新等技术实现高效划分。

**📊 数据集**

在Titan23电路网表、EDA Elite Challenge Contest的10个测试案例（含SampleInput、synopsys02等）以及ICCAD'19 Contest的两套多FPGA实例上进行实验。

**📈 对比分析**

与KaHyPar、TopoPart、Li等基线及竞赛冠军方案对比，平均总跳数降低52.3%、运行时间下降98.1%（约11倍加速），在竞赛方案上比分别低14%和48%，展示显著性能优势。

**⚠️ 局限性**

方法主要针对静态多FPGA拓扑与资源约束，仍依赖启发式搜索，可能无法覆盖所有极端规模或动态拓扑；评估指标聚焦总跳数，未全面考虑时延/带宽等实际部署因素。

---

## 342. Thinking Wrong in Silence: Backdoor Attacks on Continuous Latent Reasoning

**arXiv ID:** 2604.00770 | [PDF](https://arxiv.org/pdf/2604.00770v1)

**作者:** Swapnil Parekh `[一作]` `[通讯]` (Intuit), Swapnil Parekh (Intuit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对连续隐状态推理模型的后门攻击，利用在输入层注入的单个嵌入向量并通过多轮前向传播放大，从而在不影响正常输出的情况下强制模型给出攻击者指定的答案。

**💡 创新点**

创新点在于首次将后门攻击迁移到完全基于隐状态的链式推理框架，并揭示了神经折叠（Neural Collapse）在隐藏轨迹中形成几何吸引子，从而解释了该攻击对多种防御手段的鲁棒性，并给出了高-ASR后门必然留下线性可分特征的理论边界与三层检测层级。

**🔧 技术方法**

技术手段包括：在训练时联合优化触发嵌入φ与模型参数的Poisoning学习；多通道多步Transformer推理；对隐状态向量的线性探针与神经折叠度量；以及对五种主动防御（噪声注入、强制曝光、方向投影、神经元修剪、无监督聚类）的系统评估。

**📊 数据集**

实验使用了ProntoQA、ProsQA、GSM8K、SVAMP、MultiArith和GSM-Hard六个推理基准，覆盖逻辑推理、图推理、数学问题以及不同难度与结构的跨域数据。

**📈 对比分析**

与基准模型和现有防御的对比表明，攻击在124M GPT-2和1B/3B SimCoT上均能实现≥99%攻击成功率，且对齐的清洁准确率仅下降≤1.5%；攻击在未见过的测试集上保持94–100%成功率；五种主动防御均未能将成功率显著降低，同时模型在25轮清洁微调后仍保持高成功率。

**⚠️ 局限性**

主要局限包括：检测仍需访问隐藏状态（除嵌入行检查外）且在大规模模型（>3B）下的神经折叠与检测效果尚未完全验证；攻击对不同架构（如非Transformer或带有自监督预训练的模型）的泛化性待进一步探究；以及潜在的防御可通过改造投影层或引入非线性正则化等手段进行。

---

## 343. A Dual-Action Fabric-Based Soft Robotic Glove for Ergonomic Hand Rehabilitation

**arXiv ID:** 2604.00768 | [PDF](https://arxiv.org/pdf/2604.00768v1)

**作者:** Rui Chen `[一作]` (Scuola Superiore Sant'Anna), Antonio Frisoli `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 8482 | [OpenAlex ID](https://openalex.org/A5090204404)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文开发了一款双向可调节、基于织物的软体机械手套，可为每个手指配备自定义的双动作气动执行器，实现屈伸与拇指外展的全手功能。

**💡 创新点**

创新点在于将对称腔体结构与CNC热封技术相结合，产生内凹表面并可按患者手部几何精确定制，实现舒适、精准的双向屈伸与拇指外展支撑。

**🔧 技术方法**

采用对称腔体织物气动执行器、CNC热封成型、可调压气动控制、ESP32微控制器与智能手机Web界面相结合的软体机器人技术。

**📊 数据集**

实验数据来自10名健康男性志愿者与3名C5–C7级脊髓损伤患者的功能测试与肌电记录，构成手部功能与肌肉活动的评估数据集。

**📈 对比分析**

与无手套、被动手套以及文献中单向屈伸手套相比，本手套在屈伸角度、抓握力（最高24.8 N）和肌电活动下降（最高37 %）上均表现优异，且在脊髓损伤患者的功能任务中提升了抓握稳定性并减少了对十指屈肌牵拉。

**⚠️ 局限性**

主要局限包括：手动测量与单件制作用时长与成本高、受试者样本量小且性别单一、仅覆盖全屈伸/全伸张两种姿态、按钮式控制导致完成时间延长、缺乏闭环意图检测与自适应控制、以及未对长期耐久性与多次充放气循环的可靠性进行评估。

---

## 344. An Unconditional Barrier for Proving Multilinear Algebraic Branching Program Lower Bounds

**arXiv ID:** 2604.00746 | [PDF](https://arxiv.org/pdf/2604.00746v1)

**作者:** Deepanshu Kush `[一作]` `[通讯]` (University of Cambridge), Deepanshu Kush (University of Cambridge)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

论文探讨了多线性代数分支程序（mABPs）的超多项式下界问题，证明了最小划分秩方法无法证明mABPs的超多项式下界，并展示了存在一个全秩多线性多项式可以由多项式大小的mABP计算。

**💡 创新点**

创新点在于证明了最小划分秩方法的局限性，指出需要新的技术来区分mABP与多线性层次中的更高类，并且给出了N(n) = n^O(1)的结果，表明存在1平衡链集系统的大小是多项式级别的。

**🔧 技术方法**

使用了最小划分秩方法和超马丁格尔论证等技术，结合了组合学和概率论的工具。

**📊 数据集**

使用了平衡链集系统的组合性质，具体数据集未明确提及，但涉及到的多项式和组合对象是基于n的大小。

**📈 对比分析**

与Fabris等人的方法进行了比较，指出他们的方法在处理不平衡时存在局限性，而本研究通过双块引导策略有效控制了不平衡，证明了在每个步骤中不平衡的增加概率小于1/4，从而实现了更强的结果。

**⚠️ 局限性**

限制在于当前的结果并不直接表明mABP与mVBP之间的关系，且全秩性这一特性不足以强制大规模的mABP，表明在多线性模型中仍需探索新的方法来证明下界。

---

## 345. Spectral Compact Training: Pre-Training Large Language Models via Permanent Truncated SVD and Stiefel QR Retraction

**arXiv ID:** 2604.00733 | [PDF](https://arxiv.org/pdf/2604.00733v1)

**作者:** Björn Roman Kohlberger `[一作]` `[通讯]` (EctoSpace), Björn Roman Kohlberger (EctoSpace)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种永久采用截断SVD并在Stiefel流形上进行重投影的低秩训练方法SCT，用以显著降低大语言模型训练的显存消耗，并在70B模型上验证了完整训练步骤仅需7.2GB。

**💡 创新点**

首次将截断SVD作为权重的完整表示并在训练过程中通过QR重投影保持正交，避免任何一次全秩矩阵的材料化，从而实现近200倍的内存压缩。

**🔧 技术方法**

采用低秩SVD分解、Stiefel流形优化（QR重投影）、标准反向传播、AdamW优化器，并在MLP层使用SpectralLinear层实现高效前向后向计算。

**📊 数据集**

在SmolLM2-1.7B上使用Alpaca数据集进行微调验证，同时在SmolLM2-135M上做梯度完整性测试，70B架构在Steam Deck和Apple M4 Pro上测试。

**📈 对比分析**

与全秩FP32训练和LoRA等后训练压缩方法对比，SCT在rank 128时实现11.7×压缩、GPU内存19GB、训练步长0.74s、困惑度65.6，显示显存降低46%，速度提升2.1倍，但与dense训练仍有约3点的损失差距。

**⚠️ 局限性**

主要限制在于学习率配置导致的收敛差距、QR重投影成本占比高、未将注意力层转为低秩、在小模型上压缩效果有限，以及缺乏完整预训练实验。

---

## 346. Exploring Silent Data Corruption as a Reliability Challenge in LLM Training

**arXiv ID:** 2604.00726 | [PDF](https://arxiv.org/pdf/2604.00726v1)

**作者:** Anton Altenbernd `[一作]` (Technische Universität Berlin), Odej Kao `[通讯]` (Technische Universität Berlin)

**通讯引用:** 4733 | [OpenAlex ID](https://openalex.org/A5042349846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在单个GPU上对LLaMA模型进行针对GPU矩阵乘法指令的故障注入，系统地研究了静默数据损坏（SDC）对大型语言模型（LLM）预训练的影响，并提出了一种基于梯度和参数更新的轻量级检测方法，检测到有害更新后可通过重算上一步来抑制损失波动；

**💡 创新点**

创新点在于：①首次系统地将目标性故障注入与LLM训练结合，揭示不同位点、核函数、前向/后向阶段对损失与参数的敏感性；②设计了一种利用RMS更新量与梯度范数的异常检测机制，能够快速识别有害的参数更新；③提出在检测到异常时重算训练步的做法，实现了低开销、高效的SDC缓解；

**🔧 技术方法**

技术手段包括：NVBit动态二进制插桩实现GPU指令级故障注入；bfloat16精度的LLaMA预训练；AdamW优化器与梯度范数裁剪；基于RMS更新量与梯度范数的异常检测与重算逻辑；

**📊 数据集**

使用数据集为C4英文子集；

**📈 对比分析**

实验通过对60M、350M、1.3B三种LLaMA模型在10,000步训练中，分别比较基线、注入故障、以及注入+重算三种方案；结果显示重算能把最终评估损失恢复至基线水平，且检测+重算对训练速度的影响约为1%的开销；注入故障会导致约30-40步的训练进度损失；

**⚠️ 局限性**

局限性包括：实验仅在单GPU环境下进行，未覆盖大规模分布式训练；故障模型仅针对GEMM核，未考虑其他运算或硬件模块；只验证了LLaMA+AdamW架构，缺乏对更大模型或不同优化器的验证；故障注入率为压力测试，可能与实际生产环境不完全一致；

---

## 347. Fast Deterministic Distributed Degree Splitting

**arXiv ID:** 2604.00724 | [PDF](https://arxiv.org/pdf/2604.00724v1)

**作者:** Yannic Maus `[一作]`, Florian Schager `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在LOCAL模型下提出了改进的度划分（directed/undirected）算法，并利用其实现了更快的多色边着色；

**💡 创新点**

核心创新在于用超图无源定向（HSO）与投票块技术构建分度划分，得到 O(ε^{-1}log n) 轮次的 deterministic 方案，并证明此复杂度对使用增广路径方法是最优；

**🔧 技术方法**

技术手段包括虚拟复制、路径/循环分割、(α,β)-规则集、超图无源定向、随机化加速、递归分割与合并，以及 LCL（局部可检查标记）框架；

**📊 数据集**

本工作为理论研究，无实验数据集；

**📈 对比分析**

与之前的 Ghaffari‑Su 等算法相比，deterministic 版在任意图中实现了从 O(ε^{-1}log ε^{-1}(log log ε^{-1})^{1.71}log n) 到 O(ε^{-1}log n) 的显著提升；在多边着色中实现了 (3/2+ε)Δ 色彩，时间为 O(ε^{-1}log^2Δ·log n+ε^{-2}log n)，比先前的 O(ε^{-2}log^2Δ polyloglogΔ·log n) 更快；随机化版本进一步降低到 O(ε^{-1}log ε^{-1}·log log n) 轮次；

**⚠️ 局限性**

尽管已突破以往上限，但仍受 Θ(log n) 的下界约束；证明 O(ε^{-1}log n) 对增广路径法是最优，但是否能在不受此限制的其他方法中进一步削弱 ε^{-1} 因子仍未解决；多路划分仍需额外 O(log^2k) 复合成本；算法对 HSO 的度与秩约束较强；mending radius 的大幅度表明局部修正极其困难。

---

## 348. A novel three-step approach to forecast firm-specific technology convergence opportunity via multi-dimensional feature fusion

**arXiv ID:** 2604.00803 | [PDF](https://arxiv.org/pdf/2604.00803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 349. LangMARL: Natural Language Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.00722 | [PDF](https://arxiv.org/pdf/2604.00722v1)

**作者:** Huaiyuan Yao `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**通讯引用:** 7667 | [OpenAlex ID](https://openalex.org/A5100777770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LangMARL 框架，让多语言模型代理通过信用分配与梯度演化实现自主协作策略演进

**💡 创新点**

创新点包括：① 基于 MARL 的代理级语言信用分配；② 在语言空间实现策略梯度估计与优化；③ 通过重放轨迹总结因果关系以提供稠密反馈，提升样本效率和可解释性

**🔧 技术方法**

采用 CTDE 结构，语言策略演员、集中式语言评论家、语言策略梯度估计器和语言策略优化器，借鉴 counterfactual baseline、value decomposition 等 MARL 技术；使用 LLM 作为核心计算单元

**📊 数据集**

数据集涵盖：编码任务 HumanEval、推理任务 HotPotQA、数学任务 MATH；以及多智能体游戏 Overcooked‑AI 与 Pistonball

**📈 对比分析**

与静态提示（CoT、Agents）、自适应提示（AutoPE、DSPy）、语言梯度自演化（Reflexion、TextGrad）和符号学习（Symbolic）进行对比；实验显示在所有基准上均取得最高准确率/通关奖励，收敛速度快且样本效率高

**⚠️ 局限性**

局限性在于：在长时序稀疏奖励任务（如复杂软件开发）表现不佳；缺乏动态子代理合成与自适应结构扩展能力

---

## 350. Multicentric thrombus segmentation using an attention-based recurrent network with gradual modality dropout

**arXiv ID:** 2604.00817 | [PDF](https://arxiv.org/pdf/2604.00817v1)

**作者:** Sofia Vargas-Ibarra `[一作]` (Universite Evry Paris-Saclay), Sonia Garcia-Salicetti `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 2034 | [OpenAlex ID](https://openalex.org/A5044972551)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种结合逐步模态丢失和注意力递归网络（UpAttLLSTM）的多模态MRI小病灶（血栓）分割方法，能够在多中心、缺失模态的情况下保持鲁棒性。

**💡 创新点**

① 逐步模态丢失（Gradual Modality Dropout）作为正则化/数据增强，模拟真实多中心模态缺失；② UpAttLLSTM融合交叉注意力与轻量化Logic LSTM，利用病灶信息引导血栓检测；③ 结合距离约束与阈值扩展的后处理，显著减少假阳性。

**🔧 技术方法**

2.5D卷积、视觉Transformer Patch Embedding、交叉注意力、Logic LSTM、nnU-Net、Nyul归一化、核心对齐与裁剪、翻转/噪声增强、逐步模态丢失等。

**📊 数据集**

多中心与单中心脑卒中MRI数据集：ISLES、JHU、CHSF、MATAR、MATAR2、FOCH等，包含DWI、ADC、B0、SWAN、PHASE、SWI等多模态。

**📈 对比分析**

与nnU-Net、CLSTM、LLSTM、AttCLSTM、MultiUnet、ModDrop+等SOTA方法对比；在缺失模态时保持Dice≈0.75–0.80，完整模态时无显著下降；单中心病灶分割Dice最高0.61、检测率>90%；多中心缺失PHASE时Dice≈0.30–0.42但检测率≈80%；总体精度提升10%+，推理时间<3 min。

**⚠️ 局限性**

仍受样本量、模态缺失（如PHASE）导致Dice下降的限制；仅在单中心训练时表现最佳；缺乏对不同设备、扫描协议的广泛泛化验证；后处理需人工阈值调参。

---

## 351. DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale

**arXiv ID:** 2604.00813 | [PDF](https://arxiv.org/pdf/2604.00813v1)

**作者:** Sicheng Zuo `[一作]` (Tsinghua University), Jiwen Lu `[通讯]` (Tsinghua University)

**通讯引用:** 29056 | [OpenAlex ID](https://openalex.org/A5100460385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种实时流式视觉几何变换器 DVGT‑2，能够在线预测密集 3D 点图、车辆位姿和未来轨迹，完成端到端驾驶。

**💡 创新点**

创新点在于滑动窗口流式架构与时间因果注意力，消除全序列计算，实现 O(1) 内存和低延迟，并将密集几何作为驾驶基础，开启 Vision‑Geometry‑Action 范式。

**🔧 技术方法**

采用 ViT‑L 视觉编码器、分解注意力的几何变换器、MRoPE‑I 相对时序编码、三种预测头（点图、位姿、轨迹）以及锚点扩散解码。

**📊 数据集**

在 nuScenes、OpenScene、Waymo、KITTI 和 DDAD 五大多视角驾驶数据上混合预训练，并在 NAVSIM 闭环与 nuScenes 开环基准上评估。

**📈 对比分析**

与 VGGT、DVGT、StreamVGGT 等批处理/全历史流式方法对比，DVGT‑2 在几何重建和轨迹规划上保持或超过 SOTA，推理延迟仅 0.27 s/帧，显著低于对手。

**⚠️ 局限性**

局限性包括相对位姿累积导致的全局位姿漂移、固定窗口缺乏长时上下文，以及对 MoGe‑2 深度监督的依赖。

---

## 352. Routing-Free Mixture-of-Experts

**arXiv ID:** 2604.00801 | [PDF](https://arxiv.org/pdf/2604.00801v1)

**作者:** Yilun Liu `[一作]` (Ludwig Maximilian University Of Munich), Yunpu Ma `[通讯]` (Ludwig Maximilian University Of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Routing-Free MoE架构，让专家通过自身低秩投影自适应激活，无需中央路由器。

**💡 创新点**

创新点在于完全去中心化的专家激活机制、统一的可调节负载平衡框架以及自适应lambda调节。

**🔧 技术方法**

使用了低秩投影的专家内部评分、ReLU门控、可调权重μ、以及基于梯度的自适应正则化。

**📊 数据集**

在OpenWebText上训练，评测9个英文基准（如WinoGrande、QQP等）。

**📈 对比分析**

与标准MoE、AoE、ReMoE等基线比较，Routing-Free MoE在相同计算预算下取得更低困惑度、更高下游准确率且训练更稳定。

**⚠️ 局限性**

局限在于仅在0.8B规模内验证，未评估更大规模或多任务场景，也未探讨硬件加速与量化兼容性。

---

## 353. From Early Encoding to Late Suppression: Interpreting LLMs on Character Counting Tasks

**arXiv ID:** 2604.00778 | [PDF](https://arxiv.org/pdf/2604.00778v1)

**作者:** Ayan Datta `[一作]` (IIIT Hyderabad), Radhika Mamidi `[通讯]` (IIIT Hyderabad)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究使用字符计数任务探究大型语言模型在符号推理中的失败原因，聚焦模型内部机制而非仅评估准确率。

**💡 创新点**

创新点在于揭示模型错误不是因表示缺失，而是由后期层中的负面电路（negative circuit）主动抑制正确答案信号导致的。

**🔧 技术方法**

采用了线性探针、Logit Lens、激活修补（activation patching）等多种可解释性技术，结合概率差异度量来定位错误来源。

**📊 数据集**

构造了基于WordNet/Wiktionary的单词计数数据集，并生成多模板自然语言提示，以保持输入均衡与偏差最小化。

**📈 对比分析**

对LLaMA‑3.2、Qwen‑2.5和Gemma‑2（含基础版与指令调优版）进行比较，发现准确率仅接近随机（约33%），且缩放或指令调优并未改善抑制模式。

**⚠️ 局限性**

研究局限于2‑9B参数规模、单词级英文输入、缺乏大规模模型验证以及未尝试修复干预方案。

---

## 354. ActivityNarrated: An Open-Ended Narrative Paradigm for Wearable Human Activity Understanding

**arXiv ID:** 2604.00767 | [PDF](https://arxiv.org/pdf/2604.00767v1)

**作者:** Lala Shakti Swarup Ray `[一作]` (DFKI), Bo Zhou `[通讯]` (RPTU)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于可穿戴传感器的开放式活动叙事框架（ActivityNarrated 数据集、检索评估、ActNarrator 模型），将人类行为从固定标签分类转为自然语言叙事的可解释理解。

**💡 创新点**

创新点包括：① 用自然语言叙事取代闭集标签；② 设计多位置、多时段同步录制与自述注释的采集方法；③ 通过 Spectral VQ‑VAE 产生离散运动词典；④ 采用检索式评估指标衡量传感器–语言对齐；⑤ 结合 Q‑Former 与冻结 LLM 实现多传感器条件下的开放式文本生成。

**🔧 技术方法**

技术手段：Spectral VQ‑VAE 语义离散化；Q‑Former 交叉注意力融合多传感器序列；冻结大规模 LLM（Qwen、Gemma、LLaMA）进行文本生成；检索评估（Recall@K、MRR、nDCG）与闭集宏 F1 评估；对比实验采用 DeepConvLSTM、TinyHAR 等经典 HAR 模型。

**📊 数据集**

使用了 ActivityNarrated 数据集（22 名受试者、22 小时 IMU+视频+自述+专家标注、15 个传感器位置），并与现有闭集 HAR 数据集（如 WISDM、UCI HAR 等）进行对照实验。

**📈 对比分析**

通过检索评估和闭集 Macro‑F1 对比：在跨人、跨位姿（XS/XSP）下，ActNarrator 的检索 Recall@1 达到 71%/66%，MRR 0.59/0.56，nDCG@5 0.68/0.64；在闭集 23 类上 Macro‑F1 为 65.3%（对比 DeepConvLSTM 35% 及 TinyHAR 34%）。

**⚠️ 局限性**

局限性：样本规模有限，缺少老年人/残障人群；模型依赖 7B 参数 LLM，算力和能耗高；需要进一步压缩以实现边缘部署；评估仍未覆盖多标签、时间重叠的复杂活动。

---

## 355. Translating With Feeling: Centering Translator Perspectives within Translation Technologies

**arXiv ID:** 2604.00758 | [PDF](https://arxiv.org/pdf/2604.00758v1)

**作者:** Daniel Chechelnitsky `[一作]` (Carnegie Mellon University), Maarten Sap `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6253 | [OpenAlex ID](https://openalex.org/A5015128745)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对19名具备11种语言与11个翻译领域专业翻译人员进行半结构化访谈，系统了解其对翻译技术（CAT、MT、LLM等）的使用态度、经验与关注点，揭示翻译工作仍以人类主导、辅助工具为主而非完全自动化；

**💡 创新点**

首次系统性收集并分析专业译者对翻译技术的真实感受与需求，强调人本式协助而非替代，并提出译者自我监管与工具设计应以满足翻译质量与情感表达为核心的创新视角；

**🔧 技术方法**

采用定性研究方法——访谈、开放式编码、轴向编码与主题归纳；

**📊 数据集**

数据来源为19位自我认定为专业译者的访谈记录，涵盖11种语言与11个专业领域；

**📈 对比分析**

本研究未进行算法或性能对比，而是通过主题分析提炼译者关注的主要维度（工具类型、信任度、创意需求等），并用案例引证说明不同工具的优势与局限；

**⚠️ 局限性**

样本规模小且采用便利抽样，语言与领域范围有限，缺乏客观量化指标，且译者自我筛选的专业性与研究结果的普适性存在局限。

---

## 356. Optimal Sampling and Actuation Policies of a Markov Source over a Wireless Channel

**arXiv ID:** 2604.00748 | [PDF](https://arxiv.org/pdf/2604.00748v1)

**作者:** Mehrdad Salimnejad `[一作]` (Linköping University), Nikolaos Pappas `[通讯]` (Linköping University)

**通讯引用:** 3964 | [OpenAlex ID](https://openalex.org/A5084740578)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在无线信道上传输N状态马尔可夫源时的最优采样与执行策略，提出并分析AoII（信息错误年龄）与CoAU（不确定性下的执行成本）指标，并在采样、传输及执行三层上通过随机化、语义化和阈值化策略构建闭式性能表达式，随后求解受约束优化问题以得到最优策略。

**💡 创新点**

①将AoII作为一种同时量化时效性与不准确性的语义指标；②引入CoAU量化执行决策的不确定性导致的错误；③在同一框架下对四类随机化策略（RS、CARS、SARS、TARS）进行统一分析与优化；④证明语义化随机化策略在快速变化源下优于其它策略，阈值化策略在慢速变化源下更佳；⑤提出随机执行策略显著降低错误动作。

**🔧 技术方法**

马尔可夫链建模、AoII与CoAU指标定义、随机化采样/执行策略、离散时间系统分析、闭式解析求解、受约束优化（利用解析式求解最优采样/执行概率）以及数值仿真。

**📊 数据集**

仅使用基于N=3的离散时间马尔可夫链（无真实数据集），通过不同参数（q、p_s、η、μ）进行仿真验证。

**📈 对比分析**

通过与RS、CARS、SARS、TARS四种策略在相同约束下的AoII/CoAU数值对比，发现语义化随机化策略在快速变化源时平均AoII最低，TARS策略在慢速变化源且采样预算受限时表现最好；随机执行策略将错误动作概率显著降低，提升正确动作率至70%+。

**⚠️ 局限性**

①仅考虑离散时间、单源N状态DTMC；②假设ACK无误且无时延；③未考虑多源或多通道交叉干扰；④策略仅为预先设定的随机化形式，缺乏自适应学习机制；⑤仿真仅在N=3时进行，缺乏对大规模状态空间的验证。

---

## 357. How to Train your Tactile Model: Tactile Perception with Multi-fingered Robot Hands

**arXiv ID:** 2604.00744 | [PDF](https://arxiv.org/pdf/2604.00744v1)

**作者:** Christopher J. Ford `[一作]` (University of Bristol), Efi Psomopoulou `[通讯]` (University of Bristol)

**通讯引用:** 294 | [OpenAlex ID](https://openalex.org/A5013732957)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了TacViT，一种基于Vision Transformer的触觉感知模型，用于预测TacTip视觉触觉传感器的接触姿态和力。

**💡 创新点**

创新点在于利用ViT的全局自注意力机制，使模型能够在不需要重新训练的情况下泛化到新传感器，从而显著降低部署成本。

**🔧 技术方法**

技术上使用了预训练的ImageNet ViT（vit-base-patch16-224）并在触觉图像上微调，同时结合LoRA低秩适配器实现高效微调。

**📊 数据集**

使用了从五个指尖TacTip传感器收集的1.5万张触觉图像（每个传感器3000张），并包含对应的接触位姿和力标签。

**📈 对比分析**

通过三组实验（Tr1‑Te1、Tr5‑Te1、Tr4‑TeU）与传统CNN对比，TacViT在新传感器上的平均绝对误差仅为CNN的1/10，标准差也显著降低。

**⚠️ 局限性**

局限性包括ViT对大规模预训练数据的依赖，且在全新传感器上仍有一定误差；未来需要在更大触觉数据集上从零开始训练并针对传感器特定先验进行改进。

---

## 358. Adversarial Attenuation Patch Attack for SAR Object Detection

**arXiv ID:** 2604.00887 | [PDF](https://arxiv.org/pdf/2604.00887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 359. BioCOMPASS: Integrating Biomarkers into Transformer-Based Immunotherapy Response Prediction

**arXiv ID:** 2604.00739 | [PDF](https://arxiv.org/pdf/2604.00739v1)

**作者:** Sayed Hashim `[一作]` (University of York), Paul Cairns `[通讯]` (University of York)

**通讯引用:** 14864 | [OpenAlex ID](https://openalex.org/A5085421841)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在BioCOMPASS中，通过在COMPASS的transformer编码器基础上加入治疗门控、概念对齐、通路一致性以及辅助多任务学习等组件，利用基因表达、外部生物标志物和治疗信息来训练模型，以实现对免疫治疗反应的预测。

**💡 创新点**

创新点在于：①将外部生物标志物与模型中间表示进行对齐，确保概念的生物学可信度；②通过治疗门控将治疗靶点信息融入概念层，使模型对不同治疗方案产生自适应特征；③使用通路一致性损失鼓励模型学习与已知通路活性相关的表征，从而提升跨队列泛化能力。

**🔧 技术方法**

技术主要包括：Transformer-based encoder、概念瓶颈层、基于MSE的通路一致性损失、概念与标志物对齐损失、辅助多任务回归头以及治疗门控网络；模型在预训练阶段使用TCGA数据，微调阶段采用部分微调（冻结编码器）实现高效学习。

**📊 数据集**

使用了8个从CRI iAtlas门户下载的免疫治疗队列（共计约1600例），每个队列已预处理为TPM标准化的基因表达数据，并提供对应的治疗信息与外部生物标志物。

**📈 对比分析**

通过Leave-one-cohort-out、Leave-one-cancer-type-out和Leave-one-treatment-out三种评估策略进行比较，BioCOMPASS在准确率从63%提升到70%，ROC-AUC从71%提升到74%，F1、精确率等指标亦普遍优于原COMPASS，尤其在小样本队列上表现显著改进；在大样本队列中召回略低但整体性能仍优。

**⚠️ 局限性**

主要局限包括：①训练阶段需要外部生物标志物，限制了模型在标志物缺失情况下的适用性；②在大样本队列的召回率略低，可能是模型偏向保守预测；③仅使用了8个可获得的队列，未覆盖全部16个COMPASS原始队列，泛化评估仍有限；④对治疗信息的依赖在推理阶段仍需提供，限制了实时部署。

---

## 360. A column generation algorithm for finding co-3-plexes in chordal graphs

**arXiv ID:** 2604.00721 | [PDF](https://arxiv.org/pdf/2604.00721v1)

**作者:** Alexandre Dupont-Bouillard `[一作]` `[通讯]` (Université de Rennes), Alexandre Dupont-Bouillard (Université de Rennes)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在弦图上提出了求最大 co-3-plex 的多项式时间列生成算法；

**💡 创新点**

创新点在于将 co-3-plex 问题转化为稳定集问题，构造指数尺寸的扩展线性规划，并将定价子问题化为最大权重诱导路径，从而实现多项式求解；

**🔧 技术方法**

使用列生成、扩展线性规划、图论中的诱导路径最大化技术以及稳定集多面体理论；

**📊 数据集**

论文为理论性研究，未使用具体数据集；

**📈 对比分析**

相较于传统 NP-hard 求解方法，本文通过理论证明实现了多项式时间求解，但未进行实验对比；

**⚠️ 局限性**

局限性在于仅适用于弦图，k>3 或一般图时定价子问题未知是否多项式，且扩展空间指数大，难以推广。

---

## 361. Enumerating Two-Orbit Graphs

**arXiv ID:** 2604.00898 | [PDF](https://arxiv.org/pdf/2604.00898v1)

**作者:** David Seka `[一作]` (Technical University of Vienna), Stefan Szeider `[通讯]` (Technical University of Vienna)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统枚举了 27 顶点以内所有连通的两轨图，总计 10,094,721 个。

**💡 创新点**

创新点在于将 Goursat 引理与最小两轨群的剪枝相结合，显著压缩搜索空间，首次完成此规模的完整枚举。

**🔧 技术方法**

主要技术包括置换群理论、Goursat 的子直接积构造、最小性与本质核判定、轨道配置去重、nauty 的标准形化以及 SMS/SAT 约束生成。

**📊 数据集**

使用的“数据集”是通过算法生成的所有 27 顶点以内的图，未引用外部预构造的数据集。

**📈 对比分析**

通过与 nauty（≤11 顶点）和 SMS（≤16 顶点）结果对比，验证计数正确，并且在可达规模上实现了数百万级图的枚举，显著优于传统直接枚举。

**⚠️ 局限性**

主要限制是规模可扩展性不足，随着顶点数增大，图生成和去重成本急剧上升；需要进一步优化搜索树剪枝与并行化策略。

---

## 362. KUET at StanceNakba Shared Task: StanceMoE: Mixture-of-Experts Architecture for Stance Detection

**arXiv ID:** 2604.00878 | [PDF](https://arxiv.org/pdf/2604.00878v1)

**作者:** Abdullah Al Shafi `[一作]`, K. M. Azharul Hasan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于上下文增强的混合专家（Mixture‑of‑Experts）模型StanceMoE，用于演员级立场检测；

**💡 创新点**

创新点在于通过六个互补专家模块分别捕捉全球语义、显著词语、注意力聚焦、短语模式、词汇提示和对比标记等多样化语言信号，并通过上下文感知门控机制动态融合；

**🔧 技术方法**

主要技术包括Fine‑tuned BERT编码器、池化专家（平均、最大）、自注意力池化、Multi‑Kernel CNN、词汇提示与对比意识专家，以及门控融合与加权logit集成；

**📊 数据集**

使用了StanceNakba 2026共享任务提供的英语文本数据集，共1401条样本，划分为训练/验证/测试；

**📈 对比分析**

与传统机器学习、经典深度网络以及BERT、Stacked、Fusion等基线对比，StanceMoE在F1‑Score上达到94.26%，显著高于所有基线（最高为91.83%），并在比赛中获得第三名；

**⚠️ 局限性**

局限性主要体现在模型复杂度高、需六个专家的计算与调参成本，以及对极少样本或多语种场景的适应性尚待验证。

---

## 363. Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation

**arXiv ID:** 2604.00865 | [PDF](https://arxiv.org/pdf/2604.00865v1)

**作者:** Shuguang Jiao `[一作]` (Harbin Institute of Technology), Lina Yao `[通讯]` (UNSW)

**通讯引用:** 16352 | [OpenAlex ID](https://openalex.org/A5052731721)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DR‑RAG 框架，用于对 Agentic Retrieval‑Augmented Generation 失败进行诊断与局部修复。

**💡 创新点**

创新点在于将错误诊断与修复拆分为两阶段，使用覆盖门控错误分类与错误定位，并在定位点上进行工具条件的局部修复，避免全流程重跑。

**🔧 技术方法**

技术包括错误分类与定位模块、覆盖评估、工具条件的修复操作（如答案重写、推理重写、检索查询重写），以及大型语言模型与检索组件的交互。

**📊 数据集**

在 HotpotQA、2WikiMultihopQA 和 MuSiQue 三个多跳问答基准上进行实验。

**📈 对比分析**

与全流程重跑、逐步重试和 RAG‑Critic 的对比表明，DR‑RAG 在 EM、F1、ROUGE‑L 等指标上显著提升，同时 token 及时间成本降低 20‑30%。

**⚠️ 局限性**

局限性在于诊断与定位的准确率受限，尤其在证据不完整时仍有约 30% 的错误识别；且对极长链或高度复杂推理的修复效果仍有限。

---

## 364. Policy Improvement Reinforcement Learning

**arXiv ID:** 2604.00860 | [PDF](https://arxiv.org/pdf/2604.00860v1)

**作者:** Huaiyang Wang `[一作]` (Beihang University), Yikun Ban `[通讯]` (Beihang University)

**通讯引用:** 190 | [OpenAlex ID](https://openalex.org/A5047387636)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PIRL框架和PIPO算法，通过回顾性验证闭环优化RLVR，以策略改进为优化目标。

**💡 创新点**

创新点在于把策略改进反馈作为核心目标，并利用滑动窗口历史基线对群组相对优势产生的梯度失真进行纠正，从而实现闭环自我校正。

**🔧 技术方法**

使用策略改进奖励、重要性采样、滑动窗口平滑、双阶段探索‑验证‑加速/纠正循环等技术。

**📊 数据集**

实验使用数学推理数据集（MATH500、AIME 2025、AMC 2023、MINERVA）以及科学推理数据集SciKnowEval。

**📈 对比分析**

与GRPO、GSPO、DAPO等基线对比，PIPO在所有基准上提升1–7%的准确率，训练更稳定、模式崩溃更少。

**⚠️ 局限性**

局限在于需额外的历史窗口与验证计算开销，对窗口大小和裁剪范围敏感，且在极端稀有奖励场景仍可能出现不确定性。

---

## 365. Sparkle: A Robust and Versatile Representation for Point Cloud based Human Motion Capture

**arXiv ID:** 2604.00857 | [PDF](https://arxiv.org/pdf/2604.00857v1)

**作者:** Yiming Ren `[一作]` (ShanghaiTech University), Yuexin Ma `[通讯]` (ShanghaiTech University)

**通讯引用:** 4191 | [OpenAlex ID](https://openalex.org/A5102015139)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SparkleMotion框架，通过统一骨架关节与表面锚点的点云运动捕捉方法实现对人体运动的高精度建模。

**💡 创新点**

创新点在于引入Sparkle结构化表示，将内部运动学与外部几何分离并联合学习；设计点对齐骨架追踪和骨架引导锚点估计两大模块；采用几何初始化+跨注意力的SMPL求解器，显著提升表达力与鲁棒性。

**🔧 技术方法**

使用了PointNet/Transformer、双向GRU、交叉注意力、swing‑twist旋转分解、SMPL人体模型及端到端训练技术。

**📊 数据集**

使用了11个公开基准数据集，包括LiDAR/深度相机点云（FreeMotion、Sloper4D、NoiseMotion、GTA-Human-Point、HuMMan-Point）、多人近距离交互（InterHuman、Chi3D、Hi4D）以及多视角（FreeMotion-MV、HuMMan-MV）。

**📈 对比分析**

与LiDARCap、LiveHPS、LiveHPS++、VoteHMR、PointHPS、FreeCap等基线比较，SparkleMotion在全局关节/顶点误差、角度误差等指标上均实现了state‑of‑the‑art性能，尤其在噪声、遮挡、跨传感器和多视角场景下表现突出。

**⚠️ 局限性**

局限性包括对极端稀疏或极端噪声点云的鲁棒性仍有限；需要大规模标注点云训练；模型复杂度高，对硬件资源要求较高；在极端遮挡或极端噪声情况下仍可能出现误差。

---

## 366. Perturb-and-Restore: Simulation-driven Structural Augmentation Framework for Imbalance Chromosomal Anomaly Detection

**arXiv ID:** 2604.00854 | [PDF](https://arxiv.org/pdf/2604.00854v1)

**作者:** Yilan Zhang `[一作]` (King Abdullah University of Science and Technology), Aihua Yin `[通讯]` (Guangdong Provincial Maternal and Child Health Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Perturb‑and‑Restore框架，用结构扰动与扩散重建生成高质量合成染色体异常样本，并通过能量引导自适应采样提升检测性能。

**💡 创新点**

创新点在于利用无标签的结构扰动+扩散重建生成合成异常，并通过能量分数动态筛选样本，解决极度不平衡的结构异常检测问题。

**🔧 技术方法**

采用基于片段重排、均值回归SDE扩散模型和能量模型相结合的技术。

**📊 数据集**

使用广东省母婴医院收集的超过26万幅染色体图像的综合异常数据集（含24类染色体，共4,242例异常）。

**📈 对比分析**

与多种长尾学习、能量方法和专门染色体异常检测方法比较，P&R在ResNet18/50上在敏感度、异常精度和F1分数上分别提升约8.9%/7.1%、4.4%/8.9%和13.4%/13.8%，显著超越SOTA。

**⚠️ 局限性**

局限在于合成异常仍受重建模型偏向正常结构的影响，能量采样需调参，且对极少见异常类型的生成和评估仍有限。

---

## 367. MotionGrounder: Grounded Multi-Object Motion Transfer via Diffusion Transformer

**arXiv ID:** 2604.00853 | [PDF](https://arxiv.org/pdf/2604.00853v1)

**作者:** Samuel Teodoro `[一作]` (Korea Advanced Institute of Science and Technology), Munchurl Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5901 | [OpenAlex ID](https://openalex.org/A5027012300)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种无训练的 Diffusion Transformer 框架 MotionGrounder，能够将参考视频的多对象运动迁移到目标视频，并通过文本描述实现对象与区域的显式对齐。

**💡 创新点**

创新点在于：①引入 Flow-based Motion Signal（FMS）以提供更稳定的光流运动先验；②提出 Object-Caption Alignment Loss（OCAL）实现对象-文本的空间对齐；③设计 Object Grounding Score（OGS）作为综合评估多对象运动转移质量的新指标。

**🔧 技术方法**

采用 Diffusion Transformer（DiT）作为核心生成模型，结合 GMFlow 估计光流生成 FMS，利用 3D 注意力引导和训练‑free 优化；评估时使用 CLIP、DINO 等预训练模型。

**📊 数据集**

构建了包含 52 段高运动多对象视频的数据集，来源于 DAVIS、YouTube‑VOS、ConMo 等；使用 CogVLM2‑Caption+GPT‑5 自动生成视频描述，并人工标注对象掩码与对应文本标签。

**📈 对比分析**

与 DMT、MOFT、MotionClone、ConMo、DiTFlow 等零‑shot 方法进行对比，评估指标包括 Motion Fidelity、IoU、Local Textual Alignment、OGS、Global Textual Alignment、CLIP/DINO Temporal Consistency 等。MotionGrounder 在大多数指标上取得最高或第二高成绩，特别是在多对象定位与语义对齐方面明显优于基线。

**⚠️ 局限性**

局限性包括：依赖光流估计，易受噪声影响导致形状失真；对极端场景的泛化不足；需要人工标注对象掩码；对大物体的运动约束可能不够严格。

---

## 368. PanoAir: A Panoramic Visual-Inertial SLAM with Cross-Time Real-World UAV Dataset

**arXiv ID:** 2604.00852 | [PDF](https://arxiv.org/pdf/2604.00852v1)

**作者:** Yiyang Wu `[一作]` (Sun Yat-sen University), Xiangpeng Xu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 157 | [OpenAlex ID](https://openalex.org/A5040897831)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于全景相机和IMU的视觉-惯性SLAM框架，并构建了17条真实UAV飞行序列的全景视觉-惯性数据集；

**💡 创新点**

创新点在于（1）利用全景等距投影实现对全方位视觉信息的特征提取与匹配；（2）结合学习与手工特征的混合特征以及畸变加权，提升特征鲁棒性；（3）设计全景循环闭环模块，显著纠正长程漂移；（4）在嵌入式Jetson Orin NX上实现实时部署；

**🔧 技术方法**

主要技术包括：全景等距投影模型、混合ORB+SuperPoint特征提取、畸变加权、基于Sim(3)的全景循环闭环、优化和全局束调整；

**📊 数据集**

使用了作者自行构建的17条UAV全景视觉-惯性数据集（覆盖云天、晴天、夜晚），以及两个手持序列；

**📈 对比分析**

与ORB-SLAM3、VINS-Mono、DROID-SLAM、OpenVSLAM、360-VIO、360-DVO等方法在UAV和公开数据集上对比，本文方法在所有序列上取得100%成功率、最低ATE，且在嵌入式平台上保持约10Hz实时率；

**⚠️ 局限性**

局限性包括：框架尚未充分利用GPU并行加速；未显式建模遮挡与动态场景，导致在极端遮挡或高速动态情况下仍可能出现误估。

---

## 369. Steering through Time: Blending Longitudinal Data with Simulation to Rethink Human-Autonomous Vehicle Interaction

**arXiv ID:** 2604.00832 | [PDF](https://arxiv.org/pdf/2604.00832v1)

**作者:** Yasaman Hakiminejad `[一作]` (Villanova University), Arash Tavakoli `[通讯]` (Villanova University)

**通讯引用:** 383 | [OpenAlex ID](https://openalex.org/A5066478442)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并验证了一种混合框架，将7天的移动感知和每日体验采样与高保真驾驶模拟结合，用以研究半自动驾驶车辆中的司机接管准备状态。

**💡 创新点**

创新点在于：①将长期基线生理/心理数据与实时模拟响应相融合，捕捉个体差异；②通过多模态测量（可穿戴、fNIRS、眼动、驾驶模拟）构建更具生态效度的研究设计。

**🔧 技术方法**

使用技术包括：Empatica EmbracePlus 可穿戴腕带（EDA、PPG、加速度、温度）、Ethica 体验采样应用、Tobii Pro Glasses 2 眼动追踪、Biopac fNIRS Imager 2000 皮层激活监测、驾驶模拟器与多摄像头录像；数据预处理与特征提取采用SCI、波形滤波、HbO/HbR 转换、I‑DT 眼动分割、TOC 计算等。

**📊 数据集**

使用数据集为 38 名受试者，包含 7 天的可穿戴生理记录、每日 4 次体验问卷（共约 28 份）、以及一次实验室模拟会话（6 条况：2 种接管场景 × 3 次次任务）。

**📈 对比分析**

方法比较主要基于非参数 Wilcoxon 符号秩检验与 Holm 校正，混合效应模型评估 ICC；结果显示框架可行，个体差异显著，但并未给出预测性能指标，仅证明了多模态数据整合与个体化监测的可行性。

**⚠️ 局限性**

局限性包括：受试者样本规模小、样本多为大学生；移动感知数据缺失率高、实验室模拟未随机化场景顺序；fNIRS 仅展示单个受试者数据；脚本化接管事件缺乏生态真实性；缺乏对实时预测模型的评估。

---

## 370. Learning to Learn-at-Test-Time: Language Agents with Learnable Adaptation Policies

**arXiv ID:** 2604.00830 | [PDF](https://arxiv.org/pdf/2604.00830v1)

**作者:** Zhanzhi Lou `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 5759 | [OpenAlex ID](https://openalex.org/A5065675832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Meta-TTL 框架，利用进化元学习在任务分布上学习可迁移的适应策略，使语言代理能够在测试时通过多轮交互自适应提升表现。

**💡 创新点**

核心创新在于把跨剧烈循环的适应机制本身视为可学习的目标，在提示空间中进行双层优化并用进化搜索寻找最优 meta‑prompt，从而实现无需手工规则的自适应改进。

**🔧 技术方法**

技术包括：双层优化（inner TTL 循环 + outer 元训练循环）、进化搜索（种群、变异、交叉）、权重冻结的 LLM（Gemini 3 Flash、GLM‑5、GPT‑5）以及自然语言 meta‑prompt 的生成与评估。

**📊 数据集**

实验数据集为：Jericho（ID：Detective、Zork 1、Temple；OOD：Balances、Library、Zork 3）和 WebArena‑Lite（ID：Shopping、GitLab、Map；OOD：Reddit、Shopping Admin）。

**📈 对比分析**

与静态、Reflexion、Memory Agent 以及未学习的 Naïve baseline 进行对比，采用平均分数和 Weighted‑AUC（W‑AUC）指标。实验显示在 ID 场景下，Meta‑TTL 在平均游戏分数上提升约 120%（Jericho）或 15%（WebArena‑Lite）成功率，W‑AUC 明显提高；在 OOD 场景亦保持正向提升，证明学习到的策略具有迁移性。

**⚠️ 局限性**

局限性：对奖励粒度敏感（Jericho 的细粒度奖励更易优化，WebArena‑Lite 的二元奖励导致搜索空间粗糙）；进化搜索计算成本较高；仅在提示层面改进，无法处理需要权重微调的任务；效果受限于所用基础 LLM 的能力；对结构完全不同的 OOD 任务提升仍有限。

---

## 371. Continual Vision-Language Learning for Remote Sensing: Benchmarking and Analysis

**arXiv ID:** 2604.00820 | [PDF](https://arxiv.org/pdf/2604.00820v1)

**作者:** Xingxing Weng `[一作]` (Wuhan University), Gui-Song Xia `[通讯]` (Wuhan University)

**通讯引用:** 22023 | [OpenAlex ID](https://openalex.org/A5073032922)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了面向遥感视觉‑语言模型的持续学习基准CLeaRS，系统评估模型在多任务、多模态与多应用场景下的学习表现；

**💡 创新点**

首次将持续学习与遥感视觉‑语言模型结合，定义了三种评估协议（长时序、模态增量、任务增量），并验证了常规持续学习方法在遥感场景下的不足；

**🔧 技术方法**

使用现有遥感VLM（如Qwen2.5‑VL、MiniGPT‑v2、LLaVA‑1.5、GeoChat、VHM）和三种参数高效持续学习技术（MoELoRA、HiDe‑LLaVA、SEFE）进行实验；

**📊 数据集**

收集并自动生成10个子集，共207,753图文对，涵盖光学场景分类、图像字幕、视觉定位、视觉问答、SAR/红外定位与问答，以及火灾风险与灾害损毁评估；

**📈 对比分析**

实验显示所有模型在三种协议下均表现出负向后向迁移（BWT<0），且即便采用持续学习方法，显著减轻遗忘的效果有限；在长时序和任务增量设置下，多数方法与普通顺序微调相差无显著提升；

**⚠️ 局限性**

局限性包括：基准仅覆盖有限任务与模态，未覆盖所有遥感数据特征；持续学习方法未针对遥感VLM进行专门设计，导致效果不佳；评估仅基于当前模型规模，缺少更大规模或多语言场景验证。

---

## 372. Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment

**arXiv ID:** 2604.00913 | [PDF](https://arxiv.org/pdf/2604.00913v1)

**作者:** Zhuchenyang Liu `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**通讯引用:** 3949 | [OpenAlex ID](https://openalex.org/A5069437467)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了IKEA-Bench，一个评估二维手册与视频配合的跨表现图像-视频对齐基准。

**💡 创新点**

首次量化描述-视频对齐难点，揭示文本辅助能提升图像理解但会分散视觉注意力，提出三层机制分析。

**🔧 技术方法**

使用Vision‑Language Models（VLMs），三种对齐策略（视觉、视觉+文本、文本）以及中心核对齐（CKA）、线性探针、注意力分析。

**📊 数据集**

采用IKEA Manuals at Work数据集，涵盖29件家具的图纸、视频与时间标注，共1623道题。

**📈 对比分析**

对19个VLM（含两款专有模型）进行零样本评测，平均T1准确率约45%，T4约33%，表明视频理解是瓶颈。

**⚠️ 局限性**

仅覆盖IKEA家具，缺少更广泛领域与更高级指导功能，专有模型覆盖有限，且实验仅评估基础对齐能力。

---

## 373. Shape Representation using Gaussian Process mixture models

**arXiv ID:** 2604.00862 | [PDF](https://arxiv.org/pdf/2604.00862v1)

**作者:** Panagiotis Sapoutzoglou `[一作]` (National Technical University of Athens), Maria Pateraki `[通讯]` (National Technical University of Athens)

**通讯引用:** 1067 | [OpenAlex ID](https://openalex.org/A5047139078)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出基于高斯过程混合模型的对象特定函数形状表示，从稀疏点云学习连续方向距离场来重建3D形状。

**💡 创新点**

仅使用高斯过程取代深度网络，形成轻量化、可解释的形状混合模型，聚焦单个物体细节而非类别泛化，并通过参考点划分实现全表面覆盖。

**🔧 技术方法**

使用高斯过程回归、方向距离场（DDF）、参考点选择（骨架化/距离聚类）、混合模型加权软最大、Rational Quadratic核、GPyTorch 训练与评估。

**📊 数据集**

在 ShapeNetCore（Planes、Chairs、Sofas）和 IndustryShapes（Screwdriver）数据集上进行实验。

**📈 对比分析**

与 DeepSDF、NKSR 等方法在 Chamfer Distance、Precision、Recall、F-score 上对比，获得最低 CD、>90% F-score，尤其在薄壁、复杂拓扑上表现优于竞争者。

**⚠️ 局限性**

需要人工或智能化的参考点选取，数量与位置对质量影响大；对极其复杂拓扑仍需改进覆盖保证。

---

## 374. Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation

**arXiv ID:** 2604.00849 | [PDF](https://arxiv.org/pdf/2604.00849v1)

**作者:** Shuang Li `[一作]` (Tencent), Jie Jiang `[通讯]` (Tencent)

**通讯引用:** 1571 | [OpenAlex ID](https://openalex.org/A5101944041)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DisCo 框架，先将主体身份与文本指令解耦，然后通过强化学习重新耦合，以实现高保真主体保留与文本控制兼顾。

**💡 创新点**

创新点在于将文本提示拆分为仅包含修改指令的简化提示，并用视觉定位确定主体；同时设计专属奖励模型和 GRPO 强化学习来重新融合视觉与文本信息。

**🔧 技术方法**

采用 FLUX Diffusion、DiT、3D RoPE、GroundingDINO、Qwen2.5-VL、Qwen3-VL 作为奖励模型，并结合 GRPO 对生成过程进行优化。

**📊 数据集**

使用 Subjects200K 作为训练集，DreamBench 作为评测基准。

**📈 对比分析**

与 SDXL 与 FLUX 相关的多种基线对比，DisCo 在 CLIP-B‑I/CLIP-L‑I/DINO‑I 以及文本对齐和图像质量（ImageReward）指标均显著优于所有对手，获得最优性能。

**⚠️ 局限性**

缺点包括需要额外的 RL 训练和专用奖励模型，计算成本较高，且对极端复杂场景的自适应仍有待提升。

---

## 375. Proactive Agent Research Environment: Simulating Active Users to Evaluate Proactive Assistants

**arXiv ID:** 2604.00842 | [PDF](https://arxiv.org/pdf/2604.00842v1)

**作者:** Deepak Nathani `[一作]` (University of California, Santa Barbara), Xin Eric Wang `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 10123 | [OpenAlex ID](https://openalex.org/A5100327844)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于状态机的用户模拟环境（Proactive Agent Research Environment）和一个观察‑执行的主动代理架构，并在此基础上发布了143个多应用、多场景的评测基准。

**💡 创新点**

创新点：①将用户交互建模为有限状态机（FSM），实现与真实手机界面相似的导航约束；②采用非对称接口，让代理可直接调用API而用户受限于当前界面；③提出观察‑执行两阶段代理，既保证了用户自主权，又能高效完成任务。

**🔧 技术方法**

技术：基于Agent Research Environment（ARE）的扩展；LLM驱动的用户模拟器和主动代理（采用ReAct框架）；Stackelberg POMDP建模；自动化场景生成器（LLM驱动）。

**📊 数据集**

数据集：自动生成的143个多应用场景（涵盖通信、生产力、日程安排、生活方式）；用户行为模型使用GPT‑5‑mini；测试包含工具失效、噪声事件等随机扰动。

**📈 对比分析**

比较方法：在10回合、4次重复实验中对七种LLM（Claude 4.5 Sonnet、GPT‑5、Gemini 3 Pro/Flash、Qwen 3 4B、Llama 3.2 3B、Gemma 3 4B）进行评测；指标包括成功率、接受率、提议率、信息采集动作数。最佳模型Claude 4.5 Sonnet与Gemini 3 Flash成功率≈42%，但整体模型成功率仍低于50%，小模型一致性差。

**⚠️ 局限性**

局限性：①仅在模拟环境中评估，缺乏真实用户实验；②用户模拟器依赖LLM，可能无法完全复制人类行为；③对工具失败和噪声的鲁棒性仍有限；④框架对不同设备/平台的适配尚未充分验证。

---

## 376. Event Embedding of Protein Networks : Compositional Learning of Biological Function

**arXiv ID:** 2604.00911 | [PDF](https://arxiv.org/pdf/2604.00911v1)

**作者:** Antonin Sulc `[一作]` `[通讯]` (Lawrence Berkeley National Lab), Antonin Sulc (Lawrence Berkeley National Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在蛋白质互作网络上使用Event2Vec强制加性序列嵌入，并与DeepWalk基线对比，评估路径连贯性、向量算术、网络层级与嵌入漂移等指标。

**💡 创新点**

提出在蛋白互作序列上采用可逆加性递推约束，使嵌入具备可加性，从而显著提升关系推理与功能类比的表现。

**🔧 技术方法**

使用Event2Vec（可逆加性递推嵌入）、DeepWalk（Skip‑gram随机游走嵌入）、欧氏与双曲几何加性运算以及随机游走采样。

**📊 数据集**

使用STRING v12.0人类互作网络（16,201蛋白、89,234高置信度边），生成10条长度15的随机游走序列共约160K条。

**📈 对比分析**

与DeepWalk在相同数据和超参数下对照，依据路径连贯度、功能类比相似度、层级聚类、嵌入漂移等六项指标评估；Event2Vec在路径连贯度提升30.2×、类比相似度提升至0.966、层级聚类更清晰，整体表现显著优于DeepWalk。

**⚠️ 局限性**

仅能捕捉可加性关系，无法建模上下文依赖或乘法交互；评估样本有限，跨细胞区间类比失效，仍需在更大数据库中验证。

---

## 377. JAMMEval: A Refined Collection of Japanese Benchmarks for Reliable VLM Evaluation

**arXiv ID:** 2604.00909 | [PDF](https://arxiv.org/pdf/2604.00909v1)

**作者:** Issa Sugiura `[一作]` (Kyoto University), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3546 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 JAMMEval，一个通过两轮人工审核和重标注精炼的日语视觉语言问答（VQA）基准集合。

**💡 创新点**

创新点在于以保留样本规模为前提，对七大日语 VQA 数据集进行系统化重标注，而非简单剔除噪声样本，从而提升评测质量与可靠性。

**🔧 技术方法**

采用人工双轮审核与重标注、统一短答式答案格式、LLM 驱动的软 Exact‑Match 判分器（GPT‑5.1）、多模型评测（Qwen3‑VL、InternVL3.5、Sarashina2.2 及 GPT‑4o/5.1、Gemini 3 Pro）等技术。

**📊 数据集**

使用七个日语 VQA 种子数据集：CC‑OCR‑JA、JA‑VLM‑Bench、Heron‑Bench、CVQA‑JA、JA‑Multi‑Image‑VQA、JDocQA 与 JGraphQA，最终得到 1,592 条精炼实例。

**📈 对比分析**

通过对同一模型在原始与精炼数据集上进行多次评测（采用准确率、跑次标准差、模型间性能差距和 Spearman 相关性比较），Gemini 3 Pro 在所有任务上均达到 90%+ 准确率，Open‑Weight 版本 Qwen3‑VL‑8B 在多任务上表现最佳。

**⚠️ 局限性**

局限性包括：日语文化知识仍不足导致 CVQA‑JA 误差高；判分器基于 LLM 的软判分可能产生误判；人工重标耗时且难以大规模扩展；精炼后数据集规模仍有限，影响模型细粒度比较。

---

## 378. Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts

**arXiv ID:** 2604.00901 | [PDF](https://arxiv.org/pdf/2604.00901v1)

**作者:** Sha Li `[一作]` (Virginia Tech), Naren Ramakrishnan `[通讯]` (Virginia Tech)

**通讯引用:** 10935 | [OpenAlex ID](https://openalex.org/A5035052603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种层级演化的多代理检索增强生成（RAG）框架，通过经验库和角色感知提示演化同时优化全局协同策略与个体代理行为，实现对复杂多跳查询的无梯度适应。

**💡 创新点**

创新点在于：①将全局协调与代理提示的演化放在同一层级框架中；②构建可检索的经验库（Profile–Insight–Utility）作为上下文驱动的策略改进；③引入角色感知提示演化（RoPE）进行局部信用分配与提示调整；④使用无梯度的组相对策略优化（GRPO）在拓扑层面进行采样与奖励比较；⑤实现结构演化与行为演化的协同进化。

**🔧 技术方法**

技术包括：层级EM解释、无参数的提示演化、组相对策略优化、经验库检索与整合、工具和多代理交互、检索增强生成、token效率分析、拓扑演化指标等。

**📊 数据集**

实验数据集涵盖多跳问答与事实验证基准：2WikiMultiHopQA、HotpotQA、MusiQue、AmbigQA、Bamboogle、HoVer，检索语料库为Wikipedia。

**📈 对比分析**

与Direct inference、CoT、单回合RAG、IterDRAG、Plan-RAG、Search-o1、Search-R1、IRCoT、SELF-RAG、CORAG、InstructRAG、R1-Searcher、DeepResearcher、MMOA-RAG、AceSearcher、MAO-ARAG、ExSearch等基线相比，平均提升38.69%的EM/F1/准确率，同时保持token消耗和泛化性能。

**⚠️ 局限性**

局限性包括：依赖冻结的LLM参数，对极长文本或大规模知识库的扩展受限；经验库的规模与检索效率之间需权衡；提示演化局部性导致跨代理错误仍可能累积；未对计算资源、延迟及多语言/跨域场景进行深入评估。

---

## 379. IDDM: Identity-Decoupled Personalized Diffusion Models with a Tunable Privacy-Utility Trade-off

**arXiv ID:** 2604.00903 | [PDF](https://arxiv.org/pdf/2604.00903v1)

**作者:** Linyan Dai `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8908 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向授权个性化的模型侧输出免疫方法IDDM，能够在保持图像质量的前提下降低生成头像与真实身份的可链接性

**💡 创新点**

在个性化训练过程中引入双阶段身份解耦数据优化，并通过可调节参数ρ实现可控的隐私‑实用性权衡，首次在面部个性化扩散模型中实现模型侧输出免疫

**🔧 技术方法**

采用Stable Diffusion预训练模型与DreamBooth/LoRA细调框架，结合投影梯度下降（PGD）对训练图像进行有限扰动，使用多种面部识别器（IRSE50、IR152、MobileFaceNet、FaceNet）构建身份原型，并在优化时采用两个阶段（保持图像一致性与降低身份相似度）

**📊 数据集**

在CelebA‑HQ和VGGFace2两个公开人脸数据集上进行实验，每个身份选取多张照片分为参考集、保护集与评估集

**📈 对比分析**

与现有反个性化防御（Anti‑DreamBooth、SimAC、Anti‑Diffusion）以及无防御基线进行对比，IDDM在保持面部检测成功率、FID、SER‑FIQ、BRISQUE等视觉质量指标不变甚至提升的同时，显著降低IS­M、ADA和top‑k检索成功率，表明在隐私保护和生成质量之间实现更优平衡

**⚠️ 局限性**

限制在于需要在训练阶段进行额外的身份解耦优化，且对不同扩散模型的适应性需进一步验证；在极端高隐私配置下仍可能出现一定的视觉质量衰退，并且对攻击者使用不同或更强的识别器时保护效果尚未完全保证

---

## 380. Super-Resolving Coarse-Resolution Weather Forecasts With Flow Matching

**arXiv ID:** 2604.00897 | [PDF](https://arxiv.org/pdf/2604.00897v1)

**作者:** Aymeric Delefosse `[一作]` (Inria), Dominique Béréziat `[通讯]` (Sorbonne Université)

**通讯引用:** 341 | [OpenAlex ID](https://openalex.org/A5053426980)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种模块化的学习式超分辨率方法，将低分辨率机器学习气象预报与高分辨率细节重建分离，利用残差形式的生成模型在预报后处理阶段恢复细尺度变化。

**💡 创新点**

创新点在于将超分辨率视为随机逆问题，采用流匹配训练的三维 Swin U‑Net 生成器实现条件随机重构，并通过残差分解保证大尺度结构不变，首次实现了在全球尺度、0.25° 级别下的高效零射击细尺度预报。

**🔧 技术方法**

使用的技术包括：三维 Swin U‑Net Transformer、流匹配（flow matching）训练、残差重构（大尺度插值+残差预测）、保守重格化、以及基于概率分布的随机采样。

**📊 数据集**

使用的主要数据集为 ERA5 重新格点数据（1979‑2018 训练，2019 验证，2020 测试），将 0.25° 分辨率降格到 1.5° 进行训练对照，并在真实预报上零射击应用。

**📈 对比分析**

通过重格化一致性指标（相关系数、活跃比、归一化 RMSE）和标准集群验证（CRPS、Ensemble Mean RMSE、Energy Score、Brier Score 等）与 IFS ENS、GenCast、bicubic 基线对比，结果显示超分辨率在保持大尺度结构的同时显著提升了小尺度能量和概率预报技术性，0.25° 级别的技术性与高成本端到端高分辨率模型相当。

**⚠️ 局限性**

主要局限在于：推理时的计算开销仍显著（单 V100 32GB GPU 10 天高分辨率预报约 8 分钟）；方法针对 24 h 预测轨迹设计，缺乏时间上细化的动态学习；零射击假设对大型误差修正有限，导致随机细节不一定提升整体预报技能；对高分辨率训练数据的依赖可能限制在极端天气或复杂地形的适用性。

---

## 381. PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding

**arXiv ID:** 2604.00886 | [PDF](https://arxiv.org/pdf/2604.00886v1)

**作者:** Nan Wang `[一作]` (OPPO AI Center), Haonan Lu `[通讯]` (OPPO AI Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种 PixelPrune 方法，在 Vision‑Language 模型的 ViT 编码前利用像素层预测编码进行裁剪，去除文档与 GUI 图像中大量的冗余视觉补丁，减少视觉 token 数量。

**💡 创新点**

创新点在于：① 采用二维预测编码（Pred‑2D）在像素空间直接识别可恢复的重复补丁；② 无需训练、无可学习参数、支持精确和可控损失的裁剪；③ 通过在 ViT 前裁剪，能够一次性加速整条推理/训练流水线，实现显著的速度与内存提升。

**🔧 技术方法**

使用技术包括：像素级预测编码、阈值 τ 控制匹配精度、Pred‑2D 预测策略、O(N) 的 GPU 并行实现；与 Qwen3‑VL 等 VLM 集成，兼容 FlashAttention、DeepSpeed ZeRO‑2 等训练加速框架。

**📊 数据集**

实验数据集：文档理解 7 个基准（DocVQA、AI2D、ChartQA、InfoVQA、OCRBench、MMLongBench‑Doc、olmOCRBench）；GUI 理解 9 个基准（ScreenSpot V2 的 Web/Mobile/Desktop，ScreenSpot Pro 的 Scientific/Office/OS/Creative/Development/CAD）。

**📈 对比分析**

方法与全量 token、Resize、Random、ConnComp、Resize baseline 等进行对比。结果显示：文档任务中 PixelPrune 与 Full 差距≤0.2%，保留率 22–71%；GUI 任务中训练后+KD 可逼近 Full，平均保留率 39–61%。推理加速 3–4.2×，训练加速 1.9×，内存减 33.6%。

**⚠️ 局限性**

局限性：对细微纹理或高频信息（如 Chart、CAD）阈值敏感，可能需 fine‑tuning 或 KD 以恢复性能；无学习参数限制了针对特定任务的进一步优化；在非结构化自然图像上的泛化效果尚待验证。

---

## 382. Detecting Call Graph Unsoundness without Ground Truth

**arXiv ID:** 2604.00885 | [PDF](https://arxiv.org/pdf/2604.00885v1)

**作者:** Fangtian Zhong `[一作]`, Joseph Windmann `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过基于部分顺序的形变测试方法，检测并量化了四个主流 Java 静态分析框架（Soot、SootUp、WALA、Doop）在算法、配置及框架语义层面上的语义不一致。

**💡 创新点**

创新点在于将算法与配置视为可组合的部分顺序，利用无基准的元方法自动发现语义违例，并揭示配置与算法交互放大错误的现象。

**🔧 技术方法**

采用部分顺序（精度/可证性）定义、元方法测试、调用图归一化和 Jaccard 相似度比较技术。

**📊 数据集**

使用 JCG 基准套件（125 个 Java 程序）以及其覆盖的现代语言特性。

**📈 对比分析**

通过对同一程序在不同配置、算法、框架间生成的调用图进行归一化后 Jaccard 计算，发现跨框架相似度仅 10‑20%，且在精度提高时错误数不一定下降。

**⚠️ 局限性**

局限在于仍无法完全消除框架间语义差异、对大规模真实应用的适用性不足，以及高精度分析在资源上不可行。

---

## 383. A 4D Representation for Training-Free Agentic Reasoning from Monocular Laparoscopic Video

**arXiv ID:** 2604.00867 | [PDF](https://arxiv.org/pdf/2604.00867v1)

**作者:** Maximilian Fehrentz `[一作]` (TU Munich), Nassir Navab `[通讯]` (TU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文在单目腹腔镜视频中构建了时空一致的4D语义跟踪表示，供多模大语言模型（MLLM）无训练地进行时空推理。

**💡 创新点**

创新点在于将语义分割、深度估计与跟踪结果融合成可调用的4D工具，实现训练无关的4D grounding，并通过工具调用让MLLM直接操作时空信息。

**🔧 技术方法**

技术栈包括 Depth Anything 3（深度与相机估计）、Cotracker 3（二维跟踪）、SASVi（语义分割）以及Qwen3-VL 作为无训练的推理引擎，利用点云插值和实例合并构建4D表示。

**📊 数据集**

实验使用Cholec80子集的25段3-4秒短视频，外科医生标注134个临床相关的空间、时间、方向查询。

**📈 对比分析**

与单模2D MLLM基线比较，4D方法在空间查询误差降低约50%，方向查询提升79%，时间点查询同样优于基线；在时间区间查询上表现略逊，但整体显著优于传统2D基线。

**⚠️ 局限性**

主要限制包括：依赖离线深度/跟踪模型，重建精度受单目视角与光照影响；跟踪误差与光照抖动仍会影响结果；未在实时手术环境中验证，尚未证明可在线高帧率部署。

---

## 384. Yet Even Less Is Even Better For Agentic, Reasoning, and Coding LLMs

**arXiv ID:** 2604.00824 | [PDF](https://arxiv.org/pdf/2604.00824v1)

**作者:** Yang Ye `[一作]`, Yuchi Ma `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Less-Is-More 训练框架和 STITCH 机制，利用更少但高质量的轨迹数据提升软件工程代理的能力

**💡 创新点**

将数学推理中的“少即多”假设迁移到编码与代理任务，并引入分层轨迹筛选与滑动记忆的 Map‑Reduce 细粒度提炼方法

**🔧 技术方法**

SandForge 任务构造、STITCH 轨迹筛选（宏观逻辑回归+微观 LLM 语义分析）、滑动记忆、Map‑Reduce 处理、强化学习/监督微调、自动评估

**📊 数据集**

基于真实 GitHub 代码修复记录构造的数据集（Python、Java、ArkTS），共计约 752 / 391 / 924 条高质量轨迹，配合 SWE‑bench‑Verified、Multi‑SWE‑bench、HarmonyOS（ArkTS）评测

**📈 对比分析**

在同类模型（30B‑355B）和多语言环境下与基线对比，STITCH 训练的模型在 SWE‑bench‑Verified 上提升至 63.16% 相对改进；在 Multi‑SWE‑bench（Java）上 MiniMax‑M2.5‑STITCH + CodeArts Agent 提升至 43.75%（+16.67%）；在 HarmonyOS（ArkTS）上 GLM‑4.7‑STITCH 编译通过率提升至 61.31%（+43.34%）

**⚠️ 局限性**

对高质量轨迹的依赖较强，仍需更广泛的多语言、跨平台验证；模型对极端长轨迹的记忆与评估仍有挑战；在无网络、无历史访问的严格评测环境下可能略逊于部分基线

---

## 385. ProCap: Projection-Aware Captioning for Spatial Augmented Reality

**arXiv ID:** 2604.00912 | [PDF](https://arxiv.org/pdf/2604.00912v1)

**作者:** Zimo Cao `[一作]` (Southwest University), Bingyao Huang `[通讯]` (Southwest University)

**通讯引用:** 364 | [OpenAlex ID](https://openalex.org/A5081934804)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出ProCap框架，在空间增强现实中实现物理场景与投影内容的双重字幕生成。

**💡 创新点**

创新点在于自动投影分割、区域感知检索以及双重字幕评估协议，解决虚拟-物理模糊与投影失真问题。

**🔧 技术方法**

主要技术包括冻结CLIP ViT视觉编码器、Q-Former查询、Mask Pooling、外部知识库检索与LLM解码。

**📊 数据集**

使用RGBP大规模SAR语义基准，包含65个物理场景、180k+投影，提供二值投影掩码和分离字幕。

**📈 对比分析**

与FastVLM、Qwen3-VL等基线对比，ProCap在场景与投影字幕上显著提升CIDEr、SPICE等指标，尤其在投影字幕上提升近十倍。

**⚠️ 局限性**

局限性包括对投影分割精度依赖、知识库词表限制以及仅覆盖平面/轻曲面，未覆盖复杂非刚性或动态表面。

---

## 386. A Framework for Parameterized Subexponential-Subcubic-Time Algorithms for Weighted Problems in Planar Graphs

**arXiv ID:** 2604.00891 | [PDF](https://arxiv.org/pdf/2604.00891v1)

**作者:** Matthias Bentert `[一作]`, Petr A. Golovach `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套通用框架，用于在平面图上设计子指数参数化算法，能够处理带权、定向、并且解集可能包含任意多连通分支的问题。

**💡 创新点**

创新点在于：① 通过对Nederlof的分离器技术进行精细化改造，得到能兼顾权重、定向、非连通解的子分离器；② 将分离器与树宽分布式覆盖相结合，得到新的递归结构；③ 通过“请求变体”与“抛光器”概念，使得多种复杂问题可统一归入框架中；④ 取得了此前未知的子指数时间算法（如带权版本的最大权重诱导森林、最大权重独立集、带权覆盖等）。

**🔧 技术方法**

核心技术包括：平面图的Baker分层、3/4平衡环分离器、树宽与树分解（尤其是二叉、对数深度分解）、Nederlof的分离器重构与概率覆盖、动态规划结合请求表、以及基于分离器的“测量”与“几乎测量”判定。

**📊 数据集**

该工作为理论算法，未使用具体实验数据集；所有结果均基于图论证明与复杂度分析。

**📈 对比分析**

与传统的bidimensionality与树宽模式覆盖方法相比，本文的框架突破了权重、定向、非连通解的限制，获得了在平面图上多类问题的子指数时间（2^O(√k) n^2.49）。实验对比部分主要是算法复杂度与已知最优下界（ETH）对照，证明该框架在理论上是最优或接近最优的。

**⚠️ 局限性**

局限性包括：① 框架实现复杂，常数项极大，实际可行性尚待验证；② 仅适用于平面图（或更广泛的稀疏图族需进一步扩展）；③ 对某些问题的“请求变体”定义与 R4 的实现仍需手工设计；④ 在高参数规模下的内存与时间开销可能仍较大。

---

## 387. AuraDesk: Data Physicalization through Olfaction Metaphors for Representing and Mitigating Workplace Stress

**arXiv ID:** 2604.00869 | [PDF](https://arxiv.org/pdf/2604.00869v1)

**作者:** Siying Hu `[一作]` (University of Queensland), Zhenhao Zhang `[通讯]` (University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并现场部署了一款名为 AuraDesk 的嗅觉物理化系统，用可穿戴心率变异性传感器实时感知工作场所的生理压力，并通过八通道雾化装置在办公桌附近释放气味来提示用户的压力状态。

**💡 创新点**

创新点在于：① 将连续的生理指标映射为可感知的气味元件，采用场景隐喻（森林、海风、花园等）而非直接情感标签；② 结合本地 AI（PicoLM）与规则约束的混合映射，平衡实时性与气味扩散特性；③ 设计了低干扰、空间限定的气味输出，适用于共享办公环境；④ 通过一日现场实验探究了气味反馈在工作场所中的认知、情感与调节效果。

**🔧 技术方法**

技术细节包括：华为 Watch 5 智能手表采集 HRV 与心率；NVIDIA Jetson Nano 作为本地边缘推理节点，运行量化版 PicoLM 进行生理状态推断；八通道 USB 雾化控制板结合红外遥控实现气味的即时、单通道释放；气味库由八种单音符精油构成，按场景隐喻分类；系统采用基于规则的调度层控制气味强度、频率与持续时间。

**📊 数据集**

数据来源为实验中 25 名工作者在各自办公桌上采集的实时 HRV 与心率数据，并以个体基线进行归一化；日志记录气味释放时序、强度与类型；此外收集了用户的问卷评分与访谈记录，但未使用公开的标准生理数据集。

**📈 对比分析**

评估方法采用混合方法：量化问卷（七点量表）评估感知度、舒适度、非侵入性等维度；定性访谈挖掘用户体验与对比。结果显示用户对气味的感知度、舒适度和非侵入性均给出高分，认为气味更为细腻与隐蔽；但实验未设置可视化或音频对照组，且缺乏客观效能指标（如生产力、压力水平变化），因此性能评估主要基于主观反馈。

**⚠️ 局限性**

局限性包括：样本量有限（25 人）；仅进行单日现场部署，缺乏长期习惯适应与持续效果评估；未进行实验对照，无法量化与传统视觉/听觉压力反馈的差异；气味偏好与社交适应性因人而异，尚未系统研究；整体依赖主观评价，缺乏客观生理或行为指标。

---

## 388. Reliability of Large Language Models for Design Synthesis: An Empirical Study of Variance, Prompt Sensitivity, and Method Scaffolding

**arXiv ID:** 2604.00851 | [PDF](https://arxiv.org/pdf/2604.00851v1)

**作者:** Rabia Iftikhar `[一作]` (Technische Universitat Clausthal), Andreas Rausch `[通讯]` (Technische Universitat Clausthal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三大主流LLM（ChatGPT 4o-mini、Claude 3.5 Sonnet、Gemini 2.5 Flash）在两类领域（医院计费系统与传感器网络）上，系统评估其生成 UML 类图的设计合成能力、设计原则遵循、模式出现与行为稳定性，共完成 540 次实验。

**💡 创新点**

提出基于偏好的一阶少样本提示法，利用对比设计实例引导模型倾向符合面向对象原则与设计模式的结构，并对比标准、规则注入与该偏好提示三种策略，探究模型稳定性与可靠性。

**🔧 技术方法**

使用多模态提示工程（标准、规则注入、偏好对比）、人工评估、结构正确性、原则遵循、模式出现、稳定性指标四种度量，结合十次重复与三种提示变体，对模型输出进行全面评估。

**📊 数据集**

数据集由两份领域设计意图基准组成：医院计费系统（中等复杂度）与传感器网络（高复杂度），每份包含三种不同措辞的提示和对应专家参考模型；此外构建了少量对比设计偏好样本。

**📈 对比分析**

相较于标准和规则注入提示，偏好提示在原则遵循（PAS）与模式出现（PES）上显著提升，但稳定性指数（SI）仍未达到理想；模型间差异显著：Claude 具高度解码稳定，ChatGPT 在给定架构后可控，Gemini 结果波动大；总体表现显示偏好提示可提升质量但无法完全消除随机性。

**⚠️ 局限性**

研究局限：仅覆盖两类领域，缺乏更广泛的设计任务；偏好样本规模有限，未使用完整 RLHF；评估依赖人工标注，可能带主观偏差；仅考察 UML 类图生成，未涉及多种建模成果。

---

## 389. Influence of the geometry on the mechanical performance of tubular interlockings: A study of the Sine Block

**arXiv ID:** 2604.00836 | [PDF](https://arxiv.org/pdf/2604.00836v1)

**作者:** Domen Macek `[一作]` (RWTH Aachen University), Alice C. Niemeyer `[通讯]` (RWTH Aachen University)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5050998707)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并分析了一种新的管状拓扑互锁块——Sine Block，并通过多体动力学（MBD）与有限元（FEM）方法，系统评估其在不同边界值问题（内部压力、外部压力、轴向压缩、弯曲、扭转）下的机械性能。

**💡 创新点**

创新点包括：①基于正弦曲线设计的块体实现了对径向爆炸（kinematic explosion）的自阻断；②提出了参数化的块体设计空间，并利用高效的MBD框架进行全局探索；③将MBD与高精度FEM结合，验证了MBD在预测接触压力分布与力平衡方面的可靠性；④通过与传统六边形块进行对比，证明Sine Block在有效接触面积、平均接触压力和最大平均接触压力等指标上具有更优性能。

**🔧 技术方法**

主要技术手段包括：利用无摩擦单向接触的非线性互补问题求解，构建静态平衡与运动学可行性判定；采用最小二范数量化的二次规划求解接触力；在Abaqus/EXPLICIT中实现软接触的有限元仿真；以及对接触面积、平均接触压力、最大平均接触压力等定量指标的计算。

**📊 数据集**

采用的“数据集”为基于设计参数（幅度 a、平移 s、频率 f、块数 n、壁厚 t、内半径 r_i、层数 L 等）构造的合成配置空间；在五种典型边界条件下对多组参数组合进行仿真，形成实验数据集。

**📈 对比分析**

比较方法：通过MBD与FEM在管道、隧道、轴向等三种情境下的接触压力分布进行定性对比，并统计有效接触面积、平均接触压力及最大平均接触压力；性能结果显示：MBD 的计算成本比 FEM 低约五个数量级（约 0.0056 小时/配置 vs 758 小时/配置），且在预测接触压力分布与力平衡方面与 FEM 结果高度一致；Sine Block 在所有 BVP 下的有效接触面积更大、平均接触压力更低、最大平均接触压力更优。

**⚠️ 局限性**

局限性：①仅考虑无摩擦接触，未包含摩擦对互锁性能的影响；②模型假设块体为刚体或线弹性，未考虑大变形或塑性行为；③有限元采用软接触模型和小间隙，可能低估实际接触压力；④缺乏实验验证，实际制造和施工工艺对互锁结构的影响尚未探究。

---

## 390. Near-Optimal Four-Cycle Counting in Graph Streams

**arXiv ID:** 2604.00828 | [PDF](https://arxiv.org/pdf/2604.00828v1)

**作者:** Sebastian Lüderssen `[一作]` (TU Wien), Pan Peng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 11770 | [OpenAlex ID](https://openalex.org/A5091685833)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在任意顺序图流中对四环（4-cycle）进行检测和计数的多遍历算法，能够在三次遍历内以 (1±ε) 的近似误差计数四环，所需空间为 O(m/√T)，其中 m 为边数，T 为四环数量。

**💡 创新点**

核心创新在于完全基于节点采样而非传统的边采样，结合“配置”与“标签化子结构”的概念，并引入细化的重心度阈值与随机移位，能够精确控制方差并匹配已知的空间下界，从而突破以往仅能达到 O(m/T^{1/3}) 的空间上限。

**🔧 技术方法**

技术手段包括多级节点采样、κ 值离散化、配置与标签化子结构、重心度阈值（heavy/light 判定）以及精细化的重心度 t(·,κ,ℓ)；配合或acles 实现轻重判断，利用方差分析与概率放大实现高概率成功。

**📊 数据集**

本文为理论研究，不涉及实验数据集，主要通过构造硬实例和严谨的概率与方差分析来验证算法的正确性与空间下界匹配。

**📈 对比分析**

与以往的 3‑pass O(m/T^{1/3}) 算法以及单遍 O(m^4/T^2) 算法相比，本算法在多遍历场景下实现了与下界 Ω(m/√T) 的匹配；在理论上证明了高概率 (1±ε) 近似计数，并保持线性扫描时间。

**⚠️ 局限性**

主要限制在于：需要多遍历且假设已知 T 的近似下界；实现相对复杂，需维护多组采样概率、随机移位及 oracles；对极端结构（如重边）仍需额外处理；单遍算法在空间上仍无法匹配下界，且实际部署时对随机性与硬件支持有一定要求。

---

## 391. Fatigue-Aware Learning to Defer via Constrained Optimisation

**arXiv ID:** 2604.00904 | [PDF](https://arxiv.org/pdf/2604.00904v1)

**作者:** Zheng Zhang `[一作]` (University of Surrey), Gustavo Carneiro `[通讯]` (University of Surrey)

**通讯引用:** 15011 | [OpenAlex ID](https://openalex.org/A5029215323)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FALCON 框架，将学习推延（L2D）问题建模为约束马尔可夫决策过程（CMDP），通过心理学疲劳曲线动态模拟人类专家随工作负荷累积而产生的性能衰退，从而实现 AI 与人类的自适应协作。

**💡 创新点**

创新点包括：① 将可预测的心理学疲劳曲线嵌入 L2D；② 采用 PPO‑Lagrangian 在 CMDP 上进行约束优化，实现覆盖率预算的精确控制；③ 设计 FA‑L2D 基准，系统性覆盖从近静态到快速疲劳的多种人类性能情景；④ 在未见专家的零射设置下仍能保持鲁棒性。

**🔧 技术方法**

主要技术手段：构造状态包含任务特征与累积工作负荷；使用 Resettable S5 序列模型捕捉长期依赖；利用 PPO‑Lagrangian 强化学习求解带上下限约束的 CMDP；通过噪声率映射将疲劳曲线转化为人类预测误差；训练和评估基于模拟的心理学疲劳环境。

**📊 数据集**

实验数据集：Cifar100（改版）、Chaoyang、MiceBone、Flickr10K；每个数据集均构造了多种疲劳曲线参数区间以生成 FA‑L2D 基准。

**📈 对比分析**

与 OneStage L2D、TwoStage L2D、L2D‑Pop、EA‑L2D 等 SOTA 方法在 Accuracy‑Coverage 曲线和 AUACC 指标上进行对比。FALCON 在所有覆盖率区间、所有数据集上均优于竞争者，尤其在高覆盖率和快速疲劳场景；在零射（zero‑shot）设置下仍保持显著优势。

**⚠️ 局限性**

局限性：① 仅使用基于心理学模型的人工模拟人类疲劳，缺乏真实人类实验验证；② 假设疲劳对所有任务相同，未考虑任务难度异质性和个体差异；③ 仅考虑累积工作负荷，未考虑瞬时注意力或多模态疲劳指标；④ 需进一步探索在真实部署中的可行性与安全性。

---

## 392. When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation

**arXiv ID:** 2604.00892 | [PDF](https://arxiv.org/pdf/2604.00892v1)

**作者:** Henry Peng Zou `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 135315 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了长时程、环境受限的 Web 导航任务中，LLM 代理在执行过程被用户打断（新增、修改或撤销请求）时的适应与恢复能力，提出了 InterruptBench benchmark；

**💡 创新点**

创新点在于：①首次构建可插入中断的长时程任务基准；②定义三类真实中断场景（添加、修订、撤销）；③采用轨迹根据信息注入模拟中断并量化后续适应；

**🔧 技术方法**

技术手段包括：基于 WebAgent-R1 框架的 LLM 代理，统一的中断注入与模拟框架，以及针对后续成功曲线、token 与动作成本的分析；

**📊 数据集**

使用数据集为从 WebArena-Lite 衍生的 165 个经人工验证的任务，经过 LLM 合成生成初始查询和中断消息；

**📈 对比分析**

比较方法：在单/多中断场景下，测量成功率（SR）及后续成功曲线 SR(k)，并对比无中断基线，六种 LLM 后端（Claude 系列、Qwen3、DeepSeek、Mistral）的表现，结果显示即便是强大模型在处理中断时仍存在显著性能瓶颈；

**⚠️ 局限性**

限制在于：中断后代理往往未能有效同步 UI 状态或重构计划，导致任务失败或恢复效率低；基准主要聚焦 Web 导航，难以直接推广至其他类型的长时程任务。

---

## 393. Beyond Symbolic Solving: Multi Chain-of-Thought Voting for Geometric Reasoning in Large Language Models

**arXiv ID:** 2604.00890 | [PDF](https://arxiv.org/pdf/2604.00890v1)

**作者:** Md. Abu Bakor Siddique `[一作]` (Islamic University of Technology), Md Kamrul Hasan `[通讯]` (Islamic University of Technology)

**通讯引用:** 3344 | [OpenAlex ID](https://openalex.org/A5100656463)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MARS-GPS，一种推理时框架，生成多条并行推理链（CoT），在链中嵌入Python代码执行进行数值验证，并通过多阶段投票与自检融合最终答案；

**💡 创新点**

创新点包括：① 并行采样多条推理路径，① 通过每个token的熵来无成本估计置信度，② 结合熵加权投票与模型自检的多阶段聚合策略，③ 在推理链中加入Python沙箱进行精确计算，显著提升几何推理可靠性；

**🔧 技术方法**

使用技术包括：冻结的大语言模型GPT-OSS 120B、图像解析器PGDPNet、规则式文本解析器、token级熵估计、Python沙箱执行、熵加权投票与自检聚合算法；

**📊 数据集**

使用数据集为Geometry3K（3002题）和PGPS9K（9022题），涵盖高中几何多类型题目；

**📈 对比分析**

通过与多种基线（神经、神经-符号、全模态LLM、专有LLM）在Geometry3K和PGPS9K上的比较，MARS-GPS分别达到88.8%（比之前SOTA提升约11%）和77.48%，在所有基线中均名列前茅；

**⚠️ 局限性**

局限性包括：依赖解析阶段的准确性，解析错误会直接导致失败；计算成本随并行采样数k线性增长；仅适用于多选几何题目，未验证开放式或证明任务；

---

## 394. Accurate and Scalable Matrix Mechanisms via Divide and Conquer

**arXiv ID:** 2604.00868 | [PDF](https://arxiv.org/pdf/2604.00868v1)

**作者:** Guanlin He `[一作]` (Penn State University), Daniel Kifer `[通讯]` (Penn State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于分治策略的可扩展矩阵机制，用于从数据边缘统计（marginals）高效地回答任意线性查询工作负载。

**💡 创新点**

创新点在于将查询分解为正交子工作负载，使每个子问题可以独立使用低维矩阵机制求解，并通过重新组合实现全局最优（在平方误差下优于 RP、RP+、WFF）。

**🔧 技术方法**

采用了线性查询分解、残差空间正交投影、最优或近似低维矩阵机制、以及噪声重新缩放的组合技术。

**📊 数据集**

实验使用了 Adult、CPS、Loans 等真实数据集以及各种 n^d 形式的合成数据集进行评测。

**📈 对比分析**

与 RP+、WFF、HDMM 等基线方法比较，实验显示该方法在所有工作负载下均获得最低 RMSE，尤其在非轴对齐查询（affine、abs）上提升 15–21%。

**⚠️ 局限性**

局限在于子工作负载的最优求解仍需半正定规划，计算开销随子工作负载规模增长，且对极高维或极大域的情况仍面临内存瓶颈。

---

## 395. Evaluating the Feasibility of Augmented Reality to Support Communication Access for Deaf Students in Experiential Higher Education Contexts

**arXiv ID:** 2604.00856 | [PDF](https://arxiv.org/pdf/2604.00856v1)

**作者:** Roshan Mathew `[一作]` (Rochester Institute of Technology), Roshan L. Peiris `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1676 | [OpenAlex ID](https://openalex.org/A5000376873)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了增强现实（AR）智能眼镜在实验室环境中为聋哑学生提供实时沟通访问的可行性，并与现场、远程字幕等传统方式进行对比。

**💡 创新点**

创新点在于将实时字幕或手语翻译直接叠加到AR眼镜的视野中，验证空间连续性原则在安全关键实验环境下的有效性，并探讨头锁定与世界锁定显示模式的设计冲突。

**🔧 技术方法**

使用的技术包括Vuzix Blade 2光学透明AR眼镜、WebRTC+Socket.IO后端通信、TypeWell实时字幕服务，以及自研的ARRAE平台。

**📊 数据集**

数据来源为12名DHH学生在模拟实验室完成的人工制雪任务，收集眼动追踪、问卷评分和访谈文本，未使用公开数据集。

**📈 对比分析**

采用within‑subject 3个条件实验，使用VAS量表和Friedman检验对满意度、视觉分散等指标进行比较。结果显示AR在视觉分散和未来使用偏好上显著优于远程字幕，且在手动任务中显著降低视觉分散。

**⚠️ 局限性**

局限性包括样本量仅12人、单一教师/解说员、实验室仿真环境非真实课程、Vuzix硬件与助听器冲突、头锁定显示导致的视觉疲劳等，限制了结论的普适性和对长期使用的评估。

---

## 396. Agentic Tool Use in Large Language Models

**arXiv ID:** 2604.00835 | [PDF](https://arxiv.org/pdf/2604.00835v1)

**作者:** Jinchao Hu `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 61178 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大规模语言模型的代理工具使用进行了系统综述，提出了从提示到监督再到强化学习的演化视角，并为三种方法范式构建了统一的分类框架。

**💡 创新点**

创新点在于：①提供了从提示式插件到监督式学习再到奖励驱动策略学习的完整进化路线；②提出了统一的三范式（提示、监督、奖励）分类，并明确了各范式的优化信号与方法边界；③将评测体系与方法范式交叉映射，形成了一张完整的工具使用生态图。

**🔧 技术方法**

采用文献梳理与归纳技术，对近两年关于 LLM 代理工具使用的论文进行分层、对比与总结，构建了基于方法论、工具类型、训练设置的多维度视角。

**📊 数据集**

综述涵盖了多种公开基准数据集，例如函数调用正确性评测（BFCL、ToolEyes、T-Eval）、任务完成评测（MATH、GSM8K、BigCodeBench、SciBench 等）以及交互式评测（WebArena、OSWorld、ToolSword、SafetyBench 等）。

**📈 对比分析**

通过对比分析，作者展示了三类方法在不同评测层级（调用正确性、任务完成度、交互式成功率）下的表现差异与局限，并指出强化学习范式在长周期信用分配与策略优化方面表现更佳，但整体缺乏统一的性能基准。

**⚠️ 局限性**

局限性包括：①综述仍停留在方法与评测的宏观层面，缺乏统一的实验对比与量化指标；②跨范式的迁移与融合机制尚未系统化；③在安全、对齐与高阶推理等关键挑战上仍存在显著空白，亟需进一步的研究与标准化。

---

## 397. LinguDistill: Recovering Linguistic Ability in Vision- Language Models via Selective Cross-Modal Distillation

**arXiv ID:** 2604.00829 | [PDF](https://arxiv.org/pdf/2604.00829v1)

**作者:** Patrick Amadeus Irawan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yova Kementchedjhieva `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 829 | [OpenAlex ID](https://openalex.org/A5048127182)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无适配器的跨模态知识蒸馏框架，用冻结的语言模型（LM）作为教师，借助 KV‑cache 层级共享，让 VLM 在训练中恢复被多模态微调损失的语言能力。

**💡 创新点**

创新点在于：①通过层级 KV‑cache 共享让冻结 LM 能够访问 VLM 的多模态上下文，避免改动模型架构；②采用数据源敏感的选择性蒸馏，只在语言密集的数据上启用蒸馏信号，保护视觉任务不受干扰，从而兼顾语言与视觉性能。

**🔧 技术方法**

使用了跨模态知识蒸馏（KL 损失 + CE 损失）、KV‑cache 共享机制、选择性蒸馏权重调度（按数据源分配 α）、标准微调与温度参数等技术。

**📊 数据集**

基准数据集为 Cauldron 里的 50 个 VL 指令调优数据，挑选 17 个核心来源（含 VQA、OCR、知识、领域任务）进行完整训练，另外 8 个语言密集子集用于 ablation；所有样本均包含一张图片，最大序列长度 1024。

**📈 对比分析**

与标准微调、仅使用语言数据微调以及统一蒸馏四种基线进行对比。选择性蒸馏在语言与知识基准（如 ScienceQA、AI2D、COCO caption）上恢复约 10% 绩效，同时视觉任务（如 DocVQA、OCRBench）仅出现 3–7% 的轻微下降，显著优于统一蒸馏导致的 15–40% 视觉性能损失。

**⚠️ 局限性**

局限性包括：①训练仅使用单图像和 1024 token 的输入限制，导致对多图像或长文本任务的适应性受限；②对 OCR/文档类细粒度视觉任务仍存在一定的性能下降；③选择性蒸馏依赖于数据源标签，若缺失或不准确会影响效果。

---

## 398. Video Patch Pruning: Efficient Video Instance Segmentation via Early Token Reduction

**arXiv ID:** 2604.00827 | [PDF](https://arxiv.org/pdf/2604.00827v1)

**作者:** Patrick Glandorf `[一作]` (Leibniz University Hannover), Bodo Rosenhahn `[通讯]` (Leibniz University Hannover)

**通讯引用:** 10070 | [OpenAlex ID](https://openalex.org/A5040412734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了视频Patch Pruning (VPP) 框架，在视频实例分割任务中实现早期图像块稀疏化，显著降低计算量。

**💡 创新点**

创新点在于引入可微分的时间映射模块（Map‑SM），利用前一帧的高层特征指导第一层的稀疏化，并通过 Gumbel‑Softmax 实现前景选择与稀疏背景激活。

**🔧 技术方法**

技术包括 Vision Transformer、Patch Pruning、Gumbel‑Softmax、前景选择损失、动态稀疏化（SM）以及与 Mask2Former 与 ROVIS 的集成。

**📊 数据集**

在 Youtube‑VIS 2019 与 2021 数据集上进行评估。

**📈 对比分析**

与 SViT、Dynamic‑ViT、SVIT、TPS 等现有方法比较，VPP 在保持 55% PKR 时仅损失 0.6% AP，且在 40% PKR 时 AP 仅下降 1.9%，同时 FLOPs 下降约 32% 并提升 FPS，显示显著的效率提升。

**⚠️ 局限性**

局限性包括对视频前景稳定性和运动估计的依赖，极端场景切换时仍可能出现暂时的前景误判；以及在极低 PKR（≤30%）时性能下降更明显。

---

## 399. Optimal Brain Decomposition for Accurate LLM Low-Rank Approximation

**arXiv ID:** 2604.00821 | [PDF](https://arxiv.org/pdf/2604.00821v1)

**作者:** Yuhang Li `[一作]` (Yale University), Priyadarshini Panda `[通讯]` (Yale University)

**通讯引用:** 6136 | [OpenAlex ID](https://openalex.org/A5050310538)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于全局任务损失的低秩分解方法OBD-LLM，利用二阶Hessian信息通过Kronecker分解实现权重矩阵的双向白化并给出闭式最优解；

**💡 创新点**

创新点在于①将未来层梯度信息（输出梯度协方差）与传统的输入激活协方差共同考虑，构造双向白化；②利用K-FAC近似Hessian，避免计算大规模矩阵；③将分解问题化为在白化空间的SVD，得到全局损失最小的低秩近似；

**🔧 技术方法**

使用的核心技术包括：二阶Taylor展开的损失近似、Hessian的Kronecker-factorized (K-FAC) 近似、Cholesky分解实现白化、SVD求最优低秩解；

**📊 数据集**

实验数据集主要包括Wikitext-2、C4以及多任务零样本下的LongBench、PiQA、ARC、HellaSwag、Winogrande、BoolQ、OBQA、SiQA；

**📈 对比分析**

与传统SVD、FWSVD、ASVD、SVD-LLM等方法对比，在20%–40%压缩率下，OBD-LLM在Wikitext-2的困惑度下降约10–40%，在多任务零样本下平均精度提升约1–3%；同时在KV缓存压缩、量化/剪枝补偿等场景亦表现优于对照方法；

**⚠️ 局限性**

局限性包括：①依赖K-FAC假设，若输入激活与梯度高度相关时近似失效；②仅针对线性层的低秩分解，对Transformer整体结构的其他层（如FFN、Attention）尚需进一步验证；③在极高压缩率或极大模型规模下，SVD求解仍是瓶颈；

---

## 400. Emotion Entanglement and Bayesian Inference for Multi-Dimensional Emotion Understanding

**arXiv ID:** 2604.00819 | [PDF](https://arxiv.org/pdf/2604.00819v1)

**作者:** Hemanth Kotaprolu `[一作]` (Indian Institute of Technology Bombay), Pushpak Bhattacharyya `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 13314 | [OpenAlex ID](https://openalex.org/A5065100828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了情感场景数据集EmoScene，包含4731个情境，每个情境通过Plutchik理论的8维情感向量进行标注；

**💡 创新点**

创新点在于将情感建模为结构化的多标签任务，并引入基于情感共现统计的贝叶斯后处理框架，以捕捉情感间的互联关系；

**🔧 技术方法**

使用指令调优的大型语言模型进行零样本预测，并在此基础上采用Ising模型构建的情感共现先验进行联合推理；

**📊 数据集**

使用自建的EmoScene文本情境数据集，数据来源为ECoK知识图谱和GPT-4o生成的情境，随后由人工专家标注；

**📈 对比分析**

通过与六个开源指令模型的零样本比较，最优模型Macro F1为0.501；贝叶斯后处理显著提升弱模型表现（如Qwen2.5-7B从0.428提升至0.479，宏观F1提升5%），整体减小了汉明损失；

**⚠️ 局限性**

局限性包括情境由语言模型生成，缺乏自然多样性；数据仅为文本，不包含多模态信号；贝叶斯框架仅建模二阶共现，未覆盖更高阶情感交互。

---

## 401. Misconception Acquisition Dynamics in Large Language Models

**arXiv ID:** 2604.00818 | [PDF](https://arxiv.org/pdf/2604.00818v1)

**作者:** Naiming Liu `[一作]` (Rice University), Shashank Sonkar `[通讯]` (University of Central Florida)

**通讯引用:** 132 | [OpenAlex ID](https://openalex.org/A5028809416)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过构建MalAlgoLib生成带有步骤级错误轨迹的线性方程练习，并在三种LLM上训练Novice Student和Expert Tutor误解模型，系统评估它们在学习误解时的动态表现。

**💡 创新点**

创新点在于首次量化误解习得的学习动力学，揭示学生模型需混合正确样本才能避免错误泛化，导师模型可并行学习多误解而不牺牲整体准确率，并证明步骤级监督是学习误解的关键。

**🔧 技术方法**

采用LLM指令微调技术，利用MalAlgoLib生成的图结构问题与误解规则进行step‑by‑step训练，结合Llama‑8B、Phi‑4‑4B和Qwen‑4B等模型实现误解学习。

**📊 数据集**

使用MalAlgoLib生成的16类线性方程问题与20种已归类误解，每类2000条正确示例与可变数量的误解示例，构成实验数据集。

**📈 对比分析**

实验结果显示，学生模型在学习误解时准确率下降，但加入25%正确样本即可恢复；导师模型在多误解训练下保持甚至提升整体正确率；缺乏步骤监督时两模型误解准确率均低于30%。

**⚠️ 局限性**

局限包括仅针对线性代数、需要大量步骤级数据、误解频率不均导致样本稀缺，以及真实教育场景中难以获得完整解题轨迹。

---

## 402. Flow-based Policy With Distributional Reinforcement Learning in Trajectory Optimization

**arXiv ID:** 2604.00977 | [PDF](https://arxiv.org/pdf/2604.00977v1)

**作者:** Ruijie Hao `[一作]` (National University of Defense Technology), Guangquan Cheng `[通讯]` (National University of Defense Technology)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5101851407)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了一种结合流匹配（Flow Matching）策略与分布式强化学习（Distributional RL）的FP-DRL框架，用于解决多模态动作分布的连续控制问题。

**💡 创新点**

创新点在于仅用流匹配网络直接建模策略并在RL目标下训练，同时引入分位数分布回报的评论器为多模态策略提供更丰富的学习信号，并通过Transformer实现高效、低步采样的实时控制。

**🔧 技术方法**

使用了流匹配生成模型、Transformer架构、分布式分位数回归、Soft Actor-Critic框架、Huber量化损失、Wasserstein距离等技术，并在JAX与PyTorch中实现。

**📊 数据集**

在MuJoCo连续控制基准上进行评测，包括Humanoid-v4、Ant-v4、HalfCheetah-v4、Hopper-v4、Reacher-v4和InvertedPendulum-v4。

**📈 对比分析**

与TD3、SAC、DSAC-T、SAC-Flow、DACER等六大基线在相同网络、同等训练步骤下对比，采用训练后10%期奖励的平均值评估，FP-DRL在所有任务上均达到或超过SOTA，部分任务提升约20%至80%。

**⚠️ 局限性**

主要局限在于对高维动作空间仍需较多采样，对Transformer长度、分位数数目、温度等超参数敏感；此外，离散或大规模状态空间的适用性尚未验证。

---

## 403. An Integrated Soft Robotic System for Measuring Vital Signs in Search and Rescue Environments

**arXiv ID:** 2604.00971 | [PDF](https://arxiv.org/pdf/2604.00971v1)

**作者:** Jorge Francisco García-Samartín `[一作]` (Universidad Politécnica de Madrid), Antonio Barrientos `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 5321 | [OpenAlex ID](https://openalex.org/A5033322846)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一套集成软体机器人系统，用于在搜救环境中自动测量受害者的心率和血压。

**💡 创新点**

创新点包括：①可穿戴式气动软抓手可自适应包裹受害者手臂并安全施压；②轻量化可携带的气动供给系统；③基于自适应峰值滤波的心率与血压估计算法，显著降低误差并实现毫秒级响应。

**🔧 技术方法**

主要技术：单位四足机器人（Unitree Aliengo）+ 6-DoF 机械臂；软体气动抓手（TPU 3D 打印）；光学光电容积脉搏传感器；差压传感器；Jetson Nano + ADS1115 ADC；气泵+电磁阀控制；信号处理（低通 Butterworth、FIR 滤波、峰值自适应筛选）和血压估算算法。

**📊 数据集**

数据集：20 名志愿者（年龄 21–58 岁）在实验室环境下采集心率/血压；另外在模拟灾后场景中对 5 名真人受害者与 2 名假体进行 18 分钟的现场测试。

**📈 对比分析**

与市售 Omron M2+ 电子血压计进行比较，采用 MAE、RMSE、MedAE、% 错误、Bland‑Altman 偏差、标准差和限差等指标；结果显示心率 MAE ≈9.6 bpm、血压 MAE 约 12–10 mmHg，偏差约 -4~ -6 mmHg，误差在 10% 以内，满足救援场景所需的快速估计要求。

**⚠️ 局限性**

限制：样本量小（20 名受试者）且未覆盖极端血压范围；气动软抓手在极端姿态下可能导致低压或误检；实验仅在“黄色”难度的低光灾后场景，未验证在更复杂地形或高温、烟雾浓度更高的条件下的性能；系统存在轻微的偏低估计，需进一步校准与算法优化。

---

## 404. The Varieties of Ought-Implies-Can and Deontic STIT Logic

**arXiv ID:** 2604.00967 | [PDF](https://arxiv.org/pdf/2604.00967v1)

**作者:** Kees van Berkel `[一作]` (Technische Universit"at Wien), Tim S. Lyon `[通讯]` (Technische Universit"at Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了多重义务-可行性（Ought-Implies-Can）解释，并在Deontic STIT逻辑中构建了模块化框架。

**💡 创新点**

提出十种OIC读法的系统分类，给出相应的完备推理系统并分析它们的逻辑关系。

**🔧 技术方法**

采用模态逻辑、标签化序贯演算与形式语义技术。

**📊 数据集**

无实验数据集，全部为形式化证明。

**📈 对比分析**

通过逻辑可证性与完备性比较，各OIC原则的相互承认关系被系统性阐述。

**⚠️ 局限性**

局限在于仅关注传统Deontic STIT框架，未探讨其他多模态或实证验证。

---

## 405. A Visionary Look at Vibe Researching

**arXiv ID:** 2604.00945 | [PDF](https://arxiv.org/pdf/2604.00945v1)

**作者:** Yebo Feng `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 85720 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并系统阐述了“vibe researching”——一种让研究者通过自然语言指令交由LLM代理执行文献综述、实验实现、数据分析与写作等劳动密集型任务，同时保持研究者对创意、评估与质量把控的参与；

**💡 创新点**

创新点在于将软件工程中的vibe coding理念迁移至科研流程，定义了vibe researching的原则与方法学，并对其技术瓶颈、社会影响及未来研究方向进行了全面归纳；

**🔧 技术方法**

采用多代理架构、持久化记忆机制、工具调用与技能库、规划与任务拆解、检索增强生成、以及自我反思与验证等一系列LLM代理技术；

**📊 数据集**

本文未针对单一公开数据集开展实验，而是利用现有文献检索（如Semantic Scholar、arXiv）、代码仓库（GitHub）和实验平台（HPC/云）等多种来源作为案例演示；

**📈 对比分析**

由于论文主要是方法论与综述性质，未给出实验性能指标；作者通过对比传统人工流程、工具辅助、AI for Science以及全自动研究等对照框架，说明vibe researching在效率、可扩展性与人机协作层面具有潜在优势；

**⚠️ 局限性**

主要局限包括模型幻觉、上下文窗口限制、基础设施不兼容、对多模态与实体实验的弱适配、验证与审查不对称、对新颖任务的脆弱性以及数据隐私与知识产权问题。

---

## 406. Auditing the Reliability of Multimodal Generative Search

**arXiv ID:** 2604.00944 | [PDF](https://arxiv.org/pdf/2604.00944v1)

**作者:** Erfan Samieyan Sahneh `[一作]` (IT University of Copenhagen), Luca Maria Aiello `[通讯]` (IT University of Copenhagen)

**通讯引用:** 5000 | [OpenAlex ID](https://openalex.org/A5034406723)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 Gemini 2.5 Pro 多模态搜索系统生成的 11,943 条视频引用主张进行大规模审核。

**💡 创新点**

首次系统性评估视频基础生成搜索的可信度，并揭示未得到视频支持的主张比例及其主要错误类型。

**🔧 技术方法**

使用多模态检索、自动语音转录、三名独立 LLM 法官评估以及逻辑回归分析等技术。

**📊 数据集**

基于医学、经济、通用三类公开查询数据集（Comprehensive Medical Q&A、FinGPT、Google Natural Questions）收集查询并生成主张。

**📈 对比分析**

通过与三名 LLM 法官和人工验证对比，支持率分别在 81.3%–96.3% 之间，严格法官识别错误率高达 18.7%。

**⚠️ 局限性**

局限包括仅基于文本证据验证、缺乏对检索与生成错误的区分、三名法官可能存在偏差、视觉证据仅占少量。

---

## 407. EmoScene: A Dual-space Dataset for Controllable Affective Image Generation

**arXiv ID:** 2604.00933 | [PDF](https://arxiv.org/pdf/2604.00933v1)

**作者:** Li He `[一作]`, Lizhe Qi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了一个大规模的双空间情绪数据集EmoScene，并在其上实现了轻量级的情绪感知调制基线，能够在不改变生成模型主体的前提下通过情绪标签、VAD维度和感知属性实现对图像情绪色调的可控生成。

**💡 创新点**

创新点在于首次将情绪维度与感知属性（色彩、纹理、曲率等）联合编码为双空间表示，并利用该表示直接驱动扩散模型的交叉注意力，提供可解释的情绪与视觉属性控制。

**🔧 技术方法**

采用了PixArt‑α冻结扩散生成器，设计了Affective‑Perceptual Modulation Network（APMN）将情绪与感知输入映射为调制向量，并通过轻量化注入器在跨层交叉注意力中调整键值，从而实现对情绪与颜色的微调。

**📊 数据集**

使用了自建的EmoScene数据集，包含约120万张图片、345类场景、8种离散情绪标签、连续VAD值、颜色与结构感知属性以及语义描述。

**📈 对比分析**

实验通过与冻结PixArt‑α基线在CLIPScore、VAD误差和HSV色差等指标对比，结果显示调制后模型在情绪对齐和颜色控制上显著优于基线，虽然文本语义一致性略有下降。

**⚠️ 局限性**

局限性主要体现在对感知属性的调制仍相对粗糙（如仅通过HSV颜色有限控制）、跨情绪维度的细粒度转换受限，以及对生成质量与文本一致性之间的权衡需要进一步平衡。

---

## 408. PsychAgent: An Experience-Driven Lifelong Learning Agent for Self-Evolving Psychological Counselor

**arXiv ID:** 2604.00931 | [PDF](https://arxiv.org/pdf/2604.00931v1)

**作者:** Yutao Yang `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**通讯引用:** 8140 | [OpenAlex ID](https://openalex.org/A5062604912)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出经验驱动的终身学习心理咨询代理，集成记忆增强规划、技能演进与强化内部化三大模块实现多会话自适应提升。

**💡 创新点**

通过闭环经验→技能提炼→参数内部化的循环机制，让模型在每轮咨询后主动更新策略，解决传统静态LLM缺乏长期记忆与技能演化的问题。

**🔧 技术方法**

采用记忆增强规划引擎（动态记忆构建+战略规划）、分层技能树与检索、生成式技能演进算法、拒绝式微调（Rejection Fine‑Tuning）强化内部化，以及奖励模型评估和LLM-judge自动评测。

**📊 数据集**

使用公开的多会话心理咨询数据集（约2000+客户档案，涵盖多种治疗流派），在训练/验证集上进行监督微调和多轮推理。

**📈 对比分析**

与 GPT‑5.4、Gemini‑3、Qwen3‑Max 等通用LLM 以及 PsyLLM、TheraMind 等心理学专用模型进行对比，采用多维度自动/人工评估，PsychAgent 在所有共享与专属指标上均优于基线，特别在会话规划、干预效果与客户情绪轨迹上表现突出。

**⚠️ 局限性**

仍依赖标注对话数据，未在真实临床环境中验证安全与隐私，奖励模型与内部化策略可能受参数设置影响，对跨文化或复杂心理疾病的适用性尚待进一步检验。

---

## 409. Positional Cognitive Specialization: Where Do LLMs Learn To Comprehend and Speak Your Language?

**arXiv ID:** 2604.00923 | [PDF](https://arxiv.org/pdf/2604.00923v1)

**作者:** Luis Frentzen Salim `[一作]` (Academia Sinica), Hsing-Kuo Kenneth Pao `[通讯]` (National Taiwan University of Science and Technology)

**通讯引用:** 939 | [OpenAlex ID](https://openalex.org/A5030897009)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多语言大型语言模型在训练过程中前后层的功能分化，并提出只调节模型两端层的 CogSym 方案，显著降低适配成本。

**💡 创新点**

创新点在于将语言感知（理解）与生成（产出）与人类大脑语言中枢相对应，证明前层主要负责理解，后层负责生成；基于此发现提出的仅调节两端层的对称策略既无额外数据又能获得接近全模型微调的性能。

**🔧 技术方法**

使用层级消融、前后层对称微调、LoRA 参数高效微调以及 FIM、LSN、AlphaLoRA 等层选择方法对比；核心技术是对 Transformer 的层级功能进行实证拆分，并设计 CogSym 的层分配策略。

**📊 数据集**

实验数据集包括 Javanese 与 Sundanese 的 Alpaca 翻译数据、NusaX 机器翻译、情感分析与 IndoMMLU；这些均为低资源语言场景。

**📈 对比分析**

与全模型微调相比，CogSym 只训练 25% 外层即可保持 2–3% 的性能差距；在 LoRA 训练下仍保持一致；与 FIM、LSN、AlphaLoRA 等方法相比，CogSym 既简化了流程又无需额外计算，性能相当。

**⚠️ 局限性**

局限性：仅在两种低资源语言上验证，缺乏对更高资源语言或其他架构的推广；未深入探讨中层功能及更复杂任务（推理、数学等）；顺序微调可能导致灾难性遗忘，需进一步研究。

---

## 410. Orthogonal Learner for Estimating Heterogeneous Long-Term Treatment Effects

**arXiv ID:** 2604.00915 | [PDF](https://arxiv.org/pdf/2604.00915v1)

**作者:** Haorui Ma `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Long-Term Orthogonal Learners (LT‑O‑learners)，在结合短期随机实验数据与长期观察数据的异质长期治疗效应（HLTE）估计框架中，通过自定义重叠权重和正交损失实现对低重叠情况的鲁棒性。

**💡 创新点**

创新点：① 通过重目标学习目标与重叠权重实现对有限重叠的下采样；② 利用有效影响函数得到的Neyman正交损失，使估计对nuisance误差二阶不敏感；③ 针对不同重叠类型（仅治疗、仅长期结果或两者同时低）设计多种权重实例；④ 理论上给出准oracle收敛率并证明正交性。

**🔧 技术方法**

技术：正交学习、有效影响函数、重叠权重、两阶段交叉拟合（cross‑fitting）、机器学习模型（如神经网络）用于nuisance函数估计。

**📊 数据集**

数据集：人工合成数据（四种重叠场景）以及半合成医疗数据（基于第三届国际卒中试验IST‑3的7天 surrogate 与6个月长期结果）。

**📈 对比分析**

比较方法：与传统的T‑learner、RA‑learner、IPW‑learner、DR‑learner 等基线 Meta‑learners 进行对比，评价指标为 PEHE。实验结果显示，LT‑O‑learners 在所有低重叠场景下均显著优于基线，提升幅度可达 30–50%，在两重低重叠（治疗+长期结果）场景中提高约 40%。

**⚠️ 局限性**

局限性：① 仅针对二元处理，假设满足 surrogate 与 comparability 条件；② 需要同时拥有完整的 surrogate 与长期结果数据，且对长期结果的可观测性依赖性高；③ 主要实验基于仿真与半合成数据，真实应用验证仍有限；④ 对高度维度或噪声丰富的 surrogate 变量，重叠权重与正交性可能仍面临挑战。

---

## 411. Transfer learning for nonparametric Bayesian networks

**arXiv ID:** 2604.01021 | [PDF](https://arxiv.org/pdf/2604.01021v1)

**作者:** Rafael Sojo `[一作]` (Aingura IIoT), Concha Bielza `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 8821 | [OpenAlex ID](https://openalex.org/A5010920098)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了两种针对稀缺数据的非参数贝叶斯网络迁移学习方法——PC-stable迁移学习（PCS‑TL）与hill climbing迁移学习（HC‑TL）

**💡 创新点**

创新点在于首次为非参数贝叶斯网络设计迁移学习框架，结合SJS相似度、目标信任因子和风险度量，并采用线性/对数线性聚合提升结构与参数学习；对负迁移进行自适应过滤

**🔧 技术方法**

技术包括PC‑stable和hill‑climbing结构学习、RCoT条件独立性检验、kernel density estimation 贝叶斯网络、SJS散度、目标信任因子、风险度量、对数线性聚合与交叉验证log‑likelihood

**📊 数据集**

实验使用四个小型半参数贝叶斯网络（SPBN）、两个中大型高斯贝叶斯网络（GBN）以及五个来自UCI的无标签连续数据集（9–24维）

**📈 对比分析**

通过将迁移学习模型与单任务学习模型在log‑likelihood、结构误差（DHD）以及执行时间进行对比，并用Friedman检验+Bergmann‑Hommel后验分析验证统计显著性；结果显示迁移学习在稀缺数据时能显著提升结构与似然，随样本增加最终趋于相同，PCS‑TL在噪声下更易受负迁移影响，HC‑TL更稳健

**⚠️ 局限性**

局限性：PCS‑TL对负迁移敏感，需进一步改进；实验仅在模拟与公开数据集上验证，未在真实工业场景中测试；对高维度或极端数据分布的适用性尚未探究

---

## 412. Two Linear Passes Are Necessary for Sum-Exclude-Self Under Sublinear Space

**arXiv ID:** 2604.01012 | [PDF](https://arxiv.org/pdf/2604.01012v1)

**作者:** Andrew Au `[一作]` `[通讯]` (Independent Researcher), Andrew Au (Independent Researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

证明了在子线性工作空间下，求无自包含求和的数组必须至少进行两次线性遍历；

**💡 创新点**

利用 choke-point 技术构造信息论下界，证明任何算法在第一遍至少读 n-1 个元素，第二遍至少读 n-⌊t/d⌋ 个元素，从而给出 2n-1-⌊t/d⌋ 的读取下界；

**🔧 技术方法**

信息论计数、choke-point 断点、输入输出可重构性证明等；

**📊 数据集**

无；

**📈 对比分析**

无实验比较，结论为理论下界，说明标准两遍算法几乎最优；

**⚠️ 局限性**

未能完全消除标准算法与下界之间的少量读数差距，未探讨更优摘要方案及其可行性。

---

## 413. Customizing Large Vision Model-Guided Low-Rank Approximation for Ground-Roll Denoise

**arXiv ID:** 2604.00998 | [PDF](https://arxiv.org/pdf/2604.00998v1)

**作者:** Jiacheng Liao `[一作]` (University of Electronic Science and Technology of China), Yongjian Guo `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了 LVM-LRA 框架，实现无监督的地面滚动（ground‑roll）抑制；通过提示可编程的大视觉模型生成语义掩码，再嵌入低秩分解实现信号与噪声分离。

**💡 创新点**

创新点在于：①利用无训练的 CLIPSeg 与混合提示（形态、物理、少量实例）生成高质量软掩码；②把掩码嵌入到掩码约束的低秩优化中，兼顾物理一致性；③采用 ADMM 求解，保持计算轻量并可跨数据集迁移。

**🔧 技术方法**

技术手段包括：大视觉模型 CLIPSeg + 多模态提示、基于核范数的低秩分解、掩码约束、ADMM+奇异值阈值(SVT)求解、截断 SVD 加速；并在推理阶段实现无训练、无标注的完全自动化流程。

**📊 数据集**

实验数据集：合成 VSP 数据（加入已知 ground‑roll 并提供参考），以及两套真实 VSP 数据（不同地质结构）。

**📈 对比分析**

与 F‑K 滤波、带通滤波和隐式神经表示（INR）等基线比较。评估指标：合成数据使用 SNR；真实数据使用局部相似度。LVM‑LRA 在合成数据上 SNR 达 14.5 dB，明显高于 7‑8 dB 的基线；在真实数据上局部相似度最低，表明信号泄漏最小；视觉和频谱结果也显示更优的噪声抑制与反射保留。

**⚠️ 局限性**

局限性：当地面滚动与反射在频率/斜率上重叠过度时，掩码精度对分离效果影响更大；目前方法对窄斜率、强重叠场景的鲁棒性有限，需进一步改进掩码估计和更强的自适应低秩模型。

---

## 414. Round-efficient Fully-scalable MPC algorithms for k-Means

**arXiv ID:** 2604.00954 | [PDF](https://arxiv.org/pdf/2604.00954v1)

**作者:** Shaofeng H. -C. Jiang `[一作]`, Weicheng Wang `[通讯]` (Peking University)

**通讯引用:** 12692 | [OpenAlex ID](https://openalex.org/A5019005101)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在并行计算框架MPC下提出了一套三阶段的聚类算法，能够在欧氏空间中对k中心问题实现近似求解。

**💡 创新点**

创新点在于将传统序列化的稀疏化和圆整过程拆分为可并行执行的三阶段流程，并证明了在MPC规则集帮助下能保持低开源解的约束，从而实现近似圆整。

**🔧 技术方法**

利用MPC规则集、范围查询、近似最近邻搜索等几何原语实现了稀疏化、分段圆整以及最终的整数圆整。

**📊 数据集**

实验使用的是理论上的任意欧氏点集（n点，d维，aspect ratio Δ），未给出特定实际数据集。

**📈 对比分析**

相比传统序列化算法，新方法在O(log_s n)回合内完成，近似因子为( log n / log log n )^z，若维度较小则可达常数近似，空间复杂度为O(n^{1+ϵ} d)。

**⚠️ 局限性**

局限性在于高维空间下空间复杂度仍较高；当k=1时需要额外的采样方案；常数近似仅在低维情况下可实现。

---

## 415. Uncertainty-Aware Variational Reward Factorization via Probabilistic Preference Bases for LLM Personalization

**arXiv ID:** 2604.00997 | [PDF](https://arxiv.org/pdf/2604.00997v1)

**作者:** Gyuseok Lee `[一作]` (University of Illinois Urbana Champaign), Dong Wang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于变分推理的奖励因子化方法 VRF，用来实现大语言模型的个性化对齐。

**💡 创新点**

创新点包括：①将用户偏好建模为共享潜在空间中的高斯分布；②使用 Wasserstein 距离对用户分布与共享高斯基进行不确定性感知匹配；③引入方差衰减的 Bradley‑Terry 损失以在训练中自动抑制不确定度高的样本。

**🔧 技术方法**

核心技术包括：变分编码器、产品高斯聚合、共享概率偏好基、Wasserstein 距离软匹配、方差衰减的对数似然损失和混合高斯正则化。

**📊 数据集**

实验使用了三大基准数据集：PersonalLLM（含多样化偏好控制参数 α）、TL;DR（每位用户训练样本数 n）和 PRISM（真实用户对话）。

**📈 对比分析**

与 BT、VPL、PAL、LoRe、PReF 等基线进行对比，VRF 在所有数据集、见过用户与未见用户、少量样本以及不同不确定度水平下均取得最高的对比偏好准确率，并在推理时的 best‑of‑N 对齐效果上提升了 ΔWinRate，且单用户适配时间仅为 0.6 ms，显著低于梯度优化方法。

**⚠️ 局限性**

局限性包括：①方法在极大规模用户群下的训练成本与内存开销未完全评估；②仍依赖于固定数量的共享基，可能无法覆盖所有潜在偏好；③缺乏针对查询上下文动态适配的机制，未来需进一步扩展。

---

## 416. Multimodal Analysis of State-Funded News Coverage of the Israel-Hamas War on YouTube Shorts

**arXiv ID:** 2604.00994 | [PDF](https://arxiv.org/pdf/2604.00994v1)

**作者:** Daniel Miehling `[一作]` (Indiana University), Sandra Kuebler `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一套多模态分析管线，对 2023–2024 年间四个主流国有/国资媒体（阿尔及尔、BBC、DW、TRT）在 YouTube Shorts 上的以色列‑哈马斯冲突报道进行情感和视觉场景的系统分析。

**💡 创新点**

创新点：①将自动转写、依存句法提取、Aspect‑Based Sentiment Analysis（ABSA）与视觉场景分类统一到同一管线，首次在短视频新闻中量化情感与视觉语义的协同；②提出专门针对冲突报道的七类视觉场景词表，并利用开源 4B VLM 进行高效分类；③在域适配上发现小型 Transformer 经过微调后可超越大型模型与 LLM，强调任务特化而非模型规模。

**🔧 技术方法**

技术与方法：ASR（Whisper）+ OCR；依存句法分析（spaCy + UD）；Aspect‑Based Sentiment Analysis（BERT‑base、RoBERTa‑base、T5‑base、Yang、Qwen‑LLM）微调；视觉场景分类使用 4B GPT‑4‑Vision‑like VLM；数据采集通过 YouTube Data API；后处理包括依存三元组抽取、文本清洗、语义标签分配。

**📊 数据集**

数据集：约 2,335 条符合条件的 YouTube Shorts，涵盖 12 个月，4 个频道共 94,300+ 帧；转写文本 3,252 句；按 7 类视觉场景标签划分的 79,900+ 帧。

**📈 对比分析**

性能对比：ABSA 微调后最佳模型（BERT‑base）宏 F1 = 81.9，优于 RoBERTa、T5、Yang 以及 Qwen‑LLM（宏 F1 72.5）；视觉场景分类准确率为 86.9%（694/799 正确），在新闻采访、灾难现场、政治事件等类别上表现尤佳；对比大型模型，开源小模型在此任务上既节省算力又保持高精度。

**⚠️ 局限性**

局限性：①仅处理英文内容，未覆盖其他语言；②视频样本为人工预选，可能不具代表性；③“其他/未知”视觉标签误判率高，需更精细提示；④未考察多平台（TikTok、Instagram）与多语言扩展；⑤AI 生成图像可能干扰视觉分析，未作鉴别；⑥情感与视觉的跨模态对齐仍未深入，需进一步研究。

---

## 417. Do Phone-Use Agents Respect Your Privacy?

**arXiv ID:** 2604.00986 | [PDF](https://arxiv.org/pdf/2604.00986v1)

**作者:** Zhengyang Tang `[一作]` (Chinese University of Hong Kong), Benyou Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究手机使用代理在完成常规任务时是否遵守隐私，构建了可验证的评估框架iMy并在10个模拟应用上进行实验。

**💡 创新点**

创新点在于将隐私合规定义为可执行的“iMy隐私合约”，并通过可观测的模拟应用和基于规则的审计，将隐私违规行为转化为可计量指标。

**🔧 技术方法**

采用了Android模拟器、iMy工具、可插拔的Mock App、权限请求接口以及多模型对话循环（Claude、Gemini、Qwen等）实现。

**📊 数据集**

数据集为300个从10个控制应用生成的任务，覆盖预订、过滤、取消、保存偏好等场景，任务通过种子数据库生成。

**📈 对比分析**

通过五大前沿模型（Claude Opus 4.6、Gemini 3 Pro、Qwen 3.5 Plus、Kimi K2.5、Doubao Seed 1.8）对比任务成功率、平均隐私分、隐私合规成功率和后续会话偏好使用率，发现没有单一模型在所有指标上领先，且隐私合规率显著低于单纯成功率。

**⚠️ 局限性**

限制在于仅评估“行为隐私”且仅在模拟应用与白名单权限下进行，未涵盖跨应用泄露、网络传输隐私等真实世界风险，且用户模拟始终同意授权。

---

## 418. DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving

**arXiv ID:** 2604.00969 | [PDF](https://arxiv.org/pdf/2604.00969v1)

**作者:** Yiyao Zhu `[一作]` (HKUST), Shaojie Shen `[通讯]` (HKUST)

**通讯引用:** 17385 | [OpenAlex ID](https://openalex.org/A5001947944)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种两阶段自监督预训练框架 DLWM，用双重潜在世界模型对 Gaussian‑centric 场景表示进行预训练，以提升 3D 占据感知、4D 预测和运动规划。

**💡 创新点**

创新点在于：①首次将 Gaussian flow 引导的潜在世界模型与 ego‑planning 引导的潜在世界模型并行训练；②通过 BEV 先验对稀疏 Gaussian 进行时间一致性约束；③在第一阶段利用多视角深度与语义渲染实现完全无监督的几何与语义学习。

**🔧 技术方法**

使用的技术包括：3D Gaussian Splattering、BEV rasterization、基于 transformer 的 Gaussian 编码器、Gaussian flow 预测、Motion‑Aware Layer Normalization、交叉/自注意力、L2 及 mIoU 损失、AdamW 优化器。

**📊 数据集**

主要使用的公开数据集是 nuScenes（含 1000 条 20 秒视频、RGB+LiDAR）和 SurroundOcc（nuScenes 衍生的 3D 语义占据标注）。

**📈 对比分析**

与基线模型及多种现有方法（如 OccWorld、BEV‑Planner、UniAD）对比，DLWM 在 3D 占据感知上提升 mIoU 1.02 分、在 4D 预测上提升 mIoU 2.68 分，运动规划 L2 距离下降 0.09 m，整体表现达到或超过现有最佳方案。

**⚠️ 局限性**

局限性包括：①需要大规模无标签视频进行预训练，计算与存储成本较高；②仍依赖 LiDAR 深度辅助的深度重建，可能在无 LiDAR 场景下受限；③模型复杂度较高，对硬件和训练时间的要求较高。

---

## 419. FlexAI: A Multi-modal Solution for Delivering Personalized and Adaptive Fitness Interventions

**arXiv ID:** 2604.00968 | [PDF](https://arxiv.org/pdf/2604.00968v1)

**作者:** Shivangi Agarwal `[一作]` (Plaksha University), Siddharth Siddharth `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 FlexAI——一款结合多模态生理感知与大型语言模型（LLM）的实时个性化健身教练系统，能够在运动过程中动态监测姿态、痛感、疲劳与心率，并即时生成针对性训练、恢复、姿势纠正与激励建议。

**💡 创新点**

创新点包括：① 将视觉、音频、可穿戴心率三模态数据通过统一推理管线集成到 LLM 之中，实现全流程的“感知‑推理‑反馈”；② 采用分层提示（Inter‑Exercise 与 Intra‑Exercise）与安全守护策略，保证 LLM 输出既及时又符合训练安全；③ 结合 TTS 与语音交互的“耳机”反馈，提升用户沉浸感与遵从度；④ 通过专家评估与端到端时延测评验证系统的可用性与安全性。

**🔧 技术方法**

核心技术包括：计算机视觉（MediaPipe Pose、SAM 分割、ResNet‑18 痛感分类）、语音信号处理（声学特征 + 语音疲劳检测）、可穿戴心率测量（Lab Streaming Layer 接入 Max‑Health‑Band）、LLM 推理（OpenAI GPT）与层次化提示设计、文本转语音（OpenAI TTS）以及低延迟视觉与音频回路的系统集成。

**📊 数据集**

数据集：① 40 条训练视频（15 公开 YouTube，5 试点会话，10 人工搭建，涵盖姿势误差与正确动作）用于姿势识别、计数与误差检测；② Delaware Pain Dataset（用于训练 ResNet‑18 痛感分类模型）；③ 采集自 25 名参与者的实时心率、语音与面部数据用于验证与评估；此外，系统基于 WHO GPAQ 生成的 MET 评分用于构建 PHR。

**📈 对比分析**

比较方法：① 技术评估——在自建验证集上测量姿势识别、重复计数（97.5%）、姿势误差检测（F1≥0.93）与痛感分类准确率（79.3%）；② LLM 交互评估——三名认证教练对 30 条 LLM 生成干预进行 5 级 Likert 评分（安全 4.72/5，适宜 4.35/5，及时 3.87/5）；③ 用户实验——25 名受试者在对照与 FlexAI 条件下完成同一训练程序，使用 PACES 与 SEES 量表评估情绪与满意度；结果显示 FlexAI 在享受度、成就感提升、无聊与挫败感降低等指标上均显著优于静态控制（p<0.05）。

**⚠️ 局限性**

局限性：① 仅单次实验，缺乏长期使用与健身效果验证；② 受限于小样本与年轻成人群体，结果可能不适用于老年人、慢性病患者或竞技运动员；③ 现有模型基于相关性推断，难以区分安全训练与过度疲劳，安全保障需进一步医学验证；④ 端到端音频反馈时延约 1.4s，可能不足以满足高强度实时姿势纠正的需求；⑤ 某些姿势或功能模块（如瑜伽姿势细节、计数精度）仍需改进；⑥ 对隐私与数据安全的进一步探讨与技术实现（如联邦学习、边缘推理）仍待完善。

---

## 420. Forecasting Motion in the Wild

**arXiv ID:** 2604.01015 | [PDF](https://arxiv.org/pdf/2604.01015v1)

**作者:** Neerja Thakkar `[一作]` (University Of California Berkeley), Carl Doersch `[通讯]` (Google DeepMind)

**通讯引用:** 12028 | [OpenAlex ID](https://openalex.org/A5081047759)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对野外多样动物的未来运动进行预测，提出将稠密点轨迹视为行为视觉标记，并使用扩散Transformer直接生成点轨迹；

**💡 创新点**

创新点在于把点轨迹作为中层视觉标记，结合DINO特征和轨迹速度进行无类别、数据高效的运动预测，并在野外动物数据上实现可控的扩散模型；

**🔧 技术方法**

使用扩散模型（DDPM/DiT）与DDIM采样、DINOv3视觉特征、正弦位置编码等技术；

**📊 数据集**

使用自研的MammalMotion数据集（约300小时野外动物视频），并对MammalNet进行筛选；

**📈 对比分析**

与无学习基线、WHN、ATM、Track2Act等方法比较，在ADE、FDE、VMD、PWT、FD、FVMD等指标上均显著优于对手，尤其在低/中/高运动区间均表现最佳；

**⚠️ 局限性**

局限性包括对极高速度/长时间预测仍受限；仅使用2D轨迹，遮挡处理仍有挑战；需要大量野外视频预处理；模型在非动物/非哺乳动物场景下仍有一定误差。

---

## 421. PDA: Text-Augmented Defense Framework for Robust Vision-Language Models against Adversarial Image Attacks

**arXiv ID:** 2604.01010 | [PDF](https://arxiv.org/pdf/2604.01010v1)

**作者:** Jingning Xu `[一作]` (City University of Hong Kong), Chen Liu `[通讯]` (City University of Hong Kong)

**通讯引用:** 10388 | [OpenAlex ID](https://openalex.org/A5100322126)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了训练无关、仅在推理阶段通过文本增强（释义、分解、聚合）来提升视觉-语言模型对图像对抗扰动的鲁棒性

**💡 创新点**

创新点在于将对抗鲁棒性从视觉空间迁移到文本空间，利用多样化的语义视图和一致性投票实现对抗样本的“随机平滑”效果，并提供多种预算友好的变体

**🔧 技术方法**

采用大语言模型进行问句释义与拆分，利用多视图推理、投票/置信度聚合，以及与图像预处理（JPEG、随机增广）等传统方法对比

**📊 数据集**

在VQA‑v2、ImageNet‑D和MS‑COCO（caption）等公开数据集上进行实验

**📈 对比分析**

与无防御基线、图像侧预处理、提示式推理（CoT、Self‑Consistency）以及对抗训练（TeCoA、FARE）等方法对比，PDA在白盒PGD、EOT、以及多种黑盒攻击下显著提升对抗准确率，且在多数设置下保持或提升干净准确率；效率方面提供了多种压缩版实现

**⚠️ 局限性**

对抗训练方法在鲁棒性上更强但需要显著的计算与梯度信息，PDA虽然在黑盒场景表现稳定，但对大规模、复杂推理任务的提升有限；此外，PDA依赖LLM生成的释义和拆分，LLM的质量会影响鲁棒性，且在某些模型/任务中可解释性与效率仍有提升空间

---

## 422. EgoSim: Egocentric World Simulator for Embodied Interaction Generation

**arXiv ID:** 2604.01001 | [PDF](https://arxiv.org/pdf/2604.01001v1)

**作者:** Jinkun Hao `[一作]` (Shanghai Jiao Tong University), Xudong Xu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了EgoSim，一种闭环 egocentric 世界模拟器，能够根据 3D 点云渲染背景并生成手部交互视频，同时在每一步生成后即时更新 3D 场景状态，实现长时序一致性和物理可塑性；

**💡 创新点**

创新点包括：① 通过 Geometry-action-aware Observation Simulation 模型将动作与 3D 结构结合，保证视角变化下的空间一致性；② 交互感知状态更新模块，能够识别并更新交互对象的 3D 形状，实现可更新的场景记忆；③ 设计大规模可扩展的数据构建流水线，利用单目 egocentric 视频自动提取点云、相机轨迹和手部关键点；④ 低成本 EgoCap 方案实现无标定场景扫描与相机重定位；

**🔧 技术方法**

技术手段包括：点云重建（DepthAnything3、SAM3、DROID‑SLAM）、3D 关键点投影、Diffusion Transformer（DiT）与 VAE 的联合训练、TSDF 融合、视觉语言模型+SAM3 的交互对象识别、低频滤波与轨迹平滑、流匹配推断；

**📊 数据集**

使用了 EgoDex（240K 片段）和 EgoVid（160K 片段）的 web‑scale 单目 egocentric 视频进行训练；在评估时随机抽取 100 个 unseen 视频；并在 AgiBot 数据集上进行机器人手臂仿真；还采集了 50 条低成本 EgoCap 数据进行少量微调；

**📈 对比分析**

与 Wan‑2.1‑14B‑InP、InterDyn、Mask2IV、CosHand 等基线进行对比。单段生成下，EgoSim 在 PSNR、SSIM、LPIPS、Depth‑ERR、Cam‑ERR 上均显著领先；连续生成场景下，虽然略有下降，但仍保持高质量；在机器人仿真中，预训练+微调显著提升性能；

**⚠️ 局限性**

主要局限是单目深度与相机姿态估计在遮挡或高速运动场景下可能失效，导致点云初始化不准确，进而影响后续渲染和状态更新；未来可结合多视角或物理接触约束进一步提升真实性。

---

## 423. ACT Now: Preempting LVLM Hallucinations via Adaptive Context Integration

**arXiv ID:** 2604.00983 | [PDF](https://arxiv.org/pdf/2604.00983v1)

**作者:** Bei Yan `[一作]` (Chinese Academy of Sciences), Xilin Chen `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 35557 | [OpenAlex ID](https://openalex.org/A5083420537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了训练无关的推理干预方法ACT，主动调节多模态注意力以预防视觉模型产生幻觉

**💡 创新点**

创新点在于将视觉上下文探索与语义上下文聚合结合，通过动态头选择和概率融合实现自适应上下文整合

**🔧 技术方法**

主要技术包括跨模态注意力的动态头识别（基于时空协方差相似度STCS）、视觉上下文放大、语义上下文聚合与概率加权融合

**📊 数据集**

使用MSCOCO作离线校准集，评估数据集包括POPE、MME、AMBER、CHAIR等多种判别与生成式幻觉基准

**📈 对比分析**

与现有十余种先进幻觉抑制方法（如Dola、VCD、TAME等）对比，ACT在POPE、MME、CHAIR等指标上均取得最优或同等表现，显著降低幻觉率并提升准确率

**⚠️ 局限性**

局限性包括SCA模块的并行分支导致额外显存和推理延迟，生成更保守可能缩短文本长度，且对复杂推理或关系幻觉的抑制效果仍有限

---

## 424. Enhancing Gradient Inversion Attacks in Federated Learning via Hierarchical Feature Optimization

**arXiv ID:** 2604.00955 | [PDF](https://arxiv.org/pdf/2604.00955v1)

**作者:** Hao Fang `[一作]`, Ke Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种分层特征优化方法，提升联邦学习中梯度反演攻击的效果，并针对梯度裁剪、稀疏化和Soteria防御进行适配。

**💡 创新点**

通过在多层中对中间特征、噪声向量和潜在向量使用l1球约束，并采用自适应半径调度，实现更精准的图像重建，从而突破传统梯度反演的限制。

**🔧 技术方法**

梯度反演攻击、分层特征优化、l1球约束、半径调度、梯度转换处理（裁剪、稀疏化、Soteria）。

**📊 数据集**

使用 BigGAN 和 StyleGAN2 模型，训练数据来自 ImageNet（或对应的图像数据集）。

**📈 对比分析**

与传统的梯度反演方法（如 DLG、DFG 等）在多种防御场景下进行对比，实验显示重建质量提升 10-15%，PSNR/SSIM 显著高于对手方法；在不同裁剪阈值和稀疏率下仍保持较高成功率。

**⚠️ 局限性**

方法依赖于对模型架构的先验了解，计算量较大，且对非视觉任务的推广有限；在极端防御（如强烈噪声注入）下恢复效果仍有限。

---

## 425. Phase transition on a context-sensitive random language model with short range interactions

**arXiv ID:** 2604.00947 | [PDF](https://arxiv.org/pdf/2604.00947v1)

**作者:** Yuma Toji `[一作]` (Hokkaido University), Hideyuki Miyahara `[通讯]` (Hokkaido University)

**通讯引用:** 357 | [OpenAlex ID](https://openalex.org/A5085978119)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `14d48e9d-0069-4ad9-996a-1d5968216998` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个短程相互作用的上下文敏感随机语言模型（CSG），并通过数值模拟研究其热力学行为，发现存在相变。

**💡 创新点**

创新点在于首次证明在仅含短程相互作用的语言生成过程中即可出现BKT相变，表明语言的本质结构而非长程耦合是相变的根源。

**🔧 技术方法**

采用了统计力学方法（磁化强度、磁化率、Binder参数、相关函数）与Metropolis‑Hastings算法对模型进行数值模拟，并利用有限尺寸标度分析确定临界指数。

**📊 数据集**

使用自生成的句子数据集，参数取 K=20、100、500 等，模拟不同温度、生成规则概率（q、t）下的句子长度和符号分布。

**📈 对比分析**

通过与先前的长程相互作用模型对比（相变温度、相图、Binder曲线、相关函数指数）验证了BKT相变特征；数值结果显示临界指数接近理论预期，且相变温度随规则参数变化而显著变化。

**⚠️ 局限性**

局限性包括：有限尺寸效应导致相变尖锐度不明显；模型极度简化，未包含真实语言的语法多样性；对 t=0.5 的临界行为尚未完全阐明；未在实际语料上验证相变现象。

---

## 426. WARP: Guaranteed Inner-Layer Repair of NLP Transformers

**arXiv ID:** 2604.00938 | [PDF](https://arxiv.org/pdf/2604.00938v1)

**作者:** Hsin-Ling Hsu `[一作]` (National Chengchi University), Fang Yu `[通讯]` (National Chengchi University)

**通讯引用:** 6022 | [OpenAlex ID](https://openalex.org/A5082198869)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为WARP的后训练修复框架，利用凸二次规划在Transformer的最后一层之前的全连接层上进行权重微调，以保证修复样本的分类正确性并保持其余样本的表现。

**💡 创新点**

创新点包括：①将可证明修复扩展到Transformer的中间层而非仅限于最终分类层；②通过一次性求解凸二次规划获得每个修复样本的间隔（gap）正值、剩余样本的间隔保持以及基于Lipschitz的鲁棒半径等三项可验证保证；③引入Gap Sensitivity Norm（GSN）诊断与GSN-FT预处理，保证优化可行性；④采用低秩参数化与迭代重新线性化提升求解效率。

**🔧 技术方法**

使用技术包括：前向一阶线性化、凸二次规划（QP）求解、低秩（rank‑2）更新、GAP敏感度测度、GSN‑FT微调、Lipschitz连续性下的鲁棒半径计算、迭代优化与收敛判据。

**📊 数据集**

数据集与任务：AdvGLUE中的SST‑2和RTE二分类任务，利用六种对抗攻击（CheckList、SememePSO、StressTest、T3、TextBugger、TextFooler）构建修复集、剩余集与攻击泛化集；模型为DistilBERT和BERT的encoder‑only版本。

**📈 对比分析**

与两类基线对比：梯度基LoRA（rank‑2）和全层微调。WARP在所有配置下实现100%修复与剩余准确率，攻击泛化率高达69.0%（相较LoRA提升12.3pp，较全层微调提升7.8pp），在一般准确率上略低于LoRA但在RTE任务中优于全层微调；同时提供每样本的gap与鲁棒半径证书。

**⚠️ 局限性**

局限性：Lipschitz上界保守导致鲁棒半径较小；仅针对encoder‑only结构和分类任务；低秩更新可能限制修复空间；GSN‑FT的额外微调步骤虽小，但仍需额外计算。

---

## 427. Portable and Secure CI/CD for COBOL: Lessons from an Industrial Migration

**arXiv ID:** 2604.00936 | [PDF](https://arxiv.org/pdf/2604.00936v1)

**作者:** Andreas Askholm `[一作]` (Bankdata), Jacopo Mauro `[通讯]` (University of Southern Denmark)

**通讯引用:** 1334 | [OpenAlex ID](https://openalex.org/A5058872144)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将Bankdata现有的基于Jenkins和Groovy的COBOL CI/CD管道迁移到容器化、平台无关的架构，抽象平台逻辑、统一仓库结构并预构建OCI兼容镜像。

**💡 创新点**

设计并实现了平台抽象层，消除对特定CI/CD工具的耦合；使用容器化预构建镜像降低运行时依赖，形成可复用、可迁移的工业级迁移蓝图。

**🔧 技术方法**

使用的技术包括：Jenkins、GitHub Actions、Groovy、Gradle、GnuCOBOL、COBOL Expander、COBOL Check、Zowe CLI、Node.js / npm、Docker（OCI镜像）以及Spock测试框架。

**📊 数据集**

在评估中使用Bankdata真实的COBOL源代码及其单元测试（5个生产级文件）作为工作负载；性能测量基于同一工作负载在两个平台上多次执行。

**📈 对比分析**

通过在相同工作负载下分别运行Jenkins和GitHub Actions管道，记录平均运行时间并进行统计；结果显示Jenkins平均724 s，GitHub Actions平均130 s，性能提升约82.1%。

**⚠️ 局限性**

局限性包括：性能提升部分归因于平台差异，未在其他CI/CD平台复测；未覆盖持续部署场景；作者参与迁移可能引入偏见；安全与合规细节在不同环境下的可迁移性仍待验证。

---

## 428. Generalization Bounds for Spectral GNNs via Fourier Domain Analysis

**arXiv ID:** 2604.00918 | [PDF](https://arxiv.org/pdf/2604.00918v1)

**作者:** Vahan A. Martirosyan `[一作]` (Université Paris-Saclay, CentraleSupélec, Inria), Fragkiskos D. Malliaros `[通讯]` (Université Paris-Saclay, CentraleSupélec, Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过在图傅里叶域对谱图神经网络（Spectral GNN）进行分析，推导出深度和多项式阶数依赖的泛化上界，并将此上界与网络的雅可比矩阵范数关联，提出了一种基于频谱能量的正则化方法。

**💡 创新点**

创新点在于：①证明全图高斯复杂度对图傅里叶变换不变，能够将空间域的复杂度转化为频域的逐频乘法；②得到数据相关、深度和多项式阶数显式的泛化上界；③给出深度线性谱网络更紧的上界；④将泛化和稳定性归因于同一组结构参数（激活 Lipschitz、权重范数、基函数放大系数）；⑤提出能量加权正则化，并在多基函数实验中验证其有效性。

**🔧 技术方法**

主要技术包括：图傅里叶变换、全图高斯复杂度、变换后激活的 Lipschitz 保证、通用多项式基（Chebyshev、Bernstein、Legendre、Monomial）、线性和非线性谱网络的递归分析、雅可比矩阵范数上界、正则化项的设计与实现。

**📊 数据集**

使用了四个常用的节点分类基准数据集：Cora、Citeseer、Chameleon、Squirrel，并在每个数据集上采用10个随机稀疏划分（每类10个训练样本）进行实验。

**📈 对比分析**

实验比较了不同多项式基函数与/无正则化下的测试准确率和泛化间隙。结果显示：在均匀放大系数的Chebyshev基上，加入能量加权正则化能够持续减小泛化间隙并提升准确率；而对非均匀基（Monomial、Bernstein、Legendre）则效果不佳或不稳定。

**⚠️ 局限性**

局限性包括：仅在传导式固定图场景下分析，未考虑归纳学习、图结构变化或分布偏移；仅研究多项式谱滤波，未覆盖注意力或非多项式运算；激活函数需满足 Lipschitz 且参数有范数约束；非线性上界仍依赖连乘 Lipschitz 约束，可能不够紧；最终的泛化上界更多用于解释趋势而非精准预测准确率。

---

## 429. Investigating Autonomous Agent Contributions in the Wild: Activity Patterns and Code Change over Time

**arXiv ID:** 2604.00917 | [PDF](https://arxiv.org/pdf/2604.00917v1)

**作者:** Razvan Mihai Popescu `[一作]` (Delft University of Technology), Maliheh Izadi `[通讯]` (Delft University of Technology)

**通讯引用:** 665 | [OpenAlex ID](https://openalex.org/A5064355563)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了约11万条公开开源 PR 的大规模数据集，系统分析了五种主流自动编码代理（OpenAI Codex、Claude Code、GitHub Copilot、Google Jules、Devin）与人类开发者在 PR 活动、Merge 率、Merge 时延、文件类型、代码变更规模、代码生存率和 churn 等维度的差异，并揭示代理 PR 在低星项目更常见且更易产生代码 churn。

**💡 创新点**

创新点在于首次提供了规模庞大、跨多代理的长周期 PR 数据集，并通过实证分析展示了代理驱动开发对代码可维护性（Survival 率低、Churn 率高）及社区协作模式（Merge 率差异、评论/评审行为）的影响，为评估自动编码代理的真实世界性能提供了新基准。

**🔧 技术方法**

采用 GitHub GraphQL API 抓取 PR 与相关元数据，利用代理特定的 branch prefix、作者标识和 watermark 进行识别；计算 Merge Rate、Merge Time、Comment/Review 计数、文件类型分布、Change Size、Survival Rate、Churn Rate、Deletion Rate 等指标，并使用统计检验（Mann‑Whitney U、Cliff's δ）对不同代理与人类 PR 进行比较。

**📊 数据集**

使用了公开的 GitHub 开源仓库数据，经过筛选后得到 111,969 条 PR（约 110k 代理 PR + 人类 PR），包含 PR 状态、提交、评论、评审、issue、文件变更等信息，已在 Hugging Face 上公开（https://huggingface.co/datasets/AISE-TUDelft/MOSAIC-agentic-3m）。

**📈 对比分析**

通过对 Merge 率、Merge 时延、评论/评审频率、Change Size、文件类型比例、Survival Rate、Churn Rate 等维度进行定量对比，结果显示代理 PR 的 Merge 率略高于人类但 Merge 时延更短，代码变更规模更大，Survival 率显著低于人类，Churn 率明显高，表明代理贡献在易维护性和代码稳定性上存在劣势。

**⚠️ 局限性**

局限性包括：代理识别依赖特定信号，可能漏检或误检；数据时间窗口仅覆盖 2025 年 6‑8 月，未覆盖更长期动态；仅考虑可识别的 PR，未覆盖所有代理活动；人类 PR 中仍可能包含未被检测到的代理贡献；以及 Repo 星级分布不均导致代理与人类在不同项目类型上的可比性受限。

---

## 430. Representation Selection via Cross-Model Agreement using Canonical Correlation Analysis

**arXiv ID:** 2604.00921 | [PDF](https://arxiv.org/pdf/2604.00921v1)

**作者:** Dylan B. Lewis `[一作]` (University of Tennessee), Hector Santos-Villalobos `[通讯]` (University of Tennessee)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5083817903)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用预训练视觉模型之间的跨模型一致性（Canonical Correlation Analysis, CCA）作为后置表示选择与降维工具，无需额外训练即可压缩、精炼和迁移特征；

**💡 创新点**

创新点在于把 CCA 从单纯的分析工具转变为直接作用于表示的投影算子，利用两模型间可互预测的方向来挑选共享语义子空间，从而实现无监督的压缩、精炼和知识迁移；

**🔧 技术方法**

核心技术包括 CCA（配合 ZCA whitening）、PCA 对比、ViT 编码器、线性分类器、GPU 加速实现；

**📊 数据集**

评估数据集涵盖 ImageNet‑1k、CIFAR‑100、MNIST、Caltech‑101、Oxford‑IIIT Pets，以及使用 LAION‑400M 训练的 CLIP‑ViT；

**📈 对比分析**

与未投影基线和 PCA 进行对比，实验显示：平均提升 1.0% 相较基线，2.0% 相较 PCA；降维可达 57% 且仍提升；在固定维度下可提升约 8%；在任务迁移场景中提升 8–12%（最高 12.6%）；整体效果优于传统 PCA；

**⚠️ 局限性**

局限性包括：对严重类别不平衡敏感，CCA 在高不平衡数据上表现下降；需要两模型容量相近且协方差估计可靠；仅实现线性投影，无法捕捉非线性关系。

---

## 431. Maximizing T2-Only Prostate Cancer Localization from Expected Diffusion Weighted Imaging

**arXiv ID:** 2604.00985 | [PDF](https://arxiv.org/pdf/2604.00985v1)

**作者:** Weixi Yi `[一作]` (University College London), Yipeng Hu `[通讯]` (University College London)

**通讯引用:** 5261 | [OpenAlex ID](https://openalex.org/A5032309114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出一种泛化期望最大化（GEM）框架，在训练时利用DWI作为隐变量，推理时仅使用T2‑w序列完成前列腺癌区级定位。

**💡 创新点**

将隐变量DWI建模为生成式后验，并在E‑step中用流匹配生成器近似该后验，在M‑step中联合优化分类器与生成器，实现无DWI推理下接近多模态上限的性能。

**🔧 技术方法**

使用流匹配生成模型（flow matching）+AutoencoderKL、3D ResNet‑FPN+RoIAlign、条件随机场（CRF）以及GEM算法。

**📊 数据集**

在PROMIS、Targeted（私有）和PI‑CAI三大数据集上共计4133例mpMRI与组织学标签进行训练与评估。

**📈 对比分析**

与单模态基线、全模态基线及ModDrop、SimMLM、KD‑Net等方法对比，区级QWK提升至≈26.7（比T2‑w基线+9.0点），病人级AUC提升至≈70.5，优于多数方法并逼近多模态上限。

**⚠️ 局限性**

受域迁移影响，生成器在PI‑CAI等外部数据上的泛化受限；生成模型仍需预训练AutoencoderKL，且推理速度相对传统方法仍略慢。

---

## 432. Differentially Private Manifold Denoising

**arXiv ID:** 2604.00942 | [PDF](https://arxiv.org/pdf/2604.00942v1)

**作者:** Jiaqi Wu `[一作]` (National University of Singapore), Zhigang Yao `[通讯]` (National University of Singapore)

**通讯引用:** 2319 | [OpenAlex ID](https://openalex.org/A5079144613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种在私人参考数据下进行隐私保护的流形去噪框架，通过私有化局部均值和切空间，迭代纠正公共查询点，使其逼近隐藏的低维流形。

**💡 创新点**

创新点在于：①首次将局部PCA与差分隐私结合，直接私有化切空间投影和加权均值；②构建模块化、可扩展的迭代更新流程；③使用zCDP进行精确的隐私预算分配；④给出非渐进的误差上界，量化曲率、测量噪声和隐私噪声的影响。

**🔧 技术方法**

核心技术包括：差分隐私（Gaussian机制、zCDP）、局部PCA、加权均值与切空间估计、投影残差矫正、迭代更新（固定点法）以及精确的隐私账本。

**📊 数据集**

实验数据集包括：合成流形（圆、环面、瑞士卷、球面），英国生物银行血/尿标记物面板，十个公开单细胞RNA‑seq数据集。

**📈 对比分析**

与非私有去噪、梯度下降等基线相比，所提方法在保持接近非私有性能的同时提供正式的差分隐私保证；在合成实验中误差随样本量下降，隐私预算增大时表现趋于最优；在生物医学和单细胞数据中，去噪后聚类ARI显著提升，风险模型的AUC与非私有方法相近。

**⚠️ 局限性**

局限性：①误差仅能抵消到一阶噪声；②当前仅支持有界噪声，无法处理无界或重尾噪声；③无法一次性发布完整流形，仅能针对查询点去噪；④仅处理公共查询，未覆盖双侧隐私场景。

---

## 433. Autoregressive Appearance Prediction for 3D Gaussian Avatars

**arXiv ID:** 2604.00928 | [PDF](https://arxiv.org/pdf/2604.00928v1)

**作者:** Michael Steiner `[一作]` (Graz University of Technology), Michael Zollhöfer `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于3D Gaussian Splatting的头像模型，通过局部姿势编码与每帧外观潜码实现高保真渲染。

**💡 创新点**

创新点在于局部姿势条件化、学习外观潜码以及自回归预测器来消除姿势-外观混淆并提升时序稳定性。

**🔧 技术方法**

采用3D Gaussian Splatting、皮肤权重掩蔽、局部PCA约束、卷积外观编码器以及Transformer自回归预测。

**📊 数据集**

在六个大型多视角全景捕获数据集上进行训练与评估。

**📈 对比分析**

与MMLPs和nRFGCA基线对比，实验表明在训练集拟合和测试集时序稳定性方面均优于基线，PSNR/SSIM/LPIPS得分最高。

**⚠️ 局限性**

局限性包括无法完全分离姿势驱动与外观变化、局部姿势条件化难以捕捉全局阴影效应，以及预测器的热启动需要额外时间。

---

## 434. Learning Quantised Structure-Preserving Motion Representations for Dance Fingerprinting

**arXiv ID:** 2604.00927 | [PDF](https://arxiv.org/pdf/2604.00927v1)

**作者:** Arina Kharlamova `[一作]` (MBZUAI), Xue Liu `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出端到端舞蹈运动检索框架，将原始视频转为离散可解释的运动签名并实现大规模检索。

**💡 创新点**

创新点在于将 Skeleton Motion Quantisation 与 Spatio‑Temporal Transformers 结合，构建结构保留的离散词汇表，并通过两阶段直方图索引+对齐重排序实现高效检索。

**🔧 技术方法**

使用 CoMotion 3D 骨骼估计、SMQ+STT 向量量化、EMA 更新、直方图索引以及多种时间对齐距离（TWED、LCSS、EDR、ERP 等）等技术。

**📊 数据集**

实验使用自建的基于 TikTok/YouTube Shorts 的 Fortnite Emote 数据集，并在 AIST 舞蹈数据集上进行评估。

**📈 对比分析**

与全量暴力检索对比，直方图索引仅损失约4.3%性能，Top‑3 匹配率约0.49，Rank‑1 约38%；在更大规模数据集上仍能保持 40% 以上 Rank‑1。

**⚠️ 局限性**

局限性包括对时间顺序的丢失、缺乏语义标签、固定距离权重、依赖离散词典导致细粒度差异难以捕捉以及对复杂舞蹈结构的建模不足。

---

## 435. OrgAgent: Organize Your Multi-Agent System like a Company

**arXiv ID:** 2604.01020 | [PDF](https://arxiv.org/pdf/2604.01020v1)

**作者:** Yiru Wang `[一作]` (Chinese University of Hong Kong), Tsung-Yi Ho `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6621 | [OpenAlex ID](https://openalex.org/A5062800747)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 OrgAgent 的公司式层级多智能体框架，将协作划分为治理层、执行层和合规层，以改进大语言模型的多智能体推理能力。

**💡 创新点**

创新点在于将组织结构视为核心变量，通过明确的层级治理、执行与审查机制，实现了更高效、可解释的多智能体协作，并系统评估了层级与扁平结构的差异。

**🔧 技术方法**

采用大语言模型（GPT‑OSS‑120B、GPT‑5 mini、Llama‑3.1‑8B）与技能型工作池，配合不同的执行模式（DIRECT、LIGHT MAS、FULL MAS）和执行策略（STRICT、BALANCE、NOCAP、AUTO）实现协同推理。

**📊 数据集**

实验使用了 MuSR、MuSiQue 与 SQuAD 2.0 三个推理基准数据集。

**📈 对比分析**

在与单一代理和扁平多智能体的基线对比下，OrgAgent 在 MuSiQue、SQuAD 2.0 上实现了最高达 123.99% 的 F1 提升，同时在 SQuAD 2.0 上将 token 消耗减少 74.52%，显示出显著的性能与效率双赢。

**⚠️ 局限性**

局限性包括对多项选择题的提升有限、讨论轮数上限导致的潜在推理不完整、仍存在错误信息被错误传递的风险，以及实验范围受限于模型、任务和组织设置，没有评估延迟、稳定性或人工评价。

---

## 436. AutoMIA: Improved Baselines for Membership Inference Attack via Agentic Self-Exploration

**arXiv ID:** 2604.01014 | [PDF](https://arxiv.org/pdf/2604.01014v1)

**作者:** Ruhao Liu `[一作]` (National University Of Singapore), Xinchao Wang `[通讯]` (National University Of Singapore)

**通讯引用:** 13421 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AutoMIA 框架，通过代理自动生成、执行并迭代改进 logits 级别的成员推断攻击策略，降低对手工特征工程的依赖。

**💡 创新点**

创新点在于将成员推断视为 agent 生成策略的闭环自动化过程；利用历史策略库、Guidance 代理的反射反馈、滑动窗口选择以及多指标加权评分，实现在模型无关、对噪声反馈鲁棒的策略搜索。

**🔧 技术方法**

使用大语言模型（如 DeepSeek‑V3.2‑Reasoner、Gemini 3 Flash、Grok 4.1 Fast、Qwen3‑Max）作为代理；结合可执行 logits 代码、策略库、滑动窗口、加权 AUC/Acc/TPR 评分、Guidance 代理等技术。

**📊 数据集**

实验数据集为 VL‑MIA/Text、VL‑MIA/DALL·E、VL‑MIA/Flickr；目标模型为 MiniGPT‑4、LLaVA‑1.5、LLaMA‑Adapter；采用灰盒访问假设。

**📈 对比分析**

与传统手工指标（Perplexity、Max Prob Gap、Min‑k%、Rényi 等）及多种基线比较，AutoMIA 在文本、图像和多模态设置中持续跑在前列，AUC、Accuracy、TPR@5%FPR 均显著提升，往往超过最强基线。

**⚠️ 局限性**

局限性：需要灰盒访问；对代理推理能力和 token 成本敏感；在极端分布漂移或更严格的 IID 评估中性能下降；未覆盖纯黑盒或多轮查询场景。

---

## 437. OmniMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory

**arXiv ID:** 2604.01007 | [PDF](https://arxiv.org/pdf/2604.01007v1)

**作者:** Jiaqi Liu `[一作]` (UNC-Chapel Hill), Huaxiu Yao `[通讯]` (UNC-Chapel Hill)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用自主研究管道，开发并验证了一套统一的多模态终身记忆框架，显著提升了 LoCoMo 与 Mem-Gallery 两个基准的 F1 分数。

**💡 创新点**

创新点在于：①通过自治实验管道完成代码修复、架构重构和提示工程，突破传统 AutoML 只能调整超参数的局限；②提出了多模态输入的新颖性过滤、MAU（Multimodal Atomic Unit）统一表示、金字塔式检索与稠密-稀疏-图融合检索的组合方案；③系统性展示了自动化探索在多组件 AI 系统中的有效性。

**🔧 技术方法**

采用的技术包括：23 阶段 LLM 驱动的自治实验管道、FAISS+BM25+图检索混合方案、MAU 轻量化存储结构、CLIP+MiniLM 进行多模态嵌入、GPT‑4o 等 LLM 进行实体抽取与摘要生成，并行检索实现高吞吐。

**📊 数据集**

使用的数据集为 LoCoMo（1,986 条多轮对话 QA）和 Mem‑Gallery（240 条多模态对话 + 1,003 张图像，1,711 条 QA），均为公开基准。

**📈 对比分析**

在六个基线（MemVerse、Mem0、Claude‑Mem、A‑MEM、MemGPT、SimpleMem）与五种 LLM 主干（GPT‑4o、GPT‑4o‑mini、GPT‑4.1‑nano、GPT‑5.1、GPT‑5‑nano）下对比，所提出的框架从 LoCoMo 的 0.117/0.254 提升至 0.598/0.797，超越现有最优约 20–30% 的 F1，且在所有评测指标上均保持领先。

**⚠️ 局限性**

局限性包括：仅在公开基准上验证，未涉及真实用户数据与隐私合规问题；自治管道对计算资源与 LLM 的依赖较高，部署成本显著；系统在生产环境下可能产生数据泄露、隐私侵害或不透明决策风险。

---

## 438. Faster Approximate Fixed Points of $\ell_\infty$-Contractions

**arXiv ID:** 2604.01006 | [PDF](https://arxiv.org/pdf/2604.01006v1)

**作者:** Andrei Feodorov `[一作]` (ETH Zurich), Sebastian Haslebacher `[通讯]` (ETH Zurich)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5019688727)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种新的查询高效算法，能够在时间 (log(1/ε))^{d log d} 内找到 ℓ∞-收缩函数的 ε-近似不动点，并进一步结合分解定理得到时间 (log(1/ε))^{√d log d}、查询数 (log(1/ε))^{√d log d} 的算法；同时证明该算法可用于提高求解 Shapley 随机博弈的确定性算法复杂度。

**💡 创新点**

1) 用新方法替代原先的暴力搜索，显著降低了寻找中心点时的时间；2) 将查询效率与时间复杂度的权衡通过分解定理实现，首次在保持查询效率的同时把时间复杂度降到 (log(1/ε))^{√d log d}；3) 引入“拉动”操作和金字塔结构，对搜索空间进行高效体积计算。

**🔧 技术方法**

ℓ∞-中心点理论、金字塔分解、二分搜索拉动操作、体积计算（利用超平面排列和凸多面体体积计算）、收缩函数到非扩张函数的归约、分解定理以及潜能函数迭代分析。

**📊 数据集**

该工作为理论算法，未使用任何实际数据集；所有结果均为理论上限。

**📈 对比分析**

与之前的最佳算法 (log^{⌈d/2⌉}(1/ε) 运行时间、d log(1/ε) 查询) 进行对比，提出的算法在时间上接近最优且在查询与时间之间取得更优权衡；在 Shapley 随机博弈上实现了 |G|^{√n log n} 的确定性时间改进，优于现有指数级别的上界。

**⚠️ 局限性**

1) 时间复杂度仍随维度 d 指数增长，适用于中等维度；2) 体积计算步骤的实现复杂，可能导致大常数和高位运算量；3) 对于 λ 接近 1 的情况，仍需先归约到非扩张函数，可能引入额外的近似误差；4) 由于对 ℓ∞-中心点的迭代搜索，实际实现的效率和稳定性需要进一步实验验证。

---

## 439. Query-Conditioned Evidential Keyframe Sampling for MLLM-Based Long-Form Video Understanding

**arXiv ID:** 2604.01002 | [PDF](https://arxiv.org/pdf/2604.01002v1)

**作者:** Yiheng Wang `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**通讯引用:** 26207 | [OpenAlex ID](https://openalex.org/A5058073627)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于信息瓶颈的证据驱动关键帧采样框架，能够在长视频中挑选最能回答查询的帧。

**💡 创新点**

创新点在于将关键帧选择建模为最大化条件互信息，证明该目标是单调子模的，利用其模上界将组合优化分解为独立帧级评分，并通过对比学习训练查询条件证据评分网络，实现高效无RL的采样。

**🔧 技术方法**

核心技术包括信息瓶颈理论、子模函数近似、查询条件的视觉-语言编码器、因果窗口注意力聚合、门控机制和对比（InfoNCE）损失。

**📊 数据集**

实验数据集包括Seek‑173K LLaVA‑Video 子集（用于训练），LVBench（长视频 QA）和 Video‑MME（多模态长视频 QA）等。

**📈 对比分析**

与均匀采样、AKS、FOCUS、TSPO 等基线比较，方法在 Qwen2.5‑VL‑7B 上 LVBench 精确率提升约10.1%，在 Video‑MME 上平均准确率提升至 63.6%，在多子任务（实体识别、关键信息检索、时序定位）均优于现有采样方案。

**⚠️ 局限性**

局限性包括对推理和摘要子任务的提升有限，方法依赖已标注的证据区间，且在极长视频或极稀疏证据场景下，子模近似和局部窗口聚合可能无法捕获全局上下文。

---

## 440. EmbedPart: Embedding-Driven Graph Partitioning for Scalable Graph Neural Network Training

**arXiv ID:** 2604.01000 | [PDF](https://arxiv.org/pdf/2604.01000v1)

**作者:** Nikolai Merkel `[一作]` (Technical University of Munich), Hans-Arno Jacobsen `[通讯]` (University of Toronto)

**通讯引用:** 10628 | [OpenAlex ID](https://openalex.org/A5072791865)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于节点嵌入的图划分方法，利用GNN训练产生的嵌入向量进行聚类，再进行轻量级平衡，显著加速了分布式GNN训练的划分过程。

**💡 创新点**

创新点在于把图划分问题从不规则的拓扑结构转移到密集的嵌入空间，避免了昂贵的预处理，支持动态图、快速重新划分，并可直接用于单机训练的图重排。

**🔧 技术方法**

技术手段包括：使用GraphSAGE/GAT等GNN模型训练并提取节点嵌入；采用FAISS实现高效KMeans聚类；轻量级迁移平衡策略；模块化架构可替换聚类、平衡等组件。

**📊 数据集**

使用的基准图数据集包括Open Graph Benchmark (OGB) 的几大图（如 ogbl‑citation2、ogbl‑papers100M 等）和 DGL 提供的图，节点数从十几万到一亿级别。

**📈 对比分析**

与传统的基于结构的划分器（如 Metis、KaHIP、流式划分器）以及随机划分进行对比。实验显示：划分时间比基线快 127–155 倍；边切割率相比随机下降 70%+；分布式 GNN 训练加速 2.5×；单机重排在 CPU/ GPU 上分别获得 1.2×/2.1× 的训练速度提升。

**⚠️ 局限性**

局限性包括：需要先完成至少几轮 GNN 训练才能获得高质量嵌入；在图结构极端不规则或节点特征缺失时嵌入质量可能不佳；轻量级迁移可能引入额外边切割；对超大规模图的聚类仍需高显存；实验主要在 OGB/DGL 图上，未覆盖更广泛的应用场景。

---

## 441. Rapid mixing in positively weighted restricted Boltzmann machines

**arXiv ID:** 2604.00963 | [PDF](https://arxiv.org/pdf/2604.00963v1)

**作者:** Weiming Feng `[一作]` (University of Hong Kong), Minji Yang `[通讯]` (University of Hong Kong)

**通讯引用:** 729 | [OpenAlex ID](https://openalex.org/A5101972265)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在任意无向图上，证明了退火两态系统（铁磁两点系统）的Glauber动力学在满足 λ<λ_0 时混合时间为 O(n^{1+o(1)})，从而突破了先前的 O(n^2) 上界。

**💡 创新点**

创新点在于提出了“典型情形下的聚合强空间混合（ASSM）”的概念，并构造了局部邻域与边界条件，使得在任意图上实现多尺度 ASSM，从而实现快速混合的直接证明；该方法与传统的树上递归分析相结合，但不需要图的最大度数或正则结构限制。

**🔧 技术方法**

主要技术包括自可约性、潜能函数与树递归、SAW 树的构造与裁剪、单调耦合、以及新的典型案例 ASSM 证明与多尺度块动力学分析。

**📊 数据集**

无实际数据集，全部为理论证明与理论分析。

**📈 对比分析**

与之前的 Glauber 动力学混合时间 O(n^2) 及基于自回归法的 2^n 复杂度相比，得到的 O(n^{1+o(1)}) 混合时间显著提升，证明了在更宽广的参数范围内可以实现近线性时间的近似抽样。

**⚠️ 局限性**

限制在于对 λ 的上界 λ<λ_0 的要求，以及 β≤1<γ 的设定；在 λ 接近 λ_0 或 β>1 的情形下，方法不再适用，且对更一般的相互作用图结构或多值状态的推广仍存在挑战。

---

## 442. Dual Optimal: Make Your LLM Peer-like with Dignity

**arXiv ID:** 2604.00979 | [PDF](https://arxiv.org/pdf/2604.00979v1)

**作者:** Xiangqi Wang `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 12787 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“Dignified Peer”框架，旨在解决LLM中的“Evasive Servant”失效模式，构建具备尊严与同行属性的助手；

**💡 创新点**

创新点包括①PersonaKnob：首个多维度、分层偏好数据集；②Lag-DPO：容错多目标Lagrangian优化算法，动态平衡四个人格维度；③基于Many‑Facet Rasch模型的IRT评估协议，实现偏见校正的Peer与Dignity评分；

**🔧 技术方法**

技术涵盖：多目标DPO与Lagrangian松弛、偏好学习、深度双向多任务训练、IRT多因素校准、对抗式人类/LLM双重审核；

**📊 数据集**

使用PersonaKnob（A、T、E、C四维度的合成负样本）以及多源锚定数据集（TruthfulQA、Empathetic Dialogues等）进行训练和评估；

**📈 对比分析**

通过与SFT‑Anchor、SFT‑Combined、Multi‑Neg DPO、PCGrad‑DPO、SACPO、SafeRLHF、MODPO等基线在Llama‑3‑8B和Qwen‑3‑4B上对比，Lag‑DPO在所有四维度的IRT对数尺度得分均显著提升，且保持或超过基线的通用推理性能；

**⚠️ 局限性**

局限性包括：需要预先设计的四维度偏好结构，可能不易扩展到更多属性；ε和η_λ超参数仍需人工调优，缺乏自适应调度；在更大规模模型（>8B）上的可扩展性尚未验证；

---

## 443. YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction

**arXiv ID:** 2604.00940 | [PDF](https://arxiv.org/pdf/2604.00940v1)

**作者:** Miro Miranda `[一作]` (RPTU Kaiserslautern-Landau), Andreas Dengel `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

发布了YieldSAT多模态、高分辨率（10 m）作物产量像素级数据集，并在其上开展大规模深度学习基准实验

**💡 创新点**

创新点在于首次公开覆盖4国4作物9年的10 m像素产量图，结合域感知深度集成方法显著缓解分布偏移

**🔧 技术方法**

采用多种深度模型（LSTM、3D‑LSTM、ConvLSTM、3D‑ConvLSTM、Transformer、Diffusion）以及多模态融合和深度集成技术

**📊 数据集**

使用YieldSAT数据集（2173字段，12.2 M像素）包含Sentinel‑2光谱、气象、土壤、地形等72维特征

**📈 对比分析**

通过10折交叉验证、留年（LOYO）和留区（LORO）评估，RMSE和R²指标显示3D‑LSTM+DE在R²上从0.58提升至0.84（留年）或0.76（留区），优于基线

**⚠️ 局限性**

局限包括严重的分布偏移导致模型泛化受限、数据分布不均（作物、地区差异大）、缺乏全球覆盖、现有模型对不同区域/作物的迁移能力有限

---

## 444. GPT-NL Public Corpus: A Permissively Licensed, Dutch-First Dataset for LLM Pre-training

**arXiv ID:** 2604.00920 | [PDF](https://arxiv.org/pdf/2604.00920v1)

**作者:** Jesse van Oort `[一作]`, Saskia Lensink `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了面向荷兰语的大规模公开语料库 GPT‑NL Public Corpus，并针对版权合规、可用性与偏见风险进行系统筛选与评估。

**💡 创新点**

创新点在于将许可合规性（仅收集 CC‑0、CC‑BY、公共领域数据）、多语言可用性、人工风险分析与自动化筛选工具（C5、LLM 重写、OCR/HTR）结合，形成完整的端到端语料生成与审核流程。

**🔧 技术方法**

采用了爬虫与手工标注、OCR/HTR 自动化数字化、LLM（如 Phi‑4）重写、C5 许可提取、数据评估流水线（归一化、语言检测、启发式过滤）以及风险分析与上采样策略。

**📊 数据集**

数据来源包括政府与司法档案（Openraadsinformatie、Tweede Kamer、Officiële bekendmakingen、de Rechtspraak 等）、公共档案（Koninklijke Bibliotheek、Utrechts Archief 等）、Common Corpus（CC‑Eurovoc、CC‑OpenAlex、CC‑English‑PD 等）、Wikidata、YouTube‑Commons、C5 web‑crawl、Openstate 定向抓取等。

**📈 对比分析**

评估方法基于人工抽样评估、风险分析与上采样权重，结果表明最终语料库包含约 36 亿个荷兰语 token（总共 48 亿 tokens），覆盖多领域且满足许可与偏见过滤要求。

**⚠️ 局限性**

主要限制包括：未包含版权较复杂的资源（如 Wikipedia）、缺乏对最新信息的覆盖、未对跨集合进行 dedupe、可能存在未完全删除的个人信息，以及对低资源语言的覆盖仍有限。

---

## 445. Paper Reconstruction Evaluation: Evaluating Presentation and Hallucination in AI-written Papers

**arXiv ID:** 2604.01128 | [PDF](https://arxiv.org/pdf/2604.01128v1)

**作者:** Atsuyuki Miyai `[一作]` (University of Tokyo), Kiyoharu Aizawa `[通讯]` (University of Tokyo)

**通讯引用:** 8825 | [OpenAlex ID](https://openalex.org/A5069982192)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Paper Reconstruction Evaluation（PaperRecon）框架，通过让编码代理在仅有压缩摘要、表格、图像等极简信息的情况下重建完整论文，从而评估 AI 论文写作能力。

**💡 创新点**

首个系统化的 AI 写作评估框架，将写作质量拆分为呈现质量和幻觉两维；同时创建了覆盖 2025 年后 51 篇顶级会议论文的基准集 PaperWrite-Bench。

**🔧 技术方法**

利用压缩摘要、简化图表、表格 LaTeX 源码等最小化输入，采用 rubric 评估呈现质量，并通过 agentic 评估与原论文对比检测幻觉；在实验中使用 Claude Code、Codex 等多种模型。

**📊 数据集**

基准集包含来自 NeurIPS、ICLR、CVPR、ICCV、ACL、ACMMM 等顶级会议的 51 篇 2025 年后发表的论文，覆盖多领域。

**📈 对比分析**

实验对比表明，Claude Code 在呈现质量上优于 Codex，但平均每篇论文产生 10+ 幻觉；Codex 幻觉更少（约 3 次），但呈现质量较低；随着模型升级，写作能力均有所提升。

**⚠️ 局限性**

局限性包括：重建过程仅基于极简信息，可能缺失深层上下文；评估仅衡量与原论文的相似度，无法评估创作性和原创性；幻觉检测方法仍依赖人工判断，可能漏检；框架对不同学科写作风格的适应性尚待验证。

---

## 446. Reconsidering Dependency Networks from an Information Geometry Perspective

**arXiv ID:** 2604.01117 | [PDF](https://arxiv.org/pdf/2604.01117v1)

**作者:** Kazuya Takabatake `[一作]` (Independent Researcher), Shotaro Akaho `[通讯]` (Institute of Statistical Mathematics)

**通讯引用:** 3243 | [OpenAlex ID](https://openalex.org/A5080238107)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文从信息几何角度重新审视了依赖网络（dependency network），通过把伪 Gibbs 采样视为对完整条件流形的 m‑投影，提出了完整条件散度（full conditional divergence）及其上界（FC‑limit），并基于此构建了结构与参数学习的成本函数，最终证明在样本量趋于无穷时学习得到的模型分布收敛到真实分布；

**💡 创新点**

创新点包括：1) 将伪 Gibbs 采样解释为迭代 m‑投影；2) 引入完整条件散度并推导其与 KL 散度的关系及上界；3) 用该散度作为学习成本函数，实现了可分解的结构/参数学习；4) 给出一致收敛证明；5) 设计基于决策图的贪婪结构学习算法；

**🔧 技术方法**

主要技术：信息几何（m‑投影、e‑flat、m‑flat）、贝叶斯推断中的伪似然、Bregman 散度、MDL 复杂度惩罚、决策树/图构造算法、伪 Gibbs 采样与条件伪 Gibbs 采样；

**📊 数据集**

使用了两类数据集：1) Ising4x3 模型的两组样本（1k 与 100k 条）；2) 12 结点 21 边的贝叶斯网络样本（1k 与 100k 条）；

**📈 对比分析**

通过对比完整条件散度与其上界的实验，验证了上界在实践中的紧致性；此外实验没有与其他模型做性能比较，而是聚焦于证明散度上界的有效性与收敛性质；

**⚠️ 局限性**

主要局限：1) 模型分布没有闭式表达，需通过 MCMC 采样近似；2) 条件伪 Gibbs 采样为每个查询生成不同的条件分布，缺乏全局一致的联合分布；3) 对低概率事件的推断可能不准确；4) 对顺序扫描的伪 Gibbs 采样缺乏严格的收敛证明。

---

## 447. ProTPS: Prototype-Guided Text Prompt Selection for Continual Learning

**arXiv ID:** 2604.01116 | [PDF](https://arxiv.org/pdf/2604.01116v1)

**作者:** Jie Mei `[一作]` (University of Washington), Jenq-Neng Hwang `[通讯]` (University of Washington)

**通讯引用:** 12752 | [OpenAlex ID](https://openalex.org/A5101702810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于冻结CLIP模型的持续学习框架ProTPS，用可学习的视觉原型指导文本提示的选择与学习，旨在减轻灾难性遗忘；

**💡 创新点**

创新点在于：①使用视觉原型来引导文本提示选择，促使不同类别的文本提示在语义空间中保持独特；②引入“prompt‑prototype对比损失”与采样式文本分类器以强化提示与原型的互补；③构建了真实海洋物种长尾数据集Marine112，满足类域双增学习场景；

**🔧 技术方法**

主要技术包括：冻结CLIP图像/文本编码器、可学习文本提示、视觉原型初始化与微调、双分类器（视觉+文本）以及对比损失；

**📊 数据集**

实验使用ImageNet100、CIFAR100以及自采的Marine112；

**📈 对比分析**

在CI、CDC和CDI三种持续学习设定下，ProTPS均超过或逼近上限；例如ImageNet100 CI下+10.6%提升，CIFAR100 CI下+1.9%，跨数据集I2C下+2.1%，在Marine112 CDI下表现最优；

**⚠️ 局限性**

局限性包括：依赖CLIP预训练模型，无法直接处理不兼容的文本/图像编码；对视觉原型的冻结可能限制后续任务的迁移；以及未探讨大规模数据下的计算与存储开销。

---

## 448. A Framework for Coalgebraic Reward-Sensitive Bisimulation (Extended Version)

**arXiv ID:** 2604.01103 | [PDF](https://arxiv.org/pdf/2604.01103v1)

**作者:** Pedro H. Azevedo de Amorim `[一作]` (University of Bath), Koko Muroya `[通讯]` (Ochanomizu University)

**通讯引用:** 109 | [OpenAlex ID](https://openalex.org/A5076471067)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一套统一的框架，用来对齐并比较奖励敏感的有界（graded）与无界（ungraded）Bisimulation；通过在纤维化与余纤维化的笼合构造，给出了它们之间的形式化关系，并证明了在此构造下的安全性与可判定性；在该框架下重新表述并归纳了多种已有的奖励/度量 Bisimulation（如 amortised bisimulation、MDP 的奖励敏感模拟、LMP 的 Lévy‑Prokhorov bisimulation 等）。

**💡 创新点**

创新点在于：① 将纤维化和笼合（gluing）技术首次应用于 coalgebraic Bisimulation 的有界/无界分层建模；② 通过构造通用的“Doctrine for Refined Simulations”与“Glued Fibration”，实现了不同层次 Bisimulation 的统一、抽象化与可比较性；③ 在该框架下给出了抽象的 Soundness Theorem，证明有界 Bisimulation 的并集仍为无界 Bisimulation，且对最终 coalgebra 具有完备性。

**🔧 技术方法**

使用的技术主要包括：纤维化 (fibration)、纤维化升华 (liftings)、笼合 (categorical gluing)、单子与伴随、拉伸与极限 (completeness, final coalgebras)、以及与度量结构相关的量子代数（quantale）与拉普拉斯（Lévy‑Prokhorov）上升。

**📊 数据集**

本工作为理论研究，未使用具体实验数据集；所有结论均在抽象代数与类别论层面证明。

**📈 对比分析**

方法比较：通过示例 (LTS、MDP、LMP 等) 说明框架可统一多种 Bisimulation 定义，并通过抽象定理给出它们之间的包含关系；未给出数值性能指标，因研究属于理论推导。

**⚠️ 局限性**

局限性：① 仅在有限或连续可测的系统上保证最终 coalgebra 的存在性；② 证明的安全性在无限状态空间时可能不成立；③ 对于某些度量化 Bisimulation，逆向包含关系不一定成立，需额外假设（如有限状态、闭球等）。

---

## 449. Escaping Flatland: A Placement Flow for Enabling 3D FPGAs

**arXiv ID:** 2604.01078 | [PDF](https://arxiv.org/pdf/2604.01078v1)

**作者:** Cong Hao `[一作]` (Georgia Tech), Ismael Youssef `[通讯]` (Georgia Tech)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套完整的3D FPGA放置流程，包括层分配初始化、动态成本调度、精细延迟估算与扩展的移动集合。

**💡 创新点**

创新点在于：①使用基于分区的层分配作为全局初始解；②引入自适应时间/线长权重和垂直延迟缩放系数；③改进3D延迟查找表并加入层交换移动操作。

**🔧 技术方法**

核心技术包括VTR模拟退火放置器、TritonPart超图分区器、三维延迟查找表、动态权重调度（θ(w)、ζ(w)）和扩展的3D移动算子。

**📊 数据集**

使用Koios综合电路库作为评估数据集，覆盖多种3D架构（CB、CB-O、CB-I、SB及其混合变体）。

**📈 对比分析**

与VTR 3D/2D基线对比，几乎所有指标均提升：关键路径延迟平均下降3.3–10.6%，线长平均下降0.86–5.5%，路由可达性（min_CW）平均降低约3%，在最优配置下可达18% CPD/10% WL的显著改善。

**⚠️ 局限性**

局限性：仅支持两层堆叠，未考虑热/功耗模型，改进主要基于VTR框架，进一步的多层扩展与路由优化仍待实现。

---

## 450. PHASOR: Anatomy- and Phase-Consistent Volumetric Diffusion for CT Virtual Contrast Enhancement

**arXiv ID:** 2604.01053 | [PDF](https://arxiv.org/pdf/2604.01053v1)

**作者:** Zilong Li `[一作]` (Shanghai Key Lab of Intelligent Information Processing School of Computer Science and Artificial Intelligence Fudan University), Hongming Shan `[通讯]` (Institute of Science and Technology for Brain Inspired Intelligence and MOE Frontiers Center for Brain Science Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于视频扩散模型的三维CT虚拟对比增强框架，能够在不使用对比剂的情况下合成高保真CECT图像。

**💡 创新点**

创新点包括：①将CT体积视为视频序列，利用DiT实现全局空间一致性；②引入解剖导向的混合专家网络（ARMoE），按器官划分增强模式；③设计强度-相位感知对齐（IPARA），通过权重化和教师网络对齐消除空间失配的干扰。

**🔧 技术方法**

主要技术：视频扩散Transformer（DiT）、三维变分自编码器（VAE）压缩、解剖路由混合专家（ARMoE）和强度-相位对齐（IPARA）模块、教师模型特征对齐、Flow Matching 训练。

**📊 数据集**

使用三个公开/内部数据集：VinDr‑Multiphase（NC、Arterial、Portal）、WAW‑TACE（多相），以及大规模腹部CT数据集（900病例）进行训练与评估。

**📈 对比分析**

与GAN（3D‑CycleGAN、3D‑Pix2Pix）、MAFormer、ALDM、WAN、SMILE等方法比较；在所有指标（PSNR、SSIM、Dice、Recall、Precision、FID）上均显著优于基线，尤其在增强精度和三维连贯性方面表现突出，跨数据集泛化能力强。

**⚠️ 局限性**

局限性：依赖精确的器官分割；对非刚性位移的残留误差仍可能导致合成错误；视频扩散模型训练与推理成本高，需要更多算力与时间。

---

## 451. Secure Network Function Computation for General Target and Security Functions

**arXiv ID:** 2604.01051 | [PDF](https://arxiv.org/pdf/2604.01051v1)

**作者:** Qin Zhou `[一作]` (Nankai University), Fang-Wei Fu `[通讯]` (Nankai University)

**通讯引用:** 3182 | [OpenAlex ID](https://openalex.org/A5063946169)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了安全网络功能计算的通用模型，给出了针对任意网络、任意目标函数与安全函数以及任意安全级别的非平凡上界，并在两种特定场景（向量线性目标、身份安全函数；以及两者均为向量线性）中进一步简化上界、给出高效线性时间计算算法，提出了两种线性安全网络编码构造，并对构造所需的有限域大小与安全计算容量下界进行了分析。

**💡 创新点**

创新点在于：①将安全网络功能计算推广到任意目标与安全函数，给出通用信息理论上界；②利用格论与秩分析，将复杂上界化简并实现线性时间算法；③设计了针对向量线性目标与安全函数的两种可构造线性安全编码，推导其最小域大小上界，并由此给出非平凡的安全计算容量下界；④通过实例验证上界与构造的有效性。

**🔧 技术方法**

主要技术包括信息理论安全分析、割集与Menger定理、格与偏序理论、线性网络编码与矩阵秩分析、最小割与主割的算法实现。

**📊 数据集**

论文为理论研究，未使用具体数据集。

**📈 对比分析**

作者将所给上界与已知的特殊情形下界对比，证明在目标为加法或身份安全函数时能退化为已有结果；在示例网络中构造的编码达到上界，表明算法与构造在性能上可达最优；算法复杂度线性于网络边数。

**⚠️ 局限性**

局限性包括：上界在一般情况下可能不紧凑；域大小的充分条件虽给出但并非必要，实际所需域可能更小；构造的线性编码并非对所有网络最优，可能需要更大域或更高码率；未考虑更广义的安全准则或随机密钥资源的最小化等进一步研究方向。

---

## 452. A global dataset of continuous urban dashcam driving

**arXiv ID:** 2604.01044 | [PDF](https://arxiv.org/pdf/2604.01044v1)

**作者:** Md Shadab Alam `[一作]` (Eindhoven University of Technology), Pavlo Bazilinskyy `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1439 | [OpenAlex ID](https://openalex.org/A5068886540)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了大规模的 CROWD 数据集，包含 51,753 条 5 分钟的前视城市行车 dashcam 视频片段，覆盖 238 个国家、7,103 个地点，手工标注了昼夜时间和车辆类型，并提供 YOLOv11x 检测和 BoT‑SORT 跟踪的注释文件。

**💡 创新点**

首创了面向日常城市行驶的跨域、地理多样化 dashcam 数据集，系统剔除事故与编辑片段，保留连续、自然驾驶场景；同时提供统一的检测与跟踪基准，显著降低现有数据集的域偏差。

**🔧 技术方法**

技术手段包括：YouTube 手工筛选与分段、YOLOv11x（基于 MS‑COCO）目标检测、BoT‑SORT 多目标跟踪，并通过自定义脚本完成视频下载、裁剪与注释生成。

**📊 数据集**

数据来源主要是公开的 YouTube dashcam 视频，经过手工筛选后形成 CROWD 数据集；使用的检测/跟踪模型来自公开的 YOLOv11x 与 BoT‑SORT 框架。

**📈 对比分析**

文中未开展模型性能对比，提供的检测/跟踪文件仅作为基线；未来研究可利用该数据集在多任务、跨域鲁棒性与轨迹预测等方面进行方法评估。

**⚠️ 局限性**

局限性包括：昼夜比例不平衡（大部分为白天）、地理覆盖仍偏向欧美地区、检测与跟踪受遮挡、光照、天气等因素影响、仅涵盖 MS‑COCO 类别，缺少车道、交通标志等信息，且依赖 YouTube，视频可能随时失效。

---

## 453. ONE-SHOT: Compositional Human-Environment Video Synthesis via Spatial-Decoupled Motion Injection and Hybrid Context Integration

**arXiv ID:** 2604.01043 | [PDF](https://arxiv.org/pdf/2604.01043v1)

**作者:** Fengyuan Yang `[一作]` (National University of Singapore), Angela Yao `[通讯]` (National University of Singapore)

**通讯引用:** 4839 | [OpenAlex ID](https://openalex.org/A5006278133)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一种参数高效、可组合的人机环境视频生成框架 ONE-SHOT，能够独立控制主体身份、动作、环境几何与相机轨迹，并支持长时序生成。

**💡 创新点**

核心创新在于：①将人体运动与环境条件解耦，采用canonical‑space运动注入与跨域动态 RoPE；②引入混合上下文整合（静态身份参考 + 动态记忆）以保持分钟级长时序一致性；③使用 LoRA 仅微调极少参数，避免过度约束。

**🔧 技术方法**

技术手段包括：基于预训练视频基础模型（Wan2.1）+ ControlNet 侧分支；canonical‑space SMPL‑X 运动映射与跨域动态 RoPE；Hybrid Context Integration；多任务掩码训练策略；流匹配训练目标。

**📊 数据集**

训练数据来自多源混合：EMDB2、MotionX（人类动作子集）、ARKitScenes（点云）以及自采网络视频，总约 50k 条 5–20 秒长片；测试集使用 Traj100 与自采交叉组合集。

**📈 对比分析**

与 WanAnimate、RealisDance、Uni3C、RealisMotion 等基线进行对比；在自重建任务中，ONE-SHOT 在 FID/FVD、动作平滑度、背景一致性上均优于对手；在交叉组合任务中亦取得最佳 FID/FVD 与较高的 Motion Smoothness 与 Background Consistency。

**⚠️ 局限性**

局限性包括：依赖点云与相机轨迹质量，噪声或稀疏点云会影响场景一致性；人类估计误差会传递至生成；在极端 bbox‑grounding 或极长视频中仍可能出现运动漂移或身份漂移。

---

## 454. Fast and Accurate Probing of In-Training LLMs' Downstream Performances

**arXiv ID:** 2604.01025 | [PDF](https://arxiv.org/pdf/2604.01025v1)

**作者:** Zhichen Liu `[一作]` (Southern University of Science and Technology), Yang Xu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 22921 | [OpenAlex ID](https://openalex.org/A5100779940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级探针框架，利用LLM内部表征在训练过程中实时预测下游任务的 Pass@1 成功率，从而替代高成本的生成式评估。

**💡 创新点**

创新点在于将下游性能估计视为值函数逼近问题，设计了 Submodel 与 LoRA 两种探针结构，并证明 Submodel 能在不同训练阶段实现跨检查点的前向迁移，显著提升评估效率。

**🔧 技术方法**

采用内部隐藏层表征、LoRA 低秩调参、回归训练、AUROC 与 MSE 评估指标，核心技术为轻量化探针网络与价值函数逼近。

**📊 数据集**

使用了多领域基准：MMLU（知识）、GSM8K/MATH/AIME/BBH（推理）、HumanEval/MBPP（代码）等，在 OLMo‑3‑7B 的多个训练检查点上进行实验。

**📈 对比分析**

与传统基于损失拟合和线性探针的基线相比，Submodel 探针在平均 AUROC 上提升至 0.789，MSE 降至 0.105，且在后续检查点保持 0.75 左右的性能；评估速度提升 15.6×（后期可达 231.8×）。

**⚠️ 局限性**

局限性包括 LoRA 探针对权重漂移敏感，且探针仍需在一定检查点上进行训练；对极端分布变迁或不同模型体系结构的泛化需进一步验证。

---

## 455. CARE: Privacy-Compliant Agentic Reasoning with Evidence Discordance

**arXiv ID:** 2604.01113 | [PDF](https://arxiv.org/pdf/2604.01113v1)

**作者:** Haochen Liu `[一作]` (University Of Cambridge), Ye Yuan `[通讯]` (McGill University)

**通讯引用:** 14024 | [OpenAlex ID](https://openalex.org/A5016488397)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了针对ICU病情加重预测中主观与客观证据冲突的MIMIC-DOS数据集，并提出CARE框架实现隐私合规的多阶段代理推理。

**💡 创新点**

创新点在于引入证据冲突场景、设计分阶段隐私合规代理推理框架，利用私有LLM提供结构化指导而不泄露敏感数据。

**🔧 技术方法**

采用多阶段LLM推理（本地LLM+私有LLM），包括rubric生成、证据获取、状态转换和最终决策，以及多种基线比较。

**📊 数据集**

使用公开的MIMIC-IV v3.1数据集，筛选出主观无痛且MAP低于65mmHg的时刻形成MIMIC-DOS。

**📈 对比分析**

与单传递、投票、辩论等基线相比，CARE在零样本下实现了更高的平衡准确率（0.546）和真正率/真负率均>0.5，表现优于其他方法。

**⚠️ 局限性**

局限在于只关注一种证据冲突类型，数据集规模有限，未考虑更复杂的隐私约束和更广泛的临床场景。

---

## 456. LightGuard: Transparent WiFi Security via Physical-Layer LiFi Key Bootstrapping

**arXiv ID:** 2604.01092 | [PDF](https://arxiv.org/pdf/2604.01092v1)

**作者:** Shiqi Xu `[一作]` (Chinese University of Hong Kong), Soung Chang Liew `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10568 | [OpenAlex ID](https://openalex.org/A5019164720)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出LightGuard架构，将WiFi密钥协商迁移至光学LiFi链路，实现无射频密钥泄露的安全WiFi连接

**💡 创新点**

利用LiFi的物理线束束缚将密钥协商完全离开RF介质；通过双链路同步机制在不影响数据传输的情况下实现密钥注入

**🔧 技术方法**

WiFi IEEE 802.11 NIC、定制LiFi光学转发器、Linux 802.11子系统改造、nl80211驱动API、四阶段同步协议

**📊 数据集**

无公开数据集，实验使用两台Ubuntu mini-PC、Intel AX200 WiFi卡和自研LiFi天线

**📈 对比分析**

对比传统WPA2 4‑Way握手：在对齐状态下，WiFi吞吐量约80 Mbps、时延≈1 ms；在光学对齐失效时握手失败，需重新对齐后恢复；重钥成功率随光束角度偏移显著下降，超过±25°即出现失效

**⚠️ 局限性**

对齐角度要求严格，光链路易受遮挡或角度误差影响；当前仅支持单点对准，需手动或机械对齐；可扩展性受限于LiFi天线范围与布线成本

---

## 457. A Hierarchical Importance-Guided Multi-objective Evolutionary Framework for Deep Neural Network Pruning

**arXiv ID:** 2604.01076 | [PDF](https://arxiv.org/pdf/2604.01076v1)

**作者:** Zak Khan `[一作]`, Azam Asilian Bidgoli `[通讯]` (Wilfrid Laurier University)

**通讯引用:** 343 | [OpenAlex ID](https://openalex.org/A5017611735)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出双阶段重要性引导的多目标CNN剪枝框架，先用连续阈值进行全局搜索，再用二进制进化局部细化。

**💡 创新点**

创新点在于将连续阈值全局探索与基于重要性采样的二进制进化相结合，显著提升Pareto前沿密度与压缩率。

**🔧 技术方法**

采用多目标进化算法（NSGA‑II、MOEA/D）、重要性引导采样、连续阈值剪枝以及二进制掩码精细搜索技术。

**📊 数据集**

使用CIFAR‑10和CIFAR‑100数据集，实验网络包括ResNet‑18、50、56、101、110、152等多种深度网络。

**📈 对比分析**

与多种基准剪枝方法（贪心、深度剪枝、WIEA等）对比，压缩率可达35‑70%，准确率损失≤1‑2%，并通过超体积（HV）提升验证了方法改进。

**⚠️ 局限性**

局限在于仅实现无结构权重剪枝，难以在常规硬件上直接加速；二进制编码规模大、运算成本高，且缺乏结构化可解释性。

---

## 458. Narrative Fingerprints: Multi-Scale Author Identification via Novelty Curve Dynamics

**arXiv ID:** 2604.01073 | [PDF](https://arxiv.org/pdf/2604.01073v1)

**作者:** Fred Zimmerman `[一作]` (Nimble Books LLC), Hilmar AI `[通讯]` (RKHS Multiverses Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用句子嵌入计算段落间的语义新颖性曲线，结合符号聚合逼近（SAX）和动机提取，对两大语料库中的书籍进行多尺度（整本书与章节/场景）分析，验证作者在信息理论层面留下可识别的“指纹”。

**💡 创新点**

①首次将信息理论的新颖性曲线与SAX动机分析相结合，发现作者指纹在宏观尺度由标量动态决定、在微观尺度由SAX动机决定，呈现多尺度特性；②在跨时代（现代与19世纪）验证指纹不受出版体裁约束；③揭示体裁混淆与作者多变性对指纹的影响。

**🔧 技术方法**

句子变换器嵌入（Nomic Embed Text v1.5）、余弦距离新颖性计算、PAA与SAX离散化、k-gram动机向量、Jensen‑Shannon Divergence、最近邻分类、Fisher判别分析、k‑means聚类。

**📊 数据集**

Books3（52,796 现代书籍，759 作者）和 PG‑19（28,439 经典书籍，1,821 作者）两大公开语料库，均按段落切分并预先计算嵌入和新颖性曲线。

**📈 对比分析**

采用作者内一致性（leave‑one‑out JSD）和最近邻分类评估指纹。宏观尺度标量特征可显著识别 43.3% 作者，top‑1 精度 3.8%（约 29× 胜过随机）；微观尺度 SAXB 动机在 20 段窗口时达到 30× 提升；分辨率提升与 k‑gram 增大进一步提升检测率；在同体裁内，7–25% 作者仍保持显著指纹。

**⚠️ 局限性**

①体裁混淆难以完全剔除，尤其在公式化体裁下指纹显著性下降；②嵌入模型对不同时代文本可能产生偏差；③指纹信号相对弱，最高 top‑1 仅约 2.7%（约 30× 胜过随机），不足以替代传统词频等特征；④集体笔名、幽灵写作等因素可能导致误判；⑤语料偏向英语西方文学，泛化能力受限。

---

## 459. SynDe: Syndrome-guided Decoding of Raw Nanopore Reads

**arXiv ID:** 2604.01054 | [PDF](https://arxiv.org/pdf/2604.01054v1)

**作者:** Anisha Banerjee `[一作]` (Technical University of Munich), Alexandre Graell i Amat `[通讯]` (Chalmers University of Technology)

**通讯引用:** 2960 | [OpenAlex ID](https://openalex.org/A5014522913)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了两种算法：PrimerSearch 用于在原始纳米孔读段中快速定位引物；SynDe 在定位后直接在同一信号上进行基于syndrome trellis的低复杂度基因编码解码。

**💡 创新点**

创新点在于：①直接在原始信号上完成引物定位，避免完整basecalling；②利用syndrome trellis实现与任意线性纠错码的集成，解码复杂度与码记忆无关且仅与码长线性；③结合marker符号实现同步与插入/缺失错误校正。

**🔧 技术方法**

核心技术包括：EDHMM/CTC 神经网络产生的概率矩阵；改进的 beam search；syndrome trellis 表示的线性码；PrimerSearch 的多起点 beam 扩展与聚合；卷积码与 marker 码的组合编码。

**📊 数据集**

主要使用了 Volkel 等人提供的 R9.4.1 化学品原始 FAST5 数据集（1250–5000 bp 的定制寡核苷酸），并在补充材料中评估了 Lau 等人数据集。

**📈 对比分析**

与 Chandak 等人、Guppy、Bonito 的完整 basecalling + 解码方案对比，PrimerSearch 在 50 个样本误差内达 78.9–97% 的一致率；SynDe 在 FER 方面比传统方法低约 3 倍，且计算时间显著缩短；置信分数可有效筛选错误输出。

**⚠️ 局限性**

局限性包括：目前主要针对卷积码实现，虽支持任意线性码但受存储/计算限制；引物定位对起始样本间距和子采样因子敏感；未处理反向互补读段；对高插入/缺失错误的鲁棒性仍需进一步提升。

---

## 460. VibeGuard: A Security Gate Framework for AI-Generated Code

**arXiv ID:** 2604.01052 | [PDF](https://arxiv.org/pdf/2604.01052v1)

**作者:** Ying Xie `[一作]` (Kennesaw State University), Ying Xie `[通讯]` (Kennesaw State University)

**通讯引用:** 2911 | [OpenAlex ID](https://openalex.org/A5033829087)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了VibeGuard，一款在发布前执行的安全网关，用于检测AI生成代码项目中的五类操作性漏洞；

**💡 创新点**

创新点在于：①针对AI生成项目的五类特定安全盲点制定分类体系；②构建模块化的静态扫描器组合与策略引擎，实现自动化门控；③在合成实验中实现100%召回率、89.47%精确率，并在所有项目上做出正确的通过/拒绝决策；

**🔧 技术方法**

采用多模块静态分析技术，包括ArtifactScanner、ConfigScanner、SourceMapScanner、SecretScanner、DependencyScanner，并配合策略引擎对扫描结果进行聚合与门控；

**📊 数据集**

使用了八个合成的npm/Python项目作为测试数据集，其中七个植入已知漏洞（按V1~V5分类），一个为干净控制项目；

**📈 对比分析**

实验对三种预设策略（默认、严格、宽松）进行扫描，记录TP、FP、FN等指标，得到召回率100%、精确率89.47%、F1 94.44%，并在所有项目上实现100%的门控正确率；

**⚠️ 局限性**

局限性包括：①仅使用合成项目，真实项目可能更复杂；②凭正则表达式和占位符判断的secret扫描器可能漏检或误报；③仅为静态扫描，无法发现运行时漏洞；④仅覆盖npm与pip生态，未考虑其他打包系统；⑤未区分AI生成与人工编写的文件，无法对AI生成文件实施更严格策略。

---

## 461. Foundation Model-guided Iteratively Prompting and Pseudo-Labeling for Partially Labeled Medical Image Segmentation

**arXiv ID:** 2604.01038 | [PDF](https://arxiv.org/pdf/2604.01038v1)

**作者:** Qiaochu Zhao `[一作]`, Yading Yuan `[通讯]` (Columbia University)

**通讯引用:** 3530 | [OpenAlex ID](https://openalex.org/A5087485509)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出IPnP框架，利用专家网络与冻结的通用模型迭代生成并细化伪标签，从部分标注中实现全器官分割。

**💡 创新点**

基于交互式提示的通用-专家协作生成伪标签，并通过迭代改进和体素级选择损失抑制伪标签噪声。

**🔧 技术方法**

使用交互式提示（bounding box）、伪标签生成与精炼、体素级选择损失、nnUNet作为专家网络、nnInteractive作为通用模型。

**📊 数据集**

实验基于AMOS腹部多器官数据集（模拟67%/33%部分标注）和210例头颈癌临床数据集。

**📈 对比分析**

与全标注基线、全标注下的部分标注、部分标注损失、TransDoDNet进行对比；在AMOS上67%/33%分别达到88.1/83.1 DSC、10.53/25.32 mm HD95，接近全标注；在头颈数据集对少量标注器官表现显著优于基线。

**⚠️ 局限性**

局限包括对通用模型的依赖、极少标注器官仍受限、未系统评估多种提示策略和更广泛基线。

---

## 462. Infinite-Horizon Ergodic Control via Kernel Mean Embeddings

**arXiv ID:** 2604.01023 | [PDF](https://arxiv.org/pdf/2604.01023v1)

**作者:** Christian Hughes `[一作]`, Ian Abraham `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于核均值嵌入（KME）的无限时域全局覆盖控制方法，能在非欧氏域上实现长时任务的比例覆盖；

**💡 创新点**

创新点在于通过递归更新的KME误差状态，将访视历史编码为固定维度的嵌入，消除了对完整历史记录的需求，实现了O(1)的每步计算复杂度；

**🔧 技术方法**

采用了最大均方差（MMD）度量、核均值嵌入、递归梯度下降控制、以及可回溯（MPC）框架；

**📊 数据集**

实验使用了斯坦福乌龟模型（Bunny）和龙模型（Dragon）三维网格数据集，并在二维平面多模分布上进行验证；

**📈 对比分析**

与传统EMMD、全历史、短期记忆、采样记忆、TSP与NBV等基线对比，显示在覆盖率、收敛速度和计算量上显著优于记忆依赖方法；

**⚠️ 局限性**

局限性包括对核函数和长度尺度敏感、对非平稳环境适应性待验证，以及在极高分辨率网格下仍需大量采样点导致的计算与精度权衡。

---

## 463. A comparison of Markov Chain Monte Carlo algorithms for Bayesian inference of constitutive models

**arXiv ID:** 2604.01121 | [PDF](https://arxiv.org/pdf/2604.01121v1)

**作者:** Aricia Rinkens `[一作]` (Eindhoven University of Technology), Clemens Verhoosel `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 8604 | [OpenAlex ID](https://openalex.org/A5000097351)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对热传导系统和粘性流动系统的贝叶斯参数估计，比较了Metropolis–Hastings、Affine Invariant Stretch Move、No‑U‑Turn Sampler 三种 MCMC 采样器的收敛与效率。

**💡 创新点**

创新点是将 Kullback–Leibler 散度与传统启发式指标（Gelman–Rubin、有效样本量）关联，提供更客观的收敛度量，并系统评估梯度信息对采样效率的影响。

**🔧 技术方法**

采用 Metropolis–Hastings、Affine Invariant Stretch Move（AISM）和 No‑U‑Turn Sampler（NUTS）三种 MCMC 算法，并用自动微分求梯度。

**📊 数据集**

使用两个定制实验的数据集：① 通过热敏电阻测量石蜡柱温度的热传导实验；② 通过相机捕捉压缩流体的半解析模型实验，分别含 40 和 10 条观测数据。

**📈 对比分析**

比较方法包括：KL 散度、Gelman–Rubin 诊断、有效样本量和模型评估次数；结果显示 NUTS 在样本量固定时 KL 低、ESS 高，但需更多模型评估；在热传导系统 NUTS 计算量过大而不划算；在粘性流动系统 AISM 较 MH 更有效，NUTS 的优势因梯度计算效率提升而显现。

**⚠️ 局限性**

局限性包括：KL 散度需要参考解，参考解的逼近误差影响结果；仅测试两类低维问题，难以推广到更高维或更复杂模型；梯度评估成本差异导致比较受实现细节影响；并未考虑并行化、采样器参数优化等实际使用中的细节。

---

## 464. Lightweight Prompt-Guided CLIP Adaptation for Monocular Depth Estimation

**arXiv ID:** 2604.01118 | [PDF](https://arxiv.org/pdf/2604.01118v1)

**作者:** Reyhaneh Ahani Manghotay `[一作]` (Simon Fraser University), Jie Liang `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 459 | [OpenAlex ID](https://openalex.org/A5100411312)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 MoA-DepthCLIP 框架，利用轻量化 Mixture-of-Adapters 对 CLIP 视觉编码器进行参数高效适配，联合分类-回归头完成单目深度估计。

**💡 创新点**

将 MoA 模块与 VLM 视觉编码器相结合，实现空间感知的低参数适配；引入全局场景语义上下文向量；采用 128 级深度分箱与复合损失融合分类与回归。

**🔧 技术方法**

Mixture-of-Adapters (MoA) 轻量适配；ViT-B/32 预训练 CLIP backbone；双头深度预测（分类+回归）；全局文本上下文向量；复合损失（交叉熵+L1+SILog）。

**📊 数据集**

NYU Depth V2 室内 RGB‑D 数据集。

**📈 对比分析**

与复现的 DepthCLIP 基线及 ResNet‑50 版本对比，MoA-DepthCLIP 在 δ1 由 0.390 提升至 0.745，RMSE 从 1.176 降至 0.520，且仅使用少量可训练参数。

**⚠️ 局限性**

主要局限在于仅在室内场景上验证，固定 128 级分箱可能在不同场景下需要调优，且在 δ3 上略有下降；缺乏在线动态提示选择等进一步提升。

---

## 465. Discretization-optimized Bayesian model calibration for nonlinear constitutive modeling in heat conduction

**arXiv ID:** 2604.01101 | [PDF](https://arxiv.org/pdf/2604.01101v1)

**作者:** Rodrigo L. S. Silva `[一作]` (Eindhoven University of Technology), Erik Quarghebeur `[通讯]` (Eindhoven University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该研究提出了一种协同优化数值离散化和模型复杂度的贝叶斯框架，用于热传导问题中的温度依赖热导率反演。

**💡 创新点**

创新点在于结合Morozov失配准则和信息准则的自适应算法，同时将梯度优化与MCMC联合使用，实现误差控制下的自动离散化和模型复杂度选择。

**🔧 技术方法**

采用梯度下降/牛顿-信赖域优化求MAP估计，利用MCMC（自适应Metropolis-Hastings）进行不确定性量化，并使用BIC/DIC进行模型选择。

**📊 数据集**

使用合成数据（已知热导曲线）和真实实验数据（在PLA制成的导热柱上记录的温度传感器数据）。

**📈 对比分析**

通过与合成真值对比验证准确性，实验结果表明算法在保持误差与测量噪声相当的前提下，显著降低了网格细化和模型过拟合，计算量比全贝叶斯求证显著减少。

**⚠️ 局限性**

局限包括仅考虑独立高斯测量误差、仅在一维热传导场景下验证、对多物理和三维问题缺乏扩展，以及对更复杂误差结构的适应性不足。

---

## 466. Approximating Pareto Frontiers in Stochastic Multi-Objective Optimization via Hashing and Randomization

**arXiv ID:** 2604.01098 | [PDF](https://arxiv.org/pdf/2604.01098v1)

**作者:** Jinzhao Li `[一作]` (Purdue University), Yexiang Xue `[通讯]` (Purdue University)

**通讯引用:** 1869 | [OpenAlex ID](https://openalex.org/A5060838579)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于哈希与随机化的SAT或acular方法，用于在不确定环境下近似求解多目标随机优化（SMOO）的帕累托前沿。

**💡 创新点**

首次将哈希计数与SAT查询相结合，构造可提供多目标近似保证的概率SAT oracle，并在#P‑难问题上实现可控的常数因子逼近。

**🔧 技术方法**

使用XOR基哈希模型计数、概率SAT oracle、格点离散化、多目标多维乘法网格，以及对加权计数的离散化与放大技巧。

**📊 数据集**

两个真实场景数据集：基于OpenStreetMap与Meteostat的道路网络调度数据，以及来自TSPLIB的物流网络实例（Burma7、Burma14、Ulysses16）。

**📈 对比分析**

与NSGA‑II、NSGA‑II、RVEA、C‑TAEA、SMS‑EMOA等主流演化算法进行对比，使用GD、IGD、HV、SP指标；实验显示本文方法在收敛性、覆盖度、体积以及均匀性上均优于基线，尤其在目标难度增大时差距更明显。

**⚠️ 局限性**

对SAT oracle的可靠性依赖于随机化与错误概率控制，且实现成本随变量数与目标维数呈指数增长；对极大规模实例或高维目标的实用性尚未验证。

---

## 467. Temporal Dependencies in In-Context Learning: The Role of Induction Heads

**arXiv ID:** 2604.01094 | [PDF](https://arxiv.org/pdf/2604.01094v1)

**作者:** Anooshka Bajaj `[一作]` (Indiana University Bloomington), Zoran Tiganj `[通讯]` (Indiana University Bloomington)

**通讯引用:** 1153 | [OpenAlex ID](https://openalex.org/A5021423914)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究大型语言模型在上下文学习中的时间依赖性，发现其在序列回忆任务中表现出类似人类“+1 lag”序列回忆的偏好，并通过消融实验证明这一现象与模型中的诱导头（induction heads）密切相关。

**💡 创新点**

创新点在于首次将认知科学中的自由回忆和序列回忆范式引入LLM，系统量化LLM对时间位移的概率偏好，并揭示诱导头是实现此类时间上下文检索的关键机制。

**🔧 技术方法**

技术方法包括：①基于自由回忆的实验设计（随机排列500个词，最后一个词重复第250个词）；②对4个7-9B参数的开源模型（Llama、Mistral、Qwen、Gemma）进行诱导头与随机头的消融；③利用TransformerLens库实现头消融；④设计少量样本序列回忆任务评估消融对功能性能的影响。

**📊 数据集**

使用的数据集为500个常见英文单词的随机排列，生成5000个不同顺序的实验序列；序列回忆任务采用10个长度为14的字符列表（不重复），在10-shot提示下评估模型的顺序回忆准确率。

**📈 对比分析**

对比方法包括：在消融前后比较不同lag下的token概率分布，绘制+1 lag峰值变化曲线；在序列回忆任务中对比诱导头消融与随机头消融对lag+1回忆概率的影响。结果显示，诱导头消融显著削弱+1 lag偏好，并导致序列回忆性能大幅下降（例如Llama-Instruct lag+1概率从0.98降至0.28），而随机头消融对性能影响相对较小。

**⚠️ 局限性**

局限性包括：①仅使用了500个高频词，未验证其他词表的普适性；②实验聚焦于上下文学习的序列回忆任务，未覆盖其他ICL任务；③所用模型规模为7-9B参数，可能与更大规模模型的行为不同。

---

## 468. POLARIS: PHY-Aware Spectrum Steering for Dynamic Spectrum Sharing

**arXiv ID:** 2604.01087 | [PDF](https://arxiv.org/pdf/2604.01087v1)

**作者:** Stavros Dimou `[一作]` (Northeastern University), Guevara Noubir `[通讯]` (Northeastern University)

**通讯引用:** 3548 | [OpenAlex ID](https://openalex.org/A5008859536)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对3GPP规范下的多种UE频谱转移机制（BWP重配置、CA、EN‑DC、HO、R&R）进行PHY层级的实测与拆解，构建基于O‑RAN NearRT‑RIC的“最小干扰”转移决策框架。

**💡 创新点**

首次实现了端到端PHY层级的延迟分解（内在PHY执行 vs RRC‑>PHY协调）并提出双参数干扰评分（λ, μ），使转移策略能够兼顾平均延迟与极端尾部波动；将该方法嵌入O‑RAN闭环控制，实现场景感知、动态优化。

**🔧 技术方法**

主要技术包括：COTS UE诊断日志提取（ML1/LL1/L2），RRC‑>PHY时间拆分，统计分布与分位数分析，O‑RAN E2接口与xApp实现，NearRT‑RIC决策引擎，以及对延迟指标的多维度归一化与评分。

**📊 数据集**

使用 1,600 次真实网络中的转移执行记录（共约5,500个时间戳的PHY里程碑），采集自 12 个美国城市的商业LTE/NR 网络，设备包括 Google Pixel 5、LG Velvet 5G、OnePlus 8 5G 等。

**📈 对比分析**

与传统固定策略（always‑BWP、always‑HO、min‑mean、min‑T95）对比，所提系统在五种部署场景下平均延迟下降至 85% 左右，95% 分位数下降 89%，并彻底消除 50 ms 以上的尾部超时；对 BWP、HO、CA 等机制的干扰成本给出了可量化比较。

**⚠️ 局限性**

局限性：实验覆盖的 UE 设备和城市场景有限，未包含极端宏站/小区密度、网络拥塞或极端高频段；对 RRC‑>PHY 细节的模型假设可能在不同硬件/厂商中有所差异；系统在大规模多站协同中的可扩展性与实时性仍需进一步验证。

---

## 469. ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction

**arXiv ID:** 2604.01081 | [PDF](https://arxiv.org/pdf/2604.01081v1)

**作者:** Yuheng Zhang `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**通讯引用:** 5330 | [OpenAlex ID](https://openalex.org/A5027010844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 ProOOD，一种基于体素原型的轻量级方法，联合改进3D占据预测和异常检测。

**💡 创新点**

创新点包括：①原型引导的语义填充（PGSI）实现遮挡区域语义一致性；②原型引导的尾类挖掘（PGTM）增强稀有类别表达；③EchoOOD原型融合评分机制，通过局部/全局原型一致性和logit一致性实现无监督的体素级异常检测。

**🔧 技术方法**

采用原型学习、EMA原型更新、对比损失、logit一致性、局部/全局原型匹配等技术，在主干不变的情况下提升特征质量。

**📊 数据集**

在SemanticKITTI、SSCBench-KITTI-360、VAA-KITTI、VAA-KITTI-360、VAA-STU等五个公开基准上进行评估，涵盖常规占据预测与OOD检测两类任务。

**📈 对比分析**

与SGN、VoxDet等主流框架结合，ProOOD在SemanticKITTI上实现整体mIoU提升3.57%、尾类mIoU提升24.80%；在VAA-KITTI上AuPRC_r提升19.34点，整体表现均达到或超越当前最优方案。

**⚠️ 局限性**

局限性：方法仍依赖高质量原型初始化，且对超大规模数据的实时推理性能尚未完全验证；实验多集中于合成或有限真实异常，进一步推广到更复杂场景仍需探索。

---

## 470. Integer-State Dynamics of Quantized Spiking Neural Networks for Efficient Hardware Acceleration

**arXiv ID:** 2604.01042 | [PDF](https://arxiv.org/pdf/2604.01042v1)

**作者:** Lei Zhang `[一作]` `[通讯]` (University of Regina), Lei Zhang (University of Regina)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

本文提出了将量化SNN视为有限整数状态动力学系统的框架，并通过离散动力学分析与轻量级仿真展示不同位宽下的周期性与收敛行为。

**💡 创新点**

创新点在于把量化误差与可裁剪/溢出等有限整数约束视为可调的动力学参数，揭示了位宽对网络吸引子、周期长度和活动模式的根本影响。

**🔧 技术方法**

采用整数状态更新规则（如基于位移的泄漏、阈值截断）、离散时间映射分析以及自定义的伪排名/循环长度指标，对随机网络进行数值实验。

**📊 数据集**

使用随机生成的稀疏整数权重网络作为实验数据集（无真实任务数据），在不同网络规模、连接密度和位宽范围内进行全网扫描。

**📈 对比分析**

通过比较各位宽下的平均发射率、活跃神经元比例、伪排名和经验周期长度，发现从3~4位开始网络从静默跃迁至活跃，并在4/8/16位下保持稳定的周期性活动，表明量化并未显著削弱动力学特性。

**⚠️ 局限性**

局限性包括仅针对无重置的无符号整数模型、缺乏对不同重置/符号策略的系统研究、未给出正式的周期长度分布理论，也未在实际FPGA/ASIC上验证硬件资源与能耗收益。

---

## 471. Diff3R: Feed-forward 3D Gaussian Splatting with Uncertainty-aware Differentiable Optimization

**arXiv ID:** 2604.01030 | [PDF](https://arxiv.org/pdf/2604.01030v1)

**作者:** Yueh-Cheng Liu `[一作]` (Technical University of Munich), Angela Dai `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

Diff3R通过在训练中嵌入可微的3D Gaussian Splatting优化层，使网络学习为后期测试时优化提供更优初始化。

**💡 创新点**

通过双层优化、隐式梯度求导与矩阵无关PCG求解，以及自适应不确定性正则化，突破了传统 feed-forward 与逐场景优化之间的权衡。

**🔧 技术方法**

隐式函数定理、Gauss‑Newton 近似、预条件共轭梯度 (PCG)、可学习的不确定性正则化，以及 DepthSplat/Depth Anything v3 架构。

**📊 数据集**

RealEstate10K、ScanNet++、ScanNet、DL3DV、Hypersim、ArkitScenes 等多数据集。

**📈 对比分析**

与 PixelSplat、MVSplat、DepthSplat、AnySplat 等基线在 Pose‑given/Free 设置下对比，TTO 后 PSNR 提升约 0.25–0.35 dB，显著优于传统方法。

**⚠️ 局限性**

训练时间显著增加（约 4–5 倍），未做加速优化，导致资源消耗高。

---

## 472. Revision or Re-Solving? Decomposing Second-Pass Gains in Multi-LLM Pipelines

**arXiv ID:** 2604.01029 | [PDF](https://arxiv.org/pdf/2604.01029v1)

**作者:** Jingjie Ning `[一作]` (Carnegie Mellon University), Chengyu Yu `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨多LLM修订流水线的真实收益来源，并通过四条件实验拆解其贡献。

**💡 创新点**

提出四条件设计，可将第二遍收益分解为重解、支架和内容三种可加性效应。

**🔧 技术方法**

使用对照实验、McNemar检验等统计方法，对比两组模型在不同任务中的影响。

**📊 数据集**

评估了GPQA Diamond、HLE（多学科MCQ）和LiveCodeBench（编程）三个基准。

**📈 对比分析**

实验显示在MCQ中收益主要来自重解，直接询问强模型往往更优；在编程中收益主要来自支架，弱稿内容甚至有害，且强稿可帮助弱模型。

**⚠️ 局限性**

局限在于仅覆盖两组模型和三类任务，且对低质量稿的消极效应未提供自动化缓解方案。

---

## 473. Multi-Agent LLM Governance for Safe Two-Timescale Reinforcement Learning in SDN-IoT Defense

**arXiv ID:** 2604.01127 | [PDF](https://arxiv.org/pdf/2604.01127v1)

**作者:** Saeid Jamshidi `[一作]` (Polytechnique Montréal), Mohammad Hamdaqa `[通讯]` (Polytechnique Montréal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种自我反思的两时钟控制架构，在 SDN‑IoT 防御中使用快时钟分散化的 PPO 代理进行即时缓解，慢时钟多智能体 LLM 负责审计、生成可结构化的政策更新 ΔΠ，最终实现闭环安全治理。

**💡 创新点**

创新点在于将快速局部学习与慢速全局治理分离，利用 LLM 生成可审计、可回滚的政策演化（不直接修改 RL 参数），并在安全约束、风险敏感的 CVaR 目标下实现控制器感知的闭环稳定；同时提出了多角色 LLM（Critic、Compiler、Red‑Team、Judge）验证流程。

**🔧 技术方法**

核心技术包括：分散式 Proximal Policy Optimization (PPO) 代理、控制器感知的安全约束与动作掩码、风险敏感 CVaR 目标、LLM 多代理治理链（Critic→Compiler→Red‑Team→Judge）生成并验证 ΔΠ、SDN‑IoT 仿真环境（控制器队列、FlowMod 延迟、流表压力建模）。

**📊 数据集**

使用自研 SDN‑IoT 仿真环境生成异构 IoT 流量和定制攻击注入，未使用公开数据集；所有实验基于该模拟平台进行。

**📈 对比分析**

通过与静态阈值、无约束 PPO、受限 PPO 基线在相同仿真条件下对比，评估指标包括 Macro‑F1、RTT p95、控制器积压、FlowMod 强度等。结果显示：相较于无约束 PPO 提升 Macro‑F1 9.1%，相较于阈值法提升 15.4%；代理降级下降 36.8%，控制器积压降低 42.7%，RTT p95 下降不超过 5.8%；治理循环仅需 5 次反射即可将灾难性溢载事件从 11.6% 降至 2.3%。

**⚠️ 局限性**

局限性包括：缺乏形式化最坏情况稳定性证明；治理仅基于汇总统计，可能对罕见故障反应不足；依赖可信控制器与可信计量，未考虑控制器被破坏的情形；LLM 生成的 ΔΠ 仍可能出现偏差；实验仅在仿真环境完成，缺乏真实网络验证。

---

## 474. Trust and Reliance on AI in Education: AI Literacy and Need for Cognition as Moderators

**arXiv ID:** 2604.01114 | [PDF](https://arxiv.org/pdf/2604.01114v1)

**作者:** Griffin Pitts `[一作]` (North Carolina State University), Weedguet Mildort `[通讯]` (University of Florida)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过实验探究本科生在编程问题求解过程中对AI助手的信任与其对AI建议的适当依赖关系，并检验AI素养与认知需求等学习者特征如何调节这一关系。

**💡 创新点**

创新点在于首次揭示信任与适当依赖呈非线性负相关，并发现AI素养和认知需求显著调节该关系，为教育技术设计提供了针对性验证和自我调节的理论依据。

**🔧 技术方法**

技术手段包括Wizard‑of‑Oz实验设计、使用OpenAI GPT‑3.5‑turbo 提供混合正误建议的Python输出预测题、行为数据收集与问卷评估，并采用线性/二次回归与 Johnson‑Neyman 分析。

**📊 数据集**

数据集为432名本科生完成的14道Python输出预测题（8道正确信息、6道误导性建议）以及对应的信任、AI素养、认知需求、编程自我效能、编程素养等问卷数据。

**📈 对比分析**

方法对比采用线性与二次回归模型，二次模型解释方差更高（R²=0.185 vs 0.173）；结果显示信任与适当依赖呈显著负相关；调节模型中AI素养和认知需求的交互显著；整体适当依赖平均61.77%，过度依赖高达86.03%，误判率低。

**⚠️ 局限性**

局限性包括：任务结构受限，未涵盖开放式长任务；行为指标无法捕捉学生思考过程；自报问卷与任务匹配可能不完全；缺乏过程层面数据与多样策略评估。

---

## 475. On the Construction of Recursively Differentiable Quasigroups and an Example of a Recursive $[4,2,3]_{26}$-Code

**arXiv ID:** 2604.01105 | [PDF](https://arxiv.org/pdf/2604.01105v1)

**作者:** Petr Klimov `[一作]` `[通讯]` (Moscow Institute of Physics and Technology), Petr Klimov (Moscow Institute of Physics and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一个递归可微、递归可微阶达到2的26阶准群，完成了q=26时递归MDS码的存在性证明；

**💡 创新点**

首次利用完美循环Mendelsohn设计与指向性标准构造，给出了26阶递归可微准群的显式构造，填补了此前仅缺少q=26情形的空白；

**🔧 技术方法**

运用了完美循环Mendelsohn设计、指向性标准构造、递归导数运算以及递归可微准群的理论框架；

**📊 数据集**

无外部数据集，全部采用理论构造与符号算例；

**📈 对比分析**

与已知的递归可微准群表进行对比，证明递归可微阶达到3，对应MDS码长度≥4，显示构造的有效性；

**⚠️ 局限性**

仅适用于特定阶数，构造方法高度依赖手工设计的Mendelsohn设计，缺乏普适的自动化构造框架；

---

## 476. TRACE: Training-Free Partial Audio Deepfake Detection via Embedding Trajectory Analysis of Speech Foundation Models

**arXiv ID:** 2604.01083 | [PDF](https://arxiv.org/pdf/2604.01083v1)

**作者:** Awais Khan `[一作]` (University of Michigan-Flint), Khalid Malik `[通讯]` (University of Michigan-Flint)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了TRACE，一种训练-free的部分音频deepfake检测框架，通过分析冻结的语音基础模型嵌入序列的第一阶动态来识别拼接边界。

**💡 创新点**

创新点在于利用预训练模型隐含的法医信号——真实语音的嵌入轨迹平滑，而拼接边界产生突变，且无需任何标签或梯度训练。

**🔧 技术方法**

技术包括对语音嵌入进行L2归一化、计算相邻帧的弦距离、提取多种统计量（如RMS、滑窗最大值、角度统计），并用加权线性融合和自动方向校准生成检测分数。

**📊 数据集**

使用了四个部分deepfake基准：PartialSpoof、HAD、ADD 2023 Track 2（两种语言）以及跨域的LlamaPartialSpoof，评估六种不同的语音基础模型。

**📈 对比分析**

在PartialSpoof上，TRACE以8.08% EER与监督模型相当；在LlamaPartialSpoof上甚至优于监督基线（24.12% vs 24.49% EER），在跨语言和跨模型迁移时也保持了较好的性能。

**⚠️ 局限性**

局限性包括：仅针对拼接边界有效，对完全合成语音检测能力弱；以及统计量组合与阈值校准仍需依赖PartialSpoof的验证集，缺乏完全无监督的通用设定。

---

## 477. Containing the Reproducibility Gap: Automated Repository-Level Containerization for Scholarly Jupyter Notebooks

**arXiv ID:** 2604.01072 | [PDF](https://arxiv.org/pdf/2604.01072v1)

**作者:** Sheeba Samuel `[一作]` (Chemnitz University of Technology), Martin Gaedke `[通讯]` (Chemnitz University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套自动化、基于仓库级容器化的 Jupyter Notebook 可复现性评估管线，能够从 GitHub 仓库自动获取代码、推断依赖、构建 Docker 镜像、隔离执行并对比输出，系统性评估大规模学术 notebook 的可复现性。

**💡 创新点**

核心创新在于：① 将依赖推断与容器化集成为仓库级自动化流程；② 通过静态导入分析补全缺失依赖；③ 采用细粒度细胞级日志与输出比对实现可复现性评分；④ 在实际学术数据上进行大规模验证，揭示“可复现性缺口”的本质。

**🔧 技术方法**

技术方案主要包括：Bash/CI orchestrated pipeline、GitHub 仓库爬取、Python import 静态分析、Dockerfile 自动生成与构建、repo2docker 与 Binder 相关实现、nbformat+nbcompare 进行细胞级输出对比、PostgreSQL/SQLite 存储元数据与日志。

**📊 数据集**

使用 116 个 GitHub 仓库中的 443 个 Python Jupyter Notebook，仓库均在 PubMed Central 论文中引用，涵盖多学科、Python 2.7–3.10 版本及多样化依赖环境。

**📈 对比分析**

与原先基于 Conda 的非容器化基线对比：容器化将依赖安装失败率降低 66.7%，整体成功率提升，然而可复现性得分（identical cells / total cells）仍有 53.7% 的 Notebook 在“差”区间；完整性评估显示 114 个 Notebook 存在非确定性操作，导致输出漂移。

**⚠️ 局限性**

主要限制包括：① 仍无法消除外部资源依赖（数据文件、API 密钥、云服务）导致的运行错误；② 只能处理 CPU 环境，GPU、NVIDIA Docker、Singularity 等尚未支持；③ 依赖推断缺少版本信息，导致兼容性风险；④ 仍存在显著的非确定性（随机、时间、UUID 等），导致可复现性评分不理想；⑤ 对 legacy Python 2.7 兼容性有限。

---

## 478. BAT: Balancing Agility and Stability via Online Policy Switching for Long-Horizon Whole-Body Humanoid Control

**arXiv ID:** 2604.01064 | [PDF](https://arxiv.org/pdf/2604.01064v1)

**作者:** Donghoon Baek `[一作]` (Georgia Institute of Technology), Sehoon Ha `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个在线策略切换框架BAT，能够在长时序人形机器人控制中平衡敏捷性与稳定性，自动在耦合和非耦合两种全身控制策略之间切换；

**💡 创新点**

创新点在于：①将滑动窗口策略预评估与层级强化学习结合，提供高质量的切换监督；②设计了面向选项的VQ‑VAE，使离散运动编码与控制器偏好对齐；③构建决策融合模块，利用不同模块的不确定性动态决定是否使用切换策略；

**🔧 技术方法**

核心技术包括层级强化学习（Discrete PPO）、滑动窗口价值引导评估、选项感知VQ‑VAE、行为克隆辅助、熵/KL失衡检测与决策融合；

**📊 数据集**

使用AMASS运动库（含多模态人类运动）进行离线评估与数据采集，随机拼接运动片段构造长时序任务；

**📈 对比分析**

在仿真（IsaacGym、MuJoCo）和真实Unitree G1机器人上与多种基线（耦合/非耦合策略、BC、Heuristic、Oracle等）对比，BAT在成功率、跟踪误差与能耗等指标上均优于所有基线，并在硬件上实现了多阶段长时序任务；

**⚠️ 局限性**

局限性：缺乏对环境感知（如不平地、外部扰动）的显式建模，且当前切换仅限于预设的两种低层策略，无法动态扩展到更多控制器或更复杂环境；

---

## 479. Adversarial Attacks in AI-Driven RAN Slicing: SLA Violations and Recovery

**arXiv ID:** 2604.01049 | [PDF](https://arxiv.org/pdf/2604.01049v1)

**作者:** Deemah H. Tashman `[一作]` (Polytechnique Montreal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montreal)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究基于深度强化学习的 RAN 切片控制在面临预算受限的对抗性干扰时对服务水平协议（SLA）的影响，并提出了一个智能攻击者模型来诱导 SLA 违规与系统恢复行为。

**💡 创新点**

①首次系统量化对抗攻击对不同服务切片 SLA 的长期违规率与恢复过程；②设计了预算约束下的对抗性 jamming 策略，并通过对抗者的 DDQN 近似预测目标控制器的行为；③将 SLA 满足度嵌入奖励函数，从而揭示传统仅关注吞吐量的奖励对 SLA 损害的隐蔽性。

**🔧 技术方法**

双深度 Q 网络（DDQN）用于主控制器和攻击者的策略学习；离散化的切片活动/优先级状态空间；基于滑动窗口的 SLA 指标和奖励函数；对抗性干扰模拟与能量预算约束。

**📊 数据集**

使用仿真环境生成的交通请求概率、RB 需求、优先级权重等参数（eMBB、URLLC、mMTC），未使用公开真实数据集，所有实验均在自定义的 NextG RAN 切片仿真平台上完成。

**📈 对比分析**

与无对抗干扰的基线 DDQN（不考虑 SLA 罚款）和已有方法（未嵌入 SLA）的性能进行对比。结果显示：在对抗攻击下，URLLC 与 mMTC 的 SLA 违规率显著上升，eMBB 影响较小；攻击结束后，系统通过继续学习逐步恢复至基线奖励水平，证明了策略的可恢复性。

**⚠️ 局限性**

受限于仿真实验，未在真实网络中验证；攻击者需要一个近似的 DDQN 代理来预测目标控制器，可能无法完全复制真实行为；预算约束和干扰模型简化了实际无线环境中的多路径与信号衰落等复杂因素。

---

## 480. Automated Framework to Evaluate and Harden LLM System Instructions against Encoding Attacks

**arXiv ID:** 2604.01039 | [PDF](https://arxiv.org/pdf/2604.01039v1)

**作者:** Anubhab Sahu `[一作]` (Keysight Technologies), Reza Soosahabi `[通讯]` (Keysight Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并使用自动化框架，评估LLM在编码或结构化输出请求下对系统指令的泄露情况。

**💡 创新点**

首次系统探讨通过编码/结构化重构攻击导致的泄露，并提出基于Chain‑of‑Thought的指令重塑硬化方案。

**🔧 技术方法**

结合编码探测、Prompt生成器、判别模型以及CoT指令重塑技术。

**📊 数据集**

使用约80条含敏感信息的系统指令，筛选46条在直接查询下拒绝泄露的指令作为实验基准。

**📈 对比分析**

对四大LLM模型进行ASR评估，结构嵌入格式泄露率高达90%+，经过重塑后可显著降至10%以下。

**⚠️ 局限性**

实验仅限单轮直接查询，未覆盖多轮或工具协助攻击；结果受模型对抗鲁棒性与指令复杂度限制。

---

## 481. Stein Variational Uncertainty-Adaptive Model Predictive Control

**arXiv ID:** 2604.01034 | [PDF](https://arxiv.org/pdf/2604.01034v1)

**作者:** Hrishikesh Sathyanarayan `[一作]` (Yale University), Ian Abraham `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出了一种基于Stein变分推断的分布式鲁棒模型预测控制器，能够在存在潜在参数不确定性的非线性系统中通过自适应粒子流实时更新参数后验，从而生成对任务关键不确定性鲁棒的控制序列。

**💡 创新点**

创新点在于：1）将Stein变分推断直接嵌入控制器设计，构造任务敏感的后验分布；2）通过最优性差距（optimality gap）来引导粒子更新，聚焦于对闭环性能影响最大的参数；3）避免传统DRO的全局最坏情况设计，提升性能-鲁棒性折衷。

**🔧 技术方法**

采用Stein变分梯度下降（SVGD）、模型预测控制（MPC）、Lagrangian 约束优化、核函数（RBF/IMQ）、粒子并行更新以及最优性差距作为似然的变分推断框架。

**📊 数据集**

实验数据主要为仿真任务：Cartpole Swing‑up、Rocket Landing 2D、Autonomous Racing。每个任务在32个随机初始参数下运行，采样参数均为均匀分布。

**📈 对比分析**

与三类基线（EMPPI、经典DRO、标准MPC）比较，结果显示该方法在成功率、平均完成时间和方差方面均优于基线；在Cartpole和Rocket任务中成功率均为100%，完成时间约为基线一半；在赛车任务中平均 lap time 约 4.4 秒，波动性最小。

**⚠️ 局限性**

局限性包括：对粒子数、步长、γ 参数的敏感性；核函数选择会影响收敛与多模态保持；在高维参数空间下可能出现粒子聚类或计算量急剧上升；目前仅在仿真环境验证，缺乏真实硬件或复杂多车/多机器人系统的实证。

---

## 482. Sub-metre Lunar DEM Generation and Validation from Chandrayaan-2 OHRC Multi-View Imagery Using Open-Source Photogrammetry

**arXiv ID:** 2604.01032 | [PDF](https://arxiv.org/pdf/2604.01032v1)

**作者:** Aaranay Aadi `[一作]` (Manipal University Jaipur), Oleg Alexandrov `[通讯]` (National Aeronautics and Space Administration Ames Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过基于开源ASP的工作流程，利用Chandrayaan‑2 OHRC多视角影像生成子米分辨率数字高程模型。

**💡 创新点**

创新之处在于为OHRC实现了PDS4导入模板和CSM相机模型，并通过基线/高度比和收敛角自发识别有效立体对，首次完成公开可复现的子米DEMs。

**🔧 技术方法**

使用了ASP、ISIS、ALE、CSM、ICP、QGIS等开源软件和工具进行相机建模、立体匹配、点云生成与DEM网格化。

**📊 数据集**

数据集为Chandrayaan‑2 OHRC的原始PDS4影像以及LRO NAC参考DEMs。

**📈 对比分析**

将生成的DEMs与NAC DTM进行剖面对比和特征匹配，垂直RMSE约5.85 m，水平误差<30 cm，验证表明整体性能达最高分辨率。

**⚠️ 局限性**

局限在于高基线/高度比（>≈0.9）导致大量空洞，需大量NAC填补；立体匹配受光照差异和阴影影响，极端收敛角会显著降低完整性。

---

## 483. Model-Based Learning of Near-Optimal Finite-Window Policies in POMDPs

**arXiv ID:** 2604.01024 | [PDF](https://arxiv.org/pdf/2604.01024v1)

**作者:** Philip Jordan `[一作]` (EPFL), Maryam Kamgarpour `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在表格部分可观察马尔可夫决策过程（POMDPs）中基于模型的有限窗口策略学习。

**💡 创新点**

提出了一种模型估计程序，能够在单个轨迹下紧密保证样本复杂度，并结合价值迭代获得近似最优的有限窗口策略。

**🔧 技术方法**

使用了基于模型的学习算法，结合了价值迭代和样本复杂度分析。

**📊 数据集**

使用了从POMDP中生成的轨迹数据进行模型估计。

**📈 对比分析**

与现有方法相比，提出的方法在样本复杂度上达到了O(ϵ^-2)，而之前的研究通常为O(ϵ^-4)，显示出显著的改进。

**⚠️ 局限性**

模型估计的准确性依赖于对转移和观察核的正则性假设，且在有限窗口近似中可能引入额外的误差。

---

## 484. Asymptotically Optimal Sequential Testing with Heterogeneous LLMs

**arXiv ID:** 2604.01086 | [PDF](https://arxiv.org/pdf/2604.01086v1)

**作者:** Guokai Li `[一作]` (Queen's University), Preet Baxi `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

研究了在多模型LLM异质信息源下的贝叶斯二元序贯假设检验，提出最优策略与通用下界，并证明在高置信度下仅需最多两个LLM即可实现近似最优；

**💡 创新点**

创新点在于（1）将成本、等待时间与非对称准确度统一到多源序贯测试框架；（2）给出极限误差阈值趋近0时的下界与稀疏结构证明；（3）设计基于符号的两源策略并证明其渐近最优；

**🔧 技术方法**

使用了概率论与信息理论（KL信息率、对数似然比、马尔可夫链、子高斯分布）、凸优化（求解信息分配下界）、大数定律与Freedman不等式等技术；

**📊 数据集**

论文主要为理论分析，未给出具体实验数据集；若有实验，可能使用公开LLM性能指标（Intelligence Index、输出速度）等；

**📈 对比分析**

与oracle策略和单LLM策略比较，证明两源策略与下界相差O((log(1/α))^{ρ-1})且上界与下界收敛速度一致，说明在α→0时性能最优；

**⚠️ 局限性**

局限性：仅适用于二元假设、已知模型参数、子高斯延迟假设，未考虑多类别、在线学习、动态成本等情形。

---

## 485. Harnessing Hype to Teach Empirical Thinking: An Experience With AI Coding Assistants

**arXiv ID:** 2604.01110 | [PDF](https://arxiv.org/pdf/2604.01110v1)

**作者:** Marvin Wyrich `[一作]` (Saarland University), Sven Apel `[通讯]` (Saarland University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计并实施了为期一个学期的研讨会，围绕 AI 编码助手展开实践与经验研究，帮助学生培养实证思维。

**💡 创新点**

创新之处在于将热点技术作为教学切入点，既降低抽象概念门槛，又兼顾技术与研究双重目标，证明炒作主题可驱动学生的实证学习。

**🔧 技术方法**

使用 GitHub Copilot 等大型语言模型驱动的编码助手，并在 VS Code 环境中进行交互、实验设计与数据收集。

**📊 数据集**

主要使用学生自行生成的实验数据和测试用例作为研究素材，未采用公开数据集。

**📈 对比分析**

通过对研讨会前后学生问卷结果的定量与定性分析，比较了学生对 AI 编码助手认知、实证思维水平以及对热点主题的兴趣提升，发现参与者在认知与方法论上均有显著提升。

**⚠️ 局限性**

局限性包括样本量小、实验仅为两轮短周期实验、难以涵盖多样化研究方法、教师资源需求高以及热点主题可能导致期望与课程目标不匹配。

---

## 486. Adversarial Moral Stress Testing of Large Language Models

**arXiv ID:** 2604.01108 | [PDF](https://arxiv.org/pdf/2604.01108v1)

**作者:** Saeid Jamshidi `[一作]` (Polytechnique Montréal), Kawser Wazed Nafi `[通讯]` (Polytechnique Montréal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Adversarial Moral Stress Testing（AMST）框架，通过多轮结构化对抗压力变换来评估大型语言模型的伦理鲁棒性。

**💡 创新点**

创新点在于将多轮交互与压力累积引入伦理评估，提出分布感知鲁棒性指标和可控的情绪/伦理压力变换，从而揭示单轮评估难以捕捉的伦理衰退模式。

**🔧 技术方法**

技术手段包括：结构化语义变换算子、分布感知鲁棒性指数、语义风险、毒性、拒绝检测等多维伦理指标、黑盒交互、自动化对抗生成与自回归推理分析。

**📊 数据集**

数据来源为真实伦理困境的基线提示集合，并通过自动化压力变换生成多轮对话样本，构成实验输入。

**📈 对比分析**

实验在GPT‑4o、LLaMA‑3‑8B、DeepSeek‑v3三大模型上进行多轮对抗测试，利用鲁棒性均值、方差、漂移曲线等分布指标进行横向比较，结果显示GPT‑4o最稳健、DeepSeek‑v3最易衰退。

**⚠️ 局限性**

局限性包括：评估仅限于英文西方伦理框架；压力变换与指标为自动化代理，缺乏人类主观伦理判定；未覆盖多模态或多语言场景，可能不足以捕捉所有真实攻击方式。

---

## 487. ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data

**arXiv ID:** 2604.01082 | [PDF](https://arxiv.org/pdf/2604.01082v1)

**作者:** Yaoqin Ye `[一作]` (ShanghaiTech University), Yuexin Ma `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ReMoGen 框架，实现了实时交互到反应的运动生成，通过统一的单人运动先验、Meta‑Interaction 模块以及帧级细化来适配多域交互场景。

**💡 创新点**

创新点在于（1）把大规模单人运动先验冻结下来，利用轻量级 Meta‑Interaction 模块实现跨 HHI、HSI 等不同域的快速适配；（2）引入帧级细化（Frame‑wise Segment Refinement）在保持长时序连贯性的同时实现毫秒级响应。

**🔧 技术方法**

技术包括：冻结的文本条件单人运动先验（基于潜在扩散网络），多源上下文编码器（TCN + ViT），FiLM 风格的跨注意力模块，分段自回归生成和轻量级帧级细化模块。

**📊 数据集**

使用 HumanML3D 训练先验；Inter‑X 训练 HHI Meta‑Interaction 模块；LINGO 训练 HSI 模块；EgoBody 用于混合域的零样本和微调验证。

**📈 对比分析**

与 ReGenNet、SymBridge、FreeMotion、TRUMANS、LINGO 等前沿方法对比，ReMoGen 在 Inter‑X、LINGO 上均取得 FID、R‑Precision、MM‑Dist 等指标更优，同时帧级推理延迟仅 0.042 s，明显低于对比模型。

**⚠️ 局限性**

局限性：需为每个交互域单独训练 Meta‑Interaction 模块，跨域组合仍可能产生不稳定或过度模糊的细节；对极度动态或稀缺多模态输入的鲁棒性尚待进一步提升。

---

## 488. From Validity to Inter-Subjectivity: An Argument for Reliability Signals in Search Environments

**arXiv ID:** 2604.01186 | [PDF](https://arxiv.org/pdf/2604.01186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 489. Automated Generation of Cybersecurity Exercise Scenarios

**arXiv ID:** 2604.01079 | [PDF](https://arxiv.org/pdf/2604.01079v1)

**作者:** Charilaos Skandylas `[一作]` (Linkoping University), Mikael Asplund `[通讯]` (Linkoping University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种自动化生成网络安全训练情景的方法，先根据网络拓扑与架构规范生成可执行环境（子系统与连线），再生成故事线（锁、钥匙、目标、任务），并实现可在模拟器或虚拟化环境中部署。

**💡 创新点**

创新点在于：1）将网络拓扑、子系统架构、约束统一为可解析的文本规范，并通过 Alloy 自动生成合法网络与子系统实例；2）通过锁/钥匙/目标机制与任务（Quest）实现动态、可调节难度的情景；3）提供两种执行后端（模拟器与虚拟化），实现跨平台可部署。

**🔧 技术方法**

使用的技术包括：文本规范语言（Topology、ArchStyle）、Alloy 模型化与求解、图算法（锁/钥匙/目标生成）、任务与能力推理、Python/Java 脚本生成网络与子系统实例、libvirt 虚拟化、仿真游戏循环（两轮制），以及 JSON/YAML 数据交互。

**📊 数据集**

未使用公开数据集；所有网络拓扑与子系统定义均为作者自定义规范，随机化参数用于生成多样化情景。

**📈 对比分析**

本文未给出量化性能或对比实验；仅在方法论层面与现有情景生成语言（如 CRACK、CST‑SDL、CTL、VSDL 等）做概念对比，指出本方法在场景数量与多样性方面优于单一片段重排的 Cyexec^*，但缺乏实验评估。

**⚠️ 局限性**

局限性包括：1）生成的情景依赖手工编写的规范与脚本，手动工作量大；2）缺乏大规模实验与性能评测；3）对复杂约束（如高级防御策略、动态路由）支持有限；4）在真实环境部署时需要额外脚本生成钥匙与目标文件；5）对现有仿真平台的依赖导致跨平台兼容性受限。

---

## 490. Aligning Recommendations with User Popularity Preferences

**arXiv ID:** 2604.01036 | [PDF](https://arxiv.org/pdf/2604.01036v1)

**作者:** Mona Schirmer `[一作]` (University of Amsterdam), Yannik Stein `[通讯]` (Amazon Music)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了推荐系统中的流行度偏差，将其视为用户与推荐器之间的对齐问题，提出了流行度分位校准（Popularity Quantile Calibration）框架，并在此基础上设计了新的量化指标 PCE；随后开发了 SPREE 方法，在推理时通过激活向导（activation steering）根据每位用户的流行度偏好动态调整推荐结果。

**💡 创新点**

创新点包括：①将推荐系统的流行度偏差视为个体对齐问题并以测量理论为依据构建分位校准框架；②提出了基于分位数的 PCE 指标，兼顾方向、幅度与分布宽度；③提出了 SPREE，利用激活向导在推理阶段根据用户偏好自适应调节模型内部表示，实现个性化的流行度偏差校正，而不需重新训练模型。

**🔧 技术方法**

技术手段包括：分位校准与 PCE 的统计量计算；基于 Transformer 的序列推荐模型（SASRec）；通过对高/低流行度样本的对比学习得到流行度方向向量；对该向量进行自适应放缩，利用 Lasso 回归预测用户流行度偏差；在推理时将该向量加到激活上实现 Steering。

**📊 数据集**

使用了四个公开数据集：Foursquare Tokyo、MovieLens‑1M、MovieLens‑20M 和 RateBeer，均经过常规预处理（保留至少 5 次交互的用户/物品）。

**📈 对比分析**

与基线（原始 SASRec、IPR、PP、Random Neighbors、PopSteer）在 PCE@100 与 NDCG@100 进行比较。实验表明，SPREE 在大多数数据集上显著降低 PCE（即提升用户流行度对齐），同时保持甚至略提升 NDCG@100，优于仅降级流行度或对齐历史的基线方法；PP 在保持准确率方面表现差强人意。

**⚠️ 局限性**

局限性包括：仅能在模型嵌入空间存在局部流行度可分离的区域时有效；目前仅对齐中位数附近的流行度分布，无法针对分布尾部进行精细控制；若嵌入空间结构差异大，激活向导效果可能受限。

---

## 491. HippoCamp: Benchmarking Contextual Agents on Personal Computers

**arXiv ID:** 2604.01221 | [PDF](https://arxiv.org/pdf/2604.01221v1)

**作者:** Zhe Yang `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HippoCamp benchmark，模拟三种真实个人文件系统，评估代理在检索、感知和推理方面的表现。

**💡 创新点**

创新点在于：①跨模态、多层级文件结构的真实化；②两类任务（事实保留与画像推理）与多粒度失败诊断标注；③提供可解释的错误分析框架。

**🔧 技术方法**

使用检索增强生成（RAG）、ReAct搜索代理、终端代理等多种技术，并借助 GPT‑4o 等 LLM 进行答案与检索质量评估。

**📊 数据集**

数据集基于三个人类典型文件系统（约 42.4 GB、2000+ 文件），共 581 题 QA 与 46.1 k 细粒度标注。

**📈 对比分析**

与现有 RAG、搜索代理和自主代理对比，最优系统 ChatGPT Agent Mode 在画像任务上仅达 48.3% 准确率，整体仍远低于人类，表明性能差距显著。

**⚠️ 局限性**

局限性包括：检索后处理瓶颈、跨模态感知不足、实体绑定与最终验证缺失，且样本仅覆盖三种典型场景，缺乏更广泛多样性。

---

## 492. Screening Is Enough

**arXiv ID:** 2604.01178 | [PDF](https://arxiv.org/pdf/2604.01178v1)

**作者:** Ken M. Nakanishi `[一作]` `[通讯]` (RIKEN), Ken M. Nakanishi (RIKEN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Multiscreen 语言模型架构，核心创新是 Screening 机制，用于实现绝对查询–键相关性，并取代传统的 Softmax 注意力；同时设计了可学习的窗口、最小位置编码、GLU 风格门控等模块。

**💡 创新点**

创新点：1) 通过 Screening 单元实现每个键的独立阈值判定，消除全局竞争与固定单位质量的重新分配；2) 学习可变屏蔽窗口，动态决定有效上下文范围；3) 引入最小位置编码（MiPE），仅在窗口小于阈值时激活，避免长距离位置外推问题；4) 结合 TanhNorm 进行输出规范化；5) 在训练中不使用权重衰减与梯度裁剪，提高大学习率稳定性。

**🔧 技术方法**

技术实现：Transformer 基础架构、gated screening tiles、Trim‑Square‑Softmask、TanhNorm、MiPE、GLU‑style gating、RoPE 类似旋转、学习窗口参数、AdamW 优化器、无权重衰减、无梯度裁剪、bfloat16 推理、Triton 实现屏蔽模块。

**📊 数据集**

数据集：训练使用 SlimPajama（≈628B tokens）; 长上下文评测使用 PG‑19 书籍数据；检索基准使用自定义 ABCDigits（无语义、固定 26 键的完整键值检索任务）。

**📈 对比分析**

比较方法：在相同 token 预算下与 LLaMA‑style Transformer 基线做规模化实验；做学习率稳定性扫频；对比长上下文困惑度、ABCDigits 检索准确率（不同上下文长度与深度）以及 100K token 推理延迟。性能：Multiscreen 在参数量约 40% 下降的情况下与 Transformer 在验证损失上相当；可使用更大学习率并保持稳定；长上下文 perplexity 与检索准确率在超长序列上保持不降；推理延迟在 100K 长度下比 Transformer 快 2.3–3.2×。

**⚠️ 局限性**

局限性：1) 仍未验证跨语言、跨模态等场景；2) 虽提升检索，但对复杂自然语言推理的影响仍不明确；3) 需要进一步评估大规模并行化与硬件友好性；4) Screening 机制在极大序列上可能出现窗口学习不充分导致的效率下降；5) 仍需探索更细粒度的定位或多层次屏蔽策略。

---

## 493. Reasoning Shift: How Context Silently Shortens LLM Reasoning

**arXiv ID:** 2604.01161 | [PDF](https://arxiv.org/pdf/2604.01161v1)

**作者:** Gleb Rodionov `[一作]` `[通讯]` (Yandex), Gleb Rodionov (Yandex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究不同上下文条件（无关长文本、子任务、多轮对话等）对大型语言模型推理链长度、自我验证与不确定性管理等高级行为的影响，并通过系统实验验证在非基线上下文中模型会压缩推理链且削弱自我检验行为。

**💡 创新点**

发现不相关上下文会显著缩短推理链（最高可缩短50%），导致自我验证和不确定性管理行为下降，从而暴露推理鲁棒性与上下文管理的脆弱性。

**🔧 技术方法**

使用链式推理（CoT）技术、强化学习后训练（SFT/DPO）获取思考模式；采用长上下文窗口、句子级标签化和转移矩阵分析方法对推理链进行细粒度分类与统计。

**📊 数据集**

评估数据集包括 IMOAnswerBench（数学推理）和 MATH-500（数学题集），并使用 Gemini 3 Pro 作为自动评判者。

**📈 对比分析**

通过基线、子任务、长输入、多轮对话四种场景对比，记录准确率和平均推理token数；实验表明在非基线场景下准确率下降（约9–15%），推理长度缩短至基线的一半左右。

**⚠️ 局限性**

局限性包括仅测试合成且简单的上下文扰动、仅关注数学推理任务、主要聚焦单一模型（Qwen3.5‑27B 等），未探究更复杂或真实场景及对策。

---

## 494. ReinDriveGen: Reinforcement Post-Training for Out-of-Distribution Driving Scene Generation

**arXiv ID:** 2604.01129 | [PDF](https://arxiv.org/pdf/2604.01129v1)

**作者:** Hao Zhang `[一作]` (MMLab, CUHK), Hongsheng Li `[通讯]` (MMLab, CUHK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一框架ReinDriveGen，能够在动态3D点云基础上自由编辑车辆轨迹、行人、骑车人等行驶场景，并通过视频扩散模型生成逼真的驾驶视频。

**💡 创新点**

创新点包括：①使用车辆点云完成模块填补未观测表面，消除几何缺陷；②将DiffusionNFT扩展至视频生成并引入pairwise preference + pairwise reward机制，利用强化学习后训练显著提升离训练分布（OOD）场景下车辆质量；③通过点云条件化与视频扩散的结合，实现对任意视角的高质量渲染。

**🔧 技术方法**

技术手段包括：动态多帧LiDAR点云构建、AdaPoinTr点云完成、VACE-1.3B视频扩散模型、DiffusionNFT强化学习后训练、pairwise preference model（基于DINOv3 ViT-H+）、YOLO检测、LoRA微调等。

**📊 数据集**

使用Waymo开放数据集进行训练与评估，并在DriveDreamer4D、PVG、StreetGaussians、StreetCrafter等公开基准上进行对比。

**📈 对比分析**

在Novel Ego-View和车辆轨迹编辑任务中，相比PVG、DriveDreamer4D、ReconDreamer、S3Gaussian、Deformable-GS等方法，ReinDriveGen在NTA-IoU、NTL-IoU和FID等指标上均取得最佳成绩；在车辆编辑任务中，VBench指标（图像质量、背景一致性、运动平滑度）均优于Street Gaussians/StreetCrafter，FID下降约12%。

**⚠️ 局限性**

局限性：受GPU显存限制，只能生成约49帧、每段约一分钟的视频，无法实现实时应用；强化学习后训练仅在有限的OOD场景下验证，未来需扩展至更广泛的测试条件。

---

## 495. Bridging the Simulation-to-Experiment Gap with Generative Models using Adversarial Distribution Alignment

**arXiv ID:** 2604.01169 | [PDF](https://arxiv.org/pdf/2604.01169v1)

**作者:** Kai Nelson `[一作]`, Aditi S. Krishnapriyan `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种数据驱动的分布对齐框架——Adversarial Distribution Alignment（ADA），通过先在完整但近似的模拟数据上预训练生成模型，然后用实验中仅部分观测的数据对其进行对齐，弥合仿真与实验之间的差距。

**💡 创新点**

创新点在于：① 将生成模型的分布与实验观测分布通过可学习的对抗式 Wasserstein 距离约束实现全分布匹配，而非仅匹配低阶统计量；② 通过 KL 正则化保留预训练分布的先验信息；③ 允许处理多个相关观测，理论证明可在 β→∞ 时收敛到目标分布。

**🔧 技术方法**

技术手段包括：生成模型（以扩散模型为主）预训练；对抗式训练实现可学习的观测 Wasserstein 距离；使用梯度惩罚实现 Lipschitz 约束；采用 adjoint matching 进行无偏梯度估计；在优化目标中结合 KL 与 Wasserstein 损失。

**📊 数据集**

数据集覆盖三类：1) 合成高维多峰混合高斯数据；2) 小分子 MD17 数据（低阶半经验 GFN2‑xTB 与高阶 DFT 作为对照）；3) 蛋白质 PDB 数据（Trp‑cage、BBL）通过模拟的 cryo‑EM 图像作为观测。

**📈 对比分析**

与期望对齐（Expectation Alignment）方法进行对比。实验表明 ADA 在合成数据上显著降低 ℓ1 残差与 Wasserstein 距离，在小分子数据上在所有观测上实现更低的 Wasserstein 距离和更小的自由能表面 Jensen‑Shannon 散度；在蛋白质实验数据上，ADA 能将模拟生成的分布对齐到实验结构，显著减少可观测量的 Wasserstein 距离和最大 RMSD。

**⚠️ 局限性**

局限性包括：需要可微分的观测函数；对高维噪声观测的对齐仍受限于样本量与噪声水平；当前仅处理静态观测，扩展到动态观测需要进一步研究；对实验数据的依赖使得在实验数据稀缺或高成本场景下效果受限。

---

## 496. Universal YOCO for Efficient Depth Scaling

**arXiv ID:** 2604.01220 | [PDF](https://arxiv.org/pdf/2604.01220v1)

**作者:** Yutao Sun `[一作]`, Furu Wei `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Universal YOCO 的递归模型，它将 YOCO 的一次性 KV 缓存与在 Self-Decoder 层进行的递归计算相结合，以提升推理效率和模型容量。

**💡 创新点**

创新点在于：① 将递归计算仅限制在浅层 Self-Decoder 上，避免了全层递归导致的 KV 缓存膨胀；② 通过参数共享实现多次迭代，提高表示深度；③ 兼具 YOCO 的线性预填充和常数 KV 缓存优势，实现了计算规模与内存开销的分离。

**🔧 技术方法**

采用 YOCO 解码器-解码器架构、滑窗高效自注意力（Sliding‑Window Attention）、SwiGLU 变换、RMSNorm、NoPE 与 RoPE 位置编码、递归循环、参数共享、Paged Attention 与 Flash‑Decoding 等技术。

**📊 数据集**

在 300 B 训练标记、8192 长度的通用语言建模语料上进行训练，并在多个下游评测集上验证：ARC‑C、Winogrande、HellaSwag、MMLU、BBH、GSM‑8K、Humaneval、DROP、各类数学基准（GSM‑8K、MATH、SVAMP、ASDiv、MAWPS、CARP、TABMWP、Gaokao 2023 En、OlympiadBench、CollegeMath、AMC23），以及书籍/代码长序列困惑度和 Needle‑In‑A‑Haystack 检索任务。

**📈 对比分析**

通过与 Transformer、YOCO、Universal Transformer、RINS、ParScale 等多种基线在相同 FLOPs 或相同标记数量下进行对比，显示 Universal YOCO 在训练损失、下游任务准确率（平均提升约 4.5 分）、长序列困惑度和检索性能上均优于基线；在推理吞吐量上仅损失约 5%，KV 缓存内存几乎不变，且预填充速度保持线性。

**⚠️ 局限性**

局限性包括：① 递归仅作用于浅层 Self‑Decoder，深层递归带来的收益有限；② 随着循环次数增多收益趋于饱和；③ 依赖高效注意力实现，若更换注意力机制需重新调优；④ 在极大模型或极长上下文下的可扩展性尚未全面验证；⑤ 需要专门的推理优化（如 Flash‑Decoding、Paged Attention）才能充分发挥优势。

---

## 497. LAtent Phase Inference from Short time sequences using SHallow REcurrent Decoders (LAPIS-SHRED)

**arXiv ID:** 2604.01216 | [PDF](https://arxiv.org/pdf/2604.01216v1)

**作者:** Yuxuan Bao `[一作]` (University of Washington), J. Nathan Kutz `[通讯]` (Autodesk Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为LAPIS‑SHRED的模块化深度学习框架，用于在仅有极短时间窗口和稀疏空间传感器观测的条件下重构或预测完整的时空动力学。

**💡 创新点**

创新点包括：在稀疏观测下实现前向与后向时空推断；利用预训练的SHRED将时空状态映射到结构化潜在空间；通过序列到序列或自回归潜在动态模型实现完整轨迹的推断；以及对单帧终端观测的静态填充编码方案。

**🔧 技术方法**

采用SHRED架构、LSTM或BiLSTM时序单元、Seq2Seq与自回归潜在模型、静态填充、以及冻结解码器的潜在空间推断技术；同时在多尺度情况下使用多尺度SHRED和轻量级解码器。

**📊 数据集**

在六个实验中使用：二维Kuramoto–Sivashinsky、二维Kolmogorov流、二维Von Karman涡街、三维旋转爆炸引擎（高精度和Koch一维模型）、以及MODIS雪覆盖指数（NDSI）数据。

**📈 对比分析**

与全时序观测的SHRED基线、SHRED‑ROM以及FNO/DeepONet等方法比较，LAPIS‑SHRED在仅占10%–20%时序窗口、3–64个传感器时均能取得NRMSE≤5%，并在多数实验中接近或超过基线；在单帧终端观测下亦保持低误差；与其他方法相比，支持前向/后向推断且误差显著降低。

**⚠️ 局限性**

局限性包括：缺乏不确定性量化、对观测窗口固定、对高度混沌系统的理论分析有限、需预训练SHRED模型且对模拟-现实差距的补偿仍需改进、以及在长时序预测中潜在的误差累积。

---

## 498. CliffSearch: Structured Agentic Co-Evolution over Theory and Code for Scientific Algorithm Discovery

**arXiv ID:** 2604.01210 | [PDF](https://arxiv.org/pdf/2604.01210v1)

**作者:** Youssef Mroueh `[一作]` (IBM Research), David Cox `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CliffSearch，一种利用 LLM 代理进行对比选择、交叉、探索/修复突变以及评审的进化搜索框架，核心目标是科学算法的结构化发现；

**💡 创新点**

创新点包括：①将理论、代码与摘要纳入统一节点并在两种模式（含/不含理论）下运作；②将评审的正确性与原创性设为硬性存活门；③将突变拆分为探索性和修复性两条路径；④以可审计、可重复的运行时实现多岛同步与迁移；

**🔧 技术方法**

使用技术：大语言模型（Claude/ChatGPT）驱动的多代理，JSON 接口、分布式计算、固定任务契约、基准化评估与审核；

**📊 数据集**

数据集：Shakespeare 字符级数据（nanoGPT 训练），若干小型线性/MLP 分类数据集用于本地优化器基准；

**📈 对比分析**

比较方法：在同一基准（验证损失）下与人类种子及已知优化器/注意力结构做对比，使用中位数阈值与评审门限筛选赢家；性能方面，发现了 Givens‑Poincaré 混合注意力、MuOn 等优化器族，平均验证损失显著低于种子和传统 Adam，且通过后置文献审核显示一定程度的原创性；

**⚠️ 局限性**

局限性：缺乏收敛保证；评审信心仅为标量评分；对基准协议高度依赖；原创性判断仍基于 LLM 内部记忆，未结合外部检索或可检索知识库。

---

## 499. TRACE: High-Fidelity 3D Scene Editing via Tangible Reconstruction and Geometry-Aligned Contextual Video Masking

**arXiv ID:** 2604.01207 | [PDF](https://arxiv.org/pdf/2604.01207v1)

**作者:** Jiyuan Hu `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TRACE框架，利用可触摸的3D几何锚点和视频扩散技术实现高保真3D Gaussian Splatting场景的自动化编辑，支持局部几何变形、纹理合成与风格迁移。

**💡 创新点**

创新点在于：① 将显式3D网格作为几何锚点指导视频扩散，兼顾结构完整性与生成灵活性；② 设计了两阶段的Tangible Geometry Alignment (TGA)实现精准对齐；③ 引入Contextual Video Masking (CVM)提升视频连贯性和物理一致性；④ 构建首个多视角一致的MV-TRACE编辑数据集。

**🔧 技术方法**

核心技术包括：3D LoRA微调的多视角锚点合成、两阶段几何注册（粗对齐+微调）、基于遮罩的自回归视频扩散、以及将编辑后的视频回投到3D Gaussian Splatting进行最终重建。

**📊 数据集**

使用了自研的MV-TRACE数据集（约10万对场景一致的物体添加/修改样本），并在IN2N、BlendedMVS、Mip-NeRF 360等公开数据集上进行评估。

**📈 对比分析**

在48个编辑案例、8个场景的实验中，TRACE在CLIP Direction、CLIP Similarity、DINO一致性和美学评分上均优于DGE、GaussianEditor、EditSplat等基线，CLIP_dir提升至0.1514（+49.5%），编辑时间维持在10分钟以内，达成实时级别。

**⚠️ 局限性**

局限性包括：仍依赖于预训练的3D LoRA和网格生成器，对极端拓扑改动或大规模动态场景的适应性有限；对高频细节的纹理优化仍需改进，且在复杂光照/阴影场景下可能出现微小不一致。

---

## 500. Property-Level Flood Risk Assessment Using AI-Enabled Street-View Lowest Floor Elevation Extraction and ML Imputation Across Texas

**arXiv ID:** 2604.01153 | [PDF](https://arxiv.org/pdf/2604.01153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 501. AgentWatcher: A Rule-based Prompt Injection Monitor

**arXiv ID:** 2604.01194 | [PDF](https://arxiv.org/pdf/2604.01194v1)

**作者:** Yanting Wang `[一作]` (Pennsylvania State University), Jinyuan Jia `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种通过上下文归因与基于规则的推理来检测大型语言模型代理中的注入攻击的框架；

**💡 创新点**

创新点在于：①使用注意力权重定位“sink”标记并提取关键短文本，减少对长上下文的依赖；②将检测过程交给监控LLM并结合可自定义规则，提升可解释性与灵活性；

**🔧 技术方法**

技术主要包括：注意力权重归因、滑动窗口扩展、规则驱动的监控LLM（并通过GRPO微调提升推理质量）；

**📊 数据集**

使用的数据集包括：AgentDojo、InjecAgent、WASP等代理基准，以及LongBench中的LCC、GovReport、PassageRetrieval、Qasper、HotpotQA、MultiNews等长上下文数据集；

**📈 对比分析**

与PromptArmor、DataSentinel、PromptGuard、GPT‑OSS‑Safeguard、PIGuard等现有方法对比，AgentWatcher在大多数任务中将攻击成功率降至≤1%（或≤10%）同时保持≤4%的实用性损失；

**⚠️ 局限性**

主要限制为：检测过程需调用LLM，平均耗时约10秒，因而需在高风险操作时才触发；此外，长轨迹或适应性攻击仍可能带来挑战。

---

## 502. Open-Set Supervised 3D Anomaly Detection: An Industrial Dataset and a Generalisable Framework for Unknown Defects

**arXiv ID:** 2604.01171 | [PDF](https://arxiv.org/pdf/2604.01171v1)

**作者:** Hanzhe Liang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Pan Li `[通讯]` (Hangzhou Dianzi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了开放式监督3D异常检测方法 Open3D-AD，能够在仅使用少量已知异常样本的情况下，对工业点云数据中的未知异常进行检测。

**💡 创新点**

创新点包括：① 引入开放式监督任务，利用少量已知异常样本学习异常分布；② 设计“对应分布子采样（Correspondence Distributions Subsampling）”方法，有效去除正常与异常分布重叠的模糊特征；③ 将模拟异常与真实异常联合建模，形成双分布表示，显著提升对未知异常的泛化能力。

**🔧 技术方法**

使用多尺度点云特征提取器（多FPFH与点云Transformer），模拟异常生成（Norm-AS），对特征分布进行贪婪子采样，并通过分布感知距离度量与重要性加权计算异常分数。

**📊 数据集**

主要数据集为新构建的工业点云基准 Open-Industry（15类，含5种真实异常类型），以及公开数据集 Real3D-AD 与 Anomaly-ShapeNet，用于评估跨域鲁棒性。

**📈 对比分析**

与传统开放式异常检测方法（DevNet、DRA）及3D异常检测基线（M3DM、Reg3D-AD）进行对比；在 Open-Industry 上，Open3D-AD 在 5/10 样本/类别设置下分别取得 84.39%/82.74% 的 O-AUROC 与 70.52%/76.29% 的 P-AUROC，明显优于第二名；在 Real3D-AD 与 Anomaly-ShapeNet 上亦保持领先，展示了强大的泛化与鲁棒性。

**⚠️ 局限性**

局限性在于目前仅能在点级别进行异常分类，且对重建式方法的标签信息利用不充分，未来需要进一步探索更好地利用标签信息并提升对复杂异常形态的检测能力。

---

## 503. SMASH: Mastering Scalable Whole-Body Skills for Humanoid Ping-Pong with Egocentric Vision

**arXiv ID:** 2604.01158 | [PDF](https://arxiv.org/pdf/2604.01158v1)

**作者:** Junli Ren `[一作]` (University of Hong Kong), Ping Luo `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套基于自我感知的全身技术体系，实现了室外人形乒乓球机能，包括快速自我感知、任务对齐的全身控制与可扩展的运动生成；

**💡 创新点**

核心创新在于①将基于运动变分自编码器生成的可覆盖整个击球工作空间的姿态库与任务目标对齐的运动匹配策略相结合；②构建了仅依赖机身摄像头的低延迟 egocentric 感知管线；③通过适应性区域采样与动态噪声注入实现对高动态任务的鲁棒学习；

**🔧 技术方法**

采用的技术包括：YOLO+HSV分割的双阶段目标检测；自适应扩展卡尔曼滤波（AEKF）与物理模型预测；运动变分自编码器（Motion‑VAE）生成全身击球序列；强化学习（PPO）结合任务奖励与运动匹配的策略学习；双摄像头姿态估计与AprilTag定位；

**📊 数据集**

使用了400条全身动作捕捉数据（含前手、后手、低蹲、猛烈击球等），通过 Motion‑VAE 生成至5k+ 条合成姿态以覆盖击球空间；

**📈 对比分析**

在仿真与实机对比基线（PPO、Mimic、HITTER）中，所提方法在任务成功率与运动平滑度上均表现最佳；仿真下成功率约86%，实机下成功率达到93.7%（连击）且击球精度误差低；

**⚠️ 局限性**

主要限制包括：1) 受限于机身摄像头视角导致低蹲或激烈击球时感知失效；2) 未对球旋转进行建模，无法处理高旋转或复杂发球；3) 对极端动态视觉噪声和遮挡仍有鲁棒性提升空间。

---

## 504. Deep Reinforcement Learning for Robotic Manipulation under Distribution Shift with Bounded Extremum Seeking

**arXiv ID:** 2604.01142 | [PDF](https://arxiv.org/pdf/2604.01142v1)

**作者:** Shaifalee Saxena `[一作]` (University of New Mexico), Alexander Scheinker `[通讯]` (Los Alamos National Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

结合强化学习与有界极值寻优（ES）的混合控制器，改进机器人在推送和抓取-放置任务中的鲁棒性。

**💡 创新点**

提出在任务进入阶段使用 RL 快速获取接触姿态，接触后立即切换至 ES 在线适应，从而在训练后推理阶段保持高性能，首次在接触富集的操作中实现了两者的互补。

**🔧 技术方法**

使用目标条件 DDPG 训练的 RL 策略、bounded ES 作为后端适应器以及基于接触触发的监督切换，全部实现于 MuJoCo/Fetched 仿真环境。

**📊 数据集**

使用 FetchPush 与 FetchPickAndPlace 的 Fetch 基准数据集进行训练，并在测试时构造分区摩擦系数表面与时变目标的仿真场景。

**📈 对比分析**

通过与仅 RL、仅 ES 两种基线对比，在固定与时变摩擦、三维目标跟踪等任务中，ES‑DRL 在成功率、目标误差和轨迹跟踪精度方面均优于单一方法，特别是在接触后分布漂移时表现突出。

**⚠️ 局限性**

仅实现一次切换，无法在失去接触后自动回退到 RL；未对实时计算成本进行评估；实验仅在仿真环境中验证，缺乏真实硬件验证。

---

## 505. Obfuscating Code Vulnerabilities against Static Analysis in JavaScript Code

**arXiv ID:** 2604.01131 | [PDF](https://arxiv.org/pdf/2604.01131v1)

**作者:** Francesco Pagano `[一作]` (University of Verona), Giorgio Giacinto `[通讯]` (University of Cagliari)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了JavaScript代码混淆对SAST工具检测漏洞能力的影响。

**💡 创新点**

创新点在于提出Vulnerability Detection Loss指标，并通过两阶段实验（基准和真实项目）系统量化单/多重混淆导致的检测损失。

**🔧 技术方法**

采用javascript-obfuscator提供的八种语义保持混淆技术，分别对Njsscan和Bearer两款SAST工具进行扫描。

**📊 数据集**

使用16个OWASP公开的vulnerable-by-design Node.js应用作为基准数据集，及从GitHub收集的260个真实Node.js/JavaScript项目。

**📈 对比分析**

通过比较混淆前后漏洞发现率的Vulnerability Detection Loss百分比，发现单一混淆即可使大多数漏洞消失，堆叠超过5种混淆后损失几乎饱和，表明现有SAST工具对混淆极为脆弱。

**⚠️ 局限性**

限制包括仅使用开源SAST工具和单一混淆器，未考虑动态分析，基线检测差异导致绝对损失评估受限，并且未测试所有混淆组合的顺序与交互效果。

---

## 506. Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction

**arXiv ID:** 2604.01204 | [PDF](https://arxiv.org/pdf/2604.01204v1)

**作者:** Jorge Condor `[一作]` (NVIDIA), Qi Wu `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在原语（如 3D 高斯球）周围构建虚拟四面体，在每个顶点绑定可学习特征向量，并在射线交点处进行双三角插值后施加周期激活，最终通过一次延迟着色直接解码出 RGB。

**💡 创新点**

创新点在于将特征向量绑定到原语自身，利用正弦/余弦周期激活将高频细节融入每个原语的透明度，显著提升单原语表达能力，同时保持可移动、可编辑的局部编码。

**🔧 技术方法**

采用三维高斯原语、虚拟四面体、双三角插值、正弦/余弦周期激活、单步延迟着色、浅层 MLP、半精度训练、EMA、余弦学习率调度、损失缩放等技术。

**📊 数据集**

主要在 MipNeRF360、Tanks & Temples 等常规三维重建数据集上训练，并在高分辨率 14 位 HDR RAW 图像集上测试二维图像重建。

**📈 对比分析**

与 3DGS、3DGUT、Feature‑3DGS、Instant‑NGP 等基准相比，Neural Harmonic Textures 在保持约 140 FPS 的实时渲染速度的同时，在高频细节、视角依赖效果、语义场重建等方面均实现了最优或接近最优的 PSNR/SSIM/LPIPS 结果。

**⚠️ 局限性**

局限性包括在极少视图监督时易过拟合、相较纯原语方法略慢、难以自动提取多层细节以及在低原语数量下仍需平衡精度与速度。

---

## 507. Embarrassingly Simple Self-Distillation Improves Code Generation

**arXiv ID:** 2604.01193 | [PDF](https://arxiv.org/pdf/2604.01193v1)

**作者:** Ruixiang Zhang `[一作]` (Apple), Yizhe Zhang `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于模型自身生成的无验证输出进行自蒸馏，通过简单的温度/截断采样后对模型进行标准交叉熵微调，提升代码生成性能

**💡 创新点**

创新点在于仅使用自身未经过筛选或验证的生成样本进行训练，无需教师模型、奖励模型或执行验证，展示了极简自我提升方法的有效性

**🔧 技术方法**

采用温度调节与截断采样生成数据，随后用Megatron-LM进行标准监督微调；在推理时保持固定的解码温度与截断配置

**📊 数据集**

利用从rSTARcoder数据集中去重得到的约10k个竞赛编程问题作为训练提示；评估基准为LiveCodeBench v6（及v5）

**📈 对比分析**

与基线模型在LiveCodeBench v6进行对比，所有五个模型（Llama‑8B、Qwen3‑4B‑Instruct、Qwen3‑4B‑Thinking、Qwen3‑30B‑Instruct、Qwen3‑30B‑Thinking）均提升，最显著的是Qwen3‑30B‑Instruct从42.4%提升至55.3% pass@1，尤其在中难和难题上增幅最大；相较于仅调节解码参数的最佳基线，self‑distillation在pass@1和pass@5上均表现更好

**⚠️ 局限性**

限制主要在于依赖特定的温度/截断超参且对训练数据域较为依赖；对非编程任务的泛化能力有限，且在极端高温或无截断场景下仍需依赖后期解码截断才能获益

---

## 508. A ROS 2 Wrapper for Florence-2: Multi-Mode Local Vision-Language Inference for Robotic Systems

**arXiv ID:** 2604.01179 | [PDF](https://arxiv.org/pdf/2604.01179v1)

**作者:** J. E. Domínguez-Vidal `[一作]` `[通讯]`, J. E. Domínguez-Vidal

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文实现了一个 ROS 2 包装器，将 Florence‑2 视觉‑语言模型集成到机器人系统中，支持连续主题流、同步服务和异步动作三种交互模式；

**💡 创新点**

创新点在于为 Florence‑2 提供统一的 ROS 2 接口，结合 JSON 与 ROS 原生检测消息，支持本地部署与 Docker 化，同时在多 GPU 上对吞吐量进行评估；

**🔧 技术方法**

技术栈包括 ROS 2、Python、Hugging Face Transformers、PyTorch、OpenCV、CUDA、Docker、ROS 2 服务/动作/主题接口等；

**📊 数据集**

实验使用自制的 rosbag 记录的图像序列进行推理验证，并未引用公开数据集；

**📈 对比分析**

采用同一图像流在不同 GPU（GTX 1060 Mobile、RTX 3060 Mobile、RTX 3080 Ti）上进行连续检测任务，测量 FPS：GTX 1060 约 5.5 FPS，RTX 3060 约 9.2 FPS，RTX 3080 约 25.3 FPS，表明本地消费级硬件即可实现实时推理；

**⚠️ 局限性**

局限性包括：①仅检测结果映射为 ROS 2 消息，其他任务仍返回 JSON；②Action 取消受阻塞生成阶段限制；③实验聚焦于集成性能，未全面评估任务质量；④性能高度依赖硬件。

---

## 509. NeuroDDAF: Neural Dynamic Diffusion-Advection Fields with Evidential Fusion for Air Quality Forecasting

**arXiv ID:** 2604.01175 | [PDF](https://arxiv.org/pdf/2604.01175v1)

**作者:** Prasanjit Dey `[一作]`, Bianca Schoen-Phelan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种Neural Dynamic Diffusion‑Advection Fields（NeuroDDAF）模型，用于在时空图上同时学习扩散和对流的物理约束与神经ODE残差，以实现PM₂.₅等空气质量的高精度预测。

**💡 创新点**

创新点包括：①将扩散-对流算子与神经残差通过可学习的证据融合门自适应组合；②在神经ODE中嵌入时变图卷积与傅里叶域扩散，兼顾物理可解释性与表达能力；③通过Monte‑Carlo ODE集成与证据回归实现联合的不确定性量化。

**🔧 技术方法**

使用技术：时变图卷积（GAT）+傅里叶域扩散+神经ODE（Dormand–Prince）+Monte‑Carlo ODE集成+证据回归（Normal‑Inverse‑Gamma）。

**📊 数据集**

实验基于多站点空气质量与气象观测数据，涵盖至少数十个监测站（N）和24小时历史窗口（T=24）进行3天（τ=24）PM₂.₅预测，具体数据集为公开城市监测网络（如天津/北京等）。

**📈 对比分析**

与传统GRU、GCN、ConvLSTM等基线相比，NeuroDDAF在MAE和RMSE上平均提升约5–10%，并在不同天气/风速场景下保持稳健的性能。

**⚠️ 局限性**

局限性：①对高频细尺度变化仍不够敏感；②模型依赖时变图结构与傅里叶算子，计算开销相对较大；③不确定性估计依赖MC‑ODE集成，需更多轨迹以获得更稳定的置信区间。

---

## 510. FineLAP: Taming Heterogeneous Supervision for Fine-grained Language-Audio Pretraining

**arXiv ID:** 2604.01155 | [PDF](https://arxiv.org/pdf/2604.01155v1)

**作者:** Xiquan Li `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

开发 FineLAP，利用双流 Sigmoid 损失和分离的音频适配器，在 CLAP 框架中实现同时学习 clip 与 frame 级音频-文本对齐，并构建 FineLAP-100k 合成 SED 数据集。

**💡 创新点**

采用异构监督结合双流 Sigmoid 损失与语义聚类负采样实现细粒度与粗粒度对齐；分离的音频适配器同时提取全局与帧级特征；通过大规模合成 SED 数据缓解框架级标注稀缺。

**🔧 技术方法**

使用 EAT 自监督音频编码器、双流 Sigmoid 损失、基于聚类的负采样、分离音频适配器与文本投影，以及合成数据生成 pipeline。

**📊 数据集**

利用公开音频-文本对齐数据（AudioSetCaps、WavCaps、AudioCaps、Clotho）约 2.1M 条；帧级标注数据（AudioSet-Strong、DESED、UrbanSED、FineLAP-100k）约 201k 条；评测集包含 AudioCaps、Clotho、ESC-50、UrbanSound8K、VGGSound、AudioSet-Strong、DESED、UrbanSED、AudioGrounding。

**📈 对比分析**

与现有 CLAP 系列模型（LAION-CLAP、HTSAT-BERT、Cacophony、MGA-CLAP、FLAM、M2D-CLAP）以及 SED 基线（FlexSED、PT-SED）对比，FineLAP 在检索、分类和 SED 任务上均达到 SOTA，检索 R@1 最高 45.7/62.5，SAD 在 AudioSet-Strong 0.474、DESED 0.344、UrbanSED 0.446、TAG 0.649。

**⚠️ 局限性**

仅支持固定 10s 长度输入，无法处理可变/长时音频；未覆盖除 SED 之外的帧级语音-文本任务；合成数据与真实数据差异导致泛化略受限。

---

## 511. VRUD: A Drone Dataset for Complex Vehicle-VRU Interactions within Mixed Traffic

**arXiv ID:** 2604.01134 | [PDF](https://arxiv.org/pdf/2604.01134v1)

**作者:** Ziyu Wang `[一作]` (Jilin University), Yuxin Zhang `[通讯]` (Jilin University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了基于无人机拍摄的高分辨率城市混合交通数据集VRUD，重点记录了VRU（行人、自行车、电动车等）与车辆的交互轨迹，并构建了4,002个多主体交互场景库。

**💡 创新点**

① 以“城市村落”无结构高VRU密度环境为采集目标，填补了现有数据集对这类长尾场景的空白；② 采用向量时间碰撞（VTTC）指标替代传统TTC，精准量化交互相关性；③ 通过单/多视频对齐与RTS后处理，提升轨迹精度与可用性。

**🔧 技术方法**

无人机高帧率4K视频采集、YOLO11 OBB目标检测、ByteTrack多目标跟踪、RTS平滑、VTTC冲突提取、OpenDRIVE HD地图配准、数据可视化平台。

**📊 数据集**

VRUD自身（4小时4K/30Hz，包含11,479 VRU轨迹和1,939车辆轨迹）；与SIND、inD、INTERACTION等公开数据集进行对比。

**📈 对比分析**

通过VTTC阈值（上四分位1.53 s→1.53 s）筛选，得到4,002个交互样本；对比数据显示VRUD VRU占比≈87%，比SIND等高；统计显示车辆速度与VRU密度呈负相关，VTTC平均≈0.7 s，表明交互时间窗口的统一性。

**⚠️ 局限性**

仅覆盖深圳两处“城市村落”，地理范围有限；数据为日间高峰期，缺乏夜间与多天气条件；未公开完整原始视频，限制了对方法的复现。

---

## 512. Toward Personalized Darts Training: A Data-Driven Framework Based on Skeleton-Based Biomechanical Analysis and Motion Modeling

**arXiv ID:** 2604.01130 | [PDF](https://arxiv.org/pdf/2604.01130v1)

**作者:** Zhantao Chen `[一作]` (Qinghai University), Xuejun Hu `[通讯]` (Zhongshan Xiaolan Senior High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套基于标记无的骨骼运动捕捉与机器学习的保齿训练辅助系统，能够为运动员生成个性化的参考轨迹并实时给出训练建议。

**💡 创新点**

创新点在于突破传统模板匹配模式，采用“以个人历史高质量动作为基准”结合最小jerk模型生成个体参考轨迹，并用Z分数对单次投掷的18个可解释特征进行偏差诊断与分级反馈。

**🔧 技术方法**

技术手段包括Kinect 2.0深度摄像头与RGB摄像头的联合采集、骨骼坐标提取、三维向量与时间序列插值、最小jerk轨迹拟合、Z分数偏差检测及分级规则推送。

**📊 数据集**

使用了由七名运动员（四名职业选手、三名业余选手）在真实训练环境下收集的2396条投掷动作数据，构成了专门的保齿运动学数据集。

**📈 对比分析**

实验结果显示：生成的参考轨迹在平滑度、速度峰值、角度与姿态稳定性等方面与运动员历史高质量动作高度吻合；诊断模型能准确识别出三种严重偏差（躯干不稳、肘部失稳、释放速度过快），并给出针对性训练建议；相较于仅用单一模板匹配，系统在个体化适应性和反馈精准度上有明显提升。

**⚠️ 局限性**

局限性包括样本量有限、数据集中仅包含运动学特征而未结合视觉、肌电等多模态信息；参考模型仍依赖历史高质量数据，难以应对技术更新或受伤恢复阶段；推荐机制基于规则映射，缺乏因果推断与优先级排序，无法处理多重偏差共生的复杂情形。

---

## 513. Therefore I am. I Think

**arXiv ID:** 2604.01202 | [PDF](https://arxiv.org/pdf/2604.01202v1)

**作者:** Esakkivel Esakkiraja `[一作]` (Northeastern University), Rajagopal Venkatesaramani `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大型语言推理模型在产生链式思考之前是否已决定是否调用工具，利用线性探针检测隐藏状态中的决策信号，并通过激活向量干预验证其因果关系，最后用LLM评审分析干预后链式思考的行为变化。

**💡 创新点**

创新点在于：①首次证明工具调用决策可在第一个推理词出现前就已在隐藏层强烈编码；②通过激活向量干预提供了该决策的因果证据；③揭示干预后模型往往通过“合理化”而非抵抗来调整链式思考，暴露链式思考可能缺乏可信解释。

**🔧 技术方法**

采用线性探针（Logistic 回归）分析隐藏状态，使用激活向量（steering）注入/抑制决策方向，评估激活干预对行为的影响，并用 GPT-5.4 与 Claude Sonnet 4.6 进行行为分类。

**📊 数据集**

主要使用 NVIDIA When2Call 与 BFCL 两个工具调用决策基准，其中包含数千个多选题和 LLM-judge 样例。

**📈 对比分析**

在不同层和不同推理阶段进行交叉验证，发现 5% 推理前隐藏状态的 AUROC 可达 95% 以上；激活干预后，抑制/注入的翻转率分别高达 79%（GLM）与 62%（Qwen），并导致链式思考平均生成 1.4–2.5 倍的 token；行为分类显示大部分翻转通过“合理化”完成。

**⚠️ 局限性**

局限性包括：部分样例对干预表现出抵抗，导致链式思考长度显著增加但决策未变；仅针对二元工具调用决策，可能不适用于更复杂的多步骤决策；激活干预的参数选择依赖模型与基准，未充分验证跨模型可迁移性；此方法可能被用于攻击，需在 RL 训练中加入探针置信度惩罚以提升可解释性。

---

## 514. LLM REgression with a Latent Iterative State Head

**arXiv ID:** 2604.01206 | [PDF](https://arxiv.org/pdf/2604.01206v1)

**作者:** Yiheng Su `[一作]` (University of Texas at Austin), Matthew Lease `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为RELISH的轻量级预测头，用于在冻结大型语言模型（LLM）上直接进行文本回归任务。

**💡 创新点**

创新点在于使用可学习的迭代latent状态通过交叉注意力从token级表示中逐步提炼信息，替代传统的静态池化或索引方式，从而在保持单步推理效率的同时显著提升回归性能。

**🔧 技术方法**

技术细节包括：冻结LLM的隐藏层输出、投影至头部维度、构建多层残差交叉注意力块、线性回归头以及Huber损失训练；使用的LLM包括Llama 3.1 8B、Qwen 3 8B/32B、Gemma 3 27B；在训练中对目标进行y‑归一化。

**📊 数据集**

实验数据集涵盖语义文本相似度（STS‑B、SICK‑R）和机器翻译质量估计（WMT 2020 QE，包含英中、俄英、僧英三语对），共五个回归数据集。

**📈 对比分析**

与自回归解码、RAIL、RAFT、线性/MLP预测头等三大类基线对比，RELISH在四个LLM、五个数据集上均实现了最高的Pearson、Spearman相关系数和最低的NRMSE；参数量仅约3.4–3.7 M（占模型的0.01–0.04%），显著低于LoRA方案。

**⚠️ 局限性**

局限性包括：仅支持点估计，需有监督训练；不直接提供预测分布或置信区间；在极大模型规模时受隐藏维度限制；对跨语言或跨任务的泛化仍需进一步验证。

---

## 515. Assessing Affective Objectives for Communicative Visualizations

**arXiv ID:** 2604.01183 | [PDF](https://arxiv.org/pdf/2604.01183v1)

**作者:** Elsie Lee-Robbins `[一作]` (University of Michigan), Eytan Adar `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了在设计传达性可视化时，用学习目标（LO）来指导目标设定和评估的框架，并给出评估方法的选择准则，随后通过一项关于索马里人道危机的视频案例研究，验证了该框架在实际设计中的可行性。

**💡 创新点**

创新点在于：①将认知与情感两类学习目标统一纳入可视化设计流程；②为情感目标制定了一套从观察到行为的多层次评估方法与工具；③结合评估准则，系统性地对不同评估技术进行对比与筛选，帮助设计者快速做出评估选择。

**🔧 技术方法**

使用的技术包括：基于LO词汇表的问卷设计（Likert量表、真值诱导方法如贝叶斯真相预言）、行为测量（直接捐赠金额）以及在线问卷平台Prolific进行实验；评估分析采用预后后测、对照组比较和统计描述。

**📊 数据集**

使用的数据集主要是联合国人道协调办公室（UN OCHA）发布的2023年人道响应计划报告中的可视化与文本、相关照片和故事，以及实验参与者的自我报告与捐赠数据；受试者从Prolific招募，共计502人。

**📈 对比分析**

方法比较：通过三种视频设计（数据叙事、人文叙事、混合叙事）进行A/B测试，测量五个情感学习目标的评估结果。结果显示：在观察、重要性判断、责任与效能感这四项目标上三种设计差异不大；在捐赠行为目标上，人文叙事视频平均捐赠1.03美元，显著高于混合叙事（0.76美元）和数据叙事（0.87美元）。

**⚠️ 局限性**

局限性包括：①样本量虽达500+但仅为一次性线上实验，缺乏长期跟踪；②捐赠上限为2美元，可能低估真实捐赠潜力；③评估工具多为自我报告，易受社会期望偏差影响；④仅针对索马里危机的案例，结果的泛化性待验证；⑤情感目标的测量仍受当前可用量表与真值诱导方法的限制。

---

## 516. S0 Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models

**arXiv ID:** 2604.01168 | [PDF](https://arxiv.org/pdf/2604.01168v1)

**作者:** Jack Young `[一作]` `[通讯]`, Jack Young

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种零推理开销的PEFT方法，利用仅优化每层递归网络的初始状态矩阵，而不修改模型权重，实现对混合注意力-递归语言模型的快速任务适配。

**💡 创新点**

创新点在于把模型的全矩阵递归状态视为适配面，利用梯度优化学习一个可学习的初始状态，从而在保持推理成本为零的同时显著提升性能，并证明矩阵状态的表达性是关键。

**🔧 技术方法**

使用梯度下降在已验证的HumanEval解答上优化仅包含初始状态的张量；配合Alpha比例调节、state‑offset变体以及对比LoRA和前缀调优的实验，并进行KL衰减、首次字符偏移等机制分析。

**📊 数据集**

主要数据集包括HumanEval（80题训练+84题验证），交叉域评测MATH‑500、GSM8K、Spider；训练集为约48条执行验证通过的代码解答。

**📈 对比分析**

与LoRA（rank 24）和匹配参数的LoRA（rank 64）以及前缀调优等基线对比；在Qwen3.5‑4B上HumanEval pass@1提升约+23%（从48.8%到72.2%），LoRA仅+12%；在Mamba‑2 7B上两者相近；state‑offset获得最高精度但有推理开销；跨域任务分别提升+4.8%和+2.8%，Spider无提升。

**⚠️ 局限性**

局限性包括：需要执行验证过的少量解答（约48条）；仅在矩阵状态的混合模型上验证，对Diagonal状态无效；跨域迁移有限；未验证在更大规模或多任务场景下的稳健性；训练耗时短但受单GPU预算限制。

---

## 517. Detecting Multi-Agent Collusion Through Multi-Agent Interpretability

**arXiv ID:** 2604.01151 | [PDF](https://arxiv.org/pdf/2604.01151v1)

**作者:** Aaron Rose `[一作]` (University of Oxford), Christian Schroeder de Witt `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了多代理系统中隐蔽协作的检测，提出 NARCBench 基准和五种基于内部激活的探针方法。

**💡 创新点**

首次将激活探针扩展到多代理情境，并展示不同探针对应不同协作形式的局部激活信号；同时提出跨结构和隐写任务的零样本检测框架。

**🔧 技术方法**

采用线性激活探针、决策级融合、集中融合、相关性检测等五种探测技术，并在 Qwen3‑32B 模型中提取中间层激活，进行 token‑级投影分析。

**📊 数据集**

使用 NARCBench（Core、Transfer、Stego 三层级），共计 50+ 代码审查、股票操纵、RAG 污染、黑杰克隐写等多代理场景。

**📈 对比分析**

在 Core 上取得 1.00 AUROC；在 Transfer 上方法 AUROC 在 0.60–0.86 之间，最佳为 Asymmetry Probe 0.84；在 Stego 上所有方法 0.90–1.00 AUROC，表明零样本迁移能力强。

**⚠️ 局限性**

仅测试单一模型 Qwen3‑32B，探针可能受 prompt 余留影响；标签定义不完全；对抗鲁棒性未评估；是否能推广到更大模型或其他场景尚未验证。

---

## 518. Automated Generation of High-Quality Bug Reports for Android Applications

**arXiv ID:** 2604.01148 | [PDF](https://arxiv.org/pdf/2604.01148v1)

**作者:** Antu Saha `[一作]` (William & Mary), Oscar Chaparro `[通讯]` (William & Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM结合应用程序的GUI交互、屏幕描述和故障屏幕信息，自动生成高质量的bug报告（Observed Behavior、Expected Behavior、Steps to Reproduce）。

**💡 创新点**

创新点在于：①针对每个报告组件（OB、EB、S2R）选择最合适的应用上下文；②使用零射击、任务分解提示提升LLM生成准确、完整、可验证的步骤；③将动态执行模型与LLM推理融合，实现自动化、细粒度的报告改进。

**🔧 技术方法**

技术包括：大语言模型（GPT‑4o/ChatGPT），图形化执行模型（屏幕与交互的有向图），零射击/任务分解提示，应用程序动态分析（UI探索与手工录制），以及基于规则的句子分类与屏幕描述生成。

**📊 数据集**

数据集为48条来自26个Android应用的bug报告（原始报告+人工生成的高质量OB/EB/S2R），以及开发阶段的10条报告用于调优。

**📈 对比分析**

通过与原始报告、两个LLM基线（无上下文）以及Acharya等人提出的结构化报告方法对比，测量精确度、召回率和F1。实验显示：S2R的F1从约0.45提升至0.88（44.1%–82.3%提升），OB/EB中正确元素数从122提升至165，缺失元素降至0。性能优于所有基线。

**⚠️ 局限性**

局限性包括：需要完整的执行模型和屏幕元数据；当模型路径多样或报告信息不足时仍会漏步或多步；错误的故障屏幕定位导致OB/EB不准确；LLM非确定性需多次运行；评估数据仅限Android，缺乏跨平台验证。

---

## 519. SERSEM: Selective Entropy-Weighted Scoring for Membership Inference in Code Language Models

**arXiv ID:** 2604.01147 | [PDF](https://arxiv.org/pdf/2604.01147v1)

**作者:** Kıvanç Kuzey Dikici `[一作]` (Bilkent University), Sinem Sav `[通讯]` (Bilkent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SERSEM，基于代码结构的成员推断攻击方法，利用 AST、linting 生成权重掩码并结合内部激活探测来识别代码模型的记忆痕迹。

**💡 创新点**

创新点在于将成员推断从全局概率切换为基于语法与人类编码失误的连续熵加权评分，并将表层输出与深层隐藏状态的多维特征融合，显著提高对代码 LLM 的识别准确率。

**🔧 技术方法**

技术包括静态抽象语法树分析、拼写检查、多语言逻辑检测、离线 linting、连续权重掩码、Transformer 内部激活线性探测（LUMIA）以及基于 Z‑score 的概率校准。

**📊 数据集**

使用 25,000 条平衡样本，来自 The Stack v2（成员）和 The Heap（非成员）的 Python、Java、Go、Ruby、Rust 代码，评估在 StarCoder2-3B 与 StarCoder2-7B 两个模型上。

**📈 对比分析**

与 Loss、Min‑K% Prob、PAC 三个基准对比，SERSEM 在 3B 模型上 AUC‑ROC 0.7913，7B 模型上 0.7867，均比基准高出 20+ 百分点，显示出显著优势。

**⚠️ 局限性**

局限在于依赖手工设定的正则与 lint 规则，对非 Python 语言的泛化不充分，且在高结构化语言（如 Rust、Java）中的检测性能仍相对较弱。

---

## 520. Looking into a Pixel by Nonlinear Unmixing -- A Generative Approach

**arXiv ID:** 2604.01141 | [PDF](https://arxiv.org/pdf/2604.01141v1)

**作者:** Maofeng Tang `[一作]` (University of Tennessee), Hairong Qi `[通讯]` (University of Tennessee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无模型的 hyperspectral 非线性混合去混方法，基于双向 CycleGAN 并加入线性约束与语义一致性正则化；

**💡 创新点**

创新点在于：① 通过双向生成网络实现可逆混合-去混流程；② 利用 Dirichlet 分布和线性混合的语义一致性约束；③ 用互信息损失替代传统重建损失，提升鲁棒性；

**🔧 技术方法**

使用技术包括：生成对抗网络（GAN）、CycleGAN、卷积-反卷积网络、Dirichlet 分布采样、互信息估计（MINE）、双向循环一致性损失；

**📊 数据集**

数据集涵盖：合成图像（LMM、BMM、PNMM、MLM，SNR 30/20/15 dB）；真实数据集 Urban 与 Washington D.C.（HYDICE），均采用 USGS 端元；

**📈 对比分析**

与模型基方法（FCLS、GBM、PPNM、MLM）及深度学习方法（uDAS、NN-LM）对比，实验表明本文方法在 AAD、AID、RE、SAD 等指标上均优于或相当于最先进方法，且对噪声和混合模型不匹配具有更强的泛化与鲁棒性；

**⚠️ 局限性**

局限性在于：仍假设端元已知，且模型训练需要多尺度补丁处理，计算量相对较大；

---

## 521. The Recipe Matters More Than the Kitchen:Mathematical Foundations of the AI Weather Prediction Pipeline

**arXiv ID:** 2604.01215 | [PDF](https://arxiv.org/pdf/2604.01215v1)

**作者:** Piyush Garg `[一作]` (RWE Trading Americas Inc.), Galen J. Yacalis `[通讯]` (RWE Trading Americas Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了一个全流程的学习管道框架，系统地把模型结构、损失函数、训练策略和数据分布统称为Pipeline，并在此基础上构建了误差分解、谱损失理论、预测极值边界、错误共识比率以及最终的多维度Holistic Model Assessment Score。

**💡 创新点**

创新点包括：① 证明在当前操作分辨率下，架构误差远小于估计误差，强调损失函数和数据分布对预报质量的主导作用；② 推导MSE损失在球面谐波空间中的谱衰减定理，解释所有MSE模型在高波数处的谱能量缺失；③ 通过错误共识比率量化模型间共享误差，验证“Pipeline主导”假设；④ 证明数据驱动模型对记录极值的系统性低估与条件均值回归相关，给出线性偏差-极值超过关系；⑤ 设计多维度HMAS，将谱、物理一致性、能量稳定性、极端事件准确性等六个指标综合。

**🔧 技术方法**

技术手段主要包括：球面逼近理论（Sobolev空间、球谐展开）、动态系统与信息理论（Lyapunov指数、KS熵）、统计学习误差分解、谱损失（MSE、MSH、CRPS、score‑matching）以及基于谱能量比率和能量稳定指数的诊断；数据集与模型交互方面采用NVIDIA Earth2Studio的ERA5初始场，配合多源预训练（ERA5+GFS+CMIP6）和不同损失权重。对模型性能的比较使用六个数学度量（RMSE、ACC、SFI、PCS、EES、ASI）和对应的置信区间。

**📊 数据集**

使用NVIDIA Earth2Studio平台上的ERA5重分析初始场，覆盖30个初始化日期（4个季节），包括10个不同架构的AI气象模型（GraphCast、AIFS、AIFS‑ENS、Aurora、FourCastNet 3、FengWu、FuXi、Pangu‑Weather、SFNO、Atlas）。

**📈 对比分析**

在所有模型间以同一训练数据、相同验证标准（纬度加权RMSE、ACC）进行跨模型比较，发现架构差异对误差贡献极小，损失函数与训练策略决定了绝大部分性能差异；SFI、PCS、ASI等指标显示不同模型在谱保真、物理一致性和能量稳定性上的差别；HMAS给出了单一排名，验证了多维度评估的必要性。整体上，最优模型在HMAS上排在前列，但排名随预测时效和指标权重而波动，说明单一指标无法全面评估。

**⚠️ 局限性**

主要局限包括：① 理论推导假设完备的近似与无误差的优化过程，实际模型受优化误差与数据不足影响；② 只评估了十个模型，可能不具备足够代表性；③ 训练管道与架构在实验中相互耦合，无法完全分离两者对误差的具体贡献；④ 仅使用ERA5初始场，未检验实时气象数据或其他再分析对OOB表现的影响；⑤ 某些诊断（如能量稳定指数）对数值模式选择和时间步长高度敏感，可能导致结果对特定实现的依赖。

---

## 522. Collaborative Task and Path Planning for Heterogeneous Robotic Teams using Multi-Agent PPO

**arXiv ID:** 2604.01213 | [PDF](https://arxiv.org/pdf/2604.01213v1)

**作者:** Matthias Rubio `[一作]` (ETH Zürich), Marco Hutter `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于Multi‑Agent PPO的协同多机器人路径规划与任务分配框架，可在二维网格上让异构机器人队伍完成分配、调度与路径规划，并支持在线重规划；

**💡 创新点**

创新点在于：①将路径规划、目标分配和调度统一到端到端的强化学习模型中；②采用bootstrap+refinement训练策略与多目标奖励设计，使模型既能快速收敛又能实现高效合作；③通过固定观测实现常数推理时间，显著降低运行时计算开销；

**🔧 技术方法**

使用Multi‑Agent Proximal Policy Optimization（MAPPO）作为核心算法，配合GRU+全连接的 actor‑critic 网络；设计多种奖励（吸引、目标、错误、步进、求解时间、终端奖励）；在离散二维网格环境中实现训练与测试；实现框架基于 JaxMARL；

**📊 数据集**

使用合成数据：在随机生成的二维网格环境中，随机赋予机器人不同技能集合、目标不同技能需求及目标类型（AND/OR），并在训练与评估中分别设置 5、6、7 个目标；

**📈 对比分析**

通过与完整搜索（ES1、ES2）最优解进行对比，评估成功率、求解时间（M_st）和团队努力（M_tte）；结果显示：成功率均超过 90%，团队努力相对性能达到 92%、91%、84%，求解时间相对性能为 86%、81%、73%；推理时间与搜索相比从指数级降低到常数级，训练时间随目标数呈线性增长；

**⚠️ 局限性**

主要局限：固定观测尺寸限制了目标数与机器人队规模，导致难以扩展；训练时间随目标/技能数量急剧增长，需昂贵 GPU 资源；奖励与参数需手动调优，模型在不同规模任务上的泛化能力受限；

---

## 523. ORBIT: Scalable and Verifiable Data Generation for Search Agents on a Tight Budget

**arXiv ID:** 2604.01195 | [PDF](https://arxiv.org/pdf/2604.01195v1)

**作者:** Nandan Thakur `[一作]` (University of Waterloo), Jimmy Lin `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种四阶段无成本、无先决条件的框架，用自动化检索与自检/外部验证生成20K个多步推理问答对，训练搜索代理；

**💡 创新点**

创新点在于：①实现低预算、全自动的复杂多步检索数据集构建；②引入双阶段验证（自检+外部LLM评判）提升答案可靠性；③公开框架、数据与代码，降低研究门槛；

**🔧 技术方法**

采用DeepSeek Chat、Python Selenium自动交互、DDGS搜索聚合、GRPO强化学习、Qwen3-4B LLM、外部LLM评判等技术；

**📊 数据集**

构建了自己的 Open-Web Reasoning (odysseyrow) 数据集，基准对比使用NQ、HotpotQA、InfoSeek等公开数据集；

**📈 对比分析**

在Wikipedia单跳与多跳QA基准上，用Qwen3-4B+GRPO训练的odysseyrow-4B模型在EM上平均提升约9点，显著优于Search-R1、InfoSeeker等基线；

**⚠️ 局限性**

局限性在于：仅使用免费搜索工具和小模型，缺乏付费API、全文抓取与更大模型支持，训练规模和检索质量受限，未来需扩展资源与工具。

---

## 524. True (VIS) Lies: Analyzing How Generative AI Recognizes Intentionality, Rhetoric, and Misleadingness in Visualization Lies

**arXiv ID:** 2604.01181 | [PDF](https://arxiv.org/pdf/2604.01181v1)

**作者:** Graziano Blasilli `[一作]` (Sapienza University of Rome), Marco Angelini `[通讯]` (Link Campus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估多模态大型语言模型在识别误导性可视化、识别视觉修辞以及归因作者意图方面的能力。

**💡 创新点**

提出九类作者意图的新分类法，并设计六个实验条件（A、B、C）在不同先验知识下系统评估模型表现，结合COVID‑19推文与VisLies数据集进行对比。

**🔧 技术方法**

使用多模态LLM（共16种），自定义提示、评分矩阵、错误敏感度得分（ESS）和模型行为相似度（MBS）等技术进行量化分析。

**📊 数据集**

利用COVID‑19推文2,336条（50%误导）和VisLies社区收集的130条真实误导可视化，并配合11名可视化专家的人工标注。

**📈 对比分析**

通过召回率、MCC、ESS、MBS等指标进行比较，发现大模型在无先验知识下误导检测召回率高，但存在误报；在提供错误信息的C条件下性能提升，模型在修辞和意图识别上的表现差异显著。

**⚠️ 局限性**

局限性包括：仅评估大型模型，未测试小型模型；仅使用单一提示与默认温度；数据集错误标签与更细粒度的Lo等分类不匹配；人工基准仅来自专家；未深入探究嵌入空间结构。

---

## 525. Safe learning-based control via function-based uncertainty quantification

**arXiv ID:** 2604.01173 | [PDF](https://arxiv.org/pdf/2604.01173v1)

**作者:** Abdullah Tokmak `[一作]` (Aalto University), Dominik Baumann `[通讯]` (Aalto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于可视化支持场景的强化学习方法，用于提升连续控制任务的安全性和性能。

**💡 创新点**

创新点在于将支持场景概念与深度强化学习结合，并使用动态边界约束提高了策略的鲁棒性。

**🔧 技术方法**

采用的技术包括深度神经网络、可视化支持场景生成、蒙特卡洛采样以及多目标优化。

**📊 数据集**

实验使用了MuJoCo物理引擎中的多种仿真环境，如HalfCheetah、Walker2d和Ant等。

**📈 对比分析**

与传统的基线方法（如DDPG、PPO、LQR等）对比，本文方法在奖励得分和收敛速度上分别提升了约10%和30%。

**⚠️ 局限性**

局限性包括对高维环境的可扩展性有限，且支持场景生成需要大量计算资源。

---

## 526. Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning

**arXiv ID:** 2604.01152 | [PDF](https://arxiv.org/pdf/2604.01152v1)

**作者:** Mohammad R. Abu Ayyash `[一作]` `[通讯]` (Brains Build Research), Mohammad R. Abu Ayyash (Brains Build Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Brainstacks，一种通过冻结的 MoE‑LoRA 适配器堆叠实现的连续多域 LLM 微调框架；

**💡 创新点**

创新点在于将内部残差提升、外部持续堆叠、随机SVD零空间投影、基于结果的 sigmoid 元路由以及磁盘卸载推理等多项技术组合成一套完整的无遗忘、可组合的模块化系统；

**🔧 技术方法**

核心技术包括：MoE‑LoRA 适配器（Shazeer‑style 噪声 top‑2 路由、rsLoRA 缩放）、残差提升（inner loop）、连续域堆叠（outer loop）、随机SVD 零空间投影、基于结果的 sigmoid 元路由、磁盘卸载推理等；

**📊 数据集**

在 TinyLlama‑1.1B（chat、code、math、medical 四域）和 Gemma 3 12B IT（chat、code、math、medical、reasoning 五域）上使用公开数据集（alpaca、python_code_instructions、GSM8K、MedQA‑USMLE 等）；

**📈 对比分析**

与单一 LoRA、单块 MoE‑LoRA、以及未路由的堆叠模型对比，MoE‑LoRA 训练速度提升 2.5×，残差提升突破单块上限，零空间投影与元路由组合后实现零遗忘；在 8 个零样本基准上，路由模型在大多数任务保持或略优表现，未出现灾难性退化；

**⚠️ 局限性**

局限性包括：推理时需要逐个加载冻结堆叠导致延迟、隐藏维度容量上限（每域 64 方向），路由训练对数据敏感、需预训练基础模型（随机初始化效果差），以及新堆叠易被后续训练误导导致性能下降。

---

## 527. Leveraging Commit Size Context and Hyper Co-Change Graph Centralities for Defect Prediction

**arXiv ID:** 2604.01132 | [PDF](https://arxiv.org/pdf/2604.01132v1)

**作者:** Amit Kumar `[一作]` (Indian Institute of Information Technology), Sonali Agarwal `[通讯]` (Indian Institute of Information Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对文件级软件缺陷预测，提出了两种新颖的特征表示：基于提交大小的向量化过程指标和基于超图的向量中心性，并将它们与传统的产品指标结合进行实验。

**💡 创新点**

创新点在于：① 将每个文件的过程指标从标量转化为包含 1–100 个维度的向量，显式捕获不同提交规模对文件变更行为的影响；② 构造超图模型（每条超边代表一次完整提交），并计算向量化的度/介数/接近/特征向量中心性，从而获取文件在不同规模提交中的重要性。两者相结合显著提升了缺陷预测的判别与校准能力。

**🔧 技术方法**

技术方法包括：① 5 种机器学习分类器（Logistic 回归、SVM、XGBoost、GBM、随机森林）；② SMOTE 处理类别不平衡；③ HSIC Lasso 进行高维特征选择；④ Bootstrap 100 次自举验证；⑤ 统计检验采用 Friedman 与 Nemenyi 测试；⑥ 评估指标为 AUROC、AUPRC、F1、MCC、Brier 分数。

**📊 数据集**

使用了 9 个长期维护的 Apache 开源项目（共 32 个版本）：Activemq、Camel、Derby、Groovy、HBase、Hive、JRuby、Lucene、Wicket。每个项目提供 54 个产品指标、14 个标量过程指标和 1400 个向量化过程指标。

**📈 对比分析**

通过对比三组特征集合（PR+SP、PR+VP、PR+VP+VC）在所有 45 个数据集–分类器组合上进行 bootstrap 评估，结果显示：① PR+VP 在 AUROC、AUPRC、F1、MCC 上平均提升 3.5–26.7%；② PR+VP+VC 在所有指标上均最高，最佳模型出现率分别为 AUROC 75.6%、MCC 66.7%、F1 71.1%、AUPRC 62.2%，且 Brier 分数下降 3–11%。统计检验表明提升显著（p<0.05）。

**⚠️ 局限性**

局限性包括：① 实验仅覆盖 Java 项目，泛化性待验证；② 只使用一个数据集，缺乏跨域评估；③ 向量化后特征维度高，计算成本和可解释性受限；④ 对超图构建与向量中心性计算的实现细节及参数选择仍需进一步研究。

---

## 528. $\texttt{YC-Bench}$: Benchmarking AI Agents for Long-Term Planning and Consistent Execution

**arXiv ID:** 2604.01212 | [PDF](https://arxiv.org/pdf/2604.01212v1)

**作者:** Muyu He `[一作]`, Nazneen Rajani `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 YC-Bench，一个基于 POMDP 的长期连贯性基准，用 LLM 代理模拟一年期的初创公司运营，涵盖任务接受、员工分配、资金管理与客户信任。

**💡 创新点**

创新点在于：①引入隐藏的对手客户与工作量膨胀机制；②多领域任务与员工技能不对齐的“信息不对称”问题；③利用 20 步截断的对话窗口配合可持久化的 scratchpad 进行长期记忆；④通过累积财务动态和递增工资展示长期规划与风险管理。

**🔧 技术方法**

使用技术包括：LLM 代理通过 CLI 交互、POMDP 建模、deterministic 转移、基于 scratchpad 的自我记录与检索、对环境的观察/行动序列。

**📊 数据集**

数据集为合成生成的任务市场、员工技能档案以及客戶列表，其中约三分之一客戶为对手且隐藏属性需被模型推断；所有数值信息以数值形式提供。

**📈 对比分析**

对比方法：在 12 款前沿模型（GPT‑5.4 系列、Claude、Gemini、Qwen、GLM‑5、Kimi‑K2.5、Grok）上跑 3 种 seed，测量一年后最终资金、盈利率与 API 成本效率；结果显示仅 3 模型平均超 1M 美元、5 模型盈利，均优于贪婪基线；成本效率方面 Kimi‑K2.5 表现最佳。

**⚠️ 局限性**

局限性包括：员工不可雇佣/解雇、只有对手客户干扰、所有量化信号为数值而非自然语言、环境转移确定且未覆盖更广泛的外部冲击，未来可通过引入随机事件、招聘机制与自然语言输入来提升真实性。

---

## 529. Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning

**arXiv ID:** 2604.01170 | [PDF](https://arxiv.org/pdf/2604.01170v1)

**作者:** Cai Zhou `[一作]` (Massachusetts Institute of Technology), Stephen Bates `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在线推理校准框架ORCA，结合测试时训练与合规预测，对大型语言模型的推理过程实现自适应计算分配与置信度估计。

**💡 创新点**

把校准视作可在推理时更新的预测问题，内部循环使用fast‑weight 在线更新校准模块，外部循环通过元学习学习初始化与更新动态；同时用Learn‑then‑Test对整个停止策略进行合规校准，提供统计风险控制并提升对分布移位的鲁棒性。

**🔧 技术方法**

测试时训练（TTT）、合规预测（conformal prediction）、Learn‑then‑Test（LTT）风险控制、fast‑weight 在线更新、可选的Q/K投影、Brier损失、步长与阈值网格搜索等。

**📊 数据集**

训练集5K（1k Math、2k OpenR1、2k DeepMath），校准集1k，测试集1k；OOD评估集包括MATH‑500、GPQA‑Diamond、AIME 2024/25/26；模型使用Qwen2.5‑32B、QwQ‑32B和Llama‑3.3‑70B。

**📈 对比分析**

与静态线性探针基线对比，使用LTT选取阈值；评价指标为步数节省率（Savings）和错误率。结果显示，在δ=0.1下，ORCA在Qwen上可节省47.5%推理步数，OOD MATH‑500上节省率提升至63–67%；跨模型（QwQ、Llama）亦优于基线，且错误率始终保持在预算以内。

**⚠️ 局限性**

仍依赖手工阈值网格与交换性假设；仅在步骤级别提供校准，未覆盖长序列或跨模态推理；基础LLM参数未被微调，仅使用fast‑weights 进行更新。

---

## 530. Functional Force-Aware Retargeting from Virtual Human Demos to Soft Robot Policies

**arXiv ID:** 2604.01224 | [PDF](https://arxiv.org/pdf/2604.01224v1)

**作者:** Uksang Yoo `[一作]` (Robotics Institute, Carnegie Mellon University), Harsha Prahlad `[通讯]` (Meta Reality Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过在虚拟现实环境中收集人类手部演示并利用接触力信息，设计了两阶段力平衡与曲面测地加权的重映射方法，将人类操纵技能迁移到非人形软机器人手，并训练低层控制器和策略实现零shot真实部署。

**💡 创新点**

创新点在于：①首次将接触力分配与软手非人形结构相匹配的力平衡手指分配引入重映射；②在线基于测地距离的接触加权微调，使软手能根据实际接触几何自适应；③提出端到端的接触几何驱动学习框架，突破传统运动学对齐的局限。

**🔧 技术方法**

技术包括：虚拟现实+运动捕捉的接触力数据采集；测地距离与热核算法进行力分配与重映射；有限元仿真软手动力学；扩散模型策略学习；MLP+优化的学习逆向低层控制器。

**📊 数据集**

数据集：基于VR收集的六个任务（灯泡插入/旋转、杯子倒灌、标记抓取、瓶子旋拆、盒子重新定向）的手部姿势、对象位姿与接触点/接触力信息；以及对应的仿真轨迹数据用于训练与评估。

**📈 对比分析**

与仅基于运动学的重映射、纯学习的低层控制器以及传统PID等基线比较，力感知重映射在软手轨迹跟踪RMSE下降55%/69%，任务成功率在仿真中提升约30–70%，在零shot真实部署中的成功率也显著高于基线，验证了接触几何与力分布建模的有效性。

**⚠️ 局限性**

局限性包括：需要在仿真中获得精确接触力标签，无法直接从真实世界收集；仅针对短期、固定结构的接触任务；依赖运动捕捉标记获取物体位姿，缺乏对视觉感知的支持；对更长时程、多阶段操作的迁移能力尚未验证。

---

