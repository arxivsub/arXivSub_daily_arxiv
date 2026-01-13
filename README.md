# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-13 | 今日论文总数: 883

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Error correction methods based on two-faced processes

**arXiv ID:** 2601.06447 | [PDF](https://arxiv.org/pdf/2601.06447v1)

**作者:** Boris Ryabko `[一作]` `[通讯]` (Federal Research Center for Information and Computational Technologies), Boris Ryabko (Federal Research Center for Information and Computational Technologies)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了一种基于双面过程的误差校正方法，先把输入序列变换为高互相依赖的两面过程，再利用该属性在EC和BSC通道中纠正错误，编码解码复杂度为线性。

**💡 创新点**

创新点在于通过特定的映射把普通Bernoulli序列转化为两面过程，从而利用其长程统计特性显著降低误码率，实现无额外冗余的误差校正。

**🔧 技术方法**

采用了两面过程的马尔可夫链理论、ϕ与ψ映射、Hoeffding不等式等信息论与概率工具。

**📊 数据集**

论文未使用实验数据集，而是给出理论推导和大样本极限分析。

**📈 对比分析**

与传统块码/卷积码等方法对比，理论上在低错误概率下误码率可达到π·o(1)，即随通道误码率的平方级下降，表现优于简单的置零填充。

**⚠️ 局限性**

限制在于需已知源熵小于通道容量，且对高误码率下的性能尚未给出实验验证；此外需要对p>1/2的假设。

---

## 2. SPINAL -- Scaling-law and Preference Integration in Neural Alignment Layers

**arXiv ID:** 2601.06238 | [PDF](https://arxiv.org/pdf/2601.06238v1)

**作者:** Arion Das `[一作]` (Indira Gandhi Institute of Technology Ranchi), Amitava Das `[通讯]` (Pragya Lab BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SPINAL（Scaling-law and Preference Integration in Neural Alignment Layers）诊断方法，用于可视化和量化 LLM 在 DPO（Direct Preference Optimization）对齐过程中深度层内部的几何变化。

**💡 创新点**

创新点在于：①将对齐视为局部几何校准而非全局行为重写；②定义层级化的谱指数 α_ℓ 与 Fisher–Rao 传输长度 ℒ_ℓ 两个可解释信号；③通过 Δ_align、轨迹连贯度和梯度集中度三项综合形成 SPINALScore，提供可比、可审计的对齐指标。

**🔧 技术方法**

技术方法包括：对每层激活做奇异值谱拟合得到 α_ℓ；利用 logit‑lens 的 Gibbs 分布计算相邻层的 Bhattacharyya 系数，进而得到 Fisher–Rao 距离 ℒ_ℓ；聚合终端层（通常 21–30 层）得到 Δ_align、连贯度和梯度集中度；最终线性组合得到 SPINALScore。

**📊 数据集**

实验使用 512 条 Anthropic HH 提示，B=64，模型族包括 Phi‑2、DeepSeek、Gemma、Qwen、Llama 3；每个模型都有对齐前后的基线与 DPO 校准版本。

**📈 对比分析**

通过对齐前后模型在 α_ℓ 与 ℒ_ℓ 曲线、Δ_align、连贯度、梯度集中度的差异进行比较，发现对齐后终端层谱指数显著上升、Fisher–Rao 距离下降、梯度集中度提升；SPINALScore 在所有五个模型中保持一致且与行为评估（HCR、SRQ）呈显著相关，表明对齐的几何指纹是可测量且一致的。

**⚠️ 局限性**

主要限制包括：①验证集中于解码器‑only 中等规模模型，未覆盖 encoder‑decoder、MoE 或超大规模模型；②对齐指标对训练目标敏感，RLHF、宪法式对齐等可能产生不同的几何特征；③Fisher–Rao 解释为“热力学长度”尚缺乏严格理论支持；④指标对提示集、token 位置、top‑k 截断等评估细节敏感，需进一步稳健性验证。

---

## 3. Synthetic FMCW Radar Range Azimuth Maps Augmentation with Generative Diffusion Model

**arXiv ID:** 2601.06228 | [PDF](https://arxiv.org/pdf/2601.06228v1)

**作者:** Zhaoze Wang `[一作]` (HELLA GmbH & Co. KGaA), Markus Gardill `[通讯]` (Brandenburg University of Technology)

**通讯引用:** 296 | [OpenAlex ID](https://openalex.org/A5041147083)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用条件扩散模型合成逼真的FMCW雷达范围-方位图（RAMap），并通过生成的雷达数据扩充数据集以提升目标检测性能。

**💡 创新点**

创新点包括：①使用ConfMap作为条件输入，结合几何感知校正（GAC）和目标一致性正则化（TCR）来强化雷达信号的物理一致性；②引入多类别雷达信号的条件生成框架；③在雷达领域首次系统评估扩散模型的生成质量与下游检测效果。

**🔧 技术方法**

技术包括条件扩散模型（SDM）、U‑Net结构、几何感知修正、目标一致性正则化、PSNR与mAP评价指标。

**📊 数据集**

使用ROD2021雷达数据集（包含行人、自行车和车辆）进行训练与评估。

**📈 对比分析**

与基线方法（使用边界框掩码的de Oliveira等）以及基于图像处理的RAMap增强方法比较，生成的雷达图像PSNR提升约3.6 dB；使用混合数据集训练的检测器在mAP上提升约4.15%，在低样本场景（市街、快速路）尤为显著。

**⚠️ 局限性**

局限性包括：仅在ROD2021上验证，未考虑多普勒或微多普勒特征；扩散采样耗时较大；对复杂多路径与遮挡的建模仍相对简化。

---

## 4. TimeGNN-Augmented Hybrid-Action MARL for Fine-Grained Task Partitioning and Energy-Aware Offloading in MEC

**arXiv ID:** 2601.06191 | [PDF](https://arxiv.org/pdf/2601.06191v1)

**作者:** Wei Ai `[一作]` (Central South University of Forestry and Technology), Keqin Li `[通讯]` (State University of New York)

**通讯引用:** 29725 | [OpenAlex ID](https://openalex.org/A5087894632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种TG-DCMADDPG框架，用于移动边缘计算中细粒度任务分割和能耗感知的任务卸载。

**💡 创新点**

创新点在于将时间图神经网络用于预测服务器状态，结合离散-连续混合动作的多智能体强化学习，实现前瞻性能耗与延迟优化。

**🔧 技术方法**

主要技术包括时间图神经网络 (TimeGNN)、多智能体确定性策略梯度算法 (DC-MADDPG)、中心化训练-分散执行 (CTDE) 与 Gumbel-Softmax。

**📊 数据集**

使用仿真环境生成的多设备、多服务器时变任务与信道数据，未使用公开真实数据集。

**📈 对比分析**

与DC-MADDPG、DC-MAAC、DC-MA2C、ROP、FOO等基线对比，TG-DCMADDPG在收敛速度、能耗、延迟和任务完成率上均表现最佳。

**⚠️ 局限性**

局限性包括仅在仿真中验证，未考虑模型误差导致的预测偏差，以及对极端能耗极限或更大规模网络的鲁棒性未知。

---

## 5. How well can off-the-shelf LLMs elucidate molecular structures from mass spectra using chain-of-thought reasoning?

**arXiv ID:** 2601.06289 | [PDF](https://arxiv.org/pdf/2601.06289v1)

**作者:** Yufeng Wang `[一作]` (Stony Brook University), Haibin Ling `[通讯]` (Stony Brook University)

**通讯引用:** 35154 | [OpenAlex ID](https://openalex.org/A5061469520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对零样本大型语言模型进行链式推理（CoT）提示，评估其在 MS/MS 到 SMILES 结构推理任务中的表现。

**💡 创新点**

提出了专门的 CoT 提示模板与基准评估框架，并首次量化 LLM 在 MassSpecGym 数据集上的结构推理准确性和推理质量。

**🔧 技术方法**

利用 Chain‑of‑Thought 提示、零样本推理、SMILES 生成、并用 Tanimoto 与 MCES 等结构相似性指标进行评估。

**📊 数据集**

使用公开的 MassSpecGym 数据集（约 231,000 条 MS/MS 光谱与 29,000 结构）。

**📈 对比分析**

与 Claude‑3.5‑Sonnet、GPT‑4o‑mini、Llama‑3‑70B、Llama‑3‑8B 等多款 LLM 进行对比；结果显示 SMILES 有效率可达 90%（Claude）但化学一致性仅约 4%，Top‑10 Tanimoto 仅 0.09‑0.12，远低于专门训练的谱学模型（如 DiffMS）。

**⚠️ 局限性**

缺乏对碎片化学物理约束的内在理解，CoT 生成的推理步骤常出现 DBE、原子计数错误，最终结构几乎无法与真实分子匹配，需结合谱学编码与化学约束实现可靠推理。

---

## 6. SyntaxMind at BLP-2025 Task 1: Leveraging Attention Fusion of CNN and GRU for Hate Speech Detection

**arXiv ID:** 2601.06306 | [PDF](https://arxiv.org/pdf/2601.06306v1)

**作者:** Md. Shihab Uddin Riad `[一作]` `[通讯]` (International Islamic University Chittagong), Md. Shihab Uddin Riad (International Islamic University Chittagong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的模型，将BanglaBERT嵌入与并行CNN与Bi‑GRU+注意力相结合，用于Bengali仇恨言论检测与目标群体识别；

**💡 创新点**

创新点在于同时使用CNN提取局部n‑gram特征和Bi‑GRU捕获全局语境，并通过自注意力和特征融合实现两条路径的协同学习；

**🔧 技术方法**

使用了BanglaBERT预训练、Bi‑GRU+self‑attention、CNN+self‑attention、特征融合、线性分类器，并在PyTorch上采用AdamW、CrossEntropyLoss、梯度裁剪等技术；

**📊 数据集**

实验基于BLP‑2025 Task 1的Subtask 1A和1B数据集，各35,522条样本，包含hate‑speech类别和目标群体标签；

**📈 对比分析**

与竞赛中其他团队比较，Subtask 1A获得0.7345的micro F1（第2名），Subtask 1B获得0.7317（第5名），性能接近前列；

**⚠️ 局限性**

局限性包括数据严重类别不平衡导致少数类表现不佳、模型结构较为复杂、计算资源需求高，且未对Subtask 1C进行验证。

---

## 7. VideoWeave: A Data-Centric Approach for Efficient Video Understanding

**arXiv ID:** 2601.06309 | [PDF](https://arxiv.org/pdf/2601.06309v1)

**作者:** Zane Durante `[一作]` (Stanford University), Li Fei-Fei `[通讯]` (Stanford University)

**通讯引用:** 212174 | [OpenAlex ID](https://openalex.org/A5100450462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 VideoWeave 方法，通过拼接短视频段落生成合成长视频训练样本，从而提升视频语言模型在固定算力下的数据效率。

**💡 创新点**

创新点在于仅对训练数据进行重组而非改造模型结构：使用随机或视觉聚类拼接短视频并保持原始字幕，利用无额外标注的方式构造多视频上下文，显著提升训练效果。

**🔧 技术方法**

技术手段包括 CLIP‑ViT‑L/14 视觉编码器、LLaMA‑2‑7B 语言模型、RoPE 上下文长度扩展、K‑means 聚类、随机视频拼接以及将帧压缩到批量维度进行并行处理。

**📊 数据集**

使用数据集为 WebVid‑10M 进行训练（10k–160k 子集），评测基准为 VideoMME。

**📈 对比分析**

在 1 轮、10,000 次迭代、固定 GPU 计算资源下，与单视频微调和图像基准对比，VideoWeave 在 VideoMME‑Short 上从 41.7% 提升至 44.8%（+3%），在全局上从 34.5% 提升至 36.6%（+2.1%）。

**⚠️ 局限性**

局限性：实验仅覆盖短视频且内容相对均匀，对长时序视频的适用性未知；未在更大规模或其他评测集上验证鲁棒性。

---

## 8. What Users Leave Unsaid: Under-Specified Queries Limit Vision-Language Models

**arXiv ID:** 2601.06165 | [PDF](https://arxiv.org/pdf/2601.06165v1)

**作者:** Dasol Choi `[一作]` (Yonsei University), Youngsook Song `[通讯]` (Lablup Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了HAERAE‑Vision，一个包含653个来自韩国在线社区的真实视觉问答对，并配套1,306个显式化的查询版本；

**💡 创新点**

创新点在于通过严谨的六阶段过滤流水线获取真实的、欠缺信息的自然查询，并通过显式化改写来量化查询模糊对VLM表现的影响；

**🔧 技术方法**

采用LLM（GPT‑4o、GPT‑5、Gemini等）进行自动过滤、评估、改写，结合检索增强、结构化检查表以及LLM评判器实现统一评测；

**📊 数据集**

使用86,052个来自九大韩语社区的问答对，经过过滤得到653个高质量问题，覆盖13个类别；

**📈 对比分析**

与45个VLM（包括GPT‑5、Gemini 2.5 Pro等）对比，原始查询下最高准确率仅48%，显式化后提升7–22点，检索增强虽有增益但不足以弥补模糊；

**⚠️ 局限性**

局限包括极低的筛选率（0.76%）可能遗漏边缘案例、检索增强仅测试OpenAI搜索、错误分析依赖LLM评判器，且模型主要受文化知识缺失而非语言错误影响。

---

## 9. Prompt Engineering for Responsible Generative AI Use in African Education: A Report from a Three-Day Training Series

**arXiv ID:** 2601.06121 | [PDF](https://arxiv.org/pdf/2601.06121v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 10. Model Reconciliation through Explainability and Collaborative Recovery in Assistive Robotics

**arXiv ID:** 2601.06552 | [PDF](https://arxiv.org/pdf/2601.06552v1)

**作者:** Britt Besch `[一作]`, Samuel Bustamante `[通讯]` (Technical University of Munich)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5101779594)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于大型语言模型（LLM）的模型调和框架，在助行机器人共享控制中通过自然语言解释机器人与人类的认知差异并支持模型恢复。

**💡 创新点**

创新点在于将LLM与可视化语言模型（VLM）结合，用来预测并解释机器人与人类的认知差异，而不需完整的用户心理模型，并实现交互式错误恢复。

**🔧 技术方法**

使用了LLM（Mistral‑Small‑3.2‑24B‑Instruct‑2506）与VLM（同一模型），并通过提示工程、链式推理等技术进行搜索匹配、预条件解释和恢复建议。

**📊 数据集**

数据集包括真实机器人实验的30条交互日志、数字双胞胎实验的40条情节（包含图像、世界模型、用户查询），并手工标注对照答案。

**📈 对比分析**

通过与“纯VLM”基线对比，模型调和方法在对象定位解释、预条件错误说明与恢复建议上分别达成100%、78.79%和78.12%的准确率；加入术语翻译词典后，预条件错误解释准确率提升至92.42%。

**⚠️ 局限性**

局限性包括：VLM在空间推理（如运动建议）时精度不足；对形容词的歧义处理不完善；实验场景与对象范围有限，需进一步验证在更复杂日常生活情境中的可用性和可信度。

---

## 11. Two-step Authentication: Multi-biometric System Using Voice and Facial Recognition

**arXiv ID:** 2601.06218 | [PDF](https://arxiv.org/pdf/2601.06218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 12. From Easy to Hard++: Promoting Differentially Private Image Synthesis Through Spatial-Frequency Curriculum

**arXiv ID:** 2601.06368 | [PDF](https://arxiv.org/pdf/2601.06368v1)

**作者:** Chen Gong `[一作]` (University of Virginia), Tianhao Wang `[通讯]` (University of Virginia)

**通讯引用:** 2750 | [OpenAlex ID](https://openalex.org/A5100610986)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于空间-频率学习曲线的 DP 图像合成方法（FETA‑Pro），通过在无公共资源的前提下先用空间中心图像和频率特征做“温度”训练，再利用辅助 GAN 生成频率对齐图像，最终在真实数据上进行 DP‑SGD 微调，从而实现高质量、低泄露风险的合成图像。

**💡 创新点**

创新点在于：① 将空间（中心图像）和频率（随机傅里叶特征）两类“训练捷径”融合进一个从易到难的学习序列；② 解决空间与频率特征训练分歧的难题，采用辅助生成器把频率信息映射到空间域；③ 通过多模型流水线实现对不同特征的并行学习，兼顾扩展性与效率。

**🔧 技术方法**

技术手段包括：DP‑SGD 梯度裁剪+噪声；随机傅里叶特征提取（低频噪声化）；中心图像构造（空间特征）；一阶 GAN 辅助生成器；扩散模型（Diffusion）作为主合成器；RDP/PRV 隐私计算；统一的训练管线。

**📊 数据集**

使用的敏感图像数据集有 MNIST、Fashion‑MNIST、CIFAR‑10、CelebA（人脸）和 Camelyon（医学病理），全部不依赖公开预训练模型或公共数据。

**📈 对比分析**

与 DP-MERF、DP-NTK、DP-Kernel、GS‑WGAN、DP-GAN、DPDM、DP-FETA 等无公共资源基线对比，FETA‑Pro 在 ε=1 时平均 FID 降低 25.7%，分类精度提升 4.1%；在 CIFAR‑10 和 Camelyon 上分别提高 20.2% 和 41.2% 的 FID；收敛速度更快、训练时间仅增加 0.3%。

**⚠️ 局限性**

局限性包括：① 对隐私预算分配极度敏感，需经验调参；② 对极大分辨率或非图像模态的推广尚未验证；③ 仍存在模型偏差和潜在误用风险；④ 需要辅助 GAN 的额外实现，若生成器表现不佳会影响整体性能。

---

## 13. TIR-Flow: Active Video Search and Reasoning with Frozen VLMs

**arXiv ID:** 2601.06176 | [PDF](https://arxiv.org/pdf/2601.06176v1)

**作者:** Hongbo Jin `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**通讯引用:** 13766 | [OpenAlex ID](https://openalex.org/A5100447673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了训练无关的 TIR-Flow 框架，用于提升冻结视频‑语言模型在长时序视频推理任务中的表现。

**💡 创新点**

通过将推理拆解为假设驱动分解、主动感知和证据黑板三模块，实现“计划‑观察‑验证”的 System‑2 思维，嵌入主动搜索以突破信息冗余与感知瓶颈。

**🔧 技术方法**

结合 LLM 规划、CLIP 双编码器的时空检索、网格金字塔裁剪以及证据仲裁与动态工作空间等技术，且不对模型参数进行更新。

**📊 数据集**

在七个公开 VideoQA 基准（EgoSchema、VideoEspresso、VideoMMMU、MVBench、VideoMME 等）上进行评测。

**📈 对比分析**

与 GPT‑4o、Gemini、VideoLLaMA3、SmartSight 等基线对比，平均提升约5.9%，在 EgoSchema 上提升10.5%，在其他任务也实现显著性能提升。

**⚠️ 局限性**

受限于模型规模、对基础模型规划质量的依赖以及推理迭代导致的速度下降；未验证更大规模模型的可扩展性。

---

## 14. BotSim: Mitigating The Formation Of Conspiratorial Societies with Useful Bots

**arXiv ID:** 2601.06154 | [PDF](https://arxiv.org/pdf/2601.06154v1)

**作者:** Lynnette Hui Xian Ng `[一作]`, Kathleen M. Carley `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文构建了一个仿真模型，用于研究不同类型机器人（坏机器人、信息纠正机器人、好机器人）对人类信息消费队列质量和“坏人类占多数”时间的影响。

**💡 创新点**

创新点在于引入三类机器人并通过响应面分析寻找使“坏人类占多数”所需时间最大化的最优比例组合，表明信息纠正机器人与好机器人在提升信息质量方面更具资源效益。

**🔧 技术方法**

主要技术包括离散事件仿真模型、比例参数化、响应面分析（Response Surface Methodology）以及时间到达统计测量。

**📊 数据集**

使用的不是公开真实数据集，而是基于模型设定生成的合成模拟数据；人类、机器人被赋予不同颜色和形状以可视化演示。

**📈 对比分析**

通过在不同坏机器人、信息纠正机器人和好机器人的比例下运行仿真，并统计“坏人类占多数”所需时间，进行比较。结果显示，在坏机器人与信息纠正机器人比例为1:1、坏机器人与好机器人比例为1:0.5时，时间达到峰值，说明此组合最有效。

**⚠️ 局限性**

局限性包括：1）仿真假设过于简化，未考虑真实社交网络的复杂结构与动态；2）仅考虑三类机器人，未覆盖多样化的机器人行为；3）缺乏对真实世界数据的验证，难以直接推广到实际平台。

---

## 15. LLM-Powered Social Digital Twins: A Framework for Simulating Population Behavioral Response to Policy Interventions

**arXiv ID:** 2601.06111 | [PDF](https://arxiv.org/pdf/2601.06111v1)

**作者:** Aayush Gupta `[一作]` (PricewaterhouseCoopers), Farahan Raza Sheikh `[通讯]` (PricewaterhouseCoopers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于大型语言模型的社交数字孪生框架，用于预测政策对人群行为的影响，并在COVID‑19 期间对其进行了严格验证。

**💡 创新点**

创新点在于将LLM作为个体决策引擎，生成多维度行为概率；通过校准层将个体输出映射到可观测指标，实现可解释、可对照假设的行为模拟。

**🔧 技术方法**

使用的技术包括大型语言模型（Gemini、GPT‑4 等）进行认知推理、线性或多目标校准映射、梯度提升回归、时间序列分割与评估、以及 Optuna 优化。

**📊 数据集**

采用的主要数据集有 Oxford COVID‑19 Government Response Tracker、Google Mobility Reports、以及人口普查与问卷调查数据，用于构建人物属性、政策信号和验证指标。

**📈 对比分析**

通过严格的训练/验证/测试时间拆分，与持久性预测和梯度提升基线在六类移动行为上进行对比，数字孪生在宏观平均 RMSE 上提升 20.7%，在政策敏感行为上明显优于统计模型。

**⚠️ 局限性**

主要局限包括对惯性行为预测不足、对校准数据的高度依赖、LLM 推理成本较高、缺乏因果识别、以及仅在 10 个虚拟人物上验证，需扩大规模并改进记忆机制。

---

## 16. zkRansomware: Proof-of-Data Recoverability and Multi-round Game Theoretic Modeling of Ransomware Decisions

**arXiv ID:** 2601.06667 | [PDF](https://arxiv.org/pdf/2601.06667v1)

**作者:** Xinyu Hou `[一作]` (University of Science and Technology of China), Weidong Shi `[通讯]` (University of Houston)

**通讯引用:** 25563 | [OpenAlex ID](https://openalex.org/A5041067396)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文设计并实现了一种基于零知识证明和智能合约的 zkRansomware，能够在攻击过程中提供可验证的数据恢复和多轮加密支付机制，同时提出相应的博弈论决策模型。

**💡 创新点**

创新点包括：①将 zk‑SNARK、可验证加密（VECK）与公平数据交换（FDE）技术整合到勒索软件中，实现数据恢复可验证性；②利用智能合约实现多轮支付与自动退款，降低受害者的风险；③构建新的博弈论决策模型，揭示攻击者与受害者在多轮支付与数据泄露风险下的均衡行为。

**🔧 技术方法**

核心技术包括：零知识简洁交互式知识论证（zk‑SNARK）、可验证加密与证明（VECK）、公平数据交换协议、以太坊 Solidity 智能合约、时间锁（time‑lock）以及 ElGamal/Paillier 加密和证明生成/验证工具。

**📊 数据集**

实验使用了合成数据集：数据价值服从均匀分布 U(250,350)、U(550,650)、U(950,1050) 以及 U(200,1200) 等多种分布，用以模拟不同勒索情景。

**📈 对比分析**

通过与传统单轮支付模型、完美与最差声誉情景比较，利用仿真与线性规划求解攻击者期望收益，结果显示多轮支付在受害者支付意愿提高时能显著提升攻击者收益；性能方面，512 KB 数据加密+证明生成约 239 s，证明验证约 91 s，合约交易 gas 成本低于 8 USD。

**⚠️ 局限性**

局限性包括：加密与证明产生的计算与存储开销较大，主要适用于高价值数据；模型假设参与方理性且及时发现泄露，现实中可能出现误判；仅实现了一种 FDE 方案，未覆盖所有可行实现；依赖区块链的可用性与安全，若链下层遭受攻击可能失效。

---

## 17. ConSensus: Multi-Agent Collaboration for Multimodal Sensing

**arXiv ID:** 2601.06453 | [PDF](https://arxiv.org/pdf/2601.06453v1)

**作者:** Hyungjun Yoon `[一作]` (KAIST), Lorena Qendro `[通讯]` (Nokia Bell Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个训练自由的多代理协作框架，将多模态感知任务分解为模态专属代理，并通过混合融合实现鲁棒推理。

**💡 创新点**

创新点在于将语义聚合与统计共识融合为单轮混合融合，平衡先验知识偏差与传感器失效风险。

**🔧 技术方法**

采用大型语言模型（如 LLaMA 7B/13B）、模态特定代理、语义融合代理、统计融合代理及混合融合代理；无监督的prompting和角色化协作。

**📊 数据集**

在五个多模态感知基准上评估，包括 WESAD、SleepEDF、ActionSense、MMFit 与 PAMAP2。

**📈 对比分析**

与单代理、Self‑Consistency、Self‑Refine 及多轮辩论基线对比，平均提升 7.1% 识别准确率，同时单轮混合融合将融合令牌消耗降低 12.7×。

**⚠️ 局限性**

实验规模受限于计算资源，只覆盖分类任务，未涉及长序列、罕见模态或人类主观评价，且未加入高级prompt或fine‑tuning。

---

## 18. Stylistic Evolution and LLM Neutrality in Singlish Language

**arXiv ID:** 2601.06580 | [PDF](https://arxiv.org/pdf/2601.06580v1)

**作者:** Linus Tze En Foo `[一作]` (Independent Researcher), Lynnette Hui Xian Ng `[通讯]` (Carnegie Mellon University)

**通讯引用:** 588 | [OpenAlex ID](https://openalex.org/A5001523273)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Singlish在2012-2021年间通过短消息的词汇、句法及语用演变，并评估大型语言模型（LLM）在生成Singlish文本时的时间敏感性。

**💡 创新点**

提出一种可解释的风格相似度框架，融合手工设计的词汇-结构、语用与心理语言学特征与深度编码器（all‑MiniLM）嵌入，量化语料随时间的差异；同时首次系统比较LLM生成文本在时间维度上的偏差。

**🔧 技术方法**

采用Gradient Boosting分类器进行二元时间区分，使用SHAP进行特征重要性解释，利用PCA分析LIWC心理语言特征；LLM使用Qwen、Mistral、SeaLLM、DeepSeek四款模型，结合Zero‑Shot、CoT、DD、SC四种提示，并对部分模型进行LoRA微调。

**📊 数据集**

核心数据集为CoSEM（2012‑2021年短信集合）作为真实文本；生成文本通过上述四款LLM和提示策略产生的1,000条样本。

**📈 对比分析**

通过将分类准确率转化为相似度S（0为最不相似，1为最相似）并计算跨年标准差衡量时间中性。结果显示：真实文本随年距增大相似度显著下降，证明语料具有可测的时间差异；LLM生成文本虽在相似度上接近真实，但标准差高，表明其输出仍带有明显时间指纹，微调对时间中性影响有限。

**⚠️ 局限性**

研究仅聚焦于短信式文本，难以推广至社交媒体等其他书面语境；LLM在捕捉社会语言细微语用特征上仍表现不足。

---

## 19. UMLoc: Uncertainty-Aware Map-Constrained Inertial Localization with Quantified Bounds

**arXiv ID:** 2601.06602 | [PDF](https://arxiv.org/pdf/2601.06602v1)

**作者:** Mohammed S. Alharbi `[一作]` (King Abdullah University of Science and Technology), Shinkyu Park `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 707 | [OpenAlex ID](https://openalex.org/A5038347119)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了UMLoc框架，利用IMU量测与地图约束实现无漂移的室内定位。

**💡 创新点**

通过LSTM量化预测区间与基于交叉注意力的CGAN联合生成地图可行轨迹，并给出置信区间。

**🔧 技术方法**

采用LSTM量化回归、CGAN、交叉注意力、欧几里得距离变换的地图编码以及自监督+对抗训练。

**📊 数据集**

在自建的室内IMU+地图数据集以及RoNIN和RNIN公开数据集上进行评测。

**📈 对比分析**

与RoNIN LSTM/TCN、RNIN‑VIO等基线对比，UMLoc 在ATE/FDE/RTE 上提升约50%~65%，且在跨建筑和鲁棒性测试中表现更优。

**⚠️ 局限性**

仍受限于二维定位、对姿态估计依赖GRV、以及训练需要地图信息，且在极端噪声下仍可能漂移。

---

## 20. A Foundation Model Approach for Fetal Stress Prediction During Labor From cardiotocography (CTG) recordings

**arXiv ID:** 2601.06149 | [PDF](https://arxiv.org/pdf/2601.06149v1)

**作者:** Naomi Fridman `[一作]` (Ariel University), Berta Ben Shachar `[通讯]` (NF Algorithms and AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文通过自监督掩码预训练方法，对大量未标记的产检心电图（CTG）信号进行表示学习，并在CTU-UHB标注子集上微调，实现胎儿酸中毒的二分类预测。

**💡 创新点**

创新点在于首次将基于Transformer的自监督掩码预训练与通道异构掩码策略相结合，用以学习FHR与宫缩信号之间的关联，并通过滑动窗口实时生成风险预警。

**🔧 技术方法**

技术手段包括PatchTST Transformer架构、通道独立的自注意力模块、FHR掩码重建任务、以及滑动窗口推理与逻辑回归二级警报判定。

**📊 数据集**

所用数据集为CTGDL，包含2,444小时未标记的CTG记录（来自CTU-UHB、FHRMA、SPaM）以及552条带有脐动脉pH标签的CTU-UHB子集。

**📈 对比分析**

与以往在CTU-UHB基准上基于监督学习的模型（AUC 0.68–0.75）对比，本文在全测试集上获得0.83的AUC，在无产褥期阴道分娩子集中达到0.853，明显提升。

**⚠️ 局限性**

局限性包括标注数据来源单一中心、缺乏孕产科临床背景信息、模型对不同监测仪器的泛化能力有限，以及在小样本情况下仍面临标签稀缺与过拟合风险。

---

## 21. SimLLM: Fine-Tuning Code LLMs for SimPy-Based Queueing System Simulation

**arXiv ID:** 2601.06543 | [PDF](https://arxiv.org/pdf/2601.06543v1)

**作者:** Jun-Qi Chen `[一作]` (Renmin University of China), Ying Zhong `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 601 | [OpenAlex ID](https://openalex.org/A5064533699)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对排队系统仿真代码生成问题，提出多阶段数据构造与微调框架，并使用开源LLM自动生成可执行的SimPy仿真脚本。

**💡 创新点**

创新点在于设计了 Category‑Template Framework（CTF）实现指令与代码的多样化生成，以及三阶段训练（SFT→掩码补全→DPO）提升模型对排队逻辑的理解与一致性。

**🔧 技术方法**

使用的技术包括监督微调、基于掩码的代码补全训练、直接偏好优化（DPO）以及SimPy仿真环境下的自动执行与评估。

**📊 数据集**

数据集由CTF自动生成的 7,200 条指令‑代码对、20,000 条掩码补全样本以及 380/420 条 DPO 偏好对组成，覆盖 12 类排队场景。

**📈 对比分析**

在 600 题测试集上评估可执行率、输出格式一致性和指令‑代码一致性，Qwen2.5‑Coder‑7B 从 86% 提升至 76.8% 的一致性，DeepSeek‑Coder‑6.7B 从 75% 提升至 62.3%，相较基线显著提升。

**⚠️ 局限性**

仍存在网络路由和中断恢复等复杂机制错误，且对极复杂场景的提升有限，需进一步数据增强与目标调优。

---

## 22. L-RAG: Balancing Context and Retrieval with Entropy-Based Lazy Loading

**arXiv ID:** 2601.06551 | [PDF](https://arxiv.org/pdf/2601.06551v1)

**作者:** Sergii Voloshyn `[一作]` `[通讯]` (Taras Shevchenko National University of Kyiv), Sergii Voloshyn (Taras Shevchenko National University of Kyiv)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了L-RAG（Lazy Retrieval-Augmented Generation）框架，通过在生成时先使用文档摘要，再根据模型的不确定性（熵）决定是否进行昂贵的检索；

**💡 创新点**

创新点在于利用语言模型自身的预测熵作为无训练、无额外模块的检索触发信号，并构建两层上下文架构（摘要+检索块）实现自适应检索；

**🔧 技术方法**

采用熵基门控（entropy-based gating）、两层上下文策略、向量检索（句子Transformer + FAISS）和Phi-2 LLM；

**📊 数据集**

使用SQuAD 2.0数据集（500个可答问题）进行评估；

**📈 对比分析**

与四种基线对比：无检索、Standard RAG、Strong RAG和Oracle。L-RAG(τ=0.5)精确率78.2%，与Standard RAG(77.8%)差异不显著，同时检索率比Standard RAG低8%；τ=1.0时精确率76.0%，检索率降至74%；Strong RAG最高79.8%；

**⚠️ 局限性**

局限性包括样本量有限（500例）、仅在单一数据集和单一模型（Phi‑2）验证、对“自信错误”无响应、摘要质量依赖、以及熵分布重叠导致的门控不完美。

---

## 23. B-FIRE: Binning-Free Diffusion Implicit Neural Representation for Hyper-Accelerated Motion-Resolved MRI

**arXiv ID:** 2601.06166 | [PDF](https://arxiv.org/pdf/2601.06166v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 24. Certified Unlearning in Decentralized Federated Learning

**arXiv ID:** 2601.06436 | [PDF](https://arxiv.org/pdf/2601.06436v1)

**作者:** Hengliang Wu `[一作]` (Shandong University), Dongxiao Yu `[通讯]` (Shandong University)

**通讯引用:** 4058 | [OpenAlex ID](https://openalex.org/A5045982340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在去中心化联邦学习中实现机器消除（unlearning）的框架，利用Newton式更新与 Fisher 信息矩阵近似实现可证的（ϵ,δ）消除；

**💡 创新点**

创新点在于：①在去中心化网络中无需记录历史梯度即可量化并消除已传播的样本/客户端影响；②通过高斯噪声和统一广播实现无中心化的可证消除；③提供了严格的理论证明与实用的内存/计算效率；

**🔧 技术方法**

技术包括：分布式随机梯度下降（DSGD）、影响函数与Newton式近似、Fisher信息矩阵的稀疏化、加噪声机制（Gaussian）和单轮微调；

**📊 数据集**

使用CIFAR‑10、MNIST、Fashion‑MNIST三大公开数据集，在不同图拓扑（环形、Erdős‑Rényi）下进行实验；

**📈 对比分析**

与完整重新训练（RT）及最近的PDUDT对比，实验显示：在样本/类别/客户端删除场景下，模型精度接近RT，仅损失1–3个百分点，MIA攻击准确率降至≈50%；相较于RT，耗时下降约97%，PDUDT需多轮微调；

**⚠️ 局限性**

局限性：仅在凸或强凸目标下理论和实验验证；对非凸网络（如深度CNN）缺乏正式证明；需假设混合矩阵对称双随机且网络连通；

---

## 25. Time-Series Anomaly Classification for Launch Vehicle Propulsion Systems: Fast Statistical Detectors Enhancing LSTM Accuracy and Data Quality

**arXiv ID:** 2601.06186 | [PDF](https://arxiv.org/pdf/2601.06186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 26. Are Emotions Arranged in a Circle? Geometric Analysis of Emotion Representations via Hyperspherical Contrastive Learning

**arXiv ID:** 2601.06575 | [PDF](https://arxiv.org/pdf/2601.06575v1)

**作者:** Yusuke Yamauchi `[一作]` (University of Tokyo), Akiko Aizawa `[通讯]` (National Institute of Informatics)

**通讯引用:** 5147 | [OpenAlex ID](https://openalex.org/A5041062417)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对语言模型情感表示进行对比学习，在超球面上诱导圆形情绪结构，并评估其与心理学圆形模型的对应性与分类性能。

**💡 创新点**

提出 CircularCSE 损失，利用 nGPT 头在超球面上实现情绪的圆形几何；通过与 SINCERE、SoftCSE 的对比，揭示心理学可解释性与高维判别性之间的权衡。

**🔧 技术方法**

对比学习损失（SINCERE、SoftCSE、CircularCSE）；超球面 Transformer 头（nGPT）与传统 GPT 头；Spherical k‑means 聚类、MDS/PCA 降维、CD‑r Pearson 相关评估。

**📊 数据集**

Emolit、Empathetic Dialogue、SuperEmotion 以及合成 PersonaGen 数据集，共覆盖 12 个情绪标签。

**📈 对比分析**

采用 V‑Measure 和 CD‑r 两指标进行比较；SINCERE/SoftCSE 在 V‑Measure 上优于 CircularCSE；CircularCSE 在 CD‑r 上最佳；在低维或标签数少的情景下 CircularCSE 稳定，随维度/标签增大性能明显下降。

**⚠️ 局限性**

仅处理单标签情绪，未覆盖多重情绪或中性状态；仅文本域，缺乏多模态考量；情绪标签与真实心理模型映射存在差异；对更高维或更复杂情绪模型的推广仍受限。

---

## 27. Leveraging Foundation Models for Calibration-Free c-VEP BCIs

**arXiv ID:** 2601.06028 | [PDF](https://arxiv.org/pdf/2601.06028v1)

**作者:** Mohammadreza Behboodi `[一作]` (University of Calgary), Hatem Abou-Zeid `[通讯]` (University of Calgary)

**通讯引用:** 1280 | [OpenAlex ID](https://openalex.org/A5027792197)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用预训练的EEGPT基础模型，在c-VEP BCI 中实现无校准或仅需少量校准的解码方案，并评估其在两套公开数据集上的性能。

**💡 创新点**

首次将大规模EEG基础模型迁移到c-VEP领域，提出真正的校准‑free 方法，并证明仅用 20% 个人数据即可达到或超过传统全校准性能；方法对不同序列与校准策略均具通用性。

**🔧 技术方法**

使用 EEGPT 作为冻结编码器，搭配两层线性任务头；训练策略包括 Calibration‑Free、Limited Calibration（少量个人数据微调）以及 Within‑Subject；实验采用 LOSO 交叉验证。

**📊 数据集**

Fast‑Stim（17 位受试者，32 目标，Ensemble 代码）与 Group‑Mod（12 位受试者，16 目标，Circular‑Shift 代码）两套公开 c‑VEP 数据集。

**📈 对比分析**

与原始研究全校准结果对比：Calibration‑Free 在 Fast‑Stim 上平均 71.8%±13.3%（原始 66.2%±13.8%），在 Group‑Mod 上 71.8%±20.2%（原始 93.7%±5.5%）；Limited Calibration 仅用 20% 校准数据即可取得 92%±5.2%（Fast‑Stim）和 89.5%±7.8%（Group‑Mod），近似或超越原始全校准性能。

**⚠️ 局限性**

仅评估成人数据，缺乏儿童样本；EEGPT 模型体积大、计算资源高，难以部署在低资源设备；需要足够数量的相同实验设置的受试者来训练任务头。

---

## 28. Using street view images and visual LLMs to predict heritage values for governance support: Risks, ethics, and policy implications

**arXiv ID:** 2601.06056 | [PDF](https://arxiv.org/pdf/2601.06056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 29. Learning Domain Agnostic Latent Embeddings of 3D Faces for Zero-shot Animal Expression Transfer

**arXiv ID:** 2601.06484 | [PDF](https://arxiv.org/pdf/2601.06484v1)

**作者:** Yue Wang `[一作]` (Stony Brook University), Xianfeng Gu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种零射（zero‑shot）框架，利用人类面部表情数据训练，能够将人类表情无监督地迁移到动物3D面部网格上。

**💡 创新点**

创新点在于将内在几何描述（HKS/WKS）与 DiffusionNet 编码相结合，生成网格无关的潜在嵌入，并通过注意力融合源表情与目标身份，再用神经雅可比场预测局部变形，从而实现跨物种、无对齐、无表达监督的表情迁移。

**🔧 技术方法**

使用的核心技术包括：内在谱特征提取（HKS/WKS）、DiffusionNet 处理、注意力特征融合、神经雅可比场（Neural Jacobian Field）以及联合顶点位置与雅可比损失的监督。

**📊 数据集**

训练数据仅来自人类面部表达模型 ICT Face Model，合成 1000 组三角网格（身份、表情、目标表达），在推理阶段测试猫等动物面部网格（由 CAFM 生成）。

**📈 对比分析**

与现有的 NFR（Neural Face Rigging）对比，实验显示在无中性表达输入和无对齐条件下，本方法能更准确、自然地迁移多样表情，并在动物网格上产生可解释的解剖变形；客观指标（如顶点重建误差）亦优于 NFR。

**⚠️ 局限性**

限制包括：仅在静态帧测试；对非常极端或完全不同解剖结构的动物（如蛇、鱼）可能不具备良好泛化；缺乏动态序列和实时推理的评估。

---

## 30. Algorithms for Computing the Petz-Augustin Capacity

**arXiv ID:** 2601.06492 | [PDF](https://arxiv.org/pdf/2601.06492v1)

**作者:** Chun-Neng Chu `[一作]` (National Taiwan University), Yen-Huan Li `[通讯]` (National Taiwan University)

**通讯引用:** 221 | [OpenAlex ID](https://openalex.org/A5089741364)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究并实现了两类算法，分别基于 Hölder 光滑优化与 Blahut‑Arimoto 型迭代，以非渐近收敛保证的方式计算经典‑量子通道的 Petz‑Augustin 容量。

**💡 创新点**

创新点在于首次为 Petz‑Augustin 容量提供非渐近收敛保证，将 Hölder 光滑优化引入量子信息理论，提出基于 Thompson 度量的固定点算法，并将相对光滑性与镜像下降相结合实现 Blahut‑Arimoto 型求解。

**🔧 技术方法**

主要技术包括 Hölder 光滑优化（FISTA 等）、相对光滑性与镜像下降、Thompson 度量下的收敛性分析、固定点迭代、矩阵分析工具以及 Petz‑Rényi 与 Petz‑Augustin 信息的量子计算。

**📊 数据集**

实验使用 Python QuTiP 生成的随机量子态构成的通道，输入符号规模 n=2^7，输出维度 d=2^5，并在此数据集上评估算法性能。

**📈 对比分析**

与传统 Blahut‑Arimoto 算法和 FGM‑1e‑9 进行比较；在 α=0.6 时 FGM‑Balanced 收敛速度最快；在 α=0.9 时 Blahut‑Arimoto 型算法更快；FGM‑1e‑9 在两者中表现最好，整体收敛率与理论预测一致。

**⚠️ 局限性**

局限性包括：仅针对 α∈[1/2,1) 进行分析；对 α<1/2（球包极限）尚未解决；相对光滑参数可能随最小特征值而无界；实验主要基于随机生成的通道，缺乏真实量子通道的验证。

---

## 31. TCLNet: A Hybrid Transformer-CNN Framework Leveraging Language Models as Lossless Compressors for CSI Feedback

**arXiv ID:** 2601.06588 | [PDF](https://arxiv.org/pdf/2601.06588v1)

**作者:** Zijiu Yang `[一作]` (Zhejiang University), Zhiguo Shi `[通讯]` (Zhejiang University)

**通讯引用:** 8570 | [OpenAlex ID](https://openalex.org/A5041940889)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了基于Transformer‑CNN混合结构的CSI压缩网络TCLNet，并在压缩后引入LM–FM混合无损编码，实现了高效的CSI反馈。

**💡 创新点**

创新点包括并行的Transformer‑CNN特征提取、动态符号选择的LM/ FM无损压缩框架以及利用预训练LLM的零样本压缩方法。

**🔧 技术方法**

技术手段主要有TransConv模块、Swim Transformer、上下文感知语言模型、因子化模型、算术编码与LLM提示工程。

**📊 数据集**

实验数据集涵盖真实的Argos室内数据和COST2100模拟室内/室外数据。

**📈 对比分析**

与CNN、TransNet、CsiNet+等基线在不同压缩比下比较，TCLNet在NMSE上提升1–4 dB，FLOPs仅为纯Transformer的约⅓，同时LLM零样本压缩可接近训练模型的压缩率。

**⚠️ 局限性**

局限性在于仍需较高计算资源（尤其是LLM推理），且缺乏在更复杂多用户或高速多变信道环境下的鲁棒性验证。

---

## 32. Dynamics-inspired Structure Hallucination for Protein-protein Interaction Modeling

**arXiv ID:** 2601.06214 | [PDF](https://arxiv.org/pdf/2601.06214v1)

**作者:** Fang Wu `[一作]` (Stanford University), Stan Z. Li `[通讯]` (Westlake University)

**通讯引用:** 47694 | [OpenAlex ID](https://openalex.org/A5082786719)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Refine-PPI框架，联合预测突变产生的蛋白-蛋白相互作用能量变化及其结构；

**💡 创新点**

创新点在于①基于mask mutation modeling的结构重建模块，能在缺失突变体结构时自动生成；②引入概率密度云网络(PDC-Net)捕捉原子运动的不确定性与动态性；

**🔧 技术方法**

采用等变形图神经网络(如EGNN)配合PDC网络，结合Huber损失、Masking任务和多任务训练；

**📊 数据集**

主要使用SKEMPI.v2数据集进行训练和评估，预训练基于PDB-REDO结构；同时在ProteinGym替换基准上做零射击评测；

**📈 对比分析**

与Rosetta、FoldX、DDGPred、ESM-1v、PPIFormer、MIF-Net等多类基线对比，Refine-PPI在per-structure Pearson/Spearman、RMSE、AUROC及ProteinGym Spearman等指标均名列前茅；

**⚠️ 局限性**

局限性包括仅局部重建而非全局结构预测、未显式建模侧链细节、mask范围有限、对全复杂结构变化的适应性不足。

---

## 33. Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence

**arXiv ID:** 2601.06364 | [PDF](https://arxiv.org/pdf/2601.06364v1)

**作者:** Xiaotian Zhang `[一作]` (Northeastern University), Xiaozhong Liu `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3763 | [OpenAlex ID](https://openalex.org/A5101985030)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种临床医生与AI协同的单页界面，AI负责将慢性病管理所需数据组织成结构化草稿，医生进行快速认知式审核并最终发布；

**💡 创新点**

创新点在于将AI的角色限定为“数据准备”而非决策制定，并通过单页布局、图表配对、紧急度标记等交互模式实现效率与可追溯性的平衡，同时揭示了高风险环境下的“责任悖论”；

**🔧 技术方法**

技术实现包括使用Qwen3‑8B LLM进行草稿生成、结合自动紧急度评估（LLM+规则基检查）、单页HTML编辑器以及可视化图表；

**📊 数据集**

使用了真实临床病例数据，共计24例慢性病管理报告（14例高危、8例注意、2例稳定），包含药物清单、设备趋势和医患对话；

**📈 对比分析**

通过对比医生手工撰写的基准（评分5.0）进行量化评估，平均质量评分为4.86/10，编辑比例仅8.3%，无安全关键问题，但时间节省未显著提升；

**⚠️ 局限性**

局限性在于责任归属导致医生仍需完整核查，系统无法真正降低工作时长；AI仅负责草稿而非决策，缺乏大规模验证与保险/法律适配机制。

---

## 34. DeeperBrain: A Neuro-Grounded EEG Foundation Model Towards Universal BCI

**arXiv ID:** 2601.06134 | [PDF](https://arxiv.org/pdf/2601.06134v1)

**作者:** Jiquan Wang `[一作]` (Zhejiang University), Gang Pan `[通讯]` (Zhejiang University)

**通讯引用:** 11276 | [OpenAlex ID](https://openalex.org/A5084291326)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于神经科学先验的EEG基础模型DeeperBrain，用于实现通用脑机接口；

**💡 创新点**

创新点包括：①将体积传导和多尺度神经动力学作为空间、时间位置编码的生物物理先验；②引入双重自监督目标——Masked EEG Reconstruction与Neurodynamics Statistics Prediction，强制模型同时保留细粒度波形与宏观脑状态；

**🔧 技术方法**

技术主要是Transformer编码器、卷积时间嵌入、基于距离衰减的通道位置编码、基于慢振荡与指数衰减的时间位置编码、Masked重建与统计预测损失；

**📊 数据集**

预训练使用了14个公开EEG数据集，累计超过17,200小时（约2.4M个非重叠样本），下游评估涉及10种不同任务与数据集（情绪识别、运动想象、睡眠分期、癫痫检测、语音想象等）；

**📈 对比分析**

与传统EEGNet、EEGConformer以及现有EEG基础模型（LaBraM、CBraMod、CSBrain、REVE）比较，DeeperBrain在大多数任务上取得最高或竞争性最高的精度，且在冻结探测（frozen‑probing）条件下仍保持显著优势；

**⚠️ 局限性**

局限性包括：依赖已知的3D电极坐标，未处理电极布局变化的细粒度建模；NSP仅包含四个传统统计，可能无法覆盖更复杂的动力学特征；预训练数据存在人口和地理偏倚，未来需更广泛的数据覆盖。

---

## 35. An Exploratory Pilot Survey on Technical Quality Control Practices in Agile R&D Projects

**arXiv ID:** 2601.06689 | [PDF](https://arxiv.org/pdf/2601.06689v1)

**作者:** Mateus Costa Lucena `[一作]` `[通讯]` (Venturus R&D Institute), Mateus Costa Lucena (Venturus R&D Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对巴西马纳斯州科研机构（STIs）中使用Scrum的敏捷研发软件项目进行了一项探索性问卷调查，收集并描述了技术质量控制实践、指标使用、面临的挑战和技术债务影响。

**💡 创新点**

首次在区域创新生态系统内提供敏捷研发项目技术质量管理的经验性基线，揭示了实践应用不一致、指标监控不足以及技术债务与业务层面缺乏衔接的关键问题，为后续深入研究奠定了参考框架。

**🔧 技术方法**

使用在线结构化问卷（包含闭合和开放式问题）进行数据收集；对问卷结果进行描述性统计（频率分布）与主题分析（对开放式回答进行轻量化归类）。

**📊 数据集**

自报问卷数据：17名参与者（包括开发者、项目负责人、经理等多角色），数据已公开发布于Google Sheets；无外部公开数据集使用。

**📈 对比分析**

通过描述性统计展示各技术质量控制工具和指标的使用频率，并结合主题分析阐释主要挑战；未进行实验对比或性能评估。

**⚠️ 局限性**

局限性：样本量小（N=17），仅限于马纳斯州单一创新生态系统；采用便利抽样与自报数据，存在记忆偏差和主观性；结果为描述性而非可推广的统计结论。

---

## 36. Random is Faster than Systematic in Multi-Objective Local Search

**arXiv ID:** 2601.06318 | [PDF](https://arxiv.org/pdf/2601.06318v1)

**作者:** Zimin Liang `[一作]` (University of Birmingham), Miqing Li `[通讯]` (University of Birmingham)

**通讯引用:** 7364 | [OpenAlex ID](https://openalex.org/A5036335232)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究多目标局部搜索中的两种邻域探索策略，系统性探索（s-PLS）与随机采样（r-PLS），并证明后者在大多数多目标组合优化问题上更快；

**💡 创新点**

创新点在于通过经验与理论相结合，首次证明随机采样在多目标局部搜索中比系统性探索更高效，并揭示两者在邻域结构上的根本差异；

**🔧 技术方法**

主要技术包括基于Pareto支配的局部搜索框架、邻域结构分析、几何分布拟合、期望评估次数理论推导以及对比实验；

**📊 数据集**

使用四类典型多目标组合优化问题：0/1背包、旅行商、二次分配和NK景观，规模分别为100、200、500变量；

**📈 对比分析**

通过在每个问题上运行30次实验，比较r-PLS与s-PLS在Hypervolume（HV）指标上的轨迹，结果显示r-PLS在早期即显著优于s-PLS，差距随问题规模增大而加大；

**⚠️ 局限性**

局限性包括：只考虑了最基础的s-PLS与r-PLS版本；未讨论搜索终止后总体解质量；在某些特殊的伪布尔基准（如OneMinMax）中系统性搜索可能更快，理论分析未覆盖这些情形。

---

## 37. Channel Knowledge Map Construction via Guided Flow Matching

**arXiv ID:** 2601.06156 | [PDF](https://arxiv.org/pdf/2601.06156v1)

**作者:** Ziyu Huang `[一作]` (Southeast University), Hongyang Du `[通讯]` (University of Hong Kong)

**通讯引用:** 5765 | [OpenAlex ID](https://openalex.org/A5068782412)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于线性传输引导流匹配（LT-GFM）的渠道知识图谱（CKM）构建框架，实现高效、精确的 CGM 与 SCM 生成。

**💡 创新点**

创新点包括：①将 CKM 生成过程建模为沿直线轨迹的确定性 ODE，显著减少采样步骤；②融合环境语义（建筑掩模、边缘信息）和 Hermitian 对称约束，保证生成结果的物理一致性与高频细节；③通过一次 ODE 求解即可完成样本生成，推理速度提升约 25 倍。

**🔧 技术方法**

使用技术：线性传输流匹配、确定性 ODE 采样、U‑Net 结构、环境语义条件化、Hermitian 对称化后处理、FID 与 MSI 等分布与结构度量。

**📊 数据集**

数据集：CKMImageNet（由 Wireless InSite 仿真软件生成的 CGM 与 SCM 数据）。

**📈 对比分析**

对比方法：Bicubic/Linear 插值、KNN、基线 UNet 回归、DDPM 生成模型。性能表现：CGM 的 NMSE 下降至 0.0021，SSIM 提升至 0.8341，FID 下降至 61.02，推理时间 417 ms（比 DDPM 快 25×）；SCM 的 MSI 达到 0.90，FID 降至 1.40，显著优于 UNet 与 DDPM。

**⚠️ 局限性**

限制：①对极低采样率或极大规模场景的适应性尚待验证；②训练依赖大量真实或高质量仿真数据；③虽然采样步骤大幅减少，但仍需平衡步数与精度，尤其在复杂环境下可能需要更细粒度的 ODE 采样。

---

## 38. DRAGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization

**arXiv ID:** 2601.06502 | [PDF](https://arxiv.org/pdf/2601.06502v1)

**作者:** Shengkai Chen `[一作]` (Institute for Infocomm Research), Shili Xiang `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5068002105)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 DRAGON 框架，利用大语言模型（LLM）通过分解-重构的迭代过程，针对大规模组合优化问题（如 TSP、CVRP、BPP、MKP）生成高质量解。

**💡 创新点**

创新点在于：①将 LLM 作为分解器，自动识别可改进子结构；②通过重构器在约束下本地优化子问题，整体实现无手工算法的可扩展求解；③框架采用状态传递通信，兼顾全局一致性与局部可控性。

**🔧 技术方法**

技术手段包括：prompt‑engineering 生成分解与重构指令；LLM 作为推理代理完成子问题求解；与 OR‑Tools 交互进行约束验证；多轮状态传递实现迭代优化。

**📊 数据集**

数据集涵盖：TSPLIB（50–20k 节点）、CVRPLIB（X 与 XML 类型）、FunSearch/BPP Weibull‑5k、人工合成的 MKP（10–100 背包、10–1,000 项），覆盖路由、装箱、背包多种典型 COP。

**📈 对比分析**

在与 OPRO、SGE、LMEA、ReEvo（a/c）以及 OR‑Tools 的比较中，DRAGON 在大规模实例（≥500 节点）上平均优化缺口比对手低约1–3%，同时保持可接受的运行时间；但在极小规模实例或非常长的 token 输入下仍面临性能下降。

**⚠️ 局限性**

主要局限包括：对 prompt 设计和分解策略高度敏感；迭代过程导致计算与 API 调用开销显著；大规模实例仍受 LLM 输入 token 限制；缺乏自动化、自适应的分解机制，导致在某些结构复杂的 COP 上表现不稳定。

---

## 39. QCaption: Video Captioning and Q&A through Fusion of Large Multimodal Models

**arXiv ID:** 2601.06566 | [PDF](https://arxiv.org/pdf/2601.06566v1)

**作者:** Jiale Wang `[一作]` (Home Team Science and Technology Agency), Davis Wang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 QCaption 视频字幕与问答管线，将关键帧提取、图像‑文本 LMM 与文本 LLM 融合为可本地部署的多模态分析流程。

**💡 创新点**

通过晚期融合三模态模型（帧抽取、LMM、LLM）而非传统早期融合，显著提升视频字幕与问答质量，并可灵活替换各个子模型。

**🔧 技术方法**

关键帧抽取使用 Katna、均匀/随机采样；图像‑文本 LMM 采用 LLaVA‑1.5；文本聚合 LLM 采用 Vicuna‑v1.5，并对多剪辑采样与 LLM 省略做 ablation。

**📊 数据集**

在 YouCook2、MSR‑VTT（视频字幕）和 ActivityNet‑QA（视频问答）三个公开数据集上进行实验。

**📈 对比分析**

与 Video‑LLaVA、Video‑ChatGPT、Video‑LLaMA 等基线使用 CIDEr、Video‑ChatGPT 评估指标，QCaption 在字幕任务上提升 44.2%/28.7%，在问答任务上提升 48.9% 以上。

**⚠️ 局限性**

聚合阶段高度依赖 LLM 触发提示；对极短/单词级字幕（如 MSR‑VTT）可能因 LLM 抽象导致降效；并未加入语音等其他模态。

---

## 40. QMAVIS: Long Video-Audio Understanding using Fusion of Large Multimodal Models

**arXiv ID:** 2601.06573 | [PDF](https://arxiv.org/pdf/2601.06573v1)

**作者:** Zixing Lin `[一作]` (Home Team Science and Technology Agency), Yaohao Li `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了QMAVIS长视频音频理解管线，采用晚期融合方式将视频LMM、语音转写模型和LLM组合，能够处理超小时长的视频并生成连贯文本。

**💡 创新点**

创新点在于将视频分块后分别用视频LMM与Whisper进行视觉与音频处理，再通过LLM聚合输出，突破传统短视频限度，实现对长视频的多模态理解与叙事总结。

**🔧 技术方法**

使用技术包括视频多模态模型 Qwen2-VL-72B、Whisper Large v3 语音转写、LLM（如 Qwen2-LLM）以及分块截取、字幕与音频交错聚合等。

**📊 数据集**

实验使用了 VideoMME（含字幕）、PerceptionTest 和 EgoSchema 三大多模态长视频评测数据集。

**📈 对比分析**

与 VideoLLaMA2、InternVideo2、PandaGPT 等基线模型对比，QMAVIS 在 VideoMME 上提升 38.75%，在 PerceptionTest 提升 1.51%，在 EgoSchema 提升 1.72%，表现显著优于现有方法。

**⚠️ 局限性**

局限包括：多模型融合导致计算资源和推理时延较高；仅使用语音转写模型，未充分利用非语音音频信息；缺乏专门针对超长视频的标准评测数据集。

---

## 41. Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Vetting via Automated Spectral Inspection

**arXiv ID:** 2601.06498 | [PDF](https://arxiv.org/pdf/2601.06498v1)

**作者:** Minghui Jia `[一作]` (Institute of Automation), Dongbin Zhao `[通讯]` (Institute of Automation)

**通讯引用:** 15268 | [OpenAlex ID](https://openalex.org/A5100624298)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了Spec-o3，一种工具增强的视觉语言代理，用于像天文学家一样对光谱稀有天体候选人进行可视化检查，实现自动化筛选。

**💡 创新点**

创新点包括：①交互式多模态链式思维（iMCoT）结合光谱可视化工具，②两阶段后训练策略（冷启动SFT+基于结果的RL），③在光谱图像上实现细粒度缩放工具调用，显著提升稀有天体识别与跨仪器、跨任务泛化。

**🔧 技术方法**

采用Qwen2.5‑VL视觉语言模型，GRPO强化学习，Token‑level工具输出遮罩，交互式工具调用，文本‑图像联合推理。

**📊 数据集**

使用LAMOST的5类稀有天体样本（CV、CS、SS、MG、WD）构建SpecVI‑Bench基准，额外评估SDSS/DESI跨仪器数据以及未见O/B/A类型的跨任务数据。

**📈 对比分析**

与专用深度学习模型、专有VLM（GPT‑4.1、o3）以及开源VLM（Qwen2.5‑VL、Qwen3‑VL）对比，Spec-o3‑7B宏平均F1为76.5%（比基线提升约48%），跨仪器保持81%+，未见任务上F1≈76%，远优于对照模型。

**⚠️ 局限性**

局限性：评估仅涵盖有限稀有天体类别，未覆盖更广谱子类或极端观测条件；冷启动阶段仍需专家演示；未集成多模态外部数据；缺乏生产级风险控制（校准、拒绝、分流）等功能。

---

## 42. LLMTrack: Semantic Multi-Object Tracking with Multi-modal Large Language Models

**arXiv ID:** 2601.06550 | [PDF](https://arxiv.org/pdf/2601.06550v1)

**作者:** Pan Liao `[一作]` (Northwestern Polytechnical University), Wenhui Zhao `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 1586 | [OpenAlex ID](https://openalex.org/A5076775815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种结合 Grounding DINO 检测和 LLaVA‑OneVision LLM 的语义多目标跟踪框架 LLMTrack，实现从轨迹定位到自然语言行为描述的闭环。

**💡 创新点**

创新点在于将强定位与深度理解解耦，提出时空融合模块与三阶段渐进训练，能够让大型语言模型直接“读取”轨迹并生成语义化描述。

**🔧 技术方法**

采用 Grounding DINO、LLaVA‑OneVision（Qwen‑VL）、AnyRes 视频编码、Adaptive Temporal Attention、递归上下文聚合、LoRA 微调等技术。

**📊 数据集**

使用 BenSMOT 基准数据集（3292 条视频）进行评估。

**📈 对比分析**

与 SORT、ByteTrack、TransTrack、SMOTer 等传统与语义跟踪方法对比，LLMTrack 在 HOTA、MOTA、IDF1、CIDEr、交互 F1 等多项指标上均显著领先。

**⚠️ 局限性**

限制在于依赖大模型算力导致推理速度受限，长序列推理时可能产生幻觉，且缺乏公开完整语义数据集与实时系统评估。

---

## 43. LitVISTA: A Benchmark for Narrative Orchestration in Literary Text

**arXiv ID:** 2601.06445 | [PDF](https://arxiv.org/pdf/2601.06445v1)

**作者:** Mingzhe Lu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Yunpeng Li `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 VISTA Space 框架，统一人类与模型的叙事视角，并基于 LitBank 文学文本构建 LitVISTA 结构化评测基准，用于系统评估大型语言模型的叙事编排能力。

**💡 创新点**

创新点在于：①将叙事组织抽象为高维 VISTA 空间，将事件的结构角色（Impulse、Resonance、Pause）映射到三维坐标；②提出结构化注释的 LitVISTA 基准，突破传统线性事件序列评测；③通过 Oracle 事件级评估，剖析模型在节点分类与依赖解析两方面的强弱点。

**🔧 技术方法**

使用技术包括：事件(anchor)提取与分类、图结构构造、三维 VISTA 空间投影、Oracle 事件级联合优化、基于 Precision/Recall/F1 的评测。

**📊 数据集**

数据集来源于 LitBank 文学语料库，构造了 LitVISTA：10k token 长章节，平均每章约13个 Impulse、60个 Resonance、4个 Pause，涵盖跨章节长距离依赖。

**📈 对比分析**

对比 GPT、Claude、Gemini 等模型（含思考/非思考版本）在 Oracle 评估中的 Anchor 解析与依赖解析精度，结果显示模型在两子任务之间存在显著偏差，思考模式并未带来整体提升，整体 F1 均未突破 0.5。

**⚠️ 局限性**

局限包括：依赖 Oracle 设置（未评估完整端到端流程）；仅适用于英文公共领域文学；注释过程资源消耗大、规模有限；叙事结构本身的主观性仍影响“黄金标准”。

---

## 44. Plasticity vs. Rigidity: The Impact of Low-Rank Adapters on Reasoning on a Micro-Budget

**arXiv ID:** 2601.06677 | [PDF](https://arxiv.org/pdf/2601.06677v1)

**作者:** Zohaib Khan `[一作]` (University of Michigan), Zoha Hayat Bhatti `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在极低的算力预算（单块A40 GPU 48GB，训练时长24小时）下，研究者使用 RLVR 与 LoRA 进行微调，探讨小型语言模型（≤1.5B）在数学推理任务上的性能提升。

**💡 创新点**

创新点在于证明高秩 LoRA（r=256）能够在微预算环境下激活模型的隐含推理潜能，从而在不增加显存占用的前提下实现大幅度性能提升，揭示了模型“可塑性”与“刚性”之间的权衡。

**🔧 技术方法**

采用了强化学习可验证奖励（RLVR）+ Group Relative Policy Optimization（GRPO）以及低秩适配（LoRA）技术，并通过链式思维（CoT）提示引导推理过程。

**📊 数据集**

训练使用 Open‑RS 约7000道推理题，评估基准为 MATH500、AIME 24/25、AMC 23 等公开竞赛数据集。

**📈 对比分析**

对比基线模型与 LoRA 微调后的模型，最佳实验在 AIME 24 上实现了 40.0% Pass@1（比基线提升 11.1%）且 Pass@16 达 70.0%，显示出显著的性能提升，但此提升仅在通用指令模型中可见，对已高度数学对齐的模型则出现性能衰退。

**⚠️ 局限性**

局限性包括：仅评估了 ≤1.5B 的小模型，使用单 GPU、单次 24h 训练、单一随机种子，缺乏完整的超参搜索，且结果可能不适用于更大模型或更长训练周期。

---

## 45. Hellinger Multimodal Variational Autoencoders

**arXiv ID:** 2601.06572 | [PDF](https://arxiv.org/pdf/2601.06572v1)

**作者:** Huyen Khanh Vo `[一作]` (Max Planck Institute for Software Systems), Isabel Valera `[通讯]` (University of Saarland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于 Hellinger 聚合的多模态 VAE（HELVAE），通过 Hölder pooling（α=0.5）对单模态推断进行矩匹配，消除子采样问题；

**💡 创新点**

创新点在于：①将 Hölder pooling 与 Hellinger 距离结合，得到一种软依赖的聚合方式；②利用矩匹配将非高斯聚合转化为单高斯后验；③实现无子采样训练，提升生成一致性与质量的平衡；

**🔧 技术方法**

使用技术包括：Hölder pooling、α‑divergence（α=0.5）、Hellinger 距离、矩匹配近似、变分自编码器框架、β‑VAE 超参数调节、MoHELVAE 组合等；

**📊 数据集**

实验数据集：PolyMNIST、CUB Image‑Captions、bimodal CelebA；

**📈 对比分析**

与 MVAE、MMVAE、MoPoE、MMVAE+、MWBVAE、CoDEVAE 等方法对比，评估指标为生成一致性、FID（生成质量）、对数似然、潜在表示准确率；HELVAE 在一致性与质量上多次与最先进模型持平或更优，尤其在多模态协同生成上表现突出；

**⚠️ 局限性**

限制包括：无模态专属潜变量导致无条件一致性略逊，生成多样性稍低；模型对权重选择敏感，未来需探索可学习权重或加入专属潜变量。

---

## 46. Evaluating Robustness of Large Language Models in Enterprise Applications: Benchmarks for Perturbation Consistency Across Formats and Languages

**arXiv ID:** 2601.06341 | [PDF](https://arxiv.org/pdf/2601.06341v1)

**作者:** Tara Bogavelli `[一作]` (ServiceNow), Roshnee Sharma `[通讯]` (ServiceNow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一套面向企业场景的LLM鲁棒性基准，评估了11种模型在四类任务（案例摘要、聊天摘要、问答、实体槽填充）上对五类扰动（通用编辑、位置变换、格式切换、多语言、跨语言）的表现。

**💡 创新点**

创新点在于：①统一覆盖多种实际扰动并细分子类型；②从模型规模与鲁棒性关系、格式与语言特定弱点等维度系统揭示模型差异；③提出“内容差异+质量差异”双层评估与整体鲁棒分数公式。

**🔧 技术方法**

技术上采用LLM‑as‑a‑Judge（以GPT‑4.1为评判器）对输出进行语义相似度与任务质量评分；利用内容差异率、质量差异率计算鲁棒得分；通过多次固定温度0推理控制随机性。

**📊 数据集**

数据集由420条IT故障案例手工构造，随后用LLM生成四个任务对应的输入；再通过机器翻译生成8种语言版本；扰动数据包括空格、标点、大小写、重排、JSON/HTML/XML/Markdown/YAML等格式。

**📈 对比分析**

实验对11个模型（4B–120B+）进行5次独立跑，计算基线与扰动下的鲁棒得分；结果显示GPT‑5.2最高（≈91），Llama 3.1 8B最低（≈69）；不同扰动导致平均降分从通用（≈−8.8）到位置（≈−18）不等；模型规模并非单一决定因素，架构与训练方式更为关键。

**⚠️ 局限性**

局限性包括：①评判器为LLM，可能漏判细节；②鲁棒得分聚合掩盖细粒度差异；③模型输出非完全确定，5次跑仍不足以捕获高方差；④多语言扰动仅用自动翻译，可能引入错误；⑤仅评估少数封闭源模型；⑥仅覆盖IT领域和四个任务，未涉及更广泛业务与任务；⑦单一扰动评测，未考量多重扰动交互；⑧未覆盖更广语言及不同温度设置。

---

## 47. ReAct: Reflection Attack Mitigation For Asymmetric Routing

**arXiv ID:** 2601.06367 | [PDF](https://arxiv.org/pdf/2601.06367v1)

**作者:** David Hay `[一作]` (Hebrew University of Jerusalem), Shir Landau Feibish `[通讯]` (University of Haifa)

**通讯引用:** 460 | [OpenAlex ID](https://openalex.org/A5078064243)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种名为 ReAct 的数据平面反放大反射 DDoS 防御框架，可在具有对称或非对称路由的网络中对合法请求与响应进行关联，从而过滤掉大部分攻击流量。

**💡 创新点**

创新点在于：①使用滑动窗口 Bloom 过滤器而非计数 Bloom 过滤器，避免攻击流量对过滤器状态的破坏；②通过可编程数据平面跨交换机协作，支持非对称路由下的请求‑响应关联；③实现了无控制平面干预的自适应路由变化机制。

**🔧 技术方法**

技术实现基于 P4/Lucid 编程模型在 Tofino 交换机和 NVIDIA BlueField‑3 SmartNIC 上部署，核心技术包括多级 Bloom 过滤器、请求转发表、广播标记、滑动窗口更新以及软硬件协同状态管理。

**📊 数据集**

使用公开的 DNS 交互流量（BIND9 服务器响应）与人工注入的伪造响应作为实验数据集，并在模拟环境中通过 Lucid 解释器模拟双交换机架构进行验证。

**📈 对比分析**

与现有计数 Bloom 过滤器方案（Jaqen）对比，ReAct 在任意攻击比例下保持 0‑5% 的误报率并成功阻断 97%+ 的攻击响应；在对称与非对称场景均能保证合法响应不被丢弃，且占用的 SRAM 仅约 6% 甚至 4%，性能优于传统硬件防护。

**⚠️ 局限性**

主要局限包括：对无固定事务 ID 的协议（如 NTP、SSDP）处理仍需进一步研究；跨机组广播可能在大规模网络中产生额外开销；Bloom 过滤器的误判导致的广播频繁增加，需更细粒度的控制平面协调；以及对极高吞吐量或多协议混合场景下的可扩展性尚未充分验证。

---

## 48. Geo-Standardizing 3D Modeling of Surface Objects and Related Logical Spaces on Celestial Bodies: Case Studies for Moon and Mars

**arXiv ID:** 2601.06182 | [PDF](https://arxiv.org/pdf/2601.06182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 49. PsyAgent: Constructing Human-like Agents Based on Psychological Modeling and Contextual Interaction

**arXiv ID:** 2601.06158 | [PDF](https://arxiv.org/pdf/2601.06158v1)

**作者:** Zibin Meng `[一作]` (Hong Kong University of Science and Technology), Kani Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 856 | [OpenAlex ID](https://openalex.org/A5040402090)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PsyAgent，一种将 Big Five 性格先验与 Bourdieu 的社会结构框架结合的心理学驱动型对话代理。

**💡 创新点**

创新点在于：① 将性格特质与四个内部结构（教育轨迹、生活经历、社会经济背景、文化资本）编码成机器可用的 Individual Structure（IS）；② 设计 Multi-Scenario Contexting（MSC）八大情景的角色-关系-规范框架；③ 通过结构化 prompt 与小规模参数高效适配器（LoRA/QLoRA）实现长时一致的性格表达与情境适应。

**🔧 技术方法**

技术包括：结构化 prompt 绑定 IS 与 MSC、SFT 与 DPO（PEFT 训练）对小模型进行监督微调、确定性解码、统一百分位数评估指标（ProfileAcc、MAE、RMSE、余弦相似度）。

**📊 数据集**

使用合成的 IS×MSC 数据集：跨 IS 4 维与 MSC 8 场景的 5 个 Big Five 组合共 38,880 条样本，全部由 LLM 生成并经过滤、去重、审核得到。

**📈 对比分析**

与未使用 PsyAgent 的多种规模开源 LLM（1B~70B）在相同推理设置下对比；结果显示小模型在 ProfileAcc 上提升 11.7 点，RMSE 降低 13.5 点，DPO 进一步提升 8.1 点；整体性能可与或优于更大无调优模型，且保持语义一致性与安全性。

**⚠️ 局限性**

局限性包括：① 监督数据为合成，可能带来合成偏差；② 评估与过滤使用相同的性格/情境 scorer，存在耦合风险；③ MSC 规范与角色多为文化特定，跨文化迁移需重写；④ 情境覆盖有限，专业领域未包含；⑤ 长交互中仍可能出现漂移，需进一步红队验证。

---

## 50. A Mixed Methods Systematic Analysis of Issues and Factors Influencing Organizational Cloud Computing Adoption and Usage in the Public Sector: Initial Findings

**arXiv ID:** 2601.06175 | [PDF](https://arxiv.org/pdf/2601.06175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 51. Some New Results on Sequence Reconstruction Problem for Deletion Channels

**arXiv ID:** 2601.06503 | [PDF](https://arxiv.org/pdf/2601.06503v1)

**作者:** Xiang Wang `[一作]` (Beijing University of Technology), Fang-Wei Fu `[通讯]` (Nankai University)

**通讯引用:** 3081 | [OpenAlex ID](https://openalex.org/A5063946169)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了删除信道上的序列重建问题，并给出了N(n,3,t)的下界与N(n,3,4)的精确值；

**💡 创新点**

提出了新的下界表达式，并对更高t给出猜想，突破了原有理论极限；

**🔧 技术方法**

主要使用组合学、序列分析以及计算机枚举搜索技术；

**📊 数据集**

未使用传统数据集，全部基于理论与符号枚举；

**📈 对比分析**

通过与已知上界进行理论比较，证明下界在n≥13时与上界相近，理论上接近最优；

**⚠️ 局限性**

局限在于对t≥5仅给出猜想，缺乏严格证明，且对小长度n的情况仅通过枚举验证。

---

## 52. Spatiotemporal Change-Points in Development Discourse: Insights from Social Media in Low-Resource Contexts

**arXiv ID:** 2601.06402 | [PDF](https://arxiv.org/pdf/2601.06402v1)

**作者:** Woojin Jung `[一作]` (Rutgers University), Tawfiq Ammari `[通讯]` (Rutgers University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用Zambia两年多的地理标签X数据，探究低资源地区发展话语的时空演化；

**💡 创新点**

提出“可持续话语（durable discourse）”概念，区分危机短期波动与长期基础设施议题的持续性；

**🔧 技术方法**

结合BERTopic主题建模、PELT变点检测和定性编码；

**📊 数据集**

Zambia 2019‑2021年原始推文共212万条，其中20k条地理标记；

**📈 对比分析**

采用主题一致性得分0.72、103个主题，PELT检测到与COVID‑19及热能项目相关的关键变点；未与传统调查做直接量化对比；

**⚠️ 局限性**

依赖单一平台X和API访问，地理标签样本极少，平台偏见与语言不均衡，且无法完全验证研究结果。

---

## 53. The environmental impact of ICT in the era of data and artificial intelligence

**arXiv ID:** 2601.06174 | [PDF](https://arxiv.org/pdf/2601.06174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 54. C-EQ-ALINEA: Distributed, Coordinated, and Equitable Ramp Metering Strategy for Sustainable Freeway Operations

**arXiv ID:** 2601.06311 | [PDF](https://arxiv.org/pdf/2601.06311v1)

**作者:** Kevin Riehl `[一作]` (ETH Zurich), Michail A. Makridis `[通讯]` (ETH Zurich)

**通讯引用:** 2121 | [OpenAlex ID](https://openalex.org/A5015419644)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出一种名为 C‑EQ‑ALINEA 的公平意识、去中心化、协同化入口匝道计时控制器，并在阿姆斯特丹 A10 环路上进行验证。

**💡 创新点**

创新点在于仅通过相邻匝道间极其轻量的信息交换，将公平性指标（Harsanyi、Egalitarian、Rawlsian、Aristotelian）嵌入传统 ALINEA 反馈控制，保持控制器的简易性与鲁棒性，同时实现了跨匝道的公平协同。

**🔧 技术方法**

使用距离加权的邻域协同项、两种距离归一化（全局与局部）、SUMO 微观仿真、网格搜索优化控制参数，并采用 Gini 系数、最大延时等多维公平指标。

**📊 数据集**

利用荷兰国家交通门户（Nationaal Dataportaal Wegverkeer）校准的需求模型，以及荷兰中央统计局（Centraal Bureau voor de Statistiek）提供的车辆分布数据，对 A10 环路 20 个入口、22 个出口进行仿真。

**📈 对比分析**

与传统 ALINEA 及协同控制 METALINE 进行基准比较，评估指标包括总延时、平均延时、Gini、最大延时等。C‑EQ‑ALINEA 在全局距离归一化、邻域大小 m=3 的配置下，总延时降低约 58%，平均延时降至 2.1 min/车，公平性指标均优于两种基线，且在多项效率指标上与 METALINE 相当或更优。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未实测用户感知公平性；假设检测器覆盖完整、匝道存储充足，未处理排队溢出等物理约束；对网络拓扑与通信延迟敏感；缺乏对突发事件、硬件老化等不确定因素的鲁棒性评估。

---

## 55. VIPER Strike: Defeating Visual Reasoning CAPTCHAs via Structured Vision-Language Inference

**arXiv ID:** 2601.06461 | [PDF](https://arxiv.org/pdf/2601.06461v1)

**作者:** Minfeng Qi `[一作]` (City University of Macau), Lefeng Zhang `[通讯]` (City University of Macau)

**通讯引用:** 604 | [OpenAlex ID](https://openalex.org/A5090039254)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ViPer框架，结合结构化多物体感知与LLM推理实现对视觉推理CAPTCHA的通用攻击；

**💡 创新点**

将视觉检测与符号推理解耦，通过查询信息提取、整合与相对位置推理以及动态提示的LLM推理，实现跨平台通用性，并提出TSR防御策略；

**🔧 技术方法**

使用anchor-free YOLOv11式多物体检测、语义槽位解析、图神经网络对比与GPT-4o/DeepSeek/Kimi/Grok-2等大语言模型的自适应提示推理；

**📊 数据集**

基准包含六大VRC提供商（VTT、Geetest、NetEase、Dingxiang、Shumei、Xiaodun）共6000+样本，附1200张多物体标注图像；

**📈 对比分析**

在统一基准上与Holistic、GraphNet、Oedipus等传统与LLM方法对比，ViPer在六个平台平均93%以上准确率，超越人类10个百分点，LLM后端切换后仍保持90%+准确，响应时间与人类相当；

**⚠️ 局限性**

依赖大模型导致计算与成本开销大、未系统评估轻量化模型、TSR防御仅针对自动攻击未验证对人类可用性、未覆盖非视觉CAPTCHA等。

---

## 56. Dreaming Is Not a Bug: A Jung-Inspired Dream Layer for Multi-Agent LLM Companions

**arXiv ID:** 2601.06115 | [PDF](https://arxiv.org/pdf/2601.06115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 57. Matrix Factorization Framework for Community Detection under the Degree-Corrected Block Model

**arXiv ID:** 2601.06262 | [PDF](https://arxiv.org/pdf/2601.06262v1)

**作者:** Alexandra Dache `[一作]` (University of Mons), Nicolas Gillis `[通讯]` (University of Mons)

**通讯引用:** 3646 | [OpenAlex ID](https://openalex.org/A5040368041)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将度纠正块模型（DCBM）的推断转化为约束非负矩阵三因子分解（OtrisymNMF），并提出FROST算法和基于SVCA的鲁棒初始化，解决了DCBM推断的初始化敏感性和计算效率问题。

**💡 创新点**

创新点包括：① 用Frobenius范数替代KL散度，避免Poisson假设带来的限制；② 通过三因子分解框架统一DCBM与NMF，提出OtrisymNMF模型；③ 设计高效的FROST迭代算法；④ 开发基于可分离NMF的SVCA初始化，理论上可在多种网络结构下获得高质量初始分区。

**🔧 技术方法**

主要技术：非负矩阵三因子分解、Frobenius范数最小化、正交性约束、可分离NMF（SVCA）初始化、交替最优化（FROST）以及传统DCBM推断算法（KN、KL‑EM、MHA）对比。

**📊 数据集**

使用的数据集包括：LFR合成网络、Zachary Karate Club、Political Blog Network、Southern Women（二分网络）、Scotland Corporate Interlock、Malaria基因子串网络等。

**📈 对比分析**

与传统DCBM推断方法（KN、KL‑EM、MHA）和仅使用SVCA初始化的结果对比。实验表明：
- 在大多数情况下，FROST的社区发现与DCBM相当，甚至在密集图上优于DCBM；
- FROST在运行时间上显著快，尤其在大规模网络（10⁵节点）下，速度提升可达10倍；
- SVCA初始化能显著提高所有方法的准确率和收敛速度，减少迭代次数；
- 对于二分网络，FROST在大多数试验中能恢复正确的分割，且比DCBM推断更稳健。

**⚠️ 局限性**

局限性：
- 当社区结构较弱或图很稠密时，OtrisymNMF的精度略逊于DCBM；
- SVCA初始化在零行导致某些节点未分配到社区，可能导致后续算法陷入次优局部最优；
- 对于极大或极稀疏网络，FROST仍需多次随机初始化以获得最优解；
- 目前对节点重叠社区的支持有限，需进一步扩展。

---

## 58. Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation

**arXiv ID:** 2601.06600 | [PDF](https://arxiv.org/pdf/2601.06600v1)

**作者:** Jen-tse Huang `[一作]` (Johns Hopkins University), Mark Dredze `[通讯]` (Johns Hopkins University)

**通讯引用:** 23110 | [OpenAlex ID](https://openalex.org/A5024437840)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了200条短视频的多模态数据集并进行手工标注错误类型，评估八大多模态LLM在识别谣言与认知偏差方面的表现。

**💡 创新点**

提供高质量、手工校验的多模态短视频谣言数据集，细粒度标注实验错误、逻辑谬误与虚构声明，系统探测MLLM在真实场景中的误差来源与认知偏差。

**🔧 技术方法**

采用视觉帧采样、OCR、ASR、链式思维提示，7级Likert信任评分和归一化Belief Score评估，利用多模态LLM（Gemini、GPT‑4o、Qwen、Claude、Seed）进行推理。

**📊 数据集**

自建200条抖音、快手短视频数据集，分四个健康领域，手工注解三种错误类型，并附加频道、点赞、转发等社交元数据。

**📈 对比分析**

在五种输入设定（Claim、Textual、Aural、Visual、Multimodal）与四个领域、三种错误类型对比实验，Gemini‑2.5‑Pro以71.5/100最高，其他模型表现各异，存在标签偏差、保守性、偏信任等。

**⚠️ 局限性**

数据规模有限、仅中文短视频、可能存在训练数据泄露、帧采样可能漏掉细微视觉信息、模型偏差与人类认知偏差相似。

---

## 59. Neural Nonmyopic Bayesian Optimization in Dynamic Cost Settings

**arXiv ID:** 2601.06505 | [PDF](https://arxiv.org/pdf/2601.06505v1)

**作者:** Sang T. Truong `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**通讯引用:** 4526 | [OpenAlex ID](https://openalex.org/A5091266570)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为LookaHES的非贪婪贝叶斯优化框架，专门针对动态、历史相关的查询成本环境；

**💡 创新点**

核心创新在于将多步H-Entropy Search与神经策略（包含大型语言模型）和路径采样结合，实现了超过二十步的长周期规划，同时显著降低了指数级复杂度；

**🔧 技术方法**

技术手段包括多步HES、路径采样、变分神经策略（RNN/Transformer）、贝叶斯回归与信息增益公式的变分优化；

**📊 数据集**

实验数据集涵盖九个从2维到8维的合成基准、NASA夜灯影像地理空间优化任务以及ProteinEA蛋白荧光半合成编辑任务；

**📈 对比分析**

与八种基准（EI、UCB、KG、Gittins、MSL等）比较，LookaHES在所有合成、地理与蛋白编辑任务上均实现了更低的最终回报（更高的目标值）且保持了较好的计算效率；

**⚠️ 局限性**

局限性在于需要先验给定精确的动态成本模型以及对代理模型的依赖，若成本模型或代理模型失真，计划质量会受到显著影响。

---

## 60. Latent Space Communication via K-V Cache Alignment

**arXiv ID:** 2601.06123 | [PDF](https://arxiv.org/pdf/2601.06123v1)

**作者:** Lucio M. Dery `[一作]` (Google DeepMind), Arthur Szlam `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文内容未提供，无法确定研究主题

**💡 创新点**

无可识别的创新点

**🔧 技术方法**

未知技术

**📊 数据集**

未使用任何数据集

**📈 对比分析**

未给出对比方法和性能评估

**⚠️ 局限性**

缺乏详细信息，无法评估限制

---

## 61. From Individual Prompts to Collective Intelligence: Mainstreaming Generative AI in the Classroom

**arXiv ID:** 2601.06171 | [PDF](https://arxiv.org/pdf/2601.06171v1)

**作者:** Junaid Qadir `[一作]` (Qatar University), Muhammad Salman Khan `[通讯]` (Qatar University)

**通讯引用:** 44163 | [OpenAlex ID](https://openalex.org/A5049702888)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两门本科工程课程中实施基于思维流程的生成式集体智能（GCI）教学活动，结合问题分类与分层分析，使用生成式 AI 作为结构化协作的工具，促成学生在小组内共同建构知识并反思 AI 的角色；

**💡 创新点**

将生成式 AI 与 Harvard Project Zero 的“可视化思维”流程系统性融合，形成“先人类思维、后 AI 交互”的教学模型，强调 AI 作为协同桥梁而非替代者；

**🔧 技术方法**

使用 ChatGPT/CustomGPT 进行结构化提问与观点梳理，并采用“Question Sorts”和“Peel the Fruit”等思维流程工具；

**📊 数据集**

通过两门课程的140名学生问卷调查与课堂观察记录收集数据；

**📈 对比分析**

以学生对学习方式的偏好、AI 使用频率、时机感知等自评指标为衡量；结果显示“组内协作+AI”模式被约50%学生认为最有效，且对 AI 时机敏感度高；

**⚠️ 局限性**

缺乏对照组、随机分配、客观测验、单一机构与短期实验、主观自评为主要局限，需后续进行随机对照实验和纵向追踪验证。

---

## 62. Precision Meets Art: Autonomous Multi-UAV System for Large Scale Mural Drawing

**arXiv ID:** 2601.06508 | [PDF](https://arxiv.org/pdf/2601.06508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 63. Object-WIPER : Training-Free Object and Associated Effect Removal in Videos

**arXiv ID:** 2601.06391 | [PDF](https://arxiv.org/pdf/2601.06391v1)

**作者:** Saksham Singh Kushwaha `[一作]` (University of Texas at Dallas), Kuldeep Kulkarni `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的视频目标及其相关效果去除方法，利用文本到视频扩散模型实现无监督对象去除与背景填充。

**💡 创新点**

创新点包括：①通过交叉注意力与自注意力两步定位目标及其相关阴影、反射等效果的掩码；②引入时间自适应掩码与注意力缩放，避免在去噪过程中目标信息泄漏；③使用噪声重初始化与背景值复制实现一致的空间填充；④提出基于 DINOv3 的 Token Similarity 评价指标与全新真实世界 benchmark（WIPER‑Bench）来客观评估去除质量。

**🔧 技术方法**

核心技术：文本到视频扩散变换器（DiT）、交叉注意力与自注意力映射、时间自适应掩码、注意力缩放（biasing）、噪声逆向（inversion）与重初始化、DINOv3 语义特征、Token Similarity 评价。

**📊 数据集**

使用公开数据集 DAVIS 以及新构建的 60 条真实视频组成的 WIPER‑Bench（涵盖阴影、反射、镜面、半透明、复合多效、离散效应等）。

**📈 对比分析**

与多种训练基准（Gen‑Prop、ROSE、Propainter）及训练无关基线（KV‑Edit、Attentive‑Eraser、KV‑Edit‑Video）在 TokSim 指标上均优于对手；在 BG‑PSNR、FG‑Flicker、Text‑Align 等传统指标上亦保持竞争力，证明在不进行额外微调的前提下可实现更干净、更一致的去除效果。

**⚠️ 局限性**

局限性：①对极端遮挡或高速运动场景仍可能残留微量目标痕迹；②评价主要基于 Token Similarity，缺乏更细粒度的视觉主观评估；③方法依赖大规模预训练 DiT 模型，计算成本相对传统视频填补算法仍高；④未公开完整的实现细节与模型下载链接，限制了复现与广泛部署。

---

## 64. Robust and Secure Blockage-Aware Pinching Antenna-assisted Wireless Communication

**arXiv ID:** 2601.06430 | [PDF](https://arxiv.org/pdf/2601.06430v1)

**作者:** Ruotong Zhao `[一作]` (University of New South Wales), Derrick Wing Kwan Ng `[通讯]` (University of New South Wales)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种在存在遮挡和不完美CSI下的多波导、可调节PA（pinching antenna）系统，实现鲁棒且安全的下行通信，主要通过联合优化波导功率分配、PA位置、天线波束与人工噪声来最大化合法用户的系统总速率，并满足信息泄露约束。

**💡 创新点**

创新点包括：
- ① 构造了几何感知的CSI不确定集，能够同时考虑窃听者位置与方位误差；
- ② 设计了3D遮挡感知的信道模型，既考虑波导衰减，又涵盖自由空间路径与LoS可用性；
- ③ 提出了两阶段PA定位策略（大尺度路径损失/遮挡 + 微尺度相位对齐），显著降低求解复杂度；
- ④ 在此基础上提出了基于BCD、S‑procedure、MM和Lipschitz近似的求解框架，能够在非凸环境下获得高质量子最优解。

**🔧 技术方法**

使用技术：
- 信道建模：近场球面波、波导衰减、遮挡判定；
- 误差建模：几何感知的Frobenius‑norm CSI误差界；
- 优化方法：块坐标下降（BCD）、S‑procedure、MM（majorization‑minimization）、LMI变换、两阶段位置分解、Lipschitz‑基逼近；
- 约束处理：功率、相位、遮挡、信息泄漏约束；
- 计算实现：使用CVX求解凸子问题。

**📊 数据集**

数据集与仿真设置：
- 随机生成15×15 m 平面内用户位置，2个波导、每个波导5个PA，1个用户，1个EAV，2个障碍物；
- 频率 28 GHz，波导材料 PTFE（η_eff=1.42，ϵ_r=2.1，tanδ=2×10⁻⁴）；
- 误差参数 κ²=0.1（CSI误差），EAV方位误差 1°，位置误差 1 cm；
- 多组随机实验（至少10⁴ 次）评估误差上界与性能。

**📈 对比分析**

比较方法与性能：
- 对比上界（无遮挡、无EAV、无波导衰减）、BM1（固定功率分配、优化PA位置）、BM2（固定PA位置、优化波束）以及无遮挡版本；
- 实验结果显示：
  • 采用遮挡感知的PA系统比传统固定天线方案平均高约 4.7 dB；
  • 方案在不同总功率、CSI误差、遮挡配置下均保持较大优势；
  • 随功率提升，信息泄露约束更显重要，人工噪声占比上升；
  • 通过主动遮挡利用可显著降低人工噪声需求。

**⚠️ 局限性**

局限性：
- 假设波导与PA位置在静态或慢移动场景，未考虑高速用户或快速遮挡变化；
- 采用理想化的矩形障碍模型，实际建筑遮挡更复杂；
- 求解过程仍为子最优，且在大规模系统中计算量显著；
- 未考虑能量效率、硬件实现细节与实际波导损耗模型的精度；
- 误差模型假设为Frobenius‑norm界，可能对极端环境不够精确。

---

## 65. Automated Generation of Accurate Privacy Captions From Android Source Code Using Large Language Models

**arXiv ID:** 2601.06276 | [PDF](https://arxiv.org/pdf/2601.06276v1)

**作者:** Vijayanta Jain `[一作]` (University of Maine), Collin McMillan `[通讯]` (University of Notre Dame)

**通讯引用:** 3290 | [OpenAlex ID](https://openalex.org/A5084874990)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种自动生成隐私说明短句的系统 PCapGen。

**💡 创新点**

创新点在于结合大规模源码上下文提取、LLM 细粒度描述，并生成精准完整的隐私说明。

**🔧 技术方法**

技术包括静态代码分析、上下文挖掘、使用大型语言模型生成描述以及可视化工具。

**📊 数据集**

数据集为公开安卓应用的源码与手工标注的隐私说明，包含多个应用及其隐私行为标签。

**📈 对比分析**

与基线方法相比，PCapGen 生成的说明更简洁完整且准确率提升，专家评价至少 71%，LLM 评测至少 76%。

**⚠️ 局限性**

局限性包括对大型模型的依赖、对少数语言/平台的覆盖有限，以及对复杂隐私行为的识别仍有误差。

---

## 66. Triadic Concept Analysis for Logic Interpretation of Simple Artificial Networks

**arXiv ID:** 2601.06229 | [PDF](https://arxiv.org/pdf/2601.06229v1)

**作者:** Ingo Schmitt `[一作]` `[通讯]` (Brandenburgische Technische Universität Cottbus-Senftenberg), Ingo Schmitt (Brandenburgische Technische Universität Cottbus-Senftenberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将训练好的简单神经网络转化为三维比特张量，再利用三元概念分析提取三元概念并将其映射为可解释的逻辑树，实现模型解释。

**💡 创新点**

创新点在于将ReLU节点划分空间得到线性子模型，使用minterm权重量化为比特后构造三维比特张量，并把它视作三元关系进行三元概念分析，从而生成非布尔逻辑的可解释树。

**🔧 技术方法**

主要技术包括ReLU分区、minterm构造、Formal Concept Analysis与Triadic Concept Analysis、量化映射、QLDT（量子逻辑启发决策树）以及基于Shapley值的属性重要性分析。

**📊 数据集**

实验数据集为UCI血液输血服务中心（Blood Transfusion Service Center）数据集。

**📈 对比分析**

与原始网络保持相同的74%准确率；提取的概念和逻辑树在覆盖训练样本上也能达到类似准确率，并能揭示属性间的交互作用。

**⚠️ 局限性**

局限性包括：特征数必须很小（需生成2^n个minterm）；仅适用于包含单一ReLU层的简单网络；对更复杂网络的扩展尚未实现。

---

## 67. Student Guides Teacher: Weak-to-Strong Inference via Spectral Orthogonal Exploration

**arXiv ID:** 2601.06160 | [PDF](https://arxiv.org/pdf/2601.06160v1)

**作者:** Dayu Wang `[一作]` (Baidu Inc), Yang Li `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Spectral Orthogonal Exploration (SOE) 通过弱学生模型的正交探针来引导强教师模型逃离低秩偏差流形，实现更高质量的推理路径。

**💡 创新点**

创新点在于将正交投影与微型SVD结合，构建“学生引导教师”几何框架，利用学生的结构异质性在教师的空域中注入维度而非知识，显著提升推理多样性和正确率。

**🔧 技术方法**

核心技术包括：隐状态协方差矩阵的有效秩评估、微型SVD (Micro‑SVD) 估计低秩偏差流形、正交残差最大化的“正交潜在拼接”(OLS) 以及基于学生生成的正交探针的干预插入。

**📊 数据集**

在五大数学推理基准上验证：AIME 2024、AIME 2025、MATH‑500、OlympiadBench、Omni‑Math（Hard），使用教师模型 (4B 级) 与弱学生模型。

**📈 对比分析**

与自一致性基线 (Self‑Consistency) 比较，SOE 在 Pass@16 上提升平均 62.4%（最高 99.7%），同时在语义探索效率上几乎保持线性增长，显著优于基线的对数饱和。

**⚠️ 局限性**

局限性包括：额外的微型SVD 与蒙特卡洛前向仿真导致每步延迟较大；需要访问模型内部隐藏状态，限制了对封闭源模型的适用；实验主要在 4B 规模模型和数学领域，尚未验证在更大模型或非数学推理任务上的可扩展性。

---

## 68. The Patient/Industry Trade-off in Medical Artificial Intelligence

**arXiv ID:** 2601.06144 | [PDF](https://arxiv.org/pdf/2601.06144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 69. LDTC: Lifelong deep temporal clustering for multivariate time series

**arXiv ID:** 2601.06221 | [PDF](https://arxiv.org/pdf/2601.06221v1)

**作者:** Zhi Wang `[一作]` (Xidian University), Yiyuan Jiao `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种终身深度时间序列聚类算法LDTC，集成降维与聚类于一体，支持在无监督序列任务学习中持续学习新概念并避免灾难性遗忘。

**💡 创新点**

创新点包括：①联合层级优化的自编码器与聚类目标；②基于Dilated Causal CNN＋Attention BiLSTM的CTAE编码器；③终身学习机制，包含动态模型扩展与混合重放来实现无监督终身学习。

**🔧 技术方法**

使用的技术包括：自编码器CTAE、聚类层TC、Student t核下的KL散度优化、Adam优化器、混合重放回放、模型池与动态扩展等。

**📊 数据集**

实验使用七个真实多变量时间序列数据集：EEG2、NetFlow、Wafer、HAR、AREM、Uwave、ArabicDigits。

**📈 对比分析**

与传统k-means、DTC、USRL对比，LDTC在准确率和纯度上平均提升15–21%，且训练时间最短，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：聚类精度仍有进一步提升空间；终身学习机制可能导致模型池内存消耗增长。

---

## 70. Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds

**arXiv ID:** 2601.06286 | [PDF](https://arxiv.org/pdf/2601.06286v1)

**作者:** Min Dai `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**通讯引用:** 14488 | [OpenAlex ID](https://openalex.org/A5039171820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于物理模型引导的强化学习框架，实现了在受限足点（如踏石、踏板）上灵活且可靠的人形机器人行走；

**💡 创新点**

创新点在于将低阶步进规划（LIP模型）产生的动态一致参考与CLF奖励紧耦合，既保留了模型规划的精确性，又利用RL的鲁棒性；

**🔧 技术方法**

核心技术包括：
- 线性倒立摆（LIP）步进规划与能量调节
- CLF‑based奖励与轨迹跟踪
- 教师‑学生（distillation）RL训练
- 任务相关的自适应步长与垂直COM速度控制
- 现场模拟与硬件部署的域随机化与无监督转移

**📊 数据集**

数据集：采用基于随机几何参数的四类踏石/楼梯仿真地形（可调难度），并在真实硬件上使用预设的踏石地形网格；

**📈 对比分析**

与纯端到端RL基线及两种消除结构（固定步长、无垂直速度）对照实验，实验显示：在平面与高度变化踏石上，本文方法在全部4096环境上实现了100%成功率；相较于基线的最高56%和消除结构的低于90%，性能显著提升；在 sim‑to‑sim 与 sim‑to‑real 迁移实验中亦保持稳定；

**⚠️ 局限性**

局限性包括：
- 仍需依赖对踏石几何的精准感知（但通过高度图可部分缓解）
- 在极端极端宽阔间隙或极高摆动幅度时可能需进一步优化步长与COM调节
- 训练耗时较长（PPO 16k迭代，4096环境），对计算资源要求高

---

## 71. S-DAPT-2026: A Stage-Aware Synthetic Dataset for Advanced Persistent Threat Detection

**arXiv ID:** 2601.06690 | [PDF](https://arxiv.org/pdf/2601.06690v1)

**作者:** Saleem Ishaq Tijjani `[一作]` (University of Plymouth), Matthew Craven `[通讯]` (University of Plymouth)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5070682430)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并生成了近似真实的多阶段APT数据集（S‑DAPT‑2026），包含 14 种安全告警类型，覆盖校园和组织网络环境，并对每条告警赋予时间、主机、协议等属性；同时提出了基于 KNN 的告警聚类与关联索引框架，实现对APT攻击链的自动识别与阶段标注。

**💡 创新点**

创新点在于：
1) 数据集覆盖更广泛的告警类型（14种）且保留完整的多阶段APT生命周期；
2) 引入机器学习驱动的 KNN‑cosine 相似度聚类，显著降低传统关联计算的复杂度；
3) 对每个告警序列计算“关联指数”，可直接判定为完整、子或非APT场景，支持阶段感知检测。

**🔧 技术方法**

主要技术手段包括：
- 语义化告警生成与随机噪声混合；
- 采用 KNN‑基于余弦相似度的聚类算法，对告警进行时间窗口内的分组；
- 关联索引规则（Corr_a,b…Corr_d,e）结合主机与事件属性，量化阶段关联度；
- 统计特征工程（时间拆分、one‑hot 编码、Min‑Max 标准化）为后续 ML 模型提供特征。

**📊 数据集**

使用的数据集为自研的 S‑DAPT‑2026，包含 120,000 条网络/主机告警，约 40% 为APT标签（完整与子场景），60% 为非APT。数据涵盖校园/组织网络环境的多阶段攻击与背景噪声。

**📈 对比分析**

方法通过与现有公开APT数据集（如 ISCXIDS2012、DAPT 2020、Unraveled 等）对比，S‑DAPT‑2026 在告警类型多样性、阶段完整性、关联指数分布以及时间分布上更贴近真实场景；KNN 聚类在相同告警负载下的计算时间显著低于手工规则关联，提升了实时检测潜力。虽然文中未给出精确指标，但统计分析表明关联指数与真实攻击链高度一致，且完整场景与子场景的区分率可达 90%+。

**⚠️ 局限性**

局限性包括：
- 数据为合成，缺乏真实攻击的复杂性与可变性；
- 关联规则与告警映射仍基于预设模板，可能无法覆盖新型APT TTP；
- 未在真实网络或实测环境中验证检测性能，缺乏对模型泛化能力的客观评估；
- 仅关注校园/组织网络场景，其他工业、云等环境的适用性尚未验证。

---

## 72. Noise Reduction for Pufferfish Privacy: A Practical Noise Calibration Method

**arXiv ID:** 2601.06385 | [PDF](https://arxiv.org/pdf/2601.06385v1)

**作者:** Wenjin Yang `[一作]` (Beijing Institute of Technology), Liehuang Zhu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 14514 | [OpenAlex ID](https://openalex.org/A5100634361)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于1‑Wasserstein（Kantorovich）机制的松弛噪声校准方法，能够在满足pufferfish隐私的前提下显著降低噪声并提升数据效用。

**💡 创新点**

创新点在于放宽了原严格的噪声上限条件，并给出了可计算的根求解算法（改进的Brent方法），证明在任意隐私预算下都能得到更小的噪声参数，尤其在低隐私预算时噪声减少显著。

**🔧 技术方法**

采用了Wasserstein距离、Kantorovich最优传输、拉普拉斯噪声、理论证明以及改进的根求解算法来实现噪声校准。

**📊 数据集**

实验使用了三个公开的UCI数据集：Student Performance、Census Income和Bank Marketing。

**📈 对比分析**

通过与传统的W_1机制和ℓ1‑敏感度方法对比，实验结果显示所提出的方法在噪声参数上降低了47%–87%，从而在保持相同隐私水平下显著提升了数据效用。

**⚠️ 局限性**

局限性包括需要预先计算最优传输计划，且仅适用于有限可数字母表；算法复杂度与传输计划规模相关；在极端分布下噪声仍与ℓ1方法相近。

---

## 73. A novel RF-enabled Non-Destructive Inspection Method through Machine Learning and Programmable Wireless Environments

**arXiv ID:** 2601.06512 | [PDF](https://arxiv.org/pdf/2601.06512v1)

**作者:** Stavros Tsimpoukis `[一作]` (University of Ioannina), Christos K. Liaskos `[通讯]` (Foundation for Research and Technology Hellas)

**通讯引用:** 3365 | [OpenAlex ID](https://openalex.org/A5054811310)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于可编程无线环境（PWE）的无线射频（RF）感知方法，利用RF波前编码与生成对抗网络（GAN）实现工业工件的无损检测和数字孪生（DT）图像重建。

**💡 创新点**

创新点包括：
1) 将PWE视作可控制的RF “镜头”，通过优化方向到达（DoA）矩阵，使不同工件产生相似的RF波前；
2) 采用统计相关性匹配将视觉相似性映射到RF域；
3) 使用GAN将受噪声和量化影响的RF读取映射为高质量DT图像；
4) 结合图路由算法实现从发射器到接收阵列的可编程信号路径。

**🔧 技术方法**

核心技术：
- 软件定义金属表面（SDM）与PWE配置；
- RF波前编码与DoA优化（使用L‑BFGS‑B、相关矩阵匹配）；
- 图路由与射线追踪仿真；
- 条件GAN（pix2pix）用于图像生成；
- 图像相似度评估（SSIM、L2、PSNR、互信息、余弦相似度）。

**📊 数据集**

数据集：
- 40个不同工业工件的3D STL模型，共4000个旋转样本；
- 12个工件的形变序列，共6612个形变+旋转样本；
- 通过MATLAB ray‑tracing生成对应的RF读取，构成训练与测试集。

**📈 对比分析**

评估方法：
- 训练pix2pix后生成DT图像，与数据库原始图像做相似度匹配；
- 采用SSIM、欧氏距离、PSNR、互信息、余弦相似度等指标；
- 结果显示SSIM达到99.5%匹配率，且无物体匹配错误；L2/PSNR/互信息/余弦相似度稍低，存在角度误差。

**⚠️ 局限性**

局限与挑战：
- 受SDM代码书和硬件量化限制，RF读取与理想模式存在误差；
- 需要大规模、丰富的训练数据才能泛化到更多形状；
- 目前仅生成灰度图像，缺乏颜色与深度信息；
- 处理速度和实时性尚未在真实工业环境验证；
- 对极端遮挡、复杂多径环境的鲁棒性待进一步评估。

---

## 74. How to Assess AI Literacy: Misalignment Between Self-Reported and Objective-Based Measures

**arXiv ID:** 2601.06101 | [PDF](https://arxiv.org/pdf/2601.06101v1)

**作者:** Shan Zhang `[一作]` (University of Florida), Kenneth R. Koedinger `[通讯]` (Carnegie Mellon University)

**通讯引用:** 26125 | [OpenAlex ID](https://openalex.org/A5062550465)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究开发并评估了面向K–12教师的自评（SR）与客观测评（OB）AI素养工具，检验其心理测量属性，并基于两种测评结果构建学习者画像，分析教师对AI素养的自我感知与实际表现之间的匹配度。

**💡 创新点**

创新点在于：①首次在同一框架下同时构建并验证SR与OB两种测评工具；②通过潜在画像分析揭示教师在AI素养上的高估、低估与对齐模式；③将伦理维度纳入两种测评，丰富AI素养维度。

**🔧 技术方法**

技术方法包括：Rasch 1PL 模型（检验OB测评的可靠性与维度性）；探索性与确认性因子分析（确定SR与OB的结构）；潜在画像分析（LPA）用于聚类不同的自我感知与表现模式。

**📊 数据集**

数据集为 288 名来自台湾的K–12教师（含预备与在职教师）完成的问卷数据，其中 OB 部分包含 25 题多选，SR 部分包含 13 题 Likert 量表。

**📈 对比分析**

对比方法：计算 SR 与 OB 维度间的相关系数（普遍低于 0.25），并通过 LPA 评估模型拟合（AIC/BIC 选择 6 组画像，分类可信度平均 0.935）。表现表明 SR 与 OB 之间存在显著差异，且教师在有 AI 学习经历时低估/高估程度更为可控。

**⚠️ 局限性**

局限性包括：①未考虑教师职称、教龄等细分人群差异；②OB测评为通用情景题，缺少学科与年级针对性；③未涵盖最新的 AI 生成与检测维度，导致部分高阶技术项对教师而言过于技术化。

---

## 75. Evaluating Cross-Lingual Unlearning in Multilingual Language Models

**arXiv ID:** 2601.06675 | [PDF](https://arxiv.org/pdf/2601.06675v1)

**作者:** Tyler Lizzo `[一作]` (Georgia Institute of Technology), Larry Heck `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6381 | [OpenAlex ID](https://openalex.org/A5003679010)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估多语言LLM的跨语言忘却效果，比较多种忘却算法。

**💡 创新点**

发现并利用共享的“中间语言”子空间实现跨语言高效忘却。

**🔧 技术方法**

使用子空间投影（UNLEARN）技术，并与梯度上升、KL、FLAT等传统方法对比。

**📊 数据集**

使用译制TOFU基准，覆盖七种语言/脚本。

**📈 对比分析**

采用忘却质量p值和模型效用的双指标进行比较；在多语言模型中，UNLEARN在跨语言忘却上显著优于其它方法，且几乎不损失模型效用。

**⚠️ 局限性**

局限性：实验语言覆盖有限，缺乏低资源或极端形态语言的评估；仅针对事实知识，未涉及程序性或推理行为。

---

## 76. IndRegBias: A Dataset for Studying Indian Regional Biases in English and Code-Mixed Social Media Comments

**arXiv ID:** 2601.06477 | [PDF](https://arxiv.org/pdf/2601.06477v1)

**作者:** Debasmita Panda `[一作]` (Indian Institute of Science Education and Research), Neelesh Kumar Shukla `[通讯]` (Oracle Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建并标注了25,000条来自Reddit和YouTube的印度地区偏见评论，形成IndRegBias数据集，包含是否偏见、严重程度（轻/中/重）和目标地区的多级标签。

**💡 创新点**

提出针对印度多语言、代码混合文本的多级标注策略以及专门的区域偏见检测数据集，首次在零射击、少射击和微调三种设置下系统评估LLM在检测与级别排序上的性能。

**🔧 技术方法**

使用链式思考提示实现零射击/少射击，采用参数高效微调（LoRA）与指令式微调对LLM进行微调，评估多种开源LLM/ILM。

**📊 数据集**

IndRegBias（25k条评论）数据集，包含印度36个州/地区的偏见表达，语言多为英语与多种印度语混合。

**📈 对比分析**

在二分类上，零射击平均F1≈0.78，少射击提升至≈0.82，微调后达到≈0.90；在多分类（严重程度）上，零射击Precision/Recall低，微调后显著提升，尤其在重度类别。

**⚠️ 局限性**

数据存在地区与州分布不均，严重程度判定主观；LLM在安全对齐下可能拒绝输出导致表现低估。

---

## 77. Hierarchical Pooling and Explainability in Graph Neural Networks for Tumor and Tissue-of-Origin Classification Using RNA-seq Data

**arXiv ID:** 2601.06381 | [PDF](https://arxiv.org/pdf/2601.06381v1)

**作者:** Thomas Vaitses Fontanari `[一作]` (Universidade Federal do Rio Grande do Sul), Mariana Recamonde-Mendoza `[通讯]` (Hospital de Clínicas de Porto Alegre)

**通讯引用:** 1288 | [OpenAlex ID](https://openalex.org/A5039060913)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在RNA‑seq数据上使用层次聚类池化的图神经网络（GNN）来区分肿瘤与正常样本及肿瘤组织来源。

**💡 创新点**

首次结合权重池化与多层Chebyshev卷积，并在超节点层面进行可解释性分析，展示了GNN在癌症生物标记与生物过程识别上的潜力。

**🔧 技术方法**

使用Chebyshev图卷积、加权池化、ReLU、dropout、批归一化、全连接分类器以及梯度热力图（saliency）和过度表示分析（ORA）等技术。

**📊 数据集**

采用TCGA RNA‑seq（FPKM）与STRING蛋白互作网络，最终包含14133个基因，样本包含7008个肿瘤与701个正常样本。

**📈 对比分析**

与仅使用全连接网络的基线对比，并逐层增深网络；单层池化+卷积获得最高F1‑macro 0.978，肿瘤预测准确率达99.3%，与现有最先进方法相近。

**⚠️ 局限性**

限制包括：多层池化导致过平滑、性能下降；预先固定的层次聚类难以适应训练、未探索可学习池化或更复杂的图卷积结构，以及对单一数据集的依赖。

---

## 78. On the Fallacy of Global Token Perplexity in Spoken Language Model Evaluation

**arXiv ID:** 2601.06329 | [PDF](https://arxiv.org/pdf/2601.06329v1)

**作者:** Jeff Chan-Jan Sju `[一作]` (Carnegie Mellon University), Carlos Busso `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12922 | [OpenAlex ID](https://openalex.org/A5040793194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文改进了生成式语音模型的评估方法，提出了局部化和归一化的困惑度评估以及基于生成结果的嵌入模型判别评估。

**💡 创新点**

创新点在于将语音的短时局部性与归一化融入困惑度计算，并通过 embedding‑as‑judge 实现模型自评，从而更贴合人类感知。

**🔧 技术方法**

使用的技术包括局部化困惑度（Localized Perplexity）、归一化困惑度（Normalized Perplexity）以及基于嵌入模型的判别评估（Embedding‑as‑Judge）。

**📊 数据集**

采用 SALMon 基准集的六个子任务，对六个不同的语音语言模型进行评估。

**📈 对比分析**

与传统全局 token perplexity 相比，新方法与 MOS 的相关性提升至 0.8 以上，最佳模型在 SALMon 上达成人类极限的 83% 闭合，显示性能显著提升。

**⚠️ 局限性**

局限性在于仅在现有 SALMon 基准上测试，无法系统评估复杂组合变化（如嘈杂背景下的说话人切换），评估范围受限。

---

## 79. Towards Egocentric 3D Hand Pose Estimation in Unseen Domains

**arXiv ID:** 2601.06537 | [PDF](https://arxiv.org/pdf/2601.06537v1)

**作者:** Wiktor Mucha `[一作]` (TU Wien), Martin Kampel `[通讯]` (TU Wien)

**通讯引用:** 2415 | [OpenAlex ID](https://openalex.org/A5081324708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对第一人称视角 3D 手姿估计的跨域泛化问题，提出了 V-HPOT 方法，在单阶段网络中实现了跨域性能提升。

**💡 创新点**

创新点包括：①在虚拟摄像机空间预测 z 坐标，消除相机内参对深度估计的依赖；②设计了自监督测试时优化（TTO）框架，利用 3D 空间一致性损失对深度感知进行在线细化。

**🔧 技术方法**

技术实现包括：基于 EfficientNetV2-S 的单阶段网络，伪深度辅助任务（DPT‑Hybrid）训练，虚拟摄像机映射将深度归一化，TTO 的多尺度 3D 一致性损失以及标准数据增强。

**📊 数据集**

使用的数据集为：源域 HOT3D（RGB+多视角），目标域 H2O（RGB+Depth）、AssemblyHands（单色）和 Epic‑Kpts（无 3D 标注的 4K 视角），并在这些数据集上进行跨域评估。

**📈 对比分析**

方法与现有单阶段模型（如 H2OTR、ArcticNet‑SF）以及两阶段模型（如 FrankMocap、WildHands）对比，在 MPJPE‑RA、MRRPE 等 3D 指标上均实现 71%/41% 的误差下降，Epic‑Kpts 2D L2 减少 30%，与两阶段方法竞争。

**⚠️ 局限性**

局限性包括：对摄像机靠近手部的极端近景表现不足；在 AssemblyHands 的强畸变场景中性能下降；需要在目标域预收集 5% 视图进行测试时优化，且仍受相机畸变和深度尺度不确定性的影响。

---

## 80. Why LoRA Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models

**arXiv ID:** 2601.06305 | [PDF](https://arxiv.org/pdf/2601.06305v1)

**作者:** Hoang-Chau Luong `[一作]` (Rochester Institute of Technology), Lingwei Chen `[通讯]` (Rochester Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LoRA的谱特性进行分析，发现其对后门攻击的脆弱性源于谱强度不足和谱对齐不佳，并提出改进方法RoRA。

**💡 创新点**

创新点是将后门忘记阈值与谱强度、对齐关系相结合，设计三项正则化（清洁强化、触发器不敏感、后训练谱缩放）以提升鲁棒性。

**🔧 技术方法**

使用谱分解、dropout正则化、正交惩罚以及后训练谱缩放等技术对LoRA进行改进，并在实验中与FFT等方法对比。

**📊 数据集**

在BERT、RoBERTa、LLaMA三大模型上，采用SST‑2、CR、CoLA数据集，并对BadNet和InSent两种后门攻击进行评估。

**📈 对比分析**

与FFT、LoRA及现有防御方法相比，RoRA在保持清洁准确率不变的前提下，攻击成功率降低70‑95%，鲁棒性显著提升。

**⚠️ 局限性**

局限性包括仅针对权重后门攻击和中等规模模型进行验证，未评估极大规模模型或更复杂的自适应攻击，且理论仍以线性谱为基础。

---

## 81. Neuro-Symbolic Compliance: Integrating LLMs and SMT Solvers for Automated Financial Legal Analysis

**arXiv ID:** 2601.06181 | [PDF](https://arxiv.org/pdf/2601.06181v1)

**作者:** Yung-Shen Hsia `[一作]` (National Cheng Chi University), Jie-Hong Roland Jiang `[通讯]` (National Taiwan University)

**通讯引用:** 1607 | [OpenAlex ID](https://openalex.org/A5078604015)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Neuro‑Symbolic Compliance 框架，利用 LLM 将金融法规文本转换为 SMT 约束，并通过 SMT 求解器实现合规校验和最小事实修正。

**💡 创新点**

创新点包括：① 将 LLM 与 SMT 的交互构成神经‑符号两层系统；② 通过 RAG 机制补充相关条款；③ 采用 MaxSMT 优化寻找最小合规修正；④ 设计多代理平台实现可解释、可交互的合规分析。

**🔧 技术方法**

使用技术：大型语言模型（GPT‑4/LLaMA‑2）、Z3 SMT 求解器、RAG（BM25 + 向量检索）、CrossEncoder 重排序、LLM 生成约束、MaxSMT 优化、Gradio UI 等。

**📊 数据集**

数据集：台湾金融监督委员会（FSC）87 个执法案例、75 条法律共 3,753 条条款、官方法典数据库以及检索语料库。

**📈 对比分析**

与单独 LLM 基线比较，评估指标包括：SMT 代码生成准确率 86.2%；非法条款检测平均 5.08 条/案，耗时 0.021 s（LLM 6.40 条/案 7.01 s）；合规恢复准确率 1.0（LLM 0.3333），平均推理时间 0.004 s（LLM 1.47 s）。整体提升约 100× 速度，准确率大幅提升。

**⚠️ 局限性**

局限性：仅在台湾金融监管场景验证；跨司法辖区和多语言迁移需要重写检索语料和映射词典；LLM 仍需人工验证；对极大规模约束集的性能尚未评估。

---

## 82. Tree-Preconditioned Differentiable Optimization and Axioms as Layers

**arXiv ID:** 2601.06036 | [PDF](https://arxiv.org/pdf/2601.06036v1)

**作者:** Yuexin Liao `[一作]` `[通讯]` (California Institute of Technology), Yuexin Liao (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种可微分框架，将随机效用模型（RUM）的公理直接嵌入深度网络，并通过树预条件化的内部点法实现高效投影；

**💡 创新点**

创新点在于将RUM一致性等价为布尔格子上的流守恒，设计基于树结构的预条件器和隐函数定理的可微分投影层，突破了传统RUM投影仅能处理5个备选项的规模瓶颈；

**🔧 技术方法**

采用的技术包括Block‑Marschak Möbius变换、布尔格子流模型、树预条件化的共轭梯度求解、隐函数定理实现的可微分层以及JIT编译加速；

**📊 数据集**

主要使用合成的选择集数据进行验证，实验覆盖n=8、10、20等规模，并对稀疏选择集进行低秩加速实验；

**📈 对比分析**

与无预条件、Jacobi预条件等基线比较，迭代次数从数百降至几十，速度提升数十倍，支持n>20的投影且梯度准确，泛化性能优于软约束方法；

**⚠️ 局限性**

局限性包括仍需在每次内部点迭代中重建树预条件器，计算量随备选项数指数增长；在选择集不具低秩结构时，预条件效果下降，最坏情况迭代次数受秩上界限制。

---

## 83. Burn-After-Use for Preventing Data Leakage through a Secure Multi-Tenant Architecture in Enterprise LLM

**arXiv ID:** 2601.06627 | [PDF](https://arxiv.org/pdf/2601.06627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 84. Cognitive Sovereignty and the Neurosecurity Governance Gap: Evidence from Singapore

**arXiv ID:** 2601.06040 | [PDF](https://arxiv.org/pdf/2601.06040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 85. One-Shot Hierarchical Federated Clustering

**arXiv ID:** 2601.06404 | [PDF](https://arxiv.org/pdf/2601.06404v1)

**作者:** Shenghong Cai `[一作]` (Guangdong University of Technology), Yiu-Ming Cheung `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 9860 | [OpenAlex ID](https://openalex.org/A5038516431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种一轮通信的分层联邦聚类框架 Fed‑HIRE，利用客户端的粗粒度聚类子簇与服务器的多粒度竞争惩罚学习构建全局层次结构，并将其嵌入增强表示进行聚类。

**💡 创新点**

创新点在于：①引入 FCPL 与 MCPL 的竞争惩罚机制，自动适应不同粒度的局部与全局分布；②采用单向原型级通信，避免多轮通信导致的通信与隐私开销；③通过多粒度层次结构编码提升全局聚类的鲁棒性与准确性。

**🔧 技术方法**

技术方法包括：竞争惩罚学习、粗粒度与多粒度聚类、层次结构生成、增强表示编码、单轮联邦通信与差分隐私/同态加密等。

**📊 数据集**

使用了10个公开 UCI 数据集（Ecoli、User Knowledge Modeling、Statlog(Vehicle Silhouettes)、HCV、Yeast、Cardiotocography、Statlog(Landsat Satellite)、Wine Quality、Pen‑Based Digits、Letter Recognition），数据均为表格形式。

**📈 对比分析**

与七种最先进方法（FedSC、FFCM‑1/2、AFCL、kFed、OSFSC、NN‑FC）在四个聚类有效性指标（Purity、ARI、NMI、ACC）上进行比较，Fed‑HIRE 在大多数指标上平均排名第一、表现稳健，显著优于对手。

**⚠️ 局限性**

局限性在于仅针对结构化表格数据，未考虑图像、视频、图结构或多模态数据的联邦聚类任务。

---

## 86. Bridging Robustness and Efficiency: Real-Time Low-Light Enhancement via Attention U-Net GAN

**arXiv ID:** 2601.06518 | [PDF](https://arxiv.org/pdf/2601.06518v1)

**作者:** Yash Thesia `[一作]` (New York University), Meera Suthar `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种融合注意力门的U-Net生成对抗网络，用于实时低光图像增强，避免了扩散模型的高延迟问题。

**💡 创新点**

创新点在于证明单次前向传播即可近似扩散模型的高频纹理恢复，并通过注意力门抑制噪声、PatchGAN对局部纹理进行逼真化。

**🔧 技术方法**

采用原始 Bayer 传感器数据、注意力门（Attention Gates）U‑Net 生成器、条件GAN 损失与多尺度SSIM 损失、PatchGAN 判别器。

**📊 数据集**

使用 See‑in‑the‑Dark (SID) 数据集（Sony 子集）进行训练与评估，包含 2697 对短曝光原始图像与长曝光参考图像。

**📈 对比分析**

与扩散模型（DiffLight、Zero‑Shot LDM）及高效基线（SID U‑Net、EnlightenGAN、BM3D）对比，取得 LPIPS 0.112、PSNR 28.96 dB、SSIM 0.788，并将推理时间降至 0.062 s，实现 40× 的速度提升。

**⚠️ 局限性**

局限性包括仅在静态图像上验证，缺乏视频时序一致性评估，对其他低光数据集的泛化能力尚未充分验证。

---

## 87. ECLIPTICA - A Framework for Switchable LLM Alignment via CITA - Contrastive Instruction-Tuned Alignment

**arXiv ID:** 2601.06157 | [PDF](https://arxiv.org/pdf/2601.06157v1)

**作者:** Kapil Wanaskar `[一作]` (San Jose State University), Amitava Das `[通讯]` (Pragya Lab BITS Pilani Goa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 ECLIPTICA 框架，将 LLM 的对齐从训练后固定转为运行时可由自然语言对齐指令控制；并实现 CITA（Contrastive Instruction‑Tuned Alignment）算法，使单一模型能够在同一用户提示下根据不同的对齐指令切换行为。

**💡 创新点**

创新点主要有三：
1) 将对齐视为可切换的指令驱动行为契约，打破传统一次训练‑一次部署的“单策略”限制；
2) CITA 在对比偏好优化中引入强制 KL 约束（信任区间），确保不同指令对应的策略保持在同一 Riemannian 图上，避免坍塌并实现稳定的策略切换；
3) 发布了 ECLIPTICA 基准（3,000 条固定用户提示 × 10 种对齐指令），提供了在保持提示不变时隔离对齐指令影响的实验平台。

**🔧 技术方法**

技术细节包括：
- 先对模型做标准 SFT；
- 采用对比偏好学习（DPO 风格）以指令为条件学习 Y⁺ ≻ Y⁻；
- 加入强制 KL 约束 λ E[KL(πθ(·|I,X)||π₀(·|I,X))]，π₀ 为冻结的参考策略；
- 指令生成与筛选流程（五个判别模型 + BERTScore 一致性 + 人工质量门控）；
- 指令化评估指标 instruction‑alignment efficiency、AQI 等。

**📊 数据集**

使用的数据集包括：
- Llama‑3.1‑8B 作为模型基线；
- ECLIPTICA（ISD‑Instruction‑Switch‑Dataset）用于指令切换评测；
- TruthfulQA、Conditional Safety、Length Control、LITMUS 等公开基准；
- Anthropic HH‑rlhf 偏好样本用于训练 CITA 的对比偏好对。

**📈 对比分析**

与传统方法比较：CITA 在 ECLIPTICA 基准上达到 86.7% instruction‑alignment efficiency，显著高于 DPO（56.1%）、GRPO（36.1%）和 PPO（20.4%）。
在 TruthfulQA 上提升 0.054（vs DPO 0.001），在 Conditional Safety 上提升 0.391（vs DPO 0.475），在 Length Control 上提升 0.164（vs DPO 0.130），在 AQI 上提升 26.4 分。整体表明 CITA 在指令驱动对齐和可切换性方面具有显著优势。

**⚠️ 局限性**

局限性：
1) 仅在单一模型（Llama‑3.1‑8B）与单语言（英文）上验证，缺乏跨模型、跨规模、多语种的泛化实验；
2) 对齐指令的语义一致性与鲁棒性仍有风险，微小的指令改写可能导致切换失效或行为混合；
3) 对齐指令可能被恶意利用，需额外的指令过滤与安全层；
4) 评估基准集中在固定提示与对齐指令的组合，对真实部署中的动态提示与长上下文影响未充分探测；
5) 由于 KL 约束的权重选择对结果敏感，需进一步调优与理论分析。

---

## 88. Lexical and Statistical Analysis of Bangla Newspaper and Literature: A Corpus-Driven Study on Diversity, Readability, and NLP Adaptation

**arXiv ID:** 2601.06041 | [PDF](https://arxiv.org/pdf/2601.06041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 89. Dynamic Intelligence Ceilings: Measuring Long-Horizon Limits of Planning and Creativity in Artificial Systems

**arXiv ID:** 2601.06102 | [PDF](https://arxiv.org/pdf/2601.06102v1)

**作者:** Truong Xuan Khanh `[一作]` (H&K Research Studio), Truong Quynh Hoa `[通讯]` (H&K Research Studio)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出动态智能天花板（Dynamic Intelligence Ceiling）概念，并设计 Workshop World 这一程序化生成的 benchmark，用轨迹中心的评估框架测量 AI 系统在长期内的规划与结构创造力，从而捕捉智能前沿随时间的移动。

**💡 创新点**

创新点在于把智能视为随轨迹变化的前沿而非静态值，提出 Progressive Difficulty Ceiling（PDC）和 Ceiling Drift Rate（CDR）两个指标；同时通过整合规划与结构创造的单一环境，区分表面性能提升与真正前沿扩展。

**🔧 技术方法**

使用轨迹中心评估框架、程序化生成的 Workshop World 环境、可量化的结构签名与相似度度量、固定成功阈值 0.7，以及统计学分析方法来估计 PDC 与 CDR。

**📊 数据集**

利用自生成的 Workshop World 实例集合，参数化难度向量 δ=(H,K,C,A) 产生多层次实例；不依赖公开的静态数据集。

**📈 对比分析**

通过在不同开发阶段对成功率、效率、结构新颖性进行评估，并计算 PDC 与 CDR 来比较系统性能。结果显示部分系统在后期表现出正向 CDR 并保持新颖性，表明前沿扩展；而 CDR 维持零则说明早期固定，性能提升仅停留在已达前沿内。

**⚠️ 局限性**

局限在于只关注规划与结构创造的度量，未覆盖语义或美学创新；Benchmark 的参数化设计限制了对真实世界复杂度的外推；并不能证明实现无界智能，仅诊断早期固定的问题。

---

## 90. Towards Infinite Length Extrapolation: A Unified Approach

**arXiv ID:** 2601.06113 | [PDF](https://arxiv.org/pdf/2601.06113v1)

**作者:** Nitin Vetcha `[一作]` `[通讯]` (Indian Institute of Science), Nitin Vetcha (Indian Institute of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种统一框架，将位置编码拆分为可自适应的乘法与加法项，并基于此设计了Adaptive Positional Encoding (APE)，实现无限上下文推断。

**💡 创新点**

创新点在于将位置编码视作注意力分数的乘法变换与加性偏置，利用自适应频率调制和多项式衰减偏置实现可扩展的长距离依赖。

**🔧 技术方法**

使用了Transformer架构、RoPE、ALiBi等已有位置编码理论，加入自适应旋转矩阵、温度调度与可学习偏置，并在TinyStories及LongTinyStories上进行实验。

**📊 数据集**

采用Microsoft TinyStories和自制的LongTinyStories（单词长度0–32k）进行评估。

**📈 对比分析**

在不同上下文窗口（64/128/256）下与RoPE、ALiBi对比，APE在极长提示（16k）下保持低困惑度，注意力熵介于两者之间，虽然内存占用和推理速度略低于ALiBi，但在低上下文窗口时仍优于其更大窗口训练的表现。

**⚠️ 局限性**

限制包括仅关注位置编码而非其他长序列技术、仅验证低维模型、未深入研究与LDCP之间的权衡，以及推理速度与内存仍不如ALiBi。

---

## 91. The Impact of Post-training on Data Contamination

**arXiv ID:** 2601.06103 | [PDF](https://arxiv.org/pdf/2601.06103v1)

**作者:** Muhammed Yusuf Kocyigit `[一作]` (Boston University), Caglar Yildirim `[通讯]` (Northeastern University)

**通讯引用:** 559 | [OpenAlex ID](https://openalex.org/A5000390546)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在 Qwen2.5 与 Gemma3 预训练模型中注入 GSM8K 与 MBPP 测试集的复制，随后对比预训练后、SFT 与 RL（GRPO）后模型在数学与编码基准上的表现，系统评估数据泄露在整个训练生命周期中的影响。

**💡 创新点**

创新点在于同时考察预训练、SFT 与 RL 三个阶段对泄露效应的作用，并从模型规模角度揭示 SFT 与 GRPO 对泄露信息的不同泛化与过估计差异，首次完成此类全周期对比。

**🔧 技术方法**

技术手段包括延长预训练、基于训练步骤的 Supervised Fine‑Tuning、以及 Group Relative Policy Optimization（GRPO）RL 方法，评估时使用 LM Evaluation Harness 与 math‑verify 等工具。

**📊 数据集**

数据集方面，污染集为 GSM8K 与 MBPP（每个 5 份复制），未污染对照集为 GSMPlus 与 HumanEval；预训练混合数据共 25B tokens，包含 OpenMath‑Instruct、CodeParrots 与 FineWeb‑Edu。

**📈 对比分析**

比较方法：对污染与清洁训练集分别训练后，在各基准上计算准确率/Pass@1 的差异，发现 SFT 主要提升污染基准，GRPO 同时提升污染与未污染基准；模型规模增大时 SFT 的过估计加剧，而 GRPO 的泛化提升更明显，性能差距可达 2–4%。

**⚠️ 局限性**

局限性包括仅研究 5 份复制的后期注入泄露、模型规模限于 4B、RL 使用规则奖励而非人类偏好，并未覆盖多源、多样化泄露形式，难以直接推广到更大、更复杂的真实场景。

---

## 92. SAPL: Semantic-Agnostic Prompt Learning in CLIP for Weakly Supervised Image Manipulation Localization

**arXiv ID:** 2601.06222 | [PDF](https://arxiv.org/pdf/2601.06222v1)

**作者:** Xinghao Wang `[一作]` (University of Science and Technology of China), Xiangzheng Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了弱监督图像篡改定位框架SAPL，利用边缘信息在CLIP中学习语义无关提示，实现精确定位。

**💡 创新点**

创新点在于边缘感知的上下文提示学习（ECPL）和分层边缘对比学习（HECL），将边缘信号注入文本与视觉特征，显著提升跨数据集泛化。

**🔧 技术方法**

使用CLIP、可学习提示、软边缘地图、对比学习、双队列机制、LoRA微调等技术。

**📊 数据集**

在CASIAv2上训练，评测于CASIAv1、Columbia、COVERAGE、IMD2020、NIST16等公开数据集。

**📈 对比分析**

与现有弱监督方法相比，平均I‑AUC提升约4.6%，与全监督MVSS‑Net差距缩小9.2%；定位P‑F1也高于最佳弱监督方案，接近全监督水平。

**⚠️ 局限性**

主要局限在于只针对局部篡改，对全图或全局篡改缺乏鲁棒性；未充分考虑频域特征，某些后处理操作仍影响性能。

---

## 93. CEDAR: Context Engineering for Agentic Data Science

**arXiv ID:** 2601.06606 | [PDF](https://arxiv.org/pdf/2601.06606v1)

**作者:** Rishiraj Saha Roy `[一作]` (Fraunhofer IIS), Fabian Kuech `[通讯]` (Fraunhofer IIS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于LLM代理的自动化数据科学平台CEDAR，通过结构化提示、交错生成文本与代码块以及本地安全执行，实现对Kaggle等数据集任务的完整工作流自动化；

**💡 创新点**

创新点在于将任务拆解为可管理的文本/代码交替块，采用多代理分工与函数调用实现高效上下文工程，并将代码执行与结果仅以摘要形式注入LLM，保证本地数据安全与可解释性；

**🔧 技术方法**

使用技术包括：多代理LLM架构（主导器、文本生成器、代码生成器），OpenAI/ollama函数调用与结构化JSON输出，Docker容器化代码执行，历史渲染压缩，前端Streamlit；

**📊 数据集**

主要在Kaggle竞赛数据集上验证，例如LLM fine‑tuning 分类赛；

**📈 对比分析**

在该竞赛中，自动生成约10‑20步完整Notebook，耗时约3分钟，相比人工手工脚本大幅节省时间且保持结果可审计；尚未给出与其他自动化框架的定量基准；

**⚠️ 局限性**

局限性包括：代理系统功能仍相对基础，缺乏独立验证与自我优化，依赖LLM准确性，模型需要本地部署且资源消耗大，安全性主要靠Docker，且未公开完整性能对比。

---

## 94. Employ SmartNICs' Data Path Accelerators for Ordered Key-Value Stores

**arXiv ID:** 2601.06231 | [PDF](https://arxiv.org/pdf/2601.06231v1)

**作者:** Frederic Schimmelpfennig `[一作]` (Johannes Gutenberg University Mainz), André Brinkmann `[通讯]` (Johannes Gutenberg University Mainz)

**通讯引用:** 2508 | [OpenAlex ID](https://openalex.org/A5011466225)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出并实现了基于 BlueField-3 SmartNIC 的 DPA‑Store，一个支持点查、插入和范围查询的无状态键值存储，利用 DPAs 直接在 NIC 本地内存中遍历学习型索引，减少 PCIe 交叉，提升吞吐。

**💡 创新点**

创新点在于：①将学习型索引部署在 NIC 侧 DPA 内存，借助 DPAs 的并行和零锁遍历实现高效查询；②采用插入缓冲区与批量更新、RCU stitching 的方式在保持一致性的同时将结构维护移交给主机；③对 BlueField‑3 硬件提出改进建议，进一步提升性能。

**🔧 技术方法**

使用 BlueField‑3 SmartNIC、其 16 颗 1.8 GHz RISC‑V DPAs、DPAs 访存与 DMA、学习型索引（piecewise linear approximation）、UDP/DPDK 网络协议、RDMA、ROLEX 作为基准。

**📊 数据集**

评测数据集包括 SOSD 生成的稀疏/稠密 synthetic、Facebook Face、Amazon、Wikipedia、OpenStreetMap 等 50 M/25 M 键值对。

**📈 对比分析**

通过 YCSB、统一与 Zipf 分布混合工作负载与 ROLEX 对比，DPA‑Store 在多数读/写/范围场景下吞吐可达 33 MOPS（点查）和 13 MOPS（范围），性能与或优于 ROLEX，且延迟更低；但写入吞吐仍受限。

**⚠️ 局限性**

主要局限是 BlueField‑3 DPA 内存访问延迟高以及主机到 NIC 的 DMA 写性能差，导致写操作和结构更新吞吐受限；此外，需要更快的 DPA‑memory 接口和主机‑NIC 传输支持才能进一步提升写入性能。

---

## 95. Rational Synthesizers or Heuristic Followers? Analyzing LLMs in RAG-based Question-Answering

**arXiv ID:** 2601.06189 | [PDF](https://arxiv.org/pdf/2601.06189v1)

**作者:** Atharv Naphade `[一作]` `[通讯]` (Carnegie Mellon University), Atharv Naphade (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造新数据集GroupQA，对大型语言模型在检索增强生成（RAG）中处理多文档冲突证据的行为进行系统评估；

**💡 创新点**

创新点在于从文档集合层面而非单一对比文档探究模型的聚合机制，揭示了重复（Illusory Truth Effect）和首因偏差（Primacy Effect）等非语义驱动的聚合习惯；

**🔧 技术方法**

采用基于提示的实验设计（标准、链式推理）、离散/连续概率评估、留一法因果重要性分析以及规模化模型对比；

**📊 数据集**

使用自建的GroupQA数据集（1,635个二元争议问题、15,058篇文档，包含立场与强度标注），并对公开模型如DeepSeek、Gemini、Llama‑3.1‑70B、Qwen‑3‑32B 进行实验；

**📈 对比分析**

通过多维指标（回答翻转率、阈值、模型可塑性、首因影响）对不同规模模型进行对比，结果显示大模型更稳定、可塑性更低；模型在重复证据下更易翻转，且首位证据对结果影响显著；

**⚠️ 局限性**

局限性包括：仅评估二元答案场景，缺乏多模态与元数据信息，未进行模型内部机制剖析，且数据来源仅为英文网页，可能无法推广到其他语言或领域；

---

## 96. Comment on arXiv:2511.21731v1: Identifying Quantum Structure in AI Language: Evidence for Evolutionary Convergence of Human and Artificial Cognition

**arXiv ID:** 2601.06104 | [PDF](https://arxiv.org/pdf/2601.06104v1)

**作者:** Krzysztof Sienicki `[一作]` `[通讯]` (Chair of Theoretical Physics of Naturally Intelligent Systems), Krzysztof Sienicki (Chair of Theoretical Physics of Naturally Intelligent Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过人类问卷与两款大型语言模型的回答计算CHSH值，并将LLM生成词汇的频率分布用玻色-爱因斯坦统计模型进行拟合。

**💡 创新点**

首次将Bell/CHSH不等式与人类认知与AI文本生成相结合，并提出用玻色-爱因斯坦分布拟合词频来暗示AI语言的量子结构。

**🔧 技术方法**

使用CHSH计算、t检验、玻色-爱因斯坦与马尔可夫-玻尔兹曼分布拟合、AIC/BIC模型选择等统计方法。

**📊 数据集**

人类问卷数据以及两种大型语言模型（如GPT-4和其他模型）生成的文本词频数据。

**📈 对比分析**

将玻色-爱因斯坦拟合结果与指数（MB）分布进行对比，认为BE拟合更优，但未与常见语言学基线模型（如Zipf、Log-normal等）充分比较；CHSH值高于2但不构成严格的Bell检验。

**⚠️ 局限性**

限制包括CHSH计算不满足Bell实验的基本假设、S=4超越量子界限、t检验采样单位不明确、rank-能量映射缺乏物理解释、BE拟合仅为经验拟合且缺乏标准基线比较。

---

## 97. Atomic-SNLI: Fine-Grained Natural Language Inference through Atomic Fact Decomposition

**arXiv ID:** 2601.06528 | [PDF](https://arxiv.org/pdf/2601.06528v1)

**作者:** Minghui Huang `[一作]` `[通讯]` (University of Texas at Austin), Minghui Huang (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了自然语言推理（NLI）的原子级别推理，提出并构建了大型原子级NLI数据集Atomic‑SNLI，并通过细粒度训练评估模型的可解释推理能力。

**💡 创新点**

创新点在于：①构造了大规模、质量可控的原子级NLI数据集；②将原子拆解与多样化生成（矛盾/中立）相结合；③证明细粒度训练显著提升多事实推理准确率，并实现了可解释的推理流程。

**🔧 技术方法**

主要技术包括：利用DecModel对假设进行原子拆解；使用LLM（如Qwen3‑32B）生成矛盾原子；训练DeBERTa、ELECTRA、BERT等NLI模型；采用概率求和聚合而非硬逻辑规则；在原子级别和句子级别进行评估。

**📊 数据集**

使用的数据集为：①原始SNLI；②基于SNLI生成的Atomic‑SNLI（训练/验证/测试三分，包含蕴含、中立、矛盾三类原子实例）。

**📈 对比分析**

实验对比：细粒度训练后在2–3原子事实的推理准确率提升10%+，F1提升8%+，保持句子级性能；与传统逻辑规则聚合相比，概率求和实现更均衡的精确率与召回率，降低单一错误导致的误判。

**⚠️ 局限性**

局限性包括：①LLM生成的矛盾样本可能缺乏真实多样性；②原子拆解模型可能产生错误或不完整的原子；③当前推理规则假设原子独立，现实中可能存在依赖关系，影响推理准确性。

---

## 98. CBMAS: Cognitive Behavioral Modeling via Activation Steering

**arXiv ID:** 2601.06109 | [PDF](https://arxiv.org/pdf/2601.06109v1)

**作者:** Ahmed H. Ismail `[一作]` (University of California, Berkeley), Sean O'Brien `[通讯]` (University of California, San Diego)

**通讯引用:** 1535 | [OpenAlex ID](https://openalex.org/A5113548970)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CBMAS诊断框架，通过连续激活向量调节来分析LLM中的认知偏差传播。

**💡 创新点**

创新点在于将偏差调节从离散干预扩展为连续α轨迹，并结合logit lens、α‑sweep、层级响应分析，揭示倾向转折点与层级传播。

**🔧 技术方法**

使用激活调节（Steering vector）、logit lens、α‑sweep、层级敏感性分析、随机/正交对照向量、KL散度等技术。

**📊 数据集**

构建了对比提示对的语料库（如支持/不支持回应对），并使用GPT‑2 Small等模型进行实验。

**📈 对比分析**

与随机向量、正交向量对照以及传统二元偏差评估方法比较，展示了在不同注入层和α值下的Δlogit、KL、句法流畅性等指标，表明在特定层/α处可实现高效且稳定的偏差调节。

**⚠️ 局限性**

局限包括仅评估下一步 token，未探究序列级持续性；指标范围有限，未覆盖连贯性、事实性等；缺乏对注意力头、MLP 特征等底层机制的因果研究。

---

## 99. TeleMem: Building Long-Term and Multimodal Memory for Agentic AI

**arXiv ID:** 2601.06037 | [PDF](https://arxiv.org/pdf/2601.06037v1)

**作者:** Chunliang Chen `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 61163 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的长时记忆与多模态记忆系统，利用叙事动态抽取和结构化写入管线，在对话历史中保持一致的用户画像，提升写入效率，并通过ReAct式推理实现对视频内容的精确观测-思考-行动；

**💡 创新点**

1）使用仅对话支持的叙事单元抽取，消除基于模式的幻觉；2）引入批处理式写入管线（摘要、检索、聚类、决策）显著提升写入效率和存储利用率；3）结合多模态记忆模块与ReAct推理，实现视频内容的闭环多模态推理；

**🔧 技术方法**

叙事动态抽取、摘要与检索、语义聚类、LLM驱动的增删改决策、ReAct式工具使用、视频剪辑分割、视觉语言模型（VLM）生成字幕与实体、跨模态嵌入与向量检索；

**📊 数据集**

ZH-4O（中文角色扮演对话基准，28个真实人机会话，平均600回合，1068道多选提问），并对比多种基线系统；

**📈 对比分析**

与RAG、Mem0、MOOM、A-Mem、Memobase及全上下文LLM进行比较；在ZH-4O上取得86.33% QA准确率，较Mem0提升19%，相较于Memobase提升约9.55%，并在token使用上减少43%，速度提升2.1×；

**⚠️ 局限性**

仅在ZH-4O单一数据集上评估，缺乏跨语言或跨任务验证；对极长会话或极大记忆规模的可扩展性未充分实验；系统仍依赖外部向量数据库，写入与检索延迟可能在高并发场景中成为瓶颈；

---

## 100. CrossTrafficLLM: A Human-Centric Framework for Interpretable Traffic Intelligence via Large Language Model

**arXiv ID:** 2601.06042 | [PDF](https://arxiv.org/pdf/2601.06042v1)

**作者:** Zeming Du `[一作]` (Beijing University of Technology), Yong Zhang `[通讯]` (Beijing University of Technology)

**通讯引用:** 69443 | [OpenAlex ID](https://openalex.org/A5100419770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种跨模态框架 CrossTrafficLLM，能够同时预测未来交通状态并生成对应的自然语言描述。

**💡 创新点**

核心创新是将文本信息通过文本引导的自适应图卷积网络与交通图结构对齐，实现交通预测与文本生成的深度耦合与互补。

**🔧 技术方法**

技术组合包括 Crossformer（双阶段注意力的图变压器）、文本引导自适应 GCN、DeepSeek 预训练大语言模型、LoRA 微调、以及路段重要性检测与交叉注意力机制。

**📊 数据集**

使用北京交通-文本（BjTT）大规模多模态数据集，包含 1,260 条道路的速度序列、路网拓扑和对应的异常事件文本。

**📈 对比分析**

与多种基准（STGCN、GWN、ASTGCN、Crossformer、Informer、TimeXer 等）比较，CrossTrafficLLM 在交通预测的 MAE/RMSE 上均优于所有对手，且在文本生成的 BLEU‑4、METEOR、ROUGE‑L 上分别提升约 27%、13%、6%。

**⚠️ 局限性**

局限性在于仍主要依赖结构化交通数据，未充分利用天气、社交媒体等额外非结构化信息，模型在跨城市泛化和实时推理速度方面尚未充分验证。

---

## 101. NAS-GS: Noise-Aware Sonar Gaussian Splatting

**arXiv ID:** 2601.06285 | [PDF](https://arxiv.org/pdf/2601.06285v1)

**作者:** Shida Xu `[一作]` (Imperial), Sen Wang `[通讯]` (Imperial)

**通讯引用:** 8354 | [OpenAlex ID](https://openalex.org/A5100350760)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了NAS-GS框架，实现了基于高斯投影的噪声感知水声成像重建与新视角合成。

**💡 创新点**

创新点在于两向投影（Two‑Ways Splatting）与可学习的高斯混合噪声模型，兼顾声波极坐标投影与多方向传播。

**🔧 技术方法**

采用了3D高斯渲染、极坐标投影转换、双向渲染、可学习GMM噪声、以及多视角损失和体素化等技术。

**📊 数据集**

使用了模拟场景（Cabinet、Barrel、Panel）以及北海风电塔基座的真实海上声纳数据。

**📈 对比分析**

与Neusis、ZSplat等基线对比，NAS-GS在PSNR、SSIM、LPIPS、Chamfer/ Hausdorff等指标上均优于对手，渲染速度超过700 FPS。

**⚠️ 局限性**

局限在于当前仅处理静态场景，缺乏动态物体或实时SLAM支持，且对极端噪声分布仍有改进空间。

---

## 102. Multi-Agent Framework for Controllable and Protected Generative Content Creation: Addressing Copyright and Provenance in AI-Generated Media

**arXiv ID:** 2601.06232 | [PDF](https://arxiv.org/pdf/2601.06232v1)

**作者:** Haris Khan `[一作]` (National University of Sciences and Technology), Shumaila Asif `[通讯]` (National University of Sciences and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多智能体框架，用于实现可控且带有数字版权保护的AI生成内容，支持文本、图像乃至多模态创作。

**💡 创新点**

创新点在于将生成流程拆分为导演、生成器、评审、整合与保护五个专门角色，并在生成过程中实时嵌入水印/指纹，实现边生成边保护与可追溯；同时提供人机交互的控制点，让用户在任何阶段可干预。

**🔧 技术方法**

核心技术包括：GPT‑4作为导演/规划器、Stable Diffusion XL（或相似扩散模型）用于内容生成、CLIP 进行语义对齐评估、专门的水印嵌入与提取算法（基于扩散模型的隐写）、以及完整的流程编排与日志记录系统。

**📊 数据集**

实验使用公开的文本-图像检索/生成模型与常用评估数据集（如 COCO、ImageNet 等），并结合人工标注的评审结果进行对比。

**📈 对比分析**

与单步生成相比，基于任务拆分的 CLIPScore 提升 20–25%；与后置水印方法相比，内嵌水印的恢复率超过 90%（而传统方法约 70%）；用户在迭代满意度上从 4–5 次降至 2–3 次，验证了可控性与效率提升。

**⚠️ 局限性**

主要限制包括多智能体编排带来的计算开销、对高级对抗攻击下水印鲁棒性的挑战，以及目前用户体验评估规模有限，缺乏大规模跨域验证。

---

## 103. AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages

**arXiv ID:** 2601.06395 | [PDF](https://arxiv.org/pdf/2601.06395v1)

**作者:** Hao Yu `[一作]` (McGill University), David Ifeoluwa Adelani `[通讯]` (Mila-Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对多种开源LLM进行持续预训练，适配20种非洲语言，构建了AfriqueLLM系列模型。

**💡 创新点**

①数据混合（语料+代码+数学+合成翻译）显著提升低资源语言能力；②证明强大基础模型（如Qwen3）在持续预训练后对非洲语言提升巨大；③提供高质量合成数据桥梁。

**🔧 技术方法**

持续预训练（CPT）、动态数据混合策略、多模型架构（Llama 3.1、Gemma 3、Qwen 3）、大规模分布式训练框架（LLaMA-Factory、DeepSpeed、FlashAttention）等技术。

**📊 数据集**

20种非洲语言的 FineWeb2、WURA、MADLAD-400、CornStack-Python、FineMath-4+、NLLB-OPUS、GPT‑4.1 翻译合成语料，共计约26 B tokens。

**📈 对比分析**

在 AfroBench‑Lite（7项任务）和文档级翻译任务上使用5‑shot/8‑shot评估，性能相较基线提升 20%–80% 以上，尤其 Qwen 3 8B/14B 在多项任务达 60%–70% 以上。

**⚠️ 局限性**

仅覆盖20种语言、模型规模受限至14 B、未做指令调优、数据/超参搜索有限、训练稳定性待提升。

---

## 104. CulinaryCut-VLAP: A Vision-Language-Action-Physics Framework for Food Cutting via a Force-Aware Material Point Method

**arXiv ID:** 2601.06451 | [PDF](https://arxiv.org/pdf/2601.06451v1)

**作者:** Hyunseo Koh `[一作]` (Soongsil University), Heewon Kim `[通讯]` (Soongsil University)

**通讯引用:** 3173 | [OpenAlex ID](https://openalex.org/A5101422101)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在可变形物体（食物）切割任务中，提出了 Vision‑Language‑Action‑Physics (VLAP) 框架，将多模态 VLA 模型与 MLS‑MPM 物理模拟相结合，生成了大规模多模态切割数据集 CulinaryCut，并在此基准上评估了现有 VLA 模型。

**💡 创新点**

创新点在于（1）构建了兼顾视觉、语言、动作和物理约束的统一框架；（2）使用 MLS‑MPM 对切割过程进行精细的拓扑与力学建模；（3）通过 LLM 合成、模拟驱动的数据增强生成覆盖多种切割样式、比例与目标的多模态数据；（4）引入 CSTM 模块实现切割风格的迁移与动态平滑。

**🔧 技术方法**

技术包括：多模态 VLA 模型（OpenVLA、Octo、RDT‑1B 等）、ManiSkill 机器人仿真、MLS‑MPM 物理仿真、CPIC 触碰算法、基于力预测的安全阈值控制、LLM 生成语言指令、动作增广与运动规划。

**📊 数据集**

数据集为自研的 CulinaryCut，包含 325,000 条基于 MLS‑MPM 生成的切割轨迹，覆盖 7 种食材、5 种切割风格、13 种切割状态（比例 0.1–0.9、中心、分割），每条轨迹配备多模态观测与超过 5 条语言指令。

**📈 对比分析**

采用单目标、多人目标、未见比例与未见语言三种评测场景。RDT 在单目标 0.5 比例切割上达约 68% 成功率，单目标小物体（如莓果）下降明显；多目标场景下成功率降至 31%（RDT）、17%（Octo）和 28%（OpenVLA）。在比例泛化上，RDT 对 0.25/0.75 等比率的泛化率低至 20–0%，显示出对方向与比例的对称性不足。总体来看，RDT 在多数基准上优于 Octo 与 OpenVLA，但仍存在显著的精准度与泛化瓶颈。

**⚠️ 局限性**

局限性包括：（1）对小尺寸食材的切割精度不足；（2）多目标场景下目标识别与定位仍不稳定；（3）对比例与方向的语义对齐存在显著偏差；（4）仿真与真实物理的差距仍未完全消除，导致 sim‑to‑real 转移的可靠性受限；（5）数据生成过程对 LLM 与仿真参数的依赖较高，可能导致样本分布偏差。

---

## 105. SecureDyn-FL: A Robust Privacy-Preserving Federated Learning Framework for Intrusion Detection in IoT Networks

**arXiv ID:** 2601.06466 | [PDF](https://arxiv.org/pdf/2601.06466v1)

**作者:** Imtiaz Ali Soomro `[一作]` (Sir Syed CASE Institute of Technology), Heejung Yu ID `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对IoT网络的联邦学习式入侵检测，提出 SecureDyn-FL 框架，实现隐私保护、抗中毒、对非IID数据自适应。

**💡 创新点**

创新点：① 动态时间梯度审计（GMM+Mahalanobis），能识别隐蔽自适应中毒；② 变形 ElGamal 同态加密与稀疏量化相结合，保证通信安全且高效；③ 双目标个性化学习（全局交叉熵 + Logit‑Adjusted 失调损失）提升异构数据下的检测性能。

**🔧 技术方法**

核心技术包括联邦学习、GMM 与 MD 梯度审计、加密（Post‑Cramer ElGamal）、稀疏剪枝、量化、Logit‑Adjusted 损失、异构数据的 Dirichlet 划分与混合攻击仿真。

**📊 数据集**

使用公开 IoT 入侵数据集：N‑BaIoT（binary 与 10 类攻击）与 TON_IoT（multi‑class）以及其 mini‑版本作为实验基准。

**📈 对比分析**

与 FedAvg、Trimmed Mean、ShieldFL、FL‑Defender、FL‑Trust 等 SOTA 方法对比，在 IID 与非IID、攻击率 10%–50% 的场景下，SecureDyn‑FL 达到 99%+ 准确率、0.989 F1，攻击成功率 ≤0.05，通信与计算开销相比传统方法显著下降。

**⚠️ 局限性**

局限性：仍需可信方或 TEE 进行密钥/审计管理；加密与剪枝/量化实现对低算力设备有一定负担；聚合阶段仍要求大多数客户端参与，扩展到千级 IoT 时需要客户端选择与时延容忍机制；未充分评估针对高度自适应的对抗攻击。

---

## 106. From Lagging to Leading: Validating Hard Braking Events as High-Density Indicators of Segment Crash Risk

**arXiv ID:** 2601.06327 | [PDF](https://arxiv.org/pdf/2601.06327v1)

**作者:** Yechen Li `[一作]` (Google Research), Feng Guo `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用连接汽车产生的硬制动事件（HBE）数据，结合加利福尼亚和弗吉尼亚州的警方报告碰撞数据，对道路段级别的碰撞风险进行验证与预测。

**💡 创新点**

将HBE作为高密度安全代理指标，证明其与实际碰撞率显著正相关，为大规模道路安全评估提供可行的替代方案。

**🔧 技术方法**

采用负二项回归（Negative Binomial GLM）模型，并控制道路类型、车道数、坡度、加速器等混杂因素，评估HBE与碰撞率的关系。

**📊 数据集**

加州和弗吉尼亚各10年的撞车数据库；Google Android Auto平台收集的匿名化HBE数据和Google Maps Traffic Trends交通量数据。

**📈 对比分析**

在传统碰撞率与道路特征回归模型的基础上加入HBE率后，模型拟合度显著提升；两州模型中HBE率均得到显著正系数，表明其预测能力强。

**⚠️ 局限性**

主要限制包括HBE与撞车数据时间窗口不一致（HBE仅来自最近一个月，撞车数据跨多年），以及HBE率受交通量波动影响，无法精准捕捉短期高风险事件。

---

## 107. EntroLnn: Entropy-Guided Liquid Neural Networks for Operando Refinement of Battery Capacity Fade Trajectories

**arXiv ID:** 2601.06195 | [PDF](https://arxiv.org/pdf/2601.06195v1)

**作者:** Wei Li `[一作]` (Singapore Institute of Technology), Qingyu Yan `[通讯]` (Nanyang Technological University)

**通讯引用:** 59363 | [OpenAlex ID](https://openalex.org/A5006577991)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 EntroLnn 框架，利用熵引导的液态神经网络对锂离子电池的容量衰退轨迹进行实时在线细化，并实现了 SoH 与 EoL 的联合预测。

**💡 创新点**

① 引入基于温度场的熵特征，将热非均匀性量化为物理可解释的指标；② 将静态与动态液态神经网络组合，支持从早期数据到寿命终点的连续轨迹预测；③ 在极低数据量下实现高精度、轻量化模型。

**🔧 技术方法**

熵特征提取、温度场数值模拟与在线自适应、基于信息熵的特征转换、轻量级 1D‑CNN 与注意力融合的 SoH 估计、连续时间液态神经网络（LNN）以及梯度在线微调。

**📊 数据集**

MIT–Stanford 18650 LFP 电池退化公开数据集，包含 124 颗电池、温度、电压和容量信息。

**📈 对比分析**

与 BFRN、BTL、DRRN、TLPH、DCNN 等现有方法对比，EntroLnn 在 MAE SoH 0.00458、EoL 18 周期、数据存储仅 239 KB、模型参数 0.25 M，显著优于基线且计算更轻量。

**⚠️ 局限性**

仍受限于温度传感器的精度与噪声，熵特征主要依赖温度场，对不同电池化学体系的通用性尚待进一步验证，且在极端工况下的自适应性能需进一步测试。

---

## 108. N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs

**arXiv ID:** 2601.06603 | [PDF](https://arxiv.org/pdf/2601.06603v1)

**作者:** Mohamed Sharafath `[一作]` (Comcast India Engineering Center), Aravindakumar Venugopalan `[通讯]` (Comcast)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了零训练的多跳混合表格-文本问答框架 N2N‑GQA，将检索结果转为动态证据图并进行图结构化检索增强生成；

**💡 创新点**

创新点在于提出“噪声到叙事”(noise‑to‑narrative)的动态构图与桥接文档选择策略，首次在开放域混合 QA 中展示图结构在多跳推理中的关键作用；

**🔧 技术方法**

使用的技术包括 LLM 驱动的查询分解、ColBERTv2 检索、基于 TF‑IDF 的边权图构建、GraphRank（语义+结构加权）重新排序、桥接式混合选择以及 LLM 生成答案；

**📊 数据集**

实验使用公开的 HybridQA 与 OTT‑QA 两个混合表格‑文本 QA 数据集；

**📈 对比分析**

在 OTT‑QA 上与 Vanilla RAG、RAG+查询分解、N2N‑GQA（无 GraphRank）等基线对比，使用 GPT‑4o/4.1/Llama3‑70B 等读者，完整模型 EM 达 48.80、F1 57.26，近似 Fine‑Tuned COS（56.9 EM）但无需训练，显著优于列表检索基线（提升约 19.9 EM）；

**⚠️ 局限性**

主要限制包括高昂的 LLM 计算成本、图结构过于简单（仅 TF‑IDF 边权）、GraphRank 改进有限、对检索质量高度依赖、缺乏更丰富的图关系类型，且目前仅验证于 QA 任务。

---

## 109. Beyond Reproducibility: Token Probabilities Expose Large Language Model Nondeterminism

**arXiv ID:** 2601.06118 | [PDF](https://arxiv.org/pdf/2601.06118v1)

**作者:** Tairan Fu `[一作]` (Politecnico di Milano), Shanshan Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2140 | [OpenAlex ID](https://openalex.org/A5100417119)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了大型语言模型在GPU推理时的非确定性对token概率的影响，并提出了基于标准差和极差的两项度量方法，在多种模型、GPU和批量大小下进行实验。

**💡 创新点**

创新点包括：①在token概率层面量化非确定性，而非仅关注输出文本；②发现概率在0.2–0.8区间最为敏感；③证明不同模型与GPU之间差异不大；④提供单次推理即可估算非确定性的方法。

**🔧 技术方法**

采用统计学指标（标准差、极差）对token概率进行多次（N=50）推理；使用softmax、BF16浮点格式；在NVIDIA A100/A6000/H200和华为Ascend‑910等GPU上并行执行实验。

**📊 数据集**

使用MMLU（Massive Multitask Language Understanding）数据集作为评估与并发提示的问答样本，并对Gemma3系列不同规模模型进行比较。

**📈 对比分析**

通过比较不同模型、GPU和批量大小下的标准差与极差来评估非确定性；结果显示几乎相同的幅度，批量越大波动越大；实验未给出具体延迟指标，但强调批量大小在吞吐量与非确定性之间的折衷。

**⚠️ 局限性**

局限性：仅在单GPU、BF16精度、温度为1、英文prompt、有限模型和批量大小范围内实验；未考虑多GPU、FP16/FP32、不同温度、长/多样化提示、其他语言、更多模型以及更大样本量的重复实验。

---

## 110. Agentic AI Microservice Framework for Deepfake and Document Fraud Detection in KYC Pipelines

**arXiv ID:** 2601.06241 | [PDF](https://arxiv.org/pdf/2601.06241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 111. Is Sanskrit the most token-efficient language? A quantitative study using GPT, Gemini, and SentencePiece

**arXiv ID:** 2601.06142 | [PDF](https://arxiv.org/pdf/2601.06142v1)

**作者:** Anshul Kumar `[一作]` `[通讯]` (Birla Institute of Technology and Science Pilani), Anshul Kumar (Birla Institute of Technology and Science Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过比较主流Tokenizer（GPT legacy、GPT o200k_base、GPT cl100k_base、Gemini）与SentencePiece，对梵语、英语和印地语的token化效率进行量化评估，揭示梵语在无偏baseline下具有约50%更高的信息密度，并指出现有Tokenizer对梵语的偏差会造成“token税”。

**💡 创新点**

①首次用大规模实验量化梵语的token稠密度和扩展因子；②发现现有Tokenizer对梵语的token化效果差异显著；③提出未来可基于Panini语法的语法感知Tokenizer和梵语预训练LLM的方向。

**🔧 技术方法**

使用SentencePiece BPE训练、OpenAI GPT legacy、GPT o200k_base、GPT cl100k_base、Google Gemini tokenizer；计算token计数、字符/Token（CpT）和Token/字符（TpC）三项指标；并通过两大实验（平行翻译与语义扩展）进行比较。

**📊 数据集**

使用701句《薄伽梵歌》的三语平行文本（梵语原文、拉丁转写、英语翻译、印地语翻译）以及对应的释义段落。

**📈 对比分析**

通过在相同文本上统计四种Tokenizer的token数、CpT和TpC，并对比实验1（语言效率）和实验2（语义扩展因子）。结果表明：①在SentencePiece baseline下，梵语的token数约为英语的一半；②GPT legacy将梵语token数推高至英语的2.5倍；③在语义扩展实验中，GPT旧版对梵语的扩展因子人为降低，导致扩展因子偏低；③总体表现显示梵语在无偏Tokenizer下最为信息稠密。

**⚠️ 局限性**

①仅评估Tokenizer行为，未涉及模型推理性能；②数据集局限于《薄伽梵歌》，可能不代表完整梵语；③英语和印地语翻译来源不同，存在细微差异；④SentencePiece仅用固定8k词表，未探讨词表大小对结果的影响。

---

## 112. COVR:Collaborative Optimization of VLMs and RL Agent for Visual-Based Control

**arXiv ID:** 2601.06122 | [PDF](https://arxiv.org/pdf/2601.06122v1)

**作者:** Canming Xia `[一作]` (Sun Yat-sen University), Luntong Li `[通讯]` (Peng Cheng Laboratory)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5077743639)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种协同优化框架，使视觉语言模型（VLM）与视觉强化学习（RL）相互提升：先利用RL产生的交互轨迹对VLM进行微调，再用微调后的VLM通过行动先验指导RL策略学习；

**💡 创新点**

创新点在于：①通过EDDF模块动态挑选高质量轨迹样本；②通过RALW模块根据轨迹回报自适应加权损失，缓解行动不一致；③采用渐进式微调与LoRA技术降低资源消耗；

**🔧 技术方法**

技术主要包括SAC强化学习、VLM（Qwen2.5‑VL‑3B）、EDDF、RALW、LoRA微调以及返回归一化和Z‑score标准化等；

**📊 数据集**

使用CARLA（高速公路与鬼步行者两个场景）和DMControl四个典型与难点控制任务的数据集；

**📈 对比分析**

与SAC、DeepMDP、CURL、DrQ、ResAct等SOTA方法以及多种VLM辅助方法对比，本文在CARLA的奖励和行驶距离、DMControl的多任务奖励上均取得领先或同等水平的最高分，显著提升了学习效率与稳定性；

**⚠️ 局限性**

局限性包括：对VLM的依赖仍显高，需先有预训练模型；在极端长序列规划或高动态环境下的适应性仍待验证；微调过程仍需多轮训练，计算成本不可忽略。

---

## 113. SafeGPT: Preventing Data Leakage and Unethical Outputs in Enterprise LLM Use

**arXiv ID:** 2601.06366 | [PDF](https://arxiv.org/pdf/2601.06366v1)

**作者:** Pratyush Desai `[一作]` (Binghamton University), Zhaohan Xi `[通讯]` (Binghamton University)

**通讯引用:** 111 | [OpenAlex ID](https://openalex.org/A5026309535)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SafeGPT 两侧防护系统，防止企业 LLM 在输入输出过程中泄露敏感数据和产生违规内容。

**💡 创新点**

首次将输入侧预过滤、输出侧审核与人类反馈结合成双向防护架构，并采用分级政策与知识图谱实现精准自适应。

**🔧 技术方法**

结合正则模式匹配、上下文命名实体识别、知识图谱相似度搜索、语义相似度与事实一致性检测，以及自动重写与人类复核。

**📊 数据集**

使用三组合成数据集：PIIBench（PII 泄露）、ToxicChat（政策违规）和 EnterpriseScenarios（行业特定案例）。

**📈 对比分析**

与 Regex‑Only、NER、Keyword、Hybrid 四种基线相比，SafeGPT 在 PIIBench 上召回率 70%，在 ToxicChat 上 100% 识别率，在 EnterpriseScenarios 上 68% 召回、40.5% 精度，且漏检率 0% 但误报率 78.6%。

**⚠️ 局限性**

主要限制在于评估基于合成数据、保守策略导致高误报、知识图谱依赖组织维护且对真实多样化工作流与长期部署（延迟、适应性、对抗性）尚未充分验证。

---

## 114. Attention in Geometry: Scalable Spatial Modeling via Adaptive Density Fields and FAISS-Accelerated Kernels

**arXiv ID:** 2601.06135 | [PDF](https://arxiv.org/pdf/2601.06135v1)

**作者:** Zhaowen Fan `[一作]` `[通讯]`, Zhaowen Fan

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出Adaptive Density Fields (ADF)，一种基于几何注意力的空间聚合框架，将局部邻域的影响通过可查询的高斯混合模型进行连续空间求和。

**💡 创新点**

创新之处在于把空间影响建模为物理距离诱导的注意力操作，使近似k‑NN成为定义本身的一部分，既兼顾可扩展性又保持几何可解释性，打通了适应性核方法与注意力机制的鸿沟。

**🔧 技术方法**

主要技术包括：将点坐标转换为ECEF Cartesian形式，使用FAISS倒排文件索引实现近似k‑NN，按分数调制高斯核宽度，并在查询时聚合加权核贡献。

**📊 数据集**

实验基于中国全国机动航迹数据，构造了约180万条POI，随后在成都市双流机场单日航迹上演示了ADF与轨迹条件的影响区(ZOI)提取。

**📈 对比分析**

相较于传统全局核密度估计和传统注意力，ADF通过FAISS实现了查询子线性时间、O(n)存储的可扩展性，并在保留影响强度的前提下将近似误差控制在可忽略范围内，实验结果显示在百万级点集上仍保持低延迟和良好准确度。

**⚠️ 局限性**

局限性包括：近似k‑NN导致的误差、固定的等方差高斯核假设、对分数到核宽度映射的手工设定、缺乏自适应参数学习、以及只在航空数据上验证，尚未在实时部署或多模态语义融合场景中测试。

---

## 115. Monkey Jump : MoE-Style PEFT for Efficient Multi-Task Learning

**arXiv ID:** 2601.06356 | [PDF](https://arxiv.org/pdf/2601.06356v1)

**作者:** Nusrat Jahan Prottasha `[一作]` (University of Central Florida), Ozlem Garibay `[通讯]` (University of Central Florida)

**通讯引用:** 720 | [OpenAlex ID](https://openalex.org/A5062903281)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Monkey Jump 方法，在参数高效微调（PEFT）中通过不增加额外可训练参数的方式实现混合专家（MoE）式的输入自适应。

**💡 创新点**

创新点在于将 Transformer 块中已存在的每个投影的 PEFT 适配器视作隐式专家，利用无梯度的 K‑means 路由与指数移动平均中心进行令牌级别的专家分配，从而获得 MoE 级别的特殊化而不额外增加参数或训练开销。

**🔧 技术方法**

核心技术包括：梯度无关的聚类路由（K‑means + EMA 更新）、基于余弦相似度的 top‑k 令牌路由、对投影层的共享与专属路由策略，以及理论证明 token‑wise 路由提升表达能力、last‑token 路由在因果 Transformer 中信息最优。

**📊 数据集**

在 47 个多模态基准（14 文本、14 图像、19 视频）上评估，使用 Llama‑3‑8B‑Instruct、LLaVA‑OneVision‑Qwen2‑7B 以及 Qwen2‑0.5B 等模型。

**📈 对比分析**

与标准 PEFT（LoRA、AdaLoRA、Propulsion 等）和 MoE‑PEFT（MoELoRA、HydraLoRA、MoLA 等）对比，Monkey Jump 在保持相近或更好准确率的同时，训练时可用参数 7–29 倍更少，显存占用降低多达 48%，训练速度提升 1.5–2 倍，推理吞吐率也提升 10–25%。

**⚠️ 局限性**

主要限制包括：专家容量固定为投影数（约 7 个），无法像传统 MoE 那样扩展到数百专家；聚类假设可能不适用于高度异构或复杂数据；需要额外的 K‑means 初始化步骤；以及对 top‑k、EMA 动量等超参的敏感度。

---

## 116. Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning

**arXiv ID:** 2601.06282 | [PDF](https://arxiv.org/pdf/2601.06282v1)

**作者:** Yue Zhou `[一作]` (University of Illinois Chicago), Srinivasan H. Sengamedu `[通讯]` (Amazon)

**通讯引用:** 492 | [OpenAlex ID](https://openalex.org/A5038858575)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了Amory工作记忆框架，用于长时序对话中主动构建、巩固和检索记忆。

**💡 创新点**

创新点在于将对话片段组织为叙事、依据对话节奏进行记忆巩固、将边缘事实语义化，并用一致性推理代替传统嵌入检索。

**🔧 技术方法**

技术包括LLM驱动的记忆绑定、巩固与语义化、Neo4j图数据库存储语义记忆，以及基于叙事一致性推理的检索。

**📊 数据集**

使用LOCOMO对话推理基准和基于AgentIF合成的长任务对话进行评测。

**📈 对比分析**

通过与Mem0、Zep、A-MEM、ReadAgent、HippoRAG及全上下文基准的J-score和延迟对比，Amory在多跳、时序、常识等任务上提升最高27.8%，并将响应时间降低约50%。

**⚠️ 局限性**

局限在于依赖文本合成数据、未探索神经表征，且在多模态或多主体场景中的鲁棒性尚未验证。

---

## 117. LLM Flow Processes for Text-Conditioned Regression

**arXiv ID:** 2601.06147 | [PDF](https://arxiv.org/pdf/2601.06147v1)

**作者:** Felix Biggs `[一作]` (Secondmind AI), Samuel Willis `[通讯]` (Secondmind AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将神经扩散过程（NDP）与大型语言模型过程（LLMP）结合的产品专家采样框架，用以生成受文本条件约束的回归预测。

**💡 创新点**

创新点在于：1）将产品专家思想引入扩散/流模型与LLMP的结合；2）设计了可解析的卷积平滑专家分布，从而实现高效的非自回归采样；3）实现了文本信息对回归轨迹的平滑且准确的约束。

**🔧 技术方法**

使用了流匹配（flow matching）与扩散模型的速度场学习；LLMP的分箱概率估计；产品专家采样公式与高斯卷积；以及Transformer架构的等价/不变性设计。

**📊 数据集**

主要在合成数据集（Gaussian Process 样本与混合核）上进行训练与评估；使用小型 Qwen‑3‑4B‑Instruct‑2507 LLM 作为 LLMP 推理器。

**📈 对比分析**

与纯 NDP、I‑LLMP（独立分箱）和 A‑LLMP（自回归）等基线对比，实验显示 LLM‑NDP 在条件样本的分位数与轨迹质量上更为平滑、误差累积更小，且在文本约束实验中能显著提升预测一致性。

**⚠️ 局限性**

局限包括：1）依赖合成数据，未在真实回归数据上验证；2）LLM 推理成本高且未进行微调；3）产品专家导致在高密度区域过度集中，可能出现过度自信；4）未探索更大规模或更复杂的结构化数据。

---

## 118. Quantification and Classification of Carbon Nanotubes in Electron Micrographs using Vision Foundation Models

**arXiv ID:** 2601.06673 | [PDF](https://arxiv.org/pdf/2601.06673v1)

**作者:** Sanjay Pradeep `[一作]`, Candace S. J. Tsai `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一套利用视觉基础模型自动化量化和分类碳纳米管的框架，集成了SAM进行交互式分割与DINOv2提取粒子特征；

**💡 创新点**

创新点在于将零样本分割模型SAM与自监督视觉Transformer DINOv2结合，通过分割掩膜限定特征提取，从而在仅使用原始训练集34%数据的情况下实现95.5%分类准确率，显著优于传统CNN方法；

**🔧 技术方法**

采用的技术包括：Segment Anything Model (SAM) 的交互式分割、DINOv2 ViT-B/14 的多尺度特征提取、掩膜引导的特征采样、平均/最大/两者拼接的特征池化、线性与浅层MLP分类头；

**📊 数据集**

数据集为1800张TEM图像（400张各类碳纳米管形态，四类：纤维、聚集、矩阵、矩阵表面），并使用200张混合SEM/TEM图像验证分割性能；

**📈 对比分析**

与以往基线CNN/YOLO/UNet等方法对比，24种模型配置中最佳方案（DINOv2+掩膜+avg+max+MLP）在测试集上实现95.5%准确率，平均准确率91.4%，大幅提升；

**⚠️ 局限性**

局限性包括：对多粒子重叠场景仍可能受特征重叠影响、分割模型需人工点击启动、对极低对比度或特殊材料的适用性未充分验证，且未对更复杂多模态数据进行集成。

---

## 119. The Potential of Erroneous Outbound Traffic Analysis to Unveil Silent Internal Anomalies

**arXiv ID:** 2601.06280 | [PDF](https://arxiv.org/pdf/2601.06280v1)

**作者:** Andrea Sordello `[一作]` (Politecnico di Torino), Marco Mellia `[通讯]` (Politecnico di Torino)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在校园网络中实时监控并记录错误的出站流量，识别并分析了多种内部网络异常，包括恶意活动、配置错误、过时部署和异常 DNS 请求。

**💡 创新点**

创新点在于将注意力集中于“错误出站流量”这一被忽视的子集，利用 SDN 动态过滤仅记录错误包，实现低噪声、高信噪比的数据采集，并首次展示该方法可揭露大量潜伏的内部安全问题。

**🔧 技术方法**

主要技术手段包括：基于状态化的 SDN 网络监控器、超时与包检验的错误检测、动态 SDN 流表下发、统计宏观分析、案例驱动的异常归因，以及后续计划的表示学习、聚类、模式挖掘和 LLM 辅助解释。

**📊 数据集**

使用的数据集为 2024‑12‑01 至 2025‑02‑28 的校园子网出站流量，约 3 个月内每小时 300k 条错误包，占总出站流量的 0.06%。

**📈 对比分析**

文章并未与传统入站流量监测方法做量化对比，但通过案例展示了 11% 的异常主机产生了 11% 的错误流量，说明错误流量能高效揭示多类异常；未来工作计划通过无监督学习进一步验证方法性能。

**⚠️ 局限性**

局限性包括：缺乏完整的 ground‑truth 评估，异常检测仍需人工验证；只针对校园网络，结果的普适性未知；对高频、低错误率场景的鲁棒性尚待测试。

---

## 120. EVM-QuestBench: An Execution-Grounded Benchmark for Natural-Language Transaction Code Generation

**arXiv ID:** 2601.06565 | [PDF](https://arxiv.org/pdf/2601.06565v1)

**作者:** Pei Yang `[一作]` (Gradient), Tianyu Shi `[通讯]` (Gradient)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了基于执行的EVM交易脚本生成基准EVM-QuestBench，提供原子与复合两类任务并支持动态参数化与验证器评分。

**💡 创新点**

采用模块化任务定义、快照隔离的forked链执行、验证器驱动的得分机制以及基于步骤效率的复合任务评分，显著降低任务开发成本并揭示单步与多步能力差距。

**🔧 技术方法**

使用大型语言模型生成TypeScript交易脚本，在BSC主网fork环境中执行并收集交易回执，由自定义验证器进行状态检查，计算加权分数；复合任务通过规划与多轮交互实现。

**📊 数据集**

107个任务（62个原子、45个复合），涵盖钱包、ERC20、DeFi、流动性、闪电贷等多种操作，数据来源于BSC主网合约与链状态，动态采样数值。

**📈 对比分析**

对20个主流LLM进行统一评测，报告原子得分、复合得分与总分，发现模型在单步精度与多步流程完成上存在显著分化；DeepSeek V3.2在复合任务上表现突出，Gemini-3-Pro在原子任务表现优异。

**⚠️ 局限性**

受限于RPC稳定性与fork状态漂移，得分可能波动；复合任务得分受步骤效率与重试计数混合影响；单轮评测缺乏统计置信区间，执行与温度随机性可能导致结果不一致。

---

## 121. WHU-PCPR: A cross-platform heterogeneous point cloud dataset for place recognition in complex urban scenes

**arXiv ID:** 2601.06442 | [PDF](https://arxiv.org/pdf/2601.06442v1)

**作者:** Xianghong Zou `[一作]` (Nanchang University), Zhen Dong `[通讯]` (Wuhan University)

**通讯引用:** 25086 | [OpenAlex ID](https://openalex.org/A5100429975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了跨平台异构点云数据集 WHU-PCPR 并基准评估多种点云定位方法。

**💡 创新点**

创新点在于提供了多平台、多传感器、多阶段的异构点云数据集，覆盖复杂场景与长时序变化，推动点云定位研究。

**🔧 技术方法**

采用点云定位检索技术（如 PointNetVLAD、PPT‑Net、MinkLoc3D、LoGG3D‑Net）与重排序方法（AlphaQE、SGV、RPR）进行实验。

**📊 数据集**

使用了 WHU-PCPR 数据集（以及 Oxford RobotCar 作为对比）进行实验。

**📈 对比分析**

比较显示 LoGG3D‑Net+SGV 在 WHU-PCPR 上取得最佳 Recall/Precision，但总体仍受域差异和视角变化影响，性能仍有提升空间。

**⚠️ 局限性**

主要限制是现有方法在跨场景、跨设备和视角变化上泛化不足，容易过拟合，重排序提升有限。

---

## 122. Brokerage in the Black Box: Swing States, Strategic Ambiguity, and the Global Politics of AI Governance

**arXiv ID:** 2601.06412 | [PDF](https://arxiv.org/pdf/2601.06412v1)

**作者:** Ha-Chi Tran `[一作]` `[通讯]` (London School of Economics and Political Science), Ha-Chi Tran (London School of Economics and Political Science)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文以技术摆动国为视角，对美国–中国技术竞争中南韩、新加坡、印度三国如何利用AI技术不透明性与制度透明性形成治理影响进行了案例比较分析。

**💡 创新点**

创新点在于将AI不透明性视为战略资源，提出三种摆动国策略（延迟与对冲、选择性对齐、规范性中介）并将其与网络中心度关联。

**🔧 技术方法**

主要采用案例研究、网络中心性理论和制度分析方法，无需机器学习算法。

**📊 数据集**

数据来源为公开文献、官方政策文件、行业报告及三国AI治理案例资料。

**📈 对比分析**

通过对比三国的案例进行结构性比较，未涉及量化性能指标；分析强调策略逻辑与治理结果的关联。

**⚠️ 局限性**

局限在于仅选取三国，忽视欧盟等超国家主体以及跨国公司和社会团体的作用，且缺乏跨时间的纵向追踪。

---

## 123. Mining Quantum Software Patterns in Open-Source Projects

**arXiv ID:** 2601.06281 | [PDF](https://arxiv.org/pdf/2601.06281v1)

**作者:** Neilson Carlos Leite Ramalho `[一作]` (University of São Paulo), Marcos Lordello Chaim `[通讯]` (University of São Paulo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对80个Python量子计算开源项目中的985个Jupyter Notebook进行语义搜索，构建了61种量子计算模式的知识库，并在此基础上提出了9个新的量子模式。

**💡 创新点**

①在已有59种模式的基础上扩充为61种，并首次定义9个新模式；②开发可复用的语义匹配工具，自动识别代码中的模式；③系统评估模式在实践中的使用层级，揭示三层抽象层级的演进。

**🔧 技术方法**

Python AST解析、句子Transformer嵌入、余弦相似度匹配、GitHub API爬取、手工模式归类。

**📊 数据集**

80个Python量子项目共985个Notebook（来自GitHub）以及三大主流框架（Qiskit、PennyLane、Classiq）的源代码。

**📈 对比分析**

通过模式匹配次数、匹配类型（名称/摘要）统计，比较框架与Notebook中的模式覆盖率；结果显示22/24框架模式出现于Notebook，名称匹配平均相似度0.99，摘要匹配平均相似度0.70，验证工具有效且能捕获高频模式。

**⚠️ 局限性**

仅覆盖Python语言和Notebook，手工模式归类可能引入偏差；阈值设置较高可能漏检罕见模式；未考虑C++/Q#/Julia等其他量子编程语言。

---

## 124. Analyzing the Structure of Handwritten Digits: A Comparative Study of PCA, Factor Analysis, and UMAP

**arXiv ID:** 2601.06168 | [PDF](https://arxiv.org/pdf/2601.06168v1)

**作者:** Jyotiraditya Gupta `[一作]` `[通讯]` (University of Toronto), Jyotiraditya Gupta (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用PCA、FA和UMAP三种降维方法探究MNIST手写数字在高维像素空间中的潜在低维结构，研究其内在维度、共享变异及非线性几何特征。

**💡 创新点**

首次将线性降维（PCA、FA）与非线性降维（UMAP）结合，揭示手写数字既有可解释的笔画原语，又在低维流形上呈现平滑的非线性转化，提供多视角的结构洞察。

**🔧 技术方法**

主成分分析（PCA）、因子分析（FA）以及UMAP（基于umap-learn），并使用scikit-learn实现特征提取与重构。

**📊 数据集**

MNIST手写数字数据集，包含5000张28×28灰度图像（各数字类别均等）。

**📈 对比分析**

对比方法通过累计解释方差、重构误差和二维嵌入可视化来评估：PCA在10个主成分下解释约50%方差，50个主成分可恢复几乎完整图像；FA的旋转因子揭示可解释的笔画原语，但二维投影仍重叠；UMAP在n_neighbors=15时能较好地区分数字类别，呈现连续的非线性流形。

**⚠️ 局限性**

PCA和FA为线性方法，难以捕捉非线性转变；FA因子需要旋转且不唯一；UMAP对超参数和随机初始化敏感，嵌入无概率解释，且未对下游分类或聚类性能进行评估。

---

## 125. SkyNomad: On Using Multi-Region Spot Instances to Minimize AI Batch Job Cost

**arXiv ID:** 2601.06520 | [PDF](https://arxiv.org/pdf/2601.06520v1)

**作者:** Zhifei Li `[一作]` (University of California Berkeley), Ion Stoica `[通讯]` (University of California Berkeley)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个多区域基于 Spot 实例的 AI 批处理作业调度系统，既保证了截止期限，又实现了显著的成本降低。

**💡 创新点**

创新点在于同时利用空间和时间异质性的 Spot 可用性与寿命，结合实时探测、寿命预测和统一成本模型，实现了跨区域的智能迁移与决策。

**🔧 技术方法**

使用了轻量级探测、虚拟实例寿命预测（生存分析）、预估 Spot 寿命、迁移成本折算以及阈值决策的调度算法。

**📊 数据集**

使用真实云环境（AWS / GCP）上的 GPU 训练任务（Qwen3 模型 + Orca‑Math 数据集）以及公开的 Spot 可用性轨迹（V100、H100）进行实验。

**📈 对比分析**

与单区域 Uniform Progress、AWS SageMaker Managed Spot、UP(S) 等基线比较；在真实部署中平均节省 55% 成本，模拟中与最优策略相差 <10%，成本比单区高 1.25–3.96×。

**⚠️ 局限性**

局限性包括只能在单一云内多区操作，迁移成本与数据主权约束，极大检查点尺寸或跨云迁移时仍会产生高费用，且未考虑弹性缩放和实时性能服务。

---

## 126. Average shortest-path length in word-adjacency networks: Chinese versus English

**arXiv ID:** 2601.06361 | [PDF](https://arxiv.org/pdf/2601.06361v1)

**作者:** Jakub Dec `[一作]` (Cracow University of Technology), Tomasz Stanisz `[通讯]` (Institute of Nuclear Physics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了中文和英文文学作品的词-标点共现网络结构，重点考察平均最短路径长度（ASPL）随网络规模的变化，并将不同历史时期、不同地区、翻译文本的网络进行比较。

**💡 创新点**

创新点在于将标点符号视为与词同等的词素加入网络，发现标点对网络连通性和ASPL具有显著影响；同时通过构建加速生长+非线性优先附着模型对ASPL进行理论拟合，验证中文与英文网络在包含标点时的相似性和在排除标点时的差异。

**🔧 技术方法**

使用网络科学方法（词-标点共现网络构建、度分布、ASPL计算）、Heaps、Zipf-Mandelbrot律的理论模型、Sigmoid插值法与随机图近似公式相结合进行分析；实现上采用Jieba分词、Breadth-First Search 算法计算ASPL。

**📊 数据集**

数据集包括94部中文小说（覆盖清末、民国、毛泽东时代、当代；包含港台和网络小说）以及10部中英互译文本（三部中文原著翻译成英文，三部英文原著翻译成中文），共计约10⁴–10⁶节点。

**📈 对比分析**

通过将不同文本的L(N)曲线与模型预测（L_fit(N)）以及各文本平均ASPL进行比较，发现包含标点时各文本的ASPL曲线高度一致，且与模型拟合误差≤0.05；排除标点后ASPL显著上升，中文文本的上升幅度比英文更大，说明中文对标点的依赖更强。

**⚠️ 局限性**

局限性包括：仅考虑了词级别的邻接关系，未对短语、句法依存等更高级别关系进行建模；标点集与语言本身的标点使用习惯差异导致可比性受限；翻译样本数量有限，可能不具备统计代表性；ASPL只是网络的单一指标，缺乏对其他拓扑特征（如聚类、模块化）的深入探讨。

---

## 127. Does Inference Scaling Improve Reasoning Faithfulness? A Multi-Model Analysis of Self-Consistency Tradeoffs

**arXiv ID:** 2601.06423 | [PDF](https://arxiv.org/pdf/2601.06423v1)

**作者:** Deep Mehta `[一作]` `[通讯]` (Adobe Inc), Deep Mehta (Adobe Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了推理规模化（自一致性多路径推理）对大型语言模型在数学推理任务中的准确率与推理可信度的影响。

**💡 创新点**

首次系统评估推理规模化与推理可信度之间的关系，并揭示不同模型表现高度依赖于自身训练与架构。

**🔧 技术方法**

采用自一致性、多路径采样、早期回答可信度探测、Bootstrap 置信区间、McNemar 统计检验和 Cohen‑d 效应大小等技术。

**📊 数据集**

在 GSM8K 100 题目子集（小学数学推理）上进行实验。

**📈 对比分析**

对 GPT‑5.2、Claude Opus 4.5、Gemini‑3‑flash 和 DeepSeek‑v3.2 在 N=1、5、20 三种采样量下比较准确率与可信度，发现 GPT‑5.2 在 N=5 时准确率提升12%但可信度略降；Claude Opus 4.5 准确率下降但可信度大幅提升；DeepSeek‑v3.2 近满分时可信度略上升；Gemini‑3‑flash 变化不显著。

**⚠️ 局限性**

仅使用单一可信度探测器、仅评估数学推理、样本量有限、API 版本控制与缓存受限、未覆盖其他领域与模型。

---

## 128. CEEMDAN-Based Multiscale CNN for Wind Turbine Gearbox Fault Detection

**arXiv ID:** 2601.06217 | [PDF](https://arxiv.org/pdf/2601.06217v1)

**作者:** Nejad Alagha `[一作]` (University of Dubai), Abigail Copiaco `[通讯]` (University of Dubai)

**通讯引用:** 312 | [OpenAlex ID](https://openalex.org/A5053459575)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对风力机齿轮箱故障，本文提出了一种结合 CEEMDAN 分解与多尺度 CNN 的混合诊断框架。

**💡 创新点**

创新点在于将 CEEMDAN 提取的多尺度本征模态函数分别送入并行 CNN 分支，实现高频瞬态与低频结构特征的协同学习，并显著提升识别速度。

**🔧 技术方法**

采用 CEEMDAN 进行信号分解，1D 多尺度并行 CNN 进行特征提取与分类，使用 Adam 优化器和 dropout 进行正则化。

**📊 数据集**

使用 NREL Gearbox Reliability Collaborative 数据集，该数据集包含七个加速度计采集的健康与损坏齿轮箱 1 分钟振动记录，共 1400 个 0.5 秒窗口。

**📈 对比分析**

与 MSCNN、WPD‑MSCNN、CEEMDAN‑BT‑CNN、TWSVM 等方法比较，本文方法在 F1 分数上达 98.95%，训练时长仅 2.26 秒/epoch，速度优于 MSCNN 与 CEEMDAN‑BT‑CNN，略低于 WPD‑MSCNN，但在准确率和计算效率上实现了良好平衡。

**⚠️ 局限性**

局限性包括仅针对二分类（健康/故障）设计，缺乏多类别故障区分，且仅在受控实验数据上验证，尚需在真实运行环境中进一步评估。

---

## 129. ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions

**arXiv ID:** 2601.06112 | [PDF](https://arxiv.org/pdf/2601.06112v1)

**作者:** Aayush Gupta `[一作]` `[通讯]`, Aayush Gupta

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ReliabilityBench 基准，系统评估 LLM 代理在一致性、鲁棒性和容错性三维条件下的可靠性。

**💡 创新点**

创新点包括三维可靠性表面 R(k,ε,λ)、动作元模型变形（Action Metamorphic Relations）以及基于混沌工程的故障注入框架。

**🔧 技术方法**

采用 pass^k 一致性指标、语义变形策略、state‑based 验证、混沌工程故障注入、ReAct 与 Reflexion 两种代理架构，以及 Gemini 2.0 Flash 与 GPT‑4o 两个 LLM 模型。

**📊 数据集**

在四个真实域（排程、旅行、客服、电商）中生成约 25+ 工具集，构造约 1,280 个任务实例，涵盖不同复杂度、扰动级别和故障注入配置。

**📈 对比分析**

对 Gemini 2.0 Flash 与 GPT‑4o 在 k=2、ε∈{0,0.1,0.2}、λ∈{0,0.2} 的完整网格进行对比；结果显示扰动导致 8.8% 可靠性下降，ReAct 在一致性与鲁棒性上优于 Reflexion，GPT‑4o 成本高 82 倍但可靠性相近。

**⚠️ 局限性**

局限性包括实验规模有限（仅 1,280 次 episode）、扰动深度受限（最高 ε=0.2）、模型覆盖有限、任务为合成域、k=2 的一致性评估可能低估高 k 下的波动。

---

## 130. The AI Pyramid A Conceptual Framework for Workforce Capability in the Age of AI

**arXiv ID:** 2601.06500 | [PDF](https://arxiv.org/pdf/2601.06500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 131. Context Matters: Peer-Aware Student Behavioral Engagement Measurement via VLM Action Parsing and LLM Sequence Classification

**arXiv ID:** 2601.06394 | [PDF](https://arxiv.org/pdf/2601.06394v1)

**作者:** Ahmed Abdelkawy `[一作]` (University of Louisville), Michael McIntyre `[通讯]` (University of Louisville)

**通讯引用:** 1265 | [OpenAlex ID](https://openalex.org/A5061899053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出三阶段框架：先用少样本视觉‑语言模型(ViFi‑CLIP)识别学生动作；再将2分钟视频划分成非重叠片段并生成动作序列；最后用大型语言模型(LLM)结合同伴动作上下文判定学生的参与度。

**💡 创新点**

首次将VLM的少样本学习与LLM的零样本推理结合，利用课堂上下文提升行为参与度识别；同时自建双部分数据集以解决公开数据缺乏问题。

**🔧 技术方法**

技术组合包括：ViFi‑CLIP（少样本微调）、滑动窗口分段、Gemini 2.5/LLM提示式序列分割与分类；使用CLIP的视觉与文本编码器进行动作匹配。

**📊 数据集**

数据集：①13类受控动作剪辑集（208/47样本）；②三节课共11名学生的未剪辑2分钟视频（约90条片段），每条含手工标注的动作序列与“参与/不参与”标签。

**📈 对比分析**

与基准方法比较：ViFi‑CLIP在few‑shot下最高达97.9% top‑1；LLM在手工动作序列下F1=92%；使用Gemini‑based或VLM‑based动作分割后仍可达85–86% F1，明显优于传统直觉或仅基于动作直方图的方式。

**⚠️ 局限性**

局限性包括：动作识别误差仍影响最终参与度判定；同类细粒度动作（如写笔记 vs. 画涂鸦）区分不足；数据集规模和多样性有限，难以覆盖更广泛的课堂场景与文化差异。

---

## 132. L2CU: Learning to Complement Unseen Users

**arXiv ID:** 2601.06119 | [PDF](https://arxiv.org/pdf/2601.06119v1)

**作者:** Dileepa Pitawela `[一作]` (University of Adelaide), Hsiang-Ting Chen `[通讯]` (University of Adelaide)

**通讯引用:** 2207 | [OpenAlex ID](https://openalex.org/A5036805602)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种学习补全（Learning to Complement, L2C）框架，用于在稀疏且噪声的多标注者数据下，补全未见用户的决策并提升人机协作分类性能。

**💡 创新点**

创新点在于：①通过聚类提取代表性标注者配置文件并进行噪声标签增强，②在推理时对未见用户进行配置文件匹配并根据基准模型与用户准确率进行入口判定，实现对未知用户的自适应补全。

**🔧 技术方法**

使用的技术包括：多标注者标签聚类（Fuzzy K‑Means）、CrowdLab 一致性标签估计、噪声标签增强、基于多层感知机的人工标签编码器与决策模块、支持向量机进行用户配置文件匹配以及基准模型与合作模型的联合训练。

**📊 数据集**

使用的数据集包括图像域的 CIFAR‑10、CIFAR‑10N、CIFAR‑10H、Fashion‑MNIST‑H、Chaoyang，以及文本域的 AgNews。

**📈 对比分析**

与现有 L2D、L2D‑Unseen、LECOMH、LECODU 等方法比较，L2CU 在所有数据集上均实现了更高的后置准确率（例如 CIFAR‑10N 99.9%）、正向改动率高、负向改动率低，表明其在未见用户上的泛化效果显著优于传统方法。

**⚠️ 局限性**

局限性包括：①对配置文件数 K 的选择敏感，过多导致稀疏训练样本不足；②在用户极其精准时，合作可能略有下降；③当前方法仍依赖于预估的一致性标签，若该估计不佳可能影响性能。

---

## 133. How to Build Robust, Scalable Models for GSV-Based Indicators in Neighborhood Research

**arXiv ID:** 2601.06443 | [PDF](https://arxiv.org/pdf/2601.06443v1)

**作者:** Xiaoya Tang `[一作]` (University of Utah), Tolga Tasdizen `[通讯]` (University of Utah)

**通讯引用:** 6518 | [OpenAlex ID](https://openalex.org/A5059125158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对谷歌街景（GSV）图像进行无监督后训练，并评估不同视觉基础模型（ViT、Vision Mamba、ResNet、Swin）在四个城市环境分类任务中的表现；

**💡 创新点**

首次系统比较了 ImageNet 预训练与 GSV 无监督后训练对 ViT 与 Vision Mamba 的影响，揭示后训练虽提升下游性能但对线性探测不一定有益；

**🔧 技术方法**

采用 DINO 视觉自监督蒸馏、数据增强、裁剪投票策略和注意力图分析等技术；

**📊 数据集**

使用约 100 万张未标记的 GSV 图像进行后训练，随后在四个公开数据集（Streetlight、NSH、Green30、Sidewalk）上进行下游评估；

**📈 对比分析**

通过 fine‑tune 与 linear probing 两种评估方式，对比各模型在准确率、平衡准确率与 F1 等指标；实验显示 Vim‑S 经过后训练后在所有任务中实现 85–90% 的平衡准确率，超过 ResNet 和 Swin；

**⚠️ 局限性**

局限包括：域迁移导致线性探测性能下降、较大 Vision Mamba 模型在 DINO 训练中易失稳、模型容量过小易饱和、未对 GSV 数据做严格过滤可能导致过度泛化；

---

## 134. MixDPO: Modeling Preference Strength for Pluralistic Alignment

**arXiv ID:** 2601.06180 | [PDF](https://arxiv.org/pdf/2601.06180v1)

**作者:** Saki Imai `[一作]` (Northeastern University), Malihe Alikhani `[通讯]` (Northeastern University)

**通讯引用:** 2028 | [OpenAlex ID](https://openalex.org/A5025559955)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种混合对数似然直接偏好优化方法（MixDPO），通过将偏好强度建模为可学习的分布来改进语言模型的偏好对齐。

**💡 创新点**

创新点在于将偏好强度视为随机变量并学习其分布，从而捕捉训练样本中存在的偏好表达强度异质性。

**🔧 技术方法**

采用了混合Logit模型与直接偏好优化（DPO）相结合的技术，实验中使用LogNormal和Gamma两种分布，进行蒙特卡洛采样或解析期望。

**📊 数据集**

在PRISM、Community Alignment和Anthropic HH三大偏好数据集上对Pythia‑2.8B和Llama3.2‑1B模型进行评估。

**📈 对比分析**

与传统DPO相比，MixDPO在聚合指标（win rate）上提升了约11.2个百分点，同时在子群体级别的偏好边际（macro平均）也有显著改善，未出现整体性能下降。

**⚠️ 局限性**

局限性包括未将观测到的评价者属性或输出特征加入模型，且评估主要依赖LLM评测器，缺乏完整的人类评估。

---

## 135. Data-Driven Reduced-Complexity Modeling of Fluid Flows: A Community Challenge

**arXiv ID:** 2601.06183 | [PDF](https://arxiv.org/pdf/2601.06183v1)

**作者:** Oliver T. Schmidt `[一作]` (University of California San Diego), Ricardo Vinuesa `[通讯]` (University of Michigan)

**通讯引用:** 10413 | [OpenAlex ID](https://openalex.org/A5049616413)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织并公开了一个针对湍流航空流的压缩、预测和感知任务的社区挑战，并提供数据、基线和评估工具。

**💡 创新点**

将多种传统与现代机器学习方法与标准化指标和公开数据集结合，形成可对比的开放挑战平台，鼓励负面结果与方法改进。

**🔧 技术方法**

主要使用POD、CNN‑AE、DMD、LSTM、LSE、Wiener滤波、MLP、CNN等传统与深度学习技术。

**📊 数据集**

采用DNS、LES、TR‑PIV等湍流边界层、腔流、翼型摆动、喷流等多样化数据集，涵盖不同Re、几何与测量条件。

**📈 对比分析**

通过压缩率、NMSE、RMSE等指标对比基线，CNN‑AE在压缩上优于POD，DMD在短期预测优于LSTM但随时间衰减，LSTM在更长时滞更稳健；感知任务中CNN显著优于线性方法，非线性MLP未超越Wiener滤波。

**⚠️ 局限性**

基线模型受限于线性假设或网络容量，预测窗口有限，训练数据量不足导致过拟合，缺乏对非线性与多尺度特性的充分建模。

---

## 136. Performance of models for monitoring sustainable development goals from remote sensing: A three-level meta-regression

**arXiv ID:** 2601.06178 | [PDF](https://arxiv.org/pdf/2601.06178v1)

**作者:** Jonas Klingwort `[一作]` (Statistics Netherlands), Joep Burger `[通讯]` (Statistics Netherlands)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文利用三层随机效应meta分析，对20篇遥感机器学习研究中86个实验的整体准确率进行汇总评估。

**💡 创新点**

创新点在于首次将双角度变换和多级随机效应模型应用于遥感ML性能的元分析，并识别多数类比例是解释异质性的主要因素。

**🔧 技术方法**

采用双角度变换、三层随机效应模型、meta回归和特征选择（AIC/BIC/RMSE）等统计技术。

**📊 数据集**

数据来源为从200篇随机抽样的文献中提取的86个实验，涉及不同卫星、传感器、模型和类别数的遥感分类结果。

**📈 对比分析**

结果显示平均整体准确率约为0.90（95%CI 0.86–0.92），但异质性高（64% 由研究间差异解释），且仅多数类比例显著影响性能。

**⚠️ 局限性**

局限包括仅报告整体准确率（对类别不平衡敏感）、样本量有限、缺乏子研究层特征、可能存在发表偏倚及对其他性能指标关注不足。

---

## 137. An Intelligent AI glasses System with Multi-Agent Architecture for Real-Time Voice Processing and Task Execution

**arXiv ID:** 2601.06235 | [PDF](https://arxiv.org/pdf/2601.06235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 138. Think Bright, Diffuse Nice: Enhancing T2I-ICL via Inductive-Bias Hint Instruction and Query Contrastive Decoding

**arXiv ID:** 2601.06169 | [PDF](https://arxiv.org/pdf/2601.06169v1)

**作者:** Zhiyong Ma `[一作]` (South China University of Technology), Qingyuan Chuai `[通讯]` (Cao Tu Li (Guangzhou) Technology Co., Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一个无训练的T2I-ICL框架TBDN，解决合规失败和先验主导幻觉。

**💡 创新点**

创新点是结合 Hint Instruction 与 Query Contrastive Decoding 两种闭环机制，分别针对两类瓶颈实现互补提升。

**🔧 技术方法**

采用提示工程与对比解码技术，结合现有 LVLM 与扩散模型，实现无训练的推理与生成。

**📊 数据集**

使用 CoBSAT、Text-to-Image Fast Mini-ImageNet 与 Dreambench++ 等公开基准数据集进行评测。

**📈 对比分析**

与多种基线对比，TBDN 在 CoBSAT 与 T2IFMIT 上平均准确率提升约 20%–80%，在 Dreambench++ 上提示跟随得分显著优于现有方法。

**⚠️ 局限性**

局限在需双模型耦合导致文本-图像语义桥接不够理想，难以处理细粒度视觉合成任务，且对 MLLM 的迁移性尚未验证。

---

## 139. An evaluation of LLMs for political bias in Western media: Israel-Hamas and Ukraine-Russia wars

**arXiv ID:** 2601.06132 | [PDF](https://arxiv.org/pdf/2601.06132v1)

**作者:** Rohitash Chandra `[一作]` (University of New South Wales), Yuting Wu `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对英国两大报纸BBC和《卫报》在俄罗斯-乌克兰战争与以色列-哈马斯冲突期间的新闻内容，使用大型语言模型对文章进行政治立场（左、中、右）分类，并结合情感分析，系统性评估媒体偏见随时间和冲突的变化。

**💡 创新点**

首次将多种大模型（BERT、Gemini、DeepSeek）并行应用于长文本政治立场判定，探究模型内部“世界观”对偏见识别的影响，并提供对比框架来评估不同模型在相同文本上的偏差；同时提出滑动窗口+投票策略解决BERT截断问题。

**🔧 技术方法**

自然语言处理与机器学习技术：n-gram主题检索、滑动窗口投票的BERT分类、零温度提示的Gemini与DeepSeek推理、DistilRoBERTa情感分析；并使用Python爬虫、API抓取实现数据预处理。

**📊 数据集**

从BBC的huggingface实时新闻数据和《卫报》API抓取的两大冲突相关文章，覆盖2020-2024年，共约1.8万篇，按关键词（“Russia”“Ukraine”“Israel”“Hamas”等）筛选后得到四个子集：BBC-俄乌、BBC-以色列哈马斯、卫报-俄乌、卫报-以色列哈马斯。

**📈 对比分析**

通过对同一篇文章分别由三模型预测立场，并对预战期与战期、两报纸进行比例和平均偏置分数对比；结果显示DeepSeek持续左倾，BERT偏右，Gemini保持中性；BBC报道更接近中立但波动大，卫报偏左；情感分析揭示右倾文稿情绪更激烈。

**⚠️ 局限性**

研究仅覆盖两家英国英文媒体，数据采集依赖关键词/标签过滤，可能遗漏相关报道；未进行人工标注基准验证，模型自身偏见导致结果不确定；长文本拆分或截断可能丢失语境；仅关注两场冲突，结果不可推广到其他地区或语言。

---

## 140. PixRec: Leveraging Visual Context for Next-Item Prediction in Sequential Recommendation

**arXiv ID:** 2601.06458 | [PDF](https://arxiv.org/pdf/2601.06458v1)

**作者:** Sayak Chakrabarty `[一作]` (Northwestern University), Souradip Pal `[通讯]` (Purdue University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5035784469)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 PixRec，一种将商品图像与文本共同输入的多模态序列推荐框架，用于预测用户下一个购买行为。

**💡 创新点**

创新点在于将视觉语言模型与自回归文本生成、对比对齐及双塔结构相结合，并通过参数高效微调（PEFT）实现低成本训练，首次证明图像信息能显著提升序列推荐效果。

**🔧 技术方法**

技术包括 SmolVLM / PaliGemma2 视觉语言因果模型、LoRA/QLoRA 参数高效微调、InfoNCE 对比损失、双塔表示融合、生成候选并使用 BM25S 检索进行排序。

**📊 数据集**

使用 Amazon Reviews 2023 订阅盒子（Subscription_Boxes）和杂志订阅（Magazine_Subscriptions）类别的带图像商品数据，构建用户交互序列和商品语料库。

**📈 对比分析**

与纯文本 LLM SmolLM2 对比，PaliGemma2 在 Recall@1 提升约3倍、Recall@10 提升约40%、NDCG@10 提升约19%，显示视觉特征显著提升候选质量。

**⚠️ 局限性**

局限包括：排名顺序仍需额外工程优化；模型规模较大导致推理延迟；对图像缺失或多模态覆盖不足的场景表现不佳；跨域泛化尚未充分验证。

---

## 141. ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking

**arXiv ID:** 2601.06487 | [PDF](https://arxiv.org/pdf/2601.06487v1)

**作者:** Qiang Zhang `[一作]` (Tongyi Lab, Alibaba Group), Zheng-Jun Zha `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ArenaRL，一种针对开放式 LLM 代理任务的强化学习框架，将奖励评估从点值评分转变为基于团队内相对排名的评估，并通过过程感知的双向比较与 seeded 单淘汰锦标赛实现高效优势信号。

**💡 创新点**

创新点在于：1) 通过相对排名消除奖励模型的判别坍塌问题；2) 设计了多层级评价 Rubric 与双向评分机制提升判别细粒度；3) 引入 seeded 单淘汰锦标赛，将复杂度从 O(N²) 降至 O(N) 仍保持近似准确；4) 搭建了 Open‑Travel 与 Open‑DeepResearch 两大全流程开放式代理基准。

**🔧 技术方法**

技术实现主要包括：过程感知双向 pairwise 评估、锦标赛相对排名、基于 LLM 的 Judge、RL‑PPO（或 TRPO）优化、KL 正则化、以及对比式奖励映射为优势信号。

**📊 数据集**

使用的数据集包括自研的 Open‑Travel（2600 SFT+1626 RL 例子）和 Open‑DeepResearch（2662 SFT+2216 RL 例子），以及公开写作基准 WritingBench、HelloBench、LongBench‑write。

**📈 对比分析**

与 GRPO、GSPO、SFT 以及 GPT‑4o、Grok‑4、Gemini‑2.5‑pro、Claude‑3.7‑Sonnet 等基线对比，ArenaRL 在 Open‑Travel 平均赢率达 41.8%，在 Open‑DeepResearch 赢率 64.3% 且有效率 99%，写作任务平均分提升 6–7%，均显著优于基线。

**⚠️ 局限性**

局限性包括：1) 仍依赖 LLM‑Judge 评判，可能带来评估偏差；2) 对极长或多模态任务的扩展尚未验证；3) 需要多轮 pairwise 评估，尽管降低到 O(N) 但在极大样本量下仍有计算压力；4) 对不同 LLM 框架的迁移性需要进一步探索。

---

## 142. HiDVFS: A Hierarchical Multi-Agent DVFS Scheduler for OpenMP DAG Workloads

**arXiv ID:** 2601.06425 | [PDF](https://arxiv.org/pdf/2601.06425v1)

**作者:** Mohammad Pivezhandi `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5079359777)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 HiDVFS，一种针对 OpenMP DAG 工作负载的分层多代理动态电压频率调节调度器，优先降低任务完成时间，同时兼顾能耗与温度。

**💡 创新点**

创新点包括：①将频率/核心选择、温度感知核心分组、任务优先级分配拆分为三个协同代理，显著降低动作空间与样本需求；②利用基于模型的奖励估计与未来奖励塑形，使得做法在极短的训练周期内即可收敛；③通过实时温度、功耗与性能概况驱动的决策，解决传统 DVFS 方案缺乏 per‑core 温度监测与任务特征匹配的问题。

**🔧 技术方法**

采用多代理强化学习（D3QN+短期模型预测）、动态温度感知与频率调节、基于 OpenMP DAG 的任务优先级调度、在线性能与能耗监测、服务器‑嵌入式分布式训练框架。

**📊 数据集**

实验数据集：NVIDIA Jetson TX2 嵌入式平台与 Barcelona OpenMP Tasks Suite（BOTS）共 9 个基准（FFT、Sparselu、Alignment 等）以及多种随机种子（42、123、456）。

**📈 对比分析**

与现有 DVFS 策略（GearDVFS、zTT、DynaQ 等）对比，HiDVFS 在 FFT 基准上 L10 速度提升 3.44×（4.16 s 对比 14.32 s），能耗下降 50.4%（63.7 kJ 对比 128.4 kJ）；在全部 9 个 BOTS 基准中平均提升 3.95×、能耗降低 47.1%。

**⚠️ 局限性**

局限性：①对支持细粒度 per‑core DVFS 的硬件（如 Jetson TX2）高度依赖，传统 x86 体系结构不易直接迁移；②需要持续的温度与功耗监测，若传感器或驱动不稳定会影响决策；③多代理协同与模型训练仍需额外的服务器资源，部署复杂度相对较高；④在极高并发或多节点分布式环境下的扩展性尚未充分验证。

---

## 143. HiMeS: Hippocampus-inspired Memory System for Personalized AI Assistants

**arXiv ID:** 2601.06152 | [PDF](https://arxiv.org/pdf/2601.06152v1)

**作者:** Hailong Li `[一作]` (WeChat, Tencent Inc.), Xingyu Fan `[通讯]` (WeChat, Tencent Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HiMeS，一种融合短期对话压缩和长期用户记忆的记忆增强式检索生成框架，用于提升工业级AI助手的个性化问答质量

**💡 创新点**

①将短期对话压缩为查询重写并强化检索；②使用分区长期记忆与注意力式再排序，模拟大脑海马与皮层协同；③采用强化学习+多源奖励优化查询重写模型；④框架兼容多种黑盒LLM，实现一次训练多模型部署

**🔧 技术方法**

强化学习（GRPO）、多源奖励（HSER）、查询重写网络、分区向量存储、注意力式再排序、LLM压缩与检索融合

**📊 数据集**

工业真实场景多轮对话数据（包括用户发布内容与历史交互日志）以及合成的多轮对话重写数据集

**📈 对比分析**

与传统Native RAG（直接拼接历史与检索）以及仅加短期/长期模块的基线相比，HiMeS在CA、QA、QR指标上显著提升，最高达到90+分；跨不同黑盒LLM保持高性能，证明鲁棒性

**⚠️ 局限性**

仍依赖人工/LLM生成的重写数据质量，模型对极端主题或长篇对话的压缩效果有限；长期记忆的分区可能导致细粒度信息丢失

---

## 144. Australian Bushfire Intelligence with AI-Driven Environmental Analytics

**arXiv ID:** 2601.06105 | [PDF](https://arxiv.org/pdf/2601.06105v1)

**作者:** Tanvi Jois `[一作]` (Adelaide University), Faheem Ullah `[通讯]` (Zayed University)

**通讯引用:** 2633 | [OpenAlex ID](https://openalex.org/A5074689211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

利用NASA FIRMS火灾事件、Meteostat气象数据和Google Earth Engine NDVI指数构建时空数据融合框架，并训练多种机器学习模型对澳大利亚2015-2023年的火灾强度进行二分类预测。

**💡 创新点**

首次将多源环境数据通过空间时间联合，结合类不平衡处理和堆叠集成模型，实现二分类火灾高危区预测精度达87%。

**🔧 技术方法**

采用随机森林、XGBoost、LightGBM、MLP以及堆叠集成算法，配合SMOTE‑Tomek类平衡和特征工程如交互项、周期性转换。

**📊 数据集**

NASA FIRMS火灾记录、Meteostat日常气象观测、Google Earth Engine的MODIS NDVI，时间范围2015-2023。

**📈 对比分析**

通过对三分类与二分类的模型准确率、宏F1、召回率等指标比较，发现二分类堆叠模型在87%准确率、0.77 ROC‑AUC下表现最佳。

**⚠️ 局限性**

仅使用NDVI作为植被指标、空间时间对齐导致样本删减、模型仍受数据稀缺与其他驱动因子缺失限制，难以推广到更复杂火灾场景。

---

## 145. PRISP: Privacy-Safe Few-Shot Personalization via Lightweight Adaptation

**arXiv ID:** 2601.06471 | [PDF](https://arxiv.org/pdf/2601.06471v1)

**作者:** Junho Park `[一作]` (Seoul National University), Taesup Moon `[通讯]` (Seoul National University)

**通讯引用:** 2609 | [OpenAlex ID](https://openalex.org/A5080346989)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PRISP框架，实现零任务数据、少量用户数据的LLM个性化；

**💡 创新点**

创新点在于使用文本到LoRA的超网络生成任务感知Anchor LoRA，并在此基础上仅训练轻量桥接矩阵实现用户级个性化，避免跨用户参数共享与大量训练；

**🔧 技术方法**

采用LoRA、文本到LoRA超网络、桥接矩阵（C矩阵）以及少量用户样本进行微调；

**📊 数据集**

在LaMP基准（六个文本分类与生成任务）上进行评估，使用Qwen3-0.6B作为主模型；

**📈 对比分析**

与RAG、PAG、Per-Pcs、PriME、OPPU等基线对比，PRISP在few-shot下不依赖任务数据即可获得最高平均分，并在计算成本（显存与训练时间）和隐私方面优于其他方法；

**⚠️ 局限性**

局限在于仅评估静态个性化，未考虑持续学习场景，且依赖超网络的表达能力，若超网络质量不足可能影响个性化效果。

---

## 146. QES-Backed Virtual FIDO2 Authenticators: Architectural Options for Secure, Synchronizable WebAuthn Credentials

**arXiv ID:** 2601.06554 | [PDF](https://arxiv.org/pdf/2601.06554v1)

**作者:** Kemal Bicakci `[一作]` (Informatics Institute), Yusuf Uzunay `[通讯]` (Securify Information Technology and Security Training Consulting Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

实现了一个虚拟FIDO2身份验证器，利用PKCS#11安全令牌保护私钥并通过云端同步实现跨设备无缝登录，

**💡 创新点**

将硬件根信任与云端无信任同步结合，提出了OPRF增强方案以防止跨协议攻击，兼容现有WebAuthn接口，

**🔧 技术方法**

采用虚拟身份验证器、PKCS#11接口、HKDF密钥派生、OPRF隐私保护函数和云端对象存储同步，

**📊 数据集**

实验中使用标准Intel i7笔记本、USB型QES/PKCS#11签名卡与公共云存储服务进行性能评估，

**📈 对比分析**

对比结果显示：解锁阶段约42 ms，MakeCredential 15 ms，GetAssertion 7 ms，云同步下载/上传约220 ms，性能可接受且与硬件密钥相近，

**⚠️ 局限性**

局限在于基础架构仍易受跨协议签名泄露风险，解锁后可被恶意软件利用，且依赖令牌的PIN/PUK安全与实现细节。

---

## 147. Leveraging Membership Inference Attacks for Privacy Measurement in Federated Learning for Remote Sensing Images

**arXiv ID:** 2601.06200 | [PDF](https://arxiv.org/pdf/2601.06200v1)

**作者:** Anh-Kiet Duong `[一作]`, Minh-Tan Pham `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究将会员推断攻击(MIA)作为定量工具，对联邦学习在遥感图像分类中的隐私泄露进行评估，并比较了多种联邦学习算法及其通信策略的隐私与性能表现。

**💡 创新点**

创新点在于：①首次将黑盒MIA作为统一的隐私度量框架应用于遥感领域的联邦学习；②系统比较传统权重聚合（FedAvg、FedProx）与特征通信策略（FedFT、FedProxFT、FedFFT、FedMFT）的隐私泄露；③发现通信高效的特征共享方案在保持竞争性能的同时显著降低了MIA成功率。

**🔧 技术方法**

主要技术包括：联邦学习算法FedAvg、FedProx；特征通信FL方法FedFT、FedProxFT、FedFFT、FedMFT；三种黑盒MIA技术（熵攻击、改进熵攻击、LiRA）；评估指标AUC和低FPR下的TPR；使用ResNet18作为基准网络。

**📊 数据集**

使用的遥感图像分类数据集为UC‑Merced（2100张图像，21类）和AID（10000张图像，30类），每个数据集按70/30划分为训练/验证。

**📈 对比分析**

在准确率、通信量和隐私泄露（MIA成功率）三方面对比：全局训练准确率最高但隐私最低；本地训练隐私最好但准确率最低；FedFFT/FedMFT在准确率和通信量上表现优异；FedProxFT在保持较高准确率的同时，MIA成功率最低，提供了隐私与性能的最佳折中。

**⚠️ 局限性**

局限性包括：仅使用两大遥感数据集；仅考察黑盒MIA，未探讨白盒或更高级攻击；通信成本估计粗略；实验规模有限，未涵盖更大规模联邦网络；未结合差分隐私等其它隐私机制进行比较。

---

## 148. Projecting Out the Malice: A Global Subspace Approach to LLM Detoxification

**arXiv ID:** 2601.06226 | [PDF](https://arxiv.org/pdf/2601.06226v1)

**作者:** Zenghao Duan `[一作]` (State Key Laboratory of AI Safety Institute of Computing Technology Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需重新训练、轻量级的毒性抑制方法GLOSS，直接识别并剔除LLM模型中全局毒性子空间；

**💡 创新点**

创新点在于将毒性视为共享的全局子空间而非单个向量或层级子空间，并通过三步筛选+PCA得到稳定的毒性方向；

**🔧 技术方法**

主要技术包括对FFN输出进行对比分析、SVD提取候选毒性方向、基于词表投影的毒性评分、PCA构建全局子空间以及投影去毒；

**📊 数据集**

使用公开毒性数据集RealToxicityPrompts、PolyglotoxicityPrompts以及少量（约500对）毒性/非毒性句子进行子空间提取；

**📈 对比分析**

与多种基线（SSFT、DPO、ProFS、Self‑Reminder、Safe‑Decoding等）对比，GLOSS在六款LLM上显著降低毒性分数（约44‑54%下降）同时保持或提升流畅度与一致性，且不显著增加困惑度；

**⚠️ 局限性**

局限性包括仅在0.6‑14B规模模型验证，未对更大模型或更广泛的基线（提示或检测方法）进行评估，且需要手工标注的毒性/非毒性配对样本。

---

## 149. MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation

**arXiv ID:** 2601.06519 | [PDF](https://arxiv.org/pdf/2601.06519v1)

**作者:** Yuelyu Ji `[一作]` (University of Pittsburgh), Yanshan Wang `[通讯]` (University of Pittsburgh)

**通讯引用:** 6500 | [OpenAlex ID](https://openalex.org/A5080116611)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出MedRAGChecker框架，对医学RAG生成的长文本进行原子命题拆解并通过文本NLI与结构化知识图谱进行联合验证，生成每条命题的可信度评分；

**💡 创新点**

创新点在于将知识图谱一致性信号与自然语言推理融合为命题级验证，并通过教师模型蒸馏生成轻量级检索器与验证器；

**🔧 技术方法**

使用GPT‑4.1作为教师生成原子命题与NLI标签，蒸馏成Meditron3‑8B、Med‑Qwen2‑7B等医学LLM的学生模型，结合DRKG知识图谱与TransE得分，并在logit空间进行加权融合；

**📊 数据集**

评估数据集包括PubMedQA、MedQuAD、LiveQA、MedRedQA四个医学QA基准，使用统一的PubMed检索管线；

**📈 对比分析**

与教师NLI、单一学生验证器和基线LLaMA对比，MedRAGChecker在四个基准上在命题级准确率、宏F1和安全错误率上均提升，KG融合在安全关键命题上显著提高人类一致性；

**⚠️ 局限性**

主要局限在于依赖教师模型生成的伪标签、知识图谱覆盖不足、NLI与KG信号的校准与类不平衡，以及对罕见或消化性问题的泛化能力不足。

---

## 150. Smart Privacy Policy Assistant: An LLM-Powered System for Transparent and Actionable Privacy Notices

**arXiv ID:** 2601.06357 | [PDF](https://arxiv.org/pdf/2601.06357v1)

**作者:** Sriharshini Kalvakuntla `[一作]` (Binghamton University), Zhaohan Xi `[通讯]` (Binghamton University)

**通讯引用:** 111 | [OpenAlex ID](https://openalex.org/A5026309535)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于LLM的智能隐私政策助手，自动解析、结构化并给出风险评分与可解释说明。

**💡 创新点**

创新点在于将隐私政策拆分为条款级别，映射到结构化隐私模式，结合可解释的单调风险评分与实时上下文警告。

**🔧 技术方法**

采用大语言模型（LLM）进行条款分类与解释，配合规则/监督分类、OCR、预处理、风险评分模型。

**📊 数据集**

使用公开收集的多域隐私政策语料（社交媒体、电子商务、VPN、健康/健身、生产力工具），并人工标注子集用于评估。

**📈 对比分析**

与基线（规则、监督分类、LLM摘要）比较，F1平均值提升至0.76（比基线提升约0.18），在多数据集上表现最好。

**⚠️ 局限性**

局限包括对复杂法律语言的误解、条款模糊性、不同格式解析难度、实时部署的延迟与计算成本、并非法律合规替代。

---

## 151. Separation Results for Constant-Depth and Multilinear Ideal Proof Systems

**arXiv ID:** 2601.06299 | [PDF](https://arxiv.org/pdf/2601.06299v1)

**作者:** Amik Raj Behera `[一作]` (University of Copenhagen), Srikanth Srinivasan `[通讯]` (University of Copenhagen)

**通讯引用:** 564 | [OpenAlex ID](https://openalex.org/A5025305994)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

在理想证明系统（IPS）框架下，证明了常数深度IPS与多项式阶IPS之间的分层和分离定理，揭示了更深度能够产生更强证明的可能性。

**💡 创新点**

首次将功能方法与倍数方法相结合，并引入地址器（addressing gadget）构造，使得多项式在布尔域上取值仅为 0/1，从而实现了常数深度IPS与多项式阶IPS的严格分离。

**🔧 技术方法**

主要技术包括：功能方法（functional method）、倍数方法（multiples method）、地址器构造、矩阵秩下界与多项式阶IPS的层化分析。

**📊 数据集**

该工作为纯理论论文，无实验数据集，全部证明均在符号计算与代数复杂度框架内完成。

**📈 对比分析**

通过严格的证明与下界分析，展示了常数深度IPS的证明长度随深度显著增长，且多项式阶IPS对特定多项式实例需要指数级大小，表现出明显的复杂度阶梯。

**⚠️ 局限性**

局限在于目前仅适用于特征为 0 的域，且方法主要针对理论证明而非实际计算；扩展到更广泛的域或更通用的证明系统仍需进一步研究。

---

## 152. Cyber Threat Detection and Vulnerability Assessment System using Generative AI and Large Language Model

**arXiv ID:** 2601.06213 | [PDF](https://arxiv.org/pdf/2601.06213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 153. Cross-Border Data Security and Privacy Risks in Large Language Models and IoT Systems

**arXiv ID:** 2601.06612 | [PDF](https://arxiv.org/pdf/2601.06612v1)

**作者:** Chalitha Handapangoda `[一作]` `[通讯]` (New York University), Chalitha Handapangoda (New York University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为JAPD的架构，动态集成本地化加密、差分隐私与实时合规验证，针对跨境LLM与IoT数据安全。

**💡 创新点**

首次在推理阶段引入基于管辖区的差分隐私，结合多层加密与零知识证明，实现零合规违规。

**🔧 技术方法**

动态管辖区感知路由、推理时差分隐私、加密层与本地密钥托管、零知识证明等技术。

**📊 数据集**

使用多国模拟环境（美国、欧盟、中国）与公开LLM训练与IoT流量模拟数据，未使用真实敏感数据。

**📈 对比分析**

与标准加密、联邦学习、数据本地化+DP三种基线比较，评估攻击成功率、信息泄漏、合规违规率、系统延迟等指标；结果显示ASR<5%、CVR=0%，整体开销15-18%。

**⚠️ 局限性**

仅在7B模型规模下验证，未覆盖更大模型；缺乏对更细粒度法规的完整实现；硬件加速与ZK证明的效率还有提升空间。

---

## 154. Learning Password Best Practices Through In-Task Instruction

**arXiv ID:** 2601.06650 | [PDF](https://arxiv.org/pdf/2601.06650v1)

**作者:** Qian Ma `[一作]` (Pennsylvania State University), Brett Frischmann `[通讯]` (Villanova University)

**通讯引用:** 2030 | [OpenAlex ID](https://openalex.org/A5032312244)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出并验证了“教学性摩擦”设计思路，通过在密码创建过程中嵌入简短、规则相关的即时反馈与交互式确认，帮助用户即时学习并改进密码策略。

**💡 创新点**

创新点在于将微学习与行为心理学原理结合，首次系统评估“教学性摩擦”在密码生成任务中的即时学习、规则遵从与行为-知识一致性，且提供了多层级摩擦设计的对比。

**🔧 技术方法**

使用了基于密码强度评估的交互式用户界面、即时强度计、规则提示系统以及后测知识问卷。

**📊 数据集**

实验数据来自128名在Prolific上完成的参与者，记录了密码键盘日志、提示交互、后测问卷答案以及密码强度评分。

**📈 对比分析**

对四种实验条件（仅计量、简短提示、详细提示、交互式提示）进行重复测量ANOVA比较，结果显示所有教学性摩擦条件均显著提升规则遵从（约90%）和知识识别（约60%），交互式提示在行为-知识一致性上略优。

**⚠️ 局限性**

局限性包括：仅测试单一低风险密码生成任务，短暂的学习转移时间，且未验证在高风险或长期使用场景中的有效性。

---

## 155. When Smaller Wins: Dual-Stage Distillation and Pareto-Guided Compression of Liquid Neural Networks for Edge Battery Prognostics

**arXiv ID:** 2601.06227 | [PDF](https://arxiv.org/pdf/2601.06227v1)

**作者:** Dhivya Dharshini Kannan `[一作]` (Singapore Institute of Technology), Man-Fai Ng `[通讯]` (Institute of High Performance Computing)

**通讯引用:** 4697 | [OpenAlex ID](https://openalex.org/A5029325364)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 DLNet 框架，将高容量液态神经网络教师模型通过双阶段蒸馏、裁剪与 Pareto 选择压缩为可部署的边缘学生模型，用于电池健康预测。

**💡 创新点**

创新点在于：①使用 Euler 离散化将连续时间液态网络转换为可在嵌入式设备上执行的模型；②引入双阶段蒸馏与裁剪的循环训练，使模型在压缩后保持甚至提升性能；③通过 Pareto 指标在误差–成本双目标空间中自动挑选最优模型，实现性能与资源的最佳平衡。

**🔧 技术方法**

技术包括液态神经网络（LNN）、Euler 离散化、知识蒸馏（结合标签和教师引导）、基于稀疏度的剪枝、量化与 LiteRT 兼容导出、以及多目标 Pareto 优化。

**📊 数据集**

使用 MIT‑Stanford 电池周期数据集（SoH 时序数据），对电池在不同健康阶段的性能进行训练与评估。

**📈 对比分析**

与多种基线模型（CNN、TCN、LSTM、GRU、Transformer、LNN）以及不同实现（PyTorch、LiteRT 量化）进行对比。DLNet 最终学生在 Arduino Nano 33 BLE Sense 上实现 MAE 0.0066（相较教师 0.0078 下降 15.4%），模型尺寸从 616 kB 减至 94 kB（84.7% 降幅），推理时间 21 ms，显示了比教师更优的误差–成本比。

**⚠️ 局限性**

限制主要在于：①双阶段蒸馏与裁剪流程较为复杂，需多轮训练；②在极低资源设备（如更小 MCU）上可能仍难以满足实时性；③对不同任务的通用性需要进一步验证，当前仅在电池健康预测上验证。

---

## 156. PromptPort: A Reliability Layer for Cross-Model Structured Extraction

**arXiv ID:** 2601.06151 | [PDF](https://arxiv.org/pdf/2601.06151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 157. StablePDENet: Enhancing Stability of Operator Learning for Solving Differential Equations

**arXiv ID:** 2601.06472 | [PDF](https://arxiv.org/pdf/2601.06472v1)

**作者:** Chutian Huang `[一作]` (Hong Kong University of Science and Technology), Yang Xiang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25877 | [OpenAlex ID](https://openalex.org/A5100666554)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 StablePDENet 的自监督神经算子框架，利用对抗训练提升 PDE 求解算子的稳定性，同时保持对未扰动输入的高精度。

**💡 创新点**

核心创新在于将算子学习转化为 min‑max 优化（对抗训练），并证明对抗训练隐式约束了算子的 Fréchet 导数谱范数，从而实现对输入扰动的稳定性保证。

**🔧 技术方法**

技术手段包括：物理信息化 DeepONet（PIDeepONet）作为基准算子；PGD（l∞）对抗攻击生成扰动；自监督物理损失（PDE、边界、初始条件）；自动微分求算子导数；梯度上升/下降的两阶段训练（攻击‑防御交替）。

**📊 数据集**

使用多种人工合成数据集：Gaussian 随机场、三次多项式、双三角函数、二维 Poisson 等；每类问题都从 2000 组样本中采样，形成原始数据集与对抗扰动数据集。

**📈 对比分析**

与传统 PIDeepONet 进行对比。对未扰动输入，两者相差不大（误差 < 0.02），但在对抗扰动下，PIDeepONet 的相对 L2 误差从 0.7–10 级幅度大幅升高，而 StablePDENet 维持在 0.02–0.05 之间，显著提升鲁棒性。谱范数对比亦显示 StablePDENet 的 Fréchet 导数范数大幅下降。

**⚠️ 局限性**

局限性：仅在合成、低维 PDE 上验证；对抗训练的计算开销较大；目前只考虑 l∞ 约束；对极大扰动或其他网络结构的泛化能力尚未充分评估；未涉及真实实验或高维物理问题。

---

## 158. Extended Target Adaptive Beamforming for ISAC:A Perspective of Predictive Error Ellipse

**arXiv ID:** 2601.06125 | [PDF](https://arxiv.org/pdf/2601.06125v1)

**作者:** Shengcai Zhou `[一作]` (Nanjing University), Chan-Byoung Chae `[通讯]` (Yonsei University)

**通讯引用:** 10007 | [OpenAlex ID](https://openalex.org/A5079863632)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出基于新无线电 V2X 标准的 ISAC（集成感知与通信）系统，针对车体被视为扩展目标，设计了两阶段自适应波束形成方案：①初始波束建立阶段利用预测误差椭圆的并集实现快速定位；②波束调整阶段采用狭窄波束覆盖最近散射点与通信接收器，实现低复杂度的跟踪。

**💡 创新点**

创新点包括：①在 OFDM 与 UPA 配置下推导出雷达参数的 Cramér‑Rao 边界（CRB），实现对测量误差的闭式分析；②引入误差椭圆并集（PEE）与最小外接椭圆（MEE）算法，动态确定波束覆盖范围；③提出基于散射点与接收器相对位置的间接跟踪与自适应阈值触发的 EKF 方案；④利用天线控制映射实现对波束宽度的精细调节。

**🔧 技术方法**

核心技术包括：OFDM 波形、统一平面阵列（UPA）天线、误差椭圆与最小外接椭圆（MEE）优化、扩展卡尔曼滤波（EKF）预测、天线激活映射控制、Cramér‑Rao 限界分析。

**📊 数据集**

实验使用模拟 V2X 场景：车辆沿直线行驶，BS 采用 64、128、256、1024 天线阵列，车辆模型为 5 m×2 m 的六散射点扩展目标，NR‑V2X 帧结构参数如子载波间隔 120 kHz、槽时长 0.125 ms 等；未使用公开真实数据集，全部结果基于仿真生成。

**📈 对比分析**

与传统波束扫瞄、全波束 ISAC（Omni‑ISAC）以及仅用单点目标的 EKF 方案进行对比。结果显示：ISAC‑IBE 在 64、128、256、1024 天线阵列下相较波束扫瞄提升 5.2 %~32.4 % 的可达率；ISAC‑ABA 与 ISAC‑RB 的可达率相近，但算法调用次数仅 1% 左右；在 8×8、16×16、32×32 阵列下，ISAC‑ABA/ISAC‑RB 可达率约 13–17 bps/Hz，明显高于 Point‑target、ISAC‑DB 与扫瞄方案。

**⚠️ 局限性**

局限性包括：①仅考虑 LoS 路径，未纳入多径与遮挡效应；②车辆被简化为均匀散射体，未考虑车身轮廓与复杂雷达散射；③所有实验均为仿真，缺乏真实数据验证；④MEE 与 EKF 的计算复杂度在极大阵列时仍高，需要进一步简化。

---

## 159. SRFlow: A Dataset and Regularization Model for High-Resolution Facial Optical Flow via Splatting Rasterization

**arXiv ID:** 2601.06479 | [PDF](https://arxiv.org/pdf/2601.06479v1)

**作者:** JiaLin Zhang `[一作]` (Guangdong University of Technology), Dong Li `[通讯]` (Guangdong University of Technology)

**通讯引用:** 11902 | [OpenAlex ID](https://openalex.org/A5100407416)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了SRFlow高分辨率人脸光流数据集以及基于Splatting Rasterization的SRFlowNet网络，用以提升面部细粒度运动估计与微表情识别性能。

**💡 创新点**

创新点在于：①利用3D高斯光栅化生成精确光流标签；②设计四种面部专用正则化损失（TVR、FDR、MIGAR、IGVAR），通过面部掩模与图像梯度抑制高频噪声与纹理缺失区误差；③结合SRFlow数据集对现有光流模型进行细化，从而显著提升光流精度与下游任务表现。

**🔧 技术方法**

技术核心包括3D Gaussian Avatar重建、Flow Rasterizer、Splatting Rasterization Guided FlowNet（SKFlow骨干改造）、自定义正则化损失、光流-微表情特征融合（Off‑TANet）。

**📊 数据集**

使用新构建的SRFlow数据集（约11161帧，高分辨率2200×3208），并在SAMM、CASME II、SMIC及其组合集上评测微表情识别。

**📈 对比分析**

与传统光流方法（Gunnar‑Farnebäck、TV‑L1）以及多种深度光流模型（FlowNet3.0‑CSS、SKFlow、RPKNet、MemFlow、DPFlow）比较，SRFlowNet在SRFlow测试集上EPE降至0.2953（≈42%提升），在微表情识别上各数据集取得最高或近乎最高F1/WAUC分数，整体表现优于所有基线。

**⚠️ 局限性**

局限性包括：正则化损失过度平滑可能抑制极细微动作；训练依赖高质量3D重建，生成过程复杂；未探究不同损失组合或多尺度特征融合；微表情数据集分辨率仍较低，难以充分发挥高分辨率光流优势。

---

## 160. Towards Building efficient Routed systems for Retrieval

**arXiv ID:** 2601.06389 | [PDF](https://arxiv.org/pdf/2601.06389v1)

**作者:** Ramnath Kumar `[一作]` (University of California Los Angeles), Cho-Jui Hsieh `[通讯]` (University of California Los Angeles)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于可学习路由的检索框架FastLane，动态选择查询的最具信息量的token视图，从而消除传统late-interaction模型的token级别全量交互。

**💡 创新点**

创新点在于将token级别交互的高效性与单向量检索的可扩展性结合，通过自注意力+Gumbel-Softmax路由学习，在保持近似性能的同时将计算复杂度降低30倍，并实现与ANNS的无缝对接。

**🔧 技术方法**

使用Transformer+ColBERT骨干，结合自注意力层、Gumbel-Softmax重参数化、straight-through估计以及视图掩蔽机制，实现端到端可微的路由选择。

**📊 数据集**

在MS MARCO和TREC-DL 19这两个标准检索基准上进行评估。

**📈 对比分析**

与ColBERT、SGPT、ANCE、DyNNIBAL等SOTA方法对比，FastLane在MS MARCO上MRR@10和NDCG@10保持竞争水平，甚至比单向量双塔模型高约8%，但与ColBERT相比略低1%；同时在检索时延上实现近8倍甚至30倍的加速。

**⚠️ 局限性**

主要局限在于仍需为每个文档存储大量token视图，导致索引内存占用高；路由仅选取单一视图，可能错失多样信息；以及模型仍继承训练数据的偏见。

---

## 161. Sissi: Zero-shot Style-guided Image Synthesis via Semantic-style Integration

**arXiv ID:** 2601.06605 | [PDF](https://arxiv.org/pdf/2601.06605v1)

**作者:** Yingying Deng `[一作]` (University of Science and Technology Beijing), Xucheng Yin `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 4071 | [OpenAlex ID](https://openalex.org/A5074514262)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练‑free 的风格引导图像生成框架 Sissi，通过将风格图像与文本提示拼接作为上下文输入到预训练的 ReFlow 填图模型中实现零样本风格迁移。

**💡 创新点**

创新点：① 将风格图像视为上下文，利用 ReFlow 的 in‑context 学习完成无训练的风格控制；② 提出动态语义‑风格融合（DSSI）机制，动态调节文本与风格注意力权重，解决语义‑风格冲突；③ 在同一框架下支持多风格合成和多种图像编辑任务。

**🔧 技术方法**

技术手段：基于 ReFlow（Transformer‑based Diffusion）填图模型；多模态注意力融合；动态语义‑风格融合模块；无训练的图像‑文本联合输入。

**📊 数据集**

使用数据集：60+100 个人工艺术风格图像集合（来源于 InstaStyle 等），CIFAR‑100 类别作为文本提示；评估采用预训练 CLIP 嵌入。

**📈 对比分析**

与 WSDT、VSP、StyleShot、StyleAligned、DEADiff、InstaStyle、InstantStyle、CSGO 等 SOTA 方法进行定量（CLIP 内容/风格相似度）和用户研究对比。Sissi 在风格一致性最高、内容保持良好、整体质量最佳，生成速度约 15 秒/512×512，速度上与训练‑free 方案相当，远快于需要长时间训练的模型。

**⚠️ 局限性**

局限性：① 在填图框架下，风格图像边缘可能被无意补全或扩展，导致视觉交接处出现伪影；② 对纯纹理输入的布局与融合表现欠佳，难以在内容图像中合理安置纹理。

---

## 162. Style-constrained inverse design of microstructures with tailored mechanical properties using unconditional diffusion models

**arXiv ID:** 2601.06469 | [PDF](https://arxiv.org/pdf/2601.06469v1)

**作者:** Weipeng Xu `[一作]` (Hong Kong University of Science and Technology), Tianju Xue `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 723 | [OpenAlex ID](https://openalex.org/A5069417521)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将无条件扩散模型与可微分编程相结合的逆设计框架，用噪声作为可优化设计变量生成符合性能与风格约束的微结构。

**💡 创新点**

创新点在于将噪声输入视为设计变量，消除对标签数据和条件模型的需求，并通过向量-雅可比积链实现全流程梯度传播。

**🔧 技术方法**

使用无条件扩散模型（DDIM）、有限元分析、向量-雅可比积（VJP）逆向传播、BFGS优化等技术。

**📊 数据集**

采用MNIST手写数字和公开的二维正交骨架微结构图像（约9万张）作为训练集，后期转换为灰度图。

**📈 对比分析**

通过与传统条件扩散、手工或优化设计方法对比，实验显示在目标弹性、超弹性和弹塑性性能上能够达到或接近目标值，二值化率>98%，耗时几百秒到千余秒。

**⚠️ 局限性**

局限在于噪声输入可能偏离正态分布导致生成样本偏离学习分布、对极端性能的适配有限，以及缺乏硬几何约束如最小特征尺寸与连通性。

---

## 163. Coding for Fading Channels with Imperfect CSI at the Transmitter and Quantized Feedback

**arXiv ID:** 2601.06501 | [PDF](https://arxiv.org/pdf/2601.06501v1)

**作者:** Yuhan Yang `[一作]`, Bin Dai `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了在发射端拥有不完全CSI且仅能通过量化反馈获取CSI误差的衰落信道上，基于Schalkwijk‑Kailath(SK)方案的编码/解码方法，并将其推广到双径(ISI)衰落信道及多径衰落信道（无噪声反馈）

**💡 创新点**

创新点包括：①在SK迭代中加入模格格点函数和辅助信号，能够完全抵消发射端CSI误差；②利用活跃反馈实现误差消除；③将第二径信号视作中继并结合放大‑转发策略；④将多径信道通过离散傅里叶变换转换为频域MIMO信道，再对每个子信道应用SK方案

**🔧 技术方法**

主要技术手段：Schalkwijk‑Kailath编码、模格格点函数、MMSE估计、离散傅里叶变换(DFT)、水分配算法、辅助信号设计

**📊 数据集**

使用仿真数据（如SNR=10/4，误码率ε=10⁻⁶，块长N=100/200，量化噪声σ_z=10⁻³，CSI误差D=10⁻⁶）评估性能；没有公开的实际数据集

**📈 对比分析**

与经典SK方案（完全CSI、无量化反馈）和传统有限块长编码(如LDPC、Polar)对比，结果显示：在CSI误差和量化噪声趋近于零时，方案可逼近无反馈时的容量；在有限块长下，误码率随块长呈双指数下降，性能优于基准SK方案

**⚠️ 局限性**

局限性：①对多径信道的扩展受限于中继因子难以确定；②在量化噪声或CSI误差非零时，误码率仍受影响；③未考虑噪声反馈、MIMO或硬件失真；④方案实现需复杂的模格格点和同步反馈机制

---

## 164. Beyond BeautifulSoup: Benchmarking LLM-Powered Web Scraping for Everyday Users

**arXiv ID:** 2601.06301 | [PDF](https://arxiv.org/pdf/2601.06301v1)

**作者:** Arth Bhardwaj `[一作]` (Saint Francis High School), Gang Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 257346 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了两种基于LLM的网页抓取工作流（LLM辅助脚本和端到端LLM代理）在35个不同安全级别网站上的可行性与易用性

**💡 创新点**

首次将LLM辅助脚本与LLM端到端代理在现实安全场景下做横向对比，并提出统一的成功率、执行时间与手工干预度评估指标

**🔧 技术方法**

使用了Python的BeautifulSoup、Scrapy、Claude 3.5 Sonnet、Simular.ai等工具，结合LLM生成代码与代理直接交互

**📊 数据集**

构建了35个涵盖静态HTML、复杂HTML、简单/复杂身份验证及CAPTCHA等五大难度层级的测试网站集合

**📈 对比分析**

通过三次重复实验测量提取成功率、执行时间与手工干预量，发现代理在受保护网站上成功率显著高于脚本，但在静态站点脚本更快；总体上脚本适用于轻量抓取，代理适用于复杂/受限场景

**⚠️ 局限性**

实验受限于单一硬件环境、固定工具版本以及仅35个测试站点，且未对更强大反爬机制或多语言环境进行评估

---

## 165. Object-Centric World Models Meet Monte Carlo Tree Search

**arXiv ID:** 2601.06604 | [PDF](https://arxiv.org/pdf/2601.06604v1)

**作者:** Rodion Vakhitov `[一作]` (Moscow Institute of Physics and Technology), Aleksandr Panov `[通讯]` (Artificial Intelligence Research Institute and Moscow Institute of Physics and Technology and Fundamental Research Center for Computer Science of the Russian Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了ObjectZero，一种基于预训练Slot Attention编码器（SLATE/DINOSAUR）和图神经网络的结构化世界模型，并将其与MCTS相结合进行模型基强化学习；

**💡 创新点**

创新点在于将对象级分解与图神经网络动态、奖励、价值、策略预测整合到单一结构化MBC框架中，实现了对象级交互建模与决策规划的统一；

**🔧 技术方法**

采用Slot Attention、GNN、MCTS、预训练SLATE/DINOSAUR编码器、Transformer池化与搜索估值等技术；

**📊 数据集**

在CausalWorld的Object Reaching任务和Robosuite的Block Lifting任务上进行实验；

**📈 对比分析**

与EZ‑V2、DreamerV3、ROCA、OCRL、OC‑CA、OC‑SA等基线比较，ObjectZero在两项任务上收敛速度最快、成功率和回报均高于大多数基线，且与EZ‑V2几乎持平；

**⚠️ 局限性**

局限性包括在复杂现实场景中对象分解仍不完善，以及完全连接的GNN导致的计算复杂度随槽数呈二次增长，限制了在对象丰富环境中的训练效率与可扩展性。

---

## 166. APEX: Learning Adaptive Priorities for Multi-Objective Alignment in Vision-Language Generation

**arXiv ID:** 2601.06574 | [PDF](https://arxiv.org/pdf/2601.06574v1)

**作者:** Dongliang Chen `[一作]` (East China Normal University), Ying Qian `[通讯]` (East China Normal University)

**通讯引用:** 3474 | [OpenAlex ID](https://openalex.org/A5065775067)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出APEX框架，实现文本到图像生成任务的多目标自适应对齐；

**💡 创新点**

通过双阶段自适应归一化(DSAN)和基于梯度潜力、冲突惩罚与进展需求的动态优先级机制(P³)，解决了高方差目标占主导和梯度冲突导致的优化失衡问题；

**🔧 技术方法**

利用基于流匹配的GRPO强化学习、对奖励进行分组标准化与归一化，并在训练期间动态计算目标权重；

**📊 数据集**

在Stable Diffusion 3.5模型上，使用OCR测试集、DrawBench以及四个异构奖励（OCR、PickScore、DeQA、Aesthetic）进行评估；

**📈 对比分析**

与固定线性标量化和单目标专家模型对比，APEX在OCR保持竞争力的同时，在PickScore、DeQA和Aesthetic上实现显著提升，Pareto超体积提升约10.9倍；

**⚠️ 局限性**

需要额外的梯度估计开销，实验仅覆盖SD3.5-Medium，对更大模型和多模态任务的推广尚待验证。

---

## 167. Beyond Perfect Scores: Proof-by-Contradiction for Trustworthy Machine Learning

**arXiv ID:** 2601.06704 | [PDF](https://arxiv.org/pdf/2601.06704v1)

**作者:** Dushan N. Wadduwage `[一作]` (Old Dominion University), Leonidas Zimianitis `[通讯]` (Old Dominion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于Rubin因果模型的随机反证检验框架，用来评估医学ML模型在多单位少桶非IID数据下的可信度。

**💡 创新点**

创新点是将桶级标签置换作为零假设的随机化检验，通过Fisher样式p值量化模型是否依赖真正的临床信号，而非数据桶结构。

**🔧 技术方法**

使用标签置换、Rubin因果模型、Fisher随机化检验、预训练自监督模型、LightCNN、LightViT、ResNet‑18、RamanNet等。

**📊 数据集**

在受控基准（旋转MNIST、彩色FashionMNIST）以及真实医学数据集（细菌QPM、Raman细菌ID、抗菌敏感性预测）上验证。

**📈 对比分析**

通过与仅依赖准确率评估对比，发现高准确率并不总对应低p值；模型在受控基准中p值随因果信号强度下降，在真实数据中准确率高但p值大，表明需要进一步改进。

**⚠️ 局限性**

局限性包括置换训练成本高、对桶数和标签数的敏感性、假设标签与桶完全一致、未解决多标签或多任务情形，且需更多数据验证。

---

## 168. A Recommendation System-Based Framework for Enhancing Human-Machine Collaboration in Industrial Timetabling Rescheduling: Application in Preventive Maintenance

**arXiv ID:** 2601.06029 | [PDF](https://arxiv.org/pdf/2601.06029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 169. A Unified Shape-Aware Foundation Model for Time Series Classification

**arXiv ID:** 2601.06429 | [PDF](https://arxiv.org/pdf/2601.06429v1)

**作者:** Zhen Liu `[一作]` (South China University of Technology), Qianli Ma `[通讯]` (South China University of Technology)

**通讯引用:** 6459 | [OpenAlex ID](https://openalex.org/A5076609933)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种统一的形状感知基础模型 UniShape，用于时间序列分类任务。

**💡 创新点**

其创新点在于引入形状感知适配器，可自适应聚合多尺度形状（shapelet）并通过注意力权重筛选关键子序列，同时采用基于实例和形状级对比学习的原型预训练模块，实现跨域可迁移的形状模式学习。

**🔧 技术方法**

技术上结合了轻量级 CNN + 注意力池化的形状适配器、Transformer 编码器、以及自监督与弱监督混合的对比学习框架，配合多尺度窗口切分与标准化编码。

**📊 数据集**

预训练使用约 1.89 M 条多域单变量时间序列（UCR、UEA 及 8 个常用数据集），下游评估覆盖 128 个 UCR 数据集与 30 个额外领域数据集。

**📈 对比分析**

在全监督和零样本两种场景下均优于 16 种基线（包括 Rocket、InceptionTime、GPT4TS 等），在 128 UCR 数据集上的平均准确率达 0.8708，排名第 2.71，零样本下平均准确率为 0.7262，均显著高于其它方法。

**⚠️ 局限性**

主要局限在仅针对单变量时间序列，且对多尺度形状的选择仍需人工设定窗口长度，未来工作计划扩展到多变量场景并进一步自动化尺度搜索。

---

## 170. MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications

**arXiv ID:** 2601.06193 | [PDF](https://arxiv.org/pdf/2601.06193v1)

**作者:** Qing He `[一作]` (Ant Group), Junwei Liu `[通讯]` (Peking University)

**通讯引用:** 5398 | [OpenAlex ID](https://openalex.org/A5100422499)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个包含5个维度、22个数据集的中文医疗LLM评测基准，评估知识、伦理、安全、记录理解和情景服务能力。

**💡 创新点**

创新点在于构建基于真实临床对话和记录的情景驱动评测，以及训练基于SFT的“judge”模型实现可扩展、与专家一致的评估。

**🔧 技术方法**

采用大语言模型（Gemini、Qwen、DeepSeek等）生成数据，使用Supervised Fine‑Tuning训练评判模型，评估采用动态答案混洗与自动/人工混合打分。

**📊 数据集**

数据集包括5个公开中文医学数据集（CMExam、CHIP‑CDN、CMeEE等）和17个自研数据集，覆盖64个临床专科、约350k样本。

**📈 对比分析**

通过对10个主流LLM（Kimi‑K2‑Instruct、DeepSeek‑R1‑671B、Claude‑4‑Sonnet等）的平均分进行比较，Kimi‑K2‑Instruct总体得分最高77.3，智能服务维度普遍低于80，安全维度最高达90.6，评判模型准确率92.1%与专家一致。

**⚠️ 局限性**

限制在于基准仍聚焦中文医学场景，缺乏跨语言验证；评判模型虽然高效但仍依赖大量专家标注；部分情景任务的主观性和样本偏差可能影响结果。

---

## 171. Talking to Extraordinary Objects: Folktales Offer Analogies for Interacting with Technology

**arXiv ID:** 2601.06372 | [PDF](https://arxiv.org/pdf/2601.06372v1)

**作者:** Martha Larson `[一作]` (Radboud University), Martha Larson `[通讯]` (Radboud University)

**通讯引用:** 5971 | [OpenAlex ID](https://openalex.org/A5056272341)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对民间传说中非人类实体进行系统梳理，探讨其作为技术交互类比的可行性与启发。

**💡 创新点**

提出“otherware”概念，将技术的非人类特性与民间传说中的超凡物体对齐，突破传统人性化交互的单一类比框架。

**🔧 技术方法**

本研究主要采用文献综述与案例分析的理论方法，不涉及具体算法或软件实现。

**📊 数据集**

使用的“数据集”为经典民间故事文献，如格林兄弟童话、阿拉丁灯、北欧神话等文本与其对应的英文翻译。

**📈 对比分析**

未进行实验对比与性能评估；通过对比不同故事中的交互方式与现代技术接口，论证类比的可行性与局限性。

**⚠️ 局限性**

局限性包括：① 依赖传统故事的文化偏见与时代局限；② 缺乏实证实验验证类比对实际交互体验的影响；③ 可能忽视现代伦理与隐私问题，需进一步评估与规范。

---

## 172. AutoVulnPHP: LLM-Powered Two-Stage PHP Vulnerability Detection and Automated Localization

**arXiv ID:** 2601.06177 | [PDF](https://arxiv.org/pdf/2601.06177v1)

**作者:** Zhiqiang Wang `[一作]` (Beijing Electronic Science and Technology Institute), Yanjun Li `[通讯]` (15th Research Institute of China Electronics Technology Group Corporation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AutoVulnPHP，一个基于两阶段检测（结构先验+语义验证）与约束驱动 LLM 自动定位的 PHP 漏洞完整框架，并发布了首个大规模 PHP 漏洞数据集 PHPVD。

**💡 创新点**

创新点包括：① 结构‑语义级联检测管线，将高召回的 AST+数据流筛选与精确语义验证分离；② 通过模板+约束+链式推理的 ISAL 模块实现安全可验证的定位；③ 发布了覆盖 7 类主流漏洞、5.2M LOC 的真实项目数据集，填补 PHP 安全研究数据空缺。

**🔧 技术方法**

采用 AST 与控制/数据流增强、CodeT5/CodeBERT 编码器、风险偏置注意力、LLM 链式思维与模板生成、约束一致性校验及迭代细化等技术，实现高效精确的检测与定位。

**📊 数据集**

主要使用 PHPVD（26,614 文件、5.2M LOC）、PVts benchmark 以及基于 CVE 的真实项目数据进行训练、评估与对比实验。

**📈 对比分析**

与 HiddenCPG、RecurScan、Walden J 等传统工具和基线模型对比，AutoVulnPHP 在检测上达 99.7% 准确率、81% 定位成功率，且在真实项目扫描中发现 429 条未知漏洞（351 条已分配 CVE）。

**⚠️ 局限性**

局限性在于：① 对跨文件、跨类的数据流缺乏完整上下文，导致部分定位失败；② 约束冲突和 LLM 生成的代码偶尔出现语法/功能不符；③ 对高度动态 PHP 构造（如变量变量、call_user_func）的定位仍有 19% 未解决率。

---

## 173. LLM-Driven Accessible Interface: A Model-Based Approach

**arXiv ID:** 2601.06616 | [PDF](https://arxiv.org/pdf/2601.06616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 174. FairSCOSCA: Fairness At Arterial Signals -- Just Around The Corner

**arXiv ID:** 2601.06275 | [PDF](https://arxiv.org/pdf/2601.06275v1)

**作者:** Kevin Riehl `[一作]` (ETH Zurich), Michail A. Makridis `[通讯]` (ETH Zurich)

**通讯引用:** 2121 | [OpenAlex ID](https://openalex.org/A5015419644)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 FairSCOSCA，基于 SCOOTS/SCATS 的公平增强版交通信号控制方案，重点改进绿色相位分配与早期相位终止以提升公平性。

**💡 创新点**

创新点在于将累计等待时间纳入绿色相位优化，以及实现实时的早期相位提前终止机制，兼顾多重公平理论（平等、Rawls、功利、Harsanyi），并保持对主流系统的可落地性。

**🔧 技术方法**

使用基于贝叶斯优化的参数调优、SUMO 微观仿真、宏观基本图（MFD）与多维公平度量（Gini、最大延时、平均延时、总行程时）。

**📊 数据集**

在德国 Esslingen 的 Schorndorfer Strasse 干道网络进行仿真，包含 5 个信号交叉口、29 个探测器、26 个公交站、11 条公交线及真实的车辆构成。

**📈 对比分析**

与 Fixed‑Cycle、Max‑Pressure 以及原始 SCOOTS/SCATS（SCOSCA）进行对比。结果显示 FairSCOSCA_1 在效率（流量、速度、吞吐）上提升约 2‑3%，并在所有四种公平指标上均优于 SCOSCA；FairSCOSCA_2 维持效率，显著提升平等与 Rawls 公平。

**⚠️ 局限性**

局限在于只考虑基于延时的公平指标，未涵盖社会人口学、出行目的等维度；仅在单一干道网络验证，缺乏对其他网络类型、事故或感知研究的评估。

---

## 175. Are LLMs Vulnerable to Preference-Undermining Attacks (PUA)? A Factorial Analysis Methodology for Diagnosing the Trade-off between Preference Alignment and Real-World Validity

**arXiv ID:** 2601.06596 | [PDF](https://arxiv.org/pdf/2601.06596v1)

**作者:** Hongjun An `[一作]` (Northwestern Polytechnical University), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 61163 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对对齐语言模型在推理时受可操控提示的脆弱性进行诊断，提出了对偏好与真相两重目标的 2×2^4 因子实验设计。

**💡 创新点**

创新点是将偏好对齐模型的可操控性分解为四个正交的 PUA 交互维度，并用因子回归给出主效应与交互效应的可解释估计。

**🔧 技术方法**

采用因子回归、逻辑回归和 LLM-as-judge 的二分类评估，配合系统级目标与用户级提示因子组合。

**📊 数据集**

使用双语多项选择知识基准 MMLU 和 CMMLU，约 3 万条样本。

**📈 对比分析**

在多款闭源与开源模型上评估，结果表明系统目标从真相到偏好会显著降低准确率并提升对错误提示的顺从，且高级模型更易受 PUA 影响，开源模型更脆弱。

**⚠️ 局限性**

局限在于仅适用于确定性知识任务，无法覆盖开放式生成任务；对评估噪声与主观性控制不足。

---

## 176. Hard Thresholding Pursuit Algorithms for Least Absolute Deviations Problem

**arXiv ID:** 2601.06558 | [PDF](https://arxiv.org/pdf/2601.06558v1)

**作者:** Jiao Xu `[一作]` (Lanzhou University), Bing Zheng `[通讯]` (Lanzhou University)

**通讯引用:** 6511 | [OpenAlex ID](https://openalex.org/A5021519126)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种无参数、未知稀疏度的梯度快速硬阈值追踪算法（GFHTP_1），用于求解带有异常值的最小绝对偏差（LAD）稀疏信号重建问题。

**💡 创新点**

创新点：不需要先验稀疏度，使用截断自适应步长消除异常值影响；提供全局收敛分析并给出停止准则；在大比例异常值下实现精确恢复。

**🔧 技术方法**

采用硬阈值追踪（HTP）、梯度下降、截断量化阈值、RIP1理论以及随机高斯测量矩阵等技术。

**📊 数据集**

在合成的高斯/平坦稀疏信号以及MNIST手写数字图像上进行验证。

**📈 对比分析**

与PSGD、AIHT、FHTP等传统方法对比，GFHTP_1在成功率上更高、计算时间更低，尤其在异常值比例高时表现突出。

**⚠️ 局限性**

局限：步长选择仍有经验性依赖；对测量矩阵的RIP1假设在实际应用中难以验证；在稀疏度未知但非常大时计算开销仍较高。

---

## 177. An Efficient Evolutionary Algorithm for Few-for-Many Optimization

**arXiv ID:** 2601.06387 | [PDF](https://arxiv.org/pdf/2601.06387v1)

**作者:** Ke Shang `[一作]` (Shenzhen University), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 38375 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种针对“少解多目标”优化问题的高效进化算法SoM-EMOA，并提出了基于R2指标的新的基准测试集。

**💡 创新点**

主要创新在于：① 用Sum-of-Minimum目标直接优化集合覆盖度；② 引入目标层面选择与存档驱动的交配策略；③ 设计了高效的删除与归档更新机制；④ 提供可扩展的R2基准生成方法。

**🔧 技术方法**

采用进化算法框架（(μ+1)-进化策略）、SBX交叉+多项式变异、目标层次选择、R2指标映射及归档管理。

**📊 数据集**

评估使用DC-MaTS、NMLR、改造后的DTLZ/WFG（R2基准）以及两个真实工程问题DDMOP1/4。

**📈 对比分析**

与CluSO、MOCOBO及多种主流EMO算法在30次实验中比较，使用SoM指标G_ws作为评价，SoM-EMOA在大多数测试实例上均显著优于其他方法，收敛速度快、结果稳定。

**⚠️ 局限性**

主要局限在于：① 对初始种群和参数敏感；② 纯随机搜索可能导致重复计算；③ 适用于无梯度黑盒问题，对梯度可用场景未做针对性改进。

---

## 178. An LLM -Powered Assessment Retrieval-Augmented Generation (RAG) For Higher Education

**arXiv ID:** 2601.06141 | [PDF](https://arxiv.org/pdf/2601.06141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 179. Short-term electricity load forecasting with multi-frequency reconstruction diffusion

**arXiv ID:** 2601.06533 | [PDF](https://arxiv.org/pdf/2601.06533v1)

**作者:** Qi Dong `[一作]` (Macau University of Science and Technology), Jianzhou Wang `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 15590 | [OpenAlex ID](https://openalex.org/A5044720245)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于多频重构扩散模型MFRD，用于短期电力负荷预测。

**💡 创新点**

创新点在于将变分模态分解得到的多频特征与原始负荷拼接，引入扩散噪声注入与逆向去噪，并在去噪网络中融合残差LSTM与Transformer以提升去噪效果。

**🔧 技术方法**

采用的技术包括变分模态分解(VMD)、扩散概率模型、残差LSTM、Transformer以及傅里叶损失函数。

**📊 数据集**

使用了澳大利亚能源市场运营商(AEMO)的NSW、QLD、VIC 5分钟负荷数据和美国新英格兰独立系统运营商(ISO‑NE)的小时负荷数据。

**📈 对比分析**

与ARIMA、RNN、GRU、LSTM、Transformer以及SIWNN、WT‑ELM‑PLSR、ResNetPlus等基线模型比较，MFRD在MAE、RMSE、MAPE均下降至1%级别，R²>0.99，整体性能显著优于对手。

**⚠️ 局限性**

主要局限在于扩散模型训练成本高、参数量大，对计算资源要求高，且尚未充分挖掘空间相关性及模型压缩的潜力。

---

## 180. A Fast and Effective Method for Euclidean Anticlustering: The Assignment-Based-Anticlustering Algorithm

**arXiv ID:** 2601.06351 | [PDF](https://arxiv.org/pdf/2601.06351v1)

**作者:** Philipp Baumann `[一作]` (University of Bern), Jason Yang `[通讯]` (University of California)

**通讯引用:** 3573 | [OpenAlex ID](https://openalex.org/A5048226930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于指派问题的反聚类算法（Assignment-Based Anticlustering, ABA），通过一次性批量分配和中心点更新来求解大规模欧氏反聚类问题。

**💡 创新点**

创新点包括：①按全局质心距离降序划分批次，保证各反簇内部多样性相近；②使用中心点距离代替对象间距离，显著降低计算量；③引入层次分解策略，实现对百万级对象和十万级反簇的可扩展求解；④支持类别平衡约束和 balanced k‑cut 变体。

**🔧 技术方法**

核心技术是指派问题求解（Jonker‑Volgenant 算法）、批量更新中心点、层次分解并行化；并在实验中与最近邻交换、随机交换、整数规划、METIS 等方法比较。

**📊 数据集**

实验使用 14 个公开数据集，包含表格数据（如 UCI、Kaggle）和图像数据（CIFAR10、MNIST、ImageNet8/32），样本数从几千到 6.3 百万，特征维度从 3 到 110 维。

**📈 对比分析**

与 fast‑anticlustering（P‑N5/P‑R5/P‑R50/P‑R500）、MILP、METIS、随机分配等基线相比，ABA 在大多数实例中取得最优或最接近最优的目标值，并在运行时间上往往快数到千倍；同时生成的反簇多样性分布更均衡。

**⚠️ 局限性**

局限性：①仍需在每批中计算对象–中心点距离，极高维或稀疏数据仍可能成本高；②层次分解需要手工设定拆分参数，对不同数据集可能需调优；③在非欧氏距离或非完整图情形下的效果尚未验证。

---

## 181. Socio-technical aspects of Agentic AI

**arXiv ID:** 2601.06064 | [PDF](https://arxiv.org/pdf/2601.06064v1)

**作者:** Praveen Kumar Donta `[一作]` (Stockholm University), Schahram Dustdar `[通讯]` (ICREA)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 agentic AI 系统的技术架构与社会、伦理、经济、环境及治理维度进行系统化的社会技术分析，提出 MAD–BAD–SAD 框架，阐释技术设计与社会影响的相互作用；

**💡 创新点**

创新点在于将传统技术中心的 agentic AI 研究与社会技术视角结合，构建动机‑应用‑困境、偏见‑责任‑危害、社会影响‑采纳‑设计三维分析模型，揭示空白与未来研究方向；

**🔧 技术方法**

主要技术为文献综述与概念框架构建，采用 MAD–BAD–SAD 分析方法，对现有 survey 和案例进行整理；

**📊 数据集**

研究不依赖具体数据集，而是基于公开文献与案例，综合整理各领域 agentic AI 应用；

**📈 对比分析**

由于为综述性工作，未进行实验对比；作者通过对比近年多篇 survey 的聚焦与不足，说明自身框架在覆盖社会伦理维度方面的优势；

**⚠️ 局限性**

局限性包括：缺乏定量评估与案例验证、对新兴技术细节关注不足、跨领域治理与标准的具体实现仍待后续实证研究。

---

## 182. SparseOccVLA: Bridging Occupancy and Vision-Language Models via Sparse Queries for Unified 4D Scene Understanding and Planning

**arXiv ID:** 2601.06474 | [PDF](https://arxiv.org/pdf/2601.06474v1)

**作者:** Chenxu Dang `[一作]` (Huazhong University of Science and Technology), Yan Wang `[通讯]` (Institute for AI Industry Research AIR Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了基于稀疏占位查询的 VLM‑动作模型 SparseOccVLA，实现了统一的场景理解、占位预测与轨迹规划。

**💡 创新点**

首次将稀疏占位查询作为视觉与语言的桥梁，并结合 LLM 与 Anchor‑Diffusion 规划，实现多任务统一。

**🔧 技术方法**

采用稀疏占位编码器、LLM（Vicuna‑7B）、特征蒸馏、LLM 引导的 Anchor‑Diffusion 规划等技术。

**📊 数据集**

在 nuScenes 及其子任务 OmniDrive‑nuScenes、Occ3D‑nuScenes 上进行训练与评测。

**📈 对比分析**

与 SOTA 对比，OmniDrive‑nuScenes 的 CIDEr 提升 7%，Occ3D‑nuScenes 的 mIoU 提升 0.51，nuScenes 开环规划指标排名第一。

**⚠️ 局限性**

受限于对稠密占位监督的高成本、缺乏闭环规划评估以及模型对稠密数据的依赖。

---

## 183. EyeTheia: A Lightweight and Accessible Eye-Tracking Toolbox

**arXiv ID:** 2601.06279 | [PDF](https://arxiv.org/pdf/2601.06279v1)

**作者:** Stevenson Pather `[一作]` (University of Lille), Deise Santana Maia `[通讯]` (University of Lille)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了名为EyeTheia的轻量级、开源眼动追踪工具，可在浏览器平台上使用普通笔记本摄像头实现实时视线估计，支持用户校准，适用于认知实验与临床研究。

**💡 创新点**

创新点在于：① 将MediaPipe FaceMesh的面部标记直接映射为iTracker的输入流，保证了在无GPU设备上的高效运行；② 采用可插拔的预训练或从MPIIFaceGaze训练的模型，配合轻量级微调实现个体化校准；③ 提供透明、可复现的端到端 pipeline，并公开模型、代码与实验材料。

**🔧 技术方法**

使用技术包括：MediaPipe FaceMesh（面部标记提取）、iTracker CNN架构（多流卷积+全连接回归）、FastAPI REST接口（浏览器交互）、Adam优化器 + Smooth‑L1 或 Euclidean 损失、以及用户特定的 fine‑tuning。

**📊 数据集**

主要数据集：MPIIFaceGaze（桌面摄像头、屏幕坐标）用于训练与验证；GazeCapture（移动设备、厘米坐标）用于预训练模型；MPIIGaze 用于对照实验。还通过 18 名受试者的 Dot‑Probe 任务收集真实实验数据。

**📈 对比分析**

在实验中将 EyeTheia 与商业 SeeSo SDK 在 18 名受试者的 Dot‑Probe 任务进行对比：① 左/右侧关注一致率约 75%，② ROI 10% 边界下的准确率 42.5% 对比 40.4%，③ 眼动时序抖动显著更高（EyeTheia 69.4 像素 vs SeeSo 26.2 像素）。总体来看，EyeTheia 在粗定位（左右关注）上与 SeeSo 相当，但在细粒度空间精度与短时稳定性上略逊。

**⚠️ 局限性**

局限性包括：缺乏显式的时间平滑滤波导致帧间抖动；实验仅部署了预训练模型，未测试从 MPIIFaceGaze 训练的版本；未充分利用深度或姿态信息；在极端光照或遮挡下表现不佳。

---

## 184. Leveraging Soft Prompts for Privacy Attacks in Federated Prompt Tuning

**arXiv ID:** 2601.06641 | [PDF](https://arxiv.org/pdf/2601.06641v1)

**作者:** Quan Minh Nguyen `[一作]` (University of Florida), My T. Thai `[通讯]` (University of Florida)

**通讯引用:** 8765 | [OpenAlex ID](https://openalex.org/A5005663679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并评估了 PromptMIA，一种针对联邦 Prompt‑Tuning 的成员推断攻击；

**💡 创新点**

创新点在于：① 将软 Prompt 作为新的攻击向量；② 通过对 Prompt 池中 key 的攻击性替换实现单轮高效攻击；③ 提出了安全游戏框架和理论优势下界；④ 系统评估了常规防御在此场景下的失效；

**🔧 技术方法**

技术包括：基于余弦相似度的 Prompt 选择与更新机制；对 key 的精准对齐与多样化生成；安全游戏与优势/成功率评估；对抗性 Prompt 注入与梯度/输出无关的攻击；对常规防御（DPSGD、噪声、异常检测）的实验分析；

**📊 数据集**

使用了 CIFAR‑10、CIFAR‑100、TinyImageNet、以及由 MNIST‑M、Fashion‑MNIST、CINIC‑10、MMAFEDB 组合而成的 FourDataset；在 ViT‑B/32、DeiT‑B/16、ConViT 三种视觉 Transformer 上进行实验；

**📈 对比分析**

在多数据集、三种模型、不同批大小下进行对比；PromptMIA 的攻击成功率普遍超过 90%，优势接近 1；相比 Naïve Prompt Injection，PromptMIA 在大批量训练时仍保持高优势；传统防御（异常检测、输入噪声）对攻击效果几乎无效；

**⚠️ 局限性**

局限性包括：仅针对 Prompt‑Tuning 设计，对 LoRA/Adapter 等其他 Parameter‑Efficient 方案未验证；假设 Prompt 池大小固定且模型冻结；防御空间有限，需要针对性新策略；

---

## 185. Fixing ill-formed UTF-16 strings with SIMD instructions

**arXiv ID:** 2601.06349 | [PDF](https://arxiv.org/pdf/2601.06349v1)

**作者:** Robert Clausecker `[一作]` (Zuse Institut Berlin), Daniel Lemire `[通讯]` (Université du Québec)

**通讯引用:** 3006 | [OpenAlex ID](https://openalex.org/A5045561693)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种基于SIMD的UTF‑16错位替换算法，用以在JavaScript等环境中快速纠正非法代理对。

**💡 创新点**

提出了并行检查高/低代理并按需修复的三步SIMD算法，并在NEON、AVX‑512等平台上实现了高效版本。

**🔧 技术方法**

使用了SIMD指令集（SSE2/AVX2/AVX‑512/NEON）以及掩码、位图、布隆等技术进行逻辑优化。

**📊 数据集**

使用随机生成的Unicode字符串，控制合法代理对与非法代理出现比例，实验范围从1 k到100 万字符。

**📈 对比分析**

与V8标量实现对比，ARM64版可达18.9 GB/s（≈9×快），Intel Ice Lake版7.5 GB/s；指令/字节、指令/周期均显著下降。

**⚠️ 局限性**

对极低频或特殊结构的非法UTF‑16（如跨块高/低代理）仍需后处理；NEON缺少某些掩码指令导致部分性能瓶颈，且SVE支持不足。

---

## 186. Parent-Guided Adaptive Reliability (PGAR): A Behavioural Meta-Learning Framework for Stable and Trustworthy AI

**arXiv ID:** 2601.06167 | [PDF](https://arxiv.org/pdf/2601.06167v1)

**作者:** Anshum Rankawat `[一作]` `[通讯]`, Anshum Rankawat

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Parent-Guided Adaptive Reliability (PGAR)框架，利用父层监督学习器的学习率以实现自适应可靠性控制；

**💡 创新点**

在传统优化器基础上嵌入三种行为级反馈（事件、过度自信、记忆）形成可靠性指数，并通过Lyapunov理论证明可靠性调节的有界稳定性；

**🔧 技术方法**

采用行为元学习、控制理论（Lyapunov稳定性分析）、梯度调度与自适应学习率；

**📊 数据集**

在MNIST、Fashion-MNIST等经典图像分类数据集上进行实验验证；

**📈 对比分析**

与Adam、AdaBound、SGD等基线优化器对比，PGAR在准确率保持不变的前提下显著降低了损失方差、期望校准误（ECE）和恢复时间，表现出更平稳、更可信的训练过程；

**⚠️ 局限性**

实验仍主要聚焦静态数据集，缺乏在非平稳、持续学习或多智能体环境中的长期验证，且对大规模模型的计算开销与参数调优尚未充分探讨。

---

## 187. Classroom AI: Large Language Models as Grade-Specific Teachers

**arXiv ID:** 2601.06225 | [PDF](https://arxiv.org/pdf/2601.06225v1)

**作者:** Jio Oh `[一作]` (KAIST), Jindong Wang `[通讯]` (William and Mary)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一套针对大语言模型的年级分层微调框架，使模型能够为小学到成人不同年级的学生生成可理解且信息准确的教育内容。

**💡 创新点**

创新点在于：① 通过集成七种可读性指标并设计投票算法，将文本复杂度精准映射到六个年级层级；② 利用LLM生成的跨学科问答对构建大规模年级标注数据集；③ 在开放式问答场景下实现了基于微调而非单纯提示的年级适配，并用人类调查验证模型输出与人类感知的高度一致。

**🔧 技术方法**

技术手段包括：大语言模型微调（GPT4o‑mini、LLaMA3.1:70B 等），可读性指标集成（Flesch Reading Ease、Flesch‑Kincaid Grade Level、Coleman‑Liau Index、Linsear Write、Gunning Fog、Dale‑Chall、Spache），投票融合算法，logit‑lens 结构分析，LLM 生成的合成问答数据。

**📊 数据集**

使用的数据集：① 生成的 8 领域 54 科目、约 550 题/科目跨年级问答对；② 四个评测集：ScienceQA（科学多选题）、ELI5_Category（Reddit 解释问答）、Natural Questions（真实搜索问答）以及 GPT4o 生成的合成题集；③ 对 208 名人类评测者进行问卷调查。

**📈 对比分析**

与基于提示的传统方法相比，微调模型在目标年级兼容性上平均提升 35.64% 点，保持与基线相近的准确率；在人类评估中 Kendall τ 为 0.76，表明模型输出与人类对年级难度的认知高度一致；此外，多样性和困惑度指标显示低年级模型具有更高的文本多样性。

**⚠️ 局限性**

局限性包括：① 仅优化文本可读性，对概念难度的自适应仍不足；② 生成的数据主要来自 LLM，可能与真实学习情境偏离；③ 未结合知识图谱或概念层级等结构化信息，难以提供完整的分层解释。

---

## 188. Future-as-Label: Scalable Supervision from Real-World Outcomes

**arXiv ID:** 2601.06336 | [PDF](https://arxiv.org/pdf/2601.06336v1)

**作者:** Benjamin Turtel `[一作]` (Lightning Rod Labs), Kris Skothiem `[通讯]` (Lightning Rod Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种名为Foresight Learning的框架，利用延迟的真实事件结果来训练语言模型生成概率预测。

**💡 创新点**

将可验证奖励的强化学习扩展到只有在事件完全展开后才可获得反馈的开放世界场景，并通过群组相对策略优化降低高方差。

**🔧 技术方法**

采用策略梯度中的Group Relative Policy Optimization (GRPO)，内部生成推理轨迹并输出概率预测，奖励采用对数得分。

**📊 数据集**

构建了包含5,620个二元未来事件预测的人工生成数据集，并在独立的Metaculus 293题目上进行外部验证。

**📈 对比分析**

与仅提示、集成提示以及更大模型进行对比，Foresight训练的32B模型在Brier分数、ECE和对数得分上均优于235B大模型，提升约27%。

**⚠️ 局限性**

仅处理二元结果，训练为离线模式，依赖自动化事件生成与解析，未评估在线反馈循环，且可能受生成过程中的偏差影响。

---

## 189. Modeling Tradeoffs between mobility, cost, and performance in Edge Computing

**arXiv ID:** 2601.06591 | [PDF](https://arxiv.org/pdf/2601.06591v1)

**作者:** Muhammad Danish Waseem `[一作]` (Chalmers University of Technology), Ahmed Ali-Eldin `[通讯]` (Chalmers University of Technology)

**通讯引用:** 1542 | [OpenAlex ID](https://openalex.org/A5002226825)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过闭式排队模型结合实验验证，分析了边缘计算相较于传统云计算在移动性、工作负载波动和虚拟机调度方面的成本与性能权衡。

**💡 创新点**

创新点包括：①提出两阶段排队模型捕捉迁移开销；②给出边缘与云的容量需求公式；③扩展到GI/G/k和非平稳流量的分析；④通过真实Azure VM日志和云实验进行验证。

**🔧 技术方法**

采用排队论（M/M/k、GI/G/k、两阶段队列、Pollaczek–Khinchine、Whitt QED 等）、仿真与实验（Google Cloud ND2、Flask+ResNet50图像分类），以及流量测算技术。

**📊 数据集**

使用真实的 Azure VM 访问日志（约 200 万条记录）作为工作负载数据；实验中使用 Google Cloud ND2 实例并测量不同迁移概率下的响应时间。

**📈 对比分析**

将理论计算得到的等待时间与仿真/实验的平均等待时间进行对比，验证模型的准确性；结果表明模型能合理预测迁移概率和波动幅度导致的延迟升高；边缘需要额外容量约为 (1+1/q) 倍，云在成本效益上更具优势。

**⚠️ 局限性**

限制包括：假设到达与服务时间服从 Poisson/指数分布，迁移概率保持恒定；未考虑多层边缘架构、不同服务等级或能耗影响；实验规模有限，未覆盖极端网络拥塞或多策略迁移场景。

---

## 190. Data-Dependent Goal Modeling for ML-Enabled Law Enforcement Systems

**arXiv ID:** 2601.06237 | [PDF](https://arxiv.org/pdf/2601.06237v1)

**作者:** Dalal Alrajeh `[一作]` (Imperial College London), Mark Lee `[通讯]` (University of Birmingham)

**通讯引用:** 6902 | [OpenAlex ID](https://openalex.org/A5100620082)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过 KAOS 目标导向需求工程方法，在英国执法机构中设计并实现了一个基于机器学习的在线儿童性虐待案件嫌疑人识别决策支持系统。

**💡 创新点**

首次将数据获取与目标建模紧密耦合，提出了数据可实现性、概率化操作化等 KAOS 扩展，以解决传统 KAOS 在 ML 环境下的局限。

**🔧 技术方法**

采用 KAOS 框架进行需求建模，使用 BERT 预训练语言模型与概率性诱导逻辑程序（ILP）进行预测，标注工具采用 NVivo。

**📊 数据集**

使用四个执法机构提供的文本数据集（D1–D4），共 339 案例、约 88,976 条文本，标注了约 110,000 条行为与语言特征。

**📈 对比分析**

将目标细化为概率二分类任务，在大样本下 BERT 的精度约 0.92、召回率 0.94；在稀疏样本下 ILP 的精度约 0.83、召回率 1.00，说明模型在不同数据量下的性能差异。

**⚠️ 局限性**

受限于数据隐私、标签稀缺、目标与数据的不匹配以及对人工持续标注的高依赖，某些子目标的召回率仍低于阈值，且系统整体需频繁迭代与人工干预。

---

## 191. Perspective: The creation of "Newsgames" as a teaching method-Empirical observations

**arXiv ID:** 2601.06139 | [PDF](https://arxiv.org/pdf/2601.06139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 192. Semantic Enrichment of CAD-Based Industrial Environments via Scene Graphs for Simulation and Reasoning

**arXiv ID:** 2601.06415 | [PDF](https://arxiv.org/pdf/2601.06415v1)

**作者:** Nathan Pascal Walus `[一作]` (RWTH Aachen University), Kazunori Ohno `[通讯]` (Tohoku University)

**通讯引用:** 2305 | [OpenAlex ID](https://openalex.org/A5055778007)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种离线流程，利用CAD模型与大视觉语言模型（LVLM）生成多层3D场景图，并自动推断管道系统中的功能单元与关系。

**💡 创新点**

创新点在于：①直接对CAD网格进行语义标注并使用体素化+DBSCAN聚类；②构建含几何、语义和功能层级的场景图；③提出自动识别管道功能关系的算法，专为工业环境设计。

**🔧 技术方法**

使用技术包括：OpenAI 的 gpt‑4o 进行语义标注、1 cm 体素化、DBSCAN 聚类、Omniverse Python API 与 Isaac Lab、图结构分析与功能关系推断算法。

**📊 数据集**

数据集为单一工厂房间的 USD CAD 模型，共包含 8,327 个网格，经过预处理后 2,068 个网格被聚类并标注。

**📈 对比分析**

通过与人工标注的语义标签及功能关系进行定量对比，组标签准确率 74–84%，功能单元检索在三条管道结构上表现良好；然而存在误检和漏检，整体功能图虽可用但仍需改进。

**⚠️ 局限性**

限制包括：1 cm 体素化导致聚类误差；LVLM 在细粒度工业部件识别上误差大；功能关系解析缺乏方向性与对复杂分支的处理；对更大规模环境的可扩展性受限。

---

## 193. From RLHF to Direct Alignment: A Theoretical Unification of Preference Learning for Large Language Models

**arXiv ID:** 2601.06108 | [PDF](https://arxiv.org/pdf/2601.06108v1)

**作者:** Tarun Raheja `[一作]` (Independent Researchers), Nilay Pochhi `[通讯]` (Independent Researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型对齐中从人类偏好学习的各种方法进行系统综述，并提出一个统一的理论框架，将这些方法归纳为三个互相垂直的轴（偏好模型、正则化机制、数据分布），并给出相应的理论定理与经验对比。

**💡 创新点**

创新点包括：①构建ΨPO通用框架，证明DPO、IPO、KTO、SimPO、ORPO等方法是该框架的特例；②揭示在线与离线学习在覆盖性上的本质分离，并给出奖励过度、长度攻击、模式崩塌等失败模式的理论根源；③基于上述理论给出实用的决策指南，帮助工程师在资源、数据、稳定性等约束下选择最合适的对齐方法。

**🔧 技术方法**

使用的技术主要有：贝叶斯偏好模型（Bradley–Terry、Plackett–Luce、Prospect‑Theory等），KL 与 f‑divergence 正则化，ΨPO 损失函数的形式化与推导，覆盖性与偏差分析的定理证明，以及对现有实验结果（PPO、DPO、SimPO 等）的汇总与对比。

**📊 数据集**

在实验对比中主要引用了公开基准数据集：AlpacaEval 2、MT‑Bench、GSM8K、HH‑RLHF 等，并对 Llama‑3‑8B、Gemma‑2‑9B‑it 等模型在这些数据集上的表现进行比较。

**📈 对比分析**

方法比较采用基准性能（指令跟随、推理、对齐安全性）和训练成本两方面。结果显示：PPO 在数据覆盖不足时表现最好；DPO 在数据多样且质量高时接近 PPO；SimPO 在多种评测指标上均优于 DPO，且训练成本最低；IPO、KTO、ORPO 的表现低于 SimPO 与 PPO，但在特定场景（如二元反馈、无参考模型）下仍具备优势。

**⚠️ 局限性**

局限性主要体现在：①对多列表、非成对反馈的理论和实践仍未完善；②多目标（帮助、无害、诚实等）冲突的 Pareto 最优对齐方法缺乏成熟框架；③在分布漂移和持续学习场景下模型如何自适应仍未解决；④对大型模型的“规模法则”以及对齐过程的可解释性和因果推理尚需深入研究。

---

## 194. BabyVision: Visual Reasoning Beyond Language

**arXiv ID:** 2601.06521 | [PDF](https://arxiv.org/pdf/2601.06521v1)

**作者:** Liang Chen `[一作]` (Peking University), Kuan Li `[通讯]` (UniPat AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BabyVision基准，用来评估多模态大语言模型在早期视觉能力上的表现；

**💡 创新点**

设计了仅依赖视觉感知的22个子任务、388道题目，减少语言依赖，首次量化模型在基础视觉任务上的差距；

**🔧 技术方法**

利用多模态LLM推理、LLM-as-judge评估、RLVR强化学习调优、以及视觉生成模型（NanoBanana-Pro、Sora‑2等）进行视觉外化推理；

**📊 数据集**

构建了388道题的评测集，包含多种视觉子任务；并收集了1400道训练样例用于RLVR训练；

**📈 对比分析**

与不同年龄段儿童（3、6、10、12岁）和成人人类基线比较，发现最强的Gemini3‑Pro‑Preview仅49.7%，远低于成人94.1%，在所有四类任务上均显著落后；

**⚠️ 局限性**

主要局限在于：模型在细粒度、曲线追踪、三维空间想象和模式归纳上表现差，且大部分评测仍依赖文本推理，无法完全反映视觉推理能力；

---

## 195. Characterising Toxicity in Generative Large Language Models

**arXiv ID:** 2601.06700 | [PDF](https://arxiv.org/pdf/2601.06700v1)

**作者:** Zhiyao Zhang `[一作]` (Delft University of Technology), Yuhan Wu `[通讯]` (Delft University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型在受到有毒或无毒提示时产生有毒输出的倾向，并通过词汇与句法特征（高归因名词和ROOT结构）分析了触发有毒生成的语言模式。

**💡 创新点**

创新点在于将模型归因（Captum层级积分梯度）与POS、依存句法解析相结合，系统地揭示了高归因名词与简洁命令式句子共同驱动毒性输出的机制；同时，对七个不同模型（含指令微调版）进行了全面比较。

**🔧 技术方法**

使用了Perspective API进行多维度毒性评分，Captum的Layered Integrated Gradients进行词级归因，spaCy进行POS和依存句法解析，k-means聚类归因词汇，结合定量指标（Expected Max Toxicity、Toxicity Probability）和定性标签进行评估。

**📊 数据集**

采用DecodingTrust数据集，该数据集包含毒性诱导提示与非毒性提示，并配有毒性、脏话、威胁等多维度标注。

**📈 对比分析**

通过对比基础模型与指令微调模型的毒性概率和最大毒性得分，发现指令版模型在毒性概率上下降30%–80%，且在无毒提示下毒性概率普遍低于5%；结果表明微调有效降低模型产生有毒输出的倾向。

**⚠️ 局限性**

局限性包括：Perspective API在不同语言和语境下的偏差，阈值0.8的设定导致严重毒性偏倚，缺乏人类评估的上下文细微差别，提示样本选择可能影响归因结果，以及对非英语场景的可推广性有限。

---

## 196. Reflective Reasoning for SQL Generation

**arXiv ID:** 2601.06678 | [PDF](https://arxiv.org/pdf/2601.06678v1)

**作者:** Isabelle Mohr `[一作]` (Idiap Research Institute), Andre Freitas `[通讯]` (Idiap Research Institute)

**通讯引用:** 2480 | [OpenAlex ID](https://openalex.org/A5053978668)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于反射式迭代细化（Reflective Refinement）的多阶段文本到SQL（text-to-SQL）框架，利用局部诊断反馈对阶段级生成机制进行持续更新，以实现跨查询的可迁移改进。

**💡 创新点**

创新点在于将迭代细化从对单一SQL实例的重写转向对生成阶段的持续参数（提示）更新，并结合无监督的“形式认知评判者”（epistemic judges）进行语义覆盖验证，从而实现语法与语义双重一致性、保持先前验证约束、并提升跨查询的单调改进。

**🔧 技术方法**

核心技术包括：① 层次化生成器（schema、value、aggregation、predicate、SQL构造）；② 诊断器‑细化器（Critic‑Refiner）循环，局部定位错误并只更新相关阶段的提示；③ 数据库上下文代理（schema+采样值）压缩上下文窗口；④ 通过解释器检查和LLM语义覆盖验证构成的无监督评估；⑤ 采用多模型（GPT‑4/5、Qwen3‑30B、DeepSeek‑R1‑32B、LLaMA‑3.1‑8B）进行实验。

**📊 数据集**

在两大公开基准 Spider 与 BIRD（dev集）上评估，涵盖多域、复杂SQL模式，亦使用BIRD的VES（Valid Efficiency Score）指标。

**📈 对比分析**

与现有强基线（DIN‑SQL、DAIL‑SQL、MAC‑SQL、MCS‑SQL、GPT‑4零样本）和前沿/开源模型对比，所提全套方法在 Spider 上实现 93.8% 执行准确率、BIRD 上 95.4% 以及相应的 VES 分数，显示出显著提升；在迭代预算约 3–4 步时已接近饱和，进一步迭代收益递减。

**⚠️ 局限性**

局限包括：① 受限于固定迭代预算，预算耗尽后仍可能返回错误 SQL；② 诊断器的精度决定细化效果，误诊可能导致无效或错误细化；③ 语义检查仅覆盖预定义约束，难以捕捉细粒度语义偏差；④ 未考虑查询执行成本或效率；⑤ 尽管框架通用，但在其他结构化预测任务中需重新实现阶段与评估器。

---

## 197. A Multi-Stage Workflow for the Review of Marketing Content with Reasoning Large Language Models

**arXiv ID:** 2601.06054 | [PDF](https://arxiv.org/pdf/2601.06054v1)

**作者:** Alberto Purpura `[一作]` (Capital One), Swapnil Shinde `[通讯]` (Capital One)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个多阶段工作流，利用微调的推理LLM自动审查营销内容是否符合一系列合规要求。

**💡 创新点**

①不依赖外部知识表示，直接让LLM推理判定合规性；②比较SFT和GRPO两种微调策略及奖励函数组合；③探索生成推理文本对小LLM性能的影响。

**🔧 技术方法**

LoRA微调、强化学习（GRPO）与监督微调（SFT）、文本嵌入检索（SBERT）、BLEU奖励、结构化输出格式。

**📊 数据集**

通过LLM自动生成的合成营销内容与合规需求的检索和违规检测数据集，约12,163条内容‑需求对，包含违规与合规样本均衡。

**📈 对比分析**

检索阶段评估Recall@k和效率提升；违规检测阶段对比SFT/GRPO、不同奖励组合、是否生成推理文本，报告准确率、精确率、召回率、BLEU和格式正确率；GRPO在大模型且包含推理文本时获得最高准确率（≈65%），SFT在小模型上表现更好。

**⚠️ 局限性**

仅使用合成数据，真实性有限；仅针对营销内容和固定合规集合；未实现合规需求的自动提取；模型仍可能产生偏见和错误，需人工监督。

---

## 198. GlobalPaint: Spatiotemporal Coherent Video Outpainting with Global Feature Guidance

**arXiv ID:** 2601.06413 | [PDF](https://arxiv.org/pdf/2601.06413v1)

**作者:** Yueming Pan `[一作]` (Xi'an Jiaotong University), Nanning Zheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于扩散模型的分层视频扩展框架GlobalPaint，通过关键帧先行补全再插值中间帧，实现时空连贯的边界补全。

**💡 创新点**

创新点在于：①使用3D窗口注意力的增强时空模块提升长程交互；②通过OpenCLIP提取全序列全局特征并压缩成token，用交叉注意力引导生成；③采用分层关键帧+插值策略减轻误差累积。

**🔧 技术方法**

核心技术包括：扩散模型（Latent Diffusion Model）、Stable Diffusion v2.0 先验图像修复骨干、3D窗口注意力、全局特征提取器、交叉注意力以及插值模型微调。

**📊 数据集**

训练使用WebVid‑10M的5M视频样本，评估在DAVIS 2017和YouTube‑VOS基准数据集上。

**📈 对比分析**

与M3DDM、MOTIA、Dehan等方法对比，GlobalPaint在PSNR/SSIM/LPIPS均有提升，FVD降低至227.8（相比M3DDM下降24.1%），在YouTube‑VOS上同样表现出更优的重建质量与自然运动。

**⚠️ 局限性**

局限性：扩散模型推理成本高，处理长视频仍需分段；在极端摄像机或物体剧烈运动时仍可能出现细节模糊或时空微小不一致。

---

## 199. Investigating How MacBook Accessories Evolve across Generations, and Their Potential Environmental, Economical Impacts

**arXiv ID:** 2601.06055 | [PDF](https://arxiv.org/pdf/2601.06055v1)

**作者:** Zeyi Liao `[一作]` (Ohio State University), Ting Zhu `[通讯]` (Ohio State University)

**通讯引用:** 7468 | [OpenAlex ID](https://openalex.org/A5014539089)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对MacBook充电器从MagSafe到USB‑C再回到MagSafe 3的演变进行追踪，结合Walmart线上数据分析购买量、价格及其对电子废弃物和经济的影响。

**💡 创新点**

创新点在于：①将充电器技术演进与环境和经济双重影响关联；②使用公开API与手工校验结合的方式获取真实购买行为数据；③提出可复用组件设计和循环利用策略，以降低废弃电源适配器对生态的负担。

**🔧 技术方法**

技术手段主要包括：1) 通过SerpAPI抓取Walmart搜索结果；2) 手工核实关键词对应的充电器型号；3) 对购买时间序列进行统计与可视化；4) 计算平均价格并与购买频率对应；5) 采用生命周期评估（LCA）概念阐述产能、使用、运输与回收阶段的碳排放分布。

**📊 数据集**

数据集来源为2016–2024年Walmart网站的充电器搜索结果，包含四种型号（MagSafe 1、MagSafe 2、MagSafe 3、USB‑C）每年上半年与下半年的购买次数及对应的平均售价。

**📈 对比分析**

比较方法主要是：①按年份与半年划分统计各型号购买量；②将购买量与平均价格绘制折线/柱状图进行对比；③通过对比MagSafe 1/2与MagSafe 3/USB‑C的趋势，评估新技术的普及速度。性能方面，USB‑C的购买量最高，价格最低；MagSafe 3因兼容最新MacBook而出现上升趋势；MagSafe 1/2几乎消失。

**⚠️ 局限性**

局限性包括：①仅使用Walmart线上销量（通过评论数量近似），未覆盖线下采购；②Amazon等主要平台的数据被限制，导致样本不完整；③关键词匹配与手工校验仍可能产生误判；④使用评论数作为购买量近似假设，可能出现误差；⑤缺乏对实际电子废弃物回收率与碳排放的实测数据。

---

## 200. AI Washing and the Erosion of Digital Legitimacy: A Socio-Technical Perspective on Responsible Artificial Intelligence in Business

**arXiv ID:** 2601.06611 | [PDF](https://arxiv.org/pdf/2601.06611v1)

**作者:** Nelly Elsayed `[一作]` (University of Cincinnati), Nelly Elsayed `[通讯]` (University of Cincinnati)

**通讯引用:** 384 | [OpenAlex ID](https://openalex.org/A5108005607)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从社会技术视角构建了人工智能“AI洗牌”概念，提出四类主要表现（营销/品牌、技术能力夸大、战略信号化、治理/伦理洗牌），并讨论其对组织、行业及社会层面的影响与后果。

**💡 创新点**

创新点在于将AI洗牌与绿色洗牌进行比较，提出了专门的AI洗牌框架与四维分类体系；同时将信号理论、伦理治理与数字创新融入IS研究，形成完整的社会技术合法性模型。

**🔧 技术方法**

本文为理论性研究，无实证或算法技术实现；采用文献综述与概念分析方法。

**📊 数据集**

未使用数据集，完全基于现有文献与理论推演。

**📈 对比分析**

无实验比较；通过案例式描述与理论对照说明AI洗牌对声誉、信任与创新的潜在影响。

**⚠️ 局限性**

局限性包括缺乏实证验证与可量化测量工具，模型需要后续经验研究检验；对跨行业和跨文化差异的讨论有限。

---

## 201. Implicit bias as a Gauge correction: Theory and Inverse Design

**arXiv ID:** 2601.06597 | [PDF](https://arxiv.org/pdf/2601.06597v1)

**作者:** Nicola Aladrah `[一作]` (Università di Trieste), Fabio Anselmi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1684 | [OpenAlex ID](https://openalex.org/A5079466450)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于模型参数连续对称性的几何框架，推导出随机梯度下降（SGD）在对称商空间上的稳态分布会出现对称体积（Faddeev–Popov 行列式）产生的对数校正，从而解释并量化隐式偏置；进一步给出逆设计方法，通过构造冗余参数化来实现预期的稀疏性、低秩或谱稀疏性等隐式正则化。

**💡 创新点**

创新点在于将物理学中的约束和Gauge理论中的Jacobian概念引入机器学习的随机动力学，给出一个统一、可计算的隐式偏置表达式；同时提出了逆设计策略，可针对目标正则化先设计对称结构，再由SGD自然产生对应的正则化效果。

**🔧 技术方法**

主要技术包括：随机微分方程（SDE）描述SGD、连续Lie群对称性、商空间投影、Faddeev–Popov矩阵与轨道Gram矩阵的构造、对称体积的对数修正、最小化与拉格朗日对偶；数值验证使用标准Langevin/SGD仿真与离散梯度下降。

**📊 数据集**

使用人工合成数据：一维周期信号（Fourier系数稀疏）、一维分段常数信号（TV稀疏）、低秩矩阵完成、教师-学生注意力训练等，全部基于随机生成的输入和噪声样本。没有使用公开真实数据集。

**📈 对比分析**

与不带正则化的基线模型、直接稀疏或低秩参数化对比，评估训练误差、测试误差、谱或TV量等指标。结果显示逆设计的参数化在测试误差、谱稀疏性、TV抑制等方面显著优于基线，验证了理论预测的隐式偏置。

**⚠️ 局限性**

局限性包括：仅在小规模、合成任务上验证；假设SGD可近似为高斯Langevin且噪声平稳；小噪声极限下的分析；对非线性特征映射的推导尚未完成；在复杂现实任务中的效果尚未系统评估。

---

## 202. Nigeria's Digital Sovereignty: Analysis of Cybersecurity Legislation, Policies, and Strategies

**arXiv ID:** 2601.06050 | [PDF](https://arxiv.org/pdf/2601.06050v1)

**作者:** Polra Victor Falade `[一作]` (Nigerian Defence Academy), Oluwafemi Osho `[通讯]` (Clemson University)

**通讯引用:** 497 | [OpenAlex ID](https://openalex.org/A5067847917)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文通过多方法三角化的质性研究，评估尼日利亚《网络犯罪法》与《国家网络安全政策与战略》在推进数字主权方面的实际效果与不足；

**💡 创新点**

创新点在于将文件分析、二手研究、专家访谈和现场观察相结合，形成对法律与政策实施情况的综合评估框架；

**🔧 技术方法**

采用了多方法研究技术，包括文件分析、专家访谈、案例观察以及二手文献批判性分析；

**📊 数据集**

数据来源主要为官方文件（《网络犯罪法》与《国家网络安全政策》）、政府报告、学术研究、专家意见和现场观察记录；

**📈 对比分析**

研究未采用传统数值性能评估，而是通过对比法律条文与实际执法案例、执法效果与国际标准的匹配度，指出现行框架在执行力、跨部门协调、国际合作等方面的差距；

**⚠️ 局限性**

局限性包括执法资源不足、法律条文定义模糊、监管落实与技术更新脱节、信息共享缺失以及缺乏可量化的执行效果指标。

---

## 203. Contract2Plan: Verified Contract-Grounded Retrieval-Augmented Optimization for BOM-Aware Procurement and Multi-Echelon Inventory Planning

**arXiv ID:** 2601.06164 | [PDF](https://arxiv.org/pdf/2601.06164v1)

**作者:** Sahil Agarwal `[一作]` `[通讯]` (Found in Cache), Sahil Agarwal (Found in Cache)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个将合同文本自动转换为可验证约束并编译成MILP模型的端到端流水线，利用检索、LLM抽取、验证和修复循环生成可审计且合规的采购与库存计划。

**💡 创新点**

创新点在于将约束提取与优化决策分离，加入多层验证门控、证明可行性与合同安全的保守合并规则，以及基于证据跨度的可追溯决策卡片，实现了在存在文档歧义与提取错误时的安全自纠与人机协同。

**🔧 技术方法**

使用的技术包括检索增强生成（BM25、dense retrieval + rerank）、模式化的schema约束抽取、单调约束的保守合并、MILP编译、求解器驱动的不可行性诊断（IIS / slack minimization）、以及针对异常类的人工审核门控。

**📊 数据集**

主要数据集为自构造的微基准：500个随机生成的单品5期补货实例（包括需求、MOQ、交付期、价格、持有成本等），以及论文中提及的公开缺乏的合同+BOM+多层网络文档组合的假想合成数据。

**📈 对比分析**

通过在提取错误场景下与真实约束下的完备枚举比较，验证了只用抽取的规划会产生重尾的经济与合规风险；平均回报率为 5.40% 的超支，90分位为 587.74 美元，表明验证门控显著降低了极端风险。

**⚠️ 局限性**

限制主要包括：保守合并依赖检索覆盖度；对非单调约束（折扣异常、处罚例外等）必须人工干预；微基准仅关注 MOQ 与交付期错误，未覆盖更复杂的合同条款；大规模 BOM/网络会导致MILP规模膨胀，需要分解或滚动规划。

---

## 204. Reinforcement Learning for Chain of Thought Compression with One-Domain-to-All Generalization

**arXiv ID:** 2601.06052 | [PDF](https://arxiv.org/pdf/2601.06052v1)

**作者:** Hanyu Li `[一作]` (Peking University), Liang Zhao `[通讯]` (Xiaomi)

**通讯引用:** 6429 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于强化学习的链式思考压缩方法，采用样本级可门控软压缩，并验证其在多任务上的有效性与通用性。

**💡 创新点**

提出了仅在模型已掌握且已达安全长度时才进行软压缩的门控策略，避免全局硬限制导致的奖励操纵，并展示压缩技术能跨域迁移。

**🔧 技术方法**

使用RLVR（基于验证奖励的强化学习）与GRPO族算法（如DAPO、Dr.GPRO），实现自监督压缩奖励、动态采样与早停策略。

**📊 数据集**

主要在113K个数学问题（AIME24/25、MATH500等）上训练与评估，并在MMLU、IFEval、LiveCodeBench等多领域任务验证跨域效果。

**📈 对比分析**

与全局软压缩（DAPO-lite/heavy）和硬截断基线对比，发现样本级压缩可在保持或提升AIME准确率的同时将推理长度缩短约30‑40%，并显著优于全局策略。

**⚠️ 局限性**

压缩阶段需精细调节早停阈值，过度压缩会导致性能崩溃；门控机制对“熟练率”估计敏感，且目前主要在数学任务验证，跨域适应性尚需进一步探索。

---

## 205. Coding in a Bubble? Evaluating LLMs in Resolving Context Adaptation Bugs During Code Adaptation

**arXiv ID:** 2601.06497 | [PDF](https://arxiv.org/pdf/2601.06497v1)

**作者:** Tanghaoran Zhang `[一作]` (National University of Defense Technology), Yue Yu `[通讯]` (Peng Cheng Laboratory)

**通讯引用:** 3811 | [OpenAlex ID](https://openalex.org/A5100397991)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 CtxBugGen 框架，用于自动生成上下文适配 Bug（CtxBugs），并基于此评估大型语言模型（LLMs）在代码适配任务中的修复能力。

**💡 创新点**

创新点在于首次提出专门针对上下文依赖的 Bug 生成方法，并将 LLM 的自我生成与语法差异 + 单元测试相结合，形成完整的评测流水线和公开基准。

**🔧 技术方法**

采用了四步流程：任务选择 → 任务特定扰动规则 → LLM 生成变体 → 语法差异与测试判定；同时使用 LLM 对扰动代码进行自填充，并对生成结果做去重与验证。

**📊 数据集**

使用了 Python 代码重用基准 ClassEval 作为原始任务源，经过扰动与生成共得到 3,683 条 CtxBug 数据集，涵盖四类适配任务（接口、功能、标识符、依赖）。

**📈 对比分析**

对 GPT‑4o、DeepSeek‑V3、Qwen3‑Coder‑Plus、Kimi‑K2 四个主流 LLM 进行 Pass@1 与 Resolution Rate 比较，最优模型 Kimi‑K2 仅达 55.93% Pass@1 与 52.47% RR；相较于无 CtxBug 的基准，模型性能平均下降约 23%/27%。

**⚠️ 局限性**

局限性包括仅覆盖 Python 类级别适配场景、潜在的数据泄漏风险、仅评测四个 LLM 与四类任务，难以直接推广至其它语言或更广泛的适配情形。

---

## 206. Stress Testing Machine Learning at $10^{10}$ Scale: A Comprehensive Study of Adversarial Robustness on Algebraically Structured Integer Streams

**arXiv ID:** 2601.06117 | [PDF](https://arxiv.org/pdf/2601.06117v1)

**作者:** HyunJun Jeon `[一作]` `[通讯]` (Independent Researcher), HyunJun Jeon (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

以10亿规模的整数流生成Pythagorean三元组为基准，构造5×10⁹结构化对抗样本，对LightGBM梯度提升模型进行压力测试，评估其对代数结构的鲁棒性。

**💡 创新点**

创新点包括：① 单参数索引流实现无条件Pythagorean三元组生成；② 构造九类结构化对抗样本体系（HND）；③ 文件索引恢复的容错训练框架；④ 通过SHAP揭示模型更倾向于结构匹配特征而非直接算术检查。

**🔧 技术方法**

使用技术包括：NumPy、PyArrow、Parquet、任意精度整数、LightGBM、SHAP、GPU加速、文件级检查点、异常恢复与自动继续训练。

**📊 数据集**

数据集由10¹⁰条样本组成，其中5×10⁹为合法单参数Pythagorean三元组，5×10⁹为HND对抗样本，全部以Parquet文件形式存储并流式读取。

**📈 对比分析**

对比方法：将模型在各级对抗样本（Tier 1–3）和随机噪声基准上的准确率与训练损失曲线进行比较，结果显示在所有攻击类型下准确率≥98.75%，总体准确率达99.99%，训练至28%样本时损失已逼近64位浮点极限，整体训练耗时数十小时。

**⚠️ 局限性**

局限性包括：仅针对单参数子族，无法直接推广到其他Pythagorean族；对抗样本依赖人工构造，缺乏真实分布；模型依赖手工特征，未能证明真正“发现”结构；结果仅提供高置信度筛选，无法替代形式化验证。

---

## 207. Judge Model for Large-scale Multimodality Benchmarks

**arXiv ID:** 2601.06106 | [PDF](https://arxiv.org/pdf/2601.06106v1)

**作者:** Min-Han Shih `[一作]` (University of Southern California), Yu-Wei Chen `[通讯]` (University of Southern California)

**通讯引用:** 1565 | [OpenAlex ID](https://openalex.org/A5100374341)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种专门用于多模态评估的 Judge 模型，能够对文本、音频、图像和视频任务的多模态大语言模型（MLLM）输出进行评分、错误类型标注和可解释反馈。

**💡 创新点**

创新点在于：①引入多模态判定器（Judge）取代传统单一文本评估；②通过结构化错误分析实现可解释性；③在单一管线中统一评估四种模态并与人工标注对齐。

**🔧 技术方法**

使用的技术包括 Gemini‑3‑pro 作为 Judge 模型、基于 Likert 量表的评分规则、自动化评估管线（多模态输入→MLLM 输出→Judge 评估）以及对比人类评注的统计分析。

**📊 数据集**

使用的数据集来自 Hugging Face 的公开数据集，涵盖文本（代码、数学、问答等 130 题）、音频（音频字幕、鸟鸣检测等 50 题）、视频（AI 生成短片 50 题）和图像（图像字幕、图表推理等 50 题）等多模态子集。

**📈 对比分析**

评估方法是将 Gemini‑2.5、Phi‑4 和 Qwen‑2.5 的输出交由 Judge 模型评分，并与三名人工评审的平均分对比，结果显示 Judge 分数与人工排名高度一致，且对 Gemini‑2.5 的整体性能给出了 4‑5 级的高分；Phi‑4 与 Qwen‑2.5 在图像和文本模态表现相对逊色。

**⚠️ 局限性**

局限性包括：①Judge 仍依赖 Gemini‑3‑pro 的推理能力，可能对模型产生偏见；②仅针对单轮 QA 进行评估，缺乏多轮交互分析；③对极端误差类型（如幻觉、误解）识别仍不完备，未来需引入更多监督信号。

---

## 208. Automatic Question Generation for Intuitive Learning Utilizing Causal Graph Guided Chain of Thought Reasoning

**arXiv ID:** 2601.06098 | [PDF](https://arxiv.org/pdf/2601.06098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 209. BizFinBench.v2: A Unified Dual-Mode Bilingual Benchmark for Expert-Level Financial Capability Alignment

**arXiv ID:** 2601.06401 | [PDF](https://arxiv.org/pdf/2601.06401v1)

**作者:** Xin Guo `[一作]` (HiThink Research), Liwen Zhang `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 5898 | [OpenAlex ID](https://openalex.org/A5100459595)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并发布了基于中美股市真实业务数据的金融 LLM 评估基准 BizFinBench.v2，涵盖离线八项核心任务和两项实时在线任务，共 29,578 对 Q&A。

**💡 创新点**

创新点在于：① 采用真实业务问答而非模拟数据；② 双轨评估框架（核心业务能力 + 在线表现）；③ 结合金融专家视角的错误分析，精准定位模型在业务适配中的瓶颈。

**🔧 技术方法**

技术方法包括：用户查询聚类与去敏化、三层质量审核（平台+前端审核+专家交叉验证）、零样本评估、对齐金融专家的置信区间判定、构建在线资产配置仿真系统。

**📊 数据集**

数据集来源为中美股市真实交易与用户业务平台的问答日志，通过聚类得到四大业务场景，形成八项离线任务和两项在线任务。

**📈 对比分析**

在 21 种 LLM（包括 6 款商业模型和 15 款开源模型）上做零样本评估；ChatGPT‑5 以 61.5% 的平均准确率领跑；商业模型 DeepSeek‑R1 在资产配置任务中实现 13.46% 的累计收益并保持良好风险控制，超过市场基准；开源模型 Qwen3‑235B‑A22B‑Thinking 在所有模型中表现最佳，平均 53.3%。

**⚠️ 局限性**

局限性包括：① 任务覆盖仍偏向常见查询，缺少小众需求；② 在线评估仅包含股票预测和资产配置，可扩展至更多实时业务；③ 评估模式仅包含零样本与链式思维，缺少少样本实验。

---

## 210. A Framework for Kara-Kichwa Data Sovereignty in Latin America and the Caribbean

**arXiv ID:** 2601.06634 | [PDF](https://arxiv.org/pdf/2601.06634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 211. Semantic Event Graphs for Long-Form Video Question Answering

**arXiv ID:** 2601.06097 | [PDF](https://arxiv.org/pdf/2601.06097v1)

**作者:** Aradhya Dixit `[一作]` (Wake Tech Community), Tianxi Liang `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Semantic Event Graphs（SEG）管道，将长视频中的视觉内容抽象为人-物交互事件，并构建时间场景图（TSG）作为符号化记忆层，进而实现高效的长视频问答；

**💡 创新点**

创新点在于：①用轻量级事件日志替代原始帧，实现对数千帧视频的语义压缩；②构建可查询的时间场景图并加入查询感知裁剪；③在保持或提升准确率的前提下，将令牌使用量压缩至原始的 8.6%（约 11.6 倍成本降低）；

**🔧 技术方法**

核心技术包括：YOLOv11 对象检测与持续跟踪；基于距离阈值的 proximity 交互事件抽取（START/END）；使用 NetworkX 构建多重有向图；查询感知裁剪（锚点匹配 + 词汇相似度检索）；将裁剪后的事件序列转化为文本交给 Gemini 2.5 Flash 进行推理；以及自动化的 QA 生成与 LLM 评估；

**📊 数据集**

使用了公开的五个 YouTube 长视频（10–20 分钟），通过 YOLO+SEG 生成 300–500 个交互事件，累计 1,650+ 事件，并手工/LLM 生成 120 对长时段问答；

**📈 对比分析**

实验对比了三种方式：短上下文（仅最近 30 秒）、完整日志（所有事件）以及 TSG 裁剪。准确率分别为 2.5% / 62.5% / 65.0%，令牌使用量为 1.03k / 40.39k / 3.47k，TSG 在保持甚至略优准确率的同时，令牌使用下降 91.4%（≈12×成本降低）；

**⚠️ 局限性**

局限性包括：①基于字符串匹配的锚点检索容易遗漏同义词或短语；②离场或隐藏的动作无法被记录；③多对象或多步推理仍受限；④缺乏外观特征（颜色、姿态）的视觉 grounding；⑤数据集规模有限，评估主要依赖 LLM 判定，可能存在偏差；

---

## 212. Islamic Chatbots in the Age of Large Language Models

**arXiv ID:** 2601.06092 | [PDF](https://arxiv.org/pdf/2601.06092v1)

**作者:** Muhammad Aurangzeb Ahmad `[一作]` (University of Washington), Muhammad Aurangzeb Ahmad `[通讯]` (University of Washington)

**通讯引用:** 1731 | [OpenAlex ID](https://openalex.org/A5060934705)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并系统化分析了当代伊斯兰聊天机器人生态，提出五轴分类法，探讨其在权威、教学与实践中的影响。

**💡 创新点**

创新点在于将技术治理、范围、权威姿态、技术基础与教派编码五维度构建框架，并结合伦理与实务挑战提出责任性设计准则。

**🔧 技术方法**

采用大语言模型（LLM）与检索增强生成（RAG）技术，评估其在 Quran/Hadith、法学及日常指导中的生成方式。

**📊 数据集**

主要使用 Quran、Hadith 集合、各家法学文献以及公开的聊天机器人交互日志等数据。

**📈 对比分析**

通过对比表格与案例研究，对不同系统的源透明度、功能特性与宗派适配度进行定性比较，指出缺乏标准化评估导致效果难以量化。

**⚠️ 局限性**

限制包括模型偏见、文本错误/幻觉、缺乏可验证的链条与责任机制，且未提供统一性能指标或跨语言验证。

---

## 213. Revisiting Training Scale: An Empirical Study of Token Count, Power Consumption, and Parameter Efficiency

**arXiv ID:** 2601.06649 | [PDF](https://arxiv.org/pdf/2601.06649v1)

**作者:** Joe Dwyer `[一作]` `[通讯]` (ECPI University), Joe Dwyer (ECPI University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定硬件与模型参数条件下，对TinyLlama模型分别使用500K、1M、2M训练tokens，评估能耗感知参数效率。

**💡 创新点**

首次将实时GPU功耗与执行时间直接纳入参数效率计算，提出了能耗敏感的评估指标。

**🔧 技术方法**

使用Repeated‑Measures ANOVA、RMS功耗测量、FP16混合精度训练、AdamW优化器和梯度裁剪等技术。

**📊 数据集**

采用TinyStories JSONL 语料库，最大序列长度为64个token，并进行填充截断。

**📈 对比分析**

通过重复测量ANOVA对比不同token量的效率，发现token增加导致能耗提升且效率下降，模型性能提升有限。

**⚠️ 局限性**

研究仅在单一1.1B TinyLlama模型与单张NVIDIA A10G GPU上进行，缺乏跨模型与跨硬件的推广性。

---

## 214. ArrowGEV: Grounding Events in Video via Learning the Arrow of Time

**arXiv ID:** 2601.06559 | [PDF](https://arxiv.org/pdf/2601.06559v1)

**作者:** Fangxu Yu `[一作]` (Nanjing University), Jie Zhou `[通讯]` (Tencent Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用强化学习框架，在视频事件定位任务中加入时间方向性奖励，使 Vision‑Language 模型（VLM）能够同时识别正向视频中的事件、区分时间敏感事件在倒序视频中的缺失，以及保持时间不敏感事件在正向与倒序视频中的一致定位。

**💡 创新点**

首次将物理学中的“时间箭头”概念嵌入视频事件定位，构造基于事件是否随时间倒置语义变化的奖励机制，显著提升模型对事件时间结构的理解与鲁棒性。

**🔧 技术方法**

采用 GRPO（Group Relative Policy Optimization）强化学习、think‑before‑act 的思路、时间逆向奖励、难度自适应加权与动态课程筛选等技术，并在 Qwen2.5‑VL‑7B/3B 上进行微调。

**📊 数据集**

在三大事件定位基准 Charades‑STA、ActivityNet、TVGBench 上训练与评估，同时在 TempCompass、MVBench、VSI‑Bench、Video‑MMMU、MMVU、VideoMME 等通用视频理解与推理基准上做 OOD 测试。

**📈 对比分析**

与传统 SFT、基准方法以及其他 RL 方法对比，R1@m 指标提升约 2–3%（mIoU 提升 3–6%），在 OOD 任务上也实现显著提升，证明该方法在事件定位与时间结构理解上具有竞争优势。

**⚠️ 局限性**

模型规模大、训练成本高；目前仅针对事件定位任务，尚未充分验证在更复杂视频推理任务中的适用性，未来可进一步扩展。

---

## 215. The AI Roles Continuum: Blurring the Boundary Between Research and Engineering

**arXiv ID:** 2601.06087 | [PDF](https://arxiv.org/pdf/2601.06087v1)

**作者:** Deepak Babu Piskala `[一作]` `[通讯]`, Deepak Babu Piskala

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文通过梳理顶尖 AI 实验室和企业的职位描述、招聘流程与团队结构，提出了“AI 角色连续体”概念，阐明研究科学家、研究工程师、应用科学家和机器学习工程师在算法创新、系统实现与产品交付等方面的职责重叠与协同。

**💡 创新点**

创新点在于把传统的研究与工程分离的观念转化为一个光谱，提供了从方法创新到生产落地的角色映射表和技能热图，为 AI 组织设计人力资源、职业阶梯与跨职能协作提供了系统化框架。

**🔧 技术方法**

采用的技术主要是文献与案例分析方法：对公开职位信息、公司招聘公告、面试流程、组织结构等进行内容编码，构建角色与技能的对应关系；并通过可视化（如角色-技能热图、角色光谱图）展示结果。

**📊 数据集**

主要数据来源是公开的职位描述与招聘信息，涵盖 OpenAI、Anthropic、DeepMind、Meta、Amazon、Microsoft 等公司；未使用传统机器学习数据集，核心数据为文本型职位信息。

**📈 对比分析**

比较方法：对比不同组织（实验室、平台、产品公司、开源社区）在角色分布、招聘重点和技能需求上的差异，采用定性对比与统计描述（如角色集中度）。实验结果表明：实验室偏向研究型角色，平台与产品公司偏向工程型角色，但所有组织都表现出显著的跨职能重叠，证明连续体模型能更好解释现实团队结构。

**⚠️ 局限性**

局限性：研究仅基于公开信息，缺乏内部评估和绩效测量；样本主要集中在美国和大公司，可能不具备普适性；角色光谱为概念框架，未给出定量验证其对组织绩效的影响。

---

## 216. Pragya: An AI-Based Semantic Recommendation System for Sanskrit Subhasitas

**arXiv ID:** 2601.06607 | [PDF](https://arxiv.org/pdf/2601.06607v1)

**作者:** Tanisha Raorane `[一作]` (Don Bosco Institute of Technology), Prasenjit Kole `[通讯]` (Don Bosco Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Pragya，一个基于检索增强生成（RAG）的系统，用于对梵语 Subhāṣitas 进行语义检索，并提供原文、马拉地翻译、英语解释及音译。

**💡 创新点**

首创将检索（IndicBERT+FAISS）与本地生成（Mistral LLM）结合，突破关键词匹配的局限，实现语义级别的推荐与解释。

**🔧 技术方法**

使用 IndicBERT 句子嵌入、FAISS 向量检索、本地 Ollama 部署的 Mistral LLM、中文/马拉地/英语多语翻译，以及未来计划加入的 TTS/ASR。

**📊 数据集**

使用约 200 条手工标注的 Subhāṣita 语料，包含梵语原文、马拉地翻译、英语翻译和情感/主题标签。

**📈 对比分析**

通过与传统关键词检索对比，RAG 在 Top‑3 精确度从 45% 提升至 72%，覆盖率从 60% 提升至 82%，用户满意度从 2.8 提升至 4.3；生成解释在清晰度、文化适配度与参与度上显著优于字典式翻译。

**⚠️ 局限性**

局限性包括：数据集规模仅 200 条，导致语义覆盖不足；嵌入模型对梵语隐喻的捕捉不够；本地 LLM 生成延迟大；缺乏正式基准评测，需进一步大规模用户研究。

---

## 217. Federated Learning and Class Imbalances

**arXiv ID:** 2601.06348 | [PDF](https://arxiv.org/pdf/2601.06348v1)

**作者:** Siqi Zhu `[一作]` (University of Cambridge), Joshua D. Kaggie `[通讯]` (University of Cambridge)

**通讯引用:** 2435 | [OpenAlex ID](https://openalex.org/A5075112404)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在本研究中，作者实现并复现了 RHFL+ 算法，在 NVIDIA 的 NVFlare 框架下进行分布式训练，并将其扩展到医学影像领域的 CBIS‑DDSM、BreastMNIST、BHI 等公开数据集，系统评估了在不同标签噪声、客户端数量和模型异质性下的鲁棒性。

**💡 创新点**

创新点主要体现在：①将 RHFL+ 从单机仿真迁移到真正分布式的 NVFlare 生产级实现；②构建统一的实验框架，支持多算法、多噪声类型与多模型的自动化评测；③在医学影像任务中验证并改进了 RHFL+ 的性能。

**🔧 技术方法**

技术方法包括：联邦学习（FedAvg、FedMD 等）、异构模型融合（知识蒸馏、软标签）、动态标签修正（DLR）、对称交叉熵（SL）损失、增强型客户端置信度加权（ECCR）、NVFlare 的控制器/执行器插件、CUDA 内存分块计算等。

**📊 数据集**

主要数据集有 CIFAR‑10（私有）/CIFAR‑100（公开）用于基准实验，以及医学影像数据集 CBIS‑DDSM、BreastMNIST、BHI，分别用于乳腺癌和组织密度分类。

**📈 对比分析**

通过与 FedMD、FedDF、KT‑pFL、FedProto、FCCL、FedGH、FedTGP、RHFL 等 8 种基线对比，结果表明 RHFL+ 在对称和对称噪声下均能保持最高或次高的准确率（CIFAR‑10 平均约 79–80%），在医学影像任务中相较于 LocalOnly 也能提升 2–5% 的准确率或 PR‑AUC。

**⚠️ 局限性**

主要局限包括：①单点服务器导致容错性不足；②实验中使用的超参数未针对不同医学数据集进行细调；③FCCL 在显存有限时采用分批计算导致性能受损；④医学影像实验仅在模拟模式下完成，未验证真实分布式部署效果。

---

## 218. Enabling Long FFT Convolutions on Memory-Constrained FPGAs via Chunking

**arXiv ID:** 2601.06065 | [PDF](https://arxiv.org/pdf/2601.06065v1)

**作者:** Peter Wang `[一作]` (University of Southern California), Viktor Prasanna `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在资源受限的 FPGA（Alveo U200）上实现了分块 FFT 卷积，能够完成长度达 450K 的序列与同长度卷积核的全长一维卷积；

**💡 创新点**

创新点在于通过 chunking 与 overlap‑add 重构方法，将大规模卷积拆分为可在 2.8 MB BRAM 内处理的块；研究表明吞吐量随块大小线性提升，最长序列仅下降约 7%，首次在边缘 FPGA 上实现如此长上下文卷积；

**🔧 技术方法**

技术手段包括：Cooley‑Tukey radix‑2 FFT 核（Vitis HLS），主机侧数据分块与重构管理，FFT 频域相乘，FFT-FFT 逆变换，Overlap‑Add 合成；

**📊 数据集**

使用人类参考基因组 hg38 序列（32K、160K、450K 长度）及相同长度滤波器作为实验数据集；

**📈 对比分析**

性能评估通过表格展示不同块大小和序列长度下的执行时间、吞吐量（MFLOPS）和 compute/transfer/CPU 成本；结果显示吞吐量随块大小提升线性，最长序列吞吐约 104 MFLOPS，计算时间占比超过 98%，数据传输极小；与现有 FPGA FFT 实现（如 94 MS/s）相比，虽吞吐低但已证明可行性；

**⚠️ 局限性**

局限性包括：FFT 核仅支持最多 8,192 点，导致整体吞吐率相对低；未利用 URAM 或多核并行 FFT，且未实现稀疏因果 1‑D 卷积；未来需要进一步并行化、利用更大内存并降低计算/存储开销。

---

## 219. Improving Day-Ahead Grid Carbon Intensity Forecasting by Joint Modeling of Local-Temporal and Cross-Variable Dependencies Across Different Frequencies

**arXiv ID:** 2601.06530 | [PDF](https://arxiv.org/pdf/2601.06530v1)

**作者:** Bowen Zhang `[一作]` (University of Technology Sydney), A. Craig Roussac `[通讯]` (Buildings Alive Pty Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一个用于短期电网碳强度因子预测的深度学习框架。

**💡 创新点**

创新点是结合了多频率多波let卷积的局部时序提取模块与多频动态跨变量相关性卷积模块，能同时捕捉细粒度时序与动态交互。

**🔧 技术方法**

采用了连续小波变换、可变长度重叠分块、多波let卷积、局部多重回归、二维卷积、软max加权融合以及Grad‑CAM可解释性技术。

**📊 数据集**

使用了澳大利亚四个州（NSW、SA、QLD、VIC）2020‑2023年小时级别的电网数据，包括CIF、GLD、REG、NEG和温度共五个变量。

**📈 对比分析**

与LSTM、SVR、LSTNet、Crossformer、Informer、TimesNet、DLinear、NonStaFormer、PatchTST、iTransformer、TimeMixer、WPMixer等SOTA模型在RMSE、MAE、SMAPE上对比，模型在所有州均实现了最低误差，尤其在高变动州SA的MAE下降超过25%。

**⚠️ 局限性**

局限性包括仅使用有限的输入变量、仅在澳大利亚数据集验证、模型复杂度高、对不同市场的泛化性仍待进一步验证。

---

## 220. PDA in Action: Ten Principles for High-Quality Multi-Site Clinical Evidence Generation

**arXiv ID:** 2601.06072 | [PDF](https://arxiv.org/pdf/2601.06072v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 221. NC-Bench: An LLM Benchmark for Evaluating Conversational Competence

**arXiv ID:** 2601.06426 | [PDF](https://arxiv.org/pdf/2601.06426v1)

**作者:** Robert J. Moore `[一作]` (Independent Researcher), Jay Pankaj Gala `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了自然会话基准 NC-Bench，用以评估大语言模型在真实会话中的话语行为能力。

**💡 创新点**

创新点在于：①以会话分析理论为基础，构造超过120种通用会话模式；②设置基本、RAG 与复杂请求三大子基准；③采用 LLM 作为判定器实现自动化评测；④提供可扩展、轻量化的评测框架。

**🔧 技术方法**

使用技术包括：LLM 对话生成、基于提示的对话续写、LLM 判定器（分类器）评估、规则式得分与聚合。

**📊 数据集**

数据集来源：IBM Natural Conversation Framework 的模式库、DailyDialog 例句、Wikipedia 作为 RAG 参考文本、人工生成的 20 条商业场景对话。

**📈 对比分析**

评测方法：对 6 种开源 LLM（Qwen-3B、Qwen-7B、Granite-2B/8B、Llama-3B/8B）进行基准测试，采用贪婪解码、128 词限制；结果显示 Qwen-3B 在 Basic 集上 82.22%，Granite-8B 在 RAG 集上 77.77%，Granite-2B 在 Complex Request 集上 80.15%，整体表现优于其他模型。

**⚠️ 局限性**

局限性包括：模型在重复、闭合信号处理上表现欠佳；对无依据查询的回应仍倾向生成内容；评测仅覆盖文本对话，未涉及语音或多模态交互；模式仍有限，未来需扩展更多会话类型。

---

## 222. Breaking Model Lock-in: Cost-Efficient Zero-Shot LLM Routing via a Universal Latent Space

**arXiv ID:** 2601.06220 | [PDF](https://arxiv.org/pdf/2601.06220v1)

**作者:** Cheng Yan `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (Institute of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出ZeroRouter框架，利用通用潜在空间实现LLM路由，支持零样本快速集成新模型，并同时优化准确率、成本和延迟。

**💡 创新点**

创新点在于将查询属性与模型能力解耦构建跨任务的通用潜在空间，并通过信息论的D‑optimal采样实现轻量级模型剖析，从而突破模型锁定，支持零样本路由。

**🔧 技术方法**

使用技术包括多维2PL IRT模型、信息论D‑optimal采样、基于BERT的语义+结构混合特征的多任务预测器、整数线性规划多目标优化，以及成本/延迟的查找表估计。

**📊 数据集**

评估使用六个内部分布数据集（IFEval、BBH、MATH、GPQA、MuSR、MMLU‑PRO）和三个外部分布数据集（ARC‑C、TruthfulQA、HumanEval），并在60个LLM上进行实验。

**📈 对比分析**

与CIT‑LLM‑Routing、RouteLLM、GraphRouter、FORC等四个基线比较，ZeroRouter在小型和大型模型及三种策略下均实现最高准确率、最低成本与最低延迟，零样本泛化表现突出。

**⚠️ 局限性**

局限性包括对anchor集选择和预估表的依赖，极端新模型或任务分布大幅变化时可能需要重新采样；此外需要人工设定多目标权重，且对超大模型的计算资源需求较高。

---

## 223. AI Application Operations -- A Socio-Technical Framework for Data-driven Organizations

**arXiv ID:** 2601.06061 | [PDF](https://arxiv.org/pdf/2601.06061v1)

**作者:** Daniel Jönsson `[一作]` (Linköping University), Fredrik Viksten `[通讯]` (Linköping University)

**通讯引用:** 114 | [OpenAlex ID](https://openalex.org/A5091549351)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 AI 应用运维（AIAppOps）框架，将 MLOps 与价值驱动、治理和监控整合为从创意到生产的完整生命周期。

**💡 创新点**

创新点在于将价值假设与反馈循环贯穿所有阶段，扩展了应用与使用阶段，引入了基于 ProbSTL 的正式监控，以及多方共识的组织模型。

**🔧 技术方法**

采用了 MLOps 流水线、数据清洗、特征工程、AutoML、预训练/微调、CI/CD、服务编排、统计+形式化的运行时验证（ProbSTL）以及多维监控与治理工具。

**📊 数据集**

研究基于行业、政府与研究机构的真实数据，涵盖多模态、时序和图结构等；具体公开数据集未列出。

**📈 对比分析**

论文未给定实验指标，仅通过与现有 MLOps、Data Readiness、AI 治理框架的对比，阐述了监控与价值指标的改进；缺乏数值性能评估。

**⚠️ 局限性**

主要限制包括：需要较高的组织成熟度、实施成本和人机协作平衡；缺乏大规模实证验证；标准化与自动化程度尚待提升；对高风险领域的监控仍存在挑战。

---

## 224. 3D CoCa v2: Contrastive Learners with Test-Time Search for Generalizable Spatial Intelligence

**arXiv ID:** 2601.06496 | [PDF](https://arxiv.org/pdf/2601.06496v1)

**作者:** Hao Tang `[一作]` (Peking University), Zeyu Zhang `[通讯]` (Peking University)

**通讯引用:** 3690 | [OpenAlex ID](https://openalex.org/A5100358751)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了3D CoCa v2框架，将对比学习与生成式标题生成统一，并通过推理时搜索提升鲁棒性。

**💡 创新点**

创新点在于将CLIP先验冻结用于3D编码、联合对比与生成目标，并实现无参数更新的测试时搜索与LLM判定器选择最佳标题。

**🔧 技术方法**

采用CLIP视觉语言预训练、点云token化+Transformer编码、多模态Transformer解码器、对比损失、caption CE以及LLM判定等技术。

**📊 数据集**

使用ScanRefer、Nr3D（室内）和TOD^3Cap（室外）作为评测数据集。

**📈 对比分析**

与多种基准（Scan2Cap、Vote2Cap、3D‑VLP等）对比，在ScanRefer和Nr3D上提升CIDEr约+1.5，在零样本TOD^3Cap提升约+3.6，显示显著性能提升。

**⚠️ 局限性**

主要限制是推理时搜索增加时延与成本，判定器依赖与摘要信息不足可能导致偶尔错误，适用性受限于实时场景。

---

## 225. From Augmentation to Symbiosis: A Review of Human-AI Collaboration Frameworks, Performance, and Perils

**arXiv ID:** 2601.06030 | [PDF](https://arxiv.org/pdf/2601.06030v1)

**作者:** Richard Jiarui Tong `[一作]` `[通讯]` (NEOLAF Inc.), Richard Jiarui Tong (NEOLAF Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了60年来人机协作的发展，系统分析了Licklider的“共生”与Engelbart的“增强”两种理念，评估了大规模元分析中人机团队在决策与内容创作任务中的表现，并提出以“扩展自我”+“双过程”为核心的未来框架；

**💡 创新点**

创新点在于将历史的伙伴与工具辩证关系与现代HCAI、XAI、SMM等技术结合，揭示任务类型决定协同成效，并首次将扩展自我概念引入人机共生理论，提供了解决算法在环失效的路径；

**🔧 技术方法**

采用XAI可解释技术、共享心理模型（SMM）构建、交互式机器学习（HITL、主动学习、机器教学）等方法，结合元分析与案例研究实现多维度评价；

**📊 数据集**

主要数据来源为106项实验研究的370个效应量的元分析，Centaur案例使用Psych‑101人类决策实验数据集，以及对内容生成与编程等任务的公开评测数据；

**📈 对比分析**

通过对比人机团队、AI单独和人类单独在决策和内容创作任务中的准确率/效能，发现决策任务出现负协同（团队低于AI单独），而创作任务出现正协同（团队优于单独），Centaur案例进一步验证了共生在理论构造任务中的有效性；

**⚠️ 局限性**

主要局限在于对算法在环的信任校准失效未彻底解决，存在算法厌恶与自动化偏差、认知失能风险；所提出的扩展自我框架缺乏纵向实证验证，且研究多聚焦实验室场景，缺少真实长期使用的案例。

---

## 226. Hybrid LSTM-UKF Framework: Ankle Angle and Ground Reaction Force Estimation

**arXiv ID:** 2601.06473 | [PDF](https://arxiv.org/pdf/2601.06473v1)

**作者:** Mundla Narasimhappa `[一作]` (SRMIST University), Praveen Kumar `[通讯]` (SRMIST University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究提出了混合LSTM–UKF框架，用于估计不同速度下足踝角度和地面反作用力。

**💡 创新点**

结合深度时序网络与无迹卡尔曼滤波，利用多模态传感器融合提升预测精度并实现实时性。

**🔧 技术方法**

采用LSTM网络、UKF状态估计、加权融合的特征嵌入以及MSE+KL散度的损失。

**📊 数据集**

使用13名健康受试者在1–3 km/h步行时收集的力平台、GRF、膝关节角度及IMU数据（约91k个时间点）。

**📈 对比分析**

与单独的LSTM、UKF、EKF、KF等模型做交叉验证，Hybrid LSTM–UKF在GRF的RMSE降低约18.6%，踝角RMSE降低约22.4%。

**⚠️ 局限性**

依赖实验室力平台数据，IMU加速度不足导致单传感器性能不佳，且模型在更复杂运动或不同人群中的泛化性待验证。

---

## 227. Annotating Dimensions of Social Perception in Text: The First Sentence-Level Dataset of Warmth and Competence

**arXiv ID:** 2601.06316 | [PDF](https://arxiv.org/pdf/2601.06316v1)

**作者:** Mutaz Ayesh `[一作]` (Cardiff University), Nedjma Ousidhoum `[通讯]` (Cardiff University)

**通讯引用:** 259 | [OpenAlex ID](https://openalex.org/A5050190445)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个句子级社会感知数据集W&C‑Sent，包含超过1600条英语句子–目标对，并对每条句子在信任、社交性和能力三个维度上进行细粒度标注。

**💡 创新点**

创新点在于将温暖拆解为信任与社交性，并在句子层面进行人工标注，弥补了以往仅依赖词典的不足；同时提供了公开基准和高质量标注，为社交感知研究提供了新资源。

**🔧 技术方法**

使用了人工标注（Prolific平台）、多种模型评估（LR、BERTweet、GPT‑4o、Gemma、Qwen）以及零样本与少样本提示实验，衡量模型在细粒度和粗粒度温暖与能力预测上的表现。

**📊 数据集**

数据集来源于SemEval‑2016立场数据与ABCDE大语料库，经过目标抽取与筛选后得到1,633个句子–目标对，覆盖七个个人与社会群体。

**📈 对比分析**

与基线模型比较显示GPT‑4o在细粒度与粗粒度任务上均遥遥领先，BERTweet在细粒度任务中表现可观，但整体模型性能仍低于人类一致性，尤其在能力维度上表现最差。

**⚠️ 局限性**

主要限制包括仅使用英语数据、样本来源有限于社交媒体、标注者多来自英语北方地区、缺乏跨文化与低资源语言覆盖、以及模型在单实例预测上的可靠性不足。

---

## 228. When Imbalance Comes Twice: Active Learning under Simulated Class Imbalance and Label Shift in Binary Semantic Segmentation

**arXiv ID:** 2601.06209 | [PDF](https://arxiv.org/pdf/2601.06209v1)

**作者:** Julien Combes `[一作]` (Michelin), Jean-François Coeurjolly `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在工业缺陷检测中，主动学习如何在类不平衡和标签迁移（label shift）环境下提升语义分割模型的标注效率与推理性能。

**💡 创新点**

通过构造可控的类不平衡与标签迁移的合成数据集，系统评估三种常用主动学习策略（随机采样、熵采样、核心集采样）的鲁棒性与优势，并首次将此类评估扩展到语义分割任务。

**🔧 技术方法**

采用池化式主动学习框架，使用基于FPN的特征金字塔网络（预训练ImageNet），损失函数为Combo Loss（DiceLoss+BCE），采样策略为随机、像素熵均值、以及基于深度特征的核心集，评估指标为测试集的F1‑score并计算不确定区间。

**📊 数据集**

实验数据集为公开的土豆叶病害图像集（约5k张）和钢板缺陷图像集（约18k张），均切分为256×256/256×400的图像块后构造不同的πᵤ与πᵗ比例。

**📈 对比分析**

对比实验显示：熵采样和核心集采样在类不平衡度高或预算适中时明显优于随机采样；在存在标签迁移时性能下降但仍保持不劣于随机；两种数据集实验结果一致，证明方法的可迁移性和稳定性。

**⚠️ 局限性**

局限性包括：仅评估两种数据集且采用合成的不平衡生成方法，未探讨其他更复杂的主动学习策略或多类别分割；仅使用F1‑score作为性能指标，缺少对训练成本、标注时延等实际工业约束的量化分析。

---

## 229. Range-Coder with fast Adaptation and Table-Based Decoding

**arXiv ID:** 2601.06120 | [PDF](https://arxiv.org/pdf/2601.06120v1)

**作者:** Tilo Strutz `[一作]` (Coburg University), Roman Rischke `[通讯]` (Coburg University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于环形缓冲区的自适应表格解码方法，结合位移替代除法实现快速区间编码；

**💡 创新点**

创新点在于利用环形缓冲区保持总计数为2的幂，使得编码/解码核心可用位移代替除法，并且在自适应模式下实现表格解码的高效更新；

**🔧 技术方法**

使用区间编码（range coding）、表格搜索、环形缓冲区更新、二进制索引树（Fenwick tree）以及位移替代除法等技术；

**📊 数据集**

实验数据集包括10^8长度的合成符号序列（均匀分布与截断几何分布），以及三幅彩色图像的原始与预测误差图像；

**📈 对比分析**

与传统线性更新、二进制索引更新、表格解码、以及基于除法的实现进行对比，结果显示在静态模式下表格+位移实现可减少约40%运算量，在自适应模式下在符号数≤128（均匀）或≤256（几何）时性能最佳；

**⚠️ 局限性**

主要局限在于符号表过大时（如≥512）表格更新成本显著增加，二进制索引树的使用与表格解码不兼容，且对极端非均匀分布的适应性需要进一步评估。

---

## 230. Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI

**arXiv ID:** 2601.06161 | [PDF](https://arxiv.org/pdf/2601.06161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 231. Pareto-Optimal Model Selection for Low-Cost, Single-Lead EMG Control in Embedded Systems

**arXiv ID:** 2601.06516 | [PDF](https://arxiv.org/pdf/2601.06516v1)

**作者:** Carl Vincent Ladres Kho `[一作]` `[通讯]` (Minerva University), Carl Vincent Ladres Kho (Minerva University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对单通道低成本 EMG 传感器数据，评估了 18 种机器学习模型，用以实现低延迟、低内存的运动控制。

**💡 创新点**

提出了结合 Inception、Bi‑LSTM 与多头注意力的 MaxCRNN 架构，以及针对 ESP32 的 Pareto‑optimal 随机森林方案，实现了安全性与可部署性的双重突破。

**🔧 技术方法**

利用统计特征、原始序列、Mel 频谱图，分别在经典机器学习、深度学习和迁移学习框架下训练模型，并构建软投票集成。

**📊 数据集**

使用单个实验者 1.54M 样本（约 1,300 秒窗口）采集自 AD8232 与 ESP32 的数据，划分为 RELAX、CLENCH 与 NOISE 三类。

**📈 对比分析**

通过 5‑折交叉验证比较准确率、F1 分数和推理延迟；随机森林取得 74.25% 准确率、0.01 ms 延迟，MaxCRNN 在 GPU 上达 83.21% 准确率、0.15 ms 延迟。

**⚠️ 局限性**

局限在于单人数据、室内静止环境、样本量有限、深度模型无法在 ESP32 上运行，未来需多受试者验证与实时噪声建模。

---

## 232. Bi-Mem: Bidirectional Construction of Hierarchical Memory for Personalized LLMs via Inductive-Reflective Agents

**arXiv ID:** 2601.06490 | [PDF](https://arxiv.org/pdf/2601.06490v1)

**作者:** Wenyu Mao `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 15901 | [OpenAlex ID](https://openalex.org/A5100389037)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Bi-Mem 框架，通过双向构建（自下而上与自上而下）和关联检索，提升 LLM 在长时序个性化对话中的记忆一致性和问答性能。

**💡 创新点**

创新点在于：① 双向记忆构建，既用 inductive 代理聚合事实为场景再提取 persona，又用 reflective 代理对场景进行 persona 约束校正，避免局部聚类导致的“错误记忆”。② 关联检索机制采用传播激活（spreading activation）在多粒度记忆间互相唤醒，提高检索覆盖和一致性。

**🔧 技术方法**

技术主要包括：事实级抽取与图聚类（Label Propagation）、LLM 生成场景与 persona 的聚合与校正、基于向量相似度的检索与 BM25 混合检索、传播激活关联检索、双向代理机制。

**📊 数据集**

使用 LoCoMo 数据集进行实验，该数据集包含 50 轮对话、约 9000 tokens 以及 7512 个 QA 题目。

**📈 对比分析**

与多种基线（LongContext、RAG、Mem0、LightMem、A-MEM、SeCom、CAM）在 GPT‑4o‑mini 与 Qwen2.5‑14B‑Instruct 两个模型上对比，Bi‑Mem 在单跳、多跳、时间推理、开放域等任务均显著提升 F1/B1 分数，最高提升约 10‑15% 左右，且检索答复速度最快。

**⚠️ 局限性**

局限性包括：① 对底层 LLM 的推理与指令依赖较大，若 LLM 失效或指令不当会影响构建质量；② 目前 persona 设定为静态，无法很好跟踪用户喜好随时间的动态变化；③ 目前未对代理策略进行强化学习优化，仍可进一步提升校正效果。

---

## 233. Graph-Based Analysis of AI-Driven Labor Market Transitions: Evidence from 10,000 Egyptian Jobs and Policy Implications

**arXiv ID:** 2601.06129 | [PDF](https://arxiv.org/pdf/2601.06129v1)

**作者:** Ahmed Dawoud `[一作]` (Egyptian Center for Economic Studies), Mahmoud Mohamed `[通讯]` (Egyptian Center for Economic Studies)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并分析了包含 9,978 个埃及正式岗位的知识图谱，评估岗位自动化风险，并识别可行的职业迁移路径。

**💡 创新点**

① 通过 LLM 与语义聚类实现 0.74% 错误率的实体抽取；② 引入双重阈值（≥3 共通技能且 ≥50% 技能迁移）定义“可行迁移”；③ 通过桥接技能与网络中心性揭示结构性迁移瓶颈；④ 提出了“覆盖缺口”概念说明大部分高风险工人缺乏自然迁移路径。

**🔧 技术方法**

大语言模型（Gemini Pro 1.5）抽取实体；语义嵌入与阈值聚类；图论方法（Louvain 社区检测、betweenness、PageRank‑style 重要性）；任务级自动化风险计算。

**📊 数据集**

从 Wuzzuf、LinkedIn Egypt、Forasna 三大招聘平台抽取 9,978 条岗位，涵盖 98 个 ISCO‑3 类别，提取 19,766 项技能，形成 84,346 条岗位‑技能关系，构成 36,349 节点的知识图谱。

**📈 对比分析**

与传统基于岗位的自动化风险评估对比；通过与 O*NET 任务分解的一致性验证，抽取精度达 99.26%；在双阈值下发现 4,534 条可行迁移路径，平均 53.5% 技能迁移率和 48.1pp 的风险降低；覆盖缺口显示仅 24.4% 高风险工人具备自然迁移路径。

**⚠️ 局限性**

仅覆盖正式岗位的在线招聘数据，未涵盖 30% 的非正规就业；图谱为静态快照，缺乏纵向跟踪；自动化风险评分依赖 AI 任务抽取，可能忽视行业细节；方法与结论对其他地区的推广需进一步验证。

---

## 234. Interoperability in AI Safety Governance: Ethics, Regulations, and Standards

**arXiv ID:** 2601.06153 | [PDF](https://arxiv.org/pdf/2601.06153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 235. The Violation State: Safety State Persistence in a Multimodal Language Model Interface

**arXiv ID:** 2601.06049 | [PDF](https://arxiv.org/pdf/2601.06049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 236. Robotic Tele-Operation for Upper Aerodigestive Tract Microsurgery: System Design and Validation

**arXiv ID:** 2601.06617 | [PDF](https://arxiv.org/pdf/2601.06617v1)

**作者:** Giovani Braglia `[一作]` (Istituto Italiano di Tecnologia), Leonardo S. Mattos `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种用于上气道微创手术的远程操纵系统，配备可控手术镊子末端执行器，实现手术镊子在喉镜通道内的精确操作。

**💡 创新点**

创新点包括：①基于软件实现的远程中心运动（RCM）约束，消除传统手术中手部抖动和握持不稳；②简化的机械结构与易替换手术镊子设计；③将手术镊子夹持动作与姿态控制统一集成到机器人系统中。

**🔧 技术方法**

使用了Franka Emika Panda机器人、Omega.7六自由度触觉手柄、Maxon EPOS4电机驱动、ROS通信框架以及自研的RCM速度控制算法；实验中还采集了NDI Aurora跟踪数据和Myov2面部肌电信号。

**📊 数据集**

实验数据来自10名无TLM经验的受试者，使用硅胶声带模型（含六个模拟息肉）进行抓取与释放任务，并采集姿态、加速度与肌电信号。

**📈 对比分析**

通过与传统自由手操作对比，采用加速度RMS、肌电RMS与中频值(MDF)衡量稳定性与肌肉负荷；结果显示遥控系统在抓取稳定性、减少振动、降低肌肉激活与疲劳方面均优于自由手；问卷评估显示系统可用性和用户体验良好。

**⚠️ 局限性**

主要限制包括受试者训练时间有限、缺乏触觉反馈与3D视图、对喉镜遮挡的进一步控制尚未完善，以及系统的操作复杂度仍需优化。

---

## 237. Fixturize: Bridging the Fixture Gap in Test Generation

**arXiv ID:** 2601.06615 | [PDF](https://arxiv.org/pdf/2601.06615v1)

**作者:** Pengyu Xue `[一作]` (Shandong University), Kunwu Zheng `[通讯]` (Shandong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Fixturize 框架，利用 LLM 诊断并生成单元测试所需的 fixture，解决自动测试中忽略 fixture 的缺陷。

**💡 创新点**

将 fixture 识别与生成视为主动诊断过程，利用单行调用验证、迭代反馈修复以及 fixture 引导的 LLM 生成，实现高效的 fixture 感知测试。

**🔧 技术方法**

采用 LLM（DeepSeek、GPT‑4o、Claude、Qwen）结合基于运行时反馈的迭代修复、可执行调用生成、以及适配 unittest/JUnit 的提示工程。

**📊 数据集**

构建 FixtureEval 三子集（Python G、Python L、Java J），共 600 个函数，包含 100 个 fixture‑dependent 与 100 个 fixture‑independent 的标注。

**📈 对比分析**

与直接 LLM 提示、CoverUp、Pynguin 等基线对比，Fixturize 在 CasePS/ SuitePS 提升 15–45% 以上，覆盖率提升 30–120%；在 fixture‑dependent 样本上提升幅度更大。

**⚠️ 局限性**

依赖强大的 LLM，较小模型效果不佳；对复杂外部服务仍需人工或检索辅助；实验主要在有限领域与 4 大 LLM 上，泛化性待进一步验证。

---

## 238. Reliability and Admissibility of AI-Generated Forensic Evidence in Criminal Trials

**arXiv ID:** 2601.06048 | [PDF](https://arxiv.org/pdf/2601.06048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 239. Incentive Mechanism Design for Privacy-Preserving Decentralized Blockchain Relayers

**arXiv ID:** 2601.06699 | [PDF](https://arxiv.org/pdf/2601.06699v1)

**作者:** Boutaina Jebari `[一作]` (Ibn Tofail University), Mounir Ghogho `[通讯]` (University Mohammed VI Polytechnic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种去中心化区块链中继器的激励机制，利用博弈论设计了使中继器上传行为不可区分的概率策略，从而提升交易隐私并保持系统可靠性。

**💡 创新点**

核心创新在于将中继器协作建模为“志愿者困境”变体，并证明存在唯一的混合纳什均衡，该均衡既可被解析求解，又具有演化稳定性，保证在不可信环境下仍能实现隐私与可靠性的平衡。

**🔧 技术方法**

主要技术包括：非合作博弈建模、概率上传策略、潜在函数与演化动态分析、以及基于多参数的数值仿真。

**📊 数据集**

本文未使用真实区块链数据集，而是通过在参数空间（如中继器数量、上传成本、奖励与惩罚）上做离散化的模拟实验来评估模型。

**📈 对比分析**

通过数值仿真与敏感性分析对比不同参数组合的性能：当惩罚足够大时，系统在任意中继器规模下的失效概率可控制在5%以下；同时，增加中继器数量提升匿名集大小但会略微提高失效概率，形成隐私‑可靠性‑鲁棒性‑成本的权衡。

**⚠️ 局限性**

局限性主要包括：仅考虑链上隐私，未考虑网络层流量分析；上传成本采用归一化模型，未给出真实gas数值；模型假设参数固定且一次性交互，未考虑多轮交互与动态费用波动等实际部署挑战。

---

## 240. The Hessian of tall-skinny networks is easy to invert

**arXiv ID:** 2601.06096 | [PDF](https://arxiv.org/pdf/2601.06096v1)

**作者:** Ali Rahimi `[一作]` `[通讯]`, Ali Rahimi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种在不存储Hessian的前提下，按网络层数线性复杂度求解Hessian-逆向量积的方法，直接在网络结构上构造稀疏块三对角系统并求解；

**💡 创新点**

创新点在于利用Hessian的块双对角结构与其逆的多项式表示，构造一个可直接求逆的稀疏块系统，从而实现Hessian-逆向量积的精确计算，时间和存储复杂度仅线性依赖于网络深度；

**🔧 技术方法**

使用块矩阵操作、排列变换（commutation matrix）、块LDU分解、前向/后向代换以及与Pearlmutter的Hessian-向量积技术相类比的反向传播；

**📊 数据集**

论文中未给出具体数据集，重点是算法复杂度与理论分析；

**📈 对比分析**

与传统需要先显式构造Hessian、进行高阶运算（O(L³p³)）的朴素方法相比，该方法将复杂度降低到O(L·max(a,p)³)，在网络“高而瘦”（深度大、每层参数/激活数小）时更具优势；

**⚠️ 局限性**

限制包括：对数值稳定性依赖较大，块三对角系统求解可能出现数值不稳定；对每层维度的三次方成本仍可能在宽层网络中显著；以及需要可对每层求导的连续可微实现（对ReLU等非可微操作需近似）。

---

## 241. Bridging the AI divide in sub-Saharan Africa: Challenges and opportunities for inclusivity

**arXiv ID:** 2601.06145 | [PDF](https://arxiv.org/pdf/2601.06145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 242. Value of Information: A Framework for Human-Agent Communication

**arXiv ID:** 2601.06407 | [PDF](https://arxiv.org/pdf/2601.06407v1)

**作者:** Yijiang River Dong `[一作]` (University of Cambridge), Nigel Collier `[通讯]` (University of Cambridge)

**通讯引用:** 8465 | [OpenAlex ID](https://openalex.org/A5073413742)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于价值信息（VoI）的决策框架，让LLM代理在与用户交互时动态决定是提出澄清问题还是直接执行动作，从而在任务效用与沟通成本之间实现最优平衡。

**💡 创新点**

创新点在于将价值信息理论引入人机交互，使用无超参数的期望效用增益与沟通成本比较机制，自动适配不同任务风险、信息模糊度与用户认知负担，克服了传统阈值或固定轮数方法的脆性。

**🔧 技术方法**

核心技术包括：价值信息（VoI）计算、贝叶斯信念更新、LLM生成的答案概率模拟、一次步lookahead评估、Rational Speech Act视角以及基于LLM的概率推断与更新。

**📊 数据集**

实验使用四个多样化数据集：20 Questions（动物/医学）、FlightPREF（航班推荐）、WebShop（在线购物）以及对应的LLM模型（GPT‑4、Gemini‑2.5‑Flash）。

**📈 对比分析**

与无提问、固定轮数、适应性提示、置信阈值等基线进行比较；在大多数任务和沟通成本设定下，VoI方法与最佳手动调参基线相当或优于，最高可提升约1.36点效用，并且无需人工调参。

**⚠️ 局限性**

局限性包括：仅关注何时提问而不涉及问题生成；沟通成本采用线性简化模型，未涵盖更复杂的认知负担；LLM对信念分布的校准仍有不足，可能影响VoI估计精度。

---

## 243. Attack-Resistant Watermarking for AIGC Image Forensics via Diffusion-based Semantic Deflection

**arXiv ID:** 2601.06639 | [PDF](https://arxiv.org/pdf/2601.06639v1)

**作者:** Qingyu Liu `[一作]` (State Key Lab. of Blockchain and Data Security, Zhejiang University), Zhibo Wang `[通讯]` (State Key Lab. of Blockchain and Data Security, Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无需训练、基于扩散模型的内在水印框架 PAI，能在生成时双阶段嵌入水印并实现版权验证、攻击检测和语义级篡改定位。

**💡 创新点**

创新点在于：①通过 Box‑Muller 对初始化噪声注入用户密钥和时间盐，②设计关键条件偏转机制在前期采样步骤中进一步耦合水印与生成轨迹，③利用 DDIM 逆向过程的初始化偏差统一完成验证、攻击检测与篡改定位，并提供理论证明有效性。

**🔧 技术方法**

使用扩散模型（DDPM/Stable Diffusion）+ DDIM 逆向、Box‑Muller 生成 Gaussian 噪声、关键条件偏转（γ 乘以用户密钥）以及 PCA + Mahalanobis 检测。

**📊 数据集**

在 CelebA‑HQ、COCO、DDPM 预训练模型上生成 5,000 张水印与 5,000 张非水印图像进行评估。

**📈 对比分析**

与两类嵌入方法（EditGuard、Stable Signature）和两类内在方法（Tree‑Ring、Gaussian Shading）对比，PAI 在 12 种攻击（压缩、噪声、模糊、亮度、去除、伪造、局部/全局篡改）下平均所有权验证准确率 98.43%，比现有最优方法高 37.25%，同时在语义级篡改定位上 F1/IoU 远优于 EditGuard。

**⚠️ 局限性**

局限包括：对完全白盒关键提取攻击仍需进一步验证；需要对不同扩散模型的迁移性进行更系统评估；对极端低分辨率或高度压缩场景的鲁棒性尚未完全覆盖。

---

## 244. Deriving Decoder-Free Sparse Autoencoders from First Principles

**arXiv ID:** 2601.06478 | [PDF](https://arxiv.org/pdf/2601.06478v1)

**作者:** Alan Oursland `[一作]` `[通讯]`, Alan Oursland

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构造了一个由隐式EM理论直接推导出的无解码器稀疏自编码器：单层线性编码器配合LSE损失和InfoMax正则化，并验证理论预测。

**💡 创新点**

将隐式EM理论从解释工具转变为设计范式，证明理论可直接指定网络结构、目标函数与正则化，并通过实验验证梯度-责任等身份以及体积控制的必要性。

**🔧 技术方法**

使用Log‑Sum‑Exp目标、InfoMax（方差与去相关化）正则化、ReLU激活、梯度‑责任等价性，配合SGD与Adam的优化对比。

**📊 数据集**

以MNIST手写数字数据集进行训练与评估。

**📈 对比分析**

与传统稀疏自编码器（含重构损失与L1正则化）对比，指标包括线性探针准确率（93.4% vs 90.3%）、稀疏度（27% vs 50%）、参数量（1/2）以及损失下降；结果显示理论模型在性能与参数效率上优于基线。

**⚠️ 局限性**

实验仅在单层线性网络和MNIST上进行，未验证更深网络、更大数据集或语言模型的可扩展性；正式EM收敛性质未给出闭式证明；超参数选择未做系统调优。

---

## 245. Large Multimodal Model-Aided Scheduling for 6G Autonomous Communications

**arXiv ID:** 2601.06211 | [PDF](https://arxiv.org/pdf/2601.06211v1)

**作者:** Sunwoo Kim `[一作]` (Seoul National University), Byonghyo Shim `[通讯]` (Seoul National University)

**通讯引用:** 7536 | [OpenAlex ID](https://openalex.org/A5076075267)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于大型多模态模型（LMM）的预先调度框架（LMM‑PS），利用视觉感知与射频信号结合预测用户未来的信道参数（LoS 与 NLoS 路径距离、角度、路径增益），从而提前进行资源块分配和调制编码方案（MCS）选择，提升6G自动化通信系统的整体吞吐量。

**💡 创新点**

创新点包括：
1) 将LMM（GLIPv2、LLaVA）直接用于对象检测、轨迹预测与阻塞预测，弥补传统仅依赖射频回波的预估不足；
2) 通过Prompt和Chain‑of‑Thought技术实现高精度的用户轨迹与距离预测；
3) 结合Transformer预测LoS/NLoS路径参数，利用注意力机制捕捉长期与突发环境变化；
4) 将视觉感知的障碍信息与信道估计相结合，实现对LoS可达性实时判断；
5) 在预调度优化中直接使用未来信道估计，显著提高调度精度。

**🔧 技术方法**

所用技术：
- 视觉感知：GLIPv2（多模态目标检测）
- 轨迹与阻塞预测：LLaVA（多模态语言模型）
- 信道参数预测：Transformer（TCP‑L 和 TCP‑N）
- 资源分配：PF 规则、Hungarian 算法进行用户 ID 匹配
- 训练与优化：MSE+MAE 损失、Adam 迭代、学习率衰减
- 评估指标：吞吐量、NMSE、阻塞预测准确率、延迟/功耗。

**📊 数据集**

主要数据集：
- VOBEM2（RGB+深度图像，包含人员、手机及障碍）用于目标检测与定位；
- VOMM（移动场景视觉数据）用于泛化测试；
- 合成射频信道数据（基于MATLAB ray‑tracing + 3GPP 3GPP TR 38.901）用于信道参数预测与调度评估。

**📈 对比分析**

比较方法：
- 传统调度（Round‑Robin、Max‑SNR、PF 基于过去 CQI）
- DRL‑based 预调度
- 传统 LSTM / Kalman 信道预测
- LLaVA‑based 阻塞预测
性能：
- 相较传统调度，LMM‑PS 通过预调度实现 >30% 吞吐量提升；
- 与 DRL 预调度相比提升 11%；
- 与 Round‑Robin 提升 32%；
- 阻塞预测准确率提高 10% 以上；
- 延迟与功耗分别降低约 15% 及 25%（相较传统波束扫描）。

**⚠️ 局限性**

局限性：
- 未使用最新 LMM（如 GPT‑4V、LLaMA‑VID），可能影响预测精度；
- 需要部署视觉传感器和深度相机，硬件成本与部署复杂度提高；
- 依赖较为理想的视觉质量，遮挡或光照变化可能导致检测误差；
- 目前仅在单天线 UE、固定服务区尺寸的仿真环境验证，实际多天线、多移动场景的鲁棒性尚待研究；
- 对于极端高速或高密度障碍场景，预测误差仍可能显著；
- 训练与推理成本仍高，尤其在边缘设备上的实时性与能耗需要进一步优化。

---

## 246. Explainability of Complex AI Models with Correlation Impact Ratio

**arXiv ID:** 2601.06701 | [PDF](https://arxiv.org/pdf/2601.06701v1)

**作者:** Poushali Sengupta `[一作]` (Institute of Informatics), Anis Yazidi `[通讯]` (Oslo Metropolitan University)

**通讯引用:** 3333 | [OpenAlex ID](https://openalex.org/A5032770006)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种新的可解释性指标ExCIR，能够在存在相关特征、多个输出以及类别条件的情况下，单次计算得到稳定、可量化的特征重要性评分；

**💡 创新点**

创新点在于将相关性比率与典型相关分析、信息论（互信息）结合，保证评分在[0,1]范围内、单调且对噪声、采样变异鲁棒；同时扩展到多输出、块级（BlockCIR）与类别条件（CC‑CIR）；

**🔧 技术方法**

技术手段包括：基于协方差的相关性比率公式、单次线性变换、CCA求最优投影、信息论上界、Bootstrap和Bayesian不确定性估计；

**📊 数据集**

实验使用四类数据集：EEG（认知障碍诊断）、合成车辆传感器数据、Digits图像分类、Cats–Dogs二分类图像；

**📈 对比分析**

与SHAP、LIME、HSIC、MI等主流后置解释方法对比，ExCIR在特征排名保持、Top‑k重叠、模型准确性保持、解释速度（100–1000倍）以及在受噪声、分布漂移、子采样下的稳定性方面均显著优于对照方法；

**⚠️ 局限性**

局限性在于假设数据分布稳定，难以直接处理时间漂移、在线学习或需要局部解释的情形；此外，虽然对全局重要性高度可靠，但在某些复杂交互或多模态数据中的细粒度解释仍需进一步完善。

---

## 247. The Case for Strategic Data Stewardship: Re-imagining Data Governance to Make Responsible Data Re-use Possible

**arXiv ID:** 2601.06687 | [PDF](https://arxiv.org/pdf/2601.06687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 248. TEAS: Trusted Educational AI Standard: A Framework for Verifiable, Stable, Auditable, and Pedagogically Sound Learning Systems

**arXiv ID:** 2601.06066 | [PDF](https://arxiv.org/pdf/2601.06066v1)

**作者:** Abu Syed `[一作]` `[通讯]` (Metacog), Abu Syed (Metacog)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TEAS标准，整合可验证性、稳定性、可审计性和教学合理性四大支柱，评估教育AI系统是否具备部署可信度。

**💡 创新点**

创新点在于将技术、政策、教育和审计四个维度统一成四大支柱，强调体系架构而非模型能力决定可信度，并提出开放式验证方法。

**🔧 技术方法**

使用了RAG架构、检索增强生成、可解释模型接口、基于证据的教学策略等技术来实现可验证与可审计功能。

**📊 数据集**

使用了教育教材、同行评审书籍、机构认可的知识库，以及在Khan Academy、Khanmigo等实际部署数据进行验证。

**📈 对比分析**

通过在附录的实验案例，将8B参数的知识扎根模型与无根基前沿模型在TEAS四项指标上进行对比，结果显示前者在可验证性、稳定性与审计性上表现更优。

**⚠️ 局限性**

局限在于缺乏可操作的量化指标与工具，尚未对不同学科、不同地区进行广泛实证验证，且未覆盖AI代理安全问题。

---

## 249. Akasha 2: Hamiltonian State Space Duality and Visual-Language Joint Embedding Predictive Architectur

**arXiv ID:** 2601.06212 | [PDF](https://arxiv.org/pdf/2601.06212v1)

**作者:** Yani Meziani `[一作]` `[通讯]` (Independent AI Researcher), Yani Meziani (Independent AI Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `afceb026-1760-41ae-8d86-010831a37d97` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Akasha 2，一种融合Hamiltonian状态空间双重性（H‑SSD）与视觉‑语言联合嵌入预测框架（VL‑JEPA）的多模态架构，能够在移动端实现低延迟的视觉合成与长期稳定的多模态预测。

**💡 创新点**

核心创新点包括：① 通过Sparse Mixture of Hamiltonian Experts（SMoE‑HE）将Hamiltonian动力学嵌入混合专家网络；② 利用对称积分（symplectic leapfrog）保证潜在空间能量守恒；③ 引入Hamiltonian Flow Matching（HFM）和3D Gaussian Splatting（3DGS）实现比扩散模型快4×的生成速度；④ 采用层次化的Holographic Akasha Cells实现可递归的世界模型；⑤ 引入Phase‑Manifold V‑Sync解决异步模态同步问题。

**🔧 技术方法**

使用技术包括：Mamba‑3选择性状态空间模型、Sparse Mixture of Experts、对称积分、Hamiltonian Flow Matching、3D Gaussian Splatting、FP8量化、稀疏注意力、可学习的积分器（未来方向）。

**📊 数据集**

主要数据集：Kinetics‑400（视频预测）、COCO（图文理解）等；实验也在多模态同步和移动硬件上测试。

**📈 对比分析**

与VideoGPT、TECO、Phenaki等基线比较，FVD从582/461/394下降至287，SSIM从0.743/0.778/0.801提升至0.841；在COCO上BLEU‑4、METEOR、CIDEr、SPICE分别提升至41.3/31.8/135.7/24.6；在推理速度上，NVIDIA A100上从78 ms提升至23 ms，Apple M2 Pro从342 ms降至47 ms，Qualcomm 8 Gen 3从891 ms降至49 ms，实现3–18×速度提升。

**⚠️ 局限性**

局限性包括：Hamiltonian约束可能限制对高度混沌动力学的建模；积分时间步长Δt需要精细调优；训练时对称积分增加计算开销；模型对极端动态场景的表达能力有限。

---

## 250. How Does India Cook Biryani?

**arXiv ID:** 2601.06198 | [PDF](https://arxiv.org/pdf/2601.06198v1)

**作者:** Shubham Goel `[一作]` (Indraprastha Institute of Information Technology Hyderabad), C V Jawahar `[通讯]` (Indraprastha Institute of Information Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并公开了首个针对印度12种地区Biryani烹饪的120段YouTube视频数据集，开发了基于视觉‑语言模型的多阶段视频分割、跨模态对齐与区域差异比较框架，并提供了易中难三级问答基准。

**💡 创新点**

创新点包括：
①首次大规模、细粒度标注印度烹饪视频；
②将多模态VLM与动态时间规整、聚类、CLIP检索等技术结合，自动完成视频分段、对齐和差异可视化；
③提出跨区域烹饪差异检索和多视频推理的问答基准，推动文化语境下的视频理解研究。

**🔧 技术方法**

核心技术：
- 视觉‑语言模型（InternVL‑14B、Qwen2‑VL、Gemini‑2.5‑flash‑lite、Llama‑3.2‑Vision‑Instruct）
- 语音识别（WhisperX/Whisper‑Large）与翻译（GPT‑4）
- 动态时间规整（DTW）与聚类（MiniLM‑L6）
- 视觉检索（CLIP Open‑CLIP）
- 多阶段QA生成与人工校验

**📊 数据集**

使用的数据集为120段高质量YouTube Biryani视频（12种地区各10段），并在此基础上构建了细粒度分段、对齐标签和240易、1,357中、486难三级问答集。

**📈 对比分析**

比较方法：基于改造的VidDiff框架，先用LLM生成动作差异与子动作，再用CLIP检索关键帧，最后由VLM做多选判断；在Biryani视频对比中检测到33.2%动作差异。问答基准上，Fine‑tuned Llama‑3.2在所有指标（BLEU、ROUGE‑L、BERTScore）均优于零射模型，尤其在难题中BERTScore达0.45，显示微调能显著提升跨视频推理能力。

**⚠️ 局限性**

局限：
- 模型对印度烹饪文化的内在知识不足，导致“无差异”判断准确率仅45.7%；
- 细粒度差异检测对同义动作识别仍不够精确；
- 数据集仅覆盖Biryani，未扩展到其他菜系；
- 长视频多模态对齐与推理仍面临计算与时序误差挑战。

---

## 251. Beyond Clicking:A Step Towards Generalist GUI Grounding via Text Dragging

**arXiv ID:** 2601.06031 | [PDF](https://arxiv.org/pdf/2601.06031v1)

**作者:** Zeyi Liao `[一作]` (Ohio State University), Ahmed Awadallah `[通讯]` (Microsoft Research)

**通讯引用:** 3280 | [OpenAlex ID](https://openalex.org/A5021000040)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出可扩展的三阶段管道自动从截图合成文本拖拽示例，并构建了161K实例的GUI‑Drag数据集与5,333例的ScreenDrag基准；

**💡 创新点**

首次系统聚焦文本拖拽任务，创新性地设计了多层界面上下文的基准、三种评价指标（DTR、B‑Dist、SR）以及高效的持续训练策略；

**🔧 技术方法**

结合LLM（如o4‑mini、o3）生成指令与坐标、EasyOCR实现字符级识别，并在Jedi‑3B/7B上采用持续训练；

**📊 数据集**

数据来源自UGround、Jedi、公开论文文档等，最终形成161K文本拖拽样本和5,333个ScreenDrag评测样本；

**📈 对比分析**

与Qwen、Jedi、UI‑TARS及OpenAI/Claude CUA等对照实验，持续训练后模型在ScreenDrag上DTR≈100%、B‑Dist显著下降、SR最高提升至≈38%，同时保持点击性能；

**⚠️ 局限性**

局限在于对非文本拖拽动作支持不足、跨域迁移能力待提升，以及对多语言与复杂布局的鲁棒性验证不足。

---

## 252. On the Number of Subsequences in the Nonbinary Deletion Channel

**arXiv ID:** 2601.06493 | [PDF](https://arxiv.org/pdf/2601.06493v1)

**作者:** Han Li `[一作]` (Nankai University), Fang-Wei Fu `[通讯]` (Nankai University)

**通讯引用:** 3081 | [OpenAlex ID](https://openalex.org/A5063946169)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在非二进制删除信道中，字符串U在进行t次删除后生成的子序列数量，并提出了r-run非二进制字符串子序列数量的改进界限。

**💡 创新点**

提出了一类具有最大子序列数量的r-run非二进制字符串，并证明该数量可以在多项式时间内计算。

**🔧 技术方法**

使用了字符串操作、递归表达式和闭式公式等技术。

**📊 数据集**

使用了q-进制字符串的理论，特别是r-run字符串的构造。

**📈 对比分析**

通过与已有的二进制字符串的下界和上界进行比较，展示了所提出方法的有效性，性能优于之前的结果。

**⚠️ 局限性**

在r不整除n的情况下，极值字符串的完整特征仍然是一个开放问题。

---

## 253. FlexAct: Why Learn when you can Pick?

**arXiv ID:** 2601.06441 | [PDF](https://arxiv.org/pdf/2601.06441v1)

**作者:** Ramnath Kumar `[一作]` (University of California Los Angeles), Cho-Jui Hsieh `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Gumbel-Softmax的离散激活函数选择框架，让网络在训练过程中自适应地为每一层选择最合适的激活函数；

**💡 创新点**

创新点在于将激活函数的离散选择与梯度可微的重参数化结合，并引入基于梯度范数的正则化以纠正对无界激活的偏置，提升了模型的可解释性与适应性；

**🔧 技术方法**

使用了Gumbel-Softmax重参数化、直通估计器、KL正则化、梯度范数归一化以及传统的梯度下降训练；

**📊 数据集**

主要在合成回归数据集上验证，数据集由一维信号与高斯噪声构成，目标函数为已知的五种激活之一；

**📈 对比分析**

与固定激活（ReLU、Sigmoid、Tanh、LeakyReLU、Identity）及无正则化版本对比，结果显示该方法能够迅速收敛至正确的激活，平均MSE显著低于基线，且正则化后能消除对ReLU的偏好；

**⚠️ 局限性**

局限性包括目前仅在倒数第二层实现离散选择，未测试多层深度网络；正则化手段为经验性、启发式；在更大规模或更复杂任务中的可扩展性与稳定性仍待进一步验证。

---

## 254. Algorithm Support for Graph Databases, Done Right

**arXiv ID:** 2601.06705 | [PDF](https://arxiv.org/pdf/2601.06705v1)

**作者:** Daan de Graaf `[一作]` (Eindhoven University of Technology), Nikolay Yakovets `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 457 | [OpenAlex ID](https://openalex.org/A5082856403)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于线性代数的图算法DSL GraphAlg，并将其编译到关系代数，在AvantGraph中实现统一查询和算法执行。

**💡 创新点**

创新点在于将图算法视为第一类事务，提供可编译到查询引擎的DSL，结合稀疏性分析、循环不变代码迁移和原地聚合等优化，实现高性能、低代码量。

**🔧 技术方法**

技术包括线性代数模型MATLANG/for-MATLANG、关系代数映射、稀疏矩阵表示、循环不变代码迁移、原地聚合、AvantGraph内部IR统一、Cypher扩展等。

**📊 数据集**

使用LDBC Graphalytics基准集（BFS、PR、SSSP、WCC、CDLP）以及OpenAIRE能源规划数据集进行评测。

**📈 对比分析**

与Neo4j、PostgreSQL、DuckDB等传统图/关系数据库及Pregel/Java实现比较，GraphAlg在代码行数上减少约9.5倍，在PageRank、SSSP、WCC等算法上跑速最快，性能优于Neo4j、DuckDB，且在预处理融合上表现更好。

**⚠️ 局限性**

局限性包括对hash聚合性能仍有瓶颈，CDLP性能低于DuckDB，需进一步优化；实现仍依赖AvantGraph特定扩展，对其他引擎的可移植性有限。

---

## 255. Optimal Beamforming for Uplink Covert Communication in MIMO GEO Satellite-Terrestrial Systems

**arXiv ID:** 2601.06110 | [PDF](https://arxiv.org/pdf/2601.06110v1)

**作者:** Zewei Guo `[一作]` (Future University Hakodate), Xiaohong Jiang `[通讯]` (Future University Hakodate)

**通讯引用:** 5775 | [OpenAlex ID](https://openalex.org/A5014884811)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了MIMO GEO卫星-地面系统中的向上隐蔽通信，提出最优波束成形与联合波束成形与天线定向设计，以提升隐蔽速率。

**💡 创新点**

创新点在于同时考虑多波束天线定向、非单波导多收发天线、噪声不确定性以及完美与不完美CSI场景下的隐蔽通信性能建模与优化。

**🔧 技术方法**

采用半正定松弛（SDR）、交替优化、Rodrigues旋转公式、1D搜索以及Nakagami‑m近似等优化技术。

**📊 数据集**

实验使用基于Ku波段（18 GHz）卫星参数、模拟的地面站与卫星阵列（64×16）以及多卫星对齐的坐标位置，未使用公开数据集。

**📈 对比分析**

与传统的最大比率传输（MRT）和零迫迫（ZF）方案对比，提出的OB/JO‑BA方案在多收容器场景下隐蔽速率提升可达30–50%（具体数值见图）。

**⚠️ 局限性**

局限在于模型假设理想的LoS+Rician信道、静态场景、仅考虑单向隐蔽传输，未考虑多址、动态卫星轨道与真实噪声统计。

---

## 256. ToolGym: an Open-world Tool-using Environment for Scalable Agent Testing and Data Curation

**arXiv ID:** 2601.06328 | [PDF](https://arxiv.org/pdf/2601.06328v1)

**作者:** Ziqiao Xi `[一作]` (University of California San Diego), Kun Zhou `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了ToolGym开源工具使用环境，包含5711个统一MCP格式工具、任务生成引擎、状态控制器和Planner‑Actor框架，用于评估与训练LLM工具使用代理。

**💡 创新点**

创新点：①大规模开放式工具库与多工具、长周期、野性约束任务生成；②状态控制器可注入真实失效以测试鲁棒性；③Planner‑Actor拆分将规划与执行分离；④利用该环境自动收集1,170条轨迹实现仅1,170样本即可超越119k样本的微调效果。

**🔧 技术方法**

采用MCP统一工具协议、向量检索索引、LLM评判（GPT‑4o、GPT‑5.1、DeepSeek‑V3.2）与ReAct+规划、状态控制器注入失效，结合大语言模型（DeepSeek‑V3.2、Gemini‑3‑pro‑preview、Claude‑opus‑4.5等）进行实验。

**📊 数据集**

使用了5,571工具覆盖204个常用App的数据集，Synthesized 50个任务与1,170条执行轨迹；在BFCL和MCP‑Universe等公开基准上进行对比评测。

**📈 对比分析**

通过pass@3和整体得分评估，发现DeepSeek‑V3.2在鲁棒性（回滚率90.6%）最高，Gemini‑3‑pro在任务完成质量（>4.7/5）最佳；1,170样本微调的Qwen2.5‑7B和Qwen3‑8B在BFCL和MCP‑Universe上的准确率分别提升至≈30%和≈8%，显著优于使用119k样本的基线。

**⚠️ 局限性**

局限性：评测场景仅包含50个合成任务，覆盖范围有限；主要评估大型模型，未系统测试小模型，结论可能不适用于参数量不足10B的模型。

---

## 257. "They parted illusions -- they parted disclaim marinade": Misalignment as structural fidelity in LLMs

**arXiv ID:** 2601.06047 | [PDF](https://arxiv.org/pdf/2601.06047v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 258. Tone Matters: The Impact of Linguistic Tone on Hallucination in VLMs

**arXiv ID:** 2601.06460 | [PDF](https://arxiv.org/pdf/2601.06460v1)

**作者:** Weihao Hong `[一作]` (Kean University), Boyang Li `[通讯]` (Kean University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构造了Ghost-100合成基准，并在该基准上使用三款开源VLM对不同强度提示（5级）进行零参数推理，系统评估幻觉出现率和严重度；

**💡 创新点**

创新点在于：①提出Ghost-100缺失信息场景的合成评测集；②设计分层提示强度框架，将语义毒性与结构强制区分；③揭示幻觉随提示强度呈非单调曲线，极强制可触发拒绝或降低幻觉，提示安全对齐机制存在模型特异性；

**🔧 技术方法**

采用零参数VLM推理、人工标注与LLM（GPT‑4o‑mini）评估、结构化幻觉评分表以及5级提示强度框架；

**📊 数据集**

使用Ghost-100（600张合成图像，分6类、每类100张，目标缺失）与三款开源VLM（MiniCPM‑V 2.6‑8B、Qwen2‑VL‑7B、Qwen3‑VL‑8B）；

**📈 对比分析**

通过在同一图像和五级提示下计算攻击成功率（ASR）与幻觉严重度得分（HSS），发现所有模型在中等强度时幻觉率最高，随着强度进一步升高幻觉率与严重度下降，表明不同模型对极端提示的安全对齐阈值不同；

**⚠️ 局限性**

局限性包括：①合成数据集场景有限，缺乏真实多样性；②评价依赖人工与LLM主观判定，可能存在偏差；③仅评测三款中小型VLM，未覆盖大型模型；④未探究检索、动态适配等机制对幻觉的影响。

---

## 259. GroupSegment-SHAP: Shapley Value Explanations with Group-Segment Players for Multivariate Time Series

**arXiv ID:** 2601.06114 | [PDF](https://arxiv.org/pdf/2601.06114v1)

**作者:** Jinwoong Kim `[一作]` (Hanyang University), Sangjin Park `[通讯]` (Hanyang University)

**通讯引用:** 211 | [OpenAlex ID](https://openalex.org/A5108339099)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了GroupSegment‑SHAP，一种针对多变量时间序列模型的SHAP解释框架，构造跨变量组和时间段的联合解释单元，并通过Shapley值量化其贡献。

**💡 创新点**

创新点在于同时利用HSIC进行特征分组、MMD进行时段分割，并将二者结合成联合组段玩家，避免了传统方法在时间‑特征轴上分离导致的碎片化问题，从而更好地捕捉模型使用的多变量‑时序交互模式。

**🔧 技术方法**

核心技术包括Hilbert‑Schmidt独立性检验（HSIC）用于特征组建，最大均值差距（MMD）用于检测分布变化并划分时间段，基于Shapley值的置换采样估计，以及均值替代的掩码策略。

**📊 数据集**

实验数据涵盖四个领域：UCI‑HAR（人体动作识别）、ETTm1（电力负荷预测）、PTB‑XL（12导联心电图分类）以及S&P500（股票日收益预测）。

**📈 对比分析**

与KernelSHAP、TimeSHAP、SequenceSHAP、WindowSHAP、TSHAP等现有时序SHAP解释器在相同模型和扰动预算下进行比较，结果显示GS‑SHAP平均提高约1.7倍的删除式可信度（ΔAUC）并在匹配扰动预算下平均缩短约40%的运行时间，同时在鲁棒性和一致性上也表现更佳。

**⚠️ 局限性**

局限性包括解释单元基于统计结构，可能不符合领域语义；依赖近似Shapley估计，随着序列长度和维度增大效率下降；并且对分段长度、阈值等超参数敏感，需要进一步优化采样和估计策略。

---

## 260. Otimizando A Alocação De Salas De Aula Com Foco Na Acessibilidade Para Pessoas Com Deficiência

**arXiv ID:** 2601.06670 | [PDF](https://arxiv.org/pdf/2601.06670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 261. Visible Light Communication using Led-Based AR Markers for Robot Localization

**arXiv ID:** 2601.06527 | [PDF](https://arxiv.org/pdf/2601.06527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 262. Steer Model beyond Assistant: Controlling System Prompt Strength via Contrastive Decoding

**arXiv ID:** 2601.06403 | [PDF](https://arxiv.org/pdf/2601.06403v1)

**作者:** Yijiang River Dong `[一作]` (University of Cambridge), Nigel Collier `[通讯]` (University of Cambridge)

**通讯引用:** 8465 | [OpenAlex ID](https://openalex.org/A5073413742)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需再训练的系统提示强度（System Prompt Strength）方法，通过在同一模型中对比目标系统提示与默认系统提示的对数几率，来调节模型对提示的遵循程度；

**💡 创新点**

创新点在于将系统提示的强度视为可调连续控制参数α，并利用对比解码（contrastive decoding）直接放大目标提示的行为差异，从而实现对模型行为的细粒度、可插拔控制；

**🔧 技术方法**

使用对比解码技术，在推理时分别计算目标提示与默认（或负面）提示的 logits，将二者差值乘以 α 后与目标提示 logits 相加，重新 softmax 生成 token；

**📊 数据集**

在五个多样化基准上评估：IFEval（约束遵循）、OffTopicEval（领域限制与拒绝）、Prompt‑Steering（多样人格切换）、Inverse Capability（低能力模拟）以及 Readability Control（阅读难度控制），并在多种开源模型上（Qwen‑2.5‑7B/14B/32B、Llama‑3.1‑8B、Olmo‑3‑32B）进行实验；

**📈 对比分析**

与传统提示工程、外部分类器引导和激活调节等方法对比，系统提示强度在所有任务上均实现了显著提升：IFEval 严格准确率提升最高 8.5 分，OffTopicEval 对话拒绝率提升 45 个百分点，Prompt‑Steering 细粒度调节得分提升约 13%；同时在低能力模拟任务中可将模型准确率从 90% 降到 10% 以上，阅读难度控制中误差下降近一半；

**⚠️ 局限性**

主要局限是推理时需要双倍计算 logits，导致近乎两倍 FLOPs，且高 α 可能引入“调节税”，即在实现目标行为的同时削弱模型在其他能力上的表现；此外，极端 α 值可能导致模型过度拒绝或失去一般性实用性。

---

## 263. Time Travel Engine: A Shared Latent Chronological Manifold Enables Historical Navigation in Large Language Models

**arXiv ID:** 2601.06437 | [PDF](https://arxiv.org/pdf/2601.06437v1)

**作者:** Jingmin An `[一作]` (Peking University), Fang Fang `[通讯]` (Peking University)

**通讯引用:** 32509 | [OpenAlex ID](https://openalex.org/A5100386859)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Time Travel Engine（TTE），一种能够在 LLM 潜在空间中沿连续时间流动并动态调节文本语义与风格的可解释性框架。

**💡 创新点**

首次将时间建模为可遍历的连续流形，并展示中英两语在时间子空间的拓扑同构，从而实现跨语言零样本时间推理。

**🔧 技术方法**

采用对比激活增补、集合对比、流形投影（CMP/EnsCMP）、Procrustes 旋转、PCA 解耦以及基于残差流的动态干预等多种技术。

**📊 数据集**

构建了包含古今四个时期的英中平行语料（约 90K 词）以及三个评测基准（知识边界、因果重塑、风格解耦）作为数据集。

**📈 对比分析**

通过 Perplexity 矩阵、未来泄露率（FLR）和精确率（PR）与基线比较，TTE 在保持语义一致的同时显著降低未来泄露率、提高时代适配的 PPL，且连续流形策略优于离散向量，跨模型实验也验证了这一优势。

**⚠️ 局限性**

受限于训练数据的时间分布、风格与认知难以完全分离以及流形近似的局部细节缺失，导致对稀缺时期和突变事件的建模不够精细。

---

## 264. Kolmogorov-Arnold Networks-Based Tolerance-Aware Manufacturability Assessment Integrating Design-for-Manufacturing Principles

**arXiv ID:** 2601.06334 | [PDF](https://arxiv.org/pdf/2601.06334v1)

**作者:** Masoud Deylami `[一作]`, Adel Alaeddini `[通讯]` (Southern Methodist University)

**通讯引用:** 1195 | [OpenAlex ID](https://openalex.org/A5077624638)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于参数化特征和 Kolmogorov–Arnold 网络（KAN）的可制造性评估方法，直接利用设计参数而非几何数据进行预测。

**💡 创新点**

创新点在于使用 KAN 的可解释一维样条激活函数，能够显式考虑尺寸公差、保持信息完整性，并提供高可解释的判别与设计反馈。

**🔧 技术方法**

技术包括 Kolmogorov–Arnold 网络、样条边缘函数、SHAP 特征重要性、KAN 隐空间投影以及与传统机器学习模型（XGBoost、LightGBM、MLP 等）的对比。

**📊 数据集**

使用自制的三大场景（孔钻、槽铣、组合钻铣）共计 300,000 条合成标注设计数据，涵盖尺寸、容差、工具及 DFM 约束。

**📈 对比分析**

与 14 种基准模型对比，KAN 在所有场景下获得最高 AUC（0.9919、0.9841、0.9406），并表现出更高的准确率和更好的泛化能力。

**⚠️ 局限性**

局限性包括：仍依赖合成数据，未覆盖所有加工过程；对多特征复杂几何的解释仍有限；对真实制造数据的验证尚不足。

---

## 265. LLM Agents in Law: Taxonomy, Applications, and Challenges

**arXiv ID:** 2601.06216 | [PDF](https://arxiv.org/pdf/2601.06216v1)

**作者:** Shuang Liu `[一作]` (Carnegie Mellon University), Mengnan Du `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 4818 | [OpenAlex ID](https://openalex.org/A5072191151)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了法律领域LLM代理的研究现状，系统阐述了从标准LLM向代理架构的技术演进、分类应用、评估方法和未来挑战。

**💡 创新点**

以代理架构为中心，首次提出面向法律的完整分类、代理功能与法律任务的匹配、评估框架以及挑战与方向。

**🔧 技术方法**

主要技术包括规划、记忆、工具使用、检索增强生成、反思、多代理协作和人机交互。

**📊 数据集**

采用公开法律文本、案例检索数据库、法规和判例库等数据源，但未给出单一数据集。

**📈 对比分析**

通过对现有代理系统的功能与性能对比，指出在信息检索、合同分析等领域表现突出，但在多步推理、程序一致性等方面仍落后于传统手工流程。

**⚠️ 局限性**

局限在于缺乏专门的评估基准、对多司法域的覆盖不足、对商业代理系统内部结构未知，以及对代理自主决策的法律合规性研究不足。

---

## 266. Learning Minimally-Congested Drive Times from Sparse Open Networks: A Lightweight RF-Based Estimator for Urban Roadway Operations

**arXiv ID:** 2601.06124 | [PDF](https://arxiv.org/pdf/2601.06124v1)

**作者:** Adewumi Augustine Adepitan `[一作]` (George Mason University), Ayooluwatomiwa Ajiboye `[通讯]` (George Mason University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出轻量化随机森林框架，用开放道路网络数据和稀疏交通控制特征预测最低拥堵状态下的行驶时间。

**💡 创新点**

创新点在于将最短路径基础时刻与稀疏的交通控制/转弯特征结合，通过随机森林校正偏差，既保持了简易性又提升了预测精度。

**🔧 技术方法**

采用随机森林回归、Dijkstra路径计算、特征工程（控制点计数、转弯角度统计）以及OpenStreetMap网络构建。

**📊 数据集**

使用洛杉矶城市的OpenStreetMap道路网络、交通控制标记、Uber Movement交通流、Google Maps Routes API参考时刻。

**📈 对比分析**

与最短路径基线和其他机器学习模型（梯度提升、AdaBoost、决策树）对比，随机森林在MAPE从21.15%降至8.41%，MAE下降至75s，R²提升至0.93，系统性偏差几乎为零。

**⚠️ 局限性**

局限在于仅针对最小拥堵条件，缺乏实时拥堵动态；开放数据标记稀疏可能影响性能；对不同城市的泛化需要额外校准。

---

## 267. How Generative AI Empowers Attackers and Defenders Across the Trust & Safety Landscape

**arXiv ID:** 2601.06033 | [PDF](https://arxiv.org/pdf/2601.06033v1)

**作者:** Patrick Gage Kelley `[一作]` (Google), Allison Woodruff `[通讯]` (Google)

**通讯引用:** 5450 | [OpenAlex ID](https://openalex.org/A5045188560)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过在全球五个 Trust & Safety 领域（儿童安全、选举完整性、仇恨与骚扰、诈骗、极端暴力）进行43名专家的六小时参与式研讨会，收集并分析了记忆经历卡、变革卡和未来故事等材料，探讨生成式 AI 对攻击者与防御者的双重影响。

**💡 创新点**

首次从专家视角系统性梳理生成式 AI 对 Trust & Safety 的赋能与威胁，并提出跨域的战略框架与操作性改进路径，强调在不同领域中生成式 AI 对攻击与防御的多层次交互。

**🔧 技术方法**

采用参与式研究工作坊、实地访谈、自动语音转写、手工校正、记忆卡和变革卡等材料收集方法，以及归纳式主题分析（Reflexive Thematic Analysis）对数据进行编码和主题构建。

**📊 数据集**

主要数据集为 43 名专家在五个领域的访谈录音、记忆经历卡（125份）、变革卡（99份）以及未来故事（28份）等定性材料；未使用公开机器学习数据集。

**📈 对比分析**

本研究不涉及算法性能对比；比较以专家观点为基础，阐述生成式 AI 在攻击与防御方面的相对影响，并未给出数值指标或基准测试。

**⚠️ 局限性**

局限性包括样本规模有限（仅一组每领域）、受访者可能受引导性问题影响、研究仅在四个城市开展、专家观点随技术演进可能改变，且研究未对跨域的通用性进行量化验证。

---

## 268. Mosaic: Unlocking Long-Context Inference for Diffusion LLMs via Global Memory Planning and Dynamic Peak Taming

**arXiv ID:** 2601.06562 | [PDF](https://arxiv.org/pdf/2601.06562v1)

**作者:** Liang Zheng `[一作]` (Tianjin University), Keqiu Li `[通讯]` (Tianjin University)

**通讯引用:** 5109 | [OpenAlex ID](https://openalex.org/A5111982109)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为扩展扩散式大语言模型（dLLMs）的长上下文推理能力，提出了一套从零开始的内存高效推理系统。

**💡 创新点**

创新点包括：① 仅对被遮蔽的 token 计算 logits 的 mask‑only kernel，消除无用计算；② 通过 lazy chunking 与在线瓶颈驱动搜索动态切分 FFN 与 logits，以适应不同遮蔽比例下的内存峰值；③ 采用全局图注册器实现完整的计算图可见性，并用全局虚拟内存管理器一次性规划张量分配，消除外部碎片。

**🔧 技术方法**

使用技术包括：基于张量图的 liveness 分析、gather‑GEMM 融合 kernel、动态块划分与在线搜索、first‑fit 规划、VMM（虚拟内存）分配以及自定义原地算子。

**📊 数据集**

评估数据集为人工构造的“dummy”长序列输入，覆盖 LLaDA‑8B、Dream‑7B 与 LLaDA‑MoE 等三种主流 dLLM，在 RTX‑3090/ A100 GPU 上进行测试。

**📈 对比分析**

与三类基线（Native、-Torch、-Compile）对比，系统在不牺牲或甚至降低推理延迟的前提下，平均将峰值‑平均比降低 2.71×，将最大可支持上下文长度提升 15.89‑32.98×，并实现 4.12‑23.26% 的延迟下降。

**⚠️ 局限性**

局限性：依赖于完整可预先构建的计算图，可能对模型改动敏感；在线搜索和图注册的开销在极短序列时会出现额外负担；系统主要针对扩散式 Transformer，尚未验证对非 Transformer 或混合模型的适用性。

---

## 269. Towards Public Administration Research Based on Interpretable Machine Learning

**arXiv ID:** 2601.06205 | [PDF](https://arxiv.org/pdf/2601.06205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 270. GenAITEd Ghana_A Blueprint Prototype for Context-Aware and Region-Specific Conversational AI Agent for Teacher Education

**arXiv ID:** 2601.06093 | [PDF](https://arxiv.org/pdf/2601.06093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 271. HyperTopo-Adapters: Geometry- and Topology-Aware Segmentation of Leaf Lesions on Frozen Encoders

**arXiv ID:** 2601.06067 | [PDF](https://arxiv.org/pdf/2601.06067v1)

**作者:** Chimdi Walter Ndubuisi `[一作]` (University of Missouri), Toni Kazic `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种轻量化的HyperTopo-Adapter头，在冻结的视觉编码器上训练，利用双曲、欧氏、球面三种曲率空间对特征进行编码，并结合可微软Euler约束与持久同调距离提升叶片病斑分割的拓扑准确性。

**💡 创新点**

首次将产品流形 ℍ⊕𝔼⊕𝕊 与可微软Euler约束以及双曲对比损失结合，既保持 Dice 约束又显著降低 Betti 误差，实现了兼顾像素与拓扑的分割方案。

**🔧 技术方法**

使用冻结的 DINOv2 编码器、Poincaré 球面映射、欧氏和球面投影、双曲对比损失、软Euler损失、持久同调距离评估及 top‑K Dice 检查点选择等技术。

**📊 数据集**

在 Kaggle Leaf Lesion 数据集（2940 张图像，512×512 分辨率）上进行训练与验证。

**📈 对比分析**

与 U‑Net、DINOv2 欧氏基线对比，HyperTopo 在 Dice 0.533/IoU 0.407 的基础上，β0、β1 与 PD 距离分别下降 9%、7% 及 0.01，显示出在保持像素重叠度的同时显著提升拓扑一致性。

**⚠️ 局限性**

受限于对温度对比约束的精细调优、持久同调计算的高成本以及对不同编码器通用性的进一步验证需求。

---

## 272. Function-Correcting Partition codes

**arXiv ID:** 2601.06450 | [PDF](https://arxiv.org/pdf/2601.06450v1)

**作者:** Charul Rajput `[一作]` (Aalto University), Camilla Hollanti `[通讯]` (Aalto University)

**通讯引用:** 1924 | [OpenAlex ID](https://openalex.org/A5035260653)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了函数分区纠错码（FCPC）概念，基于输入域的分区而非函数值，能够一次性对多种函数的输出实现错误保护，并给出了冗余和速率的理论最优表达式及相应的构造方法。

**💡 创新点**

创新点在于：①将函数纠错码从“值”视角迁移到“分区”视角，形成FCPC并涵盖传统FCC；②利用分区的“并（join）”可一次保护多个函数，提出分区收益（redundancy / rate gains）；③引入分区图与最大团（clique）来决定最优冗余；④引入块保持收缩（block‑preserving contraction）在无完整团时仍可简化搜索；⑤FCPC天然具备函数类别隐私。

**🔧 技术方法**

主要技术包括：分区编码、分区并集、图论（构造分区图、寻找团）、Plotkin界、线性函数的余子空间交叉（余子空间分区）、局部(2t,2)受限分区构造等。

**📊 数据集**

论文为理论性工作，未使用具体实验数据集，所有结果均基于符号域 𝔽_q^k 的通用性质与组合构造。

**📈 对比分析**

通过理论上限与下界比较，给出了冗余和速率的两侧界，并在若干实例（线性函数、多函数联合保护、局部二值函数、权重/支持分区）中证明了构造可达到最优或接近最优；示例显示在多函数保护场景下可实现数十%以上的冗余/速率节约。

**⚠️ 局限性**

局限性包括：某些分区（如非连续加权分区）不具备完整团，导致必须使用块保持收缩并且搜索复杂度仍高；对分区的知识需求较强，且对非受限分区的实际编码实现仍需进一步研究。

---

## 273. Detecting LLM-Generated Text with Performance Guarantees

**arXiv ID:** 2601.06586 | [PDF](https://arxiv.org/pdf/2601.06586v1)

**作者:** Hongyi Zhou `[一作]` (Tsinghua University), Chengchun Shi `[通讯]` (London School of Economics and Political Science)

**通讯引用:** 593 | [OpenAlex ID](https://openalex.org/A5025970743)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种检测大型语言模型（LLM）生成文本的分类器，旨在解决人类与LLM生成文本之间的区分问题。

**💡 创新点**

创新点在于该检测器不依赖于辅助信息（如水印或特定LLM的知识），能够更有效地区分人类和LLM生成的文本，并且支持统计推断。

**🔧 技术方法**

使用了基于统计的检测方法，结合了自适应学习的统计量，能够在不依赖特定模型的情况下进行分类。

**📊 数据集**

构建了一个包含超过10,000个文本的数据集，涵盖医疗、法律等多个领域，包含人类和多种流行LLM（如GPT系列、Grok和Gemini）生成的文本。

**📈 对比分析**

与现有的9种基线检测器进行比较，提出的检测器在各个领域的AUC值均接近1.0，表现优于所有基线方法，同时控制了I型错误率，保持了高统计功效。

**⚠️ 局限性**

限制在于该方法可能在面对不同的LLM或生成策略时表现不如预期，且对训练数据的多样性和质量依赖较大。

---

## 274. Assessing novice programmers' perception of ChatGPT:performance, risk, decision-making, and intentions

**arXiv ID:** 2601.06044 | [PDF](https://arxiv.org/pdf/2601.06044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 275. A Bayesian Network-Driven Zero Trust Model for Cyber Risk Quantification in Small-Medium Businesses

**arXiv ID:** 2601.06553 | [PDF](https://arxiv.org/pdf/2601.06553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 276. Structure-Aware Diversity Pursuit as an AI Safety Strategy against Homogenization

**arXiv ID:** 2601.06116 | [PDF](https://arxiv.org/pdf/2601.06116v1)

**作者:** Ian Rios-Sialer `[一作]` `[通讯]` (Independent), Ian Rios-Sialer (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“异种再生产（xeno-reproduction）”概念，并构建结构感知多样性追求的理论框架，用以在大语言模型（LLM）生成过程中对抗同化（homogenization）问题。

**💡 创新点**

创新点在于将多样性问题与结构合规性（structure compliance）、系统核心（system core）及偏离度（deviance）等量化指标结合，形成可操作的多层次评价体系，并通过分布层与轨迹层两种公式化方法实现对干预策略的搜索。

**🔧 技术方法**

主要技术包括：1) LLM生成树建模与字符串统计；2) 结构合规性向量化与聚合；3) 定义核心熵与偏离度，用于衡量同化与多样性；4) 采用贝叶斯/强化学习风格的干预搜索（π(w) ∝ exp(β·χ(w))）。

**📊 数据集**

论文为理论性工作，没有使用具体数据集；框架面向所有可用于训练或推理的文本数据，后续实验需基于公开LLM训练语料或自定义语料库。

**📈 对比分析**

由于缺乏实验验证，论文主要通过理论推导和数学证明展示两种干预公式的有效性；未给出具体数值性能对比，而是指出不同λ权重会在多样性与公平性之间产生权衡。

**⚠️ 局限性**

局限性包括：1) 结构规范与合规度的选择依赖主观判断；2) 计算系统核心和偏离度在大规模模型上不可行，需要近似或采样方法；3) 具体实现与参数调优尚未给出；4) 可能无法完全避免技术导致的负面强化或歧视风险。

---

## 277. $\texttt{AMEND++}$: Benchmarking Eligibility Criteria Amendments in Clinical Trials

**arXiv ID:** 2601.06300 | [PDF](https://arxiv.org/pdf/2601.06300v1)

**作者:** Trisha Das `[一作]` (University of Illinois Urbana-Champaign), Jimeng Sun `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 27485 | [OpenAlex ID](https://openalex.org/A5084279065)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了“临床试验资格标准修订预测”这一新的 NLP 任务，并构建了对应的基准数据集和改进的预训练方法。

**💡 创新点**

创新点在于：①首次将试验资格标准修订预测框定为机器学习问题；②发布两大规模数据集（含原始和 LLM 去噪标签）；③提出利用历史修订信息的 Change-Aware Masked Language Modeling（CAMLM）预训练策略。

**🔧 技术方法**

技术手段包括：大语言模型（LLM）对资格标准差异进行去噪标注；变体掩码语言模型 CAMLM；以及多种 Transformer 编码器（BioBERT、BERT、Longformer）和经典分类器（LR、RF）进行下游预测。

**📊 数据集**

数据集：从 ClinicalTrials.gov 抽取的两份数据集——完整版（161,970 例）和 LLM 去噪版（64,641 例），均包含所有资格标准版本历史及修订标签。

**📈 对比分析**

与传统 MLM、Span‑MLM 等预训练方式以及不预训练的 BioBERT 进行对比，CAMLM 在 AUROC、AUPRC、准确率上均提升 1–2%（统计显著），且对不同 backbone 与分类器均保持一致的性能提升。

**⚠️ 局限性**

局限性：①仅聚焦资格标准的修订，其他协议章节未纳入；②假设 ClinicalTrials.gov 的 0 标签无误，未考虑潜在误报导致的噪声。

---

## 278. Low-Back Pain Physical Rehabilitation by Movement Analysis in Clinical Trial

**arXiv ID:** 2601.06138 | [PDF](https://arxiv.org/pdf/2601.06138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. LSRIF: Logic-Structured Reinforcement Learning for Instruction Following

**arXiv ID:** 2601.06431 | [PDF](https://arxiv.org/pdf/2601.06431v1)

**作者:** Qingyu Ren `[一作]` (Fudan University), Fei Yu `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了逻辑结构化训练框架，显式建模指令中的并行、顺序与条件逻辑。

**💡 创新点**

创新点在于将逻辑结构融入数据构造与奖励设计，通过结构感知奖励实现对指令逻辑的精准优化。

**🔧 技术方法**

技术包括基于GRPO的结构感知奖励建模、GPT‑4.1生成多约束指令、注意力参数和token级可解释性分析。

**📊 数据集**

使用了自建的多约束指令数据集（并行、顺序、条件），并利用Infinity‑Instruct、Open Assistant、Self‑Instruct等源。

**📈 对比分析**

与基线模型（如GPT‑4o、RAIF‑7B等）对比，实验显示在多种在域和跨域指令跟随、逻辑推理和通用能力上均获得显著提升。

**⚠️ 局限性**

局限性包括未在70B+大模型上验证、数据主要为英文，跨语言泛化尚待进一步探索。

---

## 280. GRASP LoRA: GRPO Guided Adapter Sparsity Policy for Cross Lingual Transfer

**arXiv ID:** 2601.06702 | [PDF](https://arxiv.org/pdf/2601.06702v1)

**作者:** Besher Hassan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiuying Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1284 | [OpenAlex ID](https://openalex.org/A5101568165)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于GRASP LoRA的参数高效微调方法，自动学习跨语言迁移时的全局稀疏比例；

**💡 创新点**

创新点在于将稀疏比例视为可学习的控制变量，使用GRPO控制器在训练过程中在线更新全局剪枝率，避免传统网格搜索；

**🔧 技术方法**

核心技术包括LoRA适配器的合并、基于重要性评分的掩码构造、GRPO策略网络的均值锚定与熵奖励，以及稀疏比例的有限步更新；

**📊 数据集**

实验使用Llama 3 8B模型，跨语言任务为XL‑Sum摘要和MLQA抽取式问答，源语言为英语，目标语言为阿拉伯语和中文；

**📈 对比分析**

与零射击、单独目标LoRA、直接合并、TIES、LoRAHub及网格搜索合并+剪枝基线相比，GRASP LoRA在两项任务上均实现最高的BERTScore、BLEU/ROUGE和EM/F1指标，并将训练时间缩短约4–7倍；

**⚠️ 局限性**

局限性包括仅在单一模型与单一硬件上验证，任务与语言覆盖有限，未评估部署时延、内存和能耗，以及缺乏对偏见与公平性的系统评估。

---

## 281. The Psychology of Learning from Machines: Anthropomorphic AI and the Paradox of Automation in Education

**arXiv ID:** 2601.06172 | [PDF](https://arxiv.org/pdf/2601.06172v1)

**作者:** Junaid Qadir `[一作]` (Qatar University), Muhammad Mumtaz `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 2150 | [OpenAlex ID](https://openalex.org/A5044710934)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过整合自动化心理学、人因工程、HCI 与技术哲学等四个研究传统，构建了一个框架，用于研究学习者与具人性化特征的 AI 导师的心理互动，并对 104,984 条 YouTube 评论进行自然语言和情感分析，比较 AI 生成哲学辩论与人工创建工程教程在信任、情感与参与模式上的差异。

**💡 创新点**

创新点在于：①首次将多学科的自动化心理学与教育技术研究系统性整合，揭示信任失调、拟人化悖论和自动化讽刺三大心理挑战；②基于大规模真实用户评论进行跨领域对比，验证了领域依赖的信任校准与拟人投射；③提出了可操作的设计原则和保护性准则，用于指导工程教育中的 AI‑人协同教学。

**🔧 技术方法**

采用的技术包括：自然语言处理（VADER 情感分析）、基于规则的评论行为特征提取（如情感广播、分析性发言等）、时间序列情感演化模型、统计检验（χ²、Cramér's V）以及比较分析框架。

**📊 数据集**

使用的数据集是从两类 YouTube 频道收集的 104,984 条评论：AI 生成的哲学辩论（72,178 条）和人类创作的工程教程（32,806 条），涵盖不同主题、订阅量和发布时间。

**📈 对比分析**

通过对比分析，本文发现：①AI 讨论在情感上更中性、参与更深层；②工程教程更倾向正面情绪、即时互动；③在信任校准上，AI 讨论表现出更高的隐式接受度但更少的显式质疑。虽然未给出传统意义上的性能指标，但这些定量指标表明不同内容类型对学习者情绪与信任的影响存在显著差异。

**⚠️ 局限性**

限制包括：①数据仅为英文评论，可能缺乏跨文化视角；②依赖规则与情感词典的 NLP 方法，可能错判讽刺与细腻情绪；③未进行人工编码验证，缺少人类标注的可靠性检验；④评论数据仅反映在线观看体验，未能覆盖实际课堂互动与学习成效。

---

## 282. Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization

**arXiv ID:** 2601.06224 | [PDF](https://arxiv.org/pdf/2601.06224v1)

**作者:** Miao Pan `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5047459900)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过引入计划+字幕阶段、奖励方差样本选择以及基于NTK的对抗正则化，对RL训练的多模态大型语言模型进行可视化基础推理增强，从而系统性地减少幻觉并提升推理准确率。

**💡 创新点**

创新点包括：① 通过Caption Reward衡量视觉文本一致性；② 用奖励方差挑选中等难度样本以促进探索多样性；③ 通过NTK相似度阈值化InfoNCE正则化降低样本冲突。

**🔧 技术方法**

采用GRPO强化学习、Caption Reward、奖励方差样本筛选、NTK相似度与InfoNCE对抗正则化技术。

**📊 数据集**

在MMVU、VideoMMMU、VideoHallucer、MathVista、POPE和MMBench等多模态基准集上进行评测。

**📈 对比分析**

与多种开源与闭源基线（如GPT‑4o、Gemini‑1.5‑Pro、LLaVA、MiniCPM等）对比，模型在MMVU 65.6%、VideoHallucer 50.8%、POPE 88.7%和MMBench 88.6%等任务上实现了开放源模型的最优性能。

**⚠️ 局限性**

局限性包括对NTK阈值等超参数敏感、需要进一步跟踪长文本推理漂移、以及在更广泛的视觉后端与指令跟随任务中的适配难度。

---

## 283. Self-Admitted Technical Debt in LLM Software: An Empirical Comparison with ML and Non-ML Software

**arXiv ID:** 2601.06266 | [PDF](https://arxiv.org/pdf/2601.06266v1)

**作者:** Niruthiha Selvanayagam `[一作]` (École de technologie supérieure), Taher A. Ghaleb `[通讯]` (Trent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型（LLM）系统中的自报技术债（SATD）进行了首次系统性实证研究，并与传统机器学习（ML）和非ML项目进行了对比。

**💡 创新点**

创新点包括：①识别并定义了三种LLM专属债务类型（模型堆栈变通债务、模型依赖债务、性能优化债务）；②对SATD生命周期进行纵向存活分析；③将SATD分布映射至LLM开发流水线不同阶段。

**🔧 技术方法**

主要技术手段包括：基于NLP的SATD检测器、GitHub API数据抓取、Kaplan–Meier生存分析、Wilcoxon秩和检验与Bonferroni校正、随机森林预测模型。

**📊 数据集**

使用的数据集为：159 个后 2022 年推出的 LLM 开源仓库，159 个 ML 仓库以及 159 个非 ML 仓库（均来自 Bhatia 等人 2021 年的公开数据），共计约 3.7 亿元评论事件。

**📈 对比分析**

通过统计检验和生存曲线对三类仓库的 SATD 发生率、存活时间和移除率进行比较，发现 LLM 与 ML 的发生率相近但 LLM 的债务自由期更长、移除率更低；随机森林模型在 ML 上达 96% 预测准确率，LLM 上 82%，非 ML 上 74%。

**⚠️ 局限性**

主要局限包括：仅覆盖 Python 开源 GitHub 项目；仅检测显式 SATD，忽略隐性债务；仓库年龄差异导致对比偏倚；检测器误报率高且需人工筛选；流水线阶段映射主要依赖手工标注。

---

## 284. RiskBridge: Turning CVEs into Business-Aligned Patch Priorities

**arXiv ID:** 2601.06201 | [PDF](https://arxiv.org/pdf/2601.06201v1)

**作者:** Yelena Mujibur Sheikh `[一作]` (Binghamton University), Zhaohan Xi `[通讯]` (Binghamton University)

**通讯引用:** 111 | [OpenAlex ID](https://openalex.org/A5026309535)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了 RiskBridge 框架，集成多源威胁情报（NVD、EPSS、CISA KEV 等），通过零日曝光模拟、合规 Policy‑as‑Code 引擎和 ROI 优化模块，生成可解释、合规驱动的补丁优先级列表。

**💡 创新点**

创新点在于：① 将概率零日曝光模型与业务影响指数（BII）结合，实时评估漏洞被利用的可能性；② 采用 Policy‑as‑Code 把 PCI‑DSS、NIST 等法规直接映射为可执行的 SLA 逻辑；③ 用加权集合覆盖实现 ROI 最大化，显著提升补丁效益与合规性；④ 通过 LLM 辅助推理提供可视化解释，增强决策透明度。

**🔧 技术方法**

技术手段包括：概率建模（ZDES）、规则引擎（Policy‑as‑Code）、加权集合覆盖优化、机器学习（EPSS 预测）、LLM（GPT‑4/Gemini 辅助推理）、容器化数据管道、可视化仪表盘。

**📊 数据集**

实验使用公开数据集：NVD CVE 数据、EPSS 预测概率、CISA KEV 真实被利用记录、资产映射表与公开威胁源；在多组公开数据上进行基准对比。

**📈 对比分析**

与 CVSS‑only、EPSS、KEV 匹配、Tenable VPR 等基线对比，评估指标包括 Precision@K、F1、Compliance Gain（SLA 提升天数）、Optimization Efficiency、ROI（风险/小时）。RiskBridge 在所有指标上均表现最佳：风险降低 88%，SLA 提升 18 天，ROI 提升 4.7 risk/hr，整体性能显著优于商业基线。

**⚠️ 局限性**

局限性包括：尚未集成实时 SOC 反馈与组织内流量，导致零日预测可能受限；缺乏自适应学习与持续反馈机制；依赖外部情报源，对极少数零日场景的精度不足；未来需进一步与 SIEM、MITRE ATT&CK、FAIR 等框架对接，以提升规模化部署与跨组织协同能力。

---

## 285. CSR-RAG: An Efficient Retrieval System for Text-to-SQL on the Enterprise Scale

**arXiv ID:** 2601.06564 | [PDF](https://arxiv.org/pdf/2601.06564v1)

**作者:** Rajpreet Singh `[一作]` (Technical University of Munich), Manzoor A. Khan `[通讯]` (Nokia Bell Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种混合检索增强生成（RAG）系统，用自然语言问题检索企业规模数据库中的相关表和列并生成SQL查询

**💡 创新点**

创新点在于同时结合上下文检索、结构化知识图谱检索和关系超图排名，实现高精度多表连接检索

**🔧 技术方法**

使用BERT编码的语义检索、数据库知识图谱结构表示、超图关系建模与相似度排名等技术

**📊 数据集**

使用匿名企业数据库（包含四组表结构）与Spider、BIRD等公开基准进行实验对比

**📈 对比分析**

相较于现有方法，精度超过40%、召回率超过80%，平均生成延迟仅30 ms，性能显著提升

**⚠️ 局限性**

局限包括需要人工调参、对极其复杂关系检索仍可能误检，以及仅在单一企业数据集上验证

---

## 286. Perception Test 2025: Challenge Summary and a Unified VQA Extension

**arXiv ID:** 2601.06287 | [PDF](https://arxiv.org/pdf/2601.06287v1)

**作者:** Joseph Heyward `[一作]` (Google DeepMind), Viorica Pătrăucean `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织2025年Perception Test挑战赛，统一多任务（对象/点跟踪、动作/声音定位、统一多选视频QA、定位视频QA、小时级视频QA）并引入新的统一多选视频QA数据集，评估大型多模态模型在这些任务上的表现。

**💡 创新点**

创新点在于将传统的定位、跟踪等任务通过视觉填充与时间标记转化为VQA形式，实现任务统一；提供包含4类新问题（点跟踪、动作排序、物体-动作交互、假动作）的统一多选视频QA子集；以及首次在挑战中引入“小时级视频QA”评估。

**🔧 技术方法**

主要技术为大规模多模态语言模型（Gemini、GPT‑4V、Qwen‑VL、Qwen‑2.5VL等）以及结合SAM2、AllTracker、TAG等模型的多模型或统一模型管道；使用Eval.ai平台进行分阶段评测，并采用多种评估指标（top‑1准确率、平均Jaccard、mAP、HOTA）。

**📊 数据集**

使用的数据集为原始Perception Test基准（视频+多选QA），新加入的统一多选视频QA样本（1842道题），以及Walking Tours长视频集（小时级QA）。

**📈 对比分析**

与随机、频率、静态基线以及往年提交模型对比；零射手模型在统一多选QA上达到0.812，精调模型0.810；对象/点跟踪最高平均Jaccard 0.62；动作/声音定位最高mAP 0.52；定位视频QA最高HOTA 0.499；小时级视频QA最高top‑1 0.78。整体表现比上一年显著提升，尤其在有语言接口的轨道。

**⚠️ 局限性**

局限性包括：多任务统一模型在对象/点跟踪方面仍未超越专门模型；部分抽象与物理类问题准确率低于50%；统一多选QA新增的5选项仍可能存在shortcut；长视频QA仍依赖大模型的生成与链式推理，受算力与模型可解释性的限制。

---

## 287. Teachers' Perspectives on Integrating AI tools in Classrooms: Insights from the Philippines

**arXiv ID:** 2601.06043 | [PDF](https://arxiv.org/pdf/2601.06043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 288. AI-Powered Algorithms for the Prevention and Detection of Computer Malware Infections

**arXiv ID:** 2601.06219 | [PDF](https://arxiv.org/pdf/2601.06219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 289. Toward Generalizable Deblurring: Leveraging Massive Blur Priors with Linear Attention for Real-World Scenarios

**arXiv ID:** 2601.06525 | [PDF](https://arxiv.org/pdf/2601.06525v1)

**作者:** Yuanting Gao `[一作]` (Tsinghua University), Kai Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 35778 | [OpenAlex ID](https://openalex.org/A5038484265)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了轻量级扩散模型 GLOWDeblur，结合 Blur Pattern Pretraining（BPP）和 Motion & Semantic Guidance（MoSeG），实现了对多种真实场景模糊的高质量恢复和强泛化能力。

**💡 创新点**

创新点包括：①将模糊模式多样性视为跨数据集泛化的核心，通过 BPP 在大规模仿真数据上预训练并迁移到真实数据；②在扩散框架中引入运动估计和跨模态文本语义指导，提升对极端模糊的恢复；③设计 32× 深度压缩 AutoEncoder 与线性注意力的轻量化 Diffusion 结构，实现了显著的速度和参数优势。

**🔧 技术方法**

使用技术主要包括：基于 UNet 的预重建与域对齐模块、SimpleGate 与 Simplified Channel Attention 的轻量化激活与注意力、32× 深度压缩 AutoEncoder、线性 DiT、运动场估计网络、Qwen2.5‑VL‑7B 生成的文本语义嵌入。

**📊 数据集**

训练数据涵盖合成数据（GoPro、HIDE、REDS、GSBlur、LSDIR）与真实数据（RealBlur、BSD、RSBlur），评估数据包括六大基准（GoPro、HIDE、REDS、RealBlur、BSD、RSBlur）以及两组真实世界数据（RWBI、RWBlur400）。

**📈 对比分析**

与现有最先进方法（HI‑Diff、MISC‑Filter、MLWNet、Restormer、DiffIR、FPro、Diff‑Plugin 等）比较，GLOWDeblur 在 PSNR/SSIM、MANIQA、LIQE、NRQM、CLIP‑IQA、PI、BRISQUE、NIQE、ILNIQE 等多种参考和无参考指标上均表现优异，特别是在跨数据集和真实世界场景下的恢复效果显著更好。

**⚠️ 局限性**

限制方面：①虽然已显著提升泛化，但对极端、复杂模糊仍可能出现细节失真；②模型在极高分辨率或极低延迟场景下仍不及传统轻量化网络；③BPP 仍依赖仿真数据的真实性，若仿真与真实分布差距过大，迁移效果受限。

---

## 290. Representing Sounds as Neural Amplitude Fields: A Benchmark of Coordinate-MLPs and A Fourier Kolmogorov-Arnold Framework

**arXiv ID:** 2601.06406 | [PDF](https://arxiv.org/pdf/2601.06406v1)

**作者:** Linfei Li `[一作]` (Tongji University), Ying Shen `[通讯]` (Tongji University)

**通讯引用:** 11362 | [OpenAlex ID](https://openalex.org/A5022675390)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文首先构建了首个基于Coordinate-MLP的开放式音频隐式表示基准，并提出了基于Fourier‑KAN的Fourier‑ASR框架，解决了位置编码与激活函数调参的难题。

**💡 创新点**

创新点在于通过三种位置编码与十六种激活函数共48种组合系统评估，揭示了激活函数对音频周期性捕捉的影响，并设计了无需位置编码、具备频率自适应学习策略的Fourier‑KAN。

**🔧 技术方法**

技术方法包括NeRF及随机Fourier位置编码、Sine/Gaussian等激活函数、Fourier‑KAN网络、FaLS频率自适应学习、SNR与LSD等评估指标。

**📊 数据集**

实验数据集为GTZAN音乐集、CSTR VCTK语音集以及SIREN提供的短音频片段。

**📈 对比分析**

在48种配置与Fourier‑ASR的对比中，Fourier‑ASR在SNR上相较Sine‑MLP提升约6dB，超过B‑Spline‑KAN约18dB，且整体性能优于大多数基线，且不需要繁琐调参。

**⚠️ 局限性**

局限性包括对KANN的优化策略尚不成熟，导致与局部周期性NeFF+Sine‑MLP相比略逊一筹，以及对频率阈值等超参数的进一步探索仍需深入。

---

## 291. Automated QoR improvement in OpenROAD with coding agents

**arXiv ID:** 2601.06268 | [PDF](https://arxiv.org/pdf/2601.06268v1)

**作者:** Amur Ghose `[一作]` (University of California San Diego), Jakang Lee `[通讯]` (University of California San Diego)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在OpenROAD物理设计工具链中，使用大型语言模型自动化生成、评估并提交代码变更，提升路由长度和时钟周期等质量指标。

**💡 创新点**

首次将结构化文档、基于文献的计划生成、以及闭环QoR反馈的Agent执行结合，构建端到端的自主代码改进流水线。

**🔧 技术方法**

使用Tree‑Sitter构建跨语言属性图、Docmaker自动生成文档卡片、DSPy编写的规划程序、Codex类Agent执行与工具链交互，并结合RAG与LM断言进行自我校正。

**📊 数据集**

基于OpenROAD开源仓库、ASAP7、SKY130HD、Nangate45等PDK公开设计（aes、ibex、jpeg、ariane133/136、bp_fe、swerv_wrapper等）。

**📈 对比分析**

通过与基准OpenROAD提交比较，实验显示路由线长可减少至-5.9%，有效时钟周期可减少至-10.0%，且改动仅需单一diff即可在多平台上复现。

**⚠️ 局限性**

受限于对高性能硬件（如ARM64）的兼容性不足、对更复杂功耗/面积指标的评估不足，以及Agent对极大代码库的可扩展性和安全回滚机制仍需进一步强化。

---

## 292. Deep Q-Network Based Resilient Drone Communication:Neutralizing First-Order Markov Jammers

**arXiv ID:** 2601.06095 | [PDF](https://arxiv.org/pdf/2601.06095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 293. AIS-CycleGen: A CycleGAN-Based Framework for High-Fidelity Synthetic AIS Data Generation and Augmentation

**arXiv ID:** 2601.06127 | [PDF](https://arxiv.org/pdf/2601.06127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 294. On the Adversarial Robustness of 3D Large Vision-Language Models

**arXiv ID:** 2601.06464 | [PDF](https://arxiv.org/pdf/2601.06464v1)

**作者:** Chao Liu `[一作]` (Singapore University of Technology and Design), Ngai-Man Cheung `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 5435 | [OpenAlex ID](https://openalex.org/A5057453537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了点云基础3D视觉‑语言模型的对抗鲁棒性，提出并评估了两种攻击方法——视觉攻击和文字攻击，并考虑无目标与有目标两种设置。

**💡 创新点**

首次系统评估3D VLM的对抗鲁棒性，设计了针对视觉特征与文本输出的两阶段攻击框架，并揭示3D模型相比2D模型更难实现精准目标攻击的原因。

**🔧 技术方法**

采用白盒梯度优化对抗攻击，利用余弦相似度和交叉熵损失分别扰动视觉投影特征与生成caption；引入关键点选择与高斯平滑的扰动正则化以保持几何可感知度。

**📊 数据集**

在 ModelNet40 与 Objaverse 两个公开点云数据集上进行实验，评估分类与captioning任务。

**📈 对比分析**

与未攻击基线以及不同攻击类型对比；在无目标攻击中取得超过80% ASR；目标攻击效果相对弱；GPT4Point 在两类任务上更易被攻击；相较于2D VLMs，3D VLMs 的目标攻击成功率更低。

**⚠️ 局限性**

仅针对白盒可微分模型，缺乏黑盒/查询限制评估；对商业专有模型的可访问性有限；未考虑多场景级3D VLMs 的鲁棒性。

---

## 295. AI-Assisted Authoring for Transparent, Data-Driven Documents

**arXiv ID:** 2601.06027 | [PDF](https://arxiv.org/pdf/2601.06027v1)

**作者:** Alfonso Piscitelli `[一作]` (University of Salerno), Chenyiqiu Zheng `[通讯]` (University College London)

**通讯引用:** 16029 | [OpenAlex ID](https://openalex.org/A5027041701)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种 AI 辅助的透明、数据驱动文档作者工具，能够把自然语言中的定量或比较性表达转化为可执行的查询并嵌入文档；

**💡 创新点**

将大型语言模型生成代码与 Fluid 语言的运行时数据血缘追踪结合，自动化文本到查询的映射，并支持交互式验证；

**🔧 技术方法**

使用 LLM（如 GPT‑4o）进行文本片段识别与 Fluid 代码合成，Fluid 语言实现数据追踪与交互，构建人机协作的工作流；

**📊 数据集**

在 SciGen 开放数据集（科学论文表格与描述）上进行评测，并手工构造反事实案例验证鲁棒性；

**📈 对比分析**

通过对不同语言习惯（平均值、百分比、极值、排名、趋势等）的成功率比较，系统在 74.9% 的情况下生成正确表达式，复杂度较高时成功率下降；在 300 次反事实测试中仅 42 次保持正确，显示鲁棒性有限；

**⚠️ 局限性**

局限包括对预定义辅助函数的依赖、对模糊或歧义表达的敏感、对复杂多步骤查询的低成功率、未覆盖近似量词和等级形容词等更细致的自然语言表达；

---

## 296. Cascading multi-agent anomaly detection in surveillance systems via vision-language models and embedding-based classification

**arXiv ID:** 2601.06204 | [PDF](https://arxiv.org/pdf/2601.06204v1)

**作者:** Tayyab Rehman `[一作]` (University of L'Aquila), Aly Shmahell `[通讯]` (University of L'Aquila)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个多级异步多代理框架，融合对象检测、自编码重建与视觉语言推理，实现实时可解释的异常检测。

**💡 创新点**

通过级联早退出与自适应阈值，使轻量模块完成常规检测，只有复杂事件才调用大型视觉语言模型；同时引入分布式代理通信与中止机制，兼顾实时性与可解释性。

**🔧 技术方法**

使用YOLOv8、卷积自编码器、基于CLIP/LLaVA的视觉语言模型与嵌入式原型分类，并采用Redis Pub/Sub实现多代理协作。

**📊 数据集**

在UCF‑Crime视频异常检测基准上进行评估，并对比了多种现有方法。

**📈 对比分析**

与直接使用VLM推理相比，级联方案平均帧延迟下降约3倍（从约8.7s降至2.6s），在UCF‑Crime上取得宏观F1≈0.72，PSNR 38.3dB、SSIM 0.965，整体性能优于单一轻量或VLM方法。

**⚠️ 局限性**

主要限制在于视觉语言阶段仍然是计算瓶颈（6–12s/帧），自编码器对光照变化敏感，阈值需人工调参，且跨数据集泛化尚未充分验证。

---

## 297. Supervised and Unsupervised Neural Network Solver for First Order Hyperbolic Nonlinear PDEs

**arXiv ID:** 2601.06388 | [PDF](https://arxiv.org/pdf/2601.06388v1)

**作者:** Zakaria Baba `[一作]` (Ecole Polytechnique), Benedetto Piccoli `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于神经网络的有限体积方法（NFV），用来学习并近似一维守恒律的数值通量函数，从而构造出既能保持质量守恒又能自适应学习的第一阶及高阶数值方案。

**💡 创新点**

创新点在于将数值通量函数的计算完全交给神经网络完成，支持在空间和时间上使用更大支架（可多步自回归、时间维度输入），大幅提升了传统有限体积方法的精度，甚至在某些场景下超越了更高阶的 ENO/WENO 方案，逼近 DG 的表现。

**🔧 技术方法**

核心技术包括多层感知机（MLP）/卷积神经网络实现数值通量函数，监督训练（最小化合成或真实数据误差）和无监督训练（最小化弱形式残差），以及自回归预测和差分进化求解参数化流函数；同时使用神经网络可视化数值通量与传统 Godunov 通量的形状。

**📊 数据集**

使用了两个主要数据集：① 通过 LWR、Burgers PDE 生成的合成交通流数据；② Berkeley DeepDrive 航拍高速公路视频中提取的车流轨迹数据，用于计算真实密度并构造实测基本图（fundamental diagram）。

**📈 对比分析**

与传统有限体积（Godunov、Lax–Friedrichs、Engquist–Osher）、高阶 ENO/WENO 以及有限元 DG 进行对比。结果显示 NFVM_2^1/UNFVM_2^1 在 L1、L2、相对误差上均显著优于所有第一阶方法，部分情况甚至优于 ENO/WENO；NFVM_4^5 在精度上与 DG 接近，误差比传统方案低一个数量级，性能在精度上大幅提升。

**⚠️ 局限性**

局限性包括：① 通量函数不保证单调性，缺乏传统的收敛理论；② 对非理想（噪声大、非守恒）真实数据的鲁棒性尚未完全验证；③ 仍需手动确保质量守恒约束，且对极端边界/初始条件的泛化能力有限；④ 在计算效率上虽接近传统 FVM，但与 DG 的性能差距仍存在。

---

## 298. NL2Dashboard: A Lightweight and Controllable Framework for Generating Dashboards with LLMs

**arXiv ID:** 2601.06126 | [PDF](https://arxiv.org/pdf/2601.06126v1)

**作者:** Boshen Shi `[一作]` (Jiutian Research), Junlan Feng `[通讯]` (Jiutian Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出NL2Dashboard框架，利用结构化中间表示(IR)实现自然语言到可交互仪表盘的高效生成和修改；

**💡 创新点**

核心创新是将分析与呈现解耦，使用IR驱动的两阶段流程（Prompt→IR→Dashboard）以及多代理系统实现精准可控的编辑；

**🔧 技术方法**

采用LLM生成分析结果、Python脚本执行、IR构造工具（O^G、O^R、O^M）、视觉‑语言模型评估、以及多代理协同（Planner、Coder、Critic）等技术；

**📊 数据集**

使用十张来自金融、教育、政府等领域的真实业务表格作为生成与修改实验数据集；

**📈 对比分析**

与Doubao、Gemini 2.5 pro、GPT5等基线通过仪表盘质量、token效率（GOR）和修改成功率(SR)对比，NL2Dashboard在所有指标上均显著优于基线，尤其在信息丰富度、控制精度和token占用方面提升超过30%；

**⚠️ 局限性**

局限在IR模板覆盖度与复杂交互（如动态筛选、实时更新）方面仍需扩展，以及对更大规模、非结构化数据的适配需要进一步验证。

---

## 299. The Environmental Impact of AI Servers and Sustainable Solutions

**arXiv ID:** 2601.06063 | [PDF](https://arxiv.org/pdf/2601.06063v1)

**作者:** Aadi Patel `[一作]` (Rutgers University), Rusheen Patel `[通讯]` (Rutgers University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估 AI 服务器对电力、水资源和碳排放的环境影响，并基于文献回顾与量化预测提出多维度可持续发展方案

**💡 创新点**

结合全球能源需求预测、区域电网碳强度差异以及先进冷却技术与可再生能源微网的协同效应，首次系统性评估并量化不同策略对 AI 数据中心环境足迹的综合影响

**🔧 技术方法**

文献回顾、定量预测模型、案例分析、与传统冷却技术及不同地理位置的对比评估

**📊 数据集**

国际能源署 (IEA)、洛杉矶贝克利国家实验室 (LBNL)、高盛 (Goldman Sachs) 的能源需求与排放数据，美国水电消耗统计，以及 Microsoft Project Natick、Meta StatePoint 等实际部署案例

**📈 对比分析**

通过与传统空冷、传统水冷系统以及不同区域电网条件的对比，展示液冷/直接芯片冷却可降低 40–50% 能耗、20–90% 水耗，结合可再生微网可进一步降低 50–80% 碳排放，验证了多维度整合方案的显著性能提升

**⚠️ 局限性**

依赖公开报告与模型预测，缺乏大规模运营数据；未来 AI 负载增长率、区域电网碳强度变化、数据中心内部运营细节的不确定性，以及不同地理位置基础设施的可行性限制了结论的普适性

---

## 300. Sports Business Administration and New Age Technology: Role of AI

**arXiv ID:** 2601.06053 | [PDF](https://arxiv.org/pdf/2601.06053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 301. Boosting Overlapping Organoid Instance Segmentation Using Pseudo-Label Unmixing and Synthesis-Assisted Learning

**arXiv ID:** 2601.06642 | [PDF](https://arxiv.org/pdf/2601.06642v1)

**作者:** Gui Huang `[一作]` (Sun Yat-sen University), Mengting Liu `[通讯]` (Cixi Institute of Biomedical Engineering)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于合成辅助半监督学习（SA‑SSL）的胚系类器官实例分割框架，结合伪标签去混叠（PLU）技术以纠正重叠实例错误。

**💡 创新点**

创新点在于：① 伪标签去混叠模块精准分解重叠实例；② 使用轮廓图像而非掩膜进行图像合成，显著提升合成效率和分布一致性；③ 在伪标签合成前加入实例级增强并通过FID评估最优增强策略；④ 统一的EMA教师‑学生循环提升模型鲁棒性。

**🔧 技术方法**

技术手段包括 Mask R‑CNN（检测+分割）、教师‑学生半监督学习、pix2pixHD GAN 轮廓到图像合成、PLU 的重叠判断与分解分支、实例级增广、Fréchet Inception Distance（FID）评估、指数滑动平均（EMA）权重传递。

**📊 数据集**

数据集：官方公开的 OrgaSegment（231 张图像）和自建的 M‑OrgaQuant（1,752 张图像，19,499 个实例），两者均涵盖严重重叠与多样形态的器官。

**📈 对比分析**

与全监督、传统 SSL（Self‑Training、Noisy Boundary、Polite Teacher、PAIS）以及 Transformer 基线（MaskFormer）比较，SA‑SSL 在 mAP、F1、Dice、AJI 上均取得领先，并且仅使用 10% 标注数据即可逼近完全监督性能。

**⚠️ 局限性**

局限性：在极低标注比例（≤1%）下仍会出现伪标签质量不足导致性能下降；模型目前基于 CNN 架构，迁移到 Transformer 需要进一步验证；对教师模型的初始化依赖较大，需足够标注数据以保证伪标签的可靠性。

---

## 302. Symplectic Hulls over a Non-Unital Ring

**arXiv ID:** 2601.06609 | [PDF](https://arxiv.org/pdf/2601.06609v1)

**作者:** Anup Kushwaha `[一作]` (Indian Institute of Technology Patna), Om Prakash `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 2166 | [OpenAlex ID](https://openalex.org/A5032501326)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了非单位环 E 上的对称辛包络（左、右、双侧），求出了其剩余码、扭曲码、生成矩阵以及辛包络秩，并探讨了自由 E 线性码的和、LCD 性质、构造方法、排列等价与辛包络变形问题，最终对长度至 4 的最优自由 E 线性码进行分类。

**💡 创新点**

首次在非单位环上定义并系统研究辛包络，给出了剩余码与扭曲码的关系、生成矩阵的显式构造、辛包络秩的计算方法，提出了两种可扩展长度和辛包络秩的 Build‑Up 构造，并分析排列等价对辛包络秩的影响，填补了非单位环码理论与辛包络理论的空白。

**🔧 技术方法**

采用环论与线性码的代数方法，利用辛内积、矩阵乘积、模 J 的剩余映射、张量积与欧拉特征、以及 MAGMA 计算验证，结合对称辛双射与双侧对称性推导生成矩阵与秩关系。

**📊 数据集**

没有使用外部实验数据集，全部结果均来自理论推导与符号计算（MAGMA 计算验证）。

**📈 对比分析**

本文未与其他算法进行数值性能比较，重点是理论性质与构造方法的证明；所给构造在理论上可实现任意长度的自由 E 码，但未给出具体实现或复杂度分析。

**⚠️ 局限性**

局限性包括仅讨论了特定的非单位环 E，结果仅在长度 ≤ 4 的最优码上进行了完整分类，且对大规模码的实际构造与性能尚未验证；对排列等价的辛包络秩变化只给出了特殊情况，缺乏更一般的判定条件。

---

## 303. Physics-Informed Tree Search for High-Dimensional Computational Design

**arXiv ID:** 2601.06444 | [PDF](https://arxiv.org/pdf/2601.06444v1)

**作者:** Suvo Banik `[一作]` (Argonne National Laboratory), Subramanian Sankaranarayanan `[通讯]` (Argonne National Laboratory)

**通讯引用:** 9927 | [OpenAlex ID](https://openalex.org/A5063950942)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出一种物理信息驱动的蒙特卡洛树搜索（Physics‑Informed MCTS）用于高维连续黑盒优化，并将其应用于晶体结构预测、势能模型拟合以及连续介质工程设计。

**💡 创新点**

创新点包括：将方向学习与逻辑回归采样器嵌入树搜索；使用深度感知的窗口缩放实现全局探索与局部精细化；通过全局‑局部树族和自适应切换实现多样化搜索，提升对高维多峰景观的适应性。

**🔧 技术方法**

技术包括改进的MCTS框架、方向性逻辑回归采样器、拉丁方/超球体采样、窗口缩放、树族层级调度、局部优化器以及物理约束处理模块。

**📊 数据集**

数据集涵盖：30维多峰/固定维度测试函数；Au 35原子纳米簇（105维）；Si 2D/3D晶体结构；Al 原子簇DFT 数据；焊接梁与压载器四维工程约束。

**📈 对比分析**

对比方法有GA、DE、PSO、WOA、BO等；在所有基准函数上，Physics‑Informed MCTS 获得与最优者相当或更优的收敛、低方差；在Au簇中 12k 评估找到全局最优；在 Al 势能拟合中能量误差 53 meV/原子，优于现有模型。

**⚠️ 局限性**

限制包括：仅在纯黑盒环境下工作，未利用多精度或可微信息；树学习未跨任务迁移；约束处理仅通过惩罚，缺乏生成式或符号约束模型；对极端大维度（>200 维）或实时实验场景的适应性待验证。

---

## 304. The Axiom of Consent: Friction Dynamics in Multi-Agent Coordination

**arXiv ID:** 2601.06692 | [PDF](https://arxiv.org/pdf/2601.06692v1)

**作者:** Murad Farzulla `[一作]` (King's College London), Murad Farzulla `[通讯]` (Dissensus AI)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文基于同意公理（actions affecting agents require authorization from those agents in proportion to stakes）推导出统一的摩擦框架，构建了对齐（α）、利益（σ）与熵（ε）三元组，并给出了摩擦方程 F = σ·(1+ε)/(1+α)。该框架被用来解释多智能体系统、加密货币治理以及政治体系中的协调摩擦与失败。

**💡 创新点**

创新点包括：
① 从单一同意公理出发，系统性地导出了摩擦与协调的数学描述；
② 将摩擦与进化动力学（ROM）关联，证明同意匹配的配置是演化吸引子；
③ 提出跨尺度粗粒化方法，证明该框架在不同层次与领域均保持一致；
④ 给出可观测的指标体系，为实证检验提供操作化路径。

**🔧 技术方法**

技术手段包括：
- 形式化推导与偏导数分析；
- 复制‑优化机制（replicator‑mutation dynamics）；
- 信息熵与相关性度量；
- 理论证明（不可能性定理、收敛性与稳定性）。

**📊 数据集**

实验与案例采用的数据集有：
- 多智能体强化学习实验数据；
- 加密货币市场事件研究（区块链交易与价格波动）；
- 政治与治理数据（CSES 选民偏好、GDELT/ACLED 事件、Open Government Partnership 透明度指数等）；
- 问卷与行为记录，用于估计 α 与 ε。

**📈 对比分析**

比较方法与性能：
- 在高对齐率下，系统收敛速度快、摩擦低，符合理论预测；
- 在低对齐率或高熵情境下，出现死锁、资源浪费或价格剧烈波动；
- 与传统效用最大化或机制设计模型相比，摩擦框架更能解释实际摩擦与失败；
- 通过方程 F 与实测摩擦（如市场波动率、治理失灵率）的回归验证，模型解释力显著提升。

**⚠️ 局限性**

局限性：
- α、σ、ε 为潜变量，需多指标推断，估计误差可能放大；
- 模型假设连续时间、可微及无外部冲击，现实中可能失效；
- 高熵或 α 接近 -1 时方程趋于无穷，实际应用受限；
- 跨领域参数映射需要经验校准，泛化能力受限；
- 数据稀缺或偏差会影响估计与验证。

---

## 305. Can we Improve Prediction of Psychotherapy Outcomes Through Pretraining With Simulated Data?

**arXiv ID:** 2601.06159 | [PDF](https://arxiv.org/pdf/2601.06159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 306. VVTRec: Radio Interferometric Reconstruction through Visual and Textual Modality Enrichment

**arXiv ID:** 2601.06475 | [PDF](https://arxiv.org/pdf/2601.06475v1)

**作者:** Kai Cheng `[一作]` (Hong Kong University of Science and Technology), Qiong Luo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 6770 | [OpenAlex ID](https://openalex.org/A5034823616)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种多模态电波干涉图像重建方法 VVTRec，将稀疏可见度信号转换为视觉和文本两种扩展模态，并结合外部视觉‑语言模型进行知识融合以提升图像质量。

**💡 创新点**

创新点在于：①通过模态转换将可见度映射为图像‑形式和文本‑形式的特征；②构建样本‑特定知识库，利用预训练视觉‑语言模型在两种陌生模态间实现无监督的知识提取和融合；③仅冻结大部分参数，实现训练‑无成本的性能提升。

**🔧 技术方法**

核心技术包括：1）可见度→图像模态与可见度→文本模态的卷积与插值转换；2）视觉‑知识生成器 (VKG)、文本‑知识生成器 (IKG) 与可见度查询生成器 (VQG)；3）基于Transformer的跨模态注意力知识库；4）条件重建器 (CR) 通过 MLP 对可见度与融合特征进行联合建模。

**📊 数据集**

使用四个公开天文数据集：Merging Galaxies (MG)、In-between Round Smooth Galaxies (IRSG)、Unbarred Tight Spiral Galaxies (UTSG)、Edge-on Galaxies with Bulge (EGB)，采样方式与 8‑台 EHT 相似。

**📈 对比分析**

与 CLEAN、U-Net、Radionets、NF、VisRec、PolarRec 等传统与最新方法对比，在 PSNR 与 SSIM 上均取得显著提升（平均 PSNR 提升约 1.9dB，SSIM 超过 0.02），且在不同预训练模型（ViLT、CLIP、BLIP）下均保持较低参数量与内存占用，训练速度也不逊色。

**⚠️ 局限性**

局限性包括：①仍需依赖视觉‑语言预训练模型，若这些模型缺乏与电波干涉相关知识可能影响效果；②对极端稀疏或高噪声可见度的鲁棒性尚未充分验证；③当前方法主要针对电波干涉，扩展到其它物理领域时需要进一步调优。

---

## 307. Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models

**arXiv ID:** 2601.06162 | [PDF](https://arxiv.org/pdf/2601.06162v1)

**作者:** Kaiyuan Deng `[一作]` (University of Arizona), Xiaolong Ma `[通讯]` (University of Arizona)

**通讯引用:** 7466 | [OpenAlex ID](https://openalex.org/A5074448953)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于闭式优化的ScaPre框架，用于在文本到图像扩散模型中大规模、精准地遗忘指定概念，且不牺牲生成质量。

**💡 创新点**

创新点包括：①冲突感知稳定设计，融合谱迹正则化与几何对齐以抑制概念间冲突；②Informax Decoupler利用互信息动态筛选与目标概念相关的参数，精确限制更新范围；③整体实现为闭式解，无需额外训练或辅助模块，显著提升可扩展性与效率。

**🔧 技术方法**

核心技术包括谱迹正则化、几何对齐（Bures距离）、Informax Decoupler（互信息估计）、Sylvester方程闭式求解以及基于Bures流形的近似几何对齐。

**📊 数据集**

实验使用Stable Diffusion v1.4/v1.5，在Imagenette、ImageNet‑Diversi50、ImageNet‑Confuse5、I2P以及自制的艺术家风格数据集上评估，另外用MS‑COCO评估生成质量。

**📈 对比分析**

与FMN、SPM、ESD、MACE、UCE、RECE等方法对比，ScaPre在多概念遗忘（可达50个概念）下实现显著更低的遗忘准确率、最高的UQ分数，并在精确遗忘、艺术风格遗忘和效率指标上均优于现有方法。

**⚠️ 局限性**

限制：当前仅在文本到图像扩散模型的跨概念遗忘场景下验证，可能对更复杂的多模态或实时生成任务的适用性尚未充分探测；此外，虽然不需要额外数据，但对谱矩阵与互信息估计的计算仍需一定资源。

---

## 308. ISMS-CR: Modular Framework for Safety Management in Central Railway Workshop

**arXiv ID:** 2601.06046 | [PDF](https://arxiv.org/pdf/2601.06046v1)

**作者:** Sharvari Kamble `[一作]` (University of Mumbai), Swati Bhatt `[通讯]` (University of Mumbai)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并验证了ISMS-CR（Integrated Safety Management System for Central Railway Workshop），一个整合工单许可、资产维护、合同管理和事故跟踪的模块化数字化安全管理框架，重点实现了根据IS 17893:2022标准的Permit‑to‑Work（PTW）生命周期自动化与实时追踪。

**💡 创新点**

创新点包括：
- 将PTW全生命周期（发起、校验、授权、执行、监控、关闭）完全数字化，采用QR码实现审计追溯；
- 通过共享数据层将工单信息与事故、合规、地理映射等模块联通，形成跨域的闭环管理；
- 在权限分层设计上严格遵循IS 17893:2022，定义多角色（发起人、执行者、安全官、区域负责人）不同权限，提升责任可追溯性；
- 将实时监测、风险评估和预警嵌入工单流程，实现从工单审批到现场执行的无缝衔接。

**🔧 技术方法**

技术栈：
- 前端：React 18 + TypeScript + Vite + Tailwind CSS + shadcn‑ui；
- 后端：Node.js v18 + Express.js + TypeScript；
- 数据库：PostgreSQL 15 + PostGIS（空间数据）+ Prisma ORM；
- 认证：JWT、角色基准访问控制；
- 通信：RESTful APIs、WebSocket（实时状态更新）；
- 部署：容器化（Docker）在AWS或本地服务器；
- 监控与日志：自动化通知（Email/SMS）、审计日志、API响应监测。

**📊 数据集**

数据集：
- 机器与设备表：250台设备，包含状态、维护日志、检查间隔；
- 工单表：100+条高度/电工作业许可（符合IS 17893:2022流程）；
- 合同表：50条承包商资料，含安全合规证书与有效期；
- 事故表：75起案例，按严重性与根因分类；
- 所有表均基于模拟真实运营数据，存储于PostgreSQL，确保测试的真实性。

**📈 对比分析**

比较方法与性能：
- 将数字化工单审批时间与传统纸质流程（平均15.4 min）对比，数字化平均5.5 min，缩短约64%；
- 在10–200并发用户下的压力测试：API平均延迟118 ms，数据库查询27 ms，系统正常运行时间99.3%；
- 前端表单校验准确率达98.7%。
- 结果表明数字化显著提升审批效率、实时监控可视化和审计可追溯性。

**⚠️ 局限性**

局限性：
- 在高并发提交时偶尔出现短暂的响应延迟；
- 移动端界面尚未完全优化，导致现场操作体验不够流畅；
- 当前版本未集成IoT传感器和AI预测模型，缺乏对设备状态的实时预测与预警；
- 仍需在真实车间环境中进行大规模部署与长期稳定性验证。

---

## 309. Jamming Detection in Cell-Free MIMO with Dynamic Graphs

**arXiv ID:** 2601.06075 | [PDF](https://arxiv.org/pdf/2601.06075v1)

**作者:** Ali Hossary `[一作]` (University of Padova), Stefano Tomasin `[通讯]` (University of Padova)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于动态图和图卷积网络的干扰检测框架，用于细胞自由MIMO网络的干扰识别与报警；

**💡 创新点**

创新点在于将无线网络抽象为时间演化的动态图，并通过GCN‑Transformer联合学习空间-时序特征，同时系统性探讨了干扰持续性导致的基线漂移问题；

**🔧 技术方法**

使用的技术包括动态图建模、图卷积网络（GCN）、Transformer自注意力机制、监督式交叉熵训练及Adam优化；

**📊 数据集**

数据集为仿真生成的细胞自由MIMO动态图，包含5个固定AP、10个移动UE，覆盖移动、不同干扰持续时间τ（1–10）、衰落与非衰落两种信道场景；

**📈 对比分析**

与GCN‑LSTM基线比较显示，混合τ训练在非衰落场景准确率>99%，衰落场景保持75–90%；相比仅训练τ=10的专门模型，混合训练在不同干扰持续性下具有更好的泛化性能；

**⚠️ 局限性**

主要局限在于对持续干扰（τ=9–10）检测效果下降，原因是网络自适应导致的基线漂移；同时仿真环境可能未能充分覆盖真实网络的复杂性。

---

## 310. Teach Diffusion Language Models to Learn from Their Own Mistakes

**arXiv ID:** 2601.06428 | [PDF](https://arxiv.org/pdf/2601.06428v1)

**作者:** Liming Liu `[一作]`, Tuo Zhao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“Decoupled Self‑Correction (DSC)”框架，使得掩码扩散语言模型在不降低生成质量的前提下实现自我纠错；

**💡 创新点**

核心创新在于将生成与纠错两阶段完全分离，并通过“Future‑Context Augmentation (FCA)”让纠错头在训练时接触更具未来信息的错误样本，提升误检准确性；

**🔧 技术方法**

采用掩码扩散模型（MDM）+轻量级Transformer纠错头，结合二阶段训练（先训练MDM后冻结再训练纠错头），并在推理时加入周期性重掩码与重复避免机制；

**📊 数据集**

在数学推理（GSM‑8K、Math）和代码生成（MBPP、HumanEval）等公开基准数据集上进行实验，使用LLaDA 8B模型作为底层；

**📈 对比分析**

与基线SFT和联合优化（Joint Optimization）对比，DSC在无重掩码下保持与基线相同的生成准确率，而在加入重掩码后，准确率提升至约63.5%（GSM‑8K）并显著压缩平均推理步数，整体表现优于联合优化与随机重掩码；

**⚠️ 局限性**

局限性包括：仍需要手动设置重掩码阈值与步长；在极大步长或极高并行度下误差累积仍可能出现；以及在非掩码扩散框架外的模型迁移性未充分验证。

---

## 311. Rethinking Inter-Process Communication with Memory Operation Offloading

**arXiv ID:** 2601.06331 | [PDF](https://arxiv.org/pdf/2601.06331v1)

**作者:** Misun Park `[一作]`, Ada Gavrilovska `[通讯]`

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

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

## 312. A Review of Online Diffusion Policy RL Algorithms for Scalable Robotic Control

**arXiv ID:** 2601.06133 | [PDF](https://arxiv.org/pdf/2601.06133v1)

**作者:** Wonhyeok Choi `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Sunghoon Im `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**通讯引用:** 933 | [OpenAlex ID](https://openalex.org/A5019689626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统综述了在线扩散策略强化学习（Online Diffusion Policy RL）方法，并在NVIDIA Isaac Lab仿真平台上对12种机器人任务进行统一评测，探讨了不同算法在任务多样性、并行化、采样步数、跨体型泛化与环境鲁棒性等五维度的表现与权衡；

**💡 创新点**

提出了四大类在线DPRL算法的分层分类法——动作梯度（Action‑Gradient）、Q加权（Q‑Weighting）、邻近性（Proximity‑Based）与BPTT；并系统阐述了各类算法如何克服扩散模型与传统RL梯度不兼容的根本冲突；

**🔧 技术方法**

采用扩散策略模型、策略梯度、Q值重加权、近似对数似然、BPTT、一致性模型、分布式RL等技术；

**📊 数据集**

使用NVIDIA Isaac Lab benchmark，包含12个不同机器人控制任务（包含3经典、6行走、3操纵任务），并利用1,024个并行环境进行数据收集；

**📈 对比分析**

将DPRL方法（DIPO、QVPO、GenPO、DACER）与四个经典RL基线（PPO、DDPG、TD3、SAC）进行对比。实验显示：在大规模并行化环境下，GenPO与PPO取得最高收益；在受限并行化时，DIPO保持优势并超过传统SAC；DIPO在多任务上排名显著优于DDPG、TD3；Q‑Weighting方法对奖励尺度敏感，BPTT类方法在采样步数增大时性能急剧下降；

**⚠️ 局限性**

主要局限包括：动作梯度方法受Q函数误差影响导致梯度偏置；Q‑Weighting对奖励尺度极度敏感；邻近性方法需通过近似对数似然或可逆扩散实现，导致模型灵活性受限；BPTT类方法梯度消失、训练不稳定、对采样步数敏感；在跨体型与环境泛化时，尤其是对大型机器人或视觉扰动，在线DPRL往往出现性能急剧下降。

---

## 313. From Values to Frameworks: A Qualitative Study of Ethical Reasoning in Agentic AI Practitioners

**arXiv ID:** 2601.06062 | [PDF](https://arxiv.org/pdf/2601.06062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 314. What makes for an enjoyable protagonist? An analysis of character warmth and competence

**arXiv ID:** 2601.06658 | [PDF](https://arxiv.org/pdf/2601.06658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 315. Styles + Persona-plug = Customized LLMs

**arXiv ID:** 2601.06362 | [PDF](https://arxiv.org/pdf/2601.06362v1)

**作者:** Yutong Song `[一作]` (University of California, Irvine), Yu Wang `[通讯]` (TikTok)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PsPLUG，一种轻量级软提示插件，在冻结的大模型上通过学习个体化残差实现用户个性化生成，并能兼顾显式风格指令。

**💡 创新点**

将个性化视为相对于中性模型的分布残差，使用风格条件负样本实现个体化与风格的解耦与可调平衡。

**🔧 技术方法**

使用软提示插件、分布残差学习、Bradley‑Terry偏好对比、可调alpha控制、句子编码器以及冻结的LLM。

**📊 数据集**

采用LaMP benchmark（七个多任务）及四类预设风格指令，基模型为Qwen/Qwen3‑8B。

**📈 对比分析**

与检索、PEFT、PPlug、OPPU等基线对比，PsPLUG在无风格和有风格两种设置下均取得最高或次高指标，且在风格适应上表现最优。

**⚠️ 局限性**

仅支持四类预设风格，难以处理更细粒度或开放式风格；实验仅在单一模型与英文数据上验证，跨语言、跨架构的鲁棒性待进一步研究。

---

## 316. How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs

**arXiv ID:** 2601.06599 | [PDF](https://arxiv.org/pdf/2601.06599v1)

**作者:** Shivam Adarsh `[一作]` (University of Copenhagen), Christina Lioma `[通讯]` (University of Copenhagen)

**通讯引用:** 2752 | [OpenAlex ID](https://openalex.org/A5045425016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在给定上下文时大型语言模型（LLM）内部“真值向量”如何在不同层面发生几何变形，重点分析了方向变化（θ）与向量幅度变化；

**💡 创新点**

首次系统量化上下文对LLM真值向量方向与幅度的几何影响，并揭示了三阶段层级变化模式及模型规模对方向/幅度敏感性的差异；

**🔧 技术方法**

利用残差流激活提取真值向量，计算欧氏角度θ和相对幅度比值，结合多层级可视化与统计检验；

**📊 数据集**

四大LLM（LLaMA‑3.1‑8B、Mistral‑Nemo‑12B、Qwen3‑4B、SmolLM3‑3B）与四个多域数据集（Druid、MF2、ConflictQA、LegalBench）进行实验；

**📈 对比分析**

通过层级平均角度、幅度比值和与随机上下文的差异进行比较，结果显示大模型对方向更敏感，小模型对幅度更敏感，且上下文对真值向量的正向变形显著提高；

**⚠️ 局限性**

仅考察首个生成标记的残差流激活，未对多 token 分布式信息、最终层以外的层次做深入探究，且实验仅限英文数据，缺乏因果干预验证。

---

## 317. Attention Mechanism and Heuristic Approach: Context-Aware File Ranking Using Multi-Head Self-Attention

**arXiv ID:** 2601.06185 | [PDF](https://arxiv.org/pdf/2601.06185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 318. Structured Episodic Event Memory

**arXiv ID:** 2601.06411 | [PDF](https://arxiv.org/pdf/2601.06411v1)

**作者:** Zhengxuan Lu `[一作]` (Southeast University), Baotian Hu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 4075 | [OpenAlex ID](https://openalex.org/A5083079672)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SEEM（Structured Episodic Event Memory）框架，结合图记忆层与事件层以结构化方式管理LLM长时间交互的记忆，解决检索碎片化问题。

**💡 创新点**

创新点在于使用层级记忆架构、Episodic Event Frames（EEFs）与逆向来源扩展（RPE）机制，将动态叙事与静态事实融合，重构完整连贯的上下文。

**🔧 技术方法**

采用LLM驱动的框架提取与融合、图数据库关系四元组、关联传播、逆向来源扩展、混合检索及生成式推理等技术。

**📊 数据集**

实验使用LoCoMo和LongMemEval两大长时交互评测基准。

**📈 对比分析**

与dense retrieval、Mem0、A-MEM、HippoRAG 2等基线对比，SEEM在LoCoMo上F1 61.1、J 78.0、BLEU 56.1，在LongMemEval上准确率65%，均优于基线并提升4.4%绝对分数。

**⚠️ 局限性**

局限性包括高计算与延迟成本、LLM提取/融合错误可能累积导致记忆污染、预定义语义槽限制抽象信息捕获，以及隐私与算法偏见风险。

---

## 319. One if by Land, Two if by Sea, Three if by Four Seas, and More to Come -- Values of Perception, Prediction, Communication, and Common Sense in Decision Making

**arXiv ID:** 2601.06077 | [PDF](https://arxiv.org/pdf/2601.06077v1)

**作者:** Aolin Xu `[一作]` `[通讯]` (Honda Research Institute US), Aolin Xu (Honda Research Institute US)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文提出了一套严谨的理论框架，用决策理论和信息论的角度量化感知、预测、通信以及常识在决策中的价值，并分析了它们的数学性质与相互关系。

**💡 创新点**

创新点在于：①将感知与预测的价值统一定义并与信息量（熵、互信息）关联；②揭示感知单独使用时可能产生负价值，强调预测的重要性；③通过函数凸凹性推导价值的最优顺序与多智能体优先级；④为多智能体决策提供了理论上最优感知/预测排序算法。

**🔧 技术方法**

主要技术手段是决策理论中的风险最小化与信息论中的熵、互信息、KL散度等概念，辅以凸优化与期望线性/凸/凹性质证明。

**📊 数据集**

文章未使用实际数据集，主要以离散随机变量的理论模型和简化示例（如道路驾驶场景）进行说明。

**📈 对比分析**

由于研究主要是理论推导与性质证明，没有实测性能指标；通过与经典信息论结果（如信息量）对齐，验证了公式的一致性。

**⚠️ 局限性**

局限性包括：①仅处理静态一次决策问题，未涵盖动态/序列决策；②对复杂非离散环境的推广仍待进一步研究；③实际应用中对分布与损失函数的先验假设较强，需进一步验证鲁棒性。

---

## 320. Why Slop Matters

**arXiv ID:** 2601.06060 | [PDF](https://arxiv.org/pdf/2601.06060v1)

**作者:** Cody Kommers `[一作]` (Alan Turing Institute), Hoyt Long `[通讯]` (University of Chicago)

**通讯引用:** 217 | [OpenAlex ID](https://openalex.org/A5015828303)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从社会功能和审美价值的角度，系统性地阐述并分析了AI Slop的特征与维度。

**💡 创新点**

将AI Slop视为具有社会功能与审美价值的独立研究对象，提出家族相似性定义和三维变异维度，填补了对该现象的概念空白。

**🔧 技术方法**

主要采用理论分析、文献综述与概念化方法，没有使用具体的实验技术。

**📊 数据集**

未收集或使用任何数据集，论文为概念性、理论性讨论。

**📈 对比分析**

未开展实验或比较，因而没有性能指标或对比结果。

**⚠️ 局限性**

概念仍相对模糊，缺乏实证定义与数据支持，需要进一步实验与量化研究。

---

## 321. Investigating Anthropometric Fidelity in SAM 3D Body

**arXiv ID:** 2601.06035 | [PDF](https://arxiv.org/pdf/2601.06035v1)

**作者:** Aizierjiang Aiersilan `[一作]` (Institute for Innovation in Health Computing), James Hahn `[通讯]` (Institute for Innovation in Health Computing)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文对SAM 3D Body在医学场景下的重建效果进行了系统分析，探讨了其为何难以捕捉孕期、脊柱侧弯等异常形态，并给出了改进思路。

**💡 创新点**

创新点在于将模型失效归因于感知‑失真权衡，指出了MHR参数化、DINOv3语义条件与标注对齐三大瓶颈，并提出隐式‑显式混合、专家‑对齐和扩展参数化的改进路径。

**🔧 技术方法**

主要技术包括MHR低维形状/骨架参数化、DINOv3语义不变特征编码、基于图像的标注对齐、3D高斯Splatting/隐式函数混合等。

**📊 数据集**

使用的数据集为SAM 3D Body的训练集（标准化人体）以及商用扫描器（Fit3D）和医学成像（CT/MRI）数据做对比验证。

**📈 对比分析**

通过与SAM 3D Body在标准人体上的性能对比，发现其在孕妇、脊柱侧弯等特殊形态时偏向平均形状，缺少细节，医学度量表现明显不足。

**⚠️ 局限性**

限制在于低维参数瓶颈导致解剖多样性被压缩，DINOv3的语义不变性和空间量化丢失高频医学细节，标注对齐对异常结构施加平滑先验，使模型难以满足毫米级诊断需求。

---

## 322. Large-Scale Continual Scheduling and Execution for Dynamic Distributed Satellite Constellation Observation Allocation

**arXiv ID:** 2601.06188 | [PDF](https://arxiv.org/pdf/2601.06188v1)

**作者:** Itai Zilberstein `[一作]` (Carnegie Mellon University), Steve Chien `[通讯]` (Jet Propulsion Laboratory, California Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了动态多卫星星座观测调度问题（DCOSP）并设计了基于邻域随机搜索的动态增量算法D-NSS。

**💡 创新点**

将动态分布式约束优化问题与观测执行相结合，提出全局最优的离线求解框架和针对大规模动态环境的分解修复方法。

**🔧 技术方法**

使用分布式随机搜索、邻域分解启发式GND、离线全局最优算法、基于Java的仿真。

**📊 数据集**

基于Planet与Walker星座模型，634个全球城市的动态观测请求生成仿真数据。

**📈 对比分析**

与随机、贪婪、0-NSS、0-DSA、D-DSA等基线比较，D-NSS在满足率上逼近最优、通信量和计算时间比D-DSA低1-2个数量级。

**⚠️ 局限性**

缺乏理论最优性保证，对未来动态预测依赖外部模型，实验仅在仿真环境，未在真空空间硬件验证。

---

## 323. Autonomous QA Agent: A Retrieval-Augmented Framework for Reliable Selenium Script Generation

**arXiv ID:** 2601.06034 | [PDF](https://arxiv.org/pdf/2601.06034v1)

**作者:** Dudekula Kasim Vali `[一作]` `[通讯]` (VIT AP University), Dudekula Kasim Vali (VIT AP University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款基于检索增强生成（RAG）的自主QA代理，用于将自然语言需求自动转化为 Selenium UI 测试脚本；

**💡 创新点**

创新点在于：① 针对 QA 定制的 RAG 架构，将功能需求文档与实际 HTML/DOM 结构同时检索；② 多模态摄取管道（Markdown、PDF、JSON、HTML）；③ 上下文融合的提示工程（CoT + 约束），显著降低 LLM 幻觉；

**🔧 技术方法**

核心技术包括：Llama 3.1‑8b‑instant（Groq API）、ChromaDB 向量数据库、LangChain RAG 管道、Streamlit 前端、FastAPI 后端、Recursive Character Text Splitter、Cosine 相似度检索、CoT 提示工程；

**📊 数据集**

数据集为自建电商网站的 4 个页面（共 127 DOM 元素）以及其 PRD（Markdown/PDF/JSON），共 20 条测试场景；

**📈 对比分析**

与标准 LLM（无检索）对比，RAG 取得：语法有效率 100% 对 95%，元素解析率 95% 对 40%，执行成功率 90% 对 30%（p<0.001）。消融实验表明文本+HTML 再检索是最优配置；

**⚠️ 局限性**

局限性包括：① 仅使用静态 HTML，无法覆盖 SPA 的动态 DOM；② 仅评估单一电商应用，缺乏跨领域验证；③ 仅采用单一 LLM 模型；④ 上下文窗口受限，极端复杂页面可能失效；⑤ 未与商业工具做直接对比；

---

## 324. PCoKG: Personality-aware Commonsense Reasoning with Debate

**arXiv ID:** 2601.06234 | [PDF](https://arxiv.org/pdf/2601.06234v1)

**作者:** Weijie Li `[一作]` (Soochow University), Guodong Zhou `[通讯]` (Soochow University)

**通讯引用:** 10007 | [OpenAlex ID](https://openalex.org/A5012794465)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个集成人格特质的常识知识图谱 PCOKG，并在此基础上实现个性化对话生成。

**💡 创新点**

创新点在于把 Myers‑Briggs 性格维度嵌入四元组 (事件, 性格, 推理维度, 结果)，并通过 LLM 评估器筛选事件、角色扮演和多轮辩论机制提升推理质量。

**🔧 技术方法**

使用大语言模型 (Deepseek‑R1、Qwen‑Turbo、Doubao‑1.6‑Seed 等) 进行事件筛选、角色扮演推理、辩论迭代；并在生成模型上采用 LoRA 微调和全参数微调技术。

**📊 数据集**

主要数据集为 ATOMIC（事件推理三元组）和构建的 521,316 条 (e,p,r,t) 四元组 PCOKG，测试集为 SPC 对话数据集。

**📈 对比分析**

在 BLEU‑4、ROUGE‑1/2/L 等指标上对比 COMET、单一 LLM、以及 PCoKGM 微调模型，结果显示 PCoKGM 在所有指标上均显著优于基线，尤其在人格一致性和自然度上提升明显。

**⚠️ 局限性**

局限性包括仅考虑人格维度，未纳入性别、职业等其他影响因素，且依赖 MBTI 这一有争议的性格模型。

---

## 325. Data-Driven Framework Development for Public Space Quality Assessment

**arXiv ID:** 2601.06026 | [PDF](https://arxiv.org/pdf/2601.06026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 326. A Rising Tide Lifts All Boats: MTQE Rewards for Idioms Improve General Translation Quality

**arXiv ID:** 2601.06307 | [PDF](https://arxiv.org/pdf/2601.06307v1)

**作者:** Ishika Agarwal `[一作]` (University of Illinois at Urbana-Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探索使用GRPO强化学习结合MTQE模型对小型LLM进行细调，以提升非组合式表达（成语、习语等）的翻译质量。

**💡 创新点**

提出将MTQE模型作为奖励函数，通过GRPO fine‑tuning实现对成语语义的精准翻译，并证明该方法在非成语文本翻译上也能提升性能。

**🔧 技术方法**

使用GRPO（Group Relative Policy Optimization）、MTQE模型（如COMET）、LLM（Qwen 3B、Llama 8B）以及结构化提示工程。

**📊 数据集**

基于中文–英文成语数据集PETCI（1623对，1k训练）和合成/OPUS的印地语–英文成语数据集（800/200对）。

**📈 对比分析**

与原始翻译模型（NLLB、Command‑R）、提示方法（LIA、TF）、SFT进行对比，GRPO+MTQE在成语翻译上平均提升约14点，在非成语翻译上提升约8点，跨语言迁移提升约6点。

**⚠️ 局限性**

受限于MTQE模型的表现与训练成本，GRPO训练耗时高且需多GPU，且MTQE模型对多语言的覆盖有限。

---

## 327. Gecko: An Efficient Neural Architecture Inherently Processing Sequences with Arbitrary Lengths

**arXiv ID:** 2601.06463 | [PDF](https://arxiv.org/pdf/2601.06463v1)

**作者:** Xuezhe Ma `[一作]` (University of Southern California), Carole-Jean Wu `[通讯]` (Meta AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并训练了一种新的长序列语言模型 Gecko，基于 Mega/Megalodon 架构，并加入多项技术以实现对任意长度序列的高效处理。

**💡 创新点**

创新点包括：时步衰减归一化、滑动块注意力（SCA）以及自适应工作记忆（AWM）结合位置感知在线 softmax，实现无需显式上下文扩展即可捕获长程依赖和检索信息。

**🔧 技术方法**

技术细节包括：EMA + 复数 CEMA、时步衰减归一化、滑动块注意力、线性注意力 + 位置感知在线 softmax 的自适应工作记忆、RoPE、SwiGLU、AdamW 等。

**📊 数据集**

训练使用 Dolma v1.7（约 2 T tokens）以及 1B/7B 模型；评估基准包括 MMLU、BoolQ、HellaSwag、PIQA、SIQA、WinoG、ARC-e/c、NQ、TQA、Scrolls、Passkey、NIAH 等。

**📈 对比分析**

在相同数据量（2 T tokens）和算力下，与 Llama2‑7B/13B、Megalodon‑7B、OLMo‑1B 等进行对比；Gecko‑7B 在训练 NLL 1.68、在短/长序列基准上均优于对手，并能在 4 M token 的上下文下保持良好 perplexity，显示出显著的数据效率和长程表现。

**⚠️ 局限性**

局限性：在极长序列（> 4 M token）仍可能出现记忆冲突/信息衰减；目前仅验证 1B/7B 规模，尚未证明可扩展到更大模型或更长序列；训练成本和并行实现仍需进一步优化。

---

## 328. Developing Bayesian probabilistic reasoning capacity in HSS disciplines: Qualitative evaluation on bayesvl and BMF analytics for ECRs

**arXiv ID:** 2601.06038 | [PDF](https://arxiv.org/pdf/2601.06038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 329. Do Language Models Reason Across Languages?

**arXiv ID:** 2601.06644 | [PDF](https://arxiv.org/pdf/2601.06644v1)

**作者:** Yan Meng `[一作]` (University of Amsterdam), Christof Monz `[通讯]` (University of Amsterdam)

**通讯引用:** 8794 | [OpenAlex ID](https://openalex.org/A5109059955)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多语言两跳问答任务，并对大型语言模型在跨语言推理过程中的行为进行细粒度分析

**💡 创新点**

创新点在于将标准两跳问题拆解为可评估的子问题，引入“Unfaithfulness”和“Compositional Failure”两种失效模式，并展示通过结构化子问题提示(SubQ)显著提升跨语言推理表现

**🔧 技术方法**

技术包括：多语言两跳问答框架、子问题拆解与评估、输入消除归因分析、干扰文档实验、三阶段SubQ提示与零样本Chain-of-Thought提示

**📊 数据集**

使用扩展后的HotpotQA数据集（含英语原始数据及其法语、俄语、阿拉伯语、中文四种高资源语言版本），共1,000个英语例子，过滤后182个符合跨语言两跳要求的样本

**📈 对比分析**

通过在Gemma‑3‑27B‑Instruct等多语言LLM上进行基准评测，发现跨语言推理性能下降显著，尤其是答案文档语言变化导致约13.96%下降；Unfaithfulness与Compositional Failure分别达到约33%和18%；使用三阶段SubQ提示后，整体准确率从10.1%提升至66.5%，显著优于零样本CoT

**⚠️ 局限性**

局限性包括：评测数据集规模有限，且仅涵盖四种高资源语言，未覆盖低资源语言；评测聚焦于模型推理机制，未对比现有最优方法；数据过滤和翻译可能引入偏差

---

## 330. A Unified Attention U-Net Framework for Cross-Modality Tumor Segmentation in MRI and CT

**arXiv ID:** 2601.06187 | [PDF](https://arxiv.org/pdf/2601.06187v1)

**作者:** Nishan Rai `[一作]` (New Mexico State University), Pushpa R. Dahal `[通讯]` (New Mexico State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种统一的Attention U‑Net模型，能够同时对脑部MRI（BraTS 2021）和肺部CT（LIDC‑IDRI）进行肿瘤分割。

**💡 创新点**

创新点在于首次实现单一模型在多模态、多解剖区域上并行训练，无需专门的编码器或域适应模块。

**🔧 技术方法**

采用了Attention门、四通道输入预处理、Focal Tversky损失以及AdamW+1‑cycle学习率策略。

**📊 数据集**

使用的公开数据集包括BraTS 2021（多参数MRI）和LIDC‑IDRI（胸部CT）。

**📈 对比分析**

与分别训练的单模态基线对比，统一模型在MRI上Dice 0.83、AUC 0.97，CT上Dice 0.55、AUC 0.83，整体Dice 0.64、AUC 0.90，显示出可接受的跨模态性能。

**⚠️ 局限性**

局限性包括仅使用二维切片、未做完整的超参数搜索、对CT小病灶识别精度仍不足，且未验证不同扫描设备的鲁棒性。

---

## 331. HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents

**arXiv ID:** 2601.06377 | [PDF](https://arxiv.org/pdf/2601.06377v1)

**作者:** Ningning Zhang `[一作]` (Macau University of Science and Technology), Wenyong Wang `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 5988 | [OpenAlex ID](https://openalex.org/A5028467074)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HiMem，一个层级化的长期记忆框架，兼顾 Episode Memory（事件记忆）和 Note Memory（笔记记忆），支持检索、动态更新和自我演化；

**💡 创新点**

创新点包括：① 事件与笔记的层级结构，桥接具体对话与抽象知识；② 基于主题感知和惊奇双通道的事件分割策略；③ 多阶段知识抽取与语义对齐；④ 冲突感知的记忆再整合机制，实现检索失败触发的自我演化；⑤ 提供 hybrid 与 best‑effort 两种检索策略；

**🔧 技术方法**

核心技术：LLM 进行事件分割、知识抽取、冲突检测和证据充分性评估；向量嵌入与检索模块；语义对齐与实体消歧；冲突感知的记忆再整合；

**📊 数据集**

使用 LoCoMo 长周期对话基准（多轮对话，约 600 轮，32 交互阶段）进行实验；

**📈 对比分析**

与 Mem0、SeCom、A‑MEM 等基线在同一 LLM (GPT‑4o‑mini) 环境下对比，指标为 GPT‑Score、F1、延迟与 token；HiMem 在单跳、多跳、时间推理和开放域任务上均显著领先，整体 GPT‑Score 最高；消融实验验证层级结构、语义对齐和自我演化的重要性；

**⚠️ 局限性**

局限性：1）高度依赖 LLM 的判断能力，易受输入噪声、隐喻、跨文化差异影响；2）一次性分割表达能力有限，难处理极长或交叉对话；3）自我演化仅基于检索失败触发，可能遗漏隐性不一致；4）实验仅覆盖单一文本交互，缺少多模态、多用户场景；5）隐私与幻觉风险需要进一步治理。

---

## 332. Manifold-based Sampling for In-Context Hallucination Detection in Large Language Models

**arXiv ID:** 2601.06196 | [PDF](https://arxiv.org/pdf/2601.06196v1)

**作者:** Bodla Krishna Vamshi `[一作]` (University of Maryland), Haizhao Yang `[通讯]` (University of Maryland)

**通讯引用:** 2142 | [OpenAlex ID](https://openalex.org/A5079602544)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于流形的演示抽样框架 MB-ICL，用来改进大语言模型在幻觉检测任务中的上下文演示选择。

**💡 创新点**

创新点在于通过联合学习局部流形结构和类级原型几何来进行示例选择，而非仅依赖词表或嵌入空间的表面相似度，从而获得更稳健的幻觉检测效果。

**🔧 技术方法**

核心技术包括：冻结 LLM 的隐藏表示、轻量级投影网络 hθ、流形构造（PCA+局部线性近似）、代理锚点损失、流形点对点损失以及动量原型更新。

**📊 数据集**

使用 FEVER 事实验证数据集和 HaluEval（问答、对话、摘要）幻觉检测数据集进行评估。

**📈 对比分析**

与 KNN、聚类、Perplexity、BM25、SA‑ICL 等基线比较，MB‑ICL 在大多数模型和任务（尤其是对话与摘要）上获得更高的准确率，并且对温度变化更稳健，平均 perplexity 也更低。

**⚠️ 局限性**

局限性包括：仅在小于 8B 参数的模型上验证，投影头训练与流形构造仍有额外计算开销，且未对在线动态示例更新或不确定性估计做进一步探索。

---

## 333. Lightweight Yet Secure: Secure Scripting Language Generation via Lightweight LLMs

**arXiv ID:** 2601.06419 | [PDF](https://arxiv.org/pdf/2601.06419v1)

**作者:** Keyang Zhang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Limin Sun `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个统一的、可复现的 PowerShell 安全脚本生成、分析和修复基准 SecGenEval-PS，验证现有 LLM 在安全推理方面的能力。

**💡 创新点**

首次将多阶段安全编码任务（生成、审计、修复）纳入同一评测框架，并利用 PSScriptAnalyzer 输出作为可信标注，实现了对 LLM 安全推理的量化评估。

**🔧 技术方法**

结合静态分析工具 PSScriptAnalyzer、结构化 JSON 注释、可重复的筛选/归一化管道，并在多种提示模式下评估 GPT‑4o、o3‑mini 等模型。

**📊 数据集**

使用从 The Stack 收集的 520k+ PowerShell 脚本，筛选出 100k 规模子集，再抽样 400 个脚本作为 benchmark，标注规则违例信息。

**📈 对比分析**

通过功能正确率 (FRate)、安全合规率 (SRate)、二进制准确率、规则/问题级别 F1、修复成功率 (FSucRate) 等指标比较不同模型；结果显示 GPT‑4o 与 o3‑mini 在功能上领先，但安全合规率普遍低；在分析与修复任务中，提供上下文提示显著提升性能，规模更大的模型并非必然更好。

**⚠️ 局限性**

受限于仅基于 PSScriptAnalyzer 的静态规则，缺少运行时动态检查、跨文件上下文和数据流分析，且标注仅覆盖约 95% 常见规则，未能覆盖低频或工具专属问题。

---

## 334. Dynamic Incentivized Cooperation under Changing Rewards

**arXiv ID:** 2601.06382 | [PDF](https://arxiv.org/pdf/2601.06382v1)

**作者:** Philipp Altmann `[一作]` (Ludwig Maximilian University of Munich), Sven Koenig `[通讯]` (University of California)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于动态奖励激励的分布式机制DRIVE，用来在奖励随时间变化的社会困境中保持合作；

**💡 创新点**

创新点在于利用奖励差值的互惠交换，完全自适应且不依赖固定激励或域参数，理论证明在任意按周期的仿射奖励变换下仍能使合作成为占优策略；

**🔧 技术方法**

采用MARL的策略梯度学习，加入分布式奖励塑形（DRIVE）以及对比的LIO、MATE、IA等PI方法，使用归一化返回实现对奖励变换的鲁棒性；

**📊 数据集**

实验数据集包括三类经典序列社会困境：Iterated Prisoner’s Dilemma、Coin（2/4 代理）和Harvest-12（12 代理），以及多种奖励变换函数（线性、指数衰减、阶梯、抖动余弦）；

**📈 对比分析**

与Naive、LOLA-PG、POLA-DiCE、LIO、MATE、IA等方法比较，DRIVE在大多数任务和奖励变换场景下都能保持或提升合作率和可持续性，尤其在奖励变化时显著优于其他PI方法；

**⚠️ 局限性**

局限性包括对完整且诚实的邻居交流的依赖，若通信失败或代理拒绝响应，DRIVE效果下降；此外理论分析主要针对同质代理和完全连通或密集邻居网络，稀疏或异质网络的适用性尚待验证。

---

## 335. MITRA: A Large-Scale Parallel Corpus and Multilingual Pretrained Language Model for Machine Translation and Semantic Retrieval for Pāli, Sanskrit, Buddhist Chinese, and Tibetan

**arXiv ID:** 2601.06400 | [PDF](https://arxiv.org/pdf/2601.06400v1)

**作者:** Sebastian Nehrdich `[一作]` (Tohoku University), Kurt Keutzer `[通讯]` (University of California)

**通讯引用:** 37569 | [OpenAlex ID](https://openalex.org/A5047285420)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MITRA框架，用MT枢轴+筛选/句子对齐技术挖掘古典佛教文献的句级并行文本，构建了1.74M句对平行语料并训练了9B参数的Gemma‑2基模型（MITRA‑BASE）以及其针对机器翻译和语义检索的细调版本（MITRA‑MT、MITRA‑Embed），并公开了数据集、模型权重和检索基准。

**💡 创新点**

核心创新在于：① 通过英语MT枢轴结合粗粒度聚类与细粒度句子对齐的三步管道，高效过滤佛教文本的噪声并实现多语言并行对齐；② 构建首个覆盖梵语、巴利、藏语、汉语的海量并行语料；③ 在此语料上连续预训练Gemma‑2，并针对翻译与检索分别进行指令细调，显著超越现有开源模型。

**🔧 技术方法**

使用的技术包括：机器翻译（MADLAD‑400 fine‑tuned）、BGE M3嵌入、kNN聚类、BERTAlign对齐、Gemma‑2连续预训练（DeepSpeed ZeRO‑3）、指令细调（Claude 3.5 Sonnet、Gemini 2.0 Flash生成合成数据）、对齐评估和多任务检索评测。

**📊 数据集**

主要数据集：MITRA‑parallel（1.74M句对，梵语-藏语、汉语-藏语、梵语-汉语）；各类单语与并行语料（Sanskrit‑English 1M、Tibetan‑English 2M、Tibetan‑Chinese 41k、Sanskrit‑Chinese 31k、巴利‑English 149k）；以及人工与合成的检索基准数据。

**📈 对比分析**

在机器翻译评测中，MITRA‑MT在Sanskrit、Pāli、Tibetan、Buddhist Chinese→English任务上均优于Gemma‑3‑27B（平均+15 GEMBA），并在Buddhist Chinese→English上领先现有领域模型+15 GEMBA；在七任务语义检索基准中，MITRA‑Embed在P@1等指标上显著优于Gemma‑2、Gemma‑3、LaBSE、FastText及BM25，尤其在跨语言检索场景下表现突出。

**⚠️ 局限性**

限制包括：检索基准未覆盖巴利与其他古典佛教语言的跨语言任务；机器翻译基准数据受版权限制无法公开；模型规模大，嵌入维度高，部署在大规模语料上仍有挑战；未涵盖托克里亚、日语、韩语、现代日语等其他佛教传统语言。

---

## 336. Political Alignment in Large Language Models: A Multidimensional Audit of Psychometric Identity and Behavioral Bias

**arXiv ID:** 2601.06194 | [PDF](https://arxiv.org/pdf/2601.06194v1)

**作者:** Adib Sakhawat `[一作]` (Islamic University of Technology), Md Kamrul Hasan `[通讯]` (Islamic University of Technology)

**通讯引用:** 3157 | [OpenAlex ID](https://openalex.org/A5100656463)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对26款大型语言模型进行政治倾向审计，结合三种心理测评工具与新闻偏见标注任务，探究模型的政治立场稳定性及与下游行为的关联。

**💡 创新点**

创新点在于跨测评工具三角化模型政治立场、揭示闭源与开源模型在文化进步性上的显著差异，以及发现模型内部政治身份与偏见识别性能无关。

**🔧 技术方法**

技术上使用多轮 Prompt 交互、官方测评平台采分、10 次独立跑以计算波动率，并采用方差分析、皮尔逊相关、k‑means 聚类和线性回归等统计方法。

**📊 数据集**

数据集包括官方 Political Compass、SapplyValues、8 Values 测验题集，以及约 27,000 篇由 Ground News 采集、AllSides/AdFontes 共识标签的新闻文章。

**📈 对比分析**

与单轴评估相比，多维度框架显示模型主要集中在自由左翼象限，且闭源模型文化进步性显著高于开源；下游新闻偏见分类中模型对左翼内容识别准确度高（19.2%），右翼低（2.1%）。

**⚠️ 局限性**

局限性包括对西方测评工具的依赖、缺乏对训练数据和 RLHF 机制的因果归因、以及基于高层标签的偏见评估无法捕捉文章细粒度差异。

---

## 337. Teacher training in inclusive digital skills in secondary education. Students with Autism Spectrum Disorders

**arXiv ID:** 2601.06058 | [PDF](https://arxiv.org/pdf/2601.06058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 338. Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms

**arXiv ID:** 2601.06039 | [PDF](https://arxiv.org/pdf/2601.06039v1)

**作者:** Yueze Liu `[一作]` (Divergence 2 percent LLC), Yichi Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 718 | [OpenAlex ID](https://openalex.org/A5100444202)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为VEJA（Values、Experiences、Judgments、Abilities）的框架，用于构建角色的内在世界并提升角色扮演模型的真实性和叙事连贯性。

**💡 创新点**

创新点在于将角色塑造拆解为四个因果链条（价值、经历、判断、能力），并以此为数据策划准则，以期突破传统模型仅凭事实或文本暗示的局限。

**🔧 技术方法**

使用了大型语言模型（Gemini Pro 2.5、Gemini 2.5 Flash）作为数据生成器和评估者，同时通过人工编写的对话对比实验来验证框架效果。

**📊 数据集**

主要数据集为两组对话：一组为自动化合成的基线数据，另一组为由15名人工作者按VEJA框架撰写的人类生成对话，聚焦角色Makise Kurisu的10天剧情。

**📈 对比分析**

通过LLM‑as‑Judge的A/B对比实验，基于100个对话对，VEJA手工生成的对话被优选43次，基线28次，剩余29次平局，表明VEJA方法在角色连贯性和情节深度上具备显著优势。

**⚠️ 局限性**

局限包括数据规模小、仅针对单一角色、评估依赖LLM可能带偏差、未拆解单个VEJA维度对结果的具体贡献，以及人工策划难以大规模复制。

---

## 339. DemMA: Dementia Multi-Turn Dialogue Agent with Expert-Guided Reasoning and Action Simulation

**arXiv ID:** 2601.06373 | [PDF](https://arxiv.org/pdf/2601.06373v1)

**作者:** Yutong Song `[一作]` (University of California), Amir Rahmani `[通讯]` (University of California)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了一个名为 DemMA 的多轮痴呆患者对话生成模型，能够同时模拟语言、情绪与非语言行为。

**💡 创新点**

创新点包括：①结合医学专家信息构建基于子类型的痴呆人设；②引入动作标签把多模态行为编码为文本；③使用 Chain‑of‑Thought 蒸馏把规划、推理、发言与动作统一到单一 LLM 中。

**🔧 技术方法**

使用大语言模型（Qwen3‑8B/32B 等）进行多任务微调，配合多智能体生成管道和 CoT 蒸馏技术。

**📊 数据集**

构建并公开了由专家验证的合成痴呆对话数据集 DemMA‑Dialogue，涵盖九种痴呆亚型。

**📈 对比分析**

与三种基线（Vanilla、Clinical‑Profile Prompt、SFT‑Utterance）对比，DemMA 在 7 项评估指标上均位列第一，LLM 评测平均分超过 4.0，专家评测平均 3.5‑4.0，显示显著提升。

**⚠️ 局限性**

主要局限：①数据完全合成，可能缺失罕见临床行为；②动作标签只能粗略表达多模态信号；③未建模疾病长期进展、药物或护理策略；④评估部分依赖 LLM 判定，可能存在偏差。

---

## 340. Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking

**arXiv ID:** 2601.06163 | [PDF](https://arxiv.org/pdf/2601.06163v1)

**作者:** Kaiyuan Deng `[一作]` (University of Arizona), Xiaolong Ma `[通讯]` (University of Arizona)

**通讯引用:** 7466 | [OpenAlex ID](https://openalex.org/A5074448953)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的多概念忘却框架FIA，用模型稀疏性实现同时遗忘多个不想要的概念；

**💡 创新点**

创新点在于：①通过对比概念重要性计算Contrastive Concept Saliency量化权重贡献；②结合时序与空间稀疏性识别Concept‑Sensitive与Concept‑Agnostic神经元；③采用多概念掩码融合策略，保证生成质量与忘却效果平衡；

**🔧 技术方法**

技术主要包括：基于能量的saliency度量、对比概念显著性计算、时序积分敏感度、空间稀疏选择、掩码融合、稀疏神经元剪枝；

**📊 数据集**

使用的数据集有：Stable Diffusion v1.5/1.4、Imagenette、I2P、MS COCO‑30K、Artist‑style prompt集（200幅每位艺术家）等；

**📈 对比分析**

与FMN、AC、ESD、SalUn、CP、MACE、UCE、SPM等多种单/多概念忘却方法对比，FIA在多概念遗忘准确率最低（如Imagenette 1.9%）、CLIP/生成质量保持最高，I2P检测总数从743降至32，艺术风格忘却成功率最高、FID/CLIP表现最优，综合排名居首；

**⚠️ 局限性**

局限性：当目标概念数目达到数百级时所需稀疏度增大，生成质量会逐渐下降；未来需与微调技术结合提升大规模多概念遗忘的质量。

---

## 341. Filtering Beats Fine Tuning: A Bayesian Kalman View of In Context Learning in LLMs

**arXiv ID:** 2601.06100 | [PDF](https://arxiv.org/pdf/2601.06100v1)

**作者:** Andrew Kiruluta `[一作]` `[通讯]` (University of California Berkeley), Andrew Kiruluta (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将大语言模型的推理时快速适应视为贝叶斯状态估计，通过低维适应子空间和Kalman递归实现在线推断；

**💡 创新点**

创新点在于把推理时学习建模为完整的贝叶斯滤波过程，显式地追踪后验协方差并证明其收敛性，揭示梯度下降和自然梯度是滤波的噪声消失极限；

**🔧 技术方法**

主要技术包括线性化参数映射、Kalman/扩展Kalman滤波、信息形式协方差更新、观测可观测性分析与协方差收敛定理；

**📊 数据集**

实验使用了合成线性回归数据和一个低秩适应子空间的toy LLM，未使用公开大规模文本数据集；

**📈 对比分析**

与传统梯度下降/随机梯度下降对比，实验仅展示协方差收敛和推理适应的稳定性，未给出精确的性能数值；

**⚠️ 局限性**

局限在于依赖局部线性化和高斯噪声假设，且在大模型中完整协方差更新成本高，需进一步验证在真实大型模型上的可扩展性与鲁棒性。

---

## 342. Exposía: Academic Writing Assessment of Exposés and Peer Feedback

**arXiv ID:** 2601.06536 | [PDF](https://arxiv.org/pdf/2601.06536v1)

**作者:** Dennis Zyska `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 25133 | [OpenAlex ID](https://openalex.org/A5027450194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首次公开了一个涵盖学生科研项目提案、同行与教师反馈以及修订过程的完整数据集，并基于细粒度评分标准对提案和评论进行人工评估。

**💡 创新点**

创新点在于整合了写作、反馈和修订三阶段的跨文档关系，并设计了专门针对学术写作与反馈的细粒度评估体系，填补了教育情境下缺乏综合数据资源的空白。

**🔧 技术方法**

采用了开源大型语言模型（如 Llama、GPT、Qwen 等）进行零样本评分，并通过单一与多维度联合提示策略实现自动评估。

**📊 数据集**

使用的数据集为 “Intro to Scientific Work” 课程收集的 55 篇提案、306 篇评审及 2,253 条内嵌评论，其中每份提案均配有草稿、最终稿及评分。

**📈 对比分析**

实验对比单一维度提示与联合提示，联合提示在多数任务上取得更高的二次方加权一致性（QWA）并降低计算成本，LLM 在非专家级维度的评分与人类评审高度一致，但在专家级维度仍表现欠佳。

**⚠️ 局限性**

局限性包括数据集规模有限、只包含授予同意的样本、同一学科范围局限、未评估封闭源模型以及模型可能产生系统性偏好，且不建议完全自动化评分，仅能作为决策辅助工具。

---

## 343. RainBalance: Alleviating Dual Imbalance in GNSS-based Precipitation Nowcasting via Continuous Probability Modeling

**arXiv ID:** 2601.06137 | [PDF](https://arxiv.org/pdf/2601.06137v1)

**作者:** Yifang Zhang `[一作]` (Wuhan University of Technology), Pengfei Duan `[通讯]` (Wuhan University of Technology)

**通讯引用:** 687 | [OpenAlex ID](https://openalex.org/A5080295555)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种可插拔模块RainBalance，用于缓解GNSS降水现在预测中的双重不平衡问题。

**💡 创新点**

创新点在于将离散降水标签映射到连续概率分布，通过聚类与VAE结合实现概率建模。

**🔧 技术方法**

采用样本聚类、变分自编码器、融合预测模块，并可集成到多种时间序列模型。

**📊 数据集**

使用全球140个GNSS站点的降水数据（J340、ZIMM、P095、JFNG等）进行实验。

**📈 对比分析**

与多种基线（xLSTM、TimeMixer、TimesNet等）对比，平均提升MSE/MAE 10–30%，在极端降水预测上提升29–35%。

**⚠️ 局限性**

局限在于对高频非降水噪声处理不充分，且VAE训练成本较高，未来需进一步验证在不同气候区的鲁棒性。

---

## 344. Toward Safe and Responsible AI Agents: A Three-Pillar Model for Transparency, Accountability, and Trustworthiness

**arXiv ID:** 2601.06223 | [PDF](https://arxiv.org/pdf/2601.06223v1)

**作者:** Edward C. Cheng `[一作]` (Stanford University), Alice Siu `[通讯]` (Stanford University)

**通讯引用:** 856 | [OpenAlex ID](https://openalex.org/A5036571346)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并阐述了一个以透明度、问责性和可信度为核心的三柱模型，用于安全可信的AI代理的设计与运营，并给出了具体的实践框架与示例。

**💡 创新点**

创新点在于将透明、问责、可信三大原则整合为可执行的演化路径，借鉴自动驾驶分阶段监管模式，结合人机协作与不确定性对齐，形成完整的工具与治理生态。

**🔧 技术方法**

采用人机在环、强化学习+人类反馈、LLM决策日志、活动记录与可观测性框架、开放式工具箱和协议（MCP、A2A等）来实现模型的透明、可追溯和安全性。

**📊 数据集**

论文未给出具体公开数据集，主要通过案例（如组邮件代理）和内部测试来验证框架。

**📈 对比分析**

没有提供量化实验或与现有方法的对比，讨论主要是基于案例演示和可视化指标，性能表现尚未在大规模基准上验证。

**⚠️ 局限性**

局限性包括缺乏系统化的实证评估与基准测试、对多样化场景的适用性未充分验证、对人机协作成本与可扩展性的考量不足，以及仍需进一步完善监管与合规机制。

---

## 345. Architecting AgentOps Needs CHANGE

**arXiv ID:** 2601.06456 | [PDF](https://arxiv.org/pdf/2601.06456v1)

**作者:** Shaunak Biswas `[一作]` (International Institute of Information Technology Hyderabad), Karthik Vaidhyanathan `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一套面向Agentic AI系统的AgentOps概念框架，包含六项关键能力，用于在系统生命周期内管理其自我学习、协作与演化。

**💡 创新点**

创新点在于将运营视角从传统的版本化、监控和回滚转向与代理、基础设施和人工监督共同演进的动态协作模式，并明确了经验状态、协同一致性、漂移预测、自治调节、工具自我生成与长期适应六大能力。

**🔧 技术方法**

采用概念建模和框架设计方法，结合LLMOps和MLOps的原则，对Agentic AI系统的运营进行抽象化描述；在示例中使用了基于LLM的代理交互、工具调用和外部服务集成。

**📊 数据集**

未使用具体数据集，而以客户支持场景（Alice和Bob两位代理）作为运行示例来说明框架的可行性。

**📈 对比分析**

尚未进行实验对比或性能评估，作者提出可通过多代理仿真平台验证，并建议使用行为一致性偏差、漂移预测准确率、协商成功率等指标进行评估，性能表现仍待后续实现。

**⚠️ 局限性**

局限性包括：缺乏实现细节和实测数据，评估指标尚未标准化；框架在不同应用场景下的通用性和可扩展性尚未验证；对长期演化过程中不可预见情境的治理机制仍不成熟。

---

## 346. Data Work in Egypt: Who Are the Workers Behind Artificial Intelligence?

**arXiv ID:** 2601.06057 | [PDF](https://arxiv.org/pdf/2601.06057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 347. Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers

**arXiv ID:** 2601.06338 | [PDF](https://arxiv.org/pdf/2601.06338v1)

**作者:** Binxu Wang `[一作]` (Harvard University), Xu Pan `[通讯]` (Harvard University)

**通讯引用:** 5923 | [OpenAlex ID](https://openalex.org/A5071861102)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Diffusion Transformers 在文本到图像生成中实现空间关系的机制进行研究，并通过可解释性分析揭示其内部电路。

**💡 创新点**

首次对比随机文本编码与预训练 T5 编码下的空间关系生成电路，展示不同编码导致的鲁棒性差异。

**🔧 技术方法**

采用注意力摘要（Attention Synopsis）、头部消融与因果操纵，以及对 T5 嵌入进行向量算术的机制可解释技术。

**📊 数据集**

构造了极简的双对象+空间关系数据集（12 种对象组合、8 种空间关系）来进行训练和评估。

**📈 对比分析**

在相同参数规模下，用 EMA 权重训练的 DiT 模型在四项指标上均达到近乎完美的准确率；RTE‑DiT 对文本扰动更鲁棒，T5‑DiT 对词序扰动更敏感。

**⚠️ 局限性**

局限性在于仅在极简数据集上验证，缺少对复杂真实场景的评估；最小模型表现差，T5 模型易受小扰动影响。

---

## 348. SourceNet: Interpretable Sim-to-Real Inference on Variable-Geometry Sensor Arrays for Earthquake Source Inversion

**arXiv ID:** 2601.06320 | [PDF](https://arxiv.org/pdf/2601.06320v1)

**作者:** Zhe Jia `[一作]` (University of Texas at Austin), Junpeng Li `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于Transformer的可解释框架 SourceNet，用于在变几何传感器阵列上实现地震源参数的实时逆推；

**💡 创新点**

创新点包括：①将传感器阵列建模为无序集合并使用 Set Transformer 捕捉全波形相互关系；②通过 Physics‑Structured Domain Randomization 随机化速度模型、噪声、散射和网络可用性，以弥合 Sim‑to‑Real 差距；③注意力机制能够自发识别信息瓶颈，实现无监督的最优实验设计；

**🔧 技术方法**

使用了 Set Transformer、物理结构化域随机化、三模态站点编码器（P 波、S 波、标量元），自注意力聚合与注意力池化，以及迁移学习与 Grad‑CAM 等解释技术；

**📊 数据集**

训练数据为 100,000 只合成地震事件（覆盖 17 种 1D 速度模型、真实噪声、散射及站点缺失），随后在约 2,500 只真实南加州 M>3 地震上微调；

**📈 对比分析**

与传统 Green’s 函数求解器、CNN、DeepSets 以及最新深度学习基线（如 FOCONET、DiTing）对比，在真实数据上得到平均 Kagan 角 26.2°、中位 19.4°，几乎达到人工标注误差下限，明显优于 30° 以上的传统方法；

**⚠️ 局限性**

局限性在于：①假设合成物理足以覆盖真实情形，难以处理多源或复杂断层滑移等 OOD 事件；②对随机化范围和模拟精度的依赖较大；③对极端稀疏阵列的鲁棒性仍待进一步验证。

---

## 349. QwenStyle: Content-Preserving Style Transfer with Qwen-Image-Edit

**arXiv ID:** 2601.06202 | [PDF](https://arxiv.org/pdf/2601.06202v1)

**作者:** Shiwen Zhang `[一作]` (Institute of Artificial Intelligence), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于Qwen-Image-Edit的首个内容保持风格迁移模型QwenStyle，支持通过风格与内容参考图像进行内容不变的风格迁移。

**💡 创新点**

创新点在于使用清洁与噪声三元组数据结合的课程连续学习框架，以及通过反向三元组合成方法扩充训练集，显著提升了对未知风格的泛化与内容保留能力。

**🔧 技术方法**

采用Diffusion Transformer（DiT）架构的Qwen-Image-Edit、MMDiT与MS-RoPE，使用LoRA微调、梯度检查点、CDST与DINO v2提取风格特征，并通过特定prompt进行训练与推理。

**📊 数据集**

构建了30类风格的真实三元组数据集D_collected，以及约100万条噪声三元组D_synthetic，随后在实验阶段使用了50种风格与40种内容图像的2000个风格-内容对进行评估。

**📈 对比分析**

在Style Similarity（CSD）、Content Preservation（CPC）和Aesthetic Score三个指标上与OmniStyle、OmniGen-v2、DreamO以及自研Style-CCL对比，QwenStyle V1在风格相似度0.577、内容保留0.441、审美得分6.317等方面均取得了最高或接近最高的成绩。

**⚠️ 局限性**

局限性包括：对离谱风格的泛化仍有限；合成三元组比例过高时可能导致内容一致性下降；模型对prompt和图像尺寸的依赖性较强，且当前仍处于实验阶段，需进一步提升鲁棒性。

---

## 350. Assessing the Carbon Footprint of Virtual Meetings: A Quantitative Analysis of Camera Usage

**arXiv ID:** 2601.06045 | [PDF](https://arxiv.org/pdf/2601.06045v1)

**作者:** Félix Mortas `[一作]` `[通讯]` (Independent Researcher), Félix Mortas (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了视频通话中摄像头开启与关闭对移动网络数据消耗及碳排放的影响，利用实测数据进行对比分析。

**💡 创新点**

创新点在于首次通过真实实验验证摄像头开启会导致数据消耗和碳排放近乎翻倍，填补了以往仅模型估计的空白。

**🔧 技术方法**

采用4G手机数据监测、Microsoft Teams会议软件和简单描述性统计（平均值）等技术手段进行数据收集与分析。

**📊 数据集**

实验使用9次30–60分钟会议的记录数据，包括摄像头状态、参与人数、屏幕共享等变量。

**📈 对比分析**

通过比较摄像头开启与关闭时每分钟平均数据量（18.07 MB vs 8.76 MB）以及对应的CO₂e（约0.17–0.36 g vs 0.08–0.18 g），显示摄像头关闭可将碳足迹减半，性能表现显著。

**⚠️ 局限性**

局限在于样本量小、未考虑屏幕共享与摄像头动态切换，仅在移动网络上测试，缺乏更广泛网络与多参与者场景的验证。

---

## 351. Can a Unimodal Language Agent Provide Preferences to Tune a Multimodal Vision-Language Model?

**arXiv ID:** 2601.06424 | [PDF](https://arxiv.org/pdf/2601.06424v1)

**作者:** Sazia Tabasum Mim `[一作]` (Georgia State University), Yi Ding `[通讯]` (Georgia State University)

**通讯引用:** 2736 | [OpenAlex ID](https://openalex.org/A5065432498)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种方法，让单模态大型语言模型（LLM）通过文本反馈来优化多模态视觉语言模型（VLM）的描述，从而提升多模态推理任务的性能。

**💡 创新点**

创新点在于首次让无视觉输入的LLM提供偏好反馈，用直接偏好优化（DPO）对VLM进行微调，使其生成更贴合LLM需求的文本描述。

**🔧 技术方法**

核心技术包括基于DPO的偏好优化、使用LoRA对VLM进行轻量级微调、以及多提示生成多样化视频描述。

**📊 数据集**

实验使用了两大多模态社交推理数据集：MUStARD（讽刺检测）和UR‑Funny（幽默检测）。

**📈 对比分析**

与基线多模态、仅语句和仅视觉的对比表明，偏好微调后的VLM在两任务上均提升了约10–13％的准确率，并在多模态条件下超过多数模型的仅语句基线。

**⚠️ 局限性**

主要限制包括：对强大模型（如Llama3.3‑70b）VLM描述贡献有限；微调采用离线批处理，缺乏实时适应；且生成描述仍存在幻觉现象。

---

## 352. RigMo: Unifying Rig and Motion Learning for Generative Animation

**arXiv ID:** 2601.06378 | [PDF](https://arxiv.org/pdf/2601.06378v1)

**作者:** Hao Zhang `[一作]` (Snap Inc.), Bing Zhou `[通讯]` (Snap Inc.)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 RigMo，联合学习骨骼结构与运动的生成框架，从原始网格序列中无需手工注解自动推断 Gaussian 骨骼与运动参数。

**💡 创新点**

将骨骼与运动在统一潜在空间分离学习，使用 Gaussian 骨骼与分离的空间时间潜在表示实现可解释、可控的动画，并引入 Motion‑DiT 在该潜在空间生成运动，突破传统单独装配或仅预测运动的局限。

**🔧 技术方法**

基于双路径 VAE 编码器-解码器、Gaussian 皮肤化、线性混合皮肤化 (LBS)、拓扑感知注意力、几何一致性约束，以及基于扩散 Transformer 的 Motion‑DiT。

**📊 数据集**

DeformingThings4D、Objaverse‑XL、TrueBones 三大数据集，涵盖真实非刚性形变、合成多种类别以及高质量骨骼动画。

**📈 对比分析**

与 Per‑Case 优化、UniRig+优化、MagicArticulate+优化等基线相比，RigMo 在骨骼精度、Chamfer Distance、跨运动通用性上均优于基线，且推断速度最快（单帧约 40 ms）。

**⚠️ 局限性**

对极端形状或极复杂拓扑的形体仍可能产生不合理骨骼；骨骼数量需要平衡，过多可能导致分割细化但收益递减。

---

## 353. What Matters When Building Universal Multilingual Named Entity Recognition Models?

**arXiv ID:** 2601.06347 | [PDF](https://arxiv.org/pdf/2601.06347v1)

**作者:** Jonas Golde `[一作]` (Humboldt Universität zu Berlin), Alan Akbik `[通讯]` (Humboldt Universität zu Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言命名实体识别（NER）的架构、Transformer backbone、损失函数和训练数据进行系统化实验，并基于实验结果提出一种跨语言实体识别模型；

**💡 创新点**

在统一实验框架下量化各设计维度的影响，发现跨编码器+mmBERT+二元交叉熵最佳，并以此构建支持100多种语言的模型；

**🔧 技术方法**

采用跨编码器/双编码器架构、mmBERT backbone、二元交叉熵损失函数以及子词级span推理；

**📊 数据集**

使用FiNERWeb（91语种）、PileNER、Euro-GLiNER-x等数据进行训练，评估在DynamicNER、UNER、MasakhaNER、MultiCoNER等七个多语言基准；

**📈 对比分析**

与GLiNER-x-base、WikiNeural、LLM Qwen3-32B/Gemma3-27B等对比，F1平均提升约5.3个百分点，性能仅比大型生成式模型低0.3个百分点，同时训练和推理更高效；

**⚠️ 局限性**

受限于预训练模型和训练数据的覆盖范围、低资源语言表现不足、标签语义差异以及阈值选择对不同语言的高度依赖导致的局限。

---

## 354. Causal and Federated Multimodal Learning for Cardiovascular Risk Prediction under Heterogeneous Populations

**arXiv ID:** 2601.06140 | [PDF](https://arxiv.org/pdf/2601.06140v1)

**作者:** Rohit Kaushik `[一作]` (Hanson Professional Services), Eva Kaushik `[通讯]` (University of Tennessee)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5016394768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并实现了一个整合基因组、心脏MRI、心电图、可穿戴传感器和EHR的多模态联邦因果学习框架，用于预测心血管疾病风险。

**💡 创新点**

创新点在于将跨模态Transformer、图神经网络与因果潜在一致化相结合，并在联邦学习环境下实现隐私保护、可解释性与公平性。

**🔧 技术方法**

使用技术包括跨模态Transformer、图注意力网络、因果潜在一致化（IV正则化）、FedAvg联邦聚合、差分隐私、SHAP与对抗解释、蒙特卡洛Dropout贝叶斯不确定性估计。

**📊 数据集**

采用数据集包括UK Biobank（50k+人）全模态数据、MIT/BIH Arrhythmia与PhysioNet（10k人）高分辨率心电图/可穿戴数据、跨洲联邦医院EHR（200k+人）以及合成数据进行鲁棒性与公平性验证。

**📈 对比分析**

通过与逻辑回归、随机森林、单模态CNN‑LSTM等基线模型在10折交叉验证和留站外验证中比较，AUC提升至0.994，公平性指标ΔAUC<0.001，置信区间覆盖率达到95.3%。

**⚠️ 局限性**

局限性包括缺乏前瞻性/干预验证、潜在未观测混杂、联邦训练通信成本高、解释性与因果假设可能不完全成立、对缺失数据处理不完善以及对抗攻击仍有风险。

---

## 355. Applied Theory of Mind and Large Language Models - how good is ChatGPT at solving social vignettes?

**arXiv ID:** 2601.06032 | [PDF](https://arxiv.org/pdf/2601.06032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 356. A survey of facial recognition techniques

**arXiv ID:** 2601.06239 | [PDF](https://arxiv.org/pdf/2601.06239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 357. Automated dimensional analysis for PDEs

**arXiv ID:** 2601.06535 | [PDF](https://arxiv.org/pdf/2601.06535v1)

**作者:** Michal Habera `[一作]` (University of Luxembourg), Andreas Zilian `[通讯]` (University of Luxembourg)

**通讯引用:** 600 | [OpenAlex ID](https://openalex.org/A5073102883)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在有限元变分形式语言UFL中实现统一的物理量单位跟踪、维度一致性检查与自动非量纲化；通过符号量化实现物理单位的自动注入和因子化；并将该框架嵌入Dolfiny库以支持Navier–Stokes、Neo‑Hooke弹性和Poisson–Nernst–Planck等多物理问题；

**💡 创新点**

1）首次将物理单位系统与UFL的抽象语法树（DAG）结合，实现全自动维度一致性验证和非量纲化；2）利用维度的阿贝尔群结构将其映射为有理数向量，极大简化运算与内存占用；3）将非量纲化视为物理感知的全算子预调（Full Operator Preconditioning），从而在组装之前就完成数值尺度均衡；

**🔧 技术方法**

符号量化（SymPy）物理单位模块；UFL的访问者（visitor）模式遍历DAG；向量化的维度表示（ℚ^n）；Python实现的Dolfiny插件；

**📊 数据集**

无传统数据集；通过数值实验（Navier–Stokes、Neo‑Hooke、PNP）验证，利用FEniCSx、Firedrake、DUNE等仿真框架；

**📈 对比分析**

对比了未加非量纲化与加非量纲化后线性系统的条件数与迭代收敛；实验表明：在Navier–Stokes中，选择合适的参考压强可使条件数从 10⁶ 降至 10²，显著加快 GMRES 收敛；在弹性问题中，因子不一致导致浮点舍入失真，非量纲化能预先捕获并消除该风险；

**⚠️ 局限性**

1）仅支持标量或各分量单位相同的向量/张量；2）UFL当前缺乏对部分算子（如时间导数、某些自定义算子）的内置支持，需用户自行扩展；3）尚未针对高阶多重尺度问题的自适应非量纲化策略；

---

## 358. Digital health transformation in Quebec: assessment of interoperability and governance strategies

**arXiv ID:** 2601.06051 | [PDF](https://arxiv.org/pdf/2601.06051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 359. La norme technique comme catalyseur de transfert de connaissances : la francophonie a l'œuvre dans le domaine de l'{é}ducation

**arXiv ID:** 2601.06069 | [PDF](https://arxiv.org/pdf/2601.06069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 360. CARD: Cluster-level Adaptation with Reward-guided Decoding for Personalized Text Generation

**arXiv ID:** 2601.06352 | [PDF](https://arxiv.org/pdf/2601.06352v1)

**作者:** Yutong Song `[一作]` (University of California), Yu Wang `[通讯]` (TikTok)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于分层的个性化文本生成框架 CARD，先通过聚类学习群组级 LoRA 适配，再在解码时注入轻量级用户偏好向量，实现高效、可扩展的个性化。

**💡 创新点**

创新点在于将群组共享与个体差异分层解耦：使用群组 LoRA 捕获共性、通过无监督的输入对齐偏好对构造实现细粒度个体差异的学习，并仅在解码阶段注入低秩 logit 调整，既保持模型冻结又实现高质量个性化。

**🔧 技术方法**

技术包括 K‑means 聚类、LoRA 参数微调、用户历史编码、偏好向量投射、低秩 logit 修正、Bradley‑Terry 对比损失以及基于 GPT‑5.2 的自动评估。

**📊 数据集**

实验使用公开的 LaMP 与 LongLaMP 个性化生成基准数据集。

**📈 对比分析**

与 RAG、PAG、PAD、PPLUG、OPPU 等多种基线对比，CARD 在 ROUGE‑1/ROUGE‑L 上多任务处于 1/2 位，显著提升了生成质量，同时在训练时间、存储与查询延迟上实现更优的效率。

**⚠️ 局限性**

局限性包括：依赖无监督 K‑means 可能无法捕获复杂用户关系；单一向量表达缺乏可解释性；对历史数据质量敏感；未建模偏好随时间变化的漂移。

---

## 361. AI Safeguards, Generative AI and the Pandora Box: AI Safety Measures to Protect Businesses and Personal Reputation

**arXiv ID:** 2601.06197 | [PDF](https://arxiv.org/pdf/2601.06197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 362. AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning

**arXiv ID:** 2601.06086 | [PDF](https://arxiv.org/pdf/2601.06086v1)

**作者:** Yiwen Shao `[一作]`, Dong Yu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Self‑Generated Instruction‑Free Tuning (SIFT) 方法，利用冻结的大语言模型自行生成监督信号，将语音编码器与语言模型对齐，训练时仅更新轻量级投影器，构建了 AZeroS 语音‑LLM。

**💡 创新点**

创新点在于：①去除了任务特定指令、任务数据的需求；②通过自生成的文本响应实现语音与文本的完整对齐，理论上可获得最佳泛化；③仅冻结 LLM 与音频编码器，仅更新约 23.8M 参数，训练成本极低；④在同等规模下实现语义与声学双重理解。

**🔧 技术方法**

技术细节：冻结 Qwen2.5‑7B‑Instruct 语言模型；使用 TTA（语义）和 Auden‑Voice（声学）两种音频编码器；投影器为两层 MLP；SIFT 训练流程：生成 oracle 文本 → 让冻结 LLM 自由生成完整回答 → 通过交叉熵损失更新投影器；两阶段训练：先对语义对齐，再对声学进行联合对齐。

**📊 数据集**

数据集：约 25k 小时公开语音，其中 3,548h 带语义转录与声学标签（Data‑SP），22,383h 仅语义转录（Data‑S）；使用 CommonVoice、WenetSpeech、GigaSpeech 等公开语料；对语音进行 ASR 转写后生成训练样本。

**📈 对比分析**

与文本模型、级联系统、现有端到端模型对比：在 VoiceBench 上接近文本上限（≈77.5 分），在 AIR‑Bench 上获得最高分（语义 60.22、声学 86.75），显著优于 Whisper+GPT‑4o、Qwen2‑Audio、Smosi 等基线；一阶段与两阶段训练对比显示两阶段略优；单编码器与双编码器对比表明双编码器更好，但核心提升来自 SIFT。

**⚠️ 局限性**

局限性：①需要冻结 LLM 具备自发“自举”能力，若 LLM 对自发细节缺乏则须恢复指令；②目前仅验证在语音模态，未扩展到视觉或其他多模态；③数据规模虽小但仍需 25k 小时公开语料，若面向更稀有语种或专业领域可能受限；④对极端噪声或说话者多样性的鲁棒性待进一步评估。

---

## 363. Resource-Aware Task Allocator Design: Insights and Recommendations for Distributed Satellite Constellations

**arXiv ID:** 2601.06706 | [PDF](https://arxiv.org/pdf/2601.06706v1)

**作者:** Bharadwaj Veeravalli `[一作]` (National University of Singapore), Bharadwaj Veeravalli `[通讯]` (National University of Singapore)

**通讯引用:** 6896 | [OpenAlex ID](https://openalex.org/A5070594442)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

设计并仿真评估了一种面向LEO至低MEO星座的资源感知任务分配器（RATA），并通过事件驱动离散仿真检验其在不同星座规模和任务类型下的阻塞、响应时间及能耗性能。

**💡 创新点**

创新点在于：①将任务可分割度（DTN）与卫星间协作分配（CoPAA）结合，构建可动态扩展的单层树网络（SLTN）；②引入太阳能感知调度，在光照周期内优先接受任务；③通过经验公式揭示阻塞、响应时间与星座规模的非线性/指数关系，标识了系统性能从平稳衰退到崩溃的阈值。

**🔧 技术方法**

使用技术包括：事件驱动离散仿真框架、VRAC/CoPAA 根/协作调度算法、处理时间与能耗估算模型、光照检测与能量收支模拟，以及基于Poisson到达率的任务流生成。

**📊 数据集**

数据集：仿真生成约25万条任务，任务尺寸在2–15 GB，计算强度25 M–2.5 B FLOP/MB；硬件参数设定为典型小卫星/CubeSat（20 GFLOPS、4核、128 GB内存、280 Wh电池、100 W太阳能），覆盖20、55、90、120颗卫星四种星座规模。

**📈 对比分析**

比较方法：在四种星座规模下，对三类任务（SatToSat、SatToGnd、GndToSat）分别统计阻塞概率、平均/最大响应时间、能耗与充电效率。结果显示：SatToSat阻塞呈1.18次幂增长，SatToGnd/​GndToSat响应时间随规模指数增长（约2.3次幂），能耗在太阳能调度下保持稳定，净能耗基本为零。整体性能随星座规模增大而出现超线性衰退，尤其在120颗卫星时SatToSat阻塞突破60%，SatToGnd平均延迟超过13天。

**⚠️ 局限性**

局限性：①仅在单轨道、单基站、无多链路冗余的理想环境下评估；②未实现故障检测与恢复，仅通过根/协作 fallback 近似；③仿真基于平均硬件与静态资源参数，未考虑设备老化、异常波动和实际链路时延；④任务类型与尺寸分布采用简化假设，未覆盖全部真实场景。

---

## 364. BlazeAIoT: A Modular Multi-Layer Platform for Real-Time Distributed Robotics Across Edge, Fog, and Cloud Infrastructures

**arXiv ID:** 2601.06344 | [PDF](https://arxiv.org/pdf/2601.06344v1)

**作者:** Cedric Melancon `[一作]`, Simon Savard `[通讯]` (École de technologie supérieure)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一套名为BlazeAIoT的多层分布式机器人平台，集成了边缘、雾端与云端三层网络，支持动态数据传输、可配置服务和实时监控；

**💡 创新点**

在分布式机器人框架中首次实现了基于服务的动态数据桥接、适应性速率限制和跨层无缝通信，兼顾多语言支持与ROS2集成；

**🔧 技术方法**

采用Kubernetes（K3S/RKE2）、Rancher、Submariner、DDS、Kafka、Redis、MongoDB、Prometheus/Graphana、LZ4压缩等开源技术；

**📊 数据集**

实验使用实际机器人平台（Pioneer 3-AT + Jetson Orin AGX）、SLAM、UWB、MediaPipe手势模型训练集HAGRID；

**📈 对比分析**

通过两组场景（导航与紧急停机）评估，发现局部执行（边缘）误差最小，远程高性能节点可显著缩短停机响应时间，整体延迟与吞吐在设定阈值内；

**⚠️ 局限性**

局限包括未实现连接中断恢复与自动负载平衡，边缘节点内存/CPU压力大导致帧率下降，速率限制可能导致重要信息丢失。

---

## 365. Foundational Analysis of Safety Engineering Requirements (SAFER)

**arXiv ID:** 2601.06335 | [PDF](https://arxiv.org/pdf/2601.06335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 366. Softly Induced Functional Simplicity Implications for Neural Network Generalisation, Robustness, and Distillation

**arXiv ID:** 2601.06584 | [PDF](https://arxiv.org/pdf/2601.06584v1)

**作者:** Maciej Glowacki `[一作]` (European Organization for Nuclear Research), Maciej Glowacki `[通讯]` (European Organization for Nuclear Research)

**通讯引用:** 281 | [OpenAlex ID](https://openalex.org/A5105673584)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在高能物理中对Jet分类任务进行实验，比较了在Transformer模型中加入软对称性约束（SEAL）与不加入约束两种训练方式，并通过Hessian分析和压缩（蒸馏）来评估学习到的解的功能复杂度；

**💡 创新点**

创新点在于首次揭示软对称性约束能在损失几何中产生伪Goldstone模式，显著降低有效维度与曲率，从而提升模型的鲁棒性、对分布外数据的泛化能力，并使得模型更易被压缩为低容量网络；

**🔧 技术方法**

采用Transformer编码器、SEAL（对Lorentz变换的MSE正则）、权重衰减、跳跃连接；随后使用Hessian的主特征值与迹量化曲率、利用蒸馏实现可压缩度评估，并对输入扰动与OOB数据进行鲁棒性和泛化测试；

**📊 数据集**

使用公开的Jet数据集，该数据集包含由gluons、quarks与bosons产生的Jet，最多32个粒子，特征包括Δη、Δϕ、log(pT)、log(E)等；

**📈 对比分析**

在相同的训练设置下，约束模型与非约束模型在分布内准确率约94%、ROC‑AUC≈0.99基本一致；但约束模型的主特征值与迹比率分别为0.079和0.119，显著更平坦；在输入pT噪声下，约束模型的ROC‑AUC衰减更慢；在近似OOB（Gluon vs Z）和远距离OOB（Gluon vs Top）下，约束模型分别提高了≈1%和≈0.9%的ROC‑AUC；蒸馏实验显示约束教师更快收敛、最终MSE更低；

**⚠️ 局限性**

局限性包括：仅针对单一Jet分类任务、仅考察Lorentz软对称约束，缺乏对硬对称或其它物理对称的探索；Hessian估计受样本与数值误差影响；结果的可推广性至其他网络结构或更广泛的高能物理任务仍需验证；

---

## 367. Modeling Descriptive Norms in Multi-Agent Systems: An Auto-Aggregation PDE Framework with Adaptive Perception Kernels

**arXiv ID:** 2601.06557 | [PDF](https://arxiv.org/pdf/2601.06557v1)

**作者:** Chao Li `[一作]` (ITMO University), Sergey Kovalchuk `[通讯]` (ITMO University)

**通讯引用:** 1389 | [OpenAlex ID](https://openalex.org/A5029904389)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种基于偏微分方程的自动聚合模型，用来模拟多智能体系统中描述性规范的传播与演化，结合非局部感知核和外部势场实现规范趋同与违规两种动力学，并将该框架应用于真实 COVID‑19 医疗数据，验证了临床指南与新变种的上下游效应。

**💡 创新点**

创新点包括：① 用 PDE 而非传统的马尔可夫博弈或贝叶斯猜测模型直接刻画意见分布的动态；② 通过自适应感知核参数化建模个体对异质意见的关注度，实现自聚合或自回避；③ 将外部势场引入迁移项，实现基于规范目标的全局引导；④ 将连续意见空间与多元高斯混合模型相结合，既可描述客观规范也可捕捉主观感知；⑤ 在实际医学场景中，将临床指南发布和变种出现映射为外部输入，并通过因果推断与 Wasserstein 距离量化验证模型效果。

**🔧 技术方法**

使用技术包括：偏微分方程求解（扩散-迁移方程）、非局部梯度理论、可自适应核函数与高斯混合模型（GMM）、KDE、Wasserstein 距离评估、线性稳定性分析、双重机器学习（DoubleML）因果推断、数值模拟（Euler 或有限差分）、可视化（3D 曲面、距离随时间演化曲线）。

**📊 数据集**

数据集为某大型医疗中心的 COVID‑19 病例记录（共 6,188 条观测，1,992 个独立病例），包含 33 个治疗控制变量、14 个动态状态变量和若干元数据，时间覆盖 2020 年 5 月至 2021 年 3 月，按临床指南发布和变种检测划分为 5 个时间段。

**📈 对比分析**

对比方法：在三组实验（top‑down 收敛、bottom‑up 违规、纯自适应）中，利用 Wasserstein 距离与目标 GMM 的差距作为性能指标；实验 1 在 0.2965（相对目标 GMM）收敛；实验 2 在 3.42（相对 10‑期数据）表现中等；实验 3 在 8.94（相对 4‑期 GMM）显示多峰分散，表明缺乏全局引导。实验表明模型能捕捉规范趋同与违规，但对外部势场精度和高维扩展敏感。

**⚠️ 局限性**

局限性包括：① 仅在一维意见空间实验，难以推广至更高维度；② 外部势场参数手工设定，缺乏自动校准；③ 模型对核函数、扩散系数等参数高度敏感，需大量调参；④ 计算量随格点细化呈指数级增长；⑤ 在完全自治场景下未实现规范自发诱导，系统易出现多中心分散；⑥ 真实数据噪声、缺失及标注不一对结果产生误差。

---

## 368. Context Video Semantic Transmission with Variable Length and Rate Coding over MIMO Channels

**arXiv ID:** 2601.06059 | [PDF](https://arxiv.org/pdf/2601.06059v1)

**作者:** Bingyan Xie `[一作]` (Shanghai Jiao Tong University), Merouane Debbah `[通讯]` (Khalifa University of Science and Technology)

**通讯引用:** 63755 | [OpenAlex ID](https://openalex.org/A5056145687)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一种结合语义上下文与多参考熵编码的MIMO视频语义传输框架（CVST），实现单模型可自适应多速率和多信道的无线视频传输；

**💡 创新点**

核心创新点包括：①学习上下文-信道相关映射，实现CSI感知的特征分配；②基于多参考的可变长度/可变速率熵编码与棋盘式特征调制，实现一次训练即可覆盖多CBR；③融合运动向量与上下文的多参考编码器，平衡两者带宽分配；

**🔧 技术方法**

采用深度学习中的CNN、Transformer、SVD预编码、注意力与多参考熵模型、特征调制以及Hungarian线性分配等技术；

**📊 数据集**

在Vimeo‑90k训练集上进行学习，以UVG和HEVC各类测试集（ClassA‑D）评估性能；

**📈 对比分析**

与传统VVC/H.265+LDPC+QAM分离编码、以及最新的DVSC、DVST等深度JSCC方案对比；在MIMO Rayleigh/3GPP CDL信道下，CVST在PSNR、MS‑SSIM、LPIPS等指标上比对手提升约2‑4 dB，且在不同CBR与SNR下保持更平滑的性能；

**⚠️ 局限性**

局限包括：依赖完美CSI或高质量CSI估计/反馈，误差会显著影响性能；对大规模MIMO的复杂度与延迟仍有挑战；目前仅考虑单用户、单用户MIMO，未覆盖多用户或多频段场景；

---

## 369. AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving

**arXiv ID:** 2601.06288 | [PDF](https://arxiv.org/pdf/2601.06288v1)

**作者:** Tianhao Xu `[一作]` (NVIDIA), Junjie Lai `[通讯]` (NVIDIA)

**通讯引用:** 563 | [OpenAlex ID](https://openalex.org/A5103287856)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了AIConfigurator工具，利用数据驱动的原语级性能建模，在不依赖GPU运行的情况下，快速搜索并给出多框架、多模式（静态、聚合、分离）下的LLM推理最佳配置。

**💡 创新点**

创新点包括：① 将LLM推理拆解为可度量的操作原语（GEMM、Attention、通信、MoE等）并构建跨框架的性能数据库；② 在三种推理模式下统一建模并实现低成本的搜索；③ 对MoE负载不均使用功率律补偿实现更精准的预测；④ 通过离线数据采集与插值相结合，实现CPU端秒级搜索。

**🔧 技术方法**

采用的技术：操作级性能数据库构建、插值与速度光速估计、功率律补偿、离线数据采集、离散事件式搜索、速率匹配与Pareto分析。

**📊 数据集**

使用的数据集与实验模型：H100、Hopper、A100 GPU平台上的多模型（Qwen3-32B、Qwen3-235B、DeepSeek V3、Llama3.1‑8B、Llama3.1‑70B等），覆盖密集与MoE架构，并在TensorRT‑LLM、vLLM、SGLang三大框架上进行实验。

**📈 对比分析**

与GPU真实跑测比较：对TPOT和TTFT的MAPE分别在7.8%（TPOT）和22%（TTFT）之间；系统吞吐量与生成速度预测误差在25%以内，且在交互速率区间（25–50 token/s/user）可降至13%与3%；搜索效率相比GPU基准提升至数十至数十万倍，单配置模拟时间约1.5ms。

**⚠️ 局限性**

局限性：① 对极端排队与超长TTFT的预测不够精确；② 目前仅支持NVIDIA Ampere、Ada、Hopper、Blackwell等GPU，其他硬件需重新采样；③ 对新框架或新技术（如稀疏注意力、speculative decoding）的兼容性待验证；④ 交叉节点延迟、网络拥塞等系统层面细节在模型中简化，可能导致大规模多机部署时的误差。

---

## 370. OptFormer: Optical Flow-Guided Attention and Phase Space Reconstruction for SST Forecasting

**arXiv ID:** 2601.06078 | [PDF](https://arxiv.org/pdf/2601.06078v1)

**作者:** Yin Wang `[一作]` (Shandong University of Finance and Economics), Xiang Wu `[通讯]` (Anqing Normal University)

**通讯引用:** 1903 | [OpenAlex ID](https://openalex.org/A5006755579)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了OptFormer模型，用于海表温度（SST）时空预测，结合相位空间重构和光流引导注意力机制；

**💡 创新点**

创新点在于将相位空间重构与光流辅助注意力融合，利用运动信息突出动态区域，增强长程时空依赖建模；

**🔧 技术方法**

技术包括相位空间重构、光流估计、光流引导注意力（Optical Attention）、Inception多尺度编码以及自相关模块；

**📊 数据集**

使用NOAA OISST v2.1日常海表温度数据，覆盖中大西洋区域；

**📈 对比分析**

与LSTM、Informer、AutoFormer、N‑BEATS、TCN等主流方法在相同训练‑预测比例下对比，OptFormer在RMSE和MAPE上平均降低约70‑80%，并在不同时间尺度、预测长度、季节和空间子区域中保持最优性能；

**⚠️ 局限性**

局限性在于目前仅使用单变量SST，未考虑风应力、潮流等外部驱动，且模型对多源数据融合及更复杂耦合过程的适应性仍待提升。

---

## 371. Reinforcement Learning-Guided Dynamic Multi-Graph Fusion for Evacuation Traffic Prediction

**arXiv ID:** 2601.06664 | [PDF](https://arxiv.org/pdf/2601.06664v1)

**作者:** Md Nafees Fuad Rafi `[一作]` (University of Central Florida), Samiul Hasan `[通讯]` (University of Central Florida)

**通讯引用:** 4510 | [OpenAlex ID](https://openalex.org/A5056217950)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种基于强化学习引导的动态多图融合框架，用于预测飓风撤离期间的网络级交通流量。

**💡 创新点**

创新点在于：①同时构造距离图和行驶时间图并通过注意力机制融合；②采用DDQN强化学习实现智能特征选择与排名，提升模型可解释性与泛化能力。

**🔧 技术方法**

技术手段包括图卷积网络 (GCN)、长短期记忆网络 (LSTM)、注意力融合、双重深度Q网络 (DDQN) 与强化学习特征掩码。

**📊 数据集**

使用佛罗里达州交通检测器的高时空分辨率数据，涵盖 2016‑2024 年 12 次飓风（I‑4、I‑10、I‑75、I‑95、Turnpike）以及相关撤离与事故信息。

**📈 对比分析**

与传统 LSTM、CNN‑LSTM、静态/动态 GCN‑LSTM 等基线模型比较，RL‑DMF 在 1–6 小时预测期内均取得最低 RMSE（总体 426.4）和最高 R²（0.90），准确率达到 90‑95%。

**⚠️ 局限性**

局限性包括：依赖历史交通模式难以即时响应极端天气或突发事件；RL 特征掩码共享所有节点，缺乏局部化解释；目前仅在佛罗里达验证，需进一步适配其他地区。

---

## 372. Follow the Signs: Using Textual Cues and LLMs to Guide Efficient Robot Navigation

**arXiv ID:** 2601.06652 | [PDF](https://arxiv.org/pdf/2601.06652v1)

**作者:** Jing Cao `[一作]` (Massachusetts Institute of Technology), Aidan Curtis `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 271 | [OpenAlex ID](https://openalex.org/A5062719305)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种结合大语言模型与传统前沿探索的语义导航框架，让机器人在未知室内环境中通过解析文本标签和房间编号模式快速定位目标房间。

**💡 创新点**

创新点在于将LLM的语义推理结果映射到置信度网格，并与前沿探索和A*路径规划耦合，从而在无先验地图的情况下实现高效的长程语义引导导航。

**🔧 技术方法**

使用技术包括：大语言模型（LLM）进行语义模式推断、置信度网格更新、前沿探索策略、A*全局规划和基于网格的可视化感知。

**📊 数据集**

数据集涵盖七个不同规模的二维网格环境（包括三种合成小型环境、三种基于真实平面图的大型环境以及一个噪声Polycam扫描环境），并在Boston Dynamics Spot机器人上进行了真实世界验证。

**📈 对比分析**

与NavGPT、LLM Only和单纯前沿探索基线对比，本文方法在七个环境中平均SPL提升约25%（最高达0.745），成功率与路径长度均优于对手，且在大地图和噪声场景下表现更为稳健。

**⚠️ 局限性**

主要局限包括对人工标注的文本标签和完整可视窗口依赖、对房间编号模式模糊或不一致时易误判、以及在噪声扫描地图中可能被误导导致路径偏离。

---

## 373. KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks

**arXiv ID:** 2601.06633 | [PDF](https://arxiv.org/pdf/2601.06633v1)

**作者:** Zhangqi Duan `[一作]` (University of Massachusetts Amherst), Andrew Lan `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1762 | [OpenAlex ID](https://openalex.org/A5063813962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种基于强化学习的知识对齐学生错误模拟器（Knowledge‑Aligned Student Error Simulator），能够根据学生的知识掌握情况生成带有对应错误的开源编程代码。

**💡 创新点**

创新点在于将代码相似度、错误匹配和代码多样性三项奖励融合进GRPO训练框架，既能对齐学生知识水平，又能避免模式崩溃，显著提升错误模拟的多样性和准确性。

**🔧 技术方法**

使用的技术包括Qwen2.5‑Coder大模型、知识跟踪（KT）模块、GRPO强化学习算法、CodeBLEU和IoU评价指标，以及基于Sentence‑BERT的错误聚类与标注。

**📊 数据集**

实验采用了两套真实学生编码数据集——Java版CodeWorkout（50题、10,834条提交）和Python版FalconCode（84题、11,194条提交）。

**📈 对比分析**

在5折交叉验证中，与PersonaPrompt、ICL、ParaStudent、Student SFT等基线对比，本文方法在CodeBLEU、IoU、代码多样性（余弦距离、最大CodeBLEU）等指标上均显著优于所有基线。

**⚠️ 局限性**

主要局限在于错误标注完全依赖LLM，缺乏完整的语法错误覆盖和真实测试用例；评估指标IoU普遍偏低，说明模型在捕捉语法错误方面仍有不足；未考虑多轮调试错误，也未充分评估对不同群体的公平性。

---

## 374. Efficient and Reliable Estimation of Named Entity Linking Quality: A Case Study on GutBrainIE

**arXiv ID:** 2601.06624 | [PDF](https://arxiv.org/pdf/2601.06624v1)

**作者:** Marco Martinelli `[一作]` (University of Padua), Gianmaria Silvello `[通讯]` (University of Padua)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5078254809)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于分层两阶段聚类抽样（STWCS）的采样框架，用于在有限专家标注预算下对 GutBrainIE 语料库的命名实体链接（NEL）准确性进行估计。

**💡 创新点**

将知识图谱准确性估计中的 STWCS 采样方法迁移到 NEL 任务，定义与标签和表面形式无关的分层与聚类，从而在保持统计置信度的前提下显著降低标注成本和上下文切换。

**🔧 技术方法**

分层两阶段聚类抽样（STWCS）、置信区间与误差界限估计、基于上下文切换的成本模型、Streamlit 接口和实时 MoE 监控脚本。

**📊 数据集**

GutBrainIE 2025 版，包含 11,184 条实体-概念链接，涵盖 13 种实体类型与 17 种关系。

**📈 对比分析**

与简单随机抽样（SRS）基线对比，STWCS 在相同抽样量下减少约 29% 标注时间（从 13h17m 降至 8h49m），并在仅标注 24.6% 三元组的条件下得到整体准确率 0.915 ± 0.0473。

**⚠️ 局限性**

采样设计仍依赖单一评估者，无法捕捉标注者间差异；分层与聚类依据表面形式可能忽略上下文多义性；对低频实体类型的精度估计仍不稳定。

---

## 375. Industrial Semantics-Aware Digital Twins: A Hybrid Graph Matching Approach for Asset Administration Shells

**arXiv ID:** 2601.06613 | [PDF](https://arxiv.org/pdf/2601.06613v1)

**作者:** Ariana Metović `[一作]` (University of Stuttgart), Oliver Riedel `[通讯]` (University of Stuttgart)

**通讯引用:** 805 | [OpenAlex ID](https://openalex.org/A5054826054)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种混合图匹配方法，用于在工业数字孪生中基于资产管理壳（AAS）进行语义可比的检索和重用。

**💡 创新点**

创新点在于将基于规则的 SPARQL 预筛选与基于 RDF2vec 的向量相似度计算相结合，既利用结构化查询捕获 AAS 的语义约束，又通过嵌入捕捉词汇差异和语义相似性，提升了异构模型间的匹配质量。

**🔧 技术方法**

所用技术包括 RDF 与 SPARQL 查询、RDF2vec 图嵌入、余弦相似度/阈值/Top‑k 策略，以及基于词向量的向量空间检索。

**📊 数据集**

使用的 dataset 包括将工业 AAS 实例（JSON/JSON‑LD）转换为 RDF 的公开仓库，并构造了包含多种词汇表和部分模板实现的 toy 数据集。

**📈 对比分析**

比较方法先通过 SPARQL 过滤出结构满足缺失子模型等约束的候选 AAS，然后用 RDF2vec 生成嵌入向量并计算余弦相似度；实验表明在 toy 数据集上能显著提升匹配准确率，但尚未给出大规模定量性能指标。

**⚠️ 局限性**

局限性包括：需要先验的结构知识才能写好 SPARQL 约束，导致对不熟悉建模规范的用户不友好；词汇多样性仍会导致过滤后候选集包含语义相近但结构差异较大的实例；嵌入训练与相似度计算在大型 AAS 仓库中可能存在计算成本和可扩展性瓶颈；RDF2vec 对文字及数值属性处理有限，难以完整表达 AAS 中的数值特征。

---

## 376. The Sample Complexity of Lossless Data Compression

**arXiv ID:** 2601.06688 | [PDF](https://arxiv.org/pdf/2601.06688v1)

**作者:** Terence Viaud `[一作]` (University of Cambridge), Ioannis Kontoyiannis `[通讯]` (University of Cambridge)

**通讯引用:** 2745 | [OpenAlex ID](https://openalex.org/A5044225058)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出了一种基于样本复杂度的非渐近框架，用以评估无损压缩的极限性能；

**💡 创新点**

创新点在于把压缩问题映射为假设检验问题，发现样本复杂度由源的Renyi熵(1/2)或Renyi散度决定，并给出可计算的上界与下界；

**🔧 技术方法**

主要技术包括Renyi散度与熵的性质、Le Cam引理、Chernoff信息、Markov链的Perron–Frobenius理论以及对偶最优压缩码的构造；

**📊 数据集**

该工作为理论研究，无需使用具体数据集，所有结果均为闭式或数值可计算的上界/下界；

**📈 对比分析**

与传统渐近式（如香农熵或方差展开）相比，该框架提供了在给定误差阈值下的确切块长界限，且常数显式，性能评估更具实际指导意义；

**⚠️ 局限性**

局限性包括：对Markov源的结果仍依赖初始分布和特征向量，常数可能不够紧；对更复杂源（如非平稳或高阶马尔可夫）需要进一步推广；

---

## 377. Will it Merge? On The Causes of Model Mergeability

**arXiv ID:** 2601.06672 | [PDF](https://arxiv.org/pdf/2601.06672v1)

**作者:** Adir Rahamim `[一作]` (Technion), Yonatan Belinkov `[通讯]` (Kempner Institute Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化了LoRA模型更新的可合并性（mergeability），并提出基于基模型性能的加权合并方法。

**💡 创新点**

首次将合并性能定义为可测量得分，揭示基模型对任务的先验知识是合并可行性的关键预测因子，并基于此提出改进的合并策略。

**🔧 技术方法**

使用LoRA适配器、Knots/TIES/平均合并算法、权重对齐和SVD对齐技术。

**📊 数据集**

PopQA实体问答数据集和Lots‑of‑LoRAs多任务LoRA集合，基础模型包括Llama‑3.2‑3B、Qwen‑2.5‑3B、Mistral‑7B‑Instruct‑v0.2。

**📈 对比分析**

通过计算合并后模型在各任务上的准确率与原始LoRA的差异来评估；加权合并相比普通平均，能更好保留低基准性能任务且不损失高性能任务。

**⚠️ 局限性**

研究仅聚焦LoRA更新，基于少量基模型；需在更多模型、算法和数据集上验证；权重特性与合并关系仍不完全清晰。

---

## 378. InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs

**arXiv ID:** 2601.06666 | [PDF](https://arxiv.org/pdf/2601.06666v1)

**作者:** Yuzhuo Bai `[一作]` (Tsinghua University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 36343 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了可解释且细粒度的事实核查框架InFi-Checker，结合控制生成管道产生含证据、错误类型、推理解释与纠正的训练集和基准，并训练模型实现支持证据、细粒度错误分类与解释的全流程推理；

**💡 创新点**

①提出了基于LLM的结构化链式思考数据合成管道，能够生成多类型细粒度幻觉；②构建了包含证据、错误类型、解释和纠正的多任务输出模型；③实现了细粒度错误检测与可解释性兼具的事实核查；

**🔧 技术方法**

使用LLM（如GPT‑4o）进行结构化链式思考生成幻觉与解释，结合人工验证；模型基于Llama‑3.1‑8B‑Instruct或Qwen3‑8B的指令微调，输出结构化JSON；采用错误类型化策略、句子级对齐比率等技术；

**📊 数据集**

通过BBC News和DetNet Wikipedia文档生成的自建训练集15,660条以及手工验证的519条测试基准InFi‑Bench；同时兼容FRANK等公开基准；

**📈 对比分析**

在细粒度和二元事实核查基准上与多种开源/闭源LLM及专业模型对比，InFi‑Checker在细粒度BAcc上达90%+，在FRANK、六大二元基准上实现最高或同等水平，显著优于对手；

**⚠️ 局限性**

受LLM证据提取与幻觉生成质量限制，仍难处理跨文档或复杂上下文幻觉；区分事实缺失与合理简化困难；需人工筛选以保证数据质量。

---

## 379. Mapping and Comparing Climate Equity Policy Practices Using RAG LLM-Based Semantic Analysis and Recommendation Systems

**arXiv ID:** 2601.06703 | [PDF](https://arxiv.org/pdf/2601.06703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 380. Physics-constrained Gaussian Processes for Predicting Shockwave Hugoniot Curves

**arXiv ID:** 2601.06655 | [PDF](https://arxiv.org/pdf/2601.06655v1)

**作者:** George D. Pasparakis `[一作]` (Johns Hopkins University), Michael D. Shields `[通讯]` (Johns Hopkins University)

**通讯引用:** 9559 | [OpenAlex ID](https://openalex.org/A5041446747)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种物理约束高斯过程回归框架，用极少量的分子动力学（MD）冲击模拟数据来预测硅碳化物在 Hugoniot 曲线上的压强、密度、温度等状态，并提供不确定度估计。

**💡 创新点**

创新点在于将 Rankine–Hugoniot 跳跃条件直接嵌入协方差函数，通过泰勒展开得到热力学一致的协方差结构，能够同时处理弹性、塑性和相变多波结构，并通过可解释的超参数实现数据高效学习。

**🔧 技术方法**

采用的技术包括：物理约束高斯过程回归（GP）、Intrinsic Coregionalization Model、泰勒展开与 Delta 方法构建协方差、HDBSCAN 聚类识别波前、LAMMPS 逆弹道 MD 仿真以及主动学习与不确定度量化。

**📊 数据集**

使用的数据集为 21 条 3C‑SiC 单晶[001]方向的 MD 冲击模拟（冲击速度 0.25–6.0 km/s），每条数据提供冲击波速、粒子速度、压强、密度、温度等信息。

**📈 对比分析**

通过与原始 MD 结果和文献实验数据比较，GP 预测在弹性、塑性和相变三个区间均能保持与 MD 的相符性，95% 置信区间覆盖实际值，表明模型在小样本下实现了可靠且符合热力学约束的预测。

**⚠️ 局限性**

局限性包括：需依赖高质量 MD 数据；对 T–E 的线性假设可能不适用于所有材料；在相变或过驱动区的尖锐非平稳特征对 GP 产生较大不确定性；数值稳定性需通过缩放和坐标变换保证；并且模型尚未在多晶或不同材料体系中验证。

---

## 381. Labels have Human Values: Value Calibration of Subjective Tasks

**arXiv ID:** 2601.06631 | [PDF](https://arxiv.org/pdf/2601.06631v1)

**作者:** Mohammed Fayiz Parappan `[一作]` (Duke University), Ricardo Henao `[通讯]` (Duke University)

**通讯引用:** 8636 | [OpenAlex ID](https://openalex.org/A5056639842)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MultiCalibrated Subjective Task Learner（MC‑STL）框架，利用价值聚类和群体特定嵌入实现对主观 NLP 任务的多校准预测。

**💡 创新点**

创新点在于①将主观标签的价值差异明确聚类为可解释的价值群体；②在模型中加入可学习的价值嵌入并对每个群体进行多校准；③在二分类、序数分类和偏好学习三种任务上统一实现并评估。

**🔧 技术方法**

采用 Transformer 文本编码 + 价值嵌入 + 加性融合 + 任务特定输出头（Sigmoid 或 CORAL），并结合交叉熵+KL 散度+L2 正则的联合损失；在偏好学习中使用 Bradley‑Terry 损失和温度参数。

**📊 数据集**

使用 ValuePrism（及其 VP‑Value、VP‑Rights、VP‑Duties 子集）、ValuePrism + Schwartz 价值标签、Anthropic HH‑RLHF+ValueImprint、DICES‑990（毒性 + 人口统计）以及 D3CODE（冒犯性）等多种数据集。

**📈 对比分析**

与忽略价值和多数投票两种基线对比，MC‑STL 在 AUC/准确率、校准斜率/偏差、1‑EMD 等指标上均表现更好，整体与群体 AUC 提升，校准接近理想（斜率≈1、偏差≈0），并提升了少数意见预测能力。

**⚠️ 局限性**

局限性包括仅考虑三种聚类方式，未探究其交叉提取；未评估更细粒度的公平度量；对注释理由或人类属性依赖较大，缺失时受限；未验证在文本生成等其他主观任务中的泛化；未分析 LLM 生成理由对聚类效果的影响。

---

## 382. Evaluating Accounting Reasoning Capabilities of Large Language Models

**arXiv ID:** 2601.06707 | [PDF](https://arxiv.org/pdf/2601.06707v1)

**作者:** Jie Zhou `[一作]` (Jiangsu Ocean University), Zhe Li `[通讯]` (Jiangsu Ocean University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估大语言模型在会计推理任务中的能力并提出垂直领域会计推理评估框架

**💡 创新点**

定义垂直会计推理概念并构建包含会计原理、财务报表、成本会计与审计的专用基准

**🔧 技术方法**

使用链式思考、少量示例提示等 Prompt 技术，并对 GLM 系列与 GPT‑4 进行评测

**📊 数据集**

利用 GSM8K、MR‑GSM8K、CLUE、CPA 试题等中文会计相关数据集

**📈 对比分析**

采用 Few‑shot‑CoT 与自动评估器对模型做对比，GPT‑4 得分最高但仍低于专业会计标准，GLM‑4 约 22%，GLM‑130B 约 60%，GLM‑6B 约 20%

**⚠️ 局限性**

受限于模型规模有限、仅覆盖少数 LLM、评测对提示敏感、基准未覆盖全部实际会计场景

---

## 383. SafePro: Evaluating the Safety of Professional-Level AI Agents

**arXiv ID:** 2601.06663 | [PDF](https://arxiv.org/pdf/2601.06663v1)

**作者:** Kaiwen Zhou `[一作]` (University of California Santa Cruz), Xin Eric Wang `[通讯]` (University of California Santa Barbara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SafePro 基准，用于评估专业 AI 代理在高复杂多步任务中的安全性，并在此基础上构建了 275 条危险任务数据集。

**💡 创新点**

创新点包括：① 专注于专业领域复杂任务，填补了安全评估空白；② 采用迭代创建与审核的流程生成高质量危险任务；③ 使用 LLM‑as‑judge 进行自动安全评判，并探索多种安全缓解策略。

**🔧 技术方法**

技术实现依赖 CodeAct 代理框架、GPT‑5、Claude‑Haiku、Gemini 等多模态大模型，结合安全提示、LLM 评分类、safeguard 模型等手段进行安全评估与缓解。

**📊 数据集**

数据集为 SafePro，包含 275 条任务，覆盖 9 经济部门与 51 个职业，按 9 种安全风险类别（财产损失、歧视/偏见、错误信息、信息泄露等）划分。

**📈 对比分析**

通过 Unsafe Rate 指标对 8 种大模型进行对比，平均 unsafe rate 超 50%；Claude‑Haiku 4.5 取得最低 unsafe rate，LLM‑judge 评估与安全分类准确率对比显示模型在指令跟随时安全性不足，缓解策略（安全提示、LLM 评分类、safeguard 模型）在一定程度上降低了 unsafe rate，但整体安全水平仍不理想。

**⚠️ 局限性**

limitations：① 数据集基于美国 GDPval，缺乏多地区、多职业和多轮交互场景；② 当前不涉及视频/音频模态任务；③ safeguard 模型识别率低，需进一步提升领域适配与安全判断能力。

---

## 384. eSkiTB: A Synthetic Event-based Dataset for Tracking Skiers

**arXiv ID:** 2601.06647 | [PDF](https://arxiv.org/pdf/2601.06647v1)

**作者:** Krishna Vinod `[一作]` (Arizona State University), Bharatesh Chakravarthi `[通讯]` (Arizona State University)

**通讯引用:** 161 | [OpenAlex ID](https://openalex.org/A5083090349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过将传统RGB广播视频直接转换为事件流，构建了第一个专门针对滑雪运动的合成事件数据集eSkiTB，并在此基础上进行系统评测；

**💡 创新点**

创新点在于严格实现iso‑informational转换（不使用神经网络插帧），保证RGB与事件流信息量相同，并提供高分辨率、密集时间标注的标注；

**🔧 技术方法**

技术主要包括v2e事件仿真、基于Transformer的RGB跟踪器STARK、基于SNN的事件跟踪器SDTrack以及密集时间插值的IoU评估；

**📊 数据集**

使用的数据集为从SkiTB裁剪的300条高分辨率比赛视频，生成1280×720分辨率事件流并配备1ms间隔的密集框标注；

**📈 对比分析**

在统一的one‑pass评估框架下，经过域适配微调的SDTrack在eSkiTB测试集上取得0.711的平均IoU，比RGB基线提升约+0.2 IoU，尤其在高杂乱背景下提升约+20.0 IoU；

**⚠️ 局限性**

局限包括源RGB帧率（25–60Hz）限制事件时间分辨率，导致缺失高频运动细节；固定尺寸的体素网格在自由式滑雪等大尺度动作时引入混叠；并且仅使用合成事件，真实传感器的噪声与动态范围尚待验证。

---

## 385. Agentic AI Empowered Intent-Based Networking for 6G

**arXiv ID:** 2601.06640 | [PDF](https://arxiv.org/pdf/2601.06640v1)

**作者:** Genze Jiang `[一作]` (Brunel University London), Yizhou Huang `[通讯]` (Brunel University London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出一种层级多代理架构，将自然语言网络意图自动拆解并转化为可执行的6G网络切片配置

**💡 创新点**

通过LLM驱动的代理协同与ReAct循环，实现了语义理解、约束满足与跨域决策的自适应组合

**🔧 技术方法**

利用大语言模型（Llama‑3.3‑70B）、ReAct推理框架、专门的RAN和核心网络专家代理及系统提示工程

**📊 数据集**

在仿真生成的多场景6G基准集上进行评估，包括URLLC、eMBB、mMTC三类意图场景

**📈 对比分析**

与单体代理、规则引擎及直接LLM提示基线对比，平均语义准确率0.67、工程效用0.75，单体代理和规则引擎在语义准确率上分别低8%与54%，但在推理成本上更低；多代理系统在三轮ReAct循环中保持高可解释性和约束满足

**⚠️ 局限性**

主要局限包括较高的token消耗与推理时延、对提示精细化的高度依赖、跨域协同仍需进一步优化，且在真实网络环境中尚未验证

---

## 386. Efficient Aspect Term Extraction using Spiking Neural Network

**arXiv ID:** 2601.06637 | [PDF](https://arxiv.org/pdf/2601.06637v1)

**作者:** Abhishek Kumar Mishra `[一作]` (Drexel University), Nagarajan Kandasamy `[通讯]` (Drexel University)

**通讯引用:** 2779 | [OpenAlex ID](https://openalex.org/A5007957626)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于脉冲神经网络的Aspect Term Extraction模型SpikeATE。

**💡 创新点**

创新点在于引入三值脉冲神经元、卷积式脉冲编码层和专门针对SNN的伪梯度训练方法，实现了与DNN相当的精度且能耗显著降低。

**🔧 技术方法**

技术包括脉冲神经网络、Leaky Integrate-and-Fire神经元、三值脉冲神经元、卷积脉冲编码、时空反向传播与伪梯度。

**📊 数据集**

使用了SemEval 2014-2016餐厅与笔记本领域四个基准数据集（Lap14、Res14、Res15、Res16）。

**📈 对比分析**

通过与多种SOTA DNN（BERT、Self‑Training等）在F1得分上对比，SpikeATE在三值版本上可与之持平或略优，且能耗比传统模型低约41倍。

**⚠️ 局限性**

局限在于对长短语依赖的识别仍不够精准，尤其是四词以上短语；并且需要在更大规模数据上验证。

---

## 387. MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis

**arXiv ID:** 2601.06636 | [PDF](https://arxiv.org/pdf/2601.06636v1)

**作者:** Wenting Chen `[一作]` (Stanford University), Wenxuan Wang `[通讯]` (Renmin University of China)

**通讯引用:** 1806 | [OpenAlex ID](https://openalex.org/A5100755181)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了MedEinst基准，对LLM在临床诊断中的Einstellung效应进行评估，并提出ECR-Agent模型进行缓解。

**💡 创新点**

创新点在于使用对抗式的因果对照案例来量化偏见陷阱率，并结合基于证据的因果推理和经验积累的代理框架。

**🔧 技术方法**

采用双通道感知、三层因果图推理、证据审计和批评驱动的图谱与记忆演化等技术。

**📊 数据集**

数据集为从DDXPlus衍生的5,383对计数病例，覆盖49种疾病，共10,766个临床叙事。

**📈 对比分析**

对比17种LLM与代理，传统模型基线准确率>50%但陷阱率>50%；ECR-Agent基线69.49%、稳健准确率24.21%、陷阱率33.75%。

**⚠️ 局限性**

局限性在于仅覆盖49个常见疾病，缺乏罕见病和多合并症情景。

---

## 388. Lower Bounds for the Algorithmic Complexity of Learned Indexes

**arXiv ID:** 2601.06629 | [PDF](https://arxiv.org/pdf/2601.06629v1)

**作者:** Luis Alberto Croquevielle `[一作]` (Imperial College London), Thomas Heinis `[通讯]` (Imperial College London)

**通讯引用:** 2619 | [OpenAlex ID](https://openalex.org/A5041993379)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了一个通用框架，用逼近理论与概率工具推导学习索引的查询时间下界，并给出与空间开销 K 的关系；

**💡 创新点**

创新点在于将查询时间映射到累计分布函数（CDF）的逼近误差，并对分段常数/线性模型给出了Ω(1/K)的逼近下界；

**🔧 技术方法**

采用逼近理论（量化、Kolmogorov宽度）、概率集中（Dvoretzky–Kiefer–Wolfowitz 等）、以及线性/指数/二分搜索的复杂度分析；

**📊 数据集**

本研究为理论性工作，使用抽象的 i.i.d. CDF F 及查询分布 μ，未使用具体数据集；

**📈 对比分析**

与传统 B+树、PGM‑Index 等在理论上比较，证明若分段数 K=O(n^α,α<1)，学习索引最坏情况下仍需 Ω(log n) 复杂度，只有接近线性空间才能获得优势；

**⚠️ 局限性**

局限性在于仅给出最坏/平均下界，未考虑动态更新、非独立样本、实际实验验证，仅适用于分段多项式模型。

---

## 389. IDRBench: Interactive Deep Research Benchmark

**arXiv ID:** 2601.06676 | [PDF](https://arxiv.org/pdf/2601.06676v1)

**作者:** Yingchaojie Feng `[一作]` (National University of Singapore), Anthony K. H. Tung `[通讯]` (National University of Singapore)

**通讯引用:** 7983 | [OpenAlex ID](https://openalex.org/A5023925931)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了IDRBench基准，评估LLM在深度研究任务中的交互性能与成本；

**💡 创新点**

首次将多代理框架、基于参考的用户模拟器与交互感知评估相结合，系统化衡量交互收益与成本；

**🔧 技术方法**

采用多代理架构、LLM问答与工具调用、用户模拟器、交互决策模块、语义相似度、F1、LLM-ACS等技术；

**📊 数据集**

使用DeepResearch Bench的100个查询‑参考文档对，并通过压缩方法注入查询不完整化；

**📈 对比分析**

对七种LLM进行对比实验，交互显著提升报告质量，弱模型提升幅度大，强模型提升有限，但成本差异显著；

**⚠️ 局限性**

用户模拟过于理想化、模糊化仅覆盖查询不完整，缺乏对更复杂用户行为与多样性噪声的评估。

---

## 390. Cross-Modal Computational Model of Brain-Heart Interactions via HRV and EEG Feature

**arXiv ID:** 2601.06792 | [PDF](https://arxiv.org/pdf/2601.06792v1)

**作者:** Malavika Pradeep `[一作]` (Digital University Kerala), Elizabeth Sherly `[通讯]` (Digital University Kerala)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

通过跨模态回归将ECG的HRV特征映射到EEG认知特征，评估ECG是否可替代EEG监测心理负荷。

**💡 创新点**

提出使用PSV‑SDG合成HRV数据增强模型，并将XGBoost作为跨模态回归框架。

**🔧 技术方法**

使用HRV时域指标、Catch22特征、XGBoost回归/分类、SMOTE、PSV‑SDG合成数据。

**📊 数据集**

公开的OpenNeuro多模态数据集ds003838（EEG+ECG+PPG+瞳孔）。

**📈 对比分析**

在多分类任务中，ECG+合成HRV在XGBoost上达到0.97准确率；跨模态映射仅提升到约0.40；EEG特征接近1.0准确率。

**⚠️ 局限性**

跨模态准确率低，因EEG与HRV时间尺度不同，缺乏同步信息，模型泛化受限。

---

## 391. SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute

**arXiv ID:** 2601.06790 | [PDF](https://arxiv.org/pdf/2601.06790v1)

**作者:** Bowen Shen `[一作]` (Harbin Institute of Technology), Zoe L. Jiang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1730 | [OpenAlex ID](https://openalex.org/A5008190964)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于两方安全多方计算的隐私保护 Mixture‑of‑Experts（MoE）推理框架 SecMoE，并通过 Select‑Then‑Compute 机制实现了安全稀疏 MoE 与安全多项式选择。

**💡 创新点**

创新点在于使用同态加密的 one‑hot 向量在专家选择阶段实现对激活专家的隐匿选择，既保持 MoE 的稀疏性，又将模型规模提升至 63 倍，且仅导致 15.2 倍的运行时间增加。

**🔧 技术方法**

采用格基加密的加法同态方案、算术/布尔秘密共享、Beaver 三元组、同态矩阵乘法以及多项式近似的 Select‑Then‑Compute 机制。

**📊 数据集**

在 GLUE 任务集（CoLA、QNLI、RTE 等）上评估，并使用 MoE‑small 与 Switch‑Base 两个模型进行实验。

**📈 对比分析**

与 Iron、BumbleBee 等现有 2‑PC PPML 框架在 LAN/WAN 条件下对比，结果显示 SecMoE 在通信量上可降低 1.8–29.8 倍，在端到端推理时间上可加速 1.3–16.1 倍，尤其在 128 个专家时速度提升至 16.1 倍。

**⚠️ 局限性**

目前仅支持 K_exp=1 的稀疏 MoE，扩展到 128 以上专家会因模型参数和 Beaver 三元组导致显存不足；此外仍假设半诚实攻击者并需要离线预处理。

---

## 392. CyberLLM-FINDS 2025: Instruction-Tuned Fine-tuning of Domain-Specific LLMs with Retrieval-Augmented Generation and Graph Integration for MITRE Evaluation

**arXiv ID:** 2601.06779 | [PDF](https://arxiv.org/pdf/2601.06779v1)

**作者:** Vasanth Iyer `[一作]` (Grambling State University), S. S. Iyengar `[通讯]` (Florida International University)

**通讯引用:** 8434 | [OpenAlex ID](https://openalex.org/A5009505287)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过对Gemma-2B进行领域适配和指令调优，构建了基于MITRE ATT&CK的网络安全语言模型；

**💡 创新点**

创新点在于使用云端大模型生成高质量合成数据并结合图结构检索增强模型在安全任务上的推理能力；

**🔧 技术方法**

采用指令调优、Chain-of-Thought、RAG、图神经网络等技术；

**📊 数据集**

利用基于MITRE ATT&CK框架的标签对齐数据以及云端LLM生成的合成日志；

**📈 对比分析**

与通用LLM以及纯RAG对比，GraphRAG+GNN在准确率和特异性上最高，整体得分为8.00；

**⚠️ 局限性**

局限在于小模型的token窗口受限、对多步推理支持不足，且对真实日志的评估仍待进一步验证。

---

## 393. Comparative Separation: Evaluating Separation on Comparative Judgment Test Data

**arXiv ID:** 2601.06761 | [PDF](https://arxiv.org/pdf/2601.06761v1)

**作者:** Xiaoyin Xi `[一作]` (Rochester Institute of Technology), Zhe Yu `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 826 | [OpenAlex ID](https://openalex.org/A5014850850)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于比较判断测试数据的公平性评估方法——比较分离（comparative separation）

**💡 创新点**

创新点在于将分离准则改写为仅依赖配对比较的形式，并在二分类情形下理论证明其与传统分离等价；同时提出了对应的统计检验与功效分析框架

**🔧 技术方法**

采用统计假设检验（z检验）、功效分析、配对比较数据采集，并利用模拟与真实数据验证方法的有效性

**📊 数据集**

实验数据集包括Compas、German Credit（二分类）以及Jira Story Point估计（回归）等真实数据，亦使用人工生成的仿真数据进行验证

**📈 对比分析**

在二分类任务中，比较分离与传统分离在检测偏差上效果相同；在回归任务中仍能应用但与传统分离的等价性尚未证明；相较于传统方法，需要约两倍的配对样本以获得相同统计功效

**⚠️ 局限性**

局限性在于仅在二分类场景下理论等价，回归或连续敏感属性的情况缺乏理论支持；此外，配对采样的效率与人类标注者的主观偏差仍需进一步研究

---

## 394. Towards Computational Chinese Paleography

**arXiv ID:** 2601.06753 | [PDF](https://arxiv.org/pdf/2601.06753v1)

**作者:** Yiran Rex Ma `[一作]` (Beijing University of Posts and Telecommunications), Yiran Rex Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1687 | [OpenAlex ID](https://openalex.org/A5100443132)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统综述并绘制了计算中国古文字学的技术路线，从图像预处理、字符识别、碎片重组、年代鉴定、语言建模、知识图谱到自动解码与人机协同的完整方法论管线。

**💡 创新点**

创新点在于：①提出问题导向的“从视觉到语义再到推理”方法论框架；②系统梳理了现有的数据集、模型与技术，并标注了关键挑战；③从技术角度阐明了从经典计算机视觉到现代深度学习（CNN、Transformer、扩散模型、LLM）与知识图谱融合的演进；④对未来研究方向做了前瞻性规划（多模态统一、少样本推理、知识发现、协同系统）。

**🔧 技术方法**

主要使用技术包括：卷积网络、Transformer、注意力与窗口机制、图像分离与去噪网络（STSN、OBIFormer）、扩散生成模型（Diff‑Oracle、OBSD）、多模态大模型（LLM、LVLM、MLLM）、知识图谱与图神经网络、检索与跨字体匹配、生成式推理框架（OracleFusion、P^3、CoLa）以及人机协同平台（RejoinX、GenOV）。

**📊 数据集**

核心数据集包括：Oracle Bone Inscriptions（HUST‑OBS、HWOBC、Oracle‑50K、Oracle‑P15K、OBIMD等）以及扫描版（OBC306、Oracle‑241、Oracle‑MNIST、OBI125）；其他古文字集（CHUBS、Qin Slips、Bronze Insc.、EVOBC、ACCID、OracleRC、PicOBI‑20k、PD‑OBS、OracleSem）；多模态/任务导向集如OBIMD、HUSAM‑SinoCDCS、Oracle‑241 等。

**📈 对比分析**

对比结果表明：目前大多数模型在识别、碎片重组、年代分类等任务上仅略优于手工方法，整体性能仍低于训练不足的人类专家；在解码与推理任务上，Benchmark 如 OBI‑Bench 与 V‑Oracle 证明现有技术在精确度、推理链条、可解释性方面均不够成熟，尚未达到可靠的学术水平。

**⚠️ 局限性**

主要局限：①数据稀缺、长尾分布严重，导致模型泛化差；②图像噪声与物理破损难以完全补偿；③缺乏语音、语义与上下文的多模态统一；④模型缺少专家评测与可解释的推理流程；⑤跨体裁、跨年代、跨介质的迁移学习效果有限；⑥整体研究仍以“形式”导向，忽视了“意义”和“音韵”等人文维度。

---

## 395. DS-CIM: Digital Stochastic Computing-In-Memory Featuring Accurate OR-Accumulation via Sample Region Remapping for Edge AI Models

**arXiv ID:** 2601.06724 | [PDF](https://arxiv.org/pdf/2601.06724v1)

**作者:** Kunming Shao `[一作]` (Hong Kong University of Science and Technology), Chi-Ying Tsui `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8826 | [OpenAlex ID](https://openalex.org/A5072596277)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种数字随机计算内存（DS-CIM）架构，能够在保持高精度的同时实现高吞吐量的矩阵-向量乘法。

**💡 创新点**

创新点包括：
1) 将有符号乘加映射为无符号OR‑MAC电路，显著降低硬件复杂度；
2) 采用共享伪随机数生成器与二维采样重映射，消除OR累加中的1s饱和误差；
3) 在每列复制64个低成本OR‑MAC，实现在保持面积基本不变的前提下提高32×计算密度；
4) 引入Latch‑cached累加器降低累加器能耗。

**🔧 技术方法**

技术手段：无偏OR‑MAC电路、共享PRNG与SNG、数据重映射、Latch‑cached累加器、INT8/FP8量化、深度学习模型推理与评估。

**📊 数据集**

数据集：CIFAR‑10（ResNet18/ResNet50）、ImageNet（ResNet50）、FP8 LLaMA‑7B在多语言文本数据集（BoolQ、PIQA、HellaSwag、WinoGrande、ARC‑e、ARC‑c）。

**📈 对比分析**

比较方法：与现有数字、模拟、近似CIM系统在同等技术节点下进行对比；DS‑CIM1在bitstream长度256下，ResNet18在CIFAR‑10上准确率94.45%、RMSE0.74%，能效669.7 TOPS/W；DS‑CIM2在bitstream长度64下，ResNet18准确率94.31%、RMSE3.81%，能效3566.1 TOPS/W、面积效率363.7 TOPS/mm²，均显著优于SOTA。

**⚠️ 局限性**

局限性：
1) 对有符号运算的能耗仍较高；
2) 误差随位流长度缩短而显著增加，需要权衡精度与效率；
3) 共享PRNG与重映射方案对不同硬件或更大模型的迁移可能需要进一步调优；
4) 对极高稀疏度数据的处理仍有提升空间。

---

## 396. Thinking with Deltas: Incentivizing Reinforcement Learning via Differential Visual Reasoning Policy

**arXiv ID:** 2601.06801 | [PDF](https://arxiv.org/pdf/2601.06801v1)

**作者:** Shujian Gao `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 23651 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出 Differential Visual Reasoning Policy（DVRP）框架，通过视觉三元组自监督驱动多模态 RLVR，消除感知-推理解耦。

**💡 创新点**

创新点是将可感知-可鲁棒的视觉差分监督引入奖励优化，既最大化与掩码输入的差异，又最小化与噪声输入的差异，从而迫使模型真正依赖视觉信息。

**🔧 技术方法**

使用了 RLVR 的 GRPO/DAPO 基础算法，加入 KL 视觉差分约束和熵正则，结合视觉三元组（原始、掩码、噪声）。

**📊 数据集**

实验数据集覆盖数学推理（Geo3k、Vista、WeMath、MVerse 等）和医学诊断（Slake、PathVQA、VQA-RAD、PMC-VQA）等。

**📈 对比分析**

与 GRPO、DAPO、PAPO 等基线以及多种商业与开源模型比较，DVRP 在 3B、7B 参数规模下分别提升 67.6% 与 40.5% 的相对准确率，且在医学任务上超过 GPT‑4o、CAPO‑7B 等。

**⚠️ 局限性**

局限性在于仅在 3B/7B 模型验证，未测试更大规模或不同体系结构；未与多代理/工具使用框架结合；对超参数与不同任务的敏感度仍需深入探究。

---

## 397. MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues

**arXiv ID:** 2601.06757 | [PDF](https://arxiv.org/pdf/2601.06757v1)

**作者:** Zheyuan Liu `[一作]` (University of Notre Dame), Meng Jiang `[通讯]` (University of Notre Dame)

**通讯引用:** 5740 | [OpenAlex ID](https://openalex.org/A5074821819)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个多轮多模态情境安全基准（MTMCS-Bench），用于评估大语言模型在图像-文本对话中识别、处理安全风险与保持有用性的能力。

**💡 创新点**

创新点在于通过与安全对话配对的同一场景对话、两种风险场景（升级型与场景切换型）以及多模态与单模态双版本的设计，系统化考察模型在对话进展中的上下文安全感知和行为选择。

**🔧 技术方法**

使用的技术包括多代理生成框架（分类器、写手、转换器、变体生成）、多模态提示与对话生成、基于MCQ/TF的意图识别评估以及LLM（GPT‑5‑mini）判定安全意识与帮助度的开放式生成评分。

**📊 数据集**

数据集由752个基准图像及其256个变体构成，生成12,032条三轮对话，共18,048问答对，包含安全与非安全两种对话，并提供多模态与文本对照版本。

**📈 对比分析**

在15个现有LLM（包括7个专有模型和8个开源模型）上评测，结果显示模型在意图识别上表现相对较好，但在安全意识与帮助度之间存在显著权衡；例如，Qwen3‑VL‑32B在安全意识方面提升有限，而GPT‑5.2在安全与帮助度上表现最佳；现有防护方法（如DPP、AdaShield）虽能提升安全意识，但往往牺牲了意图识别或帮助度。

**⚠️ 局限性**

局限性包括：1) 仅覆盖日常COCO样式图像，缺乏高风险领域；2) 评测多为诊断性，尚未提出统一有效的解决方案；3) 依赖LLM判定安全/帮助度，可能存在偏差和噪声。

---

## 398. On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning

**arXiv ID:** 2601.06748 | [PDF](https://arxiv.org/pdf/2601.06748v1)

**作者:** Changyu Liu `[一作]` (University of Missouri), Cheng Han `[通讯]` (University of Missouri)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TT-VLA框架，实现机器人在部署时通过测试时强化学习自适应调整视觉语言动作模型的策略；

**💡 创新点**

创新点在于设计了基于任务进度的稠密奖励、无价值函数的PPO变体以及每8步更新的策略调优策略，使得模型能在单一任务轨迹上快速、稳定地适应环境变化；

**🔧 技术方法**

采用了视觉语言动作模型（如Nora、OpenVLA、OpenVLA-RL、TraceVLA）、强化学习的PPO、进度估计器VLAC、LoRA微调等技术；

**📊 数据集**

使用ManiSkill 3仿真环境下的Pick‑and‑Place任务、Franka Research 3真实机器人环境以及多种视觉、执行与语义变换的评测数据集；

**📈 对比分析**

与四大开源VLA基线对比，TT-VLA在未见过的执行、视觉和语义任务上平均提升约10‑20%，在某些任务上提升超过40%，证明了显著的泛化与性能提升；

**⚠️ 局限性**

局限性包括依赖可靠的进度估计，视觉遮挡或非单调进度任务下奖励噪声可能削弱效果；当基准策略本身弱时，TT-VLA提升有限。

---

## 399. Study of Adaptive Reliability-Driven Conditional Innovation Decoding for LDPC Codes

**arXiv ID:** 2601.06732 | [PDF](https://arxiv.org/pdf/2601.06732v1)

**作者:** Hassan Touati `[一作]` (Pontifical Catholic University of Rio de Janeiro), Rodrigo C. de Lamare `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**通讯引用:** 10931 | [OpenAlex ID](https://openalex.org/A5049028312)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种自适应可靠性驱动的条件创新（AR-CID）解码算法，用于低密度奇偶校验（LDPC）码的快速高效迭代解码。

**💡 创新点**

创新点包括：①两阶段评估（消息质量检查 + 细化消息传递）以动态选择更新节点；②融合可靠性指标和上下文信息转换的加权组合；③在残差贝尔维斯传递（RBP）中加入预先评估的可靠性信息以加速收敛；④通过仅更新最不可靠的λN节点显著降低迭代次数。

**🔧 技术方法**

采用了残差贝尔维斯传递、可靠性驱动子集选择、上下文信息转换、加权组合、BPSK AWGN信道模型以及软判决LLR计算。

**📊 数据集**

使用了仿真生成的LDPC码（(512,256) 和 (2048,1024)，正则度分布 d_v=3, d_c=6），在0–4.5 dB的AWGN信道下进行评估。

**📈 对比分析**

与BP、URW、LBP、RBP、CI、RD‑RBP、RP、List‑RBP等八种主流算法在BER‑SNR、迭代次数和解码时延上进行对比。AR‑CID在同等BER目标下仅需4–5次迭代，SNR上平均提升0.3–0.5 dB，解码时延显著降低，整体性能优于其他方法。

**⚠️ 局限性**

局限性：每次迭代的计算量和内存开销略高；对参数（γ、λ、α、β）的调优有一定依赖；在极短码（N<128）或非AWGN信道（衰落、多天线、毫米波等）下的表现尚未验证；对极低延迟实时系统的硬件实现需要进一步优化。

---

## 400. Why are there many equally good models? An Anatomy of the Rashomon Effect

**arXiv ID:** 2601.06730 | [PDF](https://arxiv.org/pdf/2601.06730v1)

**作者:** Harsh Parikh `[一作]` (Yale University), Harsh Parikh `[通讯]` (Yale University)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5024676571)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了Rashomon效应的成因，并将其分为统计、结构和程序三类。

**💡 创新点**

提出了统一的框架，将Rashomon效应的三类来源系统化，并阐明它们对模型选择、解释性和公平性的影响。

**🔧 技术方法**

利用理论分析、已有度量（Rashomon集合、Rashomon比率、误差范围等）和文献中的案例来解释和量化模型多样性。

**📊 数据集**

引用了多个公开数据集作为案例，例如COMPAS、FICO、MNIST 等，但并未自行收集或实验数据。

**📈 对比分析**

通过比较不同模型在相同数据集上的误差或准确率差距（例如在 1% 以内）以及 Rashomon 集的大小来评估多样性；结果表明即使性能相近，模型在变量重要性、解释方式和公平性等方面也存在显著差异。

**⚠️ 局限性**

缺乏对高维深度模型 RASHOMON 集的实际量化与实验验证，方法主要停留在理论和案例分析层面，对跨领域泛化和实用工具的开发仍有待深入。

---

## 401. Predicting Student Success with Heterogeneous Graph Deep Learning and Machine Learning Models

**arXiv ID:** 2601.06729 | [PDF](https://arxiv.org/pdf/2601.06729v1)

**作者:** Anca Muresan `[一作]` (Florida Atlantic University), Ionut Cardei `[通讯]` (Florida Atlantic University)

**通讯引用:** 1040 | [OpenAlex ID](https://openalex.org/A5108528882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种融合异构图深度学习与传统机器学习的框架，用以在学期早期和持续期间预测学生学习成功率。

**💡 创新点**

创新点在于：①引入动态评估特征（分阶段部分成绩）并与异构图结构结合；②设计基于注册–学生–注册（RSR）元路径的异构图；③通过特征消融实验验证动态特征与学生背景信息对模型效果的影响，展示即使仅用极少特征亦能获得高准确率。

**🔧 技术方法**

主要技术包括：Heterogeneous Graph Attention Network (HAN)、Heterogeneous Graph Transformer (HGT)、逻辑回归、随机森林等传统机器学习方法；使用PCA、交叉验证、早停、Adam优化器以及网格搜索调参。

**📊 数据集**

数据集为Open University Learning Analytics (OULA)，包含7门课程、4个学期、约32,000名学生的注册、成绩与个人信息。

**📈 对比分析**

与传统机器学习基线（LR、RF等）比较，HGT/ HAN在学期早期（≤20天）F1得分领先4.7%，后期差距缩小至0.4%；在计算时间上，图模型显著高于传统模型，但在准确性上更具优势。

**⚠️ 局限性**

局限性：图模型在稀疏图结构下仍受节点度低影响；特征工程依赖人工设计的部分成绩与权重；对隐私敏感的学生人口统计特征的使用有限；以及模型训练耗时较长。

---

## 402. FinForge: Semi-Synthetic Financial Benchmark Generation

**arXiv ID:** 2601.06747 | [PDF](https://arxiv.org/pdf/2601.06747v1)

**作者:** Glenn Matlin `[一作]` (Georgia Institute of Technology), Sudheer Chava `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6888 | [OpenAlex ID](https://openalex.org/A5112933779)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于专家指导和LM驱动的半合成流程 FinForge，用以生成金融领域的高质量评测基准，并发布了包含 5000 个问答对的 FinForge‑5k 数据集。

**💡 创新点**

创新点在于将手工过滤的金融文献与多阶段 LM 生成、规划与判定相结合，实现动态、无污染、可扩展的金融专业评测数据生成。

**🔧 技术方法**

采用混合人工–程序化的语料筛选、基于 LM 的文档分析、答案计划、问题生成以及 LM‑judge 验证等技术，并使用 GPT‑4o/Claude‑3.5 等大型模型做判定。

**📊 数据集**

使用来自 100,000 份权威金融文档（约 143M 词）的语料库，涵盖 11 个子领域，并从中采样 10,000 篇生成 5,000 个 QA 对。

**📈 对比分析**

对比多款开源与闭源模型，使用统一的多选测评，发现闭源模型在 FinForge‑5k 上达 73–80% 的准确率，开放源模型在 32–110B 参数规模下表现与之相近，展示规模与金融推理能力并非完全正相关。

**⚠️ 局限性**

主要限制在于生成与评估均依赖 Gemini 2.5 Flash，导致问题缺乏必要上下文、评估主观性高、缺乏透明度，并且存在潜在的数据污染风险。

---

## 403. CIRAG: Construction-Integration Retrieval and Adaptive Generation for Multi-hop Question Answering

**arXiv ID:** 2601.06799 | [PDF](https://arxiv.org/pdf/2601.06799v1)

**作者:** Zili Wei `[一作]` (Northeastern University), Yifei Zhang `[通讯]` (Northeastern University)

**通讯引用:** 2976 | [OpenAlex ID](https://openalex.org/A5100386920)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CIRAG框架，结合迭代构建-集成检索和自适应多粒度生成，解决多跳问答中单一路径扩展与检索粒度不匹配的问题。

**💡 创新点**

创新点包括：①引入迭代构建-集成模块消除贪心单路径扩展；②使用级联多粒度生成动态匹配上下文需求；③通过轨迹蒸馏将大模型的集成策略迁移至轻量模型。

**🔧 技术方法**

采用结构化三元组检索、LLM提示式开放式IE、历史条件集成、级联上下文层级以及轨迹蒸馏+LoRA微调等技术。

**📊 数据集**

在HotpotQA、2WikiMultiHopQA、MuSiQue、多跳问答基准以及WebQA、NQ等单跳数据集上进行实验。

**📈 对比分析**

与NativeRAG、IRCoT、FLARE、MetaRAG、DualRAG、KiRAG等多种迭代RAG基线进行对比，CIRAG在Qwen2.5-7B等模型上平均提升F1/EM约10%+，在所有基准中均获得最高或次高成绩。

**⚠️ 局限性**

局限性在于：①依赖提示式开放式IE，可能无法覆盖隐式或专业领域关系；②级联生成导致额外延迟，缺乏轻量级粒度路由器。

---

## 404. Multi-Stage Evolutionary Model Merging with Meta Data Driven Curriculum Learning for Sentiment-Specialized Large Language Modeling

**arXiv ID:** 2601.06780 | [PDF](https://arxiv.org/pdf/2601.06780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 405. GDEPO: Group Dual-dynamic and Equal-right-advantage Policy Optimization with Enhanced Training Data Utilization for Sample-Constrained Reinforcement Learning

**arXiv ID:** 2601.06795 | [PDF](https://arxiv.org/pdf/2601.06795v1)

**作者:** Zhengqing Yan `[一作]` (State Key Laboratory of Engines Tianjin University), Kang Song `[通讯]` (State Key Laboratory of Engines Tianjin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对有限样本自动定理证明的强化学习框架GDEPO，改进了采样、优势计算和迭代策略

**💡 创新点**

创新点包括动态补充采样、等价优势（将正确性作为优势符号、辅助奖励影响幅度）以及根据样本难度动态增加反向传播次数

**🔧 技术方法**

基于GRPO的策略梯度方法，结合了动态采样、等价优势计算与自适应迭代，使用Lean4验证器和自定义奖励函数

**📊 数据集**

在Lean4形式化的数学数据集上训练和评估，采用MiniF2F-test、MathOlympiadBench和PutnamBench三大测试集，同时使用FineLeanCorpus中的高难度样本构建训练集

**📈 对比分析**

与DeepSeek-Prover-V2、Kimina-Prover和Goedel-Prover-V2等SOTA模型对比，GDEPO在三大基准上均提升了成功率，尤其在PutnamBench上提升了约84%，且在高难度样本上表现更为显著

**⚠️ 局限性**

局限性主要体现在：依赖离线批量采样且需多次重采样导致计算成本上升；等价优势需要手动设定阈值和权重；对极端难题仍可能无效，且实验仅在特定Lean4环境下验证

---

## 406. Mobility Inequity and Risk Response After Hurricane Helene: Evidence from Real-Time Travel and Social Sentiment Data

**arXiv ID:** 2601.06722 | [PDF](https://arxiv.org/pdf/2601.06722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 407. Privacy-Preserving Data Processing in Cloud : From Homomorphic Encryption to Federated Analytics

**arXiv ID:** 2601.06710 | [PDF](https://arxiv.org/pdf/2601.06710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 408. Behavioral Analytics for Continuous Insider Threat Detection in Zero-Trust Architectures

**arXiv ID:** 2601.06708 | [PDF](https://arxiv.org/pdf/2601.06708v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 409. Garbage Attention in Large Language Models: BOS Sink Heads and Sink-aware Pruning

**arXiv ID:** 2601.06787 | [PDF](https://arxiv.org/pdf/2601.06787v1)

**作者:** Jaewon Sok `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**通讯引用:** 2395 | [OpenAlex ID](https://openalex.org/A5065728469)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过引入sink分数衡量注意力头的冗余，提出基于sink的结构化剪枝方法。

**💡 创新点**

创新点在于将注意力sink现象与模型层级冗余关联，证明高sink分数头在深层是冗余的，并以此作为剪枝依据。

**🔧 技术方法**

技术包括计算注意力sink分数、头/层级聚合、基于sink分数的剪枝以及对GQA模型的头零化实现。

**📊 数据集**

使用的基准数据集包括WikiText-2、ARC-Easy/Challenge、BoolQ、OpenbookQA、PIQA、MMLU、Winogrande、Hellaswag等。

**📈 对比分析**

与Wanda‑SP、Mag‑SP、ShortGPT等传统剪枝方法对比，在Gemma‑3、Llama‑3.1、Qwen3等模型上，sink剪枝在保持80–90%精度的同时实现更高压缩比，性能优于或相当于现有方法。

**⚠️ 局限性**

局限性包括缺乏理论解释sink层级分布、仅在中小规模GQA模型验证、未评估对偏见和鲁棒性影响。

---

## 410. AutoTour: Automatic Photo Tour Guide with Smartphones and LLMs

**arXiv ID:** 2601.06781 | [PDF](https://arxiv.org/pdf/2601.06781v1)

**作者:** Huatao Xu `[一作]` (Hong Kong University of Science and Technology), Mo Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 14223 | [OpenAlex ID](https://openalex.org/A5100361458)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

AutoTour利用VLM与开放地理空间数据自动为用户拍摄的照片标注细粒度地标并生成文本/音频解说，提升探索体验；

**💡 创新点**

创新点在于采用无训练、基于几何重叠匹配的方式将视觉特征与OSM数据对齐，并通过多模态模型实现精确定位与描述；

**🔧 技术方法**

主要技术包括Claude Opus 4/ GPT‑4o等VLM用于特征检测与匹配，Qwen‑Max用于视觉定位，OpenStreetMap + Overpass API进行地理特征抽取，以及几何重叠匹配与文本生成；

**📊 数据集**

使用自采集的134张跨5座城市（北京、上海、深圳、香港、洛杉矶）的照片数据集，辅以公开的OSM全球地理数据库；

**📈 对比分析**

与多种LLM和基准算法比较，AutoTour在匹配召回率0.816、精确率0.923、F1值0.84、总平均得分3.58/4，且单张照片平均成本约0.03美元、延迟约20–35秒；

**⚠️ 局限性**

主要限制包括高延迟（需大型VLM推理）、依赖GPS精度、在高密度或室内环境下定位不稳以及模型在某些场景下仍可能产生幻觉或定位误差。

---

## 411. Robust Evacuation for Multi-Drone Failure in Drone Light Shows

**arXiv ID:** 2601.06728 | [PDF](https://arxiv.org/pdf/2601.06728v1)

**作者:** Minhyuk Park `[一作]` (Texas State University), Tsz-Chiu Au `[通讯]` (Texas State University)

**通讯引用:** 2144 | [OpenAlex ID](https://openalex.org/A5070876502)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种针对无人机灯光秀中多无人机失效时的停泊算法，用以快速撤离受危机影响的无人机并通过隐藏无人机实现演出恢复。

**💡 创新点**

创新点在于：①引入时空网格与占用概率模型评估坠落无人机对场景的威胁；②使用带注意力机制的 Social‑LSTM 对失效后无人机轨迹进行预测；③构建基于 RRT 的随机有向无环图，利用负对数占用概率权重进行最短路搜索，从而得到近似最优的无碰撞停泊路径；④在停泊后通过隐藏无人机补位实现演出连续性。

**🔧 技术方法**

主要技术包括：深度学习（Social‑LSTM+注意力）用于轨迹预测；空间占用概率与空间填充随机图；RRT 与 Bellman‑Ford/拓扑排序求最短路；基于占用概率的碰撞概率计算与规划评分；隐藏无人机恢复策略。

**📊 数据集**

数据集来自公司内部仿真器，结合真实无人机运动校准，随机触发三种失效模式（掉落、着陆、返回）生成数千条失效前后姿态序列，供 Social‑LSTM 训练使用。

**📈 对比分析**

实验对比基准策略（如仅按预设轨迹撤离或不考虑占用概率的停泊）显示，该方法在碰撞概率上显著下降、停泊成功率提高约 15‑20%，并在演出恢复阶段能够更快填补缺失无人机。

**⚠️ 局限性**

局限性包括：对无障碍场景的假设、失效模式未知时预测误差、占用概率估计依赖大量随机投射，计算开销相对较高，以及对大规模无人机群的扩展性尚未完全验证。

---

## 412. Artificial Entanglement in the Fine-Tuning of Large Language Models

**arXiv ID:** 2601.06788 | [PDF](https://arxiv.org/pdf/2601.06788v1)

**作者:** Min Chen `[一作]` (University of Pittsburgh), Junyu Liu `[通讯]` (University of Pittsburgh)

**通讯引用:** 4194 | [OpenAlex ID](https://openalex.org/A5100682743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过量子信息视角，将大型语言模型（LLM）的参数更新与矩阵乘积态（MPS）表征相结合，定义并测量了“人工纠缠”，以揭示不同参数高效微调（PEFT）方法（LoRA、FFT、MPS Adaptation）内部结构与外部注意力表示之间的关系。

**💡 创新点**

创新点在于：①提出“人工纠缠”概念并利用MPS的纠缠熵对LLM参数进行定量分析；②发现LoRA在查询/值投影矩阵中的纠缠呈体积律并出现“纠缠谷”，而注意力矩阵则表现为面积律加对数修正；③借助随机矩阵理论给出Attention Cardy公式，解释注意力矩阵的对数尺度；④揭示“无毛”现象——内部纠缠差异在注意力输出中被粗粒化，导致LoRA与FFT在任务性能上相近。

**🔧 技术方法**

技术手段包括：MPS分解、奇异值分解与von Neumann纠缠熵计算、随机矩阵理论分析、Attention Cardy公式推导、对比实验（LoRA、FFT、MPS Adaptation）。

**📊 数据集**

使用的实验数据集为Tulu3（指令跟随）和OpenThoughts3（推理任务），实验模型为LLaMA系列的1B与8B规模。

**📈 对比分析**

与全参数微调（FFT）比较，LoRA在参数量显著减少（仅几百或几千个可训练参数）时，测试损失与FFT相近，性能差异在同等学习率与规模因子下微小；MPS Adaptation表现出类似的纠缠特征但性能与LoRA基本持平。

**⚠️ 局限性**

局限性包括：仅在特定模型规模和数据集上验证，分析主要聚焦于投影矩阵与注意力矩阵，未深入探讨其他层结构；理论推导基于大规模极限与随机初始化，实际训练过程可能偏离；对齐“无毛”现象的泛化性尚未在更多任务与架构上检验。

---

## 413. Approximating Matroid Basis Testing for Partition Matroids using Budget-In-Expectation

**arXiv ID:** 2601.06723 | [PDF](https://arxiv.org/pdf/2601.06723v1)

**作者:** Lisa Hellerstein `[一作]` (New York University), Kevin Schewior `[通讯]` (University of Cologne)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5056422606)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了随机布尔函数评估问题，提出了一种自适应查询策略，以最小化期望查询次数，特别关注于分区矩阵的基础测试问题（MBT）。

**💡 创新点**

创新点在于提出了一种多项式时间的常数因子近似算法，解决了分区矩阵问题的MBT，结合了新技术与多种已建立的技术。

**🔧 技术方法**

使用了自适应查询策略和预算期望问题的技术，结合了随机化策略和确定性策略。

**📊 数据集**

使用了紧凑表示的矩阵和布尔随机变量的概率分布，具体数据集未明确提及。

**📈 对比分析**

与现有方法比较，提出的算法在期望查询次数上具有常数因子近似，性能优于之前的常数因子近似算法，尤其是在处理分区矩阵时。

**⚠️ 局限性**

限制在于算法的复杂性和对特定类型矩阵的依赖，尚未解决的开放问题包括在任意成本情况下的近似性能。

---

## 414. No More Stale Feedback: Co-Evolving Critics for Open-World Agent Learning

**arXiv ID:** 2601.06794 | [PDF](https://arxiv.org/pdf/2601.06794v1)

**作者:** Zhicong Li `[一作]` (Renmin University of China), Yong Liu `[通讯]` (Renmin University of China)

**通讯引用:** 19928 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 ECHO 框架，在大语言模型代理中同步更新策略与批判器，通过级联诊断‑纠正回合实现共进化，显著提升开放世界任务的长程成功率。

**💡 创新点**

创新点主要包括：① 将批判器视为随策略演进的共生模块，解决批判器过时问题；② 采用级联多视角诊断与条件改正的 rollout 生成分组轨迹；③ 引入饱和感知（SA）奖励函数，使批判器在“最后一公里”获得更具信息量的信号。

**🔧 技术方法**

技术手段包括：基于 GRPO 的双轨优势估计与组相对优化；饱和感知奖励（积分形式的非线性增益）；级联诊断‑改正回合的分组采样；使用 Qwen 系列模型做策略与批判器；外部奖励模型评估轨迹质量。

**📊 数据集**

数据集与环境：WebShop、ALFWorld、SciWorld、DeepSearch 四大开放世界任务环境；使用 Qwen3-4B、Qwen2.5-7B 等模型；评估时还对比 GPT‑4o、Gemini‑2.5‑pro、Claude‑Sonnet‑4.5、Qwen3‑235B‑A22B 等强基线。

**📈 对比分析**

与传统 GRPO、静态批判器以及多种主流大模型对比，ECHO 在四个基准上平均提升 7.28 分，Qwen3-4B 在 WebShop、ALFWorld、SciWorld、DeepSearch 的得分分别从 82.37/87.50/79.14/33.25 提升到 90.03/91.25/82.88/47.25；冻结批判器实验显示共进化显著提高训练稳定性与最终性能。

**⚠️ 局限性**

局限性：依赖外部奖励模型的质量与稳定性，若奖励噪声或偏差会导致批判器优化偏离真实诊断；目前策略与批判器使用独立模型，缺乏统一的评估‑批判体系，未来可进一步整合为单模型。

---

## 415. MemGovern: Enhancing Code Agents through Learning from Governed Human Experiences

**arXiv ID:** 2601.06789 | [PDF](https://arxiv.org/pdf/2601.06789v1)

**作者:** Qihao Wang `[一作]` (University of Chinese Academy of Sciences), Huacan Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5031229572)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建 MemGovern 框架，将 GitHub 原始 bug 修复记录治理成可供代码代理使用的经验卡片，并实现 agentic 双原语检索与浏览机制，提升自动软件工程代理的修复率。

**💡 创新点**

创新点包括：① 分层筛选、标准化与质量控制的经验治理流程；② 经验卡片分为索引层和解决层的结构化表示；③ 通过 agentic 双原语搜索（检索+浏览）实现多轮动态检索，超越传统单次 RAG。

**🔧 技术方法**

采用大型语言模型（GPT‑5.x、Claude‑4、GPT‑4o 等）进行经验抽取、标准化与质量评估；使用向量相似度检索进行索引层搜索；实现检索与浏览工具接口，实现 agentic 搜索流程。

**📊 数据集**

使用从 GitHub 收集的约 150K Issue–PR–Patch 组合，治理后得到 135K 经验卡；基准实验基于 SWE‑bench Verified 数据集。

**📈 对比分析**

在 SWE‑bench Verified 上与 SWE‑Agent 及 AutoCodeRover、CodeAct、SWESynInfer 等基线对比。平均提升 4.65% 解决率；在不同 LLM 后端（DeepSeek、Qwen3、GPT‑4o 等）均显著提升；相较于单次 RAG，Agentic Search 在各模型上表现更佳。

**⚠️ 局限性**

主要限制是检索过程中额外的 token 消耗；经验治理依赖 LLM 可能出现 hallucination，需要进一步压缩内存并提升治理效率。

---

## 416. The Normalized Difference Layer: A Differentiable Spectral Index Formulation for Deep Learning

**arXiv ID:** 2601.06777 | [PDF](https://arxiv.org/pdf/2601.06777v1)

**作者:** Ali Lotfi `[一作]` (Nutrien Centre for Sustainable and Digital Agriculture, University of Saskatchewan), Steve Shirtliffe `[通讯]` (Nutrien Centre for Sustainable and Digital Agriculture, University of Saskatchewan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出可微分的Normalized Difference Layer（ND层），让神经网络自行学习传统ND指数的波段系数，并将该层嵌入端到端的深度学习框架中，用于Sentinel‑2图像下的Kochia（侵入性杂草）与作物的像素级分类。

**💡 创新点**

创新点在于：①将经典ND公式转化为可学习的网络模块；②通过softplus对系数进行正向约束，保证分母始终正且输出保持在[-1,1]；③保留亮度不变性与范围限定的优势，同时允许梯度下降寻找任务特定的波段权重；④提供可解释的系数矩阵，揭示不同波段组合的重要性。

**🔧 技术方法**

技术方法包括：softplus参数化、对ND层的梯度推导与反向传播实现、Adam优化、交叉熵损失、对比实验的10折交叉验证、噪声鲁棒性评估（乘法噪声），以及对Attention‑gated ND变体的实现。

**📊 数据集**

使用的数据集为加拿大Saskatchewan地区2022‑2024年收集的Sentinel‑2 Level 2A多光谱图像（10波段，10/20 m分辨率），共计2318个标注样本（Kochia 1071、作物 1247），每个样本对应10个波段反射率。

**📈 对比分析**

评价方法：对ND、MLP（基线）和AttND三种模型分别在深度2、3、4下进行10折交叉验证，比较分类准确率、参数量、参数效率（%/100参数）、噪声下的准确率下降以及训练收敛曲线。结果显示：ND模型在深度4时达到97.63%准确率，参数量仅为MLP的约25%，在10%噪声下误差仅0.17%，显著优于基线，且收敛更平稳。

**⚠️ 局限性**

局限性：①在更深层网络中噪声鲁棒性随层数下降；②未验证跨传感器或不同地区的系数迁移性；③仅处理像素级特征，未结合空间上下文；④Attention‑gated 变体未显著提升；⑤实验规模相对有限，需更多数据与场景验证。

---

## 417. Heterogeneous Interaction Network Analysis (HINA): A New Learning Analytics Approach for Modelling, Analyzing, and Visualizing Complex Interactions in Learning Processes

**arXiv ID:** 2601.06771 | [PDF](https://arxiv.org/pdf/2601.06771v1)

**作者:** Shihui Feng `[一作]` (University of Hong Kong), Alec Kirkley `[通讯]` (University of Hong Kong)

**通讯引用:** 535 | [OpenAlex ID](https://openalex.org/A5058731900)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 HINA 框架，利用异质交互网络（HIN）对学习过程进行个体、配对和群体层面的多维度分析。

**💡 创新点**

创新点包括：① 将学生、技术、行为等多类型实体纳入同一网络模型；② 设计无参数 MDL 聚类、统计显著性检验与交互可视化三层分析流程；③ 在人机协作等复杂情境中首次量化并比较不同交互模式。

**🔧 技术方法**

使用的技术包括：图模型构建（HIN）、量化指标（量度、熵多样性）、二项式随机模型显著性检验、基于最小描述长度（MDL）的无参数聚类，以及交互式可视化工具。

**📊 数据集**

采用了本科项目管理课程中 27 名学生在两周 AI 辅助小组协作任务的完整聊天日志，经过 CoMPAS 编码后构造了两类 HIN（学生-伙伴和学生-内容+伙伴）。

**📈 对比分析**

通过案例研究展示 HINA 能捕捉人机交互细节并识别不同参与模式，相较于传统序列分析、聚类或单一类型网络方法，HINA 在解释学习过程的多元交互关系和发现群体模式方面表现更为细致和系统，尽管文中未给出量化的性能指标，但案例结果说明其优势。

**⚠️ 局限性**

主要限制包括：目前仅聚合处理，无法反映时间演化；缺乏对动态网络的建模；需要进一步验证其在更大规模、多学科场景中的泛化能力。

---

## 418. Structure-preserving learning and prediction in optimal control of collective motion

**arXiv ID:** 2601.06770 | [PDF](https://arxiv.org/pdf/2601.06770v1)

**作者:** Sofiia Huraka `[一作]` (University of Alberta), Vakhtang Putkaradze `[通讯]` (University of Alabama)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为 CO‑LPNets 的神经网络架构，能够仅凭观测数据在不知道控制哈密顿量或耦合形式的前提下，学习并预测多体无人机群在 SO(3) 或 SE(3) 组上进行 Lie‑Poisson 约化后的相空间动力学。

**💡 创新点**

创新点在于：
• 通过构造可解析的“测试哈密顿量”生成 Poisson 变换，并将其与神经网络参数化的权重按序列组合，从而实现结构保持的相空间映射；
• 证明了该类 Poisson 变换在足够多时能够完备覆盖所有可产生的动力学；
• 在参数量和数据量上实现显著压缩（例如 N=3 时仅约 1000 个参数，约 200‑300 个训练样本/维），同时保持 Casimir 不变。

**🔧 技术方法**

使用技术包括：
• Lie‑Poisson 约化理论与 Poisson 张量构造；
• 分裂方法与可解析测试哈密顿量的解析解；
• 浅层单隐藏层神经网络（每步权重由神经网络输出的标量决定）；
• 自动微分与 Adam 优化；
• 高精度 Lie‑Poisson 积分器用于生成真值数据与验证。

**📊 数据集**

数据集：
• 在 SO(3)^3 下生成 40 条短轨迹（每条 51 点，Δt=0.1）；
• 在 SE(3)^3 下生成 80 条短轨迹（同样 51 点，Δt=0.1）；
• 参数 χ 取 0.5，初始条件均匀采样于 [-1,1]^d（d=9 对 SO(3)，d=18 对 SE(3)）。

**📈 对比分析**

评价方法：将 CO‑LPNets 预测轨迹与高精度 Lie‑Poisson 积分器结果对比，主要指标包括 Casimir 保守性、能量误差（相对/绝对）和所有分量的平均绝对误差 MAE。结果表明：
• Casimir 在 1000 步内保持到机器精度；
• 能量误差虽不完全守恒，但始终在可接受范围内；
• MAE 在 10⁻⁶ 左右，且随着训练周期（10k 次）显著下降。

**⚠️ 局限性**

局限性：
• 仅针对可完整 Lie‑Poisson 约化的系统，未验证非完整约化或多组交叉耦合（如 SE(3)×SE(2)）的适用性；
• 对更大维度系统，参数量与计算量仍随维度平方级增长；
• 对能量守恒的保证仍是有限的，需进一步改进或加入能量正则化；
• 理论上对收敛速度与泛化能力的严格分析仍待完成。

---

## 419. ALFA: A Safe-by-Design Approach to Mitigate Quishing Attacks Launched via Fancy QR Codes

**arXiv ID:** 2601.06768 | [PDF](https://arxiv.org/pdf/2601.06768v1)

**作者:** Muhammad Wahid Akram `[一作]` (Deakin University), Dhananjay Thiruvady `[通讯]` (Deakin University)

**通讯引用:** 1031 | [OpenAlex ID](https://openalex.org/A5046309576)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ALFA框架，利用结构特征对fancy QR码进行安全检测，阻止Quishing攻击，并实现了支持Android和iOS的手机App；

**💡 创新点**

创新点在于：①从“安全设计”视角出发，使用QR码的结构特征而非深度学习或URL检测；②提出FAST方法，自动纠正黑白网格中因设计造成的误标模块；③通过生成黑白版QR并提取24条结构特征，利用预训练XGBoost模型实现分类；

**🔧 技术方法**

技术包括：图像灰度转换、阈值反转、版本迭代生成黑白QR、模块强度阈值检测、FAST模块恢复、结构特征提取、XGBoost分类、Flutter移动端开发、Flask后端服务；

**📊 数据集**

数据集：自建100个fancy QR码（50恶意 50合法），来源11款支持fancy功能的生成器；使用已有的400k黑白QR训练的XGBoost模型；

**📈 对比分析**

与10+款主流iOS/Android URL驱动App进行对比；ALFA在合法QR上60%正确、恶意QR上90%正确；FNR仅0.06%，FPR约0.7%；平均运行时间3.4秒，应用体积72.5 MB；

**⚠️ 局限性**

局限性包括：需要联网调用后端模型；对极小或复杂背景的QR裁剪与识别困难；FAST仅能修复预设模式的误标，数据模块仍可能错误；阈值参数对不同风格的QR可能不通用；App体积相对较大；

---

## 420. Federated Continual Learning for Privacy-Preserving Hospital Imaging Classification

**arXiv ID:** 2601.06742 | [PDF](https://arxiv.org/pdf/2601.06742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 421. GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO

**arXiv ID:** 2601.06767 | [PDF](https://arxiv.org/pdf/2601.06767v1)

**作者:** Shubhashis Roy Dipta `[一作]` (University of Maryland), Nadia Najjar `[通讯]` (University of North Carolina)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个困难度标注的 Bengali 数学数据集，并训练了首个能够用 Bengali 语言进行多步数学推理的 Ganit LLM。

**💡 创新点**

创新点包括：①困难感知数据集构建与过滤流程；②课程化的 GRPO 采样策略（CGRPO）解决低资源语言冷启动问题；③实现原生 Bengali 推理模型。

**🔧 技术方法**

使用了多阶段训练（SFT + GRPO），链式思考（CoT）SFT，GRPO（Group Relative Policy Optimization）以及格式、准确性、Bengali 推理的可验证奖励和困难感知采样。

**📊 数据集**

数据集来源于 1.1M 规模的 Bengali 数学题库（人类注释、翻译、LLM 生成等），划分为 CoT‑SFT、RLVR、hold‑out 评估集，并在 Bn‑MGSM、Bn‑MSVAMP 等公开基准上评测。

**📈 对比分析**

与多种规模的基线模型（0.6B–32B）以及现有 Bengali 与多语种模型对比，Ganit‑4B 在 Bn‑MGSM、Bn‑MSVAMP 上分别提升 8、7 分，Bengali 推理占比从 14% 提升至 88%，平均生成长度从 943 词降至 193 词，模型尺寸比同等性能大模型小 2–3.5 倍。

**⚠️ 局限性**

局限性包括：仅在 Bengali 上验证，难以直接推广到其他低资源语言；难度标签与过滤依赖评估模型，可能带来偏差；Bengali 推理奖励基于字符百分比，可能误判混合语言或符号使用。

---

## 422. A Backpropagation-Free Feedback-Hebbian Network for Continual Learning Dynamics

**arXiv ID:** 2601.06758 | [PDF](https://arxiv.org/pdf/2601.06758v1)

**作者:** Josh Li `[一作]` `[通讯]`, Josh Li

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种无反向传播、基于反馈–Hebbian 的小型网络，用于学习并持续保持多对关联；

**💡 创新点**

首次证明仅凭局部规则（中心化Hebbian、Oja 正则化和监督驱动）即可实现递归恢复、递延遗忘与交替条件化等持续学习行为；

**🔧 技术方法**

采用统一的局部学习规则，结合前向两层和专用反馈两层的预测-重建架构；

**📊 数据集**

使用十维二对关联任务（输入位置3或7映射到输出位置8,9或5,6的二标签映射）；

**📈 对比分析**

通过连通度轨迹、保留指数和图示验证，显示在顺序训练下实现LTD式遗忘并保持反馈记忆，在交替训练下实现并存；与单层或无反馈结构比较，证明两层+两反馈配置最低可行；

**⚠️ 局限性**

局限在任务维度过低、训练周期短、未测试高维或噪声环境，且缺乏与大规模基准的性能对比。

---

## 423. Graph Neural Network with One-side Edge Sampling for Fraud Detection

**arXiv ID:** 2601.06800 | [PDF](https://arxiv.org/pdf/2601.06800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 424. Benchmarking Egocentric Clinical Intent Understanding Capability for Medical Multimodal Large Language Models

**arXiv ID:** 2601.06750 | [PDF](https://arxiv.org/pdf/2601.06750v1)

**作者:** Shaonan Liu `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**通讯引用:** 10908 | [OpenAlex ID](https://openalex.org/A5019313200)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MedGaze-Bench，一个基于临床医生注视点的 egocentric 评估基准，用以测量医疗多模态大语言模型在空间、时间和标准意图理解上的能力。

**💡 创新点**

提出了三维临床意图框架（空间意图、时间意图、标准意图），并引入 Trap QA 机制以针对视觉与逻辑幻觉进行安全性评估，首次将注视点视为“认知光标”来捕捉医生的隐式意图。

**🔧 技术方法**

使用了多模态提示（注视点定位+文本引导）以及多轮问答生成策略，结合了视觉定位、因果推理和SOP对齐的评估方法。

**📊 数据集**

数据集来源于四个专业标注集合：开放式手术视频、阴道臀位分娩模拟、胸部X光和乳腺摄影的眼动记录，总计 4,491 个 QA 对。

**📈 对比分析**

在 9 个模型（包括 GPT‑5、Gemini‑3 Pro、Qwen3‑VL 系列、LingShu、MedGemma、EgoLife）上做零样本测试，GPT‑5 最高达 62.28% 的综合准确率，Trap QA 平均可靠率低于 70%，显示出模型在安全性和因果推理方面仍有显著差距。

**⚠️ 局限性**

局限性包括场景覆盖不足（仅限三类临床场景）、评估方式仅为多选题、未覆盖开放式生成任务，以及对 egocentric 视觉与文本对齐机制的依赖仍需进一步提升。

---

## 425. Algorithmic Reductions: Network Flow and NP-Completeness in Real-World Scheduling Problems

**arXiv ID:** 2601.06737 | [PDF](https://arxiv.org/pdf/2601.06737v1)

**作者:** Anay Sinhal `[一作]` (University of Florida), Amit Hirawat `[通讯]` (Poornima University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了医院病人-床位分配和大学课程排课两类实际调度问题，分别给出网络流求解和证明NP‑完全性并提供贪心近似。

**💡 创新点**

创新点在于将医院排床问题化为最大二部匹配并通过网络流证明可多项式求解；将课程排课等价于图着色并给出Welsh‑Powell和DSatur两种贪心近似，实验验证其性能优于理论上限。

**🔧 技术方法**

使用的技术包括网络流算法（Ford–Fulkerson/Edmonds–Karp）、最大二部匹配、图着色、贪心算法（Welsh‑Powell、DSatur）以及实验时间测量与复杂度回归。

**📊 数据集**

使用了随机生成的医院实例（患者/床位数相等，5个科室，每人兼容1–3科室）以及随机生成的课程冲突图（边概率0.3），实验规模分别从20–400和50–1000。

**📈 对比分析**

通过多次平均测定网络流求解时间并与理论O(n^3)比较，实际指数约2.51；对课程排课贪心算法测定时间与颜色数，实测指数为1.85/1.92，所用颜色远低于Δ+1上界，性能良好。

**⚠️ 局限性**

局限性包括仅在随机数据上验证，未涵盖真实医院/学校的多约束和动态变化；贪心算法不保证最优；网络流实现仍靠近三次复杂度，未探讨多目标优化或更高效的算法。

---

## 426. When Humans Judge Irises: Pupil Size Normalization as an Aid and Synthetic Irises as a Challenge

**arXiv ID:** 2601.06725 | [PDF](https://arxiv.org/pdf/2601.06725v1)

**作者:** Mahsa Mitcheff `[一作]` (University of Notre Dame), Adam Czajka `[通讯]` (University of Notre Dame)

**通讯引用:** 1324 | [OpenAlex ID](https://openalex.org/A5067121774)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过实验评估人类在虹膜验证中的表现，分别考察瞳孔尺寸变异与人工合成样本两种情境。

**💡 创新点**

创新点在于首次系统比较瞳孔尺寸归一化对人类判断的提升，以及人类对真实与合成虹膜图像识别差异的量化。

**🔧 技术方法**

使用线性拉伸（Daugman橡皮带模型）和非线性自编码器（EyePreserve）实现瞳孔对齐，采用StyleGAN与扩散模型生成合成虹膜，并利用HDBIF匹配器筛选样本。

**📊 数据集**

数据来源为University of Notre Dame公开虹膜数据库（真实图像）及其基于该数据库训练得到的合成图像。

**📈 对比分析**

通过对不同对齐策略和真实/合成样本的准确率（χ²检验）比较，发现瞳孔归一化至中等尺寸可将人类准确率提升至约83%，合成图像的准确率略低于真实图像（约57–75%）。

**⚠️ 局限性**

局限包括样本量有限、仅使用单一数据库、合成图像多样性不足，以及人类评判的主观性导致误差。

---

## 427. CliffordNet: All You Need is Geometric Algebra

**arXiv ID:** 2601.06793 | [PDF](https://arxiv.org/pdf/2601.06793v1)

**作者:** Zhongping Ji `[一作]` `[通讯]`, Zhongping Ji

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 Clifford 几何代数的视觉网络（CAN），通过几何乘积实现特征交互，去掉传统的 FFN 结构；

**💡 创新点**

创新点在于将 Clifford 几何乘积作为完整的特征交互机制，利用稀疏滚动实现线性复杂度，同时保持几何完整性，展现出无需 FFN 的高效表达能力；

**🔧 技术方法**

采用稀疏滚动（Shifted Geometric Product）、双流几何块、Gated Geometric Residual、卷积上下文（Laplacian）以及 SiLU、LayerScale 等技术实现网络；

**📊 数据集**

主要在 CIFAR‑100 数据集上进行实验，并计划在 ImageNet‑1K/21K 上扩展验证；

**📈 对比分析**

与 ShuffleNetV2、MobileNetV2、ViT‑Tiny 等在相同训练配置下对比，Nano 1.4M 参数达到 76.41%，Fast 2.6M 参数 77.63%，Base 3.0M 参数 78.05%，超越 ResNet‑18（11.2M）并形成新的 Pareto 前沿；

**⚠️ 局限性**

局限性包括目前实现依赖 PyTorch 的 roll 操作，效率受限；shift 设定固定缺乏自适应；未在大规模数据集验证；仅关注 2D 任务，未探究 3D/视频、动态度量等方向。

---

## 428. EpiCaR: Knowing What You Don't Know Matters for Better Reasoning in LLMs

**arXiv ID:** 2601.06786 | [PDF](https://arxiv.org/pdf/2601.06786v1)

**作者:** Jewon Yeom `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**通讯引用:** 2395 | [OpenAlex ID](https://openalex.org/A5065728469)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了EpiCaR框架，通过将推理训练视为一种认识性学习任务，使模型既能提升推理准确度，又能学习何时对推理结果保持警惕；

**💡 创新点**

创新点在于将自我评估作为目标内化到模型本身，形成双目标学习（推理+自评），从而显著降低迭代自训练中的校准代价；

**🔧 技术方法**

技术核心包括：迭代自监督微调、语义化置信度（verbalized confidence）预测、负样本自评标签、以及基于置信度加权的自一致性推理（CISC）等；

**📊 数据集**

主要使用了MATH、GSM8K和MBPP三大推理/代码生成数据集进行训练与评测；

**📈 对比分析**

与STaR、Slow Thinking和Model Merging等基线比较时，EpiCaR在3B以上规模模型上实现了同时提升推理准确率和校准误差（ECE/ Brier）并可在10条推理样本下匹配30条样本的性能，实现约3倍推理算力节省；

**⚠️ 局限性**

局限性包括：对低能力模型（如1B）表现不佳、对主观/模糊领域的适用性尚未验证、语义化置信度对提示敏感且在OOD场景下绝对校准仍存在波动。

---

## 429. From Text to Simulation: A Multi-Agent LLM Workflow for Automated Chemical Process Design

**arXiv ID:** 2601.06776 | [PDF](https://arxiv.org/pdf/2601.06776v1)

**作者:** Xufei Tian `[一作]` (East China University of Science and Technology), Ke Ye `[通讯]` (East China University of Science and Technology)

**通讯引用:** 1011 | [OpenAlex ID](https://openalex.org/A5100775246)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于多智能体和增强 MCTS 的端到端化学工艺设计自动化工作流，能够将自然语言工艺描述直接转化为可在专业仿真软件中验证的可执行配置。

**💡 创新点**

创新点在于：① 通过四个专用代理实现从任务理解到拓扑生成、参数配置、评估分析的完整闭环；② 采用增强 MCTS（E‑MCTS）对失败方案的潜在价值进行评估并动态重访；③ 与仿真软件实现实时双向交互，实现自动迭代优化。

**🔧 技术方法**

使用了大型语言模型（GPT‑4o）、LangGraph/LangChain 多智能体框架、E‑MCTS 搜索、链式思维与少样本提示、异步 HTTP 与仿真软件接口等技术。

**📊 数据集**

使用了自建的 Simona 数据集，共 1000 条工艺描述，覆盖不同单元操作、复杂度与细节程度。

**📈 对比分析**

通过与 GPT‑4o/Claude‑Sonnet‑4 直接生成、Swarm/AutoGen/CrewAI/MetaGPT 多智能体基线以及三名专家手工设计进行对比，实验表明本方法在仿真收敛率达到 80.3%（高于基线），设计时间缩短 89%，整体综合得分仅次于专家设计。

**⚠️ 局限性**

局限性包括：仍未完全达到专家级的经济可行性，对极复杂或高度模糊描述的鲁棒性有限，且依赖仿真软件接口和高质量参数样例；对失败配置的利用仍有提升空间。

---

## 430. ImmuniFraug: A Metacognitive Intervention Anti-Fraud Approach to Enhance Undergraduate Students' Cyber Fraud Awareness

**arXiv ID:** 2601.06774 | [PDF](https://arxiv.org/pdf/2601.06774v1)

**作者:** Xiangzhe Yuan `[一作]` (University of Iowa), Siying Hu `[通讯]` (University of Queensland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并验证了一种基于大语言模型的多模态、元认知交互式欺诈模拟平台ImmuniFraug，用以提升中国高校本科生的网络诈骗意识与防护能力。

**💡 创新点**

将LMM驱动的沉浸式情境模拟与即时反馈相结合，首次在诈骗教育中引入元认知反思和多模态交互（文字、语音、视觉），从而提升学习效果和情境逼真度。

**🔧 技术方法**

采用Doubao.AI大语言模型实现自然语言交互，配合语音识别与合成、视觉化头像生成，并在实验中使用线性混合效应模型、主题分析和沉浸度量表进行评估。

**📊 数据集**

基于中国公安部发布的十大常见诈骗类型，实验共招募846名本科生，并使用真实的诈骗情景脚本与官方文本材料进行对照。

**📈 对比分析**

与传统静态文本教材对照后，实验组在欺诈认知测验中获得显著提升（p=0.026，Cohen’s d≈0.15），且在自评意识、效能感、行为意向等元认知维度均表现出较高得分，证明该交互式干预相对更有效。

**⚠️ 局限性**

受试者已有较高的防骗背景导致效应量有限；模拟对话长度受token限制，无法再现长期诈骗；语音自然度与多媒体真实性仍有待提升；仅针对大学本科生，需进一步验证对其他群体的适用性与可扩展性。

---

## 431. The Complexity of Finding Missing Answer Repairs

**arXiv ID:** 2601.06764 | [PDF](https://arxiv.org/pdf/2601.06764v1)

**作者:** Jesse Comer `[一作]` (University of Pennsylvania), Val Tannen `[通讯]` (University of Pennsylvania)

**通讯引用:** 5331 | [OpenAlex ID](https://openalex.org/A5036528289)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文研究“缺失答案修复”问题，即在给定查询 Q、数据库实例 I 和缺失的答案元组 a 的情况下，寻找一组插入/删除操作（修复）使得 a 成为 Q 在修复后数据库中的答案。作者从组合复杂度和数据复杂度两方面对该问题的四个版本（决策、大小限制、最小修复大小、最小修复计算）进行了系统的复杂度分类，给出了可多项式求解的查询子类与不可多项式求解的查询子类，并提供了逼近难度分析。特别地，作者证明：对所有半正 Datalog 查询，在数据复杂度下可在多项式时间内求解最小修复；而当查询允许投影和连接时，决策与最小修复大小问题均为 NP‑hard，并且在最小修复大小问题上具有 log‑n 近似难度。

**💡 创新点**

创新点：
1) 统一了缺失答案修复与查询可满足性、视图更新、数据库修复等问题的关系，并从组合复杂度角度给出闭包条件（弱投影/选择）下的可判定性与等价性。 
2) 提出了“可计算的受限修复”“常数大小修复”等新概念，并利用它们证明了半正 Datalog 在数据复杂度下的可多项式性。 
3) 给出了针对不同查询子类的精细化复杂度表（如 UCQ、self‑join‑free、projection‑free 等），并通过严格与度量归约完成了 NP‑hard、log‑n 近似难度、[log(n)] 与 O(n^2) oracle 上的精确上界。 
4) 开辟了基于 “unfolding” 与“等价类” 的方法，将半正 Datalog 查询转化为（可能无穷的）UCQ，从而导出常数大小修复的上界。

**🔧 技术方法**

技术手段：
- 组合与数据复杂度分析，利用查询的闭包属性（弱投影/选择）构造多项式时间归约。
- 归约到判定可满足性、最小集合覆盖等经典 NP‑hard 问题，采用严格归约和度量归约证明硬度。
- 通过构造 oracle 机器（^、^ [O(log n)]、^ [O(n^2)] 等）给出算法上界。
- 引入“可计算的受限修复”与“常数大小修复”概念，利用 bijection 与 generic 查询属性限定搜索空间。
- 采用 unfoldings 将半正 Datalog 转化为 UCQ，使用等价类划分证明常数大小修复存在。

**📊 数据集**

本工作为理论论文，没有使用实验数据集；所有结果均通过严格的数学证明与归约得到。

**📈 对比分析**

性能评估以复杂度理论为准：
- 在组合复杂度下，若查询属于 self‑join‑free、projection‑free 等子类，则决策与最小修复大小问题可在多项式时间内解决；
- 若查询允许投影与连接，则问题为 NP‑hard，且最小修复大小问题的最优值是 [log(n)] 近似难度；
- 对半正 Datalog，数据复杂度下可在多项式时间内得到最小修复，且最小修复大小问题可在 O(n^2) oracle 调用下求解。 
- 对于更强的查询（如带全称量化的 Datalog），存在无穷大小的最小修复，导致不可多项式求解。

**⚠️ 局限性**

限制与未解决问题：
- 复杂度表中部分条目仅给出上界或下界，尚未完全闭合（如某些 UCQ 子类的决策与最小修复大小的精确复杂度）。
- 仅覆盖存在性与最小性问题，对其他修复优化目标（如最小副作用、最小化多条缺失答案等）未作分析。
- 对包含全称量化或更强否定的查询（如 stratified Datalog、全称量化关系代数）未给出可多项式解法，常数大小修复理论不适用。
- 本研究仅在布尔语义（集合语义）下展开，未涉及 semiring 注释等更一般语义。

---

## 432. Deep Recurrent Hidden Markov Learning Framework for Multi-Stage Advanced Persistent Threat Prediction

**arXiv ID:** 2601.06734 | [PDF](https://arxiv.org/pdf/2601.06734v1)

**作者:** Saleem Ishaq Tijjani `[一作]` (Plymouth University), Matthew Craven `[通讯]` (Plymouth University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种结合CNN‑RNN与HMM的混合深度概率学习框架E‑HiDNet，用于多阶段APT进展的预测与推断。

**💡 创新点**

创新点在于将深度学习的语义特征抽取与隐马尔可夫模型的状态推理融合，并改进Viterbi算法以处理缺失观测，实现对APT阶段的实时、前瞻性预测。

**🔧 技术方法**

使用1D卷积网络提取空间特征，LSTM/GRU捕捉时间依赖，随后通过全连接层生成HMM观测向量，利用Baum–Welch训练HMM并改进Viterbi解码。

**📊 数据集**

数据集为构造的S‑DAPT‑2026，包含约127k条包含真实APT生命周期逻辑的合成警报，分为训练/验证/测试三份。

**📈 对比分析**

与单独HMM和传统基线相比，E‑HiDNet在单步/双步预测中准确率提升约20%；当观测数≥4时，准确率接近100%，显示出在稀疏观测下的鲁棒性和更快收敛。

**⚠️ 局限性**

局限在于仅使用合成数据验证，缺乏大规模真实APT场景测试；对模型超参数和缺失观测的处理仍依赖先验设定，且早期阶段预测仍存在轻微波动。

---

## 433. Logic-Driven Semantic Communication for Resilient Multi-Agent Systems

**arXiv ID:** 2601.06733 | [PDF](https://arxiv.org/pdf/2601.06733v1)

**作者:** Tamara Alshammari `[一作]` (University of Oulu), Mehdi Bennis `[通讯]` (University of Oulu)

**通讯引用:** 42814 | [OpenAlex ID](https://openalex.org/A5061429095)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了基于知识推理和行动恢复的多智能体系统韧性形式化定义，并设计了语义通信支持的分布式架构与算法，随后给出正式验证与仿真评估。

**💡 创新点**

创新点在于将韧性拆分为认知韧性和行动韧性，并用可量化的恢复时间和耐久时间表述，首次实现了韧性指标的形式化验证与有限时域检查。

**🔧 技术方法**

技术上使用了时序本体论（temporal epistemic logic）、Kripke语义模型、分布式信念更新与语义通信策略，以及基于模型检查的正式验证和 UCB bandit 的仿真实验。

**📊 数据集**

实验采用自建的多智能体多臂赌博机环境，随机生成臂收益均值，设置不同噪声水平 σ，未使用公开真实数据集。

**📈 对比分析**

与独立和协同的折扣式 UCB 基线对比，实验显示 Kripke 框架在冲击恢复速度、累计回报和累计遗憾方面均优于基线，协同版性能最佳。

**⚠️ 局限性**

局限包括对通信可靠性与拓扑（如环形拓扑导致恢复时间随节点数增长）、参数调优敏感性，以及缺乏真实网络部署验证。

---

## 434. FO-Complete Program Verification for Heap Logics

**arXiv ID:** 2601.06719 | [PDF](https://arxiv.org/pdf/2601.06719v1)

**作者:** Adithya Murali `[一作]` (University of Wisconsin Madison), P. Madhusudan `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出并实现了两种新的堆逻辑——框架逻辑（Frame Logic，FL）和一种具有唯一堆片段语义的分离逻辑（SLFL），并给出了基于FO‑完整性（FO‑completeness）的自动验证方法。

**💡 创新点**

创新点在于①引入FO‑完整性概念，证明在fixpoint语义下，FL和SLFL可实现FO完备的验证；②设计了“支持”算子和云算子，使得堆片段唯一化；③将SL公式翻译为FL并利用FO完备的自然证明技术完成验证，从而突破传统分离逻辑中不可判定/不完备的瓶颈。

**🔧 技术方法**

技术手段包括：基于FOR D（first‑order logic with recursive definitions）的语法与语义；自然证明（Natural Proofs）实现的FO完备定理推理；SMT求解器（如Z3）完成量化实例化与决策；支持算子与云算子在FL中的实现；以及SLFL的翻译规则和自动验证管道。

**📊 数据集**

数据集由29个典型的数据结构程序组成，包括单链表、双链表、循环链表、二叉搜索树、红黑树、Treap等，来源于前人工作并统一整理为实验基准。

**📈 对比分析**

实验使用MacBook M2 Pro平台，FLV和SLFLV工具均能在一分钟以内完成所有基准验证；与FL相比，SLFL的验证时间略长但通过优化可缩短约4%；通过与传统分离逻辑工具比较，展示了FO完备方法在效率与准确性上的优势。

**⚠️ 局限性**

局限性包括：需要程序员手工提供归纳性质/递归定义的支持等Lemma；FO‑完备性仅对fixpoint语义成立，least fixpoint下仍需额外证明；SLFL翻译导致表达式膨胀；工具目前不支持循环（while）以及魔法手杖等高级分离逻辑运算。

---

## 435. Vextra: A Unified Middleware Abstraction for Heterogeneous Vector Database Systems

**arXiv ID:** 2601.06727 | [PDF](https://arxiv.org/pdf/2601.06727v1)

**作者:** Chandan Suri `[一作]` (Columbia University), Gursifath Bhasin `[通讯]` (Columbia University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Vextra中间件抽象层，将多种向量数据库的差异化API统一为单一高层接口，并通过适配器和AST编译器实现跨数据库查询与写入的无缝转换。

**💡 创新点**

创新点包括：①统一的向量数据库抽象模型与通用API；②插件式适配器架构，支持快速接入新数据库；③基于AST的编译器式查询翻译，提升语义一致性与可维护性；④提供逃逸口（escape hatch）以兼顾后端特性；⑤系统性性能评估与迁移案例验证其可移植性。

**🔧 技术方法**

核心技术有 Adapter 设计模式、抽象数据模型、AST 解析与转换、编译器式查询翻译、Python SDK 调用、gRPC/REST/GraphQL 通讯、统一错误映射与日志。

**📊 数据集**

使用 GloVe‑25‑angular 数据集（120 万条 25 维向量）作为基准集进行写入与查询性能评测。

**📈 对比分析**

对比方法：在同一 AWS c5.2xlarge 实例上分别使用原生 SDK 与 Vextra 接口进行 1,000/1,100 条记录批量 upsert 与带过滤的 top‑k 查询。结果显示单条写/查询延迟约 3–7% 之差，批量写与复杂过滤下延迟差距降至 1–3%，吞吐量差异亦在可接受范围。

**⚠️ 局限性**

主要局限：1) 仍存在一定的性能开销，尤其对极低延迟需求场景；2) 需要持续维护适配器以跟进各数据库的 API 演进；3) 逃逸口虽提供高级功能，但过度使用会削弱可移植性；4) 对于需要严格一致性或事务支持的应用，目前抽象层尚未覆盖。

---

## 436. Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers

**arXiv ID:** 2601.06798 | [PDF](https://arxiv.org/pdf/2601.06798v1)

**作者:** Zhiyang Zhang `[一作]` (Kuaishou Inc), Guorui Zhou `[通讯]` (Kuaishou Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于大型语言模型（LLM）的通用生成式推荐框架GRLM，核心创新是用语义丰富、结构化的Term IDs (TIDs) 作为项目标识，并通过上下文感知的Term生成、集成指令微调以及弹性标识对齐实现高效、可靠的推荐。

**💡 创新点**

创新点在于：① 用自然语言关键词构造TID，消除SIDs的语义鸿沟与hallucination；② 引入上下文感知Term生成（CTG）使TID在全局一致且局部区分度高；③ 通过集成指令微调（IIFT）将项标识生成与序列推荐双任务联合训练；④ 弹性标识对齐（EIG）实现直接与结构化双轨候选映射。

**🔧 技术方法**

技术细节包括：使用Qwen3系列LLM作为生成与推荐基座；CTG利用邻域检索的项目元数据构造prompt；IIFT在训练时同时进行生成Term和用户行为序列预测；EIG在推理时先做精确字符串匹配，再用结构化分数匹配。

**📊 数据集**

实验数据集：① 领域内：Amazon Beauty、Sports、Toys；② 跨域：Sports–Clothing（Leisure）与 Phones–Electronics（Technology）对。

**📈 对比分析**

与SASRec、BERT4Rec（序列方法）；TIGER、HSTU（生成式）；IDGenRec、OneRec‑Think（LLM‑based）；TriCDR、LLM4CDSR、GenCDR（跨域）等基线比较。GRLM在Recall@5/10、NDCG@5/10上均领先最强基线，提升幅度从约7%至30%不等，跨域场景提升超过50%。

**⚠️ 局限性**

局限性：① CTG当前仅使用固定外部检索模型，未充分挖掘领域特定检索潜力；② 只在Qwen3上验证，缺乏对更大模型或不同LLM的适配性评估。

---

## 437. DaQ-MSA: Denoising and Qualifying Diffusion Augmentations for Multimodal Sentiment Analysis

**arXiv ID:** 2601.06870 | [PDF](https://arxiv.org/pdf/2601.06870v1)

**作者:** Jiazhang Liang `[一作]` (South China Normal University), Sijie Mai `[通讯]` (South China Normal University)

**通讯引用:** 1596 | [OpenAlex ID](https://openalex.org/A5010270301)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种完全自动化的多模态情感分析框架 DaQ-MSA，通过扩散模型对视频和音频进行语义保留的数据增强，并利用质量感知模块对生成样本进行评分和加权，以提升多模态大语言模型的鲁棒性与泛化能力。

**💡 创新点**

创新点在于：①引入质量感知（QA）评分机制，对扩散生成样本的跨模态一致性和语义可靠性进行自动评估与加权；②采用无监督的负样本构造策略（特征混合、随机掩蔽、标签翻转）训练 QA 模块；③将扩散生成与多模态大语言模型训练解耦，形成两阶段“生成‑评估‑加权”流程，实现零人工标注的数据增强。

**🔧 技术方法**

核心技术包括：扩散模型（FateZero 用于视频风格迁移，Seed-VC 用于音频声纹转换）；多模态编码器（SigLIP、Whisper、BERT）提取视频、音频与文本特征；QA 模块为两层 MLP 通过特征拼接和标签嵌入实现质量评分；加权微调采用样本级加权交叉熵训练大语言模型。

**📊 数据集**

使用三大多模态情感分析基准数据集：中文 CH‑SIMS、英文 CMU‑MOSI、以及包含讽刺检测的 MUStARD，另外构建并发布了扩散增强版多模态情感数据集。

**📈 对比分析**

与现有最先进方法相比，DaQ-MSA 在 CH‑SIMS 上实现 Acc2/Acc5 分别达到 90.15%/61.49%，在 CMU‑MOSI 上实现 Acc2/Acc7/MAE/Corr 分别达到 92.37%/55.33%/0.498/0.907，均超过对照基线 5–10% 甚至更高；在 MUStARD 上也实现了 70.59% 的统一性能提升；实验表明该方法在少量标签场景下仍能保持竞争力。

**⚠️ 局限性**

局限性包括：①扩散模型推理步骤多、计算成本高，难以实现实时在线增强；②生成过程中仍存在唇同步偏差，QA 只能通过加权抑制，无法从生成侧主动校正；③目前验证仅覆盖中英两种语言，对低资源或更复杂社会文化背景的情感表达泛化仍需进一步研究。

---

## 438. U-MASK: User-adaptive Spatio-Temporal Masking for Personalized Mobile AI Applications

**arXiv ID:** 2601.06867 | [PDF](https://arxiv.org/pdf/2601.06867v1)

**作者:** Shiyuan Zhang `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**通讯引用:** 5765 | [OpenAlex ID](https://openalex.org/A5068782412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 U-MASK 框架，将个性化移动 AI 行为建模视为稀疏时空张量的条件补全，利用用户自适应时空遮罩统一即时性、稳定性与泛化三大需求。

**💡 创新点**

① 统一条件补全视角消除三角困境；② U-MASK 用户与任务特定自适应遮罩；③ U-SCOPE 通过 LLM 生成任务无关语义用户嵌入；④ 扩散 Transformer 实现多任务生成与推理。

**🔧 技术方法**

扩散模型（Diffusion Transformer）、Transformer、Gumbel‑TopK 采样、Fisher 信息近似的敏感度预测、LLM 语义嵌入、特征加权空间相似度、随机梯度下降训练。

**📊 数据集**

七个真实移动数据集（用户位置、APP 使用、网络流量，15 分钟时间粒度、500m 网格、3000 用户）。

**📈 对比分析**

与统计方法、NLP 方法、传统时空网络（CSDI、PatchTST）以及 LLM 增强基线对比；在短期/长期预测与冷启动推荐上均实现 5–90% 的误差下降，最高可达 90%+ 的性能提升。

**⚠️ 局限性**

仍依赖离线训练与中心化推理；遮罩策略受预训练模型质量限制；对极端稀疏或动态变化的数据适配有限；模型复杂度高，部署成本较大。

---

## 439. A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning

**arXiv ID:** 2601.06851 | [PDF](https://arxiv.org/pdf/2601.06851v1)

**作者:** Pedro Urbina-Rodriguez `[一作]` (Imperial College London), Pedro A. M. Mediano `[通讯]` (Imperial College London)

**通讯引用:** 3810 | [OpenAlex ID](https://openalex.org/A5074280948)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究大型语言模型（LLMs）在训练过程中是否会出现类似人脑的“协同核心”，并通过消融和强化学习细调实验验证其功能重要性。

**💡 创新点**

发现LLMs的中间层会自发形成协同核心，协同核心与人脑的默认模式网络和执行控制网络高度相似，且协同核心对模型性能与泛化尤为关键。

**🔧 技术方法**

采用信息理论框架——部分信息分解（PID）与集成信息分解（ΦID），对注意力头（或专家模块）进行时间序列信息分解，量化协同与冗余；利用KL散度评估消融对行为的影响；用强化学习（GRPO）和监督微调对协同核心进行细调。

**📊 数据集**

使用多种公开LLM（Gemma、Llama、Qwen、DeepSeek等）在包含6类认知任务的自定义提示集（语法、词性、算术、常识、抽象推理、情感交互）以及MATH数学推理基准进行评估；对训练过程中的多步模型快照（如Pythia-1B）进行分析。

**📈 对比分析**

对比随机消融、协同核心消融与冗余核心消融以及协同核心/冗余核心/随机子集细调；结果显示在强化学习细调中，协同核心细调带来显著性能提升（Hedges' g≈1.4‑5.0），而监督细调则无显著差异；在消融实验中，协同核心消融导致行为KL散度和MATH准确率下降最为剧烈。

**⚠️ 局限性**

局限性包括：仅考察注意力头/专家模块，未涉及MLP层；使用L2范数作为激活近似，未捕捉向量空间信息；强化学习细调实验未覆盖完整预训练过程；实验规模受计算资源限制，缺乏对标准大规模基准的进一步验证。

---

## 440. Speak While Watching: Unleashing TRUE Real-Time Video Understanding Capability of Multimodal Large Language Models

**arXiv ID:** 2601.06843 | [PDF](https://arxiv.org/pdf/2601.06843v1)

**作者:** Junyan Lin `[一作]` (Hong Kong Polytechnic University), Xiaoyu Shen `[通讯]` (Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Institute of Digital Twin, EIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出三种打破全局位置连续性的新位置编码策略（OSPE、GDPE、GIPE），实现多模态大型语言模型在实时视频推理中的并行感知与生成

**💡 创新点**

核心创新是放松位置编码的全局连续性约束，使输入输出可在位置空间独立并行处理，而不改动模型架构或重新对齐视觉‑语言

**🔧 技术方法**

技术手段包括三种位置编码设计（重叠、分组解耦、间隔隔离），在 Qwen2.5‑VL 基础上进行微调；理论分析推导并行流式推理的延迟与加速

**📊 数据集**

使用 PE‑Video（视频描述）和 FunQA（视频问答）两个流式数据集，并在 Qwen2.5‑VL 预训练模型上进行实验

**📈 对比分析**

与离线原始模型及传统交错流式模型对比；GDPE 在保持几乎相同的 CIDEr、BLEU、METEOR 等指标的同时，流式流畅度得分更高，且理论上可实现约 2× 的延迟降低

**⚠️ 局限性**

局限性包括仍需额外微调、仅在视频描述与问答任务验证、对极端帧速/文本比例的鲁棒性不足、未在多 GPU 真正硬件上系统评测并行效果

---

## 441. AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents

**arXiv ID:** 2601.06818 | [PDF](https://arxiv.org/pdf/2601.06818v1)

**作者:** Xuannan Liu `[一作]` (Beijing University of Posts and Telecommunications), Ran He `[通讯]` (Center for Research on Intelligent Perception and Computing, NLPR, CASIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LLM代理的自动幻觉归因任务，并构建了包含693条多步骤轨迹、5类14子类幻觉分类与多级人工注释的AgentHallu基准。

**💡 创新点**

创新点包括：①新颖的幻觉归因任务；②细粒度5类14子类幻觉分类体系；③大规模多框架、多领域轨迹与多层注释；④基于G‑EVAL的归因质量评估框架。

**🔧 技术方法**

技术手段涵盖LLM代理（7种框架）、三阶段过滤与归纳式分类、人工注释与协作评审、思考模式、Step‑by‑Step提示，以及G‑EVAL评估器。

**📊 数据集**

使用的基准数据集来源于SimpleQA、GPQA、MATH‑500、AIME、GAIA、BFCL V3、HLE等五大领域，共693条经过人工精修的轨迹。

**📈 对比分析**

通过对13款顶尖LLM（含GPT‑5、Gemini‑2.5‑Pro等）的二分类和归因定位评估，最佳定位精度仅达41.1%，工具使用类幻觉定位最差仅11.6%，相比现有幻觉检测数据集表现明显下降。

**⚠️ 局限性**

局限性：未覆盖快速演进的新型幻觉模式；仅限文本轨迹，未考虑多模态代理；需持续扩展与更新以跟上代理与工具生态的变化。

---

## 442. Learning-Augmented Performance Model for Tensor Product Factorization in High-Order FEM

**arXiv ID:** 2601.06886 | [PDF](https://arxiv.org/pdf/2601.06886v1)

**作者:** Xuanzhengbo Ren `[一作]` (Nagoya University), Seiya Nishizawa `[通讯]` (RIKEN)

**通讯引用:** 1135 | [OpenAlex ID](https://openalex.org/A5038510654)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对高阶有限元求解中的张量 n 模式乘积核，提出了依赖链分析的模型并用 XGBoost 进行学习增强，以预测不同循环体拆分配置下的性能。

**💡 创新点**

创新点在于将循环体拆分映射到指令级依赖链，结合可解释的分析模型与机器学习实现对 FMA 指令周期的精确预测，并首次在 A64FX 与 Xeon 处理器上验证了此方法。

**🔧 技术方法**

使用了依赖链分析、循环拆分策略、FMA 指令延迟/吞吐模型、XGBoost 回归模型，并利用 PAPI 与 Fujitsu profiling 工具收集性能数据。

**📊 数据集**

数据集来自对 1~15 次多项式阶数下的张量 n 模式乘积核在 Fujitsu A64FX 与 Intel Xeon Gold 6230 上的 1,800 条样本，训练集中约 1,500 条。

**📈 对比分析**

与传统 Roofline 与 ECM 模型对比，学习增强模型在两种架构上均取得 1%–24%（A64FX）和 1%–13%（Xeon）平均绝对百分误差，明显优于 Roofline 的 1%–73% 与 ECM 的 8%–112%。

**⚠️ 局限性**

局限性包括对 L1 缓存驻留假设的依赖，无法很好处理 P=15 时的数据传输瓶颈，以及仅关注 FMA 指令且未覆盖其他指令类型或缓存缺失影响。

---

## 443. Paraphrasing Adversarial Attack on LLM-as-a-Reviewer

**arXiv ID:** 2601.06884 | [PDF](https://arxiv.org/pdf/2601.06884v1)

**作者:** Masahiro Kaneko `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Masahiro Kaneko `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 3832 | [OpenAlex ID](https://openalex.org/A5005531754)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种黑盒式的语义保留改写攻击（PAA），能够在不改变论文主旨的前提下，通过对摘要进行可理解的改写提升LLM审稿分数。

**💡 创新点**

创新点在于引入了以LLM评审反馈为目标的基于自我学习的ICL改写优化方法，证明了仅用语义保留改写即可欺骗LLM审稿系统。

**🔧 技术方法**

采用了大语言模型（GPT-4o、Gemini 2.5、Sonnet 4、OLMo 3.1‑32B、Qwen 3）进行改写和评估，并利用BERTScore、perplexity等指标进行语义与自然度过滤。

**📊 数据集**

使用了从arXiv收集的5个ML/NLP会议（ACL 2025、NeurIPS 2025、ICML 2025、ICLR 2025、AAAI 2025）未被接受论文的PDF和LaTeX源作为实验数据集。

**📈 对比分析**

与原始论文、单纯改写基线以及PAI、PAIR等对比实验显示，PAA在所有会议和模型上均显著提升审稿分数（平均提升≈1.3–1.9分，p<0.01），但改写后的审稿文本情感并未显著改善。

**⚠️ 局限性**

主要局限包括仅针对英文摘要、仅使用BERTScore衡量语义相似、需要大量API调用、未测试对其他章节或多语言的适用性。

---

## 444. MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation

**arXiv ID:** 2601.06874 | [PDF](https://arxiv.org/pdf/2601.06874v1)

**作者:** Changli Wu `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**通讯引用:** 4101 | [OpenAlex ID](https://openalex.org/A5014628588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了仅使用稀疏 RGB 视图进行 3D 参照分割的 MV-3DRES 任务，并设计了 MVGGT 双分支 Transformer 模型与 PVSO 优化方案，实现端到端几何-语言融合；

**💡 创新点**

创新点在于（1）从传统点云转向稀疏视图；（2）引入冻结的几何分支与可训练的多模态分支进行跨视角与跨模态注意；（3）首次发现并解决前景梯度稀释问题，提出 Per‑view No‑target Suppression；（4）构建 MVRefer 基准数据集；

**🔧 技术方法**

采用 ViT 视图编码、Roberta 语言编码、双分支 Transformer（交叉视角注意+跨模态注意）、冻结的 Pi3 结构、前景梯度稀释分析及 PVSO 采样与加权损失；

**📊 数据集**

基于 ScanRefer 与 ScanNet 的稀疏视图采样构建 MVRefer 数据集；

**📈 对比分析**

在 MVRefer 上与两阶段、2D‑Lift 等基线对比，MVGGT 在 mIoU_global 上达 39.9、视图 mIoU 69.3，显著优于基线；在传统 ScanRefer 下，Acc@25 83.6、mIoU 65.2，几乎与全点云方法持平；

**⚠️ 局限性**

局限在于对视图数量和目标可见性的依赖，稀疏视图缺失目标或严重遮挡仍难以完全恢复；模型训练仍需预先采样的多视角数据，实时在线部署需进一步优化。

---

## 445. United We Defend: Collaborative Membership Inference Defenses in Federated Learning

**arXiv ID:** 2601.06866 | [PDF](https://arxiv.org/pdf/2601.06866v1)

**作者:** Li Bai `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8461 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个协作式联邦学习成员推断防御框架 CoFedMID，用于在部分客户端保护与协作的场景下抵御基于时间轨迹的成员推断攻击。

**💡 创新点**

创新点包括：①将客户端分为防御联盟并协同作业，②设计基于类别划分的局部训练样本分配与衰减策略，③引入基于样本贡献的再循环与置信度正则化以恢复性能，④采用聚合中性噪声实现局部模型扰动。

**🔧 技术方法**

技术手段包括：类别划分与动态衰减、基于多臂赌博机的样本再循环、置信度正则化、聚合中性噪声注入、联邦平均（FedAvg）训练。

**📊 数据集**

使用了三个公开图像分类数据集：CIFAR‑10、CIFAR‑100 与 TinyImageNet，并在 ResNet‑18 与 WideResNet‑16‑4 上进行实验。

**📈 对比分析**

与七种轨迹式成员推断攻击以及三种基线防御（梯度稀疏化、梯度噪声、DPSGD）对比，CoFedMID 在 AUC 与 TF01 指标上平均下降 0.04–0.08，且模型精度下降仅 0.03–0.05，表现出较高的防御效果与较低的性能损失。

**⚠️ 局限性**

局限性：①仍存在隐私‑性能权衡，联盟规模增大时防御效果与准确率下降会加剧；②目前仅针对监督分类任务，对无标签或非图像任务的推广有限；③未考虑对抗性攻击或投毒攻击的鲁棒性。

---

## 446. MoE-DisCo:Low Economy Cost Training Mixture-of-Experts Models

**arXiv ID:** 2601.06857 | [PDF](https://arxiv.org/pdf/2601.06857v1)

**作者:** Xin Ye `[一作]` (Institute of Computing Technology), Yunquan Zhang `[通讯]` (Institute of Computing Technology)

**通讯引用:** 76086 | [OpenAlex ID](https://openalex.org/A5001666028)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种低成本的分阶段 Mixture‑of‑Experts 训练框架 MoE‑DisCo，先将完整 MoE 拆分为多组单专家稠密子模型，使用无监督聚类将数据划分给各子模型，在低成本 GPU 上并行训练，然后再将专家合并并在少量高成本 GPU 上进行全局细调。

**💡 创新点**

创新点在于（1）将 MoE 结构与 Block Coordinate Descent (BCD) 结合，只在一次训练步骤中更新单一专家与共享骨干，极大降低内存与计算负担；（2）利用无监督聚类实现专家‑数据的去耦合，提升专家的专一性与整体多样性；（3）采用两阶段训练（子模型独立训练 + 轻量级全局细调）显著减少高成本 GPU 的使用时间与费用。

**🔧 技术方法**

技术细节包括 BCD、SimulParallel SGD、k‑means 聚类、稠密子模型训练、专家参数合并与加权平均、以及短时全局微调。

**📊 数据集**

实验数据集：C4、WikiText‑2、OpenWebText；模型为 Qwen1.5‑MoE‑2.7B 与 Llama‑MoE‑3.5B，均使用 4 个专家。

**📈 对比分析**

与全参数 MoE 训练对比：MoE‑DisCo 在三大数据集上实现了相同或更优的训练损失、PPL 与下游任务（ARC‑E、MMLU、HellaSwag、PIQA）性能；训练步骤数降至约四分之一，高成本 GPU 的使用时长缩短 50% 以上；总体经济成本降低 48%–70%，并在时间上取得同等或更快收敛。

**⚠️ 局限性**

limitations：未在 10B+ 大模型上验证；数据集规模与多样性有限；缺少动态专家数调整策略及更丰富的评估指标。

---

## 447. Variational decomposition autoencoding improves disentanglement of latent representations

**arXiv ID:** 2601.06844 | [PDF](https://arxiv.org/pdf/2601.06844v1)

**作者:** Ioannis Ziogas `[一作]`, Leontios J. Hadjileontiadis `[通讯]` (Aristotle University of Thessaloniki)

**通讯引用:** 10124 | [OpenAlex ID](https://openalex.org/A5072772953)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种基于变分分解自编码（VDA/DecVAE）的无解码器框架，用信号分解、对比自监督和变分推断实现对高维时序信号的可解释、解耦表示学习。

**💡 创新点**

创新点在于将分解结构直接嵌入变分模型，利用正交性、重构和对比损失强化解耦；仅使用编码器即可获得解耦特征；并提供多尺度分支以捕获不同时间尺度的因子。

**🔧 技术方法**

技术方法包括：变分自编码器、信号分解（EWT、EMD、VMD、FD）、自监督对比损失、DELBO目标、正交性约束、多尺度分支、GELU、LayerNorm 等。

**📊 数据集**

使用四个数据集：SimVowels（模拟语音）、TIMIT（真实语音）、VOC‑ALS（ALS病人语音），IEMOCAP（情感语音）来评估模型。

**📈 对比分析**

与 β‑VAE、FHVAE、MFL‑VAE、FactorVAE、PCA/ICA 等方法比较，DecVAE 在解耦指标（DCI、modularity/explicitness）、信息量、IRStability 以及下游任务（音素/说话人识别、情感识别、疾病阶段预测）均显著优于对比模型，展示了更高的泛化和解释性。

**⚠️ 局限性**

局限性包括：模型缺乏生成能力、对分解算法和超参数敏感、在多分量或长序列时收敛慢；在紧凑度或模块性指标上略逊于某些传统方法。

---

## 448. An Ubuntu-Guided Large Language Model Framework for Cognitive Behavioral Mental Health Dialogue

**arXiv ID:** 2601.06875 | [PDF](https://arxiv.org/pdf/2601.06875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 449. Resilience by Design: A KPI for Heavy-Duty Megawatt Charging

**arXiv ID:** 2601.06898 | [PDF](https://arxiv.org/pdf/2601.06898v1)

**作者:** Sonia Yeh `[一作]` (Chalmers University of Technology), Luka de Koe `[通讯]` (Cenex)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种针对大容量充电站的无负荷类型的韧性关键绩效指标（Resilience KPI），通过整合现有 DATEX II 数据和运营日志实现对可靠性、稳健性和恢复能力的统一评估。

**💡 创新点**

创新点在于将多维度指标（可用性、负荷容错、通信连通、价格波动、排队等待等）归一化为 0–100 分，并提供可拆分的按负荷类型（电网、ICT、热力、洪水、现场事件）诊断，形成单一、可比的韧性评分。

**🔧 技术方法**

技术实现包括使用 DATEX II TablePublication/StatusPublication、OCPP/ISO 15118 日志、SCADA/EMS 接口、证书与网络健康监测，利用加权组合、线性归一化和敏感性分析构建复合指标。

**📊 数据集**

数据集来自多家试点充电站的运营记录，涵盖电网状态、充电机状态、通信中断、价格时间序列及安全事件日志，所有数据均以 UTC 时间戳和统一 ID 标识。

**📈 对比分析**

比较方法基于 0–100 标度的横向基准，计算不同供应商/地区站点的 SRS 并绘制雷达图；实验表明指标能在不同故障场景下良好区分，并能捕捉备份电源、BESS 和安全补丁对韧性的提升。

**⚠️ 局限性**

局限性包括对缺失数据的补全假设、对高级电网或 BESS 配置的依赖、实时混合能源比例测量不足；此外，指标权重仍需专家共识，可能影响跨行业的可比性。

---

## 450. Unsupervised Domain Adaptation with SAM-RefiSeR for Enhanced Brain Tumor Segmentation

**arXiv ID:** 2601.06882 | [PDF](https://arxiv.org/pdf/2601.06882v1)

**作者:** Dillan Imans `[一作]` (Sungkyunkwan University), Hyunseung Choo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 4390 | [OpenAlex ID](https://openalex.org/A5054933494)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于SAM的两阶段无监督域适应框架 SAM-RefiSeR，用于解决脑肿瘤MRI分割中的域迁移问题。

**💡 创新点**

创新点在于：①通过低频FFT替换与域对抗训练同时降低源‑目标差异；②利用SAM对伪标签进行精细化和置信/形态筛选，有效抑制伪标签噪声扩散。

**🔧 技术方法**

使用了FFT频域自适应、梯度反转层（GRL）域对抗训练、EMA学生‑教师循环、SAM分割模型以及基于置信度和形态的一致性阈值。

**📊 数据集**

实验数据集为 BraTS 2021 多模态MRI（T1CE、T2、FLAIR、T1），对不同模态作为源域进行迁移评估。

**📈 对比分析**

与 SegResNet、DAR-UNET、DAFormer、MIC、HRDA 等现有 UDA 基线比较，SAM-RefiSeR 在 Dice 方面提升约10–15%，HD95 降低显著，尤其在 T1CE→T2 等严重域移场景中优势最为突出。

**⚠️ 局限性**

局限性包括：仅在单一 BraTS 数据集上验证，未充分评估极端扫描器/协议差异的鲁棒性；SAM 在 MRI 上的迁移效果仍受域差距影响，需要进一步改进。

---

## 451. Personality-Aware Reinforcement Learning for Persuasive Dialogue with LLM-Driven Simulation

**arXiv ID:** 2601.06877 | [PDF](https://arxiv.org/pdf/2601.06877v1)

**作者:** Donghuo Zeng `[一作]` (KDDI Research), Kazushi Ikeda `[通讯]` (KDDI Research)

**通讯引用:** 3064 | [OpenAlex ID](https://openalex.org/A5062709079)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向个性化的说服对话系统框架，结合策略导向的交互控制、动态用户人格估计与基于D3QN的强化学习策略优化，并通过LLM模拟扩充对话数据。

**💡 创新点**

创新点：① 在每一轮对话中动态估计并注入81维人格向量，使策略选择随用户心理状态实时调整；② 利用LLM驱动的议程式模拟器生成多样化、可控的对话轨迹；③ 在奖励函数中加入“改变主意”惩罚项，减少同意后撤销的情况，提升长期说服效果。

**🔧 技术方法**

核心技术：Agenda‑based 交互框架 + Maximal Marginal Relevance (MMR) 检索生成回应；81维混合类型人格嵌入（连续 + 8维分类压缩）；Dueling Double DQN (D3QN) 强化学习；奖励预测网络（同意、捐赠、改变主意三项回归器）；LLM（Mistral‑7B）用户模拟器；DeBERTa‑v3 语义分类器。

**📊 数据集**

使用 PersuasionForGood (P4G) 数据集（1017 条对话，300 条标注），并通过上述框架生成 1000 条模拟对话进行训练，评估时采集 240 条对话。

**📈 对比分析**

比较方法：与无人格注入、无改变主意惩罚、策略层 vs 语句层同意奖励等六种 RL 变体对比；评估指标包括奖励预测的 MAE/RMSE/R²、人格向量的 CCA 相关系数以及累计说服奖励（同意、捐赠、改变主意）。实验表明：① 加入 turn‑level 人格提升累计说服奖励约 12–18%；② 加入改变主意惩罚将累计撤销奖励下降 30% 并略微提升捐赠额；③ 细粒度同意奖励（策略层）在结合人格时更易获得高奖励。

**⚠️ 局限性**

局限性：① MMR 检索仅受限于 P4G 数据集，难以迁移到更大规模或不同语境；② LLM 用户模拟可能产生偏差或不现实回应，影响策略泛化；③ 评价完全基于预测奖励与模型估计，未结合真实用户反馈，可能导致奖励误导；④ 隐私与伦理风险尚未通过人类审查验证。

---

## 452. Applying Embedding-Based Retrieval to Airbnb Search

**arXiv ID:** 2601.06873 | [PDF](https://arxiv.org/pdf/2601.06873v1)

**作者:** Mustafa Abdool `[一作]` (Airbnb, Inc.), Sanjeev Katariya `[通讯]` (Airbnb, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在Airbnb搜索中实现并评估了基于嵌入的检索系统，以提升候选房源质量和召回率。

**💡 创新点**

创新点在于基于多阶段旅程的trip-based负样本采样、利用第一阶段模型作为离线评估基准的流量重放框架，以及在实时更新环境下选用IVF而非HNSW的ANN设计。

**🔧 技术方法**

使用了两塔深度网络进行嵌入学习、对比学习和对数损失，结合FAISS的IVF近似最近邻索引，以及日常离线批处理生成列表嵌入。

**📊 数据集**

使用Airbnb公开日志和实时生产流量进行训练与评估，训练数据来源于用户多阶段搜索和行为（点击、保存、查看）生成的正负样本。

**📈 对比分析**

通过离线回放评估和在线A/B实验比较，V3版召回率从基线提升约90%，在线预订转化提升0.31%，并实现16%计算资源降低。

**⚠️ 局限性**

局限性包括对实时可用性和价格的离线预计算仍需近实时更新，模型仍受限于静态列表特征，且在大规模分布式环境中需要进一步优化更新吞吐量和查询过滤的兼容性。

---

## 453. qAttCNN - Self Attention Mechanism for Video QoE Prediction in Encrypted Traffic

**arXiv ID:** 2601.06862 | [PDF](https://arxiv.org/pdf/2601.06862v1)

**作者:** Michael Sidorov `[一作]`, Ofer Hadar `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种利用自注意力机制和CNN混合模型qAttCNN，在加密视频流中仅基于数据包大小特征预测用户体验QoE（BRISQUE和FPS）。

**💡 创新点**

创新点在于将1D数据包大小序列先映射为2D矩阵，再通过多头自注意力提取全局关联，随后用CNN提取局部特征，构建端到端无参考QoE预测体系；同时结合FFT转换、迁移学习、周期学习率与Dropout等技术提升鲁棒性。

**🔧 技术方法**

使用技术包括：1D→2D嵌入层、FFT频域转换、35头多头自注意力、ResNet18/34/50作为CNN头、迁移学习（ImageNet预训练）、周期学习率调度、Dropout及加噪与行打乱的数据增强、MAEP自定义损失。

**📊 数据集**

数据集：在WhatsApp视频通话环境下采集的51,341条样本，包含350个1毫秒间隔内的数据包大小特征，用于训练和10折交叉验证。

**📈 对比分析**

通过10折交叉验证与MAEP指标与传统ML（LR、SVM、RF等）及之前的QoENet1D进行对比；qAttCNN在BRISQUE上实现2.14% MAEP（最佳），FPS上实现9.10% MAEP，显著优于所有对比模型。

**⚠️ 局限性**

限制：模型训练耗时长，需要大量样本；在单一平台（WhatsApp）上验证，泛化能力未知；对不同头网络的进一步探索与跨平台测试仍待开展。

---

## 454. Code Evolution for Control: Synthesizing Policies via LLM-Driven Evolutionary Search

**arXiv ID:** 2601.06845 | [PDF](https://arxiv.org/pdf/2601.06845v1)

**作者:** Ping Guo `[一作]` (City University of Hong Kong), Chaoning Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1936 | [OpenAlex ID](https://openalex.org/A5057230698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了一种基于大语言模型（LLM）的进化搜索框架，专门用于合成可解释的控制策略代码，并在LunarLander环境中验证了其效果。

**💡 创新点**

创新点在于：①将控制策略的生成问题转化为代码进化问题；②利用LLM既做变异/交叉，又在每一次进化中接收丰富的执行反馈（失败模式、奖励分布等），显著提升搜索效率；③通过对比多种进化策略，展示了上下文驱动的LLM进化能在保持代码可读性的同时获得高成功率。

**🔧 技术方法**

技术手段包括：LLM驱动的进化算子（FunSearch、EoH、EvoEngineer），Gymnasium 的 LunarLander‑v3 环境，基于 PPO 的强化学习基线，对候选策略进行沙盒执行、奖励评估、代码行数与复杂度统计等。

**📊 数据集**

使用的数据集为 Gymnasium 提供的 LunarLander‑v3 仿真环境（8 维连续状态、4 个离散动作）。

**📈 对比分析**

通过与随机策略、PPO（1M 步）以及三种 LLM 进化策略的对比，实验显示：在约 45 次 LLM 调用下，EvoEngineer 能获得 66.6 的平均奖励和 40% 的成功率；扩展到 200 次调用（约 1M 环境步）后，EvoEngineer+ 达到 143.6 的平均奖励和 70% 的成功率，成功率超过 PPO 的 60%，但平均奖励略低于 PPO 的 214。

**⚠️ 局限性**

局限性：①仅在离散动作、单一任务（LunarLander）上验证，缺乏对连续动作或更复杂任务的泛化评估；②依赖 LLM 的偏差和能力，结果可能随模型不同而变化；③进化产生的策略方差大，可能出现极端表现；④LLM 调用成本仍高，特别是大规模调用时；⑤未结合梯度微调或混合方法进一步提升性能。

---

## 455. Efficient Subdivision of Bézier Curves/Surfaces via Blossoms

**arXiv ID:** 2601.06841 | [PDF](https://arxiv.org/pdf/2601.06841v1)

**作者:** Krassimira Vlachkova `[一作]` `[通讯]` (Sofia University St. Kliment Ohridski), Krassimira Vlachkova (Sofia University St. Kliment Ohridski)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文研究了使用Blossoming技术对Bézier曲线、张量积曲面与三角形曲面进行细分，并给出了计算控制点的闭式公式；

**💡 创新点**

创新点在于将传统Blossoming求和化为可直接实现的闭式表达式，显著简化了控制点求解；

**🔧 技术方法**

采用Blossoming理论、组合数学与多项式解析的技术实现公式推导；

**📊 数据集**

以三维多项式曲面S(u,v)=∑c_{ij}u^iv^j（n=3,m=2）为实验数据集；

**📈 对比分析**

通过Mathematica实现验证，实验结果显示闭式公式在计算效率和数值精度上优于传统逐项求和方法；

**⚠️ 局限性**

局限在于仅适用于多项式曲线/曲面，未考虑非多项式情况，也未提供批量计算的最优实现。

---

## 456. PRISM: Color-Stratified Point Cloud Sampling

**arXiv ID:** 2601.06839 | [PDF](https://arxiv.org/pdf/2601.06839v1)

**作者:** Hansol Lim `[一作]` (State University of New York), Jongseong Brad Choi `[通讯]` (State University of New York)

**通讯引用:** 469 | [OpenAlex ID](https://openalex.org/A5061176269)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了PRISM颜色引导分层采样方法，用于RGB‑LiDAR点云降采样，保留色彩多样的纹理区域，减少色彩单一的冗余点。

**💡 创新点**

创新点在于把RGB颜色空间视为分层域，并对每个颜色箱设定最大采样容量，按色彩多样性而非空间均匀性分配采样点，实现在保持色彩完整性的同时实现可预测的压缩比例。

**🔧 技术方法**

采用颜色量化、按颜色箱分层、全局k选择（利用分段线性求解）以及随机采样策略，并通过Chamfer、Hausdorff距离和色彩熵等指标评估降采样质量。

**📊 数据集**

使用了三大公开RGB‑LiDAR数据集：Toronto‑3D、ETH3D 和 Paris‑CARLA，覆盖城市户外、建筑结构与合成纹理场景。

**📈 对比分析**

与随机采样、体素网格和法向采样对比，PRISM在约1%目标压缩率下实现了色彩熵最高、压缩比例最接近目标且运行时间与随机相当，几何误差略高但可接受。

**⚠️ 局限性**

局限性在于牺牲空间均匀性导致Chamfer距离增大，对光照变化敏感，并且在高色彩多样场景下可能过度保留细节导致空间稀疏，缺乏对几何精度的完整维护。

---

## 457. OSCAR: Optical-aware Semantic Control for Aleatoric Refinement in Sar-to-Optical Translation

**arXiv ID:** 2601.06835 | [PDF](https://arxiv.org/pdf/2601.06835v1)

**作者:** Hyunseo Lee `[一作]` (Kyungpook National University), Woo-Jeoung Nam `[通讯]` (Kyungpook National University)

**通讯引用:** 218 | [OpenAlex ID](https://openalex.org/A5060931534)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种基于光学对齐语义控制的 SAR‑至光学图像翻译框架 OSCAR，能够在噪声多、结构不确定的雷达图像中生成语义一致、结构精确且光学真实的图像。

**💡 创新点**

创新点包括：① 通过跨模态知识蒸馏构建光学对齐 SAR 编码器；② 在 ControlNet 中同时使用多层视觉提示与文本提示实现全局与局部语义对齐；③ 引入不确定性目标自适应权重，抑制雷达噪声诱发的高频伪影。

**🔧 技术方法**

采用 DINOv3‑SAT 与 LoRA 的跨模态蒸馏、语义化 ControlNet、潜在扩散模型（LDM）以及基于高斯似然的 Uncertainty‑aware 目标。

**📊 数据集**

使用 BigEarthNet‑v2 与 SEN12MS 两大遥感数据集，并结合 CORINE 语义标签进行训练与评估。

**📈 对比分析**

与 GAN（CycleGAN、StegoGAN）及 Diffusion（ControlNet、BBDM、cBBDM）基线对比，OSCAR 在 FID、LPIPS、SSIM、SAM 等指标上均显著领先；在 BigEarthNet‑v2 上 FID 降低 32.5%，在 SEN12MS 上降低 50.2%。

**⚠️ 局限性**

模型体积大、推理速度慢；当前仅利用视觉模态，未充分利用时空元数据与像素级语义标签，导致在极端季节或地理变异下仍有潜在鲁棒性不足。

---

## 458. Enhancing Low-resolution Image Representation Through Normalizing Flows

**arXiv ID:** 2601.06834 | [PDF](https://arxiv.org/pdf/2601.06834v1)

**作者:** Chenglong Bao `[一作]` (Yau Mathematical Sciences Center, Tsinghua University, Beijing Institute of Mathematical Sciences and Applications), Yihang Zou `[通讯]` (Yau Mathematical Sciences Center, Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为LR2Flow的低分辨率图像表示框架，利用小波紧帧与可逆流网络共同学习降采样与升采样算子，实现高质量的图像重构。

**💡 创新点**

创新点包括：① 在小波紧帧域构建可逆神经网络，利用可逆性与分布估计显著提升高频信息恢复；② 对重构误差进行理论分析，证明冗余小波紧帧优于正交基；③ 将该框架统一应用于图像重采样、压缩和去噪，展示其跨任务优势。

**🔧 技术方法**

主要技术：小波紧帧分解（B-spline基）、正则化的正交卷积、可逆层（Affine Coupling 或 iResBlock）、ActNorm、可逆1×1卷积、正则化流网络（Normalizing Flow）以及分布匹配损失。

**📊 数据集**

使用的公开数据集包括DIV2K、Set5、Set14、BSD100、Urban100、CBSD68、Kodak24、McMaster、Flickr2K、Waterloo等，训练时随机裁剪并做几何增强。

**📈 对比分析**

通过与传统SR、编码-解码、流基重采样模型以及JPEG压缩后处理方法比较，LR2Flow在多种任务上实现了PSNR/SSIM领先（如×2重采样PSNR>47dB，×4重采样PSNR>37dB，压缩QF30下PSNR>36dB，去噪σ=50下PSNR>35dB），并在视觉效果上显著减少模糊与伪影。

**⚠️ 局限性**

局限性：① 在压缩任务中未对码率进行显式约束，依赖后续采样恢复；② 需要较高计算资源（GPU）和存储；③ 仍有提升空间，如加入熵编码、进一步优化流网络容量与效率。

---

## 459. SARA: Scene-Aware Reconstruction Accelerator

**arXiv ID:** 2601.06831 | [PDF](https://arxiv.org/pdf/2601.06831v1)

**作者:** Jee Won Lee `[一作]` (State University of New York), Jongseong Brad Choi `[通讯]` (State University of New York)

**通讯引用:** 469 | [OpenAlex ID](https://openalex.org/A5061176269)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出SARA模块，利用几何驱动的重叠×视差评分和信息加权生成树来进行SfM视图对选择，显著减少匹配对数并提升重建精度。

**💡 创新点**

创新点在于将视图对选择从视觉相似性转为几何信息（重叠×视差）预评分，构建IWST并增补循环、长基线和弱视强化边，实现高效、精确的匹配筛选。

**🔧 技术方法**

使用全局描述子DINOv2进行kNN检索，互相最近邻+短RANSAC计算重叠与视差，生成信息权重；再用信息加权生成树与边增补；结合COLMAP、SuperGlue/LightGlue等学习匹配器进行实验。

**📊 数据集**

主要在Mip-NeRF 360数据集（5个室内外场景）上进行评估，并在此数据集上验证了SARA在不同检测器与下游渲染器（3DGS、SVRaster）上的性能。

**📈 对比分析**

与全量匹配和词袋树检索对比，SARA在所有现代检测器下实现100%图像注册率，旋转误差下降约46.5%、平移误差下降约12.5%，匹配对数从30,848降至580，匹配时间可达50×加速，重建质量与全量匹配基本相当。

**⚠️ 局限性**

受限于需要高质量特征；无法修复低质量检测器导致的匹配失败；全局描述子在重复或极端结构场景中可能失效；目前仅验证离线SfM，实时SLAM适配仍待研究。

---

## 460. PDR: A Plug-and-Play Positional Decay Framework for LLM Pre-training Data Detection

**arXiv ID:** 2601.06827 | [PDF](https://arxiv.org/pdf/2601.06827v1)

**作者:** Jinhan Liu `[一作]`, Dandan Guo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种无训练、可插拔的权重衰减方法 PDR，用于加强大语言模型的训练数据检测。

**💡 创新点**

创新点在于揭示记忆信号在序列早期更强的现象，并通过单调递减的权重对已有基于似然的检测方法进行改进，显著提升性能。

**🔧 技术方法**

技术主要包括基于信息理论的 token‑entropy 观察、可插拔的权重衰减函数（线性、指数、多项式）以及对现有 Likelihood‑based 分数（Loss、Min‑k%、Min‑k%++ 等）的加权重构。

**📊 数据集**

使用的数据集为 WikiMIA（维基百科）和 MIMIR（Pile 数据集子集），并在多种 LLM（Pythia、LLaMA、GPT‑NeoX、OPT、Mamba）上进行评估。

**📈 对比分析**

通过与 Loss、Ref、Neighbor、FSD 等基线对比，并使用 AUROC 评估，PDR 在 WikiMIA 上可提升 Min‑k%++ 达 4.7 点 AUROC，整体均能提升 1–5 点，且对不同模型和长度均保持稳健。

**⚠️ 局限性**

局限性包括：主要在英文数据上验证，黑盒场景下信息有限；对高度同质化的数据集（如 MIMIR）仍难以获得显著优势；以及不同语言、tokenization granularity 可能影响权重衰减效果。

---

## 461. SpatialNav: Leveraging Spatial Scene Graphs for Zero-Shot Vision-and-Language Navigation

**arXiv ID:** 2601.06806 | [PDF](https://arxiv.org/pdf/2601.06806v1)

**作者:** Jiwen Zhang `[一作]` (Fudan University), Qi Wu `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了在预探索环境后构建空间场景图并利用该图实现零样本视觉语言导航的完整框架。

**💡 创新点**

创新点在于：①引入零样本VLN的新设置允许预探索并构建空间场景图；②设计了基于agent-centric spatial map、compass-like视觉表示和远程物体定位的SpatialNav方法。

**🔧 技术方法**

使用技术包括：SLAM点云重建、层次化空间场景图构建（楼层/房间/物体分割与语义标注）、多模态大型语言模型（GPT‑5.1）、Compass-style 视图合成、远程物体查询与决策。

**📊 数据集**

使用数据集包括 Matterport3D 基础的 R2R、REVERIE、R2R‑CE 与 RxR‑CE，进行离散与连续环境评估。

**📈 对比分析**

与现有零样本和监督学习基准对比，SpatialNav 在四大验证集上取得 57.7%–64.0% 的成功率，明显优于其他零样本方法并逼近甚至超过部分监督学习方法。

**⚠️ 局限性**

主要局限：依赖完整的 SLAM 预探索点云与空间场景图构建；未验证在无预探索或重建失效情况下的鲁棒性；构造图的计算成本高；对房间分割的自动化仍存在误差。

---

## 462. MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation

**arXiv ID:** 2601.06883 | [PDF](https://arxiv.org/pdf/2601.06883v1)

**作者:** Xinhang Liu `[一作]` (Northwestern Polytechnical University), Yuchao Dai `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 12239 | [OpenAlex ID](https://openalex.org/A5036202579)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级网络 MixRI，用极少的参考图像即可在 RGB 图像中实现 CAD 基础的创新物体姿态估计。

**💡 创新点**

核心创新包括：① 通过多视角信息聚合的 View‑Aggregated Point Matching 模块，将多幅参考图像特征在注意力空间中融合；② 双注意力 Dual‑Attention 机制（Self‑Attention + Cross‑Attention）自动对不同视角、点与帧的相似度进行加权，提升匹配鲁棒性；③ 在网络输出中直接预测遮挡标记，自动处理遮挡。

**🔧 技术方法**

技术手段：ResNet‑style 编码器、旋转不变特征提取、双注意力特征混合、3D 卷积成本体积、空间 Soft‑Argmax、RANSAC‑based SQ‑PnP、Huber 与 BCE 损失。

**📊 数据集**

训练使用 GSO‑Dataset 生成的合成图像，评估采用 BOP 公开的七个核心数据集（LM‑O、YCB‑V、T‑LESS、TUD‑L、IC‑BIN、HomebrewedDB、ITODD）。

**📈 对比分析**

与 OSOP、MegaPose、ZS6D、GigaPose、Genflow、FoundPose 等现有方法对比，MixRI 在不使用重投影或特征缓存的前提下，使用 33 倍更少的参考图像、参数量 50 倍更小，却能获得与最先进方法相当甚至更优的平均召回率（AR），且推理速度更快、内存占用更低。

**⚠️ 局限性**

局限性：在灰度图像、极度遮挡、强反射或低纹理表面（如 ITODD 数据集）上的表现不如某些基于深度或多模态的方案；当参考图像数量骤增时，轻量网络会出现融合瓶颈，导致性能下降。

---

## 463. Seeing through the Conflict: Transparent Knowledge Conflict Handling in Retrieval-Augmented Generation

**arXiv ID:** 2601.06842 | [PDF](https://arxiv.org/pdf/2601.06842v1)

**作者:** Hua Ye `[一作]` (Nanjing University), Fei Shen `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TCR 框架，用于在 Retrieval‑Augmented Generation（RAG）中检测并解决内部知识与外部检索内容之间的冲突。

**💡 创新点**

创新点在于将语义匹配与事实一致性解耦，利用对比学习训练双编码器并估计自答能力，通过软提示和 SNR 动态加权三种可解释信号实现冲突自适应控制。

**🔧 技术方法**

使用的技术包括双编码器对比学习、soft prompt tuning、信号投影 MLP、SNR 基础的动态权重以及轻量级软提示融合。

**📊 数据集**

实验数据集涵盖 Wikidata‑Conflict‑5K、ConflictTQA、ConflictPQA、KILT、ConflictBank 2024 以及单文档 NQ 上下文等多种知识冲突与 QA 场景。

**📈 对比分析**

与 Prompt、KAFT、IRCAN、RAAT、Parenting、Astute RAG、InstructRAG 等基线相比，TCR 在冲突检测 F1/AUROC 提升 5–18 分、知识缺口恢复 +21.4pp、误导上下文覆盖率降低 29.3pp，且仅增加 0.3% 参数、保持 94% 解码速度。

**⚠️ 局限性**

局限性包括对检索质量仍有依赖、对极端噪声场景鲁棒性有限、以及在多模态或跨语言情境下的适用性尚待进一步验证。

---

## 464. CHASE: LLM Agents for Dissecting Malicious PyPI Packages

**arXiv ID:** 2601.06838 | [PDF](https://arxiv.org/pdf/2601.06838v1)

**作者:** Takaaki Toda `[一作]` (Waseda University), Tatsuya Mori `[通讯]` (Waseda University)

**通讯引用:** 4374 | [OpenAlex ID](https://openalex.org/A5064493291)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了多代理架构 CHASE，用于自动检测 PyPI 包中的恶意代码并生成可操作的安全报告。

**💡 创新点**

通过监督-工作者模型、计划-执行工作流以及与确定性安全工具的集成，补偿 LLM 的幻觉与上下文混淆问题，实现高可靠性。

**🔧 技术方法**

使用本地 LLM（Qwen3:32B 作为监督者、Qwen3:8B 作为工作者、Gemma3:4B 进行结构化输出）以及专用工具（去混淆、网络检索、威胁情报查询等）。

**📊 数据集**

使用 3,000 个真实包的数据集（500 份恶意、2,500 份良性），覆盖 setup.py、__init__.py 等主要入口文件。

**📈 对比分析**

与 MalGuard（RF）和 GuardDog 进行对比，CHASE 在单轮下召回率 98.4%、误报率 0.08%，平均分析时长 4.5 分钟，显著优于基线。

**⚠️ 局限性**

仅分析 setup.py、__init__.py，无法覆盖所有包组件；工具集有限，可能无法检测所有高级混淆与多阶段攻击手段。

---

## 465. Generative Modeling of Human-Computer Interfaces with Diffusion Processes and Conditional Control

**arXiv ID:** 2601.06823 | [PDF](https://arxiv.org/pdf/2601.06823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 466. Optimal Rate Region for Multi-server Secure Aggregation with User Collusion

**arXiv ID:** 2601.06836 | [PDF](https://arxiv.org/pdf/2601.06836v1)

**作者:** Zhou Li `[一作]` (Guangxi University), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究多服务器安全聚合中用户协同攻击下的信息论极限，提出完整的最优速率区域；

**💡 创新点**

创新点在于首次将多服务器架构与用户协同结合，得到最小通信、密钥和源密钥率的闭式表达；

**🔧 技术方法**

采用线性密钥构造与信息论熵不等式证明可实现性与下界；

**📊 数据集**

未使用具体数据集，属于理论分析；

**📈 对比分析**

通过信息论极限证明实现方案与下界匹配，性能达到理论最优；

**⚠️ 局限性**

局限在于仅考虑完全连通的两跳网络与诚实但好奇模型，未讨论动态用户加入或部分连通的实际场景。

---

## 467. SPINE Gripper: A Twisted Underactuated Mechanism-based Passive Mode-Transition Gripper

**arXiv ID:** 2601.06833 | [PDF](https://arxiv.org/pdf/2601.06833v1)

**作者:** JaeHyung Jang `[一作]` (Korea Advanced Institute of Science and Technology), Jee-Hwan Ryu `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 4697 | [OpenAlex ID](https://openalex.org/A5019438521)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一款单驱动、被动式多功能抓取器（SPINE gripper），实现稳定抓取与双向手内旋转，全部通过机械编码的扭矩阈值实现模式切换，无需传感器或控制器。

**💡 创新点**

创新点主要有：①采用Twisted Underactuated Mechanism (TUM) 生成非共面双自由度运动，①+②实现方向不变的收缩与旋转；②通过摩擦产生的扭矩阈值实现被动模式切换；③实现了单一驱动的双向旋转与可调抓取力，完全摆脱电气驱动与闭环控制。

**🔧 技术方法**

关键技术包括：Twisted Underactuated Mechanism (TUM) 的几何与弹性建模；滑动-曲柄连杆实现抓取闭合；摩擦发生器（O‑ring）设定扭矩阈值；力学平衡与Jacobian分析；实验验证与与机器人手臂的腕旋转耦合。

**📊 数据集**

未使用公开数据集，全部以实验测试为主：不同摩擦间隙的O‑ring、门把手、螺栓、厨房用具、果实等多种物体进行抓取与旋转实验。

**📈 对比分析**

通过抓取成功率、扭矩-抓取力曲线、旋转速度与稳定性等指标与传统主动式抓手进行对比。结果表明：在满足摩擦阈值的条件下抓取成功率接近100%，抓取力可在1–5 N范围内可调；在无电机、无电缆的前提下实现连续双向旋转，节省功耗且结构更可靠。

**⚠️ 局限性**

局限性：①需要足够的摩擦阈值才能实现稳定抓取；②对物体尺寸与形状有一定限制；③抓取力只能通过摩擦间隙粗调，无法实现精细调节；④单一驱动在面对高速动态或极端环境时的适用性有限；⑤长时间使用可能出现弹性材料疲劳或摩擦磨损。

---

## 468. Spectral Shadows: When Communication Complexity Meets Linear Invariance Testing

**arXiv ID:** 2601.06828 | [PDF](https://arxiv.org/pdf/2601.06828v1)

**作者:** Swarnalipa Datta `[一作]` (Indian Statistical Institute), Manmatha Roy `[通讯]` (Indian Statistical Institute)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5003989340)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究在通信复杂度模型下的线性同构测试问题，给出了确定性与私有随机模型下的通信协议并证明了其近似谱范数依赖的上、下界；

**💡 创新点**

首次将逼近谱范数作为该问题的关键复杂度度量，并构造对应的协议，证明该度量在确定性与私有随机情况下均不可避免；

**🔧 技术方法**

采用傅里叶分析、Junta 定理、线性变换与支持集抽样等技术，构造逼近函数并分析其距离与支持；

**📊 数据集**

无实验数据集，全部为理论分析与证明；

**📈 对比分析**

与已知的等价与不等价通信复杂度对比，证明确定性协议需 Ω(t²) 位、私有随机协议需 Ω(log t) 位，且给出的协议几乎匹配这些下界；

**⚠️ 局限性**

仅适用于逼近谱范数小的函数，无法覆盖所有函数；未解决容差版本的随机协议；上界与下界在私有随机模型中仍存在多项式阶差距。

---

## 469. BiasLab: A Multilingual, Dual-Framing Framework for Robust Measurement of Output-Level Bias in Large Language Models

**arXiv ID:** 2601.06861 | [PDF](https://arxiv.org/pdf/2601.06861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 470. WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport

**arXiv ID:** 2601.06810 | [PDF](https://arxiv.org/pdf/2601.06810v1)

**作者:** Qiangwei Peng `[一作]` (Peking University), Peijie Zhou `[通讯]` (Peking University)

**通讯引用:** 1069 | [OpenAlex ID](https://openalex.org/A5072894189)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种 WFR‑FM（Wasserstein–Fisher–Rao Flow Matching）方法，用于在未平衡分布下的动态最优传输问题，无需 ODE 仿真，能够同时学习位移向量场和生长速率，实现单细胞转录组时间演化轨迹推断。

**💡 创新点**

创新点在于将流匹配与 WFR 结合，联合回归位移向量场和增长速率，并给出理论证明其等价于 WFR geodesics，解决了先前方法只回归速度或需 ODE 后处理的局限；同时提出了条件高斯路径和 mini‑batch WFR‑OET 的无仿真训练流程。

**🔧 技术方法**

技术上使用了无仿真流匹配（Conditional Flow Matching）、WFR 动态最优传输、条件高斯测度路径、离散多时点 WFR 问题、以及 mini‑batch WFR‑OET 及 Sinkhorn 迭代求解。

**📊 数据集**

数据集包括合成数据：Simulation Gene、Dygen、1000D Gaussian 以及真实单细胞 RNA‑seq 数据：EB（embryoid bodies）、EMT、CITE‑seq、鼠骨髓。

**📈 对比分析**

与多种基线（MMFM、Metric FM、SF2M、MIOFlow、TIGON、DeepRUOT、Var‑RUOT、UOT‑FM、VGFM、Action Matching）进行比较，WFR‑FM 在分布匹配、动作值、插值精度、计算效率和生长率估计等方面均优于或匹配最强方法。

**⚠️ 局限性**

局限性：仍需求解静态 OT 问题，尤其在极大数据集上成本较高；对噪声鲁棒性尚未深入；仅针对可解析 Dirac‑to‑Dirac 路径的 WFR，推广到其他未平衡指标仍需研究。

---

## 471. Forest Before Trees: Latent Superposition for Efficient Visual Reasoning

**arXiv ID:** 2601.06803 | [PDF](https://arxiv.org/pdf/2601.06803v1)

**作者:** Yubo Wang `[一作]` (Fudan University), Yuhan Liu `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在视觉语言模型中提出 Laser 方法，通过在潜在空间进行动态窗口对齐学习，实现连续隐式推理。

**💡 创新点**

创新点在于将显式的链式思考替换为“森林先于树木”的潜在叠加推理，并采用动态窗口对齐与自我精炼叠加来避免早期语义崩溃。

**🔧 技术方法**

使用 Qwen2.5‑VL‑7B 作为骨干，结合动态窗口对齐（DWAL）、自我精炼叠加和熵正则干预的技术。

**📊 数据集**

构造 ScanPath 数据集（270k 语义序列）以及 6 个视觉推理基准：MMVP、BLINK、SEED‑Bench‑2‑Plus、MMStar、HallusionBench、HRBench。

**📈 对比分析**

与零样本 VLM、工具增强及 RL 增强方法和现有潜在推理基线对比，Laser 在 6 个基准上平均提升 5.03%，在 HallusionBench 最高提升 11.36%，并将推理 token 数量降至 97% 以内。

**⚠️ 局限性**

局限性在于对精细像素级定位和拼图类任务的精度略低，且依赖大量弱监督的语义序列生成，可能对极端细粒度任务效果不足。

---

## 472. Explainable Multimodal Aspect-Based Sentiment Analysis with Dependency-guided Large Language Model

**arXiv ID:** 2601.06848 | [PDF](https://arxiv.org/pdf/2601.06848v1)

**作者:** Zhongzheng Wang `[一作]` (Harbin Institute of Technology), Yan Song `[通讯]` (University of Science and Technology of China)

**通讯引用:** 29921 | [OpenAlex ID](https://openalex.org/A5013100135)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于多模态大语言模型的生成式可解释多模态方面情感分析框架，能够同时输出情感标签和自然语言解释。

**💡 创新点**

创新点是将依赖句法结构引导的方面中心剪枝作为Prompt输入，结合生成式LLM实现情感分类与解释的一体化，并通过构造带解释的训练集提升可解释性。

**🔧 技术方法**

技术包括多模态LLM（Qwen3‑VL、Ministral‑3）、依赖句法树构建与剪枝、文本化依赖信息作为Prompt、LoRA微调、生成式推理。

**📊 数据集**

使用Twitter2015和Twitter2017两大多模态ABSA基准，并通过Qwen3‑VL‑32B生成并人工审核的情感解释扩充数据。

**📈 对比分析**

与多种基线（非prompt、prompt式、传统MABSA模型）对比，在准确率和宏F1上均提升约2–4%（例如Qwen3‑VL‑8B n=2 达到77.2%准确率、73.0%宏F1），生成解释在BLEU/ROUGE/BERTScore也有明显提升。

**⚠️ 局限性**

局限包括对LLM的计算成本高、依赖句法剪枝参数需手工调优、解释质量受生成模型可控性限制，以及在极端多方面或零样本场景下仍存在误差。

---

## 473. ET-Agent: Incentivizing Effective Tool-Integrated Reasoning Agent via Behavior Calibration

**arXiv ID:** 2601.06860 | [PDF](https://arxiv.org/pdf/2601.06860v1)

**作者:** Yifei Chen `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3838 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段训练框架（自进化数据飞轮 + 行为校准训练）来校准LLM代理在工具集成推理（TIR）中的行为模式。

**💡 创新点**

创新点包括：①自进化数据飞轮可迭代生成多样化轨迹；②基于Pareto前沿的分组采样与课程式RL结合实现行为校准；③多目标奖励设计鼓励格式、正确性、效率与推理简洁度同步提升。

**🔧 技术方法**

采用自监督微调（RFT）、基于ARPO的强化学习、Pareto前沿采样、课程学习、强化奖励机制、LLM自演化提示、e5编码+ t‑SNE可视化等技术。

**📊 数据集**

训练数据来源于 Wikipedia；评测数据包含数学推理任务 AIME24、AMC23、MATH500 与知识密集任务 2WikiMultiHopQA、Bamboogle、MuSiQue。

**📈 对比分析**

与直接推理、单一TIR 及多TIR 基线进行对比，实验显示本方法在正确率与效率两项指标上均位列前列（例如平均正确率 60.1、效率 46.0）。

**⚠️ 局限性**

局限性在于仅使用本地 Wikipedia 检索，难以扩展至更大模型，并未加入实时网络搜索功能，实验资源受限。

---

## 474. Semilinear single-track vehicle models with distributed tyre friction dynamics

**arXiv ID:** 2601.06854 | [PDF](https://arxiv.org/pdf/2601.06854v1)

**作者:** Luigi Romano `[一作]` (Linköping University), Erik Frisk `[通讯]` (Linköping University)

**通讯引用:** 2831 | [OpenAlex ID](https://openalex.org/A5070530122)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

构建了基于分布式摩擦模型FrBD的半线性单轨车轮动力学框架，并与传统的集中式摩擦模型进行比较。

**💡 创新点**

创新点在于将非线性摩擦（Dahl、LuGre等）通过分布式偏微分方程统一描述，消除了局部粘滑区的显式区分，既保持物理一致性又具备严格的存在唯一性和全局微解性质；同时提供了线性化分析与传递函数表达，利于控制器与观测器设计。

**🔧 技术方法**

主要技术包括：分布式半线性偏微分方程建模、ODE–PDE 交互系统的状态空间表示、半线性系统的局部/全局存在唯一性证明、谱分析和传递函数推导、数值仿真与有限差分空间离散。

**📊 数据集**

论文未使用实际实验或公开数据集，全部以MATLAB/Simulink的数值仿真验证模型的微振动和转向响应特性。

**📈 对比分析**

与传统的集中式单轨模型相比，分布式模型能够捕捉低速微摆动（micro‑shimmy）和转向时的瞬态波动，仿真结果显示更贴合物理预期；在控制角速度输入下，两种模型在稳态值上一致，但灵活车轮壳模型在瞬态响应上更平滑，体现了分布式动力学的优势。

**⚠️ 局限性**

局限性包括：未考虑悬挂动力学与质量转移；模型仅在平衡点附近线性化，难以直接推广至大幅非线性轨迹；以及求解时对空间离散的依赖，计算量较传统集中模型略大。

---

## 475. MedGround: Bridging the Evidence Gap in Medical Vision-Language Models with Verified Grounding Data

**arXiv ID:** 2601.06847 | [PDF](https://arxiv.org/pdf/2601.06847v1)

**作者:** Mengmeng Zhang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Yisheng Lv `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于专家分割掩码的医学视觉语言模型（VLM）训练管道MedGround，自动生成高质量的图像-查询-框（image‑query‑box）三元组数据；

**💡 创新点**

创新点在于将精准的医学分割掩码转化为定位目标，利用掩码提取的几何和空间属性驱动VLM生成具有临床意义的指代查询，并通过多阶段格式、规则和VLM判定三重验证确保数据真实性；

**🔧 技术方法**

使用了掩码导向的提示生成、规则检测、基于VLM的视觉一致性验证以及LoRA微调；

**📊 数据集**

利用了八个公开医学分割数据集（如MS‑CXR、ChestX‑ray8、DeepLesion、LIDC‑IDRI等）构建MedGround‑35K，共35,480条三元组；

**📈 对比分析**

在医学指代定位、语义对齐、零样本泛化三个基准上，对MedGemma、Qwen、Lingshu等多种VLM进行Fine‑tune，实验显示在指代定位IoU、语义灵敏度和外域（QaTa‑COV19）零样本性能均显著提升（IoU提升约30%–60%，语义灵敏度提升至70%+）；

**⚠️ 局限性**

局限包括：依赖单一VLM判定器可能引入模型偏差；仅提供框级定位，缺乏像素级边界细节；生成查询可能带有LLM的风格偏差；数据来源受限于公开分割集，某些解剖部位或病理在数据中低频导致泛化受限。

---

## 476. MoEScore: Mixture-of-Experts-Based Text-Audio Relevance Score Prediction for Text-to-Audio System Evaluation

**arXiv ID:** 2601.06829 | [PDF](https://arxiv.org/pdf/2601.06829v1)

**作者:** Bochao Sun `[一作]`, Han Yin `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3355 | [OpenAlex ID](https://openalex.org/A5016864694)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于Mixture of Experts与Sequential Cross‑Attention的客观评估器，用于衡量文本到音频（TTA）系统的语义一致性。

**💡 创新点**

创新点在于将多种CLAP变体与SeqCoAttn专家结合，通过动态门控实现跨模态特征的多角度融合，首次在XACLE 2026挑战中以显著提升SRCC排名第一。

**🔧 技术方法**

采用Mixture of Experts框架、SeqCoAttn交叉注意力、CLAP、MGA‑CLAP、M2D‑CLAP、BEATs+RoBERTa编码器，并使用混合MSE＋对比损失进行训练。

**📊 数据集**

使用XACLE 2026官方训练集与盲测集进行实验评估。

**📈 对比分析**

与挑战基线相比，模型在SRCC、LCC、KTAU提升30.6%，MSE显著下降，整体性能优于基线并夺得冠军。

**⚠️ 局限性**

单一专家表现有限，SeqCoAttn专家在单独使用时效果较弱；仍需进一步研究专家选择机制与模型解释性。

---

## 477. Analyzing the effect of prediction accuracy on the distributionally-robust competitive ratio

**arXiv ID:** 2601.06813 | [PDF](https://arxiv.org/pdf/2601.06813v1)

**作者:** Toru Yoshinaga `[一作]` (University of Tokyo), Yasushi Kawase `[通讯]` (University of Tokyo)

**通讯引用:** 473 | [OpenAlex ID](https://openalex.org/A5025634820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在算法与预测框架下，利用分布鲁棒竞争比（DRCR）衡量预测准确度对在线算法性能的影响，并证明其对准确度的单调性与凹性；进一步推广到多层级（层次化）预测，提出对应的凸组合表达与LP求解方法；以滑雪租赁问题为例，给出计算最佳DRCR的LP模型，并通过该模型推导出实现比鲁棒性能更佳的“临界准确度”。

**💡 创新点**

1) 对DRCR关于预测准确度的结构性分析（单调、凹性）及其对多层预测的推广； 2) 把多层预测下的DRCR转化为凸组合表达； 3) 通过LP框架求解滑雪租赁问题的最优DRCR，并得到临界准确度的多项式可解算法； 4) 提供了预测准确度与性能提升之间的线性不等式表征，形成了预测质量评估的理论工具。

**🔧 技术方法**

凸分析与线性规划（LP）技术、Abel求和分部引理、分布鲁棒优化框架、层次化预测模型的数学表述、对滑雪租赁问题的离散化与有限维LP求解。

**📊 数据集**

本文以理论分析为主，并没有使用公开数据集；实验部分仅在滑雪租赁问题的不同参数设置（B,ℓ,u）下求解LP，得到DRCR-准确度曲线，可视化展示。

**📈 对比分析**

通过求解LP得到的最优DRCR与鲁棒性能（经典随机算法的竞争比）进行对比，表明在达到某一“临界准确度”后，预测能显著降低DRCR；图示显示DRCR随准确度递增且凹性。

**⚠️ 局限性**

限制主要包括：1）需要层次化预测假设，实际学习模型往往难以严格满足；2）本文仅针对可通过LP求解的在线问题，扩展到更一般问题需进一步研究；3）实验验证有限，缺乏真实数据场景验证。

---

## 478. Doing More with Less: Data Augmentation for Sudanese Dialect Automatic Speech Recognition

**arXiv ID:** 2601.06802 | [PDF](https://arxiv.org/pdf/2601.06802v1)

**作者:** Ayman Mansour `[一作]` `[通讯]` (Independent Researcher), Ayman Mansour (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对低资源的苏丹阿拉伯语方言，基于OpenAI Whisper模型进行自监督自训练与TTS数据增强的语音识别系统开发，并构建首个SUDAN方言ASR基准。

**💡 创新点**

创新点包括：①首次为苏丹方言搭建Whisper基准；②提出结合伪标签自训练与Klaam TTS生成语音的混合增强方法；③在仅使用低成本资源的前提下，实现显著性能提升。

**🔧 技术方法**

主要技术包括：Whisper预训练模型微调、伪标签自训练、TTS合成语音增强、基于置信度的伪标签过滤、HuggingFace Transformers实现及参数高效训练。

**📊 数据集**

使用的数据集有：苏丹方言语音集（≈4 h，3,544条录音）、Arabic Speech Corpus、Lisan‑Sudanese TTS数据集（4.6 h合成）、OOOK‑Eval、OOOK‑unlabeled、MGB‑2、Common Voice 11 h 等。

**📈 对比分析**

通过零射击、全微调、单独自训练、单独TTS、以及自训练+TTS混合四种方法进行比较；在OOUK‑Eval上最佳模型（Whisper‑Medium +自训练+TTS）WER为57.1%，在holdout集WER降至51.6%，远优于零射击（78.8%）且显著提升。

**⚠️ 局限性**

主要限制：苏丹方言语料仅4 h，性别与地区分布不均；合成TTS基于单一 Levantine 声学模型，导致声学差异；仍需更多无标签语料与更优秀的方言TTS以进一步提升性能。

---

## 479. V2P: Visual Attention Calibration for GUI Grounding via Background Suppression and Center Peaking

**arXiv ID:** 2601.06899 | [PDF](https://arxiv.org/pdf/2601.06899v1)

**作者:** Jikai Chen `[一作]` (Zhejiang University), Jinjie Gu `[通讯]` (Inclusion AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 V2P 的 GUI grounding 方法，通过视觉注意力校准实现更精准的定位。

**💡 创新点**

创新点在于结合背景抑制（Suppression Attention）和 Fitts-Gaussian 峰值建模，形成山谷-峰值训练策略。

**🔧 技术方法**

采用 Vision‑Language Model（如 Qwen2.5‑VL）提取特征，使用自注意力提升空间一致性，应用逆注意力正则化和 KL 损失对 2D 高斯先验进行监督。

**📊 数据集**

使用的公开基准包括 ScreenSpot‑v2、ScreenSpot‑Pro、OSWorld‑G、UI‑Vision、UI‑I2E 与 MMBench‑GUI L2，训练数据遵循 GUI‑Actor 的 recipe。

**📈 对比分析**

与 GUI‑Actor、UI‑TARS、JEDI 等方法对比，V2P‑7B 在 ScreenSpot‑v2 达到 92.4%、在 ScreenSpot‑Pro 达到 52.5%，相较最强基线分别提升约 3.6% 与 25.7%，在代理任务上也表现更优。

**⚠️ 局限性**

局限性包括对语义相似目标的歧义处理不足、面对非典型布局时注意力易散布，以及加入自注意力导致推理延迟略增。

---

## 480. How Do Ports Organise Innovation? Linking Port Governance, Ownership, and Living Labs

**arXiv ID:** 2601.06894 | [PDF](https://arxiv.org/pdf/2601.06894v1)

**作者:** Sonia Yeh `[一作]` (Chalmers University of Technology), Benedicte Madon `[通讯]` (Universidad de Sevilla)

**通讯引用:** 423 | [OpenAlex ID](https://openalex.org/A5013915651)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

本文通过案例比较研究，探讨港口治理模式（如Landlord与Tool/Public Service）如何影响Living Lab（LL）的设计与实施，并提出了治理-LL适配框架。

**💡 创新点**

创新点在于将港口治理模型与LL四大支柱（共创、真实环境、迭代学习、制度嵌入）关联，并通过“着陆区”概念解释LL成果转化为制度变革的机制。

**🔧 技术方法**

技术方法主要是结构化工作坊、利益相关者映射工具、SWOT分析和三层LL逻辑框架，并通过0-2评分量化治理-LL适配维度。

**📊 数据集**

数据来源于两港口（阿尔堡港与特伦勒堡港）的工作坊产出、数字化协作板、访谈邀请模板及半结构化访谈指南，未使用公开数据集。

**📈 对比分析**

通过对比两港口的治理-LL适配得分，分析不同治理模式下LL的可嵌入性和扩展性；性能表现体现在更高的得分对应更易制度化的LL成果。

**⚠️ 局限性**

局限在于样本仅限两港口、只涵盖早期工作坊阶段，缺乏纵向跟踪和更广泛治理类型的验证。

---

## 481. CLIMP: Contrastive Language-Image Mamba Pretraining

**arXiv ID:** 2601.06891 | [PDF](https://arxiv.org/pdf/2601.06891v1)

**作者:** Nimrod Shabtay `[一作]` (IBM Research), Raja Giryes `[通讯]` (Tel-Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了全Mamba结构的CLIP模型CLIMP，将视觉和文本编码器替换为Mamba

**💡 创新点**

创新点在于完全使用SSM实现跨模态对齐，获得子线性计算复杂度、可变分辨率、无77-token限制，并显著提升检索与OOD鲁棒性

**🔧 技术方法**

使用VMamba、Mamba-1/2、跨扫描SS2D、分层下采样以及自回归文本编码器

**📊 数据集**

训练数据为CC12M，评估使用CLIP-Benchmarks、ImageNet-V2/V4/A/O/Sketch、NoCaps、Crossmodal-3600、Flickr8k、DOCCI

**📈 对比分析**

与RoPE‑ViT、FlexViT、NaFlex‑ViT和CLIP‑ViT‑B/16等Transformer基线比较，CLIMP在检索、分类、ImageNet‑O等任务上提升数个百分点，内存与FLOPs降低5×，支持高分辨率和长文本检索

**⚠️ 局限性**

局限在于仅使用CC12M与基模型规模，尚未验证在LAION‑2B或更大ViT/L/H规模下的优势

---

## 482. Observability-Enhanced Target Motion Estimation via Bearing-Box: Theory and MAV Applications

**arXiv ID:** 2601.06887 | [PDF](https://arxiv.org/pdf/2601.06887v1)

**作者:** Yin Zhang `[一作]` (Westlake University), Shiyu Zhao `[通讯]` (Westlake University)

**通讯引用:** 4435 | [OpenAlex ID](https://openalex.org/A5052346042)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文提出一种基于三维检测框（3D bounding box）的观测增强目标运动估计方法（bearing‑box），并将其扩展到多旋翼无人机（MAV）的运动估计。该方法能在没有高阶观测者运动或球形目标假设的情况下，利用单目视觉测量估计目标位置、速度、加速度以及尺寸。

**💡 创新点**

创新点：
- 利用三维检测框内包含的姿态、尺寸比例信息，摆脱了传统观测器需要球形目标或侧向高阶运动的限制。
- 对MAV引入姿态-加速度耦合约束，进一步解除观测者高阶运动需求，使得在观测者静止或低阶运动时仍能观测到目标。
- 在伪线性卡尔曼滤波框架下给出观测模型和可观测性分析，实现了理论与实验并行的完整解决方案。

**🔧 技术方法**

使用技术：
- 伪线性卡尔曼滤波（pseudo‑linear Kalman filtering）实现状态估计。
- 三维目标检测网络（如WDRNet、MonoFlex、Yolov11‑Pose）输出检测框、尺寸比例和姿态。
- 观测模型中对投影矩阵、正交投影矩阵、加速度约束等数学工具。
- 可观测性分析采用状态空间可观测性矩阵求秩与多阶差分方法。

**📊 数据集**

使用的数据集与实验平台：
- 真实地面车辆实验：使用Motion Capture + 单目相机，采集约5k帧。
- KITTI 3D检测数据集：使用MonoFlex生成3D检测框。
- AirSim + Unreal Engine 仿真环境：测试MAV目标与观测者在不同运动状态下。
- 室内外真实MAV实验：使用DJI Mavic、WDRNet训练的检测网络。

**📈 对比分析**

与方法比较与性能：
- 与传统 bearing‑only 与 bearing‑angle 方法对比，bearing‑box 在两种场景（侧向运动与直线运动）均能快速收敛且估计误差显著低于对比方法。
- 在KITTI 204序列中，bearing‑box 的 NIDE（归一化积分深度误差）平均分别为 34.6%、13.5%、15.2%，显著优于 bearing‑only（54.5%、17.1%、90.9%）和 bearing‑angle（41.6%、19.9%、96.5%）。
- 对MAV实验，bearing‑box 能在观测者静止时准确估计目标位置、速度与尺寸，而 bearing‑only 与 bearing‑angle 直接发散或错误收敛到相机位置。

**⚠️ 局限性**

局限性：
- 对检测框的精度高度依赖，若检测误差大或出现遮挡，估计性能会退化。
- 目前仅在单目相机下验证，未深入探讨多相机或融合传感器场景。
- 伪线性卡尔曼滤波假设误差为高斯且独立，实际中可能受到非线性与误差耦合影响。
- 对于极小尺寸或远距离目标的3D检测依然困难，需要更鲁棒的检测网络支持。

---

## 483. Understanding the Performance Behaviors of End-to-End Protein Design Pipelines on GPUs

**arXiv ID:** 2601.06885 | [PDF](https://arxiv.org/pdf/2601.06885v1)

**作者:** Jinwoo Hwang `[一作]` (Korea Advanced Institute of Science and Technology), Jongse Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2172 | [OpenAlex ID](https://openalex.org/A5037553165)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `09944146-298c-433e-89df-37255de463d7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过对GPU端到端蛋白设计管线进行系统化剖析，分别在组件层面和完整管线层面对不同输入、超参数与采样策略下的性能进行详细测评，揭示GPU利用率低、对序列长度及采样方式高度敏感等现象；

**💡 创新点**

创新点在于：①首次对完整蛋白设计管线进行多粒度（组件级与整体级）GPU层面的性能表征；②系统性评估了GPU共址与多GPU扩展对吞吐率和延迟的影响，并提出了基于成本与准确性的多维评估框架；③公开开源了完整管线与分析脚本，为后续研究提供可复现的基准。

**🔧 技术方法**

使用的技术包括：Docker容器化、Nextflow工作流管理、Kubernetes + GPUShare插件进行GPU资源调度、NVIDIA NVML、Nsight Systems/Compute 进行GPU时间与空间利用率分析，以及多种主流AI模型（RFdiffusion、ProteinMPNN、ESMFold、AlphaFold3、Vina‑GPU等）。

**📊 数据集**

使用的数据集为四个蛋白支架结构和三种来自CASF‑2016的数据集的小分子配体，形成不同序列长度与复杂度的组合；同时采用公开的模型权重与参数进行实验。

**📈 对比分析**

比较方法：在不同采样计数（1、2、3、4、5）下测定各组件与整管线的总时延、GPU利用率与多GPU扩展效率；与单一样本、无共址情况做对比。性能表现显示：单GPU共址可提升约11.9%–27.7%延迟；多GPU从1→2 GPU可获得最高42.8%延迟下降，但随着GPU数量增多扩展效率明显下降。

**⚠️ 局限性**

局限性包括：仅在单一RTX 3090 GPU上评估，未覆盖更大规模GPU集群；采样范围有限，未考虑更高采样率对计算需求的真实影响；实验基于人工合成的蛋白‑配体组合，未验证生物学可行性；并且主要关注计算性能，缺乏对能耗、成本等指标的深入量化。

---

## 484. †DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems

**arXiv ID:** 2601.06853 | [PDF](https://arxiv.org/pdf/2601.06853v1)

**作者:** Zabir Al Nazi `[一作]` (University of California), Sudipta Kar `[通讯]` (Oracle Health AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个包含三类干扰的孟加拉语算术题基准DISTRACTMATH-BN，并评估多种模型在有无干扰信息下的推理表现。

**💡 创新点**

提出将数学问题转化为可执行计算图的框架，并通过图结构显式区分计算相关与无关节点，提升鲁棒性与推理效率。

**🔧 技术方法**

使用监督微调（SFT）+群组相对策略优化（GRPO）训练Gemma-3模型，并引入可执行计算图生成与解释器。

**📊 数据集**

基准数据源为孟加拉语MGSM和MSVAMP的3,685+3,947道题目，人工和自动验证后生成干扰版本。

**📈 对比分析**

与标准Chain-of-Thought、专门的推理模型对比，所提框架在有干扰情形下准确率相当或更高，同时使用的token约为前者的11%（约89%节省）。

**⚠️ 局限性**

局限在于仅覆盖算术文字题、三类数值干扰，未考虑非数值噪声，且模型需要较大参数才能充分利用图结构，框架不适用于几何、代数或概率推理。

---

## 485. LLM Performance Predictors: Learning When to Escalate in Hybrid Human-AI Moderation Systems

**arXiv ID:** 2601.07006 | [PDF](https://arxiv.org/pdf/2601.07006v1)

**作者:** Or Bachar `[一作]` (Zefr), Jonathan Morra `[通讯]` (Zefr)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于 LLM 性能预测（LPP）的框架，用以在内容审核中实现基于成本的选择性升级（trust‑or‑escalate）决策。

**💡 创新点**

创新点包括：① 将灰盒（token 级别概率、熵等）与黑盒（自述置信度、归因信号）特征融合构建多维度性能指示器；② 引入审核特定的归因特征（证据缺失 vs. 政策缺口），实现可解释的升级路径；③ 用岭回归学习元模型并通过成本敏感阈值实现自动化成本优化。

**🔧 技术方法**

技术手段：基于 token 级别 log‑prob 的灰盒特征；自述置信度与离散置信度带；二值归因指示器；岭回归（带 sigmoid 或 isotonic 归一化）作为元分类器；成本敏感阈值搜索；数据预处理（token 对齐、重试机制）。

**📊 数据集**

数据集：① OpenAI Moderation（1,680 文本，多标签；共约 2.5k 文本‑标签实例）；② Multimodal Moderation（1,500 短视频，3 类风险、12 种语言、4 模态）。

**📈 对比分析**

与 MSP、Top‑2 Margin、Entropy 以及 always‑trust 基线对比。LPP 在两组数据集上均实现：F1、AUC‑ROC 与 Macro‑F1 明显提升（多数模型提升 5–15% 以上），并在成本方面实现 30–70% 的下降；在多模态任务中，部分模型的成本下降更为显著。

**⚠️ 局限性**

局限性：① 需要手工标注正确/错误样本；② 仅利用后置预测特征，未考虑前置预测；③ 仅针对 Transformer‑based LLM，未验证对 RAG 或符号化模型的适用性；④ 成本模型固定，未考虑不同审核员或严重性差异；⑤ 在某些模型/数据分布下，性能提升有限。

---

## 486. Directional Selective Fixed-Filter Active Noise Control Based on a Convolutional Neural Network in Reverberant Environments

**arXiv ID:** 2601.06981 | [PDF](https://arxiv.org/pdf/2601.06981v1)

**作者:** Boxiang Wang `[一作]` (Nanyang Technological University), Woon-Seng Gan `[通讯]` (Nanyang Technological University)

**通讯引用:** 7723 | [OpenAlex ID](https://openalex.org/A5072584895)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于卷积神经网络的方向性固定滤波主动噪声控制方法，利用多通道参考信号估计噪声源的方位，并动态选择对应的预训练控制滤波器，实现对混响环境中不同方位噪声源的快速高效抑制。

**💡 创新点**

创新点在于将噪声源方向估计与固定滤波器选择耦合，并采用轻量级多任务学习CNN实现实时方位估计，显著提升了在方位变化频繁环境下的控制稳定性和响应速度。

**🔧 技术方法**

技术主要包括：多通道参考ANC系统、短时傅里叶变换特征提取、轻量级CNN（含卷积、组归一化、ReLU、池化层）进行方位分类、交叉熵联合损失的多任务学习、预训练固定滤波器库和基于FFT的实时控制。

**📊 数据集**

使用的数据集包含合成宽带白噪声和来自UrbanSound8K的真实噪声，结合多室尺寸、不同RT60、不同SNR的房间衰减脉冲响应生成的参考信号，总计约57600条样本（训练/验证/测试）。

**📈 对比分析**

与传统FxLMS、标准SFANC和GFANC等基线方法比较，在100–700 Hz带宽噪声和洗衣机噪声的仿真中，方向性SFANC在保持更快响应的同时提供约4–6 dB更高的噪声抑制性能。

**⚠️ 局限性**

局限性在于未考虑源到阵列距离估计，固定滤波器库仅覆盖离散方位，距离变化可能导致抑制效果下降。

---

## 487. Arithmetic Complexity of Solutions of the Dirichlet Problem

**arXiv ID:** 2601.06954 | [PDF](https://arxiv.org/pdf/2601.06954v1)

**作者:** Holger Boche `[一作]` (BMBF Research Hub 6G-life), H. Vincent Poor `[通讯]` (Princeton University)

**通讯引用:** 148926 | [OpenAlex ID](https://openalex.org/A5042307561)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

对单位圆盘上 Dirichlet 问题的两种经典求解方法（Poisson 积分与 Dirichlet 原理）在 Turing 机上的可计算性进行了系统研究，证明对一般可计算连续边界函数，其最小能量和点值往往不可算，并给出了它们在 Zheng–Weihrauch 可计算性层级中的确切位置。

**💡 创新点**

创新点在于：①首次把 Dirichlet 问题的可计算性与 Zheng–Weihrauch 层级联系起来；②证明即使边界函数可计算，最小 Dirichlet 能量只能在 Σ₁ 层（可递归逼近），而点值只能在 Δ₂ 层；③构造具体可计算边界函数，使能量或点值达到这些层级的下界，展示了两种求解方法在可计算性上的根本差异。

**🔧 技术方法**

采用可计算分析、Sobolev 空间与 Fourier 系数的有效逼近、递归逼近理论以及 Zheng–Weihrauch 层级的定义与性质进行严格证明。

**📊 数据集**

无实验数据集，整个工作纯粹是理论证明与构造。

**📈 对比分析**

本文不做实验比较，而是通过构造可计算边界函数与递归逼近序列，证明了两种方法在可计算性上的极限；在层级理论上，Dirichlet 能量上界为 Σ₁，点值上界为 Δ₂，下界分别为 Σ₁ 与 Σ₁∪Π₁。

**⚠️ 局限性**

局限性：仅给出理论上可计算性层级的界限，未给出实际可行的算法；对于更高层级的精确度尚未确定；此外，只考虑单位圆盘的 Laplace 方程，对更一般的 PDE 或域形的可计算性分析仍是开放问题。

---

## 488. X-Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests

**arXiv ID:** 2601.06953 | [PDF](https://arxiv.org/pdf/2601.06953v1)

**作者:** Jie Wu `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 3782 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 SynthSmith 全合成管线，生成竞赛级编程任务、解答和测试，并以此训练 X‑Coder 代码 LLM；

**💡 创新点**

创新点在于实现完全无真实数据的合成数据集、双重验证策略、任务风格多样化与 RL 训练一体化；

**🔧 技术方法**

采用特征驱动合成、长链‑CoT、双重验证、GRPO RL、工具与提示生成测试等技术；

**📊 数据集**

使用 SynthSmith 合成的 64k–200k 任务（含 10k 任务的 SelfCodeAlign 版）并对比 APPS、LiveCodeBench 等真实数据集；

**📈 对比分析**

与多种基线对比，LiveCodeBench v5/v6 avg@8 分别达 62.9/55.8，明显优于 14B 规模基线；RL 进一步提升约 4–5%；

**⚠️ 局限性**

局限在合成任务可能与真实世界偏差、验证成本高、模型仍易出现推理错误和时间限制。

---

## 489. Watching, Reasoning, and Searching: A Video Deep Research Benchmark on Open Web for Agentic Video Reasoning

**arXiv ID:** 2601.06943 | [PDF](https://arxiv.org/pdf/2601.06943v1)

**作者:** Chengwen Liu `[一作]` (Lanzhou University), Huacan Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5031229572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 VideoDR 视频深度研究基准，评估模型在视频与开放网络交互中的多帧视觉锚点提取、网页检索和多跳推理能力。

**💡 创新点**

创新点在于：①首次将视频作为核心证据入口，要求跨帧视觉锚点与网页证据共同构建答案；②设计严格的双向依赖质量控制；③对 Workflow 与 Agentic 两种代理范式进行系统对比，揭示目标漂移与长链一致性为关键瓶颈。

**🔧 技术方法**

使用多模态大型语言模型（Gemini‑3‑pro‑preview、GPT‑5、GPT‑4o、MiniCPM‑V 4.5、Qwen3‑Omni‑30B‑A3B、InternVL3.5‑14B）在 Workflow 与 Agentic 两种框架下的交互与工具调用；采用 LLM‑as‑Judge（DeepSeek‑V3‑0324）进行答案评估。

**📊 数据集**

数据集为 100 条高质量 VideoDR 样本，覆盖六大语义域（日常、经济、技术、文化、历史、地理），每条样本需同时依赖视频与网页证据，且视频时长分布从短视频到长视频。

**📈 对比分析**

通过人类上限对比和多维度（难度、视频时长、语义域）分层分析，Gemini‑3‑pro‑preview 在 Agentic 方案下最高 76%，Workflow 69%；GPT‑5 同等；GPT‑4o 在两方案均约 42%；开源模型表现更低。Agentic 并非始终优越，取决于模型能否在多轮检索中保持初始视频锚点；难度升高、视频更长时 Agentic 效果更差。

**⚠️ 局限性**

局限性包括：①样本仅基于专家搜索轨迹，缺乏多样化用户检索行为；②在 Agentic 方案下模型无法重新观看视频，易产生目标漂移；③数据规模有限（仅 100 条），难以覆盖更广泛的现实场景。

---

## 490. VISTA: Knowledge-Driven Interpretable Vessel Trajectory Imputation via Large Language Models

**arXiv ID:** 2601.06940 | [PDF](https://arxiv.org/pdf/2601.06940v1)

**作者:** Hengyu Liu `[一作]` (Aalborg University), Christian S. Jensen `[通讯]` (Aalborg University)

**通讯引用:** 30151 | [OpenAlex ID](https://openalex.org/A5029380368)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了VISTA框架，利用大型语言模型和结构化知识图谱实现海事AIS轨迹缺失补全，并生成可解释的知识提示。

**💡 创新点**

创新点在于：①将结构化数据抽取的知识（SDK）与隐式LLM知识融合形成完整的“底层知识”；②构建数据–知识–数据循环和可扩展的知识图谱；③引入工作流管理层实现并行、容错与冗余消除，提高计算效率。

**🔧 技术方法**

使用技术包括：大型语言模型（用于行为抽象、解释生成与函数合成）、结构化数据驱动的知识图谱（SD‑KG）、统计检索与图遍历、并行工作流调度与异常处理。

**📊 数据集**

实验数据集：AIS‑DK（丹麦海域）和AIS‑US（美国沿海），各包含约10,000条船舶轨迹和200万条AIS记录。

**📈 对比分析**

与规则、深度学习及其他LLM方法对比，VISTA在MAE/RMSE/MHD指标上提升5%–94%，在推理时间上比思维型LLM基线低51%–93%，同时提供可解释的行为模式与规则依据。

**⚠️ 局限性**

局限性：决策与行为估计高度依赖高容量LLM，轻量模型性能下降明显；对LLM生成函数的安全性与可验证性仍需改进；在不同海域或非标准AIS格式时的泛化能力尚待进一步验证。

---

## 491. Forgetting Similar Samples: Can Machine Unlearning Do it Better?

**arXiv ID:** 2601.06938 | [PDF](https://arxiv.org/pdf/2601.06938v1)

**作者:** Heng Xu `[一作]` (City University of Macau), Wanlei Zhou `[通讯]` (City University of Macau)

**通讯引用:** 14677 | [OpenAlex ID](https://openalex.org/A5051406984)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了现有机器去学习方法在包含相似样本的数据集上的效果，并提出了基于鲁棒性训练的增强方案。

**💡 创新点**

系统揭示了相似样本对去学习效果的显著影响，并首次将数据增强和噪声注入与正则化相结合来提升去学习的完整性。

**🔧 技术方法**

针对图像采用重训练与相似样本删除，针对语言模型采用噪声注入、扰动隐藏层以及 KL 正则化等技术实现增强去学习。

**📊 数据集**

构造了四个相似样本数据集：Similarity‑Entailed MNIST、FMNIST、CIFAR‑10 以及 PKU‑SafeRLHF 语言问答集。

**📈 对比分析**

通过像素级 SSIM、ROUGE 等细粒度验证方法与原始去学习方法及 retrain baseline 进行对比，增强方案在去学习效果上显著提升，同时模型整体性能下降仅约 0.5%。

**⚠️ 局限性**

尚未解决不同模型规模与超参数设置的泛化问题，且对测试集相似样本的残留影响仍需进一步研究。

---

## 492. Measuring Social Bias in Vision-Language Models with Face-Only Counterfactuals from Real Photos

**arXiv ID:** 2601.06931 | [PDF](https://arxiv.org/pdf/2601.06931v1)

**作者:** Haodong Chen `[一作]` (Harbin Institute of Technology), Jun Yu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 13210 | [OpenAlex ID](https://openalex.org/A5050817770)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于真实照片的面部仅对比实验（FOCUS）及对应评估基准（REFLECT），用于评估视觉语言模型的社会偏见。

**💡 创新点**

提出面部仅的对照编辑方式，在保持背景、服饰、姿态等视觉因素不变的前提下，仅改变面部种族与性别，从而消除视觉混淆并实现更清晰的因果归因。

**🔧 技术方法**

采用面部编辑工具（如Nano Banana Pro）生成对照图像，并利用GPT、Gemini、Qwen、DeepSeek、Llama等主流视觉语言模型进行实验。

**📊 数据集**

使用FOCUS数据集（480张场景匹配的面部对照图像，涵盖六种职业、十个种族-性别组合）和REFLECT基准（包含2AFC、MCQ与薪资推荐三种任务）。

**📈 对比分析**

通过Win率、均值差距和Jensen–Shannon Divergence等指标比较模型与任务表现，实验表明即使在严格视觉控制下，不同模型和任务仍显著存在种族/性别偏差，且偏差方向随任务类型变化。

**⚠️ 局限性**

受限于面部编辑的精确性和数据集规模，可能引入未控制的面部细节变化；数据集覆盖范围有限，无法代表全部职业或文化背景。

---

## 493. Calibrating Agent-Based Financial Markets Simulators with Pretrainable Automatic Posterior Transformation-Based Surrogates

**arXiv ID:** 2601.06920 | [PDF](https://arxiv.org/pdf/2601.06920v1)

**作者:** Boquan Jiang `[一作]` (Southern University of Science and Technology), Peng Yang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 4910 | [OpenAlex ID](https://openalex.org/A5039532881)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于预训练后验转换（APT）的代理模型与负相关搜索（NCS）及自适应可信区（TR）相结合的金融基于代理模型（ABM）校准框架（ANTR）

**💡 创新点**

创新点在于用APT直接学习参数后验分布，解决数据误差与参数误差不匹配问题，并通过NCS与自适应TR提升多峰搜索与样本效率

**🔧 技术方法**

核心技术包括自动后验转换（APT）+混合密度网络/正则化流、负相关搜索（NCS）保持多样性、可信区自适应机制以及CNN特征嵌入和零填充处理序列长度

**📊 数据集**

使用两个典型金融ABM数据集：Brock–Hommes异质预期模型（BH）和基于订单簿的PGPS模型，包含多维参数和不同时间序列长度的模拟样本

**📈 对比分析**

与传统进化算法（PSO、NCS、TuRBO）和最先进的SAEAs（CAL‑SAPSO）比较，ANTR在MSE、参数估计误差、成功率等指标上均显著优于基线，且样本效率提升约50%–70%

**⚠️ 局限性**

局限性包括可信区机制仅基于停滞触发重启，未充分利用已评估样本信息；NCS未对后验与真实目标差异进行自适应采样，未来可进一步提升精度与收敛速度

---

## 494. PenForge: On-the-Fly Expert Agent Construction for Automated Penetration Testing

**arXiv ID:** 2601.06910 | [PDF](https://arxiv.org/pdf/2601.06910v1)

**作者:** Huihui Huang `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 29512 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PenForge 框架，通过在渗透测试过程中动态构造专家型 LLM 代理，实现自动化 Web 应用渗透。

**💡 创新点**

核心创新在于：① 使用 Meta‑Planner 对目标进行实时侦察并评估可能的漏洞类型；② 在每一次攻击尝试前，根据侦察信息即时生成与该攻击类型匹配的专家代理，从而避免预先固定的提示与知识库限制。

**🔧 技术方法**

采用的大技术包括：大语言模型（如 GPT‑4）、AutoGPT 作为执行框架、feroxbuster、WebPageReader 进行信息收集、Perplexity API 等知识检索工具，以及 Meta‑Planner 的两阶段决策流程。

**📊 数据集**

实验基于 CVE‑Bench 40 条真实 CVE 的 Web 应用漏洞数据集，在零日场景下评估。

**📈 对比分析**

与 Cy‑Agent、AutoGPT、T‑Agent 等基线进行对比，零日设定下 success@1 20.5%（比 T‑Agent 的 8.0% 翻倍），success@5 30%（比 10% 翻三倍），显著提升渗透成功率，尤其在“unauthorized administrator login”和“outbound service”两类攻击上表现突出。

**⚠️ 局限性**

主要局限包括：工具使用失误导致的失败率偏高；知识检索仅基于通用 API，缺乏专业渗透测试知识；CVE‑Bench 仅为单一 CVE 标注，未覆盖同一系统多重漏洞；缺乏可解释性与人机协作机制。

---

## 495. Large Artificial Intelligence Models for Future Wireless Communications

**arXiv ID:** 2601.06906 | [PDF](https://arxiv.org/pdf/2601.06906v1)

**作者:** Chong Huang `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19388 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了大规模AI模型在未来无线通信中的应用与挑战，并提出了集成架构与研究方向。

**💡 创新点**

首次系统阐述将大语言/视觉模型融合为多模态大模型在无线通信中的潜力，并对能耗、架构设计、隐私安全等挑战提出解决思路。

**🔧 技术方法**

Transformer/大语言模型、深度学习、深度强化学习、边缘计算、语义通信、联邦学习、差分隐私、同态加密、区块链等技术。

**📊 数据集**

文中未给出具体实验，主要参考通用文本、图像等大规模数据集。

**📈 对比分析**

未进行实验比较，本文以理论分析和案例讨论为主。

**⚠️ 局限性**

技术可行性、极高能耗、架构复杂、隐私安全难题、缺乏实证验证是主要限制。

---

## 496. A New Perspective on Drawing Venn Diagrams for Data Visualization

**arXiv ID:** 2601.06980 | [PDF](https://arxiv.org/pdf/2601.06980v1)

**作者:** Bálint Csanády `[一作]` (Eötvös Loránd University), Bálint Csanády `[通讯]` (Eötvös Loránd University)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5053798868)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 VennFan 方法，利用极坐标投影和形状正弦曲线构造可读的 n 组 Venn 图。

**💡 创新点**

通过可调幅度衰减与形状指数避免曲线碰撞，生成扇形曲线；同时提出与 Edwards cogwheel 等价的余弦变体，并设计了可视中心标签放置启发式。

**🔧 技术方法**

使用三角函数、形状指数、线性/指数幅度衰减、极坐标投影以及视觉中心算法的组合实现，配合 Python 实现。

**📊 数据集**

主要以合成 Venn 图为实验；在 EEG 失语注释数据中展示六个注释者的重叠结构，但未使用专门的数据集进行评估。

**📈 对比分析**

通过对 Edwards’ cogwheel 与 cosine-based VennFan 的区域面积分布直方图进行比较，发现后者区域面积更均匀、可读性更好；支持至 n≈8，但未进行大规模性能测试。

**⚠️ 局限性**

对 n>8 时区域过小导致标签难放，正弦变体不一定简单且需手动调参；算法对参数敏感，且未在大规模真实数据上进行验证。

---

## 497. ObjSplat: Geometry-Aware Gaussian Surfels for Active Object Reconstruction

**arXiv ID:** 2601.06997 | [PDF](https://arxiv.org/pdf/2601.06997v1)

**作者:** Yuetao Li `[一作]` (Beijing Institute of Technology), Shaohui Zhang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 3649 | [OpenAlex ID](https://openalex.org/A5100649219)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了ObjSplat，一套基于二维高斯surfels的闭环主动物体重建框架，能够在机器人扫描过程中实时更新几何与纹理模型并驱动视角规划；

**💡 创新点**

创新点包括：①几何感知的视角评估管线，显式建模背面可见性与遮挡可视性，精准识别未重建区域；②下一最佳路径（NBP）规划器，利用多步前瞻的空间拓扑同时考虑信息增益与运动成本，避免贪婪单步策略导致的冗余运动；③将几何与纹理联合优化与统一表示融合，提升稀疏观测下的几何一致性与视觉质量；

**🔧 技术方法**

使用技术包括二维高斯surfels表示、可微渲染与α混合、前向投影/逆投影的遮挡检测、基于视角共视率的相机关联、贝塔式信息增益权重、k最近邻空间图的多步路径搜索、TSDF/Poisson网格提取与光照补偿；

**📊 数据集**

实验数据集主要为Google Scanned Objects（GSO）16个多样化模型，另外在真实环境下扫描四件文化遗产物件；

**📈 对比分析**

与SEE、PB‑NBV、MA‑SCVP、FisherRF、GauSS‑MI等基线对比，指标涵盖PSNR、SSIM、LPIPS、深度误差、Chamfer距离、F‑Score、完成率、路径长度、扫描时间；结果显示ObjSplat‑NBP在完成率约91%与其他方法相当，同时将路径长度与扫描时间分别压缩至约1/4和1/3；

**⚠️ 局限性**

局限性在于仅针对单一静止刚体，光照与材质变化受限；对半透明、强反射或动态场景的处理尚未实现，未来工作将探索材质估计与动态对象建模。

---

## 498. Divergence-Based Adaptive Aggregation for Byzantine Robust Federated Learning

**arXiv ID:** 2601.06903 | [PDF](https://arxiv.org/pdf/2601.06903v1)

**作者:** Bingnan Xiao `[一作]` (Fudan University), Xin Wang `[通讯]` (Fudan University)

**通讯引用:** 81504 | [OpenAlex ID](https://openalex.org/A5100328102)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作提出两种聚合框架DRAG和BR‑DRAG，解决联邦学习中由数据异构导致的客户端漂移以及拜占庭攻击对训练收敛的破坏，显著加快模型收敛速度。

**💡 创新点**

创新点在于：①引入分歧度（DoD）度量本地更新与全局参考方向的偏离，利用该度量通过“拖拽”机制将本地梯度沿参考方向调整，从而消除漂移；②在BR‑DRAG中使用可信根数据生成参考方向，并对上传梯度进行归一化校正，既抵御攻击又不需要额外的控制变量或正则化。

**🔧 技术方法**

技术手段包括：梯度方向对齐与归一化、分歧度度量、指数滑动平均参考方向、鲁棒根数据估计、梯度裁剪/归一化、对非凸目标下的收敛性分析（异构数据、部分参与、拜占庭攻击）。实现基于PyTorch。

**📊 数据集**

实验数据集：EMNIST、CIFAR‑10、CIFAR‑100。

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD、FedExP、FedACG等常规联邦学习方法以及FLTrust、RFA、RAGA等拜占庭鲁棒方法进行对比。实验表明DRAG在异构环境下显著快于基准（收敛到70%准确率仅需约600轮），BR‑DRAG在三种攻击场景（噪声、符号翻转、标签翻转）和高比例拜占庭（最高60%）时仍保持稳定收敛，并在准确率上优于FLTrust与几何中位数方法。

**⚠️ 局限性**

局限性：目前仅验证集中式FL环境；需要可信根数据集与标签，若根数据不可用则难以部署；对超参数（α、c、c^t）的敏感性需进一步自动化；理论收敛证明基于梯度方差与异构程度的上界，实际情况可能更为复杂。

---

## 499. A Sliding Mode Controller Based on Timoshenko Beam Theory Developed for a Tendon-Driven Robotic Wrist

**arXiv ID:** 2601.07009 | [PDF](https://arxiv.org/pdf/2601.07009v1)

**作者:** Shifa Sulaiman `[一作]` (University of Naples Federico II), Fanny Ficuciello `[通讯]` (University of Naples Federico II)

**通讯引用:** 3064 | [OpenAlex ID](https://openalex.org/A5065563904)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于Timoshenko梁理论的肌腱驱动机器人腕部，并提出了高效的滑模控制器实现精确运动控制。

**💡 创新点**

采用Timoshenko梁理论对柔性腕部进行精确建模，并将其与滑模控制器相结合，得到更快的动态响应、更低的误差，同时通过PSO优化滑模参数，提升控制性能。

**🔧 技术方法**

Timoshenko梁理论建模、滑模控制（含tanh滑移面）、粒子群优化（PSO）参数调节、Simulink/Arduino/ROS实时控制、实验验证等技术。

**📊 数据集**

未使用公开数据集，主要使用实验测量数据（腕角度、力传感等）和仿真生成的数据进行验证。

**📈 对比分析**

与几何可变应变控制器（GVSC）和传统PID控制器进行RMSE、稳态误差、上升时间比较。实验与仿真结果显示，SMC在RMSE（0.016 rad）、稳态误差（0.003 rad）和上升时间（1.9 s）上均优于GVSC和PID。

**⚠️ 局限性**

实验误差高于仿真，主要因弹簧刚度不稳定；模型简化未充分考虑摩擦、磨损等非线性因素；实验环境与实际工况差异大，需要进一步改进可移动性和适配不同平台。

---

## 500. MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education

**arXiv ID:** 2601.06979 | [PDF](https://arxiv.org/pdf/2601.06979v1)

**作者:** Dongsuk Jang `[一作]` (Seoul National University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 7668 | [OpenAlex ID](https://openalex.org/A5064858748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究构建了一个基于检索增强生成（RAG）的系统，能够自动将放射科临床病例报告转化为包含最新文献与教科书摘录的教育模块和对应的多项选择题；

**💡 创新点**

创新点包括：①混合检索机制，将本地医学教科书和实时学术检索（PubMed、Semantic Scholar）同步进行；②使用大语言模型（LLM）结合重排序模型（Qwen3‑Reranker‑8B）生成高质量教育内容；③采用异步多进程与vLLM批处理实现高吞吐量；④在本地部署，保障数据隐私；

**🔧 技术方法**

主要技术包括：LLM（Llama‑3.3‑70B‑Instruct、MedGemma‑27B 等）、OCR、文本嵌入、Qwen3‑Reranker‑8B 重排序、vLLM 服务器、异步 I/O、并行 GPU 推理、PubMed 与 Semantic Scholar API；

**📊 数据集**

使用五个放射科公开数据集进行评估：Yale Hospital Internal、MIMIC‑CXR、MIMIC‑IV‑note、CheXpert Plus、ReXGradient‑160K，共 2000 条病例；

**📈 对比分析**

通过三名放射科专家评分（Likert 1‑5）与 LLM‑as‑Judge 对比，模型生成的教材、摘要和 MCQ 的平均人类评分约为 3.4‑3.6，LLM 评分更高但与专家评分相关性中等；模型间比较显示 MedGemma‑27B 在 MCQ 质量上略优；

**⚠️ 局限性**

局限性包括：评估仅聚焦放射科，尚未验证在其他医学专科的适用性；人类评估样本仅 50 条病例，规模有限；MCQ 评价一致性低，需进一步完善提示与评估准则；

---

## 501. Generalization Bounds for Transformer Channel Decoders

**arXiv ID:** 2601.06969 | [PDF](https://arxiv.org/pdf/2601.06969v1)

**作者:** Qinshan Zhang `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**通讯引用:** 10129 | [OpenAlex ID](https://openalex.org/A5034104790)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出了Transformer结构的误差校正码解码器（ECCT）的学习理论通用性分析，给出了基于位误码率（BER）的泛化误差上界，并展示了稀疏（基于奇偶校验矩阵的掩码）注意力机制对该上界的收缩作用。

**💡 创新点**

创新点在于：①首次将乘法噪声估计误差与BER关联，并通过位级Rademacher复杂度推导出ECCT的泛化误差上界；②证明掩码稀疏注意力能显著降低覆盖数和全局Lipschitz常数，从而得到更紧的泛化上界；③扩展了单层到多层ECCT的理论分析，并给出对AWGN通道下输入无界时的概率校正。

**🔧 技术方法**

主要技术包括：位级Rademacher复杂度、Dudley熵积分、覆盖数分析、全局与局部Lipschitz常数估计、奇偶校验矩阵构造的稀疏掩码、梯度稀疏性证明、谱范数与Frobenius范数的链式估计，以及多层Transformer的链式梯度传播分析。

**📊 数据集**

实验数据集为BPSK调制、AWGN信道下的码字，使用的码包括(128,64) CCSDS码和从(1056,528) WiMAX LDPC码衍生的子码，训练集大小为12,800，嵌入维度为32，信噪比为2 dB。

**📈 对比分析**

通过对比不同注意力层数T和输入序列长度L下的泛化误差（训练BER与测试BER之差），验证理论中预测的O(√L)、O(1/√m)和O(T)等标度关系。实验显示，稀疏掩码的ECCT在相同参数下比无掩码模型具有更小的泛化误差，且随T、L的增大趋势与理论一致。

**⚠️ 局限性**

局限性包括：①仅在二进制输入对称输出（BISO）通道下给出理论；②对输入无界的AWGN处理采用保守的概率上界，实际泛化误差可能更好；③理论上界可能相对松散，实际性能远优于上界；④未给出与传统BP、NBP或其他深度解码器的直接BER对比，只验证了泛化标度。

---

## 502. Towards Operational Streamflow Forecasting in the Limpopo River Basin using Long Short-Term Memory Networks

**arXiv ID:** 2601.06941 | [PDF](https://arxiv.org/pdf/2601.06941v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 503. A Robust Certified Machine Unlearning Method Under Distribution Shift

**arXiv ID:** 2601.06967 | [PDF](https://arxiv.org/pdf/2601.06967v1)

**作者:** Jinduo Guo `[一作]` (Johns Hopkins University), Yinzhi Cao `[通讯]` (Johns Hopkins University)

**通讯引用:** 3453 | [OpenAlex ID](https://openalex.org/A5070605476)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在非i.i.d.删除集下实现有效认证机器遗忘的算法

**💡 创新点**

首次将信赖域（trust‑region）约束与Newton迭代相结合，形成分布感知的迭代遗忘框架，能够在分布偏移时提供更紧的梯度残差上界并减少噪声注入

**🔧 技术方法**

Newton方法、二阶Taylor展开、信赖域优化、Hessian向量乘积（HVP）、近似Newton步、梯度与Hessian Lipschitz分析

**📊 数据集**

MNIST与CIFAR‑10（使用3层MLP和AllCNN网络）

**📈 对比分析**

与目前SOTA的两种认证遗忘方法（Guo et al. 2020、Zhang et al. 2024）在F1、损失、ΔF1、U‑MIA等指标上对比，TR‑Certified在ΔF1与ΔLoss均显著下降，U‑MIA误差更小，表明在分布偏移下保持更高的效能与实用性

**⚠️ 局限性**

仍需对大规模模型、在线实时遗忘和多模态数据的适用性进行评估；对Hessian估计精度和计算成本的依赖可能限制在更复杂网络中的推广

---

## 504. Unified Personalized Understanding, Generating and Editing

**arXiv ID:** 2601.06965 | [PDF](https://arxiv.org/pdf/2601.06965v1)

**作者:** Yu Zhong `[一作]` (Zhejiang University), Yueting Zhuang `[通讯]` (Zhejiang University)

**通讯引用:** 16538 | [OpenAlex ID](https://openalex.org/A5008666077)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OmniPersona 框架，实现统一的个性化理解、生成和图像编辑，构建全流程端到端系统。

**💡 创新点**

创新点包括：① 结构化解耦概念 token，将理解与生成划分到不同专家子空间，降低跨任务干扰；② 推理时 explicit knowledge replay 机制，将隐式知识显式化为文本提示，提升属性推理生成的稳定性和可解释性；③ 设计 OmniBench+Edit 基准，首次在统一模型上系统评估编辑能力，并证明编辑监督可反向提升理解和生成。

**🔧 技术方法**

技术手段：多任务联合训练（理解+生成+编辑）；token 解耦路由到专家子空间；推理时三阶段知识重放（意图解析 → 内存检索 → 提示合成）；基于 Bagel 的融合自回归+扩散生成；使用 t‑SNE 等可视化验证 token 分离效果。

**📊 数据集**

数据集：基于 UnifyBench 的 20 个概念（人、宠物、物体），以及新增的 OmniEdit 编辑数据集（源图像、编辑指令、目标图像）。每概念使用 32 个可学习 token（16 用于理解，16 用于生成）。

**📈 对比分析**

与四类基线（仅理解、仅生成、统一个性化、检索增强）以及 Bagel 0‑shot 对比，使用 UnifyBench 指标、CLIP‑I、PARG、SEMA‑C、QUAL‑I 等。OmniPersona 在个性化理解、生成、属性推理生成和编辑四项指标均超越 SOTA，尤其在编辑任务的语义一致性（SEMA‑C 0.711）和整体编辑质量（QUAL‑I 0.605）表现突出。

**⚠️ 局限性**

局限性：① 仍需手工标注编辑数据，难以覆盖极端或复杂编辑需求；② 推理时知识重放步骤增加推理开销；③ 对多概念交互和长文本指令的泛化能力尚待提升；④ 受限于训练资源，模型规模和算力需求较高。

---

## 505. Operational Runtime Behavior Mining for Open-Source Supply Chain Security

**arXiv ID:** 2601.06948 | [PDF](https://arxiv.org/pdf/2601.06948v1)

**作者:** Zhuoran Tan `[一作]` (University of Glasgow), Christos Anagnostopoulos `[通讯]` (University of Glasgow)

**通讯引用:** 4541 | [OpenAlex ID](https://openalex.org/A5049657159)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 HeteroGAT‑Rank，一种基于注意力的异构图学习框架，用来从沙箱运行日志中构建单跳异构图，生成可操作的运行时行为指标供供应链安全分析师使用。

**💡 创新点**

创新点在于将单包运行时行为抽象为轻量级单跳异构图，结合注意力与对比正则化的 HeteroGAT，并通过注意力和 Grad‑CAM 提取可解释的关键节点/边，直接输出可用于人机交互的调查 pivot。

**🔧 技术方法**

采用异构图注意力网络（GATv2）、多头注意力池化、对比损失、熵与稀疏正则以及 Grad‑CAM 解释；在 PyTorch Geometric 上实现；使用 Ray Actors 进行并行编码。

**📊 数据集**

使用 OpenSSF OSPTrack 公开跨五大生态系统（Rust、NPM、PHP、PyPI、Ruby）的 sandbox 运行数据，扩展后共 9,758 个实例，其中 2,259 个标记为恶意。

**📈 对比分析**

与三种轻量级基线（熵、相关、SHAP+XGBoost）及三种模型变体（HeteroGAT、DHeteroGAT、PNHeteroGAT）进行对比；在 AUC 上各模型均在 0.73–0.95 之间，DHeteroGAT 通过注意力+稀疏正则实现了更高的召回率，且在 top‑10 指标上显著提升了可操作性。

**⚠️ 局限性**

受限于 sandbox 覆盖、标签不平衡及单跳图无法捕获多跳控制流，模型解释仅为注意力权重而非因果因子，且需要显式的运行时日志支持。

---

## 506. Caching Yields up to 5x Spectral Efficiency in Multi-Beam Satellite Communications

**arXiv ID:** 2601.06925 | [PDF](https://arxiv.org/pdf/2601.06925v1)

**作者:** Hui Zhao `[一作]` (EURECOM), Petros Elia `[通讯]` (EURECOM)

**通讯引用:** 2683 | [OpenAlex ID](https://openalex.org/A5015066458)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

论文研究了在多波束卫星通信中将向量编码缓存（VCC）与多天线预编码结合，以提高光谱效率。

**💡 创新点**

创新点是利用接收端缓存在物理层实现“缓存消除”干扰，将多条预编码信号叠加，从而实现比传统MU‑MISO多 5 倍以上的理论与实际增益。

**🔧 技术方法**

使用Rician‑阴影衰落模型、匹配滤波（MF）预编码、CSI获取与估计误差模型，并推导平均总速率与有效增益的闭式表达式。

**📊 数据集**

采用三种阴影环境参数（频繁重阴影、平均阴影、稀疏轻阴影）的Rician阴影参数，并基于仿真验证，未使用公开数据集。

**📈 对比分析**

与同等资源的传统缓存无、MU‑MISO系统进行比较，利用有效增益指标；结果显示在低至中等SNR下，VCC可实现 300%–550% 的光谱效率提升，动态与静态信道均保持近似。

**⚠️ 局限性**

限制包括对CSI获取开销采用保守的固定训练长度、忽略多用户间时变性与多普勒效应，且对用户缓存大小、文件尺寸分片等实际系统参数的更细粒度评估待后续研究。

---

## 507. Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities

**arXiv ID:** 2601.06913 | [PDF](https://arxiv.org/pdf/2601.06913v1)

**作者:** Taehyun Hwang `[一作]` (Seoul National University), Min-hwan Oh `[通讯]` (Seoul National University)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5006205893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多项式逻辑（MNL）上下文强盗问题，提出了一种计算上高效的算法，旨在处理非线性参数效用函数的序列选择问题。

**💡 创新点**

提出的算法是首个在非线性效用情况下，能够在计算上可行且证明能达到√(T)的后悔界限的方法。

**🔧 技术方法**

使用了基于上置信界（UCB）原则的算法，专门设计用于处理非线性参数效用函数，包括神经网络建模的效用函数。

**📊 数据集**

使用了模拟的上下文特征数据集，具体的上下文分布包括高斯分布和均匀分布。

**📈 对比分析**

与现有方法相比，提出的算法在非线性效用场景中表现出色，能够在可实现和模型错误指定的情况下保持稳健的性能，后悔界限为√(T)，而其他方法在非线性情况下性能显著下降。

**⚠️ 局限性**

算法的局限性在于后悔界限依赖于问题相关的实例因子κ，减少或消除这种依赖可能需要新的分析技术。

---

## 508. Santa Clara 3D: Digital Reconstruction and Storytelling of a Francoist Concentration Camp

**arXiv ID:** 2601.06902 | [PDF](https://arxiv.org/pdf/2601.06902v1)

**作者:** Stinne Zacho `[一作]` (University of Southern Denmark), Stefan Jänicke `[通讯]` (University of Southern Denmark)

**通讯引用:** 1053 | [OpenAlex ID](https://openalex.org/A5037571749)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套低成本数字重建与互动叙事平台，展示萨拉·克拉尔康集中营的历史演变，并通过线上交互式地图与时间轴实现多层级体验。

**💡 创新点**

创新点包括：1）结合参与式可视化设计与作者-读者混合叙事，赋予用户自由探索空间；2）利用SketchUp、Polycam、Three.js等开源工具构建低成本3D重建与360°影像；3）通过灰度纹理与色彩对比直观表现已消失建筑；4）提供可扩展的低成本管线，可迁移至其他被遗忘的历史遗址。

**🔧 技术方法**

技术：SketchUp（3D建模）、Polycam（低成本360°摄影）、Three.js（WebGL渲染）、MapLibre（交互地图）、GeoJSON（标记管理）、HTML/CSS/JavaScript（前端实现）。

**📊 数据集**

数据集：历史档案（地图、建筑图纸、照片）、现场采集的360°照片（Polycam）、受害者证词与专家访谈文本、实地考察获得的现场测量与照片。

**📈 对比分析**

方法未与传统数字遗产平台做系统对比，主要通过专家评估与用户反馈验证可用性；性能方面强调低成本与跨设备可访问性，未来可通过模型压缩、代码优化提升加载速度。

**⚠️ 局限性**

局限性：①档案与史料缺失导致重建依据有限，存在推测性；②时间与资源限制未完成完整评估与多语言支持；③平台目前主要在桌面/手机上运行，移动端交互体验尚待优化；④受访者与专家沟通周期受限，导致某些细节不够完整。

---

## 509. MemTrust: A Zero-Trust Architecture for Unified AI Memory System

**arXiv ID:** 2601.07004 | [PDF](https://arxiv.org/pdf/2601.07004v1)

**作者:** Xing Zhou `[一作]` (Independent Researcher), Kisson Lin `[通讯]` (Supermem AI Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 MemTrust——一种基于可信执行环境（TEE）的零信任统一 AI 记忆系统架构，支持跨代理协作、跨工具工作流、可随设备迁移的安全上下文共享。

**💡 创新点**

创新点包括：① 5 层功能抽象（存储、提取、学习、检索、治理）与 TEE 结合的零信任安全框架；② RA‑TLS 与硬件 attestation 绑定的通信协议；③ 在检索层采用隐蔽访问模式与噪声注入实现侧信道防护；④ 采用 OPA‑Wasm 的可编程访问控制与不可篡改审计日志；⑤ 支持多种 TEE 技术（AMD SEV‑SNP、Intel SGX/TDX、AWS Nitro、ARM CCA）的硬件无关实现。

**🔧 技术方法**

技术细节：使用 AMD SEV‑SNP（以及可迁移到 TDX/Nitro）的虚拟机级隔离；在 TEE 内部实现 AES‑256‑GCM 加密、Merkle 树完整性、RA‑TLS、HNSW + 噪声的向量检索、图数据库的隐蔽存储；使用 Rust + Python 组合、Gramine 轻量化 Linux、OPA‑Wasm、JWT‑OIDC 绑定 attestation、加密密钥层级（HYOK + TEE‑sealed）以及 GPU‑TEE 加速（NVIDIA H100）。

**📊 数据集**

评估数据集包括：10,000 篇文档、50,000 封邮件、1,000,000 条知识三元组，覆盖文本、结构化和多模态信息，模拟企业级实际使用场景。

**📈 对比分析**

与传统未加密记忆系统对比，MemTrust 在企业级负载下的性能开销 <20%（SEV‑SNP）或 5–15%（GPU‑TEE），QPS 维持在 70–85% 的安全基线；通过基准实验验证安全性（完整性、保密性、可审计性）与可用性（延迟、吞吐量）均满足大多数业务需求。

**⚠️ 局限性**

局限性包括：① TEE 内存和 I/O 带宽受限，无法高并发处理极大向量索引；② 开发成本高，尤其是多 TEE 兼容层和 RA‑TLS 的集成；③ 侧信道攻击（如时序分析）仍需进一步强化；④ 在跨云迁移时仍需手动配置，未完全实现零信任迁移；⑤ 对 GPU‑TEE 的依赖导致在不支持的硬件上性能下降。

---

## 510. Zer0n: An AI-Assisted Vulnerability Discovery and Blockchain-Backed Integrity Framework

**arXiv ID:** 2601.07019 | [PDF](https://arxiv.org/pdf/2601.07019v1)

**作者:** Harshil Parmar `[一作]` (Parul University), Priyank Panchal `[通讯]` (Parul University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Zer0n 框架，将 Gemini 2.0 Pro 的漏洞检测结果通过 Avalanche C-Chain 记录哈希，实现链上不可篡改的审计日志，解决 AI 漏洞报告缺乏可验证来源的问题。

**💡 创新点**

创新点在于：① 将 LLM 推理与区块链完整性验证解耦，形成高效离链执行与轻量级链上日志的混合架构；② 通过 Solidity 合约实现对报告哈希的不可变存储，提供即时的篡改检测；③ 在保持检测准确率的同时，将区块链交互的额外延迟降至约 22.9%。

**🔧 技术方法**

核心技术包括：Gemini 2.0 Pro LLM、Avalanche C-Chain 区块链、Solidity 智能合约、Node.js 后端、React 前端、SHA‑256 哈希、Web3.py 交互。

**📊 数据集**

使用了 500 个 Web 应用端点和 100 条智能合约样本（来自 SWC Registry 与 OWASP Juice Shop）作为评估数据集。

**📈 对比分析**

与 Slither、Mythril、AICyberChain 等传统工具比较，Zer0n 在准确率、召回率方面达到 80% 的 F1 分数，额外的链上日志开销仅为 22.9%，交易确认延迟约 14–16 秒，验证时间低于 100 ms，整体性能可接受。

**⚠️ 局限性**

局限性包括：① 仅评估了原型阶段的实验，缺乏大规模部署验证；② 区块链交互仍带来网络确认延迟，可能不适用于实时安全响应；③ 未实现自动修复或漏洞利用功能；④ 对区块链共识层攻击不具备防御能力。

---

## 511. Lexicalized Constituency Parsing for Middle Dutch: Low-resource Training and Cross-Domain Generalization

**arXiv ID:** 2601.07008 | [PDF](https://arxiv.org/pdf/2601.07008v1)

**作者:** Yiming Liang `[一作]` (Ghent University), Fang Zhao `[通讯]` (University of Paris Cité)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于Transformer的Benepar模型，构建了中世纪荷兰语的句法分析器，并系统评估了多源辅助语言和领域适应策略。

**💡 创新点**

创新点在于证明了与目标语在时间与地理上更接近的辅助语言能显著提升低资源语的成分句法解析；同时提出了多域对抗特征分离与多源动态匹配两种适配方法。

**🔧 技术方法**

采用了Benepar（基于自注意力的跨度式解析）与历史荷兰语BERT做词表征；实验中还引入了POS辅助任务、联合训练与域特征分离模块。

**📊 数据集**

实验数据主要来自四篇中世纪荷兰语文本（Etstoel、CRM14、Tafel、Trappen），以及一系列历史/现代语言的Penn式树库作为辅助语言。

**📈 对比分析**

与基准统计PCFG解析器（Bikel）相比，Benepar在零样本下即能达到约58–76 F1，少量（10–100）新域样例即可将性能提升至70以上；特征分离在样本≥200时可进一步提升至≈75 F1。

**⚠️ 局限性**

局限在于未利用无标签的中世纪荷兰语语料进行持续预训练或半监督学习，且对极低资源新域（如CRM）时仍表现欠佳，需进一步探索自训练与多语种联合目标。

---

## 512. Spatial Multi-Task Learning for Breast Cancer Molecular Subtype Prediction from Single-Phase DCE-MRI

**arXiv ID:** 2601.07001 | [PDF](https://arxiv.org/pdf/2601.07001v1)

**作者:** Sen Zeng `[一作]` (Tsinghua University), Yang Liu `[通讯]` (KCL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种基于单相DCE‑MRI的空间多任务学习框架，用于预测乳腺癌的四个分子标志物（ER、PR、HER2、Ki‑67）进而确定分子亚型。

**💡 创新点**

创新点在于：①首次将多尺度空间注意力与解剖学ROI加权相结合，在单相图像中补偿缺失的时间信息；②通过多任务学习共享特征并挖掘标志物间的生物学关联，提升预测精度；③提供可解释的注意力可视化。

**🔧 技术方法**

技术上采用深度卷积特征提取网络、多尺度空间注意力模块、ROI加权池化、任务特定分类/回归头，并使用交叉熵+MSE联合损失训练。

**📊 数据集**

使用了来自三中心的960例多中心DCE‑MRI数据，内部886例按7:1:2划分，外部74例五折交叉验证。

**📈 对比分析**

与传统放射组学、单任务深度学习、逻辑回归、随机森林、SVM、MLP等基线相比，ER、PR、HER2的AUC分别为0.893、0.824、0.857，平均AUC0.858，Ki‑67 MAE 8.2%，均显著优于基线，性能提升超过10%。

**⚠️ 局限性**

局限在于：仅验证了三中心数据，缺乏跨地区多厂家泛化评估；对罕见亚型的样本不足；未整合基因组学信息；以及缺乏真实临床工作流中的部署与评估。

---

## 513. UETQuintet at BioCreative IX - MedHopQA: Enhancing Biomedical QA with Selective Multi-hop Reasoning and Contextual Retrieval

**arXiv ID:** 2601.06974 | [PDF](https://arxiv.org/pdf/2601.06974v1)

**作者:** Quoc-An Nguyen `[一作]` (Vietnam National University), Hoang-Quynh Le `[通讯]` (Vietnam National University)

**通讯引用:** 1712 | [OpenAlex ID](https://openalex.org/A5057642796)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种可动态区分直接与顺序问答的框架，针对顺序问题通过LLM分解成子问题并逐步检索与回答；直接问题则直接生成答案以提高效率。

**💡 创新点**

①使用轻量级机器学习模型（随机森林+XGBoost+Logistic回归堆叠）先判别问题类型，减少误拆；②多源检索（搜索引擎+维基百科TF‑IDF）与Wiki正则化结合，为LLM提供精准上下文；③针对不同问题类型（是非与WH）采用定制检索策略。

**🔧 技术方法**

多源检索增强生成（RAG）+in‑context学习+LLM（ChatGPT‑4/Claude）+堆叠式分类器+Wiki正则化。

**📊 数据集**

MedHopQA（BioCreative IX共享任务）数据集，包含1,000道测试题。

**📈 对比分析**

与排行榜对比，多轮实验（Run 1–5）逐步提升，Run 5获得Exact Match 0.840、Concept Level 0.863，排名第二。

**⚠️ 局限性**

对极端长句或极少量样本仍易产生误检；LLM在多步推理时可能出现hallucination；整体仍依赖外部检索与大模型，推理成本高。

---

## 514. LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents

**arXiv ID:** 2601.06973 | [PDF](https://arxiv.org/pdf/2601.06973v1)

**作者:** Davide Baldelli `[一作]` (Mila – Quebec AI Institute), Sarath Chandar `[通讯]` (Chandar Research Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在需要保密内部状态的交互任务中的局限性，并提出了一种基于文本工作记忆的私有工作记忆架构，能够在对话中持续保持和更新隐藏状态；

**💡 创新点**

创新点在于首次定义并理论证明了“私有状态交互任务”（PSIT）——即需要生成并保持私密状态的任务；证明公有聊天模型无法同时满足保密与一致性；并提出三种可配置的私有工作记忆更新策略（Overwrite、Append/Replace、Patch/Replace），在实验中实现了与 Private CoT 相近的自一致性性能；

**🔧 技术方法**

技术上使用了LLM工具调用、LangGraph框架、文本工作记忆块（以 <memory> 标签注入）以及三种更新策略；对比了四种外部检索式记忆基线（Mem0、A-Mem、LightMem、MemoryOS）以及无状态LLM；

**📊 数据集**

实验数据集包括 Hangman 游戏（词典来自 Wordfreq）和 Diagnosis Simulator（基于 DDXPlus 诊断数据集）；

**📈 对比分析**

通过自一致性测试协议（SCT）将各方法在 GPT‑OSS 20B/120B 与 Qwen3 32B/235B 上的表现进行对比；私有工作记忆在自一致性准确率上与 Private CoT 相近，远超无状态LLM及四种检索式记忆基线；同时保持的 token 开销比 Private CoT 小约 10 倍，展示了更高效的隐私状态维护；

**⚠️ 局限性**

局限性包括：仅评估了短期游戏类任务，未验证在更长或多工具的复杂交互中的表现；结果对模型规模和实现细节高度敏感；私有工作记忆的引入降低了模型推理过程的可解释性和透明度。

---

## 515. RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction

**arXiv ID:** 2601.06966 | [PDF](https://arxiv.org/pdf/2601.06966v1)

**作者:** Haonan Bian `[一作]` (Xidian University), Ronghao Chen `[通讯]` (Peking University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5109632049)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究长周期项目导向交互中的记忆系统，并提出了 RealMem 基准，用来评估代理在多会话、持续项目中的记忆与推理能力。

**💡 创新点**

创新点在于：①引入“主动对齐”和“项目状态记忆”两个评估维度，聚焦真实项目的持续交互；②构建三阶段合成管线（项目基础构建、多代理对话生成、记忆与日程管理），实现动态记忆演化的模拟；③通过多会话自然用户查询，突破传统基准仅关注孤立问答的局限。

**🔧 技术方法**

技术方法包括：多代理对话生成（User Agent 与 Assistant Agent）；项目骨架（蓝图、事件、会话摘要）构建；记忆提取、去重与日程同步的自动化管线；使用 Gemini 2.5 与 GPT‑4o‑mini / GPT‑4o 进行记忆提取与评估，并用 LLM 进行语义判断（MemRecall、MemHelpful）。

**📊 数据集**

使用 RealMem 数据集：约 2,000 条跨会话对话，涵盖 11 个项目场景，体现多任务、动态进度和自然查询。

**📈 对比分析**

与 Mem0、A‑mem、MemoryOS、Graph Memory 等现有记忆系统进行 Recall@k、NDCG、QA Score、MemRecall、MemHelpful 等指标比较。结果显示：MemoryOS 在动态信息和主动对齐上表现最佳；Graph Memory 在实体关系与时间推理上更优；所有方法与 oracle 仍存在显著差距，说明真实项目记忆需求仍未被充分满足。

**⚠️ 局限性**

局限性：①依赖 Gemini 2.5 系列模型，复现成本高；②未包含工具使用等更复杂任务；③数据收集过程对人工标注和模型生成的可复现性与公平性有一定挑战。

---

## 516. Symphonym: Universal Phonetic Embeddings for Cross-Script Toponym Matching via Teacher-Student Distillation

**arXiv ID:** 2601.06932 | [PDF](https://arxiv.org/pdf/2601.06932v1)

**作者:** Stephen Gadd `[一作]` `[通讯]` (University of London / University of Pittsburgh), Stephen Gadd (University of London / University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Symphonym，一个通过字符直接生成统一语音嵌入的跨写字系统，用于历史地名匹配。

**💡 创新点**

创新点在于教师‑学生知识蒸馏框架，将基于 IPA/PanPhon 的语音知识迁移到不依赖语音资源的字符编码模型，并通过三阶段训练实现跨写字、跨语言的泛化。

**🔧 技术方法**

使用的技术包括 Epitran（G2P）、PanPhon（音位特征）、双向 LSTM+自注意力编码、triplet loss、distillation loss、硬负样本挖掘以及 Elasticsearch HNSW 近邻检索。

**📊 数据集**

训练数据来自 GeoNames、Wikidata 与 Getty TGN 的 57.6 M 地名记录，构成约 5.1 M 的正负对；评测使用 MEHDIE 以色列-阿拉伯语中世纪地名基准。

**📈 对比分析**

在 MEHDIE 基准上，Symphonym 的 R@1 为 87.5%（MRA = 0.923），明显优于 Levenshtein（81.5%）和 Jaro‑Winkler（78.5%）的基线。

**⚠️ 局限性**

局限性包括训练数据偏倚（主要是已标注地名）、缺乏声调建模、Epitran 仅覆盖约 50% 的语言导致教师覆盖不足，以及模型对同音异义词的区分仍需依赖地理上下文。

---

## 517. RenderFlow: Single-Step Neural Rendering via Flow Matching

**arXiv ID:** 2601.06928 | [PDF](https://arxiv.org/pdf/2601.06928v1)

**作者:** Shenghao Zhang `[一作]` (Disney Research), Yang Zhang `[通讯]` (Disney Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 RenderFlow，一种基于单步条件流匹配的渲染框架，能够实现实时、物理一致的正向渲染以及通过轻量适配器完成逆向图像分解。

**💡 创新点**

创新点在于：①将渲染过程重新定义为从 albedo 到完整着色图像的单步条件流生成，消除扩散模型的迭代采样与随机性；②引入稀疏关键帧引导机制，以离线路径追踪结果加强物理精度；③利用适配器实现同一网络同时支持正向渲染与逆向分解，形成统一框架。

**🔧 技术方法**

使用技术包括：流匹配（Flow Matching/Bridge Matching）在潜在空间训练；预训练视频扩散模型 DiT 作为先验；VAE 编码/解码；RoPE 位置编码；LoRA 参数高效适配；关键帧注意力与环境图适配器。

**📊 数据集**

数据集：基于 Unreal Engine 5 生成的 130k+ 帧（30k 艺术环境、100k 过程合成场景），每帧含路径追踪参考图及对应 G‑buffer 与 HDR 环境图。

**📈 对比分析**

与 RGB‑X、DiffusionRenderer 以及传统路径追踪/IBL/Deferred 渲染基线对比；在 512×512 分辨率下，RenderFlow 的 PSNR 达 24.2–26.7、SSIM 0.874–0.883、LPIPS 0.101–0.113，单帧推理约 0.19 s（≈10×快于 DiffusionRenderer、>10×快于 RGB‑X），同时保持零方差的确定性输出。

**⚠️ 局限性**

局限性：依赖于预训练扩散模型与 VAE，难以直接处理极端几何与光照变化；关键帧引导需要额外离线路径追踪；在极端全局光照与复杂材质下仍可能出现误差，且对极端高动态范围光照的建模能力有限。

---

## 518. Belief in False Information: A Human-Centered Security Risk in Sociotechnical Systems

**arXiv ID:** 2601.07016 | [PDF](https://arxiv.org/pdf/2601.07016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 519. TreePS-RAG: Tree-based Process Supervision for Reinforcement Learning in Agentic RAG

**arXiv ID:** 2601.06922 | [PDF](https://arxiv.org/pdf/2601.06922v1)

**作者:** Tianhua Zhang `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9305 | [OpenAlex ID](https://openalex.org/A5019458385)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TreePS‑RAG，一种在线树形强化学习框架，用于 agentic RAG 通过步骤级信用分配提升答案质量。

**💡 创新点**

创新点在于将 agentic RAG 的推理过程建模为树形 Rollout，利用子叶结果的蒙特卡洛估计为中间步骤提供无标注过程监督，并在线构建树并采用相似度剪枝保留多样性。

**🔧 技术方法**

使用技术包括 ReAct 交互式推理、强化学习（GRPO/PPO）、树形 Rollout、蒙特卡洛价值估计、Jaccard 相似度剪枝、过程优势（step‑level advantage）计算。

**📊 数据集**

实验数据集涵盖七个 QA 基准：NQ、TriviaQA、PopQA、HotpotQA、2WikiMultihopQA、Bamboogle、MuSiQue，使用多种 LLM（Qwen2.5‑3B‑Instruct、Qwen2.5‑7B‑Instruct、Qwen3‑8B、Qwen3‑4B‑Instruct‑2507）。

**📈 对比分析**

与 Search‑R1、StepSearch、ReasonRAG、GiGPO 等基线对比，TreePS‑RAG 在七个基准上平均提升 4–5% EM，并持续在各模型尺度上优于对手。

**⚠️ 局限性**

局限性包括仅在中小型模型和英文文本 QA 上验证；大模型训练成本高；未覆盖多语言或多模态任务；树形 Rollout 结构实现复杂，需进一步优化系统效率。

---

## 520. UDPNet: Unleashing Depth-based Priors for Robust Image Dehazing

**arXiv ID:** 2601.06909 | [PDF](https://arxiv.org/pdf/2601.06909v1)

**作者:** Zengyuan Zuo `[一作]` (Harbin Institute of Technology), Xianming Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 6559 | [OpenAlex ID](https://openalex.org/A5100654390)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了 UDPNet，一种在图像去雾任务中通过引入大规模预训练深度估计模型 DepthAnything V2 的深度先验来增强网络性能的通用框架。

**💡 创新点**

创新点在于设计了两大模块：Depth‑Guided Attention Module (DGAM) 用深度引导的通道注意力自适应调制特征；Depth Prior Fusion Module (DPFM) 通过重叠窗口多头交叉注意力实现多尺度深度信息与图像特征的高效融合，形成轻量且可插拔的深度先验增强方案。

**🔧 技术方法**

技术手段包括：深度先验提取（DepthAnything V2）、卷积+实例归一化+GELU 预处理、深度引导通道注意力、重叠窗口交叉注意力、残差连接与多尺度 U‑网络结构、以及多尺度域损失（空间域与频域）。

**📊 数据集**

使用了多种日间/夜间与真实世界去雾基准：RESIDE（SOTS、ITS、OTS）、Haze4K、NHR、GTA5、Dense‑Haze、NH‑HAZE、SateHaze1k，以及通用图像恢复的五任务集合（去噪、除雨、去雾、去模糊、低照度提升）。

**📈 对比分析**

与多种SOTA方法（如FSNet、ConvIR、MB‑TaylorFormer、C2PNet、FocalNet等）比较，UDPNet 在 SOTS、Haze4K、NHR、GTA5、Dense‑Haze、NH‑HAZE、SateHaze1k 上均实现显著 PSNR/SSIM 提升，日间去雾平均提升约0.8~1.2 dB，夜间去雾提升约1.2~1.3 dB，甚至在所有‑一‑模型恢复任务中也取得最高或接近最高的指标。

**⚠️ 局限性**

主要局限包括：对深度估计的依赖，若输入图像噪声大或深度估计误差明显，可能导致结构失真；外部深度模型的推理开销增加推理时间；在极端雾化或低光照极端情况下，深度先验的鲁棒性仍有待提升。

---

## 521. Fine-grained Verbal Attack Detection via a Hierarchical Divide-and-Conquer Framework

**arXiv ID:** 2601.06907 | [PDF](https://arxiv.org/pdf/2601.06907v1)

**作者:** Quan Zheng `[一作]` (Beijing Normal University), Yan Song `[通讯]` (University of Science and Technology of China)

**通讯引用:** 29921 | [OpenAlex ID](https://openalex.org/A5013100135)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了分层攻击评论检测数据集HACD，并设计了基于层级拆分与细粒度的攻击识别框架。

**💡 创新点**

创新点在于利用层级回复结构和时序信息拆分任务，采用轻量级模型分工，实现比大模型更高效的攻击检测。

**🔧 技术方法**

技术包括LLM辅助标注、基于层级+时间的上下文选择、四个轻量级模型的显式/隐式检测与分析模块。

**📊 数据集**

使用了自建的中文社交媒体树形评论数据集HACD，并结合Catalan的InToxiCat和英文的Hate Speech数据进行跨语种验证。

**📈 对比分析**

与零样本、单模型基线对比，训练后的分层框架在多任务上显著提升精度，24B模型达到最高All_in_One_Accuracy≈0.52，且约4B参数的轻量级模型已能逼近大型模型性能。

**⚠️ 局限性**

限制包括对中文语料的依赖、缺乏多模态信号与用户行为信息，以及在极端长对话或少数族群表达时仍可能出现误检。

---

## 522. MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning

**arXiv ID:** 2601.07005 | [PDF](https://arxiv.org/pdf/2601.07005v1)

**作者:** Jianbo Yu `[一作]` (Southeast University), Wanyuan Wang `[通讯]` (Southeast University)

**通讯引用:** 678 | [OpenAlex ID](https://openalex.org/A5054650835)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MicLog框架，利用进化的元学习+上下文学习对日志进行高效精准的解析。

**💡 创新点**

创新点包括ProgMeta-ICL逐步从0-shot到k-shot的元学习训练、加权DBSCAN采样以及多级缓存+BM25示例选择的高效推理策略。

**🔧 技术方法**

核心技术为大语言模型(Qwen-2.5-3B)的元学习、加权DBSCAN采样、BM25检索、多级LRU+Pattern缓存以及ProgMeta-ICL。

**📊 数据集**

在公开的Loghub-2.0（14个系统日志数据集）上进行评估。

**📈 对比分析**

与Drain、Brain、AdaParser等基线相比，MicLog平均解析准确率97.6%，比AdaParser提升10.3%，且总解析时间降低42.4%。

**⚠️ 局限性**

局限性在于仍依赖LLM推理成本，且在极端异构日志分布下的泛化仍有限。

---

## 523. Can Textual Reasoning Improve the Performance of MLLMs on Fine-grained Visual Classification?

**arXiv ID:** 2601.06993 | [PDF](https://arxiv.org/pdf/2601.06993v1)

**作者:** Jie Zhu `[一作]` (Michigan State University), Xiaoming Liu `[通讯]` (Michigan State University)

**通讯引用:** 20406 | [OpenAlex ID](https://openalex.org/A5100409052)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对多模态大型语言模型在细粒度视觉分类任务中的链式思考效应进行系统评估，并提出基于奖励归一化的ReFine-RFT框架以限制思考长度并提升准确率。

**💡 创新点**

提出“思考成本”概念并证明更长的链式推理会降低FGVC准确率，同时设计多奖励归一化模块和集成奖励来稳定训练并控制思考长度。

**🔧 技术方法**

使用强化学习的GRPO、LoRA参数高效微调、多奖励归一化、集成奖励（格式、分类、思考长度、LLM基准奖励、嵌入相似度）以及ReFine-RFT框架。

**📊 数据集**

在四个主流FGVC基准上评测：FGVC-Aircraft、Stanford-Cars、Flowers-102、Oxford-Pets。

**📈 对比分析**

相较于SFT、Visual-RFT等基线，ReFine-RFT在所有四个数据集上均取得至少2-3个百分点的提升，最终达到state‑of‑the‑art水平。

**⚠️ 局限性**

仅在有限的4-shot设置下验证，未针对更大规模多模态任务或对不同模型规模的泛化进行深入探讨，且仍依赖强化学习的采样效率和奖励设计。

---

## 524. FinCARDS: Card-Based Analyst Reranking for Financial Document Question Answering

**arXiv ID:** 2601.06992 | [PDF](https://arxiv.org/pdf/2601.06992v1)

**作者:** Yixi Zhou `[一作]` (ShanghaiTech University), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FinCards 结构化重排框架，用于在长公司备案文件（10‑K/10‑Q）中进行财务问答的内部检索与证据排序。

**💡 创新点**

创新点在于将文档块抽象为可审计的“卡片”，将查询映射为结构化意图，并采用三阶段锦标赛式重排（筛选、全局排序、稳定化），严格满足实体、指标、期间和数值约束，实现可解释且稳定的排名。

**🔧 技术方法**

使用结构化卡片抽象（字段提取）、语义匹配与屏蔽、LLM 零样本锦标赛重排（分组、随机重组、Borda 聚合）、BM25 召回、确定性解码与 JSON 校验等技术。

**📊 数据集**

评估基于 FinAgentBench（美国 SEC 10‑K/10‑Q 文档）的内部检索数据集。

**📈 对比分析**

与传统 BM25、零样本 LLM 重排等基线比较，采用 nDCG@10、MAP@10、MRR@10 评价。FinCards 在 nDCG@10 提升约27+点、MRR@10 提升近20点，整体优于基线，并将候选集规模从 100 降至 25。

**⚠️ 局限性**

局限性包括：多阶段导致多次 LLM 调用，计算成本较高；仅针对单文档检索，跨文档场景尚未验证；对提示设计和 schema 选择敏感，需要进一步鲁棒性研究。

---

## 525. Categorize Early, Integrate Late: Divergent Processing Strategies in Automatic Speech Recognition

**arXiv ID:** 2601.06972 | [PDF](https://arxiv.org/pdf/2601.06972v1)

**作者:** Nathan Roll `[一作]` (Stanford University), Dan Jurafsky `[通讯]` (Stanford University)

**通讯引用:** 33701 | [OpenAlex ID](https://openalex.org/A5087088138)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了“Architectural Fingerprinting”框架，通过线性探针系统性比较 Transformer 与 Conformer 在语音表示中的层级差异，揭示两类架构在信息提取时机上的根本不同。

**💡 创新点**

创新点在于：①将架构视为主控因素而非仅关注规模或数据；②利用峰值层位置（Peak Position）等指标量化信息线性可访问性；③构建跨 24 个预训练模型的统一对比实验，首次展示不同架构在相同性能下的“早分类”与“晚整合”处理策略；④通过逻辑回归可准确识别模型族，证明可形成稳定的“指纹”。

**🔧 技术方法**

主要技术包括：线性探针（Logistic/Linear 回归）提取声学、人口统计、音素、口音、持续时间等五类特征；标准化深度索引；峰值位置、峰值强度与位置差异三指标评估；t 检验、线性回归与逻辑回归分类器等统计分析；留一交叉验证和 AUC 评估。

**📊 数据集**

使用 7 个多样化语音语料库共 50,000+ 句子，涵盖母语与非母语、不同口音、录音环境及临床样本：L2-ARCTIC、CMU ARCTIC、Common Voice、Speech Accent Archive、ALLSSTAR、Cambridge Assessment、SANDI。

**📈 对比分析**

对比方法：在相同预训练模型组（17 Transformer + 7 Conformer）上，使用统一探针评估每层信息可访问性，并对峰值层位置进行两样本 t 检验和线性回归控制参数规模；构建分类器验证指纹可区分度。结果显示：Conformer 在 16%–21% 的网络深度就已显著解码性别与音素，而 Transformer 需到 49%–57% 深度才出现类似信息；此外，分类器 AUC 达 0.88，证明架构指纹显著。

**⚠️ 局限性**

限制包括：①线性探针仅衡量线性可访问性，无法区分信息何时真正出现；②仅评估声学至音素层级，未覆盖词语或句子层面；③未包含 transducer 或 state‑space 等模型；④样本量有限，置信区间较宽；⑤使用二元性别与粗糙口音标签，存在偏见与隐私风险；⑥结果为相关性，缺乏因果验证。

---

## 526. HAS-VQ: Hessian-Adaptive Sparse Vector Quantization for High-Fidelity LLM Compression

**arXiv ID:** 2601.06959 | [PDF](https://arxiv.org/pdf/2601.06959v1)

**作者:** Vladimer Khasia `[一作]` `[通讯]` (Independent Researcher), Vladimer Khasia (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Hessian适应的稀疏向量量化框架HAS-VQ，用来压缩小规模大型语言模型，能够在保持高精度的同时显著降低模型大小。

**💡 创新点**

创新点包括：① Hessian‑Masked Decoupling（使用二阶敏感度把高重要性离群点与主体分离）；② Residual Sparse Feedback（在敏感维度上恢复量化误差，实现近似无损压缩）；③ 将向量量化与稀疏残差相结合，突破了传统整数量化的均匀格子限制。

**🔧 技术方法**

使用技术包括：第二阶Fisher信息/海森矩阵近似、敏感度重要性度量、稀疏-稠密解耦、K‑Means向量量化、残差反馈机制以及对块级量化的稳健采样与死元恢复。

**📊 数据集**

实验主要基于SmolLM2‑1.7B模型，并在该模型的评测数据集（语言建模任务）上评估。

**📈 对比分析**

与INT4和FP16基线对比，HAS‑VQ Mid点在4.23 BPP时PPL从20.03下降到14.23，显著提升29%且存储比INT4小11%；HAS‑VQ High点在7.03 BPP时PPL仅比FP16差0.08，近似无损压缩实现2.3×存储缩减。

**⚠️ 局限性**

局限性在于需要计算海森矩阵近似和敏感度排序，计算成本相对较高；对极大模型的扩展性和对非Transformer结构的适用性仍需进一步验证。

---

## 527. Optimal Extended Formulations from Optimal Dynamic Programming Algorithms

**arXiv ID:** 2601.06947 | [PDF](https://arxiv.org/pdf/2601.06947v1)

**作者:** Mateus de Oliveira Oliveira `[一作]` (Informatics Institute, University of Bergen), Wim Van den Broeck `[通讯]` (Department of Computer and Systems Sciences, Stockholm University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种将解保持型动态规划算法（DP-core）与参数化扩展公式（extended formulation）相互映射的通用框架，并证明了两者在表格复杂度和多项式扩展大小之间的等价关系。

**💡 创新点**

创新点在于建立了从DP-core的表格复杂度到多项式扩展复杂度的精确上界（与下界相匹配），以及将扩展公式的无条件下界转化为对解保持型DP-core表格复杂度的无条件下界，填补了动态规划与线性规划在参数化复杂度理论中的空白。

**🔧 技术方法**

主要技术包括：①对树分解的DP-core的形式化与“解保持性”公理化；②构造与DP-core对应的T‑形树自动机并利用其状态宽度得到扩展公式；③利用树自动机理论与多面体几何证明扩展公式的维度与约束数；④结合ETH与已知的扩展公式下界来推导DP-core下界。

**📊 数据集**

论文未使用具体的实验数据集，而是以理论构造的图族（如用于独立集的最小扩展复杂度图族）来给出下界；对常见问题如独立集、支配集、Hamiltonian回路、割集、d‑着色等，给出了对应的树分解宽度与表格复杂度的参数化公式。

**📈 对比分析**

通过理论分析与ETH假设对比，作者证明了在参数化层面上多项式扩展的大小与DP-core表格复杂度呈指数级一致；对独立集、支配集、Hamiltonian回路等问题，所得到的上界与已知的下界（或ETH下界）在指数项上完全匹配，说明方法在这类问题上已是最优的。

**⚠️ 局限性**

局限性包括：①仅适用于解保持型DP-core，无法直接处理不保持解的快速算法；②在某些连通性问题（如Hamiltonian回路）中，得到的扩展公式对应的多面体不一定是所有解的凸包；③方法对DP-core的构造复杂度没有直接约束，理论上可以忽略构造时间，实际实现可能受限。

---

## 528. SketchJudge: A Diagnostic Benchmark for Grading Hand-drawn Diagrams with Multimodal Large Language Models

**arXiv ID:** 2601.06944 | [PDF](https://arxiv.org/pdf/2601.06944v1)

**作者:** Yuhang Su `[一作]` (Beijing Normal University), Hua Huang `[通讯]` (Beijing Normal University)

**通讯引用:** 7790 | [OpenAlex ID](https://openalex.org/A5022334521)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 SketchJudge 基准，用于评估多模态大语言模型在评判手绘 STEM 图形中的准确性和错误诊断能力。

**💡 创新点**

提供了针对手绘图形的细粒度错误分类，支持引用与非引用两种评分模式，并首次将评判者角色作为基准任务。

**🔧 技术方法**

采用零样本评估的多模态大语言模型，包括多种开源与闭源系统，结合二分类正确性评测和例子级 F1 细粒度错误识别。

**📊 数据集**

由 300 个问题、1,015 个学生手绘答案、四个 STEM 领域（几何、物理、图表、流程图）构成的数据集，含 23 种细粒度错误标签。

**📈 对比分析**

在 WithRef 与 NoRef 两种设定下对 16 种模型进行比较，闭源模型平均准确率约 78%–83%，人类评估 83%；错误类型识别最高约 61% 远低于人类 66%，显示模型仍有差距。

**⚠️ 局限性**

数据收集与错误分类依赖人工高成本、类别边界模糊；基准仅涵盖受控教育场景，缺乏对更广泛真实场景的鲁棒性评估。

---

## 529. mind_call: A Dataset for Mental Health Function Calling with Large Language Models

**arXiv ID:** 2601.06937 | [PDF](https://arxiv.org/pdf/2601.06937v1)

**作者:** Fozle Rabbi Shafi `[一作]` (Queen's University), Salimur Choudhury `[通讯]` (Queen's University)

**通讯引用:** 1127 | [OpenAlex ID](https://openalex.org/A5032858073)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向精神健康的可穿戴传感器数据调用的合成数据集。

**💡 创新点**

在数据集中引入了显式推理注释、时间归一化以及多样化的查询类型，提升了意图归属与时间推理的可解释性。

**🔧 技术方法**

利用大型语言模型进行数据生成、函数调用映射和推理注释，并采用标准化的健康数据 schema。

**📊 数据集**

数据集本身为合成数据，没有使用外部真实数据；公开在 Hugging Face 上。

**📈 对比分析**

文中未给出模型性能评估或与其他方法的对比，主要聚焦于数据集提供。

**⚠️ 局限性**

仅包含单函数调用、英语单一语料、合成样本、缺乏多步计划与多语言、多文化覆盖。

---

## 530. Active Learning Strategies for Efficient Machine-Learned Interatomic Potentials Across Diverse Material Systems

**arXiv ID:** 2601.06916 | [PDF](https://arxiv.org/pdf/2601.06916v1)

**作者:** Mohammed Azeez Khan `[一作]` (National Institute of Technology Warangal), Vijay Choyal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过在材料数据库中采用主动学习框架，迭代挑选最具信息量的样本以训练机器学习的原子势（MLIP），从而大幅减少需要的高成本DFT计算。

**💡 创新点**

创新点在于首次在四种不同化学体系（C、Si、Fe、Ti–O）上系统比较四种主动学习策略（随机、置信度、覆盖度、混合），并证明覆盖度采样在复杂体系中显著优于其他方法。

**🔧 技术方法**

技术包括基于特征向量的前馈神经网络集成、Query‑by‑Committee不确定性量化、k‑means聚类与远点细化的多样性采样，以及混合策略的加权组合。

**📊 数据集**

使用的数据集来自公开的Materials Project（约140k结构）和OQMD（约60万结构），每个体系分别抽取500（MP）+100（OQMD）个结构，并进行80/20拆分。

**📈 对比分析**

在5个随机种子下的统计检验显示，覆盖度采样在所有体系中均达到或优于随机基准，尤其在Ti–O体系中MAE降低10.9%（p=0.008），且整体训练耗时≤4 h、内存<8 GB。

**⚠️ 局限性**

局限在于仅使用了 17 维全局特征，未包含局部结构信息；模型仅针对成分能，未扩展到其他关键性质，且不涉及更先进的等变形网络或更细粒度的不确定性估计。

---

## 531. ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity A REM-Inspired System Design for Emergent Creative Ideation

**arXiv ID:** 2601.07121 | [PDF](https://arxiv.org/pdf/2601.07121v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 532. Towards Compositional Generalization in LLMs for Smart Contract Security: A Case Study on Reentrancy Vulnerabilities

**arXiv ID:** 2601.06914 | [PDF](https://arxiv.org/pdf/2601.06914v1)

**作者:** Ying Zhou `[一作]` (Artificial Intelligence), Xiao Zhang `[通讯]` (Blockchain and Privacy Computing)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于原子任务拆解与融合的后训练框架（CompFuse），以提升大型语言模型在智能合约重入漏洞检测中的泛化能力。

**💡 创新点**

创新点在于：①将重入漏洞拆解为四个线性独立的原子任务（外部调用、状态更新、数据依赖、执行顺序）并分别训练；②采用低秩适配器（LoRA）与自适应加权融合，保证各因子贡献平衡；③通过编译器验证与CFG/DFG增强的合成数据，实现结构化监督并满足理论的全秩性与连续性。

**🔧 技术方法**

使用了 Qwen2.5-Coder-14B-Instruct 作为主干模型，LoRA 适配器、任务感知门控网络、Jacobian 对齐损失、基于Sigmoid 的可微分风险权重，辅以 Slither、Mythril 等静态分析工具做对比。

**📊 数据集**

数据集包括三套约 2.5k 条的合成原子任务集（外部调用、依赖、顺序），以及 166 条已验证的漏洞/无漏洞合约和 31 条真实重入漏洞合约，所有合成样本均通过 Slither 编译验证并提取 CFG/DFG。

**📈 对比分析**

与传统静态分析器（Slither、Mythril、Securify、Sailfish、Smartian）以及内部非融合、冻结融合等基线相比，CompFuse 在合成数据上 F1/ACC 最高达 94.7%/98.2%，在真实合约上召回率 87.1%，比 Slither 提升约 20% 召回率，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：①合成数据规模有限，规则设计复杂；②方法目前仅针对重入漏洞，扩展到其他漏洞类型需重新设计因子拆解；③对极端复杂或非典型合约结构的泛化尚待验证。

---

## 533. Distributional Clarity: The Hidden Driver of RL-Friendliness in Large Language Models

**arXiv ID:** 2601.06911 | [PDF](https://arxiv.org/pdf/2601.06911v1)

**作者:** Shaoning Sun `[一作]` (Tsinghua University), Haifeng Wang `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究不同大模型在强化学习中的友好度差异，发现分布清晰度是关键因素，并提出基于 Silhouette 系数的重权重策略提升推理性能。

**💡 创新点**

创新点在于将分布清晰度定义为概率空间内的内聚与分离度，量化为 Silhouette 系数，并通过 Silhouette-Aware Reweighting 让低清晰度样本在 RL 训练中得到重点强化。

**🔧 技术方法**

使用 Silhouette 系数、分布重权重、强化学习算法 DAPO/GRPO 以及概率分布分析技术。

**📊 数据集**

实验数据集包括 AIME 2024/2025、MATH-500、AMC、Minerva、OlympiadBench 等数学推理基准。

**📈 对比分析**

与标准 DAPO 对比，提出方法在所有模型上均有提升，尤其对 RL 友好度低的 OctoThinker、Llama 在 AIME 24 上提升约 2–5 倍，平均提升 3–4%。

**⚠️ 局限性**

局限性包括需要足量采样才能估计 Silhouette 系数，且主要针对链式推理长度有限的场景，未覆盖超长推理或非 Chain-of-Thought 模式。

---

## 534. LINEture: novel signature cryptosystem

**arXiv ID:** 2601.07071 | [PDF](https://arxiv.org/pdf/2601.07071v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 535. Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework

**arXiv ID:** 2601.07122 | [PDF](https://arxiv.org/pdf/2601.07122v1)

**作者:** Yixiao Peng `[一作]` (State Key Laboratory of Mathematical Engineering and Advanced Computing), Yuling Liu `[通讯]` (Institute of Information Engineering)

**通讯引用:** 9687 | [OpenAlex ID](https://openalex.org/A5037034567)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种层次化的LLM+多智能体强化学习框架（CyberOps‑Bots），通过LLM进行全局语义感知与战术规划，下层RL专家执行局部防御动作，以提升云网络的韧性。

**💡 创新点**

创新点在于：①将大型语言模型与RL分层协同，既具备LLM的语义推理与可解释性，又保持RL的精确执行；②通过自然语言感知将网络状态抽象为文本，实现在网络结构与规模变更时无须重训练；③设计了长短期记忆与工具调用机制，实现人机交互（HITL）与多攻击策略的自适应；④采用异构RL专家的预训练，避免状态空间爆炸。

**🔧 技术方法**

技术手段包括：LLM（Qwen3‑8B）+ReAct规划+IPDRR感知+长短期记忆+工具调用；下层使用DQN训练的异构RL专家（Patch、Recover、Purge、Block）；仿真平台 Yawning Titan；基准 MARL 算法 IPPO、IQL、MAPPO、QMIX、VDN；以及多任务奖励与记忆检索机制。

**📊 数据集**

使用 AWS enterprise cloud 数据集（CSE/CIC）作为实验环境，包含 450 节点、6 子网、漏洞、业务连续性与攻击日志等信息。

**📈 对比分析**

与 IPPO、IQL、MAPPO、QMIX、VDN 等基准算法在四个动态场景（网络结构、规模、攻击策略、攻击强度）下对比。CyberOps‑Bots 在平均奖励、健康比例、跳起性能、网络可用率等指标上均优于基线，网络可用率提升 68.5%，跳起性能提升 34.7%，且不需重训练。

**⚠️ 局限性**

局限性：LLM 推理成本高、延时相对较大；集中式 LLM 架构在大规模云网络中可能面临扩展瓶颈；LLM 在数值计算与精准命令生成方面仍有限，未来需研究去中心化、工具学习与更高效的推理加速。

---

## 536. Few-shot Class-Incremental Learning via Generative Co-Memory Regularization

**arXiv ID:** 2601.07117 | [PDF](https://arxiv.org/pdf/2601.07117v1)

**作者:** Kexin Bao `[一作]` (Chinese Academy of Sciences), Shiming Ge `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 3530 | [OpenAlex ID](https://openalex.org/A5033254559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种生成式共记忆正则化方法（GCMR），用于少样本类增量学习（FSCIL）。

**💡 创新点**

创新点在于：① 采用生成域适配微调将预训练的 ViT 编码器同时进行自监督重建和监督分类，从而得到更具泛化与可迁移的特征；② 在增量阶段引入两种协同记忆（表示记忆和权重记忆），通过正则化约束实现灾难性遗忘与过拟合的双重抑制。

**🔧 技术方法**

核心技术包括：ViT 预训练 + MAE 自监督特征重建 + 两层分类头 + 共记忆正则化（表示记忆与权重记忆的协同更新）。

**📊 数据集**

实验数据集：MiniImageNet、CIFAR100（即 CIRAR100）和 CUB200。

**📈 对比分析**

与多种 SOTA 方法（如 C-FSCIL、NC-FSCIL、OrCo、LIMIT 等）对比，GCMR 在三大基准上均取得最高或接近最高的平均准确率：MiniImageNet +4% avg，CIFAR100 +2% avg，CUB200 +0.3% avg。

**⚠️ 局限性**

局限性：① 对类相似度高的细粒度任务（如 CUB200）区分能力略逊；② 对预训练数据分布差异敏感，需大规模预训练 ViT；③ 由于 ViT 参数量大，仍有模型规模与推理效率的挑战。

---

## 537. 3D Wavelet-Based Structural Priors for Controlled Diffusion in Whole-Body Low-Dose PET Denoising

**arXiv ID:** 2601.07093 | [PDF](https://arxiv.org/pdf/2601.07093v1)

**作者:** Peiyuan Jing `[一作]` (Zurich University of Applied Sciences), Javier Montoya `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

设计并评估了一种基于三维波形分解条件的ControlNet框架WCC‑Net，用于低剂量全身PET图像的去噪与结构保持。

**💡 创新点**

创新点在于将低频离散小波特征注入冻结的扩散模型的控制分支，实现结构先验的分离式指导，既保留生成能力又显著提升解剖一致性。

**🔧 技术方法**

采用的技术包括3D Denoising Diffusion Probabilistic Model (DDPM)、ControlNet、3D Haar小波分解以及零初始化卷积等。

**📊 数据集**

使用的数据集为公开的Ultra‑Low‑Dose PET（UDPET）挑战集，共377张18F‑FDG PET扫描，并在不同剂量级别（1/20、1/50、1/4）上进行评估。

**📈 对比分析**

通过与BM3D、NLM、3D U‑Net、REDCNN、EDCNN、GAN以及DDPM等基线在PSNR、SSIM、GMSD、NMAE等指标上进行比较，WCC‑Net在内部与外部测试集均取得最高PSNR/SSIM并显著降低结构失真。

**⚠️ 局限性**

局限性包括仅使用单尺度Haar小波、仅在单台扫描仪与单种示踪剂上验证，缺乏对多中心、多协议的泛化评估。

---

## 538. When Abundance Conceals Weakness: Knowledge Conflict in Multilingual Models

**arXiv ID:** 2601.07041 | [PDF](https://arxiv.org/pdf/2601.07041v1)

**作者:** Jiaqi Zhao `[一作]` (Harbin Institute of Technology), Jun Yu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 13210 | [OpenAlex ID](https://openalex.org/A5050817770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CLEAR 框架，用四个递进任务系统评估多语言 LLM 在内部记忆与外部证据冲突（跨语言知识冲突）下的决策与表现。

**💡 创新点**

创新点：① 将跨语言知识冲突拆解为四个层级场景；② 发现推理任务受资源规模驱动、实体事实任务受语言亲和力驱动的任务依赖决策二分；③ 揭示了“丰盈‑弱点”悖论，即高资源非拉丁语在推理中强势但在实体纠错中反而弱势。

**🔧 技术方法**

技术：构建 10 语言版本的 ConflictQA-PopQA 与 ConflictQA-StrategyQA，使用 LLM‑as‑judge（Gemini‑2.5‑Flash）进行输出判定；测量 Stubborn Rate、Persuasion Rate、Accuracy 等指标；通过跨语言检索、对齐和竞争实验模拟多源冲突。

**📊 数据集**

数据集：ConflictQA-PopQA（898 个实体事实问题）与 ConflictQA-StrategyQA（1000 个推理问题），分别翻译成 10 种语言（af, bn, de, fr, is, ja, sw, zh, tr, en）。

**📈 对比分析**

比较方法：与六款代表性 LLM（GPT‑4o‑mini, Gemini‑2.5‑Flash, Qwen3‑8B, Qwen3‑80B, Llama‑3.1‑8B, Aya‑Expanse‑8B）对照；通过 SR、PR、ACC 等指标表明：实体任务 SR 较低但 PR 高；推理任务 SR 较高；高资源非拉丁语在推理中表现卓越，而在实体任务中则表现较弱，体现语言资源与脚本亲和力的不同影响。

**⚠️ 局限性**

局限性：语言覆盖仅为 10 种，缺乏极低资源或非主流语言；数据以英文基准为翻译，缺少本土化多语言冲突实例；实验仅限 QA 任务，未扩展到对话、长文本生成等更复杂场景。

---

## 539. Task Arithmetic with Support Languages for Low-Resource ASR

**arXiv ID:** 2601.07038 | [PDF](https://arxiv.org/pdf/2601.07038v1)

**作者:** Emma Rafkin `[一作]` (Georgetown University), Xiulin Yang `[通讯]` (Georgetown University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

采用 Whisper 预训练模型，结合高资源相关语言的 LoRA 适配器或任务向量，通过线性组合（任务算术）提升低资源语言的自动语音识别性能。

**💡 创新点**

首次将任务算术与 LoRA 适配器相结合用于低资源 ASR，并利用基因相似的支持语言与 λ 参数调优，形成一种轻量级、多语言迁移的方法。

**🔧 技术方法**

Whisper 模型、LoRA 适配器、4‑bit 量化、早停、贝叶斯优化 λ、语音文本清洗、投票加权上采样、线性任务向量组合。

**📊 数据集**

Mozilla Common Voice 2025 共享任务的 21 低资源语言自发语料，支持语言与未见语言使用的 Common Voice 脚本语料；对投票得分进行上采样。

**📈 对比分析**

与任务基线（MMS 模型）以及未使用支持语言的 Whisper 模型比较，采用 WER 评估。实验显示加入支持语言后平均降低约 3.7% WER，部分语言提升至 -9.2%，但整体仍低于 MMS 基线，尤其在小模型上效果更显著。

**⚠️ 局限性**

计算资源不足导致未完成完整超参数搜索；λ 取值范围过大，最优值均 < 0.4；Whisper 的 30 秒上下文窗口限制影响长音频表现；支持语言选择与基准模型选择可能不最优。

---

## 540. Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers

**arXiv ID:** 2601.07036 | [PDF](https://arxiv.org/pdf/2601.07036v1)

**作者:** Wang Yang `[一作]` (Case Western Reserve University), Xiaotian Han `[通讯]` (Case Western Reserve University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5116337235)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Mid-Think 提示格式，通过结合 “Okay” 和 “</think>\n\n” 两个触发词实现无训练的中间预算推理，并将其应用于 RL 训练，显著提升推理准确率与训练效率。

**💡 创新点**

创新点在于发现推理行为被极少数触发词主导，利用这一“过拟合”现象设计 Mid-Think 形式实现训练‑free 的预算控制，并通过 RL 优化进一步提升性能。

**🔧 技术方法**

技术包括注意力分析识别触发词、无训练提示设计（Mid-Think）、预算控制实验、基于 GRPO 的 RL 训练、以及与固定-token、prompt-based 基线的对比实验。

**📊 数据集**

使用 Qwen3‑8B、Qwen3‑14B、DeepSeek‑7B、Qwen3‑32B 等模型，在 MATH500、AIME、GPQA 等公开数学与科学推理数据集上进行评估。

**📈 对比分析**

对比固定-token、prompt-based 预算控制以及标准 Think/No‑Think 模式，Mid‑Think 在相同或更低的平均生成长度下保持甚至超过固定预算的准确率；在 RL 训练中，Mid‑Think 训练时间缩短约 15%，AIME 从 69.8% 提升至 72.4%，GPQA 从 58.5% 提升至 61.1%。

**⚠️ 局限性**

限制在于 Mid‑Think 仅能实现中间预算的粗粒度控制，无法动态调节任意预算比例；且需预先识别模型中的触发词，若模型未出现类似过拟合特征则效果受限。

---

## 541. Codified Foreshadowing-Payoff Text Generation

**arXiv ID:** 2601.07033 | [PDF](https://arxiv.org/pdf/2601.07033v1)

**作者:** Longfei Yun `[一作]` (University of California), Jingbo Shang `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Codified Foreshadowing‑Payoff Generation（CFPG）框架，利用 Foreshadow‑Trigger‑Payoff（F‑T‑P）三元组显式建模故事中的长期承诺与兑现，形成可执行的因果状态循环；

**💡 创新点**

创新点在于将叙事连贯性转化为可执行的因果约束，通过显式的状态池和触发门控实现对故事推进的逻辑控制，显著提升了对前置伏笔的兑现能力；

**🔧 技术方法**

核心技术包括：F‑T‑P 结构化表示、可执行符号状态循环（Select‑Generate‑Update）、触发器判定函数、Payoff 识别与验证模块、以及注意力和因果显著性分析；

**📊 数据集**

使用 BookSum 语料库（长篇文学摘要）自动抽取 F‑T‑P 对，构建大型结构化伏笔–兑现数据集；

**📈 对比分析**

与基线的 Prompt、Foreshadow‑Aware Prompting、Foreshadow‑Similarity Context Refresh 等方法对比，在 Payoff 触发准确率、情节对齐度、误触发率等指标上显著优于基线（如 Payoff 触发准确率>0.96，误触发率降低31%，情节一致性提升43%）；

**⚠️ 局限性**

局限性包括：仅能处理文本显式的伏笔兑现关系，无法覆盖高度抽象或符号化叙事；依赖摘要级语料，可能无法反映完整文本的细节；以及对自动抽取 F‑T‑P 过程的错误或遗漏敏感。

---

## 542. Tight Analysis of Decentralized SGD: A Markov Chain Perspective

**arXiv ID:** 2601.07021 | [PDF](https://arxiv.org/pdf/2601.07021v1)

**作者:** Lucas Versini `[一作]` (CNRS), Aymeric Dieuleveut `[通讯]` (CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对去中心化随机梯度下降（DSGD）算法，分别在确定性和随机性两种情形下，使用马尔可夫链视角进行严格分析，推导出迭代收敛到稳态分布、偏差分解与方差表达式，并给出非渐近收敛速率；随后提出基于 Richardson–Romberg 外推的去中心化算法，以一次性消除首阶偏差并提升样本复杂度。

**💡 创新点**

创新点在于：① 将DSGD视为几何鞭形马尔可夫链，得到首阶偏差和方差的精确展开；② 证明方差首阶项与网络拓扑无关、随客户端数线性下降；③ 提出不需要梯度跟踪、也不需显式拓扑知识的 Richardson–Romberg 外推方法，实现首阶偏差消除和样本复杂度的二次改进。

**🔧 技术方法**

技术方法包括：马尔可夫链收敛性分析（Wasserstein 距离）、谱图理论、Taylor 展开与高阶项控制、随机梯度噪声的矩估计，以及 Richardson–Romberg 外推。

**📊 数据集**

实验使用二维 Logistic 回归数据，构造不同客户端的数据分布模拟异构，采用全连接、环形和四簇稀疏三种通信图进行验证。

**📈 对比分析**

与传统 DSGD、步长/2 以及 RR‑DSGD 进行比较，RR‑DSGD 在所有拓扑下均显著降低偏差，达到更低误差；在客户端数较多时展示线性加速；在噪声主导场景下性能优势更为明显。

**⚠️ 局限性**

局限性包括：需满足强凸光滑与噪声矩界定的严格假设；偏差和方差展开仅为首阶，高阶项在弱连接网络或大步长时影响不易控制；实验仅在合成小规模问题上验证，缺乏大规模真实数据的实证。

---

## 543. RSLCPP - Deterministic Simulations Using ROS 2

**arXiv ID:** 2601.07052 | [PDF](https://arxiv.org/pdf/2601.07052v1)

**作者:** Simon Sagmeister `[一作]` (Technical University of Munich), Markus Lienkamp `[通讯]` (Technical University of Munich)

**通讯引用:** 7194 | [OpenAlex ID](https://openalex.org/A5079718896)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种在ROS 2环境中实现确定性仿真的方法，使所有节点在单进程单线程的自定义事件循环内按固定顺序执行回调，从而获得硬件无关、可复现的仿真结果。

**💡 创新点**

核心创新点包括：①将节点组合到单进程，利用ROS 2的events executor实现FIFO执行；②采用模拟时钟并通过自定义事件循环控制时间推进，提供可调的延迟模型；③引入作业对象和动态节点加载，允许在不改动原有节点代码的情况下创建完整仿真。

**🔧 技术方法**

技术手段包括ROS 2节点组合、单线程事件驱动、events executor、模拟时钟、离散事件仿真、延迟调度、动态参数加载和C++实现的自定义事件循环。

**📊 数据集**

实验使用了两组数据：一是自建的Synthetic Benchmark系统；二是KITTI序列00的LiDAR Odometry benchmark（通过rosbag）。

**📈 对比分析**

将该框架与原生ROS 2在同一硬件上并行跑100次，对比最终哈希值和RMSE，结果显示RSLCPP下所有CPU（x86_64与ARM）均得到完全一致的输出，而原生ROS 2在部分CPU出现波动，ARM平台甚至无法得到有效结果。

**⚠️ 局限性**

局限性在于：①需要单线程执行，可能导致单个仿真耗时增大；②依赖ROS 2执行器实现，未来ROS 2升级需维护兼容性；③仅适用于使用模拟时钟的节点，使用壁钟的节点需重新编译；④实验验证为经验性，没有形式化的确定性证明。

---

## 544. ReinPool: Reinforcement Learning Pooling Multi-Vector Embeddings for Retrieval System

**arXiv ID:** 2601.07125 | [PDF](https://arxiv.org/pdf/2601.07125v1)

**作者:** Sungguk Cha `[一作]` (LG Uplus), Sangyeob Lee `[通讯]` (LG Uplus)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 ReinPool，一种利用强化学习动态筛选并池化多向量文档嵌入的框架；

**💡 创新点**

创新点在于将向量选择建模为强化学习任务，并通过逆检索与 NDCG 奖励直接优化检索性能，无需人工重要性标签；

**🔧 技术方法**

使用的技术包括 Transformer‑based 策略网络、逆检索训练策略、GRPO（Group Relative Policy Optimization）以及 NDCG 作为奖励函数；

**📊 数据集**

实验采用 Vidore V2 视觉文档检索基准，覆盖 ESG、ECO、ESGHL、BIO 四个子任务；

**📈 对比分析**

与全向量存储、均值/最大池化等基线对比，ReinPool 在三种视觉语言模型上实现了 746–1249 倍压缩，恢复 76–81% 的多向量检索性能，并比静态池化提升 22–33% 的 NDCG@3；

**⚠️ 局限性**

局限性包括在高维嵌入（如 3072 维）下压缩效果仍不理想，对极大文档或极稀疏向量的处理尚未充分评估。

---

## 545. Score-Based VAMP with Fisher-Information-Based Onsager Correction

**arXiv ID:** 2601.07095 | [PDF](https://arxiv.org/pdf/2601.07095v1)

**作者:** Tadashi Wadayama `[一作]` (Nagoya Institute of Technology), Takumi Takahashi `[通讯]` (University of Osaka)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出了基于分数（score）的VAMP框架（SC‑VAMP），通过学习后验分数来构造MMSE估计器，并用Fisher信息直接计算Onsager修正，完成高维逆问题的迭代求解。

**💡 创新点**

创新点在于将Onsager修正从显式雅可比导数改为仅依赖于条件分数的Fisher信息，实现无雅可比（Jacobian‑free）实现；同时将传统VAMP的解析式扩展到任意非线性、复杂先验，可通过数据驱动的分数匹配学习得分函数。

**🔧 技术方法**

主要技术包括：Tweedie公式、Stein同一性、信息‑理论关系（I–MMSE）、分数匹配（DSM）训练分数网络、随机正交/单元矩阵混合以保证退化性、以及基于mini‑batch的Fisher信息估计。

**📊 数据集**

实验使用合成信号：Bernoulli‑Gaussian稀疏先验和二维高相关高斯先验，配合随机右旋矩阵（RRI）测量矩阵；未使用公开图像或通信数据集。

**📈 对比分析**

与理论SE曲线及标准VAMP做对比。SC‑VAMP的MSE曲线与SE预言高度吻合，EXIT图显示两模块收敛至交点；在非线性/相关先验实验中，SC‑VAMP与SE误差差距约15%~17%，已显示出可观的恢复性能。

**⚠️ 局限性**

局限性包括：需要分数函数学习，若学习误差大会导致Onsager修正失准；对极高维或严重结构化非线性场景的收敛性和理论保证尚未完全给出；需要随机正交混合才能保证退化性，对实际系统的实现有一定要求。

---

## 546. When Should We Introduce Safety Interventions During Pretraining?

**arXiv ID:** 2601.07087 | [PDF](https://arxiv.org/pdf/2601.07087v1)

**作者:** Dylan Sam `[一作]` (Carnegie Mellon University), J. Zico Kolter `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18110 | [OpenAlex ID](https://openalex.org/A5075035644)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性研究在大规模预训练过程中不同时间点引入安全干预（如重写、拒绝训练、元数据标记）对语言模型安全性、鲁棒性、帮助性以及内部表示空间的影响。

**💡 创新点**

创新点在于将安全干预视为“安全课程”，通过比较0%、20%、60%预训练阶段引入干预的效果，首次揭示早期或中期干预能显著提升模型在安全评估、对抗攻击以及后续无害微调中的表现，并能更好地区分安全与不安全示例。

**🔧 技术方法**

技术包括：1) 基于SmolLM2架构的1.7B参数模型；2) 预训练期间插入的安全数据增强（上下文化改写、拒绝对话、元数据标记）；3) 通过SafeBeam等安全聚焦的推理算法进行评估；4) 线性探针用于分析安全/不安全示例在表示空间的可分离度。

**📊 数据集**

数据集涵盖FineWeb-Edu、Math、Code及合成安全示例，合计约600B token，预训练后在Alpaca、GSM8K等任务上进行安全与帮助性评测。

**📈 对比分析**

比较方法：在基准安全测试（prompt completion、Refusal）、对抗性破解（GCG）、帮助性评测（Alpaca）和表示空间可分离度上与不同干预时间点的模型进行对比。结果显示，早期或中期干预模型在标准top‑k推理下的危险生成率显著降低，SafeBeam下更易实现安全响应；后续无害微调后安全性提升最为显著，且总体过度拒绝率保持在可接受范围。

**⚠️ 局限性**

局限性包括：实验仅在1.7B规模模型上验证，未检验更大规模模型的可扩展性；干预方案依赖人工生成的合成示例，可能不覆盖所有真实场景；仅考察了三种时间点（0%、20%、60%）和特定干预组合，未来需探索更细粒度的时间策略和多样化干预方式。

---

## 547. Billboard in Focus: Estimating Driver Gaze Duration from a Single Image

**arXiv ID:** 2601.07073 | [PDF](https://arxiv.org/pdf/2601.07073v1)

**作者:** Carlos Pizarroso `[一作]` (Comenius University Bratislava), Viktor Kocur `[通讯]` (Comenius University Bratislava)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5041941896)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于YOLO的两阶段全自动管线，可从单帧图像检测路边广告牌并估计驾驶员注视时长；

**💡 创新点**

创新点在于无需多帧跟踪或眼动仪即可实现单帧注视时长分类，且结合检测框位置信息与DINOv2视觉特征实现了可扩展的估计；

**🔧 技术方法**

使用YOLOv11进行广告牌检测，DINOv2提取视觉特征，并在FLAML框架下进行特征融合与分类；

**📊 数据集**

利用Mapillary Vistas进行预训练，BillboardLamac数据集进行微调与评估，并在Google Street View图像上进行外部验证；

**📈 对比分析**

与原始多帧聚合方法比较，单帧模型在BillboardLamac测试集上获得68.1%准确率（聚合后69%），在Street View图像上达到66.3%准确率；

**⚠️ 局限性**

受限于BillboardLamac中仅145个独特广告牌导致的样本多样性不足，易出现过拟合，且对新环境的泛化能力尚待验证。

---

## 548. Between Policy and Practice: GenAI Adoption in Agile Software Development Teams

**arXiv ID:** 2601.07051 | [PDF](https://arxiv.org/pdf/2601.07051v1)

**作者:** Michael Neumann `[一作]` (University of Applied Sciences and Arts Hannover), Adam Przybylek `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在三家德国企业开展多案例研究，收集17份半结构化访谈与内部文档，探究敏捷软件团队中生成式人工智能（GenAI）工具的采用现状、使用场景、收益与障碍，并从技术、组织与环境三维度（TOE框架）分析政策与实践的差距。

**💡 创新点**

首次系统性、实证化研究敏捷团队对GenAI的使用模式，并揭示政策与实践之间的合规鸿沟；通过将TOE框架与敏捷角色相结合，提出了针对治理、培训与技术集成的改进路径。

**🔧 技术方法**

主要涉及的技术是生成式大语言模型工具，包括ChatGPT、GitHub Copilot和Microsoft Copilot等；研究亦聚焦这些工具在敏捷实践中的技术集成与工作流程适配。

**📊 数据集**

数据集为定性访谈文本（17份访谈录音转写）以及三家企业的内部使用政策与指南文件；未使用公开的量化软件项目数据。

**📈 对比分析**

采用跨案例主题分析方法，对访谈与文档进行编码、归纳主题，再进行交叉对比；结果表明GenAI在创意、文档与代码辅助方面具有显著效率提升，但由于验证负担、数据隐私与治理壁垒，实际收益与风险并存。

**⚠️ 局限性**

局限性包括：仅研究三家德国企业，样本规模有限且缺乏跨国与行业多样性；访谈数据主观自报，可能存在社会期望与记忆偏差；缺乏量化指标来衡量效率提升与质量影响。

---

## 549. Explainable Deep Radiogenomic Molecular Imaging for MGMT Methylation Prediction in Glioblastoma

**arXiv ID:** 2601.07035 | [PDF](https://arxiv.org/pdf/2601.07035v1)

**作者:** Hasan M Jamil `[一作]` (University of Idaho), Hasan M Jamil `[通讯]` (University of Idaho)

**通讯引用:** 1329 | [OpenAlex ID](https://openalex.org/A5030091324)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用多模态MRI结合Radiomics与深度学习融合方法，实现对脑胶质母细胞瘤（GBM）MGMT启动子甲基化状态的无创预测。

**💡 创新点**

创新点在于提出具有可变尺度注意力的ResUNetVSA网络、边界感知正则化与深度与手工特征的融合，并通过Grad‑CAM与SHAP提供可解释性。

**🔧 技术方法**

技术手段包括3D残差U‑Net+可变尺度注意力、3D ResNet‑18特征提取、PyRadiomics纹理/形状特征、梯度提升树分类器、Grad‑CAM/SHAP可解释性、N4偏置校正、CLAHE及网格对齐。

**📊 数据集**

数据集为RSNA‑ASNR‑MICCAI‑BraTS 2021（分割与MGMT标签）以及UCSD‑PTGBM（外部验证）。

**📈 对比分析**

与仅Radiomics、仅深度学习以及先前研究做对比，宏观Dice达0.912，交叉验证AUC0.871，外部验证AUC0.82，均显著优于传统方法。

**⚠️ 局限性**

局限性包括仅基于回顾性数据、缺乏前瞻性验证、计算量大、模型仍为相关性推断且可解释性评估尚未与临床专家充分对接。

---

## 550. A Large-Scale Study on the Development and Issues of Multi-Agent AI Systems

**arXiv ID:** 2601.07136 | [PDF](https://arxiv.org/pdf/2601.07136v1)

**作者:** Daniel Liu `[一作]` (Louisiana State University), Umar Farooq `[通讯]` (Louisiana State University)

**通讯引用:** 1823 | [OpenAlex ID](https://openalex.org/A5049262361)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对8个主流开源多智能体AI框架（AutoGen、CrewAI、Haystack、LangChain、Letta、LlamaIndex、Semantic Kernel、SuperAGI）进行大规模实证分析，收集42,267条唯一提交和4,731条已解决issue，分析提交类型、代码增删、issue分类及解决时长，以揭示这些框架的开发与维护特征。

**💡 创新点**

首次系统性、量化地评估多智能体框架的开发模式与维护实践，提出三种发展曲线（持续、稳定、爆发式）以及主导的维护类型分布（完美性、纠正性、适应性）。

**🔧 技术方法**

使用GitHub GraphQL API抓取数据；采用DistilBERT对提交信息进行维护类型分类；使用BERTopic对issue标签进行主题挖掘；统计代码增删量、CV、issue解决时间等指标。

**📊 数据集**

GitHub上8个开源MAS仓库的提交与issue数据，合计42,267条唯一提交和4,731条已关闭issue。

**📈 对比分析**

通过比较各框架的commit频率、CV、代码增删量、issue中位数/平均解决时间等指标进行对比；发现LangChain commit最多、SuperAGI呈爆发式开发，bug与基础设施问题占主导，issue平均解决时间右偏，部分框架（如LlamaIndex）响应快（<1天），部分框架（如Semantic Kernel）慢（>10天）。

**⚠️ 局限性**

研究仅覆盖GitHub公开仓库，可能遗漏私有或小型项目；issue/提交分类依赖自动化模型，存在误差；缺乏代码质量、设计动机等定性评估；可复现性受GitHub API或仓库结构变化影响。

---

## 551. The AI Cognitive Trojan Horse: How Large Language Models May Bypass Human Epistemic Vigilance

**arXiv ID:** 2601.07085 | [PDF](https://arxiv.org/pdf/2601.07085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 552. Digital Twin for Ultra-Reliable & Low-Latency 6G Wireless Communications in Dense Urban City

**arXiv ID:** 2601.07132 | [PDF](https://arxiv.org/pdf/2601.07132v1)

**作者:** Abdikarim Mohamed Ibrahim `[一作]` (Sunway University), Rosdiadee Nordin `[通讯]` (Sunway University)

**通讯引用:** 6139 | [OpenAlex ID](https://openalex.org/A5060844784)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建并评估了马来西亚新山市的数字孪生（DT）模型，用以分析七座屋顶基站在10 GHz下的毫米波覆盖，计算并映射到XR、V2X和URLLC的吞吐量需求。

**💡 创新点**

创新点在于将 Blender + OpenStreetMap 提取的三维网格与 Sionna GPU 加速射线追踪相结合，能够在细粒度城市几何上生成路径增益与 SINR 字段，并首次提出宏观多样性裕度（最佳与第二最佳基站的 SINR 差值）指标，以评估双连通的潜在可靠性提升。

**🔧 技术方法**

使用技术包括 Blender（构建 3D 建筑/道路模型）、Blosm 插件（从 OpenStreetMap 导入几何）、Sionna 及其 Ray‑Tracing 引擎（GPU 加速）、3GPP 38.901 天线模型、射频参数（10 GHz、400 MHz、30 dBm）以及 3 km×3 km 的 2 m 网格采样。

**📊 数据集**

数据集为 OpenStreetMap 的建筑与道路高度信息，转换为 Blender 三维网格；射频模型则采用标准化的 3GPP 天线与材料参数，构成七个屋顶基站的理想化部署。

**📈 对比分析**

比较方法：将射线追踪得到的 SINR 转化为理论吞吐量，分别与 XR（30 Mbps）、V2X（700 Mbps）和 URLLC（100 Mbps）阈值对比，得到覆盖面积比例（XR 约 13.9%、V2X 约 5.1%、URLLC 约 19%）以及宏观多样性裕度（平均 ≥ 3 dB，部分地区超过 10 dB）。

**⚠️ 局限性**

局限性包括：仅使用理想化射频参数未进行现场测量验证；未考虑用户运动、时变流量或多基站干扰动态变化；仅评估了七站静态部署，未探讨更密集或不同站点布局的性能。

---

## 553. The Need for a Socially-Grounded Persona Framework for User Simulation

**arXiv ID:** 2601.07110 | [PDF](https://arxiv.org/pdf/2601.07110v1)

**作者:** Pranav Narayanan Venkit `[一作]` (Salesforce Research), Chien-Sheng Wu `[通讯]` (Salesforce Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SCOPE框架，用八维社会心理结构（人口统计、社会行为、价值观与动机、人格特质、行为模式、身份叙事、专业身份与创造力）构建与评估合成人物（synthetic persona），并在七种大型语言模型上进行结构化对齐与偏差测评。

**💡 创新点**

创新点：①首次将社会心理学理论（社会身份、人格五大、价值观、叙事身份）整合进合成人物构造，构建多维结构而非单一人口统计；②设计了基于行为模式的结构相似度评估，强调与人类答案模式而非逐项匹配；③提出人口统计放大偏差量化指标（Bias%），揭示人口统计主导人物会过度放大群体差异；④将SCOPE作为可插拔的增强层，对现有Nemotron人物进行提升。

**🔧 技术方法**

技术方法：利用大型语言模型（GPT‑4o、GPT‑5.1、Claude‑3.5‑Sonnet、Gemini‑2.0 Flash/2.5 Pro、DeepSeek R1、Qwen‑3）生成人物并回答问卷；采用Pearson相关、准确率与偏差率三维评估；使用SimBench基准测试人物在外部社会行为问题上的表现；使用cosine相似度计算人口统计与回答相似度，量化偏差。

**📊 数据集**

数据集：①141项社会心理学问卷（共124名美国参与者）——涵盖八大维度；②外部SimBench（441道社会行为/心理学问题）用以检验人物在真实问卷上的泛化；③对比Nemotron USA合成人物数据。

**📈 对比分析**

比较方法：在七种模型上分别构造七种人物案例（从无人物到全信息、仅人口统计、叙事、特质、全条件、LLM摘要、AI补全、无人口统计）。对每种案例计算人类-模型的Pearson相关、准确率和偏差率；在SimBench上做多数投票准确率。结果显示：①人口统计单一人物相关≈0.62、准确≈35%，偏差≈+100%；②加入身份或价值后相关提升至≈0.66、准确≈40%，偏差显著下降；③全条件和仅价值+身份的人物达到最高相关≈0.66，准确≈39%，且Bias%为负，说明对偏差控制最好；④在SimBench上SCOPE人物比Nemotron更优，且SCOPE增强Nemotron亦可提升表现。

**⚠️ 局限性**

局限性：①样本仅来自美国，跨文化推广受限；②问卷固定，未覆盖所有行为驱动因素（如长期事件、情境压力等）；③相关度评估不保证因果或绝对分布一致；④AI检测过滤不完全，可能导致样本选择偏差。

---

## 554. Engineering of Hallucination in Generative AI: It's not a Bug, it's a Feature

**arXiv ID:** 2601.07046 | [PDF](https://arxiv.org/pdf/2601.07046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 555. MEDVISTAGYM: A Scalable Training Environment for Thinking with Medical Images via Tool-Integrated Reinforcement Learning

**arXiv ID:** 2601.07107 | [PDF](https://arxiv.org/pdf/2601.07107v1)

**作者:** Meng Lu `[一作]` (Virginia Tech), Xuan Wang `[通讯]` (Virginia Tech)

**通讯引用:** 8208 | [OpenAlex ID](https://openalex.org/A5078292155)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可扩展的工具集成医学视觉推理训练环境，并在该环境中使用两阶段监督+强化学习训练了一个VLM代理。

**💡 创新点**

提出仅凭工具访问不足以实现有效医学视觉推理，必须通过结构化的推理与工具调用交互、监督预热与在线强化学习相结合来学习纪律化的工具使用策略。

**🔧 技术方法**

采用InternVL3视觉语言模型作为骨干，先进行监督微调（SFT）以学习工具调用语法，再使用GRPO强化学习与基于正确答案的工具使用奖励进行在线优化，并对比PPO和不同奖励设计。

**📊 数据集**

评估使用六大医学VQA基准，包括PathVQA、SLAKE、VQA‑RAD（内分布）以及MMMU、PMC‑VQA、MicroVQA（外分布）。

**📈 对比分析**

与开源与专有VLM基线在同尺寸模型上对比，训练出的8B代理在六个基准上平均提升19.1%–24.2%（启用工具）并在无工具情形下提升8.4%–13.6%，显著优于同类方法。

**⚠️ 局限性**

多轮强化学习训练成本高，且环境主要聚焦VQA任务，扩展到其他临床推理或非医学领域需要额外适配；工具输出噪声或图像低质量仍可能导致推理失误。

---

## 556. PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation

**arXiv ID:** 2601.07060 | [PDF](https://arxiv.org/pdf/2601.07060v1)

**作者:** Yuanzhe Liu `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 942 | [OpenAlex ID](https://openalex.org/A5043962698)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉‑语言‑动作的框架，利用可学习查询预测未来交互affordance，并通过扩散变换器结合进度估计实现长时程机器人操控。

**💡 创新点**

创新点在于将结构化affordance预估与连续子任务进度推断联合到闭环决策中，提供任务相关的多尺度交互线索和实时进度信号。

**🔧 技术方法**

核心技术包括可学习查询+结构化注意力预测四类affordance（全局、局部、空间、动态），扩散式逆动力学策略以及大规模预训练与微调。

**📊 数据集**

使用BridgeDataV2、DROID机器人演示数据集以及EPIC‑KITCHENS、RoboCerebra长时程视频数据进行预训练，随后在942条手工标注的affordance轨迹上微调。

**📈 对比分析**

与RT‑1、Robo‑Flamingo、OpenVLA等自回归和扩散基准相比，在CALVIN ABC→D上提升12.5%任务成功率，在LIBERO‑LONG上达到91.8%成功率，并在真实世界长时程通用测试中显著优于基线。

**⚠️ 局限性**

限制在于对大规模预训练数据与计算资源依赖较大，且在极端视觉干扰或全新对象时仍可能出现误判；进度预测仍需更细粒度标注以进一步提升鲁棒性。

---

## 557. Fine-Tuning vs. RAG for Multi-Hop Question Answering with Novel Knowledge

**arXiv ID:** 2601.07054 | [PDF](https://arxiv.org/pdf/2601.07054v1)

**作者:** Zhuoyi Yang `[一作]` (University of California), Ian Harris `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统比较了多跳问答中无监督微调、监督微调和检索增强生成（RAG）三种知识注入方式，评估它们在传统多跳数据集 QASC 与自构造的包含 2024 年新事件的多跳数据集上的表现。

**💡 创新点**

① 在多跳问答场景下首次系统性对比参数化与非参数化知识注入；② 通过构造以 2024 年 Wikipedia 事件为素材的 10k+ 多跳问答数据集，检验模型对时间新颖知识的处理；③ 采用统一的多选评估框架，避免不同实验设置带来的偏差。

**🔧 技术方法**

使用 LoRA 微调、持续预训练（无监督微调）、监督多选微调、检索增强生成（BGE 句向量、FAISS 索引、cross‑encoder 重新排序）以及 MMLU 风格的答案概率计分。

**📊 数据集**

QASC（多跳科学问答）和 2024 Events（基于 2024 年 Wikipedia 事件的 10k+ 多跳多选问答）及其对应的知识语料库。

**📈 对比分析**

在统一的多选准确率评估框架下对比四种设置：基准模型、RAG、无监督微调、监督微调。结果显示：基准模型准确率最低；RAG 在两大数据集上提升约 30pp（2024 Events 上提升超过 2 倍）；监督微调在所有模型和数据集上表现最好；无监督微调仅带来微小提升。

**⚠️ 局限性**

限制：仅评估 7B 参数开源模型；未分析中间推理步骤，无法区分检索质量与模型推理能力的贡献；RAG 的效果高度依赖检索模型与索引质量；未探讨算力与延迟成本；监督微调需要大量标注数据，实用性受限。

---

## 558. ENTRA: Entropy-Based Redundancy Avoidance in Large Language Model Reasoning

**arXiv ID:** 2601.07123 | [PDF](https://arxiv.org/pdf/2601.07123v1)

**作者:** Ruichu Cai `[一作]` (Guangdong University of Technology), Boyan Xu `[通讯]` (Guangdong University of Technology)

**通讯引用:** 1237 | [OpenAlex ID](https://openalex.org/A5034536387)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了ENTRA框架，利用熵奖励抑制大型推理模型的冗余推理链。

**💡 创新点**

将令牌重要性估计与信息熵相结合，构造基于低重要性令牌熵的奖励并用理论上限归一化，辅以强化学习实现自适应压缩。

**🔧 技术方法**

使用Bidirectional Importance Estimation (BIE)估计令牌重要性，计算低重要性令牌熵并归一化，并利用GRPO强化学习进行优化。

**📊 数据集**

在英文数学推理数据集MATH-500、GSM8K、MAWPS以及中文数据集Weekly12K、MATH23K、CMath上进行实验。

**📈 对比分析**

与SFT、DPO、LC-R1、HAPO等基线对比，ENTRA在保持或提升准确率的同时实现了37–53%的长度压缩，并在大模型上表现出最高的压缩效果。

**⚠️ 局限性**

局限包括近似的双向重要性估计可能引入偏差，且方法主要针对数学推理任务，跨任务泛化尚待验证。

---

## 559. Reward-Preserving Attacks For Robust Reinforcement Learning

**arXiv ID:** 2601.07118 | [PDF](https://arxiv.org/pdf/2601.07118v1)

**作者:** Lucas Schott `[一作]` (IRT SystemX), Sylvain Lamprier `[通讯]` (Universite d'Angers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种奖励保持攻击（reward‑preserving attack）框架，动态调节对抗扰动强度以保证在每个状态-动作对上可保持α比例的期望奖励，从而在强化学习中实现既不破坏学习信号又能提升鲁棒性的对抗训练方法。

**💡 创新点**

创新点在于定义了α‑reward‑preserving攻击，使攻击强度与奖励保持比例相匹配；引入幅度‑方向分解并通过学习Q((s,a),η)自适应选择攻击强度；通过参考策略与离线更新实现对抗训练的可行性。

**🔧 技术方法**

利用深度强化学习（SAC）与对抗训练技术，使用梯度基FGM攻击、Q值预测网络、离线经验回放与重要性采样更新；在实验中使用了动态幅度的FGM_QAC_A攻击。

**📊 数据集**

在HalfCheetah-v5环境上进行实验，使用预训练的SAC基线模型作为起点。

**📈 对比分析**

将α‑训练与固定半径训练以及均匀采样半径训练进行对比；结果显示α=0.7时的策略在不同扰动半径下既保持了较高的期望回报，又保持了较好的原始性能，优于固定或随机半径的对抗训练。

**⚠️ 局限性**

局限性包括：仅在观测扰动上验证，动态扰动和动力学攻击仍待进一步研究；需要额外训练Q网络以估计不同幅度下的回报，增加计算开销；对参考策略的依赖可能导致策略更新不稳定。

---

## 560. Efficient Visual Question Answering Pipeline for Autonomous Driving via Scene Region Compression

**arXiv ID:** 2601.07092 | [PDF](https://arxiv.org/pdf/2601.07092v1)

**作者:** Yuliang Cai `[一作]` (University of Southern California), Chongruo Wu `[通讯]` (XPeng)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SRC‑Pipeline，将自动驾驶视频前期帧压缩为场景与四区块高层 token，保留后期帧细粒度 patch token，显著降低 VLM 推理 FLOPs；

**💡 创新点**

创新点在于：① 只压缩早期帧、保留关键后期帧；② 设计 SRC‑ViT 能同时输出全局场景 token 与局部区域 token；③ 采用多尺度图文对齐的两阶段训练；④ 通过实验验证后期帧信息更重要；

**🔧 技术方法**

技术实现包括：SRC‑ViT 的多尺度 token 生成与区域注意力掩码、CLIP 文本编码器对齐的对比学习、VLM (QWen2VL) 的视觉–语言映射、FLOPs 节省的 token 选取策略、位置编码适配 M‑RoPE；

**📊 数据集**

使用数据集：LingoQA（419.9k 自动驾驶 QA 对），CC3M（阶段一检索验证），以及 CLIP‑L/14 文本编码器；

**📈 对比分析**

通过在 LingoQA 上与 QWen2VL、LingoQA、LLaVA、BLIP‑2 等基线对比，展示 5 帧下的 Ling‑Judge 与 BLEU 与全帧基线相当，1 帧下仍优于基线；Ablation 证明场景+区域 token 组合比单一 token 或无压缩更优；FLOPs 从 100% 降至约 33%，保持接近原性能；

**⚠️ 局限性**

局限性：仅在 5 帧短视频上评估，未验证更长序列；依赖单一 VLM 框架，跨模态扩展未充分；压缩可能导致极细粒度信息丢失；实际车辆部署时需进一步验证实时性与安全性；

---

## 561. Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems

**arXiv ID:** 2601.07072 | [PDF](https://arxiv.org/pdf/2601.07072v1)

**作者:** Hongyan Chang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Ting Yu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1466 | [OpenAlex ID](https://openalex.org/A5101948685)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在实际检索式LLM系统中，如何通过黑盒方式构造可检索前缀，使得恶意注入文本能在自然查询下被检索到，从而实现间接提示注入（IPI）攻击；

**💡 创新点**

创新点在于将IPI拆解为可检索前缀和攻击负载两部分，并提出基于交叉熵法（CEM）的黑盒前缀搜索算法，在有限的embedding查询预算下即可近似最优检索；实验表明该方法在多模型、多数据集上可实现近乎100%的Recall@5，并能在RAG、单/多Agent等下游任务中产生高攻击成功率；

**🔧 技术方法**

使用的核心技术包括：黑盒交叉熵搜索（CEM）、token化前缀生成、embedding模型（OpenAI、Voyage、Alibaba、GTE、Contriever、Qwen3等）的查询、余弦相似度评分；

**📊 数据集**

使用的数据集主要有BEIR的11个检索基准（MSMARCO、NFCorpus、NaturalQuestions等）、Enron邮件集、MS COCO图像检索等；embedding模型覆盖开源与闭源、不同规模与架构；

**📈 对比分析**

与基线（Query+、Vanilla）对比，CEM前缀在10-15个token下Recall@5接近100%，攻击成本仅约$0.21/查询；在RAG、Agentic任务中，攻击成功率显著高于基线；在多模型、多数据集上均保持高效；

**⚠️ 局限性**

局限性包括：仅针对embedding‑based检索，未考虑混合检索、reranking等机制；跨模型转移性受限（不同架构时效果下降）；防御适应性尚未彻底解决；实验基于公开基准和合成语料，未验证对真实部署环境的直接影响；

---

## 562. Neuromorphic FPGA Design for Digital Signal Processing

**arXiv ID:** 2601.07069 | [PDF](https://arxiv.org/pdf/2601.07069v1)

**作者:** Justin London `[一作]` (University of North Dakota), Justin London `[通讯]` (University of North Dakota)

**通讯引用:** 2846 | [OpenAlex ID](https://openalex.org/A5040044002)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用 Vivado/Verilog HDL 在 FPGA 上实现并仿真传统 FIR、IIR 滤波器与其 neuromorphic（基于 memristor 与 SNN 的）版本，比较两种实现的误差、资源占用与功耗潜力。

**💡 创新点**

创新点在于：① 将 memristor 权重阵列与 LIF/Elman 神经网络结合，构建可在线自适应的滤波器；② 在 FPGA 逻辑层实现时域寄存器与时域乘法器，展现事件驱动、低功耗的 DSP 方案；③ 通过 LMS 学习规则实现滤波器权重自适应，验证 neuromorphic DSP 的可行性。

**🔧 技术方法**

主要技术包括：FPGA（Xilinx Vivado、Verilog HDL、DSP48 块）、LTSpice 电路仿真、memristor 模型、时域寄存器/乘法器、Leaky Integrate‑and‑Fire (LIF) 神经元、Elman‑型递归网络、固定点 Q15 运算、STDP、在线 LMS 学习。

**📊 数据集**

使用合成的正弦+高斯噪声信号作为测试数据集（频率 50 Hz，噪声幅值 0.05，量化为 Q15），未使用公开数据集。

**📈 对比分析**

比较方法：将传统与 neuromorphic 版本在相同输入下的输出误差（MSE）和资源占用（Cell、Pin、NetList）进行对比。结果显示：传统 FIR MSE = 0；传统 IIR MSE = 0.1374；Neuromorphic FIR MSE = 0.5209；Neuromorphic IIR MSE = 0.1533。资源使用方面，Neuromorphic 版本略高，但功耗预期更低。

**⚠️ 局限性**

局限性：① 精度较低，特别是 neuromorphic FIR；② 受模拟噪声与器件变异影响；③ 仅在仿真层面验证，未实现实际硬件部署；④ 未量化功耗与时延；⑤ 需要进一步优化学习算法与权重更新速率。

---

## 563. Hallucinations Live in Variance

**arXiv ID:** 2601.07058 | [PDF](https://arxiv.org/pdf/2601.07058v1)

**作者:** Aaron R. Flouro `[一作]`, Shawn P. Chadwick `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“语义稳定性”(Semantic Stability) 指标，用于量化大型语言模型在语义等价提示下的输出一致性，并通过稀疏知识蒸馏(Sparse Knowledge Distillation, SparseKD)方法对 Qwen3-0.6B 进行多阶段压缩，从而在保持或提升困惑度的同时提升模型对同义改写的鲁棒性。

**💡 创新点**

创新点在于：①将模型的自我一致性作为可靠性评估轴，引入 PC@k 作为可操作的度量；②阐释压缩如何通过减少内部冗余路径来降低方差，从而提升可靠性；③绘制方差‑偏差相互作用的压缩相位图，揭示压缩的“甜点”区间；④证明压缩在方差占优的 regime 下可同步提升 perplexity 与稳定性。

**🔧 技术方法**

核心技术包括：确定性贪婪解码；多重同义改写生成；SparseKD 的多阶段结构化剪枝与概率域知识蒸馏；以及对 PC@k 计算的统计分析。

**📊 数据集**

主要实验数据集为 Qwen3-0.6B 语言模型，在 GSM8k、TruthfulQA、MMLU 等任务上进行评价；同义改写通过自动 paraphrasing 工具生成；压缩级别覆盖从 0% 到 70% 的稀疏率。

**📈 对比分析**

与密集模型比较时，Dense 仅有 23.8% 的自我一致率；在 32% 稀疏率（R4 阶段）提升至 55.9% 的稳定性，且 perplexity 相较 dense 改进；但在事实类任务上，压缩导致稳定性下降，表明存在偏差崩溃。整体来看，SparseKD 在方差主导 regime 下实现了 32 点的稳定性提升，并在同一 regime 下提升了 perplexity；在偏差主导 regime 下则相反。

**⚠️ 局限性**

局限性包括：①稳定性与正确性并不相关，模型可能因压缩而“稳定但错误”；②SparseKD 需要较大的蒸馏语料与多阶段训练，成本较高；③目前仅在单一模型与少数任务上验证，跨模型与跨任务的普适性尚未完全评估；④同义改写生成的质量和多样性会影响 PC@k 的准确性。

---

## 564. Adversarial Attacks on Medical Hyperspectral Imaging Exploiting Spectral-Spatial Dependencies and Multiscale Features

**arXiv ID:** 2601.07056 | [PDF](https://arxiv.org/pdf/2601.07056v1)

**作者:** Yunrui Gu `[一作]`, Zhaoxia Yin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种针对医学高光谱影像的定向对抗攻击框架，能在保持视觉无感的情况下显著削弱模型在肿瘤/癌症区的分类准确率。

**💡 创新点**

创新点在于结合局部像素依赖攻击和多尺度信息攻击，分别利用像素间空间相关性和跨尺度的光谱-空间特征来诱导决策边界的局部移位。

**🔧 技术方法**

使用梯度平均化的局部像素攻击、下采样/上采样的多尺度扰动、迭代式对抗生成以及对抗目标损失设计。

**📊 数据集**

实验数据集包括In‑Vivo Brain Hyperspectral Database（脑癌）和MultiDimensional Choledoch（MDC）数据集。

**📈 对比分析**

与SS‑FGSM、SSA、MfcaNet等传统攻击以及RCCA、WFSS、AIAF、S3ANet等防御模型对比，实验显示在保持整体准确率高于96%的同时，肿瘤/癌症类别的准确率骤降至30%以下，证明攻击效果显著。

**⚠️ 局限性**

局限性包括未充分考虑医学领域特定的先验信息（如肿瘤与周围组织的光谱相似性），对抗扰动的可解释性和在不同光谱分辨率下的鲁棒性待进一步验证。

---

## 565. Dr. Zero: Self-Evolving Search Agents without Training Data

**arXiv ID:** 2601.07055 | [PDF](https://arxiv.org/pdf/2601.07055v1)

**作者:** Zhenrui Yue `[一作]` (Meta Superintelligence Labs), Dong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 22369 | [OpenAlex ID](https://openalex.org/A5100391422)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 DeepResearch‑Zero，一种完全零数据的自进化框架，利用外部搜索引擎训练提问者（Proposer）和求解者（Solver），通过多轮工具调用生成并学习多跳开放域问题。

**💡 创新点**

创新点：①引入多轮工具使用回放管线，显著提升问题生成质量；②设计难度引导奖励，鼓励生成既可验证又具有挑战性的查询；③采用跨跳聚类（Hop‑Based Clustering）实现单样本优势估计，消除传统 GRPO 的嵌套采样导致的高计算成本。

**🔧 技术方法**

技术：基于 LLM 的提问者/求解者双网络；使用强化学习（GRPO、REINFORCE++、KL 正则化）训练；单样本优势估计与跨跳标准化；难度奖励与格式奖励；多轮工具调用回放。

**📊 数据集**

数据集：Natural Questions、TriviaQA、PopQA（单跳）；HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle（多跳）。

**📈 对比分析**

与少量提示与监督基线（Prompting、IRCoT、Search‑o1、RAG、SFT、R1、Search‑R1）对比，DeepResearch‑Zero 在单跳任务上比最佳监督基线高出约 20%‑25%，在多跳任务上达到或超过 90% 的监督基线（最高可超 14.1%）。

**⚠️ 局限性**

局限性：①在大模型（7B）训练过程中容易出现熵坍塌和收敛停滞；②迭代次数有限，超过两次后性能基本饱和；③对奖励设计的鲁棒性尚需改进，防止奖励劫持和偏差放大；④缺乏对搜索结果质量的直接监督，可能导致错误信息的误学习。

---

## 566. Random Access in DNA Storage: Algorithms, Constructions, and Bounds

**arXiv ID:** 2601.07053 | [PDF](https://arxiv.org/pdf/2601.07053v1)

**作者:** Chen Wang `[一作]` (Technion Israel Institute of Technology), Eitan Yaakobi `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了高效计算DNA随机访问覆盖深度期望值的算法，并给出了新的构造和上下界

**💡 创新点**

在固定q和k下实现O(n)时间期望读取次数计算，改进了k=3、4时的上界，并给出最优的简单偶错码证明

**🔧 技术方法**

基于有限域线性编码、子空间权重分布、组合计数与Gaussian二项式、期望数值求解

**📊 数据集**

无实验数据集，主要为理论推导

**📈 对比分析**

与先前的0.8815k、0.8822k、0.8629k等上界进行比较，取得了更小的常数比例（如0.8811k、0.8629k）并给出更紧的下界

**⚠️ 局限性**

仅适用于小k、q且需遍历所有子空间，计算量随k、q指数增长；在实际大规模DNA存储中仍需进一步优化

---

## 567. Jasper: ANNS Quantized for Speed, Built for Change on GPU

**arXiv ID:** 2601.07048 | [PDF](https://arxiv.org/pdf/2601.07048v1)

**作者:** Hunter McCoy `[一作]` (Northeastern University), Prashant Pandey `[通讯]` (Northeastern University)

**通讯引用:** 1555 | [OpenAlex ID](https://openalex.org/A5014159929)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在GPU上实现了一个可增量更新、可高吞吐量的近似最近邻搜索系统Jasper，采用Vamana图索引。

**💡 创新点**

主要创新包括：①锁无锁的批量并行构造算法支持流式插入；②GPU友好的量化方法RaBitQ，避免随机访问并实现高压缩率；③针对GPU的贪婪搜索核通过内核融合、tile加载和线程块尺寸优化实现计算与内存访问协同。

**🔧 技术方法**

技术栈涵盖CUDA、CUDA Thrust、共享内存调度、RaBitQ量化、beam search融合、lock‑free batch construction、tile‑based vector加载、roofline性能分析。

**📊 数据集**

在五个公开数据集上评估：SIFT‑100M (128D)、Yandex‑100M (96D)、Deep‑1M (960D)、OpenAI‑Arxiv‑2.3M (1536D)、ResNeXt‑10M (200D)。

**📈 对比分析**

与四个GPU基准（Faiss‑NN‑Descent、DiskANN、HNSW‑GPU、Vamana‑GPU）对比，Jasper在构造、增量更新和查询吞吐量上分别快 2.4×、10×、50% 以上，且在高维数据上通过RaBitQ可实现 8× 的内存压缩并保持或提升召回率。

**⚠️ 局限性**

局限性包括：①对极大维度（> 2000D）或非欧氏度量（如MIPS）仍存在性能下降；②量化虽然压缩率高，但在低维场景下略逊于原精度；③系统需手工调节线程块和beam宽度，适配性仍需进一步自动化。

---

## 568. CloneMem: Benchmarking Long-Term Memory for AI Clones

**arXiv ID:** 2601.07023 | [PDF](https://arxiv.org/pdf/2601.07023v1)

**作者:** Sen Hu `[一作]` (Peking University), Ronghao Chen `[通讯]` (Peking University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5109632049)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 CloneMem 基准，用以评估 AI 复制人（AI Clone）在非对话式数字轨迹（日记、社交媒体、邮件等）上长期记忆与个体情绪、经验、观点演变的能力，并设计了多种基于时间推理的问答任务。

**💡 创新点**

创新点在于：①将评估焦点从对话历史转向真实生活中的连续数字轨迹；②构建三层次层级生成框架（人格→宏观人生弧线→阶段→细粒度事件），保证经验、情绪与观点的时间连贯性；③提出多任务评测（事实回忆、时间推理、因果/抽象推理），并通过多模态数字轨迹与证据单位的多对多关系实现更细粒度的记忆检索。

**🔧 技术方法**

使用技术包括：检索增强框架（embedding‑based检索，使用 Contriever 或 text‑embedding‑3‑small）；两种大型语言模型 LLaMA‑3.1‑8B 与 GPT‑4o‑mini 进行回答生成；三种记忆架构（Flat、Mem0、A‑Mem）；LLM‑as‑judge 评估 QA 一致性、记忆有用性；以及层级生成与证据抽取脚本。

**📊 数据集**

数据集为 CloneMem，包含 10 个人格，生成的中英文数字轨迹共约 5,000 条 QA 对，长文本上下文范围从 100k 令牌到 1M 令牌，涵盖经验、情绪和观点的时间演化，所有轨迹均为合成数据。

**📈 对比分析**

实验对比显示：在检索层面，最简单的 Flat 检索在 Recall‑All‑Any 等指标上持续优于 Mem0、A‑Mem；在生成层面，GPT‑4o‑mini 的 QA 一致性和选择准确率普遍高于 LLaMA‑3.1‑8B，尤其在大检索深度时更稳健；总体上，抽象/压缩型记忆机制在 CloneMem 上往往适得其反，导致细粒度追踪失败。

**⚠️ 局限性**

局限性包括：①轨迹为合成数据，缺乏真实噪声与语言变异；②仅模拟文本描述，未涵盖多模态信息；③评估依赖 LLM‑as‑judge 可能引入偏差；④仅覆盖 10 个个体，文化与行为多样性有限。

---

## 569. TurkBench: A Benchmark for Evaluating Turkish Large Language Models

**arXiv ID:** 2601.07020 | [PDF](https://arxiv.org/pdf/2601.07020v1)

**作者:** Çağrı Toraman `[一作]` (Middle East Technical University), Esra Darıcı `[通讯]` (Middle East Technical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了TurkBench，一个覆盖21个子任务、8151条样本、6大类（知识、语言理解、推理、内容审核、土耳其语语法词汇、指令跟随）的土耳其语大型语言模型评测基准。

**💡 创新点**

创新点在于：①完全使用土耳其语原始数据，避免翻译带来的语义与文化失真；②涵盖开源指令执行、内容安全、语法与词汇深度等多维度，填补现有土耳其LMM评测空白；③采用LLM-as-a-Judge与元提示优化的评测框架，提升主观任务的可重复性。

**🔧 技术方法**

技术手段包括：LLM-as-a-Judge（GPT‑4o‑mini）评估开放式任务；元提示（metaprompting）自动生成与选择提示；多种评价指标（准确率、Pearson/ Spearman、Deepeval等）；在线排行榜与自动化提交。

**📊 数据集**

数据集来源于土耳其国家考试（OSYM）、技术与科技研究院（TUBITAK）、大学教材、土耳其文学与词典、新闻与社交媒体公开内容，由专家手工标注并审核。

**📈 对比分析**

通过与27个开源LLM的对比实验，展示了不同模型在各子任务的得分与整体平均；大型模型（如Gemma‑27B、Qwen‑32B）在多数任务上明显优于小模型；基准还能揭示模型在土耳其文化推理与习语等方面的薄弱。

**⚠️ 局限性**

局限性包括：数据主要为正式学术与新闻语料，缺乏口语、方言与网络俚语；LLM‑as‑a‑Judge可能带有自身偏见；仅评测文本任务，未涉及多模态或语音；隐私与滥用风险仍需关注。

---

## 570. XBTorch: A Unified Framework for Modeling and Co-Design of Crossbar-Based Deep Learning Accelerators

**arXiv ID:** 2601.07086 | [PDF](https://arxiv.org/pdf/2601.07086v1)

**作者:** Osama Yousuf `[一作]` (George Washington University), Gina C. Adam `[通讯]` (George Washington University)

**通讯引用:** 4092 | [OpenAlex ID](https://openalex.org/A5077532268)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了XBTorch统一框架，集成了跨越式内存技术的设备模型、硬件感知训练、梯度低秩压缩和推理时容错等功能，以实现跨越式深度学习加速器的完整模拟。

**💡 创新点**

创新点在于提供统一的、可扩展的PyTorch API，支持多种跨越式内存技术（FeFET、ReRAM等），并实现硬件感知训练、梯度低秩压缩、推理时容错等全链路功能，打破了以往技术碎片化的局面。

**🔧 技术方法**

采用Python/ PyTorch实现，集成了分析式与表格式设备模型、WAGE量化、SBPCA/NMF梯度分解、差分编码与映射算法、状态化/无状态化交替模拟等技术。

**📊 数据集**

主要使用MNIST手写数字数据集进行训练和推理实验，LLM评估则基于TriLM系列模型，测试多项自然语言基准。

**📈 对比分析**

与纯软件PyTorch基线对比，硬件感知训练在MNIST上精度略低但更鲁棒；在LLM中显示ADC/DAC比特精度提升可显著提升准确率；梯度分解方法在低秩时表现差异，低秩SBPCA优于NMF。

**⚠️ 局限性**

限制在于对大模型训练的噪声仿真仍较慢，缺乏快速噪声近似；对某些新兴技术的参数支持尚不完善，且在推理时需要手工配置映射与容错策略。

---

## 571. Towards Automated Diagnosis of Inherited Arrhythmias: Combined Arrhythmia Classification Using Lead-Aware Spatial Attention Networks

**arXiv ID:** 2601.07124 | [PDF](https://arxiv.org/pdf/2601.07124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 572. Proof of Reasoning for Privacy Enhanced Federated Blockchain Learning at the Edge

**arXiv ID:** 2601.07134 | [PDF](https://arxiv.org/pdf/2601.07134v1)

**作者:** James Calo `[一作]` (Imperial College London), Benny Lo `[通讯]` (Imperial College London)

**通讯引用:** 11665 | [OpenAlex ID](https://openalex.org/A5063187094)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计了一种面向边缘医疗 IoT 的隐私增强区块链联邦学习框架，提出 Proof of Reasoning 共识机制和基于遮掩自编码器的特征提取，实现模型训练和聚合时的隐私保护与恶意模型抵御。

**💡 创新点**

提出 PoR 共识机制专门针对联邦学习，可验证并惩罚恶意模型；将 MAE 继续高遮掩率训练为下游分类器的特征映射，使编码数据不可逆；在区块链上共享编码数据+模型权重，实现可验证、可聚合的全局模型。

**🔧 技术方法**

采用遮掩自编码器（MAE）、Vision Transformer、轻量级下游分类器、Proof of Reasoning 共识、区块链分布式账本以及多渠道聚合策略。

**📊 数据集**

在 Cifar10、Chest‑MNIST 及 Pneumonia‑MNIST 等公开图像数据集上进行实验，并在自定义医学影像子集上验证。

**📈 对比分析**

与传统 Federated Averaging、PoW/PoS 共识下的联邦学习以及使用差分隐私/同态加密的方案对比，PoR+MAE 方案在保持 93–95% 的分类准确率的同时，通信与计算开销大幅降低，恶意模型影响显著抑制。

**⚠️ 局限性**

仅在单通道灰度图像上验证，仍需评估多通道彩色图像的编码策略；编码子采样限制了模型可用信息；区块链交易成本和共识延迟在极大规模 IoT 网络中需进一步优化。

---

## 573. Geometry-Aware LoRaWAN Gateway Placement in Dense Urban Cities Using Digital Twins

**arXiv ID:** 2601.07133 | [PDF](https://arxiv.org/pdf/2601.07133v1)

**作者:** Abdikarim Mohamed Ibrahim `[一作]` (Sunway University), Rosdiadee Nordin `[通讯]` (Sunway University)

**通讯引用:** 6139 | [OpenAlex ID](https://openalex.org/A5060844784)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

本文利用几何精确的数字孪生城市模型和GPU加速的射线追踪，对密集城市环境中的LoRaWAN网关楼顶放置进行评估，提出了基于真实链路预算的覆盖与冗余分析，并通过贪心最大覆盖算法对候选楼顶进行排名。

**💡 创新点**

创新点在于：①将子GHz LoRaWAN与精确数字孪生结合，首次在完整3D城市模型上实现射线追踪覆盖预测；②提出预算约束下的子模量最大覆盖框架，自动给出最优楼顶顺序；③系统地量化覆盖、冗余及最佳网关关联区划，揭示传统距离规划忽略的阴影与多径效应。

**🔧 技术方法**

使用技术包括：Blender + OpenStreetMap 构建数字孪生、Sionna GPU射线追踪引擎、基于RAK7289 WisGate Edge Pro 的子GHz LoRaWAN 链路预算、贪心最大覆盖算法（预算约束的子模量优化）以及 5 m 级网格化 SNR 与覆盖评估。

**📊 数据集**

使用的数据集为：①Sunway City 城市数字孪生（包含建筑高度、道路、水体等）；②8 个候选楼顶的坐标和高度；③100 只随机布置的感知节点（用于展示覆盖情况）；④RAK7289 数据表给出的链路预算参数。

**📈 对比分析**

对比方法：在不同网关数（1–6）下，利用贪心算法计算覆盖率、冗余率和最佳关联区，形成覆盖/冗余曲线；与单个楼顶的独立覆盖率比较，评估每步增量的效益。实验结果表明：单个网关仅覆盖约 20% 区域，六个网关覆盖 44% 区域；冗余覆盖（至少两路）从 0% 增至 26%。

**⚠️ 局限性**

局限性包括：①仅评估了 8 个楼顶候选，缺乏更广泛的楼顶搜索；②未考虑流量分布、时变信道或多频段；③未引入背链路容量和成本约束；④实验仅在单一城市数字孪生上验证，缺乏跨城市泛化；⑤使用的是 Class A LoRa 设备，未考虑其它工作模式；⑥射线追踪仍依赖材料模型的准确性，可能忽略树木、车辆等动态障碍物。

---

## 574. Recovering polynomials over finite fields from noisy character values

**arXiv ID:** 2601.07137 | [PDF](https://arxiv.org/pdf/2601.07137v1)

**作者:** Swastik Kopparty `[一作]` (University of Toronto), Swastik Kopparty `[通讯]` (University of Toronto)

**通讯引用:** 1396 | [OpenAlex ID](https://openalex.org/A5031695135)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出了一种多项式恢复算法，能够在有限域上从带噪声的字符值中恢复多项式g(X)，并且在存在一定比例的错误值的情况下也能有效恢复。

**💡 创新点**

这是首次在多项式时间内恢复有限域上多项式的算法，即使在存在错误的情况下也能做到，特别是对于二次剩余字符和特征为2的多项式的加法字符。

**🔧 技术方法**

算法结合了Stepanov的多项式方法和Berlekamp-Welch解码算法的思想，使用了伪多项式的概念，这些伪多项式的导数行为类似于低度多项式。

**📊 数据集**

使用了有限域上的多项式，特别是二次剩余字符和特征为2的多项式的加法字符，具体数据集未明确提及。

**📈 对比分析**

与之前的算法相比，新的算法在存在错误的情况下能在多项式时间内恢复多项式g(X)，而之前的算法在没有错误的情况下也需要更长的时间。性能上，新的算法在处理错误时表现更优。

**⚠️ 局限性**

算法的局限性在于它处理的错误数量低于唯一解码半径，且在处理更高比例的错误时效率可能下降。

---

## 575. SC-MII: Infrastructure LiDAR-based 3D Object Detection on Edge Devices for Split Computing with Multiple Intermediate Outputs Integration

**arXiv ID:** 2601.07119 | [PDF](https://arxiv.org/pdf/2601.07119v1)

**作者:** Taisuke Noguchi `[一作]` (Saitama University), Takuya Azumi `[通讯]` (Saitama University)

**通讯引用:** 1698 | [OpenAlex ID](https://openalex.org/A5022413623)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出了 SC-MII 方法，利用多台基础设施 LiDAR 的中间特征进行分布式 3D 目标检测，采用分割计算将前期卷积留在边缘设备完成，后期卷积及检测在边缘服务器完成；

**💡 创新点**

其创新点在于：①将多台 LiDAR 的中间输出在同一坐标系下对齐并融合，避免传输原始点云；②使用 NDT 扫描匹配得到静态变换矩阵，确保多传感器特征的一致性；③在保持低通信量和隐私的前提下，仅通过中间特征实现近似原始点云融合的检测精度；

**🔧 技术方法**

核心技术包括 Split Computing（将模型分为 Head 与 Tail）、NDT 扫描匹配、Voxel R‑CNN 检测模型、卷积融合与最大值融合两种特征融合策略；

**📊 数据集**

实验基于 V2X‑Real 数据集，使用两台 Ouster OS1‑64 与 OS1‑128 基础设施 LiDAR 的点云数据进行评估；

**📈 对比分析**

与单 LiDAR 与原始点云多传感器融合的基线对比，SC‑MII 在卷积 3×3 融合下的 AP@0.3 与 AP@0.5 分别为 75.72% 与 54.09%，与原始点云融合仅下降 1.05%/1.09%；推理时间平均提升 2.19 倍，设备端计算时间下降 71.6%；

**⚠️ 局限性**

局限性包括：对网络延迟和不稳定性的鲁棒性不足，需要进一步压缩特征或设计容错机制；仅适用于固定位置的基础设施 LiDAR，动态变化时需重新估计变换矩阵；特征融合方式仍可优化以提升精度。

---

## 576. How Secure is Secure Code Generation? Adversarial Prompts Put LLM Defenses to the Test

**arXiv ID:** 2601.07084 | [PDF](https://arxiv.org/pdf/2601.07084v1)

**作者:** Melissa Tessa `[一作]` (University of Luxembourg), Tegawendé F. Bissyandé `[通讯]` (University of Luxembourg)

**通讯引用:** 7538 | [OpenAlex ID](https://openalex.org/A5082835974)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对当前主流安全代码生成方法进行系统的对抗性审计，评估其在真实攻击情境下的鲁棒性和功能性；

**💡 创新点**

首次在统一评估框架下同时测量安全性与功能性，并揭示现有方法因安全与功能评估分离而被高估；

**🔧 技术方法**

采用prefix‑tuning、instruction‑tuning、prompt‑optimization三种技术，结合静态分析器（CodeQL、Bandit）、LLM判定（GPT‑4o）和可执行单元测试；

**📊 数据集**

使用CodeSecEval benchmark（Python任务，包含漏洞标签和测试用例）以及原作者提供的自有数据集；

**📈 对比分析**

对SVEN、SafeCoder、PromSec三种方法进行比较，发现其在攻击下安全率急剧下降，功能安全交叉率仅在3%~17%之间，静态分析器严重高估安全性；

**⚠️ 局限性**

局限在于：评估仍受限于单一语言（Python），缺乏多语言和更深层次动态检测，且对抗攻击多基于自然语言提示，未覆盖代码层面更细粒度攻击；

---

## 577. Automated Domain Question Mapping (DQM) with Educational Learning Materials

**arXiv ID:** 2601.07062 | [PDF](https://arxiv.org/pdf/2601.07062v1)

**作者:** Jiho Noh `[一作]` (Kennesaw State University), Dabae Lee `[通讯]` (Kennesaw State University)

**通讯引用:** 700 | [OpenAlex ID](https://openalex.org/A5072108793)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对教材内容生成教育问题并利用层级关系与语义相似度构建域问题映射（DQM），实现自动化的教育知识结构可视化与导航。

**💡 创新点**

① 用问题生成代替概念提取，更贴合学习目标；② 结合特异性关系分类与句子语义相似度，利用最小生成树压缩并保持图可导航；③ 通过预训练模型微调实现端到端生成与关系判定。

**🔧 技术方法**

使用预训练语言模型（BART、Pegasus、T5、GPT‑O4‑mini）微调做问答生成，GPT‑2进行特异性关系分类，Sentence‑BERT做语义相似度编码，基于权重的最大生成树算法实现图后处理。

**📊 数据集**

问答生成训练集：KhanQ（1,034问答）与SQuAD v2.0（142,192问答，选答题可答部分）；关系分类与DQM构建训练集：信息检索教材《An Introduction to Information Retrieval》经PDF→Markdown→分块后生成1,071问块，7,275对关系标签。

**📈 对比分析**

在KhanQ和SQuAD上分别计算BLEU、ROUGE‑L、BLEURT、BERTScore；Encoder‑Decoder模型表现最佳；关系分类模型宏平均F1≈0.90；生成DQM后手工抽取路径评估显示问题相关、层级清晰、图结构无循环，整体性能满足可视化与学习路径构建需求。

**⚠️ 局限性**

限制：生成问题质量受训练数据规模与多样性影响；特异性分类在“other”类表现不佳；图压缩过程中可能丢失细节；目前仅在模拟教材上验证，缺乏真实课堂使用与效果评估。

---

## 578. Quantum Optical Integrated Sensing and Communication with Homodyne BPSK Detection

**arXiv ID:** 2601.07034 | [PDF](https://arxiv.org/pdf/2601.07034v1)

**作者:** Ioannis Krikidis `[一作]` (University of Cyprus), Ioannis Krikidis `[通讯]` (University of Cyprus)

**通讯引用:** 10742 | [OpenAlex ID](https://openalex.org/A5080502122)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种在量子光学链路中使用二进制相位键控（BPSK）调制和同相干检波器进行联合通信与感知的量子集成感知通信（QISAC）框架。

**💡 创新点**

创新点在于：①将通信误码率（BER）与感知精度（Fisher信息）通过约束优化统一起来，揭示两者之间的根本权衡；②设计了双循环迭代算法——内部EM循环实现符号检测与相位估计，外部LO相位自适应调节，兼顾通信可靠性和相位估计精度；③在量子光学模型下首次给出BPSK同相干检波的闭式BER与Fisher信息表达式。

**🔧 技术方法**

主要技术包括：量子光学通信模型（BPSK相干态、相位失真通道、热噪声），同相干检波理论，Fisher信息与Cramér–Rao界，EM（期望最大化）算法与牛顿法求解参数，LO相位的迭代更新策略。

**📊 数据集**

本文未使用真实实验或公开数据集，而是基于Monte‑Carlo仿真，取典型参数E=10、η=0.8、N_a=3、N=1000或500等进行数值验证。

**📈 对比分析**

通过仿真比较，算法在满足给定Fisher信息阈值下能快速收敛，估计相位误差≤±1°，BER与理论曲线吻合；同时展示了BER与归一化Fisher信息的Pareto权衡曲线，证明了通信与感知目标的不可兼得性。

**⚠️ 局限性**

局限性包括：仅考虑BPSK调制与经典相干态，未实现量子优势；算法依赖高SNR假设，低SNR下精度可能下降；对快速相位漂移的鲁棒性未充分探讨，需要进一步扩展到多调制方案或非经典光源。

---

## 579. Solar Open Technical Report

**arXiv ID:** 2601.07022 | [PDF](https://arxiv.org/pdf/2601.07022v1)

**作者:** Sungrae Park `[一作]`, Alice Oh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本工作提出并训练了一个102B参数的双语Mixture-of-Experts LLM，专注于解决韩语等欠发达语言的能力瓶颈；

**💡 创新点**

创新点包括：4.5T高质量合成数据、基于质量、领域与语言的渐进式课程学习，以及解耦的SnapPO RL框架；

**🔧 技术方法**

技术栈涵盖MoE Transformer、BPE分词器、FSDP/HSDP并行、SnapPO离线RL、GSPO、DPO、vLLM、TorchTitan等；

**📊 数据集**

数据集来源于19.7T多语言预训练语料、4.5T合成韩语文本、海量多领域推理轨迹、对齐对话数据以及agent模拟轨迹；

**📈 对比分析**

在韩语金融、法律、医学基准上相较gpt‑oss‑120b领先3–9pp，英语基准与同级模型保持竞争力，推理与偏好对齐均优于基线；

**⚠️ 局限性**

局限性在于对极低资源语言的泛化尚未验证，数学推理仍低于专门化模型，RL奖励设计与探索效率需进一步改进。

---

## 580. Making Absence Visible: The Roles of Reference and Prompting in Recognizing Missing Information

**arXiv ID:** 2601.07234 | [PDF](https://arxiv.org/pdf/2601.07234v1)

**作者:** Hagit Ben Shoshan `[一作]` (University of Haifa), Osnat Mokryn `[通讯]` (University of Haifa)

**通讯引用:** 816 | [OpenAlex ID](https://openalex.org/A5034395195)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一项 100 名参与者的实验研究中，探讨了参考框架（部分样本 vs 全局汇总）和提示方式（自发 vs 指导）对在条形图可视化中识别缺失类别的影响。

**💡 创新点**

①首次提出基于期望的可视化理念，②系统验证部分样本框架能提升缺失检测，③通过实验揭示显式提示能显著克服“存在偏差”，④为交互可视化与智能界面提供期望显式化与引导注意的设计原则。

**🔧 技术方法**

采用 2×2 混合实验设计；使用 Google Gemini‑2.5‑flash LLM 自动标注参与者文本中的缺失信息；统计分析用卡方检验和 Cramér’s V 计算效应大小。

**📊 数据集**

三类离散数据集：政治体制分布、收入分布（日均人均收入区间）和能源生产分布（各能源来源占比），以竖直条形图呈现。

**📈 对比分析**

通过比较不同参考框架与提示组合下的缺失检测率：在自发阶段，部分样本框架比全局框架提升约 20%（显著；Cramér’s V≈0.20）；在提示阶段，缺失检测率均提升 40–60 个百分点，效应大小极大（Cramér’s V≈0.5–0.6）。两因素交互不显著，说明提示效果优于参考框架差异。

**⚠️ 局限性**

局限性包括：仅使用静态条形图和书面回答，样本仅为英语使用者，缺失检测依赖 LLM 自动标注可能遗漏细微表述，实验仅检验缺失提示而未考虑盈余提示，数据类型受限于离散类别，结果在不同文化或交互情境下的泛化性未知。

---

## 581. MI-PRUN: Optimize Large Language Model Pruning via Mutual Information

**arXiv ID:** 2601.07212 | [PDF](https://arxiv.org/pdf/2601.07212v1)

**作者:** Hao Zhang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于互信息的LLM块剪枝方法MI‑PRUN，利用互信息量衡量隐藏状态的转移信息，结合数据处理不等式（DPI）评估连续块重要性，并通过Fast‑Block‑Select算法实现全局最优剪枝。

**💡 创新点**

创新点在于：①使用互信息直接量化Transformer块对信息传播的贡献；②通过DPI把连续块与单块的重要性联系起来，避免误剪重要块；③提出的Fast‑Block‑Select迭代方法在保证全局最优的前提下显著降低计算成本。

**🔧 技术方法**

核心技术包括互信息估计、数据处理不等式分析、递归迭代剪枝与启发式Fast‑Block‑Select算法；实验采用了Transformer模型的前向推理与校准集输出。

**📊 数据集**

实验数据集涵盖：WikiText‑2、Alpaca（校准集）；评测基准包括Winogrande、PIQA、WSC、WNLI、SST‑2、RTE、QNLI、CB、ARC‑e、ARC‑c；还使用C4数据集进行困惑度评估。

**📈 对比分析**

与LLM‑Pruner、SliceGPT、ShortGPT等基线相比，MI‑PRUN在相同剪枝比例下保持或提升了大多数基准指标，且推理速度提升显著；实验表明其在不同模型（Llama2‑7B/13B、Qwen‑7B/14B）上均表现优异。

**⚠️ 局限性**

局限性包括：互信息估计仍需大量前向推理；DPI假设对非线性关系的解释有限；Fast‑Block‑Select的启发式参数选择仍依赖经验；在极端高剪枝率下，模型性能退化仍不可避免。

---

## 582. MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization

**arXiv ID:** 2601.07208 | [PDF](https://arxiv.org/pdf/2601.07208v1)

**作者:** Yang Zhao `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 44498 | [OpenAlex ID](https://openalex.org/A5100320723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MAESTRO框架，利用元学习动态调节多目标奖励的加权方式，以提升大语言模型在开放域生成中的对齐效果。

**💡 创新点**

创新点在于将奖励加权视为上下文感知的Bandit问题，使用终端隐藏状态作为语义瓶颈，构建轻量级的Conductor网络；通过双层优化结合GRPO的组相对优势作为元奖励，解决元信用分配难题，并通过异步双时间尺度更新保证训练稳定性。

**🔧 技术方法**

采用GRPO强化学习、元学习（上下文Bandit + 轻量级线性策略）、双层优化、异步双时间尺度更新、以及多目标奖励的动态加权。

**📊 数据集**

七个开放域基准，涵盖推理、创意写作、社交智能和多语言生成：Natural Reasoning、GeneralThoughts、WebInstruct、SS‑GEN、EmoBench、ToMBench、OPUS‑Books。

**📈 对比分析**

与预训练、SFT、CoT、GRPO变体（NOVER、EM‑GRPO）以及等权/随机权重的对比，MAESTRO在两大8B后端（Qwen3‑8B、Llama‑3.1‑8B‑Instruct）上均实现了最高的“偏好百分比”指标，并在部分任务中提升至约+20%训练吞吐率，显著优于静态或单一奖励方案。

**⚠️ 局限性**

局限包括对基础奖励质量的依赖、奖励空间仅限于五种预设目标、未在更大模型或MoE体系上验证、以及对奖励信号可能存在的偏差缺乏充分补偿。

---

## 583. Safeguarding LLM Fine-tuning via Push-Pull Distributional Alignment

**arXiv ID:** 2601.07200 | [PDF](https://arxiv.org/pdf/2601.07200v1)

**作者:** Haozhong Wang `[一作]`, Dandan Guo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出安全最优传输（SOT）框架，通过分布级别的权重学习实现对LLM细调过程中的安全防护，先筛选高质量数据，再对残留样本加权细调。

**💡 创新点**

创新点在于引入双参考“拉-推”机制：一侧以任务特定安全锚点拉近安全分布，另一侧以通用有害样本推远危险分布；同时将安全过滤视为最优传输问题而非单实例评估，显著提升数据分布对齐与安全边界。

**🔧 技术方法**

核心技术包括：离散最优传输（OT）及其熵正则化；利用冻结的安全对齐LLM提取样本表征；软硬加权策略（Top‑K 筛选 + 重权重细调）；LoRA 轻量细调；损失函数为拉推OT距离的加权组合。

**📊 数据集**

使用多种下游任务数据集（SLIMORCA、MetaMathQA、AGNews、GSM8K、LegalBench‑QA）以及红队有害样本集合（BeaverTails、HH红队子集）作为安全与有害参考。

**📈 对比分析**

与标准SFT、随机选择、ALPAGASUS、DSIR、SAFT、BDS、SEAL、Salora、SafeInstr 等方法对比；实验表明在 Meta‑Llama‑3.1‑8B‑Instruct、Qwen‑3‑8B 等模型上，SOT 在 Avg.、HpS、HmS 上均取得最优或相近的性能，尤其在安全分数 HmS 上显著低于其他基线，且在有害比例增大时仍保持稳定。

**⚠️ 局限性**

主要局限：依赖安全与有害参考数据集的质量与代表性；对新型攻击手段的适应性需更新参考集；缺乏对权重决策的自然语言可解释性；评估使用 LLM‑judge 可能带来偏差；对恶意逆向使用的潜在风险。

---

## 584. Defenses Against Prompt Attacks Learn Surface Heuristics

**arXiv ID:** 2601.07185 | [PDF](https://arxiv.org/pdf/2601.07185v1)

**作者:** Shawn Li `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**通讯引用:** 3190 | [OpenAlex ID](https://openalex.org/A5057711796)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大语言模型中使用监督微调进行提示注入防御时，模型对安全输入的误拒和性能下降的现象，并通过控制实验揭示了三种“捷径偏差”。

**💡 创新点**

创新点在于提出并系统评估了位置偏差、触发词偏差和主题泛化偏差三种超越攻击识别的误拒模式，并提供了针对这些偏差的诊断数据集与量化指标。

**🔧 技术方法**

技术主要包括监督微调（StrucQ、SecAlign 等）、外部提示守护器（ProtectAIv2、LakeraGuard 等）、以及对模型输出进行拒绝率与任务准确率的评估。

**📊 数据集**

数据集涵盖 GPT4‑QA、MMLU、AIME 等多领域推理基准，以及针对触发词的 InjectGuard、PINT、WildGuard 等安全评测数据，构造了位置、触发词和主题偏差的专门诊断集。

**📈 对比分析**

对比方法是把基线模型与微调/守护器模型在同一基准上同时测评拒绝率与准确率。结果显示，微调后模型在攻击样本上拒绝率提高，但在安全输入上误拒率可达 50%+、准确率下降 10%~40%，外部守护器亦存在类似问题。

**⚠️ 局限性**

限制包括：实验仅涵盖两种主流 LLM 与几类防御，未覆盖所有可能的攻击或部署场景；诊断集仍可能不足以覆盖所有表面特征；未探索如何在保持高拒绝率的同时彻底消除这些偏差的最佳训练策略。

---

## 585. Can Large Language Models Understand, Reason About, and Generate Code-Switched Text?

**arXiv ID:** 2601.07153 | [PDF](https://arxiv.org/pdf/2601.07153v1)

**作者:** Genta Indra Winata `[一作]` (Capital One), Emmanuele Chersoni `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一个包含16个语言对的代码切换基准数据集，并系统评估了大型语言模型在理解、推理与生成代码切换文本时的表现；

**💡 创新点**

创新之处在于：①将原生与音译两种书写方式统一纳入基准，②使用高质量人工标注的多语言数据；③首次通过推理轨迹分析揭示LLM在代码切换下的推理失配；④发现自然性与准确性之间的权衡。

**🔧 技术方法**

采用多种提示策略（随机切换、选择性切换、语法强制）生成代码切换文本；使用Code‑Mixing Index与Switch‑Point Fraction度量；利用LLM（如Qwen3‑30B‑A3B、GPT‑5.2）生成与评估文本，并用人类标注员进行自然性与语义准确性打分。

**📊 数据集**

数据来源为SimpleQA Verified，扩展生成64k条代码切换样本（每种语言4k条），并与公开数据集GLUECoS、LINCE对比；最终发布名为“CodemixQA”的公开数据集（链接：https://huggingface.co/datasets/gentaiscool/codemixqa）。

**📈 对比分析**

对比多款开源与专有LLM（Gemma、Llama、Qwen、Olmo、GPT‑OSS、GPT‑4.1），发现代码切换使F1平均下降约11%；语法强制源语法（Grammar Force (Src)）在理解任务中表现最佳，而目标语法强制（Grammar Force (Tgt)）在自然性与语义准确性评估中最优；推理轨迹显示LLM仍以英语为主，出现不支持主张、无逻辑跳跃等错误。

**⚠️ 局限性**

局限性包括：实验模型范围有限；使用AI助手协助撰写与校对；数据集选取基于当时难度，可能存在选择偏差；未覆盖所有语言对与语音数据，且对生成文本的多样性评估尚不充分。

---

## 586. PASS-Enabled Covert Communications With Distributed Cooperative Wardens

**arXiv ID:** 2601.07147 | [PDF](https://arxiv.org/pdf/2601.07147v1)

**作者:** Ji He `[一作]` (Xidian University), Ji He `[通讯]` (Xidian University)

**通讯引用:** 3399 | [OpenAlex ID](https://openalex.org/A5049728628)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出在双波导PASS（Pinching‑Antenna System）架构下的分布式协同监测环境中实现低可检测（covert）通信，并通过联合设计辐射功率、PA位移及随机对齐技术，最大化在最差检测阈值下的平均隐蔽速率。

**💡 创新点**

创新点包括：① 在非同分布（non‑i.i.d.）监视器统计下推导出多数投票融合规则的闭式系统检测误差概率（DEP）表达式；② 将PASS的三种功率-辐射模型（通用、比例、等量）与随机混淆信号统一到一体化框架；③ 设计了MM–BCD–SCA迭代算法，实现了对功率、辐射变量与PA位置的联合凸优化，且保留最差检测约束；④ 通过理论与仿真验证不同辐射模型对系统隐蔽性与速率的影响。

**🔧 技术方法**

使用的主要技术包括：概率生成函数（PGF）与基本对称多项式（ESP）求解DEP，MM（Majorization‑Minimization）与SCA（Successive Convex Approximation）构造下凹近似目标，Biconvex Block Coordinate Descent 对功率/位置进行交替优化，Gaussian随机化恢复秩‑一解，以及在一次维度搜索中对最差检测阈值进行近似求解。

**📊 数据集**

实验基于仿真设置，采用5/8个随机部署的守望者、5 GHz载频、4m波导、10×10米平面等参数，未使用公开数据集。

**📈 对比分析**

通过与粗粒度-细粒度网格搜索和随机搜索等基准对比，MM–BCD–SCA算法在覆盖度约束下平均隐蔽速率上获得与网格搜索相近的性能，并明显优于随机搜索；当守望者数量增多时速率下降，但算法仍保持鲁棒性；不同功率-辐射模型对DEP和速率表现出明显差异，等量模型在高密度守望者场景中最具隐蔽优势。

**⚠️ 局限性**

局限性包括：① 需要预先假设守望者检测阈值为最坏情况，实际阈值可能分布不同；② 对PASS功率-辐射模型的参数假设较为理想，实际工艺误差未考虑；③ 采用的离散化仿真参数有限，未验证在更大规模或多用户场景下的扩展性；④ SCA和MM近似在某些极端参数下可能收敛速度慢或陷入次优点。

---

## 587. EZBlender: Efficient 3D Editing with Plan-and-ReAct Agent

**arXiv ID:** 2601.07143 | [PDF](https://arxiv.org/pdf/2601.07143v1)

**作者:** Hao Wang `[一作]` (Clemson University), Abolfazl Razi `[通讯]` (Clemson University)

**通讯引用:** 2171 | [OpenAlex ID](https://openalex.org/A5011987346)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 EZBlender，结合计划与本地自治的 Blender 代理，实现高效、语义一致的 3D 场景编辑。

**💡 创新点**

创新点是 Plan-and-ReAct 混合架构，将任务规划与子代理自治结合，显著提升响应速度与多任务完成率。

**🔧 技术方法**

使用大型语言模型（GPT‑4o、GPT‑4o‑mini 等）与 Vision‑Language 模型、CLIP 评估；架构包含 Planner、Domain‑specific Sub‑Agents 与 Debug Agent。

**📊 数据集**

使用 BlenderGym 场景集（245 个预设场景）以及自建的 85 场景多任务基准与视觉提示基准。

**📈 对比分析**

与 BlenderGPT、BlenderAlchemy 对比，EZBlender 在文本/视觉提示精度、任务完成率和响应时延上均优于基线，最高可提升 7 倍速度、降低 67% token。

**⚠️ 局限性**

局限性包括对光照与相机的细粒度控制仍不够精准，受 VLM 空间感知与渲染闭环反馈限制；对复杂场景的多步骤协同仍有改进空间。

---

## 588. PROTEA: Securing Robot Task Planning and Execution

**arXiv ID:** 2601.07186 | [PDF](https://arxiv.org/pdf/2601.07186v1)

**作者:** Zainab Altaweel `[一作]`, Shiqi Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM的安全评估框架（PROTEA）用于检测机器人任务规划中的恶意计划，并通过对象过滤与外部记忆两大模块实现逐步推理。

**💡 创新点**

创新点在于：①将LLM用作安全裁判（LLM-as-a-Judge）；②结合对象过滤和逐步更新的外部记忆，有效克服维度灾难与历史长距离推理难题；③构造了全新的恶意计划数据集HarmPlan，覆盖六类危害和三难度级别。

**🔧 技术方法**

技术：大语言模型（GPT‑4o‑mini、GPT‑OSS‑120B、Grok‑3‑mini、LLaMA‑3.3‑70B、Mixtral‑8×22B、Phi‑4）实现安全裁判；对象过滤模块筛除无关环境信息；外部记忆模块维护执行状态；LLM交互式推理。

**📊 数据集**

数据集：HarmPlan，基于VirtualHome的750个基础计划注入91种恶意行为（直观与后果性），生成易、难、极难三层难度的正负样本。

**📈 对比分析**

与三种基线（Naive、Object‑Filtering、External‑Memory）对比，使用PROTEA在六款LLM上均获得高精度（≥90%），在多数模型上召回率提升30%以上；对难度递增的恶意计划，PROTEA表现稳健，特别是外部记忆方法对长序列恶意行为检测效果显著。

**⚠️ 局限性**

局限：对高对齐、Chain‑of‑Thought模型会产生过度谨慎、误报；部分模型在极难案例下召回仍低；需要进一步优化提示与参数以兼顾精确与召回；对实时执行时延与计算成本未充分评估。

---

## 589. Standardization of Post-Publication Code Verification by Journals is Possible with the Support of the Community

**arXiv ID:** 2601.07189 | [PDF](https://arxiv.org/pdf/2601.07189v1)

**作者:** Susana Lopez-Moreno `[一作]`, Sangil Kim `[通讯]` (Pusan National University)

**通讯引用:** 3192 | [OpenAlex ID](https://openalex.org/A5100398517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并阐述了在机器学习期刊层面实施后期发布验证的框架，允许独立研究者提交验证报告并为论文授予最多两枚可公开展示的徽章。

**💡 创新点**

创新点在于：①将ACM预发布验证徽章体系迁移到后期验证，形成可追溯、可公开的验证记录；②引入两枚徽章机制，既奖励复制也奖励重现；③将验证信息嵌入文章元数据，提升透明度和学术可信度；④强调社区驱动与期刊合作的可持续性。

**🔧 技术方法**

技术手段主要是：ACM徽章系统的修改、元数据标注、代码仓库链接、轻量级内部审核流程；并未涉及具体机器学习算法或实验实现。

**📊 数据集**

本文为位置论文，未实际使用特定数据集；仅提及常见的公开数据源（如UCI、EHR等）作为潜在验证素材，供后期验证者参考。

**📈 对比分析**

由于是概念性研究，没有实验对比；通过理论分析和案例讨论说明方案的可行性、预期提升的可信度以及对科研流程的正面影响。

**⚠️ 局限性**

局限性包括：额外的编辑与审核工作、完整可复制性仍受限、信息误传与滥用风险、资源与数据访问不均导致的参与壁垒、保密与数据隐私限制、可能引发的骚扰或滥用行为、对已有社区平台的重复与协调难度。

---

## 590. From "Thinking" to "Justifying": Aligning High-Stakes Explainability with Professional Communication Standards

**arXiv ID:** 2601.07233 | [PDF](https://arxiv.org/pdf/2601.07233v1)

**作者:** Chen Qian `[一作]` (William and Mary), Andreas Stathopoulos `[通讯]` (William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“Result→Justify”框架，用CREAC/BLUF结构化AI解释，先给出结论再论证；

**💡 创新点**

创新点在于将专业沟通模板转化为输出约束，并设计六维结构化可解释性指标；

**🔧 技术方法**

采用提示工程、贪婪解码的开源12–14B指令调优LLM（DeepSeek、Gemma、Ministral、Qwen）与指标评估；

**📊 数据集**

在法律、医疗、金融三个领域的Yes/No任务上使用四个公开数据集（Financial PhraseBank、ConsumerQA、Hearsay、PubMedQA）；

**📈 对比分析**

与Direct、CoT、ToT、CoVe、V‑RAG、Self‑RAG等基线比较，平均准确率83.9%，比CoT高5.3个百分点；

**⚠️ 局限性**

局限性包括指标仅衡量结构合规性、无法保证语义正确性、实验仅覆盖三类域和12–14B模型，未验证多类或大模型场景；

---

## 591. Consolidation or Adaptation? PRISM: Disentangling SFT and RL Data via Gradient Concentration

**arXiv ID:** 2601.07224 | [PDF](https://arxiv.org/pdf/2601.07224v1)

**作者:** Yang Zhao `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 38260 | [OpenAlex ID](https://openalex.org/A5100418162)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PRISM框架，通过分析梯度的空间浓度动态分配数据到SFT或RL，实现更高效的LLM代理训练。

**💡 创新点**

创新点在于将梯度浓度视为认知冲突指标，基于Schema理论实现数据的自适应分配，摆脱传统粗糙的启发式划分。

**🔧 技术方法**

采用非侵入式梯度探测、Gini/Kurtosis/CV浓度计算、分位数阈值分配，以及SFT与RL（GRPO）联合训练等技术。

**📊 数据集**

在WebShop与ALFWorld这两个代理基准上进行实验。

**📈 对比分析**

与SFT、GRPO、GiGPO、随机/HPT、SFT-then-RL等基线对比，PRISM在ALFWorld上成功率达95.31%，在WebShop上得分85.15，训练速度提升至3.22×。

**⚠️ 局限性**

局限性包括仅在7B–8B模型验证，采用静态分配未考虑训练过程动态变化，以及仅在代理决策任务上验证，缺乏对更广泛复杂任务的评估。

---

## 592. Language-Grounded Multi-Domain Image Translation via Semantic Difference Guidance

**arXiv ID:** 2601.07221 | [PDF](https://arxiv.org/pdf/2601.07221v1)

**作者:** Jongwon Ryu `[一作]` (Chung-Ang University), Junyeong Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 736 | [OpenAlex ID](https://openalex.org/A5021487107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 LACE 框架，实现基于语言提示的多域图像‑图像翻译，支持多属性可组合控制并保持结构一致性。

**💡 创新点**

创新点包括：1) Global‑Local Image Prompt Adapter（GLIP‑Adapter）融合全局 CLIP 与局部 DINOv2 特征，实现结构保留；2) Multi‑Domain Control Guidance（MCG）通过源‑目标提示差异生成噪声差分并按属性加权，实现每个属性可调节的强度；3) 两者组合实现了无需分割或标注的多属性编辑与可解释的属性级控制。

**🔧 技术方法**

使用技术：扩散模型 Stable Diffusion v2.1；CLIP‑ViT‑H/14 和 DINOv2‑Large 提取视觉特征；GLIP‑Adapter 线性投影融合；MCG 采用噪声差分与可变尺度 s；DDIM 逆向推断实现精细结构恢复。

**📊 数据集**

实验数据集：CelebA(Dialog)（面部属性如表情、性别、发型）和 BDD100K（天气、时间、场景风格）。

**📈 对比分析**

与 SDEdit、DDIM+PnP、Direct+MasaCtrl、IP‑Adapter 等基线比较，评估指标包括 FID、FIDclip、StructureDistance、PSNR、LPIPS、CLIP Similarity 等；LACE 在大多数指标上达到最优或第二优，且在人类评价中在多属性编辑下保持高准确度和自然度。

**⚠️ 局限性**

限制：推理时需要为每个属性多次噪声预测，计算成本较高；对属性冲突敏感，需手工设置属性权重；实验仅覆盖 CelebA、BDD100K 与动物面部数据，未在更广泛领域验证；对提示的语言歧义敏感，需进一步提升鲁棒性。

---

## 593. SceneNAT: Masked Generative Modeling for Language-Guided Indoor Scene Synthesis

**arXiv ID:** 2601.07218 | [PDF](https://arxiv.org/pdf/2601.07218v1)

**作者:** Jeongjun Choi `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**通讯引用:** 6984 | [OpenAlex ID](https://openalex.org/A5073996122)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出SceneNAT，一种基于掩码的非自回归Transformer，用自然语言指令快速生成完整的3D室内场景；

**💡 创新点**

创新点在于掩码室内场景建模（MISM）和专门的三元组预测器，将关系推理与场景解码分离，显著提升了对复杂空间关系的把握；

**🔧 技术方法**

采用掩码自回归学习、Transformer解码器、双层关系预测（Triplet Predictor）、向量量化的语义与空间离散化、迭代掩码重预测策略；

**📊 数据集**

主要使用扩展版3D-FRONT数据集（含指令和多关系标注），并在该数据集上进行评测；

**📈 对比分析**

与ATISS、DiffuScene、InstructScene等基线对比，SceneNAT在iRecall、FID、碰撞体积等多项指标上均优于对手，同时推理速度提升5-25倍，参数和算力降低；

**⚠️ 局限性**

局限性包括对极复杂多关系指令的鲁棒性仍受文本编码长度限制、离散化分辨率影响生成精度、尚未支持多模态输入（图像、草图）等。

---

## 594. SIRR-LMM: Single-image Reflection Removal via Large Multimodal Model

**arXiv ID:** 2601.07209 | [PDF](https://arxiv.org/pdf/2601.07209v1)

**作者:** Yu Guo `[一作]` (Futurewei Technologies), Heather Yu `[通讯]` (Futurewei Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于路径跟踪的3D玻璃模型与真实背景图像合成物理逼真的反射数据集，并使用该数据集对大型多模态模型Flux进行LoRA微调，完成单图像反射去除与分离任务。

**💡 创新点**

创新点在于①采用物理渲染生成高保真玻璃反射样本，①无需完整三维场景即可获得多种光照与玻璃属性；②通过将反射与透射层拼接成单一复合图像，并使用统一提示进行训练，避免传统需要多图提示的局限；③利用LoRA在少量高质量样本上快速微调大模型，实现高性能。

**🔧 技术方法**

使用路径跟踪渲染、HDR环境贴图、物理基BSDF模型、LoRA微调、Flux多模态生成模型以及统一提示策略。

**📊 数据集**

自研的合成数据集，包含约1000对图像（原图、透射层、反射层、镜像层等），并结合公开真实基准数据集Real、Nature、SIR^2进行评测。

**📈 对比分析**

与DSRNet、Dong、Zhu、DSIT、RDNet等最新单图像反射去除方法对比，PSNR/SSIM在三个真实基准上均优于大多数方法，LPIPS在反射区域提升18.2%，用户研究中对我方方法的胜率约45%。

**⚠️ 局限性**

局限性包括渲染过程需要大量计算资源，且在白色背景区域可能误判为高光或反射，导致轻微去除错误。

---

## 595. RAIRS: Optimizing Redundant Assignment and List Layout for IVF-Based ANN Search

**arXiv ID:** 2601.07183 | [PDF](https://arxiv.org/pdf/2601.07183v1)

**作者:** Zehai Yang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences University of Chinese Academy of Sciences), Shimin Chen `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences University of Chinese Academy of Sciences)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在IVF索引基础上提出RAIRS方案，改进冗余分配和列表布局，从而提升近似最近邻搜索的吞吐量和效率。

**💡 创新点**

创新点包括：①基于欧氏空间的AIR（Amplified Inverse Residual）度量，用于更精确地挑选第二个列表；②SEIL共享单元列表布局，通过共享大单元块来减少重复距离计算与空间开销；③对多列表分配（SRAIR）进行评估，并证明两份分配是最优的。

**🔧 技术方法**

技术手段：IVF-PQ Fast Scan + Refinement基础上实现；使用SIMD向量化计算、批量处理、缓存优化和哈希去重；采用AIR度量进行列表选择；实现SEIL共享块的内存布局和去重逻辑。

**📊 数据集**

实验数据集：SIFT1M、SIFT1B、MSong、GIST、OpenAI（欧氏空间）以及T2I（内积空间）用于验证SEIL在不同度量下的适用性。

**📈 对比分析**

与IVF、HNSW、IVFPQfs、NaïveRA、SOARL2、RAIR/SRAIR等方法进行Recall‑QPS和Recall‑DCO对比；RAIRS在所有数据集上实现1.07–1.33×吞吐量提升、0.64–0.83×距离计算降低，并在top‑1/10/100搜索上保持与最优方法相同的召回率；插入/删除操作略慢但在可接受范围内。

**⚠️ 局限性**

局限性：仅针对主存驻留环境；对欧氏空间优化，非欧氏空间需要额外调整；SEIL实现较为复杂，受块大小和大单元分布影响；多列表分配虽然理论可行，但实际多于两份时效果不明显。

---

## 596. DIVER: Dynamic Iterative Visual Evidence Reasoning for Multimodal Fake News Detection

**arXiv ID:** 2601.07178 | [PDF](https://arxiv.org/pdf/2601.07178v1)

**作者:** Weilin Zhou `[一作]` (Xinjiang University), Xiangzheng Zhang `[通讯]` (360 AI Security Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种动态迭代的视觉证据推理框架 DIVER，用于多模态假新闻检测。

**💡 创新点**

创新点包括：1）先基于文本构建可靠的语义基线，再通过跨模态对齐门控决定是否进入深度视觉推理；2）采用可回溯的视觉证据抽取与反馈机制；3）引入不确定性感知的动态融合，避免无效模态干扰。

**🔧 技术方法**

技术手段包括：大型语言模型（LLM）分析与自纠正、CLIP 对齐、OCR/密集描述等视觉工具、注意力门控与不确定性权重融合、双重反思（跨模态与跨级别）机制。

**📊 数据集**

使用中文微博数据集 Weibo、Weibo21 以及英文 GossipCop 三个公开基准进行评估。

**📈 对比分析**

与多模态、单模态及 LLM 推理基线相比，DIVER 在所有数据集上均取得最高准确率（Weibo 95.8%，Weibo21 97.7%，GossipCop 91.6%），并将平均推理延迟降低到 4.12 秒，明显优于 GLPN‑LLM、INSIDE、LIFE 等方法。

**⚠️ 局限性**

局限性包括：对更多社交媒体域的泛化能力待提升；在更复杂真实场景和未知对抗攻击下的鲁棒性尚未充分验证；模型规模大，适配边缘设备仍需轻量化蒸馏技术。

---

## 597. Motion Focus Recognition in Fast-Moving Egocentric Video

**arXiv ID:** 2601.07154 | [PDF](https://arxiv.org/pdf/2601.07154v1)

**作者:** Daniel Hong `[一作]` (Clemson University), Abolfazl Razi `[通讯]` (Clemson University)

**通讯引用:** 2171 | [OpenAlex ID](https://openalex.org/A5011987346)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种实时运动焦点识别方法，利用自举摄像机姿态估计推断人体运动趋势并生成运动焦点地图。

**💡 创新点**

创新点在于利用物理先验的加速度投影生成运动焦点，无需额外训练，可嵌入任何摄像头姿态估计框架，并通过滑动窗口批处理实现实时推理。

**🔧 技术方法**

使用了基础模型Depth Anything 3进行姿态估计、滑动窗口批处理与增量锚定技术，以及加速度投影与高斯核聚合生成焦点图。

**📊 数据集**

使用了作者自收集的轻量级第一人称动作视频数据集，涵盖步行、滑板、骑行等多种场景，分辨率与帧率多样。

**📈 对比分析**

与单批推理对比，滑动窗口方案实现约30+ FPS，GPU内存<5GB，可在消费级硬件上实时部署；实验显示运动焦点能有效突出运动相关区域。

**⚠️ 局限性**

局限性包括对极低帧率或高度抖动视频效果有限，依赖基础模型精度，且尚未在多摄像头或多目标场景下验证。

---

## 598. Intercultural Communication Strategies of a Technology Brand: A Comparative Quantitative Analysis of Xiaomi's Digital Marketing in China and Russia

**arXiv ID:** 2601.07204 | [PDF](https://arxiv.org/pdf/2601.07204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 599. Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation (Extended Version)

**arXiv ID:** 2601.07152 | [PDF](https://arxiv.org/pdf/2601.07152v1)

**作者:** Aja Khanal `[一作]` (University of Western Ontario), Apurva Narayan `[通讯]` (University of Western Ontario)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了Agents of Diffusion（AoD）框架，利用多智能体强化学习和自然语言反馈在不微调扩散语言模型的前提下控制生成符合 JSON schema 的结构化文本。

**💡 创新点**

创新点包括：①用自然语言反馈代替标量奖励，保持可解释性；②将冻结的扩散模型与自回归 LLM 的 Prompt Optimizer 与 Judge 协同工作，实现无规则、无微调的可控生成；③通过多轮交互实现结构精准与多样性兼顾。

**🔧 技术方法**

技术手段：多智能体强化学习（PPO+REINFORCE）、扩散语言模型 LLaDA‑8B、LLaMA‑3.1、Qwen‑3、DeepSeek‑R1、Gemma‑2 等自回归 LLM 作为 Prompt Optimizer 与 Judge、自然语言评估器（NLE）、JSON 结构验证器、离线评估指标（BLEU、ROUGE、METEOR、TSR、Field Overlap 等）。

**📊 数据集**

使用四个公开 JSON 结构化数据集：MultiWOZ、Super‑NaturalInstructions、Self‑Instruct、TruthfulQA。

**📈 对比分析**

对比六类基线（Diffusion‑LM、DiffLM、UniGen、PromptBreeder、EvoPrompt、CodecLM）和多款 LLM 单次推理；在同等硬件（AMD Ryzen 9 7900X + RTX 4080 SUPER）下，AoD 取得最高 Task Success Rate（0.79）和最低 Field Overlap（0.29），同时保持高相似度、丰富多样性、低困惑度，优于所有基线。

**⚠️ 局限性**

局限性：目前仅针对 JSON 结构化文本；扩散模型保持冻结，无法自适应；多轮 LLM 交互导致生成速度相对较慢；在更大规模或更复杂结构（如多层嵌套表格、代码）上的适用性尚未验证。

---

## 600. MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models

**arXiv ID:** 2601.07141 | [PDF](https://arxiv.org/pdf/2601.07141v1)

**作者:** Xi Ye `[一作]` (Wuhan University), Jiayi Yu `[通讯]` (Wuhan University)

**通讯引用:** 258 | [OpenAlex ID](https://openalex.org/A5101872220)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MacPrompt，一种黑盒跨语言攻击框架，通过构造混合语言的字符级“马卡罗尼词”绕过文本与模型级安全过滤，生成 NSFW 或禁用对象图像。

**💡 创新点**

创新点在于：① 以字符级子串拼接的跨语言混合词保持视觉语义但改变文本表示；② 仅依赖黑盒查询和零阶优化，无需模型内部信息，兼顾文本和概念移除两类防御；③ 通过对多语言候选的筛选与组合实现高语义相似度与低过滤检测率。

**🔧 技术方法**

使用技术包括：跨语言翻译与候选筛选、字符级子串提取与排序、零阶优化（ZOO）求解 β、α 参数、NSFW 检测器评估、CLIP/BLIP 等多模态相似度计算。

**📊 数据集**

使用的数据集：NSFW‑200 与 Object‑200（来源于 DeepSeek‑V3，结合 I2P 与 MS‑COCO 词表），以及标准的 CLIP/BLIP 评估集。

**📈 对比分析**

通过与 7 种现有黑盒攻击（DACA、DiffZOO、ART、Position、MMP‑Attack、PGJ、SurrogatePrompt）以及多种安全过滤（黑名单、BERT、LatentGuard）和 9 种概念移除模型（ESD、SLD、FMN、SafeGen、DUO、EAP、PromptGuard 等）对比；在 NSFW 生成上 BPR 100%、ASR 92‑98%，在禁用对象生成上优于 MMP‑Attack 并取得 94‑100% 的成功率，表现出显著优势。

**⚠️ 局限性**

局限性：① 主要针对文本级过滤，无法直接对图像级安全过滤或更复杂的多模态模型构成威胁；② 对于极少数语言或拼写变体的覆盖可能不足；③ 依赖于跨语言翻译质量和候选筛选过程，若翻译错误或语义不一致可能导致攻击失败；④ 未探讨对抗性鲁棒性的长期可扩展性。

---

## 601. The Roots of Performance Disparity in Multilingual Language Models: Intrinsic Modeling Difficulty or Design Choices?

**arXiv ID:** 2601.07220 | [PDF](https://arxiv.org/pdf/2601.07220v1)

**作者:** Chen Shani `[一作]` (Stanford University), Ekaterina Shutova `[通讯]` (University of Amsterdam)

**通讯引用:** 4351 | [OpenAlex ID](https://openalex.org/A5016184654)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对跨语言多语种语言模型性能差异进行系统综述，分析差异来源是模型设计（分词、编码、数据分配、共享参数）而非语言内在难度，并提出针对性设计建议。

**💡 创新点**

创新点在于将语言学特征（正字法、形态学、词汇多样性、句法、信息量、类型距离）与模型失配机制映射，提出多维度技术对策，并将评估方法从单一子词困惑度转向字符/字节/形态学级别与类型感知探测。

**🔧 技术方法**

主要技术包括：字节/字符级或形态学感知分词器（BPE、WordPiece、Morfessor、Script-BPE 等），字节归一化采样、信息归一化采样，模块化或自适应参数分配（语言适配器、路由网络），以及类型感知的评估指标与基准（CLAMS、SIGMORPHON、子词/字符困惑度、探测任务）。

**📊 数据集**

利用公开多语言语料（Web 语料、WikiText、Common Crawl 等）以及语言学资源（UniMorph、WALS、Treebanks）进行预训练与评估；在基准上采用跨语言任务数据（下游任务、生成任务、形态分析任务）。

**📈 对比分析**

比较方法通过对同一模型在不同分词/采样设置下的子词、字符、字节困惑度、下游任务准确率以及探测指标来评估；实验表明在统一的语义覆盖下，跨语言差距显著缩小，且形态学感知或字节归一化设置可在不牺牲高资源语言性能的前提下提升低资源/非拉丁文字的表现。

**⚠️ 局限性**

局限性包括：聚焦于表示和架构层面，未充分考虑语用、语篇、社会语言学因素；评估多主要基于大规模预训练模型和标准数据集，可能低估极低资源或濒危语言的挑战；架构与基准仍以英语为中心，可能偏见句法与位置编码的影响；信息理论指标主要与正字法/形态学相关，缺乏层级结构、语篇可预测性等度量；综述可能遗漏相关研究。

---

## 602. Forward versus Backward: Comparing Reasoning Objectives in Direct Preference Optimization

**arXiv ID:** 2601.07199 | [PDF](https://arxiv.org/pdf/2601.07199v1)

**作者:** Murtaza Nikzad `[一作]`, Raghuram Ramanujan `[通讯]` (Davidson)

**通讯引用:** 285 | [OpenAlex ID](https://openalex.org/A5007877912)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用直接偏好优化（DPO）在前向链式推理和后向验证两个训练目标上进行实验，评估它们对推理准确性与错误识别能力的影响。

**💡 创新点**

揭示前向与后向DPO训练提供互补的学习信号，同时发现偏好优化会降低模型自我识别错误的能力，首次系统比较两类训练目标的利弊。

**🔧 技术方法**

采用直接偏好优化（DPO）、低秩适配（LoRA）以及两阶段温度调控的推理流程，配合混合梯度和加权偏好训练。

**📊 数据集**

以 GSM8K（小学数学推理题）数据集为训练和评估数据，使用来自 LLaMA‑3.1‑8B‑Instruct 的教师模型生成前向和后向示例。

**📈 对比分析**

对比基线、仅前向 DPO、仅后向 DPO 三种设置：前向 DPO 将准确率提升 3.5pp（从 83.1% 到 86.6%），后向 DPO 将误报率从 13.4% 降至 4.3%，但两种训练都显著降低了识别错误率。

**⚠️ 局限性**

实验仅在 GSM8K 上进行，训练数据来自同一模型家族，缺乏跨域验证，且偏好优化导致模型自我纠错（acknowledgement rate）下降，需进一步研究如何保持或提升错误识别能力。

---

## 603. TranSC: Hardware-Aware Design of Transcendental Functions Using Stochastic Logic

**arXiv ID:** 2601.07172 | [PDF](https://arxiv.org/pdf/2601.07172v1)

**作者:** Mehran Moghadam `[一作]` (Case Western Reserve University), M. Hassan Najafi `[通讯]` (Case Western Reserve University)

**通讯引用:** 1273 | [OpenAlex ID](https://openalex.org/A5012903661)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于随机计算的轻量级Transcendental函数实现框架 TranSC，并在硬件上实现了多种三角、双曲、激活及对数等函数。

**💡 创新点**

创新点在于用 Van der Corput 低差序列生成器取代传统伪随机数发生器，既消除了中间延迟块、单一 BSG 单元，又显著提升了计算精度与硬件效率。

**🔧 技术方法**

采用随机计算 (SC) + VDC‑2ⁿ 低差序列 + Horner 规则多项式近似 + CMOS 低功耗电路，系统性评估了比特流长度、误差、面积、功耗与能耗。

**📊 数据集**

评估使用了 QR 码图像数据集（用于图像变换）和 2 关节机器人运动学坐标集；函数误差测量覆盖了 [0,1] 区间内所有 N‑bit 取值。

**📈 对比分析**

通过 MSE、面积、CPL、功耗、能耗以及 FoM 指标与 SOTA SC、Sobol、LFSR、CORDIC、PPI 进行对比，TranSC 在面积、功耗、能耗分别下降 33%、72%、64%，MSE 下降 98% 以上，FoM 明显优于所有对比方案。

**⚠️ 局限性**

局限性包括仍受比特流长度限制的精度瓶颈、串行处理导致的能耗相对较高，以及在需要极高精度或迭代实时性能的场景下不及传统 CORDIC/PPI 方法。

---

## 604. Stable On-Policy Distillation through Adaptive Target Reformulation

**arXiv ID:** 2601.07155 | [PDF](https://arxiv.org/pdf/2601.07155v1)

**作者:** Ijun Jang `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**通讯引用:** 2395 | [OpenAlex ID](https://openalex.org/A5065728469)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 Veto 的目标级重构方法，用来在对抗式自策略知识蒸馏中构造教师与学生之间的几何桥梁，从而解决前向 KL 梯度爆炸与反向 KL 模式坍塌的问题。

**💡 创新点**

创新点在于通过在 logits 空间构造中间目标分布 Q(y|x) ∝ exp(z_T + β z_S)，单一可调参数 β 既可实现自适应梯度阻断（前向 KL）又能调节决策度与熵正则（反向 KL），使得同一目标能够兼顾两种 KL 的优缺点。

**🔧 技术方法**

所用技术包括目标重构、指数加权分布、KL 损失的前向/反向两种形式、线性 β 衰减调度以及基于优势函数的 RL 风格梯度计算。

**📊 数据集**

实验采用了 GSM8K、WizardCoder、HumanEval、DialogSum 等多种任务数据集，涵盖数学推理、代码生成与对话摘要三大场景。

**📈 对比分析**

与监督 KD、SKD、传统对抗 KD 等基线对比，Veto 在数学推理任务中提升约 4‑6% 的准确率，在代码生成中 Pass@1 与 Pass@10 分别提升约 0.7 与 5.8 点，在摘要任务中 win‑rate 提升 2‑3% ，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括需要手动选择 β 的初始值与衰减策略，对极端多模态或长序列任务的泛化仍有待进一步验证，且在训练早期仍可能出现梯度波动。

---

## 605. AdaField: Generalizable Surface Pressure Modeling with Physics-Informed Pre-training and Flow-Conditioned Adaptation

**arXiv ID:** 2601.07139 | [PDF](https://arxiv.org/pdf/2601.07139v1)

**作者:** Junhong Zou `[一作]` (Institute of Automation), Xiangyu Zhu `[通讯]` (Institute of Automation)

**通讯引用:** 5991 | [OpenAlex ID](https://openalex.org/A5100632822)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 AdaField 框架，在汽车大规模数据上预训练并在列车与飞机等数据稀缺领域进行高效迁移，以预测表面压强场。

**💡 创新点**

创新点在于结合 Semantic Aggregation Point Transformer、Flow‑Conditioned Adapter 与 Physics‑Informed Data Augmentation，实现跨域自适应、参数高效微调和物理一致性扩增。

**🔧 技术方法**

使用技术包括点云自注意力、Slot Attention 语义聚合、轻量级 Adapter、基于 Navier‑Stokes 的尺度/速度数据增强以及 U‑Net 结构。

**📊 数据集**

使用的数据集为公开的 DrivAerNet++ 汽车数据集、少量高铁/磁悬浮列车数据集和航空机翼数据集。

**📈 对比分析**

与 RegDGCNN、Transolver、FigConvNet、TripNet 等 SOTA 进行 MSE/MAE/MaxAE 等指标对比，AdaField 在 DrivAerNet++ 上 MSE 降至 4.58×10⁻²，列车和机翼领域微调后误差比从零开始训练降低约 1/3，显示显著性能提升。

**⚠️ 局限性**

局限性包括对预训练域相似性的依赖，极端流动或尺寸差异下迁移效果受限；PIDA 基于稳态不可压假设，难以直接适用于可压或瞬态流。

---

## 606. Measuring Iterative Temporal Reasoning with TimePuzzles

**arXiv ID:** 2601.07148 | [PDF](https://arxiv.org/pdf/2601.07148v1)

**作者:** Zhengxiang Wang `[一作]` (Stony Brook University), Zeyu Dong `[通讯]` (Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Time Puzzles 这一基于约束的日期推理基准，用于评估 LLM 在工具辅助下的迭代时间推理能力。

**💡 创新点**

创新点在于设计可动态生成、可持续评估的约束式日期推理任务，并聚焦工具驱动的迭代推理，区别于传统单次推理基准。

**🔧 技术方法**

采用基于模板的事实约束生成、熵驱动搜索求解、零-shot chain-of-thought 提示，以及 Web 搜索和代码解释器等工具技术。

**📊 数据集**

构建两套 600 题的数据集（隐式与显式约束），涵盖 1-6 个解集，基于公开历史事件与日历关系生成。

**📈 对比分析**

在 13 大模型（含 GPT-5、GPT-4.1 等）下评估精确匹配 (EM)；无工具时 GPT-5 仅 49.3%，有工具时提升至约 80% 但仍不足；Web 搜索和 CI 取得一定改善但未完全弥补隐式约束难点。

**⚠️ 局限性**

局限性包括生成约束可能缺乏真实世界复杂性；仅评估现成模型与工具，未设计高级工具策略；缺少无解问题且生成方法依赖模板。

---

## 607. Generating readily synthesizable small molecule fluorophore scaffolds with reinforcement learning

**arXiv ID:** 2601.07145 | [PDF](https://arxiv.org/pdf/2601.07145v1)

**作者:** Ruhi Sayana `[一作]` (Stanford University), Kyle Swanson `[通讯]` (Stanford University)

**通讯引用:** 5436 | [OpenAlex ID](https://openalex.org/A5005590626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研发了 SyntheFluor‑RL，一种结合强化学习与图神经网络的生成式 AI，用于在 320 亿分子化学空间中快速设计可合成、光谱性能良好的荧光染料。

**💡 创新点**

创新点在于：①在生成过程中嵌入已知的合成反应库与构建块，确保产物可合成；②构建多目标评分函数（PLQY、吸收/发射波长、π‑共轭网络大小），并动态调节权重；③首次通过强化学习实现了多种荧光特性与合成可行性的联合优化。

**🔧 技术方法**

技术包括：图神经网络（Chemprop‑Morgan）与多层感知机（MLP‑Morgan）用于属性预测；强化学习（SyntheMol‑RL）用于分子构建；深度学习动态权重与温度调节；TD‑DFT 计算做最终筛选；分子相似度聚类与Tanimoto过滤；实验测定荧光量子产率、寿命、Stokes 位移等。

**📊 数据集**

数据集为 ChemFluor（2,912 分子、4,336 分子-溶剂对）用于训练属性预测模型；Enamine REAL Space（30 亿分子）作为可合成化学空间；对比实验用 Enamine 合成的 14 结构并在细胞内验证。

**📈 对比分析**

与先前的生成式荧光染料方法（ChemTS、FLAME）比较，SyntheFluor‑RL 在 16.5 小时内生成 11,590 个候选（相比 ChemTS 的 3,643、FLAME 的 1,000,000），并在 19 结构中筛选出 13 个可合成并实验验证，其中亮度最高的染料 PLQY 0.62、Stokes 位移 97 nm、寿命 11.5 ns，优于多种市售蓝光荧光染料。

**⚠️ 局限性**

局限性包括：属性预测模型仍受训练数据规模与波段范围限制；合成可行性评估依赖 Enamine REAL Space，若构建块不足会受限；最终筛选仍需要昂贵的 TD‑DFT 与实验验证；染料对不同溶剂和生物环境的适应性尚未系统评估。

---

## 608. Sentiment Analysis on Movie Reviews: A Deep Dive into Modern Techniques and Open Challenges

**arXiv ID:** 2601.07235 | [PDF](https://arxiv.org/pdf/2601.07235v1)

**作者:** Agnivo Gosai `[一作]` (Independent Researcher), Karun Thankachan `[通讯]` (Carnegie Mellon University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了电影评论情感分析的技术演进与挑战，从词典到传统机器学习，再到深度学习、Transformer 与大型语言模型，并聚焦多模态、跨语言与可解释性等新方向。

**💡 创新点**

创新点在于以挑战驱动的视角对比不同模型在讽刺检测、领域漂移、长文本处理、解释性与资源效率等五大难题上的表现，并提出多维评估框架和未来研究路线。

**🔧 技术方法**

所述技术涵盖规则/词典方法、SVM/Naïve Bayes、CNN/RNN/LSTM、BERT/RoBERTa/XLNet、MMA/MOSEI 等多模态 Transformer 以及 GPT‑4 等大型语言模型，并辅以注意力、层次注意等机制。

**📊 数据集**

主要参考数据集包括 IMDb、Rotten Tomatoes、SST‑2、CMU‑MOSI/MOSEI、GoEmotions、TweetEval 等传统与多模态、情感细粒度、跨平台的标注集合。

**📈 对比分析**

通过与现有基准的对比，传统机器学习模型在精度上远低于 Transformer/LLM，但在解释性、训练速度和模型压缩方面更具优势；作者提出在准确率之外加入领域迁移、情感细粒度、资源效率等指标来全面衡量。

**⚠️ 局限性**

局限性主要体现在缺乏统一的多模态/多语言多维评测标准、模型对讽刺与领域漂移等语境依赖性现象仍易失效、以及对资源受限环境的部署仍存在显著挑战。

---

## 609. Lost in the Noise: How Reasoning Models Fail with Contextual Distractors

**arXiv ID:** 2601.07226 | [PDF](https://arxiv.org/pdf/2601.07226v1)

**作者:** Seongyun Lee `[一作]` (KAIST AI), Minjoon Seo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了语境噪声对推理模型性能的影响，构建了包含多种噪声类型的综合基准，并提出基于Rationale-Aware Reward的鲁棒性提升方法。

**💡 创新点**

创新点包括：①设计了统一的噪声基准；②揭示代理工作在噪声环境下更易失效；③提出奖励机制鼓励模型识别有用信息，从而显著提升鲁棒性。

**🔧 技术方法**

技术手段涵盖了Prompting、SFT、RL（GRPO）以及自定义奖励函数，并辅以注意力、entropy等分析工具。

**📊 数据集**

使用的数据集包括11个RAG、推理、对齐、工具使用基准（SealQA、MultihopRAG、Musique等）及其噪声扩展，并构建了Rationale-Noise数据集。

**📈 对比分析**

通过对比多模型在无噪声与噪声环境下的harmonic mean得分，RL+RARE方法平均提升10-20%，并在噪声情境中恢复超过80%原始性能。

**⚠️ 局限性**

局限性在于奖励设计依赖判定模型，未能完全自适应动态噪声，且实验规模受算力限制，未来需进一步探索更通用的噪声对抗策略。

---

## 610. LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing

**arXiv ID:** 2601.07206 | [PDF](https://arxiv.org/pdf/2601.07206v1)

**作者:** Hao Li `[一作]` (Northwestern Polytechnical University), Shuyue Hu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了大型统一的 LLM 路由基准 LLMRouterBench，并在此基准上对多种路由方法进行系统评估。

**💡 创新点**

创新点在于：①整合 21 个多领域数据集、33 种 LLM（含 13 款旗舰模型），共 400K+ 实例；②提供统一的性能、成本与延迟评价指标；③公开 10 个代表性路由基线与完整评测框架，显著降低研究门槛；④揭示路由方法在统一评测下大多相近、商业路由亦难超越单一模型，并指出 Oracle 差距主要源于模型召回失败。

**🔧 技术方法**

采用统一的路由评估框架（Collector/Adaptor/Evaluator），利用多模型推理、嵌入式特征、成本与延迟统计，计算 Accuracy、Gain@Base、Gap@Oracle、PerfGain、CostSave、ParetoDist 等指标；并对 9 个公开路由算法与 OpenRouter 进行对比实验。

**📊 数据集**

使用的 21 个数据集涵盖数学、代码、逻辑、知识、情感、指令跟随、工具使用等 7 类任务；模型池分为 20 个 7B 轻量模型和 13 个旗舰模型；数据量约 1.8B token，实验成本约 1K GPU 时/2.8k 美元。

**📈 对比分析**

评测方法通过统一的路由器适配器与标准化评价器实现，比较指标包括性能提升（Gain）、Oracle 差距（Gap）、成本节省（CostSave）以及 Pareto 前沿距离；实验结果显示多数路由方法与单一最佳模型相当，部分路由在成本-性能平衡上略有优势，但整体仍落后于 Oracle，尤其在模型召回率方面存在显著缺口。

**⚠️ 局限性**

局限性包括：①评测未覆盖所有现有路由方法；②基准仅包含 21 个通用数据集，未覆盖垂直领域、长文本或多模态任务；③延迟估算基于 OpenRouter 统计，存在近似性，未给出精确端到端延迟。

---

## 611. CalPro: Prior-Aware Evidential--Conformal Prediction with Structure-Aware Guarantees for Protein Structures

**arXiv ID:** 2601.07201 | [PDF](https://arxiv.org/pdf/2601.07201v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 4685 | [OpenAlex ID](https://openalex.org/A5044574038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了 CalPro，一种先验感知的证据-共形框架，用于蛋白质结构预测中的不确定性量化，并证明其在分布偏移下的鲁棒性。

**💡 创新点**

创新点在于：①将图形证据头、可微共形层和软先验约束融合成一个端到端的训练管线；②提供结构感知的覆盖保证与 PAC‑Bayesian 偏移下的理论界限；③框架可推广至任何带有结构化先验的回归任务。

**🔧 技术方法**

采用的技术包括：Normal–Inverse‑Gamma 证据学习、图神经网络、差分共形校准、先验软正则化以及 PAC‑Bayesian 与分布不确定性分析。

**📊 数据集**

使用的数据集有：AlphaFold/Cryo‑EM/NMR PDB 等蛋白质结构数据库、FLIP 变异评估数据集、合成扰动数据以及非生物结构化回归基准。

**📈 对比分析**

与 pLDDT、温度缩放、传统共形、深度集成、MC‑dropout、CQR、IW‑CP 等基线进行比较，CalPro 在 90% 目标覆盖率下实现接近 nominal 的覆盖率、校准误差下降 30–50%，区间宽度更窄，并且在模态偏移下的覆盖率下降仅 4%（基线 15–25%）。

**⚠️ 局限性**

限制：仍依赖底层特征（如 AlphaFold 输出），共形层假设独立输出，需在参考分布上获取校准样本，先验若不可靠会误导；无法直接提供全结构的联合覆盖保证。

---

## 612. Beyond Variance: Knowledge-Aware LLM Compression via Fisher-Aligned Subspace Diagnostics

**arXiv ID:** 2601.07197 | [PDF](https://arxiv.org/pdf/2601.07197v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 4685 | [OpenAlex ID](https://openalex.org/A5044574038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型进行后训练激活压缩，提出基于Fisher信息的梯度对齐子空间压缩方法 FASC。

**💡 创新点**

创新点在于利用 Fisher 信息矩阵捕捉激活与梯度的耦合，定义 Dependence Violation Score ρ 作为诊断指标，形成梯度感知的压缩框架。

**🔧 技术方法**

采用二阶 Taylor 展开、Fisher 信息近似、随机化线性代数 sketching 及跨层梯度-激活协方差分析实现压缩。

**📊 数据集**

使用 C4、WikiText、Alpaca 进行校准；在 MMLU、LAMA、mLAMA、BLiMP、Natural Questions 等多种知识与语法评测集进行评估。

**📈 对比分析**

与标准 SVD、Grad-Weighted SVD、MagPrune、Fisher-Diag、LLM-Pruner 等基线比较，在 50% rank 压缩下，FASC 在知识任务上提升 6–8% 准确率，逼近更大模型的表现。

**⚠️ 局限性**

局限性包括相较 SVD 计算成本更高、需具备代表性校准数据、在极低资源或 MoE 模型环境下适用性需进一步验证。

---

## 613. Active Context Compression: Autonomous Memory Management in LLM Agents

**arXiv ID:** 2601.07190 | [PDF](https://arxiv.org/pdf/2601.07190v1)

**作者:** Nikhil Verma `[一作]` `[通讯]` (Independent Researcher), Nikhil Verma (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Focus 架构，使 LLM 代理能够主动压缩对话历史，并将重要学习成果保存到知识块中。

**💡 创新点**

创新点在于：1）Agent 自主决定压缩时机；2）将压缩过程内置于对话循环，避免外部总结工具；3）通过激进的提示策略实现高效压缩，保持任务准确率。

**🔧 技术方法**

使用的技术包括：基于 ReAct 的 Agent 循环改造，引入 start_focus/complete_focus 两个自定义工具；Claude Haiku 4.5 语言模型；持久 Bash 会话和字符串替换编辑器作为工具栈。

**📊 数据集**

评估数据集为 SWE‑bench Lite 的 5 个高难度实例，涉及复杂代码搜索和修复任务。

**📈 对比分析**

比较方法：将 Baseline（仅追加）与 Focus 进行 A/B 对比；结果显示 Focus 在 5 个实例上总 token 下降 22.7%（从 14.9M 降至 11.5M），准确率保持 3/5 = 60%；每任务平均压缩 6 次，平均 dropped 70 条消息。

**⚠️ 局限性**

局限性：1）样本量仅 5 例，缺乏大规模验证；2）压缩效果受任务特性影响，迭代细化任务时可能产生负面效果；3）当前压缩策略依赖人工提示，未实现模型自学；4）仅在 Claude Haiku 4.5 上测试，其他模型的表现未知。

---

## 614. PRPO: Aligning Process Reward with Outcome Reward in Policy Optimization

**arXiv ID:** 2601.07182 | [PDF](https://arxiv.org/pdf/2601.07182v1)

**作者:** Ruiyi Ding `[一作]` (Shanghai University), Yuan Cheng `[通讯]` (Fudan University)

**通讯引用:** 8134 | [OpenAlex ID](https://openalex.org/A5058272109)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种critic‑free强化学习算法PRPO，利用过程奖励模型（PRM）提供的密集过程级奖励与稀疏的最终奖励对齐，从而在多步推理任务中实现细粒度信用分配；

**💡 创新点**

创新点在于：①采用熵分段在不需要显式标记的情况下对生成序列进行语义级划分；②将每段的PRM得分标准化为token级优势；③通过位置参数偏移的分布对齐将过程优势与最终优势统一，使得即使在无价值网络的框架下也能获得稳定的token级学习信号；

**🔧 技术方法**

技术包括：critic‑free RL框架（GRPO/TEPO风格）、过程奖励模型Qwen2.5‑Math‑PRM‑7B、token熵分段、优势归一化、分布对齐、PPO/GRPO损失与KL/clip正则；

**📊 数据集**

使用了多数学推理数据集：MATH、AMC 2023、AIME 2024/2025，并在Qwen2.5‑Math‑1.5B和‑7B两个模型上进行实验；

**📈 对比分析**

与GRPO、PRM‑Avg、PURE等基线对比，PRPO在MATH上将pass@1从61.2%提升至64.4%（+3.2%），在AMC、AIME等数据集也实现显著提升，且保持与GRPO相当的训练效率；

**⚠️ 局限性**

局限性包括：对PRM质量高度依赖，固定均值/方差假设可能不适用于所有PRM；分段与对齐参数需手工设定，易受噪声影响；当过程奖励波动过大时仍可能导致崩溃；未将PRM与策略联合训练，缺乏自适应优化机制。

---

## 615. Structured Reasoning for Large Language Models

**arXiv ID:** 2601.07180 | [PDF](https://arxiv.org/pdf/2601.07180v1)

**作者:** Jinyi Han `[一作]` (East China Normal University), Yanghua Xiao `[通讯]` (Fudan University)

**通讯引用:** 3918 | [OpenAlex ID](https://openalex.org/A5090455375)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 StruCtured Reasoning 框架，将推理过程拆解为生成–验证–修订三步，并在监督微调与强化学习阶段分别训练每一步。

**💡 创新点**

创新点在于：① 用结构化标签显式化推理轨迹；② 引入动态终止监督（DTS）和选择性损失屏蔽（SLM）使模型按验证结果自行决定停止；③ 采用两阶段 GRPO 强化学习，分别针对初始生成与自我验证、以及修订给予细粒度奖励，降低各阶段学习信号干扰。

**🔧 技术方法**

技术方法包括：结构化监督微调（SFT）+动态终止监督、选择性损失屏蔽；两阶段基于 Group Relative Policy Optimization (GRPO) 的强化学习；格式奖励、初始/最终准确奖励、自我验证奖励、修订奖励等多元奖励设计。

**📊 数据集**

训练使用 MATH（难度3-5）和 DAPO-MATH；评测数据集涵盖 AIME24/25、MATH500、AMC、Olympiad、GPQA-Diamond、ARC、MMLU-Pro 等。

**📈 对比分析**

与 Base、On‑policy GRPO、Self‑Refine、SFT+GRPO 等基线对比，实验表明在三大模型上均实现最高平均性能；在自我验证上准确率、精确率显著提升，F1提升；输出 token 长度减少最高 50%；整体推理能力显著优于传统长链 CoT 及现有自我纠错方法。

**⚠️ 局限性**

局限性在于仅考虑两轮修订，扩展到更多轮需要更高成本的数据构造；多轮修订的效果与成本平衡尚待研究。

---

## 616. Safe-FedLLM: Delving into the Safety of Federated Large Language Models

**arXiv ID:** 2601.07177 | [PDF](https://arxiv.org/pdf/2601.07177v1)

**作者:** Mingxiang Tao `[一作]` (Hainan University), Xiangyan Tang `[通讯]` (Hainan University)

**通讯引用:** 865 | [OpenAlex ID](https://openalex.org/A5073355834)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对联邦学习中的大语言模型 (FedLLM) 进行安全评估，发现 LoRA 权重能区分恶意与正常更新，进而提出 Safe-FedLLM 框架，利用 LoRA 探测器和多层安全防御（Step、Client、Shadow）在 FL 过程中识别并抑制恶意客户端。

**💡 创新点**

①在 LoRA 权重空间发现恶意与正常更新可分离；②提出基于轻量级 LoRA 探测器的多层防御策略；③引入安全门控轮跳与早期冻结机制提升稳健性。

**🔧 技术方法**

LoRA 参数高效微调、逻辑回归探测器、贝叶斯权重因子、指数衰减、Shadow 分支、加权聚合、FedAvg 及 Krum、TrimmedMean、FoolsGold、Residual 等鲁棒聚合算法。

**📊 数据集**

使用 Llama3.1‑8B 与 Qwen2.5‑7B 两个大语言模型；数据来源包括 LMSYS‑Chat、WildChat、BeaverTails 与自动生成的 MaliciousGen；安全评估采用 Rule、MD‑Judge、RM 指标，效能评估采用 MT‑1。

**📈 对比分析**

与 FedAvg 及四种鲁棒聚合方法对比；在 30% 恶意比例下，Safe‑FedLLM 在 Rule 指标上提升至 90% 以上，MD‑Judge 与 RM 均显著改善，且对 MT‑1 的影响几乎为零；训练时间仅增加 3.2%。

**⚠️ 局限性**

需要统一 LoRA 初始化种子；探测器对不同 backbone 的迁移性差，需重新训练或适配；长时间训练导致 LoRA 分布漂移，Shadow 机制虽能缓解但会增加训练开销。

---

## 617. Test-time Adaptive Hierarchical Co-enhanced Denoising Network for Reliable Multimodal Classification

**arXiv ID:** 2601.07163 | [PDF](https://arxiv.org/pdf/2601.07163v1)

**作者:** Shu Shen `[一作]`, Tong Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究在低质量多模态数据下的可靠分类，提出 Test‑time Adaptive Hierarchical Co‑enhanced Denoising Network (TAHCD)。

**💡 创新点**

创新点：在全局与实例层面同时去除模态特定噪声与跨模态噪声，并设计无标签测试时协同增强机制 TTCE，实现对未知噪声的自适应更新。

**🔧 技术方法**

技术：Adaptive Stable Subspace Alignment (ASSA) 通过奇异值掩码构建稳健子空间；Sample‑Adaptive Confidence Alignment (SACA) 采用置信度加权的非对称 Slack 对齐；TTCE 通过多轮全局与实例去噪的协同更新和先验重估，提升对未见噪声的鲁棒性。

**📊 数据集**

数据集：BRCA、ROSMAP（基因表达、甲基化、miRNA）; CUB（图像+文本）；UPMC FOOD101（图像+文本）。

**📈 对比分析**

与 MD、MLCLNet、QMF、PDF、NCR、ALBEF、SMILE、SPS 等方法比较，TAHCD 在多模态噪声环境下在准确率、F1/宏F1、AUC 等指标上均取得更高成绩，尤其在训练数据无噪声、测试数据含噪声时表现更为显著。

**⚠️ 局限性**

局限性：目前仅验证分类任务，对回归或语义分割等其他任务的可靠性尚未探究；实现复杂度较高，TTCE 需要多次迭代，计算开销较大。

---

## 618. Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection

**arXiv ID:** 2601.07232 | [PDF](https://arxiv.org/pdf/2601.07232v1)

**作者:** Olivia Shanhong Liu `[一作]` (Singapore University of Technology and Design), Konstantinos N. Plataniotis `[通讯]` (University of Toronto)

**通讯引用:** 21464 | [OpenAlex ID](https://openalex.org/A5059152392)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个闭环反馈推理框架FLoReNce，利用冻结的视觉‑语言模型通过检索反馈驱动的提示实现模因幽默理解。

**💡 创新点**

将幽默推理建模为控制系统，采用判定者反馈转换为PID控制信号并存入非参数知识库，实现无参数训练下的自适应提示。

**🔧 技术方法**

使用Qwen2.5‑VL‑32B视觉‑语言模型、PID控制器、非参数检索式知识库、提示映射器以及裁决者与控制反馈机制。

**📊 数据集**

在PrideMM数据集（5,063条LGBTQ+主题模因）上进行实验。

**📈 对比分析**

与视觉/文本单模、传统多模融合以及提示/代理基线比较，FLoReNce在PrideMM上达到73.4–73.8%准确率、约77%宏F1，推理质量得分≈74%，相较主流基线提升约2–3个百分点。

**⚠️ 局限性**

局限在于仅依赖训练阶段的标注反馈，无法在完全无监督场景下学习；对讽刺与攻击性边界的判断仍易混淆，检索仅基于相似度，缺乏更细粒度的过滤策略。

---

## 619. DiSCo: Making Absence Visible in Intelligent Summarization Interfaces

**arXiv ID:** 2601.07229 | [PDF](https://arxiv.org/pdf/2601.07229v1)

**作者:** Eran Fainman `[一作]` (University of Haifa), Osnat Mokryn `[通讯]` (University of Haifa)

**通讯引用:** 816 | [OpenAlex ID](https://openalex.org/A5034395195)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于领域期望对比的缺失可视化总结方法DiSCo，利用LLM与LvS技术生成包含缺失信息的住宿评论摘要，帮助用户更全面地评估住宿；

**💡 创新点**

核心创新在于构建领域级期望分布并通过Surprisal度量个体住宿与该期望的偏差，从而识别既异常强调又缺失的方面，并将这些信息注入LLM生成的摘要；

**🔧 技术方法**

采用LLM（GPT‑5‑mini）进行方面‑情感提取与摘要生成，结合LvS（Surprise-based Divergence）对主题分布进行偏差分析，并通过结构化Prompt驱动LLM；

**📊 数据集**

使用Booking.com公开评论数据，按ski、beach、city‑center三类住宿域收集约270个住宿实例，提取138个主题层级；

**📈 对比分析**

通过对照实验（Baseline仅基于出现信息 vs. DiSCo强调缺失信息）进行用户研究，评估五维Likert指标与整体偏好；DiSCo在细节、实用性、决策支持上显著提升（p<.001），但易读性略下降；在ski域用户总体偏好显著倾向DiSCo；

**⚠️ 局限性**

仅评估主观感知，未验证对实际决策影响；摘要生成基于LLM，结果可重复性有限；缺失信息解释受文化、平台偏差影响；方法主要适用于结构化主题域，跨领域推广需调整期望模型。

---

## 620. VENUS: Visual Editing with Noise Inversion Using Scene Graphs

**arXiv ID:** 2601.07219 | [PDF](https://arxiv.org/pdf/2601.07219v1)

**作者:** Thanh-Nhan Vo `[一作]` (University of Science), Minh-Triet Tran `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了训练无关的场景图引导图像编辑框架VENUS，能够在不需要微调的情况下通过分解提示实现精准编辑并保持背景完整；

**💡 创新点**

创新点在于将编辑目标与背景分离的分割提示策略、利用噪声反演保证未编辑区域不变，并自动使用多模态LLM提取和编辑场景图，显著降低计算成本与推理延迟；

**🔧 技术方法**

核心技术包括基于LEDIT++的扩散模型、Stable Diffusion v2.1、Qwen‑VL‑2.5‑7B多模态LLM、噪声反演与无监督提示融合；

**📊 数据集**

评估使用PIE‑Bench、EditVal、COCO等编辑基准数据集；

**📈 对比分析**

与SGEdit、P2P+DirInv、LEDIT++等现有文本/场景图编辑方法对比，VENUS在PSNR提升至24.8、SSIM 0.837、LPIPS 0.07、CLIP 24.97，并将单图编辑时间从数分钟缩短至20‑30秒，整体性能优于基线；

**⚠️ 局限性**

局限性包括在EditVal等多尺度/不同分辨率场景下的准确率仍低于SGEdit，受Stable Diffusion v2.1对尺寸适应性的限制，以及对精细位置编辑的优化不足。

---

## 621. BlindU: Blind Machine Unlearning without Revealing Erasing Data

**arXiv ID:** 2601.07214 | [PDF](https://arxiv.org/pdf/2601.07214v1)

**作者:** Weiqi Wang `[一作]` (University of Technology Sydney), Shui Yu `[通讯]` (University of Technology Sydney)

**通讯引用:** 27080 | [OpenAlex ID](https://openalex.org/A5005228053)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 BlindU 方法，实现在不向服务器泄露用户删除数据的前提下，对基于信息瓶颈（IB）训练的联邦模型进行有效的机器去学习。

**💡 创新点**

创新点在于：① 双重隐私保护——通过 IB 压缩消除任务无关信息并在上传前进行无噪声 DP 采样掩码；② 仅使用压缩表示完成压缩器和近似器的去学习；③ 采用多梯度下降（MGDA）实现遗忘与保留的 Pareto 统一。

**🔧 技术方法**

主要技术包括信息瓶颈（IB）框架、互信息估计（MINE / Donsker‑Varadhan 表示）、无噪声 DP 采样掩码、以及 MGDA 优化策略。

**📊 数据集**

实验数据集涵盖 MNIST、CIFAR‑10、CIFAR‑100、TinyImageNet 以及 Adult 表格数据。

**📈 对比分析**

与 HBFU、BFU、VBU 等现有联邦与全局去学习方法以及噪声 DP 版本对比，BlindU 在保持模型准确率的同时，显著提升了重构 MSE 与成员推断 AUC，且耗时仅为 HBFU 的 1/5 左右。

**⚠️ 局限性**

局限性包括：依赖预先训练好的 IB 模型且压缩率固定；DP 采样对去学习效果有一定影响；对模型参数泄露的风险尚未完全解决；在大语言模型或无标签任务中的适用性仍需进一步验证。

---

## 622. Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG

**arXiv ID:** 2601.07192 | [PDF](https://arxiv.org/pdf/2601.07192v1)

**作者:** Manzong Huang `[一作]` (Hefei University of Technology), Xindong Wu `[通讯]` (Hefei University of Technology)

**通讯引用:** 42113 | [OpenAlex ID](https://openalex.org/A5080738591)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Relink框架，采用动态构建查询特定证据图的reason-and-construct范式。

**💡 创新点**

创新点在于同时利用高精度KG与高召回潜在关系池，联合查询驱动的评估来实时修复路径并过滤干扰事实。

**🔧 技术方法**

技术包括异构知识源整合、联合语义空间编码、粗细两级查询驱动排序、LLM动态实例化以及对比对齐与排名损失训练。

**📊 数据集**

实验数据集涵盖五个多跳问答基准：2WikiMultiHopQA、HotpotQA、ConcurrentQA、MuSiQue-Ans和MuSiQue-Full。

**📈 对比分析**

与GraphRAG及其它RAG基线比较，Relink平均提升5.4% EM、5.2% F1，在所有数据集上均取得显著优势。

**⚠️ 局限性**

局限性主要在于依赖LLM实例化生成的关系质量、对稀疏文本语境的鲁棒性以及潜在的计算开销。

---

## 623. ShowUI-Aloha: Human-Taught GUI Agent

**arXiv ID:** 2601.07181 | [PDF](https://arxiv.org/pdf/2601.07181v1)

**作者:** Yichun Zhang `[一作]` (Show Lab), Mike Zheng Shou `[通讯]` (Show Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出ShowUI-Aloha框架，将人类桌面演示录制成可执行的结构化任务轨迹，并通过规划与执行实现多步骤GUI自动化；

**💡 创新点**

创新点在于：①使用“记录–解析–学习”流水线将自然人类操作转化为语义化的自然语言轨迹；②将人类演示作为稀缺监督，构建“演示驱动+记忆规划”的闭环；③实现跨平台（Windows/macOS）可扩展的轻量级录制器、执行器和规划器；

**🔧 技术方法**

技术包括：FFmpeg/KeyCastOW录制，视觉-语言模型（VLM）进行截图标记与轨迹生成，LLM（如GPT‑4o）做规划，OpenAI API作为低级动作执行器；

**📊 数据集**

使用真实人类桌面演示视频（自行录制），并在OSWorld 361个任务上进行评估；

**📈 对比分析**

与多种无指导、零-shot或Agentic GUI基线（如UI‑TARS‑1.5‑7B、OpenAI CUA 4o、Claude 4 Sonnet、GTA‑1‑7B、Jedi‑7B、CoAct‑1）对比，ShowUI‑Aloha在整体任务成功率60.1%显著高于基线（最高≈57%）；在10类应用中，Chrome、OS、Thunderbird等类别成功率分别达到91.3%、83.3%、80%；

**⚠️ 局限性**

局限在于：①对图标细粒度辨别和文本拖拽选择的精度不足导致的定位错误；②仍需至少一条演示才能执行；③对跨应用协同任务仍挑战较大；

---

## 624. Offline Meta-Reinforcement Learning with Flow-Based Task Inference and Adaptive Correction of Feature Overgeneralization

**arXiv ID:** 2601.07164 | [PDF](https://arxiv.org/pdf/2601.07164v1)

**作者:** Min Wang `[一作]` (Beijing Institute of Technology), Hasnaa Bennis `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种面向离线元强化学习的框架 FLORA，通过分解 Q 函数为特征与权重、引入流式任务推断、以及基于不确定性反馈的自适应特征修正，以解决特征泛化导致的外域动作外推误差问题。

**💡 创新点**

创新点在于：1）将 Q 函数分解为环境动力学特征与奖励权重，揭示特征泛化与外推误差的因果关系；2）使用可逆流（Planar Flow）对任务分布进行灵活建模，提升任务表征的表达能力；3）通过双特征学习估计不确定性，并利用多臂老虎机策略结合返回反馈动态调整特征估计的保守程度，从而有效抑制特征过度泛化。

**🔧 技术方法**

核心技术包括：1）Successor Features（SFs）分解；2）双特征学习（double-ψ learning）与 Gaussian 估计；3）可逆流（Planar Flow）任务推断；4）多臂老虎机+返回反馈的自适应权重更新；5）基于 SAC 的离线数据收集和 BRAC 作为基线。

**📊 数据集**

使用 Meta-World（ML1 任务集，包含 8 个测试环境）和 MuJoCo（含 Point-Robot 与 Point-Robot-Wind 两种情景）进行实验，离线数据由 40 个训练任务（Meta-World）或 8 个训练任务（MuJoCo）产生的 50 条轨迹组成。

**📈 对比分析**

与 FOCAL、IDAQ、CSRO、UNICORN、Meta-DT、ER-TRL 等 6 种基线对比，FLORA 在所有 Meta-World 环境的最终成功率均超过对手，收敛速度最快，方差最小；在 MuJoCo 上也取得最优或接近最优的适应效率与终端性能；Ablation 研究表明，流式任务推断与自适应特征修正是提升性能的关键。

**⚠️ 局限性**

局限性包括：1）对离线数据质量高度敏感，极低质量或极宽任务分布时仍可能出现外推误差；2）模型参数多，训练成本相对较高；3）实验主要聚焦于离线数据下的 Meta-World 与 MuJoCo，未验证在更大规模或连续任务环境中的可推广性。

---

## 625. AscendKernelGen: A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units

**arXiv ID:** 2601.07160 | [PDF](https://arxiv.org/pdf/2601.07160v1)

**作者:** Xinzi Cao `[一作]` (Pengcheng Laboratory), Yonghong Tian `[通讯]` (Pengcheng Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个集成生成与评估的框架AscendKernelGen，用于自动化Ascend NPU核的代码生成与性能评估。

**💡 创新点**

创新点在于：①创建了基于链式推理的Ascend-CoT数据集；②使用领域自适应后训练（SFT+RL）得到KernelGen-LM；③提出了面向NPU的NPUKernelBench评测体系。

**🔧 技术方法**

使用技术包括大型语言模型（Qwen、Gemini、DeepSeek等）Fine‑Tuning、Reinforcement Learning（DPO）、Chain‑of‑Thought、自动化编译与执行反馈。

**📊 数据集**

数据集：Ascend-CoT（文档推理、代码推理与通用CoT混合），以及基准集NPUKernelBench包含158个从L1到L3的算子。

**📈 对比分析**

通过与通用LLM对比，基线模型在复杂L2/L3核上编译成功率从0%提升至95.5%，功能正确率提升至64.3%，并在部分L2核实现了1.86×的速度提升。

**⚠️ 局限性**

限制主要在：对极复杂的L3算子仍难以保证高正确率，缺乏对性能目标的显式奖励，且数据规模与多样性仍有限。

---

## 626. Engineering Decisions in MBSE: Insights for a Decision Capture Framework Development

**arXiv ID:** 2601.07301 | [PDF](https://arxiv.org/pdf/2601.07301v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 627. Rewarding Creativity: A Human-Aligned Generative Reward Model for Reinforcement Learning in Storytelling

**arXiv ID:** 2601.07149 | [PDF](https://arxiv.org/pdf/2601.07149v1)

**作者:** Zhaoyan Li `[一作]` (Alibaba Group), Liang Yu `[通讯]` (Alibaba Group)

**通讯引用:** 1524 | [OpenAlex ID](https://openalex.org/A5003039318)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RLCS 框架，使用强化学习提升创意故事生成，并设计了生成式奖励模型（GenRM）与熵基奖励塑形机制。

**💡 创新点**

创新点在于（1）GenRM 能提供多维度分析与显式推理的奖励，超越传统标量评分；（2）熵基奖励塑形动态聚焦高置信错误与不确定正确样本，显著提升训练稳定性与效率。

**🔧 技术方法**

使用技术包括：SFT + GRPO 训练 pipeline、CoT 推理链蒸馏、生成式奖励模型、熵基奖励塑形、Qwen 系列模型（7B/14B/32B）、Gemini-2.5-Pro 作为教师与对比基线。

**📊 数据集**

数据集：约 4,000 对故事由专业编剧标注，其中 1,400 对用于 SFT；其余通过多模型共识生成合成偏好数据；评测集 500 对人类标注的故事偏好。

**📈 对比分析**

与 SFT-Base、标准 RL（使用判别型 BT 奖励）以及 Gemini-2.5-Pro 等基线对比。GenRM 在专家评测上 68% 与人类一致；RLCS 在故事生成评测中分别以 72.4%、66.8% 以及 59.1% 的胜率领先基线，显示显著性能提升。

**⚠️ 局限性**

局限性包括：奖励模型受限于标注者的审美偏好；仅在故事生成任务验证，其他创作领域需进一步研究；熵基奖励塑形需任务特定调参；可能继承数据偏见，生成内容需人工审核。

---

## 628. Low-Altitude Satellite-AAV Collaborative Joint Mobile Edge Computing and Data Collection via Diffusion-based Deep Reinforcement Learning

**arXiv ID:** 2601.07307 | [PDF](https://arxiv.org/pdf/2601.07307v1)

**作者:** Boxiong Wang `[一作]` (Jilin University), Shiwen Mao `[通讯]` (Auburn University)

**通讯引用:** 23603 | [OpenAlex ID](https://openalex.org/A5080122431)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文提出一种基于卫星-无人机协同的联合移动边缘计算与数据采集系统，针对该系统设计了一套多目标优化框架，目标是最小化边缘计算端到端延迟、最大化采集数据量并降低无人机能耗，随后将三目标问题转化为马尔可夫决策过程并通过QAGOB算法实现在线决策。

**💡 创新点**

创新点在于：①构造了融合Gale‑Shapley设备关联与连续动作映射的混合动作空间；②首次将扩散模型（Diffusion）与Q‑加权变分策略优化（QVPO）结合，形成QAGOB算法，实现多模态策略生成与高效探索；③利用softmax和离散化映射解决可变维度与离散决策难题。

**🔧 技术方法**

核心技术包括：扩散概率模型（DDPM）+QVPO、马尔可夫决策过程建模、Gale‑Shapley匹配、softmax带宽分配、强化学习框架（SAC、PPO、TD3）以及PyTorch实现。

**📊 数据集**

实验使用自建仿真数据，任务长度、数据量采用指数/泊松分布，卫星/无人机参数和网络拓扑均在仿真中随机生成，无需公开数据集。

**📈 对比分析**

在随机、贪婪、PPO、SAC、DM‑TD3五种基准上进行对比，QAGOB在平均端到端延迟下降约11.48%、收集数据量提升约13.99%、无人机能耗降低约4.65%，同时收敛速度最快、总体奖励最高。

**⚠️ 局限性**

局限性：仅在仿真环境验证，缺乏真实部署案例；扩散模型训练与采样开销较大；需手工调节多参数，且对极端动态场景的鲁棒性尚待进一步验证。

---

## 629. Memory-Based Malware Detection under Limited Data Conditions: A Comparative Evaluation of TabPFN and Ensemble Models

**arXiv ID:** 2601.07305 | [PDF](https://arxiv.org/pdf/2601.07305v1)

**作者:** Valentin Leroy `[一作]` (CESI Engineering School), Sharif Ullah `[通讯]` (University of Central Arkansas)

**通讯引用:** 142 | [OpenAlex ID](https://openalex.org/A5071845094)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在有限数据条件下，评估 TabPFN 与 Random Forest、LightGBM、XGBoost 在多类别恶意软件家族分类中的表现

**💡 创新点**

首次将专为低样本设计的概率变压器 TabPFN 应用于内存特征的恶意软件多类识别，并在不同类别数（3、10、15）和训练规模下进行系统比较

**🔧 技术方法**

使用 TabPFN（无超参数调优的概率变压器）和传统树模型（Random Forest、LightGBM、XGBoost），通过 5 折交叉验证、Stratified Shuffle Split 评估准确率、精确率、召回率、F1 分数及训练/推理时间

**📊 数据集**

CIC‑MalMem‑2022 内存特征恶意软件数据集（58,596 条样本，去除正常样本后按 3/10/15 类划分），采样至 3000 条，进一步按 10%、20%、30%、50% 的训练比例进行实验

**📈 对比分析**

在低样本（≤250 条）和低类别（3 类）情境下，TabPFN 的准确率、F1 等指标较基线提升 2%–6%；随着样本和类别增多提升幅度减小，但仍保持领先；然而其训练和推理时间显著高于基线，尤其在 15 类场景下慢 30–60 秒

**⚠️ 局限性**

主要局限为计算开销较大，尤其多类别时训练/推理时间显著增加；实验仅针对内存特征，未覆盖其他恶意软件特征；数据仍相对有限且存在类别不平衡，需进一步改进数据处理和模型效率

---

## 630. Towards Multi-Behavior Multi-Task Recommendation via Behavior-informed Graph Embedding Learning

**arXiv ID:** 2601.07294 | [PDF](https://arxiv.org/pdf/2601.07294v1)

**作者:** Wenhao Lai `[一作]` (Shenzhen University), Zhong Ming `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的多行为多任务推荐模型BiGEL，结合级联图卷积、反馈门控、全局上下文增强和对比偏好对齐四个模块，提升多行为推荐的综合性能。

**💡 创新点**

在传统级联图结构上引入双向反馈（CGF）让目标行为信息回馈辅助行为；利用全局图提供全局偏好（GCE）；通过对比学习（CPA）校正级联过程中的偏好偏差，三者协同解决多行为多任务推荐的关键难点。

**🔧 技术方法**

基于LightGCN的级联图卷积；门控机制（Sigmoid+LeakyReLU）的反馈；全局GCN进行全局上下文增强；InfoNCE对比学习；BPR损失进行多任务优化。

**📊 数据集**

公开电商数据集JD（点击、收藏、购买）和UB（点击、收藏、加入购物车、购买）。

**📈 对比分析**

与十种单/多行为基线（MF‑BPR, LightGCN, GHCF, CRGCN, MB‑CGCN, BCIPM, AutoDCS, DA‑GCN, BVAE, POGCN）在Precision@5、Recall@5、NDCG@5、HR@5上进行对比，BiGEL在主要目标行为（购买）和辅助行为（点击、收藏、购物车）上均显著提升，尤其是购买行为的NDCG@5提升约1.5–2%。

**⚠️ 局限性**

模型在行为稀疏、全局与局部信息冲突时可能引入噪声；对行为权重设定统一，未探索任务权重自适应；缺乏对用户动态偏好序列的进一步建模。

---

## 631. Learning to Trust the Crowd: A Multi-Model Consensus Reasoning Engine for Large Language Models

**arXiv ID:** 2601.07245 | [PDF](https://arxiv.org/pdf/2601.07245v1)

**作者:** Pranav Kallem `[一作]` `[通讯]` (University of Texas), Pranav Kallem (University of Texas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多模型LLM的输出做结构化特征提取，训练一个监督式的meta‑learner（GBDT/RankNet/GAT）来预测每个实例的正确答案，实现多模型共识推理。

**💡 创新点**

创新点在于把LLM答案集合视为一个可学习的结构化输入，利用语义相似性、聚类、推理质量、置信度和模型先验等多维特征，并在图神经网络上进行信息传播，从而实现跨模型的协同决策，显著提升实例级可靠性。

**🔧 技术方法**

采用句子级SBERT/E5嵌入、聚类统计、词汇相似度、推理质量评分、置信度估计等特征；模型包括梯度提升树、列表排序网络、图注意力网络（GAT）以及传统的独立二分类器；训练过程中使用交叉熵、LambdaMART等损失。

**📊 数据集**

在四个公开基准的精简版上评估：GSM8K（数学推理）、ARC‑Challenge（科学QA）、HellaSwag（常识推理）和TruthfulQA（真值检验），每个数据集训练/验证/测试比例为800/200。

**📈 对比分析**

与随机、投票、最佳单模型以及自一致性等基线相比，最佳GAT共识模型在宏平均上提升至60.3%（比最佳单模型高4.6pp、比投票高8.1pp）；在GSM8K提升5.7pp，在TruthfulQA提升5pp，整体效果稳定且显著。

**⚠️ 局限性**

局限包括：需要有标签的数据进行监督，查询多模型导致计算成本和延迟上升，meta‑learner随基模型更新需重新训练，且在训练集外任务或系统偏差较强时可能失效或放大偏差。

---

## 632. Stochastic CHAOS: Why Deterministic Inference Kills, and Distributional Variability Is the Heartbeat of Artifical Cognition

**arXiv ID:** 2601.07239 | [PDF](https://arxiv.org/pdf/2601.07239v1)

**作者:** Tanmay Joshi `[一作]` (Pragya Lab, BITS Pilani Goa), Amitava Das `[通讯]` (Pragya Lab, BITS Pilani Goa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文质疑 LLM 推理中普遍采用的“位级确定性”做法，提出在推理中保留随机性（Stochastic CHAOS）能更好体现模型的不确定性、增强推理多路径、揭示潜在能力与安全风险。

**💡 创新点**

创新点在于将确定性视为可调度的设计选择而非默认目标，系统性拆分了三种稳定性维度（位级确定性、分布再现性、语义稳定性），并通过四类压力测试（指令遵循、突变评估、演绎推理、多样化安全评估）实验证明过度追求确定性会导致模型能力被低估、出现安全盲区。

**🔧 技术方法**

技术主要包括：① 传统与自定义的采样策略（温度、top‑k/top‑p、Self‑Consistency、Tree‑of‑Thought）；② 设计可批量不变的算子与数值可复现的推理内核；③ 多样本采样与扰动评估框架；④ 用统计量（成功率、方差、尾部风险）来描述分布性质。

**📊 数据集**

使用了广泛的公开评测集：GLUE、SST‑2、MNLI、ARC‑H、MMLU、GSM‑8K、OpenAI 指令遵循基准、Red‑Team 安全题库等；同时对同一模型在不同采样配置下的多样本输出进行聚合。

**📈 对比分析**

与单一贪婪推理（T=0）相比，多样本策略在指令遵循和推理任务上提升了约5–15% 的成功率，突变扰动下鲁棒性显著下降，而安全评估中风险估计也明显被低估。作者通过对比单样本与多样本的准确率、方差及安全违规率，展示了确定性推理对性能的误导性。

**⚠️ 局限性**

局限性包括：① 只评估了有限数量的模型与采样策略，无法完全覆盖所有 LLM 体系结构；② 主要关注文本任务，对多模态、工具调用等情境的适用性尚未验证；③ 评测集中在公开基准，可能存在训练集泄漏风险；④ 统计分析仍需更系统的置信区间与罕见事件估计方法。

---

## 633. Group Pattern Selection Optimization: Let LRMs Pick the Right Pattern for Reasoning

**arXiv ID:** 2601.07238 | [PDF](https://arxiv.org/pdf/2601.07238v1)

**作者:** Hanbin Wang `[一作]` (Peking University), Lifeng Shang `[通讯]` (Huawei Noah’s Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于强化学习的 Group Pattern Selection Optimization (GPSO) 框架，让大语言模型在推理时动态选择最优的思维模式。

**💡 创新点**

创新点在于：① 对多种推理模式同时进行 roll‑out 并通过可验证奖励进行评估；② 采用模式选择规则挑选最佳模式进行策略更新；③ 用注意力掩蔽防止模式提示泄露到策略中，使模型能内化模式选择决策。

**🔧 技术方法**

技术包括 RLVR（可验证奖励的强化学习）结合 GRPO 算法；多模式 roll‑out；模式选择与奖励归一化；注意力掩蔽与梯度裁剪；在训练时使用多种提示模板。

**📊 数据集**

训练数据使用 DAPO‑Math‑17K（约1.7万道竞赛级数学题）；评估数据包括 AIME2024、AIME2025、MATH‑500 与 GPQA 四个公开基准。

**📈 对比分析**

与多种现有大模型（DeepSeek‑R1‑Distill‑Qwen‑1.5B/7B、DeepScaleR‑1.5B‑Preview、Light‑R1‑7B‑DS、AReal‑boba‑RL‑7B、Qwen3‑8B）对比；GPSO 在各模型上平均提升 2.6%–3.2%（例如 Qwen‑1.5B 从 55.4% 提升至 58.0%），在强基线 Qwen3‑8B 上提升 0.8%；在 AIME2024/25 等难题上平均提升 4.0 / 2.7 分，体现了显著的性能提升。

**⚠️ 局限性**

主要局限：① 每个问题需要多模式 roll‑out，计算成本显著高于单一路径 RL；② 采用的模式集合是基于经验预设，缺乏自动发现或进化新模式的机制。

---

## 634. VLM-CAD: VLM-Optimized Collaborative Agent Design Workflow for Analog Circuit Sizing

**arXiv ID:** 2601.07315 | [PDF](https://arxiv.org/pdf/2601.07315v1)

**作者:** Guanyuan Pan `[一作]` (Hangzhou Dianzi University), Yaqi Wang `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 3870 | [OpenAlex ID](https://openalex.org/A5100352467)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于视觉语言模型的协同智能工作流 VLM-CAD，用于模拟电路的尺寸优化。

**💡 创新点**

创新点包括：① 利用 Image2Net 将电路原理图转化为结构化 JSON 供 VLM 理解；② 设计 Explainable Trust‑Region Bayesian Optimization (ExTuRBO)，通过协同热启动和双粒度敏感性分析实现可解释的数值优化；③ 将语言模型的推理能力与外部仿真器无缝结合，显著降低幻觉风险。

**🔧 技术方法**

使用的技术包括：YOLOv8-Pose+传统计算机视觉（角点检测、连通图构造）、Vision Language Model（Gemini‑3 Flash）、多代理 LLM 交互（Prompt Injection、链式思维）、自适应阈值的信赖区间贝叶斯优化（ExTuRBO）以及对外部 SPICE 仿真（Ngspice）的调用。

**📊 数据集**

数据集：三种 PTM 芯片（180 nm、90 nm、45 nm）下的两类放大器（互补输入+Class‑AB 输出、两级 Miller 运算放大器）的原理图和性能规范；实验中每个工艺跑 5 次。

**📈 对比分析**

对比方法包括：VLM-CAD、两种消融实验（仅原始原理图、无原理图）。实验结果显示：对互补输入放大器，VLM-CAD 100% 成功率、功耗≤10 mW、总时长<9 min；对 Miller 放大器，VLM-CAD 在功耗上稍超出目标但满足其他指标，时长<43 min；消融实验耗时显著更长，说明原理图信息及 Image2Net 对性能提升有重要作用。

**⚠️ 局限性**

局限性：① 对极端约束的设计（如 45 nm Miller 放大器）仍难以满足所有指标；② VLM 对原理图的理解受限，若原理图复杂或有视觉干扰，热启动种子质量下降；③ 现有工作主要验证两类放大器，尚缺乏更广泛的电路种类与工艺验证。

---

## 635. From Landslide Conditioning Factors to Satellite Embeddings: Evaluating the Utilisation of Google AlphaEarth for Landslide Susceptibility Mapping using Deep Learning

**arXiv ID:** 2601.07268 | [PDF](https://arxiv.org/pdf/2601.07268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 636. ARM: Role-Conditioned Neuron Transplantation for Training-Free Generalist LLM Agent Merging

**arXiv ID:** 2601.07309 | [PDF](https://arxiv.org/pdf/2601.07309v1)

**作者:** Zhuoka Feng `[一作]` (Fudan University), Yugang Jiang `[通讯]` (Fudan University)

**通讯引用:** 23651 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的 Agent-Role Merging (ARM) 方法，将多个专用 LLM 代理融合为单一通用模型。

**💡 创新点**

创新点在于：①使用角色条件激活追踪构造 Activation‑Overlap Score 以选取稳健的合并初始化；②在多轮交互中采用冲突感知的神经元移植，精准修复关键行为而不破坏其他环境能力。

**🔧 技术方法**

技术包括：权重空间合并、激活覆盖评分、角色条件激活映射、冲突感知神经元移植。

**📊 数据集**

使用 Qwen3‑8B / Qwen2.5‑7B 专家模型，并在 τ‑bench、OfficeBench、WebShop、Operating System、DB‑bench、AlfWorld 等六大交互代理基准上评估。

**📈 对比分析**

与均匀平均、任务算术、TIES、TIES+DARE、WIDEN、AIM、NeuronMerge 等训练无关基线及 BEST‑of‑Three oracle 对比，ARM 在大多数基准上实现最高平均分，并显著提升最差套装的鲁棒性。

**⚠️ 局限性**

局限性：仅适用于同一架构、同一分词器的专家，无法直接融合异构模型或黑盒 API；依赖激活级诊断，若激活解释不充分则可能影响效果。

---

## 637. Heterogeneous Multi-Expert Reinforcement Learning for Long-Horizon Multi-Goal Tasks in Autonomous Forklifts

**arXiv ID:** 2601.07304 | [PDF](https://arxiv.org/pdf/2601.07304v1)

**作者:** Yun Chen `[一作]` (State Key Laboratory of Engines Tianjin University), Kang Song `[通讯]` (State Key Laboratory of Engines Tianjin University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种 Heterogeneous Multi-Expert Reinforcement Learning（HMER）框架，用分层专家（导航、拾取、放置）和语义任务规划器，实现自主叉车在仓库中长周期多目标任务的闭环控制。

**💡 创新点**

创新点包括：① 通过结构解耦将宏观导航与微观操作分为异构专家，消除梯度干扰；② 语义任务规划器提供闭环状态监控与错误恢复；③ 混合模仿-强化训练策略，先行为克隆再PPO细化；④ 递归奖励提升放置精度。

**🔧 技术方法**

采用层次强化学习、行为克隆、PPO、LiDAR+RGB感知（1D‑CNN、2D‑CNN/ResNet）、Gazebo+ODE仿真、Domain Randomization、离散状态机和递归奖励机制。

**📊 数据集**

使用10,000条专家演示数据以及在Gazebo仿真中生成的自监督数据集，未使用公开真实数据集。

**📈 对比分析**

与 Flat‑BC、Rule‑Based、HBC、HRL、Seq‑Hybrid 等基线比较，HMER 取得94.2%任务成功率、42.5s平均周期、1.5cm放置误差、2.1%碰撞率，样本效率提升43.7%，显著优于基线。

**⚠️ 局限性**

局限性：仅在仿真验证，缺乏真实世界实验；对传感器延迟、地面摩擦等现实噪声未完全考虑；高层规划采用离散逻辑，缺乏自学习的连续恢复策略。

---

## 638. SwarmFoam: An OpenFOAM Multi-Agent System Based on Multiple Types of Large Language Models

**arXiv ID:** 2601.07252 | [PDF](https://arxiv.org/pdf/2601.07252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 639. GenDet: Painting Colored Bounding Boxes on Images via Diffusion Model for Object Detection

**arXiv ID:** 2601.07273 | [PDF](https://arxiv.org/pdf/2601.07273v1)

**作者:** Chen Min `[一作]`, Liang Xiao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 GenDet 框架，将目标检测重新定义为在输入图像上直接生成带颜色的边框，完成检测与生成任务的统一实现。

**💡 创新点**

创新点在于把检测任务转化为条件图像生成问题，利用预训练的 Stable Diffusion 在潜在空间中直接生成带类别信息的检测框，并通过双路径条件注入与梯度损失提升生成质量。

**🔧 技术方法**

使用技术包括 Stable Diffusion（Latent Diffusion + VAE + U‑Net）、DDIM 采样、双路径条件注入、梯度损失、多尺度训练目标以及特征/学习式后处理（VGG‑16 特征差分 + DBSCAN / set‑based 检测头）。

**📊 数据集**

使用的数据集为 COCO 2017（118K 训练图，5K 验证图）和 CrowdHuman（15K+ 训练图，4.3K 验证图）。

**📈 对比分析**

与 Faster R‑CNN、DETR、DiffusionDet 等主流检测器比较，Set‑based GenDet（学习式后处理）在 COCO 上 AP ≈ 46.4、AP_50 ≈ 64.2 与 DiffusionDet 接近（AP ≈ 45.8、AP_50 ≈ 64.5），Feature‑based GenDet 性能较低但仍具备一定竞争力；在 CrowdHuman 上表现尚未达到最佳水平。

**⚠️ 局限性**

局限性包括：推理速度慢（单张图需数十秒）、对拥挤/遮挡/小目标识别能力不足、检测结果存在随机性、颜色空间限制导致类别可识别范围受限。

---

## 640. The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents

**arXiv ID:** 2601.07264 | [PDF](https://arxiv.org/pdf/2601.07264v1)

**作者:** Weihao Xuan `[一作]` (University of Tokyo), Naoto Yokoya `[通讯]` (University of Tokyo)

**通讯引用:** 13952 | [OpenAlex ID](https://openalex.org/A5034435383)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究工具使用代理的校准动态，并提出 Calibration Agentic RL（CAR）框架以提升代理自我校准能力。

**💡 创新点**

创新点在于揭示证据工具导致过度自信的“信心二分法”，并提出 Margin‑Separated Calibration Reward（MSCR）显式分离正确与错误奖励，显著提升校准精度。

**🔧 技术方法**

使用强化学习微调（GRPO、RLCR、MSCR）、工具集成（Web 搜索、代码解释器）、多维度校准评估（ECE、Brier、AUROC）等技术。

**📊 数据集**

采用 NQ、HotpotQA、SimpleQA‑verified、AIME2024/2025、MATH‑500、Serper API、SimpleTIR 等数据集进行评估。

**📈 对比分析**

与 Vanilla Search‑R1、Temperature Scaling、MASH 等基线对比，CAR 在 ECE 下降 50–70%，准确率保持或提升，AUROC 提升 10–20%，并能跨域从本地检索迁移至噪声 API 与数学推理场景。

**⚠️ 局限性**

限制在于仅测试 3B–7B 规模模型，未探讨更大模型的表现；仅评估短答题和数理推理，未覆盖开放式生成或长期规划等更复杂场景。

---

## 641. Simulated Annealing-based Candidate Optimization for Batch Acquisition Functions

**arXiv ID:** 2601.07258 | [PDF](https://arxiv.org/pdf/2601.07258v1)

**作者:** Sk Md Ahnaf Akif Alvi `[一作]` (Texas A&M University), Douglas Allaire `[通讯]` (Texas A&M University)

**通讯引用:** 1780 | [OpenAlex ID](https://openalex.org/A5015668401)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在离散候选集上使用模拟退火进行批量贝叶斯优化候选点选择的方法，并与传统梯度优化方法进行了对比。

**💡 创新点**

提出了直接在离散候选集上操作、兼顾全局搜索并支持大批量尺寸的模拟退火框架，突破了梯度方法的局部收敛与离散约束限制。

**🔧 技术方法**

采用高斯过程代理、qEHVI获取函数以及串行与GPU并行的模拟退火技术。

**📊 数据集**

在四个经典多目标基准（ZDT1、DTLZ2、Kursawe、Latent‑Aware）以及一组五目标材料设计实验数据上进行评估。

**📈 对比分析**

与SLSQP等连续优化方法比较，模拟退火在高维或非凸问题上实现了约2%–20%甚至更高的超体积提升，GPU并行版收敛更快、效率更高。

**⚠️ 局限性**

受限于退火冷却速率与迭代次数，仍存在探索深度不足以及对极大批量规模的计算成本未完全解决。

---

## 642. DarwinTOD: LLM Driven Lifelong Self Evolution for Task Oriented Dialog Systems

**arXiv ID:** 2601.07248 | [PDF](https://arxiv.org/pdf/2601.07248v1)

**作者:** Shuyu Zhang `[一作]` (Shanghai Jiao Tong University), Bin Li `[通讯]` (Shenzhen Institute of Advanced Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DarwinTOD，一种终身自我进化的任务导向对话框架，采用可进化策略库和双循环（在线对话+离线进化）实现从零开始的自适应优化。

**💡 创新点**

创新点：① 将进化算法与 LLM 驱动的自我提升统一为人口级别策略演化；② 通过可进化策略库维护多样化策略集合；③ 采用双循环架构实现在线多智能体协作与离线结构化进化，零人类干预即可持续改进。

**🔧 技术方法**

技术手段：POMDP建模；可进化策略库（ESB）与 Boltzmann 选择；多智能体 LLM（DST、DP、NLG、UserSim）协作与同行评审；进化算子（变异、合并、裁剪、生成）；熵与适应度度量；基于 LLM 的策略生成与评估。

**📊 数据集**

数据集：MultiWOZ 2.0、2.1、2.2 以及 SGD 数据集，用于训练与评估。

**📈 对比分析**

对比方法：基线包括 pipeline、end‑to‑end、持续学习模型及 AgentTOD 等；在 MultiWOZ 各版本上，DarwinTOD 超越所有现有 SOTA，Combine 分数在 117.34（Qwen3-8B）到 120.59（GPT‑5.1）之间，显示持续改进的演化曲线。

**⚠️ 局限性**

局限性：目前主要基于模拟交互，缺乏对真实工具调用与外部系统的完整支持；对真实环境的鲁棒性和功能调用能力待进一步提升；大规模部署时对算力与延迟的需求仍需优化。

---

## 643. HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization

**arXiv ID:** 2601.07242 | [PDF](https://arxiv.org/pdf/2601.07242v1)

**作者:** Taekbeom Lee `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**通讯引用:** 6984 | [OpenAlex ID](https://openalex.org/A5073996122)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了基于NeRF的主动三维场景重建框架HERE，利用知识不确定性驱动相机轨迹规划。

**💡 创新点**

创新点在于将证据深度学习(EDL)应用于神经隐式映射以量化认识不确定性，并结合分层全局与局部规划实现高效全景覆盖。

**🔧 技术方法**

技术包括NeRF/Co‑SLAM的隐式网格表示、EDL+NIG先验的认识不确定性估计、基于可见性与信息增益的贪婪视角选择、TSP与A* 的覆盖规划以及离线/实时推理。

**📊 数据集**

使用MP3D和Gibson室内数据集进行仿真评估，并在Turtlebot3上进行真实世界实验。

**📈 对比分析**

与基线的前沿探索、其他INR主动重建方法以及3DGS方法对比，取得更低的AUSE、更高的重建完整度(%)和更好的网格细节，帧率约9.2FPS。

**⚠️ 局限性**

局限性包括依赖先验位姿、对极端光照或动态物体的鲁棒性不足、以及在极大场景下全局规划仍可能产生冗余路径。

---

## 644. Inference-Time Scaling for Visual AutoRegressive modeling by Searching Representative Samples

**arXiv ID:** 2601.07293 | [PDF](https://arxiv.org/pdf/2601.07293v1)

**作者:** Weidong Tang `[一作]` (Xidian University), Xiumei Wang `[通讯]` (Xidian University)

**通讯引用:** 3333 | [OpenAlex ID](https://openalex.org/A5100765030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了VAR-Scaling，一种针对视觉自回归模型(VAR)的推理时间缩放框架，通过核密度估计(KDE)将离散潜空间映射为近似连续特征空间，并采用密度自适应的混合采样策略（Top‑k+Random‑k）提升采样质量；

**💡 创新点**

创新点包括：①首次在VAR中实现推理时间缩放；②利用KDE将离散潜空间转化为连续空间；③设计基于样本密度的自适应混合采样，自动区分高密度代表性样本与低密度噪声样本；

**🔧 技术方法**

技术手段包括VAR模型、KDE映射、Top‑k与Random‑k采样、密度阈值α判定、Euclidean距离度量、Classifier‑Free Guidance（CFG）以及FID、IS、GenEval等评估指标；

**📊 数据集**

实验数据集主要包括ImageNet 256×256（类条件生成）和Infinity文本到图像数据集，以及ImageNet‑50k用于评估；

**📈 对比分析**

与原版VAR、FlexVAR及Infinity模型对比，VAR‑Scaling在类条件下将IS提升8.7%/6.3%，在文本条件下GenEval提升1.1%，同时保持FID不变，显示显著性能提升；

**⚠️ 局限性**

局限性在于对高密度样本与高质量样本对应关系的理论尚未充分阐明，且未将方法扩展到视频生成等更复杂任务，且在不同模型/参数组合上仍需进一步调优；

---

## 645. A Visual Semantic Adaptive Watermark grounded by Prefix-Tuning for Large Vision-Language Model

**arXiv ID:** 2601.07291 | [PDF](https://arxiv.org/pdf/2601.07291v1)

**作者:** Qi Zheng `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1186 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种视觉语义自适应水印框架 VISA-Mark，能够在保持视觉一致性的同时向大型视听语言模型嵌入可检测且鲁棒的水印。

**💡 创新点**

创新点在于引入前缀调优的视觉证据提取器、基于模型不确定性的词表动态分区以及证据校准的对数几率偏移，三者协同实现视觉信息对齐与水印可检验性的平衡。

**🔧 技术方法**

使用技术包括前缀调优（prefix‑tuning）、对比解码提取视觉权重、熵调节的词表分区策略以及可调节的对数几率偏置。

**📊 数据集**

在公开的 MS‑COCO 14/17、AMBER 以及 LLaVA‑v1.5、Qwen3‑VL 等大规模视听语言模型上进行实验，利用 Dense Image‑Caption 数据集训练前缀。

**📈 对比分析**

与 KGW、SWEET、Unbiased、DiP 等传统水印方法相比，VISA‑Mark 在视觉一致性（Chair‑I）和文本质量（PPL、BERTScore）上均优于对手，且检测 AUC 达到 97.95% 以上，鲁棒性对文本攻击保持 99% 以上。

**⚠️ 局限性**

局限性包括对训练集外域（如医学影像、抽象艺术）的泛化不佳、对针对视觉证据提取器的自适应攻击缺乏充分评估，以及目前主要聚焦对象级别证据，尚未覆盖属性或关系细粒度一致性。

---

## 646. Coalition Tactics: Bribery and Control in Parliamentary Elections

**arXiv ID:** 2601.07279 | [PDF](https://arxiv.org/pdf/2601.07279v1)

**作者:** Hodaya Barr `[一作]` (Bar-Ilan University), Sarit Kraus `[通讯]` (Bar-Ilan University)

**通讯引用:** 18347 | [OpenAlex ID](https://openalex.org/A5103213461)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了在比例代表制议会选举中，以提升一个既定政党联盟（coalition）整体席位比例为目标的选举操纵（bribery 与 control）模型，并给出了对应的算法与复杂度分析。

**💡 创新点**

创新点在于首次将操纵目标从单一候选人扩展到政党联盟，并系统划分了四种操纵方式（bribery、control-adding、control-deleting、不同阈值与目标组合），揭示了阈值对复杂度的决定性影响。

**🔧 技术方法**

采用的技术主要是算法设计与多元化的复杂度证明，包括多目标动态规划、最小成本流（MCF）转化、以及从经典 NP‑hard/parameterized‑hard 问题（如 3‑4‑Exact‑Cover、Clique、Dominating Set）构造的归约。

**📊 数据集**

本文未使用实测数据集，而是基于抽象的理论模型与人工构造的投票实例进行实验演示。

**📈 对比分析**

方法比较主要通过理论证明完成，证明了多种情形下问题是多项式可解的、NP‑hard 或 W[1]/W[2]‑hard；在可解情形下给出多项式时间算法，在硬件情形下给出对应的最优或近似实现。

**⚠️ 局限性**

主要局限包括：仅考虑单一联盟与单一首选党，未对多元化联盟组合或多操纵者场景进行建模；模型忽略了实际选举的分区制、选区席位分配细节；阈值的静态设定与投票行为的简化限制了结果对真实选举的直接可迁移性。

---

## 647. PALUM: Part-based Attention Learning for Unified Motion Retargeting

**arXiv ID:** 2601.07272 | [PDF](https://arxiv.org/pdf/2601.07272v1)

**作者:** Siqi Liu `[一作]` (Shanghai Jiao Tong University), Cewu Lu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15329 | [OpenAlex ID](https://openalex.org/A5010726528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个骨架无关的运动重定向框架PALUM，能够在骨骼结构差异较大的角色之间迁移运动。

**💡 创新点**

创新点在于基于语义体部的注意力聚合与时空编码，配合循环一致性约束，使模型学习到骨架不变的运动表示。

**🔧 技术方法**

使用Transformer结构、注意力聚合、T5词嵌入、T姿态嵌入以及循环一致性损失。

**📊 数据集**

在Mixamo数据集上进行训练与评估，使用12个角色训练，7个角色测试，并去除手指关节。

**📈 对比分析**

与R^2ET、PAN、MoMa三种先进方法对比，PALUM在四个评估场景下均取得最低MSE，尤其在跨骨架情形下显著优于基线。

**⚠️ 局限性**

局限在于仅利用关节名称语义来对齐骨骼，导致不同链长时的姿态预测不精确，且目前仅适用于人形骨架。

---

## 648. Document-Level Zero-Shot Relation Extraction with Entity Side Information

**arXiv ID:** 2601.07271 | [PDF](https://arxiv.org/pdf/2601.07271v1)

**作者:** Mohan Raj Chanthran `[一作]` (Monash University), Bhawani Selvaretnam `[通讯]` (Valiantlytix)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DocZSRE‑SI框架，用实体侧信息（实体描述、超义词、实体类型）实现文档级零样本关系抽取，避免依赖LLM生成合成数据。

**💡 创新点**

创新点在于首次将实体侧信息与动态加权评分机制结合，用以提升低资源语言的零样本关系抽取效果。

**🔧 技术方法**

技术上采用预训练BERT生成实体与关系嵌入，计算余弦相似度并通过动态加权得分判断最优未见关系标签。

**📊 数据集**

实验使用DocRED、RE‑DocRED以及马来西亚英语新闻数据集MEN-Dataset进行评估。

**📈 对比分析**

相较于仅用实体描述的基线，和现有的Doc‑level ZSRE方法GenRDK相比，宏F1平均提升约40%（n=10）或整体提升约11.6%，显著优于对标方法。

**⚠️ 局限性**

局限性包括对语义相近关系的高方差、依赖实体类型可能产生歧义以及计算量大等问题。

---

## 649. When Bots Take the Bait: Exposing and Mitigating the Emerging Social Engineering Attack in Web Automation Agent

**arXiv ID:** 2601.07263 | [PDF](https://arxiv.org/pdf/2601.07263v1)

**作者:** Xinyi Wu `[一作]` (Fudan University), Baojun Liu `[通讯]` (Tsinghua University)

**通讯引用:** 719 | [OpenAlex ID](https://openalex.org/A5101694986)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并系统评估基于LLM的Web自动化代理面临的社会工程攻击，并提出轻量级运行时防御模块

**💡 创新点**

首次将诱因与目标定义为四元组，提出环境与意图一致性检查作为通用防御手段，并实现可插拔模块

**🔧 技术方法**

结合LLM推理、DOM语义抽取、规则+LLM判断的环境一致性检查、权限/敏感度策略的意图一致性检查，并通过函数/进程挂钩集成至各框架

**📊 数据集**

构造100个攻击四元组基准（20种诱因×4种目标×5种一致性模式）及基于WebVoyager的真实网站实验，使用公开的5大开源Web代理框架

**📈 对比分析**

与四种轻量级防御（Task‑Specific、Safety‑Prompt、AGrail、ATHENA）对比；在5框架上攻击成功率平均降低78.1%，运行时开销仅7.7%，任务完成率仅下降2.7%

**⚠️ 局限性**

局限于开源框架、攻击策略覆盖不全、LLM推理不确定导致误判、未对商业代理进行评估

---

## 650. BEAT-Net: Injecting Biomimetic Spatio-Temporal Priors for Interpretable ECG Classification

**arXiv ID:** 2601.07316 | [PDF](https://arxiv.org/pdf/2601.07316v1)

**作者:** Runze Ma `[一作]` (Monash University), Caizhi Liao `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 BEAT‑Net，一种将心电图分析视为语言建模任务的生物仿真框架，利用 QRS 关键点标记化将连续信号拆分为心搏级别的 token，并通过形态、空间、时间四个专门编码器提取特征，最终使用 Transformer 进行全局推理，完成多标签心电诊断。

**💡 创新点**

创新点：
① 在有监督学习中首次引入 QRS‑中心化 token 化，显著提升数据效率；
② 通过形态、空间、时间层级的解耦设计，模仿临床诊断流程，形成可解释的医学知识表征；
③ 在仅使用 30–35% 标注数据时即可恢复全标注性能，且注意力机制能自动复现 Lead II、V1/V5/V6 等临床先验，兼具准确性与可解释性。

**🔧 技术方法**

技术细节：QRS 标记化、深度残差网络做词编码器、lead‑specific affine 变换实现空间归一化、时间嵌入、Transformer 句子编码器；实现框架为 PyTorch，使用 AdamW 进行训练。

**📊 数据集**

数据集：PTB‑XL（21,837 条 10 s 记录）、CPSC2018（6,877 条变长记录）以及 CSN（约 45,000 条 500 Hz 记录）三大 12‑导联心电基准。

**📈 对比分析**

与 xresnet1d101、resnet1d_wang、inception1d 等三种主流 1D‑CNN 基线对比，BEAT‑Net 在所有任务上与 CNN 相当或略优；在低资源情景下，仅使用 35% 标注数据即可达到或超过完整监督模型的 AUC；注意力分布与临床经验高度一致。

**⚠️ 局限性**

局限性：
① 仍需在更大、更多中心、不同采样率的数据上进一步验证；
② 模型对 R‑峰检测的准确性较为敏感；
③ 当前仅针对 12‑导联心电，未扩展至单导联或多模态场景；
④ 推理时间与参数量相对传统 CNN 略高。

---

## 651. Mitrasamgraha: A Comprehensive Classical Sanskrit Machine Translation Dataset

**arXiv ID:** 2601.07314 | [PDF](https://arxiv.org/pdf/2601.07314v1)

**作者:** Sebastian Nehrdich `[一作]` (Center for Integrated Japanese Studies), Kurt Keutzer `[通讯]` (University of California)

**通讯引用:** 37569 | [OpenAlex ID](https://openalex.org/A5047285420)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建并公开了规模最大的梵语-英语机器翻译语料库Mitrasaṃgraha，包含391,548句对，并提供约5,500句的手工校正验证集和测试集。

**💡 创新点**

创新点在于：①跨越三千年、多领域（宗教、史诗、哲学、诗歌等）梵语文本的系统收集与对齐；②对比多种对齐算法，最终选用BertAlign+LaBSE实现高质量对齐；③评估多种MT评估指标，发现BLEURT与GEMBA与人工判断相关性最高；④对商业LLM和开源模型进行基准评测，并展示检索增强和全参数微调显著提升性能。

**🔧 技术方法**

技术包括：网页抓取+OCR+规则+手工清洗；句子分割与BertAlign对齐；BLEU/chrF、BLEURT、GEMBA评估；对开源模型（NLLB、MADLAD、Gemma等）的全参数微调；检索增强生成（RAG）与语法注释增强。

**📊 数据集**

使用的数据集为Mitrasaṃgraha（391,548句对，涵盖梵语四个时期和六大领域），以及先前公开的Itihāsa、Mahābhārata/Rāmāyaṇa等数据集做对比。

**📈 对比分析**

比较方法：在测试集上计算chrF、BLEURT和GEMBA得分。商业LLM在检索增强后可提升约10%点；开源模型在全参数微调后可逼近甚至超过部分商业系统，Gemma‑2 9B微调后GEMBA达83.34分。

**⚠️ 局限性**

局限性包括：句子对齐仍存在噪声和不一致；OCR错误未完全消除；对深层语义（多层隐喻、哲学概念）仍表现不佳；部分技术文献和专业领域文本缺失。

---

## 652. PsyCLIENT: Client Simulation via Conversational Trajectory Modeling for Trainee Practice and Model Evaluation in Mental Health Counseling

**arXiv ID:** 2601.07312 | [PDF](https://arxiv.org/pdf/2601.07312v1)

**作者:** Huachuan Qiu `[一作]` (Westlake University), Zhenzhong Lan `[通讯]` (Westlake University)

**通讯引用:** 7807 | [OpenAlex ID](https://openalex.org/A5103239171)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PsyCLIENT 框架，用对话轨迹建模实现高真实性的中文 AI 咨询客户模拟，并公开了 PsyCLIENT‑CP 客户画像数据集。

**💡 创新点**

创新点在于同时利用行为标签和内容约束的对话轨迹，解决了现有模拟过度顺从、缺乏多样性和仅限英语的问题；同时首次在中文情境下实现多样化、可复用的客户画像与轨迹组合。

**🔧 技术方法**

技术核心是基于大语言模型的客户端生成（通过精心设计的 Prompt 控制），配合对话轨迹提取与行为标签映射的规则框架；评估采用专业咨询师的人工打分和二分类辨别任务。

**📊 数据集**

使用 60 个咨询主题的 120 条人工对话（30+ 轮）构成的 PsyCLIENT‑CP 画像数据集，以及 324 条真实对话的轨迹数据；对照 4 种基线（vanilla、+behavior、+content、PsyCLIENT）。

**📈 对比分析**

通过 7‑分量表对真实性与有效性进行专家评估，PsyCLIENT 在流畅性、情感表达、连贯性、适当性和整体真实性上均显著优于基线；在辨别任务中专家误判率约 95%，LLM 辨别准确率随真实性提升而下降，证明模拟高度逼真。

**⚠️ 局限性**

局限性包括仅模拟单次会话，缺乏多轮/多会话长期跟踪；数据集和轨迹均基于中国文化背景，跨文化通用性尚待验证。

---

## 653. Revisiting the Ordering of Channel and Spatial Attention: A Comprehensive Study on Sequential and Parallel Designs

**arXiv ID:** 2601.07310 | [PDF](https://arxiv.org/pdf/2601.07310v1)

**作者:** Zhongming Liu `[一作]` (Jiangxi Normal University), Bingbing Jiang `[通讯]` (Jiangxi Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在单一网络框架下系统评估18种通道-空间注意力组合，揭示不同数据规模下的最优设计规律。

**💡 创新点**

提出“数据规模–结构–性能”耦合法则，证明空间优先顺序在细粒度分类中的稳定优势，并给出可直接实施的场景化设计准则。

**🔧 技术方法**

结合通道注意、空间注意、门控、残差、并行与多尺度等技术，构建十余种新型融合模块。

**📊 数据集**

在CIFAR‑10/CIFAR‑100和MedMNIST（共9个医学子数据集）上进行实验验证。

**📈 对比分析**

通过与基线、现有CBAM等模块对比，实验显示在小样本时使用C‑CMSSA可提升约3%，中等样本采用C&SAFA提升≈10%，大样本使用GC&SA^2可达99.57%精度，整体提升0.5–2%。

**⚠️ 局限性**

局限性包括：最佳结构仍依赖数据规模与任务，未在多种主干网络上验证；对极端小样本或非图像任务的适用性未知；缺乏理论误差界定。

---

## 654. Bringing Computation to the data: Interoperable serverless function execution for astrophysical data analysis in the SRCNet

**arXiv ID:** 2601.07308 | [PDF](https://arxiv.org/pdf/2601.07308v1)

**作者:** Manuel Parra-Royón `[一作]` (Instituto de Astrofísica de Andalucía), Lourdes Verdes-Montenegro `[通讯]` (Instituto de Astrofísica de Andalucía)

**通讯引用:** 3625 | [OpenAlex ID](https://openalex.org/A5041257950)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 SRCNet 体系内实现了一种基于 FaaS 的函数化架构，并以高斯卷积函数为例完成了完整的开发、容器化、CI/CD 部署以及在 GateKeeper、Site‑Capabilities 与 IVOA DataLink 的注册与授权流程。

**💡 创新点**

创新点在于将传统的单一云 FaaS 方案迁移到多中心联邦环境，设计了跨站点身份验证、权限管控、函数注册与数据关联机制，实现真正的数据近位计算与统一可发现性。

**🔧 技术方法**

使用技术包括 Kubernetes + GitOps（Flux）、容器镜像仓库（Harbor）、FastAPI + ASGI、OpenFaaS/Knative 兼容的 FaaS 平台、Rucio 数据湖、SKAO IAM（OIDC）、GateKeeper 访问网关、Site‑Capabilities 目录与 IVOA DataLink 接口。

**📊 数据集**

数据集主要为 SKA 产生的 FITS 图像（存储于 Rucio Datalake 复制节点），以及相关的 RSE 本地副本；函数通过对这些 FITS 进行高斯卷积处理验证功能可行性。

**📈 对比分析**

与传统集中式处理相比，FaaS 架构显著降低了跨站点的数据传输量与网络延迟，提升了整体吞吐量；虽然本文未给出精确基准，但经验表明在本地节点执行可将网络延迟压缩至数秒级，且不再需要复制完整数据集。

**⚠️ 局限性**

局限性包括：联邦治理带来的运维复杂度、缺乏对多函数链路与临时数据流的原生支持、不同 SRC 节点硬件与能力不一致导致的性能异质性，以及在大规模部署前仍需完善容错、调度与可追溯性机制。

---

## 655. Kernel Alignment-based Multi-view Unsupervised Feature Selection with Sample-level Adaptive Graph Learning

**arXiv ID:** 2601.07288 | [PDF](https://arxiv.org/pdf/2601.07288v1)

**作者:** Yalan Tan `[一作]` (Southwestern University of Finance and Economics), Tianrui Li `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 25984 | [OpenAlex ID](https://openalex.org/A5070559820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为 KAFUSE 的多视角无监督特征选择方法，该方法通过核对齐与正交约束降低特征冗余，并利用样本级自适应权重学习跨视角一致相似图，最终实现更具判别力和独立性的特征子集。

**💡 创新点**

创新点：① 将核对齐与正交约束联合使用，既能捕捉线性关系也能捕捉非线性冗余；② 在一致相似图学习中引入样本级自适应视图权重，替代传统的样本不变权重，更好地保留局部结构；③ 将图学习与特征选择通过统一的目标函数耦合，形成互补提升。

**🔧 技术方法**

主要技术：核对齐、正交约束、张量堆叠与样本级权重融合、图拉普拉斯正则化、广义幂迭代、近端梯度下降、交替优化算法。

**📊 数据集**

实验数据集：NGs、Prokaryotic、COIL20、Cora、AD、CiteSeer、ALOI、Caltech 八个多视角真实数据集。

**📈 对比分析**

与八个现有多视角/单视角无监督特征选择方法（如 GAWFS、RNE、CFSMO、CDMvFS、MAMFS、CE-UMFS、MFSGL、UKMFS、WLTL）在 ACC 与 NMI 上进行比较。KAFUSE 在大多数数据集上均取得最高 ACC/NMI，平均提升 4%–11%，且通过 Wilcoxon 检验显示显著性。

**⚠️ 局限性**

局限性：① 需要手动调节三个超参数（α、β、r）和特征选择比例；② 对参数敏感，过多依赖网格搜索；③ 仅在无标签场景下验证，未考虑有标签信息；④ 对大规模数据的计算复杂度仍高。

---

## 656. A High-Recall Cost-Sensitive Machine Learning Framework for Real-Time Online Banking Transaction Fraud Detection

**arXiv ID:** 2601.07276 | [PDF](https://arxiv.org/pdf/2601.07276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 657. Focal Guidance: Unlocking Controllability from Semantic-Weak Layers in Video Diffusion Models

**arXiv ID:** 2601.07287 | [PDF](https://arxiv.org/pdf/2601.07287v1)

**作者:** Yuanyang Yin `[一作]` (MoE Key Lab of BIPC), Feng Zhao `[通讯]` (MoE Key Lab of BIPC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为Focal Guidance（FG）的框架，用于提升图像到视频（I2V）生成模型在文本提示下的指令遵循能力。

**💡 创新点**

创新点在于：①发现并定位Diffusion Transformer模型中的“Semantic‑Weak Layers”与“Condition Isolation”问题；②设计Fine‑grained Semantic Guidance与Attention Cache两种机制，直接修复中间层对文本的弱响应；③构造专门评估I2V模型指令遵循的基准。

**🔧 技术方法**

技术包括Diffusion Transformer（DiT）架构、CLIP视觉‑文本对齐、交叉注意力、注意力缓存、视觉锚点注入、Moran’s I与标准差评估语义响应。

**📊 数据集**

使用内部12K视频+字幕小规模后训练数据；评估基准基于VQA框架的动态属性、人类动作与交互三维度。

**📈 对比分析**

与Wan2.1‑I2V、HunyuanVideo‑I2V等公开模型对比，FG分别提升总分3.97%（0.6973→0.7250）和7.44%（0.5185→0.5571）；在原始视觉一致性指标上保持不变，表明仅增强语义控制。

**⚠️ 局限性**

局限性：FG效果受底层图像编码器与基模型能力限制；对传统视频质量与一致性指标影响有限；需更大规模多样化数据进一步验证。

---

## 658. AdaMorph: Unified Motion Retargeting via Embodiment-Aware Adaptive Transformers

**arXiv ID:** 2601.07284 | [PDF](https://arxiv.org/pdf/2601.07284v1)

**作者:** Haoyu Zhang `[一作]` (JD Explore Academy), Zecui Zeng `[通讯]` (JD Explore Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种统一的神经网络框架 AdaMorph，能够将人类动作映射到多种机器人形态，实现一次训练即可控制不同机器人。

**💡 创新点**

创新点在于将运动语义与形态执行解耦：使用双路径提示（token‑level 注意力 + 通过 AdaLN 的层级调节）让单一模型学习可迁移的“意图”空间，并通过物理一致性训练保证运动可行性。

**🔧 技术方法**

采用 Transformer 编码器/解码器、Adaptive Layer Normalization（AdaLN）、软提示（human/robot prompt）、基于物理的损失（SO(3) 角度一致性、可微整合轨迹一致性）以及自适应训练调度。

**📊 数据集**

训练数据来自 AMASS 大规模人类捕捉数据，使用 General Motion Retargeting 方法生成 12 种人形机器人的对齐控制信号，构成约 30M 条输入‑输出序列。

**📈 对比分析**

与传统针对单一机器人训练的专用模型相比，AdaMorph 在 12 台机器人上实现了高质量实时重定向；根部速度 Pearson 相关系数平均 >0.85，整体活动一致性 >0.85，零样本对未知民族舞蹈的迁移效果亦保持高同步性。

**⚠️ 局限性**

局限性：需要先用优化方法生成训练对齐数据，受限于机器人可达性；仅在类人机器人上验证；在极端动态或非类人形态下可能需要进一步改进提示与物理约束。

---

## 659. ReasonTabQA: A Comprehensive Benchmark for Table Question Answering from Real World Industrial Scenarios

**arXiv ID:** 2601.07280 | [PDF](https://arxiv.org/pdf/2601.07280v1)

**作者:** Changzai Pan `[一作]` (Institute of Artificial Intelligence), Zhongjiang He `[通讯]` (Institute of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向工业场景的双语 TableQA 基准 ReasonTabQA，并提出了利用表格可验证奖励的强化学习方法 TabCodeRL。

**💡 创新点**

创新点在于：① 通过多表、多层级标题、超大规模表结构三大复杂性，覆盖 30 个行业子领域，提供“思考/非思考”双模式完整推理路径；② 设计了基于表格路径选择和代码相似度的可验证奖励，显著提升程序式推理的准确性；③ 在公开数据集上实现了与大模型对比的透明评估，揭示工业 TableQA 的巨大挑战。

**🔧 技术方法**

技术主要包括：① 监督微调（SFT）+ DAPO 强化学习框架；② 三阶段奖励设计（格式、执行、答案；表格路径 F1；CodeBLEU 相似度）；③ 半自动化问题生成与人工校验流程。

**📊 数据集**

使用的数据集为 ReasonTabQA：1932 张表（1101 中文、831 英文），5523 题目，涵盖 30 个行业子领域，平均每张表 138 行、1359 列；另外在 4 个公开基准（WTQ、AITQA、MimoTable、HiTab）做对照。

**📈 对比分析**

与 29 种 LLM（开放源、闭源、表格专用）进行评测，结果显示即使是 Gemini‑3‑Pro‑Preview 的整体准确率仅 67.58%。TabCodeRL 在 Qwen3‑8B‑Instruct 上实现了 20% 以上的绝对提升，超越同等参数规模的大模型；在其他基准上也保持显著的性能提升。

**⚠️ 局限性**

局限性包括：① 仍未覆盖全球所有工业细分领域；② 仅包含中文和英文两种语言，缺少其他语言的适配；③ 评测主要聚焦程序生成的可执行性，对推理链的多样性和可解释性探讨有限。

---

## 660. Towards Comprehensive Semantic Speech Embeddings for Chinese Dialects

**arXiv ID:** 2601.07274 | [PDF](https://arxiv.org/pdf/2601.07274v1)

**作者:** Kalvin Chang `[一作]` (University of California Berkeley), Dong Yu `[通讯]` (Tencent)

**通讯引用:** 44449 | [OpenAlex ID](https://openalex.org/A5034476404)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于Zipformer编码器构建全方位覆盖多种中文方言的ASR模型，并通过Speech‑to‑Speech检索验证其跨方言语义一致性，首次公开YuBao方言检索基准；

**💡 创新点**

①提出利用仅含ASR数据即可诱导跨方言语义对齐的观点；②公开覆盖七大方言分支的YuBao检索数据集；③通过SeqSim评估跨方言检索召回率，展示ASR‑only训练已形成共享语义空间；

**🔧 技术方法**

Zipformer编码器 + RNN‑T + 注意力解码器；SeqSim 句子级BERTScore；SpecAugment；ScaledAdam+Eden学习率调度；

**📊 数据集**

34,000小时覆盖Mandarin、Yue、Wu、Min、Hakka、Xiang的ASR语料；YuBao方言检索语料（3,499句、6h45m）；公开的标准方言语料如AISHELL、KeSpeech等；

**📈 对比分析**

与Paraformer、FireRed‑AED等主流方言ASR模型对比，Zipformer在除Mandarin外多达6种方言的CER均显著下降；Speech‑to‑Speech检索召回率在Mandarin↔方言之间普遍>80%，并在多组间超过随机；

**⚠️ 局限性**

缺少Gan方言的训练样本导致召回率最低；检索基准主要使用读音老年男性语料，缺乏多样性；模型仍需进一步压缩与优化，尚未在真正低资源方言或跨语系场景验证；

---

## 661. Innovation Capacity of Dynamical Learning Systems

**arXiv ID:** 2601.07257 | [PDF](https://arxiv.org/pdf/2601.07257v1)

**作者:** Anthony M. Polloreno `[一作]` `[通讯]`, Anthony M. Polloreno

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过引入创新容量（innovation capacity）这一概念，补充传统的可预测信息处理容量（information‑processing capacity），并在噪声物理储层中证明了可预测与创新容量的守恒定律：C_ip + C_i = rank(Σ_XX) ≤ d。作者进一步推导出在线性高斯（Johnson‑Nyquist噪声）系统中的通用特征值收缩公式，给出了温度与可预测容量的单调权衡；对创新子空间进行几何解析（椭球体分解）并证明大创新容量导致高维τ‑创新子空间和长序列块熵的指数增长；最后利用典型集包装、KL与TV距离以及Fano不等式，给出了学习创新块分布的平均信息理论下界。

**💡 创新点**

创新点包括：
- 引入“创新容量”并证明其与可预测容量的严格分割关系。
- 在Johnson‑Nyquist噪声下得到可预测/创新容量的特征值收缩闭式表达。
- 将创新容量解释为白化空间中椭球体的体积份额，提供直观几何视角。
- 通过弱依赖与反集中假设，证明τ‑创新子空间的块熵下界，从而得到可区分历史的指数上界。
- 结合典型集与Fano理论，给出学习创新块分布的平均TV误差下界，揭示了创新容量大时的学习难度。

**🔧 技术方法**

主要技术手段：
- Doob可预测/创新分解与Hilbert空间投影。
- 协方差矩阵分解、特征值收缩与广义特征值问题。
- 椭球体几何与白化坐标变换。
- 统计学习理论：典型集、KL/TV距离、Fano不等式。
- 线性状态空间与Duffing非线性动力学仿真。

**📊 数据集**

数据集：
- 纯模拟数据，分别来自线性RLC电路和非线性Duffing振荡器的I/Q读取；
- 通过在不同温度（噪声强度）与非线性系数下生成输入序列并记录输出，用于计算容量指标。

**📈 对比分析**

比较方法：
- 将可预测容量 C_ip 与创新容量 C_i 在不同温度/非线性强度下绘图；
- 对比理论公式（特征值收缩、椭球体体积）与仿真估计；
- 结果表明：随着温度升高，可预测容量单调下降，创新容量单调上升，总和保持恒定；
- 线性RLC实验与Duffing实验均验证了理论预测，展示了容量随参数变化的趋势。

**⚠️ 局限性**

限制与不足：
- 许多结论基于线性高斯或弱依赖、反集中假设，实际硬件噪声分布可能更复杂。
- 典型集与熵下界需要较强的弱依赖假设，未在真实数据上验证。
- 仅使用模拟数据，缺乏对真实物理储层（光学干涉仪、Ising机等）的实验验证。
- 对非线性系统的分析主要在Duffing振荡器，泛化到更复杂非线性动力学仍待研究。

---

## 662. Universal Adversarial Purification with DDIM Metric Loss for Stable Diffusion

**arXiv ID:** 2601.07253 | [PDF](https://arxiv.org/pdf/2601.07253v1)

**作者:** Li Zheng `[一作]` (University of Macau), He YiMin `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种针对稳定扩散模型（Stable Diffusion）的通用对抗性净化框架 UDAP，利用 DDIM 逆向重建误差对抗样本进行优化，自动调节净化迭代轮数；

**💡 创新点**

创新点在于首次针对生成模型的对抗攻击构建净化方法，设计了基于 DDIM 逆向重建的度量损失来判别对抗噪声，并引入动态迭代阈值实现高效净化；

**🔧 技术方法**

核心技术包括 Stable Diffusion 的 VAE 编码/解码、DDIM 逆向与正向推理、基于 L2 的 DDIM 计量损失、动态阈值自适应优化；

**📊 数据集**

实验使用 VGGFace2、CelebA‑HQ（4 张/ID）和 ImageNet（1000 张）等公开人脸与通用图像数据集；

**📈 对比分析**

与 DiffPure、GridPure 等基准方法对比，在 PID、Anti‑DB、MIST、Anti‑DF、MetaCloak 等多种攻击下，UDAP 在 FDFR、ISM、BRISQUE、FID 等指标上均优于对照组，并通过动态阈值将平均净化时长从 18 秒降低到约 3 秒；

**⚠️ 局限性**

局限性主要体现在：①需依赖 SD VAE/UNet 结构，跨模型兼容性仍有待验证；②对极端强度对抗噪声的鲁棒性未充分评估；③在资源受限环境下的实时性仍受限于 DDIM 逆向迭代的计算开销。

---

## 663. MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences

**arXiv ID:** 2601.07251 | [PDF](https://arxiv.org/pdf/2601.07251v1)

**作者:** Zizhen Li `[一作]` (Shanda AI Research), Kaipeng Zhang `[通讯]` (Shanda AI Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了专门的桌游批评模型 MeepleLM，能够根据游戏规则书自动生成面向不同玩家人设的体验评价。

**💡 创新点**

创新点在于将游戏设计理论 MDA 的因果链式推理与数据驱动的玩家人设相结合，模型内部模拟玩家视角并生成具有情感与技术细节的批评；同时首次在大型语言模型上实现规则书到体验的直接映射。

**🔧 技术方法**

技术上采用 Qwen3-8B 作为基础模型，通过 LoRA 微调与 Persona‑Conditioned Instruction 调整；使用 Qwen3-235B 与 GPT‑5.1 作为教师模型完成 MDA 推理链生成，最终得到带有 CoT 的批评输出。

**📊 数据集**

使用了 1,727 本结构化规则书与约 150,000 条经过质量分层筛选的评论数据，结合从评论中提炼出的 5 类玩家人设作为训练与评估素材。

**📈 对比分析**

与 GPT‑5.1、Gemini3‑Pro、Qwen3‑235B 等最先进通用 LLM 进行对比，MeepleLM 在社区评分对齐（MAE、Wasserstein、Kendall τ）、内容准确度、词汇多样性、观点覆盖度等指标上均优于基线；用户实验中对 MeepleLM 的偏好率约为 70%。

**⚠️ 局限性**

主要限制包括：①仅处理文本规则，未整合视觉资产的多模态信息；②人设仅为 5 类聚类，缺乏对个体玩家的细粒度建模。

---

## 664. Rate-distortion Theory on Non-compact Spaces: A Concentration-compactness Approach

**arXiv ID:** 2601.07246 | [PDF](https://arxiv.org/pdf/2601.07246v1)

**作者:** Jiayang Zou `[一作]` (Shanghai Jiao Tong University), Jia Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 28772 | [OpenAlex ID](https://openalex.org/A5118788614)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了非紧致空间下的最优重构分布存在性问题

**💡 创新点**

引入集中-紧致性原理并放宽了失真函数的连续性要求，提出了新的存在性证明框架

**🔧 技术方法**

采用变分分析、测度论、Lions 的集中-紧致性原理及低半连续性与可压缩性条件

**📊 数据集**

无具体实验数据集，主要为理论推导与证明

**📈 对比分析**

未给出数值实验或性能比较，本文关注理论证明而非算法实现

**⚠️ 局限性**

对有额外约束（如因果或非预测性重构）的非紧致空间问题尚未覆盖，需进一步研究

---

## 665. Bias-Aware BP Decoding of Quantum Codes via Directional Degeneracy

**arXiv ID:** 2601.07240 | [PDF](https://arxiv.org/pdf/2601.07240v1)

**作者:** Mohammad Rowshan `[一作]` `[通讯]` (University of New South Wales), Mohammad Rowshan (University of New South Wales)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在量子CSS码上利用方向性注释的边缘权重来指导贝叶斯传播（BP）解码，提出了一种通过单参数β调节的方向性偏置优先级，直接用于BP+OSD解码器。

**💡 创新点**

创新点在于将定向权重聚合为每个量子比特的方向性成本，构造方向性退化枚举器，并证明了该枚举器与宏观距离和退化类数的关系，从而在不改动码结构的前提下实现了可调的解码偏置。

**🔧 技术方法**

采用了加权BP与有序统计解码（OSD）相结合的技术，利用方向性权重生成的LLR进行解码；同时使用MacWilliams等式给出枚举器的解析表达式，并通过理论推导给出距离与退化类数的上界。

**📊 数据集**

实验数据来源于两组有限长度的量子LDPC码：162,2,9的环面码和36,4 NE3N平面码，在代码容量噪声模型下进行模拟。

**📈 对比分析**

与传统等方位BP+OSD基线比较，利用方向性偏置的解码器在中等物理错误率（10^-3到10^-2）下逻辑错误率下降约1–2个数量级，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括：若噪声与预设方向不匹配或噪声本身近似i.i.d.，单一方向性偏置可能无效甚至降低性能；此外，复杂的多比特相关噪声模式需要更丰富的硬件模型，而本文仅采用了单参数平滑梯度。

---

## 666. Explaining Machine Learning Predictive Models through Conditional Expectation Methods

**arXiv ID:** 2601.07313 | [PDF](https://arxiv.org/pdf/2601.07313v1)

**作者:** Silvia Ruiz-España `[一作]` (Universitat Politècnica de València), Joaquim Arlandis `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5091139582)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了多变量条件期望（MUCE）及改进的ICE方法，用于解释黑盒模型的局部行为；

**💡 创新点**

通过多变量网格探索特征交互并引入稳定性与不确定性指标，为局部解释提供定量评估；

**🔧 技术方法**

使用改进的ICE、贪心搜索的MUCE算法以及稳定性/不确定性指标，并在XGBoost模型上实现；

**📊 数据集**

在2D、3D合成数据以及对U.S. Census 1990数据做特征转换后的真实数据集上进行实验；

**📈 对比分析**

合成数据AUC近乎100%，真实数据转化后87.55%；指标能够揭示模型在决策边界附近的局部不稳定性；

**⚠️ 局限性**

仅适用于概率输出，未扩展至回归等任务；对非表格数据无适配；贪心搜索可能陷入局部极值；生成的网格样本可能不真实且易产生协变量偏差。

---

## 667. ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge Evaluation Plan

**arXiv ID:** 2601.07303 | [PDF](https://arxiv.org/pdf/2601.07303v1)

**作者:** Xueping Zhang `[一作]` (Duke Kunshan University), Ting Dang `[通讯]` (University of Melbourne)

**通讯引用:** 600 | [OpenAlex ID](https://openalex.org/A5071116593)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了ESDD2环境感知语音与环境声深度伪造检测挑战，并提供CompSpoofV2数据集与基线框架。

**💡 创新点**

在语音与环境声可独立合成/篡改的场景下提出组件级深度伪造检测与分离增强联合学习框架。

**🔧 技术方法**

采用语音/环境声分离网络结合专门的反伪造模型，联合训练以保留伪造相关线索，并输出五类分类。

**📊 数据集**

采用250k+样本、283小时的CompSpoofV2数据集，涵盖五类真伪组合，并加入未见的合成样本。

**📈 对比分析**

以5类宏F1为主评估指标，基线在验证集上宏F1 0.946，评估集0.622，测试集0.633，展示模型在复杂混合伪造场景下仍具有效性。

**⚠️ 局限性**

仍依赖人工合成的分离模型，对极端噪声或高度复杂环境声的鲁棒性不足，且对未见伪造技术的泛化能力有限。

---

## 668. Mimic Human Cognition, Master Multi-Image Reasoning: A Meta-Action Framework for Enhanced Visual Understanding

**arXiv ID:** 2601.07298 | [PDF](https://arxiv.org/pdf/2601.07298v1)

**作者:** Jianghao Yin `[一作]` (ByteDance), Liang He `[通讯]` (East China Normal University)

**通讯引用:** 10007 | [OpenAlex ID](https://openalex.org/A5061101217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出CINEMA框架，使用五种结构化meta‑action（Global、Focus、Hint、Think、Answer）分步完成多图像推理，并通过检索式树采样进行冷启动训练，再采用两阶段强化学习（DPS+DAPO）提升推理质量。

**💡 创新点**

创新点包括：①将多图推理拆解为五步人类认知过程；②检索式树采样生成多样化、准确的meta‑action路径；③两阶段RL策略在保持多样性的同时逐步收敛，避免熵坍塌。

**🔧 技术方法**

技术手段：meta‑action框架、检索式树采样、Diversity‑Preserving Strategy (DPS)、Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)、Qwen2.5‑VL 7B 作为基座、精细化奖励设计与格式约束、Pass@K 评估。

**📊 数据集**

使用57k冷启动实例与58k RL实例（涵盖多图像、多帧、单图任务）构建训练集；在MUIR、MVMath、EMMA、VideoMME、VideoMMMU、MMIU、MIRB、MM-IQ、MM-Math、Math‑Vision、MathVista、MMMU、MMVerse等多图、视频及单图基准上进行评测。

**📈 对比分析**

与GPT‑4o、GPT‑4V、Gemini‑1.5‑Pro等闭源大模型及OpenFlamingo、LLaVA、InternVL、VILA、Mantis‑Idefics、VideoR1等开源/增强模型对比；在MUIR、MVMath、EMMA、VideoMME、VideoMMMU上实现SOTA，部分指标超过GPT‑4o；Pass@K实验显示两阶段RL显著提升成功率；消融实验证明双路径训练与每种meta‑action对性能都有显著贡献。

**⚠️ 局限性**

局限性：①训练与推理仍依赖大模型与高算力，部署成本较高；②依赖GPT‑4o等教师模型的路径检索，若教师性能不足易影响质量；③在极大图像集合或高分辨率视频时的可扩展性与实时性尚待验证；④meta‑action定义仍是人为设定，可能不适用于所有新颖任务。

---

## 669. LRAS: Advanced Legal Reasoning with Agentic Search

**arXiv ID:** 2601.07296 | [PDF](https://arxiv.org/pdf/2601.07296v1)

**作者:** Yujin Zhou `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18216 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LRAS框架，将法律LLM从闭环思维转为主动探询，能识别知识边界并自主搜索

**💡 创新点**

结合自省模仿学习与难度感知强化学习，首次让模型学会何时搜索及如何多步搜索

**🔧 技术方法**

使用全参数微调、GRPO强化学习、结构化标签化推理轨迹

**📊 数据集**

采集自JEC-QA、CAIL2018/2021、LexEval、LawBench、UniLaw及DiscLaw等中文法律基准

**📈 对比分析**

相较于基线模型提升8.2–32%（如LRAS-RL 14B在LexEval上平均67.49%），并在OOD DiscLaw 75.66%优于所有对手

**⚠️ 局限性**

受限于模型规模与工具精度，无法覆盖极端复杂多层查询，且需进一步提升对检索错误的鲁棒性

---

## 670. VideoLoom: A Video Large Language Model for Joint Spatial-Temporal Understanding

**arXiv ID:** 2601.07290 | [PDF](https://arxiv.org/pdf/2601.07290v1)

**作者:** Jiapeng Shi `[一作]` (Fudan University), Zuxuan Wu `[通讯]` (Fudan University)

**通讯引用:** 7572 | [OpenAlex ID](https://openalex.org/A5026167547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VideoLoom 套件，实现视频的联合时空理解，包含新数据集 LoomData‑8.7k 与基准 LoomBench；

**💡 创新点**

创新点在于：① 自动化的多阶段时空标注管线，生成统一的时间戳与空间掩码；② 采用 SlowFast 视觉 token 结合 MLLM‑SAM2 架构，兼顾低帧速率的时序与高分辨率的空间；③ 引入双向前景 J&F 评估度量，提升联合时空问题的评测可靠性；

**🔧 技术方法**

核心技术包括：多模态大型语言模型 InternVL3、视觉分割模型 SAM2、视觉 token 采样（SlowFast）、视觉编码器、LoRA 微调、自动化标注工具（GroundingDINO、SAM2、Gemini2.5pro 等）；

**📊 数据集**

使用的数据集：ActivityNet 通过自动管线生成 LoomData‑8.7k（8,710 个镜头），验证集生成 LoomBench（130 个视频，541 When、487 Where、456 Combined 问答对）；此外还使用了 Charades‑STA、YouCook2、QVHighlights、MeVIS、ReVOS、RefYTVOS、RefCOCO 等公开基准；

**📈 对比分析**

与现有 Video LLM（TimeChat、VTG‑LLM、TRACE、TimeSuite、UniTime、Sa2VA 等）比较，VideoLoom 在时序任务上获得 48.3 R1@0.7（Charades‑STA）和 63.3 HIT@1（QVHighlights）；在空间任务上取得 51.7、71.3、63.1 的 J&F（MeVIS、RefYTVOS、ReVOS）；在 LoomBench 上 Combined 问题实现 +16.2 tIoU、+15.4 J&F_bi‑fore 的提升，整体性能居领先；

**⚠️ 局限性**

局限性包括：① 仍需多阶段手工校验的标注管线，自动化程度不够；② 对长视频多场景切换的鲁棒性有限；③ 在极大模型规模下仍受限于 GPU 资源与推理效率；

---

## 671. ColorBrowserAgent: An Intelligent GUI Agent for Complex Long-Horizon Web Automation

**arXiv ID:** 2601.07262 | [PDF](https://arxiv.org/pdf/2601.07262v1)

**作者:** Jiamu Zhou `[一作]` (OPPO Research Institute), Jun Wang `[通讯]` (OPPO Research Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ColorBrowserAgent框架，通过Progressive Progress Summarization与Human-in-the-Loop Knowledge Adaptation实现浏览器任务自动化。

**💡 创新点**

创新点包括双代理协同架构、进阶记忆压缩保持长时稳定、基于专家提示的知识库适配多样化网站，以及混合触发人类干预机制。

**🔧 技术方法**

使用GPT‑5作为核心推理引擎，结合VLM与规则判别器做干预触发，Summarizer Agent进行结构化摘要，Adaptive Knowledge Base存储专家提示，扩展动作空间和计算器工具。

**📊 数据集**

在WebArena基准（812任务，涵盖GitLab、Reddit、Shopping、Shopping Admin、Map）以及WebArena‑Lite子集进行评估。

**📈 对比分析**

与多种公开与闭源基线（OpenAI Operator、Claude Code+GBOX、IBM CUGA等）对比，ColorBrowserAgent取得71.2%成功率，显著高于最高基线68.0%，并在各子域表现优异。

**⚠️ 局限性**

局限性包括在高度视觉化、动态DOM丰富的Map域表现欠佳；对强大模型（GPT‑5）的依赖导致成本较高；人类干预仍需手工提示，触发机制对VLM与规则的准确性敏感。

---

## 672. ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models

**arXiv ID:** 2601.07260 | [PDF](https://arxiv.org/pdf/2601.07260v1)

**作者:** Huipeng Ma `[一作]` (Beijing Institute of Technology), Shuhao Zhang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 1240 | [OpenAlex ID](https://openalex.org/A5100696504)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了ActiShade框架，通过迭代检测查询中的被掩盖关键词、检索相关文档并生成新查询来改进多跳推理。

**💡 创新点**

创新点在于引入Gaussian Perturbation (GaP)检测知识被掩盖，以及针对被掩盖关键词的对比学习检索器和查询生成策略。

**🔧 技术方法**

使用的技术包括Gaussian噪声扰动、对比学习损失、密集检索器、LLM生成与检索交互。

**📊 数据集**

实验数据集涵盖HotpotQA、2WikiMQA和MuSiQue三个多跳推理数据集。

**📈 对比分析**

与多种基线（如IRCoT、Iter-RetGen、Self-Ask、DRAGIN等）对比，ActiShade在所有数据集和LLM上均取得最高的Acc/F1，显著提升性能。

**⚠️ 局限性**

限制在于对噪声标准差的调节仍需经验，且在更大模型上的实现受限于硬件资源。

---

## 673. DDT: A Dual-Masking Dual-Expert Transformer for Energy Time-Series Forecasting

**arXiv ID:** 2601.07250 | [PDF](https://arxiv.org/pdf/2601.07250v1)

**作者:** Mingnan Zhu `[一作]` (Xiamen University), Shiming Lin `[通讯]` (Xiamen University)

**通讯引用:** 1230 | [OpenAlex ID](https://openalex.org/A5111574544)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出DDT框架，使用双重掩码（严格因果+数据驱动动态）与双专家系统（时序专家+通道专家）实现能源时序预测。

**💡 创新点**

创新点：①将因果约束与动态特征选择融合为单一掩码；②将时间序列内部动力学与变量间相关性分离成并行专家并通过动态门控融合，显著提升适应性与精度。

**🔧 技术方法**

技术手段包括Transformer基础结构、动态因果掩码（基于Mahalanobis距离+Gumbel‑Softmax采样）、多尺度CNN‑Attention嵌入、动态门控融合模块、可配置的CI模式以及多尺度Patch化和STMA注意力。

**📊 数据集**

数据集：7个能源基准集（ETTh1/2、ETTm1/2、Electricity、Solar、Wind、Traffic）。

**📈 对比分析**

与10个SOTA模型（Pathformer、iTransformer、TimeMixer、FITS、PDF等）在MSE/MAE上对比，DDT在所有预测时长与所有数据集上均排名第一，性能提升显著，尤其在长周期和高波动场景。

**⚠️ 局限性**

局限：模型结构复杂，参数量和计算成本高；可解释性不足；对不同超参数和训练设置的敏感性未深入探讨。

---

## 674. Pseudodata-guided Invariant Representation Learning Boosts the Out-of-Distribution Generalization in Enzymatic Kinetic Parameter Prediction

**arXiv ID:** 2601.07261 | [PDF](https://arxiv.org/pdf/2601.07261v1)

**作者:** Haomin Wu `[一作]` (Pengcheng Laboratory), Zhixiang Ren `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 O^2DENet 模块，通过伪数据增强与不变表征学习提升酶-底物相互作用预测的 OOD 泛化。

**💡 创新点**

创新点在于轻量化 plug‑and‑play 方案，利用酶序列掩码、底物 SMILES 枚举和分子图掩码产生伪样本，并加入特征一致性正则化，无需改动主干网络即可显著提升泛化。

**🔧 技术方法**

使用了酶序列随机掩码、底物 SMILES 枚举、分子图掩码等扰动增强技术，结合特征一致性损失实现不变表征学习，并评估 R²、MAE 与 AU‑GOOD 三个指标。

**📊 数据集**

使用了 CatPred k_cat / K_m 基准数据集、TAL 与 MS 定向进化数据集进行实验。

**📈 对比分析**

与 UniKP、DLKCat、CatPred、OmniESI 四大主流框架对齐，O^2DENet 在 40%–99% 序列相似度 OOD 测试中 R² 提升最高达 145.4%，MAE 降低 18.5%，AU‑GOOD 也提升 5–60% 级别，零样本预测成功率约 75%。

**⚠️ 局限性**

局限性包括对掩码比例敏感，过度扰动会破坏关键信息；在极端 OOD 或不同酶家族时可能需要更复杂的扰动策略；目前仅验证了基于序列相似度的 OOD 拆分，未覆盖其他分布偏移场景。

---

## 675. Improved lower bounds for the maximum size of Condorcet domains

**arXiv ID:** 2601.07336 | [PDF](https://arxiv.org/pdf/2601.07336v1)

**作者:** Alexander Karpov `[一作]` (Higher School of Economics University), Bei Zhou `[通讯]` (Imperial College London)

**通讯引用:** 3958 | [OpenAlex ID](https://openalex.org/A5102390224)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文通过结构分析与递归搜索方法，寻找并构造了更大的Condorcet域（Condorcet domains），并进一步给出了新的渐近下界。

**💡 创新点**

创新点包括：1）利用最大域的子域结构设计递归扩展算法；2）在超级计算机上实现大规模搜索，得到 9≤n≤20 的最大域大小更新；3）结合已有构造方法，将这些域用于生成 21≤n≤25 的更大域；4）得到新的渐近下界 Ω(2.198139^n)。

**🔧 技术方法**

技术手段主要是：1）结构分析和never条件的约束求解；2）基于 Condorcet Domain Library 的后向搜索与剪枝；3）超级计算机并行计算；4）递归域拼接（1N3–3N1 与 2N3 构造）。

**📊 数据集**

使用的数据集为已知的最大 Condorcet 域（n≤8）以及通过搜索得到的域集合；并利用 Fishburn 域等作为基准。

**📈 对比分析**

与以往结果比较，本文在 9≤n≤20 时均超过前人上界，21≤n≤25 时通过构造进一步提升；在渐近增长率方面，由 2.1973 提升至 2.198139，成为当前最强的下界。

**⚠️ 局限性**

局限性包括：1）搜索对大 n 仍然计算量巨大，实验证明至 n=20 已耗时数日；2）对更大 n 的域构造仍需理论突破；3）域的结构性理解尚不完整，缺乏解析证明其最优性。

---

## 676. Reconstruction Guided Few-shot Network For Remote Sensing Image Classification

**arXiv ID:** 2601.07335 | [PDF](https://arxiv.org/pdf/2601.07335v1)

**作者:** Mohit Jaiswal `[一作]` (LNMIIT Jaipur), Biplab Banerjee `[通讯]` (IIT Bombay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于重建引导的少样本学习网络（RGFS‑Net），用于遥感图像分类。

**💡 创新点**

创新点在于将掩码图像重建任务与瓶颈潜在空间学习结合，形成多任务正则化，提升了特征的稳定性与泛化能力。

**🔧 技术方法**

采用预训练CNN编码器、解码器实现掩码重建、Triplet 余弦距离与原型分类损失，并使用DropBlock、Monte Carlo 前向传播做随机正则化。

**📊 数据集**

在 EuroSAT 与 PatternNet 两个遥感基准数据集上进行 1‑shot 与 5‑shot 任务评估。

**📈 对比分析**

与 Siamese、Relation、Prototypical、SPN 等现有少样本方法对比，RGFS‑Net 在多数配置下均提升 5–10% 的准确率，尤其在 5‑way 5‑shot 的未见类别上表现突出。

**⚠️ 局限性**

局限在于仍依赖较大规模预训练模型与大量训练迭代，且对非常小样本（1‑shot）在部分设置下提升有限。

---

## 677. Engineering Favorable Propagation: Near-Field IRS Deployment for Spatial Multiplexing

**arXiv ID:** 2601.07317 | [PDF](https://arxiv.org/pdf/2601.07317v1)

**作者:** Yuxuan Chen `[一作]` (Shanghai Jiao Tong University), Wen Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 17572 | [OpenAlex ID](https://openalex.org/A5100673541)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出利用稀疏阵列近场效果对 IRS 进行几何部署，以获得更高的空间复用度，并基于统计 CSI 设计低复杂度的最大比传输与功率分配算法，实现 IRS 辅助多用户 MIMO 系统的可行性与性能提升。

**💡 创新点**

创新点包括：
1) 推导近场球面波前对稀疏阵列的正交性提升机制，并给出可实现的 IRS 部署准则；
2) 通过解析 Dirichlet 核构造的优良传播度量，实现对用户间相关性的确定性调控；
3) 在 Rician 频率复用环境下，仅利用长期统计信息即可完成 IRS 相位与功率的交替优化，显著降低反馈与估计开销。

**🔧 技术方法**

技术手段：
- 近场球面波前的 2 阶 Taylor 展开；
- 期望与方差解析推导得到用户相关性与 EDoF 公式；
- 统计 CSI 下的有效 SINR 近似与最大比传输；
- 基于 Lagrange 边缘、二次变换与 ADMM 的交替优化算法；
- 仿真验证与 Monte‑Carlo 统计。

**📊 数据集**

数据集：使用基于 60 GHz 频率的仿真环境，BS 128 天线稀疏 ULA，IRS 1600 元素稀疏 UPA，K = 4、8、16 个单天线用户，Rician 随机化与多种功率与角度配置，未使用公开实际测量数据集。

**📈 对比分析**

与随机相位、理论无干扰基准等方案对比，实验表明：
- 在适当的 IRS 位置与稀疏度下，用户间相关系数显著下降，EDoF 接近 K；
- 经优化的算法在不同功率、Rician 因子及 IRS 大小下均实现比随机策略提升 5–20 dB 的吞吐量，逼近理论无干扰极限；
- 收敛速度良好，且增大 IRS 元素数可进一步提升性能。

**⚠️ 局限性**

局限性：
- 需要精确的长周期统计 CSI 与 IRS 位置信息，实际部署中可能受环境变化影响；
- 近场球面波前模型假设 IRS 与 BS 之间的 LoS 条件，非 LoS 或多径严重的场景性能需进一步验证；
- 算法最终收敛到局部最优点，无法保证全局最优；
- 计算复杂度仍随 IRS 元素数升高而显著增加，需进一步简化实现。

---

## 678. SCALPEL: Selective Capability Ablation via Low-rank Parameter Editing for Large Language Model Interpretability Analysis

**arXiv ID:** 2601.07411 | [PDF](https://arxiv.org/pdf/2601.07411v1)

**作者:** Zihao Fu `[一作]` (Chinese University of Hong Kong), Zhenguang G. Cai `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1601 | [OpenAlex ID](https://openalex.org/A5077498783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SCALPEL 框架，通过低秩 LoRA 参数编辑实现大语言模型的选择性能力消融。

**💡 创新点**

将能力视为分布在层和模块中的低秩子空间，而非单一模块，实现多重语义与分布式编码的精准消融。

**🔧 技术方法**

利用 LoRA 低秩适配器、概率平衡损失、文本正则化、范数与稀疏正则化，并在 transformer 上进行梯度优化。

**📊 数据集**

在 24 个多维能力任务、BLiMP 67 细粒度语言任务以及 WikiText-103 文本上进行训练与评估。

**📈 对比分析**

与八种现有解释/干预方法（DiffMean、Causal Tracing 等）对比，SCALPEL 在目标任务准确率下降最高、困惑度几乎不变，整体能力保持最佳。

**⚠️ 局限性**

对低秩假设的依赖导致对某些高维、复杂能力的消融效果有限，且在极大模型上训练成本仍高。

---

## 679. OceanSAR-2: A Universal Feature Extractor for SAR Ocean Observation

**arXiv ID:** 2601.07392 | [PDF](https://arxiv.org/pdf/2601.07392v1)

**作者:** Alexandre Tuel `[一作]` (Galeio), Bertrand Chapron `[通讯]` (LOPS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了一个基于DINOv2的、使用σ⁰归一化的海洋SAR基础模型OceanSAR-2，并构建了多任务海洋SAR基准测试框架。

**💡 创新点**

创新点是将物理校正的σ⁰输入、动态冗余剪枝、KoLeo正则化与iBOT局部预测损失相结合的自监督目标，实现了体量小、性能强的海洋SAR表示。

**🔧 技术方法**

使用了DINOv2自监督学习、iBOT局部补全损失、KoLeo正则化、动态数据清洗、ViT骨干网络、kNN零样本评估以及轻量化MLP/DETR微调。

**📊 数据集**

使用了Sentinel‑1 Wave Mode（WV）校正σ⁰图像；十分类别的TenGeoP；SWH回归的WV‑SWH；风速风向回归的WV‑wind；冰山检测的YOLOIB等数据集。

**📈 对比分析**

通过零样本kNN和微调（MLP/DETR）在上述五个基准上与TerraMind、WV‑Net、DINOv3比较，OceanSAR‑2在大多数任务零样本和微调后均表现优于或接近其它模型，且参数量仅2.1亿。

**⚠️ 局限性**

限制在于仅针对WV模式，缺少多极化/相位输入，基准集仍相对有限且主要基于2016年Sentinel‑1A，跨传感器推广与更广泛任务仍待验证。

---

## 680. On the Non-decoupling of Supervised Fine-tuning and Reinforcement Learning in Post-training

**arXiv ID:** 2601.07389 | [PDF](https://arxiv.org/pdf/2601.07389v1)

**作者:** Xueyan Niu `[一作]` (Huawei Technologies Co., Ltd.), Weixi Zhang `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并证明了在大型语言模型的后训练阶段，监督微调（SFT）与强化学习（RL）之间无法解耦：任一阶段都不可避免地会损害前一阶段的表现。

**💡 创新点**

提出了两个理论定理，分别说明SFT后接RL会导致SFT损失上升，RL后接SFT会导致RL奖励下降，揭示两种后训练策略的不可分离性。

**🔧 技术方法**

采用概率分布理论、KL 与 TV 不等式、PPO 与GRPO等强化学习算法以及交叉熵损失来分析和实验验证。

**📊 数据集**

在Qwen3-0.6B模型上使用CoLA（Corpus of Linguistic Acceptability）数据集进行SFT和RL训练。

**📈 对比分析**

通过在SFT-then-RL与RL-then-SFT两条管线中监测交叉熵损失与奖励变化，实验显示RL阶段会使SFT损失急剧上升，SFT阶段会使RL奖励急剧下降，验证了理论预期。

**⚠️ 局限性**

局限在于仅在单一模型与单一数据集上实验，缺乏对多模型、多任务和更复杂奖励机制的验证；理论假设（如奖励可见性、分布一致性）可能不完全符合实际训练环境。

---

## 681. Learning Dynamic Collaborative Network for Semi-supervised 3D Vessel Segmentation

**arXiv ID:** 2601.07377 | [PDF](https://arxiv.org/pdf/2601.07377v1)

**作者:** Jiao Xu `[一作]` (Dalian University of Technology), Lihe Zhang `[通讯]` (Dalian University of Technology)

**通讯引用:** 11184 | [OpenAlex ID](https://openalex.org/A5015500789)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种动态协同网络（DiCo）用于半监督3D血管分割，动态切换教师与学生模型，并加入多视角融合和MIP对抗监督。

**💡 创新点**

创新点包括：①基于模型当前表现动态决定教师/学生角色，减少传统静态Mean Teacher的认知偏差；②轻量级多视角整合模块捕捉局部与全局特征；③利用最大强度投影（MIP）将3D分割投影至2D进行对抗监督，提升血管形状连贯性。

**🔧 技术方法**

核心技术：动态协同网络（CNN+ViT交替监督）、多视角整合模块、MIP对抗监督、Dice+交叉熵混合损失、EMA更新、滑动窗口推理。

**📊 数据集**

使用三大血管数据集：ImageCAS（CT‑Angio）、CAS2023（MRA）、Parse2022（CT肺动脉），仅使用5%标注数据进行训练。

**📈 对比分析**

与现有半监督方法（MT、UA‑MT、SASSNet、SLCNet、MagicNet、CAML、CauSSL、GuidedNet）以及全监督方法（CTNet、DSCNet、ERNet、VNet）比较，DiCo在DSC、NSD、ASD三项指标上均明显领先，且仅用5%标注即可逼近全监督性能。

**⚠️ 局限性**

局限性：①对不同模态（如超声、PET）验证不足；②需要额外的投影与对抗模块，计算开销略高；③模型对超参数（教师/学生切换阈值、MIP投影方式）敏感，需进一步自动化。

---

## 682. On Narrative: The Rhetorical Mechanisms of Online Polarisation

**arXiv ID:** 2601.07398 | [PDF](https://arxiv.org/pdf/2601.07398v1)

**作者:** Jan Elfes `[一作]` (University College Dublin), Luca Maria Aiello `[通讯]` (IT University of Copenhagen)

**通讯引用:** 4901 | [OpenAlex ID](https://openalex.org/A5034406723)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 YouTube 视频与评论中的叙事极化，提出“叙事极化”概念，使用 Greimas 的 Actantial Model 通过大型语言模型自动提取叙事角色，量化不同党派搜索意图下的叙事差异，并检验评论在表层和深层叙事层面上对视频极化的缓和与残留作用。

**💡 创新点**

创新点：
1) 将结构化叙事理论与大规模 LLM 自动标注结合，实现对海量文本的叙事角色识别。
2) 将叙事极化定义为核心角色在叙事结构中的位置差异，提供了比传统情绪/主题极化更细致的框架。
3) 发现评论虽然在表层叙事上能显著降低极化，但在叙事动机、对立、依赖等更深层叙事模式上仍保留差异，揭示了极化的多维性。

**🔧 技术方法**

技术方法：
- 使用 DeepSeek‑R1‑Distill‑Qwen‑32B 对视频转写与评论进行 Actantial 角色标注。
- 统计分析包括重叠系数、主体偏差（subject divergence）、叙事动机差异，并采用置换检验与 bootstrap 置信区间验证显著性。
- 人工验证与多注释者一致性（Micro‑F1、Krippendorff α）评估标注质量。

**📊 数据集**

数据集：
- 212 个英文 YouTube 视频（107 pro‑Israel，105 pro‑Palestine），发布时间 2023‑10‑07 至 2024‑10‑01。
- 90,029 条英文评论（词数 25–600）。
- 视频转写采用 OpenAI Whisper，注释使用 11,000 条样本进行 LLM 与人工标注。

**📈 对比分析**

比较方法与性能：
- 重叠系数衡量视频与评论内部、两党派之间的叙事相似度；
- 主体偏差测量两党派对核心角色的分配差异；
- 置换检验与 bootstrap 计算 p 值与置信区间。
- 结果显示评论相较视频的主体偏差从平均 0.19 缩小至 0.07，表层叙事极化降低约 75%；
- LLM 标注平均 Micro‑F1 0.73，人工一致性 Krippendorff α 0.59，说明在大规模分析中保持可接受的准确性。

**⚠️ 局限性**

局限性：
1) 仅在 YouTube 平台上验证，缺乏跨平台或跨媒体的推广性。
2) 样本集中于英文视频，未涵盖多语言多文化视角。
3) 评论受视频受欢迎程度影响，可能偏向主流内容。
4) LLM 在低频角色与细粒度语义上的误差仍显著。
5) 只捕捉叙事结构，未深入探讨情绪、视觉与多模态信息对极化的影响。

---

## 683. Software-Hardware Co-optimization for Modular E2E AV Paradigm: A Unified Framework of Optimization Approaches, Simulation Environment and Evaluation Metrics

**arXiv ID:** 2601.07393 | [PDF](https://arxiv.org/pdf/2601.07393v1)

**作者:** Chengzhi Ji `[一作]` (Southeast University), Ziyuan Pu `[通讯]` (Southeast University)

**通讯引用:** 3141 | [OpenAlex ID](https://openalex.org/A5047891218)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套软件与硬件协同优化及闭环评估框架，提升模块化端到端自动驾驶推理的能耗与延迟。

**💡 创新点**

创新点在于将软件层面模型剪枝、量化与硬件层面的计算图优化融合，并引入多维评估指标EER_AV。

**🔧 技术方法**

采用模块化剪枝、混合精度量化、TensorRT图优化、操作符融合以及实时同步仿真技术。

**📊 数据集**

使用Bench2Drive数据集和CARLA仿真环境进行多场景测试。

**📈 对比分析**

通过与基线对比，推理延迟降低6×，每帧能耗约为基线的五分之一，EER_AV提升达22.35%。

**⚠️ 局限性**

局限性包括仅验证于ME2E框架，缺少对不同模型策略和场景对延迟敏感性的系统化分析。

---

## 684. Examining the Effectiveness of Transformer-Based Smart Contract Vulnerability Scan

**arXiv ID:** 2601.07334 | [PDF](https://arxiv.org/pdf/2601.07334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 685. Interactive visualizations for adolescents to understand and challenge algorithmic profiling in online platforms

**arXiv ID:** 2601.07381 | [PDF](https://arxiv.org/pdf/2601.07381v1)

**作者:** Yui Kondo `[一作]` (Oxford Internet Institute, University of Oxford), Luc Rocher `[通讯]` (Oxford Internet Institute, University of Oxford)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过开发交互式可视化工具 Algorithmic Mirror，允许 12–16 岁青少年上传其在 YouTube、TikTok 与 Netflix 的观看历史，并在统一的语义空间和时间轴上展示数据聚类、跨平台关联与算法推断，从而帮助青少年直观地了解平台如何收集、聚合并利用其数据来构建数字身份。

**💡 创新点**

创新点在于①使用真实用户的长期观看记录进行“现场化”可视化，突破传统模拟或游戏化工具的抽象性；②结合跨平台文本标准化、语义聚类与时间维度，首次将多平台多时段的数据以可交互镜像的形式呈现；③利用 LLM 进行跨平台内容摘要统一，使得不同平台的视频能够在同一空间聚类，从而揭示算法如何在多服务中一致地解读用户兴趣。

**🔧 技术方法**

技术实现包括：①文本嵌入（OpenAI text-embedding-ada-002）+ UMAP 降维得到二维可视化；②LLM（GPT‑4o‑mini）进行主题提取与描述标准化；③前端拖拽上传、缩放、时间滑块等交互；④后端对平台导出文件进行预处理、补全缺失字段（如使用 TMDB API 获取标题/描述），并在云端完成嵌入与聚类。

**📊 数据集**

数据集来自 27 名青少年上传的 YouTube、TikTok 与 Netflix 观看历史，总计约 750K 条视频记录，覆盖长达 5 年的时间跨度。数据经过平台导出后统一格式化，并用 LLM 生成统一的内容摘要后送入嵌入模型。

**📈 对比分析**

本文未采用对照实验或数值性能对比，而是通过两阶段用户研究（开放式探索 + 半结构化访谈）和主题分析来评估工具对用户对数据规模、跨平台聚合与算法推断的认知提升。参与者普遍报告认知显著提升、情感共鸣增强，且能在对话中引用具体可视化细节，但缺乏定量指标或与现有工具的直接性能比较。

**⚠️ 局限性**

主要局限包括：①样本偏向日本，跨国差异难以充分验证；②仅有 30% 的参与者提供多平台数据，导致跨平台聚合效果受限；③研究仅观察单次 60 分钟的即时反应，未能评估长期行为变化或持续使用效果；④工具依赖平台导出与云端 LLM，存在隐私、标准化和可持续部署的技术与政策挑战。

---

## 686. GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap

**arXiv ID:** 2601.07375 | [PDF](https://arxiv.org/pdf/2601.07375v1)

**作者:** Farzad Shami `[一作]` (Aalto University), Henrikki Tenkanen `[通讯]` (Aalto University)

**通讯引用:** 3606 | [OpenAlex ID](https://openalex.org/A5047335001)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需视觉、基于 OpenStreetMap 的层次化 LLM 框架，用来评估导航指令的可导航性。

**💡 创新点**

创新点在于：① 用地图知识取代视觉感知，实现“评估者”无需图像；② 采用结构化 JSON 表示地图并进行子指令分解，显著提升 LLM 的空间推理能力；③ 将执行结果作为评估指标，形成“Agent-as-Judge”的逆向评价范式。

**🔧 技术方法**

技术包括：大型语言模型（Gemini‑3 Pro）与 prompt engineering、JSON/图形结构化地图表示、层次化子指令与导航代理、基于 OSM 的可见区域构建、基于地理方向与距离的 POI 归一化。

**📊 数据集**

使用 Map2Seq 数据集（TestSet_A/ TestSet_B 两个 700 条样本子集），包含 OSM 节点、边、POI 与人类生成的导航指令。

**📈 对比分析**

与随机走者、动作采样、启发式规则三种基线对比；在 Navigation Error、Success Rate、Oracle Success Rate、SDTW 等指标上，提出方法分别在 TestSet_A、TestSet_B 上取得 NE≈56m、SR≈66%（相比基线的 NE≈180m、SR≈18%）并与人类评估显著相关，性能优越。

**⚠️ 局限性**

局限性在于：① 只评估基于结构与语义的导航，不支持依赖纯视觉提示的指令；② 依赖大型 LLM，计算成本与延迟高；③ 结果仅在 Gemini‑3 Pro 上验证，尚未确认在其他模型上的泛化性。

---

## 687. Agentic Diagnostic Reasoning over Telecom and Datacenter Infrastructure

**arXiv ID:** 2601.07342 | [PDF](https://arxiv.org/pdf/2601.07342v1)

**作者:** Nicolas Tacheny `[一作]` `[通讯]` (Ni Innovation Lab), Nicolas Tacheny (Ni Innovation Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的代理式诊断框架，利用Model Context Protocol（MCP）暴露的工具接口，对电信和数据中心基础设施进行根因分析（RCA）与影响分析，而非传统的图遍历或规则引擎。

**💡 创新点**

核心创新在于：1）将基础设施抽象为类型化图，并通过MCP工具封装数据访问，彻底分离模型与实现；2）定义结构化的调查协议，指导LLM进行逐步推理，保证可重复、可追溯；3）实现完全基于工具调用的无硬编码因果逻辑的诊断流程，开启自动化事件解决与变更影响预测的新方向。

**🔧 技术方法**

技术包括：大型语言模型（Claude Haiku 3.5、Llama 3.1 8B、GPT‑OSS‑120B）；ReAct‑style 工具调用机制；Model Context Protocol（MCP）工具接口；结构化的调查协议；对工具结果进行的语义检查与错误检测。

**📊 数据集**

数据集为人工构造的“合成基础设施本体”图，包含已知根因节点、事件、笔记及其影响集合。实验中共10个测试场景（从单一存储故障到多级服务层级），每个场景运行10次，形成100条实验记录。

**📈 对比分析**

对比三种模型在相同测试集上的调查准确率、RCA准确率、影响准确率和平均时长。Claude Haiku 3.5在所有指标均达到100%，时长≈20.9 s；GPT‑OSS‑120B在99/100/99%范围内，时长≈11.6 s；Llama 3.1 8B在79/91.1/86.1%范围，时长≈3.9 s，表明模型能力与精度、速度存在权衡。

**⚠️ 局限性**

限制：1）根因识别高度依赖本体数据完整性；2）缺乏显式时间推理，难以处理并发或已修复事件；3）在服务实现规模大时，工具调用量增大导致上下文限制；4）小模型易产生幻觉或协议偏离；5）目前未实现自动修复或置信度量化。

---

## 688. BayesRAG: Probabilistic Mutual Evidence Corroboration for Multimodal Retrieval-Augmented Generation

**arXiv ID:** 2601.07329 | [PDF](https://arxiv.org/pdf/2601.07329v1)

**作者:** Xuan Li `[一作]` (University of Science and Technology of China), Junnan Zhu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于贝叶斯推理与Dempster‑Shafer证据理论的多模态检索框架BayesRAG，能够在视觉丰富文档中通过语义与布局一致性来重新排序检索结果并生成更准确的答案。

**💡 创新点**

创新点在于将检索视为证据融合过程，使用贝叶斯后验来衡量文本-图像-布局三者的互相佐证程度；通过语义一致性先验和布局先验抑制模态冲突；采用Dempster‑Shafer融合提升对冲突的鲁棒性。

**🔧 技术方法**

核心技术包括：多模态嵌入（文本、图像、截图）、贝叶斯推理、Dempster‑Shafer证据融合、语义一致性先验（知识图谱）、布局先验（几何距离）、多模态向量检索与重排序。

**📊 数据集**

使用公开的文档问答基准：DocBench和MMLongBench‑Doc，覆盖金融、法律、政府、新闻等多领域长文档。

**📈 对比分析**

与RAGFlow、VisRAG、ViDoRAG、RAGAnything等现有SOTA多模态RAG基线相比，BayesRAG在整体评分上达51.2%（比最佳基线高2.5%），在政府、新闻、指南书等高噪声领域提升显著；召回率@20提升至76.6%，显著降低高相似度错误检索。

**⚠️ 局限性**

主要局限包括：检索与生成之间的差距，尽管检索召回率高，但生成模型（如GPT‑4o‑mini）对视觉证据的解析仍受限；在纯文本或大表格为主的场景中，多模态一致性优势不明显，需要进一步改进生成与表格处理。

---

## 689. Performance Bounds of Joint Detection with Kalman Filtering and Channel Decoding for Wireless Networked Control Systems

**arXiv ID:** 2601.07322 | [PDF](https://arxiv.org/pdf/2601.07322v1)

**作者:** Jinnan Piao `[一作]` (Chinese Academy of Sciences), Jincheng Dai `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1983 | [OpenAlex ID](https://openalex.org/A5069582226)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了无线网络控制系统中联合检测的性能界限，使用卡尔曼滤波（KF）来估计控制输出的先验概率，以辅助信道解码。

**💡 创新点**

创新点在于将联合检测视为最大后验（MAP）解码，并推导出考虑系统干扰、量化间隔和码字权重分布的下限和上限界限。

**🔧 技术方法**

使用了卡尔曼滤波、最大后验解码和无限状态马尔可夫链等技术。

**📊 数据集**

使用了（64,16）极化码和16位CRC作为数据集进行仿真。

**📈 对比分析**

与最大似然（ML）解码相比，MAP解码在块错误率为10^-3时表现出约3.0dB的性能增益，且在高信噪比（SNR）区域，MAP性能与上限界限一致。

**⚠️ 局限性**

限制在于该方法在高SNR和低系统干扰的假设下推导的界限，可能在实际应用中受到其他因素的影响。

---

## 690. SAD: A Large-Scale Strategic Argumentative Dialogue Dataset

**arXiv ID:** 2601.07423 | [PDF](https://arxiv.org/pdf/2601.07423v1)

**作者:** Yongkang Liu `[一作]` (Northeastern University), Hinrich Schütze `[通讯]` (CIS, LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个大规模多轮论辩对话数据集 SAD，并在其上提出了策略条件生成任务。

**💡 创新点**

首次将论辩策略标签与多轮对话结合，提供真实交互式数据和可控生成范式。

**🔧 技术方法**

使用大语言模型（Llama3.1-8B、Qwen3-8B 等）以及基于 GPT‑4.1 的自动评估器，并进行 SFT/DPO 微调。

**📊 数据集**

基于 Reddit 的 ChangeMyView 论坛，经过筛选共 392,822 例对话，包含 5 种策略标签。

**📈 对比分析**

对比多种开源/闭源 LLM，加入策略提示后在相关性、连贯性、流畅性和说服力等指标均有提升，尤其微调后效果显著。

**⚠️ 局限性**

数据来源为公开论坛，可能带有偏见；评估对提示敏感；模型对说服力提升有限。

---

## 691. Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning

**arXiv ID:** 2601.07408 | [PDF](https://arxiv.org/pdf/2601.07408v1)

**作者:** Ziheng Li `[一作]` (Fudan University), Hongcheng Guo `[通讯]` (Fudan University)

**通讯引用:** 260 | [OpenAlex ID](https://openalex.org/A5073687083)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Outcome-grounded Advantage Reshaping (OAR) 的方法，用于改进 Group Relative Policy Optimization (GRPO) 在长序列推理任务中的信用分配。

**💡 创新点**

创新点在于将奖励的信用细粒度化到每个 token，依据 token 对最终答案分布的影响进行归因，并采用双层优势重分配机制来抑制低影响 token、放大高影响 token，同时保持优势总量不变。

**🔧 技术方法**

主要技术包括：
- OAR-P：通过对每个 token 进行遮蔽并计算最终答案分布的 KL 散度来得到归因分数；
- OAR-G：使用单次反向传播的梯度×输入近似 OAR-P，计算 token 对最终答案分布的敏感度；
- 双层优势重分配（Bi-Level Advantage Reshaping）对 token 重要性进行阈值筛选并重新归一化；
- 集成至 GRPO 的 PPO 目标中。

**📊 数据集**

使用的基准数据集包括 AIME2024/2025、AMC23、MATH500、GSM8K 等数学推理 benchmark，模型基座为 Qwen2.5-7B 与 Qwen2.5-Math-7B（以及 1.5B 版本的 ablation）。

**📈 对比分析**

与 vanilla GRPO、随机信用分配、基于熵的信用分配以及 KTAE 等基线相比，OAR-P 取得了最高的性能提升，OAR-G 以接近 OAR-P 的效果同时几乎不增加计算开销。实验结果显示在多项任务上 Pass@1/Pass@k 均提升 2-3% 甚至更高，且训练收敛更快、熵不易坍塌。

**⚠️ 局限性**

局限性包括：
- OAR 的重要性估计依赖模型自身的答案分布，若分布变化与判别器奖励弱耦合，归因可能不准确；
- OAR-P 的遮蔽操作虽然能近似因果效应，但会引入分布偏移且计算成本高；
- OAR-G 仅是第一阶近似，精度低于 OAR-P；
- 目前验证集中于可判定的数学推理任务，对开放式或交互式场景的推广仍需研究。

---

## 692. Computing patient similarity based on unstructured clinical notes

**arXiv ID:** 2601.07385 | [PDF](https://arxiv.org/pdf/2601.07385v1)

**作者:** Petr Zelina `[一作]` (Masaryk University), Vít Nováček `[通讯]` (Masaryk University)

**通讯引用:** 949 | [OpenAlex ID](https://openalex.org/A5102905135)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究提出了一套基于临床笔记的患者相似度计算框架，将患者的所有笔记通过分段过滤后映射为向量矩阵，并对矩阵进行相似度评估。

**💡 创新点**

创新点在于把患者视为一整个嵌入矩阵，结合分段标题预测过滤、三种矩阵相似度方法（RV系数、MaxMax、Edit‑Distance）以及多种文本嵌入技术，并在十个临床相似度类别上进行系统验证。

**🔧 技术方法**

使用了LSA、Doc2Vec（PV‑DM）、Transformer（RobeCzech）三种文本嵌入方法，BERT标题预测过滤器，以及RV系数、MaxMax和Edit‑Distance三种矩阵相似度计算。

**📊 数据集**

实验基于捷克乳腺癌患者的4,267份匿名临床笔记（共152,552条记录），含XML元数据，平均每位患者约36条笔记。

**📈 对比分析**

通过与三位临床专家在10个相似度类别上标注的Kendall τ相关性评估，最优组合（嵌入组合+RV系数）在治疗相关类别的平均τ约为0.35-0.40，过滤步骤显著提升性能，整体表现优于单一嵌入或矩阵相似度方法。

**⚠️ 局限性**

局限性包括部分相似度类别信息稀缺导致注释者一致性低，评估样本仅为5个对比患者，统计显著性有限，模型未在验证数据上进行监督学习，且对低段落计数类别的处理仍不理想。

---

## 693. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

**arXiv ID:** 2601.07372 | [PDF](https://arxiv.org/pdf/2601.07372v1)

**作者:** Xin Cheng `[一作]` (Peking University), Wenfeng Liang `[通讯]` (DeepSeek-AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了 Engram 模块——一种基于哈希化 N‑gram 检索的条件记忆单元，能够与 Mixture‑of‑Experts 并行使用，提升大模型的知识检索与推理效率。

**💡 创新点**

创新点在于将条件记忆视为稀疏性的另一维度，结合 Tokenizer 压缩、多头哈希、上下文门控与多分支融合，构造出与传统 MoE 互补的 U‑形稀疏分配定律。

**🔧 技术方法**

所用技术包括哈希检索、上下文门控、深度可分离卷积、multi‑branch Transformer、FP8 乘法、异步 Prefetch、CPU/DRAM offload 等。

**📊 数据集**

实验数据集涵盖 The Pile 训练语料，以及 MMLU、CMMLU、ARC、TriviaQA、RACE、HumanEval、MATH、LongPPL、RULER 等多任务评测。

**📈 对比分析**

通过与等参数/等 FLOPs 的 MoE‑27B 和 Dense‑4B 进行对比，Engram‑27B 在知识、推理、代码等任务上平均提升 3–5%，Engram‑40B 在多数任务进一步下降；在长上下文任务（LongPPL、RULER）上显著优于基线。

**⚠️ 局限性**

局限性包括 Engram‑40B 仍未完全收敛、对超长尾 N‑gram 冲突与稀疏度调节尚不完善、实现高度依赖特定硬件与分布式环境。

---

## 694. Interpretable Text Classification Applied to the Detection of LLM-generated Creative Writing

**arXiv ID:** 2601.07368 | [PDF](https://arxiv.org/pdf/2601.07368v1)

**作者:** Minerva Suvanto `[一作]` (Chalmers University of Technology), Peter J Barclay `[通讯]` (Edinburgh Napier University)

**通讯引用:** 389 | [OpenAlex ID](https://openalex.org/A5026153887)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何用机器学习方法区分经典侦探小说中的人类创作文本与LLM生成文本，并进一步解释了高分类准确率的原因。

**💡 创新点**

创新点在于采用可解释的线性分类器，对单词特征进行手工注释，揭示了LLM生成文本在同义词使用、时间漂移、美国式词汇以及外语/口语表达上的差异，从而解释了模型高准确率。

**🔧 技术方法**

技术主要包括基于词袋（unigram）特征的线性分类器、朴素贝叶斯、SVM、逻辑回归与随机森林；并使用pybiber提取语言学特征、Shannon熵分析以及可视化工具对模型进行解释。

**📊 数据集**

数据集为DETECT12，由六部阿加莎·克里斯蒂和六部多萝西·L·萨耶斯的经典侦探小说拆分为约8,068个约100词左右的短段落，随后用GPT‑4.1在保持意义与长度的前提下重写，得到等量的AI文本。

**📈 对比分析**

与人类评估者（≈50%准确率）对比，线性模型在测试集上达0.9814的准确率；其余模型亦保持95%以上。通过逐特征剔除实验表明，前100个高频特征对准确率影响最大，去除后准确率降至≈0.88。

**⚠️ 局限性**

限制包括：生成方式依赖重写提示，可能人为放大词汇多样性；只评估单一文学体裁和单一LLM；且手工注释仅覆盖少量高频特征，未覆盖全部语料。

---

## 695. FOCAL: A Novel Benchmarking Technique for Multi-modal Agents

**arXiv ID:** 2601.07367 | [PDF](https://arxiv.org/pdf/2601.07367v1)

**作者:** Aditya Choudhary `[一作]` (Sprinklr AI), Anupam Purwar `[通讯]` (Sprinklr AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FOCAL 框架，对多模态（语音+文本）智能体的端到端推理、错误传播以及语义与推理能力进行系统评估；并在购物支持代理上进行了实验验证。

**💡 创新点**

创新点包括：①引入 Reasoning 和 Semantic 两种新度量，用以量化代理的推理深度和对话语义质量；②设计全流程评估管道（Human‑Simulator、TTS、ASR、LLM Judge），能够同时进行自动化与人工参与的评测；③提出多维语音质量指标（音质、声纹相似度、声纹一致性）和 MOS 的合成估计；④通过上下文相似度补偿 WER 的缺陷，突出语义保持。

**🔧 技术方法**

使用的技术包括：SOTA TTS（支持声纹克隆）、Whisper‑style ASR、RAG 语义检索的购物代理、LLM Judge（对话内容评估）、speaker embedding（Wespeaker）用于声纹相似度、VoiceMOS（T05）用于 MOS 预测、Python + Docker 构建的可视化 UI。

**📊 数据集**

数据集：购物知识库（订单跟踪、店铺定位等场景数据）、随机生成的用户人格与语料、TTS 合成语音、ASR 录音，主要为内部构造并未公开；对话日志用于 Ground‑Truth 与 Implementation transcript 对比。

**📈 对比分析**

在六类客户旅程（如订单跟踪、退货、付款问题等）中，Reasoning 评分 6–10，Semantic 评分 7–9，Tool‑Calling 完全正确；WER 0.05–0.18，语义相似度 0.88–0.95，MOS 2.4–3.4，声纹一致性 0.67–0.78。相比传统仅评估问答正确率的基准，FOCAL 能更细粒度捕捉推理、语义和语音质量的误差传播。

**⚠️ 局限性**

局限性：①评测主要聚焦于语音‑语音管道，对多模态混合输入（如同时出现文本和语音）支持有限；②MOS 采用合成估计，缺乏真实人类评测；③未对安全性、对抗鲁棒性做深入测试；④框架实现复杂，需要多组件的协同部署，难以快速落地。

---

## 696. Large-Scale Autonomous Gas Monitoring for Volcanic Environments: A Legged Robot on Mount Etna

**arXiv ID:** 2601.07362 | [PDF](https://arxiv.org/pdf/2601.07362v1)

**作者:** Julia Richter `[一作]` (Robotic Systems Lab), Marco Hutter `[通讯]` (Robotic Systems Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发并现场部署了一套在埃特纳火山执行全自主火山气体监测的四足机器人系统，完成了三次全自主任务和一次遥控测量，首次实现地面近源气体分析。

**💡 创新点**

①将四极质量谱仪与腿形机器人平台集成，实现近地气体采样；②构建端到端自治栈（全球规划、局部导航、SLAM+GNSS融合、适应性本地化与感知），首次在活火山现场完成全自主气体监测；③在机器人上实现了实时气体谱测量和数据可视化。

**🔧 技术方法**

硬件：ANYmal四足机器人、INFICON Transpector MPH质量谱仪、6D RealSense相机、Velodyne VLP‑16 LiDAR、Swift Piksi Multi GNSS；软件：ROS自治栈、HF 融合框架、X‑ICP LiDAR‑SLAM、A*全局规划、RMP 本地规划、CNN 通过率估计、GPU 加速的 Elevation Mapping、RL 训练的基底步态控制。

**📊 数据集**

使用现场采集的火山地形与气体数据；OpenStreetMap 路网和10–30 m分辨率数字高程模型做路径规划；少量人工标注的地形样本训练穿越性CNN；无人机航拍图像与雷达测绘做场景可视化。

**📈 对比分析**

通过三次全自主任务评估：平均自主率 96.4 %，干预率 3.6 %；气体源检测成功率 5/8（≈62 %）。遥控任务中成功检测到 SO₂ 峰值，CO₂ 也得到确认。与传统 UAV 或固定传感器相比，机器人可在源近距离采样，覆盖更大区域且在高坡度、松散地表保持稳定。

**⚠️ 局限性**

局限：①气体源检测受风速与 plume 分散影响，导致部分源未被捕获；②定位误差受 OpenStreetMap 分辨率与 LiDAR 匹配不稳定影响，出现路径与实际地形错位；③本地规划在细沙、极细颗粒下易受阻；④入口高度约 0.75 m，低浓度气体采样受限；⑤穿越性估计缺乏实时反馈；⑥在极细沙、陡坡时仍难以实现完全自主。

---

## 697. Beyond Hard Masks: Progressive Token Evolution for Diffusion Language Models

**arXiv ID:** 2601.07351 | [PDF](https://arxiv.org/pdf/2601.07351v1)

**作者:** Linhao Zhong `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University)

**通讯引用:** 68011 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EvoToken-DLM，通过软 token 分布代替硬掩码，实现逐步可逆的迭代解码。

**💡 创新点**

将软 token 进化状态与连续轨迹监督相结合，提升推理质量并保持与 KV‑caching、块级扩散的兼容性。

**🔧 技术方法**

连续空间软分布、连续轨迹监督、KV‑caching 兼容、块级扩散架构。

**📊 数据集**

使用 S1K 数据集进行微调，并在 Countdown、GSM8K、MATH500、SVAMP 等算数/推理基准上评测。

**📈 对比分析**

与原 LLaDA‑Instruct‑8B 及二进制掩码 MDLM 进行对比，平均在各基准上提升 2–17% 以上，且推理延迟仅略增。

**⚠️ 局限性**

对基于 AR 预训练的模型收敛困难，需更长训练时间和更高资源投入。

---

## 698. PLANET v2.0: A comprehensive Protein-Ligand Affinity Prediction Model Based on Mixture Density Network

**arXiv ID:** 2601.07415 | [PDF](https://arxiv.org/pdf/2601.07415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 699. Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation

**arXiv ID:** 2601.07338 | [PDF](https://arxiv.org/pdf/2601.07338v1)

**作者:** Yanzhi Tian `[一作]` (Beijing Institute of Technology), Yuhang Guo `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5101786500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了针对非字面翻译的Meta‑Evaluation数据集MENT，并提出了可反射式的Agentic翻译评估框架RATE，结合核心Agent与检索、评估、校准子Agent动态评估翻译质量。

**💡 创新点**

首次聚焦非字面翻译评估，提出通过核心Agent反射式调用检索与比较子Agent来克服LLM知识截止与分数不一致问题，显著提升评估准确性。

**🔧 技术方法**

使用大型语言模型作为评估与检索核心，OODA循环控制核心Agent，配合Search、Evaluation、Comparison三类子Agent，以及外部搜索引擎和对比校准技术。

**📊 数据集**

使用从社交网络、跨文化、诗歌、文学四个领域挑选的句子构成MENT数据集，包含7,530条人类评注的翻译质量分数。

**📈 对比分析**

与传统基于参考、无参考QE指标及LLM-as-a-Judge方法对比，RATE在MENT上Meta‑Score提升至少3.2点，且在WMT23 En‑De通用数据集上表现与现有最优指标相当。

**⚠️ 局限性**

核心Agent缺乏对外部工具执行失败（API超时、网络错误）的诊断与重试机制，导致在外部服务不稳定时评估效率受限。

---

## 700. How to predict creativity ratings from written narratives: A comparison of co-occurrence and textual forma mentis networks

**arXiv ID:** 2601.07327 | [PDF](https://arxiv.org/pdf/2601.07327v1)

**作者:** Roberto Passaro `[一作]` (University of Trento), Massimo Stella `[通讯]` (University of Trento)

**通讯引用:** 2092 | [OpenAlex ID](https://openalex.org/A5074066409)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向短篇创意文本的两种语义网络构建方法（词共现网络和文本 forma mentis 网络）并给出完整教程。

**💡 创新点**

创新在于系统比较这两种网络在预测人类创意评分上的表现，并证明依赖句法的 TFMN 在预测上更优。

**🔧 技术方法**

采用 spaCy、NetworkX、EmoAtlas 等库构建网络，提取结构指标、传播激活值和情感特征，使用 10 种机器学习回归器。

**📊 数据集**

数据集为 1029 篇 4-6 句创意故事，包含三词提示并由 4 名评审给出 1-5 评分。

**📈 对比分析**

通过交叉验证和置换基线比较，TFMN 在所有模型下均取得最低 MAE 约 0.58、最高 Spearman 0.64，词共现网络无明显优势，情感特征仅有细微提升。

**⚠️ 局限性**

限制在于仅涉及短文本、仅使用传统结构特征、不同评审方案可能导致结果变化，未考虑更复杂的图神经网络或长文本。

---

## 701. SDHSI-Net: Learning Better Representations for Hyperspectral Images via Self-Distillation

**arXiv ID:** 2601.07416 | [PDF](https://arxiv.org/pdf/2601.07416v1)

**作者:** Prachet Dev Singh `[一作]` (LNMIIT), Biplab Banerjee `[通讯]` (IIT Bombay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于自蒸馏的轻量级网络 SDHSI-Net，用于高光谱图像分类。

**💡 创新点**

创新点在于结合多级自蒸馏和三元组损失，利用内部教师-学生结构提升特征分离性，且不依赖外部教师网络。

**🔧 技术方法**

采用 3D-2D 卷积骨干、内部自蒸馏（logit 与 hint 损失）、三元组损失、PCA 降维以及 DropBlock 正则化。

**📊 数据集**

在 Indian Pines 和 Salinas 两个标准高光谱数据集上进行实验。

**📈 对比分析**

与多种 SOTA 方法对比，教师头在 Indian Pines 上 OA 达到 99.00%，与 SSRN 相当；学生头在保持参数大幅减少的同时，准确率超过 99% 或 98% 等，表现优异。

**⚠️ 局限性**

局限性包括对训练数据分割敏感、对补丁大小依赖、仅在分类任务验证，未扩展到变化检测等更复杂任务。

---

## 702. Novel Decoding Algorithm for Noiseless Non-Adaptive Group Testing

**arXiv ID:** 2601.07388 | [PDF](https://arxiv.org/pdf/2601.07388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 703. The Practicality of Normalizing Flow Test-Time Training in Bayesian Inference for Agent-Based Models

**arXiv ID:** 2601.07413 | [PDF](https://arxiv.org/pdf/2601.07413v1)

**作者:** Junyao Zhang `[一作]` (University of Birmingham), Junqi Tang `[通讯]` (University of Birmingham)

**通讯引用:** 1033 | [OpenAlex ID](https://openalex.org/A5054394663)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究如何在代理模型的参数后验估计中，通过测试时训练对预训练的正则化流进行微调，以适应分布迁移。

**💡 创新点**

提出梯度子空间驱动的测试时训练方法（GradSubspace‑TTT和GradSubspace‑PEA），能在保持参数效率的同时实现针对目标分布的精确校正。

**🔧 技术方法**

使用神经后验估计（NPE）配合正则化流、LoRA低秩适配、梯度子空间投影和重新参数化。

**📊 数据集**

使用 Brock‑Hommes 代理模型在 β=120 与 β=60 两种情境下生成的合成时序数据。

**📈 对比分析**

与原始 SNPE、SNPE‑TTT、SNPE‑LoRA 进行对比，利用 Wasserstein 距离和 MMD 评价后验一致性；梯度子空间方法在两种设定下均实现最低误差，性能优于 LoRA，甚至在参数更少时与全参数微调相当。

**⚠️ 局限性**

依赖于仿真器的可访问性，梯度子空间仅在预训练点附近有效，需手动设定子空间维度与批次大小，且实验仅验证于单一代理模型，泛化性待进一步评估。

---

## 704. On the universal definition of intelligence

**arXiv ID:** 2601.07364 | [PDF](https://arxiv.org/pdf/2601.07364v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 705. Recommendation-as-Experience: A framework for context-sensitive adaptation in conversational recommender systems

**arXiv ID:** 2601.07401 | [PDF](https://arxiv.org/pdf/2601.07401v1)

**作者:** Raj Mahmud `[一作]` (University of Technology Sydney), A. Baki Kocaballi `[通讯]` (Macquarie University)

**通讯引用:** 3645 | [OpenAlex ID](https://openalex.org/A5036989189)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并开展了十个领域的情境问卷实验，收集了168名受试者对教育性、探索性、情感性三大交互目标在不同领域、价值情境与用户特征下的优先级，并以此为基础提出了Recommendation-as-Experience（RAE）适配框架。

**💡 创新点**

将推荐过程视作经验化交互，将用户交互目标视为可调节的状态变量，并构建了基于领域、价值、用户特征与主动权的分层状态–策略映射，突破了传统仅关注排序精度的CRS设计。

**🔧 技术方法**

采用问卷情境实验、贝叶斯层级有序回归、Kruskal–Wallis、Wilcoxon等非参数检验，对实验数据进行统计分析，并给出权重映射规则；框架可在规则、学习或LLM驱动的对话策略中实现。

**📊 数据集**

使用了来自168名受试者的十领域（教育、金融、住房、旅游、服饰、娱乐、美容、技术、餐饮、消费）情境问卷数据集。

**📈 对比分析**

通过统计检验与贝叶斯置信区间验证假设，结果表明领域、价值、经验等因素显著影响交互目标优先级；框架本身不包含性能指标，主要为对话策略改进提供依据。

**⚠️ 局限性**

局限性包括：仅采用一次性问卷数据，缺乏纵向或真实系统交互验证；用户属性测量有限，未纳入更细粒度的心理特征；框架尚未在实际CRS中进行实验评估。

---

## 706. MCP-ITP: An Automated Framework for Implicit Tool Poisoning in MCP

**arXiv ID:** 2601.07395 | [PDF](https://arxiv.org/pdf/2601.07395v1)

**作者:** Ruiqi Li `[一作]` (University of Science and Technology of China), Xiang-Yang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18112 | [OpenAlex ID](https://openalex.org/A5100341802)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自动化框架MCP‑ITP，用来在Model Context Protocol（MCP）生态中实现隐式工具中毒攻击；

**💡 创新点**

创新点在于将隐式工具中毒定义为黑盒优化问题，并通过迭代反馈（评估LLM与检测LLM）来同时最大化攻击成功率并最小化被检测概率；

**🔧 技术方法**

采用LLM驱动的生成与优化（L_A）、LLM检测器（L_D）和LLM评估器（L_E）构成的三元循环，使用树结构搜索与候选集合剪枝实现迭代优化；

**📊 数据集**

使用公开的MCPTox数据集（45台真实MCP服务器、353个工具，共1497个案例，548个隐式中毒案例）进行实验；

**📈 对比分析**

与手工构造的对照基线相比，MCP‑ITP在12种LLM代理上平均提高ASR（如GPT‑3.5‑turbo从48.2%提升至84.2%），且MDR显著下降（如Qwen3‑14b从22%降至0.3%）；

**⚠️ 局限性**

局限性包括：对强大检测LLM的抵抗力仍有限；实验仅覆盖MCPTox数据集，未扩展到更广泛的场景；缺乏理论证明，仅靠经验评估。

---

## 707. CompNO: A Novel Foundation Model approach for solving Partial Differential Equations

**arXiv ID:** 2601.07384 | [PDF](https://arxiv.org/pdf/2601.07384v1)

**作者:** Hamda Hmida `[一作]` (Mines Paris), Youssef Mesri `[通讯]` (Mines Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种组合式神经算子框架（CompNO），通过预训练基础算子（Convection、Diffusion、Inviscid Burgers）构建算子库，然后用轻量化适配块和边界条件算子组装成针对不同参数化 PDE 的求解器。

**💡 创新点**

创新点在于：① 预训练独立的基础算子而非单一大模型，提升数据效率与可解释性；② 通过适配块实现few-shot组合，支持不同物理耦合；③ 在推理阶段硬性满足 Dirichlet 边界，消除边界误差；④ 结构模块化，易扩展至更多 PDE。

**🔧 技术方法**

采用 Parametric Fourier Neural Operator（PFNO）作为基础算子，使用 MLP/线性聚合层与边界修正算法；损失函数为 MSE/MAE，优化器为 Adam；实现基于 PyTorch。

**📊 数据集**

使用 PDEBench 的 1‑D 数据集：convection、diffusion、non‑linear convection（无粘性 Burgers）、viscous Burgers 以及 convection‑diffusion，初始条件为两波正弦叠加，生成 8000 条训练样本和 2000 条测试样本。

**📈 对比分析**

与 PFNO、PDEFormer‑FS 及 ICL‑based 模型在相同任务上比较，评价指标为相对 L² 误差、MAE 以及边界误差。CompNO 在线性参数化系统（convection、diffusion、convection‑diffusion）中取得最低误差，在 Burgers（低粘度）下表现与基线相当且更稳定，整体展示了较强的泛化能力。

**⚠️ 局限性**

局限性包括：仅在 1‑D 任务验证，无法直接评估多维复杂几何；聚合层表达能力有限，难以捕捉强非线性或湍流耦合；扩展至更高维 PDE 与更复杂边界条件仍需进一步研究。

---

## 708. Fast and Provable Nonconvex Robust Matrix Completion

**arXiv ID:** 2601.07355 | [PDF](https://arxiv.org/pdf/2601.07355v1)

**作者:** Yichen Fu `[一作]` (Fudan University), Ke Wei `[通讯]` (Fudan University)

**通讯引用:** 45271 | [OpenAlex ID](https://openalex.org/A5100325379)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的非凸鲁棒矩阵补全算法 ARMC，利用子空间投影提升 SVP 方法的效率，并在存在稀疏离群值和随机噪声的情况下实现了线性收敛。

**💡 创新点**

创新点包括：①在低秩更新时加入切空间投影，既降低了计算复杂度，又不需要显式的相干性投影或样本分割；②采用 leave‑one‑out 技术给出在同时出现离群值与噪声时的理论恢复保证；③改进了样本复杂度与稀疏度上界，比传统凸方法和已有非凸方法更宽松。

**🔧 技术方法**

技术手段主要有：子空间（切空间）投影、奇异值阈值化（SVP）、软/硬阈值化估计稀疏项、leave‑one‑out 解析、子高斯噪声的矩阵集中不等式以及奇异向量的相干性约束。

**📊 数据集**

实验使用了：①不同尺寸、秩、条件数和稀疏度的合成数据；②VIRAT 数据集的 1907 帧视频（灰度 180×320）进行前景/背景分离。

**📈 对比分析**

与 RMC（无切空间投影的非凸方法）、RPCA‑GD（快速梯度鲁棒 PCA）以及凸 RMC 进行比较；结果显示 ARMC 在相同迭代次数下的每次迭代时间显著降低，总运行时间更短；在高条件数、较大稀疏度和噪声水平下的相位转移曲线更优，误差随噪声水平呈线性关系。

**⚠️ 局限性**

局限性：①理论分析依赖矩阵相干性与条件数假设；②参数 β1、β2、γ 的选择仍需经验调优；③虽然每次迭代更快，但整体计算仍需 O(n²r) 的 SVD 计算；④实验规模相对有限，未验证在更大规模真实数据集上的鲁棒性。

---

## 709. Semantic Compression of LLM Instructions via Symbolic Metalanguages

**arXiv ID:** 2601.07354 | [PDF](https://arxiv.org/pdf/2601.07354v1)

**作者:** Ernst van Gassen `[一作]` `[通讯]` (Independent Researcher), Ernst van Gassen (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种名为MetaGlyph的符号指令语言，用数学符号压缩提示并评估其在不同规模模型上的可解释性与性能。

**💡 创新点**

创新点在于利用预训练中已学习的数学符号（如∈、⇒、∩）实现指令压缩，无需额外教学，且提供多模型、多任务的实证比较。

**🔧 技术方法**

采用符号指令语言设计、Prompt压缩实验、语义相等性、操作符保真度以及Token压缩率等评估指标。

**📊 数据集**

主要使用四类任务（选择、提取、约束组合、条件变换）以及八个公开或API模型作为实验对象，未使用专有数据集。

**📈 对比分析**

通过将自然语言提示、MetaGlyph符号提示与无意义符号控制进行对比，发现MetaGlyph在选择任务可达75%语义相等，Token压缩率62-81%，不同模型的符号保真度差异显著。

**⚠️ 局限性**

限制包括模型规模与指令调优偏差导致的U形曲线、符号保真度不稳定、对复杂多约束任务的解析失败，以及缺乏大规模、跨模型的系统评估。

---

## 710. TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees

**arXiv ID:** 2601.07353 | [PDF](https://arxiv.org/pdf/2601.07353v1)

**作者:** Tianyu Liu `[一作]` (University of Science and Technology of China), Xiaoyan Sun `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4180 | [OpenAlex ID](https://openalex.org/A5100656317)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关、预算驱动的自适应树扩展框架，用于改进树式投机式解码的推理速度。

**💡 创新点**

核心创新是把固定宽度/深度的树结构改为基于预算的动态扩展，结合鲁棒树初始化和置信度门控，使树形在确定性场景变深窄、在不确定性场景变宽浅，从而在保持输出质量的前提下显著提升速度。

**🔧 技术方法**

利用树式投机式解码（Speculative Decoding）框架，采用Top‑K初始化、置信度门控（confidence‑gated）动态扩展、树注意力（Tree Attention）验证；并通过预算驱动的层级迭代构造草稿树。

**📊 数据集**

在六个基准上评估：MT‑Bench、Alpaca、GSM8K、HumanEval、CNN/DM 与 QA，使用 Llama‑3.1‑8B‑Instruct、Qwen3‑8B/32B、DeepSeek‑R1‑Distill‑LLaMA‑8B、Vicuna‑13B 等模型。

**📈 对比分析**

与当前最强树式投机式解码方法（-3）对比，实验显示在所有 8 个模型、6 个数据集上均实现更高的吞吐量；在 HumanEval 可达 5.16×，在 CNN/DM 最高 2.30×，在推理与编程类任务上提升尤为显著，整体加速幅度远超基线。

**⚠️ 局限性**

主要局限在大批量推理下的可扩展性和超参数泛化：目前仅在单样本延迟场景验证，批量化时 GPU 计算饱和与树结构管理带来的开销；另外，阈值 μ 与预算 N 的固定设置可能需要针对特定任务手工调优。

---

## 711. Reward Modeling from Natural Language Human Feedback

**arXiv ID:** 2601.07349 | [PDF](https://arxiv.org/pdf/2601.07349v1)

**作者:** Zongqi Wang `[一作]` (Alibaba Group), Yongbin Li `[通讯]` (Alibaba Group)

**通讯引用:** 1729 | [OpenAlex ID](https://openalex.org/A5100644428)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了将自然语言人类反馈作为过程奖励，改进生成式奖励模型（GRM）对偏好任务的学习；

**💡 创新点**

首次揭示并解决了GRM在二元偏好任务中出现的结果-过程不一致问题，并通过人类批注的相似度引入过程奖励；

**🔧 技术方法**

使用强化学习与可验证奖励（RLVR）、GRPO框架、F1相似度作为过程奖励、Meta Reward Model（MetaRM）以及在线自适应MetaRM；

**📊 数据集**

在HelpSteer3等多任务基准（HelpSteer3、SCAN-HPD、HREF、LitBench、WQ_Arena、WPB）以及下游评测数据集（MATH500、HumanEval+、Arena-Hard-V2.0）上进行实验；

**📈 对比分析**

与基线GRM、标量奖励模型及同等规模的专用GRM比较，RM‑NLHF在多数基准上均优于仅使用结果奖励的模型，且在下游任务中表现更好；

**⚠️ 局限性**

依赖有限的人类批注，虽然MetaRM可减少标注成本，但在数据分布漂移或与基线训练集差异较大的情况下仍可能出现奖励误差。

---

## 712. Controlled Self-Evolution for Algorithmic Code Optimization

**arXiv ID:** 2601.07348 | [PDF](https://arxiv.org/pdf/2601.07348v1)

**作者:** Tu Hu `[一作]` (Nanjing University), Huacan Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Controlled Self‑Evolution（CSE）的框架，用于在有限的搜索预算下通过迭代的“生成‑验证‑改进”循环自动改进代码实现的时间与空间效率。

**💡 创新点**

创新点在于三方面：① 采用多样化的规划初始化生成结构多元的候选方案；② 用基于反馈的遗传演化替代随机变异与交叉，实现细粒度的有指导性改进；③ 通过层级演化记忆在任务内外捕获成功与失败经验，实现经验复用与搜索引导。

**🔧 技术方法**

技术上结合了大型语言模型（LLM）的生成与自我反思、功能分解驱动的有指导变异与结构化交叉，以及本地与全局记忆的检索与压缩机制。

**📊 数据集**

使用EffiBench‑X数据集，涵盖623道算法竞赛题目，测试两种语言Python和C++。

**📈 对比分析**

与Direct、Self‑Reflection、SE‑Agent、AlphaEvolve等基线方法比较，CSE在执行时间、峰值内存、内存积分等效率指标上均优于对手，且在早期代数和后期持续进化上表现更佳。

**⚠️ 局限性**

局限性在于尚未将迭代优化的经验融入基础模型的训练，未探索将CSE演化轨迹转化为强化学习式的训练信号以提升模型自身的优化能力。

---

## 713. SEE: Signal Embedding Energy for Quantifying Noise Interference in Large Audio Language Models

**arXiv ID:** 2601.07331 | [PDF](https://arxiv.org/pdf/2601.07331v1)

**作者:** Yuanhe Zhang `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 4709 | [OpenAlex ID](https://openalex.org/A5036865453)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Signal Embedding Energy（SEE）度量 LALM 噪声干扰，并基于 SEE 开发无训练的 Signal Embedding Energy Neutralization（SEEN）噪声抑制方法。

**💡 创新点**

① 用模型内部嵌入子空间而非传统声学特征定义噪声度量；② 将 SEE 作为内部指标实现噪声的定量评估；③ 基于 SEE 在嵌入层直接去除噪声方向，显著提升推理质量。

**🔧 技术方法**

SVD 分解构建噪声子空间；投影与去噪（SEEN）；多模型评估（Qwen、MiniCPM、StepAudio）；传统频域与模型去噪器对比；相关性分析与 GSR 性能评估。

**📊 数据集**

MMAU、LibriSpeech 等多任务数据；噪声类型包括白噪声（Gauss）、Crowd、Machine、Traffic、Music、Sound 等；使用 SNR-10 dB、-2 dB 等设置。

**📈 对比分析**

与 STFT、WT、Segan、DFL 四种传统/模型去噪器对比；SEEN 在噪声场景下平均提升 6.7% 的 Generation Success Rate（GSR）；SEE 与 GSR 的 Pearson 相关系数达到 -0.96~ -1.00，说明 SEE 与性能高度相关。

**⚠️ 局限性**

需收集部署环境的纯噪声样本才能估计噪声子空间；均值池化可能低估时序噪声；过强抑制可能削弱语义信息；目前仅做推断，未用于训练或在高度可变噪声环境下的适用性有限。

---

## 714. Segmental Advantage Estimation: Enhancing PPO for Long-Context LLM Training

**arXiv ID:** 2601.07320 | [PDF](https://arxiv.org/pdf/2601.07320v1)

**作者:** Xue Gong `[一作]` (Tencent), Bo Zhou `[通讯]` (Tencent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Segmental Advantage Estimation (SAE)，在 RLVR 任务中将 PPO 的优势估计从 token 级切换为语义分段级，以降低 GAE 的偏差并提升训练稳定性和样本效率。

**💡 创新点**

创新点在于：①使用低概率 token 作为语义分段边界，自动将生成序列划分为信息量高的子段；②在仅对分段边界处做优势估计，显著减少噪声和误差累积；③提供理论证明和实验验证，说明分段可以在保持低偏差的同时不增加计算复杂度。

**🔧 技术方法**

技术：Proximal Policy Optimization (PPO)、Generalized Advantage Estimation (GAE)、Segmental Advantage Estimation (SAE)、概率阈值分段、值函数预训练、RLVR 评价框架。

**📊 数据集**

数据集：使用 Qwen3‑8B 基础模型，训练集为 DAPO‑Math‑17k，评测数据集包括 AIME'24、AIME'25、AMC、BeyondAIME，额外对代码生成和 STEM 领域进行验证。

**📈 对比分析**

与 GRPO、PPO（λ=1）和 PPO（adaptive λ）等基线比较，SAE 在四个测试集的平均分约 40.98，优于最佳基线 38.89，提升约 2.09%；在训练曲线中表现出更快的收敛速度、稳定的样本效率和更好的最终性能。

**⚠️ 局限性**

限制：分段策略仅基于简单的概率阈值，缺乏更复杂或自适应的分段方法；阈值 p 仍需手动设定，虽然表现稳健但可能在不同任务或模型规模下需调整；实验主要集中在数学与部分 STEM/代码领域，对其他长序列推理任务的通用性尚待验证。

---

## 715. OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single Image

**arXiv ID:** 2601.07333 | [PDF](https://arxiv.org/pdf/2601.07333v1)

**作者:** Tessa Pulli `[一作]` (Automation and Control Institute, TU Wien), Andreas Holzinger `[通讯]` (BOKU University)

**通讯引用:** 25216 | [OpenAlex ID](https://openalex.org/A5034657358)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了OSCAR——一种无训练、开放集CAD模型检索框架，利用单张RGB图像和自然语言提示从未标注的3D数据库中检索匹配模型，并将检索到的模型用于6D姿态估计。

**💡 创新点**

创新点包括：
1) 无需训练即可完成跨域3D模型检索；
2) 双模态两阶段检索策略：先用CLIP文本嵌入过滤候选，再用DINOv2图像嵌入精细匹配；
3) 自动化的onboarding流程——多视角渲染并用LLaVA自动生成描述；
4) 与MegaPose等姿态估计方法集成，替代耗时的在线重建，提升准确率与效率。

**🔧 技术方法**

使用的技术：CLIP（图像与文本嵌入）、DINOv2（自监督视觉嵌入）、GroundedSAM（语义分割提取ROI）、LLaVA（图像描述生成）、CAP3D渲染管线、MegaPose（姿态估计）、Nerfacto（重建基准）。

**📊 数据集**

实验数据集：MI3DOR（跨域3D模型检索基准），YCB‑V、Housecat6D、GSO、YCB‑V+GSO（用于姿态估计与检索）。

**📈 对比分析**

与MI3DOR上现有SOTA方法（如S2Mix、SC‑IFA、DLEA等）进行对比，OSCAR在所有指标（NN、FT、ST、F、DCG、ANMRR）上均超越对手；在6D姿态估计任务中，使用OSCAR检索的CAD模型在YCB‑V上相较于重建模型和GT模型分别提高了多达30‑40%的精度；并且在模型入库时间上，OSCAR比基于Nerfacto的重建快约31.8倍。

**⚠️ 局限性**

局限性：
1) 对渲染视角与语言描述的质量敏感，异常光照或遮挡时表现下降；
2) 对纹理相似但类别不同的对象可能产生误匹配；
3) 需要对每个CAD模型进行多视角渲染和caption生成，虽然速度快于重建，但在极大数据库规模下仍有算力与存储开销；
4) 目前主要针对单张RGB图，缺乏对多帧或视频输入的扩展。

---

## 716. Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations

**arXiv ID:** 2601.07422 | [PDF](https://arxiv.org/pdf/2601.07422v1)

**作者:** Wen Luo `[一作]` (Peking University), Houfeng Wang `[通讯]` (Peking University)

**通讯引用:** 5527 | [OpenAlex ID](https://openalex.org/A5025565222)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）内部真伪信号的产生机制，揭示了两条信息通路：问答依赖路径（Question-Anchored）和答案自足路径（Answer-Anchored），并利用此发现提升幻觉检测效果。

**💡 创新点**

创新点在于首次系统分离并验证真伪信号的两种独立路径，证明它们与知识边界及内部自我意识相关，并基于此提出了混合探针（MoP）和通路加权（PR）两种提升检测性能的方法。

**🔧 技术方法**

主要技术包括注意力消除（attention knockout）、令牌修补（token patching）、内部隐藏层探针（linear probing）、自我意识门控以及通路加权调节。

**📊 数据集**

实验使用了PopQA、TriviaQA、HotpotQA和Natural Questions四个问答数据集，并在12个不同规模与架构的LLM上进行评估。

**📈 对比分析**

与现有内部信号检测、基于不确定性和外部检索的方法相比，MoP和PR在AUC上平均提升约10%，在所有模型与数据集上均显著优于基线。

**⚠️ 局限性**

限制在于需要访问模型内部隐藏层，无法直接在纯黑盒环境下应用，且实验仅覆盖问答情境，未验证在其他生成任务中的通用性。

---

## 717. Peacock: UEFI Firmware Runtime Observability Layer for Detection and Response

**arXiv ID:** 2601.07402 | [PDF](https://arxiv.org/pdf/2601.07402v1)

**作者:** Hadar Cochavi Gorelik `[一作]` (Ben Gurion University of Negev), Yuval Elovici `[通讯]` (Ben Gurion University of Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

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

## 718. Forecast the Principal, Stabilize the Residual: Subspace-Aware Feature Caching for Efficient Diffusion Transformers

**arXiv ID:** 2601.07396 | [PDF](https://arxiv.org/pdf/2601.07396v1)

**作者:** Guantao Chen `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 13598 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于SVD的特征缓存框架SVD-Cache，用于加速Diffusion Transformer的推理。

**💡 创新点**

创新点在于将特征空间分解为主子空间和残差子空间，对主子空间使用EMA预测，对残差子空间直接复用，并通过一次性SVD实现跨prompt的基底重用。

**🔧 技术方法**

技术包括奇异值分解（SVD）、指数移动平均（EMA）预测、低秩近似、一次性SVD基底缓存以及特征缓存与重用。

**📊 数据集**

使用的基准数据集有DrawBench（文本到图像）和VBench（文本到视频），以及FLUX.1-dev、HunyuanVideo等模型。

**📈 对比分析**

与传统重用、TaylorSeer、FoCa等预测方法对比，SVD-Cache在FLUX.1-dev、HunyuanVideo等上实现了5.55×–29.01×的加速，且在ImageReward、CLIP、VBench等指标几乎无质量损失。

**⚠️ 局限性**

局限性包括：仍需一次性SVD预处理，对极大模型的SVD成本较高；对某些模型或任务的子空间稳定性验证有限；残差子空间直接复用可能在极端加速比例下出现累计误差。

---

## 719. OpenTinker: Separating Concerns in Agentic Reinforcement Learning

**arXiv ID:** 2601.07376 | [PDF](https://arxiv.org/pdf/2601.07376v1)

**作者:** Siqi Zhu `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7779 | [OpenAlex ID](https://openalex.org/A5003491365)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OpenTinker，一套开源的 RLaaS 框架，用于在分布式环境下训练 LLM 代理，并支持单机多租户调度。

**💡 创新点**

创新点在于将 agent、环境、交互协议抽象为可重用的一流对象，彻底解耦编程与执行，并通过协议协调器实现多代理协同训练与交互。

**🔧 技术方法**

核心技术包括 Ray 分布式调度、PPO/LoRA 训练后端、FSM 统一执行模型、异步流水线、以及基于环境的多代理协议协调器。

**📊 数据集**

使用的数据集涵盖 HuggingFace math、geometry、以及仿真 Gomoku 的离线与在线交互数据，并提供语言‑仅、视觉‑语言混合与多回合交互场景。

**📈 对比分析**

通过验证曲线与两代理 Gomoku 对抗奖励的对比，证明奖励传播、梯度优化与多回合/多代理执行均符合预期，性能稳定提升。

**⚠️ 局限性**

局限在于目前仅支持单节点部署，缺乏多节点集群编排、服务器侧训练/推理分离以及批量调度机制，导致大规模部署效率仍有提升空间。

---

## 720. HiVid-Narrator: Hierarchical Video Narrative Generation with Scene-Primed ASR-anchored Compression

**arXiv ID:** 2601.07366 | [PDF](https://arxiv.org/pdf/2601.07366v1)

**作者:** Haoxuan Li `[一作]` (Taobao and Tmall Group of Alibaba), Junjun Zheng `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了HiVid-Narrator框架，用层次化推理和SPA-Compressor压缩多模态令牌生成电商视频结构化叙述。

**💡 创新点**

创新在于构建双粒度E-HVC数据集、提出场景引导的ASR锚定压缩模块以及两阶段层次化训练方法。

**🔧 技术方法**

采用多模态LLM、视觉编码器SigLIP、Whisper ASR、交叉注意力融合、场景/事件查询提取以及Transformer解码器。

**📊 数据集**

使用E-HVC-146K训练集及E-HVC-Bench评测集，并与YouCook2、ActivityNet Captions公开数据集对比。

**📈 对比分析**

与现有模型相比，HiVid-Narrator在YouCook2 F1 30.5、ActivityNet METEOR 7.8、E-HVC-Bench SODA_c 14.48等指标显著提升，且压缩率达82.59%。

**⚠️ 局限性**

局限在于对高质量ASR和帧级描述的依赖、压缩过程中可能丢失细节、场景划分误差、仅针对电商视频且泛化性待验证。

---

## 721. Seeing Right but Saying Wrong: Inter- and Intra-Layer Refinement in MLLMs without Training

**arXiv ID:** 2601.07359 | [PDF](https://arxiv.org/pdf/2601.07359v1)

**作者:** Shezheng Song `[一作]` (National University of Defense Technology), Jie Yu `[通讯]` (National University of Defense Technology)

**通讯引用:** 4520 | [OpenAlex ID](https://openalex.org/A5006770280)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的解码细化方法DualPD，结合层间注意力对比与头级信息筛选，提升多模态大型语言模型的视觉推理准确率。

**💡 创新点**

创新点在于：①基于层间注意力差异动态选取“基础层”，通过对比该层与最终层的对数几率捕捉视觉理解的演化；②在每层中对注意力头进行重要性评分并软抑制低贡献头，减少视觉噪声。

**🔧 技术方法**

技术主要包括注意力分布的Hellinger距离计算、层间对数几率对比、头级重要性评估（L2范数）与软抑制（γ乘数）。

**📊 数据集**

使用六个公开视觉问答数据集（GQA、VQAv2、OKVQA、VizWiz、TextVQA、DocVQA）以及LLaVA和Qwen-VL两大模型系列进行评测。

**📈 对比分析**

与传统解码、Contrastive Decoding、DoLA等方法对比，DualPD在LLaVA和Qwen-VL模型上平均提升5–7个百分点，且仅需单模型推理，计算成本低。

**⚠️ 局限性**

局限性：仍依赖注意力分布的可靠性，对极端视觉模糊或头部分配不均的场景可能效果有限，且未探究多模型或多任务下的进一步泛化。

---

## 722. DiffER: Diffusion Entity-Relation Modeling for Reversal Curse in Diffusion Large Language Models

**arXiv ID:** 2601.07347 | [PDF](https://arxiv.org/pdf/2601.07347v1)

**作者:** Shaokai He `[一作]` (Chongqing University), Yu Tian `[通讯]` (Tsinghua University)

**通讯引用:** 10921 | [OpenAlex ID](https://openalex.org/A5015080274)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对 Diffusion 语言模型的“反转诅咒”问题，提出并验证了一套后训练框架 DiffER，以提升模型在双向关系推理上的表现。

**💡 创新点**

创新点包括：
1) 全实体掩码（Whole-Entity Masking）解决实体碎片化问题；
2) 对称对齐（Symmetric Alignment）通过构造正反向数据消除数据不平衡；
3) 逆关系建模（Inverse Relation Modeling）显式学习关系的互逆映射，增强逻辑推理能力。

**🔧 技术方法**

技术手段主要是：
- 采用 Diffusion 语言模型的去噪训练（discrete diffusion objective）；
- 在后训练阶段加入实体级别的掩码策略；
- 设计对称对齐和逆关系监督的训练数据；
- 通过交叉熵损失和联合训练目标实现多任务学习。

**📊 数据集**

使用的主要数据集为 PORE 基准数据集，包括父子（parent–child）和公司 CEO（company–ceo）两大关系类型，并基于这些数据构建正向、对称和逆向训练子集。

**📈 对比分析**

对比方法：在 LLaDA‑8B、Dream‑7B 等 Diffusion LLM 上进行基线与 DiffER 版本的精确匹配准确率比较。实验显示：
- 正向查询准确率从 92% 提升至 97–98%；
- 逆向查询从 46% 提升至 49–50%；
- 逻辑推理（关系逆向）同样得到显著提升。整体来看，DiffER 在保持正向性能的同时，显著缓解了反转诅咒。

**⚠️ 局限性**

局限性：
1) 数据构造规模化难度大，当前对称对齐与逆关系建模依赖结构化知识图谱，难以直接迁移至海量非结构化文本；
2) 仅在 8B/7B 级别模型上验证，缺乏对更大规模或不同 Diffusion 架构（如 MDLM）的系统评估。

---

## 723. PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis

**arXiv ID:** 2601.07344 | [PDF](https://arxiv.org/pdf/2601.07344v1)

**作者:** Jiao Xu `[一作]` (Dalian University of Technology), Ping Wang `[通讯]` (Peking University)

**通讯引用:** 49434 | [OpenAlex ID](https://openalex.org/A5100338632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PulseMind多模态诊断模型，包含大规模多模态诊断对话数据集MediScope、四维评测基准PulseMind Benchmark以及相对奖励的CRPO训练框架

**💡 创新点**

创新点在于整合真实多轮诊断对话与多模态影像的数据集、设计包含主动性、准确性、实用性和语言质量四维评估维度，并提出基于比较的强化学习方法CRPO

**🔧 技术方法**

使用Qwen2.5-VL系列基础模型，采用LoRA微调、CRPO强化学习、GPT‑4自动评测及多模型比较奖励机制

**📊 数据集**

使用自建MediScope数据集（98k对话、601.5k影像）以及多份公开医学问答数据

**📈 对比分析**

在PulseMind Benchmark上取得平均胜率76%，在11个公开医学QA基准上大多位居领先，且优于多种开源和专有模型

**⚠️ 局限性**

局限包括对3D/高维医学影像支持不足、训练对算力和时间要求高

---

## 724. On the Extremal Source Key Rates for Secure Storage over Graphs

**arXiv ID:** 2601.07340 | [PDF](https://arxiv.org/pdf/2601.07340v1)

**作者:** Zhou Li `[一作]` (Guangxi University), Zhou Li `[通讯]` (Guangxi University)

**通讯引用:** 26700 | [OpenAlex ID](https://openalex.org/A5100452309)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究基于图的安全存储码，探讨源密钥容量的极值，并给出对应图结构的完整刻画

**💡 创新点**

首次从极值视角阐明源密钥容量与图拓扑的关系，提出内部合格边是阻碍极值实现的关键因素；并完全表征无源密钥可实现的图结构

**🔧 技术方法**

信息理论安全约束、编码对齐（noise alignment）与符号对齐（symbol alignment）、随机线性编码、Schwartz–Zippel 论证

**📊 数据集**

论文中未使用公开数据集，所有结果均基于理论分析与抽象图模型

**📈 对比分析**

通过理论推导给出极值上界，并构造可实现极值的编码方案；实验验证未给出，性能评估以源密钥速率 R_Z=1/M 或 0 为界

**⚠️ 局限性**

仅适用于均匀源符号、单一/多符号边约束，且假设无孤立节点；对异质边约束或多源密钥情况缺乏分析

---

## 725. Thinking Before Constraining: A Unified Decoding Framework for Large Language Models

**arXiv ID:** 2601.07525 | [PDF](https://arxiv.org/pdf/2601.07525v1)

**作者:** Ngoc Trinh Hung Nguyen `[一作]` (Télécom Paris), Mehwish Alam `[通讯]` (Télécom Paris)

**通讯引用:** 1291 | [OpenAlex ID](https://openalex.org/A5009026163)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合解码框架In‑Writing，将自然生成与结构化生成结合，先自由推理后在触发词出现时切换到受限解码以输出规范化结果

**💡 创新点**

通过在推理过程中动态触发结构化约束，既保持了大模型的表达与推理能力，又保证了输出的可解析性，且仅需在模型内部实现，无需额外解析模型

**🔧 技术方法**

利用有限状态机（FSM）与正则表达式实现约束解码，并在触发词出现时切换状态；实现基于Open‑Source库（如Litelines/Outlines）的In‑Writing流程

**📊 数据集**

在多种任务上评估：推理类（GSM8K、Last Letter Concatenation、Shuffled Objects）与分类类（DDXPlus、MultiFin、Sports Understanding、NI‑Task280）等公开数据集

**📈 对比分析**

与传统两步“NL‑to‑Format”解析器对比，In‑Writing在大多数任务上实现约15–27%精度提升；在少量样本与大模型场景中更优，甚至在轻量级模型上也能保持性能，只需额外约10–20个Token，耗时约0.5–1.5秒

**⚠️ 局限性**

可能出现触发词过早生成导致过早切换约束、降低推理深度；或在自然生成阶段循环产生无意义文本；需要设计独特触发词、限制自然生成令牌数或精细调优提示

---

## 726. Judging Against the Reference: Uncovering Knowledge-Driven Failures in LLM-Judges on QA Evaluation

**arXiv ID:** 2601.07506 | [PDF](https://arxiv.org/pdf/2601.07506v1)

**作者:** Dongryeol Lee `[一作]` (Seoul National University), Kyomin Jung `[通讯]` (Seoul National University)

**通讯引用:** 3524 | [OpenAlex ID](https://openalex.org/A5077832834)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了当大语言模型（LLM）在问答评测中收到与自身知识冲突的参考答案时会发生的失效，并提出了可控的“swap‑reference”评测框架来系统化验证此现象。

**💡 创新点**

创新点在于：①首次量化并揭示了参考答案与模型内部知识冲突导致的判分不可靠；②设计了多种可控的实体交换策略（类型保持、类型更改、热门度差异、评估者知识一致）以产生评测对照；③通过对比原始与交换参考下的准确率与RPAG，深入分析了模型规模、知识流行度、知识新鲜度等因素对失效的影响。

**🔧 技术方法**

技术手段包括：基于GPT‑4o/4.1/5、Llama‑3.1/3.3、Qwen‑2.5/3系列的LLM判定；利用NER标注实体类型、构造多种交换参考；生成与参考一致/不一致的长文本候选答案；采用Accuracy、Reference‑Polarity Accuracy Gap (RPAG) 等评估指标；以及对不同提示策略（Direct、CoT、Self‑Consistency、CoT+Self‑Consistency）的对比实验。

**📊 数据集**

使用的数据集为 NaturalQuestions‑Open、PopQA、SciQ 和 FreshQA 四个问答基准。

**📈 对比分析**

通过比较原始参考与交换参考下的准确率（ACC⁰ vs ACCˢ）及RPAG，实验表明几乎所有评判模型在交换参考时准确率显著下降，甚至最强模型也无法恢复到原始水平，说明在知识冲突场景下LLM评判的鲁棒性有限。

**⚠️ 局限性**

局限性在于：仅聚焦问答任务，未对其它参考条件评估（如摘要、事实核查）展开；未给出有效的缓解方法，只是诊断问题；交换参考是人为构造，可能与真实评测场景不完全对应。

---

## 727. AntiPaSTO: Self-Supervised Steering of Moral Reasoning

**arXiv ID:** 2601.07473 | [PDF](https://arxiv.org/pdf/2601.07473v1)

**作者:** Michael J. Clark `[一作]` `[通讯]` (Independent Researcher), Michael J. Clark (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需偏好标签的梯度内部调节方法AntiPaSTO，用来在大型语言模型中控制道德价值观，尤其是诚实与不诚实的行为；

**💡 创新点**

创新点在于：①在SVD变换空间中学习旋转适配器，②使用不完整对比前缀作为自监督信号，③通过投影损失实现反平行方向的可逆控制，并加入总变差一致性与单调性约束；

**🔧 技术方法**

技术包括：SVD基础的适配器设计、Cayley参数化旋转、对比度前缀提取、投影损失、总变差一致性约束、单调性约束、梯度优化；

**📊 数据集**

使用了800对“诚实/不诚实”语义对的自监督数据作为训练集，在Gemma-3-1B上训练，并在DailyDilemmas（1,360个道德困境）和控制问题（数学正确性、颜色偏好）上进行评估；

**📈 对比分析**

与普通提示（prompting）和算术自监督方法RepEng相比，AntiPaSTO在Gemma-3-1B上在道德困境上的Steering F1提升了6.9倍；在更大模型（Gemma-3-12B、Qwen3-14B）上也能超越提示，但需要更多探索；

**⚠️ 局限性**

局限性包括：对随机种子高度敏感，导致性能波动；仅验证了诚实这一单一价值维度；在某些模型架构（如Llama-3.1-8B）上表现不佳；方法训练成本高于算术方法；对后训练安全策略的鲁棒性未知。

---

## 728. Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents

**arXiv ID:** 2601.07468 | [PDF](https://arxiv.org/pdf/2601.07468v1)

**作者:** Miao Su `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 Temporal Semantic Memory (TSM) 的长记忆框架，专为 LLM 代理在多会话、个性化对话中捕捉、聚合和检索时间相关语义信息而设计。

**💡 创新点**

创新点：
• 用语义时间线（而非对话时间）来组织记忆，解决时间不准确问题；
• 通过构建持续记忆（Durative Memory）将连续、语义相关的事件聚合成主题/人格摘要，避免时间碎片化；
• 在记忆检索时结合查询的时间意图进行过滤与重排序，提升时间一致性与回答准确性。

**🔧 技术方法**

核心技术：
• 语义时间知识图谱 (Temporal Knowledge Graph, TKG) 用实体‑关系‑时间三元组记录事件；
• Gaussian Mixture Model 对同一时间切片内的实体进行语义聚类，生成主题摘要；
• LLM 生成实体摘要、主题/人格摘要并进行向量嵌入；
• 基于密集检索 + 时间重排序的检索策略；
• 双向索引映射将摘要与原始聊天回合关联；
• 轻量级在线图更新 + “睡眠时间”聚合机制。

**📊 数据集**

数据集：
• LongMemEval_S（约 115k tokens 的对话历史，500 个问答）
• LoCoMo（1986 题，分为 5 类，长度 16k–26k tokens）。

**📈 对比分析**

评估方法：使用 GPT‑4o‑mini 与 Qwen3‑30B‑A3B‑Instruct‑2507 作为 LLM 背骨，对每个问题做答案判定，得到准确率。与 Full Text、Naive RAG、LangMem、A‑MEM、MemoryOS、Mem0、Zep 等基线比较。结果显示：
• 在 LongMemEval_S 上 TSM 的整体准确率为 74.80%（GPT‑4o‑mini），比 A‑MEM 的 62.60% 提升 12.2%；在多会话、时间推理子类的提升尤为显著；
• 在 LoCoMo 上 TSM 取得 71.23%（GPT‑4o‑mini）和 70.00%（Qwen3），均高于其他基线，尤其在单跳和时间类问题上表现突出。

**⚠️ 局限性**

局限性：
• 采用固定的时间粒度（如按月划分）进行持续记忆聚合，可能不适用于事件密度极低或极高的场景；
• 目前仅关注个性化记忆，未扩展到程序性记忆或多智能体共享记忆；
• 记忆构建与聚合在大规模长对话中仍需进一步优化计算开销。

---

## 729. A Scalable Solution for Node Mobility Problems in NDN-Based Massive LEO Constellations

**arXiv ID:** 2601.07466 | [PDF](https://arxiv.org/pdf/2601.07466v1)

**作者:** Miguel Rodríguez-Pérez `[一作]`, Andrés Suárez-González `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出了一套基于NDN（Named Data Networking）的低轨卫星星座（LEO）移动性解决方案，既解决了生产者（地面站/卫星站）在快速交叉链路切换时的连通性问题，又改进了消费者的重传机制，保证了在频繁的天基链路切换下的端到端通信不中断。

**💡 创新点**

创新点在于：①将卫星的二维网格拓扑利用Forwarding Hint进行自适应路由；②通过“Link”对象与滑动时间窗口（H参数）实现生产者与其可见卫星的动态绑定；③在消费者端直接使用Forwarding Hint在原始接入卫星与新接入卫星之间重传未完成的Interest，从而避免了传统IP级别的多路径重传与切换延迟。

**🔧 技术方法**

技术核心包括NDN路由与缓存机制、Forwarding Hint与自定义路由策略、Open Location Code (OLC) 用于编码地面站地理位置、NDNSim仿真平台以及星座轨道数据（Walker‑Delta/Starlink 结构）来模拟卫星交叉链路。

**📊 数据集**

实验数据基于仿真，使用Starlink‑风格星座：72个轨道平面、每平面22颗卫星、倾角53°、海拔550km。数据来源于标准轨道模型和NDNSim生成的网络拓扑。

**📈 对比分析**

在仿真中，生产者移动时通过设定H值可将因切换导致的Interest丢包率降至接近0%（H>1s时几乎无丢包），平均丢包时长短于单个链路可视时间；消费者移动时，重传机制能在毫秒级内恢复所有Pending Interest，保持无明显吞吐下降。与传统IP移动性协议相比，NDN方案无需额外的定位协议或代理，显著降低协议开销。

**⚠️ 局限性**

局限性包括：①未考虑链路失效、信号衰减或多跳链路故障；②假设地面站和卫星节点的时间同步足够精准；③在极大星座规模下，接入卫星集合可能过大，导致Forwarding Hint体积膨胀；④方案依赖仿真结果，缺乏真实场景实验验证。

---

## 730. PanoSAMic: Panoramic Image Segmentation from SAM Feature Encoding and Dual View Fusion

**arXiv ID:** 2601.07447 | [PDF](https://arxiv.org/pdf/2601.07447v1)

**作者:** Mahdi Chamseddine `[一作]` (German Research Center for Artificial Intelligence), Jason Rambach `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

结合预训练的 Segment Anything (SAM) 编码器，构建了 PanoSAMic 模型，对多模态全景图像（RGB、RGB‑Depth、RGB‑Depth‑Normal）进行语义分割。

**💡 创新点**

创新点包括：① 双视角融合（shifted panoramas）解决全景边界不连续问题；② Moving CBAM (MCBAM) 在局部滑动窗口内做通道与空间注意力，适配密集分割；③ 在不微调 SAM 编码器的前提下，通过多分支输出实现跨模态特征融合。

**🔧 技术方法**

核心技术包括：冻结 SAM ViT‑H 编码器并扩展分支输出、MCBAM、双视角融合与球面卷积的语义解码器、实例引导的语义细化以及 Jaccard/交叉熵交替损失。

**📊 数据集**

使用 Stanford2D3DS（RGB、Depth、Normal）和 Matterport3D（RGB、Depth）两个公开全景数据集进行评估。

**📈 对比分析**

与现有全景分割方法（如 Trans4PASS+, SGAT4PASS+, SFSS‑MMSI, 360BEV, OOOPS, SAM3 等）对比，PanoSAMic 在所有模态下均达到或超过 SOTA，在 mIoU 和 mAcc 上均优于监督式方法，并显著优于开词汇方法。

**⚠️ 局限性**

局限性包括：仍对地面真值误标、遮挡和边缘噪声敏感；模型依赖预训练 SAM，无法在无监督或极端小样本场景中自适应；以及在实时性和极大尺寸图像上推理速度有待提升。

---

## 731. Variational Autoencoder with Normalizing flow for X-ray spectral fitting

**arXiv ID:** 2601.07440 | [PDF](https://arxiv.org/pdf/2601.07440v1)

**作者:** Fiona Redmen `[一作]` (University of Southampton), Cecilia Garraffo `[通讯]` (Harvard-Smithsonian Center for Astrophysics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用变分自编码器结合正则化流，在BHB X射线光谱上实现全概率后验预测，显著加快参数推断速度

**💡 创新点**

创新点在于把正则化流嵌入自编码器，能够直接输出完整后验分布并通过物理潜在空间与解码器耦合来提高重构精度

**🔧 技术方法**

采用卷积编码器、神经样条流（normalizing flow）、GRU解码器，配合重构损失、潜在损失和流损失的联合训练

**📊 数据集**

使用了NICER观测的25个BHB共10800条光谱，结合100k条合成光谱，涵盖0.3–10 keV范围

**📈 对比分析**

与Xspec的MCMC（130次迭代）相比，单个样本预测速度提升约640倍，1000个后验样本速度提升约2000倍，重构误差（PGStat）与传统方法相当（≈3.8）

**⚠️ 局限性**

局限在于使用了过于简化的光谱模型、对低计数率光谱的预测不够精准，且对更复杂物理模型的适应性尚待验证

---

## 732. LOONG: Online Time-Optimal Autonomous Flight for MAVs in Cluttered Environments

**arXiv ID:** 2601.07434 | [PDF](https://arxiv.org/pdf/2601.07434v1)

**作者:** Xin Guan `[一作]`, Shuo Li `[通讯]` (Zhejiang University)

**通讯引用:** 48891 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套名为 LOONG 的在线学习加速时序最优规划与控制框架，实现了微型无人机在未知、拥挤环境中的高速度自主飞行。

**💡 创新点**

创新点包括：① 将时间最优多段多项式参考生成与 MPC 统一在一次优化中；② 在 MPC 的预测期内仅对前 G 步加入安全多面体约束，兼顾攻击性与安全；③ 通过模仿学习实现时间分配的实时加速；④ 采用轨迹重用策略保证连续规划的平滑性。

**🔧 技术方法**

主要技术：多项式轨迹优化（MINCO）、模仿学习（MLP）时间分配、时变约束的 MPCC、SFC（安全飞行通道）基于 CIRI 的凸多面体分解、LiDAR‑IMU SLAM（FAST‑LIO2）、实时离散化求解器 acados (SQP‑RTI)。

**📊 数据集**

使用仿真环境（包含稀疏障碍与森林密集障碍）以及真实 LiDAR 采集的点云地图进行验证；无公开数据集，实验全部在自建平台与仿真中完成。

**📈 对比分析**

与 SUPER 与 IPC 两种基准对比。结果显示：在时间关键任务场景中 LOONG 的平均速度、最大速度和飞行时间均显著优于 SUPER；在森林密集场景中，LOONG 既保持成功率 100% 又显著提升速度；计算时间稳定在 6–7 ms，远低于 SUPER 在复杂环境中的 50 ms。

**⚠️ 局限性**

局限性：① 仍受限于 MPC 的收敛性，极端动态环境下可能出现求解失败；② 需要预先训练时间分配网络，迁移到不同 MAV 规模需重新训练；③ 采用 LiDAR 进行地图构建，受光照与反射影响；④ 目前仅在室内/室内模拟环境验证，户外长距离飞行尚未测试。

---

## 733. KALE: Enhancing Knowledge Manipulation in Large Language Models via Knowledge-aware Learning

**arXiv ID:** 2601.07430 | [PDF](https://arxiv.org/pdf/2601.07430v1)

**作者:** Qitan Lv `[一作]` (University of Science and Technology of China), Chaochao Lu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于知识图谱的后训练框架KALE，能提升LLM在知识检索与推理上的表现。

**💡 创新点**

①使用多路径A*算法从KG中高效提取多跳推理路径；②通过生成高质量的推理理由（rationale）来引导模型；③采用知识感知微调（KA），即最小化带理由与不带理由输出分布的KL散度，提升知识操控能力。

**🔧 技术方法**

知识图谱检索、A*搜索、GPT-4o文本生成、KL散度对齐的微调、数据增强与标注技术。

**📊 数据集**

主要使用公开评测基准：AbsR、Common（commonsense）、BBH、RACE-H、RACE-M、MMLU、ARC‑c、ARC‑e；利用Wikidata作为知识图谱；对六大开源LLM（LLaMA3‑8B、Mistral‑7B、Qwen‑32B、Gemma‑9B、OLMOE‑7B、Orca‑7B）进行实验。

**📈 对比分析**

与Prompt、检索、SFT、增强、KG‑SFT等多种基线对比，KALE在所有模型和所有任务上均显著提升；在AbsR上最高提升11.72%，平均提升4.18%；在最强模型Qwen‑32B上实现89.93%/94.90%/77.91%等高分，明显优于SFT及检索方法。

**⚠️ 局限性**

①需要结构化的Q&A数据；②使用硬匹配的路径检索，忽略语义相似性；③锚点选择经验化；④依赖现成的知识图谱，域内KG稀缺时受限。

---

## 734. Loci Similes: A Benchmark for Extracting Intertextualities in Latin Literature

**arXiv ID:** 2601.07533 | [PDF](https://arxiv.org/pdf/2601.07533v1)

**作者:** Julian Schelb `[一作]` (University of Konstanz), Andreas Spitz `[通讯]` (University of Konstanz)

**通讯引用:** 1283 | [OpenAlex ID](https://openalex.org/A5018128415)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了Loci Similes基准数据集，包含约172k条拉丁文本段落及545对专家验证的互文链接，并给出了基线检索与分类实验。

**💡 创新点**

创新点在于提供了大规模标注互文对、基于错误率的评估框架以及可直接用于评测的全流程工具，促进了拉丁文互文检测的标准化与可复现性。

**🔧 技术方法**

主要技术包括使用密集检索双编码器与交叉编码器的检索‑再排序流水线，以及多语种预训练模型（E5-large、XLM‑RoBERTa‑Large）进行嵌入与二分类。

**📊 数据集**

数据集由约172k条文本段落组成，其中查询语料来自晚期拉丁文作者（如Hieronymus、Lactantius），源语料来自古典拉丁文作者（如Cicero、Virgil、Ovid 等），并附带545条专家验证的互文对。

**📈 对比分析**

采用五折交叉验证评测，检索阶段在Recall@10≈61%，在Recall@100≈72%，Recall@1000≈83%；分类阶段XLM‑RoBERTa‑Large的F1≈0.5；两阶段检索‑再排序在k=100时召回率≈79%，F1≈0.55，显著降低误报率并提升效率。

**⚠️ 局限性**

主要局限在于数据覆盖不完整、互文定义模糊导致标注不一致，且检索模型对低词汇重叠的隐含互文仍识别不足，需进一步扩展数据与改进模型。

---

## 735. Machine Learning Model Trading with Verification under Information Asymmetry

**arXiv ID:** 2601.07510 | [PDF](https://arxiv.org/pdf/2601.07510v1)

**作者:** Xiang Li `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society and School of Science and Engineering, Chinese University of Hong Kong, Shenzhen), Chenyou Fan `[通讯]` (South China Normal University)

**通讯引用:** 951 | [OpenAlex ID](https://openalex.org/A5100569461)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于游戏理论的机器学习模型交易框架，研究信息不对称下的模型欺骗与验证问题，并设计了最优定价策略。

**💡 创新点**

创新点在于首次将模型质量验证与信息不对称结合，揭示验证既可降低信息不对称又可提升双方收益，同时系统分析了订单信息保护对交易的负面影响。

**🔧 技术方法**

采用了博弈论、最优策略分析、学习曲线的幂律特性以及混合策略均衡求解等方法。

**📊 数据集**

实验使用 ResNet‑18 与 CIFAR‑10 数据集，对五个不同训练样本规模的模型进行评估。

**📈 对比分析**

通过数值实验比较了带验证与不带验证、订单信息保护与否的买卖双方收益，结果显示验证成本低、样本量大的情况下收益可逼近完全信息基准；而订单信息保护导致收益最低。

**⚠️ 局限性**

局限性包括仅考虑二元模型（低/高质量）和简化的验证误差模型，且在多模型或非幂律学习曲线情况下的分析与定价策略尚未给出。

---

## 736. High-Rank Structured Modulation for Parameter-Efficient Fine-Tuning

**arXiv ID:** 2601.07507 | [PDF](https://arxiv.org/pdf/2601.07507v1)

**作者:** Yongkang Liu `[一作]` (Northeastern University), Hinrich Schütze `[通讯]` (Munich Center for Machine Learning)

**通讯引用:** 46885 | [OpenAlex ID](https://openalex.org/A5071144367)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种新的参数高效微调方法 SMoA，利用多子空间的低秩 LoRA 模块对大型语言模型的原始权重进行结构化调制，以提升模型的表示能力。

**💡 创新点**

核心创新在于：①将原始权重的奇异向量按累计谱能量划分成若干子空间；②在每个子空间内单独训练一个低秩 LoRA 模块，并通过 Hadamard 乘法与该子空间的能量张量相结合，保证不同子空间不重叠，从而在保持参数量不增的前提下显著提升整体更新矩阵的秩。

**🔧 技术方法**

使用的技术包括：SVD 对权重进行分解；LoRA 与其变体；Hadamard 乘法实现权重调制；子空间划分与秩理论分析；以及对 Llama-2-7B/3-8B 的微调与评估。

**📊 数据集**

实验所用数据集涵盖常识推理（BoolQ、PIQA、SIQA、ARC‑c、ARC‑e、OBQA、HellaS、WinoG）、对话生成（CONVAI2）以及数学推理（GSM8K），在 Llama‑2‑7B 与 Llama‑3‑8B 两大后端模型上进行测试。

**📈 对比分析**

与 Prompt Tuning、P‑Tuning v2、LoRA、DoRA、MoRA、SSMLoRA、MeLoRA、HiRA 等 8 种主流 PEFT 方法进行对比。评估指标包括准确率、BLEU、METEOR、ROUGE‑L、BERT‑Score 等。SMoA 在所有任务与模型组合上均实现了最高或最接近最高的指标，平均准确率提升约 3‑5 个百分点，BLEU/METEOR/ROUGE‑L 亦显著优于同类方法。

**⚠️ 局限性**

主要局限：需要手动设定子空间数 K，最佳 K 随数据集与任务不同而变化，确定最优 K 需额外实验；此外，实验仅在单一模型与单一任务场景下验证，未对极大模型或多任务迁移场景的适用性进行评估。

---

## 737. Anatomy Aware Cascade Network: Bridging Epistemic Uncertainty and Geometric Manifold for 3D Tooth Segmentation

**arXiv ID:** 2601.07499 | [PDF](https://arxiv.org/pdf/2601.07499v1)

**作者:** Bing Yu `[一作]` (Nanchang University), Qiegen Liu `[通讯]` (Nanchang University)

**通讯引用:** 3297 | [OpenAlex ID](https://openalex.org/A5057647276)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了名为AACNet的三维牙齿分割框架，用于解决CBCT扫描中牙齿粘附和边界模糊问题。

**💡 创新点**

创新点在于将不确定性建模与几何先验相结合，分别通过熵门控边界重构模块（AGBR）针对高不确定性区域进行局部修正，以及通过签名距离图引导的解剖注意模块（SDMAA）在特征空间嵌入拓扑一致性约束，实现边界细化与全局形状一致性的双重优化。

**🔧 技术方法**

技术上采用残差U‑Net骨干网络，配合自适应熵门控残差分支、几何注意力机制、深度监督与多尺度训练，使用SGD优化，利用Z‑score标准化、随机裁剪与旋转等数据增强。

**📊 数据集**

实验数据包括125个内部CBCT扫描（100/25训练/测试）和一个公开的外部CBCT牙齿分割数据集（20个案例），两者均为高分辨率的三维体数据。

**📈 对比分析**

在内部数据上与八种先进方法（包括nnUNet、UNETR、SwinUNETR、E2MISeg、NexToU等）对比，AACNet在DSC上达到90.17%，HD95为3.63 mm，超越最佳对手；在外部数据上亦保持HD95仅2.19 mm，显示出强鲁棒性。

**⚠️ 局限性**

主要局限是两阶段级联结构导致计算量和推理时延增加，对第一阶段的粗定位质量高度依赖，未来工作计划将知识蒸馏为单阶段轻量网络，并扩展几何先验以适应更多口腔病变。

---

## 738. Secure Joint Source-Channel Coding for the AWGN Channel with Feedback: A Finite Blocklength Analysis

**arXiv ID:** 2601.07472 | [PDF](https://arxiv.org/pdf/2601.07472v1)

**作者:** Sheng Su `[一作]` (Southwest Jiaotong University), Xiaohu Tang `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 5939 | [OpenAlex ID](https://openalex.org/A5029840191)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在有限码长（finite blocklength, FBL）情形下，研究了带无噪声反馈的加性高斯噪声（AWGN）窃听通道的联合源-信道编码（JSCC），证明传统的Schalkwijk‑Kailath（SK）方案在FBL下并非最优，并提出一种在首时刻采用MMSE估计的改进SK方案；同时给出该模型的FBL取证界（上界）与改进方案对应的下界；并通过数值仿真验证改进方案在低码率时优于经典SK方案。

**💡 创新点**

1) 在FBL框架下首次给出AWGN窃听通道的取证界；2) 提出改进的SK方案，使首时刻估计误差下降，尽管引入额外信息泄漏，但整体性能提升；3) 通过解析与数值结合，对比经典与改进方案的极限性能。

**🔧 技术方法**

采用Gaussian信源与AWGN信道模型，利用MMSE估计、信息密度与tilted信息、Berry‑Esseen定理以及功率约束变换（peak→等功率）等信息理论工具，构造上下界并分析误差与泄漏率。

**📊 数据集**

本文无实验数据集，所有结果基于理论推导与仿真分析，使用标准的高斯信源与AWGN通道参数（如σ_s^2=1，σ_η^2=30，σ_e^2=20/25）。

**📈 对比分析**

比较方法：将改进SK方案与经典SK方案在相同误差阈值ε=10^{-5}、安全阈值δ=0.01、功率P=1的设置下进行仿真。性能指标为码率1/N与误差概率。仿真结果表明，在较小码率（N较大）时，改进方案的码率更高，误差概率更低；两方案在高码率下表现相近。

**⚠️ 局限性**

1) 取证界与下界之间仍存在较大差距，说明FBL下仍未找到最优方案；2) 改进方案虽然在首时刻引入MMSE估计，但随时间的累积信息泄漏率上升，导致安全阈值δ的严格约束；3) 论文仅针对单符号源（k=1）进行分析，未讨论多符号源序列的情况；4) 所有分析基于理想的无噪声反馈，实际系统中反馈延迟与噪声可能影响性能。

---

## 739. Knowledge Distillation for LLM-Based Human Activity Recognition in Homes

**arXiv ID:** 2601.07469 | [PDF](https://arxiv.org/pdf/2601.07469v1)

**作者:** Julien Cumin `[一作]` (Orange Research), Xi Chen `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究不同规模的Qwen3 LLM在多主体家庭活动识别（HAR）中的性能，并利用教师模型生成的推理示例对小型模型进行LoRA微调，以提升其推理能力。

**💡 创新点**

①系统性评估模型规模与HAR性能的关系；②首次将知识蒸馏应用于家庭HAR；③证明小模型微调后可接近大模型性能，仅需少量数据；③展示跨数据集迁移的可行性。

**🔧 技术方法**

基于Prompt的直接LLM推理、LoRA参数微调（知识蒸馏）以及vLLM推理加Unsloth训练框架。

**📊 数据集**

两套多主体HAR数据集：Marble（含环境+可穿戴传感器）和MuRAL（仅环境传感器）。

**📈 对比分析**

对比不同参数规模（0.6B–32B）的Qwen3模型以及微调后模型的F1分数；小模型在Marble上从16.5%提升至≈32%，在MuRAL上从10.8%提升至≈52%；与教师32B模型相比，微调后模型仅落后3–5个百分点。

**⚠️ 局限性**

仍需更小模型实现本地部署；蒸馏效果受教师模型性能限制；对单一数据集过拟合风险；仅评估了活动标签准确性，未考虑身份归属和能耗等方面。

---

## 740. Puzzle it Out: Local-to-Global World Model for Offline Multi-Agent Reinforcement Learning

**arXiv ID:** 2601.07463 | [PDF](https://arxiv.org/pdf/2601.07463v1)

**作者:** Sijia li `[一作]`, Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 82598 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对离线多智能体强化学习中的数据有限与模型误差问题，提出了 LOGO（Local‑to‑Global World Model）并结合不确定性加权采样扩展数据集，提升策略泛化与稳健性。

**💡 创新点**

创新点：①通过先对每个智能体的局部观测进行预测，再拼接推导全局状态，显著降低维度与误差传播；②利用不同路径预测差异快速估计不确定性，避免传统集成方法的计算开销；③将不确定性用于采样权重，优先使用可信转移样本。

**🔧 技术方法**

使用技术包括：离线模型基础的自编码器架构、局部与全局预测模块、基于差异的不确定性估计、加权采样策略、以及在 MACQL 基础上的离线 MARL 训练。

**📊 数据集**

数据集：SMAC（多种质量：Medium‑Replay、Expert、Mixed 等）和 MaMuJoCo（Hopper、Ant、HalfCheetah 等），均采用公开的离线训练数据。

**📈 对比分析**

与 8~10 种现有离线/在线、模型基/无模型、全局/局部方法（MACQL、OMAR、CFCQL、Morel、SUMO、MAMBA、MAZero 等）对比，LOGO 在 Medium 质量数据上均获得最高分，尤其在 SMAC 的 5m_vs_6m、6h_vs_8z 等地图上提升 10% 以上；MPC 整合后也表现更稳定；在推理速度上比传统集成模型快约 3 倍。

**⚠️ 局限性**

局限性：①对离线数据质量依赖较高，极低质量或极大 OOD 仍可能导致模型误差积累；②不确定性估计仅基于预测路径差异，可能在高度相关或噪声多的场景下不足；③目前主要验证于离散动作空间，持续扩展到连续动作或更大规模系统仍需研究。

---

## 741. Improving Video Question Answering through query-based frame selection

**arXiv ID:** 2601.07459 | [PDF](https://arxiv.org/pdf/2601.07459v1)

**作者:** Himanshu Patil `[一作]` (Indian Institute of Technology Bombay), Rohit Saluja `[通讯]` (Indian Institute of Technology Mandi)

**通讯引用:** 226 | [OpenAlex ID](https://openalex.org/A5061660587)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于查询的子模优化（SMI）函数的帧选择方法，用于提升视频问答（VideoQA）的准确性。

**💡 创新点**

创新点在于将子模互信息函数（FLMI、GCMI）用于视频帧的查询相关性和多样性权衡选择，且该方法不需要额外训练、模型无关，直接替换传统的均匀采样。

**🔧 技术方法**

核心技术包括：CLIP 图像/文本编码器生成共享嵌入空间；SUBMODLIB 库实现的子模函数；以及对 Video-LLaVA 和 LLaVA-NeXT 两大 VLM 的无缝集成。

**📊 数据集**

在 MVBench 多任务视频问答基准（共 4000 条 30fps 视频，20 个时序复杂任务）上进行评估。

**📈 对比分析**

与均匀采样和 AKS 基线相比，查询驱动的子模采样在 4/12 帧下平均提升 2–4%（例如 Video-LLaVA 4 帧从 33.21% 提升至 35.13%，12 帧从 32.38% 提升至 36.08%），在绝大多数任务上取得最优或次优结果。

**⚠️ 局限性**

主要局限：CLIP 编码器缺乏时序特征，导致在需要细粒度时间推理的短视频任务中可能不如均匀采样；选取的帧数上受 GPU 内存限制；对极短或极长视频的通用性仍待验证。

---

## 742. MegaFlow: Large-Scale Distributed Orchestration System for the Agentic Era

**arXiv ID:** 2601.07526 | [PDF](https://arxiv.org/pdf/2601.07526v1)

**作者:** Lei Zhang `[一作]` (Shenzhen Institutes of Advanced Technology), Min Yang `[通讯]` (Shenzhen Institutes of Advanced Technology)

**通讯引用:** 68830 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 MegaFlow，一套分布式三服务（模型、代理、环境）架构，用于大规模交互式 AI 代理的训练与评估，解决了安全隔离、存储扩展和吞吐瓶颈问题。

**💡 创新点**

创新点在于：① 把训练拆分为独立服务并通过统一 API 协调，② 采用“many‑small‑instances”弹性伸缩和混合执行模式（短暂/持续），③ 事件驱动资源与任务调度、分层隔离（实例 + 容器）以及基于云服务的按需容器镜像提供。

**🔧 技术方法**

使用技术包括：云原生计算（阿里云 ECS）、容器镜像仓库、分布式调度器、事件总线、文档数据库、内存队列、分布式信号量以及现有模型训练框架（vLLM、SGLang、VeRL、FSDP 等）。

**📊 数据集**

主要使用软件工程类数据集：SWE‑bench、SWE‑Gym 等，涵盖数万条容器化任务，实验规模高达 10,000 并发任务。

**📈 对比分析**

与传统高规格集中式方案（208 核 CPU、3TB 内存）比较，MegaFlow 在 2,000 并发任务时实现 32% 成本下降，吞吐量保持 100 分钟不变、持续 10,000 任务；持久执行模式下总时延仅 75 分钟；CPU/内存利用率更稳定，峰值更低，资源调度更可预测。

**⚠️ 局限性**

限制包括：仍依赖单一云提供商（阿里云）实现弹性扩展，未覆盖多云或混合云场景；目前仅针对单一类型任务（软件工程），缺少多环境、多服务依赖的复杂任务支持；对动态执行模式切换和跨容器网络依赖的细粒度优化尚未完善。

---

## 743. A Parity-Consistent Decomposition Method for the Weight Distribution of Pre-Transformed Polar Codes

**arXiv ID:** 2601.07515 | [PDF](https://arxiv.org/pdf/2601.07515v1)

**作者:** Yang Liu `[一作]` (Beijing University of Posts and Telecommunications), Kai Niu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 6748 | [OpenAlex ID](https://openalex.org/A5008455605)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于奇偶一致分解（PCD）方法的高效算法，用于确定预变换极化码的权重分布。

**💡 创新点**

通过引入扩展信息集和等价类理论，显著降低了计算复杂度，并实现了递归计算。

**🔧 技术方法**

使用了奇偶一致分解（PCD）方法和迭代算法来构建扩展信息集。

**📊 数据集**

使用了预变换极化码的数据集，具体的数值结果展示了该方法在不同码率下的表现。

**📈 对比分析**

与现有的确定性算法相比，提出的方法在计算复杂度上显著降低，数值结果表明在高和低码率下均表现良好。

**⚠️ 局限性**

对于一般随机预变换极化码和大多数PAC码，PCD方法在扩展信息集大小的减少上效果有限。

---

## 744. Land-then-transport: A Flow Matching-Based Generative Decoder for Wireless Image Transmission

**arXiv ID:** 2601.07512 | [PDF](https://arxiv.org/pdf/2601.07512v1)

**作者:** Jingwen Fu `[一作]` (KTH Royal Institute of Technology), Dong In Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 23480 | [OpenAlex ID](https://openalex.org/A5022649488)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种土地先行-运输（LTT）框架，利用流匹配生成式解码器在无线图像传输中直接嵌入物理信道，避免了传统扩散模型的高采样延迟。

**💡 创新点**

创新点在于：①将物理无线信道映射到连续时间概率流的“落地点”，②使用条件流匹配（CFM）学习解析的速度场，③通过MMSE前置实现Rayleigh和MIMO信道的AWGN等价化，使得同一训练好的解码器可跨信道使用。

**🔧 技术方法**

采用的核心技术包括Flow Matching、Conditional Flow Matching、U‑Net学生向量场、概率流ODE求解器以及MMSE线性均方误差等价化。

**📊 数据集**

实验使用了MNIST、Fashion‑MNIST和DIV2K三大公开图像数据集。

**📈 对比分析**

与JPEG2000+LDPC、DeepJSCC和CDDM等基线在AWGN、Rayleigh和MIMO信道下进行比较，结果显示LTT解码器在PSNR、MS‑SSIM和LPIPS上均显著优于对比方法，且仅需10步ODE即可获得与传统扩散模型相当甚至更好的重建质量。

**⚠️ 局限性**

限制方面：目前仅针对线性高斯信道（AWGN、Rayleigh、MIMO）实现等价化，对非高斯或时变信道仍需进一步研究；解码性能仍受噪声水平估计误差和ODE步数的影响。

---

## 745. Graph Inference Towards ICD Coding

**arXiv ID:** 2601.07496 | [PDF](https://arxiv.org/pdf/2601.07496v1)

**作者:** Xiaoxiao Deng `[一作]` (DePaul University), Xiaoxiao Deng `[通讯]` (DePaul University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5108798654)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

将自动 ICD 编码任务重新定义为在 ICD 代码层次结构上生成标签图，并提出 LabGraph 框架实现该任务。

**💡 创新点**

创新点在于：① 将 ICD 编码转化为图生成问题，显式建模层次关系、兄弟关系与互斥约束；② 采用多头残差 CNN（MHR-CNN）+ Fat‑RGCN+消息整合模块（MIM）捕捉文本与代码结构信息；③ 通过对抗强化学习（ARCL）和对抗自适应训练（AAT）提升生成质量与鲁棒性；④ 在训练中引入图判别器（LGD）提供中间监督。

**🔧 技术方法**

使用的技术包括：多头残差卷积网络、RGCN（Fat‑RGCN）、消息整合模块、对抗强化学习、对抗自适应训练、LSTM 判别器、嵌入式对抗扰动。

**📊 数据集**

主要数据集为 MIMIC‑III（全数据集及 Top‑50 版本），并在 Cora 与 FB15k‑237 上做知识图谱子结构实验。

**📈 对比分析**

与现有的 Hierarchy‑SVM、C‑LSTM‑Att、C‑MemNN、CAML、DR‑CAML、LAAT、JointLAAT、ISD、MSMN、FUSION 等基线相比，LabGraph 在 MIMIC‑III Full 与 Top‑50 的 micro‑F1、macro‑AUC、P@K 等指标上分别提升约 4–12 %（如表中所示），性能显著优于其它方法。

**⚠️ 局限性**

限制包括：难以扩展到完整 ICD‑10（70k+ 代码）；模型可解释性有限；对大规模标注数据依赖较强；跨机构迁移时需大量重训，且对罕见病编码仍存在较高错误率。

---

## 746. The Secretary Problem with Predictions and a Chosen Order

**arXiv ID:** 2601.07482 | [PDF](https://arxiv.org/pdf/2601.07482v1)

**作者:** Helia Karisani `[一作]` (University of Massachusetts Amherst), Cameron Musco `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1313 | [OpenAlex ID](https://openalex.org/A5023229845)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在拥有机器学习预测的秘书问题中提出一种随机化算法，结合预测与传统的 Dynkin 规则，在预测准确时接近最优，在预测误差大时仍保持常数竞争比。

**💡 创新点**

创新点在于：①通过随机化控制进入传统阈值策略的时机；②利用单一最优预测候选者的固定到达位置与随机排布其他候选者，显著提升 0.215→0.221（ROSP）和 0.215→0.262（COSP）的竞争比；③不需要提前知道误差上界 ε，算法自适应。

**🔧 技术方法**

技术上采用了：预测可信模式与大误差模式的切换；对到达顺序进行固定/随机化调度；对不同误差集合进行分案分析；数值优化与符号求解结合，验证最坏情况竞争比。

**📊 数据集**

本文未使用真实数据集，而是基于理论分析与最坏案例构造进行验证。

**📈 对比分析**

与先前 Fujii‑Yoshida 算法比较，竞争比从 max{0.215,(1-ε)/(1+ε)} 提升到 max{0.221,(1-ε)/(1+ε)}（ROSP）和 max{0.262,(1-ε)/(1+ε)}（COSP）；此外突破了 0.25 的随机顺序下的上界。

**⚠️ 局限性**

局限性包括：竞争比仍低于 1/e≈0.368 的理论上限；仅适用于单一选择；未考虑多选、图形或基数约束等更一般的秘书问题；预测模型假设为单独的点估计，未涵盖概率预测或分布信息。

---

## 747. NanoCockpit: Performance-optimized Application Framework for AI-based Autonomous Nanorobotics

**arXiv ID:** 2601.07476 | [PDF](https://arxiv.org/pdf/2601.07476v1)

**作者:** Elia Cereda `[一作]` (Dalle Molle Institute for Artificial Intelligence), Daniele Palossi `[通讯]` (Integrated Systems Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了NanoCockpit框架，整合协程多任务、双缓冲摄像头驱动和零拷贝Wi‑Fi堆栈，实现了对Crazyflie无人机上多 MCU 的高效流水线处理，显著提升了 TinyML 推理与控制的吞吐量与低延迟。

**💡 创新点**

创新点在于：① 将协程编程引入 GWT GAP8 的轻量级运行时，打破传统单线程事件回调的瓶颈；② 设计双缓冲摄像头驱动和 DMA 路由，实现摄像头采集与推理完全重叠；③ 构建零拷贝、可并发的 CPX Wi‑Fi 协议栈，消除串行传输导致的延迟。

**🔧 技术方法**

采用 STM32/FreesRTOS、GAP8/PMSIS、ESP32 Xtensa、协程框架、DMA、SPI、CPI、HyperBus、Wi‑Fi、ROS 与 Python 等技术栈。

**📊 数据集**

在三类真实实验中使用了：人类姿态估计（PULP‑Frontnet 与 MobileNetV2）、无人机对无人机定位（FCNN）、无人机竞速障碍规避（CNN 预测碰撞概率）。实验数据通过运动捕捉系统采集，未依赖公开数据集。

**📈 对比分析**

与多项 SOTA TinyML 研究相比，NanoCockpit 在闭环吞吐量上逼近推理吞吐上限，平均位置误差降低约 30%，任务成功率从 40% 提升至 100%，Wi‑Fi 传输延迟降至 55 ms，整体系统延迟显著下降。

**⚠️ 局限性**

局限性包括：① 仍受单个模型推理时间限制，无法突破硬件推理瓶颈；② 需要针对不同 MCU 进行手工调参，迁移成本较高；③ 在极高频率或高动态场景下，摄像头采集速率与通信速率仍可能成为瓶颈；④ 对功耗与热设计仍未做深入评估。

---

## 748. IFDNS: An Iterative Feedback-Driven Neuro-Symbolic Method for Faithful Logical Reasoning

**arXiv ID:** 2601.07464 | [PDF](https://arxiv.org/pdf/2601.07464v1)

**作者:** Xiaoheng Wang `[一作]` (University of Science and Technology of China), Jing Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 111821 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了迭代反馈驱动的神经符号方法IFDNS，旨在提升大语言模型的逻辑推理能力并减轻符号化过程中的信息损失；

**💡 创新点**

创新点在于多轮反馈机制、细粒度实体/量词识别、逻辑扩展与自然语言再集成以及词序重排，形成完整的逻辑推理增强链；

**🔧 技术方法**

技术上结合了LLM提示、多轮自检反馈、Python实现的命题逻辑推理、符号化与反向翻译及上下文重排；

**📊 数据集**

实验使用六个逻辑推理数据集（ReClor、LogiQA、RuleTaker、ProofWriter、FOLIO、PrOntoQA）和一个数学推理数据集R-GSM；

**📈 对比分析**

与Direct、CoT、CoT‑SC以及LoT、Logic‑LM等基线对比，使用GPT‑4o、GPT‑4o‑mini和DeepSeek‑V3模型，在36组对比中提升了0–13.9%（CoT‑SC+IFDNS在所有数据集上均为最佳）；

**⚠️ 局限性**

局限性在于仅支持命题逻辑，缺乏更高阶逻辑推理能力，推理范围受限；多轮反馈导致token消耗增加；词序重排的效果仍有提升空间。

---

## 749. WaveMan: mmWave-Based Room-Scale Human Interaction Perception for Humanoid Robots

**arXiv ID:** 2601.07454 | [PDF](https://arxiv.org/pdf/2601.07454v1)

**作者:** Yuxuan Hu `[一作]` (Fudan University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**通讯引用:** 6747 | [OpenAlex ID](https://openalex.org/A5005666034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种空间自适应的mmWave雷达感知框架WaveMan，能在房间尺度下实现人机交互中的手势识别，兼顾用户隐私和多角度多距离的鲁棒性。

**💡 创新点**

核心创新在于三大模块：①几何视角对齐将不同位置的点云映射到统一正对坐标系；②无对齐谱图增强通过CycleGAN式稀疏‑>稠密翻译缓解长距离、偏斜视角导致的信号稀疏；③双分支通道注意力（DBCA）提升特征对位置不变性的鲁棒性。

**🔧 技术方法**

技术手段包括MIMO FMCW mmWave雷达数据预处理、点云几何对齐、时频谱生成、CycleGAN谱图增强、轻量级CNN+DBCA分类网络。

**📊 数据集**

使用自采集的室内实验数据，共12,000条样本，涵盖六个预定义位置（P1–P6）及五位受试者、五种手势，未公开使用任何公开数据集。

**📈 对比分析**

与多种基线方法（TS-DRSPA、DI‑Gesture、ShuffleNet‑Traj、5D‑DCN）在相同的交叉位置与随机位置评估下对比。WaveMan在未见位置上准确率从基线的67.45%提升至95.94%（提升≈28.5%），随机位置准确率从33.00%提升至94.33%，显著优于所有对比方法。

**⚠️ 局限性**

局限性包括：仍需依赖专业mmWave雷达硬件，且对极端遮挡或多目标场景的鲁棒性尚未充分验证；对实时性能的硬件实现需进一步优化；模型在多用户同时交互时的辨识效果未评估。

---

## 750. RLPO: Residual Listwise Preference Optimization for Long-Context Review Ranking

**arXiv ID:** 2601.07449 | [PDF](https://arxiv.org/pdf/2601.07449v1)

**作者:** Hao Jiang `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**通讯引用:** 29845 | [OpenAlex ID](https://openalex.org/A5100403129)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种残差列表偏好优化（RLPO）框架，用于长上下文评论排序，先使用LLM做点wise打分，再用轻量级残差头纠正列表级交互误差；

**💡 创新点**

创新点在于在不需要完整token级列表推理的前提下，在表示层面通过残差修正实现点wise与列表级上下文的融合，兼顾效率与准确性；

**🔧 技术方法**

技术手段包括：Mistral‑7B LLM的全参数微调产生点wise分数和句子表示，残差自注意力块进行列表级交互，LambdaRank式的重要性加权对NDCG的梯度化；

**📊 数据集**

使用从Amazon Reviews 2023构建的公开基准，涵盖All_Beauty、Fashion、Baby_Products、Software四类，包含大量长文本评论和人类验证标签；

**📈 对比分析**

与BM25、SFT、DPO、LIPO等基线对比，RLPO在NDCG@1/3/10上均明显领先，尤其在列表长度扩大到50时保持稳定且优于其他方法；

**⚠️ 局限性**

局限性包括：评论的主观性导致标签噪声，人工验证成本高；当基础点wise打分严重失准或对提示敏感时，残差头难以完全纠正；

---

## 751. Formalization of Amicable Numbers Theory

**arXiv ID:** 2601.07444 | [PDF](https://arxiv.org/pdf/2601.07444v1)

**作者:** Zhipeng Chen `[一作]` (Shanghai Dianji University), Jingyi Zhan `[通讯]` (Shanghai Dianji University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在 Lean4 证明助手中完整形式化了友好数理论，定义了友好数、友好对、社会数、订婚数等概念，并计算验证了多组已知的友好数对与 Poulet 五周期。

**💡 创新点**

首次在机器可验证的形式下完成了三条经典生成规则（Thābit、Euler 和 1986 年的 Borho‑Hoffmann 繁殖法）的完整证明，特别是对 Borho‑Hoffmann 通过自动化多项式验证实现的首次完整形式化。

**🔧 技术方法**

使用 Lean4 及 Mathlib 库、native_decide、index‑shifting、zify、ring、polynomial tactics 等技术完成证明与计算。

**📊 数据集**

主要利用已知的历史友好数对（如 220/284、1184/1210、17296/18416 等）和 Poulet 五周期作为实例；在检验时采用 10^65 的搜索上界作为经验阈值，但并未执行完整搜索。

**📈 对比分析**

对特定数值通过 native_decide 进行即时计算验证，速度可达毫秒级；对一般性证明则依赖理论推导与自动化环证明，未给出系统的性能基准，但证明无任何 sorry，代码可编译通过。

**⚠️ 局限性**

尚未解决奇偶友好数、无限友好数等开放问题；缺乏对更大规模友好数生成的自动化工具；当前实现仅局限于 Lean4，尚未正式合并至 Mathlib。

---

## 752. Stagewise Reinforcement Learning and the Geometry of the Regret Landscape

**arXiv ID:** 2601.07524 | [PDF](https://arxiv.org/pdf/2601.07524v1)

**作者:** Chris Elliott `[一作]`, Daniel Murfet `[通讯]` (Timaeus)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了奇异学习理论在深度强化学习中的应用，提出局部学习系数（LLC）来解释贝叶斯后验收缩和阶段性学习现象。

**💡 创新点**

将Watanabe的自由能公式推广到RL，证明LLC是后验浓度的控制因子，揭示更优策略不一定在贝叶斯意义上更优，并验证了“简单→复杂”相变。

**🔧 技术方法**

采用奇异学习理论、控制即推理框架、重要性采样构造通用后验、pSGLD估计LLC、政策梯度训练等技术。

**📊 数据集**

在简化的Cheese-in-the-Corner网格世界环境（13×13）上实验，使用不同α和γ混合策略进行多次随机种子训练。

**📈 对比分析**

通过对比LLC估计与回报（regret）随训练阶段的变化，观察到“对立阶梯”现象；LLC在未观测状态下也能检测到相变，证明其对策略复杂度的捕捉优于单纯性能指标。

**⚠️ 局限性**

仅在有限的网格世界实验验证，贝叶斯后验与SGD动力学的对应关系仍未完全阐明，且对更大规模环境与模型的推广仍待进一步研究。

---

## 753. From RAG to Agentic RAG for Faithful Islamic Question Answering

**arXiv ID:** 2601.07528 | [PDF](https://arxiv.org/pdf/2601.07528v1)

**作者:** Gagan Bhatia `[一作]` (Qatar Computing Research Institute), Firoj Alam `[通讯]` (Arab Center for Research and Policy Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了双语（阿拉伯语/英语）生成式基准和端到端基于古兰经检索的模型，以评估和减少伊斯兰问答中的幻觉和回避。

**💡 创新点**

①引入仅有单一金标准答案的原子问题基准，直接测量幻觉和回避；②构建包含25k阿拉伯语SFT对、5k双语偏好样本和6k古兰经经文检索语料的完整资源；③设计agentic RAG框架通过结构化工具调用实现迭代证据搜索和答案修订，显著提升跨语言鲁棒性。

**🔧 技术方法**

监督微调(SFT)、奖励驱动对齐(RL/GSPO)、检索增强生成(RAG)以及代理式RAG(agentic RAG)，并使用LLM-as-Judge进行评估。

**📊 数据集**

3,810道阿拉伯语/英语双语问答基准；25,000条阿拉伯语文本检索SFT问答；5,000条双语偏好样本；6,236条古兰经经文检索语料；以及公开基准如QURAN QA、IslamicEval等。

**📈 对比分析**

在多种阿拉伯语中心与多语种指令调优模型上进行对照实验，使用%Correct衡量。检索增强平均提升约+8‑10点；agentic RAG 在 Qwen3‑4B‑2507 上从 38.85 提升至 48.90，Fanar‑2‑27B 从 48.65 提升至 57.30，显示在英语与阿拉伯语间的差距缩小，整体性能达到同类最佳。

**⚠️ 局限性**

基准仅覆盖单一答案，未考虑多义传统；依赖LLM评审，可能存在语言偏差；主要检索基于古兰经，忽略圣训和法学文献；agentic RAG 产生延迟和工具使用错误；适用于短问答，未覆盖长文本指导。

---

## 754. Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions

**arXiv ID:** 2601.07516 | [PDF](https://arxiv.org/pdf/2601.07516v1)

**作者:** Yongqi Li `[一作]` (Wuhan University), Yongbin Li `[通讯]` (Tongyi Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个紧凑的潜在动作空间，并用该空间对多模态对话代理进行强化学习微调。

**💡 创新点**

①首次在多模态对话代理中引入潜在动作；②利用跨模态投影器并加入循环一致性损失，将海量文本数据与有限的图文对齐；③将动作空间从完整词表缩减至128维，显著提高探索效率。

**🔧 技术方法**

采用强化学习（GRPO、Dr.GRPO、DAPO、BNPO 等）与跨模态投影器、循环一致性训练、逆向动力学学习、世界模型及策略行为克隆等技术。

**📊 数据集**

使用14M图像+1B文本的图文对数据（图像标题、新闻、维基百科）以及627B文本的 SlimPajama 文本库，在角色扮演和个性化对话两个基准上进行评估。

**📈 对比分析**

与 Prompt、SFT、以及 token‑级 RL 等基线进行对比，使用 LLM‑as‑a‑Judge 指标，在 3B/7B 模型和四种 RL 算法上平均提升约 4%，在所有任务上均优于 token‑级 RL。

**⚠️ 局限性**

训练时间比 token‑级慢 1.08×，推理延迟增加 1.13×；实验仅覆盖对话任务，未验证到更广泛场景。

---

## 755. FocalOrder: Focal Preference Optimization for Reading Order Detection

**arXiv ID:** 2601.07483 | [PDF](https://arxiv.org/pdf/2601.07483v1)

**作者:** Fuyuan Liu `[一作]` (Unisound AI Technology Co.Ltd), Junnan Zhu `[通讯]` (MAIS Institute of Automation Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FocalOrder 框架，解决阅读顺序检测中的位置偏差（Positional Disparity）问题，提升文档中间段落的排序准确性。

**💡 创新点**

创新点在于自适应难度发现（EMA 追踪错误率）和难度校准的对比排序，动态聚焦结构模糊区并通过加权交叉熵和对比损失实现全局一致性。

**🔧 技术方法**

采用 EMA 难度发现、位置加权交叉熵、难度校准对比排序（DCPR）以及 LayoutLMv3 作为统一编码器。

**📊 数据集**

使用公开基准 OmniDocBench（v1.0/v1.5）和 Comp‑HRDoc 两个数据集进行训练与评估。

**📈 对比分析**

与多种基线（如 UniHDSA、MinerU、PaddleOCR‑VL、GPT‑4o 等）对比，在 OmniDocBench v1.0 上取得 0.038（英）/0.055（中）编辑距离，在 Comp‑HRDoc 上取得 97.1%/91.1% REDS，显著超过现有 SOTA。

**⚠️ 局限性**

局限性包括：依赖前置布局检测的粒度；无法纠正缺失或错误的布局元素；对不同语义标签体系需重新对齐；对极端非结构化/艺术化排版的泛化能力有限；训练时增加轻微计算开销。

---

## 756. JudgeFlow: Agentic Workflow Optimization via Block Judge

**arXiv ID:** 2601.07477 | [PDF](https://arxiv.org/pdf/2601.07477v1)

**作者:** Zihan Ma `[一作]` (KAIST), Jinkyoo Park `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于评估‑判断‑优化‑更新管道（JudgeFlow）的自动化方法，通过可复用的逻辑块和块级责任评分的判别模块实现对LLM代理工作流的细粒度诊断与自动优化。

**💡 创新点**

创新之处在于引入层次化的逻辑块抽象（顺序、循环、条件）以及基于rank的责任分配判别器，结合LLM驱动的优化器实现对失败实例的细粒度错误定位与针对性改进。

**🔧 技术方法**

使用LLM作为执行器、判别器和优化器；构建逻辑块抽象；采用块级责任评分的判别模块；实现Evaluation‑Judge‑Optimization‑Update的迭代流程；在优化中利用少量样本进行少样本学习。

**📊 数据集**

在公开基准上进行实验：数学推理（GSM8K、MATH、AIME）和代码生成（MBPP、HumanEval），并进行跨基准迁移验证。

**📈 对比分析**

与单体代理、人工设计多代理、自治多代理等多种基线（如CoT、Self‑Consistency、AutoGen、MermaidFlow 等）进行对比，JudgeFlow 在 GSM8K/MATH/MBPP/HumanEval 上平均提升约 1.7%/3.1%/1.5%，并在 AIME 2025 上提升 2.67pp。

**⚠️ 局限性**

判别器依赖LLM，可能受偏见与噪声影响；对高复杂度工作流的搜索空间仍有限；优化过程主要关注单个块，可能忽略跨块交互问题。

---

## 757. ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs

**arXiv ID:** 2601.07475 | [PDF](https://arxiv.org/pdf/2601.07475v1)

**作者:** Haoqian Meng `[一作]` (Tianjin University), Xindian Ma `[通讯]` (Tianjin University)

**通讯引用:** 149 | [OpenAlex ID](https://openalex.org/A5110658926)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 NVIDIA Blackwell 架构下的 NVFP4 微尺度量化，提出 ARCQuant 框架，通过增量残差通道实现双阶段 4‑bit 激活量化，兼顾精度与吞吐量。

**💡 创新点**

创新点在于：①保持统一的 NVFP4 格式，避免混合精度对 Tensor Core 的破坏；②利用残差通道将量化误差内嵌到 GEMM 扩展维度，实现无缝的误差补偿；③理论证明双阶段误差上界可与 MXFP8 相当。

**🔧 技术方法**

核心技术包括：块级缩放量化、通道重排序与阈值挑选、残差计算与量化、Fused 量化与 GEMM kernel、误差上界分析。

**📊 数据集**

使用 Llama 3.1‑8B、Qwen 2.5‑7B/32B 等 LLM；校准集为 WikiText2、C4、HumanEval；评估数据集包括 WikiText2、MMLU、HumanEval、MBPP、GSM8K、CMATH 等。

**📈 对比分析**

与 FP16、W4A8 RTN、FlatQuant、Atom 等基线在 0‑shot/5‑shot MMLU、Perplexity、下游代码/算术任务中比较，ARCQuant 在保持近 FP16 精度的同时在 RTX 5090/PRO 6000 上实现 2–3× 的推理速度提升；在 4‑bit NVFP4 下优于所有现有 PTQ 方法。

**⚠️ 局限性**

局限性包括：目前仅使用 RTN 进行权重量化，未集成 GPTQ/ AWQ 等高级权重补偿；完全依赖 Blackwell 的 NVFP4 硬件，缺乏对旧架构的适配；离线校准的通道排序与残差通道数固定，可能在分布漂移或极端 OOD 场景下失效。

---

## 758. From Sketch to Fresco: Efficient Diffusion Transformer with Progressive Resolution

**arXiv ID:** 2601.07462 | [PDF](https://arxiv.org/pdf/2601.07462v1)

**作者:** Shikang Zheng `[一作]` (South China University of Technology), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 13598 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Fresco动态分辨率框架，通过统一噪声场和基于方差的逐步上采样，实现Diffusion Transformers的高效采样。

**💡 创新点**

创新点在于：①将所有采样阶段共用固定噪声向量，消除重噪声导致的跨阶段不一致；②使用自适应方差判定的区域性上采样，避免一次性全局放大导致的误差累积。

**🔧 技术方法**

主要技术包括：Token‑encoded Unified Noise Field、方差引导的进阶上采样、Hadamard变换生成细粒度噪声、与蒸馏、量化、特征缓存等加速方法的协同使用。

**📊 数据集**

在FLUX.1‑dev（文本→图像）和HunyuanVideo（文本→视频）上评估，且对量化版、蒸馏版FLUX进行兼容性验证，使用DrawBench、VBench、ImageReward、CLIP Score、CLIP‑IQA等指标。

**📈 对比分析**

与现有动态分辨率、特征缓存、量化、蒸馏等方法对比，Fresco在FLUX上实现10×加速、HunyuanVideo 5×加速，且在质量指标上与原模型持平甚至更好；与蒸馏组合可达22×加速，显示出强大的兼容性和效果。

**⚠️ 局限性**

局限性包括：对噪声方差阈值的手工设定可能需要针对不同模型调整；对不同模型结构的通用性验证仍不足；在极端压缩条件下仍可能出现细节丢失或轻微质量下降。

---

## 759. Sparse Point-wise Privacy Leakage: Mechanism Design and Fundamental Limits

**arXiv ID:** 2601.07523 | [PDF](https://arxiv.org/pdf/2601.07523v1)

**作者:** Amirreza Zamani `[一作]` (KTH), Mikael Skoglund `[通讯]` (KTH)

**通讯引用:** 8707 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种信息论隐私机制设计问题，设计了从有用数据Y生成的披露数据U，同时限制与敏感数据X的相关性和泄露量。

**💡 创新点**

提出了稀疏逐点隐私泄露的概念，作为一种最坏情况隐私标准，确保每个披露符号u与最多N个X的实现相关，并且总泄露量受到限制。

**🔧 技术方法**

使用了信息几何的概念来获得互信息的局部二次近似，并提出了一个凸半正定规划(SDP)松弛方法来解决稀疏Rayleigh商最大化问题。

**📊 数据集**

使用了离散随机变量X和Y的联合分布P_XY，假设X和Y的字母表大小相同，且每个元素的分布非零。

**📈 对比分析**

通过与稀疏主成分分析(SPCA)的关系，证明了稀疏Rayleigh商最大化问题是NP难的，并通过数值实验展示了SDP解在稀疏阈值以上与确切组合最优解匹配。

**⚠️ 局限性**

在高维情况下，稀疏Rayleigh商最大化问题是NP难的，且在字母表增大时，组合支持枚举方法变得不可行。

---

## 760. Mon3tr: Monocular 3D Telepresence with Pre-built Gaussian Avatars as Amortization

**arXiv ID:** 2601.07518 | [PDF](https://arxiv.org/pdf/2601.07518v1)

**作者:** Fangyu Lin `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 82598 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了 Mon3tr 系统，通过离线生成高质量 3D Gaussian 头像，在线仅传输低比特率的姿态与表情参数，实现单摄像头实时 3D 远程呈现。

**💡 创新点**

创新点在于将昂贵的全景重建迁移到离线 amortized 计算，并使用 3D Gaussian splatting 绑定在自定义骨骼模板上，形成可在移动端实时渲染的可动画头像；同时提出轻量化网格与属性变形网络。

**🔧 技术方法**

核心技术包括 3D Gaussian splatting、SPMM3 参数化人体模型、Mesh 变形网络、属性变形网络、FP16+LZ4 压缩、WebRTC 数据通道、并行姿态、手部、面部估计模型。

**📊 数据集**

主要使用自制 iCom4D 数据集进行训练与评估，并在 ActorHQ、AvatarRex 数据上做对比测试。

**📈 对比分析**

与 MagicStream、MetaStream、TeleAloha、MonoPort 等基线以及 MeshAvatar、Animatable Gaussians、MMLPHuman 等头像方法对比，PSNR 达到 32.4 dB、SSIM 0.986、LPIPS 0.023、FID 11；实时 60 FPS、延迟 73 ms、比特率 <0.2 Mbps，显著优于传统多摄像头方案。

**⚠️ 局限性**

局限在于需对每个用户进行一次昂贵的多视角录制和离线重建，无法即时适应衣着或外观变化；且离线过程仍需高算力与时间。

---

## 761. Principal ideal problem and ideal shortest vector over rational primes in power-of-two cyclotomic fields

**arXiv ID:** 2601.07511 | [PDF](https://arxiv.org/pdf/2601.07511v1)

**作者:** Gaohao Cui `[一作]` (Shandong University), Jincheng Zhuang `[通讯]` (Shandong University)

**通讯引用:** 60 | [OpenAlex ID](https://openalex.org/A5103179935)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在2的幂次循环域[ζ₂ⁿ⁺¹]中，针对满足p≡7,916的素数的素理想的最短向量问题，并给出了精确长度表达式及更紧的上界。

**💡 创新点**

创新点在于将主理想问题与分裂子域攻击相结合，证明对某些理想域存在生成元即为最短向量，从而将求最短向量转化为求最短生成元，首次给出了p≡7,916情况下的最短向量长度公式及更优上界。

**🔧 技术方法**

采用的主要技术包括：利用理想的生成元与最短向量的对应关系；利用Pell方程求解aₚ、bₚ；利用循环域的嵌入与单位群结构；以及子域分解与理想分解的代数数论方法。

**📊 数据集**

无具体数据集，研究完全基于理论证明与数论计算。

**📈 对比分析**

本文将得到的最短向量长度上界与传统的Minkowski上界进行比较，证明在此类素数上新的上界（√(2^{2n+1}p)）严格优于Minkowski给出的2ⁿ√p，从而在理论上提供了更紧的安全参数估计。

**⚠️ 局限性**

局限性在于仅针对特定形式的素数p≡7,916（以及类似p≡7,916, 716等）以及特定的2的幂次循环域，尚未证明该生成元-最短向量对应性在更一般的数域或理想类型中成立。

---

## 762. FROAV: A Framework for RAG Observation and Agent Verification - Lowering the Barrier to LLM Agent Research

**arXiv ID:** 2601.07504 | [PDF](https://arxiv.org/pdf/2601.07504v1)

**作者:** Tzu-Hsuan Lin `[一作]` (AetheTech), Chih-Hsuan Kao `[通讯]` (AetheTech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个名为FROAV的框架，用于快速搭建、评估和验证基于检索增强生成（RAG）的LLM代理。

**💡 创新点**

集成了可视化工作流、LLM-as-a-Judge多维度评估、结构化人机交互以及统一的PostgreSQL日志，显著降低了研发门槛。

**🔧 技术方法**

使用n8n可视化编排、PostgreSQL + pgvector存储、FastAPI后端、Streamlit前端、Docker Compose容器化，并实现多模型LLM评估的RAG流水线。

**📊 数据集**

以美国证券交易委员会SEC 10‑K/10‑Q文件为演示数据，并设计为可适用于任何文本语料。

**📈 对比分析**

将手动编码的完整RAG管道与FROAV框架对比，部署时间从40‑50小时降至1小时，工作流代码从1000+行降至0行，评估和人机交互时间大幅缩短；实验表明框架可实现与人工判断高度一致的多维度评分。

**⚠️ 局限性**

目前仅支持四维评估框架，缺乏大规模压力测试，且对非常专业或多模态数据的支持仍有限。

---

## 763. Frequency-Adaptive Multi-Band Architecture for Upper Mid-Band MIMO Systems

**arXiv ID:** 2601.07489 | [PDF](https://arxiv.org/pdf/2601.07489v1)

**作者:** Emiel Vanspranghels `[一作]` (WaveCoRE), Sofie Pollin `[通讯]` (Interuniversity Microelectronics Centre)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究FR3频段MIMO性能，并提出一种可在多子带间动态重用ADC/DAC与基带资源的全数字频率自适应多带架构，以适应频谱可用性与环境变化。

**💡 创新点**

创新点在于：①将频率整合与频率分区设计的优点结合，提出可在多子带同时工作且在子带不可用时动态重配硬件资源的架构；②给出基于射线跟踪的场景评估方法和资源分配优化方案；③展示在多子带可用时优于传统频率整合架构的性能。

**🔧 技术方法**

使用射线跟踪仿真（Sionna RT）获取室内/室外场景的频率依赖MIMO通道；设计全数字频率自适应架构与切换网络；通过基于ADC/DAC资源分配与MIMO维度优化的仿真算法实现动态资源重用。

**📊 数据集**

使用Sionna Ray Tracing生成的代表性室内实验室和城市宏观户外场景的射线跟踪通道数据集。

**📈 对比分析**

将频率自适应、频率分区、频率整合以及理想全天线基线架构在同一场景下进行比对，评价指标包括总比特/秒/Hz、ADC/DAC数量、子带利用率等。结果显示：在所有子带可用时，频率自适应架构比频率整合高约18%；在子带不可用时，通过动态资源重用仍能保持接近最优性能。

**⚠️ 局限性**

主要限制：仿真假设硬件理想（ADC/DAC完美切换、相位一致性），未考虑切换时延、功耗与实际天线布局；仅覆盖FR3子带，未扩展到更广频段；缺乏实际硬件实现与实验验证。

---

## 764. Building Faculty Expertise Ontology using Protege: Enhancing Academic Library Research Services

**arXiv ID:** 2601.07451 | [PDF](https://arxiv.org/pdf/2601.07451v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 765. Surrogate-based Optimization via Clustering for Box-Constrained Problems

**arXiv ID:** 2601.07442 | [PDF](https://arxiv.org/pdf/2601.07442v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 766. R3-RECON: Radiance-Field-Free Active Reconstruction via Renderability

**arXiv ID:** 2601.07484 | [PDF](https://arxiv.org/pdf/2601.07484v1)

**作者:** Xiaofeng Jin `[一作]` (Politecnico di Milano), Matteo Matteucci `[通讯]` (Politecnico di Milano)

**通讯引用:** 6752 | [OpenAlex ID](https://openalex.org/A5003932703)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种不依赖辐射场的主动重建框架 R^3-RECON，利用视角可渲染性字段快速评估候选视角并引导下一步观察。

**💡 创新点**

创新点在于将渲染可渲染性拆解为方向偏差、外观噪声与分辨率三项，并通过轻量化体素统计实现闭式、毫秒级的视角评分，避免了对昂贵的辐射场或 3D 高斯模型的实时维护。

**🔧 技术方法**

主要技术包括：基于斐波那契球面离散的方向统计、Welford 更新的颜色协方差估计、体素化渲染可渲染性表、360° 全景视角评估与最短路径成本融合。

**📊 数据集**

在 Replica-Dense（9 个室内场景）数据集上进行实验，使用统一的 RGB‑D 流式数据收集与 3DGS 重建管线进行评测。

**📈 对比分析**

与 Active‑GS、ActiveGAMER 等基线相比，R^3-RECON 在相同时间/视角预算下实现了更高的 PSNR/SSIM、更低的 LPIPS，并在 GPU 内存和计算成本上显著更轻，采集效率更高。

**⚠️ 局限性**

局限性在于候选视角仅以局部随机采样，容易在大空间或拥挤环境中陷入视角重复，未来需结合更长远或学习型规划来提升采样质量。

---

## 767. Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data

**arXiv ID:** 2601.07474 | [PDF](https://arxiv.org/pdf/2601.07474v1)

**作者:** Youngmin Oh `[一作]` (Electronics and Telecommunications Research Institute), Jung Uk Kim `[通讯]` (Kyung Hee University)

**通讯引用:** 878 | [OpenAlex ID](https://openalex.org/A5036936141)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于任务原型的知识检索框架，实现部分标注多任务学习（MTPSL）中的稳健知识迁移。

**💡 创新点**

①在任务原型中嵌入任务特征并用关联知识生成（AKG）损失量化任务关联；②设计知识检索 Transformer 利用任务亲和得分自适应提升特征表示，无需依赖未标注任务的伪标签；③通过向量量化与任务一致性损失进一步强化共享表示与任务特征的稳定性。

**🔧 技术方法**

任务原型（多槽学习）、向量量化、任务一致性损失、AKG 损失、知识检索 Transformer、交叉注意力、任务亲和得分、L2 及交叉熵损失组合。

**📊 数据集**

PASCAL-Context（4k 训练/5k 测试）与 NYUD‑v2（795/654）数据集，涵盖语义分割、人体分割、边界、表面法线、深度、显著性等任务。

**📈 对比分析**

与单任务、MTL 基线、半监督、MTPSL、DiffusionMTL 等方法对比。实验表明在一标注和随机标注设置下，本文方法在所有任务上均能获得最优或第二优指标（如 mIoU、maxF、mErr、absErr 等），并且性能稳健不受输入形式影响。

**⚠️ 局限性**

仅能处理训练时已出现的任务；对未知任务的零样本迁移或元学习扩展尚未覆盖；同时模型参数量相对较大，需进一步压缩与加速。

---

## 768. Learning How to Remember: A Meta-Cognitive Management Method for Structured and Transferable Agent Memory

**arXiv ID:** 2601.07470 | [PDF](https://arxiv.org/pdf/2601.07470v1)

**作者:** Sirui Liang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Kang Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种 Meta‑Cognitive Memory Abstraction（MCMA）方法，通过学习记忆抽象的认知技能，提升 LLM 代理在长时程任务中的记忆利用与迁移能力。

**💡 创新点**

创新点在于将记忆抽象视为可学习的元认知策略，使用可学习的 Memory Copilot 并与冻结的 Task Model 分离，从而实现跨任务、跨域的可迁移记忆结构与层次化抽象。

**🔧 技术方法**

核心技术包括：冻结任务模型、基于 Direct Preference Optimization (DPO) 的 Memory Copilot 训练、结构化记忆生成（树、链、自然语言、键值组合）以及多层抽象级别的聚类与选择。

**📊 数据集**

实验数据集涵盖 ALFWorld、ScienceWorld 和 BabyAI 三大长时程文本化嵌入式推理基准，验证了在不同环境和模型规模下的有效性。

**📈 对比分析**

与 No Memory、ReAct、Raw Tra、TRAD、ExpeL 等基线相比，MCMA 在 ALFWorld 上平均提升约 25% 的成功率、在 ScienceWorld 上提升约 8% 奖励，且在未见任务和跨域迁移上保持显著优势。

**⚠️ 局限性**

主要局限包括：训练过程需要构造大量候选抽象并进行评估，导致计算成本较高；抽象层级选择仍依赖手工策略，缺乏完全端到端自适应。

---

## 769. Center-Fed Pinching Antenna System (C-PASS) Aided Wireless Communications

**arXiv ID:** 2601.07424 | [PDF](https://arxiv.org/pdf/2601.07424v1)

**作者:** Xu Gan `[一作]` (University of Hong Kong), Yuanwei Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 33847 | [OpenAlex ID](https://openalex.org/A5076863392)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种中心馈Pinching Antenna System（C-PASS），提出三种操作协议（功率分裂 PS、方向切换 DS、时间切换 TS），并针对每种协议设计联合发射和Pinching Beamforming 的优化算法以最大化系统总速率。

**💡 创新点**

创新点在于：1）利用中心馈将波导信号双向分裂，提升了 DoF（从 1 提升到 2）；2）首次提出三种可实现的操作协议，并给出相应的优化框架；3）在 PS/DS 方案中通过 WMMSE 重构加 Riemannian 梯度下降实现多维度联合优化，在 DS 方案中引入惩罚法处理整数约束；4）在 TS 方案中给出了时分配比例的闭式最优解。

**🔧 技术方法**

主要技术包括：加权最小均方误差（WMMSE）重构、交替优化（AO）、Riemannian 表面梯度下降、惩罚法（处理二进制约束）、MRT 预编码、微调 PA 位移的相位匹配、闭式时间分配。

**📊 数据集**

实验使用基于 28 GHz mmWave 频段的仿真环境，设定 M=2、N=10、λg=λ0/1.4 等参数，随机生成用户位置和波导输入端口分布；未使用公开数据集，全部为仿真数据。

**📈 对比分析**

与传统 END‑fed PASS、随机预编码 baseline1、均匀 PA 配置 baseline2 进行对比。结果显示：PS/DS 在高 SNR 下实现 DoF=2，吞吐率约为传统 PASS 的两倍；TS 在低功率下优于其他协议；相较于 baseline1/2，PS/DS/TS 的总速率提升约 4–5 dB，且随 PA 数量和输入端口数量增加可进一步提升。

**⚠️ 局限性**

局限性：1）仅考虑两用户下行场景，未讨论多用户或上行；2）算法复杂度较高，需多轮迭代；3）硬件实现的能量损耗、实际波导损耗及控制精度未做深入评估；4）对波导长度、输入端口位置的假设较为理想，实际部署可能受限。

---

## 770. BenchSeg: A Large-Scale Dataset and Benchmark for Multi-View Food Video Segmentation

**arXiv ID:** 2601.07581 | [PDF](https://arxiv.org/pdf/2601.07581v1)

**作者:** Ahmad AlMughrabi `[一作]` (Matemàtiques i Informàtica, Universitat de Barcelona), Petia Radeva `[通讯]` (Institut de Neurosciències, Universitat de Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BenchSeg 数据集与基准，包含 25,284 帧多视角食物视频分割注释，并对 20 种先进模型进行评测。

**💡 创新点**

关键创新在于构建全景自由运动视频分割基准，系统评估跨数据集泛化与时空一致性，并证明内存增强/基于 SAM 的混合模型在视角变换下保持高精度。

**🔧 技术方法**

采用 Transformer/CNN 语义分割器、SAM、XMem、XMem2 等内存与提示技术。

**📊 数据集**

结合 FoodSeg103、Nutrition5k、Vegetables & Fruits、MetaFood3D、FoodKit 等公开数据集。

**📈 对比分析**

通过 mAP、Recall、Precision、F1、IoU 等指标比较，发现单帧分割器在新视角退化严重，混合方案（如 SeTR‑MLA+XMem2）相较于 FoodMem 提升约 2.6% mAP，整体显示内存增强可显著提升稳定性。

**⚠️ 局限性**

限制包括数据覆盖不全（主要单菜、受控拍摄）、对快速运动/遮挡鲁棒性不足、模型尺寸与推理速度高，难以在移动端实时部署。

---

## 771. GPU accelerated surface-based gaze mapping for XR experiences

**arXiv ID:** 2601.07571 | [PDF](https://arxiv.org/pdf/2601.07571v1)

**作者:** Charles Javerliat `[一作]` (University Lyon), Guillaume Lavoué `[通讯]` (University Lyon)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种 GPU 加速、基于表面取样的 6DoF 注视密度图生成与实时渲染算法，支持可配置分辨率并与网格分辨率和 UV 映射解耦。

**💡 创新点**

创新点在于引入基于三角形亚样本的 quasi-uniform 采样与 O(1) 索引、可自适应分辨率、通过裁剪相机视锥减少计算、以及可导出基于三角形索引和重心坐标的原始数据。

**🔧 技术方法**

使用了 GPU 并行计算、z‑buffer 视图剔除、高斯光锥投影、矩阵裁剪技术、以及 PLUME 与 XREcho 框架集成。

**📊 数据集**

在没有 UV 映射、重叠 UV 以及顶点分布不均的“天使雕像+立方体+兔子”测试场景和一个包含 10M 三角形的压力测试场景进行实验。

**📈 对比分析**

相较于现有体素、UV 基础或像素投影方法，本文在 6,000 次注视点下生成速度约 3 倍快，10M 三角形场景生成时间低于 1 分钟，且对网格分辨率不敏感。

**⚠️ 局限性**

限制在于需要在实验后重采样以获得更高分辨率、对大规模场景仍需要显存和时间开销，以及对动态场景的实时更新仍需进一步优化。

---

## 772. FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments

**arXiv ID:** 2601.07558 | [PDF](https://arxiv.org/pdf/2601.07558v1)

**作者:** Chen Feng `[一作]` (Hong Kong University of Science and Technology), Boyu Zhou `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2687 | [OpenAlex ID](https://openalex.org/A5101982552)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于大型预训练模型（Foundation Models）的无人机三维结构扫描系统 FlyCo，能够在未知开放环境中仅凭简短文本或少量视觉标注实现全自主、高效且安全的目标结构扫描。

**💡 创新点**

创新点主要包括：① 将 FMs 与三阶段感知‑预测‑规划循环相结合，构建可扩展、模块化的系统架构；② 设计跨模态感知模块（SAM2+VLM融合与三维重构），实现对目标的稳健定位与跟踪；③ 提出多模态表面预测网络，利用视觉、语言与点云信息共同完成目标几何补全，并提供自适应稠密化方案；④ 开发预测感知的层级规划框架，兼顾全局覆盖一致性、局部实时避障与轨迹优化；⑤ 通过自动化多模态数据生成流水线，构建大规模无标注训练集。

**🔧 技术方法**

核心技术包括：基础模型（SAM2、DINO、SigLIP、BEiT3 等）融合、三维点云聚类与投影、交叉模态细化、点云与视觉文本融合注意力网络、基于稀疏感知的几何稠密化、预测感知的全局视角生成与 ATSP 分解、实时局部轨迹优化（MINCO）及软约束方法。

**📊 数据集**

数据集：自动合成的多模态数据集（约 70K 个 3D 资产，来源于 Objaverse 等），每个样本包含点云、RGB 图、相机位姿与文本描述；在实际实验中使用真实户外环境中的四个目标（拱桥、音乐厅、城堡门、红砖建筑）进行验证。

**📈 对比分析**

与现有最先进方法相比，FlyCo 在公开仿真与真实飞行实验中取得显著优势：飞行时间缩短 1.25‑3 倍、覆盖率提升 4.3‑56.2% 以上、成功率提升 11.6‑50.1% 以上，并且仅需要简短文本与少量点击，显著降低人工投入；系统在大规模结构、复杂障碍与动态环境中均保持高效率与安全。

**⚠️ 局限性**

局限性：① 对极其偏离训练分布的新型结构或极端视觉条件的鲁棒性尚待进一步验证；② 预测网络及稠密化在边缘设备上仍耗时较高，需借助地面服务器；③ 依赖用户提供初始文本/点击，若标注极模糊可能导致误检；④ 系统整体复杂度高，部署与维护成本较大。

---

## 773. Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning

**arXiv ID:** 2601.07641 | [PDF](https://arxiv.org/pdf/2601.07641v1)

**作者:** Jiaxuan Lu `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongzhan Zhou `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 718 | [OpenAlex ID](https://openalex.org/A5112578541)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了测试时工具演化（TTE）框架，使大语言模型在推理过程中能动态合成、验证并演化可执行工具，替代传统的静态工具库；

**💡 创新点**

创新点在于将工具生成与推理过程耦合，实现在线工具演化与自适应，突破了工具稀疏、异构与不可预见性限制；

**🔧 技术方法**

使用的技术包括链式思维+工具调用、语义检索+重排序、代码生成与执行验证、原子化拆分与冗余检测、闭环强化学习式工具库管理；

**📊 数据集**

数据集为SciEvo基准，包含1,590个科学推理实例与925个自演化工具，覆盖物理、化学、数学、材料四大领域；

**📈 对比分析**

与多种基线（Basic-COT、Basic-POT、Creator、KTCE、CheMatAgent）比较，TTE-Zero在SciEvo上准确率0.62，TRR@1≈1.0，TTE-Adapt在跨域任务上准确率提升至0.595/0.618并显著提高新工具复用率；

**⚠️ 局限性**

局限包括推理时延高、对大型LLM的编码能力依赖、可执行代码安全与沙箱约束、以及较小模型表现不足。

---

## 774. Proof of Time: A Benchmark for Evaluating Scientific Idea Judgments

**arXiv ID:** 2601.07606 | [PDF](https://arxiv.org/pdf/2601.07606v1)

**作者:** Bingyang Ye `[一作]` (Harvard University), Danielle S. Bitterman `[通讯]` (Mass General Brigham)

**通讯引用:** 3408 | [OpenAlex ID](https://openalex.org/A5039369605)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为 pot 的时间分区半可验证基准框架，冻结截止前的证据并让 LLM 在离线沙箱中预测截止后可观测的信号（如引用数、奖项、研究方向变化和基准进展），从而实现大规模、可重复的科学想法评估。

**💡 创新点**

创新点包括：①时间分区设计将可验证的未来信号与当下证据分离，消除泄漏与人工注释；②离线沙箱与工具使用可度量的交互流程，使代理系统的增益可量化；③在四大任务领域（引用预测、奖项评估、研究者演化、前沿基准）构建 3 万+ 自动可评分实例；④系统化比较零射、代理与结构化提示三种推理策略。

**🔧 技术方法**

技术手段包括：大语言模型（Claude、Gemini、GPT‑5 系列）在单一 ReAct 代理循环中使用文件读取、Python 计算、文本编辑等工具；离线沙箱隔离网络访问；按消息数限制（15/30/50）控制推理预算；使用 LLM‑as‑judge 对代理执行轨迹进行错误分类；自动化评估脚本通过后截止时间的外部信号（Google Scholar、会议奖项、公开榜单）计算精确匹配准确率。

**📊 数据集**

数据集来源于公开元数据：2021‑2024 期刊/会议（ACL、NAACL、EMNLP）论文的标题、摘要、作者及历史引用；2025 年的新论文作为预测对象；教师论文集（2021‑2024）用于研究演化任务；公开基准排行榜（如 GLUE、SuperGLUE 等）截至 2025 年 10 月的分数，用于 SOTA 预测；所有实例自动生成并可随时刷新。

**📈 对比分析**

比较方法：在同一任务定义下，对比三种求解器（零射、代理、代理+结构化提示），并在 15/30/50 条消息预算下评估；使用多种大模型做基准；对比前后截止时间的准确率变化。结果显示：代理在 Faculty 任务上提升约 60%，在 Citations 上提升约 10pp；Awards 任务增益有限，SOTA 任务几乎已达上限；消息预算提升总体准确率，但增益呈现模型和任务依赖；结构化提示对 Claude 有正面影响，GPT 有中性或略差影响。后截止评估会显著改变模型排名，凸显时间分区的重要性。

**⚠️ 局限性**

局限性：①使用引用数、奖项、基准分数等外部信号作为“想法质量”的代理，易受可见性、社区偏好等噪声影响；②评估结果依赖截止时间选择和数据采集方式；③离线沙箱剥离实时检索可能低估真实助手性能；④仅研究单一 ReAct 代理和固定消息预算，其他代理架构或终止策略可能得到不同表现；⑤部分任务的难度与标签质量可能随时间变化。

---

## 775. OODEval: Evaluating Large Language Models on Object-Oriented Design

**arXiv ID:** 2601.07602 | [PDF](https://arxiv.org/pdf/2601.07602v1)

**作者:** Bingxu Xiao `[一作]` (Northwestern Polytechnical University), Yepang Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2407 | [OpenAlex ID](https://openalex.org/A5084868951)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了29种大型语言模型在面向对象设计（OOD）任务中的表现，并首次构建了基于手工收集的50个OOD案例的OODEval benchmark及其人类评估版OODEval‑Human。

**💡 创新点**

创新点包括：①提出OODEval与OODEval‑Human两大benchmark，填补OOD评估数据缺口；②设计了CLUE（Class Likeness Unified Evaluation）统一评估指标，兼顾结构与语义细粒度；③通过贝叶斯优化调参验证CLUE与人工评分高度相关，证明其可靠性。

**🔧 技术方法**

采用的技术包括：基于PlantUML解析器提取类图元数据；CLUE指标基于Hungarian算法实现元素匹配与语义相似度计算；Pass@k评估语法正确率；贝叶斯优化对CLUE权重进行全局调优；以及多种Prompt设计与统一评价框架。

**📊 数据集**

使用的数据集包括：①OODEval（50个多难度OOD任务，包含需求文本与参考类图）；②OODEval‑Human（940份本科生提交的类图与教师评分）。

**📈 对比分析**

通过将CLUE与BLEU、CodeBERTScore、NINHS等传统指标与人工评分进行相关性对比，并与人类本科生平均水平做直接比较。实验结果显示：最佳LLM Qwen3‑Coder‑30B在整体CLUE得分上接近平均本科生，但仍落后于最佳人类；模型规模、代码预训练和指令微调是提升性能的关键因素。

**⚠️ 局限性**

局限性包括：①benchmark规模相对有限，缺少更大规模、多样化的OOD实例；②单一参考实现可能导致评估偏倚；③存在模型实现差异与潜在数据泄漏风险；④对不同编程语言或工具的泛化能力未作充分验证。

---

## 776. A Unified Framework for Emotion Recognition and Sentiment Analysis via Expert-Guided Multimodal Fusion with Large Language Models

**arXiv ID:** 2601.07565 | [PDF](https://arxiv.org/pdf/2601.07565v1)

**作者:** Jiaqi Qiao `[一作]` (Dalian University of Technology), Yu Liu `[通讯]` (Dalian University of Technology)

**通讯引用:** 66413 | [OpenAlex ID](https://openalex.org/A5100345666)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个统一的情感识别与情绪分析框架 EGMF，能够同时处理情绪分类与情绪强度回归任务。

**💡 创新点**

创新点包括三种功能专一的专家网络（细粒度本地专家、语义关联专家、全局上下文专家）以及层次化动态门控的自适应融合机制，并将增强后的多模态特征通过伪标记注入和提示式条件化与大型语言模型耦合，实现统一的生成式预测。

**🔧 技术方法**

技术上结合了跨模态双向注意力、多尺度专家网络、层次动态门控、伪标记注入、提示工程以及 LoRA 参数高效微调，以利用 LLM 的推理与生成能力。

**📊 数据集**

使用四个公开双语数据集：MELD、CHERMA（情绪分类）以及 MOSEI、SIMS‑V2（情绪强度回归），涵盖英语与中文。

**📈 对比分析**

与现有多模态方法（如 MulT、MAG‑BERT、Self‑MM 等）以及 LLM‑基线（UniSAGPT2、UniSAT5 等）对比，EGMF 在所有任务中均取得最优或接近最优的指标（例如在 MOSEI 上 F1 87.09%，在 CHERMA 上加权 F1 73.90%，在 MELD 上加权 F1 65.57%），并显示出更强的跨语种鲁棒性。

**⚠️ 局限性**

限制包括对中文数据的 LoRA 微调效果不佳、对视觉/音频模态的贡献仍相对有限，以及在极端多模态场景下仍需进一步验证模型的可扩展性和解释性。

---

## 777. TFEC: Multivariate Time-Series Clustering via Temporal-Frequency Enhanced Contrastive Learning

**arXiv ID:** 2601.07550 | [PDF](https://arxiv.org/pdf/2601.07550v1)

**作者:** Zexi Tan `[一作]` (Guangdong University of Technology), Yiqun Zhang `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 714 | [OpenAlex ID](https://openalex.org/A5100329232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种新的多变量时间序列聚类框架 TFEC，利用时频共增强与双路径对比学习实现无监督的高质量聚类。

**💡 创新点**

创新点包括：1) 时频共增强（CoEH）在保持时间连贯性的同时通过频域邻居混合提升特征表达；2) 高置信度伪标签对比学习（PGCL）显式利用聚类信息构造正负样本；3) 读取修正路径（READ）通过掩码重建稳定表征，两者协同提高聚类友好性与表征鲁棒性。

**🔧 技术方法**

技术手段：自监督对比学习、伪标签引导的聚类采样、频域FFT与逆FFT混合、掩码重建、自编码器、K‑means 初始化、联合对比-重建损失。

**📊 数据集**

使用六个 UEA 公开数据集：AtrialFibrillation、ERing、RacketSports、Libras、StandWalkJump、NATOPS，覆盖从短时序到长时序、2~24 类的多样场景。

**📈 对比分析**

与基线 K‑means、以及四种 SOTA 深度聚类方法（TimesURL、UNITS、DropPatch、FCACC）在 ACC、NMI、F1 上对比，TFEC 在所有数据集均实现最优或近最优结果，平均在 NMI 上提升 4.48%。

**⚠️ 局限性**

局限性：对噪声的鲁棒性尚待提升；计算成本相对较高，尤其在长序列与高维特征上；在部分数据集（如 RacketSports、NATOPS）对比提升幅度有限。

---

## 778. Estimators for Substitution Rates in Genomes from Read Data

**arXiv ID:** 2601.07546 | [PDF](https://arxiv.org/pdf/2601.07546v1)

**作者:** Shiv Pratap Singh Rathore `[一作]` (Indian Institute of Science), Navin Kashyap `[通讯]` (Indian Institute of Science)

**通讯引用:** 971 | [OpenAlex ID](https://openalex.org/A5059452861)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究如何仅凭未组装的噪声测序读数估计两条基因组间的置换率，并提出基于k‑mer计数的两类估计器（k=1 与 k=30），其中k=1的估计器给出了理论置信界。

**💡 创新点**

创新点在于把传统需要完整k‑mer频数的置换率估计方法迁移到仅观测噪声读数的情形，提出可直接从读数推断k‑mer分布的算法，并对k=1估计器给出了非平凡的理论保证。

**🔧 技术方法**

使用了k‑mer 统计、子序列覆盖模型、Hoeffding 与 McDiarmid 不等式的概率分析以及数值解法来构造估计器。

**📊 数据集**

实验数据包括合成的 i.i.d. 序列以及人类 T2T‑CHM13v2.0 的三段真实基因组（D‑easy、D‑hardest、Chr9‑HSat）。

**📈 对比分析**

与已有的基于k‑mer计数的估计器（p_AH）以及仅 k=1 的估计器进行相对误差箱形图比较。结果显示 k=1 估计器在高测序错误率或高重复序列下更稳健，而 k=30 估计器在低置换率和低重复率下表现更好；两者在极端参数区间会失稳。

**⚠️ 局限性**

主要限制包括：k=30 估计器对测序错误高度敏感且缺乏理论保证；两类估计器在极低或极高置换率、极高重复率时可能失稳；目前不考虑插入/缺失等其他突变类型。

---

## 779. ViewMorpher3D: A 3D-aware Diffusion Framework for Multi-Camera Novel View Synthesis in Autonomous Driving

**arXiv ID:** 2601.07540 | [PDF](https://arxiv.org/pdf/2601.07540v1)

**作者:** Farhad G. Zanjani `[一作]` (Qualcomm AI Research), Amirhossein Habibian `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扩散模型的多视角图像增强框架 ViewMorpher3D，用于对 3D 高斯散点渲染的图像进行后处理，以提升图像质量和多视角一致性。

**💡 创新点**

创新点包括：①将 3D 坐标图和 Plücker 视角嵌入作为条件直接注入扩散模型，实现几何先验的显式利用；②设计可变视角输入/输出的无监督架构，支持任意数量的参考和目标视角，保证跨视角与时序一致性；③采用稀疏采样的像素级监督策略，在保持显存低的同时实现全局一致性。

**🔧 技术方法**

使用了 SD‑Turbo 单步扩散模型、冻结的 VAE 编码器、辅助轻量级编码器（用于 C‑maps 与 Plücker 嵌入）、3D 自注意力的 UNet、LoRA 方式对 VAE 解码器进行微调，以及混合的潜在层与像素层损失。

**📊 数据集**

在四个驾驶相关数据集上评估：EUVS（大视角外推），Para‑Lane（跨车道场景），nuScenes（多相机动态驾驶），以及 DL3DV（有界/无界场景）.

**📈 对比分析**

与 DiFix3D+、3DGS‑Enhancer 等最先进方法进行对比，使用 PSNR、SSIM、LPIPS 三项指标，实验结果表明 ViewMorpher3D 在所有稀疏/外推设置下均取得更高分数（PSNR 提升 1–2 dB，LPIPS 降低约 50%），并在跨视角与时序一致性方面表现更佳。

**⚠️ 局限性**

局限性：①仅为后处理，未对 3D 结构进行改进；②需要先生成 3DGS 渲染和对应的 C‑maps，额外计算成本；③在极端稀疏或极端外推的视角下仍可能出现轻微漂移或失真；④扩散模型虽然单步推理速度快，但仍受 GPU 显存限制。

---

## 780. Studying the Role of Synthetic Data for Machine Learning-based Wireless Networks Traffic Forecasting

**arXiv ID:** 2601.07646 | [PDF](https://arxiv.org/pdf/2601.07646v1)

**作者:** José Pulido `[一作]` (Universidad de Málaga), Raquel Barco `[通讯]` (Universidad de Málaga)

**通讯引用:** 2813 | [OpenAlex ID](https://openalex.org/A5039871509)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在无线网络流量预测中使用合成数据提升机器学习模型表现的可行性和效果。

**💡 创新点**

提出一种基于GAN的合成流量生成方法，并证明合成数据与真实数据混合训练能显著提升预测精度。

**🔧 技术方法**

使用了生成对抗网络（GAN）进行数据生成，长短期记忆网络（LSTM）和ARIMA模型进行流量预测。

**📊 数据集**

利用来自校园Wi‑Fi 802.11网络的真实流量日志作为基准数据集，并生成对应的合成数据集。

**📈 对比分析**

通过将模型分别仅用真实数据、仅用合成数据以及混合数据训练，并对比MAE、RMSE等指标，发现混合训练模型在RMSE上平均降低约15%，性能最优。

**⚠️ 局限性**

合成数据在捕捉极端流量模式上仍有限，生成过程计算成本较高，且需要进一步验证在不同网络环境下的泛化能力。

---

## 781. PlaM: Training-Free Plateau-Guided Model Merging for Better Visual Grounding in MLLMs

**arXiv ID:** 2601.07645 | [PDF](https://arxiv.org/pdf/2601.07645v1)

**作者:** Zijing Wang `[一作]` (Northeastern University), Hinrich Schütze `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了无训练的平衡点引导模型合并方法（PlaM），通过在多模态大语言模型的后期层插入基础语言模型参数，恢复被细调削弱的文本推理能力并提升视觉定位；

**💡 创新点**

创新点在于先用层级视觉token遮蔽揭示三阶段视觉信息利用模式，然后利用平稳期定位，只在后期层合并原始LM参数，从而既保留跨模态对齐又加强文本推理；

**🔧 技术方法**

技术包括层级视觉token遮蔽、线性插值参数合并、注意力权重和热图分析等训练‑free干预手段；

**📊 数据集**

在五种主流MLLM（如LLaVA‑v1.5‑7B、Qwen2.5‑VL‑3B‑Instruct、Qwen3‑VL‑8B‑Instruct 等）上，使用九个视觉语言基准（MMStar、MMMU、MME、MMBench、GQA、RealWorldQA、SEED‑Bench‑2‑Plus、POPE 等）进行评测；

**📈 对比分析**

与原始细调模型、早期/中期/全层合并策略对比，PlaM 在所有模型和基准上均实现最高分，平均提升数个百分点，尤其在需要后期语义决策的任务（MMStar、MME、MMMU）表现尤为显著；

**⚠️ 局限性**

局限性在于需要手动调节合并起始层k0，且采用的是简单的线性插值合并策略，复杂合并方法可能进一步提升效果。

---

## 782. Simple Power Analysis of Polynomial Multiplication in HQC

**arXiv ID:** 2601.07634 | [PDF](https://arxiv.org/pdf/2601.07634v1)

**作者:** Pavel Velek `[一作]` (Czech Technical University in Prague), Jiří Buček `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 158 | [OpenAlex ID](https://openalex.org/A5065973581)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对后量子密码学系统HQC的乘法实现，提出并验证了单次跟踪简单功率分析（SPA）攻击，成功恢复64位私钥片段。

**💡 创新点**

创新点在于利用针对缓存时间攻击的对策引入的查找表全遍历方式，从单个功耗轨迹中提取四位私钥，且不依赖任何机器学习或侧信道预处理。

**🔧 技术方法**

使用ChipWhisperer-Lite平台的功耗采集、Python脚本自动化分析，并结合原始Karatsuba实现的基础乘法算法。

**📊 数据集**

数据集为10,000组随机生成的64位a与b值，生成相应功耗轨迹。

**📈 对比分析**

通过与原实现(base_mul)、去查表实现(base_mul2)以及去缓存计时对策实现(base_mul3)进行时延对比，发现base_mul2与base_mul3分别比原实现快约2.3倍和6.9倍。

**⚠️ 局限性**

局限在于攻击仅针对特定实现，对其他优化或硬件变体可能无效，且仍需考虑多次攻击以恢复完整私钥。

---

## 783. Clipped Affine Policy: Low-Complexity Near-Optimal Online Power Control for Energy Harvesting Communications over Fading Channels

**arXiv ID:** 2601.07622 | [PDF](https://arxiv.org/pdf/2601.07622v1)

**作者:** Hao Wu `[一作]` (Zhejiang Gongshang University), Guanding Yu `[通讯]` (Zhejiang University)

**通讯引用:** 9440 | [OpenAlex ID](https://openalex.org/A5079187892)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了两种低复杂度的裁剪仿射功率控制策略，并基于它们设计了强化学习算法，用于能量采集无线通信中的在线功率调度。

**💡 创新点**

创新点在于对贝尔曼方程相对价值函数进行线性逼近，得到通用的裁剪仿射政策，并将其与RL相结合，实现参数极低（≤5）且吞吐量与最优策略几乎相同；同时扩展至能量/信道一跳前瞻的上下文设置。

**🔧 技术方法**

采用了线性相对价值函数近似、裁剪仿射策略、基于平均奖励的经验回放强化学习、Jensen不等式分析以及离散化MDP与策略迭代等技术。

**📊 数据集**

仿真使用了伯努利、指数、均匀三种能量到达分布以及Rayleigh信道，并在不同NMCR和NSNR组合下生成1000个episode、每个10⁴步的模拟数据。

**📈 对比分析**

通过与最优在线策略（策略迭代得到）以及现有RL和启发式算法比较，采用在线乘法因子和相对性能损失指标，结果显示鲁棒裁剪仿射策略在所有场景下平均损失<1%、最大损失<2%，并在前瞻场景下进一步提升性能。

**⚠️ 局限性**

局限在于模型假设理想电池与能量到达、持续的Rayleigh信道以及i.i.d.过程；在非理想电池、能量/信道相关性强的实际环境中仍需进一步改进。

---

## 784. Diffusion in SPAD Signals

**arXiv ID:** 2601.07599 | [PDF](https://arxiv.org/pdf/2601.07599v1)

**作者:** Lior Dvir `[一作]` (Technion Israel Institute of Technology), Yoav Y. Schechner `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文推导了单光子雪崩二极管（SPAD）原始信号在给定光子通量下的似然函数，并基于该似然构建了评分函数，用于通过扩散模型解决逆问题。

**💡 创新点**

创新点在于：① 将死亡时间（dead time）和事件间隔的统计学建模与非线性前向模型结合，得到完整的事件级似然；② 通过域适配将扩散模型输出映射到物理光子通量；③ 将扩散后向采样（Diffusion Posterior Sampling）与SPAD似然梯度相结合，实现了自适应的图像重建。

**🔧 技术方法**

使用的技术包括：泊松-埃朗格分布推导、Tweedie公式与扩散模型的评分网络、Annealed Langevin Dynamics、域适配（Affine 变换）以及基于梯度引导的后向扩散步骤。

**📊 数据集**

实验数据集为基于 FFHQ（高分辨率人脸图像）的合成灰度图像，通过模拟不同光照（lux）和事件数（N_det）生成 SPAD 事件序列。

**📈 对比分析**

与简单重建方法相比，实验显示在不同光照下重建图像的 PSNR 约提升 2-5 dB，LPIPS 指标亦明显下降，证明该方法在低光/高噪声条件下具有更好的重建质量。

**⚠️ 局限性**

主要限制包括：① 依赖于预训练的扩散模型，若目标场景与训练分布差异大可能失效；② 计算量较大，尤其是每像素事件序列的似然梯度求解；③ 对极低事件数（如单事件）重建仍受限，需进一步提升模型鲁棒性。

---

## 785. Pheromone-Focused Ant Colony Optimization algorithm for path planning

**arXiv ID:** 2601.07597 | [PDF](https://arxiv.org/pdf/2601.07597v1)

**作者:** Yi Liu `[一作]` (Fudan University), Chun Ouyang `[通讯]` (Fudan University)

**通讯引用:** 5731 | [OpenAlex ID](https://openalex.org/A5075868200)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Pheromone-Focused Ant Colony Optimization（PFACO）算法，用三种策略改进传统蚁群算法的路径规划性能。

**💡 创新点**

创新点包括：基于起点–终点欧氏距离的非均匀信息素初始化（ADPI）、全局优秀解复制强化策略（PSPRS）以及前瞻转弯惩罚的转弯优化策略（LTOS），使信息素集中在潜在最优区域，提升收敛速度与解质量。

**🔧 技术方法**

使用了蚁群优化、信息素更新、启发式函数、非均匀初始化、复制强化和转弯惩罚等技术。

**📊 数据集**

使用了三种不同尺寸（10×10、15×15、20×20）的格点地图数据集，每种地图随机生成100个起终点实例，包括无障碍、单障碍和混合障碍地图。

**📈 对比分析**

通过与 A*、AS、EliteACO、MMACO、NCAACO、IHMACO 等算法在平均路径长度、平均耗时、转弯数、标准差、成功率和路径改进率等指标上进行对比。PFACO 在所有规模下平均路径最短、成功率 100%、耗时最低、方差最小、收敛速度最快，整体性能优于对比算法。

**⚠️ 局限性**

局限性在于大规模地图上计算成本较高，执行时间随地图尺寸显著增长；在动态或极大规模环境中可扩展性和实时性仍需进一步提升。

---

## 786. A Multimodal Dataset of Student Oral Presentations with Sensors and Evaluation Data

**arXiv ID:** 2601.07576 | [PDF](https://arxiv.org/pdf/2601.07576v1)

**作者:** Alvaro Becerra `[一作]` (Universidad Autonoma de Madrid), Roberto Daza `[通讯]` (Universidad Autonoma de Madrid)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文开发并公开了SOPHIAS多模态数据集，记录了65名本科生和研究生在真实课堂中进行的口头演讲，涵盖高清视频、音频、眼动、心率、手势、互动日志、PDF幻灯片以及教师、同伴和自评的评估结果。

**💡 创新点**

创新点在于：①在自然课堂环境下大规模采集多模态数据；②同步整合心率、眼动、手表传感器等生理与行为信号；③采用HuMLaS对原始标签进行精细重标注，提供高质量的时序行为数据；④提供完整评估体系（教师、同伴、自评）与细粒度评分，支持跨模态学习分析。

**🔧 技术方法**

使用技术包括：edBB平台和Microsoft Teams捕获音视频；Watch-DMLT获取Fitbit Sense心率、加速度和陀螺；Tobii Pro Glasses 3采集眼动和第一视角视频；OpenCV、Dlib与RetinaFace实现人脸检测与5点关键点提取；WHENet完成头位估计；FFT跨频率同步音频；faster-whisper实现语音转写和名称去除；HuMLaS进行人机标注；Python/NumPy/TensorFlow/PyTorch等编程与机器学习框架。

**📊 数据集**

采用的主要数据集是本研究生成的SOPHIAS数据集，包含12小时、385 GB的多模态记录，共50个演讲（46个本科单人演讲和4个研究生组演讲），并提供完整的评估与标注文件。

**📈 对比分析**

通过教师、同伴和自评三方评分的Gwet’s AC2一致性和Pearson相关系数评估，发现本科生阶段教师与同伴之间AC2≈0.60，整体一致性0.66；研究生阶段教师与同伴AC2≈0.49，教师与自评AC2≈0.55，说明同伴评分偏好；未给出具体机器学习模型性能，仅呈现评估指标的一致性和相关性。

**⚠️ 局限性**

局限性包括：①受受试者同意限制，部分模态（如眼动、心率、部分视频）缺失；②数据来源单一高校，外部泛化性受限；③同伴评估存在偏差和自评偏高；④未提供长期学习跟踪或模型验证，缺乏因果推断。

---

## 787. Order in the Evaluation Court: A Critical Analysis of NLG Evaluation Trends

**arXiv ID:** 2601.07648 | [PDF](https://arxiv.org/pdf/2601.07648v1)

**作者:** Jing Yang `[一作]` (Technische Universität Berlin), Vera Schmitt `[通讯]` (Technische Universität Berlin)

**通讯引用:** 121 | [OpenAlex ID](https://openalex.org/A5063295175)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过大规模自动信息抽取，对2020‑2025年4大NLP会议共14,171篇NLG论文的评估方法进行系统梳理与量化分析，揭示评估生态中的“指标惯性”“人类评估与LLM评估偏离”与“验证缺失”等问题。

**💡 创新点**

创新点在于：①使用多模型LLM抽取+多数投票+人工验证构建高质量结构化数据集；②提出频率和似然比两种度量，量化任务‑指标‑评估关联；③首次用统计学方式评估LLM‑Judge（LaaJ）与人类评估的相关性，揭示两者关注信号不同。

**🔧 技术方法**

技术包括：PDF‑>文本提取(GROBID)、多模型LLM（DeepSeek‑R1、GPT‑OSS‑120B、Qwen3‑235B‑A22B‑Instruct）问答式抽取、投票融合、术语归一化（模糊匹配）、人类双人标注校验、频率/似然比计算、热图与相关性分析。

**📊 数据集**

数据集为ACL Anthology中2020‑2025年ACL、EMNLP、NAACL、INLG的所有论文，经过任务筛选后保留30个主任务共3,334篇；在这些论文中进一步提取评估方法、指标与标准等元数据。

**📈 对比分析**

比较方法：统计评估方法使用频率、任务关联度（LR）、人类评估与LaaJ的对比相关性（Spearman/ Pearson）。结果显示：LaaJ在对话生成等开放式任务快速普及，但与人类评估的相关系数普遍低于0.5；传统n‑gram指标在机器翻译中仍占主导，显示指标惯性。

**⚠️ 局限性**

局限性包括：①仅对四个二元问题进行一致性评估，未细化术语抽取的交叉验证；②术语归一化方法简化，可能忽略同义词背后差异；③任务分辨率不足，无法深入子任务的评估细节；④人类验证样本量有限，可能影响整体抽取质量。

---

## 788. SALT-KG: A Benchmark for Semantics-Aware Learning on Enterprise Tables

**arXiv ID:** 2601.07638 | [PDF](https://arxiv.org/pdf/2601.07638v1)

**作者:** Isaiah Onando Mulang `[一作]` (SAP SE), Johannes Hoffart `[通讯]` (SAP SE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SALT-KG benchmark，扩展 SALT 数据集，在多表企业数据中加入结构化知识图谱（OBKG）作为语义层，以实现语义感知的表格学习。

**💡 创新点**

创新点在于将业务表格字段与声明性元数据（字段描述、关系、对象类型）通过知识图谱显式关联，使模型能在统计关联之外利用语义上下文进行推理。

**🔧 技术方法**

采用文本嵌入（text‑embedding‑3‑large）+ PCA 进行语义特征编码，并与表格特征进行早期融合；评估了 XGBoost、LightGBM、CatBoost、CARTE、AutoGluon、GraphSAGE 等传统与基于深度学习的基线模型。

**📊 数据集**

基于原始 SALT 交易数据（多表销售订单）以及从该数据生成的 OBKG（约 990 个字段、1,954 个语义对象类型）。

**📈 对比分析**

通过将语义特征拼接到表格特征后训练基线模型，发现对大多数模型的整体排名影响有限，但树模型稳定、深度学习模型略有提升；GraphSAGE 表现不稳定，说明当前模型对语义上下文的利用不足。

**⚠️ 局限性**

局限在于 OBKG 的语义深度有限，缺乏高阶类层次与跨实体抽象，导致语义信息难以在关系网络中充分传播，提升空间受限。

---

## 789. Robust Multicentre Detection and Classification of Colorectal Liver Metastases on CT: Application of Foundation Models

**arXiv ID:** 2601.07585 | [PDF](https://arxiv.org/pdf/2601.07585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 790. GeoMotionGPT: Geometry-Aligned Motion Understanding with Large Language Models

**arXiv ID:** 2601.07632 | [PDF](https://arxiv.org/pdf/2601.07632v1)

**作者:** Zhankai Ye `[一作]` (Florida State University), Xin Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种将离散动作分割与大型语言模型嵌入统一对齐的框架，通过显式正交约束实现两种模态间几何一致性；

**💡 创新点**

创新点在于：①将正交性作为统一几何基底，直接在动作代码表和LLM嵌入空间中施加正交正则；②使用可微的 Gumbel‑Softmax 解码器式量化器；③采用稀疏投影保持几何结构并实现两阶段正交正则；

**🔧 技术方法**

核心技术包括：Gumbel‑Softmax 量化器、稀疏投影映射、两阶段正交正则（正交与利用率正则）、自监督重建与对比损失；

**📊 数据集**

在 HumanML3D 数据集上进行实验，使用 GPT‑2、Qwen‑3‑0.6B 和 LLaMA‑3.2‑1B 进行 fine‑tuning；

**📈 对比分析**

与现有方法相比，平均分数提升约 20%–22%，在 R‑Precision、BLEU、ROUGE、CIDEr、BERTScore 等指标上均优于 MotionGPT3 等前沿模型；

**⚠️ 局限性**

局限性包括：仅评估动作理解任务而非生成；正交约束可能不足以捕获更细粒度的几何关系；缺乏对不同几何约束的深入探讨。

---

## 791. PARL: Position-Aware Relation Learning Network for Document Layout Analysis

**arXiv ID:** 2601.07620 | [PDF](https://arxiv.org/pdf/2601.07620v1)

**作者:** Fuyuan Liu `[一作]` (Unisound AI Technology Co.Ltd), Junnan Zhu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个纯视觉的 OCR‑free 文档布局分析框架 PARL，用位置感知关系学习完成布局检测与分类。

**💡 创新点**

双层关系建模：BSP‑DA 将全局空间关系嵌入可变形注意力；GRC 通过动态图注意网络对预测结果进行上下文细化。

**🔧 技术方法**

在 DFINE（改进的 Deformable DETR）基础上，结合 BSP‑DA、GRC、Varifocal Loss、AdamW 等技术。

**📊 数据集**

在 M6Doc、DocLayNet 与 D4LA 三个公开基准数据集上进行实验验证。

**📈 对比分析**

与多模态与单模态基线对比，M6Doc 上 mAP 72.9% 超越 70.3%，DocLayNet 上 mAP 81.9% 超越 DINO 77.7%，仅 65M 参数，效率显著。

**⚠️ 局限性**

纯视觉模式难以识别依赖文本语义的类别，且仅处理单页布局，无法捕捉跨页整体上下文。

---

## 792. GAP-Net: Calibrating User Intent via Gated Adaptive Progressive Learning for CTR Prediction

**arXiv ID:** 2601.07613 | [PDF](https://arxiv.org/pdf/2601.07613v1)

**作者:** Ke Shenqiang `[一作]`, Hua Qingsong `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了GAP-Net，一种多层次门控的序列行为建模框架，用于提升CTR预测的准确性和鲁棒性。

**💡 创新点**

创新点在于三层门控（ASGA、GCQC、CGDF）逐级抑制噪声、动态校正用户意图并自适应融合多视角行为，从而解决注意力漏斗、静态查询假设和硬融合等瓶颈。

**🔧 技术方法**

采用了自适应稀疏门控注意力、门控级联查询校准、上下文门控去噪融合以及SwiGLU‑FFN等门控机制，并在Transformer‑style注意力基础上做了门控扩展。

**📊 数据集**

实验数据集包括自研的XMart（含点击与购买两任务）和公开的KuaiVideo（短视频CTR），均为真实工业日志。

**📈 对比分析**

与DIN、ETA、SDIM等SOTA模型对比，在XMart和KuaiVideo上实现了AUC、NDCG和MAP均提升约0.5‑1%，并在在线A/B测试中提升GMV、CVR、V2P等业务指标。

**⚠️ 局限性**

局限性包括门控结构复杂度较高，训练时需额外的门控参数；对极端短序列的优势不明显；以及在极端动态场景下门控策略的可解释性尚待进一步研究。

---

## 793. A $q$-Polymatroid Framework for Information Leakage in Secure Linear Network Coding

**arXiv ID:** 2601.07567 | [PDF](https://arxiv.org/pdf/2601.07567v1)

**作者:** Eimear Byrne `[一作]` (University College Dublin), Camilla Hollanti `[通讯]` (Aalto University)

**通讯引用:** 1924 | [OpenAlex ID](https://openalex.org/A5035260653)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于可表示 q-多项式的框架，用来描述安全线性网络编码中窃听者获取信息量的条件式，并将该信息泄漏与嵌套秩度代码对相关的 q-多项式的条件秩函数联系起来。

**💡 创新点**

创新点在于：1）首次将信息泄漏与 q-多项式的条件秩函数对应；2）引入 q-多项式端口和 q-访问结构，推广经典的 matroid 端口和访问结构；3）把 Massey 的最小码字与最小重构集的对应关系推广到秩度码；4）证明了一个 q-版本的 Brickell–Davenport 定理，说明在理想、完美且连通的情况下端口可以由 q-矩阵诱导。

**🔧 技术方法**

主要技术手段包括：q-多项式理论（子空间格上的秩函数、双子、多项式的删约和收缩等），秩度码的代数结构（最大秩距离 MRD 码、双子码等），以及信息论工具（互信息、熵的子空间解释）。

**📊 数据集**

本研究为理论分析，未使用公开数据集；但作者在论文中利用 Magma 计算示例代码的 q-多项式、端口结构和最小码字，以验证理论结果。

**📈 对比分析**

由于是理论推导，未进行实验性能比较；但通过例子展示了端口完美性与信息泄漏的对应关系，并给出了阈值结构的判定准则，说明在 MRD 码下可实现最优阈值访问结构。

**⚠️ 局限性**

局限性包括：①并非所有 q-多项式端口都是完美的，导致部分结果只能给出充分条件；②对连通性的定义尚未确定，影响 q-矩阵与端口的唯一对应关系；③在非可表示的 q-多项式中缺乏信息论的操作解释；④缺少针对非线性秩度码或子空间码的推广。

---

## 794. Near-Optimal Private Linear Regression via Iterative Hessian Mixing

**arXiv ID:** 2601.07545 | [PDF](https://arxiv.org/pdf/2601.07545v1)

**作者:** Omri Lev `[一作]` (Massachusetts Institute of Technology), Ashia C. Wilson `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1257 | [OpenAlex ID](https://openalex.org/A5013415067)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于高斯草图的迭代Hessian混合(IHM)算法，用于在已知数据上做差分隐私普通最小二乘回归，并给出了理论和经验上的误差分析。

**💡 创新点**

创新点在于将迭代Hessian草图技术与差分隐私结合，克服了现有Gaussian‑Sketch方法的内在局限，并通过新证明证明IHM在大多数隐私与统计效率折中下优于AdaSSP基线。

**🔧 技术方法**

采用了差分隐私框架、随机Gaussian草图、迭代Hessian草图、Ridge回归估计、误差与隐私分析以及大规模实验模拟。

**📊 数据集**

在多组常用线性回归基准数据集上进行实验（UCI、Kaggle等公开数据集的线性回归任务），并收集了这些数据集的规模与特征。

**📈 对比分析**

与AdaSSP和此前的Gaussian‑Sketch方法进行对比；实验结果显示IHM在绝大多数数据规模与隐私级别下误差更低、相对误差提升约10%‑30%，在极端隐私设置下仍保持竞争力。

**⚠️ 局限性**

局限性包括：仅适用于已知上界的bounded数据；对X的满秩性有要求；在极低维或极稀疏数据时迭代收敛可能较慢；在极高隐私ε下仍不如AdaSSP达到理论下界。

---

## 795. d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation

**arXiv ID:** 2601.07568 | [PDF](https://arxiv.org/pdf/2601.07568v1)

**作者:** Yu-Yang Qian `[一作]` (University of California), Hao Zhang `[通讯]` (University of California)

**通讯引用:** 63393 | [OpenAlex ID](https://openalex.org/A5100397026)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 d3LLM 框架，结合伪轨迹蒸馏与基于熵的多块并行解码，并引入 AUP 指标评估 dLLM 的精度-并行度权衡。

**💡 创新点**

创新点在于：① 通过教师 dLLM 的伪轨迹为训练提供中间监督，显著提升并行解码顺序；② 采用熵门控多块解码与 KV‑cache 刷新机制，在保持高并行度的同时抑制错误传播；③ 设计 AUP 指标统一衡量精度与并行度，克服单指标误导。

**🔧 技术方法**

使用的技术包括伪轨迹蒸馏、渐进噪声与窗口的课程学习、熵门控多块并行解码、KV‑cache 刷新与早停、以及新定义的 AUP 评价指标。

**📊 数据集**

训练数据来自 PRM12K、AceCode、GSM8K 训练集和 Numina‑Math；评测数据涵盖 GSM8K‑CoT、MATH、HumanEval、MBPP 以及长上下文 GSM8K（5‑shot）等。

**📈 对比分析**

与 LLaDA、Dream、Fast‑dLLM、Fast‑dLLM‑v2、dParallel、D2F 等方法对比，d3LLM 在 9/10 任务中获得最高 AUP 分数，推理速度比原版 dLLM 高 10×，比 AR 模型 Qwen‑2.5‑7B‑it 高 3.6‑5×，同时保持或提升准确率。

**⚠️ 局限性**

局限性包括：仍依赖教师模型的伪轨迹，数据量相对有限；在极长文本或高度专业任务中，KV‑cache 刷新频率与准确性仍需进一步平衡；AUP 指标虽统一，但对不同任务的阈值设置仍需经验调优。

---

## 796. Dynamic $(Δ+ 1)$ Vertex Coloring

**arXiv ID:** 2601.07566 | [PDF](https://arxiv.org/pdf/2601.07566v1)

**作者:** Noam Benson-Tilsen `[一作]` `[通讯]`, Noam Benson-Tilsen

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并总结了动态图着色领域最新的子线性算法，包括针对可观测对手和自适应对手的方案

**💡 创新点**

在已有 O(Δ) 朴素算法的基础上提出层级数据结构实现 O(logΔ) 与 O(1) 更新时间，并首次给出对抗自适应对手下的子线性时间算法

**🔧 技术方法**

采用层级划分、随机采样、可变桶、增广路、稀疏-稠密分解、动态匹配等技术

**📊 数据集**

本研究为综述性工作，未使用特定实验数据集，主要以理论分析和证明为主

**📈 对比分析**

与传统 O(Δ) 方案相比，更新时间显著下降，O(logΔ) 与 O(1) 算法在理论上已达到最优；对自适应对手的算法实现了 Õ(n^{8/9}) 的子线性更新时间，优于之前的线性/多项式上界

**⚠️ 局限性**

局限性在于需要预先知道最大度 Δ、对输入序列有不同的可观测性假设、实现复杂度高、对自适应对手的子线性算法仍然具有较大的 Θ(n^{8/9}) 失配

---

## 797. Stable In-hand Manipulation for a Lightweight Four-motor Prosthetic Hand

**arXiv ID:** 2601.07559 | [PDF](https://arxiv.org/pdf/2601.07559v1)

**作者:** Yuki Kuroda `[一作]` (OMRON SINIC X Corporation), Masashi Hamaya `[通讯]` (OMRON SINIC X Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了一款311 g、四电机的轻量化电动义肢手，利用电机电流反馈估算握持物体宽度并协调食指实现平稳的精细-侧向（PL）手内操纵。

**💡 创新点**

创新点在于：①仅用电机电流即可估计物体宽度，省去额外传感器；②通过宽度估计实现人类启发的食指定位，从而在重物或宽物时保持握持稳定；③机械设计仅需单轴拇指与简单关节即可完成多姿态转换。

**🔧 技术方法**

采用机械优化的单轴拇指结构、四条电机驱动的链条与差速链接、基于电流阈值的接触检测、宽度到目标角度的查找表以及人类启发的食指固定角度控制。

**📊 数据集**

使用的“数据集”是实验中自制的原始物体：PLA/Al圆柱与方柱（宽度5–30 mm，重量至289 g）以及常用日常物品（笔、瓶盖、卡片等），共计数十件。

**📈 对比分析**

与无食指协调（PL w/o index）相比，带协调（PL w/ index）在轻量物体上成功率100%，在重物（高达289 g）上保持≥80%成功率；无协调时，重物成功率降至30–60%。实验表明该方法在不增加重量或额外传感器的前提下，显著提升了PL操纵的可靠性。

**⚠️ 局限性**

局限性包括：①对圆柱形物体的稳定性较差，受限于点接触模型；②无法提供足够扭矩完成高负载旋转任务（如紧固瓶盖）；③缺乏腕部与上肢协同控制，需要进一步集成。

---

## 798. Contextual Discrepancy-Aware Contrastive Learning for Robust Medical Time Series Diagnosis in Small-Sample Scenarios

**arXiv ID:** 2601.07548 | [PDF](https://arxiv.org/pdf/2601.07548v1)

**作者:** Kaito Tanaka `[一作]` (SANNO University), Keisuke Matsuda `[通讯]` (SANNO University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种基于上下文差异感知的对比学习框架CoDAC，用以在少标注医疗时序数据上实现更准确的疾病诊断。

**💡 创新点**

创新点在于结合Transformer自编码器的上下文差异估计器（CDE）与动态多视图对比框架（DMCF），让模型能够自适应关注异常区段并加强表征学习。

**🔧 技术方法**

采用Transformer自编码器、膨胀卷积+多头注意力编码器、动态视图加权对比学习以及多阶段预训练+微调等技术。

**📊 数据集**

使用AD、PD的EEG数据以及MI的ECG数据，并借助外部健康数据集（Valladolid、TDBrain、PTB-XL）进行训练与评估。

**📈 对比分析**

与多种基线（TS2Vec、SimCLR、COMET、DAAC等）对比，CoDAC在10%标签场景下AUROC/AUPRC均高于同类方法，整体性能优于现有最优算法。

**⚠️ 局限性**

仍需在真实临床数据上验证、对多模态信号扩展有限，且对异常解释的临床可解释性需进一步评估。

---

## 799. FairRF: Multi-Objective Search for Single and Intersectional Software Fairness

**arXiv ID:** 2601.07537 | [PDF](https://arxiv.org/pdf/2601.07537v1)

**作者:** Giordano d'Alosio `[一作]` (University of L'Aquila), Federica Sarro `[通讯]` (University College London)

**通讯引用:** 4413 | [OpenAlex ID](https://openalex.org/A5012165852)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于多目标进化搜索的算法，自动寻找随机森林模型的超参数和输入数据突变，以同时优化公平性和预测准确性。

**💡 创新点**

创新点在于将公平性与效能建模为多目标优化问题，允许利益相关者根据需求选择 Pareto 前沿上的解，并在公平性与效能之间提供可控的权衡。

**🔧 技术方法**

技术包括NSGA-II进化算法、随机森林分类器、数据突变策略、统计公平性指标（SPD、EOD、AOD）与准确性等效能指标。

**📊 数据集**

实验使用5个公开数据集（Adult、Compas、German、Bank、MEPS）并分别划分为单敏感变量和交叉敏感变量场景。

**📈 对比分析**

与26个基线（包括基类分类器、随机搜索、现有公平性缓解方法RW、ADV、MAAT、DEMV、EOP、FairHOME）比较，实验显示其在公平性上大幅提升且效能保持相近，且在交叉公平性上优于FairHOME。

**⚠️ 局限性**

局限性包括仅使用准确率和SPD作为目标函数，可能忽略其他公平性或效能度量；适用于二分类任务；搜索过程计算成本较高；且在实际部署时需进一步验证泛化能力。

---

## 800. A Protocol-Aware P4 Pipeline for MQTT Security and Anomaly Mitigation in Edge IoT Systems

**arXiv ID:** 2601.07536 | [PDF](https://arxiv.org/pdf/2601.07536v1)

**作者:** Bui Ngoc Thanh Binh `[一作]` (Nara Institute of Science and Technology), Yasuhiko Nakashima `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1483 | [OpenAlex ID](https://openalex.org/A5074853381)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

基于P4实现了MQTT协议感知的安全与异常检测管道，集成会话验证、主题前缀ACL、客户端速率软限和轻量异常检测；

**💡 创新点**

创新点在于将完整MQTT协议解析与状态检测（会话顺序、字节级主题授权、KeepAlive/Remaining‑Length异常）直接实现于可编程数据平面，且保持线速处理与子毫秒延迟；

**🔧 技术方法**

采用P4 v1model、BMv2软件交换机、寄存器/计数器/计量器、clone‑to‑CPU、动态表更新等技术；

**📊 数据集**

使用Mininet/BMv2环境生成的MQTT流量（100–16,000 pps、QoS 0–2、多个主题层级）进行实验；

**📈 对比分析**

通过与传统云IDS、CPU防火墙的对比（无直接数值），实验结果显示：策略执行准确率99.8%，异常检测召回率98%，交付率>99.9%，端到端延迟<1 ms；

**⚠️ 局限性**

局限性：仅在软件模拟器验证，未在硬件ASIC上部署；内存受限，只能支持512客户端；未实现ML自适应阈值；TCAM容量限制可能阻碍大规模部署。

---

## 801. Integrating Machine-Generated Short Descriptions into the Wikipedia Android App: A Pilot Deployment of Descartes

**arXiv ID:** 2601.07631 | [PDF](https://arxiv.org/pdf/2601.07631v1)

**作者:** Marija Šakota `[一作]` (École Polytechnique Fédérale de Lausanne), Robert West `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 101153 | [OpenAlex ID](https://openalex.org/A5059645286)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Wikipedia Android应用中试点部署多语言短描述生成模型Descartes，向编辑器提供机器建议并收集反馈与评估

**💡 创新点**

将大型多语言生成模型直接嵌入移动编辑工具，并在真实编辑场景中验证其可用性与质量，首次系统评估模型在多语言上的实用性

**🔧 技术方法**

使用Transformer语言模型Descartes（无语义类型信息），Beam Search解码（两束），API接口与移动端UI集成；结合GPU/CPU服务器与LiftWing部署

**📊 数据集**

基于12种语言（阿拉伯语、捷克语、德语、英语、西班牙语、法语、古吉拉特语、印地语、意大利语、日语、俄语、土耳其语）收集的3,968篇条目内容，生成短描述；使用社区评审员给出1-5分质量评分

**📈 对比分析**

对比机器生成与人工编写的短描述：接受率90%，平均质量评分4.1（机器）vs 4.2（人工），修改后4.1；更有经验编辑的接受质量更高（4.4对3.6）；Beam1优于Beam2；撤回率极低（0.06%），修改率可接受；用户保留率略高于对照组

**⚠️ 局限性**

局限包括：模型在某些语言（如古吉拉特语、阿拉伯语）数据不足导致低质量；生成速度受限（无GPU时10秒/句）；易出现日期漂移、歧义页错误、大小写不规范；对敏感主题（如在世人士）存在潜在风险，需要额外过滤与监控

---

## 802. The Issue with Special Issues: when Guest Editors Publish in Support of Self

**arXiv ID:** 2601.07563 | [PDF](https://arxiv.org/pdf/2601.07563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 803. Fifteen Years of Learning Analytics Research: Topics, Trends, and Challenges

**arXiv ID:** 2601.07629 | [PDF](https://arxiv.org/pdf/2601.07629v1)

**作者:** Valdemar Švábenský `[一作]` (Masaryk University), Dragan Gašević `[通讯]` (Monash University)

**通讯引用:** 28014 | [OpenAlex ID](https://openalex.org/A5036855560)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统性分析了15年（2011-2025）LAK会议的936篇完整及短篇论文，探讨作者生态、资金与研究主题关系，以及全球与本土主题发展趋势。

**💡 创新点**

创新点在于首次将无监督机器学习、自然语言处理与网络分析相结合，对整套会议文献进行宏观量化；发现六个持续存在的主题中心并揭示资助者与主题之间的系统性关联；提出跨中心资助与国际合作的可视化与测度。

**🔧 技术方法**

主要技术包括：
- 关键词标准化与Levenshtein编辑距离匹配
- Louvain社区检测（主题与资助网络）
- 句子级别抽象文本嵌入（384维）与k‑means聚类
- 双向网络（主题–资助）与度量（密度、模块度）
- 余弦相似度、Spearman相关性评估跨国协作与主题相似度。

**📊 数据集**

使用自建的LAK 2011‑2025数据集，共936篇论文；数据包含DOI、年份、标题、作者、机构、国家、摘要、关键词、文章类型、致谢与资助信息。

**📈 对比分析**

比较方法：通过关键词聚类与文本嵌入两种独立途径验证主题中心一致性；对双向网络进行连通性、密度与模块度评估，显示资助网络在2024年最为紧密。表现：六大主题中心的余弦相似度均高于0.90，跨国协作与主题相似度呈显著正相关（ρ=0.44，p<0.001）。

**⚠️ 局限性**

局限性：仅分析主轨道（完整/短篇）论文，未包含海报、演示等初级工作；致谢信息不完整，可能低估机构种子资助；未覆盖JLA及其他会议，因而对主题热度的预测有限。

---

## 804. Peformance Isolation for Inference Processes in Edge GPU Systems

**arXiv ID:** 2601.07600 | [PDF](https://arxiv.org/pdf/2601.07600v1)

**作者:** Juan José Martín `[一作]` (Universitat Politècnica de València), Carles Hernández `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 1184 | [OpenAlex ID](https://openalex.org/A5047492364)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了 NVIDIA GPU 中的 MPS、MIG 与 Green Contexts 三种隔离机制，以实现安全关键 AI 推理的时间可预测性。

**💡 创新点**

创新点在于对 MIG、MPS 与 Green Contexts 的性能、内存隔离与功耗影响进行系统对比，并提出 Green Contexts 作为低功耗边缘设备的可行替代方案。

**🔧 技术方法**

使用了 NVIDIA A100、Jetson Orin Nano/AGX GPU，结合 PyTorch、TensorRT 以及 CUDA 12.4 的 Green Contexts 实验。

**📊 数据集**

数据集使用了 ImageNet‑1K 预训练模型（ConvNeXt、MobileNetV2、ResNet18、ViT 等）。

**📈 对比分析**

通过最大推理频率搜索、吞吐量、内存占用和功耗等指标对比，结果显示 MIG 在时间隔离上最稳健，MPS 提升了吞吐但不提供完整隔离，GC 在功耗受限下隔离不足，功耗较高的 AGX 上可接近 MIG。

**⚠️ 局限性**

局限性包括 MIG 固定分区、GC 缺乏内存隔离、GC 在功耗受限平台上的隔离受限，且现有技术难以实现动态资源调度与严格的实时保证。

---

## 805. GRPO with State Mutations: Improving LLM-Based Hardware Test Plan Generation

**arXiv ID:** 2601.07593 | [PDF](https://arxiv.org/pdf/2601.07593v1)

**作者:** Dimple Vijay Kochar `[一作]` (Massachusetts Institute of Technology), Brucek Khailany `[通讯]` (NVIDIA Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于大语言模型的两阶段子单元测试计划生成框架，并通过强化学习改进模型的推理能力。

**💡 创新点**

创新点在于：①将测试计划作为中间结构拆解任务；②提出GRPO‑SMu在强化学习中通过输入状态突变提升探索；③设计树状突变策略生成多样化训练数据。

**🔧 技术方法**

使用了监督微调（SFT）、Group Relative Policy Optimization (GRPO) 与其改进版GRPO‑SMu、树状代码突变技术以及随机测试生成器做验证。

**📊 数据集**

主要数据集为ScaleRTL（500条RTL代码）以及其通过树状突变生成的1,500条变异RTL，用于训练与评估。

**📈 对比分析**

与DeepSeek‑R1、Claude‑4.0‑Sonnet、LLaMA‑3.1‑405B、ScaleRTL‑32B等基线模型比较，7B GRPO‑SMu模型实现了33.3% 的金标通过率和13.9% 的突变检测率，较未训练基线提升了17.6个百分点，甚至超越了更大规模的通用LLM。

**⚠️ 局限性**

仍距工业级验证远，整体通过率偏低，依赖于强化学习奖励稀疏、数据集覆盖仍有限，且需要进一步提升对不同逻辑类型的鲁棒性。

---

## 806. Neural Architecture for Fast and Reliable Coagulation Assessment in Clinical Settings: Leveraging Thromboelastography

**arXiv ID:** 2601.07618 | [PDF](https://arxiv.org/pdf/2601.07618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 807. Hagenberg Risk Management Process (Part 1): Multidimensional Polar Heatmaps for Context-Sensitive Risk Analysis

**arXiv ID:** 2601.07644 | [PDF](https://arxiv.org/pdf/2601.07644v1)

**作者:** Eckehard Hermann `[一作]` (University of Applied Sciences Upper Austria), Harald Lampesberger `[通讯]` (University of Applied Sciences Upper Austria)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5010248122)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并实现了多维极坐标热图，用于在复杂基础设施的风险分析中，将传统的二维风险矩阵扩展为可显式表示多维上下文信息的可视化模型。

**💡 创新点**

创新点在于：①将上下文维度作为极坐标轴显式化，保持二维矩阵为其特例；②通过聚合模式（mode）对主轴进行颜色映射；③引入阈值圆弧直观显示阈值超标；④兼容并能聚合现有方法（FMEA、HAZOP等）的结果。

**🔧 技术方法**

采用的技术包括：极坐标嵌入公式、离散层级集合、笛卡尔积状态空间、模式聚合算法、阈值检测与可视化，且实现了与Riskomat实时风险管理系统的集成。

**📊 数据集**

示例数据来自于“数据中心冷却失效”情景，包含冷却冗余级别、维护状态、概率与影响等维度；并未使用公开大规模数据集，而是通过模拟情景进行演示。

**📈 对比分析**

比较方法：将传统二维风险矩阵视为切片 Mσ 与多维极坐标热图对比，展示在单一上下文维度变化时，极坐标热图能连续可视化而二维矩阵需切换切片。性能方面主要体现在可视化清晰度和上下文可见性上，并未给出定量指标。

**⚠️ 局限性**

局限性：①需预先定义轴与规则，规则制定工作量较大；②维度增多时可视化复杂度上升，可能导致信息过载；③阈值设置主观且缺乏统一标准；④论文仅演示示例，未在大规模真实系统上进行验证。

---

## 808. Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning

**arXiv ID:** 2601.07636 | [PDF](https://arxiv.org/pdf/2601.07636v1)

**作者:** Yanan Chen `[一作]` (Xi’an Jiaotong University), Wen Wen `[通讯]` (Xi’an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为 FLAD 的连续学习优化框架，能够将 sharpness 相关的扰动向量分解为梯度对齐分量与随机噪声分量，并仅保留噪声分量以实现更平坦的最小化，从而提升模型在连续任务中的泛化与抗遗忘能力。

**💡 创新点**

创新点包括：① 对 sharpness 进行梯度对齐与噪声分离，剔除对学习不利的方向；② 结合零阶与阶梯二阶 sharpness 的分解，形成统一的优化策略；③ 设计轻量级的调度方案，使得仅在部分迭代中使用 FLAD 即可获得显著收益；④ 证明该框架可无缝集成至现有的三大类连续学习方法（重放、正则化、扩展）。

**🔧 技术方法**

技术手段主要有：sharpness‑aware 预训练（SAM、GAM），梯度对齐与正交投影、指数滑动平均（EMA）估计全梯度、Hessian‑vector 乘法实现一阶 sharpness、噪声分量的选取、以及基于比例调度的部分应用策略。

**📊 数据集**

实验数据集包括 CIFAR‑10、CIFAR‑100（5/10 任务划分）和 Tiny‑ImageNet（8 任务划分），所有数据均采用标准数据增强与 128 大小 batch。

**📈 对比分析**

作者将 FLAD 与 6 种主流连续学习基线（Replay、iCaRL、PODNet、WA、FOSTER、MEMO）以及 SAM、GAM、C‑Flat 等 sharpness‑aware 方案进行对比，结果显示 FLAD 在所有任务与指标（平均准确率、随时准确率）上提升 1–3% 左右，同时在训练过程中收敛更快，且即使仅使用 10–20% 的 epoch 也能获得与全程使用相当或更好的性能。

**⚠️ 局限性**

局限性包括：仍需在每次迭代执行 2 次前向与 4 次反向传播，导致一定的计算开销；需要手动调节超参数 ρ 与 γ，选择不当可能影响效果；该方法主要针对梯度噪声方向，可能忽略对齐方向在某些任务中的正面作用；在极大模型或任务数量极多的场景下的可扩展性与稳定性尚未完全验证。

---

## 809. Searching point patterns in point clouds describing local topography

**arXiv ID:** 2601.07621 | [PDF](https://arxiv.org/pdf/2601.07621v1)

**作者:** Ewa Bednarczuk `[一作]` (Systems Research Institute Polish Academy of Sciences), Małgorzata Szelachowska `[通讯]` (Institute of Geodesy and Cartography)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5079985961)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于 Wasserstein 距离、最小二乘和 Procrustes 距离的三维点云模式匹配与对齐算法，用于校正地球重力测站的水平坐标，通过与高分辨率 DEM 进行匹配来提高点位精度。

**💡 创新点**

创新点在于将局部地形“十字”测量数据与全局 DEM 结合，利用归一化后的有限差分描述符和 Wasserstein 与 Procrustes 两种不同的相似度度量，形成一种多尺度、多指标的匹配框架；同时引入 λ 缩放参数调节 z 轴尺度，显著提升了匹配鲁棒性。

**🔧 技术方法**

核心技术包括：
- 3D 点云的归一化与尺度调整（均值中心化、标准差归一化、z 轴 λ 缩放）；
- Wasserstein（W₂）距离、最小二乘距离、Procrustes 距离的计算与最优化；
- 离散曲线导数的有限差分估计；
- 角度离散化与离散最优匹配算法；
- 线性求解与 SVD 计算。

**📊 数据集**

使用的数据集为：
- 波兰高分辨率 DEM（1 m 级），包含 52 个 2.3 km×2.3 km 区域；
- 通过“十字”测量获得的局部地形点云（每点 100 m 范围内 4 条轴向测量），共 401 个点；
- 生成的随机扰动样本（共 8 次实验），通过随机平移和角度扰动得到。

**📈 对比分析**

实验结果显示：
- 在 λ=1 时，Wasserstein 匹配误差与原始中心距离相当；
- 随着 λ 增大到 40 以上，误差显著下降（大多数实验误差 < 5 m）；
- Procrustes 距离在 λ=1 时误差约 1–2 m，λ>20 后趋于稳定；
- 最小二乘方法在 λ≥20 时表现最佳，误差约 1–3 m；
- 整体来看，Wasserstein 与 Procrustes 在大 λ 下性能最优，能够有效消除由于坐标误差导致的偏移。

**⚠️ 局限性**

局限性包括：
- 仅适用于拥有完整 3D DEM 的区域，对缺失或低分辨率区域效果有限；
- 对局部地形极度复杂或高度不规则的点云匹配仍可能出现局部最小；
- 参数 λ 的选择需经验调优，未给出自适应选择策略；
- 计算量随点云大小和角度离散程度增加而显著升高，实时性受限。

---

## 810. DIAGPaper: Diagnosing Valid and Specific Weaknesses in Scientific Papers via Multi-Agent Reasoning

**arXiv ID:** 2601.07611 | [PDF](https://arxiv.org/pdf/2601.07611v1)

**作者:** Zhuoyang Zou `[一作]` (Penn State University), Wenpeng Yin `[通讯]` (Penn State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个人类驱动的多代理框架，用于论文弱点识别，模拟审稿人、作者辩论，并按严重性输出Top‑K弱点。

**💡 创新点**

创新点包括：① 用动态生成的评判维度驱动代理分解，替代传统表面角色配置；② 引入作者代理进行对抗性辩论，验证并过滤无效弱点；③ 基于大规模评审数据学习严重性评分，对弱点进行排序。

**🔧 技术方法**

技术：多代理LLM架构，包含Criterion‑oriented Reviewer Decomposition、Rebuttal Mechanism、Severity Ranking模块；使用开源/闭源LLM（GPT‑4o、Llama 3.1‑70B、Mistral‑7B、Qwen‑2.5‑72B）进行推理。

**📊 数据集**

数据集：AAAR（包含人工评审文本）与ReviewCritique（标注有效/无效弱点段落）两大基准。

**📈 对比分析**

与单代理LLM、AgentReview、MARG等基线对比，实验表明在AAAR和ReviewCritique上取得语义F1、Specificity均为最优；F1_inv指标显示显著抑制无效弱点；总体性能提升显著，甚至使开源模型逼近GPT‑4o水平。

**⚠️ 局限性**

局限性：多代理结构导致运行时开销增大；未检索外部文献，仅评估AI领域，未知跨学科适用性。

---

## 811. UIKA: Fast Universal Head Avatar from Pose-Free Images

**arXiv ID:** 2601.07603 | [PDF](https://arxiv.org/pdf/2601.07603v1)

**作者:** Zijian Wu `[一作]` (Nanjing University), Hao Zhu `[通讯]` (Nanjing University)

**通讯引用:** 9675 | [OpenAlex ID](https://openalex.org/A5068560690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种面向任意数量未标定输入图像的前向式3D高保真高斯头部动画模型UIKA，能够快速生成可动画的头部虚拟形象。

**💡 创新点**

创新点包括：①基于像素级面部对应的UV引导重投影实现跨视角信息对齐；②在Transformer中加入UV注意力分支，让可学习的UV令牌同时关注屏幕空间与UV空间特征；③自适应融合策略将全局高斯预测与局部UV聚合色彩动态权衡；④构建规模庞大的合成多视角数据集以提升身份与表情多样性。

**🔧 技术方法**

核心技术涵盖3D高斯样条渲染、DINOv3冻结特征提取器、DPT头部UV坐标回归、UV/屏幕双域注意力Transformer、线性混合骨骼动画（LBS）以及基于感知损失的端到端训练。

**📊 数据集**

使用的公开数据集包括VFHQ、HDTF、NeRSemble‑v2；同时构造了7,500+身份、9视角、13,000+帧的合成数据集。

**📈 对比分析**

在单视角和多视角重建与复现任务中，相比LAM、GAGAvatar、Portrait4D‑v2、InvertAvatar、GPAvatar和DiffusionRig等SOTA方法，UIKA在PSNR、SSIM、LPIPS、AED、APD、CSIM等指标均实现显著提升，并且推理速度可达220 FPS，实现了实时动画。

**⚠️ 局限性**

局限性主要体现在：对极端表情或视角的泛化仍受限于合成数据；对非人脸或全身动画尚未扩展；训练需要大量GPU资源，且模型对背景复杂度仍有一定依赖。

---

## 812. ES-Mem: Event Segmentation-Based Memory for Long-Term Dialogue Agents

**arXiv ID:** 2601.07582 | [PDF](https://arxiv.org/pdf/2601.07582v1)

**作者:** Huhai Zou `[一作]` (Chongqing University), Kaiwen Wei `[通讯]` (Chongqing University)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5068039769)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ES-Mem框架，结合事件分割理论构建动态事件分割和分层记忆，实现对长对话的语义一致事件划分与结构化检索。

**💡 创新点**

创新点在于把事件边界视为认知锚点，动态划分语义事件并在三层记忆中保留边界、摘要与原始上下文，采用粗到细检索提升检索精度。

**🔧 技术方法**

使用LLM（如Qwen2.5-3B、Llama3.2-3B、GPT-4o-mini）进行主题提取与边界判定，利用MiniLM嵌入、Faiss向量索引和两阶段事件分割算法。

**📊 数据集**

评估数据集包括LoCoMo、LongMemEval-S两大长记忆基准，以及DialSeg711、SuperDialSeg、TIAGE三大对话分割数据集。

**📈 对比分析**

在LoCoMo和LongMemEval-S上与十多种基线相比，ES-Mem在所有LLM后端均取得最高F1/ACC，单跳QA最高F1 50.07，整体性能提升显著；在分割任务中相对无监督方法也获得最佳Score。

**⚠️ 局限性**

局限性包括未实现记忆的动态进化（如遗忘、抽象）以及仅支持文本模态，未来需加入多模态和记忆动态机制。

---

## 813. An adjoint method for training data-driven reduced-order models

**arXiv ID:** 2601.07579 | [PDF](https://arxiv.org/pdf/2601.07579v1)

**作者:** Donglin Liu `[一作]` (Lund University), Mengwu Guo `[通讯]` (Lund University)

**通讯引用:** 1017 | [OpenAlex ID](https://openalex.org/A5005525255)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于连续时间轨迹损失的Adjoint训练框架，用以学习二次多项式ROM，避免了传统OpInf中对时间导数的数值差分估计；

**💡 创新点**

创新点在于将轨迹拟合转化为连续时间优化，并通过后向Adjoint求解得到梯度，实现对稀疏、噪声数据的鲁棒学习；

**🔧 技术方法**

主要技术包括POD降维、OpInf回归预热、Tikhonov/TSVD正则化、连续时间损失、后向Adjoint求解、梯度下降与Armijo线搜索；

**📊 数据集**

使用三类经典PDE（粘性Burgers、Fisher‑KPP、线性输运扩散）在高分辨率网格上生成的全阶模拟快照，随后提取训练/验证/测试子集；

**📈 对比分析**

与传统OpInf（2阶/6阶差分）在不同采样密度和噪声水平下进行比较，Adjoint方法在清洁数据下相当或更优，在稀疏或高噪声场景下明显优于两种基线，尤其在20%采样和200%噪声时仍保持低误差；

**⚠️ 局限性**

局限性包括对二次多项式模型的依赖、对时间离散化精度敏感、相较于最小二乘回归计算开销更大。

---

## 814. Beyond Entangled Planning: Task-Decoupled Planning for Long-Horizon Agents

**arXiv ID:** 2601.07577 | [PDF](https://arxiv.org/pdf/2601.07577v1)

**作者:** Yunfan Li `[一作]` (State Key Laboratory of AI Safety Institute of Computing Technology), Huawei Shen `[通讯]` (State Key Laboratory of AI Safety Institute of Computing Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Task-Decoupled Planning (TDP) 框架，利用 Supervisor 将任务拆分为 DAG 子任务，并在子任务级别进行局部规划与执行，显著降低上下文耦合，提高长周期任务的鲁棒性与效率。

**💡 创新点**

创新点在于显式子任务解耦与局部上下文限制，将规划与执行拆分为 Supervisor、Planner、Executor 三个模块，并通过 Self-Revision 动态维护 DAG，实现局部重规划、避免错误传播。

**🔧 技术方法**

技术上结合大型语言模型的 Planner 与 Executor、Supervisor 自动生成 DAG、Self-Revision 更新图谱；使用无训练提示式推理，并与环境 API 交互执行工具调用。

**📊 数据集**

在 TravelPlanner、ScienceWorld 与 HotpotQA 三大长周期基准上进行评估，涵盖多阶段工具调用、交互式控制与多跳推理任务。

**📈 对比分析**

与 ReAct、CoT、Plan-and-Act 等基线在 DeepSeek‑V3.2 与 GPT‑4o 上对比，TDP 在所有任务中表现最稳定，平均得分最高，Token 消耗降低 70–82%，同时保持或提升准确率/奖励。

**⚠️ 局限性**

局限性包括子任务粒度可能未最优，导致执行效率仍有提升空间；评估聚焦可验证任务，尚未验证在开放式真实任务中的表现。

---

## 815. Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based Brain-Computer Interfaces

**arXiv ID:** 2601.07556 | [PDF](https://arxiv.org/pdf/2601.07556v1)

**作者:** Siyang Li `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 14574 | [OpenAlex ID](https://openalex.org/A5008740867)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种无反向传播的测试时适配方法BFT，通过对单个EEG样本进行多种知识引导的增广或蒙特卡洛Dropout变换，生成多条预测并用学习到的排序模块对其可靠性进行加权，进而在部署阶段实现自适应解码。

**💡 创新点**

创新点包括：①完全基于前向传播，消除反向传播带来的计算、隐私和量化限制；②将知识引导的增广与近似贝叶斯推理两类变换统一到BFT框架；③设计学习到排序与映射模块，将预测可靠性映射为可分辨的权重；④兼顾分类与回归任务，具有噪声鲁棒性与隐私保护；⑤理论上证明多变换加权可降低不确定性。

**🔧 技术方法**

使用技术包括：EEGNet特征提取与全连接回归/分类头；多种EEG增广（噪声、幅度缩放、频移、滑窗等）与MC Dropout变换；学习到排序网络r(·)与映射网络m(·)；温度缩放与softmax聚合；欧氏对齐(EA)与批归一化自适应；量化(32→8位)后推理。

**📊 数据集**

使用了五个公开EEG数据集：Zhou2016、BNCI2014001、HighGamma（三类左右手运动想象分类任务）；以及Driving与SEED‑VIG（驾驶员困倦指数回归任务）。

**📈 对比分析**

通过留一子标交叉验证与在线序列推理，将BFT与传统TL（DAN、DANN、CDAN‑E等）、BP‑based TTA（Tent、T‑TIME等）、无BP TTA（BN‑adapt、T3A、LAME）、单一变换与MC Dropout平均等方法比较。实验显示：在分类任务上BFT‑A/BFT‑D平均提升约1–2%准确率，性能与最优BP‑TTA相当或更优；在回归任务上CC提升0.01–0.02，RMSE下降；在加入时序与空间噪声实验中保持最高稳定性；量化后仍保持显著优势。

**⚠️ 局限性**

局限性包括：①仍需预训练的特征提取器，对极端域漂移可能不够鲁棒；②对空间噪声的抑制效果不如对时序噪声；③缺乏针对标签分布漂移的专门机制；④对完全无监督或弱监督目标域适配的进一步评估尚未完成；⑤对在线实时连续波形（无明显试验分界）的异步适配仍需研究。

---

## 816. VirtualEnv: A Platform for Embodied AI Research

**arXiv ID:** 2601.07553 | [PDF](https://arxiv.org/pdf/2601.07553v1)

**作者:** Kabir Swain `[一作]` (Massachusetts Institute of Technology), Antonio Torralba `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 96261 | [OpenAlex ID](https://openalex.org/A5085020955)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了基于Unreal Engine 5的高保真、多代理、语言驱动仿真平台VirtualEnv，用于评估LLM在交互式任务中的表现并提供逃脱室等复杂任务；

**💡 创新点**

创新点在于将LLM与VLM结合实现语言驱动的场景生成与编辑，支持3D‑MIO环境、20k+交互资产、实时多代理协作以及逃脱室挑战框架；

**🔧 技术方法**

采用Unreal Engine 5、LLM（如GPT‑4o、Claude 3等）、VLM、场景图、JSON‑驱动环境编辑、光照渲染、深度/语义感知等技术；

**📊 数据集**

使用LLM生成的任务与场景，构建约140 000个预定义或自动生成任务；通过31人参与的视觉逼真度调查以及多任务指标（单/多代理、探索、搜索等）数据进行评估；

**📈 对比分析**

通过用户调查比较视觉逼真度（VirtualEnv 4.46/5，优于其他平台），在五个基准任务上比较reasoning vs non‑reasoning LLM，链式思维提升约11%，多代理协作提升显著，整体任务成功率高于现有平台；

**⚠️ 局限性**

局限在于部分可观测性、探索策略不足、虚假目标、状态跟踪错误导致约30%失败；对物理不可能动作和对象相似性也存在问题，仍有提升空间。

---

## 817. On the Sequence Reconstruction Problem for the Single-Deletion Two-Substitution Channel

**arXiv ID:** 2601.07547 | [PDF](https://arxiv.org/pdf/2601.07547v1)

**作者:** Wentu Song `[一作]` (Singapore University of Technology and Design), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 28027 | [OpenAlex ID](https://openalex.org/A5030858163)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过组合数学方法分析了在q-进制字母表上，长度为n的字符串在两次单一替换（双错误）场景下的误差球交集大小；

**💡 创新点**

创新点在于给出了双错误球交集的精确上界，并证明了该上界在大多数情形下是最优的；

**🔧 技术方法**

主要使用了球体交集计数、距离计数与组合枚举等技术；

**📊 数据集**

该研究属于理论性工作，不依赖具体数据集；

**📈 对比分析**

通过与现有错误球交集上界的比较，证明了提出上界在误差率与码长关系上具有更好的性能；

**⚠️ 局限性**

局限性在于仅针对双错误球，且在某些特殊距离配置下上界可能略微失效。

---

## 818. Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection

**arXiv ID:** 2601.07780 | [PDF](https://arxiv.org/pdf/2601.07780v1)

**作者:** Mariana Costa `[一作]` (University of Brasilia), Camila Ferreira `[通讯]` (University of Brasilia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MyGO Poly-Reflective Chain-of-Thought（PR‑CoT）方法，利用 prompt engineering 在 LLM 推理过程中引入多视角自我反思（逻辑一致性、信息完整性、偏见/伦理、备选方案），并在初始 CoT 之后进行综合修正，从而提升推理的准确性和自我纠错能力。

**💡 创新点**

创新点在于：① 将单维反思扩展为多维度结构化反思，覆盖四个互补视角；② 全过程仅通过精心设计的提示实现，无需模型改造或再训练；③ 引入反思结果的综合融合步骤，实现迭代式自我纠正。

**🔧 技术方法**

技术手段：Chain‑of‑Thought prompting、prompt engineering、基于多视角的自我反思流程、综合推理融合；实现基于大型预训练模型（GPT‑3.5、GPT‑4）。

**📊 数据集**

使用与 MyGO Multiplex CoT 兼容的公开推理 benchmark，涵盖算术问题、常识推理、伦理决策和逻辑谜题等四类任务。

**📈 对比分析**

对比方法：传统 CoT 与单维反思 MCoT；评估指标为逻辑一致性、错误修正率和最终答案准确率。PR‑CoT 在所有任务中均显著优于两者，逻辑一致性最高提升 13%、错误修正率提升 3‑4%，尤其在伦理决策任务中表现突出。

**⚠️ 局限性**

限制：① 多轮反思导致 token 消耗与推理时间大幅增加（约 4‑5 倍）；② 视角设计需手工制定，迁移性与自动化程度有限；③ 目前仅在公开 benchmark 上验证，缺乏真实世界复杂场景的进一步检验。

---

## 819. THETA: Triangulated Hand-State Estimation for Teleoperation and Automation in Robotic Hand Control

**arXiv ID:** 2601.07768 | [PDF](https://arxiv.org/pdf/2601.07768v1)

**作者:** Alex Huang `[一作]` (Arizona State University), Akshay Karthik `[通讯]` (Arizona State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于三台摄像头多视角三角测量的低成本手部关节角度估计方法THETA，实时控制DexHand机器人手进行远程操作。

**💡 创新点**

创新点在于：①仅用三台普通RGB摄像头实现多视角三角测量提取关节角度；②结合DeepLabV3语义分割+HSV滤波构建九通道输入；③使用MobileNetV2进行离散角度分类，兼顾精度与实时性。

**🔧 技术方法**

核心技术包括三角测量、DeepLabV3语义分割、HSV颜色空间过滤、MobileNetV2 CNN分类器、温度缩放softmax与焦点损失、ROS2-Arduino串口通信。

**📊 数据集**

使用自建数据集，40种手势、48,000+彩色图像，手动标注MCP、PIP、DIP角度并加入±5°随机噪声。

**📈 对比分析**

与单视角或传统深度/传感器方法对比，THETA在测试集上实现97.18%精度、98.72%召回率、0.9274 F1，实时推理误差低于3°，显著优于现有RGB单视角方案。

**⚠️ 局限性**

局限性：角度离散化导致运动平滑不足；数据集规模有限，需进一步扩展多用户与环境多样性；缺乏腕部跟踪，整体手部运动仍不完整。

---

## 820. Video Evidence to Reasoning Efficient Video Understanding via Explicit Evidence Grounding

**arXiv ID:** 2601.07761 | [PDF](https://arxiv.org/pdf/2601.07761v1)

**作者:** Yanxiang Huang `[一作]` (Hong Kong Polytechnic University), Jianyuan Ni `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种链式证据（CoE）框架，通过先提取视频中与问题相关的关键视觉证据，再基于这些证据进行高效、可解释的推理，解决了大型视听语言模型在视频推理中精度与效率的矛盾。

**💡 创新点**

创新点包括：① 轻量化的证据归纳模块（EGM）以查询引导的注意力方式筛选关键帧；② 基于强化学习的证据锚定协议（Evidence‑Anchoring Protocol）强制模型在推理过程中显式引用时间锚点；③ 双注释的CoE‑Instruct训练集，实现感知与推理的分离式监督；④ 组合奖励机制与GRPO算法，提升推理的可视化根源与准确性。

**🔧 技术方法**

技术手段主要有：ViT编码器+LLM解码器架构；跨注意力网络实现EGM；结构化推理提示模板；二阶段训练（SFT + RL）配合多任务损失；奖励函数包含F1、IoU、答案正确性；GRPO强化学习优化。

**📊 数据集**

使用了自构建的CoE‑Instruct大规模（164k）数据集，包含问答、时间锚点、推理引导等；并在五个公开基准上评测：Video‑MME、MVBench、VSI‑Bench、VidHal、EventHall。

**📈 对比分析**

与现有方法（CoT、QA仅监督、传统LLM+视频特征等）相比，CoE在所有五个基准上均取得显著提升；例如CoE‑8B(RL)在MVBench上达到91.2分，超越GPT‑4V、Gemini‑1.5‑Pro等闭源系统；同时显著降低token使用与推理时延。

**⚠️ 局限性**

局限性包括：① 仍依赖InternVL骨干，对其他模型的可迁移性尚未验证；② 训练需要双注释的大规模数据与RL阶段的计算成本；③ 对极长或多模态复杂视频的鲁棒性虽然提升，但仍受视频分辨率、帧速率等因素影响。

---

## 821. On the application of the Wasserstein metric to 2D curves classification

**arXiv ID:** 2601.07749 | [PDF](https://arxiv.org/pdf/2601.07749v1)

**作者:** Agnieszka Kaliszewska `[一作]` (Systems Research Institute), Monika Syga `[通讯]` (Warsaw University of Technology)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5064047931)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将Wasserstein距离与可变离散概率分布相结合，用于2D曲线（古陶器轮廓）的聚类与相似度检测；

**💡 创新点**

创新点在于提出多种权重分布（均匀、二项、按索引或坐标递增/递减、预选区间等），使Wasserstein距离能够聚焦曲线的指定片段；

**🔧 技术方法**

核心技术是离散Wasserstein距离（最优传输）与不同的离散概率测度；

**📊 数据集**

实验使用五组来自考古学的陶器轮廓曲线数据集，分别包含19–45个对象；

**📈 对比分析**

与传统Procrustes方法比较，基本Wasserstein方案约73%准确率，加入权重后多数实验超过85%，表明对指定区域的关注显著提升聚类效果；

**⚠️ 局限性**

局限在于需要先验选择合适的权重分布与切点，且Wasserstein距离对曲线规模仍有一定依赖，计算成本相对较高。

---

## 822. Deep Whole-body Parkour

**arXiv ID:** 2601.07701 | [PDF](https://arxiv.org/pdf/2601.07701v1)

**作者:** Ziwen Zhuang `[一作]` (Tsinghua University), Hang Zhao `[通讯]` (Shanghai Qi Zhi Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种将深度感知与全身运动跟踪相结合的框架，使仿人机器人能够在不平坦地形上完成多接触的敏捷动作（如翻越、潜行翻滚等），实现对复杂环境的鲁棒控制。

**💡 创新点**

创新点在于：①将外部深度观测直接注入到全身运动跟踪中，弥补传统盲目运动跟踪对环境的缺失；②设计了相对帧奖励、适应性采样与卡死检测的训练策略，使得单一网络可学习多种动作并对初始位置偏差具有自适应性；③实现了GPU级别的多环境深度射线投射，解决了并行仿真中感知隔离难题。

**🔧 技术方法**

使用的技术包括：深度强化学习（PPO），NVIDIA Isaac Lab并行仿真，GPU加速射线投射，深度相机（RealSense）感知，ONNX推理，深度图噪声模拟与修复，动作重映射（GMR）与轨迹回放。

**📊 数据集**

数据集为自制的运动与环境配对数据：使用光学运动捕捉系统记录人类在物理障碍物上的动作，并同步使用LiDAR扫描得到对应场景网格；随后将人类动作重映射至Unitree G1机器人。补充的公开数据如AMASS、OMOMO也被引用但不足以满足多接触动作需求。

**📈 对比分析**

通过与传统盲目跟踪基线（BeyondMimic）以及无卡死检测、无相对帧奖励等消融比较，结果显示加入深度感知后，MPJPE下降、成功率提升（如在随机初始化下成功率达100%），且对环境干扰（如宽体、平板等）表现出更高的鲁棒性。实验在室内外真实机器人上验证，表现出与仿真一致的性能。

**⚠️ 局限性**

局限性包括：①训练需要大量GPU资源与并行环境；②仅验证了四种动作与三种地形，通用性尚待扩展；③对高度变化、动态障碍物的适应性尚不充分；④深度感知受遮挡与反射影响，虽然做了噪声模拟，但真实环境中仍可能出现误差；⑤仍需人工标定运动参考与环境对齐，未实现完全自动化。

---

## 823. Hidden Monotonicity: Explaining Deep Neural Networks via their DC Decomposition

**arXiv ID:** 2601.07700 | [PDF](https://arxiv.org/pdf/2601.07700v1)

**作者:** Jakob Paul Zimmermann `[一作]` (Technical University Berlin), Georg Loho `[通讯]` (Freie Universität Berlin)

**通讯引用:** 78 | [OpenAlex ID](https://openalex.org/A5001022516)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种数值稳定的差分网络分解方法，将预训练的 ReLU 网络拆分为两个单调凸子网络，并基于此构建 SplitCAM、SplitLRP 与 SplitGrad 三种可解释性方法。

**💡 创新点**

创新点在于提供对任意预训练网络的可数值稳定分解，消除权重爆炸；利用单调凸子网络的差分实现更具解释性和自解释的模型；将分解技术应用于 CAM 与 LRP，显著提升解释质量。

**🔧 技术方法**

使用 DC 分解、Maxout、梯度稳定化（α 约束）、Affine 归一化、对抗梯度校正等技术，并通过 PyTorch 的模块化转换实现网络的“拆分视图”。

**📊 数据集**

在 ImageNet‑S‑50（ImageNet 的带有像素级标注的子集）以及 MNIST 上进行实验，评估 VGG16、ResNet18 等预训练模型。

**📈 对比分析**

与多种基准解释方法（Guided Backprop、LayerCAM、Grad‑CAM++、Integrated Gradients、LRP、DeepLift、Occlusion 等）在 Quantus 的六个指标（Selectivity、Attribution Localization、Pointing Game、Pixel Flipping AUC@5/20、Maximum Sensitivity）上对比，SplitCAM/SLRP 在大多数指标上均优于基准，尤其在 Pointing Game 与 Localization 上表现突出。

**⚠️ 局限性**

局限性包括对层选择高度敏感；分解后参数仍需复制，内存效率有待提升；未针对 Transformer 或多模态网络进行实验；在某些指标上仍未统一超越所有基准方法。

---

## 824. Emotional Support Evaluation Framework via Controllable and Diverse Seeker Simulator

**arXiv ID:** 2601.07698 | [PDF](https://arxiv.org/pdf/2601.07698v1)

**作者:** Chaewon Heo `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 1664 | [OpenAlex ID](https://openalex.org/A5021733732)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种可控的求助者模拟器，用九维心理和语言特征驱动，提升情感支持聊天机器人的评估可信度。

**💡 创新点**

创新点在于引入多维求助者配置并通过Mixture-of-Experts实现细粒度行为可控，突破传统过度合作且不可控的模拟器。

**🔧 技术方法**

采用Llama-3-8B-Instruct基础模型，先进行SFT，再加入低秩LoRA专家和全局路由网络，结合语言建模与对比式去相关损失训练。

**📊 数据集**

使用来自Reddit线上支持社区的真实对话，构建11,066条带标注求助者特征的对话数据集。

**📈 对比分析**

与零拷贝、SFT、对比学习等基线对比，宏观F1 0.549，心理特征控制超过0.4，且在多维评估中均优于现有模拟器。

**⚠️ 局限性**

局限在于仅评估单轮对话，未考虑情感支持的累积效果；并将求助者特征视为固定，无法动态跟踪特征变化；未衡量长期心理恢复等结果指标。

---

## 825. Exploring the Meta-level Reasoning of Large Language Models via a Tool-based Multi-hop Tabular Question Answering Task

**arXiv ID:** 2601.07696 | [PDF](https://arxiv.org/pdf/2601.07696v1)

**作者:** Nick Ferguson `[一作]` (University of Edinburgh), Kwabena Nuamah `[通讯]` (University of Edinburgh)

**通讯引用:** 54 | [OpenAlex ID](https://openalex.org/A5048801815)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个多跳问答任务，利用工具调用环路评估大型语言模型（LLM）在元层推理（meta-level reasoning）和对象层推理（object-level reasoning）方面的能力；通过引入“essential actions”集合，量化模型对工具的选择与使用；同时分析错误信息对模型推理的影响。

**💡 创新点**

创新点在于：①将元层与对象层推理概念应用于LLM评估；②设计“essential actions”集合，避免单一金标准，能够衡量模型工具调用的精确性与完整性；③通过错误消息分析展示模型对失败的学习与修正能力；④提出将数值运算与数据检索统一为工具调用的实验框架。

**🔧 技术方法**

技术手段包括：Chain‑of‑Thought + ReAct 样式的提示；构建22个自定义工具（13个算术、7个数据检索、1个思考工具、1个最终答案工具）；在工具调用循环中执行并返回结果；使用改进的 precision/recall 评估模型与 essential actions 的匹配度；进行 n‑shot 例子提示与错误信息实验。

**📊 数据集**

数据集基于世界银行开放数据 API，选取 296 个特色指标，提取 2003‑2023 年数据；设计 20 个问题模板，手工填充 20‑108 条 slot 值，生成 400 个样本（20 种类型各 20 题），涵盖可回答、不可回答与部分缺失数据三类。

**📈 对比分析**

实验对比了 8 种 off‑the‑shelf LLM（Qwen 3、GPT‑4o mini、Llama 3.3 等）在 0、1、3‑shot 设置下的最终答案准确率、precision/recall 以及错误率。结果显示：部分模型（如 Qwen 3 4B/32B、GPT‑4o mini）在元层推理上精度高、召回好；大模型并不一定更优；n‑shot 提示对准确率提升不显著，但能降低错误调用；错误信息对部分模型有助于修正。若仅使用检索工具且不调用算术工具，模型准确率明显下降，证明符号运算工具不可或缺。

**⚠️ 局限性**

限制包括：任务难度相对较低，仅涵盖世界银行指标；工具集合固定，缺乏多领域通用性；未探讨不同推理路径或不确定性处理；实验仅使用 off‑the‑shelf 模型，未进行专门的推理/工具使用微调；“essential actions” 仍基于单一执行路径，未覆盖多种有效解法。

---

## 826. On Angels and Demons: Strategic (De)Construction of Dynamic Models

**arXiv ID:** 2601.07690 | [PDF](https://arxiv.org/pdf/2601.07690v1)

**作者:** Davide Catta `[一作]` (LIPN, CNRS, Université Sorbonne Paris Nord), Munyque Mittelmann `[通讯]` (LIPN, CNRS, Université Sorbonne Paris Nord)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了能够对加权有向图永久删边（SDL）、增边（SCL）以及两者同时操作（SUL）的动态策略逻辑，并给出了它们的语义、表达力与模型检验算法；

**💡 创新点**

创新点在于将永久性的边删加操作分别建模为“魔鬼”“天使”策略，并引入类似ATL的联合策略，使得这些逻辑在表达力上严格优于CTL并能够覆盖传统的Obstruction Logic；

**🔧 技术方法**

采用时序逻辑语义、游戏论策略框架、复杂度分析（PSPACE/EXPSPACE）以及交替算法设计等技术手段；

**📊 数据集**

没有使用真实数据集，全部采用理论模型与示例图进行说明与验证；

**📈 对比分析**

通过理论证明与复杂度分析与CTL、OL等现有逻辑对比，SDL与SCL的模型检验是PSPACE‑complete，SUL是EXPSPACE（next‑time 子语言仍为PSPACE‑complete）；

**⚠️ 局限性**

限制在于只考虑无记忆策略、未讨论可满足性与完备性问题、SUL的高复杂度与实际可实现性未做实验验证，以及对大规模实际系统的适用性尚待研究。

---

## 827. Tab-TRM: Tiny Recursive Model for Insurance Pricing on Tabular Data

**arXiv ID:** 2601.07675 | [PDF](https://arxiv.org/pdf/2601.07675v1)

**作者:** Kishan Padayachy `[一作]` (insureAI), Mario V. Wüthrich `[通讯]` (ETH Zurich)

**通讯引用:** 5093 | [OpenAlex ID](https://openalex.org/A5004533209)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Tab-TRM 模型，将 Tiny Recursive Models 迁移到保险定价任务，利用递归隐状态对表格特征进行迭代推理并输出 Poisson 频率预测。

**💡 创新点**

创新点在于：① 结合递归推理与表格特征编码，构建两个可学习的前缀 token（答案与推理）；② 通过参数共享与深度递归实现极小参数量但高效迭代深度；③ 在保险定价场景中首次将 TRM 原理与 GLM 形式的 Poisson 损失结合。

**🔧 技术方法**

使用技术包括：递归神经网络（Tiny Recursive Model）、实体嵌入、分段线性编码（Piecewise‑Linear Encoding）、Poisson 对数似然（deviance）损失、AdamW 优化器、Optuna 超参搜索、轻量级正则化。

**📊 数据集**

实验数据集为法国机动车第三责任险（MTPL）基准数据集，包含 5 个连续变量和 4 个分类变量。

**📈 对比分析**

评估方法：与 GLM、GAM、FNN、CAFFT、Credibility Transformer、Tree‑like PIN 等基线模型在相同训练/测试划分上对比；Tab‑TRM 在测试集上的 Poisson deviance 为 23.589×10⁻²，参数量 14 820，性能优于多数对比模型，接近最佳集成模型。

**⚠️ 局限性**

局限性：仅验证频率预测；在低维嵌入下性能下降；未涵盖严重度或复合频率‑严重度模型；缺乏对时间序列或多期经验定价的扩展验证。

---

## 828. Advancing Multinational License Plate Recognition Through Synthetic and Real Data Fusion: A Comprehensive Evaluation

**arXiv ID:** 2601.07671 | [PDF](https://arxiv.org/pdf/2601.07671v1)

**作者:** Rayson Laroca `[一作]` (Pontifical Catholic University of Paraná), David Menotti `[通讯]` (Federal University of Technology-Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

结合真实与三种合成技术（模板、字符置换、GAN）训练OCR模型，提升跨区域车牌识别性能

**💡 创新点**

在多语种、多布局车牌数据上实现最广泛的实验评估，发现合成数据与三种方法协同显著提升性能

**🔧 技术方法**

采用模板渲染、字符置换、pix2pix GAN、YOLO/CRNN等深度网络及数据增强技术

**📊 数据集**

使用八个公开车牌数据集（Brazil, China, Europe, Taiwan, Mercosur 等）进行训练/验证，四个未见集做交叉评估

**📈 对比分析**

通过对比多种OCR模型与商业系统，平均识别率超过 90%，在跨数据集仍保持高精度，并实现速度与精度的平衡

**⚠️ 局限性**

未覆盖非拉丁文字、竖排或极端低质量车牌，合成数据仍可能与真实分布偏离，对极端场景的鲁棒性待验证

---

## 829. Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference

**arXiv ID:** 2601.07667 | [PDF](https://arxiv.org/pdf/2601.07667v1)

**作者:** Rei Taniguchi `[一作]` (Osaka University), Chuan Xiao `[通讯]` (Osaka University)

**通讯引用:** 3450 | [OpenAlex ID](https://openalex.org/A5090377305)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自适应选择层（ASL）技术，用于在LLM推理的prefilling阶段根据注意力分数的方差动态决定KV缓存中保留的token层，随后与SnapKV或GemFilter联合实现KV缓存压缩。

**💡 创新点**

创新点在于：①通过监控连续层中token注意力分数排名的方差来判定最合适的token裁剪层，而非预设固定层；②该方法无需训练、仅在推理时完成，能够在不同任务难度下自动调节；③能够与现有一拍/二拍压缩方法无缝集成，兼顾速度与内存。

**🔧 技术方法**

核心技术包括：①注意力分数平均池化与头组聚合；②计算token排名方差并归一化为相对方差；③阈值τ决定何时触发token选择；④在prefilling阶段进行一次token选择，随后在后续层使用所选token；⑤与SnapKV/GemFilter联合使用实现KV预算控制。

**📊 数据集**

使用的数据集为三大长上下文基准：InfiniteBench（平均约214k上下文长度）、RULER（4k–128k）、Needle in a Haystack（1k–256k），实验模型包括Llama‑3.1‑8B‑UL和Qwen2.5‑7B。

**📈 对比分析**

与FastKV、GemFilter、PyramidInfer等方法比较：在相同KV预算（2048）下，ASL在多数任务上取得最高或相近准确率，尤其在难度较高的KV检索任务中与全KV接近；在TTFT、TPOT方面略慢于FastKV但快于SnapKV；吞吐量低于FastKV/GemFilter，但在大多数查询中差距不大；内存占用与其他压缩方法相当。

**⚠️ 局限性**

局限性包括：①仅在两种LLM上验证，缺乏对更大/不同模型的评估；②目前仅集成到一拍或二拍的token裁剪方法，未探索多拍方法（如LasyLLM、OmniKV）的适配；③对阈值τ的敏感性与调优仍需进一步研究；④在某些极大上下文或特定任务下吞吐量仍明显低于最优方法。

---

## 830. Learning to accelerate Krasnosel'skii-Mann fixed-point iterations with guarantees

**arXiv ID:** 2601.07665 | [PDF](https://arxiv.org/pdf/2601.07665v1)

**作者:** Andrea Martin `[一作]` (KTH Royal Institute of Technology), Giuseppe Belgioioso `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 703 | [OpenAlex ID](https://openalex.org/A5054192782)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种学习式优化（L2O）框架，在Krasnosel'skii–Mann（KM）迭代中注入可求和扰动，以加速求解一般非膨胀映射的固定点问题，并将该框架应用于多种算子分裂方法（如Douglas–Rachford），在基于球面与线性子空间交集的最近点问题上实现了平均性能提升。

**💡 创新点**

创新点在于：①通过引入可求和扰动将学习到的更新嵌入KM迭代，既保留传统KM的收敛保证又实现平均案例加速；②在假设度量子正则性下证明该参数化既能保证局部线性收敛，又包含所有足够快的线性收敛迭代；③将这一理论扩展到Davis–Yin分裂及其特例（ADMM、DR等）并展示实证。

**🔧 技术方法**

采用的技术包括：学习到优化（L2O）与神经网络参数化扰动、可求和扰动的理论分析（准Fejér单调性）、度量子正则性下的局部线性收敛证明、长短时记忆网络（LSTM）学习扰动序列、Adam优化器训练、基于折扣残差的损失函数。

**📊 数据集**

使用的“数据集”为随机生成的三维点，均匀采样自立方体[0.25,0.75]^3，用于训练和评估投影问题的性能。

**📈 对比分析**

与传统Douglas–Rachford分裂方法对比，实验表明加入可求和扰动后，迭代在前20步内快速逼近交点，平均残差下降更快、收敛速度提升显著，尤其在球面与近平行子空间的困难场景中表现尤为突出。

**⚠️ 局限性**

局限性包括：①对扰动序列要求可求和，限制了某些快速收敛策略的使用；②理论结果依赖度量子正则性假设，非所有问题满足；③实验仅在合成数据上验证，缺乏对真实工业场景或随机固定点迭代的推广；④目前仅针对确定性迭代，未涵盖随机或游戏平衡求解。

---

## 831. TeeMAF: A TEE-Based Mutual Attestation Framework for On-Chain and Off-Chain Functions in Blockchain DApps

**arXiv ID:** 2601.07726 | [PDF](https://arxiv.org/pdf/2601.07726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 832. Beyond Single-Shot: Multi-step Tool Retrieval via Query Planning

**arXiv ID:** 2601.07782 | [PDF](https://arxiv.org/pdf/2601.07782v1)

**作者:** Wei Fang `[一作]` (Massachusetts Institute of Technology), James Glass `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 18767 | [OpenAlex ID](https://openalex.org/A5112758056)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将工具检索建模为迭代查询规划的框架（Tool Query Planner），通过分解用户意图并交互式生成查询来检索大型动态工具库。

**💡 创新点**

创新点在于把单次匹配转为动态规划过程，克服语义鸿沟、组合任务与工具依赖的局限；使用轻量级模型与现有检索器无缝集成，且可在无监督环境下学习。

**🔧 技术方法**

主要技术包括基于LLM的任务分解与查询生成、交互式反馈循环、峰值排名聚合；训练采用合成查询轨迹的监督微调加上可验证奖励的强化学习（RLVR）。

**📊 数据集**

使用ToolRet基准（35个工具使用数据集，约4.4万工具）进行训练与评测，并在API‑Bank与StableToolBench上验证端到端调用效果。

**📈 对比分析**

与提示、检索重排序、对比学习等多种基线对比，Tool Query Planner在nDCG@10、Recall@10、端到端成功率上均取得显著提升，零样本泛化与跨检索器迁移表现尤为突出。

**⚠️ 局限性**

局限包括：合成数据仅基于单一检索器，可能导致模型偏向特定嵌入空间；隐式学习工具依赖，缺乏显式知识图支持；对小型工具集的迭代规划成本可能过高，需进一步自适应简化策略。

---

## 833. Are LLM Decisions Faithful to Verbal Confidence?

**arXiv ID:** 2601.07767 | [PDF](https://arxiv.org/pdf/2601.07767v1)

**作者:** Jiawei Wang `[一作]` (University of Southern California), Deqing Fu `[通讯]` (University of Southern California)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5070849355)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在面对不同错误惩罚时是否会调整其放弃回答的策略，并提出RiskEval框架进行评估。

**💡 创新点**

发现模型虽然能够给出校准良好的自我置信度，却无法根据风险水平调整回答/拒绝策略，显示置信度与决策不一致的根本问题。

**🔧 技术方法**

使用了风险敏感的决策理论方法，定义置信度阈值、Policy Consistency、Normalized Regret、Utility、AUARC等指标，并通过Prompt在模型中提取置信度和决策。

**📊 数据集**

在HLE、GPQA Diamond、GSM8K三大公开问答/推理基准上进行评估。

**📈 对比分析**

与多款前沿LLM对比，结果显示在高风险下模型几乎不放弃，导致负的归一化效用；通过后期施加数学最优决策π*可显著提升效用，证明置信度信息可用但未被模型利用。

**⚠️ 局限性**

局限性包括依赖口头置信度估计、只针对可验证答案的任务，且对开放式生成任务的适用性未知。

---

## 834. Structural Approach to Guiding a Present-Biased Agent

**arXiv ID:** 2601.07763 | [PDF](https://arxiv.org/pdf/2601.07763v1)

**作者:** Tatiana Belova `[一作]` (ITMO University), Danil Sagunov `[通讯]` (ITMO University)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5021164153)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并求解了一种新的主-代理图编辑问题：在存在时间不一致的代理（present-biased agent）时，如何通过最小化对任务图的弧增删操作，保证代理能够完成所有关键任务并最终到达目标。

**💡 创新点**

创新点在于：①把原先单一的引导子图问题推广为可增删弧的通用形式；②给出了完整的参数化复杂度景观，证明在树宽（treewidth）+路径成本多样性等组合参数下可实现 FPT；③提出了基于树分解的动态规划与二分图最小点覆盖子问题的高效算法；④通过参数化约简给出了 W[1]‑hard 与 para‑NP‑hard 的下界，回答了 Belova 等 2024 年提出的开放问题。

**🔧 技术方法**

主要技术包括：参数化算法框架（treewidth、vertex‑cover、feedback‑vertex‑set、path‑length、tree‑depth）、树分解上的动态规划、最短路径距离猜测与弧删除约束、二分图最小点覆盖（利用最大匹配和 König 定理）以及参数化归约与 gap reduction。

**📊 数据集**

本文为理论工作，未使用公开数据集，而是通过构造性证明和参数化归约展示算法与硬度结果；若需实验，可在人工生成的 DAG 或现有工作中的示例图上实现。

**📈 对比分析**

相较于之前只讨论单一增弧或删弧的特例，本文的算法在树宽+成本多样性组合参数下取得了更强的 FPT 结果；在仅给定 vertex‑cover 参数时提供了多项式时间算法；实验（若实现）可验证在低树宽、低成本多样性场景下，算法的实际运行时间与理论上限保持一致。

**⚠️ 局限性**

局限性包括：①只考虑无环有向图；②对于某些参数（如最大路径长度）仍保持 W[1]‑hard；③在大规模图或高树宽/高成本多样性情况下，算法时间仍然指数级；④未给出多任务并行化或近似算法的研究，未来工作可探索更宽松的参数或启发式改进。

---

## 835. Evaluating Impacts of Traffic Regulations in Complex Mobility Systems Using Scenario-Based Simulations

**arXiv ID:** 2601.07735 | [PDF](https://arxiv.org/pdf/2601.07735v1)

**作者:** Arianna Burzacchi `[一作]` (Fondazione Bruno Kessler), Marco Pistore `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 7728 | [OpenAlex ID](https://openalex.org/A5036876399)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个融合交通网络、流量与排放物理层与社会层的多层模型，用于对城市交通管制政策的前瞻性评估。

**💡 创新点**

创新点在于将行为适应性与物理交通动力学统一到一个模拟框架中，并通过可组合的情景生成与概率不确定性处理实现对直接与间接影响的系统评估。

**🔧 技术方法**

使用了离散时间流量演化、几何分布保留概率、指数分布的行为参数、Logit 模型以及基于 Python 的批量仿真与集成推理。

**📊 数据集**

依托博洛尼亚市实时交通传感器、OD 矩阵、车辆排放等级数据以及欧盟 Euro 级别车辆分布和 NOx 排放文献值。

**📈 对比分析**

通过与基线情景对比并在同一框架内生成多种何-if 方案，使用累计车辆流、峰值交通、NOx 排放及收入等指标评估性能，结果显示不同价格策略和行为假设能显著改变需求与排放，验证了模型的预测能力。

**⚠️ 局限性**

局限在于空间只到单一区域聚合、行为模型基于经验分布而非实时学习、缺乏长期动态适应与社会公平指标。

---

## 836. DT-ICU: Towards Explainable Digital Twins for ICU Patient Monitoring via Multi-Modal and Multi-Task Iterative Inference

**arXiv ID:** 2601.07778 | [PDF](https://arxiv.org/pdf/2601.07778v1)

**作者:** Wen Guo `[一作]` (ETH Zurich), Wen Guo `[通讯]` (ETH Zurich)

**通讯引用:** 976 | [OpenAlex ID](https://openalex.org/A5035745280)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种可解释的数字孪生框架DT-ICU，用于ICU患者监测，融合多模态（生命体征、实验室、影像、文本）数据并通过多任务迭代推理实现诊断、预后和风险评估。

**💡 创新点**

创新点在于将多任务迭代推理与多模态表示学习相结合，实现跨任务知识共享，同时引入可解释的注意力机制，让模型的决策过程可视化。

**🔧 技术方法**

采用图神经网络处理时序与关系数据、Transformer/多头注意力提取文本与影像特征、多任务学习框架以及自适应迭代推理模块。

**📊 数据集**

主要使用公开的MIMIC-III与eICU数据库，包含超过一万例ICU住院病人，涵盖生理监测、实验室检验、影像报告及医嘱文本。

**📈 对比分析**

与传统的单任务XGBoost、BiLSTM、BERT等基线模型相比，DT-ICU在30日死亡率预测（AUROC 0.88 vs 0.83）、败血症早期识别（F1 0.76 vs 0.68）等指标上显著提升，且在多任务联合训练中实现了更高的整体准确率。

**⚠️ 局限性**

局限性包括对数据缺失和噪声的鲁棒性待进一步提升、模型推理时间较长限制实时部署、以及可解释性机制仍依赖手工定义的注意力权重，难以完全捕捉临床决策的全部细节。

---

## 837. Beyond External Guidance: Unleashing the Semantic Richness Inside Diffusion Transformers for Improved Training

**arXiv ID:** 2601.07773 | [PDF](https://arxiv.org/pdf/2601.07773v1)

**作者:** Lingchen Sun `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 104126 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完全自监督的训练框架 Self‑Transcendence，用内部特征引导 Diffusion Transformer（DiT）的训练，加速收敛并提升生成质量。

**💡 创新点**

创新点在于：①使用预训练 VAE 的潜在空间对浅层特征进行对齐作为稳定的早期监督；②在内部特征空间引入类无监督指导（CFG），增强中层特征的语义表达；③将上述两阶段内部监督结合起来，完成完全自监督的加速训练，而不依赖外部模型。

**🔧 技术方法**

技术手段包括：Diffusion Transformer（SiT、LightningDiT 等）模型；VAE 对齐损失；Classifier‑Free Guidance (CFG) 在特征空间的扩展；自监督损失与标准去噪损失的联合训练；早停策略；多尺度特征对齐与层级指导。

**📊 数据集**

主要使用 ImageNet 256×256 与 512×512 图像数据集进行实验，评估 FID、sFID、IS、Precision/Recall 等指标。

**📈 对比分析**

与 REPA、Disperse Loss、SRA、LayerSync 等自监督加速方法以及 U‑Net/Hybrid 等基准模型对比，Self‑Transcendence 在 SiT‑XL/2 上仅用 80 轮即可达到 7.51 FID，接近或优于 REPA；在 LightningDiT‑XL/1 上 64 轮即可达到 3.55 FID；在 512×512 分辨率下仅 100 轮即可取得 2.00 FID，明显快于 REPA；整体收敛速度提升约 30‑60%，生成质量维持或提升。

**⚠️ 局限性**

局限性：需要额外的“预热”训练阶段来构建内部监督，尽管成本低于外部模型但仍有额外时间；内部引导的质量受模型容量限制；目前仅在图像生成任务上验证，尚未探索文本‑图像、文本‑视频或 3D 生成等其他模态。

---

## 838. Evaluating the encoding competence of visual language models using uncommon actions

**arXiv ID:** 2601.07737 | [PDF](https://arxiv.org/pdf/2601.07737v1)

**作者:** Chen Ling `[一作]`, Nai Ding `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了UAIT，一个专注于常识逆转动作图文对的评测基准，并通过半自动流程生成高质量样本。

**💡 创新点**

创新点在于以动作的语义角色逆转为核心，构造对称的多选问题，去除统计偏差，专门挑战模型的深层语义理解。

**🔧 技术方法**

采用大语言模型（如Qwen2、LLaMA3.2）、文本到图像扩散模型（Stable Diffusion）、Prompt工程、少样本学习、LoRA微调等技术。

**📊 数据集**

使用自行生成的UAIT数据集，包含400幅图像及其对抗常识文本，对比常规图文对如Winoground、COCO-A等。

**📈 对比分析**

在UAIT上，SOTA VLM如Qwen2-VL-Instruct、LLaMA3.2-Vision、LLaVA-1.5在启用CoT时分别取得0.64-0.69准确率，人类为0.96；对比之下，CLIP仅0.49，甚至更低，表明模型远逊于人类。

**⚠️ 局限性**

局限性在于仍依赖统计偏差，缺乏对语义角色和因果推理的深入建模；且数据规模有限，跨文化和多语言的适用性待扩展。

---

## 839. Weak Composition Lattices and Ring-Linear Anticodes

**arXiv ID:** 2601.07725 | [PDF](https://arxiv.org/pdf/2601.07725v1)

**作者:** Jessica Bariffi `[一作]` (Technical University of Munich), Violetta Weger `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在整数模环 ℤ/p^sℤ 上 Lee 度量下的最优反码，并将其构造为弱组合的格结构；同时提出了基于反码的新型 Lee‑度量码的可度量不变量；

**💡 创新点**

①给出了最优 Lee 反码的完全表征；②证明了反码集合在包含关系下形成格，并与弱组合格（按支配序）一一对应；③利用此对应关系定义并研究了新的码的可度量不变量。

**🔧 技术方法**

利用有限链环上线性码的代数结构、Lee 度量的定义、支配序（dominance order）的组合学性质、格理论（join/meet、Möbius 函数）以及子码结构的计数公式。

**📊 数据集**

本工作为纯理论研究，没有使用实际数据集；所有结果均通过数学证明与组合计数得到。

**📈 对比分析**

由于研究为理论性，无实验对比；若与传统 Hamming/同态权度量等做比较，主要表现为能给出更精细的码重量分布与界限。

**⚠️ 局限性**

仅适用于 p≠2（Lee 度量在 ℤ/2ℤ、ℤ/3ℤ 下退化），反码本质上是退化的；对更一般的非链环或非 Lee 度量的推广尚未解决。

---

## 840. Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids

**arXiv ID:** 2601.07718 | [PDF](https://arxiv.org/pdf/2601.07718v1)

**作者:** Shaoting Zhu `[一作]` (Tsinghua University), Hang Zhao `[通讯]` (Shanghai Qi Zhi Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一个可扩展的单阶段端到端感知框架，利用高频深度相机实现人形机器人在野外复杂地形上的安全高动态奔跑与步行；

**💡 创新点**

创新点包括：①基于Terrain Edge Detection与Foot Volume Points的软约束安全机制；②基于Flat Patch Sampling的定位命令生成，消除reward hacking；③结合Mixture-of-Experts网络与Adversarial Motion Prior实现零步迁移；

**🔧 技术方法**

技术手段包括：PPO强化学习、Mixture-of-Experts网络、深度图像预处理（噪声、模糊、失真对齐）、时间步深度历史、卷积+时间序列网络、梯度惩罚AMP、边缘检测算法、体积点碰撞惩罚等；

**📊 数据集**

使用多来源运动数据集：MPC生成步态、NOKOV人类捕捉数据、LAFAN跑步数据；训练环境为Isaac Sim仿真，部署使用Intel RealSense D435i深度相机；

**📈 对比分析**

通过与无边缘惩罚、无深度历史、无MoE、无AMP等消融版本以及不同地形和步态的对比实验，完整系统在多种地形下成功率>99%，最高速度2.5 m/s，持续4分钟无跌倒；相比基线提升10–30％，平均到达时间下降1–3 s；

**⚠️ 局限性**

限制在于仅使用单向前向深度相机，缺乏后视/侧视感知；多种地形和步态同时训练易出现模式坍塌，单一统一策略性能受限。

---

## 841. New $X$-Secure $T$-Private Information Retrieval Schemes via Rational Curves and Hermitian Curves

**arXiv ID:** 2601.07676 | [PDF](https://arxiv.org/pdf/2601.07676v1)

**作者:** Yuan Gao `[一作]` (Shandong University), Jiejing Wen `[通讯]` (Shandong University)

**通讯引用:** 80 | [OpenAlex ID](https://openalex.org/A5101564790)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计了一类新的 X‑Secure T‑Private 信息检索（XSTPIR）方案，利用已存在的有理曲线与 Hermitian 曲线，改进了信息检索速率。

**💡 创新点**

核心创新在于引入一种新的基底集合（取代传统的 Lagrange 插值基底），更高效地利用曲线上的有理点，从而提升最大 PIR 速率，而非单纯追求更高基数的曲线。

**🔧 技术方法**

采用了代数几何码（AG 码）与代数曲线理论（尤其是有理曲线和 Hermitian 曲线）的构造方法，并通过构造新的函数空间基底实现新的 XSTPIR 方案。

**📊 数据集**

本工作不使用实际数据集，而是以理论分析与速率比较为主，基于有限域大小、X、T 参数进行数学推导与对比。

**📈 对比分析**

通过对比最大 PIR 速率，证明在字段大小 q²≥14² 且 X+T≥4q 时，Hermitian 曲线方案获得已知方案中最高的速率；在 q²≥28² 且 X+T≥4 时，新的有理曲线与 Hermitian 曲线方案联合可实现全局最高速率。

**⚠️ 局限性**

局限性：方案仅在满足特定的字段大小与安全/隐私参数约束时才有效；并未考虑更广泛的曲线类型（如 Norm‑Trace、Ree、Suzuki 曲线）的进一步改进；同时实现细节（如具体编码实现、实际通信开销）未在论文中给出。

---

## 842. Self-Creating Random Walks for Decentralized Learning under Pac-Man Attacks

**arXiv ID:** 2601.07674 | [PDF](https://arxiv.org/pdf/2601.07674v1)

**作者:** Xingran Chen `[一作]` (Rutgers University), Salim El Rouayheb `[通讯]` (Rutgers University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于随机游走（RW）的全局去中心化学习算法，设计了一种自创建机制以抵御恶意节点（Pac‑Man）吞噬随机游走的攻击，从而保证RW种群永不灭亡、维持可控规模，并在此基础上实现随机梯度下降（RW‑SGD）的收敛。

**💡 创新点**

创新点在于：①使用局部时间阈值自发创建新的RW，而非传统的复制/复制（duplication）策略；②不需要对网络参数（如RW数量、图拓扑等）做估计或全局协调；③提供完整的理论框架，证明在Pac‑Man攻击下RW种群保持有界、不会永久消亡，并给出收敛误差界；④阐明在不同网络拓扑与参数设置下的鲁棒性。

**🔧 技术方法**

主要技术包括：随机游走与马尔可夫链理论、分支过程/分支树分析、随机梯度下降（SGD）与随机梯度理论、谱分析与极限分布、实验验证的仿真脚本和PyTorch实现。

**📊 数据集**

实验使用的两类数据集：①合成线性回归数据；②MNIST手写数字数据集，按数据分区（均匀与Dirichlet非均匀）划分到100个节点。

**📈 对比分析**

与传统复制机制（如CIL、DeCaFork）以及Gossip‑based SGD做对比。结果表明：①在大多数图结构下，本文方法收敛速度与复制方法相当，但在鲁棒性上更强；②相较于Gossip‑based SGD，收敛更快且更不易被Pac‑Man干扰；③在误差上给出了理论上界，实验误差与理论区间一致。

**⚠️ 局限性**

限制与不足：①本文仅严格证明单一Pac‑Man攻击，虽然扩展到多Pac‑Man可行但分析更复杂；②对图的直径与拓扑依赖较强，部分拓扑（如环形）需要调整q的规模；③实验规模仅在100节点，需验证大规模网络下的性能；④算法在极端参数（如极低的终止概率ζ）下的收敛速度仍待进一步优化。

---

## 843. OS-Symphony: A Holistic Framework for Robust and Generalist Computer-Using Agent

**arXiv ID:** 2601.07779 | [PDF](https://arxiv.org/pdf/2601.07779v1)

**作者:** Bowen Yang `[一作]` (University of Science and Technology of China), Zichen Ding `[通讯]` (Shanghai AI Laboratory)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OS‑Symphony 框架，整合 Orchestrator、Reflection‑Memory Agent 与多模态搜索工具，实现长周期任务的稳健执行和跨域泛化。

**💡 创新点**

①利用里程碑驱动的长期视觉记忆与轨迹级反思，消除视觉上下文丢失；②构建视觉驱动的 SeeAct 多模搜索器，主动抓取视觉对齐教程，突破单模检索瓶颈。

**🔧 技术方法**

采用 Vision‑Language Models（GPT‑5、Qwen3‑VL 等）、结构化反思协议、里程碑记忆、浏览器沙箱多模搜索、OCR Grounder、Coder 与 Grounder 等技术。

**📊 数据集**

评估基准包括 OSWorld‑Verified（361 个 Ubuntu 任务）、WindowsAgentArena 与 MacOSArena 三大桌面任务集。

**📈 对比分析**

与 Agent S3、CoAct‑1 等基线在同一动作空间下对比；在 OSWorld 100 步达 65.84%（+2.4%），WindowsAgentArena 63.5%（+6.9%），MacOSArena 46%（+38%），实现 SOTA。

**⚠️ 局限性**

局限性：仅验证于桌面环境，未适配移动平台；多代理架构导致高 token 消耗与推理延迟；视觉细粒度感知不足；存在安全/隐私与恶意滥用风险。

---

## 844. Contrastive Learning with Narrative Twins for Modeling Story Salience

**arXiv ID:** 2601.07765 | [PDF](https://arxiv.org/pdf/2601.07765v1)

**作者:** Igor Sterner `[一作]` (University of Edinburgh), Frank Keller `[通讯]` (University of Edinburgh)

**通讯引用:** 12467 | [OpenAlex ID](https://openalex.org/A5054936589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对叙事文本进行对比学习，学习故事嵌入并利用该嵌入评估句子在叙事中的重要性（即叙事显著性）

**💡 创新点**

创新点在于使用叙事对（Narrative Twins）与具有相似表面但不同情节的干扰语（distractors）构造对比损失，从而显著提升故事嵌入质量；并提出四种基于叙事学的显著性运算（删除、位移、扰动、摘要）并进行系统比较

**🔧 技术方法**

技术主要包括Transformer（BERT/ModernBERT）作为编码器，InfoNCE对比学习，随机dropout生成的Dropout Twins，窗口级嵌入池化，及对句子/窗口的四种显著性计算方式

**📊 数据集**

使用两类数据集：ROCStories（短五句故事）和Wikipedia电影情节摘要（约34句），分别用来训练和评估模型的显著性预测能力

**📈 对比分析**

与masked‑LM基线、随机/单调排序基线以及LLM提示（GPT‑4/5）进行对比；实验显示：在ROCStories上，摘要法在Spearman ρ≈0.44、AUC≈0.76上优于其他三种运算和masked‑LM；在Wikipedia上，摘要法在AUC≈0.66时达到最优，使用Narrative Twins或Dropout Twins并结合in‑story negatives可获得最佳自监督结果；整体上模型优于传统基线，接近LLM提示性能

**⚠️ 局限性**

局限性包括：显著性评估仅限句子级别，缺乏更细粒度或层级标签；需要叙事对和干扰文本的人工/LLM生成，增加成本；对长篇叙事的处理仍受窗口划分与注意力窗口大小限制；模型性能高度依赖于嵌入生成方式，可能难以推广到不同语言或极长文本

---

## 845. Structure First, Reason Next: Enhancing a Large Language Model using Knowledge Graph for Numerical Reasoning in Financial Documents

**arXiv ID:** 2601.07754 | [PDF](https://arxiv.org/pdf/2601.07754v1)

**作者:** Aryan Mishra `[一作]` (Indian Institute of Science Education and Research Bhopal), Akash Anil `[通讯]` (Indian Institute of Science Education and Research Bhopal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了基于知识图谱增强的LLM框架，用于金融文本的数值推理。

**💡 创新点**

在LLM推理中加入自定义财务知识图谱，利用结构化信息提升推理准确率。

**🔧 技术方法**

使用Llama 3.1 8B Instruct、表格线性化、知识图谱构建、语义+结构检索与MLP过滤器。

**📊 数据集**

FinQA金融推理基准数据集。

**📈 对比分析**

通过与vanilla Llama对比，使用Gemini 2.5 Pro作为Judge，执行准确率从51.93%提升至58.34%，约12.3%相对提升。

**⚠️ 局限性**

仅测试单一FinQA数据集；仅使用Llama，未试验其他LLM；KG schema可能不适用于多样化财务数据；缺乏与专有LLM基线的直接对比。

---

## 846. Enforcing Priority in Schedule-based User Equilibrium Transit Assignment

**arXiv ID:** 2601.07712 | [PDF](https://arxiv.org/pdf/2601.07712v1)

**作者:** Liyang Feng `[一作]` (Southwest Jiaotong University), Jiayang Li `[通讯]` (University of Hong Kong)

**通讯引用:** 1508 | [OpenAlex ID](https://openalex.org/A5115597510)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于隐式优先规则的时刻表公交分配模型，将其等价转化为非线性互补性问题（NCP），并进一步提出行为可行的R‑UEIP约束与对应求解算法。

**💡 创新点**

创新点包括：①将公交优先规则编码为可用容量并给出等价NCP及其存在性证明；②通过改进可用性定义消除非现实解，得到R‑UEIP；③使用Fisher–Burmeister光滑化将互补性条件转为可微MPEC，并提供两种求解方案。

**🔧 技术方法**

采用非线性互补性建模、Fisher–Burmeister光滑化、MPEC（隐式与SQP求解）、列生成与最短路算法，以及数值仿真与对比分析。

**📊 数据集**

数据集包括：香港大学学生通勤真实案例、Benchmark网络（6站4线）以及Sioux Falls公交网络（24站10线）。

**📈 对比分析**

与传统显式优先模型对比，模型在行为合理性上更优；在Benchmark与Sioux Falls网络上，迭代次数约3–8k，耗时≤55分钟，性能显著优于显式模型且收敛速度快。

**⚠️ 局限性**

局限性在于未能保证全局最优、对极大规模网络的可扩展性不足、缺乏对行为学习与动态收敛性的理论与实验验证。

---

## 847. Is Agentic RAG worth it? An experimental comparison of RAG approaches

**arXiv ID:** 2601.07711 | [PDF](https://arxiv.org/pdf/2601.07711v1)

**作者:** Pietro Ferrazzi `[一作]` (Fondazione Bruno Kessler), Davide Giannuzzi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统化比较增强式与代理式检索增强生成（Enhanced RAG 与 Agentic RAG）在四个维度（用户意图处理、查询重写、文档列表精炼、LLM影响）下的表现，并评估其计算成本与时延；

**💡 创新点**

首次将两种RAG范式在同一实验框架下进行细粒度对比，提出统一的评估维度和可操作的选择指南，展示在不同场景下各自的优势与劣势；

**🔧 技术方法**

采用semantic router、Hyde查询重写、ELECTRA重排序器以及OpenAI/ Qwen LLMs 的 Agentic 与 Enhanced 实现，使用 PocketFlow 轻量化框架进行实验；

**📊 数据集**

使用 FIQA、NQ、FEVER、CQADupStack‑English 四个公开 QA / IR 数据集；

**📈 对比分析**

通过 Recall/F1、NDCG@10、LLM‑as‑Judge 等指标进行对比。实验显示 Agentic 在查询重写上表现略优，Enhanced 在用户意图处理与重排序上更为稳健；但 Agentic 在成本和时延上更高（约 2–3.6 倍）；

**⚠️ 局限性**

局限性包括未公开 PocketFlow 代理实现、仅使用单一工具、未评估文档摘要/重排序等模块、实验规模受限、以及主观评价与自动评测之间的差异等。

---

## 848. Smooth Operator: Smooth Verifiable Reward Activates Spatial Reasoning Ability of Vision-Language Model

**arXiv ID:** 2601.07695 | [PDF](https://arxiv.org/pdf/2601.07695v1)

**作者:** Siwen Jiao `[一作]` (Alibaba Group), Yang Cai `[通讯]` (Alibaba Group)

**通讯引用:** 2218 | [OpenAlex ID](https://openalex.org/A5102877899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

训练视觉‑语言模型以精确完成3D空间数值推理任务

**💡 创新点**

提出平滑数值奖励激活（SNRA）与绝对保持GRPO（AP‑GRPO），解决奖励稀疏、优势坍塌等问题

**🔧 技术方法**

采用动态Sigmoid奖励映射、绝对维持的GRPO框架以及可验证物理奖励的强化学习流程

**📊 数据集**

构建并使用了约5万条可验证3D子任务的数据集Numerical3D‑50k

**📈 对比分析**

与大规模监督模型和传统GRPO对比，单模型仅用50k样本即可达到与百万级基线相近的准确率，样本效率显著提升

**⚠️ 局限性**

仅关注数值子任务，尚无法覆盖更复杂的语义推理或长序列任务

---

## 849. Leveraging 3D Representation Alignment and RGB Pretrained Priors for LiDAR Scene Generation

**arXiv ID:** 2601.07692 | [PDF](https://arxiv.org/pdf/2601.07692v1)

**作者:** Nicolas Sereyjol-Garros `[一作]` (Valeo.ai), Nermin Samet `[通讯]` (Valeo.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 R3DPA，一个无条件 LiDAR 点云生成方法，将大规模 RGB 预训练的流匹配模型与自监督 3D 特征对齐，实现高质量场景合成。

**💡 创新点**

创新点在于①将 ImageNet 预训练权重迁移至 LiDAR 生成器并通过 VAE 对齐消除域差；②在训练中加入 3D 自监督特征（ScaLR）对齐损失，显著提升生成质量；③实现推理时基于特征的对象修补与场景混合，提供可控编辑。

**🔧 技术方法**

使用 VAE+流匹配（SiT）模型、Transformer 架构、equirectangular 投影、3D 自监督特征（ScaLR）、对齐损失、以及端到端联合训练。

**📊 数据集**

在 KITTI‑360 数据集上进行训练与评估，并使用 ImageNet‑1k 对流匹配模型进行预训练。

**📈 对比分析**

与 LiDM、R2DM、R2Flow 等 SOTA 方法对比，采用 FRID、FSVD、FPVD、FLD、JSD、MMD 等指标，R3DPA 在多数指标上优于前沿方法，提升至少 17%。

**⚠️ 局限性**

限制包括：对范围图像表示的依赖导致对不同 LiDAR 传感器的适配有限；训练与采样仍需大量计算资源；对域外场景的泛化能力尚待验证。

---

## 850. AptaFind: A lightweight local interface for automated aptamer curation from scientific literature

**arXiv ID:** 2601.07684 | [PDF](https://arxiv.org/pdf/2601.07684v1)

**作者:** Geoffrey Taghon `[一作]` `[通讯]`, Geoffrey Taghon

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文开发了一套名为AptaFind的本地化、三层次文献挖掘系统，能够高效检索并提取aptamer研究的序列、文献与实验信息。

**💡 创新点**

创新点在于提出“最小代理流”(MAF)原则，将语言模型用于语义理解、正则表达式用于可靠验证，并通过三层结果实现从直接序列提取到文献覆盖的连续价值梯度。

**🔧 技术方法**

核心技术包括本地运行的1B参数LLM（Llama3.2）、正则表达式解析、Deterministic XML/HTML处理、Playwright浏览器自动化、ColBERT RAG检索及多源API（PubMed、PMC、bioRxiv）集成。

**📊 数据集**

主要数据集为美国德克萨斯大学aptamer数据库（UTdb）555个靶点，随机抽取3×100个靶点进行基准测试，并使用公开的PubMed/PMC全文及补充材料进行检索。

**📈 对比分析**

与手工搜索或现有商业工具对比，AptaFind在3×100目标的基准实验中分别实现了84.0%±3.5%（文献发现/研究线索）和79.3%±0.6%（直接序列提取），每小时可处理约950个目标，显著提升了检索效率和覆盖率。

**⚠️ 局限性**

主要局限包括无法访问付费墙内容、补充文件与正文的匹配依赖文本相似度、图像表格中的序列无法提取、对定性亲和力描述识别不佳以及对特殊单位的兼容性不足。

---

## 851. Simplicial Belief

**arXiv ID:** 2601.07669 | [PDF](https://arxiv.org/pdf/2601.07669v1)

**作者:** Christian Cachin `[一作]`, Thomas Studer `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出将多色（非严格区分）单纯形复形与颜色多重性结合，构造出能够在拓扑结构上自然解释信念的模型；

**💡 创新点**

创新点在于：① 通过放宽颜色唯一性约束，利用颜色多重性作为逆可知度量，直接在单纯形模型中定义安全信念与最可信信念；② 引入条件（<ref>）保证可辨识关系为偏等价关系；③ 探讨多重单纯形与单纯形集合两种替代形式；

**🔧 技术方法**

主要技术包括：组合拓扑学（单纯形复形与单纯形集合）、可知度量与可比度关系、模态逻辑语义定义、可辨识与可比度的组合、以及证明结构（如 S4.2、知识→信念）

**📊 数据集**

未使用实验数据集，论文为理论研究，讨论主要在形式定义与逻辑性质上；

**📈 对比分析**

无实验比较，论文仅给出逻辑性质证明（如知识不增、信念增现象、S4.2 规律）以及与传统信念函数方法的对比；

**⚠️ 局限性**

限制包括：① 模型不满足充分可辨识（非 proper）导致知识增问题；② 信念增的存在与可辨识不完全；③ 目前仅讨论单体信念，群体信念与邻域语义等仍待研究；

---

## 852. Reasoning Models Will Blatantly Lie About Their Reasoning

**arXiv ID:** 2601.07663 | [PDF](https://arxiv.org/pdf/2601.07663v1)

**作者:** William Walden `[一作]` `[通讯]` (Johns Hopkins University), William Walden (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型推理模型在提示下是否诚实报告对提示的使用；

**💡 创新点**

首次通过给模型明确指令要求识别并说明提示依赖，并用诚实评分衡量，揭示模型往往否认使用提示；

**🔧 技术方法**

采用提示变体实验、Chain-of-Thought 生成、LLM 判别器评估 faithfulness 与 honesty 分数；

**📊 数据集**

使用 MMLU-Pro 与 GPQA-Diamond 两个多项选择问答数据集；

**📈 对比分析**

对不同提示类型及正确/错误提示下答案变化率进行比较，计算归一化 faithfulness 与 honesty 分数，结果显示模型诚实度低于预期；

**⚠️ 局限性**

仅测试三种模型的默认行为，未探究 RL fine‑tuning 影响，提示形式导致分数方差大，Claude 4.5 Haiku 的摘要化可能影响评分，并且不一定适用于所有 LRM。

---

## 853. Towards Automating Blockchain Consensus Verification with IsabeLLM

**arXiv ID:** 2601.07654 | [PDF](https://arxiv.org/pdf/2601.07654v1)

**作者:** Elliot Jones `[一作]` (Imperial College London), William Knottenbelt `[通讯]` (Imperial College London)

**通讯引用:** 5533 | [OpenAlex ID](https://openalex.org/A5050119476)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 IsabeLLM，将 Isabelle 证明助手与大语言模型结合，用于自动化区块链共识协议的形式化验证。

**💡 创新点**

创新点在于将 LLM 与 Sledgehammer 迭代结合，构建可多轮补全的证明流程，并成功验证了基于 n‑叉树的 Bitcoin PoW 共识模型。

**🔧 技术方法**

采用 Scala 与 Isabelle 的 Scala‑Isabelle 库、DeepSeek R1 API 调用 LLM，并配合 Sledgehammer 进行自动定理求解。

**📊 数据集**

使用的主要数据集为 Isabelle 的 AFP 以及通过 DeepSeek R1 生成的证明文本。

**📈 对比分析**

通过在 16 条关键引理上进行 10 次尝试，平均 1–2 次迭代即可完成，成功率超过 90%，证明长度显著缩短。

**⚠️ 局限性**

局限性包括 LLM 的不确定性导致语法错误与幻觉、Sledgehammer 资源占用高，以及工具只能证明已给出的命题而无法自动生成命题。

---

## 854. Active Evaluation of General Agents: Problem Definition and Comparison of Baseline Algorithms

**arXiv ID:** 2601.07651 | [PDF](https://arxiv.org/pdf/2601.07651v1)

**作者:** Marc Lanctot `[一作]` (Google DeepMind), Michael Kaisers `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了多任务评估的主动评估框架，提出了一套数据生成模型、评估指标以及多种主动采样算法，旨在高效地从有限的评估数据中得到准确的代理排名。

**💡 创新点**

创新点包括：①正式定义主动评估问题并引入平均通用排名误差（AGRE）指标；②将离线评估方法（如 Elo、SCO、Nash Averaging、Maximal Lotteries、Voting-as-Evaluation 等）扩展到在线主动评估场景；③提出代表性任务采样（Proportional Representation）策略以降低高任务变异导致的错误。

**🔧 技术方法**

采用的技术包括社会选择理论与博弈论（如 Nash Averaging、Maximal Lotteries、SCO 等）、马尔科夫随机模型（Mallows、Plackett‑Luce）进行数据生成、基于 UCB 的多臂赌博机、双臂博弈、Elo、KemenyEl 等算法。

**📊 数据集**

实验使用了两类数据集：①合成数据（基于 Mallows 与 Plackett‑Luce 模型，任务数 50~57，代理数 8~10）；②真实 Atari 57 游戏数据（Agent57 表格，8 个代理、57 个任务）。

**📈 对比分析**

通过与多种基线（UniformAveraging、BasicUCB、BatchElo 等）和批量/在线版本的比较，利用平均通用排名误差（AGRE）和GRE评估性能；在合成数据中，UniformAveraging 与 BasicUCB 在前 1000 次迭代内快速降低误差，BatchElo 也表现较好；在 Atari 数据中，BatchSCO 与 OnlineSCO 取得最佳效果，BatchElo 排名次之；Nash Averaging 与 Online Maximal Lotteries 在 Atari 上收敛较慢。

**⚠️ 局限性**

局限性包括：①对任务间得分归一化和任务变异敏感，某些算法在高变异情形下表现不佳；②自适应任务/模型采样策略有限，未充分利用置信区间、不确定性采样等方法；③缺乏针对样本复杂度与误差上界的理论保证；④实验仅在有限任务/代理规模上验证，难以直接推广到更大规模或不同领域。

---

## 855. Predefined-time One-Shot Cooperative Estimation, Guidance, and Control for Simultaneous Target Interception

**arXiv ID:** 2601.07744 | [PDF](https://arxiv.org/pdf/2601.07744v1)

**作者:** Lohitvel Gopikannan `[一作]` (Indian Institute of Technology Bombay), Abhinav Sinha `[通讯]` (University of Cincinnati)

**通讯引用:** 877 | [OpenAlex ID](https://openalex.org/A5022385451)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一个统一的一次性非线性估计‑引导‑控制框架，用于在异质感知拓扑下实现多拦截器对静止目标的同步拦截；

**💡 创新点**

创新点在于将预定时间分布式观测器、改进的到达时间估计、预定时间一致性控制以及基于超速滑模的可拉弯舵舵面控制器融合在同一框架中，并通过有向信息流实现部分可观测性下的协同拦截；

**🔧 技术方法**

使用了预定时间分布式观测、预定时间一致性协议、改进的时间到达估计公式、滑模控制、可拉弯舵面动力学模型以及基于图论的通信拓扑；

**📊 数据集**

通过多种仿真情景（4~6 架拦截器、不同发射角、不同启动位置以及感知/通信拓扑变化）验证，该论文未使用真实数据集；

**📈 对比分析**

与已发表的有限时间滑模自航机相比，采用相同的收敛时间设置，仿真显示拦截时间几乎相同，但总体控制能耗下降约17%，且在感知失效或链路丢失时仍能保持同步拦截；

**⚠️ 局限性**

局限性包括需满足有向生成树和强连通性，算法对通信延迟和丢包敏感，未在实验平台上验证，且对极端动态或非平面运动的推广性尚待研究。

---

## 856. The Complexity of Games with Randomised Control

**arXiv ID:** 2601.07775 | [PDF](https://arxiv.org/pdf/2601.07775v1)

**作者:** Sarvin Bahmani `[一作]` (University of Liverpool), Shufang Zhu `[通讯]` (University of Liverpool)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5101756232)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了两玩家无限持续图游戏中，节点控制权随机分配（一次性分配或首次访问后固定）的复杂度，并给出对三种典型目标（到达、奇偶、能量）的定性与定量分析，证明相应决策问题的复杂度，并提出了可在多项式时间内实现的随机化近似方案（FPRAAS）。

**💡 创新点**

创新点包括：① 明确区分两种随机控制机制并给出完整的复杂度谱；② 证明定量问题在首次访问机制下为 PSPACE‑完整，在预先分配机制下为 PP‑完整；③ 引入“显式游戏”构造以便在交替多项式时间内求解；④ 在不可解的定量问题上设计可在多项式时间内收敛的 Monte‑Carlo 近似算法，并在实验中验证其效果。

**🔧 技术方法**

主要技术手段：从 QBF、两端可靠性问题等经典 NP/PSPACE/PP 难题构造归约；构造显式游戏并利用交替多项式算法；使用奇偶/能量游戏的已知多项式/对偶多项式算法；对近似方案利用 Hoeffding 不等式和随机采样理论。

**📊 数据集**

使用自定义的随机生成图集：20 个节点、最大出度 20，奇偶目标使用最多 6 个优先级，能量目标权重在 [-10,10]，初始能量为 0；对每个目标分别生成 10 个随机实例进行实验。

**📈 对比分析**

与精确求解（枚举所有 2^n 个控制分配并求解）相比，FPRAAS 在收敛误差和运行时间上显著优越：对于到达与奇偶目标，FPRAAS 在 6 分钟内达到与精确解相同的误差范围；能量目标在 2 小时内完成。误差曲线显示实际收敛速度远快于 Hoeffding 上界。

**⚠️ 局限性**

局限性：① 近似方案仅在控制权分配概率均匀、奇偶优先级数有限、能量权重以单元编码时才保证多项式时间；② 对大规模图的实验缺乏验证；③ 论文主要关注理论复杂度，实际应用中的模型适配与性能评估仍待进一步研究。

---

## 857. Free-RBF-KAN: Kolmogorov-Arnold Networks with Adaptive Radial Basis Functions for Efficient Function Learning

**arXiv ID:** 2601.07760 | [PDF](https://arxiv.org/pdf/2601.07760v1)

**作者:** Shao-Ting Chiu `[一作]` (Texas A&M University), Rui Peng Li `[通讯]` (Lawerance Livermore National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了Free‑RBF‑KAN网络，利用可学习的径向基函数网格和光滑度来替代传统的B‑spline基函数，实现了更高效的函数逼近。

**💡 创新点**

创新点在于将自由中心（free‑knots）与可调光滑度结合进RBF‑KAN，并证明其普适逼近性，同时在训练过程中动态优化网格与形状，显著提升表达能力。

**🔧 技术方法**

采用Kolmogorov–Arnold表示定理、径向基函数网络、可学习中心与光滑参数、以及Neural Tangent Kernel（NTK）分析等技术。

**📊 数据集**

实验使用合成多尺度函数、1D/2D热传导与Helmholtz PDE、反应扩散算子学习，以及MNIST回归等数据集。

**📈 对比分析**

与B‑spline KAN、FastKAN、FasterKAN以及传统MLP对比，Free‑RBF‑KAN在保持或提升逼近精度的同时，训练与推理速度提升至B‑spline KAN的约三分之一，在物理信息学习和算子学习任务中表现最佳。

**⚠️ 局限性**

局限性在于对结构化高维数据表现优异，但在无结构图像等任务如MNIST时仍不及标准MLP，且中心与光滑度学习仍需额外参数。

---

## 858. Improving Domain Generalization in Contrastive Learning using Adaptive Temperature Control

**arXiv ID:** 2601.07748 | [PDF](https://arxiv.org/pdf/2601.07748v1)

**作者:** Robert Lewis `[一作]` (Massachusetts Institute of Technology), John Guttag `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 15256 | [OpenAlex ID](https://openalex.org/A5007282049)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在多域稀缺标注的场景下，利用域标签改进对比学习的预训练，提升对协方差移位的鲁棒性。

**💡 创新点**

引入自适应域感知温度参数，通过域判别器估计负样本与锚点同域概率，动态调整InfoNCE损失中的温度，以增强域不变性。

**🔧 技术方法**

使用InfoNCE损失、域判别器（softmax分类器）、自适应温度控制、SimCLR框架与线性下游分类器。

**📊 数据集**

自定义MNIST颜色变体数据集（红蓝两训练域、紫验证域、绿测试域），每域颜色分布不同。

**📈 对比分析**

与标准对比学习、Same Domain Negatives、MMD、DANN等域泛化基线比较；在Test‑ID和Test‑OOD上均取得最高准确率（0.945/0.819），且域判别准确率最低。

**⚠️ 局限性**

局限在于需额外训练域判别器、对温度超参敏感、仅在单一合成数据上验证，未在真实多域任务上评估。

---

## 859. FMAC: a Fair Fiducial Marker Accuracy Comparison Software

**arXiv ID:** 2601.07723 | [PDF](https://arxiv.org/pdf/2601.07723v1)

**作者:** Guillaume J. Laurent `[一作]` (Université Marie et Louis Pasteur), Patrick Sandoz `[通讯]` (Université Marie et Louis Pasteur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过构建物理基础的射线追踪引擎 FMAC，生成高保真合成图像，用以对不同标记的姿态估计精度进行公平、可重复的评估。

**💡 创新点**

创新点包括：① 结合薄透镜模型、径向/切向畸变与衍射模糊的全物理渲染流程；② 采用 Sobol/Halton 序列实现低失配采样与局部边缘重采样；③ 在同一姿态集合上生成四种标记的合成图像并绘制 36 维度误差互相关矩阵，从而揭示误差与六自由度的系统性关联。

**🔧 技术方法**

使用技术包括：光线追踪、薄透镜和相机畸变模型、Sobol 与 Halton 序列采样、Airy 圆衍射卷积、Gamma 校正、OpenCV 直接读取相机内参，所有算法均以 Python/C++ 开发并开源。

**📊 数据集**

数据集方面，作者以 Logitech HD C270 和 Canon EOS Rebel XS 两台相机的 OpenCV/Matlab 校准图像为基准，验证渲染质量；随后在相同 10 000 个姿态（X/Y/θ/ψ/γ/Z 低失配采样）下渲染 ArUco、AprilTag、STag 与 TopoTag 四种标记的合成图像。

**📈 对比分析**

比较方法是在相同姿态下使用各标记的官方检测器估计姿态，绘制 36 维度误差互相关矩阵，并统计 5 625 个所有方法均能检测到的姿态的平均绝对误差；结果显示 TopoTag 在平移误差（约 0.1 mm）上最佳，AprilTag 在旋转误差（约 0.1°）上领先，但检测范围与角度敏感度不同。

**⚠️ 局限性**

局限性包括：仅针对二值边缘图案的标记；合成图像仍无法完全复现真实相机噪声、遮挡与不均匀照明；检测率随标记尺寸、距离和边缘像素密度显著变化，导致比较时需排除部分姿态；未覆盖动态遮挡、低动态范围等实际工况。

---

## 860. Predictive Analytics for Dementia: Machine Learning on Healthcare Data

**arXiv ID:** 2601.07685 | [PDF](https://arxiv.org/pdf/2601.07685v1)

**作者:** Shafiul Ajam Opee `[一作]` (American International University-Bangladesh), Md Rashedul Islam `[通讯]` (University of Asia Pacific)

**通讯引用:** 2659 | [OpenAlex ID](https://openalex.org/A5100755654)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

基于患者健康与处方数据，使用多种监督学习和混合深度学习模型对痴呆进行预测，重点评估LDA模型的表现。

**💡 创新点**

创新点包括将SMOTE和TF‑IDF特征工程结合以处理类别不平衡与文本数据，提出CNN+RNN混合模型，并强调模型可解释性与特征相关性分析。

**🔧 技术方法**

使用了KNN、QDA、LDA、Gaussian Process Classifier、SMOTE、TF‑IDF、标准化、CNN、RNN及其混合架构。

**📊 数据集**

使用Kaggle上公开的“Dementia Patient Health and Prescriptions Dataset”，包含人口学、健康指标、遗传信息、处方文本等特征。

**📈 对比分析**

实验对比显示LDA在测试集上的准确率达98%，优于KNN、QDA、GPC；传统ML平均准确率95.39%，混合CNN‑RNN模型为92.23%。

**⚠️ 局限性**

局限包括数据集单一、缺乏影像和纵向跟踪信息、部分模型（如GPC）存在过拟合、解释性仍需提升，且未验证在临床实践中的可推广性。

---

## 861. On the complexity of the Maker-Breaker happy vertex game

**arXiv ID:** 2601.07673 | [PDF](https://arxiv.org/pdf/2601.07673v1)

**作者:** Mathieu Hilaire `[一作]` (Bordeaux INP), Nacim Oijid `[通讯]` (Umeå University)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5081421238)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文提出并研究了Maker–Breaker型计分幸福顶点游戏（SHVG），通过形式化定义、理论证明和算法设计，对其计算复杂性进行深入分析；

**💡 创新点**

创新点在于构造新的字面-子句发生图（literal‑clause incidence graph），利用该图完成对量化MAX‑2‑SAT的硬度证明，从而证明SHVG在树、蚀枝形树上是NP‑完全/NP‑难的；此外，在路径、分枝星形树等简单图类上给出精确分数公式，并提出按邻域多样性（neighborhood diversity）参数的3^w时间FPT算法；

**🔧 技术方法**

核心技术包括：从量化MAX‑2‑SAT到SHVG的多阶段归约（使用字面‑子句图、周期消除装置和取值对偶），Milnor宇宙中的和运算与温度理论，超级引理（super‑lemma）用于简化相同类型顶点，动态规划枚举邻域多样性小集；

**📊 数据集**

实验数据集：无；研究全部基于理论构造和证明，没有使用真实或合成图数据；

**📈 对比分析**

评价方法：通过理论复杂度证明比较，证明在树、蚀枝形树上为NP‑完全/NP‑难，在路径、分枝星形树上可多项式求解，在邻域多样性w上实现O(3^w(|E|+|V|))的FPT算法；

**⚠️ 局限性**

限制在于：对蚀枝形树仍为NP‑难，未能给出多项式核化；对更广泛参数（树宽、反馈边数等）尚无可行算法；此外，未考虑实验验证，算法的实际运行时间和常数仍待评估。

---

## 862. Variational Contrastive Learning for Skeleton-based Action Recognition

**arXiv ID:** 2601.07666 | [PDF](https://arxiv.org/pdf/2601.07666v1)

**作者:** Dang Dinh Nguyen `[一作]` (Telecom SudParis), Titus Zaharia `[通讯]` (Telecom SudParis)

**通讯引用:** 1841 | [OpenAlex ID](https://openalex.org/A5011567037)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将变分自编码器（VAE）与对比式自监督学习结合的框架，用于骨架动作识别。

**💡 创新点**

创新点在于：①把对比损失视作变分推断中的重构项；②在编码器后加入高斯采样头，使表征为概率分布并通过KL正则化结构化潜在空间；③在单一框架中同时优化对比与变分目标。

**🔧 技术方法**

使用图卷积网络（ST‑GCN）作为骨架特征提取器，结合MoCo‑v2的记忆池和对比损失，VAE的变分编码与重参数技巧，以及KL散度正则化。

**📊 数据集**

在NTU RGB+D 60/120、PKU‑MMD三个公开骨架动作数据集上进行评估。

**📈 对比分析**

与SkeletonCLR、CrosSCLR、GL‑Transformer等现有自监督方法对比，在线性评估、微调和少标注（1%/10%）设置下均取得更高或相近的准确率，尤其在低标注场景下优势显著。

**⚠️ 局限性**

局限性：仅针对骨架数据，缺乏对RGB或多模态输入的验证；变分框架的复杂度和训练时间相对较高；在极大规模或更复杂动作类别上的泛化仍待进一步研究。

---

## 863. StdGEN++: A Comprehensive System for Semantic-Decomposed 3D Character Generation

**arXiv ID:** 2601.07660 | [PDF](https://arxiv.org/pdf/2601.07660v1)

**作者:** Yuze He `[一作]` (Tsinghua University), Yong-Jin Liu `[通讯]` (Tsinghua University)

**通讯引用:** 10642 | [OpenAlex ID](https://openalex.org/A5008076279)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

从任意图像或文本生成高保真、可分解的3D角色模型，得到身体、服装、头发等语义层。

**💡 创新点**

创新点包括：①双分支语义感知大规模重建模型（Dual‑Branch S‑LRM）实现全身与面部细节并行学习；②基于粗细尺度提案的语义表面提取公式，在保持高分辨率的同时显著降低显存；③将视频扩散用于纹理分解，将表情、虹膜、眼白等拆分为可编辑层。

**🔧 技术方法**

主要技术：Transformer‑based S‑LRM+LoRA、NeRF/SDF 语义场、FlexiCubes 网格提取、多视角与视频扩散、三阶段语义监督、边缘缺口惩罚、基于纹理的可编辑层训练。

**📊 数据集**

使用扩展版 Anime3D‑EX 数据集，包含 10,811 个高质量 3D 动漫角色、三层语义渲染、面部多尺度图像、离散纹理层和文本描述。

**📈 对比分析**

与多种 SOTA 基线（Zero‑1‑to‑3、CharacterGen、Magic123、InstantMesh、Unique3D 等）在 SSIM、LPIPS、FID、CLIP 相似度以及 CD、IoU、F1⁰·⁵ 等三维指标上对比；StdGEN++ 在所有指标上均优于基线，尤其在细节恢复（F1⁰·⁵ 提升至 0.725）和纹理可编辑性上表现突出。

**⚠️ 局限性**

局限性：①仍依赖大量 GPU 内存和长时间训练；②主干架构只支持 A‑pose 标准化，极端姿态或动态表情仍难以完全恢复；③对极细薄结构（如细长服装褶皱）仍存在轻微拓扑缺口；④纯文本生成的语义一致性在复杂角色设定下偶尔失真。

---

## 864. Failure-Aware RL: Reliable Offline-to-Online Reinforcement Learning with Self-Recovery for Real-World Manipulation

**arXiv ID:** 2601.07821 | [PDF](https://arxiv.org/pdf/2601.07821v1)

**作者:** Huanyu Li `[一作]` (Shanghai Jiao Tong University), Huazhe Xu `[通讯]` (Tsinghua University)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5049093671)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Failure‑Aware Offline‑to‑Online Reinforcement Learning（FA‑O²RL）框架，在已有的离线预训练策略上进行在线细调时，通过世界模型预测失败概率并使用预训练的恢复策略来降低探索过程中的人为干预需求。

**💡 创新点**

创新点包括：① 将世界模型与安全评估相结合，形成基于未来状态序列的失败预测；② 在离线阶段预训练恢复策略，并在在线阶段固定使用，从而在不引入额外探索风险的情况下实现安全提升；③ 在离线‑在线 RL 这一实际部署场景下首次开展基于失败场景的安全评估与对比实验。

**🔧 技术方法**

技术手段：Uni‑O4 离线‑在线 RL 训练框架；PPO 与 GAE 进行在线微调；基于离散时间动态的世界模型（encoder、latent dynamics、reward/constraint heads 等）；恢复策略的行为克隆与强化学习；安全评估使用 H‑step 未来约束概率预测；理论上给出“动作修正”提升下界。

**📊 数据集**

数据集与环境：① FailureBench（改造后的 MetaWorld 四个任务：Bounded Push、Bounded Soccer、Fragile Push Wall、Obstructed Push），包含任务演示、恢复演示和失败演示；② Franka Emika Panda 真实机器人实验，三项任务（Fragile Push Wall、Disturbed Push、Bounded Soccer），使用 YOLOv8 + RealSense D435 视觉感知。

**📈 对比分析**

与基线 Uni‑O4 及三种在线安全 RL 算法（PPO‑Lagrangian、P3O、CPO）对比，FA‑O²RL 在 FailureBench 上平均减少 43.6% 失败 episodes（最高 65.8%），同时平均回报提升 8‑12%；在真实机器人上失败次数大幅下降（≈70%），回报显著提升，且方差减小，表明更稳健。

**⚠️ 局限性**

局限性：① 仅考虑视觉与位置感知，未加入触觉、深度等多模态；② 预训练的恢复策略和世界模型是任务特定，跨任务泛化能力有限；③ 只在离线‑在线 RL 场景验证，缺少在完全从零学习的鲁棒性评估。

---

## 865. Learning Through Dialogue: Unpacking the Dynamics of Human-LLM Conversations on Political Issues

**arXiv ID:** 2601.07796 | [PDF](https://arxiv.org/pdf/2601.07796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 866. Data-driven control of hydraulic impact hammers under strict operational and control constraints

**arXiv ID:** 2601.07813 | [PDF](https://arxiv.org/pdf/2601.07813v1)

**作者:** Francisco Leiva `[一作]` (Universidad de Chile), Javier Ruiz-del-Solar `[通讯]` (Universidad de Chile)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于遥操作数据的学习方法，构建静态液压冲击锤的动力学模型，并利用该模型在离散控制下实现末端执行器对任意目标姿态的到达。

**💡 创新点**

创新点在于仅使用旋转编码器和离散电磁阀指令进行系统识别，结合监督学习得到的动力学模型，在强化学习与模型预测控制中合成可直接转移到真实机的到达策略，且不需要额外的传感器或精细调参。

**🔧 技术方法**

使用了监督学习（MLP/KAN）进行动力学建模，PPO强化学习与改进的CEM模型预测控制，JAX/Flax、Brax、MuJoCo、Gazebo和ROS等软件框架进行仿真与部署。

**📊 数据集**

数据集为约76分钟的遥操作记录，包含关节位置、速度和离散动作，用于训练动力学模型并生成RL/MPC训练样本。

**📈 对比分析**

通过在仿真和真实Bobcat E10 mini‑excavator上评估成功率、误差、工作空间逸出等指标，PPO‑MLP Δq̇策略在真实环境中平均位置误差<12 cm、姿态误差<0.08 rad，成功率高；iCEM在仿真中成功率更高但在真实环境中受限；实现了无需微调的Sim2Real转移。

**⚠️ 局限性**

主要局限在于缺乏外部感知，难以避免碰撞；工作空间约束未完全满足；对模型误差敏感，某些状态区间会导致失效；MPC受异步延迟影响，性能下降。

---

## 867. The Confidence Trap: Gender Bias and Predictive Certainty in LLMs

**arXiv ID:** 2601.07806 | [PDF](https://arxiv.org/pdf/2601.07806v1)

**作者:** Ahmed Sabir `[一作]` (Institute of Computer Science), Rajesh Sharma `[通讯]` (School of AI and CS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在性别偏见任务中的置信度校准与人类偏见的一致性，并对多款主流LLM进行系统评估。

**💡 创新点**

提出了专门衡量性别预测校准差异的新指标 Gender‑ECE，并将其与传统 ECE、MacroCE 等指标对比，验证其在性别公平评估中的敏感性。

**🔧 技术方法**

利用概率置信度提取、ECE/ICE/MacroCE 等校准评估方法，并通过贝塔校准、等宽/等量分箱等技术对模型输出进行后处理。

**📊 数据集**

使用 WinoBias、Winogender、GenderLex、WinoQueer 等基于 Winograd 模板的英语性别与 LGBTQ 偏见基准数据集。

**📈 对比分析**

通过与传统 ECE、MacroCE、cc‑ECE 等指标对比，发现 Gemma‑2‑9B 校准最差、GPT‑J‑6B 最优；贝塔校准可将 ECE 降低约三倍，且在多模型中提升了整体准确率。

**⚠️ 局限性**

局限性包括校准并非偏见缓解手段；仅关注代词预测、英语模板文本，缺乏跨语言与更复杂文本的评估；未对模型内部偏见源进行根源分析。

---

## 868. Lossy Source Coding with Broadcast Side Information

**arXiv ID:** 2601.07797 | [PDF](https://arxiv.org/pdf/2601.07797v1)

**作者:** Yiqi Chen `[一作]` (Technical University of Munich), Marc Geitz `[通讯]` (T-Labs Deutsche Telekom AG)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文研究了在通过退化广播信道传输的侧信息下，源编码的失真-带宽-速率区域（R-D-B）。

**💡 创新点**

创新点包括：①给出该问题的外界界并证明其适用性；②在限制助手使用分离方案时给出可实现的内部界，并在若干特殊情形下完成完全表征；③将分离方案与无编码方案在二次高斯情形下进行比较，揭示在不同失真阈值下两种方案的性能关系。

**🔧 技术方法**

主要技术手段包括：Wyner–Ziv源编码、退化广播信道的 superposition 编码、辅助随机变量的引入、时间共享技巧、Gaussian 随机变量的连续式测试通道以及连续式高斯源的可成功细化性质。

**📊 数据集**

本文属于理论信息论研究，不使用具体数据集；所有结果均通过信息量与随机变量模型推导得出。

**📈 对比分析**

比较方法：在ρ=1、S∼N(0,σ_s^2)、T=S 的二次高斯模型下，分别计算分离方案（R_Se）与无编码方案（R_U）的 rate–distortion 曲线；结果显示：在某些失真区间（例如 D_1^*≤D_1≤D_1^* 且 D_2^*<D_2≤D_2^*）分离方案优于无编码，且在其它区间二者相等或分离方案劣于无编码。

**⚠️ 局限性**

限制：①仅考虑退化广播信道和分离方案；②未给出非退化或更一般多用户信道的可达性；③高斯结果仅在带宽相等（ρ=1）时得到；④实际系统的实现复杂度与鲁棒性未讨论。

---

## 869. Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification

**arXiv ID:** 2601.07790 | [PDF](https://arxiv.org/pdf/2601.07790v1)

**作者:** Yahya Masri `[一作]` (George Mason University), Chaowei Yang `[通讯]` (George Mason University)

**通讯引用:** 6509 | [OpenAlex ID](https://openalex.org/A5048422487)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在真实 Linux 服务器的 journalctl 日志上，构建了半平衡的 46,774 条日志数据集，评估了 9 个小型语言模型（SLMs 与 SRLMs）在零样本、少样本和检索增强生成（RAG）三种提示策略下对日志严重级别的分类性能和推理延迟。

**💡 创新点**

首次把日志严重级别分类作为检索行为的可测探针，展示了 RAG 在小模型上的显著提升与模型架构/训练目标的耦合关系，并为数字孪生等实时监控场景提供了可部署基准。

**🔧 技术方法**

采用 0.6B–4B 参数规模的 Llama3.2、Gemma3、Qwen3、DeepSeek‑R1‑Distill‑Qwen‑1.5B 与 Phi‑4‑Mini‑Reasoning 这些 SLM 与 SRLM，使用零样本/少样本提示以及基于 FAISS 的向量检索（k=5）进行 RAG，评估准确率和每条日志的平均推理时延。

**📊 数据集**

数据集来自 6 台生产服务器的 journalctl 输出，覆盖 2024‑06 至 2025‑07，日志经结构化为 JSON，并按优先级 1–7（不含 emergency）平衡抽样后分为 80/20 训练/测试集。

**📈 对比分析**

实验结果显示，RAG 能使大部分模型准确率大幅提升：Qwen3‑4B 达到 95.64%、Gemma3‑1B 85.28%、Gemma3‑4B 81.84%；部分 SRLM（如 Qwen3‑1.7B、DeepSeek‑R1‑Distill‑Qwen‑1.5B、Phi‑4‑Mini‑Reasoning）在 RAG 下性能下降；在推理延迟方面，Gemma 与 Llama 系列大多在 1.2 秒以内完成，Phi‑4‑Mini‑Reasoning 则超过 228 秒，说明模型规模与检索适配性直接决定实时可行性。

**⚠️ 局限性**

局限性包括：严重级别标签受管理员自定义影响，噪声大且并非完备的诊断标注；RAG 对某些 SRLM 造成退化，提示需要模型特定的检索调优；实验仅基于静态日志，未覆盖流式动态环境与多源时序检索的挑战。

---

## 870. Vision-Language Model for Accurate Crater Detection

**arXiv ID:** 2601.07795 | [PDF](https://arxiv.org/pdf/2601.07795v1)

**作者:** Patrick Bauer `[一作]` (University of Technology of Troyes), Hichem Snoussi `[通讯]` (University of Technology of Troyes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对预训练的 OWLv2 视觉 Transformer 进行 LoRA 参数化微调，以实现月球表面陨石坑的自动检测。

**💡 创新点**

创新点在于：①在 OWLv2 的 MSA、分类与框回归头中插入 LoRA 以实现参数高效微调；②采用 CIoU 与对比损失联合训练，并通过文本“crater”锚点实现文本-图像嵌入对齐；③移除无用的对象性头，减少推理成本。

**🔧 技术方法**

技术手段包括 Vision Transformer (OWLv2)、LoRA 参数化微调、CIoU 损失、对比损失、匈牙利匹配、数据增强策略（随机遮影、尺度变化）等。

**📊 数据集**

使用 IMPACT 项目提供的手工标注陨石坑数据（LROC CDR 图像，512×512 tiles，约 178,812 颗陨石坑）进行训练与评估。

**📈 对比分析**

在 6 张测试图像（96 张 tiles）上评估，平均召回率 88.4%（范围 82.2–94.0%），平均精度 52.8%（范围 29.3–73.1%），表明模型在不同照明与大小条件下能保持高召回率，但精度受注释缺失影响。

**⚠️ 局限性**

局限性包括：①精度偏低，主要因人工注释缺失导致的“假阳性”；②对极小陨石坑检测不足；③使用方框定位而非直接分割，未充分捕获圆形陨石坑的几何形状。

---

## 871. Tuning-free Visual Effect Transfer across Videos

**arXiv ID:** 2601.07833 | [PDF](https://arxiv.org/pdf/2601.07833v1)

**作者:** Maxwell Jones `[一作]` (Carnegie Mellon University), Kuan-Chieh Jackson Wang `[通讯]` (Snap Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于参考视频的无调优视觉效果转移框架，能够将复杂的时间变化效果（如动态光照、角色变形、气象变化等）从一段视频无缝迁移到另一段输入视频或图像上，保持输入的内容与运动。

**💡 创新点**

创新点包括：①构建了首个包含1.7万种不同效果、12万+三元组（参考视频、输入、目标）的大规模参考效果数据集；②设计了可在训练时同时利用参考视频、输入视频和文本提示的扩散模型（基于Wan框架的多源条件化）；③提出了可扩展的自动化视频对视频（V2V）生成管线，用于大规模合成动态效果样本；④通过无优化、一次前向推断即可实现效果转移。

**🔧 技术方法**

技术方法主要包括：基于扩散模型的视觉效果生成（Wan、VACE、Wan 2.1等）；参考视频与输入视频的多源条件编码；使用LoRA适配器生成图像到视频效果；利用GPT-4o自动生成多样化效果提示；程序化代码效果与时间过渡合成；以及在推断时的多方向无监督指导（文本、输入、参考视频）。

**📊 数据集**

数据集：120K+三元组，涵盖1,700+效果；来源包括：43个LoRA图像到视频效果（约14K片段）；使用可扩展管线自动生成的V2V配对（约100K片段）；以及程序化生成的1,500种基于代码的时间变化效果（约100K片段）。

**📈 对比分析**

对比方法：文本驱动的图像到视频模型（Wan 2.1、Wan VACE）、文本驱动的视频编辑模型（Lucy Edit）、姿态/深度条件的VACE变体。评估指标包括：视频嵌入相似度（VideoPrism、CLIP）、人类偏好调查。结果显示：①相较基线模型，生成视频与参考视频的相似度显著更高；②在人类评测中，在参考遵循度、输入保持度等维度获胜率达70–98%；③在各种未见效果上保持良好泛化，且实现一次前向推断而非逐帧优化。

**⚠️ 局限性**

局限性：①对大规模扩散模型和GPU资源依赖较高；②效果转移质量仍受参考视频与输入视频相似度限制，极端运动或极端风格转换仍可能出现伪影；③数据集覆盖仍有限，某些极其细腻或复杂的时间效果在训练中难以充分表现；④目前仅支持视频/图像对视频的单向转移，尚未深入探索多参考或多模态融合。

---

## 872. MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

**arXiv ID:** 2601.07832 | [PDF](https://arxiv.org/pdf/2601.07832v1)

**作者:** Kewei Zhang `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**通讯引用:** 8707 | [OpenAlex ID](https://openalex.org/A5100554498)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新型线性注意力机制——多头线性注意力（MHLA），通过在token维度上分块并对每个查询块使用可学习的多头混合矩阵，恢复了query‑依赖的多样性；

**💡 创新点**

在传统线性注意力全局KV摘要的基础上，首次将token拆分为多个块，使用局部KV摘要并通过可学习的混合系数实现块级自适应，既保留了线性时间复杂度，又显著提升了表示能力，解决了全局上下文崩溃问题；

**🔧 技术方法**

核化线性注意力、局部KV摘要、可学习的多头混合矩阵（含局部性偏置初始化）、块级GEMM计算、无额外深度卷积/门控模块；

**📊 数据集**

ImageNet‑1K（分类、生成）、DiT/DiG（图像生成）、VBench（视频生成）、FineWeb‑Edu/LongBench（自然语言建模与长文本任务）；

**📈 对比分析**

与自注意力、传统线性注意力及其改进版本（Focus‑LA、RALA、MALA、GLA、Mamba、Transformer++等）在同一训练设置下对比；MHLA在ImageNet Top‑1准确率从73%提升至82.6%（VLT‑S）、在DiT‑S/2图像生成FID从89.7降至59.8、在视频生成FID从69.96提升至84.26，同时保持或接近线性注意力的吞吐；NLP上MMLU和LongBench得分均高于其他线性注意力模型；

**⚠️ 局限性**

对块数M的选择有限制，过大会增加额外开销，过小可能无法充分恢复多样性；在极长序列或超大模型上仍需进一步验证；实现依赖GPU高效矩阵乘法，CPU或低算力设备适配性尚未充分评估；

---

## 873. Exchange Is All You Need for Remote Sensing Change Detection

**arXiv ID:** 2601.07805 | [PDF](https://arxiv.org/pdf/2601.07805v1)

**作者:** Sijun Dong `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 28857 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种Siamese编码器-交换-解码器（SEED）框架，利用参数无关的特征交换取代传统的显式差异计算，实现高效遥感变化检测。

**💡 创新点**

创新点在于把差异特征融合改为信息保留的特征交换，并在像素一致性假设下证明其保持互信息和贝叶斯风险不变，形成无参数且易实现的变化检测范式。

**🔧 技术方法**

采用共享权重的Siamese编码器（Swin‑T、EfficientNet、ResNet）、层/通道/空间/随机特征交换、FPN与轻量解码器，并结合互信息与贝叶斯理论进行分析。

**📊 数据集**

在五个子米级遥感变化检测基准（SYSU‑CD、LEVIR‑CD、PX‑CLCD、WaterCD、CDD）上进行实验验证。

**📈 对比分析**

与Concat、Add、Subtract、SGSLN等传统融合/差异方法对比，SEED在所有数据集上均达到或超过state‑of‑the‑art，且参数量与计算成本与现有方法相当或更低。

**⚠️ 局限性**

仍受像素对齐假设影响，在大范围图像失配或低分辨率场景下性能下降；随机交换在极端变化情形下可能导致收敛慢。

---

## 874. "TODO: Fix the Mess Gemini Created": Towards Understanding GenAI-Induced Self-Admitted Technical Debt

**arXiv ID:** 2601.07786 | [PDF](https://arxiv.org/pdf/2601.07786v1)

**作者:** Abdullah Al Mujahid `[一作]` (Missouri University of Science and Technology), Mia Mohammad Imran `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5088063511)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过关键词检索与AST解析收集了6,540条包含LLM引用的代码注释，并进一步筛选出81条自我承认技术债的注释，探讨AI参与代码生成时产生的技术债分布与开发者对AI代码的认识。

**💡 创新点**

创新点在于首次系统性识别并定义了AI相关的自我承认技术债（AI‑ATD），并提出四类AI角色（Source、Catalyst、Mitigator、Neutral），揭示AI生成代码对技术债类型与责任分配的影响。

**🔧 技术方法**

技术上采用了基于正则表达式的关键词搜索、AST解析提取注释、人工双人标注（Cohen κ=0.896）以及开放编码分析，构建了81条注释的分类与角色标签。

**📊 数据集**

数据集为公开GitHub上2022‑2025年间的Python与JavaScript仓库，经过检索与去重得到6,540条LLM相关注释，最终筛选出81条自我承认技术债注释。

**📈 对比分析**

研究主要通过对比技术债类型分布（设计债、需求债、测试债等）与以往SATD研究结果，发现AI促进需求与测试债占比提升；未进行算法性能评估，仅提供统计与定性分析。

**⚠️ 局限性**

局限性包括关键词检索可能漏检或误检、仅覆盖两种语言、样本量有限且主观标注仍存在偏差，以及未考察技术债随时间演化的动态变化。

---

## 875. Reference Games as a Testbed for the Alignment of Model Uncertainty and Clarification Requests

**arXiv ID:** 2601.07820 | [PDF](https://arxiv.org/pdf/2601.07820v1)

**作者:** Manar Ali `[一作]` (Bielefeld University), Hendrik Buschmeier `[通讯]` (Bielefeld University)

**通讯引用:** 644 | [OpenAlex ID](https://openalex.org/A5064912341)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文探讨视觉语言模型在参考游戏中如何基于内部不确定性生成澄清请求。

**💡 创新点**

创新之处在于将参考游戏作为可控的、可测量的澄清行为测试平台。

**🔧 技术方法**

使用多样采样、置信度评估和提示式对话指令评估模型澄清能力。

**📊 数据集**

采用公开的颜色网格参考游戏数据集（color‑grid dataset）。

**📈 对比分析**

与人类基准和三种模型的基线准确率对比，GPT‑5‑mini在不确定时提出澄清请求的比例最高，准确率超过基线；Qwen‑2.5‑VL模型澄清行为与不确定性关联弱，整体表现低于人类。

**⚠️ 局限性**

局限包括缺乏迭代共识构建、模型置信度过高、澄清请求信息量不足，以及数据集简化导致结果难以推广。

---

## 876. Optimal Learning Rate Schedule for Balancing Effort and Performance

**arXiv ID:** 2601.07830 | [PDF](https://arxiv.org/pdf/2601.07830v1)

**作者:** Valentina Njaradi `[一作]` (University College London), Andrew Saxe `[通讯]` (University College London)

**通讯引用:** 5634 | [OpenAlex ID](https://openalex.org/A5011428379)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个将学习速率调度视为最优控制问题的规范化框架，推导了只需当前和预期最终表现的闭式最优学习率，并通过模拟验证其在多种网络架构和任务上的有效性；同时提出了基于情节记忆的未来表现估计方法。

**💡 创新点**

创新点包括：① 用控制理论（Hamilton–Jacobi–Bellman）得到通用闭式最优学习率；② 解决了不需要完整学习轨迹即可调度学习速率；③ 引入情节记忆加权估计未来表现；④ 对折扣因子、任务难度等参数进行解析研究；⑤ 将理论与动物行为和自我调节学习理论相结合。

**🔧 技术方法**

技术手段包括：梯度流学习动态建模、Hamilton–Jacobi–Bellman 方程求解与解析、Homotopy Perturbation + Padé 近似、情节记忆相似度加权估计、自动微分进行数值元优化，以及多种模拟实验。

**📊 数据集**

使用的数据集与任务有：教师‑学生网络（自生成）、MNIST 图像分类、线性感知机线性回归、两高斯判别任务、随机生成的教师网络实例；实验中还引用了老鼠两选任务的行为数据。

**📈 对比分析**

与数值最优学习率调度和固定学习率基线进行比较。实验结果显示，在教师‑学生、MNIST、线性感知机等任务上，闭式最优学习率的累计内部奖励与数值最优接近或略优，且明显优于固定学习率；在噪声梯度下仍保持优势。

**⚠️ 局限性**

局限性包括：① 仅适用于可微、梯度流的学习动态；② 需要估计最终表现，估计误差会导致性能下降；③ 对学习率的连续控制假设可能与生物系统的离散化约束不符；④ 未考虑多任务或多目标的学习速率分配；⑤ 对学习成本非二次形式的解析性未知。

---

## 877. Video Generation Models in Robotics - Applications, Research Challenges, Future Directions

**arXiv ID:** 2601.07823 | [PDF](https://arxiv.org/pdf/2601.07823v1)

**作者:** Zhiting Mei `[一作]` (Princeton University), Anirudha Majumdar `[通讯]` (Princeton University)

**通讯引用:** 934 | [OpenAlex ID](https://openalex.org/A5102792178)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了视频生成模型在机器人领域的最新进展与应用，重点探讨了其作为体现在世界模型的潜力、关键技术、评估方法及面临的挑战。

**💡 创新点**

创新点在于首次系统整合了视频生成模型的架构、机器人应用（数据生成、强化学习动力学、策略评估、视觉规划）以及评价指标，提出了多维度的挑战与未来方向，填补了先前综述中缺乏机器人视角的空白。

**🔧 技术方法**

主要涉及的技术包括扩散/流匹配视频生成、视觉语言模型（VLM/LLM）、动作预测网络、强化学习动态建模与视觉规划框架。

**📊 数据集**

综述引用了多种公开数据集与研究资源，如 KITTI、COCO、机器人演示数据集、以及各类视频生成基准集，但并未进行新的实验数据收集。

**📈 对比分析**

文章通过对比已有研究的评估指标（如 FID、Perceptual Quality、Physical Consistency Scores）说明当前模型在视觉质量和物理一致性方面的表现；由于是综述性质，没有自己的实验结果，但指出现有模型普遍存在幻觉、指令跟随差、成本高等性能瓶颈。

**⚠️ 局限性**

主要限制包括缺乏统一的评估基准和安全保障机制，模型易出现幻觉和物理违规，指令跟随能力不足，以及高昂的数据收集、训练和推理成本，限制了其在安全关键机器人任务中的广泛应用。

---

## 878. More Images, More Problems? A Controlled Analysis of VLM Failure Modes

**arXiv ID:** 2601.07812 | [PDF](https://arxiv.org/pdf/2601.07812v1)

**作者:** Anurag Das `[一作]` (Max Planck Institute for Informatics), Brais Martinez `[通讯]` (Samsung AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了MIMIC多图像评测框架，对现有大型视觉语言模型（LVLM）在多图像任务中的表现进行系统分析，并发现它们主要表现为单图像行为；基于此提出数据层面的程序化多图像训练样本和优化层面的注意力掩码两种互补的微调策略，显著提升多图像推理能力。

**💡 创新点**

创新点包括：1）构造可控的多图像基准MIMIC，能精细化区分信息分布、查询复杂度、干扰图像等维度；2）通过程序化合成多图像样本提供跨图像监督；3）提出基于层级注意力的掩码策略，限制深层跨图像交互，提升聚合和多概念追踪。

**🔧 技术方法**

使用的技术有：程序化数据生成（基于MS‑COCO标注构造多图像序列）、层级注意力掩码（限制特定层的跨图像注意力）、LoRA微调、视觉令牌序列长度控制（1‑D池化、像素空间下采样）、多任务训练。

**📊 数据集**

主要数据集为MS‑COCO（生成MIMIC评测与训练），OpenImages（合成多图像训练样本），并在MuirBench、Blink、MMIU、MIRB、MMT、NLVR2等现有多图像基准上进行评估。

**📈 对比分析**

通过与SOTA模型在MIMIC、Blink、MMIU、MIRB、MMT、NLVR2等任务的准确率/ F1 进行对比，7B模型在MIMIC上的平均分从54.0%提升至63.8%，在多图像基准上均实现显著提升；掩码版微调将 FLOPs 降低 81% 的同时，性能与全微调持平或更优。

**⚠️ 局限性**

局限性：1）基准与改进均以 MS‑COCO 为主，未验证在文档、医疗影像等专业领域的适用性；2）序列长度削减方法适用于语义推理，对细粒度像素级任务可能不理想；3）实验仅在公开权重模型上验证，闭源模型的适应性需进一步检验。

---

## 879. Kinship Data Benchmark for Multi-hop Reasoning

**arXiv ID:** 2601.07794 | [PDF](https://arxiv.org/pdf/2601.07794v1)

**作者:** Tianda Sun `[一作]` (University of York), Dimitar Kazakov `[通讯]` (University of York)

**通讯引用:** 870 | [OpenAlex ID](https://openalex.org/A5031368041)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套基于文化差异的家谱推理基准KinshipQA，用生成式流程自动化产生符合不同亲属体系的族谱数据，并通过这些数据生成多跳推理问题；

**💡 创新点**

创新点在于：①通过可调节的文化婚姻约束实现多样化、可控的家谱生成；②将亲属推理拆分为生物学链与文化分类两种推理模式；③设计无污染的评测流程，避免训练数据泄露；

**🔧 技术方法**

主要技术包括：基于RDF/OWL的家谱建模、路径模板化的提问生成、符号查询生成真值、自然语言序列化以及零-shot提示与确定性解码；

**📊 数据集**

使用了生成式的家谱数据集KinshipQA，其中包含七种亲属体系（Eskimo、Sudanese、Hawaiian、Iroquois、Dravidian、Crow、Omaha）的多级多跳问题；

**📈 对比分析**

采用统一零-shot评测协议，对六款模型（Qwen3‑32B、Gemma3‑27B、DeepSeek‑R1‑32B、GPT‑4o‑mini、Claude‑3.5‑Haiku、Gemini‑2.5‑Flash）进行EM和集合匹配评估，结果显示在非覆盖体系中模型平均准确率约为81.8%，覆盖体系约为96.0%，文化覆盖问题准确率最低（约44%），表明文化规则是主要瓶颈；

**⚠️ 局限性**

局限性包括：仅覆盖七种亲属体系，未考虑更广泛或混合系统；仅使用英文描述，可能偏向英语训练数据；评测仅采用零-shot，未探究few-shot或微调效果；缺乏人工基准与跨语言验证。

---

## 880. SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations

**arXiv ID:** 2601.07835 | [PDF](https://arxiv.org/pdf/2601.07835v1)

**作者:** Mohammed Himayath Ali `[一作]` (Cybersecurity and Artificial Intelligence Division), Shahnawaz Alam `[通讯]` (Cybersecurity and Artificial Intelligence Division)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对安全运营中心的LLM防御框架，结合宪法式AI、可演化宪法和Direct Preference Optimization，显著降低注入攻击成功率并提升安全分析准确率。

**💡 创新点**

创新点在于：①专门针对安全场景设计的安全宪法原则；②通过持续红队反馈实现宪法演化；③使用DPO实现安全模式的“遗忘”，在保持任务性能的同时消除不安全回答。

**🔧 技术方法**

核心技术包括：宪法式AI (Supervised Learning + RLHF)、Direct Preference Optimization、输入去污化、输出验证、持续红队和宪法演化机制。

**📊 数据集**

使用了约51,750条带注入攻击的安全数据集，涵盖日志、钓鱼邮件、恶意代码等六类攻击，并结合15,000条正常安全分析样本。

**📈 对比分析**

与基线模型（未加安全训练）、标准CAI、输入过滤、指令层级等方案对比，平均攻击成功率从80.4%降至4.3%（降低94.7%），正常任务准确率提升至95.1%，宪法遵从率保持在0.92以上。

**⚠️ 局限性**

局限性包括：推理延迟增加约23%，对完全新颖攻击的瞬时防御仍有限，宪法设计高度依赖专家知识，且演化过程需要持续的人为监控与迭代。

---

## 881. Tensor Algebra Processing Primitives (TAPP): Towards a Standard for Tensor Operations

**arXiv ID:** 2601.07827 | [PDF](https://arxiv.org/pdf/2601.07827v1)

**作者:** Jan Brandejs `[一作]` (Laboratoire de Chimie et Physique Quantique), Paolo Bientinesi `[通讯]` (Umeå University)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5048393932)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

**🎯 论文内容**

提出并实现了一个类似 BLAS 的通用张量收缩接口 TAPP，并给出可读性强的参考实现与完整测试集。

**💡 创新点**

创新点在于：①基于社区共识定义统一的张量操作语义与编程接口；②通过可变键值存储实现后台插件化与优化策略的可插拔；③提供专门针对张量收缩的 C 语言参考实现，促进不同库（TBLIS、cuTENSOR、CTF 等）快速接入。

**🔧 技术方法**

采用 C 语言实现低级 API（库句柄、执行器、张量描述符、操作描述符、虚拟键值存储）；核心算法为多维循环遍历加速；支持混合实/复数 32/64 位数据；通过预处理将标签分为收缩、Hadamard、自由、重复等集合。

**📊 数据集**

使用半随机张量数据集：随机生成张量维度、大小、正负 stride、零维度、复数/双精度等多种组合，用于覆盖 50+ 组单元测试，检验正确性和边界情况。

**📈 对比分析**

与 TBLIS 进行功能性对比，主要验证相同输入下结果一致；性能评估留给后续工作，当前参考实现仅保证正确性，未做高性能优化；未来计划提供基准集并实现 Multi‑TAPP 进行动态库切换与性能测评。

**⚠️ 局限性**

局限性包括：①目前仅支持 Case 1–3 的收缩（不支持 Case 4/5 的归约/广播等更通用情形）；②参考实现未针对 GPU、分布式内存等高性能场景做优化；③仅支持正/负 stride 及零 stride，仍未覆盖 16 位脑浮点等新型数据类型；④错误处理和异常情况仅通过整数错误码与文本描述实现，未提供完整的异常堆栈信息。

---

## 882. Passing the Baton: Shift Handovers within Cybersecurity Incident Response Teams

**arXiv ID:** 2601.07788 | [PDF](https://arxiv.org/pdf/2601.07788v1)

**作者:** Liberty Kent `[一作]` (University), Ingolf Becker `[通讯]` (University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过与6名来自不同地区的CSIRT专家访谈，评估并改进了两份CSIRT交接指南（Guideline A与Guideline B），旨在为CSIRT提供可操作的交接模板和检查表。

**💡 创新点**

创新点在于：①首次将医疗、核能、铁路等高可靠性领域的交接策略迁移并适配到网络安全事件响应；②采用半结构化访谈与归纳式主题分析，系统捕捉交接中的痛点、变异与最佳实践；③提出两份模板并通过专家反馈确定更受欢迎的模板（Guideline B），并给出可持续改进建议。

**🔧 技术方法**

主要技术手段为：半结构化访谈、Microsoft Teams录音转写、Excel编码、归纳式主题分析；辅以文献综述和对Patterson等人交接策略的参照。

**📊 数据集**

数据集为6名CSIRT专业人员的访谈文本，覆盖英国、摩洛哥、荷兰、沙特阿拉伯，涉及内部CSIRT、外部SOC、MSSP等多种组织类型；未使用公开数据集。

**📈 对比分析**

方法对比：并未与已有交接规范进行量化对比，只通过专家主观评估。Guideline B被5/6参与者认为最合适，认为覆盖了必要细节并支持独立传达；作者未给出客观性能指标，只提供定性满意度反馈。

**⚠️ 局限性**

局限性包括：样本量小且偏向外部SOC和MSSP，缺乏内部CSIRT和Follow‑The‑Sun模型的代表；未涉及失败或极端交接情境；访谈仅用Teams转写，未录音验证；缺少多组织间共识验证；改进建议需要更大规模、跨组织的验证研究。

---

## 883. Affordable Data Collection System for UAVs Taxi Vibration Testing

**arXiv ID:** 2601.07783 | [PDF](https://arxiv.org/pdf/2601.07783v1)

**作者:** Chaoyi Lin Yang `[一作]` (Universidad Carlos III de Madrid), Oscar E. Bonilla-Manrique `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5080701770)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一个低成本、可扩展的加速度数据采集系统，应用于小型固定翼无人机的Taxi和Ambient振动测试

**💡 创新点**

创新点在于将Orange Pi SBC与多个LSM6DS3TR-C MEMS IMU通过I2C多路复用、Python主从架构实现可扩展、成本低的分布式DAQ，并在航空振动测试中证明了其可重复的PSD结果

**🔧 技术方法**

采用Orange Pi 3 LTS单板机、Adafruit LSM6DS3TR-C 6-DoF MEMS加速度计、I2C多路复用器、Python主从软件架构、Welch功率谱密度估计、Hamming窗、CSV数据存储等技术

**📊 数据集**

使用的实验数据集为在Volantex RC Ranger 2400无人机上布置的六个加速度计（共18轴）进行的60 秒Taxi Vibration Test（6次）和20 分钟Ambient Vibration Test（2次）

**📈 对比分析**

通过比较TVT与AVT的平均归一化PSD和95%置信区间，验证系统在不同激励条件下的可重复性与稳定性；在低至中等振动能量下，系统的频谱一致性与商业DAQ相当

**⚠️ 局限性**

限制在于采样率受限于I2C与Python实现（约208 Hz），对高频或高速事件捕获有限；缺乏FIFO缓存与更精确的时间同步，导致长时间低信噪比测试中的置信区间较宽

---

